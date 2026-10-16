# Copyright 2025 DeepResearch authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by the Axion authors from the DeepResearch recipe: comments and
# log messages translated to English, hard-coded site paths replaced with
# environment variables or relative paths, and bug fixes. Unmodified files in
# this directory carry their original headers.
#
"""
Visit tool for DeepResearch using Jina API with SQLite cache.
Supports optional sharded cache.
"""

import asyncio
import json
import logging
import os
import random
import sqlite3
import threading
import time
import atexit
import hashlib
from typing import Any, Optional, Tuple
from uuid import uuid4

import requests
from openai import AzureOpenAI, OpenAI

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
_RECIPE_DIR = os.path.dirname(_TOOL_DIR)

import sys as _sys
if _RECIPE_DIR not in _sys.path:
    _sys.path.insert(0, _RECIPE_DIR)
from qwen_llm_utils import (  # noqa: E402
    apply_qwen_params,
    force_vllm_enabled,
    strip_thinking,
    vllm_base_urls,
    vllm_model_name,
)
_DEFAULT_CACHE_DIR = os.path.join(_RECIPE_DIR, "database")


def _bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "y"}


def _get_leader_flag(config: dict) -> bool:
    if config.get("cache_shard_leader") is not None:
        return bool(config.get("cache_shard_leader"))
    if os.getenv("VISIT_CACHE_SHARD_LEADER") is not None:
        return _bool_env("VISIT_CACHE_SHARD_LEADER", False)
    # Auto-detect common rank envs: only rank 0 is leader
    for env in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        if os.getenv(env) is not None:
            try:
                return int(os.getenv(env, "-1")) == 0
            except ValueError:
                return False
    return False


def _is_visit_error_result(result: str) -> bool:
    if not result:
        return True
    lowered = result.strip().lower()
    error_prefixes = (
        "[visit error]",
        "[visit] error",
        "[visit] invalid",
    )
    if lowered.startswith(error_prefixes):
        return True
    if "failed to read page" in lowered:
        return True
    if "could not be accessed" in lowered:
        return True
    return False


def _is_summarization_failure_result(result: str) -> bool:
    if not result:
        return True
    lowered = result.lower()
    failure_markers = (
        "summarization failed:",
        "failed to summarize content",
    )
    return any(marker in lowered for marker in failure_markers)


EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rationale**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
"""


class VisitCache:
    """SQLite cache storing URL content and (url, goal) summaries. Supports sharded DB files."""

    def __init__(
        self,
        cache_dir: str,
        cache_file: str,
        shards: int = 1,
        resume: bool = True,
        auto_shard: bool = False,
        auto_merge: bool = False,
        is_leader: bool = False,
        pool_dir: str = "",
    ) -> None:
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.master_file = cache_file
        self.shards = max(1, int(shards))
        self.resume = resume
        self.auto_shard = auto_shard
        self.auto_merge = auto_merge
        self.is_leader = is_leader

        base, _ = os.path.splitext(os.path.basename(self.master_file))
        self._shard_files = [
            os.path.join(self.cache_dir, f"{base}_shard{idx}.db") for idx in range(self.shards)
        ]

        self._conns: dict[int, sqlite3.Connection] = {}
        self._locks: dict[int, threading.Lock] = {}
        self._dead_shards: set[int] = set()
        self._shard_failures: dict[int, int] = {}
        self._max_shard_failures = max(1, int(os.getenv("VISIT_CACHE_MAX_SHARD_FAILURES", "3")))

        self._pool_dir = (pool_dir or os.getenv("VISIT_CACHE_POOL_DIR") or "").strip()
        self._pool_conns: dict[int, sqlite3.Connection] = {}
        self._pool_locks: dict[int, threading.Lock] = {}
        self._pool_dead: set[int] = set()
        self._pool_failures: dict[int, int] = {}
        if self._pool_dir and not os.path.isdir(self._pool_dir):
            logger.warning("[visit-cache] shared pool directory does not exist, ignoring: %s", self._pool_dir)
            self._pool_dir = ""
        if self._pool_dir:
            logger.warning("[visit-cache] read-only shared pool enabled: %s", self._pool_dir)

        if self.shards == 1:
            self._get_conn(0)

        if self.shards > 1 and self.auto_shard and self.is_leader:
            self._split_master()

        if self.shards > 1 and self.auto_merge and self.is_leader:
            atexit.register(self.merge_shards)

        atexit.register(self.close)

    def close(self) -> None:
        for conn in list(getattr(self, "_pool_conns", {}).values()):
            try: conn.close()
            except Exception: pass
        for conn in self._conns.values():
            try:
                conn.close()
            except Exception:
                pass

    def _open_conn(self, path: str) -> sqlite3.Connection:
        db_exists = os.path.exists(path)
        conn = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()
        _jm = (os.getenv("CACHE_JOURNAL_MODE") or "WAL").strip().upper()
        if _jm not in ("WAL", "DELETE", "TRUNCATE", "PERSIST", "MEMORY", "OFF"):
            _jm = "WAL"
        cursor.execute("PRAGMA journal_mode=" + _jm)
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=-8000")
        conn.commit()

        if self.resume and db_exists:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row["name"] for row in cursor.fetchall()}
            has_new = "url_content" in tables and "url_goal_info" in tables
            has_old = "visit_cache" in tables
            if has_new:
                self._ensure_tables(conn)
            elif has_old:
                self._ensure_tables(conn)
                self._migrate_from_old_structure(conn)
            else:
                self._ensure_tables(conn)
        else:
            self._ensure_tables(conn)
        return conn

    def _ensure_tables(self, conn: sqlite3.Connection) -> None:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS url_content (
                url TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS url_goal_info (
                url TEXT NOT NULL,
                goal TEXT NOT NULL,
                useful_information TEXT NOT NULL,
                timestamp REAL NOT NULL,
                PRIMARY KEY (url, goal)
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_url_goal_info_url
            ON url_goal_info(url)
            """
        )
        conn.commit()

    def _migrate_from_old_structure(self, conn: sqlite3.Connection) -> None:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='visit_cache'"
        )
        if cursor.fetchone() is None:
            return
        cursor.execute(
            "SELECT url, goal, content, useful_information, timestamp FROM visit_cache ORDER BY timestamp DESC"
        )
        old_records = cursor.fetchall()
        migrated_urls = set()
        for row in old_records:
            url = row["url"]
            goal = row["goal"]
            content = row["content"]
            useful_information = row["useful_information"]
            timestamp = row["timestamp"]

            if url not in migrated_urls and content:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO url_content (url, content, timestamp)
                    VALUES (?, ?, ?)
                    """,
                    (url, content, timestamp),
                )
                migrated_urls.add(url)
            if useful_information:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO url_goal_info (url, goal, useful_information, timestamp)
                    VALUES (?, ?, ?, ?)
                    """,
                    (url, goal, useful_information, timestamp),
                )
        conn.commit()

    def _get_conn(self, shard_id: int) -> sqlite3.Connection:
        if shard_id not in self._conns:
            if shard_id == 0 and self.shards == 1:
                path = self.master_file
            else:
                path = self._shard_files[shard_id]
            self._conns[shard_id] = self._open_conn(path)
            self._locks[shard_id] = threading.Lock()
        return self._conns[shard_id]

    def _get_pool_conn(self, shard_id: int) -> Optional[sqlite3.Connection]:
        """Read-only connection to the shared pool. mode=ro guarantees it is never
        written to."""
        if not self._pool_dir or shard_id in self._pool_dead:
            return None
        if shard_id not in self._pool_conns:
            base, _ = os.path.splitext(os.path.basename(self.master_file))
            name = f"{base}_shard{shard_id}.db" if self.shards > 1 else os.path.basename(self.master_file)
            path = os.path.join(self._pool_dir, name)
            if not os.path.exists(path):
                self._pool_dead.add(shard_id)
                return None
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=15.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._pool_conns[shard_id] = conn
            self._pool_locks[shard_id] = threading.Lock()
        return self._pool_conns[shard_id]

    def _note_pool_failure(self, shard_id: int, exc: BaseException) -> None:
        conn = self._pool_conns.pop(shard_id, None)
        if conn is not None:
            try: conn.close()
            except Exception: pass
        n = self._pool_failures.get(shard_id, 0) + 1
        self._pool_failures[shard_id] = n
        if n >= self._max_shard_failures:
            self._pool_dead.add(shard_id)
            logger.warning("[visit-cache] shared pool shard %d DISABLED after %d failures (%s: %s)",
                           shard_id, n, type(exc).__name__, exc)

    def _pool_lookup(self, shard_id: int, sql: str, args: tuple, col: str) -> Optional[str]:
        """One lookup in the shared pool. Any exception is swallowed and trips the
        breaker: the pool is an accelerator and must never take the tool down."""
        try:
            conn = self._get_pool_conn(shard_id)
            if conn is None:
                return None
            with self._pool_locks[shard_id]:
                cur = conn.cursor()
                cur.execute(sql, args)
                row = cur.fetchone()
                return row[col] if row else None
        except Exception as e:
            self._note_pool_failure(shard_id, e)
            return None

    def _note_shard_failure(self, shard_id: int, exc: BaseException) -> None:
        """A cache shard read/write failed: drop the connection, and once the failure
        count reaches the threshold disable that shard for good, falling back to
        fetching the page directly."""
        conn = self._conns.pop(shard_id, None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        n = self._shard_failures.get(shard_id, 0) + 1
        self._shard_failures[shard_id] = n
        if n >= self._max_shard_failures:
            self._dead_shards.add(shard_id)
            logger.warning(
                "[visit-cache] shard %d DISABLED after %d failures (%s: %s); "
                "falling back to live fetch for its URLs",
                shard_id, n, type(exc).__name__, exc,
            )
        else:
            logger.warning(
                "[visit-cache] shard %d failed (%d/%d) %s: %s",
                shard_id, n, self._max_shard_failures, type(exc).__name__, exc,
            )

    def _shard_id(self, url: str) -> int:
        if self.shards == 1:
            return 0
        digest = hashlib.md5(url.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little") % self.shards

    def _split_master(self) -> None:
        if not os.path.exists(self.master_file):
            return

        if not self.resume:
            for shard_file in self._shard_files:
                if os.path.exists(shard_file):
                    os.remove(shard_file)

        master_conn = self._open_conn(self.master_file)
        master_cursor = master_conn.cursor()

        master_cursor.execute("SELECT url, content, timestamp FROM url_content")
        content_rows = master_cursor.fetchall()
        rows_by_shard: dict[int, list[tuple[str, str, float]]] = {}
        for row in content_rows:
            shard_id = self._shard_id(row["url"])
            rows_by_shard.setdefault(shard_id, []).append(
                (row["url"], row["content"], row["timestamp"])
            )
        for shard_id, rows in rows_by_shard.items():
            conn = self._get_conn(shard_id)
            with self._locks[shard_id]:
                cursor = conn.cursor()
                cursor.executemany(
                    """
                    INSERT OR REPLACE INTO url_content (url, content, timestamp)
                    VALUES (?, ?, ?)
                    """,
                    rows,
                )
                conn.commit()

        master_cursor.execute(
            "SELECT url, goal, useful_information, timestamp FROM url_goal_info"
        )
        goal_rows = master_cursor.fetchall()
        rows_by_shard = {}
        for row in goal_rows:
            shard_id = self._shard_id(row["url"])
            rows_by_shard.setdefault(shard_id, []).append(
                (row["url"], row["goal"], row["useful_information"], row["timestamp"])
            )
        for shard_id, rows in rows_by_shard.items():
            conn = self._get_conn(shard_id)
            with self._locks[shard_id]:
                cursor = conn.cursor()
                cursor.executemany(
                    """
                    INSERT OR REPLACE INTO url_goal_info (url, goal, useful_information, timestamp)
                    VALUES (?, ?, ?, ?)
                    """,
                    rows,
                )
                conn.commit()

    def merge_shards(self) -> None:
        if self.shards <= 1:
            return
        master_conn = self._open_conn(self.master_file)
        master_cursor = master_conn.cursor()

        for shard_id, shard_file in enumerate(self._shard_files):
            if not os.path.exists(shard_file):
                continue
            conn = self._open_conn(shard_file)
            cursor = conn.cursor()

            cursor.execute("SELECT url, content, timestamp FROM url_content")
            content_rows = cursor.fetchall()
            if content_rows:
                master_cursor.executemany(
                    """
                    INSERT OR REPLACE INTO url_content (url, content, timestamp)
                    VALUES (?, ?, ?)
                    """,
                    [(row["url"], row["content"], row["timestamp"]) for row in content_rows],
                )
                master_conn.commit()

            cursor.execute(
                "SELECT url, goal, useful_information, timestamp FROM url_goal_info"
            )
            goal_rows = cursor.fetchall()
            if goal_rows:
                master_cursor.executemany(
                    """
                    INSERT OR REPLACE INTO url_goal_info (url, goal, useful_information, timestamp)
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        (row["url"], row["goal"], row["useful_information"], row["timestamp"])
                        for row in goal_rows
                    ],
                )
                master_conn.commit()

    def get_content_by_url(self, url: str) -> Optional[str]:
        shard_id = self._shard_id(url)
        if shard_id in self._dead_shards:
            return self._pool_lookup(shard_id,
                                     "SELECT content FROM url_content WHERE url = ?", (url,), "content")
        try:
            conn = self._get_conn(shard_id)
            with self._locks[shard_id]:
                cursor = conn.cursor()
                cursor.execute("SELECT content FROM url_content WHERE url = ?", (url,))
                row = cursor.fetchone()
                if row:
                    return row["content"]
        except Exception as e:
            self._note_shard_failure(shard_id, e)
        return self._pool_lookup(shard_id,
                                 "SELECT content FROM url_content WHERE url = ?", (url,), "content")

    def get_useful_information(self, url: str, goal: str) -> Optional[str]:
        shard_id = self._shard_id(url)
        if shard_id in self._dead_shards:
            return self._pool_lookup(
                shard_id,
                "SELECT useful_information FROM url_goal_info WHERE url = ? AND goal = ?",
                (url, goal), "useful_information")
        try:
            conn = self._get_conn(shard_id)
            with self._locks[shard_id]:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT useful_information FROM url_goal_info WHERE url = ? AND goal = ?",
                    (url, goal),
                )
                row = cursor.fetchone()
                if row:
                    return row["useful_information"]
        except Exception as e:
            self._note_shard_failure(shard_id, e)
        return self._pool_lookup(
            shard_id,
            "SELECT useful_information FROM url_goal_info WHERE url = ? AND goal = ?",
            (url, goal), "useful_information")

    def set(self, url: str, goal: str, content: str, useful_information: str) -> None:
        shard_id = self._shard_id(url)
        if shard_id in self._dead_shards:
            return
        try:
            self._set_unsafe(url, goal, content, useful_information)
        except Exception as e:
            self._note_shard_failure(shard_id, e)

    def _set_unsafe(self, url: str, goal: str, content: str, useful_information: str) -> None:
        shard_id = self._shard_id(url)
        conn = self._get_conn(shard_id)
        current_time = time.time()
        with self._locks[shard_id]:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO url_content (url, content, timestamp)
                VALUES (?, ?, ?)
                """,
                (url, content, current_time),
            )
            cursor.execute(
                """
                INSERT OR REPLACE INTO url_goal_info (url, goal, useful_information, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (url, goal, useful_information, current_time),
            )
            conn.commit()


def truncate_content(text: str, max_chars: int = 100000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[Content truncated due to length...]"


def _format_useful_information(url: str, goal: str, evidence: str, summary: str) -> str:
    return (
        f"The useful information in {url} for user goal {goal} as follows: \n\n"
        f"Evidence in page: \n{evidence}\n\n"
        f"Summary: \n{summary}\n\n"
    )


def jina_fetch_sync(url: str, jina_api_key: str, timeout: int = 50) -> str:
    if not jina_api_key:
        return "[visit] Error: JINA_API_KEY environment variable not set."

    headers = {"Authorization": f"Bearer {jina_api_key}"}

    max_retries = max(1, int(os.getenv("VISIT_JINA_MAX_RETRIES", "5")))
    backoff_cap = float(os.getenv("VISIT_JINA_BACKOFF_CAP", "8"))
    NO_RETRY = {400, 401, 403, 404, 405, 410, 414, 422, 451}
    last = "[visit] Failed to read page after all retries."
    for attempt in range(max_retries):
        try:
            response = requests.get(f"https://r.jina.ai/{url}", headers=headers, timeout=timeout)
            if response.status_code == 200:
                return response.text
            error_info = response.text
            try:
                error_json = json.loads(response.text)
                error_message = error_json.get("readableMessage") or error_json.get("message", response.text)
                error_info = f"[visit] Error (HTTP {response.status_code}): {error_message}"
            except Exception:
                error_info = f"[visit] Error (HTTP {response.status_code}): {response.text[:500]}"
            last = error_info
            if response.status_code in NO_RETRY:
                return error_info
        except requests.exceptions.Timeout:
            last = f"[visit] Error: Request timed out after {timeout}s"
        except requests.exceptions.RequestException as e:
            last = f"[visit] Error: Network error: {str(e)}"
        except Exception as e:
            last = f"[visit] Error: Unexpected error: {str(e)}"
        if attempt < max_retries - 1:
            wait = min(backoff_cap, 0.5 * (2 ** attempt)) * (0.5 + random.random())
            time.sleep(wait)
    return last


_RR_STATE = {}   # api_base string -> next index (per process)
_INFLIGHT = {}   # endpoint url -> number of this process's summaries currently in flight
_RR_LOCK = __import__("threading").Lock()


def _pick_base_url(api_base: str) -> str:
    """Pick one endpoint when api_base is a comma-separated list of base URLs.

    Chooses the endpoint with the fewest in-flight requests in this process,
    breaking ties round robin. Random and plain round-robin selection carry no
    feedback from a slow engine, so under high concurrency a few engines
    accumulate long queues while the rest idle. A worker runs many summaries at
    once, so its own in-flight count per endpoint tracks that endpoint's latency
    closely enough to steer new requests away from a backed-up engine.

    Call _release_base_url(url) when the request ends.
    """
    if not api_base or "," not in api_base:
        return api_base
    candidates = [x.strip() for x in api_base.split(",") if x.strip()]
    if not candidates:
        return api_base
    with _RR_LOCK:
        start = _RR_STATE.get(api_base)
        if start is None:
            start = random.SystemRandom().randrange(len(candidates))
        _RR_STATE[api_base] = (start + 1) % len(candidates)
        best = None; best_load = None
        for k in range(len(candidates)):
            url = candidates[(start + k) % len(candidates)]
            load = _INFLIGHT.get(url, 0)
            if best is None or load < best_load:
                best, best_load = url, load
        _INFLIGHT[best] = _INFLIGHT.get(best, 0) + 1
    return best


def _release_base_url(url: str) -> None:
    if not url:
        return
    with _RR_LOCK:
        n = _INFLIGHT.get(url, 0) - 1
        if n <= 0:
            _INFLIGHT.pop(url, None)
        else:
            _INFLIGHT[url] = n


def summarize_content(
    content: str,
    url: str,
    goal: str,
    api_key: str = "",
    api_base: str = "",
    model_name: str = "gpt-4o-mini",
    azure_endpoint: str = "",
    azure_api_version: str = "2024-08-01-preview",
    fallback_api_key: str = "",
    fallback_api_base: str = "",
    fallback_model_name: str = "",
    fallback_azure_endpoint: str = "",
    fallback_azure_api_version: str = "2024-08-01-preview",
    timeout_seconds: float = 300,
    max_content_length: int = 100000,
) -> str:
    if content.startswith("[visit]"):
        return _format_useful_information(
            url,
            goal,
            content,
            "The webpage could not be accessed. Please check the URL or file format.",
        )

    # Each candidate is (label, client, model_name, is_vllm). is_vllm marks the
    # local OpenAI-compatible (vLLM) path so Qwen-only sampling params /
    # chat_template_kwargs are only sent there, never to Azure.
    candidate_clients = []
    if azure_endpoint:
        candidate_clients.append(
            (
                "visit-summary primary",
                AzureOpenAI(
                    api_key=api_key,
                    api_version=azure_api_version,
                    azure_endpoint=azure_endpoint,
                    timeout=timeout_seconds,
                ),
                model_name,
                False,
            )
        )
    elif api_key:
        client_kwargs = {"api_key": api_key, "timeout": timeout_seconds}
        if api_base:
            # placeholder; a fresh least-in-flight pick is made for EVERY attempt below so a retry never
            # lands on the same backed-up engine (the old code retried 3x on one random endpoint)
            client_kwargs["base_url"] = api_base.split(",")[0].strip()
        candidate_clients.append(("visit-summary primary", OpenAI(**client_kwargs), model_name, True))

    if fallback_azure_endpoint:
        candidate_clients.append(
            (
                "visit-summary fallback",
                AzureOpenAI(
                    api_key=fallback_api_key,
                    api_version=fallback_azure_api_version,
                    azure_endpoint=fallback_azure_endpoint,
                    timeout=timeout_seconds,
                ),
                fallback_model_name,
                False,
            )
        )
    elif fallback_api_key:
        fallback_client_kwargs = {"api_key": fallback_api_key, "timeout": timeout_seconds}
        if fallback_api_base:
            fallback_client_kwargs["base_url"] = _pick_base_url(fallback_api_base)
        candidate_clients.append(("visit-summary fallback", OpenAI(**fallback_client_kwargs), fallback_model_name, True))

    if not candidate_clients:
        content = truncate_content(content, 5000)
        return _format_useful_information(
            url,
            goal,
            content,
            "The webpage content could not be processed, and therefore, no information is available.",
        )

    content = truncate_content(content, max_content_length)

    messages = [{"role": "user", "content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}]

    max_retries = 3
    last_error = None
    for client_label, client, current_model_name, is_vllm in candidate_clients:
        # Web-scrape summary is a general (non-reasoning) task and runs with
        # thinking OFF; Qwen-only params only go to the local vLLM path.
        create_kwargs = {
            "model": current_model_name,
            "messages": messages,
            "max_completion_tokens": int(os.getenv("VISIT_SUMMARY_MAX_TOKENS", "4096")),
        }
        apply_qwen_params(
            create_kwargs, mode="instruct_general", is_vllm=is_vllm,
            enable_thinking=False, overwrite=True,
        )
        for attempt in range(max_retries):
            _picked = None
            if is_vllm and client_label == "visit-summary primary" and api_base and "," in api_base:
                _picked = _pick_base_url(api_base)
                client = OpenAI(api_key=api_key, timeout=timeout_seconds, base_url=_picked)
            try:
                response = client.chat.completions.create(**create_kwargs)
                raw_result = response.choices[0].message.content
                if not raw_result:
                    continue
                # Belt-and-suspenders: drop any thinking trace that slipped through.
                raw_result = strip_thinking(raw_result)
                raw_result = raw_result.replace("```json", "").replace("```", "").strip()
                try:
                    parsed = json.loads(raw_result)
                except json.JSONDecodeError:
                    left = raw_result.find("{")
                    right = raw_result.rfind("}")
                    if left != -1 and right != -1 and left < right:
                        parsed = json.loads(raw_result[left : right + 1])
                    else:
                        return _format_useful_information(url, goal, raw_result, "")

                evidence = parsed.get("evidence", "N/A")
                summary = parsed.get("summary", "N/A")
                return _format_useful_information(url, goal, evidence, summary)
            except Exception as e:
                last_error = e
                _diag_base = getattr(client, "base_url", "?")
                _diag_status = getattr(e, "status_code", getattr(getattr(e, "response", None), "status_code", "?"))
                logger.warning(
                    "[visit] %s attempt %s/%s failed for %s: base_url=%s model=%s status=%s err=%s: %s",
                    client_label,
                    attempt + 1,
                    max_retries,
                    url,
                    _diag_base,
                    model_name,
                    _diag_status,
                    type(e).__name__,
                    e,
                )
            finally:
                _release_base_url(_picked)
        logger.warning("[visit] %s exhausted; trying next summarizer path for %s", client_label, url)

    content = truncate_content(content, 3000)
    return _format_useful_information(
        url,
        goal,
        content,
        f"Summarization failed: {str(last_error)}" if last_error else "Failed to summarize content",
    )


class DeepResearchVisitTool(BaseTool):
    """Visit tool with optional SQLite cache."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        def _env_override(name: str, fallback: str = "") -> str:
            # Respect explicit env overrides even when they are set to the empty string.
            return os.environ[name] if name in os.environ else fallback

        self.jina_api_key = (
            config.get("jina_api_key")
            or os.environ.get("JINA_API_KEY", "")
            or (os.environ.get("JINA_API_KEYS", "").split(",")[0].strip() if os.environ.get("JINA_API_KEYS") else "")
        )
        self.timeout = config.get("timeout", 50)

        # Prefer visit-summary-specific env vars over static tool config so the launcher
        # can switch providers without being overridden by tools.yaml defaults.
        self.api_key = _env_override("VISIT_SUMMARY_API_KEY", config.get("api_key") or "")
        self.api_base = _env_override("VISIT_SUMMARY_API_BASE", config.get("api_base") or "")
        self.model_name = (
            _env_override(
                "VISIT_SUMMARY_MODEL_NAME",
                config.get("summary_model_name") or "",
            )
        )
        self.azure_endpoint = _env_override("VISIT_SUMMARY_AZURE_ENDPOINT", config.get("azure_endpoint") or "")
        self.azure_api_version = (
            _env_override("VISIT_SUMMARY_AZURE_API_VERSION", config.get("azure_api_version") or "2024-08-01-preview")
        )
        self.fallback_api_key = _env_override(
            "VISIT_SUMMARY_FALLBACK_API_KEY",
            config.get("fallback_api_key") or "",
        )
        self.fallback_api_base = _env_override(
            "VISIT_SUMMARY_FALLBACK_API_BASE",
            config.get("fallback_api_base") or "",
        )
        self.fallback_model_name = _env_override(
            "VISIT_SUMMARY_FALLBACK_MODEL_NAME",
            config.get("fallback_model_name") or "",
        )
        self.fallback_azure_endpoint = _env_override(
            "VISIT_SUMMARY_FALLBACK_AZURE_ENDPOINT",
            config.get("fallback_azure_endpoint") or "",
        )
        self.fallback_azure_api_version = _env_override(
            "VISIT_SUMMARY_FALLBACK_AZURE_API_VERSION",
            config.get("fallback_azure_api_version") or "2024-08-01-preview",
        )

        # Global force switch: pin the page summarizer to the local Qwen3.5 vLLM,
        # clearing any azure endpoint / gpt-5.5 model the platform env injected.
        if force_vllm_enabled():
            _already_local = (
                (self.api_base or "").strip().startswith("http")
                and (self.model_name or "").strip().lower().startswith("qwen")
            )
            self.api_key = self.api_key or "EMPTY"
            self.azure_endpoint = ""
            # Drop the azure fallback so it cannot pull traffic back to gpt-5.5.
            self.fallback_azure_endpoint = ""
            if not _already_local:
                _vllm_urls = vllm_base_urls()
                if _vllm_urls:
                    self.api_base = ",".join(_vllm_urls)
                    self.model_name = vllm_model_name()
        self.summary_timeout_seconds = float(os.environ.get("VISIT_SUMMARY_TIMEOUT_SECONDS", "300"))
        self.max_content_length = config.get("max_content_length", 100000)

        cache_enabled = config.get("cache_enabled")
        if cache_enabled is None:
            cache_enabled = _bool_env("VISIT_CACHE_ENABLED", True)
        cache_resume = config.get("cache_resume")
        if cache_resume is None:
            cache_resume = _bool_env("VISIT_CACHE_RESUME", True)

        cache_dir = config.get("cache_dir") or os.getenv("VISIT_CACHE_DIR", "")
        if not cache_dir:
            cache_dir = _DEFAULT_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)

        cache_file = config.get("cache_file") or os.getenv("VISIT_CACHE_FILE", "")
        if not cache_file:
            cache_file = os.path.join(cache_dir, "visit_cache.db")

        cache_shards = config.get("cache_shards")
        if cache_shards is None:
            cache_shards = int(os.getenv("VISIT_CACHE_SHARDS", "1"))

        cache_auto_shard = config.get("cache_auto_shard")
        if cache_auto_shard is None:
            cache_auto_shard = _bool_env("VISIT_CACHE_AUTO_SHARD", False)

        cache_auto_merge = config.get("cache_auto_merge")
        if cache_auto_merge is None:
            cache_auto_merge = _bool_env("VISIT_CACHE_AUTO_MERGE", False)

        is_leader = _get_leader_flag(config)
        cache_pool_dir = (config.get("cache_pool_dir") or os.getenv("VISIT_CACHE_POOL_DIR", "") or "").strip()

        self.cache = (
            VisitCache(
                cache_dir=cache_dir,
                cache_file=cache_file,
                shards=cache_shards,
                resume=cache_resume,
                auto_shard=cache_auto_shard,
                auto_merge=cache_auto_merge,
                is_leader=is_leader,
                pool_dir=cache_pool_dir,
            )
            if cache_enabled
            else None
        )

        if not self.jina_api_key:
            logger.warning("JINA_API_KEY not set, visit tool will not work")

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"visited_urls": [], "results": []}
        return instance_id, ToolResponse()

    def _validate_url(self, url: Any) -> Tuple[bool, str]:
        if not isinstance(url, str):
            return False, "[Visit] URL must be a string."
        url = url.strip()
        if not url:
            return False, "[Visit] URL cannot be empty."
        lowered = url.lower()
        if lowered.startswith("view-source:"):
            return False, "[Visit] Invalid URL protocol: 'view-source:' is not allowed."
        if lowered.startswith("javascript:"):
            return False, "[Visit] Invalid URL protocol: 'javascript:' is not allowed."
        if lowered.startswith("data:"):
            return False, "[Visit] Invalid URL protocol: 'data:' is not allowed."
        if lowered.startswith("file:"):
            return False, "[Visit] Invalid URL protocol: 'file:' is not allowed."
        return True, ""

    def _fetch_and_summarize(self, url: str, goal: str) -> str:
        content = jina_fetch_sync(url, self.jina_api_key, self.timeout)
        useful_info = summarize_content(
            content,
            url,
            goal,
            self.api_key,
            self.api_base,
            self.model_name,
            self.azure_endpoint,
            self.azure_api_version,
            self.fallback_api_key,
            self.fallback_api_base,
            self.fallback_model_name,
            self.fallback_azure_endpoint,
            self.fallback_azure_api_version,
            self.summary_timeout_seconds,
            self.max_content_length,
        )

        if self.cache and content and not content.startswith("[visit]"):
            if (
                not _is_visit_error_result(useful_info)
                and not _is_summarization_failure_result(useful_info)
            ):
                self.cache.set(url, goal, content, useful_info)

        return useful_info

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        url = parameters.get("url", "")
        goal = parameters.get("goal", "Extract relevant information")

        if not url:
            return ToolResponse(text="[Visit Error] URL cannot be empty."), 0.0, {"error": "empty_url"}

        urls = url if isinstance(url, list) else [url]
        loop = asyncio.get_running_loop()
        results = []
        valid_urls = []

        for u in urls:
            is_valid, error_msg = self._validate_url(u)
            if not is_valid:
                results.append(error_msg)
                continue
            u = u.strip()
            valid_urls.append(u)

            if self.cache:
                cached_info = self.cache.get_useful_information(u, goal)
                if cached_info:
                    logger.info("[visit] cache hit (url+goal): %s %s", u, goal)
                    results.append(cached_info)
                    continue

                cached_content = self.cache.get_content_by_url(u)
                if cached_content:
                    logger.info("[visit] cache hit (url) %s, summarizing: %s", u, goal)
                    result = await loop.run_in_executor(
                        None,
                        summarize_content,
                        cached_content,
                        u,
                        goal,
                        self.api_key,
                        self.api_base,
                        self.model_name,
                        self.azure_endpoint,
                        self.azure_api_version,
                        self.fallback_api_key,
                        self.fallback_api_base,
                        self.fallback_model_name,
                        self.fallback_azure_endpoint,
                        self.fallback_azure_api_version,
                        self.summary_timeout_seconds,
                        self.max_content_length,
                    )
                    if (
                        result
                        and not _is_visit_error_result(result)
                        and not _is_summarization_failure_result(result)
                    ):
                        self.cache.set(u, goal, cached_content, result)
                    results.append(result)
                    continue

            result = await loop.run_in_executor(None, self._fetch_and_summarize, u, goal)
            results.append(result)

        self._instance_dict[instance_id]["visited_urls"].extend(valid_urls)
        self._instance_dict[instance_id]["results"].extend(results)

        combined_result = "\n=======\n".join(results)
        error_count = sum(1 for result in results if _is_visit_error_result(result))
        if error_count == len(results):
            status = "error"
        elif error_count > 0:
            status = "partial_success"
        else:
            status = "success"
        metrics = {
            "url_count": len(urls),
            "valid_url_count": len(valid_urls),
            "error_count": error_count,
            "status": status,
        }

        return ToolResponse(text=combined_result), 0.0, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
