#!/usr/bin/env python3
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: LicenseRef-Axion-NC-1.0
# Licensed under the Axion Non-Commercial Research and Responsible Use
# License, Version 1.0. See the LICENSE file at the repository root.

"""Minimal OpenAI-compatible chat client with rate limiting and checkpointed batching.

Configuration comes from the environment:

    OPENAI_API_KEY        required
    OPENAI_BASE_URL       optional, for any OpenAI-compatible endpoint
    OPENAI_MODEL          default model name, overridable per client
    OPENAI_QUERY_JSON     optional, extra query parameters as JSON, for a
                          gateway that needs them alongside the key
    OPENAI_HEADERS_JSON   optional, extra request headers as JSON

Two pieces matter for data generation at scale:

* `LLMClient.chat` retries on an empty response, a failed validator, or an API
  error. Reasoning models count reasoning tokens against max_tokens, so an
  empty response can mean either rate limiting or an exhausted budget; the
  budget is doubled on each empty response to rule out the second cause.
* `run_batch` writes every success to a checkpoint keyed by item, so an
  interrupted run resumes instead of restarting, and items that fail a pass are
  retried in the next one. Items still failing after the last pass are
  returned rather than silently dropped.
"""
import collections
import concurrent.futures as cf
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from openai import OpenAI

DEFAULT_MAX_TOKENS = 32768
MIN_MAX_TOKENS = 8192
MAX_MAX_TOKENS = 128000


class EmptyResponse(Exception):
    """The API returned a response with no content."""


class ValidationFailed(Exception):
    """The response was non-empty but failed the caller's validator."""


class RateLimiter:
    """Simple queries-per-minute limiter, shared across threads."""

    def __init__(self, qpm: int):
        self.interval = 60.0 / max(1, qpm)
        self._next = 0.0
        self._penalty = 0.0

    def acquire(self):
        now = time.monotonic()
        wait = max(0.0, self._next - now) + self._penalty
        if wait > 0:
            time.sleep(wait)
        self._penalty = 0.0
        self._next = max(now, self._next) + self.interval

    def penalize(self, seconds: float = 5.0):
        """Back off after a 429, so the next acquire waits longer."""
        self._penalty = max(self._penalty, seconds)


def _json_env(name: str):
    """Parse a JSON-valued environment variable, or None when unset or invalid."""
    raw = os.environ.get(name, '').strip()
    if not raw:
        return None
    try:
        v = json.loads(raw)
        return v if isinstance(v, dict) and v else None
    except Exception:
        print(f'[llm] ignoring {name}: not valid JSON', flush=True)
        return None


def _retry_after(err: Exception) -> Optional[float]:
    m = re.search(r'retry[- ]after[":\s]+(\d+(?:\.\d+)?)', str(err), re.I)
    return float(m.group(1)) if m else None


def _strip_fence(text: str) -> str:
    t = text.strip()
    if t.startswith('```'):
        t = re.sub(r'^```[a-zA-Z]*\s*', '', t)
        t = re.sub(r'\s*```$', '', t)
    return t.strip()


def _extract_json(text: str):
    t = _strip_fence(text)
    try:
        return json.loads(t)
    except Exception:
        pass
    for opener, closer in (('{', '}'), ('[', ']')):
        i, j = t.find(opener), t.rfind(closer)
        if 0 <= i < j:
            try:
                return json.loads(t[i:j + 1])
            except Exception:
                continue
    raise ValueError('no JSON object found in response')


class LLMClient:
    """Chat client with retries, rate limiting and per-call statistics."""

    def __init__(
        self,
        model: Optional[str] = None,
        qpm: int = 60,
        max_attempts: int = 8,
        base_backoff: float = 2.0,
        max_backoff: float = 90.0,
        timeout: float = 300.0,
        min_chars: int = 1,
    ):
        key = os.environ.get('OPENAI_API_KEY', '').strip()
        if not key:
            raise RuntimeError('OPENAI_API_KEY is not set')
        self.model = (model or os.environ.get('OPENAI_MODEL', '')).strip()
        if not self.model:
            raise RuntimeError('no model given: pass model= or set OPENAI_MODEL')
        self._client = OpenAI(
            api_key=key,
            base_url=os.environ.get('OPENAI_BASE_URL') or None,
            default_query=_json_env('OPENAI_QUERY_JSON'),
            default_headers=_json_env('OPENAI_HEADERS_JSON'),
            timeout=timeout,
        )
        self.limiter = RateLimiter(qpm)
        self.max_attempts = max_attempts
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.min_chars = min_chars
        self.stats = collections.Counter()

    def chat(
        self,
        messages: list,
        validator: Optional[Callable[[str], bool]] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 1.0,
        max_attempts: Optional[int] = None,
        tag: str = '',
    ) -> str:
        """Run one chat completion, retrying on empty, invalid or failed calls.

        Raises the last error when every attempt fails.
        """
        attempts = max_attempts or self.max_attempts
        budget = max(max_tokens, MIN_MAX_TOKENS)
        last_err: Optional[Exception] = None

        for i in range(attempts):
            self.limiter.acquire()
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=min(budget, MAX_MAX_TOKENS),
                    temperature=temperature,
                )
                text = (resp.choices[0].message.content or '').strip()
                if len(text) < self.min_chars:
                    self.stats['empty'] += 1
                    budget = min(budget * 2, MAX_MAX_TOKENS)
                    raise EmptyResponse(f'empty content (budget now {budget})')
                if validator and not validator(text):
                    self.stats['invalid'] += 1
                    raise ValidationFailed('validator rejected the response')
                self.stats['ok'] += 1
                return text
            except Exception as e:                      # noqa: BLE001
                last_err = e
                self.stats[type(e).__name__] += 1
                if '429' in str(e) or 'rate' in str(e).lower():
                    self.limiter.penalize(_retry_after(e) or 5.0)
                if i == attempts - 1:
                    break
                wait = min(self.base_backoff * (2 ** i), self.max_backoff)
                wait *= 0.5 + random.random()           # jitter, or workers resynchronise
                print(f'[llm] attempt {i + 1}/{attempts} failed{" " + tag if tag else ""}: '
                      f'{type(e).__name__}: {str(e)[:160]} -> retry in {wait:.1f}s', flush=True)
                time.sleep(wait)
        raise RuntimeError(f'all {attempts} attempts failed{" " + tag if tag else ""}: {last_err}')

    def chat_json(self, messages: list, required_keys: Iterable[str] = (), **kw) -> dict:
        """Like chat, but parse the response as JSON and require certain keys."""
        keys = tuple(required_keys)

        def ok(text: str) -> bool:
            try:
                obj = _extract_json(text)
            except Exception:
                return False
            return isinstance(obj, dict) and all(k in obj for k in keys)

        return _extract_json(self.chat(messages, validator=ok, **kw))


def run_batch(
    items: list,
    key_fn: Callable[[Any], str],
    work_fn: Callable[[Any], dict],
    checkpoint: str,
    workers: int = 6,
    passes: int = 4,
    pass_sleep: float = 30.0,
    impl: Optional[str] = None,
) -> 'tuple[dict, list]':
    """Apply work_fn to every item, checkpointing results by key.

    Keys already in the checkpoint are skipped, so a run can be resumed. Items
    that fail one pass are retried in the next, up to `passes`. Returns
    (results_by_key, permanently_failed_items).

    `impl` is a fingerprint of the code that produced a result. When given, it
    is stored on every row, and rows carrying a different fingerprint are
    discarded on resume rather than reused.
    """
    ckpt = Path(checkpoint)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    done: dict = {}
    stale = 0
    if ckpt.exists():
        with open(ckpt, encoding='utf-8') as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if impl is not None and row.get('_impl') != impl:
                    stale += 1
                    continue
                done[row['_key']] = row
        print(f'[batch] resuming with {len(done)} cached results', flush=True)
        if stale:
            print(f'[batch] WARNING: discarded {stale} cached rows written by a '
                  f'different implementation (expected _impl={impl}); they will '
                  f'be re-judged', flush=True)

    todo = [it for it in items if key_fn(it) not in done]
    for p in range(passes):
        if not todo:
            break
        print(f'[batch] pass {p + 1}/{passes}: {len(todo)} items, {workers} workers', flush=True)
        failed, n = [], 0
        with open(ckpt, 'a', encoding='utf-8') as out:
            with cf.ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(work_fn, it): it for it in todo}
                for fut in cf.as_completed(futs):
                    it = futs[fut]
                    try:
                        res = fut.result()
                    except Exception as e:              # noqa: BLE001
                        failed.append(it)
                        print(f'[batch] {key_fn(it)} failed: {type(e).__name__}: {str(e)[:120]}',
                              flush=True)
                        continue
                    res = dict(res)
                    res['_key'] = key_fn(it)
                    if impl is not None:
                        res['_impl'] = impl
                    done[res['_key']] = res
                    out.write(json.dumps(res, ensure_ascii=False) + '\n')
                    out.flush()
                    n += 1
                    if n % 20 == 0:
                        print(f'[batch]   {n}/{len(todo)} done, {len(failed)} failed', flush=True)
        todo = failed
        if todo and p < passes - 1:
            time.sleep(pass_sleep)

    if todo:
        print(f'[batch] {len(todo)} items failed in every pass', flush=True)
    return done, todo
