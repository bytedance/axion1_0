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
# Modified by the Axion authors for package paths, deployment settings and runtime fixes.
#
"""
Python execution tool for DeepResearch.
Supports sandbox_fusion and E2B backends. This tool is intentionally stateless and does not use cache.
"""

import asyncio
import json
import logging
import os
import random
import re
import time
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

try:
    import json5
except ImportError:
    json5 = None

try:
    from requests.exceptions import Timeout as RequestsTimeout
except ImportError:
    RequestsTimeout = Exception

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# The e2b SDK's httpx hook logs bare, context-free "Response 404"/"Response 429"
# lines at ERROR (e2b/api/__init__.py:_log_response), so a run's logs show the
# status but not which sandbox op failed or why. These typed exceptions let us
# attribute the failure to create/run/kill and infer the HTTP status. Import is
# best-effort so the tool still loads if the e2b package layout changes.
try:
    from e2b.exceptions import (
        AuthenticationException as _E2BAuthException,
        NotFoundException as _E2BNotFoundException,
        RateLimitException as _E2BRateLimitException,
        SandboxNotFoundException as _E2BSandboxNotFoundException,
    )
except Exception:  # pragma: no cover - e2b optional / layout drift
    _E2BAuthException = _E2BNotFoundException = None
    _E2BRateLimitException = _E2BSandboxNotFoundException = None


def _e2b_error_summary(exc: Exception) -> str:
    """Human-readable label for an E2B failure, inferring the HTTP status from
    the exception type so opaque 'Response 404'/'Response 429' lines become
    actionable: sandbox expired (404) vs concurrency/rate limit (429) vs auth."""
    name = type(exc).__name__
    status, hint = "?", ""
    if _E2BRateLimitException is not None and isinstance(exc, _E2BRateLimitException):
        status, hint = "429", "concurrent-sandbox / rate limit exceeded (lower rollout concurrency or raise E2B plan limit)"
    elif _E2BSandboxNotFoundException is not None and isinstance(exc, _E2BSandboxNotFoundException):
        status, hint = "404", "sandbox not found (expired/killed before this op; sandbox_timeout too short for the trajectory)"
    elif _E2BNotFoundException is not None and isinstance(exc, _E2BNotFoundException):
        status, hint = "404", "not found (sandbox or template missing; check E2B_TEMPLATE)"
    elif _E2BAuthException is not None and isinstance(exc, _E2BAuthException):
        status, hint = "401/403", "auth failed (check E2B_API_KEY)"
    else:
        status = str(
            getattr(exc, "status_code", getattr(getattr(exc, "response", None), "status_code", "?"))
        )
    return f"{name} (HTTP {status}){(' - ' + hint) if hint else ''}: {exc}"


def _e2b_is_rate_limit(exc: Exception) -> bool:
    """True if the failure is an E2B 429 (concurrent-sandbox / rate limit)."""
    if _E2BRateLimitException is not None and isinstance(exc, _E2BRateLimitException):
        return True
    status = getattr(exc, "status_code", getattr(getattr(exc, "response", None), "status_code", None))
    return status == 429


def _e2b_is_gone(exc: Exception) -> bool:
    """True if the sandbox no longer exists (E2B 404: expired/killed)."""
    for cls in (_E2BSandboxNotFoundException, _E2BNotFoundException):
        if cls is not None and isinstance(exc, cls):
            return True
    status = getattr(exc, "status_code", getattr(getattr(exc, "response", None), "status_code", None))
    return status == 404


# Quiet the e2b SDK's own context-free "Response <status>" hook now that the
# phase-attributed logs above carry the same information with context. Set
# E2B_QUIET_SDK_LOG=0 to restore the SDK's bare lines.
if os.getenv("E2B_QUIET_SDK_LOG", "1") == "1":
    try:
        logging.getLogger("e2b.api").setLevel(logging.CRITICAL)
    except Exception:
        pass


def _split_conf_list_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            out.extend(_split_conf_list_values(item))
        return out
    return [item.strip() for item in re.split(r"[,\s]+", str(value)) if item.strip()]


def _dedup_keep_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _normalize_sandbox_endpoint(endpoint: str) -> str:
    endpoint = (endpoint or "").strip()
    if not endpoint:
        return ""

    if "://" not in endpoint:
        endpoint = f"http://{endpoint}"

    parsed = urlsplit(endpoint)
    scheme = parsed.scheme or "http"
    netloc = parsed.netloc
    if not netloc and parsed.path:
        path_parts = parsed.path.split("/", 1)
        netloc = path_parts[0]
        path = f"/{path_parts[1]}" if len(path_parts) > 1 else ""
    else:
        path = parsed.path

    path = (path or "").rstrip("/")
    if path == "/run_code":
        path = ""

    return urlunsplit((scheme, netloc, path, parsed.query, parsed.fragment))


def _generate_sandbox_endpoints(hosts: list[str], ports: list[str]) -> list[str]:
    endpoints = []
    for host in hosts:
        host = str(host).strip()
        if not host:
            continue

        host_for_parse = host if "://" in host else f"http://{host}"
        parsed = urlsplit(host_for_parse)
        hostname = parsed.hostname
        if not hostname:
            continue
        scheme = parsed.scheme or "http"

        for port in ports:
            port = str(port).strip()
            if not port:
                continue
            endpoints.append(f"{scheme}://{hostname}:{port}")

    return _dedup_keep_order([_normalize_sandbox_endpoint(endpoint) for endpoint in endpoints if endpoint])


def _load_sandbox_nodes_conf(conf_path: str) -> dict[str, list[str]]:
    parsed = {"hosts": [], "ports": [], "endpoints": []}
    if not conf_path:
        return parsed

    try:
        with open(conf_path, encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.split("#", 1)[0].strip()
                if not line:
                    continue

                if "=" not in line:
                    if "://" in line or "/" in line or re.search(r":\d+$", line):
                        parsed["endpoints"].extend(_split_conf_list_values(line))
                    else:
                        parsed["hosts"].extend(_split_conf_list_values(line))
                    continue

                raw_key, raw_value = line.split("=", 1)
                key = raw_key.strip().lower()
                value = raw_value.strip()
                if key in {"ip", "ips", "node", "nodes", "host", "hosts"}:
                    parsed["hosts"].extend(_split_conf_list_values(value))
                elif key in {"port", "ports"}:
                    parsed["ports"].extend(_split_conf_list_values(value))
                elif key in {
                    "endpoint",
                    "endpoints",
                    "url",
                    "urls",
                    "address",
                    "addresses",
                    "base_url",
                    "base_urls",
                }:
                    parsed["endpoints"].extend(_split_conf_list_values(value))
    except Exception as e:
        logger.warning("[python] Failed to read python_nodes_conf %s: %s", conf_path, e)
        return {"hosts": [], "ports": [], "endpoints": []}

    parsed["hosts"] = _dedup_keep_order(parsed["hosts"])
    parsed["ports"] = _dedup_keep_order(parsed["ports"])
    parsed["endpoints"] = _dedup_keep_order(
        [_normalize_sandbox_endpoint(endpoint) for endpoint in parsed["endpoints"] if endpoint]
    )
    return parsed


def _load_static_sandbox_endpoints(config: dict[str, Any]) -> list[str]:
    raw_endpoints = []
    for value in (
        os.environ.get("SANDBOX_FUSION_ENDPOINTS"),
        config.get("sandbox_fusion_endpoints"),
        os.environ.get("SANDBOX_FUSION_ENDPOINT"),
        config.get("sandbox_fusion_endpoint"),
        os.environ.get("PYTHON_SERVICE_URLS"),
        config.get("python_service_urls"),
        os.environ.get("PYTHON_SERVICE_URL"),
        config.get("python_service_url"),
    ):
        raw_endpoints.extend(_split_conf_list_values(value))

    return _dedup_keep_order(
        [_normalize_sandbox_endpoint(endpoint) for endpoint in raw_endpoints if endpoint]
    )


def _extract_code_from_text(text: str) -> str:
    if not text:
        return ""
    code = text
    xml_match = re.search(r"<code>(.*?)</code>", code, re.DOTALL)
    if xml_match:
        code = xml_match.group(1)
    triple_match = re.search(r"```[^\n]*\n(.*?)```", code, re.DOTALL)
    if triple_match:
        code = triple_match.group(1)
    return code.strip()


def _normalize_code_param(params: Any) -> str:
    if isinstance(params, dict):
        return _extract_code_from_text(str(params.get("code", "") or params.get("raw", "")))
    if isinstance(params, str):
        raw = params
        try:
            parsed = json.loads(params)
            if isinstance(parsed, dict):
                raw = str(parsed.get("code", "") or parsed.get("raw", "") or raw)
        except Exception:
            if json5 is not None:
                try:
                    parsed = json5.loads(params)
                    if isinstance(parsed, dict):
                        raw = str(parsed.get("code", "") or parsed.get("raw", "") or raw)
                except Exception:
                    pass
        return _extract_code_from_text(raw)
    return _extract_code_from_text(str(params))


def _get_python_sandbox_backend(default: str = "sandbox_fusion") -> str:
    backend = str(os.getenv("PYTHON_SANDBOX_BACKEND") or default or "sandbox_fusion").strip().lower()
    if backend not in {"sandbox_fusion", "e2b", "auto"}:
        return "sandbox_fusion"
    return backend


def _has_e2b_config() -> bool:
    return bool(os.getenv("E2B_API_KEY", "").strip())


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _get_optional_float_env(env_name: str, default: float) -> float:
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _autoprint_last_expr(code: str) -> str:
    """Wrap a trailing bare expression in print(repr(...)), imitating Jupyter's
    automatic display. Kept in sync with inference/tool_python.py so training
    and evaluation behave the same.

    sandbox_fusion writes each request to a fresh main.py and runs it as a
    script rather than in a Jupyter kernel, so the value of a trailing bare
    expression is discarded. Without this, a large share of python calls
    produce no output at all, nearly always because the model wrote no print()
    and ended on a bare expression. Telling the model to use print() in the
    tool description does not work, so the tool has to cover it.

    Behaviour: a syntax error is passed through unchanged; a trailing print(...)
    is not wrapped again; a trailing assignment, loop or import is left alone.
    Set PYTHON_AUTOPRINT_LAST_EXPR=0 to disable.
    """
    if os.getenv("PYTHON_AUTOPRINT_LAST_EXPR", "1").strip().lower() in ("0", "false", "no"):
        return code
    try:
        import ast
        tree = ast.parse(code)
    except Exception:
        return code
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return code
    last = tree.body[-1]
    if isinstance(last.value, ast.Call) and getattr(last.value.func, "id", "") == "print":
        return code
    try:
        new = ast.Expr(ast.Call(
            func=ast.Name(id="print", ctx=ast.Load()),
            args=[ast.Call(func=ast.Name(id="repr", ctx=ast.Load()), args=[last.value], keywords=[])],
            keywords=[]))
        tree.body[-1] = ast.fix_missing_locations(new)
        return ast.unparse(tree)
    except Exception:
        return code


def _format_python_result(stdout: str = "", stderr: str = "", timed_out: bool = False, fallback_text: str = (
    "Finished execution with no output. NOTE: this sandbox runs your code as a standalone "
    "script in a FRESH process (python main.py), NOT a Jupyter kernel. Variables from "
    "previous calls do NOT persist, and a bare expression on the last line is NOT displayed. "
    "Make each snippet self-contained and use print() explicitly."
)) -> str:
    result_parts = []
    if stdout:
        result_parts.append(f"stdout:\n{stdout}")
    if stderr:
        result_parts.append(f"stderr:\n{stderr}")
    if timed_out:
        result_parts.append("[Python Interpreter Error] TimeoutError: Execution timed out.")
    return "\n".join(result_parts).strip() or fallback_text


class DeepResearchPythonTool(BaseTool):
    """Python execution tool backed by sandbox_fusion or E2B."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.timeout = int(config.get("timeout", 50))
        self.client_timeout = int(config.get("client_timeout", self.timeout + 30))
        self.max_retries = int(config.get("max_retries") or os.getenv("PYTHON_MAX_RETRIES", "5"))
        self.backend = _get_python_sandbox_backend(str(config.get("backend", "sandbox_fusion")))
        self._python_nodes_conf = str(os.environ.get("PYTHON_NODES_CONF") or config.get("python_nodes_conf") or "").strip()
        self._static_sandbox_endpoints = _load_static_sandbox_endpoints(config)
        if self._python_nodes_conf or self._static_sandbox_endpoints:
            logger.info(
                "[python] sandbox backend configured (backend=%s, nodes_conf=%s, static_endpoints=%s)",
                self.backend,
                self._python_nodes_conf or "(none)",
                ",".join(self._static_sandbox_endpoints) or "(none)",
            )

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {}
        return instance_id, ToolResponse()

    def _get_sandbox_endpoints(self) -> list[str]:
        if self._python_nodes_conf:
            conf = _load_sandbox_nodes_conf(self._python_nodes_conf)
            if conf["endpoints"]:
                return conf["endpoints"]
            if conf["hosts"] and conf["ports"]:
                return _generate_sandbox_endpoints(conf["hosts"], conf["ports"])
        return self._static_sandbox_endpoints

    def _resolve_backend(self) -> str:
        if self.backend != "auto":
            return self.backend
        if _has_e2b_config():
            return "e2b"
        return "sandbox_fusion"

    def _run_with_sandbox_fusion(self, code: str) -> tuple[str, dict]:
        endpoints = self._get_sandbox_endpoints()
        if not endpoints:
            return (
                "[Python Interpreter Error]: sandbox endpoint is not configured. Set SANDBOX_FUSION_ENDPOINT or python_nodes_conf.",
                {"error": "missing_sandbox_endpoint", "backend": "sandbox_fusion"},
            )

        try:
            from sandbox_fusion import RunCodeRequest, run_code
        except ImportError as e:
            return (
                f"[Python Interpreter Error]: sandbox_fusion is not installed: {e}",
                {"error": "missing_sandbox_fusion", "backend": "sandbox_fusion"},
            )

        last_error = None
        attempts = max(1, self.max_retries)
        retry_endpoints = list(endpoints)
        if len(retry_endpoints) > 1:
            random.shuffle(retry_endpoints)
        for attempt in range(attempts):
            endpoint = retry_endpoints[attempt % len(retry_endpoints)]
            if attempt >= len(retry_endpoints) and len(retry_endpoints) > 1 and attempt % len(retry_endpoints) == 0:
                random.shuffle(retry_endpoints)
            try:
                started_at = time.time()
                code_result = run_code(
                    RunCodeRequest(code=_autoprint_last_expr(code), language="python", run_timeout=self.timeout),
                    max_attempts=1,
                    client_timeout=self.client_timeout,
                    endpoint=endpoint,
                )
                elapsed = time.time() - started_at
                run_result = getattr(code_result, "run_result", None)
                stdout = _stringify(getattr(run_result, "stdout", "") or "")
                stderr = _stringify(getattr(run_result, "stderr", "") or "")
                execution_time = getattr(run_result, "execution_time", 0.0) or 0.0
                text = _format_python_result(
                    stdout=stdout,
                    stderr=stderr,
                    timed_out=execution_time >= self.timeout - 1,
                )
                return text, {
                    "endpoint": endpoint,
                    "execution_time": round(elapsed, 6),
                    "status": "success",
                    "backend": "sandbox_fusion",
                }
            except RequestsTimeout:
                last_error = f"[Python Interpreter Error] TimeoutError: Execution timed out on endpoint {endpoint}."
            except Exception as e:
                last_error = f"[Python Interpreter Error]: {e} on endpoint {endpoint}"
            logger.warning("[python] sandbox_fusion attempt %s/%s failed: %s", attempt + 1, attempts, last_error)
            if attempt < attempts - 1:
                _cap = float(os.getenv("PYTHON_RETRY_BACKOFF_CAP", "4"))
                time.sleep(min(_cap, 0.3 * (2 ** attempt)) * (0.5 + random.random()))

        return last_error or "[Python Interpreter Error]: All attempts failed.", {
            "error": "execution_failed",
            "backend": "sandbox_fusion",
        }

    def _run_with_e2b(self, code: str) -> tuple[str, dict]:
        api_key = os.getenv("E2B_API_KEY", "").strip()
        if not api_key:
            return (
                "[Python Interpreter Error]: E2B_API_KEY is not configured.",
                {"error": "missing_e2b_api_key", "backend": "e2b"},
            )

        try:
            from e2b_code_interpreter import Sandbox
        except ImportError as e:
            return (
                f"[Python Interpreter Error]: e2b_code_interpreter is not installed: {e}",
                {"error": "missing_e2b", "backend": "e2b"},
            )

        # Match mirothinker's stateless E2B style as closely as possible.
        default_template = "1av7fdjfvcparqo8efq6"
        template = os.getenv("E2B_TEMPLATE", "").strip() or default_template
        sandbox_timeout = min(self.timeout, 300)

        # Resilience knobs (env-overridable) so a Free-plan concurrency cap
        # (20 sandboxes) does not turn into failed executions:
        #   - create 429 -> wait for a slot and retry, instead of hard-failing.
        #   - run_code 404 -> the sandbox expired; recreate a fresh one and retry
        #     (retrying the SAME dead sandbox, as before, could never recover).
        create_max_attempts = max(1, int(os.getenv("E2B_CREATE_MAX_RETRIES", "8")))
        create_backoff_cap = float(os.getenv("E2B_CREATE_BACKOFF_CAP_S", "20"))
        session_max_attempts = max(1, int(os.getenv("E2B_SESSION_MAX_RETRIES", "3")))
        run_max_attempts = 2

        started_at = time.time()
        last_error: Optional[Exception] = None

        for session in range(1, session_max_attempts + 1):
            sandbox = None
            recreate = False
            try:
                # --- create sandbox, backing off on 429 until a slot frees ---
                # e2b>=2.x exposes Sandbox.create(...); e2b 1.x uses Sandbox(...).
                # The installed e2b-code-interpreter is 2.x, whose Sandbox.__init__
                # rejects template/timeout/api_key -> the bare Sandbox(...) path
                # TypeErrors on every call, so prefer create().
                for c_attempt in range(1, create_max_attempts + 1):
                    try:
                        if hasattr(Sandbox, "create"):
                            sandbox = Sandbox.create(
                                template=template,
                                timeout=sandbox_timeout,
                                api_key=api_key,
                            )
                        else:
                            sandbox = Sandbox(
                                template=template,
                                timeout=sandbox_timeout,
                                api_key=api_key,
                            )
                        break
                    except Exception as e:
                        if _e2b_is_rate_limit(e) and c_attempt < create_max_attempts:
                            delay = min(2.0 ** c_attempt, create_backoff_cap) + random.uniform(0, 1.0)
                            logger.warning(
                                "[python][e2b] create rate-limited (429), waiting %.1fs for a "
                                "sandbox slot (attempt %s/%s): %s",
                                delay,
                                c_attempt,
                                create_max_attempts,
                                _e2b_error_summary(e),
                            )
                            time.sleep(delay)
                            continue
                        logger.warning(
                            "[python][e2b] create failed (template=%s timeout=%ss): %s",
                            template,
                            sandbox_timeout,
                            _e2b_error_summary(e),
                        )
                        raise

                sandbox_id = getattr(sandbox, "sandbox_id", "?")

                # --- run code; a 404 means the sandbox is gone -> recreate ---
                execution = None
                for attempt in range(1, run_max_attempts + 1):
                    try:
                        execution = sandbox.run_code(_autoprint_last_expr(code))
                        break
                    except Exception as e:
                        last_error = e
                        logger.warning(
                            "[python][e2b] run_code failed (sandbox=%s session=%s/%s attempt=%s/%s): %s",
                            sandbox_id,
                            session,
                            session_max_attempts,
                            attempt,
                            run_max_attempts,
                            _e2b_error_summary(e),
                        )
                        if _e2b_is_gone(e):
                            # Dead sandbox: stop retrying it; recreate if budget left.
                            recreate = session < session_max_attempts
                            break
                        if attempt == run_max_attempts:
                            raise e

                if execution is None:
                    if recreate:
                        continue  # finally kills the dead sandbox; outer loop recreates
                    raise last_error  # gone on the last session, or otherwise unrecovered

                elapsed = time.time() - started_at
                logs = getattr(execution, "logs", None)
                stdout = _stringify(getattr(logs, "stdout", "") if logs is not None else "")
                stderr = _stringify(getattr(logs, "stderr", "") if logs is not None else "")
                execution_error = _stringify(getattr(execution, "error", "") or "")
                if execution_error:
                    stderr = f"{stderr}\n{execution_error}".strip() if stderr else execution_error
                if not stdout and getattr(execution, "text", None):
                    stdout = _stringify(getattr(execution, "text", None))
                text = _format_python_result(stdout=stdout, stderr=stderr, fallback_text=str(execution))
                return text, {
                    "status": "success",
                    "backend": "e2b",
                    "execution_time": round(elapsed, 6),
                    "e2b_sessions": session,
                }
            except RequestsTimeout:
                return (
                    "[Python Interpreter Error] TimeoutError: Execution timed out.",
                    {"error": "timeout", "backend": "e2b"},
                )
            except Exception as e:
                # create/run_code failures are already logged with phase context
                # above; keep the E2B status on the metrics so it is queryable
                # downstream even when phase logs are filtered out.
                return (
                    f"[Python Interpreter Error]: {str(e)}",
                    {
                        "error": "execution_failed",
                        "backend": "e2b",
                        "e2b_error": _e2b_error_summary(e),
                    },
                )
            finally:
                if sandbox is not None:
                    try:
                        sandbox.kill()
                    except Exception as e:
                        # A kill-time 404 is benign (sandbox already expired/gone);
                        # log at debug so it explains the paired e2b "Response 404"
                        # without adding ERROR noise.
                        logger.debug(
                            "[python][e2b] kill failed (sandbox=%s): %s",
                            getattr(sandbox, "sandbox_id", "?"),
                            _e2b_error_summary(e),
                        )

        # All sessions exhausted (repeated 404 recreations never succeeded).
        return (
            f"[Python Interpreter Error]: {last_error}",
            {
                "error": "execution_failed",
                "backend": "e2b",
                "e2b_error": _e2b_error_summary(last_error) if last_error else "sandbox unavailable",
            },
        )

    def _run_python(self, parameters: Any) -> tuple[str, dict]:
        code = _normalize_code_param(parameters)
        if not code:
            return "[Python Interpreter Error]: Empty code.", {"error": "empty_code"}

        backend = self._resolve_backend()
        if backend == "e2b":
            return self._run_with_e2b(code)
        return self._run_with_sandbox_fusion(code)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        loop = asyncio.get_running_loop()
        text, metrics = await loop.run_in_executor(None, self._run_python, parameters)
        return ToolResponse(text=text), 0.0, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
