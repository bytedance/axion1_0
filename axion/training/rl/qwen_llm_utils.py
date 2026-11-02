#
# Modified by the Axion authors from the DeepResearch recipe: comments and
# log messages translated to English, hard-coded site paths replaced with
# environment variables or relative paths, and bug fixes. Unmodified files in
# this directory carry their original headers.
#
"""Shared helpers for driving the local Qwen3.5-27B eval/summary/memory vLLM.

Centralises three concerns shared by the auxiliary LLM call sites (reward
judges, obj/citation judges, memory condenser, visit summary):

  1. Thinking control. Qwen3.5 has thinking ON by default and, unless the
     server is launched with a reasoning parser, leaks the whole reasoning
     trace into ``message.content`` (typically ``...reasoning...</think>answer``,
     the opening ``<think>`` living in the chat template rather than the
     output). ``enable_thinking=False`` turns it off at the source; strip_thinking()
     is the belt-and-suspenders cleanup for any response that still carries it.

  2. Qwen3.5 recommended sampling parameters, split by task family.

  3. Provider awareness. ``top_k`` / ``min_p`` / ``repetition_penalty`` and
     ``chat_template_kwargs`` are vLLM-only extensions and must travel via
     ``extra_body``; sending them to Azure/OpenAI errors out. So the vLLM-only
     bits are emitted only when the target is the local vLLM, letting Azure
     stay a valid (if unselected) fallback.

All values follow the Qwen3.5 team's published recommendations.
"""

import os
import re

# --- Qwen3.5 recommended sampling parameters -------------------------------
# Only the non-thinking ("instruct") profiles are wired up today because every
# auxiliary call runs with thinking disabled; the thinking profiles are kept
# for completeness / future use.
QWEN35_SAMPLING = {
    # Thinking mode, general tasks.
    "thinking_general": {
        "temperature": 1.0, "top_p": 0.95, "top_k": 20,
        "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0,
    },
    # Thinking mode, precise coding tasks (e.g. WebDev).
    "thinking_coding": {
        "temperature": 0.6, "top_p": 0.95, "top_k": 20,
        "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0,
    },
    # Instruct (non-thinking) mode, general tasks -> summaries, memory condense.
    "instruct_general": {
        "temperature": 0.7, "top_p": 0.8, "top_k": 20,
        "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0,
    },
    # Instruct (non-thinking) mode, reasoning tasks -> LLM-as-judge / graders.
    "instruct_reasoning": {
        "temperature": 1.0, "top_p": 0.95, "top_k": 20,
        "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0,
    },
    # Binary graders that emit a single token of verdict ("true"/"false",
    # "correct: yes/no"). These feed the RL reward directly, so they are sampled
    # greedily: a sampling profile tuned for reasoning makes the verdict
    # unstable on borderline answers, which shows up as the same
    # (question, reference, prediction) triple flipping between runs.
    # presence_penalty must be 0 as well -- it is meaningless for a one-token
    # answer and only perturbs the logits.
    "judge_binary": {
        "temperature": 0.0, "top_p": 1.0, "top_k": -1,
        "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0,
    },
}

# Standard OpenAI chat params (safe for Azure/OpenAI too) vs vLLM-only ones.
_OPENAI_STD_KEYS = ("temperature", "top_p", "presence_penalty")
_VLLM_ONLY_KEYS = ("top_k", "min_p", "repetition_penalty")

# Allow an operator to globally flip behaviour without code edits.
_DEFAULT_ENABLE_THINKING = os.environ.get("QWEN_ENABLE_THINKING", "0") == "1"


# --- Global backend force switch ------------------------------------------
# The platform env template re-injects EVAL_LLM_PROVIDER=azure / model=gpt-5.5
# into the trainer environment, overriding the launcher's local exports. This
# switch is consulted with the HIGHEST priority in every provider resolver so
# that, when set, all auxiliary LLM traffic is pinned to the local Qwen3.5 vLLM
# regardless of the polluted per-profile env. Azure remains fully wired up and
# is used again the moment DR_LLM_BACKEND is unset (or set to "azure").


def force_vllm_enabled():
    """True when DR_LLM_BACKEND selects the local vLLM as the forced backend."""
    return os.environ.get("DR_LLM_BACKEND", "").strip().lower() in {"vllm", "local", "local_openai"}


def vllm_model_name():
    """The served model name to force on the local vLLM (default qwen35-27b-eval)."""
    return (os.environ.get("DR_LLM_VLLM_MODEL") or "qwen35-27b-eval").strip() or "qwen35-27b-eval"


def _normalize_base_url(x):
    x = x.strip()
    if not x:
        return ""
    if not x.startswith("http://") and not x.startswith("https://"):
        x = "http://" + x
    x = x.rstrip("/")
    return x if x.endswith("/v1") else x + "/v1"


def vllm_base_urls():
    """Local vLLM OpenAI-compatible base URLs (each ending in /v1).

    Priority: DR_LLM_VLLM_BASE_URLS, then LOCAL_OPENAI_BASE_URLS /
    LOCAL_OPENAI_BASE_URL, then MEMORY_API_BASE (the 27B summary/memory pool),
    then derived from EVAL_LLM_IPS + EVAL_LLM_PORTS.
    """
    raw = (
        os.environ.get("DR_LLM_VLLM_BASE_URLS")
        or os.environ.get("LOCAL_OPENAI_BASE_URLS")
        or os.environ.get("LOCAL_OPENAI_BASE_URL")
        or ""
    ).strip()
    urls = []
    if raw:
        for x in re.split(r"[,\s]+", raw):
            u = _normalize_base_url(x)
            if u:
                urls.append(u)
        return urls
    # Prefer the 27B memory/summary pool (MEMORY_API_BASE, e.g. 6000-6002) over a
    # raw EVAL_LLM_PORTS fan-out: EVAL_LLM_PORTS also includes the split-out 9B
    # judge port (6003), which rejects this util's 27B served-model name
    # (vllm_model_name()) with a 404 and wastes retries. Mirrors memory.py's guard.
    mem_api_base = (os.environ.get("MEMORY_API_BASE") or "").strip()
    if mem_api_base:
        urls = [u for u in (_normalize_base_url(x) for x in re.split(r"[,\s]+", mem_api_base) if x.strip()) if u]
        if urls:
            return urls
    ips = [x for x in re.split(r"[,\s]+", (os.environ.get("EVAL_LLM_IPS") or "").strip()) if x]
    ports = [x for x in re.split(r"[,\s]+", (os.environ.get("EVAL_LLM_PORTS") or "6000,6001").strip()) if x]
    for ip in ips:
        host = f"[{ip}]" if (not ip.startswith("[") and ip.count(":") >= 2) else ip
        for p in ports:
            if p.isdigit():
                urls.append(f"http://{host}:{p}/v1")
    return urls


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text):
    """Remove Qwen thinking traces from a model response.

    Handles three shapes: a full ``<think>...</think>`` block, a dangling
    ``</think>`` with the opening tag supplied by the chat template, and a lone
    unterminated ``<think>`` (truncated output). Non-str input is returned
    unchanged.
    """
    if not isinstance(text, str) or not text:
        return text
    # 1) Drop any complete <think>...</think> blocks.
    cleaned = _THINK_BLOCK_RE.sub("", text)
    # 2) Dangling close tag (opening tag was in the prompt/template): keep the
    #    tail after the last </think>.
    lower = cleaned.lower()
    if "</think>" in lower:
        idx = lower.rfind("</think>")
        cleaned = cleaned[idx + len("</think>"):]
    # 3) Lone opening tag with no close (truncated reasoning): nothing useful
    #    after it, keep only the pre-think text.
    lower = cleaned.lower()
    if "<think>" in lower:
        cleaned = cleaned[: lower.find("<think>")]
    return cleaned.strip()


def strip_thinking_from_message(message):
    """Strip thinking from an OpenAI-style message object or dict, in place-ish.

    Returns the cleaned content string. Accepts a pydantic message (``.content``)
    or a mapping (``["content"]``).
    """
    content = None
    if hasattr(message, "content"):
        content = message.content
        cleaned = strip_thinking(content)
        try:
            message.content = cleaned
        except Exception:
            pass
        return cleaned
    if isinstance(message, dict):
        cleaned = strip_thinking(message.get("content"))
        message["content"] = cleaned
        return cleaned
    return strip_thinking(message)


def qwen_sampling_kwargs(mode="instruct_reasoning", is_vllm=True,
                         enable_thinking=None):
    """Build request kwargs + extra_body for a Qwen3.5 chat.completions call.

    Args:
        mode: key into QWEN35_SAMPLING.
        is_vllm: when True, emit vLLM-only params (top_k/min_p/repetition_penalty)
            and chat_template_kwargs via extra_body. When False (Azure/OpenAI),
            only the standard OpenAI sampling params are returned so the call
            does not fail on unknown fields.
        enable_thinking: override; defaults to env QWEN_ENABLE_THINKING (off).

    Returns:
        (kwargs, extra_body) where kwargs holds standard OpenAI params and
        extra_body holds vLLM-only params (empty dict when not is_vllm).
    """
    if enable_thinking is None:
        enable_thinking = _DEFAULT_ENABLE_THINKING
    params = QWEN35_SAMPLING[mode]
    kwargs = {k: params[k] for k in _OPENAI_STD_KEYS}
    extra_body = {}
    if is_vllm:
        extra_body = {k: params[k] for k in _VLLM_ONLY_KEYS}
        # enable_thinking is consumed by the Qwen chat template.
        extra_body["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}
    return kwargs, extra_body


def apply_qwen_params(request_kwargs, mode="instruct_reasoning", is_vllm=True,
                      enable_thinking=None, overwrite=False):
    """Merge Qwen3.5 sampling params into an existing chat kwargs dict.

    vLLM-only params and chat_template_kwargs are merged into
    request_kwargs["extra_body"]. Existing caller-supplied values are preserved
    unless overwrite=True.
    """
    kwargs, extra_body = qwen_sampling_kwargs(
        mode=mode, is_vllm=is_vllm, enable_thinking=enable_thinking
    )
    for k, v in kwargs.items():
        if overwrite or k not in request_kwargs or request_kwargs.get(k) is None:
            request_kwargs[k] = v
    if extra_body:
        merged = dict(request_kwargs.get("extra_body") or {})
        for k, v in extra_body.items():
            if k == "chat_template_kwargs":
                ctk = dict(merged.get("chat_template_kwargs") or {})
                ctk.update(v)
                merged["chat_template_kwargs"] = ctk
            elif overwrite or k not in merged:
                merged[k] = v
        request_kwargs["extra_body"] = merged
    return request_kwargs
