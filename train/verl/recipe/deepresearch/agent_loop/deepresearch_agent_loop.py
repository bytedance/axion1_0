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
DeepResearch Agent Loop for verl framework.
Implements multi-turn tool calling with memory/condenser support.
"""

import asyncio
import copy
import json
import logging
import os
import re
import time
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.chat_template import (
    apply_chat_template as safe_apply_chat_template,
    initialize_system_prompt,
)
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

from ..memory import (
    build_fallback_memory_state,
    call_condenser_async,
    extract_events_from_messages,
    format_prev_state_section,
    parse_memory_result,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, fallback to default=%s", name, raw, default)
        return default


def _is_rank_zero() -> bool:
    for env in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        raw = os.getenv(env)
        if raw is not None:
            try:
                return int(raw) == 0
            except ValueError:
                return False
    return True


def _find_subseq(seq, subseq, start=0):
    """Find the starting index of subseq in seq, starting from start."""
    for i in range(start, len(seq) - len(subseq) + 1):
        if seq[i:i + len(subseq)] == subseq:
            return i
    return -1



def _diagnose_tool_call_json(payload) -> str:
    """Turn __invalid_tool_call_json__ into a message the model can act on.

    The two common causes are a response truncated by max_response_length
    before </code> or the closing brace, and a tool call written as
    {"name":"PythonInterpreter","arguments":{}} followed by <code> with no
    closing </code>. Echoing the raw payload back leaves the model guessing,
    so this names the actual defect instead.
    """
    s = payload if isinstance(payload, str) else str(payload)
    head = s[:200] + ("..." if len(s) > 200 else "")
    if "<code>" in s and "</code>" not in s:
        return ("Your tool_call was cut off: found <code> without a closing </code>. "
                "Either keep the code short enough to finish in one turn, or pass it as a JSON "
                'string in arguments.code, e.g. {"name": "PythonInterpreter", "arguments": '
                '{"code": "print(1)"}}. Received (truncated): ' + head)
    if s.count("{") > s.count("}") or s.count('"') % 2 == 1:
        return ("Your tool_call JSON is incomplete (unbalanced braces or quotes) - it was most "
                "likely truncated. Retry with a shorter tool_call, or split the work across turns. "
                "Received (truncated): " + head)
    return ("Your tool_call is not valid JSON. It must be a single JSON object with a string "
            '"name" and an object "arguments". Received (truncated): ' + head)

class AgentState(Enum):
    """Agent state machine states."""
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    MEMORY_TRIGGER = "memory_trigger"
    FORCING_ANSWER = "forcing_answer"
    TERMINATED = "terminated"


SEARCH_TOOL_PROMPT = """{"type": "function", "function": {"name": "search", "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}}"""

SCHOLAR_TOOL_PROMPT = """{"type": "function", "function": {"name": "google_scholar", "description": "Leverage Google Scholar to retrieve relevant information from academic publications. Accepts multiple queries. This tool will also return results from google search", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries for Google Scholar."}}, "required": ["query"]}}}"""

PYTHON_TOOL_PROMPT = """{"type": "function", "function": {"name": "PythonInterpreter", "description": "Executes Python code in a sandboxed environment. To use this tool, you must follow this format:\n1. The 'arguments' JSON object must be empty: {}.\n2. The Python code to be executed must be placed immediately after the JSON block, enclosed within <code> and </code> tags.\n\nIMPORTANT: Any output you want to see MUST be printed to standard output using the print() function.\n\nExample of a correct call:\n<tool_call>\n{\"name\": \"PythonInterpreter\", \"arguments\": {}}\n<code>\nimport numpy as np\n# Your code here\nprint(f\"The result is: {np.mean([1,2,3])}\")\n</code>\n</tool_call>", "parameters": {"type": "object", "properties": {}}}}"""

VISIT_TOOL_PROMPT = """{"type": "function", "function": {"name": "visit", "description": "Visit webpage(s) and return the summary of the content.", "parameters": {"type": "object", "properties": {"url": {"type": "array", "items": {"type": "string"}, "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."}, "goal": {"type": "string", "description": "The specific information goal for visiting webpage(s)."}}, "required": ["url", "goal"]}}}"""


# System prompt for DeepResearch agent
SYSTEM_PROMPT = """You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
""" + SEARCH_TOOL_PROMPT + """
""" + SCHOLAR_TOOL_PROMPT + """
""" + PYTHON_TOOL_PROMPT + """
""" + VISIT_TOOL_PROMPT + """
</tools>

# Using prev_state (Research State Summary)

If you see a "RESEARCH STATE SUMMARY (prev_state)" section in the user message, it contains a compressed summary of previous research progress. Use it to:

1. **Avoid redundant work**:
   - Check `search_queries` to avoid repeating searches that have already been executed.
   - Check `visited_sources` to avoid visiting URLs that have already been visited.

2. **Use verified information**:
   - Check `information_state.trusted` for facts that have been verified from visited sources. You can use these directly in your answer without re-searching or re-visiting.
   - Check `information_state.untrusted` for claims that have been contradicted or proven unreliable.

3. **Follow up on uncertain information**:
   - Check `information_state.uncertain` for claims that need more evidence. The `need` field specifies the exact next action (e.g., "visit <URL>" or "search <query>") to resolve the uncertainty.

IMPORTANT: Do NOT search for or visit information that is already in `prev_state`, unless it's insufficient to answer the user's question. Only in this case, you are encouraged to search for more information or even visit the same URL. Instead, use the information from `prev_state` directly, or follow the specific actions suggested in `information_state.uncertain.need` if more information is needed.

The final answer must exclude any information that remains uncertain or pending. All statements included must be fully verified.

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Current date: """


def _truncate_routed_experts(experts, prompt_ids, response_ids, response_length):
    # routed experts of the session's last request: [len(prompt)+len(response)-1, layers, topk]; clip to the row
    if experts is None:
        return None
    n = len(prompt_ids) + min(len(response_ids), response_length)
    return experts[:n]


class SessionData:
    """Data for a completed session (segment between condenser triggers)."""

    def __init__(
        self,
        session_id: int,
        prompt_ids: List[int],
        original_user_content: str,
        memory_state: Optional[Dict] = None,
    ):
        self.session_id = session_id
        self.prompt_ids = prompt_ids
        self.original_user_content = original_user_content
        self.memory_state = memory_state  # prev_state at start of this session

        # Response data (filled in after generation)
        self.response_ids: List[int] = []
        self.response_mask: List[int] = []
        self.response_logprobs: List[float] = []
        self.routed_experts = None  # np.ndarray [T-1, layers, topk] from the last request of this session
        self.response_text: str = ""
        self.messages: List[Dict[str, Any]] = []


class AgentData:
    """Encapsulates all state variables for the DeepResearch agent loop."""

    def __init__(
        self,
        messages: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        request_id: str,
        tools_kwargs: Dict[str, Any],
        original_user_content: str,
    ):
        self.messages = messages
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.original_user_content = original_user_content

        # State variables
        self.prompt_ids: List[int] = []
        self.response_ids: List[int] = []
        self.response_mask: List[int] = []
        self.response_logprobs: List[float] = []
        self.routed_experts = None  # last generation request's routed experts (covers the session so far)
        self.turn_scores: List[float] = []
        self.tool_rewards: List[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Tool calls from current response
        self.tool_calls: List[Dict[str, Any]] = []

        # Memory state
        self.pending_messages: Optional[List[Dict[str, Any]]] = None
        self.memory_state: Optional[Dict] = None
        self.session_id: int = 0
        self.condenser_call_id: int = 0
        self.condenser_update_id: int = 0

        # Completed sessions (each condenser trigger saves a session)
        self.completed_sessions: List[SessionData] = []

        # Full response tracking for reward computation (accumulates across all sessions)
        self.full_response: str = ""
        # Current session's response text (reset on condenser trigger)
        self.session_response: str = ""

        # Extra fields for partial-rollout recovery metadata.
        self.extra_fields: Dict[str, Any] = {}

        # Termination reason flags
        self.terminated_by_consecutive_invalid: bool = False
        # Set once a forced final-answer turn has been attempted (FORCE_FINAL_ANSWER)
        self.forced_answer_done: bool = False

        # Full messages tracking (accumulates across all sessions, no condenser rewrite)
        self.full_messages: List[Dict[str, Any]] = []


@register("deepresearch_agent")
class DeepResearchAgentLoop(AgentLoopBase):
    """DeepResearch Agent Loop with multi-turn tool calling and memory support.

    This agent loop implements the DeepResearch pattern with:
    - Multi-turn tool calling (search, google_scholar, PythonInterpreter, visit)
    - Memory/condenser system for context compression
    - Session-based generation with state management
    """

    _class_initialized = False

    def __init__(self, trainer_config, server_manager, tokenizer, processor, dataset_cls, dataset_config, **kwargs):
        super().__init__(trainer_config, server_manager, tokenizer, processor, dataset_cls, dataset_config, **kwargs)
        self.init_class(self.config, tokenizer, processor)

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        logger.info("Performing class-level DeepResearchAgentLoop initialization")

        # Initialize tokenizer and processor
        cls.tokenizer = tokenizer
        cls.processor = processor

        # Multi-turn configuration
        cls.max_turns = config.actor_rollout_ref.rollout.multi_turn.get("max_turns", 100)
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.get("max_user_turns", 50)
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.get("max_assistant_turns", 50)
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.get("max_parallel_calls", 1)
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.get("max_tool_response_length", 10000)
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.get("tool_response_truncate_side", "middle")

        # Memory configuration
        # In fully-async mode, MultiTurnConfig is structured and may reject extra keys.
        # Keep backward compatibility by allowing env vars to override these knobs.
        cls.memory_enabled = _env_bool(
            "MEMORY_ENABLED", config.actor_rollout_ref.rollout.multi_turn.get("memory_enabled", True)
        )
        cls.context_threshold = _env_int(
            "CONTEXT_THRESHOLD", config.actor_rollout_ref.rollout.multi_turn.get("context_threshold", 16000)
        )
        cls.memory_tier1_enabled = _env_bool("MEMORY_TIER1", True)
        cls.memory_tier1_keep_msgs = _env_int("MEMORY_TIER1_KEEP_MSGS", 6)
        cls.memory_tier1_stub_chars = _env_int("MEMORY_TIER1_STUB_CHARS", 240)
        cls.memory_tier1_target = _env_int("MEMORY_TIER1_TARGET", 0)

        # Tool configuration
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.get("tool_config_path")
        if tool_config_path:
            tool_list = initialize_tools_from_config(tool_config_path)
            cls.tools = {tool.name: tool for tool in tool_list}
            cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        else:
            cls.tools = {}
            cls.tool_schemas = []
        logger.info(f"Initialized tools: {list(cls.tools.keys())}")

        # Chat template configuration
        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.turn_response_length = _env_int("MAX_TURN_RESPONSE_LENGTH", cls.response_length)
        # Weak-model answer-completion aids (default OFF; enabled only by the 4B
        # launcher). FORCE_FINAL_ANSWER: on no-answer termination (max-turns or
        # 3x consecutive-invalid) do one extra prompted turn to elicit a final
        # <answer> instead of dumping an answerless trajectory (which scores 0 and
        # gives GRPO no signal). SOFTEN_ANSWER_PROMPT: relax the "only answer if
        # fully verified" instruction so an uncertain model still commits.
        cls.force_final_answer_enabled = _env_bool("FORCE_FINAL_ANSWER", False)
        cls.soften_answer_prompt = _env_bool("SOFTEN_ANSWER_PROMPT", False)
        # vLLM model context limit; used to avoid requesting generation when prompt + response would exceed it
        _max_model_len = getattr(config.actor_rollout_ref.rollout, "max_model_len", None)
        cls.model_max_len = _max_model_len if _max_model_len is not None else (cls.prompt_length + cls.response_length)

        # System prompt tokens
        cls.system_prompt_tokens = initialize_system_prompt(tokenizer, **cls.apply_chat_template_kwargs)

        # Pre-compute turn boundary token sequences
        cls._compute_turn_boundary_tokens()

        # Per-turn printing controlled by env var
        cls._print_turns = os.getenv("DEEPRESEARCH_PRINT_TURNS", "0").strip().lower() in {
            "1", "true", "yes", "y", "on",
        }
        try:
            cls._print_turns_max_chars = max(64, int(os.getenv("DEEPRESEARCH_PRINT_TURNS_MAX_CHARS", "1000")))
        except ValueError:
            cls._print_turns_max_chars = 1000
        try:
            cls._print_turns_budget = max(0, int(os.getenv("DEEPRESEARCH_PRINT_TURNS_BUDGET", "0")))
        except ValueError:
            cls._print_turns_budget = 0
        cls._print_turns_used = 0

        # Per-search-query printing (the "[tool] req=.. query=.." lines). Off by
        # default: on rank 0 this fires once per query and, with dozens of queries
        # per rollout, dominates the training log. Set DEEPRESEARCH_PRINT_SEARCH=1
        # to re-enable for debugging.
        cls._print_search = os.getenv("DEEPRESEARCH_PRINT_SEARCH", "0").strip().lower() in {
            "1", "true", "yes", "y", "on",
        }

        # Real-time trajectory streaming directory (set env var to enable)
        cls._stream_dir = os.getenv("DEEPRESEARCH_STREAM_DIR", "")

    @staticmethod
    def _remove_prev_state_from_messages(messages: List[Dict]) -> List[Dict]:
        """Remove prev_state sections from user messages to get the original conversation."""
        cleaned = copy.deepcopy(messages)
        for msg in cleaned:
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                marker = "====================\nRESEARCH STATE SUMMARY"
                idx = msg["content"].find(marker)
                if idx != -1:
                    msg["content"] = msg["content"][:idx].rstrip()
        return cleaned

    @staticmethod
    def _unwrap_scalar(value: Any) -> Any:
        """Convert common scalar wrappers (e.g., numpy scalar) to plain python values."""
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return value
        return value

    @classmethod
    def _safe_path_component(cls, value: Any, default: str) -> str:
        """Sanitize arbitrary value for filesystem-safe path component."""
        value = cls._unwrap_scalar(value)
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        text = str(value).strip() if value is not None else ""
        if not text:
            text = default
        text = text.replace(os.sep, "_")
        if os.altsep:
            text = text.replace(os.altsep, "_")
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
        text = text.strip("._-")
        return text[:128] if text else default

    @classmethod
    def _extract_stream_task_id(cls, kwargs: Dict[str, Any]) -> str:
        """Extract task id from agent-loop kwargs for stream folder naming."""
        reward_model = cls._unwrap_scalar(kwargs.get("reward_model"))
        if isinstance(reward_model, dict):
            for key in ("task_id", "original_task_id", "question_id"):
                value = cls._unwrap_scalar(reward_model.get(key))
                if value is not None and str(value).strip():
                    return cls._safe_path_component(value, default="unknown_task")

            ground_truth = cls._unwrap_scalar(reward_model.get("ground_truth"))
            if isinstance(ground_truth, dict):
                for key in ("task_id", "original_task_id", "question_id"):
                    value = cls._unwrap_scalar(ground_truth.get(key))
                    if value is not None and str(value).strip():
                        return cls._safe_path_component(value, default="unknown_task")

        extra_info = cls._unwrap_scalar(kwargs.get("extra_info"))
        if isinstance(extra_info, dict):
            for key in ("task_id", "original_task_id", "question_id", "id"):
                value = cls._unwrap_scalar(extra_info.get(key))
                if value is not None and str(value).strip():
                    return cls._safe_path_component(value, default="unknown_task")

        for key in ("task_id", "id", "index", "uid"):
            value = cls._unwrap_scalar(kwargs.get(key))
            if value is not None and str(value).strip():
                return cls._safe_path_component(value, default="unknown_task")

        return "unknown_task"

    @classmethod
    def _extract_stream_step(cls, kwargs: Dict[str, Any]) -> str:
        """Extract current global step for stream folder naming."""
        for key in ("trajectory_step", "global_steps", "step"):
            if key not in kwargs:
                continue
            value = cls._unwrap_scalar(kwargs.get(key))
            if value is None:
                continue
            try:
                return str(int(value))
            except Exception:
                text = str(value).strip()
                if text:
                    return cls._safe_path_component(text, default="unknown")
        return "unknown"

    @classmethod
    def _extract_stream_rollout(cls, kwargs: Dict[str, Any]) -> str:
        """Extract rollout index (fallback to uid) for stream folder naming."""
        for key in ("trajectory_rollout_n", "rollout_n", "rollout_index"):
            if key not in kwargs:
                continue
            value = cls._unwrap_scalar(kwargs.get(key))
            if value is None:
                continue
            try:
                return str(int(value))
            except Exception:
                text = str(value).strip()
                if text:
                    return cls._safe_path_component(text, default="0")

        uid = cls._unwrap_scalar(kwargs.get("uid"))
        if uid is not None and str(uid).strip():
            return cls._safe_path_component(uid, default="0")
        return "0"

    def _stream_trajectory(
        self,
        agent_data: AgentData,
        question: str,
        is_final: bool,
        trigger_reason: str,
        stream_filepath: str,
    ) -> None:
        """Append a JSONL entry to the streaming trajectory file.

        Called after each condenser call and at final termination.
        Format matches the evaluation trajectories.jsonl schema.
        Enable by setting DEEPRESEARCH_STREAM_DIR to a directory path.

        At final termination, also writes trajectories_no_memory.jsonl (single line)
        using full_messages with prev_state stripped — the original conversation
        without any memory/condenser artifacts.
        """
        if not stream_filepath:
            return
        try:
            now = datetime.now()
            request_text = self._messages_to_text(agent_data.messages, add_generation_prompt=True)
            token_count = len(self.tokenizer.encode(request_text, add_special_tokens=False))
            entry = {
                "question": question,
                "is_final": is_final,
                "condenser_update_id": agent_data.condenser_update_id if not is_final else None,
                "condenser_call_id": agent_data.condenser_call_id if not is_final else None,
                "timestamp": now.strftime("%Y%m%d_%H%M%S_%f"),
                "datetime": now.isoformat(),
                "round": agent_data.assistant_turns,
                "trigger_reason": trigger_reason,
                "prev_state": agent_data.memory_state,
                "messages": agent_data.messages,
                "message_count": len(agent_data.messages),
                "token_count": token_count,
            }
            stream_dir = os.path.dirname(stream_filepath)
            os.makedirs(stream_dir, exist_ok=True)
            with open(stream_filepath, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())

            # At final termination, write no_memory version (single line)
            if is_final:
                full_msgs = agent_data.full_messages if agent_data.full_messages else agent_data.messages
                no_memory_msgs = self._remove_prev_state_from_messages(full_msgs)
                no_memory_text = self._messages_to_text(no_memory_msgs, add_generation_prompt=True)
                no_memory_token_count = len(self.tokenizer.encode(no_memory_text, add_special_tokens=False))
                no_memory_entry = {
                    "question": question,
                    "is_final": True,
                    "condenser_update_id": None,
                    "condenser_call_id": None,
                    "timestamp": now.strftime("%Y%m%d_%H%M%S_%f"),
                    "datetime": now.isoformat(),
                    "round": agent_data.assistant_turns,
                    "trigger_reason": trigger_reason,
                    "prev_state": None,
                    "messages": no_memory_msgs,
                    "message_count": len(no_memory_msgs),
                    "token_count": no_memory_token_count,
                }
                no_memory_filepath = os.path.join(stream_dir, "trajectories_no_memory.jsonl")
                with open(no_memory_filepath, "a") as f:
                    f.write(json.dumps(no_memory_entry, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
        except Exception as e:
            logger.debug(f"[Stream] Failed to write trajectory: {e}")

    def _stream_condenser_call(
        self,
        agent_data: AgentData,
        new_state: Dict,
        original_message_count: int,
        compressed_message_count: int,
        trigger_reason: str,
        stream_filepath: str,
    ) -> None:
        """Write a condenser call JSON log file alongside the trajectory JSONL.

        Filename: condenser_call_{call_id}_{timestamp}.json
        """
        if not stream_filepath:
            return
        try:
            now = datetime.now()
            ts = now.strftime("%Y%m%d_%H%M%S_%f")
            stream_dir = os.path.dirname(stream_filepath)
            filename = f"condenser_call_{agent_data.condenser_call_id}_{ts}.json"
            filepath = os.path.join(stream_dir, filename)
            entry = {
                "condenser_call_number": agent_data.condenser_call_id,
                "timestamp": ts,
                "datetime": now.isoformat(),
                "round": agent_data.assistant_turns,
                "trigger_reason": trigger_reason,
                "state": new_state,
                "original_message_count": original_message_count,
                "compressed_message_count": compressed_message_count,
                "messages_removed": original_message_count - compressed_message_count,
            }
            with open(filepath, "w") as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            logger.debug(f"[Stream] Failed to write condenser call log: {e}")



    _toolstat_counts: Dict[tuple, int] = {}
    _toolstat_calls: Dict[str, int] = {}
    _toolstat_every = None
    _toolstat_tsum: Dict[str, float] = {}
    _toolstat_tcnt: Dict[str, int] = {}
    _toolstat_tmax: Dict[str, float] = {}
    _toolstat_tsamp: Dict[str, list] = {}

    @classmethod
    def _record_toolstat(cls, tool_name: Any, metrics: Any, elapsed: float = None) -> None:
        try:
            if cls._toolstat_every is None:
                cls._toolstat_every = max(0, int(os.getenv("DEEPRESEARCH_TOOLSTAT_EVERY", "200")))
            if not cls._toolstat_every:
                return
            name = str(tool_name)
            m = metrics if isinstance(metrics, dict) else {}
            err = m.get("error")
            if err:
                outcome = "error:" + str(err)[:60].replace(" ", "_")
            else:
                outcome = "status:" + str(m.get("status", "unknown"))
            ec = m.get("error_count")
            if isinstance(ec, int) and ec > 0:
                outcome += f"+perr{ec}"
            key = (name, outcome)
            cls._toolstat_counts[key] = cls._toolstat_counts.get(key, 0) + 1
            n = cls._toolstat_calls.get(name, 0) + 1
            cls._toolstat_calls[name] = n
            if elapsed is not None and elapsed >= 0:
                cls._toolstat_tsum[name] = cls._toolstat_tsum.get(name, 0.0) + float(elapsed)
                cls._toolstat_tcnt[name] = cls._toolstat_tcnt.get(name, 0) + 1
                if float(elapsed) > cls._toolstat_tmax.get(name, 0.0):
                    cls._toolstat_tmax[name] = float(elapsed)
                samp = cls._toolstat_tsamp.setdefault(name, [])
                samp.append(float(elapsed))
                if len(samp) > 1000:
                    del samp[: len(samp) - 1000]
            if n % cls._toolstat_every == 0:
                parts = " ".join(
                    f"{o}x{c}" for (t, o), c in sorted(cls._toolstat_counts.items()) if t == name
                )
                tc = cls._toolstat_tcnt.get(name, 0)
                if tc:
                    avg = cls._toolstat_tsum[name] / tc
                    sp = sorted(cls._toolstat_tsamp.get(name, []))
                    p50 = sp[len(sp) // 2] if sp else float("nan")
                    p90 = sp[int(len(sp) * 0.9)] if sp else float("nan")
                    tstr = (f" | t_avg={avg:.2f}s p50={p50:.2f}s p90={p90:.2f}s "
                            f"max={cls._toolstat_tmax.get(name, 0.0):.2f}s n_t={tc}")
                else:
                    tstr = ""
                print(f"[TOOLSTAT] pid={os.getpid()} tool={name} calls={n} {parts}{tstr}", flush=True)
        except Exception:
            pass

    @classmethod
    def _print_budget_ok(cls) -> bool:
        """True while the print budget lasts; on the call that exhausts it, log one
        line and stay off for good."""
        budget = getattr(cls, "_print_turns_budget", 0)
        if budget <= 0:
            return True
        used = getattr(cls, "_print_turns_used", 0)
        if used < budget:
            cls._print_turns_used = used + 1
            return True
        if used == budget:
            cls._print_turns_used = used + 1
            print(f"[deepresearch] trajectory print budget of {budget} exhausted, no further "
                  f"(DEEPRESEARCH_PRINT_TURNS_BUDGET)", flush=True)
        return False

    @classmethod
    def _compute_turn_boundary_tokens(cls):
        """Pre-compute turn boundary token sequences using apply_chat_template.

        This computes the exact token sequences needed for:
        - assistant->user transition (e.g., <|im_end|>\\n<|im_start|>user\\n)
        - user->assistant transition (e.g., <|im_end|>\\n<|im_start|>assistant\\n)
        """
        ref_msgs = [
            {"role": "assistant", "content": "XBOUNDARYMARKERX"},
            {"role": "user", "content": "YBOUNDARYMARKERY"},
        ]
        ref_ids = safe_apply_chat_template(
            cls.tokenizer,
            ref_msgs,
            add_generation_prompt=True,
            tokenize=True,
            **cls.apply_chat_template_kwargs,
        )
        x_ids = cls.tokenizer.encode("XBOUNDARYMARKERX", add_special_tokens=False)
        y_ids = cls.tokenizer.encode("YBOUNDARYMARKERY", add_special_tokens=False)
        x_pos = _find_subseq(ref_ids, x_ids)
        x_end = x_pos + len(x_ids)
        y_pos = _find_subseq(ref_ids, y_ids, x_end)
        y_end = y_pos + len(y_ids)
        # assistant->user boundary: e.g. <|im_end|>\n<|im_start|>user\n
        cls._boundary_a2u = ref_ids[x_end:y_pos]
        # user->assistant boundary: e.g. <|im_end|>\n<|im_start|>assistant\n
        cls._boundary_u2a = ref_ids[y_end:]
        cls._im_end_id = cls._boundary_a2u[0] if cls._boundary_a2u else None
        logger.info(
            "Turn boundaries: a2u=%r, u2a=%r",
            cls.tokenizer.decode(cls._boundary_a2u, skip_special_tokens=False),
            cls.tokenizer.decode(cls._boundary_u2a, skip_special_tokens=False),
        )

    async def _compute_env_turn_tokens(self, agent_data, content):
        """Compute tokens for an environment turn (tool response / error) with proper role boundaries.

        This wraps the content with assistant->user and user->assistant transition tokens,
        ensuring the model sees properly formatted multi-turn conversation.
        """
        if (self._im_end_id is not None
                and agent_data.prompt_ids
                and agent_data.prompt_ids[-1] == self._im_end_id):
            # Model already generated <|im_end|>, skip it in the boundary
            pre = list(self._boundary_a2u[1:])
        else:
            pre = list(self._boundary_a2u)
        content_ids = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.encode(content, add_special_tokens=False))
        post = list(self._boundary_u2a)
        return pre + content_ids + post

    def _append_env_tokens(self, agent_data, env_tokens):
        """Append environment tokens to agent data with mask=0 (not model-generated)."""
        agent_data.response_ids += env_tokens
        agent_data.response_mask += [0] * len(env_tokens)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(env_tokens)
        agent_data.prompt_ids += env_tokens

    @classmethod
    def _remaining_response_budget(cls, agent_data, extra_tokens: int = 0) -> int:
        """Return remaining response-segment budget for the current session."""
        return cls.response_length - len(agent_data.response_mask) - extra_tokens

    @classmethod
    def _remaining_turn_budget(cls) -> int:
        """Return max tokens allowed for a single assistant generation turn."""
        return cls.turn_response_length

    @classmethod
    def _build_budgeted_sampling_params(cls, agent_data, sampling_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Clamp one generation turn to the remaining response and model-context budget."""
        remaining_response_budget = cls._remaining_response_budget(agent_data)
        remaining_context_budget = cls.model_max_len - len(agent_data.prompt_ids)
        generation_budget = min(
            remaining_response_budget,
            remaining_context_budget,
            cls._remaining_turn_budget(),
        )
        if generation_budget <= 0:
            return None

        budgeted_sampling_params = dict(sampling_params)
        if "max_new_tokens" in budgeted_sampling_params:
            budgeted_sampling_params["max_new_tokens"] = min(
                int(budgeted_sampling_params["max_new_tokens"]), generation_budget
            )
        elif "max_tokens" in budgeted_sampling_params:
            budgeted_sampling_params["max_tokens"] = min(int(budgeted_sampling_params["max_tokens"]), generation_budget)
        else:
            budgeted_sampling_params["max_tokens"] = generation_budget
        return budgeted_sampling_params

    @classmethod
    def _budget_exhausted_state(cls, agent_data) -> AgentState:
        """Prefer condenser rollover when the current session runs out of response budget."""
        if cls.memory_enabled and (
            len(agent_data.response_mask) > 0 or agent_data.assistant_turns > 0 or agent_data.user_turns > 0
        ):
            return AgentState.MEMORY_TRIGGER
        return AgentState.TERMINATED

    def _get_system_prompt(self) -> str:
        """Get system prompt with current date."""
        prompt = SYSTEM_PROMPT + date.today().strftime("%Y-%m-%d")
        if self.soften_answer_prompt:
            prompt += (
                "\n\nIMPORTANT: Do not refuse to answer or leave the answer blank because you are "
                "uncertain. When you have researched enough, commit to your single best-supported "
                "answer based on the evidence gathered, and always place it inside <answer></answer> "
                "tags. Never end your response without a final <answer>."
            )
        return prompt

    def _messages_to_text(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
        """Build chat-formatted text from messages."""
        try:
            return safe_apply_chat_template(
                self.tokenizer,
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **self.apply_chat_template_kwargs,
            )
        except Exception:
            # Fallback: concatenate with minimal role markers
            parts = []
            for m in messages:
                role, content = m.get("role", ""), m.get("content", "")
                parts.append(f"<|{role}|>\n{content}")
            if add_generation_prompt:
                parts.append("<|assistant|>\n")
            return "\n".join(parts)

    def _parse_tool_call(self, content: str) -> tuple[Optional[Any], Optional[Any]]:
        """Parse tool call from model output (last match to skip fake calls in <think>)."""
        pattern = r"<tool_call>(.*?)</tool_call>"
        matches = list(re.finditer(pattern, content, re.DOTALL))

        if not matches:
            return None, None

        # Use the last match to be consistent with _postprocess_response (rfind)
        try:
            tool_call_str = matches[-1].group(1).strip()
            code_match = re.search(r"<code>(.*?)</code>", tool_call_str, re.DOTALL)
            extracted_code = None
            json_part = tool_call_str
            if code_match:
                extracted_code = code_match.group(1).strip()
                json_part = tool_call_str[:code_match.start()].strip()
            else:
                open_match = re.search(r"<code>", tool_call_str)
                if open_match:
                    extracted_code = tool_call_str[open_match.end():].strip()
                    json_part = tool_call_str[:open_match.start()].strip()
            tool_call = json.loads(json_part)
            if not isinstance(tool_call, dict):
                # Let _call_tool produce a structured tool error via <tool_response>.
                return tool_call, {}
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("arguments", {})
            if extracted_code and isinstance(tool_name, str) and tool_name.strip() == "PythonInterpreter":
                if tool_args is None:
                    tool_args = {}
                if isinstance(tool_args, dict):
                    tool_args["code"] = extracted_code
            return tool_name, tool_args
        except (json.JSONDecodeError, TypeError, ValueError):
            # Preserve tool-call feedback path: malformed JSON still becomes a tool error response.
            return ["__invalid_tool_call_json__", tool_call_str], {}

    @staticmethod
    def _visible(content: str) -> str:
        """Content after the model's reasoning block -- everything up to and including the
        last `</think>` is the model thinking out loud, not its reply. Models draft the
        required output format inside it (literal `<answer></answer>`), and scanning the raw
        text treats that draft as a submission and ends the episode early. Same rule as
        `inference/react_agent.py:visible_content`. Note rollout content carries the closing
        tag only (the vLLM reasoning parser strips the opener), so splitting on `</think>`
        is the reliable form -- do not switch this to a `<think>...</think>` regex."""
        if not content:
            return ""
        if "</think>" in content:
            return content.rsplit("</think>", 1)[-1]
        return content

    def _check_for_answer(self, content: str) -> bool:
        """Whether the response submits a final answer: a non-empty `<answer>` outside the
        reasoning block. An empty pair is the model quoting the format instruction."""
        return self._extract_answer(content) is not None

    def _extract_answer(self, content: str) -> Optional[str]:
        """Last non-empty `<answer>` payload outside the reasoning block; None if absent."""
        pattern = r"<answer>(.*?)</answer>"
        for m in reversed(list(re.finditer(pattern, self._visible(content), re.DOTALL))):
            if m.group(1).strip():
                return m.group(1).strip()
        return None

    @staticmethod
    def _normalize_tool_name(tool_name: Any) -> Optional[str]:
        if isinstance(tool_name, list):
            if len(tool_name) == 1 and isinstance(tool_name[0], str):
                tool_name = tool_name[0]
            else:
                return None
        if not isinstance(tool_name, str):
            return None
        tool_name = tool_name.strip()
        return tool_name or None

    def _postprocess_response(self, resp: str) -> str:
        """Post-process response to ensure tag completeness (longest match)."""
        if "</tool_call>" in resp:
            last_idx = resp.rfind("</tool_call>")
            return resp[:last_idx + len("</tool_call>")]
        if "</answer>" in resp:
            last_idx = resp.rfind("</answer>")
            return resp[:last_idx + len("</answer>")]
        return resp

    def _reached_max_turns(self, agent_data: AgentData) -> bool:
        """Check whether configured total turn limit has been reached."""
        if not self.max_turns:
            return False
        return (agent_data.user_turns + agent_data.assistant_turns) >= self.max_turns

    def _terminate_or_force(self, agent_data: AgentData) -> AgentState:
        """Route a no-answer termination to one final prompted answer turn when
        FORCE_FINAL_ANSWER is on and no <answer> has been emitted yet; else end."""
        if (
            self.force_final_answer_enabled
            and not agent_data.forced_answer_done
            and not self._check_for_answer(agent_data.full_response)
        ):
            return AgentState.FORCING_ANSWER
        return AgentState.TERMINATED

    @rollout_trace_op
    async def run(self, sampling_params: Dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run the DeepResearch agent loop."""
        messages = list(kwargs["raw_prompt"])
        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Get original user content
        original_user_content = ""
        for msg in messages:
            if msg.get("role") == "user":
                original_user_content = msg.get("content", "")
                break

        # Create AgentData instance
        agent_data = AgentData(
            messages=messages,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            original_user_content=original_user_content,
        )

        # State machine loop
        state = AgentState.PENDING
        consecutive_invalid_count = 0

        # Real-time trajectory streaming path
        stream_filepath = ""
        if self._stream_dir:
            step_tag = self._extract_stream_step(kwargs)
            task_tag = self._extract_stream_task_id(kwargs)
            rollout_tag = self._extract_stream_rollout(kwargs)
            stream_filepath = os.path.join(
                self._stream_dir,
                f"step_{step_tag}",
                f"task_{task_tag}",
                f"rollout_{rollout_tag}",
                "trajectories.jsonl",
            )

        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state, consecutive_invalid_count = await self._handle_generating_state(
                    agent_data, sampling_params, consecutive_invalid_count
                )
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.MEMORY_TRIGGER:
                state = await self._handle_memory_trigger_state(agent_data, original_user_content, stream_filepath)
            elif state == AgentState.FORCING_ANSWER:
                state = await self._handle_forcing_answer_state(agent_data, sampling_params)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Stream final trajectory at termination
        self._stream_trajectory(agent_data, original_user_content, is_final=True, trigger_reason="final", stream_filepath=stream_filepath)

        # Save the final session (current one)
        final_session_prompt_ids = list(agent_data.prompt_ids[:len(agent_data.prompt_ids) - len(agent_data.response_ids)])
        final_session = SessionData(
            session_id=agent_data.session_id,
            prompt_ids=final_session_prompt_ids,
            original_user_content=agent_data.original_user_content,
            memory_state=agent_data.memory_state,
        )
        final_session.response_ids = list(agent_data.response_ids)
        final_session.response_mask = list(agent_data.response_mask)
        final_session.response_logprobs = list(agent_data.response_logprobs) if agent_data.response_logprobs else []
        final_session.routed_experts = agent_data.routed_experts
        final_session.response_text = agent_data.session_response
        final_session.messages = copy.deepcopy(agent_data.messages)

        # Combine all sessions
        all_sessions = agent_data.completed_sessions + [final_session]
        num_sessions = len(all_sessions)

        # Build session data list for expansion in manager
        session_outputs = []
        for session in all_sessions:
            session_outputs.append({
                "session_id": session.session_id,
                "prompt_ids": session.prompt_ids,
                "response_ids": session.response_ids[:self.response_length],
                "response_mask": session.response_mask[:self.response_length],
                "response_logprobs": session.response_logprobs[:self.response_length] if session.response_logprobs else None,
                "routed_experts": _truncate_routed_experts(session.routed_experts, session.prompt_ids, session.response_ids, self.response_length),
                "memory_state": session.memory_state,
                "messages": session.messages,
            })

        # Return single output (use the last session for the primary output)
        # All session data is embedded in extra_fields for expansion by manager
        output = AgentLoopOutput(
            prompt_ids=final_session.prompt_ids,
            response_ids=final_session.response_ids[:self.response_length],
            response_mask=final_session.response_mask[:self.response_length],
            multi_modal_data={},
            response_logprobs=final_session.response_logprobs[:self.response_length] if final_session.response_logprobs else None,
            routed_experts=_truncate_routed_experts(final_session.routed_experts, final_session.prompt_ids, final_session.response_ids, self.response_length),
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields={},
        )
        output.extra_fields.update({
            "session_id": final_session.session_id,
            "num_sessions": num_sessions,
            "session_outputs": session_outputs,  # All sessions for expansion
            "full_response": agent_data.full_response,  # For reward computation
            "memory_state": final_session.memory_state,
            "turn_scores": agent_data.turn_scores,
            "tool_rewards": agent_data.tool_rewards,
            "terminated_by_consecutive_invalid": agent_data.terminated_by_consecutive_invalid,
            "answered": bool(self._check_for_answer(agent_data.full_response)),
            "reached_max_turns": bool(
                (self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns)
                or (self.max_user_turns and agent_data.user_turns >= self.max_user_turns)
                or self._reached_max_turns(agent_data)
            ),
            "agent_messages": copy.deepcopy(agent_data.messages),
            "full_messages": copy.deepcopy(agent_data.full_messages),
        })

        return output

    async def _handle_pending_state(
        self, agent_data: AgentData, sampling_params: Dict[str, Any]
    ) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        # Build messages with system prompt
        system_prompt = self._get_system_prompt()

        # Add prev_state to user content if memory state exists
        user_content = agent_data.original_user_content
        if agent_data.memory_state:
            state_section = format_prev_state_section(agent_data.memory_state)
            user_content = agent_data.original_user_content + state_section

        if agent_data.pending_messages:
            session_messages = agent_data.pending_messages
            agent_data.pending_messages = None
        else:
            session_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

        # Update agent_data messages
        agent_data.messages = session_messages

        # Track full messages (only for the first session, i.e., before any condenser)
        if agent_data.session_id == 0:
            agent_data.full_messages = copy.deepcopy(session_messages)

        # Tokenize prompt
        agent_data.prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: safe_apply_chat_template(
                self.tokenizer,
                session_messages,
                add_generation_prompt=True,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )

        return AgentState.GENERATING

    async def _handle_generating_state(
        self,
        agent_data: AgentData,
        sampling_params: Dict[str, Any],
        consecutive_invalid_count: int,
    ) -> tuple[AgentState, int]:
        """Handle the generating state: generate model response and check for tool calls."""
        if self._reached_max_turns(agent_data):
            logger.info(
                "Terminating due to max_turns=%s (user_turns=%s, assistant_turns=%s)",
                self.max_turns,
                agent_data.user_turns,
                agent_data.assistant_turns,
            )
            return self._terminate_or_force(agent_data), consecutive_invalid_count

        budgeted_sampling_params = self._build_budgeted_sampling_params(agent_data, sampling_params)
        if budgeted_sampling_params is None:
            logger.warning(
                "Generation budget exhausted: prompt_ids=%d, response_tokens=%d, response_limit=%d, turn_limit=%d, model_max_len=%d",
                len(agent_data.prompt_ids),
                len(agent_data.response_mask),
                self.response_length,
                self.turn_response_length,
                self.model_max_len,
            )
            return self._budget_exhausted_state(agent_data), consecutive_invalid_count

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=budgeted_sampling_params,
                image_data=None,
            )

        agent_data.assistant_turns += 1
        agent_data.response_ids += output.token_ids
        agent_data.prompt_ids += output.token_ids
        agent_data.response_mask += [1] * len(output.token_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs
        if getattr(output, "routed_experts", None) is not None:
            agent_data.routed_experts = output.routed_experts  # last request wins

        # Decode only newly generated tokens for this round
        cur_response = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
        )

        # Post-process response
        cur_response = self._postprocess_response(cur_response)

        if self._print_turns and _is_rank_zero() and self._print_budget_ok():
            _text = cur_response if len(cur_response) <= self._print_turns_max_chars else cur_response[:self._print_turns_max_chars] + "...<truncated>"
            print(f"[TURN] req={agent_data.request_id[:8]} assistant_turn={agent_data.assistant_turns} tokens={len(output.token_ids)}\n{_text}", flush=True)

        # Update response tracking
        agent_data.full_response += cur_response
        agent_data.session_response += cur_response

        # Add assistant message
        agent_data.messages.append({"role": "assistant", "content": cur_response})
        agent_data.full_messages.append({"role": "assistant", "content": cur_response})

        # Check turn-based termination conditions
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return self._terminate_or_force(agent_data), consecutive_invalid_count
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return self._terminate_or_force(agent_data), consecutive_invalid_count
        if self._reached_max_turns(agent_data):
            return self._terminate_or_force(agent_data), consecutive_invalid_count

        # Check for final answer BEFORE response length check
        # so a valid answer is never discarded due to buffer size
        if self._check_for_answer(cur_response):
            return AgentState.TERMINATED, consecutive_invalid_count

        # Parse tool call
        tool_name, tool_args = self._parse_tool_call(cur_response)

        if tool_name is not None:
            consecutive_invalid_count = 0
            agent_data.tool_calls = [{"name": tool_name, "arguments": tool_args}]
            return AgentState.PROCESSING_TOOLS, consecutive_invalid_count

        if self._remaining_response_budget(agent_data) <= 0:
            logger.info(
                "Response budget exhausted after assistant turn: response_tokens=%d, response_limit=%d",
                len(agent_data.response_mask),
                self.response_length,
            )
            return self._budget_exhausted_state(agent_data), consecutive_invalid_count

        # Invalid response (no tool call, no answer)
        consecutive_invalid_count += 1
        if consecutive_invalid_count >= 3:
            logger.warning(f"Terminating: 3 consecutive invalid responses")
            agent_data.terminated_by_consecutive_invalid = True
            return self._terminate_or_force(agent_data), consecutive_invalid_count

        # Add error message with required formats (from system prompt)
        error_msg = (
            "\n[System] Invalid response. Please either:\n"
            "1) Call a tool with <tool_call>{\"name\": <function-name>, \"arguments\": <args-json-object>}</tool_call>\n"
            "2) Or provide the final answer inside <answer>...</answer> tags.\n"
        )

        env_tokens = await self._compute_env_turn_tokens(agent_data, error_msg)
        if self._remaining_response_budget(agent_data, extra_tokens=len(env_tokens)) < 0:
            logger.info(
                "Skipping retry prompt because it would overflow response budget: response_tokens=%d, env_tokens=%d, response_limit=%d",
                len(agent_data.response_mask),
                len(env_tokens),
                self.response_length,
            )
            return self._budget_exhausted_state(agent_data), consecutive_invalid_count

        agent_data.messages.append({"role": "user", "content": error_msg})
        agent_data.full_messages.append({"role": "user", "content": error_msg})
        self._append_env_tokens(agent_data, env_tokens)
        agent_data.full_response += error_msg
        agent_data.session_response += error_msg
        agent_data.user_turns += 1
        agent_data.request_id = uuid4().hex

        if self._reached_max_turns(agent_data):
            return self._terminate_or_force(agent_data), consecutive_invalid_count

        return AgentState.GENERATING, consecutive_invalid_count

    async def _handle_forcing_answer_state(
        self, agent_data: AgentData, sampling_params: Dict[str, Any]
    ) -> AgentState:
        """One last prompted turn to elicit a final <answer> before terminating an
        otherwise answerless trajectory (FORCE_FINAL_ANSWER). Mirrors the token
        bookkeeping of the normal generating path; always terminates afterward."""
        agent_data.forced_answer_done = True
        force_msg = (
            "\n[System] You must stop researching now and give your final answer based on what "
            "you have gathered. Output ONLY your single best final answer wrapped in "
            "<answer>...</answer> tags. Do not call any tools. Even if you are not fully certain, "
            "commit to your best-supported answer; do not leave it blank."
        )

        env_tokens = await self._compute_env_turn_tokens(agent_data, force_msg)
        if self._remaining_response_budget(agent_data, extra_tokens=len(env_tokens)) <= 0:
            return AgentState.TERMINATED
        agent_data.messages.append({"role": "user", "content": force_msg})
        agent_data.full_messages.append({"role": "user", "content": force_msg})
        self._append_env_tokens(agent_data, env_tokens)
        agent_data.full_response += force_msg
        agent_data.session_response += force_msg
        agent_data.user_turns += 1
        agent_data.request_id = uuid4().hex

        budgeted_sampling_params = self._build_budgeted_sampling_params(agent_data, sampling_params)
        if budgeted_sampling_params is None:
            return AgentState.TERMINATED

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=budgeted_sampling_params,
                image_data=None,
            )

        agent_data.assistant_turns += 1
        agent_data.response_ids += output.token_ids
        agent_data.prompt_ids += output.token_ids
        agent_data.response_mask += [1] * len(output.token_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs
        if getattr(output, "routed_experts", None) is not None:
            agent_data.routed_experts = output.routed_experts  # last request wins

        cur_response = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
        )
        cur_response = self._postprocess_response(cur_response)
        agent_data.full_response += cur_response
        agent_data.session_response += cur_response
        agent_data.messages.append({"role": "assistant", "content": cur_response})
        agent_data.full_messages.append({"role": "assistant", "content": cur_response})
        return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        tasks = []
        for tool_call in agent_data.tool_calls[:self.max_parallel_calls]:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("arguments", {})
            tasks.append(self._call_tool(tool_name, tool_args, agent_data.tools_kwargs))

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Combine tool responses
        tool_response_texts = []
        for tc, (response, reward, metrics) in zip(agent_data.tool_calls[:self.max_parallel_calls], responses):
            tool_response_texts.append(response.text or "")
            agent_data.tool_rewards.append(reward)
            # Only log retrieval-tool query, channel, and similarity; upstream uses content only.
            normalized_tool_name = self._normalize_tool_name(tc.get("name"))
            if normalized_tool_name in {"search", "google_scholar"} and metrics and metrics.get("sources"):
                args = tc.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                elif isinstance(args, list):
                    args = args[0] if len(args) == 1 and isinstance(args[0], dict) else {}
                elif args is None:
                    args = {}
                if not isinstance(args, dict):
                    args = {}

                q = args.get("query")
                if isinstance(q, str):
                    queries = [q]
                elif isinstance(q, (list, tuple)):
                    queries = [item for item in q if isinstance(item, str)]
                else:
                    queries = []
                tool_name = normalized_tool_name or "tool"
                for qry, s in zip(queries, metrics["sources"]):
                    src = s.get("source", "?")
                    sim = s.get("similarity")
                    sim_str = f", similarity={sim}" if sim is not None else ""
                    if self._print_search and _is_rank_zero():
                        print(f"[{tool_name}] req={agent_data.request_id[:8]} query={qry!r} source={src}{sim_str}", flush=True)

        combined_response = "\n".join(tool_response_texts)
        tool_response = f"\n<tool_response>\n{combined_response}\n</tool_response>\n"

        if self._print_turns and _is_rank_zero() and self._print_budget_ok():
            for tc, (resp, _, _) in zip(agent_data.tool_calls[:self.max_parallel_calls], responses):
                _resp_text = resp.text or ""
                if len(_resp_text) > self._print_turns_max_chars:
                    _resp_text = _resp_text[:self._print_turns_max_chars] + "...<truncated>"
                display_tool_name = self._normalize_tool_name(tc.get("name")) or tc.get("name")
                print(
                    f"[TURN] req={agent_data.request_id[:8]} tool={display_tool_name} args={tc.get('arguments')}\n  response={_resp_text}",
                    flush=True,
                )

        # Tokenize tool response with proper role boundary tokens
        env_tokens = await self._compute_env_turn_tokens(agent_data, tool_response)

        # Add tool response to messages
        agent_data.messages.append({"role": "user", "content": tool_response})
        agent_data.full_messages.append({"role": "user", "content": tool_response})
        agent_data.user_turns += 1
        agent_data.full_response += tool_response
        agent_data.session_response += tool_response

        if self._remaining_response_budget(agent_data, extra_tokens=len(env_tokens)) < 0:
            logger.info(
                "Tool response overflowed response budget: response_tokens=%d, env_tokens=%d, response_limit=%d",
                len(agent_data.response_mask),
                len(env_tokens),
                self.response_length,
            )
            return self._budget_exhausted_state(agent_data)

        # Append env tokens first so memory check sees full context
        self._append_env_tokens(agent_data, env_tokens)

        if self._reached_max_turns(agent_data):
            return AgentState.TERMINATED

        # Check memory threshold: if context is too long, trigger condenser
        if self.memory_enabled:
            request_text = self._messages_to_text(agent_data.messages, add_generation_prompt=True)
            current_tokens = len(self.tokenizer.encode(request_text, add_special_tokens=False))
            logger.debug(f"[Memory] Turn {agent_data.assistant_turns}: tokens = {current_tokens}, threshold = {self.context_threshold}")

            if current_tokens >= self.context_threshold:
                return AgentState.MEMORY_TRIGGER

        # Change request_id to avoid caching
        agent_data.request_id = uuid4().hex
        return AgentState.GENERATING


    _TOOL_RESP_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)

    @classmethod
    def _stub_tool_responses(cls, text: str) -> tuple[str, int]:
        """Replace a <tool_response> body with its first N characters plus an
        elision note. Returns (new text, characters saved)."""
        saved = 0

        def _repl(m):
            nonlocal saved
            body = m.group(1)
            if len(body) <= cls.memory_tier1_stub_chars:
                return m.group(0)
            saved += len(body) - cls.memory_tier1_stub_chars
            head = body[: cls.memory_tier1_stub_chars].rstrip()
            return (f"<tool_response>{head}\n"
                    f"[... {len(body) - cls.memory_tier1_stub_chars} chars of this tool output were dropped to free "
                    f"context; re-run the tool if you need the rest ...]</tool_response>")

        return cls._TOOL_RESP_RE.sub(_repl, text), saved

    @classmethod
    def _strip_think(cls, text: str) -> tuple[str, int]:
        """Strip the thinking segment from earlier assistant turns, keeping only
        what follows the last </think>."""
        if "</think>" not in text:
            return text, 0
        tail = text.rsplit("</think>", 1)[-1]
        return tail.lstrip(), len(text) - len(tail)

    def _tier1_compact(self, messages: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Tier 1: keep the system message, the first user message (the task) and
        the last keep_msgs messages verbatim. In between, only tool-response
        bodies and assistant thinking are trimmed; no message is dropped.
        Returns (compacted messages, stats)."""
        keep = max(0, self.memory_tier1_keep_msgs)
        head_n = 2 if len(messages) >= 2 and messages[0].get("role") == "system" else 1
        if len(messages) <= head_n + keep:
            return list(messages), {"stubbed": 0, "think_stripped": 0, "chars_saved": 0}
        out = [dict(m) for m in messages[:head_n]]
        mid = messages[head_n: len(messages) - keep] if keep else messages[head_n:]
        stubbed = think_stripped = chars_saved = 0
        for m in mid:
            m2 = dict(m)
            content = m2.get("content")
            if not isinstance(content, str):
                out.append(m2)
                continue
            if m2.get("role") == "assistant":
                content, cut = self._strip_think(content)
                if cut:
                    think_stripped += 1
                    chars_saved += cut
            else:
                content, cut = self._stub_tool_responses(content)
                if cut:
                    stubbed += 1
                    chars_saved += cut
            m2["content"] = content
            out.append(m2)
        out += [dict(m) for m in (messages[len(messages) - keep:] if keep else [])]
        return out, {"stubbed": stubbed, "think_stripped": think_stripped, "chars_saved": chars_saved}

    def _tier1_tokens(self, messages: List[Dict[str, Any]]) -> int:
        return len(self.tokenizer.encode(self._messages_to_text(messages, add_generation_prompt=True),
                                         add_special_tokens=False))



    def _keep_recent_messages(self, messages: List[Dict[str, Any]], budget_tokens: int) -> List[Dict[str, Any]]:
        """Walk backwards from the end, taking recent messages until budget_tokens
        is full; anything that does not fit is stubbed as in tier 1 and retried.
        Tier 2 summarises only the older middle, so the recent tail is always
        carried verbatim -- this function produces that tail."""
        head_n = 2 if len(messages) >= 2 and messages[0].get("role") == "system" else 1
        tail = messages[head_n:]
        if not tail or budget_tokens <= 0:
            return []
        keep: List[Dict[str, Any]] = []
        used = 0
        for m in reversed(tail):
            cand = dict(m)
            c = cand.get("content")
            if isinstance(c, str):
                n = len(self.tokenizer.encode(c, add_special_tokens=False))
                if used + n > budget_tokens:
                    stub, saved = (self._strip_think(c) if cand.get("role") == "assistant"
                                   else self._stub_tool_responses(c))
                    n2 = len(self.tokenizer.encode(stub, add_special_tokens=False)) if saved else n
                    if not saved or used + n2 > budget_tokens:
                        break
                    cand["content"] = stub; n = n2
                used += n
            keep.append(cand)
        keep.reverse()
        while keep and keep[0].get("role") != "assistant":
            keep.pop(0)
        return keep

    def _rotate_session(self, agent_data: "AgentData", question: str, stream_filepath: str,
                        trigger_reason: str, new_state: Optional[Dict],
                        pending_messages: Optional[List[Dict[str, Any]]], stream: bool = True) -> None:
        """Archive the current session, advance the counters and clear the response
        buffer; the next session starts from pending_messages. Same as the
        condenser branch, except a non-empty pending_messages carries the
        compacted context into the new session."""
        if stream:
            self._stream_trajectory(agent_data, question, is_final=False,
                                    trigger_reason=trigger_reason, stream_filepath=stream_filepath)
        session_prompt_ids = list(agent_data.prompt_ids[:len(agent_data.prompt_ids) - len(agent_data.response_ids)])
        current_session = SessionData(
            session_id=agent_data.session_id,
            prompt_ids=session_prompt_ids,
            original_user_content=agent_data.original_user_content,
            memory_state=agent_data.memory_state,
        )
        current_session.response_ids = list(agent_data.response_ids)
        current_session.response_mask = list(agent_data.response_mask)
        current_session.response_logprobs = list(agent_data.response_logprobs) if agent_data.response_logprobs else []
        current_session.routed_experts = agent_data.routed_experts
        current_session.response_text = agent_data.session_response
        current_session.messages = copy.deepcopy(agent_data.messages)
        agent_data.completed_sessions.append(current_session)

        agent_data.memory_state = new_state
        agent_data.session_id += 1
        agent_data.pending_messages = pending_messages
        agent_data.response_ids = []
        agent_data.response_mask = []
        agent_data.response_logprobs = []
        agent_data.routed_experts = None
        agent_data.session_response = ""

    async def _handle_memory_trigger_state(self, agent_data: AgentData, question: str = "", stream_filepath: str = "") -> AgentState:
        """Handle a memory trigger: try tier 1 (deterministic trimming) first and
        only call the condenser (tier 2) if that does not free enough space."""
        if self.memory_tier1_enabled:
            compacted, st = self._tier1_compact(agent_data.messages)
            target = self.memory_tier1_target or max(1, (self.context_threshold * 3) // 5)
            after = self._tier1_tokens(compacted) if st["chars_saved"] else None
            if after is not None and after <= target:
                logger.info(
                    "[Memory] tier1 freed %d chars (%d tool outputs stubbed, %d think blocks stripped); "
                    "context %d -> %d tokens <= target %d, rotating session without condenser",
                    st["chars_saved"], st["stubbed"], st["think_stripped"], self.context_threshold, after, target)
                agent_data.metrics["memory_tier1_rotations"] = agent_data.metrics.get("memory_tier1_rotations", 0) + 1
                self._rotate_session(agent_data, question, stream_filepath,
                                     trigger_reason="CONTEXT_THRESHOLD_TIER1",
                                     new_state=agent_data.memory_state,
                                     pending_messages=compacted)
                return AgentState.PENDING
            logger.info(
                "[Memory] tier1 relief insufficient (freed %d chars, context still %s vs target %d); escalating to condenser",
                st["chars_saved"], after, target)
            agent_data.metrics["memory_tier2_rotations"] = agent_data.metrics.get("memory_tier2_rotations", 0) + 1

        logger.info(f"[Memory] CONTEXT_THRESHOLD triggered, calling condenser...")

        # Extract events from messages
        events = extract_events_from_messages(agent_data.messages)

        # Call condenser
        result = await call_condenser_async(events, agent_data.memory_state)
        new_state = parse_memory_result(result)
        used_manual_fallback = False
        agent_data.condenser_call_id += 1

        if not new_state:
            new_state = build_fallback_memory_state(events, agent_data.memory_state)
            used_manual_fallback = True
            logger.warning("[Memory] Condenser failed; using heuristic fallback state and rotating session.")

        state_changed = bool(new_state) and (used_manual_fallback or new_state != agent_data.memory_state)

        if new_state:
            if state_changed:
                agent_data.condenser_update_id += 1
            else:
                logger.info("[Memory] Condenser returned unchanged state; rotating session to enforce context reset.")
            original_message_count = len(agent_data.messages)

            # Stream trajectory BEFORE resetting session state (so messages reflect the pre-condenser conversation)
            if used_manual_fallback:
                trigger_reason = "CONTEXT_THRESHOLD_FALLBACK"
            elif state_changed:
                trigger_reason = "CONTEXT_THRESHOLD"
            else:
                trigger_reason = "CONTEXT_THRESHOLD_UNCHANGED"
            self._stream_trajectory(agent_data, question, is_final=False, trigger_reason=trigger_reason, stream_filepath=stream_filepath)

            recent = []
            if self.memory_tier1_enabled:
                state_tokens = len(self.tokenizer.encode(format_prev_state_section(new_state), add_special_tokens=False))
                budget = max(0, (self.memory_tier1_target or max(1, (self.context_threshold * 3) // 5)) - state_tokens)
                recent = self._keep_recent_messages(agent_data.messages, budget)
            pending = None
            if recent:
                user_content = agent_data.original_user_content + format_prev_state_section(new_state)
                pending = ([{"role": "system", "content": self._get_system_prompt()},
                            {"role": "user", "content": user_content}] + recent)
                logger.info("[Memory] tier2: condenser summary + %d recent messages carried into next session", len(recent))

            # Stream condenser call log
            self._stream_condenser_call(
                agent_data, new_state, original_message_count,
                compressed_message_count=len(pending) if pending else 2, trigger_reason=trigger_reason,
                stream_filepath=stream_filepath,
            )
            n_resp = len(agent_data.response_ids)
            self._rotate_session(agent_data, question, stream_filepath,
                                 trigger_reason=trigger_reason, new_state=new_state,
                                 pending_messages=pending, stream=False)
            logger.info(f"[Memory] Saved session {agent_data.session_id - 1} with {n_resp} response tokens")
            logger.info(f"[Memory] Starting new session {agent_data.session_id}")
            return AgentState.PENDING
        else:
            logger.warning("[Memory] Condenser produced no usable state after fallback; continuing current session.")
            agent_data.request_id = uuid4().hex
            return AgentState.GENERATING

    async def _call_tool(
        self, tool_name: Any, tool_args: Any, tools_kwargs: Dict[str, Any]
    ) -> tuple[ToolResponse, float, Dict]:
        """Call a tool and return the response."""
        raw_tool_name = tool_name
        raw_tool_args = tool_args

        if (isinstance(raw_tool_name, list) and len(raw_tool_name) == 2
                and raw_tool_name[0] == "__invalid_tool_call_json__"):
            return (
                ToolResponse(text="[Tool Error] " + _diagnose_tool_call_json(raw_tool_name[1])),
                0.0,
                {"error": "invalid_tool_call_json"},
            )
        if isinstance(tool_name, list):
            if len(tool_name) == 1 and isinstance(tool_name[0], str):
                tool_name = tool_name[0]
            else:
                return (
                    ToolResponse(text=f"[Tool Error] Invalid tool call format: name must be string, got {raw_tool_name!r}"),
                    0.0,
                    {"error": "invalid_tool_name"},
                )
        if not isinstance(tool_name, str) or not tool_name.strip():
            return (
                ToolResponse(text=f"[Tool Error] Invalid tool call format: name must be string, got {raw_tool_name!r}"),
                0.0,
                {"error": "invalid_tool_name"},
            )
        tool_name = tool_name.strip()

        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except json.JSONDecodeError:
                return (
                    ToolResponse(text=f"[Tool Error] Invalid tool arguments JSON for {tool_name}: {tool_args!r}"),
                    0.0,
                    {"error": "invalid_tool_arguments_json"},
                )
        elif isinstance(tool_args, list):
            if len(tool_args) == 1 and isinstance(tool_args[0], dict):
                tool_args = tool_args[0]
            else:
                return (
                    ToolResponse(
                        text=f"[Tool Error] Invalid tool call format for {tool_name}: arguments must be object, got {raw_tool_args!r}"
                    ),
                    0.0,
                    {"error": "invalid_tool_arguments"},
                )
        elif tool_args is None:
            tool_args = {}

        if not isinstance(tool_args, dict):
            return (
                ToolResponse(
                    text=f"[Tool Error] Invalid tool call format for {tool_name}: arguments must be object, got {raw_tool_args!r}"
                ),
                0.0,
                {"error": "invalid_tool_arguments"},
            )

        if tool_name not in self.tools:
            error_msg = f"[Tool Error] Unknown tool: {tool_name}. Available tools: {list(self.tools.keys())}"
            return ToolResponse(text=error_msg), 0.0, {"error": "unknown_tool"}

        tool = self.tools[tool_name]
        instance_id = None
        _tool_t0 = time.perf_counter()
        _tool_elapsed = None

        try:
            instance_id, _ = await tool.create(create_kwargs={})
            tool_response, reward, metrics = await tool.execute(instance_id, tool_args, **tools_kwargs)
            _tool_elapsed = time.perf_counter() - _tool_t0
        except Exception as e:
            logger.warning(f"Error executing tool {tool_name}: {e}")
            self._record_toolstat(tool_name, {"error": type(e).__name__},
                                  elapsed=time.perf_counter() - _tool_t0)
            return ToolResponse(text=f"[Tool Error] {str(e)}"), 0.0, {"error": str(e)}
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        # Truncate tool response if needed
        tool_response_text = tool_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[:self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length:]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]
            tool_response = ToolResponse(text=tool_response_text)

        self._record_toolstat(tool_name, metrics, elapsed=_tool_elapsed)
        return tool_response, reward, metrics
