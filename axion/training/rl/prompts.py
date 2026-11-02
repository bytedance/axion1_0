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
"""Shared exact initial prompt construction for filtering and rollout."""
import os
from datetime import date
from pathlib import Path

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


def build_system_prompt(*, soften_answer_prompt=False):
    prompt = SYSTEM_PROMPT + date.today().strftime("%Y-%m-%d")
    if soften_answer_prompt:
        prompt += (
            "\n\nIMPORTANT: Do not refuse to answer or leave the answer blank because you are "
            "uncertain. When you have researched enough, commit to your single best-supported "
            "answer based on the evidence gathered, and always place it inside <answer></answer> "
            "tags. Never end your response without a final <answer>."
        )
    return prompt


def build_initial_messages(question, *, soften_answer_prompt=False):
    if not isinstance(question, str):
        raise TypeError("The paper RL recipe requires a text question")
    return [
        {"role": "system", "content": build_system_prompt(soften_answer_prompt=soften_answer_prompt)},
        {"role": "user", "content": question},
    ]


def prompt_template_kwargs(configured):
    """Mirror AgentLoopBase's supported optional template file override."""
    result = dict(configured or {})
    template_file = os.environ.get("VERL_CHAT_TEMPLATE_FILE")
    if template_file and "chat_template" not in result:
        result["chat_template"] = Path(template_file).read_text()
    return result


def tokenize_session_prompt(tokenizer, messages, template_kwargs):
    from verl.utils.tokenizer import normalize_token_ids
    from verl.utils.tokenizer.chat_template import apply_chat_template

    return normalize_token_ids(apply_chat_template(
        tokenizer, messages, add_generation_prompt=True, tokenize=True,
        **template_kwargs,
    ))
