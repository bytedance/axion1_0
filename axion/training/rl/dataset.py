"""Dataset filter using the exact DeepResearch session prefix, without truncation."""
import os

from verl.utils.dataset.rl_dataset import RLHFDataset

from axion.training.rl.prompts import build_initial_messages, prompt_template_kwargs, tokenize_session_prompt


class PaperRLHFDataset(RLHFDataset):
    """Filter system + question; cache exact initial tokens for the agent factory."""

    def _initial_prompt(self, row):
        messages = row[self.prompt_key]
        question = next((message.get("content") for message in messages if message.get("role") == "user"), None)
        if not isinstance(question, str):
            raise ValueError("The paper RL dataset requires a text user question")
        soften = os.environ.get("SOFTEN_ANSWER_PROMPT", "0").strip().lower() in {"1", "true", "t", "yes", "y", "on"}
        actual_messages = build_initial_messages(question, soften_answer_prompt=soften)
        kwargs = prompt_template_kwargs(self.apply_chat_template_kwargs)
        ids = tokenize_session_prompt(self.tokenizer, actual_messages, kwargs)
        return actual_messages, ids

    def maybe_filter_out_long_prompts(self, dataframe=None):
        if not self.filter_overlong_prompts:
            raise ValueError("PaperRLHFDataset requires filter_overlong_prompts=True")
        return dataframe.filter(
            lambda row: len(self._initial_prompt(row)[1]) <= self.max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering complete research prompts over {self.max_prompt_length} tokens",
        )

    def __getitem__(self, item):
        row = super().__getitem__(item)
        messages, ids = self._initial_prompt(row)
        if len(ids) > self.max_prompt_length:
            raise ValueError("Session prompt changed after filtering and now exceeds max_prompt_length")
        row["axion_initial_messages"] = messages
        row["axion_initial_prompt_ids"] = ids
        return row
