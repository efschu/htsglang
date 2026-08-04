import json
from abc import ABC, abstractmethod
from array import array
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import dill
import orjson
import torch

from sglang.srt.sampling.thinking_budget import (
    THINKING_BUDGET_KEY,
    THINKING_BUDGET_TOKEN_IDS_KEY,
    ThinkingBudgetTokenIds,
    ThinkingBudgetUnsupportedError,
    validate_thinking_budget,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


@lru_cache(maxsize=None)
def _cache_from_str(json_str: str):
    """Deserialize a json string to a Callable object.
    This function is cached to avoid redundant deserialization.
    """
    data = orjson.loads(json_str)
    return dill.loads(bytes.fromhex(data["callable"]))


class CustomLogitProcessor(ABC):
    """Abstract base class for callable functions."""

    @abstractmethod
    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        """Define the callable behavior."""
        raise NotImplementedError

    @classmethod
    def to_str(cls) -> str:
        """Serialize the callable function to a JSON-compatible string."""
        return json.dumps({"callable": dill.dumps(cls).hex()})

    @classmethod
    def from_str(cls, json_str: str):
        """Deserialize a callable function from a JSON string."""
        return _cache_from_str(json_str)()


class DisallowedTokensLogitsProcessor(CustomLogitProcessor):
    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        disallowed_token_ids = custom_param_list[0]["token_ids"]
        assert all(
            disallowed_token_ids == c["token_ids"] for c in custom_param_list
        ), f"{custom_param_list=}"
        logits[..., disallowed_token_ids] = -float("inf")
        return logits


class ThinkingBudgetLogitProcessor(CustomLogitProcessor):
    """Caps the length of the thinking section of a reasoning model.

    Once the section started by the model's thinking-start marker has produced
    ``thinking_budget`` tokens, the next sampled token is forced: first a
    newline (so the close lands on its own line), then the thinking-end marker.

    Token ids come from ``custom_params["thinking_budget_token_ids"]``, which
    the tokenizer manager derives from the tokenizer that actually serves the
    request (see ``sglang/srt/sampling/thinking_budget.py``). The class-level
    constants below are the legacy per-model fallback for callers that
    instantiate a subclass outside the server; when a checkpoint's tokenizer
    disagrees with them the budget would be silently ignored, which is exactly
    the defect behind sgl-project/sglang#25536 and #20274. Server-side requests
    therefore always carry derived ids, and a request that asks for a budget
    with no ids available raises instead of quietly doing nothing.
    """

    THINKING_START_TOKEN_ID: Optional[int] = None
    THINKING_END_TOKEN_ID: Optional[int] = None
    NEW_LINE_TOKEN_ID: Optional[int] = None

    def _resolve_token_ids(
        self, param_dict: Dict[str, Any]
    ) -> "ThinkingBudgetTokenIds":
        derived = ThinkingBudgetTokenIds.parse_param_list(
            param_dict.get(THINKING_BUDGET_TOKEN_IDS_KEY)
        )
        if derived is not None:
            return derived

        if self.THINKING_START_TOKEN_ID is None or self.THINKING_END_TOKEN_ID is None:
            raise ThinkingBudgetUnsupportedError(
                f"{type(self).__name__} was asked to enforce a thinking budget "
                "but the request carries no tokenizer-derived marker ids "
                f"('{THINKING_BUDGET_TOKEN_IDS_KEY}') and this class defines no "
                "fallback ids. Refusing rather than silently ignoring the "
                "budget."
            )
        return ThinkingBudgetTokenIds(
            start=self.THINKING_START_TOKEN_ID,
            end=self.THINKING_END_TOKEN_ID,
            newline=self.NEW_LINE_TOKEN_ID,
            start_str="",
            end_str="",
        )

    def __call__(self, logits, custom_param_list: list[dict[str, Any]]):
        if custom_param_list is None or not custom_param_list:
            return logits
        for i, param_dict in enumerate(custom_param_list):
            if param_dict is None:
                continue

            # A malformed value is a client error, not a reason to drop the
            # budget on the floor: validate_thinking_budget raises.
            thinking_budget = validate_thinking_budget(
                param_dict.get(THINKING_BUDGET_KEY)
            )
            if thinking_budget is None:
                continue

            token_ids = self._resolve_token_ids(param_dict)
            req: Req = param_dict.get("__req__")
            cur_ids: list[int] = [*req.origin_input_ids, *req.output_ids]

            # Check if out of thinking stage
            if token_ids.start not in cur_ids or token_ids.end in cur_ids:
                continue

            # Find the index of the thinking start token
            start_index = cur_ids.index(token_ids.start)

            # Count the number of tokens after the thinking start token
            num_tokens_after_start = len(cur_ids) - start_index - 1

            if num_tokens_after_start < thinking_budget:
                continue

            # Ensure new line token before thinking end token
            if token_ids.newline is not None and (
                not req.output_ids or req.output_ids[-1] != token_ids.newline
            ):
                logits[i, :] = -float("inf")
                logits[i, token_ids.newline] = 0.0
                continue

            # Assign highest probability to the thinking end token
            logits[i, :] = -float("inf")
            logits[i, token_ids.end] = 0.0

        return logits


class Glm4MoeThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """Thinking budget for GLM-4.5 / GLM-4.6 / GLM-4.5V / GLM-4.6V models.

    The ids below are a fallback only; server-side requests use ids derived
    from the deployed tokenizer.
    """

    THINKING_START_TOKEN_ID: Optional[int] = 151350
    THINKING_END_TOKEN_ID: Optional[int] = 151351
    NEW_LINE_TOKEN_ID: Optional[int] = 198


class Qwen3ThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """Thinking budget for Qwen3 models.

    The ids below are a fallback only; server-side requests use ids derived
    from the deployed tokenizer.
    """

    THINKING_START_TOKEN_ID: Optional[int] = 151667
    THINKING_END_TOKEN_ID: Optional[int] = 151668
    NEW_LINE_TOKEN_ID: Optional[int] = 198


class DeepSeekR1ThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    """Thinking budget for DeepSeek-R1 models.

    The ids below are a fallback only; server-side requests use ids derived
    from the deployed tokenizer.
    """

    THINKING_START_TOKEN_ID: Optional[int] = 128798
    THINKING_END_TOKEN_ID: Optional[int] = 128799
    NEW_LINE_TOKEN_ID: Optional[int] = 201


# Adapted from DeepSeek's implementation: https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/DeepSeek-OCR-master/DeepSeek-OCR-vllm/process/ngram_norepeat.py
class DeepseekOCRNoRepeatNGramLogitProcessor(CustomLogitProcessor):
    """Block n-gram repetitions within a sliding window for DeepSeek-OCR outputs."""

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        if not custom_param_list:
            return logits

        for batch_idx, params in enumerate(custom_param_list):
            if not params:
                continue

            req = params.get("__req__")
            if req is None:
                continue

            try:
                ngram_size = int(params.get("ngram_size") or 0)
                window_size = int(params.get("window_size") or 0)
            except (TypeError, ValueError):
                continue

            if ngram_size <= 0 or window_size <= 0:
                continue

            sequence = req.origin_input_ids + req.output_ids
            if len(sequence) < ngram_size:
                continue

            search_start = max(0, len(sequence) - window_size)
            search_end = len(sequence) - ngram_size + 1
            if search_end <= search_start:
                continue

            if ngram_size > 1:
                current_prefix = sequence[-(ngram_size - 1) :]
            else:
                current_prefix = array("q")

            banned_tokens: Set[int] = set()
            for idx in range(search_start, search_end):
                ngram = sequence[idx : idx + ngram_size]
                if ngram_size == 1 or ngram[:-1] == current_prefix:
                    banned_tokens.add(ngram[-1])

            whitelist_ids = params.get("whitelist_token_ids") or []
            try:
                whitelist = {int(token_id) for token_id in whitelist_ids}
            except (TypeError, ValueError):
                whitelist = set()

            banned_tokens.difference_update(whitelist)

            if not banned_tokens:
                continue

            indices = list(banned_tokens)
            logits[batch_idx, indices] = -float("inf")

        return logits
