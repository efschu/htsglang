# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
"""Per-request thinking budget: token-id derivation and request attachment.

The enforcement vehicle is ``ThinkingBudgetLogitProcessor`` in
``sglang/srt/sampling/custom_logit_processor.py``. This module owns everything
that must happen *before* the request reaches the scheduler:

* validating the per-request scalar (``SamplingParams.thinking_budget``),
* deriving the reasoning marker token ids from the *deployed* tokenizer
  instead of a hardcoded per-model constant,
* attaching the built-in processor so clients do not have to serialize one.

Why tokenizer-derived ids: the original design carried the marker ids as class
constants per model family. When the deployed checkpoint tokenizes ``<think>``
to a different id, the processor's membership checks never fire and the budget
is silently ignored -- see sgl-project/sglang#25536 (Qwen3.6) and #20274
(GLM-5). A budget that cannot be enforced must be a named refusal at admission
time, never a silent no-op, so every id used at runtime is derived here from
the tokenizer that actually serves the request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from sglang.srt.sampling.sampling_params import SamplingParams

logger = logging.getLogger(__name__)

# ``custom_params`` keys. ``THINKING_BUDGET_KEY`` is the documented,
# client-visible legacy key (docs/basic_usage/glm45.md, deepseek_v3.md); the
# other two are server-owned and stripped from client input.
THINKING_BUDGET_KEY = "thinking_budget"
THINKING_BUDGET_TOKEN_IDS_KEY = "thinking_budget_token_ids"
THINKING_BUDGET_INTERNAL_KEY = "thinking_budget_internal"

SERVER_OWNED_CUSTOM_PARAM_KEYS = frozenset(
    {THINKING_BUDGET_TOKEN_IDS_KEY, THINKING_BUDGET_INTERNAL_KEY}
)

# Sentinel used inside the JSON-safe ``custom_params`` list for "this tokenizer
# has no single-token newline". The processor then closes the thinking section
# without emitting a separating newline first.
NO_NEWLINE_TOKEN_ID = -1


class ThinkingBudgetUnsupportedError(ValueError):
    """A thinking budget was requested that this deployment cannot enforce.

    Raised at request admission so the caller gets a 400 naming the model and
    the markers, instead of a request that runs to completion with the budget
    quietly ignored.
    """


@dataclass(frozen=True)
class ThinkingBudgetTokenIds:
    """Reasoning marker token ids derived from the serving tokenizer."""

    start: int
    end: int
    newline: Optional[int]
    start_str: str
    end_str: str

    def to_param_list(self) -> List[int]:
        """JSON-safe representation carried in ``custom_params``."""
        return [
            self.start,
            self.end,
            self.newline if self.newline is not None else NO_NEWLINE_TOKEN_ID,
        ]

    @staticmethod
    def parse_param_list(value: Any) -> Optional[ThinkingBudgetTokenIds]:
        """Inverse of :meth:`to_param_list`; ``None`` when unusable."""
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return None
        try:
            start, end, newline = (int(v) for v in value)
        except (TypeError, ValueError):
            return None
        return ThinkingBudgetTokenIds(
            start=start,
            end=end,
            newline=None if newline < 0 else newline,
            start_str="",
            end_str="",
        )


def validate_thinking_budget(value: Any) -> Optional[int]:
    """Validate a per-request thinking budget; return ``None`` when unset.

    Mirrors vLLM's ``validate_thinking_token_budget``: a non-negative integer
    is a budget in tokens of the thinking section, ``-1`` and ``None`` both
    mean "no budget". ``bool`` is rejected explicitly because it is an ``int``
    subclass and ``True`` would otherwise become a 1-token budget.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            "thinking_budget must be a non-negative integer or -1 for "
            f"unlimited, got {value!r}."
        )
    if value == -1:
        return None
    if value < 0:
        raise ValueError(
            "thinking_budget must be a non-negative integer or -1 for "
            f"unlimited, got {value}."
        )
    return value


def _encode_single_token(tokenizer, text: str) -> Optional[int]:
    """Encode ``text`` and return its id iff it is exactly one token."""
    if not text:
        return None
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except Exception as e:  # tokenizer backends raise their own error types
        logger.debug("Failed to encode %r for thinking budget: %s", text, e)
        return None
    if ids is None or len(ids) != 1:
        return None
    return int(ids[0])


@lru_cache(maxsize=8)
def resolve_thinking_budget_token_ids(
    tokenizer,
    reasoning_parser: Optional[str],
    model_name: str = "",
) -> ThinkingBudgetTokenIds:
    """Derive the thinking marker token ids from the serving tokenizer.

    The marker *strings* come from the configured reasoning parser (the same
    detector the output path uses), the *ids* come from the tokenizer that
    serves this deployment. Anything that prevents an unambiguous derivation
    raises :class:`ThinkingBudgetUnsupportedError` naming the model and the
    markers -- callers turn that into a 400.

    Cached because it runs on every budgeted request; the cache key includes
    the tokenizer object, so a reload produces a fresh derivation.
    """
    model_label = model_name or getattr(tokenizer, "name_or_path", "") or "unknown"

    if not reasoning_parser:
        raise ThinkingBudgetUnsupportedError(
            f"thinking_budget was requested for model '{model_label}' but the "
            "server was launched without --reasoning-parser, so the thinking "
            "section markers are unknown and the budget cannot be enforced."
        )
    if tokenizer is None:
        raise ThinkingBudgetUnsupportedError(
            f"thinking_budget was requested for model '{model_label}' but the "
            "server has no tokenizer (skip_tokenizer_init=True), so the "
            "thinking markers cannot be resolved to token ids."
        )

    # Imported lazily: the reasoning parser module pulls in the OpenAI
    # protocol types, which must not become an import-time dependency of the
    # sampling package.
    from sglang.srt.parser.reasoning_parser import ReasoningParser

    try:
        detector = ReasoningParser(
            model_type=reasoning_parser,
            stream_reasoning=False,
            tokenizer=tokenizer,
        ).detector
    except ValueError as e:
        raise ThinkingBudgetUnsupportedError(
            f"thinking_budget was requested for model '{model_label}' but the "
            f"reasoning parser '{reasoning_parser}' could not be built: {e}"
        ) from e

    start_str = getattr(detector, "think_start_token", "") or ""
    end_str = getattr(detector, "think_end_token", "") or ""

    start_id = _encode_single_token(tokenizer, start_str)
    end_id = _encode_single_token(tokenizer, end_str)
    if start_id is None or end_id is None:
        raise ThinkingBudgetUnsupportedError(
            f"thinking_budget was requested for model '{model_label}' but the "
            f"reasoning markers of parser '{reasoning_parser}' do not encode "
            f"to exactly one token each with this checkpoint's tokenizer "
            f"(start={start_str!r} -> {'ok' if start_id is not None else 'not a single token'}, "
            f"end={end_str!r} -> {'ok' if end_id is not None else 'not a single token'}). "
            "Enforcing a budget would require rewriting multi-token markers "
            "mid-generation, so the request is refused instead of silently "
            "ignoring the budget."
        )

    # The separating newline is cosmetic: it keeps the forced close on its own
    # line the way the model would have written it. A tokenizer without a
    # single-token newline simply closes the section directly.
    newline_id = _encode_single_token(tokenizer, "\n")
    if newline_id is None:
        logger.debug(
            "Thinking budget: no single-token newline for model '%s'; the "
            "thinking section will be closed without a separating newline.",
            model_label,
        )

    return ThinkingBudgetTokenIds(
        start=start_id,
        end=end_id,
        newline=newline_id,
        start_str=start_str,
        end_str=end_str,
    )


@lru_cache(maxsize=1)
def internal_thinking_budget_processor_str() -> str:
    """Serialized built-in processor attached by the server itself.

    Clients do not have to send a ``custom_logit_processor`` (and this path
    does not require ``--enable-custom-logit-processor``): the string is
    produced here from the in-tree class, never from client input.
    """
    from sglang.srt.sampling.custom_logit_processor import (
        ThinkingBudgetLogitProcessor,
    )

    return ThinkingBudgetLogitProcessor.to_str()


def _strip_server_owned_keys(
    custom_params: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(custom_params, dict):
        return custom_params
    if not SERVER_OWNED_CUSTOM_PARAM_KEYS & custom_params.keys():
        return custom_params
    return {
        k: v
        for k, v in custom_params.items()
        if k not in SERVER_OWNED_CUSTOM_PARAM_KEYS
    }


def attach_thinking_budget(
    sampling_params: SamplingParams,
    custom_logit_processor: Optional[str],
    tokenizer,
    reasoning_parser: Optional[str],
    model_name: str = "",
) -> Optional[str]:
    """Resolve and wire the thinking budget for one request.

    Mutates ``sampling_params.custom_params`` in place (replacing the dict, not
    editing the caller's) and returns the custom logit processor string the
    request should carry.

    Two client-facing forms are supported and both end up on the same
    enforcement path with tokenizer-derived ids:

    1. ``sampling_params.thinking_budget`` -- the first-class scalar
       (``extra_body={"thinking_budget": N}`` on the OpenAI chat front, or the
       Anthropic ``thinking.budget_tokens``). The server attaches the built-in
       processor itself.
    2. ``custom_params["thinking_budget"]`` together with a client-supplied
       ``custom_logit_processor`` -- the documented legacy form. The ids are
       injected here as well, which is what makes the legacy form immune to the
       hardcoded-id mismatch of #25536 / #20274.
    """
    custom_params = sampling_params.custom_params
    requested = getattr(sampling_params, "thinking_budget", None)
    legacy_requested = (
        custom_params.get(THINKING_BUDGET_KEY)
        if isinstance(custom_params, dict)
        else None
    )

    if requested is None and legacy_requested is None:
        # No budget requested: make sure a client cannot smuggle server-owned
        # keys into the scheduler.
        sampling_params.custom_params = _strip_server_owned_keys(custom_params)
        return custom_logit_processor

    if requested is not None and custom_logit_processor is not None:
        raise ValueError(
            "thinking_budget cannot be combined with a client-supplied "
            "custom_logit_processor: a request carries exactly one processor. "
            "Either drop custom_logit_processor and let the server attach the "
            "built-in thinking budget processor, or pass the budget as "
            "custom_params={'thinking_budget': N} for your own processor."
        )

    budget = validate_thinking_budget(
        requested if requested is not None else legacy_requested
    )
    if budget is None:
        # Explicit "unlimited" (-1): behave exactly like an absent budget.
        sampling_params.thinking_budget = None
        sampling_params.custom_params = _strip_server_owned_keys(custom_params)
        return custom_logit_processor

    token_ids = resolve_thinking_budget_token_ids(
        tokenizer, reasoning_parser, model_name
    )

    new_params: Dict[str, Any] = dict(custom_params or {})
    new_params[THINKING_BUDGET_KEY] = budget
    new_params[THINKING_BUDGET_TOKEN_IDS_KEY] = token_ids.to_param_list()
    if custom_logit_processor is None:
        new_params[THINKING_BUDGET_INTERNAL_KEY] = True
        custom_logit_processor = internal_thinking_budget_processor_str()
    else:
        new_params.pop(THINKING_BUDGET_INTERNAL_KEY, None)

    sampling_params.thinking_budget = budget
    sampling_params.custom_params = new_params
    return custom_logit_processor
