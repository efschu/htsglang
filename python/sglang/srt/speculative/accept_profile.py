"""Per-request acceptance profile of speculative decoding (fnFL2 H36).

WHY. The flip form's decode acceptance was compared with Form A's through
window means (``Decode batch ... accept len``, one number per ~40 rounds of
whatever request happened to be running) and through the probe's
``completion_tokens / stream deltas``. Neither says how the rounds of ONE
request are distributed over 0..k correct drafts, nor whether a flipped
request -- whose draft KV over the prompt is the #993 zero fill (DRAFT-COLD)
-- accepts less in its first rounds than later, which is the one signature a
cold draft context would leave. The per-request histogram
(``Req.spec_correct_drafts_histogram``) already exists on the CPU; it only
never reached the log, and it has no head/rest split.

WHAT. ``note_round`` keeps a second histogram over the request's first
``SGLANG_LOG_SPEC_ACCEPT_PROFILE_HEAD_ROUNDS`` verify rounds, next to the
whole-request one the result processor keeps. ``log_finished`` writes one
``SPEC-ACCEPT-PROFILE`` line at finish on TP rank 0 (every rank holds the same
counts -- the accept decision is broadcast group-wide -- so one line suffices).
Both read Python ints the processor already moved to the CPU for its own
bookkeeping: no device sync, no collective.

Naming (speculative-naming): ``correct_drafts`` excludes the bonus token,
``accept_length`` includes it (tokens per verify round, the EAGLE tau).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

#: Req attribute holding the correct-drafts histogram of the first N rounds.
HEAD_HISTOGRAM_ATTR = "spec_head_correct_drafts_histogram"
#: phase_flip_draft_bootstrap.COLD_ARMED_ATTR, read by name so this module
#: does not import the flip machinery (a non-flip server never sets it).
DRAFT_COLD_ATTR = "phase_flip_draft_cold_armed"


def head_rounds() -> int:
    """First-N-rounds window of the profile; 0 = profile off."""
    value = envs.SGLANG_LOG_SPEC_ACCEPT_PROFILE_HEAD_ROUNDS.get()
    return max(0, int(value or 0))


def _bump(histogram: List[int], num_correct_drafts: int) -> None:
    if len(histogram) <= num_correct_drafts:
        histogram.extend([0] * (num_correct_drafts - len(histogram) + 1))
    histogram[num_correct_drafts] += 1


def note_round(req, num_correct_drafts: int, head: int) -> None:
    """Count one verify round into the head histogram.

    Call AFTER ``req.spec_verify_ct`` was incremented for this round, so the
    first round has ``spec_verify_ct == 1``.
    """
    if head <= 0 or req.spec_verify_ct > head:
        return
    histogram = getattr(req, HEAD_HISTOGRAM_ATTR, None)
    if histogram is None:
        histogram = []
        setattr(req, HEAD_HISTOGRAM_ATTR, histogram)
    _bump(histogram, int(num_correct_drafts))


def rest_histogram(total: Sequence[int], head: Sequence[int]) -> List[int]:
    width = max(len(total), len(head))
    return [
        (total[i] if i < len(total) else 0) - (head[i] if i < len(head) else 0)
        for i in range(width)
    ]


def accept_length(histogram: Sequence[int]) -> Optional[float]:
    """Tokens per verify round incl. the bonus token; None without rounds."""
    rounds = sum(histogram)
    if rounds <= 0:
        return None
    return sum((k + 1) * n for k, n in enumerate(histogram)) / rounds


def _fmt_len(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _fmt_share(histogram: Sequence[int]) -> str:
    rounds = sum(histogram)
    if rounds <= 0:
        return "n/a"
    return "/".join(f"{k}:{n / rounds:.3f}" for k, n in enumerate(histogram))


def format_profile(req, head: int) -> Optional[str]:
    """The SPEC-ACCEPT-PROFILE line of a finished request, or None when the
    request ran no verify round (no speculation, or a finish at prefill)."""
    total = list(getattr(req, "spec_correct_drafts_histogram", None) or [])
    if head <= 0 or sum(total) <= 0:
        return None
    head_hist = list(getattr(req, HEAD_HISTOGRAM_ATTR, None) or [])
    rest = rest_histogram(total, head_hist)
    prompt = len(getattr(req, "origin_input_ids", None) or [])
    return (
        f"SPEC-ACCEPT-PROFILE rid={getattr(req, 'rid', '?')} "
        f"draft_cold={'yes' if getattr(req, DRAFT_COLD_ATTR, False) else 'no'} "
        f"prompt={prompt} rounds={sum(total)} "
        f"accept_length={_fmt_len(accept_length(total))} "
        f"correct_drafts_hist={total} share={_fmt_share(total)} | "
        f"head{head} rounds={sum(head_hist)} "
        f"accept_length={_fmt_len(accept_length(head_hist))} hist={head_hist} | "
        f"rest rounds={sum(rest)} accept_length={_fmt_len(accept_length(rest))} "
        f"hist={rest} (per verify round, bonus included; a draft_cold "
        f"request's 1-node seed verify is one 0-draft round of its head)"
    )


def _is_log_rank() -> bool:
    try:
        from sglang.srt.distributed import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank() == 0
    except Exception:  # noqa: BLE001 - no TP group (unit tests, tools): log
        return True


def log_finished(req) -> Optional[str]:
    """Log the profile of a finished request on TP rank 0; returns the line."""
    head = head_rounds()
    if head <= 0 or not _is_log_rank():
        return None
    line = format_profile(req, head)
    if line is not None:
        logger.info(line)
    return line
