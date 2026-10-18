# SPDX-License-Identifier: Apache-2.0
"""MIXED-D (S32, 18.10.): the SHORT gate in front of the upstream mixed-chunk path (SGLang PR #36933, ported 1:1 elsewhere).

User order 18.10.: "mixed chunk fuer kurze D-Prefills wieder an." Upstream's mixed chunk (+ EAGLE, #36933) is taken as it is;
this file is only the gate that restricts it to group D and to SHORT prefills (SCHEDULER-SOLL R2: u < halbX, halbX = F_halb x X,
SGLANG_WEG2_HALF_X_FRAC). Never P (P has no D riegel, X = 0), never a big prefill, never a chunk of a chunked request.
Rank agreement (R11) is taken by the caller: ``Scheduler._weg2_group_min_flags``. Switch ``SGLANG_WEG2_MIXED_D_SHORT``, default ON.
"""
from __future__ import annotations

import logging

from sglang.srt.weg2.phase_policy import half_x_of

logger = logging.getLogger(__name__)
MARK = "MIXED-D"
_COUNT = [0]


def enabled() -> bool:
    """Default ON (new feature); ``SGLANG_WEG2_MIXED_D_SHORT=0`` = the old halt-the-decodes path."""
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_WEG2_MIXED_D_SHORT.get())


def short_gate(u_tokens: int, x_tokens: int, half_frac: float, chunked: bool) -> bool:
    """Rank-local verdict: fold the running decodes into this prefill batch?  x_tokens <= 0 = not group D."""
    if int(x_tokens) <= 0 or chunked:
        return False
    return int(u_tokens) < half_x_of(x_tokens, half_frac)


def note_mixed(u_tokens: int, n_decode_rows: int, x_tokens: int, half_frac: float) -> None:
    """The instrument line, one per mixed batch (cumulative count rides on it)."""
    _COUNT[0] += 1
    logger.info(
        "%s mixed n=%d n_prefill_tok=%d n_decode_rows=%d halfX=%d X=%d",
        MARK, _COUNT[0], u_tokens, n_decode_rows, half_x_of(x_tokens, half_frac), x_tokens,
    )
