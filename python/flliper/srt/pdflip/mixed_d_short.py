# SPDX-License-Identifier: Apache-2.0
"""MIXED-D (S32, 18.10.): the SHORT gate in front of the upstream mixed-chunk path (fLLiper PR #36933, ported 1:1 elsewhere).

User order 18.10.: "mixed chunk fuer kurze D-Prefills wieder an." Upstream's mixed chunk (+ EAGLE, #36933) is taken as it is;
this file is only the gate that restricts it to group D and to SHORT prefills (SCHEDULER-SOLL R2: u < halbX). Never P (P has no
D riegel, X = 0), never a big prefill, never a chunk of a chunked request, never a batch that runs no target forward.
Rank agreement (R11) is taken by the caller: ``Scheduler._pdflip_group_min_flags``. Switch ``FLLIPER_PDFLIP_MIXED_D_SHORT``, default ON.

S33 review A1 (SOLL section 2 "X nie ein fester Wert", R2a, R11): halbX is the one THE FRONT carried with each D leg
(``weg2_front_halfx``, F_halb x the X of its decision at the request's cache depth, computed with the front's flag), the
smallest over the batch's requests (R2a: S against the smallest X of the members). NOT 0.5 x the rank's riegel
(``--tp-prefill-max-tokens`` = 12288 on NF -> 6144, above every curve X of 4096..5800 the front decides with: the very bound
H16V review A3 removed from the resumed-tail extend). A request without a carried halbX is not mixed (no invented bound).
S33 review A2: a batch holding an E2 END-state hand-off (``tail_adopt.SKIP_PLANS``) runs NO target forward; folding decode
rows into it makes ``tail_adopt.skip_tokens`` refuse the batch ("PDFLIP-TAIL-SKIP-EXTEND mixed batch") on every rank.
"""
from __future__ import annotations

import logging
from typing import Iterable

logger = logging.getLogger(__name__)
MARK = "MIXED-D"
_COUNT = [0]


def enabled() -> bool:
    """Default ON (new feature); ``FLLIPER_PDFLIP_MIXED_D_SHORT=0`` = the old halt-the-decodes path."""
    from flliper.srt.environ import envs

    return bool(envs.FLLIPER_PDFLIP_MIXED_D_SHORT.get())


def batch_half_x(reqs: Iterable) -> int:
    """The smallest front-carried halbX of the batch's requests; 0 when any request carries none (or the batch is empty).

    Replicated: ``weg2_front_halfx`` rides the broadcast request to every TP rank (scheduler.handle_generate_request)."""
    from flliper.srt.pdflip.front_x import ATTR_HALFX

    best = 0
    for r in reqs or ():
        try:
            hx = int(getattr(r, ATTR_HALFX, None) or 0)
        except (TypeError, ValueError):
            return 0
        if hx <= 0:
            return 0
        best = hx if best == 0 else min(best, hx)
    return best


def end_state_handoff(reqs: Iterable) -> bool:
    """True when a request of the batch was admitted with P's END state (E2 skip plan): that batch runs no target forward."""
    try:
        from flliper.srt.pdflip import tail_adopt
    except Exception:  # noqa: BLE001 -- no tail adopt in this build: no skip plan can exist
        return False
    plans = tail_adopt.SKIP_PLANS
    return bool(plans) and any(str(getattr(r, "rid", "")) in plans for r in (reqs or ()))


def short_gate(u_tokens: int, x_tokens: int, half_x_tokens: int, chunked: bool, no_forward: bool = False) -> bool:
    """Rank-local verdict: fold the running decodes into this prefill batch?

    ``x_tokens`` <= 0 = not group D (P: no riegel). ``half_x_tokens`` = :func:`batch_half_x` (the front's halbX, 0 = none
    carried -> never). ``chunked`` = a chunked request in or behind this batch (R6 rest). ``no_forward`` = an E2 hand-off."""
    if int(x_tokens) <= 0 or chunked or no_forward:
        return False
    hx = int(half_x_tokens)
    if hx <= 0:
        return False
    return int(u_tokens) < hx


def note_mixed(u_tokens: int, n_decode_rows: int, half_x_tokens: int, x_tokens: int) -> None:
    """The instrument line, one per mixed batch (cumulative count rides on it)."""
    _COUNT[0] += 1
    logger.info(
        "%s mixed n=%d n_prefill_tok=%d n_decode_rows=%d halfX=%d X=%d (halfX = the front's, min over the batch; X = D riegel)",
        MARK, _COUNT[0], u_tokens, n_decode_rows, half_x_tokens, x_tokens,
    )
