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

S50 (18.10., same user order): CHUNKED IS NOT BIG. The S32 chunk stamp (``_mixed_d_was_chunked``, lifetime-sticky) stood
for "big prefill" (R6 rest: the small last chunk of a big prefill must not read short). But D chunks by its own chunk budget
(rc13 D log: chunks of 512, 1024, 1088, 1536, 1664, 2240 tokens), so a request of 745 uncached tokens under halbX 3116
(pdflip-14-180: 512 + 233) was chunked -- and then neither of its chunks mixed, although it is a SHORT D prefill by R2.
rc13, first 400 extend batches with running decodes: 274 mixed, 67 E2 hand-offs (no forward), 50 chunk forwards unmixed --
44 of the 57 chunked request instances in them had a total under halbX. The bound now is the request's WHOLE uncached
amount at its first D chunk (``seqlen - prefix``, stamped once with the chunk stamp, :func:`stamp_chunked`): under the
front's halbX = short, every chunk may mix (each chunk is smaller still); >= halbX = big, no chunk mixes (unchanged). A chunked
request without a stamped total is big (no invented bound). Switch ``FLLIPER_PDFLIP_MIXED_D_SHORT_CHUNKED`` (default ON).

S50 measurement note (rc13 D log 1018_200305): S48's "391 of 398 single-extend gaps ran unmixed" is a LOG-ORDER artifact --
a ``Decode rank batch`` line is written one round late, so the ``MIXED-D mixed`` line of the forward between rounds k and
k+2 stands BEFORE round k's line; counting only between the two decode lines finds 10 of 579, one line earlier 482 of 579.
642 of the 836 extend batches with running decodes were mixed. A mixed forward still lasts ~1.5 s (mean, the MoE
expert-major prefill stream), during which the decodes advance one token: the census below names the share per window.
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
    carried -> never). ``chunked`` = a BIG chunked request in or behind this batch (:func:`chunked_big`, R6 rest;
    S50: a chunked request whose whole uncached amount is under halbX is short). ``no_forward`` = an E2 hand-off."""
    if always():
        return int(x_tokens) > 0 and not no_forward
    if int(x_tokens) <= 0 or chunked or no_forward:
        return False
    hx = int(half_x_tokens)
    if hx <= 0:
        return False
    return int(u_tokens) < hx


def always() -> bool:
    """Measurement arm (user 29.10.): mix every D prefill batch, any size (D-only boot, no halbX). Default OFF."""
    from flliper.srt.environ import envs

    return bool(envs.FLLIPER_PDFLIP_MIXED_D_ALWAYS.get())


def chunked_enabled() -> bool:
    """S50: default ON; ``FLLIPER_PDFLIP_MIXED_D_SHORT_CHUNKED=0`` = any chunked request blocks the mix (S32 form)."""
    from flliper.srt.environ import envs

    return bool(envs.FLLIPER_PDFLIP_MIXED_D_SHORT_CHUNKED.get())


#: S50: the request's whole uncached amount at its FIRST D chunk (tokens), stamped once with the chunk stamp.
ATTR_TOTAL = "_mixed_d_u_total"
ATTR_CHUNKED = "_mixed_d_was_chunked"
#: S50 review A1: a stamp that could not be read once -- the request stays big for life
ATTR_UNREADABLE = "_mixed_d_u_total_unreadable"


def stamp_chunked(req) -> None:
    """The chunk stamp (S32, lifetime-sticky) plus the request's whole uncached amount at its first D chunk.

    Called where the scheduler counts a chunk of ``req`` into this pass's batch; at the FIRST such pass
    ``extend_range.start`` is the prefix the request was admitted on, so ``seqlen - start`` is what D must prefill in all
    (the amount R2 classifies). Later chunks of the same admission read a smaller rest and keep the first value (the last
    chunk of a big request does not read short). S50 review A1: the stamp is the MAXIMUM over its passes, never only the
    first -- a request re-admitted on a shorter prefix (retract / park / resume lost part of it: ``seqlen - start`` is now
    larger) is judged on its new whole amount, not on a smaller total of an earlier admission. An unreadable stamp makes
    the request big for life (no invented bound)."""
    setattr(req, ATTR_CHUNKED, True)
    if getattr(req, ATTR_UNREADABLE, False):
        return
    try:
        er = getattr(req, "extend_range", None)
        prefix_indices = getattr(req, "prefix_indices", None)  # a torch.Tensor: never ``or ()`` on it
        start = int(er.start) if er is not None else (0 if prefix_indices is None else len(prefix_indices))
        seqlen = int(getattr(req, "seqlen"))
        total = max(0, seqlen - start)
    except Exception:  # noqa: BLE001 -- no total = big (chunked_big), never an invented bound
        setattr(req, ATTR_UNREADABLE, True)
        setattr(req, ATTR_TOTAL, None)
        return
    old = getattr(req, ATTR_TOTAL, None)
    if old is None or total > int(old):
        setattr(req, ATTR_TOTAL, total)


def chunked_big(reqs: Iterable, chunked_req, half_x_tokens: int) -> bool:
    """Rank-local term for :func:`short_gate`'s ``chunked``: True when a chunked request in (or behind) this batch is BIG.

    S50: big = its stamped total (:data:`ATTR_TOTAL`) is missing or >= the batch's halbX. With the switch off every chunked
    request counts as big (S32). A request never chunked is judged by the batch sum ``u`` alone (unchanged)."""
    cands = [r for r in (reqs or ()) if getattr(r, ATTR_CHUNKED, False)]
    if chunked_req is not None:
        cands.append(chunked_req)
    if not cands:
        return False
    if not chunked_enabled():
        return True
    hx = int(half_x_tokens or 0)
    for r in cands:
        total = getattr(r, ATTR_TOTAL, None)
        if total is None or hx <= 0 or int(total) >= hx:
            return True
    return False


def refusal_reason(u_tokens: int, x_tokens: int, half_x_tokens: int, chunked: bool, no_forward: bool, local_ok: bool) -> str:
    """The census name of a rank-local verdict (first matching term, the order of :func:`short_gate`); "" = local yes."""
    if int(x_tokens) <= 0:
        return "not_d"
    if no_forward:
        return "e2_handoff"
    if always():
        return "" if local_ok else "local_block"
    if chunked:
        return "big_chunk"
    if int(half_x_tokens) <= 0:
        return "no_halfx"
    if int(u_tokens) >= int(half_x_tokens):
        return "u_ge_halfx"
    if not local_ok:
        return "local_block"
    return ""


_CENSUS = {"n": 0, "mixed": 0, "refused": {}}


def note_verdict(local_reason: str, agreed: bool) -> None:
    """S50 instrument: count one gate decision (entered with running decodes); every N decisions one census line.

    ``local_reason`` = :func:`refusal_reason` of this rank; a local yes the group MIN turned down counts as ``group_veto``."""
    try:
        c = _CENSUS
        c["n"] += 1
        if agreed:
            c["mixed"] += 1
        else:
            k = local_reason or "group_veto"
            c["refused"][k] = c["refused"].get(k, 0) + 1
        from flliper.srt.environ import envs

        every = int(envs.FLLIPER_PDFLIP_MIXED_D_CENSUS_N.get() or 0)
        if every > 0 and c["n"] % every == 0:
            logger.info(
                "%s CENSUS decisions=%d mixed=%d share=%.3f refused=%s (decisions = prefill batches with running decodes on D; "
                "refused names the first blocking term)",
                MARK, c["n"], c["mixed"], c["mixed"] / max(1, c["n"]), dict(sorted(c["refused"].items())),
            )
    except Exception:  # noqa: BLE001 -- an instrument never stops a batch
        pass


def note_mixed(u_tokens: int, n_decode_rows: int, half_x_tokens: int, x_tokens: int) -> None:
    """The instrument line, one per mixed batch (cumulative count rides on it)."""
    _COUNT[0] += 1
    logger.info(
        "%s mixed n=%d n_prefill_tok=%d n_decode_rows=%d halfX=%d X=%d (halfX = the front's, min over the batch; X = D riegel)",
        MARK, _COUNT[0], u_tokens, n_decode_rows, half_x_tokens, x_tokens,
    )
