"""#1268/#1460 z30j: a forwarded /flush_cache is decided ONCE, on PP0.

THE DEFECT (27B boot rc12z30j, bb82fbcb68, flip epoch=120, 00:44:12Z). The
front polls /flush_cache on group P. Only PP0 receives the RPC; it forwards
the ``FlushCacheReqInput`` down the request chain (``#1460 CTRL-FWD``) BEFORE
its own dispatch (commit, forward, then process -- #633), and every follower
then ran ``flush_cache`` on ITS OWN idleness (``group_idle_verdict``: "a
follower's answer never leaves the group"). PP0 answered "Cache not flushed
... GROUP VERDICT PENDING" (no idle-vote lap landed yet), PP1/PP2 answered
"Cache flushed successfully!" and dropped their Mamba anchors
(``PDFLIP-ANCHOR-LOST at=flush n=2 depths=[16383, 20990]``). The ranks were
uneins: PP0 kept a tree its followers had reset, and the pass after it parked
12 s (PP0 in the idle-vote-home recv, PP1/PP2 in CHAIN-RECV) until three
expiries let it go -- the 14283 ms P->D flip. The follower's answer does not
leave the group, but its RESET does. rc12o27 b1 was the same class (PP0
refused, PP1/PP2 flushed a released group); FD closed only its dormant case.

THE FORM (NF's recommendation, operator 29.09.). PP0 stays authoritative and
the forward path stays. PP0 stamps each ``FlushCacheReqInput`` it puts on the
wire with a sequence number (``pdflip_flush_seq``); a follower that receives a
stamped flush does NOT decide it -- it parks it. PP0 decides it in its own
dispatch exactly as before and records the verdict; the NEXT pass puts
``PdFlipFlushVerdict(seq, passed)`` at the FRONT of the wire, so it reaches every
follower before anything the front sends after PP0's answer (the sleep leg's
release follows a 200 and therefore lands behind it). A follower flushes only
on ``passed``, and drops the parked flush by name otherwise.

WHAT A FOLLOWER STILL CHECKS ON ``passed``: its dormant refusal and its own
idleness, both inside the ordinary flush path. They are interlocks, not a
second decision -- a reset under running work or on unmapped pools corrupts
the rank. A follower that is busy under PP0's "passed" was a refusal before
this change too; it is now printed by name (``UNEINS``).

BYTE-IDENTITY. Off PP (``pp_size <= 1``) nothing is stamped and nothing
changes. On PP, a flush PP0 passes is run on the followers one PP0 pass later
than before (on the verdict, not on the forward); a flush PP0 refuses is no
longer run on the followers at all -- that is the fix. The follower follows the
WIRE, never its own env: an unstamped flush takes the old path.
``FLLIPER_PDFLIP_FLUSH_PP0_VERDICT=0`` (read on PP0) stops the stamping and so
restores the old form everywhere.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List

from flliper.srt.managers.io_struct import FlushCacheReqInput

logger = logging.getLogger(__name__)

ENV = "FLLIPER_PDFLIP_FLUSH_PP0_VERDICT"

#: A follower parks at most this many undecided flushes; past it the oldest is
#: dropped by name (a PP0 that never answers must not grow a follower's heap).
PARK_CAP = 256


@dataclass
class PdFlipFlushVerdict:
    """PP0's decision on the forwarded flush ``seq``; rides the request chain."""

    seq: int
    passed: bool
    detail: str = ""


def env_on(env=None) -> bool:
    e = os.environ if env is None else env
    raw = (e.get(ENV, "") or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _pp(scheduler):
    ps = getattr(scheduler, "ps", None)
    return (int(getattr(ps, "pp_size", 1) or 1), int(getattr(ps, "pp_rank", 0) or 0))


def _state(scheduler) -> dict:
    st = getattr(scheduler, "_pdflip_flush_verdict_state", None)
    if st is None:
        st = {"seq": 0, "out": [], "parked": {}, "drops": 0}
        scheduler._pdflip_flush_verdict_state = st
    return st


def pp0_armed(scheduler) -> bool:
    """PP0 stamps: resolved ONCE per scheduler (a boot constant)."""
    cached = getattr(scheduler, "_pdflip_flush_verdict_armed", None)
    if cached is not None:
        return cached
    pp_size, pp_rank = _pp(scheduler)
    value = bool(pp_size > 1 and pp_rank == 0 and env_on())
    scheduler._pdflip_flush_verdict_armed = value
    if value:
        logger.info(
            "PDFLIP-FLUSH-VERDICT ARMED pp_size=%d: PP0 decides every forwarded "
            "/flush_cache; followers flush only on PP0's passed verdict", pp_size)
    return value


def pp0_wire(scheduler, wire_reqs: List) -> List:
    """PP0, in the forward block: stamp the flushes going out and put the
    verdicts decided since the last pass at the FRONT of the wire."""
    if not pp0_armed(scheduler):
        return wire_reqs
    st = _state(scheduler)
    for r in wire_reqs or ():
        if isinstance(r, FlushCacheReqInput) and r.pdflip_flush_seq is None:
            st["seq"] += 1
            r.pdflip_flush_seq = st["seq"]
    if not st["out"]:
        return wire_reqs
    out, st["out"] = st["out"], []
    return out + list(wire_reqs or ())


def pp0_record(scheduler, recv_req, success: bool, detail: str = "") -> None:
    """PP0, at every terminal outcome of a stamped flush."""
    seq = getattr(recv_req, "pdflip_flush_seq", None)
    if seq is None or not pp0_armed(scheduler):
        return
    _state(scheduler)["out"].append(PdFlipFlushVerdict(seq=int(seq), passed=bool(success), detail=detail))


def follower_park(scheduler, recv_req) -> bool:
    """A follower, at dispatch: a stamped flush waits for PP0's verdict."""
    seq = getattr(recv_req, "pdflip_flush_seq", None)
    if seq is None:
        return False
    pp_size, pp_rank = _pp(scheduler)
    if pp_size <= 1 or pp_rank == 0:
        return False
    st = _state(scheduler)
    parked: Dict[int, object] = st["parked"]
    parked[int(seq)] = recv_req
    while len(parked) > PARK_CAP:
        old = next(iter(parked))
        parked.pop(old)
        logger.warning("PDFLIP-FLUSH-VERDICT park-cap seq=%d dropped undecided (cap=%d)", old, PARK_CAP)
    return True


def follower_absorb(scheduler, recv_reqs: List, apply) -> List:
    """A follower, after the forward: take PP0's verdicts off the list and
    apply each to its parked flush. ``apply(recv_req, passed, detail)`` runs
    the flush (passed) or answers the named refusal (not passed)."""
    if not recv_reqs or not any(isinstance(r, PdFlipFlushVerdict) for r in recv_reqs):
        return recv_reqs
    st = _state(scheduler)
    rest = []
    for r in recv_reqs:
        if not isinstance(r, PdFlipFlushVerdict):
            rest.append(r)
            continue
        req = st["parked"].pop(int(r.seq), None)
        if req is None:
            logger.warning("PDFLIP-FLUSH-VERDICT seq=%d passed=%s for no parked flush on this rank",
                           r.seq, r.passed)
            continue
        if not r.passed:
            st["drops"] += 1
            n = st["drops"]
            if n <= 8 or (n & (n - 1)) == 0:
                logger.info(
                    "PDFLIP-FLUSH-VERDICT dropped seq=%d n=%d: PP0 refused this flush (%s); "
                    "this follower does not flush on its own verdict",
                    r.seq, n, r.detail or "refused")
        apply(req, bool(r.passed), r.detail)
    return rest
