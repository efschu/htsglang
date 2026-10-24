# SPDX-License-Identifier: Apache-2.0
"""R2-KURZ-CHUNK (27B 24.10., SCHEDULER-SOLL v2 R2 / R5 / R7; R6k untouched): a kurz request gets its share of the pass
chunk BEFORE the carried continuation takes it.

THE SPECIMEN (27B int4ar boot weg2..._1023_223241, D, TP0; X=4096, halbX=2048 from the front)
---------------------------------------------------------------------------------------------
The corridor actuator narrowed D's pass chunk from 4096 to 256 tokens (``#794 GROUP-NARROWED ... to 256``). A request
longer than one chunk then stays D's ``chunked_req`` for several passes, and ``Scheduler.get_new_batch_prefill`` adds that
continuation through ``PrefillAdder.add_chunked_req`` BEFORE the waiting-queue loop runs: it takes
``min(rem_chunk_tokens, rem_total_tokens)`` = the whole 256. Every kurz request behind it reads ``rem_chunk=0``, does not
fit whole, may not become a second continuation, and is refused ``gate=trunc_len_zero`` pass after pass:

    22:38:06 PDFLIP D-ADMIT-REFUSED rid=pdflip-2-22 gate=trunc_len_zero rem_chunk=0 uncached=188 limit=2047
    22:38:06 #969 EXTENT fwd=1 reqs=[('pdflip-2-21', 65249, 65505, 65249, 256)]     <- the continuation (kurz u=1265)
    22:56:48 ... rid=pdflip-20-190 gate=trunc_len_zero rem_chunk=0 uncached=26     <- 26 tokens waited 1.3 s

48 TP0 refusals for 18 kurz rids in that boot (u 26..908), each waiting 0.4..4.3 s (``D-ADMIT-AFTER-REFUSALS``); the
holder was a kurz continuation (pdflip-2-21 u=1265, 4-57 u=1807, 4-61, 16-136, 20-162 u=1488) or a halb after its timer
(R3 D_AFTER_TIMER: 2-23 u=3805, 14-122, 24-214). D had the seat (the front granted D_NOW, R5) and the KV (decider budget
>= 42000); only the pass chunk was spent. R2: "sofort auf D prefillt, ohne Wartezeit"; R7: "kann D einen Request unter X
nicht nehmen, ist das ein Defekt".

THE RULE (Sarathi-style: the small work first into the token budget)
-------------------------------------------------------------------
Around the continuation's ``add_chunked_req`` (the ``lanes_p.chunk_scope`` seam), the chunk budget is lowered by the
needs of the waiting kurz requests that fit WHOLE next to it; the continuation is then charged only what it really took,
so the waiting loop finds exactly those tokens left and admits the kurz requests in the same pass.

* kurz = the FRONT's R2 class (SOLL R11: no rank-local class): the request carries the front's halbX (``weg2_front_halfx``)
  and its first-admission price ``weg2_front_priced`` below it (a MEASURE barrier carries price 0: its class is D's exact
  extent below halbX, the barrier's own rule). A halb/groß request is never reserved for.
* need = D's own GROUP-priced uncached extent from its last X gate (``d_admit_refusal.XGATE_ATTR``, MIN-reduced over the
  group, set on every rank in the same pass) rounded up to a page, plus one page of slack (end anchor / page rounding).
  No X-gate reading yet = no reservation this pass (the request is offered as before; after its first chunk-bound refusal
  it carries the reading). Never the front's price as the size: the price can be 0 (barrier) or a different depth.
* the continuation keeps at least ``max(page_size, chunk grain)`` tokens: it always progresses (liveness; a kurz stream
  cannot stall a running halb/kurz prefill). Kurz requests that do not fit next to that floor are skipped (they need a
  chunk of their own and wait for the continuation as before -- named ``R2-KURZ-CHUNK-NOFIT``).
* a kurz request is reserved for at most ``MAX_TRIES`` passes: a reservation that did not get it admitted (another gate:
  lifetime, loadback room, seat) stops costing the continuation after that. Admission outcomes are group-agreed, so the
  count is the same on every rank.
* group D only (``tp_prefill_max_tokens > 0``); never on a rank that executes a forwarded schedule
  (``adder.scheduled_extents``); no reservation when the switch is off.

GROUP UNIFORMITY (R11): the inputs are the replicated waiting queue, the broadcast headers, the group-priced X-gate
extent, the group-narrowed chunk and launch constants; and ``add_chunked_req`` MIN-reduces the continuation's length over
the group (CHUNKGEOM ``tp_admission_agree.chunk_min``) anyway, so the charge restored here is the group's on every rank.
No new collective. A request that still does not fit is refused by name (``second_continuation_chunked`` /
``trunc_len_zero``), never minted as a second continuation (#959/#996 guard unchanged).

R6k: nothing moves off D; a running kurz continuation finishes on D, only shares the pass with a younger kurz.
R1a/R4: flips are not touched. Switch ``FLLIPER_PDFLIP_R2_KURZ_CHUNK_RESERVE`` (default ON, user order 14.10.).
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

MARK = "R2-KURZ-CHUNK"
ENV = "FLLIPER_PDFLIP_R2_KURZ_CHUNK_RESERVE"
#: Req attribute: how many passes this request was reserved for (group-agreed: admission outcomes are)
ATTR_TRIES = "_pdflip_r2kurz_reserve_n"
MAX_TRIES = 3
_COUNTS = {"reserve": 0, "nofit": 0}


def enabled() -> bool:
    """Default ON (user order 14.10.: new features on); ``0/false/off/no`` = the old continuation-first budget."""
    return (os.environ.get(ENV, "1") or "1").strip().lower() not in ("0", "false", "off", "no")


def _ceil_page(n: int, page: int) -> int:
    return -(-int(n) // page) * page


def kurz_need(req: Any, page_size: int) -> Optional[int]:
    """Chunk tokens a waiting request needs to be admitted WHOLE this pass, or ``None`` when it is not a kurz request
    by the front's R2 verdict or D has not priced it yet. Never raises."""
    try:
        from flliper.srt.pdflip import d_admit_refusal, front_x

        fx, priced = front_x.request_front_x(req)  # first admission only (no output yet)
        if fx is None:
            return None
        hx = getattr(req, front_x.ATTR_HALFX, None)
        if hx is None or int(hx) <= 0:
            return None
        xg = getattr(req, d_admit_refusal.XGATE_ATTR, None)
        if not xg:
            return None
        uncached = int(xg[0])
        if uncached <= 0:
            return None
        hx = int(hx)
        # the front's class: priced < halbX; a MEASURE barrier (price 0) is classed by D's exact extent (< halbX)
        if priced is not None and int(priced) > 0:
            if int(priced) >= hx:
                return None
        elif uncached >= hx:
            return None
        page = max(1, int(page_size or 1))
        return _ceil_page(uncached, page) + page
    except Exception:  # noqa: BLE001 -- unreadable = not kurz = no reservation
        return None


@dataclass
class Plan:
    reserve: int = 0
    floor: int = 0
    rids: List[Tuple[str, int]] = field(default_factory=list)
    nofit: List[Tuple[str, int]] = field(default_factory=list)


def plan_reserve(queue: Iterable[Any], rem_chunk: int, cont_floor: int, page_size: int) -> Plan:
    """Pure: greedy in queue order, the kurz needs that fit next to the continuation's floor."""
    plan = Plan(floor=int(cont_floor))
    room = int(rem_chunk) - int(cont_floor)
    if room <= 0:
        return plan
    for req in queue or ():
        need = kurz_need(req, page_size)
        if need is None:
            continue
        rid = str(getattr(req, "rid", "?"))
        if int(getattr(req, ATTR_TRIES, 0) or 0) >= MAX_TRIES:
            continue
        if plan.reserve + need <= room:
            plan.reserve += need
            plan.rids.append((rid, need))
        else:
            plan.nofit.append((rid, need))
    return plan


def _group_d(sched: Any) -> bool:
    try:
        return int(getattr(getattr(sched, "server_args", None), "tp_prefill_max_tokens", 0) or 0) > 0
    except (TypeError, ValueError):
        return False


def _cont_floor(adder: Any) -> int:
    page = max(1, int(getattr(adder, "page_size", 1) or 1))
    grain = 1
    g = getattr(adder, "_pdflip_chunk_grain", None)
    if callable(g):
        try:
            grain = max(1, int(g()))
        except Exception:  # noqa: BLE001
            grain = 1
    return max(page, grain)


def plan_for_pass(adder: Any, sched: Any, cont: Any) -> Optional[Plan]:
    """The reservation of this pass for the continuation ``cont``, or ``None`` (nothing to do / not applicable)."""
    if cont is None or not enabled() or not _group_d(sched):
        return None
    rem = getattr(adder, "rem_chunk_tokens", None)
    if rem is None or getattr(adder, "scheduled_extents", None):
        return None
    queue = list(getattr(sched, "waiting_queue", None) or ())
    if not queue:
        return None
    return plan_reserve(queue, int(rem), _cont_floor(adder), int(getattr(adder, "page_size", 1) or 1))


@contextmanager
def reserve_scope(adder: Any, sched: Any, cont: Any):
    """Lower the chunk budget for the continuation by the kurz reservation; afterwards charge only what it took."""
    try:
        plan = plan_for_pass(adder, sched, cont)
    except Exception:  # noqa: BLE001 -- a reservation may never break admission
        plan = None
    if plan is not None and plan.nofit and not plan.rids:
        _COUNTS["nofit"] += 1
        n = _COUNTS["nofit"]
        if n <= 8 or n % 64 == 0:
            logger.info(
                "PDFLIP %s-NOFIT cont=%s chunk=%s floor=%d waiting_kurz=%s occurrence=%d soll=R2 -- the kurz request "
                "needs a chunk of its own and waits for the continuation (one continuation at a time)",
                MARK, str(getattr(cont, "rid", "?")), getattr(adder, "rem_chunk_tokens", None), plan.floor,
                ",".join(f"{r}:{n_}" for r, n_ in plan.nofit[:6]), n,
            )
    if plan is None or plan.reserve <= 0:
        yield
        return
    before = int(adder.rem_chunk_tokens)
    lowered = before - plan.reserve
    adder.rem_chunk_tokens = lowered
    try:
        yield
    finally:
        taken = lowered - int(adder.rem_chunk_tokens)
        adder.rem_chunk_tokens = before - taken
        by_rid = {str(getattr(r, "rid", "")): r for r in (getattr(sched, "waiting_queue", None) or ())}
        for rid, _need in plan.rids:
            r = by_rid.get(rid)
            if r is not None:
                try:
                    setattr(r, ATTR_TRIES, int(getattr(r, ATTR_TRIES, 0) or 0) + 1)
                except Exception:  # noqa: BLE001 -- a slotted stand-in
                    pass
        _COUNTS["reserve"] += 1
        n = _COUNTS["reserve"]
        if n <= 8 or n % 64 == 0:
            logger.info(
                "PDFLIP %s-RESERVE cont=%s chunk=%d reserve=%d cont_took=%d left=%d kurz=%s nofit=%s occurrence=%d "
                "soll=R2/R7 -- the kurz requests get their share of this pass before the continuation (R6k: nothing "
                "leaves D)",
                MARK, str(getattr(cont, "rid", "?")), before, plan.reserve, taken, int(adder.rem_chunk_tokens),
                ",".join(f"{r}:{n_}" for r, n_ in plan.rids[:6]),
                ",".join(f"{r}:{n_}" for r, n_ in plan.nofit[:6]) or "-", n,
            )
