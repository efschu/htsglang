# SPDX-License-Identifier: Apache-2.0
"""RA (23.10., SOLL v2 R11 "Front entscheidet allein, keine rang-lokale Entscheidung"): on a plain TP group the
ADMISSION of a request is ONE group verdict, and the built extend set is checked before anything branches on it.

THE DEATH (27B NVFP4 duo, rc35n4duo5 = desk/27b-nvfp4-duo2-1023 @ 77bdedf7b4, boot ...duo10231145_..._114625, D log
641479-641536 and the WEDGE-STACKs at 12:27:04Z; W17 at 12:33:12Z). Group D is TP2 uneven (5090 + 3080, DCP), pp 1,
no Form A role plan. pdflip-36-127 (a P->D hand-back, N=113072, d_uncached=1) was refused 186 times in 41.3 s, every
time on BOTH ranks by the ``lifetime`` gate of ``PrefillAdder.add_one_req`` with identical numbers
(``D-ADMIT-REFUSED ... gate=lifetime price=117169 budget=114615.8 ... refusals=128`` on TP0 and TP1, 12:26:21Z). At
12:26:33Z the previous turn of the same session (pdflip-36-126, 113408 tokens) finished; its turn-end L2 copies ran
on each rank's copier thread (``#737 drain depth [writing_check]: TP0 ready=1 pending=1`` / ``TP1 ready=2
pending=0`` -- "rank-local; no group agreement is taken here"), the rank-local tree terms of ``rem_total_tokens``
(``available`` + the peel's deliverable evictable, ``cap_above``) moved apart, and in the very next pass TP0's gate
admitted (``#988 LOADBACK``, ``D-ADMIT-AFTER-REFUSALS refusals=186``, ``#969 EXTENT``) while TP1's refused again.
Off a Form A group nothing makes that verdict the group's: ``add_one_req`` returns the first failing LOCAL gate
(``if _gate is not None and _fa_follow is None: return``). So:

* TP1 (refused) took the pass's D-HOL head (``hol_overtake.HolPass.may_overtake``) and, after the loop, entered PW-R's
  group MIN (``d_wall_head.pick_unservable_head`` -> ``Scheduler._pdflip_group_min_flags`` -> all_reduce MIN);
* TP0 (admitted) had no HOL head, skipped PW-R and entered ``prepare_for_extend`` ->
  ``dcp/prefix_lens_check.assert_prefix_lens_rank_uniform`` -> all_reduce.

Two different collectives, both ranks blocked (FORWARD-HANG forward_ct 11140 from 12:26:34Z, queued=0 on TP0 against
queued=1 on TP1). The S35 note (``pdflip/form_a_post_admit``) said "off a Form A group the budgets are the group's
already (``_update_uniform_pool_budget``)" -- only the FLOOR term is; ``rem_total_tokens`` is
``min(local, floor - offset)`` and the local term (available + deliverable evictable, rank-local tree) can bind on one
rank alone.

THE RULE.

1. GATE: on a plain TP group (``applies``: tp > 1, pp 1, no DP attention, no Form A follow -- that one has H105) every
   rank's gates are REAL (each rank owns its own KV share), so ``add_one_req`` does not return on a local refusal:
   after the in-lock re-check, BEFORE the load-back, every rank gathers ``("gate", rid, code, price, budget)`` over
   the TP cpu group (one all_gather_object per gate call, the same call on every rank). The group code is the MIN:
   ADMIT only when every rank admits, else the first refusing rank's code in TP order -- so no rank is ever made to
   admit against its own pool, and a rank whose own gate admitted takes the group's refusal. A gather for another
   rid, or one that is not a gate verdict, is a named stop (``RankAgreeSplit``).
2. POST-ADMIT: an admitted request never reads NO_TOKEN from this rank's own ``budget_state()`` (the S35 rule, now on
   this group too): whether the NEXT request fits is that request's own group verdict.
3. RIEGEL: after the loop -- before PW-R, before the queue removal, before ``prepare_for_extend`` -- every rank gathers
   ``("set", admitted rids in order, HOL head, gate calls)``. Different sets or different HOL heads (a rank-local exit
   after the gate: load-back WAIT, AgreedRoomShort, chain mismatch) are a named stop; a rank at the riegel while a
   peer is at a gate (the loops made different numbers of gate calls) pairs its gather with the peer's and is a
   named stop on both, instead of two ranks parked in two different collectives.

Every collective here is the same kind (``all_gather_object`` on the TP cpu group, under the blocked-recv stamp), so a
count mismatch PAIRS with a peer's gather and is named by payload, never left as a hang. Ranks nie uneins: a split is a
crash-stop by name. ``FLLIPER_TP_ADMISSION_AGREE=0`` switches it off (default ON)."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, List, Optional, Sequence

logger = logging.getLogger(__name__)

ENV = "FLLIPER_TP_ADMISSION_AGREE"
MARK = "R11 TP-ADMISSION"
ADMIT = "ADMIT"
SITE_GATE = "tp-admission/tp<->gate"
SITE_SET = "tp-admission/tp<->extend-set"

_STATS = {"gate": 0, "follow": 0, "riegel": 0}


class RankAgreeSplit(RuntimeError):
    """The ranks of one TP group are about to take different admission branches: a named stop (raenge-nie-uneins)."""


def enabled(env=None) -> bool:
    e = os.environ if env is None else env
    return (e.get(ENV, "1") or "1").strip().lower() not in ("0", "false", "no", "off")


def applies(sched: Any, env=None) -> bool:
    """Group-uniform: the same boot config on every rank of the group. True on a plain TP group (tp > 1, pp 1) without
    DP attention and without the Form A follow (H105 already makes the verdict the group's there)."""
    if not enabled(env):
        return False
    ps = getattr(sched, "ps", None)
    if ps is None or int(getattr(ps, "tp_size", 1) or 1) <= 1 or int(getattr(ps, "pp_size", 1) or 1) > 1:
        return False
    sa = getattr(sched, "server_args", None)
    if sa is not None and bool(getattr(sa, "enable_dp_attention", False)):
        return False
    is_fa = getattr(sched, "_form_a_is_host", None)
    if callable(is_fa) and is_fa() is not None:
        return False
    return True


def _code(gate: Any) -> str:
    return ADMIT if gate is None else str(getattr(gate, "name", gate))


def gate_verdict(rid: str, gate: Any, *, gather: Callable[[tuple], list], tp_rank: int = 0,
                 price: Any = None, budget: Any = None, sink: Optional[dict] = None) -> str:
    """The group's code for ``rid``: ADMIT when every rank's gate admitted, else the first refusing rank's code in TP
    order. ``gate`` = this rank's first failing gate (an ``AddReqResult``) or None. ``sink`` receives the deciding
    rank's ``tp``/``price``/``budget``."""
    rid = str(rid)
    local = _code(gate)
    got = gather(("gate", rid, local, price, budget))
    if not isinstance(got, list) or not got or any(not isinstance(v, tuple) or len(v) < 2 for v in got):
        raise RankAgreeSplit(
            f"{MARK} MALFORMED rid={rid} tp={tp_rank} got={str(got)[:160]}: this rank is at the admission gate but "
            "the group's gather carries no verdict from every rank -- stopping instead of admitting on a guess "
            "(raenge-nie-uneins)."
        )
    tags = [str(v[0]) for v in got]
    if any(t != "gate" for t in tags):
        raise RankAgreeSplit(
            f"{MARK} CALL-COUNT SPLIT rid={rid} tp={tp_rank} peers={tags}: this rank is at the admission gate while "
            "a peer is at the post-loop riegel -- the ranks' admission loops made different numbers of gate calls "
            "(a rank-local skip or break before add_one_req); stopping by name instead of two ranks in two "
            "collectives (raenge-nie-uneins)."
        )
    rids = [str(v[1]) for v in got]
    if any(r != rid for r in rids):
        raise RankAgreeSplit(
            f"{MARK} RID SPLIT tp={tp_rank} local_rid={rid} group_rids={rids}: the ranks reached the admission gate "
            "for different requests -- the queues or the loop's skips diverged; stopping by name instead of "
            "building different extend sets (raenge-nie-uneins)."
        )
    code, decider = ADMIT, None
    for i, v in enumerate(got):
        c = str(v[2]) if len(v) > 2 else ADMIT
        if c != ADMIT:
            code, decider = c, i
            break
    _STATS["gate"] += 1
    if sink is not None:
        src = got[decider] if decider is not None else got[0]
        sink["tp"] = decider
        sink["price"] = src[3] if len(src) > 3 else None
        sink["budget"] = src[4] if len(src) > 4 else None
    if code != local:
        _STATS["follow"] += 1
        n = _STATS["follow"]
        if n <= 20 or (n & (n - 1)) == 0 or n % 256 == 0:
            d = got[decider] if decider is not None else got[0]
            logger.warning(
                "%s FOLLOW rid=%s tp=%d local=%s group=%s decider=tp%s price=%s budget=%s local_price=%s "
                "local_budget=%s (n=%d): this rank's own gate said %s, a peer's pool refused -- the group verdict "
                "is the MIN, every rank refuses alike (SOLL R11)",
                MARK, rid, int(tp_rank), local, code, decider, d[3] if len(d) > 3 else None,
                d[4] if len(d) > 4 else None, price, budget, n, local,
            )
    return code


def extend_set_riegel(admitted: Sequence[Any], hol_head: Optional[str], gate_calls: int, *,
                      gather: Callable[[tuple], list], tp_rank: int = 0) -> None:
    """After the admission loop: every rank's admitted rids (in order) and HOL head must be the group's. Raises
    :class:`RankAgreeSplit` on any difference; returns None when the ranks agree."""
    rids = tuple(str(getattr(r, "rid", r)) for r in admitted)
    head = None if hol_head is None else str(hol_head)
    got = gather(("set", rids, head, int(gate_calls)))
    if not isinstance(got, list) or not got or any(not isinstance(v, tuple) or len(v) < 1 for v in got):
        raise RankAgreeSplit(
            f"{MARK} RIEGEL MALFORMED tp={tp_rank} got={str(got)[:160]}: stopping by name (raenge-nie-uneins)."
        )
    tags = [str(v[0]) for v in got]
    if any(t != "set" for t in tags):
        raise RankAgreeSplit(
            f"{MARK} CALL-COUNT SPLIT tp={tp_rank} at=riegel peers={tags} gate_calls={int(gate_calls)} "
            f"admitted={list(rids)}: this rank finished its admission loop while a peer is still at a gate -- the "
            "loops made different numbers of gate calls; stopping by name instead of two ranks in two collectives "
            "(raenge-nie-uneins)."
        )
    sets = [tuple(v[1]) if len(v) > 1 else () for v in got]
    heads = [v[2] if len(v) > 2 else None for v in got]
    calls = [v[3] if len(v) > 3 else None for v in got]
    _STATS["riegel"] += 1
    if any(s != sets[0] for s in sets) or any(h != heads[0] for h in heads):
        raise RankAgreeSplit(
            f"{MARK} EXTEND-SET SPLIT tp={tp_rank} sets={[list(s) for s in sets]} hol_heads={heads} "
            f"gate_calls={calls}: the ranks built different extend sets or hold different queue heads after an "
            "agreed gate (a rank-local exit behind it: load-back wait, room short, chain mismatch) -- the next "
            "collectives (PW-R's group MIN, the DCP prefix check, the forward) would differ per rank; stopping by "
            "name instead (raenge-nie-uneins, SOLL R11)."
        )


def uniform_result(res: Any, added: bool, agree: Any) -> Any:
    """S35 on a plain TP group: NO_TOKEN of an ADDED request -> CONTINUE (``agree`` = the adder's tp_admission_agree
    callable, None off this group). Everything else unchanged."""
    if agree is None or not added or getattr(res, "name", None) != "NO_TOKEN":
        return res
    return type(res).CONTINUE


def make_agree(sched: Any) -> Optional[Callable]:
    """The callable ``PrefillAdder.add_one_req`` hands its gate verdict to (None off a plain TP group). Built per pass
    with the adder; ``calls`` counts its gate calls for the riegel."""
    if not applies(sched):
        return None
    tp_rank = int(getattr(getattr(sched, "tp_group", None), "rank", 0) or 0)

    def _gather(payload):
        return sched._form_a_tp_gather(SITE_GATE, payload)

    def _agree(req, gate, price=None, budget=None):
        _agree.calls += 1
        sink: dict = {}
        code = gate_verdict(str(req.rid), gate, gather=_gather, tp_rank=tp_rank, price=price, budget=budget,
                            sink=sink)
        _agree.last_verdict = (str(req.rid), code, sink.get("price"), sink.get("budget"))
        return code

    _agree.calls = 0
    _agree.last_verdict = None
    _agree.tp_rank = tp_rank
    return _agree


def riegel(sched: Any, agree: Any, admitted: List[Any], hol_head: Optional[str]) -> None:
    """The post-loop riegel for the pass whose adder carries ``agree`` (no-op when None)."""
    if agree is None:
        return
    extend_set_riegel(
        admitted, hol_head, int(getattr(agree, "calls", 0) or 0),
        gather=lambda p: sched._form_a_tp_gather(SITE_SET, p),
        tp_rank=int(getattr(agree, "tp_rank", 0) or 0),
    )
