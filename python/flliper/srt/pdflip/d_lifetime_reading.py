"""KVOFFSET-1024: D publishes the terms of its lifetime gate in the same KV reading the front decides on.

SCHEDULER-SOLL v2 R11 ("Front entscheidet allein"): the front forms the hand-off verdict; D delivers NUMBERS only.
R5 (seat = seat + KV) and R7 (a request under X that D does not take is a defect): a hand-off the front lets through
must be one D's admission takes.

THE GAP (27B int4ar ``_1023_223241`` / ``_1024_003732``, report front-handover-kvfits-1024, review S2). The front's KV
test (``arrival_seat_rule.kv_fits``) read ``available + evictable`` from ``/server_info`` ``pdflip_kv``. D's lifetime
gate (``PrefillAdder.add_one_req``: ``total_tokens >= rem_total_tokens`` refuses) reads

    min(available + DELIVERABLE evictable - offset - dcp_deficit, fundable_floor - offset) - ledger

with ``offset`` = the decode reserve of every running request (``min(max_new - out, CLIP) * new_token_ratio``, plus the
mixed decode rows). Two optimistic terms, both in D and both invisible to the front:

* the decode reserve: pdflip-25-238 / 26-256 / 37-295 / 9-62 pass the front's test and are refused only by ``offset``
  (25-238: 87 refusals in 23.4 s);
* the reported instead of the deliverable evictable count on the hybrid-SSM 27B (``rem_total_tokens``' ED branch).

THE READING (``lifetime_reading``, merged into ``pdflip_kv`` by ``Scheduler.get_internal_state``):

``lifetime_budget``
    the LARGEST price D's gate admits right now (``ceil(rem_total_tokens) - 1``: the gate refuses at ``price >=
    budget``), computed by ``schedule_policy.lifetime_budget`` from the adder's own terms
    (``lifetime_available_and_evictable``, ``running_decode_reserve``, the published group floor, the uneven-DCP
    deficit, the commitment ledger). Nothing is added: no term D does not subtract itself.
``price_extra``
    what the gate adds to a new request's ``extend + decode clip``: one page plus the shared-gap Mamba cost.
``decode_reserve`` / ``decode_reserve_by_rid``
    the reserve term and its split over the running decodes (a displaced decode gives its share back --
    ``PrefillAdder.released_by_leaving``; the front's AGE PLAN credits it to the victim).
``admitted_rids``
    requests D has admitted whose rows are already in the reading but which are not in ``running_rids`` yet (the
    extend batch the merge has not taken, the chunked continuation): the front's burst debit must not charge them
    a second time.
``ledger_outstanding`` / ``ledger_rids``
    the chunked commitments the budget already subtracts.

RANK UNITY. ``new_token_ratio`` is RANK-LOCAL (NF y6h 10011531: TP0 priced a reservation at 4964, TP1/TP2 at 6592), and
each rank's gate subtracts its own reserve; the group admits only when every rank admits (``tp_admission_agree``), so
the binding reserve is the LARGEST. The ratio therefore rides the per-iteration MIN reduce
(``Scheduler._update_uniform_pool_budget``, one negated slot: group MAX) and the running decodes -- the same list on
every D rank -- are priced with ``max(group ratio, TP0's current ratio)``. ``decode_reserve_scope`` names which ratio
priced it (``group``; ``tp0`` where no reduce runs: the kv-session-offload path, a named gap). The pool terms
(available, deliverable evictable) stay TP0's, as ``available`` / ``evictable`` always were (S94 A1); under uneven DCP
the published deficit pins TP0's budget to the binding rank's.

Observation only: nothing on D decides on these numbers; a failing read leaves the fields out (the front then decides
as before -- an old D)."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

#: the ratio rides an int64 MIN reduce in parts per million (ceil: the gate's direction)
RATIO_SCALE = 1_000_000


def ratio_ppm(ratio) -> int:
    """``new_token_ratio`` as an int for the MIN reduce, rounded UP (a larger reserve is the refusing direction)."""
    try:
        return int(math.ceil(max(0.0, float(ratio)) * RATIO_SCALE))
    except (TypeError, ValueError):
        return 0


def ratio_from_ppm(v) -> float:
    return float(int(v)) / RATIO_SCALE


def local_ratio(sched) -> float:
    """This rank's current ``new_token_ratio`` (0.0 for a scheduler stand-in without a tracker)."""
    tracker = getattr(sched, "new_token_ratio_tracker", None)
    try:
        return max(0.0, float(getattr(tracker, "current", 0.0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _reserve_of(req, ratio: float) -> float:
    """``schedule_policy.running_request_reserve``; 0 for a request double without sampling params."""
    from flliper.srt.managers import schedule_policy as _sp

    if getattr(req, "sampling_params", None) is None:
        return 0.0
    return _sp.running_request_reserve(req, ratio)


def _reserve_sum(reqs, ratio: float, mixed: int) -> float:
    return mixed + sum(_reserve_of(r, ratio) for r in reqs)


def _form_a_token_cut() -> bool:
    try:
        from flliper.srt.rank_role import form_a_token_cut_active

        return bool(form_a_token_cut_active())
    except Exception:  # noqa: BLE001
        return False


def _form_a_host(sched) -> bool:
    """True on the Form A attention host (TP0, the publisher) whose gate verdict every rank adopts (H105)."""
    fn = getattr(sched, "_form_a_is_host", None)
    if not callable(fn):
        return False
    try:
        return fn() is True
    except Exception:  # noqa: BLE001 -- a split is the scheduler's own named stop, never this instrument's
        return False


def reserve_ratio(sched) -> Tuple[float, str]:
    """``(ratio, scope)``: the ratio the published reserve is priced with -- the group MAX of the last reduce, never
    below this rank's current one; ``tp0`` when no reduce published a group value (kv-session-offload path).
    Review S5 (NF Form A, H105): the attention host's gate verdict is the group's -- the workers' ratios never decide --
    so the host publishes ITS OWN ratio (``host``); under the token cut every rank's gate counts again (MIN gather):
    the group MAX."""
    local = local_ratio(sched)
    if _form_a_host(sched) and not _form_a_token_cut():
        return local, "host"
    group = getattr(sched, "_kvoffset_group_ratio", None)
    if group is None:
        return local, "tp0"
    return max(float(group), local), "group"


def _live(reqs):
    out = []
    for r in list(reqs or ()):
        fin = getattr(r, "finished", None)
        if callable(fin) and fin():
            continue
        out.append(r)
    return out


def lifetime_reading(sched) -> Dict[str, Any]:
    """The fields above, from THIS scheduler (TP0 when served on ``/server_info``). Raises on a missing pool -- the
    caller (``get_internal_state``) catches, an instrument never breaks the info route."""
    from flliper.srt.managers import schedule_policy as _sp
    from flliper.srt.mem_cache.common import cap_blind_gap_published, published_fundable_floor
    from flliper.srt.pdflip.d_seat_vram import _unmerged_extend

    alloc = sched.token_to_kv_pool_allocator
    tree = sched.tree_cache
    avail, evict = _sp.lifetime_available_and_evictable(alloc, tree)
    ratio, scope = reserve_ratio(sched)
    running = _live(getattr(getattr(sched, "running_batch", None), "reqs", None))
    # the extend batch that ran last is merged into the running batch before the next admission builds its adder:
    # its requests are in that adder's offset (and their rows already in ``available``)
    unmerged = _unmerged_extend(sched, running)
    decoding = running + list(unmerged)
    mixed = len(decoding) if getattr(sched, "is_mixed_chunk", False) else 0
    reserve = _reserve_sum(decoding, ratio, mixed)
    by_rid = {str(r.rid): int(math.ceil(_reserve_of(r, ratio))) for r in decoding}
    admitted = [str(r.rid) for r in unmerged]
    chunked = getattr(sched, "chunked_req", None)
    if chunked is not None and not (callable(getattr(chunked, "finished", None)) and chunked.finished()):
        if all(chunked is not r for r in decoding):
            admitted.append(str(chunked.rid))
    ledger = getattr(sched, "chunked_commitment_ledger", None)
    ledger_out = int(ledger.outstanding_tokens()) if ledger is not None else 0
    ledger_rids = list(ledger.request_ids()) if ledger is not None and hasattr(ledger, "request_ids") else []
    floor = published_fundable_floor(tree)
    dcp = 0
    ubd = getattr(sched, "uniform_budget_deficit", None)
    if callable(ubd):
        dcp = int(ubd() or 0)
    kvso = getattr(sched, "kv_session_offload", None)
    if kvso is not None and callable(getattr(kvso, "dcp_budget_deficit", None)):
        dcp = int(kvso.dcp_budget_deficit() or 0)
    budget = _sp.lifetime_budget(avail, evict, reserve, dcp, floor, ledger_out)
    page = max(1, int(getattr(sched, "page_size", 1) or 1))
    return {
        # the gate refuses at price >= budget: the largest price it admits
        "lifetime_budget": int(math.ceil(budget)) - 1,
        "price_extra": page + int(_sp.mamba_slot_full_token_cost(alloc)),
        "lifetime_available": int(avail),
        "deliverable_evictable": int(evict),
        "decode_reserve": int(math.ceil(reserve)),
        "decode_reserve_ratio": round(float(ratio), 6),
        "decode_reserve_scope": scope,
        "decode_reserve_by_rid": dict(list(by_rid.items())[:64]),
        "admitted_rids": admitted[:64],
        "ledger_outstanding": ledger_out,
        "ledger_rids": [str(r) for r in ledger_rids][:64],
        "fundable_floor": None if floor is None else int(floor),
        "dcp_deficit": int(dcp),
        # review S1: D's gate lifts the lifetime budget by this group gap for a request whose IMMEDIATE demand fits
        # (``PrefillAdder._cap_lifetime_waiver``, an engaged residency cap); 0 without one
        "cap_blind_gap": int(cap_blind_gap_published(tree)),
    }


def resume_budget(sched, running_reqs) -> Optional[int]:
    """THIS rank's lifetime budget for the pressure-park resume test (``d_park_runtime._resume_fit_reading``, which
    takes the group MIN): ``available + deliverable evictable - this rank's own offset - dcp deficit - ledger``,
    capped by the published floor -- the adder this pass will build, before any admission of the pass. Own ratio, not
    the group's: the MIN over the ranks' own budgets IS the group verdict the gate takes (every rank must admit).

    NEVER RAISES (review A2, S75 2.1 -- "every path enters the group MIN exactly once"): the caller sends the result
    into a gloo MIN its peers enter too; a raise on ANY term (pool, dcp deficit, ledger, a request's reserve, the
    floor) would leave this rank out of the collective and its peers hung (W17). Any failure is None -- the caller's
    sentinel (ok=0), and the group refuses the early resume on every rank alike."""
    try:
        from flliper.srt.managers import schedule_policy as _sp
        from flliper.srt.mem_cache.common import published_fundable_floor

        alloc = sched.token_to_kv_pool_allocator
        tree = sched.tree_cache
        avail, evict = _sp.lifetime_available_and_evictable(alloc, tree)
        ratio = local_ratio(sched)
        running = _live(running_reqs)
        mixed = len(running) if getattr(sched, "is_mixed_chunk", False) else 0
        reserve = _reserve_sum(running, ratio, mixed)
        ledger = getattr(sched, "chunked_commitment_ledger", None)
        ledger_out = int(ledger.outstanding_tokens()) if ledger is not None else 0
        dcp = 0
        ubd = getattr(sched, "uniform_budget_deficit", None)
        if callable(ubd):
            dcp = int(ubd() or 0)
        floor = published_fundable_floor(tree)
        budget = _sp.lifetime_budget(avail, evict, reserve, dcp, floor, ledger_out)
        return int(math.ceil(budget)) - 1
    except Exception:  # noqa: BLE001 -- the sentinel, never a skipped collective
        return None


def resume_price(req, sched) -> int:
    """The price the gate will charge a resumed parked request: its whole context as the extend (a parked span is
    re-matched at admission; the resume test assumes none of it is still on the device, as before -- the
    ``origin + output`` it measured) plus the decode clip, one page and the Mamba gap for a request without a state
    (``schedule_policy.lifetime_price``)."""
    from flliper.srt.managers import schedule_policy as _sp

    ctx = len(getattr(req, "origin_input_ids", None) or ()) + len(getattr(req, "output_ids", None) or ())
    page = max(1, int(getattr(sched, "page_size", 1) or 1))
    gap = 0
    if getattr(req, "mamba_pool_idx", None) is None:
        try:
            gap = _sp.mamba_slot_full_token_cost(sched.token_to_kv_pool_allocator)
        except Exception:  # noqa: BLE001
            gap = 0
    if getattr(req, "sampling_params", None) is None:
        return ctx + page + int(gap)          # a request double without sampling params: no decode term
    return _sp.lifetime_price(ctx, req, page, gap)
