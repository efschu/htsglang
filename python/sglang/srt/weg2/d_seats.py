"""H91 Teil B (Nutzer-Design 25.09., Stufe 1): group D decodes with SEATS > 1,
and a request that has to leave the device PARKS instead of being thrown away.

Two ways a running D request parks, one write path and one read path for both:

* ``pressure`` -- the unified 262,144-token KV pool is shared by every seat.
  When the seats' contexts together outgrow it, decode-OOM retracts; on group
  D that retraction already RETAINS the computed span in the tree
  (``weg2/retract_retain.py``: KV rows, the node's GDN/Mamba checkpoint, the
  draft rows carried with the KV).  The span is evictable, so the older
  request's growth demotes it to the host tier through the ordinary HiCache
  write-back (L2 = the shared arena, L3 = the storage backend).  This module
  decides WHO parks (the YOUNGEST, by the kv-session-offload protection key --
  spill class, fast lane, then FCFS) and WHEN it comes back (when no older
  request is live any more; an optional hysteresis for the later dynamic-seat
  form).  The resume is the ordinary re-admission: prefix match, ``load_back``
  from host/storage, Mamba anchor from the node, one extend of the last token
  -- it continues at exactly the position it left.
* ``flip`` -- before group D sleeps (D->P flip) the front calls
  ``POST /weg2/park_running``.  Every running request is retracted the same
  way, stamped for a forced host write-through (the tree is flushed at the
  sleep, the store is the only surviving copy -- the #969D/#1068 carrier), and
  kept aside.  At the sleep's dormant point they enter the #1443 dormant hold
  FIRST, oldest first, with their storage prefetch issued during the flip; the
  wake releases them ahead of every request that arrived while D slept.

What this deliberately is NOT: kv-session-offload's own machinery.  kvso keeps
a spilled session DECODING from a separate pinned host pool (its spill tick);
the user design parks, and wants no second pinned pool.  kvso's POLICY pieces
are reused as they are (``session_priority_key`` for the victim,
``RestoreHysteresis`` for the optional early resume); its manager stays off.

Ranks never disagree: every input here is replicated scheduler state
(``kv_arrival_seq``, the park marks set by group-uniform events, the
group-reduced available count), so every rank computes the same order and the
same gate without a collective.

Stage 2 (H95, Nutzer-Design 25.09.): the seats are DYNAMIC, n = 1..--d-bs
per D phase.  The front knows n at the P->D flip (``handoff_n`` + ``parked_n``
on the kv_cache resume, H91c rule 2); :func:`phase_seats` turns that into the
phase's seat count on every rank from the SAME replicated request -- no
collective, no rank-local input.  Nothing here assumes two seats -- the order,
the gate and the park are all functions of the live set; bs2 (H91b) is n = 2.

What a seat costs and where (the per-seat posts; H95 report, file:line in
the commit): the GDN/Mamba state slots (``mamba_slots_for_seats``), the
speculative verify state (ReplaySSM spec ring rows ``spec_state_size + 1``),
one decode CUDA graph per batch size 1..--d-bs, and -- before H95 -- expert
pool scratch growing with n (``min(n x 4 x 10, E - R)``).  H95 A removes the
last one (overflow waves, the scratch of bs1 serves every n); the first two
are allocated at boot for --d-bs seats.  Loading them only for the n occupied
seats and handing the rest to the experts per flip needs a VRAM re-partition
at the wake (open, metal).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence

PARK_ENV = "SGLANG_WEG2_D_PARK"
#: mirrors ``corridor_guard.GROUP_ENV`` / ``retract_retain.GROUP_ENV``.
GROUP_ENV = "SGLANG_WEG2_GROUP"
#: A park_running whose sleep never comes (the front changed its mind) must not
#: strand the parked requests: after this many seconds awake they re-join the
#: queue head.  <= 0 disables the fallback.
AWAKE_REQUEUE_ENV = "SGLANG_WEG2_D_PARK_AWAKE_REQUEUE_S"
AWAKE_REQUEUE_DEFAULT_S = 30.0
#: Stage-2 hook: a pressure-parked request may resume BEFORE the older one
#: finishes once ``avail - need >= margin`` held for RESUME_STEPS passes.
#: Default -1 = off: the stage-1 rule is literally "the older one is done".
RESUME_MARGIN_ENV = "SGLANG_WEG2_D_PARK_RESUME_MARGIN_TOKENS"
RESUME_STEPS_ENV = "SGLANG_WEG2_D_PARK_RESUME_STEPS"
RESUME_STEPS_DEFAULT = 8

SITE_ATTR = "_weg2_d_park_site"
EPOCH_ATTR = "_weg2_d_park_epoch"
SINCE_ATTR = "_weg2_d_park_since"
SITE_PRESSURE = "pressure"
SITE_FLIP = "flip"
SITES = (SITE_PRESSURE, SITE_FLIP)

_OFF = ("0", "false", "no", "off")


def d_park_active(env: Optional[Mapping[str, str]] = None) -> bool:
    """Standard on group D; ``SGLANG_WEG2_D_PARK=0`` turns it off (A/B)."""
    env = os.environ if env is None else env
    if str(env.get(PARK_ENV, "1")).strip().lower() in _OFF:
        return False
    return str(env.get(GROUP_ENV, "")).strip().upper() == "D"


def awake_requeue_s(env: Optional[Mapping[str, str]] = None) -> float:
    env = os.environ if env is None else env
    try:
        return float(env.get(AWAKE_REQUEUE_ENV, AWAKE_REQUEUE_DEFAULT_S))
    except (TypeError, ValueError):
        return AWAKE_REQUEUE_DEFAULT_S


def _arrival(req) -> float:
    """FCFS position; a request that never passed the queue sorts last."""
    seq = getattr(req, "kv_arrival_seq", None)
    return float("inf") if seq is None else float(seq)


def park_site(req) -> Optional[str]:
    site = getattr(req, SITE_ATTR, None)
    return site if site in SITES else None


def mark_parked(req, site: str, *, epoch: Optional[int] = None, now: float = 0.0) -> None:
    if site not in SITES:
        raise ValueError(f"unknown park site {site!r}")
    setattr(req, SITE_ATTR, site)
    setattr(req, EPOCH_ATTR, epoch)
    setattr(req, SINCE_ATTR, float(now))


def clear_park(req) -> None:
    setattr(req, SITE_ATTR, None)


def park_running_order(reqs: Iterable) -> List:
    """The order a flip park keeps and the wake resumes: oldest first."""
    return sorted(list(reqs), key=_arrival)


def retraction_order(reqs: Sequence, *, spec_active: bool) -> Optional[List[int]]:
    """Indices from most- to least-protected; ``retract_decode`` pops from the
    END, so the YOUNGEST (least protected by kvso's ``session_priority_key``)
    parks first.

    Under speculative decoding a request may only leave the batch from the
    BACK (``filter_batch`` / kvso ``spec_decline_non_back_spill``).  Then the
    stock back-only order stands and this returns it unchanged when the back
    IS the youngest; ``None`` when it is not, so the caller keeps the stock
    order and names the deviation instead of reordering a spec batch.
    """
    from sglang.srt.managers.kv_session_offload import session_priority_key

    n = len(reqs)
    if n == 0:
        return []
    by_protection = sorted(range(n), key=lambda i: session_priority_key(reqs[i]), reverse=True)
    if not spec_active:
        return by_protection
    youngest = by_protection[-1]
    if youngest == n - 1:
        return list(range(n))
    return None


def order_waiting(waiting: Sequence) -> List:
    """D's queue order under the park: every parked request first (oldest
    first), then the rest in the order they already had.

    The rest is deliberately NOT re-sorted: the queue arrives here in the
    group's order (#823 W9 ``_apply_uniform_head_order`` puts the requests
    held on every rank first), and a re-sort could lift a request one rank
    does not hold above that head.  The park marks themselves are set by
    group-uniform events, so moving the parked requests is replicated."""
    parked = [r for r in waiting if park_site(r) is not None]
    rest = [r for r in waiting if park_site(r) is None]
    return park_running_order(parked) + rest


def head_of_queue(waiting: Sequence, reqs: Sequence) -> List:
    """``reqs`` (a park) move to the queue head, oldest first; the rest keep
    their order."""
    ids = {id(r) for r in reqs}
    return park_running_order(reqs) + [q for q in waiting if id(q) not in ids]


@dataclass(frozen=True)
class AdmissionGate:
    """One pass's verdict for D's admission loop.

    ``barrier`` -- a parked request is still waiting (in the queue or outside
    it, e.g. a post-wake settle): new requests are not admitted this pass, so
    no newcomer takes the seat a parked request is coming back to.
    ``blocked`` -- pressure-parked rids that may not resume yet.
    """

    barrier: bool = False
    blocked: FrozenSet[str] = frozenset()
    note: str = ""

    def skip(self, req) -> Optional[str]:
        """Census key when ``req`` is skipped this pass, else None."""
        site = park_site(req)
        if site is not None:
            return "weg2_d_park_older_live" if str(req.rid) in self.blocked else None
        return "weg2_d_park_first" if self.barrier else None


@dataclass
class ResumeBook:
    """Stage-2 early resume (default off): per parked rid, kvso's
    ``RestoreHysteresis`` over ``avail - need >= margin``.  Updated once per
    pass with replicated inputs, so it is rank-uniform."""

    margin_tokens: int = -1
    steps: int = RESUME_STEPS_DEFAULT
    _hyst: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "ResumeBook":
        env = os.environ if env is None else env
        try:
            margin = int(str(env.get(RESUME_MARGIN_ENV, "-1")).strip())
        except ValueError:
            margin = -1
        try:
            steps = int(str(env.get(RESUME_STEPS_ENV, RESUME_STEPS_DEFAULT)).strip())
        except ValueError:
            steps = RESUME_STEPS_DEFAULT
        return cls(margin_tokens=margin, steps=max(1, steps))

    def early_ok(self, rid: str, *, avail_tokens: int, need_tokens: int) -> bool:
        if self.margin_tokens < 0:
            return False
        from sglang.srt.managers.kv_session_offload import RestoreHysteresis

        h = self._hyst.get(rid)
        if h is None:
            h = self._hyst[rid] = RestoreHysteresis(self.steps)
        return bool(h.update(int(avail_tokens) - int(need_tokens) >= self.margin_tokens))

    def forget(self, live_rids: Iterable[str]) -> None:
        keep = set(live_rids)
        for rid in [r for r in self._hyst if r not in keep]:
            del self._hyst[rid]


def admission_gate(
    waiting: Sequence,
    *,
    running: Sequence,
    pending_outside: Sequence = (),
    avail_tokens: Optional[int] = None,
    resume_book: Optional[ResumeBook] = None,
) -> AdmissionGate:
    """The stage-1 rule, per pass.

    * A FLIP-parked request resumes as soon as it fits (oldest first; the
      adder's own NO_TOKEN stops the pass).
    * A PRESSURE-parked request resumes when it is the oldest live request --
      "ist der aeltere fertig, wird der juengere zurueckgeholt" -- or, only
      with the stage-2 margin armed, after the hysteresis held.
    * While any parked request waits, no newcomer is admitted.
    """
    parked_waiting = [r for r in waiting if park_site(r) is not None]
    parked_outside = [r for r in pending_outside if park_site(r) is not None]
    if not parked_waiting and not parked_outside:
        if resume_book is not None:
            resume_book.forget(())
        return AdmissionGate()
    live = list(running) + parked_waiting + parked_outside
    blocked = set()
    for r in parked_waiting:
        if park_site(r) != SITE_PRESSURE:
            continue
        mine = _arrival(r)
        older_live = any(x is not r and _arrival(x) < mine for x in live)
        if not older_live:
            continue
        if resume_book is not None and avail_tokens is not None:
            need = len(getattr(r, "origin_input_ids", None) or ()) + len(
                getattr(r, "output_ids", None) or ()
            )
            if resume_book.early_ok(str(r.rid), avail_tokens=int(avail_tokens), need_tokens=need):
                continue
        blocked.add(str(r.rid))
    if resume_book is not None:
        resume_book.forget(str(r.rid) for r in parked_waiting)
    note = (
        f"gate=weg2_d_park(parked_waiting={len(parked_waiting)} "
        f"parked_outside={len(parked_outside)} blocked={len(blocked)})"
    )
    return AdmissionGate(barrier=True, blocked=frozenset(blocked), note=note)


def awake_requeue_due(parked: Sequence, *, now: float, bound_s: float) -> bool:
    """A flip park whose sleep never came: due once the OLDEST park is older
    than ``bound_s`` (rank-local clock -- the caller min-reduces the verdict
    over the group before acting, the weg2xsn296 rule)."""
    if bound_s <= 0 or not parked:
        return False
    stamps = [getattr(r, SINCE_ATTR, None) for r in parked]
    since = min(float(now) if t is None else float(t) for t in stamps)
    return (now - since) >= bound_s


# ---------------------------------------------------------------------------
# The seat posts on the attention host (Form A TP0).  The budget arithmetic
# itself lives in planner/expert_residency.py (``seat_rescale_reference``);
# this is the runtime's slot formula restated, bound to
# model_runner_kv_cache_mixin's constants by the unit test.
# ---------------------------------------------------------------------------

#: model_runner_kv_cache_mixin.MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO
MAMBA_RATIO_BASE = 3
#: ... + MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP (D runs the overlap schedule
#: with --mamba-radix-cache-strategy extra_buffer, launcher.argv_d)
MAMBA_RATIO_OVERLAP_EXTRA = 2
#: model_runner_kv_cache_mixin.MAMBA_AUTO_SAFETY_MARGIN
MAMBA_SAFETY = 1.25


def mamba_slots_for_seats(
    seats: int,
    *,
    ratio: int = MAMBA_RATIO_BASE + MAMBA_RATIO_OVERLAP_EXTRA,
    safety: float = MAMBA_SAFETY,
) -> int:
    """``_auto_mamba_demand_size``: ceil(seats x ratio x safety), floored at
    ``ratio`` (the hard floor ``seats x slots_per_running_req`` lies below it
    for this shape)."""
    s = max(1, int(seats))
    return int(max(math.ceil(s * int(ratio) * float(safety)), int(ratio)))


@dataclass(frozen=True)
class PhaseSeats:
    """H95: one D phase's seat count, decided at the wake of D."""

    n: int
    handoff_n: int
    parked_n: int
    cap: int
    epoch: Optional[str] = None

    @property
    def clamped(self) -> bool:
        return self.handoff_n + self.parked_n > self.cap

    def line(self) -> str:
        return (
            "WEG2 D-PHASE-SEATS (H95) epoch=%s handoff_n=%d parked_n=%d -> n=%d of "
            "cap %d%s: decode batch bs%d, GDN slots in use <= %d of %d (boot), "
            "replicated from the wake request, no collective"
            % (
                self.epoch, self.handoff_n, self.parked_n, self.n, self.cap,
                " (CLAMPED: the front handed more than --d-bs)" if self.clamped else "",
                self.n, mamba_slots_for_seats(self.n), mamba_slots_for_seats(self.cap),
            )
        )


def phase_seats(
    handoff_n: Optional[int], parked_n: Optional[int], *, cap: int,
    epoch: Optional[str] = None,
) -> Optional[PhaseSeats]:
    """H95: the seats of the D phase that starts with this wake -- every
    request the ending P phase handed over plus the wait-bound-parked ones D
    resumes first, at least 1, at most ``cap`` (= D's --max-running-requests,
    the --d-bs upper bound). ``None`` when the wake carries no count (a P
    wake, a pre-H91c front, a stock resume): nothing is decided then.

    A pure function of the wake request's two integers and of the boot's
    ``cap`` -- both identical on every rank (the scheduler receives the SAME
    control request on each rank), so the ranks cannot disagree."""
    if handoff_n is None and parked_n is None:
        return None
    h = max(0, int(handoff_n or 0))
    p = max(0, int(parked_n or 0))
    c = max(1, int(cap))
    return PhaseSeats(n=max(1, min(h + p, c)), handoff_n=h, parked_n=p, cap=c, epoch=epoch)


def graph_bs_covers(graph_bs: Optional[Sequence[int]], seats: int) -> bool:
    """Does the decode graph list capture every batch size 1..seats?  A seat
    count without its graph runs that batch EAGER -- no refusal, just slow."""
    if graph_bs is None:
        return True  # the runtime's own default list; not ours to judge
    have = {int(b) for b in graph_bs}
    return all(b in have for b in range(1, int(seats) + 1))
