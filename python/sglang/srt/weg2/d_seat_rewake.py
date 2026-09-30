"""D-SEAT-REWAKE: D's phase seat count n moves LIVE, both ways.

Nutzer 30.09. (wörtlich): "einfache lösung, es wird sofort auf P geflippt
prefillt und dann wird D mit mehr sitzen gewaked. - oder D sleeped (ohne
wirklich "runterzufaren") und waket sofort wieder mit mehr sitzen, macht den
decode selbst (weil kurz) und decoded dann alle sitze weiter"; and: "dann kann
der D teil auch wenn sitze nicht mehr belegt sein "intern flippen" zu einem
layout mit weniger sitzen, dann gibts mehr platz für experten - selbe funktion
nur auch wieder in die andere rictung um scneller zu werden".

Befund (NF y4s 09301432, front): a D phase has the seats its wake fixed (H95c,
n = handoff_n + parked_n); an arrival whose KV fits waited "why=seat taken=n
n=n" with 441613 tokens free (weg2-10-16 6.43 s, weg2-4-9 6.4 s).

At a round boundary, rank-uniform, no weight legs, no P (variant B):

* GROW n -> n + k when k waiting requests (not pressure-parked) find every
  seat taken and n < cap: at once -- a waiting request is real demand. The
  new seats' GDN cells map; the expert rows their pages fund go OFF coldest
  first before (``SeatVram.reseat_live``).
* SHRINK n -> n - k when seats stand free and nobody waits -- only once the
  continuous free time exceeds the MEASURED price of a re-plan round trip
  (grow + shrink; ski rental, the flip policy's shape: a short gap does not
  flap). The freed GDN cells go back to expert rows. Before a live re-plan
  is measured the wake's own page apply (``PhaseState.apply_ms``) is the
  price; with neither, no shrink (a guess is not a price).

Every verdict is REPLICATED: its inputs are the request lists, the phase and
the allocator (the same on every D rank); the page part is rank-local and its
success -- and each rank's shrink verdict, which reads its own clock and its
own measured price -- goes through the group MIN, so the ranks never
disagree on n (RAENGE-NIE-UNEINS). A shrink is asked at most every
``SHRINK_ASK_ROUNDS`` rounds of the replicated idle precondition (one small
CPU all-reduce then, none on a decode round otherwise).

Variant A (over X: the flip) keeps its path: the P->D wake fixes n, and a
waiter the wake's n leaves without a seat grows it here at the next round.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

MARKER = "WEG2 D-SEAT-REWAKE"
ATTR = "_weg2_d_seat_rewake"
#: rounds of the replicated idle precondition between two shrink questions
SHRINK_ASK_ROUNDS = 32


def enabled() -> bool:
    try:
        from sglang.srt.environ import envs

        return bool(envs.SGLANG_WEG2_D_SEAT_REWAKE.get())
    except Exception:  # noqa: BLE001
        return False


@dataclass
class RewakeState:
    """Per rank. ``grow_ms``/``shrink_ms``: the last measured live re-plans
    (decode pause, ms) of THIS rank; ``idle_since``: when the replicated idle
    precondition began on this rank's clock; ``idle_rounds``: its length in
    rounds (replicated)."""

    grow_ms: Optional[float] = None
    shrink_ms: Optional[float] = None
    idle_since: Optional[float] = None
    idle_rounds: int = 0
    refused: Dict[str, int] = field(default_factory=dict)
    epoch: Optional[str] = None
    counters: Dict[str, int] = field(default_factory=lambda: {"grow": 0, "shrink": 0,
                                                              "grow_refused": 0, "shrink_refused": 0,
                                                              "shrink_asked": 0})


def price_ms(rs: RewakeState, wake_apply_ms: Optional[float]) -> Optional[float]:
    """The ski-rental price of a shrink: a re-plan ROUND TRIP (the shrink now
    and the grow the next arrival may need), from measurements only. Missing
    directions take the other measured one; with no live measurement, twice
    the wake's measured page apply; None = nothing measured, no shrink."""
    g, s = rs.grow_ms, rs.shrink_ms
    if g is not None or s is not None:
        g = g if g is not None else s
        s = s if s is not None else g
        return float(g) + float(s)
    if wake_apply_ms is not None:
        return 2.0 * float(wake_apply_ms)
    return None


def grow_target(n: int, cap: int, running: int, waiting: int) -> Optional[int]:
    """GROW: every seat taken and ``waiting`` requests stand behind them ->
    ``n + k`` (at most cap), k = the waiters the new seats take. None: no grow."""
    n, cap = int(n), int(cap)
    if waiting <= 0 or n >= cap or running < n:
        return None
    return min(cap, n + int(waiting))


def shrink_target(n: int, running: int, waiting: int) -> Optional[int]:
    """SHRINK candidate: nobody waits and seats stand free -> the running
    count (at least 1). None: no shrink."""
    n = int(n)
    if waiting > 0 or running >= n or n <= 1:
        return None
    return max(1, int(running))


def shrink_due(idle_s: Optional[float], price: Optional[float]) -> bool:
    """Ski rental: the free seats have stood longer than a re-plan round trip
    costs. No price (nothing measured) -> never."""
    return idle_s is not None and price is not None and float(idle_s) * 1000.0 >= float(price)


def _group_min(sched: Any, flags: Sequence[bool]) -> List[int]:
    gm = getattr(sched, "_weg2_group_min_flags", None)
    return list(gm(list(flags))) if callable(gm) else [1 if f else 0 for f in flags]


def _lists(sched: Any):
    from sglang.srt.weg2 import d_seats
    from sglang.srt.weg2.d_seat_vram import _demand_lists

    running, _adm = _demand_lists(sched)
    queue = list(getattr(sched, "waiting_queue", None) or ())
    waiting = [q for q in queue if d_seats.park_site(q) != d_seats.SITE_PRESSURE]
    return running, waiting


def _set_limit(sched: Any, st: Any, n_new: int) -> bool:
    """The allocator's slot limit and the phase's n (REPLICATED inputs)."""
    from sglang.srt.weg2 import d_seat_vram as V

    pool = V._req_pool(sched)
    allocator = getattr(pool, "mamba_allocator", None)
    if allocator is not None and hasattr(allocator, "set_phase_limit"):
        size = int(getattr(allocator, "size", 0) or 0)
        want = None if n_new >= st.cap else V.phase_slot_limit(size, n_new, st.cap)
        if not allocator.set_phase_limit(want, seats=n_new):
            return False
        st.slot_limit = want
    st.n = int(n_new)
    return True


def _reseat(sched: Any, st: Any, n_new: int, grow: bool) -> Optional[float]:
    """One live re-plan; this rank's measured ms (the decode pause), or None
    when the group refused.

    GROW: the pages first (rank-local), then the group MIN of their success;
    only an agreed grow raises the slot limit and n (a slot is never handed
    out before every rank has its pages). SHRINK: the slot limit and n come
    down FIRST (the allocator's ledger is replicated, the caller checked the
    slots above are free), then the pages -- a rank that could not free them
    keeps them mapped, which is safe below a lowered limit."""
    from sglang.srt.weg2 import d_seat_vram as V

    t0 = time.perf_counter()
    if not grow and not _set_limit(sched, st, n_new):
        return None
    ok = True
    ctl = V.controller(sched)
    applied = None
    if ctl is not None:
        try:
            applied = ctl.reseat_live(n_new, st.stage)
        except V.Weg2DSeatVramRefused as exc:
            ok = False
            logger.warning("%s %s n->%d refused on this rank: %s: %s", MARKER,
                           "GROW" if grow else "SHRINK", n_new, type(exc).__name__, exc)
    if grow:
        agreed = bool(_group_min(sched, [ok])[0])
        if not agreed or not _set_limit(sched, st, n_new):
            return None
    ms = (time.perf_counter() - t0) * 1000.0
    if applied is not None and ctl is not None:
        logger.info("%s", V.phase_line(st, applied, ctl))
    return ms


def parts_text(ctl: Any, pause_ms: float) -> str:
    """The pause in parts: rows OFF (copies), cells released, cells mapped,
    device syncs, and the rest (the group MIN, the slot limit) -- per rank."""
    p = dict(getattr(ctl, "last_reseat_parts", None) or {})
    keys = ("off_copy_ms", "unmap_ms", "map_ms", "sync_ms")
    rest = max(0.0, float(pause_ms) - sum(float(p.get(k, 0.0)) for k in keys))
    return " ".join("%s=%.1f" % (k, float(p.get(k, 0.0))) for k in keys) + " other_ms=%.1f" % rest


def round_boundary(sched: Any) -> Optional[str]:
    """THE D round boundary's memory step: ONE of the two live re-plans per
    iteration, never both against each other. The seat re-plan goes first
    (a waiting request is demand now; its GROW/SHRINK is replicated); only
    when it did not move does the D-MEM-SCHED tick move the KV stage -- with
    the live n, beside the live GDN pages (``apply_stage``). A moved seat
    count lets the stage tick run at the next iteration, on the new n."""
    from sglang.srt.weg2 import d_seat_vram as V

    moved = tick(sched)
    if moved is None:
        V.runtime_tick(sched)
        return None
    V._tick_noop(sched)
    return moved


def tick(sched: Any) -> Optional[str]:
    """Once per iteration of an AWAKE D, after the D-MEM-SCHED tick. Returns
    "grow"/"shrink" when n moved, else None."""
    from sglang.srt.weg2 import d_seat_vram as V

    if not enabled() or not V.armed() or getattr(sched, "weg2_dormant", False):
        return None
    st = getattr(sched, V.PHASE_ATTR, None)
    if st is None or not st.done or st.n is None:
        return None
    rs = getattr(sched, ATTR, None)
    if rs is None:
        rs = RewakeState()
        setattr(sched, ATTR, rs)
    if rs.epoch != st.epoch:
        rs.epoch, rs.idle_since, rs.idle_rounds, rs.refused = st.epoch, None, 0, {}
    running, waiting = _lists(sched)
    n, cap = int(st.n), int(st.cap)
    # a SHRINK that leaves a KV stage above what fewer seats may take, or a
    # GROW into one, is not a form the capture holds (replicated inputs)
    form = V.stage_form()

    def _stage_ok(n_new: int) -> bool:
        return form is None or st.stage is None or int(st.stage) <= form.max_stage(n_new)

    up = grow_target(n, cap, len(running), len(waiting))
    if up is not None:
        rs.idle_since, rs.idle_rounds = None, 0
        while up > n and not _stage_ok(up):
            up -= 1
        key = "grow:%d->%d" % (n, up)
        if up <= n or rs.refused.get(key):
            return None
        rows0 = getattr(V.controller(sched), "rows_on", None)
        ms = _reseat(sched, st, up, grow=True)
        if ms is None:
            rs.refused[key] = 1
            rs.counters["grow_refused"] += 1
            logger.warning("%s GROW REFUSED n=%d->%d waiting=%d -- a rank could not map the new "
                           "seats' pages; n stays (asked again after the next wake or n change)",
                           MARKER, n, up, len(waiting))
            return None
        rs.grow_ms = ms
        rs.counters["grow"] += 1
        V._reopen_admission(sched)
        logger.warning("%s GROW n=%d->%d waiting=%d running=%d pause_ms=%.1f (%s) rows_on %s->%s "
                       "mamba_keep=%s (the waiting requests take the new seats at this round; "
                       "no weight legs, no P)", MARKER, n, up, len(waiting), len(running), ms,
                       parts_text(V.controller(sched), ms), rows0,
                       getattr(V.controller(sched), "rows_on", None),
                       getattr(V.controller(sched), "mamba_keep", None))
        return "grow"
    down = shrink_target(n, len(running), len(waiting))
    if down is None or not _stage_ok(down):
        rs.idle_since, rs.idle_rounds = None, 0
        return None
    now = time.monotonic()
    if rs.idle_since is None:
        rs.idle_since, rs.idle_rounds = now, 0
    rs.idle_rounds += 1
    if rs.idle_rounds % SHRINK_ASK_ROUNDS:
        return None
    # the slots above the new limit must be free: the allocator decides (replicated)
    pool = V._req_pool(sched)
    allocator = getattr(pool, "mamba_allocator", None)
    if allocator is not None and hasattr(allocator, "slot_used"):
        size = int(getattr(allocator, "size", 0) or 0)
        lim = V.phase_slot_limit(size, down, cap)
        try:
            if bool(allocator.slot_used[lim + 1:].any()):
                return None
        except Exception:  # noqa: BLE001 -- unreadable ledger: no shrink
            return None
    price = price_ms(rs, getattr(st, "apply_ms", None))
    idle_s = now - rs.idle_since
    rs.counters["shrink_asked"] += 1
    if not bool(_group_min(sched, [shrink_due(idle_s, price)])[0]):
        return None
    rows0 = getattr(V.controller(sched), "rows_on", None)
    ms = _reseat(sched, st, down, grow=False)
    if ms is None:
        rs.counters["shrink_refused"] += 1
        rs.idle_since, rs.idle_rounds = now, 0
        logger.warning("%s SHRINK REFUSED n=%d->%d -- a rank could not move its pages; n stays",
                       MARKER, n, down)
        return None
    rs.shrink_ms = ms
    rs.counters["shrink"] += 1
    rs.idle_since, rs.idle_rounds = None, 0
    logger.warning("%s SHRINK n=%d->%d running=%d idle_s=%.2f price_ms=%s pause_ms=%.1f (%s) rows_on %s->%s "
                   "mamba_keep=%s (the free seats' pages went back to expert rows)", MARKER, n, down,
                   len(running), idle_s, "-" if price is None else "%.1f" % price, ms,
                   parts_text(V.controller(sched), ms), rows0,
                   getattr(V.controller(sched), "rows_on", None),
                   getattr(V.controller(sched), "mamba_keep", None))
    return "shrink"
