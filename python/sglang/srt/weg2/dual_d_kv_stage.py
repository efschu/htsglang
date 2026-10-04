"""DUAL-TP3PP3 unified KV per card (C): group D's KV actor.

D has priority on the card's ONE KV pool (user order 30.09. 07:25Z). D grows
from what is free and, when short, raises pressure on P through the card
ledger -- P pauses at its next chunk boundary and gives its context back
(front ``_dual_pause_inflight``, P actor ``dual_p_kv_stage``). D gives memory
back when its running work no longer needs it, so a waiting P prompt is not
starved by D's idle cache.

Why not ``d_seat_vram.apply_stage``: its cells are priced against D's BOOT
form ("the total never above the boot form") -- D could never grow into P's
share. The primitives are the same (born at a TOP range and trimmed,
``slot_spans`` on a fixed lattice, the allocator cap ``_engage_kv_cap``,
``max_live_page``); the decision is this module's (NF 30.09.: the stage
decision for the dual layout lives in the dual arbiter, not in MemSched).

GROUP-CONSISTENT: D's allocator is replicated over its TP ranks (uneven DCP:
each rank holds its owner share, ``KvTensorGeom.owner_block``). Every rank
computes the same wanted level from GLOBAL demand (``d_seat_vram.
global_demand``); the grant is local per card, so the ranks agree with one
MIN collective (``sched._weg2_group_min_ints``) -- entered only when the level
wants to move, which is decided from replicated inputs.
Armed only by ``SGLANG_WEG2_DUAL_LAYOUT=1`` + group D +
``SGLANG_WEG2_DUAL_D_KV_MAX_TOKENS`` > 0.
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional, Sequence, Tuple

from sglang.srt.weg2 import dual_p_kv_stage as _pk

logger = logging.getLogger(__name__)

MARK = "DUAL-TP3PP3 D-KV"
MAX_TOKENS_ENV = "SGLANG_WEG2_DUAL_D_KV_MAX_TOKENS"
#: rounds a shrink condition must hold before D gives memory back (no flap)
SHRINK_HOLD_ROUNDS = 64
#: Q-696: a RUNNING D gave its unlocked cache to a P that waited for a card
LIVE_YIELD_MARK = "DUAL D-CACHE-YIELD-LIVE"
#: Q-696: P waits for a card, D maps more than it needs, and the shrink cannot go
SHRINK_BLOCKED_MARK = "SHRINK-BLOCKED"


def live_yield_wait_s() -> float:
    """Q-696: P's card wait (seconds, group MAX) before a running D yields its
    cache; <= 0 = off (only an idle D yields, the pre-Q-696 rule)."""
    from sglang.srt.environ import envs

    return float(envs.SGLANG_WEG2_DUAL_D_LIVE_YIELD_WAIT_S.get())


def regrow_hold_s() -> float:
    """Q-696: no shrink this many seconds after a grow (no grow/shrink/grow flap)."""
    from sglang.srt.environ import envs

    return max(0.0, float(envs.SGLANG_WEG2_DUAL_D_REGROW_HOLD_S.get()))


def live_yield_due(p_wait_s: float, wait_s: float) -> bool:
    """Q-696 (pure): has P waited long enough for a running D to yield its cache?"""
    return float(wait_s) > 0 and float(p_wait_s) >= float(wait_s)


def shrink_blocked_reason(*, mapped: int, need: int, floor: int, step: int, holds: bool,
                          p_missing: bool, regrow_hold: bool) -> Optional[str]:
    """Q-696 (pure): why a P-waiting tick did not shrink although D maps more
    than its work needs (``need``: the level demand + air ask, before the live
    floor). None when D maps nothing beyond ``need``."""
    if int(mapped) - int(need) < int(step):
        return None
    if holds:
        return "holds"
    if p_missing:
        return "p_missing"
    if regrow_hold:
        return "regrow_hold"
    if _pk.round_up(int(floor), int(step)) >= int(mapped):
        return "live_floor"
    return "d_air"      # decide()'s D-PRIORITY room: air + two lattice steps stay free (gmps12)


_D_BORN: List[Tuple[int, object]] = []
#: D's boot level (GLOBAL tokens), set at _config_from_budget before the pool
#: is built; None = not the target pool (the draft's pool passes untouched)
_BOOT_TOKENS: Optional[int] = None


def armed(env=None) -> bool:
    env = os.environ if env is None else env
    if str(env.get("SGLANG_WEG2_DUAL_LAYOUT", "")).strip() != "1":
        return False
    if str(env.get("SGLANG_WEG2_GROUP", "")).strip().upper() != "D":
        return False
    try:
        return int(env.get(MAX_TOKENS_ENV, "0") or 0) > 0
    except ValueError:
        return False


def max_tokens(env=None) -> int:
    env = os.environ if env is None else env
    return int(env.get(MAX_TOKENS_ENV, "0") or 0)


def pool_tokens(boot_tokens: int) -> int:
    return max(int(boot_tokens), max_tokens()) if armed() else int(boot_tokens)


def _owner_block_for_rows(rows: int, top: int, page: int) -> Tuple[int, int]:
    from sglang.srt.weg2.d_seat_vram import kv_owner_block

    S, ratio = kv_owner_block()
    if S <= 0 or ratio <= 0 or ratio >= S:
        return (0, 0)
    compact = (int(top) // S + 1) * ratio
    return (S, ratio) if int(rows) in (compact, compact + int(page)) else (0, 0)


def born(pool, t, name: str, *, boot_tokens: Optional[int] = None, spans=None,
         granule: Optional[int] = None):
    """A D K/V buffer keeps its TOP range and D's BOOT level mapped. Only the
    target pool sized at the top (whole or compacted owner share) is touched."""
    if boot_tokens is None:
        boot_tokens = _BOOT_TOKENS
    if not armed() or t is None or not t.numel() or boot_tokens is None:
        return t
    top_ = max_tokens()
    size, page_ = int(getattr(pool, "size", 0) or 0), int(getattr(pool, "page_size", 1) or 1)
    if size < top_ and _owner_block_for_rows(size, top_, page_) == (0, 0):
        return t
    from sglang.srt.weg2.d_seat_vram import (
        KvTensorGeom,
        SlotTensorGeom,
        Weg2DSeatVramRefused,
        _sync_before_unmap,
        granule_for,
        slot_spans,
        tms,
    )

    spans = tms() if spans is None else spans
    info = spans.info(t.data_ptr()) if spans.available else None
    if info is None:
        raise Weg2DSeatVramRefused("%s: the KV tensor %s is not a saver allocation" % (MARK, name))
    g = int(granule or granule_for(t.device))
    rows = int(t.shape[0])
    slot_bytes = (t.numel() * t.element_size()) // max(1, rows)
    top = max_tokens()
    geom = KvTensorGeom(SlotTensorGeom(name, 1, rows, int(slot_bytes), int(info.size)),
                        token_ratio=1, token_pad=int(pool.page_size),
                        owner_block=_owner_block_for_rows(rows, top, int(pool.page_size)))
    step = _pk.step_tokens()
    boot = min(top, _pk.round_up(boot_tokens, step))
    cuts = [geom.slots_for(k) for k in _pk.lattice(top, step)]
    _sync_before_unmap(t)
    rc = spans.set_spans(t.data_ptr(), slot_spans(geom.geom, geom.slots_for(boot), g, cuts=cuts), now=True)
    if rc != 0:
        raise Weg2DSeatVramRefused("%s: trimming %s to %d tokens failed (rc=%d)" % (MARK, name, boot, rc))
    _D_BORN.append((int(t.data_ptr()), geom))
    pool.set_stage_backed_rows(geom.slots_for(boot))
    return t


class DKvStage(_pk.PKvStage):
    """D's share of its card. Same mapping mechanics as P's actor; the
    decision is group-wide (MIN over the D ranks) and D may press P."""

    def __init__(self, *a, gmin=None, **kw):
        super().__init__(*a, **kw)
        self.gmin = gmin or (lambda vals: list(vals))
        self._below = 0

    def group_grow(self, want: int) -> bool:
        """Every rank asks its card for the bytes up to ``want``; all map it or
        nobody does (MIN of the per-rank verdict). A short card's request left
        pressure on P there -- P pauses and releases, the next tick succeeds."""
        want = min(self.top, _pk.round_up(want, self.step))
        if want <= self.mapped_tokens:
            return True
        need = self.bytes_for(want) - self.bytes_for(self.mapped_tokens)
        granted, pressure = self.ledger.request(need)
        ok = 1 if granted >= need else 0
        if int(self.gmin([ok])[0]) != 1:
            if granted:
                self.ledger.release(granted)
            t = _pk._now()
            self._gw_n = int(getattr(self, "_gw_n", 0)) + 1
            if t >= getattr(self, "_gw_next", 0.0):   # 1 line/s at most, backing off (metal dual13)
                self._gw_iv = 2.0 * float(getattr(self, "_gw_iv", 0.5))
                self._gw_next = t + self._gw_iv
                logger.info("%s GROUP-WAIT want=%d need=%d granted=%d pressure_on_P=%d waits=%d", MARK, want,
                            need, granted, pressure, self._gw_n)
            return False
        self._gw_n, self._gw_next, self._gw_iv = 0, 0.0, 0.5
        # metal dual14: cuMemCreate OOM here killed D. A physically short card
        # is "card short": EVERY rank rolls back (second MIN), the grant goes
        # back, the failing card's ledger is reconciled against cuMemGetInfo
        # and the shortfall is asked again -- which presses P.
        moved, short = 1, None
        try:
            self._move(want)
        except _pk.Weg2DualKvMapShort as exc:
            moved, short = 0, exc
        if int(self.gmin([moved])[0]) != 1:
            self._move(self.mapped_tokens)                   # unmap what this rank got; rows/cap back
            self.ledger.release(granted)
            over = 0
            phys = _pk.phys_free_bytes() if short is not None else None
            if phys is not None:
                over = self.ledger.reconcile(phys)
            got, pressure = self.ledger.request(need)       # registers D's demand; presses P if it holds any
            if got:
                self.ledger.release(got)
            logger.warning("%s GROW-SHORT want=%d need=%d: %s -- grow refused on every rank, rolled back to %d; "
                           "ledger reconciled by -%d B against phys_free=%s; pressure_on_P=%d",
                           MARK, want, need, short if short is not None else "another rank's card is short",
                           self.mapped_tokens, over, phys, pressure)
            return False
        self._committed = int(getattr(self, "_committed", 0) or 0) + need
        self._grow_t = _pk._now()                           # Q-696: the regrow hold starts here
        logger.info("%s GROW %d -> %d tokens (+%d B)", MARK, self.mapped_tokens, want, need)
        self.mapped_tokens = want
        _pk.check_cover(self, "grow")
        return True

    def group_shrink(self, target: int, live_floor_tokens: int) -> int:
        """Give back everything above ``target`` once no page above it is live
        (``live_floor_tokens``: the GROUP's highest live page, MAX over ranks).
        The allocator cap goes to ``target`` first, so new pages land below."""
        target = min(self.mapped_tokens, _pk.round_up(max(target, live_floor_tokens), self.step))
        if target >= self.mapped_tokens:
            return 0
        self._engage_cap(self.allocator, int(target), self.page)
        live = _pk.max_live_id(self.allocator, self.page) * self.page
        if live > target:
            # cannot happen with the tick's group floor; if it does, the ranks
            # would part company -- stop by name instead of holding locally
            raise _pk.Weg2DualKvCapBreach(
                "%s SHRINK under a live row: token %d is live above the target %d on this rank although "
                "the group floor was %d" % (MARK, live, target, int(live_floor_tokens)))
        self._sync()
        n = self.bytes_for(self.mapped_tokens) - self.bytes_for(target)
        self._move(target)
        self.ledger.release(n)
        self._committed = max(0, int(getattr(self, "_committed", 0) or 0) - n)
        logger.info("%s SHRINK %d -> %d tokens (-%d B back to the card pool)", MARK, self.mapped_tokens,
                    target, n)
        self.mapped_tokens = target
        _pk.check_cover(self, "shrink")
        return n


def want_tokens(used: int, incoming: int, air: int, step: int) -> int:
    """The level D's running work needs: what the running requests hold, the
    queue head, and the decode/verify air -- on the lattice."""
    return _pk.round_up(max(0, int(used)) + max(0, int(incoming)) + max(0, int(air)), int(step))


def d_locked_rows(sched, actor) -> int:
    """The rows D's pool holds that nothing can evict: mapped minus free minus
    the tree's evictable cache. Counts what the request bookkeeping misses --
    a #243 hand-off hold, a D-PARK, a retract-retain. Metal gmps7 (D 17:53:00-15):
    weg2-0-13's #243 hold kept 61625 rows on the device, d_demand did not see
    them, D shrank 221184 -> 204800 for a waiting P prompt and ran full 10 s later
    ("KV cache pool is full. Retract requests. #retracted_reqs: 4")."""
    alloc = getattr(actor, "allocator", None)
    try:
        free = int(alloc.available_size())
    except Exception:  # noqa: BLE001 -- no reading, no extra term (the bookkeeping still counts)
        return 0
    ev = 0
    tree = getattr(sched, "tree_cache", None)
    if tree is not None:
        try:
            ev = int(tree.evictable_size() or 0)
        except Exception:  # noqa: BLE001
            ev = 0
    return max(0, int(actor.mapped_tokens) - free - ev)


def want_local_tokens(demand: int, locked: int, air: int, step: int) -> int:
    """D PRIORITY: the level this rank needs -- the request bookkeeping OR the
    locked rows, whichever is larger, plus the decode/verify air (the next round
    of every seat with the draft's tokens, one extend chunk), on the lattice."""
    return want_tokens(max(int(demand), int(locked)), 0, air, step)


def decide(mapped: int, want: int, p_waiting: bool, below_rounds: int, step: int,
           hold: int = SHRINK_HOLD_ROUNDS, *, avail_min: Optional[int] = None, air: int = 0,
           holds: bool = False) -> Tuple[str, int]:
    """REPLICATED, pure: ('grow', want) / ('shrink', target) / ('hold', mapped),
    plus the new below-counter via the caller. Grow at once; shrink when a P
    prompt waits (D's cache must not starve it) or after ``hold`` rounds with
    at least two lattice steps of slack.

    D PRIORITY (gmps12, D 00:16:45/00:16:50: SHRINK 167936 -> 69632 twice for a
    waiting P prompt, then a 35736-token hand-back and kv_full): ``avail_min`` --
    the group's free rows (MIN over the ranks) -- must keep ``air`` + two lattice
    steps AFTER the shrink: the shrink goes only as low as that allows (and not
    at all below one step); and while a hold exists (``holds``: a parked /
    W50-midstream request, a D-HOLD-FOR-GROW episode) a waiting P prompt shrinks
    nothing -- the held context comes back to these rows. ``avail_min=None`` (no
    reading, or nothing running, queued or held): the old rule."""
    if want > mapped:
        return "grow", want
    slack = mapped - want
    if avail_min is not None:
        room = int(avail_min) - int(air) - 2 * int(step)       # rows a shrink may still take
        slack = min(slack, (room // int(step)) * int(step)) if room > 0 else 0
        want = mapped - slack
    p_shrink = bool(p_waiting) and not bool(holds)
    if slack >= int(step) and (p_shrink or (slack >= 2 * int(step) and below_rounds >= int(hold))):
        return "shrink", want
    return "hold", mapped


#: the group-free reading's "no reading" value (a MIN collective ignores it)
NO_AVAIL = 1 << 62


def d_avail_rows(sched, actor) -> int:
    """This rank's rows a decode round can get WITHOUT the card ledger:
    allocator.available_size() -- the reading ``uniform_min_avail`` /
    ``uniform_avail_floor`` reduce for the retract check -- plus the tree's
    evictable cache, which ``update_running_batch`` evicts before it allocates.
    The evictable term on purpose: D's cached prefix is not demand (the
    cache_yield rule); counting only available_size would grow D -- and press P
    -- for cache. The retract path re-reads available_size after the eviction
    (dual_d_priority.grow_or_hold), so both read the same rows. NO_AVAIL when
    there is no reading."""
    try:
        free = int(actor.allocator.available_size())
    except Exception:  # noqa: BLE001 -- no reading: the MIN takes the other ranks', or the old rule
        return NO_AVAIL
    tree = getattr(sched, "tree_cache", None)
    try:
        ev = int(tree.evictable_size() or 0) if tree is not None else 0
    except Exception:  # noqa: BLE001 -- a tree without the counter: available alone (the stricter reading)
        ev = 0
    return free + max(0, ev)


def d_holds(sched) -> int:
    """1 while D holds rows the bookkeeping does not schedule: a parked request
    (``weg2_d_parked`` -- the D-PARK and the W50 midstream hold of
    resume_via_p.keep_on_d both land there) or a D-HOLD-FOR-GROW episode."""
    return 1 if (getattr(sched, "weg2_d_parked", None) or getattr(sched, "_weg2_d_hold", None)) else 0


def floor_want(want: int, mapped: int, avail_min: int, air: int, step: int, demand: int) -> int:
    """D PRIORITY: the level the TIGHTEST rank needs. The token sums (``want``)
    do not see the binding rank; the retract check does (MIN over the ranks).
    When the group has demand and its free rows fall under ``air`` + two lattice
    steps, D asks for the gap (at least one step) -- before the decode round
    finds the pool full."""
    if int(demand) <= 0 or int(avail_min) >= NO_AVAIL:
        return int(want)
    gap = int(air) + 2 * int(step) - int(avail_min)
    if gap <= 0:
        return int(want)
    return max(int(want), int(mapped) + max(int(step), _pk.round_up(gap, int(step))))


# -- wiring (D process, dual only) -------------------------------------------

ACTOR_ATTR = "dual_d_kv"


def attach(runner) -> Optional[DKvStage]:
    if not armed() or getattr(runner, "is_draft_worker", False) or not _D_BORN:
        return None
    import torch

    from sglang.srt.weg2 import d_seat_vram as _sv
    from sglang.srt.weg2.card_kv_ledger import CardKvLedger, ledger_path

    dev = torch.device("cuda", int(runner.gpu_id))
    card = str(torch.cuda.get_device_properties(dev).uuid)
    tag = os.environ.get("SGLANG_WEG2_DUAL_KV_TAG", "") or os.environ.get("SGLANG_WEG2_TAG", "weg2")
    ledger = CardKvLedger(ledger_path(tag, card), "D")
    pools = _pk.stage_pools(runner.token_to_kv_pool)
    actor = DKvStage(list(_D_BORN), ledger, allocator=runner.token_to_kv_pool_allocator, pools=pools,
                     page_size=int(runner.page_size), granule=_sv.granule_for(dev), top_tokens=max_tokens(),
                     sync=lambda: torch.cuda.synchronize(dev))
    boot = min(actor.top, _pk.round_up(int(getattr(runner, "_dual_d_boot_tokens", 0) or 0), actor.step))
    boot_bytes = actor.bytes_for(boot) - actor.bytes_for(0)
    ledger.contribute(boot_bytes, committed=boot_bytes)
    actor.mapped_tokens = boot
    actor._committed = boot_bytes
    actor._engage_cap(actor.allocator, boot, actor.page)
    setattr(runner, ACTOR_ATTR, actor)
    logger.info("%s JOIN card=%s boot_tokens=%d contributed=%d B (kept mapped) top=%d", MARK, card[-12:],
                boot, boot_bytes, actor.top)
    return actor


def d_demand(sched) -> int:
    """The tokens D's requests need NOW: every running request (and the
    chunked one) plus EVERY request in the waiting queue -- a deferred one
    (X-DEFER prefetch_pending: its store read landed short and is re-issued),
    a requeue returner and a request not yet admitted all need their rows for
    the loadback and the prefill. Metal bsffsv: counting only the queue head let
    D shrink under a waiting 40767-token request. The front bounds the queue by
    D's seats, so the sum is bounded too."""
    from sglang.srt.weg2 import d_seat_vram as _sv

    running = list(getattr(getattr(sched, "running_batch", None), "reqs", None) or ())
    chunked = getattr(sched, "chunked_req", None)
    if chunked is not None and all(chunked is not r for r in running):
        running.append(chunked)
    queue = list(getattr(sched, "waiting_queue", None) or ())
    return sum(_sv._req_tokens(r) for r in running) + sum(_sv._req_tokens(r) for r in queue)


def cache_yield(sched, actor, live: Optional[Tuple[int, float]] = None) -> int:
    """Evict D's whole EVICTABLE device cache (unlocked nodes only -- a live
    seat's rows are locked and stay). Backed prefixes keep their L2 copy.
    Metal dual13 (3q33cu): D held 65536 mapped rows of cache with 0 running and
    0 queued while P's grant starved on the 5090. Returns the evicted tokens.

    ``live`` (Q-696): ``(group floor, P's card wait s)`` -- the yield of a D
    that still runs work, named LIVE_YIELD_MARK. Dual y8z fs10031909: D held
    225280 rows (163929 evictable) for ONE running request while P's grant
    waited 73.9 s; the idle-only rule gave them back only when D ran empty."""
    tree = getattr(sched, "tree_cache", None)
    if tree is None:
        return 0
    try:
        ev = int(tree.evictable_size() or 0)
    except Exception:  # noqa: BLE001 -- a tree without the counter has nothing to yield
        return 0
    if ev <= 0:
        return 0
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams

    tree.evict(EvictParams(num_tokens=ev))
    t = _pk._now()
    if live is not None:
        if t >= getattr(actor, "_live_yield_log_next", 0.0):
            actor._live_yield_log_iv = min(60.0, 2.0 * float(getattr(actor, "_live_yield_log_iv", 0.5)))
            actor._live_yield_log_next = t + actor._live_yield_log_iv
            logger.info("%s %s freed=%d floor=%d p_wait_s=%.1f mapped=%d: P waits for a card and D's cache "
                        "is not demand -- unlocked nodes evicted under running work (backed prefixes stay in "
                        "L2), the next tick shrinks to the live floor (Q-696)", MARK, LIVE_YIELD_MARK, ev,
                        int(live[0]), float(live[1]), actor.mapped_tokens)
        return ev
    if t >= getattr(actor, "_yield_log_next", 0.0):
        actor._yield_log_iv = min(300.0, 2.0 * float(getattr(actor, "_yield_log_iv", 0.5)))
        actor._yield_log_next = t + actor._yield_log_iv
        logger.info("%s CACHE-YIELD evicted=%d tokens: P waits and D has no running or waiting request "
                    "(backed prefixes stay in L2) mapped=%d", MARK, ev, actor.mapped_tokens)
    return ev


def _arena_census() -> Tuple[int, int]:
    """(COMPLETE slots, slots) of the shared Mamba-anchor arena this process
    reads through its registered host pools (the anchor pool; else the first
    arena); (0, 0) without one."""
    from sglang.srt.weg2 import handoff_pending as _hp

    pools = [p for p in (r() for r in list(_hp._POOLS)) if p is not None and getattr(p, "arena", None) is not None]
    pools.sort(key=lambda p: getattr(p, _hp.ROLE_ATTR, "kv") != "anchor")
    for pool in pools:
        arena = pool.arena
        if hasattr(arena, "ref_census"):
            _pinned, _refs, complete = arena.ref_census()
            return int(complete), int(arena.slots)
    return 0, 0


def publish_d_signal(sched, actor) -> None:
    """Q-660: D's shortage beyond the ledger bytes -- its id space and the Mamba
    arena -- for the front's stages (dual_d_priority.d_signal_short). TP0 only,
    at most every D_SIGNAL_EVERY_S; an instrument, never raises."""
    from sglang.srt.weg2 import dual_d_priority as _ddp

    if int(getattr(sched, "tp_rank", 0) or 0) != 0:
        return
    t = _pk._now()
    if t < float(getattr(actor, "_dsig_next", 0.0) or 0.0):
        return
    actor._dsig_next = t + _ddp.D_SIGNAL_EVERY_S
    try:
        size = int(actor.allocator.size)
        tree = getattr(sched, "tree_cache", None)
        evictable = int(tree.evictable_size() or 0) if tree is not None else 0
        used = size - (int(actor.allocator.available_size()) + evictable)
        complete, slots = _arena_census()
        tag = os.environ.get("SGLANG_WEG2_DUAL_KV_TAG", "") or os.environ.get("SGLANG_WEG2_TAG", "weg2")
        _ddp.publish_d_signal(_ddp.d_signal_file(tag), id_frac=used / max(1, size), arena_complete=complete,
                              arena_slots=slots, now=time.time())
    except Exception as exc:  # noqa: BLE001 -- the front then sees no D signal: the ledger demand still counts
        logger.info("%s D-SIGNAL not published: %r", MARK, exc)


def _note_shrink_blocked(actor, reason: Optional[str], need: int, floor: int, p_wait_s: float) -> None:
    """Q-696: name why D keeps rows it does not need while P waits for a card
    (rate-limited, the gap doubling to 60 s; reset once nothing is blocked)."""
    if reason is None:
        actor._blocked_log_iv = 0.5
        return
    t = _pk._now()
    if t < float(getattr(actor, "_blocked_log_next", 0.0) or 0.0):
        return
    actor._blocked_log_iv = min(60.0, 2.0 * float(getattr(actor, "_blocked_log_iv", 0.5) or 0.5))
    actor._blocked_log_next = t + actor._blocked_log_iv
    logger.info("%s %s reason=%s mapped=%d need=%d live_floor=%d p_wait_s=%.1f: P waits for a card and D "
                "maps more than its work needs, the shrink cannot go now (Q-696)", MARK, SHRINK_BLOCKED_MARK,
                reason, actor.mapped_tokens, int(need), int(floor), float(p_wait_s))


def tick(sched) -> Optional[str]:
    """Once per scheduler iteration on every D rank. ONE collective per tick
    (MAX of want, P-waiting and the highest live row), so every rank decides on
    the SAME numbers -- metal bsffsv: a local view let TP1/TP2 shrink while TP0
    held, and the collective of the shrink path then met ranks that were not
    in it."""
    runner = getattr(getattr(sched, "tp_worker", None), "model_runner", None)
    actor = getattr(runner, ACTOR_ATTR, None)
    if actor is None:
        return None
    from sglang.srt.weg2 import d_seat_vram as _sv
    from sglang.srt.weg2.card_kv_ledger import peek

    actor.gmin = getattr(sched, "_weg2_group_min_ints", None) or actor.gmin
    demand_local = d_demand(sched)
    air = int(_sv._air(sched))
    want_local = want_local_tokens(demand_local, d_locked_rows(sched, actor), air, actor.step)
    st = peek(actor.ledger.path)
    p_wait_local = 1 if (st is not None and int(st.demand.get("P", 0)) > 0) else 0
    live_local = int(_pk.max_live_id(actor.allocator, actor.page)) * int(actor.page)
    # metal dual14: D unmapped its boot pool before P sized its KV on the card,
    # P counted those bytes as free -> the budget held them twice. D keeps its
    # boot pool until P has JOINED every D card (group decision).
    p_missing_local = 1 if (st is None or not int(st.pid.get("P", 0) or 0)) else 0
    # Q-696: how long THIS rank's card has shown P waiting, and whether this rank
    # grew within the regrow hold -- both as group MAX on the same collective, so
    # the live yield and the hold are decided on the same numbers on every rank
    t_now = _pk._now()
    if p_wait_local:
        if getattr(actor, "_p_wait_since", None) is None:
            actor._p_wait_since = t_now
        p_wait_ms_local = int(1000.0 * (t_now - actor._p_wait_since))
    else:
        actor._p_wait_since, p_wait_ms_local = None, 0
    grow_t = getattr(actor, "_grow_t", None)
    recent_grow_local = 1 if (grow_t is not None and t_now - float(grow_t) < regrow_hold_s()) else 0
    # Q-1190: a claim the SHARED KV arena refused (P's hand-off / write-through, or D's own
    # backup) left its page need next to the arena; TP0 takes it, the MAX rides this
    # collective, every D rank yields the same pages of its host-only leaves
    from sglang.srt.weg2 import dual_arena_spill as _das

    arena_need_local = _das.d_take_need(sched) if int(getattr(sched, "tp_rank", 0) or 0) == 0 else 0
    # D PRIORITY: the tightest rank's free rows (MIN) and the holds (MAX) ride the
    # SAME collective -- every rank decides on the same numbers
    g = actor.gmin([-int(want_local), -int(p_wait_local), -int(live_local), -int(demand_local),
                    -int(p_missing_local), int(d_avail_rows(sched, actor)), -int(d_holds(sched)),
                    -int(p_wait_ms_local), -int(recent_grow_local), -int(arena_need_local)])
    arena_need = max(0, -int(g[9])) if len(g) > 9 else 0
    if arena_need > 0:
        _das.d_yield_arena(sched, arena_need)
    want, p_waiting, floor = -int(g[0]), -int(g[1]) > 0, -int(g[2])
    p_missing = len(g) > 4 and -int(g[4]) > 0
    avail_min = int(g[5]) if len(g) > 5 else NO_AVAIL
    holds = len(g) > 6 and -int(g[6]) > 0
    p_wait_s = max(0, -int(g[7])) / 1000.0 if (len(g) > 7 and p_waiting) else 0.0
    recent_grow = len(g) > 8 and -int(g[8]) > 0
    want = floor_want(want, actor.mapped_tokens, avail_min, air, actor.step, -int(g[3]))
    need, freed = want, 0
    live_due = p_waiting and live_yield_due(p_wait_s, live_yield_wait_s())
    if p_waiting and -int(g[3]) == 0:
        # the GROUP has no running or waiting request and P waits: D's cached
        # prefix is not demand (user rule) -- every rank gives it up alike, the
        # next tick's live floor lets the shrink through
        cache_yield(sched, actor)
    elif live_due and not holds:
        # Q-696: D runs work, P has waited for a card past the bound -- the cache
        # is still not demand. Unlocked nodes only (a running seat's rows are
        # locked); a held context (D-PARK, W50, D-HOLD-FOR-GROW) keeps its rows.
        freed = cache_yield(sched, actor, live=(floor, p_wait_s))
    want = max(want, _pk.round_up(floor, actor.step))       # never below a live row of any rank
    below = actor._below + 1 if want < actor.mapped_tokens else 0
    # the room rule of decide() guards running, queued or held work; an idle D's
    # level is its look-ahead already (the cache_yield path shrinks it to that)
    guarded = (-int(g[3]) > 0 or holds) and avail_min < NO_AVAIL
    verdict, level = decide(actor.mapped_tokens, want, p_waiting, below, actor.step,
                            avail_min=avail_min if guarded else None, air=air, holds=holds)
    if verdict == "shrink" and p_missing:
        verdict, level = "hold", actor.mapped_tokens
    if verdict == "shrink" and recent_grow:
        # Q-696 REGROW HOLD: a shrink right after a grow is the flap (dual y8z
        # 19:26:04 GROW 110592->196608, SHRINK back, GROUP-WAIT for 196608 again
        # in the same second -> pressure on P -> a second P pause)
        verdict, level = "hold", actor.mapped_tokens
    if live_due and verdict != "shrink" and not freed:     # a yield tick's floor predates its eviction
        _note_shrink_blocked(actor, shrink_blocked_reason(
            mapped=actor.mapped_tokens, need=need, floor=floor, step=actor.step, holds=holds,
            p_missing=p_missing, regrow_hold=recent_grow), need, floor, p_wait_s)
    if verdict != "grow" and st is not None and (int(st.pressure.get("P", 0) or 0) > 0
                                                 or int(st.demand.get("D", 0) or 0) > 0):
        # D's demand fits what it maps (seats ended, aborted or shrunk): the pressure
        # on P goes with it (metal dual20: it stood 4 min after the L seats were gone)
        actor.ledger.clear_pressure()
    _pk.phys_check(actor, "D")
    publish_d_signal(sched, actor)                          # Q-660: id space + arena for the front
    actor._below = below
    if verdict == "grow":
        actor.group_grow(level)
    elif verdict == "shrink":
        if actor.group_shrink(level, floor):
            actor._below = 0
    return verdict
