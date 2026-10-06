"""L15-HOSTLOCK-NAMED (06.10., boot ..._2b8ea5bf5a_1006_050049, L15-CHECK REFUSED
epoch 66): the sleep's HOSTLOCK pin must survive the same sleep's reset.

Metal: ``L15-HOSTLOCK kv-sleep-hold +1 slots=129792 took=129792`` and, in the
same second, ``#1424g ARENA-REF RESET-ORPHANS at=reset pool=FULL
released=129792 slots=129792`` -- the pin's reset (``reset_keep`` ->
``_reset_full`` -> ``_weg2_release_orphan_refs``) gave every reference back,
because the record lived only on ``sched._l15_host_hold`` and no holder named
it. The wake's release then took 0 / 44 / 59 (102 wakes), and the first epoch
with a FULL arena (e66) let P's write-through recycle the held L2 slots ->
the wake sample check found 17/7/20 bad rows -> fallback.

Hermetic like test_arena_reset_orphans_1424g: the real C arena with the
per-process ledger, the real ``UnifiedRadixCache._reset_full`` on a tree shell,
the real ``l15_hostlock`` pin/release and the real wake release
(``SchedulerWeightUpdaterManager._l15_release_host_hold_refs``); the balance
is read from the slot headers and the ledger."""
from __future__ import annotations

import ctypes
import logging
import os
import queue as _queue
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.managers.scheduler_components.weight_updater import (  # noqa: E402
    SchedulerWeightUpdaterManager as WU,
)
from sglang.srt.mem_cache.pool_host import arena_pool as ap  # noqa: E402
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena  # noqa: E402
from sglang.srt.weg2 import l15_hostlock  # noqa: E402

P = 4
SLOTS = 16
SLOT_BYTES = 256
S = 5

HELD = ["h0", "h1", "h2", "h3", "h4", "h5"]      # the held chain's L2 pages
OTHER = ["o%d" % i for i in range(SLOTS - len(HELD))]


@pytest.fixture
def arena(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_QUEUE_REFS", "1")
    a = ShmArena(str(tmp_path / "arena-kv-1006.bin"), SLOT_BYTES, SLOTS)
    yield a
    a.close()


def _publish(arena, stems):
    out = {}
    for stem in stems:
        (s, st, g), = arena.claim_slots([stem], [SLOT_BYTES])
        assert st == 0, (stem, st)
        assert arena.complete_slots([s], [g], [(0, SLOT_BYTES)]) == [1]
        out[stem] = (s, g)
    return out


def _refs(arena):
    out = (ctypes.c_int64 * 6)()
    arena._lib.arena_layout(arena.slots, arena.slot_bytes, out)
    hb, hoff = int(out[0]), int(out[3])
    u32 = np.frombuffer(arena._mm, dtype=np.uint32, count=arena.slots * hb // 4, offset=hoff)
    return u32.reshape(arena.slots, hb // 4)[:, 1].astype(np.int64).tolist()


def _own(arena):
    return int(arena._ledger.held.sum())


def _pool(arena):
    pool = object.__new__(ap.ArenaMHAHostPool)
    pool._arena_page_tokens = P
    pool.staging_rows = S
    pool.arena = arena
    pool.arena_tokens = SLOTS * P
    pool.arena_slots = SLOTS
    pool.row_slot = None
    pool._page_bytes = SLOT_BYTES
    pool._backend = types.SimpleNamespace(_get_suffixed_key=lambda k: k)
    pool._pending = {}
    pool._pending_mask = torch.zeros(SLOTS, dtype=torch.bool)
    pool._pending_gen = torch.zeros(SLOTS, dtype=torch.int64)
    pool._pending_fresh = torch.zeros(SLOTS, dtype=torch.bool)
    return pool


class _Controller:
    def __init__(self, pool):
        self.enable_storage = True
        self.host_mem_release_queue = _queue.Queue()
        self.extra_host_mem_release_queues = {}
        # live_host_pools unwraps the group's "kv" entry (HostPoolGroup shape)
        self.mem_pool_host = types.SimpleNamespace(
            arena_read=True, clear=lambda: None,
            entry_map={"kv": types.SimpleNamespace(host_pool=pool)})

    def entry_for_extra_release(self, name):
        return None

    def reset(self):
        self.host_mem_release_queue.queue.clear()


def _tree(pool):
    from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    FULL = ComponentType.FULL
    t = object.__new__(UnifiedRadixCache)
    t._components_tuple = (types.SimpleNamespace(component_type=FULL, _full_kv_pool_host=pool),)
    t.tree_components = (FULL,)
    t.cache_controller = _Controller(pool)
    t.ongoing_prefetch = {}
    t._retired_prefetch = []
    t._weg2_carrier_rows = {}
    t.session = types.SimpleNamespace(slots={})
    t.device = "cpu"
    t._drop_staging_write_ring = lambda: None
    t._init_pin_trace = lambda: None
    t._record_all_cleared_event = lambda: None
    t._weg2_orphan_sweep_armed = True
    t.root_node = types.SimpleNamespace(children={})
    return t


class _Env:
    """One D rank's sleep: a full arena (every page COMPLETE, unreferenced --
    P's earlier write-through), a held chain, the tree and the scheduler."""

    def __init__(self, arena):
        self.arena = arena
        self.slot_of = _publish(arena, HELD + OTHER)       # arena FULL
        self.pool = _pool(arena)
        self.tree = _tree(self.pool)
        self.sched = types.SimpleNamespace(tree_cache=self.tree, _l15_host_hold=None)
        self.logs = []
        self.held_slots = [self.slot_of[k][0] for k in HELD]

    def sleep(self, named=True):
        """retain step (hostlock) -> sink -> reset_keep's _reset_full, the
        scheduler hook's wiring (put = record + name on the tree)."""
        from sglang.srt.weg2 import l15_bind

        sched, tree, log = self.sched, self.tree, self.logs.append

        def put(rec):
            sched._l15_host_hold = rec
            if named:
                l15_hostlock.name_hold_on_tree(
                    tree, l15_bind.live_host_pools(tree), rec, log)

        sink = l15_hostlock.rearm_sink(
            get=lambda: sched._l15_host_hold, put=put,
            pools=lambda: l15_bind.live_host_pools(tree), log=log)
        rec = l15_hostlock.hold_sleep_refs(self.pool, None, self.held_slots, [], log)
        assert rec is not None
        sink(rec)
        tree._reset_full()

    def p_phase_claims(self):
        """P's write-through while D sleeps: one new page per call. The arena
        is full, so every claim first asks the pool's room-making
        (``_evict_for_claim``: the oldest UNREFERENCED complete slots) and
        retries -- the metal order of ArenaMHAHostPool._claim_np."""
        fresh = refused = 0
        for i in range(SLOTS):
            stem = "p%d" % i
            (s, st, g), = self.arena.claim_slots([stem], [SLOT_BYTES])
            if st == 4:
                self.pool._evict_for_claim(self.arena, 1)
                (s, st, g), = self.arena.claim_slots([stem], [SLOT_BYTES])
            if st == 0:
                fresh += 1
                assert self.arena.complete_slots([s], [g], [(0, SLOT_BYTES)]) == [1]
                # P's tree holds what it just wrote (a reader reference of
                # ANOTHER process: header only, not this ledger) -- else its
                # own pages would be the next victims
                c = (ctypes.c_int64 * 1)(s)
                assert self.arena._lib.arena_ref_slots(self.arena._base, 1, c, 1) == 1
            else:
                refused += 1
        return fresh, refused

    def gens_now(self):
        cs, cg, _lo, _hi = self.arena.complete_census()
        return dict(zip(cs.tolist(), cg.tolist()))

    def wake_release(self):
        return WU._l15_release_host_hold_refs(types.SimpleNamespace(), self.sched)


def test_pin_survives_the_sleeps_own_reset_and_is_released_at_the_wake(arena):
    """Base: RESET-ORPHANS gave the 6 pins back inside the sleep (refs 0, wake
    release took 0). Fixed: the pins stand through the reset, the census names
    them (gap 0), and the wake's release takes exactly the slots it pinned."""
    e = _Env(arena)
    e.sleep()
    refs = _refs(arena)
    assert [refs[s] for s in e.held_slots] == [1] * len(HELD), refs
    assert _own(arena) == len(HELD)
    line = e.tree.weg2_arena_holder_census(arena)
    assert "l15_hold=6 " in line and "gap=0" in line, line
    assert e.wake_release() == len(HELD)
    assert _refs(arena) == [0] * SLOTS and _own(arena) == 0
    assert arena._ledger.refused == 0
    assert e.sched._l15_host_hold is None and e.tree._weg2_l15_hold == []


def test_full_arena_p_phase_cannot_recycle_the_held_slots(arena):
    """Arena-voll (the e66 shape): every page COMPLETE, P claims a page per
    call. The held slots keep their generation (the wake's L2 source is intact)
    and P is refused only beyond the unpinned part -- no reserve, just the pin.
    Base: all 16 claims succeed, the held slots are recycled (gen bump)."""
    e = _Env(arena)
    gens0 = e.gens_now()
    e.sleep()
    fresh, refused = e.p_phase_claims()
    assert fresh == SLOTS - len(HELD) and refused == len(HELD), (fresh, refused)
    gens1 = e.gens_now()
    for s in e.held_slots:
        assert gens1.get(s) == gens0[s], (s, gens0.get(s), gens1.get(s))
    assert e.wake_release() == len(HELD)
    assert _own(arena) == 0 and arena._ledger.refused == 0


def test_without_the_name_the_orphan_pass_gives_the_pin_back(arena):
    """The defect, kept as the mutant's witness (naming switched off = the
    base wiring): the pins are gone after the sleep, the wake release takes
    nothing, and P recycles every held slot."""
    e = _Env(arena)
    gens0 = e.gens_now()
    e.sleep(named=False)
    assert _own(arena) == 0 and [_refs(arena)[s] for s in e.held_slots] == [0] * len(HELD)
    assert e.wake_release() == 0
    fresh, refused = e.p_phase_claims()
    assert fresh == SLOTS and refused == 0
    assert any(e.gens_now().get(s) != gens0[s] for s in e.held_slots)


def test_fallback_drop_shape_reset_then_release(arena):
    """The fallback act resets the tree BEFORE it releases: the pins must stand
    through that reset too, then go back once (took == slots)."""
    e = _Env(arena)
    e.sleep()
    e.tree._reset_full()                      # _l15_fallback_drop -> tree_cache.reset()
    assert _own(arena) == len(HELD)
    assert e.wake_release() == len(HELD)
    assert _own(arena) == 0 and arena._ledger.refused == 0


def test_second_retain_in_one_sleep_keeps_one_pin_per_slot(arena):
    """rearm_sink: the second retain pins again, names the new record, then
    releases the old one -- one reference per slot stands through BOTH resets
    (the dense count never dips to 0), and the wake gives back the last one."""
    e = _Env(arena)
    e.sleep()
    e.sleep()
    assert [_refs(arena)[s] for s in e.held_slots] == [1] * len(HELD)
    assert _own(arena) == len(HELD)
    assert e.wake_release() == len(HELD)
    assert _own(arena) == 0 and arena._ledger.refused == 0


def test_after_the_wake_the_name_is_gone_and_a_later_orphan_is_swept(arena):
    """A name left behind would shield a LATER reference on the same slot from
    the orphan pass: after the wake release the same slot, referenced by
    nobody's class, is given back by the next reset again."""
    e = _Env(arena)
    e.sleep()
    e.wake_release()
    s = e.held_slots[0]
    assert arena.ref_slots([s], +1) == 1
    e.tree._reset_full()
    assert _refs(arena)[s] == 0 and _own(arena) == 0


def test_other_orphans_are_still_swept_while_the_pin_stands(arena):
    """The name covers the pin only: an unrelated unnamed reference on another
    slot is still given back by the same reset (#1424g unchanged)."""
    e = _Env(arena)
    other = e.slot_of["o0"][0]
    assert arena.ref_slots([other], +1) == 1
    e.sleep()
    refs = _refs(arena)
    assert refs[other] == 0 and [refs[s] for s in e.held_slots] == [1] * len(HELD)
    assert e.wake_release() == len(HELD)


def test_a_foreign_pin_on_the_same_slot_is_not_taken(arena):
    """P's own reader reference on a held slot (another process, header only)
    stays; the pin + the name are this process's one reference."""
    e = _Env(arena)
    s = e.held_slots[0]
    c = (ctypes.c_int64 * 1)(s)
    assert arena._lib.arena_ref_slots(arena._base, 1, c, 1) == 1
    e.sleep()
    assert _refs(arena)[s] == 2
    assert e.wake_release() == len(HELD)
    assert _refs(arena)[s] == 1


def test_tree_without_the_api_names_nothing():
    """A stub tree (tests of other paths, no UnifiedRadixCache): no name, no
    raise -- today's behaviour."""
    assert l15_hostlock.name_hold_on_tree(types.SimpleNamespace(), (None, None),
                                          ((1, 2), ()), lambda m: None) == 0
    assert l15_hostlock.clear_hold_on_tree(None) == 0


def test_scheduler_sink_wiring_names_the_hold():
    """The sleep hook's put must name the tree (the one place the record is
    stored): guards the wiring the hermetic tests above simulate."""
    import pathlib

    src = pathlib.Path(__import__("sglang").__path__[0], "srt", "managers", "scheduler.py").read_text()
    i = src.index("hold_sink=l15_hostlock.rearm_sink(")
    block = src[i:i + 1400]
    assert "l15_hostlock.name_hold_on_tree(" in block, block
    wu = pathlib.Path(__import__("sglang").__path__[0], "srt", "managers", "scheduler_components",
                      "weight_updater.py").read_text()
    j = wu.index("def _l15_release_host_hold_refs")
    assert "clear_hold_on_tree(" in wu[j:j + 2500]
