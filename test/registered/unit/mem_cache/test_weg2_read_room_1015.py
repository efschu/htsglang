# SPDX-License-Identifier: Apache-2.0
"""READ-ROOM 1015: an L3 -> L2 read never ends at a FULL arena that holds L3-backed tree copies.

MEASURED (boots ..._226aad7105_1015_{115748,122948,124245,133855}, group P, PP0 ``WEG2-READ-DEPTH``):
46 ``cause=arena_full`` cuts. weg2-4-25 (124245, 12:52:35): the store held 1568 pages of the 100423-token
prompt, ``WEG2-READ-STAGES pages=54 l3fill_pages=977``, ``L3-FILL ROOM-WAIT stems=970 claimed=0 still_full=970
rounds=102 waited_ms=5025`` on all three stages, ``ARENA-REF-HOLDERS tree=6478 tree_in_use=0 prefetch=54
arena_pinned=6485 arena_complete=6485`` (PP1, 12:52:39), ``#257 PREFETCH BELOW-ANCHOR read=3456 of 100352
anchored=3008`` -> 97408 tokens prefilled again although every page was on the L3 disk.

The fill's only room-maker is the arena clock (COMPLETE slots no reference holds); the trees' spill
(``_w3_arena_spill``) was reachable from write-back claims only. Driven here through the REAL C arena and the
REAL ``HiCacheFile.arena_fill_from_disk`` on a prefetch-io-named thread, with a scheduler-side tree stand-in
whose spill gives references back like the real one (and the real ``_w3_arena_spill`` through ``serve``).
RED on 226aad7105 (no ``weg2/read_room.py``: the read ends short), GREEN with it.
"""

import logging
import os
import shutil
import threading
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import HiCacheFile
from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena
from sglang.srt.weg2 import pp_slot_fidelity as SF

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

TOTAL = 4096
IO = "hicache-prefetch-io-0"


class _NoRoom:
    """The BASE tree (226aad7105) has no ``weg2/read_room.py``: a stand-in that does nothing, so the scenarios run the
    base behaviour (the fill never asks, nobody serves) and go red by BEHAVIOUR, not by an ImportError."""

    @staticmethod
    def tick(cache):
        return 0

    @staticmethod
    def reset_for_tests():
        return None


def _rr():
    try:
        from sglang.srt.weg2 import read_room
    except ImportError:
        return _NoRoom
    return read_room


def _armed(arena):
    """The state ``tick`` leaves after a round on a local-PP tree: armed, and this arena resolved as the one the
    scheduler serves (no scheduler thread in the scenarios that drive ``post`` / ``_unseen`` by hand)."""
    rr = _rr()
    rr._ST.armed = True
    rr._ST.arena = arena


def _ov(name, value):
    """``envs.<name>.override(value)``; a no-op on the base tree (the switch does not exist there)."""
    import contextlib

    field = getattr(envs, name, None)
    return field.override(value) if field is not None else contextlib.nullcontext()


def _backend(root):
    be = object.__new__(HiCacheFile)

    def _path(stem):
        return os.path.join(root, stem + ".bin")

    def _clock(arena, want, need=None):
        # the arena clock: COMPLETE slots NO reference holds go (no disk write needed in the stand-in)
        got = arena.evict_candidates(int(want))
        if got:
            arena.free_evicted([c[0] for c in got], reason="test_clock")
        return len(got)

    be._existing_path = _path
    be._stat_stems = lambda stems: {s: os.path.getsize(_path(s)) for s in stems if os.path.exists(_path(s))}
    be._arena_evict_to_disk = _clock
    return be


class _TreeSim:
    """One PP rank's tree: it holds a reference on ``held`` slots (host copies of finished contexts,
    tree_in_use=0); its spill hands the references of the first ``max(need, 64)`` back, like ``_w3_arena_spill``
    (L3 copy first, then the host copy goes)."""

    def __init__(self, arena, held):
        self.arena = arena
        self.held = list(held)
        self.spills = []
        setattr(self, SF.FLOOR_LOCAL_PP_ATTR, True)

    def _weg2_direct_pool(self):
        return types.SimpleNamespace(arena=self.arena)

    def _w3_arena_spill(self, pool, need_pages, claimer=None):
        take = self.held[: max(int(need_pages), 64)]
        del self.held[: len(take)]
        if take:
            self.arena.ref_slots(take, -1)
        self.spills.append(len(take))
        return len(take)


def _fill_arena_with_tree_pages(arena, n, tag="old"):
    """n COMPLETE pages, each carrying one tree reference (the P tree's host copy of a finished context)."""
    got = arena.claim_slots([f"{tag}{i}" for i in range(n)], [TOTAL] * n)
    assert all(st == 0 for _s, st, _g in got)
    slots = [s for s, _st, _g in got]
    assert arena.complete_slots(slots, [g for _s, _st, g in got], [(0, TOTAL)]) == [1] * n
    arena.ref_slots(slots, +1)
    return slots


def _setup(tmp_path, stems, slots):
    root = tmp_path / "store"
    root.mkdir()
    for s in stems:
        (root / (s + ".bin")).write_bytes(b"\x22" * TOTAL)
    arena = ShmArena(str(tmp_path / "kv.bin"), TOTAL, slots)
    return _backend(str(root)), arena


def _run_on(thread_name, fn):
    box = {}
    th = threading.Thread(target=lambda: box.setdefault("r", fn()), name=thread_name)
    th.start()
    th.join(60)
    return box["r"]


class _Scheduler:
    """The scheduler thread of one rank: ``check_hicache_events`` calls ``tick`` once per round."""

    def __init__(self, tree, period=0.02):
        self.tree, self.period, self.stop = tree, period, threading.Event()
        self.th = threading.Thread(target=self._run, name="scheduler")

    def _run(self):
        while not self.stop.is_set():
            _rr().tick(self.tree)
            time.sleep(self.period)

    def __enter__(self):
        self.th.start()
        time.sleep(0.1)       # the first round arms the post
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.th.join(5)


@pytest.fixture(autouse=True)
def _clean():
    _rr().reset_for_tests()
    with envs.SGLANG_WEG2_L3FILL_ROOM_WAIT_MS.override(300), \
            _ov("SGLANG_WEG2_READ_ROOM_WAIT_MS", 4000), \
            envs.SGLANG_WEG2_L3FILL_SELF_JOIN_TAKEOVER.override(False):
        yield
    _rr().reset_for_tests()


def _read(be, arena, stems):
    return _run_on(IO, lambda: HiCacheFile.arena_fill_from_disk(
        be, arena, stems, TOTAL, prefix=True, take_ref=True))


# --- the cases ---------------------------------------------------------------------------------------

def test_weg2_4_25_a_pinned_arena_gives_the_room_the_read_needs(tmp_path):
    """The 12:52:35 shape, scaled 1:100: arena full of tree-held pages (tree=all, tree_in_use=0), a read of
    ~1/6 of the arena. RED before: nothing lands (claimed=0 for the whole wait). GREEN: every page lands."""
    stems = [f"r{i}" for i in range(16)]
    be, arena = _setup(tmp_path, stems, slots=64)
    held = _fill_arena_with_tree_pages(arena, 64)
    tree = _TreeSim(arena, held)
    with _Scheduler(tree):
        out = _read(be, arena, stems)
    landed = sum(1 for s in out if s is not None)
    assert landed == len(stems), (landed, tree.spills)
    assert tree.spills, "the scheduler was asked for room"
    assert _rr().census(arena)["claimed"] == 0


def test_the_ratio_of_the_boot_977_stems_against_6485_slots_is_a_deficit_of_970():
    """The census arithmetic of 12:52:35: free 0, clock 0 (every COMPLETE slot referenced) -> the whole need."""
    class A:
        path = "x"
        def stats(self):
            return {"slots": 6485, "complete": 6485, "claimed": 0}
        def ref_census(self):
            return (6485, 16376, 6485)

    d, c = _rr().deficit(A(), 970)
    assert d == 970 and c["free"] == 0 and c["clock"] == 0


def test_the_clock_and_free_slots_count_as_room_before_the_trees_are_asked():
    class A:
        path = "x"
        def stats(self):
            return {"slots": 100, "complete": 90, "claimed": 4}
        def ref_census(self):
            return (60, 180, 90)       # 30 COMPLETE unreferenced, 6 free

    d, c = _rr().deficit(A(), 50)
    assert (c["free"], c["clock"], d) == (6, 30, 14)


def test_without_a_scheduler_the_read_is_cut_and_the_cut_is_named(tmp_path, caplog):
    """Nobody serves (no scheduler round in the wait): the old cut, but never silent -- ``READ-ROOM SHORT``."""
    stems = [f"r{i}" for i in range(8)]
    be, arena = _setup(tmp_path, stems, slots=16)
    _fill_arena_with_tree_pages(arena, 16)
    tree = _TreeSim(arena, [])
    _armed(arena)                   # armed by an earlier round, but no round runs now
    with _ov("SGLANG_WEG2_READ_ROOM_WAIT_MS", 300), caplog.at_level(logging.INFO):
        out = _read(be, arena, stems)
    assert all(s is None for s in out)
    line = [r.getMessage() for r in caplog.records if "READ-ROOM SHORT" in r.getMessage()]
    assert line and "class=unserved" in line[-1] and "pinned=16" in line[-1], line
    assert not tree.spills


def test_live_reads_that_nothing_can_spill_end_named_as_nothing_to_spill(tmp_path, caplog):
    """Everything pinned by LIVE holders (the stand-in tree holds nothing): the spill answers 0 -- the cut is
    classified ``served_nothing_to_spill``, the physical case the order allows a named stop for."""
    stems = [f"r{i}" for i in range(8)]
    be, arena = _setup(tmp_path, stems, slots=16)
    _fill_arena_with_tree_pages(arena, 16)       # referenced, but by nobody this tree can release
    tree = _TreeSim(arena, [])
    with _Scheduler(tree), _ov("SGLANG_WEG2_READ_ROOM_WAIT_MS", 500), caplog.at_level(logging.INFO):
        out = _read(be, arena, stems)
    assert all(s is None for s in out)
    msgs = [r.getMessage() for r in caplog.records]
    short = [m for m in msgs if "READ-ROOM SHORT" in m]
    assert short and "served_nothing_to_spill" in short[-1], short
    assert any("READ-ROOM SERVE" in m and "released_pages=0" in m for m in msgs)


def test_the_request_reaches_a_peer_rank_through_the_shared_file(tmp_path):
    """PP0 posts, PP1's scheduler (another process: its own 'seen') serves the same request -- a slot is free
    only when EVERY rank has let go."""
    be, arena = _setup(tmp_path, ["r0"], slots=16)
    _fill_arena_with_tree_pages(arena, 16)
    rr = _rr()
    _armed(arena)
    assert rr.post(arena, 10, site="fill")
    assert rr._unseen(arena) == 10
    assert rr._unseen(arena) == 0, "served once per rank"
    rr._ST.seen.clear()                 # the peer process has not seen it
    assert rr._unseen(arena) == 10


def test_two_ranks_whose_pids_collide_modulo_64_do_not_share_a_slot(tmp_path):
    """PP0 (pid 5) and PP2 (pid 69) would both have taken slot 5 under pid % 64: the later post overwrote the
    earlier one and a rank's room request vanished. A poster owns the slot that carries its pid."""
    import mmap

    rr = _rr()
    m = mmap.mmap(-1, rr._SIZE)
    a, _ = rr._pick_slot(m, 5, 100)
    m[a * rr._REC.size:(a + 1) * rr._REC.size] = rr._REC.pack(1, 100, rr._now_ms(), 5)
    b, _ = rr._pick_slot(m, 69, 50)
    assert a != b
    m[b * rr._REC.size:(b + 1) * rr._REC.size] = rr._REC.pack(1, 50, rr._now_ms(), 69)
    assert rr._pick_slot(m, 5, 10)[0] == a, "the same process keeps its slot"
    assert rr._pick_slot(m, 5, 10)[1] == 100, "a second post within 3 s keeps the larger need"


def test_a_stale_request_is_not_served(tmp_path):
    be, arena = _setup(tmp_path, ["r0"], slots=16)
    rr = _rr()
    _armed(arena)
    assert rr.post(arena, 10)
    rr._ST.seen.clear()
    real = rr._now_ms
    try:
        rr._now_ms = lambda: real() + rr._STALE_MS + 1000
        assert rr._unseen(arena) == 0
    finally:
        rr._now_ms = real


def test_re_posts_are_paced(tmp_path):
    be, arena = _setup(tmp_path, ["r0"], slots=16)
    rr = _rr()
    _armed(arena)
    assert rr.post(arena, 10)
    assert not rr.post(arena, 10), "paced by SGLANG_WEG2_READ_ROOM_REPOST_MS"


def test_switch_off_is_the_old_fill(tmp_path):
    """SGLANG_WEG2_READ_ROOM=0: nothing posted, the scheduler never spills, the read ends short as before."""
    stems = [f"r{i}" for i in range(8)]
    be, arena = _setup(tmp_path, stems, slots=16)
    held = _fill_arena_with_tree_pages(arena, 16)
    tree = _TreeSim(arena, held)
    with _ov("SGLANG_WEG2_READ_ROOM", False), _Scheduler(tree):
        out = _read(be, arena, stems)
    assert all(s is None for s in out)
    assert not tree.spills and _rr()._ST.posted == 0


def test_a_tp_group_is_not_armed_and_keeps_the_old_wait(tmp_path):
    """Replicas never edit rank-locally: a tree without the local-PP floor serves nothing, posts nothing."""
    be, arena = _setup(tmp_path, ["r0"], slots=16)
    tree = _TreeSim(arena, [])
    setattr(tree, SF.FLOOR_LOCAL_PP_ATTR, False)
    _rr().tick(tree)
    assert not _rr()._ST.armed
    assert not _rr().post(arena, 5)
    assert _rr().wait_idle_ms(5000, arena) == 5000


def test_the_room_wait_idle_bound_is_longer_than_a_scheduler_round_when_armed(tmp_path):
    be, arena = _setup(tmp_path, ["r0"], slots=16)
    rr = _rr()
    _armed(arena)
    with _ov("SGLANG_WEG2_READ_ROOM_WAIT_MS", 10000):
        assert rr.wait_idle_ms(5000, arena) == 10000
    rr._ST.armed = False
    assert rr.wait_idle_ms(5000, arena) == 5000


def test_the_mamba_anchor_arena_is_not_served_and_keeps_the_old_wait(tmp_path):
    """A second arena of the same process (the 32-slot mamba anchors) is not the tree's KV arena the scheduler
    serves: a post there would wait 10 s for an answer nobody gives. It posts nothing and keeps the old bound."""
    be, kv = _setup(tmp_path, ["r0"], slots=16)
    other = ShmArena(str(tmp_path / "mamba.bin"), TOTAL, 4)
    rr = _rr()
    _armed(kv)
    assert rr.post(kv, 3) and not rr.post(other, 3)
    with _ov("SGLANG_WEG2_READ_ROOM_WAIT_MS", 10000):
        assert rr.wait_idle_ms(5000, kv) == 10000 and rr.wait_idle_ms(5000, other) == 5000
    assert not os.path.exists(str(tmp_path / "mamba.bin") + ".room")


def test_two_readers_one_arena_both_complete(tmp_path):
    """weg2-8-29 / 8-30 / 4-25 asked in the same seconds: two fills compete for the room one spill makes."""
    a = [f"a{i}" for i in range(12)]
    b = [f"b{i}" for i in range(12)]
    be, arena = _setup(tmp_path, a + b, slots=32)
    held = _fill_arena_with_tree_pages(arena, 32)
    tree = _TreeSim(arena, held)
    res = {}

    def one(name, stems):
        res[name] = _read(be, arena, stems)

    with _Scheduler(tree):
        ts = [threading.Thread(target=one, args=(n, s)) for n, s in (("a", a), ("b", b))]
        for t in ts:
            t.start()
        for t in ts:
            t.join(60)
    landed = {n: sum(1 for s in r if s is not None) for n, r in res.items()}
    assert landed == {"a": 12, "b": 12}, landed


def test_a_slow_scheduler_round_is_waited_for(tmp_path):
    """A P stage's scheduler round is a 2-3 s chunk forward: the old 5 s idle bound of the room wait is shorter
    than a round of THREE stages. Scaled: the room wait's own bound 300 ms, the scheduler answers after 0.8 s --
    the armed bound (4 s) waits for it, the old bound would have cut the read."""
    stems = [f"r{i}" for i in range(8)]
    be, arena = _setup(tmp_path, stems, slots=16)
    held = _fill_arena_with_tree_pages(arena, 16)
    tree = _TreeSim(arena, held)
    with _Scheduler(tree, period=0.8):
        out = _read(be, arena, stems)
    assert all(s is not None for s in out), (out, tree.spills)


# --- the real spill behind ``serve`` ----------------------------------------------------------------

def test_serve_runs_the_real_w3_spill_for_the_deficit(tmp_path):
    """``serve`` -> the REAL ``UnifiedRadixCache._w3_arena_spill`` on a real tree: the node with a host copy is
    released (L3 copy first), the deficit is what the header says."""
    from unittest import mock

    from sglang.srt.mem_cache.base_prefix_cache import InsertParams
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache_components.tree_component import (
        BASE_COMPONENT_TYPE as FULL,
        ComponentType,
    )
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    from test_unified_radix_cache_unittest import CacheConfig, build_fixture

    cache, allocator, _ = build_fixture(CacheConfig(page_size=1, components=(ComponentType.FULL,)))
    for first, n in ((1, 64), (5000, 48)):
        cache.insert(InsertParams(key=RadixKey(list(range(first, first + n)), None),
                                  value=allocator.alloc(n).to(dtype=torch.int64)))
    row = 100
    for node in cache._collect_all_nodes():
        if node is cache.root_node:
            continue
        k = len(node.key)
        node.component_data[FULL].host_value = torch.arange(row, row + k, dtype=torch.int64)
        row += k

    class Pool:
        staging_rows = 0

        def __init__(self, arena):
            self.arena = arena
            self.secured = 0

        def secure_rows_to_l3(self, hv):
            self.secured += int(hv.numel())
            return {"pages": int(hv.numel()), "on_disk": int(hv.numel()), "written": 0, "lost": 0}

    class A:
        path = str(tmp_path / "fake.bin")

        def stats(self):
            return {"slots": 6485, "complete": 6485, "claimed": 0}

        def ref_census(self):
            return (6485, 16376, 6485)

    released_ids = []

    def _release(self_, node, comp, target=None, tracker=None):
        released_ids.append(node.id)
        hv = node.component_data[FULL].host_value
        node.component_data[FULL].host_value = None
        return 0, int(hv.numel())

    pool = Pool(A())
    with mock.patch.object(UnifiedRadixCache, "_evict_component_and_detach_lru", _release):
        got = _rr().serve(cache, pool, pool.arena, 100)
    assert got == 112 and len(released_ids) == 2 and pool.secured == 112, (got, released_ids)
    assert all(n.l3_present for n in cache._collect_all_nodes() if n.id in released_ids)


def test_the_hook_is_wired_into_the_scheduler_round():
    """``check_hicache_events`` calls ``read_room.tick`` once per round (source pin, the way the tree's other
    per-round hooks are pinned)."""
    import inspect

    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    src = inspect.getsource(UnifiedRadixCache.check_hicache_events)
    assert "_read_room_tick(self)" in src


def test_the_fill_posts_before_it_waits():
    import inspect

    src = inspect.getsource(HiCacheFile.arena_fill_from_disk)
    assert src.index("_read_room.post(arena, len(still_full)") < src.index("self._await_fill_room(")


# --- the real geometry of the boot's cases ------------------------------------------------------------
# 6485 slots, the need of the ``L3-FILL ROOM-WAIT stems=`` line of PP0 (claimed=0: not one free slot, so the
# arena is FULL at that moment) and the pages the nearest ``ARENA-REF-HOLDERS`` census puts on the trees. The
# rest of the full arena is held by what a tree spill cannot hand back (running requests' pages, loads, other
# reads' claims, parks). Boots ..._226aad7105_1015_{115748,122948,124245}.
SLOTS = 6485

#            case (boot rid, time of the cut)   stems  tree-held (spillable by the P tree)
GEOMETRY = [
    ("124245 weg2-4-25 12:52:35",                970, 6478),     # tree=6478 tree_in_use=0 on PP1/PP2
    ("124245 weg2-0-4 12:47:32",                 892, 5364),
    ("115748 weg2-0-6 12:01:09",                 384, 5807),
    ("122948 weg2-0-12 12:35:06",               1024, 4347),     # three reads of 1024/339/1396 asked in the same second
    ("122948 weg2-3-25 12:37:16",                964, 3032),
    ("115748 weg2-3-29 12:05:02",                851, 5804),
    ("115748 weg2-5-34 12:06:37",                634, 4047),     # 2297 more pages live (running requests)
    ("124245 weg2-70-293 13:35:39",              977,    0),     # tree=0, tree_in_use=2258, parks: nothing to hand back
]


@pytest.mark.parametrize("case,stems_n,tree_n", GEOMETRY, ids=[g[0] for g in GEOMETRY])
def test_the_boots_cases_at_their_real_geometry(tmp_path, caplog, case, stems_n, tree_n):
    """The read of every case of the boots at 6485 slots, the arena FULL (the ROOM-WAIT line says claimed=0): where
    the tree holds at least the need, the whole read lands; where live holders alone fill the arena, the cut is
    the physical case and names itself."""
    stems = [f"r{i}" for i in range(stems_n)]
    be, arena = _setup(tmp_path, stems, slots=SLOTS)
    held = _fill_arena_with_tree_pages(arena, tree_n, tag="tree") if tree_n else []
    _fill_arena_with_tree_pages(arena, SLOTS - tree_n, tag="live")      # the rest: referenced by nobody who can spill
    assert _rr().census(arena)["claimed"] == 0
    tree = _TreeSim(arena, held)
    with _Scheduler(tree), _ov("SGLANG_WEG2_READ_ROOM_WAIT_MS", 2000), caplog.at_level(logging.INFO):
        out = _read(be, arena, stems)
    landed = sum(1 for s in out if s is not None)
    if tree_n >= stems_n:
        assert landed == len(stems), (case, landed, tree.spills)
    else:
        assert landed < len(stems)
        shorts = [r.getMessage() for r in caplog.records if "READ-ROOM SHORT" in r.getMessage()]
        assert shorts and "served_nothing_to_spill" in shorts[-1], shorts


def test_three_pp_ranks_each_hold_the_page_and_the_slot_is_free_only_after_all_three_gave_it_back(tmp_path):
    """The arena census says refs > slots (16376 refs on 6485 slots): every PP stage's tree holds the same pages.
    PP0 posts; its own scheduler and two peer stages (their own ledger of seen requests) serve the request on
    their rounds. A slot is free only when the last of the three let go."""
    import threading

    stems = [f"r{i}" for i in range(24)]
    be, arena = _setup(tmp_path, stems, slots=64)
    held = _fill_arena_with_tree_pages(arena, 64)
    arena.ref_slots(held, +1)
    arena.ref_slots(held, +1)                       # three references on every slot
    trees = [_TreeSim(arena, held) for _ in range(3)]
    rr = _rr()
    stop = threading.Event()

    def peer(tr):
        seen = {}
        pool = tr._weg2_direct_pool()
        while not stop.is_set():
            n = rr._unseen(arena, seen)
            if n > 0:
                rr.serve(tr, pool, arena, n)
            time.sleep(0.02)

    peers = [threading.Thread(target=peer, args=(t,), name=f"peer{i}") for i, t in enumerate(trees[1:])]
    for th in peers:
        th.start()
    try:
        with _Scheduler(trees[0]):
            out = _read(be, arena, stems)
    finally:
        stop.set()
        for th in peers:
            th.join(5)
    assert all(s is not None for s in out), [i for i, s in enumerate(out) if s is None]
    assert all(t.spills for t in trees), [t.spills for t in trees]


def test_the_poll_of_the_request_file_is_paced(tmp_path):
    """An idle P stage runs ~1500 rounds/s: ``tick`` reads the 64-slot request file at most every 20 ms."""
    be, arena = _setup(tmp_path, ["r0"], slots=8)
    tree = _TreeSim(arena, [])
    rr = _rr()
    calls = []
    real = rr._unseen
    rr._unseen = lambda a, seen=None: (calls.append(1), 0)[1]
    try:
        for _ in range(200):
            rr.tick(tree)
    finally:
        rr._unseen = real
    assert 1 <= len(calls) <= 3, len(calls)
