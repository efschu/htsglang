# SPDX-License-Identifier: Apache-2.0
"""READ-ROOM review conditions A1-A4, A6 (review wf_605481af-f2e of f5278f4ff2, done/nf-zero-arenaread-review-auflagen-1015.md).

A1 (HIGH B1): 124245 PP0 pdflip-4-25 -- reads 5114/7748/10032 ms, then ``PF TOLD-FALLBACK told=43392 -> 0 reason=frist
    after=0.16s``: PP0's own reads (each with its room wait) ate the told Frist's ``intake + TOTAL_S`` (16 s), the
    Frist was due at the publication and the acks on their way (PP2 own=43392 at +4 s) met told=0. READ-ROOM is now
    integrated WITH told2 (d736852bcf, T2 WAIT-PROGRESS / DEFER-LIFESIGN) and coupled to it: an armed room wait is a
    lifesign of the read (``read_room.lifesign_s``), the clamp moves by it (``pdflip_told_fallback.deadline``), told2's
    absolute ceiling bounds it. The combined case: 3 real fills x 10 s of room wait without progress, then PP0's
    told -- RED on f5278f4ff2 (``reason=frist`` at +0.16 s), GREEN here (the acks at +4 s admit the told).
A2 (B2): the SHORT class of a wait comes from THAT wait's serves (reviewer probe R1, unchanged).
A3 (B3): a rank never spills again for a read it already let go of while a peer still holds the pages (reviewer
    probe R2, unchanged; plus the peer rank's post of the same read, by content id).
A4 (B4): a geometry case on a REAL UnifiedRadixCache with the REAL ``_w3_arena_spill`` (device+host nodes, a node kept
    by W3-KEEP-PAYABLE, a host-only chain with a child) on the real C arena; RED on 226aad7105 by behaviour.
A6 (B5): the ``.room`` file lies in the boot's arena directory; a rebind closes the maps of other arenas.
"""

import inspect
import logging
import os
import sys
import threading
import time
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(__file__))

import pytest
import torch

from flliper.srt.environ import envs
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from flliper.srt.mem_cache.unified_radix_cache import EvictLayer, UnifiedRadixCache
from flliper.srt.pdflip import pp_slot_fidelity as SF

import test_pdflip_read_room_1015 as T
from test_pdflip_read_room_1015 import _armed, _fill_arena_with_tree_pages, _ov, _read, _rr, _Scheduler, _setup, _TreeSim

FULL = ComponentType.FULL


@pytest.fixture(autouse=True)
def _clean():
    _rr().reset_for_tests()
    with envs.FLLIPER_PDFLIP_L3FILL_ROOM_WAIT_MS.override(300), \
            _ov("FLLIPER_PDFLIP_READ_ROOM_WAIT_MS", 4000), \
            envs.FLLIPER_PDFLIP_L3FILL_SELF_JOIN_TAKEOVER.override(False):
        yield
    _rr().reset_for_tests()


# === A1: READ-ROOM x told2 =================================================================================

def _told_sched(pp_size=3):
    """PP0's scheduler as the told verdict reads it: a queued request, nothing running, no first token and no
    prefill chunk for a minute (the box idles while PP0 reads -- the wedge detector would call it wedged)."""
    now = time.perf_counter()
    return types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_size=pp_size), waiting_queue=[object()],
        running_batch=types.SimpleNamespace(reqs=[]),
        last_first_token_progress_time=now - 60.0, last_prefill_progress_time=now - 60.0)


def _room_s(intake, now):
    from flliper.srt.managers import pdflip_store_told as st

    f = getattr(st, "_read_room_lifesign", None)
    return f(intake, now) if f is not None else 0.0


def _open(fb, sched, rid, told, now, own_read_s, window, room):
    kw = {"room_wait_s": room} if "room_wait_s" in inspect.signature(fb.pp0_open).parameters else {}
    return fb.pp0_open(sched, rid, told, now, own_read_s, window, published_pass=None, **kw)


def test_three_reads_of_10s_room_wait_without_progress_do_not_end_in_frist_told0(tmp_path, caplog):
    """pdflip-4-25's shape: PP0 reads the rid three times (first read + two re-reads), each read finds the arena full
    and NOTHING spillable (live references only), each room wait runs its real 10 s idle bound and ends short. Then
    PP0 publishes the told; the followers' acks arrive 4 s later naming exactly the told.
    RED on f5278f4ff2: the Frist is due at the publication -> ``(0, "frist")`` at +0.16 s (the 124245 line).
    GREEN: the room waits are a lifesign, the Frist keeps its window, the acks admit the told."""
    import flliper.srt.managers.scheduler_components.invariant_checker  # noqa: F401 -- loaded, as in the scheduler
    from flliper.srt.managers import pdflip_store_told as st
    from flliper.srt.managers import pdflip_told_fallback as fb

    be, arena = _setup(tmp_path, [f"s{k}_{i}" for k in range(3) for i in range(4)], slots=16)
    _fill_arena_with_tree_pages(arena, 16, tag="live")          # running requests: no tree can hand them back
    tree = _TreeSim(arena, [])
    with _ov("FLLIPER_PDFLIP_READ_ROOM_WAIT_MS", 10000), _Scheduler(tree), caplog.at_level(logging.INFO):
        intake = time.monotonic()
        for k in range(3):
            out = _read(be, arena, [f"s{k}_{i}" for i in range(4)])
            assert all(x is None for x in out)                     # no progress: nothing landed
        now = time.monotonic()
    own_read_s = now - intake
    assert own_read_s >= 30.0, own_read_s
    told = 43392
    window = st.pace_window_s(own_read_s, told)
    sched = _told_sched()
    _open(fb, sched, "pdflip-4-25", told, now, own_read_s, window, _room_s(intake, now))
    first = fb.pp0_decide(sched, "pdflip-4-25", now + 0.16)
    assert first is None, ("the Frist was due at the publication: told=0 for acks on their way", first)
    o = fb._pp0_open_map(sched)["pdflip-4-25"]
    o.acks[1] = told
    o.acks[2] = told
    assert fb.pp0_decide(sched, "pdflip-4-25", now + 4.0) == (told, fb.REASON_ACKS)


def test_without_an_armed_room_wait_the_frist_is_the_old_one_and_the_lifesign_is_capped():
    from flliper.srt.managers import pdflip_told_fallback as fb

    pub, own, win = 1000.0, 23.0, 6.0
    old = max(pub, min(pub + fb.frist_s(win), pub - own + fb.TOTAL_S_DEFAULT))
    assert fb.deadline(pub, own, win) == old == pub
    assert fb.deadline(pub, own, win, 0.0) == old
    # the lifesign moves only the intake clamp, never past published + frist_s(window)
    assert fb.deadline(pub, own, win, 20.0) == pub + fb.frist_s(win)
    # capped by the read time and by told2's absolute ceiling
    assert fb.deadline(pub, own, win, 5.0) == max(pub, min(pub + fb.frist_s(win), pub - own + 16.0 + 5.0))
    with mock.patch.dict(os.environ, {fb.ENV_WAIT_ABS_S: "2"}):
        assert fb.deadline(pub, own, win, 20.0) == max(pub, pub - own + 16.0 + 2.0)


def test_the_lifesign_is_the_union_of_armed_room_waits_inside_the_window(tmp_path):
    rr = _rr()
    be, arena = _setup(tmp_path, ["r0"], slots=4)
    assert rr.room_wait_begin(arena) is None, "an unarmed wait is no lifesign (READ-ROOM off = the old Frist)"
    _armed(arena)
    rr._ST.waits.extend([[10.0, 14.0], [12.0, 16.0], [20.0, 21.0], [30.0, float("inf")]])
    assert rr.lifesign_s(0.0, 25.0) == pytest.approx(7.0)            # [10,16] + [20,21], overlap once
    assert rr.lifesign_s(13.0, 20.5) == pytest.approx(3.5)
    assert rr.lifesign_s(29.0, 31.0) == pytest.approx(1.0)           # an open wait counts up to ``until``


def test_pp0_passes_its_room_lifesign_to_the_frist():
    """The call site: PP0's publication hands the lifesign of [intake, now] to ``pp0_open`` (source pin)."""
    from flliper.srt.managers import pdflip_store_told as st

    src = inspect.getsource(st)
    assert "_room_s = _read_room_lifesign(_intake, now)" in src
    assert "room_wait_s=_room_s" in src


# === A2 / A3: reviewer probes R1, R2 (unchanged) =========================================================

def test_R1_short_class_is_per_wait_not_process_lifetime(tmp_path, caplog):
    """An earlier wait's successful spill must not turn a later wait whose serves released NOTHING into
    ``served_peers_hold``: the class is the evidence the next order (Park/Hand-off rest) is cut from."""
    rr = _rr()
    be, arena = T._setup(tmp_path, ["r0"], slots=16)
    T._fill_arena_with_tree_pages(arena, 16)
    rr._ST.armed = True
    rr._ST.arena = arena
    rr._ST.posted = 3
    rr._ST.spilled_pages = 900          # an EARLIER read's room was served (process lifetime counter)
    before = rr._ST.served
    rr._ST.served += 2                  # this wait: two serves, both released 0
    with caplog.at_level(logging.INFO):
        rr.note_short(arena, 8, 100, 10000.0, before)
    line = [r.getMessage() for r in caplog.records if "READ-ROOM SHORT" in r.getMessage()][-1]
    assert "class=served_nothing_to_spill" in line, line


def test_R2_a_rank_does_not_spill_again_for_a_deficit_only_peers_still_hold(tmp_path):
    """Two ranks hold every slot. Rank A serves the first post (spills d). A second post of the SAME read (the
    peer's fill, other pid) arrives before rank B served: the header still shows the same deficit, rank A has
    already let go of d slots -- a second d from A frees nothing the read needs (over-spill)."""
    rr = _rr()
    be, arena = T._setup(tmp_path, ["r0"], slots=512)
    held = T._fill_arena_with_tree_pages(arena, 512)
    arena.ref_slots(held, +1)            # rank B's reference on the same slots
    a = T._TreeSim(arena, held)
    rr._ST.armed = True
    rr._ST.arena = arena
    pool = a._pdflip_direct_pool()
    assert rr.serve(a, pool, arena, 200) == 200
    # the peer's post of the same read: deficit unchanged (B still holds) -> A serves again
    second = rr.serve(a, pool, arena, 200)
    assert second == 0, ("rank A spilled again although its own references on the deficit are gone", a.spills)


def test_the_posts_of_one_read_by_three_ranks_and_its_re_posts_spill_once_per_rank(tmp_path):
    """Through the shared file: PP0, PP1, PP2 fill the SAME stems of one rid (same content id, three pids) and re-post;
    rank A serves every post on its rounds but spills for the read once while rank B still holds the pages. A
    DIFFERENT read is served again. Once B let go, the pages are free -- A's ledger counts them no more."""
    import mmap

    rr = _rr()
    be, arena = _setup(tmp_path, ["r0"], slots=512)
    held = _fill_arena_with_tree_pages(arena, 512)
    arena.ref_slots(held, +1)                                       # rank B
    a = _TreeSim(arena, held)
    _armed(arena)
    rid = rr.new_read_id(["k0", "k1", "kLAST"])
    assert rid == rr.new_read_id([(0, "x"), (5, "kLAST")]), "the id is the read's content: every rank names it alike"
    m = rr._map(arena, create=True)
    for k, pid in enumerate((101, 102, 103)):                        # three posters, one read
        m[k * rr._REC.size:(k + 1) * rr._REC.size] = rr._REC.pack(rid, 200, rr._now_ms(), pid)
    pool = a._pdflip_direct_pool()
    seen = {}
    rr.serve_unseen(a, pool, arena, seen)
    assert a.spills == [200], a.spills
    m[0:rr._REC.size] = rr._REC.pack(rid, 200, rr._now_ms() + 1, 101)   # a re-post of the same read
    rr.serve_unseen(a, pool, arena, seen)
    assert a.spills == [200], ("a re-post spilled again", a.spills)
    other = rr.new_read_id(["other"])
    m[3 * rr._REC.size:4 * rr._REC.size] = rr._REC.pack(other, 100, rr._now_ms(), 101)
    rr.serve_unseen(a, pool, arena, seen)
    assert a.spills == [200, 100], ("another read is a new deficit", a.spills)
    mine = [x for x, _st in rr._ST.released[(id(a), rid)][1]]
    arena.ref_slots(mine, -1)                                       # rank B let go of the first read's pages
    assert rr._still_pinned_mine(arena, (id(a), rid)) == 0
    # the clock takes them and the slots hold OTHER pages now, referenced by their readers: still not "mine"
    got = arena.evict_candidates(len(mine))
    arena.free_evicted([c[0] for c in got], reason="test_clock")
    _fill_arena_with_tree_pages(arena, len(got), tag="newpage")
    assert rr._still_pinned_mine(arena, (id(a), rid)) == 0, "a reused slot is not the page this rank let go of"


def test_a_wait_after_a_served_wait_names_its_own_class_end_to_end(tmp_path, caplog):
    """A2 through the real fill: read 1 gets its room from the tree (spills), read 2 finds only live pages --
    its SHORT line says served_nothing_to_spill, not what read 1's wait did."""
    stems1 = [f"a{i}" for i in range(8)]
    stems2 = [f"b{i}" for i in range(8)]
    be, arena = _setup(tmp_path, stems1 + stems2, slots=32)
    held = _fill_arena_with_tree_pages(arena, 8, tag="tree")       # exactly read 1's room (the stand-in spills all)
    _fill_arena_with_tree_pages(arena, 24, tag="live")
    tree = _TreeSim(arena, held)
    with _ov("FLLIPER_PDFLIP_READ_ROOM_WAIT_MS", 1500), _Scheduler(tree), caplog.at_level(logging.INFO):
        out1 = _read(be, arena, stems1)
        assert all(s is not None for s in out1)
        arena.ref_slots([s for s in out1], +1)                       # read 1's request keeps its pages
        out2 = _read(be, arena, stems2)
    assert sum(1 for s in out2 if s is not None) < len(stems2)
    shorts = [r.getMessage() for r in caplog.records if "READ-ROOM SHORT" in r.getMessage()]
    assert shorts and "class=served_nothing_to_spill" in shorts[-1], shorts


# === A4: geometry on a REAL tree with the REAL spill =======================================================

class _ArenaPool:
    """P's arena KV host pool as the spill reads it: host rows = arena slots (staging 0, one row per page); the L3
    copy of a slot's page is written under the slot header's stem before its reference goes (#257 d)."""

    staging_rows = 0

    def __init__(self, arena, root, slots):
        self.arena, self.root, self.arena_slots = arena, root, slots
        self.written = 0

    def secure_rows_to_l3(self, hv):
        from flliper.srt.mem_cache.pool_host.arena_pool import arena_ref_slots

        slots = arena_ref_slots(self, hv).tolist()
        on_disk = w = 0
        for s in slots:
            p = os.path.join(self.root, self.arena.slot_stem(int(s)) + ".bin")
            if os.path.exists(p):
                on_disk += 1
            else:
                with open(p, "wb") as f:
                    f.write(b"\x44" * T.TOTAL)
                w += 1
        self.written += w
        return {"pages": len(slots), "on_disk": on_disk, "written": w, "lost": 0}


def _release_host(self_, node, comp, target=None, tracker=None):
    """The host-layer release of a P tree node: its rows' arena references go (one per rank)."""
    ct = getattr(comp, "component_type", comp)
    cd = node.component_data[ct]
    if target is None or EvictLayer.HOST not in target or cd.host_value is None:
        return 0, 0
    hv, cd.host_value = cd.host_value, None
    self_._t_arena.ref_slots(hv.tolist(), -1)
    self_._t_released.append(int(list(node.key.token_ids)[0]))
    return 0, int(hv.numel())


def _evict_leaf(self_, node, tracker):
    cd = node.component_data[FULL]
    hv, cd.host_value = cd.host_value, None
    self_._t_arena.ref_slots(hv.tolist(), -1)
    self_._t_released.append(int(list(node.key.token_ids)[0]))
    for k, v in list(node.parent.children.items()):
        if v is node:
            node.parent.children.pop(k)
    tracker[FULL] = tracker.get(FULL, 0) + int(hv.numel())


def _real_p_tree(arena, pool):
    """A real UnifiedRadixCache on the local-PP floor: A (24 pages, device+host), B (24, device+host) whose 8-page
    tail T went to the host and is host-locked (W3-KEEP-PAYABLE keeps B), a host-only chain HP (16) -> HC (16).
    Every host row names a COMPLETE arena slot with this tree's reference."""
    from flliper.srt.mem_cache.base_prefix_cache import InsertParams
    from flliper.srt.mem_cache.radix_cache import RadixKey

    from test_unified_radix_cache_unittest import CacheConfig, build_fixture

    cache, allocator, _ = build_fixture(CacheConfig(page_size=1, components=(FULL,)))
    for first, n in ((1, 24), (1001, 32), (1001, 24), (2001, 32), (2001, 16)):
        cache.insert(InsertParams(key=RadixKey(list(range(first, first + n)), None),
                                  value=allocator.alloc(n).to(dtype=torch.int64)))
    nodes = {}
    for node in cache._collect_all_nodes():
        if node is cache.root_node:
            continue
        nodes[int(list(node.key.token_ids)[0])] = node
    for first, node in nodes.items():
        n = len(node.key)
        slots = _fill_arena_with_tree_pages(arena, n, tag=f"t{first}_")
        node.component_data[FULL].host_value = torch.tensor(slots, dtype=torch.int64)
        node.hash_value = [f"h{first}_{i}" for i in range(n)]
    tail, hp, hc = nodes[1025], nodes[2001], nodes[2017]
    tail.component_data[FULL].value = None
    tail.component_data[FULL].host_lock_ref = 1
    hp.component_data[FULL].value = None
    hc.component_data[FULL].value = None
    SF.mark_floor_scope(cache, True)
    cache._pdflip_direct_pool = lambda: pool
    cache._t_arena = arena
    cache._t_released = []
    return cache, nodes


@pytest.fixture
def _floor_env():
    with mock.patch.dict(os.environ, {SF.ENV: "1"}), \
            mock.patch.object(UnifiedRadixCache, "_evict_component_and_detach_lru", _release_host), \
            mock.patch.object(UnifiedRadixCache, "_evict_host_leaf", _evict_leaf):
        yield


@pytest.mark.parametrize("need,lands_all", [(48, True), (72, False)], ids=["need48_of_56_spillable",
                                                                        "need72_over_56_spillable"])
def test_geometry_on_a_real_tree_with_the_real_spill(tmp_path, caplog, _floor_env, need, lands_all):
    """128 slots, FULL: 88 tree pages (A 24, B 24 + tail 8, HP 16 + HC 16) and 40 live pages. The real spill may
    hand back A and the host-only chain (24 + 32 = 56 pages); B is kept (its host-locked tail: W3-KEEP-PAYABLE), the
    tail never goes. need=48: every page lands (RED on 226aad7105: nothing lands). need=72: the 56 land, the read
    ends short and names it ``served_spilled_short``; B and the tail are untouched."""
    stems = [f"r{i}" for i in range(need)]
    be, arena = _setup(tmp_path, stems, slots=128)
    pool = _ArenaPool(arena, str(tmp_path / "store"), 128)
    cache, nodes = _real_p_tree(arena, pool)
    _fill_arena_with_tree_pages(arena, 40, tag="live")
    assert _rr().census(arena)["claimed"] == 0 and _rr().deficit(arena, need)[1]["free"] == 0
    with _ov("FLLIPER_PDFLIP_READ_ROOM_WAIT_MS", 2000), _Scheduler(cache), caplog.at_level(logging.INFO):
        out = _read(be, arena, stems)
    landed = sum(1 for s in out if s is not None)
    assert 1001 not in cache._t_released and 1025 not in cache._t_released, cache._t_released
    if lands_all:
        assert landed == need, (landed, cache._t_released)
    else:
        assert landed == 56, (landed, cache._t_released)
        shorts = [r.getMessage() for r in caplog.records if "READ-ROOM SHORT" in r.getMessage()]
        assert shorts and "class=served_spilled_short" in shorts[-1], shorts
    assert sorted(set(cache._t_released)) == [1, 2001, 2017]
    assert pool.written == 56, "every released page got its L3 copy first"


# === A6 ===================================================================================================

def test_the_room_file_lies_in_the_arena_directory_and_a_rebind_closes_old_maps(tmp_path):
    import importlib.util

    rr = _rr()
    be, arena = _setup(tmp_path, ["r0"], slots=4)
    assert os.path.dirname(rr._path(arena)) == os.path.dirname(arena.path)
    # the boot's arena directory goes with the launcher's shm sweep (SHM_OWN_PREFIXES) -- read as text, the module is
    # the whole launcher
    src = open(importlib.util.find_spec("flliper.srt.pdflip.launcher").origin).read()
    assert '    "pdflip-arena-",\n)' in src
    _armed(arena)
    assert rr.post(arena, 2)
    assert rr._ST.maps.get(rr._path(arena)) is not None
    other = types.SimpleNamespace(path=str(tmp_path / "other.bin"))
    rr._forget_other_maps(other)
    assert rr._path(arena) not in rr._ST.maps


def test_two_reads_of_one_process_take_two_slots_and_their_needs_add_up(tmp_path):
    """Review B5: two fills of one rank (two rids read at once) posted into ONE slot (pid) and the larger need won
    -- the other read's need vanished. Each read has its own slot; a re-post of a read keeps its slot."""
    rr = _rr()
    be, arena = _setup(tmp_path, ["r0"], slots=16)
    _armed(arena)
    a, b = rr.new_read_id(["x", "rid-a-last"]), rr.new_read_id(["x", "rid-b-last"])
    assert rr.post(arena, 10, read_id=a) and rr.post(arena, 6, read_id=b)
    reads = dict(rr._unseen_reads(arena, {}))
    assert reads == {a: 10, b: 6}, reads
