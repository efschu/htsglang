"""S20 (h16v3 17.10.): the holder of D's 6485 arena slots was the Form A WORKERS' stale host rows.

Boot ..._4c655e6fd1_1017_224250, D.log: TP0 released the host rows of pdflip-0-3..9 after their load-back
(``#248 LOADED-HOST-RELEASE``, 22:48:13-23, 307848 tokens = 4810 pages) and sent the R12 verdicts seq 1-4 (``HOST-VERDICT
SENT``); TP1/TP2 printed no APPLIED line before seq 5 at 22:49:08 -- ``consume`` found no tree (``_S.tree`` was set only
by worker_keeps / worker_short, the first SHADOW-KEEP of the boot is 22:49:08) and parked the events in ``_S.pending``.
ARENA-REF-HOLDERS 22:48:41: TP1/TP2 ``tree=6043 own_held=7765``, TP0 ``tree=1233 own_held=2955`` (6043 - 1233 = 4810);
refs 18485 = 2 x 7765 + 2955; arena pinned 6485 of 6485 for 45 s -> READ-ROOM SHORT class=not_posted x13, cause=arena_full
x13, the ~56 s CAPACITY-PARK of pdflip-1-14.

Fix: every rank registers its tree when it is built (``UnifiedRadixCache._reset_full`` runs in the constructor), so a
verdict is applied at the broadcast it rides. SOLL R11: the front/TP0 decides, the ranks apply the same verdict at the
same pass -- never a verdict held back by a rank-local "I have not seen a tree yet".

Hermetic: the harness of test_form_a_host_shadow_r12 (REAL UnifiedRadixCache host-life methods on real nodes, one
pickle round trip for the broadcast) -- except that a rank's tree is NOT handed to ``consume`` (the request receiver
never does) and NOT pre-registered (``_as_rank`` does that and hid this defect from every earlier test).
"""

from __future__ import annotations

import contextlib
import pickle

import pytest

import test_form_a_host_shadow_r12 as H
from flliper.srt import rank_role
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache as U

m = H.m
pytestmark = pytest.mark.skipif(m is None, reason="needs form_a_host_shadow")


@pytest.fixture(autouse=True)
def _clean():
    """``attach`` calls read_room.serve_at_broadcast -> ``form_a_role()``, which caches the role for a second: left
    behind it turns a later test's plain PP tree into a Form A host (the h16t F1 room-set tests went red in one
    process). Reset around every test."""
    from flliper.srt.pdflip import read_room as _rr

    _rr.reset_for_tests()
    yield
    _rr.reset_for_tests()


@contextlib.contextmanager
def _as_rank_untreed(rank):
    """H._as_rank without ``m._S.tree = rank``: the process state of this rank as the boot has it before any
    worker_keeps ran."""
    prev = (rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK)
    rank_role.set_form_a_role_plan(rank_role.RankRolePlan(H.ROLES), rank.idx)
    prev_state, m._S = m._S, rank.state
    try:
        yield
    finally:
        rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK = prev
        m._S = prev_state


def _construct(rank):
    """The construction edge of a real tree: ``UnifiedRadixCache._reset_full`` (the constructor runs ``reset``). The
    harness rank is not a full cache, so the body stops at the first attribute it lacks -- registration is the first
    statement of the method, which is all this edge contributes."""
    with _as_rank_untreed(rank):
        with contextlib.suppress(Exception):
            U._reset_full(rank)


def _broadcast_untreed(ranks):
    """``recv_requests``: TP0 attaches, the list travels (pickle), every rank consumes WITHOUT a tree argument."""
    with _as_rank_untreed(ranks[0]):
        wire = pickle.dumps(m.attach(["req-a"]))
    out = []
    for r in ranks:
        with _as_rank_untreed(r):
            out.append(m.consume(pickle.loads(wire)))
    return out


def test_construction_registers_the_tree_on_every_rank():
    ranks = H._ranks()
    for r in ranks:
        assert r.state.tree is None
        _construct(r)
        assert r.state.tree is r, f"rank {r.idx}: the tree must be known to R12 once it is built"


def test_a_worker_applies_tp0s_verdict_at_the_broadcast_it_rides():
    """The 22:48:14 shape: TP0 releases a node's host rows and sends the verdict; both workers must hold the same host
    entries right after THAT broadcast (before: they kept them for 45 s and pinned the arena). RED without the
    registration: TP0 (0, 0) for the middle node, the workers (1, 1) -- the verdict sits in ``_S.pending``."""
    ranks = H._ranks()
    ranks[0].fail_rebind.add(f"h{2 * H.NODE}")   # TP0's rebind fails: it releases the middle node's rows (a verdict)
    with H._switch(True):
        for r in ranks:
            _construct(r)
        # only TP0 acks here: a worker's store ack (worker_keeps) would register its tree by itself -- exactly
        # the first SHADOW-KEEP at 22:49:08 that ended the window; at 22:48:14 no worker had acked yet
        with _as_rank_untreed(ranks[0]):
            ranks[0].ack_store_writes()
        _broadcast_untreed(ranks)
    snaps = [r.snapshot() for r in ranks]
    assert snaps[0] == snaps[1] == snaps[2], f"TP0 {snaps[0]} TP1 {snaps[1]} TP2 {snaps[2]}"
    assert snaps[0] == {H.NODE: (1, 1), 2 * H.NODE: (0, 0), 3 * H.NODE: (1, 1)}
    for r in ranks[1:]:
        assert not r.state.pending, "no verdict may stay parked on a worker after the broadcast it rode"


# === the h16v3 TP0 host-arena-full cut (coordinator, 00:16-00:19Z, pdflip-54-635 / -623) ====================
def test_h16v3_tp0_arena_full_workers_free_the_tp0_own_store_read_lands_and_the_vote_is_not_zero(tmp_path):
    """D.log 00:17-00:19: ARENA-REF-HOLDERS TP0 ``tree=6013-6018 own_held=6485-6952`` against TP1/TP2 ``tree=3796-3801
    own_held=4449-4735`` (arena 6485 slots; the workers' own trees do NOT hold ~2200 of TP0's pages), 20x ``READ-ROOM
    SHORT class=not_posted`` / ``cause=arena_full`` on D, then 24 W31 with TP0 ``why=host_admission my_vote=0`` while the
    workers vote 60608: D's own store read of ~59584 tokens could not land on TP0 because TP0's host tree held the
    arena and nobody on D answered a room request.
    Replica (1:100): 48 slots, ALL pinned by TP0's tree (L3-backed spans, one reference each); the worker's tree holds
    only a small span of them (a second reference). D's own read of 16 stems runs on TP0 (its process) with READ-ROOM-D
    armed: TP0 serves the request at its broadcast, releases L3-backed childless spans, the clock takes the slots, EVERY
    stem lands (= TP0's vote is the full depth, not 0 -- no W31). The worker applies the same verdict."""
    import threading
    import time

    import test_pdflip_read_room_d_1015 as DD
    from test_pdflip_read_room_1015 import _fill_arena_with_tree_pages, _read, _rr, _setup

    stems = [f"r{i}" for i in range(16)]
    be, arena = _setup(tmp_path, stems, slots=48)
    t0_slots = _fill_arena_with_tree_pages(arena, 48, tag="dpage")            # TP0's tree holds the whole arena
    arena.ref_slots(t0_slots[0:8], +1)                                        # the worker holds one 8-page span of it
    stem_of = {s: f"dpage{i}" for i, s in enumerate(t0_slots)}
    root = DD._store_files(tmp_path, [f"dpage{i}" for i in range(48)])
    tp0_spans = [(1, t0_slots[0:8]), (2001, t0_slots[8:20]), (4001, t0_slots[20:34]), (6001, t0_slots[34:48])]
    tp0 = DD._real_d_tree(tp0_spans, arena, "tp0")
    worker = DD._real_d_tree([(1, t0_slots[0:8])], arena, "worker")
    pool = DD._Pool(arena, stem_of, root)
    tp0._pdflip_direct_pool = lambda: pool
    mp = pytest.MonkeyPatch()
    roles = DD._Roles()
    mp.setattr(DD.R12, "role", roles)
    mp.setattr(_rr(), "form_a_role", roles, raising=False)
    _rr().reset_for_tests()
    DD.R12._reset_state_for_tests()
    _rr()._ST.arena = arena
    stop = threading.Event()
    th = threading.Thread(target=DD._group_d_loop, args=(roles, tp0, worker, stop), name="d-group")
    th.start()
    try:
        time.sleep(0.1)                                    # D's first broadcast arms READ-ROOM on this arena
        out = _read(be, arena, stems)
    finally:
        stop.set()
        th.join(5)
        mp.undo()
        _rr().reset_for_tests()
        DD.R12._reset_state_for_tests()
    landed = sum(1 for s in out if s is not None)
    assert landed == len(stems), f"TP0's own read landed {landed}/{len(stems)}: its vote would be short, W31"
    assert DD._released(tp0)[:0] == [], "diagnostics"
    # the worker's span (shared by both ranks) is released on both or on neither -- never rank-uneven
    assert (1 in DD._released(tp0)) == (1 in DD._released(worker))


# === review S20 A2: RRSTATE counts as a STATE in the K21 pairing sets ==========================================
def test_k21_tp0_orphaned_worker_clean_with_only_an_rrstate_is_no_stop_and_resolves_kv():
    """TP0's ack orphaned the node (K21 event), this worker's own ack was clean, and the only verdict that follows is a
    READ-ROOM STATE (RRSTATE). The pairing must see a STATE for the key (no 'no STATE follows' deferred stop) and the
    worker must end ``kv`` (k21_expect resolved). Mutant: RRSTATE out of ``st_keys``/``state_keys`` -> deferred stop."""
    ranks = H._ranks()
    w = ranks[1]
    node = w.node_at(H.NODE)
    key = m.node_key(node)
    with H._switch(True):
        with H._as_rank(w):
            m.note_k21_ack(node, 0)                                   # own ack: clean
            m.apply(w, [(m.K21, key[0], key[1], 1, 0, 0),             # TP0: orphaned
                        (m.RRSTATE, key[0], key[1], 1, 0, 1)], seq=1)  # + a READ-ROOM STATE: KV host rows gone, anchor kept
            assert m._S.k21_stop_msg is None, m._S.k21_stop_msg
            assert key not in m._S.k21_expect, "k21_expect must be resolved with 'kv'"
            assert node.component_data[H.FULL].host_value is None
