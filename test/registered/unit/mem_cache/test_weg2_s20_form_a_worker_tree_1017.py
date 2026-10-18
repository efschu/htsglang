"""S20 (h16v3 17.10.): the holder of D's 6485 arena slots was the Form A WORKERS' stale host rows.

Boot ..._4c655e6fd1_1017_224250, D.log: TP0 released the host rows of weg2-0-3..9 after their load-back
(``#248 LOADED-HOST-RELEASE``, 22:48:13-23, 307848 tokens = 4810 pages) and sent the R12 verdicts seq 1-4 (``HOST-VERDICT
SENT``); TP1/TP2 printed no APPLIED line before seq 5 at 22:49:08 -- ``consume`` found no tree (``_S.tree`` was set only
by worker_keeps / worker_short, the first SHADOW-KEEP of the boot is 22:49:08) and parked the events in ``_S.pending``.
ARENA-REF-HOLDERS 22:48:41: TP1/TP2 ``tree=6043 own_held=7765``, TP0 ``tree=1233 own_held=2955`` (6043 - 1233 = 4810);
refs 18485 = 2 x 7765 + 2955; arena pinned 6485 of 6485 for 45 s -> READ-ROOM SHORT class=not_posted x13, cause=arena_full
x13, the ~56 s CAPACITY-PARK of weg2-1-14.

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
from sglang.srt import rank_role
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache as U

m = H.m
pytestmark = pytest.mark.skipif(m is None, reason="needs form_a_host_shadow")


@pytest.fixture(autouse=True)
def _clean():
    """``attach`` calls read_room.serve_at_broadcast -> ``form_a_role()``, which caches the role for a second: left
    behind it turns a later test's plain PP tree into a Form A host (the h16t F1 room-set tests went red in one
    process). Reset around every test."""
    from sglang.srt.weg2 import read_room as _rr

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
