"""#239 S4b part 4: the R12 host shadow under the TOKEN CUT (kv=qsa_forma_dcp).

Under the cut a Form A worker owns token rows of every page -- its host rows
carry the page's bytes -- and TP0 holds share 0 in the target form 0/48/16:
it writes nothing. TP0's own store ack therefore says nothing about the
page. Before part 4 R12 decided a node's host life on TP0's ack alone:

* TP0 acks first (nothing to write), its arena rebind finds the page not yet
  complete, it records TRANSIT; at the next broadcast every rank runs the
  release -- a worker whose own store write is still in flight holds a host
  lock, so its release is a silent no-op: TP0 has no host rows, the worker
  has them. The trees part (the rc9m class R12 exists to close).

Part 4: a worker keeps its rows at its own ack; TP0 rebinds only a COMPLETE
page (every owner wrote -- the shared arena header) and sends REBIND, on
which the workers rebind the same node; an incomplete page waits, then falls
back to the pre-cut decision; TRANSIT/REBIND on a busy node stays pending.

Hermetic, on the R12 harness (test_form_a_host_shadow_r12): the REAL drain,
release and eviction methods on real nodes, each rank under its installed
Form A plan, the broadcast a pickle round trip. The arena is modelled by one
shared set of complete pages; a page completes when BOTH owner workers'
writes landed. RED on ebeccfd190, GREEN with part 4.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "_r12_harness", os.path.join(_HERE, "test_form_a_host_shadow_r12.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

m = H.m
NODE = H.NODE
FULL, MAMBA = H.FULL, H.MAMBA
OWNERS = (1, 2)  # the token cut 0/48/16: the two workers own rows, TP0 none


class _CutRank(H._Rank):
    """A rank whose arena rebind succeeds only on a COMPLETE page (every
    owner's write landed -- the arena header, shared by all ranks)."""

    def __init__(self, idx, arena):
        super().__init__(idx)
        self.arena = arena
        self.rebound = []
        self.cache_controller.mem_pool_host.arena_read = True  # part 3: workers have the arena

    def _pdflip_rebind_host_to_arena(self, node):
        h = node.hash_value[-1]
        if not all(h in self.arena["written"][o] for o in OWNERS):
            return False
        self.rebound.append(h)
        return True


def _cut_ranks(n_nodes=3):
    arena = {"written": {o: set() for o in OWNERS}}
    ranks = [_CutRank(i, arena) for i in range(3)]
    for r in ranks:
        r.add_chain(n_nodes)
    return ranks, arena


def _write(arena, owner, depth):
    arena["written"][owner].add(f"h{depth}")


@pytest.fixture
def cut(monkeypatch):
    """The token cut is active on every rank (group-uniform by construction)."""
    from flliper.srt import rank_role

    monkeypatch.setattr(rank_role, "form_a_token_cut_active", lambda: True)
    yield


def _ack_one(rank, node):
    """One store ack of ``node`` drained on ``rank`` (H74: rank-local)."""
    from types import SimpleNamespace

    rank._ids += 1
    op = SimpleNamespace(id=rank._ids, completed_tokens=0)
    rank.ongoing_backup[op.id] = (node, None)
    rank.cache_controller.ack_backup_queue.put(op)
    with H._as_rank(rank):
        rank._drain_storage_control_queues_impl(
            n_revoke=None, n_backup=None, n_release=None,
            extra_release_counts=None, log_metrics=False)


def test_tp0_acks_first_while_a_worker_still_writes_the_trees_stay_equal(cut):
    """THE ROOT. TP0 (share 0) acks node 2 before the owners' writes landed;
    worker TP1 still writes it (host lock). RED on ebeccfd190: TP0 recorded
    TRANSIT, the broadcast freed TP0's rows, TP1's release was a no-op under
    its lock -> TP0 (0,0) vs TP1 (1,1) at that depth."""
    ranks, arena = _cut_ranks()
    with H._switch(True):
        mid = [r.node_at(2 * NODE) for r in ranks]
        mid[1].component_data[FULL].host_lock_ref = 1  # TP1's own write in flight
        _ack_one(ranks[0], mid[0])  # TP0 first: nothing to write, page incomplete
        H._broadcast(ranks)
        H._same(ranks)  # nobody dropped a row TP0's ack could not vouch for
        mid[1].component_data[FULL].host_lock_ref = 0
        _write(arena, 1, 2 * NODE)
        _write(arena, 2, 2 * NODE)
        _ack_one(ranks[1], mid[1])
        _ack_one(ranks[2], mid[2])
        H._broadcast(ranks)  # the page is complete now: TP0 rebinds, sends REBIND
    snap = H._same(ranks)
    assert snap[2 * NODE] == (1, 1)
    assert f"h{2 * NODE}" in ranks[0].rebound
    assert all(f"h{2 * NODE}" in r.rebound for r in ranks[1:])  # the workers followed


def test_a_worker_never_rebinds_on_its_own_ack(cut):
    """Both owners wrote and acked before TP0: the workers keep their rows
    and wait for TP0's verdict; they rebind only at the REBIND broadcast."""
    ranks, arena = _cut_ranks(1)
    with H._switch(True):
        node = [r.node_at(NODE) for r in ranks]
        _write(arena, 1, NODE)
        _write(arena, 2, NODE)
        _ack_one(ranks[1], node[1])
        _ack_one(ranks[2], node[2])
        assert ranks[1].rebound == [] and ranks[2].rebound == []
        _ack_one(ranks[0], node[0])
        H._broadcast(ranks)
    assert all(r.rebound == [f"h{NODE}"] for r in ranks)
    assert H._same(ranks) == {NODE: (1, 1)}


def test_a_page_that_never_completes_falls_back_to_transit_on_every_rank(cut):
    """An owner's write never lands: after REBIND_WAIT_PASSES broadcasts TP0
    takes the pre-cut decision (TRANSIT for a transit host) -- on every rank
    at the same broadcast, so the trees stay equal throughout."""
    ranks, arena = _cut_ranks(1)
    with H._switch(True):
        node = [r.node_at(NODE) for r in ranks]
        _write(arena, 1, NODE)  # TP2's rows never arrive
        for r, n in zip(ranks, node):
            _ack_one(r, n)
        for _ in range(m.REBIND_WAIT_PASSES - 1):
            H._broadcast(ranks)
            assert H._same(ranks) == {NODE: (1, 1)}
        H._broadcast(ranks)
    assert H._same(ranks) == {NODE: (0, 0)}
    assert all(r.rebound == [] for r in ranks)


def test_a_rebind_on_a_busy_worker_node_waits_for_the_lock(cut):
    """REBIND reaches a worker whose node is under a host lock (a load in
    flight): it stays pending and is applied at the next broadcast."""
    ranks, arena = _cut_ranks(1)
    with H._switch(True):
        node = [r.node_at(NODE) for r in ranks]
        _write(arena, 1, NODE)
        _write(arena, 2, NODE)
        for r, n in zip(ranks, node):
            _ack_one(r, n)
        node[2].component_data[FULL].host_lock_ref = 1
        H._broadcast(ranks)
        assert ranks[2].rebound == [] and ranks[1].rebound == [f"h{NODE}"]
        node[2].component_data[FULL].host_lock_ref = 0
        H._broadcast(ranks)
    assert ranks[2].rebound == [f"h{NODE}"]
    assert H._same(ranks) == {NODE: (1, 1)}


def test_a_tree_reset_drops_the_waiting_pages(cut):
    """No wait outlives the tree it names: a reset clears TP0's rebind wait,
    so a re-inserted prefix of the same hash is never rebound on stale news."""
    ranks, _arena = _cut_ranks(1)
    with H._switch(True):
        _ack_one(ranks[0], ranks[0].node_at(NODE))
        with H._as_rank(ranks[0]):
            assert m._S.rebind_wait  # waiting on the owners
            m.on_tree_reset(ranks[0])
            assert not m._S.rebind_wait
            assert m.attach(["x"]) == ["x"]


def test_without_the_cut_the_pre_part4_path(monkeypatch):
    """Guard: no cut -> R12 as before (TP0 rebinds at its ack, a worker keeps
    its byteless rows, no REBIND on the wire)."""
    from flliper.srt import rank_role

    monkeypatch.setattr(rank_role, "form_a_token_cut_active", lambda: False)
    ranks, arena = _cut_ranks(1)
    with H._switch(True):
        _write(arena, 1, NODE)
        _write(arena, 2, NODE)
        _ack_one(ranks[0], ranks[0].node_at(NODE))
        with H._as_rank(ranks[0]):
            assert m.attach(["x"]) == ["x"]  # nothing to broadcast
    assert ranks[0].rebound == [f"h{NODE}"]
