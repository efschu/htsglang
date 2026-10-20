"""K12/H1 (h16p review, open point H1 after A2): on a Form A D group
``l3_present`` is the GROUP's completeness -- TP0 decides, the workers mirror.

A2 (e6ec15e1e0) sets ``l3_present`` in ``_drain_backup`` only from a COMPLETE
ack -- read rank-locally. A write that lands only partly (or is refused stale /
fails, K12) on ONE rank then leaves the TP ranks with different
``l3_present`` for the same node; its readers (the write_backup parent rule,
``publish_unbacked_sweep``, the anchor-only backup) issue backups on that rank
alone -> rank-asymmetric ``hicache_backup(n)`` -> the sleep drain's named
refusal, the K12 class.

K12b: under R12 (Form A role plan installed, ``FLLIPER_PDFLIP_L3_PRESENT_GROUP``
default on) a rank sets ``l3_present`` only by TP0's verdict: TP0 on its own
complete ack (no cut) or on the COMPLETE page (cut: every owner wrote, the
rebind), sent as an R12 ``L3`` event on the ``tp<-reqs`` broadcast; a worker
from that event only, never from its own ack. Off R12 (P, classic) A2 stands.

Hermetic, on the R12 harness (test_form_a_host_shadow_r12): the REAL drain on
real nodes, each rank under its installed plan, the broadcast a pickle round
trip. RED on 97e49033c2: the workers' l3_present follows their own acks.
"""

from __future__ import annotations

import importlib.util
import os
from types import SimpleNamespace

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "_r12_harness_k12b", os.path.join(_HERE, "test_form_a_host_shadow_r12.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

m = H.m
NODE, PAGE = H.NODE, H.PAGE
OWNERS = (1, 2)


def _l3(ranks, depth):
    return [bool(getattr(r.node_at(depth), "l3_present", False)) for r in ranks]


def _ack(rank, node, *, complete=True):
    """One store ack of ``node`` on ``rank`` (H74: rank-local). ``complete``:
    every page written (A2) or one page short."""
    rank._ids += 1
    pages = len(node.hash_value)
    op = SimpleNamespace(id=rank._ids, hash_value=list(node.hash_value),
                         completed_tokens=(pages if complete else pages - 1) * PAGE)
    rank.ongoing_backup[op.id] = (node, None)
    rank.cache_controller.ack_backup_queue.put(op)
    with H._as_rank(rank):
        rank._drain_storage_control_queues_impl(
            n_revoke=None, n_backup=None, n_release=None,
            extra_release_counts=None, log_metrics=False)


@pytest.fixture
def group_on(monkeypatch):
    monkeypatch.delenv("FLLIPER_PDFLIP_L3_PRESENT_GROUP", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_L3_PRESENT_STRICT", raising=False)


def test_tp0_partial_worker_complete_no_rank_reads_l3(group_on):
    """THE H1 FORM. TP0's write is one page short, the workers' are complete.
    RED on 97e49033c2: TP0 l3_present False, TP1/TP2 True (rank-asymmetric)."""
    ranks = H._ranks(1)
    with H._switch(True):
        nodes = [r.node_at(NODE) for r in ranks]
        _ack(ranks[0], nodes[0], complete=False)
        _ack(ranks[1], nodes[1])
        _ack(ranks[2], nodes[2])
        H._broadcast(ranks)
    assert _l3(ranks, NODE) == [False, False, False]


def test_tp0_complete_every_rank_after_the_broadcast(group_on):
    """TP0 complete: the workers do NOT set it on their own ack; after TP0's
    broadcast every rank reads l3_present."""
    ranks = H._ranks(1)
    with H._switch(True):
        nodes = [r.node_at(NODE) for r in ranks]
        _ack(ranks[1], nodes[1])
        _ack(ranks[2], nodes[2])
        assert _l3(ranks, NODE) == [False, False, False]  # no rank-local verdict
        _ack(ranks[0], nodes[0])
        assert _l3(ranks, NODE) == [True, False, False]
        H._broadcast(ranks)
    assert _l3(ranks, NODE) == [True, True, True]


def test_a_failed_tp0_backup_is_no_l3_anywhere(group_on):
    """K12 + H1: TP0's backup thread failed the operation; nobody reads L3."""
    from flliper.srt.mem_cache import pdflip_backup_guard as g

    ranks = H._ranks(1)
    with H._switch(True):
        nodes = [r.node_at(NODE) for r in ranks]
        ranks[0]._ids += 1
        op = SimpleNamespace(id=ranks[0]._ids, hash_value=list(nodes[0].hash_value),
                             completed_tokens=0)
        g.mark_failed(op, "IndexError: index 72085 is out of bounds for dimension 0 with size 6549")
        ranks[0].ongoing_backup[op.id] = (nodes[0], None)
        ranks[0].cache_controller.ack_backup_queue.put(op)
        with H._as_rank(ranks[0]):
            ranks[0]._drain_storage_control_queues_impl(
                n_revoke=None, n_backup=None, n_release=None,
                extra_release_counts=None, log_metrics=False)
        _ack(ranks[1], nodes[1])
        _ack(ranks[2], nodes[2])
        H._broadcast(ranks)
    assert _l3(ranks, NODE) == [False, False, False]
    H._same(ranks)  # and the host entries stay equal (TP0 kept its rows)


# ------------------------------------------------------------------ the cut
class _CutRank(H._Rank):
    def __init__(self, idx, arena):
        super().__init__(idx)
        self.arena = arena
        self.cache_controller.mem_pool_host.arena_read = True

    def _pdflip_rebind_host_to_arena(self, node):
        h = node.hash_value[-1]
        return all(h in self.arena[o] for o in OWNERS)


@pytest.fixture
def cut(monkeypatch, group_on):
    from flliper.srt import rank_role

    monkeypatch.setattr(rank_role, "form_a_token_cut_active", lambda: True)
    yield


def _cut_ranks():
    arena = {o: set() for o in OWNERS}
    ranks = [_CutRank(i, arena) for i in range(3)]
    for r in ranks:
        r.add_chain(1)
    return ranks, arena


def test_cut_l3_only_for_the_complete_page(cut):
    """Under the cut TP0 (share 0) acks first, the page is not complete: no
    rank reads L3; once both owners wrote, TP0's rebind sets it and every
    worker mirrors it at the same broadcast."""
    ranks, arena = _cut_ranks()
    with H._switch(True):
        nodes = [r.node_at(NODE) for r in ranks]
        _ack(ranks[0], nodes[0])
        H._broadcast(ranks)
        assert _l3(ranks, NODE) == [False, False, False]
        for o in OWNERS:
            arena[o].add(f"h{NODE}")
        _ack(ranks[1], nodes[1])
        _ack(ranks[2], nodes[2])
        assert _l3(ranks, NODE) == [False, False, False]  # workers: never their own ack
        H._broadcast(ranks)
    assert _l3(ranks, NODE) == [True, True, True]


def test_switch_off_is_a2_rank_local(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_L3_PRESENT_GROUP", "0")
    ranks = H._ranks(1)
    with H._switch(True):
        nodes = [r.node_at(NODE) for r in ranks]
        _ack(ranks[0], nodes[0], complete=False)
        _ack(ranks[1], nodes[1])
        _ack(ranks[2], nodes[2])
    assert _l3(ranks, NODE) == [False, True, True]  # A2's reading, kept on the switch
