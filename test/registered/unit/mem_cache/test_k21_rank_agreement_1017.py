"""K21 A1 (review nf-review-h16v-1017 p5, law raenge-nie-uneins-crash-stop): the write-back peel's
EVICT-SKIP after an A2-orphan drain ack must hold on EVERY rank, not on TP0 alone.

Two trees, Form A (host + worker), the REAL peel/ack/STATE code: the drain's own ack orphans the
node on every rank; TP0 gives the host layer back (R12 STATE), a worker keeps its byteless rows.
RED on 77d9fe7b08: TP0 skipped (not backuped), the worker still saw backuped=True and demoted the
node to host in the same peel -> the trees part (TP0 device, worker evicted), the STATE then
deleted/orphaned it on the worker. GREEN: all ranks keep the node on the device, the STATE
apply ends 'kv' (orphan rows dropped), the ranks' device and host flags are equal. A STATE that
cannot reconcile to 'kv' is the named stop K21RankDivergence.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import test_form_a_host_shadow_r12 as T  # noqa: E402
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache as U  # noqa: E402

FULL, MAMBA, m = T.FULL, T.MAMBA, T.m


def _wire(r):
    """Rank r: write_back policy, a leaf with no host copy, a write_backup that backs it and a drain
    whose ack orphans it (the same A2 event on every rank)."""
    r.cache_controller.write_policy = "write_back"
    r.pp_size = 1
    r._weg2_l3_drop = staticmethod(lambda *a, **k: False)
    leaf = r.node_at(3 * T.NODE)
    for ct in T.COMPS:
        leaf.component_data[ct].host_value = None

    def write_backup(node, write_back=False, **kw):
        node.component_data[FULL].host_value = T.torch.arange(0, T.NODE)
        node.component_data[MAMBA].host_value = T.torch.tensor([0])
        return 1

    r.write_backup = write_backup
    r.writing_check = lambda write_back=False: U._k12_direct_l3_verdict(r, leaf, 1)
    r._k12_republish_orphaned = lambda node, o, _r=r: U._k12_republish_orphaned(_r, node, o)
    r._weg2_release_chain_piece_host = lambda node, _r=r: U._weg2_release_chain_piece_host(_r, node)
    return leaf


def _peel(r):
    leaf = r.node_at(3 * T.NODE)
    r._evict_device_leaf(leaf, {ct: 0 for ct in T.COMPS})


def _setup():
    ranks = T._ranks()
    leaves = [_wire(r) for r in ranks]
    return ranks, leaves


def test_every_rank_keeps_the_orphaned_node_on_the_device_and_the_state_agrees():
    ranks, leaves = _setup()
    with T._switch(True):
        T._each(ranks, _peel)
        assert [l.evicted for l in leaves] == [False, False, False], (
            "TP0 skipped, a worker demoted the same node in the same peel (rank-local K21)")
        T._broadcast(ranks)
    assert [l.evicted for l in leaves] == [False, False, False]
    snaps = [r.snapshot() for r in ranks]
    assert snaps[1] == snaps[0] and snaps[2] == snaps[0], snaps
    assert snaps[0][3 * T.NODE] == (0, 0)  # orphan rows gone everywhere
    # the same next leaf for every rank: the tree is unchanged on all of them
    assert len({tuple(sorted(r.snapshot())) for r in ranks}) == 1


def test_state_that_cannot_reconcile_to_kv_is_a_named_group_stop():
    ranks, leaves = _setup()
    with T._switch(True):
        T._each(ranks, _peel)
        # the worker lost the node from the device meanwhile: STATE would delete/orphan it there
        w = leaves[1]
        w.component_data[FULL].value = None
        with pytest.raises(m.K21RankDivergence):
            T._broadcast(ranks)


def test_without_an_orphan_ack_the_peel_demotes_as_before():
    ranks, leaves = _setup()
    for r in ranks:
        r.writing_check = lambda write_back=False: None  # a clean drain: nothing orphaned
    with T._switch(True):
        T._each(ranks, _peel)
    assert [l.evicted for l in leaves] == [True, True, True]
