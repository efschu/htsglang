"""K21 N1 (nf-nachreview-h16v-1017, law raenge-nie-uneins-crash-stop): the A2-orphan verdict of a
drain ack is RANK-LOCAL (each rank acks against the shared arena at its own time; metal:
1013_122103 only TP0, 1007_123249 only TP2, 1014_030008 first only TP1, later only TP0).  The
ranks must notice a ONE-SIDED orphan in BOTH directions without a new collective:

* TP0 sends an R12 event "K21" at EVERY own orphan ack (also when the host release fails);
* a worker stops by name (K21RankDivergence) when TP0 reports K21, it has no mark of its own and
  its STATE does not end in 'kv' (or no STATE follows);
* V1 (S11): TP0 reports EVERY direct ack (orphaned=0|1); a worker pairs it with its own ack finding
  per key (FIFO) and stops by name only when both exist and disagree (test_k21_ack_pairing_1017.py).

Host + two workers, the REAL peel / ack / STATE code of test_k21_rank_agreement_1017.py.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import test_form_a_host_shadow_r12 as T  # noqa: E402
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache as U  # noqa: E402

FULL, MAMBA, m = T.FULL, T.MAMBA, T.m


def _wire(r, orphan, release_fails=False):
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
    if orphan:
        r.writing_check = lambda write_back=False: U._k12_direct_l3_verdict(r, leaf, 1)
    else:
        r.writing_check = lambda write_back=False: U._k12_direct_l3_verdict(r, leaf, 0)  # a clean ack (V1: reported too)
    r._k12_republish_orphaned = lambda node, o, _r=r: U._k12_republish_orphaned(_r, node, o)
    if release_fails:
        r._weg2_release_chain_piece_host = lambda node: None
    else:
        r._weg2_release_chain_piece_host = lambda node, _r=r: U._weg2_release_chain_piece_host(_r, node)
    return leaf


def _peel(r):
    r._evict_device_leaf(r.node_at(3 * T.NODE), {ct: 0 for ct in T.COMPS})


def _setup(orphans, release_fails_on_tp0=False):
    ranks = T._ranks()
    leaves = [_wire(r, o, release_fails=(release_fails_on_tp0 and i == 0))
              for i, (r, o) in enumerate(zip(ranks, orphans))]
    return ranks, leaves


def _k21_sent(ranks):
    with T._as_rank(ranks[0]):
        return [e for e in m._S.ledger if e[0] == m.K21 and e[3] == 1]  # orphan verdicts


def test_only_tp0_orphans_the_workers_stop_by_name():
    ranks, leaves = _setup([True, False, False])
    with T._switch(True):
        T._each(ranks, _peel)
        assert len(_k21_sent(ranks)) == 1  # TP0 reports its orphan
        with pytest.raises(m.K21RankDivergence):
            T._broadcast(ranks)


def test_only_a_worker_orphans_the_next_handover_stops_it_by_name():
    ranks, leaves = _setup([False, True, False])
    with T._switch(True):
        T._each(ranks, _peel)
        assert _k21_sent(ranks) == []  # TP0 did not orphan: only a clean (orphaned=0) verdict
        with pytest.raises(m.K21RankDivergence):
            T._broadcast(ranks)


def test_symmetric_orphan_every_rank_agrees_and_tp0_reported_it():
    ranks, leaves = _setup([True, True, True])
    with T._switch(True):
        T._each(ranks, _peel)
        assert len(_k21_sent(ranks)) == 1
        T._broadcast(ranks)
        T._broadcast(ranks)  # a later handover must not trip a stale mark
    assert [l.evicted for l in leaves] == [False, False, False]
    assert T._same(ranks)[3 * T.NODE] == (0, 0)


def test_symmetric_orphan_with_failed_release_on_tp0_does_not_stop():
    ranks, leaves = _setup([True, True, True], release_fails_on_tp0=True)
    with T._switch(True):
        T._each(ranks, _peel)
        assert len(_k21_sent(ranks)) == 1  # sent although the release then failed
        T._broadcast(ranks)
        T._broadcast(ranks)
    assert [l.evicted for l in leaves] == [False, False, False]
