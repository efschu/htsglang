"""S22 (18.10.): R12 SPILL-DIVERGED seq=46 node=90 out=unreconcilable, NF h16v4b 01:40:12 (law raenge-nie-uneins, SOLL R11).

THE ROOT: device life is rank-local, host life is TP0's. TP0 spilled a node that was device-resident on TP0 while
both workers had already taken it off the device (an interior node: its KV exists there only as its host rows) --
the worker has no state to follow into (``unreconcilable``; a device-evicted leaf: ``deleted``). The fix is at TP0's
choice: a worker publishes its device-evicted content keys next to the arena (``_evict_to_host`` ->
``worker_device_evicted``), TP0 spills only nodes no worker has evicted. Hermetic, the dual15 fixture (real arena, real
pools, real ``_w3_arena_spill`` / ``attach`` / ``apply``).
"""
import logging
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from test_w3_arena_spill_forma_dual15_0930 import FULL, _as, _claimer, group  # noqa: E402,F401

from sglang.srt.mem_cache import form_a_host_shadow as R  # noqa: E402


def _worker_peels(g, idxs):
    """The worker's own device peel, leaf first, through the real ``_evict_to_host`` (the fixture tree has no LRU
    machinery: the component free is the one stand-in, it only takes the KV off the device)."""
    def _free(node, comp, target=None, tracker=None):
        node.component_data[FULL].value = None
        return 0, 0

    g.t1._evict_component_and_detach_lru = _free
    g.t1._cascade_evict = lambda node, trigger, tracker: None
    g.t1._record_remove_event = lambda node, medium=None: None
    g.t1._for_each_component_lru = lambda *a, **k: None
    for i in sorted(idxs, reverse=True):
        n = g.n1[i]
        with _as("worker"):
            g.t1._evict_to_host(n)
        assert n.evicted and n.component_data[FULL].host_value is not None


def _tp0_serves(g, expect_verdict=True):
    with _as("worker"):
        m1 = _claimer(g.t1, "b")
        assert g.t1._weg2_direct_claim(m1) is False
    with _as("host"):
        R.register_tree(g.t0)
        sent = R.attach([])
    if not expect_verdict:
        return sent
    assert sent and isinstance(sent[0], R.FormAHostVerdict)
    return sent[0]


def test_s22_tp0_does_not_spill_a_node_a_worker_has_device_evicted(group, caplog):
    g = group
    evicted = [4, 5, 6, 7]                         # the worker's tail left its device (interior 4..6 + leaf 7)
    _worker_peels(g, evicted)
    v = _tp0_serves(g)
    for i in evicted:                              # TP0 keeps those host copies: no worker could follow
        assert g.n0[i].component_data[FULL].host_value is not None, i
        assert not g.n0[i].l3_present
    assert [i for i in range(4) if g.n0[i].component_data[FULL].host_value is None], "TP0 spilled nothing at all"
    caplog.set_level(logging.WARNING, logger=R.logger.name)
    with _as("worker"):
        c = R.apply(g.t1, v.events, seq=v.seq)
    assert c["unreconcilable"] == 0 and c["deleted"] == 0 and c["short"] == 0 and c["missing"] == 0
    assert not [r for r in caplog.records if "SPILL-DIVERGED" in r.getMessage() or "UNRECONCILABLE" in r.getMessage()]
    for a, b in zip(g.n0, g.n1):                   # one host life on every rank
        assert (a.component_data[FULL].host_value is None) == (b.component_data[FULL].host_value is None)
        assert bool(a.l3_present) == bool(b.l3_present)


def test_s22_the_h16v4b_shape_without_the_worker_state_diverges(group, monkeypatch):
    """Anchor of the repro: with the publish switched off (the pre-fix TP0 choice) the same scene IS unreconcilable."""
    g = group
    monkeypatch.setattr(R, "worker_device_evicted", lambda tree, node: None)
    monkeypatch.setattr(R, "_dev_refresh", lambda tree: None)
    _worker_peels(g, [4, 5, 6, 7])
    v = _tp0_serves(g)
    with _as("worker"):
        c = R.apply(g.t1, v.events, seq=v.seq)
    assert c["unreconcilable"] + c["deleted"] >= 1


def test_s22_a_clean_group_still_spills_every_node(group):
    g = group
    v = _tp0_serves(g)
    assert all(n.component_data[FULL].host_value is None for n in g.n0)
    with _as("worker"):
        c = R.apply(g.t1, v.events, seq=v.seq)
    assert c["spill"] == len([e for e in v.events if e[0] == R.SPILL]) and c["unreconcilable"] == 0


def test_s22_a_node_back_on_the_device_is_spillable_again(group):
    g = group
    _worker_peels(g, [7])
    assert R.worker_evicted_keys(g.t0)
    g.n1[7].component_data[FULL].value = g.n0[7].component_data[FULL].value   # loaded back
    with _as("worker"):
        R._dev_refresh(g.t1)
    assert not R.worker_evicted_keys(g.t0)


def test_s22_two_workers_union_and_a_dead_process_is_ignored(group):
    g = group
    _worker_peels(g, [6])
    pre = R._dev_dir_prefix(R._tree_arena(g.t1))
    import json
    with open(pre + "999999999", "w") as f:        # a worker process that no longer exists
        json.dump([["zz", 1]], f)
    own = R.worker_evicted_keys(g.t0)
    assert R.node_key(g.n1[6]) in own and ("zz", 1) not in own


def test_s22_an_unreadable_worker_file_is_named(group, caplog):
    g = group
    pre = R._dev_dir_prefix(R._tree_arena(g.t1))
    with open(pre + str(os.getpid()), "w") as f:
        f.write("{not json")
    caplog.set_level(logging.WARNING, logger=R.logger.name)
    R.worker_evicted_keys(g.t0)
    assert [r for r in caplog.records if "SPILL-APPLICABILITY read" in r.getMessage()]


def test_s22_tree_reset_releases_the_keys(group):
    g = group
    _worker_peels(g, [7])
    with _as("worker"):
        R.on_tree_reset(g.t1)
    assert not R.worker_evicted_keys(g.t0)


def test_s22_unkeyed_nodes_are_no_tp0_candidates(group):
    g = group
    for n in g.n0:
        n.hash_value = None
    sent = _tp0_serves(g, expect_verdict=False)
    assert all(n.component_data[FULL].host_value is not None for n in g.n0)
    assert not [v for v in sent if isinstance(v, R.FormAHostVerdict) and any(e[0] == R.SPILL for e in v.events)]
