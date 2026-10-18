"""S22 part 2 (18.10.): the SPILL-DIVERGED class in every TP0 spill path (law raenge-nie-uneins, SOLL R11).

* ``read_room.form_a_spill`` (READ-ROOM-D, STATE verdict): a worker has the node off the device -> TP0 keeps its host
  copy (counted ``kept_worker_evicted`` in the D-SPILL line), also when the worker peels BETWEEN the decision and the
  broadcast (re-read in ``flush_forma_pending``): no ``deleted`` / ``unreconcilable`` on the worker.
* ``host_only_spill`` (K23, STATE exists=0): TP0 drops a host-only leaf only when no worker keeps the node (device-resident
  or with children) -- the worker could not delete it.
"""
import json
import logging
import os
import sys
import threading
import types
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from test_weg2_read_room_d_1015 import (  # noqa: E402,F401
    FULL, _Pool, _broadcast, _clean, _real_d_tree, _released, _rr, evict_mock, roles,
)

from sglang.srt.mem_cache import form_a_host_shadow as R12  # noqa: E402


@pytest.fixture
def chan(tmp_path, monkeypatch):
    """The arena file the w3dev channel sits next to (the unit trees have no direct pool)."""
    arena = types.SimpleNamespace(path=str(tmp_path / "kv.bin"))
    monkeypatch.setattr(R12, "_tree_arena", lambda tree: arena)
    return arena


def _store(tmp_path, n=8):
    root = tmp_path / "store"
    root.mkdir()
    for s in range(n):
        (root / f"s{s}.bin").write_bytes(b"\x01")
    return str(root)


def _worker_peels(roles, worker, node):
    me = threading.current_thread().name
    roles.now[me] = "worker"
    node.component_data[FULL].value = None          # the device peel (the real funnel is covered in the dual15 test)
    R12.worker_device_evicted(worker, node)
    roles.now[me] = "host"


def test_form_a_spill_keeps_a_node_a_worker_has_device_evicted(tmp_path, roles, evict_mock, chan, caplog):
    me = threading.current_thread().name
    roles.now[me] = "host"
    root = _store(tmp_path)
    spans = [(1, list(range(8)))]
    tp0 = _real_d_tree(spans, None, "tp0")
    worker = _real_d_tree(spans, None, "worker")
    _worker_peels(roles, worker, worker._t_nodes[1])
    caplog.set_level(logging.INFO)
    got = _rr().form_a_spill(tp0, _Pool(None, {s: f"s{s}" for s in range(8)}, root), 1)
    assert got == 0
    line = [r.getMessage() for r in caplog.records if "D-SPILL n=" in r.getMessage()][-1]
    assert "kept_worker_evicted=1" in line, line
    v = _broadcast(roles, tp0, worker)
    assert _released(tp0) == [] and _released(worker) == []
    assert not [e for e in (v[0].events if v else ())], v


def test_a_peel_between_decision_and_broadcast_keeps_the_copy(tmp_path, roles, evict_mock, chan):
    me = threading.current_thread().name
    roles.now[me] = "host"
    root = _store(tmp_path)
    spans = [(1, list(range(8)))]
    tp0 = _real_d_tree(spans, None, "tp0")
    worker = _real_d_tree(spans, None, "worker")
    assert _rr().form_a_spill(tp0, _Pool(None, {s: f"s{s}" for s in range(8)}, root), 1) == 8   # decided
    _worker_peels(roles, worker, worker._t_nodes[1])                                           # then the worker peels
    _broadcast(roles, tp0, worker)
    assert _released(tp0) == [], "TP0 released a copy a worker could not follow on"
    assert _released(worker) == []


def test_a_clean_form_a_spill_still_releases(tmp_path, roles, evict_mock, chan):
    me = threading.current_thread().name
    roles.now[me] = "host"
    root = _store(tmp_path)
    spans = [(1, list(range(8)))]
    tp0 = _real_d_tree(spans, None, "tp0")
    worker = _real_d_tree(spans, None, "worker")
    assert _rr().form_a_spill(tp0, _Pool(None, {s: f"s{s}" for s in range(8)}, root), 1) == 8
    _broadcast(roles, tp0, worker)
    assert _released(tp0) == [1] and _released(worker) == [1]


# ------------------------------------------------------------------------------------------ K23 host-only leaves
from test_k23_form_a_host_leaf_spill_1017 import SF, SPAN, _AnchorPool, _death_shape  # noqa: E402


def _k23(cache, pool, claimer):
    setattr(cache, SF.FLOOR_LOCAL_PP_ATTR, False)
    with mock.patch.object(R12, "role", return_value="host"):
        return cache._w3_arena_spill(pool, SPAN, claimer=claimer)


def test_k23_tp0_keeps_a_host_only_leaf_a_worker_cannot_delete():
    R12._reset_state_for_tests()
    cache, top, mid, leaf = _death_shape()
    with mock.patch.object(R12, "worker_nodelete_keys", return_value={R12.node_key(leaf)}):
        got = _k23(cache, _AnchorPool(), top)
    assert leaf in cache._collect_all_nodes(), "TP0 dropped a leaf a worker keeps (device-resident / has children)"
    assert not [e for e in R12._S.ledger if e[0] == R12.STATE and e[3] == 0 and (e[1], int(e[2])) == R12.node_key(leaf)]
    assert got == 0


def test_k23_without_a_worker_state_the_leaves_still_go():
    R12._reset_state_for_tests()
    cache, top, mid, leaf = _death_shape()
    with mock.patch.object(R12, "worker_nodelete_keys", return_value=set()):
        assert _k23(cache, _AnchorPool(), top) == 2 * SPAN
    assert leaf not in cache._collect_all_nodes()


def test_the_nodelete_set_is_published_and_read(tmp_path, roles, evict_mock, chan):
    me = threading.current_thread().name
    spans = [(1, list(range(8))), (101, list(range(8, 16)))]
    worker = _real_d_tree(spans, None, "worker")
    roles.now[me] = "worker"
    R12.register_tree(worker)
    R12._dev_refresh(worker)
    roles.now[me] = "host"
    nd = R12.worker_nodelete_keys(worker)
    assert {R12.node_key(n) for n in worker._t_nodes.values()} == nd     # both device-resident: neither deletable
    roles.now[me] = "worker"
    for n in worker._t_nodes.values():
        n.component_data[FULL].value = None
    R12._dev_refresh(worker)
    roles.now[me] = "host"
    assert R12.worker_nodelete_keys(worker) == set() and len(R12.worker_evicted_keys(worker)) == 2
