# SPDX-License-Identifier: Apache-2.0
"""S41 (18.10., review of S39 FORMA-L3-SUBTREE): two conditions on the S39 drop.

A1 RANK DISAGREEMENT (R11, raenge-nie-uneins). The device life of a node is rank-local on D (S22: TP0 5090, the
workers 3080, every rank peels its own LRU). S39's TP0 check read the workers' S22 ``nd`` set only for the CHILDLESS
nodes of the subtree -- for an interior node ``nd`` says nothing (every interior node has children). A subtree whose
interior node a worker still holds on its device: TP0 dropped it and sent L3DROP, the worker's ``apply_drop`` met
``child_on_device`` and answered DIVERGED -- for good, a Form A worker never evicts host on its own and TP0 never
names the node again. Now every worker also publishes its device-resident keys (``dv``) and TP0 leaves such a drop
alone (KEPT ``worker_cannot_delete``), like the S22 rule for spills.

A2 THE PEEL MUST DELIVER BEFORE PW-R ANSWERS. PW-R (``d_wall_head.pick_unservable_head``, W50) runs after the pass's
admission -- after the peel that QUEUED the store-backed leaves and before the broadcast that drops them. With TP0's
payable memo standing (``backup_wall_up``) PW-R could hand the head to P one broadcast before S39 makes its room
(a flip for a request D can hold: SCHEDULER-SOLL R7). TP0's half of PW-R now says False while its S39 queue holds
leaves of that tree; the group MIN keeps the head on D on every rank. The queue empties at the next broadcast
(dropped, or KEPT by name after MAX_PASSES), then PW-R decides as before; PW-F stays the replicated named end.
"""

import os
import threading

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from sglang.srt.mem_cache import form_a_host_shadow as R12
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.weg2 import d_park_runtime as DP
from sglang.srt.weg2 import d_wall_head as W
from sglang.srt.weg2 import forma_l3_subtree as S39

from test_weg2_read_room_d_1015 import _broadcast, _real_d_tree, _WbController, roles  # noqa: F401 (fixture)
from test_weg2_s39_forma_l3_subtree_1018 import _attached, _peel, _rank

FULL = ComponentType.FULL


@pytest.fixture(autouse=True)
def _clean():
    R12._reset_state_for_tests()
    S39._reset_for_tests()
    yield
    R12._reset_state_for_tests()
    S39._reset_for_tests()


def _rank2(roles, tag, child_on_device=False):
    """leaf 1..8 (device, l3_present, no host copy) -> child 9..12 -> grandchild 13..16, both host-only and
    l3_present; ``child_on_device``: this rank still holds the INTERIOR child on its device (its own device life)."""
    me = threading.current_thread().name
    roles.now[me] = "host" if tag == "tp0" else "worker"
    t = _real_d_tree([(1, list(range(8)))], None, tag)
    t.cache_controller = _WbController()
    leaf = t._t_nodes[1]
    leaf.component_data[FULL].host_value = None
    leaf.l3_present = True
    t.insert(InsertParams(key=RadixKey(list(range(1, 13)), None), value=t._t_alloc.alloc(12).to(dtype=torch.int64)))
    c = next(iter(leaf.children.values()))
    t.insert(InsertParams(key=RadixKey(list(range(1, 17)), None), value=t._t_alloc.alloc(16).to(dtype=torch.int64)))
    g = next(iter(c.children.values()))
    for node, host, hv in ((c, [90, 91, 92, 93], ["c9", "c10", "c11", "c12"]),
                           (g, [94, 95, 96, 97], ["g13", "g14", "g15", "g16"])):
        node.component_data[FULL].host_value = torch.tensor(host, dtype=torch.int64)
        node.hash_value = hv
        node.l3_present = True
    for node in ((g,) if child_on_device else (g, c)):
        t._t_alloc.free(node.component_data[FULL].value)
        node.component_data[FULL].value = None
    t.write_backup = lambda *a, **k: 0       # #1421 arena_claim: the arena is full
    t._t_child, t._t_grand = c, g
    return t


class _Arena:
    def __init__(self, path):
        self.path = path


def _publish_worker(roles, worker, monkeypatch, tmp_path):
    """The worker's consume of the PREVIOUS pass: its S22 key sets go next to the arena (here: tmp)."""
    arena = _Arena(str(tmp_path / "arena-0.bin"))
    monkeypatch.setattr(R12, "_tree_arena", lambda tree: arena)
    me = threading.current_thread().name
    roles.now[me] = "worker"
    R12._dev_refresh(worker)
    roles.now[me] = "host"


def test_s41_the_worker_publishes_its_device_keys_and_tp0_reads_them(roles, monkeypatch, tmp_path):
    """The channel: a worker that holds the interior child on its device names it in ``dv``; ``nd`` alone could not
    (the child has children on every rank)."""
    tp0, worker = _rank2(roles, "tp0"), _rank2(roles, "worker", child_on_device=True)
    _publish_worker(roles, worker, monkeypatch, tmp_path)
    kc, kg = R12.node_key(worker._t_child), R12.node_key(worker._t_grand)
    dv = R12.worker_device_keys(tp0)
    assert kc in dv and kg not in dv, dv
    assert kc in R12.worker_nodelete_keys(tp0)          # nd has it too -- but so has every interior node


def test_s41_tp0_never_drops_a_subtree_whose_interior_node_a_worker_holds_on_its_device(roles, monkeypatch, tmp_path):
    """A1, danger direction RANK DISAGREEMENT: the worker could not delete its device-resident child -- TP0 leaves the
    whole drop alone, both trees keep leaf, child and grandchild, nobody names DIVERGED."""
    tp0, worker = _rank2(roles, "tp0"), _rank2(roles, "worker", child_on_device=True)
    _peel(roles, tp0), _peel(roles, worker)
    _publish_worker(roles, worker, monkeypatch, tmp_path)
    _broadcast(roles, tp0, worker)
    assert S39._ST.n["diverged"] == 0, S39._ST.n
    assert S39._ST.n["dropped"] == 0, S39._ST.n
    for t in (tp0, worker):
        assert _attached(t, t._t_nodes[1]) and _attached(t, t._t_child) and _attached(t, t._t_grand)


def test_s41_the_same_subtree_off_every_device_still_leaves_on_both_ranks(roles, monkeypatch, tmp_path):
    """The guard is no blanket refusal: with the child off the worker's device too, the two-level subtree and the leaf
    leave on both ranks at the broadcast (children first), the leaf's device rows come free on both."""
    tp0, worker = _rank2(roles, "tp0"), _rank2(roles, "worker")
    free0 = (int(tp0._t_alloc.available_size()), int(worker._t_alloc.available_size()))
    _peel(roles, tp0), _peel(roles, worker)
    _publish_worker(roles, worker, monkeypatch, tmp_path)
    _broadcast(roles, tp0, worker)
    assert S39._ST.n["dropped"] == 1 and S39._ST.n["worker_dropped"] == 1 and S39._ST.n["diverged"] == 0, S39._ST.n
    freed = (int(tp0._t_alloc.available_size()) - free0[0], int(worker._t_alloc.available_size()) - free0[1])
    assert freed == (8, 8), freed
    for t in (tp0, worker):
        assert not t.root_node.children


def _wall(monkeypatch):
    monkeypatch.setattr(DP, "backup_wall_up", lambda tree: True)
    monkeypatch.setattr(W, "_payable", lambda tree: 0)


def _verdicts(tp0, worker):
    return [W.wall_unservable(tree=t, available=0, need=10 ** 6) for t in (tp0, worker)]


def test_s41_pw_r_waits_for_the_broadcast_that_drops_the_queued_leaves(roles, monkeypatch):
    """A2: after the peel queued the store-backed leaf, TP0's half of PW-R is False (the group MIN keeps the head on
    D); after the broadcast dropped it the queue is empty and PW-R decides from the wall again."""
    tp0, worker = _rank(roles, "tp0"), _rank(roles, "worker")
    _wall(monkeypatch)
    _peel(roles, tp0), _peel(roles, worker)
    assert S39.room_pending(tp0) == 8 and S39.room_pending(worker) == 0
    v = _verdicts(tp0, worker)
    assert v == [False, True] and min(v) is False, v
    _broadcast(roles, tp0, worker)
    assert S39.room_pending(tp0) == 0
    assert _verdicts(tp0, worker) == [True, True]


def test_s41_pw_r_is_not_held_forever_by_a_leaf_that_cannot_leave(roles, monkeypatch):
    """A2, danger direction LOOP: a queued leaf that cannot leave (its child is outside the store) holds PW-R only
    until S39 names it KEPT after MAX_PASSES broadcasts -- then PW-R answers the unservable head again."""
    tp0, worker = _rank(roles, "tp0", child_l3=False), _rank(roles, "worker", child_l3=False)
    _wall(monkeypatch)
    _peel(roles, tp0), _peel(roles, worker)
    assert _verdicts(tp0, worker)[0] is False
    for _ in range(S39.MAX_PASSES + 1):
        _broadcast(roles, tp0, worker)
    assert S39._ST.n["kept"] == 1 and S39.room_pending(tp0) == 0, S39._ST.n
    assert _verdicts(tp0, worker) == [True, True]
    for t in (tp0, worker):
        assert _attached(t, t._t_nodes[1]) and _attached(t, t._t_child)


def test_s41_another_trees_queue_does_not_hold_pw_r(roles, monkeypatch):
    """The hold reads the queue of THIS tree only."""
    tp0, worker = _rank(roles, "tp0"), _rank(roles, "worker")
    other = _rank(roles, "tp0")
    _wall(monkeypatch)
    _peel(roles, other)
    assert S39.room_pending(other) == 8 and S39.room_pending(tp0) == 0
    assert _verdicts(tp0, worker) == [True, True]
