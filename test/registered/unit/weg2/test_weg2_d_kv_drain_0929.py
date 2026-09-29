"""D-KV-DRAIN (29.09., NF1d/NF1e): a pending shrink the RETAINED TREE holds.

THE METAL. NF1d z30y3f 09291811: the 245k cell ended at S7 and its end event
left ``pending S0`` (cap 32768) with the floor at 248832 -- no request held
those pages, the retained radix tree did. Nothing evicts a tree nobody asks
to evict (the peel runs only under allocation pressure, and under the pending
cap that pressure lands below it), so the stage stayed up until the next sleep
and the stage cell's expert rows stayed OFF (Grundgesetz: free VRAM is experts).

WHAT MUST HOLD. While a shrink is pending and its floor is blocked, the tick
demotes the tree nodes above the pending cap to the host -- only unlocked
nodes whose L2 copy is acked (nothing dropped, decode's pages untouched), the
whole device subtree children-first, and group-uniformly: a node moves on
every rank or on none. A hot node without its L2 copy gets its write-through
issued and waits. The next tick shrinks; the counters reach the RankState
stats file.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import logging  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from test_weg2_d_mem_sched_0929 import FLOOR_LADDER, _req, floor_env, tick_env  # noqa: E402,F401

PAGE = 64
CAP = 32768            # S0: page 512 is the last id under the cap


def _full():
    from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType

    return ComponentType.FULL


class _Node:
    _ids = 0

    def __init__(self, parent, key, pages, backed=True, lock=0):
        _Node._ids += 1
        self.id = _Node._ids
        self.parent = parent
        self.children = {}
        full = int(_full())
        self.component_data = [types.SimpleNamespace(value=None, lock_ref=0)
                               for _ in range(full + 1)]
        cd = self.component_data[full]
        cd.value = (None if pages is None else
                    torch.tensor([p * PAGE for p in pages], dtype=torch.int64))
        cd.lock_ref = lock
        self.backuped = backed
        self.l3_present = False
        if parent is not None:
            parent.children[key] = self


class _Tree:
    def __init__(self):
        self.root_node = _Node(None, None, None)
        self.disable = False
        self.tree_components = (_full(),)
        self.evicted, self.backups, self.dropped = [], [], []
        self.on_evict = None
        self.cache_controller = types.SimpleNamespace(write_policy="write_through")
        self.ongoing_write_through = {}

    def _is_device_leaf(self, node):
        cd = node.component_data[int(_full())]
        if cd.value is None or any(c.lock_ref > 0 for c in node.component_data):
            return False
        return all(c.component_data[int(_full())].value is None for c in node.children.values())

    def _evict_to_host(self, node, tracker):
        assert self._is_device_leaf(node) and node.backuped
        cd = node.component_data[int(_full())]
        tracker[_full()] += int(cd.value.numel())
        cd.value = None
        self.evicted.append(node.id)
        if self.on_evict is not None:
            self.on_evict(node)

    def _evict_device_leaf(self, node, tracker):
        assert self._is_device_leaf(node) and not node.backuped and node.l3_present
        cd = node.component_data[int(_full())]
        tracker[_full()] += int(cd.value.numel())
        cd.value = None
        self.dropped.append(node.id)

    def write_backup(self, node, write_back=False):
        self.backups.append(node.id)
        return 1


def _drain(tree, gmin=None, cap=CAP):
    from sglang.srt.weg2 import d_kv_drain

    return d_kv_drain.drain(tree, cap, PAGE, gmin)


def test_a_backed_leaf_above_the_cap_goes_to_the_host_the_rest_stays():
    t = _Tree()
    a = _Node(t.root_node, (1,), range(1, 40))            # below the cap: stays
    b = _Node(a, (2,), range(513, 520))                    # the retained tail: above
    res = _drain(t)
    assert t.evicted == [b.id] and a.component_data[int(_full())].value is not None
    assert res.nodes == 1 and res.tokens == 7 and not res.mismatch


def test_a_hot_inner_node_goes_with_its_device_subtree_children_first():
    t = _Tree()
    a = _Node(t.root_node, (1,), range(600, 610))          # hot
    b = _Node(a, (2,), range(1, 5))                        # cold child: must go first
    c = _Node(b, (3,), range(5, 9))
    _drain(t)
    assert t.evicted == [c.id, b.id, a.id]


def test_a_locked_node_is_never_touched():
    t = _Tree()
    _Node(t.root_node, (1,), range(600, 610), lock=1)      # a running request reads it
    res = _drain(t)
    assert t.evicted == [] and res.locked == 1 and res.nodes == 0


def test_a_hot_node_without_its_l2_copy_gets_its_write_through_and_waits():
    t = _Tree()
    a = _Node(t.root_node, (1,), range(600, 610), backed=False)
    res = _drain(t)
    assert t.evicted == [] and t.backups == [a.id]
    assert res.waiting_backup == 1 and res.backup_issued == 1


def _peer(shape_fn, flags):
    """A two-rank group: element-wise MIN of this rank's values and a peer's.
    Call 1 is the shape vote (``shape_fn`` maps ours to the peer's), call 2
    the flag vote (``flags`` = the peer's hot/dev/host/l3 vector, None = ours)."""
    calls = []

    def gmin(vals):
        vals = list(vals)
        calls.append(vals)
        peer = shape_fn(vals) if len(calls) == 1 else (vals if flags is None else flags)
        return [min(a, b) for a, b in zip(vals, peer)]
    return gmin


def test_a_node_hot_only_on_a_peer_moves_here_too():
    t = _Tree()
    a = _Node(t.root_node, (1,), range(1, 10))             # cold HERE
    n = 1
    # the peer: same shape, node 0 hot (-1) and device-on (-1), ok (1)
    gmin = _peer(lambda v: v, [-1] * n + [-1] * n + [1] * n + [0] * n)
    _drain(t, gmin)
    assert t.evicted == [a.id]


def test_a_node_the_peer_cannot_move_stays_on_every_rank():
    t = _Tree()
    _Node(t.root_node, (1,), range(600, 610))              # hot and ok HERE
    n = 1
    gmin = _peer(lambda v: v, [-1] * n + [-1] * n + [0] * n + [0] * n)  # peer unbacked
    res = _drain(t, gmin)
    assert t.evicted == [] and res.nodes == 0


def test_a_store_acked_node_without_host_copy_leaves_the_device_by_the_write_through_path():
    t = _Tree()
    a = _Node(t.root_node, (1,), range(600, 610), backed=False)
    a.l3_present = True                                    # the store holds its bytes
    res = _drain(t)
    assert t.dropped == [a.id] and t.backups == [] and res.l3_dropped == 1


def test_a_store_acked_node_under_write_back_is_not_dropped():
    t = _Tree()
    t.cache_controller.write_policy = "write_back"
    a = _Node(t.root_node, (1,), range(600, 610), backed=False)
    a.l3_present = True
    res = _drain(t)
    assert t.dropped == [] and t.evicted == [] and res.nodes == 0


def test_a_node_whose_write_through_is_out_gets_no_second_one():
    t = _Tree()
    a = _Node(t.root_node, (1,), range(600, 610), backed=False)
    t.ongoing_write_through[a.id] = object()
    res = _drain(t)
    assert t.backups == [] and res.waiting_backup == 1 and res.backup_issued == 0


def test_mixed_classes_across_ranks_wait():
    """host copy HERE, only the store copy on the peer: no uniform way -> wait."""
    t = _Tree()
    _Node(t.root_node, (1,), range(600, 610))              # backed here
    n = 1
    gmin = _peer(lambda v: v, [-1] * n + [-1] * n + [0] * n + [1] * n)
    res = _drain(t, gmin)
    assert t.evicted == [] and t.dropped == [] and res.nodes == 0


def test_a_tree_shape_that_differs_abstains():
    t = _Tree()
    _Node(t.root_node, (1,), range(600, 610))
    gmin = _peer(lambda v: [v[0] - 1, v[1], v[2], v[3]], None)
    res = _drain(t, gmin)
    assert res.mismatch and t.evicted == []


def _pending_with_tree(dsv, sched, caps, floor):
    t = _Tree()
    a = _Node(t.root_node, (1,), range(1, 400))
    _Node(a, (2,), range(400, 518))                        # pages 513..517 above the cap
    t.on_evict = lambda node: floor.__setitem__("page", 399)
    sched.tree_cache = t
    sched.running_batch.reqs = [_req("weg2-2-10", 32835, 258)]
    dsv.runtime_tick(sched)
    sched.running_batch.reqs = []
    floor["page"] = 517
    return t


def test_the_tick_drains_a_pending_shrink_held_by_the_tree_and_then_shrinks(floor_env, caplog):
    """NF1d: pending S0 with floor 248832 stayed until the sleep. With the
    drain the tree's tail goes to the host in the pending tick, and the next
    tick takes the shrink (the stage cell's expert rows come back)."""
    dsv, sched, caps, floor = floor_env
    t = _pending_with_tree(dsv, sched, caps, floor)
    caplog.set_level(logging.INFO)
    st = dsv.runtime_tick(sched)
    ms = getattr(sched, dsv.MEM_SCHED_ATTR)
    assert not st.changed and ms.pending == 0 and caps[-1] == FLOOR_LADDER[0]
    assert t.evicted, "the retained tail above the pending cap was never demoted"
    assert ms.counters["drain_nodes"] >= 1 and ms.counters["drain_tokens"] > 0
    assert any("WEG2 D-MEM-SCHED DRAIN" in m and "nodes=1" in m for m in caplog.messages)
    st = dsv.runtime_tick(sched)
    assert st.changed and ms.stage == 0 and ms.pending is None


def test_the_emergency_stop_keeps_the_tree(floor_env, monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_DISABLE_D_KV_DRAIN", "1")
    dsv, sched, caps, floor = floor_env
    t = _pending_with_tree(dsv, sched, caps, floor)
    dsv.runtime_tick(sched)
    dsv.runtime_tick(sched)
    ms = getattr(sched, dsv.MEM_SCHED_ATTR)
    assert t.evicted == [] and ms.pending == 0


def test_while_requests_run_the_drain_waits_for_its_round(floor_env):
    dsv, sched, caps, floor = floor_env
    t = _pending_with_tree(dsv, sched, caps, floor)
    sched.running_batch.reqs = [_req("small", 100, 5)]     # fits S0: the shrink stays pending
    from sglang.srt.weg2 import d_kv_drain

    ms = None
    for _ in range(d_kv_drain.DRAIN_EVERY + 1):
        dsv.runtime_tick(sched)
        ms = getattr(sched, dsv.MEM_SCHED_ATTR)
        if t.evicted:
            break
    assert t.evicted and ms._round % d_kv_drain.DRAIN_EVERY == 0


def test_the_counters_reach_the_rankstate_stats_file(floor_env):
    from sglang.srt.weg2 import rankstats

    dsv, sched, caps, floor = floor_env
    _pending_with_tree(dsv, sched, caps, floor)
    dsv.runtime_tick(sched)
    block = rankstats._mem_sched_block(sched)
    assert block is not None and block["drain_nodes"] >= 1 and block["pending"] == 0
    assert rankstats._mem_sched_block(types.SimpleNamespace()) is None
