# SPDX-License-Identifier: Apache-2.0
"""UD-H: a refused write_back leaf with host-only children is dropped on the local-PP floor.

NF int19 (617c6b6541), P log boot_weg2_dkrnfint4h6ablxcbar1dauer10081332_617c6b6541_
1008_133232.P.log 13:37:49Z PP2: ``EVICT-FRONTIER-CENSUS request=16448
delivered_before=9600 ... reported_evictable=132224 on_frontier=3456
behind_device_child=128768 leaves 2->2``, then ``EXTEND-RELIEF evicted=0 asked=16384
avail 15232->15232 evictable_left=132224`` -> ``Prefill out of memory ... Try to
allocate 16384 tokens`` -> RANK-DEATH. Same death in int18 without PW (12:57:14Z
PP2, ``EXTEND-RELIEF evicted=0 asked=6702``) and int16 (10:37:41Z PP0). The shape the
int18 log names (12:53:27Z PP2 ``#1421 ... node=341 backuped=True parent=277
parent_backuped=False``): a backed child under an un-backed parent; once the child
went to the host, the parent is a device leaf whose backup the full arena refuses
and which UD may not drop (#841: it has a child) -- the peel stops there.

Driven through the REAL ``UnifiedRadixCache._evict_device_leaf`` (and the real
``_ud_clear_host_children`` / ``_is_host_leaf``); the tree's I/O steps are stubs.
RED on ccf0daefbe: the leaf stays, 0 paid. GREEN: the host-only child leaves the
host, the leaf is dropped, its rows are paid. Off the local-PP floor (D, any TP
group) and with a host-locked child nothing changes.
"""

import os
import types
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.mem_cache.unified_cache_components.tree_component import (  # noqa: E402
    BASE_COMPONENT_TYPE as FULL,
)
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from flliper.srt.pdflip import pp_slot_fidelity as SF  # noqa: E402

LEAF_ROWS = 3456  # 'on_frontier=3456'
HOST_ROWS = 4096


class _CD:
    def __init__(self, value=None, host_value=None, host_lock=0):
        self.value = value
        self.host_value = host_value
        self.lock_ref = 0
        self.host_lock_ref = host_lock


class _Node:
    _n = 0

    def __init__(self, parent, *, device=0, host=0, evicted=False, backuped=False, host_lock=0):
        _Node._n += 1
        self.id = _Node._n
        self.parent = parent
        self.children = {}
        self.evicted = evicted
        self.backuped = backuped
        self.component_data = [
            _CD(list(range(device)) if device else None, list(range(host)) if host else None, host_lock),
            _CD(), _CD(),
        ]
        if parent is not None:
            parent.children[self.id] = self


class _Tree:
    _evict_device_leaf = UnifiedRadixCache._evict_device_leaf
    _is_device_leaf = UnifiedRadixCache._is_device_leaf
    _is_host_leaf = UnifiedRadixCache._is_host_leaf
    _ud_clear_host_children = getattr(UnifiedRadixCache, "_ud_clear_host_children", None)

    def __init__(self, local_pp=True):
        self.root_node = _Node(None)
        self.cache_controller = types.SimpleNamespace(write_policy="write_back")
        self.ongoing_write_through = {}
        self.tree_components = (FULL,)
        setattr(self, SF.FLOOR_LOCAL_PP_ATTR, local_pp)
        self.dropped, self.host_evicted = [], []

    def write_backup(self, node, write_back=False, kv_only_if_mamba_refused=False):
        return 0  # '#1421 BACKUP-REFUSED why=arena_claim'

    def writing_check(self, write_back=False):
        return None

    def _evict_host_leaf(self, node, tracker):
        assert self._is_host_leaf(node)
        node.parent.children.pop(node.id)
        tracker[FULL] += len(node.component_data[FULL].host_value)
        self.host_evicted.append(node.id)

    def _ud_drop_unbacked_leaf(self, node, tracker):
        assert not node.children, "#841: never drop a node with children"
        node.parent.children.pop(node.id)
        tracker[FULL] += len(node.component_data[FULL].value)
        self.dropped.append(node.id)


def _shape(tree, host_lock=0):
    leaf = _Node(tree.root_node, device=LEAF_ROWS)  # un-backed write_back leaf
    child = _Node(leaf, host=HOST_ROWS, evicted=True, backuped=True, host_lock=host_lock)
    return leaf, child


class UdHostChildrenTest(unittest.TestCase):
    def setUp(self):
        self.env = {SF.ENV: os.environ.get(SF.ENV)}
        os.environ[SF.ENV] = "1"

    def tearDown(self):
        if self.env[SF.ENV] is None:
            os.environ.pop(SF.ENV, None)
        else:
            os.environ[SF.ENV] = self.env[SF.ENV]

    def test_13_37_49_the_blocking_leaf_is_paid(self):
        tree = _Tree()
        leaf, child = _shape(tree)
        self.assertTrue(tree._is_device_leaf(leaf), "a frontier leaf: its child holds no device rows")
        tracker = {FULL: 0}
        tree._evict_device_leaf(leaf, tracker)
        self.assertEqual(tracker[FULL], LEAF_ROWS)
        self.assertEqual(tree.host_evicted, [child.id])
        self.assertEqual(tree.dropped, [leaf.id])

    def test_a_host_locked_child_keeps_everything(self):
        tree = _Tree()
        leaf, _ = _shape(tree, host_lock=1)
        tracker = {FULL: 0}
        tree._evict_device_leaf(leaf, tracker)
        self.assertEqual((tracker[FULL], tree.dropped, tree.host_evicted), (0, [], []))

    def test_off_the_local_pp_floor_nothing_changes(self):
        """D / any TP group: a rank-local drop would split the replicas."""
        tree = _Tree(local_pp=False)
        leaf, _ = _shape(tree)
        tracker = {FULL: 0}
        tree._evict_device_leaf(leaf, tracker)
        self.assertEqual((tracker[FULL], tree.dropped, tree.host_evicted), (0, [], []))

    def test_a_device_grandchild_keeps_everything(self):
        tree = _Tree()
        leaf, child = _shape(tree)
        _Node(child, device=64)  # a device row below: not a host-only subtree
        tracker = {FULL: 0}
        tree._evict_device_leaf(leaf, tracker)
        self.assertEqual((tracker[FULL], tree.dropped, tree.host_evicted), (0, [], []))


if __name__ == "__main__":
    unittest.main()
