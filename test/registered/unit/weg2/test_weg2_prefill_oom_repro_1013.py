# SPDX-License-Identifier: Apache-2.0
"""PREFILL-OOM fix 43, the two REPRO tests (NF 13.10.2026 11:59:14Z; P.log 1214208-1214253 of boot ..._b60520f01d_1013_073910).

Depends on nothing the fix adds except the attribute under test, so it runs RED on 0ea6d17efd (int24f6):
(A) ``estimate_payable`` counts the un-backable leaf [6193] (135872 rows) payable although the peel demotes its two
host-locked device children first and then cannot drop it -- 150656 instead of 320, the 320 rows the peel really delivered;
(B) the allocation raises a bare ``RuntimeError`` (no ``mem_cache.common.PrefillOutOfPages`` with the terms).
GREEN on the fix. Hermetic: stubs only, no CUDA.
"""

import os
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.mem_cache import common as mc  # noqa: E402
from sglang.srt.weg2 import pp_room_vote as PR  # noqa: E402


# ---------------------------------------------------------------------------
# (A) the walk follows the peel's order
# ---------------------------------------------------------------------------


class _CD:
    def __init__(self, n=0, lock=0, host_lock=0):
        self.value = torch.arange(n) if n else None
        self.lock_ref = lock
        self.host_lock_ref = host_lock


class _Node:
    def __init__(self, parent, *, n=0, backuped=False, evicted=False, lock=0, host_lock=0):
        self.id = id(self)
        self.parent = parent
        self.children = {}
        self.backuped = backuped
        self.evicted = evicted
        self.component_data = [_CD(n, lock, host_lock), _CD(), _CD()]
        if parent is not None:
            parent.children[self.id] = self


def _incident(child_host_lock=1, with_children=True):
    """[6182] 3008 locked -> [6190] 14464 backed -> [6193] 135872 UN-backed -> [6194] 192, [6195] 128 (backed by the tail
    write-through at 11:59:13, device nodes, holding the end anchor -> host-locked)."""
    root = _Node(None)
    a = _Node(root, n=3008, lock=1)
    b = _Node(a, n=14464, backuped=True)
    leaf = _Node(b, n=135872, backuped=False)
    kids = []
    if with_children:
        kids = [_Node(leaf, n=192, backuped=True, host_lock=child_host_lock),
                _Node(leaf, n=128, backuped=True, host_lock=child_host_lock)]
    return root, leaf, kids


def _tree(root, ongoing=None):
    return types.SimpleNamespace(
        root_node=root, ongoing_write_through=ongoing or {},
        cache_controller=types.SimpleNamespace(write_policy="write_back", mem_pool_host=None),
    )


def _payable(root, *, local_floor=True, host_free=0, ongoing=None):
    return PR.estimate_payable(_tree(root, ongoing), local_floor=local_floor, host_free=host_free)


class WalkFollowsThePeelTest(unittest.TestCase):
    def test_6193_is_not_payable_when_the_children_it_needs_gone_are_host_locked(self):
        """THE REPRO. The peel demotes [6194]/[6195] (320 rows, exactly what it delivered), then cannot clear them from the
        host (host-locked) -> [6193] cannot be dropped -> nothing behind it is paid. RED: 150656."""
        root, _, _ = _incident(child_host_lock=1)
        self.assertEqual(_payable(root), 320)

    def test_the_same_leaf_is_payable_when_the_demoted_children_can_be_cleared(self):
        root, _, _ = _incident(child_host_lock=0)
        self.assertEqual(_payable(root), 14464 + 135872 + 192 + 128)

    def test_a_childless_unbacked_leaf_is_dropped(self):
        root, _, _ = _incident(with_children=False)
        self.assertEqual(_payable(root), 14464 + 135872)

    def test_write_through_in_flight_or_no_local_floor_blocks_the_drop(self):
        root, leaf, _ = _incident(child_host_lock=0)
        self.assertEqual(_payable(root, ongoing={leaf.id: 1}), 320)
        self.assertEqual(_payable(root, local_floor=False), 320)

    def test_a_host_only_child_that_is_host_locked_blocks_the_drop_as_before(self):
        root = _Node(None)
        leaf = _Node(root, n=1000, backuped=False)
        _Node(leaf, backuped=True, evicted=True, host_lock=1)
        self.assertEqual(_payable(root), 0)
        root = _Node(None)
        leaf = _Node(root, n=1000, backuped=False)
        _Node(leaf, backuped=True, evicted=True)
        self.assertEqual(_payable(root), 1000)

    def test_a_demoted_chain_is_unchanged(self):
        """Backed nodes are demoted, no UD involved: the old arithmetic."""
        root = _Node(None)
        a = _Node(root, n=4096, backuped=True)
        _Node(a, n=4096, backuped=True)
        self.assertEqual(_payable(root), 8192)


# ---------------------------------------------------------------------------
# (B) the typed exception
# ---------------------------------------------------------------------------


class _Alloc:
    page_size = 64

    def __init__(self, free=13568):
        self._free = free

    def available_size(self):
        return self._free

    def alloc_extend(self, *a, **k):
        return None

    def backup_state(self):
        return "state"


def _alloc_tree(alloc, evictable=150336):
    tc = mock.MagicMock()
    tc.is_chunk_cache.return_value = False
    tc.token_to_kv_pool_allocator = alloc
    tc.uniform_avail_floor = None
    tc.uniform_admitted_since_floor = 0
    tc.evictable_size.return_value = evictable
    tc.full_evictable_size.return_value = evictable
    tc.available_and_evictable_str.return_value = f"avail={alloc._free} evictable={evictable}"
    return tc


class TypedExceptionTest(unittest.TestCase):
    def setUp(self):
        self._saved = list(mc._extend_relief_providers)
        mc.clear_extend_relief_providers()

    def tearDown(self):
        mc.clear_extend_relief_providers()
        mc._extend_relief_providers.extend(self._saved)

    def test_the_raise_names_its_terms_and_stays_a_runtime_error(self):
        t = torch.tensor([0], dtype=torch.int64)
        with self.assertRaises(mc.PrefillOutOfPages) as cm:
            mc.alloc_paged_token_slots_extend(
                tree_cache=_alloc_tree(_Alloc()), prefix_lens=t, prefix_lens_cpu=t, seq_lens=t, seq_lens_cpu=t,
                last_loc=t, extend_num_tokens=16384)
        e = cm.exception
        self.assertIsInstance(e, RuntimeError)
        self.assertIn("Prefill out of memory", str(e))
        self.assertEqual(e.need, 16384 + 64)
        self.assertEqual(e.free, 13568)
        self.assertEqual(e.evictable_reported, 150336)
        self.assertGreaterEqual(e.evict_asked, 0)


if __name__ == "__main__":
    unittest.main()
