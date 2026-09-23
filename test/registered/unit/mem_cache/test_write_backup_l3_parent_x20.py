"""fnFL2x19/x20 (2026-09-23): a parent whose store write acked is backed.

THE LOOP. Under the shared arena group P's host copy is transit: the store ack
frees a node's host rows again (``_drain_backup`` ->
``_weg2_release_chain_piece_host``), and the node is left ``l3_present`` but no
longer ``backuped``. ``write_backup`` enforces the write-through contiguity law
by backing an un-backed parent up first -- and asked only ``backuped``. So at the
sleep flush every child backup (and every poll of a refused child) copied the
whole acked chain above it to the host again and wrote it to the store again,
whose ack freed it again. Measured on P's PP1/PP2: 1088 / 1024 store writes for
the 64 nodes of one 259k prompt, the in-flight count pinned at 54,
``/flush_cache`` answering 400 for 90 s, the P->D flip stopped on W3. PP0 and
boot fnFL2x18 escaped only because their acks landed after the sweep had issued
every node.

``publish_unbacked_sweep`` already treats ``l3_present`` as backed (#1317
C2/R-5); the parent law must ask the same question.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import types
import unittest

import torch

from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.test.test_utils import CustomTestCase

from test_unified_radix_cache_unittest import CacheConfig, build_fixture


class _RecordingController:
    """The two members ``write_backup`` touches on its way to a D->H copy."""

    write_policy = "write_back"

    def __init__(self):
        self.copied = []
        self.mem_pool_host = types.SimpleNamespace(available_size=lambda: 1 << 20)

    def write(self, device_indices, node_id=None, extra_pools=None):
        self.copied.append(node_id)
        return torch.arange(len(device_indices), dtype=torch.int64)


def _chain():
    """root -> parent (tokens 1..16) -> child (17..32), both on the device."""
    cache, allocator, _ = build_fixture(
        CacheConfig(page_size=1, components=(ComponentType.FULL,))
    )
    for n in (16, 32):
        value = allocator.alloc(n)
        cache.insert(
            InsertParams(
                key=RadixKey(list(range(1, n + 1)), None),
                value=value.to(dtype=torch.int64),
            )
        )
    parent = next(iter(cache.root_node.children.values()))
    child = next(iter(parent.children.values()))
    cc = _RecordingController()
    cache.cache_controller = cc
    return cache, parent, child, cc


class TestAStoredParentIsNotCopiedAgain(CustomTestCase):
    def test_the_child_backup_leaves_an_acked_parent_alone(self):
        """RED ON 0f614eebfa: the parent is copied again before the child."""
        cache, parent, child, cc = _chain()
        parent.l3_present = True  # store ack landed, transit host copy freed
        self.assertFalse(parent.backuped)

        self.assertGreater(cache.write_backup(child), 0)

        self.assertEqual(
            cc.copied,
            [child.id],
            "the acked parent was copied to the host again; at the flush every "
            "child re-issues its whole stored chain and the write-throughs "
            "never drain",
        )

    def test_a_parent_that_is_nowhere_is_still_backed_first(self):
        """The contiguity law itself stays: a parent neither on the host nor in
        the store is backed up before its child."""
        cache, parent, child, cc = _chain()
        self.assertFalse(parent.backuped)
        self.assertFalse(parent.l3_present)

        self.assertGreater(cache.write_backup(child), 0)

        self.assertEqual(cc.copied, [parent.id, child.id])


if __name__ == "__main__":
    unittest.main()
