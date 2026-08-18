"""#745: the anchor WRITE path is reachable on the composite lineage.

WHY THIS EXISTS. comp4's load window showed ZERO anchor lines under
``--mamba-checkpoint-interval 8192``. One reading is the missing emitter
(#758, being built); the #742-class worse reading is that the write path
itself is silent-inert on the composite's cherry-picked #747 stack -- the
flag accepted, the grid never consulted, the harvest boot's ARM II
acceptance a hope. This suite discriminates the two AT THE DESK: it drives
the real chunk-end chain of the unified mamba component (the live
hierarchical-cache lineage) across two anchor boundaries and asserts every
link.

THE CHAIN, each link asserted:

    chunk end (n x 512 tokens)                 scheduler cache_unfinished
      -> prepare_for_caching_req  no_buffer    grid gate (#747 retention
         seam): off-grid ends cache NOTHING,   seam mirrors :652-659)
         the 16th chunk end (8192 = 16 x 512,
         legal per #750 divisibility) DONATES
      -> commit_insert_component_data          the donated value becomes a
                                               radix-retained node
      -> build_hicache_transfers(BACKUP_HOST)  the node yields a MAMBA
                                               PoolTransfer -- host-tier
                                               eligible, the #747
                                               composition claim

Red-first is the mutation run recorded in the commit: breaking the grid
rule (is_on_interval always-False) produces ZERO anchors across the whole
drive and reds the boundary assertions.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

INTERVAL = 8192
CHUNK = 512


class _CopySpy:
    def __init__(self):
        self.copies = []

    def copy_from(self, src, dst):
        self.copies.append((src, dst))

    replayssm_write_pos = None


def _component(interval=INTERVAL):
    from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
        MambaComponent,
    )

    comp = object.__new__(MambaComponent)
    comp.enable_mamba_extra_buffer = False
    comp.mamba_checkpoint_interval = interval
    pool_spy = _CopySpy()
    comp.cache = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            mamba_pool=pool_spy,
            mamba_ckpt_pool=None,
            translate_mamba_indices=lambda ids: ids,
        ),
    )
    comp._allocs = []

    def _alloc():
        slot = torch.tensor([40 + len(comp._allocs)], dtype=torch.int64)
        comp._allocs.append(slot)
        return slot

    comp._alloc_mamba_slot = _alloc
    return comp, pool_spy


def _req():
    return SimpleNamespace(
        rid="anchor-drive", mamba_pool_idx=torch.tensor(3, dtype=torch.int64)
    )


def _insert_params():
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams

    return InsertParams(prev_prefix_len=0)


class TestChunkEndDecision(CustomTestCase):
    """Link 1: the grid decision fires at EXACTLY every 16th chunk end."""

    def test_two_boundaries_over_33_chunk_ends(self):
        comp, pool = _component()
        anchored = []
        for n in range(1, 34):  # 512 .. 16896: crosses 8192 and 16384
            token_len = n * CHUNK
            out = comp.prepare_for_caching_req(
                req=_req(),
                insert_params=_insert_params(),
                token_ids_len=token_len,
                is_finished=False,
            )
            if out and out > 0:
                anchored.append((n, out))
        self.assertEqual(
            anchored,
            [(16, 8192), (32, 16384)],
            "the anchor decision must fire at the 16th and 32nd chunk ends "
            "and NOWHERE between -- comp4's zero-anchor window is either the "
            "emitter or THIS assertion",
        )
        self.assertEqual(
            len(comp._allocs), 2, "exactly one donation slot per anchor"
        )
        self.assertEqual(len(pool.copies), 2, "each donation copies the state")

    def test_off_grid_ends_never_touch_the_allocator(self):
        """Cache-nothing must mean NOTHING: no slot alloc, no copy, no
        donated value on the params -- a skipped step may not leak."""
        comp, pool = _component()
        params = _insert_params()
        out = comp.prepare_for_caching_req(
            req=_req(), insert_params=params, token_ids_len=15 * CHUNK,
            is_finished=False,
        )
        self.assertEqual(out, 0)
        self.assertEqual(comp._allocs, [])
        self.assertEqual(pool.copies, [])
        self.assertIsNone(params.mamba_value)

    def test_interval_off_donates_every_chunk_end(self):
        """The gate is what sparsifies: with no interval the same drive
        donates at every end (the pre-interval behaviour) -- proving this
        harness can tell a dead gate from a sparse one."""
        comp, _ = _component(interval=None)
        fired = 0
        for n in range(1, 9):
            out = comp.prepare_for_caching_req(
                req=_req(), insert_params=_insert_params(),
                token_ids_len=n * CHUNK, is_finished=False,
            )
            fired += 1 if out and out > 0 else 0
        self.assertEqual(fired, 8)


class TestAnchorBecomesARetainedHostEligibleNode(CustomTestCase):
    """Links 2+3, walked END TO END with the SAME value object: the donated
    slot becomes a radix-retained node, and that node yields a BACKUP_HOST
    transfer -- the #747 'host-tier-eligible like any other retained node'
    claim, executed rather than quoted."""

    def _commit_env(self, comp):
        from sglang.srt.mem_cache.unified_cache_components.tree_component import (
            ComponentType,
        )

        ct = ComponentType.MAMBA

        class _Lru:
            def __init__(self):
                self.mru = []

            def insert_mru(self, node):
                self.mru.append(node)

            def in_list(self, node):
                return False

            def reset_node_mru(self, node):
                pass

        comp.cache.lru_lists = {ct: _Lru()}
        comp.cache.host_lru_lists = {ct: _Lru()}
        comp.cache.component_evictable_size_ = {ct: 0}
        return ct

    def _node(self, ct):
        return SimpleNamespace(
            component_data={
                ct: SimpleNamespace(
                    value=None, host_value=None, lock_ref=0, host_lock_ref=0
                )
            },
            hash_value=["h"],
        )

    def test_donated_value_reaches_the_host_transfer(self):
        from sglang.srt.mem_cache.unified_cache_components.tree_component import (
            CacheTransferPhase,
        )
        from sglang.srt.mem_cache.hicache_storage import PoolName

        comp, _ = _component()
        ct = self._commit_env(comp)

        # Link 1: the 16th chunk end donates.
        params = _insert_params()
        out = comp.prepare_for_caching_req(
            req=_req(), insert_params=params, token_ids_len=16 * CHUNK,
            is_finished=False,
        )
        self.assertEqual(out, 8192)
        self.assertIsNotNone(params.mamba_value)

        # Link 2: the donation commits onto an on-grid leaf and is retained.
        node = self._node(ct)
        params.key = list(range(8192))  # on-grid key length for the backstop
        result = SimpleNamespace(mamba_exist=False)
        comp.commit_insert_component_data(
            node=node, is_new_leaf=True, params=params, result=result
        )
        self.assertIs(node.component_data[ct].value, params.mamba_value)
        self.assertIn(node, comp.cache.lru_lists[ct].mru)
        self.assertEqual(comp.cache.component_evictable_size_[ct], 1)

        # Link 3: the retained anchor is host-writeback eligible.
        transfers = comp.build_hicache_transfers(
            node, CacheTransferPhase.BACKUP_HOST
        )
        self.assertIsNotNone(transfers, "the anchor node must yield a host transfer")
        self.assertEqual(len(transfers), 1)
        self.assertEqual(transfers[0].name, PoolName.MAMBA)
        self.assertIs(transfers[0].device_indices, params.mamba_value)

    def test_an_off_grid_commit_never_reaches_the_host_tier(self):
        """The backstop direction: an off-grid leaf is refused at commit
        (tombstone), and a tombstone yields NO host transfer."""
        from sglang.srt.mem_cache.unified_cache_components.tree_component import (
            CacheTransferPhase,
        )

        comp, _ = _component()
        ct = self._commit_env(comp)
        node = self._node(ct)
        params = _insert_params()
        params.mamba_value = torch.tensor([7], dtype=torch.int64)
        params.key = list(range(8192 + CHUNK))  # off-grid
        comp._off_grid_insert_refusals = 0
        result = SimpleNamespace(mamba_exist=False)
        comp.commit_insert_component_data(
            node=node, is_new_leaf=True, params=params, result=result
        )
        self.assertTrue(result.mamba_exist, "off-grid commit must be refused")
        self.assertIsNone(node.component_data[ct].value)
        self.assertIsNone(
            comp.build_hicache_transfers(node, CacheTransferPhase.BACKUP_HOST)
        )


if __name__ == "__main__":
    unittest.main()
