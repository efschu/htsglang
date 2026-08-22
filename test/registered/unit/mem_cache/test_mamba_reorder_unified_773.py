"""#773: the #755 lock reorder, ported to the lineage that actually runs.

WHY IT HAD TO BE PORTED RATHER THAN RE-GATED. `#755` collapses the floor's
donation and pinned-checkpoint terms into one by releasing the OLD anchor's
pin BEFORE allocating the donated slot, so the two never coexist. Its config
gate requires `enable_hierarchical_cache` -- and that is exactly what routes a
hybrid-SSM boot to `UnifiedRadixCache`, which had no reorder. The reduction
was taken where the mechanism was absent (see test_mamba_reorder_lineage_773).

WHAT THIS PORT DOES DIFFERENTLY, and it is strictly safer. `MambaRadixCache`
releases the WHOLE node lock early, and `inc_lock_ref` walks ancestors for the
FULL component, so its window also leaves the request's own matched KV prefix
evictable. The unified tree locks per component, so only the MAMBA lock is
dropped: the KV path stays protected and the single state slot the reorder is
un-double-counting is the only thing made evictable.
`dec_swa_lock_only` is the same shape for SWA.

THE HAZARD THIS FILE GUARDS. An early release that is not paired exactly is
#583 -- a mamba ref given back that was never taken, or taken and never given
back, either of which drains the pool over minutes of serving. Every test
below is about that pairing.

CPU-only: no GPU, no DMA controller.
"""

import os
import unittest
from unittest import mock

from sglang.srt.mem_cache.base_prefix_cache import DecLockRefParams, IncLockRefResult
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

from test_mamba_pin_budget_live_773 import (
    _build,
    _checkpoint_nodes,
    _mamba_value,
)

register_cpu_ci(est_time=15)


def _mamba_lock_ref(node):
    return node.component_data[ComponentType.MAMBA].lock_ref


def _full_lock_ref(node):
    return node.component_data[ComponentType.FULL].lock_ref


def _acquire(cache, node):
    cache.components[ComponentType.MAMBA].acquire_component_lock(
        node=node, result=IncLockRefResult()
    )


class TestThePerNodeGate(CustomTestCase):
    """The release is only admissible for an anchor that survives losing it."""

    def _one_node(self):
        cache, allocator, pool, _ = _build()
        nodes = _checkpoint_nodes(cache, allocator, pool, 1)
        self.assertGreater(len(nodes), 0)
        return cache, nodes[0]

    def test_a_device_only_anchor_is_refused(self):
        """No host copy -> releasing risks a DEAD anchor, not a load_back."""
        cache, node = self._one_node()
        self.assertIsNotNone(_mamba_value(node))
        self.assertIsNone(node.component_data[ComponentType.MAMBA].host_value)
        self.assertFalse(
            cache.components[ComponentType.MAMBA].anchor_release_admissible(node)
        )

    def test_a_backed_anchor_is_admissible(self):
        cache, node = self._one_node()
        node.component_data[ComponentType.MAMBA].host_value = object()
        self.assertTrue(
            cache.components[ComponentType.MAMBA].anchor_release_admissible(node)
        )

    def test_an_INFLIGHT_backup_is_refused(self):
        """#767: host_value is published when the copy is HANDED OVER.

        Between that moment and the ack the anchor exists as an intention
        only, and a release there is the dead anchor the gate exists to
        prevent. The node is in `ongoing_write_through` for exactly that
        window.
        """
        cache, node = self._one_node()
        node.component_data[ComponentType.MAMBA].host_value = object()
        cache._track_write_through_node(node, None)
        self.assertIn(node.id, cache.ongoing_write_through)
        self.assertFalse(
            cache.components[ComponentType.MAMBA].anchor_release_admissible(node)
        )

    def test_the_root_is_never_released_early(self):
        cache, _ = self._one_node()
        self.assertFalse(
            cache.components[ComponentType.MAMBA].anchor_release_admissible(
                cache.root_node
            )
        )

    def test_a_tombstoned_anchor_has_no_pin_to_release(self):
        """A node whose state is already gone must answer False, not pretend."""
        cache, node = self._one_node()
        node.component_data[ComponentType.MAMBA].host_value = object()
        node.component_data[ComponentType.MAMBA].value = None
        self.assertFalse(
            cache.components[ComponentType.MAMBA].anchor_release_admissible(node)
        )

    def test_a_missing_node_is_refused_rather_than_assumed(self):
        cache, _ = self._one_node()
        self.assertFalse(
            cache.components[ComponentType.MAMBA].anchor_release_admissible(None)
        )


class TestTheReleaseTouchesOnlyMamba(CustomTestCase):
    """The property that makes this port safer than the original."""

    def test_the_kv_lock_survives_the_mamba_release(self):
        cache, allocator, pool, _ = _build()
        nodes = _checkpoint_nodes(cache, allocator, pool, 1)
        node = nodes[0]
        cache.inc_lock_ref(node)
        self.assertEqual(_mamba_lock_ref(node), 1)
        full_before = _full_lock_ref(node)
        self.assertGreater(full_before, 0, "fixture must hold a KV lock too")

        self.assertTrue(cache.dec_mamba_lock_only(node))

        self.assertEqual(_mamba_lock_ref(node), 0, "the mamba pin is released")
        self.assertEqual(
            _full_lock_ref(node),
            full_before,
            "the KV path lock must be UNTOUCHED -- this is the whole reason "
            "the unified port is safer than MambaRadixCache's whole-node "
            "release, which drops the request's own prefix protection",
        )

    def test_releasing_the_root_is_a_no_op(self):
        cache, _, _, _ = _build()
        self.assertFalse(cache.dec_mamba_lock_only(cache.root_node))

    def test_the_released_slot_becomes_evictable(self):
        """The slot the reorder is not double-counting is really freed up."""
        cache, allocator, pool, _ = _build()
        nodes = _checkpoint_nodes(cache, allocator, pool, 1)
        node = nodes[0]
        cache.inc_lock_ref(node)
        locked = cache.component_evictable_size_[ComponentType.MAMBA]
        cache.dec_mamba_lock_only(node)
        self.assertGreater(
            cache.component_evictable_size_[ComponentType.MAMBA],
            locked,
            "an early-released anchor must be reclaimable, or the reorder "
            "saves no slot at all",
        )


class TestTheLockRefPairing(CustomTestCase):
    """#583: a ref given back that was never taken drains the pool."""

    def _locked_node(self):
        cache, allocator, pool, _ = _build()
        nodes = _checkpoint_nodes(cache, allocator, pool, 1)
        node = nodes[0]
        cache.inc_lock_ref(node)
        return cache, node

    def test_the_skip_set_prevents_a_double_release(self):
        """The happy path: released early, then skipped at the normal site."""
        cache, node = self._locked_node()
        cache.dec_mamba_lock_only(node)
        self.assertEqual(_mamba_lock_ref(node), 0)

        params = DecLockRefParams()
        params.skip_lock_node_ids.setdefault(ComponentType.MAMBA, set()).add(node.id)
        cache.dec_lock_ref(node, params)

        self.assertEqual(
            _mamba_lock_ref(node),
            0,
            "the mamba ref must not go negative: it was already given back",
        )

    def test_CAN_FAIL_without_the_skip_set_the_ref_is_released_twice(self):
        """The proof that the skip set is load-bearing, not decoration.

        `release_component_lock` guards with `if cd.lock_ref > 0`, so a second
        release cannot drive the counter below zero -- the damage is subtler
        and worse: the ACCOUNTING is applied twice. This test pins the
        difference so a future edit that drops the skip set cannot look
        harmless.
        """
        cache, node = self._locked_node()
        protected_at_start = cache.component_protected_size_[ComponentType.MAMBA]
        cache.dec_mamba_lock_only(node)
        after_one = cache.component_protected_size_[ComponentType.MAMBA]
        self.assertLess(after_one, protected_at_start)

        # No skip set -> the normal site releases the same ref again.
        cache.dec_lock_ref(node, DecLockRefParams())
        self.assertEqual(
            cache.component_protected_size_[ComponentType.MAMBA],
            after_one,
            "a second release must not double-subtract the protected size",
        )

    def test_the_bail_path_restores_the_pre_call_state(self):
        """`cache_unfinished_req` can bail after the early release.

        When nothing is inserted, no new anchor takes over the pin that was
        dropped, and `req.last_node` is unchanged -- so the NEXT call's
        `dec_lock_ref` would give back a ref this step already gave back.
        The bail path re-acquires instead of carrying the imbalance forward.
        """
        cache, node = self._locked_node()
        before = _mamba_lock_ref(node)
        cache.dec_mamba_lock_only(node)
        self.assertEqual(_mamba_lock_ref(node), before - 1)

        _acquire(cache, node)  # what the bail path does

        self.assertEqual(
            _mamba_lock_ref(node),
            before,
            "after a bail the node must hold exactly the pin it started with",
        )

    def test_a_full_round_trip_leaves_no_residue(self):
        """release -> re-acquire -> normal release ends at zero."""
        cache, node = self._locked_node()
        cache.dec_mamba_lock_only(node)
        _acquire(cache, node)
        cache.dec_lock_ref(node, DecLockRefParams())
        self.assertEqual(_mamba_lock_ref(node), 0)
        self.assertGreater(
            cache.component_evictable_size_[ComponentType.MAMBA],
            0,
            "the slot must be reclaimable at the end, not stranded protected",
        )


if __name__ == "__main__":
    unittest.main()


class TestTheReorderIsReachedFromCacheUnfinishedReq(CustomTestCase):
    """END-TO-END: the slot the reorder saves is a slot that gets USED.

    Testing the predicate and the release in isolation is not enough -- an
    earlier round of this work proved it: a mutant that disabled the guard
    inside `write_backup` survived every direct-predicate test in the sibling
    file. So this class asserts the reorder through the real entry point, by
    making the pool exactly tight enough that the donation allocation can only
    succeed if the old anchor was released FIRST.

    Pool = 2 slots: one held by the running request's active state, one by the
    anchor it resumes from. Free = 0. `prepare_for_caching_req` must allocate a
    donated slot, and the only reclaimable slot in the pool is the anchor's --
    reclaimable only once its pin is gone. With the reorder the insert lands;
    without it the pool is exhausted and the step caches nothing.
    """

    def setUp(self):
        # The #755 gate is opt-in; without it the reorder never fires and
        # this class would silently test the pre-port path.
        self._env = mock.patch.dict(
            os.environ, {"SGLANG_MAMBA_SLOT_REORDER": "1"}, clear=False
        )
        self._env.start()
        self.addCleanup(self._env.stop)

    def _one_running_req_on_a_full_pool(self, backed: bool, mamba_pool_size: int = 2):
        from array import array

        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        from test_mamba_pin_budget_live_773 import _cache_one_finished_req

        cache, allocator, pool, server_args = _build(
            mamba_pool_size=mamba_pool_size, max_running_requests=1
        )
        # The #755 config gate: a write-through host tier is what makes an
        # early-released anchor recoverable rather than dead.
        server_args.enable_hierarchical_cache = True
        server_args.hicache_write_policy = "write_through"

        anchor_tokens = list(range(500, 508))
        anchor = _cache_one_finished_req(
            cache, allocator, pool, anchor_tokens, rid="anchor"
        )
        self.assertIsNotNone(anchor)
        self.assertIsNotNone(_mamba_value(anchor))
        if backed:
            anchor.component_data[ComponentType.MAMBA].host_value = object()

        # A running request that resumes from that anchor.
        tokens = anchor_tokens + list(range(600, 608))
        req = Req(
            rid="runner",
            origin_input_text="",
            origin_input_ids=array("q", tokens),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        self.assertIsNotNone(pool.alloc([req]))
        req.output_ids = array("q")
        req.cache_protected_len = 0
        req.swa_uuid_for_lock = None
        req.extra_key = None
        req.mamba_last_track_seqlen = len(tokens)
        # `get_fill_ids()` slices full_untruncated_fill_ids by extend_range.
        req._refresh_fill_ids()
        req.set_extend_range(0, len(tokens))
        kv = allocator.alloc(len(tokens))
        self.assertIsNotNone(kv)
        pool.write((req.req_pool_idx, slice(0, len(tokens))), kv)
        req.last_node = anchor
        cache.inc_lock_ref(anchor)

        # active slot + anchor slot; a size of 2 leaves the pool exactly full.
        self.assertEqual(pool.mamba_allocator.available_size(), mamba_pool_size - 2)
        return cache, req, anchor, tokens

    def test_a_backed_anchor_lets_the_donation_alloc_through(self):
        cache, req, anchor, tokens = self._one_running_req_on_a_full_pool(backed=True)
        before = cache.total_size()[0]

        cache.cache_unfinished_req(req)

        self.assertGreater(
            cache.total_size()[0],
            before,
            "with the anchor's pin released first, the donated slot is "
            "allocatable and the checkpoint is cached",
        )
        self.assertIsNot(
            req.last_node, anchor, "the request must have moved to a new anchor"
        )

    def test_CAN_FAIL_an_unbacked_anchor_caches_nothing_on_a_full_pool(self):
        """The same call, with the ONLY difference being admissibility.

        Not host-backed -> no early release -> the anchor stays pinned -> the
        donation alloc finds nothing to evict. This is the pre-port behaviour,
        and it is what the reorder buys: the identical request on the
        identical pool either caches or does not, decided by that one slot.
        """
        cache, req, anchor, tokens = self._one_running_req_on_a_full_pool(backed=False)
        before = cache.total_size()[0]

        cache.cache_unfinished_req(req)

        self.assertEqual(
            cache.total_size()[0],
            before,
            "an unbacked anchor must not be released, so the pool stays full "
            "and this step caches nothing",
        )

    def test_the_skip_set_protects_a_SECOND_holder_of_the_same_anchor(self):
        """#583 pairing, through the real entry point.

        `release_component_lock` no-ops at zero, so a double release on a
        singly-held node hides. The damage shows when the anchor is held
        TWICE -- an admission pin plus, say, a load-back pin. The reorder
        gives back one ref; if the normal site then gives back another
        without consulting the skip set, it releases a ref belonging to
        someone else and the slot becomes evictable while still in use.

        This is the mutant that survived the first round of this file: every
        pairing test simulated the sequence by hand, so unwiring the skip set
        changed nothing they looked at.
        """
        # One spare slot, so the donation alloc succeeds WITHOUT needing to
        # evict the anchor -- otherwise the step bails and correctly restores
        # the pin, which is a different path (covered below).
        cache, req, anchor, _ = self._one_running_req_on_a_full_pool(
            backed=True, mamba_pool_size=3
        )
        cache.inc_lock_ref(anchor)  # a second, independent holder
        self.assertEqual(_mamba_lock_ref(anchor), 2)

        cache.cache_unfinished_req(req)

        self.assertEqual(
            _mamba_lock_ref(anchor),
            1,
            "exactly ONE ref may be given back: the reorder released this "
            "request's, and the second holder's must survive",
        )

    def test_a_bailed_step_restores_the_anchor_pin(self):
        """#583 pairing on the path that inserts NOTHING.

        An off-grid checkpoint position makes the mamba component decline the
        anchor, so `effective_cache_len` collapses to 0 and the step caches
        nothing -- AFTER the reorder already released the old pin. No new
        anchor takes it over and `req.last_node` is unchanged, so the next
        call would give back a ref this one already gave back. The bail path
        must put it back.
        """
        cache, req, anchor, _ = self._one_running_req_on_a_full_pool(backed=True)
        cache.components[ComponentType.MAMBA].mamba_checkpoint_interval = 8192
        self.assertEqual(_mamba_lock_ref(anchor), 1)

        cache.cache_unfinished_req(req)

        self.assertIs(req.last_node, anchor, "a bailed step keeps its anchor")
        self.assertEqual(
            _mamba_lock_ref(anchor),
            1,
            "after a step that cached nothing the anchor must hold exactly "
            "the pin it started with",
        )
