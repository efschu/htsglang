"""#1012 -- hermetic, CPU-only, drives the two handover points directly.

Neither test needs a GPU, a model or a scheduler process. Both pin the fix in
BOTH directions, because both defects are of the class where the wrong sign is
silent: a request wrongly held leaks its first prefill's rows, and a retention
length wrongly returned as 0 empties the whole cache while every request still
answers 200.
"""

import sys
import types
import unittest


class _Req:
    def __init__(self, rid, output_ids=()):
        self.rid = rid
        self.output_ids = list(output_ids)


class _Batch:
    def __init__(self, reqs):
        self.reqs = list(reqs)


class Test1008HeldPredicate(unittest.TestCase):
    """A result still queued is not a missing result."""

    @staticmethod
    def _held(result_queue, batch):
        from sglang.srt.managers.scheduler import Scheduler

        fake = types.SimpleNamespace(result_queue=result_queue)
        return Scheduler._1008_held_indices(fake, batch)

    def test_pending_result_is_not_held(self):
        # The overlap case: the request's prefill batch is in flight, its
        # result pops later this iteration. Holding it here is what re-queued
        # it and orphaned its first prefill's KV rows.
        r = _Req("a")
        launched = _Batch([r])
        running = _Batch([r])
        self.assertEqual(self._held([(launched, object())], running), [])

    def test_absent_result_is_still_held(self):
        # The case #1008/#1010 were built on: no batch anywhere holds the
        # request and its result will never land. The guard must still fire --
        # this is the direction a careless fix would break.
        r = _Req("a")
        self.assertEqual(self._held([], _Batch([r])), [0])

    def test_only_the_pending_one_is_spared(self):
        pending, stranded, decoding = _Req("p"), _Req("s"), _Req("d", [7])
        launched = _Batch([pending])
        running = _Batch([pending, stranded, decoding])
        self.assertEqual(self._held([(launched, None)], running), [1])

    def test_identity_not_equality(self):
        # Two distinct requests with identical fields must not shadow each
        # other; the queue is matched by object identity.
        a, b = _Req("same"), _Req("same")
        self.assertEqual(self._held([(_Batch([a]), None)], _Batch([b])), [0])

    def test_no_result_queue_attribute(self):
        # PP loops build their own structures; a scheduler without the
        # attribute must fall back to the pre-#1012 behaviour, not crash.
        from sglang.srt.managers.scheduler import Scheduler

        fake = types.SimpleNamespace()
        self.assertEqual(Scheduler._1008_held_indices(fake, _Batch([_Req("a")])), [0])


class TestFinishedRetentionWithoutTrackedPosition(unittest.TestCase):
    """No tracked mamba position must not empty the full-attention cache."""

    def _component(self):
        from sglang.srt.mem_cache.unified_cache_components.mamba_component import (
            MambaComponent,
        )

        comp = MambaComponent.__new__(MambaComponent)
        comp.enable_mamba_extra_buffer = True
        comp.mamba_checkpoint_interval = 4096
        comp._off_grid_retention_declines = 0
        comp._991_absent_active_slot = 0
        comp.cache = types.SimpleNamespace()
        return comp

    def _req(self, tracked):
        return types.SimpleNamespace(
            rid="r0",
            mamba_pool_idx=object(),
            mamba_last_track_seqlen=tracked,
            cache_protected_len=0,
        )

    def test_no_tracked_position_declines_the_anchor_not_the_length(self):
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams

        comp = self._component()
        params = InsertParams()
        # `mamba_last_track_seqlen is None` is every request shorter than the
        # checkpoint interval. `0` here collapses the shared
        # effective_cache_len and caches NOTHING (measured: TREE CENSUS
        # nodes=1 after three finished requests). `None` means "no constraint
        # from mamba": the KV is cached at full length under a tombstone.
        out = comp.prepare_for_caching_req(
            req=self._req(None),
            insert_params=params,
            token_ids_len=40,
            is_finished=True,
        )
        self.assertIsNone(out, "a finished request must impose no length cap")
        self.assertIsNone(params.mamba_value, "the anchor must be declined")
        self.assertEqual(comp._off_grid_retention_declines, 1, "must be counted")

    def test_unfinished_step_still_caches_nothing(self):
        # The other half of #783 and the direction that must NOT move: an
        # unfinished step inserts and immediately re-matches, and an anchorless
        # node is unmatchable, so retaining there loses the rows instead.
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams

        comp = self._component()
        params = InsertParams()
        out = comp.prepare_for_caching_req(
            req=self._req(None),
            insert_params=params,
            token_ids_len=40,
            is_finished=False,
        )
        self.assertEqual(out, 0, "an unfinished step must still cache nothing")

    def test_unfinished_off_grid_still_caches_nothing(self):
        # Same direction as above, but through `_decline_retention` itself --
        # the branch an over-retaining edit there would silently flip. An
        # unfinished step that retained an anchorless node would hand the tree
        # rows the request keeps using and later frees.
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams

        comp = self._component()
        params = InsertParams()
        out = comp.prepare_for_caching_req(
            req=self._req(1234),
            insert_params=params,
            token_ids_len=4000,
            is_finished=False,
        )
        self.assertEqual(out, 0)

    def test_off_grid_tracked_position_unchanged(self):
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams

        comp = self._component()
        params = InsertParams()
        out = comp.prepare_for_caching_req(
            req=self._req(1234),  # not a multiple of 4096
            insert_params=params,
            token_ids_len=4000,
            is_finished=True,
        )
        self.assertIsNone(out)
        self.assertIsNone(params.mamba_value)


if __name__ == "__main__":
    unittest.main(verbosity=2, argv=[sys.argv[0]])
