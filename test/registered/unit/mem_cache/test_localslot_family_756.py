"""#756: the local_slot contract holds for the WHOLE KVCache family.

THE CRASH (comp3, 06:14:18, boot_735_comp3.log:1292-1392 -- the #752
sibling): under hicache the first full-attention forward calls
``HybridLinearKVPool.get_kv_buffer`` -> ``_wait_for_layer`` ->
``local_slot``, which reads ``self._local_slot_of`` -- assigned only in
``KVCache.__init__``, which HybridLinearKVPool (and MiniMaxSparseKVPool)
deliberately bypass. AttributeError ("Did you mean: local_slot?"), dead
scheduler. The two MiniMax hicache-integration tests were failing with the
same error before this fix -- pre-existing specimens of the same family bug.

THE FAMILY FIX, two layers:

* base cover: ``_local_slot_of = None`` as a KVCache CLASS attribute --
  every subclass that bypasses the init degenerates to the historic
  ``layer_id - start_layer`` subtraction instead of crashing;
* real ownership map on the two GLOBAL-frame wrappers that bypass the init
  (Hybrid, MiniMax), because a bare None would silently mis-slot under
  #753's gapped layer ownership: their waits would index another layer's
  counter slot, confidently.

Pinned here: both wrappers survive a counter-gated read (contiguous), both
build the REAL map under an active layer set, the base cover reaches every
KVCache subclass, and the standard pool's wait SURVIVES (the can-fail
direction -- defanging local_slot or the wait reds it).
"""

import os
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

SET_ENV = "SGLANG_PP_LAYER_SET"
WIRE_ENV = "SGLANG_PP_CROSSING_WIRE"


class _CounterSpy:
    def __init__(self):
        self.waits = []

    def wait_until(self, threshold):
        self.waits.append(threshold)


class _SubPoolStub:
    def get_key_buffer(self, layer_id):
        return f"k{layer_id}"

    def get_value_buffer(self, layer_id):
        return f"v{layer_id}"

    def get_kv_buffer(self, layer_id):
        return (f"k{layer_id}", f"v{layer_id}")


def _hybrid(counter, start_layer=24):
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    pool = object.__new__(HybridLinearKVPool)
    pool.layer_transfer_counter = counter
    pool.start_layer = start_layer
    pool.full_kv_pool = _SubPoolStub()
    pool._transfer_full_attention_id = lambda lid: lid - start_layer
    return pool


def _minimax(counter):
    from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool

    pool = object.__new__(MiniMaxSparseKVPool)
    pool.layer_transfer_counter = counter
    pool.start_layer = 0
    pool.main_pool = _SubPoolStub()
    return pool


class TestTheCrashSites(CustomTestCase):
    """The comp3 specimen and its MiniMax sibling, on __new__ instances --
    which is exactly the bypass-the-init state the wrappers are in."""

    def test_hybrid_counter_gated_read_does_not_raise(self):
        spy = _CounterSpy()
        pool = _hybrid(spy, start_layer=24)
        out = pool.get_kv_buffer(30)
        self.assertEqual(out, ("k6", "v6"))
        self.assertEqual(spy.waits, [6], "contiguous default: id - start_layer")

    def test_minimax_counter_gated_read_does_not_raise(self):
        spy = _CounterSpy()
        pool = _minimax(spy)
        self.assertEqual(pool.get_key_buffer(3), "k3")
        self.assertEqual(spy.waits, [3])

    def test_the_wait_survives(self):
        """CAN-FAIL: the counter consult must not have been defanged to fix
        the crash -- the overlapped KV load-back depends on it."""
        spy = _CounterSpy()
        pool = _hybrid(spy)
        pool.get_key_buffer(24)
        pool.get_value_buffer(25)
        self.assertEqual(spy.waits, [0, 1])

    def test_no_counter_stays_inert(self):
        self.assertEqual(_hybrid(None).get_key_buffer(24), "k0")


class TestTheBaseCoverReachesEverySubclass(CustomTestCase):
    def test_every_kvcache_subclass_inherits_the_attribute(self):
        """The one-line cover, proven over the LIVE class tree rather than a
        hand-list: every subclass -- including future ones -- resolves
        ``_local_slot_of`` without its __init__ having run."""
        from sglang.srt.mem_cache import memory_pool as mp

        def walk(cls):
            yield cls
            for sub in cls.__subclasses__():
                yield from walk(sub)

        classes = list(walk(mp.KVCache))
        self.assertGreater(len(classes), 5, "the subclass walk found nothing")
        for cls in classes:
            with self.subTest(cls=cls.__name__):
                self.assertIsNone(getattr(cls, "_local_slot_of"))

    def test_a_bypassing_instance_degenerates_to_the_subtraction(self):
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        pool = object.__new__(HybridLinearKVPool)
        pool.start_layer = 24
        self.assertEqual(pool.local_slot(30), 6)


class TestLayerSetsStillMapForReal(CustomTestCase):
    """#753's gapped future: a bare None-only cover would silently wait on
    another layer's counter slot. Both wrappers must build the REAL map when
    a layer set is active -- proven through their actual __init__ source, and
    behaviorally through the base init's own helper."""

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in (SET_ENV, WIRE_ENV)}
        self.addCleanup(self._restore)

    def _restore(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_both_wrappers_build_the_ownership_map_in_init(self):
        import inspect

        from sglang.srt.mem_cache.memory_pool import (
            HybridLinearKVPool,
            MiniMaxSparseKVPool,
        )

        for cls in (HybridLinearKVPool, MiniMaxSparseKVPool):
            with self.subTest(cls=cls.__name__):
                src = inspect.getsource(cls.__init__)
                self.assertIn("_owned_layers_for_pool", src)
                self.assertIn("_local_slot_of", src)

    def test_the_map_ranks_owned_layers_not_subtracts(self):
        """The mapping the wrappers now build, driven through the same
        helper the base init uses (`_owned_layers_for_pool` reads
        get_pp_group + get_pp_layer_set), under a gapped three-stage set."""
        from types import SimpleNamespace
        from unittest import mock

        import sglang.srt.distributed as dist
        from sglang.srt.mem_cache.memory_pool import _owned_layers_for_pool

        # #815: THE CROSSING WIRE IS NOW PART OF THE PREMISE, not decoration.
        # This test's whole point is a GAPPED ownership set, and 4b2e43465d
        # [#753] made a gapped set illegal unless the crossing wire carries it
        # (`parse_pp_layer_sets(..., allow_gapped=)` is only passed True by
        # `get_pp_layer_set` when `pp_crossing_wire_enabled()`). Without the
        # flag the refusal raises, `current_stage_layer_set` swallows it with a
        # bare `except Exception: return None`, and this test saw None.
        #
        # So the fix is to state the configuration the scenario actually needs,
        # NOT to drop the gap: a contiguous set would still map correctly under
        # plain subtraction and would stop testing anything. #753 built the wire
        # precisely so this configuration could be carried safely.
        os.environ[SET_ENV] = "0-15,32-39;16-31;40-47"
        os.environ[WIRE_ENV] = "1"
        group = SimpleNamespace(num_hidden_layers=48, rank_in_group=0, world_size=3)
        with mock.patch.object(dist, "get_pp_group", return_value=group):
            owned = _owned_layers_for_pool()
        self.assertIsNotNone(owned, "the layer set did not resolve")
        mapping = {layer: slot for slot, layer in enumerate(sorted(owned))}
        # Gapped ownership: layer 32 is stage 0's slot 16, while the
        # subtraction would confidently say 32.
        self.assertEqual(mapping[32], 16)
        self.assertEqual(mapping[0], 0)

    def test_the_gapped_set_is_refused_without_the_crossing_wire(self):
        """TODAY'S CONTRACT, and the reason the test above must name the wire.

        #753 refuses a gapped set the wire cannot carry. That refusal is not
        visible at this seam -- `current_stage_layer_set` catches everything and
        returns None -- so without this test the silent None would look like an
        ordinary "no layer set configured" and the test above would be one env
        var away from measuring nothing again.
        """
        from types import SimpleNamespace
        from unittest import mock

        import sglang.srt.distributed as dist
        from sglang.srt.mem_cache.memory_pool import _owned_layers_for_pool

        os.environ[SET_ENV] = "0-15,32-39;16-31;40-47"
        os.environ.pop(WIRE_ENV, None)
        group = SimpleNamespace(num_hidden_layers=48, rank_in_group=0, world_size=3)
        with mock.patch.object(dist, "get_pp_group", return_value=group):
            self.assertIsNone(
                _owned_layers_for_pool(),
                "a gapped set without the crossing wire must not resolve",
            )


class TestStandardPoolUnchanged(CustomTestCase):
    def test_mha_pool_wait_still_runs_through_local_slot(self):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        spy = _CounterSpy()
        pool = object.__new__(MHATokenToKVPool)
        pool.layer_transfer_counter = spy
        pool.start_layer = 0
        pool.dtype = torch.float16
        pool.store_dtype = torch.float16
        pool.k_buffer = [torch.zeros(1, 1, 1, dtype=torch.float16)]
        pool.v_buffer = [torch.zeros(1, 1, 1, dtype=torch.float16)]
        pool.get_key_buffer(0)
        self.assertEqual(spy.waits, [0])


if __name__ == "__main__":
    unittest.main()
