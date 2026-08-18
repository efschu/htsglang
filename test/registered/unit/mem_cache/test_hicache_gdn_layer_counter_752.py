"""#752: the hicache layer counter is a KV-layer instrument -- the mamba
pool must not consult it.

THE CRASH (specimen boot_735_hicache.log:1097-1196, PP0 05:55:39): under
--enable-hierarchical-cache the first GDN forward calls
``HybridReqToTokenPool.mamba2_layer_cache``, which consulted the installed
``layer_transfer_counter``. On the review lineage that consult is
``wait_until(self.local_slot(layer_id))`` and ``local_slot`` is the KVCache
indexing half (utils/common.py) that the req-to-token pool never had --
AttributeError, dead scheduler, gloo collateral on the peers. On this
lineage the consult is ``wait_until(layer_id - self.start_layer)``, which
does not raise but waits on a KV-layer slot index that mamba transfers
never advance.

WHY SKIPPING IS CORRECT, not a weakening: the counter exists for the ONE
transfer that overlaps the forward -- the layer-by-layer KV load-back --
and the cache controller wires it to the FULL-ATTENTION pool alone
(cache_controller.py unwraps ``HybridLinearKVPool`` to ``full_kv_pool``
for every layer-wise op). Mamba states move as WHOLE BLOBS through the
mamba component's PoolTransfer and complete before the batch launches;
there is no per-layer mamba progress for the counter to report, so a wait
keyed by mamba layer id is a category error. Pinned in both directions:
the mamba path never consults the counter, the KV path still does.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _CounterSpy:
    def __init__(self):
        self.waits = []

    def wait_until(self, threshold):
        self.waits.append(threshold)


class _MambaPoolStub:
    def __init__(self):
        self.asked = []

    def mamba2_layer_cache(self, slot):
        self.asked.append(slot)
        return f"state-for-slot-{slot}"


def _hybrid_pool(counter):
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

    pool = object.__new__(HybridReqToTokenPool)
    pool.mamba_map = {30: 0, 31: 1}
    pool.mamba_pool = _MambaPoolStub()
    pool.layer_transfer_counter = counter
    pool.start_layer = 24
    return pool


class TestMambaPathSkipsTheLayerCounter(CustomTestCase):
    def test_mamba2_layer_cache_never_consults_the_counter(self):
        """RED-FIRST for #752: with a transfer counter installed (the
        hicache boot), the mamba cache read must neither raise nor wait --
        the counter has no slot that mamba transfers advance."""
        spy = _CounterSpy()
        pool = _hybrid_pool(spy)
        out = pool.mamba2_layer_cache(31)
        self.assertEqual(out, "state-for-slot-1")
        self.assertEqual(
            spy.waits,
            [],
            "mamba2_layer_cache consulted the KV layer counter; on the "
            "review lineage that exact consult is the AttributeError crash "
            "of the first GDN forward under hicache",
        )

    def test_without_a_counter_nothing_changes(self):
        pool = _hybrid_pool(None)
        self.assertEqual(pool.mamba2_layer_cache(30), "state-for-slot-0")

    def test_the_skip_is_documented_at_the_site(self):
        """A silent skip reads as an oversight and gets 'fixed' back. The
        decision must be stated where the counter is NOT consulted."""
        import inspect

        from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

        src = inspect.getsource(HybridReqToTokenPool.mamba2_layer_cache)
        low = src.lower()
        self.assertIn("whole", low)
        self.assertIn("layer", low)
        self.assertTrue("#752" in src or "752" in src)


class TestKvPathStillWaits(CustomTestCase):
    """The can-fail direction: the guard must not defang the KV pool's
    layer wait, which IS the overlapped transfer the counter exists for.
    Mutating the mamba skip into a blanket removal reds this."""

    def _kv_pool(self, counter):
        import torch

        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        pool = object.__new__(MHATokenToKVPool)
        pool.layer_transfer_counter = counter
        pool.start_layer = 0
        # Contiguous ownership (no SGLANG_PP_LAYER_SET): local_slot then
        # reduces to layer_id - start_layer, the pre-layer-set arithmetic.
        pool._local_slot_of = None
        pool.dtype = torch.float16
        pool.store_dtype = torch.float16
        pool.k_buffer = [torch.zeros(1, 1, 1, dtype=torch.float16)]
        pool.v_buffer = [torch.zeros(1, 1, 1, dtype=torch.float16)]
        return pool

    def test_key_buffer_read_waits_on_the_counter(self):
        spy = _CounterSpy()
        pool = self._kv_pool(spy)
        pool.get_key_buffer(0)
        self.assertEqual(spy.waits, [0], "the KV layer wait must survive #752")


if __name__ == "__main__":
    unittest.main()
