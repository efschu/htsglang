"""#752: the mamba read must not wait in the KV pool's index frame.

SUPERSEDED IN PART BY #904, and the surviving half is pinned here.

THE CRASH (specimen boot_735_hicache.log:1097-1196, PP0 05:55:39): under
--enable-hierarchical-cache the first GDN forward calls
``HybridReqToTokenPool.mamba2_layer_cache``, which consulted the installed
``layer_transfer_counter`` as ``wait_until(self.local_slot(layer_id))``.
``local_slot`` is the ``KVCache`` indexing half that a req-to-token pool
never had -- AttributeError, dead scheduler, gloo collateral on the peers.
On a lineage where it does not raise it waits on a KV-layer slot index,
which is a DIFFERENT layer's step. That finding stands and this file keeps
it: the KV frame is not the mamba frame, in either direction.

WHAT #752 GOT WRONG, and what #904 replaced: the fix was "consult nothing",
justified by "mamba states move as WHOLE blobs through PoolTransfer and are
complete before the batch launches". Traced rather than assumed, the blob
rides the same asynchronous per-layer ``load_stream`` loop as the KV, at its
own GLOBAL layer index, with no join to the compute stream before the
recurrent step reads it. The correct fix is to wait in the MAMBA transfer
frame -- see ``test_mamba_read_waits_for_transfer_904.py``, which owns that
contract. This file now guards only the negative: never ``local_slot``,
never a mamba slot id, and never a wait on a counter that has no mamba step.
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _code_lines(fn):
    """Source of ``fn`` with comment lines removed.

    The guard below is about what the code CALLS, not about what the comment
    is allowed to name -- and the comment must be free to name `local_slot`,
    because explaining why it is the wrong frame is the whole point of it.
    """
    import inspect

    return "\n".join(
        line
        for line in inspect.getsource(fn).splitlines()
        if not line.lstrip().startswith("#")
    )


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


def _hybrid_pool(counter, frame=None):
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

    pool = object.__new__(HybridReqToTokenPool)
    pool.mamba_map = {30: 0, 31: 1}
    pool.mamba_pool = _MambaPoolStub()
    pool.layer_transfer_counter = None
    pool.start_layer = 24
    pool._mamba_transfer_frame = None
    if counter is not None:
        pool.register_layer_transfer_counter(counter, mamba_transfer_frame=frame)
    return pool


class TestMambaPathNeverUsesTheKvFrame(CustomTestCase):
    def test_mamba_read_does_not_wait_on_a_kv_local_slot(self):
        """The #752 finding, kept live. With the mamba transfer frame
        registered the read DOES wait (that is #904) -- but on the global
        layer id, never on ``local_slot`` and never on the mamba slot."""
        spy = _CounterSpy()
        pool = _hybrid_pool(spy, frame=48)
        out = pool.mamba2_layer_cache(31)
        self.assertEqual(out, "state-for-slot-1")
        self.assertEqual(
            spy.waits,
            [31],
            "the mamba wait must be keyed by the GLOBAL layer id; 1 would be "
            "the mamba slot and 7 the pre-#752 local_slot form that crashed",
        )
        self.assertNotIn(1, spy.waits)
        self.assertNotIn(31 - pool.start_layer, spy.waits)

    def test_a_counter_with_no_mamba_step_is_not_consulted(self):
        """A KV-only controller counts attention layers only; waiting on it
        would block on another pool's progress or index past its events.
        This is the half of #752 that is still literally 'skip'."""
        spy = _CounterSpy()
        pool = _hybrid_pool(spy, frame=None)
        self.assertEqual(pool.mamba2_layer_cache(31), "state-for-slot-1")
        self.assertEqual(spy.waits, [])

    def test_without_a_counter_nothing_changes(self):
        pool = _hybrid_pool(None)
        self.assertEqual(pool.mamba2_layer_cache(30), "state-for-slot-0")

    def test_the_decision_is_documented_at_the_site(self):
        """A silent choice reads as an oversight and gets 'fixed' back --
        which is what happened here once already. Both tickets must be
        traceable from the read site."""
        import inspect

        from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

        src = inspect.getsource(HybridReqToTokenPool.mamba2_layer_cache)
        self.assertIn("752", src)
        self.assertIn("904", src)
        self.assertNotIn(
            "local_slot", _code_lines(HybridReqToTokenPool.mamba2_layer_cache)
        )


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
