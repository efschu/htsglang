"""#790 -- the #767 carry instrument must not charge the path it observes.

THE ERROR CLASS, stated once: an instrument that violates the path it is
watching. The #767 carry counter logs a WARNING from inside prefill admission
(`alloc` <- `alloc_for_extend` <- `prepare_for_extend` <-
`get_new_batch_prefill`) and passed `req.mamba_pool_idx` -- a 1-element CUDA
tensor -- as a format argument. Formatting a device tensor calls
`Tensor.__repr__`, which is a D2H copy plus a stream synchronize, because that
is the only way to put the number on the host. On a device busy with a spinning
kernel that sync does not return: PP0's MainThread sat inside `logging.emit`
for 25+ minutes while PP1/PP2 starved on `pp_chain_receiver.recv`.

TWO HALVES, and only the first was already in the tree at cc5babf995:

  1. the ARGUMENT is sync-free (`sync_free_tensor_repr`, which reports
     shape/dtype/device/id and deliberately never the slot NUMBER). Present
     before this change; pinned here because nothing pinned it, and a future
     edit that interpolates the tensor back in would restore the wedge.
  2. the LINE only fires when asked for. A rate limit is NOT a gate:
     `n <= 3 or n % 500 == 0` still emitted at WARNING level on a healthy
     server. This is the half added here.

The counters stay unconditional on purpose: two integer increments, no device
contact, still readable from a debugger on a server nobody started with the
gate on -- which is the situation an incident is actually diagnosed in.

SCOPE, so this file does not restate its sibling. `test_admission_log_no_device_
sync_790.py` already drives the REAL `alloc` branch through a tripwire and pins
half 1 there, including a negative control; the gate is pinned on that same real
path in that file too. What is pinned HERE is the instrument's own contract in
isolation: the three exits, the counters' independence from the gate, the rate
limit, and that the admission path holds no logging call of its own.
"""

import inspect
import logging
import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache import memory_pool as mp
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_MODULE_LOGGER = "sglang.srt.mem_cache.memory_pool"


class _HostReadError(AssertionError):
    """Raised when anything tries to materialize the tensor's VALUE."""


class _ExplodingTensor(torch.Tensor):
    """A tensor that is safe to hold and fatal to read.

    Every route to the VALUE raises; the metadata a sync-free formatter is
    allowed to use (shape, dtype, device) keeps working. On a real CUDA tensor
    those same routes would block on a stream sync instead of raising, which is
    the failure this stands in for -- a hang cannot be asserted on, so it is
    turned into an exception.
    """

    @staticmethod
    def __new__(cls, data):
        return torch.Tensor._make_subclass(cls, data, False)

    def _boom(self, *args, **kwargs):
        raise _HostReadError(
            "the instrument tried to materialize the slot value on the host; "
            "on a busy device this is the #790 wedge"
        )

    __repr__ = _boom
    __str__ = _boom
    __format__ = _boom
    item = _boom
    tolist = _boom
    numpy = _boom
    cpu = _boom
    __int__ = _boom
    __float__ = _boom


def _req(slot, cow=None, rid="r0", needs_clear=False):
    class _R:
        pass

    r = _R()
    r.mamba_pool_idx = slot
    r.mamba_cow_src_index = cow
    r.rid = rid
    r.mamba_needs_clear = needs_clear
    return r


def _pool():
    class _P:
        pass

    return _P()


class TestExplodingTensorIsAFaithfulStandIn(CustomTestCase):
    """The instrument of this test file, tested before it is trusted.

    A stand-in that quietly failed to explode would make every assertion below
    pass for the wrong reason -- the indicator has to be checked in BOTH
    directions before any finding rests on it.
    """

    def test_it_really_raises_on_every_host_route(self):
        t = _ExplodingTensor(torch.tensor([7]))
        for name, call in (
            ("repr", lambda: repr(t)),
            ("str", lambda: str(t)),
            ("format", lambda: f"{t}"),
            ("percent-s", lambda: "%s" % (t,)),
            ("item", t.item),
            ("tolist", t.tolist),
            ("cpu", t.cpu),
            ("int", lambda: int(t)),
            ("float", lambda: float(t)),
        ):
            with self.assertRaises(_HostReadError, msg=f"{name} did not raise"):
                call()

    def test_metadata_stays_readable(self):
        t = _ExplodingTensor(torch.tensor([7]))
        self.assertEqual(tuple(t.shape), (1,))
        self.assertEqual(t.device.type, "cpu")
        self.assertIsNotNone(t.dtype)


class TestTheGate(CustomTestCase):
    """Half 2: off by default, and the counters do not depend on it."""

    def test_silent_by_default(self):
        pool, req = _pool(), _req(_ExplodingTensor(torch.tensor([7])))
        with envs.SGLANG_DEBUG_MAMBA_CARRY.override(False):
            with self.assertNoLogs(_MODULE_LOGGER, level=logging.DEBUG):
                mp.note_mamba_carry_without_copy(pool, req)

    def test_counters_increment_with_the_gate_off(self):
        pool, req = _pool(), _req(_ExplodingTensor(torch.tensor([7])))
        with envs.SGLANG_DEBUG_MAMBA_CARRY.override(False):
            for _ in range(4):
                mp.note_mamba_carry_without_copy(pool, req)
        self.assertEqual(pool._m767_carry_total, 4)
        self.assertEqual(pool._m767_carry_nocopy, 4)

    def test_speaks_when_asked(self):
        pool, req = _pool(), _req(_ExplodingTensor(torch.tensor([7])))
        with envs.SGLANG_DEBUG_MAMBA_CARRY.override(True):
            with self.assertLogs(_MODULE_LOGGER, level=logging.WARNING) as cm:
                mp.note_mamba_carry_without_copy(pool, req)
        self.assertIn("#767 carry-without-copy #1", cm.output[0])

    def test_emitting_never_reads_the_slot_value(self):
        # The gate being ON is exactly when the device is busy, so the argument
        # must stay sync-free behind the gate too. The record is formatted here
        # (assertLogs renders it), and the tensor would raise if touched.
        pool, req = _pool(), _req(_ExplodingTensor(torch.tensor([7])))
        with envs.SGLANG_DEBUG_MAMBA_CARRY.override(True):
            with self.assertLogs(_MODULE_LOGGER, level=logging.WARNING) as cm:
                mp.note_mamba_carry_without_copy(pool, req)
        self.assertIn("<tensor shape=(1,)", cm.output[0])

    def test_a_pending_copy_is_not_a_carry_without_copy(self):
        pool = _pool()
        req = _req(_ExplodingTensor(torch.tensor([7])), cow=object())
        with envs.SGLANG_DEBUG_MAMBA_CARRY.override(True):
            with self.assertNoLogs(_MODULE_LOGGER, level=logging.DEBUG):
                mp.note_mamba_carry_without_copy(pool, req)
        self.assertEqual(pool._m767_carry_total, 1)
        self.assertFalse(hasattr(pool, "_m767_carry_nocopy"))

    def test_the_rate_limit_survives_the_gate(self):
        pool, req = _pool(), _req(_ExplodingTensor(torch.tensor([7])))
        with envs.SGLANG_DEBUG_MAMBA_CARRY.override(True):
            with self.assertLogs(_MODULE_LOGGER, level=logging.WARNING) as cm:
                for _ in range(10):
                    mp.note_mamba_carry_without_copy(pool, req)
        # 1, 2, 3 and then nothing until 500.
        self.assertEqual(len(cm.output), 3)
        self.assertEqual(pool._m767_carry_nocopy, 10)


class TestAdmissionPathIsClean(CustomTestCase):
    """Pin: the admission path holds no ungated log call of its own.

    A source pin because driving `alloc` needs a real pool, allocator and
    device. It is narrow: it fails if a logging call is ever put back inline in
    the branch, which is the exact regression.
    """

    def test_alloc_delegates_the_instrument(self):
        src = inspect.getsource(mp.HybridReqToTokenPool.alloc)
        self.assertIn("note_mamba_carry_without_copy(self, req)", src)
        self.assertNotIn("logger.warning", src)
        self.assertNotIn("carry-without-copy", src)

    def test_the_instrument_is_gated_and_sync_free(self):
        src = inspect.getsource(mp.note_mamba_carry_without_copy)
        self.assertIn("envs.SGLANG_DEBUG_MAMBA_CARRY.get()", src)
        self.assertIn("sync_free_tensor_repr(", src)
        # The tensor itself must never reach the format arguments.
        self.assertNotIn("req.mamba_pool_idx,", src)


if __name__ == "__main__":
    unittest.main()
