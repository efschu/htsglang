"""#224: the pointer-I/O park contract, exercised on CPU without a transport.

The remote host-RAM tier (``mooncake``) is the one #224 promise that needs a
second machine, and this rig cannot reach one -- #659 measured Rig-2 RAM as
unroutable (75 MB/s on the only routable path, swap-backed). So the tier was
implemented, accepted by validation, and NEVER EXERCISED: the module's own
"supported park tiers" note marks ``file`` as "validated end-to-end on CPU"
and marks ``mooncake`` with no such claim, and the existing suite only ever
asserts that ``destinations_error(["local", "mooncake"])`` returns None -- a
CONFIG check, not a byte check. The one test double in the tree sets
``pointer_io = False``, so the branch mooncake uses had no coverage at all.

The transport needs hardware. The CONTRACT does not: a ``pointer_io`` tier
moves bytes as ``(data_ptr, nbytes)`` pairs, and ``make_tier`` already lets a
``dynamic`` tier opt into that same branch via ``extra_config["pointer_io"]``
(kv_session_spill_destination.py:507). So the branch can be driven with a fake
store that reads and writes through the pointer exactly as a registered-buffer
backend does, on CPU, hermetically.

What this does NOT claim: that mooncake works. It claims the pointer-I/O half
of the destination layer moves the right bytes in the right direction, so that
when a routable peer exists the remaining risk is the transport, not this code.
"""

import ctypes
import unittest

import torch

from sglang.srt.managers.kv_session_spill_destination import DestinationTier
from sglang.test.test_utils import CustomTestCase


class _PointerStore:
    """A registered-buffer backend, faked: it copies through the pointer.

    This is what mooncake's contract looks like from this side -- ``set``
    reads nbytes FROM the caller's buffer, ``get`` writes nbytes INTO it.
    Keeping a bytes copy is what lets the test assert direction, not just
    that a call happened.
    """

    def __init__(self):
        self.blobs = {}
        self.calls = []

    def set(self, key, target_location=None, target_sizes=None, value=None):
        self.calls.append(("set", key, target_location, target_sizes))
        if target_location is None or target_sizes is None:
            raise AssertionError("pointer_io set must carry (ptr, nbytes)")
        buf = (ctypes.c_char * int(target_sizes)).from_address(int(target_location))
        self.blobs[key] = bytes(buf)
        return True

    def get(self, key, target_location=None, target_sizes=None):
        self.calls.append(("get", key, target_location, target_sizes))
        blob = self.blobs.get(key)
        if blob is None:
            return False
        n = min(int(target_sizes), len(blob))
        ctypes.memmove(int(target_location), blob, n)
        return True


class _TensorStore:
    """The non-pointer contract (file/dynamic), as the control arm."""

    def __init__(self):
        self.blobs = {}

    def set(self, key, target_location=None, target_sizes=None, value=None):
        self.blobs[key] = value.clone()
        return True

    def get(self, key, target_location=None, target_sizes=None):
        blob = self.blobs.get(key)
        if blob is None:
            return None
        target_location.copy_(blob)
        return target_location


def _payload(n=257):
    """Odd length on purpose: a power-of-two size can hide an off-by-one in
    the nbytes arithmetic."""
    return torch.arange(n, dtype=torch.uint8)


class TestThePointerIoContractMovesBytes(CustomTestCase):
    def test_a_pointer_io_round_trip_is_byte_exact(self):
        store = _PointerStore()
        tier = DestinationTier("mooncake", store, pointer_io=True)
        src = _payload()

        self.assertTrue(tier.put("k1", src))

        dst = torch.zeros_like(src)
        self.assertTrue(tier.get_into("k1", dst))
        self.assertTrue(torch.equal(src, dst), (src[:8], dst[:8]))

    def test_the_pointer_call_carries_the_real_size_not_the_element_count(self):
        """nbytes, not numel: a multi-byte dtype would under-copy 4x if the
        element count were passed."""
        store = _PointerStore()
        tier = DestinationTier("mooncake", store, pointer_io=True)
        src = torch.arange(64, dtype=torch.int32)  # 256 bytes, 64 elements

        tier.put("k", src)

        _, _, _, nbytes = store.calls[0]
        self.assertEqual(nbytes, 256)

    def test_a_wide_dtype_round_trips_intact(self):
        store = _PointerStore()
        tier = DestinationTier("mooncake", store, pointer_io=True)
        src = torch.arange(37, dtype=torch.int64) * 7

        self.assertTrue(tier.put("k", src))
        dst = torch.zeros_like(src)
        self.assertTrue(tier.get_into("k", dst))
        self.assertTrue(torch.equal(src, dst))

    def test_a_non_contiguous_tensor_is_parked_as_its_values_not_its_stride(self):
        """put() calls .contiguous(); without it the pointer would address a
        strided view and park whatever lay between the elements."""
        store = _PointerStore()
        tier = DestinationTier("mooncake", store, pointer_io=True)
        base = torch.arange(128, dtype=torch.uint8)
        src = base[::2]  # non-contiguous
        self.assertFalse(src.is_contiguous())

        self.assertTrue(tier.put("k", src))

        dst = torch.zeros(src.numel(), dtype=torch.uint8)
        self.assertTrue(tier.get_into("k", dst))
        self.assertTrue(torch.equal(src.contiguous(), dst))

    def test_a_missing_key_is_a_clean_false_not_an_exception(self):
        tier = DestinationTier("mooncake", _PointerStore(), pointer_io=True)
        dst = torch.zeros(8, dtype=torch.uint8)
        self.assertFalse(tier.get_into("absent", dst))

    def test_a_raising_backend_is_swallowed_into_false(self):
        """The declared contract: a failing tier falls over to the next one,
        so a raise here would abort a spill that had somewhere else to go."""

        class _Boom:
            def set(self, *a, **k):
                raise RuntimeError("link down")

            def get(self, *a, **k):
                raise RuntimeError("link down")

        tier = DestinationTier("mooncake", _Boom(), pointer_io=True)
        self.assertFalse(tier.put("k", _payload()))
        self.assertFalse(tier.get_into("k", torch.zeros(4, dtype=torch.uint8)))


class TestTheTensorArmStillBehaves(CustomTestCase):
    """The control: the validated `file`/`dynamic` path must be unchanged by
    anything the pointer arm does."""

    def test_a_tensor_tier_round_trips(self):
        tier = DestinationTier("file", _TensorStore(), pointer_io=False)
        src = _payload()
        self.assertTrue(tier.put("k", src))
        dst = torch.zeros_like(src)
        self.assertTrue(tier.get_into("k", dst))
        self.assertTrue(torch.equal(src, dst))

    def test_a_tensor_tier_reports_a_miss_as_false(self):
        tier = DestinationTier("file", _TensorStore(), pointer_io=False)
        self.assertFalse(tier.get_into("absent", torch.zeros(4, dtype=torch.uint8)))


if __name__ == "__main__":
    unittest.main()
