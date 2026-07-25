"""HTCCL transport seam: pluggable (`shm | device | ucx`), no leaked assumptions.

Pins the refactor that replaced the per-transport attributes
(`self.shm_transport` / `self.device_transport`) and the caller-side slot-size
test with a registry plus a `handles(op, nbytes)` capability query.

Two things are being protected:
  1. BEHAVIOUR IS UNCHANGED. The selection the communicator makes today must
     equal the selection the old hard-coded conditions made -- including the
     quirk that shm serves all_reduce ONLY, and only within slot_bytes.
  2. A NEW TRANSPORT IS CONFIGURATION, NOT SURGERY. Registering one must make
     it dispatchable without touching any call site.

CPU only: nothing here constructs a real transport or touches a device.
"""

import unittest

from sglang.srt.distributed.device_communicators import htccl as htccl_mod
from sglang.srt.distributed.device_communicators.htccl import (
    TRANSPORT_REGISTRY,
    HTCCLCommunicator,
    _build_transport,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

SLOT = 64 * 1024 * 1024
OPS = ("all_reduce", "all_gather", "reduce_scatter")


class _FakeDevice:
    """Mirrors HTCCLDeviceTransport's declared capability."""

    HTCCL_OPS = frozenset(OPS)

    def handles(self, op, nbytes):
        return op in self.HTCCL_OPS


class _FakeShm:
    """Mirrors HTCCLShmTransport's declared capability."""

    HTCCL_OPS = frozenset({"all_reduce"})

    def __init__(self, slot_bytes=SLOT):
        self.slot_bytes = slot_bytes

    def handles(self, op, nbytes):
        return op in self.HTCCL_OPS and nbytes <= self.slot_bytes


def _select(transport, op, nbytes):
    """Drive HTCCLCommunicator._select without building a real communicator."""
    comm = HTCCLCommunicator.__new__(HTCCLCommunicator)
    comm.transport = transport
    return HTCCLCommunicator._select(comm, op, nbytes)


class TestCapabilityMatchesOldConditions(CustomTestCase):
    """Property 1: identical selection to the pre-refactor hard-coded logic."""

    def test_device_serves_all_three_ops_at_any_size(self):
        t = _FakeDevice()
        for op in OPS:
            for nbytes in (0, 1, SLOT - 1, SLOT, SLOT + 1, 1 << 30):
                self.assertIs(
                    _select(t, op, nbytes), t, f"{op}/{nbytes} should use device"
                )

    def test_shm_serves_only_all_reduce_and_only_within_the_slot(self):
        """The exact quirk the old caller-side condition encoded."""
        t = _FakeShm()
        self.assertIs(_select(t, "all_reduce", SLOT - 1), t)
        self.assertIs(_select(t, "all_reduce", SLOT), t)
        # Oversized all_reduce -> gloo plane, as before.
        self.assertIsNone(_select(t, "all_reduce", SLOT + 1))
        # all_gather / reduce_scatter were never shm-served. Pinned so the
        # KNOWN op-coverage gap cannot change silently in either direction.
        for op in ("all_gather", "reduce_scatter"):
            for nbytes in (1, SLOT, SLOT + 1):
                self.assertIsNone(_select(t, op, nbytes))

    def test_no_transport_means_gloo_plane(self):
        for op in OPS:
            self.assertIsNone(_select(None, op, 1))


class TestRegistry(CustomTestCase):
    """Property 2: adding a transport is one registry entry."""

    def test_known_names_are_registered(self):
        self.assertIn("device", TRANSPORT_REGISTRY)
        self.assertIn("shm", TRANSPORT_REGISTRY)

    def test_gloo_and_unknown_names_yield_the_inline_plane(self):
        for name in ("gloo", "", "ucx-not-yet", "nonsense"):
            self.assertIsNone(_build_transport(name, None, None, disabled=False))

    def test_world_size_one_never_builds_a_transport(self):
        for name in list(TRANSPORT_REGISTRY) + ["gloo"]:
            self.assertIsNone(_build_transport(name, None, None, disabled=True))

    def test_a_new_transport_becomes_dispatchable_without_touching_call_sites(self):
        """The whole point of the seam. Register a fake 'ucx', and it is
        selected for the ops it declares -- no dispatch code involved."""

        class _FakeUcx:
            HTCCL_OPS = frozenset({"all_reduce", "all_gather"})

            def handles(self, op, nbytes):
                return op in self.HTCCL_OPS

        made = _FakeUcx()
        TRANSPORT_REGISTRY["ucx-test"] = lambda cpu_group, device: made
        try:
            built = _build_transport("ucx-test", None, None, disabled=False)
            self.assertIs(built, made)
            self.assertIs(_select(built, "all_reduce", 1 << 20), made)
            self.assertIs(_select(built, "all_gather", 1 << 20), made)
            # Undeclared op falls through to gloo rather than erroring.
            self.assertIsNone(_select(built, "reduce_scatter", 1 << 20))
        finally:
            TRANSPORT_REGISTRY.pop("ucx-test", None)


class TestFallbackPolicy(CustomTestCase):
    """The device transport must NEVER silently degrade to gloo.

    CUDA graphs were allowed on the strength of it being graph-capturable; a
    CPU-orchestrated replacement would be captured and crash later. shm, by
    contrast, is allowed to fall back.
    """

    def _boom(self, *_a, **_k):
        raise RuntimeError("transport init failed")

    def test_device_failure_propagates(self):
        saved = TRANSPORT_REGISTRY["device"]
        TRANSPORT_REGISTRY["device"] = self._boom
        try:
            with self.assertRaises(RuntimeError):
                _build_transport("device", None, None, disabled=False)
        finally:
            TRANSPORT_REGISTRY["device"] = saved

    def test_shm_failure_falls_back_to_gloo(self):
        saved = TRANSPORT_REGISTRY["shm"]
        TRANSPORT_REGISTRY["shm"] = self._boom
        try:
            self.assertIsNone(
                _build_transport("shm", None, None, disabled=False)
            )
        finally:
            TRANSPORT_REGISTRY["shm"] = saved

    def test_device_is_the_only_no_fallback_transport(self):
        self.assertEqual(htccl_mod._NO_FALLBACK, frozenset({"device"}))


class TestRealTransportsDeclareTheExpectedCapability(CustomTestCase):
    """Guard against the real classes drifting from what the seam assumes,
    without importing anything device-specific at module scope."""

    def test_shm_class_declares_all_reduce_only(self):
        from sglang.srt.distributed.device_communicators.htccl_shm import (
            HTCCLShmTransport,
        )

        self.assertEqual(HTCCLShmTransport.HTCCL_OPS, frozenset({"all_reduce"}))
        self.assertTrue(hasattr(HTCCLShmTransport, "handles"))
        self.assertTrue(hasattr(HTCCLShmTransport, "htccl_all_reduce"))

    def test_device_class_declares_all_three(self):
        from sglang.srt.distributed.device_communicators.htccl_device import (
            HTCCLDeviceTransport,
        )

        self.assertEqual(HTCCLDeviceTransport.HTCCL_OPS, frozenset(OPS))
        for name in ("handles", "htccl_all_reduce", "htccl_all_gather",
                     "htccl_reduce_scatter"):
            self.assertTrue(hasattr(HTCCLDeviceTransport, name), name)


if __name__ == "__main__":
    unittest.main()
