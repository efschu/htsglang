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


class TestAsyncSeam(CustomTestCase):
    """The issue/wait split's dispatch contract (task #198).

    supports_async must depend only on group-uniform state (transport class
    and coverage), never on payloads -- the same property handles() has, and
    for the same reason: no rank may go async while a peer goes sync.
    """

    def _comm(self, transport, disabled=False):
        comm = HTCCLCommunicator.__new__(HTCCLCommunicator)
        comm.transport = transport
        comm.disabled = disabled
        return comm

    def test_sync_only_transport_does_not_support_async(self):
        comm = self._comm(_FakeDevice())
        self.assertFalse(comm.supports_async())
        self.assertIsNone(comm.all_reduce_async(__import__("torch").zeros(4)))

    def test_no_transport_does_not_support_async(self):
        comm = self._comm(None)
        self.assertFalse(comm.supports_async())

    def test_disabled_group_does_not_support_async(self):
        class _FakeAsync(_FakeDevice):
            def all_reduce_async(self, comm, inp):
                return "handle"

        comm = self._comm(_FakeAsync(), disabled=True)
        self.assertFalse(comm.supports_async())

    def test_async_transport_dispatches_issue_and_wait(self):
        import torch

        issued = []

        class _FakeAsync(_FakeDevice):
            def all_reduce_async(self, comm, inp):
                issued.append(inp)
                return ("h", inp)

            def wait_async(self, handle):
                return handle[1] * 2

        comm = self._comm(_FakeAsync())
        self.assertTrue(comm.supports_async())
        x = torch.ones(4)
        h = comm.all_reduce_async(x)
        self.assertEqual(len(issued), 1)
        out = comm.wait_async(h)
        self.assertTrue(torch.equal(out, x * 2))


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

        # `broadcast` joined the three data-plane ops: the communicator's
        # generic broadcast is host-staged and therefore ILLEGAL inside a CUDA
        # graph capture, which the speculative draft-pick sync performs once
        # PyNccl is suppressed under HTCCL. The device transport declares and
        # implements a capturable one (all_gather + src slice).
        self.assertEqual(
            HTCCLDeviceTransport.HTCCL_OPS,
            frozenset(OPS) | {"broadcast"},
        )
        for name in ("handles", "htccl_all_reduce", "htccl_all_gather",
                     "htccl_reduce_scatter", "htccl_broadcast"):
            self.assertTrue(hasattr(HTCCLDeviceTransport, name), name)

    def test_device_broadcast_is_built_on_a_byte_exact_primitive(self):
        """It must be all_gather-based, not zero+all_reduce.

        The draft-pick sync carries an int64 `topk_index`. The device reduce
        kernel dispatches only kHalf/kBFloat16/kFloat, so a zero+all_reduce
        emulation would either fail outright or, if bit-cast into a float
        dtype, silently corrupt indices (-0.0 and NaN payloads are not
        preserved by `x + 0.0`). all_gather's kernel is pure byte movement,
        so it is exact for every dtype.
        """
        import ast
        import inspect
        import textwrap

        from sglang.srt.distributed.device_communicators.htccl_device import (
            HTCCLDeviceTransport,
        )

        fn = ast.parse(
            textwrap.dedent(
                inspect.getsource(HTCCLDeviceTransport.htccl_broadcast)
            )
        ).body[0]
        # judge the CODE, not the prose: the docstring names the rejected
        # alternative on purpose
        if (fn.body and isinstance(fn.body[0], ast.Expr)
                and isinstance(fn.body[0].value, ast.Constant)):
            fn.body = fn.body[1:]
        body = ast.unparse(fn)
        self.assertIn("all_gather", body)
        self.assertNotIn("all_reduce", body)


if __name__ == "__main__":
    unittest.main()
