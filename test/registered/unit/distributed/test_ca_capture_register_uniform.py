"""#195: the graph-buffer registration in ca_comm.capture()'s exit is a GROUP
collective entered on a RANK-LOCAL condition.

Family (fourth-plus sighting, siblings #194/#94): a rank-local condition
decides whether a group collective is entered. `CustomAllreduce.capture()`
runs `register_graph_buffers()` at context exit -- a per-rank
`broadcast_object_list` over the whole group (v2: an `all_gather`) -- while
entry into the context can be rank-local:

* a solo-placed draft runner (spec_solo_rank_local_graphs) captures its
  graphs ALONE inside parallel_state.graph_capture(), which opens
  ca_comm.capture() on the solo rank only (recorded in the #194 fix commit as
  a separate bug -- this one);
* dual-group lane runners capture on a lane subset;
* `disabled` itself is rank-local (sgl_kernel import, cached can_p2p verdict,
  constructor exception), yet gated the registration.

The fix is rank-uniform derivation, never a conditional collective:
1. register_graph_buffers()/v2-capture skip -- without any collective -- when
   the capture recorded no custom-AR call. A captured custom-AR call is
   itself a group collective, so its count is group-uniform in a uniform
   capture and zero in a rank-local one.
2. GroupCoordinator._harmonize_ca_comm_enablement forces `disabled` to the
   group consensus at construction time, the one point where every rank is
   provably present.

Hermetic: CPU only, no torch.cuda calls (attributes are patched, never
invoked), no process groups -- the group protocol is simulated with an
in-process rendezvous that DETECTS unbalanced entry instead of hanging.
"""

import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

# THE REAL MODULES, imported -- not re-implementations.
import sglang.srt.distributed.device_communicators.custom_all_reduce as car_mod  # noqa: E402
import sglang.srt.distributed.device_communicators.custom_all_reduce_ops as car_ops  # noqa: E402
from sglang.srt.distributed.device_communicators.custom_all_reduce import (  # noqa: E402
    CustomAllreduce,
)
from sglang.srt.distributed.parallel_state import GroupCoordinator  # noqa: E402

RENDEZVOUS_TIMEOUT_S = 5.0


class HangDetected(Exception):
    """A simulated rank waited on a collective its peers never entered."""


class FakeGroup:
    """Stands in for a ProcessGroup; also carries the simulated caller rank,
    so the fake dist functions can tell WHO entered a collective."""

    def __init__(self, rank, ranks):
        self.rank = rank
        self.ranks = list(ranks)


class FakeDistRendezvous:
    """broadcast_object_list with real rendezvous semantics and a watchdog.

    Publisher (caller rank == src) deposits its payload and signals; every
    consumer waits for the signal with a timeout. Balanced entry -> all
    threads complete with identical data. Unbalanced entry -> HangDetected
    instead of an indefinite hang, which is the property under test.
    """

    def __init__(self):
        self._store = {}
        self._events = {}
        self._lock = threading.Lock()
        self.entered_ranks = []

    def _event_for(self, key):
        with self._lock:
            if key not in self._events:
                self._events[key] = threading.Event()
            return self._events[key]

    def broadcast_object_list(self, obj_list, src=None, group=None, device=None):
        with self._lock:
            self.entered_ranks.append(group.rank)
        event = self._event_for(src)
        if group.rank == src:
            with self._lock:
                self._store[src] = [x for x in obj_list]
            event.set()
        else:
            if not event.wait(timeout=RENDEZVOUS_TIMEOUT_S):
                raise HangDetected(
                    f"rank {group.rank} waited on broadcast(src={src}) "
                    "that no peer entered"
                )
            with self._lock:
                obj_list[:] = [x for x in self._store[src]]

    @staticmethod
    def get_world_size(group=None):
        return len(group.ranks)

    @staticmethod
    def get_process_group_ranks(group=None):
        return list(group.ranks)


def make_ca(rank, ranks, meta_by_rank):
    """A CustomAllreduce with __init__ bypassed: only the state the capture
    exit path reads."""
    ca = object.__new__(CustomAllreduce)
    ca.disabled = False
    ca._IS_CAPTURING = False
    ca.rank = rank
    ca.world_size = len(ranks)
    ca.group = FakeGroup(rank, ranks)
    ca._ptr = 1000 + rank
    ca._meta = meta_by_rank[rank]
    # __del__ -> close() would hand the fake _ptr to the real ops.dispose()
    # (a segfault); the instance attribute shadows the method.
    ca.close = lambda: None
    return ca


class TestV1CaptureExitBalance(CustomTestCase):
    """The v1 (custom_all_reduce.py) capture exit."""

    def _run_ranks(self, cas, fake):
        """Run each rank's capture() exit concurrently, as the real boot does."""
        errors = {}
        registered = {}

        def fake_meta(ptr):
            ca = next(c for c in cas if c._ptr == ptr)
            return ca._meta

        def fake_register(ptr, handles, offsets):
            rank = next(c.rank for c in cas if c._ptr == ptr)
            registered[rank] = (handles, offsets)

        def one_rank(ca):
            try:
                with ca.capture():
                    pass
            except Exception as e:  # noqa: BLE001 -- recorded for assertions
                errors[ca.rank] = e

        with (
            mock.patch.object(
                car_ops, "get_graph_buffer_ipc_meta", create=True, new=fake_meta
            ),
            mock.patch.object(
                car_ops, "register_graph_buffers", create=True, new=fake_register
            ),
            mock.patch.object(
                car_mod.dist, "broadcast_object_list", fake.broadcast_object_list
            ),
            mock.patch.object(car_mod.dist, "get_world_size", fake.get_world_size),
            mock.patch.object(
                car_mod.dist,
                "get_process_group_ranks",
                fake.get_process_group_ranks,
            ),
        ):
            threads = [threading.Thread(target=one_rank, args=(ca,)) for ca in cas]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=RENDEZVOUS_TIMEOUT_S + 5)
                self.assertFalse(t.is_alive(), "simulated rank did not finish")
        return errors, registered

    def test_rank_local_capture_enters_no_collective(self):
        """THE #195 CASE: one rank captures alone (solo draft), records no
        custom-AR call, and must not enter the group broadcast."""
        meta = {0: ([], [])}
        ca = make_ca(0, [0, 1], meta)
        fake = FakeDistRendezvous()
        errors, registered = self._run_ranks([ca], fake)
        self.assertEqual(errors, {}, f"solo capture exit raised: {errors}")
        self.assertEqual(
            fake.entered_ranks,
            [],
            "solo rank entered a group collective nobody else will join",
        )
        self.assertEqual(registered, {}, "nothing was captured, nothing to register")

    def test_uniform_capture_with_buffers_registers_identically(self):
        """Uniform case: both ranks captured the same AR calls; the exchange
        runs, is balanced, and every rank registers every rank's meta."""
        meta = {
            0: (b"handle-0", [0, 16]),
            1: (b"handle-1", [0, 16]),
        }
        cas = [make_ca(r, [0, 1], meta) for r in (0, 1)]
        fake = FakeDistRendezvous()
        errors, registered = self._run_ranks(cas, fake)
        self.assertEqual(errors, {})
        # both ranks entered the exchange, twice each (one broadcast per src)
        self.assertEqual(sorted(set(fake.entered_ranks)), [0, 1])
        self.assertEqual(sorted(registered.keys()), [0, 1])
        for rank in (0, 1):
            handles, offsets = registered[rank]
            self.assertEqual(handles, [b"handle-0", b"handle-1"])
            self.assertEqual(offsets, [[0, 16], [0, 16]])
        # identical exchange result on every rank -- the byte-unchanged check
        self.assertEqual(registered[0], registered[1])

    def test_uniform_empty_capture_skips_everywhere(self):
        """A uniform capture that recorded no AR call skips the exchange on
        every rank -- balanced by construction (registering zero buffers was
        a no-op before the fix)."""
        meta = {0: ([], []), 1: ([], [])}
        cas = [make_ca(r, [0, 1], meta) for r in (0, 1)]
        fake = FakeDistRendezvous()
        errors, registered = self._run_ranks(cas, fake)
        self.assertEqual(errors, {})
        self.assertEqual(fake.entered_ranks, [])
        self.assertEqual(registered, {})

    def test_disabled_rank_skips_before_touching_meta(self):
        """A (group-uniformly) disabled communicator's capture exit reads no
        ops state at all."""
        ca = make_ca(0, [0, 1], {0: (b"x", [0])})
        ca.disabled = True
        fake = FakeDistRendezvous()

        def must_not_be_called(*a, **k):
            raise AssertionError("disabled rank touched the register path")

        with (
            mock.patch.object(
                car_ops,
                "get_graph_buffer_ipc_meta",
                create=True,
                new=must_not_be_called,
            ),
            mock.patch.object(
                car_mod.dist, "broadcast_object_list", fake.broadcast_object_list
            ),
        ):
            with ca.capture():
                pass
        self.assertEqual(fake.entered_ranks, [])


class TestInitEnablementHarmonisation(CustomTestCase):
    """GroupCoordinator._harmonize_ca_comm_enablement: `disabled` becomes the
    group consensus at construction time."""

    def _run(self, local_ca, gathered_verdicts):
        me = SimpleNamespace(
            ca_comm=local_ca,
            world_size=len(gathered_verdicts),
            ranks=list(range(len(gathered_verdicts))),
            cpu_group=object(),
            unique_name="tp:test",
        )

        def fake_all_gather_object(out, obj, group=None):
            out[:] = list(gathered_verdicts)

        with mock.patch(
            "torch.distributed.all_gather_object", new=fake_all_gather_object
        ):
            GroupCoordinator._harmonize_ca_comm_enablement(me)
        return me

    def test_divergent_group_disables_everywhere(self):
        ca = SimpleNamespace(disabled=False, original_disabled=False)
        self._run(ca, [True, False])
        self.assertTrue(ca.disabled, "enabled rank must join the group consensus")
        self.assertTrue(ca.original_disabled)

    def test_uniform_enabled_group_is_untouched(self):
        ca = SimpleNamespace(disabled=False, original_disabled=False)
        self._run(ca, [True, True])
        self.assertFalse(ca.disabled)
        self.assertFalse(ca.original_disabled)

    def test_uniform_disabled_group_is_untouched(self):
        ca = SimpleNamespace(disabled=True, original_disabled=True)
        self._run(ca, [False, False])
        self.assertTrue(ca.disabled)

    def test_failed_construction_rank_does_not_crash(self):
        """The rank whose constructor raised has ca_comm=None; it still
        participates in the agreement and returns cleanly."""
        me = self._run(None, [True, False])
        self.assertIsNone(me.ca_comm)

    def test_v2_shaped_comm_without_original_disabled(self):
        ca = SimpleNamespace(disabled=False)  # CustomAllReduceV2 shape
        self._run(ca, [True, False])
        self.assertTrue(ca.disabled)
        self.assertFalse(hasattr(ca, "original_disabled"))


class TestV2CaptureExitBalance(CustomTestCase):
    """custom_all_reduce_v2.py: same path, same family, same gates."""

    @classmethod
    def setUpClass(cls):
        try:
            import sglang.srt.distributed.device_communicators.custom_all_reduce_v2 as v2_mod
        except ImportError as e:  # jit_kernel unavailable in this build
            raise unittest.SkipTest(f"custom_all_reduce_v2 not importable: {e}")
        cls.v2_mod = v2_mod

    def _make_v2(self, raw_ptrs, disabled=False):
        v2 = object.__new__(self.v2_mod.CustomAllReduceV2)
        v2.disabled = disabled
        v2.group = object()  # for __del__ -> close() on the mock instance
        v2.tms_cudagraph = False
        v2.obj = mock.Mock()
        v2.obj.get_graph_capture_ptrs.return_value = raw_ptrs
        v2._vmm_graph_input_manager = mock.Mock()
        v2._register_graph_inputs_ipc = mock.Mock()
        return v2

    def _exit_capture(self, v2):
        # patched attribute, never the real torch.cuda call
        with mock.patch.object(
            self.v2_mod.torch.cuda,
            "is_current_stream_capturing",
            create=True,
            return_value=False,
        ):
            with v2.capture():
                pass

    def test_empty_capture_registers_nothing(self):
        """THE #195 CASE for v2: an empty (rank-local) capture must not walk
        into the group-wide all_gather of the IPC registration."""
        v2 = self._make_v2(raw_ptrs=[])
        self._exit_capture(v2)
        v2._register_graph_inputs_ipc.assert_not_called()
        v2._vmm_graph_input_manager.register_graph_inputs.assert_not_called()

    def test_nonempty_capture_registers_ipc(self):
        v2 = self._make_v2(raw_ptrs=[0x1000])
        with mock.patch.object(self.v2_mod, "is_vmm_pointer", return_value=False):
            self._exit_capture(v2)
        v2._register_graph_inputs_ipc.assert_called_once()
        v2._vmm_graph_input_manager.register_graph_inputs.assert_not_called()

    def test_nonempty_vmm_capture_registers_vmm(self):
        v2 = self._make_v2(raw_ptrs=[0x2000])
        with mock.patch.object(self.v2_mod, "is_vmm_pointer", return_value=True):
            self._exit_capture(v2)
        v2._vmm_graph_input_manager.register_graph_inputs.assert_called_once()
        v2._register_graph_inputs_ipc.assert_not_called()

    def test_disabled_capture_touches_nothing(self):
        v2 = self._make_v2(raw_ptrs=[0x3000], disabled=True)
        with v2.capture():
            pass
        v2.obj.set_cuda_graph_capture.assert_not_called()
        v2.obj.get_graph_capture_ptrs.assert_not_called()
        v2._register_graph_inputs_ipc.assert_not_called()


if __name__ == "__main__":
    unittest.main()
