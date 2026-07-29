"""#286 offload register, GPU-phase prebuild -- hermetic movement tests.

No GPU, no CUDA: torch.cuda is NEVER called; every device touch goes through
``FakeDeviceOps``. Covers the movement state machine (park-in-flight /
parked / wave-in-flight, double-park, wave-in during an in-flight park,
error paths), the Erg.-7c park-target ladder including the DIRECTED P2P
capability model (measured-vs-unmeasured paths, effective-aperture window
policies reject|chunk, asymmetric pairs, clean degradation to host_ram),
the capacity ledger (peer budget = explicit grant, never silent poaching),
the policy-syntax extension (``@target>target``) and the live size sources.
"""

import logging
import os
import unittest
from unittest.mock import patch

from sglang.srt.model_executor.offload_movement import (
    DEFAULT_PARK_TARGET_ORDER,
    STATE_PARK_IN_FLIGHT,
    STATE_PARKED,
    STATE_RESIDENT,
    CapacityLedger,
    FakeDeviceOps,
    MovementError,
    PeerPathCapability,
    PeerProbe,
    RealMovementBackend,
    SuspendPayload,
    TagPayload,
    TensorPayload,
)
from sglang.srt.model_executor.offload_register import (
    CpuFakeMovementBackend,
    OffloadItem,
    OffloadRegister,
    configure_global_register,
    get_global_register,
    maybe_bind_movement_payload,
    maybe_refresh_item_sizes,
    maybe_register_item,
    parse_class_policy_overrides,
    parse_park_target_order,
    reset_global_register,
    resolve_class_policies,
)
from sglang.srt.model_executor.offload_sizes import resolve_size_bytes
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def make_item(item_id="item", size=1000, va_stable=False, klass="lane_workspaces"):
    return OffloadItem(
        item_id=item_id,
        offload_class=klass,
        size_bytes=size,
        restore_cost_ms=lambda: 1.0,
        hot=lambda: False,
        va_stable_required=va_stable,
    )


def make_backend(ops=None, **kwargs):
    return RealMovementBackend(ops if ops is not None else FakeDeviceOps(), **kwargs)


class _FakeTensor:
    """numel()/element_size() duck-type -- what the size resolver and the
    payloads see; deliberately not a torch tensor."""

    def __init__(self, numel=10, element_size=4, device_index=0):
        self._numel = numel
        self._element_size = element_size

        class _Dev:
            index = device_index

        self.device = _Dev()

    def numel(self):
        return self._numel

    def element_size(self):
        return self._element_size


class TestParkTargetParser(unittest.TestCase):
    def test_default_and_chain(self):
        self.assertEqual(parse_park_target_order(None, "--x"), ("host_ram",))
        self.assertEqual(parse_park_target_order("", "--x"), ("host_ram",))
        self.assertEqual(
            parse_park_target_order("peer_vram>host_ram", "--x"),
            ("peer_vram", "host_ram"),
        )
        self.assertEqual(DEFAULT_PARK_TARGET_ORDER, ("host_ram",))

    def test_rejects_own_vram_unknown_and_duplicates(self):
        with self.assertRaisesRegex(ValueError, "tier 0"):
            parse_park_target_order("own_vram>host_ram", "--x")
        with self.assertRaisesRegex(ValueError, "unknown park target"):
            parse_park_target_order("moon", "--x")
        with self.assertRaisesRegex(ValueError, "twice"):
            parse_park_target_order("host_ram>host_ram", "--x")

    def test_class_policy_target_suffix(self):
        p = parse_class_policy_overrides(
            "drafter_heads=ram:0.5@peer_vram>host_ram,graph_rungs=auto"
        )
        self.assertEqual(p["drafter_heads"].targets, ("peer_vram", "host_ram"))
        self.assertEqual(p["drafter_heads"].fraction, 0.5)
        self.assertIsNone(p["graph_rungs"].targets)

    def test_class_policy_target_suffix_errors(self):
        with self.assertRaisesRegex(ValueError, "unknown park target"):
            parse_class_policy_overrides("drafter_heads=ram@moon")
        with self.assertRaisesRegex(ValueError, "empty park-target"):
            parse_class_policy_overrides("drafter_heads=ram@")
        with self.assertRaisesRegex(ValueError, "tier 0"):
            parse_class_policy_overrides("drafter_heads=ram@own_vram")


class TestPeerProbe(unittest.TestCase):
    """The directed P2P capability model. Every parameter of a path is a
    placeholder until the post-driver-update probe run feeds it; an
    unmeasured path is never assumed usable."""

    def test_unmeasured_path_is_refused_even_with_p2p(self):
        probe = PeerProbe(FakeDeviceOps(peer_matrix={(0, 1): True}))
        ok, chunk, why = probe.path_admits(0, 1, 100)
        self.assertFalse(ok)
        self.assertIn("unmeasured", why)

    def test_no_p2p_path(self):
        probe = PeerProbe(FakeDeviceOps())
        ok, _chunk, why = probe.path_admits(0, 1, 100)
        self.assertFalse(ok)
        self.assertEqual(why, "no P2P path")

    def test_measured_full_bar_admits(self):
        probe = PeerProbe(
            FakeDeviceOps(),
            capabilities={(0, 1): PeerPathCapability(p2p=True, measured=True)},
        )
        ok, chunk, _ = probe.path_admits(0, 1, 10**9)
        self.assertTrue(ok)
        self.assertIsNone(chunk)

    def test_effective_aperture_reject_and_chunk_policies(self):
        # aperture_bytes is the EFFECTIVE measured window, injected -- no
        # nominal constant anywhere in the code.
        cap = PeerPathCapability(
            p2p=True,
            measured=True,
            aperture_bytes=256,
            nominal_bar_bytes=512,
        )
        probe = PeerProbe(FakeDeviceOps(), capabilities={(0, 1): cap})
        ok, chunk, _ = probe.path_admits(0, 1, 200, "reject")
        self.assertTrue(ok)
        self.assertIsNone(chunk)  # fits the window in one piece
        ok, _chunk, why = probe.path_admits(0, 1, 1000, "reject")
        self.assertFalse(ok)
        self.assertIn("aperture", why)
        ok, chunk, _ = probe.path_admits(0, 1, 1000, "chunk")
        self.assertTrue(ok)
        self.assertEqual(chunk, 256)

    def test_asymmetric_pairs_are_independent(self):
        # 3080 -> 5090 (full BAR) measured usable; 5090 -> 3080 unmeasured.
        probe = PeerProbe(
            FakeDeviceOps(peer_matrix={(1, 0): True}),
            capabilities={(0, 1): PeerPathCapability(p2p=True, measured=True)},
        )
        self.assertTrue(probe.path_admits(0, 1, 100)[0])
        self.assertFalse(probe.path_admits(1, 0, 100)[0])

    def test_set_capability_feeds_measurement_later(self):
        probe = PeerProbe(FakeDeviceOps(peer_matrix={(0, 1): True}))
        self.assertFalse(probe.path_admits(0, 1, 100)[0])
        probe.set_capability(0, 1, PeerPathCapability(p2p=True, measured=True))
        self.assertTrue(probe.path_admits(0, 1, 100)[0])


class TestCapacityLedger(unittest.TestCase):
    def test_host_ram_default_unlimited(self):
        ledger = CapacityLedger()
        self.assertIsNone(ledger.headroom("host_ram", None))
        self.assertTrue(ledger.book("host_ram", None, 10**12))

    def test_peer_vram_requires_explicit_grant(self):
        # No silent poaching of the target card's KV / group budget: without
        # a granted budget the booking is refused.
        ledger = CapacityLedger()
        self.assertFalse(ledger.book("peer_vram", 1, 1))
        ledger.set_budget("peer_vram", 1, 1000)
        self.assertTrue(ledger.book("peer_vram", 1, 800))
        self.assertFalse(ledger.book("peer_vram", 1, 300))  # over budget
        ledger.release("peer_vram", 1, 800)
        self.assertTrue(ledger.book("peer_vram", 1, 1000))

    def test_remote_is_never_bookable(self):
        self.assertFalse(CapacityLedger().book("remote", None, 1))

    def test_host_budget_when_set_is_enforced(self):
        ledger = CapacityLedger()
        ledger.set_budget("host_ram", None, 100)
        self.assertTrue(ledger.book("host_ram", None, 100))
        self.assertFalse(ledger.book("host_ram", None, 1))


class TestBackendStateMachine(unittest.TestCase):
    def _bound_backend(self, ops=None, size=1000, **kwargs):
        backend = make_backend(ops, **kwargs)
        item = make_item(size=size)
        backend.bind(item.item_id, TensorPayload((_FakeTensor(),)), 0)
        return backend, item

    def test_park_without_binding_is_an_error(self):
        backend = make_backend()
        with self.assertRaisesRegex(MovementError, "no bound movement payload"):
            backend.park(make_item())

    def test_tensor_roundtrip_host_ram(self):
        ops = FakeDeviceOps()
        backend, item = self._bound_backend(ops)
        backend.park(item)
        # Tensor route: transfer enqueued behind compute -> in flight until
        # settled at a boundary.
        self.assertEqual(backend.state_of(item.item_id), STATE_PARK_IN_FLIGHT)
        self.assertEqual(backend.target_of(item.item_id), "host_ram")
        self.assertEqual(backend.ledger.booked("host_ram"), 1000)
        backend.settle(item.item_id)
        self.assertEqual(backend.state_of(item.item_id), STATE_PARKED)
        backend.wave_in(item)
        self.assertEqual(backend.state_of(item.item_id), STATE_RESIDENT)
        self.assertEqual(backend.ledger.booked("host_ram"), 0)
        ops_seq = [c[0] for c in ops.calls]
        self.assertEqual(
            ops_seq, ["copy_out", "wait", "wait", "copy_in", "wait", "free_destination"]
        )
        self.assertEqual(backend.stats.parks, 1)
        self.assertEqual(backend.stats.wave_ins, 1)

    def test_double_park_is_a_noop(self):
        backend, item = self._bound_backend()
        backend.park(item)
        backend.park(item)  # in flight
        backend.settle(item.item_id)
        backend.park(item)  # parked
        self.assertEqual(backend.stats.parks, 1)

    def test_wave_in_during_in_flight_park_joins_first(self):
        ops = FakeDeviceOps()
        backend, item = self._bound_backend(ops)
        backend.park(item)
        self.assertEqual(backend.state_of(item.item_id), STATE_PARK_IN_FLIGHT)
        backend.wave_in(item)  # joins the park, then reverses it
        self.assertEqual(backend.state_of(item.item_id), STATE_RESIDENT)
        names = [c[0] for c in ops.calls]
        # The park's copy_out is WAITED before the reverse copy starts.
        self.assertLess(names.index("wait"), names.index("copy_in"))
        self.assertEqual(backend.ledger.booked("host_ram"), 0)

    def test_settle_is_idempotent_and_resident_wave_in_is_noop(self):
        backend, item = self._bound_backend()
        backend.settle(item.item_id)  # nothing in flight
        backend.wave_in(item)  # resident -> no-op
        self.assertEqual(backend.stats.wave_ins, 0)

    def test_failed_park_releases_booking_and_stays_resident(self):
        ops = FakeDeviceOps(fail_ops={"copy_out": RuntimeError("bus fell off")})
        backend, item = self._bound_backend(ops)
        with self.assertRaisesRegex(MovementError, "park of item"):
            backend.park(item)
        self.assertEqual(backend.state_of(item.item_id), STATE_RESIDENT)
        self.assertEqual(backend.ledger.booked("host_ram"), 0)
        self.assertEqual(backend.stats.park_failures, 1)

    def test_failed_wave_in_stays_parked_and_keeps_booking(self):
        ops = FakeDeviceOps(fail_ops={"copy_in": RuntimeError("boom")})
        backend, item = self._bound_backend(ops)
        backend.park(item)
        backend.settle(item.item_id)
        with self.assertRaisesRegex(MovementError, "wave-in of item"):
            backend.wave_in(item)
        self.assertEqual(backend.state_of(item.item_id), STATE_PARKED)
        self.assertEqual(backend.ledger.booked("host_ram"), 1000)
        self.assertEqual(backend.stats.wave_in_failures, 1)


class TestTargetLadder(unittest.TestCase):
    def _peer_ready_backend(self, size=1000, window_policy="reject", cap=None):
        ops = FakeDeviceOps()
        probe = PeerProbe(
            ops,
            capabilities={(0, 1): cap or PeerPathCapability(p2p=True, measured=True)},
        )
        backend = RealMovementBackend(
            ops,
            target_order=("peer_vram", "host_ram"),
            probe=probe,
            window_policy=window_policy,
        )
        backend.ledger.set_budget("peer_vram", 1, 10_000)
        item = make_item(size=size)
        backend.bind(item.item_id, TensorPayload((_FakeTensor(),)), 0)
        return backend, item, ops

    def test_peer_vram_park_books_the_peer_budget(self):
        backend, item, ops = self._peer_ready_backend()
        backend.park(item)
        self.assertEqual(backend.target_of(item.item_id), "peer_vram")
        self.assertEqual(backend.ledger.booked("peer_vram", 1), 1000)
        copy_out = next(c for c in ops.calls if c[0] == "copy_out")
        self.assertEqual(copy_out[1:], ("peer_vram", 1, None))
        backend.wave_in(item)
        self.assertEqual(backend.ledger.booked("peer_vram", 1), 0)

    def test_unmeasured_p2p_degrades_to_host_ram_with_log(self):
        ops = FakeDeviceOps(peer_matrix={(0, 1): True})  # p2p, unmeasured
        backend = RealMovementBackend(ops, target_order=("peer_vram", "host_ram"))
        backend.ledger.set_budget("peer_vram", 1, 10_000)
        item = make_item()
        backend.bind(item.item_id, TensorPayload((_FakeTensor(),)), 0)
        with self.assertLogs(
            "sglang.srt.model_executor.offload_movement", logging.INFO
        ) as logs:
            backend.park(item)  # degrades, never errors
        self.assertEqual(backend.target_of(item.item_id), "host_ram")
        self.assertEqual(backend.stats.peer_degradations, 1)
        self.assertTrue(any("degrading to host_ram" in m for m in logs.output))

    def test_no_peer_budget_degrades_to_host_ram(self):
        ops = FakeDeviceOps()
        probe = PeerProbe(
            ops,
            capabilities={(0, 1): PeerPathCapability(p2p=True, measured=True)},
        )
        backend = RealMovementBackend(
            ops, target_order=("peer_vram", "host_ram"), probe=probe
        )
        # No set_budget for the peer: booking must not silently poach.
        item = make_item()
        backend.bind(item.item_id, TensorPayload((_FakeTensor(),)), 0)
        backend.park(item)
        self.assertEqual(backend.target_of(item.item_id), "host_ram")

    def test_asymmetric_pair_direction_matters(self):
        ops = FakeDeviceOps()
        probe = PeerProbe(
            ops,
            capabilities={
                (0, 1): PeerPathCapability(p2p=True, measured=True),
                (1, 0): PeerPathCapability(p2p=False),
            },
        )
        backend = RealMovementBackend(
            ops, target_order=("peer_vram", "host_ram"), probe=probe
        )
        backend.ledger.set_budget("peer_vram", 0, 10_000)
        backend.ledger.set_budget("peer_vram", 1, 10_000)
        fwd = make_item("fwd")
        rev = make_item("rev")
        backend.bind("fwd", TensorPayload((_FakeTensor(),)), 0)
        backend.bind("rev", TensorPayload((_FakeTensor(),)), 1)
        backend.park(fwd)
        backend.park(rev)
        self.assertEqual(backend.target_of("fwd"), "peer_vram")
        self.assertEqual(backend.target_of("rev"), "host_ram")

    def test_window_policy_reject_degrades_oversized_items(self):
        cap = PeerPathCapability(p2p=True, measured=True, aperture_bytes=100)
        backend, item, _ops = self._peer_ready_backend(
            size=1000, window_policy="reject", cap=cap
        )
        backend.park(item)
        self.assertEqual(backend.target_of(item.item_id), "host_ram")

    def test_window_policy_chunk_moves_in_aperture_chunks(self):
        cap = PeerPathCapability(p2p=True, measured=True, aperture_bytes=100)
        backend, item, ops = self._peer_ready_backend(
            size=1000, window_policy="chunk", cap=cap
        )
        backend.park(item)
        self.assertEqual(backend.target_of(item.item_id), "peer_vram")
        copy_out = next(c for c in ops.calls if c[0] == "copy_out")
        self.assertEqual(copy_out[3], 100)  # chunk_bytes = effective aperture
        self.assertEqual(backend.stats.chunked_transfers, 1)

    def test_remote_is_a_stub_and_ladder_falls_through(self):
        backend = make_backend(target_order=("remote", "host_ram"))
        item = make_item()
        backend.bind(item.item_id, TensorPayload((_FakeTensor(),)), 0)
        backend.park(item)
        self.assertEqual(backend.target_of(item.item_id), "host_ram")

    def test_explicit_remote_target_names_224(self):
        backend = make_backend()
        item = make_item()
        backend.bind(item.item_id, TensorPayload((_FakeTensor(),)), 0)
        with self.assertRaisesRegex(MovementError, "#224"):
            backend.park(item, target="remote")

    def test_explicit_own_vram_and_unknown_target_are_value_errors(self):
        backend = make_backend()
        item = make_item()
        backend.bind(item.item_id, TensorPayload((_FakeTensor(),)), 0)
        with self.assertRaisesRegex(ValueError, "tier 0"):
            backend.park(item, target="own_vram")
        with self.assertRaises(ValueError):
            backend.park(item, target="moon")

    def test_per_class_target_override(self):
        policies = resolve_class_policies("capacity", "lane_workspaces=ram@peer_vram")
        ops = FakeDeviceOps()
        probe = PeerProbe(
            ops,
            capabilities={(0, 1): PeerPathCapability(p2p=True, measured=True)},
        )
        backend = RealMovementBackend(
            ops,
            target_order=("host_ram",),
            class_policies=policies,
            probe=probe,
        )
        backend.ledger.set_budget("peer_vram", 1, 10_000)
        item = make_item(klass="lane_workspaces")
        backend.bind(item.item_id, TensorPayload((_FakeTensor(),)), 0)
        backend.park(item)
        # The class override beats the global host_ram-only order.
        self.assertEqual(backend.target_of(item.item_id), "peer_vram")

    def test_va_stable_tensor_payload_is_refused_with_guidance(self):
        backend = make_backend()
        item = make_item(va_stable=True)
        backend.bind(item.item_id, TensorPayload((_FakeTensor(),)), 0)
        with self.assertRaisesRegex(MovementError, "TagPayload"):
            backend.park(item)


class TestRoutes(unittest.TestCase):
    def test_tag_route_pause_resume(self):
        ops = FakeDeviceOps()
        backend = make_backend(ops)
        item = make_item(va_stable=True)
        backend.bind(item.item_id, TagPayload("graph_rung_k3"), 0)
        backend.park(item)
        # Tag route settles synchronously (#93 pause is not an async copy).
        self.assertEqual(backend.state_of(item.item_id), STATE_PARKED)
        backend.wave_in(item)
        self.assertEqual(
            [c[0] for c in ops.calls],
            ["pause_tag", "resume_tag"],
        )

    def test_tag_route_parks_to_host_ram_only(self):
        backend = make_backend()
        item = make_item(va_stable=True)
        backend.bind(item.item_id, TagPayload("t"), 0)
        with self.assertRaisesRegex(MovementError, "host_ram only"):
            backend.park(item, target="peer_vram")

    def test_suspend_route_pauses_and_resumes_tags_in_order(self):
        ops = FakeDeviceOps()
        backend = make_backend(ops)
        item = make_item(va_stable=True, klass="cold_lane")
        backend.bind(item.item_id, SuspendPayload(("lane1/own",)), 0)
        backend.park(item)
        backend.wave_in(item)
        self.assertEqual(
            ops.calls,
            [
                ("suspend_tags", ("lane1/own",)),
                ("resume_suspended_tags", ("lane1/own",)),
            ],
        )

    def test_unknown_payload_type_is_rejected_at_bind(self):
        backend = make_backend()
        with self.assertRaisesRegex(ValueError, "unknown payload type"):
            backend.bind("x", object())


class TestRegisterIntegration(unittest.TestCase):
    """The register drives the real backend: policy half decides, movement
    half executes; park() passes the target through."""

    def _register_with_backend(self):
        ops = FakeDeviceOps()
        backend = make_backend(ops)
        reg = OffloadRegister(
            policies=resolve_class_policies("capacity"),
            backend=backend,
            hysteresis_window_s=0.0,
        )
        return reg, backend, ops

    def test_register_park_target_passthrough(self):
        reg, backend, _ops = self._register_with_backend()
        reg.register("w", "lane_workspaces", 100, 1.0)
        backend.bind("w", TensorPayload((_FakeTensor(),)), 0)
        reg.park("w", target="host_ram")
        self.assertTrue(reg.is_parked("w"))
        self.assertEqual(backend.target_of("w"), "host_ram")

    def test_register_rejects_unknown_target_early(self):
        reg, _backend, _ops = self._register_with_backend()
        reg.register("w", "lane_workspaces", 100, 1.0)
        with self.assertRaises(ValueError):
            reg.park("w", target="moon")

    def test_backend_failure_leaves_register_resident(self):
        ops = FakeDeviceOps(fail_ops={"copy_out": RuntimeError("nope")})
        backend = make_backend(ops)
        reg = OffloadRegister(
            policies=resolve_class_policies("capacity"),
            backend=backend,
            hysteresis_window_s=0.0,
        )
        reg.register("w", "lane_workspaces", 100, 1.0)
        backend.bind("w", TensorPayload((_FakeTensor(),)), 0)
        with self.assertRaises(MovementError):
            reg.park("w")
        self.assertFalse(reg.is_parked("w"))

    def test_cpu_fake_backend_records_targets_and_bindings(self):
        fake = CpuFakeMovementBackend()
        reg = OffloadRegister(
            policies=resolve_class_policies("capacity"),
            backend=fake,
            hysteresis_window_s=0.0,
        )
        reg.register("w", "lane_workspaces", 100, 1.0)
        fake.bind("w", TagPayload("t"))
        reg.park("w", target="host_ram")
        self.assertEqual(fake.parked_targets, ["host_ram"])
        self.assertIn("w", fake.bound)


class TestSizeResolution(unittest.TestCase):
    """Task 3: real item sizes from the registered objects, resolver
    injectable and meta/cpu-safe -- on GPU the true figure appears without
    further change."""

    def test_tensor_like_and_nested_sources(self):
        t = _FakeTensor(numel=10, element_size=4)
        t2 = _FakeTensor(numel=10, element_size=4)
        self.assertEqual(resolve_size_bytes(t), 40)
        self.assertEqual(resolve_size_bytes([t, t2]), 80)
        # Identity dedup: an ALIASED tensor is one allocation, one count.
        self.assertEqual(resolve_size_bytes([t, t]), 40)
        self.assertEqual(resolve_size_bytes({"a": t}), 40)
        self.assertEqual(resolve_size_bytes(lambda: t), 40)
        self.assertEqual(resolve_size_bytes(None), 0)
        self.assertEqual(resolve_size_bytes(123), 123)
        self.assertEqual(resolve_size_bytes(object()), 0)

    def test_footprint_bytes_source(self):
        class _Record:
            footprint_bytes = 4096

        self.assertEqual(resolve_size_bytes(_Record()), 4096)

    def test_module_like_and_runner_like_sources(self):
        tensors = [_FakeTensor(numel=8, element_size=2) for _ in range(3)]

        class _Module:
            def parameters(self):
                return tensors[:2]

            def buffers(self):
                return tensors[2:]

        class _Runner:
            model = _Module()

        self.assertEqual(resolve_size_bytes(_Module()), 48)
        self.assertEqual(resolve_size_bytes(_Runner()), 48)

    def test_meta_tensor_is_size_resolvable_without_cuda(self):
        import torch

        t = torch.empty(16, dtype=torch.float32, device="meta")
        self.assertEqual(resolve_size_bytes(t), 64)

    def test_cycle_safety(self):
        d = {}
        d["self"] = d
        self.assertEqual(resolve_size_bytes(d), 0)

    def test_register_size_source_and_refresh(self):
        reg = OffloadRegister(
            policies=resolve_class_policies("capacity"),
            hysteresis_window_s=0.0,
        )
        holder = {"t": None}
        reg.register(
            "rung",
            "graph_rungs",
            0,
            1.0,
            size_source=lambda: holder["t"],
        )
        self.assertEqual(reg.get("rung").size_bytes, 0)
        # The allocation appears later (capture) -- refresh turns 0 real.
        holder["t"] = _FakeTensor(numel=100, element_size=2)
        self.assertEqual(reg.refresh_sizes(), 1)
        self.assertEqual(reg.get("rung").size_bytes, 200)
        # Idempotent when nothing changed.
        self.assertEqual(reg.refresh_sizes(), 0)

    def test_parked_items_keep_their_park_time_size(self):
        reg = OffloadRegister(
            policies=resolve_class_policies("capacity"),
            hysteresis_window_s=0.0,
        )
        holder = {"t": _FakeTensor(numel=10, element_size=1)}
        reg.register("w", "lane_workspaces", 0, 1.0, size_source=lambda: holder["t"])
        self.assertEqual(reg.get("w").size_bytes, 10)
        reg.park("w")
        holder["t"] = _FakeTensor(numel=999, element_size=1)
        self.assertEqual(reg.refresh_sizes(), 0)
        self.assertEqual(reg.get("w").size_bytes, 10)


class TestAdapterHelpers(unittest.TestCase):
    def setUp(self):
        reset_global_register()
        self.addCleanup(reset_global_register)

    def test_maybe_helpers_are_noops_when_flag_off(self):
        with patch.dict(os.environ, {"SGLANG_OFFLOAD_REGISTER": "0"}):
            self.assertEqual(maybe_refresh_item_sizes(), 0)
            maybe_bind_movement_payload("x", TagPayload("t"))  # must not raise
            self.assertIsNone(get_global_register())

    def test_maybe_bind_reaches_the_backend(self):
        with patch.dict(os.environ, {"SGLANG_OFFLOAD_REGISTER": "1"}):
            reg = configure_global_register("capacity")
            maybe_register_item(
                "w", "lane_workspaces", 0, 1.0, size_source=lambda: _FakeTensor(4, 4)
            )
            maybe_bind_movement_payload("w", TagPayload("t"))
            self.assertIn("w", reg.backend.bound)
            # The live source already resolved at registration...
            self.assertEqual(reg.get("w").size_bytes, 16)
            # ...so a refresh finds nothing changed (idempotent).
            self.assertEqual(maybe_refresh_item_sizes(), 0)


if __name__ == "__main__":
    unittest.main()
