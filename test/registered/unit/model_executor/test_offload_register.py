"""#286 offload register, CPU phase -- hermetic unit tests (no GPU, no CUDA).

Covers the policy half of the generic VRAM item tiering (DESIGN_201
Nachtrag-13 Erg. 7/7b/7c): registration lifecycle, preset expansion and
override precedence (including the 7c depth fraction), the enforced
invariants (hysteresis gate per time-constant tier, hot refusal, priority
protection, fraction cap), the stage-2 phase-boundary planner (plans
deterministically, moves nothing), the #279 latency-term API, ServerArgs
parsing including the failure cases, and the flag-off no-op guarantee of the
adapters.
"""

import os
import unittest
from unittest.mock import patch

from sglang.srt.model_executor.offload_register import (
    OFFLOAD_CLASSES,
    ClassPolicy,
    CpuFakeMovementBackend,
    OffloadRefused,
    OffloadRegister,
    configure_global_register,
    get_global_register,
    maybe_register_item,
    maybe_touch_item,
    parse_class_policy_overrides,
    reset_global_register,
    resolve_class_policies,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class _Clock:
    """Injectable monotonic clock so hysteresis is testable without sleeping."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_register(profile="capacity", overrides=None, **kwargs):
    kwargs.setdefault("backend", CpuFakeMovementBackend())
    kwargs.setdefault("clock", _Clock())
    kwargs.setdefault("hysteresis_window_s", 5.0)
    return OffloadRegister(
        policies=resolve_class_policies(profile, overrides), **kwargs
    )


class TestRegistration(unittest.TestCase):
    def test_register_and_query(self):
        reg = make_register()
        item = reg.register("lane0/drafter_head", "drafter_heads", 1_800_000_000, 50.0)
        self.assertEqual(item.offload_class, "drafter_heads")
        self.assertFalse(reg.is_parked("lane0/drafter_head"))
        self.assertEqual(reg.resident_bytes(), 1_800_000_000)
        self.assertEqual(reg.stats.registered, 1)

    def test_double_registration_is_a_hard_error(self):
        reg = make_register()
        reg.register("x", "graph_rungs", 10, 1.0)
        with self.assertRaisesRegex(ValueError, "already registered"):
            reg.register("x", "graph_rungs", 10, 1.0)

    def test_unknown_class_and_tier_are_hard_errors(self):
        reg = make_register()
        with self.assertRaisesRegex(ValueError, "unknown offload item class"):
            reg.register("x", "kv_cache", 10, 1.0)
        with self.assertRaisesRegex(ValueError, "unknown time-constant tier"):
            reg.register("x", "graph_rungs", 10, 1.0, time_constant_tier="ns")

    def test_negative_size_and_cost_are_hard_errors(self):
        reg = make_register()
        with self.assertRaises(ValueError):
            reg.register("x", "graph_rungs", -1, 1.0)
        with self.assertRaises(ValueError):
            reg.register("x", "graph_rungs", 1, -1.0)

    def test_experts_class_is_registrable_as_existing_class(self):
        # 7b: experts are the existing class, referenced for unified
        # accounting -- registration must work like any other class.
        reg = make_register()
        reg.register("layer3/experts", "experts", 123, 1.0)
        self.assertEqual(len(reg.items_of_class("experts")), 1)


class TestPresetsAndOverrides(unittest.TestCase):
    def test_latency_preset_is_all_resident_fraction_zero(self):
        policies = resolve_class_policies("latency")
        self.assertEqual(set(policies), set(OFFLOAD_CLASSES))
        for policy in policies.values():
            self.assertEqual(policy, ClassPolicy("resident", 0.0))

    def test_capacity_preset_is_all_ram_fraction_one(self):
        for policy in resolve_class_policies("capacity").values():
            self.assertEqual(policy, ClassPolicy("ram", 1.0))

    def test_auto_preset_fraction_is_measurement_placeholder(self):
        for policy in resolve_class_policies("auto").values():
            self.assertEqual(policy.mode, "auto")
            self.assertIsNone(policy.fraction)

    def test_single_override_beats_the_preset(self):
        policies = resolve_class_policies("latency", "drafter_heads=ram")
        self.assertEqual(policies["drafter_heads"], ClassPolicy("ram", 1.0))
        self.assertEqual(policies["graph_rungs"], ClassPolicy("resident", 0.0))

    def test_override_with_fraction(self):
        policies = resolve_class_policies(
            "latency", "drafter_heads=ram:0.5,graph_rungs=auto"
        )
        self.assertEqual(policies["drafter_heads"], ClassPolicy("ram", 0.5))
        self.assertEqual(policies["graph_rungs"], ClassPolicy("auto", None))

    def test_unknown_profile_is_a_hard_error(self):
        with self.assertRaisesRegex(ValueError, "lane-offload-profile"):
            resolve_class_policies("balanced")

    def test_parse_rejects_unknown_class_policy_and_fraction(self):
        with self.assertRaisesRegex(ValueError, "unknown item class"):
            parse_class_policy_overrides("kv=ram")
        with self.assertRaisesRegex(ValueError, "unknown policy"):
            parse_class_policy_overrides("drafter_heads=disk")
        with self.assertRaisesRegex(ValueError, "not of the form"):
            parse_class_policy_overrides("drafter_heads")
        with self.assertRaisesRegex(ValueError, "not a number"):
            parse_class_policy_overrides("drafter_heads=ram:half")
        with self.assertRaisesRegex(ValueError, r"within\s*\[0, 1\]"):
            parse_class_policy_overrides("drafter_heads=ram:1.5")
        with self.assertRaisesRegex(ValueError, "twice"):
            parse_class_policy_overrides("drafter_heads=ram,drafter_heads=auto")

    def test_parse_empty_is_empty(self):
        self.assertEqual(parse_class_policy_overrides(None), {})
        self.assertEqual(parse_class_policy_overrides("  "), {})


class TestParkInvariants(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        self.backend = CpuFakeMovementBackend()
        self.reg = make_register("capacity", backend=self.backend, clock=self.clock)

    def _cold_item(self, item_id="item", klass="graph_rungs", **kwargs):
        self.reg.register(item_id, klass, 100, 10.0, **kwargs)
        self.clock.advance(10.0)  # age past the 5 s hysteresis window
        return item_id

    def test_park_and_wave_in_roundtrip(self):
        item_id = self._cold_item()
        self.reg.park(item_id)
        self.assertTrue(self.reg.is_parked(item_id))
        self.assertEqual(self.backend.parked_ids, [item_id])
        self.reg.wave_in(item_id)
        self.assertFalse(self.reg.is_parked(item_id))
        self.assertEqual(self.backend.waved_in_ids, [item_id])

    def test_hysteresis_gate_refuses_young_items(self):
        self.reg.register("young", "graph_rungs", 100, 10.0)
        self.clock.advance(4.9)  # inside the 5 s window
        with self.assertRaisesRegex(OffloadRefused, "hysteresis window"):
            self.reg.park("young")
        self.clock.advance(0.2)  # now outside
        self.reg.park("young")
        self.assertTrue(self.reg.is_parked("young"))

    def test_touch_resets_the_hysteresis_clock(self):
        item_id = self._cold_item()
        self.reg.touch(item_id)
        with self.assertRaisesRegex(OffloadRefused, "hysteresis window"):
            self.reg.park(item_id)

    def test_hot_items_are_refused(self):
        hot = {"flag": True}
        self.reg.register(
            "active_rung", "graph_rungs", 100, 10.0, hot=lambda: hot["flag"]
        )
        self.clock.advance(10.0)
        with self.assertRaisesRegex(OffloadRefused, "hot"):
            self.reg.park("active_rung")
        hot["flag"] = False
        self.reg.park("active_rung")

    def test_prio_protected_never_parked_to_make_room(self):
        item_id = self._cold_item("protected", prio_protected=True)
        with self.assertRaisesRegex(OffloadRefused, "priority-protected"):
            self.reg.park(item_id, make_room_for="someone_else")
        # A direct policy park (not displacement) is still allowed.
        self.reg.park(item_id)
        self.assertTrue(self.reg.is_parked(item_id))

    def test_resident_policy_refuses(self):
        reg = make_register("latency", backend=self.backend, clock=self.clock)
        reg.register("x", "graph_rungs", 100, 10.0)
        self.clock.advance(10.0)
        with self.assertRaisesRegex(OffloadRefused, "'resident'"):
            reg.park("x")
        self.assertEqual(reg.stats.park_refusals, 1)

    def test_auto_policy_requires_saturation_pressure(self):
        reg = make_register("auto", backend=self.backend, clock=self.clock)
        reg.register("x", "graph_rungs", 100, 10.0)
        self.clock.advance(10.0)
        with self.assertRaisesRegex(OffloadRefused, "sensor"):
            reg.park("x")  # no sensor attached
        pressure = {"on": False}
        reg.set_saturation_sensor(lambda: pressure["on"])
        with self.assertRaisesRegex(OffloadRefused, "sensor"):
            reg.park("x")  # sensor attached, no pressure
        pressure["on"] = True
        reg.park("x")
        self.assertTrue(reg.is_parked("x"))

    def test_fraction_cap_bounds_parked_depth(self):
        # ram:0.5 over two 100-B items = at most 100 B parked.
        reg = make_register(
            "latency",
            "graph_rungs=ram:0.5",
            backend=self.backend,
            clock=self.clock,
        )
        reg.register("a", "graph_rungs", 100, 10.0)
        reg.register("b", "graph_rungs", 100, 10.0)
        self.clock.advance(10.0)
        reg.park("a")
        with self.assertRaisesRegex(OffloadRefused, "fraction cap"):
            reg.park("b")
        reg.wave_in("a")
        self.clock.advance(10.0)
        reg.park("b")  # depth freed -> the other item may park

    def test_park_is_idempotent_and_unknown_id_is_an_error(self):
        item_id = self._cold_item()
        self.reg.park(item_id)
        self.reg.park(item_id)  # no second backend call
        self.assertEqual(self.backend.parked_ids, [item_id])
        with self.assertRaisesRegex(ValueError, "unknown offload item"):
            self.reg.park("ghost")


class TestLatencyTermApi(unittest.TestCase):
    def test_parked_is_a_latency_term_never_unavailable(self):
        clock = _Clock()
        reg = make_register("capacity", clock=clock)
        reg.register("head", "drafter_heads", 100, 55.0)
        reg.register("rung", "graph_rungs", 100, 7.5)
        clock.advance(10.0)
        self.assertEqual(reg.retrieval_latency_ms("head"), 0.0)
        reg.park("head")
        reg.park("rung")
        self.assertEqual(reg.retrieval_latency_ms("head"), 55.0)
        self.assertEqual(reg.latency_term_ms(), 62.5)
        self.assertEqual(reg.latency_term_ms("graph_rungs"), 7.5)
        # wave_in is never refused: retrieval cost was the whole answer.
        reg.wave_in("head")
        self.assertEqual(reg.retrieval_latency_ms("head"), 0.0)

    def test_restore_cost_estimator_is_live(self):
        clock = _Clock()
        reg = make_register("capacity", clock=clock)
        measured = {"ms": 40.0}
        reg.register("head", "drafter_heads", 100, lambda: measured["ms"])
        clock.advance(10.0)
        reg.park("head")
        self.assertEqual(reg.retrieval_latency_ms("head"), 40.0)
        measured["ms"] = 72.0  # telemetry updates the estimate in place
        self.assertEqual(reg.retrieval_latency_ms("head"), 72.0)


class TestPhaseBoundaryPlanner(unittest.TestCase):
    """Stage-2 hook (Erg. 7c): plans deterministically, moves nothing."""

    def setUp(self):
        self.clock = _Clock()
        self.backend = CpuFakeMovementBackend()
        self.reg = make_register(
            "capacity",
            backend=self.backend,
            clock=self.clock,
            phase_hysteresis_window_s=0.0,
            overlap_budget_fn=lambda item, cur, nxt: True,
        )

    def _phase_item(self, item_id, phases, **kwargs):
        self.reg.register(
            item_id,
            kwargs.pop("klass", "drafter_heads"),
            kwargs.pop("size_bytes", 100),
            10.0,
            phase_mask=phases,
            time_constant_tier="phase",
            **kwargs,
        )

    def test_candidates_are_items_the_next_phase_does_not_need(self):
        self._phase_item("draft_only", ("draft",))
        self._phase_item("verify_only", ("verify",))
        self._phase_item("both", ("draft", "verify"))
        plan = self.reg.on_phase_boundary("draft", "verify")
        self.assertEqual(plan.park_candidates, ["draft_only"])
        self.assertIn("needed in phase 'verify'", plan.skipped["verify_only"])
        # Planning must not move anything (CPU phase: no-op interface).
        self.assertEqual(self.backend.parked_ids, [])
        self.assertFalse(self.reg.is_parked("draft_only"))

    def test_plan_is_deterministic_sorted_order(self):
        for item_id in ("c", "a", "b"):
            self._phase_item(item_id, ("draft",))
        plan = self.reg.on_phase_boundary("draft", "verify")
        self.assertEqual(plan.park_candidates, ["a", "b", "c"])

    def test_empty_mask_means_needed_in_every_phase(self):
        self._phase_item("always", ())
        plan = self.reg.on_phase_boundary("draft", "verify")
        self.assertEqual(plan.park_candidates, [])
        self.assertIn("every phase", plan.skipped["always"])

    def test_turn_tier_items_are_not_stage2_candidates(self):
        self.reg.register(
            "turn_item",
            "graph_rungs",
            100,
            10.0,
            phase_mask=("draft",),
            time_constant_tier="turn",
        )
        plan = self.reg.on_phase_boundary("draft", "verify")
        self.assertEqual(plan.park_candidates, [])
        self.assertNotIn("turn_item", plan.skipped)  # not even considered

    def test_prio_protected_hot_and_young_are_excluded(self):
        self._phase_item("protected", ("draft",), prio_protected=True)
        self._phase_item("hot", ("draft",), hot=lambda: True)
        plan = self.reg.on_phase_boundary("draft", "verify")
        self.assertEqual(plan.park_candidates, [])
        self.assertIn("priority-protected", plan.skipped["protected"])
        self.assertIn("hot", plan.skipped["hot"])
        # Phase hysteresis: with a window, a just-touched item is excluded.
        reg = make_register(
            "capacity",
            clock=self.clock,
            phase_hysteresis_window_s=0.050,
            overlap_budget_fn=lambda item, cur, nxt: True,
        )
        reg.register(
            "young",
            "drafter_heads",
            100,
            10.0,
            phase_mask=("draft",),
            time_constant_tier="phase",
        )
        plan = reg.on_phase_boundary("draft", "verify")
        self.assertIn("phase hysteresis", plan.skipped["young"])
        self.clock.advance(0.060)
        plan = reg.on_phase_boundary("draft", "verify")
        self.assertEqual(plan.park_candidates, ["young"])

    def test_overlap_budget_gates_candidates(self):
        # Honest 7c physics via injected fake: 12 GB/s PCIe, 30 ms round =>
        # a 1.8 GB drafter (round trip) is NOT hideable, a 4 MiB workspace is.
        pcie_bytes_per_s = 12e9
        hideable_ms = 30.0

        def fake_overlap(item, cur, nxt):
            move_ms = 2 * item.size_bytes / pcie_bytes_per_s * 1000.0
            return move_ms <= hideable_ms

        reg = make_register(
            "capacity", clock=self.clock, overlap_budget_fn=fake_overlap
        )
        reg.register(
            "big_head",
            "drafter_heads",
            1_800_000_000,
            70.0,
            phase_mask=("draft",),
            time_constant_tier="phase",
        )
        reg.register(
            "small_ws",
            "lane_workspaces",
            4_000_000,
            1.0,
            phase_mask=("draft",),
            time_constant_tier="phase",
        )
        plan = reg.on_phase_boundary("draft", "verify")
        self.assertEqual(plan.park_candidates, ["small_ws"])
        self.assertIn("not hideable", plan.skipped["big_head"])

    def test_without_overlap_budget_fn_the_hook_is_a_true_noop(self):
        reg = make_register("capacity", clock=self.clock)
        reg.register(
            "x",
            "drafter_heads",
            100,
            10.0,
            phase_mask=("draft",),
            time_constant_tier="phase",
        )
        plan = reg.on_phase_boundary("draft", "verify")
        self.assertEqual(plan.park_candidates, [])
        self.assertIn("overlap-budget", plan.skipped["x"])

    def test_fraction_cap_applies_to_the_plan(self):
        reg = make_register(
            "latency",
            "drafter_heads=ram:0.5",
            clock=self.clock,
            overlap_budget_fn=lambda item, cur, nxt: True,
        )
        for item_id in ("a", "b"):
            reg.register(
                item_id,
                "drafter_heads",
                100,
                10.0,
                phase_mask=("draft",),
                time_constant_tier="phase",
            )
        plan = reg.on_phase_boundary("draft", "verify")
        self.assertEqual(plan.park_candidates, ["a"])
        self.assertIn("fraction cap", plan.skipped["b"])


class TestServerArgsParsing(unittest.TestCase):
    """model_path='dummy' short-circuits __post_init__, so the handler is
    driven in isolation with exactly the fields under test."""

    def _args(self, **kwargs):
        return ServerArgs(model_path="dummy", **kwargs)

    def test_defaults_resolve(self):
        args = self._args()
        self.assertEqual(args.lane_offload_profile, "latency")
        self.assertIsNone(args.lane_offload_class_policy)
        args._handle_lane_offload_register()  # must not raise

    def test_valid_profile_and_overrides(self):
        args = self._args(
            lane_offload_profile="capacity",
            lane_offload_class_policy="drafter_heads=ram:0.5,graph_rungs=auto",
        )
        args._handle_lane_offload_register()  # must not raise

    def test_unknown_profile_is_rejected(self):
        args = self._args(lane_offload_profile="balanced")
        with self.assertRaisesRegex(ValueError, "lane-offload-profile"):
            args._handle_lane_offload_register()

    def test_unknown_class_and_policy_are_rejected(self):
        for policy in ("kv=ram", "drafter_heads=disk", "drafter_heads=ram:2"):
            args = self._args(lane_offload_class_policy=policy)
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                args._handle_lane_offload_register()

    def test_cli_roundtrip(self):
        import argparse

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        parsed = parser.parse_args(
            [
                "--model-path",
                "dummy",
                "--lane-offload-profile",
                "capacity",
                "--lane-offload-class-policy",
                "drafter_heads=ram:0.5",
            ]
        )
        self.assertEqual(parsed.lane_offload_profile, "capacity")
        self.assertEqual(parsed.lane_offload_class_policy, "drafter_heads=ram:0.5")
        with self.assertRaises(SystemExit):  # choices enforced at the CLI
            parser.parse_args(["--model-path", "dummy", "--lane-offload-profile", "x"])


class TestFlagGate(unittest.TestCase):
    """SGLANG_OFFLOAD_REGISTER off (default) => zero behavior anywhere."""

    def setUp(self):
        reset_global_register()
        self.addCleanup(reset_global_register)

    def test_flag_off_means_no_register_and_noop_helpers(self):
        with patch.dict(os.environ, {"SGLANG_OFFLOAD_REGISTER": "0"}):
            self.assertIsNone(get_global_register())
            self.assertIsNone(maybe_register_item("x", "graph_rungs", 10, 1.0))
            maybe_touch_item("x")  # must not raise

    def test_flag_on_creates_default_latency_register(self):
        with patch.dict(os.environ, {"SGLANG_OFFLOAD_REGISTER": "1"}):
            reg = get_global_register()
            self.assertIsNotNone(reg)
            self.assertEqual(
                reg.policies["drafter_heads"], ClassPolicy("resident", 0.0)
            )
            item = maybe_register_item("x", "graph_rungs", 10, 1.0)
            self.assertIsNotNone(item)
            # Adapter re-registration refreshes instead of raising.
            item2 = maybe_register_item("x", "graph_rungs", 20, 1.0)
            self.assertEqual(item2.size_bytes, 20)

    def test_configure_from_server_args_knobs(self):
        with patch.dict(os.environ, {"SGLANG_OFFLOAD_REGISTER": "1"}):
            reg = configure_global_register("capacity", "drafter_heads=resident")
            self.assertIs(get_global_register(), reg)
            self.assertEqual(
                reg.policies["drafter_heads"], ClassPolicy("resident", 0.0)
            )
            self.assertEqual(reg.policies["graph_rungs"], ClassPolicy("ram", 1.0))

    def test_workspace_adapter_books_only_when_enabled(self):
        from sglang.srt.runtime_context import (
            _note_workspace_in_offload_register,
        )

        with patch.dict(os.environ, {"SGLANG_OFFLOAD_REGISTER": "0"}):
            _note_workspace_in_offload_register(None, "ws", object(), True)
            self.assertIsNone(get_global_register())
        with patch.dict(os.environ, {"SGLANG_OFFLOAD_REGISTER": "1"}):
            _note_workspace_in_offload_register(None, "ws", object(), True)
            reg = get_global_register()
            item = reg.get("lane_workspace/None/ws")
            self.assertIsNotNone(item)
            self.assertEqual(item.offload_class, "lane_workspaces")
            # Reuse touches instead of re-registering.
            before = item.last_access_s
            _note_workspace_in_offload_register(None, "ws", object(), False)
            self.assertGreaterEqual(reg.get(item.item_id).last_access_s, before)

    def test_input_buffer_adapter_books_only_when_enabled(self):
        import torch

        from sglang.srt.model_executor.input_buffers import share_input_buffer

        buf = torch.zeros(8, dtype=torch.int64)
        with patch.dict(os.environ, {"SGLANG_OFFLOAD_REGISTER": "0"}):
            share_input_buffer("test_offload_reg_off", buf)
            self.assertIsNone(get_global_register())
        with patch.dict(os.environ, {"SGLANG_OFFLOAD_REGISTER": "1"}):
            share_input_buffer("test_offload_reg_on", buf)
            reg = get_global_register()
            items = reg.items_of_class("lane_workspaces")
            self.assertTrue(any("test_offload_reg_on" in i.item_id for i in items))
            booked = next(i for i in items if "test_offload_reg_on" in i.item_id)
            self.assertEqual(booked.size_bytes, buf.numel() * buf.element_size())


if __name__ == "__main__":
    unittest.main()
