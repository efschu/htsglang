"""#286 Erg. 8 -- gdn_state_sets as an offload-register class, CPU phase.

Hermetic unit tests (no GPU, no CUDA; fakes/meta tensors only) for:

* ladder parsing + validation (nonsense rungs = hard error),
* the SessionSetLadder contract (plateaus, immediate raise, lowering
  hysteresis in admission cycles, the floor rung),
* the admission-boundary hook (deterministic park/wave-in plans, moves
  nothing in the CPU phase),
* the enforced invariant "an arriving session never meets a parked set",
* hot protection of active sessions' sets,
* the per-set size model from real tensor shapes (fakes + meta tensors),
* the MambaPool adapter incl. the allocator activity probe,
* ServerArgs parsing of the new flags,
* flag off (SGLANG_OFFLOAD_REGISTER unset) = zero behavior.
"""

import dataclasses
import os
import unittest
from typing import List, Optional
from unittest.mock import patch

from sglang.srt.model_executor.offload_gdn_states import (
    GDN_STATE_SET_CLASS,
    SessionSetLadder,
    attach_state_set_activity_probe,
    mamba_state_set_nbytes,
    parse_gdn_state_set_ladder,
    register_mamba_state_sets,
    state_set_item_id,
)
from sglang.srt.model_executor.offload_register import (
    CpuFakeMovementBackend,
    OffloadRegister,
    get_global_register,
    reset_global_register,
    resolve_class_policies,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _Clock:
    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakeTensor:
    """Only what the size model consults: numel and element_size."""

    def __init__(self, numel: int, element_size: int):
        self._numel = numel
        self._element_size = element_size

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return self._element_size


@dataclasses.dataclass(frozen=True, kw_only=True)
class _FakeState:
    """Shape-compatible stand-in for MambaPool.State (dataclass walk)."""

    conv: List[_FakeTensor]
    temporal: _FakeTensor
    replayssm_d: Optional[_FakeTensor] = None
    intermediate_ssm: Optional[_FakeTensor] = None
    intermediate_conv_window: Optional[List[_FakeTensor]] = None


def _make_register(profile="capacity", clock=None, **kwargs):
    kwargs.setdefault("backend", CpuFakeMovementBackend())
    kwargs.setdefault("hysteresis_window_s", 5.0)
    return OffloadRegister(
        policies=resolve_class_policies(profile),
        clock=clock or _Clock(),
        **kwargs,
    )


def _register_sets(reg, count, hot_slots=(), size_bytes=100):
    """Book `count` state-set items; sets in hot_slots report an active
    session."""
    active = set(hot_slots)
    for slot in range(1, count + 1):
        reg.register(
            state_set_item_id(slot),
            GDN_STATE_SET_CLASS,
            size_bytes,
            1.0,
            hot=lambda s=slot: s in active,
            va_stable_required=True,
            time_constant_tier="turn",
        )
    return active


class TestLadderParsing(unittest.TestCase):
    def test_valid_specs(self):
        self.assertEqual(parse_gdn_state_set_ladder("4,2,1"), (4, 2, 1))
        self.assertEqual(parse_gdn_state_set_ladder(" 8 , 3 "), (8, 3))
        self.assertEqual(parse_gdn_state_set_ladder("2"), (2,))

    def test_off_is_none(self):
        self.assertIsNone(parse_gdn_state_set_ladder(None))
        self.assertIsNone(parse_gdn_state_set_ladder(""))
        self.assertIsNone(parse_gdn_state_set_ladder("   "))

    def test_nonsense_rungs_are_hard_errors(self):
        with self.assertRaisesRegex(ValueError, "not an integer"):
            parse_gdn_state_set_ladder("4,x,1")
        with self.assertRaisesRegex(ValueError, "positive"):
            parse_gdn_state_set_ladder("4,0")
        with self.assertRaisesRegex(ValueError, "positive"):
            parse_gdn_state_set_ladder("-2")
        with self.assertRaisesRegex(ValueError, "strictly descending"):
            parse_gdn_state_set_ladder("2,4")
        with self.assertRaisesRegex(ValueError, "strictly descending"):
            parse_gdn_state_set_ladder("4,4,1")
        with self.assertRaisesRegex(ValueError, "empty rung"):
            parse_gdn_state_set_ladder("4,,1")


class TestSessionSetLadder(unittest.TestCase):
    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "exceeds the pool"):
            SessionSetLadder((8, 2), max_sets=4)
        with self.assertRaisesRegex(ValueError, "hysteresis_cycles"):
            SessionSetLadder((4, 2), max_sets=4, hysteresis_cycles=0)
        with self.assertRaisesRegex(ValueError, "max_sets"):
            SessionSetLadder((1,), max_sets=0)
        with self.assertRaisesRegex(ValueError, "at least one rung"):
            SessionSetLadder((), max_sets=4)

    def test_starts_on_top_rung(self):
        ladder = SessionSetLadder((4, 2, 1), max_sets=8)
        self.assertEqual(ladder.current_rung, 4)

    def test_lowering_waits_for_hysteresis(self):
        ladder = SessionSetLadder((4, 2, 1), max_sets=8, hysteresis_cycles=3)
        # Two below-threshold cycles: rung holds.
        self.assertEqual(ladder.on_admission_cycle(2), 4)
        self.assertEqual(ladder.on_admission_cycle(2), 4)
        # Third consecutive cycle: rung drops to the covering plateau.
        self.assertEqual(ladder.on_admission_cycle(2), 2)
        self.assertEqual(ladder.current_rung, 2)

    def test_any_at_or_above_cycle_resets_the_below_counter(self):
        ladder = SessionSetLadder((4, 2, 1), max_sets=8, hysteresis_cycles=2)
        self.assertEqual(ladder.on_admission_cycle(1), 4)
        # Back at the current rung: counter resets.
        self.assertEqual(ladder.on_admission_cycle(4), 4)
        self.assertEqual(ladder.on_admission_cycle(1), 4)
        # Only now the second CONSECUTIVE below cycle lowers.
        self.assertEqual(ladder.on_admission_cycle(1), 1)

    def test_raising_is_immediate_and_covers_needed(self):
        ladder = SessionSetLadder((4, 2, 1), max_sets=8, hysteresis_cycles=1)
        self.assertEqual(ladder.on_admission_cycle(1), 1)
        # One admission boundary later three sessions arrive: plateau 4, now.
        self.assertEqual(ladder.on_admission_cycle(3), 4)
        self.assertEqual(ladder.current_rung, 4)

    def test_above_top_rung_the_target_follows_the_session_count(self):
        # Correctness outranks the plateaus: rungs cap comfort, not sessions.
        ladder = SessionSetLadder((4, 2, 1), max_sets=8, hysteresis_cycles=1)
        self.assertEqual(ladder.on_admission_cycle(6), 6)
        # Capped at the pool size; admission beyond it is the allocator's
        # failure, not the ladder's.
        self.assertEqual(ladder.on_admission_cycle(11), 8)

    def test_bottom_rung_is_the_floor(self):
        ladder = SessionSetLadder((4, 2), max_sets=8, hysteresis_cycles=1)
        self.assertEqual(ladder.on_admission_cycle(0), 2)
        self.assertEqual(ladder.on_admission_cycle(0), 2)


class TestAdmissionBoundaryHook(unittest.TestCase):
    def test_without_ladder_the_hook_plans_nothing(self):
        reg = _make_register()
        _register_sets(reg, 4)
        plan = reg.on_admission_boundary(2, 1)
        self.assertEqual(plan.target_sets, 4)
        self.assertEqual(plan.park_candidates, [])
        self.assertEqual(plan.wave_in_candidates, [])
        self.assertEqual(reg.stats.admission_plans, 1)

    def test_lowering_plan_parks_highest_free_sets_deterministically(self):
        def build():
            clock = _Clock()
            reg = _make_register(clock=clock)
            _register_sets(reg, 4, hot_slots=(1, 2))
            reg.set_session_ladder(
                SessionSetLadder((4, 2, 1), max_sets=4, hysteresis_cycles=1)
            )
            clock.advance(10.0)  # outside the turn hysteresis window
            return reg

        plans = [build().on_admission_boundary(2, 0) for _ in range(2)]
        for plan in plans:
            self.assertEqual(plan.needed_sets, 2)
            self.assertEqual(plan.target_sets, 2)
            self.assertEqual(
                plan.park_candidates,
                [state_set_item_id(4), state_set_item_id(3)],
            )
            self.assertEqual(plan.wave_in_candidates, [])
        # Deterministic: identical inputs, identical plans.
        self.assertEqual(plans[0], plans[1])

    def test_cpu_phase_plans_but_moves_nothing(self):
        clock = _Clock()
        backend = CpuFakeMovementBackend()
        reg = _make_register(clock=clock, backend=backend)
        _register_sets(reg, 4)
        reg.set_session_ladder(
            SessionSetLadder((4, 1), max_sets=4, hysteresis_cycles=1)
        )
        clock.advance(10.0)
        plan = reg.on_admission_boundary(0, 0)
        self.assertEqual(len(plan.park_candidates), 3)
        self.assertEqual(backend.parked_ids, [])
        self.assertEqual(backend.waved_in_ids, [])
        for item_id in plan.park_candidates:
            self.assertFalse(reg.is_parked(item_id))

    def test_hot_sets_of_active_sessions_are_never_park_candidates(self):
        clock = _Clock()
        reg = _make_register(clock=clock)
        # Sessions sit on the HIGH slots -- exactly where the planner would
        # otherwise park first.
        _register_sets(reg, 4, hot_slots=(3, 4))
        reg.set_session_ladder(
            SessionSetLadder((4, 2, 1), max_sets=4, hysteresis_cycles=1)
        )
        clock.advance(10.0)
        plan = reg.on_admission_boundary(2, 0)
        self.assertEqual(
            plan.park_candidates,
            [state_set_item_id(2), state_set_item_id(1)],
        )
        self.assertEqual(plan.skipped[state_set_item_id(4)], "hot (active session)")
        self.assertEqual(plan.skipped[state_set_item_id(3)], "hot (active session)")

    def test_hot_set_park_is_refused_by_the_register_too(self):
        from sglang.srt.model_executor.offload_register import OffloadRefused

        clock = _Clock()
        reg = _make_register(clock=clock)
        _register_sets(reg, 2, hot_slots=(2,))
        clock.advance(10.0)
        with self.assertRaisesRegex(OffloadRefused, "hot"):
            reg.park(state_set_item_id(2))

    def test_turn_hysteresis_window_gates_the_plan(self):
        clock = _Clock()
        reg = _make_register(clock=clock)
        _register_sets(reg, 2)
        reg.set_session_ladder(
            SessionSetLadder((2, 1), max_sets=2, hysteresis_cycles=1)
        )
        # Freshly registered = freshly touched: nothing may park yet.
        plan = reg.on_admission_boundary(0, 0)
        self.assertEqual(plan.park_candidates, [])
        self.assertIn("hysteresis window", plan.skipped[state_set_item_id(2)])
        clock.advance(10.0)
        plan = reg.on_admission_boundary(0, 0)
        self.assertEqual(plan.park_candidates, [state_set_item_id(2)])

    def test_resident_policy_keeps_every_set_resident(self):
        clock = _Clock()
        reg = _make_register(profile="latency", clock=clock)
        _register_sets(reg, 3)
        reg.set_session_ladder(
            SessionSetLadder((3, 1), max_sets=3, hysteresis_cycles=1)
        )
        clock.advance(10.0)
        plan = reg.on_admission_boundary(0, 0)
        self.assertEqual(plan.park_candidates, [])
        for slot in (2, 3):
            self.assertEqual(
                plan.skipped[state_set_item_id(slot)],
                "class policy is 'resident'",
            )

    def test_incoming_sessions_trigger_wave_ins_before_admission(self):
        clock = _Clock()
        reg = _make_register(clock=clock)
        _register_sets(reg, 4)
        reg.set_session_ladder(
            SessionSetLadder((4, 2, 1), max_sets=4, hysteresis_cycles=1)
        )
        clock.advance(10.0)
        # Park down to 1 resident set (needed=0 -> floor rung 1).
        plan = reg.on_admission_boundary(0, 0)
        for item_id in plan.park_candidates:
            reg.park(item_id)
        self.assertEqual(len(plan.park_candidates), 3)
        # One session runs, two arrive: the plan must wave sets back BEFORE
        # the admission, lowest parked ids first, and park nothing.
        plan = reg.on_admission_boundary(1, 2)
        self.assertEqual(plan.needed_sets, 3)
        self.assertEqual(plan.target_sets, 4)  # plateau above 3
        self.assertEqual(
            plan.wave_in_candidates,
            [
                state_set_item_id(2),
                state_set_item_id(3),
                state_set_item_id(4),
            ],
        )
        self.assertEqual(plan.park_candidates, [])

    def test_invariant_admission_never_meets_a_parked_set(self):
        """Drive a full churn sequence, executing every plan, and assert the
        invariant at every boundary: after executing the plan, at least
        needed = running + incoming sets are resident, and the specific sets
        an admission would take (lowest inactive slots) are resident."""
        clock = _Clock()
        reg = _make_register(clock=clock)
        total = 6
        active = set()
        _register_sets(reg, total)
        # Rebind hot to the mutable active-session set of this simulation.
        for slot in range(1, total + 1):
            reg.get(state_set_item_id(slot)).hot = lambda s=slot: s in active
        reg.set_session_ladder(
            SessionSetLadder((6, 3, 1), max_sets=total, hysteresis_cycles=2)
        )

        def boundary(incoming: int):
            clock.advance(10.0)
            plan = reg.on_admission_boundary(len(active), incoming)
            for item_id in plan.wave_in_candidates:
                reg.wave_in(item_id)
            for item_id in plan.park_candidates:
                reg.park(item_id)
            # Admit: each incoming session takes the lowest inactive slot.
            for _ in range(incoming):
                slot = min(s for s in range(1, total + 1) if s not in active)
                self.assertFalse(
                    reg.is_parked(state_set_item_id(slot)),
                    f"admission met parked set {slot}",
                )
                active.add(slot)
                reg.touch(state_set_item_id(slot))
            resident = sum(
                1
                for s in range(1, total + 1)
                if not reg.is_parked(state_set_item_id(s))
            )
            self.assertGreaterEqual(resident, min(len(active), total))

        # Ramp up, drain, idle long enough to lower, then a burst arrives.
        for incoming in (2, 2, 1, 0):
            boundary(incoming)
        active.clear()
        for _ in range(4):  # hysteresis-lowering cycles while idle
            boundary(0)
        boundary(4)  # burst: wave-ins must precede these admissions
        boundary(2)

    def test_negative_counts_are_hard_errors(self):
        reg = _make_register()
        with self.assertRaises(ValueError):
            reg.on_admission_boundary(-1, 0)
        with self.assertRaises(ValueError):
            reg.on_admission_boundary(0, -1)


class TestSetSizeModel(unittest.TestCase):
    def test_per_set_bytes_from_fake_shapes(self):
        num_slots = 5
        state = _FakeState(
            conv=[
                _FakeTensor(numel=num_slots * 100, element_size=2),
                _FakeTensor(numel=num_slots * 60, element_size=2),
            ],
            temporal=_FakeTensor(numel=num_slots * 400, element_size=2),
            replayssm_d=_FakeTensor(numel=num_slots * 30, element_size=4),
        )
        self.assertEqual(
            mamba_state_set_nbytes(state, num_slots),
            100 * 2 + 60 * 2 + 400 * 2 + 30 * 4,
        )

    def test_transient_spec_scratch_is_excluded(self):
        num_slots = 4
        base = dict(
            conv=[_FakeTensor(numel=num_slots * 10, element_size=2)],
            temporal=_FakeTensor(numel=num_slots * 20, element_size=2),
        )
        with_spec = _FakeState(
            **base,
            intermediate_ssm=_FakeTensor(numel=10_000, element_size=2),
            intermediate_conv_window=[_FakeTensor(numel=10_000, element_size=2)],
        )
        self.assertEqual(
            mamba_state_set_nbytes(with_spec, num_slots),
            mamba_state_set_nbytes(_FakeState(**base), num_slots),
        )

    def test_none_fields_contribute_nothing(self):
        state = _FakeState(
            conv=[_FakeTensor(numel=8, element_size=2)],
            temporal=_FakeTensor(numel=16, element_size=2),
            replayssm_d=None,
        )
        self.assertEqual(mamba_state_set_nbytes(state, 2), 8 + 16)

    def test_meta_tensors_resolve_without_cuda(self):
        import torch

        num_slots = 3
        state = _FakeState(
            conv=[
                torch.empty((2, num_slots, 7, 3), dtype=torch.bfloat16, device="meta")
            ],
            temporal=torch.empty(
                (2, num_slots, 4, 8), dtype=torch.bfloat16, device="meta"
            ),
        )
        expected = (2 * 7 * 3 + 2 * 4 * 8) * 2
        self.assertEqual(mamba_state_set_nbytes(state, num_slots), expected)

    def test_invalid_slot_count_is_a_hard_error(self):
        with self.assertRaises(ValueError):
            mamba_state_set_nbytes(_FakeState(conv=[], temporal=_FakeTensor(1, 1)), 0)


class _FakePool:
    """MambaPool stand-in: only what the adapter consults."""

    def __init__(self, size: int, num_layers: int = 2):
        self.size = size
        num_slots = size + 1
        self.mamba_cache = _FakeState(
            conv=[_FakeTensor(numel=num_slots * 100, element_size=2)],
            temporal=_FakeTensor(numel=num_slots * 200, element_size=2),
        )


class TestMambaPoolAdapter(unittest.TestCase):
    def setUp(self):
        reset_global_register()
        self.addCleanup(reset_global_register)

    def test_flag_off_registers_nothing(self):
        with patch.dict(os.environ, {"SGLANG_OFFLOAD_REGISTER": "0"}):
            self.assertEqual(register_mamba_state_sets(_FakePool(4)), [])
            self.assertIsNone(get_global_register())

    def test_registers_one_item_per_session_slot_never_slot_zero(self):
        with patch.dict(os.environ, {"SGLANG_OFFLOAD_REGISTER": "1"}):
            ids = register_mamba_state_sets(_FakePool(4))
            self.assertEqual(ids, [state_set_item_id(s) for s in (1, 2, 3, 4)])
            reg = get_global_register()
            items = reg.items_of_class(GDN_STATE_SET_CLASS)
            self.assertEqual(len(items), 4)
            self.assertIsNone(reg.get(state_set_item_id(0)))
            for item in items:
                self.assertTrue(item.va_stable_required)
                self.assertEqual(item.time_constant_tier, "turn")
                # Per-set share of the [layers, slots, ...] tensors.
                self.assertEqual(item.size_bytes, 100 * 2 + 200 * 2)

    def test_without_probe_every_set_is_hot_the_safe_direction(self):
        with patch.dict(os.environ, {"SGLANG_OFFLOAD_REGISTER": "1"}):
            register_mamba_state_sets(_FakePool(2))
            reg = get_global_register()
            for item in reg.items_of_class(GDN_STATE_SET_CLASS):
                self.assertTrue(item.hot())

    def test_activity_probe_marks_allocated_slots_hot(self):
        from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator

        with patch.dict(os.environ, {"SGLANG_OFFLOAD_REGISTER": "1"}):
            pool = _FakePool(3)
            register_mamba_state_sets(pool)
            allocator = MambaSlotAllocator(size=3, device="cpu")
            self.assertTrue(attach_state_set_activity_probe(pool, allocator))
            reg = get_global_register()
            hot = {i.item_id: i.hot() for i in reg.items_of_class(GDN_STATE_SET_CLASS)}
            self.assertEqual(set(hot.values()), {False})  # all free
            taken = allocator.alloc(2)  # slots 1 and 2
            self.assertIsNotNone(taken)
            self.assertTrue(reg.get(state_set_item_id(1)).hot())
            self.assertTrue(reg.get(state_set_item_id(2)).hot())
            self.assertFalse(reg.get(state_set_item_id(3)).hot())

    def test_probe_attach_is_a_noop_when_disabled(self):
        pool = _FakePool(2)
        self.assertFalse(
            attach_state_set_activity_probe(pool, allocator=None, enabled=False)
        )
        self.assertIsNone(getattr(pool, "_offload_slot_active_fn", None))

    def test_ladder_from_global_server_args(self):
        from sglang.srt.server_args import (
            ServerArgs,
            set_global_server_args_for_scheduler,
        )

        args = ServerArgs(
            model_path="dummy",
            gdn_state_set_ladder="3,1",
            gdn_state_set_ladder_hysteresis=4,
        )
        set_global_server_args_for_scheduler(args)
        try:
            with patch.dict(os.environ, {"SGLANG_OFFLOAD_REGISTER": "1"}):
                register_mamba_state_sets(_FakePool(4))
                ladder = get_global_register().session_ladder
                self.assertIsNotNone(ladder)
                self.assertEqual(ladder.rungs, (3, 1))
                self.assertEqual(ladder.max_sets, 4)
                self.assertEqual(ladder.hysteresis_cycles, 4)
        finally:
            set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_no_ladder_flag_means_no_ladder(self):
        from sglang.srt.server_args import (
            ServerArgs,
            set_global_server_args_for_scheduler,
        )

        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        with patch.dict(os.environ, {"SGLANG_OFFLOAD_REGISTER": "1"}):
            register_mamba_state_sets(_FakePool(4))
            self.assertIsNone(get_global_register().session_ladder)

    def test_real_mamba_pool_cpu_registers_sets(self):
        """End-to-end on a REAL MambaPool built on the CPU device (no CUDA):
        the pool constructor books one item per session slot with the true
        per-set byte figure."""

        from sglang.srt.configs.mamba_utils import (
            Mamba2CacheParams,
            Mamba2StateShape,
        )
        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.memory_pool import MambaPool

        with patch.dict(os.environ, {"SGLANG_OFFLOAD_REGISTER": "1"}):
            with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
                shape = Mamba2StateShape.create(
                    tp_world_size=1,
                    intermediate_size=512,
                    n_groups=4,
                    num_heads=8,
                    head_dim=64,
                    state_size=32,
                    conv_kernel=4,
                )
                params = Mamba2CacheParams(shape=shape, layers=[0, 1, 2])
            size = 5
            pool = MambaPool(
                size=size,
                spec_state_size=size,
                cache_params=params,
                mamba_layer_ids=[0, 1, 2],
                device="cpu",
            )
            reg = get_global_register()
            items = reg.items_of_class(GDN_STATE_SET_CLASS)
            self.assertEqual(len(items), size)
            expected = mamba_state_set_nbytes(pool.mamba_cache, size + 1)
            self.assertGreater(expected, 0)
            for item in items:
                self.assertEqual(item.size_bytes, expected)
            # Sanity: per-set bytes recompose (x num_slots) to the pool's
            # own accounting of the persistent state (conv + temporal here;
            # no spec intermediates, no rings in this configuration).
            self.assertEqual(
                expected * (size + 1),
                int(pool.mem_usage * (1 << 30)),
            )


class TestServerArgsFlags(unittest.TestCase):
    """model_path='dummy' short-circuits __post_init__ (same pattern as the
    lane-offload flag tests), so the handler is driven explicitly."""

    def _args(self, **kwargs):
        from sglang.srt.server_args import ServerArgs

        return ServerArgs(model_path="dummy", **kwargs)

    def test_defaults_are_ladder_off(self):
        args = self._args()
        self.assertIsNone(args.gdn_state_set_ladder)
        self.assertEqual(args.gdn_state_set_ladder_hysteresis, 2)
        args._handle_gdn_state_set_ladder()  # must not raise

    def test_valid_ladder_passes(self):
        args = self._args(
            gdn_state_set_ladder="4,2,1", gdn_state_set_ladder_hysteresis=1
        )
        args._handle_gdn_state_set_ladder()  # must not raise
        self.assertEqual(args.gdn_state_set_ladder, "4,2,1")

    def test_invalid_ladder_fails_the_boot(self):
        for spec, message in (
            ("1,2,4", "strictly descending"),
            ("vier", "not an integer"),
            ("4,0", "positive"),
        ):
            args = self._args(gdn_state_set_ladder=spec)
            with self.subTest(spec=spec), self.assertRaisesRegex(ValueError, message):
                args._handle_gdn_state_set_ladder()

    def test_invalid_hysteresis_fails_the_boot(self):
        args = self._args(gdn_state_set_ladder="4,1", gdn_state_set_ladder_hysteresis=0)
        with self.assertRaisesRegex(ValueError, "hysteresis"):
            args._handle_gdn_state_set_ladder()

    def test_cli_roundtrip(self):
        import argparse

        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        args = parser.parse_args(
            [
                "--model-path",
                "dummy",
                "--gdn-state-set-ladder",
                "4,2,1",
                "--gdn-state-set-ladder-hysteresis",
                "3",
            ]
        )
        server_args = ServerArgs.from_cli_args(args)
        self.assertEqual(server_args.gdn_state_set_ladder, "4,2,1")
        self.assertEqual(server_args.gdn_state_set_ladder_hysteresis, 3)


if __name__ == "__main__":
    unittest.main()
