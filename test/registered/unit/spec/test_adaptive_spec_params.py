import json
import tempfile
import unittest
from types import SimpleNamespace

from sglang.srt.speculative.adaptive_runtime_state import (
    SpecRuntimeState,
    assert_runtime_state_isolation,
)
from sglang.srt.speculative.adaptive_spec_params import (
    AdaptiveSpeculativeParams,
    AdaptiveStepSlot,
    RungMetrics,
    resolve_candidate_steps_from_config,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_xpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
register_xpu_ci(est_time=10, suite="stage-a-test-1-gpu-xpu")

# Disables the T156 stage-1 anti-flap guards (windowed estimator inertia,
# minimum dwell, noise-scaled deadzone) so the threshold/hysteresis tests
# below keep pinning that logic in isolation. The guards have their own
# test classes (TestStage1AntiFlap, TestRungMetrics).
_STAGE1_GUARDS_OFF = {
    "window_size": 1,
    "min_dwell_rounds": 0,
    "deadzone_sigma": 0.0,
}


class TestAdaptiveStepSlot(unittest.TestCase):
    def _make_params_from_config(self, initial_steps: int, config: dict):
        return AdaptiveStepSlot(
            initial_steps=initial_steps, cfg={**_STAGE1_GUARDS_OFF, **config}
        )

    def test_initial_steps_snaps_to_middle_when_missing(self):
        params = self._make_params_from_config(2, {"candidate_steps": [1, 3, 7]})

        self.assertEqual(params.candidate_steps, [1, 3, 7])
        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.ema_accept_len, 2.0)

    def test_update_respects_warmup_and_interval(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 1,
                "update_interval": 2,
            },
        )

        self.assertFalse(params.update([0, 0]))
        self.assertEqual(params.current_steps, 3)

        self.assertFalse(params.update([0, 0]))
        self.assertEqual(params.current_steps, 3)

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 1)

    def test_empty_batches_do_not_consume_warmup_or_shift_steps(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 1,
                "update_interval": 1,
            },
        )

        self.assertFalse(params.update([]))
        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.ema_accept_len, 2.0)

        self.assertFalse(params.update([0, 0]))
        self.assertEqual(params.current_steps, 3)

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 1)

    def test_update_scales_up_across_candidates(self):
        params = self._make_params_from_config(
            1,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "up_hysteresis": 0.0,
            },
        )

        self.assertTrue(params.update([1, 1]))
        self.assertEqual(params.current_steps, 3)

        self.assertTrue(params.update([3, 3]))
        self.assertEqual(params.current_steps, 7)

    def test_update_can_scale_down_across_candidates_in_one_recompute(self):
        params = self._make_params_from_config(
            7,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
            },
        )

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 1)

    def test_exact_rise_threshold_does_not_upshift(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "up_hysteresis": 0.0,
            },
        )

        self.assertFalse(params.update([2, 3]))
        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.ema_accept_len, 2.5)

        self.assertTrue(params.update([3, 3]))
        self.assertEqual(params.current_steps, 7)

    def test_exact_drop_threshold_does_downshift(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "down_hysteresis": 0.0,
                "up_hysteresis": 0.5,
            },
        )

        self.assertTrue(params.update([0, 1]))
        self.assertEqual(params.current_steps, 1)
        self.assertEqual(params.ema_accept_len, 0.5)

    def test_hysteresis_can_prevent_premature_upshift(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "up_hysteresis": 0.75,
            },
        )

        self.assertFalse(params.update([3, 3]))
        self.assertEqual(params.current_steps, 3)

        self.assertTrue(params.update([4, 4]))
        self.assertEqual(params.current_steps, 7)

    def test_down_hysteresis_can_prevent_premature_downshift(self):
        params = self._make_params_from_config(
            7,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "down_hysteresis": -0.75,
            },
        )

        self.assertFalse(params.update([2, 2]))
        self.assertEqual(params.current_steps, 7)

        self.assertTrue(params.update([1, 1]))
        self.assertEqual(params.current_steps, 3)

    def test_multi_batch_sequence_can_ramp_up_then_back_down(self):
        # window_size=2 shows the windowed estimator's inertia: a single bad
        # round after a good one averages to the middle, so descent from 7 is
        # gradual (one rung per decision), mirroring the old EMA behavior.
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 0.5,
                "warmup_batches": 0,
                "update_interval": 1,
                "up_hysteresis": 0.0,
                "down_hysteresis": 0.0,
                "window_size": 2,
            },
        )

        self.assertTrue(params.update([4, 4]))
        self.assertEqual(params.current_steps, 7)
        # ema_accept_len is the observability gauge; it keeps EMA smoothing.
        self.assertEqual(params.ema_accept_len, 3.0)

        # Window was cleared on the switch: this decision sees only [4.0]
        # (post-switch round), so 7 holds.
        self.assertFalse(params.update([4, 4]))
        self.assertEqual(params.current_steps, 7)

        # Window [4.0, 0.0] -> mean 2.0: down exactly one rung, not to floor.
        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 3)

        # Fresh window [0.0] after the switch -> mean 0.0: down to the floor.
        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 1)

    def test_zero_step_mixed_slot_drops_probes_and_rechecks(self):
        params = self._make_params_from_config(
            3,
            {
                "candidate_steps": [0, 3],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "down_hysteresis": 0.0,
            },
        )

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 0)
        self.assertEqual(params.ema_accept_len, 0.0)

        self.assertTrue(params.update([3, 3]))
        self.assertEqual(params.current_steps, 3)
        self.assertEqual(params.ema_accept_len, 0.0)

        self.assertTrue(params.update([0, 0]))
        self.assertEqual(params.current_steps, 0)
        self.assertEqual(params.ema_accept_len, 0.0)

    def test_ceiling_coeff_caps_steps(self):
        params = self._make_params_from_config(
            7,
            {
                "candidate_steps": [1, 3, 7],
                "ema_alpha": 1.0,
                "warmup_batches": 0,
                "update_interval": 1,
                "ceiling_coeff": 1.0,
            },
        )
        # Force low ema to trigger ceiling
        params.ema_accept_len = 1.0
        self.assertTrue(params.update([1, 1]))
        # ceiling = ceil(1.0 * 1.0) = 1, target capped to 1
        self.assertEqual(params.current_steps, 1)

    def test_ceiling_disabled_by_default(self):
        params = self._make_params_from_config(3, {"candidate_steps": [1, 3, 7]})
        self.assertEqual(params.ceiling_coeff, 0)


class TestAdaptiveSpeculativeParams(unittest.TestCase):
    def test_default_config_loads(self):
        params = AdaptiveSpeculativeParams(initial_steps=3)
        self.assertEqual(params._bs_list, [1, 8, 32, 64])
        self.assertEqual(params._slots[1].candidate_steps, [1, 2, 3])
        self.assertEqual(params._slots[8].candidate_steps, [0, 1, 3])
        self.assertEqual(params._slots[32].candidate_steps, [0, 1])
        self.assertEqual(params._slots[64].candidate_steps, [0])

    def test_config_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "1": {"candidate_steps": [1, 5], "up_hysteresis": 0.3},
                    "32": {"candidate_steps": [1, 2]},
                },
                f,
            )
            f.flush()
            params = AdaptiveSpeculativeParams(initial_steps=5, cfg_path=f.name)
        self.assertEqual(params._bs_list, [1, 32])
        # Slots are built straight from the config; the launch flag never pollutes
        # them. initial_steps just selects the smallest slot's starting step.
        self.assertEqual(params._slots[1].candidate_steps, [1, 5])
        self.assertEqual(params._slots[1].current_steps, 5)
        self.assertEqual(params._slots[1].up_hysteresis, 0.3)
        self.assertEqual(params._slots[32].candidate_steps, [1, 2])

    def test_launch_flag_not_injected_into_slots(self):
        # initial_steps lives only in a larger slot. It must NOT be merged into
        # any other slot's candidates: slots come straight from the config.
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "1": {"candidate_steps": [1, 5]},
                    "8": {"candidate_steps": [1, 3, 7]},
                },
                f,
            )
            f.flush()
            params = AdaptiveSpeculativeParams(initial_steps=7, cfg_path=f.name)
        self.assertEqual(params._slots[1].candidate_steps, [1, 5])
        self.assertEqual(params._slots[8].candidate_steps, [1, 3, 7])
        # The slot that does not own initial_steps starts at its own median.
        self.assertEqual(params._slots[1].current_steps, 5)
        self.assertEqual(params._slots[8].current_steps, 7)

    def test_invalid_config_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump({"not_a_bs": "bad"}, f)
            f.flush()
            with self.assertRaises(ValueError):
                AdaptiveSpeculativeParams(initial_steps=3, cfg_path=f.name)

    def test_invalid_steps_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump({"1": {"candidate_steps": "bad"}}, f)
            f.flush()
            with self.assertRaises(ValueError):
                AdaptiveSpeculativeParams(initial_steps=3, cfg_path=f.name)

    def test_empty_steps_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump({"1": {"candidate_steps": []}}, f)
            f.flush()
            with self.assertRaises(ValueError):
                AdaptiveSpeculativeParams(initial_steps=3, cfg_path=f.name)

    def test_global_hysteresis_inherited(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "up_hysteresis": 0.5,
                    "1": {"candidate_steps": [1, 3]},
                },
                f,
            )
            f.flush()
            params = AdaptiveSpeculativeParams(initial_steps=3, cfg_path=f.name)
        self.assertEqual(params._slots[1].up_hysteresis, 0.5)

    def test_entry_hysteresis_overrides_global(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "up_hysteresis": 0.5,
                    "1": {"candidate_steps": [1, 3], "up_hysteresis": 0.1},
                },
                f,
            )
            f.flush()
            params = AdaptiveSpeculativeParams(initial_steps=3, cfg_path=f.name)
        self.assertEqual(params._slots[1].up_hysteresis, 0.1)


class TestBatchSizeRouting(unittest.TestCase):
    """BS-aware routing: batch size selects the slot, CUDA-graph BS pads first."""

    def _params(self):
        # Default slots: bs=1 -> [1,2,3], bs=8 -> [0,1,3], bs=32 -> [0,1],
        # bs=64 -> [0].
        return AdaptiveSpeculativeParams(initial_steps=3)

    def test_routes_to_floor_slot_without_cuda_graph(self):
        params = self._params()
        # A batch maps to the largest slot BS <= batch (floor), capped at the top slot.
        self.assertEqual(params._route(1).candidate_steps, [1, 2, 3])
        self.assertEqual(params._route(7).candidate_steps, [1, 2, 3])
        self.assertEqual(params._route(8).candidate_steps, [0, 1, 3])
        self.assertEqual(params._route(31).candidate_steps, [0, 1, 3])
        self.assertEqual(params._route(32).candidate_steps, [0, 1])
        self.assertEqual(params._route(1000).candidate_steps, [0])

    def test_cuda_graph_bs_pads_batch_up_before_routing(self):
        params = self._params()
        params.set_cuda_graph_bs([4, 8, 16, 32])
        # bs=5 pads up to the captured graph BS 8 -> slot bs=8.
        self.assertEqual(params._route(5).candidate_steps, [0, 1, 3])
        # bs=17 pads up to 32 -> slot bs=32.
        self.assertEqual(params._route(17).candidate_steps, [0, 1])
        # A batch larger than every captured BS keeps its own value -> top slot.
        self.assertEqual(params._route(100).candidate_steps, [0])

    def test_cuda_graph_bs_for_step_prunes_unreachable_graphs(self):
        params = self._params()
        params.set_cuda_graph_bs([4, 8, 16, 32])
        # step=1 is reachable from every slot.
        self.assertEqual(params.cuda_graph_bs_for_step(1), [4, 8, 16, 32])
        # step=3 lives in the bs=1 and bs=8 slots: graphs 4,8,16 floor into them.
        self.assertEqual(params.cuda_graph_bs_for_step(3), [4, 8, 16])
        # step=2 lives only in the bs=1 slot: only graph BS 4 floors into it.
        self.assertEqual(params.cuda_graph_bs_for_step(2), [4])

    def test_cuda_graph_bs_for_step_returns_none_when_disabled(self):
        params = self._params()
        self.assertIsNone(params.cuda_graph_bs_for_step(3))
        params.set_cuda_graph_bs(None)
        self.assertIsNone(params.cuda_graph_bs_for_step(3))

    def test_observe_verify_feeds_the_routed_slot(self):
        params = self._params()
        # Drive the bs=1 slot up with perfect acceptance; the bs=32 slot is
        # untouched and stays at its own step.
        for _ in range(40):
            params.on_verify_complete([3, 3, 3], batch_size=1)
        self.assertGreater(params.get_steps_for_batch(1), 1)
        # The bs axis is debounced: reaching the bs=32 slot takes bs_debounce
        # (default 3) consecutive decode steps routed there.
        for _ in range(params._bs_debounce):
            steps_32 = params.get_steps_for_batch(32)
        self.assertEqual(steps_32, 1)


class TestBsDebounce(unittest.TestCase):
    """G3: the bs axis must not flap slots on a single boundary crossing."""

    def _params(self, bs_debounce=None):
        cfg = {
            "1": {"candidate_steps": [5]},
            "8": {"candidate_steps": [2]},
        }
        if bs_debounce is not None:
            cfg["bs_debounce"] = bs_debounce
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(cfg, f)
            f.flush()
            return AdaptiveSpeculativeParams(initial_steps=5, cfg_path=f.name)

    def test_default_debounce_value(self):
        self.assertEqual(self._params()._bs_debounce, 3)

    def test_invalid_debounce_raises(self):
        with self.assertRaises(ValueError):
            self._params(bs_debounce=0)

    def test_boundary_flap_does_not_switch_slot(self):
        params = self._params()
        self.assertEqual(params.get_steps_for_batch(1), 5)
        # Alternating 8,1,8,1,... never accumulates 3 consecutive steps in the
        # bs=8 slot, so the active slot stays bs=1 (no graph-swap thrash).
        for _ in range(10):
            self.assertEqual(params.get_steps_for_batch(8), 5)
            self.assertEqual(params.get_steps_for_batch(1), 5)

    def test_consecutive_occupancy_switches(self):
        params = self._params()
        self.assertEqual(params.get_steps_for_batch(1), 5)
        self.assertEqual(params.get_steps_for_batch(8), 5)  # pending 1
        self.assertEqual(params.get_steps_for_batch(8), 5)  # pending 2
        self.assertEqual(params.get_steps_for_batch(8), 2)  # switch on 3rd
        # Switching back is debounced symmetrically.
        self.assertEqual(params.get_steps_for_batch(1), 2)
        self.assertEqual(params.get_steps_for_batch(1), 2)
        self.assertEqual(params.get_steps_for_batch(1), 5)

    def test_return_to_active_slot_resets_pending_run(self):
        params = self._params()
        self.assertEqual(params.get_steps_for_batch(1), 5)
        params.get_steps_for_batch(8)  # pending 1
        params.get_steps_for_batch(8)  # pending 2
        params.get_steps_for_batch(1)  # reset
        self.assertEqual(params.get_steps_for_batch(8), 5)  # pending 1 again
        self.assertEqual(params.get_steps_for_batch(8), 5)  # pending 2
        self.assertEqual(params.get_steps_for_batch(8), 2)  # pending 3: switch

    def test_debounce_one_switches_immediately(self):
        params = self._params(bs_debounce=1)
        self.assertEqual(params.get_steps_for_batch(1), 5)
        self.assertEqual(params.get_steps_for_batch(8), 2)
        self.assertEqual(params.get_steps_for_batch(1), 5)

    def test_verify_feed_reads_active_slot_without_advancing(self):
        params = self._params()
        self.assertEqual(params.get_steps_for_batch(1), 5)
        # Verify feeds at bs=8 must neither advance the debounce nor touch the
        # bs=8 slot's EMA while the bs=1 slot is active.
        ema_8_before = params._slots[8].ema_accept_len
        for _ in range(10):
            params.on_verify_complete([2, 2], batch_size=8)
        self.assertEqual(params._slots[8].ema_accept_len, ema_8_before)
        self.assertEqual(params.get_steps_for_batch(8), 5)  # still pending 1


class TestStage1AntiFlap(unittest.TestCase):
    """T156 stage 1: controller repair against the 2026-07-22 flap incident
    (12 switches in 18 s, median dwell 1.0 s, perfect 3<->2 alternation on
    an input signal that swings wider than the 0.25 hysteresis band)."""

    def _run(self, cfg: dict, seq: list[list[int]]) -> int:
        slot = AdaptiveStepSlot(
            initial_steps=3, cfg={"candidate_steps": [1, 2, 3], **cfg}
        )
        return sum(1 for r in seq if slot.update(r))

    def test_flapping_sequence_is_dwell_limited(self):
        # Synthetic bs=1 regime of the incident: alternating 5-round bursts
        # of full acceptance (code) and full rejection (prose), whose overall
        # mean (1.5) sits exactly on the 2->3 rise threshold.
        rounds = 600
        seq = [[3] if (i // 5) % 2 == 0 else [0] for i in range(rounds)]

        # Guard-free configuration (the pre-stage-1 behavior): every decision
        # point samples one phase or the other and flips -> mass flapping.
        old_switches = self._run(
            {"warmup_batches": 0, "update_interval": 5, **_STAGE1_GUARDS_OFF},
            seq,
        )
        # Stage-1 defaults (window 16, min_dwell_rounds 64, deadzone_sigma 1):
        # the window straddles multiple phases, its mean stays inside the
        # noise-widened deadzone, and any switch is dwell-limited.
        new_switches = self._run(
            {"warmup_batches": 0, "update_interval": 5}, seq
        )

        self.assertGreater(old_switches, 20, "flap reproduction is dead")
        slot_dwell = AdaptiveStepSlot(
            initial_steps=3, cfg={"candidate_steps": [1, 2, 3]}
        ).min_dwell_rounds
        self.assertLessEqual(new_switches, rounds // slot_dwell + 1)
        self.assertLess(new_switches, old_switches / 4)

    def test_min_dwell_blocks_consecutive_switches(self):
        cfg = {
            "candidate_steps": [1, 2, 3],
            "warmup_batches": 0,
            "update_interval": 1,
            "window_size": 1,
            "deadzone_sigma": 0.0,
            "min_dwell_rounds": 10,
        }
        slot = AdaptiveStepSlot(initial_steps=3, cfg=cfg)
        switch_rounds = []
        for i in range(100):
            # Alternates faster than the dwell: would switch on nearly every
            # update without the guard.
            if slot.update([3] if (i // 2) % 2 else [0]):
                switch_rounds.append(i)
        self.assertGreaterEqual(len(switch_rounds), 2)
        gaps = [b - a for a, b in zip(switch_rounds, switch_rounds[1:])]
        self.assertTrue(all(g >= 10 for g in gaps), gaps)

    def test_noise_scaled_deadzone_blocks_noisy_threshold_straddle(self):
        cfg = {
            "candidate_steps": [2, 3],
            "warmup_batches": 0,
            "update_interval": 1,
            "window_size": 8,
            "min_dwell_rounds": 0,
            "deadzone_sigma": 2.0,
            "down_hysteresis": 0.0,
        }
        slot = AdaptiveStepSlot(initial_steps=3, cfg=cfg)
        # Noisy accepts around the 3->2 drop threshold (1.5): the window mean
        # stays ~1.5 but its spread widens the deadzone -> no switch, ever.
        for i in range(200):
            self.assertFalse(slot.update([0] if i % 2 else [3]))
        self.assertEqual(slot.current_steps, 3)
        # A stable low regime (spread -> 0) still drops promptly.
        switched_after = None
        for i in range(20):
            if slot.update([1]):
                switched_after = i
                break
        self.assertIsNotNone(switched_after)
        self.assertEqual(slot.current_steps, 2)

    def test_window_reset_on_switch_requires_fresh_data(self):
        cfg = {
            "candidate_steps": [1, 3],
            "warmup_batches": 0,
            "update_interval": 1,
            "window_size": 8,
            "min_dwell_rounds": 0,
            "deadzone_sigma": 0.0,
        }
        slot = AdaptiveStepSlot(initial_steps=1, cfg=cfg)
        switch_round = None
        for i in range(4):
            if slot.update([3]):
                switch_round = i
        # Decisions wait for the window to half-fill (8 // 2 = 4 rounds).
        self.assertEqual(switch_round, 3)
        self.assertEqual(slot.current_steps, 3)
        # The switch cleared the window: dropping back needs 4 FRESH rounds
        # measured at the new rung, even though the signal is unambiguous.
        results = [slot.update([0]) for _ in range(4)]
        self.assertEqual(results, [False, False, False, True])
        self.assertEqual(slot.current_steps, 1)


class TestRungMetrics(unittest.TestCase):
    """Stage-4 preparation: per-rung reward measurement (observability-only
    in stage 1; wall-clock term must never feed the rank-replicated step
    decision — see the RungMetrics determinism warning)."""

    def test_accept_and_duration_ema(self):
        m = RungMetrics()
        m.observe(3, [2, 3], batch_size=1, now=10.0)
        snap = m.snapshot()
        self.assertEqual(snap[3]["accept_len_ema"], 3.5)  # mean 2.5 + bonus
        self.assertEqual(snap[3]["time_samples"], 0)  # no previous round yet
        m.observe(3, [3, 3], batch_size=1, now=10.05)
        snap = m.snapshot()
        self.assertEqual(snap[3]["time_samples"], 1)
        self.assertAlmostEqual(snap[3]["round_s_ema"], 0.05, places=5)

    def test_idle_gaps_and_bs_changes_do_not_pollute_duration(self):
        m = RungMetrics()
        m.observe(3, [1], batch_size=1, now=10.0)
        m.observe(3, [1], batch_size=2, now=10.05)  # bs changed: not comparable
        m.observe(3, [1], batch_size=2, now=15.0)  # idle gap: not round cost
        self.assertEqual(m.snapshot()[3]["time_samples"], 0)
        self.assertEqual(m.snapshot()[3]["accept_samples"], 3)

    def test_attribution_by_producing_rung(self):
        # Delayed overlap results must land on the rung that produced them,
        # not the currently active one — hence separate per-steps buckets.
        m = RungMetrics()
        m.observe(2, [2], batch_size=1, now=1.0)
        m.observe(3, [3], batch_size=1, now=1.02)
        snap = m.snapshot()
        self.assertEqual(snap[2]["accept_samples"], 1)
        self.assertEqual(snap[3]["accept_samples"], 1)


def _fake_ws(ptr: int):
    return SimpleNamespace(data_ptr=lambda: ptr)


def _fake_backend(ws_ptr: int | None = None, **extra):
    # init_forward_metadata marks this as a real attention backend for the
    # wrapper-unwrap duck-type check in _iter_state_backends.
    ns = SimpleNamespace(init_forward_metadata=lambda *a, **k: None, **extra)
    if ws_ptr is not None:
        ns.workspace_buffer = _fake_ws(ws_ptr)
    return ns


def _make_state(steps, draft=None, target=None, draft_extend=None):
    return SpecRuntimeState(
        speculative_num_steps=steps,
        speculative_num_draft_tokens=steps + 1,
        draft_attn_backend=draft,
        cuda_graph_runner=None,
        target_attn_backend=target,
        target_graph_runner=None,
        draft_extend_attn_backend=draft_extend,
        cuda_graph_runner_for_draft_extend=None,
    )


class TestRuntimeStateIsolation(unittest.TestCase):
    """G4: the M16/#50 invariant — no backend / flashinfer workspace may be
    reachable from two different adaptive runtime states."""

    def test_distinct_states_pass(self):
        states = {
            1: _make_state(1, draft=_fake_backend(0x100), target=_fake_backend(0x200)),
            3: _make_state(3, draft=_fake_backend(0x300), target=_fake_backend(0x400)),
        }
        assert_runtime_state_isolation(states)  # must not raise

    def test_shared_backend_object_across_states_raises(self):
        shared = _fake_backend(0x100)
        states = {
            1: _make_state(1, draft=shared, target=_fake_backend(0x200)),
            3: _make_state(3, draft=shared, target=_fake_backend(0x400)),
        }
        with self.assertRaisesRegex(RuntimeError, "backend shared between states"):
            assert_runtime_state_isolation(states)

    def test_shared_workspace_across_states_raises(self):
        # Distinct backend objects, but their flashinfer workspaces alias the
        # same allocation (e.g. both fell back to the process-global buffer).
        states = {
            1: _make_state(1, draft=_fake_backend(0x100), target=_fake_backend(0x200)),
            3: _make_state(3, draft=_fake_backend(0x100), target=_fake_backend(0x400)),
        }
        with self.assertRaisesRegex(RuntimeError, "workspace shared between states"):
            assert_runtime_state_isolation(states)

    def test_sharing_within_one_state_is_allowed(self):
        # The registered static/initial state legitimately aliases the global
        # workspace across its own draft/target/draft-extend backends.
        states = {
            1: _make_state(
                1,
                draft=_fake_backend(0x100),
                target=_fake_backend(0x100),
                draft_extend=_fake_backend(0x100),
            ),
            3: _make_state(3, draft=_fake_backend(0x300), target=_fake_backend(0x400)),
        }
        assert_runtime_state_isolation(states)  # must not raise

    def test_hybrid_wrapper_leaves_are_checked(self):
        # A leaf backend hidden inside a hybrid prefill/decode wrapper of one
        # state must not alias another state's workspace.
        leaf = _fake_backend(0x500)
        hybrid = SimpleNamespace(
            init_forward_metadata=lambda *a, **k: None,
            prefill_backend=leaf,
            decode_backend=_fake_backend(0x600),
        )
        states = {
            1: _make_state(1, draft=hybrid, target=_fake_backend(0x700)),
            3: _make_state(3, draft=_fake_backend(0x500), target=_fake_backend(0x800)),
        }
        with self.assertRaisesRegex(RuntimeError, "workspace shared between states"):
            assert_runtime_state_isolation(states)

    def test_single_state_never_raises(self):
        shared = _fake_backend(0x100)
        assert_runtime_state_isolation(
            {3: _make_state(3, draft=shared, target=shared)}
        )

    def test_global_workspace_alias_with_baseline_is_allowed(self):
        # Real-world case (found on GPU boot): the EAGLE draft-extend backend
        # is constructed without init_new_workspace, so every built state's
        # draft-extend aliases the PROCESS-GLOBAL workspace — the same buffer
        # the registered static state uses. With baseline_steps naming the
        # registered state, this is logged, not fatal.
        g = 0x1000  # global workspace ptr
        states = {
            3: _make_state(  # registered static state (baseline)
                3,
                draft=_fake_backend(g),
                target=_fake_backend(g),
                draft_extend=_fake_backend(g),
            ),
            1: _make_state(
                1,
                draft=_fake_backend(0x200),
                target=_fake_backend(0x300),
                draft_extend=_fake_backend(g),
            ),
            2: _make_state(
                2,
                draft=_fake_backend(0x400),
                target=_fake_backend(0x500),
                draft_extend=_fake_backend(g),
            ),
        }
        assert_runtime_state_isolation(states, baseline_steps={3})  # no raise

    def test_private_share_still_raises_with_baseline(self):
        # A pointer shared between two built states that is NOT part of the
        # baseline (global) family stays a hard failure.
        states = {
            3: _make_state(3, draft=_fake_backend(0x1000), target=_fake_backend(0x1000)),
            1: _make_state(1, draft=_fake_backend(0x200), target=_fake_backend(0x300)),
            2: _make_state(2, draft=_fake_backend(0x200), target=_fake_backend(0x500)),
        }
        with self.assertRaisesRegex(RuntimeError, "workspace shared between states"):
            assert_runtime_state_isolation(states, baseline_steps={3})

    def test_string_backend_selector_attrs_are_not_unwrapped(self):
        # Regression (found on GPU boot): FlashInferAttnBackend stores plain
        # STRING kernel selectors in .prefill_backend/.decode_backend ("fa2").
        # Interned strings alias across states — must not be treated as
        # shared wrapped backends.
        s1 = _fake_backend(0x100, prefill_backend="fa2", decode_backend="fa2")
        s2 = _fake_backend(0x200, prefill_backend="fa2", decode_backend="fa2")
        assert_runtime_state_isolation(
            {
                1: _make_state(1, draft=s1, target=_fake_backend(0x300)),
                2: _make_state(2, draft=s2, target=_fake_backend(0x400)),
            }
        )  # must not raise

    def test_baseline_alias_without_baseline_arg_still_raises(self):
        # Default (no baseline_steps) keeps the strict behavior.
        states = {
            3: _make_state(3, draft=_fake_backend(0x1000)),
            1: _make_state(1, draft=_fake_backend(0x1000)),
        }
        with self.assertRaisesRegex(RuntimeError, "workspace shared between states"):
            assert_runtime_state_isolation(states)


class TestResolveCandidateSteps(unittest.TestCase):
    def test_default_config(self):
        steps = resolve_candidate_steps_from_config()
        # Generic default union: step 0 (nospec, bs>=8 slots) plus the [1,2,3]
        # ladder; the k=7 tier was removed (net-negative at accept p<=0.8).
        self.assertEqual(steps, [0, 1, 2, 3])

    def test_config_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump({"1": {"candidate_steps": [2, 4]}}, f)
            f.flush()
            steps = resolve_candidate_steps_from_config(cfg_path=f.name)
        self.assertEqual(steps, [2, 4])

    def test_unions_and_dedups_across_slots(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "1": {"candidate_steps": [1, 5]},
                    "8": {"candidate_steps": [3, 5, 7]},
                },
                f,
            )
            f.flush()
            steps = resolve_candidate_steps_from_config(cfg_path=f.name)
        self.assertEqual(steps, [1, 3, 5, 7])


if __name__ == "__main__":
    unittest.main()
