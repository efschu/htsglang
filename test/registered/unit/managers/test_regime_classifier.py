"""#363 phase 1 -- regime classification, hysteresis, dwell, stage admission.

Hermetic: no GPU, no CUDA, no torch, no scheduler. Everything here runs on
synthetic per-round traces, which is the point -- the two failure modes this
controller can have (it thrashes, or it acts on a number inside its own noise)
are both reachable at a desk, and a controller that only becomes falsifiable
once it is inside the scheduler loop is one that gets debugged on a live
server.

Two of the design's falsifiers live here in full:

* **F0 -- the interlock refusal.** A stage flip whose pool cannot hold the
  current working set must be refused, with the arithmetic in the message.
  Driven with the measured #354 numbers: the FP8 prefill arm funds 96 256
  tokens against the decode arm's 453 632, so a 100 000-token working set does
  not fit the stage a prefill-heavy regime would ask for.
* **F1 -- oscillation under adversarial alternation.** A trace that alternates
  across the hysteresis boundary at the worst frequency for each mechanism,
  with the flip count bounded by the dwell.

The remaining falsifiers (F2 self-conditioning replay, F3 per-signal A-vs-A,
F4 do-nothing baseline) need a card and belong to phase 2; F3's band
arithmetic is exercised here because the controller and the experiment must
compute it with the same code.
"""

import math
import unittest

from sglang.srt.managers.regime_classifier import (
    DEFAULT_ENTER_PREFILL,
    DWELL_AMORTIZATION,
    KV_ASCEND_MARK,
    KV_DESCEND_MARK,
    REGIME_DECODE_HEAVY,
    REGIME_KV_PRESSURE,
    REGIME_MIXED,
    REGIME_PREFILL_HEAVY,
    REGIMES,
    DwellGate,
    RegimeError,
    RegimeSample,
    RegimeSensor,
    Stage,
    StageTable,
    classify_sample,
    clears_band,
    min_dwell_rounds,
    pack_proposal,
    signal_band,
    unpack_reduced,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# The measured stage table (#354, four boots, 27B point, TP=3 on 5090+2x3080).
# docs/rig-runbook.md section 4.1.0.
# ---------------------------------------------------------------------------

#: Benchmark harness noise floor, planner/key_solver.py NOISE_FLOOR_PCT.
_HARNESS_FLOOR_PCT = 4.2

FP8_DECODE = Stage(
    name="fp8-decode",
    regime=REGIME_DECODE_HEAVY,
    weight_vector=None,
    kv_token_vector=(7, 3, 3),
    vram_budget_mib=(29607, 17780, 17780),
    max_total_num_tokens=453_632,
    measured_gain_pct=0.0,
    measured_band_pct=_HARNESS_FLOOR_PCT,
    flip_cost_s=0.3,
)
FP8_PREFILL = Stage(
    name="fp8-prefill",
    regime=REGIME_PREFILL_HEAVY,
    weight_vector=(16, 1, 1),
    kv_token_vector=(2, 11, 10),
    vram_budget_mib=(28107, 17780, 17780),
    max_total_num_tokens=96_256,
    measured_gain_pct=22.6,
    measured_band_pct=_HARNESS_FLOOR_PCT,
    flip_cost_s=4.5,
)


def _table():
    return StageTable([FP8_DECODE, FP8_PREFILL], reference="fp8-decode")


def _sample(**kw):
    base = dict(
        round_index=0,
        prefill_rounds=0,
        decode_rounds=64,
        held_tokens=0,
        capacity_tokens=453_632,
    )
    base.update(kw)
    return RegimeSample(**base)


# ---------------------------------------------------------------------------
# 1. The sample: absence is absence
# ---------------------------------------------------------------------------


class TestSample(CustomTestCase):
    def test_an_idle_window_has_no_share_rather_than_a_zero_share(self):
        """A window with no forwards is not 0 % prefill. Returning 0.0 would
        make every idle window read as decode-heavy."""
        s = _sample(prefill_rounds=0, decode_rounds=0)
        self.assertIsNone(s.prefill_share)
        self.assertIsNone(s.decode_share)

    def test_no_capacity_means_no_occupancy(self):
        self.assertIsNone(_sample(capacity_tokens=0).occupancy)

    def test_shares_and_occupancy_are_the_plain_ratios(self):
        s = _sample(prefill_rounds=16, decode_rounds=48, held_tokens=113_408)
        self.assertAlmostEqual(s.prefill_share, 0.25)
        self.assertAlmostEqual(s.decode_share, 0.75)
        self.assertAlmostEqual(s.occupancy, 0.25)

    def test_negative_counters_are_refused_at_construction(self):
        with self.assertRaises(RegimeError):
            _sample(held_tokens=-1)
        with self.assertRaises(RegimeError):
            RegimeSample(
                round_index=0,
                prefill_rounds=0,
                decode_rounds=1,
                held_tokens=0,
                capacity_tokens=1,
                rank_ms_spread_pct=-3.0,
            )


# ---------------------------------------------------------------------------
# 2. Classification: tier-R only, fixed precedence
# ---------------------------------------------------------------------------


class TestClassify(CustomTestCase):
    def test_kv_pressure_outranks_everything(self):
        """The constraint regime, not a fourth peer: a pool at the ascend mark
        is the answer even in the middle of a prefill burst."""
        s = _sample(
            prefill_rounds=64,
            decode_rounds=0,
            held_tokens=int(0.90 * 453_632),
            queued_prompt_tokens=1_000_000,
        )
        self.assertEqual(classify_sample(s, burst_tokens=8192), REGIME_KV_PRESSURE)

    def test_prefill_share_over_the_mark_is_prefill_heavy(self):
        s = _sample(prefill_rounds=32, decode_rounds=32)
        self.assertEqual(classify_sample(s), REGIME_PREFILL_HEAVY)

    def test_a_queued_burst_is_prefill_heavy_before_a_single_prefill_ran(self):
        """The predictive half: every actuator commits at a group-idle
        boundary, so a controller that waits for the prefill to show up in the
        window has already lost the window."""
        s = _sample(
            prefill_rounds=0,
            decode_rounds=64,
            queued_reqs=1,
            queued_prompt_tokens=32_768,
        )
        # Without a burst threshold the queue is invisible to the classifier:
        # decode dominates the window, but the queue is not empty, so the
        # honest answer is MIXED rather than DECODE_HEAVY.
        self.assertEqual(classify_sample(s), REGIME_MIXED)
        self.assertEqual(classify_sample(s, burst_tokens=8192), REGIME_PREFILL_HEAVY)

    def test_decode_heavy_needs_an_empty_queue(self):
        s = _sample(prefill_rounds=0, decode_rounds=64, queued_reqs=3)
        self.assertEqual(classify_sample(s), REGIME_MIXED)
        self.assertEqual(
            classify_sample(_sample(prefill_rounds=0, decode_rounds=64)),
            REGIME_DECODE_HEAVY,
        )

    def test_the_middle_is_mixed_and_that_is_an_answer(self):
        s = _sample(prefill_rounds=16, decode_rounds=48)  # 25 % prefill, 75 % decode
        self.assertEqual(classify_sample(s), REGIME_MIXED)

    def test_classification_reads_no_rank_local_field(self):
        """Tier-L must not reach a branch. Two samples differing only in the
        reduced rank spread must classify identically -- if they ever do not,
        a rank-local value has become a classifier input."""
        common = dict(prefill_rounds=32, decode_rounds=32)
        a = _sample(rank_ms_spread_pct=0.0, **common)
        b = _sample(rank_ms_spread_pct=95.0, **common)
        self.assertEqual(classify_sample(a), classify_sample(b))


# ---------------------------------------------------------------------------
# 3. Hysteresis: enforced, not recommended
# ---------------------------------------------------------------------------


class TestHysteresisContract(CustomTestCase):
    def test_equal_entry_and_exit_marks_are_refused(self):
        with self.assertRaises(RegimeError) as cm:
            RegimeSensor(enter_prefill=0.35, exit_prefill=0.35)
        self.assertIn("hysteresis", str(cm.exception))

    def test_an_exit_window_no_longer_than_the_entry_window_is_refused(self):
        with self.assertRaises(RegimeError) as cm:
            RegimeSensor(enter_window=4, exit_window=4)
        self.assertIn("asymmetry is the contract", str(cm.exception))

    def test_inverted_kv_marks_are_refused(self):
        with self.assertRaises(RegimeError):
            RegimeSensor(kv_ascend_mark=0.55, kv_descend_mark=0.85)

    def test_the_inherited_kv_marks_are_the_287_numbers(self):
        """Two independently-chosen thresholds on one physical quantity is how
        two controllers end up disagreeing about the same pool."""
        from sglang.srt.model_executor import kv_pressure_ladder as ladder

        self.assertEqual(KV_ASCEND_MARK, ladder.DEFAULT_ASCEND_THRESHOLD)
        self.assertEqual(KV_DESCEND_MARK, ladder.DEFAULT_DESCEND_THRESHOLD)


class TestHysteresisBehaviour(CustomTestCase):
    def _run(self, sensor, samples):
        return [sensor.observe(s) for s in samples]

    def test_a_single_sample_never_moves_the_state(self):
        sensor = RegimeSensor(enter_window=2, exit_window=4)
        steady = _sample(prefill_rounds=0, decode_rounds=64)
        for _ in range(10):
            sensor.observe(steady)
        self.assertEqual(sensor.regime, REGIME_DECODE_HEAVY)
        sensor.observe(_sample(prefill_rounds=64, decode_rounds=0))
        self.assertEqual(sensor.regime, REGIME_DECODE_HEAVY)

    def test_a_sustained_change_does_move_the_state(self):
        sensor = RegimeSensor(enter_window=2, exit_window=4)
        for _ in range(10):
            sensor.observe(_sample(prefill_rounds=0, decode_rounds=64))
        self.assertEqual(sensor.regime, REGIME_DECODE_HEAVY)
        for _ in range(8):
            sensor.observe(_sample(prefill_rounds=64, decode_rounds=0))
        self.assertEqual(sensor.regime, REGIME_PREFILL_HEAVY)

    def test_a_signal_sitting_exactly_on_the_entry_mark_does_not_toggle(self):
        """The classic hysteresis failure: a value parked on the threshold,
        dithered by one round in each direction."""
        sensor = RegimeSensor(enter_window=2, exit_window=4)
        on_mark = int(round(DEFAULT_ENTER_PREFILL * 64))  # 22 of 64
        transitions_before = sensor.transitions
        for i in range(200):
            p = on_mark + (1 if i % 2 else -1)
            sensor.observe(_sample(prefill_rounds=p, decode_rounds=64 - p))
        # It may settle into one regime; it may not move at all. What it may
        # not do is track the dither.
        self.assertLessEqual(sensor.transitions - transitions_before, 2)


# ---------------------------------------------------------------------------
# 4. F1 -- oscillation under adversarial alternation
# ---------------------------------------------------------------------------


class TestF1Oscillation(CustomTestCase):
    """FALSIFIER F1. A controller that thrashes a live server fails here.

    Two adversarial traces, one per mechanism:

    * dither at the entry threshold (hysteresis' job),
    * a genuine square wave whose half-period is below the dwell (dwell's
      job -- the signal is clean, and the move is still unaffordable).
    """

    @staticmethod
    def _drive(sensor, gate, samples):
        """Run the observe/select/dwell loop.

        Returns ``(flips, regime_history, commits)``. The two timelines are
        deliberately separate: the classifier reports the regime TRUTHFULLY
        every round, and the gate decides what was affordable. A regime that
        alternates faster than the dwell is a fact about the load and belongs
        in the observe-only log; what must not alternate is the SERVER's
        configuration.
        """
        table = _table()
        current = "fp8-decode"
        flips = 0
        history = []
        commits = []  # (round_index, stage_name)
        for i, s in enumerate(samples):
            regime = sensor.observe(s)
            history.append(regime)
            target, _why = table.select(regime, s, current=current)
            if target is None:
                continue
            ok, _why = gate.allows(i)
            if not ok:
                continue
            gate.record_flip(i)
            current = target.name
            commits.append((i, current))
            flips += 1
        return flips, history, commits

    def test_dither_at_the_boundary_commits_almost_no_flips(self):
        sensor = RegimeSensor(enter_window=2, exit_window=4)
        gate = DwellGate(min_rounds=64)
        on_mark = int(round(DEFAULT_ENTER_PREFILL * 64))
        samples = [
            _sample(
                round_index=i,
                prefill_rounds=on_mark + (1 if i % 2 else -1),
                decode_rounds=64 - (on_mark + (1 if i % 2 else -1)),
            )
            for i in range(512)
        ]
        flips, _history, _commits = self._drive(sensor, gate, samples)
        self.assertLessEqual(flips, 2, f"dither committed {flips} flips")

    def test_a_square_wave_faster_than_the_dwell_is_bounded_by_the_dwell(self):
        """The signal is clean -- full prefill, then full decode -- so
        hysteresis has no objection. The dwell must still bound the flips: a
        workload that alternates faster than a flip amortizes is a workload
        that alternates, not a regime that changed."""
        min_rounds = 64
        sensor = RegimeSensor(enter_window=2, exit_window=4)
        gate = DwellGate(min_rounds=min_rounds)
        rounds = 1024
        half = 16  # well under the dwell
        samples = []
        for i in range(rounds):
            prefill = (i // half) % 2 == 0
            samples.append(
                _sample(
                    round_index=i,
                    prefill_rounds=64 if prefill else 0,
                    decode_rounds=0 if prefill else 64,
                )
            )
        flips, _history, _commits = self._drive(sensor, gate, samples)
        bound = rounds // min_rounds
        self.assertLessEqual(flips, bound, f"{flips} flips against bound {bound}")
        self.assertGreater(gate.refusals, 0, "the dwell gate never bound")

    def test_no_stage_is_returned_to_within_one_dwell_interval(self):
        """The A -> B -> A signature of a controller chasing itself.

        The check is on the SERVER's configuration, not on the classification.
        The regime is allowed -- required, in observe-only mode -- to track a
        load that alternates every 24 rounds; putting the server back on a
        configuration it left 24 rounds ago is the failure.
        """
        sensor = RegimeSensor(enter_window=2, exit_window=4)
        gate = DwellGate(min_rounds=64)
        samples = []
        for i in range(512):
            prefill = (i // 24) % 2 == 0
            samples.append(
                _sample(
                    round_index=i,
                    prefill_rounds=64 if prefill else 0,
                    decode_rounds=0 if prefill else 64,
                )
            )
        _flips, history, commits = self._drive(sensor, gate, samples)
        # The classifier did see the alternation -- otherwise this trace would
        # prove nothing about the gate.
        self.assertGreater(len(set(history)), 1)
        for (round_a, stage_a), (round_b, stage_b) in zip(commits, commits[2:]):
            if stage_a == stage_b:
                self.assertGreaterEqual(
                    round_b - round_a,
                    gate.min_rounds,
                    f"stage {stage_a!r} was returned to after "
                    f"{round_b - round_a} rounds, inside the "
                    f"{gate.min_rounds}-round dwell",
                )


# ---------------------------------------------------------------------------
# 5. Dwell: derived, not typed
# ---------------------------------------------------------------------------


class TestDwell(CustomTestCase):
    def test_the_dwell_is_the_amortization_of_a_measured_cost(self):
        # A 4.5 s weight-class flip on a 40 ms round.
        self.assertEqual(min_dwell_rounds(4.5, 0.040), math.ceil(20 * 4.5 / 0.040))
        # A 0.3 s KV reshard on the same round: two orders cheaper to hold.
        self.assertEqual(min_dwell_rounds(0.3, 0.040), math.ceil(20 * 0.3 / 0.040))
        self.assertGreater(min_dwell_rounds(4.5, 0.040), min_dwell_rounds(0.3, 0.040))

    def test_a_missing_round_time_is_refused_rather_than_substituted(self):
        """Without a measured round time the dwell would be set by whatever
        was substituted for it."""
        with self.assertRaises(RegimeError) as cm:
            min_dwell_rounds(4.5, 0.0)
        self.assertIn("substituted", str(cm.exception))

    def test_the_amortization_factor_is_the_only_judgement_in_the_number(self):
        self.assertEqual(DWELL_AMORTIZATION, 20)
        self.assertEqual(
            min_dwell_rounds(1.0, 0.1, amortization=1),
            10,
        )

    def test_the_gate_permits_the_first_flip_and_then_holds(self):
        gate = DwellGate(min_rounds=100)
        ok, why = gate.allows(0)
        self.assertTrue(ok)
        self.assertIn("no flip yet", why)
        gate.record_flip(0)
        ok, why = gate.allows(50)
        self.assertFalse(ok)
        self.assertIn("50 rounds since the last flip", why)
        self.assertTrue(gate.allows(100)[0])


# ---------------------------------------------------------------------------
# 6. F0 -- the interlock refusal, on the measured numbers
# ---------------------------------------------------------------------------


class TestF0Interlock(CustomTestCase):
    """FALSIFIER F0. A stage is a tuple, and its pool is part of it.

    The #354 measurement: the FP8 prefill arm buys +22.6 % prefill and pays
    -79 % of ``max_total_num_tokens`` (453 632 -> 96 256). A controller that
    moved the weight cut alone, or that moved the tuple without checking the
    pool, would collapse capacity by 4.7x underneath a live working set.
    """

    def test_a_working_set_that_does_not_fit_the_target_pool_is_refused(self):
        held = 100_000  # fits the decode arm at 22 %; does not fit the prefill arm
        ok, why = FP8_DECODE.admits(held)
        self.assertTrue(ok, why)
        ok, why = FP8_PREFILL.admits(held)
        self.assertFalse(ok)
        # The refusal names both numbers, per the design's "fail loudly with
        # the arithmetic" rule.
        self.assertIn("100000", why)
        self.assertIn("96256", why)

    def test_the_admission_ceiling_is_the_pressure_sensor_s_own_mark(self):
        """Admissibility and pressure are the same arithmetic on the same
        sample; a second mark here would let the two disagree about one pool."""
        ceiling = int(KV_ASCEND_MARK * FP8_PREFILL.max_total_num_tokens)
        self.assertTrue(FP8_PREFILL.admits(ceiling)[0])
        self.assertFalse(FP8_PREFILL.admits(ceiling + 1_000)[0])

    def test_select_refuses_the_flip_the_regime_asked_for(self):
        table = _table()
        s = _sample(
            round_index=0,
            prefill_rounds=64,
            decode_rounds=0,
            held_tokens=100_000,
            capacity_tokens=453_632,
        )
        self.assertEqual(classify_sample(s), REGIME_PREFILL_HEAVY)
        target, why = table.select(REGIME_PREFILL_HEAVY, s, current="fp8-decode")
        self.assertIsNone(target)
        self.assertIn("96256", why)

    def test_the_same_flip_is_allowed_once_the_working_set_is_small_enough(self):
        table = _table()
        s = _sample(
            prefill_rounds=64,
            decode_rounds=0,
            held_tokens=40_000,
            capacity_tokens=453_632,
        )
        target, why = table.select(REGIME_PREFILL_HEAVY, s, current="fp8-decode")
        self.assertIsNotNone(target)
        self.assertEqual(target.name, "fp8-prefill")
        self.assertIn("+22.6%", why)


# ---------------------------------------------------------------------------
# 7. The stage table earns its entries (#360)
# ---------------------------------------------------------------------------


class TestStageTable(CustomTestCase):
    def test_a_gain_inside_its_own_band_is_not_a_stage(self):
        """The #354 INT8 prefill arm: +6.1 % from one boot against the 4.2 %
        harness floor, with its decode column explicitly undecided at n=1. A
        candidate, not a stage."""
        int8_prefill = Stage(
            name="int8-prefill",
            regime=REGIME_PREFILL_HEAVY,
            weight_vector=(10, 1, 1),
            kv_token_vector=(2, 11, 10),
            vram_budget_mib=(28107, 17780, 17780),
            max_total_num_tokens=137_664,
            measured_gain_pct=6.1,
            measured_band_pct=6.1,  # its own repeat moved it as much
            flip_cost_s=4.5,
        )
        self.assertFalse(int8_prefill.clears_its_band)
        with self.assertRaises(RegimeError) as cm:
            StageTable([FP8_DECODE, int8_prefill], reference="fp8-decode")
        self.assertIn("int8-prefill", str(cm.exception))
        self.assertIn("band", str(cm.exception))

    def test_the_measured_fp8_pair_is_admitted(self):
        table = _table()
        self.assertEqual(len(table), 2)
        self.assertIs(table.for_regime(REGIME_PREFILL_HEAVY), FP8_PREFILL)
        self.assertIs(table.for_regime(REGIME_DECODE_HEAVY), FP8_DECODE)
        self.assertIsNone(table.for_regime(REGIME_MIXED))

    def test_a_reference_that_is_not_in_the_table_is_refused(self):
        with self.assertRaises(RegimeError):
            StageTable([FP8_DECODE], reference="nope")

    def test_kv_pressure_hands_the_decision_to_the_pressure_arithmetic(self):
        table = _table()
        s = _sample(held_tokens=int(0.9 * 453_632))
        target, why = table.select(REGIME_KV_PRESSURE, s, current="fp8-decode")
        self.assertIsNone(target)
        self.assertIn("outranks", why)

    def test_mixed_selects_nothing_and_says_so(self):
        table = _table()
        target, why = table.select(REGIME_MIXED, _sample(), current="fp8-decode")
        self.assertIsNone(target)
        self.assertIn("no stage is solved", why)

    def test_already_on_the_stage_is_not_a_flip(self):
        table = _table()
        target, why = table.select(REGIME_DECODE_HEAVY, _sample(), current="fp8-decode")
        self.assertIsNone(target)
        self.assertIn("already on stage", why)

    def test_a_stage_with_no_pool_is_refused_at_construction(self):
        with self.assertRaises(RegimeError):
            Stage(
                name="empty",
                regime=REGIME_MIXED,
                weight_vector=None,
                kv_token_vector=(1,),
                vram_budget_mib=(1,),
                max_total_num_tokens=0,
                measured_gain_pct=50.0,
                measured_band_pct=1.0,
                flip_cost_s=0.0,
            )

    def test_an_unknown_regime_name_is_refused_at_construction(self):
        with self.assertRaises(RegimeError):
            Stage(
                name="bogus",
                regime="fast",
                weight_vector=None,
                kv_token_vector=(1,),
                vram_budget_mib=(1,),
                max_total_num_tokens=1,
                measured_gain_pct=50.0,
                measured_band_pct=1.0,
                flip_cost_s=0.0,
            )


# ---------------------------------------------------------------------------
# 8. The #360 A-vs-A band
# ---------------------------------------------------------------------------


class TestSignalBand(CustomTestCase):
    def test_the_band_is_what_the_arm_measures_against_its_own_repeat(self):
        a = [10.0, 11.0, 10.5, 12.0]
        a_repeat = [10.2, 10.6, 10.9, 11.1]
        self.assertAlmostEqual(signal_band(a, a_repeat), 0.9)

    def test_unaligned_windows_are_refused_rather_than_reported_as_noise(self):
        with self.assertRaises(RegimeError) as cm:
            signal_band([1.0, 2.0], [1.0])
        self.assertIn("misalignment", str(cm.exception))

    def test_an_empty_band_is_refused(self):
        with self.assertRaises(RegimeError) as cm:
            signal_band([], [])
        self.assertIn("trivially", str(cm.exception))

    def test_a_threshold_must_clear_its_band_by_a_factor(self):
        band = 2.0
        self.assertFalse(clears_band(3.0, band))  # inside 2x
        self.assertTrue(clears_band(5.0, band))
        self.assertTrue(clears_band(-5.0, band))  # direction does not matter

    def test_the_harness_floor_would_reject_a_six_percent_claim(self):
        """The concrete case from the stage table, as arithmetic."""
        # At the 2x margin the design requires of a THRESHOLD, +6.1 % does
        # not clear a 4.2 % floor and +22.6 % does. That is the whole
        # difference between the two #354 prefill arms.
        self.assertFalse(clears_band(6.1, _HARNESS_FLOOR_PCT))
        self.assertTrue(clears_band(22.6, _HARNESS_FLOOR_PCT))
        # It does clear at 1x, which is why the INT8 arm is a CANDIDATE and
        # not simply noise -- it is one boot short of being a stage.
        self.assertTrue(clears_band(6.1, _HARNESS_FLOOR_PCT, margin=1.0))


# ---------------------------------------------------------------------------
# 9. Routing the rank-local tier through the reduction
# ---------------------------------------------------------------------------


def _min_reduce(payloads):
    """The consensus channel's contract: element-wise MIN across the group."""
    return [min(vals) for vals in zip(*payloads)]


class TestConsensusPayload(CustomTestCase):
    def test_agreeing_ranks_commit(self):
        payloads = [
            pack_proposal(REGIME_PREFILL_HEAVY, 1, 7, 41.0),
            pack_proposal(REGIME_PREFILL_HEAVY, 1, 7, 40.0),
            pack_proposal(REGIME_PREFILL_HEAVY, 1, 7, 42.0),
        ]
        out = unpack_reduced(_min_reduce(payloads))
        self.assertTrue(out["agreed"])
        self.assertEqual(out["regime"], REGIME_PREFILL_HEAVY)
        self.assertEqual(out["target_stage_index"], 1)
        self.assertEqual(out["epoch"], 7)

    def test_the_rank_spread_survives_the_reduction(self):
        """Ranks disagreeing on TIMING is data, not a desync -- that spread is
        the whole reason the tier-L field is carried at all."""
        payloads = [
            pack_proposal(REGIME_PREFILL_HEAVY, 1, 7, 50.0),
            pack_proposal(REGIME_PREFILL_HEAVY, 1, 7, 40.0),
        ]
        out = unpack_reduced(_min_reduce(payloads))
        self.assertTrue(out["agreed"])
        self.assertAlmostEqual(out["rank_ms_spread_pct"], 20.0, places=3)

    def test_ranks_disagreeing_on_the_regime_is_a_named_desync(self):
        payloads = [
            pack_proposal(REGIME_PREFILL_HEAVY, 1, 7, 40.0),
            pack_proposal(REGIME_DECODE_HEAVY, 0, 7, 40.0),
        ]
        out = unpack_reduced(_min_reduce(payloads))
        self.assertFalse(out["agreed"])
        self.assertTrue(
            any("regime_code" in d for d in out["disagreements"]), out["disagreements"]
        )

    def test_one_blind_rank_suppresses_the_spread_rather_than_faking_it(self):
        """A graph-covered forward reports no split at all rather than a wrong
        zero; that absence has to survive the reduction, or a blind rank reads
        as an infinitely fast one."""
        payloads = [
            pack_proposal(REGIME_MIXED, 0, 3, 40.0),
            pack_proposal(REGIME_MIXED, 0, 3, None),
        ]
        out = unpack_reduced(_min_reduce(payloads))
        self.assertTrue(out["agreed"])
        self.assertIsNone(out["rank_ms_spread_pct"])

    def test_a_wrong_length_payload_is_refused(self):
        with self.assertRaises(RegimeError) as cm:
            unpack_reduced([0, 0])
        self.assertIn("element-wise MIN", str(cm.exception))

    def test_an_unknown_regime_cannot_be_packed(self):
        with self.assertRaises(RegimeError):
            pack_proposal("fast", 0, 0, None)

    def test_every_regime_round_trips(self):
        for regime in REGIMES:
            out = unpack_reduced(_min_reduce([pack_proposal(regime, 0, 0, 1.0)]))
            self.assertEqual(out["regime"], regime)


# ---------------------------------------------------------------------------
# 10. Phase-1 scope: nothing here reaches the loop
# ---------------------------------------------------------------------------


class TestPhaseOneScope(CustomTestCase):
    def test_the_module_imports_no_torch_and_no_scheduler(self):
        """Observe-only means the classifier must be desk-importable: a phase-1
        component that drags the runtime in cannot be tested without one."""
        import ast
        import pathlib

        import sglang.srt.managers.regime_classifier as mod

        tree = ast.parse(pathlib.Path(mod.__file__).read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertEqual(imported & {"torch", "sglang"}, set(), sorted(imported))

    def test_no_actuator_is_reachable_from_this_module(self):
        import pathlib

        import sglang.srt.managers.regime_classifier as mod

        src = pathlib.Path(mod.__file__).read_text()
        for forbidden in ("kv_reshard", "vram_dial", ".arm(", "apply_budget_request"):
            self.assertNotIn(forbidden, src, forbidden)


if __name__ == "__main__":
    unittest.main()
