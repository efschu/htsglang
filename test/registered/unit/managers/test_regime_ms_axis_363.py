"""#363 defects 5 and 8 -- the ms/round axis, and the flag/resolved-state split.

Hermetic: no GPU, no server, no CUDA events. Every clock here is driven by
plain floats, because what these gates pin is WIRING and FRESHNESS, not the
device timing itself (that is the CollectiveClock's own suite).

THREE PROPERTIES, and each one is a defect the R14 window handed over.

DEFECT 8a -- THE CLOCK MUST EXIST IN OBSERVE.
    ``build_regime_observer`` built the ms/round clock only in ACT mode, on
    the reasoning that "a clock whose verdict cannot move anything would be
    an expensive observe under a misleading name". That reasoning is right
    about the ADMISSION GATE and wrong about the CLOCK, and the difference
    closes a bootstrap deadlock:

        stage_measure_pass reads OBSERVE traces to build the canon
        -> the canon is what makes a stage a flip target
        -> a flip target is what act mode requires
        -> but the rows the pass reads exist only if the clock ran
        -> and the clock ran only in act mode.

    So the canon could never be bootstrapped on any rig. R14 fixed the INNER
    half of this deadlock (#363 defect 7, the "measurement only" branch of
    ``_intra_phase_decide``) -- a branch that could not execute, because in
    observe the clock it hangs off was never constructed. Measured on R14's
    own B3 trace: 82 549 verdict rows, ``ms_decision`` null on every one,
    and ``"stage_clock": null`` in all three ranks' summary lines.

    The ADMISSION gate stays act-only. Observe gains a measurement
    instrument, not a path to an actuator.

DEFECT 8b -- A SAMPLE MUST BE COUNTED ONCE.
    ``RankPrefillLog`` keeps ``last_gpu_ms``/``last_wait_ms`` at the values of
    the last forward that WAS measurable, and ``last_split_known`` stays True
    from then on. The accessor therefore answers every later boundary with
    the same retired forward, and the observer accumulated it once per
    boundary. On R14's B3 that is 82 341 of 82 549 boundaries carrying a
    number, against 2 574 prefill forwards actually measured in the whole
    boot -- a mean over roughly thirty copies of each sample, which reads
    exactly like a measurement and is not one.

    A stale carry is the right behaviour for a LOG LINE (it is what was last
    seen) and the wrong one for a MEAN. So the split now carries a sequence,
    and a boundary accumulates only a sample it has not already counted.

DEFECT 5 -- THE BOOTED WEIGHT VECTOR MUST COME FROM RESOLVED STATE.
    ``_booted_stage`` read ``server_args.rank_mlp_ratio`` -- the FLAG. Under
    ``--rank-tp-ratio auto-performance`` that flag is None while the server
    runs a concrete resolved partition, so the booted stage reported
    ``weights=None``, every planner candidate carries a concrete vector, and
    ``reachability`` -- which checks weights FIRST -- returned
    REACH_NO_WEIGHT_MOVER for every candidate whatever its KV vector.

    The two sides live in different spaces and must be canonicalised before
    they are compared: a candidate reports a gcd-reduced ratio of MLP UNITS,
    the installed plan is a raw ratio. Comparing them raw trades a false
    mismatch for a false match, which is worse, so the comparison reduces
    both and says so.
"""

import types
import unittest

from sglang.srt.managers.regime_runtime import (
    MODE_OBSERVE,
    RegimeObserver,
    _booted_stage,
    build_regime_observer,
    rank_split_ms_from,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


# --------------------------------------------------------------------------
# fakes


class _Log:
    """Stands in for RankPrefillLog's structured tap."""

    def __init__(self, gpu_ms=None, wait_ms=None, known=False, seq=0):
        self.last_gpu_ms = gpu_ms
        self.last_wait_ms = wait_ms
        self.last_split_known = known
        self.last_split_seq = seq

    def put(self, gpu_ms, wait_ms):
        """One newly flushed, measurable forward."""
        self.last_gpu_ms = gpu_ms
        self.last_wait_ms = wait_ms
        self.last_split_known = True
        self.last_split_seq += 1


def _sched_with_log(log):
    return types.SimpleNamespace(
        metrics_reporter=types.SimpleNamespace(rank_prefill_log=log)
    )


def _server_args(**kw):
    base = dict(
        regime_controller="observe",
        regime_stage_clock=True,
        tp_size=1,
        kv_pressure_consensus_interval=8,
        regime_trace=None,
        regime_gate_evidence=None,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def _scheduler(server_args):
    return types.SimpleNamespace(
        server_args=server_args,
        regime_stage_table=None,
        tp_cpu_group=None,
        tp_rank=0,
        kv_reshard_runtime=None,
        max_total_num_tokens=453632,
    )


def _observer(**kw):
    """An observer with the ms axis wired, driven by plain floats."""
    from sglang.srt.managers.regime_ms_clock import MsStageDecider

    params = dict(
        consensus_interval=1,
        tp_size=1,
        collective_min=None,
        mode=MODE_OBSERVE,
        table=None,
        table_plan=None,
        current_stage=None,
        stage_clock=MsStageDecider(),
    )
    params.update(kw)
    return RegimeObserver(**params)


def _round(obs, **kw):
    params = dict(
        phase="decode",
        held_tokens=100,
        capacity_tokens=1000,
        running_bs=1,
        queued_reqs=0,
        queued_prompt_tokens=0,
        max_queued_prompt_tokens=0,
        rank_forward_ms=10.0,
    )
    params.update(kw)
    return obs.on_round(**params)


# --------------------------------------------------------------------------
# defect 8a -- the clock exists in observe


class TestTheClockIsWiredInObserve(CustomTestCase):
    def test_observe_with_the_flag_builds_the_clock(self):
        """The bootstrap deadlock, in one assertion.

        This is the gate that was red for four windows: an observe boot with
        --regime-stage-clock produced an observer whose _stage_clock was None,
        so _intra_phase_decide was never called and no ms_decision row was
        ever written -- and those rows are the only input the measurement
        pass has.
        """
        obs = build_regime_observer(_scheduler(_server_args()))
        self.assertIsNotNone(obs)
        self.assertIsNotNone(
            obs._stage_clock,
            "observe + --regime-stage-clock must build the ms/round clock: "
            "the measurement pass reads observe traces, so a clock that only "
            "exists in act mode can never produce the canon act requires",
        )

    def test_observe_holds_no_admission_gate(self):
        """The half of the old reasoning that was RIGHT stays enforced.

        The admission gate prices a flip against the corridor. In observe
        there is no flip to price, and the gate is a path toward acting --
        so it stays act-only. Only the measurement instrument crosses over.
        """
        obs = build_regime_observer(_scheduler(_server_args()))
        self.assertIsNone(obs._admission)

    def test_without_the_flag_no_clock_is_built(self):
        """Neutrality: the default boot is unchanged in both modes."""
        obs = build_regime_observer(_scheduler(_server_args(regime_stage_clock=False)))
        self.assertIsNotNone(obs)
        self.assertIsNone(obs._stage_clock)
        self.assertIsNone(obs._admission)

    def test_the_clock_records_a_measurement_row_in_observe(self):
        """Wiring is not enough -- the row has to arrive.

        The end-to-end shape of the fix: an observe boundary with a fresh
        split writes an ms_decision carrying the measurement fields, which is
        exactly what stage_measure_pass reads.
        """
        obs = _observer()
        rec = _round(obs, rank_compute_ms=6.0, rank_wait_ms=4.0, rank_split_seq=1)
        self.assertIsNotNone(rec)
        self.assertIsNotNone(
            rec.get("ms_decision"),
            "an observe boundary with a fresh split must carry an "
            "ms_decision; that record IS the measurement",
        )
        self.assertEqual(rec["ms_decision"]["mean_total_ms"], 10.0)


# --------------------------------------------------------------------------
# defect 8b -- a sample is counted once


class TestSplitFreshness(CustomTestCase):
    def test_the_accessor_reports_the_sample_sequence(self):
        log = _Log()
        self.assertEqual(rank_split_ms_from(_sched_with_log(log)), (None, None, 0))
        log.put(10.0, 4.0)
        self.assertEqual(rank_split_ms_from(_sched_with_log(log)), (6.0, 4.0, 1))

    def test_a_repeated_sample_is_counted_once(self):
        """The defect, stated as arithmetic.

        Two boundaries, one retired forward. The mean must be that forward,
        not that forward averaged with a copy of itself -- and the count the
        mean divides by must be 1.
        """
        obs = _observer(consensus_interval=100)
        _round(obs, rank_compute_ms=6.0, rank_wait_ms=4.0, rank_split_seq=7)
        _round(obs, rank_compute_ms=6.0, rank_wait_ms=4.0, rank_split_seq=7)
        self.assertEqual(obs._ms_split_n, 1)

    def test_a_new_sample_is_counted(self):
        obs = _observer(consensus_interval=100)
        _round(obs, rank_compute_ms=6.0, rank_wait_ms=4.0, rank_split_seq=7)
        _round(obs, rank_compute_ms=8.0, rank_wait_ms=2.0, rank_split_seq=8)
        self.assertEqual(obs._ms_split_n, 2)
        self.assertEqual(obs._ms_compute_sum, 14.0)
        self.assertEqual(obs._ms_wait_sum, 6.0)

    def test_a_boundary_with_no_new_forward_reports_no_split(self):
        """Stale carry must abstain, not answer.

        A boundary that saw no measurable forward has no ms/round split. The
        honest report is the sentinel the group reduction already understands,
        never the last forward's numbers wearing this boundary's timestamp.
        """
        obs = _observer(consensus_interval=1)
        _round(obs, rank_compute_ms=6.0, rank_wait_ms=4.0, rank_split_seq=7)
        rec = _round(obs, rank_compute_ms=6.0, rank_wait_ms=4.0, rank_split_seq=7)
        self.assertIsNotNone(rec)
        decision = rec.get("ms_decision")
        if decision is not None:
            self.assertIsNone(
                decision.get("mean_total_ms"),
                "a boundary with no fresh forward must not report a mean",
            )


# --------------------------------------------------------------------------
# defect 5 -- resolved state, not the flag


class TestBootedWeightsComeFromResolvedState(CustomTestCase):
    def setUp(self):
        from sglang.srt.distributed.utils import set_tp_partition_ratios

        self.addCleanup(set_tp_partition_ratios, None, None)

    def _sched(self, **sa):
        args = types.SimpleNamespace(
            rank_mlp_ratio=None,
            rank_tp_ratio=None,
            rank_gpu_memory_mib=[27107, 16680, 16680],
            vram_budget_mib=None,
            model_path="/nonexistent/model",
            tp_size=3,
        )
        for k, v in sa.items():
            setattr(args, k, v)
        return types.SimpleNamespace(
            server_args=args,
            max_total_num_tokens=453632,
            kv_reshard_runtime=types.SimpleNamespace(
                current_vector=(30, 17, 17), allowed_vectors=((30, 17, 17), (1, 1, 1))
            ),
        )

    def test_the_flag_is_not_the_answer_when_state_disagrees(self):
        """The falsifier: flag unset, resolved plan concrete.

        This is the auto-performance boot. The flag says nothing; the server
        is running [27107, 16680, 16680]. Reading the flag reports
        weights=None and makes every candidate NO_WEIGHT_MOVER.
        """
        from sglang.srt.distributed.utils import set_tp_partition_ratios

        set_tp_partition_ratios([27107, 16680, 16680], None)
        stage = _booted_stage(self._sched(rank_mlp_ratio=None))
        self.assertIsNotNone(stage)
        self.assertIsNotNone(
            stage.weight_vector,
            "the booted stage must report the RESOLVED partition, not the unset flag",
        )

    def test_the_vector_is_canonicalised_before_comparison(self):
        """Same layout, different scale, must compare equal.

        A resolved plan of [60, 34, 42] and a candidate ratio of [30, 17, 21]
        describe ONE partition. Comparing them raw is a false mismatch.
        """
        from sglang.srt.distributed.utils import set_tp_partition_ratios

        set_tp_partition_ratios([60, 34, 42], None)
        stage = _booted_stage(self._sched(rank_mlp_ratio=None))
        self.assertEqual(tuple(stage.weight_vector), (30, 17, 21))

    def test_an_explicit_flag_still_wins_and_is_reduced(self):
        """The pinned-ratio boot keeps working, in the same space."""
        from sglang.srt.distributed.utils import set_tp_partition_ratios

        set_tp_partition_ratios([27107, 16680, 16680], {"mlp": [94, 13, 29]})
        stage = _booted_stage(self._sched(rank_mlp_ratio=[94, 13, 29]))
        self.assertEqual(tuple(stage.weight_vector), (94, 13, 29))

    def test_a_reachable_candidate_stops_being_no_weight_mover(self):
        """The consequence the defect actually had, end to end."""
        from sglang.srt.distributed.utils import set_tp_partition_ratios
        from sglang.srt.managers.regime_classifier import REGIME_MIXED, Stage
        from sglang.srt.managers.regime_stages import REACH_RESHARD, reachability

        set_tp_partition_ratios([188, 26, 58], None)
        booted = _booted_stage(self._sched(rank_mlp_ratio=None))
        candidate = Stage(
            name="planner:maxkv",
            regime=REGIME_MIXED,
            weight_vector=(94, 13, 29),
            kv_token_vector=(1, 1, 1),
            vram_budget_mib=tuple(booted.vram_budget_mib),
            max_total_num_tokens=booted.max_total_num_tokens,
            measured_gain_pct=0.0,
            measured_band_pct=0.0,
            flip_cost_s=0.0,
        )
        code, reason = reachability(booted, candidate, ((30, 17, 17), (1, 1, 1)))
        self.assertEqual(code, REACH_RESHARD, reason)


if __name__ == "__main__":
    unittest.main()
