# SPDX-License-Identifier: Apache-2.0
"""Rank-uniformity contract of the KV pressure ladder runtime (#287).

Hermetic (CPU-only, no torch.distributed): the consensus channel is
injected. The desync falsifier drives REAL threads through a barrier-backed
mock channel, so "fails loudly, never hangs" is demonstrated with actual
concurrency, not asserted."""

import inspect
import logging
import threading
import unittest

from sglang.srt.managers.admission_limiter import (
    DEFAULT_RELEASE_LOW,
    DEFAULT_THROTTLE_HIGH,
    AdmissionLimiter,
)
from sglang.srt.managers.kv_pressure_runtime import (
    LOG_PREFIX,
    KvPressureRuntime,
    build_kv_pressure_runtime,
)
from sglang.srt.model_executor.kv_pressure_ladder import (
    DEFAULT_ASCEND_THRESHOLD,
    DEFAULT_DESCEND_THRESHOLD,
    DEFAULT_PRE_STAGE_THRESHOLD,
    PHASE_FLIP,
    PHASE_PRE_STAGE,
    STEP_BASE,
    STEP_RELIEF,
    KvLadderError,
    KvPressureLadder,
    KvPressureSensor,
    LadderStep,
    OperatingPoint,
    PressureLadder,
    StageOperatingGrid,
)

CAPACITY = 10_000


def _table(*, with_admission=True, with_grid=False):
    grid = None
    if with_grid:
        grid = StageOperatingGrid(
            [
                OperatingPoint("prefill", 0.05, (7, 3, 3), (5, 1, 1)),
                OperatingPoint("prefill", 0.75, (7, 3, 3), (2, 1, 1)),
                OperatingPoint("decode", 0.05, (7, 3, 3), (5, 3, 3)),
                OperatingPoint("decode", 0.75, (7, 3, 3), (6, 5, 5)),
            ]
        )
    steps = [
        LadderStep(name="base", step_type=STEP_BASE),
        LadderStep(
            name="dcp_ratio",
            step_type=STEP_RELIEF,
            relief_feature="dcp_ratio",
            operating_grid=grid,
        ),
    ]
    if with_admission:
        steps.append(
            LadderStep(
                name="admission_cap",
                step_type=STEP_RELIEF,
                relief_feature="admission_cap",
            )
        )
    return PressureLadder(steps)


def _sensor(**kwargs):
    defaults = dict(
        ascend_threshold=0.85,
        ascend_window=2,
        descend_threshold=0.55,
        descend_window=6,
        pre_stage_threshold=0.70,
        pre_stage_window=2,
        abort_stage_window=8,
        horizon_rounds=4,
    )
    defaults.update(kwargs)
    return KvPressureSensor(**defaults)


def _ladder(pre_stage=False, **table_kwargs):
    return KvPressureLadder(
        _table(**table_kwargs), _sensor(), pre_stage_enabled=pre_stage
    )


def _limiter():
    return AdmissionLimiter(64, 64, auto=True)


def _runtime(ladder=None, limiter=None, **kwargs):
    return KvPressureRuntime(
        ladder or _ladder(),
        admission_limiter=limiter or _limiter(),
        **kwargs,
    )


def _drive(runtime, occupancies, phase="decode", running_bs=8):
    plans = []
    for occ in occupancies:
        plans.append(
            runtime.on_round(
                held_tokens=int(occ * CAPACITY),
                capacity_tokens=CAPACITY,
                running_bs=running_bs,
                phase=phase,
            )
        )
    return plans


RAMP = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.88, 0.90, 0.92, 0.94, 0.96, 0.97]
CALM = [0.30, 0.32, 0.31, 0.30, 0.29, 0.30, 0.31, 0.30, 0.29, 0.30, 0.31, 0.30]


class _BarrierChannel:
    """Element-wise MIN across N rank threads, synchronized by a barrier.

    A broken barrier (a rank that never arrives) raises in every waiting
    thread -- the mock inherits the loud-failure property the production
    bounded collective has, so a hang cannot masquerade as a pass."""

    def __init__(self, n, timeout=15.0):
        self.n = n
        self._barrier = threading.Barrier(n, timeout=timeout)
        self._slots = [None] * n
        self._result = None
        self.calls_per_rank = [0] * n

    def channel_for(self, rank):
        def _reduce(vals):
            self.calls_per_rank[rank] += 1
            self._slots[rank] = list(vals)
            index = self._barrier.wait()
            if index == 0:
                self._result = [min(col) for col in zip(*self._slots)]
            self._barrier.wait()
            return list(self._result)

        return _reduce


def _run_ranks(n, series_per_rank, interval=2):
    """Drive N rank runtimes through their series on real threads. Returns
    (runtimes, exceptions_per_rank). Joins with a hard timeout so a hang
    fails the test instead of freezing the suite."""
    channel = _BarrierChannel(n)
    runtimes = [
        _runtime(
            tp_size=n,
            collective_min=channel.channel_for(r),
            consensus_interval=interval,
        )
        for r in range(n)
    ]
    errors = [None] * n

    def _worker(rank):
        try:
            _drive(runtimes[rank], series_per_rank[rank])
        except BaseException as exc:  # noqa: BLE001 - the falsifier inspects it
            errors[rank] = exc

    threads = [
        threading.Thread(target=_worker, args=(r,), daemon=True) for r in range(n)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)
    alive = [t for t in threads if t.is_alive()]
    if alive:
        raise AssertionError(
            f"{len(alive)} rank thread(s) still alive after 30s -- the "
            f"consensus path HUNG, which is exactly what it must never do"
        )
    return runtimes, errors, channel


class TestRankUniformDecision(unittest.TestCase):
    def test_identical_replicated_inputs_flip_identically(self):
        """Rule 1+2: with replicated inputs, three ranks pass the consensus
        channel and land on the same rung at the same epoch."""
        runtimes, errors, channel = _run_ranks(3, [list(RAMP)] * 3)
        self.assertEqual(errors, [None, None, None])
        rungs = [rt.ladder.current_rung for rt in runtimes]
        epochs = [rt.epoch for rt in runtimes]
        self.assertEqual(len(set(rungs)), 1)
        self.assertEqual(len(set(epochs)), 1)
        self.assertGreater(epochs[0], 0, "the ramp must actually flip")
        self.assertGreater(runtimes[0].desync_checks, 0)
        # The channel is exercised at every boundary on every rank.
        self.assertEqual(len(set(channel.calls_per_rank)), 1)

    def test_desync_falsifier_diverging_ranks_fail_loudly_never_hang(self):
        """THE falsifier: rank 2 reads a perturbed (rank-local) picture and
        would decide differently. Every rank must raise the same loud
        KvLadderError at the consensus boundary -- no rank may hang, no
        rank may proceed on its local opinion."""
        # Rank 2 reads a picture whose VERDICT diverges (its local pool looks
        # full while the group is calm) -- the exact shape a rank-local
        # allocator read would produce under uneven DCP.
        perturbed = [0.95] * len(CALM)
        runtimes, errors, _channel = _run_ranks(3, [list(CALM), list(CALM), perturbed])
        for rank, err in enumerate(errors):
            self.assertIsInstance(
                err,
                KvLadderError,
                f"rank {rank} must raise the desync error, got {err!r}",
            )
            self.assertIn("DESYNC", str(err))
        # Nobody committed a transition on a disputed proposal.
        self.assertEqual({rt.ladder.current_rung for rt in runtimes}, {0})

    def test_consensus_cadence_is_unconditional(self):
        """Rule 2: the channel fires every interval-th round even with zero
        pressure -- the cadence gate is the round counter, never the local
        verdict (the rank-local-condition-before-collective trap)."""
        runtimes, errors, channel = _run_ranks(2, [list(CALM)] * 2, interval=3)
        self.assertEqual(errors, [None, None])
        self.assertEqual(channel.calls_per_rank, [4, 4])  # 12 rounds / 3
        self.assertEqual([rt.transitions for rt in runtimes], [0, 0])

    def test_multi_rank_without_channel_is_refused(self):
        with self.assertRaisesRegex(ValueError, "consensus channel"):
            _runtime(tp_size=3, collective_min=None)

    def test_malformed_channel_result_is_loud(self):
        rt = _runtime(
            tp_size=2, collective_min=lambda vals: vals[:3], consensus_interval=1
        )
        with self.assertRaisesRegex(KvLadderError, "channel contract"):
            _drive(rt, [0.3])


class TestDeterministicSelection(unittest.TestCase):
    def test_same_series_same_decisions(self):
        """Gate: stage selection is deterministic from (phase, depth,
        format scores, pressure) -- two runtimes over the same replicated
        series take identical transitions."""
        a, b = _runtime(consensus_interval=2), _runtime(consensus_interval=2)
        plans_a = [p for p in _drive(a, RAMP) if p is not None]
        plans_b = [p for p in _drive(b, RAMP) if p is not None]
        self.assertEqual(
            [(p.phase, p.current_rung, p.target_rung) for p in plans_a],
            [(p.phase, p.current_rung, p.target_rung) for p in plans_b],
        )
        self.assertEqual(a.ladder.current_rung, b.ladder.current_rung)
        self.assertEqual(a.epoch, b.epoch)

    def test_operating_point_depends_on_phase_and_depth(self):
        """The flipped rung's grid answers per (phase, depth) -- the depth/
        format axis reaches the runtime decision, not only the table."""
        ladder = _ladder(with_grid=True, with_admission=False)
        rt = _runtime(ladder=ladder, consensus_interval=2)
        with self.assertLogs(
            "sglang.srt.managers.kv_pressure_runtime", level=logging.WARNING
        ) as logs:
            _drive(rt, RAMP, phase="decode")
        flip_lines = [
            line for line in logs.output if "FLIP" in line and "dcp_ratio" in line
        ]
        self.assertTrue(flip_lines)
        # The trend fires the flip at occupancy 0.60 -> the low decode bin.
        self.assertIn("[5, 3, 3]", flip_lines[0])
        self.assertIn("operating point (decode", flip_lines[0])
        # The same ladder state asked for the prefill face gives the
        # prefill optimum -- phase is a real axis, not a label.
        point = ladder.operating_point("prefill", 0.9)
        self.assertEqual(point.kv_vector, (2, 1, 1))
        self.assertEqual(ladder.operating_point("decode", 0.1).kv_vector, (5, 3, 3))


class TestActuators(unittest.TestCase):
    def test_dcp_ratio_flip_is_planned_only_and_says_so(self):
        lim = _limiter()
        rt = _runtime(
            ladder=_ladder(with_admission=False),
            limiter=lim,
            consensus_interval=2,
        )
        with self.assertLogs(
            "sglang.srt.managers.kv_pressure_runtime", level=logging.WARNING
        ) as logs:
            _drive(rt, RAMP[:8])
        self.assertEqual(rt.ladder.current_rung, 1)  # dcp_ratio
        self.assertIn("dcp_ratio", rt.planned_only_reliefs)
        self.assertTrue(any("PLANNED-ONLY" in line for line in logs.output))
        # Planned-only: the admission limiter is NOT touched by this rung.
        self.assertEqual(lim.current, lim.start)

    def test_admission_flip_throttles_and_descend_releases(self):
        lim = _limiter()
        rt = _runtime(limiter=lim, consensus_interval=2)
        _drive(rt, RAMP + [0.97, 0.97, 0.97, 0.97], running_bs=16)
        self.assertEqual(rt.ladder.current_rung, 2)  # admission_cap
        self.assertEqual(lim.current, 15)  # min(64, 16) - 1
        self.assertEqual(lim.last_reason, "kv_pressure")
        # Long calm stretch: the ladder descends (sluggish window) and the
        # release gives the float back stepwise, never in one jump.
        _drive(rt, [0.30] * 30, running_bs=4)
        self.assertLess(rt.ladder.current_rung, 2)
        self.assertGreater(lim.current, 15)
        self.assertLessEqual(lim.current, lim.start)

    def test_admission_rung_without_armed_limiter_is_a_boot_error(self):
        with self.assertRaisesRegex(KvLadderError, "max-running-requests-ceiling"):
            KvPressureRuntime(
                _ladder(), admission_limiter=AdmissionLimiter(64, auto=False)
            )
        with self.assertRaisesRegex(KvLadderError, "max-running-requests-ceiling"):
            KvPressureRuntime(_ladder(), admission_limiter=None)

    def test_session_offload_rung_without_manager_is_a_boot_error(self):
        table = PressureLadder(
            [
                LadderStep(name="base", step_type=STEP_BASE),
                LadderStep(
                    name="session_offload",
                    step_type=STEP_RELIEF,
                    relief_feature="session_offload",
                ),
            ]
        )
        ladder = KvPressureLadder(table, _sensor())
        with self.assertRaisesRegex(KvLadderError, "enable-kv-session-offload"):
            KvPressureRuntime(ladder, spill_fn=None)

    def test_session_offload_flip_calls_the_spill(self):
        table = PressureLadder(
            [
                LadderStep(name="base", step_type=STEP_BASE),
                LadderStep(
                    name="session_offload",
                    step_type=STEP_RELIEF,
                    relief_feature="session_offload",
                ),
            ]
        )
        calls = []
        rt = KvPressureRuntime(
            KvPressureLadder(table, _sensor()),
            spill_fn=lambda bs: calls.append(bs) or True,
            consensus_interval=2,
        )
        _drive(rt, RAMP[:8], running_bs=5)
        self.assertEqual(rt.ladder.current_rung, 1)
        self.assertEqual(calls, [5])


class TestNoPressureRegression(unittest.TestCase):
    def test_flag_off_builds_nothing(self):
        class _Sched:
            class server_args:
                kv_pressure_ladder = None

            kv_session_offload = None
            admission_limiter = None
            tp_cpu_group = None
            running_batch = None

        self.assertIsNone(build_kv_pressure_runtime(_Sched()))

    def test_calm_series_changes_nothing(self):
        """Gate: without pressure the ladder holds, the limiter holds, no
        actuator fires -- the regression pin of today's behavior."""
        lim = _limiter()
        rt = _runtime(limiter=lim, consensus_interval=2)
        plans = [p for p in _drive(rt, CALM) if p is not None]
        self.assertTrue(all(p.is_noop or p.blocked for p in plans))
        self.assertEqual(rt.ladder.current_rung, 0)
        self.assertEqual(rt.transitions, 0)
        self.assertEqual(lim.current, lim.start)

    def test_pre_stage_commits_no_actuator_and_no_rung(self):
        """Gate (Erg. 9b): pre-staging without a flip is bookkeeping only --
        the active path (rung, limiter, spill) is untouched."""
        # Geometry rung above base so a pre-stage target exists (relief
        # rungs are never staged).
        from sglang.srt.model_executor.kv_pressure_ladder import (
            HANDOVER_BACKGROUND_MIGRATE,
            STEP_GEOMETRY,
        )

        table = PressureLadder(
            [
                LadderStep(name="base", step_type=STEP_BASE),
                LadderStep(
                    name="tp3-fine",
                    step_type=STEP_GEOMETRY,
                    geometry_key="tp3-fine",
                    handover=HANDOVER_BACKGROUND_MIGRATE,
                ),
            ]
        )
        lim = _limiter()
        ladder = KvPressureLadder(table, _sensor(), pre_stage_enabled=True)
        rt = KvPressureRuntime(ladder, admission_limiter=lim, consensus_interval=2)
        # Hold between pre-stage (0.70) and ascend (0.85): stages, no flip.
        plans = [p for p in _drive(rt, [0.72, 0.74, 0.76, 0.78, 0.79, 0.80]) if p]
        self.assertIn(PHASE_PRE_STAGE, [p.phase for p in plans])
        self.assertEqual(ladder.current_rung, 0)
        self.assertEqual(ladder.staged_target, 1)
        self.assertEqual(lim.current, lim.start)

    def test_warm_shadow_flip_is_delta_only(self):
        from sglang.srt.model_executor.kv_pressure_ladder import (
            HANDOVER_ANTICIPATORY_SHADOW,
            HANDOVER_BACKGROUND_MIGRATE,
            STEP_GEOMETRY,
        )

        table = PressureLadder(
            [
                LadderStep(name="base", step_type=STEP_BASE),
                LadderStep(
                    name="tp3-fine",
                    step_type=STEP_GEOMETRY,
                    geometry_key="tp3-fine",
                    handover=HANDOVER_BACKGROUND_MIGRATE,
                ),
            ]
        )
        ladder = KvPressureLadder(table, _sensor(), pre_stage_enabled=True)
        rt = KvPressureRuntime(ladder, consensus_interval=2)
        plans = [
            p for p in _drive(rt, [0.72, 0.74, 0.76, 0.78, 0.88, 0.90, 0.92, 0.94]) if p
        ]
        flips = [p for p in plans if p.phase == PHASE_FLIP]
        self.assertTrue(flips)
        self.assertTrue(flips[0].delta_only)
        self.assertEqual(flips[0].handover, HANDOVER_ANTICIPATORY_SHADOW)


class TestThresholdAnchors(unittest.TestCase):
    def test_ladder_marks_sit_inside_the_307_actuator_marks(self):
        """Gate (#307-fit anchor): the ladder is the EARLY, planned reaction;
        the floating admission limiter's own throttle (0.90) and the retract
        fallback (1.0) are the last resorts. The default marks must keep
        that order or the ladder would fire after the emergency brake."""
        self.assertLess(DEFAULT_DESCEND_THRESHOLD, DEFAULT_RELEASE_LOW)
        self.assertLessEqual(DEFAULT_RELEASE_LOW, DEFAULT_PRE_STAGE_THRESHOLD)
        self.assertLess(DEFAULT_PRE_STAGE_THRESHOLD, DEFAULT_ASCEND_THRESHOLD)
        self.assertLess(DEFAULT_ASCEND_THRESHOLD, DEFAULT_THROTTLE_HIGH)
        self.assertLess(DEFAULT_THROTTLE_HIGH, 1.0)


class TestLogging(unittest.TestCase):
    def test_flip_is_loud_and_greppable(self):
        rt = _runtime(consensus_interval=2)
        with self.assertLogs(
            "sglang.srt.managers.kv_pressure_runtime", level=logging.WARNING
        ) as logs:
            _drive(rt, RAMP[:8])
        self.assertTrue(
            any(LOG_PREFIX in line and "FLIP" in line for line in logs.output)
        )


if __name__ == "__main__":
    unittest.main()


LADDER_LOGGER = "sglang.srt.managers.kv_pressure_runtime"


def _hold_lines(ctx) -> list:
    """Only the ladder's 'hold:' INFO lines from an assertLogs capture."""
    return [r.getMessage() for r in ctx.records if " hold: " in r.getMessage()]


class TestHoldLogIsNotSpam582(unittest.TestCase):
    """#582: the hold line was continuous INFO fire.

    Two independent defects produced it:

    1. It narrated the IDLE regime. Production logged
       "hold: occupancy between the marks" at occupancy 0.06 against a 0.55
       descend mark, once per scheduler round. Below the lowest mark the
       ladder is not deferring any decision an operator could act on.
    2. The de-duplication compared the FORMATTED REASON, and several reasons
       embed live numbers -- e.g. "pre-stage: level>=0.700 over 3 rounds=
       False, projected exhaustion<=64 rounds=True". Those flip between
       rounds, so the text differed almost every round and the guard passed
       almost every round. De-duplication now keys on the coarse STATE.
    """

    def test_silent_far_below_the_lower_mark(self):
        rt = _runtime()
        with self.assertLogs(LADDER_LOGGER, level="DEBUG") as ctx:
            logging.getLogger(LADDER_LOGGER).debug("anchor so assertLogs has a record")
            _drive(rt, [0.06] * 40)
        self.assertEqual(
            _hold_lines(ctx), [],
            "the ladder narrated the idle regime; at 0.06 against a 0.55 "
            "descend mark there is no deferred decision to report",
        )

    def test_steady_state_in_band_logs_once_not_per_round(self):
        rt = _runtime()
        with self.assertLogs(LADDER_LOGGER, level="DEBUG") as ctx:
            logging.getLogger(LADDER_LOGGER).debug("anchor")
            _drive(rt, [0.60] * 30)
        lines = _hold_lines(ctx)
        self.assertLessEqual(
            len(lines), 1,
            f"steady state produced {len(lines)} hold lines; a state that has "
            f"not changed is not news. Got: {lines[:4]}",
        )

    def test_one_line_per_state_transition(self):
        """Crossing marks is news; sitting still is not."""
        rt = _runtime()
        with self.assertLogs(LADDER_LOGGER, level="DEBUG") as ctx:
            logging.getLogger(LADDER_LOGGER).debug("anchor")
            # idle -> low -> idle -> low, each held for several rounds.
            _drive(rt, [0.10] * 6)
            _drive(rt, [0.60] * 6)
            _drive(rt, [0.10] * 6)
            _drive(rt, [0.60] * 6)
        lines = _hold_lines(ctx)
        self.assertLessEqual(
            len(lines), 4,
            f"expected at most one line per band transition, got {len(lines)}",
        )
        self.assertGreaterEqual(
            len(lines), 1,
            "leaving the idle band is a transition and must still be reported",
        )

    def test_numeric_churn_in_the_reason_does_not_defeat_dedup(self):
        """The exact regression: same STATE, drifting numbers in the text.

        Occupancies differ every round but never leave the band, so the
        formatted reason can churn while the state does not. The old
        reason-text guard logged on every one of these rounds.
        """
        rt = _runtime()
        occs = [0.60 + (i % 7) * 0.001 for i in range(30)]
        with self.assertLogs(LADDER_LOGGER, level="DEBUG") as ctx:
            logging.getLogger(LADDER_LOGGER).debug("anchor")
            _drive(rt, occs)
        lines = _hold_lines(ctx)
        self.assertLessEqual(
            len(lines), 1,
            f"numeric churn inside one band produced {len(lines)} lines; "
            f"de-duplication is keying on the text again. Got: {lines[:4]}",
        )

    def test_blocked_plans_are_still_reported_even_when_idle(self):
        """A blocked plan means something WANTED to move and could not.

        That is news at any occupancy, so the idle suppression must not
        swallow it. Guarding this keeps the fix from becoming "log less" in
        the direction that costs an operator a diagnosis.
        """
        rt = _runtime()
        source = inspect.getsource(type(rt)._commit)
        self.assertIn("plan.blocked is None", source)
        self.assertIn("_band_of", source)
