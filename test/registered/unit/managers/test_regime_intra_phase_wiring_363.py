"""#363 intra-phase axis, wired into the observer. Hermetic, no GPU.

The two module suites pin the clock and the gate in isolation. This one pins
that they are actually IN the loop, because a correct component that nothing
calls is the failure mode the #578 stage table already demonstrated: it
solved, and nothing actuated from it for a full phase.

Three properties:

1. DEFAULT OFF IS BYTE-IDENTICAL. Without the flag the observer holds neither
   object, adds no collective, and its record carries the same keys.
2. THE GATE IS IN THE ACT PATH. An unfunded stage flip is refused by the
   admission interlock and NOTHING reaches the actuator -- pinned by a commit
   callable that records whether it was called.
3. THE CLOCK IS IN THE DECISION. A wired clock is fed the group-reduced
   split, and a stage the ms axis proposes is the one the act path receives.
"""

import unittest

from sglang.srt.managers.regime_admission import CorridorAdmission
from sglang.srt.managers.regime_classifier import (
    REGIME_DECODE_HEAVY,
    REGIME_MIXED,
    RegimeSensor,
    Stage,
    StageTable,
)
from sglang.srt.managers.regime_ms_clock import MsStageDecider
from sglang.srt.managers.regime_runtime import MODE_ACT, MODE_OBSERVE, RegimeObserver
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_stage(
    name, *, gain, band=1.0, vram=(8000, 8000), kv=(1, 1), regime=REGIME_MIXED
):
    return Stage(
        name=name,
        regime=regime,
        weight_vector=None,
        kv_token_vector=tuple(kv),
        vram_budget_mib=tuple(vram),
        max_total_num_tokens=1_000_000,
        measured_gain_pct=gain,
        measured_band_pct=band,
        flip_cost_s=1.0,
    )


BOOTED = make_stage("balanced", gain=0.0)
# Solved for the decode-heavy regime, so the LABEL axis proposes it under a
# decode load -- which is what lets the admission tests below exercise the
# interlock without a clock wired.
FASTER = make_stage(
    "split-heavy",
    gain=30.0,
    vram=(8512, 8512),
    kv=(2, 1),
    regime=REGIME_DECODE_HEAVY,
)


class RecordingCommit:
    """Stands in for RegimeActuator.apply and records whether it was called."""

    def __init__(self):
        self.calls = []

    def __call__(self, stage, current):
        self.calls.append((stage.name, current.name))

        class _R:
            armed = True

            def as_dict(self_inner):
                return {"stage": stage.name, "armed": True}

        return _R()


class StubGuardResult:
    def __init__(self, ok):
        self.ok = ok
        self.free_before = 20_000 * 1024 * 1024
        self.free_after = self.free_before
        self.reclaimed = 0
        self.detail = f"stub ok={ok}"


class StubGuard:
    def __init__(self, ok):
        self.ok = ok
        self.calls = []

    def ensure_headroom(self, want_bytes, *, reason="", **_kw):
        self.calls.append(want_bytes)
        return StubGuardResult(self.ok)


#: Same stage, solved for MIXED. With both stages mixed the LABEL axis has
#: nothing to propose (the table's mixed entry is the one already in force),
#: so anything that moves came from the ms clock -- which is what the clock
#: tests need in order to be about the clock.
MIXED_FASTER = make_stage(
    "split-heavy", gain=30.0, vram=(8512, 8512), kv=(2, 1), regime=REGIME_MIXED
)


def make_observer(
    *, stage_clock=None, admission=None, commit=None, mode=MODE_ACT, target=FASTER
):
    table = StageTable([BOOTED, target], reference=BOOTED.name)
    return RegimeObserver(
        sensor=RegimeSensor(),
        table=table,
        table_plan=None,
        consensus_interval=1,
        tp_size=1,
        mode=mode,
        current_stage=BOOTED.name,
        commit_fn=commit,
        stage_clock=stage_clock,
        admission=admission,
    )


def drive(observer, n, *, compute=50.0, wait=50.0, phase="decode"):
    last = None
    for i in range(n):
        last = observer.on_round(
            phase=phase,
            held_tokens=1000,
            capacity_tokens=1_000_000,
            running_bs=4,
            rank_forward_ms=compute + wait,
            rank_compute_ms=compute,
            rank_wait_ms=wait,
        )
    return last


class TestDefaultOffIsUnchanged(unittest.TestCase):
    def test_observe_mode_holds_neither_object(self):
        obs = make_observer(mode=MODE_OBSERVE)
        self.assertIsNone(obs._stage_clock)
        self.assertIsNone(obs._admission)

    def test_record_shape_is_unchanged_and_ms_decision_is_none(self):
        obs = make_observer(mode=MODE_OBSERVE)
        record = drive(obs, 4)
        self.assertIsNone(record["ms_decision"])
        self.assertIsNone(obs.summary()["stage_clock"])
        self.assertIsNone(obs.summary()["admission"])
        self.assertEqual(obs.stage_clock_proposals, 0)

    def test_the_split_is_accepted_but_unused_when_off(self):
        """An off boot may be handed the split; it must change nothing."""
        obs = make_observer(mode=MODE_OBSERVE)
        with_split = drive(obs, 6)
        self.assertIsNone(with_split["ms_decision"])


class TestAdmissionIsInTheActPath(unittest.TestCase):
    def test_an_unfunded_flip_never_reaches_the_actuator(self):
        """The interlock, not the actuator, is what refuses."""
        commit = RecordingCommit()
        guard = StubGuard(ok=False)
        admission = CorridorAdmission(
            guard_fn=lambda: guard,
            load_state_fn=lambda: REGIME_MIXED,
            transient_fn=lambda stage: {REGIME_MIXED: 1500.0},
            rank=0,
            tp_size=1,
        )
        obs = make_observer(admission=admission, commit=commit)
        drive(obs, 40)
        self.assertEqual(commit.calls, [])
        self.assertGreater(guard.calls, [])
        self.assertEqual(obs.actuations, 0)

    def test_a_missing_census_refuses_rather_than_pricing_at_zero(self):
        commit = RecordingCommit()
        guard = StubGuard(ok=True)
        admission = CorridorAdmission(
            guard_fn=lambda: guard,
            load_state_fn=lambda: REGIME_MIXED,
            transient_fn=lambda stage: {},
            rank=0,
            tp_size=1,
        )
        obs = make_observer(admission=admission, commit=commit)
        drive(obs, 40)
        self.assertEqual(commit.calls, [])
        # The guard was never even asked: the price could not be formed.
        self.assertEqual(guard.calls, [])

    def test_a_funded_flip_does_reach_the_actuator(self):
        """The gate must OPEN, or the two tests above prove nothing."""
        commit = RecordingCommit()
        guard = StubGuard(ok=True)
        admission = CorridorAdmission(
            guard_fn=lambda: guard,
            load_state_fn=lambda: REGIME_MIXED,
            transient_fn=lambda stage: {REGIME_MIXED: 1500.0},
            rank=0,
            tp_size=1,
        )
        obs = make_observer(admission=admission, commit=commit)
        drive(obs, 40)
        self.assertGreater(len(commit.calls), 0)
        self.assertEqual(commit.calls[0][0], FASTER.name)


class TestClockIsInTheDecision(unittest.TestCase):
    def test_a_wait_bound_rig_proposes_the_faster_stage(self):
        commit = RecordingCommit()
        obs = make_observer(
            stage_clock=MsStageDecider(), commit=commit, target=MIXED_FASTER
        )
        drive(obs, 60, compute=50.0, wait=50.0)
        self.assertGreater(obs.stage_clock_proposals, 0)
        self.assertGreater(len(commit.calls), 0)
        self.assertEqual(commit.calls[0][0], FASTER.name)

    def test_a_compute_bound_rig_proposes_nothing_on_the_ms_axis(self):
        """Same table, same load shape, only the split differs."""
        commit = RecordingCommit()
        obs = make_observer(
            stage_clock=MsStageDecider(), commit=commit, target=MIXED_FASTER
        )
        drive(obs, 60, compute=95.0, wait=5.0)
        self.assertEqual(obs.stage_clock_proposals, 0)

    def test_the_record_carries_the_ms_verdict(self):
        obs = make_observer(
            stage_clock=MsStageDecider(),
            commit=RecordingCommit(),
            target=MIXED_FASTER,
        )
        record = drive(obs, 20, compute=50.0, wait=50.0)
        self.assertIsNotNone(record["ms_decision"])
        self.assertIn("signal_pct", record["ms_decision"])
        self.assertIsNotNone(obs.summary()["stage_clock"])

    def test_a_blind_rank_makes_the_axis_abstain(self):
        """No split reported at all -- the axis must not invent one."""
        obs = make_observer(
            stage_clock=MsStageDecider(),
            commit=RecordingCommit(),
            target=MIXED_FASTER,
        )
        for _ in range(40):
            record = obs.on_round(
                phase="decode",
                held_tokens=1000,
                capacity_tokens=1_000_000,
                running_bs=4,
                rank_forward_ms=100.0,
                rank_compute_ms=None,
                rank_wait_ms=None,
            )
        self.assertIsNone(record["ms_decision"])
        self.assertEqual(obs.stage_clock_proposals, 0)

    def test_multi_rank_without_a_channel_abstains(self):
        """A rank-local clock on replicated state is the forbidden thing."""
        table = StageTable([BOOTED, MIXED_FASTER], reference=BOOTED.name)
        obs = RegimeObserver(
            table=table,
            consensus_interval=1,
            tp_size=2,
            collective_min=None,
            mode=MODE_ACT,
            current_stage=BOOTED.name,
            commit_fn=RecordingCommit(),
            stage_clock=MsStageDecider(),
        )
        record = drive(obs, 40)
        self.assertIsNone(record["ms_decision"])
        self.assertEqual(obs.stage_clock_proposals, 0)


if __name__ == "__main__":
    unittest.main()
