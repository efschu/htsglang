"""#363 phase 2 -- falsifiers F2-F5 and the observe-only wiring contract.

Hermetic: no GPU, no CUDA, no server. Synthetic ``RegimeSample`` streams and
an injected consensus channel, which is all three of the remaining desk-
reachable falsifiers need.

* **F2 -- self-conditioning replay** (the #156 lesson, with the mechanism
  DESIGN_363 section 7.3 named). Run the same workload open-loop (nothing
  actuates, so nothing in the trace is self-caused) and closed-loop, and
  compare the regime sequences. The test does it twice: once against a NAIVE
  controller that ignores the stage-admissibility interlock, to prove the trap
  is real on these numbers, and once against the guarded one, to prove the
  interlock closes it.
* **F3 -- noise floor per input signal.** The #360 rule: the band a signal has
  to clear is what the arm measures against its own repeat. Two properties are
  pinned -- the shipped thresholds clear their bands at the required margin,
  and a threshold placed inside its band produces a controller that flips on
  noise.
* **F4 -- do-nothing baseline.** The card half is phase 3. The hermetic half
  is the property that makes the comparison meaningful at all: the observe arm
  must be indistinguishable from the off arm except in its log, so that it can
  serve as its own do-nothing baseline.
* **F5 -- consensus desync.** An injected channel that merges different ranks'
  payloads. In observe-only the verdict is that the observer COUNTS and LOGS
  the disagreement and does not raise -- an instrument that takes the server
  down while proving it is safe has failed at its own job -- and that the
  count is visible in the summary, because it is the phase-3 gate.

F0 (interlock refusal) and F1 (oscillation) live in test_regime_classifier.py
and stay green there.
"""

import logging
import unittest

from sglang.srt.managers.regime_classifier import (
    _MS_ABSENT,
    DEFAULT_ENTER_PREFILL,
    KV_ASCEND_MARK,
    RANK_MS_QUANTUM,
    REGIME_CODES,
    REGIME_DECODE_HEAVY,
    REGIME_KV_PRESSURE,
    REGIME_MIXED,
    REGIME_PREFILL_HEAVY,
    DwellGate,
    RegimeSample,
    RegimeSensor,
    Stage,
    StageTable,
    clears_band,
    signal_band,
)
from sglang.srt.managers.regime_runtime import (
    ENV_MODE,
    MODE_OBSERVE,
    MODE_OFF,
    PHASE_DECODE,
    PHASE_IDLE,
    PHASE_PREFILL,
    RegimeObserver,
    observe_mode,
    phase_of_last_batch,
    rank_forward_ms_from,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# The measured #354 stage pair (docs/rig-runbook.md section 4.1.0).
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# F2 -- self-conditioning replay
# ---------------------------------------------------------------------------


class _Workload:
    """A fixed sequence of (prefill_rounds, decode_rounds, held_tokens).

    The workload is the INPUT and does not depend on the controller. What the
    controller changes is the pool the held tokens are weighed against, which
    is exactly the coupling DESIGN_363 section 7.3 identified: occupancy is
    the pressure input, and a stage flip changes its denominator.
    """

    def __init__(self, rounds: int, held_tokens: int):
        self.rounds = rounds
        self.held = held_tokens

    def step(self, i: int):
        # Prefill-heavy throughout: the load never changes shape, so any
        # regime change in the closed loop came from the controller.
        return 48, 16, self.held


def _run_loop(workload, *, honour_interlock: bool, capacity_follows_stage: bool):
    """Drive sensor + table over ``workload``; return (regimes, commits).

    ``capacity_follows_stage=False`` is the OPEN loop: nothing the controller
    does reaches the sample, so the trace is free of self-caused effects.
    ``True`` is the closed loop.

    ``honour_interlock=False`` is the naive controller -- it selects the stage
    its regime names and ignores whether the working set fits the pool. It
    exists to prove the trap is real before the guard is credited with closing
    it.
    """
    table = _table()
    sensor = RegimeSensor(enter_window=2, exit_window=4)
    gate = DwellGate(min_rounds=8)
    stage = FP8_DECODE
    regimes, commits = [], []
    for i in range(workload.rounds):
        prefill_rounds, decode_rounds, held = workload.step(i)
        capacity = (
            stage.max_total_num_tokens
            if capacity_follows_stage
            else FP8_DECODE.max_total_num_tokens
        )
        sample = RegimeSample(
            round_index=i,
            prefill_rounds=prefill_rounds,
            decode_rounds=decode_rounds,
            held_tokens=held,
            capacity_tokens=capacity,
        )
        regime = sensor.observe(sample)
        regimes.append(regime)

        if honour_interlock:
            target, _why = table.select(regime, sample, current=stage.name)
        else:
            candidate = table.for_regime(regime)
            target = (
                candidate
                if candidate is not None and candidate.name != stage.name
                else None
            )
        if target is None:
            continue
        if not gate.allows(i)[0]:
            continue
        gate.record_flip(i)
        stage = target
        commits.append((i, stage.name))
    return regimes, commits


def _transitions(regimes):
    return [r for i, r in enumerate(regimes) if i == 0 or r != regimes[i - 1]]


class TestF2SelfConditioning(CustomTestCase):
    """FALSIFIER F2. The controller must not react to effects it caused.

    The workload holds 90 000 tokens and never changes shape. Against the
    decode stage's 453 632-token pool that is 20 % occupancy -- nowhere near
    pressure. Against the prefill stage's 96 256 it is 93 %, above the 85 %
    ascend mark. So a controller that flips to the prefill stage manufactures
    the KV_PRESSURE regime out of its own denominator.
    """

    WORKLOAD = _Workload(rounds=240, held_tokens=90_000)

    def test_the_trap_is_real_on_these_numbers(self):
        """Red by construction against a naive controller. If this ever goes
        green, the arithmetic behind section 7.3 has changed and the guard
        below is being credited for nothing."""
        open_regimes, open_commits = _run_loop(
            self.WORKLOAD, honour_interlock=False, capacity_follows_stage=False
        )
        closed_regimes, closed_commits = _run_loop(
            self.WORKLOAD, honour_interlock=False, capacity_follows_stage=True
        )
        # Open loop: prefill-heavy, and never anything else.
        self.assertNotIn(REGIME_KV_PRESSURE, open_regimes)
        self.assertIn(REGIME_PREFILL_HEAVY, open_regimes)
        # Closed loop: a regime the load never produced.
        self.assertIn(
            REGIME_KV_PRESSURE,
            closed_regimes,
            "the naive closed loop did not manufacture pressure; check the "
            "stage pools against the workload",
        )
        self.assertTrue(open_commits)
        # And the flip that caused it.
        self.assertEqual(closed_commits[0][1], "fp8-prefill")

    def test_every_closed_loop_transition_appears_in_the_open_loop_replay(self):
        """The F2 verdict. With the admissibility interlock, the guarded
        controller's regime sequence contains no transition the open-loop
        trace does not produce -- so nothing it observed was its own doing."""
        open_regimes, _ = _run_loop(
            self.WORKLOAD, honour_interlock=True, capacity_follows_stage=False
        )
        closed_regimes, _ = _run_loop(
            self.WORKLOAD, honour_interlock=True, capacity_follows_stage=True
        )
        self.assertEqual(_transitions(closed_regimes), _transitions(open_regimes))
        self.assertNotIn(REGIME_KV_PRESSURE, closed_regimes)

    def test_the_guard_is_what_makes_the_difference(self):
        """Same workload, same loop, one flag: the interlock. The naive arm
        diverges from its own open loop and the guarded arm does not."""
        naive_open, _ = _run_loop(
            self.WORKLOAD, honour_interlock=False, capacity_follows_stage=False
        )
        naive_closed, _ = _run_loop(
            self.WORKLOAD, honour_interlock=False, capacity_follows_stage=True
        )
        self.assertNotEqual(_transitions(naive_closed), _transitions(naive_open))

        guarded_open, _ = _run_loop(
            self.WORKLOAD, honour_interlock=True, capacity_follows_stage=False
        )
        guarded_closed, _ = _run_loop(
            self.WORKLOAD, honour_interlock=True, capacity_follows_stage=True
        )
        self.assertEqual(_transitions(guarded_closed), _transitions(guarded_open))

    def test_a_working_set_that_fits_both_pools_has_no_trap_to_spring(self):
        """The control on the control: at 40 000 held tokens the prefill pool
        is not the constraint, the flip commits, and the closed loop still
        matches its own replay. The guard is not simply refusing everything."""
        small = _Workload(rounds=240, held_tokens=40_000)
        open_regimes, _ = _run_loop(
            small, honour_interlock=True, capacity_follows_stage=False
        )
        closed_regimes, closed_commits = _run_loop(
            small, honour_interlock=True, capacity_follows_stage=True
        )
        self.assertEqual(_transitions(closed_regimes), _transitions(open_regimes))
        self.assertTrue(closed_commits, "the guard refused a flip that fits")
        self.assertEqual(closed_commits[0][1], "fp8-prefill")


# ---------------------------------------------------------------------------
# F3 -- the noise floor, per input signal
# ---------------------------------------------------------------------------


class TestF3NoiseFloor(CustomTestCase):
    """FALSIFIER F3. No threshold may sit inside its own signal's A-vs-A band.

    The band is measured the way the house measures it (#360): the largest
    difference an arm shows against its own repeat. The traces below stand in
    for two identical runs; on a card they are two boots, and the arithmetic
    that judges them is the same function.
    """

    #: Two "identical" runs of the prefill-share signal, jittering around 0.20.
    #: On a card these are two boots; here they stand in for them, and the
    #: arithmetic that judges them is the same function either way.
    RUN_A = [0.20, 0.22, 0.19, 0.21, 0.18, 0.23, 0.20, 0.21]
    RUN_A_REPEAT = [0.21, 0.20, 0.21, 0.19, 0.20, 0.21, 0.22, 0.19]

    #: The signal's own noise, as runs long enough for a windowed sensor to
    #: act on -- which is what makes it a threat rather than a wobble.
    NOISE_RUNS = ([0.19] * 4 + [0.22] * 4) * 24

    def test_the_band_is_measured_not_assumed(self):
        band = signal_band(self.RUN_A, self.RUN_A_REPEAT)
        # Largest disagreement the arm shows against its own repeat. Not a
        # constant chosen in advance: a guessed floor would decide the verdict
        # by the guess.
        self.assertAlmostEqual(band, 0.02, places=6)

    def test_the_shipped_threshold_gap_clears_the_measured_band(self):
        """``enter_prefill`` (0.35) against ``exit_prefill`` (0.15): the gap
        has to survive the band twice over, or the hysteresis is decorative."""
        band = signal_band(self.RUN_A, self.RUN_A_REPEAT)
        sensor = RegimeSensor()
        gap = sensor.enter_prefill - sensor.exit_prefill
        self.assertTrue(
            clears_band(gap, band),
            f"threshold gap {gap} does not clear 2x the measured band {band}",
        )

    def test_a_threshold_inside_its_band_flips_on_noise(self):
        """The failure F3 exists to catch, demonstrated rather than asserted.

        One trace, two sensors. The marks whose GAP is smaller than the
        signal's own noise track the noise; the shipped marks, whose gap is
        an order above it, do not move at all.
        """
        band = signal_band(self.RUN_A, self.RUN_A_REPEAT)

        def _count(enter, exit_):
            sensor = RegimeSensor(
                enter_prefill=enter,
                exit_prefill=exit_,
                enter_window=2,
                exit_window=4,
            )
            for share in self.NOISE_RUNS:
                p = int(round(share * 64))
                sensor.observe(
                    RegimeSample(
                        round_index=0,
                        prefill_rounds=p,
                        decode_rounds=64 - p,
                        held_tokens=0,
                        capacity_tokens=453_632,
                    )
                )
            return sensor.transitions

        # Marks straddling the noise with a 0.01 gap, against a 0.02 band.
        self.assertFalse(clears_band(0.01, band))
        inside = _count(0.21, 0.20)
        # The shipped marks: a 0.20 gap against the same band.
        shipped = _count(DEFAULT_ENTER_PREFILL, 0.15)
        self.assertTrue(clears_band(0.20, band))
        self.assertGreater(inside, 4, "the in-band sensor did not track the noise")
        self.assertEqual(shipped, 0, "the shipped marks moved on pure noise")

    def test_an_occupancy_threshold_is_not_re_derived(self):
        """The KV marks are #287's, so their band is #287's problem and not a
        second measurement that could disagree with the first."""
        from sglang.srt.model_executor import kv_pressure_ladder as ladder

        self.assertEqual(KV_ASCEND_MARK, ladder.DEFAULT_ASCEND_THRESHOLD)


# ---------------------------------------------------------------------------
# The observer: cadence, tiers, and the observe-only contract
# ---------------------------------------------------------------------------


def _min_reduce(payloads):
    return [min(vals) for vals in zip(*payloads)]


class _Channel:
    """A consensus channel with one synthetic peer.

    By default the peer MIRRORS this rank's proposal -- an agreeing group,
    which is the only way to exercise the happy path without also having to
    keep a second observer's epoch in step by hand. ``divergent_regime``
    rewrites exactly one field of the mirrored payload, so a desync test
    disagrees about the regime and about nothing else; ``peer_ms`` rewrites
    the timing field, which is the one field where disagreement is DATA.
    """

    def __init__(self, *, divergent_regime=None, peer_ms="mirror"):
        self.divergent_regime = divergent_regime
        self.peer_ms = peer_ms
        self.calls = 0

    def _peer(self, payload):
        peer = list(payload)
        if self.divergent_regime is not None:
            code = REGIME_CODES[self.divergent_regime]
            peer[0], peer[1] = code, -code
        if self.peer_ms != "mirror":
            q = (
                _MS_ABSENT
                if self.peer_ms is None
                else max(0, int(round(self.peer_ms / RANK_MS_QUANTUM)))
            )
            peer[6], peer[7] = q, -q
        return peer

    def __call__(self, payload):
        self.calls += 1
        return _min_reduce([payload, self._peer(payload)])


def _drive(observer, rounds, *, phase="prefill", held=0, capacity=453_632, ms=1.0):
    out = []
    for _ in range(rounds):
        rec = observer.on_round(
            phase=phase,
            held_tokens=held,
            capacity_tokens=capacity,
            running_bs=1,
            rank_forward_ms=ms,
        )
        if rec is not None:
            out.append(rec)
    return out


class TestObserverContract(CustomTestCase):
    def test_it_verdicts_only_on_the_replicated_cadence(self):
        obs = RegimeObserver(consensus_interval=8, tp_size=1)
        records = _drive(obs, 24)
        self.assertEqual([r["round"] for r in records], [8, 16, 24])

    def test_the_cadence_gate_is_the_round_counter_not_the_verdict(self):
        """Rule 2. A rank whose load looks completely different must still
        reach the boundary in the same round, or the collective is entered by
        a subset of the group."""
        a = RegimeObserver(consensus_interval=8, tp_size=1)
        b = RegimeObserver(consensus_interval=8, tp_size=1)
        rounds_a = [r["round"] for r in _drive(a, 32, phase="prefill")]
        # Deliberately IDLE, not decode: an idle rank must still enter the
        # collective in the same round, or a quiet rig splits the group.
        rounds_b = [r["round"] for r in _drive(b, 32, phase="idle")]
        self.assertEqual(rounds_a, rounds_b)

    def test_it_never_actuates(self):
        obs = RegimeObserver(consensus_interval=4, tp_size=1, table=_table())
        records = _drive(obs, 64, held=40_000)
        self.assertTrue(records)
        self.assertTrue(all(r["actuated"] is False for r in records))
        self.assertEqual(obs.summary()["actuations"], 0)

    def test_it_names_the_stage_it_would_have_selected(self):
        obs = RegimeObserver(consensus_interval=4, tp_size=1, table=_table())
        records = _drive(obs, 64, phase="prefill", held=40_000)
        self.assertTrue(any(r["would_flip_to"] == "fp8-prefill" for r in records))
        self.assertGreater(obs.summary()["proposals"], 0)

    def test_without_a_stage_table_it_reports_the_regime_and_says_so(self):
        obs = RegimeObserver(consensus_interval=4, tp_size=1, table=None)
        records = _drive(obs, 16, held=40_000)
        self.assertTrue(all(r["would_flip_to"] is None for r in records))
        self.assertIn("no stage table declared", records[0]["reason"])

    def test_the_interlock_refusal_is_reported_verbatim(self):
        """The observe run's most useful line: the regime asked for a stage
        and the arithmetic refused it. That is the phase-3 design input."""
        obs = RegimeObserver(consensus_interval=4, tp_size=1, table=_table())
        records = _drive(obs, 64, phase="prefill", held=90_000)
        refusals = [r for r in records if "96256" in r["reason"]]
        self.assertTrue(refusals, [r["reason"] for r in records])
        self.assertTrue(all(r["would_flip_to"] is None for r in refusals))

    def test_a_single_rank_needs_no_channel_and_reports_no_spread(self):
        obs = RegimeObserver(consensus_interval=4, tp_size=1)
        records = _drive(obs, 8)
        self.assertFalse(obs.uncoordinated)
        self.assertTrue(all(r["rank_ms_spread_pct"] is None for r in records))
        self.assertEqual(obs.summary()["consensus_rounds"], 0)

    def test_a_multi_rank_group_without_a_channel_says_it_is_unchecked(self):
        """Deliberately not the #287 refusal: nothing here acts, so a missing
        channel is a degraded observation rather than a hang risk -- but it
        must not read as a clean run."""
        obs = RegimeObserver(consensus_interval=4, tp_size=3, collective_min=None)
        _drive(obs, 8)
        self.assertTrue(obs.uncoordinated)
        self.assertTrue(obs.summary()["uncoordinated"])


class TestIdleIsNotPrefill(CustomTestCase):
    """The defect the 2026-08-01 card window found.

    The hook used to pass a BOOLEAN, computed by negating #287's
    "is this a decode round". An EMPTY batch fell on the prefill side, so a
    mostly-idle rig read PREFILL_HEAVY on 93 600 of 93 603 verdicts. The
    hermetic tests fed the boolean directly and therefore never exercised the
    mapping that produced it -- which is why this test drives the PHASE.
    """

    def test_an_idle_window_classifies_as_mixed(self):
        obs = RegimeObserver(consensus_interval=4, tp_size=1)
        records = [
            r
            for r in (
                obs.on_round(
                    phase="idle",
                    held_tokens=0,
                    capacity_tokens=453_632,
                    running_bs=0,
                )
                for _ in range(32)
            )
            if r is not None
        ]
        self.assertTrue(records)
        self.assertTrue(all(r["regime"] == REGIME_MIXED for r in records), records[0])
        self.assertIsNone(records[0]["prefill_share"])
        self.assertIsNone(records[0]["decode_share"])

    def test_idle_rounds_are_counted_but_in_neither_share(self):
        obs = RegimeObserver(consensus_interval=8, tp_size=1)
        for _ in range(4):
            obs.on_round(
                phase="prefill", held_tokens=0, capacity_tokens=1, running_bs=1
            )
        for _ in range(4):
            obs.on_round(phase="idle", held_tokens=0, capacity_tokens=1, running_bs=0)
        rec = obs.last_record
        self.assertEqual(rec["idle_rounds"], 4)
        # 4 prefill forwards out of 4 FORWARD rounds -- the idle ones are not
        # in the denominator either.
        self.assertEqual(rec["prefill_share"], 1.0)

    def test_an_unknown_phase_is_refused_by_name(self):
        obs = RegimeObserver(consensus_interval=2, tp_size=1)
        with self.assertRaises(ValueError) as cm:
            obs.on_round(phase="busy", held_tokens=0, capacity_tokens=1, running_bs=0)
        self.assertIn("not a kind of prefill", str(cm.exception))

    def test_the_scheduler_hook_maps_all_three_phases(self):
        """The mapping itself, driven -- not grepped.

        It moved out of the scheduler and into ``phase_of_last_batch`` with
        #388 precisely so it could be driven: the mapping is the thing that
        was wrong twice now, and a test that greps for its source text cannot
        tell a right mapping from a wrong one.
        """
        self.assertEqual(phase_of_last_batch(None), PHASE_IDLE)
        self.assertEqual(
            phase_of_last_batch(_Batch(ForwardMode.IDLE)),
            PHASE_IDLE,
        )
        self.assertEqual(
            phase_of_last_batch(_Batch(ForwardMode.EXTEND)),
            PHASE_PREFILL,
        )
        self.assertEqual(
            phase_of_last_batch(_Batch(ForwardMode.DECODE)),
            PHASE_DECODE,
        )


# ---------------------------------------------------------------------------
# #388 -- the retired-batch attribution, driven through a synthetic loop.
# ---------------------------------------------------------------------------


class _Batch:
    """The two fields the hook reads off a ``ScheduleBatch``, and no more.

    ``is_prefill_only`` is here because the PRE-#388 attribution read it, and
    a falsifier that cannot express the broken version cannot show it broken.
    It defaults False for the same reason it is False on a real generating
    batch: it means ``max_new_tokens == 0`` (and is forced False under spec),
    not "this round is a prefill".
    """

    def __init__(self, forward_mode, reqs=(), is_prefill_only=False):
        self.forward_mode = forward_mode
        self.reqs = list(reqs)
        self.is_prefill_only = is_prefill_only

    def is_empty(self):
        return not self.reqs

    def merge_batch(self, other):
        self.reqs.extend(other.reqs)
        self.is_prefill_only = self.is_prefill_only and other.is_prefill_only


def _old_attribution(running_batch, last_batch):
    """The hook exactly as it stood before #388, transcribed.

    Kept in the test rather than in the tree so the can-fail proof has
    something to fail on: this is the code whose output the gate windows
    recorded, and the assertion below is that it reports 0 % prefill on a run
    that is nothing but prefill.
    """
    if running_batch.is_empty():
        return PHASE_IDLE
    if running_batch.is_prefill_only:
        return PHASE_PREFILL
    return PHASE_DECODE


def _new_attribution(running_batch, last_batch):
    return phase_of_last_batch(last_batch)


def _drive_ticks(ticks, attribute):
    """Replay a sequence of scheduler ticks through one attribution.

    The ORDER is the whole point, and it is the scheduler's own
    (``get_next_batch_to_run``):

    1. a retired EXTEND batch is merged into ``running_batch`` -- which is why
       ``running_batch`` reads as a decode batch in the middle of a prefill
       burst;
    2. the regime hook runs;
    3. only THEN is this iteration's batch selected and dispatched, becoming
       ``last_batch`` for the next boundary.

    ``ticks`` names what each iteration dispatches: ``prefill`` an extend
    batch for one freshly admitted request, ``decode`` the running batch,
    ``idle`` nothing.
    """
    running = _Batch(ForwardMode.DECODE)
    last = None
    phases = []
    for i, tick in enumerate(ticks):
        if last is not None and last.forward_mode.is_extend():
            running.merge_batch(last)
        phases.append(attribute(running, last))
        if tick == "prefill":
            last = _Batch(ForwardMode.EXTEND, reqs=[f"req{i}"])
        elif tick == "decode":
            running.forward_mode = ForwardMode.DECODE
            last = None if running.is_empty() else running
        elif tick == "idle":
            last = None
        else:  # pragma: no cover -- a typo in a test plan, not a state
            raise AssertionError(f"unknown tick {tick!r}")
    return phases


def _share(phases):
    """``prefill_share`` as the observer computes it over one window."""
    obs = RegimeObserver(consensus_interval=len(phases), tp_size=1)
    record = None
    for phase in phases:
        record = (
            obs.on_round(
                phase=phase, held_tokens=0, capacity_tokens=453_632, running_bs=1
            )
            or record
        )
    return record["prefill_share"]


class TestRetiredBatchAttribution(CustomTestCase):
    """#388. The hook read the phase off the wrong batch.

    Gate 3 measured the consequence twice, independently: ``prefill_share``
    max **0.000** on both arms, 29 active boundaries each -- and 0.000 across
    all 34 954 boundaries of the window before it. ``enter_prefill = 0.35``
    was reported UNREACHED, which was true and was not the finding.
    """

    #: Eight admitted prompts, one extend forward each. Nothing else runs.
    BURST = ["prefill"] * 8
    #: A generation drain: the running batch decodes, no new arrivals.
    DRAIN = ["prefill"] + ["decode"] * 7
    #: Short arrivals during long generations -- the workload's `mixed` phase.
    MIXED = ["prefill", "decode"] * 4

    def test_the_unfixed_hook_reports_no_prefill_during_a_prefill_burst(self):
        """THE CAN-FAIL PROOF. A window of nothing but prefill forwards, and
        the pre-#388 attribution calls it 0 % prefill -- because the burst's
        own retired batches have been merged into ``running_batch`` by the
        time it looks, and ``is_prefill_only`` was never about phases."""
        phases = _drive_ticks(self.BURST, _old_attribution)
        self.assertNotIn(PHASE_PREFILL, phases)
        self.assertEqual(_share(phases), 0.0)

    def test_the_fixed_hook_reports_the_true_share_of_a_prefill_burst(self):
        phases = _drive_ticks(self.BURST, _new_attribution)
        # The first boundary has no retired batch yet: nothing had run.
        self.assertEqual(phases[0], PHASE_IDLE)
        self.assertEqual(phases[1:], [PHASE_PREFILL] * (len(self.BURST) - 1))
        self.assertEqual(_share(phases), 1.0)

    def test_a_decode_drain_stays_decode_under_both(self):
        """The regression the fix must not introduce. Decode was the ONE
        answer the broken hook got right, and it still has to be right --
        otherwise #388 would trade a silent 0 for a silent 1."""
        old = _drive_ticks(self.DRAIN, _old_attribution)
        new = _drive_ticks(self.DRAIN, _new_attribution)
        self.assertEqual(new.count(PHASE_DECODE), len(self.DRAIN) - 2)
        self.assertEqual(_share(new), 1.0 / (len(self.DRAIN) - 1))
        # ... and the old one saw a pure decode run here, which is why the
        # defect was invisible on a decode-dominated rig.
        self.assertEqual(_share(old), 0.0)
        self.assertIn(PHASE_DECODE, old)

    def test_a_mixed_window_reports_a_mixed_share(self):
        # One trailing tick that dispatches nothing, so the last dispatched
        # forward still gets its boundary -- the one-boundary lag, paid at the
        # end of a window rather than hidden.
        ticks = self.MIXED + ["idle"]
        self.assertEqual(_share(_drive_ticks(ticks, _new_attribution)), 0.5)
        # The broken hook flattens the same window to zero, so a mixed window
        # and a pure decode drain were indistinguishable in the traces.
        self.assertEqual(_share(_drive_ticks(ticks, _old_attribution)), 0.0)

    def test_an_idle_stretch_is_still_no_measurement(self):
        phases = _drive_ticks(["idle"] * 8, _new_attribution)
        self.assertEqual(phases, [PHASE_IDLE] * 8)
        self.assertIsNone(_share(phases))

    def test_a_spec_verify_is_decode_not_prefill(self):
        """``ForwardMode.is_extend()`` returns True for TARGET_VERIFY. Reading
        it as prefill would report the reference NEXTN recipe's decode drain
        -- the recipe both gate-3 arms booted -- as a prefill burst, which is
        the same defect with the sign flipped."""
        self.assertTrue(ForwardMode.TARGET_VERIFY.is_extend())
        self.assertEqual(
            phase_of_last_batch(_Batch(ForwardMode.TARGET_VERIFY)), PHASE_DECODE
        )
        self.assertEqual(
            phase_of_last_batch(_Batch(ForwardMode.DLLM_EXTEND)), PHASE_DECODE
        )
        for mode in (ForwardMode.MIXED, ForwardMode.SPLIT_PREFILL):
            self.assertEqual(phase_of_last_batch(_Batch(mode)), PHASE_PREFILL, mode)

    def test_every_dispatched_forward_is_attributed_exactly_once(self):
        """The one-boundary lag, pinned. The fix trades a structural zero for
        a shift of one boundary, and the claim that this is affordable rests
        on the shift being uniform -- no forward dropped, none counted twice.
        """
        ticks = self.MIXED + self.BURST + ["idle"] * 3 + self.DRAIN
        phases = _drive_ticks(ticks + ["idle"], _new_attribution)
        dispatched_prefill = ticks.count("prefill")
        self.assertEqual(phases.count(PHASE_PREFILL), dispatched_prefill)


class TestTierSplit(CustomTestCase):
    """The rank-local input reaches a decision only through the reduction."""

    def test_the_rank_ms_is_released_only_inside_the_payload(self):
        """A peer reporting a slower forward: the group spread is the output,
        and it exists only after the reduction. One reduction per boundary,
        not one per round."""
        channel = _Channel(peer_ms=20.0)
        obs = RegimeObserver(consensus_interval=4, tp_size=2, collective_min=channel)
        records = _drive(obs, 4, ms=10.0)
        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(records[0]["rank_ms_spread_pct"], 50.0, places=3)
        self.assertEqual(channel.calls, 1)

    def test_the_spread_reaches_the_sample_one_boundary_later(self):
        """Stated in the module docstring and pinned here: the classification
        that produced the proposal ran BEFORE the reduction, so the spread can
        only be an input to the NEXT boundary. Not a shortcut -- a property of
        the reduction, and phase 3 inherits it as a one-boundary-stale veto."""
        channel = _Channel(peer_ms=20.0)
        obs = RegimeObserver(consensus_interval=2, tp_size=2, collective_min=channel)
        first, second = _drive(obs, 4, ms=10.0)
        # Boundary 1: the sample went in blind, the reduction came back with
        # the spread.
        self.assertIsNone(first["sample_spread_pct"])
        self.assertAlmostEqual(first["rank_ms_spread_pct"], 50.0, places=3)
        # Boundary 2: the previous reduction is now an input.
        self.assertAlmostEqual(second["sample_spread_pct"], 50.0, places=3)

    def test_a_blind_rank_suppresses_the_spread_instead_of_faking_it(self):
        """A graph-covered forward reports nothing rather than a wrong zero;
        that absence must survive the reduction, or a blind rank reads as an
        infinitely fast one."""
        channel = _Channel(peer_ms=None)
        obs = RegimeObserver(consensus_interval=2, tp_size=2, collective_min=channel)
        rec = _drive(obs, 2, ms=10.0)[0]
        self.assertIsNone(rec["rank_ms_spread_pct"])

    def test_this_rank_being_blind_also_suppresses_the_spread(self):
        channel = _Channel(peer_ms=20.0)
        obs = RegimeObserver(consensus_interval=2, tp_size=2, collective_min=channel)
        rec = _drive(obs, 2, ms=None)[0]
        self.assertIsNone(rec["rank_mean_forward_ms"])
        self.assertIsNone(rec["rank_ms_spread_pct"])

    def test_no_rank_local_value_changes_the_classification(self):
        """The tier rule, end to end through the observer: two runs differing
        only in this rank's timing must classify identically."""
        fast = RegimeObserver(consensus_interval=4, tp_size=1, table=_table())
        slow = RegimeObserver(consensus_interval=4, tp_size=1, table=_table())
        a = _drive(fast, 32, held=40_000, ms=1.0)
        b = _drive(slow, 32, held=40_000, ms=999.0)
        self.assertEqual(
            [(r["regime"], r["would_flip_to"]) for r in a],
            [(r["regime"], r["would_flip_to"]) for r in b],
        )


class TestF5Desync(CustomTestCase):
    """FALSIFIER F5. A disagreement is counted and logged, never fatal."""

    @staticmethod
    def _diverging():
        # A peer that classified DECODE_HEAVY while this rank sees prefill,
        # agreeing about every other field -- so the reduction reports one
        # disagreement and it is the one under test.
        return _Channel(divergent_regime=REGIME_DECODE_HEAVY)

    def test_a_desync_is_counted_and_does_not_raise(self):
        obs = RegimeObserver(
            consensus_interval=2, tp_size=2, collective_min=self._diverging()
        )
        records = _drive(obs, 8, phase="prefill", ms=10.0)
        self.assertTrue(records)
        self.assertGreater(obs.desyncs, 0)
        self.assertFalse(records[-1]["agreed"])

    def test_the_desync_is_logged_loudly_with_the_gate_named(self):
        obs = RegimeObserver(
            consensus_interval=2, tp_size=2, collective_min=self._diverging()
        )
        with self.assertLogs(
            "sglang.srt.managers.regime_runtime", level=logging.WARNING
        ) as cm:
            _drive(obs, 2, phase="prefill", ms=10.0)
        joined = "\n".join(cm.output)
        self.assertIn("DESYNC", joined)
        self.assertIn("BLOCKS wiring any actuator", joined)

    def test_the_count_survives_into_the_summary(self):
        """It is the phase-3 gate, so it has to be readable at the end of a
        run rather than only in a log line somebody has to grep."""
        obs = RegimeObserver(
            consensus_interval=2, tp_size=2, collective_min=self._diverging()
        )
        _drive(obs, 8, phase="prefill", ms=10.0)
        self.assertEqual(obs.summary()["desyncs"], obs.desyncs)
        self.assertGreater(obs.summary()["desyncs"], 0)

    def test_agreeing_ranks_record_no_desync(self):
        """The happy path: a peer that classified the same way. The mirrored
        channel is what makes this checkable -- the epoch has to agree too,
        and a hand-built peer payload would drift out of step with it."""
        channel = _Channel()
        obs = RegimeObserver(consensus_interval=2, tp_size=2, collective_min=channel)
        records = _drive(obs, 16, ms=10.0)
        self.assertEqual(obs.desyncs, 0)
        self.assertTrue(records)
        self.assertTrue(all(r["agreed"] for r in records))
        self.assertEqual(channel.calls, len(records))


class TestF4DoNothingBaseline(CustomTestCase):
    """FALSIFIER F4, hermetic half.

    The card half -- off / observe / on over one workload, judged on ms/verify
    and ms/prefill against the off arm's own A-vs-A band, at equal or better
    capacity -- is phase 3. What can be settled here is the property that
    makes the comparison mean anything: the observe arm has to be a legitimate
    do-nothing baseline, i.e. differ from off only in what it writes down.
    """

    def test_the_observe_arm_reports_zero_actuations_by_construction(self):
        obs = RegimeObserver(consensus_interval=4, tp_size=1, table=_table())
        _drive(obs, 128, held=40_000)
        summary = obs.summary()
        self.assertEqual(summary["actuations"], 0)
        self.assertGreater(summary["proposals"], 0, "nothing was even proposed")

    def test_the_module_reaches_no_actuator(self):
        """Checked on the syntax tree, not on the prose: the module docstring
        NAMES the actuators it must not reach, and a substring grep would
        either fail on the documentation or force the documentation out."""
        import ast
        import pathlib

        import sglang.srt.managers.regime_runtime as mod

        tree = ast.parse(pathlib.Path(mod.__file__).read_text())
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
        for forbidden in ("kv_reshard", "vram_dial"):
            self.assertFalse(
                any(forbidden in m for m in modules),
                f"{forbidden} is imported: {sorted(modules)}",
            )
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        for forbidden in ("arm", "apply_budget_request", "throttle", "try_spill"):
            self.assertNotIn(forbidden, called)

    def test_off_is_the_default(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(observe_mode(), MODE_OFF)
        with mock.patch.dict(os.environ, {ENV_MODE: "0"}):
            self.assertEqual(observe_mode(), MODE_OFF)
        with mock.patch.dict(os.environ, {ENV_MODE: "1"}):
            self.assertEqual(observe_mode(), MODE_OBSERVE)
        with mock.patch.dict(os.environ, {ENV_MODE: "observe"}):
            self.assertEqual(observe_mode(), MODE_OBSERVE)

    def test_an_unknown_mode_is_refused_rather_than_rounded(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {ENV_MODE: "on"}):
            with self.assertRaises(ValueError) as cm:
                observe_mode()
            self.assertIn("not a known mode", str(cm.exception))

    def test_the_env_override_cannot_authorize_acting(self):
        """UPDATED DELIBERATELY by #363 phase 3: 'act' now exists as a mode,
        and the env override still may not select it. Acting is authorized by
        the entry gate at argument time, and an environment variable cannot
        carry evidence."""
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {ENV_MODE: "act"}):
            with self.assertRaises(ValueError) as cm:
                observe_mode()
            self.assertIn("cannot carry evidence", str(cm.exception))


class TestSensingAdapter(CustomTestCase):
    """The #252 reading, and its documented absences."""

    class _Log:
        def __init__(self, gpu_ms, split_known):
            self.last_gpu_ms = gpu_ms
            self.last_wait_ms = None
            self.last_split_known = split_known

    class _Reporter:
        def __init__(self, log):
            self.rank_prefill_log = log

    class _Sched:
        def __init__(self, reporter):
            self.metrics_reporter = reporter

    def test_a_measured_prefill_is_read(self):
        sched = self._Sched(self._Reporter(self._Log(41.5, True)))
        self.assertEqual(rank_forward_ms_from(sched), 41.5)

    def test_a_graph_covered_forward_reads_as_absent_not_as_zero(self):
        """DESIGN_363 section 7.1: the split is prefill-only and reports
        nothing under a replayed graph. The adapter must carry that absence,
        because a zero here is a rank that looks infinitely fast."""
        sched = self._Sched(self._Reporter(self._Log(41.5, False)))
        self.assertIsNone(rank_forward_ms_from(sched))

    def test_a_scheduler_without_the_reporter_is_absent_not_an_error(self):
        class _Bare:
            pass

        self.assertIsNone(rank_forward_ms_from(_Bare()))


class TestSchedulerHookContract(CustomTestCase):
    """The wiring itself, checked on the source rather than by booting a
    server: the default path must cost one attribute compare and nothing else,
    and the hook must sit at the same between-tick boundary as #287/#364.
    """

    @staticmethod
    def _scheduler_src():
        import pathlib

        import sglang.srt.managers.scheduler as mod

        return pathlib.Path(mod.__file__).read_text()

    def test_the_attributes_default_to_none(self):
        src = self._scheduler_src()
        self.assertIn("self.regime_observer = None", src)
        self.assertIn("self._regime_observer_mode = None", src)

    def test_the_hook_is_gated_before_anything_is_built(self):
        """Unset env: the mode resolves once to 'off' and the round-loop cost
        is one compare. Nothing is imported, nothing is constructed."""
        src = self._scheduler_src()
        self.assertIn('if self._regime_observer_mode != "off":', src)
        gate_at = src.index('if self._regime_observer_mode != "off":')
        build_at = src.index("build_regime_observer(self)")
        self.assertLess(gate_at, build_at, "the build is not behind the gate")

    def test_the_hook_sits_at_the_between_tick_boundary(self):
        """Between the #364 executor and the batch selection -- the previous
        forward is retired (so its device timing is complete) and the next
        batch is not chosen yet."""
        src = self._scheduler_src()
        gdn_at = src.index("self.gdn_slot_executor.on_round(")
        regime_at = src.index("self.regime_observer.on_round(")
        # The batch-selection chain of THIS loop, i.e. the first one after the
        # hook -- the string also occurs earlier in the file.
        select_at = src.index("if self.dllm_config is not None:", regime_at)
        self.assertLess(gdn_at, regime_at)
        self.assertLess(regime_at, select_at)

    def test_the_hook_passes_only_replicated_state_plus_the_named_rank_local(self):
        """Every keyword the scheduler hands the observer is tier-R except
        the named rank-local ones. A new keyword here needs a tier decision,
        and this test is where that decision gets made deliberately.

        UPDATED DELIBERATELY (#363 intra-phase axis). Two keywords joined the
        list: ``rank_compute_ms`` and ``rank_wait_ms``. The tier decision is
        that both are TIER-L, exactly like ``rank_forward_ms`` -- they are the
        same retired forward's device time, split into the two terms, and they
        are subject to the same discipline: accumulated rank-locally, never
        branched on locally, and released only through a MIN reduction
        (``regime_ms_clock.pack_ms_sample``). The ms clock that consumes them
        is fed the GROUP statistic, never this rank's own, which is why adding
        a rank-local input here does not add a rank-local decision.

        Why a split rather than reusing the existing total: the stage axes the
        intra-phase controller moves (#297 KV vector, #330 budget) change how
        work is DIVIDED, not how fast a GEMM runs, so the wait term is the
        only part of the round they can be credited against. A total would
        hide it."""
        import ast
        import pathlib

        import sglang.srt.managers.scheduler as mod

        tree = ast.parse(pathlib.Path(mod.__file__).read_text())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "on_round"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "regime_observer"
        ]
        self.assertEqual(len(calls), 1, "expected exactly one observer hook")
        passed = {kw.arg for kw in calls[0].keywords}
        self.assertEqual(
            passed,
            {
                "phase",
                "held_tokens",
                "capacity_tokens",
                "running_bs",
                "queued_reqs",
                "queued_prompt_tokens",
                "max_queued_prompt_tokens",
                "rank_forward_ms",
                # Tier-L, #363 intra-phase. See the docstring above.
                "rank_compute_ms",
                "rank_wait_ms",
            },
        )
        self.assertEqual(calls[0].args, [], "the hook must be keyword-only")

    def test_the_phase_is_attributed_to_the_retired_batch(self):
        """UPDATED DELIBERATELY, twice, and this is the second time.

        The 2026-08-01 gate window made the split three-way (idle stopped
        falling on the prefill side). #388 changed WHICH BATCH it is read off:
        ``running_batch`` has already absorbed the finished extend batch by
        the time the hook runs, and ``is_prefill_only`` is a request kind
        rather than an execution phase, so the prefill phase was unreportable.
        The hook now hands the observer the phase of ``last_batch``.

        Pinned on the source because it is a wiring property: which argument
        reaches the hook, not what the hook does with it.
        """
        import ast
        import pathlib

        import sglang.srt.managers.scheduler as mod

        src = pathlib.Path(mod.__file__).read_text()
        tree = ast.parse(src)
        call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "on_round"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "regime_observer"
        )
        phase = next(kw.value for kw in call.keywords if kw.arg == "phase")
        # phase=phase_of_last_batch(last_batch), and nothing else.
        self.assertIsInstance(phase, ast.Call)
        self.assertEqual(phase.func.id, "phase_of_last_batch")
        self.assertEqual([a.id for a in phase.args], ["last_batch"])
        # The pre-#388 reading must not come back by another route.
        self.assertNotIn("prefill_active = not (", src)
        self.assertNotIn('phase = "prefill"', src)
        # And the provenance of `last_batch` itself, which is the one link in
        # the chain the hermetic driver stands in for: both event loops
        # publish the batch they dispatched, and only that.
        self.assertEqual(src.count("self.last_batch = batch"), 2)
        self.assertIn(
            "plan = self.get_next_batch_to_run(\n"
            "                running_batch=self.running_batch, "
            "last_batch=self.last_batch\n"
            "            )",
            src,
        )


if __name__ == "__main__":
    unittest.main()
