"""Unit tests for #156-4: cross-algo lazy single capture + ctx retirement.

CPU-only tests of the pure decision logic in
``sglang.srt.speculative.cross_algo_lazy``:

* the ctx-collapse RETIREMENT rule, including the ctx-vs-content
  distinction that keeps a prose passage from killing DFLASH for later code,
  and its monotonicity;
* the lazy-capture PHASE MACHINE (steady -> warm-up -> measure -> adopt or
  return), the capture/warm-keep/eager outputs of each phase, and the
  invariant that no family change ever bypasses the warm-up;
* the SIGNAL GATE (probe only when NEXTN's own accept EMA says the content
  is structured), its hysteresis, and the improve-since-failed-probe rule;
* parameter parsing/validation and the flag-OFF equivalences.

GPU behavior (the actual capture toggle and the eager verify fallback) is
covered by the live validation protocol in PLAN_crossalgo_lazy_capture.md,
not here.
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload, relay_field
from sglang.srt.speculative.adaptive_spec_params import RungMetrics
from sglang.srt.speculative.draft_worker_common import make_draft_input_v2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.cross_algo_utils import (
    _resolve_lazy_stash,
    _resolve_retire_stash,
)
from sglang.srt.speculative.cross_algo_lazy import (
    DECIDE_END_MEASURE,
    DECIDE_ENTER_MEASURE,
    DECIDE_RETIRE_EVICT,
    DECIDE_STEADY_K,
    LAZY_ENV_PREFIX,
    PHASE_MEASURE,
    PHASE_STEADY,
    PHASE_WARMUP,
    RETIRE_CTX_COLLAPSE,
    RETIRE_CTX_HARD,
    RETIRE_DISABLED,
    RETIRE_KEEP_CONTENT,
    RETIRE_KEEP_MEASURED,
    RETIRE_KEEP_NO_DATA,
    LazyCaptureConfig,
    LazyCaptureController,
    RetirePolicy,
    parse_retire_ctx_value,
    resolve_retire_policy,
    retire_verdict,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _cfg(**kw):
    base = dict(
        warmup_rounds=5,
        probe_window_rounds=10,
        probe_interval_rounds=100,
        min_dwell_rounds=20,
        decide_interval_rounds=16,
    )
    base.update(kw)
    return LazyCaptureConfig(**base)


def _retire(collapse=10000, **kw):
    base = dict(collapse_ctx=collapse, band_frac=0.8, accept_ratio=1.0)
    base.update(kw)
    return RetirePolicy(**base)


def _run(ctl, rounds, ctx=1000, nextn=4.0, dflash=6.0, commit=None):
    """Drive the controller for *rounds* rounds, auto-committing decisions.

    ``commit`` picks the family for an end-of-measure decision (the rank-0
    part); by default the candidate is adopted. Returns the list of
    (round, verdict) pairs where a decision was due.
    """
    events = []
    for r in range(ctl._round + 1, ctl._round + 1 + rounds):
        ctl.observe(r, ctx, nextn, dflash)
        v = ctl.step(r)
        if v.decision_due:
            if v.decision_kind == DECIDE_END_MEASURE:
                fam = commit(v) if commit else v.candidate
            else:
                fam = v.family_target or v.adopted
            ctl.commit(fam, r)
            events.append((r, v))
    return events


class TestRetireVerdict(CustomTestCase):
    def test_disabled_policy_never_retires(self):
        # The default: no measurement exists yet, so the whole retirement
        # path must be unreachable -- exactly today's behavior.
        pol = RetirePolicy()
        self.assertFalse(pol.enabled)
        for ctx in (0, 4096, 100000):
            self.assertEqual(
                retire_verdict(ctx, 1.0, 9.0, pol), (False, RETIRE_DISABLED)
            )

    def test_low_accept_at_low_ctx_is_content_not_collapse(self):
        # THE trap: DFLASH is bad on prose at EVERY context length. Retiring
        # inside a prose passage would kill it for the code that follows.
        pol = _retire(10000)
        self.assertEqual(pol.band_low, 8000)
        for ctx in (0, 1000, 4096, 7999):
            self.assertEqual(
                retire_verdict(ctx, 2.9, 4.3, pol),
                (False, RETIRE_KEEP_CONTENT),
                f"retired at ctx {ctx} on a content-driven accept drop",
            )

    def test_low_accept_inside_the_band_is_ctx_collapse(self):
        pol = _retire(10000)
        retire, reason = retire_verdict(8500, 3.0, 4.3, pol)
        self.assertTrue(retire)
        self.assertEqual(reason, RETIRE_CTX_COLLAPSE)

    def test_healthy_accept_inside_the_band_keeps_dflash(self):
        pol = _retire(10000)
        self.assertEqual(
            retire_verdict(8500, 6.0, 4.3, pol), (False, RETIRE_KEEP_MEASURED)
        )

    def test_missing_accept_data_inside_the_band_keeps_dflash(self):
        pol = _retire(10000)
        self.assertEqual(
            retire_verdict(8500, None, 4.3, pol), (False, RETIRE_KEEP_NO_DATA)
        )
        self.assertEqual(
            retire_verdict(8500, 6.0, None, pol), (False, RETIRE_KEEP_NO_DATA)
        )

    def test_beyond_the_collapse_ctx_retires_without_data(self):
        pol = _retire(10000)
        self.assertEqual(
            retire_verdict(10000, None, None, pol), (True, RETIRE_CTX_HARD)
        )
        self.assertEqual(
            retire_verdict(50000, 9.0, 1.0, pol), (True, RETIRE_CTX_HARD)
        )

    def test_accept_ratio_is_relative_to_the_nextn_baseline(self):
        # ratio 0.5: DFLASH may fall to half the baseline before retiring.
        pol = _retire(10000, accept_ratio=0.5)
        self.assertFalse(retire_verdict(8500, 2.5, 4.0, pol)[0])
        self.assertTrue(retire_verdict(8500, 1.9, 4.0, pol)[0])


class TestRetireResolution(CustomTestCase):
    def test_parse_values(self):
        self.assertEqual(parse_retire_ctx_value(None), "off")
        self.assertEqual(parse_retire_ctx_value("off"), "off")
        self.assertEqual(parse_retire_ctx_value("AUTO"), "auto")
        self.assertEqual(parse_retire_ctx_value("9000"), 9000)
        with self.assertRaises(ValueError):
            parse_retire_ctx_value("0")
        with self.assertRaises(ValueError):
            parse_retire_ctx_value("nonsense")

    def test_default_is_disabled(self):
        pol = resolve_retire_policy(None, 2048)
        self.assertFalse(pol.enabled)
        self.assertIn("off", pol.source)

    def test_auto_without_a_measured_factor_stays_disabled(self):
        # NO GUESSED DEFAULT: the collapse point is unmeasured, so 'auto'
        # without an explicit factor must degrade to today's behavior and
        # SAY so, never to a made-up number.
        os.environ.pop("SGLANG_CROSS_RETIRE_CTX_FACTOR", None)
        pol = resolve_retire_policy("auto", 2048)
        self.assertFalse(pol.enabled)
        self.assertIn("UNMEASURED", pol.source)

    def test_auto_with_factor_derives_from_the_sliding_window(self):
        os.environ["SGLANG_CROSS_RETIRE_CTX_FACTOR"] = "4.5"
        try:
            pol = resolve_retire_policy("auto", 2048)
        finally:
            del os.environ["SGLANG_CROSS_RETIRE_CTX_FACTOR"]
        self.assertEqual(pol.collapse_ctx, 9216)
        self.assertEqual(pol.band_low, int(0.8 * 9216))

    def test_auto_without_a_sliding_window_stays_disabled(self):
        os.environ["SGLANG_CROSS_RETIRE_CTX_FACTOR"] = "4.5"
        try:
            pol = resolve_retire_policy("auto", None)
        finally:
            del os.environ["SGLANG_CROSS_RETIRE_CTX_FACTOR"]
        self.assertFalse(pol.enabled)

    def test_explicit_token_count(self):
        pol = resolve_retire_policy("8500", 2048)
        self.assertEqual(pol.collapse_ctx, 8500)
        self.assertTrue(pol.enabled)

    def test_stash_roundtrip(self):
        pol = resolve_retire_policy("8500", 2048)
        again = RetirePolicy.from_stash(pol.to_stash())
        self.assertEqual(pol, again)
        self.assertFalse(RetirePolicy.from_stash(None).enabled)


class TestRetireLatchIsMonotone(CustomTestCase):
    def test_latch_never_clears(self):
        ctl = LazyCaptureController(_cfg(), retire=_retire(10000))
        ctl.observe(1, 1000, 4.3, 6.0)
        self.assertFalse(ctl.retired)
        ctl.observe(2, 12000, 4.3, 6.0)
        self.assertTrue(ctl.retired)
        self.assertEqual(ctl.retire_reason, RETIRE_CTX_HARD)
        # A later batch back at low context (a fresh short request) must NOT
        # un-retire: ctx is monotone WITHIN a session, and the whole point of
        # the permanent latch is that the tail pays no dual tax any more.
        ctl.observe(3, 500, 4.3, 9.0)
        self.assertTrue(ctl.retired)
        self.assertEqual(ctl.retire_round, 2)

    def test_prose_at_low_ctx_does_not_latch(self):
        ctl = LazyCaptureController(_cfg(), retire=_retire(10000))
        for r in range(1, 200):
            ctl.observe(r, 2000, 3.6, 2.9)  # prose: DFLASH clearly worse
        self.assertFalse(ctl.retired)


class TestPhaseMachine(CustomTestCase):
    def test_steady_state_is_single_capture_and_no_warmkeep(self):
        ctl = LazyCaptureController(_cfg())
        ctl.observe(1, 1000, 4.0, 6.0)
        v = ctl.step(1)
        self.assertEqual(v.phase, PHASE_STEADY)
        self.assertEqual(v.running, "nextn")
        self.assertFalse(v.aux_capture)  # NEXTN needs no aux concat
        self.assertFalse(v.warmkeep)  # the expensive half of the tax
        self.assertFalse(v.eager)  # the aux-OFF NEXTN graph set replays

    def test_steady_dflash_keeps_aux_but_still_no_warmkeep(self):
        ctl = LazyCaptureController(_cfg(), initial_family="dflash")
        ctl.observe(1, 1000, 4.0, 6.0)
        v = ctl.step(1)
        self.assertTrue(v.aux_capture)  # DFLASH's own input
        self.assertFalse(v.warmkeep)
        # The DFLASH graph set is baked aux-ON, so replay is fine.
        self.assertFalse(v.eager)

    def test_probe_window_shape(self):
        cfg = _cfg(warmup_rounds=5, probe_window_rounds=10)
        ctl = LazyCaptureController(cfg)
        events = _run(ctl, 140)
        kinds = [v.decision_kind for _r, v in events]
        self.assertIn(DECIDE_ENTER_MEASURE, kinds)
        self.assertIn(DECIDE_END_MEASURE, kinds)
        enter = next(r for r, v in events if v.decision_kind == DECIDE_ENTER_MEASURE)
        end = next(r for r, v in events if v.decision_kind == DECIDE_END_MEASURE)
        self.assertEqual(end - enter, cfg.probe_window_rounds)

    def test_warmup_precedes_every_family_change(self):
        # Invariant: a direct steady->switch would hand the incoming rung a
        # cold draft KV AND an unpopulated draft-seed field.
        ctl = LazyCaptureController(_cfg())
        seen_warmup = False
        prev_running = "nextn"
        for r in range(1, 300):
            ctl.observe(r, 1000, 4.0, 6.0)
            v = ctl.step(r)
            if v.decision_due:
                fam = (
                    v.candidate
                    if v.decision_kind == DECIDE_END_MEASURE
                    else (v.family_target or v.adopted)
                )
                ctl.commit(fam, r)
                v = ctl.step(r)
            if v.phase == PHASE_WARMUP:
                seen_warmup = True
            if v.running != prev_running:
                self.assertTrue(
                    seen_warmup, f"family changed at round {r} without a warm-up"
                )
                seen_warmup = False
                prev_running = v.running
        self.assertGreater(ctl.window_count, 0)

    def test_window_runs_dual_and_the_nextn_direction_runs_eager(self):
        ctl = LazyCaptureController(_cfg())
        saw_warmup_eager = False
        for r in range(1, 130):
            ctl.observe(r, 1000, 4.0, 6.0)
            v = ctl.step(r)
            if v.phase == PHASE_WARMUP:
                # incumbent NEXTN still running, DFLASH being re-primed
                self.assertTrue(v.aux_capture)
                self.assertTrue(v.warmkeep)
                self.assertTrue(v.eager)  # HAKEN 1: aux-OFF graphs are dead here
                saw_warmup_eager = True
            if v.decision_due:
                fam = (
                    v.candidate
                    if v.decision_kind == DECIDE_END_MEASURE
                    else (v.family_target or v.adopted)
                )
                ctl.commit(fam, r)
        self.assertTrue(saw_warmup_eager)

    def test_measure_phase_runs_the_candidate(self):
        ctl = LazyCaptureController(_cfg())
        _run(ctl, 106)  # into the window
        self.assertEqual(ctl.phase, PHASE_MEASURE)
        v = ctl.step(ctl._round)
        self.assertEqual(v.running, "dflash")
        self.assertEqual(v.adopted, "nextn")
        # DFLASH runs on its own aux-ON graph set -> no eager needed.
        self.assertFalse(v.eager)

    def test_probe_return_restores_the_incumbent(self):
        ctl = LazyCaptureController(_cfg())
        _run(ctl, 200, commit=lambda v: v.adopted)  # rank 0: candidate lost
        self.assertEqual(ctl.adopted, "nextn")
        self.assertEqual(ctl.phase, PHASE_STEADY)
        self.assertEqual(ctl.adopt_count, 0)

    def test_probe_adopt_switches_the_family(self):
        ctl = LazyCaptureController(_cfg())
        _run(ctl, 120, commit=lambda v: v.candidate)
        self.assertEqual(ctl.adopted, "dflash")
        self.assertEqual(ctl.adopt_count, 1)

    def test_refused_switch_aborts_the_window(self):
        # The worker's first-boot swap guard can decline the incoming rung.
        # Measuring the incumbent against itself is meaningless.
        ctl = LazyCaptureController(_cfg())
        for r in range(1, 200):
            ctl.observe(r, 1000, 4.0, 6.0)
            v = ctl.step(r)
            if v.decision_due:
                ctl.commit("nextn", r)  # the switch never happened
                if v.decision_kind == DECIDE_ENTER_MEASURE:
                    self.assertEqual(ctl.phase, PHASE_STEADY)
                    self.assertIsNone(ctl.candidate)
                    return
        self.fail("no probe window was ever opened")

    def test_steady_k_decisions_run_on_the_cheap_cadence(self):
        cfg = _cfg(decide_interval_rounds=16, probe_interval_rounds=10**6)
        ctl = LazyCaptureController(cfg)
        events = _run(ctl, 100)
        kinds = {v.decision_kind for _r, v in events}
        self.assertEqual(kinds, {DECIDE_STEADY_K})
        rounds = [r for r, _v in events]
        self.assertEqual(rounds, [16, 32, 48, 64, 80, 96])

    def test_uncommitted_decision_is_re_emitted_not_skipped(self):
        # A dropped commit must never let one rank's phase run ahead of the
        # broadcast -- that is the NCCL-hang shape.
        ctl = LazyCaptureController(_cfg())
        for r in range(1, 200):
            ctl.observe(r, 1000, 4.0, 6.0)
            v = ctl.step(r)
            if v.decision_due:
                again = ctl.step(r)
                self.assertEqual(again.decision_kind, v.decision_kind)
                self.assertEqual(again.family_target, v.family_target)
                return
        self.fail("no decision was ever due")


class TestRetirementInTheController(CustomTestCase):
    def test_retired_dflash_is_never_probed_again(self):
        ctl = LazyCaptureController(_cfg(), retire=_retire(10000))
        ctl.observe(1, 12000, 4.3, 6.0)
        self.assertTrue(ctl.retired)
        _run(ctl, 2000, ctx=12000)
        self.assertEqual(ctl.window_count, 0)
        self.assertEqual(ctl.adopted, "nextn")

    def test_retired_dflash_drops_capture_and_warmkeep(self):
        ctl = LazyCaptureController(_cfg(warmkeep_stride=4), retire=_retire(10000))
        ctl.observe(1, 12000, 4.3, 6.0)
        for r in range(2, 40):
            ctl.observe(r, 12000, 4.3, 6.0)
            v = ctl.step(r)
            self.assertFalse(v.aux_capture, f"aux still on at round {r}")
            self.assertFalse(v.warmkeep, f"warm-keep still on at round {r}")
            self.assertFalse(v.eager)

    def test_retirement_while_dflash_is_adopted_evicts_through_a_warmup(self):
        ctl = LazyCaptureController(
            _cfg(), retire=_retire(10000), initial_family="dflash"
        )
        events = _run(ctl, 40, ctx=12000)
        kinds = [v.decision_kind for _r, v in events]
        self.assertIn(DECIDE_RETIRE_EVICT, kinds)
        self.assertEqual(ctl.adopted, "nextn")
        # ... and the eviction went through a warm-up window, so the
        # incoming NEXTN rung's draft KV was re-primed first.
        self.assertEqual(ctl.window_count, 1)


class TestSignalGate(CustomTestCase):
    def test_low_signal_suppresses_probing(self):
        # Mixed session: a code burst sets the peak, then prose. The prose
        # stretch must not pay the dual tax for a DFLASH probe that cannot
        # win there (RESULT_draft_crossover.md, kernaussage 3). The CONTROL
        # arm is the identical run with the code signal kept up -- without it
        # the assertion would pass vacuously.
        def run(signal):
            ctl = LazyCaptureController(_cfg())
            # Adopt on every probe so the improve-since-failed-probe rule
            # never fires; the signal gate is what is under test here.
            _run(ctl, 30, nextn=4.3, commit=lambda v: v.candidate)
            _run(ctl, 1000, nextn=signal, commit=lambda v: v.candidate)
            return ctl.window_count

        prose = run(2.9)  # 0.67 of the peak -> below the low threshold
        code = run(4.3)  # at the peak
        self.assertEqual(prose, 0)
        self.assertGreater(code, 1)

    def test_high_signal_after_a_prose_stretch_re_enables_probing(self):
        ctl = LazyCaptureController(_cfg())
        _run(ctl, 30, nextn=4.3, commit=lambda v: v.candidate)
        _run(ctl, 300, nextn=2.9, commit=lambda v: v.candidate)
        windows_after_prose = ctl.window_count
        self.assertEqual(windows_after_prose, 0)
        _run(ctl, 300, nextn=4.3, commit=lambda v: v.candidate)  # code is back
        self.assertGreater(ctl.window_count, windows_after_prose)

    def test_hysteresis_band_does_not_flap(self):
        cfg = _cfg(signal_high_frac=0.9, signal_low_frac=0.75)
        ctl = LazyCaptureController(cfg)
        ctl.observe(1, 1000, 4.0, 6.0)
        self.assertTrue(ctl._signal_high)
        ctl.observe(2, 1000, 3.3, 6.0)  # 0.825 -- inside the band
        self.assertTrue(ctl._signal_high, "left the high state inside the band")
        ctl.observe(3, 1000, 2.9, 6.0)  # 0.725 -- below low
        self.assertFalse(ctl._signal_high)
        ctl.observe(4, 1000, 3.3, 6.0)  # 0.825 -- inside the band again
        self.assertFalse(ctl._signal_high, "re-armed inside the band")
        ctl.observe(5, 1000, 3.7, 6.0)  # 0.925 -- above high
        self.assertTrue(ctl._signal_high)

    def test_failed_probe_needs_an_improved_signal_to_repeat(self):
        cfg = _cfg(reprobe_improve_frac=0.05)
        ctl = LazyCaptureController(cfg)
        # First probe loses at signal 4.0.
        _run(ctl, 120, nextn=4.0, commit=lambda v: v.adopted)
        after_first = ctl.window_count
        # Same signal forever: no point asking again.
        _run(ctl, 2000, nextn=4.0, commit=lambda v: v.adopted)
        self.assertEqual(ctl.window_count, after_first)
        # A genuinely better signal re-opens the question.
        _run(ctl, 400, nextn=4.5, commit=lambda v: v.adopted)
        self.assertGreater(ctl.window_count, after_first)

    def test_min_dwell_and_probe_interval_bound_the_cadence(self):
        cfg = _cfg(min_dwell_rounds=20, probe_interval_rounds=100)
        ctl = LazyCaptureController(cfg)
        # Adopt every probe, so probing is never suppressed by the
        # improve-since-failed-probe rule and only the cadence guards bind.
        events = _run(ctl, 1000, commit=lambda v: v.candidate)
        starts = [r for r, v in events if v.decision_kind == DECIDE_ENTER_MEASURE]
        self.assertGreater(len(starts), 3, "cadence never exercised")
        ends = [r for r, v in events if v.decision_kind == DECIDE_END_MEASURE]
        for end, nxt in zip(ends, starts[1:]):
            self.assertGreaterEqual(
                nxt - end,
                cfg.probe_interval_rounds,
                "a window opened before probe_interval_rounds had elapsed",
            )
            self.assertGreaterEqual(
                nxt - end,
                cfg.min_dwell_rounds,
                "a window opened before the family dwell had elapsed",
            )

    def test_absolute_floor_blocks_probing_when_configured(self):
        ctl = LazyCaptureController(_cfg(signal_min_accept=4.0))
        _run(ctl, 1000, nextn=3.5)
        self.assertEqual(ctl.window_count, 0)


class TestWarmkeepStride(CustomTestCase):
    def test_stride_zero_is_full_lazy(self):
        ctl = LazyCaptureController(_cfg(warmkeep_stride=0))
        for r in range(1, 20):
            ctl.observe(r, 1000, 4.0, 6.0)
            self.assertFalse(ctl.step(r).warmkeep)

    def test_stride_n_keeps_every_nth_round_warm(self):
        ctl = LazyCaptureController(_cfg(warmkeep_stride=4))
        warm = []
        for r in range(1, 21):
            ctl.observe(r, 1000, 4.0, 6.0)
            v = ctl.step(r)
            if v.phase == PHASE_STEADY and v.warmkeep:
                warm.append(r)
        self.assertEqual(warm, [4, 8, 12, 16, 20])

    def test_stride_rounds_in_the_nextn_steady_state_need_eager(self):
        # The NEXTN target-verify graphs are baked aux-OFF, so a warm-keep
        # round there cannot replay them.
        ctl = LazyCaptureController(_cfg(warmkeep_stride=4))
        ctl.observe(4, 1000, 4.0, 6.0)
        v = ctl.step(4)
        self.assertTrue(v.warmkeep)
        self.assertTrue(v.aux_capture)
        self.assertTrue(v.eager)

    def test_stride_rounds_in_the_dflash_steady_state_replay_graphs(self):
        ctl = LazyCaptureController(
            _cfg(warmkeep_stride=4), initial_family="dflash"
        )
        ctl.observe(4, 1000, 4.0, 6.0)
        v = ctl.step(4)
        self.assertTrue(v.warmkeep)
        self.assertFalse(v.eager)  # the DFLASH set already emits both


class TestConfig(CustomTestCase):
    def test_env_override(self):
        os.environ[LAZY_ENV_PREFIX + "WARMUP_ROUNDS"] = "17"
        os.environ[LAZY_ENV_PREFIX + "SIGNAL_HIGH_FRAC"] = "0.95"
        try:
            cfg = LazyCaptureConfig.from_env()
        finally:
            del os.environ[LAZY_ENV_PREFIX + "WARMUP_ROUNDS"]
            del os.environ[LAZY_ENV_PREFIX + "SIGNAL_HIGH_FRAC"]
        self.assertEqual(cfg.warmup_rounds, 17)
        self.assertAlmostEqual(cfg.signal_high_frac, 0.95)

    def test_env_overrides_the_seeded_bandit_values(self):
        os.environ[LAZY_ENV_PREFIX + "MIN_DWELL_ROUNDS"] = "7"
        try:
            cfg = LazyCaptureConfig.from_env(min_dwell_rounds=64)
        finally:
            del os.environ[LAZY_ENV_PREFIX + "MIN_DWELL_ROUNDS"]
        self.assertEqual(cfg.min_dwell_rounds, 7)

    def test_validation_rejects_a_broken_hysteresis_band(self):
        with self.assertRaises(ValueError):
            LazyCaptureConfig(signal_high_frac=0.5, signal_low_frac=0.9).validate()
        with self.assertRaises(ValueError):
            LazyCaptureConfig(warmup_rounds=0).validate()
        with self.assertRaises(ValueError):
            LazyCaptureConfig(warmkeep_stride=-1).validate()


class TestRankUniformInputs(CustomTestCase):
    def test_accept_len_accessor_is_the_rank_uniform_half_of_rungmetrics(self):
        m = RungMetrics()
        rung = ("nextn", 3)
        self.assertIsNone(m.accept_len(rung))
        for _ in range(10):
            m.observe(rung, [2, 2], batch_size=2, now=0.0)
        # accept_len needs only accept counts (broadcast from rank 0),
        # while reward()/round_s() need wall clock -- which is exactly why
        # the lazy controller consumes the former and never the latter.
        self.assertAlmostEqual(m.accept_len(rung), 3.0)
        self.assertIsNone(m.round_s(rung, min_time_samples=3))
        self.assertIsNone(m.accept_len(rung, min_samples=11))

    def test_two_controllers_with_identical_inputs_stay_in_lockstep(self):
        # The rank-uniformity proof in miniature: two "ranks" fed the same
        # rank-uniform observations must produce identical decision rounds,
        # identical kinds and identical phases -- otherwise the ranks would
        # reach different collectives.
        a = LazyCaptureController(_cfg(), retire=_retire(10000))
        b = LazyCaptureController(_cfg(), retire=_retire(10000))
        ctxs = [500 + 40 * i for i in range(400)]
        accepts = [4.3 if (i // 50) % 2 == 0 else 2.9 for i in range(400)]
        trace_a, trace_b = [], []
        for i, r in enumerate(range(1, 401)):
            for ctl, trace in ((a, trace_a), (b, trace_b)):
                ctl.observe(r, ctxs[i], accepts[i], 6.0)
                v = ctl.step(r)
                trace.append(
                    (v.phase, v.running, v.aux_capture, v.warmkeep, v.eager,
                     v.decision_kind, v.retired)
                )
                if v.decision_due:
                    fam = (
                        v.candidate
                        if v.decision_kind == DECIDE_END_MEASURE
                        else (v.family_target or v.adopted)
                    )
                    ctl.commit(fam, r)
        self.assertEqual(trace_a, trace_b)
        self.assertEqual(a.snapshot(), b.snapshot())


class TestArgResolution(CustomTestCase):
    """The launcher-side resolution: it runs ONCE, in the launcher process,
    and the frozen result travels to every scheduler inside the pickled
    cross-shapes stash -- which is what makes both features rank-uniform by
    construction, before any round has run."""

    def test_lazy_stash_is_none_when_the_flag_is_off(self):
        args = SimpleNamespace(speculative_cross_algorithm_lazy_capture=False)
        self.assertIsNone(_resolve_lazy_stash(args, "auto"))

    def test_lazy_stash_rebuilds_the_config(self):
        args = SimpleNamespace(speculative_cross_algorithm_lazy_capture=True)
        stash = _resolve_lazy_stash(args, "auto")
        cfg = LazyCaptureConfig(**stash)
        cfg.validate()
        self.assertEqual(cfg.warmup_rounds, LazyCaptureConfig().warmup_rounds)

    def test_lazy_stash_seeds_the_round_knobs_from_the_bandit_config(self):
        os.environ["SGLANG_CROSS_BANDIT_MIN_DWELL_ROUNDS"] = "123"
        args = SimpleNamespace(speculative_cross_algorithm_lazy_capture=True)
        try:
            stash = _resolve_lazy_stash(args, "auto")
        finally:
            del os.environ["SGLANG_CROSS_BANDIT_MIN_DWELL_ROUNDS"]
        self.assertEqual(stash["min_dwell_rounds"], 123)

    def test_lazy_requires_force_auto(self):
        args = SimpleNamespace(speculative_cross_algorithm_lazy_capture=True)
        for force in ("nextn", "dflash", "policy", "schedule"):
            with self.assertRaises(ValueError):
                _resolve_lazy_stash(args, force)

    def test_retire_stash_defaults_to_disabled(self):
        args = SimpleNamespace(speculative_cross_algorithm_retire_ctx="off")
        stash = _resolve_retire_stash(args, "/nonexistent")
        self.assertIsNone(RetirePolicy.from_stash(stash).collapse_ctx)


class TestWorkerWiring(CustomTestCase):
    """The thin worker-side glue, exercised on a hand-built shell (no GPU,
    no sub-workers). What matters here is the flag-OFF equivalence and the
    capture/eager pointer discipline."""

    def _shell(self, lazy=None, retire=None, retired=False):
        from sglang.srt.speculative.cross_algo_worker import CrossAlgoWorker

        w = object.__new__(CrossAlgoWorker)
        w._lazy = lazy
        w._lazy_aux_active = None if lazy is None else lazy.aux_required_now
        w._lazy_eager_active = False
        w._lazy_warmkeep = True
        w._dflash_retired = retired
        w._retire_policy = retire if retire is not None else RetirePolicy()
        w._switching = True
        w._force = "auto"
        w._active_name = "nextn"
        w._active_k = 3
        w._primary_k = 3
        w._dflash_block_size = 16
        w._rounds_total = 0
        w._ctx_gate_last_ctx = 0
        return w

    def test_runtime_aux_enabled_flag_off_is_the_old_value(self):
        # Flag OFF must be byte-identical: every switching mode bakes dual
        # capture into both graph sets, exactly as before #156-4.
        w = self._shell()
        self.assertTrue(w._runtime_aux_enabled())
        w._switching = False
        w._force = "nextn"
        self.assertFalse(w._runtime_aux_enabled())
        w._force = "dflash"
        self.assertTrue(w._runtime_aux_enabled())

    def test_runtime_aux_enabled_under_lazy_follows_the_boot_family(self):
        ctl = LazyCaptureController(_cfg(), initial_family="nextn")
        self.assertFalse(self._shell(lazy=ctl)._runtime_aux_enabled())
        ctl2 = LazyCaptureController(_cfg(), initial_family="dflash")
        self.assertTrue(self._shell(lazy=ctl2)._runtime_aux_enabled())

    def test_capture_mode_toggles_aux_and_the_graph_runner(self):
        ctl = LazyCaptureController(_cfg())
        w = self._shell(lazy=ctl)
        toggles = []
        w._set_target_aux_capture = toggles.append
        graph_runner = object()
        w._active_target_graph_runner = lambda: graph_runner
        mr = SimpleNamespace(decode_cuda_graph_runner=graph_runner)
        w._target_worker = SimpleNamespace(model_runner=mr)

        ctl.observe(1, 1000, 4.0, 6.0)
        w._lazy_apply_capture_mode(ctl.step(1))
        # Steady NEXTN: no aux, no warm-keep, graphs replay.
        self.assertEqual(toggles, [])
        self.assertIs(mr.decode_cuda_graph_runner, graph_runner)
        self.assertFalse(w._lazy_warmkeep)

        # Force the controller into a warm-up window.
        ctl._last_window_end = -10**6
        ctl._last_adopt_round = -10**6
        ctl.observe(2, 1000, 4.0, 6.0)
        w._lazy_apply_capture_mode(ctl.step(2))
        self.assertEqual(toggles, [True])
        self.assertIsNone(
            mr.decode_cuda_graph_runner,
            "the aux-OFF NEXTN graph set must be bypassed while aux is on",
        )
        self.assertTrue(w._lazy_warmkeep)

        # Back to steady: the pointer must be restored, not left None.
        ctl._adopt("nextn")
        ctl.observe(3, 1000, 4.0, 6.0)
        w._lazy_apply_capture_mode(ctl.step(3))
        self.assertEqual(toggles, [True, False])
        self.assertIs(mr.decode_cuda_graph_runner, graph_runner)

    def test_retire_latch_needs_accept_data_and_is_permanent(self):
        w = self._shell(retire=_retire(10000))
        accepts = {}
        w._bandit_or_none = lambda: SimpleNamespace(
            metrics=SimpleNamespace(
                accept_len=lambda rung, min_samples=1: accepts.get(rung)
            )
        )
        w._maybe_retire_dflash(8500)  # in-band but no data yet
        self.assertFalse(w._dflash_retired)
        accepts[("dflash", 16)] = 3.0
        accepts[("nextn", 3)] = 4.3
        w._maybe_retire_dflash(2000)  # low ctx -> content, not collapse
        self.assertFalse(w._dflash_retired)
        w._maybe_retire_dflash(8500)  # in-band -> collapse
        self.assertTrue(w._dflash_retired)
        accepts[("dflash", 16)] = 9.0
        w._maybe_retire_dflash(2000)
        self.assertTrue(w._dflash_retired, "retirement must be permanent")

    def test_retire_disabled_never_touches_the_latch(self):
        w = self._shell()  # default RetirePolicy: disabled
        w._bandit_or_none = lambda: None
        for ctx in (0, 4096, 10**6):
            w._maybe_retire_dflash(ctx)
        self.assertFalse(w._dflash_retired)


class TestOverlapRelayAcrossFamilyTransition(CustomTestCase):
    """Regression for the crash lazy capture exposed on GPU (round 549).

    THE LESSON: this bug needed ~550 rounds of a real run to appear, because
    it only bites once the run is DFLASH-only for more than one round -- and
    the SHAPE of a smoke test (20 tokens) cannot reach that. So it is tested
    here as a ROUND SEQUENCE over the relay, not as a single call.

    What broke: DFlashDraftInputV2's Eagle-shaped fields are zero-width
    PLACEHOLDERS ((bs, 0)); the drafter documents them as unused. The
    per-round warm-keep of the idle NEXTN rung used to overwrite all three
    with real seeds on every DFLASH round, so a placeholder never reached
    FutureMap.stash. Lazy capture drops that warm-keep -- on purpose, it is
    the expensive half of the tax -- and the placeholder went straight into
    a slot sized from a real NEXTN payload:
        RuntimeError: shape mismatch: value tensor of shape [0] cannot be
        broadcast to indexing result of shape [1, 1]
    """

    ALGO = "EAGLE"  # the global relay type under every switching mode

    def _future_map(self, req_pool_size=8):
        pool = SimpleNamespace(
            req_to_token=torch.zeros((req_pool_size, 4), dtype=torch.int32)
        )
        return FutureMap(
            torch.device("cpu"),
            SpeculativeAlgorithm.from_string(self.ALGO),
            pool,
        )

    @staticmethod
    def _nextn_payload(token, topk_width=1, hidden=16):
        return RelayPayload(
            bonus_tokens=torch.tensor([token], dtype=torch.int64),
            topk_p=torch.full((1, topk_width), float(token)),
            topk_index=torch.full((1, topk_width), token, dtype=torch.int64),
            hidden_states=torch.full((1, hidden), float(token)),
        )

    @staticmethod
    def _dflash_payload(token):
        """The REAL DFLASH decode payload, built by the production helper."""
        return RelayPayload.from_draft_input(
            make_draft_input_v2(
                bonus_tokens=torch.tensor([token], dtype=torch.int64),
                new_seq_lens=torch.tensor([100 + token], dtype=torch.int64),
            )
        )

    def test_dflash_placeholders_are_normalized_to_absent(self):
        d = make_draft_input_v2(
            bonus_tokens=torch.tensor([7], dtype=torch.int64),
            new_seq_lens=torch.tensor([10], dtype=torch.int64),
        )
        # Precondition of the whole bug: they are (bs, 0) tensors, NOT None.
        self.assertEqual(tuple(d.topk_p.shape), (1, 0))
        self.assertEqual(tuple(d.topk_index.shape), (1, 0))
        self.assertEqual(tuple(d.hidden_states.shape), (1, 0))
        payload = RelayPayload.from_draft_input(d)
        self.assertIsNone(payload.topk_p)
        self.assertIsNone(payload.topk_index)
        self.assertIsNone(payload.hidden_states)
        self.assertIsNotNone(payload.bonus_tokens)

    def test_relay_field_passes_real_tensors_through_unchanged(self):
        t = torch.rand(2, 3)
        self.assertIs(relay_field(t), t)
        self.assertIsNone(relay_field(None))
        self.assertIsNone(relay_field(torch.empty((2, 0))))
        self.assertIsNone(relay_field(torch.empty((0,))))

    def test_long_dflash_segment_after_a_nextn_segment(self):
        """The crash, as the run produced it: many NEXTN rounds, a switch,
        then a LONG DFLASH-only stretch. Every DFLASH round must stash
        cleanly and the last real NEXTN seeds must survive."""
        idx = torch.tensor([0])
        with mock.patch(
            "sglang.srt.speculative.spec_utils.spec_need_hidden_states",
            return_value=True,
        ):
            fm = self._future_map()
            for tok in range(1, 40):  # NEXTN segment
                fm.stash(idx, self._nextn_payload(tok))
            self.assertEqual(tuple(fm.topk_p_buf.shape), (8, 1))
            last_seeds = fm.topk_p_buf[0].clone()
            last_hidden = fm.hidden_states_buf[0].clone()

            for tok in range(40, 600):  # DFLASH-only stretch (the crash zone)
                fm.stash(idx, self._dflash_payload(tok))

            # bonus tokens keep flowing (DFLASH owns that field) ...
            self.assertEqual(int(fm.output_tokens_buf[0]), 599)
            # ... and the absent fields kept their last relayed value, which
            # is exactly the documented absent-field contract.
            self.assertTrue(torch.equal(fm.topk_p_buf[0], last_seeds))
            self.assertTrue(torch.equal(fm.hidden_states_buf[0], last_hidden))

    def test_round_trip_nextn_dflash_nextn_refreshes_the_seeds(self):
        """The switch BACK: the warm-up window re-primes the NEXTN seeds
        before the family change, so the first NEXTN round after a long
        DFLASH segment does not draft from pre-segment seeds."""
        idx = torch.tensor([0])
        with mock.patch(
            "sglang.srt.speculative.spec_utils.spec_need_hidden_states",
            return_value=True,
        ):
            fm = self._future_map()
            for tok in range(1, 20):
                fm.stash(idx, self._nextn_payload(tok))
            stale = fm.topk_p_buf[0].clone()
            for tok in range(20, 400):
                fm.stash(idx, self._dflash_payload(tok))
            self.assertTrue(torch.equal(fm.topk_p_buf[0], stale))
            # Warm-up rounds: the catch-up attaches real seeds to the DFLASH
            # draft input again (cross_algo_worker._warmkeep_nextn_after_
            # dflash_round), so the payload stops being a placeholder.
            for tok in range(400, 405):
                fm.stash(idx, self._nextn_payload(tok))
            self.assertFalse(torch.equal(fm.topk_p_buf[0], stale))
            self.assertAlmostEqual(float(fm.topk_p_buf[0][0]), 404.0)

    def test_dflash_first_then_nextn_sizes_the_buffers_from_real_data(self):
        """Reverse order: a placeholder must never SIZE the pool buffers.
        A (req_pool_size, 0) buffer would swallow every write silently and
        then mismatch the first real payload -- the same bug, one round
        later and much harder to read."""
        idx = torch.tensor([0])
        with mock.patch(
            "sglang.srt.speculative.spec_utils.spec_need_hidden_states",
            return_value=True,
        ):
            fm = self._future_map()
            for tok in range(1, 30):
                fm.stash(idx, self._dflash_payload(tok))
            self.assertIsNone(fm.topk_p_buf)
            self.assertIsNone(fm.hidden_states_buf)
            fm.stash(idx, self._nextn_payload(99))
            self.assertEqual(tuple(fm.topk_p_buf.shape), (8, 1))
            self.assertEqual(tuple(fm.hidden_states_buf.shape), (8, 16))
            self.assertAlmostEqual(float(fm.topk_p_buf[0][0]), 99.0)

    def test_relayed_extras_do_not_depend_on_stash_order(self):
        """Second bug of the same family, found while fixing the first:
        need_hidden_states used to be latched from the FIRST payload's shape.
        Under cross-algorithm switching a DFLASH round can be stashed first,
        and it carries no hidden states -- which pinned the flag to False for
        the FutureMap's whole life and silently stripped the NEXTN rung of
        its hidden-state relay. Silent accept-rate loss, not a crash, so it
        needs an explicit assertion."""
        idx = torch.tensor([0])
        with mock.patch(
            "sglang.srt.speculative.spec_utils.spec_need_hidden_states",
            return_value=True,
        ):
            nextn_first = self._future_map()
            nextn_first.stash(idx, self._nextn_payload(1))
            dflash_first = self._future_map()
            dflash_first.stash(idx, self._dflash_payload(1))
            bonus_first = self._future_map()
            bonus_first.stash(
                idx,
                RelayPayload(bonus_tokens=torch.tensor([1], dtype=torch.int64)),
            )
            for fm in (nextn_first, dflash_first, bonus_first):
                self.assertTrue(fm.need_topk)
                self.assertTrue(fm.need_hidden_states)
            # ... and the two that started without the field pick the real
            # shape up as soon as a payload carries it.
            for fm in (dflash_first, bonus_first):
                self.assertIsNone(fm.hidden_states_buf)
                fm.stash(idx, self._nextn_payload(2))
                self.assertEqual(tuple(fm.hidden_states_buf.shape), (8, 16))

    def test_bonus_only_payload_still_works(self):
        """Flag-OFF / non-spec shape: a payload with no spec extras at all
        was already tolerated and must stay tolerated."""
        idx = torch.tensor([0])
        with mock.patch(
            "sglang.srt.speculative.spec_utils.spec_need_hidden_states",
            return_value=True,
        ):
            fm = self._future_map()
            fm.stash(idx, self._nextn_payload(5))
            fm.stash(
                idx,
                RelayPayload(bonus_tokens=torch.tensor([6], dtype=torch.int64)),
            )
            self.assertEqual(int(fm.output_tokens_buf[0]), 6)
            self.assertAlmostEqual(float(fm.topk_p_buf[0][0]), 5.0)


if __name__ == "__main__":
    unittest.main()
