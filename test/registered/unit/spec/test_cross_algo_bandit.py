"""Unit tests for T156 stage 4: the cross-algorithm rung bandit.

CPU-only tests of the pure decision logic: the score formula
(EMA accept / EMA round seconds), dwell and deadzone guards, burn-in
exclusion after switches, staleness probing (cold start, periodic re-probe,
clearly-inferior suppression, probe adoption/return), and the broadcast
no-decision fallback (decide returns the current rung). GPU behavior is
covered by the live validation protocol, not here.
"""

import json
import os
import tempfile
import unittest

from sglang.srt.speculative.cross_algo_bandit import (
    ENV_PREFIX,
    CrossAlgoBandit,
    CrossBanditConfig,
)
from sglang.srt.speculative.cross_algo_utils import (
    ctx_gate_eligible,
    derive_ctx_gate_threshold,
    parse_ctx_gate_value,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

NEXTN3 = ("nextn", 3)
NEXTN1 = ("nextn", 1)
DFLASH = ("dflash", 16)
RUNGS = [NEXTN1, NEXTN3, DFLASH]


def _cfg(**kw):
    base = dict(
        decide_interval_rounds=16,
        min_dwell_rounds=64,
        burn_in_rounds=0,
        deadzone=0.06,
        probe_interval_rounds=256,
        probe_window_rounds=32,
        probe_inferior_factor=1.5,
        min_time_samples=3,
    )
    base.update(kw)
    return CrossBanditConfig(**base)


def _bandit(cfg=None, rungs=RUNGS, initial=NEXTN3):
    return CrossAlgoBandit(rungs=rungs, initial=initial, cfg=cfg or _cfg())


def _feed(bandit, rung, accept, rounds, dt=0.040, t0=0.0, bs=1):
    """Feed `rounds` back-to-back results for `rung` (accept drafts each,
    dt seconds apart). Returns the timestamp after the last result."""
    t = t0
    for _ in range(rounds):
        t += dt
        bandit.observe_result(rung, [accept], bs, now=t)
    return t


class TestScoreFormula(CustomTestCase):
    def test_reward_is_accept_ema_over_round_seconds_ema(self):
        b = _bandit()
        # Constant stream: EMAs converge to the constants exactly.
        _feed(b, NEXTN3, accept=3, rounds=10, dt=0.040)
        s = b.score(NEXTN3)
        # accept_len includes the bonus token: 1 + 3 = 4; dt 40 ms.
        self.assertAlmostEqual(s, 4.0 / 0.040, places=6)

    def test_unmeasured_rung_scores_none(self):
        b = _bandit()
        self.assertIsNone(b.score(NEXTN3))
        # One result: accept data exists but no dt gap yet -> still None.
        b.observe_result(NEXTN3, [3], 1, now=1.0)
        self.assertIsNone(b.score(NEXTN3))

    def test_min_time_samples_gate(self):
        b = _bandit(_cfg(min_time_samples=3))
        _feed(b, NEXTN3, accept=3, rounds=3, dt=0.040)  # 2 dt samples
        self.assertIsNone(b.score(NEXTN3))
        _feed(b, NEXTN3, accept=3, rounds=1, dt=0.040, t0=3 * 0.040)
        self.assertIsNotNone(b.score(NEXTN3))

    def test_idle_gaps_do_not_poison_duration(self):
        b = _bandit()
        t = _feed(b, NEXTN3, accept=3, rounds=6, dt=0.040)
        s_before = b.score(NEXTN3)
        # A 5 s scheduler idle gap must be ignored by the duration EMA.
        b.observe_result(NEXTN3, [3], 1, now=t + 5.0)
        self.assertAlmostEqual(b.score(NEXTN3), s_before, places=6)


class TestBurnIn(CustomTestCase):
    def test_first_results_after_rung_change_are_excluded(self):
        b = _bandit(_cfg(burn_in_rounds=4))
        _feed(b, NEXTN3, accept=3, rounds=10, dt=0.040)
        samples_before = b.metrics._per_rung[NEXTN3][1]
        # Switch the attribution stream to DFLASH: 4 burn-in results with
        # atypically LOW accept must not enter the estimator.
        t = 10 * 0.040
        t = _feed(b, DFLASH, accept=0, rounds=4, dt=0.040, t0=t)
        self.assertNotIn(DFLASH, b.metrics._per_rung)
        # The 5th+ result records; the dt chain stayed intact (dt=40 ms, not
        # a multi-round gap).
        _feed(b, DFLASH, accept=8, rounds=2, dt=0.040, t0=t)
        m = b.metrics._per_rung[DFLASH]
        self.assertEqual(m[1], 2)  # accept samples
        self.assertEqual(m[3], 2)  # dt samples survived the burn-in chain
        self.assertAlmostEqual(m[2], 0.040, places=6)
        # And switching BACK re-applies burn-in to the first rung.
        t = 20 * 0.040
        _feed(b, NEXTN3, accept=3, rounds=4, dt=0.040, t0=t)
        self.assertEqual(b.metrics._per_rung[NEXTN3][1], samples_before)


class TestDwellAndDeadzone(CustomTestCase):
    def _measured_bandit(self, cur_score_dt, other_score_dt):
        """NEXTN3 active with dt=cur_score_dt; DFLASH measured with
        dt=other_score_dt (same accept), scores ~ 1/dt."""
        b = _bandit(rungs=[NEXTN3, DFLASH])
        _feed(b, NEXTN3, accept=3, rounds=10, dt=cur_score_dt)
        _feed(b, DFLASH, accept=3, rounds=10, dt=other_score_dt, t0=100.0)
        # Reset burn-in bookkeeping noise: attribution ends on DFLASH but
        # burn_in=0 in _cfg so nothing was excluded.
        return b

    def test_dwell_blocks_early_switches(self):
        b = self._measured_bandit(0.060, 0.040)  # DFLASH 50% better
        b._last_switch_round = 100
        self.assertEqual(b.decide(100 + 63), NEXTN3)  # dwell holds
        self.assertEqual(b.decide(100 + 64), DFLASH)  # dwell expired

    def test_deadzone_blocks_marginal_switches(self):
        b = self._measured_bandit(0.0412, 0.0400)  # +3% < 6% deadzone
        b._last_active_round[DFLASH] = 999  # fresh estimate, no probe due
        self.assertEqual(b.decide(1000), NEXTN3)

    def test_clear_win_switches(self):
        b = self._measured_bandit(0.0480, 0.0400)  # +20% > deadzone
        self.assertEqual(b.decide(1000), DFLASH)
        self.assertEqual(b.active, DFLASH)
        self.assertEqual(b.switch_count, 1)

    def test_no_decision_returns_current_rung(self):
        """The broadcast fallback: when nothing warrants a change, decide()
        returns the CURRENT rung -- broadcasting it is the explicit
        no-decision message every rank can apply as a no-op."""
        b = self._measured_bandit(0.040, 0.040)  # equal scores
        b._last_active_round[DFLASH] = 999  # fresh estimate, no probe due
        self.assertEqual(b.decide(1000), NEXTN3)  # argmax==active -> hold
        self.assertEqual(b.switch_count, 0)
        # Within the dwell, decide() holds unconditionally.
        b._last_switch_round = 990
        self.assertEqual(b.decide(1010), NEXTN3)


class TestProbing(CustomTestCase):
    def test_cold_start_probes_unmeasured_rung_after_dwell(self):
        b = _bandit(rungs=[NEXTN3, DFLASH])
        _feed(b, NEXTN3, accept=3, rounds=10, dt=0.040)
        # Dwell not yet expired -> hold.
        self.assertEqual(b.decide(32), NEXTN3)
        # Dwell expired -> probe the never-measured DFLASH rung.
        self.assertEqual(b.decide(64), DFLASH)
        self.assertIsNotNone(b._probe_until)
        self.assertEqual(b._probe_until, 64 + 32)

    def test_probe_returns_to_incumbent_when_it_loses(self):
        b = _bandit(rungs=[NEXTN3, DFLASH])
        _feed(b, NEXTN3, accept=3, rounds=10, dt=0.040)
        self.assertEqual(b.decide(64), DFLASH)  # probe_start
        # During the probe window: hold on the probe rung.
        _feed(b, DFLASH, accept=3, rounds=10, dt=0.060, t0=50.0)  # worse
        self.assertEqual(b.decide(80), DFLASH)  # probe_hold
        # Window over: probe rung lost -> back to the incumbent.
        self.assertEqual(b.decide(96), NEXTN3)
        self.assertEqual(b.active, NEXTN3)

    def test_probe_adopts_winner(self):
        b = _bandit(rungs=[NEXTN3, DFLASH])
        _feed(b, NEXTN3, accept=3, rounds=10, dt=0.060)
        self.assertEqual(b.decide(64), DFLASH)  # probe_start
        _feed(b, DFLASH, accept=3, rounds=10, dt=0.040, t0=50.0)  # better
        self.assertEqual(b.decide(96), DFLASH)  # adopted
        self.assertEqual(b.active, DFLASH)
        # Adoption restarts the dwell clock.
        self.assertEqual(b._last_switch_round, 96)

    def test_stale_rung_reprobed_after_interval(self):
        b = _bandit(rungs=[NEXTN3, DFLASH])
        _feed(b, NEXTN3, accept=3, rounds=10, dt=0.040)
        _feed(b, DFLASH, accept=3, rounds=10, dt=0.041, t0=100.0)  # ~equal
        b._last_active_round[DFLASH] = 1000
        b._last_switch_round = 1000
        # Not stale yet at 1064 (age 64 < 256) and not better -> hold.
        self.assertEqual(b.decide(1064), NEXTN3)
        # Stale at 1256+ -> probe.
        self.assertEqual(b.decide(1300), DFLASH)

    def test_clearly_inferior_rung_is_not_probed(self):
        b = _bandit(rungs=[NEXTN3, NEXTN1])
        _feed(b, NEXTN3, accept=3, rounds=10, dt=0.040)  # score 100
        _feed(b, NEXTN1, accept=1, rounds=10, dt=0.040, t0=100.0)  # score 50
        b._last_active_round[NEXTN1] = 0
        # 50 * 1.5 = 75 < 100 -> suppressed; no probe, hold forever.
        self.assertEqual(b.decide(10_000), NEXTN3)
        self.assertEqual(b.active, NEXTN3)

    def test_once_competitive_rung_stays_probe_eligible(self):
        """The optimistic gate: a rung whose FROZEN score is clearly inferior
        but whose (decayed) best-ever score was competitive must still be
        probed -- and must outrank a never-competitive rung for the probe
        slot (the 2026-07-22 k=1-snipes-DFLASH incident)."""
        b = _bandit(rungs=[NEXTN3, NEXTN1, DFLASH], cfg=_cfg(
            probe_inferior_factor=1.2))
        _feed(b, NEXTN3, accept=3, rounds=10, dt=0.040)  # score 100
        _feed(b, NEXTN1, accept=1.4, rounds=10, dt=0.040, t0=100.0)  # ~60
        _feed(b, DFLASH, accept=4, rounds=10, dt=0.040, t0=200.0)  # 125
        b.decide(0)  # records best: dflash 125, nextn1 60 (dwell_hold)
        # Content shift: DFLASH's live estimate collapses below NEXTN1's.
        _feed(b, DFLASH, accept=1, rounds=60, dt=0.040, t0=300.0)  # -> ~50
        self.assertLess(b.score(DFLASH), b.score(NEXTN1))
        b._last_active_round[NEXTN1] = 0
        b._last_active_round[DFLASH] = 0
        b._last_switch_round = 0
        target = b.decide(1000)
        # NEXTN1: best 60 * 1.2 = 72 < 100 -> suppressed.
        # DFLASH: best ~125 (decayed) * 1.2 > 100 -> probed.
        self.assertEqual(target, DFLASH)
        self.assertIsNotNone(b._probe_until)


NO_DFLASH = frozenset([NEXTN1, NEXTN3])


class TestCtxGateDecide(CustomTestCase):
    """decide(eligible=...): selection filter, probe suppression, eviction,
    probe abort, and the off-mode identity."""

    def test_eligible_none_is_identical_to_no_arg(self):
        """Gate off (eligible=None) must reproduce the pre-gate decision."""
        for kwargs in ({}, {"eligible": None}):
            b = _bandit(rungs=[NEXTN3, DFLASH])
            _feed(b, NEXTN3, accept=3, rounds=10, dt=0.060)
            _feed(b, DFLASH, accept=3, rounds=10, dt=0.040, t0=100.0)
            self.assertEqual(b.decide(1000, **kwargs), DFLASH)

    def test_ineligible_rung_not_selected_by_argmax(self):
        b = _bandit(rungs=[NEXTN3, DFLASH])
        _feed(b, NEXTN3, accept=3, rounds=10, dt=0.060)
        _feed(b, DFLASH, accept=3, rounds=10, dt=0.040, t0=100.0)  # 50% better
        b._last_active_round[DFLASH] = 999  # fresh, no probe due
        self.assertEqual(
            b.decide(1000, eligible=frozenset([NEXTN3])), NEXTN3
        )
        self.assertEqual(b.switch_count, 0)

    def test_ineligible_rung_not_probed_cold_start(self):
        b = _bandit(rungs=[NEXTN3, DFLASH])
        _feed(b, NEXTN3, accept=3, rounds=10, dt=0.040)
        # Pre-gate behavior: the never-measured DFLASH rung is probed at 64
        # (see TestProbing). Gated: it must not be.
        self.assertEqual(b.decide(64, eligible=frozenset([NEXTN3])), NEXTN3)
        self.assertIsNone(b._probe_until)
        # Gate lifted -> the cold-start probe happens.
        self.assertEqual(b.decide(80), DFLASH)
        self.assertIsNotNone(b._probe_until)

    def test_ineligible_stale_rung_not_probed(self):
        b = _bandit(rungs=RUNGS)
        _feed(b, NEXTN3, accept=3, rounds=10, dt=0.040)
        _feed(b, NEXTN1, accept=3, rounds=10, dt=0.041, t0=100.0)
        _feed(b, DFLASH, accept=3, rounds=10, dt=0.041, t0=200.0)
        for r in (NEXTN1, DFLASH):
            b._last_active_round[r] = 0  # both long stale
        b._last_switch_round = 1000
        b._best_score[DFLASH] = 1000.0  # would win the probe slot
        # Gated: the probe goes to the eligible stale rung instead.
        self.assertEqual(b.decide(2000, eligible=NO_DFLASH), NEXTN1)
        self.assertEqual(b._pre_probe_rung, NEXTN3)

    def test_active_ineligible_rung_evicted_overriding_dwell(self):
        b = _bandit(rungs=RUNGS, initial=DFLASH)
        _feed(b, DFLASH, accept=8, rounds=10, dt=0.040)
        _feed(b, NEXTN3, accept=3, rounds=10, dt=0.040, t0=100.0)
        _feed(b, NEXTN1, accept=1, rounds=10, dt=0.040, t0=200.0)
        b._last_switch_round = 995  # dwell far from expired
        target = b.decide(1000, eligible=NO_DFLASH)
        # Initial rung (DFLASH) is itself ineligible -> fallback: the best
        # ELIGIBLE measured rung, despite DFLASH scoring highest and the
        # dwell not being over.
        self.assertEqual(target, NEXTN3)
        self.assertEqual(b.active, NEXTN3)
        self.assertEqual(b.switch_count, 1)

    def test_eviction_prefers_initial_rung_over_frozen_argmax(self):
        """Eviction falls back to the INITIAL (boot) rung, not the frozen-
        score argmax: at eviction time every eligible score is frozen from
        the pre-crossing regime (the 2026-07-22 nextn:2 lock-in)."""
        b = _bandit(rungs=RUNGS, initial=NEXTN3)
        _feed(b, NEXTN3, accept=2, rounds=10, dt=0.040)
        _feed(b, NEXTN1, accept=5, rounds=10, dt=0.040, t0=100.0)  # argmax
        _feed(b, DFLASH, accept=8, rounds=10, dt=0.040, t0=200.0)
        b.active = DFLASH
        target = b.decide(1000, eligible=NO_DFLASH)
        self.assertEqual(target, NEXTN3)  # initial, NOT the higher NEXTN1
        self.assertEqual(b.active, NEXTN3)

    def test_eviction_with_no_measured_eligible_rung(self):
        b = _bandit(rungs=[NEXTN3, DFLASH], initial=DFLASH)
        _feed(b, DFLASH, accept=8, rounds=10, dt=0.040)
        target = b.decide(1000, eligible=frozenset([NEXTN3]))
        self.assertEqual(target, NEXTN3)  # first eligible, unmeasured

    def test_probe_aborted_when_probe_rung_becomes_ineligible(self):
        b = _bandit(rungs=[NEXTN3, DFLASH])
        _feed(b, NEXTN3, accept=3, rounds=10, dt=0.040)
        self.assertEqual(b.decide(64), DFLASH)  # probe_start
        # Mid-window (would be probe_hold at 70) the context crosses the
        # threshold: abort, return to the incumbent, probe state cleared.
        self.assertEqual(b.decide(70, eligible=frozenset([NEXTN3])), NEXTN3)
        self.assertEqual(b.active, NEXTN3)
        self.assertIsNone(b._probe_until)
        self.assertIsNone(b._pre_probe_rung)

    def test_gate_skip_logged_once_per_phase(self):
        b = _bandit(rungs=[NEXTN3, DFLASH])
        _feed(b, NEXTN3, accept=3, rounds=10, dt=0.040)
        with self.assertLogs(
            "sglang.srt.speculative.cross_algo_bandit", level="INFO"
        ) as cm:
            b.decide(64, eligible=frozenset([NEXTN3]))
            b.decide(80, eligible=frozenset([NEXTN3]))
        skips = [m for m in cm.output if "skipped: ctx gate" in m]
        self.assertEqual(len(skips), 1)
        # Eligibility restored, then lost again -> a fresh phase logs again.
        b.decide(96)
        self.assertIsNotNone(b._probe_until)  # probe started while eligible
        b.decide(200)  # probe window over (adopt or return; irrelevant here)
        b.active = NEXTN3
        b._probe_until = None
        b._pre_probe_rung = None
        with self.assertLogs(
            "sglang.srt.speculative.cross_algo_bandit", level="INFO"
        ) as cm:
            b.decide(1000, eligible=frozenset([NEXTN3]))
        self.assertTrue(any("skipped: ctx gate" in m for m in cm.output))


class TestCtxGateEligibility(CustomTestCase):
    """The pure eligibility predicate (threshold, bs>1 max, turn-start
    pre-emption nearness criterion)."""

    def test_below_threshold_eligible(self):
        ok, max_ctx = ctx_gate_eligible([(4000, 100)], 8192, 0.8)
        self.assertTrue(ok)
        self.assertEqual(max_ctx, 4000)

    def test_at_or_above_threshold_ineligible(self):
        self.assertFalse(ctx_gate_eligible([(8192, 10)], 8192, 0.8)[0])
        self.assertFalse(ctx_gate_eligible([(50_000, 10)], 8192, 0.8)[0])

    def test_bs2_max_over_requests(self):
        ok, max_ctx = ctx_gate_eligible(
            [(2000, 100), (50_000, 100)], 8192, 0.8
        )
        self.assertFalse(ok)
        self.assertEqual(max_ctx, 50_000)
        self.assertTrue(ctx_gate_eligible([(2000, 100), (3000, 100)], 8192, 0.8)[0])

    def test_turn_start_preemption_near_and_crossing(self):
        # near (7000 > 0.8*8192=6553.6) AND budget crosses (7000+2000>8192).
        self.assertFalse(ctx_gate_eligible([(7000, 2000)], 8192, 0.8)[0])
        # near but budget CANNOT cross -> eligible.
        self.assertTrue(ctx_gate_eligible([(7000, 1000)], 8192, 0.8)[0])
        # NOT near: a huge budget alone does not pre-empt (no answer-length
        # estimator; short-context requests usually stop early).
        self.assertTrue(ctx_gate_eligible([(2000, 16_000)], 8192, 0.8)[0])
        # near with unbounded budget (None) -> treated as crossing.
        self.assertFalse(ctx_gate_eligible([(7000, None)], 8192, 0.8)[0])

    def test_empty_batch_eligible(self):
        ok, max_ctx = ctx_gate_eligible([], 8192, 0.8)
        self.assertTrue(ok)
        self.assertEqual(max_ctx, 0)


class TestCtxGateDerivation(CustomTestCase):
    """Threshold auto-derivation from the drafter's config.json and flag
    value parsing."""

    def _write_cfg(self, tmpdir, cfg):
        with open(os.path.join(tmpdir, "config.json"), "w") as f:
            json.dump(cfg, f)

    def test_zlab_swa_drafter_derives_8k(self):
        # The z-lab Qwen3.6-27B drafter shape: target-inherited (useless)
        # max_position_embeddings, declared SWA window 2048 -> 4*2048 = 8192.
        with tempfile.TemporaryDirectory() as d:
            self._write_cfg(
                d,
                {
                    "max_position_embeddings": 262144,
                    "sliding_window": 2048,
                    "use_sliding_window": True,
                },
            )
            thr, source = derive_ctx_gate_threshold(d)
        self.assertEqual(thr, 8192)
        self.assertIn("sliding_window", source)

    def test_factor_env_override(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_cfg(d, {"sliding_window": 2048})
            os.environ["SGLANG_CROSS_CTX_GATE_FACTOR"] = "8"
            try:
                thr, _ = derive_ctx_gate_threshold(d)
            finally:
                del os.environ["SGLANG_CROSS_CTX_GATE_FACTOR"]
        self.assertEqual(thr, 16384)

    def test_swa_threshold_capped_at_mpe(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_cfg(
                d, {"max_position_embeddings": 4096, "sliding_window": 2048}
            )
            thr, _ = derive_ctx_gate_threshold(d)
        self.assertEqual(thr, 4096)

    def test_long_context_drafter_lifts_threshold(self):
        # No SWA declared -> the drafter's own max_position_embeddings IS the
        # threshold; a genuine long-context drafter raises the gate itself.
        with tempfile.TemporaryDirectory() as d:
            self._write_cfg(d, {"max_position_embeddings": 65536})
            thr, source = derive_ctx_gate_threshold(d)
        self.assertEqual(thr, 65536)
        self.assertIn("max_position_embeddings", source)

    def test_use_sliding_window_false_ignores_window(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_cfg(
                d,
                {
                    "max_position_embeddings": 32768,
                    "sliding_window": 2048,
                    "use_sliding_window": False,
                },
            )
            thr, _ = derive_ctx_gate_threshold(d)
        self.assertEqual(thr, 32768)

    def test_unreadable_config_disables_gate(self):
        with tempfile.TemporaryDirectory() as d:
            thr, source = derive_ctx_gate_threshold(d)  # no config.json
        self.assertIsNone(thr)
        self.assertIn("gate disabled", source)

    def test_parse_flag_values(self):
        self.assertEqual(parse_ctx_gate_value("auto"), "auto")
        self.assertEqual(parse_ctx_gate_value("off"), "off")
        self.assertEqual(parse_ctx_gate_value("12000"), 12000)
        self.assertEqual(parse_ctx_gate_value(None), "auto")
        for bad in ("0", "-5", "5.5", "on", ""):
            with self.assertRaises(ValueError):
                parse_ctx_gate_value(bad)


class TestConfig(CustomTestCase):
    def test_env_overrides(self):
        os.environ[ENV_PREFIX + "MIN_DWELL_ROUNDS"] = "128"
        os.environ[ENV_PREFIX + "DEADZONE"] = "0.2"
        try:
            cfg = CrossBanditConfig.from_env()
            self.assertEqual(cfg.min_dwell_rounds, 128)
            self.assertAlmostEqual(cfg.deadzone, 0.2)
            self.assertEqual(cfg.decide_interval_rounds, 16)  # default kept
        finally:
            del os.environ[ENV_PREFIX + "MIN_DWELL_ROUNDS"]
            del os.environ[ENV_PREFIX + "DEADZONE"]

    def test_validation_rejects_bad_values(self):
        with self.assertRaises(ValueError):
            _cfg(min_dwell_rounds=0).validate()
        with self.assertRaises(ValueError):
            _cfg(deadzone=-0.1).validate()
        with self.assertRaises(ValueError):
            _cfg(probe_window_rounds=300).validate()  # >= probe_interval
        with self.assertRaises(ValueError):
            _cfg(probe_inferior_factor=0.5).validate()


if __name__ == "__main__":
    unittest.main()
