"""Unit tests for T156 stage 4: the cross-algorithm rung bandit.

CPU-only tests of the pure decision logic: the score formula
(EMA accept / EMA round seconds), dwell and deadzone guards, burn-in
exclusion after switches, staleness probing (cold start, periodic re-probe,
clearly-inferior suppression, probe adoption/return), and the broadcast
no-decision fallback (decide returns the current rung). GPU behavior is
covered by the live validation protocol, not here.
"""

import os
import unittest

from sglang.srt.speculative.cross_algo_bandit import (
    ENV_PREFIX,
    CrossAlgoBandit,
    CrossBanditConfig,
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
