"""FLLIPER_ADAPTIVE_K: verify width k = argmax E[tokens](k) / step_ms(k).

Pure-Python controller (``adaptive_verify_width``) for the dense 27B DFlash
path; Basalt ``Controller::choose`` (basalt-ref src/spec/controller.cpp:8-66)
is the model, DSpark's ``SpsAdditiveCostTable`` the cost table. Each case
names the failure it guards.
"""

import math
import unittest

from flliper.srt.speculative.adaptive_verify_width import (
    AcceptEma,
    AdaptiveKConfig,
    AdaptiveVerifyWidth,
    VerifyCostCurve,
    choose_width,
    expected_accept_tokens,
    flat_curve,
    initial_ema,
    linear_sps,
    load_curve_file,
    observe_round,
    parse_curve_text,
    round_cost_ms,
    state_digest,
)
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

B = 8
#: A1 (28.10.), all 27B boots: S_j = P(correct drafts >= j), j = 1..7.
A1_SURVIVAL = (0.7411, 0.5243, 0.3653, 0.2646, 0.1989, 0.1544, 0.1233)


def _ema(rate: float) -> AcceptEma:
    return AcceptEma(accept_rate=(rate,) * (B - 1), observed_ct=(0,) * (B - 1))


def _curve(*, a_ms: float, c_ms: float, widths=tuple(range(2, B + 1)), prior=()) -> VerifyCostCurve:
    return VerifyCostCurve(
        widths=tuple(widths),
        sps=linear_sps(a_ms=a_ms, c_ms_per_row=c_ms),
        prior_survival=tuple(prior),
        source="test",
    )


def _cfg(**kw) -> AdaptiveKConfig:
    base = dict(block_size=B, warmup_rounds=0, probe_every=0, min_dwell=0, margin_pct=0.0)
    base.update(kw)
    return AdaptiveKConfig(**base)


class TestChooseWidth(unittest.TestCase):
    def test_flat_curve_optimum_is_block(self):
        """Width-independent cost: every extra verified slot is free -> k = N.

        Guards an argmax that prefers short widths on cost ties (e.g. a score
        of E/k instead of E/cost) or drops B from the candidates."""
        k = choose_width(
            cfg=_cfg(), curve=flat_curve(widths=range(2, B + 1)), ema=_ema(0.6),
            batch_size=1, incumbent=B,
        )
        self.assertEqual(k, B)

    def test_steep_curve_picks_smaller_width(self):
        """rates 0.5, cost 10+5k: E(k)=2-0.5^(k-1); E/C peaks at k=2
        (1.5/20=0.075 > 1.75/25=0.070 > ... > ~2/50=0.040 at k=8)."""
        k = choose_width(
            cfg=_cfg(), curve=_curve(a_ms=10.0, c_ms=5.0), ema=_ema(0.5),
            batch_size=1, incumbent=B,
        )
        self.assertEqual(k, 2)

    def test_a1_curve_bs1_keeps_8_bs3_takes_5(self):
        """A1's 27B numbers, tiers {5,8}: T = 21.5 + 0.93 * rows.
        bs1: E8/T = 3.372/28.94 = 0.1165 > E5/T = 2.895/26.15 = 0.1107 -> 8;
        bs3: E5/35.45 = 0.0817 > E8/43.82 = 0.0770 -> 5 (+6.1 %, A1: +6.0 %).
        Guards the batch-rows term reaching the cost (budget = bs*(k-1)) and
        the measured prior reaching the EMA."""
        curve = _curve(a_ms=21.5, c_ms=0.93, widths=(5, B), prior=A1_SURVIVAL)
        cfg = _cfg()
        ema = initial_ema(cfg=cfg, curve=curve)
        self.assertAlmostEqual(expected_accept_tokens(ema=ema, width=B), 3.372, places=2)
        pick = {
            bs: choose_width(cfg=cfg, curve=curve, ema=ema, batch_size=bs, incumbent=B)
            for bs in (1, 3)
        }
        self.assertEqual(pick, {1: B, 3: 5})

    def test_no_curve_or_curve_without_block_keeps_block(self):
        """Without a curve the feature must equal today's round; a curve
        lacking B has nothing to price the incumbent against."""
        ctl = AdaptiveVerifyWidth(cfg=_cfg(), curve=None)
        self.assertFalse(ctl.active)
        self.assertEqual({ctl.next_width(batch_size=4) for _ in range(50)}, {B})
        no_b = _curve(a_ms=10.0, c_ms=5.0, widths=range(2, B))
        self.assertEqual(
            choose_width(cfg=_cfg(), curve=no_b, ema=_ema(0.5), batch_size=1, incumbent=B), B
        )

    def test_forced_width(self):
        """FLLIPER_ADAPTIVE_K_FORCE=N is the self-test arm (bit-identical to
        today); a forced width outside the curve falls back to N."""
        curve = _curve(a_ms=10.0, c_ms=5.0)
        for force, want in ((B, B), (4, 4), (B + 3, B)):
            ctl = AdaptiveVerifyWidth(cfg=_cfg(force_width=force), curve=curve)
            self.assertEqual({ctl.next_width(batch_size=1) for _ in range(20)}, {want})


class TestRankUniformity(unittest.TestCase):
    def _run(self, feed):
        ctl = AdaptiveVerifyWidth(
            cfg=_cfg(min_dwell=4, margin_pct=2.0, probe_every=16, warmup_rounds=8),
            curve=_curve(a_ms=20.0, c_ms=1.5),
        )
        seq = []
        for rows in feed:
            w = ctl.next_width(batch_size=len(rows))
            seq.append(w)
            ctl.observe(num_correct_drafts=rows, width=w)
        return seq, state_digest(state=ctl.state)

    def test_same_inputs_same_widths_and_digest(self):
        """Two ranks fed the same synced counts pick the same width every
        round; the digest is what the worker cross-checks per census."""
        feed = [[(r * 7 + i * 3) % B for i in range(1 + r % 4)] for r in range(400)]
        a = self._run(feed)
        b = self._run(feed)
        self.assertEqual(a, b)
        self.assertGreater(len(set(a[0])), 1, "feed must exercise switches")

    def test_digest_detects_one_differing_count(self):
        """Negative contract: a single rank-local count must change the digest."""
        feed = [[3, 5]] * 100
        other = [list(r) for r in feed]
        other[50] = [3, 4]
        self.assertNotEqual(self._run(feed)[1], self._run(other)[1])


class TestHysteresis(unittest.TestCase):
    def test_margin_holds_incumbent_inside_band(self):
        """Candidate 1 % better than the incumbent: margin 2 % keeps it,
        margin 0 takes it."""
        ema = _ema(0.5)
        e4 = expected_accept_tokens(ema=ema, width=4)
        e8 = expected_accept_tokens(ema=ema, width=B)
        # c = 1 ms/row at bs1; solve e4/(a+4) = 1.01 * e8/(a+8) for a.
        a = (1.01 * 4 * e8 - 8 * e4) / (e4 - 1.01 * e8)
        curve = _curve(a_ms=a, c_ms=1.0, widths=(4, B))
        held = choose_width(cfg=_cfg(margin_pct=2.0), curve=curve, ema=ema, batch_size=1, incumbent=B)
        taken = choose_width(cfg=_cfg(margin_pct=0.0), curve=curve, ema=ema, batch_size=1, incumbent=B)
        self.assertEqual((held, taken), (B, 4))

    def test_dwell_and_margin_stop_flapping(self):
        """Acceptance alternating around the break-even of two widths flaps
        without hysteresis; dwell + margin bound switches to rounds/dwell."""
        # T(4)=18, T(8)=27 ms at bs1 sits at this feed's break-even: measured
        # 342 switches in 512 rounds without hysteresis, 48 with dwell 8 + 2 %.
        curve = _curve(a_ms=9.0, c_ms=2.25, widths=(4, B))

        def switches(cfg):
            ctl = AdaptiveVerifyWidth(cfg=cfg, curve=curve)
            for r in range(512):
                w = ctl.next_width(batch_size=1)
                ctl.observe(num_correct_drafts=[w - 1 if r % 3 == 0 else 1], width=w)
            return ctl.state.switch_ct

        bare = switches(_cfg(ema_alpha=0.5))
        damped = switches(_cfg(ema_alpha=0.5, min_dwell=8, margin_pct=2.0))
        self.assertGreater(bare, 128, "feed must flap without hysteresis")
        self.assertLessEqual(damped, 512 // 9)


class TestLearning(unittest.TestCase):
    def test_probe_rounds_run_block_and_keep_incumbent(self):
        """Every probe_every-th round verifies all N slots (fresh deep-slot
        evidence) without counting as a switch or dropping the incumbent."""
        ctl = AdaptiveVerifyWidth(cfg=_cfg(probe_every=10), curve=_curve(a_ms=10.0, c_ms=5.0))
        widths = []
        for _ in range(40):
            w = ctl.next_width(batch_size=1)
            widths.append(w)
            ctl.observe(num_correct_drafts=[0], width=w)
        self.assertEqual([widths[i] for i in (9, 19, 29, 39)], [B] * 4)
        self.assertEqual(ctl.state.incumbent, 2)
        self.assertEqual(ctl.state.probe_ct, 4)

    def test_tail_pull_lets_width_grow(self):
        """Closed loop from a pessimistic deep-slot prior: the true draft is
        always right. Without Basalt's pull (fully accepted rows drag the
        unverified tail toward the deepest verified rate) slots >= k never
        get evidence and the width stays small (mutation-checked: 5 != 8)."""
        ctl = AdaptiveVerifyWidth(
            cfg=_cfg(prior_accept_rate=0.1, ema_alpha=0.2),
            curve=_curve(a_ms=20.0, c_ms=0.5),
        )
        first = ctl.next_width(batch_size=1)
        ctl.observe(num_correct_drafts=[first - 1], width=first)
        self.assertLess(first, B)
        for _ in range(300):
            w = ctl.next_width(batch_size=1)
            ctl.observe(num_correct_drafts=[w - 1], width=w)
        self.assertEqual(ctl.state.incumbent, B)

    def test_observe_only_verified_slots(self):
        """A width-3 round with one correct draft: slot 1 hit, slot 2 miss,
        slots 3.. untouched (no full row -> no pull)."""
        out = observe_round(ema=_ema(0.5), num_correct_drafts=[1], width=3, alpha=0.5)
        self.assertEqual(out.accept_rate[:2], (0.75, 0.25))
        self.assertEqual(out.accept_rate[2:], (0.5,) * (B - 3))
        self.assertEqual(out.observed_ct[:3], (1, 1, 0))


class TestCurveFile(unittest.TestCase):
    def test_markdown_json_block_and_linear_cost(self):
        """A curve in Markdown (fenced json, the deskq report form) with the
        linear shorthand: cost is a + c * bs * k (rows), unknown widths are
        ineligible (inf)."""
        text = """# Kurve
```json
{"widths": [8, 5], "linear": {"a_ms": 21.5, "c_ms_per_row": 0.93}}
```
"""
        curve = parse_curve_text(text=text, source="x.md")
        self.assertEqual(curve.widths, (5, 8))
        self.assertAlmostEqual(round_cost_ms(curve=curve, width=5, batch_size=3), 21.5 + 0.93 * 15)
        self.assertTrue(math.isinf(round_cost_ms(curve=curve, width=6, batch_size=1)))

    def test_bundled_27b_curve_loads(self):
        """The shipped default (FLLIPER_ADAPTIVE_K_CURVE="") must parse with
        tiers {5,8} and A1's prior; a broken file would silently leave the
        default-ON feature inert."""
        from flliper.srt.speculative.dflash_verify_width import DEFAULT_CURVE_PATH

        curve = load_curve_file(path=DEFAULT_CURVE_PATH)
        self.assertIsNotNone(curve)
        self.assertEqual(curve.widths, (5, B))
        self.assertEqual(curve.prior_survival, A1_SURVIVAL)


if __name__ == "__main__":
    unittest.main()
