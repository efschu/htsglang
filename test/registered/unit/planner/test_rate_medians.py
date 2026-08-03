# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Unit tests for rate_medians.py -- the "0 (median 43.2)" badge behind the
landing-page rate tiles.

Hermetic: pure functions, no server, no GPU, no clock.

The load-bearing property is that IDLE POLLS NEVER ENTER THE WINDOW. A median
over every poll of a mostly-idle rig is 0 by construction, and a badge reading
"0 (median 0.0)" is worse than no badge at all. ``test_idle_polls_never_enter_
the_window`` is the falsifier: drop the delta predicates in
``processing_samples`` and it goes red.
"""

import unittest

from sglang.srt.planner import rate_medians as rm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _counters(gen=0.0, prompt=0.0, cached=0.0, running=None, accept=None):
    out = {
        "generation_tokens_total": gen,
        "prompt_tokens_total": prompt,
        "cached_total": cached,
        "num_running_reqs": running,
        "spec": None,
    }
    if accept is not None:
        out["spec"] = {"accept_rate": accept, "adaptive_k": 3,
                       "ema_accept_len": 2.5}
    return out


def _rates(dt=1.0, decode=0.0, prefill=0.0):
    return {"dt": dt, "decode_tok_s": decode, "prefill_tok_s": prefill,
            "prefill_tok_s_gross": prefill, "cached_tok_s": 0.0,
            "gen_throughput_server": decode}


class TestMedianOf(CustomTestCase):
    def test_empty_window_is_none_not_zero(self):
        self.assertIsNone(rm.median_of([]))

    def test_single_element(self):
        self.assertEqual(rm.median_of([43.2]), 43.2)

    def test_odd_and_even_lengths(self):
        self.assertEqual(rm.median_of([3.0, 1.0, 2.0]), 2.0)
        self.assertEqual(rm.median_of([4.0, 1.0, 3.0, 2.0]), 2.5)

    def test_order_independent(self):
        self.assertEqual(rm.median_of([9, 1, 5]), rm.median_of([5, 9, 1]))

    def test_a_real_zero_sample_still_counts(self):
        """Excluding idle polls is done by the PREDICATE, not by dropping
        zeros here -- a genuine measured 0 inside a processing window is a
        datum."""
        self.assertEqual(rm.median_of([0.0, 0.0, 10.0]), 0.0)


class TestProcessingPredicate(CustomTestCase):
    def test_idle_polls_never_enter_the_window(self):
        """THE falsifier: counters did not move, so nothing is sampled even
        though the rates dict is present and full of honest zeros."""
        prev = _counters(gen=1000, prompt=500, cached=100, running=0)
        cur = _counters(gen=1000, prompt=500, cached=100, running=0)
        self.assertEqual(
            rm.processing_samples(_rates(), prev, cur, None), {})

    def test_decode_activity_samples_decode_only(self):
        prev = _counters(gen=1000, prompt=500, cached=100)
        cur = _counters(gen=1080, prompt=500, cached=100)
        got = rm.processing_samples(_rates(decode=80.0), prev, cur, None)
        self.assertEqual(set(got), {"decode_tok_s"})
        self.assertEqual(got["decode_tok_s"], 80.0)

    def test_prefill_activity_samples_prefill_only(self):
        prev = _counters(gen=1000, prompt=500, cached=100)
        cur = _counters(gen=1000, prompt=900, cached=100)
        got = rm.processing_samples(_rates(prefill=400.0), prev, cur, None)
        self.assertEqual(set(got), {"prefill_tok_s"})

    def test_prompt_delta_fully_served_from_cache_is_not_prefill_work(self):
        prev = _counters(gen=1000, prompt=500, cached=100)
        cur = _counters(gen=1000, prompt=900, cached=500)   # all 400 cached
        got = rm.processing_samples(_rates(prefill=0.0), prev, cur,
                                    {"overall": 1.0})
        self.assertNotIn("prefill_tok_s", got)
        self.assertIn("cache_hit_overall", got)   # a prompt window DID exist

    def test_per_request_needs_a_running_request(self):
        prev = _counters(gen=1000, running=2)
        cur = _counters(gen=1080, running=2)
        got = rm.processing_samples(_rates(decode=80.0), prev, cur, None)
        self.assertEqual(got["decode_tok_s_per_request"], 40.0)
        cur0 = _counters(gen=1080, running=0)
        got0 = rm.processing_samples(_rates(decode=80.0), prev, cur0, None)
        self.assertNotIn("decode_tok_s_per_request", got0)

    def test_missing_concurrency_gauge_drops_the_per_request_sample_only(self):
        prev = _counters(gen=1000, running=None)
        cur = _counters(gen=1080, running=None)
        got = rm.processing_samples(_rates(decode=80.0), prev, cur, None)
        self.assertIn("decode_tok_s", got)
        self.assertNotIn("decode_tok_s_per_request", got)

    def test_spec_accept_is_sampled_only_while_decode_runs(self):
        prev = _counters(gen=1000, accept=0.8)
        moving = _counters(gen=1080, accept=0.8)
        idle = _counters(gen=1000, accept=0.8)
        self.assertIn("spec_accept_rate",
                      rm.processing_samples(_rates(decode=80.0), prev, moving, None))
        self.assertNotIn("spec_accept_rate",
                         rm.processing_samples(_rates(), prev, idle, None))

    def test_cache_hit_needs_a_prompt_window(self):
        prev = _counters(prompt=500)
        cur = _counters(prompt=500)
        self.assertNotIn("cache_hit_overall",
                         rm.processing_samples(_rates(), prev, cur,
                                               {"overall": 0.9}))

    def test_no_rates_or_no_prev_yields_nothing(self):
        cur = _counters(gen=1080)
        self.assertEqual(rm.processing_samples(None, _counters(), cur, None), {})
        self.assertEqual(rm.processing_samples(_rates(), None, cur, None), {})

    def test_nonpositive_dt_yields_nothing(self):
        prev = _counters(gen=1000)
        cur = _counters(gen=1080)
        self.assertEqual(
            rm.processing_samples(_rates(dt=0.0, decode=0.0), prev, cur, None), {})


class TestWindowRolling(CustomTestCase):
    def test_rolls_at_the_window_size(self):
        w = None
        for i in range(rm.RATE_MEDIAN_WINDOW + 10):
            w = rm.update_windows(w, {"decode_tok_s": float(i)})
        self.assertEqual(len(w["decode_tok_s"]), rm.RATE_MEDIAN_WINDOW)
        self.assertEqual(w["decode_tok_s"][0],
                         float(rm.RATE_MEDIAN_WINDOW + 10 - rm.RATE_MEDIAN_WINDOW))
        self.assertEqual(w["decode_tok_s"][-1], float(rm.RATE_MEDIAN_WINDOW + 9))

    def test_an_idle_tick_does_not_age_history_out(self):
        w = rm.update_windows(None, {"decode_tok_s": 40.0})
        for _ in range(100):
            w = rm.update_windows(w, {})          # 100 idle polls
        self.assertEqual(w["decode_tok_s"], [40.0])
        self.assertEqual(rm.medians(w)["decode_tok_s"]["median"], 40.0)

    def test_unknown_keys_are_dropped_from_carried_state(self):
        w = rm.update_windows({"bogus": [1.0]}, {"decode_tok_s": 2.0})
        self.assertEqual(set(w), {"decode_tok_s"})

    def test_previous_window_is_not_mutated(self):
        prev = {"decode_tok_s": [1.0]}
        rm.update_windows(prev, {"decode_tok_s": 2.0})
        self.assertEqual(prev, {"decode_tok_s": [1.0]})

    def test_custom_window_size_is_honoured(self):
        w = None
        for i in range(10):
            w = rm.update_windows(w, {"decode_tok_s": float(i)}, window=3)
        self.assertEqual(w["decode_tok_s"], [7.0, 8.0, 9.0])


class TestMediansBlock(CustomTestCase):
    def test_empty_window_produces_no_entry_at_all(self):
        self.assertEqual(rm.medians(None), {})
        self.assertEqual(rm.medians({"decode_tok_s": []}), {})

    def test_one_element_reports_n_one(self):
        got = rm.medians(rm.update_windows(None, {"decode_tok_s": 43.2}))
        self.assertEqual(got["decode_tok_s"],
                         {"median": 43.2, "n": 1, "window": rm.RATE_MEDIAN_WINDOW})

    def test_the_idle_scenario_end_to_end(self):
        """Three processing polls, then idle: the tile shows 0 and the badge
        keeps the median of the three -- not 0."""
        w, prev = None, _counters(gen=0, prompt=0, cached=0, running=1)
        for gen, dec in ((100, 100.0), (150, 50.0), (190, 40.0)):
            cur = _counters(gen=gen, prompt=0, cached=0, running=1)
            w = rm.update_windows(
                w, rm.processing_samples(_rates(decode=dec), prev, cur, None))
            prev = cur
        for _ in range(20):                       # idle polls
            cur = dict(prev)
            w = rm.update_windows(
                w, rm.processing_samples(_rates(), prev, cur, None))
            prev = cur
        self.assertEqual(rm.medians(w)["decode_tok_s"]["median"], 50.0)
        self.assertEqual(rm.medians(w)["decode_tok_s"]["n"], 3)

    def test_window_constant_is_documented_and_sane(self):
        self.assertEqual(rm.RATE_MEDIAN_WINDOW, 30)
        self.assertIn("60 s", rm.__doc__)


if __name__ == "__main__":
    unittest.main()
