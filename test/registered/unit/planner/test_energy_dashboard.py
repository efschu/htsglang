# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Unit tests for the energy-dashboard Bundle 1 (#149 + #150) — no GPU, no boot.

Covers the pure logic added for the power-comparability / live-monitoring
feature (#149) and the configurable scenario builder (#150):

  * power-state TAGGING schema + OC/UV classification (fake NVML),
  * power-limit efficiency SWEEP with the ALWAYS-restore guarantee (incl. the
    restore-on-exception path) and constraint clamping,
  * LIVE delta-rate math (client-side counter deltas, reset/dt<=0 guards),
  * SCENARIO expansion (prefill-once/decode-split, multiturn growing context,
    validation) and
  * the cache-flush-warning gate (mandatory vs informative).

Also asserts the work-normalized per-card efficiency column.
"""

import unittest

from sglang.srt.planner.energy import (
    CODE_BEHAVIOR,
    PROSE_BEHAVIOR,
    GpuPowerState,
    LiveSnapshot,
    PowerLimitController,
    ScenarioConfig,
    SweepResult,
    _work_per_watt_rel,
    cache_flush_warning,
    clamp_live_resolution_ms,
    compute_live_rates,
    parse_prometheus_metrics,
    power_limit_sweep,
    read_gpu_power_states,
)


class _FakeCard:
    def __init__(self, name, watts, limit, default, mn, mx, sm, mem, temp,
                 offset=None):
        self.name = name
        self.watts = watts          # W
        self.limit = limit          # W current enforced
        self.default = default      # W stock TDP
        self.mn = mn
        self.mx = mx
        self.sm = sm
        self.mem = mem
        self.temp = temp
        self.offset = offset
        self.set_calls = []         # every SetPowerManagementLimit (mW)


class _FakeNvml:
    """Minimal pynvml stand-in driven by a list of _FakeCard."""

    NVML_CLOCK_SM = 1
    NVML_CLOCK_MEM = 2
    NVML_TEMPERATURE_GPU = 0

    def __init__(self, cards):
        self.cards = cards
        self.init_calls = 0
        self.shutdown_calls = 0

    def nvmlInit(self):
        self.init_calls += 1

    def nvmlShutdown(self):
        self.shutdown_calls += 1

    def nvmlDeviceGetCount(self):
        return len(self.cards)

    def nvmlDeviceGetHandleByIndex(self, i):
        return self.cards[i]

    def nvmlDeviceGetName(self, h):
        return h.name

    def nvmlDeviceGetPowerUsage(self, h):
        return h.watts * 1000.0

    def nvmlDeviceGetPowerManagementLimit(self, h):
        return h.limit * 1000.0

    def nvmlDeviceGetPowerManagementDefaultLimit(self, h):
        return h.default * 1000.0

    def nvmlDeviceGetPowerManagementLimitConstraints(self, h):
        return (h.mn * 1000.0, h.mx * 1000.0)

    def nvmlDeviceSetPowerManagementLimit(self, h, mw):
        h.set_calls.append(mw)
        h.limit = mw / 1000.0

    def nvmlDeviceGetClockInfo(self, h, which):
        return h.sm if which == self.NVML_CLOCK_SM else h.mem

    def nvmlDeviceGetTemperature(self, h, sensor):
        return h.temp

    def nvmlDeviceGetGpcClkVfOffset(self, h):
        if h.offset is None:
            raise RuntimeError("offset not supported")
        return h.offset


def _rig():
    return _FakeNvml([
        _FakeCard("RTX 5090", watts=380, limit=575, default=575,
                  mn=400, mx=600, sm=2800, mem=1500, temp=62),
        _FakeCard("RTX 3080", watts=280, limit=320, default=320,
                  mn=100, mx=370, sm=1900, mem=1180, temp=70),
    ])


# ---------------------------------------------------------------------------
# #149 — power-state tagging.
# ---------------------------------------------------------------------------


class TestPowerStateTagging(unittest.TestCase):
    def test_tag_schema(self):
        states = read_gpu_power_states(nvml=_rig())
        self.assertEqual(len(states), 2)
        s0 = states[0]
        self.assertIsInstance(s0, GpuPowerState)
        self.assertEqual(s0.name, "RTX 5090")
        self.assertAlmostEqual(s0.power_watts, 380.0)
        self.assertAlmostEqual(s0.power_limit_w, 575.0)
        self.assertAlmostEqual(s0.default_limit_w, 575.0)
        self.assertAlmostEqual(s0.limit_pct_of_default, 1.0)
        self.assertEqual(s0.sm_clock_mhz, 2800)
        self.assertEqual(s0.mem_clock_mhz, 1500)
        self.assertEqual(s0.temperature_c, 62)
        self.assertEqual(s0.oc_uv, "stock")   # limit == default
        self.assertIn("nvml_index", s0.to_json())

    def test_power_limited_and_raised_classification(self):
        rig = _FakeNvml([
            _FakeCard("A", 200, limit=250, default=320, mn=100, mx=370,
                      sm=1, mem=1, temp=1),   # limit < default
            _FakeCard("B", 200, limit=360, default=320, mn=100, mx=370,
                      sm=1, mem=1, temp=1),   # limit > default
        ])
        states = read_gpu_power_states(nvml=rig)
        self.assertEqual(states[0].oc_uv, "power-limited")
        self.assertEqual(states[1].oc_uv, "power-raised")

    def test_clock_offset_overrides_to_oc_uv(self):
        rig = _FakeNvml([
            _FakeCard("OC", 200, 320, 320, 100, 370, 1, 1, 1, offset=150),
            _FakeCard("UV", 200, 320, 320, 100, 370, 1, 1, 1, offset=-120),
        ])
        states = read_gpu_power_states(nvml=rig)
        self.assertEqual(states[0].oc_uv, "clock-OC")
        self.assertEqual(states[0].clock_offset_mhz, 150)
        self.assertEqual(states[1].oc_uv, "clock-UV")

    def test_injected_nvml_not_shutdown(self):
        rig = _rig()
        read_gpu_power_states(nvml=rig)
        self.assertEqual(rig.shutdown_calls, 0)  # caller owns injected nvml


# ---------------------------------------------------------------------------
# #149 — power-limit controller + sweep (ALWAYS restore).
# ---------------------------------------------------------------------------


class TestPowerLimitController(unittest.TestCase):
    def test_restore_returns_original_limits(self):
        rig = _rig()
        ctrl = PowerLimitController(nvml=rig)
        applied = ctrl.set_limit_pct(0.8)
        # 80% of each stock TDP, within constraints (460W > 5090 min 400W)
        self.assertAlmostEqual(applied[0], 575 * 0.8, places=0)
        self.assertAlmostEqual(rig.cards[0].limit, 575 * 0.8, places=0)
        ctrl.restore()
        self.assertAlmostEqual(rig.cards[0].limit, 575.0)
        self.assertAlmostEqual(rig.cards[1].limit, 320.0)

    def test_set_pct_clamps_below_min(self):
        # 60% of the 5090's 575W = 345W, below its 400W min -> clamped up.
        rig = _rig()
        ctrl = PowerLimitController(nvml=rig)
        applied = ctrl.set_limit_pct(0.6)
        self.assertAlmostEqual(applied[0], 400.0)          # clamped to min
        self.assertAlmostEqual(applied[1], 320 * 0.6, places=0)  # 3080 ok
        ctrl.restore()

    def test_restore_is_idempotent(self):
        rig = _rig()
        ctrl = PowerLimitController(nvml=rig)
        ctrl.set_limit_watts(200)
        ctrl.restore()
        n = len(rig.cards[0].set_calls)
        ctrl.restore()  # second restore must not re-issue sets
        self.assertEqual(len(rig.cards[0].set_calls), n)

    def test_clamp_to_constraints(self):
        rig = _rig()
        ctrl = PowerLimitController(nvml=rig)
        # request below the 3080's 100W min and above the 5090's 600W max
        applied = ctrl.set_limit_watts(5)     # -> clamped up to each min
        self.assertAlmostEqual(applied[0], 400.0)   # 5090 min
        self.assertAlmostEqual(applied[1], 100.0)   # 3080 min
        applied = ctrl.set_limit_watts(9999)  # -> clamped down to each max
        self.assertAlmostEqual(applied[0], 600.0)
        self.assertAlmostEqual(applied[1], 370.0)
        ctrl.restore()

    def test_context_manager_restores_on_exception(self):
        rig = _rig()
        try:
            with PowerLimitController(nvml=rig) as ctrl:
                ctrl.set_limit_pct(0.6)
                raise RuntimeError("boom mid-sweep")
        except RuntimeError:
            pass
        # restored despite the exception
        self.assertAlmostEqual(rig.cards[0].limit, 575.0)
        self.assertAlmostEqual(rig.cards[1].limit, 320.0)


class _FakePoint:
    def __init__(self, decode_tok_s, j_per_decode_token, avg_decode_watts):
        self.decode_tok_s = decode_tok_s
        self.j_per_decode_token = j_per_decode_token
        self.avg_decode_watts = avg_decode_watts
        self.prefill_tok_s = 0.0
        self.j_per_prefill_token = 0.0


class TestPowerLimitSweep(unittest.TestCase):
    def test_sweep_records_points_and_restores(self):
        rig = _rig()
        seen_limits = []

        def measure():
            # capture the limit the card is at during this rung
            seen_limits.append(round(rig.cards[0].limit))
            # fake: efficiency improves as the limit drops
            return _FakePoint(decode_tok_s=100.0, j_per_decode_token=rig.cards[0].limit / 100.0,
                              avg_decode_watts=rig.cards[0].limit)

        res = power_limit_sweep(
            measure, pcts=(1.0, 0.8, 0.6), nvml=rig, settle_s=0,
            sleep_fn=lambda s: None)
        self.assertIsInstance(res, SweepResult)
        self.assertEqual(len(res.points), 3)
        # measured at descending caps; 60% (345W) clamps up to the 5090's
        # 400W min -> the physical floor, not the requested value.
        self.assertEqual(seen_limits, [575, round(575 * 0.8), 400])
        # ALWAYS restored to stock afterwards
        self.assertAlmostEqual(rig.cards[0].limit, 575.0)
        # x-axis modes
        self.assertEqual(res.x_axis("pct_tdp"), [1.0, 0.8, 0.6])
        self.assertEqual(len(res.x_axis("abs_w")), 3)
        with self.assertRaises(ValueError):
            res.x_axis("nonsense")
        # most efficient = lowest J/tok = the 60% rung
        self.assertAlmostEqual(res.most_efficient().limit_pct, 0.6)
        self.assertAlmostEqual(res.stock_total_watts, 575 + 320)

    def test_sweep_restores_when_measure_raises(self):
        rig = _rig()

        def measure():
            raise RuntimeError("measurement blew up")

        with self.assertRaises(RuntimeError):
            power_limit_sweep(measure, pcts=(0.6,), nvml=rig, settle_s=0,
                              sleep_fn=lambda s: None)
        self.assertAlmostEqual(rig.cards[0].limit, 575.0)  # still restored
        self.assertAlmostEqual(rig.cards[1].limit, 320.0)


# ---------------------------------------------------------------------------
# #149 — live delta-rate math.
# ---------------------------------------------------------------------------


class TestLiveRates(unittest.TestCase):
    def _snap(self, t, p, g, ar=0.0, k=0.0, ema=0.0):
        return LiveSnapshot(t=t, prompt_tokens_total=p, generation_tokens_total=g,
                            spec_accept_rate=ar, spec_num_steps=k,
                            spec_ema_accept_len=ema, gen_throughput=0.0)

    def test_counter_delta_rates(self):
        prev = self._snap(100.0, 1000, 5000)
        cur = self._snap(100.5, 1000 + 250, 5000 + 100, ar=0.8, k=3, ema=2.6)
        r = compute_live_rates(prev, cur)
        self.assertAlmostEqual(r["prefill_tok_s"], 250 / 0.5)   # 500 tok/s
        self.assertAlmostEqual(r["decode_tok_s"], 100 / 0.5)    # 200 tok/s
        self.assertAlmostEqual(r["accept_rate"], 0.8)
        self.assertAlmostEqual(r["adaptive_k"], 3)
        self.assertAlmostEqual(r["ema_accept_len"], 2.6)

    def test_zero_or_negative_dt_is_safe(self):
        prev = self._snap(100.0, 1000, 5000)
        r = compute_live_rates(prev, self._snap(100.0, 2000, 6000))
        self.assertEqual(r["prefill_tok_s"], 0.0)
        self.assertEqual(r["decode_tok_s"], 0.0)

    def test_counter_reset_does_not_go_negative(self):
        prev = self._snap(100.0, 5000, 9000)
        cur = self._snap(101.0, 10, 20)   # server restarted -> counters reset
        r = compute_live_rates(prev, cur)
        self.assertEqual(r["prefill_tok_s"], 0.0)
        self.assertEqual(r["decode_tok_s"], 0.0)

    def test_resolution_floor(self):
        self.assertEqual(clamp_live_resolution_ms(5), 30.0)
        self.assertEqual(clamp_live_resolution_ms(250), 250.0)
        self.assertEqual(clamp_live_resolution_ms("bad"), 250.0)

    def test_parse_prometheus_sums_label_sets(self):
        text = (
            "# HELP sglang:generation_tokens_total foo\n"
            "# TYPE sglang:generation_tokens_total counter\n"
            'sglang:generation_tokens_total{worker="0"} 100\n'
            'sglang:generation_tokens_total{worker="1"} 55\n'
            "sglang:prompt_tokens_total 1000\n"
            'sglang:spec_ema_accept_len{model="m"} 2.75\n'
        )
        m = parse_prometheus_metrics(text)
        self.assertAlmostEqual(m["sglang:generation_tokens_total"], 155.0)
        self.assertAlmostEqual(m["sglang:prompt_tokens_total"], 1000.0)
        self.assertAlmostEqual(m["sglang:spec_ema_accept_len"], 2.75)
        snap = LiveSnapshot.from_metrics(m, t=1.0)
        self.assertAlmostEqual(snap.generation_tokens_total, 155.0)
        self.assertAlmostEqual(snap.spec_ema_accept_len, 2.75)


# ---------------------------------------------------------------------------
# #150 — scenario builder.
# ---------------------------------------------------------------------------


class TestScenarioConfig(unittest.TestCase):
    def test_prefill_once_decode_split(self):
        units = ScenarioConfig(phases="both").expand()
        prefill = [u for u in units if u["phase"] == "prefill"]
        decode = [u for u in units if u["phase"] == "decode"]
        # PREFILL measured ONCE (content-independent), DECODE split code+prose
        self.assertEqual(len(prefill), 1)
        self.assertEqual(prefill[0]["behavior"], "any")
        self.assertEqual({u["behavior"] for u in decode},
                         {CODE_BEHAVIOR, PROSE_BEHAVIOR})

    def test_short_test_defaults(self):
        cfg = ScenarioConfig()
        self.assertEqual(cfg.scaled_prefill_tokens(), 10_000)
        self.assertEqual(cfg.scaled_decode_tokens(), 1_000)
        self.assertAlmostEqual(cfg.duration_s, 10.0)

    def test_scale(self):
        cfg = ScenarioConfig(scale=0.1)
        self.assertEqual(cfg.scaled_prefill_tokens(), 1_000)
        self.assertEqual(cfg.scaled_decode_tokens(), 100)

    def test_phases_prefill_only_and_decode_only(self):
        self.assertTrue(all(u["phase"] == "prefill"
                            for u in ScenarioConfig(phases="prefill").expand()))
        self.assertTrue(all(u["phase"] == "decode"
                            for u in ScenarioConfig(phases="decode").expand()))

    def test_multiturn_growing_context(self):
        cfg = ScenarioConfig(phases="prefill", multiturn=True, turns=3,
                             prefill_tokens=10_000, turn_growth_tokens=2_000)
        units = cfg.expand()
        self.assertEqual(len(units), 3)
        ctx = [u["prompt_tokens"] for u in units]
        self.assertEqual(ctx, [10_000, 12_000, 14_000])   # grows -> re-prefill
        self.assertEqual([u["turn"] for u in units], [0, 1, 2])

    def test_cold_prefill_flag(self):
        cold = ScenarioConfig(phases="prefill", cold_prefill=True).expand()
        self.assertTrue(cold[0]["cold"])
        warm = ScenarioConfig(phases="prefill", cold_prefill=False).expand()
        self.assertFalse(warm[0]["cold"])

    def test_validation(self):
        with self.assertRaises(ValueError):
            ScenarioConfig(phases="bogus")
        with self.assertRaises(ValueError):
            ScenarioConfig(concurrency=0)
        with self.assertRaises(ValueError):
            ScenarioConfig(scale=0)
        with self.assertRaises(ValueError):
            ScenarioConfig(behaviors=("code", "sql"))
        with self.assertRaises(ValueError):
            ScenarioConfig(multiturn=True, turns=0)

    def test_concurrency_propagates(self):
        units = ScenarioConfig(phases="both", concurrency=8).expand()
        self.assertTrue(all(u["concurrency"] == 8 for u in units))


# ---------------------------------------------------------------------------
# #150 — cache-flush warning gate.
# ---------------------------------------------------------------------------


class TestCacheFlushWarning(unittest.TestCase):
    def test_mandatory_for_running_server(self):
        w = cache_flush_warning(will_flush=True, target_running_server=True)
        self.assertTrue(w["warn"])
        self.assertTrue(w["mandatory"])
        self.assertIn("lost", w["message"].lower())
        self.assertIn("/flush_cache", w["endpoints"])
        self.assertIn("/clear_hicache_storage_backend", w["endpoints"])

    def test_informative_for_fresh_server(self):
        w = cache_flush_warning(will_flush=True, target_running_server=False)
        self.assertTrue(w["warn"])
        self.assertFalse(w["mandatory"])   # non-blocking

    def test_no_warning_when_not_flushing(self):
        w = cache_flush_warning(will_flush=False, target_running_server=True)
        self.assertFalse(w["warn"])
        self.assertFalse(w["mandatory"])
        self.assertEqual(w["endpoints"], [])


# ---------------------------------------------------------------------------
# Work-normalized per-card efficiency column (the coordinator's column).
# ---------------------------------------------------------------------------


class TestWorkPerWattColumn(unittest.TestCase):
    def test_relative_efficiency_least_efficient_is_one(self):
        # 5090 does 46% of the work at ~equal J/tok as the 3080s (27% each) ->
        # the 5090 is the efficient workhorse, the 3080s the relative drain.
        shares = [0.46, 0.27, 0.27]
        j_per_tok = [2.90, 2.97, 2.97]
        rel = _work_per_watt_rel(shares, j_per_tok)
        self.assertAlmostEqual(rel[1], 1.0)   # 3080 = baseline
        self.assertAlmostEqual(rel[2], 1.0)
        self.assertGreater(rel[0], 1.6)       # 5090 clearly more efficient
        self.assertLess(rel[0], 1.9)          # ~1.78x

    def test_none_shares_yield_none(self):
        rel = _work_per_watt_rel([None, None], [2.0, 3.0])
        self.assertEqual(rel, [None, None])

    def test_zero_jtok_is_safe(self):
        rel = _work_per_watt_rel([0.5, 0.5], [0.0, 2.0])
        self.assertIsNone(rel[0])
        self.assertAlmostEqual(rel[1], 1.0)


if __name__ == "__main__":
    unittest.main()
