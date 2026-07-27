"""CPU unit tests for rate semantics: group throughput vs per-rank work."""

import unittest

from sglang.srt.rigmon.rates import (
    PeakCapability,
    engine_rank_rates,
    group_throughput,
    idle_floor_from_power_profile,
    pacing_rank,
    peaks_from_hw_profile,
    rank_shares,
)
from sglang.srt.rigmon.sources import CardSample
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


#: Shaped after the real cached probe on this rig.
HW_PROFILE = {
    "created": "2026-07-21 07:06:43",
    "gpus": {
        "GPU-5090": {"name": "RTX 5090", "gemm_tflops": 233.91, "membw_gbs": 1664.1},
        "GPU-3080a": {"name": "RTX 3080", "gemm_tflops": 63.17, "membw_gbs": 718.2},
        "GPU-3080b": {"name": "RTX 3080", "gemm_tflops": 61.24, "membw_gbs": 718.2},
    },
    "links": {"GPU-5090|GPU-3080a": {"p2p_gbs": 5.11}},
}
POWER_PROFILE = {
    "cards": [
        {"uuid": "GPU-5090", "p_idle_w": 47.8},
        {"uuid": "GPU-3080a", "p_idle_w": 73.2},
        {"uuid": "GPU-3080b", "p_idle_w": 72.5},
    ]
}


def card(index, uuid, name, **kw):
    base = dict(
        mem_total_mib=32607,
        mem_used_mib=20000,
        temp_c=60.0,
        power_w=200.0,
        sm_clock_mhz=1900,
        sm_clock_max_mhz=2100,
        util_gpu_pct=50.0,
        util_mem_pct=40.0,
        sm_active=0.5,
        dram_active=0.4,
        activity_source="nvml-gpm",
    )
    base.update(kw)
    return CardSample(index=index, uuid=uuid, name=name, **base)


class TestGroupThroughput(CustomTestCase):
    def test_no_per_card_breakdown_is_offered(self):
        gt = group_throughput({"gen_throughput": 42.0})
        d = gt.to_json()
        self.assertEqual(d["gen_tok_s"], 42.0)
        self.assertNotIn("per_rank_tok_s", d)
        self.assertIn("invented", d["per_card_note"])

    def test_counter_delta_is_preferred_over_the_gauge(self):
        gt = group_throughput(
            {"generation_tokens_total": 2000.0, "gen_throughput": 999.0},
            {"generation_tokens_total": 1000.0},
            dt_s=10.0,
        )
        self.assertAlmostEqual(gt.gen_tok_s, 100.0)
        self.assertIn("counter delta", gt.source)

    def test_gauge_fallback_names_itself(self):
        gt = group_throughput({"gen_throughput": 12.0})
        self.assertIn("engine-internal window", gt.source)

    def test_counter_reset_falls_back_instead_of_going_negative(self):
        gt = group_throughput(
            {"generation_tokens_total": 5.0, "gen_throughput": 7.0},
            {"generation_tokens_total": 1000.0},
            dt_s=10.0,
        )
        self.assertEqual(gt.gen_tok_s, 7.0)

    def test_engine_down(self):
        gt = group_throughput(None)
        self.assertIsNone(gt.gen_tok_s)
        self.assertIn("unreachable", gt.source)


class TestPeaks(CustomTestCase):
    def test_peaks_from_probe_and_power_profile(self):
        peaks = peaks_from_hw_profile(HW_PROFILE, POWER_PROFILE)
        self.assertAlmostEqual(peaks["GPU-5090"].membw_gbs, 1664.1)
        self.assertAlmostEqual(peaks["GPU-5090"].idle_w, 47.8)
        self.assertEqual(peaks["GPU-5090"].probe_created, "2026-07-21 07:06:43")

    def test_idle_floor_extraction(self):
        self.assertEqual(idle_floor_from_power_profile(POWER_PROFILE)["GPU-3080a"], 73.2)
        self.assertEqual(idle_floor_from_power_profile(None), {})


class TestRankShares(CustomTestCase):
    def _cards(self):
        return [
            card(0, "GPU-5090", "RTX 5090", sm_active=0.9, dram_active=0.8,
                 tensor_active=0.30, power_w=400.0),
            card(1, "GPU-3080a", "RTX 3080", sm_active=0.4, dram_active=0.5,
                 tensor_active=0.10, power_w=180.0),
        ]

    def test_achieved_is_peak_times_activity(self):
        v = rank_shares(self._cards(), peaks_from_hw_profile(HW_PROFILE, POWER_PROFILE))
        r0 = v.ranks[0]
        self.assertAlmostEqual(r0.achieved_gbs, 0.8 * 1664.1, places=3)
        self.assertAlmostEqual(r0.achieved_tflops, 0.30 * 233.91, places=3)
        self.assertAlmostEqual(r0.membw_achieved_frac, 0.8)

    def test_active_and_wait_are_reported_separately(self):
        v = rank_shares(self._cards(), peaks_from_hw_profile(HW_PROFILE))
        r0, r1 = v.ranks
        self.assertAlmostEqual(r0.active_share, 0.9)
        self.assertAlmostEqual(r0.wait_share, 0.1)
        self.assertAlmostEqual(r1.wait_share, 0.6)

    def test_work_share_sums_to_one(self):
        v = rank_shares(self._cards(), peaks_from_hw_profile(HW_PROFILE))
        self.assertAlmostEqual(sum(r.byte_work_share for r in v.ranks), 1.0, places=6)
        self.assertAlmostEqual(sum(r.flop_work_share for r in v.ranks), 1.0, places=6)

    def test_work_per_watt_has_a_dynamic_variant(self):
        """Total-power efficiency punishes a fast card for waiting; the
        above-idle variant is the measured decomposition that does not."""
        v = rank_shares(self._cards(), peaks_from_hw_profile(HW_PROFILE, POWER_PROFILE))
        r0 = v.ranks[0]
        self.assertAlmostEqual(r0.idle_w, 47.8)
        self.assertAlmostEqual(r0.dynamic_w, 400.0 - 47.8, places=3)
        self.assertGreater(r0.gbs_per_dynamic_w, r0.gbs_per_total_w)

    def test_roofline_position(self):
        cards = [
            card(0, "GPU-5090", "RTX 5090", sm_active=0.9, dram_active=0.9,
                 tensor_active=0.01),  # lots of bytes, little compute
        ]
        v = rank_shares(cards, peaks_from_hw_profile(HW_PROFILE))
        self.assertEqual(v.ranks[0].bound_by, "memory")
        cards[0].tensor_active = 0.95
        cards[0].dram_active = 0.05
        v = rank_shares(cards, peaks_from_hw_profile(HW_PROFILE))
        self.assertEqual(v.ranks[0].bound_by, "compute")

    def test_idle_card_is_not_called_memory_bound(self):
        cards = [card(0, "GPU-5090", "RTX 5090", sm_active=0.0, dram_active=0.0,
                      tensor_active=0.0)]
        v = rank_shares(cards, peaks_from_hw_profile(HW_PROFILE))
        self.assertEqual(v.ranks[0].bound_by, "idle")

    def test_colocated_ranks_do_not_double_count_the_card(self):
        """Two ranks on one physical card share its power and its work; the
        group sums must count the card once."""
        cards = [card(0, "GPU-5090", "RTX 5090", power_w=400.0)]
        v = rank_shares(cards, peaks_from_hw_profile(HW_PROFILE), rank_gpu_id=[0, 0])
        self.assertEqual(len(v.ranks), 2)
        self.assertEqual([r.rank for r in v.ranks], [0, 1])
        self.assertAlmostEqual(v.group_power_w, 400.0)

    def test_rank_gpu_id_maps_ranks_onto_cards(self):
        cards = self._cards()
        v = rank_shares(cards, {}, rank_gpu_id=[1, 0])
        self.assertEqual([r.rank for r in v.ranks], [0, 1])
        self.assertEqual([r.gpu_index for r in v.ranks], [1, 0])

    def test_throttles_are_surfaced_but_idle_is_not_a_throttle(self):
        cards = [
            card(0, "GPU-5090", "RTX 5090", throttle=["gpu_idle"]),
            card(1, "GPU-3080a", "RTX 3080", throttle=["sw_thermal_slowdown"], temp_c=88.0),
        ]
        v = rank_shares(cards, {})
        self.assertEqual(v.ranks[0].throttles, [])
        self.assertEqual(v.ranks[1].throttles, ["sw_thermal_slowdown"])

    def test_clock_ratio_exposes_a_held_back_card(self):
        cards = [card(0, "GPU-3080a", "RTX 3080", sm_clock_mhz=1695,
                      sm_clock_max_mhz=1905)]
        v = rank_shares(cards, {})
        self.assertAlmostEqual(v.ranks[0].clock_ratio, 1695 / 1905, places=5)


class TestCaveats(CustomTestCase):
    def test_power_comparability_is_always_stated(self):
        v = rank_shares([card(0, "GPU-5090", "RTX 5090")], {})
        keys = {c.key for c in v.caveats}
        self.assertIn("power_comparability", keys)
        self.assertTrue(v.group_power_approximate)

    def test_missing_probe_is_stated(self):
        v = rank_shares([card(0, "GPU-5090", "RTX 5090")], {})
        self.assertIn("no_peaks", {c.key for c in v.caveats})

    def test_coarse_activity_is_stated_and_tensor_stays_absent(self):
        c = card(0, "GPU-5090", "RTX 5090",
                 activity_source="nvml-utilization (coarse fallback)",
                 tensor_active=None)
        v = rank_shares([c], peaks_from_hw_profile(HW_PROFILE))
        self.assertIn("coarse_activity", {x.key for x in v.caveats})
        self.assertIsNone(v.ranks[0].achieved_tflops)
        self.assertIsNotNone(v.ranks[0].achieved_gbs)

    def test_wait_caveat_only_when_wait_is_known(self):
        c = card(0, "GPU-5090", "RTX 5090", sm_active=None)
        v = rank_shares([c], {})
        self.assertNotIn("wait_costs_power", {x.key for x in v.caveats})


class TestPacingRank(CustomTestCase):
    def test_lowest_wait_share_paces_the_group(self):
        cards = [
            card(0, "GPU-5090", "RTX 5090", sm_active=0.95),
            card(1, "GPU-3080a", "RTX 3080", sm_active=0.40),
        ]
        v = rank_shares(cards, {}, rank_gpu_id=[0, 1])
        self.assertEqual(v.pacer_rank, 0)
        self.assertIn("waits on this rank", v.pacer_basis)

    def test_no_pacer_named_when_the_spread_is_noise(self):
        cards = [
            card(0, "GPU-5090", "RTX 5090", sm_active=0.50),
            card(1, "GPU-3080a", "RTX 3080", sm_active=0.53),
        ]
        v = rank_shares(cards, {}, rank_gpu_id=[0, 1])
        self.assertIsNone(v.pacer_rank)
        self.assertIn("noise", v.pacer_basis)

    def test_coarse_source_is_flagged_in_the_basis(self):
        cards = [
            card(0, "GPU-5090", "RTX 5090", sm_active=0.95,
                 activity_source="nvml-utilization (coarse fallback)"),
            card(1, "GPU-3080a", "RTX 3080", sm_active=0.30,
                 activity_source="nvml-utilization (coarse fallback)"),
        ]
        v = rank_shares(cards, {}, rank_gpu_id=[0, 1])
        self.assertIn("direction, not a measurement", v.pacer_basis)

    def test_single_rank_has_no_pacer(self):
        self.assertEqual(pacing_rank([]), (None, None))


class TestEngineDerivedWork(CustomTestCase):
    """The engine already publishes per-TP-rank busy time and estimated work.
    That is an exact attribution and must outrank the device counters."""

    def _rates(self, dt=2.0):
        prev = {
            0: {"forward_execution_seconds_total": 10.0,
                "estimated_flops_per_gpu_total": 1e12,
                "estimated_read_bytes_per_gpu_total": 1e9,
                "estimated_write_bytes_per_gpu_total": 0.0},
            1: {"forward_execution_seconds_total": 10.0,
                "estimated_flops_per_gpu_total": 1e12,
                "estimated_read_bytes_per_gpu_total": 1e9,
                "estimated_write_bytes_per_gpu_total": 0.0},
        }
        cur = {
            0: {"forward_execution_seconds_total": 11.8,
                "estimated_flops_per_gpu_total": 1e12 + 200e12,
                "estimated_read_bytes_per_gpu_total": 1e9 + 800e9,
                "estimated_write_bytes_per_gpu_total": 200e9},
            1: {"forward_execution_seconds_total": 10.8,
                "estimated_flops_per_gpu_total": 1e12 + 40e12,
                "estimated_read_bytes_per_gpu_total": 1e9 + 300e9,
                "estimated_write_bytes_per_gpu_total": 100e9},
        }
        return engine_rank_rates(cur, prev, dt)

    def test_busy_time_delta_is_the_active_share(self):
        r = self._rates()
        self.assertAlmostEqual(r[0]["active_share"], 0.9)
        self.assertAlmostEqual(r[1]["active_share"], 0.4)

    def test_estimated_counters_become_achieved_rates(self):
        r = self._rates()
        self.assertAlmostEqual(r[0]["achieved_tflops"], 100.0)
        self.assertAlmostEqual(r[0]["achieved_gbs"], 500.0)

    def test_reads_and_writes_are_both_counted(self):
        r = self._rates()
        # rank 1: (300 + 100) GB over 2 s
        self.assertAlmostEqual(r[1]["achieved_gbs"], 200.0)

    def test_engine_rates_outrank_coarse_device_counters(self):
        cards = [
            card(0, "GPU-5090", "RTX 5090", sm_active=0.10, dram_active=0.10,
                 activity_source="nvml-utilization (coarse fallback)"),
            card(1, "GPU-3080a", "RTX 3080", sm_active=0.10, dram_active=0.10,
                 activity_source="nvml-utilization (coarse fallback)"),
        ]
        v = rank_shares(cards, peaks_from_hw_profile(HW_PROFILE),
                        rank_gpu_id=[0, 1], engine_rates=self._rates())
        self.assertAlmostEqual(v.ranks[0].active_share, 0.9)
        self.assertEqual(v.ranks[0].work_source, "engine forward-time counter")
        self.assertAlmostEqual(v.ranks[0].achieved_gbs, 500.0)
        # ...and the coarse caveat is gone, replaced by the estimation caveat.
        keys = {c.key for c in v.caveats}
        self.assertIn("engine_work_estimated", keys)
        self.assertNotIn("coarse_activity", keys)

    def test_achieved_fraction_against_the_probed_peak(self):
        v = rank_shares(
            [card(0, "GPU-5090", "RTX 5090")],
            peaks_from_hw_profile(HW_PROFILE),
            rank_gpu_id=[0],
            engine_rates=self._rates(),
        )
        # 500 GB/s of a measured 1664.1 GB/s peak
        self.assertAlmostEqual(v.ranks[0].membw_achieved_frac, 500.0 / 1664.1, places=5)

    def test_roofline_from_engine_rates(self):
        v = rank_shares(
            [card(0, "GPU-5090", "RTX 5090")],
            peaks_from_hw_profile(HW_PROFILE),
            rank_gpu_id=[0],
            engine_rates=self._rates(),
        )
        # 100 TFLOP/s over 500 GB/s = 200 flop/byte, balance ~140 -> compute
        self.assertAlmostEqual(v.ranks[0].intensity_flop_per_byte, 200.0, places=6)
        self.assertEqual(v.ranks[0].bound_by, "compute")

    def test_pacer_from_engine_rates(self):
        v = rank_shares(
            [card(0, "GPU-5090", "RTX 5090"), card(1, "GPU-3080a", "RTX 3080")],
            peaks_from_hw_profile(HW_PROFILE),
            rank_gpu_id=[0, 1],
            engine_rates=self._rates(),
        )
        self.assertEqual(v.pacer_rank, 0)
        self.assertNotIn("direction, not a measurement", v.pacer_basis)

    def test_single_rank_export_is_flagged(self):
        v = rank_shares([card(0, "GPU-5090", "RTX 5090")], {},
                        single_rank_export=True)
        keys = {c.key for c in v.caveats}
        self.assertIn("single_rank_export", keys)
        text = [c.text for c in v.caveats if c.key == "single_rank_export"][0]
        self.assertIn("--enable-metrics-for-all-schedulers", text)

    def test_no_previous_sample_yields_no_rates(self):
        self.assertEqual(engine_rank_rates({0: {"a": 1.0}}, None, 1.0), {})
        self.assertEqual(engine_rank_rates({0: {"a": 1.0}}, {0: {"a": 0.0}}, 0.0), {})

    def test_counter_reset_is_ignored_rather_than_reported_negative(self):
        r = engine_rank_rates(
            {0: {"forward_execution_seconds_total": 1.0}},
            {0: {"forward_execution_seconds_total": 100.0}},
            2.0,
        )
        self.assertNotIn(0, r)


if __name__ == "__main__":
    unittest.main()
