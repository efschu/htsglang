"""#1240 launcher half: the layer-set FLAG, the design depth, D's measured defaults.

Three things this file pins, all of them launcher-level and none of them
requiring a GPU or a launch:

  * ``SGLANG_PP_LAYER_SET`` stops being an INPUT. ``--pp-layer-set`` is, and an
    inherited environment value is refused by name (W44) rather than honoured,
    because ``build_env`` starts from ``os.environ`` and an exported map would
    otherwise reach group P's ranks without ever passing the solver -- the
    PP-CUT provenance line would then describe a layout the boot did not run.
  * The DESIGN PREFIX is read from a boot's own prefill census. No line states
    a prefix, so it is accumulated per request out of ``#new-token`` and
    ``#cached-token``, on ONE rank, and its absence is an absence of the
    instrument rather than a measured zero.
  * Group D's decode defaults are MEASURED (BOOT_weg2pp1_0907.md arms table
    row D2) and the override flags still work.
"""

import os
import tempfile
import unittest

import pytest

try:
    from sglang.srt.weg2.launcher import (
        ATTN_ANCHOR_MS,
        ATTN_ANCHOR_PREFIX_TOKENS,
        CALIBRATION_PREFIX_TOKENS,
        D_NUM_CONTINUOUS_DECODE_STEPS,
        DESIGN_PREFIX_FALLBACK_TOKENS,
        PP_LAYER_SET_ENV,
        BubbleMeasurement,
        Weg2LaunchRefused,
        argv_d,
        newest_prefill_census_log,
        pcie_lanes,
        per_pair_crossing_ms,
        read_mean_prefill_prefix,
        refuse_inherited_layer_set,
        solve_p_depth,
    )
    from sglang.test.ci.ci_register import register_cpu_ci
except RuntimeError as _import_err:  # pragma: no cover - leak-dependent
    pytest.skip(
        f"#249 default-device collection leak broke the import chain: {_import_err}",
        allow_module_level=True,
    )

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PY = "/nonexistent/python"
MODEL = "/nonexistent/model"

#: Two requests, chunk size 4096: one of ~8.5k tokens (chunks at prefix 0 and
#: 4096, then a short remainder at 8192) and one of ~4.1k. Written as the
#: runtime writes them, including the PP1/PP2 echo lines that must NOT be
#: counted a second time.
CENSUS = "\n".join(
    "[2026-09-08 00:00:0%d %s] Prefill batch, #new-seq: 1, #new-token: %d, "
    "#cached-token: %d, full token usage: 0.01, mamba usage: 0.20, "
    "#running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: False, "
    "input throughput (token/s): 100.0" % (i, rank, new, cached)
    for i, (new, cached) in enumerate(
        [(4096, 0), (4096, 0), (306, 0), (4096, 0), (1, 0)]
    )
    for rank in ("PP0", "PP1", "PP2")
)


class TheEnvIsNoLongerAnInput(unittest.TestCase):
    """W44: an inherited stage map is refused, never honoured silently."""

    def test_an_inherited_layer_set_is_refused_by_name(self):
        with self.assertRaises(Weg2LaunchRefused) as ctx:
            refuse_inherited_layer_set({PP_LAYER_SET_ENV: "0-2,4-6;3,7"})
        msg = str(ctx.exception)
        self.assertIn("W44 Weg2LayerSetEnvRefused", msg)
        # It names the offending value AND the flag that replaces it.
        self.assertIn("0-2,4-6;3,7", msg)
        self.assertIn("--pp-layer-set", msg)

    def test_an_absent_or_blank_env_is_the_normal_case(self):
        refuse_inherited_layer_set({})
        refuse_inherited_layer_set({PP_LAYER_SET_ENV: ""})
        refuse_inherited_layer_set({PP_LAYER_SET_ENV: "   "})

    def test_the_refusal_reads_the_environment_it_is_handed(self):
        # It is called on os.environ in main() BEFORE build_env copies it; the
        # test hands it a dict so the discipline is provable without exporting
        # anything into this process.
        self.assertIsNone(refuse_inherited_layer_set(dict(os.environ) | {}))


class TheDesignPrefixIsMeasured(unittest.TestCase):
    """Accumulated per request, one rank, and absent means absent."""

    def test_the_prefix_is_accumulated_and_resets_on_a_short_chunk(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "boot.P.log")
            with open(path, "w") as fh:
                fh.write(CENSUS)
            mean, n, rank = read_mean_prefill_prefix(path)
        # ONE rank only: five census lines, not fifteen.
        self.assertEqual(n, 5)
        self.assertEqual(rank, "PP0")
        # Prefixes: 0, 4096, 8192 (short chunk ends the request), then 0, 4096.
        self.assertAlmostEqual(mean, (0 + 4096 + 8192 + 0 + 4096) / 5.0, places=6)

    def test_the_cache_hit_counts_towards_the_prefix(self):
        line = CENSUS.split("\n")[0].replace("#cached-token: 0", "#cached-token: 512")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "boot.P.log")
            with open(path, "w") as fh:
                fh.write(line)
            mean, n, _ = read_mean_prefill_prefix(path)
        self.assertEqual(n, 1)
        self.assertAlmostEqual(mean, 512.0, places=6)

    def test_a_log_without_the_instrument_returns_none_not_zero(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "empty.P.log")
            with open(path, "w") as fh:
                fh.write("[2026-09-08 00:00:00 PP0] nothing to see here\n")
            self.assertIsNone(read_mean_prefill_prefix(path))
        self.assertIsNone(read_mean_prefill_prefix("/nonexistent/nope.P.log"))

    def test_the_newest_log_carrying_the_census_is_preferred_over_the_newest_log(self):
        with tempfile.TemporaryDirectory() as d:
            old = os.path.join(d, "old.P.log")
            new = os.path.join(d, "new.P.log")
            with open(old, "w") as fh:
                fh.write(CENSUS)
            with open(new, "w") as fh:
                fh.write("[2026-09-08 00:00:00 PP0] died before the first prefill\n")
            os.utime(old, (1, 1))
            os.utime(new, (2, 2))
            self.assertEqual(newest_prefill_census_log(d), old)

    def test_the_fallback_is_one_chunk_and_is_a_named_constant(self):
        self.assertEqual(DESIGN_PREFIX_FALLBACK_TOKENS, 4096)
        self.assertEqual(CALIBRATION_PREFIX_TOKENS, 4096)
        self.assertEqual(ATTN_ANCHOR_PREFIX_TOKENS, 262144)
        self.assertEqual(ATTN_ANCHOR_MS, 400.0)


class TheCrossingPricesComeFromTheMeasuredTable(unittest.TestCase):
    """Bottleneck lanes, and a lane count with no measurement is omitted."""

    def test_the_pair_price_uses_the_bottleneck_lane_count(self):
        prices = per_pair_crossing_ms([8, 4, 8], 41_943_040)
        # 0<->2 is x8 on both ends; anything touching ordinal 1 is x4.
        self.assertLess(prices[(0, 2)], prices[(0, 1)])
        self.assertAlmostEqual(prices[(0, 1)], prices[(1, 2)], places=9)
        self.assertAlmostEqual(prices[(0, 2)], prices[(2, 0)], places=9)

    def test_an_unmeasured_lane_count_is_omitted_and_never_interpolated(self):
        # The measured table has entries for 4 and 8 lanes only. A bottleneck
        # of 16 or 2 is DELIBERATELY absent rather than interpolated, and the
        # pair drops out entirely instead of acquiring a plausible number.
        self.assertEqual(per_pair_crossing_ms([16, 16, 1], 41_943_040), {})
        prices = per_pair_crossing_ms([8, 16, 2], 41_943_040)
        self.assertEqual(sorted(prices), [(0, 1), (1, 0)])

    def test_an_unknown_lane_width_does_not_crash_the_launcher(self):
        self.assertEqual(per_pair_crossing_ms([None, None, None], 1), {})

    def test_pcie_lanes_answers_one_entry_per_card_or_none(self):
        lanes = pcie_lanes([])
        self.assertEqual(lanes, [])


class TheDepthFollowsTheLayout(unittest.TestCase):
    """A gapped map admits ONE pass; that is derived, an override is refused."""

    def depth(self, **over):
        kwargs = dict(
            measured=None,
            pool_tokens=500_000.0,
            attn_counts=(0, 8, 8),
            kv_mib_per_token_per_attn_layer=2048.0 / (1024.0 * 1024.0),
            hidden_size=5120,
            chunk_tokens=4096,
            cap_tokens=262_144,
        )
        kwargs.update(over)
        return solve_p_depth(**kwargs)

    def test_a_gapped_layout_takes_the_depth_the_layout_admits(self):
        d = self.depth(gapped_layer_set="0-2,4-6;3;7")
        self.assertEqual(d.depth, 0)
        self.assertTrue(d.gapped_layout)
        self.assertIn("GAPPED layer set", d.line())
        self.assertIn("BY THE LAYOUT", d.line())

    def test_a_measured_bubble_asking_for_depth_is_taken_down_by_the_layout(self):
        # The bubble says two passes; the gapped layout admits one. That is a
        # DERIVATION -- the PP-CUT solver already priced this candidate at one
        # pass -- and the alternative shapes are both wrong: raising W43 would
        # refuse a layout the solver chose, and honouring the 1 would die in
        # init_pp_loop_state after the weights are loaded.
        measured = BubbleMeasurement(
            source="synthetic",
            rank=0,
            windows=1,
            forward_ms=600.0,
            bubble_ms=400.0,
            starved_ms=0.0,
            n_forwards=10,
        )
        contiguous = self.depth(measured=measured, gapped_layer_set="")
        self.assertGreater(contiguous.depth, 0)
        gapped = self.depth(measured=measured, gapped_layer_set="0-2,4-6;3;7")
        self.assertEqual(gapped.depth, 0)
        self.assertEqual(gapped.passes_in_flight, 1)
        self.assertTrue(gapped.gapped_layout)
        self.assertIn("BY THE LAYOUT", gapped.line())

    def test_a_pinned_depth_on_a_gapped_layout_is_refused_W43(self):
        with self.assertRaises(Weg2LaunchRefused) as ctx:
            self.depth(gapped_layer_set="0-2,4-6;3;7", pinned_depth=1)
        self.assertIn("W43 Weg2DepthGapped", str(ctx.exception))

    def test_a_contiguous_layout_is_unaffected(self):
        d = self.depth(gapped_layer_set="", pinned_depth=2)
        self.assertEqual(d.depth, 2)
        self.assertFalse(d.gapped_layout)
        self.assertNotIn("GAPPED layer set", d.line())


class GroupDDefaultsAreMeasured(unittest.TestCase):
    """Row D2 of BOOT_weg2pp1_0907.md, and the overrides still work."""

    def argv(self, **over):
        kwargs = dict(
            py=PY,
            model=MODEL,
            budgets=[1, 2, 3],
            s_gb=1,
            m_mib=1,
            store_gib=1.0,
            extra=[],
        )
        kwargs.update(over)
        return argv_d(**kwargs)

    def test_the_measured_default_is_two_steps(self):
        self.assertEqual(D_NUM_CONTINUOUS_DECODE_STEPS, 2)
        argv = self.argv(num_continuous_decode_steps=D_NUM_CONTINUOUS_DECODE_STEPS)
        self.assertEqual(argv[argv.index("--num-continuous-decode-steps") + 1], "2")

    def test_overlap_is_on_by_absence_and_pairs_with_extra_buffer(self):
        argv = self.argv(disable_overlap=False)
        self.assertNotIn("--disable-overlap-schedule", argv)
        self.assertEqual(
            argv[argv.index("--mamba-radix-cache-strategy") + 1], "extra_buffer"
        )

    def test_the_override_puts_the_shipped_pair_back(self):
        argv = self.argv(disable_overlap=True, num_continuous_decode_steps=1)
        self.assertIn("--disable-overlap-schedule", argv)
        self.assertEqual(
            argv[argv.index("--mamba-radix-cache-strategy") + 1], "no_buffer"
        )
        self.assertEqual(argv[argv.index("--num-continuous-decode-steps") + 1], "1")

    def test_the_strategy_follows_the_overlap_choice_and_is_not_a_second_knob(self):
        for disable in (True, False):
            argv = self.argv(disable_overlap=disable)
            want = "no_buffer" if disable else "extra_buffer"
            self.assertEqual(argv[argv.index("--mamba-radix-cache-strategy") + 1], want)


if __name__ == "__main__":
    unittest.main()
