"""#692 microbatch depth for group P: derived from the bubble, priced on the pool.

BOOT weg2pp1 (record 1r) solved group P's layer cut and still measured
``bubble_remains = TRUE`` -- 21.3 % of the binding stage's time on arm A1,
24.1 % on its A/A repeat. The cut moved the makespan; it cannot move the gap
BETWEEN forwards, because that gap is the output/proxy exchange serialising
with the launch. ``pp_async_batch_depth`` is the knob that overlaps them
(``scheduler_pp_mixin.init_pp_loop_state``: it both widens ``pp_loop_size``
and moves ``_pp_commit_send_output_work_and_preprocess_output_tensors`` from
after the launch to before it).

These are launcher tests: they read what the launcher would launch and what it
would print, without launching anything and without touching a GPU.
"""

import math
import os
import tempfile
import unittest
from unittest import mock

import pytest

try:
    from sglang.srt.weg2 import launcher as launcher_mod
    from sglang.srt.weg2.launcher import (
        BubbleMeasurement,
        DepthDecision,
        Weg2LaunchRefused,
        argv_d,
        argv_p,
        chunked_prefill_size_of,
        newest_bubble_log,
        read_pp_bubble,
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
BUDGETS = [27960, 17064, 16552]

#: Three real window lines from boot weg2pp1 arm A1
#: (/root/.claude/jobs/1481bb40/tmp/perf/P_A1.log, 2026-09-07 23:01:37Z), the
#: measurement record 1r reports as "21.3 % on the binding stage". PP0 carries
#: the largest forward total, so PP0 is the binding stage.
REAL_LINES = """\
[2026-09-07 23:01:37 PP0] PP-BUBBLE rank=0 share=22.0% of gap+forward, mean=98.4 ms, max=273.8 ms, n=12 (n_gaps=12, forward_ms=4191.7, bubble_ms=1181.1, starved_ms=0.0; bubble = host time BETWEEN forwards, a different term from the 'wait' inside one; starved = the part of it in which this rank visited the loop with nothing to launch, so bubble_ms - starved_ms is the pipeline stall)
[2026-09-07 23:01:37 PP1] PP-BUBBLE rank=1 share=27.0% of gap+forward, mean=122.7 ms, max=353.8 ms, n=12 (n_gaps=12, forward_ms=3979.3, bubble_ms=1472.3, starved_ms=0.0; bubble = host time BETWEEN forwards, a different term from the 'wait' inside one; starved = the part of it in which this rank visited the loop with nothing to launch, so bubble_ms - starved_ms is the pipeline stall)
[2026-09-07 23:01:38 PP2] PP-BUBBLE rank=2 share=36.1% of gap+forward, mean=163.5 ms, max=300.8 ms, n=12 (n_gaps=12, forward_ms=3474.9, bubble_ms=1962.1, starved_ms=0.0; bubble = host time BETWEEN forwards, a different term from the 'wait' inside one; starved = the part of it in which this rank visited the loop with nothing to launch, so bubble_ms - starved_ms is the pipeline stall)
"""

#: The pool facts of that same boot: P pool 373,785 tokens at the solved cut
#: 42,11,11 attn 10,3,3, floor --max-kv-per-request 262,144, KV cell
#: 2048 B/token/attention-layer (fp8_e4m3, consumed from config), hidden 5120.
POOL_TOKENS = 373785.0
ATTN = (10, 3, 3)
KV_MIB = 2048.0 / (1024.0 * 1024.0)
HIDDEN = 5120
CHUNK = 4096
CAP = 262144


def write_log(text):
    fh = tempfile.NamedTemporaryFile("w", suffix=".P.log", delete=False)
    fh.write(text)
    fh.close()
    return fh.name


def depth(measured=None, pool_tokens=POOL_TOKENS, cap=CAP, attn=ATTN, **kw):
    return solve_p_depth(
        measured=measured,
        pool_tokens=pool_tokens,
        attn_counts=attn,
        kv_mib_per_token_per_attn_layer=KV_MIB,
        hidden_size=HIDDEN,
        chunk_tokens=CHUNK,
        cap_tokens=cap,
        **kw,
    )


class TestReadingTheMeasurement(unittest.TestCase):
    """The share is READ from the previous boot's own line, never a literal."""

    def test_parses_every_rank_and_names_the_binding_stage(self):
        m = read_pp_bubble(write_log(REAL_LINES))
        self.assertIsInstance(m, BubbleMeasurement)
        # PP0 has the largest forward total (4191.7 > 3979.3 > 3474.9), so it
        # is the stage whose idle costs throughput.
        self.assertEqual(m.rank, 0)
        self.assertAlmostEqual(m.forward_ms, 4191.7, places=1)
        self.assertAlmostEqual(m.bubble_ms, 1181.1, places=1)
        self.assertEqual(m.windows, 1)

    def test_share_reproduces_the_line_it_was_read_from(self):
        m = read_pp_bubble(write_log(REAL_LINES))
        self.assertAlmostEqual(m.bubble_share * 100.0, 22.0, places=1)
        self.assertAlmostEqual(m.forward_share * 100.0, 78.0, places=1)

    def test_windows_of_one_rank_are_summed_not_averaged(self):
        """Sum of numerators over sum of denominators -- the denominator law."""
        doubled = REAL_LINES + REAL_LINES
        m = read_pp_bubble(write_log(doubled))
        self.assertEqual(m.windows, 2)
        self.assertAlmostEqual(m.forward_ms, 2 * 4191.7, places=1)
        self.assertAlmostEqual(m.bubble_share * 100.0, 22.0, places=1)

    def test_a_log_without_the_line_is_absence_not_zero(self):
        self.assertIsNone(read_pp_bubble(write_log("nothing to see\n")))

    def test_a_missing_file_is_absence(self):
        self.assertIsNone(read_pp_bubble("/nonexistent/never.P.log"))

    def test_newest_bubble_log_skips_logs_without_the_line(self):
        d = tempfile.mkdtemp()
        empty = os.path.join(d, "boot_weg2_a_0000000000_0101_000000.P.log")
        with open(empty, "w") as fh:
            fh.write("no bubble here\n")
        os.utime(empty, (2_000_000_000, 2_000_000_000))
        real = os.path.join(d, "boot_weg2_b_0000000000_0101_000001.P.log")
        with open(real, "w") as fh:
            fh.write(REAL_LINES)
        os.utime(real, (1_000_000_000, 1_000_000_000))
        # `empty` is NEWER but carries no measurement: absence of the line is
        # not a measurement of zero bubble.
        self.assertEqual(newest_bubble_log(d), real)

    def test_no_candidate_at_all_is_none(self):
        self.assertIsNone(newest_bubble_log(tempfile.mkdtemp()))


class TestTheDerivation(unittest.TestCase):
    """depth follows the bubble; every term in the line comes from a measurement."""

    def test_measured_bubble_yields_one_extra_pass(self):
        m = read_pp_bubble(write_log(REAL_LINES))
        d = depth(measured=m)
        # 1/(1-0.220) = 1.282 -> 2 passes must be in flight to cover the gap;
        # one of them is the pass being forwarded, so the FLAG (extra passes)
        # is 1. What SHIPS is 0: BOOT_weg2pp2_0907.md measured that depth on
        # this workload and it lost (+6.9 % TTFT p50, bubble unmoved), so the
        # derivation is retracted until a measured exchange-bound share exists
        # to re-ground it. The arithmetic stays visible in derived_depth.
        self.assertEqual(d.derived_depth, 1)
        self.assertEqual(d.depth, 0)
        self.assertTrue(d.retracted)

    def test_the_formula_is_the_reciprocal_of_the_forward_share(self):
        for share in (0.10, 0.213, 0.35, 0.55, 0.70):
            m = BubbleMeasurement(
                source="<synthetic>",
                rank=0,
                windows=1,
                forward_ms=(1.0 - share) * 1000.0,
                bubble_ms=share * 1000.0,
                starved_ms=0.0,
                n_forwards=1,
            )
            d = depth(measured=m, pool_tokens=1.0e9)
            self.assertEqual(
                d.derived_depth + 1,
                math.ceil(1.0 / (1.0 - share)),
                f"share={share}",
            )
            # And a PIN of the derived number ships it, unpriced by nothing:
            # the retraction is about what the launcher chooses, never about
            # what the arithmetic says.
            pinned = depth(measured=m, pool_tokens=1.0e9, pinned_depth=d.derived_depth)
            self.assertEqual(pinned.depth, d.derived_depth)
            self.assertEqual(pinned.passes_in_flight, d.derived_depth + 1)

    def test_the_retraction_is_not_a_silent_zero(self):
        """The line has to SAY it, or a 0 reads as 'no bubble was measured'."""
        m = read_pp_bubble(write_log(REAL_LINES))
        line = depth(measured=m).line()
        self.assertIn("RETRACTED (derived 1, shipped 0)", line)
        self.assertIn("BOOT_weg2pp2_0907.md", line)
        self.assertIn("bubble_share=", line)

    def test_a_bubble_that_is_all_starvation_buys_no_depth(self):
        """Depth overlaps stages; it cannot manufacture work that is not queued."""
        m = BubbleMeasurement(
            source="<synthetic>",
            rank=0,
            windows=1,
            forward_ms=780.0,
            bubble_ms=220.0,
            starved_ms=220.0,
            n_forwards=1,
        )
        d = depth(measured=m)
        self.assertEqual(d.derived_depth, 0)
        self.assertEqual(d.depth, 0)
        self.assertFalse(d.retracted)
        self.assertIn("starved", d.line())

    def test_a_boot_with_no_bubble_buys_no_depth(self):
        m = BubbleMeasurement(
            source="<synthetic>",
            rank=0,
            windows=1,
            forward_ms=1000.0,
            bubble_ms=0.0,
            starved_ms=0.0,
            n_forwards=1,
        )
        self.assertEqual(depth(measured=m).derived_depth, 0)

    def test_absent_measurement_is_todays_behaviour_and_says_so(self):
        d = depth(measured=None)
        self.assertIsInstance(d, DepthDecision)
        self.assertEqual(d.depth, 0)
        line = d.line()
        self.assertIn("WEG2 P-DEPTH solver:", line)
        self.assertIn("depth=0", line)
        self.assertIn("no PP-BUBBLE measurement", line)

    def test_the_line_carries_every_term_the_briefing_names(self):
        m = read_pp_bubble(write_log(REAL_LINES))
        line = depth(measured=m).line()
        for term in (
            "WEG2 P-DEPTH solver:",
            "bubble_share=",
            "forward_share=",
            "depth=",
            "price_rows=",
            "price_mib=",
            "pool_after=",
            "constraint pool >=",
        ):
            self.assertIn(term, line, f"missing {term!r} in {line!r}")

    def test_the_line_names_the_file_the_share_came_from(self):
        path = write_log(REAL_LINES)
        line = depth(measured=read_pp_bubble(path)).line()
        self.assertIn(path, line)


class TestThePrice(unittest.TestCase):
    """One currency: pool tokens. No second budget beside the pool model."""

    def test_kv_rows_are_charged_per_stage_per_extra_pass(self):
        # PINNED, because since FOLLOW FIX 1 that is the only path that ships
        # a depth > 0 -- and the price is what a shipped depth costs.
        m = read_pp_bubble(write_log(REAL_LINES))
        d = depth(measured=m, pinned_depth=1)
        self.assertEqual(d.price_rows, d.depth * CHUNK)

    def test_the_crossing_frame_is_charged_too(self):
        """The activation frame is NOT in the pool model, so it must be added."""
        m = read_pp_bubble(write_log(REAL_LINES))
        d = depth(measured=m, pinned_depth=1)
        act_mib = CHUNK * HIDDEN * 2 / (1024.0 * 1024.0)
        self.assertAlmostEqual(d.act_mib_per_pass, act_mib, places=3)
        # 40 MiB at chunk 4096 -- the same order as the measured 40 MiB per
        # stage crossing quoted in the user's physics note.
        self.assertGreater(d.act_mib_per_pass, 35.0)
        self.assertLess(d.act_mib_per_pass, 45.0)
        self.assertGreater(d.price_tokens, d.price_rows)

    def test_the_binding_stage_of_the_price_is_the_one_with_least_attention(self):
        """A stage with fewer attention layers pays more TOKENS per MiB."""
        m = read_pp_bubble(write_log(REAL_LINES))
        wide = depth(measured=m, attn=(10, 10, 10), pinned_depth=1)
        narrow = depth(measured=m, attn=(10, 3, 3), pinned_depth=1)
        self.assertGreater(narrow.price_tokens, wide.price_tokens)

    def test_pool_after_is_the_pool_minus_the_price(self):
        m = read_pp_bubble(write_log(REAL_LINES))
        d = depth(measured=m, pinned_depth=1)
        self.assertAlmostEqual(d.pool_after, POOL_TOKENS - d.price_tokens, places=3)

    def test_the_measured_boot_can_fund_one_extra_pass(self):
        """373,785 - one pass is still above the 262,144 floor."""
        m = read_pp_bubble(write_log(REAL_LINES))
        d = depth(measured=m, pinned_depth=1)
        self.assertEqual(d.depth, 1)
        self.assertGreater(d.pool_after, float(CAP))


class TestTheRefusals(unittest.TestCase):
    def test_W42_when_the_pool_cannot_fund_the_depth(self):
        m = read_pp_bubble(write_log(REAL_LINES))
        with self.assertRaises(Weg2LaunchRefused) as ctx:
            depth(measured=m, pool_tokens=float(CAP) + 10.0, pinned_depth=1)
        msg = str(ctx.exception)
        self.assertIn("W42 Weg2DepthUnfunded", msg)
        # Fail fast NAMING the numbers, per the launcher's own refusal style.
        self.assertIn("262144", msg.replace(",", ""))
        self.assertIn("depth", msg)

    def test_W42_never_silently_lowers_the_depth(self):
        """A refusal, not a cap: a quietly reduced depth is a hand number."""
        m = read_pp_bubble(write_log(REAL_LINES))
        with self.assertRaises(Weg2LaunchRefused):
            depth(measured=m, pool_tokens=float(CAP) + 10.0, pinned_depth=1)

    def test_W42_on_a_PIN_names_the_pin_and_not_the_measurement(self):
        """FOLLOW FIX 2 / finding 2: the only reachable W42 today is a PIN.

        With the #692 derivation retracted a DERIVED depth is always 0, so this
        refusal can only fire for an override. Attributing the pin to "the
        measured bubble", printing that measurement's stall_share beside it,
        and closing with "a depth the bubble did not ask for is a hand number"
        was three sentences describing a mechanism that did not act.
        """
        m = read_pp_bubble(write_log(REAL_LINES))
        with self.assertRaises(Weg2LaunchRefused) as ctx:
            depth(measured=m, pool_tokens=float(CAP) + 10.0, pinned_depth=3)
        msg = str(ctx.exception)
        self.assertIn("--p-microbatch-depth 3 was PINNED", msg)
        self.assertNotIn("the measured bubble asks for", msg)
        self.assertNotIn("a depth the bubble did not ask for", msg)
        # The derivation is still named -- it is what the pin overrode.
        self.assertIn("the measurement derived", msg)

    def test_W42_on_a_DERIVED_depth_is_reachable_and_says_the_bubble_asked(self):
        """The other branch is not dead text: it is live the day #692 re-grounds.

        Proved by flipping the one constant that gates the retraction, rather
        than by asserting a string nothing can print.
        """
        m = read_pp_bubble(write_log(REAL_LINES))
        with mock.patch.object(launcher_mod, "DEPTH_DERIVATION_GROUNDED", True):
            with self.assertRaises(Weg2LaunchRefused) as ctx:
                depth(measured=m, pool_tokens=float(CAP) + 10.0)
        msg = str(ctx.exception)
        self.assertIn("the measured bubble asks for", msg)
        self.assertIn("a depth the bubble did not ask for", msg)
        self.assertNotIn("was PINNED", msg)

    def test_W43_when_a_PINNED_depth_meets_a_gapped_layer_set(self):
        """init_pp_loop_state raises on gapped + depth>0; say so BEFORE the boot.

        REVISED BY #1240. When #692 was built, a gapped layer set could only
        arrive by an operator's export, so any depth against it was a
        contradiction and W43 was the whole answer. Since #1240 the SOLVER
        chooses the map, and it prices a gapped candidate at exactly ONE pass
        in flight -- so a DERIVED depth against a gapped map is not a
        contradiction, it is the layout's own bound and is taken (the test
        below). W43 keeps the half that is still a contradiction: an OVERRIDE
        asking for a depth the boot cannot run.
        """
        m = read_pp_bubble(write_log(REAL_LINES))
        with self.assertRaises(Weg2LaunchRefused) as ctx:
            depth(measured=m, gapped_layer_set="0,1,2:48", pinned_depth=1)
        msg = str(ctx.exception)
        self.assertIn("W43 Weg2DepthGapped", msg)
        self.assertIn("init_pp_loop_state", msg)

    def test_a_derived_depth_against_a_gapped_map_is_taken_down_by_the_layout(self):
        m = read_pp_bubble(write_log(REAL_LINES))
        self.assertGreater(depth(measured=m).derived_depth, 0)
        d = depth(measured=m, gapped_layer_set="0,1,2:48")
        self.assertEqual(d.depth, 0)
        self.assertTrue(d.gapped_layout)
        self.assertIn("BY THE LAYOUT", d.line())

    def test_a_gapped_set_with_depth_zero_is_not_refused(self):
        self.assertEqual(
            depth(measured=None, gapped_layer_set="0,1,2:48").depth, 0
        )


class TestFix3TheLineAndTheRefusalPriceWhatShips(unittest.TestCase):
    """FOLLOW FIX 3 (review findings 1+2 on 558ad4bb6b).

    Finding 1: the bracket printed the formula label over the SHIPPED pass
    count, so a retracted depth read ``ceil(1/(1-stall_share))=1`` while the
    formula gave 2. Finding 2: the W42 sentence priced "one extra pass" with
    the TOTAL for every pass. Both are instrument-text-lies of class A.
    """

    def test_the_line_names_shipped_and_derived_passes_separately(self):
        m = read_pp_bubble(write_log(REAL_LINES))
        d = depth(measured=m)
        self.assertTrue(d.retracted)
        line = d.line()
        self.assertIn("passes_in_flight (shipped)=1 = depth+1", line)
        self.assertIn(
            "derived passes=ceil(1/(1-stall_share))=%d (NOT shipped: retraction)"
            % (d.derived_depth + 1),
            line,
        )
        # The old form must be gone: a formula label over the shipped value.
        self.assertNotIn("passes_in_flight=ceil", line)

    def test_a_pin_that_ships_the_derived_depth_prints_no_not_shipped(self):
        m = read_pp_bubble(write_log(REAL_LINES))
        d = depth(measured=m, pinned_depth=1)
        self.assertEqual(d.depth, d.derived_depth)
        self.assertNotIn("NOT shipped", d.line())
        self.assertIn("passes_in_flight (shipped)=2 = depth+1", d.line())

    def test_a_pin_above_the_derivation_says_the_pin_overrides(self):
        m = read_pp_bubble(write_log(REAL_LINES))
        d = depth(measured=m, pool_tokens=1.0e9, pinned_depth=3)
        self.assertIn("(NOT shipped: pin overrides)", d.line())
        self.assertIn("passes_in_flight (shipped)=4 = depth+1", d.line())

    def test_no_measurement_prints_no_formula(self):
        line = depth(measured=None).line()
        self.assertIn("derived passes=n/a (no measurement)", line)
        self.assertNotIn("ceil(", line)

    def test_the_price_is_per_pass_times_depth(self):
        m = read_pp_bubble(write_log(REAL_LINES))
        one = depth(measured=m, pool_tokens=1.0e9, pinned_depth=1)
        two = depth(measured=m, pool_tokens=1.0e9, pinned_depth=2)
        self.assertAlmostEqual(one.price_tokens, one.price_tokens_per_pass, places=3)
        self.assertAlmostEqual(two.price_tokens, 2 * two.price_tokens_per_pass, places=3)
        self.assertAlmostEqual(one.price_tokens_per_pass, two.price_tokens_per_pass, places=3)
        self.assertEqual(depth(measured=m).price_tokens, 0.0)

    def test_the_line_prints_per_pass_and_total(self):
        m = read_pp_bubble(write_log(REAL_LINES))
        d = depth(measured=m, pool_tokens=1.0e9, pinned_depth=2)
        line = d.line()
        self.assertIn(
            "= %d pool tokens at the stage that converts worst, %d in total for depth 2"
            % (int(d.price_tokens_per_pass), int(d.price_tokens)),
            line,
        )

    def test_W42_prices_each_pass_and_the_total_separately(self):
        m = read_pp_bubble(write_log(REAL_LINES))
        per_pass = depth(measured=m, pool_tokens=1.0e9, pinned_depth=1).price_tokens
        with self.assertRaises(Weg2LaunchRefused) as ctx:
            depth(measured=m, pool_tokens=float(CAP) + per_pass + 10.0, pinned_depth=2)
        msg = str(ctx.exception)
        self.assertIn("EACH extra pass", msg)
        self.assertIn("= %d pool tokens at the worst-converting stage" % int(per_pass), msg)
        self.assertIn("depth 2 costs %d in total" % int(2 * per_pass), msg)
        self.assertNotIn("one extra pass", msg)


class TestThePin(unittest.TestCase):
    """An override replaces the DERIVATION, never the price."""

    def test_a_pin_is_announced_in_the_line(self):
        d = depth(measured=None, pinned_depth=1)
        self.assertEqual(d.depth, 1)
        self.assertTrue(d.pinned)
        self.assertIn("PINNED (user override)", d.line())

    def test_a_pin_is_still_priced_and_still_refusable(self):
        with self.assertRaises(Weg2LaunchRefused) as ctx:
            depth(measured=None, pinned_depth=40, pool_tokens=float(CAP) + 10.0)
        self.assertIn("W42 Weg2DepthUnfunded", str(ctx.exception))

    def test_a_pin_of_zero_is_a_pin_not_an_absence(self):
        d = depth(measured=read_pp_bubble(write_log(REAL_LINES)), pinned_depth=0)
        self.assertEqual(d.depth, 0)
        self.assertTrue(d.pinned)

    def test_a_negative_pin_is_refused(self):
        with self.assertRaises(Weg2LaunchRefused):
            depth(measured=None, pinned_depth=-1)

    def test_a_pin_still_collides_with_a_gapped_set(self):
        with self.assertRaises(Weg2LaunchRefused) as ctx:
            depth(measured=None, pinned_depth=1, gapped_layer_set="0,1,2:48")
        self.assertIn("W43 Weg2DepthGapped", str(ctx.exception))

    def test_an_unpinned_decision_says_so(self):
        self.assertNotIn("PINNED", depth(measured=None).line())


class TestTheFlagReachesTheGroup(unittest.TestCase):
    def test_group_P_states_the_depth_on_its_argv(self):
        argv = argv_p(PY, MODEL, BUDGETS, 1, 2400, 8.0, [], depth=1)
        self.assertIn("--pp-async-batch-depth", argv)
        self.assertEqual(argv[argv.index("--pp-async-batch-depth") + 1], "1")

    def test_depth_zero_is_still_stated_rather_than_omitted(self):
        """The argv is an honest statement of what the boot runs."""
        argv = argv_p(PY, MODEL, BUDGETS, 1, 2400, 8.0, [], depth=0)
        self.assertIn("--pp-async-batch-depth", argv)
        self.assertEqual(argv[argv.index("--pp-async-batch-depth") + 1], "0")

    def test_group_D_never_carries_it(self):
        """D runs pp_size=1: a pipeline depth is meaningless there."""
        self.assertNotIn(
            "--pp-async-batch-depth", argv_d(PY, MODEL, BUDGETS, 1, 2400, 8.0, [])
        )

    def test_the_depth_is_a_group_constant_not_a_per_rank_value(self):
        """Rank uniformity: one argv token, published by the launcher."""
        argv = argv_p(PY, MODEL, BUDGETS, 1, 2400, 8.0, [], depth=2)
        self.assertEqual(argv.count("--pp-async-batch-depth"), 1)
        self.assertNotIn(",", argv[argv.index("--pp-async-batch-depth") + 1])

    def test_the_chunk_size_is_read_off_the_argv_not_restated(self):
        """A second copy of --chunked-prefill-size would drift the day it moves."""
        argv = argv_p(PY, MODEL, BUDGETS, 1, 2400, 8.0, [], depth=0)
        self.assertEqual(chunked_prefill_size_of(argv), CHUNK)

    def test_a_missing_chunk_flag_refuses_rather_than_defaults(self):
        with self.assertRaises(Weg2LaunchRefused):
            chunked_prefill_size_of(["python", "-m", "sglang.launch_server"])


class TestTheReadersSeeIt(unittest.TestCase):
    """The knob has ONE reader chain: server_args -> scheduler_pp_mixin."""

    def test_server_args_exposes_the_cli_flag(self):
        import argparse

        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        dests = {a.dest for a in parser._actions}
        self.assertIn("pp_async_batch_depth", dests)

    def test_the_loop_size_is_pp_size_plus_depth(self):
        """init_pp_loop_state: `pp_loop_size = ps.pp_size + pp_async_batch_depth`."""
        import inspect

        from sglang.srt.managers import scheduler_pp_mixin

        src = inspect.getsource(scheduler_pp_mixin.SchedulerPPMixin.init_pp_loop_state)
        self.assertIn(
            "self.pp_loop_size: int = self.ps.pp_size "
            "+ self.server_args.pp_async_batch_depth",
            " ".join(src.split()),
        )

    def test_the_runtime_still_refuses_gapped_plus_depth(self):
        """W43 mirrors a live refusal; if that one goes, W43 is a lie."""
        import inspect

        from sglang.srt.managers import scheduler_pp_mixin

        src = " ".join(
            inspect.getsource(
                scheduler_pp_mixin.SchedulerPPMixin.init_pp_loop_state
            ).split()
        )
        self.assertIn("if self.server_args.pp_async_batch_depth > 0:", src)
        self.assertIn("a gapped PP layer set cannot be combined with", src)


if __name__ == "__main__":
    unittest.main()
