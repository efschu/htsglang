"""Group P and group D no longer share flags whose reason belongs to one.

BSSCALE_0907.md D4: ``--disable-overlap-schedule`` sat in ``common_flags``, so
group D inherited it although its only justification (#1030, server_args.py
:19507 "Pipeline parallelism is not compatible with overlap schedule") is a
``pp_size > 1`` reason and D runs ``pp_size=1``. D3: ``num_continuous_decode_
steps`` sat at its minimum with no way to move it per group. P5: the
``write_through`` store tax was never A/B'd because there was no arm.

These are argv tests: they read what the launcher would launch, without
launching anything.
"""

import unittest

import pytest

try:
    from sglang.srt.weg2.launcher import (
        argv_d,
        argv_p,
        common_flags,
        d_mamba_ping_pong_cost,
        d_overlap_cost_line,
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


def p(**kw):
    return argv_p(PY, MODEL, BUDGETS, 1, 2400, 8.0, [], **kw)


def d(**kw):
    return argv_d(PY, MODEL, BUDGETS, 1, 2400, 8.0, [], **kw)


class TestOverlapScheduleIsGroupPsFlag(unittest.TestCase):
    def test_common_flags_no_longer_carry_it(self):
        # max_kv_per_request IS REQUIRED and was missing here, so this
        # assertion had been raising TypeError instead of checking anything.
        # Same class as the _max_running_requests boot killer train fix 2
        # closed: merge 3's union signature inserted a required parameter and
        # a positional caller was left behind.
        self.assertNotIn(
            "--disable-overlap-schedule",
            common_flags(MODEL, 1, 2400, 8.0, 262144),
        )

    def test_group_P_carries_it(self):
        self.assertIn("--disable-overlap-schedule", p())

    def test_group_D_does_not(self):
        self.assertNotIn("--disable-overlap-schedule", d())

    def test_the_escape_hatch_puts_it_back_on_D(self):
        """W41: a gate that refuses is honoured, never weakened."""
        self.assertIn("--disable-overlap-schedule", d(disable_overlap=True))


class TestContinuousDecodeSteps(unittest.TestCase):
    def test_D_states_the_value_and_defaults_to_the_shipped_one(self):
        argv = d()
        self.assertIn("--num-continuous-decode-steps", argv)
        self.assertEqual(argv[argv.index("--num-continuous-decode-steps") + 1], "1")

    def test_the_value_reaches_the_argv(self):
        argv = d(num_continuous_decode_steps=4)
        self.assertEqual(argv[argv.index("--num-continuous-decode-steps") + 1], "4")

    def test_P_does_not_get_it(self):
        self.assertNotIn("--num-continuous-decode-steps", p())


class TestWritePolicyMeasurementArm(unittest.TestCase):
    def test_default_is_the_shipped_write_through_on_both_groups(self):
        for argv in (p(), d()):
            i = argv.index("--hicache-write-policy")
            self.assertEqual(argv[i + 1], "write_through")

    def test_the_arm_moves_P_only(self):
        argv_pp = p(write_policy="write_back")
        self.assertEqual(
            argv_pp[argv_pp.index("--hicache-write-policy") + 1], "write_back"
        )
        argv_dd = d()
        self.assertEqual(
            argv_dd[argv_dd.index("--hicache-write-policy") + 1], "write_through"
        )


class TestTheStageRatioIsNoLongerAConstantInTheArgv(unittest.TestCase):
    def test_the_solved_cut_reaches_the_argv(self):
        argv = p(stage_ratio="42,11,11", attn_stage_ratio="10,3,3")
        self.assertEqual(argv[argv.index("--pp-stage-ratio") + 1], "42,11,11")
        self.assertEqual(argv[argv.index("--pp-attn-stage-ratio") + 1], "10,3,3")



class TestTheOverlapChoiceIsPricedAndStated(unittest.TestCase):
    """FIX 1r/1: turning the overlap schedule ON for D also flips its mamba
    radix-cache strategy to ``extra_buffer`` and DOUBLES D's device mamba
    floor -- 16 -> 32 state slots at ``--max-running-requests 8`` -- out of
    D's FIXED ``--rank-gpu-memory-mib`` budgets. MEASURED with the runtime's
    own functions against the real argv (``mamba_pool_floor.
    mamba_ping_pong_slots`` returns 2 instead of 0). The shipped commit
    announced the resolution and priced only the overlap benefit: an unpriced
    term does not read as unknown, it reads as free (#1009).
    """

    def test_D_states_its_strategy_instead_of_inheriting_it(self):
        argv = d()
        self.assertIn("--mamba-radix-cache-strategy", argv)
        self.assertEqual(
            argv[argv.index("--mamba-radix-cache-strategy") + 1], "extra_buffer"
        )

    def test_the_escape_hatch_states_the_other_one(self):
        argv = d(disable_overlap=True)
        self.assertEqual(
            argv[argv.index("--mamba-radix-cache-strategy") + 1], "no_buffer"
        )

    def test_the_slot_delta_is_derived_from_the_runtime_not_typed(self):
        strategy, per_req, extra, mrr = d_mamba_ping_pong_cost(MODEL, False)
        self.assertEqual(strategy, "extra_buffer")
        self.assertEqual(per_req, 2)
        self.assertEqual(mrr, 8)
        self.assertEqual(extra, 16)
        strategy, per_req, extra, mrr = d_mamba_ping_pong_cost(MODEL, True)
        self.assertEqual(strategy, "no_buffer")
        self.assertEqual(per_req, 0)
        self.assertEqual(extra, 0)

    def test_the_number_appears_in_the_line_that_announces_the_change(self):
        line = d_overlap_cost_line(MODEL, False)
        self.assertIn("16 extra device mamba state slots", line)
        self.assertIn("2 extra ping-pong state slots per running request", line)
        self.assertIn("extra_buffer", line)
        # The MiB conversion is REFUSED here, and the refusal is named rather
        # than left as a silent omission.
        self.assertIn("not converted to MiB", line)
        self.assertIn("mamba_cache_per_req", line)

    def test_the_hatch_line_charges_nothing(self):
        line = d_overlap_cost_line(MODEL, True)
        self.assertIn("no_buffer", line)
        self.assertIn("0 extra device mamba state slots", line)


if __name__ == "__main__":
    unittest.main()
