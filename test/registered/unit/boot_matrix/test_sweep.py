"""The sweep's card-less surfaces (#349).

build_command is a pure function -- the base recipe plus the arm's delta -- and
render_plan is a pure table. Both are exercised here so the plan an operator
reads before spending an hour of card time is provably the plan that runs.
"""

import unittest

from sglang.srt.boot_matrix.arms import BASE_FLAGS, arm_by_name
from sglang.srt.boot_matrix.sweep import build_command, render_plan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestBuildCommand(CustomTestCase):
    def test_base_recipe_is_present(self):
        arm = arm_by_name("A_default")
        env, argv = build_command(arm, model_path="/m", port=30000)
        line = " ".join(argv)
        self.assertIn("sglang.launch_server", line)
        self.assertIn("--model-path /m", line)
        self.assertIn("--tp-size 3", line)
        self.assertEqual(env["SGLANG_UNEVEN_DCP"], "1")

    def test_arm_flags_are_added_on_top(self):
        arm = arm_by_name("B_offload")
        _, argv = build_command(arm, model_path="/m", port=30000)
        self.assertIn("--enable-kv-session-offload", argv)
        # base flags still there
        self.assertIn("--speculative-algorithm", argv)

    def test_arm_h_removes_the_spec_flags_instead_of_renaming_them(self):
        """Arm H means "no speculation", and that is a REMOVAL.

        It used to append ``--speculative-algorithm none`` and rely on argparse
        taking the last value. That does not disable anything:
        ``speculative_algorithm`` is a free-form string, so "none" is just an
        algorithm nobody registered, truthy at every call site. Sweep 1 watched
        the arm die on a guard whose message read "does not yet support
        speculative decoding (--speculative-algorithm=none)".
        """
        arm = arm_by_name("H_ps2_prefill_spill")
        _, argv = build_command(arm, model_path="/m", port=30000)
        self.assertNotIn("--speculative-algorithm", argv)
        self.assertNotIn("none", argv)
        for gone in (
            "--speculative-num-steps",
            "--speculative-eagle-topk",
            "--speculative-num-draft-tokens",
        ):
            self.assertNotIn(gone, argv, f"{gone} must go with the algorithm")
        # The rest of the base recipe is untouched -- a drop must not be a rewrite.
        self.assertIn("--tp-size", argv)
        self.assertIn("--kv-cache-dtype", argv)

    def test_a_later_flag_still_overrides_a_base_value(self):
        """Removal is the new mechanism; plain override must keep working.

        I_dflash_shards appends --speculative-algorithm DFLASH over the base
        NEXTN and relies on argparse taking the last value. (This used to be
        demonstrated on reject_dcp_offlane's ratio, which now DROPS the base
        flag instead -- an explicit all-identical ratio is rejected by name.)
        """
        arm = arm_by_name("I_dflash_shards")
        _, argv = build_command(arm, model_path="/m", port=30000)
        i_first = argv.index("--speculative-algorithm")
        i_last = len(argv) - 1 - argv[::-1].index("--speculative-algorithm")
        self.assertGreater(i_last, i_first, "the arm's value must come last")
        self.assertEqual(argv[i_last + 1], "DFLASH")

    def test_dropping_a_flag_takes_its_value_with_it(self):
        """An orphan value would be read as a positional argument."""
        arm = arm_by_name("reject_dcp_offlane")
        _, argv = build_command(arm, model_path="/m", port=30000)
        self.assertNotIn("--rank-auto-reserve-mib", argv)
        self.assertNotIn("3000,2700,2700", argv)

    def test_a_drop_that_matches_nothing_is_an_error(self):
        """A silent no-op here is how an arm runs something it did not declare."""
        from sglang.srt.boot_matrix.sweep import _without

        with self.assertRaises(ValueError) as caught:
            _without(("--tp-size", "3"), ("--not-a-base-flag",))
        self.assertIn("--not-a-base-flag", str(caught.exception))

    def test_reject_arm_env_override(self):
        arm = arm_by_name("reject_dcp_offlane")
        env, _ = build_command(arm, model_path="/m", port=30000)
        self.assertEqual(env["SGLANG_UNEVEN_DCP"], "0")

    def test_barlink_arm_sets_transport(self):
        env, _ = build_command(arm_by_name("E_barlink"), model_path="/m", port=1)
        self.assertEqual(env["SGLANG_BARLINK_TRANSPORT"], "device")

    def test_base_flags_unmutated_between_arms(self):
        """build_command must not mutate the shared BASE_FLAGS tuple."""
        before = tuple(BASE_FLAGS)
        build_command(arm_by_name("G_all_axes"), model_path="/m", port=1)
        self.assertEqual(tuple(BASE_FLAGS), before)


class TestRenderPlan(CustomTestCase):
    def test_plan_lists_every_arm_and_a_total(self):
        text = render_plan()
        self.assertIn("A_default", text)
        self.assertIn("reject_dcp_topk", text)
        self.assertIn("total estimated card time", text)

    def test_plan_flags_the_bar1_caveat(self):
        self.assertIn("cold graph cache", render_plan())


if __name__ == "__main__":
    unittest.main()
