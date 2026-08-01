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

    def test_an_arm_can_override_a_base_flag(self):
        """Arm H turns spec off; argparse takes the last value, so the arm's
        --speculative-algorithm none wins over the base NEXTN."""
        arm = arm_by_name("H_ps2_prefill_spill")
        _, argv = build_command(arm, model_path="/m", port=30000)
        # base sets NEXTN, arm appends none AFTER it
        i_base = argv.index("--speculative-algorithm")
        i_last = len(argv) - 1 - argv[::-1].index("--speculative-algorithm")
        self.assertGreater(i_last, i_base, "arm override must come after the base")
        self.assertEqual(argv[i_last + 1], "none")

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
