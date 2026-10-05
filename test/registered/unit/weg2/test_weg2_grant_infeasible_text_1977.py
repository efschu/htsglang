# SPDX-License-Identifier: Apache-2.0
"""deskq 1988 (a) / report 1977: the '#1640 GRANT-INFEASIBLE' line TEXT (dual P, PP0). ``tokens`` is the GRANT-SUM of
all held prompts; when the P bytes of just-taken grants close the gap the line says 'waits for the return', not
'only a falling level frees it'. Text only: ``infeasible_cards`` (the decision) is untouched.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import dual_p_kv_stage as S  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class InfeasibleLineText1977(CustomTestCase):
    def test_p_closing_the_gap_says_waits_for_the_return_and_names_the_sum(self):
        # need 1000 > have 400 + D 100, but have + P + D = 400 + 700 + 100 >= 1000
        line = S._infeasible_line("weg2-0-86", 449044, 266240, [(2, 1000, 400, 700, 100)])
        self.assertIn("tokens=449044(GRANT-SUM of all held prompts, not this prompt)", line)
        self.assertIn("level_tokens=266240", line)
        self.assertIn("2:need=1000,pool=1200,have=400,P=700,D=100", line)
        self.assertIn("waits for the return of the followers' pre-charge", line)
        self.assertNotIn("only a falling level frees it", line)

    def test_p_not_closing_the_gap_keeps_the_hard_verdict(self):
        line = S._infeasible_line("weg2-0-9", 62343, 196608, [(0, 4080, 1100, 0, 2000)])
        self.assertIn("0:need=4080,pool=3100,have=1100,P=0,D=2000", line)
        self.assertIn("only a falling level or a lower grant sum frees it", line)
        self.assertNotIn("waits for the return", line)

    def test_one_card_not_covered_by_p_is_the_hard_verdict(self):
        line = S._infeasible_line("r", 1, 1, [(1, 1000, 400, 700, 100), (0, 4080, 1100, 0, 2000)])
        self.assertIn("only a falling level", line)

    def test_decision_function_untouched(self):
        self.assertEqual(S.infeasible_cards([], {}, 4096), [])
