# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#274 slice D v1: the spreading decider.

The decider has exactly one measured rule -- do not put an SM-saturating load
on a card that already carries a lane -- so the tests are about that rule, the
analytic label it rests on, and the four honesty guards: unmeasured regimes
report no E, the pair matrix may only break ties, an infeasible rig is named
rather than solved, and a fixed lane stays where it was pinned.
"""

from __future__ import annotations

import json
import unittest

from sglang.srt.planner import spread as sp
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _load(key, label, **kw):
    return sp.LaneLoad(key=key, label=label, **kw)


class TestLabel(CustomTestCase):
    """The label is analytic: step width against the card's machine balance."""

    # RTX 5090 as the card probe measured it on this rig.
    GEMM_FP8 = 566.88
    MEMBW = 1660.4

    def test_machine_balance_matches_the_probed_rates(self):
        bal = sp.machine_balance(self.GEMM_FP8, self.MEMBW)
        self.assertAlmostEqual(bal, 341.4, delta=1.0)

    def test_the_same_weights_flip_label_with_the_step_width(self):
        # A Q3_K-ish checkpoint: ~0.44 bytes per parameter.
        params, wbytes = 27e9, 27e9 * 0.44
        bal = sp.machine_balance(self.GEMM_FP8, self.MEMBW)
        verify = sp.step_intensity(params=params, weight_bytes=wbytes, tokens=4)
        prefill = sp.step_intensity(params=params, weight_bytes=wbytes, tokens=2048)
        self.assertEqual(sp.label_from_intensity(verify, bal), sp.BW_BOUND)
        self.assertEqual(sp.label_from_intensity(prefill, bal), sp.SM_BOUND)
        # And the crossover is a token count, not a model property: that is
        # what makes batch/chunk size the intensity knob.
        self.assertLess(verify, bal)
        self.assertGreater(prefill, bal)

    def test_an_unlabelled_rate_is_unknown_not_guessed(self):
        self.assertEqual(sp.label_from_intensity(None, 341.0), sp.UNKNOWN)
        self.assertEqual(sp.label_from_intensity(100.0, None), sp.UNKNOWN)
        self.assertIsNone(sp.step_intensity(params=0.0, weight_bytes=1.0, tokens=8))

    def test_label_from_a_cost_model(self):
        from types import SimpleNamespace

        model = SimpleNamespace(
            fixed_params=[1e9, 0.0],
            unit_params=1e9,
            weight_bytes=lambda units: [1.0e10, 1.0],
        )
        rates = SimpleNamespace(gemm_tflops=[566.88], membw_gbs=[1660.4])
        load = sp.lane_load_from_model(
            "pd", model, [9, 0], rates, tokens=2048, card_count=1
        )
        self.assertEqual(load.label, sp.SM_BOUND)
        self.assertIn("analytic", load.basis)


class TestObjective(CustomTestCase):
    def test_two_lanes_prefer_separate_cards(self):
        ans = sp.spread_plan(
            [_load("pd", sp.SM_BOUND), _load("main", sp.BW_BOUND)], [0, 1]
        )
        self.assertTrue(ans.ok)
        self.assertNotEqual(ans.assignment["pd"], ans.assignment["main"])
        self.assertAlmostEqual(ans.score, 2.0, places=6)
        self.assertEqual(ans.sm_colocations, 0)
        self.assertAlmostEqual(ans.expected_e, 2.0, places=6)

    def test_when_a_card_must_be_shared_the_saturating_lane_is_kept_off_it(self):
        """The dir1 constellation: one card free, so exactly one of the two
        movable lanes must share with the pinned serving rank."""
        loads = [
            _load("serving", sp.BW_BOUND, cards=(0,)),
            _load("prefill_lane", sp.SM_BOUND, allowed_cards=(0, 1)),
            _load("decode_lane", sp.BW_BOUND, allowed_cards=(0, 1)),
        ]
        ans = sp.spread_plan(loads, [0, 1])
        self.assertTrue(ans.ok)
        self.assertEqual(ans.assignment["prefill_lane"], (1,))
        self.assertEqual(ans.assignment["decode_lane"], (0,))
        # 1.440 on the shared card + 1.0 on the card that got the prefill lane.
        self.assertAlmostEqual(ans.score, 2.44, places=6)

    def test_the_bad_pairing_scores_below_the_chosen_one(self):
        """The A/B of the rig test, on the desk: same lanes, same card, only
        the load shape differs, and the decider ranks them the way slice C
        measured them."""
        shared_sm = sp.spread_plan(
            [
                _load("serving", sp.BW_BOUND, cards=(0,)),
                _load("lane", sp.SM_BOUND, cards=(0,)),
            ],
            [0],
        )
        shared_bw = sp.spread_plan(
            [
                _load("serving", sp.BW_BOUND, cards=(0,)),
                _load("lane", sp.BW_BOUND, cards=(0,)),
            ],
            [0],
        )
        self.assertLess(shared_sm.score, shared_bw.score)
        self.assertAlmostEqual(shared_sm.expected_e, 1.130, places=6)
        self.assertAlmostEqual(shared_bw.expected_e, 1.440, places=6)

    def test_a_forced_bad_placement_is_named_not_hidden(self):
        ans = sp.spread_plan(
            [
                _load("serving", sp.BW_BOUND, cards=(0,)),
                _load("lane", sp.SM_BOUND, cards=(0,)),
            ],
            [0],
        )
        self.assertEqual(ans.sm_colocations, 1)
        self.assertTrue(any("not fully met" in r for r in ans.reasons))

    def test_fewer_sm_colocations_breaks_a_tie(self):
        # Both placements score 1.130 (one SM lane shared), but one of them
        # co-locates TWO saturating lanes, which was never measured and can
        # only be worse. The decider must not be indifferent.
        loads = [
            _load("a", sp.SM_BOUND, allowed_cards=(0,)),
            _load("b", sp.SM_BOUND, allowed_cards=(0, 1)),
        ]
        ans = sp.spread_plan(loads, [0, 1])
        self.assertEqual(ans.assignment["b"], (1,))
        self.assertEqual(ans.sm_colocations, 0)


class TestHonesty(CustomTestCase):
    def test_two_saturating_lanes_report_no_expected_e(self):
        ans = sp.spread_plan(
            [
                _load("a", sp.SM_BOUND, cards=(0,)),
                _load("b", sp.SM_BOUND, cards=(0,)),
            ],
            [0],
        )
        self.assertIsNone(ans.expected_e)
        self.assertEqual(ans.per_card[0].provenance, "bounded")
        self.assertIn("never measured", ans.per_card[0].note)

    def test_three_lanes_on_one_card_report_no_expected_e(self):
        ans = sp.spread_plan(
            [_load(k, sp.BW_BOUND, cards=(0,)) for k in ("a", "b", "c")], [0]
        )
        self.assertIsNone(ans.expected_e)
        self.assertEqual(ans.per_card[0].provenance, "absent")
        self.assertIn("outside the measured regime", ans.per_card[0].note)

    def test_an_unlabelled_lane_reports_no_expected_e_and_says_why(self):
        ans = sp.spread_plan(
            [
                _load("a", sp.BW_BOUND, cards=(0,)),
                _load("b", sp.UNKNOWN, cards=(0,)),
            ],
            [0],
        )
        self.assertIsNone(ans.expected_e)
        self.assertTrue(any("could not be labelled" in r for r in ans.reasons))

    def test_an_impossible_rig_is_named_not_solved(self):
        ans = sp.spread_plan([_load("a", sp.BW_BOUND, card_count=4)], [0, 1])
        self.assertFalse(ans.ok)
        self.assertTrue(any("no placement exists" in r for r in ans.reasons))

    def test_an_infeasible_predicate_is_named_not_solved(self):
        ans = sp.spread_plan(
            [_load("a", sp.BW_BOUND)], [0, 1], feasible=lambda _a: False
        )
        self.assertFalse(ans.ok)
        self.assertTrue(any("infeasible" in r for r in ans.reasons))

    def test_the_feasibility_seam_is_consulted_not_reimplemented(self):
        seen = []

        def feasible(assignment):
            seen.append(dict(assignment))
            return assignment["a"] == (1,)

        ans = sp.spread_plan([_load("a", sp.BW_BOUND)], [0, 1], feasible=feasible)
        self.assertTrue(ans.ok)
        self.assertEqual(ans.assignment["a"], (1,))
        self.assertEqual(len(seen), 2)

    def test_max_lanes_per_card_is_honoured(self):
        ans = sp.spread_plan(
            [_load(k, sp.BW_BOUND) for k in ("a", "b")], [0, 1], max_lanes_per_card=1
        )
        self.assertTrue(ans.ok)
        self.assertNotEqual(ans.assignment["a"], ans.assignment["b"])


class TestPairMatrix(CustomTestCase):
    """The pair matrix informs, it does not decide."""

    PROBE = {
        "cards": [
            {"uuid": "A", "cuda_index": 0},
            {"uuid": "B", "cuda_index": 1},
            {"uuid": "C", "cuda_index": 2},
        ],
        "pairs": [
            {"src_uuid": "A", "dst_uuid": "B", "bandwidth_gbs": 4.44},
            {"src_uuid": "B", "dst_uuid": "A", "bandwidth_gbs": 4.40},
            {"src_uuid": "A", "dst_uuid": "C", "bandwidth_gbs": 9.10},
            {"src_uuid": "C", "dst_uuid": "A", "bandwidth_gbs": 9.00},
            {"src_uuid": "B", "dst_uuid": "C", "bandwidth_gbs": 3.10},
            {"src_uuid": "C", "dst_uuid": "B", "bandwidth_gbs": 3.05},
        ],
    }

    def test_translation_from_uuids_to_indices(self):
        pairs = sp.pair_bandwidth_from_probe(self.PROBE, [0, 1, 2])
        self.assertAlmostEqual(pairs[(0, 1)], 4.44)
        self.assertAlmostEqual(pairs[(2, 1)], 3.05)
        self.assertEqual(len(pairs), 6)

    def test_a_two_card_lane_takes_the_fastest_pair_when_all_else_is_equal(self):
        pairs = sp.pair_bandwidth_from_probe(self.PROBE, [0, 1, 2])
        ans = sp.spread_plan(
            [_load("wide", sp.BW_BOUND, card_count=2)],
            [0, 1, 2],
            pair_bandwidth_gbs=pairs,
        )
        self.assertEqual(ans.assignment["wide"], (0, 2))  # 9.0/9.1, not 4.4 or 3.05

    def test_the_pair_matrix_never_overrides_the_load_shape(self):
        pairs = sp.pair_bandwidth_from_probe(self.PROBE, [0, 1, 2])
        loads = [
            _load("serving", sp.BW_BOUND, cards=(0,)),
            _load("sm_lane", sp.SM_BOUND, allowed_cards=(0, 1)),
        ]
        ans = sp.spread_plan(loads, [0, 1], pair_bandwidth_gbs=pairs)
        self.assertEqual(ans.assignment["sm_lane"], (1,))
        self.assertEqual(ans.sm_colocations, 0)


class TestPayload(CustomTestCase):
    def test_the_answer_is_json_serializable(self):
        ans = sp.spread_plan(
            [
                _load("serving", sp.BW_BOUND, cards=(0,)),
                _load("lane", sp.SM_BOUND, allowed_cards=(0, 1)),
            ],
            [0, 1],
        )
        blob = json.dumps(ans.to_json())
        self.assertIn("assignment", blob)
        self.assertIn("per_card", blob)

    def test_the_objective_is_stated_with_its_numbers(self):
        text = sp.describe_objective()
        self.assertIn("1.130", text)
        self.assertIn("1.440", text)


if __name__ == "__main__":
    unittest.main()
