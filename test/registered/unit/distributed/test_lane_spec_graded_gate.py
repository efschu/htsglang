# SPDX-License-Identifier: Apache-2.0
"""#274/#284: the graded lane-spec gate, and the control that reframed it.

Hermetic. No server, no card, no fixture from a run -- the scorers and the
world-decision rule are pure functions and are tested as such, so a change to
either fails here before it costs a card window.
"""

import importlib.util
import os
import sys
import unittest

_R12 = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "..", "scripts", "dual_group", "r12",
)


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_R12, f"{name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


graded = _load("graded")
verdict = _load("verdict")


class TestAlphabetScorer(unittest.TestCase):
    def test_the_determined_tail_scores_full(self):
        self.assertEqual(graded.score("alphabet", "w\nx\ny\nz\n")["score"], 4)

    def test_scoring_stops_at_the_first_wrong_letter(self):
        self.assertEqual(graded.score("alphabet", "w\nx\nq\nz\n")["score"], 2)

    def test_nothing_right_is_zero_not_negative(self):
        self.assertEqual(graded.score("alphabet", "hello there")["score"], 0)

    def test_text_past_z_does_not_change_the_score(self):
        """The prompt determines four letters; what follows is not graded.

        This is the property that makes the score usable as a gate: two arms
        that agree on w..z and then wander differently must score the same,
        or the gate is an identity check wearing a number.
        """
        a = graded.score("alphabet", "w\nx\ny\nz\na\nb\nc\n")["score"]
        b = graded.score("alphabet", "w\nx\ny\nz\n\nThe sequence ends.")["score"]
        self.assertEqual(a, b)
        self.assertEqual(a, 4)


class TestSquaresScorer(unittest.TestCase):
    def test_consecutive_correct_lines_count(self):
        self.assertEqual(
            graded.score("squares", "12 144\n13 169\n14 196\n")["score"], 3
        )

    def test_a_wrong_square_stops_the_count(self):
        self.assertEqual(graded.score("squares", "12 144\n13 170\n14 196\n")["score"], 1)

    def test_a_wrong_index_stops_the_count(self):
        self.assertEqual(graded.score("squares", "12 144\n14 196\n")["score"], 1)


class TestUnscoredPromptIsNotZero(unittest.TestCase):
    def test_missing_scorer_returns_minus_one(self):
        """`no scorer` and `scored zero` must not read the same in a table."""
        self.assertEqual(graded.score("code", "anything")["score"], -1)


class TestTrajectoryClassifier(unittest.TestCase):
    """`verdict.compare` must stay equivalent to the r8 gate's classifier."""

    def test_identical(self):
        self.assertEqual(
            verdict.compare([1, 2, 3], [1, 2, 3])["classification"], "identical"
        )

    def test_length_end_only(self):
        got = verdict.compare([1, 2, 3], [1, 2, 3, 4])
        self.assertEqual(got["classification"], "length_end_only")
        self.assertIsNone(got["first_divergent_index"])

    def test_content_divergence_reports_the_index(self):
        got = verdict.compare([1, 2, 3], [1, 9, 3])
        self.assertEqual(got["classification"], "content_divergence")
        self.assertEqual(got["first_divergent_index"], 1)

    def test_it_agrees_with_the_r8_gate_on_every_shape(self):
        """Literal cross-check against the harness this control mirrors."""
        spec = importlib.util.spec_from_file_location(
            "lane_spec_window",
            os.path.join(_R12, "..", "r8", "lane_spec_window.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["lane_spec_window"] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception as exc:  # pragma: no cover - import guard only
            self.skipTest(f"r8 harness not importable here: {exc}")
        cases = [
            ([1, 2, 3], [1, 2, 3]),
            ([1, 2, 3], [1, 2, 3, 4]),
            ([1, 2, 3], [1, 9, 3]),
            ([], []),
        ]
        for ref, got in cases:
            self.assertEqual(
                verdict.compare(ref, got)["classification"],
                mod.compare_trajectories(ref, got)["classification"],
                msg=f"classifiers disagree on {ref} vs {got}",
            )


class TestNearTieRule(unittest.TestCase):
    def test_the_threshold_is_generous_to_the_defect_world(self):
        """0.05 nats is ~50x the 1e-3 a reassociated reduction moves a logit.

        Pinned so a later edit cannot quietly widen the band until every
        divergence reads as a near tie.
        """
        self.assertLessEqual(verdict.NEAR_TIE_ABS, 0.05)
        self.assertGreater(verdict.NEAR_TIE_ABS, 1e-3)


class TestWorldDecision(unittest.TestCase):
    """The rule was written before the numbers; these pin all four branches."""

    @staticmethod
    def _arm(ids, margins=None, text="", void=None):
        entry = {
            "run_a": {"output_ids": ids, "margins": margins or [], "text": text},
            "run_b": {"output_ids": ids, "margins": margins or [], "text": text},
        }
        if void:
            entry["void"] = void
        return entry

    def test_stock_identical_points_at_the_lane(self):
        p = verdict.judge_prompt(
            "alphabet",
            self._arm([1, 2, 3], [5.0, 5.0, 5.0], "w\nx\ny\nz\n"),
            self._arm([1, 2, 3], [5.0, 5.0, 5.0], "w\nx\ny\nz\n"),
        )
        self.assertEqual(p["classification"], "identical")
        self.assertIsNone(p["first_divergent_index"])

    def test_stock_diverges_at_a_near_tie(self):
        p = verdict.judge_prompt(
            "alphabet",
            self._arm([1, 2, 3], [5.0, 0.001, 5.0], "w\nx\ny\nz\n"),
            self._arm([1, 9, 3], [5.0, 0.001, 5.0], "w\nx\ny\nz\n"),
        )
        self.assertEqual(p["classification"], "content_divergence")
        self.assertEqual(p["first_divergent_index"], 1)
        self.assertTrue(p["near_tie"])
        self.assertEqual(p["graded"]["delta"], 0)

    def test_stock_diverges_at_a_comfortable_margin(self):
        p = verdict.judge_prompt(
            "alphabet",
            self._arm([1, 2, 3], [5.0, 4.0, 5.0], "w\nx\ny\nz\n"),
            self._arm([1, 9, 3], [5.0, 4.0, 5.0], "w\nx\nq\n"),
        )
        self.assertFalse(p["near_tie"])
        self.assertEqual(p["graded"]["delta"], -2)

    def test_a_void_arm_carries_no_verdict(self):
        p = verdict.judge_prompt(
            "alphabet",
            self._arm([1, 2, 3], void="floor not byte-identical"),
            self._arm([1, 9, 3]),
        )
        self.assertEqual(p["verdict"], "void")



class TestPerturbationBand(unittest.TestCase):
    """The band is measured from the pair, and it must stop at divergence."""

    @staticmethod
    def _pair(ref_ids, got_ids, ref_m, got_m):
        return (
            {"run_a": {"output_ids": ref_ids, "margins": ref_m, "text": ""}},
            {"run_a": {"output_ids": got_ids, "margins": got_m, "text": ""}},
        )

    def test_it_measures_the_gap_difference_where_the_arms_agree(self):
        ns, sp = self._pair([1, 2, 3], [1, 2, 3], [5.0, 4.0, 3.0], [5.1, 4.3, 3.0])
        band = verdict.perturbation_band(ns, sp)
        self.assertEqual(band["n"], 3)
        self.assertAlmostEqual(band["max"], 0.3, places=6)

    def test_positions_from_the_divergence_on_are_excluded(self):
        """Past the flip the arms condition on different prefixes.

        Including those would let the consequence of the divergence widen the
        band that is supposed to explain it.
        """
        ns, sp = self._pair([1, 2, 3], [1, 9, 3], [5.0, 4.0, 3.0], [5.1, 99.0, 88.0])
        band = verdict.perturbation_band(ns, sp)
        self.assertEqual(band["n"], 1)
        self.assertAlmostEqual(band["max"], 0.1, places=6)

    def test_an_empty_band_does_not_explain_anything(self):
        ns, sp = self._pair([1], [9], [5.0], [5.0])
        self.assertEqual(verdict.perturbation_band(ns, sp)["n"], 0)

    def test_a_degenerate_band_cannot_alone_explain_a_flip(self):
        """Fewer than three agreeing positions is not a band.

        Guard against the shape where an arm diverges almost immediately and
        the one or two samples before it happen to be wide.
        """
        ns = {"run_a": {"output_ids": [1, 2, 9], "margins": [0.1, 0.2, 5.0],
                        "text": ""}}
        sp = {"run_a": {"output_ids": [1, 2, 3], "margins": [0.9, 1.4, 5.0],
                        "text": ""}}
        p = verdict.judge_prompt("alphabet", ns, sp)
        self.assertEqual(p["perturbation_band"]["n"], 2)
        self.assertFalse(p["inside_band"])



class TestGateVerdictIsTheScoreNotTheTokens(unittest.TestCase):
    """The reframed r8 gate: identity is reported, the score decides.

    Exercised through the real ``run_gate`` with the server calls stubbed, so
    the rule under test is the one the recipe runs.
    """

    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "lane_spec_window",
            os.path.join(_R12, "..", "r8", "lane_spec_window.py"),
        )
        self.mod = importlib.util.module_from_spec(spec)
        sys.modules["lane_spec_window"] = self.mod
        try:
            spec.loader.exec_module(self.mod)
        except Exception as exc:  # pragma: no cover - import guard only
            self.skipTest(f"r8 harness not importable here: {exc}")

    def _run(self, nospec_text, spec_text):
        m = self.mod
        texts = {"nospec": nospec_text, "spec": spec_text}

        def fake_lane_run(base, job, **kw):
            key = "spec" if job.get("spec") else "nospec"
            # ids are opaque here; detokenize is stubbed to map them back
            return [{"output_ids": [key], "decode_ms_mean": 1.0,
                     "accept_len_mean": 1.0, "verify_graph_rounds": 0,
                     "spec_rounds": 1}]

        orig = (m.lane_run, m.tokenize, m.detokenize)
        m.lane_run = fake_lane_run
        m.tokenize = lambda base, text, tok: [1, 2, 3]
        m.detokenize = lambda ids, tok: texts[ids[0]]
        try:
            return m.run_gate(
                "http://x", "tok", ["alphabet"], 8, 1, "target_verify",
                deadline=9e18,
            )
        finally:
            m.lane_run, m.tokenize, m.detokenize = orig

    def test_different_tokens_same_score_is_coherent(self):
        """The r12 finding, as a rule: a near-tie flip must not fail the gate."""
        out = self._run("w\nx\ny\nz\n", "w\nx\ny\nz\n\n")
        p = out["prompts"]["alphabet"]
        self.assertEqual(p["graded_delta"], 0)
        self.assertTrue(p["graded_within_band"])
        self.assertEqual(out["verdict"], "coherent")

    def test_a_worse_score_still_fails(self):
        out = self._run("w\nx\ny\nz\n", "w\nx\nq\nq\n")
        p = out["prompts"]["alphabet"]
        self.assertLess(p["graded_delta"], 0)
        self.assertFalse(p["graded_within_band"])
        self.assertEqual(out["verdict"], "divergent")

    def test_the_identity_rate_is_reported_not_required(self):
        out = self._run("w\nx\ny\nz\n", "w\nx\ny\nz\n\n")
        self.assertEqual(out["token_identity_rate"], 0.0)
        self.assertEqual(out["verdict"], "coherent")
        self.assertIn("Token identity", out["verdict_criterion"])



class TestLaneMarginProbe(unittest.TestCase):
    """The lane's own top-2 margin, wired but off by default (#284 -> r12).

    The r12 control answered the near-tie question for STOCK speculation over
    /generate's logprob channel. The lane commits its own tokens and reports
    only ``output_ids``, so the same question could not be put to the lane at
    all. These pin the three properties that make the added probe usable and
    safe: it is off unless asked for, it lines up with the committed tokens,
    and it costs nothing when off.
    """

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:  # pragma: no cover - torch is a hard dep here
            self.skipTest("torch not importable")
        from sglang.srt.model_executor import dual_group_lane as dgl

        self.dgl = dgl
        self.lane = dgl.DualGroupLane.__new__(dgl.DualGroupLane)

    def tearDown(self):
        os.environ.pop("SGLANG_LANE_MARGIN_PROBE", None)

    def test_it_is_off_unless_the_env_var_is_set(self):
        os.environ.pop("SGLANG_LANE_MARGIN_PROBE", None)
        self.assertFalse(self.lane._margin_probe_on())
        os.environ["SGLANG_LANE_MARGIN_PROBE"] = "1"
        self.assertTrue(self.lane._margin_probe_on())

    def test_the_margin_is_the_gap_between_the_two_best_logits(self):
        import torch

        logits = torch.tensor([[1.0, 7.5, 3.25, -2.0]])
        self.assertAlmostEqual(
            self.dgl.DualGroupLane._top2_margin(logits), 4.25, places=5
        )

    def test_a_tie_reads_as_zero(self):
        import torch

        logits = torch.tensor([[5.0, 5.0, 1.0]])
        self.assertAlmostEqual(
            self.dgl.DualGroupLane._top2_margin(logits), 0.0, places=6
        )

    def test_record_appends_one_entry_per_committed_token(self):
        job = {}
        self.lane._record_margin(job, 0.25)
        self.lane._record_margin(job, 1.5)
        self.assertEqual(job["_margins"], [0.25, 1.5])

    def test_a_none_margin_is_not_recorded(self):
        """Probe off must leave no hole in the list, not a None in it.

        `verdict.py` walks margins positionally against output_ids; a None
        placeholder would silently shift every later position by one.
        """
        job = {}
        self.lane._record_margin(job, None)
        self.lane._record_margin(job, 0.5)
        self.assertEqual(job["_margins"], [0.5])

if __name__ == "__main__":
    unittest.main()
