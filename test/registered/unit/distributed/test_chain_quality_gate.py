# SPDX-License-Identifier: Apache-2.0
"""#328: the semantic chain-quality gate.

Hermetic. The gate is pure arithmetic over graded scores, so the whole
contract is testable without a server, a card, or a fixture from a run.

The tests are written from the rule as prose, because the two ways this gate
can be wrong are both silent:

* judging content by TEXT IDENTITY (the #360/#365 finding), which fails on a
  near-tie flip that changed nothing about the answer;
* judging the cross-arm delta against a PRE-REGISTERED constant (the #274
  lesson: a 1e-3 threshold would have reported world B where the measurement
  said world A).

So there is a test for each: a flip that keeps the content scores equal must
be GREEN, and the band must come from the runs rather than from a constant.
"""

import importlib.util
import json
import os
import sys
import tempfile
import unittest

_SCRIPTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "..", "scripts", "dual_group",
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gate = _load("chain_quality_gate", os.path.join(_SCRIPTS, "chain_quality_gate.py"))


def _run(text):
    return {"text": text, "output_ids": [1, 2, 3]}


def _arm(name, alphabet=("w x y z", "w x y z"), squares=None):
    """A stock_spec_control-shaped report for one arm."""
    prompts = {"alphabet": {"run_a": _run(alphabet[0]), "run_b": _run(alphabet[1])}}
    if squares is not None:
        prompts["squares"] = {"run_a": _run(squares[0]), "run_b": _run(squares[1])}
    return {"arm": name, "prompts": prompts}


class TestBand(unittest.TestCase):
    def test_band_is_the_wider_of_the_two_repeat_spreads(self):
        # The candidate's own jitter must not be charged against a quieter
        # reference arm's floor.
        self.assertEqual(gate.band_of(4, 4, 3, 1), 2)
        self.assertEqual(gate.band_of(4, 1, 3, 3), 3)

    def test_a_perfectly_still_pair_has_a_zero_band(self):
        self.assertEqual(gate.band_of(4, 4, 4, 4), 0)


class TestVerdictRule(unittest.TestCase):
    def test_delta_inside_the_measured_band_is_green(self):
        # ref repeats 4/3 -> band 1; candidate 3 vs ref 4 -> margin 1 <= 1.
        e = gate.judge_scores("alphabet", 4, 3, 3, 3)
        self.assertEqual(e["verdict"], gate.GREEN)
        self.assertEqual(e["band"], 1)
        self.assertEqual(e["margin"], 1)

    def test_delta_outside_the_band_is_red_with_a_reason(self):
        e = gate.judge_scores("alphabet", 4, 4, 1, 1)
        self.assertEqual(e["verdict"], gate.RED)
        self.assertEqual(e["band"], 0)
        self.assertEqual(e["margin"], 3)
        self.assertIn("exceeds the same-boot A-vs-A band", e["reason"])

    def test_a_better_candidate_is_green_and_labelled_better(self):
        e = gate.judge_scores("alphabet", 2, 2, 4, 4)
        # Scoring HIGHER than the reference is not a regression; the gate
        # still reports it as outside the band, but the direction says why.
        self.assertEqual(e["direction"], "better")

    def test_an_unstable_candidate_widens_its_own_band(self):
        # Candidate repeats 4/1 -> band 3, so a cross-arm delta of 3 passes:
        # the arm's own noise is that large, and the gate says so rather than
        # calling its jitter a regression.
        e = gate.judge_scores("alphabet", 4, 4, 1, 4)
        self.assertEqual(e["band"], 3)
        self.assertEqual(e["verdict"], gate.GREEN)

    def test_a_still_agreeing_pair_is_named_not_just_passed(self):
        e = gate.judge_scores("alphabet", 4, 4, 4, 4)
        self.assertEqual(e["verdict"], gate.GREEN)
        self.assertIn("held still and agree", e["note"])


class TestNoConstantAndNoIdentity(unittest.TestCase):
    """The two lessons this gate exists to encode."""

    def test_the_band_comes_from_the_runs_not_from_a_constant(self):
        # Identical cross-arm delta, different measured noise -> different
        # verdicts. A pre-registered threshold cannot produce this.
        quiet = gate.judge_scores("alphabet", 4, 4, 2, 2)
        noisy = gate.judge_scores("alphabet", 4, 2, 2, 4)
        self.assertEqual(quiet["margin"], noisy["margin"])
        self.assertEqual(quiet["verdict"], gate.RED)
        self.assertEqual(noisy["verdict"], gate.GREEN)

    def test_a_near_tie_flip_that_keeps_the_content_is_green(self):
        # Different TEXT, same graded content: the trajectory diverged at a
        # near tie and still emitted the determined sequence. Text identity
        # would call this a failure; the gate must not.
        ref = _arm("nospec", alphabet=("w x y z", "w x y z"))
        cand = _arm("spec", alphabet=("w  x\ny\nz extra", "w x y z"))
        res = gate.judge_run(ref, cand)
        self.assertEqual(res["verdict"], gate.GREEN)

    def test_content_that_actually_collapsed_is_red(self):
        ref = _arm("nospec", alphabet=("w x y z", "w x y z"))
        cand = _arm("spec", alphabet=("q q q q", "q q q q"))
        res = gate.judge_run(ref, cand)
        self.assertEqual(res["verdict"], gate.RED)


class TestVoidIsNotAPass(unittest.TestCase):
    def test_a_missing_repeat_is_void(self):
        e = gate.judge_scores("alphabet", 4, None, 4, 4)
        self.assertEqual(e["verdict"], gate.VOID)
        self.assertIn("no constant stands in for it", e["reason"])

    def test_an_ungraded_prompt_is_void_not_zero(self):
        e = gate.judge_scores("mystery", -1, -1, -1, -1)
        self.assertEqual(e["verdict"], gate.VOID)
        self.assertIn("no scorer", e["reason"])

    def test_the_harness_void_marker_is_honoured(self):
        ref = _arm("nospec")
        ref["prompts"]["alphabet"]["void"] = "a-vs-a floor not byte-identical"
        res = gate.judge_run(ref, _arm("spec"))
        self.assertEqual(res["verdict"], gate.VOID)
        self.assertIn("marked void by the harness", res["prompts"][0]["reason"])

    def test_a_prompt_present_in_only_one_arm_is_void(self):
        ref = _arm("nospec", squares=("12 144 13 169", "12 144 13 169"))
        res = gate.judge_run(ref, _arm("spec"))
        self.assertEqual(res["verdict"], gate.VOID)

    def test_void_outranks_green_in_the_run_verdict(self):
        # One clean prompt plus one unmeasurable one is not a pass.
        ref = _arm("nospec", squares=("12 144", "12 144"))
        cand = _arm("spec", squares=("12 144", "12 144"))
        cand["prompts"]["squares"]["run_b"] = {"output_ids": []}  # no text
        res = gate.judge_run(ref, cand)
        self.assertEqual(res["verdict"], gate.VOID)
        self.assertEqual(res["n_green"], 1)

    def test_red_outranks_void(self):
        ref = _arm("nospec", squares=("12 144", "12 144"))
        cand = _arm("spec", alphabet=("q q q q", "q q q q"),
                    squares=("12 144", "12 144"))
        cand["prompts"]["squares"]["run_b"] = {"output_ids": []}
        res = gate.judge_run(ref, cand)
        self.assertEqual(res["verdict"], gate.RED)


class TestMachineSurface(unittest.TestCase):
    def test_the_verdict_is_json_serialisable_and_carries_the_method(self):
        res = gate.judge_run(_arm("nospec"), _arm("spec"))
        payload = json.loads(json.dumps(res))
        self.assertEqual(payload["gate"], "chain_quality")
        self.assertIn("never text identity", payload["method"])
        self.assertIn("never a pre-registered constant", payload["method"])

    def test_per_prompt_bands_are_reported_not_averaged(self):
        ref = _arm("nospec", squares=("12 144 13 169", "12 144 13 169"))
        cand = _arm("spec", squares=("12 144 13 169", "12 144 13 169"))
        res = gate.judge_run(ref, cand)
        self.assertEqual(res["n_prompts"], 2)
        for e in res["prompts"]:
            self.assertIn("band", e)

    def test_cli_exit_codes_distinguish_green_red_and_void(self):
        with tempfile.TemporaryDirectory() as d:
            ref_p = os.path.join(d, "ref.json")
            cand_p = os.path.join(d, "cand.json")
            with open(ref_p, "w") as f:
                json.dump(_arm("nospec"), f)
            with open(cand_p, "w") as f:
                json.dump(_arm("spec"), f)
            self.assertEqual(
                gate.main(["--reference", ref_p, "--candidate", cand_p]), 0
            )
            with open(cand_p, "w") as f:
                json.dump(_arm("spec", alphabet=("q q", "q q")), f)
            self.assertEqual(
                gate.main(["--reference", ref_p, "--candidate", cand_p]), 1
            )
            broken = _arm("spec")
            broken["prompts"]["alphabet"]["run_b"] = {"output_ids": []}
            with open(cand_p, "w") as f:
                json.dump(broken, f)
            self.assertEqual(
                gate.main(["--reference", ref_p, "--candidate", cand_p]), 2
            )

    def test_the_human_rendering_names_the_band(self):
        text = gate.format_report(gate.judge_run(_arm("nospec"), _arm("spec")))
        self.assertIn("chain-quality gate:", text)
        self.assertIn("band", text)


if __name__ == "__main__":
    unittest.main()
