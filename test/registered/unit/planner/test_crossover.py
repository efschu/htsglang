"""CPU unit tests for the MLP-concentration crossover: what it may claim.

The crossover between the prefill gain and the decode cost of an MLP weight
concentration is a property of a RIG -- of its card mix, its interconnect, its
model and its quantisation -- not a constant of the feature. These tests pin
that: a number measured on one rig may be shown as that rig's finding and may
never be used as this rig's answer, and with no local measurement the module
says so instead of guessing.
"""

import json
import os
import tempfile
import time
import unittest

from sglang.srt.planner.crossover import (
    MEASURED_ELSEWHERE,
    MEASURED_HERE,
    MODELLED,
    MODELLED_NET_REFUSED,
    MODELLED_PREFILL_NOTE,
    REFERENCE_FINDING,
    STALE_AFTER_S,
    STUDY_KEY,
    STUDY_TIERS,
    ConcentrationPoint,
    CrossoverFinding,
    RigDescriptor,
    load_finding,
    save_finding,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _local(points=None, **kw):
    """A synthetic finding that IS from this rig, for the advice path."""
    base = dict(
        rig=RigDescriptor(
            cards=("card A", "card B", "card B"),
            model="a-model",
            quant="fp8",
            tp_size=3,
            base_vector=(63, 37, 36),
        ),
        points=list(points if points is not None else REFERENCE_FINDING.points),
        provenance=MEASURED_HERE,
        measured_at=time.time(),
        cache_bypass_proven=True,
    )
    base.update(kw)
    return CrossoverFinding(**base)


class TestTheArithmetic(CustomTestCase):
    def test_break_even_is_the_ratio_where_the_two_terms_cancel(self):
        p = ConcentrationPoint(
            vector=(3, 1, 1),
            prefill_ms_per_prompt_token_saved=0.05,
            decode_ms_per_output_token_cost=1.0,
        )
        self.assertAlmostEqual(p.break_even_ratio, 20.0)
        self.assertAlmostEqual(p.net_ms_per_output_token(20.0), 0.0)
        self.assertLess(p.net_ms_per_output_token(10.0), 0.0)
        self.assertGreater(p.net_ms_per_output_token(40.0), 0.0)

    def test_a_vector_that_saves_nothing_never_pays(self):
        p = ConcentrationPoint((2, 1, 1), 0.0, 0.5)
        self.assertIsNone(p.break_even_ratio)

    def test_a_vector_that_costs_nothing_pays_immediately(self):
        p = ConcentrationPoint((2, 1, 1), 0.05, 0.0)
        self.assertEqual(p.break_even_ratio, 0.0)


class TestTheEnvelope(CustomTestCase):
    """Which candidates may be proposed at all -- a structural rule, computed
    from whatever numbers a rig produced, not a list of blessed vectors."""

    def test_a_candidate_that_is_never_the_best_choice_is_pruned(self):
        f = _local()
        on = [p.vector for p in f.envelope()]
        self.assertIn((3, 1, 1), on)
        self.assertIn((6, 1, 1), on)
        self.assertNotIn(
            (4, 1, 1),
            on,
            "4,1,1 is on the envelope, but there is no prompt:output ratio at "
            "which it is the best positive-net choice",
        )

    def test_the_pruned_candidate_carries_the_reason(self):
        f = _local()
        pruned = dict(f.pruned())
        self.assertIn((4, 1, 1), pruned)
        self.assertTrue(pruned[(4, 1, 1)])

    def test_pruning_follows_the_numbers_not_the_vector(self):
        """The same vector survives when the measurement says it should.

        The rule has to be structural, or it is a hardcoded blacklist that
        happens to be right on one rig.
        """
        f = _local(
            points=[
                ConcentrationPoint((3, 1, 1), 0.02, 1.00),
                ConcentrationPoint((4, 1, 1), 0.09, 1.10),
                ConcentrationPoint((6, 1, 1), 0.10, 4.00),
            ]
        )
        self.assertIn((4, 1, 1), [p.vector for p in f.envelope()])

    def test_the_best_choice_moves_with_the_workload(self):
        f = _local()
        self.assertIsNone(
            f.best_for_ratio(5.0), "a concentration is proposed where it loses"
        )
        self.assertEqual(f.best_for_ratio(16.0).vector, (3, 1, 1))
        self.assertEqual(f.best_for_ratio(80.0).vector, (6, 1, 1))

    def test_the_pruned_candidate_is_never_returned_at_any_ratio(self):
        f = _local()
        r = 1.0
        while r < 500.0:
            best = f.best_for_ratio(r)
            if best is not None:
                self.assertNotEqual(best.vector, (4, 1, 1), f"proposed at {r}:1")
            r += 0.5


class TestProvenance(CustomTestCase):
    def test_the_reference_finding_is_labelled_as_another_rigs_result(self):
        self.assertEqual(REFERENCE_FINDING.provenance, MEASURED_ELSEWHERE)
        self.assertFalse(
            REFERENCE_FINDING.usable_for_advice(),
            "a finding measured on another rig is being used as this rig's "
            "answer",
        )

    def test_the_reference_finding_names_its_rig(self):
        label = REFERENCE_FINDING.rig.label()
        for token in ("5090", "3080", "fp8"):
            self.assertIn(token.lower(), label.lower())

    def test_a_local_measurement_is_usable(self):
        self.assertTrue(_local().usable_for_advice())

    def test_a_result_without_cache_bypass_proof_is_refused(self):
        f = _local(cache_bypass_proven=False)
        self.assertFalse(f.usable_for_advice())
        self.assertTrue(any("cache" in c.lower() for c in f.caveats()))

    def test_a_stale_result_is_refused_and_says_how_old(self):
        f = _local(measured_at=time.time() - STALE_AFTER_S - 60.0)
        self.assertTrue(f.is_stale())
        self.assertFalse(f.usable_for_advice())
        self.assertTrue(any("old" in c.lower() or "stale" in c.lower()
                            for c in f.caveats()))

    def test_a_throttled_result_is_kept_and_marked(self):
        """Dropping it would hide a real measurement; using it silently would
        report a throttled rig's numbers as the rig's numbers."""
        f = _local(throttled=True, throttle_reason="sw_thermal_slowdown on card A")
        self.assertTrue(f.usable_for_advice())
        self.assertTrue(any("throttl" in c.lower() for c in f.caveats()))

    def test_provenance_is_validated(self):
        with self.assertRaises(ValueError):
            _local(provenance="pretty sure")


class TestTheModelledSide(CustomTestCase):
    def test_the_prefill_model_carries_its_known_bias(self):
        self.assertIn("1.8", MODELLED_PREFILL_NOTE)
        self.assertIn(MODELLED, (MEASURED_HERE, MEASURED_ELSEWHERE, MODELLED))

    def test_a_modelled_net_is_refused_rather_than_printed(self):
        self.assertTrue(MODELLED_NET_REFUSED)
        self.assertIn("net", MODELLED_NET_REFUSED.lower())


class TestTheStore(CustomTestCase):
    def test_roundtrip(self):
        f = _local()
        back = CrossoverFinding.from_json(json.loads(json.dumps(f.to_json())))
        self.assertEqual(back.rig, f.rig)
        self.assertEqual([p.vector for p in back.points],
                         [p.vector for p in f.points])
        self.assertEqual(back.provenance, f.provenance)

    def test_save_and_load_rig_local(self):
        f = _local()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mlp_crossover.json")
            save_finding(f, path)
            back = load_finding(path, rig=f.rig)
            self.assertIsNotNone(back)
            self.assertEqual(back.rig, f.rig)

    def test_a_finding_for_a_different_rig_is_not_returned(self):
        f = _local()
        other = RigDescriptor(
            cards=("card A", "card A", "card A"),
            model="a-model",
            quant="fp8",
            tp_size=3,
        )
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mlp_crossover.json")
            save_finding(f, path)
            self.assertIsNone(load_finding(path, rig=other))

    def test_a_missing_store_is_absent_not_an_error(self):
        self.assertIsNone(load_finding("/nonexistent/mlp_crossover.json"))


class TestTheStudyOffer(CustomTestCase):
    """The measurement has to be offerable, or nobody will run it."""

    def test_two_tiers_with_a_time_estimate_each(self):
        self.assertGreaterEqual(len(STUDY_TIERS), 2)
        for t in STUDY_TIERS:
            self.assertTrue(t.key)
            self.assertGreater(t.est_runtime_min, 0)
            self.assertTrue(t.what)

    def test_the_quick_tier_is_the_cheaper_one(self):
        by = {t.key: t for t in STUDY_TIERS}
        self.assertIn("quick", by)
        self.assertIn("thorough", by)
        self.assertLess(by["quick"].est_runtime_min, by["thorough"].est_runtime_min)

    def test_every_tier_points_at_a_study_file_that_exists(self):
        import pathlib

        root = pathlib.Path("tools/rig_dashboard/studies")
        if not root.is_dir():
            self.skipTest("study files not in this checkout")
        for t in STUDY_TIERS:
            self.assertTrue(
                (root / t.study_file).is_file(), f"{t.key}: {t.study_file} missing"
            )

    def test_the_tiers_drive_the_registered_scenario(self):
        from sglang.srt.planner.scenarios import SCENARIOS

        self.assertIn(STUDY_KEY, SCENARIOS)

    def test_every_tier_study_is_loadable_by_the_executor(self):
        """A recipe that the runner refuses is not a recipe. Loading also
        validates every arm's launch settings, so a bad rank map fails here
        rather than after the first boot."""
        import pathlib

        from sglang.srt.planner.runner import load_study

        root = pathlib.Path("tools/rig_dashboard/studies")
        if not root.is_dir():
            self.skipTest("study files not in this checkout")
        for t in STUDY_TIERS:
            with self.subTest(tier=t.key):
                study = load_study(str(root / t.study_file))
                self.assertEqual(study.scenario.key, STUDY_KEY)
                self.assertGreaterEqual(len(study.arms), 2)
                self.assertEqual(study.arms[0].label, "base")

    def test_the_arms_pin_the_kv_ownership_vector(self):
        """Only --rank-mlp-ratio may move. Without the pin the weight split
        and the token split move together and neither is attributable."""
        import pathlib

        from sglang.srt.planner.runner import load_study

        root = pathlib.Path("tools/rig_dashboard/studies")
        if not root.is_dir():
            self.skipTest("study files not in this checkout")
        for t in STUDY_TIERS:
            with self.subTest(tier=t.key):
                study = load_study(str(root / t.study_file))
                self.assertTrue(study.policy.pin_token_vector)

    def test_the_prefill_points_bypass_the_radix_cache(self):
        """Unique random input ids per request. A prefill number taken over a
        warm cache measures the cache: the same prompt twice went 5505 ->
        1319 ms in the campaign that established this."""
        import json as _json
        import pathlib

        root = pathlib.Path("tools/rig_dashboard/studies")
        if not root.is_dir():
            self.skipTest("study files not in this checkout")
        for t in STUDY_TIERS:
            with self.subTest(tier=t.key):
                spec = _json.loads((root / t.study_file).read_text())
                self.assertEqual(spec["point"]["phase"], "random-ids")
                self.assertEqual(spec["point"]["output_tokens"], 1)


if __name__ == "__main__":
    unittest.main()
