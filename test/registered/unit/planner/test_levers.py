"""CPU unit tests for the five levers."""

import time
import unittest

from sglang.srt.planner.crossover import (
    MEASURED,
    MEASURED_HERE,
    MODELLED,
    REFERENCE_FINDING,
    CrossoverFinding,
    RigDescriptor,
)
from sglang.srt.planner.levers import (
    LEVERS,
    RATES_KNOWN,
    STRUCTURE_ONLY,
    Confidence,
    Evidence,
    FlagSpec,
    Lever,
    flag_available,
    missing_flags,
    render_levers_text,
    suggest_levers,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PROBE = {
    "created": "2026-07-21 07:06:43",
    "gpus": {"GPU-a": {"gemm_tflops": 233.9, "membw_gbs": 1664.1}},
}

#: A finding that IS from this rig, so the prefill lever may act on it. The
#: slopes are the reference rig's, but the point of the fixture is the
#: provenance, not the values.
LOCAL = CrossoverFinding(
    rig=RigDescriptor(
        cards=("card A", "card B", "card B"),
        model="a-model",
        quant="fp8",
        tp_size=3,
        base_vector=(63, 37, 36),
    ),
    points=list(REFERENCE_FINDING.points),
    provenance=MEASURED_HERE,
    measured_at=time.time(),
    cache_bypass_proven=True,
)


def _lever(key, **kw):
    kw.setdefault("probe", PROBE)
    return [x for x in suggest_levers(keys=[key], **kw) if x.lever.key == key][0]


class TestLeverDefinitions(CustomTestCase):
    def test_the_five_directions_exist(self):
        self.assertEqual(
            set(LEVERS),
            {"context", "decode_speed", "prefill_speed", "ttft_loaded", "energy"},
        )

    def test_every_lever_states_a_price(self):
        """A lever that lists only what it improves is a sales pitch."""
        for key, p in LEVERS.items():
            self.assertTrue(p.gains, key)
            self.assertTrue(p.costs, f"{key} names no cost")
            self.assertTrue(p.maximises, key)
            self.assertTrue(p.vague_statement, f"{key} has no pre-probe statement")

    def test_opposing_levers_name_each_other(self):
        self.assertIn("decode_speed", LEVERS["context"].tradeoff_against)
        self.assertIn("context", LEVERS["decode_speed"].tradeoff_against)
        self.assertIn("decode_speed", LEVERS["energy"].tradeoff_against)

    def test_every_flag_states_why(self):
        for key, p in LEVERS.items():
            for f in p.flags:
                self.assertTrue(f.why, f"{key}: {f.flag} has no rationale")


class TestBuildGate(CustomTestCase):
    def test_flag_availability_is_read_from_server_args(self):
        self.assertTrue(flag_available("rank_tp_ratio"))
        self.assertFalse(flag_available("definitely_not_a_field"))

    def test_a_lever_whose_knob_is_absent_says_so(self):
        """A lever must never emit a flag the running build cannot parse.

        Exercised on a synthetic lever rather than on a real one: which flags
        exist is a property of the branch under test, so asserting a specific
        absence would make the test a claim about the branch instead of about
        the gate.
        """
        synthetic = Lever(
            key="synthetic",
            label="Synthetic",
            maximises="nothing",
            flags=[
                FlagSpec(
                    "--definitely-not-a-flag",
                    "1",
                    "definitely_not_a_field",
                    why="exists only to be missing",
                )
            ],
            gains=["none"],
            costs=["none"],
            vague_statement="none",
        )
        missing = missing_flags(synthetic)
        self.assertEqual([f.flag for f in missing], ["--definitely-not-a-flag"])

    def test_the_decode_lever_has_its_knob_on_this_build(self):
        """#210 shipped --rank-kv-ratio; the decode lever sits in the KV-token
        split, so on this build it must resolve to a real flag rather than to
        an unavailability notice."""
        self.assertTrue(flag_available("rank_kv_ratio"))
        self.assertEqual(missing_flags(LEVERS["decode_speed"]), [])
        s = [x for x in suggest_levers(probe=PROBE) if x.lever.key == "decode_speed"][0]
        self.assertEqual(s.unavailable_flags, [])
        self.assertIn("--rank-kv-ratio speed", " ".join(s.command_flags))
        # 'speed' reads the per-rank bandwidth scores, which only the
        # auto-performance plan resolves; without it the mode degrades to the
        # opposite direction, so the lever must carry it.
        self.assertIn("--rank-tp-ratio auto-performance", " ".join(s.command_flags))

    def test_the_context_lever_asks_for_capacity_ownership(self):
        s = [x for x in suggest_levers(probe=PROBE) if x.lever.key == "context"][0]
        self.assertIn("--rank-kv-ratio capacity", " ".join(s.command_flags))

    def test_available_flags_are_still_emitted(self):
        s = [x for x in suggest_levers(probe=PROBE) if x.lever.key == "context"][0]
        self.assertIn("--rank-tp-ratio auto", " ".join(s.command_flags))


class TestStaging(CustomTestCase):
    def test_without_a_probe_suggestions_are_vague(self):
        for s in suggest_levers(probe=None):
            if s.confidence == Confidence.BLOCKED:
                continue
            self.assertEqual(s.stage, STRUCTURE_ONLY)
            self.assertEqual(s.confidence, Confidence.VAGUE)
            self.assertEqual(s.statement, s.lever.vague_statement)

    def test_rate_dependent_flags_are_withheld_before_a_probe(self):
        s = [x for x in suggest_levers(probe=None) if x.lever.key == "prefill_speed"][0]
        self.assertNotIn("--rank-perf-tune", " ".join(s.command_flags))

    def test_with_a_probe_suggestions_become_concrete(self):
        s = [x for x in suggest_levers(probe=PROBE) if x.lever.key == "decode_speed"][0]
        self.assertEqual(s.stage, RATES_KNOWN)
        self.assertEqual(s.confidence, Confidence.DETAILED)
        self.assertIn("--rank-kv-ratio speed", " ".join(s.command_flags))
        self.assertIn("Price:", s.statement)

    def test_a_probe_alone_does_not_make_the_prefill_lever_concrete(self):
        """Card rates say which rank is compute-strong. They do not say what
        concentrating onto it costs in decode, and that term is half the
        decision."""
        s = _lever("prefill_speed", probe=PROBE)
        self.assertEqual(s.stage, RATES_KNOWN)
        self.assertEqual(s.confidence, Confidence.UNMEASURED)

    def test_fit_levers_do_not_need_a_probe_for_their_main_flag(self):
        """Fit questions are answerable without measurement; speed questions
        are not."""
        s = [x for x in suggest_levers(probe=None) if x.lever.key == "context"][0]
        self.assertIn("--rank-tp-ratio auto", " ".join(s.command_flags))


class TestHomogeneousRig(CustomTestCase):
    def test_heterogeneous_only_flags_are_dropped_on_a_uniform_rig(self):
        s = [
            x
            for x in suggest_levers(heterogeneous=False, probe=PROBE)
            if x.lever.key == "context"
        ][0]
        self.assertNotIn("--rank-tp-ratio", " ".join(s.command_flags))


class TestPreconditions(CustomTestCase):
    def test_ttft_lever_needs_a_second_node(self):
        s = [x for x in suggest_levers(probe=PROBE, node_count=1)
             if x.lever.key == "ttft_loaded"][0]
        self.assertTrue(s.unmet_preconditions)
        self.assertIn("second node", " ".join(s.unmet_preconditions))
        s2 = [x for x in suggest_levers(probe=PROBE, node_count=2)
              if x.lever.key == "ttft_loaded"][0]
        self.assertNotIn("second node", " ".join(s2.unmet_preconditions))

    def test_energy_lever_needs_power_control(self):
        s = [x for x in suggest_levers(probe=PROBE, facility_keys_available=[])
             if x.lever.key == "energy"][0]
        self.assertIn("Power-target", " ".join(s.unmet_preconditions))
        s2 = [
            x
            for x in suggest_levers(
                probe=PROBE, facility_keys_available=["power_target"]
            )
            if x.lever.key == "energy"
        ][0]
        self.assertEqual(s2.unmet_preconditions, [])


class TestRendering(CustomTestCase):
    def test_json_carries_gains_costs_and_counter_reckoning(self):
        d = suggest_levers(probe=PROBE)[0].to_json()
        for k in ("gains", "costs", "counter_reckoning", "statement", "confidence"):
            self.assertIn(k, d)

    def test_text_states_the_evidence_stage(self):
        txt = render_levers_text(suggest_levers(probe=None))
        self.assertIn("structure only", txt)
        txt2 = render_levers_text(suggest_levers(probe=PROBE))
        self.assertIn("measured rates", txt2)

    def test_text_shows_the_counter_reckoning(self):
        txt = render_levers_text(
            suggest_levers(probe=PROBE, keys=["context"])
        )
        self.assertIn("costs decode_speed", txt)

    def test_filtering_by_key(self):
        out = suggest_levers(probe=PROBE, keys=["energy"])
        self.assertEqual([s.lever.key for s in out], ["energy"])


class TestThePrefillLeverDoesNotGuess(CustomTestCase):
    """What the split costs and what it buys are both rig-specific.

    The lever used to hand out a concentration vector on the strength of a
    hardware probe alone. A probe measures cards; the crossover is a property
    of cards plus model plus quantisation path, and a vector proposed without
    it is a guess with a flag attached.
    """

    def test_without_a_local_crossover_no_vector_is_proposed(self):
        s = _lever("prefill_speed")
        self.assertEqual(s.confidence, Confidence.UNMEASURED)
        joined = " ".join(s.command_flags)
        self.assertNotIn("--rank-mlp-ratio", joined)
        self.assertNotIn("--rank-perf-tune", joined)

    def test_it_offers_the_measurement_instead_of_a_number(self):
        s = _lever("prefill_speed")
        offer = s.crossover.get("offer") or []
        self.assertEqual({t["key"] for t in offer}, {"quick", "thorough"})
        for t in offer:
            self.assertGreater(t["est_runtime_min"], 0, t["key"])

    def test_another_rigs_finding_does_not_unlock_it(self):
        s = _lever(
            "prefill_speed",
            crossover=REFERENCE_FINDING,
            prompt_to_output_ratio=100.0,
        )
        self.assertEqual(s.confidence, Confidence.UNMEASURED)
        self.assertNotIn("--rank-mlp-ratio", " ".join(s.command_flags))

    def test_a_workload_ratio_alone_does_not_unlock_it(self):
        s = _lever("prefill_speed", prompt_to_output_ratio=100.0)
        self.assertEqual(s.confidence, Confidence.UNMEASURED)
        self.assertNotIn("--rank-mlp-ratio", " ".join(s.command_flags))

    def test_a_local_finding_without_a_workload_shows_the_turn_and_stops(self):
        s = _lever("prefill_speed", crossover=LOCAL)
        self.assertNotIn("--rank-mlp-ratio", " ".join(s.command_flags))
        self.assertIn("13.7", s.statement)

    def test_below_the_turn_the_base_split_is_the_answer(self):
        s = _lever("prefill_speed", crossover=LOCAL, prompt_to_output_ratio=4.0)
        self.assertNotIn("--rank-mlp-ratio", " ".join(s.command_flags))
        self.assertIn("base split", s.statement.lower())

    def test_above_the_turn_it_names_the_measured_vector(self):
        s = _lever("prefill_speed", crossover=LOCAL, prompt_to_output_ratio=16.0)
        self.assertEqual(s.confidence, Confidence.DETAILED)
        self.assertIn("--rank-mlp-ratio 3,1,1", " ".join(s.command_flags))

    def test_the_deep_vector_needs_a_prompt_dominated_mix(self):
        s = _lever("prefill_speed", crossover=LOCAL, prompt_to_output_ratio=100.0)
        self.assertIn("--rank-mlp-ratio 6,1,1", " ".join(s.command_flags))

    def test_a_candidate_that_is_never_best_is_never_proposed(self):
        r = 1.0
        while r < 400.0:
            s = _lever("prefill_speed", crossover=LOCAL, prompt_to_output_ratio=r)
            self.assertNotIn(
                "--rank-mlp-ratio 4,1,1",
                " ".join(s.command_flags),
                f"proposed at {r}:1 although no ratio makes it the best choice",
            )
            r += 2.5

    def test_no_flag_is_emitted_without_the_vector_it_belongs_to(self):
        """A context budget for a concentration that is not being set is a
        flag that says nothing, and it made the lever look resolved."""
        for kw in (
            {},
            {"prompt_to_output_ratio": 100.0},
            {"crossover": LOCAL, "prompt_to_output_ratio": 2.0},
            {"probe": None, "crossover": LOCAL, "prompt_to_output_ratio": 100.0},
            {"heterogeneous": False, "crossover": LOCAL,
             "prompt_to_output_ratio": 100.0},
        ):
            with self.subTest(**kw):
                s = _lever("prefill_speed", **kw)
                if "--rank-mlp-ratio" not in " ".join(s.command_flags):
                    self.assertEqual(s.command_flags, [])

    def test_the_optimizer_is_not_asked_to_pick_the_vector(self):
        """--rank-perf-tune runs a decode-knee guard that rejects the very
        concentration a prompt-dominated workload justifies. Naming both would
        emit a command whose two halves disagree."""
        s = _lever("prefill_speed", crossover=LOCAL, prompt_to_output_ratio=100.0)
        self.assertNotIn("--rank-perf-tune", " ".join(s.command_flags))


class TestCounterReckoningCarriesNumbers(CustomTestCase):
    def test_the_prefill_lever_states_what_decode_pays(self):
        s = _lever("prefill_speed", crossover=LOCAL, prompt_to_output_ratio=100.0)
        against = s.counter_reckoning["decode_speed"]
        self.assertIn("%", against)
        self.assertIn("15.5", against)

    def test_without_a_measurement_it_says_the_size_is_unknown(self):
        s = _lever("prefill_speed")
        against = s.counter_reckoning["decode_speed"]
        self.assertNotIn("%", against.split(".")[0])
        self.assertIn("not measured", against.lower())

    def test_the_decode_lever_states_what_prefill_is_left_on_the_table(self):
        s = _lever("decode_speed", crossover=LOCAL)
        self.assertIn("prefill_speed", s.counter_reckoning)
        self.assertIn("%", s.counter_reckoning["prefill_speed"])

    def test_every_lever_still_names_at_least_one_opposing_direction(self):
        for s in suggest_levers(probe=PROBE, crossover=LOCAL):
            if s.confidence == Confidence.BLOCKED:
                continue
            self.assertTrue(s.counter_reckoning, s.lever.key)


class TestMeasuredAgainstModelled(CustomTestCase):
    def test_every_evidence_entry_declares_which_it_is(self):
        for s in suggest_levers(probe=PROBE, crossover=LOCAL):
            for e in s.evidence:
                self.assertIn(e.kind, (MEASURED, MODELLED), s.lever.key)

    def test_a_measured_entry_carries_its_setup(self):
        for s in suggest_levers(probe=PROBE, crossover=LOCAL):
            for e in s.evidence:
                if e.kind == MEASURED:
                    self.assertTrue(e.setup, f"{s.lever.key}: {e.statement}")

    def test_a_modelled_entry_carries_its_known_error(self):
        for s in suggest_levers(probe=PROBE, crossover=LOCAL):
            for e in s.evidence:
                if e.kind == MODELLED:
                    self.assertTrue(e.caveat, f"{s.lever.key}: {e.statement}")

    def test_the_prefill_model_bias_is_on_the_prefill_lever(self):
        s = _lever("prefill_speed", crossover=LOCAL, prompt_to_output_ratio=100.0)
        modelled = [e for e in s.evidence if e.kind == MODELLED]
        self.assertTrue(modelled)
        self.assertTrue(any("1.8" in e.caveat for e in modelled))

    def test_no_lever_prints_a_modelled_net(self):
        """The decode half of the cost model is fitted to measurement and the
        prefill half is not, so their difference has no bound."""
        for s in suggest_levers(probe=PROBE, crossover=LOCAL):
            for e in s.evidence:
                if e.kind != MODELLED:
                    continue
                self.assertNotIn("net", e.statement.lower(), s.lever.key)

    def test_the_rendered_text_labels_both_kinds(self):
        txt = render_levers_text(
            suggest_levers(
                probe=PROBE,
                crossover=LOCAL,
                prompt_to_output_ratio=100.0,
                keys=["prefill_speed"],
            )
        )
        self.assertIn("[measured]", txt)
        self.assertIn("[modelled]", txt)

    def test_a_borrowed_finding_is_rendered_as_another_rigs(self):
        txt = render_levers_text(
            suggest_levers(
                probe=PROBE, crossover=REFERENCE_FINDING, keys=["prefill_speed"]
            )
        )
        self.assertIn("another rig", txt.lower())

    def test_json_carries_the_evidence_and_the_crossover(self):
        d = _lever("prefill_speed", crossover=LOCAL).to_json()
        self.assertIn("evidence", d)
        self.assertIn("crossover", d)
        for e in d["evidence"]:
            self.assertIn(e["kind"], (MEASURED, MODELLED))


class TestEvidenceType(CustomTestCase):
    def test_an_unknown_kind_is_refused(self):
        with self.assertRaises(ValueError):
            Evidence(kind="a hunch", statement="x")


if __name__ == "__main__":
    unittest.main()
