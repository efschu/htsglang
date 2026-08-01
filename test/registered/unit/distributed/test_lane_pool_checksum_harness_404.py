"""#404: the checksum reader and the STEPS=3 bracket's arm contracts.

Hermetic. No server, no card: ``pool_checksum_diff`` is arithmetic over
records and the bracket's expectations are arithmetic over a result row, so
both are tested as such -- a change to either fails here rather than in the
one boot the window gets.

The pair this file guards is easy to get backwards. The append-only reading
lives inside ONE job and needs no reference; the cross-job reading joins on
``committed_len`` and must never join ``map``, which hashes physical slot ids
that two correct jobs draw differently. A reader that compared ``map`` across
jobs would report a leak on every clean run, and the window would have spent a
boot on it.
"""

import importlib.util
import os
import sys
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_R404 = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "..",
    "scripts",
    "dual_group",
    "r404",
)


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_R404, f"{name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


diff = _load("pool_checksum_diff")


def _chain(n, prompt=10, kept=2, tag="a", plant=None):
    """A clean record chain, optionally with a planted break at ``plant``.

    Content digests depend on the committed length alone, so two chains with
    different tags agree on ``kv`` / ``conv`` / ``ssm`` and disagree on
    ``map`` -- the mock of the exact asymmetry the reader has to respect.
    """
    out = []
    committed = prompt
    prev = 0
    prev_map = prev_kv = None
    for round_index in range(n):
        if round_index:
            committed += kept
        rec = {
            "tag": tag,
            "round": round_index,
            "path": "prefill" if round_index == 0 else "spec",
            "n_accept": None if round_index == 0 else kept - 1,
            "rung": 0 if round_index == 0 else kept - 1,
            "committed_len": committed,
            "prev_committed_len": prev,
            "region": "committed_prefix",
            "map": f"map-{tag}-{committed}",
            "map_stable": prev_map,
            "kv": f"kv-{committed}",
            "kv_stable": prev_kv,
            "conv": f"conv-{committed}",
            "ssm": f"ssm-{committed}",
        }
        if plant is not None and round_index == plant:
            rec["kv_stable"] = "kv-PLANTED"
            rec["kv"] = f"kv-PLANTED-{committed}"
        prev = committed
        prev_map, prev_kv = rec["map"], rec["kv"]
        out.append(rec)
    return out


class TestTheAppendOnlyReading(CustomTestCase):
    def test_a_clean_chain_has_no_violations(self):
        self.assertEqual(diff.append_only_violations(_chain(8)), [])

    def test_a_planted_break_is_named_by_round_and_surface(self):
        found = diff.append_only_violations(_chain(8, plant=3))
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["round"], 3)
        self.assertEqual(found[0]["surface"], "kv")

    def test_a_record_without_a_stable_digest_is_skipped_not_failed(self):
        """The freed-tail can-fail arm withholds the stable digests.

        Skipped is the honest answer: that arm cannot see an append-only break,
        and reporting it as clean would be the reader lying on the probe's
        behalf.
        """
        recs = _chain(6)
        for rec in recs:
            rec["kv_stable"] = None
            rec["map_stable"] = None
            rec["region"] = "freed_tail"
        self.assertEqual(diff.append_only_violations(recs), [])
        report = diff.analyse(recs)
        self.assertEqual(report["region"], ["freed_tail"])

    def test_a_gap_in_the_prefix_bookkeeping_is_skipped(self):
        recs = _chain(6)
        recs[3]["prev_committed_len"] = 999
        self.assertEqual(diff.append_only_violations(recs), [])


class TestTheCrossJobReading(CustomTestCase):
    def test_two_clean_jobs_with_different_slots_agree(self):
        ref = _chain(8, tag="ref", kept=1)
        spec = _chain(5, tag="spec", kept=2)
        out = diff.cross_job_differences(ref, spec)
        self.assertEqual(out["differences"], [])
        self.assertGreater(out["compared_positions"], 2)

    def test_map_is_never_joined_across_jobs(self):
        """The guard that stops a clean run reading as a leak."""
        self.assertNotIn("map", diff.CROSS_JOB_SURFACES)
        ref = _chain(8, tag="ref", kept=1)
        spec = _chain(5, tag="spec", kept=2)
        self.assertNotEqual(ref[1]["map"], spec[1]["map"])
        self.assertEqual(diff.cross_job_differences(ref, spec)["differences"], [])

    def test_a_planted_difference_names_the_surface_and_the_position(self):
        ref = _chain(8, tag="ref", kept=1)
        spec = _chain(5, tag="spec", kept=2, plant=2)
        out = diff.cross_job_differences(ref, spec)
        self.assertTrue(out["differences"])
        first = out["differences"][0]
        self.assertEqual(first["surface"], "kv")
        self.assertEqual(first["spec_round"], 2)

    def test_per_position_digests_narrow_a_kv_difference_to_a_token(self):
        ref = _chain(3, tag="ref", kept=1)
        spec = _chain(3, tag="spec", kept=1)
        for rec, other in zip(ref, spec):
            rec["kv_pos"] = ["a", "b", "c", "d"]
            other["kv_pos"] = ["a", "b", "c", "d"]
        spec[2]["kv"] = "moved"
        spec[2]["kv_pos"] = ["a", "b", "X", "d"]
        first = diff.cross_job_differences(ref, spec)["differences"][0]
        self.assertEqual(first["positions"], [2])
        self.assertEqual(first["n_positions"], 1)


class TestTheVerdict(CustomTestCase):
    def test_clean_is_clean_on_both_readings(self):
        report = diff.analyse(_chain(6, tag="spec"), _chain(9, tag="ref", kept=1))
        self.assertTrue(report["clean"])
        self.assertIn("clean", report["verdict"])

    def test_a_break_makes_the_verdict_name_the_round(self):
        report = diff.analyse(
            _chain(6, tag="spec", plant=4), _chain(9, tag="ref", kept=1)
        )
        self.assertFalse(report["clean"])
        self.assertIn("round 4", report["verdict"])

    def test_the_reading_works_without_a_reference_at_all(self):
        report = diff.analyse(_chain(6, tag="spec", plant=4))
        self.assertFalse(report["clean"])
        self.assertNotIn("cross_job", report)


def _numeric_chain(n, prompt=10, kept=2, tag="a", draw=0, noise=0.0, plant=None):
    """A chain that also carries the NUMERIC cross-job fields.

    ``draw`` and ``noise`` reproduce the defect the 2026-08-02 window measured
    on two NO-SPEC reference jobs of the same prompt: every digest different
    (the per-position sets were disjoint), every number within a last mantissa
    bit of each other, and the emitted tokens identical. That is the pair the
    cross-job reading has to call clean, and before this change it called it
    dirty on 100 % of positions.

    ``plant`` is the other side of the gate: a real content difference at one
    logical position, which has to stay visible through the tolerance.
    """
    out = []
    committed = prompt
    prev = 0
    prev_map = prev_kv = None
    for round_index in range(n):
        if round_index:
            committed += kept
        values = [float(100 + p) for p in range(committed)]
        values = [v * (1.0 + noise) for v in values]
        if plant is not None and round_index >= plant:
            values[3] = 999.0
        rec = {
            "tag": tag,
            "round": round_index,
            "path": "prefill" if round_index == 0 else "spec",
            "n_accept": None if round_index == 0 else kept - 1,
            "rung": 0 if round_index == 0 else kept - 1,
            "committed_len": committed,
            "prev_committed_len": prev,
            "region": "committed_prefix",
            # Every digest is draw-specific: bytes do not join across jobs on a
            # stack whose forwards are not bitwise reproducible, and pretending
            # otherwise is exactly the bug being fixed.
            "map": f"map-{tag}-{committed}",
            "map_stable": prev_map,
            "kv": f"kv-{draw}-{committed}",
            "kv_stable": prev_kv,
            "conv": f"conv-{draw}-{committed}",
            "ssm": f"ssm-{draw}-{committed}",
            "kv_num": [[v, v] for v in values],
            "conv_num": [sum(values), max(values)],
            "ssm_num": [sum(values) / 2.0, max(values)],
        }
        prev = committed
        prev_map, prev_kv = rec["map"], rec["kv"]
        out.append(rec)
    return out


class TestTheCrossJobReadingHasAMeasuredFloor(CustomTestCase):
    """The round-2 fix: the reading that fired on 15 of 15 arms.

    Including ``spec_steps_0``, whose speculative side takes zero draft steps,
    at round 0, which is the prefill. Two reference draws read the same way
    against each other. A reading whose control arm is 100 % red has no
    resolution, and the fix is not a bigger hammer -- it is measuring the floor
    and comparing against it.
    """

    def _draws(self, noise=1e-6):
        a = _numeric_chain(6, tag="ref0", draw=0, noise=0.0)
        b = _numeric_chain(6, tag="ref1", draw=1, noise=noise)
        return a, b

    def test_pre_fix_the_byte_reading_calls_two_clean_draws_dirty(self):
        """Reproduced, not asserted from the report: the pair IS byte-dirty."""
        a, b = self._draws()
        byte = diff.cross_job_differences(a, b)
        self.assertEqual(len(byte["differences"]), 3 * byte["compared_positions"])

    def test_the_floor_says_the_byte_reading_is_unusable_here(self):
        a, b = self._draws()
        floor = diff.noise_floor([a, b])
        self.assertTrue(floor["measured"])
        self.assertFalse(floor["byte_reading_usable"])
        self.assertLess(floor["numeric_max_deviation"], 1e-3)

    def test_a_plain_versus_plain_pair_reads_clean_after_the_fix(self):
        """The can-fail gate of the fix, in the direction that was broken."""
        a, b = self._draws()
        control = _numeric_chain(6, tag="ref2", draw=2, noise=2e-6)
        report = diff.analyse(b, a, [control])
        self.assertEqual(report["resolution"], "numeric")
        self.assertTrue(report["clean"], report["verdict"])
        self.assertEqual(report["cross_job_numeric"]["differences"], [])
        # ... and the byte reading is still recorded, still red, and no longer
        # allowed to decide. Silence about it would hide the floor.
        self.assertTrue(report["cross_job"]["differences"])

    def test_a_planted_logical_content_difference_is_still_dirty(self):
        a, b = self._draws()
        control = _numeric_chain(6, tag="ref2", draw=2, noise=2e-6)
        planted = _numeric_chain(6, tag="spec", draw=3, noise=1e-6, plant=2)
        report = diff.analyse(planted, a, [control])
        self.assertEqual(report["resolution"], "numeric")
        self.assertFalse(report["clean"])
        first = report["first_cross_job_numeric"]
        self.assertEqual(first["surface"], "kv_num")
        self.assertEqual(first["spec_round"], 2)
        self.assertEqual(first["positions"], [3])

    def test_the_planted_position_is_named_and_not_merely_counted(self):
        a, b = self._draws()
        control = _numeric_chain(6, tag="ref2", draw=2, noise=2e-6)
        planted = _numeric_chain(6, tag="spec", draw=3, noise=1e-6, plant=1)
        report = diff.analyse(planted, a, [control])
        kv = [
            d
            for d in report["cross_job_numeric"]["differences"]
            if d["surface"] == "kv_num"
        ]
        self.assertTrue(kv)
        self.assertTrue(all(d["positions"] == [3] for d in kv))

    def test_without_a_control_the_verdict_names_its_own_unmeasured_floor(self):
        a, b = self._draws()
        report = diff.analyse(b, a)
        self.assertEqual(report["resolution"], "unmeasured")
        self.assertIn("UNMEASURED", report["verdict"])

    def test_a_control_pair_that_is_byte_clean_keeps_the_byte_reading(self):
        """An exact stack loses nothing: the floor is zero and bytes decide."""
        a = _numeric_chain(6, tag="ref0", draw=0)
        b = _numeric_chain(6, tag="ref0", draw=0)
        spec = _numeric_chain(6, tag="ref0", draw=0)
        report = diff.analyse(spec, a, [b])
        self.assertTrue(report["noise_floor"]["byte_reading_usable"])
        self.assertEqual(report["noise_floor"]["numeric_max_deviation"], 0.0)
        self.assertTrue(report["clean"])

    def test_no_numeric_fields_and_a_dirty_control_is_reported_as_no_resolution(self):
        """The state the 2026-08-02 records are actually in."""
        a = _chain(6, tag="ref0")
        b = _chain(6, tag="ref1")
        for rec in b:
            rec["kv"] = rec["kv"] + "-drawn-again"
        report = diff.analyse(_chain(6, tag="spec"), a, [b])
        self.assertEqual(report["resolution"], "none")
        self.assertIn("NO resolution", report["verdict"])


class TestTheReferenceSet(CustomTestCase):
    """--ref-draws: a divergence is a trajectory outside the reference SET.

    The GDN bimodality lesson. One draw is a sample of one, and a spec arm that
    lands on the reference's other mode was being reported as a defect of the
    arm.
    """

    def setUp(self):
        self.bracket = _load("bracket_arms")

    def test_a_trajectory_matching_the_second_draw_is_within_modes(self):
        a = [1, 2, 3, 4]
        b = [1, 2, 9, 4]
        out = self.bracket.compare_against_reference_set([a, b], list(b))
        self.assertEqual(out["classification"], "identical")
        self.assertEqual(out["matched_ref_draw"], 1)
        self.assertEqual(out["ref_modes"], 2)
        self.assertTrue(out["ref_bimodal"])

    def test_can_fail_the_single_draw_reading_calls_the_same_run_divergent(self):
        a = [1, 2, 3, 4]
        b = [1, 2, 9, 4]
        out = self.bracket.compare_against_reference_set([a, b], list(b))
        self.assertEqual(out["single_draw_classification"], "content_divergence")
        self.assertTrue(out["reading_changed_by_the_set"])

    def test_a_trajectory_outside_every_draw_is_still_a_divergence(self):
        a = [1, 2, 3, 4]
        b = [1, 2, 9, 4]
        out = self.bracket.compare_against_reference_set([a, b], [1, 2, 7, 4])
        self.assertEqual(out["classification"], "content_divergence")
        self.assertFalse(out["reading_changed_by_the_set"])

    def test_a_reproducible_reference_reports_one_mode(self):
        a = [1, 2, 3, 4]
        out = self.bracket.compare_against_reference_set([a, list(a)], list(a))
        self.assertEqual(out["ref_modes"], 1)
        self.assertFalse(out["ref_bimodal"])
        self.assertEqual(out["classification"], "identical")

    def test_the_least_severe_match_wins_over_a_longer_one(self):
        a = [1, 2, 3]
        b = [1, 2, 3, 4]
        out = self.bracket.compare_against_reference_set([a, b], [1, 2, 3, 4])
        self.assertEqual(out["classification"], "identical")
        self.assertEqual(out["per_draw_classification"][0], "length_end_only")


class TestTheBracketArmContracts(CustomTestCase):
    """The expectations the previous window did not have and needed."""

    def setUp(self):
        # ``bracket_arms`` reaches for the r8/r12 helpers, which import
        # transformers lazily; importing the module itself must not.
        self.bracket = _load("bracket_arms")

    def test_the_k3_arms_expect_a_captured_verify(self):
        for name in ("k3_plain", "k3_tv0", "plain", "tv_max_accept_0"):
            self.assertIn("captured", self.bracket.ARM_REGISTRY[name]["expect"], name)

    def test_the_mixed_arms_expect_a_mixed_rung_job(self):
        for name in ("mixed_0_1", "mixed_0_3"):
            self.assertIn("mixed", self.bracket.ARM_REGISTRY[name]["expect"], name)
            self.assertIsInstance(
                self.bracket.ARM_REGISTRY[name]["spec_steps"], list, name
            )

    def test_the_adaptive_arms_drop_the_pin_and_carry_no_expectation(self):
        """What the policy decides is a measurement, not a contract."""
        for name in ("adaptive", "adaptive_tv0"):
            entry = self.bracket.ARM_REGISTRY[name]
            self.assertIsNone(entry["spec_steps"])
            self.assertTrue(entry["adaptive"])
            self.assertEqual(entry.get("expect"), None)

    def test_expect_is_not_sent_to_the_server_as_a_job_override(self):
        for name, entry in self.bracket.ARM_REGISTRY.items():
            self.assertNotIn("expect", self.bracket._overrides(entry), name)

    def test_an_eager_arm_fails_the_capture_expectation(self):
        row = {"spec_rounds": 40, "verify_graph_rounds": 0}
        checks = self.bracket._expectations({"expect": ["captured"]}, row)
        self.assertTrue(checks["failed"])
        self.assertIn("captured", checks["failed"][0])

    def test_a_fully_captured_arm_passes(self):
        row = {"spec_rounds": 40, "verify_graph_rounds": 40}
        self.assertEqual(
            self.bracket._expectations({"expect": ["captured"]}, row)["failed"], []
        )

    def test_a_partially_captured_arm_is_reported_and_still_fails(self):
        """Half a ladder is not the captured path, and silence would hide it."""
        row = {"spec_rounds": 40, "verify_graph_rounds": 21}
        checks = self.bracket._expectations({"expect": ["captured"]}, row)
        self.assertTrue(checks["capture"]["partial"])
        self.assertTrue(checks["failed"])

    def test_a_single_rung_job_fails_the_mixed_expectation(self):
        row = {"spec_rounds": 40, "decode_steps": 40, "rungs": {1: 40}}
        checks = self.bracket._expectations({"expect": ["mixed"]}, row)
        self.assertTrue(checks["failed"])
        self.assertFalse(checks["kv_len"]["mixed"])

    def test_a_zero_and_one_job_meets_the_mixed_expectation(self):
        row = {"spec_rounds": 20, "decode_steps": 40, "rungs": {0: 20, 1: 20}}
        checks = self.bracket._expectations({"expect": ["mixed"]}, row)
        self.assertEqual(checks["failed"], [])
        self.assertTrue(checks["kv_len"]["mixed"])
        self.assertEqual(checks["kv_len"]["k0_rounds"], 20)

    def test_an_all_k0_job_is_not_mixed(self):
        """The previous window's ``spec_steps_0`` arm: the WRITE side only."""
        row = {"spec_rounds": None, "decode_steps": 63, "rungs": {0: 63}}
        kv = self.bracket._kv_len_exercised(row)
        self.assertTrue(kv["exercised"])
        self.assertFalse(kv["mixed"])


class TestTheReferenceSetSmokeThroughMain(CustomTestCase):
    """``main()`` end to end against the bimodal fake server.

    Desk-written-never-executed is a standing risk label, so the reference-set
    gating is driven through the harness's own entry point rather than through
    its helper. The server is the shipped smoke vehicle with one further
    can-fail arm: under ``--dirty bimodal`` the NO-SPEC reference draws two
    different trajectories and the speculative arm lands on the second one.

    Both directions are asserted from ONE run of the vehicle:
    ``--ref-draws 2`` classifies the arm as within-modes, ``--ref-draws 1``
    classifies the same server's answer as a content divergence.
    """

    def setUp(self):
        import threading

        self.bracket = _load("bracket_arms")
        self.server_mod = _load("fake_lane_server")
        self.server_mod.set_dirty("bimodal")
        self.server = self.server_mod.make_server(0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        # The tokenizer is the one thing in this path that needs a model
        # directory, and it has nothing to do with what is under test.
        self.bracket.tokenize = lambda base, text, tok: list(range(10))
        self.bracket.detokenize = lambda ids, tok: " ".join(str(i) for i in ids)
        self.bracket.graded_score = lambda prompt, text: {"score": 0}

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.server_mod.set_dirty("none")

    def _run(self, ref_draws):
        import json as _json
        import tempfile

        self.server_mod.reset_state()
        with tempfile.NamedTemporaryFile("r", suffix=".json") as handle:
            rc = self.bracket.main(
                [
                    "--port",
                    str(self.port),
                    "--arms",
                    "plain",
                    "--tokens",
                    "64",
                    "--ref-draws",
                    str(ref_draws),
                    "--out",
                    handle.name,
                ]
            )
            self.assertEqual(rc, 0)
            return _json.load(open(handle.name))["arms"]["plain"]

    def test_the_set_reading_places_the_arm_within_the_reference_modes(self):
        arm = self._run(2)
        self.assertEqual(arm["classification"], "identical")
        self.assertEqual(arm["ref_modes"], 2)
        self.assertTrue(arm["ref_bimodal"])
        self.assertEqual(len(arm["no_spec_draws"]), 2)

    def test_can_fail_a_single_draw_calls_the_same_server_divergent(self):
        arm = self._run(1)
        self.assertEqual(arm["classification"], "content_divergence")
        self.assertEqual(arm["ref_modes"], 1)

    def test_the_single_draw_reading_is_kept_beside_the_set_reading(self):
        arm = self._run(2)
        self.assertEqual(arm["single_draw_classification"], "content_divergence")
        self.assertTrue(arm["reading_changed_by_the_set"])

    def test_a_bimodal_reference_leaves_the_checksum_reading_without_resolution(self):
        """And it says so, rather than reporting the floor as a finding.

        The two reference draws committed different tokens, so the floor they
        measure is O(1) -- larger than any leak could be. The reading refuses
        to decide instead of returning the green that a huge tolerance would
        have manufactured, and the arm lands in
        ``checksum_no_resolution_arms`` rather than in either verdict list.
        """
        arm = self._run(2)
        self.assertEqual(arm["pool_checksum"]["resolution"], "none")
        self.assertIn("NO resolution", arm["pool_checksum"]["verdict"])


if __name__ == "__main__":
    unittest.main()
