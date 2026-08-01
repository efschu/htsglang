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


if __name__ == "__main__":
    unittest.main()
