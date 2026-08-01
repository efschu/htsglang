"""The coherence gate: short byte-exact + long graded (#349).

The design constraint under test: NO byte-identity on long output. The byte
tier is short and exact; the graded tier scores against an A-vs-A floor and
treats text identity as framing only. The real #274 grader is exercised once
to prove the reuse wiring; the rest inject a deterministic grader so the tier
logic is provable without the scripts tree.
"""

import unittest

from sglang.srt.boot_matrix.coherence import (
    BYTE_TIER_MAX_CHARS,
    _find_grader,
    grade_probes,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _fake_grader(name, text):
    """A stand-in for the #274 grader: counts non-empty whitespace tokens, so
    a test can drive any score without the scripts tree."""
    toks = [t for t in (text or "").split() if t]
    return {"score": len(toks), "max_score": 8}


class TestByteTier(CustomTestCase):
    def test_leading_bytes_match_passes(self):
        probes = [{"name": "b", "tier": "byte", "text": "4\n5\n6\n", "ref_text": "4\n5"}]
        res = grade_probes(probes, grader=_fake_grader)
        self.assertTrue(res.passed)
        self.assertTrue(res.probes[0].ok)

    def test_leading_bytes_differ_fails(self):
        probes = [{"name": "b", "tier": "byte", "text": "9\n5\n", "ref_text": "4\n5"}]
        res = grade_probes(probes, grader=_fake_grader)
        self.assertFalse(res.passed)

    def test_a_byte_reference_past_the_window_is_a_probe_error(self):
        """Too long to byte-gate: the result flags it so the check renders STOP
        (mis-designed probe), never a false FAIL."""
        long_ref = "x" * (BYTE_TIER_MAX_CHARS + 1)
        probes = [{"name": "b", "tier": "byte", "text": long_ref, "ref_text": long_ref}]
        res = grade_probes(probes, grader=_fake_grader)
        self.assertTrue(res.byte_probe_too_long)
        self.assertFalse(res.passed)


class TestGradedTier(CustomTestCase):
    def test_score_at_or_above_floor_passes(self):
        probes = [
            {"name": "squares", "tier": "graded", "text": "a b c d", "min_score": 3}
        ]
        res = grade_probes(probes, grader=_fake_grader)
        self.assertTrue(res.passed)
        self.assertEqual(res.probes[0].score, 4)

    def test_score_below_floor_fails(self):
        probes = [
            {"name": "squares", "tier": "graded", "text": "a b", "min_score": 3}
        ]
        res = grade_probes(probes, grader=_fake_grader)
        self.assertFalse(res.passed)

    def test_text_identity_is_framing_not_criterion(self):
        """A trajectory that diverged in text but still scored above the floor
        PASSES; byte_identical is reported, never gated on -- the #360 rule."""
        probes = [
            {
                "name": "squares",
                "tier": "graded",
                "text": "a b c d",  # differs from ref
                "ref_text": "w x y z",
                "min_score": 3,
            }
        ]
        res = grade_probes(probes, grader=_fake_grader)
        self.assertTrue(res.passed)
        self.assertFalse(res.probes[0].byte_identical)


class TestGraderAvailability(CustomTestCase):
    def test_missing_grader_is_flagged_not_crashed(self):
        """When the locator finds no grader (e.g. a wheel with no scripts
        tree), the result is flagged unavailable, not crashed -- the check
        turns that into STOP, never a false FAIL."""
        from sglang.srt.boot_matrix import coherence as coh_mod

        saved = coh_mod._find_grader
        try:
            coh_mod._find_grader = lambda: None
            res = grade_probes([{"name": "x", "tier": "graded", "text": "a"}])
            self.assertFalse(res.grader_available)
            self.assertFalse(res.passed)
        finally:
            coh_mod._find_grader = saved

    def test_the_real_grader_is_locatable_and_scores(self):
        """Proves the reuse wiring against the actual checkout file: the #274
        grader must be found and score the alphabet continuation exactly."""
        grader = _find_grader()
        self.assertIsNotNone(
            grader, "the #274 grader must be reachable from a source checkout"
        )
        got = grader("alphabet", "w\nx\ny\nz")
        self.assertEqual(got["score"], 4)
        self.assertEqual(got["max_score"], 4)


if __name__ == "__main__":
    unittest.main()
