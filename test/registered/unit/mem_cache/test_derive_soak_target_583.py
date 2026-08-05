"""The soak harness must aim itself from a crash reference, not from a guess.

Two aiming errors already cost GPU windows on 2026-08-05, and the second is the
one this module exists to make impossible:

  arm 2      unbounded prefix diversity saturated mamba at 1.00 and died on
             allocation asserts before barlink was ever stressed.
  soak arm 1 ran fraction 0.17 against the reference's 0.25 and was rejected as
             "too light" -- but ABSOLUTE occupancy was identical (~16 slots).
             Only the denominator differed (96 vs 64). Acting on the fraction
             would have pushed real load ABOVE the reference while the number
             on screen moved toward it.

So: convert to absolute slots using the reference's OWN denominator, and target
absolutes. These tests pin that, and pin the traps that make it subtle.
"""

import sys
import unittest
from pathlib import Path


def _repo_root() -> Path:
    rel = Path("scripts/repro/derive_soak_target.py")
    for parent in Path(__file__).resolve().parents:
        if (parent / rel).is_file():
            return parent
    raise AssertionError(f"could not locate {rel}")


# Self-contained: the module under test lives in scripts/, not on the package
# path, and a registered test must not depend on the caller's PYTHONPATH.
sys.path.insert(0, str(_repo_root() / "scripts" / "repro"))

from derive_soak_target import (  # noqa: E402
    DEFAULT_SLOTS_PER_RUNNING_REQ,
    read_reference,
    solve,
)

RESULTS = Path("/spinning/spill-night-20260804/results")
BARLINK_REF = RESULTS / "CRASH_20260805_boot5_barlink_full.log"
MAMBA_REF = RESULTS / "CRASH_20260805_boot6_mamba_pingpong.log"


def _skip_unless(path: Path):
    if not path.is_file():
        raise unittest.SkipTest(f"reference log absent: {path}")


class TestAbsoluteConversion(unittest.TestCase):
    def test_uses_the_references_own_denominator(self):
        _skip_unless(BARLINK_REF)
        ref = read_reference(BARLINK_REF)
        self.assertEqual(ref.denominator, 64)
        # 0.25 * 64 = 16 slots. If this ever reads 96 (production's pool) the
        # whole derivation is rebuilt on the wrong denominator.
        self.assertEqual(ref.median_slots, 16)

    def test_peak_comes_from_prefill_lines_not_just_mamba_num(self):
        """`mamba num` is printed on DECODE lines only.

        In this reference that is 14 samples, all 16, while the regime's peak
        (51 slots) occurs on PREFILL lines carrying no `mamba num`. A deriver
        reading only `mamba num` reports a flat regime and misses the peak by
        more than 3x -- which is exactly how soak arm 1 came out dead flat.
        """
        _skip_unless(BARLINK_REF)
        ref = read_reference(BARLINK_REF)
        self.assertGreater(
            ref.peak_slots, 3 * ref.median_slots // 2,
            "peak collapsed toward the median: the deriver is probably "
            "reading decode-only samples",
        )
        self.assertEqual(ref.peak_slots, 51)

    def test_crosscheck_against_mamba_num_is_tight(self):
        """usage*denominator must agree with the printed absolute count."""
        _skip_unless(BARLINK_REF)
        ref = read_reference(BARLINK_REF)
        self.assertLessEqual(
            ref.crosscheck_error, 1.0,
            "reconstructed absolute slots disagree with the log's own "
            "'mamba num' by more than a slot -- the denominator or the "
            "rounding is wrong",
        )

    def test_full_sample_coverage(self):
        _skip_unless(BARLINK_REF)
        ref = read_reference(BARLINK_REF)
        self.assertGreater(len(ref.slots), 100)


class TestSolve(unittest.TestCase):
    def test_prefix_pool_is_solved_from_the_peak_not_the_median(self):
        _skip_unless(BARLINK_REF)
        ref = read_reference(BARLINK_REF)
        target = solve(ref)
        structural = max(ref.running) * DEFAULT_SLOTS_PER_RUNNING_REQ
        self.assertEqual(target.prefix_pool, ref.peak_slots - structural)
        # Solving on the median would give a pool far too small to ever reach
        # the reference's excursions.
        self.assertGreater(target.prefix_pool, ref.median_slots - structural)

    def test_mamba_cache_matches_the_reference_denominator(self):
        _skip_unless(BARLINK_REF)
        target = solve(read_reference(BARLINK_REF))
        self.assertEqual(target.mamba_cache, 64)

    def test_known_barlink_reference_regression_pin(self):
        """The aiming for the #583 soak, pinned."""
        _skip_unless(BARLINK_REF)
        target = solve(read_reference(BARLINK_REF))
        self.assertEqual(
            (target.mamba_cache, target.prefix_pool, target.sessions),
            (64, 31, 3),
        )

    def test_solved_pool_clears_the_581_floor(self):
        """A derived MAMBA_CACHE below the #581 hard floor would refuse to boot."""
        _skip_unless(BARLINK_REF)
        ref = read_reference(BARLINK_REF)
        target = solve(ref)
        floor = max(ref.running) * DEFAULT_SLOTS_PER_RUNNING_REQ
        self.assertGreaterEqual(target.mamba_cache, floor)


class TestReferenceSelectionGuards(unittest.TestCase):
    def test_saturating_reference_is_detected(self):
        """boot6 is the MAMBA crash and saturated its pool.

        Targeting it reproduces a saturating regime, which the harness's own
        gate scores REGIME FAIL. The deriver has to be able to say so, or the
        next operator aims at the most recent crash by reflex.
        """
        _skip_unless(MAMBA_REF)
        ref = read_reference(MAMBA_REF)
        self.assertGreaterEqual(
            ref.peak_slots, ref.denominator,
            "boot6 is expected to have saturated; if it no longer does, "
            "re-check which log this is",
        )

    def test_barlink_reference_does_not_saturate(self):
        _skip_unless(BARLINK_REF)
        ref = read_reference(BARLINK_REF)
        self.assertLess(ref.peak_slots, ref.denominator)

    def test_missing_denominator_is_a_hard_error_not_a_guess(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as fh:
            fh.write("mamba usage: 0.25\nmamba usage: 0.80\n")
            path = Path(fh.name)
        try:
            with self.assertRaises(SystemExit):
                read_reference(path)
        finally:
            path.unlink()

    def test_spikiness_is_reported(self):
        _skip_unless(BARLINK_REF)
        ref = read_reference(BARLINK_REF)
        self.assertAlmostEqual(ref.spikiness, 51 / 16, places=2)


if __name__ == "__main__":
    unittest.main()
