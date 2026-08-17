"""#602: the draft runner's residency, measured, with its overlap credit.

WHY THIS TERM EXISTS. The KV-floor solver's first cut priced weights, KV, a
transient and a seam, and had NO term for the NEXTN / draft runner at all. On
the live boot that runner holds 18226 MiB on rank 0 and ~10.8 GiB on each 3080
-- first-order, not a rounding error. Feeding those raw made the model declare
the CURRENTLY RUNNING configuration infeasible, which is the loudest possible
proof that raw draft weights are the wrong number.

THE LOAD-BEARING PART IS THE NON-ADDITIVITY. The two runners do not simply
add. The flight recorder measures an ``inter_runner_gap`` of -15320 / -8546 /
-8550 MiB: between sizing the target pool and loading the draft weights the
process RELEASES that much and builds again into the same space. So the draft
runner's true cost is

    net = weights_draft - overlap_credit

which is 2906 / 2250 / 2250 MiB on the reference rig -- an order of magnitude
below the raw weights, and the difference between a model that says the running
config cannot run and one that says it can.

MEASURED OR REFUSED, NEVER DEFAULTED (the #606 getattr lesson). A missing
recorder post is not zero. Zero is a real, meaningful value here -- it means
"this deployment has no draft runner" -- so silently substituting it for "the
posts were not found" would let an uncalibrated solve look calibrated, which is
exactly the failure this term was added to fix. Absence raises; only an
explicit declaration that no draft runner exists is allowed to mean zero.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

from sglang.srt.planner import pp_cut
from sglang.test.ci.ci_register import register_cpu_ci

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_pp_family_cut_485 as ref  # noqa: E402

register_cpu_ci(est_time=15)

MIB = 1 << 20

#: Measured on boot 3439375-1786877650, /spinning/flight_605.
LIVE_WEIGHTS_DRAFT_MIB = (18226.0, 10796.0, 10800.0)
LIVE_OVERLAP_CREDIT_MIB = (15320.0, 8546.0, 8550.0)
LIVE_NET_DRAFT_MIB = (2906.0, 2250.0, 2250.0)

LIVE_BUDGET_MIB = (31800.0, 18800.0, 19800.0)
LIVE_SEAM_STAGING_MIB = (1289.0, 329.0, 384.0)
LIVE_ARENA_TOKENS = 471638
LIVE_CUT = [28, 20, 16]


def _mark(pid, phase, resident_mib, *, draft=None, boot="boot-1", t=0.0):
    extra = {} if draft is None else {"draft_worker": draft}
    return {
        "pid": pid,
        "phase": phase,
        "nvml_self_bytes": int(resident_mib * MIB),
        "card_uuid": f"GPU-{pid}",
        "boot_id": boot,
        "monotonic": t,
        "rank": 0,
        "extra": extra,
    }


def _write_marks(directory, marks):
    path = os.path.join(directory, "flight_marks_rank0.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for m in marks:
            fh.write(json.dumps(m) + "\n")
    return path


def _one_pid_boot(pid=100, *, with_draft=True):
    """A minimal but REAL boot shape: target loads, pool sized, the process
    releases, the draft loads."""
    seq = [
        _mark(pid, "process_start", 0, t=0),
        _mark(pid, "pre_weight_load", 500, draft=False, t=1),
        _mark(pid, "weights_loaded", 10500, draft=False, t=2),
        _mark(pid, "kv_pool_sized", 16000, draft=False, t=3),
    ]
    if with_draft:
        seq += [
            # the release: -6000 MiB, an inter_runner_gap
            _mark(pid, "pre_weight_load", 10000, draft=True, t=4),
            # the draft weights: +8000 MiB
            _mark(pid, "weights_loaded", 18000, draft=True, t=5),
        ]
    seq.append(_mark(pid, "boot_complete", 18000 if with_draft else 16000, t=6))
    return seq


class TheParserMeasuresBothPosts(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="flight602-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def test_weights_and_credit_are_read_separately(self):
        _write_marks(self.dir, _one_pid_boot())
        got = pp_cut.draft_residency_from_flight(self.dir)
        self.assertEqual(len(got), 1)
        res = got[100]
        self.assertAlmostEqual(res.weights_draft_mib, 8000.0, places=3)
        self.assertAlmostEqual(res.overlap_credit_mib, 6000.0, places=3)

    def test_the_net_is_weights_minus_credit(self):
        _write_marks(self.dir, _one_pid_boot())
        res = pp_cut.draft_residency_from_flight(self.dir)[100]
        self.assertAlmostEqual(res.net_mib, 2000.0, places=3)

    def test_the_reading_is_provenance_stamped(self):
        """A calibrated number with no provenance is indistinguishable from a
        guess six weeks later."""
        _write_marks(self.dir, _one_pid_boot())
        res = pp_cut.draft_residency_from_flight(self.dir)[100]
        self.assertEqual(res.pid, 100)
        self.assertEqual(res.boot_id, "boot-1")
        self.assertEqual(res.card_uuid, "GPU-100")
        self.assertIn(self.dir, res.source)


class AbsenceRaisesRatherThanDefaultingToZero(unittest.TestCase):
    """#606. Zero is a real value here ('no draft runner'), so it must never
    stand in for 'not measured'."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="flight602-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def test_a_boot_without_draft_posts_raises(self):
        _write_marks(self.dir, _one_pid_boot(with_draft=False))
        with self.assertRaises(pp_cut.DraftResidencyUnavailable):
            pp_cut.draft_residency_from_flight(self.dir)

    def test_an_empty_directory_raises(self):
        with self.assertRaises(pp_cut.DraftResidencyUnavailable):
            pp_cut.draft_residency_from_flight(self.dir)

    def test_declaring_a_draft_runner_without_supplying_it_refuses_the_solve(self):
        with self.assertRaises(ValueError):
            ref._inputs(
                budgets=LIVE_BUDGET_MIB,
                seam_staging=LIVE_SEAM_STAGING_MIB,
                pool=LIVE_ARENA_TOKENS,
                draft_runner_present=True,
            )

    def test_declaring_no_draft_runner_is_allowed_to_mean_zero(self):
        inputs = ref._inputs(
            budgets=LIVE_BUDGET_MIB,
            seam_staging=LIVE_SEAM_STAGING_MIB,
            pool=LIVE_ARENA_TOKENS,
        )
        self.assertIsNotNone(pp_cut.world_kv_floor(LIVE_CUT, inputs))


def _live_inputs(*, draft=LIVE_NET_DRAFT_MIB, **over):
    kw = dict(
        budgets=LIVE_BUDGET_MIB,
        seam_staging=LIVE_SEAM_STAGING_MIB,
        pool=LIVE_ARENA_TOKENS,
        corridor=1024.0,
        draft_residency=draft,
        draft_runner_present=True,
    )
    kw.update(over)
    return ref._inputs(**kw)


class TheCalibrationAcceptance(unittest.TestCase):
    """The whole point: with the credit the RUNNING config must price as
    runnable, and without it must not. If the live config prices infeasible,
    the term is still wrong -- that is the acceptance, stated as a test."""

    def test_the_running_config_is_feasible_with_the_overlap_credit(self):
        floor = pp_cut.world_kv_floor(LIVE_CUT, _live_inputs())
        self.assertIsNotNone(
            floor,
            "the model still calls the running configuration infeasible, so "
            "the draft term is not yet calibrated",
        )
        self.assertGreater(floor, 0)

    def test_the_raw_draft_weights_alone_price_the_running_config_infeasible(self):
        """The credit is load-bearing, demonstrated rather than asserted."""
        raw = pp_cut.world_kv_floor(
            LIVE_CUT, _live_inputs(draft=LIVE_WEIGHTS_DRAFT_MIB)
        )
        self.assertIsNone(
            raw,
            "charging the raw draft weights no longer breaks the model, so "
            "this test has stopped proving that the overlap credit matters",
        )

    def test_the_draft_term_lowers_the_floor_it_does_not_raise_it(self):
        with_draft = pp_cut.world_kv_floor(LIVE_CUT, _live_inputs())
        without = pp_cut.world_kv_floor(LIVE_CUT, _live_inputs(draft=(0.0, 0.0, 0.0)))
        self.assertLess(with_draft, without)


class TheCutIsReSolvedOnTheCalibratedInputs(unittest.TestCase):
    def test_the_solve_is_feasible_and_covers_every_layer(self):
        sol = pp_cut.solve_pp_cut_for_kv_floor(_live_inputs())
        self.assertTrue(sol.feasible, sol.refusals)
        self.assertEqual(sum(sol.counts), 64)

    def test_the_solved_cut_still_beats_the_shipping_cut(self):
        inputs = _live_inputs()
        incumbent = pp_cut.world_kv_floor(LIVE_CUT, inputs)
        sol = pp_cut.solve_pp_cut_for_kv_floor(inputs)
        self.assertGreater(sol.floor_tokens, incumbent)

    def test_the_dp_still_matches_a_full_enumeration(self):
        import itertools

        inputs = _live_inputs()
        best = None
        for cuts in itertools.combinations(range(1, 64), 2):
            b = list(cuts) + [64]
            counts = [b[0], b[1] - b[0], b[2] - b[1]]
            f = pp_cut.world_kv_floor(counts, inputs)
            if f is not None and (best is None or f > best):
                best = f
        sol = pp_cut.solve_pp_cut_for_kv_floor(inputs)
        self.assertAlmostEqual(sol.floor_tokens, best, places=6)


class TheSeamSlopeDoesNotFlipTheCut(unittest.TestCase):
    """The per-token seam scaling stays DOCUMENTED, not solved -- but only if
    it does not change the answer at the operating point. Rank 0's measured
    slope is 2360.1 B/token; charge it against the arena and re-solve."""

    def test_the_cut_is_stable_under_the_measured_seam_slope(self):
        base = pp_cut.solve_pp_cut_for_kv_floor(_live_inputs())
        slopes = (2360.1, 424.1, 547.6)
        fixed = (227.0, 138.0, 138.0)
        scaled = tuple(fx + s * LIVE_ARENA_TOKENS / MIB for fx, s in zip(fixed, slopes))
        moved = pp_cut.solve_pp_cut_for_kv_floor(_live_inputs(seam_staging=scaled))
        self.assertEqual(
            base.counts,
            moved.counts,
            "the seam's per-token term changes the chosen cut, so it can no "
            "longer be left documented-but-unsolved",
        )


if __name__ == "__main__":
    unittest.main()


LIVE_SEAM_FIXED_MIB = (227.0, 138.0, 138.0)
LIVE_SEAM_SLOPE_B_PER_TOKEN = (2360.1, 424.1, 547.6)


class TheSeamIsAFixedPointBecauseItFlipsTheCut(unittest.TestCase):
    """The sensitivity check came back POSITIVE, so the seam had to be solved.

    Pricing the seam at the live arena while solving for a larger one
    understates it on exactly the stage the solve wants to load. On this rig
    that changes the answer, which is the stated condition for promoting the
    seam from 'documented' to 'solved'.
    """

    def _fp(self, **over):
        return pp_cut.solve_pp_cut_for_kv_floor_at_seam_fixed_point(
            _live_inputs(**over),
            seam_fixed_mib=LIVE_SEAM_FIXED_MIB,
            seam_slope_bytes_per_token=LIVE_SEAM_SLOPE_B_PER_TOKEN,
        )

    def test_the_fixed_point_converges(self):
        got = self._fp()
        self.assertTrue(got.converged, f"did not converge in {got.iterations}")
        self.assertTrue(got.solution.feasible)

    def test_the_one_shot_solve_and_the_fixed_point_disagree(self):
        """If these ever agree, the seam term stopped mattering and this whole
        class should be revisited rather than left passing vacuously."""
        one_shot = pp_cut.solve_pp_cut_for_kv_floor(_live_inputs())
        fixed = self._fp().solution
        self.assertNotEqual(
            one_shot.counts,
            fixed.counts,
            "the seam's per-token term no longer changes the chosen cut",
        )

    def test_the_seam_charged_at_the_fixed_point_exceeds_the_live_one(self):
        got = self._fp()
        for used, base in zip(got.seam_staging_mib, LIVE_SEAM_STAGING_MIB):
            self.assertGreater(used, base)

    def test_the_incumbent_is_scored_at_its_own_fixed_point(self):
        """Comparing a fixed-point candidate against a one-shot incumbent
        would credit the candidate with the correction."""
        inputs = _live_inputs()
        inc = pp_cut.world_kv_floor_at_seam_fixed_point(
            LIVE_CUT,
            inputs,
            seam_fixed_mib=LIVE_SEAM_FIXED_MIB,
            seam_slope_bytes_per_token=LIVE_SEAM_SLOPE_B_PER_TOKEN,
        )
        self.assertIsNotNone(inc)
        one_shot = pp_cut.world_kv_floor(LIVE_CUT, inputs)
        self.assertLess(inc, one_shot)

    def test_the_solved_cut_still_wins_on_equal_terms(self):
        inputs = _live_inputs()
        inc = pp_cut.world_kv_floor_at_seam_fixed_point(
            LIVE_CUT,
            inputs,
            seam_fixed_mib=LIVE_SEAM_FIXED_MIB,
            seam_slope_bytes_per_token=LIVE_SEAM_SLOPE_B_PER_TOKEN,
        )
        fixed = self._fp().solution
        self.assertGreater(fixed.floor_tokens, inc)

    def test_a_mis_sized_seam_vector_is_refused(self):
        with self.assertRaises(ValueError):
            pp_cut.solve_pp_cut_for_kv_floor_at_seam_fixed_point(
                _live_inputs(),
                seam_fixed_mib=(227.0,),
                seam_slope_bytes_per_token=LIVE_SEAM_SLOPE_B_PER_TOKEN,
            )
