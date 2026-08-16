"""#602: read the per-rank fixed overhead and transient from the recorder.

THE LAST TWO TERMS THE #485 REFERENCE BENCH WAS STILL SUPPLYING. The draft
runner is calibrated (``draft_residency_from_flight``) and the seam is solved
as a fixed point; ``fixed_overhead_mib`` and ``transient_mib`` were still the
bench's numbers, measured on a different checkpoint.

FIXED OVERHEAD IS AN AT-REST QUANTITY and the boot marks determine it exactly:
everything resident at ``boot_complete`` that is not weights, not the KV pool
and not the draft runner -- CUDA context, the NCCL init, the attention
workspace, graph capture, the boot tail. On the reference rig 1610 / 1222 /
1198 MiB.

THE TRANSIENT IS NOT AN AT-REST QUANTITY AND THE RECORDER DOES NOT SETTLE IT.
What the serving marks measure is the draw over the window they cover. On the
2026-08-16 boot that is 742 / 440 / 584 MiB, while law 31's worst measured
state is 1989-3148 MiB on a 22-minute soak. A gate fed the optimistic number
admitted cuts that metal then broke the corridor on, twice. So this reader
returns the OBSERVED draw with its sample count and window, and refuses to call
it a worst-case: the caller charges ``max(observed, worst known)`` and the
bench value survives as a floor until a soak-length window replaces it.

MEASURED OR REFUSED (#606), same separation as the draft term: zero overhead
and zero transient are meaningful values, so absence raises instead of
producing them.
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

#: Measured, boot 3439375-1786877650, /spinning/flight_605.
LIVE_FIXED_OVERHEAD_MIB = (1610.0, 1222.0, 1198.0)
LIVE_OBSERVED_TRANSIENT_MIB = (742.0, 440.0, 584.0)
#: Law 31: the worst measured load state, from the #485 soak. Still the
#: governing number until a soak-length recorder window replaces it.
BENCH_WORST_TRANSIENT_MIB = (1346.0, 1120.0, 982.0)

LIVE_NET_DRAFT_MIB = (2906.0, 2250.0, 2250.0)
LIVE_BUDGET_MIB = (31800.0, 18800.0, 19800.0)
LIVE_SEAM_FIXED_MIB = (227.0, 138.0, 138.0)
LIVE_SEAM_SLOPE_B_PER_TOKEN = (2360.1, 424.1, 547.6)
LIVE_ARENA_TOKENS = 471638
LIVE_CUT = [28, 20, 16]


def _mark(pid, phase, resident_mib, *, draft=None, t=0.0):
    extra = {} if draft is None else {"draft_worker": draft}
    return {
        "pid": pid,
        "phase": phase,
        "nvml_self_bytes": int(resident_mib * MIB),
        "card_uuid": f"GPU-{pid}",
        "boot_id": "boot-1",
        "monotonic": t,
        "rank": 0,
        "extra": extra,
    }


def _write(directory, stem, marks):
    with open(
        os.path.join(directory, f"{stem}_rank0.jsonl"), "w", encoding="utf-8"
    ) as fh:
        for m in marks:
            fh.write(json.dumps(m) + "\n")


def _boot_marks(pid=100):
    return [
        _mark(pid, "process_start", 0, t=0),
        _mark(pid, "pre_weight_load", 500, draft=False, t=1),  # ctx+nccl 500
        _mark(pid, "weights_loaded", 10500, draft=False, t=2),  # weights 10000
        _mark(pid, "kv_pool_sized", 16000, draft=False, t=3),  # kv 5500
        _mark(pid, "pre_weight_load", 10000, draft=True, t=4),  # gap -6000
        _mark(pid, "weights_loaded", 18000, draft=True, t=5),  # draft 8000
        _mark(pid, "boot_complete", 18300, t=6),  # tail 300
    ]


def _serving_marks(pid=100, peak=19100):
    return [
        _mark(pid, "serving", 18400, t=10),
        _mark(pid, "serving", peak, t=11),
        _mark(pid, "serving", 18500, t=12),
    ]


class TheOverheadIsDifferencedFromTheBootMarks(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="resid602-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        _write(self.dir, "flight_marks", _boot_marks())
        _write(self.dir, "flight_serving", _serving_marks())

    def test_overhead_is_everything_not_weights_kv_or_draft(self):
        got = pp_cut.residency_terms_from_flight(self.dir)[100]
        # boot_complete 18300 - weights 10000 - kv 5500 - net draft 2000 = 800
        self.assertAlmostEqual(got.fixed_overhead_mib, 800.0, places=3)

    def test_the_observed_transient_is_the_peak_over_boot_complete(self):
        got = pp_cut.residency_terms_from_flight(self.dir)[100]
        self.assertAlmostEqual(got.observed_transient_mib, 800.0, places=3)

    def test_it_is_provenance_stamped_with_its_sample_count(self):
        got = pp_cut.residency_terms_from_flight(self.dir)[100]
        self.assertEqual(got.pid, 100)
        self.assertEqual(got.boot_id, "boot-1")
        self.assertEqual(got.card_uuid, "GPU-100")
        self.assertIn(self.dir, got.source)
        self.assertEqual(got.serving_samples, 3)

    def test_the_observed_transient_refuses_to_call_itself_a_worst_case(self):
        """The field name and the flag both have to say so, or a caller will
        charge a short window as if it were a soak."""
        got = pp_cut.residency_terms_from_flight(self.dir)[100]
        self.assertFalse(got.covers_worst_load_state)


class AbsenceRaisesHere_Too(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="resid602-")
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def test_no_marks_at_all_raises(self):
        with self.assertRaises(pp_cut.DraftResidencyUnavailable):
            pp_cut.residency_terms_from_flight(self.dir)

    def test_boot_marks_without_serving_marks_raise(self):
        """A boot with no serving window cannot report a transient, and zero
        is a real value ('nothing drawn'), so it must not be produced."""
        _write(self.dir, "flight_marks", _boot_marks())
        with self.assertRaises(pp_cut.DraftResidencyUnavailable):
            pp_cut.residency_terms_from_flight(self.dir)

    def test_a_boot_without_boot_complete_raises(self):
        marks = [m for m in _boot_marks() if m["phase"] != "boot_complete"]
        _write(self.dir, "flight_marks", marks)
        _write(self.dir, "flight_serving", _serving_marks())
        with self.assertRaises(pp_cut.DraftResidencyUnavailable):
            pp_cut.residency_terms_from_flight(self.dir)


def _live_inputs(*, transients=None, **over):
    """The live case with EVERY calibrated term in place."""
    charged = transients or tuple(
        max(o, b)
        for o, b in zip(LIVE_OBSERVED_TRANSIENT_MIB, BENCH_WORST_TRANSIENT_MIB)
    )
    kw = dict(
        budgets=LIVE_BUDGET_MIB,
        seam_staging=(1289.0, 329.0, 384.0),
        pool=LIVE_ARENA_TOKENS,
        corridor=1024.0,
        draft_residency=LIVE_NET_DRAFT_MIB,
        draft_runner_present=True,
        overheads=LIVE_FIXED_OVERHEAD_MIB,
        transients=charged,
    )
    kw.update(over)
    return ref._inputs(**kw)


class TheCalibrationAcceptanceHolds(unittest.TestCase):
    """Same acceptance shape as the draft term: the RUNNING configuration must
    still price as runnable once the measured overhead is charged."""

    def test_the_running_config_is_still_feasible(self):
        floor = pp_cut.world_kv_floor_at_seam_fixed_point(
            LIVE_CUT,
            _live_inputs(),
            seam_fixed_mib=LIVE_SEAM_FIXED_MIB,
            seam_slope_bytes_per_token=LIVE_SEAM_SLOPE_B_PER_TOKEN,
        )
        self.assertIsNotNone(
            floor,
            "charging the measured fixed overhead makes the running "
            "configuration infeasible, so the term is wrong",
        )
        self.assertGreater(floor, 0)

    def test_charging_the_overhead_lowers_the_floor(self):
        with_oh = pp_cut.world_kv_floor(LIVE_CUT, _live_inputs())
        without = pp_cut.world_kv_floor(
            LIVE_CUT, _live_inputs(overheads=(0.0, 0.0, 0.0))
        )
        self.assertLess(with_oh, without)

    def test_the_worst_load_state_is_charged_not_the_observed_one(self):
        """Law 31. The observed window is gentler than the known worst state,
        so the gate must not get cheaper by looking at a shorter window."""
        observed = pp_cut.world_kv_floor(
            LIVE_CUT, _live_inputs(transients=LIVE_OBSERVED_TRANSIENT_MIB)
        )
        charged = pp_cut.world_kv_floor(LIVE_CUT, _live_inputs())
        self.assertLess(charged, observed)


class TheCutUnderFullCalibration(unittest.TestCase):
    def _fp(self, **over):
        return pp_cut.solve_pp_cut_for_kv_floor_at_seam_fixed_point(
            _live_inputs(**over),
            seam_fixed_mib=LIVE_SEAM_FIXED_MIB,
            seam_slope_bytes_per_token=LIVE_SEAM_SLOPE_B_PER_TOKEN,
        )

    def test_the_solve_converges_and_is_feasible(self):
        got = self._fp()
        self.assertTrue(got.converged)
        self.assertTrue(got.solution.feasible, got.solution.refusals)

    def test_it_still_beats_the_shipping_cut_on_equal_terms(self):
        inc = pp_cut.world_kv_floor_at_seam_fixed_point(
            LIVE_CUT,
            _live_inputs(),
            seam_fixed_mib=LIVE_SEAM_FIXED_MIB,
            seam_slope_bytes_per_token=LIVE_SEAM_SLOPE_B_PER_TOKEN,
        )
        self.assertGreater(self._fp().solution.floor_tokens, inc)

    def test_every_layer_is_covered(self):
        self.assertEqual(sum(self._fp().solution.counts), 64)


if __name__ == "__main__":
    unittest.main()
