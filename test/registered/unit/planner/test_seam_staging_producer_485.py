# SPDX-License-Identifier: Apache-2.0
"""#485 T1/T3: the gate stops funding the seam at zero, and the census can see it.

WHAT THIS FILE PINS, AND WHY IT EXISTS

``RankResources.seam_staging_mib`` has carried a ``0.0`` default since the
field was added, and NOTHING IN THE TREE SUPPLIED IT -- a grep for the name
outside ``pp_cut.py`` returned nothing. So ``runnable_headroom_mib ==
headroom_mib`` on every wired boot, and the field's own docstring says what
that makes the answer: *"left at zero the verdict is a residency verdict again
and says nothing about runnability."*

THE COST, MEASURED, NOT ARGUED. The gate admitted the #485 planner cut
``40,12,12`` reporting **374.9 MiB** of headroom. Pooled over the 196 tp->pp
flips of the two certification windows (s50 + s51), the seam it funded at zero
drew **5800 MiB** modally and **7055 MiB** at its worst on rank 0 -- and one
of those flips took the corridor to 668 MiB against a 1024 MiB law. The admit
was arithmetically unreachable, not unlucky. See
``docs/dev/485/EXCURSION_ANALYSIS_485.md``.

THE TWO HALVES, AND WHY THEY ARE ONE TICKET.

* **T1** is the producer: ``ServerArgs._pp_cut_seam_staging`` reads the
  measured seam demand out of the census and refuses rather than defaulting,
  exactly as ``_pp_cut_transients`` already does for the load-state table.
* **T3** is the reason there was nothing to read. ``transient_census.note()``
  is called from exactly one site -- ``Scheduler.process_batch_result`` --
  and labels its sample ``batch.forward_mode.name``. A cutover is not a
  batch, so the census the gate funds "the WORST load state" from could not
  take one sample inside the largest transient in the system. The cutover now
  feeds it under a per-direction seam load state.

Fixing T1 without T3 would give the producer nothing to produce from; fixing
T3 without T1 would measure the seam and still price it at zero.

NOT PINNED HERE: the gate's own arithmetic for the seam term. That already
has coverage in ``test_pp_family_cut_485.py`` with synthetic staging values.
What was missing is that no real boot ever put a real value into it, and that
is what this file is about.
"""

import sys
import types
import unittest
from pathlib import Path

from sglang.srt.managers import phase_flip_seam_census as seam_census
from sglang.srt.planner import pp_cut, transient_census
from sglang.srt.server_args import ServerArgs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pp_family_cut_485 import _inputs  # noqa: E402

#: The #485 planner cut, as admitted and as booted.
CUT = [40, 12, 12]

#: Fixed overheads chosen so the fixture reproduces the REAL admit's reported
#: headroom -- 374.9 MiB on the binding rank -- rather than the shared
#: fixture's roomier default. Pinning the refusal against a headroom the gate
#: never actually reported would prove nothing about the boot that broke.
OVERHEADS_AT_THE_REAL_ADMIT = (9371.1, 8000.0, 8000.0)

#: Measured on rank 0 over 196 tp_to_pp flips, s50 + s51.
SEAM_MODAL_MIB = 5800.0
SEAM_WORST_MIB = 7055.0


def _verdict(seam_mib, overheads=OVERHEADS_AT_THE_REAL_ADMIT):
    return pp_cut.validate_pp_cut(
        CUT,
        _inputs(
            pool=280000,
            overheads=overheads,
            seam_staging=(seam_mib, seam_mib, seam_mib),
        ),
    )


class TestTheAdmitThatMetalBroke(unittest.TestCase):
    """The headline. The gate refuses its own prior verdict once it is told."""

    def test_the_fixture_reproduces_the_reported_headroom(self):
        # Guard on the guard: if this drifts, every refusal below is being
        # measured against a configuration the gate never admitted, and the
        # file silently stops testing the thing it names.
        sol, _ = _verdict(0.0)
        self.assertTrue(sol.feasible)
        self.assertAlmostEqual(
            min(s.headroom_mib for s in sol.stages), 374.9, places=1
        )

    def test_zero_seam_is_what_admitted_it(self):
        # The can-fail counterpart: with the seam priced at zero -- the
        # shipped default -- the cut is ADMITTED. So the refusals below are
        # produced by the term, not by the fixture being tight.
        sol, violations = _verdict(0.0)
        self.assertTrue(sol.feasible, violations)

    def test_the_modal_seam_alone_refuses_it(self):
        sol, violations = _verdict(SEAM_MODAL_MIB)
        self.assertFalse(sol.feasible)
        self.assertTrue(violations)
        joined = " ".join(violations)
        # The refusal must say it FITS AT REST, or a reader goes hunting for
        # a residency problem that does not exist -- the cut really does fit.
        self.assertIn("FITS AT REST", joined)
        self.assertIn("seam staging", joined)

    def test_the_worst_seam_refuses_it_by_more(self):
        modal, _ = _verdict(SEAM_MODAL_MIB)
        worst, _ = _verdict(SEAM_WORST_MIB)
        self.assertFalse(worst.feasible)
        self.assertLess(
            min(s.runnable_headroom_mib for s in worst.stages),
            min(s.runnable_headroom_mib for s in modal.stages),
        )

    def test_the_shortfall_is_the_measured_one(self):
        # 374.9 of headroom against a 5800 MiB seam is short by 5425.1. The
        # number matters: it is the difference between "tune the cut" and
        # "this cut cannot host this seam at all".
        sol, _ = _verdict(SEAM_MODAL_MIB)
        self.assertAlmostEqual(
            min(s.runnable_headroom_mib for s in sol.stages), -5425.1, places=1
        )


class TestTheProducerReadsAMeasurementOrRefuses(unittest.TestCase):
    """``_pp_cut_seam_staging`` -- T1, exercised without a ServerArgs boot."""

    @staticmethod
    def _call(transients, pp_size=3, census_dir="/tmp/census"):
        # Unbound: the method reads nothing off self but ``pp_size``, and
        # building a real ServerArgs here would drag in argument parsing that
        # has nothing to do with the behaviour under test.
        stub = types.SimpleNamespace(pp_size=pp_size)
        return ServerArgs._pp_cut_seam_staging(stub, transients, census_dir)

    def test_it_takes_the_worst_seam_state(self):
        tp_to_pp = transient_census.seam_load_state("tp_to_pp")
        pp_to_tp = transient_census.seam_load_state("pp_to_tp")
        tables = [
            {"EXTEND": 1989.0, tp_to_pp: 5800.0, pp_to_tp: 0.0},
            {"EXTEND": 900.0, tp_to_pp: 12.0, pp_to_tp: 3.0},
            {"DECODE": 800.0, tp_to_pp: 7.0, pp_to_tp: 1.0},
        ]
        self.assertEqual(self._call(tables), (5800.0, 12.0, 7.0))

    def test_the_batch_load_states_do_not_leak_into_the_seam_term(self):
        # 1989 MiB is the census's worst BATCH state. Funding it as seam
        # staging would double-charge it -- it is already the transient term
        # -- and would still be 3811 MiB short of the real seam.
        tp_to_pp = transient_census.seam_load_state("tp_to_pp")
        tables = [{"EXTEND": 1989.0, tp_to_pp: 5800.0}] * 3
        self.assertEqual(self._call(tables), (5800.0, 5800.0, 5800.0))

    def test_a_census_with_no_seam_state_is_REFUSED_not_zeroed(self):
        # The whole defect in one assertion: silently pricing this at 0.0 is
        # what admitted the cut, and the failure would look identical while
        # the code looked calibrated.
        with self.assertRaises(ValueError) as ctx:
            self._call([{"EXTEND": 1989.0}, {"EXTEND": 900.0}, {"DECODE": 8.0}])
        msg = str(ctx.exception)
        self.assertIn("no measured SEAM staging", msg)
        # And it must name the fix, not merely the fault.
        self.assertIn("--enable-phase-flip", msg)
        self.assertIn("SGLANG_TRANSIENT_CENSUS", msg)

    def test_a_boot_that_never_flipped_is_refused_too(self):
        # An empty seam table is the signature of a boot that armed the flip
        # and never reached a cutover. It measures no seam, so it may not
        # certify one.
        with self.assertRaises(ValueError):
            self._call([{}, {}, {}])

    def test_one_rank_missing_the_seam_refuses_the_whole_census(self):
        tp_to_pp = transient_census.seam_load_state("tp_to_pp")
        with self.assertRaises(ValueError):
            self._call([{tp_to_pp: 5800.0}, {tp_to_pp: 12.0}, {"EXTEND": 8.0}])


class TestTheSeamLoadState(unittest.TestCase):
    """T3 -- the census can finally see the cutover."""

    def setUp(self):
        transient_census._CENSUS = None

    def tearDown(self):
        transient_census._CENSUS = None

    def test_the_state_is_per_direction(self):
        a = transient_census.seam_load_state("tp_to_pp")
        b = transient_census.seam_load_state("pp_to_tp")
        self.assertNotEqual(a, b)
        # Folding the legs together would hide WHICH leg owes the demand, and
        # on this rig only one of them has any: rank 0 measured a transient of
        # 0 on all 86 pp_to_tp flips of s50.
        for s in (a, b):
            self.assertTrue(s.startswith(transient_census.SEAM_LOAD_STATE_PREFIX))

    def test_note_free_records_the_value_it_is_given(self):
        c = transient_census.TransientCensus(0, "5090", 8100 << 20)
        transient_census._CENSUS = c
        state = transient_census.seam_load_state("tp_to_pp")
        transient_census.note_free(state, 668 << 20)
        self.assertEqual(c.min_free_bytes[state], 668 << 20)

    def test_note_free_is_NOT_strided(self):
        # A flip's trough is one event, not a stream. ``note`` samples one in
        # ``_stride()``; applying that here would drop the rare flip, which is
        # the only one that ever mattered -- s50 breached on 1 flip in 86.
        import os

        old = os.environ.get("SGLANG_TRANSIENT_CENSUS_STRIDE")
        os.environ["SGLANG_TRANSIENT_CENSUS_STRIDE"] = "1000"
        try:
            c = transient_census.TransientCensus(0, "5090", 8100 << 20)
            transient_census._CENSUS = c
            state = transient_census.seam_load_state("tp_to_pp")
            transient_census.note_free(state, 668 << 20)
            self.assertIn(state, c.min_free_bytes)
        finally:
            if old is None:
                os.environ.pop("SGLANG_TRANSIENT_CENSUS_STRIDE", None)
            else:
                os.environ["SGLANG_TRANSIENT_CENSUS_STRIDE"] = old

    def test_it_keeps_the_MINIMUM_across_flips(self):
        c = transient_census.TransientCensus(0, "5090", 8100 << 20)
        transient_census._CENSUS = c
        state = transient_census.seam_load_state("tp_to_pp")
        for free_mib in (1925, 668, 1356):
            transient_census.note_free(state, free_mib << 20)
        self.assertEqual(c.min_free_bytes[state], 668 << 20)

    def test_an_unarmed_census_is_a_no_op_not_an_error(self):
        # This is called from the cutover, which is the no-return region. An
        # instrument may not be the reason a flip dies.
        transient_census._CENSUS = None
        transient_census.note_free("SEAM_TP_TO_PP", 668 << 20)

    def test_a_census_that_raises_is_swallowed(self):
        class _Boom:
            def note(self, *a, **k):
                raise RuntimeError("nope")

        transient_census._CENSUS = _Boom()
        transient_census.note_free("SEAM_TP_TO_PP", 1)


class TestTheCensusLinePrintsAllocated(unittest.TestCase):
    """The M0/M1 precondition: the column that decides S1/S2/S3."""

    @staticmethod
    def _census(rows):
        c = seam_census.SeamCensus("tp_to_pp", 0, probe=lambda: None)
        c.stages = list(rows)
        return c

    def test_allocated_and_reserved_are_both_printed(self):
        mib = 1 << 20
        line = self._census(
            [("plan", 7725 * mib, 32040 * mib, 31634 * mib)]
        ).format_line()
        self.assertIn("alloc=31634", line)
        self.assertIn("res=32040", line)
        # slack stays, so every existing reader keeps working.
        self.assertIn("slack=406", line)

    def test_the_S3_signature_is_readable_off_the_line(self):
        # allocated FLAT while free drops => the bytes went somewhere outside
        # torch. This is the one clean split in the suspect set, and before
        # this change the line could not express it: slack is a DIFFERENCE and
        # would have been identical in both readings.
        mib = 1 << 20
        line = self._census(
            [
                ("gdn_state", 6203 * mib, 32040 * mib, 31634 * mib),
                ("weights_refill", 668 * mib, 32040 * mib, 31634 * mib),
            ]
        ).format_line()
        self.assertIn("weights_refill free=668 step-5535", line)
        self.assertEqual(line.count("alloc=31634"), 2)

    def test_the_torch_side_signature_differs_on_the_same_column(self):
        mib = 1 << 20
        line = self._census(
            [
                ("gdn_state", 6203 * mib, 32040 * mib, 30376 * mib),
                ("weights_refill", 668 * mib, 33298 * mib, 31634 * mib),
            ]
        ).format_line()
        self.assertIn("alloc=30376", line)
        self.assertIn("alloc=31634", line)

    def test_a_failed_probe_row_still_prints_nothing_misleading(self):
        line = self._census([("plan", -1, -1, -1)]).format_line()
        self.assertIn("plan=probe-failed", line)
        self.assertNotIn("alloc=-1", line)


if __name__ == "__main__":
    unittest.main()
