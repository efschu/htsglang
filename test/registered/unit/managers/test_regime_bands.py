"""#363 gate 3 -- the A-vs-A band script.

Hermetic. The script's own ``--smoke`` runs here too, so a gate tool cannot
rot between the desk pass and the window; what is pinned directly is the
arithmetic a reader would otherwise have to trust: the band value, the
alignment of ragged spans, and the three verdicts that are NOT "clears".

The design decision this file exists to protect: a band is only a noise floor
if the two arms were doing the same thing. Three different ways of not being
one are separated -- too few samples (UNDERPOWERED), arms that do not line up
(ARMS_DISSIMILAR), and a threshold the signal never approaches (UNREACHED) --
because each needs a different fix and collapsing them into "failed" loses
the instruction.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

_SCRIPTS = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "scripts", "regime_gates"
    )
)
_PY_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "python")
)
sys.path.insert(0, _SCRIPTS)

import bands  # noqa: E402


def _write(path, rows):
    with open(path, "w") as f:
        f.write(json.dumps({"kind": "header", "mode": "observe", "rank": 0}) + "\n")
        for i, r in enumerate(rows):
            rec = {"kind": "verdict", "rank": 0, "round": (i + 1) * 8}
            rec.update(r)
            f.write(json.dumps(rec) + "\n")


def _pair(tmp, rows_a, rows_b):
    a, b = os.path.join(tmp, "a.jsonl"), os.path.join(tmp, "b.jsonl")
    _write(a, rows_a)
    _write(b, rows_b)
    return a, b


class TestBoundaryCounting(CustomTestCase):
    def test_a_three_rank_file_counts_each_boundary_once(self):
        """Otherwise a 3-rank run looks like three times the samples and every
        band shrinks by the square root of a lie."""
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "t.jsonl")
            with open(p, "w") as f:
                for rnd in (8, 16, 24):
                    for rank in (0, 1, 2):
                        f.write(
                            json.dumps(
                                {
                                    "kind": "verdict",
                                    "rank": rank,
                                    "round": rnd,
                                    "occupancy": 0.1,
                                }
                            )
                            + "\n"
                        )
            self.assertEqual(len(bands.load_boundaries(p)), 3)

    def test_a_rankless_file_is_de_duplicated_on_the_round(self):
        """The 2026-08-01 traces predate the rank stamp and must still read."""
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "t.jsonl")
            with open(p, "w") as f:
                for rnd in (8, 16):
                    for _ in range(3):
                        f.write(
                            json.dumps(
                                {"kind": "verdict", "round": rnd, "occupancy": 0.1}
                            )
                            + "\n"
                        )
            self.assertEqual(len(bands.load_boundaries(p)), 2)


class TestAlignment(CustomTestCase):
    def test_absent_samples_are_dropped_not_filled(self):
        """An idle window carries no share. A zero there is a different claim
        and would drag the band toward the full range."""
        rows = [{"prefill_share": None}, {"prefill_share": 0.4}]
        with tempfile.TemporaryDirectory() as tmp:
            a, b = _pair(tmp, rows, rows)
            self.assertEqual(
                bands.series(bands.load_boundaries(a), "prefill_share"), [0.4]
            )

    def test_ragged_spans_are_resampled_to_the_shorter(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = _pair(
                tmp,
                [{"decode_share": 0.5} for _ in range(30)],
                [{"decode_share": 0.5} for _ in range(9)],
            )
            rep = bands.report(a, b)
            band = rep["bands"]["decode_share"]
            self.assertEqual(band["paired"], 9)
            self.assertAlmostEqual(band["span_ratio"], 30 / 9, places=3)

    def test_resampling_picks_real_samples_never_interpolates(self):
        """An interpolated value is one the run never produced, and the band
        would then be partly a property of the interpolation."""
        got = bands.resample([0.0, 1.0], 3)
        self.assertTrue(all(v in (0.0, 1.0) for v in got), got)


#: One idle boundary as the observer writes it: no shares, and a REAL zero for
#: occupancy and the queue mass. That asymmetry is the whole of #388 item B.
_IDLE = {
    "prefill_share": None,
    "decode_share": None,
    "occupancy": 0.0,
    "queued_prompt_tokens": 0,
}


def _busy(occ, queued=1000):
    return {
        "prefill_share": 0.0,
        "decode_share": 1.0,
        "occupancy": occ,
        "queued_prompt_tokens": queued,
    }


class TestActiveOnlySignals(CustomTestCase):
    """#388 item B. Two signals are present on EVERY boundary, so per-signal
    absent-dropping does not drop idle stretches for them.

    Gate 3 (2026-08-01) measured the consequence: two arms whose idle lengths
    differed by 25 % (19 402 against 15 504 boundaries) had a quiet stretch of
    one aligned against a busy stretch of the other, and both signals came
    back ARMS_DISSIMILAR. The guard was right; the alignment was not.
    """

    def test_the_idle_boundary_that_causes_it_is_real(self):
        """The premise, checked rather than asserted in prose: an idle
        boundary carries no share and a genuine 0.0 occupancy."""
        self.assertFalse(bands.is_active(_IDLE))
        self.assertTrue(bands.is_active(_busy(0.5)))
        self.assertIsNone(_IDLE["prefill_share"])
        self.assertIsNotNone(_IDLE["occupancy"])
        self.assertEqual(_IDLE["occupancy"], 0.0)

    def test_dropping_absent_samples_does_not_drop_idle_for_these_two(self):
        """The defect itself, in one assertion pair. Same boundaries, and the
        share loses the idle ones while occupancy keeps them."""
        rows = [_IDLE] * 10 + [_busy(0.16)] * 3
        with tempfile.TemporaryDirectory() as tmp:
            a, _ = _pair(tmp, rows, rows)
            got = bands.load_boundaries(a)
            self.assertEqual(len(bands.series(got, "decode_share")), 3)
            self.assertEqual(len(bands.series(got, "occupancy")), 13)
            self.assertEqual(len(bands.series(got, "occupancy", active_only=True)), 3)

    def test_the_two_named_signals_are_restricted_and_the_shares_need_not_be(self):
        self.assertEqual(
            bands.ACTIVE_ONLY_SIGNALS, {"occupancy", "queued_prompt_tokens"}
        )
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_IDLE] * 10 + [_busy(0.16)] * 10
            a, b = _pair(tmp, rows, rows)
            rep = bands.report(a, b)
            self.assertTrue(rep["bands"]["occupancy"]["active_only"])
            self.assertTrue(rep["bands"]["queued_prompt_tokens"]["active_only"])
            self.assertFalse(rep["bands"]["decode_share"]["active_only"])
            self.assertFalse(rep["bands"]["rank_ms_spread_pct"]["active_only"])

    def test_arms_with_different_idle_lengths_are_comparable_again(self):
        """THE FALSIFIER, and it can fail: the same two arms, same identical
        working stretch, differing only in how long they idled first. Under
        the old rule the pairing puts arm A's idle against arm B's workload
        and the band is the signal's whole range; under the new one the two
        workloads line up and the band is the real (here: zero) difference.
        """
        work = [_busy(0.10), _busy(0.13), _busy(0.16)] * 4
        rows_a = [_IDLE] * 200 + work
        rows_b = [_IDLE] * 20 + work
        with tempfile.TemporaryDirectory() as tmp:
            a, b = _pair(tmp, rows_a, rows_b)
            loaded_a, loaded_b = bands.load_boundaries(a), bands.load_boundaries(b)

            # The unrestricted alignment, which is what gate 3 ran.
            old = bands.band_for(loaded_a, loaded_b, "occupancy", active_only=False)
            self.assertEqual(old["status"], "ARMS_DISSIMILAR")
            self.assertGreaterEqual(old["band"], 0.9 * 0.16)

            new = bands.report(a, b)["bands"]["occupancy"]
            self.assertEqual(new["status"], "OK")
            self.assertEqual(new["band"], 0.0)
            self.assertEqual(new["paired"], len(work))

    def test_a_restricted_signal_can_still_be_underpowered(self):
        """Restricting is not free: an almost-idle run has few active
        boundaries, and a band from a handful of them is a number rather than
        a measurement. UNDERPOWERED must therefore block -- the runsheet's
        table says only CLEARS is a pass."""
        # Occupancy above the ascend mark, so the constant is REACHED and the
        # verdict is about the sample count rather than about reachability.
        rows = [_IDLE] * 100 + [_busy(0.9)] * 3
        with tempfile.TemporaryDirectory() as tmp:
            a, b = _pair(tmp, rows, rows)
            rep = bands.report(a, b)
            self.assertEqual(rep["bands"]["occupancy"]["status"], "UNDERPOWERED")
            self.assertFalse(rep["passed"])
            self.assertIn("kv_ascend_mark: UNDERPOWERED", rep["blocking"])

    def test_the_report_says_which_signals_were_restricted(self):
        """A restriction the analysis applied has to be visible in the
        analysis's own output, or the next reader compares two reports that
        measured different things."""
        with tempfile.TemporaryDirectory() as tmp:
            rows = [_IDLE] * 5 + [_busy(0.1 + 0.01 * i) for i in range(10)]
            a, b = _pair(tmp, rows, rows)
            rendered = bands.render(bands.report(a, b))
            self.assertIn("occupancy*", rendered)
            self.assertIn("restricted to boundaries that ran a forward", rendered)
            self.assertIn("(10 active)", rendered)


class TestBandArithmetic(CustomTestCase):
    def test_a_known_offset_is_the_band(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = _pair(
                tmp,
                [{"decode_share": 0.50} for _ in range(20)],
                [{"decode_share": 0.53} for _ in range(20)],
            )
            rep = bands.report(a, b)
            self.assertAlmostEqual(rep["bands"]["decode_share"]["band"], 0.03)

    def test_it_uses_the_controllers_own_band_function(self):
        """Experiment and runtime must not drift: if the runtime's definition
        of a band changes, this report changes with it."""
        import inspect

        from sglang.srt.managers import regime_classifier

        src = inspect.getsource(bands)
        self.assertIn("signal_band", src)
        self.assertIn("clears_band", src)
        self.assertIs(bands.signal_band, regime_classifier.signal_band)
        self.assertIs(bands.clears_band, regime_classifier.clears_band)


class TestVerdictsThatAreNotClears(CustomTestCase):
    """Three ways to not be a noise floor, kept apart because the fixes are
    different."""

    def test_too_few_samples_is_underpowered_not_a_band(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = _pair(
                tmp,
                [{"decode_share": 0.5 + 0.01 * i} for i in range(4)],
                [{"decode_share": 0.5 + 0.01 * i} for i in range(4)],
            )
            rep = bands.report(a, b)
            self.assertEqual(rep["bands"]["decode_share"]["status"], "UNDERPOWERED")

    def test_arms_that_do_not_line_up_are_flagged_not_averaged(self):
        """A band as large as one arm's own movement is a comparability
        failure, not a floor. Found on the real trace: two halves of ONE run,
        idle first and workload second, produced a band equal to the whole
        occupancy range.

        Every boundary here carries a share, i.e. is ACTIVE: since #388
        occupancy is measured on active boundaries only, and the guard has to
        keep firing on arms that differ WHILE BOTH ARE WORKING -- which is the
        case it was written for and the one restriction cannot fix.
        """
        with tempfile.TemporaryDirectory() as tmp:
            a, b = _pair(
                tmp,
                [{"occupancy": 0.0, "decode_share": 1.0} for _ in range(20)],
                [{"occupancy": i / 20.0, "decode_share": 1.0} for i in range(20)],
            )
            rep = bands.report(a, b)
            self.assertEqual(rep["bands"]["occupancy"]["status"], "ARMS_DISSIMILAR")
            self.assertIn("within a single arm", rep["bands"]["occupancy"]["why"])

    def test_two_steady_arms_at_different_levels_are_a_real_band(self):
        """The control on the guard above: a reproducible bias IS the
        measurement, and calling it an alignment failure would throw it away."""
        with tempfile.TemporaryDirectory() as tmp:
            a, b = _pair(
                tmp,
                [{"decode_share": 0.50} for _ in range(20)],
                [{"decode_share": 0.53} for _ in range(20)],
            )
            rep = bands.report(a, b)
            self.assertEqual(rep["bands"]["decode_share"]["status"], "OK")

    def test_a_threshold_the_signal_never_approaches_is_unreached(self):
        """The finding the re-run handed us: prefill_share peaked at 0.000
        across 34 954 boundaries, so enter_prefill=0.35 gates a regime that
        cannot be entered. UNREACHED, not 'clears'."""
        with tempfile.TemporaryDirectory() as tmp:
            a, b = _pair(
                tmp,
                [{"prefill_share": 0.01, "decode_share": 0.99} for _ in range(20)],
                [{"prefill_share": 0.01, "decode_share": 0.99} for _ in range(20)],
            )
            rep = bands.report(a, b)
            v = next(c for c in rep["constants"] if c["constant"] == "enter_prefill")
            self.assertEqual(v["verdict"], "UNREACHED")
            self.assertIn("does not choose between those", v["why"])
            self.assertFalse(rep["passed"])

    def test_unreached_outranks_the_band_check(self):
        """A dead threshold that happens to clear its band is still dead, and
        reporting 'clears' for it would be true and useless."""
        with tempfile.TemporaryDirectory() as tmp:
            a, b = _pair(
                tmp,
                [{"prefill_share": 0.001 * i} for i in range(20)],
                [{"prefill_share": 0.001 * i} for i in range(20)],
            )
            rep = bands.report(a, b)
            v = next(c for c in rep["constants"] if c["constant"] == "enter_prefill")
            self.assertEqual(v["verdict"], "UNREACHED")


class TestReportContract(CustomTestCase):
    def test_every_section_34_constant_is_judged(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = _pair(
                tmp,
                [{"decode_share": 0.5} for _ in range(20)],
                [{"decode_share": 0.5} for _ in range(20)],
            )
            rep = bands.report(a, b)
            named = {c["constant"] for c in rep["constants"]}
            self.assertEqual(named, {c.name for c in bands.CONSTANTS})
            self.assertIn("enter_prefill", named)
            self.assertIn("spread_veto_pct", named)

    def test_the_inherited_kv_marks_carry_their_do_not_re_derive_note(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = _pair(tmp, [{"occupancy": 0.5}] * 20, [{"occupancy": 0.5}] * 20)
            rep = bands.report(a, b)
            v = next(c for c in rep["constants"] if c["constant"] == "kv_ascend_mark")
            self.assertIn("INHERITED from #287", v["note"])
            self.assertIn("not a licence", v["note"])

    def test_the_margin_is_two_bands(self):
        self.assertEqual(bands.THRESHOLD_MARGIN, 2.0)

    def test_one_file_as_both_arms_is_refused(self):
        """A band measured against itself is zero and every threshold would
        clear it for free."""
        with tempfile.TemporaryDirectory() as tmp:
            a, _b = _pair(
                tmp, [{"decode_share": 0.5}] * 20, [{"decode_share": 0.5}] * 20
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    os.path.join(_SCRIPTS, "bands.py"),
                    "--arm-a",
                    a,
                    "--arm-b",
                    a,
                ],
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": _PY_ROOT},
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("same file", proc.stderr)

    def test_a_failing_report_writes_no_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = _pair(
                tmp,
                [{"prefill_share": 0.01, "decode_share": 0.99}] * 20,
                [{"prefill_share": 0.01, "decode_share": 0.99}] * 20,
            )
            ev = os.path.join(tmp, "gate.json")
            proc = subprocess.run(
                [
                    sys.executable,
                    os.path.join(_SCRIPTS, "bands.py"),
                    "--arm-a",
                    a,
                    "--arm-b",
                    b,
                    "--evidence",
                    ev,
                ],
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": _PY_ROOT},
            )
            self.assertEqual(proc.returncode, 1)
            self.assertIn("GATE 3 NOT PASSED", proc.stdout)
            self.assertFalse(os.path.exists(ev))


class TestSmokeRuns(CustomTestCase):
    def test_the_scripts_own_smoke_passes(self):
        proc = subprocess.run(
            [sys.executable, os.path.join(_SCRIPTS, "bands.py"), "--smoke"],
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": _PY_ROOT},
        )
        self.assertEqual(proc.returncode, 0, proc.stdout[-1500:] + proc.stderr[-500:])
        self.assertIn("SMOKE OK", proc.stdout)


if __name__ == "__main__":
    unittest.main()
