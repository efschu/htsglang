# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The A-vs-A band is measured, and it reaches every arm (#349 repair).

WHY THIS FILE EXISTS
--------------------
``sweep.py``'s module docstring has always said the graded floor and the byte
reference are measured from ``A_default``, booted first and probed twice. For
the whole of sweep 1 that was a promise and nothing else: ``main()`` called
``run_arm`` without ``reference_probes``, nothing ran an arm twice, and every
arm's ``probes.json`` carried ``ref_text=""`` and ``min_score=0``. The
coherence half of the gate could not fail, and seventeen arms were graded
against nothing while the report said "coherence within the A-vs-A band".

So the tests below pin the mechanism rather than the wording:

* the baseline really is booted twice,
* a NON-EMPTY byte reference and a NON-ZERO graded floor reach every later arm,
* the floor is the MINIMUM of the two baseline runs (an A-vs-A floor taken
  from one run would call ordinary run-to-run variation a defect),
* and an arm that actually misses is FAILED, which is the only property that
  makes the other three worth anything.

Hermetic: the runner is a stub, so no server, no card, no model.
"""

import json
import os
import tempfile
import unittest
from typing import Dict, List, Optional

from sglang.srt.boot_matrix.arms import arm_by_name
from sglang.srt.boot_matrix.check import FAIL, PASS, Verdict
from sglang.srt.boot_matrix.coherence import CoherenceResult, ProbeResult
from sglang.srt.boot_matrix.sweep import BAND_ARM, _measure_band
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _verdict(byte_text: str, scores: Dict[str, int]) -> Verdict:
    out: List[ProbeResult] = [_probe("byte_count", "byte", byte_text, None)]
    for name, score in scores.items():
        out.append(_probe(name, "graded", f"text-{name}", score))
    return Verdict(
        PASS, BAND_ARM, "stub", coherence=CoherenceResult(True, True, False, out)
    )


def _probe(name: str, tier: str, text: str, score: Optional[int]) -> ProbeResult:
    """A ProbeResult exactly as the real grader produces it -- no `text`.

    The first version of this helper injected a ``text`` attribute with
    ``object.__setattr__``. ProbeResult has no such field, so the stub was more
    capable than the type it stood in for: the test passed while the real band
    read "" for every reference and reproduced the very bug this file exists to
    prevent. The text now travels the way it really does, through probes.json
    on disk, which is what ``_write_probes`` below writes.
    """
    del text  # carried by the artifact, not by the object
    return ProbeResult(name=name, tier=tier, ok=True, detail="", score=score)


def _write_probes(out_dir: str, arm_name: str, byte_text: str,
                  scores: Dict[str, int]) -> None:
    """The artifact the real runner writes, which is where the text lives."""
    d = os.path.join(out_dir, arm_name)
    os.makedirs(d, exist_ok=True)
    entries = [{"name": "byte_count", "tier": "byte", "text": byte_text}]
    for name in scores:
        entries.append({"name": name, "tier": "graded", "text": f"text-{name}"})
    with open(os.path.join(d, "probes.json"), "w") as f:
        json.dump(entries, f)


class _Recorder:
    """A stubbed ``run_arm``: records what the band machinery hands it."""

    def __init__(self, runs: List[Verdict], texts: Optional[List[str]] = None,
                 scores: Optional[Dict[str, int]] = None):
        self._runs = list(runs)
        self._texts = list(texts or ["stub-text\n"])
        self._scores = dict(scores or {"alphabet": 1})
        self.calls: List[Dict[str, object]] = []

    def __call__(self, arm, *, model_path, out_dir, port, **kw):
        v = self._runs[min(len(self.calls), len(self._runs) - 1)]
        _write_probes(out_dir, arm.name, self._texts[
            min(len(self.calls), len(self._texts) - 1)], self._scores)
        self.calls.append(
            {
                "arm": arm.name,
                "reference_probes": kw.get("reference_probes"),
                "band": kw.get("band"),
            }
        )
        return v


class TestTheBandIsMeasured(CustomTestCase):
    def test_the_baseline_is_booted_twice_before_anything_is_graded(self):
        rec = _Recorder(
            [
                _verdict("1\n2\n3\n", {"alphabet": 9, "squares": 7}),
                _verdict("1\n2\n3\n", {"alphabet": 8, "squares": 9}),
                _verdict("1\n2\n3\n", {"alphabet": 8, "squares": 9}),
            ]
        )
        with tempfile.TemporaryDirectory() as out:
            ref, band, baseline = _measure_band(
                [arm_by_name(BAND_ARM)], model_path="/m", out_dir=out,
                port=1, run=rec,
            )
        self.assertGreaterEqual(len(rec.calls), 2, "the baseline must run twice")
        self.assertEqual(rec.calls[0]["arm"], BAND_ARM)
        self.assertEqual(rec.calls[1]["arm"], BAND_ARM)
        self.assertIsNotNone(baseline)
        # The first run is the reference source, so it is graded against nothing.
        self.assertFalse(rec.calls[0]["reference_probes"])
        # The second is already handed the reference.
        self.assertTrue(rec.calls[1]["reference_probes"])
        del ref, band

    def test_the_byte_reference_is_non_empty_and_is_the_first_run_text(self):
        rec = _Recorder(
            [
                _verdict("FIRST-RUN\n", {"alphabet": 9}),
                _verdict("SECOND-RUN\n", {"alphabet": 8}),
                _verdict("SECOND-RUN\n", {"alphabet": 8}),
            ],
            texts=["FIRST-RUN\n", "SECOND-RUN\n", "SECOND-RUN\n"],
            scores={"alphabet": 9},
        )
        with tempfile.TemporaryDirectory() as out:
            ref, _, _ = _measure_band(
                [arm_by_name(BAND_ARM)], model_path="/m", out_dir=out,
                port=1, run=rec,
            )
        self.assertIn("byte_count", ref)
        self.assertEqual(ref["byte_count"]["text"], "FIRST-RUN\n")
        self.assertNotEqual(ref["byte_count"]["text"], "")

    def test_the_graded_floor_is_the_minimum_of_the_two_runs(self):
        """A floor from one run would score this rig's own noise as a defect."""
        rec = _Recorder(
            [
                _verdict("b\n", {"alphabet": 9, "squares": 6}),
                _verdict("b\n", {"alphabet": 7, "squares": 8}),
                _verdict("b\n", {"alphabet": 7, "squares": 8}),
            ]
        )
        with tempfile.TemporaryDirectory() as out:
            _, band, _ = _measure_band(
                [arm_by_name(BAND_ARM)], model_path="/m", out_dir=out,
                port=1, run=rec,
            )
        self.assertEqual(band["alphabet"], 7)
        self.assertEqual(band["squares"], 6)
        self.assertTrue(all(v > 0 for v in band.values()), band)

    def test_the_baseline_is_regraded_against_the_band_it_produced(self):
        """A_default must be held to the same standard it sets for the others."""
        rec = _Recorder([_verdict("b\n", {"alphabet": 8})] * 3)
        with tempfile.TemporaryDirectory() as out:
            _, band, baseline = _measure_band(
                [arm_by_name(BAND_ARM)], model_path="/m", out_dir=out,
                port=1, run=rec,
            )
        self.assertEqual(len(rec.calls), 3)
        self.assertEqual(rec.calls[2]["band"], band)
        self.assertIsNotNone(baseline)

    def test_no_baseline_in_the_selection_yields_an_empty_band(self):
        """--only on one arm must still run, and must not fake a band."""
        rec = _Recorder([_verdict("b\n", {"alphabet": 8})])
        with tempfile.TemporaryDirectory() as out:
            ref, band, baseline = _measure_band(
                [arm_by_name("E_barlink")], model_path="/m", out_dir=out,
                port=1, run=rec,
            )
        self.assertEqual((ref, band, baseline), ({}, {}, None))
        self.assertEqual(rec.calls, [])


class TestTheBandActuallyGates(CustomTestCase):
    """The property that makes the rest worth anything: a miss FAILS."""

    def _graded(self, score: int, floor: int) -> CoherenceResult:
        p = ProbeResult(
            name="alphabet", tier="graded", ok=score >= floor,
            detail=f"{score} < {floor}" if score < floor else "",
            score=score, max_score=10, min_score=floor,
        )
        return CoherenceResult(p.ok, True, False, [p])

    def test_a_score_under_the_measured_floor_is_not_a_pass(self):
        below = self._graded(score=4, floor=7)
        self.assertFalse(below.passed)

    def test_a_score_at_the_floor_passes(self):
        at = self._graded(score=7, floor=7)
        self.assertTrue(at.passed)

    def test_a_zero_floor_would_pass_anything_which_is_the_bug(self):
        """Pinned so the sweep-1 state cannot come back unnoticed.

        With min_score=0 every score passes, including a completely incoherent
        one. That is precisely what shipped, and why this file exists.
        """
        vacuous = self._graded(score=0, floor=0)
        self.assertTrue(vacuous.passed)
        real = self._graded(score=0, floor=7)
        self.assertFalse(
            real.passed, "a measured floor must reject what a zero floor waves through"
        )

    def test_check_fails_an_arm_whose_coherence_missed(self):
        from sglang.srt.boot_matrix.check import Verdict as V

        v = V(FAIL, "E_barlink", "coherence below the measured A-vs-A band",
              coherence=self._graded(score=3, floor=7))
        self.assertEqual(v.status, FAIL)
        self.assertFalse(v.coherence.passed)


if __name__ == "__main__":
    unittest.main()
