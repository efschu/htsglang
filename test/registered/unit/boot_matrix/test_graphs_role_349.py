# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""``graphs`` must be confirmed by the TARGET worker, not the draft one (#349).

THE DEFECT THIS PINS. ``report_effective`` resolved ``graphs`` with

    r"Capture (draft )?(decode|extend|prefill) CUDA graph"

while ``model_runner.py:3907`` builds the line as
``role = "draft" if self.is_draft_worker else "target"``. So the alternation
covered the DRAFT role and missed the TARGET role entirely, and it also missed
the ``verify`` phase. A real 2026-08-01 arm log carries exactly this:

    Capture draft decode CUDA graph begin      <- matched
    Capture draft extend CUDA graph begin      <- matched
    Capture target verify CUDA graph begin     <- matched NOTHING (x3)

Every boot arm runs with speculation, so the draft worker's line was always
present and ``graphs`` always resolved True -- on the strength of the DRAFT
model alone. The matrix has never once observed whether the TARGET model
captured its graphs. A boot where target capture silently fell back to eager
while draft capture succeeded reported ``graphs=True`` and went green.

That is a FALSE GREEN, which is worse than the STOP an always-absent marker
usually produces: ``BASE_EXPECT["graphs"] = True`` means "full CUDA graphs,
not eager" about the SERVED model, and the served model is the target.

Every boot has a target worker; a draft worker exists only under speculation.
So the target line is the honest evidence and the draft line is not evidence
about the target at all.
"""

from __future__ import annotations

import unittest

from sglang.srt.boot_matrix.effective import report_effective

_READY = "[2026-08-01 00:00:09 TP0] The server is fired up and ready to roll!"


def _log(*lines: str) -> str:
    return "\n".join(["[2026-08-01 00:00:00 TP0] start", *lines, _READY])


class TestTargetCaptureConfirmsGraphs(unittest.TestCase):
    def test_target_verify_is_recognised(self):
        # The exact line shape a real arm log carries, and the one the old
        # alternation could not match: neither "target" nor "verify" was in it.
        eff = report_effective(
            _log("[2026-08-01 00:00:05 TP0] Capture target verify CUDA graph begin.")
        )
        self.assertTrue(eff.graphs)

    def test_target_decode_and_prefill_are_recognised(self):
        for phase in ("decode", "extend", "prefill", "verify"):
            with self.subTest(phase=phase):
                eff = report_effective(
                    _log(f"[t TP0] Capture target {phase} CUDA graph begin.")
                )
                self.assertTrue(eff.graphs)

    def test_a_roleless_line_still_reads_as_target(self):
        # Pre-role-prefix logs had no role word. Those are target captures;
        # historical artifacts must stay readable.
        eff = report_effective(_log("[t TP0] Capture decode CUDA graph begin."))
        self.assertTrue(eff.graphs)


class TestDraftCaptureIsNotEvidenceAboutTheTarget(unittest.TestCase):
    def test_draft_only_does_not_confirm_graphs(self):
        # THE POINT. Draft graphs captured, target graphs not mentioned: the
        # honest answer is "unconfirmed", never True.
        eff = report_effective(
            _log(
                "[t TP0] Capture draft decode CUDA graph begin.",
                "[t TP0] Capture draft extend CUDA graph begin.",
            )
        )
        self.assertIsNone(eff.graphs)

    def test_draft_plus_target_confirms(self):
        eff = report_effective(
            _log(
                "[t TP0] Capture draft decode CUDA graph begin.",
                "[t TP0] Capture target verify CUDA graph begin.",
            )
        )
        self.assertTrue(eff.graphs)


class TestEagerStillReadsFalse(unittest.TestCase):
    def test_disable_notice_wins(self):
        eff = report_effective(_log("[t TP0] Disable cuda graph"))
        self.assertIs(eff.graphs, False)

    def test_silent_log_is_unconfirmed(self):
        self.assertIsNone(report_effective(_log()).graphs)


if __name__ == "__main__":
    unittest.main()
