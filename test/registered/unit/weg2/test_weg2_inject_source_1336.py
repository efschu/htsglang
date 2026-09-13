# SPDX-License-Identifier: Apache-2.0
"""#1336 -- one source for the inject mode, and a named absent state.

TWO HALVES OF ONE DEFECT, both measured on this rig:

(1) THE LAUNCHER AND THE RANKS READ DIFFERENT SOURCES. `prepare_xchg_env`
    publishes INJECT_ENV into the dict handed to the RANKS and never into the
    launcher's OWN environment, so every launcher-process caller of
    `inject_mode()` got the `shadow` default regardless of the boot's arm.
    Measured in #1361 [23a]: a boot armed `authoritative` printed
    `bounce_slots=3` (computed as if shadow) beside `bounce_inject=authoritative`
    (from the argv) on the SAME line.

(2) AN ABSENT MODE WAS READ AS A SHADOW FINDING. The bounce emitter's
    `inject=` field was a two-way branch, so a record with NO mode fell into
    `NOTHING-COMPARED` -- a claim that a shadow leg had graded nothing,
    invented out of missing evidence, while the `mode=` field one column to its
    left already said `unset`. Two fields on one line contradicting each other
    about the same absence.

PRIOR ART, found by the gate before the first edit: `INJECT_MODE_UNSET` was
ALREADY defined for this ticket (`weight_exchange.py`) and already used by the
emitter's `mode=` field. Only the `inject=` field never learned it. This commit
finishes a constant that was built and half-wired, rather than adding one.
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange as wx
from sglang.test.test_utils import CustomTestCase


class AnAuthoritativeBootMustNotReadShadow(CustomTestCase):
    """Direction 1: the launcher process has no INJECT_ENV, ever."""

    def test_explicit_beats_an_environment_that_was_never_published(self):
        old = os.environ.pop(wx.INJECT_ENV, None)
        try:
            # The launcher's own process: INJECT_ENV absent, boot armed
            # authoritative. Without `explicit` this answers "shadow" and every
            # number derived from it describes a different arm.
            self.assertEqual(
                wx.inject_mode(wx.INJECT_AUTHORITATIVE), wx.INJECT_AUTHORITATIVE)
            self.assertTrue(wx.inject_authoritative(wx.INJECT_AUTHORITATIVE))
            # and the unchanged reading is still the default, so no rank moves
            self.assertEqual(wx.inject_mode(), wx.INJECT_SHADOW)
        finally:
            if old is not None:
                os.environ[wx.INJECT_ENV] = old

    def test_a_typo_is_ignored_rather_than_trusted(self):
        """A bad `explicit` must NOT become `authoritative`.

        That is the direction INJECT_MODE_UNSET exists to guard: claiming
        ownership of 27 GiB of weights on evidence that did not parse.
        """
        old = os.environ.pop(wx.INJECT_ENV, None)
        try:
            self.assertEqual(wx.inject_mode("authoritatve"), wx.INJECT_SHADOW)
            self.assertEqual(wx.inject_mode(""), wx.INJECT_SHADOW)
            self.assertFalse(wx.inject_authoritative("authoritatve"))
        finally:
            if old is not None:
                os.environ[wx.INJECT_ENV] = old

    def test_the_ranks_still_read_the_environment(self):
        old = os.environ.get(wx.INJECT_ENV)
        os.environ[wx.INJECT_ENV] = wx.INJECT_AUTHORITATIVE
        try:
            self.assertEqual(wx.inject_mode(), wx.INJECT_AUTHORITATIVE)
        finally:
            if old is None:
                os.environ.pop(wx.INJECT_ENV, None)
            else:
                os.environ[wx.INJECT_ENV] = old


class NoInjectObjectMustNotPrintAModeItDoesNotHave(CustomTestCase):
    """Direction 2: an absent mode is `unset`, not a shadow finding."""

    def _line(self, mode):
        from sglang.srt.weg2 import weight_exchange_bounce as wb

        r = wb.BounceLegReport.__new__(wb.BounceLegReport)
        r.inject = None
        r.mode = mode
        r.banded = []
        r.overlap = 0
        r.verdict = "MATCH"
        for f in ("units", "covered", "uncovered", "leg", "group", "rank"):
            if not hasattr(r, f):
                setattr(r, f, 0)
        return r.line() if hasattr(r, "line") else str(r)

    def test_an_absent_mode_prints_unset_not_NOTHING_COMPARED(self):
        import inspect

        from sglang.srt.weg2 import weight_exchange_bounce as wb

        src = inspect.getsource(wb)
        i = src.index('"inject=by-design-authoritative "')
        window = src[i:i + 500]
        self.assertIn("INJECT_MODE_UNSET", window,
                      "an absent mode must print `unset`, not fall into the "
                      "NOTHING-COMPARED arm, which is a claim about a SHADOW leg")
        self.assertIn(f"== wx.{'INJECT_SHADOW'}", window,
                      "NOTHING-COMPARED must be reached only for a mode that "
                      "IS shadow, never for one that is merely not authoritative")

    def test_the_two_fields_agree_about_the_same_absence(self):
        """`mode=` already said `unset`; `inject=` used to contradict it."""
        import inspect

        from sglang.srt.weg2 import weight_exchange_bounce as wb

        src = inspect.getsource(wb)
        self.assertIn("mode={self.mode or wx.INJECT_MODE_UNSET}", src)
        self.assertEqual(
            src.count("INJECT_MODE_UNSET"), 2,
            "both the mode field and the inject field must name the absence",
        )


if __name__ == "__main__":
    unittest.main()
