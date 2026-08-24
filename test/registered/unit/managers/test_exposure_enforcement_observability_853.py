"""#853(i) -- every seam evaluation of the EXPOSURE law says what it decided.

THE W24 SPECIMEN, and it is an ABSENCE rather than a line::

    "#851 EXPOSURE ENFORCED"   0 occurrences   UNDETERMINABLE

Zero is multi-valued here, and that is the whole defect. `_enforce_exposure_at_seam`
has FOUR exits that return 0 in silence -- the spill import failing, the rung
attribute missing, the clamp raising (logged at DEBUG, which the boot does not
capture), and the ordinary `withdrawn == 0` -- and only the `withdrawn > 0` case
ever reaches the log. So a count of zero conflates:

    * the law ran and found nothing to correct   (healthy)
    * the law could not find its actuator        (inert)
    * the law threw and was swallowed            (broken)
    * the law never ran at all                   (unreachable)

On W24 the true answer was the fourth: grep finds ONE call site, on the cutover
path, and the 23.6-minute stuck phase had ZERO cutovers. An instrument whose
reading is identical whether it is healthy or absent measured nothing for the
entire window it was installed to measure.

This is the #851 defect class occurring INSIDE the #851 fix, which is why it is
worth a test rather than a log tweak: the corpus's signature failure mode is a
law connected to nothing, and the only defence that has ever worked is asserting
that each path SAYS which one it took.

WHAT THIS FILE PINS. Each of the five outcomes emits its own distinct marker, so
a reader (and a window count) can tell them apart. The markers are asserted to be
mutually exclusive -- a shared prefix that appears in all five would pass a naive
"is something logged" test while preserving exactly the ambiguity above.

Hermetic: no CUDA, no NVML, no pool. CUDA_VISIBLE_DEVICES="".
"""

import logging
import sys
import types
import unittest

from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

SPILL_MODULE = "sglang.srt.managers.phase_flip_spill"


class _Rung:
    """The actuator, modelled by what it returns to the seam."""

    def __init__(self, withdrawn=0, raises=False):
        self._withdrawn = withdrawn
        self._raises = raises
        self.calls = []

    def clamp_exposure_to_backing(self, why):
        self.calls.append(why)
        if self._raises:
            raise RuntimeError("the clamp refused")
        return self._withdrawn


def _runtime(rung=None):
    r = PhaseFlipRuntime.__new__(PhaseFlipRuntime)
    if rung is not None:
        from sglang.srt.managers.phase_flip_spill import KV_BACKING_RELIEF_ATTR

        sched = types.SimpleNamespace()
        setattr(sched, KV_BACKING_RELIEF_ATTR, rung)
        r._census_scheduler = sched
    else:
        r._census_scheduler = None
    return r


class EverySeamEvaluationIsOnTheRecord(unittest.TestCase):
    def _run(self, runtime, when="pp_to_tp cutover"):
        with self.assertLogs(
            "sglang.srt.managers.phase_flip_runtime", level=logging.DEBUG
        ) as caught:
            withdrawn = runtime._enforce_exposure_at_seam(when)
        return withdrawn, "\n".join(caught.output)

    def test_a_clean_evaluation_is_logged_rather_than_silent(self):
        """THE W24 CASE THAT MATTERS MOST. `withdrawn == 0` is the healthy
        outcome and it was indistinguishable from never running."""
        rung = _Rung(withdrawn=0)
        withdrawn, out = self._run(_runtime(rung))
        self.assertEqual(withdrawn, 0)
        self.assertIn("EXPOSURE CHECKED", out)
        self.assertEqual(rung.calls, ["pp_to_tp cutover"])

    def test_an_enforcement_still_names_the_rows_it_withdrew(self):
        withdrawn, out = self._run(_runtime(_Rung(withdrawn=7)))
        self.assertEqual(withdrawn, 7)
        self.assertIn("EXPOSURE ENFORCED", out)
        self.assertIn("7", out)

    def test_a_missing_rung_says_INERT_rather_than_returning_a_quiet_zero(self):
        """The actuator is absent: the law is installed and unable to act. That
        is not the same fact as 'nothing needed correcting'."""
        withdrawn, out = self._run(_runtime(None))
        self.assertEqual(withdrawn, 0)
        self.assertIn("EXPOSURE NOT ENFORCEABLE", out)

    def test_a_raising_clamp_says_FAILED_at_a_level_a_boot_captures(self):
        """W24 ran at INFO. The old handler logged this at DEBUG, so a clamp
        that threw on every seam would have left no trace in the boot log."""
        with self.assertLogs(
            "sglang.srt.managers.phase_flip_runtime", level=logging.INFO
        ) as caught:
            withdrawn = PhaseFlipRuntime._enforce_exposure_at_seam(
                _runtime(_Rung(raises=True)), "tp_to_pp cutover"
            )
        out = "\n".join(caught.output)
        self.assertEqual(withdrawn, 0)
        self.assertIn("EXPOSURE CHECK FAILED", out)

    def test_a_missing_actuator_module_is_its_own_marker(self):
        """The import exit. Distinct from the missing-rung exit because the
        remedies differ: one is a build/packaging fault, the other a wiring one."""
        saved = sys.modules.get(SPILL_MODULE)
        sys.modules[SPILL_MODULE] = types.ModuleType(SPILL_MODULE)
        try:
            withdrawn, out = self._run(_runtime(None))
        finally:
            if saved is not None:
                sys.modules[SPILL_MODULE] = saved
            else:  # pragma: no cover - the module is always importable here
                del sys.modules[SPILL_MODULE]
        self.assertEqual(withdrawn, 0)
        self.assertIn("EXPOSURE ACTUATOR MISSING", out)

    def test_the_five_markers_are_mutually_exclusive(self):
        """THE CAN-FAIL DIRECTION FOR THE WHOLE FILE. One shared marker on all
        five paths would satisfy every assertion above while preserving exactly
        the ambiguity this ticket exists to remove: each reading must exclude
        the other four."""
        markers = [
            "EXPOSURE CHECKED",
            "EXPOSURE ENFORCED",
            "EXPOSURE NOT ENFORCEABLE",
            "EXPOSURE CHECK FAILED",
            "EXPOSURE ACTUATOR MISSING",
        ]
        outs = {}
        _, outs["EXPOSURE CHECKED"] = self._run(_runtime(_Rung(withdrawn=0)))
        _, outs["EXPOSURE ENFORCED"] = self._run(_runtime(_Rung(withdrawn=3)))
        _, outs["EXPOSURE NOT ENFORCEABLE"] = self._run(_runtime(None))
        with self.assertLogs(
            "sglang.srt.managers.phase_flip_runtime", level=logging.INFO
        ) as caught:
            PhaseFlipRuntime._enforce_exposure_at_seam(
                _runtime(_Rung(raises=True)), "x"
            )
        outs["EXPOSURE CHECK FAILED"] = "\n".join(caught.output)
        saved = sys.modules.get(SPILL_MODULE)
        sys.modules[SPILL_MODULE] = types.ModuleType(SPILL_MODULE)
        try:
            _, outs["EXPOSURE ACTUATOR MISSING"] = self._run(_runtime(None))
        finally:
            sys.modules[SPILL_MODULE] = saved

        for mine, out in outs.items():
            for other in markers:
                if other == mine:
                    self.assertIn(other, out)
                else:
                    self.assertNotIn(other, out, f"{mine} also emitted {other}")


class TheSeamNamesTheEventItEvaluated(unittest.TestCase):
    """A marker that does not carry `when` cannot tell an arm-time evaluation
    from a cutover one, which is precisely the distinction #853(i) adds."""

    def test_the_when_string_reaches_every_marker(self):
        for runtime, level in (
            (_runtime(_Rung(withdrawn=0)), logging.DEBUG),
            (_runtime(_Rung(withdrawn=5)), logging.DEBUG),
            (_runtime(None), logging.DEBUG),
            (_runtime(_Rung(raises=True)), logging.INFO),
        ):
            with self.assertLogs(
                "sglang.srt.managers.phase_flip_runtime", level=level
            ) as caught:
                PhaseFlipRuntime._enforce_exposure_at_seam(runtime, "tp_to_pp arm")
            self.assertIn("tp_to_pp arm", "\n".join(caught.output))


if __name__ == "__main__":
    unittest.main()
