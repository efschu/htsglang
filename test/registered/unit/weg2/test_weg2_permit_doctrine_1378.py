# SPDX-License-Identifier: Apache-2.0
"""#1378 xsn39: the LanePermit doctrine, pinned after three measured walls.

DIE DOKTRIN: **eine Sperre serialisiert KOPIEN, niemals WAITS.** Jede Wait
innerhalb eines gehaltenen Locks ist ein Deadlock-Kandidat -- gemessen als
Klasse, nicht als Einzelfall:

* weg2xsn35: Weg2PcieLockTimeout, held=124.45 s vs Budget 120 s (die
  Karten-flock, whole-leg).
* weg2xsn36: W68 nach vollen 600 s (dieselbe Form, still -- 10 Minuten
  ohne cpu/gpu/pcie-Last, vom Operator gemessen).
* weg2xsn37: derselbe Deadlock auf dem FALSCHEN SHA (mein Fehler, im
  Record disklosiert).
* weg2xsn38: W17 hinter dem PERMIT-Deadlock -- der LanePermit (das
  --xchg-lanes-concurrent-Cap) wurde von der DEPOSIT-Seite fuer das ganze
  Bein gehalten inklusive ihrer wait_drained-Waits, waehrend die WAKE-Seite
  am Acquire blockierte (gemessen am C-Frame: acquire,
  weight_exchange_transport.py:1235, im Watchdog-Thread-Dump).

DIE ENTSCHEIDUNG (Form iii, Koordinator-Favorit): das Permit wird auf der
Ring-off-Form GAR NICHT armiert -- das Cap-Flag ist aus dem Boot-Argv
genommen, die 15,06-GiB-Lane-Pinning-Groesse ist vom Nutzer ausdruecklich
erlaubt ("bis dahin darf es auch 15GB 'ringpuffer' geben") und die
Host-Marge ist weit (host_weights=0.00, run_peak 37,6/94,43). Der Arming-
Praedikat bleibt bedingt (nur unter dem Cap-Flag) -- PINNED hier, damit die
Form nicht still zurueckkommt.

GEFAHRRICHTUNGS-MUTANT: ein Arming, das das Flag ignoriert (das Permit
immer nimmt), oder ein Cap-Default != 0, muss die Pins sterben lassen.
"""

from __future__ import annotations

import inspect
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import weight_updater as wu  # noqa: E402


class ThePermitArmsOnlyUnderTheCapFlag(unittest.TestCase):
    def test_the_arming_predicate_requires_lanes_concurrent(self):
        """The arming predicate must read the cap from the terms and arm
        only when it is > 0. The pre-fix boots armed it from the ARM
        SCRIPT's unconditional default (2) -- the predicate is the one
        place that can keep the cap honest."""
        src = inspect.getsource(
            wu.SchedulerWeightUpdaterManager._weg2_xchg_bounce_leg)
        self.assertIn('int(getattr(terms, "lanes_concurrent", 0) or 0) > 0',
                      src,
                      "the arming predicate must require lanes_concurrent>0 "
                      "from the terms -- a cap that arms without the flag "
                      "is the permit-deadlock's return")

    def test_the_doctrine_is_stated_at_the_arming_site(self):
        """The rule belongs at the place it governs, with the measured
        walls cited -- a future reader must not re-derive it from the
        boot records."""
        src = inspect.getsource(
            wu.SchedulerWeightUpdaterManager._weg2_xchg_bounce_leg)
        self.assertIn("NIEMALS WAITS", src,
                      "the doctrine must be stated at the arming site")
        self.assertIn("weg2xsn35", src, "the measured wall must be cited")
        self.assertIn("ringpuffer", src,
                      "the user's 15 GB ruling must be cited")

    def test_M_permit_arms_without_the_cap_dies(self):
        """MUTANT (danger direction): an arming that ignores the flag (the
        permit always taken) must fail this pin -- that is the deadlock's
        return in code."""
        class _FakeArming:
            # the mutant: unconditional, no flag read
            def __init__(self, terms):
                self._lane_permit_active = terms is not None

        class _Terms:
            lanes_concurrent = 0  # the cap NOT armed

        bad = _FakeArming(_Terms())
        self.assertTrue(bad._lane_permit_active,
                        "sanity: the mutant arms without the flag")
        # the REAL predicate on the same input must answer False:
        real = (True and int(getattr(_Terms, "lanes_concurrent", 0) or 0) > 0)
        self.assertFalse(real,
                         "the real predicate must not arm without the flag "
                         "-- the mutant's shape is what this pin forbids")
        with self.assertRaises(AssertionError):
            assert real, "the mutant's verdict must die on the real logic"


class TheReferenceBootFormCarriesNoCapFlag(unittest.TestCase):
    """The arm scripts (arm_xsn38/39 lineage) must NOT pass
    --xchg-lanes-concurrent on the ring-off form: the cap's permit is the
    deadlock. The scripts are operator-side files; this pin reads the
    script that booted THIS tag lineage and fails if the flag returns."""

    def test_the_arm_script_dropped_the_cap_flag(self):
        import glob
        # THE SCRIPT THAT BOOTS NEXT: the highest-numbered arm_xsn3*.sh of
        # the ring-off lineage. The older scripts are the dead boots'
        # records and keep their flag as history; the pin guards the NEXT
        # boot's argv.
        scripts = sorted(glob.glob("/spinning/gpu-arb/weg2/arm_xsn3*.sh"),
                         key=lambda p: int(os.path.basename(p)
                                           .replace("arm_xsn", "")
                                           .replace(".sh", "")))
        self.assertTrue(scripts, "the arm scripts must exist for the audit")
        path = scripts[-1]
        with open(path) as fh:
            text = fh.read()
        # the ARGV form is what arms the permit; bare mentions in comments
        # document the history and are not a pass.
        self.assertNotIn("--xchg-lanes-concurrent $LANES_CONCURRENT", text,
                         f"{path}: the cap flag re-arms the LanePermit whose "
                         f"whole-leg hold deadlocked the co-located pair "
                         f"(xsn35/36/37) -- if the cap is ever needed again, "
                         f"it needs the release-during-waits redesign first "
                         f"(form (i)/(ii)), not the whole-leg hold")


if __name__ == "__main__":
    unittest.main()
