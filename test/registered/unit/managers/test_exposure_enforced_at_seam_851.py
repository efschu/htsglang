"""#851 F1: the EXPOSURE law is enforced at the seam events, not only audited.

W22's root: the allocator exposed 470755 row ids against 126976 committed rows
-- gap 343779, equal to the census `withheld` field exactly. The authority SAW
it and said so 48 times in that one boot. Nothing was held to it. The #816
clamp fired twice.

`fe43b09e52` (#822) names this as its own open item: `_retire_row_id_space`
"does not yet REFUSE such an id at the allocator -- that is enforcement".
O-2 recorded that as an owner decision (audit, not enforcement) on the grounds
of hot-path blast radius; the reversal is on record in WINDOW-QUEUE with the
scope that answers it -- enforcement at THE TWO SEAM EVENTS ONLY, never per
allocation. The hot allocation path is untouched; the law is enforced where the
id space changes regime.

THE ACTUATOR ALREADY EXISTS and is used here rather than reinvented:
`KvBackingRelief.clamp_exposure_to_backing` (#816). Its own contract is why it
is safe to call at a seam:

    "It only ever LOWERS exposure toward `_current_rows()` -- a MEASURED
    committed count, never a remembered one (the #684 lesson) -- and it never
    lowers the BACKING. If the backing already sits below the live set, that is
    the #722 state and it is reported here rather than papered over."

So the direction guard holds by construction: this can never cap below a live
set, which is the thing `test_residency_cap_flip_levelling_792::
TheLevellingMustNotCapBelowTheLiveSet` forbids and which the retracted
floor-clamp remedy would have done.

WHAT THIS DOES NOT CLOSE, so nobody cites it wrongly: `floor > cap` SURVIVES
this fix. With exposed == committed == 126976, `max_live` can still sit at the
cap and the floor is `max_live + 1 + margin + reserve` = 131073 by design --
the admission reserve is deliberately above the high-water mark. F1 converts a
permanent SILENT veto into an explicit grow requirement at the seam; F2 makes
that grow fundable or refuses it at boot. See docs/dev/NOTE_851_build_caveats.md.

Hermetic: duck-typed rank and rung, no scheduler, no pool, no CUDA.
"""

import unittest

from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime


class _Rung:
    """Stands in for KvBackingRelief, recording what the seam asked of it."""

    def __init__(self, exposed, committed):
        self.exposed = exposed
        self.committed = committed
        self.calls = []

    def exposed_rows(self):
        return self.exposed

    def _current_rows(self):
        return self.committed

    def clamp_exposure_to_backing(self, why):
        self.calls.append(why)
        withdrawn = max(0, self.exposed - self.committed)
        # The real actuator lowers exposure and never touches backing.
        self.exposed = min(self.exposed, self.committed)
        return withdrawn


class _Rank:
    """The smallest object `_enforce_exposure_at_seam` needs."""

    def __init__(self, rung):
        from sglang.srt.managers.phase_flip_spill import KV_BACKING_RELIEF_ATTR

        sched = type("S", (), {})()
        setattr(sched, KV_BACKING_RELIEF_ATTR, rung)
        self._census_scheduler = sched


def _enforce(rank, when):
    return PhaseFlipRuntime._enforce_exposure_at_seam(rank, when)


class TestTheSeamEnforcesTheLaw(unittest.TestCase):
    def test_the_w22_violation_is_corrected_at_the_seam(self):
        """RED before F1. The exact W22 numbers."""
        rung = _Rung(exposed=470755, committed=126976)
        withdrawn = _enforce(_Rank(rung), "tp_to_pp cutover")

        self.assertEqual(withdrawn, 343779)
        self.assertEqual(rung.exposed, 126976)
        self.assertEqual(rung.committed, 126976, "backing must NOT be lowered")
        self.assertTrue(rung.calls)

    def test_a_sound_id_space_is_left_alone(self):
        """CAN-FAIL TWIN. Enforcement must be a no-op on a legal state."""
        rung = _Rung(exposed=126976, committed=126976)
        withdrawn = _enforce(_Rank(rung), "tp_to_pp cutover")

        self.assertEqual(withdrawn, 0)
        self.assertEqual(rung.exposed, 126976)

    def test_backing_below_the_live_set_is_NOT_papered_over(self):
        """THE DIRECTION GUARD, in the shape that matters here.

        Under-backing is the #722 state and a clamp cannot repair it; the
        actuator reports it instead. Enforcement must never respond by lowering
        the backing -- that is the reverted floor-clamp remedy wearing a
        different hat.
        """
        rung = _Rung(exposed=100_000, committed=126_976)
        _enforce(_Rank(rung), "tp_to_pp cutover")
        self.assertEqual(rung.committed, 126_976)
        self.assertEqual(rung.exposed, 100_000, "exposure must not be RAISED")

    def test_a_missing_rung_is_survivable(self):
        """A seam must not die because there is nothing to enforce against."""
        blank = type("R", (), {"_census_scheduler": None})()
        self.assertEqual(_enforce(blank, "tp_to_pp cutover"), 0)

    def test_a_raising_rung_does_not_kill_the_cutover(self):
        """The cutover is mid-flight; enforcement may refuse, never explode."""

        class _Boom:
            def clamp_exposure_to_backing(self, why):
                raise RuntimeError("nvml exploded")

        from sglang.srt.managers.phase_flip_spill import KV_BACKING_RELIEF_ATTR

        rank = type("R", (), {})()
        sched = type("S", (), {})()
        setattr(sched, KV_BACKING_RELIEF_ATTR, _Boom())
        rank._census_scheduler = sched
        self.assertEqual(_enforce(rank, "tp_to_pp cutover"), 0)


if __name__ == "__main__":
    unittest.main()
