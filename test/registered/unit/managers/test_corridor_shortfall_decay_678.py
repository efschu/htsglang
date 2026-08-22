"""#678: a measured corridor breach must expire, or one intrusion sizes forever.

WHAT IT COST, MEASURED. `corridor_shortfall_bytes` is added straight to the
arming floor's load margin -- `(DEFAULT_MARGIN_MIB << 20) + measured` -- and the
arming floor is the binding constraint on two of three ranks. On 2026-08-16 the
rank-0 record carried 1004 MiB of it while every record written on 2026-08-15
carried 0, and the boot that read it logged NO breach of its own: it was
inherited. The breach it descends from is almost certainly the 02:36:30 event on
that exact card, where a test harness belonging to this strand held 4.29 GiB and
drove free to 76 MiB. A one-off intrusion, taxing the pool on every subsequent
boot.

THE OLD SEMANTICS WERE HALF RIGHT, AND THE HALF THAT WAS RIGHT IS KEPT.
`record_corridor_shortfall` is documented as "A MONOTONIC MAXIMUM, deliberately
-- a shallower breach later does not mean the deeper one cannot recur; the pool
must be sized for the worst instant that has ever been seen". That is correct
WITHIN an observation: while a breach keeps being seen, the worst instant is the
honest number. It is wrong ACROSS boots that never see it again, because "ever"
had no end and nothing could retire a number nobody could reproduce.

So the rule is now: monotonic maximum while it is being OBSERVED, geometric
decay across boots that observe nothing. A breach that recurs is re-observed and
re-raised to its worst on the spot; a breach that does not recur is halved by
each clean flip boot and forgotten below `SHORTFALL_FORGET_BYTES`.

"OBSERVED BY THIS PROCESS" IS THE DISCRIMINATOR, and it is a pid rather than a
timestamp because both writers live in the same process: the runtime's corridor
audit stamps the record mid-run, and `write_seam_reserve` rewrites it at the end
of the same boot's flip measurement. Same pid means this boot saw the breach and
the value stands; a different pid means the value was inherited and this boot,
having measured its seam without seeing a breach, is evidence against it.

FOURTH LATCH OF THIS SESSION, and the same cure shape as the other three: #681's
eviction count that could not be paid, #682's guard ceiling the scheduler never
held, #684's `_exhausted_at_rows` process-lifetime latch. Each one was a number
that could only ever ratchet one way.

RANK-LOCAL. The record is per (configuration, rank) and the shortfall is one
card's own measurement; ranks legitimately differ (1004 / 0 / 0 on this boot).
No collective reads or writes it, and this change adds none.
"""

import json
import os
import tempfile
import unittest

from sglang.srt.managers import phase_flip_seam_reserve as sr
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)

MIB = 1 << 20
BREACH = 1004 * MIB  # the value the rank-0 record actually carried


class _Recorded(unittest.TestCase):
    """Drives the real writers over a temp record; no rig cache is touched."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._dir.name, "seam-rank0.json")
        self._real_record_path = sr.record_path
        sr.record_path = lambda server_args, world_rank: self.path

    def tearDown(self):
        sr.record_path = self._real_record_path
        self._dir.cleanup()

    # -- helpers ---------------------------------------------------------

    def _write_boot_record(self, **extra):
        """One end-of-flip-boot measurement, as the production writer makes it."""
        sr.write_seam_reserve(
            None,
            0,
            fixed_bytes=228 * MIB,
            per_row_bytes=2360.3,
            detail="test",
            have_bytes=3880139776,
            id_space=435319,
            **extra,
        )

    def _record(self):
        with open(self.path) as fh:
            return json.load(fh)

    def _seed(self, shortfall, pid):
        """A record carrying a shortfall observed by ``pid``."""
        self._write_boot_record()
        rec = self._record()
        rec["corridor_shortfall_bytes"] = int(shortfall)
        rec["corridor_shortfall_pid"] = int(pid)
        with open(self.path, "w") as fh:
            json.dump(rec, fh)


class TheBreachThisBootSawStands(_Recorded):
    def test_a_breach_observed_by_this_process_is_preserved(self):
        """The half of the old rule that was right."""
        self._write_boot_record()
        sr.record_corridor_shortfall(None, 0, BREACH)
        self._write_boot_record()
        self.assertEqual(BREACH, self._record()["corridor_shortfall_bytes"])

    def test_a_deeper_breach_still_raises_the_record(self):
        self._write_boot_record()
        sr.record_corridor_shortfall(None, 0, 200 * MIB)
        sr.record_corridor_shortfall(None, 0, BREACH)
        self.assertEqual(BREACH, self._record()["corridor_shortfall_bytes"])

    def test_a_shallower_breach_does_not_lower_it_within_one_observation(self):
        """The worst instant is the honest number while it keeps being seen."""
        self._write_boot_record()
        sr.record_corridor_shortfall(None, 0, BREACH)
        sr.record_corridor_shortfall(None, 0, 20 * MIB)
        self.assertEqual(BREACH, self._record()["corridor_shortfall_bytes"])


class TheInheritedBreachExpires(_Recorded):
    def test_an_inherited_breach_is_decayed_by_a_clean_boot(self):
        """RED before #678: preserved verbatim by ``max(0, prior_shortfall)``."""
        self._seed(BREACH, pid=os.getpid() + 1)
        self._write_boot_record()
        got = self._record()["corridor_shortfall_bytes"]
        self.assertLess(got, BREACH, "a boot that saw no breach must not renew it")
        self.assertEqual(BREACH // 2, got)

    def test_a_one_off_is_forgotten_in_a_bounded_number_of_clean_boots(self):
        """It must reach EXACTLY zero, not decay to a long tail that never
        stops taxing the floor."""
        self._seed(BREACH, pid=os.getpid() + 1)
        for boot in range(12):
            # Each iteration is a fresh boot: nothing observed a breach.
            rec = self._record()
            rec["corridor_shortfall_pid"] = os.getpid() + 1 + boot
            with open(self.path, "w") as fh:
                json.dump(rec, fh)
            self._write_boot_record()
            if self._record()["corridor_shortfall_bytes"] == 0:
                break
        self.assertEqual(
            0,
            self._record()["corridor_shortfall_bytes"],
            "a breach nobody can reproduce must expire within a dozen boots",
        )

    def test_the_1004_mib_intrusion_stops_taxing_the_floor(self):
        """The production case, end to end.

        1004 MiB on the arming floor is 1004 MiB the pool may not hold. After
        enough clean boots the margin is back to its default and the sizer is
        answering to measurements again.
        """
        self._seed(BREACH, pid=os.getpid() + 1)
        for boot in range(12):
            rec = self._record()
            rec["corridor_shortfall_pid"] = os.getpid() + 100 + boot
            with open(self.path, "w") as fh:
                json.dump(rec, fh)
            self._write_boot_record()
        reserve = sr.SeamReserve(
            corridor_shortfall_bytes=self._record()["corridor_shortfall_bytes"]
        )
        self.assertEqual(
            sr.DEFAULT_MARGIN_MIB << 20,
            sr.seam_margin_bytes(reserve)
            if hasattr(sr, "seam_margin_bytes")
            else (sr.DEFAULT_MARGIN_MIB << 20)
            + max(0, reserve.corridor_shortfall_bytes),
            "the load margin must return to its default once the breach expires",
        )


class TheRecurringBreachIsNotForgotten(_Recorded):
    def test_a_breach_that_keeps_happening_holds_at_its_worst(self):
        """The property the decay must not cost: a real, reproducible breach.

        Without this the fix would be indistinguishable from deleting the term.
        """
        self._seed(BREACH, pid=os.getpid() + 1)
        for _ in range(6):
            self._write_boot_record()  # decays
            sr.record_corridor_shortfall(None, 0, BREACH)  # observed again
        self.assertEqual(
            BREACH,
            self._record()["corridor_shortfall_bytes"],
            "a breach observed on every boot must not be decayed away",
        )


if __name__ == "__main__":
    unittest.main()
