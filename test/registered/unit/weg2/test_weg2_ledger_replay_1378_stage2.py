# SPDX-License-Identifier: Apache-2.0
"""#1378 Desk-Leiter Stufe 2 -- Ledger-Replay ueber ALLE vorhandenen xsn31-
Records, MESSUNG statt Umbau: dieselben zwei Invarianten
(``test_weg2_cushion_gate_1377.py``'s eigene Kalibrierung, hier auf jeden
Record der xsn31-Kette erweitert, statt nur auf das eine Paar) gegen die
AUFGEZEICHNETEN Zahlen jeder xsn31-Attempt gefahren, hermetisch, kein Boot.

DATEIGRENZE: dieser Sitz aendert `host_ledger.py`/`launcher.py` NICHT (sie
gehoeren gerade dem Ledger-Sitz fuer den HiCache-Schalter und den
Flaggengruppen-Nachtrag). Diese Datei IMPORTIERT beide nur lesend und ist
selbst das Deliverable -- der Replay-Harness.

KALIBRIER-ANKER (woertlich aus test_weg2_cushion_gate_1377.py, nicht neu
erfunden): xsn31/2 GRUEN (6.99), xsn31/3 ROT (-0.97). Ein Replay, der beide
gruen liest, misst nicht -- dann ist der Replay defekt, nicht die Welt. Diese
Datei pinnt dieselben zwei Zahlen als `test_the_calibration_anchors_hold`
BEVOR sie auf die restliche Kette erweitert wird.

DIE ZWEI INVARIANTEN, je Record:
  (1) predicted_run_peak + RATE_LATCH_CUSHION_FLOOR_GIB <= hard_bound_gib
  (2) cushion_headroom_gib(cushion_min_of_PREV, bounce_of_CURR, bounce_of_PREV)
      >= RATE_LATCH_CUSHION_FLOOR_GIB
      (fuer den ERSTEN Record einer Kette: PREV = CURR, Delta 0 -- das ist,
      was xsn31/2s eigene Kalibrierung tut: `cushion_headroom_gib(6.99, 7.79,
      7.79) == 6.99`.)

QUELLEN JE ZAHL (file:line im Prosaprotokoll, kein Boot in diesem Sitz):
  xsn31/2, xsn31/3: BOOT_weg2xsn31_0913.md (Peak/Cushion-Tabelle Zeile
    195-199) UND test_weg2_cushion_gate_1377.py (90.66/94.43/1.50, das
    dort schon gepinnte predicted_run_peak/bound-Paar -- WIEDERVERWENDET,
    nicht neu geschaetzt).
  xsn31/4, /5, /6, /7: BOOT_weg2xsn31_0913_4.md, je Versuch-Abschnitt
    zitiert (Zeilennummern in den Docstrings der einzelnen Testfaelle
    unten, weil sie sich mit jeder Bearbeitung des Dokuments verschieben
    koennten -- die Abschnittsueberschriften ("VERSUCH 4/5/6/7") sind der
    stabile Anker).
"""

from __future__ import annotations

import inspect
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger as hl
from sglang.test.test_utils import CustomTestCase

FLOOR = hl.RATE_LATCH_CUSHION_FLOOR_GIB  # 1.50, read from the ledger, not restated


# ---------------------------------------------------------------------------
# THE RECORDS. Every field cites its own source; nothing here is invented.
# `bounce_priced_gib` is what the ARM charged AT THE TIME (the number a
# pre-boot dry run could see); `bounce_real_gib` is what a 0.2 s filesystem
# poller or the lane's own HOST-SLOT lines measured DURING the actual boot,
# where that instrument existed (xsn31/4 onward -- xsn31/2 and /3 predate it
# and are NOT backfilled with a guessed number).
# ---------------------------------------------------------------------------

RECORDS = {
    "xsn31/2": dict(
        # BOOT_weg2xsn31_0913.md:195-199 (comparison table); predicted_run_peak
        # and hard_bound are the SAME pair test_weg2_cushion_gate_1377.py
        # already pins for this boot (90.66 / 94.43) -- reused, not re-derived.
        bounce_priced_gib=7.79,
        bounce_real_gib=None,  # poller instrument did not exist yet
        cushion_min_gib=6.99,
        predicted_run_peak_gib=90.66,
        hard_bound_gib=94.43,
        note="pre-#1385 F1b Option-1 pricing; the boot never reached a bounce "
             "allocation at all (died on WEG2-DORMANT set / AttributeError "
             "before deposit) -- 7.79 is the ARM's priced figure, not a "
             "measured one either",
    ),
    "xsn31/3": dict(
        bounce_priced_gib=15.75,
        bounce_real_gib=None,  # plausibly ALSO inflated by the pre-#1385
        # allocator bug (leg_slot_bytes not yet fixed at this date), but
        # UNMEASURED -- the poller did not exist, and backfilling a guess
        # here would be exactly the "null nur bei erreichtem Emitter" rule
        # this desk's own memory forbids.
        cushion_min_gib=0.20,
        predicted_run_peak_gib=90.66,
        hard_bound_gib=94.43,
        note="W98 cushion=0.20, now=92.40, shmem=67.51 (front.log:202); "
             "measured 'Peak nicht-reklaimierbar' 92.40 GiB is a DIFFERENT, "
             "run-moment quantity from predicted_run_peak_gib and is not "
             "substituted for it here",
    ),
    "xsn31/4": dict(
        # BOOT_weg2xsn31_0913_4.md, VERSUCH 4 section. lanes_concurrent=2,
        # PRE-size-fix allocator (leg_slot_bytes bug still present).
        bounce_priced_gib=6.75,
        bounce_real_gib=2 * 16.905,  # 2 lanes x 16.905 GiB/lane, poller-measured
        cushion_min_gib=0.19,
        predicted_run_peak_gib=84.74,  # worst of the "84.39-84.74" clean dry-run range
        hard_bound_gib=94.43,
        note="W98 cushion=0.19 now=93.81 shmem=69.76; the size bug (#1385/"
             "xsn31-4, fixed in commit 461fdebca0) is STILL PRESENT here",
    ),
    "xsn31/5": dict(
        bounce_priced_gib=3.75,
        bounce_real_gib=1 * 16.905,
        cushion_min_gib=0.85,
        predicted_run_peak_gib=82.19,
        hard_bound_gib=94.43,
        note="W98 cushion=0.85 now=91.15 shmem=67.66; cap correctly holds at "
             "1 file all boot (3018 poll ticks), size bug STILL present "
             "(same 16.905 GiB/file as xsn31/4)",
    ),
    "xsn31/6": dict(
        # Size fix landed (leg_slot_bytes). Per-lane size now matches the
        # price (3.00 GiB), but a SEPARATE, still-open gap (named, not
        # traced to file:line, in this same document's VERSUCH 6 section)
        # let up to 4 lanes run concurrently despite lanes_concurrent=1.
        bounce_priced_gib=3.75,
        bounce_real_gib=12.00,  # 4 x 3.00 GiB, poller peak
        cushion_min_gib=0.20,
        predicted_run_peak_gib=82.48,  # worst of "81.41-82.48"
        hard_bound_gib=94.43,
        note="W98 cushion=0.20; size fix BOOT-PROVEN correct (3.00 GiB/file "
             "exact), but 4 files coexisted instead of 1 -- a cross-group "
             "concurrency gap, NOT the size bug",
    ),
    "xsn31/7-projected": dict(
        # VERSUCH 7 crashed at MODEL LOAD (--speculative-draft-kv-only needs
        # --hicache-canonical-kv-page, server_args.py:8420-8432) BEFORE any
        # bounce allocation -- there is no measured cushion for this attempt.
        # This row is a PROJECTION using xsn31/6's own recorded bounce (the
        # cross-group gap is unrelated to HiCache and is assumed UNCHANGED)
        # plus BOOT7's own measured HiCache saving.
        bounce_priced_gib=3.75,       # unchanged from xsn31/6 -- HiCache does
        bounce_real_gib=12.00,        # not touch lane pricing or the gap
        cushion_min_gib=None,          # NEVER MEASURED -- projection only
        predicted_run_peak_gib=73.50,  # worst of "73.35-73.38"/"73.49-73.50"
        hard_bound_gib=94.43,
        # BOOT7's own measured delta (VERSUCH 7 section, "Koordinator-
        # Pflichtpruefung"): run_peak 81.79 -> 73.62 GiB, i.e. -8.17 GiB when
        # --weg2-disable-hicache is armed; ANALYSE_57GIB_SHMEM_0913.md's
        # earlier, independent estimate was -8.14 GiB (rings+anchors+
        # overhead) -- the two agree within 0.03 GiB.
        hicache_saving_gib=8.17,
        note="PROJECTED, never booted to a flip; cushion_min is None by "
             "construction, not a guess",
    ),
}

#: The chain order for invariant (2)'s consecutive-pair replay. A tuple of
#: (prev, curr) names, in the order the actual boots ran.
CHAIN = [
    ("xsn31/2", "xsn31/2"),   # the FIRST record: self-check, delta 0 --
                              # this IS how xsn31/2's own 6.99 anchor is read
    ("xsn31/2", "xsn31/3"),   # THE CALIBRATION PAIR
    ("xsn31/3", "xsn31/4"),   # priced-basis only (xsn31/3 has no real bounce)
    ("xsn31/4", "xsn31/5"),   # real-basis available for both
    ("xsn31/5", "xsn31/6"),   # real-basis available for both -- THE NON-
                              # MONOTONIC TRANSITION (bounce fell, cushion fell)
    ("xsn31/6", "xsn31/7-projected"),  # HiCache-adjusted projection
]


def _invariant1(rec: dict) -> bool:
    """``predicted_run_peak + FLOOR <= hard_bound`` -- the form
    test_weg2_cushion_gate_1377.py already pins as funding BOTH xsn31/2 and
    xsn31/3 (hence "not the gate"). Reused verbatim, extended to every
    record."""
    return rec["predicted_run_peak_gib"] + FLOOR <= rec["hard_bound_gib"]


def _invariant2(prev: dict, curr: dict, *, basis: str) -> "float | None":
    """``cushion_headroom_gib`` -- THE gate, called through host_ledger's own
    function (never re-implemented here). ``basis`` selects which bounce
    figures to feed it: 'priced' (what the ARM believed at the time) or
    'real' (what a poller/HOST-SLOT line actually measured, where it
    exists)."""
    key = f"bounce_{basis}_gib"
    bounce_now = curr.get(key)
    bounce_then = prev.get(key)
    if bounce_now is None or bounce_then is None:
        return None
    return hl.cushion_headroom_gib(prev["cushion_min_gib"], bounce_now, bounce_then)


class TheCalibrationAnchorsHold(CustomTestCase):
    """A replay that reads BOTH of these green measures nothing -- it means
    the replay itself is defective, not the world (coordinator order,
    verbatim). This class fails LOUDLY if that ever happens."""

    def test_xsn31_slash_2_alone_is_GREEN(self):
        v = _invariant2(RECORDS["xsn31/2"], RECORDS["xsn31/2"], basis="priced")
        self.assertAlmostEqual(v, 6.99, places=2)
        self.assertGreaterEqual(v, FLOOR)

    def test_xsn31_slash_2_to_3_is_RED(self):
        v = _invariant2(RECORDS["xsn31/2"], RECORDS["xsn31/3"], basis="priced")
        self.assertAlmostEqual(v, -0.97, places=2)
        self.assertLess(v, FLOOR)

    def test_the_two_anchors_are_not_the_same_reading(self):
        """The literal guard against 'a replay that reads both green'."""
        g = _invariant2(RECORDS["xsn31/2"], RECORDS["xsn31/2"], basis="priced")
        r = _invariant2(RECORDS["xsn31/2"], RECORDS["xsn31/3"], basis="priced")
        self.assertGreaterEqual(g, FLOOR)
        self.assertLess(r, FLOOR)
        self.assertNotAlmostEqual(g, r, places=1)


class Invariant1IsConsistentlyWeak(CustomTestCase):
    """``predicted_run_peak + FLOOR <= bound`` PASSES for every single
    record in this chain, including the four (xsn31/3,4,5,6) that actually
    latched W98 on the metal. This is not a defect in the arithmetic --
    both this file and test_weg2_cushion_gate_1377.py already document it
    as 'a floor, not the gate' -- but the replay must show it holds across
    the WHOLE chain, not just the original calibration pair, or invariant
    (2) below would be redundant instead of the thing that actually
    discriminates."""

    def test_every_record_passes_invariant_1(self):
        for name, rec in RECORDS.items():
            with self.subTest(record=name):
                self.assertTrue(_invariant1(rec), (name, rec))

    def test_invariant_1_alone_would_have_fundable_every_wall(self):
        """The four records that actually hit W98 on the metal
        (xsn31/3,4,5,6) all pass invariant 1 -- named explicitly so a
        reader cannot mistake 'passes invariant 1' for 'the boot will
        succeed'."""
        walls = ("xsn31/3", "xsn31/4", "xsn31/5", "xsn31/6")
        for name in walls:
            with self.subTest(record=name):
                self.assertTrue(_invariant1(RECORDS[name]))


class Invariant2ReplayedOverTheWholeChain(CustomTestCase):
    """The verdict table the message asks for, as executable assertions
    rather than prose."""

    def test_xsn31_3_to_4_priced_basis_mismatches_reality(self):
        """xsn31/3 has no measured real bounce (the poller did not exist
        yet), so this transition can only be replayed on the PRICED figures
        -- and the priced figures were WRONG at the time (the #1385/
        xsn31-4 slot-size bug, already found and fixed this session,
        commit 461fdebca0). The formula itself is not at fault: feeding it
        a price that was 5.6x too small produces an optimistic verdict by
        construction. Named here so a future reader does not mistake this
        MISMATCH for a defect in cushion_headroom_gib."""
        v = _invariant2(RECORDS["xsn31/3"], RECORDS["xsn31/4"], basis="priced")
        self.assertGreaterEqual(v, FLOOR, "priced-basis predicts GREEN")
        # Reality: xsn31/4 measured cushion 0.19 -- RED. The mismatch is the
        # finding, asserted as an inequality rather than left implicit.
        self.assertLess(RECORDS["xsn31/4"]["cushion_min_gib"], FLOOR)

    def test_xsn31_4_to_5_real_basis_correctly_predicts_improvement(self):
        """Real bounce nearly halved (33.81 -> 16.905 GiB, the lane cap
        going from 2 to 1). The formula predicts a large GREEN improvement,
        and reality DID improve (0.19 -> 0.85) -- just not enough to clear
        the floor. Direction agrees; magnitude overshoots. Recorded, not
        smoothed over."""
        v = _invariant2(RECORDS["xsn31/4"], RECORDS["xsn31/5"], basis="real")
        self.assertGreaterEqual(v, FLOOR, "real-basis predicts a large GREEN")
        self.assertLess(RECORDS["xsn31/5"]["cushion_min_gib"], FLOOR,
                        "reality improved but stayed under the floor")

    def test_xsn31_5_to_6_is_the_named_NONMONOTONIC_case(self):
        """THE ARITHMETIC FINDING (coordinator: 'die Cushion war zweimal
        nicht monoton zur Bounce-Groesse'). Real bounce FELL
        (16.905 -> 12.00 GiB -- the cross-group concurrency gap notwith-
        standing, fewer total simultaneous bytes than xsn31/5's single
        16.905 GiB file). cushion_headroom_gib's own model has exactly ONE
        term (the bounce delta) and holds every other shmem-consuming term
        constant between the two records BY CONSTRUCTION (host_ledger.py,
        function body, 4 lines: `cushion_min_gib - (bounce_now - bounce_
        then)`, no other term enters it) -- so it predicts an IMPROVEMENT
        here. Measured reality went the OTHER way: cushion FELL
        (0.85 -> 0.20). Not a bug in the subtraction -- a documented
        BLIND SPOT: the model has no term for whatever ELSE varied between
        these two boots (the boot record's own conclusion: 'der naechste
        Hebel ist der Host-Ring/die Baseline-Terme, NICHT der Lane-Knopf').
        """
        v = _invariant2(RECORDS["xsn31/5"], RECORDS["xsn31/6"], basis="real")
        self.assertGreaterEqual(v, FLOOR,
                                "the formula predicts GREEN: bounce fell")
        self.assertLess(RECORDS["xsn31/6"]["cushion_min_gib"],
                        RECORDS["xsn31/5"]["cushion_min_gib"],
                        "reality: cushion fell too -- the non-monotonic case")

    def test_xsn31_6_to_7_projected_HiCache_adjusted(self):
        """xsn31/7 never reached a flip (crashed at model load on an
        UNRELATED --speculative-draft-kv-only / --hicache-canonical-kv-page
        incompatibility, server_args.py:8420-8432 -- named in BOOT_weg2xsn31
        _0913_4.md's own VERSUCH 7 section). This is a PROJECTION, not a
        measurement: bounce is held IDENTICAL to xsn31/6 (HiCache does not
        touch lane pricing or the still-open cross-group concurrency gap),
        and the ONLY delta fed to cushion_headroom_gib is BOOT7's own
        measured HiCache saving (-8.17 GiB predicted_run_peak, cross-checked
        against ANALYSE_57GIB_SHMEM_0913.md's independent -8.14 GiB
        estimate). `cushion_headroom_gib` takes no HiCache-specific
        parameter -- it is a generic shmem-currency subtraction, and this
        call uses that generality deliberately (bounce_now=0, bounce_then=
        the saving) rather than reading a new parameter into it."""
        saving = RECORDS["xsn31/7-projected"]["hicache_saving_gib"]
        v = hl.cushion_headroom_gib(
            RECORDS["xsn31/6"]["cushion_min_gib"], 0.0, saving)
        self.assertAlmostEqual(v, 0.20 + 8.17, places=2)
        self.assertGreaterEqual(v, FLOOR)


class CushionHeadroomGibIsWiredIntoChooseAndOnlyTightens(CustomTestCase):
    """HISTORY (2026-09-13, this class's original form, kept for the record
    rather than deleted): "THE WIRING FINDING: `cushion_headroom_gib` --
    the ONLY function in this tree that correctly separates xsn31/2 from
    xsn31/3 -- is never called by `choose()`, by the dry-run FUNDABLE path,
    or by any other ARM-time decision in `host_ledger.py` or `launcher.py`.
    [...] Every xsn31 boot that invariant (2) would have refused pre-flight
    (xsn31/3, /4, /5, /6) was instead allowed to dry-run FUNDABLE and
    discovered the breach LIVE (W98, after VRAM had already been mutated)."

    2026-09-14, commit 695c29dd9e (#1378 Stage 2, W105) CLOSED that gap:
    `choose()` now calls `cushion_headroom_gib` at its ARM point and folds
    the result into the verdict by AND ONLY -- it may refuse an arm the
    other checks funded, it can never fund one they refused (the order's
    own "GRUEN refused nichts und fundiert nichts"). This class inverted
    from the finding to the ratchet on it: the ORIGINAL two tests here
    asserted the ABSENCE of a call site as proof of the gap; deleting them
    once the gap closed would have thrown away the one thing that caught
    the gap in the first place, so instead they now assert the wiring's
    two SPECIFIC shapes -- exactly one production call site, and it binds
    by AND, never OR -- which is exactly what a future refactor could
    silently break without ever touching this file's own assertions.
    """

    def test_exactly_one_production_call_site_and_it_is_inside_choose(self):
        """GENAU EIN Produktions-Call-Site (order 2026-09-14): the def line
        plus exactly one call, and that call must resolve inside `choose`'s
        own source -- not merely somewhere in the module, which would pass
        just as well for a second, looser reader added later beside it."""
        src = inspect.getsource(hl)
        calls = src.count("cushion_headroom_gib(")
        defs = src.count("def cushion_headroom_gib(")
        self.assertEqual(defs, 1)
        self.assertEqual(
            calls, 2,
            "expected exactly the def plus ONE production call site in "
            "host_ledger.py (695c29dd9e wired the one inside choose()); "
            "a THIRD occurrence is a second, looser reader this test does "
            "not yet know about and must be named, not silently accepted")
        choose_src = inspect.getsource(hl.choose)
        self.assertIn(
            "cushion_headroom_gib(", choose_src,
            "the one production call site must be inside choose() itself, "
            "not merely somewhere else in the module")

    def test_choose_reads_cushion_headroom_and_only_tightens_the_verdict(self):
        """Round 1's own mutant is the authority here (test_weg2_cushion_
        headroom_gate_1378_stage2.py::TheGateCanOnlyTightenNeverLoosen):
        removing `and headroom_ok` from `ok = ...` must still kill that
        test. This test only pins the STRUCTURE the mutant depends on --
        the call exists, and it feeds an AND, never an OR or a bare
        replacement of an existing term."""
        src = inspect.getsource(hl.choose)
        self.assertIn("cushion_headroom_gib(", src,
                      "the ARM decision function must consult the one "
                      "invariant that discriminates xsn31/2 from xsn31/3")
        i = src.index("ok = moments_ok and peak_ok and cushion_ok")
        line = src[i:i + 120].splitlines()[0]
        self.assertIn("and headroom_ok", line,
                      "the fold must be conjunctive: it can only ever "
                      "refuse an arm the other checks funded, never fund "
                      "one they refused")


if __name__ == "__main__":
    unittest.main()
