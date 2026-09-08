# SPDX-License-Identifier: Apache-2.0
"""#1284: the host-ring NEED series and the W51 refusal, against weg2sb5e.

THE ROWS THIS FILE IS BUILT ON are the three ``W31 Weg2HostRingExhausted``
lines that ended boot weg2sb5e (2026-09-08, tip ``3ea18deb95``), copied
verbatim from ``boot_weg2_weg2sb5e_3ea18deb95_0908_205950.D.log``::

    card=GPU-31d7ef41 tag=weights_2 need=78 free=44 MiB  waited_ms=110043
    card=GPU-62dbbae1 tag=weights_2 need=20 free= 4 MiB  waited_ms=110246
    card=GPU-5c648f96 tag=weights_2 need=48 free=10 MiB  waited_ms=110148

Each is a per-ALLOCATION acquire inside ``weights_2`` that could not be funded,
after the leg had already placed the earlier tags.  All three ranks sat out the
full ~110 s acquire budget in silence and then exited 1.

WHAT IS ASSERTED, and the danger direction each assertion covers:

* The three rows REFUSE, and refuse BEFORE any waiter is parked -- within
  ``stall_probe_s``, not within the C++ budget.  Danger: a guard that only
  makes the same wedge louder.
* A HEALTHY interleaved leg -- weg2sb5e's own D->P shape, where the leg opens
  ~10 GiB short and the waking peer funds it tag by tag -- is NOT refused.
  Danger direction, and the one that matters most: this shortfall is NORMAL and
  happened on all 30 good flips; a threshold guard would have refused every one
  of them and no boot would ever have run.
* ABSENCE is not zero: a boot with no ring published gets no refusal.
* The refusal carries the WHOLE series, not one row.

MUTANTS.  Four, each a source-level edit of ``ring_guard.py`` re-executed as a
fresh module, each asserting the suite goes RED -- the can-fail proof, because a
guard that cannot be shown to fail has not been shown to test anything.
"""

import itertools
import logging
import os
import re
import sys
import types
import unittest
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import ring_guard
from sglang.srt.weg2.ring_guard import (
    RingNeedGuard,
    Weg2HostRingUnfunded,
    need_granules,
    need_mib,
)
from sglang.test.test_utils import CustomTestCase

MIB = 1024 * 1024
GRANULE = 2 * MIB

#: The three W31 rows, verbatim.  ``need``/``free`` are MiB as the C++ printed
#: them (granules x 2), so they convert back to bytes exactly.
SB5E_W31_ROWS = (
    ("GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d", "weights_2", 78, 44),
    ("GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4", "weights_2", 20, 4),
    ("GPU-5c648f96-be1d-42d5-0221-34d11ab137f7", "weights_2", 48, 10),
)

#: weg2sb5e's D group per-tag bytes on nvml1 (card GPU-31d7ef41), from that
#: boot's ``WEG2-FLIP-TAG group=D ... dir=d2h`` lines.  Every one of these was
#: byte-IDENTICAL across all 15 saves -- the measured answer to "does the need
#: creep": it does not.  Kept here so the healthy-leg case is this boot's real
#: shape and not an invented one.
SB5E_D_TAGS_NVML1 = (
    ("weights_0", 1608), ("weights_1", 1338), ("weights_2", 1352),
    ("weights_3", 1352), ("weights_4", 1352), ("weights_5", 1352),
    ("weights_6", 1352), ("weights_7", 1352), ("weights", 2856),
)

#: Ring H and the free the D leg opens against on that card: ring 17541 MiB
#: total, P's parked image 13860 MiB, so 3681 MiB free at leg start.
SB5E_NVML1_FREE_AT_LEG_START = 3681


class FakeClock:
    """A clock that only moves when someone sleeps.

    Makes the timing assertions EXACT rather than flaky: "refused after 2.0 s
    of guard time" is a statement about the guard's own arithmetic, not about
    how loaded the box was.
    """

    def __init__(self):
        self.t = 0.0
        self.sleeps = 0

    def __call__(self):
        return self.t

    def sleep(self, dt):
        self.t += dt
        self.sleeps += 1


def stats(free_mib, granule_bytes=GRANULE, total_mib=20000):
    """A ``ring_stats()`` dict with ``free_mib`` free."""
    return {
        "granule_bytes": granule_bytes,
        "granules_total": total_mib * MIB // granule_bytes,
        "granules_free": free_mib * MIB // granule_bytes,
    }


def flat_free(free_mib):
    """A peer that releases NOTHING -- the weg2sb5e wedge."""
    return lambda: stats(free_mib)


def rising_free(start_mib, step_mib):
    """A peer that releases ``step_mib`` per slice -- a healthy interleave."""
    box = {"v": start_mib}

    def fn():
        out = stats(box["v"])
        box["v"] += step_mib
        return out

    return fn


def guard(card="GPU-test", **kw):
    clock = FakeClock()
    kw.setdefault("clock", clock)
    kw.setdefault("sleep", clock.sleep)
    kw.setdefault("log", logging.getLogger("weg2.ring_guard.test"))
    g = RingNeedGuard(card, group="D", rank=0, **kw)
    return g, clock


class TestRingGuardOnSb5eRows(CustomTestCase):
    """The three W31 rows must refuse, and refuse early."""

    def test_sb5e_rows_refuse_before_the_cpp_budget(self):
        for card, tag, need, free in SB5E_W31_ROWS:
            with self.subTest(card=card):
                g, clock = guard(card, stall_probe_s=2.0, slice_s=0.25)
                with self.assertRaises(Weg2HostRingUnfunded) as cm:
                    g.guard_tag(tag, need * MIB, flat_free(free))
                # BEFORE any waiter is parked: the guard's own elapsed time is
                # the stall probe, not the 110 s the C++ acquire spent.
                self.assertLess(clock.t, 3.0, "guard sat longer than its probe")
                self.assertGreaterEqual(clock.t, 2.0, "refused before probing")
                msg = str(cm.exception)
                self.assertIn("W51 Weg2HostRingUnfunded", msg)
                self.assertIn(f"card={card}", msg)
                self.assertIn(f"tag={tag}", msg)
                self.assertIn(f"need_mib={need}", msg)
                self.assertIn(f"free_mib={free}", msg)
                self.assertIn(f"delta_mib={need - free}", msg)

    def test_refusal_carries_the_whole_series_not_one_row(self):
        g, _ = guard(stall_probe_s=2.0, slice_s=0.25)
        # Two tags that fit, then the sb5e row that does not.
        g.record("weights_0", 100, 900)
        g.record("weights_1", 100, 800)
        with self.assertRaises(Weg2HostRingUnfunded) as cm:
            g.guard_tag("weights_2", 78 * MIB, flat_free(44))
        msg = str(cm.exception)
        for tag in ("weights_0", "weights_1", "weights_2"):
            self.assertIn(f"tag={tag}", msg)
        self.assertIn("need series for this leg", msg)

    def test_no_waiter_is_parked_on_refusal(self):
        """The refusal happens BEFORE ``pause(tag)`` -- nothing was acquired.

        Modelled by counting pauses in a leg driver that mirrors the
        weight_updater loop: guard, then pause.  The wedged tag must never be
        paused, so its device bytes are still mapped when the leg unwinds.
        """
        paused = []
        g, _ = guard(stall_probe_s=2.0, slice_s=0.25)
        with self.assertRaises(Weg2HostRingUnfunded):
            for tag, mib in (("weights_0", 10), ("weights_1", 10), ("weights_2", 78)):
                g.guard_tag(tag, mib * MIB, flat_free(44))
                paused.append(tag)
        self.assertEqual(paused, ["weights_0", "weights_1"])


class TestHealthyInterleaveIsNotRefused(CustomTestCase):
    """The danger direction: refusing the 30 flips that worked."""

    def test_sb5e_healthy_leg_completes(self):
        """weg2sb5e's real D leg on nvml1: opens 10 GiB short, is funded."""
        total = sum(mib for _, mib in SB5E_D_TAGS_NVML1)
        self.assertGreater(total, SB5E_NVML1_FREE_AT_LEG_START,
                           "the leg must open SHORT or this proves nothing")
        free = {"v": SB5E_NVML1_FREE_AT_LEG_START}

        def peer_releases():
            out = stats(free["v"])
            free["v"] += 400          # the waking peer, tag by tag
            return out

        g, clock = guard(stall_probe_s=2.0, slice_s=0.25)
        for tag, mib in SB5E_D_TAGS_NVML1:
            g.guard_tag(tag, mib * MIB, peer_releases)
            free["v"] -= mib          # this tag's acquire takes its granules
            free["v"] = max(free["v"], 0)
        self.assertGreater(len(g.samples), len(SB5E_D_TAGS_NVML1))

    def test_fits_now_never_sleeps(self):
        g, clock = guard()
        g.guard_tag("weights_0", 100 * MIB, flat_free(900))
        self.assertEqual(clock.sleeps, 0)
        self.assertEqual(clock.t, 0.0)

    def test_slow_but_progressing_peer_is_not_refused(self):
        g, _ = guard(stall_probe_s=2.0, slice_s=0.25, max_wait_s=20.0)
        # 60 MiB per slice: slower than the probe window is long, but it never
        # stalls, so it must be allowed to finish.
        g.guard_tag("weights_2", 600 * MIB, rising_free(100, 60))


class TestAbsenceIsNotZero(CustomTestCase):
    def test_no_ring_published_stands_down(self):
        g, clock = guard()
        g.guard_tag("weights_2", 78 * MIB, lambda: None)
        self.assertEqual(g.samples, [])
        self.assertEqual(clock.sleeps, 0)

    def test_stats_without_counters_stands_down(self):
        g, _ = guard()
        g.guard_tag("weights_2", 78 * MIB, lambda: {"card_uuid": "x"})
        self.assertEqual(g.samples, [])

    def test_ring_vanishing_mid_wait_is_not_our_verdict(self):
        seq = [stats(4), stats(4), None]
        g, _ = guard(stall_probe_s=5.0, slice_s=0.25)
        g.guard_tag("weights_2", 78 * MIB, lambda: seq.pop(0) if seq else None)


class TestNeedLineShape(CustomTestCase):
    """The per-flip line the next boot's series is read from."""

    FIELDS = ("tag=", "card=", "need_mib=", "free_mib=", "delta_mib=")

    def test_line_carries_the_five_fixed_fields_in_order(self):
        log = logging.getLogger("weg2.ring_guard.shape")
        with self.assertLogs(log, level="INFO") as cap:
            g = RingNeedGuard("GPU-abc", group="D", rank=1, log=log)
            g.record("weights_2", 1352, 3681)
        line = [ln for ln in cap.output if "WEG2-RING NEED" in ln][0]
        head = line.split("WEG2-RING NEED", 1)[1]
        pos = [head.index(f) for f in self.FIELDS]
        self.assertEqual(pos, sorted(pos), f"fields out of order in {head!r}")
        self.assertIn("need_mib=1352", head)
        self.assertIn("free_mib=3681", head)
        self.assertIn("delta_mib=-2329", head)

    def test_absence_line_says_n_a_never_zero(self):
        log = logging.getLogger("weg2.ring_guard.shape2")
        with self.assertLogs(log, level="INFO") as cap:
            g = RingNeedGuard("GPU-abc", log=log)
            g.guard_tag("weights_2", 78 * MIB, lambda: None)
        line = [ln for ln in cap.output if "WEG2-RING NEED" in ln][0]
        self.assertIn("free_mib=n/a", line)
        self.assertNotIn("free_mib=0", line)


class TestGranuleRounding(CustomTestCase):
    def test_rounds_up_like_the_ring(self):
        self.assertEqual(need_granules(1), 1)
        self.assertEqual(need_granules(GRANULE), 1)
        self.assertEqual(need_granules(GRANULE + 1), 2)
        self.assertEqual(need_granules(0), 0)

    def test_need_mib_matches_the_w31_rows(self):
        for _card, _tag, need, _free in SB5E_W31_ROWS:
            self.assertEqual(need_mib(need * MIB), need)


class TestCeiling(CustomTestCase):
    def test_a_peer_dribbling_one_granule_hits_the_ceiling(self):
        """Bounds the guard itself: never anywhere near the 110 s budget."""
        g, clock = guard(stall_probe_s=2.0, slice_s=0.25, max_wait_s=5.0)
        with self.assertRaises(Weg2HostRingUnfunded) as cm:
            g.guard_tag("weights_2", 100000 * MIB, rising_free(10, 2))
        self.assertLessEqual(clock.t, 5.5)
        self.assertIn("ceiling", str(cm.exception))


# ---------------------------------------------------------------------------
# MUTANTS -- the can-fail proof.
# ---------------------------------------------------------------------------

SOURCE = Path(ring_guard.__file__).read_text()

#: ``(name, pattern, replacement, why)``.  Every one is on the DANGER
#: direction: three make the guard miss the wedge, one makes it refuse a
#: healthy flip.  A mutant that does not change behaviour is a test gap, so
#: each is asserted to actually change the substitution count too.
MUTANTS = (
    (
        "no-trend-test",
        "if now - last_progress >= self.stall_probe_s:",
        "if False:",
        "drops the stall probe -- the guard then waits out its ceiling exactly "
        "as the C++ acquire waited out its budget",
    ),
    (
        "flat-counts-as-progress",
        "if free_now > best:",
        "if free_now >= best:",
        "treats an unchanged free as funding -- the weg2sb5e wedge never refuses",
    ),
    (
        "refuse-on-shortfall-alone",
        "        if sample.fits:\n            return",
        "        if not sample.fits:\n            self._refuse(tag, want, free, '', 0.0, 'shortfall')\n        return",
        "refuses on need > free -- would have killed all 30 good flips",
    ),
    (
        "absence-reads-as-zero",
        "        if not stats:\n            return None",
        "        if not stats:\n            return 0",
        "turns 'no ring published' into 'zero free' -- refuses every non-ring boot",
    ),
)


_MUTANT_SEQ = itertools.count()


def load_mutant(pattern, replacement):
    """Re-exec ``ring_guard.py`` with one edit applied, as a fresh module.

    THE MODULE MUST BE IN ``sys.modules`` BEFORE THE EXEC.  ``ring_guard`` uses
    ``from __future__ import annotations``, so every dataclass field annotation
    is a STRING, and ``dataclasses._process_class`` resolves those by looking
    the class's own module up in ``sys.modules`` -- an unregistered module makes
    it dereference ``None`` and the whole mutant fails to import for a reason
    that has nothing to do with the mutation.  Measured on the first remote run
    of this file (4 failed, all four mutants, same AttributeError).
    """
    assert SOURCE.count(pattern) == 1, f"mutant pattern not unique: {pattern!r}"
    name = f"ring_guard_mutant_{next(_MUTANT_SEQ)}"
    mod = types.ModuleType(name)
    mod.__file__ = ring_guard.__file__
    sys.modules[name] = mod
    try:
        exec(compile(SOURCE.replace(pattern, replacement), "<mutant>", "exec"),
             mod.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


class TestMutantsGoRed(CustomTestCase):
    def _guard(self, mod, **kw):
        clock = FakeClock()
        kw.setdefault("clock", clock)
        kw.setdefault("sleep", clock.sleep)
        kw.setdefault("log", logging.getLogger("weg2.ring_guard.mutant"))
        return mod.RingNeedGuard("GPU-31d7ef41", group="D", rank=0, **kw), clock

    def test_every_mutant_pattern_is_present_exactly_once(self):
        for name, pattern, _repl, _why in MUTANTS:
            with self.subTest(mutant=name):
                self.assertEqual(SOURCE.count(pattern), 1)

    def test_mutant_no_trend_test_loses_the_early_refusal(self):
        mod = load_mutant(MUTANTS[0][1], MUTANTS[0][2])
        g, clock = self._guard(mod, stall_probe_s=2.0, slice_s=0.25, max_wait_s=20.0)
        with self.assertRaises(mod.Weg2HostRingUnfunded):
            g.guard_tag("weights_2", 78 * MIB, flat_free(44))
        # The live guard refuses at 2.0 s; this one only at the ceiling.
        self.assertGreaterEqual(clock.t, 20.0)

    def test_mutant_flat_counts_as_progress_never_refuses_sb5e(self):
        mod = load_mutant(MUTANTS[1][1], MUTANTS[1][2])
        g, clock = self._guard(mod, stall_probe_s=2.0, slice_s=0.25, max_wait_s=20.0)
        with self.assertRaises(mod.Weg2HostRingUnfunded) as cm:
            g.guard_tag("weights_2", 78 * MIB, flat_free(44))
        # Not the stall verdict any more -- it ran to the ceiling instead.
        self.assertIn("ceiling", str(cm.exception))
        self.assertGreaterEqual(clock.t, 20.0)

    def test_mutant_refuse_on_shortfall_alone_kills_a_healthy_leg(self):
        """The discriminating case: SHORT at first read, then funded.

        A tag that already fits cannot tell the two apart -- neither guard
        refuses it -- so this uses weg2sb5e's own mid-leg shape instead:
        ``weights_3`` (1352 MiB) meeting 600 MiB of free while the waking peer
        is still releasing.  The live guard waits and proceeds; the mutant
        refuses on the shortfall alone, which is what would have killed all 30
        good flips.  (First remote run of this file used ``weights_0`` against
        3681 MiB free -- it FITS, the mutant never fired, and the test passed
        the mutant it was written to catch.)
        """
        mod = load_mutant(MUTANTS[2][1], MUTANTS[2][2])
        g, _ = self._guard(mod, stall_probe_s=2.0, slice_s=0.25)
        with self.assertRaises(mod.Weg2HostRingUnfunded):
            g.guard_tag("weights_3", 1352 * MIB, rising_free(600, 400))
        # The live guard must NOT refuse the same leg.
        live, clock = guard(stall_probe_s=2.0, slice_s=0.25)
        live.guard_tag("weights_3", 1352 * MIB, rising_free(600, 400))
        self.assertGreater(clock.sleeps, 0, "the live guard must have WAITED, "
                                            "or this proves nothing")

    def test_mutant_absence_reads_as_zero_refuses_a_ringless_boot(self):
        mod = load_mutant(MUTANTS[3][1], MUTANTS[3][2])
        g, _ = self._guard(mod, stall_probe_s=2.0, slice_s=0.25)
        with self.assertRaises(mod.Weg2HostRingUnfunded):
            g.guard_tag("weights_2", 78 * MIB, lambda: None)
        # The live guard stands down instead.
        live, clock = guard()
        live.guard_tag("weights_2", 78 * MIB, lambda: None)
        self.assertEqual(clock.sleeps, 0)


class TestWiredIntoTheSleepingLeg(CustomTestCase):
    """PRESENT-AND-WIRED, not merely present (#859's three states).

    A guard module that nothing imports is the failure mode this codebase has
    hit repeatedly, so the wiring is asserted at the call site rather than
    assumed from the import.
    """

    def test_weight_updater_guards_before_the_pause(self):
        from sglang.srt.managers.scheduler_components import weight_updater

        src = Path(weight_updater.__file__).read_text()
        self.assertIn("RingNeedGuard(", src)
        self.assertIn("weg2_ring_guard.guard_tag(", src)
        block = src.split('direction="d2h"')[1]
        guard_at = block.index("weg2_ring_guard.guard_tag(")
        pause_at = block.index("self.memory_saver_adapter.pause(tag)")
        self.assertLess(guard_at, pause_at,
                        "the guard must run BEFORE the pause that acquires")

    def test_w51_code_is_unassigned_elsewhere(self):
        """W31 named two different things once; the number is enumerated."""
        root = Path(ring_guard.__file__).resolve().parents[3]
        used = set()
        pat = re.compile(r"\b(W\d{1,2}[a-z]?)\s+(Weg2[A-Za-z0-9_]+)")
        for sub in ("srt/weg2", "srt/managers", "srt/mem_cache"):
            d = root / "sglang" / sub
            if not d.is_dir():
                continue
            for p in d.rglob("*.py"):
                for code, name in pat.findall(p.read_text(errors="replace")):
                    used.add((code, name))
        w51 = {name for code, name in used if code == "W51"}
        self.assertEqual(w51, {"Weg2HostRingUnfunded"},
                         f"W51 is not exclusively ours: {w51}")


if __name__ == "__main__":
    unittest.main()
