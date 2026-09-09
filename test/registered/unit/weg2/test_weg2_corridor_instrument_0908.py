# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The Weg-2 corridor sampler measures ALLOCATABLE free, and says so.

THE DEFECT (boot weg2rg6, 2026-09-08).  ``front._nvml_free()`` shelled out to
``nvidia-smi --query-gpu=index,uuid,memory.used,memory.total`` and returned
``total - used``.  nvidia-smi's ``memory.used`` -- like NVML's v2 ``used`` --
already EXCLUDES the driver's RM carve-out, so that subtraction returns FREE
PLUS THE CARVE-OUT.  Measured at ONE instant, 07:31:30Z, against
``memory.free``: nvml0 1454 vs 1030, nvml1 859 vs 341, nvml2 1276 vs 852 --
overstatements of exactly 424 / 518 / 424 MiB.  The 5090 was 478 MiB BELOW the
corridor floor while the operator's line showed a number inside the band.

THE FAKE DRIVER.  Every NVML test here runs through the real
``registry.nvml`` code path with ``_pynvml`` swapped for :class:`FakeNvml`,
whose two memory structs are built the way the driver builds them:

* v1 ``nvmlMemory_t``: ``used = total - free``, i.e. the carve-out is INSIDE
  ``used`` and ``free`` is already allocatable.
* v2 ``nvmlMemory_v2``: ``used = total - free - reserved``, i.e. the carve-out
  is its own field and out of ``used``; ``free`` is the same figure as v1.

That is not an assumption: :class:`FakeDriverMatchesMetalTest` pins the fake
against the numbers the real driver returned on this rig on 2026-09-08 with
nothing running.  ``nvidia-smi`` printed ``total 20480 used 1 free 20055
reserved 426`` for an RTX 3080; ``pynvml`` on the same idle card printed v1
``total 20480 free 20054 used 425`` and v2 ``used 0 reserved 425``.  The 1 MiB
spread between the two tools is byte-level rounding (neither ``free`` nor
``reserved`` is a round MiB), so the fake works in exact MiB and models an idle
card as ``free = total - reserved`` -- which reproduces v1 ``used`` = 425 and
v2 ``used`` = 0 exactly.  A fake that got the two structs backwards would make
every assertion below pass against the defect.
"""

from __future__ import annotations

import ast
import logging
import os
import tempfile
import unittest

from sglang.srt.managers import corridor_guard
from sglang.srt.registry import nvml as nvml_registry
from sglang.srt.weg2 import corridor_arm, front, launcher, ring_table

MIB = 1024 * 1024

#: The rg6 instant, 07:31:30Z: (nvml index, allocatable free MiB, carve-out MiB,
#: and what the pre-fix `total - used` line printed for it).  The last column is
#: not derived here -- it is quoted from the boot log.
RG6_INSTANT = (
    (0, 1030, 425, 1454),
    (1, 341, 518, 859),
    (2, 852, 425, 1276),
)

CARD_TOTAL_MIB = {0: 20480, 1: 32607, 2: 20480}
CARD_NAME = {0: "NVIDIA GeForce RTX 3080", 1: "NVIDIA GeForce RTX 5090", 2: "NVIDIA GeForce RTX 3080"}


class _V1:
    """``nvmlMemory_t``: used INCLUDES the carve-out, free EXCLUDES it."""

    def __init__(self, total_b: int, free_b: int):
        self.total = total_b
        self.free = free_b
        self.used = total_b - free_b


class _V2:
    """``nvmlMemory_v2``: reserved is its own field and is OUT of used."""

    def __init__(self, total_b: int, free_b: int, reserved_b: int):
        self.version = 2
        self.total = total_b
        self.free = free_b
        self.reserved = reserved_b
        self.used = total_b - free_b - reserved_b


class _Pci:
    def __init__(self, index: int):
        self.busId = f"0000:0{index}:00.0"


class FakeNvml:
    """The subset of ``pynvml`` the registry touches, driver-accurate.

    ``cards`` is ``{index: (total_mib, free_mib, reserved_mib)}``.  The class
    attribute ``nvmlMemory_v2`` is what the registry probes to decide whether
    the v2 struct exists at all, so its presence here is load-bearing.
    """

    nvmlMemory_v2 = 2

    def __init__(self, cards):
        self.cards = dict(cards)
        self.calls = 0
        self.init_calls = 0

    def nvmlInit(self):
        self.init_calls += 1

    def nvmlShutdown(self):
        pass

    def nvmlDeviceGetCount(self):
        return len(self.cards)

    def nvmlDeviceGetHandleByIndex(self, i):
        return i

    def nvmlDeviceGetUUID(self, h):
        return f"GPU-fake{h:04d}-0000-0000-0000-000000000000".encode()

    def nvmlDeviceGetName(self, h):
        return CARD_NAME[h].encode()

    def nvmlDeviceGetPciInfo(self, h):
        return _Pci(h)

    def nvmlDeviceGetMemoryInfo(self, h, version=None):
        self.calls += 1
        total, free, reserved = self.cards[h]
        if version is None:
            return _V1(total * MIB, free * MIB)
        return _V2(total * MIB, free * MIB, reserved * MIB)


def _fake(cards):
    return FakeNvml(cards)


def _rig(free_by_index):
    """``{index: (total, free, reserved)}`` for the reference rig."""
    return {
        i: (CARD_TOTAL_MIB[i], free, res)
        for i, free, res in (
            (i, free_by_index[i], 425 if i != 1 else 518) for i in sorted(free_by_index)
        )
    }


#: Every card idle: free = total - carve-out, nothing held by any process.
IDLE_FREE = {i: CARD_TOTAL_MIB[i] - (518 if i == 1 else 425) for i in CARD_TOTAL_MIB}


class _PatchNvml:
    """Swap ``registry.nvml._pynvml`` for a fake, restore on exit."""

    def __init__(self, fake):
        self.fake = fake

    def __enter__(self):
        self._orig = nvml_registry._pynvml
        nvml_registry._pynvml = lambda: self.fake
        return self.fake

    def __exit__(self, *exc):
        nvml_registry._pynvml = self._orig
        return False


class FakeDriverMatchesMetalTest(unittest.TestCase):
    """The fake reproduces what the real driver returned on this rig.

    Indicator law: an instrument test is worth nothing until the instrument is
    shown to measure what it claims.  Numbers below are from
    ``pynvml`` + ``nvidia-smi`` on 2026-09-08 with the cards idle.
    """

    #: An idle RTX 3080: total, free, carve-out (MiB).
    IDLE_3080 = (20480, 20480 - 425, 425)

    def test_v1_used_includes_the_carve_out(self):
        total, free, res = self.IDLE_3080
        f = _V1(total * MIB, free * MIB)
        self.assertEqual(f.used // MIB, res)

    def test_v2_used_excludes_the_carve_out(self):
        total, free, res = self.IDLE_3080
        f = _V2(total * MIB, free * MIB, res * MIB)
        self.assertEqual(f.used // MIB, 0)
        self.assertEqual(f.reserved // MIB, res)

    def test_total_minus_used_over_a_v2_used_is_free_plus_the_carve_out(self):
        """The defect, in arithmetic. nvidia-smi's memory.used is the v2 one."""
        total, free, res = self.IDLE_3080
        v2 = _V2(total * MIB, free * MIB, res * MIB)
        self.assertEqual((v2.total - v2.used) // MIB, free + res)


class RegistryReaderTest(unittest.TestCase):
    def test_snapshot_free_is_allocatable_and_reserved_is_carried(self):
        with _PatchNvml(_fake(_rig({0: 1030, 1: 341, 2: 852}))):
            snap = nvml_registry.memory_snapshot()
        got = {d.index: (m.free_mib, m.reserved_mib) for d, m in snap}
        self.assertEqual(got, {0: (1030, 425), 1: (341, 518), 2: (852, 425)})

    def test_snapshot_never_returns_free_plus_carve_out(self):
        with _PatchNvml(_fake(_rig({0: 1030, 1: 341, 2: 852}))):
            snap = nvml_registry.memory_snapshot()
        for idx, free, _res, prefix_value in RG6_INSTANT:
            mem = dict((d.index, m) for d, m in snap)[idx]
            self.assertEqual(mem.free_mib, free)
            self.assertNotEqual(
                mem.free_mib, prefix_value,
                f"nvml{idx} read {prefix_value} MiB -- that is the pre-fix total-used value",
            )

    def test_tenant_used_excludes_the_carve_out(self):
        """#539: a v1 `used` read as tenancy calls an idle card 425 MiB busy."""
        with _PatchNvml(_fake(_rig(IDLE_FREE))):
            snap = nvml_registry.memory_snapshot()
        for dev, mem in snap:
            self.assertEqual(mem.tenant_used_mib, 0, dev.index)
            self.assertEqual(mem.used_bytes // MIB, mem.reserved_mib, dev.index)

    def test_one_nvml_session_for_the_whole_rig(self):
        """All cards from one instant: three sessions would be three instants."""
        with _PatchNvml(_fake(_rig({0: 1030, 1: 341, 2: 852}))) as fake:
            nvml_registry.memory_snapshot()
        self.assertEqual(fake.init_calls, 1)


class CorridorVerdictTest(unittest.TestCase):
    def test_band_edges_are_inclusive(self):
        self.assertEqual(front.corridor_verdict(819), "IN")
        self.assertEqual(front.corridor_verdict(1229), "IN")

    def test_below_the_floor(self):
        self.assertEqual(front.corridor_verdict(818), "BELOW")
        self.assertEqual(front.corridor_verdict(341), "BELOW")

    def test_above_the_ceiling(self):
        self.assertEqual(front.corridor_verdict(1230), "ABOVE")

    def test_band_is_the_corridor_law(self):
        """FIX 2, finding 2: the band is READ from its one declaration.

        The predecessor asserted ``front.CORRIDOR_FLOOR_MIB == 819`` -- which
        pinned the private copy rather than the law, and would have stayed
        green while the guard's law moved underneath it.  Same default, but
        now the identity is what is asserted; the override tests below are
        what make it a fix rather than a rename.
        """
        self.assertEqual(front.corridor_band_mib(), (819, 1229))
        self.assertEqual(
            front.corridor_band_mib(),
            (corridor_guard.corridor_band_floor_mib(), corridor_guard.corridor_band_ceiling_mib()),
        )


def _front():
    return front.Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0)


class CorridorLineTest(unittest.TestCase):
    def _line(self, free_by_index, awake="D"):
        f = _front()
        f.awake = awake
        with _PatchNvml(_fake(_rig(free_by_index))):
            return f, f.corridor_sample()

    def test_prints_allocatable_free_not_total_minus_used(self):
        """The rg6 instant. 341, never 859."""
        _f, line = self._line({0: 1030, 1: 341, 2: 852})
        for idx, free, res, prefix_value in RG6_INSTANT:
            self.assertIn(f"nvml{idx}:free={free}MiB", line)
            self.assertIn(f"reserved={res}MiB", line)
            self.assertNotIn(f"nvml{idx}:free={prefix_value}MiB", line)

    def test_names_its_instrument(self):
        _f, line = self._line({0: 1030, 1: 341, 2: 852})
        self.assertIn("instrument=nvml_v2_free,allocatable", line)

    def test_prints_the_band_it_grades_against(self):
        _f, line = self._line({0: 1030, 1: 341, 2: 852})
        self.assertIn("band=819-1229MiB", line)

    #: #1257c: the per-card field gained ``floor=``/``source=``/``reserve=``
    #: between the free reading and the verdict, because the floor is DERIVED
    #: per card now and a reader must be able to see which one graded this
    #: card.  On a rig with no measured footprint and no user reserve the
    #: values are the named fallback, so the VERDICTS below are byte-identical
    #: to the pre-#1257c ones -- which is the point of pinning them here.
    # #1257c refuter fix 1: the segment carries BOTH numbers -- the floor
    # and the number the ``verdict=`` beside it is actually graded
    # against. Pinning only ``floor=`` is how the front and the arm came
    # to contradict each other on the same line.
    FALLBACK_FIELDS = (
        "floor=1024MiB verdict_floor=819MiB "
        "source=UNMEASURED-FALLBACK reserve=0MiB"
    )

    def test_verdict_per_card(self):
        _f, line = self._line({0: 1030, 1: 341, 2: 852})
        f = self.FALLBACK_FIELDS
        self.assertIn(f"nvml0:free=1030MiB reserved=425MiB {f} verdict=IN", line)
        self.assertIn(f"nvml1:free=341MiB reserved=518MiB {f} verdict=BELOW", line)
        self.assertIn(f"nvml2:free=852MiB reserved=425MiB {f} verdict=IN", line)

    def test_verdict_flips_at_the_floor(self):
        f = self.FALLBACK_FIELDS
        _f, low = self._line({0: 818, 1: 818, 2: 818})
        self.assertIn(f"nvml0:free=818MiB reserved=425MiB {f} verdict=BELOW", low)
        _f, edge = self._line({0: 819, 1: 819, 2: 819})
        self.assertIn(f"nvml0:free=819MiB reserved=425MiB {f} verdict=IN", edge)

    def test_verdict_flips_at_the_ceiling(self):
        f = self.FALLBACK_FIELDS
        _f, edge = self._line({0: 1229, 1: 1229, 2: 1229})
        self.assertIn(f"nvml0:free=1229MiB reserved=425MiB {f} verdict=IN", edge)
        _f, high = self._line({0: 1230, 1: 1230, 2: 1230})
        self.assertIn(f"nvml0:free=1230MiB reserved=425MiB {f} verdict=ABOVE", high)

    def test_the_rg6_5090_would_have_been_graded_BELOW(self):
        """The whole point: 859 grades IN, 341 grades BELOW, same instant."""
        self.assertEqual(front.corridor_verdict(859), "IN")
        self.assertEqual(front.corridor_verdict(341), "BELOW")

    def test_min_so_far_is_in_the_same_unit_and_says_so(self):
        f = _front()
        with _PatchNvml(_fake(_rig({0: 1030, 1: 341, 2: 852}))):
            f.corridor_sample()
        with _PatchNvml(_fake(_rig({0: 900, 1: 900, 2: 900}))):
            line = f.corridor_sample()
        self.assertEqual(f.corridor_min["D"], {0: 900, 1: 341, 2: 852})
        self.assertIn("min_so_far={0: 900, 1: 341, 2: 852}", line)
        self.assertIn("(nvml_v2_free,allocatable, MiB)", line)

    def test_one_read_per_sample(self):
        """Printed numbers and min_so_far must be ONE instant, not two reads."""
        with _PatchNvml(_fake(_rig({0: 1030, 1: 341, 2: 852}))) as fake:
            _front().corridor_sample()
        self.assertEqual(fake.init_calls, 1)

    def test_sample_is_logged(self):
        f = _front()
        with _PatchNvml(_fake(_rig({0: 1030, 1: 341, 2: 852}))):
            with self.assertLogs("weg2.front", level=logging.INFO) as cap:
                line = f.corridor_sample()
        self.assertIn(line, "\n".join(cap.output))

    def test_phase_is_the_awake_group(self):
        _f, line = self._line({0: 1030, 1: 341, 2: 852}, awake="P")
        self.assertIn("WEG2-CORRIDOR phase=P(awake)", line)

    def test_no_nvml_no_sample_and_no_fallback(self):
        """A dead NVML yields None -- never a total-used fallback line."""
        f = _front()
        front._nvml_unavailable_logged = False
        orig = nvml_registry.memory_snapshot
        nvml_registry.memory_snapshot = lambda: (_ for _ in ()).throw(RuntimeError("no driver"))
        try:
            with self.assertLogs("weg2.front", level=logging.WARNING) as cap:
                self.assertIsNone(f.corridor_sample())
        finally:
            nvml_registry.memory_snapshot = orig
            front._nvml_unavailable_logged = False
        self.assertEqual(f.corridor_min["D"], {})
        self.assertIn("no corridor samples", "\n".join(cap.output).lower())


class RingTableStillParsesTheLineTest(unittest.TestCase):
    """The new line must stay readable by the ring table's own parser."""

    def _log(self, line):
        fd, path = tempfile.mkstemp(prefix="weg2corridor-", suffix=".front.log")
        with os.fdopen(fd, "w") as f:
            f.write("[2026-09-08 07:31:30,000] INFO weg2.front: " + line + "\n")
        self.addCleanup(os.unlink, path)
        return path

    def test_free_and_phase_still_parse(self):
        f = _front()
        with _PatchNvml(_fake(_rig({0: 1030, 1: 341, 2: 852}))):
            line = f.corridor_sample()
        got = ring_table.parse_front_corridor(self._log(line))
        self.assertEqual(got, {"D": {0: 1030, 1: 341, 2: 852}})

    def test_reserved_field_is_not_mistaken_for_free(self):
        f = _front()
        with _PatchNvml(_fake(_rig({0: 1030, 1: 341, 2: 852}))):
            line = f.corridor_sample()
        got = ring_table.parse_front_corridor(self._log(line))
        self.assertNotIn(425, got["D"].values())
        self.assertNotIn(518, got["D"].values())

    def test_instrument_is_reported_for_a_post_fix_log(self):
        f = _front()
        with _PatchNvml(_fake(_rig({0: 1030, 1: 341, 2: 852}))):
            line = f.corridor_sample()
        self.assertEqual(
            ring_table.front_corridor_instrument(self._log(line)),
            "nvml_v2_free,allocatable",
        )

    def test_instrument_is_reported_for_a_pre_fix_log(self):
        pre = (
            "WEG2-CORRIDOR phase=D(awake) epoch=20 nvml0:free=1454MiB "
            "nvml1:free=859MiB nvml2:free=1276MiB min_so_far={0: 1454}"
        )
        self.assertEqual(
            ring_table.front_corridor_instrument(self._log(pre)),
            ring_table.CORRIDOR_INSTRUMENT_PRE_FIX,
        )

    def test_a_log_without_corridor_samples_says_so(self):
        path = self._log("WEG2-FRONT up tag=x")
        self.assertEqual(ring_table.front_corridor_instrument(path), "no WEG2-CORRIDOR samples")


class RealBootLogsAreProvenPreFixTest(unittest.TestCase):
    """The three boots this fix re-derives are all in the pre-fix unit.

    Evidence-tree bound: skipped where ``/spinning/evidence-665-f1`` is absent.
    """

    EVIDENCE = "/spinning/evidence-665-f1"
    STEMS = {
        "rg3": "boot_weg2_weg2rg3_5b015ad139_0908_041053",
        "rg5": "boot_weg2_weg2rg5_15a46a611a_0908_050519",
        "rg6": "boot_weg2_weg2rg6_7f88b1c75d_0908_070324",
    }

    def _front_log(self, key):
        p = os.path.join(self.EVIDENCE, self.STEMS[key] + ".front.log")
        if not os.path.exists(p):
            self.skipTest(f"evidence tree absent: {p}")
        return p

    def test_all_three_boots_are_pre_fix(self):
        for key in self.STEMS:
            self.assertEqual(
                ring_table.front_corridor_instrument(self._front_log(key)),
                ring_table.CORRIDOR_INSTRUMENT_PRE_FIX,
                key,
            )

    def test_rg6_minimum_in_front_units_matches_the_recorded_finding(self):
        """rg6's printed min on the 5090 was 859; the truth at that instant 341."""
        got = ring_table.parse_front_corridor(self._front_log("rg6"))
        self.assertEqual(got["D"][1], 859)
        self.assertEqual(got["D"][1] - 518, 341)
        self.assertEqual(front.corridor_verdict(859), "IN")
        self.assertEqual(front.corridor_verdict(859 - 518), "BELOW")

    def test_rg5_front_minimum_minus_carve_out_is_the_samplers_161(self):
        """679 (front) - 518 (carve-out) = 161, the R4 sampler's memory.free."""
        got = ring_table.parse_front_corridor(self._front_log("rg5"))
        self.assertEqual(got["D"][1], 679)
        self.assertEqual(got["D"][1] - 518, 161)

    def test_no_weg2_module_queries_nvidia_smi_for_card_memory(self):
        """The deletion target is gone, not merely bypassed.

        AST, not grep: every prose mention of ``memory.used`` in this tree is
        an explanation of the defect (this file included), and a text scan
        cannot tell those from a live query -- it flagged six of its own
        comments on the first run.  Only ``--query-gpu=`` string LITERALS that
        ask for card memory are executable second readers.
        ``--query-compute-apps=`` is deliberately not swept: per-process
        attribution is a different question and NVML answers it per PID, not
        per card.
        """
        weg2 = os.path.dirname(os.path.abspath(front.__file__))
        hits = []
        for name in sorted(os.listdir(weg2)):
            if not name.endswith(".py"):
                continue
            path = os.path.join(weg2, name)
            with open(path, errors="replace") as f:
                tree = ast.parse(f.read(), filename=path)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                if node.value.startswith("--query-gpu=") and "memory" in node.value:
                    hits.append(f"{name}:{node.lineno}: {node.value}")
        self.assertEqual(hits, [], "a second NVML card-memory reader survives")


class LauncherPreflightTest(unittest.TestCase):
    #: card 0 with a 2000 MiB foreign tenant on it, the others idle.
    BUSY_FREE = {**IDLE_FREE, 0: IDLE_FREE[0] - 2000}

    def _cards(self):
        with _PatchNvml(_fake(_rig(IDLE_FREE))):
            return launcher.resolve_cards()

    def test_resolve_cards_carries_the_carve_out(self):
        cards = self._cards()
        self.assertEqual([c.reserved_mib for c in cards], [425, 518, 425])
        self.assertEqual([c.total_mib for c in cards], [20480, 32607, 20480])

    def test_idle_rig_is_not_refused_over_the_carve_out(self):
        """#539 regression: 425 MiB of carve-out is not 425 MiB of tenancy."""
        cards = self._cards()
        lines = []
        with _PatchNvml(_fake(_rig(IDLE_FREE))):
            launcher.cards_free_check(cards, lines.append)
        self.assertIn("instrument: nvml_v2_free, allocatable", lines[0])
        self.assertIn("driver-reserved", lines[0])

    def test_a_real_tenant_is_still_refused(self):
        cards = self._cards()
        with _PatchNvml(_fake(_rig(self.BUSY_FREE))):
            with self.assertRaises(launcher.Weg2LaunchRefused) as e:
                launcher.cards_free_check(cards, lambda _m: None)
        self.assertIn("2000 MiB held by processes", str(e.exception))

    def test_refusal_names_its_instrument(self):
        cards = self._cards()
        with _PatchNvml(_fake(_rig(self.BUSY_FREE))):
            with self.assertRaises(launcher.Weg2LaunchRefused) as e:
                launcher.cards_free_check(cards, lambda _m: None)
        self.assertIn("nvml_v2_used", str(e.exception))

    def test_missing_card_refuses_rather_than_keyerrors(self):
        cards = self._cards()
        with _PatchNvml(_fake({0: (20480, 20054, 425)})):
            with self.assertRaises(launcher.Weg2LaunchRefused) as e:
                launcher.cards_free_check(cards, lambda _m: None)
        self.assertIn("no memory row", str(e.exception))


# ===========================================================================
# FIX 2 -- the four defects the review found IN the fix above.
# ===========================================================================

#: The rg6 front log's own prose about corridor samples, verbatim from
#: ``boot_weg2_weg2rg6_7f88b1c75d_0908_070324.front.log`` at 07:03:46Z, 127 s
#: BEFORE that boot's first real sample.  ``solve()`` writes it on any boot
#: that skips a newer candidate, so it is a recurring input, not a one-off.
RG6_PROSE_LINE = (
    "[2026-09-08T07:03:46Z] WEG2-LAUNCH WEG2-HOST-RING SKIPPED (newer than the "
    "chosen table) boot_weg2_weg2rg5_15a46a611a_0908_050519: front carries no "
    "WEG2-CORRIDOR phase=P(awake)/D(awake) samples"
)

#: One genuine post-fix sample line, as the fixed sampler emits it.
POST_FIX_LINE = (
    "[2026-09-08T07:05:53Z] INFO weg2.front: WEG2-CORRIDOR phase=D(awake) epoch=0 "
    "instrument=nvml_v2_free,allocatable band=819-1229MiB "
    "nvml0:free=1030MiB reserved=425MiB verdict=IN "
    "nvml1:free=341MiB reserved=518MiB verdict=BELOW "
    "nvml2:free=852MiB reserved=425MiB verdict=IN "
    "min_so_far={0: 1030, 1: 341, 2: 852} (nvml_v2_free,allocatable, MiB)"
)

#: One genuine PRE-fix sample line (no instrument= token), rg6 front units.
PRE_FIX_LINE = (
    "[2026-09-08T07:31:30Z] INFO weg2.front: WEG2-CORRIDOR phase=D(awake) epoch=20 "
    "nvml0:free=1454MiB nvml1:free=859MiB nvml2:free=1276MiB "
    "min_so_far={0: 1454, 1: 859, 2: 1276}"
)


def _write_log(case, *lines):
    fd, path = tempfile.mkstemp(prefix="weg2fix2-", suffix=".front.log")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n")
    case.addCleanup(os.unlink, path)
    return path


class ProseIsNotASampleTest(unittest.TestCase):
    """FIX 2, finding 1: the #995 prose trap, inside the instrument-namer.

    ``front_corridor_instrument`` scanned for lines CONTAINING
    ``WEG2-CORRIDOR`` and classified any line matching the phase regex.  The
    front log's own skip-reason sentence matches that regex, sits BEFORE the
    first real sample, and the function returns on the first match -- so a
    POST-fix boot was labelled ``total_minus_used(carve-out-blind)`` and
    ``RingTable.provenance()`` appended a warning about an instrument that
    boot never used.  A printed claim the code did not measure, produced by
    the one function added to make instrument claims true.
    """

    def test_prose_before_a_real_sample_does_not_set_the_instrument(self):
        path = _write_log(self, RG6_PROSE_LINE, POST_FIX_LINE)
        self.assertEqual(
            ring_table.front_corridor_instrument(path), "nvml_v2_free,allocatable"
        )

    def test_prose_before_a_pre_fix_sample_still_reports_pre_fix(self):
        """The fix must not flip the OTHER way: a real pre-fix boot stays pre-fix."""
        path = _write_log(self, RG6_PROSE_LINE, PRE_FIX_LINE)
        self.assertEqual(
            ring_table.front_corridor_instrument(path),
            ring_table.CORRIDOR_INSTRUMENT_PRE_FIX,
        )

    def test_a_log_of_nothing_but_prose_reports_no_samples(self):
        path = _write_log(self, RG6_PROSE_LINE)
        self.assertEqual(
            ring_table.front_corridor_instrument(path), "no WEG2-CORRIDOR samples"
        )

    def test_prose_creates_no_phase_entry(self):
        """The nonblocking sibling: an EMPTY 'P' satisfied solve()'s guard.

        ``if "P" not in corridor or "D" not in corridor`` reads a key as "this
        phase has measurements".  The prose line minted ``{'P': {}}``, the
        guard passed, and every card's P credit silently became
        ``free_p.get(uuid, 0) == 0`` instead of the intended refusal.
        """
        path = _write_log(self, RG6_PROSE_LINE, POST_FIX_LINE)
        got = ring_table.parse_front_corridor(path)
        self.assertEqual(got, {"D": {0: 1030, 1: 341, 2: 852}})
        self.assertNotIn("P", got)

    def test_a_sample_line_needs_a_measurement_not_just_the_marker(self):
        self.assertIsNone(ring_table._corridor_sample_phase(RG6_PROSE_LINE))
        self.assertEqual(ring_table._corridor_sample_phase(POST_FIX_LINE), "D")
        self.assertEqual(ring_table._corridor_sample_phase(PRE_FIX_LINE), "D")

    def test_a_phase_key_still_appears_for_a_real_P_sample(self):
        """Can-fail proof for the gate: it must not reject genuine samples."""
        p_line = POST_FIX_LINE.replace("phase=D(awake)", "phase=P(awake)")
        got = ring_table.parse_front_corridor(_write_log(self, p_line, POST_FIX_LINE))
        self.assertEqual(sorted(got), ["D", "P"])
        self.assertEqual(got["P"], {0: 1030, 1: 341, 2: 852})

    def test_provenance_does_not_claim_a_pre_fix_instrument_for_a_post_fix_boot(self):
        """The printed consequence, end to end.

        FIX 3 moved what the provenance line SAYS about a pre-fix source: it no
        longer warns that the credits are over-stated (they are converted now),
        it NAMES the source instrument beside the unit the credits are in.  The
        round-1 property under test is unchanged -- a post-fix boot must not be
        described as a pre-fix one.
        """
        def prov(*lines):
            return ring_table.RingTable(
                boot="b", instrument="i", lines_read=1, image_source="s",
                credit_instrument=ring_table.front_corridor_instrument(
                    _write_log(self, *lines)
                ),
            ).provenance()

        post = prov(RG6_PROSE_LINE, POST_FIX_LINE)
        self.assertIn("CREDIT unit: ALLOCATABLE free", post)
        self.assertIn("source instrument nvml_v2_free,allocatable", post)
        self.assertNotIn(ring_table.CORRIDOR_INSTRUMENT_PRE_FIX, post)
        pre = prov(RG6_PROSE_LINE, PRE_FIX_LINE)
        self.assertIn("CREDIT unit: ALLOCATABLE free", pre)
        self.assertIn(
            f"source instrument {ring_table.CORRIDOR_INSTRUMENT_PRE_FIX}", pre
        )


class TheProseLineIsRealTest(unittest.TestCase):
    """Evidence-tree bound: the trigger is in a real log, not constructed."""

    RG6 = (
        "/spinning/evidence-665-f1/"
        "boot_weg2_weg2rg6_7f88b1c75d_0908_070324.front.log"
    )

    def _log(self):
        if not os.path.exists(self.RG6):
            self.skipTest(f"evidence tree absent: {self.RG6}")
        return self.RG6

    def test_rg6_front_log_contains_the_prose_line(self):
        with open(self._log(), errors="replace") as f:
            hits = [ln for ln in f if "front carries no WEG2-CORRIDOR" in ln]
        self.assertEqual(len(hits), 1, "the trigger line is not in the rg6 log")
        self.assertIsNone(ring_table._corridor_sample_phase(hits[0]))

    def test_rg6_is_still_classified_pre_fix_from_its_real_samples(self):
        """The gate must not have made the real answer worse."""
        self.assertEqual(
            ring_table.front_corridor_instrument(self._log()),
            ring_table.CORRIDOR_INSTRUMENT_PRE_FIX,
        )
        got = ring_table.parse_front_corridor(self._log())
        self.assertEqual(sorted(got), ["D", "P"])
        self.assertTrue(got["P"] and got["D"], "a real phase lost its samples")


class BandHasOneDeclarationTest(unittest.TestCase):
    """FIX 2, finding 2: the front held a private copy of the corridor band.

    ``corridor_guard.py:141`` states the rule: "THE ONE DECLARATION.  Every
    other module that needs the law imports it from here rather than repeating
    the literal."  The commit whose thesis was "ONE reader" froze 819/1229
    beside it, under the comment "the corridor law, verbatim" -- a copy that
    cannot follow the law it quotes.
    """

    def setUp(self):
        self._saved = os.environ.get(corridor_guard.LAW_ENV)
        self.addCleanup(self._restore)

    def _restore(self):
        if self._saved is None:
            os.environ.pop(corridor_guard.LAW_ENV, None)
        else:
            os.environ[corridor_guard.LAW_ENV] = self._saved

    def _move_the_law(self):
        os.environ[corridor_guard.LAW_ENV] = "1536"
        return corridor_guard.corridor_band_floor_mib(), corridor_guard.corridor_band_ceiling_mib()

    def test_front_declares_no_band_literal_of_its_own(self):
        """AST, not grep: prose may quote 819-1229, code may not hold it."""
        src = open(front.__file__, errors="replace").read()
        held = [
            f"line {n.lineno}: {n.value}"
            for n in ast.walk(ast.parse(src, filename=front.__file__))
            if isinstance(n, ast.Constant)
            and isinstance(n.value, int)
            and not isinstance(n.value, bool)
            and n.value in (819, 1229)
        ]
        self.assertEqual(held, [], "front.py holds a frozen copy of the corridor band")

    def test_the_band_follows_the_law(self):
        moved = self._move_the_law()
        self.assertEqual(moved, (1228, 1843))
        self.assertEqual(front.corridor_band_mib(), moved)

    def test_the_verdict_follows_the_law(self):
        self.assertEqual(front.corridor_verdict(1500), "ABOVE")
        self._move_the_law()
        self.assertEqual(front.corridor_verdict(1500), "IN")
        self.assertEqual(front.corridor_verdict(1227), "BELOW")

    def test_the_printed_band_follows_the_law(self):
        self._move_the_law()
        f = _front()
        with _PatchNvml(_fake(_rig({0: 1030, 1: 341, 2: 852}))):
            line = f.corridor_sample()
        self.assertIn("band=1228-1843MiB", line)
        # #1257c: an ENV-set law is still a STATED law, so it keeps the +-20 %
        # band (only a MEASURED transient peak loses the tolerance) -- and it
        # is stamped ENV-OVERRIDE so a reader can tell it from the shipped
        # fallback.
        self.assertIn(
            "nvml0:free=1030MiB reserved=425MiB floor=1536MiB "
            "verdict_floor=1228MiB source=ENV-OVERRIDE reserve=0MiB "
            "verdict=BELOW",
            line,
        )

    def test_the_exported_state_follows_the_law(self):
        self._move_the_law()
        self.assertEqual(_front().state_dict()["corridor_band_mib"], [1228, 1843])

    def test_the_default_band_is_unchanged(self):
        """Same numbers as before the fix, by derivation rather than by copy."""
        self.assertEqual(front.corridor_band_mib(), (819, 1229))
        self.assertEqual(_front().state_dict()["corridor_band_mib"], [819, 1229])


class InstrumentNameFollowsTheReadTest(unittest.TestCase):
    """FIX 2, finding 3: ``nvml_v2_free`` named a struct the code did not read.

    ``memory_snapshot`` fetched the v2 struct, kept ``reserved``, threw its
    ``free`` and ``used`` away, and read them again from the V1 struct -- while
    every consumer printed ``nvml_v2_free`` / ``nvml_v2_used``.  The two agree
    on this driver (v1 free 20054 == v2 free 20054, measured), so the fake here
    makes them DISAGREE: only a reader that actually takes the v2 field can
    pass.
    """

    class _SplitFake(FakeNvml):
        """v1 and v2 report different ``free``. No real driver does this."""

        V1_FREE_MIB = 9999

        def nvmlDeviceGetMemoryInfo(self, h, version=None):
            self.calls += 1
            total, free, reserved = self.cards[h]
            if version is None:
                return _V1(total * MIB, self.V1_FREE_MIB * MIB)
            return _V2(total * MIB, free * MIB, reserved * MIB)

    def test_free_is_the_v2_field(self):
        with _PatchNvml(self._SplitFake(_rig({0: 1030, 1: 341, 2: 852}))):
            snap = nvml_registry.memory_snapshot()
        self.assertEqual([m.free_mib for _d, m in snap], [1030, 341, 852])
        for _d, m in snap:
            self.assertNotEqual(m.free_mib, self._SplitFake.V1_FREE_MIB)
            self.assertEqual(m.free_instrument, nvml_registry.FREE_INSTRUMENT_V2)

    def test_tenant_used_is_the_v2_used_field_not_a_derivation(self):
        """``used`` read, not ``(total - reserved)//MiB - free//MiB`` twice-floored."""
        with _PatchNvml(_fake(_rig({0: 1030, 1: 341, 2: 852}))):
            snap = nvml_registry.memory_snapshot()
        for dev, mem in snap:
            total, free, reserved = _rig({0: 1030, 1: 341, 2: 852})[dev.index]
            self.assertEqual(mem.tenant_used_bytes // MIB, total - free - reserved)
            self.assertEqual(mem.tenant_used_mib, total - free - reserved)
            self.assertEqual(mem.tenant_used_instrument, nvml_registry.USED_INSTRUMENT_V2)

    def test_a_hand_built_MemoryInfo_still_derives(self):
        """Backwards compatible: no v2 ``used`` supplied means the old path."""
        m = nvml_registry.MemoryInfo(
            total_bytes=20480 * MIB, free_bytes=1030 * MIB,
            used_bytes=19450 * MIB, reserved_bytes=425 * MIB,
        )
        self.assertIsNone(m.tenant_used_bytes)
        self.assertEqual(m.tenant_used_mib, 20480 - 425 - 1030)


class CarveOutBlindModeIsNamedTest(unittest.TestCase):
    """FIX 2, finding 3 (degradation half), and the disclosed nonblocking item.

    With no v2 struct the registry returns ``reserved=0``, ``tenant_used``
    collapses to the v1 ``used`` (carve-out INCLUDED -- the #539 trap), and the
    predecessor still printed ``instrument=nvml_v2_free,allocatable`` beside
    ``reserved=0MiB``.  The value degrades; the CLAIM must degrade with it.
    """

    class _NoV2(FakeNvml):
        nvmlMemory_v2 = None

    def _blind(self, free_by_index):
        return self._NoV2(_rig(free_by_index))

    def test_registry_says_the_carve_out_is_unknown(self):
        with _PatchNvml(self._blind(IDLE_FREE)):
            snap = nvml_registry.memory_snapshot()
        for _d, m in snap:
            self.assertFalse(m.carve_out_known)
            self.assertIsNone(m.tenant_used_bytes)
            self.assertEqual(m.free_instrument, nvml_registry.FREE_INSTRUMENT_V1)
            self.assertEqual(m.tenant_used_instrument, nvml_registry.USED_INSTRUMENT_V1)
            self.assertIn("carve-out INCLUDED", m.tenant_used_instrument)

    def test_the_corridor_line_degrades_its_instrument_token(self):
        f = _front()
        with _PatchNvml(self._blind({0: 1030, 1: 341, 2: 852})):
            line = f.corridor_sample()
        self.assertIn(f"instrument={front.CORRIDOR_INSTRUMENT_NO_V2}", line)
        self.assertNotIn("instrument=nvml_v2_free", line)
        self.assertIn("reserved=0MiB", line)

    def test_the_exported_state_degrades_with_the_line(self):
        f = _front()
        self.assertEqual(f.state_dict()["corridor_instrument"], front.CORRIDOR_INSTRUMENT)
        with _PatchNvml(self._blind({0: 1030, 1: 341, 2: 852})):
            f.corridor_sample()
        self.assertEqual(
            f.state_dict()["corridor_instrument"], front.CORRIDOR_INSTRUMENT_NO_V2
        )

    def test_one_blind_card_downgrades_the_whole_line(self):
        """One token stands for the line, so the weakest card sets it."""
        good = front.CardFree(0, "u0", 1030, 425, carve_out_known=True)
        bad = front.CardFree(1, "u1", 341, 0, carve_out_known=False)
        self.assertEqual(front.corridor_instrument([good]), front.CORRIDOR_INSTRUMENT)
        self.assertEqual(
            front.corridor_instrument([good, bad]), front.CORRIDOR_INSTRUMENT_NO_V2
        )

    def test_the_launcher_refusal_names_v1_when_v2_is_absent(self):
        """The #539 trap returns in this mode; the message must not hide it."""
        busy = {**IDLE_FREE, 0: IDLE_FREE[0] - 2000}
        with _PatchNvml(self._blind(busy)):
            cards = launcher.resolve_cards()
            with self.assertRaises(launcher.Weg2LaunchRefused) as e:
                launcher.cards_free_check(cards, lambda _m: None)
        self.assertIn("carve-out INCLUDED", str(e.exception))
        self.assertNotIn("instrument nvml_v2_used", str(e.exception))

    def test_the_launcher_free_line_names_v1_when_v2_is_absent(self):
        lines = []
        with _PatchNvml(self._blind(IDLE_FREE)):
            cards = launcher.resolve_cards()
            launcher.cards_free_check(cards, lines.append)
        self.assertIn("instrument: nvml_v1_free, allocatable", lines[0])
        self.assertNotIn("nvml_v2_free", lines[0])


class LauncherThresholdTest(unittest.TestCase):
    """The mutant gap the review's own probe found and left open.

    Replacing the guard quantity at ``launcher.py`` ``if m.tenant_used_mib >
    1500`` with ``m.used_bytes // MIB > 1500`` -- the #539 mirror trap at the
    threshold, message unchanged -- SURVIVED all 37 tests: the idle case is
    425 either way and the busy case is 2000 either way.  A 3080 holding
    1076-1500 MiB of real tenancy reads 1501-1925 in the v1 figure, so the
    mutant refuses a card the law admits.  1200 is inside that band.
    """

    def _cards(self):
        with _PatchNvml(_fake(_rig(IDLE_FREE))):
            return launcher.resolve_cards()

    def test_a_tenant_inside_the_mutant_band_is_not_refused(self):
        tenancy = 1200
        free = {**IDLE_FREE, 0: IDLE_FREE[0] - tenancy}
        with _PatchNvml(_fake(_rig(free))) as fake:
            self.assertEqual(
                fake.nvmlDeviceGetMemoryInfo(0).used // MIB, tenancy + 425,
                "the v1 figure must be over 1500 or this case proves nothing",
            )
            launcher.cards_free_check(self._cards(), lambda _m: None)

    def test_the_threshold_still_bites_just_above_it(self):
        free = {**IDLE_FREE, 0: IDLE_FREE[0] - 1501}
        with _PatchNvml(_fake(_rig(free))):
            with self.assertRaises(launcher.Weg2LaunchRefused) as e:
                launcher.cards_free_check(self._cards(), lambda _m: None)
        self.assertIn("1501 MiB held by processes", str(e.exception))


class NvmlWarningLatchTest(unittest.TestCase):
    """Nonblocking: one transient failure muted the warning for the boot."""

    def setUp(self):
        front._nvml_unavailable_logged = False
        self.addCleanup(setattr, front, "_nvml_unavailable_logged", False)

    def _fail_once_then(self, results):
        it = iter(results)

        def snapshot():
            r = next(it)
            if isinstance(r, Exception):
                raise r
            return r

        return snapshot

    def test_a_good_read_re_arms_the_warning(self):
        orig = nvml_registry.memory_snapshot
        good = None
        with _PatchNvml(_fake(_rig({0: 1030, 1: 341, 2: 852}))):
            good = nvml_registry.memory_snapshot()
        nvml_registry.memory_snapshot = self._fail_once_then(
            [RuntimeError("transient"), good, RuntimeError("again")]
        )
        try:
            with self.assertLogs("weg2.front", level=logging.WARNING):
                front._nvml_free()
            self.assertTrue(front._nvml_free())
            self.assertFalse(front._nvml_unavailable_logged, "the latch stayed set")
            with self.assertLogs("weg2.front", level=logging.WARNING):
                front._nvml_free()
        finally:
            nvml_registry.memory_snapshot = orig


class BootArmTest(unittest.TestCase):
    """FIX 2, finding 4: the acceptance [SECTION 1x] pointed at and lacked.

    ``arm_report`` grades a boot's own front log (did it sample, in which
    unit); ``pair_verdict`` is the proof, because a sampler printing the right
    LABEL over the wrong NUMBER passes the first and fails the second.
    """

    def test_a_post_fix_log_passes(self):
        rep = corridor_arm.arm_report(_write_log(self, RG6_PROSE_LINE, POST_FIX_LINE))
        self.assertTrue(rep.ok, rep.report())
        self.assertEqual(rep.instrument, "nvml_v2_free,allocatable")
        self.assertEqual((rep.samples, rep.prose_mentions), (1, 1))
        self.assertIn("verdict=PASS", rep.report())

    def test_a_pre_fix_log_fails_and_says_why(self):
        rep = corridor_arm.arm_report(_write_log(self, PRE_FIX_LINE))
        self.assertFalse(rep.ok)
        self.assertIn("pre-fix total-minus-used sampler", " ".join(rep.problems))

    def test_a_log_with_no_samples_fails(self):
        rep = corridor_arm.arm_report(_write_log(self, RG6_PROSE_LINE))
        self.assertFalse(rep.ok)
        self.assertEqual((rep.samples, rep.prose_mentions), (0, 1))
        self.assertIn("the sampler did not run", " ".join(rep.problems))

    def test_a_carve_out_blind_log_fails_the_arm(self):
        blind = POST_FIX_LINE.replace(
            "instrument=nvml_v2_free,allocatable",
            f"instrument={front.CORRIDOR_INSTRUMENT_NO_V2}",
        )
        rep = corridor_arm.arm_report(_write_log(self, blind))
        self.assertFalse(rep.ok)
        self.assertIn("carve-out-blind", " ".join(rep.problems))

    def test_band_check_is_opt_in(self):
        """341 is BELOW the floor: a capacity finding, not an instrument one."""
        path = _write_log(self, POST_FIX_LINE)
        self.assertTrue(corridor_arm.arm_report(path).ok)
        strict = corridor_arm.arm_report(path, require_in_band=True)
        self.assertFalse(strict.ok)
        self.assertIn(
            "nvml1 minimum 341 MiB (allocatable free) is BELOW",
            " ".join(strict.problems),
        )

    def test_the_band_check_grades_BOTH_edges(self):
        """Both edges are still GRADED. Only one of them still FAILS.

        REVERSED 2026-09-09 by user decision (#1257c, consequence 5): "the
        upper band edge stays a FINDING ('unmobilised free') never a FAIL by
        itself". Below the floor is a breach and still fails the arm; above
        the ceiling is VRAM buying no tokens, which is a capacity question for
        the planner and not a breach of the corridor law, so it is reported in
        ``findings`` and the arm still passes on it alone.

        The original intent of this case -- that a check must not look only
        downward -- is preserved: the ABOVE card must still be NAMED, with its
        number, in the report. What changed is which list it lands in.
        (Mutant 4g, dropping the upper comparison entirely, still dies here.)
        """
        above = POST_FIX_LINE.replace("nvml0:free=1030MiB", "nvml0:free=5000MiB")
        rep = corridor_arm.arm_report(_write_log(self, above), require_in_band=True)
        self.assertFalse(rep.ok, "the BELOW card still fails the arm")
        joined = " ".join(rep.problems)
        self.assertIn("nvml1 minimum 341 MiB (allocatable free) is BELOW", joined)
        self.assertNotIn("is ABOVE", joined)
        found = " ".join(rep.findings)
        self.assertIn("nvml0", found)
        self.assertIn("unmobilised_free_mib=", found)
        self.assertIn("a FINDING, not a failure", found)

    def test_an_over_filled_card_alone_does_not_fail_the_arm(self):
        """#1257c consequence 5, isolated: idle VRAM is not a breach.

        The predecessor failed an acceptance outright on a card resting above
        the ceiling. Under the user decision that boot passes its corridor arm
        and carries a finding.
        """
        only_above = POST_FIX_LINE
        for a, b in (
            ("nvml0:free=1030MiB", "nvml0:free=5000MiB"),
            ("nvml1:free=341MiB", "nvml1:free=5000MiB"),
            ("nvml2:free=852MiB", "nvml2:free=5000MiB"),
        ):
            only_above = only_above.replace(a, b)
        rep = corridor_arm.arm_report(
            _write_log(self, only_above), require_in_band=True
        )
        self.assertEqual(rep.problems, [], rep.problems)
        self.assertTrue(rep.ok)
        self.assertEqual(len(rep.findings), 3, rep.findings)

    def test_an_in_band_log_passes_the_strict_check(self):
        """Can-fail the other way: the strict check must be satisfiable."""
        good = (
            POST_FIX_LINE.replace("nvml1:free=341MiB", "nvml1:free=1000MiB")
            .replace("nvml2:free=852MiB", "nvml2:free=900MiB")
        )
        rep = corridor_arm.arm_report(_write_log(self, good), require_in_band=True)
        self.assertTrue(rep.ok, rep.report())

    def test_the_report_grades_each_minimum(self):
        rep = corridor_arm.arm_report(_write_log(self, POST_FIX_LINE))
        self.assertIn("phase=D(awake) nvml1: min_free=341MiB verdict=BELOW", rep.report())
        self.assertIn("phase=D(awake) nvml0: min_free=1030MiB verdict=IN", rep.report())

    def test_pair_passes_when_the_two_readers_agree(self):
        res = corridor_arm.pair_verdict({0: 1030, 1: 341, 2: 852}, {0: 1030, 1: 341, 2: 852})
        self.assertTrue(res.ok, res.report())
        self.assertIn("verdict=PASS", res.report())

    def test_pair_tolerates_the_one_mib_rounding_spread(self):
        """pynvml floors 20054 where nvidia-smi prints 20055; measured."""
        self.assertTrue(corridor_arm.pair_verdict({0: 20054}, {0: 20055}).ok)

    def test_pair_fails_on_the_rg6_instant_and_names_the_defect(self):
        """The exact form that produced 1030/341/852 vs 1454/859/1276."""
        res = corridor_arm.pair_verdict(
            {0: 1454, 1: 859, 2: 1276}, {0: 1030, 1: 341, 2: 852}
        )
        self.assertFalse(res.ok)
        self.assertEqual(res.disagree, (0, 1, 2))
        self.assertEqual([res.rows[i][2] for i in (0, 1, 2)], [424, 518, 424])
        self.assertTrue(res.looks_like_the_carve_out_defect)
        self.assertIn("pre-fix `total - used` subtraction", res.report())

    def test_pair_fails_without_calling_every_disagreement_the_carve_out(self):
        res = corridor_arm.pair_verdict({0: 1030}, {0: 700})
        self.assertFalse(res.ok)
        self.assertFalse(res.looks_like_the_carve_out_defect)

    def test_pair_names_a_card_only_one_reader_saw(self):
        res = corridor_arm.pair_verdict({0: 1030, 1: 341}, {0: 1030})
        self.assertFalse(res.ok)
        self.assertEqual(res.only_in_tree, (1,))
        self.assertIn("cards only the in-tree reader saw: [1]", res.report())

    def test_an_empty_pairing_is_not_a_pass(self):
        self.assertFalse(corridor_arm.pair_verdict({}, {}).ok)

    def test_smi_rows_parse_and_junk_is_dropped(self):
        got = corridor_arm.parse_smi_free("0, 1030\n1, 341\nFailed to initialize NVML\n2, 852\n")
        self.assertEqual(got, {0: 1030, 1: 341, 2: 852})

    def test_the_pairing_cannot_be_widened_past_the_defect(self):
        """A tolerance at or above the smallest carve-out cannot fail on it."""
        cli = _cli()
        with self.assertRaises(SystemExit) as e:
            cli.main(["--pair", "--tolerance-mib", "424"])
        self.assertEqual(e.exception.code, 2)

    def test_cli_exit_codes(self):
        cli = _cli()
        self.assertEqual(cli.main(["--log", _write_log(self, POST_FIX_LINE)]), 0)
        self.assertEqual(cli.main(["--log", _write_log(self, PRE_FIX_LINE)]), 1)
        with self.assertRaises(SystemExit) as e:
            cli.main([])
        self.assertEqual(e.exception.code, 2)

    def test_cli_pairs_from_supplied_output_without_touching_a_card(self):
        cli = _cli()
        with _PatchNvml(_fake(_rig({0: 1030, 1: 341, 2: 852}))):
            self.assertEqual(cli.main(["--pair", "--smi-output", "0, 1030\n1, 341\n2, 852\n"]), 0)
            self.assertEqual(cli.main(["--pair", "--smi-output", "0, 1454\n1, 859\n2, 1276\n"]), 1)

    def test_live_pair_reads_the_in_tree_sampler(self):
        with _PatchNvml(_fake(_rig({0: 1030, 1: 341, 2: 852}))):
            res = corridor_arm.live_pair(smi_text="0, 1030\n1, 341\n2, 852\n")
        self.assertTrue(res.ok, res.report())


class BootArmAgainstRealBootsTest(unittest.TestCase):
    """Evidence-tree bound: the arm must FAIL on all three pre-fix boots.

    A can-fail proof against real inputs -- an acceptance that passes on the
    boots whose instrument this commit corrects would be measuring nothing.
    """

    EVIDENCE = RealBootLogsAreProvenPreFixTest.EVIDENCE
    STEMS = RealBootLogsAreProvenPreFixTest.STEMS

    def test_all_three_pre_fix_boots_fail_the_arm(self):
        for key, stem in self.STEMS.items():
            path = os.path.join(self.EVIDENCE, stem + ".front.log")
            if not os.path.exists(path):
                self.skipTest(f"evidence tree absent: {path}")
            rep = corridor_arm.arm_report(path)
            self.assertFalse(rep.ok, f"{key} passed an arm it must fail")
            self.assertGreater(rep.samples, 0, key)
            self.assertIn("pre-fix total-minus-used sampler", " ".join(rep.problems), key)


#: The rg6 front log's own ``NVML -> CUDA ordinal map`` line, verbatim (line 5
#: of ``boot_weg2_weg2rg6_7f88b1c75d_0908_070324.front.log``).  A carve-out
#: belongs to a CARD, so a fixture that wants one converted has to name its
#: cards the way a real boot does.
REAL_ORDINAL_MAP = (
    "[2026-09-08T07:03:24Z] WEG2-LAUNCH NVML -> CUDA ordinal map: "
    "ordinal 0 = nvml 1 NVIDIA GeForce RTX 5090 "
    "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d total 32607 MiB, "
    "ordinal 1 = nvml 0 NVIDIA GeForce RTX 3080 "
    "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7 total 20480 MiB, "
    "ordinal 2 = nvml 2 NVIDIA GeForce RTX 3080 "
    "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4 total 20480 MiB"
)


class ArmGradesInOneUnitTest(unittest.TestCase):
    """FIX 3, the round-2 blocker: the arm graded front units against the
    allocatable band and blessed the one card the strand exists for.

    ``arm_report`` read a pre-fix log's minima -- allocatable free PLUS the
    driver carve-out -- and graded every one of them with
    ``front.corridor_verdict``, i.e. against the 819-1229 MiB ALLOCATABLE band,
    in the same call that had already appended "every free is over-stated by
    that card's driver carve-out" as a problem.  On the real rg6 log that
    printed ``nvml1: min_free=859MiB verdict=IN`` for the 5090 that sat at
    341 MiB, 478 MiB below the floor, and reported the two 3080s that were
    genuinely in band as ABOVE it.  Every assertion here fails on the parent
    commit 4302724e8b.
    """

    def _pre_fix(self, *free, identity=True):
        line = (
            "[2026-09-08T07:31:30Z] INFO weg2.front: WEG2-CORRIDOR phase=D(awake) "
            "epoch=20 " + " ".join(f"nvml{i}:free={v}MiB" for i, v in free)
        )
        return _write_log(self, *(([REAL_ORDINAL_MAP] if identity else []) + [line]))

    def test_a_pre_fix_minimum_is_graded_in_the_bands_unit(self):
        """859 front units = 341 allocatable = BELOW, not IN."""
        rep = corridor_arm.arm_report(self._pre_fix((0, 1454), (1, 859), (2, 1276)))
        line = rep.report()
        self.assertIn("nvml1: min_free=341MiB", line)
        self.assertIn("verdict=BELOW", line)
        self.assertNotIn("min_free=859MiB verdict=IN", line)

    def test_the_report_shows_the_source_number_and_the_correction(self):
        """The operator greps the boot log, which prints the SOURCE unit."""
        rep = corridor_arm.arm_report(self._pre_fix((1, 859)))
        self.assertIn(
            "nvml1: min_free=341MiB(allocatable; the log printed 859MiB, "
            "-518 MiB carve-out) verdict=BELOW",
            rep.report(),
        )

    def test_the_two_3080s_are_no_longer_reported_ABOVE_the_band(self):
        """1454/1276 front units are 1029/851 allocatable: IN and IN."""
        rep = corridor_arm.arm_report(
            self._pre_fix((0, 1454), (1, 859), (2, 1276)), require_in_band=True
        )
        joined = " ".join(rep.problems)
        self.assertNotIn("nvml0 minimum", joined)
        self.assertNotIn("nvml2 minimum", joined)
        self.assertIn("nvml1 minimum 341 MiB (allocatable free) is BELOW", joined)

    def test_the_band_check_can_still_pass_on_a_converted_log(self):
        """Can-fail the other way: conversion must not make the check vacuous.

        A pre-fix log always fails the arm on its INSTRUMENT -- that problem is
        the point of the arm and converting the numbers does not retire it --
        so what must be satisfiable here is the BAND half: 1454/1400/1276 front
        units convert to 1029/882/851, all three in band, and no band problem
        may be raised.
        """
        rep = corridor_arm.arm_report(
            self._pre_fix((0, 1454), (1, 1400), (2, 1276)), require_in_band=True
        )
        band = [p for p in rep.problems if "band" in p]
        self.assertEqual(band, [], rep.report())
        self.assertEqual(len(rep.problems), 1, rep.report())
        self.assertIn("pre-fix total-minus-used sampler", rep.problems[0])

    def test_a_pre_fix_log_with_no_card_identity_is_ungraded_not_graded(self):
        """No ordinal map -> the nvml indices name no card -> no carve-out."""
        rep = corridor_arm.arm_report(self._pre_fix((1, 859), identity=False))
        self.assertIn("verdict=UNGRADED", rep.report())
        self.assertNotIn("verdict=IN", rep.report())

    def test_an_ungradeable_log_FAILS_the_strict_check(self):
        """A strict check must never pass by declining to look."""
        rep = corridor_arm.arm_report(
            self._pre_fix((1, 859), identity=False), require_in_band=True
        )
        self.assertFalse(rep.ok)
        self.assertIn("cannot be graded against the 819-1229 MiB band",
                      " ".join(rep.problems))
        self.assertIn("no measured or recorded driver carve-out",
                      " ".join(rep.problems))

    def test_an_unknown_instrument_token_is_never_graded(self):
        """Mixed units in their least visible form: a unit nobody declared."""
        line = PRE_FIX_LINE.replace(
            "epoch=20", "epoch=20 instrument=free_by_some_future_reader"
        )
        rep = corridor_arm.arm_report(
            _write_log(self, REAL_ORDINAL_MAP, line), require_in_band=True
        )
        self.assertFalse(rep.ok)
        self.assertIn("names neither the allocatable unit", " ".join(rep.problems))
        self.assertIn("verdict=UNGRADED", rep.report())

    def test_a_post_fix_log_is_not_corrected_twice(self):
        """Already allocatable: correction 0, and the line stays as it was."""
        rep = corridor_arm.arm_report(_write_log(self, REAL_ORDINAL_MAP, POST_FIX_LINE))
        self.assertTrue(rep.ok, rep.report())
        self.assertIn("nvml1: min_free=341MiB verdict=BELOW", rep.report())
        self.assertFalse(rep.units["D"].converted)
        self.assertIn("already allocatable free", rep.report())

    def test_a_measured_carve_out_wins_over_the_recorded_one(self):
        """The launcher's own NVML read is the live registry snapshot."""
        rep = corridor_arm.arm_report(
            self._pre_fix((1, 859)),
            carve_out_by_uuid={"GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d": 500},
        )
        self.assertEqual(rep.units["D"].allocatable[1], 359)
        self.assertIn("matched by UUID", rep.report())

    def test_the_recorded_table_is_keyed_by_card_never_by_index(self):
        """NVML order shifts between boots; 518 must not land on a 3080."""
        self.assertEqual(
            set(ring_table.RECORD_CARVE_OUT_MIB),
            {
                "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7",
                "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d",
                "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4",
            },
        )
        swapped = REAL_ORDINAL_MAP.replace(
            "nvml 1 NVIDIA GeForce RTX 5090", "nvml 7 NVIDIA GeForce RTX 5090"
        )
        line = (
            "[2026-09-08T07:31:30Z] INFO weg2.front: WEG2-CORRIDOR phase=D(awake) "
            "epoch=20 nvml7:free=859MiB"
        )
        rep = corridor_arm.arm_report(_write_log(self, swapped, line))
        self.assertEqual(rep.units["D"].allocatable[7], 341, "the CARD carries 518")

    def test_the_two_carve_out_tables_cannot_drift_apart(self):
        """The pairing's delta tuple and the correction table are one rig.

        ``KNOWN_CARVE_OUT_MIB`` recognises a pairing's DELTAS (which carry the
        1 MiB spread between pynvml and nvidia-smi); ``RECORD_CARVE_OUT_MIB``
        corrects a NUMBER.  Different quantities, different roles -- but the
        same three cards, so a value that appears in one and not the other is
        a drift, not a design.
        """
        for uuid, mib in ring_table.RECORD_CARVE_OUT_MIB.items():
            self.assertTrue(
                any(abs(mib - k) <= 1 for k in corridor_arm.KNOWN_CARVE_OUT_MIB),
                f"{uuid} carve-out {mib} matches no KNOWN_CARVE_OUT_MIB entry",
            )

    def test_the_arm_and_the_ring_credit_share_one_converter(self):
        """ONE converter, no second table (AST, so a copy cannot creep back).

        ``arm_report`` must reach its numbers through
        ``ring_table.corridor_allocatable`` and must not do the arithmetic
        itself: a second subtraction here is exactly how the two readers came
        to disagree about the unit in the first place.  (The subtractions
        elsewhere in the module belong to :func:`pair_verdict`, whose whole
        subject is the DIFFERENCE between two readers and which corrects
        nothing.)
        """
        src = open(corridor_arm.__file__.replace(".pyc", ".py")).read()
        fn = next(
            n for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.FunctionDef) and n.name == "arm_report"
        )
        self.assertEqual(
            [n for n in ast.walk(fn)
             if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub)],
            [],
            "arm_report subtracts something itself instead of calling "
            "ring_table.corridor_allocatable",
        )
        called = {
            ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)
        }
        self.assertIn("ring_table.corridor_allocatable", called)
        self.assertFalse(
            [
                n for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.Dict)
                and any(
                    isinstance(k, ast.Constant) and str(k.value).startswith("GPU-")
                    for k in n.keys
                )
            ],
            "the arm carries its own per-card carve-out table",
        )


class ArmAgainstTheRealPreFixBootsTest(unittest.TestCase):
    """The blocker, reproduced and closed on the logs it was found on."""

    EVIDENCE = RealBootLogsAreProvenPreFixTest.EVIDENCE
    STEMS = RealBootLogsAreProvenPreFixTest.STEMS

    #: Phase D whole-boot minima, allocatable free, and the verdict the band
    #: gives them.  Re-derived from each boot's own front log minus that card's
    #: carve-out; cross-validated against the independent ``memory.free``
    #: samplers of rg5 (161) and rg6 (1030/341/852) to 0-1 MiB.
    EXPECTED_D = {
        "rg3": {0: (1125, "IN"), 1: (579, "BELOW"), 2: (893, "IN")},
        "rg5": {0: (969, "IN"), 1: (161, "BELOW"), 2: (745, "BELOW")},
        "rg6": {0: (1029, "IN"), 1: (341, "BELOW"), 2: (851, "IN")},
    }

    def _log(self, key):
        p = os.path.join(self.EVIDENCE, self.STEMS[key] + ".front.log")
        if not os.path.exists(p):
            self.skipTest(f"evidence tree absent: {p}")
        return p

    def test_every_pre_fix_boot_is_graded_in_allocatable_free(self):
        for key, expected in self.EXPECTED_D.items():
            rep = corridor_arm.arm_report(self._log(key))
            unit = rep.units["D"]
            self.assertTrue(unit.ok, f"{key}: {unit.reason}")
            for idx, (mib, verdict) in expected.items():
                self.assertEqual(unit.allocatable[idx], mib, f"{key} nvml{idx}")
                self.assertEqual(front.corridor_verdict(mib), verdict, f"{key} nvml{idx}")

    def test_the_rg6_5090_is_no_longer_blessed(self):
        """The exact line the round-2 review quoted, and its correction."""
        rep = corridor_arm.arm_report(self._log("rg6"), require_in_band=True)
        line = rep.report()
        self.assertNotIn("nvml1: min_free=859MiB verdict=IN", line)
        self.assertIn("nvml1: min_free=341MiB", line)
        joined = " ".join(rep.problems)
        self.assertIn("nvml1 minimum 341 MiB (allocatable free) is BELOW", joined)
        self.assertNotIn("phase=D nvml0 minimum", joined)
        self.assertNotIn("phase=D nvml2 minimum", joined)

    def test_the_instrument_problem_is_still_raised(self):
        """Converting the number does not make the pre-fix SAMPLER acceptable."""
        for key in self.STEMS:
            rep = corridor_arm.arm_report(self._log(key))
            self.assertFalse(rep.ok, key)
            self.assertIn("pre-fix total-minus-used sampler", " ".join(rep.problems), key)


def _cli():
    """The boot-arm CLI, loaded from its durable path in the tree."""
    import importlib.util

    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(front.__file__)))))
    path = os.path.join(os.path.dirname(root), "scripts", "weg2", "corridor_arm_check.py")
    spec = importlib.util.spec_from_file_location("weg2_corridor_arm_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if __name__ == "__main__":
    unittest.main()
