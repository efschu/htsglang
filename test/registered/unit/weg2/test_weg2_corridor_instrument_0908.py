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

from sglang.srt.registry import nvml as nvml_registry
from sglang.srt.weg2 import front, launcher, ring_table

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
        self.assertEqual((front.CORRIDOR_FLOOR_MIB, front.CORRIDOR_CEIL_MIB), (819, 1229))


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

    def test_verdict_per_card(self):
        _f, line = self._line({0: 1030, 1: 341, 2: 852})
        self.assertIn("nvml0:free=1030MiB reserved=425MiB verdict=IN", line)
        self.assertIn("nvml1:free=341MiB reserved=518MiB verdict=BELOW", line)
        self.assertIn("nvml2:free=852MiB reserved=425MiB verdict=IN", line)

    def test_verdict_flips_at_the_floor(self):
        _f, low = self._line({0: 818, 1: 818, 2: 818})
        self.assertIn("nvml0:free=818MiB reserved=425MiB verdict=BELOW", low)
        _f, edge = self._line({0: 819, 1: 819, 2: 819})
        self.assertIn("nvml0:free=819MiB reserved=425MiB verdict=IN", edge)

    def test_verdict_flips_at_the_ceiling(self):
        _f, edge = self._line({0: 1229, 1: 1229, 2: 1229})
        self.assertIn("nvml0:free=1229MiB reserved=425MiB verdict=IN", edge)
        _f, high = self._line({0: 1230, 1: 1230, 2: 1230})
        self.assertIn("nvml0:free=1230MiB reserved=425MiB verdict=ABOVE", high)

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


if __name__ == "__main__":
    unittest.main()
