"""#606: two silent-default getattr sites on capacity paths.

FIX 1 -- registry/nvml.py ``_memory_info``: the old code used
``getattr(v2, "reserved", 0)`` which silently returned 0 when the v2
struct was absent *and* when the struct existed but had no ``reserved``
field. These are two different conditions:

  (a) No v2 support at all -- the binding or driver predates it. A
      carve-out of 0 is the correct fallback, but it must be **visible**.
      A one-time ``logger.warning`` per process replaces the silent
      default.

  (b) v2 struct is present but missing ``reserved`` -- this is a data
      error, not an old driver. Raise ``RuntimeError`` instead of
      swallowing the absence.

FIX 2 -- planner/runner.py ``own_vram_gate``: the old code used
``getattr(proc, "usedGpuMemory", 0)`` which conflated "0 MiB used" with
"NVML hid the value from us". The new code uses ``getattr(..., None)``
and adds a ``used_known`` boolean to every process entry, so the
provenance record can distinguish the two.

Both fixes follow the same principle: the producer (registry/nvml.py)
must not be softer than the consumer (mem_ledger). A silently-zero
carve-out overbooks every card by ~425-518 MiB (#602).
"""

import logging
import types
import unittest

from sglang.srt.planner.runner import own_vram_gate
from sglang.srt.registry.nvml import _memory_info
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MIB = 1 << 20
TOTAL_BYTES = 20480 * MIB  # RTX 3080 nominal


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakePynvml:
    """Minimal pynvml stub. ``nvmlMemory_v2`` can be toggled on/off."""

    def __init__(self, has_v2=True, v2_raises=False, v2_has_reserved=True):
        self.has_v2 = has_v2
        self.v2_raises = v2_raises
        self.v2_has_reserved = v2_has_reserved
        if has_v2:
            self.nvmlMemory_v2 = types.SimpleNamespace()
        else:
            # Deliberately no attribute at all.
            object.__setattr__(self, "_no_v2", True)

    def nvmlDeviceGetMemoryInfo(self, handle, version=None):
        if version is not None and self.v2_raises:
            raise RuntimeError("NVML_ERROR_NOT_SUPPORTED")
        if version is not None:
            obj = types.SimpleNamespace(total=TOTAL_BYTES, free=19000 * MIB, used=1480 * MIB)
            if self.v2_has_reserved:
                obj.reserved = 425 * MIB
            return obj
        return types.SimpleNamespace(total=TOTAL_BYTES, free=19000 * MIB, used=1480 * MIB)


class _FakeHandle:
    """Duck-type for an NVML device handle -- identity is irrelevant."""


# ---------------------------------------------------------------------------
# FIX 1: _memory_info in registry/nvml.py
# ---------------------------------------------------------------------------


class _WarningCapture(logging.Handler):
    """Collect warning-level log records for inspection."""

    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


class TestMemoryInfoNoV2(unittest.TestCase):
    """Case (a): pynvml lacks nvmlMemory_v2 or the v2 call raises."""

    def _reset_flag(self):
        import sglang.srt.registry.nvml as mod

        mod._nv2_warning_emitted = False

    def setUp(self):
        self._reset_flag()
        self.handler = _WarningCapture()
        self.handler.setLevel(logging.WARNING)
        logger = logging.getLogger("sglang.srt.registry.nvml")
        logger.addHandler(self.handler)

    def tearDown(self):
        logger = logging.getLogger("sglang.srt.registry.nvml")
        logger.removeHandler(self.handler)
        self._reset_flag()

    def test_no_v2_attribute_returns_zero_and_warns_once(self):
        """pynvml without nvmlMemory_v2: reserved is 0, one warning fires."""
        pynvml = _FakePynvml(has_v2=False)
        handle = _FakeHandle()

        total, reserved = _memory_info(pynvml, handle)
        self.assertEqual(total, TOTAL_BYTES)
        self.assertEqual(reserved, 0)
        self.assertEqual(len(self.handler.records), 1)
        self.assertIn("nvmlMemory_v2", self.handler.records[0].getMessage())

    def test_v2_call_raises_returns_zero_and_warns_once(self):
        """pynvml with v2 attribute but the call raises: same fallback."""
        pynvml = _FakePynvml(has_v2=True, v2_raises=True)
        handle = _FakeHandle()

        total, reserved = _memory_info(pynvml, handle)
        self.assertEqual(total, TOTAL_BYTES)
        self.assertEqual(reserved, 0)
        self.assertEqual(len(self.handler.records), 1)

    def test_warning_fires_only_once_across_two_calls(self):
        """The module-level guard prevents duplicate warnings."""
        pynvml = _FakePynvml(has_v2=False)
        handle = _FakeHandle()

        _memory_info(pynvml, handle)
        _memory_info(pynvml, handle)
        self.assertEqual(
            len(self.handler.records),
            1,
            "the one-time warning must not repeat on subsequent calls",
        )


class TestMemoryInfoV2HappyPath(unittest.TestCase):
    """Case (a) reversed: v2 is available and has ``reserved``."""

    def _reset_flag(self):
        import sglang.srt.registry.nvml as mod

        mod._nv2_warning_emitted = False

    def setUp(self):
        self._reset_flag()
        self.handler = _WarningCapture()
        self.handler.setLevel(logging.WARNING)
        logger = logging.getLogger("sglang.srt.registry.nvml")
        logger.addHandler(self.handler)

    def tearDown(self):
        logger = logging.getLogger("sglang.srt.registry.nvml")
        logger.removeHandler(self.handler)
        self._reset_flag()

    def test_v2_with_reserved_passes_through_no_warning(self):
        """Happy path: reserved value flows through, no warning emitted."""
        pynvml = _FakePynvml(has_v2=True, v2_raises=False, v2_has_reserved=True)
        handle = _FakeHandle()

        total, reserved = _memory_info(pynvml, handle)
        self.assertEqual(total, TOTAL_BYTES)
        self.assertEqual(reserved, 425 * MIB)
        self.assertEqual(
            len(self.handler.records),
            0,
            "happy path must not emit a warning",
        )


class TestMemoryInfoV2MissingReserved(unittest.TestCase):
    """Case (b): v2 struct exists but lacks the ``reserved`` attribute."""

    def _reset_flag(self):
        import sglang.srt.registry.nvml as mod

        mod._nv2_warning_emitted = False

    def setUp(self):
        self._reset_flag()

    def tearDown(self):
        self._reset_flag()

    def test_missing_reserved_raises_runtime_error(self):
        """RuntimeError names both the struct type and the field."""
        pynvml = _FakePynvml(has_v2=True, v2_raises=False, v2_has_reserved=False)
        handle = _FakeHandle()

        with self.assertRaises(RuntimeError) as cm:
            _memory_info(pynvml, handle)
        msg = str(cm.exception)
        self.assertIn("reserved", msg)
        self.assertIn("nvmlMemory_v2", msg)

    def test_old_getattr_default_would_silently_return_zero(self):
        """Can-fail proof: if someone restores ``getattr(v2, "reserved", 0)``
        this test goes red because no exception is raised.

        This is the mechanical proof that the RuntimeError path is active:
        the old code would return (total, 0) silently.
        """
        pynvml = _FakePynvml(has_v2=True, v2_raises=False, v2_has_reserved=False)
        handle = _FakeHandle()

        # This MUST raise -- if it does not, the code silently falls back to 0,
        # which is the exact defect #606 fixes.
        with self.assertRaises(RuntimeError):
            _memory_info(pynvml, handle)


# ---------------------------------------------------------------------------
# FIX 2: own_vram_gate in planner/runner.py
# ---------------------------------------------------------------------------


class _FakeNvmlRunner:
    """Fake NVML for ``own_vram_gate``. Each proc can omit usedGpuMemory."""

    def __init__(self, procs_by_gpu):
        """procs_by_gpu: {gpu_index: [(pid, usedGpuMemory_or_None)]}"""
        self.procs_by_gpu = procs_by_gpu

    def nvmlDeviceGetCount(self):
        return max(self.procs_by_gpu.keys()) + 1 if self.procs_by_gpu else 0

    def nvmlDeviceGetHandleByIndex(self, i):
        return i

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        entries = []
        for pid, mem in self.procs_by_gpu.get(handle, []):
            if mem is None:
                proc = types.SimpleNamespace(pid=pid)
                # Deliberately no usedGpuMemory attribute.
            else:
                proc = types.SimpleNamespace(pid=pid, usedGpuMemory=mem)
            entries.append(proc)
        return entries


class TestOwnVramGateUsedKnown(unittest.TestCase):
    """The provenance entry distinguishes '0 MiB' from 'value not visible'."""

    def test_proc_without_used_gpu_memory_has_used_known_false(self):
        """Fake proc that omits usedGpuMemory entirely -> used_known=False."""
        nvml = _FakeNvmlRunner({0: [(999, None)]})
        out = own_vram_gate(
            own_pids=[], indices=[0], nvml=nvml, timeout_s=1, sleep=lambda s: None
        )
        self.assertTrue(out["clear"])
        entry = out["foreign"][0]
        self.assertEqual(entry["pid"], 999)
        self.assertEqual(entry["used_mib"], 0)
        self.assertFalse(
            entry["used_known"],
            "used_known must be False when NVML hid the value",
        )

    def test_proc_with_used_gpu_memory_has_used_known_true(self):
        """Fake proc with usedGpuMemory=4 GiB -> correct MiB, used_known=True."""
        nvml = _FakeNvmlRunner({0: [(999, 4 * 1024 * MIB)]})
        out = own_vram_gate(
            own_pids=[], indices=[0], nvml=nvml, timeout_s=1, sleep=lambda s: None
        )
        entry = out["foreign"][0]
        self.assertEqual(entry["used_mib"], 4 * 1024)
        self.assertTrue(
            entry["used_known"],
            "used_known must be True when NVML reported the value",
        )

    def test_zero_usage_is_still_used_known_true(self):
        """A process using exactly 0 MiB is different from NVML hiding it."""
        nvml = _FakeNvmlRunner({0: [(999, 0)]})
        out = own_vram_gate(
            own_pids=[], indices=[0], nvml=nvml, timeout_s=1, sleep=lambda s: None
        )
        entry = out["foreign"][0]
        self.assertEqual(entry["used_mib"], 0)
        self.assertTrue(
            entry["used_known"],
            "0 MiB from NVML is an honest zero -- used_known must be True",
        )


if __name__ == "__main__":
    unittest.main()
