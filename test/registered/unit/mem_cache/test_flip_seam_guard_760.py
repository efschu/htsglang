# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#760: the HiCache phase guard follows the flip runtime, not the routing global.

THE GAP THESE TESTS PIN. The guard's phase source used to be the
parallel_state routing global, which is toggled INSIDE the cutover -- one
step among many. For the whole seam (waves moving KV rows, movers releasing
the outgoing backing, the cutover rebuilding topology) that global names one
of the two phases while pool bytes are in motion, so a guard reading it can
approve device-tier I/O mid-seam. Both #760 crash specimens are copies that
rode exactly that window into released backing, 3 s after a pp_to_tp
cutover, below the Python seam.

The flip runtime itself is the only component that knows the seam's extent,
and its ``_phase`` field is the value the PHASE-FLIP DONE line reports --
which the 3 s crash correlation proves truthful. So it registers itself as
the guard's phase authority, and these tests pin:

  1. no authority registered -> the routing-global fallback (deployments
     without a flip runtime stay byte-identical);
  2. a registered authority's SEAM disarms the device tier even when phase
     and binding agree (red-first: impossible before this change);
  3. a registered authority's phase WINS over a disagreeing routing global,
     and the disagreement is logged (the #754-shape instrument);
  4. a dead (garbage-collected) authority never gates I/O;
  5. the controller's seam quiesce drains both device-tier streams;
  6. the runtime's quiesce helper reaches the controller through the census
     scheduler handle and survives its absence.
"""

from __future__ import annotations

import gc
import types
import unittest

from sglang.srt.mem_cache import hicache_phase_guard as guard
from sglang.srt.mem_cache.hicache_phase_binding import binding_state


class _FakeRuntime:
    """The two attributes the authority contract reads."""

    def __init__(self, phase: str = "pp", seam: bool = False):
        self.phase = phase
        self.hicache_seam_active = seam


class _GuardCase(unittest.TestCase):
    def setUp(self):
        guard.clear_flip_phase_authority()
        guard.reset_warnings()
        binding_state().reset()

    def tearDown(self):
        guard.clear_flip_phase_authority()
        guard.reset_warnings()
        binding_state().reset()


class TestFallbackWithoutAuthority(_GuardCase):
    def test_no_authority_reads_the_routing_global(self):
        # No flip runtime was ever built: the guard reduces to the #718
        # behaviour exactly (routing off -> "pp" -> matches the boot binding).
        self.assertEqual(guard.active_phase(), "pp")
        self.assertFalse(guard.device_tier_disarmed("write"))


class TestSeamDisarmsEverything(_GuardCase):
    def test_seam_disarms_even_a_matching_binding(self):
        # Phase pp, binding pp -- WITHOUT the seam this is armed. The seam
        # must override the match: bytes are moving, nothing may bind.
        rt = _FakeRuntime(phase="pp", seam=True)
        guard.register_flip_phase_authority(rt)
        self.assertEqual(guard.active_phase(), guard.SEAM)
        self.assertTrue(
            guard.device_tier_disarmed("write"),
            "device-tier I/O must be refused during the flip seam even when "
            "the active phase equals the bound phase",
        )
        self.assertTrue(guard.device_tier_disarmed("load"))

    def test_the_seam_ends_when_the_runtime_says_so(self):
        rt = _FakeRuntime(phase="pp", seam=True)
        guard.register_flip_phase_authority(rt)
        self.assertTrue(guard.device_tier_disarmed("write"))
        rt.hicache_seam_active = False
        self.assertFalse(guard.device_tier_disarmed("write"))


class TestAuthorityBeatsTheRoutingGlobal(_GuardCase):
    def test_authority_phase_wins_and_the_disagreement_is_logged(self):
        # The runtime says tp; the routing global (never touched in this
        # process) says pp. The guard must follow the runtime -- and with the
        # boot binding still pp, that means DISARMED. Before this change the
        # guard read the global and would have approved the copy.
        rt = _FakeRuntime(phase="tp", seam=False)
        guard.register_flip_phase_authority(rt)
        with self.assertLogs(guard.logger, level="WARNING") as captured:
            self.assertEqual(guard.active_phase(), "tp")
        self.assertTrue(
            any("PREDICATE DISAGREEMENT" in line for line in captured.output),
            "a phase disagreement between runtime and routing global is the "
            "#754 shape and must be logged, once",
        )
        self.assertTrue(guard.device_tier_disarmed("write"))

    def test_agreement_logs_nothing(self):
        rt = _FakeRuntime(phase="pp", seam=False)
        guard.register_flip_phase_authority(rt)
        self.assertEqual(guard.active_phase(), "pp")
        self.assertFalse(guard.device_tier_disarmed("write"))


class TestADeadAuthorityNeverGates(_GuardCase):
    def test_collected_runtime_falls_back(self):
        guard.register_flip_phase_authority(_FakeRuntime(phase="tp", seam=True))
        gc.collect()  # the only reference was the weakref
        self.assertEqual(guard.active_phase(), "pp")
        self.assertFalse(guard.device_tier_disarmed("write"))


class _AllocMustNotRun:
    """A pool whose alloc failing loudly proves the guard ran FIRST."""

    def alloc(self, n):
        raise AssertionError(
            "the guard must refuse before the host/device pool is touched"
        )


class TestHybridControllerIsGuardedToo(_GuardCase):
    """#760: the metal specimen of 2026-08-19 20:40 went through
    HybridCacheController.write -- an OVERRIDE of the guarded base method
    that never asked the guard. These pin both overrides to the same
    contract as the base class: refuse (return None) while disarmed,
    BEFORE touching any pool."""

    def _seam(self):
        # Held on self: the registration is weak by design, so a discarded
        # runtime would fall back to the routing global mid-test.
        self._rt = _FakeRuntime(phase="pp", seam=True)
        guard.register_flip_phase_authority(self._rt)

    def test_hybrid_write_refuses_while_disarmed(self):
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )

        self._seam()
        cc = object.__new__(HybridCacheController)
        cc.mem_pool_host = _AllocMustNotRun()
        self.assertIsNone(HybridCacheController.write(cc, _FakeTensor()))

    def test_hybrid_load_refuses_while_disarmed(self):
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )

        self._seam()
        cc = object.__new__(HybridCacheController)
        cc.mem_pool_device_allocator = _AllocMustNotRun()
        self.assertIsNone(HybridCacheController.load(cc, _FakeTensor()))


class _FakeTensor:
    """Just enough tensor for the pre-guard surface of write()/load()."""

    def __len__(self):
        return 4

    def numel(self):
        return 4


class _RecordingStream:
    def __init__(self, log, name):
        self._log = log
        self._name = name

    def synchronize(self):
        self._log.append(self._name)


class TestQuiesceDrainsBothStreams(unittest.TestCase):
    def test_quiesce_synchronizes_write_and_load_streams(self):
        from sglang.srt.managers.cache_controller import HiCacheController

        calls: list[str] = []
        fake = types.SimpleNamespace(
            write_stream=_RecordingStream(calls, "write"),
            load_stream=_RecordingStream(calls, "load"),
        )
        elapsed = HiCacheController.quiesce_device_io(fake, "test seam")
        self.assertEqual(calls, ["write", "load"])
        self.assertGreaterEqual(elapsed, 0.0)


class TestRuntimeQuiesceHelper(unittest.TestCase):
    def _runtime_shell(self):
        # The helper only reads _census_scheduler; no full runtime needed.
        from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

        rt = object.__new__(PhaseFlipRuntime)
        rt._census_scheduler = None
        return rt

    def test_reaches_the_controller_through_the_scheduler(self):
        rt = self._runtime_shell()
        seen: list[str] = []
        controller = types.SimpleNamespace(
            quiesce_device_io=lambda reason: seen.append(reason)
        )
        rt._census_scheduler = types.SimpleNamespace(
            tree_cache=types.SimpleNamespace(cache_controller=controller)
        )
        rt._quiesce_hicache("pp_to_tp")
        self.assertEqual(seen, ["phase flip pp_to_tp"])

    def test_survives_an_absent_scheduler_and_controller(self):
        rt = self._runtime_shell()
        rt._quiesce_hicache("pp_to_tp")  # no scheduler: must be a no-op
        rt._census_scheduler = types.SimpleNamespace(tree_cache=None)
        rt._quiesce_hicache("tp_to_pp")

    def test_a_raising_quiesce_never_raises_into_the_seam(self):
        rt = self._runtime_shell()

        def _boom(reason):
            raise RuntimeError("stream died")

        rt._census_scheduler = types.SimpleNamespace(
            tree_cache=types.SimpleNamespace(
                cache_controller=types.SimpleNamespace(quiesce_device_io=_boom)
            )
        )
        rt._quiesce_hicache("pp_to_tp")  # must log, not raise


if __name__ == "__main__":
    unittest.main()
