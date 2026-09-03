"""#1158 (c) -- THE OPENER HONOURS THE CAP TOO.

THE SPECIMEN (boot weg1b3, 23:59:54 -> 00:06:47). PP0 and PP2 opened the
#1033c cutover warmup window (`cold_build_window` wraps the whole forward) and
waited on a peer that never joined. The PEERS reading a published window stop
extending at `barlink_build_window.build_cap_s()` (60 s on this rig); the
OPENER's own readers -- `barlink_liveness.wait_timeout_s` and
`jit_cold_build.resolve_timeout_cycles` -- multiplied by the bare x40 with no
cap, so the opener sat past every peer deadline until the 300 s scheduler
watchdog ('#1033c CUTOVER FORWARD WARMUP begin' with no done, then
'forward-counter-frozen').

THE BOUND: `min(base * mult, base + cap)` on BOTH readers, from ONE formula
(`jit_cold_build.capped_cold_build_deadline`). MUTANT: remove the min on one
reader -> that reader's assertion below goes red.
"""

import inspect
import os
import unittest
from unittest import mock

from sglang.srt.distributed.device_communicators import barlink_liveness as live
from sglang.srt.utils import jit_cold_build as jcb
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_BASE = 60_000_000_000  # BarlinkDeviceTransport._TIMEOUT_CYCLES
# Read with a default so the module COLLECTS on a tree without the fix and
# every case goes red on its own assertion (a collection error is a
# coarser red than the per-reader failures this file exists to show).
_HZ = getattr(jcb, "_NOMINAL_CYCLES_PER_S", 2_000_000_000)


def _env(**kv):
    return mock.patch.dict(os.environ, {k: str(v) for k, v in kv.items()}, clear=False)


class TheOpenerIsCappedLikeItsPeers(CustomTestCase):
    def test_host_reader_is_base_plus_cap_when_the_multiplier_would_exceed_it(self):
        with _env(
            SGLANG_BARLINK_PEER_TIMEOUT_S=10.0,
            SGLANG_JIT_COLD_BUILD_TIMEOUT_MULT=40,
            SGLANG_BARLINK_BUILD_WINDOW_CAP_S=60,
        ):
            self.assertEqual(live.wait_timeout_s(), 10.0)
            with jcb.cold_build_window("1158"):
                self.assertEqual(live.wait_timeout_s(), 70.0, "base + cap, not base x 40")
            self.assertEqual(live.wait_timeout_s(), 10.0)

    def test_device_reader_is_scaled_identically(self):
        with _env(
            SGLANG_JIT_COLD_BUILD_TIMEOUT_MULT=40,
            SGLANG_BARLINK_BUILD_WINDOW_CAP_S=60,
        ):
            self.assertEqual(jcb.resolve_timeout_cycles(_BASE), _BASE)
            with jcb.cold_build_window("1158"):
                self.assertEqual(
                    jcb.resolve_timeout_cycles(_BASE),
                    _BASE + 60 * _HZ,
                    "base + cap in nominal cycles, not base x 40",
                )
            self.assertEqual(jcb.resolve_timeout_cycles(_BASE), _BASE)

    def test_a_non_binding_cap_leaves_the_multiplier_alone(self):
        """The pre-#1158 pins (x7 under the 900 s default) still hold: the
        cap tightens nothing a legitimate build needs."""
        with _env(
            SGLANG_BARLINK_PEER_TIMEOUT_S=100.0,
            SGLANG_JIT_COLD_BUILD_TIMEOUT_MULT=7,
            SGLANG_BARLINK_BUILD_WINDOW_CAP_S=900,
        ):
            with jcb.cold_build_window("1158"):
                self.assertEqual(live.wait_timeout_s(), 700.0)
                self.assertEqual(jcb.resolve_timeout_cycles(_BASE), _BASE * 7)

    def test_a_zero_cap_disables_the_extension_on_both_readers(self):
        """The peers' bisect switch (cap 0 = no extension) governs the opener too."""
        with _env(
            SGLANG_BARLINK_PEER_TIMEOUT_S=10.0,
            SGLANG_JIT_COLD_BUILD_TIMEOUT_MULT=40,
            SGLANG_BARLINK_BUILD_WINDOW_CAP_S=0,
        ):
            with jcb.cold_build_window("1158"):
                self.assertEqual(live.wait_timeout_s(), 10.0)
                self.assertEqual(jcb.resolve_timeout_cycles(_BASE), _BASE)

    def test_one_formula_feeds_both_readers(self):
        self.assertIn("capped_cold_build_deadline", inspect.getsource(live.wait_timeout_s))
        self.assertIn(
            "capped_cold_build_deadline", inspect.getsource(jcb.resolve_timeout_cycles)
        )
        self.assertEqual(jcb.capped_cold_build_deadline(10.0, 60.0), min(10.0 * jcb.cold_build_timeout_mult(), 70.0))


if __name__ == "__main__":
    unittest.main()
