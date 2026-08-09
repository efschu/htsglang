"""#631: a flip must not spend the VRAM reserve to stage its own bytes.

MEASURED FAILURE this pins (2026-08-09, metal, all three ranks died):

    one session, 276214 tokens, 0.995 of the PP pool
    -> policy flip -> kv_reshard._exchange -> torch.empty(584 MiB)
    -> torch.OutOfMemoryError ("600.38 MiB is free")
    -> Fatal Python error: Aborted

The flip is not free of memory. It packs this rank's outgoing rows and
pre-allocates one receive buffer per peer, out of the SAME device memory
the KV pool has been filling. Nothing checked whether those buffers fit,
so a full pool turned a routine layout change into an instance-wide crash
-- and the free VRAM at that moment (600 MiB) was already below the
1024 MiB corridor floor, so even a flip that had squeezed through would
have broken the corridor.

The fix folds an affordability term into the EXISTING pre-flight verdict,
the one that already refuses a flip whose live set does not fit the target
pool. Same properties, deliberately: computed from the plan before a byte
is allocated, reduced across ranks so the abandon is unanimous, and
answered by abandoning the FLIP rather than the SERVER.

Two quantities that must not be conflated, and both directions are wrong:

* the allocator's cached-but-unhanded-out bytes are reusable and invisible
  to NVML, so spending them cannot move the corridor number -- counting
  only driver-free would abandon flips that fit;
* driver bytes DO move it, so only the amount above the reserve is
  spendable -- counting the allocator's view alone would spend the
  corridor.
"""

import unittest

from sglang.srt.managers.phase_flip_runtime import (
    DEFAULT_STAGING_RESERVE_BYTES,
    PhaseFlipRuntime,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MIB = 1024 * 1024


class _Probe:
    """A stand-in for the two memory readings, in MiB for readability."""

    def __init__(self, driver_free_mib, cached_free_mib):
        self.driver_free = int(driver_free_mib) * MIB
        self.cached_free = int(cached_free_mib) * MIB

    def __call__(self):
        return self.driver_free, self.cached_free


def _runtime(probe, reserve_mib=1024):
    r = PhaseFlipRuntime.__new__(PhaseFlipRuntime)
    r._mem_probe = probe
    r._staging_reserve_bytes = int(reserve_mib) * MIB
    return r


class TestStagingReserveDefault(CustomTestCase):
    def test_the_default_is_the_corridor_floor(self):
        # 1024 MiB, the same promise --rank-user-reserve-mib makes. If this
        # ever drifts, the flip starts spending the user's reserve.
        self.assertEqual(DEFAULT_STAGING_RESERVE_BYTES, 1024 * MIB)


class TestStagingAffordable(CustomTestCase):
    def test_the_measured_crash_is_now_refused(self):
        # The metal numbers: 600 MiB driver-free, nothing cached, a 584 MiB
        # buffer. Previously allocated and fatal; now refused.
        ok, detail = _runtime(_Probe(600, 0))._staging_affordable(584 * MIB)
        self.assertFalse(ok)
        self.assertIn("584", detail)
        self.assertIn("reserve", detail)

    def test_room_above_the_reserve_is_spendable(self):
        # 2048 free, reserve 1024 -> 1024 spendable; 584 fits.
        ok, detail = _runtime(_Probe(2048, 0))._staging_affordable(584 * MIB)
        self.assertTrue(ok)
        self.assertEqual(detail, "")

    def test_the_allocator_cache_counts_and_does_not_touch_the_corridor(self):
        # Driver-free is AT the reserve (nothing spendable from the driver),
        # but the allocator already holds 800 MiB it can hand back. Those
        # bytes are invisible to NVML, so the corridor is untouched and the
        # flip is affordable. Counting only driver-free would abandon here.
        ok, _ = _runtime(_Probe(1024, 800))._staging_affordable(584 * MIB)
        self.assertTrue(ok)

    def test_the_cache_does_not_excuse_dipping_below_the_reserve(self):
        # 200 MiB cached, driver free BELOW the reserve: spendable is the
        # cache only, so a 584 MiB staging is still refused. The falsifier
        # for "just add the two numbers".
        ok, _ = _runtime(_Probe(700, 200))._staging_affordable(584 * MIB)
        self.assertFalse(ok)

    def test_exactly_fitting_is_allowed(self):
        # Boundary: spendable == needed. Refusing here would abandon flips
        # that provably fit.
        ok, _ = _runtime(_Probe(1024 + 584, 0))._staging_affordable(584 * MIB)
        self.assertTrue(ok)

    def test_one_byte_over_is_refused(self):
        ok, _ = _runtime(_Probe(1024 + 584, 0))._staging_affordable(584 * MIB + 1)
        self.assertFalse(ok)

    def test_a_larger_reserve_is_honoured(self):
        # The reserve is configuration, not a constant: at 2048 MiB the same
        # memory picture refuses.
        ok, _ = _runtime(_Probe(2048, 0), reserve_mib=2048)._staging_affordable(
            584 * MIB
        )
        self.assertFalse(ok)


class TestStagingBytesFromThePlan(CustomTestCase):
    """The byte count must match what the exchange actually allocates."""

    class _Plan:
        # One send peer (2 layers x 4 rows) and one recv peer (3 layers x
        # 2 rows), shaped like PhaseFlipTransition's fields.
        def __init__(self):
            self.send_layers = {1: [0, 1]}
            self.recv_layers = {1: [0, 1, 2]}
            self.send_rows = {1: _N(4)}
            self.recv_rows = {1: _N(2)}

    class _Side:
        def __init__(self, row_nbytes):
            self._n = row_nbytes

        def row_nbytes(self, _layer):
            return self._n

    def test_outgoing_is_counted_twice_and_incoming_once(self):
        r = _runtime(_Probe(0, 0))
        r._src_layer_idx = lambda _d, f: f
        r._dst_layer_idx = lambda _d, f: f
        src = self._Side(1000)
        dst = self._Side(100)
        got = r._staging_bytes(self._Plan(), "pp_to_tp", src, dst)
        # out: (2 layers * 4 rows * 1000 + 8) * 2 = 16016
        # in : 3 layers * 2 rows * 100 + 8       =   608
        self.assertEqual(got, 16016 + 608)

    def test_an_empty_plan_stages_nothing(self):
        r = _runtime(_Probe(0, 0))
        plan = self._Plan()
        plan.send_layers = {}
        plan.recv_layers = {}
        self.assertEqual(r._staging_bytes(plan, "pp_to_tp", None, None), 0)


class _N:
    """Minimal stand-in for a row tensor: only numel() is read."""

    def __init__(self, n):
        self._n = n

    def numel(self):
        return self._n


if __name__ == "__main__":
    unittest.main()
