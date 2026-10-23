"""barlink-switch-1023: ONE calibration for the switch-direct topology.

The shipped ``DEFAULT_ROUND_US`` / ``DEFAULT_WIRE_GBPS`` now come from the
switch-direct arm (ACS 0011, 2026-10-23 09:10Z, world 2). The task asked
whether the binder has to calibrate per topology (ACS redirect on / off).
The measured answer, pinned here:

* the round term does NOT move with the topology (155.5 / 156.3 / 148.1 us),
  only the wire term does (20.45 switch-direct vs 11.93 / 12.26 GB/s host
  detour);
* no DECISION moves with it either: ``round_budget`` is scale-invariant and
  has no root while one round's bar1 time (round term + that round's wire
  bytes) stays below the next rung's time for the same payload; every
  measured calibration -- both topologies and the retired 0907 one -- clears
  that by >= 4x at every geometry the benches ran (worst: 0907, world 3,
  w16, 4.85x), so ``coverage_ceiling``
  answers "unbounded" for all four ops;
* a negative control (a 20-GB/s next rung, faster than one bar1 round) DOES
  produce a ceiling, so
  the invariance test can fail when a future measurement makes the topology
  matter.

The operator made ACS 0011 permanent (``pci=disable_acs_redir``), so there
is no topology switch to build; this file is where a measurement that
demanded one would show up first.

CPU-only. Pure arithmetic and a GPU-less stub.
"""

import inspect
import os
import unittest
from unittest import mock

from flliper.srt.distributed.device_communicators import barlink_bar1
from flliper.srt.distributed.device_communicators.barlink_bar1 import (
    DEFAULT_CALIBRATION_SOURCE,
    DEFAULT_NEXT_RUNG_GBPS,
    DEFAULT_ROUND_US,
    DEFAULT_WIRE_GBPS,
    ROUND_CAP_AUTO,
    BarlinkBar1Transport,
    round_budget,
    wire_bytes_for,
)
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

#: Every calibration measured on this rig, ``name -> (round_us, wire_gbps)``.
#: Sources: /spinning/gpu-arb/weg2/barlink-1023/*.json (fits recomputed by the
#: bench's own ``_least_squares``) and roundbench_fixed_0907.json.
MEASURED = {
    "switch_direct_0909": (155.52106929144708, 20.45292680652535),
    "host_detour_0909": (156.34038261850637, 11.934881900879528),
    "host_detour_0901": (148.1, 12.26),
    "pre_switch_0907_world3": (323.2, 6.02),
}

#: The six switch-direct rows (window MiB, nbytes, rounds, wire, ms).
SWITCH_ROWS = (
    (16, 25165824, 5, 25165824, 2.0489715000167052),
    (16, 100663296, 19, 100663296, 7.853801999999632),
    (40, 25165824, 2, 25165824, 1.6057719999480469),
    (40, 100663296, 8, 100663296, 6.188763499949346),
    (128, 25165824, 1, 25165824, 1.3701689999834343),
    (128, 100663296, 3, 100663296, 5.365819000189731),
)

#: ``chunk_max`` the benches reported, per (world, window MiB).
CHUNK_MAX = {
    (2, 16): 2703360, (2, 40): 6766592, (2, 128): 21667840,
    (3, 16): 1351680, (3, 40): 3383296, (3, 128): 10829824,
}

OPS = ("all_reduce", "all_gather", "broadcast", "all_to_all")


def _stub(world: int, chunk_max: int, round_us: float, wire_gbps: float,
          next_rung_gbps: float = DEFAULT_NEXT_RUNG_GBPS):
    t = BarlinkBar1Transport.__new__(BarlinkBar1Transport)
    t.world = world
    t.rank = 0
    t.group = "calib"
    t._geo = {"chunk_max": chunk_max, "a2a_slot": chunk_max}
    for attr in ("ar_max_rounds", "ag_max_rounds", "bc_max_rounds",
                 "a2a_max_rounds"):
        setattr(t, attr, ROUND_CAP_AUTO)
    t.round_us = round_us
    t.wire_gbps = wire_gbps
    t.next_rung_gbps = next_rung_gbps
    return t


class TestShippedConstantsAreTheSwitchDirectRun(CustomTestCase):
    def test_constants_are_the_switch_direct_fit(self):
        r, w = MEASURED["switch_direct_0909"]
        self.assertAlmostEqual(DEFAULT_ROUND_US, r, delta=0.1)
        self.assertAlmostEqual(DEFAULT_WIRE_GBPS, w, delta=0.01)

    def test_every_switch_direct_row_is_reproduced_within_0_1_ms(self):
        for wmib, nbytes, rounds, wire, ms in SWITCH_ROWS:
            pred = (DEFAULT_ROUND_US / 1e3 * rounds
                    + wire / (DEFAULT_WIRE_GBPS * 1e9) * 1e3)
            self.assertLess(abs(pred - ms), 0.1, (wmib, nbytes, pred, ms))

    def test_the_host_detour_rows_are_NOT_reproduced(self):
        """Discriminating power: the shipped wire term is the switch one.
        The host-detour w128 96 MiB row (8.892 ms) misses by > 3 ms."""
        pred = DEFAULT_ROUND_US / 1e3 * 3 + 100663296 / (DEFAULT_WIRE_GBPS * 1e9) * 1e3
        self.assertGreater(8.891564500004279 - pred, 3.0)

    def test_the_bind_proof_names_artifact_topology_and_world3_caveat(self):
        src = inspect.getsource(barlink_bar1)
        a = src.index("DEFAULT_CALIBRATION_SOURCE = (")
        block = src[a:src.index("def parse_round_cap", a)]
        for needle in ("p2p_rebar_1023_0909.B_switch_direct.json",
                       "/spinning/gpu-arb/weg2/barlink-1023/",
                       "ACS 0011", "residual", "WORLD 3 IS NOT DESCRIBED",
                       "p2p3sw_rebar_1023_0921.B_switch_direct.json",
                       "foreign desktop load"):
            self.assertIn(needle, block, needle)
        self.assertNotIn("INTERIM", block)


class TestNoDecisionDependsOnTheTopology(CustomTestCase):
    def test_every_measured_calibration_clears_the_root_condition(self):
        """Per round: next-rung ms over bar1 ms, >= 4x everywhere."""
        worst = None
        for (world, wmib), cm in CHUNK_MAX.items():
            unit = (cm // 16) * world * 16
            wire = wire_bytes_for("all_reduce", unit, world)
            nxt = unit / (DEFAULT_NEXT_RUNG_GBPS * 1e9) * 1e3
            for name, (r, w) in MEASURED.items():
                bar1 = r / 1e3 + wire / (w * 1e9) * 1e3
                ratio = nxt / bar1
                worst = ratio if worst is None else min(worst, ratio)
                self.assertGreater(ratio, 4.0, (world, wmib, name, ratio))
        self.assertIsNotNone(worst)

    def test_coverage_is_unbounded_for_every_op_geometry_and_calibration(self):
        for (world, wmib), cm in CHUNK_MAX.items():
            for name, (r, w) in MEASURED.items():
                t = _stub(world, cm, r, w)
                for op in OPS:
                    self.assertIsNone(
                        t.coverage_ceiling(op),
                        f"{name} world {world} w{wmib} {op}")

    def test_budgets_agree_on_admit_for_every_bench_payload(self):
        """Same verdict (budget >= rounds needed) under every calibration."""
        for (world, wmib), cm in CHUNK_MAX.items():
            unit = (cm // 16) * world * 16
            for nbytes in (24 << 20, 96 << 20, 1 << 30):
                rounds = -(-nbytes // unit)
                wire = wire_bytes_for("all_reduce", nbytes, world)
                verdicts = {
                    name: round_budget(nbytes, wire, r, w,
                                       DEFAULT_NEXT_RUNG_GBPS) >= rounds
                    for name, (r, w) in MEASURED.items()
                }
                self.assertEqual(set(verdicts.values()), {True},
                                 (world, wmib, nbytes, verdicts))

    def test_negative_control_a_slow_wire_does_bind(self):
        """Mutant direction: if a topology ever priced the wire below twice
        the rung, the derived ceiling would appear -- the tests above would
        then fail instead of passing silently."""
        t = _stub(2, CHUNK_MAX[(2, 16)], DEFAULT_ROUND_US, DEFAULT_WIRE_GBPS,
                  next_rung_gbps=20.0)
        t.handles = lambda op, n: True       # the physical refusals are not under test
        self.assertIsNotNone(t.coverage_ceiling("all_reduce"))


class TestTheBootLineNamesItsCalibration(CustomTestCase):
    def _l1(self, t):
        t.window_bytes = 16 << 20
        t._up = True
        return t.coverage_lines()[0]

    def test_defaults_name_the_artifact(self):
        t = _stub(2, CHUNK_MAX[(2, 16)], DEFAULT_ROUND_US, DEFAULT_WIRE_GBPS)
        line = self._l1(t)
        self.assertIn(DEFAULT_CALIBRATION_SOURCE, line)
        self.assertIn("calibration MEASURED", line)

    def test_an_env_override_is_named_as_such(self):
        t = _stub(2, CHUNK_MAX[(2, 16)], 300.0, DEFAULT_WIRE_GBPS)
        line = self._l1(t)
        self.assertIn("calibration from the environment", line)
        self.assertNotIn("calibration MEASURED", line)


if __name__ == "__main__":
    unittest.main()
