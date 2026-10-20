# SPDX-License-Identifier: Apache-2.0
"""W98 int16 (boot ...dkrnfint4h6ablxcbar1dauer10080701 @ 8a50452564,
2026-10-08 front 09:47:35.906): the rate latch tore down a serving NF boot on
the D park's tail hand-off of an ordinary D->P flip (epoch 70).

MEASURED (front.log 27641/28102, D.log 09:47:30, memts_*.csv):
* the D park published 15 tail parts, 0.370 GiB, into
  ``/dev/shm/pdflip-arena-<tag>/handoff`` (tail_handoff._dir) at 09:47:30;
  memts shmem 61.01 -> 61.51 -> 60.96 GiB at :30/:35/:40 -- P consumed and
  removed them within seconds (the whole flip: shmem_delta -0.37);
* the arena slot files were FULL (ARENA-DROP, "write refused status=4"), so
  the arena itself did not grow;
* cushion 0.40 < floor 1.50 (the expert store's tmpfs displaces the page cache
  for the whole boot), free pool = min(cgroup room 13.97, host MemAvailable)
  < 3.00, headroom 6.48 <= 32 -> W98, while memory.max still had 13.97 GiB.

Mechanism: ``arena_fill_gib`` -- the latch's only reading of a writer with a
known end -- listed the arena directory's top level and skipped the
``handoff/`` subdirectory, so the hand-off's bytes were not in ``touched``
and their rise counted as an UNBOUNDED writer still rising. The same 0.371 GiB
park at 09:40 (epoch 68) passed only because MemAvailable was above 3.00 then.

The hand-off is part of the arena tree: a published file is complete, so it
adds to ``touched`` and ``size`` alike and the rise is priced as a bounded
fill (remaining + floor <= cushion + free pool -> note, no latch). A rise the
arena tree does not show still latches.
"""
from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import host_ledger
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MIB = 1 << 20
GIB = 1 << 30
REAP_MARK = 95.90     # front.py constructs the latch with the CT999 constant
CEILING = 104.0       # memory.max of this boot (FLIP-CUSHION line 28102)
CUSHION = 0.40        # W98 line: cushion=0.40
FREE = 2.90           # < 3.00: the latch fired, so the tick read below 3.00
NONRECLAIM = 89.42    # W98 line: now=89.42
SHMEM0 = 61.32        # shmem before the park; the W98 line read 61.69 after
TAG = "w98int16"
ARENA_MIB = 8         # a FULL arena file (every page materialised)
TAIL_MIB = 16         # > RATE_LATCH_FILL_RISING_GIB (10.24 MiB): "rising"


def _write(path, mib):
    with open(path, "wb") as f:
        f.write(b"\1" * (mib * MIB))
        f.flush()
        os.fsync(f.fileno())


class TestHandoffIsTheArenasWriter(CustomTestCase):
    def setUp(self):
        if not os.path.isdir("/dev/shm"):
            self.skipTest("no /dev/shm (the arena's own tmpfs)")
        self._tmp = tempfile.TemporaryDirectory(dir="/dev/shm")
        self.root = self._tmp.name
        self.arena = os.path.join(self.root, f"pdflip-arena-{TAG}")
        os.makedirs(os.path.join(self.arena, "handoff"))
        _write(os.path.join(self.arena, "arena-786432.bin"), ARENA_MIB)

    def tearDown(self):
        self._tmp.cleanup()

    def _tick(self, latch, t, shmem):
        return latch.observe(
            t, NONRECLAIM, cushion_gib=CUSHION, shmem_gib=shmem, free_gib=FREE,
            ceiling_gib=CEILING,
            free_source="min(cgroup room memory.max-memory.current, host MemAvailable)",
            bounded_shm=host_ledger.arena_fill_gib(TAG, root=self.root),
        )

    def test_the_d_park_tail_handoff_is_priced_as_a_bounded_fill(self):
        """09:47:30-35: the park writes its tails into arena/handoff and shmem
        rises by exactly those bytes. RED on 8a50452564: W98 (the hand-off is
        invisible to arena_fill_gib, so the rise reads as unbounded)."""
        latch = host_ledger.RateLatch(reap_mark_gib=REAP_MARK)
        self.assertIsNone(self._tick(latch, 0.0, SHMEM0))
        _write(os.path.join(self.arena, "handoff", "pdflip-70-676.tail.dpark0-2243.pt"),
               TAIL_MIB)
        line = self._tick(latch, 0.5, SHMEM0 + TAIL_MIB / 1024)
        self.assertFalse(latch.latched, line)
        self.assertIsNotNone(line)
        self.assertNotIn("W98", line)
        self.assertIn("BOUNDED-FILL", line)

    def test_a_rise_the_arena_tree_does_not_show_still_latches(self):
        """CONTROL at the same readings: shmem +16 MiB with nothing new in the
        arena or its hand-off -- an unbounded writer, W98 as before. Guards
        against a reader that would credit every /dev/shm byte to the arena."""
        latch = host_ledger.RateLatch(reap_mark_gib=REAP_MARK)
        self.assertIsNone(self._tick(latch, 0.0, SHMEM0))
        _write(os.path.join(self.root, "not-the-arena.bin"), TAIL_MIB)
        line = self._tick(latch, 0.5, SHMEM0 + TAIL_MIB / 1024)
        self.assertTrue(latch.latched)
        self.assertIn("W98", line or "")

    def test_the_reader_counts_handoff_files_as_complete(self):
        """A published hand-off file has nothing left to write: it adds to
        touched and size alike, so it never inflates the arena's remainder."""
        t0, s0 = host_ledger.arena_fill_gib(TAG, root=self.root)
        _write(os.path.join(self.arena, "handoff", "pdflip-70-679.json"), TAIL_MIB)
        t1, s1 = host_ledger.arena_fill_gib(TAG, root=self.root)
        self.assertAlmostEqual(s1 - s0, TAIL_MIB / 1024, places=6)
        self.assertAlmostEqual(t1 - t0, TAIL_MIB / 1024, places=6)
        self.assertAlmostEqual(s1 - t1, s0 - t0, places=6)


if __name__ == "__main__":
    unittest.main()
