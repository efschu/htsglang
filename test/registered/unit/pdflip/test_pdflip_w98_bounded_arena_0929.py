# SPDX-License-Identifier: Apache-2.0
"""W98 dmatrix (boot ...z30x2bar1dauer09291358, 424346f693, 2026-09-29 14:07:48):
the rate latch tore down a serving boot on the arena's own, bounded fill.

MEASURED (front.log 1776-1783, mem_*bar1dauer.csv, state events.jsonl):
* the cushion (file - shmem) sat BELOW the 1.5 GiB floor for the WHOLE boot
  (every flip 0.32-1.33 GiB): the expert store's tmpfs displaces the page
  cache, the fnFL2 v14 shape;
* cg_room fell under the 3.0 GiB free-pool bound at flip 14 (2.86) and 15
  (2.73), so the free-pool exemption no longer applied;
* shmem rose 49.72 -> 53.95 GiB over the serving with anon flat (24.5 GiB):
  the arena files (5461 x 768 KiB + 32 x 56 MiB = 5.75 GiB of tmpfs)
  materialise their pages as the KV arrives -- a writer that STOPS at its size;
* the P->D flip of bs4 x 40k (epoch 16) touched +0.06 GiB more and W98 fired
  1 s after it: bs4 x 40k decoded 0 rounds, bs6 never started, the boot went
  down by the named STOP (not at the stop: the STOP was W98's teardown).

The fix prices a rise of the arena alone against what can absorb it:
remaining (size - touched) + floor <= cushion + free pool -> a note, no latch.
Any rise of the rest of shmem (writers without a bound) latches as before.
"""
from __future__ import annotations

import inspect
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import host_ledger
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

GIB = 2**30
REAP_MARK = 98.99   # the boot's REAP MODEL line
CEILING = 84.0      # memory.max of the Docker form
ARENA_SIZE = 5.75


def _observe(latch, t, *, shmem, cushion, free, touched, nonreclaim=80.20):
    """observe() as the front calls it; a tree without the bounded writer
    (424346f693) gets the call without it -- its behaviour is the red one."""
    kw = dict(cushion_gib=cushion, shmem_gib=shmem, free_gib=free,
              ceiling_gib=CEILING, free_source="cgroup")
    if "bounded_shm" in inspect.signature(latch.observe).parameters:
        kw["bounded_shm"] = (touched, ARENA_SIZE)
    return latch.observe(t, nonreclaim, **kw)


def _run(seq):
    latch = host_ledger.RateLatch(reap_mark_gib=REAP_MARK)
    lines = []
    for i, (shmem, cushion, free, touched) in enumerate(seq):
        ln = _observe(latch, 1000.0 + 0.5 * i, shmem=shmem, cushion=cushion,
                      free=free, touched=touched)
        if ln:
            lines.append(ln)
    return latch, lines


class TestBoundedArenaFill(CustomTestCase):
    def test_metal_shape_the_arena_tail_does_not_latch(self):
        """14:07:47-48: cushion 0.93 -> 0.98, cg room 2.73, shmem +0.05 GiB,
        all of it the arena (touched 5.50 -> 5.55 of 5.75: 0.20 remaining,
        + floor 1.50 = 1.70 <= absorb 3.71). RED on 424346f693: W98."""
        latch, lines = _run([
            (53.90, 0.93, 2.73, 5.50),
            (53.95, 0.98, 2.73, 5.55),
        ])
        self.assertFalse(latch.latched, lines)
        self.assertFalse(any("W98" in ln for ln in lines), lines)
        self.assertTrue(any("BOUNDED-FILL" in ln for ln in lines), lines)

    def test_a_large_arena_remainder_still_latches(self):
        """Same readings, but the arena has 4.7 GiB still to materialise:
        4.7 + 1.5 > 3.71 -- the latch fires as it did."""
        latch, lines = _run([
            (48.90, 0.93, 2.73, 1.00),
            (48.95, 0.98, 2.73, 1.05),
        ])
        self.assertTrue(latch.latched)
        self.assertTrue(any("W98" in ln for ln in lines), lines)

    def test_a_rise_outside_the_arena_still_latches(self):
        """shmem +0.05 GiB with the arena unchanged: an unbounded writer."""
        latch, lines = _run([
            (53.90, 0.93, 2.73, 5.50),
            (53.95, 0.98, 2.73, 5.50),
        ])
        self.assertTrue(latch.latched)
        self.assertTrue(any("W98" in ln for ln in lines), lines)

    def test_the_note_is_not_a_verdict(self):
        latch, lines = _run([
            (53.90, 0.93, 2.73, 5.50),
            (53.95, 0.98, 2.73, 5.55),
        ])
        from flliper.srt.pdflip.front import latch_line_is_verdict

        for ln in lines:
            self.assertFalse(latch_line_is_verdict(ln, latch), ln)

    def test_arena_fill_reads_the_charged_pages_of_a_sparse_file(self):
        fn = getattr(host_ledger, "arena_fill_gib", None)
        self.assertIsNotNone(fn, "no reader for the arena's touched pages")
        # tmpfs, the arena's own filesystem: st_blocks is exactly what is
        # charged (a compressing fs like ZFS would under-report it)
        if not os.path.isdir("/dev/shm"):
            self.skipTest("no /dev/shm")
        with tempfile.TemporaryDirectory(dir="/dev/shm") as root:
            d = os.path.join(root, "pdflip-arena-t1")
            os.makedirs(d)
            p = os.path.join(d, "arena-786432.bin")
            with open(p, "wb") as f:
                f.truncate(64 << 20)             # 64 MiB, sparse
                f.write(b"\1" * (4 << 20))       # 4 MiB materialised
                f.flush()
                os.fsync(f.fileno())
            touched, size = fn("t1", root=root)
            self.assertAlmostEqual(size, 64 / 1024, places=6)
            self.assertGreaterEqual(touched, 4 / 1024 - 1e-9)
            self.assertLess(touched, size)
            self.assertIsNone(fn("absent", root=root))
            self.assertIsNone(fn(None, root=root))

    def test_the_front_passes_the_arena_to_the_latch(self):
        from flliper.srt.pdflip import front

        src = inspect.getsource(front)
        i = src.index("_line = rate_latch.observe(")
        self.assertIn("bounded_shm=host_ledger.arena_fill_gib(self.tag)", src[i:i + 2500])


if __name__ == "__main__":
    unittest.main()
