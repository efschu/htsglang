"""#796: runtime_set_backing_tokens must say what it was asked and what it had.

The shrink investigation stalled on a contradiction that could not be resolved
from outside the process: one reading of the pool reported `current` as one
number while the target span implied another, and both were said to come from
the same committed-by-offset map. The quantity that separates those two
readings is the pool's own `size` at the moment of the call, together with
`uniform_backed_rows` -- what the arena actually has mapped in EVERY buffer.

`runtime_set_backing_tokens` had ZERO logger calls, so neither was observable
on a live boot. This test pins the instrumentation that makes them observable.

It exercises the real entry point (not the owner underneath it) in BOTH
directions -- grow and shrink -- plus the no-op branch, because a line that
only fires on one branch would leave exactly half the question open.
"""

import logging
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu")

LOGGER_NAME = "sglang.srt.mem_cache.memory_pool"


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA (driver VMM API)")
class TestBackingDialLogging(CustomTestCase):
    def _pool(self, size):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        return MHATokenToKVPool(
            size=size,
            page_size=1,
            dtype=torch.float16,
            head_num=4,
            head_dim=64,
            layer_num=2,
            device=f"cuda:{torch.cuda.current_device()}",
            enable_memory_saver=False,
            post_capture_active=True,
            vmm_commit_chunk_bytes=2 * 1024 * 1024,
        )

    @staticmethod
    def _fields(line):
        """Parse 'k=v' tokens out of one emitted log line."""
        out = {}
        for tok in line.split():
            if "=" in tok:
                k, _, v = tok.partition("=")
                out[k] = v
        return out

    def test_dial_reports_prev_size_and_backed_rows_in_both_directions(self):
        pool = self._pool(8192)
        try:
            pool.runtime_set_backing_tokens(4096)

            # ---------- GROW ----------
            with self.assertLogs(LOGGER_NAME, level=logging.INFO) as grow:
                pool.runtime_set_backing_tokens(6144)
            grow_call = [m for m in grow.output if "BACKING-DIAL call" in m]
            grow_done = [m for m in grow.output if "BACKING-DIAL grow done" in m]
            self.assertTrue(grow_call, "grow emitted no BACKING-DIAL call line")
            self.assertTrue(grow_done, "grow emitted no BACKING-DIAL grow done line")

            f = self._fields(grow_call[0])
            self.assertEqual(f["branch"], "grow")
            self.assertEqual(int(f["request"]), 6144)
            # THE POINT: prev_size is the size BEFORE the call, not after.
            self.assertEqual(int(f["prev_size"]), 4096)
            self.assertEqual(int(f["delta"]), 2048)
            # uniform_backed_rows is reported and is a real arena reading:
            # it must be >= the pre-call size, never 0 on a backed pool.
            self.assertGreaterEqual(int(f["uniform_backed_rows"]), 4096)

            # ---------- SHRINK ----------
            with self.assertLogs(LOGGER_NAME, level=logging.INFO) as shrink:
                released = pool.runtime_set_backing_tokens(2048)
            shrink_call = [m for m in shrink.output if "BACKING-DIAL call" in m]
            shrink_done = [m for m in shrink.output if "BACKING-DIAL shrink done" in m]
            self.assertTrue(shrink_call, "shrink emitted no BACKING-DIAL call line")
            self.assertTrue(
                shrink_done, "shrink emitted no BACKING-DIAL shrink done line"
            )

            f = self._fields(shrink_call[0])
            self.assertEqual(f["branch"], "shrink")
            self.assertEqual(int(f["request"]), 2048)
            self.assertEqual(int(f["prev_size"]), 6144)
            self.assertEqual(int(f["delta"]), -4096)
            self.assertGreaterEqual(int(f["uniform_backed_rows"]), 6144)

            # The done line must carry the bytes actually handed back, so a
            # shrink that decided correctly but paid nothing is visible as
            # released_bytes=0 rather than as silence.
            fd = self._fields(shrink_done[0])
            self.assertEqual(int(fd["released_bytes"]), int(released))

            # ---------- NO-OP ----------
            with self.assertLogs(LOGGER_NAME, level=logging.INFO) as noop:
                self.assertEqual(pool.runtime_set_backing_tokens(2048), 0)
            noop_call = [m for m in noop.output if "BACKING-DIAL call" in m]
            self.assertTrue(noop_call, "no-op emitted no BACKING-DIAL call line")
            self.assertEqual(self._fields(noop_call[0])["branch"], "noop")
        finally:
            pool._clear_buffers()


if __name__ == "__main__":
    unittest.main()
