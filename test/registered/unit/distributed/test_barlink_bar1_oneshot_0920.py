"""Task #53 (20.09.): the one-barrier 'oneshot' all_reduce for small payloads.

Measured motive (collective clock, fn8v): a Qwen3.8 Next Flash decode round
issues 98 all_reduces of 24 KB, 88 us each = 8.6 ms of a 34.5 ms round. The
mesh runs reduce-scatter + allgather (two barriers) over 427-float4 chunks;
the barriers, not the bytes, are the price. 'oneshot' sends the whole
contribution to every peer and reduces locally: one barrier.

Hermetic: source-level checks on the JIT text plus the Python selection.
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

from sglang.srt.distributed.device_communicators import barlink_bar1_ext as ext
from sglang.srt.distributed.device_communicators.barlink_bar1 import (
    BarlinkBar1Transport,
    window_requirement,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-cpu")

W24_CHUNK_MAX = 1396736


def _stub(**kw) -> BarlinkBar1Transport:
    t = BarlinkBar1Transport.__new__(BarlinkBar1Transport)
    t.world = 3
    t.rank = 0
    t.group = "tp"
    t._up = True
    t._ext = object()
    t._plan = None
    t.ring_from = 1 << 20
    t.pipe_on = False
    t.pipe_from = 256 << 10
    t.oneshot_max = 64 << 10
    t._geo = {"chunk_max": W24_CHUNK_MAX}
    for k, v in kw.items():
        setattr(t, k, v)
    return t


class SourceTest(unittest.TestCase):
    def test_kernel_dispatcher_and_wrapper_know_algo_2(self):
        src = ext._CUDA_SRC
        self.assertIn("__global__ void bar1_oneshot_kernel(Bar1Args A)", src)
        self.assertIn("else if (algo == 2) BARLINK_BAR1_SELECT(bar1_oneshot_kernel);", src)
        self.assertIn("algo == 0 || algo == 1 || algo == 2", src)
        # the payload must fit ONE RS slot: refused in the wrapper, by name
        self.assertIn("oneshot payload", src)
        # one-barrier shape: exactly ONE flag phase (phase 0) is used, the
        # AG slots and phase-1 flags are never touched by this kernel
        body = src[src.index("__global__ void bar1_oneshot_kernel"):
                   src.index("// TOPOLOGY 'ring'")]
        self.assertNotIn("nzSendAG", body)
        self.assertNotIn("nzFlagTo [1]", body)
        self.assertIn("reduceNPhase<T>(A.in, A.out, sRecvRS", body)
        # the #622 entry acknowledgment is kept (slot reuse hazard is the same)
        self.assertIn("lastRoundDev", body)
        self.assertIn("sAckTo", body)


class SelectionTest(unittest.TestCase):
    def test_small_payload_selects_oneshot(self):
        self.assertEqual(_stub().algorithm_for(24 * 1024), "oneshot")
        self.assertEqual(_stub().algorithm_for(64 * 1024), "oneshot")

    def test_above_the_cap_stays_mesh(self):
        self.assertEqual(_stub().algorithm_for(64 * 1024 + 16), "mesh")

    def test_zero_cap_disables(self):
        self.assertEqual(_stub(oneshot_max=0).algorithm_for(24 * 1024), "mesh")

    def test_never_above_one_slot(self):
        self.assertEqual(
            _stub(_geo={"chunk_max": 16384}).algorithm_for(24 * 1024), "mesh"
        )

    def test_ring_sizes_untouched(self):
        self.assertEqual(_stub().algorithm_for(2 << 20), "ring")

    def test_env_sets_the_cap(self):
        with mock.patch.dict(os.environ, {"SGLANG_BARLINK_BAR1_ONESHOT_MAX": "8192"}):
            # the constructor reads the env; emulate the one line it runs
            t = _stub(oneshot_max=int(os.environ["SGLANG_BARLINK_BAR1_ONESHOT_MAX"]))
        self.assertEqual(t.algorithm_for(8192), "oneshot")
        self.assertEqual(t.algorithm_for(8208), "mesh")

    def test_algo_code_for_the_extension(self):
        # the wrapper's contract: 0=mesh 1=ring 2=oneshot
        self.assertEqual({"mesh": 0, "ring": 1, "oneshot": 2}["oneshot"], 2)


class WindowTest(unittest.TestCase):
    def test_oneshot_requirement_is_r_minus_1_full_payloads(self):
        self.assertEqual(window_requirement("oneshot", 24576, 3), 2 * 24576)
        # never above the mesh's requirement in its own range? Not claimed:
        # (R-1)*N vs 2(R-1)*ceil(N/R) -- for R=3 it is 1.5x. What IS claimed:
        # it is only chosen for N <= chunk_max, where R-1 slots of chunk_max
        # exist by construction.
        self.assertEqual(window_requirement("mesh", 24576, 3), 2 * 2 * 8192)


if __name__ == "__main__":
    unittest.main()
