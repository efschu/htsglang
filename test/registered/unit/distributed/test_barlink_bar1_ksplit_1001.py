# SPDX-License-Identifier: Apache-2.0
"""barlink-BAR1 K_SPLIT (variant 2): the path choice, default OFF.

K_SPLIT runs the mesh/ring all_reduce as normal launches (multi-block payload
phases, single-thread flag waits) -- multi-block without the co-residency of
the cooperative grid, so it may run where the grid may not (under MPS,
tatcwj C1 wedge) and wins back the 1blk solo loss (tatcwj F3solo 40 MiB:
13.3 ms 1blk vs 9.9 ms grid).

DANGER DIRECTIONS guarded here:
* default OFF: no env -> the choice is exactly the old 0/1 (no new path in any boot);
* ON + MPS (grid threshold 1<<62): all_reduce >= SPLIT_FROM -> 2, below -> 1blk;
* never for all_to_all_single / mesh_pipe (their kernels have no split variant);
* where the grid applies (size >= grid threshold, no MPS) the grid still wins;
* the ext routes variant 2 to the split launcher for mesh/ring only, and its
  flag waits are single-thread kernels (no block waits for another block).
"""
from __future__ import annotations

import inspect
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.distributed.device_communicators import barlink_bar1 as B
from sglang.srt.distributed.device_communicators import barlink_bar1_ext as X
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MPS_NO_GRID = 1 << 62


def _t(on, frm=4 << 20):
    t = types.SimpleNamespace(split_on=on, split_from=frm, graph_grid=True, _graph_grid_reported=True)
    return lambda moved, thr, where: B.BarlinkBar1Transport._kernel(t, moved, thr, where)


class Bar1KSplit(CustomTestCase):
    def test_default_off(self):
        self.assertEqual(B.split_variant_default({}), (False, 4 << 20))
        k = _t(*B.split_variant_default({}))
        self.assertEqual(k(40 << 20, MPS_NO_GRID, "all_reduce"), 0)
        self.assertEqual(k(40 << 20, 4 << 20, "all_reduce"), 1)
        self.assertEqual(k(1 << 20, 4 << 20, "all_reduce"), 0)

    def test_env(self):
        self.assertEqual(B.split_variant_default({"SGLANG_BARLINK_BAR1_SPLIT": "1"}), (True, 4 << 20))
        self.assertEqual(B.split_variant_default({"SGLANG_BARLINK_BAR1_SPLIT": "1",
                                                  "SGLANG_BARLINK_BAR1_SPLIT_FROM": "1048576"}), (True, 1 << 20))
        self.assertEqual(B.split_variant_default({"SGLANG_BARLINK_BAR1_SPLIT": "0"})[0], False)

    def test_on_under_mps(self):
        k = _t(True)
        self.assertEqual(k(40 << 20, MPS_NO_GRID, "all_reduce"), 2)
        self.assertEqual(k(4 << 20, MPS_NO_GRID, "all_reduce"), 2)
        self.assertEqual(k((4 << 20) - 16, MPS_NO_GRID, "all_reduce"), 0)

    def test_never_for_other_collectives(self):
        k = _t(True)
        self.assertEqual(k(40 << 20, MPS_NO_GRID, "all_to_all_single"), 0)
        self.assertEqual(k(40 << 20, MPS_NO_GRID, "mesh_pipe"), 0)

    def test_grid_still_wins_where_it_applies(self):
        self.assertEqual(_t(True)(40 << 20, 4 << 20, "all_reduce"), 1)

    def test_ext_routes_and_waits_single_thread(self):
        src = X._CUDA_SRC
        self.assertIn("#define K_SPLIT 2", src)
        self.assertIn("if (kernel_variant == K_SPLIT && (algo == 0 || algo == 1))", src)
        body = src[src.index("__global__ void bar1_split_flagwait"):src.index("template<typename T>\n__global__ void bar1_split_reduce")]
        self.assertIn("if (blockIdx.x != 0 || threadIdx.x != 0) return;", body)
        self.assertNotIn("this_grid", src[src.index("// K_SPLIT -- multi-block"):src.index("// Launches the chosen kernel.")])

    def test_transport_wires_it(self):
        self.assertIn("self.split_on, self.split_from = split_variant_default()",
                      inspect.getsource(B.BarlinkBar1Transport.__init__))
