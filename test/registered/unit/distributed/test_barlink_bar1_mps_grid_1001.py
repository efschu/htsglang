# SPDX-License-Identifier: Apache-2.0
"""barlink-BAR1 under MPS: no cooperative full-card grid unless asked for.

ROOT (repro scjhru S1 + metal ndktv4, 30.09.): from GRID_THRESHOLD (4 MiB) a
spin collective is a cudaLaunchCooperativeKernel sized perSM x all SMs, i.e.
it can only start when the whole card is free of resident blocks. Under MPS
P and D share one server context with no time-slice preemption: a spin block
of the other group (waiting on its partner on another card) keeps the grid
from starting, and the two groups' spins form a cycle across the cards that
only the capCycles device deadline breaks (S1: D rounds 20.6/61.7/62.0 s =
multiples of the 60e9-cycle cap; P rc=124). Without MPS the time slice
preempts whole contexts, so the same load is slow but never cyclic.

DANGER DIRECTIONS guarded here:
* an MPS client (CUDA_MPS_PIPE_DIRECTORY set) gets NO cooperative variant by
  default -- every spin kernel is one block, no co-residency demand;
* an explicit SGLANG_BARLINK_BAR1_GRID_THRESHOLD still wins (measurements);
* a non-MPS process keeps the 4 MiB default (default path unchanged);
* the transport consults this one function.
"""
from __future__ import annotations

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.distributed.device_communicators import barlink_bar1 as B
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class Bar1MpsGrid(CustomTestCase):
    def test_default_without_mps_unchanged(self):
        self.assertEqual(B.grid_threshold_default({}), 4 << 20)
        self.assertEqual(B.grid_threshold_default({"CUDA_MPS_PIPE_DIRECTORY": ""}), 4 << 20)

    def test_mps_client_gets_no_cooperative_grid(self):
        thr = B.grid_threshold_default({"CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps/pipe"})
        self.assertGreaterEqual(thr, 1 << 62)

    def test_explicit_threshold_wins(self):
        env = {"CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps/pipe", "SGLANG_BARLINK_BAR1_GRID_THRESHOLD": "8388608"}
        self.assertEqual(B.grid_threshold_default(env), 8 << 20)
        self.assertEqual(B.grid_threshold_default({"SGLANG_BARLINK_BAR1_GRID_THRESHOLD": "1024"}), 1024)

    def test_transport_uses_it(self):
        src = inspect.getsource(B.BarlinkBar1Transport.__init__)
        self.assertIn("self.grid_from = grid_threshold_default()", src)
