"""Wave-aware KV split for the eager FlashInfer prefill (27B line, 2026-09-24).

Desk half (no GPU): the chooser reproduces the measured geometry -- 512-token
chunks, 24 q / 4 KV heads, head_dim 256 -> 48 q tiles -> 192 CTAs -- splits on
the 5090 (170 SMs) and declines on a 3080 (68 SMs), stays inside the float
workspace flashinfer 0.6.14 reserves, and the default environment is empty.
The kernel half (split vs unsplit output) needs CUDA:
test_fi_prefill_wave_split_gpu_0924.py.
"""

import math
import os
import unittest
from unittest import mock

from sglang.srt.layers.attention import fi_prefill_wave_split as W
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

WS = 384 * 1024 * 1024  # SGLANG_FLASHINFER_WORKSPACE_SIZE default
GEOM = dict(num_qo_heads=24, num_kv_heads=4, head_dim=256, float_workspace_bytes=WS)


def _choose(prefix, num_sm, qo=512, **kw):
    args = dict(GEOM)
    args.update(kw)
    return W.choose_prefill_kv_split([qo], [prefix + qo], num_sm=num_sm, **args)


class TestFlashinferMirror(CustomTestCase):
    def test_cta_tile_q_mirrors_fa2_determine(self):
        # utils.cuh FA2DetermineCtaTileQ, flashinfer 0.6.14
        self.assertEqual(W.fa2_cta_tile_q(3072, 256), 64)  # our case
        self.assertEqual(W.fa2_cta_tile_q(3072, 128), 128)
        self.assertEqual(W.fa2_cta_tile_q(16, 256), 16)
        self.assertEqual(W.fa2_cta_tile_q(40, 128), 64)
        self.assertEqual(W.fa2_cta_tile_q(3072, 512), 32)
        self.assertEqual(W.fa2_cta_tile_q(8, 512), 16)


class TestChooser(CustomTestCase):
    def test_stock_geometry_is_192_ctas(self):
        c = _choose(36000, 170)
        self.assertEqual(c.q_tiles, 48)
        self.assertEqual(c.cta_tile_q, 64)
        self.assertEqual(c.ctas_stock, 192)
        self.assertEqual(c.rounds_stock, 2)

    def test_5090_splits_five_ways_at_depth(self):
        c = _choose(36000, 170)
        self.assertEqual(c.reason, "split")
        self.assertEqual(c.chunks, 5)
        self.assertEqual(c.fixed_split_size, math.ceil(36512 / 5))
        self.assertEqual(c.ctas_split, 960)
        self.assertEqual(c.rounds_split, 6)
        self.assertLess(c.predicted_ratio, 0.65)

    def test_3080_finds_no_gain(self):
        for prefix in (8192, 36000, 131072, 262144):
            c = _choose(prefix, 68)
            self.assertIsNone(c.fixed_split_size, prefix)
            self.assertEqual(c.reason, "stock:no_gain")

    def test_shallow_prefix_keeps_the_stock_plan(self):
        c = _choose(4096, 170, min_prefix_tokens=10240)
        self.assertIsNone(c.fixed_split_size)
        self.assertEqual(c.reason, "stock:shallow")
        c = _choose(10240, 170, min_prefix_tokens=10240)
        self.assertEqual(c.reason, "split")

    def test_workspace_bounds_the_chunk_count(self):
        # 0.6.14 reserves num_qo_heads * work_items * cta_tile_q * (hd+1) * 4 B
        per_chunk = 24 * 48 * 64 * 257 * 4
        for prefix in (36000, 131072, 262144):
            c = _choose(prefix, 170)
            work_items = 48 * c.chunks
            self.assertLessEqual(24 * work_items * 64 * 257 * 4, W.WORKSPACE_FILL_CAP * WS)
        # a workspace for two chunks only: n <= 2, which is no gain on 170 SMs
        c = _choose(36000, 170, float_workspace_bytes=int(2.2 * per_chunk / W.WORKSPACE_FILL_CAP))
        self.assertLessEqual(c.chunks, 2)

    def test_no_chunk_below_the_minimum(self):
        c = _choose(3000, 170)  # kv 3512: two chunks would be < 2048 tokens
        self.assertIsNone(c.fixed_split_size)

    def test_every_chunk_reaches_the_first_rows_causal_range(self):
        # qo 4096 over kv 12288: a chunk of 2458 tokens (n=5) would leave the
        # last chunk (2456 tokens) short of the 4096 new rows -> skipped
        c = W.choose_prefill_kv_split([4096], [12288], num_sm=170, **GEOM)
        if c.fixed_split_size is not None:
            n = math.ceil(12288 / c.fixed_split_size)
            last = 12288 - (n - 1) * c.fixed_split_size
            self.assertGreaterEqual(last, 4096)

    def test_bad_inputs_keep_the_stock_plan(self):
        self.assertIsNone(W.choose_prefill_kv_split([], [], num_sm=170, **GEOM).fixed_split_size)
        bad = dict(GEOM)
        bad["num_kv_heads"] = 5
        self.assertIsNone(W.choose_prefill_kv_split([512], [9000], num_sm=170, **bad).fixed_split_size)


class TestEnvironment(CustomTestCase):
    def test_default_is_off(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(W.wave_split_on())
            self.assertEqual(W.wave_split_from_prefix(), 0)
        self.assertEqual(W.launcher_env_p_deep_split(0), {})
        self.assertEqual(W.launcher_env_p_deep_split(None), {})

    def test_launcher_env_names_both_variables(self):
        env = W.launcher_env_p_deep_split(10240)
        self.assertEqual(env, {W.WAVE_SPLIT_ENV: "1", W.WAVE_SPLIT_FROM_PREFIX_ENV: "10240"})
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertTrue(W.wave_split_on())
            self.assertEqual(W.wave_split_from_prefix(), 10240)

    def test_graph_threshold_is_not_this_modules(self):
        # the graph side is Agent H's SGLANG_PREFILL_GRAPH_MAX_PREFIX
        env = W.launcher_env_p_deep_split(10240)
        self.assertFalse(any("GRAPH" in k for k in env))


if __name__ == "__main__":
    unittest.main()
