"""KV split inside the full prefill graph (27B line): the desk half, no GPU.

Pinned: the work-item arrays are flashinfer's own fixed-split layout (the
PrefillSplitQOKVIndptr step-3 port), the captured layout pads them to q_tiles
x N with a valid-item mask, the plan vector points run() at our region and
keeps the stock plan's live-row fields, the replay write lands byte for byte
in the wrapper's int workspace, the chooser uses up to N chunks on 170 SMs and
none on 68, and the default environment is empty. The kernel half (graph split
vs eager stock; our arrays vs the C++ planner's) is
test_fi_graph_split_gpu_0924.py.
"""

import math
import os
import struct
import unittest
from unittest import mock

import torch

from sglang.srt.layers.attention import fi_graph_split as G
from sglang.srt.layers.attention import fi_jit_cache_check as J
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

HQ, HKV, HD = 24, 4, 256
# the graph-mode stock plan of a 512-row bucket on the 5090 (padded 85)
STOCK = [85, 512, 1568, 64, 0, 352, 704, 1584, 1056, 1072, 0, 133693440, 3664, 1, 1]


def _layout(max_chunks=7, int_ws=8 << 20, float_ws=384 << 20):
    return G.layout_from_stock(
        STOCK, slots=1, max_chunks=max_chunks, num_qo_heads=HQ, num_kv_heads=HKV,
        head_dim_vo=HD, out_bytes=2, int_workspace_bytes=int_ws, float_workspace_bytes=float_ws)


class TestFlashinferLayoutPort(CustomTestCase):
    def test_one_request_five_chunks(self):
        a = G.split_arrays([512], [36512], gqa=6, cta_tile_q=64, kv_chunk=7303)
        self.assertEqual(len(a["request_indices"]), 48 * 5)
        self.assertEqual(set(a["request_indices"]), {0})
        self.assertEqual(a["qo_tile_indices"][:6], [0, 0, 0, 0, 0, 1])
        self.assertEqual(a["kv_tile_indices"][:6], [0, 1, 2, 3, 4, 0])
        self.assertEqual(a["merge_indptr"][:3], [0, 5, 10])
        self.assertEqual(a["merge_indptr"][-1], 512 * 5)
        self.assertEqual(a["o_indptr"], [0, 512 * 5])
        self.assertEqual(a["kv_chunk_size"], [7303])

    def test_per_request_chunk_counts_follow_the_kernel(self):
        # the kernel's own count: ceil(min(kv, kv + CTA_TILE_Q) / chunk)
        qo, kv, chunk = [100, 7], [9000, 2000], 2048
        a = G.split_arrays(qo, kv, gqa=6, cta_tile_q=64, kv_chunk=chunk)
        n = [math.ceil(min(k, k + 64) / chunk) for k in kv]
        tiles = [math.ceil(q * 6 / 64) for q in qo]
        self.assertEqual(len(a["request_indices"]), sum(t * c for t, c in zip(tiles, n)))
        self.assertEqual(a["o_indptr"], [0, qo[0] * n[0], qo[0] * n[0] + qo[1] * n[1]])
        self.assertEqual(a["merge_indptr"][-1], qo[0] * n[0] + qo[1] * n[1])
        self.assertEqual(a["merge_indptr"][qo[0]], qo[0] * n[0])

    def test_page_size_other_than_one_is_refused(self):
        with self.assertRaises(ValueError):
            G.split_arrays([512], [4096], gqa=6, cta_tile_q=64, kv_chunk=1024, page_size=16)


class TestLayout(CustomTestCase):
    def test_capture_layout(self):
        lay, why = _layout()
        self.assertEqual(why, "ok")
        self.assertEqual(lay.padded, 48 * 7)
        o = lay.offsets
        for k, v in o.items():
            self.assertEqual(v % 16, 0, k)
        self.assertGreaterEqual(o["request_indices"], G.INT_REGION_BASE)
        self.assertEqual(o["v"], 0)
        self.assertEqual(o["s"], 512 * 7 * HQ * HD * 2)
        self.assertLess(o["float_end"], 45 * 10**6)

    def test_refusals(self):
        self.assertIsNone(G.layout_from_stock(STOCK[:14], slots=1, max_chunks=7, num_qo_heads=HQ,
            num_kv_heads=HKV, head_dim_vo=HD, out_bytes=2, int_workspace_bytes=8 << 20,
            float_workspace_bytes=384 << 20)[0])
        eager = list(STOCK)
        eager[13] = 0
        self.assertIsNone(G.layout_from_stock(eager, slots=1, max_chunks=7, num_qo_heads=HQ,
            num_kv_heads=HKV, head_dim_vo=HD, out_bytes=2, int_workspace_bytes=8 << 20,
            float_workspace_bytes=384 << 20)[0])
        self.assertIsNone(_layout(float_ws=10 << 20)[0])
        self.assertIsNone(_layout(int_ws=1 << 20)[0])

    def test_plan_vector_points_at_our_region_and_keeps_the_live_rows(self):
        lay, _ = _layout()
        v = lay.plan_vector(STOCK)
        o = lay.offsets
        self.assertEqual(len(v), 15)
        self.assertEqual(v[0], lay.padded)
        self.assertEqual(v[1], STOCK[1])  # captured max rows (merge grid)
        self.assertEqual(v[2], STOCK[2])  # the stock plan writes the live rows there
        self.assertEqual(v[3], 64)
        self.assertEqual(v[4:13], [o["request_indices"], o["qo_tile_indices"], o["kv_tile_indices"],
                                   o["merge_indptr"], o["o_indptr"], o["kv_chunk_size"],
                                   o["v"], o["s"], o["block_valid_mask"]])
        self.assertEqual(v[13:], [1, 1])


class TestRegionBytes(CustomTestCase):
    def test_bytes_decode_back(self):
        lay, _ = _layout()
        a = G.split_arrays([512], [36512], gqa=6, cta_tile_q=64, kv_chunk=5216)
        buf = G.int_region_bytes(lay, a)
        o, base = lay.offsets, lay.int_base
        items = len(a["request_indices"])
        got = struct.unpack_from("<%di" % items, buf, o["kv_tile_indices"] - base)
        self.assertEqual(list(got), a["kv_tile_indices"])
        self.assertEqual(struct.unpack_from("<i", buf, o["kv_chunk_size"] - base)[0], 5216)
        self.assertEqual(struct.unpack_from("<2i", buf, o["o_indptr"] - base), (0, 512 * 7))
        mask = buf[o["block_valid_mask"] - base: o["block_valid_mask"] - base + lay.padded]
        self.assertEqual(sum(mask), items)
        self.assertEqual(items, lay.padded)  # 7 chunks fill the captured grid

    def test_more_items_than_the_grid_is_refused(self):
        lay, _ = _layout(max_chunks=2)
        a = G.split_arrays([512], [36512], gqa=6, cta_tile_q=64, kv_chunk=5216)
        with self.assertRaises(ValueError):
            G.int_region_bytes(lay, a)


class _FakeWrapper:
    def __init__(self):
        self._int_workspace_buffer = torch.zeros(8 << 20, dtype=torch.uint8)
        self._float_workspace_buffer = torch.zeros(1, dtype=torch.uint8)
        self._plan_info = list(STOCK)


class TestApply(CustomTestCase):
    def _run(self, prefix, num_sm=170):
        lay, _ = _layout()
        st = G.GraphSplitState(lay, STOCK)
        w = _FakeWrapper()
        chunk, choice = G.apply(w, st, [512], [prefix + 512], num_kv_heads=HKV, num_sm=num_sm)
        return lay, st, w, chunk, choice

    def test_deep_chunk_uses_all_seven(self):
        lay, st, w, chunk, choice = self._run(36000)
        self.assertEqual(st.last_chunks, 7)
        self.assertEqual(chunk, math.ceil(36512 / 7))
        self.assertEqual(list(w._plan_info), lay.plan_vector(STOCK))
        a = G.split_arrays([512], [36512], gqa=6, cta_tile_q=64, kv_chunk=chunk)
        want = G.int_region_bytes(lay, a)
        got = bytes(w._int_workspace_buffer[lay.int_base: lay.int_base + lay.int_bytes].tolist())
        self.assertEqual(got, bytes(want))
        self.assertLess(choice.predicted_ratio, 0.62)

    def test_first_chunk_is_one_chunk(self):
        _lay, st, _w, chunk, choice = self._run(0)
        self.assertEqual(st.last_chunks, 1)
        self.assertEqual(chunk, 512)
        self.assertIsNone(choice.fixed_split_size)

    def test_shallow_chunk_splits_inside_the_graph(self):
        _lay, st, _w, _chunk, choice = self._run(2048)
        self.assertGreater(st.last_chunks, 1)
        self.assertLess(choice.predicted_ratio, 0.9)

    def test_3080_keeps_one_chunk(self):
        for prefix in (2048, 36000, 131072):
            _lay, st, _w, _chunk, _choice = self._run(prefix, num_sm=68)
            self.assertEqual(st.last_chunks, 1, prefix)


class TestEnvironment(CustomTestCase):
    def test_default_off(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(G.graph_split_max_chunks(), 0)
            self.assertEqual(G.graph_split_min_chunk(), 512)
        self.assertEqual(G.launcher_env_p_graph_split(0), {})
        self.assertEqual(G.launcher_env_p_graph_split(1), {})
        self.assertEqual(G.launcher_env_p_graph_split(7), {G.GRAPH_SPLIT_ENV: "7"})


class TestJitCacheCheck(CustomTestCase):
    def test_uri_matches_the_cache_directory_names(self):
        # the names the metal's cache carries (ls ~/.cache/flashinfer/0.6.14/120f/cached_ops)
        self.assertEqual(
            J.prefill_uri("e4m3"),
            "batch_prefill_with_kv_cache_dtype_q_bf16_dtype_kv_e4m3_dtype_o_bf16_dtype_idx_i32_"
            "head_dim_qk_256_head_dim_vo_256_posenc_0_use_swa_False_use_logits_cap_False_f16qk_False",
        )

    def test_no_single_gpu_is_a_refusal_not_a_build(self):
        ok, lines = J.check_prefill_modules()
        if not torch.cuda.is_available():
            self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
