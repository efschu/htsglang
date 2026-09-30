# SPDX-License-Identifier: Apache-2.0
"""BOOTZEIT (A) 0928: the O_DIRECT stream reads offset-coalesced RUNS instead of
one pread per tensor (FLLIPER_WEIGHT_LOADER_COALESCE_MIB).

Metal (NF rc12z30c): 135 032 tensors per load, name order jumps between the
dtype sections of a safetensors file on every tensor, PP0 read 0,56 GB/s against
3,56 GB/s the disk gives. The coalesced path must yield exactly what the
per-tensor path yields -- same names, same order, same bytes, should_load asked
once per name, post_load applied -- for an MoE-shaped and a 27B-shaped (dense,
PP/TP stage filter) checkpoint, while issuing a handful of reads instead of one
per tensor.
"""

import os
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from flliper.srt.model_loader import weight_utils as W

MIB = 1 << 20


def _moe_file(root, idx, n_experts=24, layers=(0, 1)):
    """MoE-shaped: per expert a small int shape tensor, a packed int32 weight and
    a bf16 scale -> three dtype sections, names interleaving them."""
    g = torch.Generator().manual_seed(11 + idx)
    d = {}
    for L in layers:
        for e in range(n_experts):
            p = f"model.layers.{L}.mlp.experts.{e}"
            d[f"{p}.down_proj.weight_shape"] = torch.tensor([64, 32], dtype=torch.int64)
            d[f"{p}.down_proj.weight_packed"] = torch.randint(-2**31, 2**31 - 1, (64, 128),
                                                              dtype=torch.int32, generator=g)
            d[f"{p}.down_proj.weight_scale"] = torch.randn(64, 4, generator=g).to(torch.bfloat16)
        d[f"model.layers.{L}.self_attn.q_proj.weight"] = torch.randn(96, 64, generator=g).to(torch.bfloat16)
    path = os.path.join(root, f"model-{idx:05d}-of-00002.safetensors")
    save_file(d, path)
    return path


def _dense_27b_file(root, idx, layers):
    """27B-shaped (dense INT8 + scales + bf16 norms), the PP3/TP3 case."""
    g = torch.Generator().manual_seed(100 + idx)
    d = {}
    for L in layers:
        p = f"model.layers.{L}"
        d[f"{p}.mlp.down_proj.weight"] = torch.randint(-128, 127, (256, 192), dtype=torch.int8, generator=g)
        d[f"{p}.mlp.down_proj.weight_scale"] = torch.randn(256, 1, generator=g)
        d[f"{p}.input_layernorm.weight"] = torch.randn(192, generator=g).to(torch.bfloat16)
        d[f"{p}.self_attn.o_proj.weight"] = torch.randint(-128, 127, (192, 192), dtype=torch.int8, generator=g)
    path = os.path.join(root, f"model-{idx:05d}-of-00003.safetensors")
    save_file(d, path)
    return path


def _collect(paths, should_load=None, post_load=None, coalesce_mib=None, gap_kib=None,
             budget=None, workers=3):
    env = {}
    if coalesce_mib is not None:
        env[W.COALESCE_ENV] = str(coalesce_mib)
    if gap_kib is not None:
        env[W.COALESCE_GAP_ENV] = str(gap_kib)
    with mock.patch.dict(os.environ, env, clear=False):
        if coalesce_mib is None:
            os.environ.pop(W.COALESCE_ENV, None)
        return list(W.pread_safetensors_stream(
            paths, should_load, post_load=post_load, direct_io=False, workers=workers,
            budget_bytes=budget, log=False))


class TestCoalescedStream(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def _same(self, a, b):
        self.assertEqual([n for n, _ in a], [n for n, _ in b])
        for (n, x), (_, y) in zip(a, b):
            self.assertEqual(x.dtype, y.dtype, n)
            self.assertEqual(tuple(x.shape), tuple(y.shape), n)
            if x.device.type != "meta":
                self.assertTrue(torch.equal(x, y), n)

    def test_default_off(self):
        self.assertEqual(W.coalesce_run_bytes(), 0) if W.COALESCE_ENV not in os.environ else None
        with mock.patch.dict(os.environ, {W.COALESCE_ENV: "0"}):
            self.assertEqual(W.coalesce_run_bytes(), 0)
        with mock.patch.dict(os.environ, {W.COALESCE_ENV: "32"}):
            self.assertEqual(W.coalesce_run_bytes(), 32 * MIB)

    def test_moe_identical_to_per_tensor(self):
        paths = [_moe_file(self.root, 0), _moe_file(self.root, 1, layers=(2, 3))]
        ref = _collect(paths)
        got = _collect(paths, coalesce_mib=8)
        self._same(ref, got)

    def test_moe_expert_window_filter_and_meta(self):
        """D-rank shape: a window of expert ids read, the rest vetoed, the
        attention weights handed over as meta (Form-A worker)."""
        paths = [_moe_file(self.root, 0)]
        asked = []

        def should_load(name):
            asked.append(name)
            if "self_attn" in name:
                return "meta"
            m = name.split(".experts.")
            return len(m) == 2 and 5 <= int(m[1].split(".")[0]) < 17

        ref = _collect(paths, should_load)
        n_ref = len(asked)
        asked.clear()
        got = _collect(paths, should_load, coalesce_mib=8, gap_kib=4)
        self._same(ref, got)
        self.assertEqual(len(asked), n_ref, "should_load asked once per name")
        self.assertEqual(len(set(asked)), len(asked))
        self.assertTrue(any(t.device.type == "meta" for _, t in got))

    def test_27b_dense_pp_stage_filter(self):
        """27B shape: dense INT8 shards, a PP stage keeps layers 2..5 only."""
        paths = [_dense_27b_file(self.root, 0, range(0, 4)),
                 _dense_27b_file(self.root, 1, range(4, 8))]

        def stage(name):
            return 2 <= int(name.split(".layers.")[1].split(".")[0]) <= 5

        ref = _collect(paths, stage)
        got = _collect(paths, stage, coalesce_mib=4)
        self._same(ref, got)
        self.assertEqual(len(got), 16)

    def test_post_load_applied_in_reader(self):
        paths = [_dense_27b_file(self.root, 0, range(0, 2))]

        def post(name, t):
            return t.to(torch.float32) * 2 if t.dtype == torch.bfloat16 else t

        ref = _collect(paths, post_load=post)
        got = _collect(paths, post_load=post, coalesce_mib=4)
        self._same(ref, got)

    def test_few_reads_not_one_per_tensor(self):
        paths = [_moe_file(self.root, 0)]
        items, runs = W.plan_coalesced_runs(paths, None, 32 * MIB, 256 << 10)
        self.assertEqual(len(runs), 1, "one contiguous file -> one run")
        items, runs = W.plan_coalesced_runs(paths, None, 16 << 10, 256 << 10)
        self.assertGreater(len(runs), 1)
        for r in runs:  # a run never outgrows the cap unless one tensor does
            if len(r.items) > 1:
                self.assertLessEqual(r.nbytes, 16 << 10)

    def test_tiny_budget_still_completes_in_order(self):
        """Window smaller than one run: every needed run is still submitted,
        the order holds, nothing deadlocks."""
        paths = [_moe_file(self.root, 0), _moe_file(self.root, 1, layers=(2,))]
        ref = _collect(paths)
        got = _collect(paths, coalesce_mib=1, budget=1, workers=2)
        self._same(ref, got)

    def test_reader_error_surfaces(self):
        paths = [_moe_file(self.root, 0)]
        with mock.patch.object(W, "_read_run", side_effect=IOError("disk gone")):
            with self.assertRaises(IOError):
                _collect(paths, coalesce_mib=8)


if __name__ == "__main__":
    unittest.main()
