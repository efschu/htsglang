# SPDX-License-Identifier: Apache-2.0
"""BOOTZEIT 3 Stufe 2b (29.09.): the non-expert tensors read ONCE per boot.

H2 (store adopt) lets group D take P's EXPERT rows from the host store; the
dense/embed/lm_head/draft tensors were still read from disk by both groups.
With SGLANG_WEIGHT_LOADER_SHARED_CACHE=keep (P) / drop (D) they travel through
the page cache: P leaves them there (budgeted), D reads them from there and
drops each range right after. Off = the plan and the reads are unchanged.
"""

import os
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from sglang.srt.model_loader import weight_utils as W

MIB = 1 << 20


def _mixed_file(root):
    g = torch.Generator().manual_seed(7)
    d = {}
    for L in (0, 1):
        for e in range(8):
            d[f"model.layers.{L}.mlp.experts.{e}.down_proj.weight_packed"] = torch.randint(
                -2**31, 2**31 - 1, (64, 128), dtype=torch.int32, generator=g)
        d[f"model.layers.{L}.self_attn.q_proj.weight"] = torch.randn(256, 64, generator=g).to(torch.bfloat16)
        d[f"model.layers.{L}.input_layernorm.weight"] = torch.randn(64, generator=g).to(torch.bfloat16)
    d["model.embed_tokens.weight"] = torch.randn(512, 64, generator=g).to(torch.bfloat16)
    path = os.path.join(root, "model-00001-of-00001.safetensors")
    save_file(d, path)
    return path


def _collect(paths, mode, extra_env=None, log=False):
    env = {W.COALESCE_ENV: "8"}
    env["SGLANG_WEIGHT_LOADER_SHARED_CACHE"] = mode
    env.update(extra_env or {})
    with mock.patch.dict(os.environ, env, clear=False):
        return list(W.pread_safetensors_stream(
            paths, None, direct_io=False, workers=2, log=log))


class TestSharedCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = _mixed_file(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _same(self, a, b):
        self.assertEqual([n for n, _ in a], [n for n, _ in b])
        for (n, x), (_, y) in zip(a, b):
            self.assertTrue(torch.equal(x, y), n)

    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {"SGLANG_WEIGHT_LOADER_SHARED_CACHE": ""}):
            self.assertEqual(W.shared_cache_mode(), "")
        with mock.patch.dict(os.environ, {"SGLANG_WEIGHT_LOADER_SHARED_CACHE": "bogus"}):
            self.assertEqual(W.shared_cache_mode(), "")

    def test_classifier_is_everything_but_experts(self):
        self.assertFalse(W.is_shared_cache_tensor("model.layers.3.mlp.experts.7.w1.weight_packed"))
        self.assertTrue(W.is_shared_cache_tensor("model.layers.3.self_attn.q_proj.weight"))
        self.assertTrue(W.is_shared_cache_tensor("model.embed_tokens.weight"))
        self.assertTrue(W.is_shared_cache_tensor("lm_head.weight"))

    def test_runs_never_mix_the_classes_when_on(self):
        _items, runs = W.plan_coalesced_runs(
            [self.path], None, 64 * MIB, 1 << 20, shared_class=W.is_shared_cache_tensor)
        for r in runs:
            classes = {W.is_shared_cache_tensor(_items[i][0]) for i in r.items}
            self.assertEqual(len(classes), 1)
            self.assertEqual(r.shared, classes.pop())
        # off: the plan is exactly the old one (one run for one contiguous file)
        _i2, runs_off = W.plan_coalesced_runs([self.path], None, 64 * MIB, 1 << 20)
        self.assertEqual(len(runs_off), 1)
        self.assertFalse(runs_off[0].shared)

    def test_keep_and_drop_yield_identical_bytes(self):
        ref = _collect([self.path], "")
        self._same(ref, _collect([self.path], "keep"))
        self._same(ref, _collect([self.path], "drop"))

    def test_drop_finds_what_keep_left_and_says_so(self):
        _collect([self.path], "keep")  # P: the first reader, buffered
        with self.assertLogs(W.logger, "INFO") as cm:
            _collect([self.path], "drop", log=True)
        line = next(r for r in cm.output if "SHARED-CACHE" in r)
        self.assertIn("mode=drop", line)
        hit = float(line.split("hit_mib=")[1].split()[0])
        dropped = float(line.split("dropped_mib=")[1].split()[0])
        self.assertGreaterEqual(dropped, 0.0)
        self.assertGreaterEqual(hit, 0.0)

    def test_keep_budget_is_honoured(self):
        import threading

        def counters():
            return {"lock": threading.Lock(), "shared_kept_bytes": 0,
                    "shared_over_budget_bytes": 0, "shared_dropped_bytes": 0}

        fd = os.open(self.path, os.O_RDONLY)
        try:
            with mock.patch.dict(os.environ, {"SGLANG_WEIGHT_LOADER_SHARED_CACHE_MAX_MIB": "1"}):
                c = counters()
                W._shared_cache_after_read(fd, 0, MIB // 2, "keep", c)
                W._shared_cache_after_read(fd, 0, MIB, "keep", c)  # would pass 1 MiB
                self.assertEqual(c["shared_kept_bytes"], MIB // 2)
                self.assertEqual(c["shared_over_budget_bytes"], MIB)
            c = counters()
            W._shared_cache_after_read(fd, 0, 4096, "drop", c)
            self.assertEqual(c["shared_dropped_bytes"], 4096)
            self.assertEqual(c["shared_kept_bytes"], 0)
        finally:
            os.close(fd)

    def test_expert_runs_keep_their_read_path(self):
        seen = []
        real = W._read_run

        def spy(run, direct_io, post_load, counters):
            seen.append((run.shared, counters.get("share_mode")))
            return real(run, direct_io, post_load, counters)

        with mock.patch.object(W, "_read_run", side_effect=spy):
            _collect([self.path], "keep")
        self.assertTrue(any(s for s, _ in seen))
        self.assertTrue(any(not s for s, _ in seen))

    def test_resident_bytes_reports_a_cached_file(self):
        fd = os.open(self.path, os.O_RDONLY)
        try:
            os.pread(fd, os.path.getsize(self.path), 0)  # into the page cache
            got = W._resident_bytes(fd, 0, os.path.getsize(self.path))
        finally:
            os.close(fd)
        self.assertGreaterEqual(got, 0)
        self.assertLessEqual(got, os.path.getsize(self.path))


if __name__ == "__main__":
    unittest.main()
