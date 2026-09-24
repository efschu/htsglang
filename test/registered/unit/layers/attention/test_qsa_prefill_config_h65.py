"""fnFL2 H65 (F2 of H58): SGLANG_WEG2_QSA_PREFILL_CONFIG -- the launch config
of the prefix-free QSA prefill kernel (_sparse_gqa_prefill: the first chunk of
every prompt and every short prefill), in the grammar of H58's
SGLANG_FORCE_QSA_ROWS_CONFIG. The device-name-keyed table gives the rig's
3080 and 5090 (16, 1, 2) above 512 rows: offline compiled for head_dim 256
and top-k width 2051, REG 255 (STACK 24-32 B) in 24.6 KB smem per 1-warp CTA
= 3 warps per SM on sm86 and sm120.
CPU: the choice per arch, the default (the table) and that the rows knob and
this one never cross, with a recorded launch. Metal (GPU window): the
configs agree with the table's output within bf16 summation-order noise.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa import sparse_attn as sa
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _RecordedKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


class PrefillConfigTest(unittest.TestCase):
    """F2 of H58: SGLANG_WEG2_QSA_PREFILL_CONFIG moves only the prefix-free
    prefill launch; the rows knob and this one never cross."""

    def setUp(self):
        sa._PREFILL_CONFIG_CACHE.clear()
        sa._ROWS_CONFIG_CACHE.clear()
        self.addCleanup(sa._PREFILL_CONFIG_CACHE.clear)
        self.addCleanup(sa._ROWS_CONFIG_CACHE.clear)

    def _launch(self, capability, prefill="", rows=""):
        rec = _RecordedKernel()
        total_q = 16384
        q = torch.zeros(total_q, 24, 8, dtype=torch.bfloat16)
        k = torch.zeros(total_q, 2, 8, dtype=torch.bfloat16)
        idx = torch.zeros(total_q, 4, dtype=torch.int32)
        cu = torch.tensor([0, total_q], dtype=torch.int32)
        with envs.SGLANG_WEG2_QSA_PREFILL_CONFIG.override(prefill), \
                envs.SGLANG_FORCE_QSA_ROWS_CONFIG.override(rows), \
                mock.patch.object(sa, "_sparse_gqa_prefill", rec), \
                mock.patch.object(sa.torch.cuda, "get_device_capability", lambda *a: capability), \
                mock.patch.object(sa.torch.cuda, "get_device_name", lambda *a: "NVIDIA GeForce RTX 5090"):
            sa.sparse_gqa_fwd_interface_triton(q, k, k.clone(), total_q, idx, cu, 0.3)
        self.assertEqual(len(rec.calls), 1)
        _, _, kw = rec.calls[0]
        return kw["BLOCK_N"], kw["num_warps"], kw["num_stages"]

    def test_default_is_the_table(self):
        self.assertEqual(envs.SGLANG_WEG2_QSA_PREFILL_CONFIG.get(), "")
        self.assertEqual(self._launch((8, 6)), (16, 1, 2))
        self.assertEqual(self._launch((12, 0)), (16, 1, 2))

    def test_override_per_arch_and_no_crosstalk(self):
        self.assertEqual(self._launch((8, 6), prefill="sm86:inf=32/8/2"), (32, 8, 2))
        self.assertEqual(self._launch((12, 0), prefill="sm86:inf=32/8/2"), (16, 1, 2))
        self.assertEqual(self._launch((12, 0), prefill="inf=64/8/2"), (64, 8, 2))
        # the rows knob does not move the prefix-free launch
        self.assertEqual(self._launch((8, 6), rows="inf=64/8/2"), (16, 1, 2))
        with self.assertRaises(ValueError):
            self._launch((8, 6), prefill="inf=24/8/2")


class RowsLaunchCrosstalkTest(unittest.TestCase):
    def setUp(self):
        sa._ROWS_CONFIG_CACHE.clear()
        sa._PREFILL_CONFIG_CACHE.clear()
        self.addCleanup(sa._ROWS_CONFIG_CACHE.clear)
        self.addCleanup(sa._PREFILL_CONFIG_CACHE.clear)

    def test_prefill_knob_does_not_move_the_rows_launch(self):
        rec = _RecordedKernel()
        q = torch.zeros(16384, 24, 8, dtype=torch.bfloat16)
        k = torch.zeros(32, 2, 8, dtype=torch.bfloat16)
        rows = torch.zeros(16384, 4, dtype=torch.int32)
        with envs.SGLANG_WEG2_QSA_PREFILL_CONFIG.override("inf=64/8/2"), \
                mock.patch.object(sa, "_sparse_attn_rows_fwd", rec), \
                mock.patch.object(sa.torch.cuda, "get_device_capability", lambda *a: (8, 6)), \
                mock.patch.object(sa.torch.cuda, "get_device_name", lambda *a: "NVIDIA GeForce RTX 3080"):
            sa.sparse_attn_rows_triton(q, k, k.clone(), rows, 0.3)
        kw = rec.calls[0][2]
        self.assertEqual((kw["BLOCK_N"], kw["num_warps"], kw["num_stages"]), (16, 1, 2))


@unittest.skipUnless(torch.cuda.is_available(), "metal: run in a GPU window, once per arch")
class MetalTest(unittest.TestCase):
    def test_prefill_configs_agree_on_the_metal(self):
        g = torch.Generator(device="cuda").manual_seed(652)
        tq, hq, hkv, d, topk = 3000, 24, 2, 256, 2051
        q = torch.randn(tq, hq, d, device="cuda", generator=g).bfloat16()
        k = torch.randn(tq, hkv, d, device="cuda", generator=g).bfloat16()
        v = torch.randn(tq, hkv, d, device="cuda", generator=g).bfloat16()
        r = torch.arange(tq, device="cuda").unsqueeze(1)
        j = torch.arange(topk, device="cuda").unsqueeze(0)
        idx = torch.where(j <= r, r - j, torch.full_like(r - j, -1)).to(torch.int32)
        cu = torch.tensor([0, tq], dtype=torch.int32, device="cuda")
        with envs.SGLANG_WEG2_QSA_PREFILL_CONFIG.override(""):
            base = sa.sparse_gqa_fwd_interface_triton(q, k, v, tq, idx, cu, 0.0625)
        for config in ("inf=32/8/2", "inf=64/8/2", "inf=32/4/2"):
            with envs.SGLANG_WEG2_QSA_PREFILL_CONFIG.override(config):
                out = sa.sparse_gqa_fwd_interface_triton(q, k, v, tq, idx, cu, 0.0625)
            self.assertTrue(torch.allclose(out.float(), base.float(), atol=2e-2, rtol=2e-2), config)


if __name__ == "__main__":
    unittest.main()
