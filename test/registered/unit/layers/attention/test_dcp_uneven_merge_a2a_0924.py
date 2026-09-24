# SPDX-License-Identifier: Apache-2.0
"""FlashInfer uneven-DCP LSE merge: SGLANG_DCP_LSE_MERGE=a2a for the 27B path.

``_dcp_uneven_merge`` (layers/attention/flashinfer_backend.py) routes the merge
of the paged-prefix / decode partials through ``cp_lse_ag_out_a2a_mha_uneven``
when ``lse_merge_mode() == 'a2a'`` -- one uneven all_to_all of the heads each
peer owns instead of an all_reduce of all 24 heads (the 27B verify's
``dcp.all_reduce``: 16 per round, ~81 us each, [8, 24, 256] fp32 = 196 KB,
above the barlink one-shot bound so it runs the two-barrier mesh).

Pinned on CPU with a fake 3-rank group that computes every collective from all
ranks' inputs:

* a2a == ar == the reference LSE merge, per rank, for the uneven head split
  [12, 6, 6] (and an even one), with and without ``return_lse``;
* the bf16 wire stays within bf16 rounding of the fp32 result;
* the dispatcher: default (unset) is the all_reduce, 'a2a' takes the
  all_to_all, and a weightless-KV boot stays on the all_reduce (its workers'
  merge sites do);
* every non-weightless merge site of the backend goes through the dispatcher.
"""

import inspect
import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

from sglang.srt.layers.dcp import comm  # noqa: E402


class _World:
    def __init__(self, counts, T=8, D=16, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.counts = list(counts)
        self.W = len(counts)
        H = sum(counts)
        self.o = [torch.randn(T, H, D, generator=g) for _ in range(self.W)]
        self.lse = [torch.randn(T, H, generator=g) * 3 for _ in range(self.W)]
        lses = torch.stack(self.lse, 0)
        self.g = torch.logsumexp(lses, dim=0)
        self.scaled = [o * torch.exp(l - self.g).unsqueeze(-1) for o, l in zip(self.o, self.lse)]
        self.total = sum(self.scaled)

    def bounds(self, r):
        s = sum(self.counts[:r])
        return s, s + self.counts[r]


class _FakeGroup:
    def __init__(self, world: _World, rank: int):
        self.w = world
        self.rank_in_group = rank
        self.world_size = world.W

    def all_gather(self, t, dim=0):
        return torch.cat([l.to(t.dtype) for l in self.w.lse], dim=0)

    def all_reduce(self, t):
        return self.w.total.to(t.dtype)

    def all_to_all_single_v(self, output, input, output_split_sizes=None, input_split_sizes=None):
        r = self.rank_in_group
        s0, s1 = self.w.bounds(r)
        blocks = [
            self.w.scaled[s][:, s0:s1, :].transpose(0, 1).to(output.dtype)
            for s in range(self.w.W)
        ]
        output.copy_(torch.cat(blocks, dim=0))
        return output


def _reset_merge_cache():
    comm._LSE_MERGE["mode"] = None
    comm._LSE_MERGE["dtype"] = None


class TestA2aEqualsAllReduce(CustomTestCase):
    def setUp(self):
        _reset_merge_cache()

    def tearDown(self):
        _reset_merge_cache()

    def _run(self, counts):
        w = _World(counts)
        for r in range(w.W):
            grp = _FakeGroup(w, r)
            s0, s1 = w.bounds(r)
            want = w.total[:, s0:s1, :]
            ar = comm.cp_lse_ag_out_ar_mha_uneven(w.o[r], w.lse[r], grp, counts)
            a2a = comm.cp_lse_ag_out_a2a_mha_uneven(w.o[r], w.lse[r], grp, counts)
            torch.testing.assert_close(ar, want, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(a2a, want, rtol=1e-6, atol=1e-6)
            self.assertEqual(a2a.dtype, torch.float32)
            self.assertEqual(tuple(a2a.shape), tuple(ar.shape))
            ar_o, ar_l = comm.cp_lse_ag_out_ar_mha_uneven(
                w.o[r], w.lse[r], grp, counts, return_lse=True)
            a2_o, a2_l = comm.cp_lse_ag_out_a2a_mha_uneven(
                w.o[r], w.lse[r], grp, counts, return_lse=True)
            torch.testing.assert_close(a2_o, ar_o, rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(a2_l, ar_l, rtol=0, atol=0)
            torch.testing.assert_close(a2_l, w.g[:, s0:s1], rtol=0, atol=0)

    def test_uneven_27b_heads(self):
        self._run([12, 6, 6])

    def test_even_heads(self):
        self._run([8, 8, 8])

    def test_bf16_wire_is_bf16_rounding(self):
        w = _World([12, 6, 6], seed=5)
        with mock.patch.dict("os.environ", {"SGLANG_DCP_LSE_MERGE_DTYPE": "bf16"}):
            _reset_merge_cache()
            for r in range(3):
                grp = _FakeGroup(w, r)
                s0, s1 = w.bounds(r)
                got = comm.cp_lse_ag_out_a2a_mha_uneven(w.o[r], w.lse[r], grp, [12, 6, 6])
                want = w.total[:, s0:s1, :]
                torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)


class TestDispatcher(CustomTestCase):
    def setUp(self):
        _reset_merge_cache()

    def tearDown(self):
        _reset_merge_cache()

    def _call(self, mode, weightless=False):
        from sglang.srt.layers.attention import flashinfer_backend as fb

        with mock.patch.object(comm, "lse_merge_mode", return_value=mode), \
             mock.patch.object(comm, "weightless_kv_active", return_value=weightless), \
             mock.patch.object(comm, "cp_lse_ag_out_a2a_mha_uneven", return_value="A2A") as a2a, \
             mock.patch.object(fb, "cp_lse_ag_out_ar_mha_uneven", return_value="AR") as ar:
            out = fb._dcp_uneven_merge("o", "lse", "grp", [12, 6, 6], return_lse=True)
        return out, a2a, ar

    def test_default_is_the_all_reduce(self):
        out, a2a, ar = self._call("ar")
        self.assertEqual(out, "AR")
        ar.assert_called_once_with("o", "lse", "grp", [12, 6, 6], return_lse=True)
        a2a.assert_not_called()

    def test_a2a_mode(self):
        out, a2a, ar = self._call("a2a")
        self.assertEqual(out, "A2A")
        a2a.assert_called_once_with("o", "lse", "grp", [12, 6, 6], return_lse=True)
        ar.assert_not_called()

    def test_weightless_stays_on_the_all_reduce(self):
        out, a2a, ar = self._call("a2a", weightless=True)
        self.assertEqual(out, "AR")
        a2a.assert_not_called()

    def test_env_default_mode_is_ar(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("SGLANG_DCP_LSE_MERGE", None)
            _reset_merge_cache()
            self.assertEqual(comm.lse_merge_mode(), "ar")

    def test_every_head_merge_site_uses_the_dispatcher(self):
        from sglang.srt.layers.attention import flashinfer_backend as fb

        src = inspect.getsource(fb.FlashInferAttnBackend)
        # the decode head path and both extend prefix merges
        self.assertEqual(src.count("_dcp_uneven_merge("), 3)
        # only the weightless WORKER sites still call the all_reduce directly
        direct = [l for l in src.splitlines() if "cp_lse_ag_out_ar_mha_uneven(" in l]
        self.assertEqual(len(direct), 2, direct)
        for name in ("forward_decode_weightless_worker", "forward_extend_weightless_worker"):
            body = inspect.getsource(getattr(fb.FlashInferAttnBackend, name))
            self.assertIn("cp_lse_ag_out_ar_mha_uneven(", body)


if __name__ == "__main__":
    unittest.main()
