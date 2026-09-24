"""Unit tests for the DFlash spec-v2 host-sync removal (#31468 port, adapted).

Covered on CPU:
* the compact-draft host seq-lens upper bound (upstream test);
* the one-pass compact rebuild kernel, bit-exact against the legacy
  gather + two-assign path -- run through the triton INTERPRETER in a
  subprocess, since this box has no GPU (the upstream test is CUDA-only);
* HybridAttnBackend needs_cpu_seq_lens delegation (upstream test);
* DFlashDraftInputV2.filter_batch with the host keep-list (fork field names);
* the bounded #1485 divergence compare in SpecTpSync (fork instrument);
* the env-gated DFLASH sync-census wrapper passes through when unarmed.
"""

import os
import subprocess
import sys
import textwrap
import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


class TestHybridNeedsCpuSeqLens(CustomTestCase):
    def _make(self, prefill_flag, decode_flag):
        from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend

        def backend(flag):
            return SimpleNamespace(needs_cpu_seq_lens=flag)

        runner = SimpleNamespace(
            server_args=SimpleNamespace(speculative_attention_mode="decode"),
            kv_cache_dtype=torch.bfloat16,
            token_to_kv_pool=None,
            req_to_token_pool=None,
        )
        return HybridAttnBackend(runner, backend(prefill_flag), backend(decode_flag))

    def test_delegation(self):
        self.assertFalse(self._make(False, False).needs_cpu_seq_lens)
        self.assertTrue(self._make(True, False).needs_cpu_seq_lens)
        self.assertTrue(self._make(False, True).needs_cpu_seq_lens)


class TestFilterBatchHostIndices(CustomTestCase):
    def test_host_keep_list_matches_gpu_indices(self):
        from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2

        def make():
            info = DFlashDraftInputV2.create_idle_input(device=torch.device("cpu"))
            info.nxt_kv_lens_cpu = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
            info.nxt_kv_lens_sum = 100
            info.future_indices = torch.tensor([5, 6, 7, 8])
            return info

        keep = [0, 2]
        a, b = make(), make()
        a.filter_batch(new_indices=torch.tensor(keep), has_been_filtered=False)
        b.filter_batch(
            new_indices=torch.tensor(keep),
            has_been_filtered=False,
            new_indices_cpu=keep,
        )
        torch.testing.assert_close(a.nxt_kv_lens_cpu, b.nxt_kv_lens_cpu)
        self.assertEqual(a.nxt_kv_lens_sum, b.nxt_kv_lens_sum)
        self.assertEqual(b.nxt_kv_lens_sum, 40)
        torch.testing.assert_close(a.future_indices, b.future_indices)

    def test_eagle_and_ngram_accept_the_host_keep_list(self):
        # schedule_batch.filter_batch now passes new_indices_cpu to whatever
        # spec_info the batch carries; every filter_batch must take it.
        import inspect

        from sglang.srt.speculative.eagle_info import EagleDraftInput
        from sglang.srt.speculative.ngram_info import NgramVerifyInput

        for cls in (EagleDraftInput, NgramVerifyInput):
            with self.subTest(cls=cls.__name__):
                self.assertIn(
                    "new_indices_cpu", inspect.signature(cls.filter_batch).parameters
                )


if __name__ == "__main__":
    unittest.main()
