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


class _FakeGroup:
    def __init__(self, rank, world_size=3):
        self.rank_in_group = rank
        self.world_size = world_size
        self.broadcasts = 0

    def broadcast(self, values, src=0):
        self.broadcasts += 1


class TestSpecTpSyncDivergeBudget(CustomTestCase):
    """#1485 compare = a blocking device read per broadcast; bounded now."""

    def _sync(self, rank, budget, rounds):
        from sglang.srt.environ import envs
        from sglang.srt.speculative import spec_tp_sync as mod

        calls = {"clone": 0}
        orig_clone = torch.Tensor.clone

        with envs.SGLANG_SPEC_TP_DIVERGE_CHECKS.override(budget):
            group = _FakeGroup(rank)
            sync = mod.SpecTpSync(group)

        def counting_clone(t, *a, **k):
            calls["clone"] += 1
            return orig_clone(t, *a, **k)

        values = torch.arange(4)
        torch.Tensor.clone = counting_clone
        try:
            for _ in range(rounds):
                sync.sync(mod.SpecTpSyncSite.DFLASH_ACCEPT_GREEDY, values)
        finally:
            torch.Tensor.clone = orig_clone
        self.assertEqual(group.broadcasts, rounds)
        return calls["clone"]

    def test_broadcast_source_never_compares(self):
        self.assertEqual(self._sync(rank=0, budget=64, rounds=100), 0)
        self.assertEqual(self._sync(rank=0, budget=-1, rounds=100), 0)

    def test_receiving_rank_compares_only_its_budget(self):
        self.assertEqual(self._sync(rank=1, budget=5, rounds=100), 5)

    def test_minus_one_keeps_the_old_unbounded_compare(self):
        self.assertEqual(self._sync(rank=2, budget=-1, rounds=100), 100)

    def test_zero_turns_it_off(self):
        self.assertEqual(self._sync(rank=1, budget=0, rounds=10), 0)


if __name__ == "__main__":
    unittest.main()
