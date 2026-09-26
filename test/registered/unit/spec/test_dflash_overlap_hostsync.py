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


def _compact_lens_exact(seq_lens, window, page):
    fake_self = SimpleNamespace(
        device=seq_lens.device, draft_window_size=window, page_size=page
    )
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    return DFlashWorkerV2._compute_compact_draft_seq_lens(fake_self, seq_lens)


def _compact_lens_host(seq_lens, window, page):
    fake_self = SimpleNamespace(draft_window_size=window, page_size=page)
    out = torch.empty(seq_lens.numel(), dtype=torch.int32)
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    DFlashWorkerV2._compute_compact_draft_seq_lens_host(fake_self, seq_lens, out)
    return out


class TestCompactSeqLensHostBound(CustomTestCase):
    def test_upper_bound_of_exact(self):
        g = torch.Generator().manual_seed(0)
        for window, page in [(4096, 64), (4096, 1), (128, 32), (64, 1), (2048, 1)]:
            seq = torch.randint(1, 3 * window, (512,), generator=g)
            exact = _compact_lens_exact(seq, window, page).to(torch.int64)
            bound = _compact_lens_host(seq, window, page).to(torch.int64)
            self.assertTrue(
                bool((bound >= exact).all()),
                f"host bound under-shoots exact at window={window} page={page}",
            )

    def test_exact_at_page_size_one(self):
        # The 27B D group: page_size 1, window 2048, exact host mirror
        # (overlap_utils.resolve_seq_lens_cpu) -> the bound IS the device value.
        g = torch.Generator().manual_seed(1)
        seq = torch.randint(1, 3 * 2048, (256,), generator=g)
        exact = _compact_lens_exact(seq, 2048, 1).to(torch.int64)
        bound = _compact_lens_host(seq, 2048, 1).to(torch.int64)
        torch.testing.assert_close(bound, exact, rtol=0, atol=0)

    def test_sawtooth_counterexample(self):
        # exact(4160) = 4096 < exact(4100) = 4100 at window=4096 page=64:
        # a host mirror of the exact math fed the reserved over-estimate
        # (4160 >= true 4100) would under-shoot; the envelope must not.
        window, page = 4096, 64
        true_len = torch.tensor([4100])
        reserved = torch.tensor([4160])
        exact_true = _compact_lens_exact(true_len, window, page).to(torch.int64)
        exact_reserved = _compact_lens_exact(reserved, window, page).to(torch.int64)
        self.assertLess(int(exact_reserved), int(exact_true))
        bound = _compact_lens_host(reserved, window, page).to(torch.int64)
        self.assertGreaterEqual(int(bound), int(exact_true))


_INTERPRETED_REBUILD = textwrap.dedent(
    """
    import os
    os.environ["TRITON_INTERPRET"] = "1"
    import torch
    from sglang.kernels.ops.speculative.cache_locs import (
        assign_req_to_token_pool_func,
        rebuild_compact_draft_req_to_token_func,
    )

    def compact_lens(seq, window, page):
        vis = torch.clamp(seq, max=window)
        if page <= 1:
            return vis.to(torch.int32)
        start = seq - vis
        aligned = start - torch.remainder(start, page)
        return (seq - aligned).to(torch.int32)

    def legacy(draft, target, req_idx, start, lens, verify_2d, bs, block):
        # DFlashWorkerV2._gather_req_to_token_segments + the two assigns.
        lens64 = lens.to(torch.int64)
        max_len = int(lens64.max().item())
        offs = torch.arange(max_len).unsqueeze(0)
        pos2d = start.to(torch.int64).unsqueeze(1) + offs
        mask = offs < lens64.unsqueeze(1)
        packed = target[req_idx.to(torch.int64)[:, None], pos2d.masked_fill(~mask, 0)][
            mask
        ].to(torch.int64)
        assign_req_to_token_pool_func(
            req_idx, draft, torch.zeros_like(lens), lens, packed, bs
        )
        assign_req_to_token_pool_func(
            req_idx, draft, lens, lens + block, verify_2d.reshape(-1), bs
        )

    n = 0
    for bs, window, page, block, seed in [
        (1, 64, 1, 8, 0),
        (6, 2048, 1, 8, 4),
        (16, 64, 32, 8, 1),
        (13, 128, 64, 8, 2),
        (7, 512, 64, 16, 3),
    ]:
        g = torch.Generator().manual_seed(seed)
        pool_rows, width = 4 * bs, 4 * window
        seq = torch.randint(1, width - block - 1, (bs,), generator=g).to(torch.int64)
        lens = compact_lens(seq, window, page)
        start = seq - lens.to(torch.int64)
        req_idx = torch.randperm(pool_rows, generator=g)[:bs]
        target = torch.randint(0, 2**30, (pool_rows, width), generator=g).to(
            torch.int32
        )
        verify_2d = torch.randint(0, 2**30, (bs, block), generator=g).to(torch.int64)
        draft_width = window + page + block + 8
        draft_a = torch.full((pool_rows, draft_width), -1, dtype=torch.int32)
        draft_b = draft_a.clone()
        legacy(draft_a, target, req_idx, start, lens, verify_2d, bs, block)
        rebuild_compact_draft_req_to_token_func(
            draft_req_to_token=draft_b,
            target_req_to_token=target,
            req_pool_indices=req_idx,
            suffix_start=start,
            draft_prefix_lens=lens,
            verify_out_cache_loc_2d=verify_2d,
            batch_size=bs,
            block_size=block,
        )
        assert torch.equal(draft_b, draft_a), (bs, window, page, block)
        for i in range(bs):
            total = int(lens[i]) + block
            assert bool((draft_b[req_idx[i], total:] == -1).all()), "wrote past block"
        n += 1
    print("REBUILD-BITEXACT-OK", n)
    """
)


class TestRebuildCompactDraftReqToTokenInterpreted(CustomTestCase):
    """The #31468 kernel against the legacy path, in the triton interpreter.

    TRITON_INTERPRET must be set before the kernels are decorated, hence the
    subprocess (this test process may already have imported cache_locs).
    """

    def test_bitexact_vs_legacy(self):
        env = dict(os.environ)
        env["TRITON_INTERPRET"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = ""
        proc = subprocess.run(
            [sys.executable, "-c", _INTERPRETED_REBUILD],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-4000:])
        self.assertIn("REBUILD-BITEXACT-OK 5", proc.stdout)


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


class TestDflashSyncTraceUnarmed(CustomTestCase):
    def test_passthrough_without_cuda_or_for_prefill(self):
        from sglang.srt.speculative.dflash_worker_v2 import _dflash_sync_traced

        seen = []

        def fn(batch, on_publish=None):
            seen.append(batch)
            return "result"

        wrapped = _dflash_sync_traced(fn, budget=4, tp_rank=0)
        extend = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_extend=lambda: True, is_idle=lambda: False
            ),
            is_extend_in_batch=False,
        )
        self.assertEqual(wrapped(extend), "result")
        self.assertEqual(wrapped(object()), "result")  # no forward_mode at all
        self.assertEqual(len(seen), 2)


if __name__ == "__main__":
    unittest.main()
