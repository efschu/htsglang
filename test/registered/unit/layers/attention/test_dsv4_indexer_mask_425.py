"""#425 -- the two pure-torch DSV4 indexer references must mask the same way.

Upstream sgl-project/sglang#33247 reports that `fp8_paged_mqa_logits_torch`
leaves positions beyond `seq_len` at 0.0 while `fp8_paged_mqa_logits_torch_sm120`
fills them with -inf, and concludes that padding therefore displaces real tokens
in the top-k that consumes these logits.

The divergence is real and is pinned here. The conclusion is not reachable on
this fork's call path, and that is pinned here too, because "harmless today"
decays quietly:

* the dispatch cannot pick the unmasked one. Since #417 Cut 3
  `select_paged_mqa_logits_fn` returns the `_sm120` variant for every card that
  takes the torch backend, over the whole capability x environment matrix;
* no consumer would be fooled if it did. All three top-k paths bound
  themselves by `seq_len` independently -- the two CUDA kernels scan only
  `[0, seq_len)`, the torch fallback re-masks with -inf first.

So the tail of the logits is undefined by contract, not -inf by contract (the
tilelang and aiter producers leave uninitialized memory there). What #425
changes is narrower: the unmasked function is the ORACLE the SM120 kernel test
compares against, and an oracle that disagrees with the implementation it
validates is the #418 failure class. Aligning it costs nothing -- the function
is unreachable from serving -- and buys a full-width equality check.

GPU-free: everything here is CPU float32.
"""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4 import indexer_arch
from sglang.srt.layers.attention.dsv4.indexer import (
    FP8_DTYPE,
    fp8_paged_mqa_logits_torch,
    fp8_paged_mqa_logits_torch_sm120,
    select_paged_mqa_logits_fn,
    topk_transform_512_pytorch_vectorized,
)
from sglang.srt.layers.attention.dsv4.indexer_arch import deepgemm_indexer_supported
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

PAGE_SIZE = 64
HEAD_DIM = 128
PAGE_BYTES = PAGE_SIZE * HEAD_DIM + PAGE_SIZE * 4

# The exact setup from the upstream report.
ISSUE_SEQ_LENS = [64, 128, 200, 256]
ISSUE_MAX_SEQ_LEN = 256


def _build_inputs(
    seq_lens: list[int],
    *,
    num_heads: int = 8,
    seed: int = 0,
    signed_weights: bool = False,
    extra_pages: int = 0,
):
    """Paged FP8 index cache in the production layout, on CPU.

    `extra_pages` widens the page table beyond `ceil(max_seq_len / 64)`, which
    is what a CUDA-graph capture does: the two implementations walk different
    numbers of pages there and must still agree.
    """
    batch_size = len(seq_lens)
    max_seq_len = max(seq_lens)
    max_pages = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE + extra_pages
    num_pages_total = batch_size * max_pages + 1

    g = torch.Generator().manual_seed(seed)
    raw = torch.empty(num_pages_total, PAGE_BYTES, dtype=torch.uint8)
    # Restrict the value bytes to [0, 0x70) so no FP8 code decodes to NaN/Inf.
    raw[:, : PAGE_SIZE * HEAD_DIM] = torch.randint(
        0,
        0x70,
        (num_pages_total, PAGE_SIZE * HEAD_DIM),
        generator=g,
        dtype=torch.uint8,
    )
    scales = torch.rand((num_pages_total, PAGE_SIZE), generator=g) * 0.5 + 0.05
    raw[:, PAGE_SIZE * HEAD_DIM :] = scales.contiguous().view(torch.uint8)
    kvcache = raw.view(num_pages_total, PAGE_SIZE, 1, HEAD_DIM + 4).view(
        dtype=FP8_DTYPE
    )

    q_fp8 = (
        torch.randn((batch_size, 1, num_heads, HEAD_DIM), generator=g)
        .clamp_(-2.0, 2.0)
        .to(FP8_DTYPE)
    )
    weight = torch.rand((batch_size, num_heads), generator=g)
    if signed_weights:
        # `compute_weights` is an unconstrained linear projection, so the
        # weighted sum of the (non-negative) relu'd scores can go negative.
        # That is the case the upstream report's mechanism needs.
        weight = weight - 0.5
    else:
        weight = weight * 0.5

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
    page_table = torch.zeros((batch_size, max_pages), dtype=torch.int32)
    for i in range(batch_size):
        page_table[i] = torch.arange(
            1 + i * max_pages, 1 + (i + 1) * max_pages, dtype=torch.int32
        )
    return q_fp8, kvcache, weight, seq_lens_t, page_table, max_seq_len


def _finite_per_row(logits: torch.Tensor) -> list[int]:
    return torch.isfinite(logits).sum(-1).tolist()


class TestBothReferencesMaskByLength(CustomTestCase):
    """The falsifier. Fails on the tree before #425, passes after."""

    def test_untrimmed_reference_masks_beyond_seq_len(self):
        q, kv, w, sl, pt, msl = _build_inputs(ISSUE_SEQ_LENS)
        out = fp8_paged_mqa_logits_torch(q, kv, w, sl, pt, None, msl, False)
        self.assertEqual(out.shape, (len(ISSUE_SEQ_LENS), ISSUE_MAX_SEQ_LEN))
        # Before the fix this is [256, 256, 256, 256].
        self.assertEqual(_finite_per_row(out), ISSUE_SEQ_LENS)

    def test_trimmed_reference_masks_beyond_seq_len(self):
        q, kv, w, sl, pt, msl = _build_inputs(ISSUE_SEQ_LENS)
        out = fp8_paged_mqa_logits_torch_sm120(q, kv, w, sl, pt, None, msl, False)
        self.assertEqual(_finite_per_row(out), ISSUE_SEQ_LENS)

    def test_the_two_are_now_equal_everywhere_not_only_on_valid_positions(self):
        """The kernel test's oracle comparison could only ever check `[:seq_len]`
        while the tail values differed. Full-width equality is the property that
        makes it an oracle."""
        for extra_pages in (0, 1, 3):
            for signed_weights in (False, True):
                with self.subTest(extra_pages=extra_pages, signed=signed_weights):
                    q, kv, w, sl, pt, msl = _build_inputs(
                        ISSUE_SEQ_LENS,
                        extra_pages=extra_pages,
                        signed_weights=signed_weights,
                    )
                    a = fp8_paged_mqa_logits_torch(q, kv, w, sl, pt, None, msl, False)
                    b = fp8_paged_mqa_logits_torch_sm120(
                        q, kv, w, sl, pt, None, msl, False
                    )
                    self.assertTrue(
                        torch.equal(a, b),
                        f"max diff {(a - b)[torch.isfinite(a)].abs().max()}",
                    )

    def test_sm120_output_is_unchanged_by_425(self):
        """#425 touches only the other function. Pin the trimmed variant against
        values recorded before the change, so a later 'tidy up the pair' cannot
        move the one that actually serves."""
        q, kv, w, sl, pt, msl = _build_inputs(ISSUE_SEQ_LENS)
        out = fp8_paged_mqa_logits_torch_sm120(q, kv, w, sl, pt, None, msl, False)
        golden = torch.tensor(
            [
                [
                    146.8665771484375,
                    129.6529541015625,
                    71.98947143554688,
                    93.57316589355469,
                ],
                [
                    12.615509033203125,
                    83.8802719116211,
                    66.0515365600586,
                    29.279504776000977,
                ],
                [
                    48.928619384765625,
                    170.20152282714844,
                    147.86595153808594,
                    75.19402313232422,
                ],
                [
                    58.9664192199707,
                    87.86997985839844,
                    10.431636810302734,
                    143.80868530273438,
                ],
            ]
        )
        torch.testing.assert_close(out[:, :4], golden, atol=0.0, rtol=0.0)
        self.assertEqual(_finite_per_row(out), ISSUE_SEQ_LENS)

    def test_padding_would_have_outranked_real_tokens(self):
        """Why the divergence is worth removing rather than documenting: with an
        unmasked tail the 0.0 padding is not below the real tokens. relu makes
        the per-head scores non-negative, so a real token can score exactly 0.0
        and tie with padding; with a sign-carrying weight projection it can go
        strictly negative and lose outright."""
        q, kv, w, sl, pt, msl = _build_inputs(ISSUE_SEQ_LENS, signed_weights=True)
        logits = fp8_paged_mqa_logits_torch(q, kv, w, sl, pt, None, msl, False)
        real = logits[0, : ISSUE_SEQ_LENS[0]]
        self.assertTrue(
            bool((real <= 0.0).any()),
            "expected at least one real token at or below the 0.0 pad value",
        )


class TestTheDispatchCannotPickTheUnmaskedOne(CustomTestCase):
    """Rescue 1: no card reaches the divergent implementation."""

    _CAPABILITIES = [(8, 0), (8, 6), (8, 9), (12, 0), (12, 1)]

    def setUp(self):
        super().setUp()
        deepgemm_indexer_supported.cache_clear()
        self.addCleanup(deepgemm_indexer_supported.cache_clear)

    def _select(self, capability, torch_impl=False):
        deepgemm_indexer_supported.cache_clear()
        with mock.patch.multiple(
            indexer_arch,
            is_cuda=lambda: True,
            get_device_capability_no_init=lambda device_id: capability,
        ), envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(torch_impl):
            return select_paged_mqa_logits_fn(
                device=torch.device("cuda", 0), use_fp4_indexer=False
            )

    def test_no_capability_or_env_combination_returns_it(self):
        for capability in self._CAPABILITIES:
            for torch_impl in (False, True):
                with self.subTest(sm=capability, torch_impl=torch_impl):
                    self.assertIsNot(
                        self._select(capability, torch_impl=torch_impl),
                        fp8_paged_mqa_logits_torch,
                    )

    def test_the_torch_backend_gets_the_masking_variant(self):
        for capability in self._CAPABILITIES:
            with self.subTest(sm=capability):
                self.assertIs(
                    self._select(capability), fp8_paged_mqa_logits_torch_sm120
                )


class TestTheTopKCannotAdmitAPaddingPosition(CustomTestCase):
    """Rescue 2: the consumer bounds itself by `seq_len` whatever it is fed.

    The two CUDA top-k kernels do it by construction -- `topk_v1.cuh` calls
    `radix_topk(score_ptr, ..., seq_len)` and `topk_v2.cuh` derives every loop
    bound from `problem.seq_len`, so neither ever loads a score past the end.
    Only the torch fallback scans the full row, and it is the one testable
    without a GPU; it re-masks first.
    """

    TOPK = 96

    def _run_topk(self, logits, seq_lens, page_table):
        batch_size = logits.shape[0]
        out_pages = torch.empty((batch_size, self.TOPK), dtype=torch.int32)
        raw_indices = torch.empty((batch_size, self.TOPK), dtype=torch.int32)
        topk_transform_512_pytorch_vectorized(
            logits, seq_lens, page_table, out_pages, PAGE_SIZE, raw_indices
        )
        return raw_indices

    def _assert_in_range(self, raw_indices, seq_lens):
        for i, seq_len in enumerate(seq_lens):
            row = raw_indices[i]
            selected = row[row >= 0]
            self.assertTrue(
                bool((selected < seq_len).all()),
                f"row {i}: selected {selected[selected >= seq_len].tolist()} "
                f"at or beyond seq_len={seq_len}",
            )

    def test_unmasked_tail_is_rejected_by_the_consumer(self):
        """Feed the top-k a deliberately unmasked tail -- the shape the
        producer had before #425, and the shape tilelang and aiter still hand
        over. Nothing beyond `seq_len` may come out."""
        q, kv, w, sl, pt, msl = _build_inputs(ISSUE_SEQ_LENS)
        logits = fp8_paged_mqa_logits_torch_sm120(q, kv, w, sl, pt, None, msl, False)
        positions = torch.arange(msl)
        invalid = positions.unsqueeze(0) >= sl.unsqueeze(1)
        # 0.0 tail (the pre-#425 producer) and a hostile tail that outscores
        # every real token, which is what uninitialized memory can look like.
        for tail_value in (0.0, 1e30):
            with self.subTest(tail=tail_value):
                unmasked = logits.masked_fill(invalid, tail_value)
                self._assert_in_range(self._run_topk(unmasked, sl, pt), ISSUE_SEQ_LENS)

    def test_masked_tail_selects_the_same_positions(self):
        """And the rescue is not doing anything else: with a properly masked
        tail the selection is identical, so the mask alignment of #425 does not
        move any decision."""
        q, kv, w, sl, pt, msl = _build_inputs(ISSUE_SEQ_LENS)
        masked = fp8_paged_mqa_logits_torch_sm120(q, kv, w, sl, pt, None, msl, False)
        positions = torch.arange(msl)
        invalid = positions.unsqueeze(0) >= sl.unsqueeze(1)
        unmasked = masked.masked_fill(invalid, 0.0)

        from_masked = self._run_topk(masked.clone(), sl, pt)
        from_unmasked = self._run_topk(unmasked, sl, pt)
        for i in range(len(ISSUE_SEQ_LENS)):
            self.assertEqual(
                sorted(from_masked[i].tolist()),
                sorted(from_unmasked[i].tolist()),
                f"row {i}",
            )

    def test_zero_length_row_selects_nothing(self):
        q, kv, w, sl, pt, msl = _build_inputs([0, 64, 200, 256])
        logits = fp8_paged_mqa_logits_torch_sm120(q, kv, w, sl, pt, None, msl, False)
        raw_indices = self._run_topk(
            logits.masked_fill(~torch.isfinite(logits), 0.0), sl, pt
        )
        self.assertTrue(bool((raw_indices[0] == -1).all()))
        self._assert_in_range(raw_indices, [0, 64, 200, 256])


if __name__ == "__main__":
    import sys

    sys.exit(unittest.main())
