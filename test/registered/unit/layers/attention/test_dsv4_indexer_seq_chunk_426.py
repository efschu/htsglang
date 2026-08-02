"""#426 -- the torch paged-MQA-logits path must not size its peak by context.

Upstream sgl-project/sglang#33246: ``fp8_paged_mqa_logits_torch_sm120`` builds
the whole ``[batch, seq, num_heads]`` bmm product and then sums the head axis
away on the very next line, so the largest allocation in the function is
``num_heads`` times the value it returns and grows linearly with context --
~15 GiB per rank for one allocation at a 1M-token DeepSeek-V4 context, with
all ranks OOMing together.

For this fork it is not a fallback concern. Since #417 Cut 3
``select_paged_mqa_logits_fn`` hands this function to EVERY card that takes the
torch backend (Ampere, Ada, consumer Blackwell), so it is the production
indexer path on the rig this fork exists for.

Two properties are pinned, and they pull against each other, which is the point:

* BOUNDED PEAK -- the largest intermediate scales with the chunk, not with the
  sequence. Measured by intercepting ``torch.bmm``: the unfixed function calls
  it once with an ``[B, S, H]`` result, the fixed one calls it once per chunk
  with ``[B, chunk, H]``.
* EXACT -- chunking is a regrouping of independent rows, not an approximation.
  Every output element reduces over ``head_dim`` (inside the bmm) and over
  heads (inside one row); neither reduction crosses a chunk boundary. Pinned
  at atol=0/rtol=0 against the single-pass result, and against the unchunked
  reference ``fp8_paged_mqa_logits_torch``.

The chunk default (8192 positions) is far above the shapes the #425 golden pins
use, so those keep running the single-pass expression op for op.

GPU-free: everything here is CPU float32.
"""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.indexer import (
    FP8_DTYPE,
    fp8_paged_mqa_logits_torch,
    fp8_paged_mqa_logits_torch_sm120,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

PAGE_SIZE = 64
HEAD_DIM = 128
PAGE_BYTES = PAGE_SIZE * HEAD_DIM + PAGE_SIZE * 4


def _build_inputs(seq_lens, *, num_heads=8, seed=0):
    """Paged FP8 index cache in the production layout, on CPU.

    Same construction as test_dsv4_indexer_mask_425.py; kept local so the two
    files can drift apart without silently coupling their fixtures.
    """
    batch_size = len(seq_lens)
    max_seq_len = max(seq_lens)
    max_pages = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    num_pages_total = batch_size * max_pages + 1

    g = torch.Generator().manual_seed(seed)
    raw = torch.empty(num_pages_total, PAGE_BYTES, dtype=torch.uint8)
    values = (torch.randn(num_pages_total, PAGE_SIZE * HEAD_DIM, generator=g) * 0.5).to(
        FP8_DTYPE
    )
    raw[:, : PAGE_SIZE * HEAD_DIM] = values.view(dtype=torch.uint8)
    scales = torch.rand(num_pages_total, PAGE_SIZE, generator=g) + 0.5
    raw[:, PAGE_SIZE * HEAD_DIM :] = scales.contiguous().view(dtype=torch.uint8)

    kvcache = raw.view(num_pages_total, PAGE_SIZE, 1, HEAD_DIM + 4)

    page_table = torch.arange(1, batch_size * max_pages + 1, dtype=torch.int32).view(
        batch_size, max_pages
    )

    q = (torch.randn(batch_size, 1, num_heads, HEAD_DIM, generator=g) * 0.5).to(
        FP8_DTYPE
    )
    weight = torch.rand(batch_size, num_heads, generator=g) + 0.1
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)

    return dict(
        q_fp8=q,
        kvcache_fp8=kvcache,
        weight=weight,
        seq_lens=seq_lens_t,
        page_table=page_table,
        deep_gemm_metadata=None,
        max_seq_len=max_seq_len,
        clean_logits=False,
    )


class _BmmProbe:
    """Records the shape of every ``torch.bmm`` result inside the call."""

    def __init__(self):
        self.result_shapes = []

    def __enter__(self):
        real_bmm = torch.bmm

        def probed(a, b, **kwargs):
            out = real_bmm(a, b, **kwargs)
            self.result_shapes.append(tuple(out.shape))
            return out

        self._patch = mock.patch.object(torch, "bmm", probed)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        return False

    @property
    def peak_elements(self):
        return max(
            (s[0] * s[1] * s[2] for s in self.result_shapes),
            default=0,
        )


def _chunk_env():
    """The chunk knob, or None on a tree that does not have it yet.

    Resolved by name so the assertions below fail on their own terms against an
    unfixed tree (one oversized bmm) instead of erroring at import -- an
    ImportError is not evidence that the peak is unbounded.
    """
    return getattr(envs, "SGLANG_DSV4_INDEXER_LOGITS_SEQ_CHUNK", None)


def _chunk_pages_fn():
    from sglang.srt.layers.attention.dsv4 import indexer

    return getattr(indexer, "_indexer_logits_chunk_pages", None)


def _run(inputs, chunk_positions):
    env = _chunk_env()
    if env is None:
        return fp8_paged_mqa_logits_torch_sm120(**inputs)
    with env.override(chunk_positions):
        return fp8_paged_mqa_logits_torch_sm120(**inputs)


class TestPeakIsBoundedByTheChunk(CustomTestCase):
    """The falsifier. Unfixed, the single bmm is [B, S, H] whatever the env says."""

    def test_bmm_result_never_exceeds_one_chunk(self):
        seq_lens = [4096, 4096, 3000, 2048]
        num_heads = 8
        inputs = _build_inputs(seq_lens, num_heads=num_heads)
        chunk_positions = 512

        with _BmmProbe() as probe:
            _run(inputs, chunk_positions)

        batch_size = len(seq_lens)
        allowed = batch_size * chunk_positions * num_heads
        unfixed = batch_size * max(seq_lens) * num_heads
        self.assertLessEqual(
            probe.peak_elements,
            allowed,
            f"largest bmm result {probe.peak_elements} elements exceeds one "
            f"chunk ({allowed}); the single-pass shape would be {unfixed}",
        )
        # The bound has to be tight enough to fail on the unfixed tree.
        self.assertLess(allowed, unfixed)

    def test_the_peak_does_not_follow_the_context_length(self):
        """Doubling the context must not double the largest intermediate.

        This is the shape of the upstream report -- their diagnostic signal was
        that lowering --chunked-prefill-size barely moved the allocation, i.e.
        the dominant axis is the sequence, not the batch.
        """
        peaks = {}
        for seq in (2048, 4096):
            inputs = _build_inputs([seq, seq], num_heads=8)
            with _BmmProbe() as probe:
                _run(inputs, 512)
            peaks[seq] = probe.peak_elements
        self.assertEqual(peaks[2048], peaks[4096])

    def test_the_probe_sees_the_unchunked_shape_when_chunking_is_off(self):
        """Can-fail arm: with chunking disabled the peak IS the [B,S,H] tensor,
        so the assertions above are measuring something real."""
        seq_lens = [1024, 1024]
        inputs = _build_inputs(seq_lens, num_heads=8)
        with _BmmProbe() as probe:
            _run(inputs, 0)
        self.assertEqual(probe.result_shapes, [(2, 1024, 8)])


class TestChunkingIsExact(CustomTestCase):
    def test_every_chunk_width_gives_the_identical_result(self):
        seq_lens = [640, 512, 200, 64]
        inputs = _build_inputs(seq_lens, num_heads=8, seed=3)
        single = _run(inputs, 0)
        for chunk_positions in (64, 128, 192, 256, 512, 8192):
            with self.subTest(chunk=chunk_positions):
                torch.testing.assert_close(
                    _run(inputs, chunk_positions), single, atol=0.0, rtol=0.0
                )

    def test_a_chunk_boundary_that_splits_the_tail_page(self):
        """max_seq_len is not a multiple of the page, so the last chunk is the
        one that has to clamp its copy width."""
        seq_lens = [200, 130, 64]
        inputs = _build_inputs(seq_lens, num_heads=4, seed=7)
        single = _run(inputs, 0)
        for chunk_positions in (64, 128):
            with self.subTest(chunk=chunk_positions):
                torch.testing.assert_close(
                    _run(inputs, chunk_positions), single, atol=0.0, rtol=0.0
                )

    def test_chunked_still_matches_the_unchunked_reference_implementation(self):
        """The #425 contract (the two torch references agree bit for bit) must
        survive chunking -- the reference is deliberately left single-pass."""
        seq_lens = [256, 192, 128, 64]
        inputs = _build_inputs(seq_lens, num_heads=8, seed=11)
        reference = fp8_paged_mqa_logits_torch(**inputs)
        for chunk_positions in (64, 128, 8192):
            with self.subTest(chunk=chunk_positions):
                torch.testing.assert_close(
                    _run(inputs, chunk_positions), reference, atol=0.0, rtol=0.0
                )


class TestChunkWidthResolution(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.chunk_pages = _chunk_pages_fn()
        self.env = _chunk_env()
        if self.chunk_pages is None or self.env is None:
            self.fail("the indexer has no sequence-axis chunk width to resolve")

    def test_disabled_and_oversized_both_mean_one_pass(self):
        for chunk_positions in (0, -1, 10**9):
            with self.subTest(chunk=chunk_positions):
                with self.env.override(chunk_positions):
                    self.assertEqual(self.chunk_pages(64, 17), 17)

    def test_the_width_is_rounded_down_to_whole_pages(self):
        with self.env.override(200):
            self.assertEqual(self.chunk_pages(64, 100), 3)

    def test_a_sub_page_request_still_walks_one_page(self):
        """The page is the unit the gather indexes in; it cannot go below it."""
        with self.env.override(1):
            self.assertEqual(self.chunk_pages(64, 100), 1)

    def test_the_default_leaves_the_golden_pin_shapes_single_pass(self):
        """The #425 pins run at max_seq_len=256 -- four pages, one chunk."""
        self.assertEqual(self.chunk_pages(64, 4), 4)


if __name__ == "__main__":
    unittest.main()
