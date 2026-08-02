"""#427 -- the DSV4 torch/triton twins the #425 family sweep left disagreeing.

#425 fixed one torch reference that diverged from the implementation it was
supposed to validate. The sweep that found it named four more places in the
same class, and this file pins each of them:

F1  `build_causal_swa_page_indices` masked its raw indices with -1 and THEN
    indexed `full_to_swa_mapping[raw_indices]`. PyTorch wraps -1 to the last
    mapping entry, so every out-of-window column came back holding a real SWA
    cache id while the triton twin stored -1. The dspark parity test compared
    only the attended region, so the oracle could not see it -- the same blind
    spot as #425. Production always takes the triton path (dispatch is by
    input placement, and serving inputs are on CUDA), so this is an oracle
    fix; the reachability claim is pinned here rather than asserted.

F2  `topk_transform_512` (the v1 wrapper, and the DEFAULT top-k on the serving
    path) documented no precondition on `seq_lens`, while both kernels behind
    it read the length as uint32 and take an illegal memory access on a
    negative value. `plan_topk_v2` already documented it. The wrapper now
    documents it and, under SGLANG_DSV4_CHECK_TOPK_SEQ_LENS, refuses before
    the launch.

F4  `build_page_table_positions` floored (`//`) where its triton twin
    truncates. Negative entries cannot reach `req_to_token` today -- it is
    allocated with `torch.zeros` and every writer stores an allocator slot id
    from `arange(1, N+1)` -- so the divergence is latent, not live. Both the
    invariant and the now-matching rounding are pinned.

F6  `_compress_forward_c128_fallback` diverged from `_compress_forward_c128_triton`
    three ways: an invalid plan entry pooled page 0 instead of yielding zeros,
    the output dtype differed, and the degenerate case returned zeros against
    the twin's uninitialized buffer. Both are unwired (they arrived
    never-called with upstream sglang#26208), so they are held to one written
    contract instead of being deleted, since upstream owns the wiring.

F8  `SetKAndS.torch` is equivalent to its triton twin. Nothing was wrong; the
    byte layout it computes is pinned so it stays that way.

Everything except the explicitly CUDA-gated `SetKAndS` parity case is CPU-only.
"""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from sglang.jit_kernel.dsv4 import topk as topk_mod
from sglang.jit_kernel.dsv4.compress import (
    CompressorDecodePlan,
    CompressorPrefillPlan,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4 import attn_metadata_kernels, compressor_v2
from sglang.srt.layers.attention.dsv4.index_buf_accessor import (
    NopeFp8RopeBf16Pack,
    SetKAndS,
    fp8_dtype,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# F1: build_causal_swa_page_indices
# ---------------------------------------------------------------------------

SWA_WINDOW = 8
PAGE_ALIGN = 12
POOL_ROWS = 4
POOL_LEN = 32
NUM_FULL_SLOTS = 64

# full_to_swa_mapping[-1] is what the buggy order returned for an out-of-window
# column. Give it a value nothing else can produce so a regression is unmistakable.
WRAP_SENTINEL = 777_777


class _FakeCudaTensor(torch.Tensor):
    """A CPU tensor that reports is_cuda, to drive the dispatch decision."""

    @property
    def is_cuda(self) -> bool:  # type: ignore[override]
        return True


def _swa_inputs(seq_lens):
    g = torch.Generator().manual_seed(427)
    req_to_token = torch.randint(
        0, NUM_FULL_SLOTS, (POOL_ROWS, POOL_LEN), dtype=torch.int32, generator=g
    )
    full_to_swa_mapping = torch.randint(
        0, 1 << 16, (NUM_FULL_SLOTS,), dtype=torch.int64, generator=g
    )
    full_to_swa_mapping[-1] = WRAP_SENTINEL
    return dict(
        req_to_token=req_to_token,
        full_to_swa_mapping=full_to_swa_mapping,
        req_pool_indices_repeated=torch.zeros(len(seq_lens), dtype=torch.int32),
        seq_lens_casual=torch.tensor(seq_lens, dtype=torch.int32),
        swa_window=SWA_WINDOW,
        page_index_aligned_size=PAGE_ALIGN,
    )


class TestCausalSwaPageIndicesReference(CustomTestCase):
    def test_out_of_window_columns_are_minus_one(self):
        # Lengths short of, exactly at, and past the window.
        seq_lens = [1, 3, SWA_WINDOW, SWA_WINDOW + 5]
        out = attn_metadata_kernels.build_causal_swa_page_indices(
            **_swa_inputs(seq_lens)
        )
        padded_width = ((SWA_WINDOW + PAGE_ALIGN - 1) // PAGE_ALIGN) * PAGE_ALIGN
        self.assertEqual(out.shape, (len(seq_lens), padded_width))
        self.assertEqual(out.dtype, torch.int32)

        col = torch.arange(padded_width).view(1, -1)
        attended = col < torch.clamp(
            torch.tensor(seq_lens, dtype=torch.int32), max=SWA_WINDOW
        ).view(-1, 1)
        # Non-degenerate: rows 0 and 1 really do have out-of-window columns
        # INSIDE the window, which is the region the old order got wrong.
        self.assertTrue(bool((~attended[:, :SWA_WINDOW]).any()))
        self.assertTrue(bool((out[~attended] == -1).all()))
        self.assertTrue(bool((out[attended] >= 0).all()))

    def test_no_column_wraps_to_the_last_mapping_entry(self):
        # The direct falsifier for the -1-then-index order: it would have put
        # full_to_swa_mapping[-1] into every out-of-window column.
        out = attn_metadata_kernels.build_causal_swa_page_indices(
            **_swa_inputs([1, 2, 3, 4])
        )
        self.assertFalse(bool((out == torch.tensor(WRAP_SENTINEL).int()).any()))

    def test_full_length_rows_are_untouched(self):
        # The other direction: rows that fill the window must still get real
        # mapped ids, so the fix cannot be "mask everything".
        kw = _swa_inputs([SWA_WINDOW, SWA_WINDOW + 3])
        out = attn_metadata_kernels.build_causal_swa_page_indices(**kw)
        pos = (kw["seq_lens_casual"] - 1).view(-1, 1) - torch.arange(SWA_WINDOW).view(
            1, -1
        )
        raw = kw["req_to_token"][0][pos]
        expected = kw["full_to_swa_mapping"][raw].to(torch.int32)
        self.assertTrue(torch.equal(out[:, :SWA_WINDOW], expected))
        self.assertTrue(bool((out[:, SWA_WINDOW:] == -1).all()))

    def test_dispatch_takes_triton_for_cuda_inputs(self):
        # Pins the reachability claim: the torch reference above is an ORACLE,
        # not a serving path. `execute` routes purely on input placement.
        cls = attn_metadata_kernels.BuildCausalSwaPageIndices
        kw = _swa_inputs([4])
        with mock.patch.object(
            cls, "triton", classmethod(lambda c, **k: "triton")
        ), mock.patch.object(cls, "torch", classmethod(lambda c, **k: "torch")):
            self.assertEqual(cls.execute(**kw), "torch")
            cuda_kw = dict(kw)
            cuda_kw["req_to_token"] = kw["req_to_token"].as_subclass(_FakeCudaTensor)
            self.assertEqual(cls.execute(**cuda_kw), "triton")


# ---------------------------------------------------------------------------
# F2: topk_transform_512 non-negative seq_len precondition
# ---------------------------------------------------------------------------


class _KernelReached(Exception):
    pass


def _refuse_to_load(*args, **kwargs):
    raise _KernelReached("the guard let a bad seq_lens through to the kernel")


class TestTopkV1SeqLenPrecondition(CustomTestCase):
    def _call(self, seq_lens):
        n = seq_lens.numel()
        topk_mod.topk_transform_512(
            torch.zeros(n, 1024),
            seq_lens,
            torch.zeros(n, 16, dtype=torch.int32),
            torch.zeros(n, 512, dtype=torch.int32),
            64,
        )

    def test_negative_seq_len_is_refused_before_the_kernel(self):
        with envs.SGLANG_DSV4_CHECK_TOPK_SEQ_LENS.override(True), mock.patch.object(
            topk_mod, "_jit_topk_v1_module", _refuse_to_load
        ), mock.patch.object(topk_mod, "is_hip_runtime", lambda: False):
            with self.assertRaises(topk_mod.NegativeSeqLenError) as cm:
                self._call(torch.tensor([16, -4, 32], dtype=torch.int32))
        # The message has to name the value and the uint32 reinterpretation,
        # since that is the whole reason the kernel cannot report it itself.
        self.assertIn("-4", str(cm.exception))
        self.assertIn("4294967292", str(cm.exception))

    def test_non_negative_seq_len_reaches_the_kernel(self):
        # Other direction: the guard must not be a blanket refusal, and zero
        # is the documented way to say "no tokens".
        with envs.SGLANG_DSV4_CHECK_TOPK_SEQ_LENS.override(True), mock.patch.object(
            topk_mod, "_jit_topk_v1_module", _refuse_to_load
        ), mock.patch.object(topk_mod, "is_hip_runtime", lambda: False):
            with self.assertRaises(_KernelReached):
                self._call(torch.tensor([0, 16, 32], dtype=torch.int32))

    def test_check_is_off_by_default(self):
        # It costs a device sync, so the serving default is off. Pinned so the
        # default cannot flip unnoticed.
        self.assertFalse(envs.SGLANG_DSV4_CHECK_TOPK_SEQ_LENS.get())
        with mock.patch.object(
            topk_mod, "_jit_topk_v1_module", _refuse_to_load
        ), mock.patch.object(topk_mod, "is_hip_runtime", lambda: False):
            with self.assertRaises(_KernelReached):
                self._call(torch.tensor([-4], dtype=torch.int32))

    def test_v2_wrappers_carry_the_same_guard(self):
        bad = torch.tensor([-1, 8], dtype=torch.int32)
        with envs.SGLANG_DSV4_CHECK_TOPK_SEQ_LENS.override(True), mock.patch.object(
            topk_mod, "_jit_topk_v2_module", _refuse_to_load
        ):
            with self.assertRaises(topk_mod.NegativeSeqLenError):
                topk_mod.plan_topk_v2(bad)
            with self.assertRaises(topk_mod.NegativeSeqLenError):
                topk_mod.topk_transform_512_v2(
                    torch.zeros(2, 1024),
                    bad,
                    torch.zeros(2, 16, dtype=torch.int32),
                    torch.zeros(2, 512, dtype=torch.int32),
                    64,
                    torch.zeros(3, 2, dtype=torch.int32),
                )


# ---------------------------------------------------------------------------
# F4: build_page_table_positions rounding
# ---------------------------------------------------------------------------

PAGE_SIZE = 256


class TestPageTablePositionsRounding(CustomTestCase):
    def _run(self, req_to_token):
        return attn_metadata_kernels.build_page_table_positions(
            req_to_token=req_to_token,
            req_pool_indices_repeated=torch.zeros(1, dtype=torch.int32),
            seq_lens_casual=torch.tensor([512], dtype=torch.int32),
            max_seq_len=req_to_token.shape[1],
            page_size=PAGE_SIZE,
            swa_window=128,
        )

    def test_negative_entry_truncates_like_the_triton_twin(self):
        # triton's `//` on tl.int32 lowers to sdiv, which truncates toward
        # zero: -1 // 256 == 0, not -1. Latent today (see the invariant test
        # below) but it is what the kernel does.
        rt = torch.zeros(1, 4 * PAGE_SIZE, dtype=torch.int32)
        rt[0, 0] = -1
        rt[0, PAGE_SIZE] = -PAGE_SIZE
        rt[0, 2 * PAGE_SIZE] = -PAGE_SIZE - 1
        got = self._run(rt).page_table
        self.assertEqual([int(v) for v in got[0, :3]], [0, -1, -1])

    def test_non_negative_entries_are_unchanged(self):
        # Other direction: the rounding change must be a no-op on the domain
        # that actually occurs.
        rt = torch.arange(4 * PAGE_SIZE, dtype=torch.int32).view(1, -1)
        got = self._run(rt).page_table
        expected = rt[0, ::PAGE_SIZE] // PAGE_SIZE
        self.assertTrue(torch.equal(got[0], expected.to(torch.int32)))

    def test_req_to_token_pool_is_zero_filled(self):
        # The invariant the F4 verdict rests on: the pool is torch.zeros, so
        # columns past a request's seq_len -- which this kernel does read,
        # since it slices by max_seq_len -- are 0 or a stale positive slot id.
        from sglang.srt.mem_cache.memory_pool import ReqToTokenPool

        pool = ReqToTokenPool(
            size=4, max_context_len=8, device="cpu", enable_memory_saver=False
        )
        self.assertEqual(pool.req_to_token.dtype, torch.int32)
        self.assertTrue(bool((pool.req_to_token >= 0).all()))
        # Slot ids handed out start at 1; slot 0 is the always-zero pad row.
        self.assertTrue(all(i >= 1 for i in pool.free_slots))


# ---------------------------------------------------------------------------
# F6: the unwired C128 compress twins
# ---------------------------------------------------------------------------

HEAD_DIM = 8
COMPRESS_RATIO = 128


def _c128_buffer(num_pages):
    g = torch.Generator().manual_seed(6427)
    return torch.randn(
        num_pages, COMPRESS_RATIO, HEAD_DIM * 2, generator=g, dtype=torch.float32
    )


def _decode_plan(seq_lens, write_locs, read_pages):
    raw = torch.zeros(len(seq_lens), 4, dtype=torch.int32)
    raw[:, 0] = torch.tensor(seq_lens, dtype=torch.int32)
    raw[:, 1] = torch.tensor(write_locs, dtype=torch.int32)
    raw[:, 2] = torch.tensor(read_pages, dtype=torch.int32)
    return CompressorDecodePlan(128, raw.view(torch.uint8))


def _prefill_plan(read_pages):
    raw = torch.zeros(len(read_pages), 4, dtype=torch.int32)
    raw[:, 2] = torch.tensor(read_pages, dtype=torch.int32)
    plan_w = torch.zeros(0, 2, dtype=torch.int32).view(torch.uint8)
    return CompressorPrefillPlan(128, raw.view(torch.uint8), plan_w)


class TestCompressC128FallbackContract(CustomTestCase):
    def test_the_pair_is_still_unwired(self):
        # The named decision for F6 is "align, do not delete, because upstream
        # owns the wiring". That decision is only valid while nothing calls
        # them; if a call site appears, revisit rather than trust this file.
        import inspect

        src = inspect.getsource(compressor_v2)
        for name in (
            "_compress_forward_c128_fallback",
            "_compress_forward_c128_triton",
        ):
            self.assertEqual(src.count(name + "("), 1, f"{name} gained a caller")

    def test_invalid_plan_entry_yields_zeros_not_a_page_0_pool(self):
        buf = _c128_buffer(3)
        inp = torch.randn(2, HEAD_DIM * 2)
        ape = torch.randn(COMPRESS_RATIO, HEAD_DIM)
        out = compressor_v2._compress_forward_c128_fallback(
            buf, inp, ape, _prefill_plan([-1, 1]), HEAD_DIM
        )
        # Row 0 names no page: zeros, and specifically NOT the page-0 pool.
        self.assertTrue(bool((out[0] == 0).all()))
        page0 = compressor_v2._compress_forward_c128_fallback(
            buf, inp, ape, _prefill_plan([0, 1]), HEAD_DIM
        )[0]
        self.assertFalse(bool((page0 == 0).all()))
        # Row 1 is untouched by the invalid-row handling.
        self.assertTrue(bool((out[1] != 0).any()))

    def test_output_dtype_matches_the_triton_twin(self):
        buf = _c128_buffer(2)
        ape = torch.randn(COMPRESS_RATIO, HEAD_DIM)
        for inp_dtype in (torch.float32, torch.bfloat16, torch.float16):
            inp = torch.randn(1, HEAD_DIM * 2).to(inp_dtype)
            out = compressor_v2._compress_forward_c128_fallback(
                buf.to(inp_dtype), inp, ape, _prefill_plan([0]), HEAD_DIM
            )
            self.assertEqual(out.dtype, torch.float32, f"from {inp_dtype}")

    def test_degenerate_shapes_are_defined_and_float32(self):
        ape = torch.randn(COMPRESS_RATIO, HEAD_DIM)
        # A non-float32 input, so that "follow kv_score_input.dtype" is
        # distinguishable from "always float32" on this path too.
        for inp_dtype in (torch.float32, torch.bfloat16):
            # No pages at all.
            out = compressor_v2._compress_forward_c128_fallback(
                _c128_buffer(0).to(inp_dtype),
                torch.randn(2, HEAD_DIM * 2).to(inp_dtype),
                ape,
                _prefill_plan([0, 0]),
                HEAD_DIM,
            )
            self.assertEqual(out.shape, (2, HEAD_DIM))
            self.assertEqual(out.dtype, torch.float32, f"from {inp_dtype}")
            self.assertTrue(bool((out == 0).all()))
            # No tokens.
            out = compressor_v2._compress_forward_c128_fallback(
                _c128_buffer(2).to(inp_dtype),
                torch.randn(0, HEAD_DIM * 2).to(inp_dtype),
                ape,
                _prefill_plan([]),
                HEAD_DIM,
            )
            self.assertEqual(out.shape, (0, HEAD_DIM))
            self.assertEqual(out.dtype, torch.float32, f"from {inp_dtype}")

    def test_decode_non_boundary_rows_stay_zero(self):
        # Unchanged behaviour, pinned so the invalid-row masking above cannot
        # accidentally displace it.
        buf = _c128_buffer(2)
        inp = torch.randn(2, HEAD_DIM * 2)
        ape = torch.randn(COMPRESS_RATIO, HEAD_DIM)
        out = compressor_v2._compress_forward_c128_fallback(
            buf, inp, ape, _decode_plan([256, 200], [0, 1], [0, 1]), HEAD_DIM
        )
        self.assertTrue(bool((out[0] != 0).any()))  # 256 % 128 == 0
        self.assertTrue(bool((out[1] == 0).all()))  # 200 % 128 != 0


# ---------------------------------------------------------------------------
# F8: SetKAndS.torch
# ---------------------------------------------------------------------------

NOPE_DIM = 448
ROPE_DIM = 64
SCALE_DIM = 7


def _set_k_and_s_case(page_size, num_pages, locs, device="cpu"):
    g = torch.Generator(device=device).manual_seed(8427)
    buf_numel_per_page = page_size * (NOPE_DIM + ROPE_DIM * 2 + SCALE_DIM + 1)
    buf = torch.zeros(num_pages, buf_numel_per_page, dtype=torch.uint8, device=device)
    n = len(locs)
    pack = NopeFp8RopeBf16Pack(
        k_nope_fp8=torch.randint(
            0, 200, (n, NOPE_DIM), dtype=torch.uint8, device=device, generator=g
        ).view(fp8_dtype),
        k_rope_bf16=torch.randn(
            n, ROPE_DIM, device=device, generator=g, dtype=torch.float32
        ).to(torch.bfloat16),
        scale_k_nope_ue8m0=torch.randint(
            1, 254, (n, SCALE_DIM), dtype=torch.uint8, device=device, generator=g
        ),
    )
    return buf, pack, torch.tensor(locs, dtype=torch.int64, device=device)


class _Pool:
    def __init__(self, page_size):
        self.page_size = page_size


class TestSetKAndSTorch(CustomTestCase):
    def test_torch_writes_the_documented_byte_layout(self):
        # SetKAndS.torch was audited as equivalent to its triton twin; nothing
        # was wrong with it. This pins the layout it produces so a later tidy-up
        # of the pair cannot move the reference out from under the kernel.
        page_size, num_pages = 4, 3
        locs = [0, 5, 6, 11]
        buf, pack, loc = _set_k_and_s_case(page_size, num_pages, locs)
        SetKAndS.torch(_Pool(page_size), buf, loc, pack)

        buf_numel_per_page = buf.shape[1]
        nope_rope_bytes = NOPE_DIM + ROPE_DIM * 2
        s_offset_in_page = page_size * nope_rope_bytes
        flat = buf.flatten()
        for i, location in enumerate(locs):
            page, off = divmod(location, page_size)
            base = page * buf_numel_per_page + off * nope_rope_bytes
            self.assertTrue(
                torch.equal(
                    flat[base : base + NOPE_DIM],
                    pack.k_nope_fp8[i].view(torch.uint8),
                )
            )
            self.assertTrue(
                torch.equal(
                    flat[base + NOPE_DIM : base + nope_rope_bytes].view(torch.bfloat16),
                    pack.k_rope_bf16[i],
                )
            )
            s_base = (
                page * buf_numel_per_page + s_offset_in_page + off * (SCALE_DIM + 1)
            )
            self.assertTrue(
                torch.equal(
                    flat[s_base : s_base + SCALE_DIM], pack.scale_k_nope_ue8m0[i]
                )
            )

    def test_untouched_slots_stay_zero(self):
        # Other direction: the writes must not spill into neighbouring slots.
        page_size, num_pages = 4, 2
        buf, pack, loc = _set_k_and_s_case(page_size, num_pages, [1])
        SetKAndS.torch(_Pool(page_size), buf, loc, pack)
        self.assertTrue(bool((buf[1] == 0).all()))
        nope_rope_bytes = NOPE_DIM + ROPE_DIM * 2
        self.assertTrue(bool((buf[0, :nope_rope_bytes] == 0).all()))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU for the triton twin")
    def test_torch_matches_triton(self):
        page_size, num_pages = 4, 3
        locs = [0, 5, 6, 11]
        buf_t, pack, loc = _set_k_and_s_case(page_size, num_pages, locs, "cuda")
        buf_r = buf_t.clone()
        SetKAndS.torch(_Pool(page_size), buf_r, loc, pack)
        SetKAndS.triton(_Pool(page_size), buf_t, loc, pack)
        self.assertTrue(torch.equal(buf_t, buf_r))


if __name__ == "__main__":
    import sys

    sys.exit(unittest.main())
