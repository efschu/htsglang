"""Numerical falsifier for the Ampere fp8-KV decode bug class (vLLM #39137 / #41343).

Bug class under test
--------------------
Two failure modes have been reported against another engine on Ampere-class
hardware (sm80 / sm86):

  * an ``fp8_e4m3`` KV cache is decoded with the wrong format (as ``e5m2``, or
    through a code path that only exists for ``e5m2``), so every cached K/V
    value comes back with a wrong exponent bias and a wrong mantissa width;
  * an ``e5m2`` KV cache combined with default (unloaded) k/v scales silently
    diverges because the store and load sides disagree about the scale.

Both are *silent*: no exception, no warning, only wrong numbers.

Diagnosis in this stack
-----------------------
``ModelRunner.configure_kv_cache_dtype`` maps the CLI string straight onto
``torch.float8_e4m3fn`` / ``torch.float8_e5m2`` with no architecture gate and no
coupling to the checkpoint's quantization algorithm, and the KV pool stores the
bytes as ``uint8`` and hands the attention backend a ``.view(self.dtype)`` of
them.  The format therefore lives in exactly one place -- the dtype of the view
-- and the dequantization happens inside the attention kernel.  This test pins
that end to end on whatever GPU it runs on, which on this rig is sm86.

Evidence produced here (four levels, cheapest first)
----------------------------------------------------
1. ``torch``'s own e4m3 decode of a known byte grid on the GPU, against a
   pure-Python IEEE decoder that never touches a torch fp8 type.
2. The quantization direction (bf16 -> e4m3) on the GPU against the CPU cast.
3. A real ``MHATokenToKVPool`` round trip (``set_kv_buffer`` -> ``get_key_buffer``),
   i.e. the uint8 store / fp8 view aliasing the server actually uses.
4. The FlashInfer paged-decode call path the server uses for MHA decode
   (``BatchDecodeWithPagedKVCacheWrapper``, page_size=1, NHD, kv_data_type =
   the pool dtype), with one KV token per request so the softmax weight is
   exactly 1.0 and the attention output *is* the dequantized V row.

Every level carries a cross-probe: the same bytes reinterpreted as ``e5m2``
must deviate grossly.  Without that cross-probe an "it matches" result would be
worthless, because a test that cannot see the confusion cannot rule it out.

Measured on an RTX 3080, capability (8, 6), over all 256 finite fp8 byte
patterns::

    flashinfer paged decode, e4m3 pool: max|out - ref_e4m3| = 0.0
                                        max|out - ref_e5m2| = 56992.0
    flashinfer paged decode, e5m2 pool: max|out - ref_e5m2| = 0.0
                                        max|out - ref_e4m3| = 56992.0
    triton kernel load of e4m3:  ValueError("type fp8e4nv not supported in
                                 this architecture. The supported fp8 dtypes
                                 are ('fp8e4b15', 'fp8e5')")
    triton kernel load of e5m2:  compiles

Pin the device by PCI order when reproducing -- with the default
``CUDA_DEVICE_ORDER=FASTEST_FIRST`` a bare ``CUDA_VISIBLE_DEVICES=0`` selects
the newest card on a mixed rig, not the Ampere one this file is about::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<ampere> \
        python -m pytest test/registered/unit/layers/attention/test_fp8_e4m3_kv_decode.py -v
"""

import math
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

_HAS_CUDA = torch.cuda.is_available()

# Bytes that are NaN or Inf under e4m3fn or under e5m2. Excluded from the value
# grid so both interpretations stay finite and directly comparable:
#   e4m3fn NaN : 0x7f, 0xff            (exp=15 and mantissa=7; e4m3fn has no Inf)
#   e5m2       : 0x7c-0x7f, 0xfc-0xff  (exp=31 -> Inf when mantissa=0, else NaN)
_NON_FINITE_BYTES = frozenset({0x7C, 0x7D, 0x7E, 0x7F, 0xFC, 0xFD, 0xFE, 0xFF})

# FlashInfer's fp8 decode kernels are specialized on a small set of head dims.
_HEAD_DIM = 128


def _decode_fp8_byte(byte: int, *, ebits: int, mbits: int, bias: int, has_inf: bool):
    """Decode one fp8 byte by the IEEE-754 rules, in pure Python.

    Deliberately free of any torch fp8 type: if torch's own decode were wrong
    on this architecture, a torch-based reference would be wrong in the same
    way and the test would pass vacuously.
    """
    sign = -1.0 if (byte >> 7) & 1 else 1.0
    exp = (byte >> mbits) & ((1 << ebits) - 1)
    mant = byte & ((1 << mbits) - 1)
    exp_max = (1 << ebits) - 1

    if exp == exp_max:
        if has_inf:
            return math.nan if mant else sign * math.inf
        if mant == (1 << mbits) - 1:
            # e4m3fn: only the all-ones mantissa is NaN, the rest are normals.
            return math.nan
    if exp == 0:
        return sign * (2.0 ** (1 - bias)) * (mant / (1 << mbits))
    return sign * (2.0 ** (exp - bias)) * (1.0 + mant / (1 << mbits))


def _decode_e4m3fn(byte: int) -> float:
    return _decode_fp8_byte(byte, ebits=4, mbits=3, bias=7, has_inf=False)


def _decode_e5m2(byte: int) -> float:
    return _decode_fp8_byte(byte, ebits=5, mbits=2, bias=15, has_inf=True)


def _finite_byte_grid(pad_to: int = 1) -> list:
    """All byte patterns that are finite under both e4m3fn and e5m2.

    ``pad_to`` repeats the benign 0x00 pattern until the grid length is a
    multiple of it, so the whole grid fits an exact number of KV rows.
    """
    grid = [b for b in range(256) if b not in _NON_FINITE_BYTES]
    while len(grid) % pad_to:
        grid.append(0x00)
    return grid


def _e4m3_representable_values() -> list:
    """A spread of values that are exactly representable in e4m3fn.

    Chosen so a bf16 -> e4m3 -> bf16 round trip is lossless, which lets the
    round-trip tests assert exact equality instead of a tolerance that could
    hide a systematic bias.
    """
    return [_decode_e4m3fn(b) for b in _finite_byte_grid()]


class _FakeLayer:
    """Minimal stand-in for ``RadixAttention`` as ``set_kv_buffer`` uses it."""

    def __init__(self, layer_id: int = 0):
        self.layer_id = layer_id


def _make_pool(dtype: torch.dtype, *, size: int, head_dim: int, head_num: int = 1):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    return MHATokenToKVPool(
        size=size,
        page_size=1,
        dtype=dtype,
        head_num=head_num,
        head_dim=head_dim,
        layer_num=1,
        device="cuda",
        enable_memory_saver=False,
    )


@unittest.skipUnless(_HAS_CUDA, "fp8 KV decode falsifier requires CUDA")
class TestFp8E4M3KVDecode(CustomTestCase):
    """e4m3 KV must decode as e4m3 -- and the test must be able to see it if it did not."""

    @classmethod
    def setUpClass(cls):
        cls.capability = torch.cuda.get_device_capability()
        cls.device_name = torch.cuda.get_device_name()
        # 248 finite bytes, padded to 256 == 2 rows of head_dim=128, so the
        # FlashInfer arm covers every finite fp8 bit pattern.
        cls.byte_grid = _finite_byte_grid(pad_to=_HEAD_DIM)
        cls.ref_e4m3 = [_decode_e4m3fn(b) for b in cls.byte_grid]
        cls.ref_e5m2 = [_decode_e5m2(b) for b in cls.byte_grid]

    def _arch_note(self) -> str:
        return f"device={self.device_name} sm{self.capability[0]}{self.capability[1]}"

    # ---------------------------------------------------------------- level 0

    def test_reference_decoders_disagree(self):
        """The cross-probe is only meaningful if the two decoders differ.

        Guards the guard: if e4m3 and e5m2 decoded the same byte grid to the
        same numbers, every "matches e4m3, not e5m2" assertion below would be
        untestable.
        """
        differing = sum(
            1
            for a, b in zip(self.ref_e4m3, self.ref_e5m2)
            if not (a == b or (math.isnan(a) and math.isnan(b)))
        )
        # Only the zeros and a handful of shared power-of-two-ish patterns can
        # coincide; the overwhelming majority must differ.
        self.assertGreater(
            differing,
            0.9 * len(self.byte_grid),
            f"e4m3 and e5m2 references agree on too many of {len(self.byte_grid)} "
            f"bytes ({len(self.byte_grid) - differing} identical) -- the cross-probe "
            "would not detect a format confusion",
        )

    # ---------------------------------------------------------------- level 1

    def test_gpu_e4m3_view_decode_matches_reference(self):
        """Dequantizing the byte grid on the GPU must give the e4m3 values, exactly."""
        raw = torch.tensor(self.byte_grid, dtype=torch.uint8, device="cuda")
        decoded = raw.view(torch.float8_e4m3fn).to(torch.float64).cpu()

        ref_e4m3 = torch.tensor(self.ref_e4m3, dtype=torch.float64)
        ref_e5m2 = torch.tensor(self.ref_e5m2, dtype=torch.float64)

        self.assertTrue(
            torch.equal(decoded, ref_e4m3),
            f"GPU e4m3 decode deviates from the IEEE reference on {self._arch_note()}: "
            f"max|diff|={(decoded - ref_e4m3).abs().max().item()}",
        )
        # Cross-probe: had the bytes been decoded as e5m2, this comparison would
        # have caught it by a wide margin.
        self.assertGreater(
            (ref_e4m3 - ref_e5m2).abs().max().item(),
            1.0,
            "e5m2 cross-probe is too weak to be evidence",
        )

    # ---------------------------------------------------------------- level 2

    def test_gpu_bf16_to_e4m3_quantization_matches_cpu(self):
        """The store direction (bf16 -> e4m3) must produce the same bytes as the CPU cast.

        This is the cast ``MHATokenToKVPool.set_kv_buffer`` performs
        (``cache_k.to(self.dtype)``) before the uint8 view.
        """
        values = torch.tensor(
            [
                0.0,
                -0.0,
                1.0,
                -1.0,
                0.5,
                1.5,
                448.0,
                -448.0,
                2.0**-9,  # smallest e4m3 subnormal
                2.0**-6,  # smallest e4m3 normal
                0.09375,
                -0.3125,
                3.5,
                -7.0,
                240.0,
                17.0,
            ],
            dtype=torch.float32,
        )
        # Sample on the CPU: on-GPU generation is not architecture-identical,
        # and the point here is the cast, not the input.
        cpu_bytes = values.to(torch.bfloat16).to(torch.float8_e4m3fn).view(torch.uint8)
        gpu_bytes = (
            values.cuda()
            .to(torch.bfloat16)
            .to(torch.float8_e4m3fn)
            .view(torch.uint8)
            .cpu()
        )
        self.assertTrue(
            torch.equal(cpu_bytes, gpu_bytes),
            f"bf16 -> e4m3 quantization differs between CPU and GPU on "
            f"{self._arch_note()}: cpu={cpu_bytes.tolist()} gpu={gpu_bytes.tolist()}",
        )
        # Cross-probe: quantizing the same values as e5m2 must produce a
        # different byte string, otherwise the byte comparison proves nothing.
        e5m2_bytes = values.to(torch.bfloat16).to(torch.float8_e5m2).view(torch.uint8)
        self.assertFalse(
            torch.equal(cpu_bytes, e5m2_bytes),
            "e4m3 and e5m2 quantization of the probe values coincide -- "
            "the probe cannot distinguish the formats",
        )

    # ---------------------------------------------------------------- level 3

    def test_kv_pool_roundtrip_preserves_e4m3(self):
        """``set_kv_buffer`` -> ``get_key_buffer`` must round trip e4m3 values exactly.

        Exercises the real pool: the uint8 store dtype, the ``.view(self.dtype)``
        on read, and the JIT store kernel -- i.e. the aliasing that a format
        confusion would live in.
        """
        head_dim = _HEAD_DIM
        values = _e4m3_representable_values()
        num_tokens = 4
        needed = num_tokens * head_dim
        flat = (values * (needed // len(values) + 1))[:needed]

        pool = _make_pool(torch.float8_e4m3fn, size=num_tokens + 4, head_dim=head_dim)
        try:
            src = torch.tensor(flat, dtype=torch.float32).reshape(
                num_tokens, 1, head_dim
            )
            cache_k = src.to(torch.bfloat16).cuda()
            cache_v = (-src).to(torch.bfloat16).cuda()
            loc = torch.arange(num_tokens, dtype=torch.int64, device="cuda")

            pool.set_kv_buffer(_FakeLayer(0), loc, cache_k.clone(), cache_v.clone())

            k_back = pool.get_key_buffer(0)[:num_tokens].to(torch.float32).cpu()
            v_back = pool.get_value_buffer(0)[:num_tokens].to(torch.float32).cpu()

            self.assertEqual(pool.get_key_buffer(0).dtype, torch.float8_e4m3fn)
            self.assertTrue(
                torch.equal(k_back, src),
                f"K round trip through the e4m3 pool is lossy on {self._arch_note()}: "
                f"max|diff|={(k_back - src).abs().max().item()}",
            )
            self.assertTrue(
                torch.equal(v_back, -src),
                f"V round trip through the e4m3 pool is lossy on {self._arch_note()}: "
                f"max|diff|={(v_back + src).abs().max().item()}",
            )

            # Cross-probe: reading the very same bytes as e5m2 must be grossly wrong.
            raw = pool.k_buffer[0][:num_tokens]
            as_e5m2 = raw.view(torch.float8_e5m2).to(torch.float32).cpu()
            finite = torch.isfinite(as_e5m2) & torch.isfinite(src)
            self.assertGreater(
                (as_e5m2[finite] - src[finite]).abs().max().item(),
                1.0,
                "reinterpreting the pool bytes as e5m2 barely changes the values -- "
                "this round trip could not detect a format confusion",
            )
        finally:
            del pool
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------- level 4

    def _run_flashinfer_single_token_decode(self, pool_dtype: torch.dtype, byte_grid):
        """Decode one KV token per request through the server's FlashInfer path.

        With exactly one KV token per request the softmax weight is 1.0, so the
        attention output equals the dequantized V row bit for bit. Mirrors what
        ``FlashInferAttnBackend`` does: page_size=1, NHD, ``kv_data_type`` set to
        the pool dtype, ``paged_kv_cache`` = ``get_kv_buffer(layer_id)``.
        """
        import flashinfer

        head_dim = _HEAD_DIM
        num_heads = 1
        self.assertEqual(len(byte_grid) % head_dim, 0)
        bs = len(byte_grid) // head_dim
        self.assertGreater(bs, 0)

        pool = _make_pool(pool_dtype, size=bs + 4, head_dim=head_dim)
        try:
            grid = torch.tensor(byte_grid[: bs * head_dim], dtype=torch.uint8).reshape(
                bs, num_heads, head_dim
            )
            # Write raw bytes straight into the uint8 store so the exact bit
            # patterns under test reach the kernel (the bf16 -> fp8 cast is
            # covered by the round-trip test above and would clip the grid).
            pool.v_buffer[0][:bs] = grid.cuda()
            pool.k_buffer[0][:bs] = 0  # byte 0x00 == +0.0 in both fp8 formats

            k_buf, v_buf = pool.get_kv_buffer(0)
            self.assertEqual(k_buf.dtype, pool_dtype)

            workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
            kv_indptr = torch.arange(bs + 1, dtype=torch.int32, device="cuda")
            kv_indices = torch.arange(bs, dtype=torch.int32, device="cuda")
            last_page_len = torch.ones(bs, dtype=torch.int32, device="cuda")
            wrapper.plan(
                kv_indptr,
                kv_indices,
                last_page_len,
                num_heads,
                num_heads,
                head_dim,
                1,  # page_size
                q_data_type=torch.bfloat16,
                kv_data_type=pool_dtype,
            )
            q = torch.ones(bs, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
            out = wrapper.run(q, (k_buf, v_buf))
            return out.to(torch.float64).cpu().reshape(-1)
        finally:
            del pool
            torch.cuda.empty_cache()

    def test_flashinfer_paged_decode_dequantizes_e4m3(self):
        """The server's decode path must return the e4m3 values, not the e5m2 ones."""
        grid = self.byte_grid
        ref_e4m3 = torch.tensor([_decode_e4m3fn(b) for b in grid], dtype=torch.float64)
        ref_e5m2 = torch.tensor([_decode_e5m2(b) for b in grid], dtype=torch.float64)

        out = self._run_flashinfer_single_token_decode(torch.float8_e4m3fn, grid)

        # e4m3 values are exactly representable in bf16 (4-bit significand,
        # exponent well inside bf16's range), so the softmax-weight-1.0 output
        # is exact; a tolerance here would let a systematic bias through.
        err_e4m3 = (out - ref_e4m3).abs().max().item()
        err_e5m2 = (out - ref_e5m2).abs().max().item()
        self.assertTrue(
            torch.equal(out, ref_e4m3),
            f"FlashInfer paged decode of an e4m3 KV pool deviates from the IEEE "
            f"e4m3 reference on {self._arch_note()}: max|diff|={err_e4m3} "
            f"(distance to the e5m2 reading: {err_e5m2})",
        )
        # Cross-probe: an e5m2 misread would have shown up as a large deviation.
        self.assertGreater(
            err_e5m2,
            1.0,
            "the e5m2 reading of the same bytes is within 1.0 of the output -- "
            "this test could not detect the confusion it is meant to rule out",
        )

    def test_flashinfer_paged_decode_dequantizes_e5m2(self):
        """Positive control: the same bytes in an e5m2 pool must decode as e5m2.

        Proves the kernel dispatches on the pool dtype rather than always using
        one format. Together with the e4m3 test this pins both directions.
        """
        grid = self.byte_grid
        ref_e4m3 = torch.tensor([_decode_e4m3fn(b) for b in grid], dtype=torch.float64)
        ref_e5m2 = torch.tensor([_decode_e5m2(b) for b in grid], dtype=torch.float64)

        out = self._run_flashinfer_single_token_decode(torch.float8_e5m2, grid)

        self.assertTrue(
            torch.equal(out, ref_e5m2),
            f"FlashInfer paged decode of an e5m2 KV pool deviates from the IEEE "
            f"e5m2 reference on {self._arch_note()}: "
            f"max|diff|={(out - ref_e5m2).abs().max().item()}",
        )
        self.assertGreater(
            (out - ref_e4m3).abs().max().item(),
            1.0,
            "the e4m3 reading of the same bytes is within 1.0 of the output",
        )


@unittest.skipUnless(_HAS_CUDA, "Triton lane probe requires CUDA")
class TestTritonLaneFp8KvBehaviour(CustomTestCase):
    """Pins *how* the Triton attention lane reacts to an fp8 KV pool.

    ``triton_backend.py`` contains no ``kv_cache_dtype`` handling at all: it
    hands ``get_key_buffer(...)`` -- an fp8 view -- straight to the Triton
    extend / decode kernels, which do ``tl.dot(q.to(k.dtype), k)``. The Triton
    NVIDIA backend only adds ``fp8e4nv`` to ``supported_fp8_dtypes`` at
    capability >= 89, so on Ampere the kernel must fail to compile rather than
    produce a wrong number. This test exists to keep it that way: a future
    change that makes the fp8 type compile on sm86 without giving it the right
    decode would turn a loud failure into exactly the silent corruption this
    file is about.
    """

    def _compile_probe(self, dtype: torch.dtype):
        import triton
        import triton.language as tl

        @triton.jit
        def _load_fp8(src_ptr, dst_ptr, N: tl.constexpr):
            offs = tl.arange(0, N)
            tl.store(dst_ptr + offs, tl.load(src_ptr + offs).to(tl.float32))

        src = torch.zeros(64, dtype=torch.uint8, device="cuda").view(dtype)
        dst = torch.zeros(64, dtype=torch.float32, device="cuda")
        _load_fp8[(1,)](src, dst, 64)
        torch.cuda.synchronize()
        return dst

    def test_e4m3_in_triton_kernel_is_loud_on_ampere(self):
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 9):
            # Native fp8: the load must simply work and decode as e4m3.
            self.assertTrue(torch.all(self._compile_probe(torch.float8_e4m3fn) == 0.0))
            return
        with self.assertRaises(Exception) as ctx:
            self._compile_probe(torch.float8_e4m3fn)
        self.assertIn(
            "fp8",
            str(ctx.exception).lower(),
            f"expected a Triton fp8 architecture rejection on sm{capability[0]}"
            f"{capability[1]}, got: {ctx.exception}",
        )

    def test_e5m2_in_triton_kernel_compiles_everywhere(self):
        """e5m2 is in Triton's base NVIDIA fp8 set, so it compiles on Ampere too.

        Recorded because it makes the two formats behave differently on the
        Triton lane, which is the asymmetry a reader needs to know about.
        """
        self.assertTrue(torch.all(self._compile_probe(torch.float8_e5m2) == 0.0))


if __name__ == "__main__":
    unittest.main()
