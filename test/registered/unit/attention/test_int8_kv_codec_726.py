"""#726: reference int8 group-64 KV codec, as a correctness ORACLE.

WHAT THIS IS. A pure-torch transcription of the codec NInfer uses for INT8 KV
(ANALYSE_NINFER.md section 3.2, from `gqa_attention_kv_quant.cuh:19-55`):
symmetric, NO zero point, per-token per-64-element-group absmax, scale stored
as FP16, RNE rounding, clamp to [-127, 127].

WHAT IT IS FOR. Any IMMA-QK kernel must be validated against something, and
"validated against itself" is not a gate. This is the something. It is also
the only way to price the quantization error before a kernel exists, which the
mandatory quality gate needs.

WHAT IT IS NOT. Not a serving path, not an implementation, and deliberately
not placed under `python/sglang/srt/` — int8 KV is UNBUILT in this fork
(`server_args.py:1008` choices exclude it) and #489 stands as
evaluated-and-declined. A reference living beside production code would read
as a half-landed feature. If #726 is reopened and built, this moves next to
the kernel it validates.

THE CODEC, stated so the pins are readable:

    per group g of 64 elements:
        scale_g = fp16(absmax(x_g) / 127)
        q_g     = clamp(rne(x_g / scale_g), -127, 127)   as int8
        x'_g    = q_g * scale_g

-127 rather than -128 is deliberate and is NInfer's choice: a symmetric codec
that admits -128 has no positive counterpart for it, so the extra code point
buys nothing and costs the symmetry the dequant relies on.
"""

import unittest

import torch

GROUP = 64
QMAX = 127


def quantize_group64(x: torch.Tensor):
    """(codes int8, scales fp16) for the last dim, split into 64-groups.

    ``x`` is [..., D] with D a multiple of 64. Returns codes of the same shape
    and scales of [..., D // 64].
    """
    if x.shape[-1] % GROUP != 0:
        raise ValueError(
            f"last dim {x.shape[-1]} is not a multiple of the {GROUP}-element "
            f"group; the codec has no defined scale for a partial group"
        )
    grouped = x.reshape(*x.shape[:-1], x.shape[-1] // GROUP, GROUP).to(torch.float32)
    absmax = grouped.abs().amax(dim=-1)
    # FP16-STORED SCALE, and the round-trip through fp16 happens HERE rather
    # than at the end: the kernel reads an fp16 scale, so the reference must
    # quantize against the same value the kernel will use, not against a
    # float32 ideal it never sees.
    scales = (absmax / QMAX).to(torch.float16)
    safe = scales.to(torch.float32).clamp_min(torch.finfo(torch.float16).tiny)
    codes = torch.round(grouped / safe.unsqueeze(-1))
    codes = codes.clamp(-QMAX, QMAX).to(torch.int8)
    return codes.reshape(x.shape), scales


def dequantize_group64(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    grouped = codes.reshape(
        *codes.shape[:-1], codes.shape[-1] // GROUP, GROUP
    ).to(torch.float32)
    return (grouped * scales.to(torch.float32).unsqueeze(-1)).reshape(codes.shape)


def roundtrip(x: torch.Tensor) -> torch.Tensor:
    codes, scales = quantize_group64(x)
    return dequantize_group64(codes, scales)


def _sample(shape, seed=0):
    """CPU-sampled inputs. Never sample on device for a cross-arch oracle --
    the CUDA generator is not bit-comparable across architectures (the
    randn-cross-arch trap)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float32)


class TestTheCodecContract(unittest.TestCase):
    def test_codes_stay_in_the_symmetric_range(self):
        codes, _ = quantize_group64(_sample((4, 256)))
        self.assertGreaterEqual(int(codes.min()), -QMAX)
        self.assertLessEqual(int(codes.max()), QMAX)

    def test_minus_128_is_never_emitted(self):
        """int8 admits -128; this codec must not use it. A symmetric codec
        with no +128 counterpart gains nothing and loses the symmetry the
        dequant assumes."""
        x = _sample((32, 256), seed=3) * 100.0
        codes, _ = quantize_group64(x)
        self.assertEqual(int((codes == -128).sum()), 0)

    def test_one_scale_per_64_elements(self):
        codes, scales = quantize_group64(_sample((7, 256)))
        self.assertEqual(codes.shape, (7, 256))
        self.assertEqual(scales.shape, (7, 4), "head_dim 256 -> 4 groups")
        self.assertEqual(scales.dtype, torch.float16)

    def test_a_partial_group_is_refused_not_guessed(self):
        with self.assertRaises(ValueError):
            quantize_group64(_sample((2, 100)))

    def test_the_group_extreme_maps_to_the_end_of_the_range(self):
        """absmax/127 as the scale means the largest magnitude in each group
        lands on +-127. If it does not, the scale is not absmax-derived."""
        x = torch.zeros(1, GROUP)
        x[0, 5] = 3.5
        x[0, 9] = -3.5
        codes, _ = quantize_group64(x)
        self.assertEqual(int(codes[0, 5]), QMAX)
        self.assertEqual(int(codes[0, 9]), -QMAX)


class TestGroupsAreIndependent(unittest.TestCase):
    """The whole point of group-64 over per-tensor: one loud group must not
    crush the resolution of its neighbours."""

    def test_an_outlier_in_one_group_does_not_degrade_another(self):
        x = _sample((1, 128), seed=7)
        x[0, :GROUP] *= 1000.0  # group 0 becomes huge
        recon = roundtrip(x)
        quiet_err = (recon[0, GROUP:] - x[0, GROUP:]).abs().max().item()
        quiet_scale = x[0, GROUP:].abs().max().item()
        self.assertLess(
            quiet_err / quiet_scale,
            0.02,
            "the quiet group's relative error must not follow the loud one",
        )

    def test_scales_differ_when_group_magnitudes_differ(self):
        x = _sample((1, 128), seed=11)
        x[0, :GROUP] *= 100.0
        _, scales = quantize_group64(x)
        self.assertGreater(float(scales[0, 0]), float(scales[0, 1]) * 10)


class TestRoundTripErrorIsPriced(unittest.TestCase):
    """Numbers for the mandatory quality gate, measured rather than assumed.

    These are LOOSE bounds on purpose: they exist to catch a codec that has
    silently become much worse, not to certify model quality. Certifying
    quality is the gate's job and needs a model, not a tensor.
    """

    def test_relative_error_on_normal_data(self):
        x = _sample((256, 256), seed=1)
        err = (roundtrip(x) - x).abs()
        rel = err.max().item() / x.abs().max().item()
        self.assertLess(rel, 0.02, f"max relative error {rel:.4f}")

    def test_rms_error_is_well_under_a_percent(self):
        x = _sample((256, 256), seed=2)
        recon = roundtrip(x)
        rms = ((recon - x) ** 2).mean().sqrt().item()
        rms_ref = (x**2).mean().sqrt().item()
        self.assertLess(rms / rms_ref, 0.01, f"relative RMS {rms / rms_ref:.5f}")

    def test_the_codec_is_deterministic(self):
        """Bit-identical on repeat. A kernel compared against a
        non-deterministic oracle proves nothing."""
        x = _sample((16, 256), seed=5)
        a, sa = quantize_group64(x)
        b, sb = quantize_group64(x)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(sa, sb))

    def test_zeros_survive_exactly(self):
        """An all-zero group has absmax 0. The codec must not divide by it
        and must reproduce zeros exactly."""
        recon = roundtrip(torch.zeros(2, GROUP))
        self.assertTrue(torch.equal(recon, torch.zeros(2, GROUP)))


class TestTheFootprintClaim(unittest.TestCase):
    """The 1.94x that motivates the whole ticket, as arithmetic.

    NInfer's measured figure (ANALYSE_NINFER section 3.2): 264 B per token per
    KV head per plane against 512 B in BF16, i.e. the full 2x MINUS the scale
    overhead. Pinned so a future layout change that quietly doubles the scale
    cost cannot keep claiming 1.94x.
    """

    def test_bytes_per_token_per_head_per_plane(self):
        head_dim = 256
        groups = head_dim // GROUP
        int8_bytes = head_dim * 1 + groups * 2  # codes + fp16 scales
        bf16_bytes = head_dim * 2
        self.assertEqual(int8_bytes, 264)
        self.assertEqual(bf16_bytes, 512)
        self.assertAlmostEqual(bf16_bytes / int8_bytes, 1.939, places=3)

    def test_the_scale_overhead_is_what_costs_the_last_0_06x(self):
        head_dim = 256
        without_scales = (head_dim * 2) / (head_dim * 1)
        with_scales = (head_dim * 2) / (head_dim + (head_dim // GROUP) * 2)
        self.assertEqual(without_scales, 2.0)
        self.assertLess(with_scales, without_scales)


if __name__ == "__main__":
    unittest.main()


class TestTheDtypeKeyAlreadySeparatesFormats(unittest.TestCase):
    """#241/#513 key-completeness, checked rather than assumed.

    The brief expected the HiCache key to need teaching about a new KV dtype.
    It does not: ``compute_model_identity_hash`` already folds
    ``kv_cache_dtype`` into the identity (hicache_storage.py:59), for exactly
    this reason -- its own docstring says a later run differing in
    ``--kv-cache-dtype`` "would silently read pages written in another byte
    format", and the hash "turns that silent wrong hit into a clean miss".

    So adding an int8 value to the dtype choices separates its pages
    automatically. This pin holds that property so a future refactor of the
    identity recipe cannot quietly drop the field and reopen the collision.
    """

    class _Args:
        model_path = "/models/qwen"
        revision = None
        dtype = "bfloat16"
        quantization = "compressed-tensors"
        kv_cache_dtype = "fp8_e4m3"

    def _hash(self, kv_dtype):
        from sglang.srt.mem_cache.hicache_storage import compute_model_identity_hash

        args = TestTheDtypeKeyAlreadySeparatesFormats._Args()
        args.kv_cache_dtype = kv_dtype
        return compute_model_identity_hash(args, include_parallel_vectors=False)

    def test_int8_pages_cannot_collide_with_fp8_pages(self):
        self.assertNotEqual(
            self._hash("int8"),
            self._hash("fp8_e4m3"),
            "an int8-KV boot must not read fp8-written pages",
        )

    def test_every_pair_of_kv_dtypes_is_separated(self):
        dtypes = ["auto", "fp8_e5m2", "fp8_e4m3", "bf16", "fp4_e2m1", "int8"]
        hashes = {d: self._hash(d) for d in dtypes}
        self.assertEqual(
            len(set(hashes.values())),
            len(dtypes),
            f"two KV dtypes share an identity hash: {hashes}",
        )

    def test_the_field_is_actually_in_the_recipe(self):
        """Behavioural pins above would also pass if the hash were salted by
        something incidental. This states WHERE the separation comes from."""
        import inspect

        from sglang.srt.mem_cache import hicache_storage

        src = inspect.getsource(hicache_storage.compute_model_identity_hash)
        self.assertIn("server_args.kv_cache_dtype", src)
