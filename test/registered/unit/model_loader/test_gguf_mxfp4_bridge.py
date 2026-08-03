# SPDX-License-Identifier: Apache-2.0
"""#391 blocker 1: is the GGUF MXFP4 block bridgeable to an executable path?

The published DeepSeek V4 Flash GGUF stores every routed ``down`` projection as
GGML type 39 (MXFP4), 47.8 GiB of the 119.4 GiB file. sglang has no MXFP4 GGUF
kernel. This module answers, executably, the only question that decides which
repair is cheapest: *what exactly is in an MXFP4 block, and which existing
contracts can consume it without loss?*

Three contracts are pinned here, and each is checked value-exactly (fp32, zero
tolerance) against ``gguf.quants`` -- the same reference implementation
llama.cpp's C code is kept in sync with:

1. **Block geometry.** 32 values in 17 bytes: byte 0 is an e8m0 exponent, bytes
   1..16 hold the 4-bit codes in llama.cpp's *split-half* order (element ``j``
   is the low nibble of byte ``j``, element ``j + 16`` the high nibble) -- the
   same interleave Q4_0/Q5_0/IQ4_NL use, NOT adjacent pairs.
2. **Bridge A -- the OCP / safetensors mxfp4 contract.** Re-interleaving the
   nibbles into adjacent pairs and lifting byte 0 out into a separate scale
   tensor produces exactly the ``(packed uint8, e8m0 uint8)`` pair that
   ``Mxfp4MoEMethod`` registers as ``w2_weight`` / ``w2_weight_scale``. The two
   formats carry the same lattice: llama.cpp doubles its code table
   (0,1,2,3,4,6,8,12) and halves the scale (``2**(e-128)``), OCP keeps the E2M1
   values (0,.5,1,1.5,2,3,4,6) and the full scale (``2**(e-127)``). Identical
   products. So the LAYOUT bridges.
3. **Bridge B -- an exact GGUF-side repack.** The MXFP4 lattice is a subset of
   Q5_0's, because ``2 * E2M1`` is a set of integers within Q5_0's ``[-16, 15]``
   code range. MXFP4 -> Q5_0 is therefore lossless, and Q5_0 is a type the GGUF
   dequant/MMVQ/MMQ kernels already dispatch on. The price is bytes: 22 per
   block instead of 17.

What the layout bridging does NOT establish is that the safetensors mxfp4
expert *kernels* can run this checkpoint -- they cannot, for reasons that are
about the layer contract rather than the bytes (``Mxfp4MoEMethod`` needs BOTH
projections in mxfp4, and this export pairs MXFP4 ``down`` with IQ3_XXS
``gate``/``up``). Bridge B is the one that shipped: ``gguf_mxfp4_repack``
performs it inside the weight stream, and the tests here call that module
rather than a copy of it.

The last two tests pinned the premise the shipped repair rested on -- MXFP4
absent from every executable GGUF type set, Q5_0 present in all of them. #398
built the MXFP4 kernels, so the first is now a CONDITIONAL: membership must
track the build's own capability flag. On a wheel with the kernels the repack
is optional (and off); on one without it, it is still load-bearing.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch
from gguf.constants import GGMLQuantizationType as GGMLType
from gguf.quants import dequantize, quantize

from sglang.srt.layers.quantization.mxfp4_tensor import MXFP4QuantizeUtil
from sglang.srt.model_loader.gguf_mxfp4_repack import repacked_gguf_bytes
from sglang.test.gguf_mxfp4_state import ForcesRepackPath

#: llama.cpp's ``kvalues_mxfp4`` (ggml-common.h): the E2M1 lattice, doubled so
#: it fits in int8. Indexed by the raw 4-bit code.
GGUF_MXFP4_KVALUES = np.array(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.int8
)

MXFP4_BLOCK_SIZE = 32
MXFP4_TYPE_SIZE = 17
Q5_0_TYPE_SIZE = 22


def _sample_blocks(seed: int = 0x391) -> np.ndarray:
    """Random rows plus the degenerate blocks a bridge is most likely to get
    wrong: an all-zero block, and blocks saturating both signs."""
    rng = np.random.default_rng(seed)
    rows, cols = 96, 256
    magnitudes = rng.choice([1e-3, 1.0, 50.0], size=(rows, 1)).astype(np.float32)
    x = rng.standard_normal((rows, cols)).astype(np.float32) * magnitudes
    x[0] = 0.0
    x[1, :MXFP4_BLOCK_SIZE] = 6.0
    x[2, :MXFP4_BLOCK_SIZE] = -6.0
    return x


def unpack_gguf_mxfp4(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(n, 17)`` raw MXFP4 blocks -> ``(e8m0 (n,), codes (n, 32))``.

    ``codes`` is in ELEMENT order; the split-half interleave is undone here and
    nowhere else, so every consumer below shares one decode.
    """
    e8m0 = blocks[:, 0]
    qs = blocks[:, 1:]
    low = qs & np.uint8(0x0F)
    high = (qs >> np.uint8(4)) & np.uint8(0x0F)
    return e8m0, np.concatenate([low, high], axis=-1)


def gguf_mxfp4_to_ocp(blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bridge A: GGUF MXFP4 blocks -> the safetensors mxfp4 parameter pair.

    Returns ``(packed, scale)`` where ``packed`` holds adjacent element pairs
    (element ``2j`` low nibble, ``2j + 1`` high nibble) as ``Mxfp4MoEMethod``'s
    ``*_weight`` expects, and ``scale`` is the untouched e8m0 byte as its
    ``*_weight_scale`` expects. Pure permutation: no value is recomputed.
    """
    e8m0, codes = unpack_gguf_mxfp4(blocks)
    packed = (codes[:, 0::2] & np.uint8(0x0F)) | (codes[:, 1::2] << np.uint8(4))
    return packed, e8m0


def gguf_mxfp4_to_q5_0(blocks: np.ndarray) -> np.ndarray:
    """Bridge B: GGUF MXFP4 blocks -> GGUF Q5_0 blocks, value-exact.

    This is the shipped repack (``model_loader/gguf_mxfp4_repack``) under a
    block-shaped call, not a second implementation: the analysis and the loader
    must not be able to drift apart. ``(n, 17)`` in, ``(n, 22)`` out, one block
    per row.
    """
    return repacked_gguf_bytes(GGMLType.MXFP4, blocks, "test_bridge")


class TestGGUFMXFP4BlockGeometry(unittest.TestCase):
    """Contract 1: what gguf-py actually writes."""

    def test_block_size_and_type_size(self):
        from gguf.constants import GGML_QUANT_SIZES

        self.assertEqual(
            GGML_QUANT_SIZES[GGMLType.MXFP4],
            (MXFP4_BLOCK_SIZE, MXFP4_TYPE_SIZE),
        )
        self.assertEqual(int(GGMLType.MXFP4), 39)

    def test_nibbles_are_split_half_not_adjacent_pairs(self):
        """Element j is byte j's low nibble, element j+16 its high nibble."""
        x = _sample_blocks()
        raw = quantize(x, GGMLType.MXFP4).reshape(-1, MXFP4_TYPE_SIZE)
        reference = dequantize(quantize(x, GGMLType.MXFP4), GGMLType.MXFP4).reshape(
            -1, MXFP4_BLOCK_SIZE
        )

        e8m0, codes = unpack_gguf_mxfp4(raw)
        scale = np.exp2(e8m0.astype(np.float32) - 128.0)
        rebuilt = GGUF_MXFP4_KVALUES[codes].astype(np.float32) * scale[:, None]
        np.testing.assert_array_equal(rebuilt, reference)

        # The adjacent-pair reading of the same bytes must NOT reproduce the
        # reference -- otherwise this test could not tell the orders apart.
        adjacent = np.empty_like(codes)
        adjacent[:, 0::2] = raw[:, 1:] & np.uint8(0x0F)
        adjacent[:, 1::2] = (raw[:, 1:] >> np.uint8(4)) & np.uint8(0x0F)
        wrong = GGUF_MXFP4_KVALUES[adjacent].astype(np.float32) * scale[:, None]
        self.assertFalse(np.array_equal(wrong, reference))

    def test_lattice_is_doubled_e2m1(self):
        """Every stored value is a doubled-E2M1 code times ``2**(e - 128)``."""
        x = _sample_blocks()
        raw = quantize(x, GGMLType.MXFP4).reshape(-1, MXFP4_TYPE_SIZE)
        e8m0, _ = unpack_gguf_mxfp4(raw)
        values = dequantize(quantize(x, GGMLType.MXFP4), GGMLType.MXFP4).reshape(
            -1, MXFP4_BLOCK_SIZE
        )
        scale = np.exp2(e8m0.astype(np.float32) - 128.0)
        ratios = np.unique(values / scale[:, None])
        np.testing.assert_array_equal(
            np.sort(ratios), np.unique(GGUF_MXFP4_KVALUES).astype(np.float32)
        )


class TestBridgeToSafetensorsMXFP4(unittest.TestCase):
    """Contract 2: the layout bridges to what ``Mxfp4MoEMethod`` registers."""

    def test_repack_is_bit_exact_under_the_ocp_dequantizer(self):
        x = _sample_blocks()
        packed_file = quantize(x, GGMLType.MXFP4)
        reference = dequantize(packed_file, GGMLType.MXFP4)

        packed, scale = gguf_mxfp4_to_ocp(packed_file.reshape(-1, MXFP4_TYPE_SIZE))
        got = MXFP4QuantizeUtil.dequantize(
            torch.from_numpy(packed),
            torch.float32,
            torch.from_numpy(scale.reshape(-1, 1)),
            [MXFP4_BLOCK_SIZE],
        ).numpy()

        np.testing.assert_array_equal(got.reshape(reference.shape), reference)

    def test_bridged_shapes_match_the_moe_parameter_contract(self):
        """``w2_weight`` is ``[.., K // 2]`` uint8 and ``w2_weight_scale`` is
        ``[.., K // 32]`` uint8; the bridge output must slot straight in."""
        rows, cols = 8, 256
        x = np.zeros((rows, cols), dtype=np.float32)
        packed, scale = gguf_mxfp4_to_ocp(
            quantize(x, GGMLType.MXFP4).reshape(-1, MXFP4_TYPE_SIZE)
        )
        self.assertEqual(packed.reshape(rows, -1).shape, (rows, cols // 2))
        self.assertEqual(
            scale.reshape(rows, -1).shape, (rows, cols // MXFP4_BLOCK_SIZE)
        )
        self.assertEqual(packed.dtype, np.uint8)
        self.assertEqual(scale.dtype, np.uint8)


class TestBridgeToQ5_0(ForcesRepackPath, unittest.TestCase):
    """Contract 3: MXFP4 is exactly representable in a supported GGUF type.

    The repack path is forced (#529): on a post-#398 wheel
    ``repacked_gguf_bytes`` is the identity, which turned this contract into
    three assertions about 17-byte payloads that were never converted. The
    representability claim is a property of the two LATTICES, not of which
    kernel the wheel happens to carry, so it is pinned where the conversion
    actually runs.
    """

    def test_repack_is_value_exact(self):
        x = _sample_blocks()
        packed_file = quantize(x, GGMLType.MXFP4)
        reference = dequantize(packed_file, GGMLType.MXFP4)

        q5 = gguf_mxfp4_to_q5_0(packed_file.reshape(-1, MXFP4_TYPE_SIZE))
        got = dequantize(q5.reshape(x.shape[0], -1), GGMLType.Q5_0)

        np.testing.assert_array_equal(got, reference)

    def test_byte_cost_of_the_repack(self):
        x = _sample_blocks()
        packed_file = quantize(x, GGMLType.MXFP4)
        q5 = gguf_mxfp4_to_q5_0(packed_file.reshape(-1, MXFP4_TYPE_SIZE))
        self.assertEqual(q5.shape[-1], Q5_0_TYPE_SIZE)
        self.assertEqual(
            q5.nbytes * MXFP4_TYPE_SIZE, packed_file.nbytes * Q5_0_TYPE_SIZE
        )

    def test_unrepresentable_scale_is_refused(self):
        """A gate that has never failed is not known to be a gate."""
        block = np.zeros((1, MXFP4_TYPE_SIZE), dtype=np.uint8)
        block[0, 0] = 250  # 2**122, far outside fp16
        block[0, 1] = 0x07  # one non-zero code, so the all-zero shortcut misses
        with self.assertRaises(ValueError):
            gguf_mxfp4_to_q5_0(block)


class TestMXFP4IsStillUnexecutable(unittest.TestCase):
    """When the repack is load-bearing, and when it is not.

    #398 built the MXFP4 kernels, so the #391 premise ("no executable MXFP4
    GGUF path") now holds only on a wheel that predates them -- or with
    ``SGLANG_GGUF_MXFP4_NATIVE=0``. The assertion therefore follows the
    build's own capability flag instead of being a constant. Both states are
    exercised in-process by
    ``test/registered/unit/quantization/test_gguf_mxfp4_native.py``
    (``TestDispatchFlip``); here the point is only that the repack stays
    correct and load-bearing exactly while the kernels are absent.
    """

    def test_mxfp4_type_set_membership_follows_the_kernel_capability(self):
        from sglang.srt.layers.quantization.gguf import (
            DEQUANT_TYPES,
            MMQ_QUANT_TYPES,
            MMVQ_QUANT_TYPES,
            MXFP4_NATIVE,
        )

        mxfp4 = int(GGMLType.MXFP4)
        for name, type_set in (
            ("DEQUANT_TYPES", DEQUANT_TYPES),
            ("MMVQ_QUANT_TYPES", MMVQ_QUANT_TYPES),
            ("MMQ_QUANT_TYPES", MMQ_QUANT_TYPES),
        ):
            present = mxfp4 in {int(t) for t in type_set}
            self.assertEqual(
                present,
                MXFP4_NATIVE,
                f"{name}: MXFP4 present={present} but the build reports "
                f"native MXFP4 kernels={MXFP4_NATIVE}. These must agree -- "
                "either the #398 kernels are in this wheel and the type is "
                "executable, or they are not and the Q5_0 repack carries it.",
            )

    def test_q5_0_is_executable_everywhere_mxfp4_would_need_to_be(self):
        from sglang.srt.layers.quantization.gguf import (
            DEQUANT_TYPES,
            MMQ_QUANT_TYPES,
            MMVQ_QUANT_TYPES,
        )

        q5_0 = int(GGMLType.Q5_0)
        for type_set in (DEQUANT_TYPES, MMVQ_QUANT_TYPES, MMQ_QUANT_TYPES):
            self.assertIn(q5_0, {int(t) for t in type_set})


if __name__ == "__main__":
    unittest.main()
