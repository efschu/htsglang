# SPDX-License-Identifier: Apache-2.0
"""#398: the native GGUF MXFP4 (ggml type 39) kernel path.

Four things are pinned here, all hermetic (no GPU, no wheel rebuild needed):

1. **The block layout the CUDA kernels assume**, as a python host reference.
   :func:`mxfp4_host_dequant` is the decoder every other test in the #398 set
   compares against -- including the CUDA numerical tests, which cannot run in
   this process. It was validated against a REAL tensor slice out of the
   DeepSeek V4 Flash UD-IQ3_XXS export (``blk.26.ffn_down_exps.weight``, ggml
   type 39, 2048 elements = 64 blocks x 17 bytes per row) by hand-decoding
   blocks 0, 5 and 63 byte by byte and comparing with ``gguf.quants.MXFP4``;
   two of those real rows are the fixture below so the check does not need the
   50 GiB shard.

2. **The dispatch flip (the falsifier).** Before #398 MXFP4 is in no GGUF type
   set and the linear path raises; after it, on a wheel carrying the kernels,
   it is in all three. The switch is one predicate -- existence of the
   ``ggml_mxfp4_native`` op -- so both states can be produced in-process by
   registering/not registering that op and reloading the module. Red and green
   are therefore both EXECUTED here, not asserted about a build elsewhere.

3. **The repack hand-off.** ``gguf_mxfp4_repack`` must become a no-op exactly
   when the kernels are native, and must still work when they are not.

4. **The shard-boundary arithmetic** an uneven-TP split has to satisfy for a
   32-element block type (#109/#385 family).
"""

from __future__ import annotations

import importlib
import os
import unittest
from pathlib import Path

import numpy as np
import torch
from gguf.constants import GGMLQuantizationType as GGMLType

#: llama.cpp's ``kvalues_mxfp4`` (ggml-common.h): the E2M1 lattice doubled so
#: it fits in int8, indexed by the raw 4-bit code. Mirrored in the CUDA kernels
#: as ``kvalues_mxfp4`` (sgl-kernel/csrc/quantization/gguf/ggml-common.h).
MXFP4_KVALUES = np.array(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.int8
)

BLOCK_SIZE = 32
TYPE_SIZE = 17

#: Two real MXFP4 rows (2 x 1088 bytes = 2 x 64 blocks) copied out of
#: ``blk.26.ffn_down_exps.weight`` of
#: ``DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00003-of-00004.gguf``, expert 0, rows
#: 0..1. Kept as a file next to this test so the layout claim rests on shipped
#: bytes rather than on synthetic data that could encode the same mistake.
_FIXTURE = Path(__file__).with_name("mxfp4_real_rows.npy")


def e8m0_to_fp32_half(e: np.ndarray) -> np.ndarray:
    """``2**(e - 128)``: the OCP scale ``2**(e-127)`` already halved against
    the doubled lattice above.

    ``e < 2`` lands in the fp32 subnormal range and is built by shifting the
    subnormal bit pattern instead of writing the exponent field -- the exact
    arithmetic ``ggml_cuda_e8m0_to_fp32_half`` performs on device.
    """
    e = np.asarray(e, dtype=np.uint8)
    bits = np.where(
        e < 2,
        np.uint32(0x00200000) << e.astype(np.uint32),
        (e.astype(np.uint32) - np.uint32(1)) << np.uint32(23),
    ).astype(np.uint32)
    return bits.view(np.float32)


def mxfp4_host_dequant(blocks: np.ndarray) -> np.ndarray:
    """``(n, 17)`` MXFP4 blocks -> ``(n, 32)`` fp32 values.

    Split-half nibble order: element ``j`` is the LOW nibble of byte ``1 + j``,
    element ``j + 16`` the HIGH nibble of the same byte.
    """
    blocks = np.ascontiguousarray(blocks).reshape(-1, TYPE_SIZE)
    d = e8m0_to_fp32_half(blocks[:, 0]).reshape(-1, 1)
    packed = blocks[:, 1:]
    codes = np.concatenate(
        [packed & np.uint8(0x0F), (packed >> np.uint8(4)) & np.uint8(0x0F)], axis=-1
    )
    return d * MXFP4_KVALUES[codes].astype(np.float32)


def synthetic_blocks(n: int, seed: int = 398) -> np.ndarray:
    """``n`` MXFP4 blocks covering all 16 codes and a spread of exponents."""
    rng = np.random.default_rng(seed)
    e = rng.integers(110, 136, size=(n, 1)).astype(np.uint8)
    qs = rng.integers(0, 256, size=(n, TYPE_SIZE - 1)).astype(np.uint8)
    return np.concatenate([e, qs], axis=-1)


# ---------------------------------------------------------------------------
# 1. Block layout / host reference
# ---------------------------------------------------------------------------
class TestMXFP4HostReference(unittest.TestCase):
    def test_block_geometry_matches_the_ggml_constants(self):
        block_size, type_size = __import__("gguf").GGML_QUANT_SIZES[GGMLType.MXFP4]
        self.assertEqual((block_size, type_size), (BLOCK_SIZE, TYPE_SIZE))
        # The 17-byte stride is why the kernels must use get_int_b1: `qs`
        # starts at an odd offset in every other block.
        self.assertEqual(TYPE_SIZE % 2, 1)

    def test_host_reference_matches_gguf_quants_on_synthetic_blocks(self):
        from gguf.quants import dequantize

        blocks = synthetic_blocks(512)
        got = mxfp4_host_dequant(blocks)
        want = dequantize(blocks.reshape(-1), GGMLType.MXFP4).reshape(-1, BLOCK_SIZE)
        np.testing.assert_array_equal(got, want)

    def test_host_reference_matches_gguf_quants_on_real_tensor_bytes(self):
        rows = np.load(_FIXTURE)
        self.assertEqual(rows.shape, (2, 64 * TYPE_SIZE))
        from gguf.quants import dequantize

        got = mxfp4_host_dequant(rows.reshape(-1, TYPE_SIZE))
        want = dequantize(rows.reshape(-1), GGMLType.MXFP4).reshape(-1, BLOCK_SIZE)
        np.testing.assert_array_equal(got, want)

    def test_hand_decoded_block_zero_of_the_real_tensor(self):
        """The ground truth, spelled out: no library on either side."""
        import struct

        blk = np.load(_FIXTURE)[0, :TYPE_SIZE]
        e = int(blk[0])
        self.assertEqual(e, 120)  # 2**(120-128) = 2**-8 = 0.00390625
        d = struct.unpack("<f", struct.pack("<I", (e - 1) << 23))[0]
        self.assertEqual(d, 0.00390625)
        kv = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]
        by_hand = [0.0] * 32
        for j in range(16):
            b = int(blk[1 + j])
            by_hand[j] = d * kv[b & 0x0F]
            by_hand[j + 16] = d * kv[b >> 4]
        np.testing.assert_array_equal(
            np.array(by_hand, dtype=np.float32), mxfp4_host_dequant(blk)[0]
        )

    def test_all_sixteen_codes_are_exercised_by_the_fixture(self):
        packed = np.load(_FIXTURE).reshape(-1, TYPE_SIZE)[:, 1:]
        seen = set((packed & 0x0F).ravel().tolist()) | set(
            (packed >> 4).ravel().tolist()
        )
        self.assertEqual(seen, set(range(16)))

    def test_subnormal_exponents_round_trip(self):
        """e = 0 and e = 1 are the two scales that are NOT a plain exponent
        write; the kernel builds them by shifting the subnormal pattern."""
        self.assertEqual(float(e8m0_to_fp32_half(np.uint8(0))), 2.0**-128)
        self.assertEqual(float(e8m0_to_fp32_half(np.uint8(1))), 2.0**-127)
        self.assertEqual(float(e8m0_to_fp32_half(np.uint8(2))), 2.0**-126)
        self.assertEqual(float(e8m0_to_fp32_half(np.uint8(128))), 1.0)


# ---------------------------------------------------------------------------
# 2. The dispatch flip -- both states executed
# ---------------------------------------------------------------------------
def _reload_gguf():
    import sglang.srt.layers.quantization.gguf as g

    return importlib.reload(g)


class _FakeNativeOp:
    """Registers ``sgl_kernel::ggml_mxfp4_native`` for the duration of a block.

    The real op comes from the wheel; the probe only asks whether it EXISTS,
    so registering an equivalent schema here reproduces the post-#398 wheel
    faithfully for everything the python dispatch decides.
    """

    def __enter__(self):
        self._lib = None
        if hasattr(torch.ops.sgl_kernel, "ggml_mxfp4_native"):
            return self  # real wheel already has it
        self._lib = torch.library.Library("sgl_kernel", "FRAGMENT")
        self._lib.define("ggml_mxfp4_native() -> int")
        self._lib.impl("ggml_mxfp4_native", lambda: 1, "CUDA")
        return self

    def __exit__(self, *exc):
        if self._lib is not None:
            self._lib._destroy()
            torch.ops.sgl_kernel  # keep the namespace object alive
        return False


class TestDispatchFlip(unittest.TestCase):
    """The #398 falsifier: the same code, one predicate apart, in both states."""

    def setUp(self):
        self._env = os.environ.get("SGLANG_GGUF_MXFP4_NATIVE")

    def tearDown(self):
        if self._env is None:
            os.environ.pop("SGLANG_GGUF_MXFP4_NATIVE", None)
        else:
            os.environ["SGLANG_GGUF_MXFP4_NATIVE"] = self._env
        _reload_gguf()

    def test_red_mxfp4_is_unexecutable_without_the_kernels(self):
        os.environ["SGLANG_GGUF_MXFP4_NATIVE"] = "0"
        g = _reload_gguf()
        self.assertFalse(g.MXFP4_NATIVE)
        mxfp4 = int(GGMLType.MXFP4)
        for name in ("DEQUANT_TYPES", "MMVQ_QUANT_TYPES", "MMQ_QUANT_TYPES"):
            self.assertNotIn(mxfp4, {int(t) for t in getattr(g, name)}, name)
        self.assertNotIn(39, g._GGML_MOE_MMQ_TYPES)
        # ...and the linear path refuses it by name rather than reaching a
        # kernel that would be a null function pointer.
        with self.assertRaises(NotImplementedError):
            g.apply_gguf_embedding(
                torch.zeros(1, dtype=torch.long),
                torch.zeros(1, TYPE_SIZE, dtype=torch.uint8),
                mxfp4,
                BLOCK_SIZE,
            )

    def test_green_mxfp4_is_executable_with_the_kernels(self):
        os.environ["SGLANG_GGUF_MXFP4_NATIVE"] = "1"
        with _FakeNativeOp():
            g = _reload_gguf()
            self.assertTrue(g.MXFP4_NATIVE)
            mxfp4 = int(GGMLType.MXFP4)
            for name in ("DEQUANT_TYPES", "MMVQ_QUANT_TYPES", "MMQ_QUANT_TYPES"):
                self.assertIn(mxfp4, {int(t) for t in getattr(g, name)}, name)
            self.assertIn(39, g._GGML_MOE_MMQ_TYPES)
            # MoE expert offload coverage follows MMVQ membership.
            self.assertTrue(g.gguf_moe_offload_covered_type(mxfp4))

    def test_env_kill_switch_beats_a_present_op(self):
        with _FakeNativeOp():
            os.environ["SGLANG_GGUF_MXFP4_NATIVE"] = "0"
            g = _reload_gguf()
            self.assertFalse(g.MXFP4_NATIVE)


# ---------------------------------------------------------------------------
# 3. The repack hand-off
# ---------------------------------------------------------------------------
class TestRepackHandoff(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.get("SGLANG_GGUF_MXFP4_NATIVE")

    def tearDown(self):
        if self._env is None:
            os.environ.pop("SGLANG_GGUF_MXFP4_NATIVE", None)
        else:
            os.environ["SGLANG_GGUF_MXFP4_NATIVE"] = self._env
        _reload_gguf()

    def test_repack_still_converts_when_the_kernels_are_absent(self):
        os.environ["SGLANG_GGUF_MXFP4_NATIVE"] = "0"
        _reload_gguf()
        from sglang.srt.model_loader import gguf_mxfp4_repack as r

        importlib.reload(r)
        self.assertFalse(r.native_mxfp4_kernels())
        blocks = synthetic_blocks(8).reshape(1, -1)
        out = r.repacked_gguf_bytes(GGMLType.MXFP4, blocks, "t")
        self.assertEqual(out.shape, (1, 8 * 22))
        self.assertEqual(r.repacked_gguf_type(GGMLType.MXFP4, "t"), GGMLType.Q5_0)

    def test_repack_is_a_no_op_when_the_kernels_are_native(self):
        os.environ["SGLANG_GGUF_MXFP4_NATIVE"] = "1"
        with _FakeNativeOp():
            _reload_gguf()
            from sglang.srt.model_loader import gguf_mxfp4_repack as r

            importlib.reload(r)
            self.assertTrue(r.native_mxfp4_kernels())
            blocks = synthetic_blocks(8).reshape(1, -1)
            out = r.repacked_gguf_bytes(GGMLType.MXFP4, blocks, "t")
            # identity: same bytes, same type, no 22/17 growth
            np.testing.assert_array_equal(out, blocks)
            self.assertEqual(r.repacked_gguf_type(GGMLType.MXFP4, "t"), GGMLType.MXFP4)
            self.assertEqual(r.repack_source_types(), set())

    def test_the_saved_bytes_are_the_22_over_17_the_repack_would_have_added(self):
        """The #398 prize, as arithmetic on the shipped file's own numbers."""
        # blk.26 + blk.42 ffn_down_exps of the UD-IQ3_XXS export: 2 tensors of
        # 1 140 850 688 B each (2048 x 4096 x 256 elements at 17 B / 32 values).
        n_bytes = 2 * 1140850688
        self.assertEqual(n_bytes / (1 << 30), 2.125)
        grown = n_bytes * 22 / 17
        self.assertAlmostEqual((grown - n_bytes) / (1 << 30), 0.625, places=6)


# ---------------------------------------------------------------------------
# 4. Shard boundaries: 32-element blocks vs 16-element MLP units
# ---------------------------------------------------------------------------
class TestUnevenTPShardBoundaries(unittest.TestCase):
    """#109/#385 family. A GGUF quantized dimension may only be cut on ggml
    BLOCK boundaries; the uneven-TP MLP family is element-granular before
    coarsening (16-element units after the #82 fix). The two granularities
    agree only at their lcm, so a 32-element block type needs the coupled-dim
    rule to be at least 32 -- which GGUFConfig already over-satisfies at 256."""

    def test_lcm_of_the_two_granularities(self):
        import math

        self.assertEqual(math.lcm(16, BLOCK_SIZE), 32)

    def test_gguf_block_alignment_covers_the_32_element_mxfp4_block(self):
        """The coarsening GGUF installs must be a multiple of 32, or an MXFP4
        shard boundary can land inside a block."""
        from sglang.srt.layers.quantization.gguf import GGUFConfig

        cfg = GGUFConfig()
        for axis, block in enumerate(cfg.weight_block_size):
            self.assertEqual(block % BLOCK_SIZE, 0, f"axis {axis}: {block}")
        # ...and also of the 256-element super-block the dequant launcher
        # rounds up to, which is what makes the guard's tail always empty for
        # a real GGUF shard.
        self.assertEqual(cfg.weight_block_size[1] % 256, 0)

    def test_real_partitioner_emits_block_aligned_shards(self):
        """Through the SHIPPED code path: _quant_block_aligned_units coarsens
        the element-granular unit family, partition_sizes cuts it."""
        from sglang.srt.distributed.utils import partition_sizes
        from sglang.srt.layers.linear import _quant_block_aligned_units
        from sglang.srt.layers.quantization.gguf import GGUFConfig

        cfg = GGUFConfig()
        # DSV4F ffn_down quantized dim (2048) and two other real MLP widths,
        # split by the rig's measured capacity vectors.
        for total in (2048, 4096, 7168):
            # element-granular unit family, as a fine-grained layer declares it
            units = _quant_block_aligned_units(total, total, cfg, 1)
            self.assertIsNotNone(units)
            for weights in ([9, 5, 5], [2, 1, 1], [1, 1, 1], [7, 3, 3]):
                parts = partition_sizes(total, weights, units)
                self.assertEqual(sum(parts), total, (total, weights))
                for p in parts:
                    self.assertGreater(p, 0)
                    self.assertEqual(p % BLOCK_SIZE, 0, (total, weights, parts))

    def test_dequant_launcher_guard_covers_a_non_256_multiple(self):
        """``dequantize_row_mxfp4_cuda`` rounds the launch up to whole
        256-element super-blocks and guards per 32-element block, so a shard
        that is a multiple of 32 but not of 256 is safe. This pins the
        arithmetic the guard implements; the device-side check is in
        ``test_gguf_mxfp4_cuda.py`` (GPU-pending)."""
        for k in (32, 96, 704, 2048, 2016):
            self.assertEqual(k % BLOCK_SIZE, 0)
            nblocks = k // BLOCK_SIZE
            launched_blocks = ((k + 255) // 256) * 8
            self.assertGreaterEqual(launched_blocks, nblocks)
            # the guard discards exactly the over-launched tail
            self.assertLess(launched_blocks - nblocks, 8)


if __name__ == "__main__":
    unittest.main()
