# SPDX-License-Identifier: Apache-2.0
"""#391 blocker 1: the shipped MXFP4 -> Q5_0 load-time repack.

``test_gguf_mxfp4_bridge.py`` establishes that the conversion is possible.
This file pins the thing that actually loads a model:

* the contract is **dequant equality**, never byte arithmetic. Every value test
  compares ``gguf.quants.dequantize`` of the repacked Q5_0 payload against
  ``gguf.quants.dequantize`` of the original MXFP4 payload, in fp32 with zero
  tolerance. A future rewrite of the packing is free to produce different bytes
  as long as the values survive;
* it must hold on the REAL file, not only on synthetic blocks -- one block out
  of ``blk.0.ffn_down_exps`` and one out of ``blk.26.ffn_gate_exps`` (the only
  layer whose ``gate``/``up`` are MXFP4 too, and the K = 4096 case beside the
  K = 2048 down projections);
* it must survive the per-expert split. The stacked ``ffn_*_exps`` tensors are
  cut into 256 per-expert tensors by ``gguf_quant_weights_iterator``; a Q5_0
  block is self-contained, so repacking a slice and slicing a repack have to
  agree byte for byte;
* the gates must be able to fail. An e8m0 exponent fp16 cannot hold is refused
  by name, and ``SGLANG_GGUF_MXFP4_REPACK=0`` restores the refusal instead of
  quietly loading something unexecutable.

The end-to-end test writes a small GGUF containing MXFP4 tensors and runs the
real iterator over it, which is the only way to check that ALL consumers --
dense linear, stacked experts, embedding -- see Q5_0 and that no ``qweight_type``
marker still says 39.

WHICH PATH THIS FILE PINS (#529). Since #398 the shipped wheel executes MXFP4
directly, which makes ``gguf_mxfp4_repack`` the identity -- so every assertion
below silently stopped describing anything: the value tests failed against
17-byte payloads that were never rewritten, and the ones that still passed
(``test_slice_of_repack_equals_repack_of_slice``, for instance) passed
VACUOUSLY, because slicing the identity trivially commutes with it. The repack
is nevertheless still shipped and still reachable -- the wheel is pinned
separately from the source, and ``SGLANG_GGUF_MXFP4_NATIVE=0`` is the standing
A/B lever -- so this file now forces that state in-process
(``ForcesRepackPath``) instead of inheriting whatever the wheel happens to do.
Deterministic on every wheel, and no capability skip that would sleep forever
on a native one. The NATIVE path -- the one this rig actually serves -- is
pinned separately in ``test_gguf_mxfp4_native_path_529.py``.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
from gguf.constants import GGMLQuantizationType as GGMLType
from gguf.quants import dequantize, quantize

from sglang.srt.environ import envs
from sglang.srt.model_loader.gguf_mxfp4_repack import (
    log_gguf_repack_plan,
    repack_enabled,
    repack_source_types,
    repacked_gguf_bytes,
    repacked_gguf_type,
)
from sglang.test.gguf_mxfp4_state import ForcesRepackPath

MXFP4_TYPE_SIZE = 17
Q5_0_TYPE_SIZE = 22
BLOCK_SIZE = 32

#: The real vehicle. Absent on a machine that has not downloaded it; the tests
#: that need it skip rather than pass vacuously.
REAL_GGUF = (
    "/spinning/llm_stuff/club-3090/models-cache/DeepSeek-V4-Flash-0731-GGUF/"
    "UD-Q3_K_XL/DeepSeek-V4-Flash-0731-UD-Q3_K_XL-00001-of-00004.gguf"
)


def _mxfp4_bytes(x: np.ndarray) -> np.ndarray:
    """Quantize ``x`` to MXFP4 and return the raw block bytes, rows intact."""
    return quantize(x, GGMLType.MXFP4).reshape(x.shape[:-1] + (-1,))


def _assert_dequant_identical(mxfp4_bytes: np.ndarray, name: str = "t") -> np.ndarray:
    """Repack ``mxfp4_bytes`` and assert the two dequants agree exactly."""
    q5 = repacked_gguf_bytes(GGMLType.MXFP4, mxfp4_bytes, name)
    want = dequantize(np.ascontiguousarray(mxfp4_bytes), GGMLType.MXFP4)
    got = dequantize(q5, GGMLType.Q5_0)
    np.testing.assert_array_equal(got, want)
    return q5


class TestRepackValueExactness(ForcesRepackPath, unittest.TestCase):
    """Synthetic blocks, including the ones a bridge gets wrong."""

    def test_random_rows_dequantize_identically(self):
        rng = np.random.default_rng(0x391)
        magnitudes = rng.choice([1e-3, 1.0, 50.0], size=(64, 1)).astype(np.float32)
        x = rng.standard_normal((64, 256)).astype(np.float32) * magnitudes
        self._check(x)

    def test_degenerate_blocks(self):
        """All-zero, both saturations, and a single non-zero element."""
        x = np.zeros((4, BLOCK_SIZE), dtype=np.float32)
        x[1] = 6.0
        x[2] = -6.0
        x[3, 7] = 1.5
        self._check(x)

    def test_scale_range_edges_are_exact(self):
        """The fp16 window's two ends survive; nothing is clipped silently.

        ``2**-24`` is fp16's smallest subnormal and ``2**15`` its largest power
        of two, so a block scaled to either edge is the last one the repack may
        accept. Both are built directly as MXFP4 blocks rather than quantized
        from floats, because the quantizer would never choose these exponents.
        """
        for exponent in (-24, 15, -23, 14, 0):
            with self.subTest(exponent=exponent):
                block = np.zeros((1, MXFP4_TYPE_SIZE), dtype=np.uint8)
                block[0, 0] = exponent + 128
                # Codes 1..15 across both nibble halves, so every lattice point
                # (positive and negative) rides on this scale.
                block[0, 1:] = np.arange(16, dtype=np.uint8) | (
                    np.arange(16, dtype=np.uint8) << 4
                )
                self.assertEqual(
                    _assert_dequant_identical(block).shape, (1, Q5_0_TYPE_SIZE)
                )

    def test_shape_and_size_of_the_repacked_payload(self):
        x = np.zeros((8, 256), dtype=np.float32)
        raw = _mxfp4_bytes(x)
        q5 = repacked_gguf_bytes(GGMLType.MXFP4, raw, "t")
        self.assertEqual(raw.shape, (8, 256 // BLOCK_SIZE * MXFP4_TYPE_SIZE))
        self.assertEqual(q5.shape, (8, 256 // BLOCK_SIZE * Q5_0_TYPE_SIZE))
        self.assertEqual(q5.dtype, np.uint8)

    def test_type_marker_is_rewritten_only_for_covered_types(self):
        self.assertEqual(repacked_gguf_type(GGMLType.MXFP4, "t"), GGMLType.Q5_0)
        for untouched in (GGMLType.Q8_0, GGMLType.IQ3_XXS, GGMLType.F32, GGMLType.BF16):
            self.assertEqual(repacked_gguf_type(untouched, "t"), untouched)
            payload = np.zeros((4,), dtype=np.uint8)
            self.assertIs(repacked_gguf_bytes(untouched, payload, "t"), payload)

    def _check(self, x: np.ndarray) -> None:
        _assert_dequant_identical(_mxfp4_bytes(x))


class TestRepackGatesCanFail(ForcesRepackPath, unittest.TestCase):
    """A gate that has never failed is not known to be a gate."""

    def _out_of_range_block(self, e8m0: int) -> np.ndarray:
        block = np.zeros((1, MXFP4_TYPE_SIZE), dtype=np.uint8)
        block[0, 0] = e8m0
        block[0, 1] = 0x07  # one non-zero code, so the all-zero shortcut misses
        return block

    def test_scale_above_fp16_is_refused_naming_tensor_and_scale(self):
        with self.assertRaises(ValueError) as caught:
            repacked_gguf_bytes(
                GGMLType.MXFP4,
                self._out_of_range_block(250),
                "blk.7.ffn_down_exps.weight",
            )
        message = str(caught.exception)
        self.assertIn("blk.7.ffn_down_exps.weight", message)
        self.assertIn("250", message)
        self.assertIn("2**122", message)

    def test_scale_below_fp16_is_refused(self):
        # e8m0 = 103 -> 2**-25, one exponent below the smallest fp16 subnormal.
        with self.assertRaises(ValueError) as caught:
            repacked_gguf_bytes(
                GGMLType.MXFP4, self._out_of_range_block(103), "blk.0.some.weight"
            )
        self.assertIn("2**-25", str(caught.exception))

    def test_the_edge_one_step_inside_is_accepted(self):
        """Pins that the refusal above is about the range, not about the
        method: 2**-24 and 2**15 pass the same call that 2**-25 and 2**122
        fail."""
        for e8m0 in (104, 143):
            repacked_gguf_bytes(GGMLType.MXFP4, self._out_of_range_block(e8m0), "t")

    def test_row_not_a_whole_number_of_blocks_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            repacked_gguf_bytes(
                GGMLType.MXFP4, np.zeros((2, 20), dtype=np.uint8), "blk.1.odd.weight"
            )
        self.assertIn("blk.1.odd.weight", str(caught.exception))

    def test_opt_out_restores_the_refusal(self):
        self.assertTrue(repack_enabled())
        self.assertEqual(repack_source_types(), {GGMLType.MXFP4})
        with envs.SGLANG_GGUF_MXFP4_REPACK.override(False):
            self.assertFalse(repack_enabled())
            # Nothing is claimed executable any more...
            self.assertEqual(repack_source_types(), set())
            # ...and both stream entry points refuse loudly rather than
            # returning the unexecutable payload untouched.
            with self.assertRaises(RuntimeError) as caught:
                repacked_gguf_type(GGMLType.MXFP4, "blk.3.ffn_down_exps.weight")
            self.assertIn("SGLANG_GGUF_MXFP4_REPACK", str(caught.exception))
            self.assertIn("blk.3.ffn_down_exps.weight", str(caught.exception))
            with self.assertRaises(RuntimeError):
                repacked_gguf_bytes(
                    GGMLType.MXFP4,
                    np.zeros((1, MXFP4_TYPE_SIZE), dtype=np.uint8),
                    "blk.3.ffn_down_exps.weight",
                )
        self.assertTrue(repack_enabled())

    def test_deepseek4_executability_gate_follows_the_opt_out(self):
        """The family gate is the user-visible refusal; it must track the
        switch, not carry a second opinion."""
        from sglang.srt.model_loader.gguf_deepseek4 import _supported_ggml_types

        self.assertIn(GGMLType.MXFP4, _supported_ggml_types())
        with envs.SGLANG_GGUF_MXFP4_REPACK.override(False):
            self.assertNotIn(GGMLType.MXFP4, _supported_ggml_types())


class TestPerExpertSplitInteraction(ForcesRepackPath, unittest.TestCase):
    """The stacked ``ffn_*_exps`` tensors are split into per-expert tensors.
    A Q5_0 block is self-contained, so the split and the repack commute."""

    def _stacked(self, n_experts: int, rows: int, k: int) -> np.ndarray:
        rng = np.random.default_rng(26)
        x = rng.standard_normal((n_experts, rows, k)).astype(np.float32)
        return _mxfp4_bytes(x)

    def test_slice_of_repack_equals_repack_of_slice(self):
        for k in (2048, 4096):  # down_proj input dim, gate/up input dim
            with self.subTest(k=k):
                stacked = self._stacked(3, 2, k)
                whole = repacked_gguf_bytes(GGMLType.MXFP4, stacked, "stacked")
                for expert in range(stacked.shape[0]):
                    part = repacked_gguf_bytes(
                        GGMLType.MXFP4, stacked[expert], "stacked"
                    )
                    np.testing.assert_array_equal(part, whole[expert])

    def test_per_expert_payload_dequantizes_to_that_expert(self):
        stacked = self._stacked(3, 2, 2048)
        for expert in range(stacked.shape[0]):
            want = dequantize(np.ascontiguousarray(stacked[expert]), GGMLType.MXFP4)
            got = dequantize(
                repacked_gguf_bytes(GGMLType.MXFP4, stacked[expert], "s"),
                GGMLType.Q5_0,
            )
            np.testing.assert_array_equal(got, want)

    def test_block_alignment_of_the_real_geometries(self):
        """K = 2048 (down) and K = 4096 (gate/up on layer 26) are both whole
        numbers of 32-element blocks, so no expert slice ever straddles one."""
        for k in (2048, 4096):
            self.assertEqual(k % BLOCK_SIZE, 0)
            self.assertEqual((k // BLOCK_SIZE * MXFP4_TYPE_SIZE) % MXFP4_TYPE_SIZE, 0)


def _write_synthetic_gguf(path: str) -> dict:
    """A minimal GGUF exercising all three consumer shapes: a stacked expert
    tensor (split per expert), a dense linear, and the embedding."""
    import gguf

    rng = np.random.default_rng(7)
    writer = gguf.GGUFWriter(path, "llama")
    writer.add_block_count(1)

    payloads = {}
    for name, shape in (
        ("blk.0.ffn_down_exps.weight", (4, 2, 64)),  # experts x rows x K
        ("blk.0.attn_q.weight", (3, 64)),
        ("token_embd.weight", (5, 64)),
    ):
        x = rng.standard_normal(shape).astype(np.float32)
        raw = _mxfp4_bytes(x)
        payloads[name] = raw
        writer.add_tensor(name, raw, raw_shape=raw.shape, raw_dtype=GGMLType.MXFP4)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return payloads


class TestIteratorEndToEnd(ForcesRepackPath, unittest.TestCase):
    """The real iterator over a real (small) file: no consumer sees type 39."""

    def test_stream_is_q5_0_everywhere_and_value_identical(self):
        from sglang.srt.model_loader.weight_utils import gguf_quant_weights_iterator

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "synthetic.gguf")
            payloads = _write_synthetic_gguf(path)
            name_map = {
                "blk.0.attn_q.weight": "model.layers.0.self_attn.q_proj.weight",
                "token_embd.weight": "model.embed_tokens.weight",
                "blk.0.ffn_down_exps.weight": "model.layers.0.mlp.experts.down_proj.weight",
            }
            stream = dict(gguf_quant_weights_iterator(path, name_map))

        markers = {k: v for k, v in stream.items() if k.endswith(".qweight_type")}
        self.assertTrue(markers)
        for key, marker in markers.items():
            self.assertEqual(
                int(marker), int(GGMLType.Q5_0), f"{key} still carries type 39"
            )

        # Dense linear and embedding: the whole payload, value-identical.
        for gguf_name, hf_name in (
            ("blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.qweight"),
            ("token_embd.weight", "model.embed_tokens.qweight"),
        ):
            got = dequantize(stream[hf_name].numpy(), GGMLType.Q5_0)
            want = dequantize(payloads[gguf_name], GGMLType.MXFP4)
            np.testing.assert_array_equal(got, want)

        # Stacked experts: one tensor per expert, each self-contained.
        stacked = payloads["blk.0.ffn_down_exps.weight"]
        for expert in range(stacked.shape[0]):
            key = f"model.layers.0.mlp.experts.{expert}.down_proj.qweight"
            got = dequantize(stream[key].numpy(), GGMLType.Q5_0)
            want = dequantize(np.ascontiguousarray(stacked[expert]), GGMLType.MXFP4)
            np.testing.assert_array_equal(got, want)
            self.assertEqual(
                int(
                    stream[
                        f"model.layers.0.mlp.experts.{expert}.down_proj.qweight_type"
                    ]
                ),
                int(GGMLType.Q5_0),
            )

    def test_iterator_refuses_when_the_repack_is_switched_off(self):
        from sglang.srt.model_loader.weight_utils import gguf_quant_weights_iterator

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "synthetic.gguf")
            _write_synthetic_gguf(path)
            with envs.SGLANG_GGUF_MXFP4_REPACK.override(False):
                with self.assertRaises(RuntimeError) as caught:
                    list(gguf_quant_weights_iterator(path, {}))
        self.assertIn("SGLANG_GGUF_MXFP4_REPACK", str(caught.exception))

    def test_repack_plan_is_logged_with_the_inflation(self):
        import gguf

        class _Tensor:
            def __init__(self, tensor_type, n_bytes):
                self.tensor_type = tensor_type
                self.n_bytes = n_bytes

            name = "t"

        # 45 tensors of 1.0625 GiB each = the real UD-Q3_K_XL MXFP4 inventory.
        tensors = [
            _Tensor(gguf.GGMLQuantizationType.MXFP4, 1140850688) for _ in range(45)
        ]
        with self.assertLogs(
            "sglang.srt.model_loader.gguf_mxfp4_repack", level="INFO"
        ) as logs:
            log_gguf_repack_plan(tensors)
        line = "\n".join(logs.output)
        self.assertIn("MXFP4->Q5_0", line)
        self.assertIn("45 tensor(s)", line)
        self.assertIn("47.81 -> 61.88 GiB", line)

    def test_no_log_line_without_mxfp4(self):
        import gguf

        class _Tensor:
            tensor_type = gguf.GGMLQuantizationType.Q8_0
            n_bytes = 1024
            name = "t"

        logger_name = "sglang.srt.model_loader.gguf_mxfp4_repack"
        with self.assertNoLogs(logger_name, level="INFO"):
            log_gguf_repack_plan([_Tensor()])


@unittest.skipUnless(os.path.exists(REAL_GGUF), f"{REAL_GGUF} not present")
class TestRealFileBlocks(ForcesRepackPath, unittest.TestCase):
    """The published DeepSeek V4 Flash export, one real block at a time.

    Header-only reads plus a handful of bytes: the shards are memory-mapped and
    nothing near 119 GiB is touched.
    """

    @classmethod
    def setUpClass(cls):
        from sglang.srt.model_loader.gguf_shards import (
            iter_gguf_tensors,
            resolve_gguf_shard_paths,
        )

        cls.tensors = {
            str(t.name): t
            for t in iter_gguf_tensors(resolve_gguf_shard_paths(REAL_GGUF))
            if t.name in ("blk.0.ffn_down_exps.weight", "blk.26.ffn_gate_exps.weight")
        }

    def _one_block(self, name: str):
        tensor = self.tensors[name]
        self.assertEqual(tensor.tensor_type, GGMLType.MXFP4)
        # First block of the first row of the first expert.
        return np.array(tensor.data[0, 0, :MXFP4_TYPE_SIZE]).reshape(1, MXFP4_TYPE_SIZE)

    def test_real_down_proj_block_k2048(self):
        tensor = self.tensors["blk.0.ffn_down_exps.weight"]
        self.assertEqual(tensor.data.shape[-1] // MXFP4_TYPE_SIZE * BLOCK_SIZE, 2048)
        _assert_dequant_identical(
            self._one_block("blk.0.ffn_down_exps.weight"),
            "blk.0.ffn_down_exps.weight",
        )

    def test_real_layer26_gate_block_k4096(self):
        tensor = self.tensors["blk.26.ffn_gate_exps.weight"]
        self.assertEqual(tensor.data.shape[-1] // MXFP4_TYPE_SIZE * BLOCK_SIZE, 4096)
        _assert_dequant_identical(
            self._one_block("blk.26.ffn_gate_exps.weight"),
            "blk.26.ffn_gate_exps.weight",
        )

    def test_real_expert_row_survives_the_split(self):
        """A whole row of one real expert -- the unit the per-expert split
        hands to the kernels -- repacks value-exactly."""
        for name in ("blk.0.ffn_down_exps.weight", "blk.26.ffn_gate_exps.weight"):
            with self.subTest(name=name):
                row = np.array(self.tensors[name].data[0, 0])
                q5 = _assert_dequant_identical(row.reshape(1, -1), name)
                self.assertEqual(
                    q5.shape[-1] * MXFP4_TYPE_SIZE, row.shape[-1] * Q5_0_TYPE_SIZE
                )


if __name__ == "__main__":
    unittest.main()
