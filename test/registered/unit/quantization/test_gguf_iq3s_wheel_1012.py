# SPDX-License-Identifier: Apache-2.0
"""GGUF-NF G12 (2026-10-12): the IQ3_S grid of the sgl-kernel WHEEL (dequantize, dense MMVQ, MoE moe_vec) and the dispatch that keeps IQ3_S off it.

G11 found on metal that the vendored JIT MMQ decoded IQ3_S with ggml's pre-final ``iq3xs_grid`` (grid value 15 as the byte 62 -> 15.5
instead of 15, +3.3 % on ~1/8 of the weights) and fixed the JIT kernel (prefill, M >= 128).  The sgl-kernel wheel carries the very same table and
the scale form ``d * (0.5 + s) * 0.5``: ``ggml_dequantize``, ``ggml_mul_mat_vec_a8`` (MMVQ) and ``ggml_moe_a8_vec`` (moe_vec = every NF decode step).

Belegt on the desk (this file pins the code half; the evidence of the report):
* the wheel of the image (sha256 605c54e7..., BUILD_INFO kernel_wheel) is built from THIS repo's sgl-kernel/ (build_kernel_wheel_cu13_120a.sh);
  its sm_86 and sm_120 cubins contain the 2048-byte iq3xs_grid with 150 bytes of 62 and not the corrected table;
* upstream sgl-project/sglang main (e6072935bf, 2026-10-11) carries the identical table in
  python/sglang/kernels/aot/csrc/quantization/gguf/ggml-common.h (byte-identical to ours) and uses it in dequantize.cuh / vecdotq.cuh;
* NF IQ3_S weights: ffn_gate_exps + ffn_up_exps of 47 of 48 layers (the 48th is IQ4_XS); everything dense is Q8_0 (header census, report).

The G12 fix:
* wheel SOURCE: ``iq3s_grid_ggml.cuh`` (ggml's values x 4, 15 -> 60) used by ``vec_dot_iq3_s_q8_1`` and ``dequantize_block_iq3_s``, plus the marker op
  ``ggml_iq3s_grid_ggml`` (registered like ``ggml_mxfp4_native``).  ggml-common.h stays byte-identical (the JIT copy is a verbatim prefix of it);
* DISPATCH for a wheel without the marker (the image's wheel until it is rebuilt): IQ3_S never reaches a wheel kernel -- dense M <= 16 and MoE of every
  M through the JIT MMQ (fixed in G11), dense M > 16 / embedding rows / per-token fallback through ``jit_kernel/gguf_iq3s_dequant.py`` (torch,
  gguf-py arithmetic); a load that cannot do that refuses (``IQ3S-WHEEL-GRID``).

What the tests do (all CPU, no GPU; GPU twins are LOCKED behind GGUF_GPU_TESTS=1):
* source pins and a CPU emulation of the wheel's dequantize kernel / MMVQ with the table the SOURCE really indexes: exact format with the G12 table, the
  +3.3 % defect with the old one (mutant), on synthetic and real NF IQ3_S blocks, in the decode form (M = 1..8, top_k 10, K = 2560);
* the torch dequant against gguf-py, bit for bit;
* the dispatch with kernels as CPU stubs: marker absent -> no wheel IQ3_S kernel is ever called; marker present -> every pre-G12 branch unchanged.

Usage:  /spinning/gpu-arb/pytest_gedeckelt.sh test/registered/unit/quantization/test_gguf_iq3s_wheel_1012.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import re
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import gguf as gguf_lib
import numpy as np
import torch

from sglang.jit_kernel import gguf_iq3s_dequant as DQ
from sglang.jit_kernel import gguf_iq_mmq_policy as P
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=120, suite="base-a-test-cpu")

_ROOT = Path(__file__).resolve().parents[4]
_SGL = _ROOT / "sgl-kernel" / "csrc" / "quantization" / "gguf"
_JIT_CSRC = _ROOT / "python" / "sglang" / "jit_kernel" / "csrc" / "gguf_iq_mmq"
_HERE = Path(__file__).parent


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_g11 = _load("_gguf_iq3s_grid_1011_for_g12", "test_gguf_iq3s_grid_1011.py")  # the CPU model of the kernel (q8_1, kernel weights, bf16)
_g9 = _g11._base  # test_gguf_iq_mmq_1009: block generator, NF header reader, dispatch harness

GPU_TESTS = os.environ.get("GGUF_GPU_TESTS") == "1"
GPU_SKIP = "needs a gpuq window: set GGUF_GPU_TESTS=1 with CUDA_VISIBLE_DEVICES=<card>"
IQ3_S = 21
IQ4_NL = 20
Q8_0 = 8
NF_OK = (_g9.NF_GGUF_DIR / "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf").is_file()


# ---------------------------------------------------------------------------------------------------------------------
# sources
def _read(path: Path) -> str:
    return path.read_text()


def _function_body(src: str, signature_regex: str) -> str:
    """Text of the function whose signature matches ``signature_regex`` up to the next top-level ``\\n}\\n``."""
    m = re.search(signature_regex, src)
    assert m, signature_regex
    end = src.index("\n}\n", m.end())
    return src[m.start() : end]


def _table(text: str, name: str) -> np.ndarray:
    m = re.search(r"\b%s\[512\]\s*=\s*\{(.*?)\};" % re.escape(name), text, re.S)
    assert m, name
    return np.array([int(x, 16) for x in re.findall(r"0x[0-9a-fA-F]{8}", m.group(1))], dtype=np.uint32)


def _bytes(table: np.ndarray) -> np.ndarray:
    return table.view(np.uint8).reshape(len(table), 4).astype(np.int32)


_WHEEL_FILES = {"vecdotq.cuh": _SGL / "vecdotq.cuh", "dequantize.cuh": _SGL / "dequantize.cuh"}


def _wheel_table_name(fname: str, signature_regex: str) -> str:
    body = _function_body(_read(_WHEEL_FILES[fname]), signature_regex)
    names = set(re.findall(r"\b(iq3\w*_grid\w*)\s*\+", body))
    assert len(names) == 1, (fname, names)
    return names.pop()


def wheel_grid_of(fname: str, signature_regex: str) -> np.ndarray:
    """The table the wheel function REALLY indexes, read from the source: its name from the function body, its bytes from the file that defines it."""
    name = _wheel_table_name(fname, signature_regex)
    for f in (_SGL / "iq3s_grid_ggml.cuh", _SGL / "ggml-common.h"):
        if f.is_file() and re.search(r"\b%s\[512\]\s*=" % name, _read(f)):
            return _bytes(_table(_read(f), name))
    raise AssertionError(f"{name} defined nowhere")


_DEQ_SIG = r"static __global__ void dequantize_block_iq3_s\("
_VEC_SIG = r"vec_dot_iq3_s_q8_1\(const void\* __restrict__ vbq"


def wheel_dequant_grid() -> np.ndarray:
    return wheel_grid_of("dequantize.cuh", _DEQ_SIG)


def wheel_vecdot_grid() -> np.ndarray:
    return wheel_grid_of("vecdotq.cuh", _VEC_SIG)


def old_wheel_grid() -> np.ndarray:
    """The table the wheel used before G12 (and upstream still uses): iq3xs_grid of the byte-identical ggml-common.h."""
    return _bytes(_table(_read(_SGL / "ggml-common.h"), "iq3xs_grid"))


def ggml_grid_x4() -> np.ndarray:
    return _g11.ggml_grid_x4()


# ---------------------------------------------------------------------------------------------------------------------
# CPU emulation of the wheel's dequantize_block_iq3_s (literal thread layout of the CUDA kernel) -> float32 [rows, K]
def emulate_wheel_dequant(raw: np.ndarray, grid: np.ndarray) -> np.ndarray:
    rows = raw.shape[0]
    d, qs, qh, signs, scales = _g11.split_blocks(raw)
    nb = d.shape[1]
    y = np.zeros((rows, nb, 256), np.float32)
    for tid in range(32):  # blockDim 32 of dequantize_row_iq3_s_cuda, blockIdx = block
        il, ib = tid // 8, tid % 8
        qs_b = qs[:, :, 8 * ib : 8 * ib + 8].astype(np.int64)
        qh_b = qh[:, :, ib].astype(np.int64)
        i1 = qs_b[:, :, 2 * il] | ((qh_b << (8 - 2 * il)) & 256)
        i2 = qs_b[:, :, 2 * il + 1] | ((qh_b << (7 - 2 * il)) & 256)
        sc = ((scales[:, :, ib // 2].astype(np.int64) >> (4 * (ib % 2))) & 0xF).astype(np.float32)
        dd = (d * (np.float32(0.5) + sc)) * np.float32(0.5)  # __half2float(d) * (0.5f + s) * 0.5f
        sg = signs[:, :, 4 * ib + il].astype(np.int64)
        base = 32 * ib + 8 * il
        for j in range(4):
            neg0 = (sg & (1 << j)) != 0  # kmask_iq2xs[j + 0]
            neg1 = (sg & (1 << (j + 4))) != 0
            y[:, :, base + j] = dd * grid[i1][..., j].astype(np.float32) * np.where(neg0, -1.0, 1.0).astype(np.float32)
            y[:, :, base + 4 + j] = dd * grid[i2][..., j].astype(np.float32) * np.where(neg1, -1.0, 1.0).astype(np.float32)
    return y.reshape(rows, nb * 256)


def reference_rows(raw: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(gguf_lib.quants.dequantize(raw, gguf_lib.GGMLQuantizationType.IQ3_S)).astype(np.float32)


def synth(rows: int, k: int, seed: int) -> np.ndarray:
    return _g9._synth_blocks(IQ3_S, rows, k, seed=seed)


# ---------------------------------------------------------------------------------------------------------------------
class TestWheelSources(CustomTestCase):
    def test_new_header_is_the_ggml_grid_times_four_index_for_index(self):
        np.testing.assert_array_equal(_bytes(_table(_read(_SGL / "iq3s_grid_ggml.cuh"), "iq3s_grid_ggml")), ggml_grid_x4())

    def test_new_header_equals_the_jit_table_of_g11(self):
        jit = _bytes(_table(_read(_JIT_CSRC / "iq_mmq_tiles.cuh"), "iq3s_grid_ggml"))
        np.testing.assert_array_equal(_bytes(_table(_read(_SGL / "iq3s_grid_ggml.cuh"), "iq3s_grid_ggml")), jit)

    def test_old_table_differs_from_the_format_only_at_the_byte_62(self):
        old, new = old_wheel_grid(), ggml_grid_x4()
        self.assertEqual(sorted(set(old.flatten().tolist())), [4, 12, 20, 28, 36, 44, 52, 62])
        diff = old != new
        self.assertEqual(int(diff.sum()), 150)
        self.assertTrue(((old[diff] == 62) & (new[diff] == 60)).all())

    def test_ggml_common_h_is_untouched_and_still_carries_the_old_table(self):
        """The JIT header is a verbatim prefix of it (test_gguf_iq_mmq_1009); only IQ3_S stopped reading the old table."""
        self.assertEqual(
            hashlib.sha256((_SGL / "ggml-common.h").read_bytes()).hexdigest(), _g9.BASE_SHA256["ggml-common.h"]
        )

    def test_both_wheel_functions_index_the_ggml_table(self):
        self.assertEqual(_wheel_table_name("dequantize.cuh", _DEQ_SIG), "iq3s_grid_ggml")
        self.assertEqual(_wheel_table_name("vecdotq.cuh", _VEC_SIG), "iq3s_grid_ggml")
        np.testing.assert_array_equal(wheel_dequant_grid(), ggml_grid_x4())
        np.testing.assert_array_equal(wheel_vecdot_grid(), ggml_grid_x4())

    def test_no_wheel_code_line_indexes_the_old_table(self):
        """Outside comments no line of any wheel GGUF source but its definition in ggml-common.h names iq3xs_grid."""
        for p in sorted(_SGL.iterdir()):
            if p.suffix in (".cu", ".cuh") or (p.suffix == ".h" and p.name != "ggml-common.h"):
                for line in _read(p).splitlines():
                    self.assertNotIn("iq3xs_grid", line.split("//", 1)[0], f"{p.name}: {line.strip()}")

    def test_scale_form_is_unchanged_so_the_x4_table_is_the_right_one(self):
        self.assertIn("(0.5f + ((x[i].scales[ib / 2] >> 4 * (ib % 2)) & 0xf)) * 0.5f", _read(_SGL / "dequantize.cuh"))
        vec = _function_body(_read(_SGL / "vecdotq.cuh"), _VEC_SIG)
        self.assertIn("(0.5f + ((bq2->scales[ib32 / 2] >> 4 * (ib32 % 2)) & 0xf))", vec)
        self.assertIn("__low2float(bq8_1[ib32].ds) * 0.5f", vec)

    def test_header_include_order_and_marker_registration(self):
        kern = _read(_SGL / "gguf_kernel.cu")
        self.assertLess(kern.index('#include "ggml-common.h"'), kern.index('#include "iq3s_grid_ggml.cuh"'))
        self.assertLess(kern.index('#include "iq3s_grid_ggml.cuh"'), kern.index('#include "vecdotq.cuh"'))
        self.assertIn("int64_t ggml_iq3s_grid_ggml() {\n  return 1;\n}", kern)
        ext = _read(_ROOT / "sgl-kernel" / "csrc" / "common_extension.cc")
        self.assertIn('m.def("ggml_iq3s_grid_ggml() -> int");\n  m.impl("ggml_iq3s_grid_ggml", &ggml_iq3s_grid_ggml);', ext)
        self.assertIn("int64_t ggml_iq3s_grid_ggml();", _read(_ROOT / "sgl-kernel" / "include" / "sgl_kernel_ops.h"))
        self.assertEqual(P.IQ3S_WHEEL_MARKER_OP, "ggml_iq3s_grid_ggml")
        self.assertEqual(_read(_SGL / "iq3s_grid_ggml.cuh").count("#pragma once"), 1)  # ggml-common.h has no guard; the new header must not re-include it
        self.assertNotIn("#include", _read(_SGL / "iq3s_grid_ggml.cuh").replace("// Include AFTER", ""))

    def test_marker_op_is_in_the_catch_all_form_of_its_siblings(self):
        """No device key: a probe without a tensor cannot be routed by the dispatcher (the #518 lesson of ggml_mxfp4_native)."""
        ext = _read(_ROOT / "sgl-kernel" / "csrc" / "common_extension.cc")
        self.assertNotIn("torch::kCUDA, &ggml_iq3s_grid_ggml", ext)


class TestWheelKernelModel(CustomTestCase):
    """The wheel's dequantize kernel, emulated with the table the source indexes, against gguf-py (the format)."""

    def test_wheel_dequant_equals_the_format_on_synthetic_blocks(self):
        raw = synth(96, 512, seed=1)
        np.testing.assert_allclose(emulate_wheel_dequant(raw, wheel_dequant_grid()), reference_rows(raw), rtol=1e-6, atol=1e-7)

    def test_mutant_old_table_is_plus_3_3_percent_on_grid_15(self):
        raw = synth(96, 512, seed=1)
        got, ref = emulate_wheel_dequant(raw, old_wheel_grid()), reference_rows(raw)
        wrong = np.abs(got - ref) > 1e-6
        self.assertGreater(float(wrong.mean()), 0.05)
        self.assertLess(float(wrong.mean()), 0.12)
        np.testing.assert_allclose((got[wrong] - ref[wrong]) / ref[wrong], 62 / 60 - 1, rtol=2e-3)

    @unittest.skipUnless(NF_OK, "NF UD-IQ4_XS GGUF header files not on this host")
    def test_wheel_dequant_equals_the_format_on_real_nf_blocks_and_the_old_table_does_not(self):
        reader = gguf_lib.GGUFReader(str(_g9.NF_GGUF_DIR / "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf"))
        t = {x.name: x for x in reader.tensors}["blk.4.ffn_up_exps.weight"]
        self.assertEqual(int(t.tensor_type), IQ3_S)
        raw = np.ascontiguousarray(t.data[3, :48])  # one expert, 48 rows of K = 2560
        ref = reference_rows(raw)
        np.testing.assert_allclose(emulate_wheel_dequant(raw, wheel_dequant_grid()), ref, rtol=1e-6, atol=1e-7)
        old = emulate_wheel_dequant(raw, old_wheel_grid())
        rel = float(np.sqrt(((old - ref) ** 2).mean()) / np.sqrt((ref**2).mean()))
        self.assertGreater(rel, 0.01)  # the grid defect alone: ~1.6 % of the weight RMS


def _chain_rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt())


@unittest.skipUnless(NF_OK, "NF UD-IQ4_XS GGUF header files not on this host")
class TestDecodeFormRealBlocks(CustomTestCase):
    """Production DECODE form: real NF IQ3_S blocks (layer 0 gate, layer 4 up), K = 2560, M = 1..8 tokens, top_k 10 distinct experts, E = 16 of 512.

    The model is the wheel's MMVQ (moe_vec): per 32 weights an exact integer dot of the grid bytes with the q8_1 activation, scaled by
    ``d * (0.5 + s) * y_d * 0.5`` -- the same arithmetic as the MMQ model of G11 (identical products, different thread layout), here fed with the
    table of ``vec_dot_iq3_s_q8_1``.  Bounds relative to the output scale (the PR's atol is blind for real blocks): relative RMS < 1 %, max < 5 % of the
    output RMS -- the G11 production bounds; CPU-model values (unmeasured on metal).
    """

    ROWS, E, TOP_K, K = 96, 16, 10, 2560

    @classmethod
    def setUpClass(cls):
        reader = gguf_lib.GGUFReader(str(_g9.NF_GGUF_DIR / "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf"))
        t = {x.name: x for x in reader.tensors}
        cls.raw = {}
        for layer, which in ((0, "gate"), (4, "up")):
            ten = t[f"blk.{layer}.ffn_{which}_exps.weight"]
            assert int(ten.tensor_type) == IQ3_S and list(ten.shape) == [2560, 640, 512]
            cls.raw[(layer, which)] = np.ascontiguousarray(ten.data[: cls.E, : cls.ROWS])
        del reader

    def _errors(self, key, m: int, act: str, grid: np.ndarray):
        raws = self.raw[key]
        g = torch.Generator().manual_seed(11 + m)
        x = (torch.rand((m, self.K), generator=g) if act == "uniform" else torch.randn((m, self.K), generator=g)).to(torch.bfloat16).float()
        ids = torch.stack([torch.randperm(self.E, generator=g)[: self.TOP_K] for _ in range(m)])
        ref = torch.einsum("tk,tjnk->tjn", x, _g11.reference_weights(raws)[ids])
        y = _g11.bf16(_g11.moe_model(x.numpy(), ids.numpy(), raws, grid))
        err = y - ref
        rms = float(ref.pow(2).mean().sqrt())
        return float(err.pow(2).mean().sqrt()) / rms, float(err.abs().max()) / rms

    def test_wheel_table_of_the_source_is_inside_the_decode_bound_for_m_1_to_8(self):
        grid = wheel_vecdot_grid()
        for key in self.raw:
            for m in (1, 2, 4, 8):
                for act in ("uniform", "normal"):
                    with self.subTest(layer=key[0], which=key[1], m=m, act=act):
                        rel, mx = self._errors(key, m, act, grid)
                        self.assertLess(rel, 0.01, f"relative RMS {rel:.4f}")
                        self.assertLess(mx, 0.05, f"max |err| / ref rms {mx:.4f}")

    def test_mutant_old_table_leaves_the_decode_bound(self):
        """The pre-G12 wheel: every decode step of NF.  Over the M = 1..8 forms the old table is outside the bound in aggregate."""
        grid = old_wheel_grid()
        outside = total = 0
        for key in self.raw:
            for m in (1, 2, 4, 8):
                for act in ("uniform", "normal"):
                    rel, mx = self._errors(key, m, act, grid)
                    total += 1
                    outside += int(rel >= 0.01 or mx >= 0.05)
        self.assertEqual(outside, total, f"{outside}/{total} forms outside the bound with the old table")


# ---------------------------------------------------------------------------------------------------------------------
class TestTorchDequantEqualsGgufPy(CustomTestCase):
    def test_equals_gguf_py_bit_for_bit_float32(self):
        raw = synth(64, 1024, seed=3)
        got = DQ.dequantize_iq3_s(torch.from_numpy(raw), torch.float32).numpy()
        np.testing.assert_array_equal(got, reference_rows(raw))

    def test_output_dtypes_are_one_rounding_of_the_float32_result(self):
        raw = synth(8, 512, seed=4)
        ref = torch.from_numpy(reference_rows(raw))
        for dt in (torch.float16, torch.bfloat16):
            got = DQ.dequantize_iq3_s(torch.from_numpy(raw), dt)
            self.assertEqual(got.dtype, dt)
            self.assertTrue(torch.equal(got, ref.to(dt)))

    def test_grid_values_are_the_odd_numbers_1_to_15(self):
        self.assertEqual(sorted(set(DQ.ggml_grid().astype(int).flatten().tolist())), [1, 3, 5, 7, 9, 11, 13, 15])

    def test_every_grid_15_weight_is_decoded_as_15_not_15_5(self):
        """One block, scale nibble 0 (factor 1), d = 1: the output of an entry is exactly its grid value times the sign."""
        raw = np.zeros((1, 110), np.uint8)
        raw[0, 0:2] = np.array([1.0], np.float16).view(np.uint8)
        idx15 = int(np.argwhere(ggml_grid_x4()[:, 0] == 60)[0][0])
        raw[0, 2] = idx15 & 0xFF
        raw[0, 66] = (idx15 >> 8) & 1
        out = DQ.dequantize_iq3_s(torch.from_numpy(raw), torch.float32)[0, 0].item()
        self.assertEqual(out, 15.0)

    def test_rejects_non_block_shapes(self):
        with self.assertRaises(ValueError):
            DQ.dequantize_iq3_s(torch.zeros(2, 100, dtype=torch.uint8), torch.float32)
        with self.assertRaises(ValueError):
            DQ.dequantize_iq3_s(torch.zeros(2, 110, dtype=torch.int8), torch.float32)

    @unittest.skipUnless(NF_OK, "NF UD-IQ4_XS GGUF header files not on this host")
    def test_real_nf_blocks(self):
        reader = gguf_lib.GGUFReader(str(_g9.NF_GGUF_DIR / "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf"))
        t = {x.name: x for x in reader.tensors}["blk.0.ffn_gate_exps.weight"]
        raw = np.ascontiguousarray(t.data[5, :64])
        np.testing.assert_array_equal(DQ.dequantize_iq3_s(torch.from_numpy(raw), torch.float32).numpy(), reference_rows(raw))


# ---------------------------------------------------------------------------------------------------------------------
class TestPolicy(CustomTestCase):
    def test_blocked_only_for_iq3s_without_the_marker(self):
        self.assertTrue(P.iq3s_wheel_blocked(IQ3_S, False))
        self.assertFalse(P.iq3s_wheel_blocked(IQ3_S, True))
        for t in sorted(set(range(0, 40)) - {IQ3_S}):
            self.assertFalse(P.iq3s_wheel_blocked(t, False), t)

    def test_dense_window_is_1_to_16(self):
        self.assertFalse(P.dense_iq3s_window_ok(0))
        self.assertTrue(all(P.dense_iq3s_window_ok(m) for m in range(1, 17)))
        self.assertFalse(P.dense_iq3s_window_ok(17))

    def test_moe_decode_shape_has_no_token_or_assignment_floor_only_the_launch_limit(self):
        self.assertFalse(P.moe_mmq_shape_ok(1, 512, 10))  # the speed gates of the PR: refused (that path is the wheel moe_vec)
        self.assertTrue(P.moe_mmq_decode_shape_ok(1, 512, 10))
        self.assertTrue(P.moe_mmq_decode_shape_ok(8, 512, 10))
        self.assertTrue(P.moe_mmq_decode_shape_ok(127, 512, 10))
        self.assertFalse(P.moe_mmq_decode_shape_ok(0, 512, 10))
        self.assertFalse(P.moe_mmq_decode_shape_ok(1 << 15, 512, 10))  # grid.y limit: 65535 blocks of 4

    def test_load_line_name(self):
        self.assertEqual(P.IQ3S_DECODE_LOAD_LINE, "IQ3_S decode via JIT (wheel grid off by one)")


# ---------------------------------------------------------------------------------------------------------------------
class _Harness:
    """The stub harness of test_gguf_iq_mmq_1009.TestDispatch (setUp / tearDown / _dense / _moe) WITHOUT its tests."""


for _name in ("setUp", "tearDown", "_dense", "_moe"):
    setattr(_Harness, _name, getattr(_g9.TestDispatch, _name))


class TestDispatch(_Harness, CustomTestCase):
    """gguf.py dispatch, kernels as CPU stubs (the 1009 harness), with the wheel marker ABSENT (the image's wheel): IQ3_S never reaches a wheel kernel."""

    def setUp(self):
        super().setUp()
        self.put("IQ3S_WHEEL_GRID_FIXED", False)

    def _fixed(self, flag: bool):
        self.put("IQ3S_WHEEL_GRID_FIXED", flag)

    # -- dense ---------------------------------------------------------------------------------------------------------
    def test_dense_iq3s_decode_window_goes_to_the_jit_mmq_for_every_m_1_to_16(self):
        for m in (1, 2, 4, 8, 9, 12, 16):
            self.assertEqual(self._dense(21, m), ["iq_dense"], m)
            self.assertEqual(self._dense(21, m, rows=4000), ["iq_dense"], m)

    def test_dense_iq3s_above_16_is_a_torch_dequant_never_the_wheel_dequant(self):
        called = []
        real = DQ.dequantize_iq3_s

        def spy(w, dt):
            called.append(tuple(w.shape))
            return torch.zeros(w.shape[0], w.shape[1] // 110 * 256, dtype=dt)

        DQ.dequantize_iq3_s = spy
        try:
            self.assertEqual(self._dense(21, 17), [])  # neither "dequant" (the wheel stub), nor "mmvq", nor "iq_dense"
            self.assertEqual(self._dense(21, 300), [])
        finally:
            DQ.dequantize_iq3_s = real
        self.assertGreater(len(called), 0)

    def test_dense_iq3s_never_uses_mmvq_or_the_wheel_dequant_when_the_jit_is_not_ready(self):
        self.G._iq_mmq_mod.is_ready = lambda t: False
        orig = DQ.dequantize_iq3_s
        DQ.dequantize_iq3_s = lambda w, dt: torch.zeros(w.shape[0], w.shape[1] // 110 * 256, dtype=dt)
        try:
            for m in (1, 4, 12, 17):
                self.assertEqual(self._dense(21, m), [], m)
        finally:
            DQ.dequantize_iq3_s = orig

    def test_dense_other_types_unchanged_by_the_blocked_flag(self):
        self.assertEqual(self._dense(20, 4), ["mmvq"])  # IQ4_NL at M=4 <= mmvq_safe 8
        self.assertEqual(self._dense(23, 4), ["mmvq"])  # IQ4_XS
        self.assertEqual(self._dense(23, 17), ["dequant"])
        self.assertEqual(self._dense(8, 8), ["mmq"])  # Q8_0 stub path

    def test_marker_present_restores_every_pre_g12_branch(self):
        self._fixed(True)
        self.assertEqual(self._dense(21, 4), ["mmvq"])
        self.assertEqual(self._dense(21, 12), ["iq_dense"])
        self.assertEqual(self._dense(21, 17), ["dequant"])
        self.assertEqual([c[0] for c in self._moe(21, 20, 4, E=64)], ["moe_vec", "moe_vec"])
        self.assertEqual([c[0] for c in self._moe(21, 20, 256, E=64)], ["iq_moe", "iq_moe"])

    # -- MoE -----------------------------------------------------------------------------------------------------------
    def test_nf_decode_m_1_to_127_goes_through_the_jit_moe_mmq_not_moe_vec(self):
        for tokens in (1, 2, 4, 8, 16, 64, 127):
            calls = self._moe(21, 20, tokens, E=64)  # IQ3_S gate/up + IQ4_NL down, top_k 10
            self.assertEqual([c[0] for c in calls], ["iq_moe", "iq_moe"], tokens)
            self.assertEqual([c[1] for c in calls], [21, 20])
            self.assertEqual(calls[0][2:5], (1280, 10, tokens))
            self.assertEqual(calls[1][2:5], (2560, 1, tokens * 10))

    def test_row_class_b_iq3s_gate_up_with_q8_0_down_decode_uses_jit_plus_the_wheel_mmq_for_the_q8_side(self):
        calls = self._moe(21, 8, 4, E=64)
        self.assertEqual([c[0] for c in calls], ["iq_moe", "wheel_moe"])

    def test_iq3s_pair_the_jit_cannot_take_never_goes_to_moe_vec(self):
        """IQ3_S + a type the JIT MoE does not serve and the wheel MMQ does not either (IQ1_M = 29, MMVQ-only): the per-token loop, which is correct."""
        calls = self._moe(21, 29, 4, E=16, top_k=2)
        names = [c[0] for c in calls]
        self.assertNotIn("moe_vec", names)
        self.assertNotIn("dequant", [n for n, c in zip(names, calls) if len(c) > 1 and c[1] == 21])
        self.assertIn("iq_dense", names)  # the IQ3_S side of every (token, expert) is M = 1 through the JIT dense MMQ

    def test_non_iq3s_moe_unchanged(self):
        self.assertEqual([c[0] for c in self._moe(20, 20, 4, E=64)], ["moe_vec", "moe_vec"])  # IQ4_NL pair at decode: moe_vec as before
        self.assertEqual([c[0] for c in self._moe(23, 20, 4, E=64)], ["moe_vec", "moe_vec"])
        self.assertEqual([c[0] for c in self._moe(23, 20, 256, E=64)], ["iq_moe", "iq_moe"])

    def test_jit_not_ready_for_iq3s_never_falls_to_moe_vec(self):
        self.G._iq_mmq_mod.is_ready = lambda t: int(t) != 21
        names = [c[0] for c in self._moe(21, 20, 4, E=16, top_k=2)]
        self.assertNotIn("moe_vec", names)
        self.assertNotIn("wheel_moe", names)

    # -- embedding -----------------------------------------------------------------------------------------------------
    def test_embedding_iq3s_is_the_torch_dequant_and_equals_gguf_py(self):
        raw = synth(32, 512, seed=9)
        qw = torch.from_numpy(raw)
        ids = torch.tensor([[3, 7, 7], [31, 0, 5]])
        got = self.G.apply_gguf_embedding(ids, qw, 21, 512, dtype=torch.float32)
        want = torch.from_numpy(reference_rows(raw))[ids.flatten()].view(2, 3, 512)
        self.assertTrue(torch.equal(got, want))
        self.assertEqual(self.calls, [])  # no wheel kernel

    # -- chunked GEMM --------------------------------------------------------------------------------------------------
    def test_chunked_torch_dequant_gemm_equals_the_unchunked_one(self):
        raw = synth(40, 512, seed=10)
        qw = torch.from_numpy(raw)
        x = torch.randn(5, 512)
        want = x @ torch.from_numpy(reference_rows(raw)).T
        for chunk in (1 << 24, 512 * 7, 512):
            old = self.G._IQ3S_TORCH_DEQUANT_CHUNK_ELEMS
            self.G._IQ3S_TORCH_DEQUANT_CHUNK_ELEMS = chunk
            try:
                got = self.G._mul_mat_iq3s_dequant(x, qw)
                rows = self.G._iq3s_dequant_rows(qw, torch.float32)
            finally:
                self.G._IQ3S_TORCH_DEQUANT_CHUNK_ELEMS = old
            torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)  # same fp32 math, different GEMM blocking: rounding-order noise only
            self.assertTrue(torch.equal(rows, torch.from_numpy(reference_rows(raw))), chunk)


class TestLoadGuard(CustomTestCase):
    """iq_mmq_prepare / _iq3s_load_guard: the named load line, the refusal, and the marker-present no-op."""

    def setUp(self):
        from sglang.srt.layers.quantization import gguf as G

        self.G = G
        self._old = {k: getattr(G, k) for k in ("_iq_mmq_mod", "IQ3S_WHEEL_GRID_FIXED", "_is_cuda", "_has_sgl_gguf_kernels", "_iq3s_load_line_logged")}
        G._is_cuda = False  # the torch dequant table is prepared only on CUDA; here no device exists
        G._iq3s_load_line_logged = False

    def tearDown(self):
        for k, v in self._old.items():
            setattr(self.G, k, v)

    def _mod(self, ready):
        m = types.SimpleNamespace()
        m.is_ready = lambda t: int(t) in ready
        m.prepare = lambda t: int(t) in ready
        m.state_text = lambda t: "IQ-MMQ-OFF: SGLANG_GGUF_IQ_MMQ=0"
        return m

    def test_ready_logs_the_named_line_once_and_does_not_raise(self):
        self.G.IQ3S_WHEEL_GRID_FIXED = False
        self.G._iq_mmq_mod = self._mod({21})
        with self.assertLogs(self.G.logger, level="WARNING") as cm:
            self.G._iq3s_load_guard({21})
            self.G._iq3s_load_guard({21, 20})
        lines = [r for r in cm.output if "IQ3_S decode via JIT (wheel grid off by one)" in r]
        self.assertEqual(len(lines), 1)
        self.assertIn("ggml_iq3s_grid_ggml", lines[0])

    def test_not_ready_refuses_with_the_reason(self):
        self.G.IQ3S_WHEEL_GRID_FIXED = False
        self.G._iq_mmq_mod = self._mod(set())
        with self.assertRaises(RuntimeError) as cm:
            self.G._iq3s_load_guard({21})
        self.assertIn("IQ3S-WHEEL-GRID", str(cm.exception))
        self.assertIn("IQ-MMQ-OFF", str(cm.exception))

    def test_module_missing_refuses(self):
        self.G.IQ3S_WHEEL_GRID_FIXED = False
        self.G._iq_mmq_mod = False
        with self.assertRaises(RuntimeError):
            self.G._iq3s_load_guard({21})

    def test_marker_present_or_other_types_is_a_no_op(self):
        self.G._iq_mmq_mod = self._mod(set())
        self.G.IQ3S_WHEEL_GRID_FIXED = True
        self.G._iq3s_load_guard({21})
        self.G.IQ3S_WHEEL_GRID_FIXED = False
        self.G._iq3s_load_guard({20, 23})  # no IQ3_S among the types: nothing to say

    def test_prepare_calls_the_guard_even_when_the_kernel_module_is_unavailable(self):
        self.G.IQ3S_WHEEL_GRID_FIXED = False
        # G13: ``mock.patch.object`` puts the module function back. The G12 form (``G._iq_mmq_kernels = ...`` then
        # ``del G._iq_mmq_kernels``) DELETED the module-level ``def`` -- every later caller in the process (here:
        # ``TestGpuDecodeForms.setUpClass`` -> ``iq_mmq_prepare``) got ``NameError: name '_iq_mmq_kernels' is not defined``.
        with mock.patch.object(self.G, "_iq_mmq_kernels", lambda: None):
            with self.assertRaises(RuntimeError):
                self.G.iq_mmq_prepare(21)

    def test_the_module_function_is_still_there_after_the_patched_test(self):
        """The G12 defect, pinned: after the test above the real ``_iq_mmq_kernels`` must exist and be the original."""
        original = self.G._iq_mmq_kernels
        self.G.IQ3S_WHEEL_GRID_FIXED = False
        with mock.patch.object(self.G, "_iq_mmq_kernels", lambda: None):
            self.assertIsNot(self.G._iq_mmq_kernels, original)
        self.assertIs(self.G._iq_mmq_kernels, original)


# ---------------------------------------------------------------------------------------------------------------------
#: gather budget of one reference chunk, in fp32 elements (2**28 = 1 GiB): the NF production form needs 32.8 M per token
_MOE_REF_CHUNK_ELEMS = 1 << 28


def moe_reference(x, wd1, wd2, tw, ids, chunk_tokens: int | None = None):
    """The fused-MoE reference (two GEMMs + silu_and_mul + weighted sum, float32, gguf-py weights), computed in token
    chunks.  Each token's result depends on that token alone, so chunking only bounds the memory of the expert gather
    (``wd1[ids]`` is [chunk, top_k, 2 * inter, K]); it does not change a single expected value.

    x [m, K], wd1 [E, 2 * inter, K], wd2 [E, K, inter], tw [m, top_k], ids [m, top_k] -> [m, K]."""
    m, top_k = int(ids.shape[0]), int(ids.shape[1])
    inter = int(wd1.shape[1]) // 2
    if chunk_tokens is None:
        per_token = top_k * int(wd1.shape[1]) * int(wd1.shape[2])
        chunk_tokens = max(1, _MOE_REF_CHUNK_ELEMS // per_token)
    out = []
    for lo in range(0, m, chunk_tokens):
        hi = min(m, lo + chunk_tokens)
        sel = ids[lo:hi].long()
        h1 = torch.einsum("tk,tjnk->tjn", x[lo:hi].float(), wd1[sel])
        h = torch.nn.functional.silu(h1[..., :inter]) * h1[..., inter:]
        out.append((torch.einsum("tjk,tjnk->tjn", h, wd2[sel]) * tw[lo:hi].float()[..., None]).sum(1))
    return torch.cat(out, dim=0)


class TestMoeReferenceChunking(CustomTestCase):
    """CPU: the chunked reference is the one-shot reference (G13, harness fix for the sm86 OOM of m = 200)."""

    def _case(self, m, E=6, top_k=3, K=16, inter=8, seed=0):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn((m, K), generator=g)
        wd1 = torch.randn((E, 2 * inter, K), generator=g)
        wd2 = torch.randn((E, K, inter), generator=g)
        ids = torch.topk(torch.rand((m, E), generator=g), top_k, dim=1).indices.to(torch.int32)
        tw = torch.softmax(torch.randn((m, top_k), generator=g), -1)
        return x, wd1, wd2, tw, ids

    @staticmethod
    def _one_shot(x, wd1, wd2, tw, ids):
        inter = wd1.shape[1] // 2
        h1 = torch.einsum("tk,tjnk->tjn", x.float(), wd1[ids.long()])
        h = torch.nn.functional.silu(h1[..., :inter]) * h1[..., inter:]
        return (torch.einsum("tjk,tjnk->tjn", h, wd2[ids.long()]) * tw.float()[..., None]).sum(1)

    def test_chunked_equals_one_shot_for_every_chunk_size(self):
        args = self._case(m=37)
        want = self._one_shot(*args)
        for chunk in (1, 2, 5, 36, 37, 1000, None):
            with self.subTest(chunk=chunk):
                got = moe_reference(*args, chunk_tokens=chunk)
                self.assertEqual(tuple(got.shape), tuple(want.shape))
                torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)

    def test_the_default_chunk_keeps_the_nf_production_gather_under_a_gigabyte_of_elements(self):
        per_token = 10 * 1280 * 2560  # top_k x (gate+up rows) x K, the NF form
        chunk = max(1, _MOE_REF_CHUNK_ELEMS // per_token)
        self.assertGreaterEqual(chunk, 1)
        self.assertLessEqual(chunk * per_token * 4, 1 << 30)  # fp32 bytes of the biggest gather
        self.assertLess(chunk, 200)  # the one-shot 200-token gather (24.41 GiB) is gone


# ---------------------------------------------------------------------------------------------------------------------
@unittest.skipUnless(GPU_TESTS, GPU_SKIP)
class TestGpuDecodeForms(CustomTestCase):
    """WRITTEN, LOCKED (GGUF_GPU_TESTS=1, one card, a gpuq window).  Reference = gguf-py dequantisation (float32) + torch -- NOT the wheel.

    They run THE PATH THE IMAGE TAKES: with the marker absent that is the JIT / torch route, with a rebuilt wheel the wheel kernels -- the asserts are
    the same format bounds either way (relative RMS < 1 %, max < 5 % of the output RMS, the G11 production bounds).
    """

    @classmethod
    def setUpClass(cls):
        assert torch.cuda.is_available()
        from sglang.srt.layers.quantization import gguf as G

        cls.G = G
        cls.dev = torch.device("cuda")
        G.iq_mmq_prepare(IQ3_S, IQ4_NL)  # arms the JIT (or refuses by name -> the test errors, as the load would)

    def _ref_w(self, raw, qtype):
        return torch.from_numpy(np.ascontiguousarray(gguf_lib.quants.dequantize(raw, gguf_lib.GGMLQuantizationType(qtype)))).to(self.dev).float()

    def test_wheel_marker_matches_the_observed_wheel_grid(self):
        """The marker is honest: a crafted block through the WHEEL ggml_dequantize gives 15 with the marker and 15.5 without."""
        from sgl_kernel.quantization import ggml_dequantize

        raw = np.zeros((1, 110), np.uint8)
        raw[0, 0:2] = np.array([1.0], np.float16).view(np.uint8)
        idx15 = int(np.argwhere(ggml_grid_x4()[:, 0] == 60)[0][0])
        raw[0, 2] = idx15 & 0xFF
        raw[0, 66] = (idx15 >> 8) & 1
        w = torch.from_numpy(raw).to(self.dev)
        out = float(ggml_dequantize(w, IQ3_S, 1, 256, torch.float32).reshape(-1)[0])
        self.assertEqual(out, 15.0 if self.G.IQ3S_WHEEL_GRID_FIXED else 15.5)

    def test_torch_dequant_equals_gguf_py_on_the_card(self):
        raw = synth(64, 1024, seed=5)
        got = DQ.dequantize_iq3_s(torch.from_numpy(raw).to(self.dev), torch.float32).cpu().numpy()
        np.testing.assert_array_equal(got, reference_rows(raw))

    def test_wheel_dequantize_equals_gguf_py_when_the_wheel_carries_the_marker(self):
        if not self.G.IQ3S_WHEEL_GRID_FIXED:
            self.skipTest("wheel without ggml_iq3s_grid_ggml: its IQ3_S kernels are known wrong and unused (see test_wheel_marker_matches_the_observed_wheel_grid)")
        from sgl_kernel.quantization import ggml_dequantize

        raw = synth(64, 1024, seed=6)
        got = ggml_dequantize(torch.from_numpy(raw).to(self.dev), IQ3_S, 64, 1024, torch.float32).cpu().numpy()
        np.testing.assert_allclose(got, reference_rows(raw), rtol=1e-6, atol=1e-7)

    def test_dense_decode_and_verify_forms_through_fused_mul_mat_gguf(self):
        """M = 1 (decode), 4 (MTP verify), 8, 16, 17, 40 (dequant branch): synthetic IQ3_S and a real NF block (layer 4 up), K = 2560."""
        cases = [("synthetic", synth(256, 2560, seed=21))]
        if NF_OK:
            reader = gguf_lib.GGUFReader(str(_g9.NF_GGUF_DIR / "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf"))
            ten = {x.name: x for x in reader.tensors}["blk.4.ffn_up_exps.weight"]
            cases.append(("real", np.ascontiguousarray(ten.data[2, :256])))
        for name, raw in cases:
            wd = self._ref_w(raw, IQ3_S)
            w = torch.from_numpy(raw).to(self.dev)
            for m in (1, 2, 4, 8, 16, 17, 40):
                with self.subTest(case=name, m=m):
                    torch.manual_seed(m)
                    x = torch.randn((m, 2560), dtype=torch.bfloat16, device=self.dev)
                    y = self.G.fused_mul_mat_gguf(x, w, IQ3_S).float()
                    ref = x.float() @ wd.T
                    self.assertTrue(torch.isfinite(y).all())
                    rms = float(ref.pow(2).mean().sqrt())
                    self.assertLess(float(((y - ref).pow(2).mean().sqrt()) / rms), 0.01)
                    self.assertLess(float((y - ref).abs().max()) / rms, 0.05)

    def _moe_decode_case(self, m: int, layer: int):
        """The full fused_moe_gguf (two GEMMs + silu_and_mul + weighted sum) in the NF production shape: E = 16 of 512 (real blocks), 2560 -> 640 -> 2560,
        top_k 10.  Down type = the real one of the layer (layer 0 IQ4_NL, layer 4 Q8_0).  Reference: gguf-py weights, float32, no activation rounding."""
        reader = gguf_lib.GGUFReader(str(_g9.NF_GGUF_DIR / "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf"))
        T = {x.name: x for x in reader.tensors}
        E, top_k = 16, 10
        g, u, d = (np.ascontiguousarray(T[f"blk.{layer}.ffn_{n}_exps.weight"].data[:E]) for n in ("gate", "up", "down"))
        qd = int(T[f"blk.{layer}.ffn_down_exps.weight"].tensor_type)
        self.G.iq_mmq_prepare(IQ3_S, qd)
        w1 = torch.from_numpy(np.concatenate([g, u], axis=1)).to(self.dev)
        w2 = torch.from_numpy(d).to(self.dev)
        wd1 = torch.stack([self._ref_w(np.concatenate([g[e], u[e]], axis=0), IQ3_S) for e in range(E)])
        wd2 = torch.stack([self._ref_w(d[e], qd) for e in range(E)])
        torch.manual_seed(100 + m)
        x = torch.randn((m, 2560), dtype=torch.bfloat16, device=self.dev)
        ids = torch.topk(torch.rand((m, E), device=self.dev), top_k, dim=1).indices.to(torch.int32)
        tw = torch.softmax(torch.randn((m, top_k), device=self.dev), -1).to(torch.bfloat16)
        y = self.G.fused_moe_gguf(x, w1, w2, tw, ids, IQ3_S, qd, "silu").float()
        # G13: the reference in TOKEN CHUNKS (``moe_reference``): the one-shot gather ``wd1[ids]`` is
        # m x top_k x 1280 x 2560 fp32 = 24.41 GiB at m = 200 and did not fit a 20 GB card.
        ref = moe_reference(x, wd1, wd2, tw, ids)
        self.assertTrue(torch.isfinite(y).all())
        return y, ref

    @unittest.skipUnless(NF_OK, "NF UD-IQ4_XS GGUF header files not on this host")
    def test_moe_decode_form_real_nf_blocks(self):
        """M = 1..8 (decode / MTP verify), M = 127 (below the PR's token floor), M = 200 (the G11 MMQ window): layer 0 (IQ3_S + IQ4_NL), layer 4 (IQ3_S + Q8_0).
        Bound: relative RMS < 1.8 % of the output RMS.  CPU chain model (two q8_1 stages, bf16 stages, real blocks, M = 1..8, uniform and normal activations,
        layers 0 and 4): 1.00-1.25 % with the ggml grid, 2.2-2.9 % with the old one -- the bound sits between; unmeasured on metal."""
        for layer in (0, 4):
            for m in (1, 2, 4, 8, 127, 200):
                with self.subTest(layer=layer, m=m):
                    y, ref = self._moe_decode_case(m, layer)
                    rms = float(ref.pow(2).mean().sqrt())
                    self.assertLess(float((y - ref).pow(2).mean().sqrt()) / rms, 0.018)


if __name__ == "__main__":
    unittest.main()
