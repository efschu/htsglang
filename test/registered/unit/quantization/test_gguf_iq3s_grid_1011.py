# SPDX-License-Identifier: Apache-2.0
"""GGUF-NF G11 (2026-10-11): the IQ3_S grid of the vendored IQ-MMQ kernel, reproduced and pinned on the desk (no GPU, no card).

Stage 0 on metal (image rc12z30y9int24egguf10, nvcc 13.0.88, sm86 and sm120, bit-identical) failed ONE case:
``test_moe_synthetic_all_types[IQ3_S]`` (k=512, rows=96, E=8, 130 tokens, top_k=4, bf16): 30 of 49920 values outside
``atol 1.0 / rtol 0.1``, greatest absolute difference 1.7513828 at (55, 1, 14), greatest relative difference 9.8544 at (62, 2, 74).
The G9 CPU model (exact dequantised weights, q8_1 activation rounding) had predicted a pass.

Root cause (this file): the kernel decoded IQ3_S with the vendored ``iq3xs_grid`` (bytes 4,12,...,52,62 = ggml's pre-final table) and the
scale ``d * (0.5 + s) * 0.5``.  The ggml format (gguf-py ``IQ3_S.grid_map``, ggml ``iq3s_grid``) has the grid 1,3,...,13,15 and the scale
``d * (1 + 2 s)``; ``(1 + 2 s) / 4 * {4,...,52,62}`` equals the format for 1..13 but turns every grid value 15 into 62 / 4 = 15.5 (+3.3 %).
Fix: ``iq3s_grid_ggml`` in ``iq_mmq_tiles.cuh`` (same indices, 62 -> 60).

What the tests do:
* a CPU MODEL of the kernel's MoE MMQ (kernel integer weights from the grid the kernel really looks up, q8_1 as ``quantize_q8_1``,
  integer dot per 32, the float order of ``vec_dot_iq3_s_q8_1_mul_mat``, bf16 output) and an emulation of torch's CUDA Philox for
  ``torch.manual_seed(1); torch.rand(130, 512, bf16, cuda); torch.randint(0, 8, (130, 4), int32, cuda)`` -- so the test case of the metal
  run is reproduced with ITS inputs: with the vendored table 30 / 49920 values, 1.7514 at (55, 1, 14), 9.85 at (62, 2, 74); with the
  shipped table 0 values;
* the table pins (shipped = ggml x 4, vendored differs only at the byte 62, the four other grids of the vendored kernels equal gguf-py);
* the production form (real NF IQ3_S expert blocks, 192 tokens, top_k 10, K = 2560): relative RMS of the model with the vendored table is
  ~1.7 %, with the shipped one ~0.5 % -- the bound used here (1 %) sits between (unmeasured on metal, CPU-model values).

Usage:  /spinning/gpu-arb/pytest_gedeckelt.sh test/registered/unit/quantization/test_gguf_iq3s_grid_1011.py
"""

from __future__ import annotations

import importlib.util
import os
import re
import unittest
from pathlib import Path

import gguf as gguf_lib
import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

_ROOT = Path(__file__).resolve().parents[4]
_CSRC = _ROOT / "python" / "sglang" / "jit_kernel" / "csrc" / "gguf_iq_mmq"
_HEADER = _CSRC / "iq_mmq_ggml_common.h"
_TILES = _CSRC / "iq_mmq_tiles.cuh"

_spec = importlib.util.spec_from_file_location("_gguf_iq_mmq_1009_for_g11", Path(__file__).with_name("test_gguf_iq_mmq_1009.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
_synth_blocks = _base._synth_blocks  # THE block generator of the metal test (numpy seeds, so the weights are bit-identical)

IQ3_S = 21
NF_GGUF_DIR = _base.NF_GGUF_DIR

# the stage-0 metal numbers (done/gguf-stufe0-g10-1011/gputests_sm86.txt and gputests_sm120.txt, identical)
METAL = dict(
    mismatched=30,
    total=49920,
    max_abs=1.751382827758789,
    max_abs_at=(55, 1, 14),
    max_rel=9.854445457458496,
    max_rel_at=(62, 2, 74),
)


# ---------------------------------------------------------------------------------------------------------------------
# tables
def _table_from(path: Path, name: str) -> np.ndarray:
    text = path.read_text()
    m = re.search(r"\b%s\[\d+\]\s*=\s*\{(.*?)\};" % re.escape(name), text, re.S)
    assert m, f"{name} not found in {path.name}"
    return np.array([int(x, 16) for x in re.findall(r"0x[0-9a-fA-F]{8}", m.group(1))], dtype=np.uint32)


def _grid_bytes(table: np.ndarray) -> np.ndarray:
    return table.view(np.uint8).reshape(len(table), 4).astype(np.int32)


def vendored_grid() -> np.ndarray:
    """iq3xs_grid of the verbatim ggml-common prefix (what the kernel used before G11)."""
    return _grid_bytes(_table_from(_HEADER, "iq3xs_grid"))


def shipped_grid() -> np.ndarray:
    """The table the IQ3_S vec_dot of iq_mmq_tiles.cuh really indexes (read from its source line)."""
    src = _TILES.read_text()
    names = set(re.findall(r"(\w+)\[__byte_perm\(qs_word", src))
    assert len(names) == 1, names
    name = names.pop()
    return _grid_bytes(_table_from(_HEADER if name == "iq3xs_grid" else _TILES, name))


def ggml_grid_x4() -> np.ndarray:
    gguf_lib.quants.IQ3_S.init_grid()
    return (np.array(gguf_lib.quants.IQ3_S.grid).reshape(512, 4) * 4).astype(np.int32)


# ---------------------------------------------------------------------------------------------------------------------
# CPU model of the kernel
def split_blocks(raw: np.ndarray):
    b = raw.reshape(raw.shape[0], -1, 110)
    d = np.ascontiguousarray(b[:, :, 0:2]).view(np.float16)[:, :, 0].astype(np.float32)
    return d, b[:, :, 2:66], b[:, :, 66:74], b[:, :, 74:106], b[:, :, 106:110]


def kernel_weights(raw: np.ndarray, grid: np.ndarray):
    """What vec_dot_iq3_s_q8_1_mul_mat builds from one row-block array [rows, nb * 110]: integer weights (grid byte * sign), the
    per-32 factor ``x_d * sub_scale * 0.5`` (sub_scale = 0.5 + 4-bit scale), float32 as in the kernel."""
    d, qs, qh, signs, scales = split_blocks(raw)
    rows, nb = d.shape
    w = np.zeros((rows, nb, 8, 4, 8), np.int32)
    for ib32 in range(8):
        qh_val = qh[:, :, ib32].astype(np.int32)
        for l in range(4):
            i1 = qs[:, :, 8 * ib32 + 2 * l].astype(np.int32) | ((qh_val << (8 - 2 * l)) & 256)
            i2 = qs[:, :, 8 * ib32 + 2 * l + 1].astype(np.int32) | ((qh_val << (7 - 2 * l)) & 256)
            sb = signs[:, :, 4 * ib32 + l].astype(np.int32)
            sg = np.stack([np.where(sb & (1 << j), -1, 1) for j in range(8)], -1)
            w[:, :, ib32, l, 0:4] = grid[i1] * sg[..., 0:4]
            w[:, :, ib32, l, 4:8] = grid[i2] * sg[..., 4:8]
    sub = np.zeros((rows, nb, 8), np.float32)
    for ib32 in range(8):
        sub[:, :, ib32] = np.float32(0.5) + ((scales[:, :, ib32 // 2] >> (4 * (ib32 % 2))) & 0xF).astype(np.float32)
    fac = (d[:, :, None] * sub) * np.float32(0.5)  # x_d * sub_scale * 0.5
    return w.reshape(rows, nb * 8, 32), fac.reshape(rows, nb * 8)


def q8_1(x: np.ndarray):
    """quantize_q8_1 of gguf_iq_mmq.cuh: d = amax / 127 in fp32, q = roundf(x / d) (half away from zero), the stored scale is fp16(d)."""
    m, k = x.shape
    xb = x.reshape(m, k // 32, 32).astype(np.float32)
    amax = np.abs(xb).max(-1, keepdims=True)
    d = amax / np.float32(127)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(amax == 0, 0, xb / d)
    q = np.where(r >= 0, np.floor(r + 0.5), -np.floor(-r + 0.5)).astype(np.int32)
    return q, d[..., 0].astype(np.float16).astype(np.float32)


def moe_model(x: np.ndarray, topk_ids: np.ndarray, raws: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Kernel MoE MMQ on the CPU: y[t, j, row] (float32, before the bf16 output cast).  Integer dot per 32 is exact (float64 holds it)."""
    tokens, k = x.shape
    E, rows = raws.shape[0], raws.shape[1]
    xq, dy = q8_1(x)
    nsb = k // 32
    out = np.zeros((tokens, topk_ids.shape[1], rows), np.float32)
    for e in range(E):
        tt, jj = np.nonzero(topk_ids == e)
        if len(tt) == 0:
            continue
        w, fac = kernel_weights(raws[e], grid)
        sumi = torch.einsum(
            "rsk,tsk->trs", torch.from_numpy(w).double(), torch.from_numpy(xq[tt].reshape(len(tt), nsb, 32)).double()
        ).numpy()
        acc = np.zeros((len(tt), rows), np.float32)
        for s in range(nsb):  # the float order of the kernel: ((x_d * sub * 0.5) * y_d) * sumi, summed in sub-block order
            acc = acc + ((fac[None, :, s] * dy[tt][:, None, s]) * sumi[:, :, s].astype(np.float32)).astype(np.float32)
        out[tt, jj] = acc
    return out


def reference_weights(raws: np.ndarray) -> torch.Tensor:
    return torch.stack(
        [torch.from_numpy(np.ascontiguousarray(gguf_lib.quants.dequantize(raws[e], gguf_lib.GGMLQuantizationType.IQ3_S))).float() for e in range(raws.shape[0])]
    )


def bf16(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(a, np.float32)).to(torch.bfloat16).float()


# ---------------------------------------------------------------------------------------------------------------------
# torch's CUDA Philox path for the activations / routing of the metal test (no GPU needed)
_M0, _M1, _W0, _W1 = 0xD2511F53, 0xCD9E8D57, 0x9E3779B9, 0xBB67AE85


def _philox_first_word(numel: int, seed: int, offset: int) -> np.ndarray:
    """Philox4x32-10 as curand_init(seed, subsequence = idx, offset) + curand4(): first output word of thread idx, counter (offset / 4, 0, idx, 0)."""
    c0 = np.full(numel, offset // 4, np.uint32)
    c1 = np.zeros(numel, np.uint32)
    c2 = np.arange(numel, dtype=np.uint32)
    c3 = np.zeros(numel, np.uint32)
    k0, k1 = seed & 0xFFFFFFFF, seed >> 32
    mask = np.uint64(0xFFFFFFFF)
    for r in range(10):
        p0 = np.uint64(_M0) * c0.astype(np.uint64)
        p1 = np.uint64(_M1) * c2.astype(np.uint64)
        hi0, lo0 = (p0 >> np.uint64(32)).astype(np.uint32), (p0 & mask).astype(np.uint32)
        hi1, lo1 = (p1 >> np.uint64(32)).astype(np.uint32), (p1 & mask).astype(np.uint32)
        c0, c1, c2, c3 = hi1 ^ c1 ^ np.uint32(k0), lo1, hi0 ^ c3 ^ np.uint32(k1), lo0
        if r < 9:
            k0, k1 = (k0 + _W0) & 0xFFFFFFFF, (k1 + _W1) & 0xFFFFFFFF
    return c0


def cuda_rand_bf16(shape, seed: int, offset: int) -> np.ndarray:
    """torch.rand(shape, dtype=bfloat16, device='cuda') right after torch.manual_seed(seed) (offset 0 = the first RNG op).
    numel <= 256 * grid on every card, so only the first word of each thread is used."""
    n = int(np.prod(shape))
    u = _philox_first_word(n, seed, offset).astype(np.float32) * np.float32(2.3283064e-10) + np.float32(1.1641532e-10)
    v = bf16(u).numpy()
    return np.where(v == np.float32(1.0), np.float32(0.0), v).reshape(shape)


def cuda_randint_i32(shape, high: int, seed: int, offset: int) -> np.ndarray:
    n = int(np.prod(shape))
    return (_philox_first_word(n, seed, offset) % np.uint32(high)).astype(np.int64).reshape(shape)


# the metal case: torch.manual_seed(1); x = rand(130, 512); topk_ids = randint(0, 8, (130, 4))  (the second RNG op: offset 4)
def metal_case():
    tokens, k, rows, E, top_k = 130, 512, 96, 8, 4
    raws = np.stack([_synth_blocks(IQ3_S, rows, k, seed=100 + e) for e in range(E)])
    x = torch.from_numpy(cuda_rand_bf16((tokens, k), 1, 0))
    ids = torch.from_numpy(cuda_randint_i32((tokens, top_k), E, 1, 4))
    wd = reference_weights(raws)
    ref = torch.einsum("tk,tjnk->tjn", x, wd[ids])
    return raws, x, ids, ref


def _assert_close_report(y: torch.Tensor, ref: torch.Tensor, atol=1.0, rtol=1e-1):
    """What torch.testing.assert_close reports: the greatest differences are taken over the MISMATCHED elements only."""
    d = (y - ref).abs()
    bad = d > atol + rtol * ref.abs()
    dm = torch.where(bad, d, torch.zeros_like(d))
    rm = torch.where(bad, d / ref.abs(), torch.zeros_like(d))
    at = lambda t: tuple(int(a) for a in np.unravel_index(int(t.argmax()), t.shape))  # noqa: E731
    return int(bad.sum()), float(dm.max()), at(dm), float(rm.max()), at(rm)


# ---------------------------------------------------------------------------------------------------------------------
class TestIq3sGridTable(CustomTestCase):
    def test_shipped_table_is_the_ggml_grid_times_four_index_for_index(self):
        np.testing.assert_array_equal(shipped_grid(), ggml_grid_x4())

    def test_ggml_values_are_the_odd_numbers_1_to_15(self):
        gguf_lib.quants.IQ3_S.init_grid()
        self.assertEqual(sorted(set(np.array(gguf_lib.quants.IQ3_S.grid).astype(int).flatten().tolist())), [1, 3, 5, 7, 9, 11, 13, 15])
        self.assertEqual(sorted(set(shipped_grid().flatten().tolist())), [4, 12, 20, 28, 36, 44, 52, 60])

    def test_the_vendored_table_differs_only_at_the_byte_62(self):
        v, s = vendored_grid(), shipped_grid()
        self.assertEqual(sorted(set(v.flatten().tolist())), [4, 12, 20, 28, 36, 44, 52, 62])
        diff = v != s
        self.assertTrue(((v[diff] == 62) & (s[diff] == 60)).all())
        self.assertEqual(int(diff.sum()), int((v == 62).sum()))
        self.assertEqual(int(diff.sum()), 150)  # bytes of 2048 (7.3 %)

    def test_vec_dot_looks_up_the_ggml_table_and_keeps_the_scale_form_the_model_assumes(self):
        src = _TILES.read_text()
        self.assertEqual(src.count("iq3s_grid_ggml[__byte_perm("), 2)
        self.assertNotIn("iq3xs_grid[", src)
        self.assertEqual(src.count("__byte_perm("), 4)  # IQ1_S 1, IQ2_S 1, IQ3_S 2 (test_gguf_iq_mmq_1009)
        self.assertIn("const float sub_scale = 0.5f + ((sc_bytes[ib32 / 2] >> (4 * (ib32 % 2))) & 0xf);", src)
        self.assertIn("x_dmf[i * (WARP_SIZE_GGUF / QI_IQ3_S_MMQ) + i / QI_IQ3_S_MMQ] * sub_scale * 0.5f * ds8.x * sumi;", src)

    def test_the_other_grids_of_the_vendored_kernels_equal_gguf_py(self):
        """Audit, not a fix: IQ3_XXS / IQ2_XXS / IQ2_XS / IQ2_S tables are value- and index-identical to gguf-py (only IQ3_S was off)."""
        for cls, name, bits in (
            (gguf_lib.quants.IQ3_XXS, "iq3xxs_grid", 32),
            (gguf_lib.quants.IQ2_XXS, "iq2xxs_grid", 64),
            (gguf_lib.quants.IQ2_XS, "iq2xs_grid", 64),
            (gguf_lib.quants.IQ2_S, "iq2s_grid", 64),
        ):
            cls.init_grid()
            g = np.array(cls.grid).astype(np.int64)
            text = _HEADER.read_text()
            m = re.search(r"\b%s\[\d+\]\s*=\s*\{(.*?)\};" % name, text, re.S)
            vals = np.array([int(x, 16) for x in re.findall(r"0x[0-9a-fA-F]+", m.group(1))], dtype=np.uint32 if bits == 32 else np.uint64)
            k = vals.view(np.uint8).reshape(len(vals), -1).astype(np.int64)
            np.testing.assert_array_equal(k, g.reshape(len(vals), -1), err_msg=name)


class TestKernelWeightsEqualTheFormat(CustomTestCase):
    """The weights the kernel builds == gguf-py dequantisation (the format), for the synthetic blocks of the metal test."""

    def _effective(self, grid):
        raws = np.stack([_synth_blocks(IQ3_S, 96, 512, seed=100 + e) for e in range(8)])
        ref = reference_weights(raws).numpy()
        got = []
        for e in range(8):
            w, fac = kernel_weights(raws[e], grid)
            got.append((w.astype(np.float32) * fac[:, :, None]).reshape(96, 512))
        return np.stack(got), ref

    def test_shipped_table_gives_the_format(self):
        got, ref = self._effective(shipped_grid())
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-7)

    def test_vendored_table_gives_plus_3_3_percent_on_grid_value_15_only(self):
        got, ref = self._effective(vendored_grid())
        self.assertGreater(float(np.abs(got - ref).max()), 0.1)
        wrong = np.abs(got - ref) > 1e-6
        self.assertGreater(float(wrong.mean()), 0.05)  # ~1/8 of the weights are grid value 15
        self.assertLess(float(wrong.mean()), 0.12)
        rel = (got[wrong] - ref[wrong]) / ref[wrong]
        np.testing.assert_allclose(rel, 62 / 60 - 1, rtol=2e-3)


class TestMetalStage0FailureIsReproduced(CustomTestCase):
    """The stage-0 case with ITS inputs (seed 1 Philox activations / routing, numpy-seeded weights)."""

    @classmethod
    def setUpClass(cls):
        cls.raws, cls.x, cls.ids, cls.ref = metal_case()

    def _y(self, grid):
        return bf16(moe_model(self.x.numpy(), self.ids.numpy(), self.raws, grid))

    def test_philox_emulation_looks_like_the_cuda_inputs(self):
        self.assertAlmostEqual(float(self.x.mean()), 0.5, delta=0.01)
        self.assertGreaterEqual(float(self.x.min()), 0.0)
        self.assertLess(float(self.x.max()), 1.0)
        self.assertEqual(sorted(set(self.ids.flatten().tolist())), list(range(8)))

    def test_vendored_table_reproduces_the_metal_numbers(self):
        n, max_abs, at_abs, max_rel, at_rel = _assert_close_report(self._y(vendored_grid()), self.ref)
        self.assertEqual((n, self.ref.numel()), (METAL["mismatched"], METAL["total"]))
        self.assertEqual(at_abs, METAL["max_abs_at"])
        self.assertEqual(at_rel, METAL["max_rel_at"])
        self.assertAlmostEqual(max_abs, METAL["max_abs"], delta=1e-4)  # metal ref = cuBLAS fp32, here CPU fp32
        self.assertAlmostEqual(max_rel, METAL["max_rel"], delta=5e-3)

    def test_shipped_table_passes_the_pr_bound_on_the_same_case(self):
        y = self._y(shipped_grid())
        n, *_ = _assert_close_report(y, self.ref)
        self.assertEqual(n, 0)
        torch.testing.assert_close(y, self.ref, atol=1.0, rtol=1e-1)
        self.assertLess(float((y - self.ref).abs().max()), 0.7)  # the q8_1 + bf16 floor (model: ~0.56) -- 1.75 before the fix

    def test_the_outliers_cluster_on_a_few_expert_rows_with_many_grid_15_weights(self):
        """Not a random kernel fault: the same few (expert, row) pairs again and again -- expert 3 row 74 alone carries 11 of the 30."""
        y = self._y(vendored_grid())
        bad = ((y - self.ref).abs() > 1.0 + 0.1 * self.ref.abs()).nonzero()
        pairs = [(int(self.ids[t, j]), int(r)) for t, j, r in bad.tolist()]
        self.assertEqual(len(pairs), 30)
        self.assertEqual(pairs.count((3, 74)), 11)
        self.assertEqual(len(set(pairs)), 11)


@unittest.skipUnless((NF_GGUF_DIR / "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf").is_file(), "NF UD-IQ4_XS GGUF header files not on this host")
class TestProductionFormRealBlocks(CustomTestCase):
    """Real NF IQ3_S expert blocks (layer 0 gate, layer 4 up), K = 2560, 192 tokens, top_k 10 (distinct experts), E = 16 of the 512.

    The PR bound (atol 1.0) is blind here: real blocks have w_rms ~0.013, outputs ~0.4, so every error below 1 passes.  A bound relative to
    the output scale is what sees the grid defect: CPU model, relative RMS 1.6-1.7 % with the vendored table, 0.4-0.6 % with the shipped one.
    """

    ROWS, E, TOKENS, TOP_K, K = 96, 16, 192, 10, 2560

    @classmethod
    def setUpClass(cls):
        reader = gguf_lib.GGUFReader(str(NF_GGUF_DIR / "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf"))
        t = {x.name: x for x in reader.tensors}
        cls.raw = {}
        for layer, which in ((0, "gate"), (4, "up")):
            ten = t[f"blk.{layer}.ffn_{which}_exps.weight"]
            assert int(ten.tensor_type) == IQ3_S and list(ten.shape) == [2560, 640, 512]
            cls.raw[(layer, which)] = np.ascontiguousarray(ten.data[: cls.E, : cls.ROWS])
        del reader

    def _rel_rms_and_max(self, key, act, grid):
        raws = self.raw[key]
        g = torch.Generator().manual_seed(3)
        x = (torch.rand((self.TOKENS, self.K), generator=g) if act == "uniform" else torch.randn((self.TOKENS, self.K), generator=g)).to(torch.bfloat16).float()
        ids = torch.stack([torch.randperm(self.E, generator=g)[: self.TOP_K] for _ in range(self.TOKENS)])
        ref = torch.einsum("tk,tjnk->tjn", x, reference_weights(raws)[ids])
        err = bf16(moe_model(x.numpy(), ids.numpy(), raws, grid)) - ref
        rms = float(ref.pow(2).mean().sqrt())
        return float(err.pow(2).mean().sqrt()) / rms, float(err.abs().max()) / rms

    def test_shipped_table_is_inside_the_production_bound_and_the_vendored_one_is_outside(self):
        for key in self.raw:
            for act in ("uniform", "normal"):
                with self.subTest(layer=key[0], which=key[1], act=act):
                    rel, mx = self._rel_rms_and_max(key, act, shipped_grid())
                    self.assertLess(rel, 0.01, f"shipped: relative RMS {rel:.4f}")
                    self.assertLess(mx, 0.05, f"shipped: max |err| / ref rms {mx:.4f}")
                    rel_v, mx_v = self._rel_rms_and_max(key, act, vendored_grid())
                    self.assertGreater(rel_v, 0.012, f"vendored table would not show: relative RMS {rel_v:.4f}")
                    self.assertGreater(mx_v, 0.06, f"vendored table would not show: max |err| / ref rms {mx_v:.4f}")


class TestGpuReferenceHelper(CustomTestCase):
    def test_per_expert_reference_equals_the_gathering_einsum(self):
        torch.manual_seed(0)
        tokens, top_k, E, rows, k = 20, 3, 5, 7, 16
        x = torch.rand((tokens, k))
        wd = torch.randn((E, rows, k))
        ids = torch.stack([torch.randperm(E)[:top_k] for _ in range(tokens)]).int()
        want = torch.einsum("tk,tjnk->tjn", x, wd[ids.long()])
        torch.testing.assert_close(_base._moe_ref_by_expert(x, wd, ids), want, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
