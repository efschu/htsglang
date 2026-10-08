"""H88-A: Marlin W4A8 (int4 weights x int8 activations) -- kernel port tests.

Port: vLLM PR #24722 (jinzhen-lin; Apache-2.0; Marlin (C) 2024 Elias Frantar,
IST-DASLab) into the htsglang JIT kernels, new files only:
  csrc/gemm/marlin_a8/      dense GEMM + repack  (+ shared device headers)
  csrc/gemm/marlin_a8_moe/  MoE GEMM
  gptq_marlin.py / gptq_marlin_repack.py / awq_marlin_repack.py /
  moe_wna16_marlin.py       new functions appended below the unchanged A16 code
  marlin_w4a8_utils.py      format contract + CPU reference

DESK PART (no GPU, no CUDA call, runs under pytest_gedeckelt.sh):
  * CPU layout/scale/zero-point reference and its invariants (g32 and g128,
    sym uint4b8 and asym uint4+zp, channelwise),
  * the kernel arithmetic emulated in fp64 against (dequant(W) . quant_int8(A)),
  * the Python wrappers' shape/dtype contract,
  * the kernel instance table (96 kernels per module: 2 weight types x 4 group
    sizes x 12 thread configs) against the vLLM generator rules,
  * the scalar-type ids Python hands to the kernel equal the C++ ids,
  * A16 unchanged: pinned sha256 of every A16 source, the Python wrappers' A16
    prefix is a byte prefix of the current files, A16 signatures pinned.

GPU PART (skipped unless torch.cuda.is_available() and H88_GPU_TESTS=1; to be
run by the NF seat inside its metal window, one card per run):

    H88_GPU_TESTS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<nvml idx> \\
      PYTHONPATH=python:sgl-kernel/python python3 -m pytest -q -x \\
      python/sglang/jit_kernel/tests/test_marlin_w4a8_1007.py -k Gpu

  run once on an RTX 3080 (sm86) and once on the RTX 5090 (sm120). First run per
  (module, dtype, arch) JITs the kernel: expect minutes (see the build-time
  estimate in the H88-A report), use TVM_FFI_CACHE_DIR to keep the cache.
  Expected tolerances (same criterion as upstream vLLM test_marlin_gemm):
    * repack kernels: bit-exact against marlin_weights(..., is_a_8bit=True)
    * GEMM: mean|out - ref| / mean|ref| < 0.04 with ref = (a_q * a_scale) @ w_ref
      in fp64 (typical observed upstream: ~0.005-0.02; bf16 is the looser one)
    * MoE: same criterion per output row.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import itertools
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

_JIT = Path(__file__).resolve().parents[1]
_REPO = _JIT.parents[2]
_CSRC = _JIT / "csrc" / "gemm"

# make the module importable without importing the whole sglang package (the
# utils module is pure torch/numpy and has no sglang import)
sys.path.insert(0, str(_JIT))
import marlin_w4a8_utils as U  # noqa: E402

BASE_COMMIT = "6cb6580482"  # desk/nf-vorlauf-hebel-1007, the A16 baseline

GPU_TESTS = bool(torch.cuda.is_available() and os.environ.get("H88_GPU_TESTS") == "1")
GPU_SKIP_REASON = "needs a GPU window: set H88_GPU_TESTS=1 with CUDA_VISIBLE_DEVICES (see module docstring)"


# ---------------------------------------------------------------------------
# helpers shared by the CPU tests and the GPU tests (builders run on CPU)
# ---------------------------------------------------------------------------


def awq_pack(q_w: torch.Tensor, num_bits: int, size_k: int, size_n: int) -> torch.Tensor:
    """AWQ checkpoint packing: interleave the columns, pack along N."""
    assert num_bits == 4
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    q = q_w.reshape((-1, len(interleave)))[:, interleave].ravel().reshape((-1, size_n)).contiguous()
    return U.pack_cols(q, num_bits, size_k, size_n)


def gptq_pack(q_w: torch.Tensor, num_bits: int, size_k: int, size_n: int) -> torch.Tensor:
    """GPTQ checkpoint packing: pack along K."""
    assert num_bits == 4 and size_k % 8 == 0
    q = q_w.cpu().numpy().astype(np.uint32)
    res = np.zeros((size_k // 8, size_n), dtype=np.uint32)
    for i in range(8):
        res |= q[i::8, :] << (4 * i)
    return torch.from_numpy(res.astype(np.int32))


def build_weight(size_k, size_n, group_size, asym, dtype, seed=0):
    """Quantise a random weight and produce everything a W4A8 GEMM consumes."""
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(size_k, size_n, generator=g) * 0.05).float()
    w_ref, q_stored, s, zp = U.quantize_weights_ref(w, group_size, asym)
    # the kernel only ever sees scales in the output dtype: the reference weight
    # is built from those (as vLLM's test does by quantising in the working dtype)
    s = s.to(dtype).float()
    gs_eff = size_k if group_size == -1 else group_size
    centre = zp.repeat_interleave(gs_eff, dim=0).float() if asym else 8.0
    w_ref = (q_stored.float() - centre) * s.repeat_interleave(gs_eff, dim=0)
    perm = U.get_weight_perm(4, True)
    marlin_q = U.marlin_weights(q_stored, size_k, size_n, 4, perm, True)
    # the checkpoint-format packing the CUDA repack kernels read
    ckpt = awq_pack(q_stored, 4, size_k, size_n) if asym else gptq_pack(q_stored, 4, size_k, size_n)
    s_m = U.marlin_permute_scales(s.to(dtype), size_k, size_n, group_size, True)
    groups = s_m.shape[0]
    factor = None
    if groups > 1:
        s_m, factor = U.marlin_act_int8_process_scales(s_m)
    zp_m = U.marlin_zero_points(zp, groups, size_n, 4, True) if asym else None
    return dict(
        w=w, w_ref=w_ref, q_stored=q_stored, s=s, zp=zp, marlin_q=marlin_q, ckpt=ckpt,
        s_m=s_m, factor=factor, zp_m=zp_m, groups=groups, group_size=group_size, asym=asym,
    )  # fmt: skip


def build_activation(size_m, size_k, dtype, factor, seed=1):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(size_m, size_k, generator=g).to(dtype)
    a_q, a_scales = U.per_token_quant_int8_ref(a)
    a_scales_kernel = a_scales * factor if factor is not None else a_scales
    return a, a_q, a_scales, a_scales_kernel.float().contiguous()


def max_diff(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float((out.double() - ref.double()).abs().mean() / ref.double().abs().mean())


def moe_align_block_size_ref(topk_ids: torch.Tensor, block_size: int, num_experts: int):
    """vLLM moe_align_block_size(..., ignore_invalid_experts=True) semantics:
    experts without tokens get no block, padding entries hold size_m * top_k."""
    flat = topk_ids.reshape(-1)
    sentinel = flat.numel()
    sorted_ids, expert_ids = [], []
    for e in range(num_experts):
        ids = torch.nonzero(flat == e).flatten().tolist()
        if not ids:
            continue
        pad = (-len(ids)) % block_size
        sorted_ids.extend(ids + [sentinel] * pad)
        expert_ids.extend([e] * ((len(ids) + pad) // block_size))
    num_post = len(sorted_ids)
    max_len = flat.numel() + num_experts * (block_size - 1)
    sorted_ids = sorted_ids + [sentinel] * (max_len - len(sorted_ids))
    return (
        torch.tensor(sorted_ids, dtype=torch.int32),
        torch.tensor(expert_ids + [0] * (max_len // block_size - len(expert_ids)), dtype=torch.int32),
        torch.tensor([num_post], dtype=torch.int32),
    )


def moe_reference(a_q, a_scales, w_refs, topk_ids, topk_weights, top_k, mul_topk_weights):
    """c[id] for id = token * top_k + k, fp64."""
    flat_e = topk_ids.reshape(-1)
    out = torch.zeros((flat_e.numel(), w_refs[0].shape[1]), dtype=torch.float64)
    for idx in range(flat_e.numel()):
        row = idx // top_k
        v = (a_q[row].double() * float(a_scales[row])) @ w_refs[int(flat_e[idx])].double()
        if mul_topk_weights:
            v = v * float(topk_weights.reshape(-1)[idx])
        out[idx] = v
    return out


# ---------------------------------------------------------------------------
# DESK: layout, scales, zero points, numerics (CPU)
# ---------------------------------------------------------------------------


class TestW4A8CpuLayout(unittest.TestCase):
    def test_weight_perm_is_a_permutation_for_both_layouts(self):
        for is_a_8bit in (True, False):
            for nb in (4, 8):
                perm = U.get_weight_perm(nb, is_a_8bit)
                self.assertEqual(perm.numel(), 1024)
                self.assertTrue(torch.equal(perm.sort().values, torch.arange(1024)))

    def test_a8_layout_differs_from_a16_layout(self):
        # the int8 kernel reads m16n8k32 fragments: different tile order
        p8 = U.get_weight_perm(4, True)
        p16 = U.get_weight_perm(4, False)
        self.assertFalse(torch.equal(p8, p16))

    def test_marlin_weights_shape_and_roundtrip(self):
        for asym, (k, n) in itertools.product((False, True), ((128, 64), (256, 128), (512, 192))):
            c = build_weight(k, n, 128 if k >= 128 else -1, asym, torch.float16)
            self.assertEqual(tuple(c["marlin_q"].shape), U.w4a8_expected_shapes(k, n, 128, asym)["b_q_weight"])
            back = U.marlin_unpack_weights(c["marlin_q"], k, n, 4, U.get_weight_perm(4, True), True)
            self.assertTrue(torch.equal(back.to(torch.int32), c["q_stored"]))

    def test_k_must_be_multiple_of_32_for_int8_tile(self):
        q = torch.zeros(48, 64, dtype=torch.int32)  # 48 % 32 != 0
        with self.assertRaises(AssertionError):
            U.marlin_weights(q, 48, 64, 4, U.get_weight_perm(4, True), True)

    def test_scale_permutation_is_a_permutation_per_group_size(self):
        for gs in (-1, 32, 128):
            k, n = 256, 128
            groups = 1 if gs == -1 else k // gs
            s = torch.arange(groups * n, dtype=torch.float32).reshape(groups, n)
            sp = U.marlin_permute_scales(s, k, n, gs, True)
            self.assertEqual(tuple(sp.shape), (groups, n))
            self.assertTrue(torch.equal(sp.flatten().sort().values, s.flatten()))
            # a8 uses the 'single' permutation for EVERY group size
            single = U.marlin_permute_scales(s, k, n, -1, True)
            self.assertTrue(torch.equal(sp, single))

    def test_scale_permutation_a16_grouped_differs(self):
        k, n = 256, 128
        s = torch.arange(2 * n, dtype=torch.float32).reshape(2, n)
        self.assertFalse(torch.equal(U.marlin_permute_scales(s, k, n, 128, False), U.marlin_permute_scales(s, k, n, 128, True)))

    def test_int16_scale_trick_error_bound(self):
        g = torch.Generator().manual_seed(3)
        for dtype in (torch.float16, torch.bfloat16):
            s = (torch.rand(4, 128, generator=g) * 0.02 + 1e-4).to(dtype)
            s_int, factor = U.marlin_act_int8_process_scales(s)
            self.assertEqual(s_int.dtype, dtype)
            ints = s_int.view(torch.int16)
            self.assertLessEqual(int(ints.max()), U.W4A8_SCALE_INT_RANGE)  # 8x overflow reserve in int16
            self.assertGreaterEqual(int(ints.min()), 0)
            dec = U.marlin_act_int8_decode_scales(s_int, factor, torch.float64)
            err = (dec - s.double()).abs()
            # round-to-nearest on a grid of one 'factor' per int step
            self.assertLessEqual(float(err.max()), 0.5 * float(factor) * (1 + 1e-6))
            self.assertAlmostEqual(float(factor), float(s.float().max()) / 4096, places=9)

    def test_zero_point_packing_roundtrip_without_interleave(self):
        k, n, gs = 256, 128, 32
        groups = k // gs
        g = torch.Generator().manual_seed(5)
        zp = torch.randint(0, 16, (groups, n), generator=g, dtype=torch.int32)
        packed = U.marlin_zero_points(zp, groups, n, 4, True)
        self.assertEqual(tuple(packed.shape), (groups, n // 8))
        unpacked = U.unpack_cols(packed, 4, groups, n)
        scale_perm, _ = U.get_scale_perms()
        inv = np.argsort(np.array(scale_perm))
        back = unpacked.reshape((-1, len(scale_perm)))[:, torch.from_numpy(inv)].reshape(groups, n)
        self.assertTrue(torch.equal(back, zp))

    def test_zero_point_a16_interleaves_a8_does_not(self):
        n, groups = 64, 2
        zp = torch.arange(groups * n, dtype=torch.int32).reshape(groups, n) % 16
        self.assertFalse(torch.equal(U.marlin_zero_points(zp, groups, n, 4, True), U.marlin_zero_points(zp, groups, n, 4, False)))


class TestW4A8CpuNumerics(unittest.TestCase):
    """dequant(W) . quant_int8(A) reference vs the kernel arithmetic."""

    CASES = list(itertools.product((-1, 32, 128), (False, True)))

    def test_emulated_kernel_matches_reference(self):
        for dtype in (torch.float16, torch.bfloat16):
            for gs, asym in self.CASES:
                k, n, m = 256, 128, 37
                c = build_weight(k, n, gs, asym, dtype, seed=gs + 7)
                a, a_q, a_scales, _ = build_activation(m, k, dtype, None)
                ref = U.reference_w4a8_gemm(a_q, a_scales, c["w_ref"])
                use_int16 = c["groups"] > 1
                # emulate on the scales exactly as handed to the kernel (permutation
                # does not change the per-column group scale: use the unpermuted ones)
                emu = U.emulate_w4a8_kernel(a_q, a_scales, c["q_stored"], c["s"], c["zp"], gs, use_int16)
                # int16 rounding of the group scales is the only difference
                # (channelwise path: exact up to fp64 summation order)
                tol = 1e-3 if use_int16 else 1e-9
                self.assertLess(max_diff(emu, ref), tol, f"{dtype} g={gs} asym={asym}")

    def test_int8_activation_error_vs_float_matmul(self):
        # end-to-end sanity: W4 + A8 stays within the budget the GPU test uses
        for gs, asym in self.CASES:
            k, n, m = 512, 128, 16
            c = build_weight(k, n, gs, asym, torch.float16, seed=11)
            a, a_q, a_scales, _ = build_activation(m, k, torch.float16, None)
            ref = U.reference_w4a8_gemm(a_q, a_scales, c["w_ref"])
            exact = a.double() @ c["w_ref"].double()
            self.assertLess(max_diff(ref, exact), 0.02)  # int8 activation rounding only

    def test_group_scale_kernel_inputs_are_consistent(self):
        # (a_scales * factor) * int16 scale == a_scales * original scale (<= int16 rounding)
        k, n, gs = 256, 128, 32
        c = build_weight(k, n, gs, False, torch.bfloat16)
        s_dec = U.marlin_act_int8_decode_scales(c["s_m"], c["factor"], torch.float64)
        s_ref = U.marlin_permute_scales(c["s"].to(torch.bfloat16), k, n, gs, True).double()
        self.assertLess(float((s_dec - s_ref).abs().max() / s_ref.max()), 1.5e-4)

    def test_moe_reference_equals_dense_reference_per_expert(self):
        e_n, m, k, n, top_k = 4, 6, 128, 64, 2
        ws = [build_weight(k, n, 32, False, torch.float16, seed=20 + e) for e in range(e_n)]
        a, a_q, a_scales, _ = build_activation(m, k, torch.float16, None)
        g = torch.Generator().manual_seed(9)
        topk_ids = torch.stack([torch.randperm(e_n, generator=g)[:top_k] for _ in range(m)]).to(torch.int32)
        topk_w = torch.rand(m, top_k, generator=g)
        out = moe_reference(a_q, a_scales, [w["w_ref"] for w in ws], topk_ids, topk_w, top_k, True)
        for idx in range(m * top_k):
            row, e = idx // top_k, int(topk_ids.reshape(-1)[idx])
            d = U.reference_w4a8_gemm(a_q[row : row + 1], a_scales[row : row + 1], ws[e]["w_ref"])[0] * float(topk_w.reshape(-1)[idx])
            self.assertTrue(torch.allclose(out[idx], d))

    def test_moe_align_ref_contract(self):
        topk_ids = torch.tensor([[0, 2], [2, 3], [0, 0], [3, 2]], dtype=torch.int32)
        s, e, npost = moe_align_block_size_ref(topk_ids, 16, 5)
        self.assertEqual(int(npost), 16 * 3)  # experts 0, 2, 3 -> one block each, expert 1 and 4 none
        self.assertEqual(e[:3].tolist(), [0, 2, 3])
        self.assertNotIn(-1, e.tolist())
        valid = s[: int(npost)]
        self.assertEqual(int((valid < 8).sum()), 8)  # every (token, k) pair exactly once
        self.assertEqual(sorted(valid[valid < 8].tolist()), list(range(8)))


# ---------------------------------------------------------------------------
# DESK: Python wrapper contract
# ---------------------------------------------------------------------------


class TestW4A8WrapperContract(unittest.TestCase):
    def _dense(self, gs=32, asym=False, m=16, k=128, n=64, dtype=torch.float16):
        c = build_weight(k, n, gs, asym, dtype)
        a, a_q, a_scales, a_sk = build_activation(m, k, dtype, c["factor"])
        return c, a_q, a_sk

    def test_expected_shapes(self):
        sh = U.w4a8_expected_shapes(2560, 640, 128, False)
        self.assertEqual(sh["b_q_weight"], (160, 1280))
        self.assertEqual(sh["b_scales"], (20, 640))
        self.assertEqual(sh["b_zeros"], (0,))
        sh = U.w4a8_expected_shapes(2560, 640, 32, True, num_experts=512)
        self.assertEqual(sh["b_q_weight"], (512, 160, 1280))
        self.assertEqual(sh["b_scales"], (512, 80, 640))
        self.assertEqual(sh["b_zeros"], (512, 80, 80))
        with self.assertRaises(ValueError):
            U.w4a8_expected_shapes(2560, 640, 48, False)  # no kernel for g48
        with self.assertRaises(ValueError):
            U.w4a8_expected_shapes(2560, 100, 128, False)  # n % 64

    def test_accepts_valid_dense_args_for_all_group_sizes_and_types(self):
        for gs, asym in itertools.product((-1, 32, 64, 128), (False, True)):
            c, a_q, a_sk = self._dense(gs, asym, k=256)
            got = U.check_w4a8_gemm_args(
                a_q, a_sk, c["marlin_q"], c["s_m"], c["zp_m"], None, "uint4" if asym else "uint4b8", 16, 64, 256
            )  # fmt: skip
            self.assertEqual(got, gs)

    def test_rejects_bad_args(self):
        c, a_q, a_sk = self._dense()
        ok = dict(a=a_q, a_scales=a_sk, b_q_weight=c["marlin_q"], b_scales=c["s_m"], b_zeros=None, b_bias=None,
                  b_q_type_name="uint4b8", size_m=16, size_n=64, size_k=128)  # fmt: skip
        U.check_w4a8_gemm_args(**ok)

        def bad(**kw):
            with self.assertRaises(ValueError, msg=str(list(kw))):
                U.check_w4a8_gemm_args(**{**ok, **kw})

        bad(a=a_q.to(torch.float16))  # activations must be int8
        bad(a_scales=a_sk.to(torch.float16))
        bad(a_scales=a_sk[:-1])
        bad(b_q_weight=c["marlin_q"][:-1])
        bad(b_q_weight=c["marlin_q"].to(torch.int64))
        bad(b_scales=c["s_m"].float())
        bad(b_scales=c["s_m"][:, :-1])
        bad(b_q_type_name="uint4")  # asymmetric type without zero points
        bad(b_zeros=torch.zeros(1, 8, dtype=torch.int32))  # zero points with uint4b8
        bad(b_q_type_name="uint8b128")
        bad(size_m=15)
        # a row stride of 144 is 16-aligned: accepted (a view into a wider buffer is fine)
        U.check_w4a8_gemm_args(**{**ok, "a": torch.zeros(16, 144, dtype=torch.int8)[:, :128]})
        # misaligned row stride: stride 136 is not a multiple of 16
        wide = torch.zeros(16, 136, dtype=torch.int8)
        bad(a=wide[:, :128])

    def test_moe_contract(self):
        e_n, k, n, m, tk = 4, 128, 64, 8, 2
        ws = [build_weight(k, n, 32, False, torch.bfloat16, seed=30 + e) for e in range(e_n)]
        s_all = torch.stack([w["s_m"] for w in ws])
        q_all = torch.stack([w["marlin_q"] for w in ws])
        a, a_q, a_scales, _ = build_activation(m, k, torch.bfloat16, None)
        U.check_w4a8_moe_args(a_q, a_scales, q_all, s_all, None, None, "uint4b8", 16, tk, m, n, k)
        for bs in (8, 24, 128):
            with self.assertRaises(ValueError):
                U.check_w4a8_moe_args(a_q, a_scales, q_all, s_all, None, None, "uint4b8", bs, tk, m, n, k)
        with self.assertRaises(ValueError):
            U.check_w4a8_moe_args(a_q, a_scales, q_all[:, :, :-8], s_all, None, None, "uint4b8", 16, tk, m, n, k)

    def test_arch_dispatch_table(self):
        for cap in ((8, 6), (12, 0)):
            ok, why = U.w4a8_arch_support(*cap)
            self.assertTrue(ok, why)
            self.assertIn(cap, U.W4A8_TESTED_ARCHS)
        self.assertTrue(U.w4a8_arch_support(8, 0)[0])
        self.assertTrue(U.w4a8_arch_support(8, 9)[0])
        self.assertFalse(U.w4a8_arch_support(7, 5)[0])
        self.assertFalse(U.w4a8_arch_support(6, 1)[0])


# ---------------------------------------------------------------------------
# DESK: source-level checks of the CUDA port (no compiler needed except cpp/g++)
# ---------------------------------------------------------------------------


def _read(p: Path) -> str:
    return p.read_text()


class TestW4A8CudaSources(unittest.TestCase):
    NEW_FILES = [
        "marlin_a8/marlin.cuh", "marlin_a8/marlin_dtypes.cuh", "marlin_a8/dequant.h", "marlin_a8/marlin_mma.h",
        "marlin_a8/marlin_template.h", "marlin_a8/kernel.h", "marlin_a8/gptq_marlin_a8.cuh",
        "marlin_a8/gptq_marlin_repack_a8.cuh", "marlin_a8/awq_marlin_repack_a8.cuh",
        "marlin_a8_moe/kernel.h", "marlin_a8_moe/marlin_template.h", "marlin_a8_moe/moe_wna16_marlin_a8.cuh",
    ]  # fmt: skip

    def test_all_new_files_exist(self):
        for f in self.NEW_FILES:
            self.assertTrue((_CSRC / f).is_file(), f)

    def test_licence_and_provenance_in_file_heads(self):
        for f in self.NEW_FILES:
            head = "\n".join(_read(_CSRC / f).splitlines()[:45])
            self.assertIn("Apache License", head, f)
            self.assertTrue("Elias Frantar" in head or "vLLM" in head, f)
        for f in ("marlin_a8/gptq_marlin_a8.cuh", "marlin_a8_moe/moe_wna16_marlin_a8.cuh"):
            head = "\n".join(_read(_CSRC / f).splitlines()[:45])
            self.assertIn("24722", head, f)
            self.assertIn("IST-DASLab", head, f)

    def test_no_torch_dependency_and_no_a16_namespace_clash(self):
        for f in self.NEW_FILES:
            # the provenance block names `vllm::` on purpose (it documents the edit)
            t = re.sub(r"// H88-A-PROVENANCE-BEGIN.*?// H88-A-PROVENANCE-END\n", "", _read(_CSRC / f), count=1, flags=re.S)
            self.assertNotIn("torch/csrc", t, f)
            self.assertNotIn("STD_TORCH_CHECK", t, f)
            if not f.endswith("_a8.cuh"):  # the hand-written wrappers name `vllm::` in their comments
                self.assertNotIn("vllm::", t, f)
            self.assertNotRegex(t, r"namespace\s+device::marlin\b", f)  # A16 namespace stays A16's
            self.assertNotRegex(t, r"namespace\s+marlin\s*\{", f)

    def test_no_a16_include_from_a8_sources(self):
        for f in self.NEW_FILES:
            for inc in re.findall(r'#include\s+"([^"]+)"', _read(_CSRC / f)):
                self.assertNotIn("marlin/", inc.replace("marlin_a8", ""), f"{f} includes {inc}")

    def test_vendored_device_code_differs_from_vllm_only_mechanically(self):
        vllm = Path("/spinning/shvllm/csrc/libtorch_stable")
        if not vllm.is_dir():
            self.skipTest("local vLLM fork not present")
        pairs = {
            "marlin_a8/marlin.cuh": "quantization/marlin/marlin.cuh",
            "marlin_a8/marlin_dtypes.cuh": "quantization/marlin/marlin_dtypes.cuh",
            "marlin_a8/dequant.h": "quantization/marlin/dequant.h",
            "marlin_a8/marlin_mma.h": "quantization/marlin/marlin_mma.h",
            "marlin_a8/marlin_template.h": "quantization/marlin/marlin_template.h",
            "marlin_a8/kernel.h": "quantization/marlin/kernel.h",
            "marlin_a8_moe/marlin_template.h": "moe/marlin_moe_wna16/marlin_template.h",
            "marlin_a8_moe/kernel.h": "moe/marlin_moe_wna16/kernel.h",
        }

        def norm(t: str) -> str:
            t = t.replace("vllm::", "host::")
            t = t.replace('#include "core/scalar_type.hpp"', "#include <sgl_kernel/scalar_type.hpp>")
            t = t.replace("libtorch_stable/quantization/marlin/", "../marlin_a8/")
            t = re.sub(r"#define MARLIN_NAMESPACE_NAME marlin_moe_wna16\b", "#define MARLIN_NAMESPACE_NAME marlin_a8_moe", t)
            t = re.sub(r"#define MARLIN_NAMESPACE_NAME marlin$", "#define MARLIN_NAMESPACE_NAME marlin_a8", t, flags=re.M)
            t = t.replace("_marlin_cuh", "_marlin_a8_cuh").replace("_data_types_cuh", "_data_types_a8_cuh")
            return t

        for mine, theirs in pairs.items():
            a = norm(_read(vllm / theirs))
            b = _read(_CSRC / mine)
            # the H88-A provenance block at the file head is the one addition
            b = re.sub(r"// H88-A-PROVENANCE-BEGIN.*?// H88-A-PROVENANCE-END\n", "", b, count=1, flags=re.S)
            if a != b:
                # a formatter may have re-wrapped lines: compare token streams
                ta, tb = re.sub(r"\s+", "", a), re.sub(r"\s+", "", b)
                self.assertEqual(ta, tb, f"{mine} deviates from {theirs} beyond the mechanical edits")

    @staticmethod
    def _expected_instances():
        """vLLM generate_kernels.py rules for a_type=kS8: thread_m_blocks 1..4,
        THREAD_CONFIGS pruned (256 threads: (128,128) for m == 1, (64,256) for m > 1)."""
        out = set()
        for group_blocks, m in itertools.product((-1, 2, 4, 8), (1, 2, 3, 4)):
            for thread_k, thread_n, threads in ((128, 128, 256), (64, 256, 256), (64, 128, 128), (128, 64, 128)):
                if threads == 256:
                    if m <= 1 and (thread_k, thread_n) != (128, 128):
                        continue
                    if m > 1 and (thread_k, thread_n) != (64, 256):
                        continue
                out.add((m, thread_n // 16, thread_k // 16, group_blocks, threads))
        return out

    def _table(self, header: str, prefix: str, root: str) -> set:
        gcc = shutil.which("gcc") or shutil.which("cc")
        if not gcc:
            self.skipTest("no C preprocessor")
        text = _read(_CSRC / header)
        start = text.index(f"#define {prefix}_GET_IF(")
        end = text.index("template <host::ScalarTypeId c_type_id>", start)
        snippet = text[start:end].replace("\\\n", " ")
        results = {}
        for b in ("host::kU4B8", "host::kU4"):
            src = snippet + f"\n{prefix}_GET_IF_ALL({b})\n"
            with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as fh:
                fh.write(src)
                path = fh.name
            try:
                pre = subprocess.run([gcc, "-E", "-P", "-x", "c", path], capture_output=True, text=True, check=True).stdout
            finally:
                os.unlink(path)
            rows = re.findall(
                r"thread_m_blocks == (\d+) && thread_n_blocks == (\d+) && thread_k_blocks == (\d+) && group_blocks == (-?\d+) && threads == (\d+)",
                pre,
            )
            results[b] = rows
            # the selected kernel is always a_type kS8, one dtype for c and s, 4 stages, no m8, no float zp
            self.assertEqual(len(re.findall(r"Marlin<\s*host::kS8\.id\(\)", pre)), len(rows))
            for tmpl in re.findall(r"Marlin<([^;]*?)>;", pre):
                parts = [p.strip() for p in tmpl.split(",")]
                self.assertEqual(parts[2], "c_type_id")
                self.assertEqual(parts[3], "c_type_id")
                self.assertEqual((parts[8], parts[9], parts[11]), ("false", "4", "false"))
        return results

    def test_dense_kernel_table(self):
        res = self._table("marlin_a8/gptq_marlin_a8.cuh", "MARLIN_A8", "dense")
        exp = self._expected_instances()
        for b, rows in res.items():
            got = {tuple(int(x) for x in r) for r in rows}
            self.assertEqual(len(rows), len(got), "duplicate instance rows")
            self.assertEqual(got, exp, b)
            self.assertEqual(len(rows), 48)

    def test_moe_kernel_table(self):
        res = self._table("marlin_a8_moe/moe_wna16_marlin_a8.cuh", "MARLIN_A8_MOE", "moe")
        exp = self._expected_instances()
        for b, rows in res.items():
            got = {tuple(int(x) for x in r) for r in rows}
            self.assertEqual(got, exp, b)
            self.assertEqual(len(rows), 48)

    def test_cpp_scalar_type_ids_equal_python_ids(self):
        cxx = shutil.which("g++") or shutil.which("c++")
        sk = _REPO / "sgl-kernel" / "python"
        if not cxx or not (sk / "sgl_kernel" / "scalar_type.py").is_file():
            self.skipTest("no C++ compiler or sgl_kernel python sources")
        prog = (
            '#include <cstdint>\n#include <cstdio>\n#include <string>\n#include <tuple>\n#include <utility>\n'
            '#include <sgl_kernel/scalar_type.hpp>\n'
            'int main(){ std::printf("%lld %lld %lld %lld\\n", (long long)host::kU4B8.id(), (long long)host::kU4.id(),'
            ' (long long)host::kFloat16.id(), (long long)host::kBFloat16.id()); }\n'
        )
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "t.cpp"
            src.write_text(prog)
            exe = Path(d) / "t"
            subprocess.run([cxx, "-std=c++20", f"-I{_JIT / 'include'}", str(src), "-o", str(exe)], check=True, capture_output=True)
            cpp_ids = [int(x) for x in subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.split()]
        spec = importlib.util.spec_from_file_location("sgl_scalar_type_h88", sk / "sgl_kernel" / "scalar_type.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod  # dataclass needs the module registered
        spec.loader.exec_module(mod)
        st = mod.scalar_types
        py_ids = [st.uint4b8.id, st.uint4.id, st.float16.id, st.bfloat16.id]
        self.assertEqual(cpp_ids, py_ids)
        self.assertEqual(str(st.uint4b8), "uint4b8")
        self.assertEqual(str(st.uint4), "uint4")



# ---------------------------------------------------------------------------
# DESK: A16 unchanged
# ---------------------------------------------------------------------------

A16_PY_SHA256 = {
    "gptq_marlin.py": "9f48dd270b70fa3782692ccfcffbffea3f16ceb4059e90159675af76c5425f3a",
    "gptq_marlin_repack.py": "87d5ae101c2e7ad74ab622dd4f1efdca50a94505c7cb555b4ef75a62dd47f9f2",
    "awq_marlin_repack.py": "c40be49c74c5012c9241f59e4338cfb95825315035bddee7bf5ced8e183f1813",
    "moe_wna16_marlin.py": "7d87488ed3c17c4224504a32b12f770ae363ae78178f9c5cecfd2314e8d380d3",
    "marlin_switches.py": "572e4465c6b8121e34865f42ff519a11008f1bf6d3777a5484f80e4a4e5b299c",
}
A16_CUDA_SHA256 = {
    "marlin/awq_marlin_repack.cuh": "ae135c9a5c4291ebdf11df4de3303e82d849b1907bc61480a8767672399e8fd2",
    "marlin/dequant.h": "f5d3e4308c3c856cce2eb2405d6b8075d006258cc88bbf85c726383304791b79",
    "marlin/gptq_marlin.cuh": "2712929fa86a88bef190162b05774c7bfbbad15e32d9591f4c4893d0e78013ab",
    "marlin/gptq_marlin_repack.cuh": "a1fd81cbce9dacc6bb9d54129a2d266ed594a243d4fdd8fad7d1eb93cf02ad0b",
    "marlin/kernel.h": "2e184c24d34934351d481f303bb6b7d9a6eeb9f6e9aef8601c68b603b9d289d7",
    "marlin/marlin.cuh": "267bf4b6ff979ac35b47a9849e8c81bcb58c7e78441b9feae2d58c23602179f7",
    "marlin/marlin_dtypes.cuh": "a5f3f9cd4b8a7196e7d92a6c1bd47075e9624179674edcabf9138e02f048019d",
    "marlin/marlin_template.h": "0e0564d110353213c5b83c2155c40d96080e5734da55e4f45dee5de9ec4f85a0",
    "marlin_moe/kernel.h": "04d19e3aeb1c3b6dac4bcf05471e2b18776ddbc5c1aa45f323f9c96e263b1f22",
    "marlin_moe/marlin_template.h": "87b7f33b4f0da9503366dc766a76ea924271d81d2d2f70919559c9583afb12e1",
    "marlin_moe/moe_wna16_marlin.cuh": "a04df0cce825c467a9bbd075db1b35915905e9adb4223c38c3055022644a0e33",
}
# signature of every top-level def of the A16 wrappers at the baseline commit
A16_SIGNATURES = {
    ("gptq_marlin.py", "_jit_gptq_marlin_module"): "dtype: torch.dtype",
    ("gptq_marlin.py", "_or_empty"): "t: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype",
    ("gptq_marlin.py", "gptq_marlin_gemm"): "a: torch.Tensor, c: Optional[torch.Tensor], b_q_weight: torch.Tensor, b_scales: torch.Tensor, global_scale: Optional[torch.Tensor], b_zeros: Optional[torch.Tensor], g_idx: Optional[torch.Tensor], perm: Optional[torch.Tensor], workspace: torch.Tensor, b_q_type: ScalarType, size_m: int, size_n: int, size_k: int, is_k_full: bool=True, use_atomic_add: bool=False, use_fp32_reduce: bool=False, is_zp_float: bool=False",
    ("gptq_marlin_repack.py", "_jit_gptq_marlin_repack_module"): "",
    ("gptq_marlin_repack.py", "gptq_marlin_repack"): "b_q_weight: torch.Tensor, perm: torch.Tensor, size_k: int, size_n: int, num_bits: int",
    ("awq_marlin_repack.py", "_jit_awq_marlin_repack_module"): "",
    ("awq_marlin_repack.py", "awq_marlin_repack"): "b_q_weight: torch.Tensor, size_k: int, size_n: int, num_bits: int",
    ("awq_marlin_repack.py", "awq_marlin_moe_repack"): "b_q_weight: torch.Tensor, perm: torch.Tensor, size_k: int, size_n: int, num_bits: int",
    ("moe_wna16_marlin.py", "_log_marlin_switches"): "device",
    ("moe_wna16_marlin.py", "_jit_moe_wna16_marlin_module"): "dtype: torch.dtype",
    ("moe_wna16_marlin.py", "_log_override_artefact"): "override, args=None",
    ("moe_wna16_marlin.py", "_or_empty"): "t: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype",
    ("moe_wna16_marlin.py", "moe_wna16_marlin_gemm"): "a: torch.Tensor, c_or_none: Optional[torch.Tensor], b_q_weight: torch.Tensor, b_bias_or_none: Optional[torch.Tensor], b_scales: torch.Tensor, global_scale_or_none: Optional[torch.Tensor], b_zeros_or_none: Optional[torch.Tensor], g_idx_or_none: Optional[torch.Tensor], perm_or_none: Optional[torch.Tensor], workspace: torch.Tensor, sorted_token_ids: torch.Tensor, expert_ids: torch.Tensor, num_tokens_post_padded: torch.Tensor, topk_weights: torch.Tensor, moe_block_size: int, top_k: int, mul_topk_weights: bool, is_ep: bool, b_q_type: ScalarType, size_m: int, size_n: int, size_k: int, is_k_full: bool=True, use_atomic_add: bool=False, use_fp32_reduce: bool=False, is_zp_float: bool=False",
}  # fmt: skip


def _git_show_base(rel: str):
    try:
        return subprocess.run(
            ["git", "-C", str(_REPO), "show", f"{BASE_COMMIT}:{rel}"], capture_output=True, check=True
        ).stdout
    except Exception:
        return None


class TestA16PathUnchanged(unittest.TestCase):
    def test_a16_cuda_sources_byte_identical_to_baseline(self):
        for rel, sha in A16_CUDA_SHA256.items():
            data = (_CSRC / rel).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), sha, f"A16 source {rel} changed")

    def test_a16_python_prefix_byte_identical(self):
        # new functions are APPENDED: the baseline content must be a byte prefix of the file
        for name, sha in A16_PY_SHA256.items():
            cur = (_JIT / name).read_bytes()
            if name == "marlin_switches.py":
                self.assertEqual(hashlib.sha256(cur).hexdigest(), sha)
                continue
            base = _git_show_base(f"python/sglang/jit_kernel/{name}")
            if base is not None:
                self.assertEqual(hashlib.sha256(base).hexdigest(), sha, "pin does not match the baseline commit")
                self.assertTrue(cur.startswith(base), f"{name}: A16 part is not a byte prefix of the file")
            else:
                # no git history here: hash the same number of leading lines the baseline had
                n_lines = {"gptq_marlin.py": 117, "gptq_marlin_repack.py": 40, "awq_marlin_repack.py": 62, "moe_wna16_marlin.py": 299}[name]
                head = b"".join(cur.splitlines(keepends=True)[:n_lines])
                self.assertEqual(hashlib.sha256(head).hexdigest(), sha, f"{name}: A16 prefix changed")

    def test_a16_signatures_pinned(self):
        seen = set()
        for name in sorted({k[0] for k in A16_SIGNATURES}):
            tree = ast.parse((_JIT / name).read_text())
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and (name, node.name) in A16_SIGNATURES:
                    self.assertEqual(ast.unparse(node.args), A16_SIGNATURES[(name, node.name)], f"{name}:{node.name}")
                    seen.add((name, node.name))
        self.assertEqual(seen, set(A16_SIGNATURES), "an A16 function disappeared")

    def test_new_functions_only_added_below_marker(self):
        new_names = {
            "gptq_marlin.py": {"_jit_gptq_marlin_a8_module", "_b_q_type_name", "gptq_marlin_gemm_w4a8"},
            "gptq_marlin_repack.py": {"_jit_gptq_marlin_repack_a8_module", "gptq_marlin_repack_w4a8", "gptq_marlin_moe_repack_w4a8"},
            "awq_marlin_repack.py": {"_jit_awq_marlin_repack_a8_module", "awq_marlin_repack_w4a8", "awq_marlin_moe_repack_w4a8"},
            "moe_wna16_marlin.py": {"_jit_moe_wna16_marlin_a8_module", "moe_wna16_marlin_gemm_w4a8", "_w4a8_b_q_type_name"},
        }
        for name, expected in new_names.items():
            src = (_JIT / name).read_text()
            marker = src.index("H88-A (2026-10-07)")
            tree = ast.parse(src)
            marker_line = src[:marker].count("\n") + 1
            added = {n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.lineno > marker_line}
            self.assertEqual(added, expected, name)
            # nothing new above the marker: every def above is an A16 def
            above = {n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.lineno < marker_line}
            self.assertTrue(all((name, a) in A16_SIGNATURES for a in above), f"{name}: unexpected def above marker: {above}")

    def test_a16_jit_module_names_not_reused(self):
        # a8 JIT modules use their own names, so no A16 cache entry is ever shared or shadowed
        a16 = {"gptq_marlin", "moe_wna16_marlin", "gptq_marlin_repack", "awq_marlin_repack"}
        used = set()
        for name in ("gptq_marlin.py", "moe_wna16_marlin.py", "gptq_marlin_repack.py", "awq_marlin_repack.py"):
            used |= set(re.findall(r'load_jit\(\s*"([a-z0-9_]+)"', (_JIT / name).read_text()))
        self.assertTrue(a16 <= used)
        self.assertTrue({"gptq_marlin_a8", "moe_wna16_marlin_a8", "gptq_marlin_repack_a8", "awq_marlin_repack_a8"} <= used)


# ---------------------------------------------------------------------------
# GPU (NF seat, metal window): kernel numerics on sm86 and sm120
# ---------------------------------------------------------------------------


@unittest.skipUnless(GPU_TESTS, GPU_SKIP_REASON)
class TestW4A8GpuRepack(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(_JIT.parents[1]))  # python/ for the sglang package
        from sglang.jit_kernel.awq_marlin_repack import awq_marlin_repack_w4a8
        from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack_w4a8

        self.awq, self.gptq = awq_marlin_repack_w4a8, gptq_marlin_repack_w4a8

    def test_repack_bit_exact_against_cpu_layout(self):
        for (k, n), asym in itertools.product(((128, 64), (256, 256), (640, 192), (2560, 640)), (False, True)):
            c = build_weight(k, n, 128 if k % 128 == 0 else -1, asym, torch.float16, seed=k + n)
            fn = self.awq if asym else self.gptq
            out = fn(c["ckpt"].cuda(), k, n, 4)
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(out.cpu(), c["marlin_q"]), f"k={k} n={n} asym={asym}")


@unittest.skipUnless(GPU_TESTS, GPU_SKIP_REASON)
class TestW4A8GpuDenseGemm(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(_JIT.parents[1]))
        from sgl_kernel.scalar_type import scalar_types  # sgl-kernel/python on PYTHONPATH

        from sglang.jit_kernel.gptq_marlin import gptq_marlin_gemm_w4a8

        self.gemm, self.types = gptq_marlin_gemm_w4a8, scalar_types
        self.ws = U.make_workspace(torch.device("cuda"))

    def _run(self, dtype, gs, asym, m, k, n, bias=False, fp32_reduce=True, atomic=False):
        c = build_weight(k, n, gs, asym, dtype, seed=m + k + n + gs)
        a, a_q, a_scales, a_sk = build_activation(m, k, dtype, c["factor"], seed=m)
        b = None
        if bias:
            b = (torch.randn(n, generator=torch.Generator().manual_seed(4)) * 0.1).to(dtype)
        out = self.gemm(
            a_q.cuda(), a_sk.cuda(), c["marlin_q"].cuda(), c["s_m"].cuda(),
            c["zp_m"].cuda() if asym else None,
            U.marlin_permute_bias(b).cuda() if bias else None,
            self.ws, self.types.uint4 if asym else self.types.uint4b8, m, n, k,
            use_atomic_add=atomic, use_fp32_reduce=fp32_reduce,
        )  # fmt: skip
        torch.cuda.synchronize()
        ref = U.reference_w4a8_gemm(a_q, a_scales, c["w_ref"])
        if bias:
            ref = ref + b.double().reshape(1, -1)
        self.assertEqual(out.dtype, dtype)
        d = max_diff(out.cpu(), ref)
        self.assertLess(d, U.W4A8_GPU_MAX_DIFF, f"{dtype} gs={gs} asym={asym} m={m} k={k} n={n} diff={d}")

    def test_g32_g128_channelwise_sym_asym_fp16_bf16(self):
        for dtype, gs, asym in itertools.product((torch.float16, torch.bfloat16), (-1, 32, 128), (False, True)):
            for m in (1, 3, 16, 33, 64, 200):
                self._run(dtype, gs, asym, m, 512, 256)

    def test_real_model_shapes(self):
        # NF group_1 experts: hidden 2560, moe_inter 640 (w13 = 1280 wide)
        for dtype, gs in itertools.product((torch.float16, torch.bfloat16), (32, 128)):
            self._run(dtype, gs, False, 8, 2560, 1280)
            self._run(dtype, gs, False, 130, 640, 2560)

    def test_bias_and_reduce_variants(self):
        self._run(torch.float16, 128, False, 24, 512, 128, bias=True)
        self._run(torch.bfloat16, 32, True, 24, 512, 128, bias=True)
        self._run(torch.float16, 128, False, 4, 512, 128, fp32_reduce=False)
        self._run(torch.float16, 128, False, 4, 512, 128, fp32_reduce=False, atomic=True)

    def test_workspace_locks_restored(self):
        self._run(torch.float16, 128, False, 64, 512, 256)
        self.assertEqual(int(self.ws.abs().sum()), 0)


@unittest.skipUnless(GPU_TESTS, GPU_SKIP_REASON)
class TestW4A8GpuMoe(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(_JIT.parents[1]))
        from sgl_kernel.scalar_type import scalar_types

        from sglang.jit_kernel.moe_wna16_marlin import moe_wna16_marlin_gemm_w4a8

        self.gemm, self.types = moe_wna16_marlin_gemm_w4a8, scalar_types
        self.ws = U.make_workspace(torch.device("cuda"))

    def _run(self, dtype, gs, asym, e_n, m, k, n, top_k, block, mul_w, second_gemm=False):
        ws = [build_weight(k, n, gs, asym, dtype, seed=40 + e) for e in range(e_n)]
        q_all = torch.stack([w["marlin_q"] for w in ws])
        s_all = torch.stack([w["s_m"] for w in ws])
        factor = None
        if s_all.shape[1] > 1:
            # ONE factor for the whole [E, groups, N] tensor, as the layer would do
            s_f = torch.stack([U.marlin_permute_scales(w["s"].to(dtype), k, n, gs, True) for w in ws])
            s_all, factor = U.marlin_act_int8_process_scales(s_f)
        zp_all = torch.stack([w["zp_m"] for w in ws]) if asym else None
        g = torch.Generator().manual_seed(m)
        topk_ids = torch.stack([torch.randperm(e_n, generator=g)[:top_k] for _ in range(m)]).to(torch.int32)
        topk_w = torch.rand(m, top_k, generator=g)
        if second_gemm:
            # second GEMM of a MoE layer: every (token, k) pair is its own row, top_k = 1
            ids_eff, w_eff, eff_top_k = topk_ids.reshape(-1, 1), topk_w.reshape(-1, 1), 1
        else:
            ids_eff, w_eff, eff_top_k = topk_ids, topk_w, top_k
        size_m = ids_eff.shape[0]
        a, a_q, a_scales, a_sk = build_activation(size_m, k, dtype, factor, seed=m + 5)
        sorted_ids, expert_ids, num_post = moe_align_block_size_ref(ids_eff, block, e_n)
        out = self.gemm(
            a_q.cuda(), a_sk.cuda(), None, q_all.cuda(), None, s_all.cuda(),
            zp_all.cuda() if asym else None, self.ws,
            sorted_ids.cuda(), expert_ids.cuda(), num_post.cuda(), w_eff.reshape(-1).float().cuda(),
            block, eff_top_k, mul_w, self.types.uint4 if asym else self.types.uint4b8, size_m, n, k,
        )  # fmt: skip
        torch.cuda.synchronize()
        ref = moe_reference(a_q, a_scales, [w["w_ref"] for w in ws], ids_eff, w_eff, eff_top_k, mul_w)
        d = max_diff(out.cpu(), ref)
        self.assertLess(d, U.W4A8_GPU_MAX_DIFF, f"{dtype} gs={gs} asym={asym} e={e_n} m={m} block={block} diff={d}")

    def test_moe_g32_g128_sym_asym(self):
        for dtype, gs, asym in itertools.product((torch.float16, torch.bfloat16), (32, 128), (False, True)):
            for m, block in ((4, 16), (64, 16), (64, 32), (200, 48), (256, 64)):
                self._run(dtype, gs, asym, 8, m, 512, 256, 2, block, False)

    def test_moe_mul_topk_weights_and_second_gemm(self):
        self._run(torch.float16, 128, False, 8, 32, 512, 256, 2, 16, True)
        self._run(torch.bfloat16, 32, False, 8, 32, 256, 256, 2, 16, True, second_gemm=True)


if __name__ == "__main__":
    unittest.main()
