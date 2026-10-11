# SPDX-License-Identifier: Apache-2.0
"""GGUF-NF G6 (2026-10-09): IQ-type MMQ / MoE-MMQ kernels (sglang PR #36122, vendored as NEW JIT sources).

What this file proves on the desk (no GPU, no card):

* CPU:  the existing sgl-kernel GGUF sources are byte-identical (sha256 pins) -- G6 adds files, it changes none of them;
        the vendored files carry their provenance and differ from their sgl-kernel originals only by the documented edits;
        the policy (thresholds with sources, K alignment, nvcc / Blackwell gate) in every branch; the dispatch in gguf.py
        with the kernels stubbed (IQ pair armed -> IQ MMQ, not armed / M < 128 / env off / Blackwell-refused -> the pre-G6
        branch, non-IQ pairs untouched); the K-tail of the IQ4_NL tile loader (K = 640 = 20 blocks, the NF ffn_down) as an
        index model; the #28784 __byte_perm rewrite as an index model; the real NF header (expert shapes, types, K).
* nvcc: syntax / codegen check of the new modules for sm_86 and sm_120 without a card (H88-A pattern), skipped when the
        toolchain is absent.
* GPU:  numerics against dequant + matmul -- WRITTEN, LOCKED: runs only with GGUF_GPU_TESTS=1 (a gpuq window, one card).
        Blocks are SYNTHETIC (random bytes with a finite fp16 scale; any bit pattern is a valid IQ block) and REAL (one expert
        row-block per NF row class A/B/C taken from the unsloth UD-IQ4_XS header files). Tolerance: the PR's own
        (atol 1.5 / rtol 0.1 dense, atol 1 / rtol 0.1 MoE, inputs uniform [0,1)) plus a relative-RMS guard of 2 % that is
        ~5x the q8_1 activation rounding floor (step amax/127 -> ~0.4 % of the signal); the guard is a G6 choice, unmeasured.

Usage:
    /spinning/gpu-arb/pytest_gedeckelt.sh test/registered/unit/quantization/test_gguf_iq_mmq_1009.py
    GGUF_GPU_TESTS=1 CUDA_VISIBLE_DEVICES=<card> python3 -m pytest ...   # inside a gpuq window only
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import gguf as gguf_lib
import numpy as np
import torch

from sglang.jit_kernel import gguf_iq_mmq_policy as P
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

_ROOT = Path(__file__).resolve().parents[4]
_JIT = _ROOT / "python" / "sglang" / "jit_kernel"
_CSRC = _JIT / "csrc" / "gguf_iq_mmq"
_SGL = _ROOT / "sgl-kernel" / "csrc" / "quantization" / "gguf"

GPU_TESTS = os.environ.get("GGUF_GPU_TESTS") == "1"
GPU_SKIP = "needs a gpuq window: set GGUF_GPU_TESTS=1 with CUDA_VISIBLE_DEVICES=<card>"

NF_GGUF_DIR = Path(
    os.environ.get(
        "GGUF_NF_IQ_PATH",
        "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-Flash-Next-GGUF-unsloth/UD-IQ4_XS",
    )
)

# sha256 of the UNCHANGED sgl-kernel GGUF sources at base c651892375 -- G6 must not touch any of them.
BASE_SHA256 = {
    "ggml-common.h": "788004323faaf8017db4aa3a55c65325691ae106e91876a049ed93e43a47a215",
    "dequantize.cuh": "28aeb052e48158c1606dd3755e06ba6737eaf6065b7c74fa9ad03d8a203ae2a4",
    "mmq.cuh": "f45d9cb48d24ffea631bcb302532f35781a2f76f22f3f91262f70225fcc23f76",
    "mmvq.cuh": "834a024377c7deacff177fddd5727b39a1a1302f63e6a63d54f1f4602c37dfdb",
    "moe.cuh": "01b6cbd5a46c3a6fd9e26ea68650e32242ee49e75a423919c125836f63d6945f",
    "moe_vec.cuh": "ee6c024771ace60055002dace94d882ce56750558f617cdfafe4a6172e8260cd",
    "vecdotq.cuh": "0e1bb0cc1be73e06760a0c9538cc94a9d3d8e5381b008215dc3b31da77b5a95c",
    "gguf_kernel.cu": "c81d758382fb71dbba02fde6ff22c1dee872c8e94bec5986014ee19e4521338b",
}
PR_DIFF_SHA256 = "cf93506bd6b55eca441794007b1d5e92f6a66bf5f0524f288f460f1418bcee38"
NEW_FILES = [
    _CSRC / "iq_mmq_ggml_common.h",
    _CSRC / "iq_mmq_common.cuh",
    _CSRC / "iq_mmq_tiles.cuh",
    _CSRC / "iq_mmq_kernels.cuh",
    _CSRC / "gguf_iq_mmq.cuh",
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
class TestExistingKernelsUntouched(CustomTestCase):
    def test_sgl_kernel_gguf_sources_are_byte_identical(self):
        for name, want in BASE_SHA256.items():
            self.assertEqual(_sha(_SGL / name), want, f"{name} changed -- G6 must only ADD files")

    def test_new_files_exist_and_carry_provenance(self):
        for p in NEW_FILES:
            self.assertTrue(p.is_file(), p)
            head = p.read_text()[:3000]
            self.assertIn("G6", head, p.name)
        for p in NEW_FILES[1:4]:
            head = p.read_text()[:2500]
            self.assertIn("#36122", head, p.name)
            self.assertIn("0a39ec3a8754e3c4426950a5a93046c048a03f07", head, p.name)
            self.assertIn(PR_DIFF_SHA256, head, p.name)

    def test_vendored_common_header_is_a_verbatim_prefix_of_ggml_common(self):
        marker = "// ---- begin verbatim copy of sgl-kernel/csrc/quantization/gguf/ggml-common.h lines 1-997 ----\n"
        vend = (_CSRC / "iq_mmq_ggml_common.h").read_text()
        self.assertIn(marker, vend)
        body = vend.split(marker, 1)[1]
        base = (_SGL / "ggml-common.h").read_text()
        self.assertTrue(base.startswith(body), "vendored ggml-common prefix drifted from sgl-kernel")
        self.assertEqual(body.count("\n"), 997)
        self.assertNotIn("c10::", body)  # torch-free

    def test_generic_mmq_template_equals_the_sgl_kernel_one_modulo_documented_edits(self):
        base = (_SGL / "mmq.cuh").read_text()
        cut = base.index("#if defined(USE_ROCM)\n#define MMQ_X_Q4_0")
        want = base[:cut]
        for a, b in (
            ("allocate_tiles_cuda_t allocate_tiles", "iq_allocate_tiles_cuda_t allocate_tiles"),
            ("load_tiles_cuda_t load_tiles", "iq_load_tiles_cuda_t load_tiles"),
            ("vec_dot_q_mul_mat_cuda_t vec_dot", "iq_vec_dot_q_mul_mat_cuda_t vec_dot"),
            ("static __device__ __forceinline__ void mul_mat_q(", "static __device__ __forceinline__ void iq_mul_mat_q("),
            ("        threadIdx.x,\n        blocks_per_row_x);\n", "        threadIdx.x,\n        blocks_per_row_x,\n        blocks_per_row_x - ib0);\n"),
        ):
            self.assertEqual(want.count(a), 1, a)
            want = want.replace(a, b)
        want = "\n".join(l for l in want.split("\n") if not l.startswith("//"))  # header comments differ
        kern = (_CSRC / "iq_mmq_kernels.cuh").read_text()
        kern_nocomment_lines = [l for l in kern.split("\n")]
        # the template body, line by line, must appear in order in the vendored file
        body_lines = [l for l in want.split("\n") if l.strip()]
        it = iter(l for l in kern_nocomment_lines if l.strip())
        missing = [l for l in body_lines if l not in it]
        self.assertEqual(missing[:3], [], "iq_mul_mat_q drifted from mmq.cuh mul_mat_q beyond the documented edits")

    def test_moe_q_keeps_the_fork_guards_and_has_the_pr_fixes(self):
        kern = (_CSRC / "iq_mmq_kernels.cuh").read_text()
        self.assertIn("const int64_t exp_stride,\n    const int num_experts,", kern)  # int64 stride (#512) + local expert count
        self.assertIn("if (exp_idx >= num_experts || exp_idx < 0) {", kern)  # #109/#112 bound, zero-fill (PR #36122)
        self.assertNotIn("exp_idx > 255 ||", kern)  # the literal bound of the upstream moe_q is gone (the fork comment still names it)
        self.assertIn("if (col_dst_0 >= num_tokens_post_padded[0]) return;", kern)
        self.assertIn("sorted_token_ids[col_dst_0 + ids] / top_k", kern)  # per-column scale load (PR #36122)
        self.assertNotIn("token_offs[threadIdx.y] / top_k", kern)
        self.assertEqual(kern.count("        blocks_per_row_x - ib0);"), 2)  # dense + MoE loop both pass blocks_left

    def test_every_moe_kernel_and_launcher_carries_num_experts(self):
        kern = (_CSRC / "iq_mmq_kernels.cuh").read_text()
        self.assertEqual(kern.count("const int num_experts,"), 1 + 8 + 8)  # iq_moe_q + 8 kernels + 8 launchers
        self.assertEqual(len(re.findall(r"\n\s+num_experts,\n\s+ncols_x,", kern)), 8 + 16)


# ---------------------------------------------------------------------------------------------------------------------
class TestPolicy(CustomTestCase):
    def test_type_ids_match_the_gguf_package(self):
        G = gguf_lib.GGMLQuantizationType
        for name, tid in (("IQ2_XXS", 16), ("IQ2_XS", 17), ("IQ3_XXS", 18), ("IQ1_S", 19), ("IQ4_NL", 20), ("IQ3_S", 21), ("IQ2_S", 22), ("IQ4_XS", 23)):
            self.assertEqual(int(G[name]), tid)
            self.assertEqual(P.IQ_TYPE_NAMES[tid], name)
        self.assertEqual(set(P.IQ_TYPE_NAMES), set(P.IQ_MMQ_TYPES))
        self.assertNotIn(int(G.IQ1_M), P.IQ_MMQ_TYPES)  # the PR has no IQ1_M kernel

    def test_k_alignment_matches_the_block_geometry(self):
        for tid in P.IQ_MMQ_TYPES:
            block, _ = gguf_lib.GGML_QUANT_SIZES[gguf_lib.GGMLQuantizationType(tid)]
            self.assertEqual(P.K_ALIGNMENT[tid] % block, 0)
        self.assertEqual(P.K_ALIGNMENT[P.IQ4_NL], 128)
        for tid in P.IQ_MMQ_TYPES - {P.IQ4_NL}:
            self.assertEqual(P.K_ALIGNMENT[tid], 256)
        # host and policy agree (gguf_iq_mmq.cuh k_alignment table)
        host = (_CSRC / "gguf_iq_mmq.cuh").read_text()
        for tid, a in P.K_ALIGNMENT.items():
            self.assertRegex(host, rf"case {tid}: return {a};")

    def test_constants_carry_the_pr_values(self):
        self.assertEqual(P.IQ_MOE_MMQ_MIN_TOKENS, 128)
        self.assertEqual(P.IQ_MMQ_MAX_BATCH_SIZE, 16)
        self.assertEqual(P.IQ_MOE_MMQ_MIN_ASSIGNMENTS_PER_EXPERT, 2)
        self.assertEqual(P.IQ_MOE_MMQ_BLOCK_SIZE, 4)
        kern = (_CSRC / "iq_mmq_kernels.cuh").read_text()
        for t in ("IQ4_NL", "IQ4_XS", "IQ3_S", "IQ3_XXS", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ1_S"):
            self.assertIn(f"#define MOE_X_{t} 4\n", kern)  # the CUDA branch tile width the Python side assumes

    def test_nvcc_version_parse(self):
        txt = "Cuda compilation tools, release 13.2, V13.2.78\nBuild cuda_13.2.r13.2/compiler.123"
        self.assertEqual(P.parse_nvcc_version(txt), (13, 2, 78))
        self.assertEqual(P.parse_nvcc_version("Cuda compilation tools, release 12.9, V12.9.86"), (12, 9, 86))
        self.assertEqual(P.parse_nvcc_version("Cuda compilation tools, release 13.0, V13.0.88"), (13, 0, 88))
        self.assertIsNone(P.parse_nvcc_version("garbage"))
        self.assertIsNone(P.parse_nvcc_version("release 13.2, V13.3.1"))

    def test_broken_toolkits_are_exactly_13_2_0_and_13_2_1(self):
        self.assertTrue(P.nvcc_is_broken_132((13, 2, 51)))  # CUDA 13.2.0
        self.assertTrue(P.nvcc_is_broken_132((13, 2, 78)))  # CUDA 13.2.1
        self.assertFalse(P.nvcc_is_broken_132((13, 2, 86)))  # CUDA 13.2.2
        self.assertFalse(P.nvcc_is_broken_132((13, 4, 92)))
        self.assertFalse(P.nvcc_is_broken_132((13, 0, 88)))  # our image's nvcc
        self.assertFalse(P.nvcc_is_broken_132((12, 9, 86)))
        self.assertFalse(P.nvcc_is_broken_132(None))

    def test_blackwell_gate_table(self):
        bad = (13, 2, 78)
        for t in (P.IQ3_S, P.IQ2_S, P.IQ1_S):
            msg = P.blackwell_refusal(t, 12, bad)
            self.assertIsNotNone(msg)
            self.assertIn("IQ-MMQ-BLACKWELL-NVCC", msg)
            self.assertIn(P.IQ_TYPE_NAMES[t], msg)
            self.assertIn("13.2.78", msg)
            self.assertIn(P.ENV_FIX_VERIFIED, msg)
            self.assertIsNone(P.blackwell_refusal(t, 8, bad), "sm_86 is not affected")
            self.assertIsNone(P.blackwell_refusal(t, 12, (13, 2, 86)))
            self.assertIsNone(P.blackwell_refusal(t, 12, (13, 0, 88)))
            self.assertIsNone(P.blackwell_refusal(t, 12, None))
            self.assertIsNone(P.blackwell_refusal(t, 12, bad, verified=True), "the metal-verified override lifts it")
        for t in (P.IQ4_NL, P.IQ4_XS, P.IQ3_XXS, P.IQ2_XXS, P.IQ2_XS):
            self.assertIsNone(P.blackwell_refusal(t, 12, bad), f"{P.IQ_TYPE_NAMES[t]} is not in the reported failing set")

    def test_env_switches(self):
        self.assertTrue(P.env_enabled({}))
        self.assertTrue(P.env_enabled({P.ENV_ENABLE: "1"}))
        self.assertFalse(P.env_enabled({P.ENV_ENABLE: "0"}))
        self.assertFalse(P.env_enabled({P.ENV_ENABLE: "off"}))
        self.assertFalse(P.fix_verified({}))
        self.assertTrue(P.fix_verified({P.ENV_FIX_VERIFIED: "1"}))

    def test_moe_shape_rule(self):
        e, k = 512, 10  # NF
        self.assertFalse(P.moe_mmq_shape_ok(127, e, k))  # below the PR threshold
        self.assertTrue(P.moe_mmq_shape_ok(128, e, k))  # 1280 assignments >= 2 * 512
        self.assertFalse(P.moe_mmq_shape_ok(128, 700, k))  # < 2 assignments per expert
        self.assertTrue(P.moe_mmq_shape_ok(8192, e, k))  # prefill chunk
        # grid.y bound: 65535 blocks of 4
        self.assertFalse(P.moe_mmq_shape_ok(27000, e, k))  # 270000 + 1539 > 262140
        self.assertTrue(P.moe_mmq_shape_ok(26000, e, k))
        # no 256-expert cap (the PR's cap belongs to its 255 literal; iq_moe_q guards with the local expert count)
        self.assertTrue(P.moe_mmq_shape_ok(4096, 512, 10))

    def test_dense_window(self):
        self.assertFalse(P.dense_mmq_shape_ok(8, 8))  # MMVQ owns M <= mmvq_safe
        self.assertTrue(P.dense_mmq_shape_ok(9, 8))
        self.assertTrue(P.dense_mmq_shape_ok(16, 8))
        self.assertFalse(P.dense_mmq_shape_ok(17, 8))  # dequant + cuBLAS wins above 16
        self.assertFalse(P.dense_mmq_shape_ok(16, 16))  # rows <= 5120: MMVQ covers M <= 16

    def test_k_aligned(self):
        self.assertTrue(P.k_aligned(P.IQ3_S, 2560))
        self.assertFalse(P.k_aligned(P.IQ3_S, 640))
        self.assertTrue(P.k_aligned(P.IQ4_NL, 640))  # NF ffn_down
        self.assertFalse(P.k_aligned(P.IQ4_NL, 96))
        self.assertFalse(P.k_aligned(8, 256))  # Q8_0 is not an IQ type


# ---------------------------------------------------------------------------------------------------------------------
class TestIndexModels(CustomTestCase):
    def test_iq4_nl_k_tail_model(self):
        """Which 32-element blocks does the IQ4_NL MMQ loop process, and which does the tile loader zero-fill?

        mul_mat_q / moe_q: 8 blocks per K window (blocks_per_warp = 32 / QI4_NL), `ir` selects 4 of them; an `ir` group is run
        only while ib0 + ir * 8 / 2 < blocks_per_row. The loader loads 8 blocks per window and (G6) zero-fills kbx >= blocks_left.
        """
        bpw, qr = 8, 2

        def run(K):
            nb = K // 32
            processed, loaded_past_row = [], 0
            for ib0 in range(0, nb, bpw):
                blocks_left = nb - ib0
                loaded_past_row += sum(1 for kbx in range(bpw) if kbx >= blocks_left)  # zero-filled, NOT read
                for ir in range(qr):
                    if not (ib0 + ir * bpw // qr < nb):
                        break
                    processed += list(range(ib0 + ir * (bpw // qr), ib0 + (ir + 1) * (bpw // qr)))
            return nb, processed, loaded_past_row

        for K in (128, 256, 384, 640, 2560):
            nb, processed, past = run(K)
            self.assertEqual(processed, list(range(nb)), f"K={K}: every block exactly once, none past the row")
        nb, _, past = run(640)
        self.assertEqual((nb, past), (20, 4))  # 20 blocks; the last window has 4 valid + 4 zero-filled
        # why 128 and not 32: K = 96 would process blocks that do not exist
        nb, processed, _ = run(96)
        self.assertNotEqual(processed, list(range(nb)))

    def test_byte_perm_rewrite_equals_the_byte_pointer_read(self):
        """#28784 analogue: __byte_perm(w, 0, 0x4440 | n) == byte n of w (zero-extended)."""

        def byte_perm(x: int, y: int, s: int) -> int:
            src = (x & 0xFFFFFFFF) | ((y & 0xFFFFFFFF) << 32)
            out = 0
            for i in range(4):
                sel = (s >> (4 * i)) & 0xF
                b = (src >> (8 * (sel & 7))) & 0xFF
                if sel & 8:  # sign replicate mode
                    b = 0xFF if (b & 0x80) else 0x00
                out |= b << (8 * i)
            return out

        rng = np.random.default_rng(0)
        for w in rng.integers(0, 2**32, size=64, dtype=np.uint64).tolist():
            for n in range(4):
                self.assertEqual(byte_perm(w, 0, 0x4440 | n), (w >> (8 * n)) & 0xFF)
        # IQ3_S: qs bytes 2*sub, 2*sub+1 of the 8 bytes of one ib32 (words w0, w1)
        for sub in range(4):
            word = 0 if sub < 2 else 1
            for off in (0, 1):
                n = 2 * sub + off
                self.assertEqual((word, (2 * sub + off) & 3), (n // 4, n % 4))
        # IQ2_S: qs[4 * ib32 + sub] is byte `sub` of word ib32; IQ1_S: qs[sub] is byte `sub` of the word

    def test_byte_perm_is_used_for_every_grid_index_of_the_three_affected_types(self):
        tiles_full = (_CSRC / "iq_mmq_tiles.cuh").read_text()
        tiles = "\n".join(l.split("//")[0] for l in tiles_full.split("\n"))  # code only, comments stripped
        self.assertEqual(tiles.count("__byte_perm("), 4)  # IQ1_S 1, IQ2_S 1, IQ3_S 2
        self.assertEqual(tiles_full.count("#28784 analogue"), 3)
        self.assertNotIn("qs[sub] |", tiles)
        self.assertNotIn("qs[2 * sub", tiles)
        self.assertNotIn("qs[4 * ib32 + sub]", tiles)
        self.assertNotRegex(tiles, r"iq3xs_grid\[qs\[")

    def test_loaders_take_blocks_left_and_only_iq4_nl_uses_it(self):
        tiles = (_CSRC / "iq_mmq_tiles.cuh").read_text()
        self.assertEqual(tiles.count("const int& blocks_left) {"), 8)
        self.assertEqual(len(re.findall(r"kbx < blocks_left|kbxd < blocks_left", tiles)), 2)


# ---------------------------------------------------------------------------------------------------------------------
def _nf_part2():
    p = NF_GGUF_DIR / "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf"
    return p if p.is_file() else None


@unittest.skipUnless(_nf_part2() is not None, "NF UD-IQ4_XS GGUF header files not on this host")
class TestRealNfHeader(CustomTestCase):
    """The real header: expert tensors of the three row classes A / B / C (GGUF_NF_IST_1009.md section 4)."""

    @classmethod
    def setUpClass(cls):
        cls.reader = gguf_lib.GGUFReader(str(_nf_part2()))
        cls.t = {t.name: t for t in cls.reader.tensors}

    def test_row_classes_and_k(self):
        G = gguf_lib.GGMLQuantizationType
        cases = {  # layer -> (gate/up type, down type)
            0: (G.IQ3_S, G.IQ4_NL),  # class A
            4: (G.IQ3_S, G.Q8_0),  # class B
            2: (G.IQ4_XS, G.Q8_0),  # class C
        }
        for layer, (gu, dn) in cases.items():
            gate = self.t[f"blk.{layer}.ffn_gate_exps.weight"]
            up = self.t[f"blk.{layer}.ffn_up_exps.weight"]
            down = self.t[f"blk.{layer}.ffn_down_exps.weight"]
            self.assertEqual((gate.tensor_type, up.tensor_type, down.tensor_type), (gu, gu, dn))
            self.assertEqual(list(gate.shape), [2560, 640, 512])  # K, rows, experts
            self.assertEqual(list(down.shape), [640, 2560, 512])
            self.assertTrue(P.k_aligned(int(gu), int(gate.shape[0])))
            if int(dn) in P.IQ_MMQ_TYPES:
                self.assertTrue(P.k_aligned(int(dn), int(down.shape[0])), "NF ffn_down K=640 must be admitted for IQ4_NL")
                self.assertFalse(int(down.shape[0]) % 256 == 0, "K=640 is the reason IQ4_NL needs the 128 alignment")
            else:
                self.assertEqual(int(down.shape[0]) % 128, 0)  # Q8_0: wheel MMQ alignment 128
            # block geometry of the policy == the file
            block, size = gguf_lib.GGML_QUANT_SIZES[gate.tensor_type]
            self.assertEqual(gate.data.shape, (512, 640, 2560 // block * size))

    def test_gate_up_merge_geometry(self):
        # w13 = (E, 2 * 640, 2560): n13 // 2 == 640 == K of w2, the value _iq_moe_mmq_selected checks
        self.assertEqual(2 * 640 // 2, 640)


def _align_buffer_len(numel: int, num_experts: int, block: int = 4) -> int:
    """Length of the sorted_token_ids buffer moe_align_block_size allocates (triton_utils/moe_align_block_size.py:62-66)."""
    if numel < num_experts + 1:
        return numel * block
    return numel + (num_experts + 1) * (block - 1)


# ---------------------------------------------------------------------------------------------------------------------
def _install_fake_iq_module(G, ready_types, calls):
    mod = types.SimpleNamespace()
    mod.is_ready = lambda t: int(t) in ready_types

    def mul_mat_a8(W, X, t, row):
        calls.append(("iq_dense", int(t), tuple(X.shape), row))
        return torch.zeros(X.shape[0], row, dtype=X.dtype)

    def moe_a8(X, W, sorted_ids, expert_ids, ntpp, t, row, top_k, tokens):
        calls.append(("iq_moe", int(t), row, top_k, tokens, expert_ids.tolist()))
        return torch.zeros(tokens * top_k, row, dtype=X.dtype)

    mod.mul_mat_a8, mod.moe_a8 = mul_mat_a8, moe_a8
    return mod


class TestDispatch(CustomTestCase):
    """gguf.py dispatch with the kernels as CPU stubs: the logic under test is the real one."""

    def setUp(self):
        from sglang.srt.layers.quantization import gguf as G

        self.G = G
        self.calls = []
        self._patched = []

        def put(name, value):
            self._patched.append((name, getattr(G, name, "<absent>")))
            setattr(G, name, value)

        self.put = put
        put("_is_cuda", True)
        put("_has_sgl_gguf_kernels", True)
        put("_iq_mmq_mod", _install_fake_iq_module(G, {20, 21, 23}, self.calls))
        put("_mmvq_safe_for_device", lambda: 2)
        put("_ggml_dequantize_ws", lambda w, t, n, k, dt: self.calls.append(("dequant", int(t))) or torch.zeros(n, k, dtype=dt))
        put("ggml_mul_mat_vec_a8", lambda w, x, t, n: self.calls.append(("mmvq", int(t))) or torch.zeros(x.shape[0], n, dtype=x.dtype))
        put("ggml_mul_mat_a8", lambda w, x, t, n: self.calls.append(("mmq", int(t))) or torch.zeros(x.shape[0], n, dtype=x.dtype))
        put("ggml_moe_get_block_size", lambda t: 4)
        put("ggml_moe_a8", lambda x, w, s, e, n, t, row, k, tok: self.calls.append(("wheel_moe", int(t), row, k, tok)) or torch.zeros(tok * k, row, dtype=x.dtype))
        put("ggml_moe_a8_vec", lambda x, w, ids, k, t, row, tok: self.calls.append(("moe_vec", int(t), row, k, tok)) or torch.zeros(tok * k, row, dtype=x.dtype))
        put("moe_sum", lambda out, dst: dst.zero_())
        put("silu_and_mul", lambda t: t[..., : t.shape[-1] // 2], )

        fake_align = types.ModuleType("sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size")

        def moe_align_block_size(topk_ids, block, E):
            # G9: the REAL allocation of moe_align_block_size (triton_utils/moe_align_block_size.py:62-66): an upper bound that is
            # generally NOT a multiple of the block size. Until G9 this fake rounded n up to a multiple of 4, which hid the
            # stage-0 metal failure ("sorted_token_ids length must be a multiple of the MoE block size 4") from the desk.
            n = _align_buffer_len(topk_ids.numel(), E, block)
            return (
                torch.zeros(n, dtype=torch.int32),
                torch.tensor([0, 1, E, 77, 300], dtype=torch.int32),
                torch.tensor([n], dtype=torch.int32),
            )

        fake_align.moe_align_block_size = moe_align_block_size
        self._mods = {}
        for name in (
            "sglang.srt.layers.moe.moe_runner.triton_utils",
            "sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size",
        ):
            self._mods[name] = sys.modules.get(name)
        pkg = sys.modules.get("sglang.srt.layers.moe.moe_runner.triton_utils")
        sys.modules["sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size"] = fake_align
        if pkg is None:
            sys.modules["sglang.srt.layers.moe.moe_runner.triton_utils"] = types.ModuleType(
                "sglang.srt.layers.moe.moe_runner.triton_utils"
            )
            sys.modules["sglang.srt.layers.moe.moe_runner.triton_utils"].__path__ = []

    def tearDown(self):
        for name, old in reversed(self._patched):
            if old == "<absent>":
                delattr(self.G, name)
            else:
                setattr(self.G, name, old)
        for name, mod in self._mods.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod

    # -- dense ---------------------------------------------------------------------------------------------------------
    def _dense(self, qtype, m, rows=6000, k=2560):
        x = torch.zeros(m, k, dtype=torch.bfloat16)
        block, size = gguf_lib.GGML_QUANT_SIZES[gguf_lib.GGMLQuantizationType(qtype)]
        w = torch.zeros(rows, k // block * size, dtype=torch.uint8)
        self.calls.clear()
        self.G.fused_mul_mat_gguf(x, w, qtype)
        return [c[0] for c in self.calls]

    def test_dense_iq_window_goes_to_the_iq_kernel(self):
        self.assertEqual(self._dense(21, 12), ["iq_dense"])
        self.assertEqual(self._dense(20, 16), ["iq_dense"])
        self.assertEqual(self._dense(23, 9), ["iq_dense"])

    def test_dense_iq_outside_the_window_keeps_the_old_path(self):
        self.assertEqual(self._dense(21, 4), ["mmvq"])  # M <= mmvq_safe (8 for rows > 5120)
        self.assertEqual(self._dense(21, 17), ["dequant"])  # above 16 dequant + cuBLAS
        self.assertEqual(self._dense(21, 12, rows=4000), ["mmvq"])  # rows <= 5120: mmvq_safe 16

    def test_dense_iq_not_armed_or_off_or_misaligned_keeps_the_old_path(self):
        self.G._iq_mmq_mod.is_ready = lambda t: False
        self.assertEqual(self._dense(21, 12), ["dequant"])
        self.G._iq_mmq_mod.is_ready = lambda t: True
        self.assertEqual(self._dense(20, 12, k=2528), ["dequant"])  # IQ4_NL K=2528 = 79 blocks, 2528 % 128 != 0

    def test_dense_non_iq_types_are_untouched(self):
        for q, want in ((8, ["mmq"]), (14, ["mmq"])):  # Q8_0, Q6_K at M=8: mmvq_safe stub 2 -> MMQ branch
            self.assertEqual(self._dense(q, 8), want)
        self.assertNotIn("iq_dense", self._dense(2, 8))

    # -- MoE -----------------------------------------------------------------------------------------------------------
    def _moe(self, t13, t2, tokens, E=512, top_k=10, hidden=2560, inter=640, dtype=torch.bfloat16):
        def w(qtype, rows, k):
            block, size = gguf_lib.GGML_QUANT_SIZES[gguf_lib.GGMLQuantizationType(qtype)]
            return torch.zeros(E, rows, k // block * size, dtype=torch.uint8)

        x = torch.zeros(tokens, hidden, dtype=dtype)
        topk_w = torch.zeros(tokens, top_k, dtype=dtype)
        topk_ids = torch.zeros(tokens, top_k, dtype=torch.int32)
        self.calls.clear()
        self.G.fused_moe_gguf(x, w(t13, 2 * inter, hidden), w(t2, hidden, inter), topk_w, topk_ids, t13, t2, "silu")
        return list(self.calls)

    def test_nf_class_a_prefill_goes_through_iq_mmq(self):
        calls = self._moe(21, 20, 256)  # IQ3_S gate/up + IQ4_NL down, K = 640
        self.assertEqual([c[0] for c in calls], ["iq_moe", "iq_moe"])
        self.assertEqual([c[1] for c in calls], [21, 20])
        self.assertEqual(calls[0][2:5], (1280, 10, 256))
        self.assertEqual(calls[1][2:5], (2560, 1, 2560))
        # expert ids >= E (garbage / the zero-pad expert id E of an uneven shard) are sanitised to -1 before the kernel
        self.assertEqual(calls[0][5], [0, 1, -1, 77, 300])  # only ids >= E (512) are masked; ids < E are valid blocks

    def test_nf_class_a_prefill_with_the_real_wrapper_and_the_real_buffer_length(self):
        """G9: the dispatch -> real gguf_iq_mmq.moe_a8 wrapper -> kernel host checks, with moe_align_block_size's real length.

        The fake module of setUp stands in for the wrapper and the fake aligner used to round the length up, so neither the
        wrapper nor the length contract was ever exercised on the desk; here only the nvcc-built module is replaced, by the
        host checks of gguf_iq_mmq.cuh. Without the G9 floor in the wrapper this raises the metal error.
        """
        from sglang.jit_kernel import gguf_iq_mmq as K

        host = _HostCheckModule()
        old = K._module
        K._module = lambda type_id: host
        self.addCleanup(setattr, K, "_module", old)
        self.G._iq_mmq_mod = types.SimpleNamespace(is_ready=lambda t: int(t) in {20, 21, 23}, moe_a8=K.moe_a8)
        # realistic routing shapes (sorted_ids upper-bound length, expert_ids of ceil(len / 4) blocks) instead of setUp's 5-entry stub
        sys.modules["sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size"].moe_align_block_size = (
            lambda topk_ids, block, E: _align_model(topk_ids, E, block)[:3]
        )
        for tokens, top_k, E in ((256, 10, 512), (192, 10, 16)):
            host.calls.clear()
            self.calls.clear()
            x = torch.zeros(tokens, 2560, dtype=torch.bfloat16)
            w1 = torch.zeros(E, 2 * 640, 2560 // 256 * 110, dtype=torch.uint8)  # IQ3_S rows
            w2 = torch.zeros(E, 2560, 640 // 32 * 18, dtype=torch.uint8)  # IQ4_NL rows
            topk_ids = torch.zeros(tokens, top_k, dtype=torch.int32)
            self.G.fused_moe_gguf(x, w1, w2, torch.zeros(tokens, top_k, dtype=x.dtype), topk_ids, 21, 20, "silu")
            self.assertEqual(len(host.calls), 2, (tokens, top_k, E))
            for sorted_ids, _, _ in host.calls:
                self.assertEqual(sorted_ids.shape[0] % 4, 0)
                self.assertEqual(sorted_ids.shape[0], _align_buffer_len(tokens * top_k, E) // 4 * 4)

    def test_nf_class_b_mixed_pair_uses_iq_and_the_wheel_kernel(self):
        calls = self._moe(21, 8, 256)  # IQ3_S + Q8_0 (wheel MMQ, K = 640 % 128 == 0)
        self.assertEqual([c[0] for c in calls], ["iq_moe", "wheel_moe"])

    def test_nf_class_c(self):
        calls = self._moe(23, 8, 256)  # IQ4_XS + Q8_0
        self.assertEqual([c[0] for c in calls], ["iq_moe", "wheel_moe"])

    def test_below_128_tokens_and_unarmed_fall_back_to_moe_vec_exactly_as_before(self):
        self.assertEqual([c[0] for c in self._moe(21, 20, 127)], ["moe_vec", "moe_vec"])
        self.G._iq_mmq_mod.is_ready = lambda t: False
        self.assertEqual([c[0] for c in self._moe(21, 20, 256)], ["moe_vec", "moe_vec"])

    def test_off_switch_and_blackwell_refusal_mean_not_armed(self):
        # the gate acts at prepare() time (not ready); the dispatch only ever asks is_ready -- a refused type is never ready
        self.G._iq_mmq_mod.is_ready = lambda t: int(t) == 20  # IQ3_S refused (Blackwell gate), IQ4_NL armed
        self.assertEqual([c[0] for c in self._moe(21, 20, 256)], ["moe_vec", "moe_vec"])

    def test_misaligned_k_falls_back(self):
        calls = self._moe(21, 20, 256, hidden=2560, inter=320)  # K of the down proj 320: IQ4_NL needs 128 -> 320 % 128 != 0
        self.assertEqual([c[0] for c in calls], ["moe_vec", "moe_vec"])

    def test_non_iq_pairs_never_reach_the_iq_branch(self):
        for t13, t2 in ((8, 8), (14, 14), (12, 13)):
            names = [c[0] for c in self._moe(t13, t2, 256, E=64, top_k=4, hidden=2048, inter=512)]
            self.assertNotIn("iq_moe", names)
            self.assertEqual(names, ["wheel_moe", "wheel_moe"])
        # and below the old M > 64 gate the old moe_vec branch decides as before
        self.assertEqual([c[0] for c in self._moe(8, 8, 32, E=64, top_k=4, hidden=2048, inter=512)], ["moe_vec", "moe_vec"])

    def test_float32_activations_keep_the_old_path(self):
        names = [c[0] for c in self._moe(21, 20, 256, dtype=torch.float32)]
        self.assertEqual(names, ["moe_vec", "moe_vec"])

    def test_prepare_is_called_at_load_time_only(self):
        src = Path(self.G.__file__).read_text()
        self.assertEqual(src.count("iq_mmq_prepare("), 3)  # def + MoE hook + Linear hook
        # the per-forward path never builds: is_ready is a lookup and gguf_iq_mmq.is_ready has no load_jit call
        mod_src = (_JIT / "gguf_iq_mmq.py").read_text()
        ready = mod_src.split("def is_ready", 1)[1].split("\ndef ", 1)[0]
        self.assertNotIn("load_jit", ready)
        self.assertNotIn("_module(", ready)


# ---------------------------------------------------------------------------------------------------------------------
class _HostCheckModule:
    """Stands in for the JIT module: replays the HOST checks of gguf_iq_mmq.cuh moe_a8 (lines 326-330) and records the call."""

    def __init__(self):
        self.calls = []

    def moe_a8(self, X, W, sorted_ids, expert_ids, ntpp, Y, quant_x, type_id, row, top_k, tokens):
        n = sorted_ids.shape[0]
        if n % 4 != 0:
            raise RuntimeError("sorted_token_ids length must be a multiple of the MoE block size 4")
        if n // 4 > 65535:
            raise RuntimeError("MoE MMQ grid.y limit")
        if expert_ids.shape[0] < n // 4:
            raise RuntimeError("expert_ids shorter than the number of blocks")
        self.calls.append((sorted_ids, expert_ids, ntpp))


def _align_model(topk_ids: torch.Tensor, E: int, block: int = 4):
    """CPU model of moe_align_block_size: (sorted_ids buffer, expert_ids buffer, num_tokens_post_padded, real used length)."""
    n_buf = _align_buffer_len(topk_ids.numel(), E, block)
    ids = topk_ids.reshape(-1).tolist()
    sorted_ids, expert_ids = [], []
    for e in range(E + 1):  # E + 1 ids: the extra one is the filtered / zero-pad expert
        mine = [i for i, v in enumerate(ids) if v == e]
        padded = -(-len(mine) // block) * block
        sorted_ids += mine + [len(ids)] * (padded - len(mine))
        expert_ids += [e] * (padded // block)
    used = len(sorted_ids)
    buf = torch.full((n_buf,), len(ids), dtype=torch.int32)
    buf[:used] = torch.tensor(sorted_ids, dtype=torch.int32)
    eb = torch.full((-(-n_buf // block),), 7777, dtype=torch.int32)
    eb[: len(expert_ids)] = torch.tensor(expert_ids, dtype=torch.int32)
    return buf, eb, torch.tensor([used], dtype=torch.int32), used


class TestMoeRoutingContract(CustomTestCase):
    """G9 (stage-0 metal, sm86 + sm120: 13 MoE cases red): the routing buffer length is an upper bound, not a multiple of 4."""

    def test_buffer_length_is_in_general_not_a_multiple_of_the_block(self):
        # the failing metal case: tokens=192, top_k=10, E=16 (test_moe_iq4_nl_k640_ffn_down_shape)
        self.assertEqual(_align_buffer_len(192 * 10, 16), 1971)
        self.assertNotEqual(1971 % 4, 0)
        # tiny batch branch (numel < E + 1): numel * block, always aligned
        self.assertEqual(_align_buffer_len(5, 16), 20)

    def test_moe_routing_len_floors_to_the_block(self):
        self.assertEqual(P.moe_routing_len(1971), 1968)
        self.assertEqual(P.moe_routing_len(1968), 1968)
        self.assertEqual(P.moe_routing_len(4), 4)
        self.assertEqual(P.moe_routing_len(7), 4)
        self.assertEqual(P.moe_routing_len(3), 0)

    def test_the_floor_never_cuts_a_used_block(self):
        """num_tokens_post_padded (a multiple of 4, <= buffer) <= floor(buffer): for random routings incl. the filtered expert."""
        g = torch.Generator().manual_seed(5)
        for tokens, top_k, E in ((192, 10, 16), (130, 4, 8), (1, 1, 1), (3, 2, 512), (257, 10, 64), (128, 10, 512), (64, 8, 3)):
            for _ in range(4):
                topk = torch.randint(0, E + 1, (tokens, top_k), generator=g, dtype=torch.int32)  # E = the zero-pad expert
                buf, eb, ntpp, used = _align_model(topk, E)
                self.assertEqual(used % 4, 0)
                self.assertLessEqual(used, buf.shape[0])
                self.assertLessEqual(used, P.moe_routing_len(buf.shape[0]), (tokens, top_k, E))
                self.assertGreaterEqual(eb.shape[0], P.moe_routing_len(buf.shape[0]) // 4)

    def _call(self, K, buf, eb, ntpp):
        X = torch.zeros(4, 512, dtype=torch.bfloat16)
        W = torch.zeros(2, 8, 10, dtype=torch.uint8)
        return K.moe_a8(X, W, buf, eb, ntpp, P.IQ4_NL, 8, 2, 4)

    def _wrapper(self):
        from sglang.jit_kernel import gguf_iq_mmq as K

        fake = _HostCheckModule()
        old = K._module
        K._module = lambda type_id: fake
        self.addCleanup(setattr, K, "_module", old)
        return K, fake

    def test_wrapper_hands_the_kernel_an_aligned_routing_for_the_metal_shape(self):
        K, fake = self._wrapper()
        topk = torch.randint(0, 16, (192, 10), dtype=torch.int32)
        buf, eb, ntpp, used = _align_model(topk, 16)
        self.assertEqual(buf.shape[0], 1971)
        X = torch.zeros(192, 640, dtype=torch.bfloat16)
        W = torch.zeros(16, 256, 10, dtype=torch.uint8)
        K.moe_a8(X, W, buf, eb, ntpp, P.IQ4_NL, 256, 10, 192)  # raises RuntimeError without the G9 floor
        (sorted_ids, expert_ids, _), = fake.calls
        self.assertEqual(sorted_ids.shape[0] % 4, 0)
        self.assertEqual(sorted_ids.shape[0], 1968)
        self.assertGreaterEqual(sorted_ids.shape[0], used)
        # order and content preserved, zero copy: the kernel reads the very same memory
        self.assertTrue(torch.equal(sorted_ids, buf[:1968]))
        self.assertEqual(sorted_ids.data_ptr(), buf.data_ptr())
        self.assertIs(expert_ids, eb)

    def test_wrapper_keeps_an_already_aligned_buffer_whole(self):
        K, fake = self._wrapper()
        buf, eb, ntpp, _ = _align_model(torch.zeros(5, 1, dtype=torch.int32), 16)  # numel < E + 1 -> numel * 4 = 20
        self.assertEqual(buf.shape[0], 20)
        X = torch.zeros(5, 512, dtype=torch.bfloat16)
        W = torch.zeros(16, 8, 10, dtype=torch.uint8)
        K.moe_a8(X, W, buf, eb, ntpp, P.IQ4_NL, 8, 1, 5)
        self.assertEqual(fake.calls[0][0].shape[0], 20)

    def test_the_kernel_check_is_not_loosened(self):
        src = (_CSRC / "gguf_iq_mmq.cuh").read_text()
        self.assertIn('RuntimeCheck(tokens_post_padded % 4 == 0, "sorted_token_ids length must be a multiple of the MoE block size 4");', src)
        self.assertEqual(_sha(_CSRC / "gguf_iq_mmq.cuh"), "61294f69dc6bafe7fff5253b587b77a5e3ece2cce7d36142f96ebedeed91bdb0")

    def test_production_dispatch_and_the_gpu_test_take_the_same_wrapper_and_the_same_aligner(self):
        """gguf.py _moe_mm_a8 -> _iq_mmq_mod.moe_a8 (the wrapper under test); the metal test calls K.moe_a8 on moe_align_block_size output."""
        g_src = Path(__import__("sglang.srt.layers.quantization.gguf", fromlist=["x"]).__file__).read_text()
        body = g_src.split("def _fused_moe_gguf_iq_mmq", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("moe_align_block_size(\n        topk_ids, _iq_policy.IQ_MOE_MMQ_BLOCK_SIZE, E\n    )", body)
        self.assertEqual(body.count("_moe_mm_a8("), 2)
        mm = g_src.split("def _moe_mm_a8", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("_iq_mmq_mod.moe_a8(", mm)
        t_src = Path(__file__).read_text()
        self.assertIn("moe_align_block_size(topk_ids, P.IQ_MOE_MMQ_BLOCK_SIZE, E)", t_src)
        self.assertIn("self.K.moe_a8(x, w, sorted_ids, expert_ids, ntpp, qtype, rows, top_k, tokens)", t_src)


class TestSyntheticToleranceIsAttainable(CustomTestCase):
    """G9 (dense IQ4_XS red on sm86 + sm120): the GPU test bounds must be attainable by the EXACT kernel math on its own inputs.

    CPU model of the kernel: exact dequantised weights (gguf-py), q8_1 activation rounding (per 32: d = amax / 127, q = round),
    exact integer dot, output rounded to the activation dtype. A bound that this model exceeds cannot be met by a correct kernel.
    """

    @staticmethod
    def _q81(x: torch.Tensor) -> torch.Tensor:
        m, k = x.shape
        xb = x.reshape(m, k // 32, 32)
        d = xb.abs().amax(-1, keepdim=True) / 127
        q = torch.round(xb / d.clamp_min(1e-30))
        return (q * d).reshape(m, k)

    def _utilisation(self, qtype, atol, rtol, moe, seeds=4, dtype=torch.bfloat16):
        """max |err| / (atol + rtol |ref|) over seeds; > 1 means assert_close fails."""
        rows, k = (96, 512) if moe else (128, 512)
        raw = _synth_blocks(qtype, rows, k, seed=qtype)
        q = gguf_lib.GGMLQuantizationType(qtype)
        wd = torch.from_numpy(np.ascontiguousarray(gguf_lib.quants.dequantize(raw, q))).float()
        worst = 0.0
        for s in range(seeds):
            torch.manual_seed(s)
            x = torch.rand((130 if moe else 16, k), dtype=dtype)
            y = (self._q81(x.float()) @ wd.T).to(dtype).float()
            ref = x.float() @ wd.T
            worst = max(worst, float(((y - ref).abs() / (atol + rtol * ref.abs())).max()))
        return worst, float(wd.pow(2).mean().sqrt())

    def test_every_type_stays_well_inside_the_pr_bounds(self):
        for q in sorted(P.IQ_MMQ_TYPES):
            for moe, atol in ((False, 1.5), (True, 1.0)):
                u, w_rms = self._utilisation(q, atol, 0.1, moe)
                self.assertLess(u, 0.6, f"{P.IQ_TYPE_NAMES[q]} moe={moe}: tolerance utilisation {u:.2f} (w_rms {w_rms:.2f})")

    def test_iq4_xs_synthetic_amplitude_is_in_the_family(self):
        _, w_xs = self._utilisation(P.IQ4_XS, 1.5, 0.1, False, seeds=1)
        _, w_nl = self._utilisation(P.IQ4_NL, 1.5, 0.1, False, seeds=1)
        self.assertLess(w_xs, 3 * w_nl)  # before G9: 14.25 vs 0.84


# ---------------------------------------------------------------------------------------------------------------------
def _nvcc():
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    cand = [Path(cuda_home) / "bin" / "nvcc"] if cuda_home else []
    which = shutil.which("nvcc")
    if which:
        cand.append(Path(which))
    cand.append(Path("/usr/local/cuda/bin/nvcc"))
    for c in cand:
        if c.is_file():
            return str(c)
    return None


def _tvm_include():
    try:
        import tvm_ffi.libinfo as li

        return str(li.find_include_path())
    except Exception:
        return None


@unittest.skipUnless(_nvcc() and _tvm_include(), "needs nvcc and tvm_ffi headers (no card needed)")
class TestNvccSyntax(CustomTestCase):
    """nvcc codegen of the new module for sm_86 and sm_120 (-c only, no card, no run). The H88-A pattern."""

    WRAP = (
        '#include <tvm/ffi/function.h>\n#include "gguf_iq_mmq/gguf_iq_mmq.cuh"\n'
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC(mul_mat_a8, (gguf_iq_mmq::mul_mat_a8));\n"
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_a8, (gguf_iq_mmq::moe_a8));\n"
    )

    def _compile(self, arch: int, mask: int):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "wrap.cu"
            src.write_text(self.WRAP)
            out = Path(td) / "wrap.o"
            cmd = [
                _nvcc(), "-std=c++20", "-O3", "--expt-relaxed-constexpr",
                "-gencode", f"arch=compute_{arch},code=sm_{arch}", f"-DSGL_CUDA_ARCH={arch}0",
                f"-DGGUF_IQ_MMQ_TYPE_MASK={mask}",
                f"-I{_JIT / 'include'}", f"-I{_JIT / 'csrc'}", f"-I{_tvm_include()}",
                "-c", str(src), "-o", str(out),
            ]  # fmt: skip
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            self.assertEqual(r.stderr.strip(), "", "nvcc warnings:\n" + r.stderr[-2000:])
            cuobjdump = str(Path(_nvcc()).with_name("cuobjdump"))
            n_fn = n_dp4a = None
            if Path(cuobjdump).is_file():
                sass = subprocess.run([cuobjdump, "-sass", str(out)], capture_output=True, text=True, timeout=300).stdout
                n_fn = sass.count("Function :")
                n_dp4a = len(re.findall(r"IDP\.4A|DP4A", sass))
            return n_fn, n_dp4a

    def test_all_eight_types_sm86(self):
        n_fn, n_dp4a = self._compile(86, 0xFF)
        if n_fn is not None:
            # 8 types x 2 dtypes x need_check{0,1} x (dense, moe) + 2 q8_1 quantisers
            self.assertEqual(n_fn, 8 * 2 * 2 * 2 + 2)
            self.assertGreater(n_dp4a, 0)

    def test_all_eight_types_sm120(self):
        n_fn, n_dp4a = self._compile(120, 0xFF)
        if n_fn is not None:
            self.assertEqual(n_fn, 8 * 2 * 2 * 2 + 2)
            self.assertGreater(n_dp4a, 0)

    def test_single_nf_types_compile_alone(self):
        for tid in (20, 21, 23):
            n_fn, _ = self._compile(86, 1 << (tid - 16))
            if n_fn is not None:
                self.assertEqual(n_fn, 1 * 2 * 2 * 2 + 2)


# ---------------------------------------------------------------------------------------------------------------------
def _synth_blocks(qtype: int, rows: int, k: int, seed: int) -> np.ndarray:
    """Random raw ggml blocks: any bit pattern is a valid IQ block; the fp16 scale `d` (first 2 bytes) is set finite."""
    block, size = gguf_lib.GGML_QUANT_SIZES[gguf_lib.GGMLQuantizationType(qtype)]
    nb = k // block
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 256, size=(rows, nb, size), dtype=np.uint8)
    d = rng.uniform(0.002, 0.02, size=(rows, nb))
    if qtype == P.IQ4_XS:
        # G9: IQ4_XS carries a second, 6-bit per-sub-block scale (-32..31) on top of d and kvalues of +-127, so with the same d
        # the random block is ~14x larger in RMS than any other type (w_rms 14.3 vs 0.1-1.8; reference output RMS 228 vs 1-21).
        # The q8_1 activation rounding error scales with it and the PR's ABSOLUTE bound (atol 1.5 dense / 1.0 MoE) is then
        # exceeded (tolerance utilisation 1.44 dense / 2.14 MoE, CPU model of the exact kernel math) -- a property of the
        # synthetic input, not of the kernel (real NF IQ4_XS blocks: w_rms 0.013, utilisation 0.02). d / 16 brings w_rms to 0.9.
        d = d / 16
    d = d.astype(np.float16)
    raw[:, :, 0:2] = d.view(np.uint8).reshape(rows, nb, 2)
    return raw.reshape(rows, nb * size)


def _real_expert_block(layer: int, which: str, n_experts: int = 8) -> tuple[np.ndarray, int]:
    reader = gguf_lib.GGUFReader(str(_nf_part2()))
    for t in reader.tensors:
        if t.name == f"blk.{layer}.ffn_{which}_exps.weight":
            return np.ascontiguousarray(t.data[:n_experts]), int(t.tensor_type)
    raise KeyError(which)


def _rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float(((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-12)).item())


@unittest.skipUnless(GPU_TESTS, GPU_SKIP)
class TestGpuNumerics(CustomTestCase):
    """WRITTEN, LOCKED. Reference = gguf-py dequantisation (float32) + torch matmul; kernel = the vendored IQ MMQ."""

    @classmethod
    def setUpClass(cls):
        assert torch.cuda.is_available()
        from sglang.jit_kernel import gguf_iq_mmq as K

        cls.K = K
        cls.dev = torch.device("cuda")

    def _ref_w(self, raw: np.ndarray, qtype: int) -> torch.Tensor:
        deq = gguf_lib.quants.dequantize(raw, gguf_lib.GGMLQuantizationType(qtype))
        return torch.from_numpy(np.ascontiguousarray(deq)).to(self.dev).float()

    def _arm(self, qtype: int):
        why = self.K.refusal(qtype)
        if why is not None:
            self.skipTest(why)
        self.assertTrue(self.K.prepare(qtype), f"IQ MMQ build failed for type {qtype}")

    # -- dense ---------------------------------------------------------------------------------------------------------
    def _dense_case(self, qtype, k, rows=128, m=16, dtype=torch.bfloat16, raw=None):
        self._arm(qtype)
        torch.manual_seed(0)
        raw = _synth_blocks(qtype, rows, k, seed=qtype) if raw is None else raw
        wd = self._ref_w(raw, qtype)
        x = torch.rand((m, k), dtype=dtype, device=self.dev)
        w = torch.from_numpy(raw).to(self.dev)
        y = self.K.mul_mat_a8(w, x, qtype, rows)
        ref = x.float() @ wd.T
        self.assertTrue(torch.isfinite(y).all())
        torch.testing.assert_close(y.float(), ref, atol=1.5, rtol=1e-1)  # the PR's own bound
        self.assertLess(_rel_rms(y, ref), 0.02)

    def test_dense_synthetic_all_types(self):
        for q in sorted(P.IQ_MMQ_TYPES):
            with self.subTest(qtype=P.IQ_TYPE_NAMES[q]):
                self._dense_case(q, 512)
                self._dense_case(q, 512, dtype=torch.float16, m=9)

    def test_dense_iq4_nl_k_tail(self):
        for k in (128, 384, 640):
            with self.subTest(k=k):
                self._dense_case(P.IQ4_NL, k)

    def test_dense_iq4_nl_never_reads_past_the_row_end(self):
        """The bytes after the tensor are 0xFF (fp16 NaN scale): a loader that read past the last row would return NaN."""
        self._arm(P.IQ4_NL)
        rows, k = 64, 640
        raw = _synth_blocks(P.IQ4_NL, rows, k, seed=7)
        buf = torch.full((raw.size + 8192,), 0xFF, dtype=torch.uint8, device=self.dev)
        buf[: raw.size] = torch.from_numpy(raw.reshape(-1)).to(self.dev)
        w = buf[: raw.size].view(rows, -1)
        x = torch.rand((16, k), dtype=torch.bfloat16, device=self.dev)
        y = self.K.mul_mat_a8(w, x, P.IQ4_NL, rows)
        self.assertTrue(torch.isfinite(y).all())

    # -- MoE -----------------------------------------------------------------------------------------------------------
    def _moe_case(self, qtype, k, rows, E=8, tokens=130, top_k=4, dtype=torch.bfloat16, raws=None, invalid=False):
        self._arm(qtype)
        from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size

        torch.manual_seed(1)
        raw = np.stack([_synth_blocks(qtype, rows, k, seed=100 + e) for e in range(E)]) if raws is None else raws
        wd = torch.stack([self._ref_w(raw[e], qtype) for e in range(E)])
        w = torch.from_numpy(np.ascontiguousarray(raw)).to(self.dev)
        x = torch.rand((tokens, k), dtype=dtype, device=self.dev)
        topk_ids = torch.randint(0, E, (tokens, top_k), dtype=torch.int32, device=self.dev)
        sorted_ids, expert_ids, ntpp = moe_align_block_size(topk_ids, P.IQ_MOE_MMQ_BLOCK_SIZE, E)
        expert_ids = expert_ids.masked_fill(expert_ids >= E, -1)
        y = self.K.moe_a8(x, w, sorted_ids, expert_ids, ntpp, qtype, rows, top_k, tokens)
        ref = torch.einsum("tk,tjnk->tjn", x.float(), wd[topk_ids.long()])  # [tokens, top_k, rows]
        self.assertTrue(torch.isfinite(y).all())
        torch.testing.assert_close(y.float().reshape(tokens, top_k, rows), ref, atol=1.0, rtol=1e-1)  # the PR's MoE bound
        self.assertLess(_rel_rms(y.reshape(tokens, top_k, rows), ref), 0.02)

    def test_moe_synthetic_all_types(self):
        for q in sorted(P.IQ_MMQ_TYPES):
            with self.subTest(qtype=P.IQ_TYPE_NAMES[q]):
                self._moe_case(q, 512, 96)

    def test_moe_iq4_nl_k640_ffn_down_shape(self):
        self._moe_case(P.IQ4_NL, 640, 256, tokens=192, top_k=10, E=16)

    def test_moe_real_nf_row_classes(self):
        """One expert block per NF row class from the real header: A (IQ3_S gate/up, IQ4_NL down), B/C (IQ3_S / IQ4_XS gate/up)."""
        for layer, which in ((0, "gate"), (0, "down"), (4, "up"), (2, "gate")):
            raw, qtype = _real_expert_block(layer, which, n_experts=8)
            if qtype not in P.IQ_MMQ_TYPES:
                continue
            with self.subTest(layer=layer, which=which, qtype=P.IQ_TYPE_NAMES[qtype]):
                k = raw.shape[2] // gguf_lib.GGML_QUANT_SIZES[gguf_lib.GGMLQuantizationType(qtype)][1] * gguf_lib.GGML_QUANT_SIZES[gguf_lib.GGMLQuantizationType(qtype)][0]
                self._moe_case(qtype, k, raw.shape[1], E=raw.shape[0], raws=raw, tokens=160, top_k=4)

    def test_moe_invalid_expert_block_writes_zero(self):
        self._arm(P.IQ3_S)
        rows, k, E = 64, 512, 4
        raw = np.stack([_synth_blocks(P.IQ3_S, rows, k, seed=e) for e in range(E)])
        w = torch.from_numpy(raw).to(self.dev)
        x = torch.rand((1, k), dtype=torch.bfloat16, device=self.dev)
        sorted_ids = torch.tensor([0, 1, 1, 1], dtype=torch.int32, device=self.dev)
        expert_ids = torch.tensor([-1], dtype=torch.int32, device=self.dev)
        ntpp = torch.tensor([4], dtype=torch.int32, device=self.dev)
        y = self.K.moe_a8(x, w, sorted_ids, expert_ids, ntpp, P.IQ3_S, rows, 1, 1)
        torch.testing.assert_close(y, torch.zeros_like(y), atol=0, rtol=0)

    def test_k_alignment_is_refused_by_the_host(self):
        self._arm(P.IQ3_S)
        w = torch.zeros((8, 110 * 2), dtype=torch.uint8, device=self.dev)
        x = torch.zeros((9, 384), dtype=torch.bfloat16, device=self.dev)  # 384 % 256 != 0
        with self.assertRaisesRegex(Exception, "requires an input size divisible by"):
            self.K.mul_mat_a8(w, x, P.IQ3_S, 8)


if __name__ == "__main__":
    unittest.main()
