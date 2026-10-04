# SPDX-License-Identifier: Apache-2.0
"""Auftrag 1301: sm_89 Triton offline-compile self-test (no GPU, no driver).

SM89-DURCHSPIEL-1002 listed "do generic Triton fp8-KV kernels compile on sm_89"
as needs-hardware ("fp8e4nv -> bf16 emits cvt.bf16.f16, ptxas takes it only from
sm_90"). Triton can compile for a ``GPUTarget`` without a device, so the
question is answerable at the desk. Measured with Triton 3.6.0 + its bundled
ptxas 12.8 (the installed env, no GPU):

* sm_86 REFUSES the type itself (``type fp8e4nv not supported in this
  architecture``) -- the reason the rig's 3080 ranks decode fp8 by bytes;
* sm_89 COMPILES every fp8e4nv conversion shape below (-> bf16 / f32 / f16, <-
  bf16 / f32, masked load + where, fp8 operand cast into tl.dot) and the PLE
  staged gather in its native form (FP8_DECODE=3) -- the ptxas error recorded in
  qwen4_exp_ple_fp8.NATIVE_FP8_MIN_ARCH does NOT reproduce here;
* the QSA rows kernel at sm_89: (16, 1, 2) spills (STACK > 0), (32, 8, 2),
  (64, 8, 2), (32, 4, 2) do not -- the basis of sparse_attn._SM89_ROWS_CONFIGS.

What this does NOT prove: the kernels RUN correctly (that needs an Ada card), nor
that the image's Triton/ptxas (CUDA 13) agrees with this env's; it pins the
compile-level facts so a Triton bump that changes them is seen at the desk.
The test skips when Triton or ptxas is not usable here.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:  # pragma: no cover -- environment gate
    import triton
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    _TRITON_OK = True
except Exception:  # noqa: BLE001
    _TRITON_OK = False

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=120, suite="base-a-test-cpu")

if _TRITON_OK:

    @triton.jit
    def _k_to(x_ptr, y_ptr, BLOCK: tl.constexpr, OUT: tl.constexpr):
        o = tl.arange(0, BLOCK)
        tl.store(y_ptr + o, tl.load(x_ptr + o).to(OUT))

    @triton.jit
    def _k_masked_where(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        o = tl.arange(0, BLOCK)
        m = o < n
        a = tl.load(x_ptr + o, mask=m, other=0.0).to(tl.bfloat16)
        b = tl.load(x_ptr + o + 7, mask=m, other=0.0).to(tl.bfloat16)
        v = tl.where(n > 3, a, b)
        tl.store(y_ptr + o, tl.where(n > 1, v, 0.0), mask=m)

    @triton.jit
    def _k_dot(a_ptr, b_ptr, c_ptr, BLOCK: tl.constexpr):
        o = tl.arange(0, BLOCK)
        a = tl.load(a_ptr + o[:, None] * BLOCK + o[None, :])
        b = tl.load(b_ptr + o[:, None] * BLOCK + o[None, :])
        tl.store(c_ptr + o[:, None] * BLOCK + o[None, :], tl.dot(a, b.to(tl.bfloat16)))


def _compile(fn, signature, constexprs, cc, **options):
    sig = dict(signature)
    sig.update({k: "constexpr" for k in constexprs})
    return triton.compile(
        ASTSource(fn=fn, signature=sig, constexprs=constexprs),
        target=GPUTarget("cuda", cc, 32),
        options=options or None,
    )


def _toolchain_ready() -> bool:
    if not _TRITON_OK:
        return False
    try:
        _compile(_k_to, {"x_ptr": "*bf16", "y_ptr": "*fp32"},
                 {"BLOCK": 32, "OUT": tl.float32}, 89)
        return True
    except Exception:  # noqa: BLE001 -- no ptxas / no driver stub: skip, never fail
        return False


_READY = _toolchain_ready()


def _cuobjdump():
    if not _TRITON_OK:
        return None
    p = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "cuobjdump")
    return p if os.path.exists(p) else None


def _res_usage(compiled):
    """(REG, STACK) from ``cuobjdump -res-usage`` of the compiled cubin."""
    tool = _cuobjdump()
    if tool is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as f:
        f.write(compiled.asm["cubin"])
        path = f.name
    try:
        out = subprocess.run([tool, "-res-usage", path], capture_output=True, text=True).stdout
    finally:
        os.unlink(path)
    for line in out.splitlines():
        if "REG:" in line and "STACK:" in line:
            fields = dict(p.split(":", 1) for p in line.split() if ":" in p)
            return int(fields["REG"]), int(fields["STACK"])
    return None


@unittest.skipUnless(_READY, "triton + ptxas not usable for an offline GPUTarget compile here")
class TestFp8e4nvOfflineCompile(unittest.TestCase):
    CONVERSIONS = [
        ("fp8->bf16", "*fp8e4nv", "*bf16", "bfloat16"),
        ("fp8->f32", "*fp8e4nv", "*fp32", "float32"),
        ("fp8->f16", "*fp8e4nv", "*fp16", "float16"),
        ("bf16->fp8", "*bf16", "*fp8e4nv", "float8e4nv"),
        ("f32->fp8", "*fp32", "*fp8e4nv", "float8e4nv"),
    ]

    def test_sm86_refuses_the_type_so_the_bytes_decoders_stay_the_sm86_path(self):
        for name, x, y, out in self.CONVERSIONS:
            with self.assertRaises(Exception) as cm:
                _compile(_k_to, {"x_ptr": x, "y_ptr": y},
                         {"BLOCK": 32, "OUT": getattr(tl, out)}, 86)
            self.assertIn("fp8e4nv not supported in this architecture", str(cm.exception), name)

    def test_sm89_compiles_every_fp8e4nv_conversion(self):
        for name, x, y, out in self.CONVERSIONS:
            c = _compile(_k_to, {"x_ptr": x, "y_ptr": y},
                         {"BLOCK": 32, "OUT": getattr(tl, out)}, 89)
            self.assertGreater(len(c.asm["cubin"]), 0, name)

    def test_sm89_compiles_masked_where_and_fp8_into_dot(self):
        c = _compile(_k_masked_where,
                     {"x_ptr": "*fp8e4nv", "y_ptr": "*bf16", "n": "i32"}, {"BLOCK": 128}, 89)
        self.assertGreater(len(c.asm["cubin"]), 0)
        c = _compile(_k_dot, {"a_ptr": "*bf16", "b_ptr": "*fp8e4nv", "c_ptr": "*fp32"},
                     {"BLOCK": 32}, 89)
        self.assertGreater(len(c.asm["cubin"]), 0)

    def test_ple_staged_gather_native_form_compiles_for_sm89(self):
        """The recorded ptxas 'cvt.bf16.f16 needs sm_90' error does not reproduce
        with this Triton/ptxas: both the native (3) and the byte decode (1)
        forms compile for sm_89. NATIVE_FP8_MIN_ARCH stays 90 (a metal run
        decides); this pins the offline fact."""
        from sglang.srt.models.qwen4_exp_ple_decode_pread import (
            _gather_ple_embedding_staged_kernel as k,
        )

        sig = {"bases_ptr": "*i64", "shard_rows": "i64", "ids_ptr": "*i64",
               "stage_addrs_ptr": "*i64", "counters_ptr": "*i32", "output_ptr": "*bf16",
               "embedding_dim": "i32", "tp_vocab_start": "i64", "tp_vocab_end": "i64"}
        for decode in (3, 1):
            c = _compile(k, sig, {"is_fp8": True, "BLOCK_D": 256, "FP8_DECODE": decode}, 89)
            self.assertGreater(len(c.asm["cubin"]), 0, decode)


@unittest.skipUnless(_READY and _cuobjdump() is not None,
                     "triton + ptxas + cuobjdump not usable here")
class TestQsaRowsKernelOnSm89(unittest.TestCase):
    HD, HQ, HKV = 256, 16, 2

    def _rows(self, cc, block_n, warps, stages):
        from sglang.srt.layers.attention.qsa.sparse_attn import _sparse_attn_rows_fwd as k

        hd, hq, hkv = self.HD, self.HQ, self.HKV
        group = hq // hkv
        cexpr = {
            "sq_m": hq * hd, "sq_h": hd, "sq_d": 1, "sk_n": hkv * hd, "sk_h": hd, "sk_d": 1,
            "sv_n": hkv * hd, "sv_h": hd, "sv_d": 1, "so_m": hq * hd, "so_h": hd, "so_d": 1,
            "sl_m": hq, "sl_h": 1, "sr_m": 2051, "sr_n": 1, "NUM_KV_HEADS": hkv,
            "GROUP_SIZE": group, "BLOCK_M": max(16, group), "BLOCK_N": block_n,
            "HEAD_DIM": hd, "KV_FP8": True, "USE_COUNTS": False, "FP8_DECODE": 0,
        }
        sig = {"q": "*bf16", "k": "*u8", "v": "*u8", "out": "*bf16", "lse": "*fp32",
               "rows": "*i32", "scale": "fp32", "topk": "i32", "counts": "*i32"}
        return _res_usage(_compile(k, sig, cexpr, cc, num_warps=warps, num_stages=stages))

    def test_the_l20_top_row_spills_on_sm89_the_h101_row_does_not(self):
        reg, stack = self._rows(89, 16, 1, 2)
        self.assertEqual(reg, 255)
        self.assertGreater(stack, 0, "(16,1,2) is the spilling build on sm89 too")
        for cfg in ((32, 8, 2), (64, 8, 2), (32, 4, 2)):
            reg, stack = self._rows(89, *cfg)
            self.assertEqual(stack, 0, cfg)
            self.assertLess(reg, 255, cfg)

    def test_sm89_and_sm86_build_the_same_shape_of_kernel(self):
        """sm86 (the rig's 3080) is the measured reference; sm89 reproduces its
        spill / no-spill split per config, which is why the sm86 P stages' (64,8,2)
        is the safe sm89 default."""
        for cfg in ((16, 1, 2), (32, 8, 2), (64, 8, 2), (32, 4, 2)):
            r86, s86 = self._rows(86, *cfg)
            r89, s89 = self._rows(89, *cfg)
            self.assertEqual(s86 > 0, s89 > 0, cfg)


if __name__ == "__main__":
    unittest.main()
