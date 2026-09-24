"""fnFL2 H65: SGLANG_WEG2_QSA_FP8_DECODE -- the in-kernel fp8 decode of the QSA
rows kernel (qsa/sparse_attn.py).

The rows kernel decodes every selected fp8 K/V byte once per (query, kv head)
program; offline compiled (Triton 3.6, cuobjdump), the default exp2 decode is
~23 SASS instructions per element and ~95 % of the loop of every spill-free
launch config on sm86 and sm120. Two decodes with the SAME value for all 256
codes replace it on request: "bits" (fp32 bit construction) and "ptx" (packed
inline PTX, four bytes per instance). Default = exp2, unchanged.

CPU only, three independent pins:
* the PTX decode: a mini PTX interpreter executes the asm text OF THE KERNEL
  (read from the source, not a hand-written twin) for all 256 codes in all
  four byte lanes against torch's float8_e4m3fn -> bfloat16 conversion;
* the exp2 and bits decodes: run under Triton's interpreter (a private copy
  of the module loaded with TRITON_INTERPRET=1) against torch's conversion;
* the selection: grammar, per-arch choice, the launch wiring and the default
  (FP8_DECODE=0, the table config) with a recorded launch.
The metal tests at the end (skipped without CUDA) are for the GPU window.
"""

import importlib.util
import inspect
import os
import re
import struct
import unittest
from unittest import mock

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa import sparse_attn as sa
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _torch_bf16_bits(codes):
    """float8_e4m3fn codes -> bfloat16 bit patterns (int, 0..0xFFFF), with the
    one deliberate difference to torch: 0x80 is +0.0. The default exp2 decode
    negates with Triton's ``-x`` = ``0 - x`` and has always produced +0.0
    there; the new decodes match the kernel, not torch, bit for bit."""
    t = torch.tensor(codes, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.bfloat16)
    bits = [int(x) & 0xFFFF for x in t.view(torch.int16).tolist()]
    return [0 if b == 0x8000 else b for b in bits]


def _is_bf16_nan(h):
    return (h & 0x7F80) == 0x7F80 and (h & 0x7F) != 0


# --------------------------------------------------------------------------
# The PTX decode, executed from the kernel's own asm text.
# --------------------------------------------------------------------------


def _ptx_program():
    src = inspect.getsource(sa._fp8_e4m3_bytes_to_bf16_ptx.fn)
    m = re.search(r'asm="""(.*?)"""', src, re.S)
    assert m, "the PTX decode no longer carries its asm as a literal"
    prog = []
    for line in m.group(1).splitlines():
        line = line.split("//")[0].strip()
        if not line or line in ("{", "}") or line.startswith(".reg"):
            continue
        assert line.endswith(";"), line
        op, _, rest = line[:-1].partition(" ")
        prog.append((op.strip(), [a.strip() for a in rest.split(",")]))
    return prog


def _bf16_value(h):
    return struct.unpack("<f", struct.pack("<I", (h & 0xFFFF) << 16))[0]


def _bf16_bits_exact(x):
    """bf16 bits of x; x must be exactly representable (the decode is exact)."""
    if x != x:
        return 0x7FC0
    b = struct.unpack("<I", struct.pack("<f", x))[0]
    assert b & 0xFFFF == 0 and _bf16_value(b >> 16) == x, f"{x!r} is not a bf16 value"
    return b >> 16


def _f16_value(h):
    return float(np.array([h & 0xFFFF], dtype=np.uint16).view(np.float16)[0])


def _run_ptx(prog, word):
    """Execute the decode's PTX for one input word; returns ($0, $1)."""
    regs = {"$2": word & 0xFFFFFFFF}

    def val(x):
        if x in regs:
            return regs[x]
        if re.fullmatch(r"0[xX][0-9a-fA-F]+", x):
            return int(x, 16)
        if re.fullmatch(r"\d+", x):
            return int(x)
        raise KeyError(f"PTX operand {x!r} read before written")

    def halves(v):
        return v & 0xFFFF, (v >> 16) & 0xFFFF

    for op, args in prog:
        d = args[0]
        if op == "mov.b32":
            regs[d] = val(args[1]) & 0xFFFFFFFF
        elif op == "prmt.b32":
            a, b, c = (val(x) for x in args[1:4])
            src = (b << 32) | a
            out = 0
            for i in range(4):
                sel = (c >> (4 * i)) & 0xF
                byte = (src >> (8 * (sel & 7))) & 0xFF
                if sel & 8:
                    byte = 0xFF if byte & 0x80 else 0
                out |= byte << (8 * i)
            regs[d] = out
        elif op == "shr.b32":
            regs[d] = (val(args[1]) & 0xFFFFFFFF) >> val(args[2])
        elif op == "and.b32":
            regs[d] = val(args[1]) & val(args[2]) & 0xFFFFFFFF
        elif op == "lop3.b32":
            a, b, c, lut = (val(x) for x in args[1:5])
            out = 0
            for bit in range(32):
                idx = (((a >> bit) & 1) << 2) | (((b >> bit) & 1) << 1) | ((c >> bit) & 1)
                out |= ((lut >> idx) & 1) << bit
            regs[d] = out
        elif op == "fma.rn.bf16x2":
            ha, hb, hc = (halves(val(x)) for x in args[1:4])
            res = [
                _bf16_bits_exact(_bf16_value(x) * _bf16_value(y) + _bf16_value(z))
                for x, y, z in zip(ha, hb, hc)
            ]
            regs[d] = res[0] | (res[1] << 16)
        elif op == "set.eq.u32.f16x2":
            ha, hb = (halves(val(x)) for x in args[1:3])
            res = [0xFFFF if _f16_value(x) == _f16_value(y) else 0 for x, y in zip(ha, hb)]
            regs[d] = res[0] | (res[1] << 16)
        else:
            raise AssertionError(f"PTX op {op!r} has no emulation -- extend the test with it")
    return regs["$0"], regs["$1"]


class PtxDecodeTest(unittest.TestCase):
    def test_every_code_in_every_lane_matches_torch(self):
        prog = _ptx_program()
        want = _torch_bf16_bits(list(range(256)))
        filler = [0x3C, 0xC1, 0x00, 0xFE]  # 1.5, -2.25, 0, -448
        checked = 0
        for code in range(256):
            for lane in range(4):
                codes = list(filler)
                codes[lane] = code
                word = sum(c << (8 * i) for i, c in enumerate(codes))
                o0, o1 = _run_ptx(prog, word)
                got = [o0 & 0xFFFF, o0 >> 16, o1 & 0xFFFF, o1 >> 16]
                for i, c in enumerate(codes):
                    if _is_bf16_nan(want[c]):
                        self.assertTrue(_is_bf16_nan(got[i]), f"code {c:#04x} lane {i}: {got[i]:#06x} not NaN")
                    else:
                        self.assertEqual(got[i], want[c], f"code {c:#04x} lane {i}")
                    checked += 1
        self.assertEqual(checked, 256 * 4 * 4)
        # the anchors spelled out: subnormal min, normal min, max, both zeros
        o0, _ = _run_ptx(prog, 0x08010080)  # lanes: 0x80, 0x00, 0x01, 0x08
        self.assertEqual((o0 & 0xFFFF, o0 >> 16), (0x0000, 0x0000))
        _, o1 = _run_ptx(prog, 0x08010080)
        self.assertEqual(_bf16_value(o1 & 0xFFFF), 2.0 ** -9)
        self.assertEqual(_bf16_value(o1 >> 16), 2.0 ** -6)
        o0, _ = _run_ptx(prog, 0x0000FE7E)
        self.assertEqual((_bf16_value(o0 & 0xFFFF), _bf16_value(o0 >> 16)), (448.0, -448.0))

    def test_the_emulated_program_is_the_kernels(self):
        ops = [op for op, _ in _ptx_program()]
        self.assertIn("fma.rn.bf16x2", ops)
        self.assertIn("set.eq.u32.f16x2", ops)
        # 7 working instructions per 32-bit output word, plus the constants
        self.assertEqual(sum(op != "mov.b32" for op in ops), 14)


# --------------------------------------------------------------------------
# The exp2 and bits decodes under Triton's interpreter.
# --------------------------------------------------------------------------

_INTERP = {}


def _decode_codes_kernel(x_ptr, y_ptr, MODE: tl.constexpr):
    offs = tl.arange(0, 256)
    x = tl.load(x_ptr + offs)
    if MODE == 1:
        y = _INTERP["bits"](x)
    else:
        y = _INTERP["exp2"](x)
    tl.store(y_ptr + offs, y)


def _interpreted():
    """A private copy of sparse_attn with every @triton.jit interpreted, and
    the test kernel above -- the module the rest of the process imported
    stays compiled."""
    if "kernel" in _INTERP:
        return _INTERP["kernel"]
    old = os.environ.get("TRITON_INTERPRET")
    os.environ["TRITON_INTERPRET"] = "1"
    try:
        spec = importlib.util.spec_from_file_location("_qsa_sparse_attn_interp_h65", sa.__file__)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        kernel = triton.jit(_decode_codes_kernel)
    finally:
        if old is None:
            os.environ.pop("TRITON_INTERPRET", None)
        else:
            os.environ["TRITON_INTERPRET"] = old
    if type(kernel).__name__ != "InterpretedFunction":
        raise unittest.SkipTest("this Triton does not honour TRITON_INTERPRET at decoration")
    _INTERP["bits"] = mod._fp8_e4m3_bytes_to_f32_bits
    _INTERP["exp2"] = mod._fp8_e4m3_bytes_to_f32
    _INTERP["kernel"] = kernel
    return kernel


class InterpretedDecodeTest(unittest.TestCase):
    def _decode(self, mode):
        kernel = _interpreted()
        x = torch.arange(256, dtype=torch.int32).to(torch.uint8)
        y = torch.empty(256, dtype=torch.float32)
        kernel[(1,)](x, y, MODE=mode)
        return y

    def _check(self, got):
        ref = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).float()
        nan = torch.isnan(ref)
        self.assertEqual(int(nan.sum()), 2)
        self.assertTrue(torch.equal(nan, torch.isnan(got)))
        self.assertTrue(torch.equal(got[~nan], ref[~nan]))
        # sign of zero: Triton's -x is 0 - x, so 0x80 is +0.0 (torch: -0.0)
        self.assertFalse(bool(torch.signbit(got[0x00])) or bool(torch.signbit(got[0x80])))
        signs = torch.signbit(got[~nan])
        self.assertTrue(torch.equal(signs, torch.signbit(ref[~nan]) & (ref[~nan] != 0)))

    def test_bits_decode_matches_torch_for_every_code(self):
        self._check(self._decode(1))

    def test_default_exp2_decode_matches_torch_for_every_code(self):
        self._check(self._decode(0))

    def test_bits_equals_the_default_decode_bit_for_bit(self):
        a, b = self._decode(1), self._decode(0)
        live = ~torch.isnan(b)
        self.assertTrue(torch.equal(a[live].view(torch.int32), b[live].view(torch.int32)))


# --------------------------------------------------------------------------
# Selection: grammar, arch, wiring, default.
# --------------------------------------------------------------------------


class ParseTest(unittest.TestCase):
    def test_modes_and_arch_groups(self):
        p = sa.parse_fp8_decode
        self.assertEqual(p("", 86), sa.FP8_DECODE_EXP2)
        self.assertEqual(p("ptx", 86), sa.FP8_DECODE_PTX)
        self.assertEqual(p("bits", 120), sa.FP8_DECODE_BITS)
        self.assertEqual(p("sm86:ptx", 86), sa.FP8_DECODE_PTX)
        self.assertEqual(p("sm86:ptx", 120), sa.FP8_DECODE_EXP2)
        # an arch group wins over a generic one, in either order
        self.assertEqual(p("bits;sm86:ptx", 86), sa.FP8_DECODE_PTX)
        self.assertEqual(p("sm86:ptx;bits", 86), sa.FP8_DECODE_PTX)
        self.assertEqual(p("sm86:ptx;bits", 120), sa.FP8_DECODE_BITS)
        self.assertEqual(p("sm86:ptx;sm120:exp2", 120), sa.FP8_DECODE_EXP2)
        self.assertEqual(p(" PTX ", 120), sa.FP8_DECODE_PTX)

    def test_refusals_name_the_problem(self):
        with self.assertRaisesRegex(ValueError, "mode must be one of"):
            sa.parse_fp8_decode("fast", 86)
        with self.assertRaisesRegex(ValueError, "needs sm80"):
            sa.parse_fp8_decode("ptx", 75)
        # bits has no arch floor
        self.assertEqual(sa.parse_fp8_decode("bits", 75), sa.FP8_DECODE_BITS)


class _RecordedKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


class LaunchTest(unittest.TestCase):
    def setUp(self):
        sa._FP8_DECODE_CACHE.clear()
        sa._ROWS_CONFIG_CACHE.clear()
        sa._ROWS_LAUNCH_SEEN.clear()
        self.addCleanup(sa._FP8_DECODE_CACHE.clear)
        self.addCleanup(sa._ROWS_CONFIG_CACHE.clear)
        self.addCleanup(sa._ROWS_LAUNCH_SEEN.clear)

    def _launch(self, capability, kv_dtype=torch.float8_e4m3fn, total_q=16384, decode="", config=""):
        rec = _RecordedKernel()
        q = torch.zeros(total_q, 24, 8, dtype=torch.bfloat16)
        k = torch.zeros(32, 2, 8).to(kv_dtype)
        rows = torch.zeros(total_q, 4, dtype=torch.int32)
        with envs.SGLANG_WEG2_QSA_FP8_DECODE.override(decode), \
                envs.SGLANG_FORCE_QSA_ROWS_CONFIG.override(config), \
                mock.patch.object(sa, "_sparse_attn_rows_fwd", rec), \
                mock.patch.object(sa.torch.cuda, "get_device_capability", lambda *a: capability), \
                mock.patch.object(sa.torch.cuda, "get_device_name", lambda *a: "NVIDIA GeForce RTX 3080"):
            sa.sparse_attn_rows_triton(q, k, k.clone(), rows, 0.3)
        self.assertEqual(len(rec.calls), 1)
        grid, _, kw = rec.calls[0]
        self.assertEqual(grid, (total_q, 2))
        return kw

    def test_default_launch_is_the_exp2_decode_and_the_table(self):
        self.assertEqual(envs.SGLANG_WEG2_QSA_FP8_DECODE.get(), "")
        for cap in ((8, 6), (12, 0)):
            kw = self._launch(cap)
            self.assertIs(kw["KV_FP8"], True)
            self.assertEqual(kw["FP8_DECODE"], sa.FP8_DECODE_EXP2)
            self.assertEqual((kw["BLOCK_N"], kw["num_warps"], kw["num_stages"]), (16, 1, 2))

    def test_switch_moves_only_the_named_arch(self):
        kw86 = self._launch((8, 6), decode="sm86:ptx")
        kw120 = self._launch((12, 0), decode="sm86:ptx")
        self.assertEqual(kw86["FP8_DECODE"], sa.FP8_DECODE_PTX)
        self.assertEqual(kw120["FP8_DECODE"], sa.FP8_DECODE_EXP2)
        self.assertEqual(self._launch((12, 0), decode="bits")["FP8_DECODE"], sa.FP8_DECODE_BITS)

    def test_a_16bit_pool_never_takes_an_fp8_decode(self):
        kw = self._launch((8, 6), kv_dtype=torch.bfloat16, decode="ptx")
        self.assertIs(kw["KV_FP8"], False)
        self.assertEqual(kw["FP8_DECODE"], sa.FP8_DECODE_EXP2)

    def test_decode_and_launch_config_compose(self):
        kw = self._launch((8, 6), decode="ptx", config="inf=64/8/2")
        self.assertEqual(kw["FP8_DECODE"], sa.FP8_DECODE_PTX)
        self.assertEqual((kw["BLOCK_N"], kw["num_warps"], kw["num_stages"]), (64, 8, 2))

    def test_one_launch_line_per_form(self):
        with self.assertLogs(sa.logger, level="INFO") as logs:
            self._launch((8, 6), decode="sm86:ptx", config="inf=64/8/2")
            self._launch((8, 6), decode="sm86:ptx", config="inf=64/8/2")
        lines = [r.getMessage() for r in logs.records if "QSA-ROWS-LAUNCH" in r.getMessage()]
        self.assertEqual(len(lines), 1)
        self.assertIn("arch=sm86 kv=fp8 decode=ptx cfg=64/8/2 first_total_q=16384", lines[0])
        self.assertIn("SGLANG_WEG2_QSA_FP8_DECODE='sm86:ptx'", lines[0])

    def test_kernel_default_is_the_exp2_branch(self):
        """The kernel's own default (FP8_DECODE=0) runs the unchanged exp2
        decode -- the default path's SASS equals the pre-H65 kernel's."""
        src = inspect.getsource(sa._sparse_attn_rows_fwd.fn)
        self.assertIn("FP8_DECODE: tl.constexpr = 0", src)
        block = src[src.index("if KV_FP8:"):src.index("scores = tl.where")]
        self.assertRegex(
            block,
            r"else:\s+keys = _fp8_e4m3_bytes_to_f32\(keys_raw\)\.to\(q_values\.dtype\)\s+"
            r"values = _fp8_e4m3_bytes_to_f32\(values_raw\)\.to\(q_values\.dtype\)",
        )
        self.assertIn("if FP8_DECODE == 2:", block)
        self.assertIn("elif FP8_DECODE == 1:", block)


# --------------------------------------------------------------------------
# Metal (GPU window only; skipped on CPU).
# --------------------------------------------------------------------------


@triton.jit
def _metal_decode_kernel(x_ptr, y_ptr, MODE: tl.constexpr):
    offs = tl.arange(0, 256)
    x = tl.load(x_ptr + offs)
    if MODE == 2:
        y = sa._fp8_e4m3_bytes_to_bf16_ptx(x).to(tl.float32)
    elif MODE == 1:
        y = sa._fp8_e4m3_bytes_to_f32_bits(x)
    else:
        y = sa._fp8_e4m3_bytes_to_f32(x)
    tl.store(y_ptr + offs, y)


@unittest.skipUnless(torch.cuda.is_available(), "metal: run in a GPU window, once per arch")
class MetalTest(unittest.TestCase):
    def test_every_code_decodes_exactly_on_the_metal(self):
        x = torch.arange(256, dtype=torch.int32, device="cuda").to(torch.uint8)
        ref = x.view(torch.float8_e4m3fn).float()
        nan = torch.isnan(ref)
        ys = {}
        for mode in (0, 1, 2):
            y = torch.empty(256, dtype=torch.float32, device="cuda")
            _metal_decode_kernel[(1,)](x, y, MODE=mode, num_warps=2)
            self.assertTrue(torch.equal(torch.isnan(y), nan), mode)
            self.assertTrue(torch.equal(y[~nan], ref[~nan]), mode)
            ys[mode] = y
        for mode in (1, 2):  # bit for bit the default decode, 0x80 -> +0.0 included
            self.assertTrue(torch.equal(ys[mode][~nan].view(torch.int32), ys[0][~nan].view(torch.int32)), mode)

    def test_decodes_are_bit_identical_through_the_rows_kernel(self):
        g = torch.Generator(device="cuda").manual_seed(65)
        tq, hq, hkv, d, n, topk = 700, 24, 2, 256, 4096, 2051
        q = torch.randn(tq, hq, d, device="cuda", generator=g).bfloat16()
        k = (torch.randn(n, hkv, d, device="cuda", generator=g) * 3).to(torch.float8_e4m3fn)
        v = (torch.randn(n, hkv, d, device="cuda", generator=g) * 3).to(torch.float8_e4m3fn)
        rows = torch.randint(0, n, (tq, topk), device="cuda", generator=g, dtype=torch.int32)
        rows[:, 2048:] = -1
        for config in ("", "inf=64/8/2", "inf=32/4/2"):
            outs = {}
            for decode in ("exp2", "bits", "ptx"):
                with envs.SGLANG_WEG2_QSA_FP8_DECODE.override(decode), \
                        envs.SGLANG_FORCE_QSA_ROWS_CONFIG.override(config):
                    outs[decode] = sa.sparse_attn_rows_triton(q, k, v, rows, 0.0625)
            for decode in ("bits", "ptx"):
                self.assertTrue(torch.equal(outs[decode][0], outs["exp2"][0]), (config, decode))
                self.assertTrue(torch.equal(outs[decode][1], outs["exp2"][1]), (config, decode))


if __name__ == "__main__":
    unittest.main()
