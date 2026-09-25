"""H68d: the fp8 PLE table on sm86 (SGLANG_WEG2_PLE_FP8_DECODE, qwen4_exp_ple_fp8.py).

The nvidia NVFP4 export keeps the PLE n-gram table in float8_e4m3fn. Every PLE
gather typed it as ``tl.float8e4nv``, which Triton refuses below sm89 -- the
3080 of the NVFP4 slice smoke died at its first forward
(h68b_slice_x172s_card0, "type fp8e4nv not supported in this architecture").
The four gathers (pinned, shards, H40 staged, H69 gated) now read the bytes as
uint8 below sm90 and decode them with the H65 QSA decoders; from sm90 on they
keep the native pointer.

CPU only, four pins:
* the selection: grammar, per-arch default, refusals, the launch argument,
  the one proof line per (arch, mode);
* offline compile (Triton 3.6, no GPU): on sm86 the native read fails with
  exactly the metal error, every byte decode builds (ptx included, so ptxas
  saw the inline asm); on sm120 the native read builds;
* the four kernels under Triton's interpreter on host memory against torch's
  float8_e4m3fn -> bfloat16 conversion over all 256 codes, for exp2 and bits
  (the ptx decoder is pinned by test_qsa_rows_fp8_decode_h65's PTX
  interpreter; it cannot run in Triton's) -- table rows, staged rows, gate
  closed and open, out-of-range ids;
* the wiring: every launch passes FP8_DECODE from ple_fp8_decode_arg, and
  every kernel that types a pointer fp8e4nv does so only under
  ``if FP8_DECODE == 3:`` (bf16 and native branches are the pre-H68d code;
  hfmeta/ple_fp8_ptx_identity.py showed their PTX unchanged at the desk).
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import contextlib
import inspect
import logging
import re
import unittest
from unittest import mock

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.runtime.interpreter import InterpretedFunction

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa import sparse_attn as sa
from sglang.srt.models import qwen4_exp as qe
from sglang.srt.models import qwen4_exp_ple_decode_pread as dp
from sglang.srt.models import qwen4_exp_ple_fp8 as pf8
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=90, suite="base-a-test-cpu")

DIM = 160
BLOCK_D = 256
ROWS_PER_SHARD = 4
SHARDS = 2
TOTAL = ROWS_PER_SHARD * SHARDS
LOGGER = "sglang.srt.models.qwen4_exp_ple_fp8"


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


class TestSelection(CustomTestCase):
    def test_default_native_from_sm90_bits_below(self):
        p = pf8.parse_ple_fp8_decode
        self.assertEqual(p("", 86), pf8.PLE_FP8_BITS)
        self.assertEqual(p("", 80), pf8.PLE_FP8_BITS)
        # sm89 types fp8e4nv but its fp8e4nv -> bf16 needs sm90's cvt.bf16.f16
        self.assertEqual(p("", 89), pf8.PLE_FP8_BITS)
        self.assertEqual(p("", 90), pf8.PLE_FP8_NATIVE)
        self.assertEqual(p("", 120), pf8.PLE_FP8_NATIVE)
        self.assertEqual(p("", None), pf8.PLE_FP8_BITS)  # no GPU (interpreter)

    def test_grammar_arch_group_wins_over_generic(self):
        p = pf8.parse_ple_fp8_decode
        self.assertEqual(p("ptx", 86), pf8.PLE_FP8_PTX)
        self.assertEqual(p("ptx", 120), pf8.PLE_FP8_PTX)
        self.assertEqual(p("sm86:ptx", 86), pf8.PLE_FP8_PTX)
        self.assertEqual(p("sm86:ptx", 120), pf8.PLE_FP8_NATIVE)  # untouched arch: default
        self.assertEqual(p("sm86:exp2; bits", 120), pf8.PLE_FP8_BITS)
        self.assertEqual(p("bits;sm86:exp2", 86), pf8.PLE_FP8_EXP2)
        self.assertEqual(p("sm120:bits;sm86:ptx", 120), pf8.PLE_FP8_BITS)
        self.assertEqual(p(" SM86:PTX ".lower(), 86), pf8.PLE_FP8_PTX)
        # the QSA numbering: one decoder authority, one set of numbers
        self.assertEqual(
            (pf8.PLE_FP8_EXP2, pf8.PLE_FP8_BITS, pf8.PLE_FP8_PTX),
            (sa.FP8_DECODE_EXP2, sa.FP8_DECODE_BITS, sa.FP8_DECODE_PTX),
        )

    def test_refusals_by_name(self):
        p = pf8.parse_ple_fp8_decode
        with self.assertRaisesRegex(ValueError, r"'native' needs sm90\+.*sm86"):
            p("native", 86)
        with self.assertRaisesRegex(ValueError, r"'native' needs sm90\+.*sm89"):
            p("sm89:native", 89)
        with self.assertRaisesRegex(ValueError, r"'ptx' needs sm80\+"):
            p("ptx", 75)
        with self.assertRaisesRegex(ValueError, "mode must be one of"):
            p("sm86:fast", 86)
        self.assertEqual(p("native", None), pf8.PLE_FP8_NATIVE)  # no arch: nothing to refuse

    def test_launch_argument_and_proof_line(self):
        self.assertEqual(pf8.ple_fp8_decode_arg(torch.bfloat16, "cpu"), pf8.PLE_FP8_NATIVE)
        self.assertEqual(pf8.ple_fp8_decode_arg(torch.float8_e4m3fn, "cpu"), pf8.PLE_FP8_BITS)
        for cap, raw, want in (((8, 6), "", pf8.PLE_FP8_BITS),
                               ((8, 6), "sm86:ptx", pf8.PLE_FP8_PTX),
                               ((12, 0), "", pf8.PLE_FP8_NATIVE),
                               ((12, 0), "sm86:ptx", pf8.PLE_FP8_NATIVE)):
            with mock.patch.dict(pf8._ARCH_OF, {}, clear=True), \
                    mock.patch.dict(pf8._MODE_CACHE, {}, clear=True), \
                    mock.patch.object(pf8, "_NOTED", set()), \
                    mock.patch.object(pf8.torch.cuda, "get_device_capability", return_value=cap), \
                    envs.SGLANG_WEG2_PLE_FP8_DECODE.override(raw), \
                    self.assertLogs(LOGGER, level=logging.INFO) as logs:
                dev = torch.device("cuda", 0)
                self.assertEqual(pf8.ple_fp8_decode_arg(torch.float8_e4m3fn, dev), want)
                self.assertEqual(pf8.ple_fp8_decode_arg(torch.float8_e4m3fn, dev), want)
            lines = [r.getMessage() for r in logs.records if "PLE-FP8-DECODE" in r.getMessage()]
            self.assertEqual(len(lines), 1, lines)  # one line per (arch, mode), not per launch
            self.assertIn(f"arch=sm{cap[0] * 10 + cap[1]} mode={pf8._NAMES[want]}", lines[0])


# --------------------------------------------------------------------------
# offline compile: the metal error reproduced, the byte decodes build
# --------------------------------------------------------------------------

_COMMON = {"embedding_dim": "i32", "tp_vocab_start": "i32", "tp_vocab_end": "i32"}
_KERNELS = {
    "pinned": (lambda: qe._gather_ple_embedding_from_pinned_kernel,
               {"weight_ptr": "i64", "ids_ptr": "*i64", "output_ptr": "*bf16", **_COMMON}),
    "shards": (lambda: qe._gather_ple_embedding_from_shards_kernel,
               {"bases_ptr": "*i64", "shard_rows": "i64", "ids_ptr": "*i64", "output_ptr": "*bf16", **_COMMON}),
    "staged": (lambda: dp._gather_ple_embedding_staged_kernel,
               {"bases_ptr": "*i64", "shard_rows": "i64", "ids_ptr": "*i64", "stage_addrs_ptr": "*i64",
                "counters_ptr": "*i32", "output_ptr": "*bf16", **_COMMON}),
    "gated": (lambda: dp._gather_ple_embedding_gated_kernel,
              {"bases_ptr": "*i64", "shard_rows": "i64", "ids_ptr": "*i64", "stage_addrs_ptr": "*i64",
               "go_ptr": "*i32", "counters_ptr": "*i32", "output_ptr": "*bf16", **_COMMON}),
}


def _compile(name, arch, fp8_decode, is_fp8=True):
    fn, sig = _KERNELS[name]
    sig = dict(sig)
    const = {"is_fp8": is_fp8, "BLOCK_D": BLOCK_D, "FP8_DECODE": fp8_decode}
    for k in const:
        sig[k] = "constexpr"
    return triton.compile(ASTSource(fn=fn(), signature=sig, constexprs=const),
                          target=GPUTarget("cuda", arch, 32))


def _chain(exc):
    out, e = [], exc
    while e is not None and len(out) < 8:
        out.append(str(e))
        e = e.__cause__ or e.__context__
    return " | ".join(out)


class TestOfflineCompile(CustomTestCase):
    def test_sm86_native_is_the_metal_error_and_every_byte_decode_builds(self):
        for name in _KERNELS:
            with self.subTest(kernel=name):
                with self.assertRaises(Exception) as ctx:
                    _compile(name, 86, pf8.PLE_FP8_NATIVE)
                self.assertIn("fp8e4nv not supported in this architecture", _chain(ctx.exception))
                for mode in (pf8.PLE_FP8_BITS, pf8.PLE_FP8_PTX):
                    k = _compile(name, 86, mode)
                    self.assertNotIn("e4m3", k.asm["ptx"])  # no hardware fp8 conversion on sm86
                # the pre-H68d bf16 table still builds on sm86
                _compile(name, 86, pf8.PLE_FP8_NATIVE, is_fp8=False)
        _compile("shards", 86, pf8.PLE_FP8_EXP2)

    def test_sm120_native_builds(self):
        for name in _KERNELS:
            with self.subTest(kernel=name):
                k = _compile(name, 120, pf8.PLE_FP8_NATIVE)
                self.assertIn("e4m3", k.asm["ptx"])  # the hardware conversion, as before H68d


# --------------------------------------------------------------------------
# the kernels under the interpreter, all 256 codes
# --------------------------------------------------------------------------


@contextlib.contextmanager
def _interpreted_helpers():
    """The byte branch calls the decoder (a jit function, and it calls the
    H65 decoders); the interpreter runs a called jit function only when it is
    an InterpretedFunction, so they are swapped in their modules' globals for
    the test. The bf16/native branches call none (H40/H69 interpret them
    unpatched)."""
    wrap = lambda f: InterpretedFunction(f.fn)  # noqa: E731
    dec = wrap(pf8.ple_fp8_bytes_to_bf16)
    with mock.patch.object(qe, "ple_fp8_bytes_to_bf16", dec), \
            mock.patch.object(dp, "ple_fp8_bytes_to_bf16", dec), \
            mock.patch.object(pf8, "_fp8_e4m3_bytes_to_f32", wrap(sa._fp8_e4m3_bytes_to_f32)), \
            mock.patch.object(pf8, "_fp8_e4m3_bytes_to_f32_bits", wrap(sa._fp8_e4m3_bytes_to_f32_bits)):
        yield


def _expected(u8: torch.Tensor, mode: int) -> torch.Tensor:
    """torch's float8_e4m3fn -> bfloat16 bits; the byte decodes give +0.0 for
    0x80 (the H65 exp2/bits contract)."""
    bits = u8.view(torch.float8_e4m3fn).to(torch.bfloat16).view(torch.int16).clone()
    if mode != pf8.PLE_FP8_NATIVE:
        bits[u8 == 0x80] = 0
    return bits


def _assert_rows(test, got_bf16, want_bits, u8):
    got = got_bf16.contiguous().view(torch.int16)
    nan = (u8 & 0x7F) == 0x7F
    test.assertTrue(bool(torch.isnan(got_bf16[nan].float()).all()))
    test.assertTrue(torch.equal(got[~nan], want_bits[~nan]))


class _Table:
    """TOTAL rows of DIM fp8 bytes in SHARDS host shards; every code 0..255
    appears (row r, lane i = (r * DIM + i) % 256)."""

    def __init__(self):
        codes = torch.arange(TOTAL * DIM, dtype=torch.int64) % 256
        self.u8 = codes.to(torch.uint8).reshape(TOTAL, DIM)
        self.shards = [self.u8[s * ROWS_PER_SHARD:(s + 1) * ROWS_PER_SHARD].clone() for s in range(SHARDS)]
        self.bases = torch.tensor([t.data_ptr() for t in self.shards], dtype=torch.int64)
        self.flat = self.u8.clone()  # the pinned kernel's contiguous table


# the byte decodes only: the interpreter's own fp8e4nv -> bf16 conversion is
# no reference (it turns the NaN codes into numbers); the native read is pinned
# by the offline compile above and runs on the 5090 as before H68d
MODES = (pf8.PLE_FP8_EXP2, pf8.PLE_FP8_BITS)


class TestKernelsInterpreted(CustomTestCase):
    def setUp(self):
        self.t = _Table()
        # every row, twice in mixed order, plus one id out of range
        self.ids = torch.tensor(list(range(TOTAL)) + list(reversed(range(TOTAL))) + [TOTAL + 5], dtype=torch.int64)
        self.n = int(self.ids.numel())

    def _want(self, mode):
        rows = self.t.u8[self.ids.clamp(max=TOTAL - 1)]
        return rows, _expected(rows, mode)

    def _check(self, out, mode):
        rows, want = self._want(mode)
        _assert_rows(self, out[: self.n - 1], want[: self.n - 1], rows[: self.n - 1])
        self.assertTrue(bool((out[self.n - 1].float() == 0).all()))  # out of range -> zeros

    def test_pinned_and_shards_all_codes(self):
        pinned = InterpretedFunction(qe._gather_ple_embedding_from_pinned_kernel.fn)
        shards = InterpretedFunction(qe._gather_ple_embedding_from_shards_kernel.fn)
        with _interpreted_helpers():
            for mode in MODES:
                with self.subTest(mode=mode):
                    out = torch.full((self.n, DIM), 3.0, dtype=torch.bfloat16)
                    pinned[(self.n,)](self.t.flat.data_ptr(), self.ids, out, embedding_dim=DIM,
                                      tp_vocab_start=0, tp_vocab_end=TOTAL, is_fp8=True,
                                      BLOCK_D=BLOCK_D, FP8_DECODE=mode)
                    self._check(out, mode)
                    out = torch.full((self.n, DIM), 3.0, dtype=torch.bfloat16)
                    shards[(self.n,)](self.t.bases, ROWS_PER_SHARD, self.ids, out, embedding_dim=DIM,
                                      tp_vocab_start=0, tp_vocab_end=TOTAL, is_fp8=True,
                                      BLOCK_D=BLOCK_D, FP8_DECODE=mode)
                    self._check(out, mode)

    def _stage(self):
        """Stage rows for every EVEN slot: the id matches for slot r when
        ids[r] is even; the staged bytes are the table row's bytes XOR 0x01
        so a row taken from the stage is told apart from a table read."""
        stage_ids = torch.full((self.n,), -1, dtype=torch.int64)
        stage_rows = torch.zeros((self.n, DIM), dtype=torch.uint8)
        hit = torch.zeros(self.n, dtype=torch.bool)
        for r in range(self.n - 1):
            gid = int(self.ids[r])
            if gid % 2 == 0:
                stage_ids[r] = gid
                stage_rows[r] = self.t.u8[gid] ^ 0x01
                hit[r] = True
        return stage_ids, stage_rows, hit

    def _want_staged(self, stage_rows, hit, mode, use_stage=True):
        rows = self.t.u8[self.ids.clamp(max=TOTAL - 1)].clone()
        if use_stage:
            rows[hit] = stage_rows[hit]
        return rows, _expected(rows, mode)

    def test_staged_and_gated_all_codes(self):
        staged = InterpretedFunction(dp._gather_ple_embedding_staged_kernel.fn)
        gated = InterpretedFunction(dp._gather_ple_embedding_gated_kernel.fn)
        stage_ids, stage_rows, hit = self._stage()
        addrs = torch.tensor([stage_ids.data_ptr(), stage_rows.data_ptr()], dtype=torch.int64)
        with _interpreted_helpers():
            for mode in MODES:
                with self.subTest(mode=mode, kernel="staged"):
                    out = torch.full((self.n, DIM), 3.0, dtype=torch.bfloat16)
                    ctr = torch.zeros(2, dtype=torch.int32)
                    staged[(self.n,)](self.t.bases, ROWS_PER_SHARD, self.ids, addrs, ctr, out,
                                      embedding_dim=DIM, tp_vocab_start=0, tp_vocab_end=TOTAL,
                                      is_fp8=True, BLOCK_D=BLOCK_D, FP8_DECODE=mode)
                    rows, want = self._want_staged(stage_rows, hit, mode)
                    _assert_rows(self, out[: self.n - 1], want[: self.n - 1], rows[: self.n - 1])
                    self.assertEqual(ctr.tolist(), [self.n - 1, int(hit.sum())])
                for go in (0, 1):
                    with self.subTest(mode=mode, kernel="gated", go=go):
                        out = torch.full((self.n, DIM), 3.0, dtype=torch.bfloat16)
                        ctr = torch.zeros(2, dtype=torch.int32)
                        gated[(self.n,)](self.t.bases, ROWS_PER_SHARD, self.ids, addrs,
                                         torch.tensor([go], dtype=torch.int32), ctr, out,
                                         embedding_dim=DIM, tp_vocab_start=0, tp_vocab_end=TOTAL,
                                         is_fp8=True, BLOCK_D=BLOCK_D, FP8_DECODE=mode)
                        rows, want = self._want_staged(stage_rows, hit, mode, use_stage=bool(go))
                        _assert_rows(self, out[: self.n - 1], want[: self.n - 1], rows[: self.n - 1])
                        self.assertEqual(ctr.tolist(), [self.n - 1, int(hit.sum()) if go else 0])


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------


class TestWiring(CustomTestCase):
    def test_every_launch_passes_the_decode(self):
        for mod, kernels in ((qe, ("_gather_ple_embedding_from_shards_kernel", "_gather_ple_embedding_from_pinned_kernel")),
                             (dp, ("_gather_ple_embedding_staged_kernel", "_gather_ple_embedding_gated_kernel"))):
            src = inspect.getsource(mod)
            for k in kernels:
                launches = [m.start() for m in re.finditer(re.escape(k) + r"\[", src)]
                self.assertTrue(launches, k)
                for pos in launches:
                    # the launch's own argument list: up to the next statement
                    # that closes a call at the launch's indentation or less
                    end = re.search(r"\n\s*\)\n", src[pos:])
                    call = src[pos: pos + end.end()]
                    self.assertIn("FP8_DECODE=ple_fp8_decode_arg(", call, k)

    def test_every_fp8e4nv_kernel_has_the_byte_branch(self):
        """A PLE kernel that types a pointer fp8e4nv must take FP8_DECODE and
        type it only under ``if FP8_DECODE == 3:`` -- the next kernel copied
        from the pre-H68d pattern would kill the 3080 ranks again."""
        seen = 0
        for mod in (qe, dp):
            for name, obj in vars(mod).items():
                fn = getattr(obj, "fn", None)
                if not isinstance(obj, triton.runtime.jit.JITFunction) or fn is None:
                    continue
                src = inspect.getsource(fn)
                if "float8e4nv" not in src:
                    continue
                seen += 1
                self.assertIn("FP8_DECODE", obj.arg_names, name)
                lines = src.splitlines()
                for i, line in enumerate(lines):
                    if "float8e4nv" in line:
                        guard = [l for l in lines[:i] if l.strip().startswith(("if ", "else:", "elif "))][-1]
                        self.assertEqual(guard.strip(), "if FP8_DECODE == 3:", f"{name}: {line.strip()}")
        self.assertEqual(seen, 4)  # pinned, shards, staged, gated


if __name__ == "__main__":
    unittest.main()
