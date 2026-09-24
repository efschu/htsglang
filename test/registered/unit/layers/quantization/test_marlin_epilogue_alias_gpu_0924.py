"""Dense Marlin epilogue alias -- window stress test. NEEDS CUDA, operator window only.

Run (seconds per card; LOADS the gptq_marlin / gptq_marlin_repack JIT modules
the boots already built, compiles nothing, writes nothing under the JIT cache):
  TEST27B_GPU=1 CUDA_VISIBLE_DEVICES=<uuid of ONE card> PATH=/usr/local/cuda/bin:/usr/bin:/bin \
  /root/.claude/jobs/1ab4cd30/tmp/test27b.sh env PYTHONPATH=<tree>/python \
  /spinning/htsglang-gpu/.venv/bin/python -m pytest -q -s -p no:cacheprovider \
  test/registered/unit/layers/quantization/test_marlin_epilogue_alias_gpu_0924.py
once on the 5090 (sm_120, GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d) and once on
a 3080 (sm_86, UUID from ``nvidia-smi -L``). The JIT cache is keyed by the
visible arch, so exactly one card may be visible.

JIT hygiene (K's fi_jit_cache_check rule, 24.09.: a check writes nothing under
/root/.cache, a window test only loads and computes). The module-level
pre-check runs the REAL ``load_jit`` path of both loaders with every writer of
that path replaced by a refusal -- ``heal_entry`` / ``building_marker`` (the
build path), ``purge_entry`` (provenance mismatch or unloadable .so),
``_selfheal_jit_cache_once`` (the residue sweep) -- and with ``load_module``
replaced by a probe: only "a cached .so with matching provenance exists" lets
the file run; anything else skips with the reason. The same refusals stay in
place while the tests run, so an unexpected write path raises instead of
writing. The refusal is a BaseException on purpose: ``load_jit`` wraps
``load_module`` in ``except Exception: purge_entry(...)``.

What it measures. The desk proof (test_marlin_epilogue_alias_0924.py) says the
alias ``sh_red == sh_b`` can only hit a dead prefetch, so the UNCHANGED kernel
must be clean. This hammers that kernel at the running DFlash2-W8 draft's
per-rank shapes (compressed-tensors W8A16 g128 -> Marlin uint8b128, bf16
activations; D group rank-tp-ratio 58,25,25 as logged by xsn436, split derived
from it: q 16/8/8 and kv 4/2/2 heads of 128, MLP 73/32/31 units of 128 --
representative, the kernel config is what matters) plus two small-N/large-K
shapes that take the atomicAdd reduce on sm_90+, over M = 1/8/16 (decode: the
DFlash block is 8 tokens per request) and 256/2048 (draft extend), in four
arms: the production dispatch ``apply_gptq_marlin_linear`` and the explicit
fp32-reduce, bf16-reduce and (sm_90+ only, as in production) atomicAdd paths.

Weights are 128 +- 4 around the uint8b128 bias, i.e. |w| <= 4 * scale, so any
byte the kernel reads that is NOT a weight (a reduction partial read as int8)
is ~32x larger than a real one: a live alias hit cannot hide inside rounding.
The reference dequantizes exactly as Marlin does ((q-128) in bf16 times the
bf16 group scale, rounded to bf16) and multiplies in fp32, so the only honest
difference is bf16 output rounding (and bf16 partials on the non-fp32 arms).

FAILS on: any non-finite output, any element beyond the arm's bound. REPORTS
(does not fail on): bitwise run-to-run differences inside the bound -- the
#190 class (gptq_marlin_gemm on sm80..88 is not run-to-run reproducible at
M >= 128, ~1 %, measured 2026-07-26, 4de61b5c6f), which is accumulation order,
not garbage. A red here is corruption on the running draft path, whatever its
mechanism -- escalate, do not re-run until green.

Env knobs (defaults are the 27B draft): SGLANG_MARLIN_ALIAS_TEST_SHAPES="N:K,..."
SGLANG_MARLIN_ALIAS_TEST_MS="1,8,..." SGLANG_MARLIN_ALIAS_TEST_GROUP=128
SGLANG_MARLIN_ALIAS_TEST_REPS=20. The NF line runs the same dense template
byte for byte (its dense layers: group 64, pass its own N:K).
"""

import contextlib
import os
import time
import unittest

import pytest
import torch

if not torch.cuda.is_available():  # pragma: no cover - desk
    pytest.skip("needs CUDA (window test: TEST27B_GPU=1)", allow_module_level=True)
if torch.cuda.device_count() != 1:  # pragma: no cover - window hygiene
    pytest.skip(
        "exactly ONE visible GPU (the JIT cache is keyed by its arch); visible=%d"
        % torch.cuda.device_count(),
        allow_module_level=True,
    )

import sglang.jit_kernel.utils as JU  # noqa: E402
from sglang.jit_kernel import gptq_marlin as GM  # noqa: E402
from sglang.jit_kernel import gptq_marlin_repack as GR  # noqa: E402


class _Refused(BaseException):
    """Not an Exception: load_jit's ``except Exception`` must not swallow it."""


@contextlib.contextmanager
def read_only_jit_cache():
    def refuse(tag):
        def _f(*args, **kwargs):
            raise _Refused("%s %s" % (tag, args[0] if args else ""))

        return _f

    patch = {
        "heal_entry": refuse("WOULD-BUILD"),
        "building_marker": refuse("WOULD-BUILD"),
        "purge_entry": refuse("WOULD-PURGE"),
        "_selfheal_jit_cache_once": lambda: [],
    }
    saved = {name: getattr(JU, name) for name in patch}
    for name, fn in patch.items():
        setattr(JU, name, fn)
    try:
        yield
    finally:
        for name, fn in saved.items():
            setattr(JU, name, fn)


def precheck(loader, *args):
    """Would ``loader`` load a cached .so without writing? -> (ok, detail)."""
    import tvm_ffi

    real = tvm_ffi.load_module

    def probe(path):
        raise _Refused("LOADS %s" % (path,))

    tvm_ffi.load_module = probe
    try:
        with read_only_jit_cache():
            loader.__wrapped__(*args)  # bypass the per-arch memo
        return False, "loader returned without reaching load_module"
    except _Refused as r:
        return str(r).startswith("LOADS "), str(r)
    finally:
        tvm_ffi.load_module = real


_OK_GEMM, _WHY_GEMM = precheck(GM._jit_gptq_marlin_module, torch.bfloat16)
_OK_REPACK, _WHY_REPACK = precheck(GR._jit_gptq_marlin_repack_module)
if not (_OK_GEMM and _OK_REPACK):  # pragma: no cover - window hygiene
    pytest.skip(
        "JIT module not loadable read-only here (no build/purge in a window): "
        "gemm: %s | repack: %s" % (_WHY_GEMM, _WHY_REPACK),
        allow_module_level=True,
    )

from sgl_kernel.scalar_type import scalar_types  # noqa: E402

from sglang.srt.layers.quantization import marlin_utils as MU  # noqa: E402
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=90, suite="nightly-1-gpu")

DEV = torch.device("cuda", 0)
CAP = torch.cuda.get_device_capability(0)

# (label, N, K) per D rank -- see the module docstring for the derivation.
DRAFT_SHAPES = [
    ("tp0.qkv", 3072, 5120),
    ("tp0.o", 5120, 2048),
    ("tp0.gate_up", 18688, 5120),
    ("tp0.down", 5120, 9344),
    ("tp12.qkv", 1536, 5120),
    ("tp12.o", 5120, 1024),
    ("tp1.gate_up", 8192, 5120),
    ("tp2.gate_up", 7936, 5120),
    ("tp1.down", 5120, 4096),
    ("tp2.down", 5120, 3968),
    ("fc", 5120, 25600),
]
EXTRA_SHAPES = [("small_n.a", 512, 17408), ("small_n.b", 1024, 8192)]


def _env_list(name, default, conv):
    raw = os.environ.get(name, "").strip()
    return [conv(x) for x in raw.split(",") if x.strip()] if raw else list(default)


SHAPES = (
    [("env", int(nk.split(":")[0]), int(nk.split(":")[1])) for nk in os.environ["SGLANG_MARLIN_ALIAS_TEST_SHAPES"].split(",")]
    if os.environ.get("SGLANG_MARLIN_ALIAS_TEST_SHAPES", "").strip()
    else DRAFT_SHAPES + EXTRA_SHAPES
)
MS = _env_list("SGLANG_MARLIN_ALIAS_TEST_MS", (1, 8, 16, 256, 2048), int)
GROUP = int(os.environ.get("SGLANG_MARLIN_ALIAS_TEST_GROUP", "128"))
REPS = int(os.environ.get("SGLANG_MARLIN_ALIAS_TEST_REPS", "20"))

# Bounds, element-wise: rel * |ref| + absf * rms(ref).
# TIGHT: fp32 reduce -- the output is ONE bf16 rounding of an fp32 sum of exact
#   bf16 x bf16 products, so |err| <= 2^-9 |ref| deterministically.
# LOOSE: bf16 partials (bf16 global reduce, atomicAdd) -- one bf16 rounding per
#   k-slice of a random-walk partial sum; err std ~ 2^-9 rms sqrt((slices+1)/6),
#   ~5 sigma over 10^7 elements -> 0.013 rms at 8 slices, hence 0.03.
# A weight byte read from the alias would add ~1.6 rms (|w| <= 4 s vs <= 128 s).
TIGHT = (0.008, 0.002)
LOOSE = (0.03, 0.03)


def make_w8(n, k, group, seed):
    """Marlin uint8b128 weights at 128 +- 4, their Marlin form, and the exact
    bf16 dequant reference (fp32 tensor of the bf16 values)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randint(124, 133, (k, n), device=DEV, generator=g, dtype=torch.int64)
    s = (torch.rand((k // group, n), device=DEV, generator=g) * 0.02 + 0.01).to(torch.bfloat16)
    w_ref = ((q - 128).to(torch.bfloat16) * s.repeat_interleave(group, dim=0)).float()
    qv = q.view(k // 4, 4, n)  # GPTQ packing along K: row 4r+i at bits 8i of word r
    packed = qv[:, 0] | (qv[:, 1] << 8) | (qv[:, 2] << 16) | (qv[:, 3] << 24)
    packed = torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32).contiguous()
    del q, qv
    perm = torch.empty(0, dtype=torch.int, device=DEV)
    b_q = GR.gptq_marlin_repack(packed, perm, k, n, 8)
    b_s = MU.marlin_permute_scales(s, k, n, group)
    return b_q, b_s, w_ref


class TestDenseMarlinEpilogueAliasOnMetal(unittest.TestCase):
    # Plain unittest.TestCase (K's lesson): CustomTestCase's retry() hides the cause.

    @classmethod
    def setUpClass(cls):
        print(
            "\n[marlin-alias] device=%s cap=%d.%d sms=%d | %s | %s"
            % (
                torch.cuda.get_device_name(0),
                CAP[0],
                CAP[1],
                torch.cuda.get_device_properties(0).multi_processor_count,
                _WHY_GEMM,
                _WHY_REPACK,
            )
        )

    def _arms(self, m, n, k):
        prod_atomic = MU.should_use_atomic_add_reduce(m=m, n=n, k=k, device=DEV, dtype=torch.bfloat16)
        arms = [
            ("prod", None, None, LOOSE if prod_atomic else TIGHT),
            ("fp32", False, True, TIGHT),
            ("bf16red", False, False, LOOSE),
        ]
        if CAP[0] >= 9:  # production only takes atomicAdd with bf16 on sm_90+
            arms.append(("atomic", True, True, LOOSE))
        return arms, prod_atomic

    def test_draft_shapes_are_clean(self):
        empty = torch.empty(0, dtype=torch.int, device=DEV)
        workspace = MU.marlin_make_workspace(DEV)
        wtype = scalar_types.uint8b128
        failures, t0 = [], time.time()
        with read_only_jit_cache():
            for si, (label, n, k) in enumerate(SHAPES):
                b_q, b_s, w_ref = make_w8(n, k, GROUP, seed=1000 + si)
                for m in MS:
                    g = torch.Generator(device=DEV).manual_seed(7 * m + si)
                    a = torch.randn((m, k), device=DEV, generator=g).to(torch.bfloat16)
                    ref = a.float() @ w_ref
                    rms = ref.pow(2).mean().sqrt().item()
                    arms, prod_atomic = self._arms(m, n, k)
                    for arm, atomic, fp32, (rel, absf) in arms:
                        bound = rel * ref.abs() + absf * rms
                        base, worst, bitdiff, nonfinite = None, 0.0, 0, 0
                        for _ in range(REPS):
                            if arm == "prod":
                                out = MU.apply_gptq_marlin_linear(
                                    a, b_q, b_s, empty, empty, empty, workspace, wtype,
                                    output_size_per_partition=n, input_size_per_partition=k,
                                    is_k_full=True, bias=None,
                                )
                            else:
                                out = GM.gptq_marlin_gemm(
                                    a, None, b_q, b_s, None, empty, empty, empty, workspace, wtype,
                                    size_m=m, size_n=n, size_k=k, is_k_full=True,
                                    use_atomic_add=atomic, use_fp32_reduce=fp32, is_zp_float=False,
                                )
                            if not torch.isfinite(out).all():
                                nonfinite += 1
                            worst = max(worst, ((out.float() - ref).abs() / bound).max().item())
                            if base is None:
                                base = out.clone()
                            elif not torch.equal(out, base):
                                bitdiff += 1
                        torch.cuda.synchronize()
                        line = "%-11s N=%-5d K=%-5d M=%-4d %-7s ratio=%.3f bitdiff=%d/%d nonfinite=%d%s" % (
                            label, n, k, m, arm, worst, bitdiff, REPS - 1, nonfinite,
                            " (prod takes atomicAdd)" if arm == "prod" and prod_atomic else "")
                        print("[marlin-alias] " + line)
                        if nonfinite or worst > 1.0:
                            failures.append(line)
                del b_q, b_s, w_ref
                torch.cuda.empty_cache()
        print("[marlin-alias] %d shapes x %d M, %d reps, %.1f s, %d red"
              % (len(SHAPES), len(MS), REPS, time.time() - t0, len(failures)))
        self.assertEqual(failures, [], "corrupted Marlin output on the draft path:\n" + "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
