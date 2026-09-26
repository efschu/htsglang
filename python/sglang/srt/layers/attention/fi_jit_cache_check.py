"""READ-ONLY check: will loading group P's flashinfer prefill module compile?

Operator order 24.09.: nothing under /root/.cache is ever written by a check
(the NF seat shares the JIT cache), and a GPU test in a window must only load
and compute (seconds, < 3 GiB). A flashinfer JIT build of the hd256 prefill
module takes several GiB of host RAM (the 19:42Z window test was OOM-killed at
the 3 GiB cgroup cap).

WHY A STANDALONE TEST REBUILDS WHAT A BOOT LOADS (the root cause, measured):
flashinfer fixes its cache DIRECTORY at import from the visible device
(flashinfer/jit/env.py + jit/core.py:138 -> ``0.6.14/120f`` for the 5090) but
takes the -gencode FLAGS from FLASHINFER_CUDA_ARCH_LIST when it writes a
module's build.ninja (jit/cpp_ext.py:210, a fresh CompilationContext). The
sglang server sets that variable AFTER importing flashinfer --
``model_runner.py:2494 set_cuda_arch()`` -> ``"12.0a"`` (utils/common.py:1547)
-- so every boot builds and loads ``120f`` modules with ``compute_120a``. A
bare test process never calls set_cuda_arch(), writes ``compute_120f``, and
ninja rebuilds the whole module (the 120f bf16 module's build.ninja was
rewritten with compute_120f at 19:55Z; its objects are still the 14.08. ones).
:func:`mirror_server_arch` reproduces the server's order.

The check, all reads plus ``ninja -n`` on a COPY of the ninja state:
1. exactly ONE visible GPU (the cache dir is keyed by the visible arch set;
   seeing the 5090 and a 3080 selects ``86_120f``, whose copies are stale);
2. the module's .so and build.ninja exist in THIS process's directory;
3. the on-disk build.ninja with THIS process's -gencode flags substituted,
   plus copies of .ninja_log / .ninja_deps, in a temp dir; ``ninja -n`` there
   with the ``ninja`` flashinfer runs (first on PATH -- the venv's ninja 1.13
   reads the 1.11-written logs as dirty) must say "no work to do". Outputs and
   inputs are absolute paths, so ninja stats the real files and writes none.

CLI: ``python -m sglang.srt.layers.attention.fi_jit_cache_check [bf16] [e4m3]``
(exit 0 = loads without building).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import List, Sequence, Tuple

_DTYPE_NAME = {"bf16": "bf16", "e4m3": "e4m3", "f16": "f16", "e5m2": "e5m2"}
_GENCODE_RE = re.compile(r"-gencode=arch=compute_\w+,code=sm_\w+")


def prefill_uri(kv: str, q: str = "bf16", o: str = "bf16", head_dim: int = 256) -> str:
    """flashinfer 0.6.14 ``get_batch_prefill_uri`` for fa2, int32 indices, no
    RoPE, no window, no soft cap, no fp16 QK reduction (pure string)."""
    return (
        "batch_prefill_with_kv_cache_dtype_q_%s_dtype_kv_%s_dtype_o_%s_dtype_idx_i32_"
        "head_dim_qk_%d_head_dim_vo_%d_posenc_0_use_swa_False_use_logits_cap_False_f16qk_False"
        % (_DTYPE_NAME[q], _DTYPE_NAME[kv], _DTYPE_NAME[o], head_dim, head_dim)
    )


def mirror_server_arch() -> str:
    """Do what the sglang server does before its first flashinfer build:
    import flashinfer (cache dir from the visible device), THEN set
    FLASHINFER_CUDA_ARCH_LIST the way model_runner does. Returns the value."""
    import flashinfer  # noqa: F401  (the directory is fixed by this import)

    from sglang.srt.utils.common import set_cuda_arch

    set_cuda_arch()
    return os.environ.get("FLASHINFER_CUDA_ARCH_LIST", "")


def substitute_gencode(build_ninja: str, want: Sequence[str]) -> str:
    """The on-disk build.ninja with its -gencode flags replaced by ``want``
    (the only arch-dependent content: a desk diff of a regenerated file against
    an untouched one differs in exactly that line)."""
    found = _GENCODE_RE.findall(build_ninja)
    if not found:
        return build_ninja
    out = build_ninja
    first = True
    for flag in found:
        out = out.replace(flag, " ".join(want) if first else "", 1)
        first = False
    return out


def ninja_would_build(module_dir: str, build_ninja_text: str, ninja: str) -> Tuple[bool, str]:
    """``ninja -n`` against a temp copy of the ninja state; nothing in
    ``module_dir`` is written."""
    tmp = tempfile.mkdtemp(prefix="fi_jit_check_")
    try:
        with open(os.path.join(tmp, "build.ninja"), "w") as fh:
            fh.write(build_ninja_text)
        for name in (".ninja_log", ".ninja_deps"):
            src = os.path.join(module_dir, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(tmp, name))
        r = subprocess.run(
            [ninja, "-n", "-C", tmp, "-f", os.path.join(tmp, "build.ninja")],
            capture_output=True,
            text=True,
        )
        last = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
        tail = last[-1][:160] if last else ""
        return ("no work to do" not in (r.stdout or "")), tail
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_prefill_modules(
    kv_dtypes: Sequence[str] = ("bf16", "e4m3"), mirror_server: bool = True
) -> Tuple[bool, List[str]]:
    lines: List[str] = []
    try:
        import torch

        n = torch.cuda.device_count()
    except Exception as exc:  # noqa: BLE001
        return False, ["no CUDA: %s" % (exc,)]
    if n != 1:
        return False, [
            "%d visible GPUs: the JIT cache is keyed by the visible arch set; run with "
            "exactly one (CUDA_VISIBLE_DEVICES=<uuid>)" % n
        ]
    arch = mirror_server_arch() if mirror_server else os.environ.get("FLASHINFER_CUDA_ARCH_LIST", "")
    from flashinfer.compilation_context import CompilationContext
    from flashinfer.jit import env as jit_env

    want = [f for f in CompilationContext().get_nvcc_flags_list() if f.startswith("-gencode")]
    ninja = shutil.which("ninja")
    lines.append("jit dir %s, FLASHINFER_CUDA_ARCH_LIST=%r -> %s, ninja %s" % (
        jit_env.FLASHINFER_JIT_DIR, arch, want, ninja))
    if ninja is None:
        return False, lines + ["no ninja on PATH"]
    ok = True
    for kv in kv_dtypes:
        uri = prefill_uri(kv)
        d = jit_env.FLASHINFER_JIT_DIR / uri
        so, bn = d / (uri + ".so"), d / "build.ninja"
        if not so.exists() or not bn.exists():
            lines.append("%s: NOT BUILT here (would compile)" % kv)
            ok = False
            continue
        on_disk = bn.read_text()
        would, tail = ninja_would_build(str(d), substitute_gencode(on_disk, want), ninja)
        note = "" if all(g in on_disk for g in want) else " (build.ninja on disk carries other flags; flashinfer will rewrite it)"
        lines.append("%s: %s%s" % (kv, ("WOULD COMPILE: " + tail) if would else "loads without building", note))
        ok = ok and not would
    return ok, lines


def main(argv: Sequence[str] = ()) -> int:
    kvs = list(argv) or ["bf16", "e4m3"]
    ok, lines = check_prefill_modules(kvs)
    for line in lines:
        print(line)
    print("LOADS WITHOUT BUILDING" if ok else "WOULD BUILD OR CANNOT CHECK -- do not run the GPU test")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
