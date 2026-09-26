"""Build the sm_86 W4A8 NVFP4 JIT modules without a GPU (Docker image pre-build).

Under ``--fp4-gemm-backend native-mixed`` every sm_8x rank computes the NVFP4 linears -- main model and NVFP4
draft -- on two tvm-ffi JIT modules (user order 25.09. ~17:33Z):

* ``nvfp4_w4a8_decode_sm86`` (``sglang.jit_kernel.nvfp4_w4a8_decode``, agent N4D): M <= 48;
* ``nvfp4_w4a8_sm86``        (``sglang.jit_kernel.nvfp4_w4a8``, agent N4A): M > 48 and the activation quantiser.

Both are content-addressed under ``$TVM_FFI_CACHE_DIR`` (default ``~/.cache/tvm-ffi``). Built here with the
image's own nvcc and the target forced to 8.6 (``override_jit_cuda_arch``), the first boot finds a complete entry
with provenance and loads it instead of compiling. No CUDA driver is needed: the artefacts link libcudart only.

  python -m sglang.jit_kernel.prebuild_nvfp4_w4a8 [--arch 8.6] [--report out.json]

Exit status 0 only if every module left a ``<name>.so`` and its provenance record in the cache.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time


def _modules():
    from sglang.jit_kernel import nvfp4_w4a8, nvfp4_w4a8_decode

    return (
        ("nvfp4_w4a8_decode_sm86", nvfp4_w4a8_decode._jit_module),
        ("nvfp4_w4a8_sm86", nvfp4_w4a8._jit_nvfp4_w4a8_module),
    )


def _entry_of(cache: pathlib.Path, name: str):
    """The newest complete cache entry of module ``name`` (dir with <mod>.so + provenance), or None."""
    best = None
    for d in cache.glob(f"sgl_kernel_jit_{name}_*"):
        sos = list(d.glob("*.so"))
        if sos and (d / "sgl_jit_provenance.json").is_file():
            if best is None or d.stat().st_mtime > best.stat().st_mtime:
                best = d
    return best


def build(arch: str = "8.6") -> dict:
    from sglang.jit_kernel.utils import override_jit_cuda_arch

    major, minor = (int(v) for v in arch.split("."))
    cache = pathlib.Path(os.environ.get("TVM_FFI_CACHE_DIR", "~/.cache/tvm-ffi")).expanduser()
    rows = []
    with override_jit_cuda_arch(major, minor):
        for name, fn in _modules():
            t0 = time.time()
            err = None
            try:
                fn()
            except Exception as e:  # noqa: BLE001 -- reported, and the entry check below decides
                err = f"{type(e).__name__}: {e}"[:400]
            entry = _entry_of(cache, name)
            rows.append(
                {
                    "module": name,
                    "arch": arch,
                    "entry": str(entry) if entry else None,
                    "ok": entry is not None,
                    "error": err,
                    "seconds": round(time.time() - t0, 1),
                }
            )
    return {"cache": str(cache), "rows": rows, "ok": all(r["ok"] for r in rows)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", default="8.6")
    ap.add_argument("--report", default="")
    ns = ap.parse_args(argv)
    rep = build(ns.arch)
    txt = json.dumps(rep, indent=1)
    print(txt)
    if ns.report:
        pathlib.Path(ns.report).write_text(txt)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
