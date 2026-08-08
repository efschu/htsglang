#!/usr/bin/env python
"""Pre-serving GPU sanity guard for the APU laptop (#651).

MEASURED 2026-08-08: after a suspend/resume cycle the gfx1103 K-quant defect
family WIDENS -- Q5_K dequantize, byte-identical across runs on a fresh boot,
goes non-deterministically wrong (~2e-2) until the machine is REBOOTED. A
server booted in that state serves garbage with HTTP 200s.

This guard runs the cheapest known discriminator (8-run Q5_K dequantize
determinism + Q4_K correctness, ~2 s) and exits non-zero if the GPU is in the
poisoned state. Boot scripts MUST run it first and refuse to serve on failure.

  source /root/lh/venv/bin/activate
  HSA_OVERRIDE_GFX_VERSION=11.0.0 PYTHONPATH=/root/lh/ggufbuild \
    python gpu_sanity_guard.py || exit 1

Exit codes: 0 = sane; 1 = POISONED (reboot the machine); 2 = cannot test.
"""

import json
import os
import sys

import numpy as np
import torch
import gguf
from gguf.constants import GGMLQuantizationType as QT

FIXTURE_DIR = os.environ.get("GGUF_FIXTURE_DIR", "/root/lh/ggufbuild")
NRUNS = 8


def main() -> int:
    try:
        import gguf_rocm_probe as K
    except ImportError as exc:
        print(f"GUARD: cannot import kernel extension: {exc}")
        return 2

    try:
        slices = json.load(open(os.path.join(FIXTURE_DIR, "slices.json")))
    except OSError as exc:
        print(f"GUARD: fixtures unavailable: {exc}")
        return 2

    os.chdir(FIXTURE_DIR)  # slice paths are relative
    by_name = {s["name"]: s for s in slices}
    verdicts = []

    # Q5_K determinism is the suspend/resume canary.
    s = by_name["q5_K"]
    raw = np.fromfile(s["path"], dtype=np.uint8).reshape(s["rows"], s["row_bytes"])
    W = torch.from_numpy(raw).cuda()
    outs = []
    for _ in range(NRUNS):
        o = K.ggml_dequantize(W, s["type"], s["rows"], s["cols"], torch.float16, None)
        torch.cuda.synchronize()
        outs.append(o.float().cpu().numpy())
    ident = all(np.array_equal(np.nan_to_num(outs[0]), np.nan_to_num(o)) for o in outs[1:])
    verdicts.append(("Q5_K dequant 8-run byte-identical", ident))

    # Q4_K correctness vs the numpy oracle.
    s = by_name["q4_K"]
    raw = np.fromfile(s["path"], dtype=np.uint8).reshape(s["rows"], s["row_bytes"])
    ref = gguf.quants.dequantize(raw, QT.Q4_K).astype(np.float32)
    W = torch.from_numpy(raw).cuda()
    o = K.ggml_dequantize(W, s["type"], s["rows"], s["cols"], torch.float16, None)
    torch.cuda.synchronize()
    err = float(np.abs(np.nan_to_num(o.float().cpu().numpy(), posinf=0, neginf=0) - ref).max())
    verdicts.append((f"Q4_K dequant correct (max|d| {err:.2e})", err < 1e-3))

    ok = all(v for _, v in verdicts)
    for label, v in verdicts:
        print(f"GUARD: {'PASS' if v else 'FAIL'}  {label}")
    if not ok:
        print("GUARD: GPU IS IN THE POISONED STATE (suspend/resume defect family).")
        print("GUARD: REBOOT THE MACHINE before serving or measuring. Refusing to serve.")
        return 1
    print("GUARD: GPU sane.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
