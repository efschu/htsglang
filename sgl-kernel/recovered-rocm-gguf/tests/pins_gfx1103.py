#!/usr/bin/env python
"""gfx1103 GGUF kernel regression pins (#651, 2026-08-08).

Pins the measured per-kernel x per-type verdict on the Radeon 780M so that
(a) the containment (never route the broken combinations on this GPU) cannot
be silently dropped, and (b) a ROCm/compiler upgrade can be re-judged in
minutes against the same fixtures.

Exit code 0 iff the world still matches KNOWN_BROKEN exactly:
  - every combination NOT in KNOWN_BROKEN must be deterministic (8 runs
    byte-identical) AND correct (rel error < TOL vs the numpy oracle);
  - every combination IN KNOWN_BROKEN must still be broken -- if one turns
    green, that is a FIX (exit 2): update KNOWN_BROKEN and re-validate the
    containment, do not ignore it.

Run on the laptop:
  cd /root/lh/ggufbuild && HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    PYTHONPATH=/root/lh/ggufbuild python pins_gfx1103.py [checkpoint.gguf ...]

Fixtures default to real tensor slices from the checkpoints named on the
command line (one representative tensor per quant type per file), falling
back to the slice_*.bin fixtures beside this script.

Measured baseline being pinned (8-run fixed-input harnesses, 2026-08-08,
fresh boot -- see docs/dev/HANDOFF_651_laptop.md section 12):
  Q6_K:  dequantize / MMVQ / moe_a8 / moe_a8_vec non-deterministic (spreads
         2.3e-2 .. 1.1e-1, non-finite values in dequantize); only MMQ clean.
  IQ2_XS, IQ3_XXS, IQ4_XS: MMQ catastrophically wrong (up to 6.5e+04,
         non-finite, non-deterministic); dequantize and MMVQ clean.
  Q4_K, Q5_K, Q8_0, Q2_K, Q3_K: all five kernels clean.
ALSO PINNED OPERATIONALLY (not testable here): after suspend/resume the
defect family widens (Q5_K dequantize goes non-deterministic) until REBOOT.
The boot guard (gpu_sanity_guard.py) covers that state.
"""

import sys

import numpy as np
import torch
import gguf
from gguf import GGUFReader
from gguf.constants import GGMLQuantizationType as QT

import gguf_rocm_probe as K

NRUNS = 8
ROWS = 512
E, T = 4, 8
TOL_REL = 5e-2  # generous: catches the 1e-1..1e5 defects, not quant noise

#: (type_name, op) combinations that are broken on gfx1103 today.
KNOWN_BROKEN = {
    ("Q6_K", "dequant"),
    ("Q6_K", "mmvq"),
    ("Q6_K", "moe_a8"),
    ("Q6_K", "moe_a8_vec"),
    ("IQ2_XS", "mmq"),
    ("IQ3_XXS", "mmq"),
    ("IQ4_XS", "mmq"),
}


def pick_fixtures(paths):
    picked = {}
    for path in paths:
        reader = GGUFReader(path)
        for t in reader.tensors:
            tt = t.tensor_type
            if tt.name in ("F32", "F16", "BF16") or tt.name in picked:
                continue
            d = t.data
            if d.ndim == 3:
                d = d.reshape(-1, d.shape[-1])
            if d.ndim != 2 or d.shape[0] < ROWS:
                continue
            picked[tt.name] = (int(tt), d[:ROWS].copy())
    return picked


def run_op(call, rf, shape):
    outs = []
    for _ in range(NRUNS):
        o = call()
        torch.cuda.synchronize()
        outs.append(o.float().cpu().numpy().reshape(shape))
    ident = all(
        np.array_equal(np.nan_to_num(outs[0]), np.nan_to_num(o)) for o in outs[1:]
    )
    worst = max(
        np.abs(np.nan_to_num(o, posinf=0, neginf=0) - rf).max() for o in outs
    )
    rel = worst / (np.abs(rf).max() + 1e-12)
    nonfin = max(int((~np.isfinite(o)).sum()) for o in outs)
    return ident and rel < TOL_REL and nonfin == 0, ident, rel, nonfin


def main() -> int:
    paths = sys.argv[1:] or None
    if paths:
        fixtures = pick_fixtures(paths)
    else:
        import json

        fixtures = {}
        for s in json.load(open("slices.json")):
            raw = np.fromfile(s["path"], dtype=np.uint8).reshape(
                s["rows"], s["row_bytes"]
            )
            fixtures[s["name"].upper().replace("Q", "Q", 1)] = (s["type"], raw)
            fixtures = {
                (k if k.startswith("Q") else k): v for k, v in fixtures.items()
            }

    rng = np.random.default_rng(29)
    unexpected_broken = []
    unexpected_fixed = []

    for name, (tid, raw) in sorted(fixtures.items()):
        try:
            ref = gguf.quants.dequantize(raw, QT(tid)).astype(np.float64)
        except Exception:
            continue
        rows, cols = ref.shape
        W = torch.from_numpy(raw).cuda()
        X1 = torch.from_numpy(
            (rng.standard_normal((1, cols), dtype=np.float32) * 0.1)
        ).cuda().half()
        Xb = torch.from_numpy(
            (rng.standard_normal((16, cols), dtype=np.float32) * 0.1)
        ).cuda().half()
        Wm = torch.from_numpy(np.repeat(raw[None], E, axis=0).copy()).cuda()
        Xt = torch.from_numpy(
            (rng.standard_normal((T, cols), dtype=np.float32) * 0.1)
        ).cuda().half()
        topk = torch.from_numpy(
            rng.integers(0, E, size=(T, 1)).astype(np.int32)
        ).cuda()
        b = int(K.ggml_moe_get_block_size(tid))
        has_moe = b > 0
        npad = ((T + b - 1) // b) * b if has_moe else T
        sids = torch.full((npad,), T, dtype=torch.int32)
        sids[:T] = torch.arange(T, dtype=torch.int32)
        eids = torch.full((max(npad // b, 1) if has_moe else 1,), 2, dtype=torch.int32)
        npost = torch.tensor([npad], dtype=torch.int32)
        sids, eids, npost = sids.cuda(), eids.cuda(), npost.cuda()

        ref1 = X1.float().cpu().numpy().astype(np.float64) @ ref.T
        refb = Xb.float().cpu().numpy().astype(np.float64) @ ref.T
        reft = Xt.float().cpu().numpy().astype(np.float64) @ ref.T

        ops = {
            "dequant": (lambda: K.ggml_dequantize(W, tid, rows, cols, torch.float16, None), ref),
            "mmvq": (lambda: K.ggml_mul_mat_vec_a8(W, X1, tid, rows), ref1),
            "mmq": (lambda: K.ggml_mul_mat_a8(W, Xb, tid, rows), refb),
        }
        if has_moe:
            ops["moe_a8"] = (
                lambda: K.ggml_moe_a8(Xt, Wm, sids, eids, npost, tid, rows, 1, T),
                reft,
            )
            ops["moe_a8_vec"] = (
                lambda: K.ggml_moe_a8_vec(Xt, Wm, topk, 1, tid, rows, T),
                reft,
            )

        for op, (call, rf) in ops.items():
            try:
                clean, ident, rel, nonfin = run_op(call, rf, np.asarray(rf).shape)
            except Exception as exc:
                clean, ident, rel, nonfin = False, False, float("inf"), -1
                detail = f"EXC {type(exc).__name__}"
            else:
                detail = f"det={ident} rel={rel:.2e} nonfin={nonfin}"
            expected_broken = (name, op) in KNOWN_BROKEN
            status = "BROKEN" if not clean else "clean"
            flag = ""
            if clean and expected_broken:
                unexpected_fixed.append((name, op))
                flag = "  <-- UNEXPECTEDLY FIXED"
            elif not clean and not expected_broken:
                unexpected_broken.append((name, op))
                flag = "  <-- REGRESSION"
            print(f"{name:<8} {op:<11} {status:<7} {detail}{flag}")

    print()
    if unexpected_broken:
        print(f"REGRESSIONS (newly broken): {unexpected_broken}")
        return 1
    if unexpected_fixed:
        print(f"FIXED vs pin (update KNOWN_BROKEN + containment): {unexpected_fixed}")
        return 2
    print("PIN HOLDS: world matches the recorded gfx1103 baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
