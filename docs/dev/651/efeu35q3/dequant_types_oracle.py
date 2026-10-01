#!/usr/bin/env python
"""efeu-TP14: per-type dequantize correctness vs the numpy oracle, per launch.

Counts, per launch, elements whose |gpu - oracle| exceeds TOL (fp16 rounding is
~6e-5 on these weights) -- signed zeros never count. Controls (Q3_K/Q4_K/Q8_0)
run next to the suspects (Q5_K/Q6_K) so a GLOBAL GPU fault (every type wrong)
is distinguishable from a TYPE-specific kernel defect.

    python dequant_types_oracle.py <gguf> [launches] [elems_per_type]
"""

import importlib
import json
import os
import sys

import numpy as np
import torch
import gguf
from gguf import GGUFReader

TOL = 1e-3


def main():
    src = sys.argv[1]
    launches = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    want = int(sys.argv[3]) if len(sys.argv) > 3 else 1 << 21
    K = importlib.import_module(os.environ.get("GGUF_MODULE", "sglang_gguf_rocm"))
    r = GGUFReader(src)
    picked = {}
    for t in r.tensors:
        n = t.tensor_type.name
        if n in ("F32", "F16", "BF16") or n in picked:
            continue
        d = t.data
        d = d.reshape(-1, d.shape[-1])
        bs, ts = gguf.GGML_QUANT_SIZES[t.tensor_type]
        cols = d.shape[1] // ts * bs
        rows = max(1, min(d.shape[0], want // cols))
        picked[n] = (t.name, t.tensor_type, np.array(d[:rows]), rows, cols)
    out = {"variant": {"HSA": os.environ.get("HSA_OVERRIDE_GFX_VERSION"), "module": K.__file__}, "types": {}}
    for n, (tn, qt, raw, rows, cols) in sorted(picked.items()):
        ref = gguf.quants.dequantize(raw, qt).astype(np.float32)
        W = torch.from_numpy(raw).cuda()
        bad_l, bad_e, worst, nonfin = 0, 0, 0.0, 0
        for _ in range(launches):
            o = K.ggml_dequantize(W, int(qt), rows, cols, torch.float16, None)
            torch.cuda.synchronize()
            g = o.float().cpu().numpy()
            del o
            nf = ~np.isfinite(g)
            diff = np.abs(np.where(nf, 0, g) - ref)
            e = int((diff > TOL).sum() + nf.sum())
            nonfin += int(nf.sum())
            if e:
                bad_l += 1
                bad_e += e
                worst = max(worst, float(diff.max()))
        out["types"][n] = {"tensor": tn, "elems": rows * cols, "launches": launches,
                           "bad_launches": bad_l, "bad_elems_total": bad_e,
                           "bad_frac": bad_e / (rows * cols * launches), "worst": worst, "nonfinite": nonfin}
        print(f"{n:6s} {tn:34s} elems {rows * cols:9d}  bad launches {bad_l:3d}/{launches}  "
              f"bad elems {bad_e:8d} ({bad_e / (rows * cols * launches):.2e})  worst {worst:.2e}  nonfin {nonfin}")
    if len(sys.argv) > 4:
        json.dump(out, open(sys.argv[4], "w"), indent=1)


if __name__ == "__main__":
    main()
