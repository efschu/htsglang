#!/usr/bin/env python
"""#651: what exactly is non-deterministic about the Q5_K dequantize canary?

WHY THIS EXISTS. The pre-serving guard (gpu_sanity_guard.py) calls the GPU
"poisoned" whenever 8 Q5_K dequantize runs are not byte-identical. A 5-cycle
battery on 2026-08-08 scored 8/15, with failures spread evenly over baseline,
post-load and post-idle phases -- i.e. UNCORRELATED with any state transition
-- while the Q4_K correctness check passed 15/15 with a bit-identical maximum
error, and the server was concurrently answering coherence probes correctly and
deterministically. A canary that cries wolf half the time on an otherwise
healthy machine is a suspect, not an oracle.

Two hypotheses, and they have opposite consequences:

  H1 "poisoned GPU": the dequantize result is genuinely unstable. The
     differences should then be spread over the whole output and the result
     should be WRONG against the numpy oracle at least some of the time.

  H2 "uninitialized output region": the kernel does not write every element of
     its freshly allocated output (tail/padding rows or a block-count edge).
     Unwritten elements then hold whatever the caching allocator left in that
     block, so run 1 differs from runs 2..8 while runs 2..8 agree with each
     other (each reuses the previous run's freed block). The WRITTEN region is
     perfectly correct, and the machine is fine.

H2 is a kernel bug in our own code and is harmless to serving unless the tail
is read. H1 would invalidate every measurement on this laptop. They are
distinguished by WHERE the differences sit and whether the oracle agrees, which
is what this script reports:

  * per-run correctness against the numpy oracle, restricted to the region all
    runs agree on, and over the full tensor;
  * the exact index set that differs, as row/column extents, so a tail or a
    padding edge is visible as such;
  * the pairwise agreement matrix over the 8 runs (H2 predicts run 1 alone
    against a unanimous 2..8);
  * a control in which the output block is poisoned deliberately: allocate and
    fill a same-shaped tensor with a sentinel, free it, then dequantize. Under
    H2 the sentinel reappears in the unwritten region -- that is a positive
    identification of unwritten memory, not an inference from a difference.
"""

import json
import os
import sys

import numpy as np
import torch
import gguf
from gguf.constants import GGMLQuantizationType as QT

FIXTURE_DIR = os.environ.get("GGUF_FIXTURE_DIR", "/root/lh/ggufbuild")
# Which built extension to exercise. The point of making this switchable is the
# gfx1100-vs-gfx1103 A/B: the default probe extension is compiled for gfx1100
# and only runs here because HSA_OVERRIDE_GFX_VERSION=11.0.0 tells the runtime
# to accept it on gfx1103 silicon, while gguf_rocm_native is compiled with
# --offload-arch=gfx1103 and needs no override. If the non-determinism belongs
# to the code-object mismatch rather than to the machine, it lives in exactly
# one of these two arms.
PROBE_MODULE = os.environ.get("GGUF_PROBE_MODULE", "gguf_rocm_probe")
NRUNS = 8
# Discarded launches before the measured ones. Several observed disagreements
# have run 0 standing alone against a unanimous 1..7, which is a cold-first-
# launch signature rather than a random per-launch event; NWARMUP makes that
# hypothesis testable instead of assumed.
NWARMUP = int(os.environ.get("NWARMUP", "0"))
SENTINEL = 12345.0


def describe_diff(a: np.ndarray, b: np.ndarray, cols: int):
    """Locate differing elements as row/column extents."""
    d = np.nan_to_num(a) != np.nan_to_num(b)
    n = int(d.sum())
    if n == 0:
        return "identical"
    idx = np.flatnonzero(d.reshape(-1))
    rows = idx // cols
    colsi = idx % cols
    frac = n / d.size * 100
    return (
        f"{n} elems ({frac:.4f}%) rows {rows.min()}..{rows.max()} "
        f"cols {colsi.min()}..{colsi.max()} "
        f"first_flat={idx[0]} last_flat={idx[-1]}"
    )


def main() -> int:
    import importlib

    try:
        K = importlib.import_module(PROBE_MODULE)
    except ImportError as exc:
        print(f"cannot import kernel extension {PROBE_MODULE!r}: {exc}")
        return 2
    print(f"extension: {PROBE_MODULE} ({getattr(K, '__file__', '?')})")
    print(f"HSA_OVERRIDE_GFX_VERSION={os.environ.get('HSA_OVERRIDE_GFX_VERSION', '<unset>')}")

    slices = json.load(open(os.path.join(FIXTURE_DIR, "slices.json")))
    os.chdir(FIXTURE_DIR)
    s = {x["name"]: x for x in slices}["q5_K"]
    rows, cols = s["rows"], s["cols"]
    raw = np.fromfile(s["path"], dtype=np.uint8).reshape(rows, s["row_bytes"])
    ref = gguf.quants.dequantize(raw, QT.Q5_K).astype(np.float32)
    W = torch.from_numpy(raw).cuda()

    print(f"Q5_K slice: rows={rows} cols={cols} row_bytes={s['row_bytes']} "
          f"elements={rows*cols}")
    print(f"oracle shape {ref.shape}")

    # NO TORCH GPU KERNELS ANYWHERE BELOW. ROCm torch on this laptop carries no
    # gfx1103 code objects, so a single `.float()` on the device would abort
    # with `invalid device function` whenever the gfx1100 override is off --
    # which is exactly the arm this script has to be able to run. Device work
    # is therefore restricted to the extension's own kernel plus plain memory
    # copies (`.cuda()` / `.cpu()` are memcpies, not kernel launches), and the
    # fp16->fp32 widening happens on the host in numpy.
    for _ in range(NWARMUP):
        w = K.ggml_dequantize(W, s["type"], rows, cols, torch.float16, None)
        torch.cuda.synchronize()
        del w

    outs = []
    for i in range(NRUNS):
        o = K.ggml_dequantize(W, s["type"], rows, cols, torch.float16, None)
        torch.cuda.synchronize()
        outs.append(o.cpu().numpy().astype(np.float32))
        del o

    # 1. pairwise agreement against run 0
    print("\n--- agreement with run 0 ---")
    for i in range(1, NRUNS):
        print(f"  run{i}: {describe_diff(outs[0], outs[i], cols)}")

    # 2. unanimity among runs 1..7 (H2 predicts they agree with each other)
    print("\n--- agreement among runs 1..7 ---")
    tail_unanimous = all(
        np.array_equal(np.nan_to_num(outs[1]), np.nan_to_num(o)) for o in outs[2:]
    )
    print(f"  runs 1..7 byte-identical to each other: {tail_unanimous}")

    # 3. correctness against the oracle, full tensor and agreed region
    print("\n--- correctness vs numpy oracle ---")
    stable = np.ones(outs[0].shape, dtype=bool)
    for o in outs[1:]:
        stable &= np.nan_to_num(outs[0]) == np.nan_to_num(o)
    print(f"  elements stable across all {NRUNS} runs: "
          f"{int(stable.sum())}/{stable.size} ({stable.sum()/stable.size*100:.4f}%)")
    for i, o in enumerate(outs):
        oc = np.nan_to_num(o.astype(np.float32), posinf=0, neginf=0)
        full = float(np.abs(oc - ref).max())
        onstable = float(np.abs(oc[stable] - ref[stable]).max()) if stable.any() else float("nan")
        print(f"  run{i}: max|d| full {full:.3e}   on stable region {onstable:.3e}")

    # 4. sentinel control: is the differing region simply never written?
    print("\n--- sentinel control (unwritten-memory identification) ---")
    torch.cuda.empty_cache()
    # Filled on the host and copied up, again to avoid a torch device kernel.
    poison = torch.from_numpy(
        np.full((rows, cols), SENTINEL, dtype=np.float16)
    ).cuda()
    torch.cuda.synchronize()
    del poison  # returns the block to the caching allocator
    o = K.ggml_dequantize(W, s["type"], rows, cols, torch.float16, None)
    torch.cuda.synchronize()
    got = o.cpu().numpy().astype(np.float32)
    hits = int((got == SENTINEL).sum())
    print(f"  sentinel value survives in {hits} elements "
          f"({hits/got.size*100:.4f}% of the output)")
    if hits:
        idx = np.flatnonzero((got == SENTINEL).reshape(-1))
        print(f"  sentinel rows {idx//cols} (min {idx.min()//cols}, "
              f"max {idx.max()//cols}), flat {idx.min()}..{idx.max()}")

    print()
    if hits:
        print("VERDICT: H2 CONFIRMED -- the kernel leaves output elements "
              "unwritten; the guard canary was reading allocator garbage, not "
              "a poisoned GPU.")
    elif tail_unanimous and not np.array_equal(
        np.nan_to_num(outs[0]), np.nan_to_num(outs[1])
    ):
        print("VERDICT: H2 LIKELY -- run 0 stands alone against a unanimous "
              "1..7, the signature of allocator reuse, but no sentinel "
              "survived; inspect the differing region above.")
    elif stable.all():
        print("VERDICT: no instability observed in this sample.")
    else:
        print("VERDICT: H1 NOT EXCLUDED -- differences are not explained by "
              "unwritten memory; treat the GPU as suspect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
