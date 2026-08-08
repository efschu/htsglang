#!/usr/bin/env python
"""#651: is the Q5_K non-determinism in the KERNEL or in the DEVICE->HOST COPY?

Established so far (2026-08-08, this laptop, Radeon 780M / gfx1103):

  * Eight consecutive Q5_K dequantize launches occasionally disagree, in about
    3 process-runs out of 25.
  * A disagreement is tiny and structured: ~64 contiguous elements of one row,
    magnitude ~1.2e-2 while every other element stays bit-identical at the
    3.86e-05 quantization error. 64 fp16 elements is 128 bytes.
  * It is NOT unwritten memory (a sentinel planted in the freed block never
    reappears).
  * It is NOT the gfx1100-vs-gfx1103 code-object mismatch: a native gfx1103
    build and the overridden gfx1100 build show the same 3/25 rate.
  * It is NOT correlated with load, idle, or any state transition, and the
    server serves coherent deterministic text throughout.

That leaves two candidates with very different consequences:

  K "kernel/output memory": the launch itself, or the device buffer it wrote,
    briefly holds wrong values. Serving would be exposed, because serving reads
    those dequantized weights on-device.

  C "copy path": the device buffer is correct and the corruption happens while
    the result is being DMA'd to host memory. 128 bytes is a plausible burst
    granularity. Serving would NOT be exposed at all -- dequantized weights are
    produced and consumed on-device and never travel to the host. The guard's
    canary would then be measuring a host-transfer defect and calling it a
    poisoned GPU.

The experiment separates them by construction: run the kernel N times keeping
every output tensor ALIVE on the device, then copy each tensor to the host
TWICE. Two host copies of the SAME device tensor can only differ if the copy
path is at fault. Two different tensors differing while each is internally
self-consistent implicates the kernel. Both effects can appear, and the script
reports them independently rather than assuming one.
"""

import importlib
import json
import os
import sys

import numpy as np
import torch

FIXTURE_DIR = os.environ.get("GGUF_FIXTURE_DIR", "/root/lh/ggufbuild")
PROBE_MODULE = os.environ.get("GGUF_PROBE_MODULE", "gguf_rocm_probe")
NRUNS = int(os.environ.get("NRUNS", "8"))
NCOPIES = int(os.environ.get("NCOPIES", "3"))


def diff_summary(a, b, cols):
    d = np.nan_to_num(a) != np.nan_to_num(b)
    n = int(d.sum())
    if n == 0:
        return None
    idx = np.flatnonzero(d.reshape(-1))
    return (
        f"{n} elems, rows {idx.min()//cols}..{idx.max()//cols}, "
        f"flat {idx.min()}..{idx.max()}, "
        f"max|d| {np.abs(a[d] - b[d]).max():.3e}"
    )


def main() -> int:
    try:
        K = importlib.import_module(PROBE_MODULE)
    except ImportError as exc:
        print(f"cannot import {PROBE_MODULE!r}: {exc}")
        return 2

    slices = json.load(open(os.path.join(FIXTURE_DIR, "slices.json")))
    os.chdir(FIXTURE_DIR)
    s = {x["name"]: x for x in slices}["q5_K"]
    rows, cols = s["rows"], s["cols"]
    raw = np.fromfile(s["path"], dtype=np.uint8).reshape(rows, s["row_bytes"])
    W = torch.from_numpy(raw).cuda()

    print(f"extension={PROBE_MODULE} runs={NRUNS} copies_per_run={NCOPIES}")

    # Keep every output tensor alive so each can be copied more than once.
    devs = []
    for _ in range(NRUNS):
        o = K.ggml_dequantize(W, s["type"], rows, cols, torch.float16, None)
        devs.append(o)
    torch.cuda.synchronize()

    copies = []
    for o in devs:
        per = [o.cpu().numpy().astype(np.float32) for _ in range(NCOPIES)]
        copies.append(per)

    # C: do repeated copies of the SAME device tensor disagree?
    copy_faults = 0
    print("\n--- copy-path check (same device tensor, copied "
          f"{NCOPIES}x) ---")
    for i, per in enumerate(copies):
        for j in range(1, NCOPIES):
            d = diff_summary(per[0], per[j], cols)
            if d:
                copy_faults += 1
                print(f"  run{i} copy0 vs copy{j}: {d}")
    if not copy_faults:
        print("  all repeated copies byte-identical")

    # K: do different launches disagree, each being self-consistent?
    kernel_faults = 0
    print("\n--- kernel check (launch i vs launch 0, first copy) ---")
    for i in range(1, NRUNS):
        d = diff_summary(copies[0][0], copies[i][0], cols)
        if d:
            kernel_faults += 1
            print(f"  run0 vs run{i}: {d}")
    if not kernel_faults:
        print("  all launches byte-identical")

    print()
    if copy_faults and not kernel_faults:
        print("VERDICT: COPY PATH -- the device results agree; only host "
              "transfers differ. Serving does not read weights over this path.")
    elif kernel_faults and not copy_faults:
        print("VERDICT: KERNEL/OUTPUT MEMORY -- each tensor copies "
              "reproducibly, but launches disagree. Serving is exposed.")
    elif kernel_faults and copy_faults:
        print("VERDICT: BOTH effects present in this sample.")
    else:
        print("VERDICT: nothing observed in this sample (the event is rare; "
              "run this repeatedly).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
