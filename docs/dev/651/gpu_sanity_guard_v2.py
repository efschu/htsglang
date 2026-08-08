#!/usr/bin/env python
"""Pre-serving GPU sanity guard, v2 (#651). Correctness-gated, not
determinism-gated.

WHY v1 HAD TO BE REPLACED
-------------------------
v1 declared "GPU IS IN THE POISONED STATE (suspend/resume defect family) --
REBOOT THE MACHINE" whenever 8 Q5_K dequantize launches were not byte-identical.
Measured on 2026-08-08 on this laptop (Radeon 780M / gfx1103), that verdict is
wrong in both halves:

  * There is no poisoned STATE. A 5-cycle battery scored 8/15 with failures
    spread evenly over baseline, post-load and post-idle phases -- uncorrelated
    with every transition ever suspected. That is why system suspend, runtime
    PM and GFXOFF were each falsified as triggers: there was never a trigger.
  * A REBOOT is not a remedy for something that is not a state. v1 refuses
    roughly a third to a half of all boots of a perfectly healthy machine, and
    the recommended fix cannot change that rate.

What actually exists is a rare per-launch fault in the dequantize kernel's
output: about 1.6% of launches, corrupting 32/64/128 contiguous elements (whole
K-quant sub-blocks) by 1.2e-2..2.7e-2 while every other element stays
bit-identical at the 3.86e-05 quantization error. It was shown NOT to be
unwritten memory (a planted sentinel never reappears), NOT the
gfx1100-code-object-under-override mismatch (a native gfx1103 build shows the
same 3/25 rate), NOT the device-to-host copy path (0 faults in 40 trials with
each device tensor copied 3x), and NOT a cold-start effect (a discarded warmup
launch leaves the rate unchanged: 5/40 with, 3/25 without).

WHAT THIS GUARD GATES ON INSTEAD
--------------------------------
The question serving actually needs answered is "are the dequantized weights
CORRECT", not "are two launches bit-identical". So:

  * CORRECTNESS is the gate. Every K-quant type is dequantized and compared to
    the numpy oracle. The consensus across launches must be within TOL_CORRECT,
    and no single launch may deviate by more than TOL_GROSS.
  * The background transient is REPORTED, not fatal: the fraction of elements
    that disagree across launches and the largest disagreement are printed
    every time, so a change in that rate is visible rather than silently
    tolerated.
  * A GROSS failure -- a large fraction of elements wrong, or a wrong value far
    outside the transient's magnitude -- still fails hard. That is the class the
    guard exists for: the int64-topk_ids bug that started #651 made routed
    experts 0.54-correlated with the oracle, which is orders of magnitude
    outside anything here.

CAN-FAIL PROOF
--------------
Run with --self-test: the fixture bytes are deliberately corrupted and the
guard must FAIL. A gate that has never been shown to fail is not a gate.

Exit codes: 0 = fit to serve; 1 = FAIL (do not serve); 2 = cannot test.
"""

import argparse
import importlib
import json
import os

import numpy as np
import torch
import gguf
from gguf.constants import GGMLQuantizationType as QT

FIXTURE_DIR = os.environ.get("GGUF_FIXTURE_DIR", "/root/lh/ggufbuild")
PROBE_MODULE = os.environ.get("GGUF_PROBE_MODULE", "gguf_rocm_probe")
NRUNS = 5

# Consensus must be this close to the numpy oracle. The observed quantization
# error of a healthy dequantize is 3.86e-05, so this is ~25x headroom.
TOL_CORRECT = 1e-3
# No individual launch may deviate further than this. The measured transient
# tops out at 2.7e-2; real corruption is orders of magnitude larger.
TOL_GROSS = 1e-1
# Nor may the transient touch more than this fraction of the tensor. Measured
# worst case is 128/262144 = 0.049%.
MAX_TRANSIENT_FRAC = 0.01

QUANT_TYPES = {"q4_K": QT.Q4_K, "q5_K": QT.Q5_K, "q6_K": QT.Q6_K}


def consensus(stack: np.ndarray) -> np.ndarray:
    """Element-wise median across launches.

    The median is the right consensus for this fault: a corrupted launch moves
    a few elements far away, and with NRUNS>=3 the median is unaffected by a
    minority of outliers, so the consensus is the clean result without needing
    to know WHICH launch was faulty.
    """
    return np.median(stack, axis=0)


def check_type(K, name, qtype, raw, rows, cols, corrupt: bool):
    # The oracle is ALWAYS computed from the clean bytes. The injection below
    # corrupts only what the GPU is given, which is the situation the guard
    # exists to catch: the kernel returning something other than what the
    # checkpoint says. Corrupting both sides instead would leave them in
    # agreement and prove nothing -- and would only ever trip the guard through
    # NaN bookkeeping, which is not the failure mode of interest.
    ref = gguf.quants.dequantize(raw, qtype).astype(np.float32)
    if corrupt:
        raw = raw.copy()
        raw[:, :4] ^= 0xFF
    W = torch.from_numpy(raw).cuda()
    runs = []
    for _ in range(NRUNS):
        o = K.ggml_dequantize(W, int(qtype), rows, cols, torch.float16, None)
        torch.cuda.synchronize()
        runs.append(np.nan_to_num(o.cpu().numpy().astype(np.float32),
                                  posinf=0, neginf=0))
        del o
    stack = np.stack(runs)

    cons = consensus(stack)
    err_consensus = float(np.abs(cons - ref).max())

    # Transient characterisation: elements not identical across all launches.
    varies = (stack != stack[0]).any(axis=0)
    frac = float(varies.mean())
    spread = float((stack.max(axis=0) - stack.min(axis=0)).max())
    worst_launch = max(float(np.abs(r - ref).max()) for r in runs)

    ok_correct = err_consensus < TOL_CORRECT
    ok_gross = worst_launch < TOL_GROSS
    ok_frac = frac < MAX_TRANSIENT_FRAC
    ok = ok_correct and ok_gross and ok_frac

    print(
        f"GUARD: {'PASS' if ok else 'FAIL'}  {name}: consensus max|d| "
        f"{err_consensus:.3e} (tol {TOL_CORRECT:.0e}); worst launch "
        f"{worst_launch:.3e} (tol {TOL_GROSS:.0e}); transient "
        f"{frac*100:.4f}% of elements, spread {spread:.3e}"
    )
    if not ok_correct:
        print(f"GUARD:   -> consensus is WRONG against the oracle: {name} "
              "dequantize is broken, not merely jittering.")
    if not ok_gross:
        print(f"GUARD:   -> a launch deviated by {worst_launch:.3e}, far "
              "outside the known transient magnitude (2.7e-2).")
    if not ok_frac:
        print(f"GUARD:   -> {frac*100:.4f}% of elements unstable, far above "
              "the known transient footprint (<0.05%).")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="corrupt what the GPU is given; the guard MUST fail",
    )
    ap.add_argument(
        "--require",
        default="q4_K,q5_K",
        help=(
            "quant types whose verdict gates serving. Types outside this list "
            "are still measured and reported but do not block a boot. The "
            "default omits q6_K deliberately: q6_K carries a far heavier "
            "transient here (0.24%% of elements at ~4.9e-01, versus 0.05%% at "
            "2.7e-02 for q5_K), and the served checkpoint is the requantized "
            "noQ6K build precisely because of it. Gating on a type the "
            "checkpoint does not contain would block boots for nothing."
        ),
    )
    args = ap.parse_args()
    required = {t.strip() for t in args.require.split(",") if t.strip()}

    try:
        K = importlib.import_module(PROBE_MODULE)
    except ImportError as exc:
        print(f"GUARD: cannot import kernel extension {PROBE_MODULE!r}: {exc}")
        return 2
    try:
        slices = json.load(open(os.path.join(FIXTURE_DIR, "slices.json")))
    except OSError as exc:
        print(f"GUARD: fixtures unavailable: {exc}")
        return 2

    os.chdir(FIXTURE_DIR)
    by_name = {s["name"]: s for s in slices}

    results = []
    for name, qtype in QUANT_TYPES.items():
        s = by_name.get(name)
        if s is None:
            print(f"GUARD: no fixture for {name}, skipping")
            continue
        raw = np.fromfile(s["path"], dtype=np.uint8).reshape(
            s["rows"], s["row_bytes"]
        )
        verdict = check_type(
            K, name, qtype, raw, s["rows"], s["cols"], args.self_test
        )
        gating = name in required
        if not gating:
            print(f"GUARD:   ({name} is reported only, not gating)")
        results.append((name, verdict, gating))

    if not results:
        print("GUARD: nothing testable")
        return 2

    ok = all(v for _, v, gating in results if gating)
    if args.self_test:
        # Inverted expectation: corrupted input must NOT pass.
        if ok:
            print("GUARD SELF-TEST FAILED: corrupted fixtures still passed. "
                  "This guard cannot detect the thing it exists to detect.")
            return 1
        print("GUARD SELF-TEST PASSED: corrupted fixtures were rejected.")
        return 0

    if not ok:
        print("GUARD: dequantize is not fit to serve. This is NOT a "
              "'reboot the machine' condition -- v1's advice was wrong; "
              "investigate the kernel or the checkpoint.")
        return 1
    print("GUARD: dequantize correct on all types; fit to serve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
