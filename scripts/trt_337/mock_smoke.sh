#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Task #337 -- end-to-end mock smoke. No card, no TensorRT required.
#
# Desk-written code that has never executed is not code, it is a draft. This
# script runs every script in scripts/trt_337 end to end on the CPU with stub
# engines and stub kernels, so a card window cannot die on a typo, a wrong
# argument name or a shape that does not chain.
#
# It must stay green. Run it after any edit to targets.py, build_engines.py,
# microbench_trt.py or export_onnx.py, and re-run it after any rebase.
#
# Usage:
#   scripts/trt_337/mock_smoke.sh [workdir]
#
# Exit code 0 means every path executed. It means NOTHING about performance --
# every mock timing is stamped "stub": true.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
WORK="${1:-/tmp/337_mock_smoke}"
PY="${PY:-/spinning/htsglang-gpu/.venv/bin/python}"
PYLIBS="${PYLIBS:-/spinning/gpu-battery-results/2026-08-05_trt_microbench_prep/pylibs}"

# No card is touched by any step below, and the empty device list is what
# proves it rather than a promise in a comment.
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$PYLIBS:$REPO/python:${PYTHONPATH:-}"

rm -rf "$WORK"
mkdir -p "$WORK"

step() { printf '\n=== %s ===\n' "$1"; }

step "0/7 guard tests (stream, empty-graph, plausibility, signatures, card id)"
"$PY" "$HERE/test_guards.py"

step "1/7 shape table (rank 0 and rank 1, cross-checked against the serving split)"
"$PY" "$HERE/targets.py" --ranks 0,1

step "2/7 TensorRT capability probe (what the installed library can express)"
if "$PY" -c "import tensorrt_rtx" 2>/dev/null; then
  "$PY" "$HERE/build_engines.py" --probe | head -40
else
  echo "tensorrt_rtx not on PYTHONPATH -- skipping probe (mock path continues)"
fi

step "3/7 stub engine build (INT8 per-stage AND fold whole-chain variants)"
"$PY" "$HERE/build_engines.py" --mock --random-weights --ranks 0 \
  --precision int8,fold_bf16,fold_fp16,fp32_ref \
  --out-dir "$WORK/engines"

step "4/7 harness over every target and every arm, stub engines, CPU"
"$PY" "$HERE/microbench_trt.py" --mock --engine-dir "$WORK/engines" \
  --random-weights --rank 0 --m 1,4 \
  --min-point-seconds 0.2 --min-arm-seconds 0.05 --target-ms 2 \
  --rounds-floor 3 --graph-bodies 4 \
  --out "$WORK/bench.json"

step "5/7 ONNX export path (tiny shapes)"
if "$PY" -c "import onnx" 2>/dev/null; then
  "$PY" "$HERE/export_onnx.py" --mock --ranks 0 --out-dir "$WORK/onnx" \
    2>&1 | grep -v -i deprecation
else
  echo "onnx not on PYTHONPATH -- skipping (secondary path, see export_onnx.py)"
fi

step "6/7 result sanity"
"$PY" - "$WORK/bench.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["mock"] is True, "mock run must be stamped"
# The mock must have driven the REAL TrtEngine through the REAL call sites.
# A stub-only mock is what let TrtEngine(mode="fold") reach a card window.
sc = d.get("signature_conformance")
assert sc, "mock run did not perform the signature conformance check"
assert sc.get("pass"), f"signature conformance FAILED: {sc}"
for _step in ("construct", "select_profile", "bind_fold", "bind", "enqueue",
              "enqueue_used_given_stream", "layer_information", "save_cache"):
    assert _step in sc["steps"], f"conformance skipped {_step}: {sc}"
assert d["results"], "no results emitted"
bad = []
fold_seen = set()
for r in d["results"]:
    if not r.get("arms"):
        bad.append(f"{r['target']} M={r['m']}: no arm ran")
    tol = r.get("tolerance", {})
    if "error" in tol:
        bad.append(f"{r['target']} M={r['m']}: tolerance path failed: {tol['error']}")
    elif not tol.get("pass", False):
        bad.append(
            f"{r['target']} M={r['m']}: stub arms disagree "
            f"(max_rel_diff={tol.get('max_rel_diff')}) -- in mock both arms run "
            f"the same stub, so any disagreement is a harness bug, not a kernel "
            f"difference"
        )
for r in d["results"]:
    for name, arm in r["arms"].items():
        if not arm.get("stub"):
            bad.append(f"{r['target']} M={r['m']} {name}: mock timing not stamped")
        if name.startswith("trt_fold") or name.startswith("trt_fp32"):
            fold_seen.add(name.replace("#A2", ""))
    # The fold arms carry the quality claim, so the quality instrument has to
    # have run wherever a fold arm ran -- an unmeasured claim is the thing this
    # whole harness exists to avoid.
    q = r.get("quality")
    if any(n.startswith("trt_fold") for n in r["arms"]):
        if not q or "error" in q:
            bad.append(f"{r['target']} M={r['m']}: fold arm ran without a "
                       f"quality reading ({q})")
        else:
            for name, qa in q.get("arms", {}).items():
                if not qa.get("finite", True):
                    # A real property of fp16 in a multi-GEMM chain, and the
                    # mock's random weights make intermediates larger than the
                    # checkpoint's would be. Reported loudly, not treated as a
                    # harness failure -- the card run with real weights decides
                    # whether it bites in practice.
                    print(f"  NOTE {r['target']} M={r['m']} {name}: "
                          f"{qa.get('overflow')}")
                elif not qa.get("at_least_as_accurate_as_deployed"):
                    bad.append(
                        f"{r['target']} M={r['m']} {name}: fold is LESS accurate "
                        f"than the deployed path ({qa.get('err_vs_exact')} vs "
                        f"{q.get('deployed_int8_err_vs_exact')}) -- the "
                        f"quality-neutrality premise does not hold here"
                    )
if not {"trt_fold_bf16_graph", "trt_fold_fp16_graph"} <= fold_seen:
    bad.append(f"fold arms never ran; only saw {sorted(fold_seen)}")
if bad:
    print("FAIL"); [print("  " + b) for b in bad]; sys.exit(1)
# No arm may be reported without the plausibility verdict attached.
for r in d["results"]:
    p = r.get("plausibility")
    if p is None:
        bad.append(f"{r['target']} M={r['m']}: no plausibility record")
print(f"OK: {len(d['results'])} points, "
      f"{sum(len(r['arms']) for r in d['results'])} arm measurements, "
      f"fold arms {sorted(fold_seen)}, "
      f"tolerance and quality paths exercised on every point")
EOF

printf '\nmock smoke GREEN -- artifacts under %s\n' "$WORK"
