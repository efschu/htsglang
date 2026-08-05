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

step "1/6 shape table (rank 0 and rank 1, cross-checked against the serving split)"
"$PY" "$HERE/targets.py" --ranks 0,1

step "2/6 TensorRT capability probe (what the installed library can express)"
if "$PY" -c "import tensorrt_rtx" 2>/dev/null; then
  "$PY" "$HERE/build_engines.py" --probe | head -40
else
  echo "tensorrt_rtx not on PYTHONPATH -- skipping probe (mock path continues)"
fi

step "3/6 stub engine build"
"$PY" "$HERE/build_engines.py" --mock --random-weights --ranks 0 \
  --out-dir "$WORK/engines"

step "4/6 harness over every target, stub engines, CPU"
"$PY" "$HERE/microbench_trt.py" --mock --engine-dir "$WORK/engines" \
  --random-weights --rank 0 --m 1,4 \
  --min-point-seconds 0.2 --min-arm-seconds 0.05 --target-ms 2 \
  --rounds-floor 3 --graph-bodies 4 \
  --out "$WORK/bench.json"

step "5/6 ONNX export path (tiny shapes)"
if "$PY" -c "import onnx" 2>/dev/null; then
  "$PY" "$HERE/export_onnx.py" --mock --ranks 0 --out-dir "$WORK/onnx" \
    2>&1 | grep -v -i deprecation
else
  echo "onnx not on PYTHONPATH -- skipping (secondary path, see export_onnx.py)"
fi

step "6/6 result sanity"
"$PY" - "$WORK/bench.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["mock"] is True, "mock run must be stamped"
assert d["results"], "no results emitted"
bad = []
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
if bad:
    print("FAIL"); [print("  " + b) for b in bad]; sys.exit(1)
print(f"OK: {len(d['results'])} points, "
      f"{sum(len(r['arms']) for r in d['results'])} arm measurements, "
      f"tolerance path exercised on every point")
EOF

printf '\nmock smoke GREEN -- artifacts under %s\n' "$WORK"
