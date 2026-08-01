#!/usr/bin/env bash
# #350/#375: tok/J one-flag A/B. Thin wrapper over run_ab.py.
#
# WHAT CHANGED AND WHY (#375). This script used to boot a server itself and
# then point the #146 harness at it with --base-url. Both were wrong, and the
# #350 window found out the expensive way:
#   * the harness's flag is --port, not --base-url; and
#   * the harness OWNS the boot (run_measurement builds the launch command
#     from a MeasurementConfig), so the two boots raced for the port.
# The logic now lives in run_ab.py, which builds one config per arm differing
# ONLY in --objective and lets the harness boot each. run_ab.py has a
# --dry-run that executes every line without a card, and this wrapper is
# smoke-run through it before any card window.
#
# Card discipline: claim through /spinning/gpu-arb/ BEFORE running, release
# after. This script does not arbitrate.
#
# Usage:
#   scripts/energy350/tokj_validation.sh                 # decode point
#   PROBE=1 scripts/energy350/tokj_validation.sh         # prefill-heavy probe
#   DRY_RUN=1 scripts/energy350/tokj_validation.sh       # no card, shape only

set -euo pipefail

WT="${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV="${VENV:-/spinning/htsglang-gpu/.venv}"
MODEL_ROOT="${MODEL_ROOT:-/spinning/llm_stuff/club-3090/models-cache}"
MODEL="${MODEL:-$MODEL_ROOT/Qwen3.6-27B-FP8}"
PORT="${PORT:-31350}"
OUT="${OUT:-/spinning/gpu-battery-results/$(date +%Y-%m-%d)_375_tokj}"

export PYTHONPATH="$WT/python"
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

ARGS=(--model "$MODEL" --port "$PORT" --out "$OUT")

if [[ "${PROBE:-0}" == "1" ]]; then
  # The #375 probe point: prefill-heavy at higher batch, where the 5090-vs-3080
  # power asymmetry has room to express itself. Verdict read on the PREFILL
  # axis, and the decode-knee guard made advisory so the objective has more
  # than one admissible candidate to choose between (the #350 window found the
  # guards left exactly one, which is why that A/B was structurally A-vs-A).
  ARGS+=(--bucket "${BUCKET:-8}" --axis prefill --workload prose
         --perf-tune phase-prefill --max-running-requests 16)
else
  ARGS+=(--bucket "${BUCKET:-1}" --axis decode --workload code)
fi

[[ "${DRY_RUN:-0}" == "1" ]] && ARGS+=(--dry-run)

mkdir -p "$OUT"
cd "$WT"
"$VENV/bin/python" scripts/energy350/run_ab.py "${ARGS[@]}" | tee "$OUT/VERDICT.txt"
