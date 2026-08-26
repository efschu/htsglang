#!/usr/bin/env bash
# #868 -- SIBLING SWEEP for the loadscope class of defect.
#
# Runs ONE module alone under `-n 2 --dist loadscope`. Because only that module
# is present, any difference from its solo-serial result cannot come from
# another module: it is xdist splitting the FILE by CLASS. Quantifies how many
# modules in the gate path carry an intra-file, cross-class state dependency.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${GATE_PY:-/spinning/htsglang-gpu/.venv/bin/python3}"
OUTDIR="${SCOPE_OUTDIR:-/tmp/868_scope}"
mod="$1"; base="$(basename "$mod" .py)"
mkdir -p "$OUTDIR"; cd "$ROOT"
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/python" \
  timeout "${SCOPE_TIMEOUT:-900}" "$PY" -m pytest "$mod" \
    -q -p no:randomly -p no:cacheprovider --color=no -rfE -n 2 --dist loadscope \
    > "$OUTDIR/$base.log" 2>&1
printf '#SOLO_RC %d\n#SOLO_MODULE %s\n' "$?" "$mod" >> "$OUTDIR/$base.log"
echo "$base done"
