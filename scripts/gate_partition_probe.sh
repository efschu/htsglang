#!/usr/bin/env bash
# #868 -- solo-probe for ONE test module.
#
# The admission criterion for the parallel tier-2 track is a PROOF, not an
# observation:
#
#   a module is admissible iff, ALONE in a fresh process, it produces exactly
#   the failure set it produces inside the full serial run.
#
# One measurement covers both directions of the #860 dual:
#   * a module that DEPENDS on being poisoned (`global_server_args`) fails
#     solo and passes serially   -> sets differ -> refused;
#   * a module that is POISONED by a neighbour passes solo and fails serially
#     -> sets differ -> refused.
# The second direction is the one a "run it parallel and see what breaks"
# approach cannot see, because separating victim from poisoner turns a real
# serial failure into a parallel PASS -- a false green that no number of
# parallel repeats exposes.
#
# CUDA_VISIBLE_DEVICES is forced empty here for the same reason as in
# gate_tier2.sh: the probe runs at the desk while a window may hold the cards.
# Modules that genuinely need a device are excluded BY NAME (see
# gate_partition.txt), never accommodated by loosening this line.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${GATE_PY:-/spinning/htsglang-gpu/.venv/bin/python3}"
OUTDIR="${SOLO_OUTDIR:-/tmp/868_solo}"
TMO="${SOLO_TIMEOUT:-900}"

mod="$1"                       # path relative to $ROOT
base="$(basename "$mod" .py)"
log="$OUTDIR/$base.log"

mkdir -p "$OUTDIR"
cd "$ROOT"
start=$(date +%s.%N)
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/python" \
  timeout "$TMO" "$PY" -m pytest "$mod" \
    -q -p no:randomly -p no:cacheprovider --color=no -rfE \
    > "$log" 2>&1
rc=$?
end=$(date +%s.%N)
printf '#SOLO_RC %d\n#SOLO_WALL %.2f\n#SOLO_MODULE %s\n' \
  "$rc" "$(echo "$end - $start" | bc)" "$mod" >> "$log"
echo "$base rc=$rc"
