#!/bin/bash
# run_pair.sh <profile> <old-tree> <new-tree> <outdir>
# F0-G fix round 1.  Dry-run of ONE profile twice:
#   old = <new-tree>/docker/flliper/<dir>/<profile>.env.alt  (the pure old-spelling chain: it sources its siblings' .alt) in the PRE-RENAME tree
#   new = <new-tree>/docker/flliper/<dir>/<profile>.env      (the converted profile) in the RENAMED tree
# Logs: <outdir>/old_<profile>.log, new_<profile>.log (last line EXIT=<rc>).  No GPU, no docker, no request to a serving port: the launcher is
# run with --dry-run and CUDA_VISIBLE_DEVICES="".  Compare with drycmp.py.
set -u
P=$1; OLDT=$2; NEWT=$3; OUT=$4
HERE=$(cd "$(dirname "$0")" && pwd)
if [ "$P" = nf-int4-h6-abl ]; then SUB=profiles; else SUB=profiles_release; fi
PD=$NEWT/docker/flliper/$SUB
[ -f "$PD/$P.env.alt" ] && [ -f "$PD/$P.env" ] || { echo "no $P in $PD" >&2; exit 2; }
export RELEASE_WORK=${RELEASE_WORK:-$OUT/work}
export RELEASE_KIT=$NEWT/tools/release
mkdir -p "$RELEASE_WORK" "$OUT"
nice -n 15 bash "$HERE/dry_force.sh" "$P" "$OLDT" "$OUT/old_$P.log" "$PD/$P.env.alt" >/dev/null 2>&1
nice -n 15 bash "$HERE/dry_force.sh" "$P" "$NEWT" "$OUT/new_$P.log" "$PD/$P.env" >/dev/null 2>&1
echo "done $P: old $(tail -n 1 "$OUT/old_$P.log")  new $(tail -n 1 "$OUT/new_$P.log")"
