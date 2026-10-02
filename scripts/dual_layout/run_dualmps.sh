#!/bin/bash
# DUAL-TP3PP3 MPS repro v3 (01.10., 27B Opus seat): does "no cooperative grid under MPS" (grid_threshold_default,
# barlink_bar1.py) end the S1 wedge of repro v2 scjhru, and what does a big collective cost on the 1blk path?
# ROOT (scjhru S1 + metal ndktv4): from 4 MiB a spin all-reduce is a cooperative full-card grid; under MPS a resident
# spin block of the OTHER group keeps it from starting -> cross-card wait cycle, broken only by capCycles.
# Every arm gets a FRESH MPS daemon (a wedged client poisons the server: scjhru S2 died "device busy or unavailable").
# The control arm C1 (old behaviour, explicit 4 MiB threshold) runs LAST with a short cap (~3 s at 2 GHz), so a wedge
# shows as aborted rounds, not as a 150 s hang.
# Arms (DUR s each, TO bounds a hang, rc=124 = wedged; grid=True/False per group in the json):
#   F1 MPS, D 8 MiB AR, P 1 MiB AR + GEMM 2048, P 50 % SM   (= scjhru S1 load, fix on)
#   F2 MPS, D 8 MiB AR, P 8 MiB AR + GEMM, no SM cap         (both groups above the old grid threshold)
#   F3 MPS, D 40 MiB AR (extend 4096 x 5120 x 2 B), P 10 MiB AR (a 1024 chunk frame) + GEMM
#   N3 no MPS, F3 load                                       (latency baseline without MPS)
#   C1 MPS, F1 load, SGLANG_BARLINK_BAR1_GRID_THRESHOLD=4194304 (old) + cap 6e9   (control: expected aborts/wedge)
# Usage (CT999, cards 0,1,2 free, no boot): run_dualmps.sh <outdir>   ~12 min
set -u
PY=${PY:-/spinning/htsglang-gpu/.venv/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$(cd "$HERE/../.." && pwd)/python
unset CUDA_VISIBLE_DEVICES SGLANG_BARLINK_BAR1_GRID_THRESHOLD
B="$PY $HERE/bar1_two_groups.py"
OUT=$1; mkdir -p "$OUT"
DUR=${DUR:-30}; TO=${TO:-150}
MPSROOT=/tmp/dual-mps-v3-$$
start_mps() { mkdir -p $MPSROOT/pipe $MPSROOT/log
  CUDA_MPS_PIPE_DIRECTORY=$MPSROOT/pipe CUDA_MPS_LOG_DIRECTORY=$MPSROOT/log nvidia-cuda-mps-control -d; sleep 1
  export CUDA_MPS_PIPE_DIRECTORY=$MPSROOT/pipe; }
stop_mps() { [ -d "$MPSROOT/pipe" ] && CUDA_MPS_PIPE_DIRECTORY=$MPSROOT/pipe timeout 10 bash -c 'echo quit | nvidia-cuda-mps-control' >/dev/null 2>&1
  sleep 2; rm -rf "$MPSROOT"; unset CUDA_MPS_PIPE_DIRECTORY; }
reap() { local pids; pids=$(pgrep -f "^$PY $HERE/bar1_two_groups.py" || true)
  [ -n "$pids" ] && { echo "reaping leftover pids: $pids"; kill $pids 2>/dev/null; sleep 2; kill -9 $pids 2>/dev/null; } || true; }
trap 'reap; stop_mps' EXIT INT TERM
pair() {  # <tag> "<d-args>" "<p-args>"
  local tag=$1 dx=$2 px=$3; local st=$(( $(date +%s) + 40 ))
  env ${DENV:-} timeout $TO $B --name D --port $((29700 + RANDOM % 50)) --start-at $st --dur $DUR --out "$OUT/$tag.D.json" $dx > "$OUT/$tag.D.log" 2>&1 &
  local a=$!
  env ${PENV:-} timeout $TO $B --name P --port $((29760 + RANDOM % 50)) --start-at $st --dur $DUR --out "$OUT/$tag.P.json" $px > "$OUT/$tag.P.log" 2>&1 &
  local b=$!
  wait $a; echo "$tag D rc=$?"; wait $b; echo "$tag P rc=$?"; reap
}
P50="CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50"
echo "== F1 MPS fix, D 8 MiB, P 1 MiB + GEMM, P 50 %"; start_mps; PENV=$P50 DENV= pair F1 "--size 8388608" "--size 1048576 --gemm-m 2048"; stop_mps
echo "== F2 MPS fix, D 8 MiB, P 8 MiB + GEMM";       start_mps; PENV= DENV= pair F2 "--size 8388608" "--size 8388608 --gemm-m 2048"; stop_mps
echo "== F3 MPS fix, D 40 MiB, P 10 MiB + GEMM";     start_mps; PENV= DENV= pair F3 "--size 41943040" "--size 10485760 --gemm-m 2048"; stop_mps
echo "== N3 no MPS, D 40 MiB, P 10 MiB + GEMM";                  PENV= DENV= pair N3 "--size 41943040" "--size 10485760 --gemm-m 2048"
OLD="SGLANG_BARLINK_BAR1_GRID_THRESHOLD=4194304 SGLANG_BARLINK_BAR1_CAP_CYCLES=6000000000"
echo "== C1 MPS OLD grid (control), D 8 MiB, P 1 MiB + GEMM, P 50 %, cap 6e9"; start_mps
PENV="$P50 $OLD" DENV="$OLD" pair C1 "--size 8388608" "--size 1048576 --gemm-m 2048"; stop_mps
$PY - "$OUT" <<'PYEOF'
import glob, json, os, sys
for f in sorted(glob.glob(sys.argv[1] + "/*.json")):
    d = json.load(open(f))
    print(f"{os.path.basename(f):8s} ok={d['ok']} mps={d['mps']} " + " | ".join(
        f"r{x.get('rank')} grid={x.get('grid')} p50 {x.get('p50_ms', float('nan')):.3f} p99 {x.get('p99_ms', float('nan')):.2f} max {x.get('max_ms', float('nan')):.1f} n {x.get('n', 0)} bad {x.get('bad')} {x.get('err', '')[:50]}" for x in d["ranks"]))
PYEOF
