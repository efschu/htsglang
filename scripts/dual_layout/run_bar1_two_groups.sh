#!/bin/bash
# DUAL-TP3PP3 risk 1b driver: barlink BAR1, two groups (D, P) on the same three cards, without / with a private MPS
# daemon. Usage: run_bar1_two_groups.sh <outdir>. Devices are torch indices (0 = 5090 on this rig, FASTEST_FIRST).
set -u
PY=${PY:-/spinning/htsglang-gpu/.venv/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$(cd "$HERE/../.." && pwd)/python
unset CUDA_VISIBLE_DEVICES
B="$PY $HERE/bar1_two_groups.py"
OUT=$1; mkdir -p "$OUT"
DUR=${DUR:-8}; TO=${TO:-300}
MPSROOT=/tmp/dual-mps-b1-$$
stop_mps() { [ -d "$MPSROOT/pipe" ] && CUDA_MPS_PIPE_DIRECTORY=$MPSROOT/pipe timeout 10 bash -c 'echo quit | nvidia-cuda-mps-control' >/dev/null 2>&1; rm -rf "$MPSROOT"; }
reap() {  # leftovers of a timed-out arm: exact PIDs of THIS script only (no broad pkill)
  local pids; pids=$(pgrep -f "$HERE/bar1_two_groups.py" || true)
  [ -n "$pids" ] && { echo "reaping leftover pids: $pids"; kill $pids 2>/dev/null; sleep 2; kill -9 $pids 2>/dev/null; } || true
}
trap 'reap; stop_mps' EXIT INT TERM
pair() {  # <tag> <p-extra...>: D + P concurrently
  local tag=$1; shift; local st=$(( $(date +%s) + 40 ))
  timeout $TO $B --name D --port $((29700 + RANDOM % 50)) --size 40960 --start-at $st --dur $DUR --out "$OUT/$tag.D.json" > "$OUT/$tag.D.log" 2>&1 &
  local a=$!
  env ${PENV:-} timeout $TO $B --name P --port $((29760 + RANDOM % 50)) --start-at $st --dur $DUR --out "$OUT/$tag.P.json" "$@" > "$OUT/$tag.P.log" 2>&1 &
  local b=$!
  wait $a; echo "$tag D rc=$?"; wait $b; echo "$tag P rc=$?"; reap
}
echo "== A: D solo, no MPS"; timeout $TO $B --name D --port 29690 --size 40960 --dur $DUR --out "$OUT/A.D.json" > "$OUT/A.D.log" 2>&1; echo "A rc=$?"; reap
echo "== B: D + P (AR 1 MiB + GEMM 2048), no MPS"; pair B --size 1048576 --gemm-m 2048
mkdir -p $MPSROOT/pipe $MPSROOT/log
CUDA_MPS_PIPE_DIRECTORY=$MPSROOT/pipe CUDA_MPS_LOG_DIRECTORY=$MPSROOT/log nvidia-cuda-mps-control -d; sleep 1
export CUDA_MPS_PIPE_DIRECTORY=$MPSROOT/pipe
echo "== C: D solo, MPS"; timeout $TO $B --name D --port 29691 --size 40960 --dur $DUR --out "$OUT/C.D.json" > "$OUT/C.D.log" 2>&1; echo "C rc=$?"; reap
echo "== D: D + P (AR 1 MiB + GEMM), MPS, P 50% SM"; PENV="CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50" pair D --size 1048576 --gemm-m 2048
echo "== E: D + P (GEMM only), MPS, P 50% SM"; PENV="CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50" pair E --size 1048576 --gemm-m 4096 --gemm-only
stop_mps
unset CUDA_MPS_PIPE_DIRECTORY
echo "== F: D + P (GEMM only), no MPS"; pair F --size 1048576 --gemm-m 4096 --gemm-only
$PY - "$OUT" <<'PYEOF'
import glob, json, os, sys
for f in sorted(glob.glob(sys.argv[1] + "/*.json")):
    d = json.load(open(f))
    rs = d["ranks"]
    print(f"{os.path.basename(f):10s} ok={d['ok']} mps={d['mps']} sm={d['sm_pct']} " + " | ".join(
        f"r{x.get('rank')} p50 {x.get('p50_ms', float('nan')):.3f} p99 {x.get('p99_ms', float('nan')):.3f} n {x.get('n', 0)} bad {x.get('bad')} {x.get('err', '')[:80]}" for x in rs))
PYEOF
