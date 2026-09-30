#!/bin/bash
# DUAL-TP3PP3: does a COLD triton module load in one group wedge the other group's (or its own) barlink collectives when
# both groups are clients of ONE MPS daemon? Metal kw6pft 04:16:56: all three D ranks opened "triton cold module load:
# _fused_sigmoid_mul_kernel" and never closed it; P and D froze at the same second (PP1 forward 154 s, then RingCommitTimeout).
# Arms (TO bounds a hang, rc=124 = wedged):
#   M0 MPS, no cold load (control)    M1 MPS, D cold-loads every 2 s    M2 MPS, P cold-loads every 2 s
#   M3 MPS, D cold-loads, P GEMM only (does P need device-spin collectives for the wedge?)
#   N1 no MPS, D cold-loads every 2 s
# Usage: run_coldjit.sh <outdir>
set -u
PY=${PY:-/spinning/htsglang-gpu/.venv/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$(cd "$HERE/../.." && pwd)/python
unset CUDA_VISIBLE_DEVICES
B="$PY $HERE/bar1_two_groups.py"
OUT=$1; mkdir -p "$OUT"
DUR=${DUR:-12}; TO=${TO:-150}
MPSROOT=/tmp/dual-mps-cj-$$
stop_mps() { [ -d "$MPSROOT/pipe" ] && CUDA_MPS_PIPE_DIRECTORY=$MPSROOT/pipe timeout 10 bash -c 'echo quit | nvidia-cuda-mps-control' >/dev/null 2>&1; rm -rf "$MPSROOT"; }
reap() { local pids; pids=$(pgrep -f "$HERE/bar1_two_groups.py" || true)
  [ -n "$pids" ] && { echo "reaping leftover pids: $pids"; kill $pids 2>/dev/null; sleep 2; kill -9 $pids 2>/dev/null; } || true; }
trap 'reap; stop_mps' EXIT INT TERM
pair() {  # <tag> "<d-extra>" "<p-extra>"
  local tag=$1 dx=$2 px=$3; local st=$(( $(date +%s) + 40 ))
  timeout $TO $B --name D --port $((29700 + RANDOM % 50)) --size 40960 --start-at $st --dur $DUR --out "$OUT/$tag.D.json" $dx > "$OUT/$tag.D.log" 2>&1 &
  local a=$!
  env ${PENV:-} timeout $TO $B --name P --port $((29760 + RANDOM % 50)) --size 1048576 --gemm-m 2048 --start-at $st --dur $DUR --out "$OUT/$tag.P.json" $px > "$OUT/$tag.P.log" 2>&1 &
  local b=$!
  wait $a; echo "$tag D rc=$?"; wait $b; echo "$tag P rc=$?"; reap
}
JIT="--jit-at 3 --jit-every 2"
mkdir -p $MPSROOT/pipe $MPSROOT/log
CUDA_MPS_PIPE_DIRECTORY=$MPSROOT/pipe CUDA_MPS_LOG_DIRECTORY=$MPSROOT/log nvidia-cuda-mps-control -d; sleep 1
export CUDA_MPS_PIPE_DIRECTORY=$MPSROOT/pipe
echo "== M0 MPS control"; PENV="CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50" pair M0 "" ""
echo "== M1 MPS, D cold loads"; PENV="CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50" pair M1 "$JIT" ""
echo "== M2 MPS, P cold loads"; PENV="CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50" pair M2 "" "$JIT"
echo "== M3 MPS, D cold loads, P GEMM only (no P collectives)"; PENV="CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50" pair M3 "$JIT" "--gemm-only"
stop_mps; unset CUDA_MPS_PIPE_DIRECTORY
echo "== N1 no MPS, D cold loads"; pair N1 "$JIT" ""
$PY - "$OUT" <<'PYEOF'
import glob, json, os, sys
for f in sorted(glob.glob(sys.argv[1] + "/*.json")):
    d = json.load(open(f))
    print(f"{os.path.basename(f):8s} ok={d['ok']} mps={d['mps']} " + " | ".join(
        f"r{x.get('rank')} p50 {x.get('p50_ms', float('nan')):.3f} max {x.get('max_ms', float('nan')):.1f} n {x.get('n', 0)} jit {x.get('jit_ms')} {x.get('err', '')[:60]}" for x in d["ranks"]))
PYEOF
