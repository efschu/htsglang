#!/bin/bash
# DUAL-TP3PP3: BAR1 aperture check with the PRODUCTION window shapes of both groups on the same three cards:
# D = world 16 + tp 32 + dcp 40 MiB, P = world <PW> + pp <PP> MiB; snapshots nvidia-smi BAR1 with all windows alive,
# then runs the D all-reduce loop with P's load, both as MPS clients. Usage: run_bar1_budget.sh <outdir> [PW] [PP]
set -u
PY=${PY:-/spinning/htsglang-gpu/.venv/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$(cd "$HERE/../.." && pwd)/python
unset CUDA_VISIBLE_DEVICES
OUT=$1; PW=${2:-16}; PPW=${3:-64}; mkdir -p "$OUT"
MPSROOT=/tmp/dual-mps-bb-$$
mkdir -p $MPSROOT/pipe $MPSROOT/log
CUDA_MPS_PIPE_DIRECTORY=$MPSROOT/pipe CUDA_MPS_LOG_DIRECTORY=$MPSROOT/log nvidia-cuda-mps-control -d; sleep 1
trap 'CUDA_MPS_PIPE_DIRECTORY=$MPSROOT/pipe timeout 10 bash -c "echo quit | nvidia-cuda-mps-control" >/dev/null 2>&1; rm -rf $MPSROOT' EXIT
export CUDA_MPS_PIPE_DIRECTORY=$MPSROOT/pipe
st=$(( $(date +%s) + 50 ))
timeout 240 $PY $HERE/bar1_two_groups.py --name D --port 29811 --size 40960 --windows-mib 16,32,40 --bar1-snapshot --start-at $st --dur 6 --out "$OUT/D.json" > "$OUT/D.log" 2>&1 &
a=$!
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 timeout 240 $PY $HERE/bar1_two_groups.py --name P --port 29851 --size 1048576 --gemm-m 2048 --windows-mib $PW,$PPW --bar1-snapshot --start-at $st --dur 6 --out "$OUT/P.json" > "$OUT/P.log" 2>&1 &
b=$!
wait $a; echo "D rc=$?"; wait $b; echo "P rc=$?"
$PY - "$OUT" <<'PYEOF'
import json, sys
for g in ("D", "P"):
    d = json.load(open(f"{sys.argv[1]}/{g}.json"))
    print(g, "ok", d["ok"], [(r.get("rank"), r.get("p50_ms"), r.get("bad"), (r.get("err") or "")[:160]) for r in d["ranks"]])
    for r in d["ranks"][:1]:
        print("  BAR1 snapshot:", r.get("bar1_q"))
PYEOF
