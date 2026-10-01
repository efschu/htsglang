#!/bin/bash
# DUAL-TP3PP3 K_SPLIT repro (01.10., 27B Opus seat): multi-block all_reduce WITHOUT a cooperative launch under MPS.
# Questions: (1) does K_SPLIT win back the 1blk solo loss (tatcwj: D 40 MiB solo 9.9 ms grid/no MPS vs 13.3 ms 1blk/MPS)?
# (2) does the wedge stay away with BOTH groups on K_SPLIT? (3) bit-exact (verify every op)?
# Fresh MPS daemon per arm; control C1 (old cooperative grid under MPS, short cap) LAST.
# Arms (DUR s each; json per group: variant 0 1blk / 1 grid / 2 split, algo ring|mesh, bad = byte mismatches):
#   N3solo  no MPS, D 40 MiB alone (grid)                 F3solo  MPS, D 40 MiB alone (1blk)
#   KSsolo  MPS + SPLIT, D 40 MiB alone (split)
#   F3      MPS, D 40 MiB + P 10 MiB + GEMM (1blk)        KS3     same, both groups SPLIT
#   KS1     MPS + SPLIT, D 8 MiB + P 1 MiB + GEMM, P 50 % (scjhru S1 load)
#   KM      MPS + SPLIT + mesh forced (RING_THRESHOLD 1<<40), D 8 MiB + P 1 MiB + GEMM
#   C1      MPS, old grid (GRID_THRESHOLD 4 MiB) + cap 6e9 (control: expected aborts)
# Usage (CT999, cards 0,1,2 free, no boot): run_ksplit.sh <outdir>   ~14 min
set -u
PY=${PY:-/spinning/htsglang-gpu/.venv/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$(cd "$HERE/../.." && pwd)/python
unset CUDA_VISIBLE_DEVICES SGLANG_BARLINK_BAR1_GRID_THRESHOLD SGLANG_BARLINK_BAR1_SPLIT
B="$PY $HERE/bar1_two_groups.py"
OUT=$1; mkdir -p "$OUT"
DUR=${DUR:-30}; TO=${TO:-150}
MPSROOT=/tmp/dual-ksplit-$$
start_mps() { mkdir -p $MPSROOT/pipe $MPSROOT/log
  CUDA_MPS_PIPE_DIRECTORY=$MPSROOT/pipe CUDA_MPS_LOG_DIRECTORY=$MPSROOT/log nvidia-cuda-mps-control -d; sleep 1
  export CUDA_MPS_PIPE_DIRECTORY=$MPSROOT/pipe; }
stop_mps() { [ -d "$MPSROOT/pipe" ] && CUDA_MPS_PIPE_DIRECTORY=$MPSROOT/pipe timeout 10 bash -c 'echo quit | nvidia-cuda-mps-control' >/dev/null 2>&1
  sleep 2; rm -rf "$MPSROOT"; unset CUDA_MPS_PIPE_DIRECTORY; }
reap() { local pids; pids=$(pgrep -f "^$PY $HERE/bar1_two_groups.py" || true)
  [ -n "$pids" ] && { echo "reaping leftover pids: $pids"; kill $pids 2>/dev/null; sleep 2; kill -9 $pids 2>/dev/null; } || true; }
trap 'reap; stop_mps' EXIT INT TERM
solo() {  # <tag> "<d-args>"
  local st=$(( $(date +%s) + 20 ))
  env ${DENV:-} timeout $TO $B --name D --port $((29700 + RANDOM % 50)) --start-at $st --dur $DUR --verify-every 1 --out "$OUT/$1.D.json" $2 > "$OUT/$1.D.log" 2>&1
  echo "$1 D rc=$?"; reap; }
pair() {  # <tag> "<d-args>" "<p-args>"
  local tag=$1 dx=$2 px=$3; local st=$(( $(date +%s) + 40 ))
  env ${DENV:-} timeout $TO $B --name D --port $((29700 + RANDOM % 50)) --start-at $st --dur $DUR --verify-every 1 --out "$OUT/$tag.D.json" $dx > "$OUT/$tag.D.log" 2>&1 &
  local a=$!
  env ${PENV:-} timeout $TO $B --name P --port $((29760 + RANDOM % 50)) --start-at $st --dur $DUR --verify-every 1 --out "$OUT/$tag.P.json" $px > "$OUT/$tag.P.log" 2>&1 &
  local b=$!
  wait $a; echo "$tag D rc=$?"; wait $b; echo "$tag P rc=$?"; reap; }
SPL="SGLANG_BARLINK_BAR1_SPLIT=1"
P50="CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50"
echo "== N3solo no MPS, D 40 MiB alone";               DENV= solo N3solo "--size 41943040"
echo "== F3solo MPS 1blk, D 40 MiB alone";  start_mps; DENV= solo F3solo "--size 41943040"; stop_mps
echo "== KSsolo MPS split, D 40 MiB alone"; start_mps; DENV=$SPL solo KSsolo "--size 41943040"; stop_mps
echo "== F3 MPS 1blk, D 40 MiB + P 10 MiB"; start_mps; DENV= PENV= pair F3 "--size 41943040" "--size 10485760 --gemm-m 2048"; stop_mps
echo "== KS3 MPS split both, D 40 + P 10";  start_mps; DENV=$SPL PENV=$SPL pair KS3 "--size 41943040" "--size 10485760 --gemm-m 2048"; stop_mps
echo "== KS1 MPS split both, S1 load";      start_mps; DENV=$SPL PENV="$SPL $P50" pair KS1 "--size 8388608" "--size 1048576 --gemm-m 2048"; stop_mps
MESH="SGLANG_BARLINK_BAR1_RING_THRESHOLD=1099511627776"
echo "== KM MPS split mesh, D 8 + P 1";     start_mps; DENV="$SPL $MESH" PENV="$SPL $MESH" pair KM "--size 8388608" "--size 1048576 --gemm-m 2048"; stop_mps
OLD="SGLANG_BARLINK_BAR1_GRID_THRESHOLD=4194304 SGLANG_BARLINK_BAR1_CAP_CYCLES=6000000000"
echo "== C1 MPS OLD grid (control)";        start_mps; DENV="$OLD" PENV="$P50 $OLD" pair C1 "--size 8388608" "--size 1048576 --gemm-m 2048"; stop_mps
$PY - "$OUT" <<'PYEOF'
import glob, json, os, sys
for f in sorted(glob.glob(sys.argv[1] + "/*.json")):
    d = json.load(open(f))
    print(f"{os.path.basename(f):11s} ok={d['ok']} mps={d['mps']} " + " | ".join(
        f"r{x.get('rank')} v={x.get('variant')} {x.get('algo')} p50 {x.get('p50_ms', float('nan')):.3f} p99 {x.get('p99_ms', float('nan')):.2f} max {x.get('max_ms', float('nan')):.1f} n {x.get('n', 0)} bad {x.get('bad')} {x.get('err', '')[:40]}" for x in d["ranks"]))
PYEOF
