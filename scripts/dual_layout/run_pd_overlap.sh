#!/bin/bash
# DUAL-TP3PP3 risk 1 driver: solo / inproc / 2proc / 2proc-mps per card.
# Usage: run_pd_overlap.sh <outdir> <nvml_idx> [<nvml_idx> ...]
# Cards are addressed by UUID (resolved via nvidia-smi = NVML), never by torch order.
# The MPS daemon uses a PRIVATE pipe directory, so no other process on the rig
# can attach to it; it is stopped by a trap on every exit path.
set -u
PY=${PY:-/spinning/shvllm/.venv/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
B="$PY $HERE/pd_overlap_bench.py"
OUT=$1; shift
mkdir -p "$OUT"
DUR=${DUR:-8}
ROWS=${ROWS:-4}
PM=${PM:-4096}
MPSROOT=/tmp/dual-mps-$$
stop_mps() {
  if [ -d "$MPSROOT/pipe" ]; then
    CUDA_MPS_PIPE_DIRECTORY=$MPSROOT/pipe CUDA_MPS_LOG_DIRECTORY=$MPSROOT/log \
      timeout 10 bash -c 'echo quit | nvidia-cuda-mps-control' >/dev/null 2>&1
    sleep 1
  fi
}
trap 'stop_mps' EXIT INT TERM

for IDX in "$@"; do
  UUID=$(nvidia-smi -i "$IDX" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
  NAME=$(nvidia-smi -i "$IDX" --query-gpu=name --format=csv,noheader | tr -d ' ')
  T="$OUT/card${IDX}_${NAME}"
  echo "== card $IDX $NAME $UUID"
  export CUDA_VISIBLE_DEVICES=$UUID
  common="--dur $DUR --rows $ROWS --prefill-m $PM"
  s() { echo $(( $(date +%s) + ${1:-20} )); }
  # solo floors
  $B --role decode  $common --out "$T.solo_decode.json"  >/dev/null
  $B --role prefill $common --out "$T.solo_prefill.json" >/dev/null
  # one process, two streams (decode high prio)
  $B --role both $common --out "$T.inproc.json" >/dev/null
  # one process, two streams, equal prio
  $B --role both $common --decode-prio 0 --out "$T.inproc_eqprio.json" >/dev/null
  # two processes, no MPS
  st=$(s 15)
  $B --role decode  $common --start-at $st --out "$T.2proc_decode.json" >/dev/null &
  p1=$!
  $B --role prefill $common --start-at $st --out "$T.2proc_prefill.json" >/dev/null &
  p2=$!
  wait $p1 $p2
  if [ "${PROBES:-1}" = 1 ]; then
  $PY $HERE/ipc_mps_probe.py > "$T.ipc_nomps.jsonl" 2>"$T.ipc_nomps.err"
  $PY $HERE/vmm_mps_probe.py > "$T.vmm_nomps.jsonl" 2>"$T.vmm_nomps.err"
  fi
  # two processes under a private MPS daemon
  mkdir -p $MPSROOT/pipe $MPSROOT/log
  CUDA_MPS_PIPE_DIRECTORY=$MPSROOT/pipe CUDA_MPS_LOG_DIRECTORY=$MPSROOT/log \
    nvidia-cuda-mps-control -d
  sleep 1
  if [ "${PROBES:-1}" = 1 ]; then
  CUDA_MPS_PIPE_DIRECTORY=$MPSROOT/pipe $PY $HERE/ipc_mps_probe.py > "$T.ipc_mps.jsonl" 2>"$T.ipc_mps.err"
  CUDA_MPS_PIPE_DIRECTORY=$MPSROOT/pipe $PY $HERE/vmm_mps_probe.py > "$T.vmm_mps.jsonl" 2>"$T.vmm_mps.err"
  fi
  for arm in mps mps_p50; do
    st=$(s 15)
    pct=100; [ $arm = mps_p50 ] && pct=50
    CUDA_MPS_PIPE_DIRECTORY=$MPSROOT/pipe $B --role decode $common --start-at $st \
      --out "$T.2proc_${arm}_decode.json" >/dev/null &
    p1=$!
    CUDA_MPS_PIPE_DIRECTORY=$MPSROOT/pipe CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$pct \
      $B --role prefill $common --start-at $st --out "$T.2proc_${arm}_prefill.json" >/dev/null &
    p2=$!
    wait $p1 $p2
  done
  stop_mps
  rm -rf $MPSROOT
done
# NCCL between two MPS clients on two cards (only with >= 2 cards)
if [ $# -ge 2 ] && [ "${PROBES:-1}" = 1 ]; then
  U0=$(nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
  U1=$(nvidia-smi -i "$2" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
  export CUDA_VISIBLE_DEVICES=$U0,$U1
  $PY $HERE/nccl_mps_probe.py > "$OUT/nccl_nomps.jsonl" 2>"$OUT/nccl_nomps.err"
  mkdir -p $MPSROOT/pipe $MPSROOT/log
  CUDA_MPS_PIPE_DIRECTORY=$MPSROOT/pipe CUDA_MPS_LOG_DIRECTORY=$MPSROOT/log nvidia-cuda-mps-control -d
  sleep 1
  CUDA_MPS_PIPE_DIRECTORY=$MPSROOT/pipe $PY $HERE/nccl_mps_probe.py > "$OUT/nccl_mps.jsonl" 2>"$OUT/nccl_mps.err"
  stop_mps
  rm -rf $MPSROOT
fi
$PY - "$OUT" <<'EOF'
import json, glob, os, sys
d = sys.argv[1]
def L(p):
    try: return json.load(open(p))
    except Exception as e: return None
for base in sorted({f.split('.')[0] for f in glob.glob(d + '/card*.json')}):
    sd, sp = L(base + '.solo_decode.json'), L(base + '.solo_prefill.json')
    if not sd or not sp: continue
    D0, P0 = sd['decode_steps_per_s'], sp['prefill_tflops']
    print(f"\n{os.path.basename(base)}  solo decode {D0:.1f} step/s p50 {sd['decode_p50_ms']:.2f} ms | solo prefill {P0:.1f} TFLOP/s")
    arms = [('inproc', L(base+'.inproc.json'), None), ('inproc_eqprio', L(base+'.inproc_eqprio.json'), None)]
    for a in ('2proc', '2proc_mps', '2proc_mps_p50'):
        arms.append((a, L(f"{base}.{a}_decode.json"), L(f"{base}.{a}_prefill.json")))
    for name, x, y in arms:
        if x is None: print(f"  {name:14s} missing"); continue
        y = y or x
        if 'decode_steps_per_s' not in x or 'prefill_tflops' not in y: print(f"  {name:14s} incomplete"); continue
        sd_ = x['decode_steps_per_s'] / D0; sp_ = y['prefill_tflops'] / P0
        print(f"  {name:14s} share_dec {sd_:.3f} share_pre {sp_:.3f}  E {sd_+sp_:.3f}  dec p50 {x['decode_p50_ms']:.2f} p99 {x['decode_p99_ms']:.2f} max {x['decode_max_ms']:.1f} ms")
for f in sorted(glob.glob(d + '/*.jsonl')):
    print(os.path.basename(f), open(f).read().strip()[:300])
EOF
