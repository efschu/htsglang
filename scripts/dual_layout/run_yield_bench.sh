#!/bin/bash
# DUAL-TP3PP3 lever (b) measurement (no MPS, time-slicing): D wake-up latency and P's GEMM share per D wait mode.
# Per card (NVML order: CUDA_DEVICE_ORDER=PCI_BUS_ID): P alone, then for each mode D alone and D + P concurrently.
# Read: P it/s with D 'spin' vs 'wait' (does a non-kernel wait hand the slice to P?) and D p50/p99 latency per mode.
# Usage (CT999, cards free, no boot, NO MPS daemon): run_yield_bench.sh <outdir> [cards "0 1"]   ~8 min for 2 cards
set -u
PY=${PY:-/spinning/htsglang-gpu/.venv/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$(cd "$HERE/../.." && pwd)/python CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_VISIBLE_DEVICES CUDA_MPS_PIPE_DIRECTORY
OUT=$1; CARDS=${2:-"0 1"}; mkdir -p "$OUT"
B="$PY $HERE/yield_wait_bench.py"; T=${T:-25}
for c in $CARDS; do
  echo "== card $c: P alone"; $B --role p --dev $c --seconds $T --out $OUT/c${c}_p_alone.json
  for m in spin wait host; do
    echo "== card $c: D $m alone"; timeout 120 $B --role d --dev $c --mode $m --iters 3000 --seconds $T --out $OUT/c${c}_d_${m}_alone.json
    echo "== card $c: D $m + P"
    $B --role p --dev $c --seconds $((T + 6)) --out $OUT/c${c}_p_with_${m}.json & pp=$!
    sleep 3; timeout 120 $B --role d --dev $c --mode $m --iters 100000 --seconds $T --out $OUT/c${c}_d_${m}_with_p.json; wait $pp
  done
done
$PY - "$OUT" <<'PYEOF'
import glob, json, os, sys
for f in sorted(glob.glob(sys.argv[1] + "/*.json")):
    d = json.load(open(f)); n = os.path.basename(f)[:-5]
    print(f"{n:22s} " + (f"P {d['it_s']} it/s {d['tflops']} TF" if d.get("role") == "p" else
          f"D n={d.get('n')} p50 {d.get('p50_us')} p90 {d.get('p90_us')} p99 {d.get('p99_us')} max {d.get('max_us')} us"))
PYEOF
