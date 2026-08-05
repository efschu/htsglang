#!/bin/bash
# Window 3: the boot-to-boot floor, which windows 1 and 2 both lacked.
#
# Window 2 compared an eager BOOT against a graph BOOT and found 4/8 text
# divergences. That number only means something if two EAGER boots agree with
# each other. NOTE_452 §2 experiment 2 flags exactly this floor as missing.
# If E1 vs E2 already diverges, the graph-vs-eager result is boot noise and
# must be withdrawn.
#
# Arms: E1 eager, E2 eager (identical flags), G graphs.
# Each arm also runs the prefill probe at three prompt sizes, so the perf
# question gets its own boot-to-boot noise floor for free.
set -u

WT=/spinning/wt-prefill-graphs
VENV=/spinning/htsglang-gpu/.venv
OUT=${OUT:-/spinning/gpu-battery-results/2026-08-05_prefill_graphs_w3}
PORT=30042
RESERVE="5500,3800,3800"

mkdir -p "$OUT"
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WT/python"
export SGLANG_UNEVEN_DCP=1
export SGLANG_UNEVEN_DCP_WEIGHTED=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16

boot() {
  local arm="$1"; shift
  local log="$OUT/boot_${arm}.log"
  echo "=== booting arm $arm"
  setsid "$VENV/bin/python" -m sglang.launch_server \
    --model-path /spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8 \
    --served-model-name default \
    --tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio auto-performance \
    --rank-perf-tune phase-decode \
    --rank-auto-reserve-mib "$RESERVE" \
    --kv-cache-dtype fp8_e4m3 --context-length 262144 \
    --max-running-requests 4 \
    --speculative-algorithm NEXTN --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --kv-pressure-ladder auto --max-mamba-cache-size 96 \
    --enable-fast-lane --retraction-policy priority \
    `# pin the seed so boot-to-boot RNG cannot be blamed for a divergence` \
    --random-seed 12345 \
    --disable-radix-cache --trust-remote-code \
    --host 127.0.0.1 --port $PORT \
    "$@" > "$log" 2>&1 &
  echo $! > "$OUT/${arm}.pgid"
  local waited=0
  until curl -s -o /dev/null -m 3 -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null | grep -q 200; do
    sleep 5; waited=$((waited+5))
    [ $waited -gt 600 ] && { echo "ARM $arm BOOT TIMEOUT"; tail -40 "$log"; return 1; }
    kill -0 "$(cat "$OUT/${arm}.pgid")" 2>/dev/null || { echo "ARM $arm DIED"; tail -50 "$log"; return 1; }
  done
  echo "arm $arm healthy after ${waited}s"
}

stop_arm() {
  local pg; pg=$(cat "$OUT/${1}.pgid" 2>/dev/null) || return 0
  kill -TERM -- -"$pg" 2>/dev/null
  local w=0; while kill -0 "$pg" 2>/dev/null && [ $w -lt 90 ]; do sleep 3; w=$((w+3)); done
  kill -KILL -- -"$pg" 2>/dev/null; sleep 8
}

run_arm() {
  local arm="$1"; shift
  boot "$arm" "$@" || return 1
  "$VENV/bin/python" "$WT/tests/prefill_graphs/content_gate.py" record \
      --port $PORT --out "$OUT/${arm}_gate.json" || return 1
  # SIZES is per-stage: the NCCL stage sweeps the length axis, the barlink
  # stage only needs the two points the 2x2 is stated over.
  for sz in ${SIZES:-256 900 1900}; do
    "$VENV/bin/python" "$WT/tests/prefill_graphs/prefill_perf.py" \
        --port $PORT --tokens $sz --n 12 --seed 4242 \
        --out "$OUT/${arm}_perf_${sz}.json" || return 1
  done
  if [ "${CONC:-1}" = "1" ]; then
    # Agent-like arrivals: short prompts, 4 in flight, so the scheduler
    # actually forms bs>1 prefill batches. This is the regime where a
    # captured prefill could plausibly pay -- launch-train bound, not GEMM
    # bound (the 68-75% collective share of the prefill window, #252) -- and
    # it is the regime the barlink hypothesis is really about.
    "$VENV/bin/python" "$WT/tests/prefill_graphs/prefill_perf.py" \
        --port $PORT --tokens 256 --n 24 --seed 777 --concurrency 4 \
        --out "$OUT/${arm}_perf_256c4.json" || return 1
  fi
  stop_arm "$arm"
}

STAGE="${STAGE:-nccl}"

if [ "$STAGE" = "nccl" ] || [ "$STAGE" = "all" ]; then
  unset SGLANG_BARLINK
  run_arm E1 || { stop_arm E1; exit 1; }
  run_arm E2 || { stop_arm E2; exit 1; }
  run_arm G --cuda-graph-backend-prefill breakable || { stop_arm G; exit 1; }

  # Determinism question: can this recipe ever be byte-strict under graphs, or
  # only distribution-level? --enable-deterministic-inference is NOT in the
  # breakable rule list (server_args.py:8487) -- only tc_piecewise rejects it --
  # so BCG + deterministic is a legal combination. Content gate only.
  SIZES="" CONC=0 run_arm ED --enable-deterministic-inference \
      || { stop_arm ED; exit 1; }
  SIZES="" CONC=0 run_arm GD --enable-deterministic-inference \
      --cuda-graph-backend-prefill breakable || { stop_arm GD; exit 1; }
fi

if [ "$STAGE" = "barlink" ] || [ "$STAGE" = "all" ]; then
  # PRECONDITION, enforced not assumed: barlink arms only run on a tree that
  # carries the #583 fix (b001d102fa, tripped spin kernel must not kill the
  # CUDA context), and only once barlink-583's repro window has confirmed the
  # fix holds under live load. Operator passes BARLINK_VERDICT=confirmed.
  if ! git -C "$WT" merge-base --is-ancestor b001d102fa HEAD 2>/dev/null; then
    echo "REFUSING barlink stage: #583 fix b001d102fa is not in this tree."
    exit 2
  fi
  if [ "${BARLINK_VERDICT:-}" != "confirmed" ]; then
    echo "REFUSING barlink stage: BARLINK_VERDICT is not 'confirmed'"
    echo "(barlink-583's repro window must clear the fix under live load first)."
    exit 2
  fi
  export SGLANG_BARLINK=1
  # Only the two points the 2x2 is stated over: the long-prompt point and the
  # bs>1 short-prompt concurrency mix.
  SIZES="1900" run_arm BE1 || { stop_arm BE1; exit 1; }
  SIZES="1900" run_arm BE2 || { stop_arm BE2; exit 1; }
  SIZES="1900" run_arm BG --cuda-graph-backend-prefill breakable \
      || { stop_arm BG; exit 1; }
  unset SGLANG_BARLINK
fi


# ---------------------------------------------------------------- reporting
# Every comparison is skipped rather than faked when its arms did not run, so
# a barlink-less window still produces a valid (smaller) report.
gate() {  # gate <label> <armA> <armB>
  [ -f "$OUT/${2}_gate.json" ] && [ -f "$OUT/${3}_gate.json" ] || return 0
  echo "--- $1"
  "$VENV/bin/python" "$WT/tests/prefill_graphs/content_gate.py" \
      compare "$OUT/${2}_gate.json" "$OUT/${3}_gate.json"
}

echo
echo "########## REPORT HEADER: measurement conditions ##########"
echo "Enforced GPU power caps at report time (measured, not assumed):"
nvidia-smi --query-gpu=index,name,power.limit,power.default_limit \
           --format=csv,noheader | sed 's/^/  /'
cat <<'HDR'
  Caps were reduced on this rig (3080 320->200 W, 5090 525->400 W).
  Every arm below is SAME-RIG and shares these caps, so the A/B deltas and the
  eager-vs-eager floors are unaffected. Any comparison against ARCHIVE numbers
  taken before the change -- the #320 Messbuendel tables in particular -- is
  CONFOUNDED by the power change and must not be read as a like-for-like
  regression or gain.
HDR

echo
echo "########## CONTENT FLOORS AND GATES ##########"
echo "A graphs-vs-eager divergence only counts if the eager-vs-eager floor above"
echo "it passes. NCCL and barlink each get their own floor."
gate "NCCL   floor  : eager vs eager (E1/E2)"        E1 E2
gate "NCCL   gate   : eager vs graphs (E1/G)"        E1 G
gate "barlink floor : eager vs eager (BE1/BE2)"      BE1 BE2
gate "barlink gate  : eager vs graphs (BE1/BG)"      BE1 BG
gate "transport     : NCCL eager vs barlink eager"   E1 BE1
gate "determinism   : eager-det vs graphs-det"       ED GD
gate "determinism   : eager vs eager-det"            E1 ED

echo
echo "########## THROUGHPUT: 2x2 transport x prefill-backend ##########"
"$VENV/bin/python" - "$OUT" <<'PY'
import json, os, statistics as st, sys
out = sys.argv[1]


def med(arm, sz):
    p = f"{out}/{arm}_perf_{sz}.json"
    if not os.path.exists(p):
        return None, None
    d = json.load(open(p))
    if d["cached_tokens_total"]:
        print(f"  !! {arm}/{sz}: cache hits present, number is contaminated")
    # Under concurrency the per-request rate understates throughput, so the
    # aggregate is the honest column there.
    if d.get("concurrency", 1) > 1:
        return d["aggregate_tok_s"], d["wall_seconds"]
    return st.median(d["prefill_tok_s_all"]), None


def block(label, e1, e2, g, sizes):
    print(f"\n=== {label} ===")
    for sz in sizes:
        a, _ = med(e1, sz)
        b, _ = med(e2, sz)
        c, _ = med(g, sz)
        if a is None or c is None:
            continue
        if b is None:
            print(f"{sz:>6}: eager {a:8.1f}  graphs {c:8.1f}  "
                  f"delta {(c/a-1)*100:+.1f}%  (NO FLOOR -- second eager boot missing)")
            continue
        noise = (b / a - 1) * 100
        emean = (a + b) / 2
        delta = (c / emean - 1) * 100
        verdict = "INSIDE NOISE" if abs(delta) <= abs(noise) else "outside noise"
        print(f"{sz:>6}: eager {a:8.1f}/{b:8.1f}  graphs {c:8.1f}  "
              f"floor {noise:+6.1f}%  delta {delta:+6.1f}%  -> {verdict}")


block("NCCL", "E1", "E2", "G", ("256", "900", "1900", "256c4"))
block("barlink", "BE1", "BE2", "BG", ("1900", "256c4"))

# The user's hypothesis, stated as one number per point: does the prefill-graph
# delta become positive under barlink where it was flat/negative under NCCL?
print("\n=== HYPOTHESIS: does the graph delta improve under barlink? ===")
print("(prefill-graph delta under NCCL vs under barlink, same point;")
print(" each compared against its OWN transport's eager floor)")
for sz in ("1900", "256c4"):
    n_e1, _ = med("E1", sz); n_e2, _ = med("E2", sz); n_g, _ = med("G", sz)
    b_e1, _ = med("BE1", sz); b_e2, _ = med("BE2", sz); b_g, _ = med("BG", sz)
    if None in (n_e1, n_g, b_e1, b_g):
        print(f"{sz:>6}: barlink arms absent -- hypothesis UNTESTED at this point")
        continue
    nd = (n_g / ((n_e1 + n_e2) / 2 if n_e2 else n_e1) - 1) * 100
    bd = (b_g / ((b_e1 + b_e2) / 2 if b_e2 else b_e1) - 1) * 100
    print(f"{sz:>6}: NCCL {nd:+.1f}%   barlink {bd:+.1f}%   swing {bd - nd:+.1f} pp")
print("\nA swing only supports the hypothesis if it exceeds BOTH transports'")
print("eager floors; otherwise it is noise wearing a mechanism story.")
PY
