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
  if [ "${GATE:-1}" = "1" ]; then
    "$VENV/bin/python" "$WT/tests/prefill_graphs/content_gate.py" record \
        --port $PORT --out "$OUT/${arm}_gate.json" || return 1
  fi
  # Every perf point is a SUSTAINED ~5 s window of back-to-back draws with a
  # discarded warmup, scored on aggregate tok/s, with SM clock and P-state
  # sampled during the measured window. A lone short draw on idle cards would
  # measure the clock ramp, not the code (#483).
  # SIZES is per-stage: the NCCL stage sweeps the length axis, the barlink
  # stage only needs the two points the 2x2 is stated over.
  for sz in ${SIZES:-256 900 1900}; do
    "$VENV/bin/python" "$WT/tests/prefill_graphs/prefill_perf.py" \
        --port $PORT --tokens $sz --seconds 5 --warmup-seconds 2 --seed 4242 \
        --out "$OUT/${arm}_perf_${sz}.json" || return 1
  done
  if [ "${CONC:-1}" = "1" ]; then
    # Agent-like arrivals: short prompts, 4 in flight, so the scheduler
    # actually forms bs>1 prefill batches. This is the regime where a
    # captured prefill could plausibly pay -- launch-train bound, not GEMM
    # bound (the 68-75% collective share of the prefill window, #252) -- and
    # it is the regime the barlink hypothesis is really about.
    "$VENV/bin/python" "$WT/tests/prefill_graphs/prefill_perf.py" \
        --port $PORT --tokens 256 --seconds 5 --warmup-seconds 2 --seed 777 \
        --concurrency 4 \
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
  # PRECONDITION 1, enforced not assumed: the #583 fix (b001d102fa, a tripped
  # spin kernel must not kill the CUDA context) must be in the tree.
  # STATUS 2026-08-17: MET on the current integration lineage. Kept enforced
  # because this script also runs on older trees.
  if ! git -C "$WT" merge-base --is-ancestor b001d102fa HEAD 2>/dev/null; then
    echo "REFUSING barlink stage: #583 fix b001d102fa is not in this tree."
    exit 2
  fi

  # PRECONDITION 2: an operator attestation that barlink holds under live
  # load. STILL REQUIRED -- an attestation is not something a script may grant
  # itself -- but its EVIDENCE BASE has moved, and a stale reason is worse
  # than a strict one because it sends the operator to the wrong window.
  #
  # It used to read "barlink-583's repro window must clear the fix under live
  # load first". No such window is recorded. What HAS since been recorded is
  # stronger and closer to what this stage actually stresses: #632
  # (b42405c0ee) found the bar1 mesh/a2a peer barrier deadlocking INSIDE GRAPH
  # REPLAY -- the one defect class that would specifically break a captured
  # arm -- replaced it with a consumption-ack barrier that is device-side and
  # capture/replay-safe, and merged it (c6f5b57c3f) on a 2h26m+ amplified-load
  # soak: zero aborts, zero FREEZE, 9.2M ack-barrier rounds with strict
  # cross-rank watermark growth THROUGH graph replay.
  #
  # There is also a direct data point this stage's own family produced:
  # window 4 (3b4526c4ac, 2026-08-05) booted and completed 8 barlink arms with
  # no fault, recorded there as "a clean data point for #583".
  #
  # So the question the operator is attesting is no longer "did #583's repro
  # window run" but "does the soaked barlink stack hold on THIS tree under
  # THIS recipe". Confirm that, then pass BARLINK_VERDICT=confirmed.
  #
  # NOTE ON SCOPE: window 4 has since answered the barlink THROUGHPUT question
  # this stage was built to ask -- interleaved, both points, each against its
  # own floor: +10.25% on 1900 single-stream, -4.62% on 256x4 concurrent. This
  # stage is therefore a CONFIRMATION run, not the primary experiment, and its
  # 1900-only arms cover the point already measured to lose. Re-run it for a
  # boot-floor cross-check; do not re-derive the verdict from it.
  if [ "${BARLINK_VERDICT:-}" != "confirmed" ]; then
    echo "REFUSING barlink stage: BARLINK_VERDICT is not 'confirmed'"
    echo "(attest that the #622/#632-soaked barlink stack holds on this tree"
    echo " under this recipe; see c6f5b57c3f's 2h26m soak. Precondition 1," 
    echo " the #583 fix b001d102fa, is already satisfied here.)"
    exit 2
  fi

  # PRECONDITION 3: the BG arm CAPTURES, so the transport must be capturable.
  # A host-staged transport (shm/gloo/ucx) raises
  # cudaErrorStreamCaptureUnsupported from whichever kernel happens to be
  # capturing, which reads as an unrelated CUDA fault deep inside a boot we
  # already paid for. Checked here, before any card is touched.
  # Capturable: device/host always; bar1/matrix while the #369 release switch
  # SGLANG_BARLINK_GRAPH_ENABLE is on (its default). Default transport is
  # "device", so an unset environment passes.
  _tp="${SGLANG_BARLINK_TRANSPORT:-device}"
  case "$_tp" in
    device|host) ;;
    bar1|matrix)
      case "${SGLANG_BARLINK_GRAPH_ENABLE:-1}" in
        0|false|False|off|OFF)
          echo "REFUSING barlink stage: transport '$_tp' is capturable only"
          echo "while SGLANG_BARLINK_GRAPH_ENABLE is on, and it is off here."
          exit 2 ;;
      esac ;;
    *)
      echo "REFUSING barlink stage: SGLANG_BARLINK_TRANSPORT='$_tp' is"
      echo "host-staged and cannot be captured; the BG arm would fault."
      exit 2 ;;
  esac

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
    """Aggregate tok/s over the sustained window, plus its evidence."""
    p = f"{out}/{arm}_perf_{sz}.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    if d["cached_tokens_total"]:
        print(f"  !! {arm}/{sz}: cache hits present, number is contaminated")
    if d["window_seconds"] < 4.0:
        print(f"  !! {arm}/{sz}: window only {d['window_seconds']:.2f}s -- "
              f"below the ~5s sustained standard, treat as indicative")
    return d


def band(d):
    c = d.get("clocks") or {}
    if not c:
        return "no clock telemetry"
    return " ".join(
        f"g{i}:{v['sm_min']}-{v['sm_max']}/{'|'.join(v['pstates'])}"
        for i, v in c.items()
    )


def block(label, e1, e2, g, sizes):
    print(f"\n=== {label} ===")
    print("Each point: aggregate tok/s over a sustained window. "
          "'win' = window seconds / draws aggregated.")
    for sz in sizes:
        a, b, c = med(e1, sz), med(e2, sz), med(g, sz)
        if a is None or c is None:
            continue
        av, cv = a["aggregate_tok_s"], c["aggregate_tok_s"]
        print(f"\n  [{sz}]  win {a['window_seconds']:.1f}s/{a['draws']}draws "
              f"(eager)  {c['window_seconds']:.1f}s/{c['draws']}draws (graphs)")
        print(f"        clocks eager  {band(a)}")
        print(f"        clocks graphs {band(c)}")
        if b is None:
            print(f"        eager {av:8.1f}  graphs {cv:8.1f}  "
                  f"delta {(cv/av-1)*100:+.1f}%  (NO FLOOR -- E2 missing)")
            continue
        bv = b["aggregate_tok_s"]
        noise = (bv / av - 1) * 100
        emean = (av + bv) / 2
        delta = (cv / emean - 1) * 100
        verdict = "INSIDE NOISE" if abs(delta) <= abs(noise) else "outside noise"
        print(f"        eager {av:8.1f} / {bv:8.1f}   graphs {cv:8.1f}")
        print(f"        eager floor {noise:+6.2f}%   graph delta {delta:+6.2f}%"
              f"   -> {verdict}")


block("NCCL", "E1", "E2", "G", ("256", "900", "1900", "256c4"))
block("barlink", "BE1", "BE2", "BG", ("1900", "256c4"))

# The user's hypothesis, stated as one number per point: does the prefill-graph
# delta become positive under barlink where it was flat/negative under NCCL?
print("\n=== HYPOTHESIS: does the graph delta improve under barlink? ===")
print("(prefill-graph delta under NCCL vs under barlink, same point;")
print(" each compared against its OWN transport's eager floor)")
for sz in ("1900", "256c4"):
    n_e1, n_e2, n_g = med("E1", sz), med("E2", sz), med("G", sz)
    b_e1, b_e2, b_g = med("BE1", sz), med("BE2", sz), med("BG", sz)
    if None in (n_e1, n_g, b_e1, b_g):
        print(f"{sz:>6}: barlink arms absent -- hypothesis UNTESTED at this point")
        continue
    A = lambda d: d["aggregate_tok_s"]
    nd = (A(n_g) / ((A(n_e1) + A(n_e2)) / 2 if n_e2 else A(n_e1)) - 1) * 100
    bd = (A(b_g) / ((A(b_e1) + A(b_e2)) / 2 if b_e2 else A(b_e1)) - 1) * 100
    print(f"{sz:>6}: NCCL {nd:+.1f}%   barlink {bd:+.1f}%   swing {bd - nd:+.1f} pp")
print("\nA swing only supports the hypothesis if it exceeds BOTH transports'")
print("eager floors; otherwise it is noise wearing a mechanism story.")
PY
