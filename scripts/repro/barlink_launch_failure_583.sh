#!/usr/bin/env bash
# #583 -- GPU repro harness for the barlink "unspecified launch failure" crash.
#
# WHAT THIS IS FOR
# ----------------
# The 2026-08-05 production boot died after 7 minutes with a sticky
# cudaErrorLaunchFailure surfacing inside Triton's load_binary. The desk
# analysis established the AMPLIFIER (barlink's device spin kernels called
# __trap() on deadline expiry, which destroys the CUDA context and makes every
# later CUDA call fail at an unrelated site) and fixed it: a tripped kernel now
# writes seq_dev[1] and the host raises DeviceCollectiveAborted naming the
# kernel, the rank and the deadline.
#
# What the desk could NOT decide is the TRIGGER: why a spin kernel reached a
# ~30 s deadline at all. This harness exists to answer that, and the fix above
# is what makes it answerable -- before it, the context died and every
# diagnostic lied about which kernel was at fault.
#
# ARMS
#   baseline    the crash configuration, unmodified. With the #583 fix in the
#               tree the expected outcome is DeviceCollectiveAborted naming the
#               kernel and rank -- that alone identifies WHICH wait expired.
#   blocking    CUDA_LAUNCH_BLOCKING=1. Serializes launches so an async fault
#               is reported at its true call site rather than at the next
#               unrelated CUDA call. Slow; expect the deadline to be reached
#               more easily, so it is the arm most likely to trip the abort.
#   sanitizer   compute-sanitizer --tool memcheck. Catches an illegal access
#               that would otherwise only show up as context corruption. Very
#               slow (10-50x); the time box is raised accordingly.
#
# THIS SCRIPT DOES NOT TOUCH THE PRODUCTION SERVER. It boots its own instance
# on its own port. The stop/restart protocol for the production server is
# printed by --print-serving-protocol and is NEVER executed automatically:
# main grants the window and runs it.
#
# Usage:
#   ./barlink_launch_failure_583.sh --arm baseline [--minutes 12]
#   ./barlink_launch_failure_583.sh --print-serving-protocol
set -uo pipefail

WT="${WT:-/spinning/wt-barlink-583}"
VENV="${VENV:-/spinning/htsglang-gpu/.venv}"
PY="$VENV/bin/python"
MODEL="${MODEL:-/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8}"
PORT="${PORT:-30077}"          # NOT 30030 (spill-night) and NOT 30099 (router)
ARM="baseline"
MINUTES=""
OUTDIR="${OUTDIR:-/spinning/spill-night-20260804/results/583_repro}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arm) ARM="$2"; shift 2 ;;
    --minutes) MINUTES="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --print-serving-protocol) PRINT_PROTOCOL=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# Serving stop/restart protocol -- PRINTED, NEVER EXECUTED.
# ---------------------------------------------------------------------------
if [[ -n "${PRINT_PROTOCOL:-}" ]]; then
  cat <<'PROTOCOL'
PRODUCTION SERVING STOP / RESTART PROTOCOL (#583 repro window)

The rig's production server backs the local Qwen sub-subagents through the
router on 30099. Stopping it stops every agent on the box, so this runs ONLY
inside a window main has granted, and main runs it -- not this script.

BEFORE the window
  1. Announce the window and confirm no other agent holds a GPU claim:
       ls -la /spinning/gpu-arb/
  2. Record what is running, so it can be restored byte-for-byte:
       ps -o pid,etime,args -C python | grep -F sglang | tee /tmp/583_serving_cmdline.txt
     Keep this file. It is the ONLY record of the exact production flags.
  3. Note the free VRAM on all three cards (the corridor rule: >= 400 MiB free
     on EVERY card after teardown, or the next boot fails for the wrong reason):
       nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv

DURING teardown
  4. Stop ONLY the production scheduler PIDs recorded in step 2. Never a broad
     pkill -- it kills sibling agents' work and has caused a self-kill before:
       kill <pid>            # SIGTERM first, give it 30 s
       kill -9 <pid>         # only if it has not exited
  5. Verify VRAM actually came back before booting anything:
       nvidia-smi --query-gpu=index,memory.free --format=csv
     A card still holding GiB means an orphaned process -- find it with
     `fuser -v /dev/nvidia*` and clear it, or the repro measures the wrong rig.

AFTER the window
  6. Restore production from /tmp/583_serving_cmdline.txt, exactly as recorded.
  7. Confirm the router is healthy again before releasing the window:
       curl -s -m 5 http://127.0.0.1:30099/health && echo ROUTER-OK
  8. Confirm an agent round-trips (the router being up is not the same as the
     model answering).

NEVER kill the router on 30099 from inside an agent session.
PROTOCOL
  exit 0
fi

case "$ARM" in
  baseline)  DEFAULT_MINUTES=12 ;;
  blocking)  DEFAULT_MINUTES=15 ;;
  sanitizer) DEFAULT_MINUTES=25 ;;
  *) echo "unknown arm: $ARM (baseline|blocking|sanitizer)" >&2; exit 2 ;;
esac
MINUTES="${MINUTES:-$DEFAULT_MINUTES}"

mkdir -p "$OUTDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$OUTDIR/583_${ARM}_${STAMP}.log"

# ---------------------------------------------------------------------------
# Physical GPU identity via NVML -- never a hardcoded index. NVML/nvidia-smi
# enumeration and torch's CUDA order can and do diverge on this rig.
# ---------------------------------------------------------------------------
echo "== NVML physical inventory ==" | tee "$LOG"
"$PY" - <<'NVML' 2>&1 | tee -a "$LOG"
import pynvml
pynvml.nvmlInit()
for i in range(pynvml.nvmlDeviceGetCount()):
    h = pynvml.nvmlDeviceGetHandleByIndex(i)
    name = pynvml.nvmlDeviceGetName(h)
    name = name.decode() if isinstance(name, bytes) else name
    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
    print(f"  nvml index {i}: {name}  total={mem.total // 1024**2} MiB "
          f"free={mem.free // 1024**2} MiB")
NVML

# The crash boot ran rank_gpu_id=[0,1,2] with rank_gpu_memory_mib
# [27107,16680,16680] -- the 5090 first, then the two 3080s. Keep that shape;
# it is part of the configuration under test (uneven DCP over mixed cards).
RANK_GPU_ID="${RANK_GPU_ID:-0,1,2}"
RANK_GPU_MEM="${RANK_GPU_MEM:-27107,16680,16680}"

# ---------------------------------------------------------------------------
# Boot flags: the crash configuration from CRASH_20260805_boot5_barlink_full.log.
# The load-bearing combination is barlink device transport + eager prefill +
# FULL decode CUDA graphs + write-through HiCache on the direct io backend --
# the barlink chunk transfers and the HiCache write-through share the PCIe copy
# engines, which is the leading trigger hypothesis for the missed deadline.
# ---------------------------------------------------------------------------
BOOT=(
  "$PY" -m sglang.launch_server
  --model-path "$MODEL"
  --served-model-name Qwen3.6-27B-583
  --trust-remote-code
  --port "$PORT" --host 127.0.0.1
  --tp-size 3 --dcp-size 3
  --rank-gpu-id "$RANK_GPU_ID"
  # auto-performance, exactly as production and the crash boot ran it. The
  # crash log's rank_gpu_memory_mib=[27107,16680,16680] and
  # rank_tp_ratio=[27107,16680,16680] are the DERIVED values, not flags --
  # passing them explicitly instead collides with
  # "--rank-auto-reserve-mib only applies with --rank-tp-ratio auto"
  # (server_args.py:9656).
  --rank-tp-ratio auto-performance
  --rank-perf-tune phase-decode
  --rank-auto-reserve-mib 5500,3800,3800
  --kv-cache-dtype fp8_e4m3
  --context-length 262144
  --max-running-requests 4
  --chunked-prefill-size 2048
  --page-size 1
  --attention-backend flashinfer
  --mamba-backend triton
  # Production's value. The default (64) is not enough for four concurrent
  # long-prefix sessions: the run dies in _alloc_ping_pong_buffer
  # (memory_pool.py:1494, "Not enough space for mamba ping pong idx") long
  # before any barlink deadline is approached, which masks what this harness
  # exists to measure. That assert killing the scheduler is a separate defect
  # -- the merged "Mamba slot starvation must not kill the scheduler" fix does
  # not cover this allocation site.
  --max-mamba-cache-size 96
  # EAGLE, exactly as the crash boot's server_args recorded it
  # (speculative_algorithm='EAGLE', speculative_draft_model_path=None -- Qwen3.6
  # carries its own MTP head, so no draft model is needed). NOT "NEXTN": that
  # alias is resolved by the arg hook only AFTER _handle_dcp_validation has
  # already called SpeculativeAlgorithm.from_string on the raw string, so under
  # --dcp-size it raises "Unknown speculative algorithm name: NEXTN".
  --speculative-algorithm EAGLE
  --speculative-num-steps 3
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4
  --enable-hierarchical-cache
  --hicache-ratio 2.0
  --hicache-write-policy write_through
  --hicache-io-backend direct
  --hicache-mem-layout page_first_direct
  --cuda-graph-config '{"decode":{"backend":"full","max_bs":24},"prefill":{"backend":"disabled"}}'
  --disable-custom-all-reduce
  --enable-metrics
  --decode-log-interval 40
)

export SGLANG_BARLINK=1
# spec + DCP is only permitted on the uneven-hybrid WEIGHTED path, which is
# what the crash boot ran (rank_tp_ratio=[27107,16680,16680], non-uniform).
# Without these two, _handle_dcp_validation refuses the combination outright.
export SGLANG_UNEVEN_DCP=1
export SGLANG_UNEVEN_DCP_WEIGHTED=1
# Do NOT silence the #583 abort check: the whole point of the window is to see
# the structured DeviceCollectiveAborted instead of a dead context.
unset SGLANG_BARLINK_BAR1_ABORT_CHECK
export PYTHONPATH="$WT/python"
export PYTHONUNBUFFERED=1

case "$ARM" in
  blocking)  export CUDA_LAUNCH_BLOCKING=1 ;;
  sanitizer) BOOT=(compute-sanitizer --tool memcheck --launch-timeout 120
                   --error-exitcode 88 "${BOOT[@]}") ;;
esac

echo "== arm=$ARM  time box=${MINUTES} min  port=$PORT ==" | tee -a "$LOG"
echo "== log: $LOG ==" | tee -a "$LOG"

timeout --signal=INT "$(( MINUTES * 60 ))" "${BOOT[@]}" >>"$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill -INT $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null' EXIT INT TERM

# ---------------------------------------------------------------------------
# Wait for readiness, then drive the mixed load.
# ---------------------------------------------------------------------------
echo "== waiting for readiness (max 15 min) ==" | tee -a "$LOG"
for _ in $(seq 1 900); do
  curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SERVER_PID 2>/dev/null || { echo "SERVER DIED BEFORE READY -- see $LOG"; exit 1; }
  sleep 1
done
echo "== ready, starting mixed load ==" | tee -a "$LOG"

# The crash profile, reproduced: 3-4 concurrent sessions, each with a LONG
# cached prefix (40-80k cached tokens) plus a fresh 600-2000 token chunk, and a
# streaming decode running alongside. That is what interleaves eager prefill
# with full-graph decode at high frequency -- the mix the desk analysis
# identified as the one no NCCL/gloo boot ever exercised.
PORT="$PORT" MINUTES="$MINUTES" "$PY" - <<'LOAD' 2>&1 | tee -a "$LOG"
import json, os, random, threading, time, urllib.request

port = os.environ["PORT"]
deadline = time.time() + int(os.environ["MINUTES"]) * 60 - 90
url = f"http://127.0.0.1:{port}/v1/chat/completions"
random.seed(583)
stop = threading.Event()
errors = []

def session(idx: int):
    # A growing conversation: each turn re-sends the whole history, so the
    # cached-token count climbs into the tens of thousands exactly as in the
    # crash log, while each turn still adds a fresh chunk to prefill.
    #
    # BOUNDED ON PURPOSE (2026-08-05 window). An unbounded version of this
    # loop drove `mamba usage` to 1.00 and killed both arms on mamba
    # allocation asserts long before any barlink deadline was approached --
    # first "Not enough space for mamba ping pong idx", then, with
    # --max-mamba-cache-size 96, "Not enough space for mamba cache". Neither
    # is a barlink fault and both masked what this harness measures. The crash
    # boot ran at mamba usage 0.25-0.28 with 3-4 running requests, so the load
    # has to stay in THAT regime: distinct long prefixes are what consume
    # mamba states, so each session recycles its prefix instead of growing one
    # forever. Watch the "mamba usage" field in the log -- if it climbs past
    # ~0.5 the arm is measuring the wrong thing.
    base = f"Session {idx}. Explain memory coherence. " + "Detail. " * 120
    history = [{"role": "user", "content": base}]
    turns = 0
    while not stop.is_set() and time.time() < deadline:
        turns += 1
        if turns % 6 == 0:
            # Recycle: drop back to the seed prefix so the number of distinct
            # cached prefixes (and therefore mamba states) stays bounded.
            history = [{"role": "user", "content": base}]
        body = json.dumps({
            "model": "Qwen3.6-27B-583",
            "messages": history,
            "max_tokens": random.choice([48, 96, 192]),
            "stream": False,
            "temperature": 0.7,
        }).encode()
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                out = json.loads(resp.read())
            text = out["choices"][0]["message"]["content"]
            history.append({"role": "assistant", "content": text})
            history.append({"role": "user",
                            "content": "Continue in more detail. " * 200})
            if len(history) > 9:            # keep the prefix long but bounded
                history = history[:1] + history[-6:]
        except Exception as exc:            # noqa: BLE001 - report, keep load on
            errors.append(f"session {idx}: {exc!r}")
            if len(errors) > 40:
                stop.set()
            time.sleep(2)

threads = [threading.Thread(target=session, args=(i,), daemon=True)
           for i in range(4)]
for t in threads:
    t.start()
while any(t.is_alive() for t in threads) and time.time() < deadline:
    time.sleep(5)
stop.set()
for t in threads:
    t.join(timeout=30)

print(f"== load finished, {len(errors)} request errors ==")
for e in errors[:20]:
    print("  ", e)
LOAD

kill -INT $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

# ---------------------------------------------------------------------------
# Verdict.
# ---------------------------------------------------------------------------
echo "================ VERDICT (arm=$ARM) ================" | tee -a "$LOG"
if grep -q "DeviceCollectiveAborted" "$LOG"; then
  echo "TRIGGER REPRODUCED, and named. The #583 fix converted the context kill" | tee -a "$LOG"
  echo "into a structured abort. The kernel/rank in the message below is the" | tee -a "$LOG"
  echo "wait that actually expired -- that is the trigger to chase next:" | tee -a "$LOG"
  grep -A6 "DeviceCollectiveAborted" "$LOG" | head -30 | tee -a "$LOG"
elif grep -q "unspecified launch failure" "$LOG"; then
  echo "CONTEXT STILL DIED. Either the fix is not in the booted tree (check" | tee -a "$LOG"
  echo "PYTHONPATH=$WT/python) or there is a SECOND context-killer beyond the" | tee -a "$LOG"
  echo "spin-kernel trap. Check for a compute-sanitizer report above." | tee -a "$LOG"
elif grep -qE "compute-sanitizer.*error|ERROR SUMMARY: [1-9]" "$LOG"; then
  echo "SANITIZER FOUND AN ILLEGAL ACCESS -- this is the true first fault:" | tee -a "$LOG"
  grep -B4 -A12 "ERROR SUMMARY" "$LOG" | head -40 | tee -a "$LOG"
else
  echo "NO FAULT IN ${MINUTES} MIN. The crash took 7 min under real agent load;" | tee -a "$LOG"
  echo "this arm did not reach it. Raise --minutes, or raise the PCIe pressure" | tee -a "$LOG"
  echo "(more concurrent sessions / longer prefixes) since copy-engine" | tee -a "$LOG"
  echo "contention with HiCache write-through is the leading trigger theory." | tee -a "$LOG"
fi
echo "full log: $LOG" | tee -a "$LOG"
