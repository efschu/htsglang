#!/usr/bin/env bash
# Task #340, posten 2 -- one card window, four boots, one question.
#
# THE QUESTION
#   #336 ARM B left a two-ingredient result: a TP=2 serving group that was
#   BOTH 3:1-uneven AND carrying a dual-group lane diverges from the TP=1
#   reference at index 1 on all three prompts, while the TP=1 reference is
#   byte-identical on a 5090 and on a 3080. That run changed two things at
#   once, so it cannot name the one that did it.
#
#   This window separates them, on the SAME vehicle (dense Llama-3.1-8B-
#   Instruct), the same three prompts, the same greedy sampling and the same
#   --disable-radix-cache:
#
#     tp1_ref_recheck      TP=1 on the big card. The reference, re-measured on
#                          THIS commit rather than carried over from #336.
#     uneven31_tp2_lane    TP=2, 3:1, WITH lane. The #336 group, re-measured
#                          here so the comparison never crosses worktrees.
#     uneven31_tp2_nolane  TP=2, 3:1, no lane.   Lane removed, unevenness kept.
#     even_tp2_nolane      TP=2, even, no lane.  Both removed, TP=2 kept.
#     even_tp2_stock       TP=2, even, no placement flags at all. The control
#                          that separates "TP=2" from "the fork's placement
#                          path"; last in the queue because it is the one the
#                          budget may drop without voiding the readout.
#
#   Reference first, deliberately: it is the cheapest boot and everything
#   downstream is VOID without it. Then the uneven arm, because the known
#   result already contains the lane -- if uneven-without-lane deviates, the
#   lane is exonerated in one boot.
#
# TIME BOUNDS -- nothing here can wait forever.
#   Every curl carries -m 10. Boot is a bounded poll (BOOT_TIMEOUT_S, 180 s).
#   The probe is bounded twice (its own --deadline-s, composed with the
#   per-request cap, and an outer `timeout`). Teardown SIGTERMs the process
#   GROUP, waits <= 40 s, then SIGKILLs, then waits for the port to be
#   released. Worst case per arm is 180 + 160 + 70 = 410 s, and an arm is only
#   STARTED while that much of BUDGET_S is left -- so the window ends early
#   rather than overrunning. Expected real cost is far lower: roughly 200 s
#   per TP=2 arm and 120 s for the TP=1 reference.
#
# USAGE
#   bash /spinning/wt-340/scripts/dual_group/r10/posten2_window.sh
#   ARMS=tp1_ref_recheck,uneven31_tp2_nolane bash .../posten2_window.sh
#   DRY_RUN=1 bash .../posten2_window.sh      # print the launch lines, no GPU

set -uo pipefail

R10="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../dcp_report.sh
source "$R10/../dcp_report.sh"
WT="${WT:-/spinning/wt-340}"
VENV="${VENV:-/spinning/htsglang-gpu/.venv}"
PY="${PY:-$VENV/bin/python}"
MODEL_ROOT="${MODEL_ROOT:-/spinning/llm_stuff/club-3090/models-cache}"
MODEL="${MODEL:-$MODEL_ROOT/Llama-3.1-8B-Instruct}"
OUT="${OUT:-/spinning/gpu-battery-results/2026-07-31_340_gpu}"
# The #340 window carries more than this posten, so everything written here
# lands in its own subdirectory of the shared result root.
RES="${RES:-$OUT/posten2}"
PORT="${PORT:-30340}"

BUDGET_S="${BUDGET_S:-1500}"
BOOT_TIMEOUT_S="${BOOT_TIMEOUT_S:-180}"
PROBE_DEADLINE_S="${PROBE_DEADLINE_S:-120}"
REQ_TIMEOUT_S="${REQ_TIMEOUT_S:-60}"
TEARDOWN_WAIT_S="${TEARDOWN_WAIT_S:-40}"
# The worst case one arm can cost: boot timeout + probe cap + teardown +
# port release. An arm is only started when that much budget is still left, so
# the window can never overrun BUDGET_S -- it ends early instead.
ARM_MIN_S="${ARM_MIN_S:-$(( BOOT_TIMEOUT_S + PROBE_DEADLINE_S + 40 + TEARDOWN_WAIT_S + 30 ))}"
TOKENS="${TOKENS:-12}"
ARMS="${ARMS:-tp1_ref_recheck,uneven31_tp2_lane,uneven31_tp2_nolane,even_tp2_nolane,even_tp2_stock}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
USE_ARB="${USE_ARB:-1}"
ARB="${ARB:-/spinning/gpu-arb}"
OWNER="${OWNER:-agent-340-posten2}"

# Memory budgets. Two forms, because server_args.py allows exactly two:
#   scalar  -- the only form accepted WITHOUT --rank-tp-ratio ("with even TP
#              all ranks are structurally equal - use a single scalar").
#   list    -- requires --rank-tp-ratio; kept at the #336 values so the uneven
#              arm is a one-flag delta from the run whose result it explains.
EVEN_MIB="${EVEN_MIB:-14000}"
UNEVEN_MIB="${UNEVEN_MIB:-16000,8000}"
REF_FRAC="${REF_FRAC:-0.70}"
STOCK_FRAC="${STOCK_FRAC:-0.60}"

# --- environment -------------------------------------------------------------
# The card resolution below must see the whole rig, so any inherited masking is
# dropped here rather than trusted.
unset CUDA_VISIBLE_DEVICES
export PYTHONPATH="$WT/python"
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
# Identical across all four arms on purpose: an arm-to-arm comparison is only
# readable if the environment is not one of the things that varies.
export SGLANG_UNEVEN_DCP=1
export SGLANG_UNEVEN_DCP_WEIGHTED=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
export FLASHINFER_DISABLE_VERSION_CHECK=1

mkdir -p "$RES/logs"
SUMMARY="$RES/window_summary.txt"
T_START=$(date +%s)
SRV_PID=""
CUR_LABEL=""
HEARTBEAT=""

log() { printf '[%s +%4ss] %s\n' "$(date -u +%H:%M:%S)" "$(( $(date +%s) - T_START ))" "$*"; }
note() { log "$*"; printf '%s\n' "$*" >> "$SUMMARY"; }

is_dry() { [ "$DRY_RUN" = "1" ]; }

pid_alive() {  # $1 = pid; a zombie is not alive
  local pid="${1:-}" state
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  state="$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null)"
  [ "$state" = "Z" ] && return 1
  return 0
}

port_busy() {
  if command -v ss >/dev/null 2>&1; then
    ss -ltn 2>/dev/null | grep -qE "[:.]${PORT}[[:space:]]" && return 0
    return 1
  fi
  # No ss on this box: a server that answers is certainly there. A server that
  # is bound but not yet answering is missed, which only costs a boot failure
  # with "address already in use" in the log -- not a hang.
  curl -sf -m 10 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1
}

wait_port_free() {  # $1 = seconds
  local t0; t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt "${1:-30}" ]; do
    port_busy || return 0
    sleep 2
  done
  return 1
}

teardown() {  # kills the server's whole process group, never the whole box
  local pid="$SRV_PID" pgid
  SRV_PID=""
  [ -n "$pid" ] || return 0
  pgid="$(ps -o pgid= "$pid" 2>/dev/null | tr -d ' ')"
  if [ -n "$pgid" ]; then kill -TERM "-$pgid" 2>/dev/null; else kill -TERM "$pid" 2>/dev/null; fi
  local t0; t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt "$TEARDOWN_WAIT_S" ]; do
    pid_alive "$pid" || break
    sleep 1
  done
  if pid_alive "$pid"; then
    log "teardown: SIGTERM ignored after ${TEARDOWN_WAIT_S}s, SIGKILL"
    [ -n "$pgid" ] && kill -KILL "-$pgid" 2>/dev/null
    kill -KILL "$pid" 2>/dev/null
  fi
  [ -n "$CUR_LABEL" ] && rm -f "$RES/logs/${CUR_LABEL}.pid"
  wait_port_free 30 || log "teardown: port $PORT still bound after 30s"
}

release() {
  [ "$USE_ARB" = "1" ] && [ -f "$ARB/holder" ] && {
    rm -f "$ARB/holder"
    printf '%s %s FREE cards=all #340 posten 2 window done\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$OWNER" >> "$ARB/log" 2>/dev/null
  }
  [ -n "$HEARTBEAT" ] && kill "$HEARTBEAT" 2>/dev/null
  return 0
}

on_exit() { teardown; release; }
trap 'on_exit' EXIT
trap 'log "interrupted"; exit 130' INT TERM

# --- preflight ---------------------------------------------------------------
fail() { log "ABORT: $*"; exit 1; }

[ -x "$PY" ] || fail "python not found at $PY"
[ -d "$MODEL" ] || fail "model not found at $MODEL"
[ -f "$WT/scripts/dual_group/lane_accept_probe.py" ] \
  || fail "lane_accept_probe.py not found under $WT/scripts/dual_group"
[ -f "$R10/probe_arm.py" ] || fail "probe_arm.py missing next to this script"

if ! is_dry; then
  if port_busy; then
    fail "port $PORT is already in use -- another server is running"
  fi
  if [ "$USE_ARB" = "1" ] && [ -f "$ARB/holder" ]; then
    holder_age=$(( $(date +%s) - $(stat -c %Y "$ARB/holder" 2>/dev/null || echo 0) ))
    if [ "$holder_age" -lt 900 ] && [ "$FORCE" != "1" ]; then
      log "another session holds the cards (holder ${holder_age}s old):"
      cat "$ARB/holder"
      fail "cards claimed elsewhere; re-run with FORCE=1 only if that claim is stale"
    fi
  fi
  busy="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
          | awk -F'[, ]+' '$2 > 500 {print $1}')"
  if [ -n "$busy" ] && [ "$FORCE" != "1" ]; then
    nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
    fail "GPUs busy (>500 MiB): $busy -- re-run with FORCE=1 to override"
  fi
fi

: > "$SUMMARY"
{
  echo "# #340 posten 2 window"
  echo "# started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# worktree $WT @ $(git -C "$WT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "# model $MODEL"
  echo "# budget ${BUDGET_S}s, boot timeout ${BOOT_TIMEOUT_S}s, port $PORT"
} >> "$SUMMARY"

# --- card resolution ---------------------------------------------------------
# Never hardcoded: CUDA order and NVML order disagree on this rig, and either
# can shift with a driver state. "big" is whichever card reports the most VRAM.
CARDS_TXT="$RES/cards_resolved.txt"
# DRY_CARDS pins a recorded resolution instead of asking the driver -- used by
# the dry run, and available to an operator who wants a known mapping.
if [ -n "${DRY_CARDS:-}" ]; then
  cp "$DRY_CARDS" "$CARDS_TXT"
else
  "$PY" "$R10/resolve_cards.py" > "$CARDS_TXT" 2>>"$RES/logs/resolve.err" \
    || fail "card resolution failed, see $RES/logs/resolve.err"
fi
eval "$(grep -E '^(CUDA|CVD|NAME|MIB)_[A-Z0-9]+=[A-Za-z0-9_.:-]+$' "$CARDS_TXT")"
CUDA_BIG="${CUDA_BIG:?}"; CUDA_SMALL0="${CUDA_SMALL0:?}"
CVD_BIG="${CVD_BIG:?}"; CVD_SMALL0="${CVD_SMALL0:?}"
note "cards: big=cuda:$CUDA_BIG ${NAME_BIG:-?} ${MIB_BIG:-?}MiB | small0=cuda:$CUDA_SMALL0 ${NAME_SMALL0:-?} ${MIB_SMALL0:-?}MiB"

if [ "$USE_ARB" = "1" ] && ! is_dry; then
  printf "session=%s  cards=all  purpose=#340 posten 2 (budget %ss)  since=%s\n" \
    "$OWNER" "$BUDGET_S" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$ARB/holder"
  printf '%s %s BUSY cards=all #340 posten 2\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$OWNER" >> "$ARB/log" 2>/dev/null
  # The holder's mtime IS the heartbeat; a long boot must not look orphaned.
  ( while true; do sleep 120; touch "$ARB/holder" 2>/dev/null || exit 0; done ) &
  HEARTBEAT=$!
fi

# --- one arm -----------------------------------------------------------------
COMMON_FLAGS=(
  --model-path "$MODEL" --tokenizer-path "$MODEL"
  --attention-backend flashinfer
  --context-length 8192
  --trust-remote-code
  --max-running-requests 2
  --disable-radix-cache
  --host 127.0.0.1 --port "$PORT"
)

launch() {  # $1 = label, $2 = CUDA_VISIBLE_DEVICES ("-" for unset), rest = flags
  local label="$1" cvd="$2"; shift 2
  local server_log="$RES/logs/${label}.server.log"
  CUR_LABEL="$label"
  if is_dry; then
    echo "DRY RUN launch: CUDA_VISIBLE_DEVICES=${cvd} $PY -m sglang.launch_server $*"
    return 0
  fi
  cd "$WT" || return 1
  if [ "$cvd" = "-" ]; then
    setsid "$PY" -m sglang.launch_server "$@" > "$server_log" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="$cvd" setsid "$PY" -m sglang.launch_server "$@" \
      > "$server_log" 2>&1 &
  fi
  SRV_PID=$!
  echo "$SRV_PID" > "$RES/logs/${label}.pid"
  return 0
}

wait_up() {  # $1 = label
  local label="$1" t0 up=0
  t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt "$BOOT_TIMEOUT_S" ]; do
    if curl -sf -m 10 "http://127.0.0.1:$PORT/health_generate" >/dev/null 2>&1; then
      up=1; break
    fi
    pid_alive "$SRV_PID" || { log "$label: server process died"; break; }
    sleep 3
  done
  if [ "$up" = 1 ]; then
    log "$label: up after $(( $(date +%s) - t0 ))s"
    return 0
  fi
  tail -30 "$RES/logs/${label}.server.log" > "$RES/logs/${label}.boot_tail.txt" 2>/dev/null
  return 1
}

run_arm() {  # $1 = label, $2 = config text, $3 = cvd, rest = extra flags
  local label="$1" config="$2" cvd="$3"; shift 3
  note "=== arm $label ($config) ==="
  launch "$label" "$cvd" "${COMMON_FLAGS[@]}" "$@" || { note "$label: LAUNCH FAILED"; return 1; }
  if is_dry; then note "$label: DRY RUN ok"; CUR_LABEL=""; return 0; fi

  local rc=1
  if wait_up "$label"; then
    # HARNESS DUTY (#345): state the EFFECTIVE dcp geometry per arm. The env
    # DCP flags only bite under --rank-tp-ratio, so an unprinted matrix
    # compares two changes at once (that is what #340 did).
    report_dcp "$label" "$RES/logs/${label}.server.log"
    timeout -k 10 "$(( PROBE_DEADLINE_S + 40 ))" \
      "$PY" "$R10/probe_arm.py" \
        --port "$PORT" --tokenizer "$MODEL" --tokens "$TOKENS" \
        --label "$label" --config "$config" \
        --module-dir "$WT/scripts/dual_group" \
        --deadline-s "$PROBE_DEADLINE_S" --req-timeout-s "$REQ_TIMEOUT_S" \
        --out "$RES/${label}.json" 2>&1 | tee -a "$SUMMARY"
    rc=${PIPESTATUS[0]}
    curl -sf -m 10 "http://127.0.0.1:$PORT/get_server_info" \
      > "$RES/logs/${label}.server_info.json" 2>/dev/null
  else
    note "$label: BOOT FAILED (see $RES/logs/${label}.boot_tail.txt)"
  fi
  teardown
  CUR_LABEL=""
  note "$label: rc=$rc"
  return "$rc"
}

arm_tp1_ref_recheck() {
  # The reference. TP=1 on the big card, no split and no lane -- nothing that
  # either disagreeing side of the #336 result is made of. mem-fraction 0.70 is
  # far above what 15.0 GiB of weights plus a 12-token generation needs.
  run_arm tp1_ref_recheck "tp1 on cuda:$CUDA_BIG, mem-frac $REF_FRAC" "$CVD_BIG" \
    --tp-size 1 --mem-fraction-static "$REF_FRAC"
}

arm_uneven31_tp2_nolane() {
  # The #336 serving group with the lane taken out, and nothing else changed:
  # same cards, same 3:1 ratio, same 16000,8000 budgets.
  run_arm uneven31_tp2_nolane \
    "tp2 3:1 on cuda:$CUDA_BIG,$CUDA_SMALL0, mib $UNEVEN_MIB, no lane" "-" \
    --tp-size 2 --rank-gpu-id "$CUDA_BIG,$CUDA_SMALL0" \
    --rank-tp-ratio 3,1 --rank-gpu-memory-mib "$UNEVEN_MIB"
}

arm_even_tp2_nolane() {
  # Same placement path, even split. server_args.py rejects --rank-gpu-id on
  # its own ("requires --rank-gpu-memory-mib to be set") and rejects a per-rank
  # MiB LIST without --rank-tp-ratio, so the even form of this arm is
  # --rank-gpu-id plus a single SCALAR budget. That scalar is each rank's whole
  # budget on its own card, which is the equal-budget form the even split wants
  # anyway: 14000 MiB fits both the big card and a 20480 MiB small one.
  run_arm even_tp2_nolane \
    "tp2 even on cuda:$CUDA_BIG,$CUDA_SMALL0, mib $EVEN_MIB (scalar), no lane" "-" \
    --tp-size 2 --rank-gpu-id "$CUDA_BIG,$CUDA_SMALL0" \
    --rank-gpu-memory-mib "$EVEN_MIB"
}

arm_even_tp2_stock() {
  # The control with NO fork placement flags at all: the two ranks land on the
  # same two cards through CUDA_VISIBLE_DEVICES instead. If this one deviates
  # too, the finding is about TP=2 as such and not about anything this fork
  # adds. Last in the queue: it sharpens a verdict, it does not carry one.
  run_arm even_tp2_stock \
    "tp2 even via CUDA_VISIBLE_DEVICES=$CVD_BIG,$CVD_SMALL0, mem-frac $STOCK_FRAC" \
    "$CVD_BIG,$CVD_SMALL0" \
    --tp-size 2 --mem-fraction-static "$STOCK_FRAC"
}

arm_uneven31_tp2_lane() {
  # The #336 serving group EXACTLY as it was measured there, re-run on this
  # commit. Two jobs at once:
  #   * posten 2 -- without it the other arms compare against a number carried
  #     over from another worktree, and "the lane's presence perturbs the
  #     serving group" could not be stated, only inferred.
  #   * posten 1 -- this is the dense bf16 coherence smoke for the two-card
  #     lane. bf16 is arch-neutral, so unlike the FP8 boot it can actually
  #     reach a result on a 5090 + 3080, and the device guard now sits in its
  #     forward path.
  run_arm uneven31_tp2_lane \
    "tp2 3:1 on cuda:$CUDA_BIG,$CUDA_SMALL0, mib $UNEVEN_MIB, WITH lane" "-" \
    --tp-size 2 --rank-gpu-id "$CUDA_BIG,$CUDA_SMALL0" \
    --rank-tp-ratio 3,1 --rank-gpu-memory-mib "$UNEVEN_MIB" \
    --dual-group-lane --dual-group-lane-budget-mib 1600 \
    --dual-group-lane-part-gpu-id "$CUDA_BIG,$CUDA_SMALL0"
}

# --- the queue ---------------------------------------------------------------
for arm in ${ARMS//,/ }; do
  left=$(( BUDGET_S - ($(date +%s) - T_START) ))
  if [ "$left" -lt "$ARM_MIN_S" ]; then
    note "$arm: SKIPPED (window budget exhausted, ${left}s left, needs >=${ARM_MIN_S}s)"
    continue
  fi
  if ! declare -F "arm_$arm" >/dev/null; then
    note "$arm: SKIPPED (unknown arm)"
    continue
  fi
  "arm_$arm"
  is_dry || sleep 5
  if ! is_dry; then
    { date -u +%Y-%m-%dT%H:%M:%SZ
      nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
    } >> "$RES/logs/between_arms.txt" 2>/dev/null
  fi
done

# --- readout -----------------------------------------------------------------
note "--- window finished after $(( $(date +%s) - T_START ))s of ${BUDGET_S}s ---"
if ! is_dry; then
  "$PY" "$R10/compare_arms.py" --out "$RES" --write "$RES/verdict.txt"
  note "verdict written to $RES/verdict.txt"
fi
exit 0
