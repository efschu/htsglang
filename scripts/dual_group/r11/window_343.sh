#!/usr/bin/env bash
# Task #343 -- one card window, three boots, one question.
#
# THE QUESTION
#   #340 localised the uneven-TP decode bug to the split itself: TP=1 and an
#   EVEN TP=2 agree byte for byte on three greedy prompts, while a 3:1
#   --rank-tp-ratio TP=2 diverges at output index 1 on all three, with and
#   without the dual-group lane, identically. The first token is right, so
#   prefill holds and the first DECODE step is where it goes wrong.
#
#   That is as far as a token stream can take it. This window opens the
#   forward pass instead: the layer fingerprint tap
#   (sglang.srt.model_executor.layer_fingerprint) records a hash of every
#   decoder layer's output per forward step, on every rank, and the comparison
#   names the first tensor whose hash stops matching the TP=1 reference.
#
#     tp1_ref       TP=1 on the big card. The reference; everything else is
#                   VOID without it, so it boots first and it is the cheapest.
#     uneven31_a    TP=2, 3:1. The arm under test. THE DELIVERABLE.
#     uneven31_b    TP=2, 3:1, byte-identical settings, separate boot. Answers
#                   a second question #340 left open: is the 3:1 arm at least
#                   deterministic with respect to ITSELF, or does it carry its
#                   own nondeterminism on top of the divergence? Last, because
#                   it sharpens the verdict rather than carrying it.
#
#   CUDA graphs are off in every arm. A forward hook that copies to host is
#   illegal inside a capture and would silently record nothing at replay, so
#   graph capture would leave the decode steps -- the whole point -- blank.
#   All three arms pay it equally, so the comparison stays readable.
#
# TIME BOUNDS -- nothing here waits forever. Same discipline as the #340 r10
#   window: curl -m 10, bounded boot poll, a probe bounded twice, teardown by
#   process GROUP with SIGTERM then SIGKILL, and an arm is only STARTED while
#   its worst case still fits in BUDGET_S.
#
# USAGE
#   bash scripts/dual_group/r11/window_343.sh
#   DRY_RUN=1 bash scripts/dual_group/r11/window_343.sh   # launch lines only
#   ARMS=tp1_ref,uneven31_a bash scripts/dual_group/r11/window_343.sh

set -uo pipefail

R11="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../dcp_report.sh
source "$R11/../dcp_report.sh"
WT="${WT:-/spinning/wt-343-probe}"
VENV="${VENV:-/spinning/htsglang-gpu/.venv}"
PY="${PY:-$VENV/bin/python}"
MODEL_ROOT="${MODEL_ROOT:-/spinning/llm_stuff/club-3090/models-cache}"
MODEL="${MODEL:-$MODEL_ROOT/Llama-3.1-8B-Instruct}"
RES="${RES:-/spinning/gpu-battery-results/2026-07-31_343_probe}"
PORT="${PORT:-30343}"

BUDGET_S="${BUDGET_S:-1700}"
BOOT_TIMEOUT_S="${BOOT_TIMEOUT_S:-240}"
PROBE_DEADLINE_S="${PROBE_DEADLINE_S:-180}"
REQ_TIMEOUT_S="${REQ_TIMEOUT_S:-90}"
TEARDOWN_WAIT_S="${TEARDOWN_WAIT_S:-40}"
ARM_MIN_S="${ARM_MIN_S:-$(( BOOT_TIMEOUT_S + PROBE_DEADLINE_S + 40 + TEARDOWN_WAIT_S + 30 ))}"
TOKENS="${TOKENS:-12}"
ARMS="${ARMS:-tp1_ref,uneven31_a,uneven31_b}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
USE_ARB="${USE_ARB:-1}"
ARB="${ARB:-/spinning/gpu-arb}"
OWNER="${OWNER:-agent-343-probe}"

UNEVEN_MIB="${UNEVEN_MIB:-16000,8000}"
EVEN_MIB="${EVEN_MIB:-14000}"
REF_FRAC="${REF_FRAC:-0.70}"

# --- environment -------------------------------------------------------------
# Card resolution must see the whole rig, so inherited masking is dropped here.
unset CUDA_VISIBLE_DEVICES
export PYTHONPATH="$WT/python"
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export SGLANG_UNEVEN_DCP=1
export SGLANG_UNEVEN_DCP_WEIGHTED=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
export FLASHINFER_DISABLE_VERSION_CHECK=1
# The second gate of the layer tap. Full tensors only for astep 1 -- the first
# decode step, the one under investigation; the prefill step's verdict comes
# from its hashes, which is the same verdict for a fraction of the bytes.
export SGLANG_DETERMINISM_LAYER_FINGERPRINT=1
export SGLANG_DETERMINISM_LAYER_FULL_STEPS="${SGLANG_DETERMINISM_LAYER_FULL_STEPS:-1}"

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

teardown() {  # kills the server's whole process group, never anything else
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
    printf '%s %s FREI cards=0,1,2 #343 layer-delta probe window done\n' \
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
[ -f "$WT/scripts/dual_group/lane_accept_probe.py" ] || fail "lane_accept_probe.py missing"
[ -f "$R11/probe_arm.py" ] || fail "probe_arm.py missing next to this script"

if ! is_dry; then
  port_busy && fail "port $PORT is already in use"
  if [ "$USE_ARB" = "1" ] && [ -f "$ARB/holder" ]; then
    holder_age=$(( $(date +%s) - $(stat -c %Y "$ARB/holder" 2>/dev/null || echo 0) ))
    if [ "$holder_age" -lt 900 ] && [ "$FORCE" != "1" ]; then
      cat "$ARB/holder"
      fail "cards claimed elsewhere (holder ${holder_age}s old); FORCE=1 only if stale"
    fi
  fi
  busy="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader \
          | awk -F'[, ]+' '$2 > 500 {print $1}')"
  if [ -n "$busy" ] && [ "$FORCE" != "1" ]; then
    nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
    fail "GPUs busy (>500 MiB): $busy -- FORCE=1 to override"
  fi
fi

: > "$SUMMARY"
{
  echo "# #343 layer-delta probe window"
  echo "# started $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# worktree $WT @ $(git -C "$WT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "# model $MODEL"
  echo "# budget ${BUDGET_S}s, boot timeout ${BOOT_TIMEOUT_S}s, port $PORT"
} >> "$SUMMARY"

# --- card resolution ---------------------------------------------------------
# Never hardcoded: CUDA order and NVML order disagree on this rig.
CARDS_TXT="$RES/cards_resolved.txt"
if [ -n "${DRY_CARDS:-}" ]; then
  cp "$DRY_CARDS" "$CARDS_TXT"
else
  "$PY" "$R11/resolve_cards.py" > "$CARDS_TXT" 2>>"$RES/logs/resolve.err" \
    || fail "card resolution failed, see $RES/logs/resolve.err"
fi
eval "$(grep -E '^(CUDA|CVD|NAME|MIB)_[A-Z0-9]+=[A-Za-z0-9_.:-]+$' "$CARDS_TXT")"
CUDA_BIG="${CUDA_BIG:?}"; CUDA_SMALL0="${CUDA_SMALL0:?}"
CVD_BIG="${CVD_BIG:?}"
note "cards: big=cuda:$CUDA_BIG ${NAME_BIG:-?} ${MIB_BIG:-?}MiB | small0=cuda:$CUDA_SMALL0 ${NAME_SMALL0:-?} ${MIB_SMALL0:-?}MiB"

if [ "$USE_ARB" = "1" ] && ! is_dry; then
  printf "session=%s  cards=0,1,2  purpose=#343 layer-delta probe (budget %ss)  since=%s\n" \
    "$OWNER" "$BUDGET_S" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$ARB/holder"
  printf '%s %s BELEGT cards=0,1,2 purpose=#343 layer-delta probe (budget %ss)\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$OWNER" "$BUDGET_S" >> "$ARB/log" 2>/dev/null
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
  --cuda-graph-backend-decode disabled
  --cuda-graph-backend-prefill disabled
  --enable-metrics
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
  local dump="$RES/dump_${label}"
  rm -rf "$dump"; mkdir -p "$dump"
  note "=== arm $label ($config) ==="
  launch "$label" "$cvd" "${COMMON_FLAGS[@]}" \
    --determinism-logits-dump-dir "$dump" "$@" \
    || { note "$label: LAUNCH FAILED"; return 1; }
  if is_dry; then note "$label: DRY RUN ok"; CUR_LABEL=""; return 0; fi

  local rc=1
  if wait_up "$label"; then
    # HARNESS DUTY (#345): every arm states its EFFECTIVE dcp geometry before
    # it produces a number. The env DCP flags only bite under
    # --rank-tp-ratio, so without this line a ratio arm and its control differ
    # in two things at once and the matrix cannot say which one moved.
    report_dcp "$label" "$RES/logs/${label}.server.log"
    # ARM the layer tap only now: boot warmup and memory profiling have run
    # their forwards, so astep 0 is the probe's first prefill in every arm.
    touch "$dump/ARM"
    timeout -k 10 "$(( PROBE_DEADLINE_S + 40 ))" \
      "$PY" "$R11/probe_arm.py" \
        --port "$PORT" --tokenizer "$MODEL" --tokens "$TOKENS" \
        --label "$label" --config "$config" \
        --module-dir "$WT/scripts/dual_group" \
        --deadline-s "$PROBE_DEADLINE_S" --req-timeout-s "$REQ_TIMEOUT_S" \
        --out "$RES/${label}.json" 2>&1 | tee -a "$SUMMARY"
    rc=${PIPESTATUS[0]}
  else
    note "$label: BOOT FAILED (see $RES/logs/${label}.boot_tail.txt)"
  fi
  teardown
  CUR_LABEL=""
  note "$label: rc=$rc, $(ls "$dump" | wc -l) dump files"
  return "$rc"
}

arm_tp1_ref() {
  run_arm tp1_ref "tp1 on cuda:$CUDA_BIG, mem-frac $REF_FRAC" "$CVD_BIG" \
    --tp-size 1 --mem-fraction-static "$REF_FRAC"
}

_uneven31() {  # $1 = label; the two 3:1 boots differ in nothing but the label
  run_arm "$1" \
    "tp2 3:1 on cuda:$CUDA_BIG,$CUDA_SMALL0, mib $UNEVEN_MIB" "-" \
    --tp-size 2 --rank-gpu-id "$CUDA_BIG,$CUDA_SMALL0" \
    --rank-tp-ratio 3,1 --rank-gpu-memory-mib "$UNEVEN_MIB"
}

arm_uneven31_a() { _uneven31 uneven31_a; }
arm_uneven31_b() { _uneven31 uneven31_b; }

arm_uneven31_nodcp() {
  # SEPARATES THE TWO THINGS --rank-tp-ratio TURNS ON AT ONCE.
  #
  # With SGLANG_UNEVEN_DCP=1 in the environment (which the #340 harness env
  # sets, and this window inherited), installing a ratio plan ALSO switches
  # the full-attention KV cache to the uneven-DCP geometry: dcp_size becomes
  # tp_size, the KV is TOKEN-sharded across the ranks, and each rank's local
  # paged attention is LSE-combined across the DCP group. The two control arms
  # ran dcp_size=1. So "3:1 differs from even" so far means "3:1 head split
  # AND token-sharded KV differ from even", which cannot name either.
  #
  # This arm is the 3:1 head split with the DCP geometry taken back out:
  # identical in every other flag. If the decode divergence survives, it
  # belongs to the uneven head split; if it vanishes, it belongs to the
  # uneven-DCP KV path, which is also the only one of the two that is a
  # decode-time mechanism -- prefill runs the ragged path over local tokens.
  local prev="${SGLANG_UNEVEN_DCP:-}"
  export SGLANG_UNEVEN_DCP=0
  _uneven31 uneven31_nodcp
  local rc=$?
  export SGLANG_UNEVEN_DCP="$prev"
  return $rc
}

arm_uneven31_dcp_unweighted() {
  # Splits the uneven-DCP path itself in two. Both halves keep dcp_size=2 and
  # the replicated-kv-head geometry (uneven_dcp_kv_replicated is true as soon
  # as a ratio plan is installed); what differs is the OWNER RULE:
  #
  #   WEIGHTED   (SGLANG_UNEVEN_DCP_WEIGHTED=1) a [3,1] token vector, so a
  #              global slot L is owned by the rank whose prefix range covers
  #              L % 4, and its compact slot is (L // 4) * width + (L % 4 - lo).
  #   UNWEIGHTED even modulo: L % 2 == rank, compact slot L // 2.
  #
  # If unweighted is correct and weighted is not, the defect is the weighted
  # prefix-range compaction, not DCP-with-uneven-TP as such.
  local prev="${SGLANG_UNEVEN_DCP_WEIGHTED:-}"
  export SGLANG_UNEVEN_DCP_WEIGHTED=0
  _uneven31 uneven31_dcp_unweighted
  local rc=$?
  export SGLANG_UNEVEN_DCP_WEIGHTED="$prev"
  return $rc
}

arm_even_tp2() {
  # The NOISE FLOOR for the layer comparison, and it is not optional. Any TP=2
  # split re-associates every row-parallel reduction against TP=1, so a
  # per-layer diff of the uneven arm against TP=1 alone cannot say which part
  # of the delta is the bug: an EVEN TP=2 differs from TP=1 at the same
  # tensors. #340 showed the even split still produces byte-identical TOKENS,
  # so its layer deltas are, by construction, a harmless magnitude. The uneven
  # arm's finding is where its delta leaves that envelope.
  #
  # server_args.py rejects a per-rank MiB LIST without --rank-tp-ratio, so the
  # even form is --rank-gpu-id plus a single SCALAR budget -- which is what an
  # equal split wants anyway, and 14000 MiB fits both cards.
  run_arm even_tp2 \
    "tp2 even on cuda:$CUDA_BIG,$CUDA_SMALL0, mib $EVEN_MIB (scalar)" "-" \
    --tp-size 2 --rank-gpu-id "$CUDA_BIG,$CUDA_SMALL0" \
    --rank-gpu-memory-mib "$EVEN_MIB"
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
done

note "--- window finished after $(( $(date +%s) - T_START ))s of ${BUDGET_S}s ---"
exit 0
