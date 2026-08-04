#!/usr/bin/env bash
# ARM 1 -- #478: the quant swap, UD-IQ3_XXS vs UD-Q3_K_XL.
#
#   ARM=iq3xxs  ./boot_478_quant.sh     # the active driver, 98 GiB
#   ARM=q3kxl   ./boot_478_quant.sh     # the #478 candidate, 120 GiB
#
# The two arms differ in exactly ONE thing: the model path. Everything else --
# env, flags, per-rank vectors, probes, prompt set -- is identical, and BOTH
# ARMS MUST RUN IN THE SAME WINDOW AT THE SAME POWER STATE. The user lowered
# every card's power target on 2026-08-03, so a comparison across power states
# is not a comparison; `power_tag` records the state at the start and end of
# each arm so that is verifiable after the fact rather than assumed.
#
# Base recipe verbatim from
# /spinning/gpu-battery-results/2026-08-02_394_linkshards/boot394.sh -- the
# only recipe that has ever served DSV4F on this rig. Nothing is invented:
# the additions are --reasoning-parser / --tool-call-parser / --chat-template
# (standing user order that every serving boot carries them) and nothing else.
#
# DESK-WRITTEN, NEVER EXECUTED. `bash -n` only.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
. "$HERE/lib.sh"

ARM_KIND="${ARM:-iq3xxs}"
PORT="${PORT:-30478}"
ARM="478_${ARM_KIND}"

case "$ARM_KIND" in
  iq3xxs)
    QUANT_DIR="$GGUF_ROOT/UD-IQ3_XXS"
    FIRST_SHARD="$QUANT_DIR/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf"
    # 98 GiB stream; launch->ready measured ~5.5-6 min on the prior window.
    READY_ITERS="${READY_ITERS:-90}"     # 15 min ceiling
    ;;
  q3kxl)
    QUANT_DIR="$GGUF_ROOT/UD-Q3_K_XL"
    FIRST_SHARD="$QUANT_DIR/DeepSeek-V4-Flash-0731-UD-Q3_K_XL-00001-of-00004.gguf"
    # 120 GiB measured with du, +22 GiB over IQ3_XXS. Scale the ceiling by the
    # size ratio and round up rather than reusing the IQ3 number.
    READY_ITERS="${READY_ITERS:-132}"    # 22 min ceiling
    ;;
  *)
    die "ARM must be iq3xxs or q3kxl, got '${ARM_KIND}'"
    ;;
esac

[ -f "$FIRST_SHARD" ] || die "first shard not found: $FIRST_SHARD"
# The loader auto-resolves all four shards from the FIRST shard path; do not
# enumerate them here.

log "=== ARM ${ARM} on port ${PORT} ==="
preflight "$ARM"
power_tag "$ARM" start
resolve_cards "$ARM"
assert_rank0_is_5090 "$RANK_GPU_ID"
assert_chat_template
rammon_start "$ARM"

export_base_env "$ARM"

BOOT_LOG="$RUN/boot_${ARM}.log"

# Quantization auto-detects as gguf and the attention backend resolves to
# dsv4 by default -- neither is passed, per the proven recipe.
BOOT_ARGS=(
  --model-path "$FIRST_SHARD"
  --tp-size 3 --rank-gpu-id "$RANK_GPU_ID" --rank-tp-ratio auto
  --rank-auto-reserve-mib "$AUTO_RESERVE_MIB"
  --rank-moe-resident-fraction "$RESIDENT_FRACTION"
  --kv-cache-dtype fp8_e4m3
  --context-length "$CONTEXT_LENGTH" --max-running-requests "$MAX_RUNNING"
  --chunked-prefill-size "$CHUNKED_PREFILL"
  --disable-cuda-graph
  --reasoning-parser "$REASONING_PARSER"
  --tool-call-parser "$TOOL_CALL_PARSER"
  --chat-template "$CHAT_TEMPLATE"
  --trust-remote-code --enable-metrics
  --host 127.0.0.1 --port "$PORT"
)
# EXTRA_ARGS: window-time levers (e.g. --max-total-tokens) that the desk recipe
# did not need. Kept as an explicit opt-in so the base recipe stays verbatim.
if [ -n "${EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  BOOT_ARGS+=( ${EXTRA_ARGS} )
  log "EXTRA_ARGS applied: ${EXTRA_ARGS}"
fi
assert_metrics_flag "${BOOT_ARGS[@]}"

log "launching: ${FIRST_SHARD}"
setsid "$PY" -u -m sglang.launch_server "${BOOT_ARGS[@]}" \
    > "$BOOT_LOG" 2>&1 < /dev/null &
record_pids "$ARM" $!

if ! wait_ready "$ARM" "$PORT" "$READY_ITERS"; then
    rammon_stop "$ARM"
    power_tag "$ARM" end
    stop_server "$ARM"
    die "arm ${ARM} never reached /health_generate -- see $BOOT_LOG and $RUN/pyspy_${ARM}.txt"
fi

# --- arm self-identification: an arm that fails its own check is reported
# --- failed, not repaired (TICKET_462 §2, the same discipline applies here).
assert_log_absent "$BOOT_LOG" "requires --disable-cuda-graph" \
    "that string means the boot died on a graph-config refusal"
log "offload trace lines: $(count_log "$BOOT_LOG" '[moe-staging-trace]')"

# --- probes. Order matters: the A-vs-A floor exists BEFORE any delta. -------
PROBE_ARGS=(--port "$PORT" --arm "$ARM" --run "$RUN" --window-seconds "${WINDOW_SECONDS:-15}")

"$PY" "$HERE/probes.py" --selftest || die "the probe instruments failed their own can-discriminate check; no number from this arm counts"
"$PY" "$HERE/probes.py" avsa        "${PROBE_ARGS[@]}"
"$PY" "$HERE/probes.py" chatprobe   "${PROBE_ARGS[@]}" --model "$(basename "$QUANT_DIR")"
"$PY" "$HERE/probes.py" prefill     "${PROBE_ARGS[@]}"
"$PY" "$HERE/probes.py" decode      "${PROBE_ARGS[@]}"
"$PY" "$HERE/probes.py" determined  "${PROBE_ARGS[@]}" --model "$(basename "$QUANT_DIR")"

# expert_stats is armed in EVERY arm (free), so arm 4 is harvested here rather
# than costing its own boot. Capture the dump BEFORE teardown -- the SIGTERM
# revision left on disk is not the headline artifact.
cp -a "$RUN/expert_stats_${ARM}"* "$RUN/" 2>/dev/null || true
for f in "$RUN/expert_stats_${ARM}"*; do
    [ -e "$f" ] && cp -a "$f" "${f}.preteardown"
done
log "expert stats captured pre-teardown"

rammon_stop "$ARM"
power_tag "$ARM" end
stop_server "$ARM"

log "=== ARM ${ARM} complete. Artifacts in $RUN ==="
log "Both #478 arms must run in THIS window at THIS power state; compare only"
log "powerstate_478_iq3xxs.json against powerstate_478_q3kxl.json first."
