#!/usr/bin/env bash
# ARM 3 -- TICKET #462: the breakable route + the #494 break-cost instrument.
#
# TICKET_462's §-ORDER IS MANDATORY and this script enforces it:
#   §3  F2 first, probe ON, plus a clean eager control      -> ARM=eager, ARM=f2
#   §4  replay-without-recapture gate, only if F2 leaves it worth running
#   §5  ms/verify A/B with the probe OFF                    -> ARM=breakable_clean
#
#   ARM=eager           ./boot_462_f2.sh   # the control. Must run.
#   ARM=f2              ./boot_462_f2.sh   # breakable + #494 probe ON (§3, §4)
#   ARM=breakable_clean ./boot_462_f2.sh   # breakable, probe OFF (§5 A/B)
#
# ROUTE ON = env AND flags, both:
#   SGLANG_MOE_OFFLOAD_GRAPH_MODE=breakable
#   --cuda-graph-backend-decode=breakable --cuda-graph-backend-prefill=disabled
# validate_breakable_boot (layers/moe/offload_capture_gate.py:311-420) refuses
# the boot if the resolved decode backend is not 'breakable' or prefill is not
# disabled/None. So this arm does NOT pass --disable-cuda-graph -- that is the
# one place it deliberately departs from the base recipe, and it is the exact
# mechanism under test.
#
# NEVER set SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1 or GRAPH_MODE=capturable: both are
# refused by name as the REFUTED path (measured 6.60x slower than eager).
# export_base_env() unsets them defensively on every arm.
#
# DESK-WRITTEN, NEVER EXECUTED. `bash -n` only.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
. "$HERE/lib.sh"

ARM_KIND="${ARM:-eager}"
PORT="${PORT:-30462}"
ARM="462_${ARM_KIND}"
READY_ITERS="${READY_ITERS:-90}"

QUANT_DIR="$GGUF_ROOT/UD-IQ3_XXS"
FIRST_SHARD="$QUANT_DIR/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf"
[ -f "$FIRST_SHARD" ] || die "first shard not found: $FIRST_SHARD"

# Expected crossings per captured decode step on DSV4F. A different number
# means the arm is not the one you think it is (TICKET_462 §3).
EXPECT_CROSSINGS="${EXPECT_CROSSINGS:-43}"

case "$ARM_KIND" in
  eager)
    # --disable-cuda-graph, NOT --cuda-graph-backend-*=disabled. The offload
    # path's own guard demands that exact flag by name and refused the boot:
    # "MoE expert-offload / routing-trace ... requires --disable-cuda-graph".
    # It is also what the proven base recipe uses, so the control arm is the
    # recipe unmodified -- which is what a control should be.
    GRAPH_FLAGS=(--disable-cuda-graph)
    ROUTE_ON=0; PROBE_ON=0
    ;;
  f2)
    GRAPH_FLAGS=(--cuda-graph-backend-decode=breakable --cuda-graph-backend-prefill=disabled)
    ROUTE_ON=1; PROBE_ON=1
    [ -e "$RUN/probes_462_eager_all.json" ] || die "the F2 arm refuses without its clean eager control. TICKET_462 §3 wants F2 measured against the control from the same window; run ARM=eager first."
    ;;
  breakable_clean)
    GRAPH_FLAGS=(--cuda-graph-backend-decode=breakable --cuda-graph-backend-prefill=disabled)
    ROUTE_ON=1; PROBE_ON=0
    [ -e "$RUN/F2_break_cost.txt" ] || die "§5 refuses before §3. TICKET_462: 'If F2 says the breaks cost more than the graph saves, steps 2 and 3 are not worth their slot.' Run ARM=f2 and read $RUN/F2_break_cost.txt first."
    ;;
  *)
    die "ARM must be eager, f2 or breakable_clean, got '${ARM_KIND}'"
    ;;
esac

log "=== ARM ${ARM} on port ${PORT} (route=${ROUTE_ON}, probe=${PROBE_ON}) ==="
preflight "$ARM"
power_tag "$ARM" start
resolve_cards "$ARM"
assert_rank0_is_5090 "$RANK_GPU_ID"
assert_chat_template
rammon_start "$ARM"

export_base_env "$ARM"
export SGLANG_MOE_HOST_SHARD_RATIO=1,1,1     # TICKET_462 §1

BOOT_LOG="$RUN/boot_${ARM}.log"

BOOT_ARGS=(
  --model-path "$FIRST_SHARD"
  --tp-size 3 --rank-gpu-id "$RANK_GPU_ID" --rank-tp-ratio auto
  --rank-auto-reserve-mib "$AUTO_RESERVE_MIB"
  --rank-moe-resident-fraction "$RESIDENT_FRACTION"
  --kv-cache-dtype fp8_e4m3
  --context-length "$CONTEXT_LENGTH" --max-running-requests "$MAX_RUNNING"
  --chunked-prefill-size "$CHUNKED_PREFILL"
  "${GRAPH_FLAGS[@]}"
  --reasoning-parser "$REASONING_PARSER"
  --tool-call-parser "$TOOL_CALL_PARSER"
  --chat-template "$CHAT_TEMPLATE"
  --trust-remote-code --enable-metrics
  --host 127.0.0.1 --port "$PORT"
)

if [ "$ROUTE_ON" = "1" ]; then
    export SGLANG_MOE_OFFLOAD_GRAPH_MODE=breakable
    # Scratch sizing, TICKET_462 §1: the bound is
    #   min(max_captured_bs x top_k, E_local - R)
    # and it counts graph-PADDED rows, which carry real routed ids. top_k = 6;
    # E_local 114/71/71, R 56/30/30 at this geometry, so the cold set is
    # 58/41/41. At --cuda-graph-bs-decode 1, C = 6 -- which is exactly the
    # SGLANG_MOE_SCRATCH_SLOTS=6 the base env already sets. Raising the
    # captured bs raises C and therefore resident VRAM: a corridor decision,
    # not a free knob. Undersizing surfaces as a named BreakableScratchOverflow,
    # never as a wrong answer.
    BOOT_ARGS+=(--cuda-graph-bs-decode "${CAPTURED_BS:-1}")
else
    unset SGLANG_MOE_OFFLOAD_GRAPH_MODE || true
fi

if [ "$PROBE_ON" = "1" ]; then
    # #494 instrument. SGLANG_BREAK_COST_PATH is DELIBERATELY LEFT UNSET.
    #
    # CONTRADICTION, resolved in favour of the code: both the briefing
    # ('SGLANG_BREAK_COST_PATH="$RUN/break_cost"') and TICKET_462 §3
    # ('"$RUN/break_cost.jsonl" # becomes one file per rank') assume the path
    # is expanded per rank. It is NOT. break_cost_clock.py:513 reads
    #     path = os.environ.get(ENV_PATH) or f"/tmp/break_cost.{rank_tag}.jsonl"
    # and uses the value VERBATIM -- no rank tag is interpolated into a
    # user-supplied path. Setting it would make all three TP ranks append to
    # ONE file, and the ticket's own readout glob
    # ("$RUN"/break_cost.rank*.jsonl) would then match nothing.
    #
    # Leaving it unset gives the documented per-rank default, which is the
    # shape everything downstream expects; the files are copied into $RUN
    # after the run so the artifact still survives with the window.
    unset SGLANG_BREAK_COST_PATH || true
    export SGLANG_BREAK_COST_PROBE=1
    export SGLANG_BREAK_COST_DEFER_ROUNDS="${SGLANG_BREAK_COST_DEFER_ROUNDS:-2}"
    export SGLANG_BREAK_COST_WARMUP_ROUNDS="${SGLANG_BREAK_COST_WARMUP_ROUNDS:-20}"
    export SGLANG_BREAK_COST_DETAIL="${SGLANG_BREAK_COST_DETAIL:-1}"
    # Stale records from an earlier run would silently pollute the table.
    rm -f /tmp/break_cost.rank*.jsonl
    # Second, independent count of the 43 crossings: DEBUG on exactly one
    # module. See logging_break_debug.json for why this is not --log-level debug.
    export SGLANG_LOGGING_CONFIG_PATH="${SGLANG_LOGGING_CONFIG_PATH:-$HERE/logging_break_debug.json}"
    log "break-cost probe ARMED; per-rank files land in /tmp and are copied to $RUN"
else
    unset SGLANG_BREAK_COST_PROBE SGLANG_BREAK_COST_PATH SGLANG_LOGGING_CONFIG_PATH || true
    log "break-cost probe OFF (its per-round harvest cost, probe_sink_ms, must not sit inside a §5 A/B number)"
fi

assert_metrics_flag "${BOOT_ARGS[@]}"

log "launching ${ARM}"
setsid "$PY" -u -m sglang.launch_server "${BOOT_ARGS[@]}" \
    > "$BOOT_LOG" 2>&1 < /dev/null &
record_pids "$ARM" $!

if ! wait_ready "$ARM" "$PORT" "$READY_ITERS"; then
    rammon_stop "$ARM"; power_tag "$ARM" end; stop_server "$ARM"
    die "arm ${ARM} never reached /health_generate -- see $BOOT_LOG and $RUN/pyspy_${ARM}.txt"
fi

# ------------------------------------------------------------------------
# §2 -- arm self-identification. RUN BEFORE QUOTING ANY NUMBER.
# An arm that fails its own check is reported failed, not repaired.
# ------------------------------------------------------------------------
SELFID="$RUN/selfid_${ARM}.txt"
{
    printf 'arm=%s route_on=%s probe_on=%s utc=%s\n' "$ARM" "$ROUTE_ON" "$PROBE_ON" "$(utc)"
    printf 'cuda graph True   : %s\n' "$(count_log "$BOOT_LOG" 'cuda graph: True')"
    printf 'cuda graph False  : %s\n' "$(count_log "$BOOT_LOG" 'cuda graph: False')"
    printf 'capture unsupported: %s\n' "$(count_log "$BOOT_LOG" 'cudaErrorStreamCaptureUnsupported')"
    printf 'capture invalidated: %s\n' "$(count_log "$BOOT_LOG" 'cudaErrorStreamCaptureInvalidated')"
    printf 'REFUTED mentions  : %s\n' "$(count_log "$BOOT_LOG" 'REFUTED')"
    printf 'UNSAFE mentions   : %s\n' "$(count_log "$BOOT_LOG" 'SGLANG_MOE_OFFLOAD_CUDA_GRAPH_UNSAFE')"
    printf 'moe-staging-trace : %s\n' "$(count_log "$BOOT_LOG" '[moe-staging-trace]')"
    printf 'break-graph DEBUG : %s\n' "$(count_log "$BOOT_LOG" 'Break graph due to function: _moe_offload_fetch_step')"
    printf -- '--- resolved cuda_graph_config (the only post-cascade view) ---\n'
    grep -o 'cuda_graph_config=CudaGraphConfig([^)]*)[^)]*)' "$BOOT_LOG" | head -3
} | tee "$SELFID"

# These are hard: they say the boot is not the arm it claims to be.
assert_log_absent "$BOOT_LOG" "requires --disable-cuda-graph" \
    "that string means the boot died; the breakable route is not running"
assert_log_absent "$BOOT_LOG" "cudaErrorStreamCaptureUnsupported" \
    "a capture error voids the arm"
assert_log_absent "$BOOT_LOG" "cudaErrorStreamCaptureInvalidated" \
    "a capture error voids the arm"
assert_log_absent "$BOOT_LOG" "SGLANG_MOE_OFFLOAD_CUDA_GRAPH_UNSAFE" \
    "the REFUTED path must not be involved"
assert_log_contains "$BOOT_LOG" "[moe-staging-trace]" \
    "expert offload is not ON, so there is nothing for this route to be about"

GRAPH_TRUE="$(count_log "$BOOT_LOG" 'cuda graph: True')"
if [ "$ROUTE_ON" = "1" ]; then
    [ "${GRAPH_TRUE:-0}" -ge 1 ] || die "breakable arm shows 0 'cuda graph: True' decode-batch lines -- it never replayed a graph, so it is an eager run wearing the breakable label."
else
    [ "${GRAPH_TRUE:-0}" -eq 0 ] || die "the eager CONTROL replayed a graph (${GRAPH_TRUE} 'cuda graph: True' lines). It is not a control."
fi

# ------------------------------------------------------------------------
# Probes. §5's A/B is only quotable from the probe-OFF arms.
# ------------------------------------------------------------------------
PROBE_ARGS=(--port "$PORT" --arm "$ARM" --run "$RUN" --window-seconds "${WINDOW_SECONDS:-15}")
"$PY" "$HERE/probes.py" --selftest || die "the probe instruments failed their own can-discriminate check; no number from this arm counts"
"$PY" "$HERE/probes.py" all "${PROBE_ARGS[@]}" --model "$(basename "$QUANT_DIR")"

# §4.4 -- correctness, same window: greedy prompt, 3 runs, breakable vs eager.
# Each arm must FIRST be internally deterministic (3 identical hashes), then
# the two arms are compared. This is B2 re-asked for THIS route: #452's B2
# divergence was never localised, so a divergence here is not automatically
# this route's fault -- but it IS a stop-and-report.
if [ "$ROUTE_ON" = "0" ]; then
    "$PY" "$HERE/probes.py" idem-record "${PROBE_ARGS[@]}"
else
    "$PY" "$HERE/probes.py" idem-compare "${PROBE_ARGS[@]}" \
        --reference "$RUN/idem_reference_462_eager.json"
fi

# ------------------------------------------------------------------------
# §3 readout -- the left-hand side of the verdict.
# ------------------------------------------------------------------------
if [ "$PROBE_ON" = "1" ]; then
    cp -a /tmp/break_cost.rank*.jsonl "$RUN/" 2>/dev/null \
        || log "WARNING: no /tmp/break_cost.rank*.jsonl to copy -- the probe never armed, or no captured decode step ran"
    if ls "$RUN"/break_cost.rank*.jsonl >/dev/null 2>&1; then
        "$PY" "$WT/scripts/dev/494_break_cost/summarise.py" \
            --drop-rounds "${SGLANG_BREAK_COST_WARMUP_ROUNDS:-20}" \
            "$RUN"/break_cost.rank*.jsonl | tee "$RUN/F2_break_cost.txt"

        # Assert crossings/round == 43, and cross-check it against the
        # INDEPENDENT DEBUG count. Two counts of the same event; a mismatch
        # means one of the two instruments is not seeing what it claims to.
        "$PY" - "$RUN/F2_break_cost.txt" "$EXPECT_CROSSINGS" <<'PYEOF' | tee -a "$RUN/F2_break_cost.txt"
import re, sys
text = open(sys.argv[1]).read()
want = float(sys.argv[2])
found = [float(m) for m in re.findall(r"crossings/round\s*:\s*([0-9.]+)", text)]
if not found:
    print(f"CROSSINGS CHECK: no 'crossings/round' line in the summary -- the arm produced no records")
    raise SystemExit(1)
bad = [v for v in found if abs(v - want) > 0.5]
print(f"CROSSINGS CHECK: per-rank crossings/round = {found}, expected {want}")
if bad:
    print(
        f"CROSSINGS CHECK FAILED: {bad} != {want}. TICKET_462 §3: 'a different "
        f"number means the arm is not the one you think it is'. A rank showing "
        f"0 is the pass-through case and INVALIDATES the arm."
    )
    raise SystemExit(2)
print("CROSSINGS CHECK OK")
PYEOF
        DEBUG_BREAKS="$(count_log "$BOOT_LOG" 'Break graph due to function: _moe_offload_fetch_step')"
        printf 'INDEPENDENT DEBUG COUNT: %s occurrences of "Break graph due to function: _moe_offload_fetch_step"\n' \
            "${DEBUG_BREAKS:-0}" | tee -a "$RUN/F2_break_cost.txt"
        if [ "${DEBUG_BREAKS:-0}" -eq 0 ]; then
            log "WARNING: the DEBUG cross-check counted 0 breaks. Either SGLANG_LOGGING_CONFIG_PATH did not take, or the break never fired. The summariser's count alone is ONE instrument, not two -- say so in the report."
        fi
    else
        log "WARNING: no break-cost records at all. F2 has no left-hand side; §4 and §5 must not be run on this arm."
    fi

    # --- §4 replay-without-recapture gate -----------------------------------
    # Only observable at the graph level, so it is harvested here rather than
    # asserted from a hermetic test.
    {
        printf 'replay-without-recapture evidence for %s\n' "$ARM"
        printf 'capture events    : %s\n' "$(count_log "$BOOT_LOG" 'Capture cuda graph')"
        printf 'cuda graph True   : %s\n' "$(count_log "$BOOT_LOG" 'cuda graph: True')"
        printf '\n-- residency.fetches / h2d_bytes over time (must rise MONOTONICALLY;\n'
        printf -- '-- a FLAT fetch counter under rising steps means the eager phase is NOT\n'
        printf -- '-- re-running and the graph is replaying STALE slots: the single most\n'
        printf -- '-- dangerous failure mode of this design.\n'
        grep -oE 'residency\.(fetches|h2d_bytes)[=: ]+[0-9]+' "$BOOT_LOG" | tail -60
    } > "$RUN/F2_replay_gate_${ARM}.txt"
    log "§4 replay evidence -> $RUN/F2_replay_gate_${ARM}.txt (read it; it is not auto-judged)"
fi

grep -F "Decode batch" "$BOOT_LOG" > "$RUN/decode_ticks_${ARM}.txt" 2>/dev/null || true

for f in "$RUN/expert_stats_${ARM}"*; do
    [ -e "$f" ] && cp -a "$f" "${f}.preteardown"
done

rammon_stop "$ARM"
power_tag "$ARM" end
stop_server "$ARM"

log "=== ARM ${ARM} complete ==="
log "VERDICT RULE (TICKET_462 §3): 43 x (break + rendezvous + planning + publish)"
log "against the launch-overhead saving the graph buys. Report both numbers and"
log "the ratio. There is NO kill threshold -- Aufwand/Ertrag decides, and a"
log "small win that is cheap to keep is still a win."
log "Do NOT quote F1's 5.3-8.4x: that is a CEILING measured on Qwen3.6-35B-A3B."
log "The 43 rendezvous/step are irreducible on this route (DESIGN_462 §4); their"
log "removal is not an achievable optimisation."
