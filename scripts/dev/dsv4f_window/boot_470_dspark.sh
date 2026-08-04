#!/usr/bin/env bash
# ARM 2 -- TICKET #470: the DSpark solo arm. A TWO-BOOT GATE.
#
#   SUBARM=a_base   ./boot_470_dspark.sh   # baseline residency, no draft
#   SUBARM=a_cut    ./boot_470_dspark.sh   # residency cut by ~11 GiB, no draft
#   SUBARM=b_dspark ./boot_470_dspark.sh   # the DSpark arm, same cut residency
#
# ORDERING IS ENFORCED, not documented. TICKET_470 §5, verbatim:
#   "If Boot A cannot be run at all, do not run Boot B: an unattributed
#    multiplier is not a result."
# b_dspark REFUSES to launch unless both a_base and a_cut left their artifacts
# in $RUN.
#
# ------------------------------------------------------------------------
# DEVIATION FROM THE TICKET, stated rather than smoothed over
# ------------------------------------------------------------------------
# TICKET_470 §2 asks for the residency cut to be measured in the SAME boot as
# the baseline ("Same boot: reduce rank 0's resident budget by ~11 GiB and
# measure again"). That is not expressible on this build: the resident set is
# fixed by --rank-moe-resident-fraction, a launch-time ServerArgs field, and
# there is no runtime endpoint that changes it (searched http_server.py and
# srt/managers/ for any resident/moe update route -- none exists).
#
# The ticket anticipates exactly this and says what to do:
#   "If the cut cannot be expressed as a budget knob on this build, say so and
#    price it by the closest available lever rather than skipping the boot:
#    without Boot A, Boot B's multiplier is unattributable."
#
# So Boot A is split into two boots, a_base and a_cut, and EACH carries its
# OWN same-boot A-vs-A floor (the §5 discipline: every point carries its own
# floor). The cut is then priced as a boot-to-boot delta gated on the larger
# of the two floors, which is strictly more conservative than a within-boot
# delta would have been.
#
# DESK-WRITTEN, NEVER EXECUTED. `bash -n` only.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
. "$HERE/lib.sh"

SUBARM="${SUBARM:-a_base}"
PORT="${PORT:-30470}"
ARM="470_${SUBARM}"
READY_ITERS="${READY_ITERS:-90}"

QUANT_DIR="$GGUF_ROOT/UD-IQ3_XXS"
FIRST_SHARD="$QUANT_DIR/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00001-of-00004.gguf"
[ -f "$FIRST_SHARD" ] || die "first shard not found: $FIRST_SHARD"

# The residency cut. NOW MEASURED (window 2026-08-04, Boot A --
# TICKET_470_RESULT_first_boot.md §1): 0.485 -> 0.23 on rank 0 frees 10.21 GiB
# against a DSpark head that needs 10.12 GiB, and costs ~1.3-1.4 % of decode
# ms/round. Ranks 1 and 2 stay at 0.42 -- the cut is asymmetric by design, and
# a scalar here would silently cut them too and invalidate the comparison
# against a_cut.
#
# The previous default, 0.383, was ARITHMETIC FROM THE TICKET (0.485 x 0.79,
# ANALYSE_463 §4.4; TICKET_470 §7.6 flagged it as unmeasured). Measurement put
# rank 0's resident set ~5 GiB below the desk model, so 0.383 frees only
# ~4.1 GiB of the 10.12 GiB needed and OOMs rank 0 partway through the draft
# build. It is replaced rather than kept as a fallback: a value that cannot
# boot is not a safe default.
RESIDENT_FRACTION_CUT="${RESIDENT_FRACTION_CUT:-0.23,0.42,0.42}"

case "$SUBARM" in
  a_base)
    USE_RESIDENT="$RESIDENT_FRACTION"
    WITH_DRAFT=0
    ;;
  a_cut)
    USE_RESIDENT="$RESIDENT_FRACTION_CUT"
    WITH_DRAFT=0
    ;;
  b_dspark)
    USE_RESIDENT="$RESIDENT_FRACTION_CUT"
    WITH_DRAFT=1
    for need in "$RUN/probes_470_a_base_all.json" "$RUN/probes_470_a_cut_all.json"; do
        [ -e "$need" ] || die "Boot B refuses: ${need} is missing. TICKET_470 §5 -- 'If Boot A cannot be run at all, do not run Boot B: an unattributed multiplier is not a result.' Run SUBARM=a_base and SUBARM=a_cut first."
    done
    [ -e "$RUN/idem_reference_470_a_cut.json" ] || die "Boot B refuses: no greedy reference from a_cut ($RUN/idem_reference_470_a_cut.json). The ANALYSE_447 §2.4 idempotence question is answered by comparing the draft arm against the SAME-residency no-draft greedy output; without the reference the correctness question cannot be answered, and it outranks the perf numbers."
    ;;
  *)
    die "SUBARM must be a_base, a_cut or b_dspark, got '${SUBARM}'"
    ;;
esac

log "=== ARM ${ARM} on port ${PORT} (resident=${USE_RESIDENT}, draft=${WITH_DRAFT}) ==="
preflight "$ARM"
power_tag "$ARM" start
resolve_cards "$ARM"
assert_rank0_is_5090 "$RANK_GPU_ID"
assert_chat_template
rammon_start "$ARM"

export_base_env "$ARM"

# SGLANG_DSV4_FP4_DEQUANT must be unset/0. At 1 it asserts against a non-auto
# runner backend and inflates the head to 18.6 GiB (TICKET_470 §3).
unset SGLANG_DSV4_FP4_DEQUANT || true
export SGLANG_DSV4_FP4_DEQUANT=0

BOOT_LOG="$RUN/boot_${ARM}.log"

BOOT_ARGS=(
  --model-path "$FIRST_SHARD"
  --tp-size 3 --rank-gpu-id "$RANK_GPU_ID" --rank-tp-ratio auto
  --rank-auto-reserve-mib "$AUTO_RESERVE_MIB"
  --rank-moe-resident-fraction "$USE_RESIDENT"
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

if [ "$WITH_DRAFT" = "1" ]; then
    [ -d "$DSPARK_HEAD" ] || die "DSpark head not found: $DSPARK_HEAD"

    # --speculative-draft-gpu takes a CUDA ORDINAL, not an NVML index.
    # server_args.py:3581-3589 verbatim: "CUDA device index (torch.cuda order,
    # same space as --rank-gpu-id)". TICKET_470 §3 and the window briefing
    # both say "NVML index of the 5090" -- that is WRONG on this rig, where
    # the 5090 is CUDA 0 / NVML 1. Passing the NVML index would NOT error
    # (rank 1 maps to cuda:1 legitimately); it would silently put the solo
    # draft head on a 3080, where the MXFP4 Marlin path does not exist
    # (SM90/SM120 only). Silent wrongness, so it is asserted.
    DRAFT_GPU="${DRAFT_GPU:-$CARD_5090_CUDA}"
    assert_draft_gpu_is_5090 "$DRAFT_GPU"
    log "draft-gpu = CUDA ordinal ${DRAFT_GPU} (5090; its NVML index is ${CARD_5090_NVML} and is NOT what this flag takes)"

    BOOT_ARGS+=(
      --speculative-algorithm DSPARK
      --speculative-draft-model-path "$DSPARK_HEAD"
      --speculative-draft-placement solo
      --speculative-draft-gpu "$DRAFT_GPU"
      --speculative-moe-runner-backend marlin
      # The draft head is safetensors; the TARGET is GGUF. Unset, this flag
      # inherits --load-format (server_args.py:3268-3273 says so), so the draft
      # loader was handed a directory while expecting a single .gguf file and
      # refused with "... is not a file." Name the draft's own format.
      --speculative-draft-load-format "${DRAFT_LOAD_FORMAT:-auto}"
      --speculative-dspark-block-size 5
      --speculative-num-draft-tokens 6
      --speculative-num-steps 1
      --speculative-eagle-topk 1
    )
    READY_ITERS="${READY_ITERS_B:-108}"   # 18 min: target stream + draft head
fi

assert_metrics_flag "${BOOT_ARGS[@]}"

log "launching ${ARM}"
setsid "$PY" -u -m sglang.launch_server "${BOOT_ARGS[@]}" \
    > "$BOOT_LOG" 2>&1 < /dev/null &
record_pids "$ARM" $!

if ! wait_ready "$ARM" "$PORT" "$READY_ITERS"; then
    rammon_stop "$ARM"
    power_tag "$ARM" end
    stop_server "$ARM"      # py-spy dump happens inside, BEFORE any signal
    die "arm ${ARM} never reached /health_generate. A hang here on b_dspark is most likely a missing shadow participant in one of the four round collectives (embed all_reduce, hidden broadcast, vocab all_gather, round payload) -- read $RUN/pyspy_${ARM}.txt before anything else."
fi

# ------------------------------------------------------------------------
# §3.1 first-boot checks, cheap before expensive. An arm that fails its own
# check is reported failed, not repaired.
# ------------------------------------------------------------------------
if [ "$WITH_DRAFT" = "1" ]; then
    # 1 -- solo placement took. The shadow line is the one whose exact text is
    # verified in this tree (model_runner.py:544, "Draft-solo placement: rank
    # %d is a draft SHADOW"). The HOST half of the sentence is asserted with
    # the shorter, safer substring so a wording drift does not fail the arm
    # for the wrong reason.
    assert_log_contains "$BOOT_LOG" "Draft-solo placement:" \
        "solo placement never announced itself -- --speculative-draft-placement solo did not take"
    assert_log_contains "$BOOT_LOG" "is a draft SHADOW" \
        "no rank reported itself a shadow, so the draft was not placed solo (model_runner.py:544)"

    # 2 -- the markov_w2 TP-shard optimisation must be off under solo.
    if ! grep -qiE "markov_w2.*(disabled|disabling|off)|SGLANG_DSPARK_OPT_MARKOV_W2_TP_SHARD" "$BOOT_LOG"; then
        log "WARNING: §3.1 check 2 (markov_w2 TP-shard disabled under solo) found no matching log line. NOT failing the arm on it -- the exact wording of that line is not pinned anywhere in this tree, so its absence may be a log-string mismatch rather than a wiring failure. Read the log and record which it was."
    fi

    # 3 -- the marlin runner backend actually reached the draft's expert
    # layers. Exact string verified at
    # python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py:133.
    assert_log_contains "$BOOT_LOG" "Preparing MXFP4 experts for Marlin backend" \
        "--speculative-moe-runner-backend marlin did NOT reach the draft build (the draft_worker_common.py flag wiring did not take). Everything after this would measure the wrong kernel, so the arm is void."

    # The refuted graph paths must not be anywhere near this boot.
    assert_log_absent "$BOOT_LOG" "SGLANG_MOE_OFFLOAD_CUDA_GRAPH_UNSAFE" \
        "the REFUTED capturable path must not be involved"
fi

# The Decode-batch tick lines carry the interval accept length. TICKET_462 §5:
# take the accept length from THIS line, not the EMA printed beside it.
grep -F "Decode batch" "$BOOT_LOG" > "$RUN/decode_ticks_${ARM}.txt" 2>/dev/null || true
log "decode tick lines harvested: $(wc -l < "$RUN/decode_ticks_${ARM}.txt" 2>/dev/null || echo 0)"

# ------------------------------------------------------------------------
# Probes. Greedy only: solo placement REFUSES non-greedy rounds by name. That
# is the v1 limit and is not worked around.
# ------------------------------------------------------------------------
PROBE_ARGS=(--port "$PORT" --arm "$ARM" --run "$RUN" --window-seconds "${WINDOW_SECONDS:-15}")

"$PY" "$HERE/probes.py" --selftest || die "the probe instruments failed their own can-discriminate check; no number from this arm counts"
"$PY" "$HERE/probes.py" all "${PROBE_ARGS[@]}" --model "$(basename "$QUANT_DIR")"

# §3.2 -- the correctness question, in the same window, outranking perf.
if [ "$WITH_DRAFT" = "1" ]; then
    "$PY" "$HERE/probes.py" idem-compare "${PROBE_ARGS[@]}" \
        --reference "$RUN/idem_reference_470_a_cut.json"
    log "ANALYSE_447 §2.4 answered behaviourally above. The static half is a read of"
    log "  python/sglang/srt/layers/attention/dsv4/compressor_v2.py:516-596"
    log "  (forward_unified writes state_pool.kv_score_buffer.kv_score and, when"
    log "   online_c128_mtp is present, write_prefix_states) -- do that read and put"
    log "  both halves in the report. A DIVERGENCE verdict is a STOP AND REPORT."
else
    "$PY" "$HERE/probes.py" idem-record "${PROBE_ARGS[@]}"
fi

for f in "$RUN/expert_stats_${ARM}"*; do
    [ -e "$f" ] && cp -a "$f" "${f}.preteardown"
done

rammon_stop "$ARM"
power_tag "$ARM" end
stop_server "$ARM"

log "=== ARM ${ARM} complete ==="
if [ "$SUBARM" = "a_cut" ]; then
    log "Boot A deliverable: ONE number -- the ms/round cost of making room for"
    log "the head, a_cut vs a_base, gated on the LARGER of the two arms' own"
    log "A-vs-A floors. Everything after is measured against it."
fi
if [ "$SUBARM" = "b_dspark" ]; then
    log "Reference band 0.49-0.77 accept (llama.cpp PR #25784) is THEIR domains,"
    log "order of magnitude only -- never a 1:1 comparison. Below ~0.45 on a"
    log "comparable mix means the block/Markov chaining is wrong, not merely slow."
fi
