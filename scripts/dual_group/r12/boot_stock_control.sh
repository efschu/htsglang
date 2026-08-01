#!/usr/bin/env bash
# #274 / #284 -- the control arm the DIVERGENT verdict never had.
#
# QUESTION
#   #284 (b4a7def95c) re-ran the lane-spec coherence gate standalone with all
#   three A-vs-A floors green on BOTH sides and got DIVERGENT: the lane's
#   speculative trajectory leaves the lane's greedy one at alphabet index 7
#   and squares index 18, with both speculative runs agreeing with each other.
#   That self-consistency rules out noise and leaves exactly two worlds:
#
#     (A) the 2-row verify forward is not bit-identical to the 1-row decode
#         forward, so near-tie positions flip. Inherent, quality-neutral, and
#         present in ANY speculative decoder on this model.
#     (B) the lane's verify path computes something different. A defect.
#
#   Both worlds predict the observation. Only (B) needs the LANE to produce
#   it. This recipe therefore boots WITHOUT A LANE and asks whether stock
#   NEXTN speculation leaves the stock greedy trajectory on the same prompts.
#
# WHY TWO BOOTS AND NOT ONE
#   Speculation is a launch flag. There is no request-level way to turn it off
#   on a speculative boot (checked: no sampling-param or set_internal_state
#   path), so the greedy reference and the speculative arm are two boots by
#   construction. Both arms are self-checked with an A-vs-A floor inside their
#   own boot, and the runbook's own #139 note records that `topk 1` on this
#   model is byte-identical ACROSS boots -- so a cross-boot comparison is
#   legitimate here, and each arm says so with its own floor rather than
#   assuming it.
#
# VEHICLE  Qwen3.6-27B-MTP-Q3_K_M-GGUF, TP=3 uneven (5090 rank 0 + 2x 3080).
#   r8's corrected vehicle. NOT the FP8 checkpoint: 28.75 GiB of weights
#   against 31.34 GiB usable is not a lane vehicle on this rig, and while THIS
#   recipe has no lane, using a different checkpoint than the gate would put a
#   second variable inside the control.
#
# FEASIBILITY ARITHMETIC (before the first boot, per the #274 rule)
#   r8 booted this exact model and partition WITH the lane at
#   --rank-gpu-memory-mib 21000,17780,17780 and reported 2794 MiB free on the
#   5090 at peak and 12.3 GiB free on each 3080. This recipe REMOVES the lane:
#   -2684 MiB of head weights and -700 MiB of lane pool on rank 0, and the
#   concurrency graph pool / GGUF dequant workspace / flashinfer workspace
#   (+412 MiB in r8) with it. Keeping the same rank budget therefore leaves
#   rank 0 with ~3.4 GiB MORE headroom than a boot that is known to fit, and
#   the 3080s untouched. The freed MiB go to the KV pool, which changes
#   max_total_num_tokens and nothing about decode numerics.
#   The `spec` arm additionally carries the NEXTN draft head; that head is the
#   SAME 2684 MiB the lane's copy was measured at, so the spec arm sits at
#   roughly r8's own peak. Both arms are inside the >=400 MiB corridor with
#   margin, and the sampler records the real number either way.
#
# TIME  ~20 min per arm (load 10-14, probe 2, teardown 2, slack 2).
#       Run as: ARM=nospec ./boot_stock_control.sh ; ARM=spec ./boot_stock_control.sh
#
# ABORT
#   - server not up in 20 min -> abort, keep the log, count the boot
#   - an A-vs-A floor is not byte-identical -> the prompt is VOID and the
#     control carries no verdict on it. Not an abort: it is the finding.
#   - spec != nospec -> NOT an abort. That is the measurement.

set -uo pipefail
cd "$(dirname "$0")"
WT="${WT:-/spinning/wt-274-lane-spec}"
OWNER="${OWNER:-agent-274-r12}"
source ../r7c/common.sh

TARGET_DIR="$MODEL_ROOT/Qwen3.6-27B-MTP-Q3_K_M-GGUF"
TARGET="$TARGET_DIR/Qwen3.6-27B-Q3_K_M.gguf"
ARM="${ARM:-nospec}"
PORT="${PORT:-30082}"
LOG="${LOG:-/tmp/r12-stock-$ARM.server.log}"
OUT="${OUT:-/tmp/r12}"
RANK_MIB="${RANK_MIB:-21000,17780,17780}"
GATE_TOKENS="${GATE_TOKENS:-64}"
PROMPTS="${PROMPTS:-alphabet,squares}"
DRIVER_DEADLINE_S="${DRIVER_DEADLINE_S:-420}"

case "$ARM" in
  nospec) SPEC_FLAGS=() ;;
  spec)   SPEC_FLAGS=(--speculative-algorithm NEXTN --speculative-num-steps 3
                      --speculative-eagle-topk 1 --speculative-num-draft-tokens 4) ;;
  *) echo "ARM must be nospec or spec, got '$ARM'" >&2; exit 2 ;;
esac

mkdir -p "$OUT"
require_file "$TARGET" "GGUF target missing: $TARGET"

assert_cards_free || exit 1
load_card_order "$OUT/cards.txt" || exit 1
claim_cards "274-r12-stock-control-$ARM"
trap 'stop_vram_sampler; release_cards "r12 $ARM abgebrochen"; exit 1' INT TERM
start_vram_sampler "$OUT/vram-$ARM.csv"

cd "$WT" || exit 1
launch_server "$LOG" "/tmp/r12-stock-$ARM.pid" \
  "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$TARGET" \
  --tokenizer-path "$TARGET_DIR" \
  --tp-size 3 --rank-gpu-id 0,1,2 \
  --rank-tp-ratio 2,1,1 --rank-mlp-ratio 6,1,1 --rank-vocab-ratio 6,1,1 \
  --rank-gpu-memory-mib "$RANK_MIB" \
  --attention-backend flashinfer \
  --kv-cache-dtype fp8_e4m3 --context-length 16384 --trust-remote-code \
  --max-running-requests 4 \
  ${SPEC_FLAGS[@]+"${SPEC_FLAGS[@]}"} \
  --enable-metrics \
  --host 127.0.0.1 --port "$PORT"

if dry_run; then
  stop_vram_sampler
  release_cards "r12 dry run ($ARM)"
  echo "DRY RUN ok: r12 stock control $ARM"
  exit 0
fi

if ! wait_for_server "$PORT" 1200; then
  echo "--- letzte 40 Zeilen Serverlog (nur bei Abbruch) ---" >&2
  tail -40 "$LOG" >&2
  stop_vram_sampler
  release_cards "r12 $ARM: server nicht oben, verbraucht"
  exit 1
fi

"$VENV/bin/python" "$WT/scripts/dual_group/r12/stock_spec_control.py" \
  --port "$PORT" --tokenizer "$TARGET_DIR" --arm "$ARM" \
  --prompts "$PROMPTS" --tokens "$GATE_TOKENS" \
  --deadline-s "$DRIVER_DEADLINE_S" \
  --out "$OUT/$ARM.json" | tee "$OUT/$ARM.txt"
DRIVER_RC=${PIPESTATUS[0]}

kill "$(cat "/tmp/r12-stock-$ARM.pid")" 2>/dev/null
stop_vram_sampler
summarise_vram "$OUT/vram-$ARM.csv" | tee "$OUT/vram-$ARM-summary.txt"
release_cards "r12 $ARM DONE (driver rc=$DRIVER_RC), Rohdaten in $OUT"
echo "fertig: $OUT/$ARM.json (driver rc=$DRIVER_RC)"
exit "$DRIVER_RC"
