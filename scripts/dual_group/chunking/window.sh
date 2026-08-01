#!/usr/bin/env bash
# #274 §13.10 -- ONE boot: chunked lane prefill, priced and gated.
#
# WHAT THIS WINDOW ANSWERS (measurement duties of the chunking posten):
#   1. ms/chunk against chunk size (512 / 1024 / 2048 -- all on the lane's
#      prefill tier keep-list, so each chunk replays a rung, no padding).
#   2. Prefill rate chunked vs single-forward, SAME BOOT (the reference
#      floor is re-collected here because the instrument's grain changed --
#      never compared against an older window's floors).
#   3. Coherence WITH the graded three-state gate: chunked trajectory vs a
#      reference SET of --ref-draws single-forward runs (GDN prefill is not
#      reproducible past ~109 tokens, so byte identity alone is not the
#      criterion; the reference set's own spread is the band).
#   4. Both spec arms: the chunked NEXTN head priming (§13.10 point 3) only
#      exists under spec, so "spec on" is the arm that can actually fail.
#
# ABORT RULES (r404 conventions):
#   - server not up in 20 min -> abort, keep the log, count the boot.
#   - a red STRUCTURE verdict is a BROKEN VEHICLE -> driver exits non-zero.
#   - a red COHERENCE verdict is NOT an abort. That is the measurement.
#   - VRAM corridor: free >= 400 MiB after captures, or no arms are run.
#
# TIME: 3 ref draws + 3 chunk arms, x2 spec modes, 1600-token prompt,
#   64 new tokens each = 12 lane jobs. Well under a 30-minute window on the
#   C3 operating point.

set -uo pipefail
WT="${WT:-/spinning/wt-274-lanespec}"
OWNER="${OWNER:-agent-274-chunking}"
export WT OWNER
source "$WT/scripts/dual_group/r7c/common.sh"

MODEL_ROOT="${MODEL_ROOT:-/spinning/llm_stuff/club-3090/models-cache}"
TARGET_DIR="$MODEL_ROOT/Qwen3.6-27B-MTP-Q3_K_M-GGUF"
TARGET="$TARGET_DIR/Qwen3.6-27B-Q3_K_M.gguf"
PORT="${PORT:-30086}"
OUT="${OUT:-/spinning/gpu-battery-results/$(date +%F)_274_chunking}"
LOG="${LOG:-/tmp/274-chunking.server.log}"
PIDFILE=/tmp/274-chunking.pid
LANE_BUDGET="${LANE_BUDGET:-700}"
RANK_MIB="${RANK_MIB:-21000,17780,17780}"
CHUNKS="${CHUNKS:-512,1024,2048}"
PROMPT_TOKENS="${PROMPT_TOKENS:-1600}"
REF_DRAWS="${REF_DRAWS:-3}"
DRIVER="$WT/scripts/dual_group/chunking/probe_arms.py"

mkdir -p "$OUT"

assert_cards_free || exit 1
load_card_order "$OUT/cards.txt" || exit 1
claim_cards "#274 chunking window: ms/chunk + coherence gate"
trap 'stop_vram_sampler; kill "$(cat $PIDFILE 2>/dev/null)" 2>/dev/null; release_cards "#274 chunking abgebrochen"; exit 1' INT TERM
start_vram_sampler "$OUT/vram.csv"

( while true; do touch "$ARB/holder" 2>/dev/null; sleep 300; done ) &
HB_PID=$!

cd "$WT" || exit 1
# The boot does NOT set --dual-group-lane-prefill-chunk: every arm comes
# from the per-job override, so chunked and unchunked share one boot, one
# capture set, one pool state. The server-side flag is exercised by the
# hermetic suite; the window measures.
launch_server "$LOG" "$PIDFILE" \
  "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$TARGET" \
  --tokenizer-path "$TARGET_DIR" \
  --tp-size 3 --rank-gpu-id 0,1,2 \
  --rank-tp-ratio 2,1,1 --rank-mlp-ratio 6,1,1 --rank-vocab-ratio 6,1,1 \
  --rank-gpu-memory-mib "$RANK_MIB" \
  --attention-backend flashinfer \
  --kv-cache-dtype fp8_e4m3 --context-length 16384 --trust-remote-code \
  --max-running-requests 4 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --dual-group-lane --dual-group-lane-budget-mib "$LANE_BUDGET" \
  --dual-group-lane-concurrent --dual-group-lane-admission-ms 2.0 \
  --dual-group-lane-spec \
  --enable-metrics \
  --host 127.0.0.1 --port "$PORT"

if dry_run; then
  kill $HB_PID 2>/dev/null
  stop_vram_sampler
  release_cards "#274 chunking dry run"
  exit 0
fi

"$VENV/bin/python" "$DRIVER" \
  --base "http://127.0.0.1:$PORT" \
  --tokenizer "$TARGET_DIR" \
  --chunks "$CHUNKS" \
  --prompt-tokens "$PROMPT_TOKENS" \
  --ref-draws "$REF_DRAWS" \
  --spec both \
  --out "$OUT/chunking_results.json"
DRIVER_RC=$?

kill "$(cat $PIDFILE 2>/dev/null)" 2>/dev/null
kill $HB_PID 2>/dev/null
stop_vram_sampler
summarise_vram "$OUT/vram.csv" || true
release_cards "#274 chunking window done rc=$DRIVER_RC"
exit $DRIVER_RC
