#!/usr/bin/env bash
# #274 slice D -- ONE card window: pairing-policy A/B + r8 E re-measure (#328-1).
#
# QUESTION
#   Does pairing saturating lane grains away from saturating serving grains
#   move E?  Both arms come from ONE boot: the policy is flipped through
#   set_internal_state, so they share floors, captures and content.  The same
#   boot re-takes r8's four E windows with the #284-repaired window reader
#   (their shared lane rates carried the drain tail and were over-estimates).
#
# VEHICLE  Qwen3.6-27B-MTP-Q3_K_M-GGUF, TP=3 uneven (5090 rank 0 + 2x 3080),
#   the r8/r9 recipe unchanged except that this worktree carries the pairing
#   policy (OFF at boot -- the regression posture; the driver flips it).
#
# TIME WINDOW  ~25 min (load ~10-12, pairing 6 windows + r8redo 5 windows at
#   30/45 s + drains ~9, teardown/summary ~3)
# ABORT  server not up in 20 min -> abort, keep the log, count the boot.

set -uo pipefail
cd "$(dirname "$0")"
WT="${WT:-/spinning/wt-slice-d}"
export WT
source ../r7c/common.sh

TARGET_DIR="$MODEL_ROOT/Qwen3.6-27B-MTP-Q3_K_M-GGUF"
TARGET="$TARGET_DIR/Qwen3.6-27B-Q3_K_M.gguf"
PORT="${PORT:-30086}"
LOG="${LOG:-/tmp/slice-d-pairing.server.log}"
OUT="${OUT:-/tmp/slice-d-pairing}"
LANE_BUDGET="${LANE_BUDGET:-700}"
RANK_MIB="${RANK_MIB:-21000,17780,17780}"
WINDOW_S="${WINDOW_S:-30}"
R8_WINDOW_S="${R8_WINDOW_S:-45}"
STEPS="${STEPS:-1}"
PHASES="${PHASES:-pairing,r8redo}"
REPEATS="${REPEATS:-2}"
DRIVER_DEADLINE_S="${DRIVER_DEADLINE_S:-1080}"
mkdir -p "$OUT"

assert_cards_free || exit 1
load_card_order "$OUT/cards.txt" || exit 1
claim_cards "274-slice-d-pairing-ab"
trap 'stop_vram_sampler; release_cards "slice-d abgebrochen"; exit 1' INT TERM
start_vram_sampler "$OUT/vram.csv"

cd "$WT" || exit 1
launch_server "$LOG" /tmp/slice-d-pairing.pid \
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
  --dual-group-lane-spec --dual-group-lane-spec-steps "$STEPS" \
  --dual-group-lane-share-window-s 1.0 \
  --enable-metrics \
  --host 127.0.0.1 --port "$PORT"

if dry_run; then
  stop_vram_sampler
  release_cards "slice-d dry run"
  echo "DRY RUN ok: slice-d pairing A/B"
  exit 0
fi

if ! wait_for_server "$PORT" 1200; then
  echo "--- letzte 40 Zeilen Serverlog (nur bei Abbruch) ---" >&2
  tail -40 "$LOG" >&2
  stop_vram_sampler
  release_cards "slice-d: server nicht oben, verbraucht"
  exit 1
fi

grep -E "lane budget [0-9]+ MiB =|verify graph captured|NEXTN head graph captured|HEAD pool sizing|lane 0 ready|worker started|pairing" \
  "$LOG" | tee "$OUT/contract_lines.txt"

"$VENV/bin/python" "$WT/scripts/dual_group/slice_d/pairing_ab.py" \
  --port "$PORT" --tokenizer "$TARGET_DIR" \
  --steps "$STEPS" --window-s "$WINDOW_S" --r8-window-s "$R8_WINDOW_S" \
  --phases "$PHASES" --repeats "$REPEATS" \
  --deadline-s "$DRIVER_DEADLINE_S" \
  --out "$OUT/report.json" | tee "$OUT/report.txt"
DRIVER_RC=${PIPESTATUS[0]}

curl -sf -m 30 "http://127.0.0.1:$PORT/get_server_info" > "$OUT/server_info.json"

kill "$(cat /tmp/slice-d-pairing.pid)" 2>/dev/null
stop_vram_sampler
echo "== VRAM/Leistung je Karte im Minimum =="
summarise_vram "$OUT/vram.csv" | tee "$OUT/vram_summary.txt"
release_cards "slice-d DONE (driver rc=$DRIVER_RC), Rohdaten in $OUT"
echo "fertig: $OUT (driver rc=$DRIVER_RC)"
exit "$DRIVER_RC"
