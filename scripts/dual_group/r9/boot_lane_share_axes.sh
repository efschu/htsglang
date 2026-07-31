#!/usr/bin/env bash
# #284 -- ONE card window: the carrier of the lane's loss, and the gate rerun.
#
# QUESTION
#   Round 4 measured share_lane 1.002 and round 8 measured 0.19-0.30 on the
#   same claim. The recipes differ on several axes at once, so neither number
#   accuses anything on its own. This boot rotates the axes ONE AT A TIME
#   inside a single boot -- accept length and rate are content- and
#   boot-driven, so two boots would carry that variance inside every
#   difference they report -- and adds a second, floor-free instrument: the
#   lane's device clock, differenced per window, which says whether a lane
#   that lost rate lost CARD TIME or lost SPEED ON THE CARD.
#   Round 8's corrected coherence gate has also never run on a card; it runs
#   first here, by itself.
#
# VEHICLE  Qwen3.6-27B-MTP-Q3_K_M-GGUF, TP=3 uneven (5090 rank 0 + 2x 3080),
#   byte-identical to the r8 recipe except for the three instrument flags
#   below. NOT FP8: it cannot host a lane on this rig (28.75 GiB of weights
#   against 31.34 usable, and the lane needs the full weights once more on top
#   of both pools).
#
# WHAT IS DIFFERENT FROM r8, AND WHY IT IS SAFE TO COMPARE
#   --dual-group-lane-share-window-s 1.0 switches the ONLINE estimator on
#   (r8 ran with it off), and the device clock is new in every arm. Both cost
#   scheduler-thread work, so the recipe carries its own falsifier: the SOLO
#   serving floor and the SOLO lane floor are the same measurements r8 took
#   (54.044 and 57.155 tok/s). If they reproduce, the instruments did not move
#   the operating point; if they do not, that is the first thing the report
#   says. The axis isolation itself is WITHIN this boot and does not depend on
#   the comparison either way.
#
# TIME WINDOW  ~30 min (load ~10-12, gate ~5, nine 30 s windows + drains ~6,
#              teardown/summary ~3, slack ~4)
# ABORT
#   - server not up in 20 min -> abort, keep the log, count the boot
#   - a gate prompt's A-vs-A floor is not byte-identical -> the prompt is void,
#     the driver says so and carries on; it is not an abort

set -uo pipefail
cd "$(dirname "$0")"
WT="${WT:-/spinning/wt-284}"
export WT
source ../r7c/common.sh

TARGET_DIR="$MODEL_ROOT/Qwen3.6-27B-MTP-Q3_K_M-GGUF"
TARGET="$TARGET_DIR/Qwen3.6-27B-Q3_K_M.gguf"
PORT="${PORT:-30084}"
LOG="${LOG:-/tmp/r9-lane-axes.server.log}"
OUT="${OUT:-/tmp/r9-lane-axes}"
LANE_BUDGET="${LANE_BUDGET:-700}"
RANK_MIB="${RANK_MIB:-21000,17780,17780}"
WINDOW_S="${WINDOW_S:-30}"
GATE_TOKENS="${GATE_TOKENS:-64}"
STEPS="${STEPS:-1}"
PHASES="${PHASES:-gate,axes}"
ARMS="${ARMS:-A_baseline,B_eager_lane,C_light_load,D_depth1}"
DRIVER_DEADLINE_S="${DRIVER_DEADLINE_S:-1080}"
# The standing gate this round formulates: the lane keeps at least 30 % of its
# solo rate under the named load. 0.30 is deliberately the round-8 measurement
# and not a wish -- a threshold picked above what the rig does would report red
# forever and say nothing; this one turns red the moment the number regresses.
SHARE_MIN="${SHARE_MIN:-0.30}"
SHARE_LOAD="${SHARE_LOAD:-4 concurrent 128-token serving requests}"
mkdir -p "$OUT"

assert_cards_free || exit 1
load_card_order "$OUT/cards.txt" || exit 1
claim_cards "284-lane-share-axes"
trap 'stop_vram_sampler; release_cards "284 abgebrochen"; exit 1' INT TERM
start_vram_sampler "$OUT/vram.csv"
export SGLANG_ACCEPT_POSITION_PROBE=1

cd "$WT" || exit 1
launch_server "$LOG" /tmp/r9-lane-axes.pid \
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
  --dual-group-lane-share-min "$SHARE_MIN" \
  --dual-group-lane-share-load "$SHARE_LOAD" \
  --enable-metrics \
  --host 127.0.0.1 --port "$PORT"

if dry_run; then
  stop_vram_sampler
  release_cards "284 dry run"
  echo "DRY RUN ok: r9 lane share axes"
  exit 0
fi

if ! wait_for_server "$PORT" 1200; then
  echo "--- letzte 40 Zeilen Serverlog (nur bei Abbruch) ---" >&2
  tail -40 "$LOG" >&2
  stop_vram_sampler
  release_cards "284: server nicht oben, verbraucht"
  exit 1
fi

grep -E "lane budget [0-9]+ MiB =|verify graph captured|NEXTN head graph captured|HEAD pool sizing|lane 0 ready|worker started" \
  "$LOG" | tee "$OUT/contract_lines.txt"

"$VENV/bin/python" "$WT/scripts/dual_group/r9/lane_share_axes.py" \
  --port "$PORT" --tokenizer "$TARGET_DIR" \
  --steps "$STEPS" --gate-tokens "$GATE_TOKENS" \
  --window-s "$WINDOW_S" --phases "$PHASES" --arms "$ARMS" \
  --deadline-s "$DRIVER_DEADLINE_S" \
  --out "$OUT/report.json" | tee "$OUT/report.txt"
DRIVER_RC=${PIPESTATUS[0]}

curl -sf -m 30 "http://127.0.0.1:$PORT/get_server_info" > "$OUT/server_info.json"

kill "$(cat /tmp/r9-lane-axes.pid)" 2>/dev/null
stop_vram_sampler
echo "== VRAM/Leistung je Karte im Minimum =="
summarise_vram "$OUT/vram.csv" | tee "$OUT/vram_summary.txt"
release_cards "284 DONE (driver rc=$DRIVER_RC), Rohdaten in $OUT"
echo "fertig: $OUT (driver rc=$DRIVER_RC)"
exit "$DRIVER_RC"
