#!/usr/bin/env bash
# #274 round 8 -- ONE card window: the lane's chain gated, then priced.
#
# QUESTION
#   Rounds 5-7a built the chain and captured all three of its pieces (verify
#   graph, head graph, rung ladder) and measured them SERIALLY. The card
#   equivalents WITH speculation on the lane were measured once, in round 4,
#   with an EAGER seqdecode verify -- 67 ms per verify against a 16 ms decode.
#   Round 7a predicted, in writing, that the concurrency gain would SHRINK once
#   the verify was captured, because a large part of what concurrency collected
#   was the lane's own CPU-side gaps. That number has never been taken.
#   This boot takes it, and gates coherence in the same boot first.
#
# VEHICLE  Qwen3.6-27B-MTP-Q3_K_M-GGUF, TP=3 uneven (5090 rank 0 + 2x 3080).
#   NOT Qwen3.6-27B-FP8: it carries 28.75 GiB of weights against 31.34 GiB
#   usable on the lane card and the lane needs the full weights ONCE on top of
#   both pools (rig-runbook 4.11, INTEGRATION_R3 families slice arm B). FP8 is
#   the vehicle of the accept-REFERENCE boot, which runs --no-lane; it is not
#   a lane vehicle on this rig and pinning it here would produce no boot.
#
# MODE   concurrent. Serial tick-sharing is by construction a zero-sum split of
#   one wall clock (measured E 0.91-0.97 no-spec, 1.069 with the eager chain)
#   and is carried from those recorded runs rather than re-measured: the mode
#   is a LAUNCH flag, two modes are two boots, and the open number is the
#   concurrent one.
#
# BUDGETS  Concurrency owns its own cuda-graph pool, its own GGUF dequant
#   workspace and its own flashinfer float workspace (+412 MiB measured), so
#   the lane pool must be SMALLER than the serial recipe's: 700 MiB carries,
#   1600 dies in the lane's breakable prefill capture. Rank 0 gives back for
#   the head's 2684 MiB of weights (22800 -> 21000 measured in round 5).
#
# TIME WINDOW  ~35 min (load ~10-12, gate ~5, three 45 s windows + drains ~5,
#              teardown/summary ~4, slack ~8)
# ABORT
#   - server not up in 20 min -> abort, keep the log, count the boot
#   - the gate's A-vs-A floor is not byte-identical -> the prompt is void, the
#     driver says so and carries on; it is not an abort
#   - spec output != no-spec output -> NOT an abort. That is the measurement.

set -uo pipefail
cd "$(dirname "$0")"
source ../r7c/common.sh

TARGET_DIR="$MODEL_ROOT/Qwen3.6-27B-MTP-Q3_K_M-GGUF"
TARGET="$TARGET_DIR/Qwen3.6-27B-Q3_K_M.gguf"
PORT="${PORT:-30081}"
LOG="${LOG:-/tmp/r8-lane-spec.server.log}"
OUT="${OUT:-/tmp/r8-lane-spec}"
LANE_BUDGET="${LANE_BUDGET:-700}"
RANK_MIB="${RANK_MIB:-21000,17780,17780}"
WINDOW_S="${WINDOW_S:-45}"
GATE_TOKENS="${GATE_TOKENS:-64}"
STEPS="${STEPS:-1}"
PHASES="${PHASES:-gate,e_spec,e_nospec}"
DRIVER_DEADLINE_S="${DRIVER_DEADLINE_S:-900}"
FALSIFY_PROMPT="${FALSIFY_PROMPT:-alphabet}"
# Only read by the "history" phase, which holds the arm fixed and varies
# the preceding request sequence. Unset means the driver's own default
# order (none, alphabet_only, repeat_only, gate_full) and the flag is not
# passed at all, so every other recipe stays byte-identical.
HISTORY_VARIANTS="${HISTORY_VARIANTS:-}"
# CONCURRENT=0 boots the same recipe in SERIAL tick mode. It is the one axis
# that separates "the chain diverges" from "the concurrency machinery makes it
# diverge": rounds 5-7a gated this chain green SERIALLY, so a red gate under
# --dual-group-lane-concurrent and a green one without it names the carrier in
# one boot. Serial mode does not allocate the second graph pool, so it also
# carries the larger lane budget the serial recipe uses.
CONCURRENT="${CONCURRENT:-1}"
MODE_FLAGS=()
if [ "$CONCURRENT" = "1" ]; then
  MODE_FLAGS=(--dual-group-lane-concurrent --dual-group-lane-admission-ms 2.0)
  MODE_LABEL=concurrent
else
  MODE_LABEL=serial
fi
mkdir -p "$OUT"

assert_cards_free || exit 1
load_card_order "$OUT/cards.txt" || exit 1
claim_cards "274-r8-lane-spec-$MODE_LABEL"
trap 'stop_vram_sampler; release_cards "r8 abgebrochen"; exit 1' INT TERM
start_vram_sampler "$OUT/vram.csv"
export SGLANG_ACCEPT_POSITION_PROBE=1

cd "$WT" || exit 1
launch_server "$LOG" /tmp/r8-lane-spec.pid \
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
  ${MODE_FLAGS[@]+"${MODE_FLAGS[@]}"} \
  --dual-group-lane-spec --dual-group-lane-spec-steps "$STEPS" \
  --enable-metrics \
  --host 127.0.0.1 --port "$PORT"

if dry_run; then
  stop_vram_sampler
  release_cards "r8 dry run"
  echo "DRY RUN ok: r8 lane spec"
  exit 0
fi

if ! wait_for_server "$PORT" 1200; then
  echo "--- letzte 40 Zeilen Serverlog (nur bei Abbruch) ---" >&2
  tail -40 "$LOG" >&2
  stop_vram_sampler
  release_cards "r8: server nicht oben, verbraucht"
  exit 1
fi

# The two boot lines that decide whether the numbers below mean anything: the
# budget ledger (which post got how many MiB and why) and the two captures.
# Grepped, not eyeballed, and never the whole log.
grep -E "lane budget [0-9]+ MiB =|verify graph captured|NEXTN head graph captured|HEAD pool sizing|lane 0 ready" \
  "$LOG" | tee "$OUT/contract_lines.txt"

"$VENV/bin/python" "$WT/scripts/dual_group/r8/lane_spec_window.py" \
  --port "$PORT" --tokenizer "$TARGET_DIR" \
  --steps "$STEPS" --gate-tokens "$GATE_TOKENS" \
  --window-s "$WINDOW_S" --phases "$PHASES" \
  --falsify-prompt "$FALSIFY_PROMPT" \
  ${HISTORY_VARIANTS:+--history-variants "$HISTORY_VARIANTS"} \
  --deadline-s "$DRIVER_DEADLINE_S" \
  --out "$OUT/report.json" | tee "$OUT/report.txt"
DRIVER_RC=${PIPESTATUS[0]}

curl -sf -m 30 "http://127.0.0.1:$PORT/get_server_info" > "$OUT/server_info.json"

kill "$(cat /tmp/r8-lane-spec.pid)" 2>/dev/null
stop_vram_sampler
echo "== VRAM/Leistung je Karte im Minimum =="
summarise_vram "$OUT/vram.csv" | tee "$OUT/vram_summary.txt"
release_cards "r8 DONE (driver rc=$DRIVER_RC), Rohdaten in $OUT"
echo "fertig: $OUT (driver rc=$DRIVER_RC)"
exit "$DRIVER_RC"
