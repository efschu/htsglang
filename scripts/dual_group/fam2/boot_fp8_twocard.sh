#!/usr/bin/env bash
# #274 families slice 2, FP8 TWO-CARD arm -- the constellation section 11.19
# left open.
#
# QUESTION
#   Qwen3.6-27B-FP8 carries 28.77 GiB of weights and the lane card has 31.3
#   usable, so a one-card lane is arithmetically out (section 11.19). The
#   two-card constellation removes exactly that wall: the lane's second part
#   lives on the second serving card, and the lane's collectives -- which are
#   local tensor ops, not wire operations -- become one activation hop per
#   shell.
#
# TWO FINDINGS THIS RECIPE STANDS ON, BOTH FROM THE DESK
#   1. Section 11.19's arithmetic was still too kind. The HULL was built with
#      real storage for every linear-attention family, on the reasoning that
#      quantized weights are lazily allocated. That is true of GGUF
#      (GGUFUninitializedParameter) and false of fp8 (create_weights calls
#      torch.empty), so an FP8-GDN lane allocated the whole model a SECOND
#      time before assembling it away. The hull now builds on meta for that
#      family too, and the two composed-by-value tensors (conv kernel,
#      per-head GDN vectors) are materialized where they are filled.
#   2. At TP=2 the lane geometry equals the serving geometry, so every nesting
#      probe holds trivially -- checked on CPU, including the block-quant axis
#      (weight_block_size [128,128] against intermediate 17408).
#
# FIXPOSTEN
#   FIRST ATTEMPT, and why the numbers below are not the first ones: at ratio
#   3:1 with budgets 25000/10000 the SERVING group died before the lane, in
#   its own hybrid-state sizing (rank 1 total_rest_memory -0.01 GB). The lane
#   part on the foreign card has to be paid for out of THAT card's budget, so
#   the ratio moves the weight off it instead: at 5:1 rank 1 carries ~4910 MiB
#   and can afford both its own state cache and the lane's part.
#
#   weights 29460 MiB at 5:1 -> rank 0 ~24550, rank 1 ~4910.
#   5090 (32607): serving budget 27000, lane pool 1000 + meta hull, eager.
#   3080 (20480): serving budget 10500 + lane part 4910 + second CUDA
#         context ~400 = ~15.8 GiB of ~19.4.
#
# TIME  ~14 min (load ~8-10, gate ~3)

set -uo pipefail
cd "$(dirname "$0")"
WT="${WT:-/spinning/wt-lane-fam2}"
export WT
source ../r7c/common.sh

MODEL="$MODEL_ROOT/Qwen3.6-27B-FP8"
PORT="${PORT:-30093}"
LOG="${LOG:-/tmp/fam2-fp8.server.log}"
OUT="${OUT:-/tmp/fam2-fp8}"
LANE_BUDGET="${LANE_BUDGET:-1000}"
RANK_MIB="${RANK_MIB:-27000,10500}"
GATE_TOKENS="${GATE_TOKENS:-12}"
BOOT_TIMEOUT_S="${BOOT_TIMEOUT_S:-1200}"
GATE_DEADLINE_S="${GATE_DEADLINE_S:-300}"
mkdir -p "$OUT"

load_card_order "$OUT/cards.txt" || exit 1
SMALL0="${CUDA_SMALL%%,*}"

cd "$WT" || exit 1
launch_server "$LOG" /tmp/fam2-fp8.pid \
  "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" \
  --tokenizer-path "$MODEL" \
  --tp-size 2 --rank-gpu-id "$CUDA_BIG,$SMALL0" \
  --rank-tp-ratio 5,1 \
  --rank-gpu-memory-mib "$RANK_MIB" \
  --attention-backend flashinfer \
  --kv-cache-dtype fp8_e4m3 --context-length 4096 --trust-remote-code \
  --max-running-requests 1 \
  --dual-group-lane --dual-group-lane-budget-mib "$LANE_BUDGET" \
  --dual-group-lane-part-gpu-id "$CUDA_BIG,$SMALL0" \
  --host 127.0.0.1 --port "$PORT"

if dry_run; then
  echo "DRY RUN ok: fam2 FP8 two-card lane"
  exit 0
fi

if ! wait_for_server "$PORT" "$BOOT_TIMEOUT_S"; then
  echo "--- letzte 40 Zeilen Serverlog (nur bei Abbruch) ---" >&2
  tail -40 "$LOG" >&2
  exit 1
fi

grep -E "lane spans cards|dual-group lane plan verified|part rank .* loaded on cuda|model assembled|memory items|foreign card|-> added by the lane|lane 0 ready" \
  "$LOG" | tee "$OUT/contract_lines.txt"

"$VENV/bin/python" "$WT/scripts/dual_group/fam2/family_gate.py" \
  --port "$PORT" --tokenizer "$MODEL" --tokens "$GATE_TOKENS" \
  --deadline-s "$GATE_DEADLINE_S" --out "$OUT/gate.json" | tee "$OUT/gate.txt"
RC=${PIPESTATUS[0]}
curl -sf -m 30 "http://127.0.0.1:$PORT/get_server_info" > "$OUT/server_info.json"
exit "$RC"
