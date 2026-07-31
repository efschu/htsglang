#!/usr/bin/env bash
# #274 families slice 2, EXPERT arm: the first MoE lane on a card.
#
# QUESTION
#   Slice C built the fifth shell class (LaneFusedMoEShell) and pinned its
#   reduce algebra on CPU, but no MoE model ever reached the lane build: all
#   three candidates died in the SERVING group first. So the two things that
#   only a card can say are open -- does the data_ptr gate hold on real
#   expert tensors (w13/w2 and their scales), and does the lane compute the
#   same model as the Verband.
#
# VEHICLE  Qwen3.6-35B-A3B-AWQ-4bit (23.29 GiB, moe_wna16 path, 256 experts,
#   moe_intermediate 512, 2 kv heads), TP=2 on 5090 (rank 0) + one 3080.
#
# WHY TP=2 AND NOT TP=3
#   2 kv heads against 3 ranks forces REPLICATED-KV for the serving group and
#   not for the lane, which makes nesting UNDEFINED (DESIGN_121 section 3.2) --
#   the exact wall the gemma candidate hit. At TP=2 the lane geometry EQUALS
#   the serving geometry, so every probe nests trivially (checked on CPU with
#   lane_plan_probe.py before this recipe was written).
#   TP=2 is also what needed the slice-2 build at all: both segments are
#   singletons there, and the one that is not this process's own shard has to
#   be MATERIALIZED rather than aliased.
#
# FIXPOSTEN (before the window, per the feasibility gate)
#   weights 23.29 GiB at ratio 3:1 -> rank 0 ~17.5 GiB, rank 1 ~5.8 GiB.
#   5090 (32607 MiB): serving 22000 budget + lane part 5.8 GiB + lane pool
#     1200 + graphs ~= 30 GiB of 31.3 usable.
#   3080 (20480 MiB): serving 12000 budget, nothing from the lane.
#   KV is cheap here: 40 layers x 1 kv head x 128 = 10 KiB/token at fp8.
#
# TIME  ~10 min (load ~5, gate ~3, teardown ~2)

set -uo pipefail
cd "$(dirname "$0")"
WT="${WT:-/spinning/wt-lane-fam2}"
export WT
source ../r7c/common.sh

MODEL="$MODEL_ROOT/Qwen3.6-35B-A3B-AWQ-4bit"
PORT="${PORT:-30091}"
LOG="${LOG:-/tmp/fam2-moe.server.log}"
OUT="${OUT:-/tmp/fam2-moe}"
LANE_BUDGET="${LANE_BUDGET:-1200}"
RANK_MIB="${RANK_MIB:-22000,12000}"
GATE_TOKENS="${GATE_TOKENS:-12}"
BOOT_TIMEOUT_S="${BOOT_TIMEOUT_S:-900}"
GATE_DEADLINE_S="${GATE_DEADLINE_S:-360}"
mkdir -p "$OUT"

load_card_order "$OUT/cards.txt" || exit 1
SMALL0="${CUDA_SMALL%%,*}"

cd "$WT" || exit 1
launch_server "$LOG" /tmp/fam2-moe.pid \
  "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" \
  --tokenizer-path "$MODEL" \
  --tp-size 2 --rank-gpu-id "$CUDA_BIG,$SMALL0" \
  --rank-tp-ratio 3,1 \
  --rank-gpu-memory-mib "$RANK_MIB" \
  --attention-backend flashinfer \
  --kv-cache-dtype fp8_e4m3 --context-length 8192 --trust-remote-code \
  --max-running-requests 2 \
  --dual-group-lane --dual-group-lane-budget-mib "$LANE_BUDGET" \
  --host 127.0.0.1 --port "$PORT"

if dry_run; then
  echo "DRY RUN ok: fam2 MoE lane"
  exit 0
fi

if ! wait_for_server "$PORT" "$BOOT_TIMEOUT_S"; then
  echo "--- letzte 40 Zeilen Serverlog (nur bei Abbruch) ---" >&2
  tail -40 "$LOG" >&2
  exit 1
fi

grep -E "dual-group lane plan verified|model assembled|memory items|-> added by the lane|lane 0 ready" \
  "$LOG" | tee "$OUT/contract_lines.txt"

"$VENV/bin/python" "$WT/scripts/dual_group/fam2/family_gate.py" \
  --port "$PORT" --tokenizer "$MODEL" --tokens "$GATE_TOKENS" \
  --deadline-s "$GATE_DEADLINE_S" --out "$OUT/gate.json" | tee "$OUT/gate.txt"
RC=${PIPESTATUS[0]}
curl -sf -m 30 "http://127.0.0.1:$PORT/get_server_info" > "$OUT/server_info.json"
exit "$RC"
