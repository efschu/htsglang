#!/usr/bin/env bash
# #274 families slice 2, TWO-CARD arm on the DENSE vehicle.
#
# QUESTION
#   DESIGN_121 section 11.11 point 3 booked "a lane larger than one card" as
#   its own build, on the reasoning that it needs REAL lane collectives. It
#   does not: the lane's collectives are already local tensor ops, so a lane
#   that spans two cards moves one ACTIVATION to the foreign part and the
#   result back. This boot asks whether that is true on hardware -- on the
#   small, fast vehicle, so that the expensive FP8 boot is a repeat of a
#   proven mechanism rather than a first attempt.
#
# VEHICLE  Llama-3.1-8B-Instruct (14.97 GiB, dense full attention), TP=2 on
#   5090 (rank 0) + one 3080, lane rank 1 placed on the 3080.
#
# WHAT IS NEW HERE COMPARED WITH THE ONE-CARD DENSE ARM (section 11.18)
#   1. --dual-group-lane-part-gpu-id: the host process is given the second
#      card in CUDA_VISIBLE_DEVICES before it starts.
#   2. The lane part is a MATERIALIZED SINGLETON: at TP=2 both segments are
#      one serving rank each, and the one that is not this process's own has
#      to be loaded rather than aliased.
#   3. The shells hop: column/row/vocab/lm_head each send their activation to
#      the foreign part and bring the result home.
#
# FIXPOSTEN
#   weights 15330 MiB at 3:1 -> rank 0 ~11500, rank 1 ~3830.
#   5090: serving budget 16000 + lane pool 1600 + meta hull residue.
#   3080: serving budget 8000 + lane part 3830 + a second CUDA context ~400
#         = ~12.3 GiB of ~19.4 usable.
#
# TIME  ~6 min (load ~2 x 2, gate ~2)

set -uo pipefail
cd "$(dirname "$0")"
WT="${WT:-/spinning/wt-lane-fam2}"
export WT
source ../r7c/common.sh

MODEL="$MODEL_ROOT/Llama-3.1-8B-Instruct"
PORT="${PORT:-30092}"
LOG="${LOG:-/tmp/fam2-dense2.server.log}"
OUT="${OUT:-/tmp/fam2-dense2}"
LANE_BUDGET="${LANE_BUDGET:-1600}"
RANK_MIB="${RANK_MIB:-16000,8000}"
GATE_TOKENS="${GATE_TOKENS:-12}"
BOOT_TIMEOUT_S="${BOOT_TIMEOUT_S:-900}"
GATE_DEADLINE_S="${GATE_DEADLINE_S:-300}"
# Extra launch flags, appended verbatim. The first user is the follow-up boot
# this arm's own result asked for: the serving floor was red on all three
# prompts while the lane floor was green, and the named suspect is the prefix
# cache (a second identical request taking a different kernel path). Passing
# --disable-radix-cache here answers that with one boot and no code change --
# which is exactly why it is a flag and not an edit.
EXTRA_ARGS="${EXTRA_ARGS:-}"
mkdir -p "$OUT"

load_card_order "$OUT/cards.txt" || exit 1
SMALL0="${CUDA_SMALL%%,*}"

cd "$WT" || exit 1
launch_server "$LOG" /tmp/fam2-dense2.pid \
  "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" \
  --tokenizer-path "$MODEL" \
  --tp-size 2 --rank-gpu-id "$CUDA_BIG,$SMALL0" \
  --rank-tp-ratio 3,1 \
  --rank-gpu-memory-mib "$RANK_MIB" \
  --attention-backend flashinfer \
  --context-length 8192 --trust-remote-code \
  --max-running-requests 2 \
  --dual-group-lane --dual-group-lane-budget-mib "$LANE_BUDGET" \
  --dual-group-lane-part-gpu-id "$CUDA_BIG,$SMALL0" \
  --host 127.0.0.1 --port "$PORT" \
  ${EXTRA_ARGS}

if dry_run; then
  echo "DRY RUN ok: fam2 dense two-card lane"
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
