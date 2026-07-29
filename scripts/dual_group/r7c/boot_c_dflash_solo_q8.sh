#!/usr/bin/env bash
# BOOT C -- #274 round 7c posten 1: the quantised DFLASH drafter, on the card
# the serving group's biggest shard is NOT on.
#
# QUESTION
#   Can a DFLASH drafter live on a second card at all on this rig, and what
#   does it buy? Round 7b's answer was "no": 3300 MiB of BF16 weights against
#   1710 MiB free. Round 7c changed two of the three terms -- the Q8_0 GGUF is
#   1753 MiB instead of 3300, and the solo-draft path already takes a card
#   index, so no lane rebuild is needed.
#
# WHY THIS IS NOT A LANE
#   Said plainly because the difference matters for reading the result: the
#   solo drafter drafts FOR THE SERVING GROUP. It is not a second concurrent
#   inference lane. It is, however, exactly the right shape for the
#   architecture-vs-algorithm A/B (nachtrag 13b), because both arms then serve
#   the same group.
#
# WHY NO LANE IN THIS BOOT
#   The lane's NEXTN head nests into the head the SERVING group is already
#   running -- that shared complement is why it costs 2684 MiB instead of a
#   fresh copy. A serving group drafting with DFLASH builds no NEXTN head, so
#   a NEXTN lane beside it would have to load its own. The reseed A/B
#   therefore rides on boot D, not here.
#
# PLACEMENT
#   --speculative-draft-gpu takes a CUDA index and the hosting RANK is derived
#   from it. Default here is the first 3080 (CUDA_SMALL), which is the whole
#   point: put the drafter where the big shard is not. Override with
#   DRAFT_GPU=$CUDA_BIG to run the 5090 arm instead.
#
# THE RESERVE IS THE PRICE
#   A 3080's 2700 MiB reserve is NOT free space -- rig-runbook 153-157 records
#   that 2200 boots and then OOMs in the GDN prefill scratch. The drafter's
#   ~1753 MiB weights + KV + graphs must come out of KV, so the reserve is
#   raised on the hosting card AND the context is shortened. Both are stated
#   here rather than tuned during the run.
#
# EXPECTATION
#   Primary: the Q8_0 GGUF drafter LOADS (58 tensors, CPU-gated on this branch)
#   and generates coherent text. That alone closes round 7b's "not buildable".
#   Secondary: DFLASH accept and ms/round against the NEXTN numbers on the same
#   target. DFLASH is known to be poor on prose and better on code.
#
# TIME WINDOW   ~45 min   (GGUF target load ~10-15 min, drafter load ~2 min,
#                          measurement ~15 min, slack ~15 min)
# ABORT
#   - drafter load raises on a tensor name -> abort and report the name; the
#     CPU gate says 94/94 resolve, so a failure here is new information
#   - OOM on the hosting card -> abort; raise RESERVE_HOST next time rather
#     than retrying inside the window
#   - server not up in 30 min -> abort
#   - output is incoherent -> that is a RESULT, not an abort: record it and
#     stop (a quantised drafter that proposes garbage is the Q4-fragility the
#     EVAL doc warned about, one step up the ladder)

set -uo pipefail
cd "$(dirname "$0")"
source ./common.sh

TARGET_DIR="$MODEL_ROOT/Qwen3.6-27B-MTP-Q3_K_M-GGUF"
TARGET="$TARGET_DIR/Qwen3.6-27B-Q3_K_M.gguf"
DRAFT="$MODEL_ROOT/qwen3.6-27b-dflash-gguf"
PORT="${PORT:-30079}"
CTX="${CTX:-16384}"
LOG=/tmp/r7c-boot-c.server.log
OUT=/tmp/r7c-boot-c
mkdir -p "$OUT"

[ -f "$DRAFT/config.json" ] || { echo "ABBRUCH: $DRAFT/config.json fehlt" >&2; exit 1; }
[ -f "$DRAFT/Qwen3.6-27B-DFlash-Q8_0.gguf" ] || { echo "ABBRUCH: Q8_0-GGUF fehlt" >&2; exit 1; }

assert_cards_free || exit 1
load_card_order | tee "$OUT/cards.txt"

# Host the drafter on the first 3080 unless told otherwise.
DRAFT_GPU="${DRAFT_GPU:-${CUDA_SMALL%%,*}}"
echo "Drafter-Karte: cuda:$DRAFT_GPU   (CUDA_BIG=$CUDA_BIG, CUDA_SMALL=$CUDA_SMALL)"

# Reserve per CUDA rank: raise it on the hosting card by the drafter's full
# footprint (weights 1753 + KV/graph headroom ~550).
RESERVE_HOST="${RESERVE_HOST:-5000}"
RESERVE=""
for i in 0 1 2; do
  if [ "$i" = "$DRAFT_GPU" ]; then RESERVE="$RESERVE,$RESERVE_HOST"
  elif [ "$i" = "$CUDA_BIG" ]; then RESERVE="$RESERVE,3000"
  else RESERVE="$RESERVE,2700"; fi
done
RESERVE="${RESERVE#,}"
echo "--rank-auto-reserve-mib $RESERVE"

claim_cards "274-r7c-boot-c-dflash-solo-q8"
trap 'stop_vram_sampler; release_cards "boot C abgebrochen"; exit 1' INT TERM
start_vram_sampler "$OUT/vram.csv"
export SGLANG_ACCEPT_POSITION_PROBE=1

cd "$WT"
setsid "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$TARGET" \
  --tokenizer-path "$TARGET_DIR" \
  --tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio auto-performance \
  --rank-auto-reserve-mib "$RESERVE" \
  --attention-backend flashinfer \
  --kv-cache-dtype fp8_e4m3 --context-length "$CTX" --trust-remote-code \
  --max-running-requests 8 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "$DRAFT" \
  --speculative-draft-placement solo \
  --speculative-draft-gpu "$DRAFT_GPU" \
  --enable-metrics \
  --host 127.0.0.1 --port "$PORT" \
  > "$LOG" 2>&1 &
echo $! > /tmp/r7c-boot-c.pid

if ! wait_for_server "$PORT" 1800; then
  echo "--- letzte 60 Zeilen Serverlog (nur bei Abbruch) ---" >&2
  tail -60 "$LOG" >&2
  stop_vram_sampler
  release_cards "boot C: server nicht oben, verbraucht"
  exit 1
fi

# Coherence FIRST: a drafter that loads but proposes garbage is the outcome
# this boot is most likely to produce, and it is cheap to see.
"$VENV/bin/python" "$WT/scripts/dual_group/lane_accept_probe.py" \
  --port "$PORT" --tokens 192 --steps 3 --no-lane \
  --prompts alphabet,squares,repeat,code,prose \
  --tokenizer "$MODEL_ROOT/Qwen3.6-27B-MTP-Q3_K_M-GGUF" \
  --out "$OUT/accept.json" | tee "$OUT/accept.txt"

curl -sf "http://127.0.0.1:$PORT/get_server_info" > "$OUT/server_info.json"
grep -iE "dflash|draft|solo|gguf" "$LOG" | tail -60 > "$OUT/loader_lines.txt"

kill "$(cat /tmp/r7c-boot-c.pid)" 2>/dev/null
stop_vram_sampler
summarise_vram "$OUT/vram.csv" | tee "$OUT/vram_summary.txt"
release_cards "boot C DONE, Rohdaten in $OUT"
echo "fertig: $OUT"
