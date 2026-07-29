#!/usr/bin/env bash
# BOOT B -- #274 round 7c: the head-precision axis, closed at the top.
#
# QUESTION
#   Round 7b ruled out MTP-head quantisation by comparing a Q3_K head against
#   a Q6_K one -- same band, 1.14-1.51. By the sample-width rule that verdict
#   only covers Q3..Q6; an unquantised head was never run, and round 7c's
#   inventory showed why that gap matters: this head is 425 M parameters, so
#   it is large enough for its precision to cost something.
#
# VEHICLE
#   Huihui-Qwen3.6-27B-abliterated-AWQ-MTP. Read out of the checkpoint, not
#   assumed: all 15 mtp.* tensors are BF16 (424,699,392 parameters, 810 MiB)
#   on an AWQ INT4 body, and `mtp` is in modules_to_not_convert. That is a
#   COARSE target with a FULL-PRECISION head -- the exact arm that was missing,
#   and it needs no conversion.
#
# WHAT IT SEPARATES
#   Boot A moves the target quantisation with the head following it. This boot
#   holds the target coarse (INT4, comparable to Q3_K_M's 3.9 bpw) and lifts
#   ONLY the head to BF16. Together the two decide between "the head's
#   precision is the lever" and "the target's is".
#
# EXPECTATION
#   If the head is the lever: accept clearly above the 1.15-1.53 band.
#   If the target is: accept stays in that band despite a BF16 head, and the
#   head-precision axis is then closed from Q3 to BF16 rather than Q3 to Q6.
#   Either outcome is a result; there is no null here.
#
# TIME WINDOW   ~40 min   (AWQ load is slower than FP8; ~12-18 min load,
#                          ~10 min measurement, slack ~12 min)
# ABORT
#   - AWQ x uneven-TP x MTP has never been booted on this branch. If the load
#     rejects the shape (the round-7b GPTQ arm died on "Dimension of size 136
#     is not a multiple of its unit count 1088"), abort and count the boot --
#     do NOT start tuning the ratio inside this window.
#   - server not up in 30 min -> abort
#   - probe reports rounds == 0 -> the spec path is not running, abort

set -uo pipefail
cd "$(dirname "$0")"
source ./common.sh

MODEL="$MODEL_ROOT/Huihui-Qwen3.6-27B-abliterated-AWQ-MTP"
PORT="${PORT:-30078}"
TOKENS="${TOKENS:-192}"
LOG=/tmp/r7c-boot-b.server.log
OUT=/tmp/r7c-boot-b
mkdir -p "$OUT"

assert_cards_free || exit 1
load_card_order | tee "$OUT/cards.txt"
claim_cards "274-r7c-boot-b-dense-head"
trap 'stop_vram_sampler; release_cards "boot B abgebrochen"; exit 1' INT TERM

start_vram_sampler "$OUT/vram.csv"
export SGLANG_ACCEPT_POSITION_PROBE=1

cd "$WT"
setsid "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio auto-performance \
  --rank-auto-reserve-mib 3000,2700,2700 \
  --kv-cache-dtype fp8_e4m3 --context-length 32768 --trust-remote-code \
  --max-running-requests 16 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --enable-metrics \
  --host 127.0.0.1 --port "$PORT" \
  > "$LOG" 2>&1 &
echo $! > /tmp/r7c-boot-b.pid

if ! wait_for_server "$PORT" 1800; then
  echo "--- letzte 40 Zeilen Serverlog (nur bei Abbruch) ---" >&2
  tail -40 "$LOG" >&2
  stop_vram_sampler
  release_cards "boot B: server nicht oben, verbraucht"
  exit 1
fi

"$VENV/bin/python" "$WT/scripts/dual_group/lane_accept_probe.py" \
  --port "$PORT" --tokens "$TOKENS" --steps 3 --no-lane \
  --prompts alphabet,squares,repeat,code,prose \
  --tokenizer "$MODEL" \
  --out "$OUT/accept.json" | tee "$OUT/accept.txt"

curl -sf "http://127.0.0.1:$PORT/get_server_info" > "$OUT/server_info.json"

kill "$(cat /tmp/r7c-boot-b.pid)" 2>/dev/null
stop_vram_sampler
summarise_vram "$OUT/vram.csv" | tee "$OUT/vram_summary.txt"
release_cards "boot B DONE, Rohdaten in $OUT"
echo "fertig: $OUT"
