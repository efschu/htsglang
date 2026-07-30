#!/usr/bin/env bash
# BOOT A -- #274 round 7c posten 0: the one-axis accept falsifier.
#
# QUESTION
#   The GGUF-Q3 vehicle measures accept 1.15-1.53 (serving group, round 7b).
#   The reference for this model class is 2.69-3.28. Round 7c's desk work
#   removed four of the five axes that separated the two measurements: the
#   cross-algo path and adaptive K are ruled out by docs/benchmarks/
#   htsglang_tp3.json:87-90, which has the SAME FP8 vehicle at pure NEXTN and
#   K=3 measuring 3.279 (code) / 2.688 (prose). What is left is the TARGET
#   QUANTISATION, and this boot moves only that: same K, same spec config,
#   same five prompts, same probe as round 7b -- different vehicle.
#
# VEHICLE
#   Qwen3.6-27B-FP8, the BASE checkpoint. Not AEON: the reference was measured
#   on the base one, and AEON's recipe.yaml grafts the identical MTP block, so
#   base holds the fine-tune axis still rather than adding it back.
#   No lane: 27B-FP8 does not fit beside one (rig-runbook 1034-1036), which is
#   what --no-lane exists for.
#
# EXPECTATION
#   Accept 2.6-3.3 on code/prose, position 0 around 65%. That confirms the
#   reference exists on this rig at K=3 and makes the GGUF ceiling a
#   target-quantisation property.
#   The falsifying outcome is accept ~1.5 with position 0 at 24-45%, i.e. the
#   same band as GGUF -- then the reference is a property of the old measuring
#   path after all and the difference to the htsglang_tp3 setup has to be named.
#
# TIME WINDOW   ~35 min   (load ~8-12 min, 5 prompts x 192 tokens ~10 min,
#                          teardown + summary ~5 min, slack ~10 min)
# ABORT
#   - server not up in 25 min -> abort, keep the log, count the boot
#   - accept_len_mean is None after the first prompt -> the probe is off or the
#     spec path is not running; abort rather than collect zeros
#   - any OOM in the load -> abort, do NOT retry with a lower reserve in the
#     same window (that is a different experiment)

set -uo pipefail
cd "$(dirname "$0")"
source ./common.sh

PORT="${PORT:-30077}"
TOKENS="${TOKENS:-192}"
LOG="${LOG:-/tmp/r7c-boot-a.server.log}"
OUT="${OUT:-/tmp/r7c-boot-a}"
mkdir -p "$OUT"

assert_cards_free || exit 1
load_card_order "$OUT/cards.txt" || exit 1
claim_cards "274-r7c-boot-a-fp8-referenz"
trap 'stop_vram_sampler; release_cards "boot A abgebrochen"; exit 1' INT TERM

start_vram_sampler "$OUT/vram.csv"

# Probe ON: this is the whole point of the boot. It is a diagnostic and costs
# nothing that the comparison cares about (raw counters, no extra forward).
export SGLANG_ACCEPT_POSITION_PROBE=1

cd "$WT" || exit 1
launch_server "$LOG" /tmp/r7c-boot-a.pid \
  "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL_ROOT/Qwen3.6-27B-FP8" \
  --tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio auto-performance \
  --rank-auto-reserve-mib 3000,2700,2700 \
  --kv-cache-dtype fp8_e4m3 --context-length 32768 --trust-remote-code \
  --max-running-requests 16 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --enable-metrics \
  --host 127.0.0.1 --port "$PORT"

if dry_run; then
  stop_vram_sampler
  release_cards "boot A dry run"
  echo "DRY RUN ok: boot A"
  exit 0
fi

if ! wait_for_server "$PORT" 1500; then
  stop_vram_sampler
  release_cards "boot A: server nicht oben, verbraucht"
  exit 1
fi

# The five round-7b contents, at K=3, serving group only.
"$VENV/bin/python" "$WT/scripts/dual_group/lane_accept_probe.py" \
  --port "$PORT" --tokens "$TOKENS" --steps 3 --no-lane \
  --prompts alphabet,squares,repeat,code,prose \
  --tokenizer "$MODEL_ROOT/Qwen3.6-27B-FP8" \
  --out "$OUT/accept.json" | tee "$OUT/accept.txt"

# Raw before analysis: the whole probe snapshot, not just what the driver
# formatted.
curl -sf "http://127.0.0.1:$PORT/get_server_info" > "$OUT/server_info.json"

kill "$(cat /tmp/r7c-boot-a.pid)" 2>/dev/null
stop_vram_sampler
echo "== VRAM, freie MiB je Karte im Minimum =="
summarise_vram "$OUT/vram.csv" | tee "$OUT/vram_summary.txt"
release_cards "boot A DONE, Rohdaten in $OUT"
echo "fertig: $OUT"
