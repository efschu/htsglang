#!/usr/bin/env bash
# BOOT D -- #274 round 7c posten 2: does the target-hidden re-seed change
# anything?
#
# QUESTION
#   Until this round the lane left the head's OWN hidden in the KV of the
#   accepted positions, while the serving group overwrites exactly those with
#   the TARGET's (EagleWorkerV2._draft_extend_for_decode). That was the last
#   structural difference between the two chains. It is now closed; what it
#   costs and what it buys is unmeasured.
#
# WHY A BOOT OF ITS OWN
#   It cannot ride on boot C: the lane's NEXTN head nests into the head the
#   SERVING group runs, and boot C's serving group drafts with DFLASH. This is
#   the round-7b configuration exactly -- known bootable, so it is the cheapest
#   boot in the queue.
#
# BOTH ARMS FROM ONE BOOT
#   `draft_reseed: false` per job keeps the old behaviour, so re-seeded and not
#   re-seeded are two arms on the SAME token ids in the SAME boot. Accept is
#   content-driven; comparing it across boots would carry the content
#   difference as if it were the effect. Same reason the five switches before
#   it exist.
#
# EXPECTATION
#   On THIS vehicle: little or nothing. Round 7b measured the head to be
#   practically insensitive to its own KV (the rollback fix was byte-neutral),
#   and the chain barely reaches position 1 here. The number that matters is
#   `reseed_forwards` -- the price -- and `decode_ms_mean`, so the cost is on
#   the record before a vehicle with real accept makes the benefit visible.
#   A LARGE effect here would be the surprise, and would mean the two chains
#   were further apart than round 7b's curves suggested.
#
# TIME WINDOW   ~30 min   (GGUF load ~10-15 min, 2 arms x 3 prompts ~8 min,
#                          slack ~8 min)
# ABORT
#   - server not up in 25 min -> abort
#   - lane job raises -> abort and report the traceback; the CPU gates cover
#     the arithmetic, so a runtime failure is new information
#   - output_ids differ between the arms: NOT an abort. That is the measurement.

set -uo pipefail
cd "$(dirname "$0")"
source ./common.sh

TARGET_DIR="$MODEL_ROOT/Qwen3.6-27B-MTP-Q3_K_M-GGUF"
TARGET="$TARGET_DIR/Qwen3.6-27B-Q3_K_M.gguf"
PORT="${PORT:-30080}"
LOG="${LOG:-/tmp/r7c-boot-d.server.log}"
OUT="${OUT:-/tmp/r7c-boot-d}"
mkdir -p "$OUT"
# The inline driver below reads these out of the environment.
export WT PORT MODEL_ROOT OUT TARGET_DIR

assert_cards_free || exit 1
load_card_order "$OUT/cards.txt" || exit 1
claim_cards "274-r7c-boot-d-lane-reseed"
trap 'stop_vram_sampler; release_cards "boot D abgebrochen"; exit 1' INT TERM
start_vram_sampler "$OUT/vram.csv"
export SGLANG_ACCEPT_POSITION_PROBE=1

cd "$WT" || exit 1
launch_server "$LOG" /tmp/r7c-boot-d.pid \
  "$VENV/bin/python" -m sglang.launch_server \
  --model-path "$TARGET" \
  --tokenizer-path "$TARGET_DIR" \
  --tp-size 3 --rank-gpu-id 0,1,2 \
  --rank-tp-ratio 2,1,1 --rank-mlp-ratio 6,1,1 --rank-vocab-ratio 6,1,1 \
  --rank-gpu-memory-mib 22800,17780,17780 \
  --attention-backend flashinfer \
  --kv-cache-dtype fp8_e4m3 --context-length 16384 --trust-remote-code \
  --max-running-requests 4 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --dual-group-lane --dual-group-lane-budget-mib 1600 \
  --dual-group-lane-spec --dual-group-lane-spec-steps 3 \
  --enable-metrics \
  --host 127.0.0.1 --port "$PORT"

if dry_run; then
  stop_vram_sampler
  release_cards "boot D dry run"
  echo "DRY RUN ok: boot D"
  exit 0
fi

if ! wait_for_server "$PORT" 1500; then
  echo "--- letzte 40 Zeilen Serverlog (nur bei Abbruch) ---" >&2
  tail -40 "$LOG" >&2
  stop_vram_sampler
  release_cards "boot D: server nicht oben, verbraucht"
  exit 1
fi

# Two arms, one boot. The driver's --rollback-arms switch drives draft_rollback;
# the re-seed arm is driven the same way through the job body, so the runs are
# issued explicitly here rather than through that flag.
"$VENV/bin/python" - <<'PY' | tee "$OUT/reseed.txt"
import json, os, sys
sys.path.insert(0, os.environ["WT"] + "/scripts/dual_group")
from lane_accept_probe import PROMPTS, lane_run, tokenize

BASE = "http://127.0.0.1:" + os.environ["PORT"]
TOKENIZER = os.environ["MODEL_ROOT"] + "/Qwen3.6-27B-MTP-Q3_K_M-GGUF"
report = {"arms": []}
for name in ("squares", "code", "prose"):
    ids = tokenize(BASE, PROMPTS[name], TOKENIZER)
    row = {"prompt": name, "prompt_tokens": len(ids), "arms": {}}
    for reseed in (True, False):
        res = lane_run(BASE, {
            "lane_id": 0,
            "input_ids": ids,
            "max_new_tokens": 192,
            "spec_steps": 3,
            "verify": "target_verify",
            "draft_reseed": reseed,
        })[-1]
        row["arms"][str(reseed)] = {
            "accept_len_mean": res.get("accept_len_mean"),
            "curve": (res.get("accept_positions") or {}).get("rate"),
            "decode_ms_mean": res.get("decode_ms_mean"),
            "head_forwards": res.get("head_forwards"),
            "reseed_forwards": res.get("reseed_forwards"),
            "spec_rounds": res.get("spec_rounds"),
            "output_ids": res.get("output_ids"),
        }
    a, b = row["arms"]["True"], row["arms"]["False"]
    row["output_identical"] = a["output_ids"] == b["output_ids"]
    print(f"== {name}: reseed accept {a['accept_len_mean']} "
          f"({a['reseed_forwards']} reseed fwd, {a['decode_ms_mean']} ms) "
          f"| ohne {b['accept_len_mean']} ({b['decode_ms_mean']} ms) "
          f"| output identisch: {row['output_identical']}")
    report["arms"].append(row)
json.dump(report, open(os.environ["OUT"] + "/reseed.json", "w"), indent=2)
PY

curl -sf "http://127.0.0.1:$PORT/get_server_info" > "$OUT/server_info.json"

kill "$(cat /tmp/r7c-boot-d.pid)" 2>/dev/null
stop_vram_sampler
summarise_vram "$OUT/vram.csv" | tee "$OUT/vram_summary.txt"
release_cards "boot D DONE, Rohdaten in $OUT"
echo "fertig: $OUT"
