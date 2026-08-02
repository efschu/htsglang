#!/usr/bin/env bash
# LIVE handover byte gate (#261 second half) -- the card-window sequence.
#
# Precondition: BOTH servers are already up and stay up the whole time:
#   source A (TP=1) on $PORT_A with file store $STORE_A,
#   destination B on $PORT_B with file store $STORE_B,
# same checkpoint, same dtype/kv-dtype (the manifest's identity gate will
# refuse anything else). GPU window rules apply: arbitrate via
# /spinning/gpu-arb/, own PIDs only, bounded runs.
#
# Order of proof (A-vs-A floor FIRST, then the cross-server claim):
#   1. seed the session on A
#   2. reference continuation on A, twice -> must be byte-identical
#      (if this floor fails, STOP: nothing cross-server can be claimed)
#   3. export on A (prefix parked), prove the park + both servers' liveness
#   4. manifest-scoped umsharder into B's store (source store is LIVE;
#      the manifest is what makes that safe)
#   5. verify_import on B, commit on A
#   6. continue on B with --expect-cached -> compare byte-for-byte vs ref
set -euo pipefail

PORT_A="${PORT_A:?source server port}"
PORT_B="${PORT_B:?destination server port}"
STORE_A="${STORE_A:?source hicache file store dir}"
STORE_B="${STORE_B:?destination hicache file store dir}"
TOKENIZER="${TOKENIZER:?tokenizer path}"
STATE="${STATE:-/tmp/live_handover_state.json}"
MANIFEST="${MANIFEST:-/tmp/live_handover_manifest.json}"
# Destination geometry for the umsharder (forward 1 -> N):
TARGET_TP="${TARGET_TP:?destination tp size}"
TARGET_RATIOS="${TARGET_RATIOS:?destination resolved --rank-tp-ratio vector}"
MODEL_CONFIG="${MODEL_CONFIG:?model config.json (GDN layout)}"
NUM_LINEAR_LAYERS="${NUM_LINEAR_LAYERS:?}"
GDN_UNITS="${GDN_UNITS:?}"
PY="${PY:-python3}"
PROBE="$(dirname "$0")/live_handover_probe.py"

rm -f "$STATE"

echo "== 1. seed on A =="
"$PY" "$PROBE" seed --port "$PORT_A" --tokenizer "$TOKENIZER" --state "$STATE"

echo "== 2. A-vs-A floor =="
"$PY" "$PROBE" continue --port "$PORT_A" --state "$STATE" --label ref
"$PY" "$PROBE" continue --port "$PORT_A" --state "$STATE" --label ref2
"$PY" "$PROBE" compare --state "$STATE" --label ref --other-label ref2

echo "== 3. export on A (park) =="
"$PY" "$PROBE" export --port "$PORT_A" --state "$STATE"
"$PY" - "$STATE" "$MANIFEST" <<'EOF'
import json, sys
state = json.load(open(sys.argv[1]))
json.dump(state["manifest"], open(sys.argv[2], "w"))
EOF
"$PY" "$PROBE" parked --port "$PORT_A" --state "$STATE"
"$PY" "$PROBE" liveness --port "$PORT_A" --tokenizer "$TOKENIZER" --state "$STATE"
"$PY" "$PROBE" liveness --port "$PORT_B" --tokenizer "$TOKENIZER" --state "$STATE"

echo "== 4. manifest-scoped umsharder (source store stays LIVE) =="
"$PY" -m sglang.srt.mem_cache.hicache_migrate \
  --source-dir "$STORE_A" --target-dir "$STORE_B" \
  --manifest "$MANIFEST" \
  --target-tp-size "$TARGET_TP" --target-ratios "$TARGET_RATIOS" \
  --model-config "$MODEL_CONFIG" \
  --num-linear-layers "$NUM_LINEAR_LAYERS" --gdn-units "$GDN_UNITS" \
  --draft-cold-start \
  --verify

echo "== 5. verify_import on B, commit on A =="
"$PY" "$PROBE" verify --port "$PORT_B" --state "$STATE"
"$PY" "$PROBE" commit --port "$PORT_A" --state "$STATE"

echo "== 6. resume on B, byte gate =="
"$PY" "$PROBE" continue --port "$PORT_B" --state "$STATE" --label migrated \
  --expect-cached
"$PY" "$PROBE" compare --state "$STATE" --label ref --other-label migrated

echo "LIVE HANDOVER GATE PASSED"
