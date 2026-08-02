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
# Both servers need, at minimum:
#   --page-size 1 --enable-hierarchical-cache
#   --hicache-storage-backend file --hicache-write-policy write_through
#   --hicache-mem-layout page_first_direct
# and SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR pointing at their store. A
# hybrid-GDN model has no choice about the layout: MambaPoolHost accepts
# page_first_direct only.
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

# The seed must be long enough that the DESTINATION will prefetch the prefix
# from storage at all: UnifiedRadixCache.prefetch_threshold is 256 tokens, and
# on a hybrid-GDN model the handover length is clamped further, to the deepest
# GDN checkpoint. A short seed makes step 6 fail with cached_tokens == 0 for a
# reason that has nothing to do with the handover.
SEED_PROMPT_TOKENS="${SEED_PROMPT_TOKENS:-1200}"
SEED_NEW_TOKENS="${SEED_NEW_TOKENS:-64}"

# Itemsize of ONE element of the GDN recurrent state. The CLI default is 2,
# but a bf16 Qwen3.5 checkpoint keeps the temporal (SSM) state in fp32 while
# the conv state stays bf16 -- so the default aborts the umsharder with a size
# mismatch naming both numbers. Check against the real blob before changing.
TEMPORAL_ITEMSIZE="${TEMPORAL_ITEMSIZE:-4}"
CONV_ITEMSIZE="${CONV_ITEMSIZE:-2}"

# KV page naming. In dcp_owner_mode (weighted uneven DCP on the destination)
# pages are rank-shared and carry no rank suffix; a plain boot writes and reads
# per-rank names instead. Getting this wrong migrates the pages under names the
# destination never looks up: verify_import passes, and step 6 then silently
# re-prefills. Default here is the plain boot, because that is what a TP=1
# destination is.
DCP_OWNER_MODE="${DCP_OWNER_MODE:-0}"
DCP_FLAG=()
if [ "$DCP_OWNER_MODE" != "1" ]; then
  DCP_FLAG=(--no-dcp-owner-mode)
fi

PY="${PY:-python3}"
PROBE="$(dirname "$0")/live_handover_probe.py"

rm -f "$STATE"

echo "== 1. seed on A =="
"$PY" "$PROBE" seed --port "$PORT_A" --tokenizer "$TOKENIZER" --state "$STATE" \
  --seed-prompt-tokens "$SEED_PROMPT_TOKENS" \
  --seed-new-tokens "$SEED_NEW_TOKENS"

echo "== 2. A-vs-A floor =="
"$PY" "$PROBE" continue --port "$PORT_A" --state "$STATE" --label ref
"$PY" "$PROBE" continue --port "$PORT_A" --state "$STATE" --label ref2
"$PY" "$PROBE" compare --state "$STATE" --label ref --other-label ref2

echo "== 3. export on A (park) =="
"$PY" "$PROBE" export --port "$PORT_A" --state "$STATE"
"$PY" - "$STATE" "$MANIFEST" <<'EOF'
import json, sys
state = json.load(open(sys.argv[1]))
manifest = state["manifest"]
json.dump(manifest, open(sys.argv[2], "w"))
# The #212 gate is the POINT of this run: on a hybrid model a manifest without
# a GDN blob means the recurrent state never travels and the destination
# re-prefills a wrong session. Refuse to continue the gate on a silent miss.
if manifest["hybrid_gdn"] and not manifest["mamba_key"]:
    raise SystemExit(
        "manifest declares hybrid_gdn but names no GDN blob -- the source "
        "gate should have refused this export (#212)"
    )
print(
    f"manifest: hybrid_gdn={manifest['hybrid_gdn']} "
    f"kv={len(manifest['kv_keys'])} "
    f"gdn={'present' if manifest['mamba_key'] else 'ABSENT'}"
)
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
  --temporal-itemsize "$TEMPORAL_ITEMSIZE" --conv-itemsize "$CONV_ITEMSIZE" \
  "${DCP_FLAG[@]}" \
  --draft-cold-start \
  --verify

echo "== 5. verify_import on B, commit on A =="
"$PY" "$PROBE" verify --port "$PORT_B" --state "$STATE"
"$PY" "$PROBE" commit --port "$PORT_A" --state "$STATE"

echo "== 6. resume on B, byte gate =="
# --expect-cached is load-bearing: without it a destination that re-prefilled
# the whole prefix reproduces the same tokens (same model, greedy) and the
# comparison below would "pass" while proving nothing.
"$PY" "$PROBE" continue --port "$PORT_B" --state "$STATE" --label migrated \
  --expect-cached
"$PY" "$PROBE" compare --state "$STATE" --label ref --other-label migrated

echo "LIVE HANDOVER GATE PASSED"
