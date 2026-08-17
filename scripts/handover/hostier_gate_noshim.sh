#!/usr/bin/env bash
# #261 / #441(3): HiCache HOST-TIER handover gate, WITHOUT the umsharder.
# FILED FOR A CARD WINDOW -- not run at authoring time.
#
# ---------------------------------------------------------------------------
# WHY THIS SCRIPT IS SAME-GEOMETRY ONLY, WHICH IS THE WHOLE POINT
# ---------------------------------------------------------------------------
#
# The brief asked for "the #261 gate short-run WITHOUT the harness shim".
# Verifying at the code first, as required: the step in question is step 4 of
# live_handover_gate.sh,
#
#     python -m sglang.srt.mem_cache.hicache_migrate \
#         --source-dir A --target-dir B --manifest ... \
#         --target-tp-size N --target-ratios ... \
#         --num-linear-layers ... --gdn-units ... \
#         --temporal-itemsize ... --conv-itemsize ...
#
# That is NOT a harness shim. It is LOAD-BEARING, for two independent reasons:
#
#   1. GEOMETRY. The live gate runs source A at TP=1 and destination B at
#      TP=N ("Destination geometry for the umsharder (forward 1 -> N)",
#      live_handover_gate.sh:37). KV pages and GDN state blobs are written in
#      the SOURCE's shard geometry. Without the conversion, B is not reading a
#      host tier "the hard way" -- it is being handed bytes in a layout it
#      cannot interpret. A no-shim 1->N run does not test the host tier; it
#      tests nothing, and would fail for a reason unrelated to the claim.
#
#   2. IT IS THE FEATURE. The cross-geometry handover IS the umsharder.
#      Removing it removes the thing under test.
#
# So a no-shim run of the gate AS WRITTEN is structurally meaningless, and
# this script does not attempt one.
#
# What IS meaningful, and what this script does: hold the geometry EQUAL and
# drop the migration. Then no conversion is needed by construction, and what
# remains under test is exactly the HiCache host tier end to end -- write
# through to the store on A, park/export, prefetch on B, resume from cache,
# byte-identical continuation. That is a weaker claim than the 1->N gate and a
# real one, and it is the claim the host tier alone can support.
#
# The preflight below REFUSES to run when the geometries differ, rather than
# producing a red that a reader could mistake for a host-tier finding. That
# refusal is the safety property of this script.
#
# ---------------------------------------------------------------------------
# PRECONDITIONS
# ---------------------------------------------------------------------------
# Both servers up for the whole run, SAME checkpoint, SAME dtype/kv-dtype, and
# SAME parallel geometry (tp size and --rank-tp-ratio vector). Both need:
#   --page-size 1 --enable-hierarchical-cache
#   --hicache-storage-backend file --hicache-write-policy write_through
#   --hicache-mem-layout page_first_direct
# with SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR pointing at their store.
# A hybrid-GDN model has no choice about the layout: MambaPoolHost accepts
# page_first_direct only.
#
# STORE_B may be the SAME directory as STORE_A (shared tier -- the strongest
# form), or a byte copy of it. It must NOT be produced by hicache_migrate:
# that would put the shim back in.
#
# GPU window rules apply: arbitrate via /spinning/gpu-arb/, own PIDs only,
# bounded run.
#
# Usage:
#   PORT_A=30040 PORT_B=30041 STORE_A=/tmp/hc_a STORE_B=/tmp/hc_a \
#   TOKENIZER=/path/to/tokenizer ./scripts/handover/hostier_gate_noshim.sh

set -euo pipefail

PORT_A="${PORT_A:?source server port}"
PORT_B="${PORT_B:?destination server port}"
STORE_A="${STORE_A:?source hicache file store dir}"
STORE_B="${STORE_B:?destination hicache file store dir}"
TOKENIZER="${TOKENIZER:?tokenizer path}"
STATE="${STATE:-/tmp/hostier_noshim_state.json}"

# Above UnifiedRadixCache.prefetch_threshold (256), or the destination never
# prefetches from storage at all and step 5 fails with cached_tokens == 0 for a
# reason that has nothing to do with the host tier. Short-run default, kept
# well clear of the threshold rather than near it.
SEED_PROMPT_TOKENS="${SEED_PROMPT_TOKENS:-600}"
SEED_NEW_TOKENS="${SEED_NEW_TOKENS:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"

PY="${PY:-/spinning/htsglang-gpu/.venv/bin/python}"
PROBE="$(dirname "$0")/live_handover_probe.py"

echo "== 0. PREFLIGHT: the geometries must match, or this run proves nothing =="
geom() {
  curl -sf -m 10 "http://127.0.0.1:$1/get_server_info" \
    | "$PY" -c 'import json,sys; d=json.load(sys.stdin); print(d.get("tp_size"), d.get("rank_tp_ratio"), d.get("model_path"), d.get("kv_cache_dtype"))'
}
GEOM_A="$(geom "$PORT_A")" || { echo "cannot read server info on A"; exit 2; }
GEOM_B="$(geom "$PORT_B")" || { echo "cannot read server info on B"; exit 2; }
echo "  A: $GEOM_A"
echo "  B: $GEOM_B"
if [[ "$GEOM_A" != "$GEOM_B" ]]; then
  cat >&2 <<'REFUSED'
REFUSED: the two servers do not share a geometry/checkpoint.

Without the umsharder, a destination with a different shard geometry cannot
interpret the source's stored pages at all, so this run would fail for a
reason that says nothing about the HiCache host tier. That failure would look
like a host-tier result and would not be one.

Use scripts/handover/live_handover_gate.sh for cross-geometry handover -- its
hicache_migrate step is exactly the conversion this configuration needs, and
it is load-bearing rather than a harness convenience.
REFUSED
  exit 3
fi

if [[ "$STORE_A" != "$STORE_B" ]]; then
  echo "  NOTE: distinct stores; STORE_B must be a byte copy of STORE_A and"
  echo "        must NOT have been produced by hicache_migrate."
fi

echo "== 1. seed the session on A =="
"$PY" "$PROBE" seed --port "$PORT_A" --tokenizer "$TOKENIZER" --state "$STATE" \
  --seed-prompt-tokens "$SEED_PROMPT_TOKENS" --seed-new-tokens "$SEED_NEW_TOKENS"

echo "== 2. A-vs-A floor: same continuation twice on A, byte-identical =="
# FIRST, and the run stops here if it fails. A cross-server byte claim on a
# rig whose own repeat is not byte-identical is not a claim about handover.
"$PY" "$PROBE" continue --port "$PORT_A" --state "$STATE" --label ref \
  --max-new-tokens "$MAX_NEW_TOKENS"
"$PY" "$PROBE" continue --port "$PORT_A" --state "$STATE" --label ref2 \
  --max-new-tokens "$MAX_NEW_TOKENS"
"$PY" "$PROBE" compare --state "$STATE" --label ref --other-label ref2

echo "== 3. export on A (prefix parked), prove the park and both liveness =="
"$PY" "$PROBE" export --port "$PORT_A" --state "$STATE"
"$PY" "$PROBE" parked --port "$PORT_A" --state "$STATE"
"$PY" "$PROBE" liveness --port "$PORT_A" --state "$STATE"
"$PY" "$PROBE" liveness --port "$PORT_B" --state "$STATE"

echo "== 4. NO MIGRATION STEP. This is the whole difference from the #261 gate. =="
echo "     B reads the host tier directly; geometry equality (step 0) is what"
echo "     makes that legitimate."

echo "== 5. verify_import on B, commit on A =="
"$PY" "$PROBE" verify --port "$PORT_B" --state "$STATE"
"$PY" "$PROBE" commit --port "$PORT_A" --state "$STATE"

echo "== 6. resume on B from the host tier, byte gate =="
# --expect-cached is load-bearing: without it a destination that simply
# re-prefilled the whole prefix reproduces the same tokens (same model,
# greedy) and the comparison below "passes" while proving nothing. That was
# observed for real on the #261 gate before it was added.
"$PY" "$PROBE" continue --port "$PORT_B" --state "$STATE" --label hostier \
  --max-new-tokens "$MAX_NEW_TOKENS" --expect-cached
"$PY" "$PROBE" compare --state "$STATE" --label ref --other-label hostier

echo "HOST-TIER HANDOVER GATE (NO UMSHARDER, SAME GEOMETRY) PASSED"
