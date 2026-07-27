#!/usr/bin/env bash
# Round-trip byte gate for the HiCache geometry umsharder (#261).
#
# TP=1 store -> TP=3 uneven-DCP -> TP=1, at the REAL blob sizes of both model
# geometries, and a byte diff against the original at the end. Needs no GPU, no
# checkpoint and no boot: the migration is a pure byte permutation, so proving
# it needs only files of the right sizes.
#
#   scripts/handover/roundtrip_gate.sh [workdir]
#
# Exits non-zero on the first divergence.
set -euo pipefail

WORK="${1:-/tmp/hicache-roundtrip}"
PY="${VENV:-/spinning/htsglang-gpu/.venv}/bin/python"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$REPO/python:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=99

# The vector `--rank-tp-ratio auto-performance` resolved to on the rig.
RATIOS="29607,17780,17780"

run_arm() {
  local arm="$1" layers="$2" units="$3"
  local a="$WORK/$arm/a" b="$WORK/$arm/b" c="$WORK/$arm/c"
  rm -rf "$WORK/$arm"
  mkdir -p "$a" "$b" "$c"

  # config.json fields the umsharder reads for the GDN state layout.
  if [[ "$arm" == "27b" ]]; then
    cat >"$WORK/$arm/config.json" <<'EOF'
{"linear_num_value_heads": 48, "linear_value_head_dim": 128,
 "linear_key_head_dim": 128, "linear_num_key_heads": 16,
 "linear_conv_kernel_dim": 4}
EOF
  else
    cat >"$WORK/$arm/config.json" <<'EOF'
{"linear_num_value_heads": 32, "linear_value_head_dim": 128,
 "linear_key_head_dim": 128, "linear_num_key_heads": 16,
 "linear_conv_kernel_dim": 4}
EOF
  fi

  echo "=== $arm: build TP=1 store"
  "$PY" "$REPO/scripts/handover/roundtrip_store.py" "$arm" "$a"

  echo "=== $arm: forward TP=1 -> TP=3"
  "$PY" -m sglang.srt.mem_cache.hicache_migrate \
    --source-dir "$a" --target-dir "$b" \
    --target-tp-size 3 --target-ratios "$RATIOS" \
    --model-config "$WORK/$arm/config.json" \
    --num-linear-layers "$layers" --gdn-units "$units" --verify

  echo "=== $arm: reverse TP=3 -> TP=1"
  "$PY" -m sglang.srt.mem_cache.hicache_migrate --reverse \
    --source-dir "$b" --target-dir "$c" \
    --source-tp-size 3 --source-ratios "$RATIOS" \
    --model-config "$WORK/$arm/config.json" \
    --num-linear-layers "$layers" --gdn-units "$units" --verify

  echo "=== $arm: byte diff against the original store"
  diff -rq "$a" "$c"
  echo "ROUND TRIP $arm: BYTE-IDENTICAL ($(ls "$c" | wc -l) files)"
}

run_arm 27b 48 8
run_arm moe 30 16
echo "round-trip gate PASSED for both geometries"
