#!/bin/bash
# Task #103 per-arm measurement protocol.
#
# Enforces the methodology gate established by the A-vs-A control:
#   * clock pinning is IMPOSSIBLE on this rig (driver refuses -lgc on these
#     GeForce cards even as root), so thermal state is controlled by protocol:
#     a mandatory warm-up burn discards the transient, in which GPU2 (3080)
#     drifts 1875 -> 1637 MHz / 75 -> 88 C over ~3 minutes.
#   * every block records clock/temp/throttle so a drifted point is
#     identifiable after the fact rather than silently averaged in.
#   * every block records the output text hash + accept length, because the
#     plateau control measured r(tok/s, accept) = 0.98 -- content variance,
#     not hardware, dominates the residual end-to-end scatter.
#
# Usage: arm_measure.sh <tag> <port> [blocks_single] [blocks_dual]
#
# VENV comes from the environment (source your local rig env file); its
# fallback is a placeholder, so an unsourced run fails on a visibly bogus
# interpreter path instead of using somebody else's.
set -u
TAG="$1"
PORT="${2:-30000}"
NB="${3:-6}"
ND="${4:-4}"
PY="${VENV:-<VENV>}/bin/python"
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

echo "=== arm $TAG: capacity line ==="
curl -s "http://127.0.0.1:$PORT/get_server_info" \
  | $PY -c 'import json,sys; d=json.load(sys.stdin); print("max_total_num_tokens =", d.get("max_total_num_tokens"))'

echo "=== arm $TAG: warm-up burn (4 blocks, DISCARDED) ==="
$PY noise_floor.py "$PORT" "warm_${TAG}" 4 code 2>&1 | tail -4

for c in code prosa misch; do
  echo "=== arm $TAG: single / $c ==="
  $PY noise_floor.py "$PORT" "${TAG}_single_${c}" "$NB" "$c" 2>&1 | tail -6
done

for c in code prosa misch; do
  echo "=== arm $TAG: dual / $c ==="
  NF_DUAL=1 $PY noise_floor.py "$PORT" "${TAG}_dual_${c}" "$ND" "$c" 2>&1 | tail -6
done
echo "=== arm $TAG done ==="
