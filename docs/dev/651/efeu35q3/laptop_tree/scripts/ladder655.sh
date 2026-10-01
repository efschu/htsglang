#!/bin/bash
# #655: find the honest prefill depth wall on this iGPU.
#
# Context: a ~10k-token prompt took GPU reset(6) ("device wedged, but recovered
# through reset"), while a 978-token prompt has run repeatedly without one. The
# amdgpu MES fault is documented as INTERMITTENT per launch, so a single
# failure is not a threshold and a single success is not a clearance. This
# ladder climbs in steps, stops at the first wedge, and then retries that same
# depth once -- a second failure at the same depth is what turns "it happened"
# into "it reproduces".
#
# The GPU recovers through the reset by itself; no reboot is involved.
set -u
TS=$(date +%H%M%S)
LOG=/root/651-p2/results/ladder655_${TS}.log
exec > >(tee -a "$LOG") 2>&1

resets() { dmesg -T 2>/dev/null | grep -cE "GPU reset\("; }
mark() { echo "[$(date -Is)] $*"; }

wait_up() {
  for _ in $(seq 1 90); do
    case "$(curl -s -m 8 localhost:31651/ondemand/status)" in
      *'"state": "up"'*) return 0;;
    esac
    sleep 5
  done
  return 1
}

mark "ladder start; resets=$(resets)"

WALL=""
for depth in 1000 2000 4000 6000 8000 10000; do
  R0=$(resets)
  mark "--- depth=$depth (resets before=$R0) ---"
  # A parked service loads on the first request; the probe's own timeout covers
  # the ~191 s cold load, so no separate wake is needed.
  timeout 1800 python3 /root/651-p2/scripts/needle655.py "$depth" \
    "/root/651-p2/results/ladder655_${depth}_${TS}.txt" 2>&1 | \
    grep -E "achieved_prompt_tokens|prefill_wall_s|prefill_tok_s|decode_tok_s|total_wall_s|NEEDLE-PROBE|FOUND|MISSING|Error"
  RC=${PIPESTATUS[0]}
  sleep 8   # let a crashing backend finish writing its reset to dmesg
  R1=$(resets)
  mark "depth=$depth rc=$RC resets after=$R1"

  if [ "$R1" != "$R0" ]; then
    mark "WEDGE at depth=$depth (resets $R0 -> $R1). Retrying this depth once."
    wait_up || mark "service did not come back up before retry"
    R2=$(resets)
    timeout 1800 python3 /root/651-p2/scripts/needle655.py "$depth" \
      "/root/651-p2/results/ladder655_${depth}_retry_${TS}.txt" 2>&1 | \
      grep -E "achieved_prompt_tokens|prefill_wall_s|decode_tok_s|NEEDLE-PROBE|Error"
    RC2=${PIPESTATUS[0]}
    sleep 8
    R3=$(resets)
    if [ "$R3" != "$R2" ]; then
      mark "RETRY ALSO WEDGED at depth=$depth -> reproducible wall"
      WALL="$depth reproducible"
    else
      mark "RETRY SURVIVED at depth=$depth rc=$RC2 -> intermittent, not a hard wall"
      WALL="$depth intermittent"
    fi
    break
  fi
  mark "depth=$depth CLEAN"
done

mark "=== ladder done. wall=${WALL:-none up to 10000} final resets=$(resets) ==="
mark "log=$LOG"
