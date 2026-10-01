#!/bin/bash
# #655: the N=3 bar. Three back-to-back 10k needle probes, warm.
#
# One clean pass proves nothing against an intermittent firmware fault -- that
# lesson was paid for: a 1k probe was clean at 06:04 and the identical probe
# wedged at 06:20. So this runs three, records the reset counter around EACH,
# and reports how many were clean rather than a single verdict.
#
# The service is warmed first so the ~195 s cold load is not counted as prefill.
set -u
TS=$(date +%H%M%S)
LOG=/root/651-p2/results/probe3x655_${TS}.log
exec > >(tee -a "$LOG") 2>&1

resets() { dmesg -T 2>/dev/null | grep -cE "GPU reset\("; }
mark() { echo "[$(date -Is)] $*"; }

mark "=== N=3 10k probe run ==="
mark "kernel=$(uname -r) cwsr_enable=$(cat /sys/module/amdgpu/parameters/cwsr_enable 2>/dev/null)"
mark "MES fw: $(grep -E '^MES ' /sys/kernel/debug/dri/*/amdgpu_firmware_info 2>/dev/null | head -1)"
mark "CTX: $(systemctl show htsglang-ondemand -p Environment | tr ' ' '\n' | grep '^CTX=')"
mark "resets at start: $(resets)"

mark "warming (cold load ~195 s)"
curl -sS -m 1800 -X POST localhost:31651/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen36-35b-a3b","messages":[{"role":"user","content":"What is 17 times 23? Answer with just the number."}],"max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  | head -c 200
echo
mark "warm. status: $(curl -s -m 10 localhost:31651/ondemand/status)"

CLEAN=0
for run in 1 2 3; do
  R0=$(resets)
  mark "--- run $run/3 (resets before=$R0) ---"
  timeout 1800 python3 /root/651-p2/scripts/needle655.py 10000 \
    "/root/651-p2/results/probe3x655_run${run}_${TS}.txt" 2>&1 | \
    grep -E "achieved_prompt_tokens|prefill_wall_s|prefill_tok_s|decode_tok_s|total_wall_s|NEEDLE-PROBE|FOUND|MISSING|HTTPError|URLError"
  RC=${PIPESTATUS[0]}
  sleep 8
  R1=$(resets)
  if [ "$R1" != "$R0" ]; then
    mark "run $run WEDGED (resets $R0 -> $R1) rc=$RC"
  elif [ "$RC" = "0" ]; then
    mark "run $run CLEAN rc=0 (resets steady at $R1)"
    CLEAN=$((CLEAN+1))
  else
    mark "run $run FAILED WITHOUT WEDGE rc=$RC (resets steady at $R1)"
  fi
  # If the backend died the service parks; the next probe's own timeout covers
  # the reload, so the loop keeps going rather than aborting on the first loss.
done

mark "=== RESULT: $CLEAN/3 clean 10k probes, final resets=$(resets) ==="
[ "$CLEAN" = "3" ] && mark "N=3 BAR: MET" || mark "N=3 BAR: NOT MET"
mark "log=$LOG"
