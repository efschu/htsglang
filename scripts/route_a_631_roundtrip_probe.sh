#!/usr/bin/env bash
# #631: the PP -> TP -> PP round trip, as one unattended sequence.
#
# WHY THIS EXISTS. The proxy reproducer only ever ABANDONS, so it exercises
# the armed window and nothing after it. The two defects that actually kill
# the instance live on the other side of the cutover:
#
#   * defect Q (slot index): the ranks leave an armed window on different
#     microbatch slots and every later proxy is mispaired;
#   * defect Q's sibling (cadence counter): the ranks leave an armed window
#     incongruent modulo the consensus interval, and the FIRST periodic
#     consensus in the TP phase deadlocks -- rank 0 in the reduction, its
#     peers in the broadcast rank 0 owes them. Measured 2026-08-09
#     08:09:39Z, 120 s no progress, then SIGQUIT.
#
# So the sequence deliberately runs armed windows FIRST (to create the
# divergence both defects feed on), then commits, then serves, then WAITS
# past the collective's own 120 s bound, then flips back. A pass means the
# instance survived the state its own armed windows put it in.
set -uo pipefail

PORT="${PORT:-30030}"
LOG="${LOG:-/spinning/serving-30030.boot.log}"
OUT="${OUT:-/tmp/route-a-631/roundtrip.txt}"
# Longer than the barlink bound (120 s), or the deadlock this probes for
# would simply not have had time to fire.
IDLE_SOAK_S="${IDLE_SOAK_S:-150}"

say() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$OUT"; }
: > "$OUT"

gen() {  # gen <label> <prompt> <max_tokens>
  local label="$1" prompt="$2" n="$3" body
  body=$(curl -s -m 120 -X POST "http://127.0.0.1:$PORT/generate" \
    -H 'Content-Type: application/json' \
    -d "{\"text\":$(printf '%s' "$prompt" | /spinning/htsglang-gpu/.venv/bin/python -c 'import json,sys;print(json.dumps(sys.stdin.read()))' 2>/dev/null || echo "\"$prompt\""),\"sampling_params\":{\"max_new_tokens\":$n,\"temperature\":0}}" 2>&1)
  if [ -z "$body" ]; then
    say "$label: NO RESPONSE (timeout or dead)"
    return 1
  fi
  say "$label: ${body:0:200}"
  return 0
}

flip() {  # flip <direction>
  local dir="$1" r
  r=$(curl -s -m 30 -X POST "http://127.0.0.1:$PORT/phase_flip" \
    -H 'Content-Type: application/json' -d "{\"direction\":\"$dir\"}")
  say "arm $dir -> ${r:0:160}"
}

health() { curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health"; }

W=$(wc -c < "$LOG")
say "watermark $W"

say "STEP 1: armed windows, to create the divergence the defects feed on"
/spinning/htsglang-gpu/.venv/bin/python scripts/route_a_631_proxy_strand_repro.py \
  --cycles 6 --arm-period 7 --warmup 15 --out /tmp/route-a-631/roundtrip_repro.json \
  >> "$OUT" 2>&1
say "STEP 1 repro exit=$?"

say "STEP 2: commit a flip on an idle server"
flip pp_to_tp
sleep 15
say "cutovers so far: $(tail -c +$W "$LOG" | grep -ac 'cutover complete')"

say "STEP 3: serve in the TP phase"
gen "TP generate" "What is 17 times 23? Answer with the number only." 32

say "STEP 4: idle soak ${IDLE_SOAK_S}s -- longer than the 120 s collective bound"
for i in $(seq 1 $((IDLE_SOAK_S / 15))); do
  sleep 15
  h=$(health)
  say "  soak ${i}: health=$h"
  [ "$h" = "200" ] || { say "DIED DURING SOAK"; break; }
done

say "STEP 5: serve again, then the return leg"
gen "TP generate 2" "Name three primary colours." 32
flip tp_to_pp
sleep 20
say "cutovers total: $(tail -c +$W "$LOG" | grep -ac 'cutover complete')"
gen "PP generate" "What is 8 plus 5? Answer with the number only." 32

say "SUMMARY"
say "  health          $(health)"
say "  cutovers        $(tail -c +$W "$LOG" | grep -ac 'cutover complete')"
say "  slot AGREED     $(tail -c +$W "$LOG" | grep -ac 'RESUME SLOTS .* -- AGREED')"
say "  slot DIVERGED   $(tail -c +$W "$LOG" | grep -ac 'RESUME SLOTS .* -- DIVERGED')"
say "  proxy refusals  $(tail -c +$W "$LOG" | grep -ac 'LEFTOVER REFUSED')"
say "  collective TO   $(tail -c +$W "$LOG" | grep -ac 'made no progress for')"
say "  SIGQUIT/Fatal   $(tail -c +$W "$LOG" | grep -acE 'SIGQUIT|Fatal Python')"
