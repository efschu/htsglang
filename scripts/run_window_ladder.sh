#!/usr/bin/env bash
# WINDOW_LADDER_0818 runner: evaluates the ladder's grep/curl acceptances
# against a LIVE server + boot log and prints a PASS/FAIL/UNOBS table.
#
# It never boots, never kills, never touches the soak driver. Traffic from
# the soak driver is expected and tolerated: every loaded-phase check is a
# log observation, not a latency assertion. The one latency check (#713
# TTFT) runs only in --phase idle, and asserting the router is quiet is the
# OPERATOR's act of choosing that phase.
#
# Usage:
#   run_window_ladder.sh --log /var/log/htsglang/boot.log \
#       [--url http://127.0.0.1:30030] [--phase idle|loaded|all] \
#       [--arm I|II] [--dry-run <fixture-dir>]
#
# Dry-run: <fixture-dir>/log.txt replaces the boot log, curl checks are
# skipped (marked DRY), proving the mechanics without a server.

set -u
URL="http://127.0.0.1:30030"
LOG=""
PHASE="all"
ARM="I"
DRY=""

while [ $# -gt 0 ]; do
  case "$1" in
    --log) LOG="$2"; shift 2 ;;
    --url) URL="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --arm) ARM="$2"; shift 2 ;;
    --dry-run) DRY="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

if [ -n "$DRY" ]; then
  LOG="$DRY/log.txt"
fi
if [ -z "$LOG" ] || [ ! -r "$LOG" ]; then
  echo "FATAL: boot log not readable: '$LOG'" >&2
  exit 2
fi

PASS=0; FAIL=0; UNOBS=0
row() { # status name detail
  case "$1" in
    PASS) PASS=$((PASS+1));;
    FAIL) FAIL=$((FAIL+1));;
    *) UNOBS=$((UNOBS+1));;
  esac
  printf "%-6s %-38s %s\n" "$1" "$2" "$3"
}

# count occurrences of a pattern in the log
lc() { grep -cE "$1" "$LOG" 2>/dev/null || true; }

echo "== WINDOW_LADDER_0818  arm=$ARM phase=$PHASE log=$LOG"
echo

# ---------------- phase 0/boot observations (always evaluated from log)
n=$(lc "images .* host \(file-backed reclaimable\)")
if [ "$n" -ge 1 ]; then row PASS "flip-images-file-backed" "$n boot summary line(s)"; else row FAIL "flip-images-file-backed" "no file-backed summary line"; fi

# ---------------- phase idle
if [ "$PHASE" = "idle" ] || [ "$PHASE" = "all" ]; then
  if [ -n "$DRY" ]; then
    row DRY "713-ttft-idle" "curl skipped in dry-run"
    row DRY "health-generation" "curl skipped in dry-run"
  else
    t0=$(date +%s.%N)
    code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' "$URL/health")
    t1=$(date +%s.%N)
    dt=$(echo "$t1 $t0" | awk '{printf "%.2f", $1-$2}')
    if [ "$code" = "200" ]; then row PASS "health" "200 in ${dt}s"; else row FAIL "health" "code $code"; fi
    if [ "$PHASE" = "idle" ]; then
      t0=$(date +%s.%N)
      body=$(curl -s -m 30 "$URL/generate" -H 'Content-Type: application/json' \
        -d '{"text":"2+2=","sampling_params":{"max_new_tokens":4,"temperature":0}}')
      t1=$(date +%s.%N)
      dt=$(echo "$t1 $t0" | awk '{printf "%.2f", $1-$2}')
      ok=$(echo "$body" | grep -c '"text"' || true)
      if [ "$ok" -ge 1 ] && awk "BEGIN{exit !($dt < 3.0)}"; then
        row PASS "713-ttft-idle" "${dt}s < 3s (OPERATOR asserted quiet router by choosing --phase idle)"
      elif [ "$ok" -ge 1 ]; then
        row FAIL "713-ttft-idle" "${dt}s >= 3s"
      else
        row FAIL "713-ttft-idle" "no generation"
      fi
    else
      row UNOBS "713-ttft-idle" "needs --phase idle (quiet router)"
    fi
  fi
fi

# ---------------- phase loaded: log-observation checks (soak-tolerant)
if [ "$PHASE" = "loaded" ] || [ "$PHASE" = "all" ]; then
  # negative checks: pattern must be ABSENT
  while IFS='|' read -r name pat; do
    [ -z "$name" ] && continue
    n=$(lc "$pat")
    if [ "$n" -eq 0 ]; then row PASS "$name" "0 occurrences"; else row FAIL "$name" "$n occurrence(s)"; fi
  done <<'NEG'
757-race-holds|#631 PROXY
748-no-provider|no KV provider
748-idle-lock|IDLE-LOCKED: no batch of either work class
gate-c-crash|Scheduler hit an exception
NEG
  # positive checks: pattern must be PRESENT
  while IFS='|' read -r name pat; do
    [ -z "$name" ] && continue
    n=$(lc "$pat")
    if [ "$n" -ge 1 ]; then row PASS "$name" "$n line(s)"; else row FAIL "$name" "0 lines"; fi
  done <<'POS'
744-717-rung-funded|KV backing relief returned [0-9]+ MiB.*before the gate
690-refill-census|refill_highwater
POS
  # vacuous relief: allowed only if that direction never commits; simple form:
  n=$(lc "relief returned NOTHING before the gate")
  if [ "$n" -eq 0 ]; then row PASS "748-vacuous-relief" "0 occurrences"; else row FAIL "748-vacuous-relief" "$n occurrence(s) (check whether that direction committed)"; fi
  # host-tier resume (WT_745 line 3 / 758-2)
  n=$(lc "load_back|host.*resume|mamba_host_hit")
  if [ "$n" -ge 1 ]; then row PASS "758-2-mamba-host-resume" "$n line(s)"; else row UNOBS "758-2-mamba-host-resume" "no resume line (needs host-backed re-hit under load)"; fi
  # anchor cadence (ARM II only; emitters may not exist yet -> UNOBS not FAIL)
  if [ "$ARM" = "II" ]; then
    n=$(lc "anchor")
    if [ "$n" -ge 1 ]; then row PASS "758-1-anchor-cadence" "$n line(s) -- verify 8192-multiples by hand"; else row UNOBS "758-1-anchor-cadence" "0 lines: emitters missing (F4-r5 building) or grid silent-inert"; fi
  fi
  # 602 placeholder: owner fills the grep
  row UNOBS "602-instrument-lines" "slot reserved; owner supplies the grep patterns"
fi

# ---------------- ARM III: the 735 step-2 gapped boot (WINDOW_TICKET_735_STEP2)
if [ "$ARM" = "III" ]; then
  # A: gapped set accepted
  n=$(lc "PPLayerSetError")
  if [ "$n" -eq 0 ]; then row PASS "735s2-A-set-accepted" "no PPLayerSetError"; else row FAIL "735s2-A-set-accepted" "$n refusal(s)"; fi
  # B: crossing count EXACTLY 31
  cnt=$(grep -oE "crossings?[ =:]+[0-9]+" "$LOG" | grep -oE "[0-9]+" | tail -1)
  if [ "${cnt:-x}" = "31" ]; then row PASS "735s2-B-crossings" "31 exactly"; elif [ -z "$cnt" ]; then row UNOBS "735s2-B-crossings" "no crossing-count line (emitter?)"; else row FAIL "735s2-B-crossings" "counted $cnt, expected exactly 31 -- wrong map, abort window"; fi
  # D: the deliberate refusal probes (operator runs them pre-boot; lines land in the same log)
  n=$(lc "must be CONTIGUOUS")
  if [ "$n" -ge 1 ]; then row PASS "735s2-D-refusal-fired" "$n contiguity refusal(s) observed"; else row UNOBS "735s2-D-refusal-fired" "refusal probe not run in this log"; fi
  # G: no 737 wedge
  n=$(lc "ack-wait|ACK WAIT HANG")
  if [ "$n" -eq 0 ]; then row PASS "735s2-G-no-737-wedge" "0 ack hangs"; else row FAIL "735s2-G-no-737-wedge" "$n"; fi
  # H: GDN slot ceiling readout (record, never fail)
  slots=$(grep -oE "mamba slots?[ =:]+[0-9]+" "$LOG" | grep -oE "[0-9]+" | tail -1)
  if [ -n "$slots" ]; then row PASS "735s2-H-slot-readout" "recorded $slots slots (compare vs 20-23 with arm C / ~7 without)"; else row UNOBS "735s2-H-slot-readout" "no slot line"; fi
  row UNOBS "735s2-C-text-identity" "operator row: greedy probe vs step-1 reference outputs"
  row UNOBS "735s2-E-throughput" "operator row: soak-load comparison + #706 across-a-flip cached-token form"
fi

echo
echo "== summary: PASS=$PASS FAIL=$FAIL UNOBS/DRY=$UNOBS"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
