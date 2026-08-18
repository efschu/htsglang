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
#       [--arm I|II] [--leg-seconds 22.5] [--dry-run <fixture-dir>]
#
# --leg-seconds is the measured refill-leg cost of the regime the boot under
# test actually ran. GATE A's hard ceiling is 60/leg_seconds flips per minute,
# so it is NOT a constant: 22.5 s (the file-backed arm, NOTE_690 §1) gives
# 2.67/min, while the pinned regime's ~3.1 s marks line gives ~19/min. Default
# 22.5 matches the arm this ladder was written for; pass the boot's own number
# when it differs, and the runner reprices the ceiling rather than baking one
# regime into the bar.
#
# Dry-run: <fixture-dir>/log.txt replaces the boot log, curl checks are
# skipped (marked DRY), proving the mechanics without a server.

set -u
URL="http://127.0.0.1:30030"
LOG=""
PHASE="all"
ARM="I"
DRY=""
LEG_S="22.5"

while [ $# -gt 0 ]; do
  case "$1" in
    --log) LOG="$2"; shift 2 ;;
    --url) URL="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --arm) ARM="$2"; shift 2 ;;
    --leg-seconds) LEG_S="$2"; shift 2 ;;
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

# effective value of a server_args field, read off the boot's own
# `server_args=ServerArgs(...)` line. The point of reading it back rather than
# trusting the command line is server_args' silent rewrites -- see the #760
# rewrite trap in WINDOW_LADDER_0818.md.
sa() { grep -m1 -oE "$1=[^,)]*" "$LOG" 2>/dev/null | head -1 | cut -d= -f2- || true; }

echo "== WINDOW_LADDER_0818  arm=$ARM phase=$PHASE log=$LOG"
echo

# ---------------- phase 0/boot observations (always evaluated from log)
n=$(lc "images .* host \(file-backed reclaimable\)")
if [ "$n" -ge 1 ]; then row PASS "flip-images-file-backed" "$n boot summary line(s)"; else row FAIL "flip-images-file-backed" "no file-backed summary line"; fi

# host ledger: the gate decides on EARNED terms. The dirty-page transient is
# reported, not charged (host_ledger.sh:113 hardcodes FLIP_DIRTY_GB=0 after the
# term failed its own discrimination test), so a REFUSE here is a real refusal
# on priced bytes, and the dirty window is a risk note beside it -- never read
# the note as part of the gate.
nref=$(lc "\[HOST-LEDGER [^]]*\].*-> REFUSE")
nok=$(lc "\[HOST-LEDGER [^]]*\].*-> OK")
if [ "$nref" -ge 1 ]; then row FAIL "host-ledger-verdict" "$nref REFUSE line(s) on earned terms"
elif [ "$nok" -ge 1 ]; then row PASS "host-ledger-verdict" "$nok OK line(s); dirty-page transient reported, not charged"
else row UNOBS "host-ledger-verdict" "no HOST-LEDGER pre/post line in this log"; fi

# #760: page_first_direct segfaults inside the copy on this rig with the shape
# guard armed and shapes MATCHING (specimen 2026-08-18T0903Z). Read the
# EFFECTIVE layout, because page_first + direct io is silently rewritten to
# page_first_direct (server_args.py:16464-16471).
layout=$(sa "hicache_mem_layout")
iob=$(sa "hicache_io_backend")
case "$layout" in
  *page_first_direct*) row FAIL "760-layout-gated" "effective layout $layout (io $iob): kernel proven broken on this rig -- boot must not run it" ;;
  "") row UNOBS "760-layout-gated" "no server_args line in this log" ;;
  *) row PASS "760-layout-gated" "effective layout $layout (io $iob): not the broken kernel path" ;;
esac
narm=$(lc "KV-TRANSFER-GUARD ARMED")
nmis=$(lc "KvTransferShapeMismatch")
if [ "$narm" -ge 1 ]; then row PASS "760-guard-armed" "$narm ARMED line(s), $nmis mismatch refusal(s) -- silence without an ARMED line is not evidence"
else row UNOBS "760-guard-armed" "0 ARMED lines: guard inert, absent, or this boot predates 9ba46eb31a"; fi

# ARM II precondition: the flag must be ON the boot. server_args returns early
# when the interval is None, so a forgotten flag is byte-identical to ARM I and
# every ARM II row below would read UNOBS while looking like a measurement.
if [ "$ARM" = "II" ]; then
  ck=$(sa "mamba_checkpoint_interval")
  if [ "$ck" = "8192" ]; then row PASS "armII-precondition-interval" "mamba_checkpoint_interval=8192"
  elif [ -z "$ck" ]; then row UNOBS "armII-precondition-interval" "no server_args line in this log"
  else row FAIL "armII-precondition-interval" "mamba_checkpoint_interval=$ck -- this is NOT ARM II; its rows are void, not failed"; fi
fi

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
  # ---- GATE A, as a RATE. The 135 flips/15 min baseline is retired as a bar
  # (it is 9.0/min, 3.4x above the physical ceiling -- a diagnosis of aborting
  # or overlapping arms, not a target). Bar: sustained < 1/min; hard fail at or
  # above the ceiling 60/leg_seconds, regardless of backlog.
  # Only PP0 emits `PHASE-POLICY arming`, so one line is one arm.
  read -r ARMS WMIN PEAK <<EOF
$(awk -v ts='^\\[[0-9][0-9-]+ [0-9][0-9:]+' '
  function ep(s,  a) { if (match(s, ts) == 0) return -1;
    a = substr(s, RSTART+1, 19); gsub(/[-:]/, " ", a); return mktime(a) }
  { e = ep($0); if (e < 0) next; last = e
    if (first_policy == 0 && /PHASE-POLICY armed:/) first_policy = e
    if (/PHASE-POLICY arming/) { n++; t[n] = e } }
  END {
    win = (first_policy > 0 && last > first_policy) ? (last - first_policy)/60.0 : 0
    peak = 0
    for (i = 1; i <= n; i++) { c = 0
      for (j = i; j <= n; j++) if (t[j] - t[i] < 60) c++
      if (c > peak) peak = c }
    printf "%d %.3f %d", n+0, win, peak }' "$LOG")
EOF
  CEIL=$(awk -v l="$LEG_S" 'BEGIN{ printf "%.2f", (l>0)? 60.0/l : 0 }')
  if [ "${WMIN:-0}" = "0.000" ] || [ "${ARMS:-0}" -eq 0 ]; then
    if [ "${ARMS:-0}" -eq 0 ]; then
      row PASS "gate-a-flip-rate" "0 arming lines (bar: sustained < 1/min; ceiling ${CEIL}/min at ${LEG_S}s/leg)"
    else
      row UNOBS "gate-a-flip-rate" "$ARMS arm(s) but no policy-live window (no 'PHASE-POLICY armed:' line)"
    fi
  else
    RATE=$(awk -v a="$ARMS" -v w="$WMIN" 'BEGIN{printf "%.2f", a/w}')
    DET="$ARMS arm(s) / ${WMIN} min = ${RATE}/min sustained, peak ${PEAK} in any 60s; ceiling ${CEIL}/min at ${LEG_S}s/leg"
    if awk "BEGIN{exit !($RATE >= $CEIL)}"; then
      row FAIL "gate-a-flip-rate" "HARD FAIL (sustained at/above ceiling) regardless of backlog -- $DET"
    elif awk "BEGIN{exit !($PEAK >= $CEIL)}"; then
      row FAIL "gate-a-flip-rate" "HARD FAIL (60s burst at/above ceiling: more arms than the seam can physically complete, so arms aborted or overlapped -- the 135-flip diagnosis) -- $DET"
    elif awk "BEGIN{exit !($RATE < 1.0)}"; then
      row PASS "gate-a-flip-rate" "$DET"
    else
      row FAIL "gate-a-flip-rate" "above the < 1/min bar -- $DET"
    fi
  fi
  # ---- GATE A arm mix: WHICH path churns. Informational, never a FAIL: an
  # economic count at or near zero is EXPECTED on this workload (the soak
  # driver's 48000-char cap keeps total backlog at or below the ~49250-token
  # break-even), and IDLE-LOCK is the one arm that bypasses both the #688
  # layout-hold verdict and the #689 formation gate by construction.
  nidle=$(grep -E "PHASE-POLICY arming" "$LOG" 2>/dev/null | grep -c "IDLE-LOCKED" || true)
  necon=$((${ARMS:-0} - nidle))
  row UNOBS "gate-a-arm-mix" "$nidle idle-lock / $necon economic of ${ARMS:-0} -- 0 economic is expected here, not a defect"
  # ---- cache reuse ACROSS A FLIP (#706), not absolute hit rate: the soak
  # driver truncates to hist[-48000:], so an absolute rate is invalid by
  # construction. Mechanical half only -- the runner cannot resolve session
  # identity, so the operator confirms the request came from a session that
  # already ran a turn BEFORE the flip.
  fl=$(grep -nE "PHASE-POLICY arming" "$LOG" 2>/dev/null | head -1 | cut -d: -f1)
  if [ -z "$fl" ]; then
    row UNOBS "706-cache-across-flip" "no flip in this log"
  else
    nhit=$(tail -n +"$fl" "$LOG" | grep -cE "#cached-token: [1-9]" || true)
    if [ "$nhit" -ge 1 ]; then
      row PASS "706-cache-across-flip" "$nhit post-flip request(s) with cached-token > 0 -- OPERATOR confirms one is a session with a pre-flip turn"
    else
      row FAIL "706-cache-across-flip" "0 post-flip requests with cached-token > 0"
    fi
  fi
  # ---- #743 agent-soak prefix reuse: PREREQUISITE, not a row that can pass.
  row UNOBS "743-slot-eviction-instrument" "PREREQUISITE ABSENT: no discrete slot-eviction event is logged (NOTE_743 2.2); cc4ac02321 is docs-only, the counter at evict_mamba:1161 is proposed, not built"
  # 602 placeholder: owner fills the grep
  row UNOBS "602-instrument-lines" "slot reserved; owner supplies the grep patterns"
fi

echo
echo "== summary: PASS=$PASS FAIL=$FAIL UNOBS/DRY=$UNOBS"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
