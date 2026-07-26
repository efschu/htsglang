#!/bin/bash
# CPU-level tests for the L0 guardrails. NO GPU, NO model, NO sglang: the ranks
# are fakes, so the launcher's CONTROL FLOW is what gets exercised. The boot
# proof belongs to a GPU window; this proves the logic that must hold before one
# is worth spending.
#
# Every process these tests create carries a per-test L0_RUN_TAG and is only
# ever addressed through that tag. Nothing here can touch another agent's
# server -- which is the whole reason the tooling stopped pattern-killing.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/l0_lib.sh"

TMP=$(mktemp -d /tmp/l0guard.XXXXXX)
FAKE="$TMP/fake_rank.sh"
PASS=0; FAIL=0
ok()   { echo "  PASS  $*"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $*"; FAIL=$((FAIL+1)); }

cat > "$FAKE" <<'FAKEEOF'
#!/bin/bash
# Fake rank: same OBSERVABLE contract as l0_rank.sh (stagger marker, then the
# ready line), and supervised the same way so the launcher's process handling is
# exercised for real.
set -u
MODE=${FAKE_MODE:-ready}
sleep "${FAKE_START_DELAY:-1}"
echo "[fake] rank $RANK Init torch distributed begin."
if [ "$MODE" = crash ]; then
  sleep 1
  echo "Traceback (most recent call last):"
  echo "  fake rank $RANK died on purpose"
  exit 1
fi
sleep "${FAKE_LOAD_DELAY:-1}"
if [ "$MODE" = ready ] && [ "$RANK" = 0 ]; then
  echo "[fake] The server is fired up and ready to roll!"
fi
sleep "${FAKE_LIFETIME:-120}" &
CHILD=$!
wait $CHILD
FAKEEOF
chmod +x "$FAKE"

echo "== 1. happy path: staggered starts, READY, ranks survive =="
TAG_BEFORE=$(l0_tagged_pids "none" | wc -l)
out=$(L0_LOCAL_ONLY=1 L0_RANK_SCRIPT="$FAKE" LOGDIR="$TMP/logs1" L0_MIN_FREE_MIB=1 \
      L0_STAGGER_TIMEOUT=30 L0_READY_TIMEOUT=60 FAKE_LIFETIME=25 \
      timeout 180 "$HERE/l0_launch.sh" 2>&1)
rc=$?
echo "$out" | grep -q "^READY$" && [ $rc -eq 0 ] && ok "launcher reports READY (rc=0)" \
    || bad "expected READY/rc=0, got rc=$rc: $(echo "$out" | tail -3)"
# staggering: rank N+1 must start only after rank N printed the marker
s0=$(grep -c "Init torch distributed begin" "$TMP/logs1/r0.log" 2>/dev/null)
[ "${s0:-0}" -ge 1 ] && ok "stagger marker observed in rank 0's log" \
    || bad "no stagger marker in rank 0's log"
l_r1=$(echo "$out" | grep -n "local rank 1 started" | head -1 | cut -d: -f1)
l_m0=$(echo "$out" | grep -n "seen in r0.log" | head -1 | cut -d: -f1)
if [ -n "$l_r1" ] && [ -n "$l_m0" ] && [ "$l_r1" -gt "$l_m0" ]; then
    ok "rank 1 started only AFTER rank 0 reached the marker (staggered)"
else
    bad "rank 1 was not gated on rank 0's marker (r1_line='$l_r1' marker_line='$l_m0')"
fi
echo "$out" | grep -q "MemAvailable" && ok "free-RAM floor was evaluated before each start" \
    || bad "no RAM floor evaluation in output"

echo "== 2. crash: launcher must NOT return while ranks live =="
TAG2=""
out=$(L0_LOCAL_ONLY=1 L0_RANK_SCRIPT="$FAKE" LOGDIR="$TMP/logs2" L0_MIN_FREE_MIB=1 \
      L0_STAGGER_TIMEOUT=30 L0_READY_TIMEOUT=60 FAKE_MODE=crash FAKE_LIFETIME=300 \
      timeout 180 "$HERE/l0_launch.sh" 2>&1)
rc=$?
TAG2=$(echo "$out" | sed -n 's/^l0: run tag \([^ ]*\).*/\1/p' | head -1)
[ $rc -ne 0 ] && ok "crash run exits non-zero (rc=$rc)" || bad "crash run returned 0"
echo "$out" | grep -qE "CRASH_ON_START|CRASHED" && ok "crash is named in the verdict" \
    || bad "no crash verdict: $(echo "$out" | tail -3)"
left=$(l0_tagged_pids "$TAG2" | wc -l)
[ "$left" -eq 0 ] && ok "NO tagged process survives the launcher's return (the container-kill rule)" \
    || bad "$left tagged process(es) still alive after the launcher returned"
echo "$out" | grep -q "all tagged processes gone" && ok "launcher waited for the set to empty" \
    || bad "launcher did not report waiting for the tagged set"

echo "== 3. free-RAM floor refuses to start =="
out=$(L0_LOCAL_ONLY=1 L0_RANK_SCRIPT="$FAKE" LOGDIR="$TMP/logs3" \
      L0_MIN_FREE_MIB=999999999 L0_READY_TIMEOUT=30 \
      timeout 120 "$HERE/l0_launch.sh" 2>&1)
rc=$?
[ $rc -ne 0 ] && echo "$out" | grep -q "REFUSING to start a rank" \
    && ok "RAM floor refuses and explains (rc=$rc)" || bad "RAM floor did not refuse: rc=$rc"
echo "$out" | grep -q "RAM_FLOOR" && ok "verdict names the RAM floor" || bad "no RAM_FLOOR verdict"

echo "== 4. orphan detection (PPID 1) =="
OTAG="l0test-orphan-$$"
env L0_RUN_TAG="$OTAG" setsid --fork sleep 30 >/dev/null 2>&1
sleep 1
opid=$(l0_tagged_pids "$OTAG" | head -1)
if [ -n "$opid" ] && [ "$(l0_ppid_of "$opid")" = "1" ]; then
    ok "built a genuinely orphaned tagged process (pid $opid, PPID 1)"
    if msg=$(l0_check_orphans "$OTAG" 2>&1); then
        bad "l0_check_orphans returned CLEAN for an orphan"
    else
        echo "$msg" | grep -q "ORPHAN: pid $opid has PPID 1" \
            && ok "l0_check_orphans names the pid and the consequence" \
            || bad "orphan message unclear: $msg"
    fi
else
    bad "could not construct an orphan for the test (pid='$opid')"
fi
l0_kill_tagged "$OTAG" 20 >/dev/null 2>&1

echo "== 5. clean set reports clean =="
CTAG="l0test-clean-$$"
env L0_RUN_TAG="$CTAG" sleep 20 &
cpid=$!
sleep 1
l0_check_orphans "$CTAG" >/dev/null 2>&1 && ok "a supervised tagged process is NOT flagged" \
    || bad "false positive: supervised process flagged as orphan"

echo "== 6. tag isolation: a foreign tag is never touched =="
FTAG="l0test-foreign-$$"
env L0_RUN_TAG="$FTAG" sleep 20 &
fpid=$!
sleep 1
l0_kill_tagged "$CTAG" 20 >/dev/null 2>&1
if kill -0 "$fpid" 2>/dev/null; then
    ok "killing tag '$CTAG' left the foreign-tagged process alive (no pattern kill)"
else
    bad "a foreign-tagged process was killed -- tag scoping is broken"
fi
kill "$fpid" 2>/dev/null; kill "$cpid" 2>/dev/null

echo "== 7. l0_rank.sh supervises rather than execs =="
grep -q '^exec .*launch_server' "$HERE/l0_rank.sh" \
    && bad "l0_rank.sh still execs the server (server would be reparented)" \
    || ok "l0_rank.sh does not exec the server"
grep -q 'SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION=1' "$HERE/l0_rank.sh" \
    && ok "SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION is set" || bad "killpg env not set"
grep -q 'REPARENTED TO INIT' "$HERE/l0_rank.sh" \
    && ok "per-rank live orphan watchdog present" || bad "no per-rank orphan watchdog"
grep -q 'setsid' "$HERE/l0_launch.sh" && \
  { grep -q 'nohup "\$R"' "$HERE/l0_launch.sh" \
      && ok "local ranks are started WITHOUT setsid (remote keeps it, by necessity)" \
      || bad "local ranks still use setsid"; }

rm -rf "$TMP"
echo
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
