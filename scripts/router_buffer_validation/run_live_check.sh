#!/usr/bin/env bash
# Live test-port validation for the router's local-backend buffer (task #675).
#
# Runs a SECOND, fully isolated router instance on a free high port against a
# real (separate-process) fake local backend -- never touches the live
# claude-local-router systemd unit on 30099, nor 30030.
#
# Proves, against real processes and real sockets (not the hermetic test
# harness):
#   A) a request issued while the local backend is down completes
#      successfully once the backend comes up (held, not failed)
#   B) max-wait expiry returns an honest 502 naming the waited seconds
#   C) the healthy pass-through path is fast and unchanged
#
# Usage: bash run_live_check.sh
set -uo pipefail

REPO=/spinning/wt-anthropic-front
VENV_PY=/spinning/htsglang-gpu/.venv/bin/python
ROUTER_PORT=47101
LOCAL_PORT=47100
WAIT_S=4
POLL_S=0.3

FAKE_PID=""
ROUTER_PID=""

cleanup() {
    if [ -n "$ROUTER_PID" ] && kill -0 "$ROUTER_PID" 2>/dev/null; then
        kill "$ROUTER_PID" 2>/dev/null
        wait "$ROUTER_PID" 2>/dev/null
    fi
    if [ -n "$FAKE_PID" ] && kill -0 "$FAKE_PID" 2>/dev/null; then
        kill "$FAKE_PID" 2>/dev/null
        wait "$FAKE_PID" 2>/dev/null
    fi
}
trap cleanup EXIT

fail() { echo "LIVE-CHECK FAIL: $*"; exit 1; }

echo "=== router buffer live validation (task #675) ==="
echo "router test port: $ROUTER_PORT   fake-backend port: $LOCAL_PORT"
echo "(live 30099/30030 are never touched by this script)"
echo

for p in $ROUTER_PORT $LOCAL_PORT; do
    if (echo >"/dev/tcp/127.0.0.1/$p") 2>/dev/null; then
        fail "port $p already in use, pick a different test port"
    fi
done

echo "--- starting router test instance on :$ROUTER_PORT (local-backend down for now) ---"
PYTHONPATH="$REPO/python" "$VENV_PY" -m sglang.srt.entrypoints.anthropic.router \
    --host 127.0.0.1 --port "$ROUTER_PORT" \
    --local-base "http://127.0.0.1:$LOCAL_PORT" \
    --local-model TestModel --no-thinking-shim \
    --local-wait-s "$WAIT_S" --local-poll-interval-s "$POLL_S" \
    > /tmp/router_live_check_router.log 2>&1 &
ROUTER_PID=$!

for i in $(seq 1 50); do
    curl -s -m 1 "http://127.0.0.1:$ROUTER_PORT/__router/stats" >/dev/null 2>&1 && break
    sleep 0.1
done
curl -s -m 1 "http://127.0.0.1:$ROUTER_PORT/__router/stats" >/dev/null 2>&1 \
    || fail "router test instance never came up (see /tmp/router_live_check_router.log)"
echo "router test instance is up (pid $ROUTER_PID)"
echo

BODY='{"model":"TestModel","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}'

# --- Test A: request issued while backend down, completes once backend comes up ---
echo "--- Test A: held while down, succeeds once backend starts ---"
OUT_A=/tmp/router_live_check_a.out
t0=$(date +%s.%N)
curl -s -m 20 -o "$OUT_A" -w "%{http_code}" -X POST \
    "http://127.0.0.1:$ROUTER_PORT/v1/messages" \
    -H 'content-type: application/json' -d "$BODY" > /tmp/router_live_check_a.code &
CURL_A_PID=$!

sleep 1.0
QUEUED=$(curl -s -m 1 "http://127.0.0.1:$ROUTER_PORT/__router/stats" | grep -o '"buffer_queued": *[0-9]*' | grep -o '[0-9]*')
echo "  after 1.0s (backend still down): buffer_queued=$QUEUED (expect 1)"
[ "$QUEUED" = "1" ] || fail "request A was not held (buffer_queued=$QUEUED, expected 1)"

echo "  starting fake local backend on :$LOCAL_PORT"
"$VENV_PY" "$(dirname "$0")/fake_local_backend.py" "$LOCAL_PORT" \
    > /tmp/router_live_check_fake.log 2>&1 &
FAKE_PID=$!

wait "$CURL_A_PID"
t1=$(date +%s.%N)
CODE_A=$(cat /tmp/router_live_check_a.code)
ELAPSED_A=$(awk "BEGIN{printf \"%.2f\", $t1 - $t0}")
echo "  request A finished: http=$CODE_A elapsed=${ELAPSED_A}s"
[ "$CODE_A" = "200" ] || fail "request A did not succeed once backend came up (got $CODE_A, body: $(cat "$OUT_A"))"
echo "  PASS: held request completed successfully once local backend came up"
echo

# --- stop fake backend again for Test B ---
kill "$FAKE_PID" 2>/dev/null; wait "$FAKE_PID" 2>/dev/null; FAKE_PID=""
sleep 0.3

# --- Test B: backend stays down for the whole wait, honest timeout 502 ---
echo "--- Test B: backend stays down -> honest 502 with waited seconds ---"
OUT_B=/tmp/router_live_check_b.out
t0=$(date +%s.%N)
CODE_B=$(curl -s -m 20 -o "$OUT_B" -w "%{http_code}" -X POST \
    "http://127.0.0.1:$ROUTER_PORT/v1/messages" \
    -H 'content-type: application/json' -d "$BODY")
t1=$(date +%s.%N)
ELAPSED_B=$(awk "BEGIN{printf \"%.2f\", $t1 - $t0}")
echo "  request B finished: http=$CODE_B elapsed=${ELAPSED_B}s (limit was ${WAIT_S}s)"
[ "$CODE_B" = "502" ] || fail "request B did not 502 on timeout (got $CODE_B, body: $(cat "$OUT_B"))"
grep -Eq "held the request for [0-9]+s" "$OUT_B" || fail "502 body does not name the waited/held seconds: $(cat "$OUT_B")"
grep -qi "gave up" "$OUT_B" || fail "502 body does not read as an honest timeout message: $(cat "$OUT_B")"
NEAR_LIMIT=$(awk "BEGIN{print ($ELAPSED_B >= ${WAIT_S} - 1.0) ? 1 : 0}")
[ "$NEAR_LIMIT" = "1" ] || fail "request B returned after only ${ELAPSED_B}s, well short of the configured ${WAIT_S}s wait"
echo "  body: $(cat "$OUT_B")"
echo "  PASS: honest 502 naming the waited duration, took roughly the configured wait (~${WAIT_S}s)"
echo

# --- Test C: healthy path is fast and unchanged ---
echo "--- Test C: healthy pass-through has no added latency ---"
"$VENV_PY" "$(dirname "$0")/fake_local_backend.py" "$LOCAL_PORT" \
    > /tmp/router_live_check_fake2.log 2>&1 &
FAKE_PID=$!
for i in $(seq 1 30); do
    curl -s -m 1 "http://127.0.0.1:$LOCAL_PORT/health" >/dev/null 2>&1 && break
    sleep 0.1
done

OUT_C=/tmp/router_live_check_c.out
t0=$(date +%s.%N)
CODE_C=$(curl -s -m 5 -o "$OUT_C" -w "%{http_code}" -X POST \
    "http://127.0.0.1:$ROUTER_PORT/v1/messages" \
    -H 'content-type: application/json' -d "$BODY")
t1=$(date +%s.%N)
ELAPSED_C=$(awk "BEGIN{printf \"%.2f\", $t1 - $t0}")
echo "  request C finished: http=$CODE_C elapsed=${ELAPSED_C}s"
[ "$CODE_C" = "200" ] || fail "healthy-path request did not succeed (got $CODE_C, body: $(cat "$OUT_C"))"
FAST=$(awk "BEGIN{print ($ELAPSED_C < 1.0) ? 1 : 0}")
[ "$FAST" = "1" ] || fail "healthy-path request took ${ELAPSED_C}s, expected well under 1s (no buffering overhead)"
echo "  PASS: healthy request succeeded fast (${ELAPSED_C}s), no buffering overhead"
echo

echo "--- final /__router/stats snapshot ---"
curl -s -m 1 "http://127.0.0.1:$ROUTER_PORT/__router/stats"
echo
echo

echo "=== ALL LIVE CHECKS PASSED (A: held-then-succeeds, B: honest timeout 502, C: healthy unchanged) ==="
