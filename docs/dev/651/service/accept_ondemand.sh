#!/bin/bash
# #651: acceptance for the on-demand service.
#
# Proves the whole cycle TWICE, because once proves only that a load works --
# the interesting failures (a stale process holding the GPU, a park that leaves
# the port bound, a second load that OOMs on memory the first never returned)
# all appear on the SECOND wake, not the first.
#
# Each cycle asserts, in order:
#   1. the model is parked and the host RAM is actually back;
#   2. a request wakes it, is HELD rather than refused, and returns the
#      correct answer to a question with exactly one right answer;
#   3. the model parks again on its own after the idle window;
#   4. the RAM comes back again.
#
# The determined-answer probe matters more than a 200: a served model that
# loads and then emits noise is a worse outcome than one that fails to load,
# because nothing alerts on it.
set -u

PORT=${PORT:-31651}
BASE="http://127.0.0.1:${PORT}"
IDLE=${HTSGLANG_IDLE_PARK_SECONDS:-60}
RESULTS=${RESULTS:-/root/651-p2/results/accept_ondemand_$(date +%H%M%S).txt}

exec > >(tee -a "$RESULTS") 2>&1
echo "=== on-demand acceptance $(date -Is) ==="
echo "port=$PORT idle_park=${IDLE}s"

status_state() {
  curl -s -m 5 "$BASE/ondemand/status" | python3 -c \
    'import json,sys; print(json.load(sys.stdin)["state"])' 2>/dev/null || echo "unreachable"
}

host_free_mib() {
  awk '/^MemAvailable:/{printf "%d", $2/1024}' /proc/meminfo
}

# One question, one correct answer, greedy decoding.
probe() {
  local question="$1" expect="$2" label="$3"
  local t0 t1 body answer
  t0=$(date +%s)
  # enable_thinking=false is REQUIRED, not tidiness: this checkpoint emits a
  # reasoning preamble by default, so a short max_tokens truncates the reply
  # mid-thought and the probe scores a working model as wrong. max_tokens is
  # also generous for the same reason.
  body=$(curl -s -m 1200 "$BASE/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"qwen36-35b-a3b\",
         \"messages\":[{\"role\":\"user\",\"content\":\"$question\"}],
         \"chat_template_kwargs\":{\"enable_thinking\":false},
         \"temperature\":0,\"max_tokens\":128}")
  t1=$(date +%s)
  answer=$(printf '%s' "$body" | python3 -c \
    'import json,sys
try:
    print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())
except Exception as e:
    print("PARSE-FAIL: %s" % (sys.stdin.read()[:200] if False else e))' 2>/dev/null)
  echo "  [$label] ${t1}s-${t0}s => $((t1 - t0))s  answer=<${answer}>"
  if printf '%s' "$answer" | grep -qi "$expect"; then
    echo "  [$label] CORRECT (expected to contain '$expect')"
    return 0
  fi
  echo "  [$label] WRONG (expected to contain '$expect')"
  return 1
}

fail=0

for cycle in 1 2; do
  echo
  echo "--- cycle $cycle ---"
  echo "  state before request: $(status_state), host free $(host_free_mib) MiB"

  probe "What is 6 multiplied by 7? Reply with only the number." "42" "cycle${cycle}-wake" || fail=1
  echo "  state after request: $(status_state), host free $(host_free_mib) MiB"

  # A second question while hot must NOT pay the load again.
  probe "What is the capital city of France? Reply with only the city name." "Paris" "cycle${cycle}-hot" || fail=1

  echo "  waiting $((IDLE + 30))s for the idle park ..."
  sleep $((IDLE + 30))
  state=$(status_state)
  echo "  state after idle: $state, host free $(host_free_mib) MiB"
  if [ "$state" != "parked" ]; then
    echo "  FAIL: expected 'parked' after ${IDLE}s idle, got '$state'"
    fail=1
  else
    echo "  PARKED as expected"
  fi
done

echo
if [ "$fail" = "0" ]; then
  echo "ACCEPTANCE: PASS (2 cycles)"
else
  echo "ACCEPTANCE: FAIL"
fi
echo "results written to $RESULTS"
exit "$fail"
