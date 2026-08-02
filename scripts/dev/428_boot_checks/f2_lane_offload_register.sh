#!/usr/bin/env bash
# Task #428 / #421 F2 boot check: --lane-offload-profile / -class-policy /
# -park-targets must reach the process-global offload register.
#
# Desk state: configure_global_register_from_server_args is called at the top
# of ModelRunner.__init__ and hermetically tested, including a call-site pin.
# What only a boot can show:
#   1. the configure step really runs BEFORE the first adapter registration
#      (the pools, input buffers and lane workspaces are built in the same
#      __init__, and the fallback register would look identical from outside);
#   2. it runs once per process even with a draft runner in the picture
#      (NEXTN adds a second ModelRunner -- a rebuild would drop the target
#      runner's already-booked items);
#   3. a planted typo refuses the boot rather than degrading to the default.
#
# Exits non-zero on the first failed check.

set -uo pipefail

source /root/rig-env.sh 2>/dev/null || true
REPO_ROOT="${REPO_ROOT:?set REPO_ROOT or provide /root/rig-env.sh}"
VENV="${VENV:?set VENV}"
MODEL_ROOT="${MODEL_ROOT:?set MODEL_ROOT}"
WT="${WT:-$(dirname "$REPO_ROOT")/wt-428-wiring}"
PORT="${PORT:-30429}"
MODEL="${MODEL:-$MODEL_ROOT/Qwen3.6-27B-FP8}"

fail=0
say() { printf '%-6s %s\n' "$1" "$2"; [ "$1" = FAIL ] && fail=1; return 0; }

export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WT/python"
export SGLANG_UNEVEN_DCP=1
export SGLANG_UNEVEN_DCP_WEIGHTED=1
export SGLANG_MAMBA_SSM_DTYPE=bfloat16
# The whole feature is dark-launched behind this. Without it the register is
# None and the flags are legitimately inert -- which is what let F2 hide.
export SGLANG_OFFLOAD_REGISTER=1

boot() {
  # $1 = log path, $2.. = extra flags
  local log="$1"; shift
  : > "$log"
  cd "$WT" || return 1
  setsid "$VENV/bin/python" -u -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp-size 3 --rank-gpu-id 0,1,2 --rank-tp-ratio auto-performance \
    --rank-auto-reserve-mib 5500,3800,3800 \
    --kv-cache-dtype fp8_e4m3 --context-length 32768 --trust-remote-code \
    --max-running-requests 8 \
    --speculative-algorithm NEXTN --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --enable-metrics \
    --host 127.0.0.1 --port "$PORT" \
    "$@" \
    > "$log" 2>&1 &
  echo $! > /tmp/428_f2.pid
}

stop() {
  if [ -f /tmp/428_f2.pid ]; then
    kill -TERM "-$(cat /tmp/428_f2.pid)" 2>/dev/null
    sleep 5
    kill -KILL "-$(cat /tmp/428_f2.pid)" 2>/dev/null
    rm -f /tmp/428_f2.pid
  fi
}
trap stop EXIT

wait_up() {
  for _ in $(seq 1 180); do
    curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    kill -0 "$(cat /tmp/428_f2.pid)" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}

# --- run A: a non-default configuration must be observable -----------------
LOG_A=/tmp/428_f2_capacity.log
boot "$LOG_A" \
  --lane-offload-profile capacity \
  --lane-offload-class-policy graph_rungs=resident \
  --lane-offload-park-targets peer_vram\>host_ram

if wait_up; then
  say PASS "server up with the three --lane-offload-* flags"
else
  say FAIL "server did not come up"
  grep -m3 -iE "lane.offload|Traceback" "$LOG_A" | head -5
  exit 1
fi

if grep -q "offload-register.*configured from server args" "$LOG_A"; then
  say PASS "the register was configured from the server args"
  grep -m1 "configured from server args" "$LOG_A" | tail -c 240
  echo
else
  say FAIL "no 'configured from server args' line -- #421 F2 has regressed"
fi

if grep -q "configured from server args: profile=capacity" "$LOG_A"; then
  say PASS "profile=capacity reached the register (the F2 defect verbatim)"
else
  say FAIL "the operator's profile did not reach the register"
fi

if grep -q "park-targets=peer_vram>host_ram" "$LOG_A"; then
  say PASS "the park-target chain reached the register"
else
  say FAIL "the park-target chain did not reach the register"
fi

# Once per process: NEXTN builds a draft ModelRunner in the same process, so
# a second configure would show up as a second line.
n=$(grep -c "configured from server args" "$LOG_A")
if [ "$n" -eq 1 ]; then
  say PASS "configured exactly once per process (draft runner did not rebuild)"
else
  say FAIL "configured $n times -- a rebuild drops items the first runner booked"
fi

if grep -qiE "Traceback|CUDA error" "$LOG_A"; then
  say FAIL "run A log contains a traceback / CUDA error"
fi
stop

# --- run B: a planted typo must refuse the boot ----------------------------
LOG_B=/tmp/428_f2_typo.log
boot "$LOG_B" --lane-offload-profile capacty
sleep 45
if wait_up; then
  say FAIL "a typo'd --lane-offload-profile started the server (silent default)"
  stop
else
  if grep -q "capacty" "$LOG_B"; then
    say PASS "the planted typo refused the boot and named the bad value"
  else
    say FAIL "the boot failed but not for the typo -- check $LOG_B"
  fi
fi

exit "$fail"
