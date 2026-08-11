#!/usr/bin/env bash
# #656 rung 2: relaunch the live instance with --phase-flip-spill-depth draft.
#
# WHY A REPLAY AND NOT THE BOOT SCRIPT. route_a_631_prod_boot.sh carries its
# own defaults (a different model checkpoint, a different max-total-tokens, a
# different per-rank MiB vector) than the instance actually running. Booting
# through it would change several things at once and the corridor/throughput
# numbers afterwards could not be attributed to the spill. This replays the
# LIVE process's argv and environment verbatim and substitutes exactly one
# token, so the depth is the only variable.
#
# STOPPING IS EXPLICIT AND BY PID. `pkill -f` has self-killed an agent shell
# in this chain eleven times, most recently one screen after the author quoted
# the warning against it; it is also forbidden by the brief. And
# seam_scaling_reboot.py --from-capture silently SKIPS its stop step, so
# "reboot" scripts here do not imply anything was stopped.
set -euo pipefail

WT="${WT:-/spinning/wt-631-routea}"
PY="${PY:-/spinning/htsglang-gpu/.venv/bin/python}"
PORT="${PORT:-30030}"
DEPTH="${DEPTH:-draft}"
# EXTRA_ENV: "K=V" pairs, newline separated, appended to the replayed env.
EXTRA_ENV="${EXTRA_ENV:-}"
LOG="${LOG:-/spinning/evidence-631/s30/serving-guard.log}"
OLD_PID="${OLD_PID:-}"

mkdir -p "$(dirname "$LOG")"

# A live peer may be mid-boot on these cards. <120 s of heartbeat means wait.
# SELF names THIS session's heartbeat so the check does not refuse on the
# caller's own liveness proof. It was hardcoded to one session, which meant
# the next shift's first reboot refused against its own heartbeat and looked
# like a peer collision.
SELF="${SELF:-successor30}"
now=$(date -u +%s)
for hb in /spinning/gpu-arb/heartbeat.*; do
  [ -e "$hb" ] || continue
  case "$hb" in *"$SELF") continue ;; esac
  age=$(( now - $(stat -c %Y "$hb") ))
  if [ "$age" -lt 120 ]; then
    echo "REFUSE: $hb is ${age}s old -- a live peer may hold these cards." >&2
    exit 3
  fi
done

if [ -z "$OLD_PID" ]; then
  # MATCH ON ARGV STRUCTURE, NEVER `pgrep -f <pattern>`.
  #
  # `pgrep -f "sglang.launch_server.*--port 30030"` matches ANY process whose
  # command line CONTAINS that text -- including the shell that is running the
  # pgrep, and including any monitoring loop that mentions it. On 2026-08-10
  # this cost two incidents in one shift: the reboot captured a bash wrapper's
  # 5-entry argv as if it were the server's 58-entry argv and relaunched a
  # stub, and separately a health-wait loop could never exit because its own
  # `! pgrep` clause matched itself. It is the same family as the `pkill -f`
  # self-kill the brief forbids -- eleven occurrences in this chain.
  #
  # A real server is: argv[0] is a python interpreter, argv[2] is exactly
  # "sglang.launch_server", and "--port <PORT>" appears as adjacent argv
  # entries. A shell that merely quotes those strings satisfies none of it.
  OLD_PID=$(
    for d in /proc/[0-9]*; do
      pid=${d#/proc/}
      [ -r "$d/cmdline" ] || continue
      mapfile -d '' -t a < "$d/cmdline" 2>/dev/null || continue
      [ "${#a[@]}" -ge 4 ] || continue
      case "${a[0]}" in *python*) ;; *) continue ;; esac
      [ "${a[2]}" = "sglang.launch_server" ] || continue
      for i in "${!a[@]}"; do
        if [ "${a[$i]}" = "--port" ] && [ "${a[$((i+1))]}" = "$PORT" ]; then
          echo "$pid"; break
        fi
      done
    done | head -1
  )
fi

if [ -n "$OLD_PID" ]; then
  echo "[reboot] capturing argv/env from live pid $OLD_PID"
  tr '\0' '\n' < "/proc/$OLD_PID/cmdline" > /tmp/s30_argv.txt
  tr '\0' '\n' < "/proc/$OLD_PID/environ" > /tmp/s30_env.txt
else
  echo "REFUSE: no live instance on port $PORT and no OLD_PID given; there is" >&2
  echo "        nothing to replay from. Boot via route_a_631_prod_boot.sh." >&2
  exit 4
fi

# Substitute the depth token: the value on the line AFTER the flag.
$PY - "$DEPTH" <<'PYEOF'
import sys
depth = sys.argv[1]
argv = open("/tmp/s30_argv.txt").read().split("\n")
if argv and argv[-1] == "":
    argv.pop()
try:
    i = argv.index("--phase-flip-spill-depth")
    argv[i + 1] = depth
except ValueError:
    argv += ["--phase-flip-spill-depth", depth]
open("/tmp/s30_argv_new.txt", "w").write("\n".join(argv))
print(f"[reboot] argv rewritten, {len(argv)} entries, depth={depth}")
PYEOF

echo "[reboot] stopping pid $OLD_PID (SIGTERM, by PID -- never pkill -f)"
kill -TERM "$OLD_PID" 2>/dev/null || true
for _ in $(seq 1 60); do
  kill -0 "$OLD_PID" 2>/dev/null || break
  sleep 1
done
if kill -0 "$OLD_PID" 2>/dev/null; then
  echo "[reboot] still alive after 60 s; py-spy dump before escalating"
  "$PY" -m py_spy dump --pid "$OLD_PID" 2>&1 | head -40 || true
  kill -KILL "$OLD_PID" 2>/dev/null || true
  sleep 5
fi
echo "[reboot] old instance gone; waiting for its workers to release VRAM"
# WAIT ON THE CHILDREN, NOT THE PARENT. The VRAM is held by the
# `sglang::scheduler_PP*` worker processes, which do NOT match the
# launch_server pattern -- the first version of this loop polled only the
# parent, saw it gone immediately, and printed free-memory figures
# (1625/4154/1895) that were the OLD instance's, while the relaunch was
# already starting into a card that had not been given back yet. Poll the
# DRIVER instead: free memory is the thing the next boot actually needs.
for _ in $(seq 1 90); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
         | awk '$1 > 2000 {n++} END {print n+0}')
  [ "$busy" -eq 0 ] && break
  sleep 2
done
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader

echo "[reboot] relaunching with depth=$DEPTH, log -> $LOG"
# execve from Python rather than splicing the environment through the shell.
# A captured environment contains values with spaces and braces (this boot's
# --chat-template-default-kwargs is literally '{"preserve_thinking": true}'),
# and word-splitting them through `env -i $(...)` corrupts the argv silently
# -- which would produce a "measurement" of a differently-configured server.
REPLAY_EXTRA_ENV="$EXTRA_ENV" setsid "$PY" - "$LOG" <<'PYEOF' &
import os, sys

log = sys.argv[1]
argv = [a for a in open("/tmp/s30_argv_new.txt").read().split("\n") if a != ""]
env = {}
for line in open("/tmp/s30_env.txt").read().split("\n"):
    if not line or "=" not in line:
        continue
    k, v = line.split("=", 1)
    env[k] = v
for line in os.environ.get("REPLAY_EXTRA_ENV", "").split("\n"):
    if line and "=" in line:
        k, v = line.split("=", 1)
        env[k] = v
        print(f"[reboot] extra env {k}={v}", flush=True)
fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(fd, 1)
os.dup2(fd, 2)
os.setsid() if os.getpgrp() != os.getpid() else None
os.execve(argv[0], argv, env)
PYEOF
echo "[reboot] launched; tail $LOG for progress"
