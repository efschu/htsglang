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
LOG="${LOG:-/spinning/evidence-631/s29/serving-depth-${DEPTH}.log}"
OLD_PID="${OLD_PID:-}"

mkdir -p "$(dirname "$LOG")"

# A live peer may be mid-boot on these cards. <120 s of heartbeat means wait.
now=$(date -u +%s)
for hb in /spinning/gpu-arb/heartbeat.*; do
  [ -e "$hb" ] || continue
  case "$hb" in *successor29) continue ;; esac
  age=$(( now - $(stat -c %Y "$hb") ))
  if [ "$age" -lt 120 ]; then
    echo "REFUSE: $hb is ${age}s old -- a live peer may hold these cards." >&2
    exit 3
  fi
done

if [ -z "$OLD_PID" ]; then
  OLD_PID=$(pgrep -f "sglang.launch_server.*--port $PORT" | head -1 || true)
fi

if [ -n "$OLD_PID" ]; then
  echo "[reboot] capturing argv/env from live pid $OLD_PID"
  tr '\0' '\n' < "/proc/$OLD_PID/cmdline" > /tmp/s29_argv.txt
  tr '\0' '\n' < "/proc/$OLD_PID/environ" > /tmp/s29_env.txt
else
  echo "REFUSE: no live instance on port $PORT and no OLD_PID given; there is" >&2
  echo "        nothing to replay from. Boot via route_a_631_prod_boot.sh." >&2
  exit 4
fi

# Substitute the depth token: the value on the line AFTER the flag.
$PY - "$DEPTH" <<'PYEOF'
import sys
depth = sys.argv[1]
argv = open("/tmp/s29_argv.txt").read().split("\n")
if argv and argv[-1] == "":
    argv.pop()
try:
    i = argv.index("--phase-flip-spill-depth")
    argv[i + 1] = depth
except ValueError:
    argv += ["--phase-flip-spill-depth", depth]
open("/tmp/s29_argv_new.txt", "w").write("\n".join(argv))
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
for _ in $(seq 1 60); do
  remaining=$(pgrep -f "sglang.launch_server.*--port $PORT" | head -1 || true)
  [ -z "$remaining" ] || { sleep 1; continue; }
  break
done
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader

echo "[reboot] relaunching with depth=$DEPTH, log -> $LOG"
# execve from Python rather than splicing the environment through the shell.
# A captured environment contains values with spaces and braces (this boot's
# --chat-template-default-kwargs is literally '{"preserve_thinking": true}'),
# and word-splitting them through `env -i $(...)` corrupts the argv silently
# -- which would produce a "measurement" of a differently-configured server.
setsid "$PY" - "$LOG" <<'PYEOF' &
import os, sys

log = sys.argv[1]
argv = [a for a in open("/tmp/s29_argv_new.txt").read().split("\n") if a != ""]
env = {}
for line in open("/tmp/s29_env.txt").read().split("\n"):
    if not line or "=" not in line:
        continue
    k, v = line.split("=", 1)
    env[k] = v
fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(fd, 1)
os.dup2(fd, 2)
os.setsid() if os.getpgrp() != os.getpid() else None
os.execve(argv[0], argv, env)
PYEOF
echo "[reboot] launched; tail $LOG for progress"
