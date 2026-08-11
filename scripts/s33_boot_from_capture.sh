#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# #656 successor 33: boot from a PREVIOUSLY CAPTURED argv/env, with edits.
#
# s30_reboot_corridor_guard.sh replays the LIVE process and refuses when
# nothing is running -- correctly, because it cannot invent a configuration.
# But an instance that CRASHED leaves the same need: relaunch exactly what it
# was, minus the thing that killed it. The capture from the last reboot is
# still on disk (/tmp/s30_argv.txt, /tmp/s30_env.txt), so this boots from that
# instead of from route_a_631_prod_boot.sh, whose own defaults are a different
# checkpoint, a different per-rank MiB vector and a different pool.
#
# Same discipline as s30: peer-heartbeat check, ARGV_SET for the edits, execve
# rather than shell word-splitting (the captured argv contains
# --chat-template-default-kwargs '{"preserve_thinking": true}', which any
# `env -i $(...)` splices into garbage).
set -euo pipefail

PY="${PY:-/spinning/htsglang-gpu/.venv/bin/python}"
LOG="${LOG:?LOG required}"
SELF="${SELF:-successor33}"
ARGV_SRC="${ARGV_SRC:-/tmp/s30_argv.txt}"
ENV_SRC="${ENV_SRC:-/tmp/s30_env.txt}"
EXTRA_ENV="${EXTRA_ENV:-}"

mkdir -p "$(dirname "$LOG")"

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

# Refuse to boot on top of a live server: two instances on one set of cards is
# an OOM, and the capture cannot tell whether the running one is wanted.
for d in /proc/[0-9]*; do
  [ -r "$d/cmdline" ] || continue
  mapfile -d '' -t a < "$d/cmdline" 2>/dev/null || continue
  [ "${#a[@]}" -ge 4 ] || continue
  case "${a[0]}" in *python*) ;; *) continue ;; esac
  [ "${a[2]}" = "sglang.launch_server" ] || continue
  echo "REFUSE: pid ${d#/proc/} is already a live launch_server." >&2
  exit 4
done

busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
       | awk '$1 > 2000 {n++} END {print n+0}')
if [ "$busy" -ne 0 ]; then
  echo "[boot] $busy card(s) still hold >2 GiB; waiting for release"
  for _ in $(seq 1 60); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
           | awk '$1 > 2000 {n++} END {print n+0}')
    [ "$busy" -eq 0 ] && break
    sleep 2
  done
fi
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader

# THE SOURCE MAY NOT BE THE SINK. This script WRITES the prepared argv to
# /tmp/s33_argv.txt, so pointing ARGV_SRC at that same file makes every run
# inherit the previous run's edits. #658 lost a boot to exactly that: a failed
# attempt left --enable-vram-dial in the file, and the next boot -- which had
# deliberately dropped the flag -- picked it up again and died with the same
# error, which reads as "my edit did nothing" rather than "my source is
# poisoned". Refuse instead of accumulating.
if [ "$(readlink -f "$ARGV_SRC")" = "$(readlink -f /tmp/s33_argv.txt)" ]; then
  echo "REFUSE: ARGV_SRC is /tmp/s33_argv.txt, which is also this script's" >&2
  echo "        OUTPUT. Copy it to a pristine path and point ARGV_SRC there," >&2
  echo "        or each boot will inherit the last one's ARGV_SET edits." >&2
  exit 5
fi

ARGV_SET="${ARGV_SET:-}" ARGV_SRC="$ARGV_SRC" $PY - <<'PYEOF'
import os

argv = [a for a in open(os.environ["ARGV_SRC"]).read().split("\n") if a != ""]


def put(flag, value):
    # A BARE FLAG IS A FLAG, NOT A FLAG WITH AN EMPTY ARGUMENT. store_true
    # options (--enable-vram-dial, --enable-dynamic-chunking) carry no value,
    # and appending ["--flag", ""] hands argparse an empty POSITIONAL, which
    # fails the boot in a way that reads like a config error rather than a
    # quoting one. #658 hit exactly that.
    if value == "":
        if flag not in argv:
            argv.append(flag)
        return
    try:
        i = argv.index(flag)
        argv[i + 1] = value
    except ValueError:
        argv.extend([flag, value])


for line in os.environ.get("ARGV_SET", "").split("\n"):
    line = line.strip()
    if not line:
        continue
    flag, _, value = line.partition(" ")
    put(flag, value.strip())
    print(f"[boot] argv set {flag}={value.strip()}", flush=True)
open("/tmp/s33_argv.txt", "w").write("\n".join(argv))
print(f"[boot] argv prepared, {len(argv)} entries")
PYEOF

REPLAY_EXTRA_ENV="$EXTRA_ENV" ENV_SRC="$ENV_SRC" setsid "$PY" - "$LOG" <<'PYEOF' &
import os, sys

log = sys.argv[1]
argv = [a for a in open("/tmp/s33_argv.txt").read().split("\n") if a != ""]
env = {}
for line in open(os.environ["ENV_SRC"]).read().split("\n"):
    if not line or "=" not in line:
        continue
    k, v = line.split("=", 1)
    env[k] = v
for line in os.environ.get("REPLAY_EXTRA_ENV", "").split("\n"):
    if line and "=" in line:
        k, v = line.split("=", 1)
        env[k] = v
        print(f"[boot] extra env {k}={v}", flush=True)
fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(fd, 1)
os.dup2(fd, 2)
os.setsid() if os.getpgrp() != os.getpid() else None
os.execve(argv[0], argv, env)
PYEOF
echo "[boot] launched; log -> $LOG"
