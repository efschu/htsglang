#!/bin/bash
# Canonical host-RAM watchdog for the DeepSeek V4 Flash GGUF boots (#391).
#
# Samples the cgroup every INTERVAL seconds and hard-stops the launched
# process group before the kernel OOM killer gets the chance. Replaces the
# per-boot rammonN.sh copies that used to live only in the battery result
# directories; the runbook (section 4.5.4) points here.
#
# WHY THE GUARD IS ON memory.current AND NOT ON anon
# ---------------------------------------------------------------------------
# Boot attempt 8 (2026-08-01, /spinning/gpu-battery-results/2026-08-01_391_
# dsv4flash8/ram8.log) guarded on `anon > 93 GiB` and never fired. It could not
# have fired: the process died with anon at 32.9 GiB. What actually filled the
# 98.5 GiB swapless box was the GGUF page cache -- `file` was 81.6 GiB while
# anon was 15.1 GiB, and for the next seven minutes the kernel traded one for
# the other almost byte for byte (file 81.6 -> 63.8, anon 15.1 -> 32.9) with
# memory.current pinned at 98.3-98.4 GiB the whole time. Then a 2.4 GiB anon
# burst inside one 15 s window outran reclaim and oom_kill fired at
# memory.peak 98.55 GiB.
#
# An anon-only guard is therefore not "set too high", it is structurally blind:
# it cannot see the half of the budget that was full. The same argument rules
# out the obvious repairs:
#
#   * `anon + file`               -- double counts nothing but still needs a
#                                    threshold per term;
#   * `file - inactive_file`      -- "reclaimable-honest available" heuristics
#                                    are exactly the trap. Reclaimable is not
#                                    reclaimable IN TIME; boot 8 proves the
#                                    kernel can be reclaiming at full tilt and
#                                    still lose to an allocation burst;
#   * free / MemAvailable         -- lxcfs-mangled in this container.
#
# memory.current is the number the OOM killer itself compares against the
# limit. Guard on that, with a margin big enough to survive one sampling
# interval's worth of allocation (default 6 GiB against the observed ~2.4 GiB
# per 15 s burst).
set -u

RUN_DIR=""
PIDFILE=""
OUT=""
MARGIN_GIB=6
INTERVAL=15

usage() {
  cat >&2 <<'EOF'
usage: rammon.sh --pidfile FILE [--run-dir DIR] [--out FILE]
                 [--margin-gib N] [--interval SECONDS]

  --pidfile     file holding the PID of the launched server; its process GROUP
                is what gets stopped, and its disappearance ends the watch
  --run-dir     directory for the log (default: the pidfile's directory)
  --out         log path (default: <run-dir>/ram.log)
  --margin-gib  how far below the cgroup limit to stop (default 6)
  --interval    seconds between samples (default 15)
EOF
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --pidfile) PIDFILE="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --margin-gib) MARGIN_GIB="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "rammon.sh: unknown argument '$1'" >&2; usage ;;
  esac
done

[ -n "$PIDFILE" ] || usage
[ -n "$RUN_DIR" ] || RUN_DIR=$(dirname "$PIDFILE")
[ -n "$OUT" ] || OUT="$RUN_DIR/ram.log"

CG=/sys/fs/cgroup

# The memory limit that actually binds this process. memory.max is "max" on the
# leaf cgroup of an LXC container, so walk up for the nearest numeric one and
# fall back to MemTotal -- which lxcfs reports as the container's limit, i.e.
# the honest ceiling here even though it is the wrong number on a bare host.
resolve_limit_bytes() {
  local dir="$CG" value
  while [ -n "$dir" ] && [ "$dir" != "/" ]; do
    if [ -r "$dir/memory.max" ]; then
      value=$(cat "$dir/memory.max")
      if [ "$value" != "max" ]; then
        echo "$value"
        return
      fi
    fi
    dir=$(dirname "$dir")
    [ "$dir" = "/sys/fs" ] && break
  done
  awk '/^MemTotal:/{print $2 * 1024}' /proc/meminfo
}

LIMIT=$(resolve_limit_bytes)
if [ -z "$LIMIT" ] || [ "$LIMIT" -le 0 ] 2>/dev/null; then
  echo "rammon.sh: cannot resolve a memory limit; refusing to run blind" >&2
  exit 3
fi
GUARD=$(awk -v l="$LIMIT" -v m="$MARGIN_GIB" 'BEGIN{printf "%d", l - m * 1073741824}')
gib() { awk -v a="$1" 'BEGIN{printf "%.1f", a/1073741824}'; }

: > "$OUT"
printf 'rammon: limit=%s GiB margin=%s GiB guard=%s GiB (on memory.current) interval=%ss\n' \
    "$(gib "$LIMIT")" "$MARGIN_GIB" "$(gib "$GUARD")" "$INTERVAL" >> "$OUT"

while :; do
  cur=$(cat "$CG/memory.current")
  anon=$(awk '/^anon /{print $2}' "$CG/memory.stat")
  file=$(awk '/^file /{print $2}' "$CG/memory.stat")
  peak=$(cat "$CG/memory.peak" 2>/dev/null || echo 0)
  printf '%s current=%sGiB anon=%sGiB file=%sGiB peak=%sGiB\n' \
      "$(date -u +%H:%M:%S)" "$(gib "$cur")" "$(gib "$anon")" "$(gib "$file")" \
      "$(gib "$peak")" >> "$OUT"

  if [ "$cur" -gt "$GUARD" ]; then
    printf 'GUARD memory.current %s GiB > %s GiB -- stopping own pgid\n' \
        "$(gib "$cur")" "$(gib "$GUARD")" >> "$OUT"
    if [ -f "$PIDFILE" ]; then
      p=$(cat "$PIDFILE")
      g=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')
      # Only ever our own launched group -- never a broad pkill, which on this
      # shared box would take other agents' servers with it.
      [ -n "$g" ] && kill -TERM -"$g" 2>/dev/null
    fi
    exit 1
  fi

  if [ -f "$PIDFILE" ]; then
    kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { echo "server gone" >> "$OUT"; exit 0; }
  fi
  sleep "$INTERVAL"
done
