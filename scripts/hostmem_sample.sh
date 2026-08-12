#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# #485/C40 host-memory sampler: the corridor sampler's counterpart on the HOST
# side, for the failure mode where a rank dies with GPU memory in hand.
#
# WHY THIS EXISTS. A rank that is SIGKILLed by the kernel's cgroup OOM killer
# leaves no traceback, no core and no in-container kernel record: inside an LXC
# container /dev/kmsg does not exist and dmesg is not permitted, so the only
# in-container trace of such a kill is the cgroup's cumulative oom_kill
# counter. That counter is useless after the fact -- it carries no timestamp
# and no victim -- so it has to be SAMPLED while the run is happening. This
# script does that, and records which rank pids are alive at every sample, so
# a vanished pid can be dated to the second and joined against the counter.
#
# Reads only. Writes one CSV and prints a verdict.
#
# Usage:  bash scripts/hostmem_sample.sh <seconds> [out.csv]
set -euo pipefail

SECS="${1:-600}"
OUT="${2:-/tmp/hostmem_$(date -u +%H%M%S).csv}"
INTERVAL="${INTERVAL:-1}"
# Match the rank processes by their argv-visible comm. NEVER pattern-kill on
# this -- it is a read-only join key.
RANK_COMM="${RANK_COMM:-sglang::schedul}"

CG=/sys/fs/cgroup
[ -r "$CG/.lxc/memory.stat" ] && CG="$CG/.lxc"

read_counter() { grep -E "^oom_kill " "$CG/memory.events" 2>/dev/null | awk '{print $2}'; }
read_stat() { awk -v k="$1" '$1==k {printf "%d", $2/1048576}' "$CG/memory.stat" 2>/dev/null; }

OOM_START="$(read_counter)"; OOM_START="${OOM_START:-0}"
echo "cgroup=$CG  oom_kill at start=$OOM_START" >&2

echo "ts_ms,mem_current_mib,anon_mib,shmem_mib,file_mib,mem_available_mib,oom_kill,n_ranks,rank_pids,rank_rss_mib" > "$OUT"
END=$(( $(date +%s) + SECS ))
PREV_PIDS=""
PREV_OOM="$OOM_START"

while [ "$(date +%s)" -lt "$END" ]; do
    TS=$(date +%s%3N)
    CUR=$(( $(cat "$CG/memory.current" 2>/dev/null || echo 0) / 1048576 ))
    ANON=$(read_stat anon); SHMEM=$(read_stat shmem); FILE=$(read_stat file)
    AVAIL=$(awk '/^MemAvailable:/{printf "%d", $2/1024}' /proc/meminfo)
    OOM=$(read_counter); OOM="${OOM:-0}"

    PIDS=""; RSS=""
    for p in $(pgrep -x "$RANK_COMM" 2>/dev/null || true); do
        r=$(awk '/^VmRSS:/{printf "%d", $2/1024}' "/proc/$p/status" 2>/dev/null || true)
        [ -n "$r" ] && { PIDS="$PIDS $p"; RSS="$RSS $r"; }
    done
    PIDS="${PIDS# }"; RSS="${RSS# }"
    N=$(echo "$PIDS" | wc -w)

    echo "$TS,$CUR,$ANON,$SHMEM,$FILE,$AVAIL,$OOM,$N,${PIDS// /|},${RSS// /|}" >> "$OUT"

    # Event lines go to stdout so a monitor can act on them the moment they
    # happen; the CSV is the record.
    if [ "$OOM" != "$PREV_OOM" ]; then
        echo "OOM_KILL_EVENT $(date -u +%H:%M:%SZ) counter $PREV_OOM -> $OOM  mem_current=${CUR}MiB avail=${AVAIL}MiB"
        PREV_OOM="$OOM"
    fi
    if [ -n "$PREV_PIDS" ] && [ "$PIDS" != "$PREV_PIDS" ]; then
        for q in $PREV_PIDS; do
            case " $PIDS " in *" $q "*) ;; *)
                echo "RANK_PID_GONE $(date -u +%H:%M:%SZ) pid=$q  oom_kill=$OOM (start $OOM_START)  mem_current=${CUR}MiB avail=${AVAIL}MiB" ;;
            esac
        done
    fi
    PREV_PIDS="$PIDS"
    sleep "$INTERVAL"
done

OOM_END="$(read_counter)"; OOM_END="${OOM_END:-0}"
awk -F, -v s="$OOM_START" -v e="$OOM_END" '
NR>1 {
    n++
    if ($2 > peak) peak = $2
    if (minavail == 0 || $6 < minavail) minavail = $6
    if ($4 > peakshm) peakshm = $4
    if (maxranks == 0 || $8 > maxranks) maxranks = $8
    if (n == 1 || $8 < minranks) minranks = $8
}
END {
    printf "samples=%d\n", n
    printf "  cgroup memory.current  peak=%d MiB\n", peak
    printf "  shmem                  peak=%d MiB\n", peakshm
    printf "  MemAvailable           min =%d MiB\n", minavail
    printf "  rank processes         max =%d  min=%d\n", maxranks, minranks
    printf "  oom_kill               %d -> %d  (delta %d)\n", s, e, e - s
    if (e > s) print "  VERDICT: THE KERNEL OOM-KILLED SOMETHING IN THIS CGROUP DURING THE RUN"
    else if (minranks < maxranks) print "  VERDICT: a rank process vanished and it was NOT a cgroup OOM kill"
    else print "  VERDICT: all rank processes survived, no cgroup OOM kill"
}' "$OUT"
echo "series: $OUT"
