#!/bin/bash
# Nordstern L0 -- shared guardrail helpers. Sourced by l0_launch.sh, l0_rank.sh
# and test_l0_guardrails.sh so there is ONE definition of each rule.
#
# WHY THIS FILE EXISTS
# --------------------
# On 2026-07-25 the main rig's LXC container died mid-campaign. It was not an
# OOM. The journal of the dead boot ends with:
#
#   systemd[1]: Caught <QUIT> from PID 273445.
#   systemd[1]: Caught <QUIT>, dumped core as pid 273622.
#   systemd[1]: Exiting PID 1...
#
# sglang signals its PARENT when a scheduler dies
# (managers/scheduler.py: parent_process.send_signal(signal.SIGQUIT), parent
# resolved via os.getppid()), and kill_process_tree sends SIGQUIT too
# (utils/common.py, whose own comment names PID 1 as a case it expects to hit).
# The launch script had returned while two ranks were still starting; those
# ranks had been detached with `nohup setsid`, so they were reparented to init.
# Their crash signal therefore went to the container's PID 1, which dumped core
# and exited, taking the container -- and every other agent on the box -- down.
#
# The rules below exist to make that unreachable. The load never mattered:
# 108 GB, no swap, and every peak-RAM moment that day was survived.

# ---------------------------------------------------------------------------
# Tagged process identification.
#
# NEVER pattern-kill on this box: it is shared, and `pkill -f sglang` has twice
# taken down someone else's server (and once the killer's own session). Every
# rank this tooling starts carries L0_RUN_TAG in its ENVIRONMENT, and every
# lookup and every kill is restricted to processes carrying THIS run's tag.
# Environment matching cannot collide with another agent's processes the way a
# cmdline pattern can.
# ---------------------------------------------------------------------------

l0_new_run_tag() {
    echo "l0-$(date +%s)-$$-${RANDOM}"
}

# Every live PID whose environment carries the given tag, EXCLUDING the caller
# and its own ancestry.
#
# The exclusion is not cosmetic. The launcher exports L0_RUN_TAG so its ranks
# inherit it -- which means the launcher itself, and every command-substitution
# subshell it spawns, also carry the tag. Without this, l0_kill_tagged TERMs
# the launcher mid-cleanup and then waits forever for a process that is itself:
# measured as "1 tagged process still alive after 90s" in
# test_l0_guardrails.sh case 2, before any of this reached a GPU window.
l0_tagged_pids() {
    local tag="$1" p pid skip walk
    [ -n "$tag" ] || return 0
    # self + ancestors: BASHPID is this subshell, $$ the shell that sourced us.
    skip=" $BASHPID $$ "
    walk=$BASHPID
    while [ -n "$walk" ] && [ "$walk" != "1" ] && [ "$walk" != "0" ]; do
        walk=$(awk '{print $4}' "/proc/$walk/stat" 2>/dev/null)
        [ -n "$walk" ] && skip="$skip$walk "
    done
    # PURE BASH from here on: no tr, no grep, no pipeline. Helper processes
    # spawned during the scan INHERIT L0_RUN_TAG and are then found by the scan
    # itself, so the set never empties -- measured as a permanent "1 tagged
    # process still alive after 90s" in test_l0_guardrails.sh case 2. A matcher
    # that forks cannot count its own tag correctly.
    local kv
    for p in /proc/[0-9]*; do
        pid=${p#/proc/}
        case "$skip" in *" $pid "*) continue ;; esac
        [ -r "$p/environ" ] || continue
        while IFS= read -r -d '' kv; do
            if [ "$kv" = "L0_RUN_TAG=$tag" ]; then
                echo "$pid"
                break
            fi
        done < "$p/environ" 2>/dev/null
    done
}

# Count without a pipeline, for the same reason.
l0_tagged_count() {
    local -a pids=()
    mapfile -t pids < <(l0_tagged_pids "$1")
    echo "${#pids[@]}"
}

l0_ppid_of() {
    awk '{print $4}' "/proc/$1/stat" 2>/dev/null
}

l0_cmd_of() {
    tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | cut -c1-120
}

# ---------------------------------------------------------------------------
# THE ORPHAN CHECK -- the live form of the rule that was violated.
#
# Any tagged process whose PPID is 1 has been reparented to init, which means
# its next crash signals init instead of a supervisor. Report it loudly; the
# caller aborts. Returns 0 when clean, 1 when at least one orphan was found.
# ---------------------------------------------------------------------------
l0_check_orphans() {
    local tag="$1" pid ppid found=0
    for pid in $(l0_tagged_pids "$tag"); do
        ppid=$(l0_ppid_of "$pid")
        if [ "$ppid" = "1" ]; then
            echo "ORPHAN: pid $pid has PPID 1 -- reparented to init." >&2
            echo "        cmd: $(l0_cmd_of "$pid")" >&2
            echo "        Its crash path would send SIGQUIT to PID 1 and kill" >&2
            echo "        the container (measured 2026-07-25 20:48:34)." >&2
            found=1
        fi
    done
    return $found
}

# ---------------------------------------------------------------------------
# Wait until every tagged process is GONE, bounded.
#
# The launcher must not return while ranks are still coming up: that is exactly
# what orphaned the two ranks that killed the container. On any non-READY exit
# it kills the tagged set and then waits here until the set is empty.
# ---------------------------------------------------------------------------
l0_kill_tagged() {
    local tag="$1" timeout="${2:-60}" pid waited=0 remaining
    for pid in $(l0_tagged_pids "$tag"); do
        kill -TERM "$pid" 2>/dev/null
    done
    while [ "$waited" -lt "$timeout" ]; do
        remaining=$(l0_tagged_count "$tag")
        [ "$remaining" -eq 0 ] && { echo "l0: all tagged processes gone after ${waited}s"; return 0; }
        sleep 2; waited=$((waited + 2))
        if [ "$waited" -eq 20 ]; then
            for pid in $(l0_tagged_pids "$tag"); do kill -KILL "$pid" 2>/dev/null; done
        fi
    done
    echo "l0: WARNING -- $(l0_tagged_count "$tag") tagged process(es) still alive after ${timeout}s" >&2
    for pid in $(l0_tagged_pids "$tag"); do echo "     pid $pid: $(l0_cmd_of "$pid")" >&2; done
    return 1
}

# ---------------------------------------------------------------------------
# Free-RAM gate. This box has NO SWAP, so a miscalculation is an OOM kill, not
# a slowdown. MemAvailable (not MemFree) is the right number: it accounts for
# reclaimable page cache, which a 16 GB checkpoint read fills.
#
# Not the cause of the 2026-07-25 crash -- included because it is cheap and
# removes a real risk class on a swapless host.
# ---------------------------------------------------------------------------
l0_avail_mib() {
    awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo
}

l0_require_ram() {
    local need="$1" have
    have=$(l0_avail_mib)
    if [ "${have:-0}" -lt "$need" ]; then
        echo "l0: REFUSING to start a rank -- MemAvailable ${have} MiB < required ${need} MiB." >&2
        echo "    This host has no swap; starting anyway means the OOM killer picks" >&2
        echo "    the victim, not you. Lower L0_MIN_FREE_MIB only with a reason." >&2
        return 1
    fi
    echo "l0: MemAvailable ${have} MiB >= ${need} MiB required"
    return 0
}

# ---------------------------------------------------------------------------
# Wait for a marker line in a log, bounded. Used to STAGGER local rank starts:
# three concurrent 16 GB checkpoint reads on a swapless box is a risk taken for
# no benefit, since the ranks rendezvous anyway.
# Returns 0 on marker, 1 on timeout, 2 if the log shows a crash first.
# ---------------------------------------------------------------------------
l0_wait_for_marker() {
    local log="$1" marker="$2" timeout="${3:-300}" waited=0
    while [ "$waited" -lt "$timeout" ]; do
        if [ -f "$log" ] && grep -qF "$marker" "$log" 2>/dev/null; then
            echo "l0: '$marker' seen in $(basename "$log") after ${waited}s"; return 0
        fi
        if [ -f "$log" ] && grep -qE "^Traceback|Received sigquit|Scheduler hit an exception" "$log" 2>/dev/null; then
            echo "l0: $(basename "$log") crashed before '$marker'" >&2; return 2
        fi
        sleep 3; waited=$((waited + 3))
    done
    echo "l0: timeout (${timeout}s) waiting for '$marker' in $(basename "$log")" >&2
    return 1
}
