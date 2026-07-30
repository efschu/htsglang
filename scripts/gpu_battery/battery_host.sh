#!/usr/bin/env bash
# Host-side plumbing for the BAR1 steps (s10-s12).
#
# The BAR1 work does NOT run where the battery runs. The patched driver, the
# holder module and /dev/dmabuf_holder live on the PVE host; CT999 cannot open
# the holder device at all (major 10 is not in the container's device
# allowlist). So s10-s12 drive the host over ssh while the battery, its state
# and its artifacts stay in the container.
#
# Two consequences run through everything in this file:
#
#   * TWO /tmp NAMESPACES. /tmp/gpu-card-N.lock in CT999 and the identically
#     named directory on the host are DIFFERENT locks. An agent in the
#     container and a session on the host do not see each other's arbitration.
#     The BAR1 steps therefore take BOTH: run_step.sh takes the container-side
#     locks (step table: locks="battery"), and host_locks_acquire takes the
#     host-side ones. A foreign lock on either side is never broken.
#   * EVERY ssh CALL IS BOUNDED. An ssh that blocks forever inside one bash
#     call makes the agent unresponsive without anyone seeing it, which is the
#     wedge trap in its purest form. host_ssh always carries a local timeout, a
#     ConnectTimeout and keepalives.
#
# Path mapping: the container's root filesystem is the host's
# $BAR1_HOST_SUBVOL, so EVERY absolute container path has a host path, and
# both directions are writable. That is what lets the host write measurement
# results straight into the run directory while the big server logs stay on
# the host.
#
# NOT executable on its own. Sourced by the BAR1 step scripts.

set -uo pipefail

BAR1_HOST="${BAR1_HOST:-192.168.0.1}"
BAR1_HOST_KEY="${BAR1_HOST_KEY:-/root/.ssh/id_root@proxmox}"
BAR1_HOST_SUBVOL="${BAR1_HOST_SUBVOL:-/spinning/subvol-999-disk-0}"

# Where the host keeps what must NOT come into an agent's context: server logs,
# JIT build noise, NCCL debug output. Host-local on purpose.
BAR1_HOST_LOGDIR="${BAR1_HOST_LOGDIR:-/root/battery-bar1}"

# The patched driver tree and the holder module, as CONTAINER paths (they get
# mapped). Both come straight from 04_BETRIEB.md of the P2P handover.
BAR1_NV_SOURCE="${BAR1_NV_SOURCE:-/spinning/nvidia-open-595}"
BAR1_HOLDER_KO="${BAR1_HOLDER_KO:-/spinning/nvidia-smallbar-p2p/dmabuf_holder/dmabuf_holder.ko}"
BAR1_REGKEY="${BAR1_REGKEY:-RMSmallBarP2PPeerBar1=1}"

# The JIT extension cache. Shared across boots on purpose: a cold build of the
# BAR1 extension costs minutes and would be paid eight times in s12.
BAR1_EXTCACHE="${BAR1_EXTCACHE:-/spinning/htccl_extcache_host}"

BAR1_PORT="${BAR1_PORT:-30030}"

# Terminating a viewer is a USER decision and it does not carry over between
# runs (05_FALLEN: get the user's approval, it does not hold permanently).
# Default off: the step stops and names the pids instead of killing anything.
BAR1_VIEWER_KILL_OK="${BAR1_VIEWER_KILL_OK:-0}"

export BAR1_HOST BAR1_HOST_KEY BAR1_HOST_SUBVOL BAR1_HOST_LOGDIR
export BAR1_NV_SOURCE BAR1_HOLDER_KO BAR1_REGKEY BAR1_EXTCACHE BAR1_PORT
export BAR1_VIEWER_KILL_OK

# --- path mapping -----------------------------------------------------------
# host_path <absolute container path> -> the same file seen from the host.
host_path() {
    local p="$1"
    case "$p" in
        /*) printf '%s%s\n' "$BAR1_HOST_SUBVOL" "$p" ;;
        *) echo "host_path: absolute paths only ($p)" >&2; return 1 ;;
    esac
}

# --- bounded ssh ------------------------------------------------------------
# One place, one policy. BATTERY_HOST_SSH_S is the wall for a single call; a
# step that needs longer passes it explicitly rather than dropping the bound.
host_ssh() {  # $@ = remote command
    local budget="${BATTERY_HOST_SSH_S:-120}"
    # -n is not cosmetic: ssh reads stdin by default, and an ssh inside a
    # `while read ... done < file` loop swallows the rest of that file. The
    # loop over the viewer pids in s10 is exactly that shape.
    timeout --signal=TERM --kill-after=15 "$budget" \
        ssh -n -i "$BAR1_HOST_KEY" \
            -o BatchMode=yes \
            -o StrictHostKeyChecking=accept-new \
            -o ConnectTimeout=10 \
            -o ServerAliveInterval=15 \
            -o ServerAliveCountMax=4 \
            "root@$BAR1_HOST" "$@"
}

host_ssh_for() {  # $1 = seconds, rest = remote command
    local budget="$1"; shift
    BATTERY_HOST_SSH_S="$budget" host_ssh "$@"
}

# Runs a script that lives in the run directory (container side) on the host,
# through the path mapping. The script is an ARTIFACT: what ran is readable
# afterwards, byte for byte, next to its output.
host_run_script() {  # $1 = seconds, $2 = container path of the script, $3.. = args
    local budget="$1" script="$2" remote
    shift 2
    remote="$(host_path "$script")" || return 2
    host_ssh_for "$budget" "bash $remote $*"
}

# grep -c prints "0" and exits 1 on no match; taking the exit code for the
# count is how a "0" becomes a "00" and an empty list starts looking occupied.
host_count_nonempty() {  # $1 = file
    local n=0
    if [ -f "$1" ]; then
        n="$(grep -c . "$1" 2>/dev/null)" || n=0
    fi
    printf '%s\n' "${n:-0}"
}

host_reachable() {
    host_ssh_for 30 true >/dev/null 2>&1
}

# --- host locks -------------------------------------------------------------
# Same convention as battery_common.sh, different /tmp. Directories, mkdir is
# the atomic acquire, an info file carries holder and heartbeat.
BATTERY_HOST_LOCKS=()
BATTERY_HOST_HEARTBEAT_PID=""

host_lock_count() {
    local n
    n="$(host_ssh_for 60 'nvidia-smi -L 2>/dev/null | grep -c "^GPU"' 2>/dev/null)"
    n="${n//[^0-9]/}"
    printf '%s\n' "${n:-0}"
}

host_locks_release() {
    if [ -n "$BATTERY_HOST_HEARTBEAT_PID" ]; then
        kill "$BATTERY_HOST_HEARTBEAT_PID" 2>/dev/null
        BATTERY_HOST_HEARTBEAT_PID=""
    fi
    local d
    for d in ${BATTERY_HOST_LOCKS[@]+"${BATTERY_HOST_LOCKS[@]}"}; do
        host_ssh_for 60 "rm -rf $d" >/dev/null 2>&1
    done
    BATTERY_HOST_LOCKS=()
}

host_locks_acquire() {  # $1 = step id
    local step="$1" n i lock info
    n="$(host_lock_count)"
    if [ "${n:-0}" -lt 1 ]; then
        echo "STOP: the host reports no GPU (nvidia-smi -L is empty)" >&2
        return 2
    fi
    for i in $(seq 0 $((n - 1))); do
        lock="/tmp/gpu-card-$i.lock"
        if host_ssh_for 60 "mkdir $lock 2>/dev/null" >/dev/null 2>&1; then
            host_ssh_for 60 "printf 'holder=gpu_battery_host\nstep=%s\nfrom=CT999\nacquired=%s\nheartbeat=%s\n' \
                '$step' \"\$(date -Is)\" \"\$(date -Is)\" > $lock/info" >/dev/null 2>&1
            BATTERY_HOST_LOCKS+=("$lock")
        else
            info="$(host_ssh_for 60 "cat $lock/info 2>/dev/null" 2>/dev/null)"
            echo "STOP: host lock $lock is held:" >&2
            printf '%s\n' "$info" | sed 's/^/    /' >&2
            echo "foreign locks are never broken -- ask the operator." >&2
            host_locks_release
            return 2
        fi
    done
    (
        while true; do
            for d in ${BATTERY_HOST_LOCKS[@]+"${BATTERY_HOST_LOCKS[@]}"}; do
                host_ssh_for 60 "sed -i \"s/^heartbeat=.*/heartbeat=\$(date -Is)/\" $d/info" \
                    >/dev/null 2>&1
            done
            sleep 60
        done
    ) &
    BATTERY_HOST_HEARTBEAT_PID=$!
    return 0
}

# --- host process hygiene ---------------------------------------------------
# py-spy runs ON THE HOST, before any kill, out of the container venv (the host
# has no py-spy of its own; the venv binary runs there unchanged). A hang whose
# stack nobody captured has to be reproduced.
host_dump_and_kill() {  # $1 = host pid, $2 = container path for the dump
    local pid="$1" dump="$2" pyspy
    [ -z "$pid" ] && return 0
    pyspy="$(host_path "$VENV/bin/py-spy")" || return 0
    # Nothing there (or the host is unreachable) -- there is nothing to dump and
    # nothing to kill either way.
    if ! host_ssh_for 60 "kill -0 $pid 2>/dev/null" >/dev/null 2>&1; then
        return 0
    fi
    host_ssh_for 120 "timeout 60 $pyspy dump --pid $pid 2>&1" > "$dump" 2>&1 || true
    # The server is a setsid process group leader; the group takes the signal so
    # the tp workers go with it. Only ever OUR pid, never a pattern.
    host_ssh_for 90 "kill -TERM -$pid 2>/dev/null; kill -TERM $pid 2>/dev/null; sleep 8; \
                     kill -9 -$pid 2>/dev/null; kill -9 $pid 2>/dev/null; true" >/dev/null 2>&1
    return 0
}

# Bounded wait for a server that listens on the HOST's loopback. Every curl
# carries -m and the loop carries a budget.
host_wait_for_server() {  # $1 = port, $2 = budget_s
    local port="$1" budget="${2:-900}" t0
    t0=$(date +%s)
    while [ $(( $(date +%s) - t0 )) -lt "$budget" ]; do
        if host_ssh_for 40 "curl -sf -m 5 http://127.0.0.1:$port/health >/dev/null" \
            >/dev/null 2>&1; then
            echo "host server up after $(( $(date +%s) - t0 ))s"
            return 0
        fi
        sleep 10
    done
    echo "host server not up within ${budget}s" >&2
    return 1
}

# --- remote grep ------------------------------------------------------------
# The server log stays on the host. What comes into the run directory is the
# GREP RESULT plus a bounded tail -- never the whole log, and never into an
# agent's context.
#
# Lines the server QUOTED from a helper subprocess it ran, caught and recovered
# from (the stage-0 hardware probe above all) are dropped from every harvest.
# They carry the emitter's marker; a fatal pattern inside one describes the
# helper, not this boot, and harvesting it is how a deliberately killed probe
# scored as a BATTERY-FAIL. Keep the literal in step with
# uneven_perf.QUOTED_SUBLOG_PREFIX / check_common.QUOTED_SUBLOG_PREFIX.
HOST_QUOTED_SUBLOG_PREFIX='[probe-subprocess] '
host_grep_into() {  # $1 = host log path, $2 = out file (container), $3.. = patterns
    local log="$1" out="$2"; shift 2
    local pat args=""
    for pat in "$@"; do
        args="$args -e '$pat'"
    done
    : > "$out"
    host_ssh_for 120 "grep -n -F $args $log 2>/dev/null \
        | grep -v -F -e '$HOST_QUOTED_SUBLOG_PREFIX' | head -400" >> "$out" 2>/dev/null
    return 0
}

host_tail_into() {  # $1 = host log path, $2 = out file (container), $3 = lines
    local log="$1" out="$2" lines="${3:-400}"
    host_ssh_for 120 "tail -n $lines $log 2>/dev/null" > "$out" 2>/dev/null
    return 0
}
