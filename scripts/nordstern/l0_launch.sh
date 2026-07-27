#!/bin/bash
# Nordstern L0 -- start all five ranks: 0-2 here, 3-4 on the second host.
# Run l0_preflight.sh first and get GREEN.
#
# Env: STAGE (s1|s2|s3|s4), RATIO, CTX, LOGDIR,
#      L0_MIN_FREE_MIB   free-RAM floor before each local rank (default 20000)
#      L0_LOCAL_ONLY=1   skip the second host (used by test_l0_guardrails.sh)
#      L0_RANK_SCRIPT    override the per-rank script (tests inject a fake)
#      L0_READY_TIMEOUT  seconds to wait for rank 0 to serve (default 750)
#
# GUARDRAILS (see l0_lib.sh for the incident these come from):
#   * this script NEVER returns while ranks are still starting -- on any
#     non-READY outcome it kills its own tagged ranks and waits until they are
#     gone. Returning early is what orphaned two ranks and killed the container
#     on 2026-07-25;
#   * every rank carries L0_RUN_TAG in its environment, and every kill is
#     restricted to that tag -- never a cmdline pattern, on a shared box;
#   * local ranks start STAGGERED and behind a free-RAM floor;
#   * orphans (PPID 1) are detected live and abort the run.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=l0_lib.sh
source "$HERE/l0_lib.sh"

# Host and key come from the environment (RIG2_HOST, RIG2_KEY); source your
# local rig env file first. The placeholder fallbacks make an unsourced run
# fail at the first ssh rather than reaching some other machine.
SECOND=${SECOND:-${RIG2_HOST:-<RIG2_IP>}}
KEY=${KEY:-${RIG2_KEY:-<RIG2_SSH_KEY>}}
SSH="ssh -i $KEY -o IdentitiesOnly=yes -o ConnectTimeout=10"
STAGE=${STAGE:-s1}; RATIO=${RATIO:-4,3,3,2,1}; CTX=${CTX:-4096}
LOGDIR=${LOGDIR:-/tmp/nordstern}; mkdir -p "$LOGDIR"; rm -f "$LOGDIR"/r*.log
R=${L0_RANK_SCRIPT:-$HERE/l0_rank.sh}
MIN_FREE=${L0_MIN_FREE_MIB:-20000}
READY_TIMEOUT=${L0_READY_TIMEOUT:-750}
LOCAL_ONLY=${L0_LOCAL_ONLY:-0}

RUN_TAG=$(l0_new_run_tag)
export L0_RUN_TAG=$RUN_TAG
echo "l0: run tag $RUN_TAG (every rank carries it; kills are restricted to it)"

# One exit path, always. Whatever happens below -- crash, timeout, Ctrl-C --
# control comes through here, and here we do not leave until our own ranks are
# gone or explicitly kept.
KEEP_ON_READY=${KEEP_ON_READY:-1}
finish() {
    local verdict="$1"
    if [ "$verdict" = READY ] && [ "$KEEP_ON_READY" = 1 ]; then
        echo "READY"
        # The server is meant to outlive this script; that is safe now because
        # each rank is supervised by its own l0_rank.sh, so nothing the server
        # can crash into has PID 1 as its parent.
        exit 0
    fi
    echo "l0: verdict=$verdict -- taking this run's ranks down and WAITING for them"
    l0_kill_tagged "$RUN_TAG" 90
    if [ "$LOCAL_ONLY" != 1 ]; then
        $SSH root@$SECOND "L0_RUN_TAG='$RUN_TAG' bash -s" <<'REMOTE' 2>/dev/null
source /root/nordstern/l0_lib.sh
l0_kill_tagged "$L0_RUN_TAG" 90
REMOTE
    fi
    echo "$verdict"
    [ "$verdict" = READY ] && exit 0
    exit 1
}
trap 'finish INTERRUPTED' INT TERM

if [ "$LOCAL_ONLY" != 1 ]; then
    $SSH root@$SECOND "mkdir -p /root/nordstern && rm -f /root/nordstern/r*.log"
    scp -q -i "$KEY" -o IdentitiesOnly=yes "$HERE/l0_rank.sh" "$HERE/l0_lib.sh" \
        root@$SECOND:/root/nordstern/
    # ranks 4 and 3 first: slowest to import; a late joiner is cheaper than a
    # rendezvous master that has already timed out.
    for r in 4 3; do
        # Forward EVERY knob that shapes the run. MAXTOK was missing here, so
        # `MAXTOK=0` (uncapped) applied to the main rig while ranks 3/4 silently
        # kept the default cap: rank 0 sized its KV pool to 133802 tokens and
        # ranks 3/4 to 2048, which is a capacity number that means nothing.
        # Measured on the S4 run of window #2.
        $SSH root@$SECOND "RANK=$r SIDE=second STAGE=$STAGE RATIO='$RATIO' CTX=$CTX \
           MAXTOK='${MAXTOK:-}' MEMFRAC='${MEMFRAC:-}' EXTRA='${EXTRA:-}' \
           L0_RUN_TAG='$RUN_TAG' \
           nohup setsid /root/nordstern/l0_rank.sh > /root/nordstern/r$r.log 2>&1 < /dev/null &" &
    done
    sleep 6
fi

# STAGGERED local starts.
#
# The gate is "Init torch distributed begin.", NOT "Load weight end". That is
# not a detail: weight loading happens AFTER the five-rank rendezvous, so a
# rank cannot reach it until every other rank has joined. Gating rank 1 on rank
# 0's weight load would deadlock the group by construction. The marker below is
# the last thing a rank prints alone, and it is exactly the expensive part
# worth serialising on a swapless box -- python import, torch, CUDA context.
for r in 0 1 2; do
    l0_require_ram "$MIN_FREE" || finish RAM_FLOOR
    RANK=$r SIDE=main STAGE=$STAGE RATIO="$RATIO" CTX=$CTX L0_RUN_TAG="$RUN_TAG" \
        nohup "$R" > "$LOGDIR/r$r.log" 2>&1 < /dev/null &
    echo "l0: local rank $r started (supervisor pid $!)"
    # NOTE: called directly, not in $(...). Wrapping it in a command
    # substitution swallowed both its message and its exit status, so the
    # crash branch below could never fire -- caught by
    # test_l0_guardrails.sh case 1 before this ever reached a GPU window.
    l0_wait_for_marker "$LOGDIR/r$r.log" \
        "${L0_STAGGER_MARKER:-Init torch distributed begin.}" \
        "${L0_STAGGER_TIMEOUT:-240}"
    case $? in
        2) finish CRASH_ON_START ;;
        1) echo "l0: rank $r did not reach the stagger marker; continuing anyway" ;;
    esac
done
echo "l0: five ranks launched (stage=$STAGE ratio=$RATIO ctx=$CTX); watching rank 0"

waited=0
while [ "$waited" -lt "$READY_TIMEOUT" ]; do
    grep -qF "fired up and ready to roll" "$LOGDIR/r0.log" 2>/dev/null && finish READY
    grep -qE "^Traceback|Received sigquit|Scheduler hit an exception" "$LOGDIR"/r*.log 2>/dev/null \
        && finish CRASHED
    # Live orphan check: a tagged process whose parent is init is one crash away
    # from taking the container down.
    l0_check_orphans "$RUN_TAG" || finish ORPHANED
    sleep 5; waited=$((waited + 5))
done
finish TIMEOUT
