#!/usr/bin/env bash
# #631: WHICH LAYOUT IS THE INSTANCE IN RIGHT NOW?
#
# Written because the question was asked twice from the outside and there
# was no way to answer it except by grepping the serving log by hand.
#
# WHY UTILISATION CANNOT ANSWER IT, which is the trap this exists for:
# under PP with pp_size=3 the stages are PIPELINED, so during a long
# chunked prefill stage 0 computes chunk n+1 while stage 1 computes chunk
# n -- all three cards sit near 100% simultaneously, exactly as they do
# under TP. High simultaneous utilisation is therefore NOT evidence of the
# TP layout, and reading the layout off nvidia-smi will mislead you.
#
# The authoritative record is the cutover line, whose single writer is
# _cutover in phase_flip_runtime.py:
#   "PHASE-FLIP cutover complete: active stack tp, ps tp=3 pp=1"
# Before the first flip of a boot there is no such line, and the instance
# is in its BOOT layout, which for a Route A boot is PP.
set -euo pipefail

LOG="${SERVING_LOG:-/spinning/serving-30030.boot.log}"

if [ ! -s "$LOG" ]; then
    echo "phase: UNKNOWN (no serving log at $LOG)"
    exit 2
fi

line="$(grep -a 'cutover complete: active stack' "$LOG" | tail -1 || true)"
if [ -z "$line" ]; then
    echo "phase: pp (boot layout; no flip has committed on this boot)"
    exit 0
fi

phase="$(printf '%s' "$line" | sed -n 's/.*active stack \([a-z]*\).*/\1/p')"
when="$(printf '%s' "$line" | sed -n 's/^\[\([0-9-]* [0-9:]*\).*/\1/p')"
echo "phase: ${phase:-UNKNOWN}   (since $when)"

# The standing policy reason, when the automatic policy is running: it
# says WHY the instance is in this layout rather than the other one.
hold="$(grep -aE 'PHASE-POLICY (holding|arming)' "$LOG" | tail -1 || true)"
[ -n "$hold" ] && printf 'policy: %s\n' \
    "$(printf '%s' "$hold" | sed 's/^\[[^]]*\] *//')"
exit 0
