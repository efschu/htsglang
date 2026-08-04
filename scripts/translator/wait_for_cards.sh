#!/bin/bash
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
#
# Wait until the GPUs are genuinely free, then exit 0. Bounded; never loops
# forever.
#
# WHY THIS EXISTS AS A SCRIPT. The first version of this watcher was one shell
# line and it LIED: it summed `nvidia-smi` output with awk and compared against
# a threshold, so a transient query failure produced an empty string, awk
# summed it to 0, and 0 < 1500 reported "cards free" while all three cards were
# in fact holding 16-28 GB. Acting on that would have taken cards out from
# under another agent's window.
#
# That is the erfolgsmeldung-vs-zustand class from CLAUDE.md in its purest
# form: a measurement counts only after the instrument is PROVEN to have
# worked. A missing reading is not a low reading.
#
# So three independent conditions must all hold, and each is checked rather
# than assumed:
#
#   1. the query SUCCEEDED  -- exit code 0, exactly $EXPECTED_CARDS lines, and
#      every line numeric. Anything else is "unknown", never "free".
#   2. the reading is STABLE -- two consecutive successful readings below the
#      threshold, at least $STABLE_GAP_S apart, so a momentary dip between two
#      of another agent's boots does not fire.
#   3. arbitration agrees   -- /spinning/gpu-arb/holder is absent. Hardware and
#      arbitration must BOTH report free; either one alone can be stale.
#
# Usage:  wait_for_cards.sh [timeout_s] [threshold_mib]
# Exit:   0 cards free   1 timed out   2 usage error

set -u

TIMEOUT_S="${1:-3600}"
THRESHOLD_MIB="${2:-1500}"
EXPECTED_CARDS="${EXPECTED_CARDS:-3}"
HOLDER="${HOLDER:-/spinning/gpu-arb/holder}"
POLL_S="${POLL_S:-60}"
STABLE_GAP_S="${STABLE_GAP_S:-10}"
SMI="${SMI:-nvidia-smi}"

# Returns 0 and echoes the total MiB in use, or returns 1 and echoes nothing.
# Every failure mode of the query collapses to "return 1" -- there is no path
# on which an unreadable card contributes 0 to the sum.
read_total_mib() {
    local out rc lines total line
    out=$("$SMI" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    rc=$?
    [ "$rc" -ne 0 ] && return 1
    [ -z "$out" ] && return 1
    lines=$(printf '%s\n' "$out" | grep -c .)
    [ "$lines" -ne "$EXPECTED_CARDS" ] && return 1
    total=0
    while IFS= read -r line; do
        line="${line// /}"
        # Reject anything that is not a plain integer, e.g. "[N/A]" from a
        # card in a bad state -- which is precisely a card we must not
        # declare free.
        case "$line" in
            ''|*[!0-9]*) return 1 ;;
        esac
        total=$((total + line))
    done <<< "$out"
    printf '%s' "$total"
    return 0
}

deadline=$(( $(date +%s) + TIMEOUT_S ))
confirmed_at=0

while [ "$(date +%s)" -lt "$deadline" ]; do
    if total=$(read_total_mib); then
        now=$(date +%s)
        if [ "$total" -lt "$THRESHOLD_MIB" ]; then
            if [ "$confirmed_at" -ne 0 ] \
               && [ $((now - confirmed_at)) -ge "$STABLE_GAP_S" ]; then
                if [ -e "$HOLDER" ]; then
                    echo "hardware free (${total} MiB) but holder still present; waiting"
                    confirmed_at="$now"
                else
                    echo "CARDS-FREE total=${total}MiB confirmed twice, holder absent"
                    exit 0
                fi
            else
                [ "$confirmed_at" -eq 0 ] && confirmed_at="$now"
                echo "low reading ${total} MiB; awaiting a second confirmation"
            fi
        else
            # Any reading at or above the threshold resets the streak: two
            # confirmations must be CONSECUTIVE, not merely two ever seen.
            confirmed_at=0
            echo "in use: ${total} MiB"
        fi
    else
        # Unknown, not free. Also resets the streak.
        confirmed_at=0
        echo "nvidia-smi query FAILED or malformed; treating as OCCUPIED"
    fi
    sleep "$POLL_S"
done

echo "timed out after ${TIMEOUT_S}s without a confirmed free reading"
exit 1
