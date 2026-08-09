#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# #631 phase-evidence extract: prove prefill ran in PP and decode in TP.
#
# The throughput signature alone (~4200-5500 tok/s PP vs ~1500 TP) is
# suggestive, not proof. This CORRELATES every prefill/decode record with
# the active stack at that moment, by tracking the most recent
# "active stack now <pp|tp>" line on rank PP0 and tagging each record with
# it. Nothing is inferred from the rate.
#
# Usage: bash scripts/phase_evidence_extract.sh [logfile]
set -euo pipefail

LOG="${1:-/spinning/serving-30030.boot.log}"

# ONE RANK ONLY (PP0). All three ranks log both the stack marker and the
# batch records, and they do not interleave in a fixed order, so counting
# every rank against a single global "current stack" mis-tags records
# around each cutover and triples the counts. Rank 0 is the intake rank
# and sees every record.
grep -F "PP0]" "$LOG" | awk '
/\(active stack now pp\)/ { stack = "PP" }
/\(active stack now tp\)/ { stack = "TP" }
/Prefill batch/ && stack != "" {
    tok = ""; rate = ""
    if (match($0, /#new-token: [0-9]+/))              tok  = substr($0, RSTART+12, RLENGTH-12)
    if (match($0, /throughput \(token\/s\): [0-9.]+/)) rate = substr($0, RSTART+22, RLENGTH-22)
    if (tok + 0 >= 512) { pf[stack]++; pfsum[stack] += rate; if (rate+0 > 20000) pfout[stack]++ }
    next
}
/Decode batch/ && stack != "" {
    dc[stack]++
    if ($0 ~ /cuda graph: True/) graph[stack]++
    if (match($0, /accept len: [0-9.]+/)) { al = substr($0, RSTART+12, RLENGTH-12); alsum[stack] += al; aln[stack]++ }
    next
}
END {
    printf "PHASE EVIDENCE (records correlated with the active stack, not with the rate)\n\n"
    printf "  PREFILL records (>=512 new tokens)\n"
    for (s in pf)
        printf "    in %s: %6d records, mean %8.1f tok/s (%d implausible >20k excluded from judgement)\n",
               s, pf[s], pfsum[s]/pf[s], pfout[s]+0
    printf "\n  DECODE records\n"
    for (s in dc)
        printf "    in %s: %6d records, cuda graph True on %d (%.1f%%), mean accept len %.2f\n",
               s, dc[s], graph[s]+0, 100*(graph[s]+0)/dc[s], (aln[s]?alsum[s]/aln[s]:0)
    printf "\n  THE CLAIM THIS TESTS: prefill belongs to PP, decode to TP.\n"
    printf "  A healthy build shows prefill records concentrated in PP and\n"
    printf "  decode records concentrated in TP with graphs live.\n"
}'
