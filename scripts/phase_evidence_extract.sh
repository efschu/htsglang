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
grep -F "PP0]" "$LOG" | awk -v assert_on="${PHASE_PURITY_ASSERT:-1}" '
/\(active stack now pp\)/ { stack = "PP" }
/\(active stack now tp\)/ { stack = "TP" }
/Prefill batch/ && stack != "" {
    tok = ""; rate = ""
    if (match($0, /#new-token: [0-9]+/))              tok  = substr($0, RSTART+12, RLENGTH-12)
    if (match($0, /throughput \(token\/s\): [0-9.]+/)) rate = substr($0, RSTART+22, RLENGTH-22)
    # PURITY counts every record and every TOKEN, with NO size floor: the
    # rule is "not a single token prefilled in TP", so the >=512 floor that
    # keeps the throughput MEAN honest must not also decide the verdict.
    pfany[stack]++; pftok[stack] += tok + 0
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

    # -- STRICT PHASE PURITY VERDICT ------------------------------------
    # "Concentrated in" is what the report above shows, and it is not the
    # rule. The user rule is absolute: ZERO decode records in PP, ZERO
    # prefilled tokens in TP. So this block does not summarise, it JUDGES,
    # and the script exits non-zero when the rule is broken -- a green
    # criterion that cannot fail proves nothing.
    printf "\n  STRICT PHASE PURITY VERDICT\n"
    bad = 0
    if (dc["PP"] + 0 > 0) {
        printf "    VIOLATION: %d decode record(s) executed in the PP layout\n", dc["PP"]
        bad = 1
    } else printf "    ok: no decode record executed in the PP layout\n"
    if (pftok["TP"] + 0 > 0) {
        printf "    VIOLATION: %d token(s) prefilled in the TP layout across %d record(s)\n",
               pftok["TP"], pfany["TP"]
        bad = 1
    } else printf "    ok: not a single token prefilled in the TP layout\n"
    # Both layouts must actually have been VISITED, or "no violation" is
    # just an instance that never flipped -- the starvation defect would
    # pass a purity check trivially by never leaving PP.
    if (pfany["PP"] + 0 == 0) {
        printf "    VIOLATION: no prefill ran in PP at all (did the instance ever enter the PP phase?)\n"
        bad = 1
    }
    if (dc["TP"] + 0 == 0) {
        printf "    VIOLATION: no decode ran in TP at all (starvation: the instance never reached the decode layout)\n"
        bad = 1
    }
    printf "    => %s\n", (bad ? "PURITY BROKEN" : "PURITY HELD, both layouts used")
    exit (bad && assert_on + 0 ? 1 : 0)
}'
