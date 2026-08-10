#!/usr/bin/env bash
# #631/#656 successor20 judged-evidence extract. BOUNDED greps only --
# the serving log is hundreds of MiB and must never be read whole.
#
# Every axis the acceptance criterion names, and each one answers WHERE the
# work ran, not merely that it ran: a flip count without a purity split is
# compatible with an instance that flips and then does everything in one
# layout anyway.
set -uo pipefail
L="${1:-/spinning/serving-30030.boot.log}"

echo "=== #631/#656 EVIDENCE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "log: $L  ($(du -h "$L" | cut -f1))"
echo
echo "-- DEFECT R: the leak must be silent --"
echo "resident-set corruption reports : $(grep -c 'resident set is corrupted' "$L" 2>/dev/null || echo 0)"
echo "repairs performed               : $(grep -c 'PHASE-FLIP-CARRY REPAIR' "$L" 2>/dev/null || echo 0)"
echo "policy refusals after repair    : $(grep -c 'still corrupted after repair' "$L" 2>/dev/null || echo 0)"
echo "self-merge refusals (s18 guard) : $(grep -c 'SELF-MERGE REFUSED' "$L" 2>/dev/null || echo 0)"
echo
echo "-- FLIPS: both layouts must be visited --"
echo "pp_to_tp cutovers : $(grep -c 'PHASE-FLIP cutover.*to_tp\|cutover pp_to_tp' "$L" 2>/dev/null || echo 0)"
echo "tp_to_pp cutovers : $(grep -c 'cutover tp_to_pp' "$L" 2>/dev/null || echo 0)"
grep -oE 'PHASE-FLIP [a-z_]+ (committed|complete)' "$L" 2>/dev/null | sort | uniq -c | head
echo
echo "-- STRICT PURITY: prefill only in PP, decode only in TP --"
echo "Prefill batches seen in PP ranks : $(grep -cE '\[PP[012]\] Prefill batch' "$L" 2>/dev/null || echo 0)"
echo "Decode  batches seen in PP ranks : $(grep -cE '\[PP[012]\] Decode batch' "$L" 2>/dev/null || echo 0)"
echo "  (the PP/TP split is reported by the phase counters below; the rank"
echo "   tag alone does not name the layout)"
grep -oE 'PHASE-PURITY[^|]*' "$L" 2>/dev/null | tail -4
echo
echo "-- SPEC + GRAPHS --"
grep -oE 'accept_len[= ][0-9.]+' "$L" 2>/dev/null | tail -3
grep -oE 'cuda graph[^,]*' "$L" 2>/dev/null | tail -3
echo
echo "-- ERRORS --"
echo "500s / tracebacks : $(grep -cE 'Traceback|500 Internal' "$L" 2>/dev/null || echo 0)"
grep -E 'Traceback' "$L" 2>/dev/null | tail -3
