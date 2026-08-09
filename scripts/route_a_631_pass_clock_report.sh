#!/usr/bin/env bash
# #631 defect Q: read the pass-clock evidence out of a serving log.
#
# Prints, per armed window: the abandon lines and the PASS-CLOCK line that
# reports every rank's slot-iteration count across that window. A non-zero
# SPREAD is the measurement that confirms the ranks left the armed window
# out of phase; a spread of 0 on repeated abandons under load KILLS the
# hypothesis, which is the outcome this script exists to make cheap.
LOG="${1:-/spinning/serving-30030.q-repro.log}"
echo "=== arms / abandons / commits ==="
grep -aE "PHASE-FLIP (phase flip armed|FLIP ABANDONED|DONE)" "$LOG" \
  | sed -E 's/(.{200}).*/\1/' | tail -40
echo
echo "=== PASS-CLOCK (defect Q instrument) ==="
grep -a "PASS-CLOCK" "$LOG" | sed -E 's/(.{240}).*/\1/' | tail -40
echo
echo "=== spreads only ==="
grep -ao "SPREAD [0-9]*" "$LOG" | sort | uniq -c
echo
echo "=== faults ==="
grep -acE "illegal memory access|ssm_state_indices|Traceback|SIGQUIT" "$LOG"
