#!/usr/bin/env bash
# S2 -- r7c Boot A: Qwen3.6-27B-FP8, the one-axis accept falsifier.
#
# A is first because it is the only boot whose outcome changes what the others
# mean. If the reference reproduces, the GGUF accept ceiling is a target-
# quantisation property; if it does not, B and C are measuring against nothing.
#
# REPORT-GATE: the executor stops after this step and reports, PASS or not,
# before B starts. That is the queue's order, not a safety measure.
#
# Both outcomes are results. accept ~2.6-3.3 with position 0 near 65 % means
# the reference exists on this rig; accept ~1.5 with position 0 at 24-45 %
# means it was a property of the old measuring path. The check does not judge
# which happened -- it verifies that a per-position curve and a reference
# column exist at all, because a boot that produced neither answered nothing.

set -uo pipefail
cd "$(dirname "$0")"
source ./battery_common.sh
source ./_r7c_boot.sh

run_r7c_boot a boot_a_fp8_reference.sh
RC=$?
emit_reference_column boot_a
exit "$RC"
