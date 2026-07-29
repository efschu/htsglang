#!/usr/bin/env bash
# S3 -- r7c Boot B: Huihui-AWQ-MTP, INT4 body with a BF16 head.
#
# B is the other half of A's question and shares its apparatus: A moves the
# target quantisation with the head following it, B holds the target coarse and
# lifts only the head. Read together they separate "the head's precision is the
# lever" from "the target's is". Run alone, neither does.
#
# KNOWN RISK, stated before the run: AWQ x uneven TP x MTP has never been
# booted on this branch, and the round-7b GPTQ arm died on a unit-count
# mismatch. If the load rejects the shape, that is one consumed boot and the
# answer is "not on this vehicle". Do NOT tune the ratio inside the window --
# that is a different experiment and the executor is not authorised to run it.

set -uo pipefail
cd "$(dirname "$0")"
source ./battery_common.sh
source ./_r7c_boot.sh

run_r7c_boot b boot_b_dense_head.sh
RC=$?
emit_reference_column boot_b
exit "$RC"
