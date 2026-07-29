#!/usr/bin/env bash
# S4 -- r7c Boot C: GGUF-Q3 target with a quantised DFLASH-Q8_0 drafter solo
# on a 3080.
#
# Independent of A and B; third in the queue because it is the boot most likely
# to fail in a way that costs a retry, and a retry is cheaper once the accept
# questions are settled.
#
# The drafter card is resolved at RUNTIME by the recipe (DRAFT_GPU defaults to
# the first small card from the CUDA/NVML join). Nothing here hardcodes an
# index; the 5090 arm is one variable away (DRAFT_GPU=$CUDA_BIG) and is worth
# it only after the 3080 arm shows the drafter working -- placement is the
# second question, not the first.
#
# ABORT semantics from the recipe, restated because they decide FAIL vs result:
#   * drafter load raises on a tensor name -> abort, report the name (FAIL),
#   * OOM on the hosting card -> abort; raising RESERVE_HOST is the NEXT run's
#     business, not a retry inside this window,
#   * INCOHERENT OUTPUT IS A RESULT, NOT AN ABORT. The check does not judge
#     coherence; it verifies the drafter loaded and the probe produced curves.

set -uo pipefail
cd "$(dirname "$0")"
source ./battery_common.sh
source ./_r7c_boot.sh

run_r7c_boot c boot_c_dflash_solo_q8.sh
RC=$?
emit_reference_column boot_c
exit "$RC"
