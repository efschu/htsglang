#!/usr/bin/env bash
# S5 -- r7c Boot D: lane re-seed A/B on the round-7b configuration.
#
# Last and cheapest: this is exactly the round-7b setup, so it is the one boot
# known to come up. Two arms in one boot, driven through the job body rather
# than the driver's --rollback-arms switch.
#
# It writes reseed.json, not accept.json -- the check is a different one.
# output_ids differing between the arms is NOT a failure. That is the
# measurement: what aligning the lane's chain with the serving group's costs
# and buys. The expectation is "little or nothing, and the price is the point".

set -uo pipefail
cd "$(dirname "$0")"
source ./battery_common.sh
source ./_r7c_boot.sh

run_r7c_boot d boot_d_lane_reseed.sh
exit $?
