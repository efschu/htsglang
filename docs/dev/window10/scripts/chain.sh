#!/usr/bin/env bash
# One-variable-at-a-time ladder from the record pin to today's production.
# EVERY arm is barlink (user law). Each step changes exactly one thing.
set -u
D=/spinning/gpu-battery-results/2026-08-05_window10
S=$D/scripts
PIN=/spinning/wt-w10-pin
INT=/spinning/wt-w10-int

run() {  # run <arm> <wt> <flagset> <envfile>
  echo "########## $(date -u +%H:%M:%SZ) START $1 ##########"
  bash "$S/run_arm.sh" "$1" "$2" "$3" "$4" > "$D/driver_$1.out" 2>&1
  echo "########## $(date -u +%H:%M:%SZ) END $1 rc=$? ##########"
  sleep 8
}

# A: the A-vs-A anchor. Pin, record config, barlink.
run arm1_pin_record_bl        "$PIN" record "$S/env_bl.sh"
# B: TREE delta -- same config, same transport, 523 commits later.
run arm2_int_record_bl        "$INT" record "$S/env_bl.sh"
# C: CONFIG delta -- today's production flagset on the same tree, with the
#    #600 EVERY=32 mitigation production actually carries.
run arm0_int_today_every32    "$INT" today  "$S/env_bl_every32.sh"
# D: the #517 fix, default-on, EVERY back to 1.
run arm0_int_today_517on      "$INT" today  "$S/env_bl.sh"
# E: the #517 control -- watchdog off, so the hot path reads in line again.
run arm0_int_today_wdoff      "$INT" today  "$S/env_bl_wdoff.sh"
echo "CHAIN COMPLETE $(date -u +%H:%M:%SZ)"
