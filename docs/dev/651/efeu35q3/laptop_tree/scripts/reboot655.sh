#!/bin/bash
# Stop the ladder and reboot to clear accumulated amdgpu MES state.
#
# Justification, so this is not read as a casual reboot: resets went 5 -> 11 in
# one session, and a configuration that served a clean 1k probe at 06:04
# wedged on the identical probe at 06:20. That is the documented "MES wedges
# accumulate" behaviour, and a reboot is the documented escalation. The
# htsglang-ondemand unit is enabled and has been verified to come back parked
# on its own after a reboot.
#
# sync first: the GPU is wedged, the filesystem is not, and there is no reason
# to risk it. A normal `systemctl reboot` is tried; the caller escalates to
# `reboot --force --force` only if the box is still up afterwards.
set -u
SELF=$$

for p in $(ls /proc | grep -E '^[0-9]+$'); do
  [ "$p" = "$SELF" ] && continue
  [ -r "/proc/$p/cmdline" ] || continue
  CMD=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)
  case "$CMD" in
    *ladder655*|*needle655*|*finalize655*|*accept_omp655*)
      kill -TERM "$p" 2>/dev/null && echo "stopped $p"
      ;;
  esac
done

echo "resets before reboot: $(dmesg -T | grep -c 'GPU reset(')"
dmesg -T | grep -cE 'GPU reset\(' > /root/651-p2/results/resets_before_reboot_$(date +%H%M%S).txt
sync
sleep 2
echo "rebooting now"
systemctl reboot &
exit 0
