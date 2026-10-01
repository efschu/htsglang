#!/bin/bash
# #655 verification driver. Detached: the whole chain is tens of minutes on
# this machine, and it is chained rather than run piecemeal so the model stays
# warm -- the service parks after 60 s idle and a cold load is ~191 s.
set -u
TS=$(date +%H%M%S)
LOG=/root/651-p2/results/drive655_${TS}.log
exec > >(tee -a "$LOG") 2>&1

mark() { echo "[$(date -Is)] MARK: $*"; }

mark "driver start"
mark "status: $(curl -s -m 10 localhost:31651/ondemand/status)"
mark "resets before: $(dmesg -T | grep -c 'GPU reset(')"

mark "phase A: needle probe at ~10k tokens"
python3 /root/651-p2/scripts/needle655.py 10000 \
  /root/651-p2/results/needle655_10k_${TS}.txt
mark "phase A rc=$? -> /root/651-p2/results/needle655_10k_${TS}.txt"

mark "status after A: $(curl -s -m 10 localhost:31651/ondemand/status)"
mark "resets after A: $(dmesg -T | grep -c 'GPU reset(')"

mark "phase B: coding agent end-to-end as user efeu"
/root/651-p2/scripts/accept_omp655.sh
mark "phase B rc=$?"

mark "status after B: $(curl -s -m 10 localhost:31651/ondemand/status)"
mark "resets after B: $(dmesg -T | grep -c 'GPU reset(')"
mark "driver done -> $LOG"
