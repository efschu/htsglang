#!/bin/bash
# #655: give the fp8 startup warmup a bounded window to finish its Triton
# compile, record the verdict either way, then put the machine on the lossless
# configuration and prove that one serves.
set -u
LOG=/root/651-p2/logs/backend_021701.log
DEADLINE=$((SECONDS+480))
while [ $SECONDS -lt $DEADLINE ]; do
  if grep -aq "Prefill batch" "$LOG" 2>/dev/null; then
    echo "FP8-VERDICT: reached first prefill after $(grep -ao 'Prefill batch' "$LOG" | wc -l) batches"
    sleep 90
    journalctl -u kv655probe2 -o cat --no-pager | tail -8
    break
  fi
  sleep 20
done
grep -aq "Prefill batch" "$LOG" 2>/dev/null || \
  echo "FP8-VERDICT: NO prefill after 480s more; startup warmup still inside triton make_amdgcn"

echo "SWITCHING to lossless config (KVDTYPE=auto, MAMBASLOTS=4)"
systemctl stop kv655probe2 2>/dev/null
cp /root/651-p2/scripts/30-kv655-final.conf \
   /etc/systemd/system/htsglang-ondemand.service.d/30-kv655.conf
systemctl daemon-reload
timeout 240 systemctl restart htsglang-ondemand || echo "restart rc=$?"
sleep 10
for i in 1 2; do
  /root/651-p2/scripts/run_load655.sh m4-$i
  sleep 70
done
echo FINISH655-DONE
