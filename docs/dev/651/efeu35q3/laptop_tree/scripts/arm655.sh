#!/bin/bash
# #655: run one configuration arm. Switches the drop-in, restarts the
# supervisor so the new environment reaches boot_ondemand.sh, then takes N
# instrumented loads.
set -u
ARM="$1"; KVD="$2"; SLOTS="$3"; N="${4:-2}"
D=/etc/systemd/system/htsglang-ondemand.service.d/30-kv655.conf
{
  echo "[Service]"
  echo "Environment=SNAP655_FILE=/root/651-p2/logs/kv655_snap.tsv"
  echo "Environment=KVDTYPE=$KVD"
  echo "Environment=MAMBASLOTS=$SLOTS"
} > "$D"
systemctl daemon-reload
timeout 180 systemctl restart htsglang-ondemand || echo "[$ARM] restart returned $?"
sleep 5
echo "[$ARM] ARM-START kvdtype=$KVD mamba_slots=${SLOTS:-auto} loads=$N"
for i in $(seq 1 "$N"); do
  /root/651-p2/scripts/run_load655.sh "$ARM-$i"
  sleep 70
done
echo "[$ARM] ARM-DONE"
