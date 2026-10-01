#!/bin/bash
# efeu-TP14: switch the on-demand service between the two checkpoints.
#   switch_model.sh q38   Qwen3.8-35B-A3B-Distill Q3_K_M (fixed kernels, graphs on)
#   switch_model.sh q4    Qwen3.6-35B-A3B Q4KM-noQ6K, the #655 state (kt CPU experts)
#   switch_model.sh       print the active one
# Only the 50-q38.conf drop-in is toggled; nothing else is touched.
set -eu
D=/etc/systemd/system/htsglang-ondemand.service.d
active() { [ -f "$D/50-q38.conf" ] && echo q38 || echo q4; }
case "${1:-}" in
  "") active; exit 0 ;;
  q38) [ -f "$D/50-q38.conf.disabled" ] && mv "$D/50-q38.conf.disabled" "$D/50-q38.conf" ;;
  q4)  [ -f "$D/50-q38.conf" ] && mv "$D/50-q38.conf" "$D/50-q38.conf.disabled" ;;
  *) echo "usage: $0 [q38|q4]"; exit 2 ;;
esac
# The coding agent's context window must match the served --context-length
# (q38: 32768, q4/kt: 16384), so its model registry follows the switch.
OMP=/home/efeu/.omp/agent/models.yml
if [ -f "$OMP.$1" ]; then
  install -o efeu -g efeu -m 0644 "$OMP.$1" "$OMP"
  echo "omp registry: $OMP.$1"
fi
systemctl daemon-reload
systemctl restart htsglang-ondemand
sleep 2
echo "active: $(active)"
curl -s -m 5 localhost:31651/ondemand/status; echo
