#!/bin/bash
# efeu-TP14: stop any measurement, bring the user's service back (q38, 128k).
set -u
pkill -f "^/bin/bash tp2/sweep_tp2.sh" ; pkill -f "^bash tp2/sweep_tp2.sh"; pkill -f "^/bin/bash pp2/pp2_llama.sh"
bash /root/efeu35q3/stopall.sh
pkill -f "^/root/651-p2/llama.cpp/build/bin/llama-server|^/root/efeu35q3/llama-hip-build/bin/llama-" 2>/dev/null
echo "--- sweep stopped for service restore $(date -Is)" >> /root/efeu35q3/logs/sweep_tp2.log
install -m 0644 /root/efeu35q3/50-q38.conf /etc/systemd/system/htsglang-ondemand.service.d/50-q38.conf
O=/home/efeu/.omp/agent/models.yml
install -o efeu -g efeu -m 0644 /root/efeu35q3/omp-models.yml $O.q38
install -o efeu -g efeu -m 0644 $O.q38 $O
systemctl daemon-reload
systemctl enable --now htsglang-ondemand
sleep 3
systemctl is-active htsglang-ondemand
systemctl show htsglang-ondemand -p Environment | tr ' ' '\n' | grep -E "^CTX="
free -g; swapon --show
