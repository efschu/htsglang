#!/bin/bash
cd /root/efeu35q3
pkill -f "^/bin/bash ./apply_when_quiet.sh"
sleep 1
bash -n apply_when_quiet.sh.new || exit 1
chmod +x apply_when_quiet.sh.new && mv apply_when_quiet.sh.new apply_when_quiet.sh
grep -E "HICACHE_RATIO=|MAMBA_HOST_RATIO=" 50-q38.conf
PROOF=0 setsid nohup ./apply_when_quiet.sh < /dev/null > /dev/null 2>&1 &
sleep 1
pgrep -af "^/bin/bash ./apply_when_quiet.sh|^/bin/bash ab18k/window_runner.sh"
curl -s -m5 localhost:31651/ondemand/status
