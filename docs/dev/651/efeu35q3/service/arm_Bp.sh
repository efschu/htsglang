#!/bin/bash
cd /root/efeu35q3
pkill -f "^/bin/bash ./apply_when_quiet.sh"
/root/lh/venv/bin/python patch_poolstats_log.py || exit 1
cp 50-q38.conf.Bp 50-q38.conf
grep -E "^Environment=(HICACHE_RATIO|MAMBASLOTS|SGLANG_HICACHE_MAMBA_HOST_RATIO)" 50-q38.conf
PROOF=0 setsid nohup ./apply_when_quiet.sh < /dev/null > /dev/null 2>&1 &
sleep 1; pgrep -af "^/bin/bash ./apply_when_quiet.sh"
curl -s -m5 localhost:31651/ondemand/status
