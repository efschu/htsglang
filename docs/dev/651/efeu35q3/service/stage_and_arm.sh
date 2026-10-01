#!/bin/bash
set -e
cd /root/efeu35q3
F=sglang_src/python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py
[ -f $F.orig-efeu ] || cp $F $F.orig-efeu
cp staged_hybrid_cache_controller.py $F.new && mv $F.new $F
grep -c "MAMBA-SNAPSHOT-FENCE" $F
grep -E "MAMBASLOTS|MAXRUN|MAXTOTAL|MAMBA_HOST_RATIO" 50-q38.conf
setsid nohup ./apply_when_quiet.sh < /dev/null > /dev/null 2>&1 &
sleep 1
pgrep -af "^/bin/bash ./apply_when_quiet.sh"
curl -s -m5 localhost:31651/ondemand/status
