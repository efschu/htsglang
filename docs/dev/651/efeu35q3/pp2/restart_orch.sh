#!/bin/bash
cd /root/efeu35q3
pkill -f "^/bin/bash pp2/pp2_when_idle.sh"
sleep 1
mv pp2/pp2_when_idle.sh.new pp2/pp2_when_idle.sh; chmod +x pp2/pp2_when_idle.sh; bash -n pp2/pp2_when_idle.sh || exit 1
setsid nohup pp2/pp2_when_idle.sh < /dev/null > /dev/null 2>&1 &
sleep 1; pgrep -af "^/bin/bash pp2/pp2_when_idle.sh|^/bin/bash /root/efeu35q3/energy/probe_when_idle.sh"
