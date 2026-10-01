#!/bin/bash
cd /root/efeu35q3
pkill -f "^/bin/bash pp2/pp2_when_idle.sh"
chmod +x ab18k/window_runner.sh
bash -n ab18k/window_runner.sh || exit 1
setsid nohup ab18k/window_runner.sh < /dev/null > /dev/null 2>&1 &
sleep 1
pgrep -af "^/bin/bash (ab18k/window_runner.sh|./apply_when_quiet.sh|/root/efeu35q3/energy/probe_when_idle.sh|pp2/pp2_when_idle.sh)"
tail -2 logs/apply_when_quiet.log
