#!/bin/bash
cd /root/efeu35q3
pkill -f "^/bin/bash ab18k/window_runner.sh"
sleep 1
bash -n ab18k/window_runner.sh.new || exit 1
mv ab18k/window_runner.sh.new ab18k/window_runner.sh; chmod +x ab18k/window_runner.sh
setsid nohup ab18k/window_runner.sh < /dev/null > /dev/null 2>&1 &
sleep 1; pgrep -af "^/bin/bash (ab18k/window_runner.sh|serving_tree_patches/logprob_tiling_probe.sh)"
