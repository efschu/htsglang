#!/bin/bash
pkill -f "^/bin/bash ./apply_when_quiet.sh"
pkill -f "^/bin/bash ab18k/window_runner.sh"
sleep 1
pgrep -af "^/bin/bash (./apply_when_quiet.sh|ab18k/window_runner.sh)" || echo "runners paused"
