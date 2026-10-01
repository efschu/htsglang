#!/bin/bash
pkill -f "^/root/lh/venv/bin/python (probe_q38|bench_prefill|bench_decode|prefix_reuse_bench)" 
P=$(pgrep -f "^python -m sglang.launch_server"); [ -n "$P" ] && kill -TERM $P
for i in $(seq 1 30); do pgrep -f "^python -m sglang.launch_server" >/dev/null || break; sleep 2; done
pgrep -f "^python -m sglang" || echo "no sglang"
