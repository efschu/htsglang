#!/bin/bash
# #655: 2 s memory time series. The pre-launch snapshot cannot explain the
# lottery on its own, because ~150 s of weight loading happens between it and
# the moment the KV pool is sized -- and reading a 21.6 GiB checkpoint is itself
# the largest memory event in the whole load. This samples through it so the
# state AT the sizing instant can be read off by timestamp.
out=/root/651-p2/logs/kv655_series.tsv
d=/sys/class/drm/card1/device
while true; do
  read -r free avail cached shmem < <(awk '/^MemFree:/{f=$2}/^MemAvailable:/{a=$2}/^Cached:/{c=$2}/^Shmem:/{s=$2}END{print f,a,c,s}' /proc/meminfo)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -u +%H:%M:%S)" \
    "$free" "$avail" "$cached" "$shmem" \
    "$(cat $d/mem_info_gtt_used 2>/dev/null || echo 0)" \
    "$(cat $d/mem_info_vram_used 2>/dev/null || echo 0)" >> "$out"
  sleep 2
done
