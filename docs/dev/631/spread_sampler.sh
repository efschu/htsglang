#!/usr/bin/env bash
# Item 16 time series: NVML free per card at 100 ms, plus the SPREAD.
# free = the NVML FREE column (never total-used: the carve-out is invisible
# to that subtraction, vram-korridor-regel).
OUT="${1:?out csv}"; DUR="${2:-600}"
echo "t_unix_ms,free0_mib,free1_mib,free2_mib,spread_mib,min_mib" > "$OUT"
END=$(( $(date +%s) + DUR ))
while [ "$(date +%s)" -lt "$END" ]; do
  read -r f0 f1 f2 <<<"$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr '\n' ' ')"
  [ -z "$f2" ] && { sleep 0.1; continue; }
  mn=$f0; mx=$f0
  for v in $f1 $f2; do [ "$v" -lt "$mn" ] && mn=$v; [ "$v" -gt "$mx" ] && mx=$v; done
  echo "$(($(date +%s%N)/1000000)),$f0,$f1,$f2,$((mx-mn)),$mn" >> "$OUT"
  sleep 0.1
done
