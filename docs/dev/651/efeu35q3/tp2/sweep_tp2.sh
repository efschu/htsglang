#!/bin/bash
# efeu-TP14 2026-10-01: TP2-form sweep (iGPU + CPU computing the SAME layer
# concurrently, split by experts via kt LLAMAFILE) + llama.cpp CPU reference.
# Every config: coherence probe, prefill 2k/6k (A/A'), decode (A/A').
# Balanced power profile throughout; the on-demand service is stopped meanwhile.
#   CONFIGS="solo_eager:0 solo_graphs:0 kt32:224 ..."  (name:KTEXPERTS, 0 = no kt)
set -u
cd /root/efeu35q3
R=results/tp2
mkdir -p $R
LOG=logs/sweep_tp2.log
exec >> $LOG 2>&1
PY=/root/lh/venv/bin/python
CONFIGS=${CONFIGS:-"llamacpp_cpu:- solo_eager:0 kt32:224 kt64:192 kt128:128 kt160:96"}
powerprofilesctl set balanced
echo "=== sweep start $(date -Is) profile=$(powerprofilesctl get) resets=$(dmesg | grep -c 'GPU reset')"
wait_up() {  # $1 log
  for i in $(seq 1 120); do
    grep -q "fired up" "$1" && return 0
    grep -qE "Traceback|Killed|guard v2 failed" "$1" && return 1
    pgrep -f "^python -m sglang.launch_server" >/dev/null || { [ $i -gt 6 ] && return 1; }
    sleep 5
  done
  return 1
}
stop_server() {
  P=$(pgrep -f "^python -m sglang.launch_server"); [ -n "$P" ] && kill -TERM $P
  for i in $(seq 1 40); do pgrep -f "^python -m sglang.launch_server" >/dev/null || break; sleep 2; done
  pkill -f "^/root/651-p2/llama.cpp/build/bin/llama-server" 2>/dev/null; sleep 2
}
for C in $CONFIGS; do
  NAME=${C%%:*}; KT=${C##*:}
  echo "--- $NAME (KTEXPERTS=$KT) $(date +%T)"
  stop_server
  if [ "$NAME" = "llamacpp_cpu" ]; then
    /root/651-p2/llama.cpp/build/bin/llama-server -m models/Qwen3.8-35B-A3B-Q3_K_M.gguf -ngl 0 --jinja -c 8192 -t 8 \
      -b 512 -ub 512 --port 31690 --host 127.0.0.1 > logs/llama_tp2.log 2>&1 &
    for i in $(seq 1 120); do curl -sf -m 3 localhost:31690/health >/dev/null && break; sleep 3; done
    $PY tp2/llama_cpu_bench.py 31690 $R/$NAME.json
    stop_server
    continue
  fi
  TS=$(date +%H%M%S); BL=logs/boot_tp2_${NAME}_$TS.log
  ENVS="PORT=31671 CTX=16384 MEMFRAC=0.92 CHUNKED_PREFILL=512 WEDGE_CP_MEASURED=512"
  case $NAME in
    solo_eager) ENVS="$ENVS EAGER=1" ;;
    solo_graphs) ;;
    *) ENVS="$ENVS KTMETHOD=LLAMAFILE KTEXPERTS=$KT KTCPUINFER=8 KTPOOLS=1 KTDEFER=0" ;;
  esac
  env $ENVS setsid ./boot_q38.sh > $BL 2>&1 < /dev/null &
  if ! wait_up $BL; then echo "BOOT FAILED $NAME"; tail -5 $BL; continue; fi
  grep -E "Load weight end|max_total" $BL | cut -c1-200
  $PY probe_q38.py 31671 qwen38-35b-a3b --json $R/${NAME}_probe.json | tail -2
  $PY bench_prefill.py --port 31671 --model qwen38-35b-a3b --label $NAME --lengths 2048,8192 --seconds 12 --warmup-s 8 \
    --out $R/${NAME}_prefill.json | grep -E "\[A|floor"
  $PY bench_decode.py --port 31671 --model qwen38-35b-a3b --max-tokens 256 --seconds 20 --warmup-s 8 --label $NAME \
    --out $R/${NAME}_decode.json | grep -E "\[A"
  grep "Prefill rank batch, #new-token: 512" $BL | tail -3 | cut -c1-140
  free -m | sed -n 2p
  echo "resets=$(dmesg | grep -c 'GPU reset')"
done
stop_server
powerprofilesctl set power-saver
echo "=== SWEEP DONE $(date -Is)"
