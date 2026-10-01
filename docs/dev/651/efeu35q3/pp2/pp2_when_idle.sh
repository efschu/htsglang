#!/bin/bash
# efeu-TP14: run the PP2 build + measurement ONLY while the user's service is
# truly idle (parked, inflight 0, idle >= 900 s). Never touches the service.
#  * build (llama.cpp HIP): SIGSTOP'ed the moment a request arrives, SIGCONT
#    once idle again (make is incremental anyway).
#  * each -ngl measurement: killed the moment a request arrives (the front door
#    then loads the model; the llama.cpp process must be gone before the ~17 GB
#    of weights land), and retried in the next idle window.
set -u
cd /root/efeu35q3
LOG=logs/pp2_when_idle.log
exec >> $LOG 2>&1
idle_ok() {
  curl -s -m5 localhost:31651/ondemand/status | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d["state"]=="parked" and d["inflight"]==0 and d["idle_seconds"]>=900 else 1)' 2>/dev/null
}
busy() { ! curl -s -m5 localhost:31651/ondemand/status | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d["state"]=="parked" and d["inflight"]==0 else 1)' 2>/dev/null; }
wait_idle() { until idle_ok; do sleep 30; done; }
guard() {  # $1 pid; $2 action on busy: stop|kill
  while kill -0 $1 2>/dev/null; do
    if busy; then
      if [ "$2" = stop ]; then
        pkill -STOP -g $(ps -o pgid= $1 | tr -d " "); echo "$(date +%T) request -> build paused"
        wait_idle; pkill -CONT -g $(ps -o pgid= $1 | tr -d " "); echo "$(date +%T) idle -> build resumed"
      else
        pkill -KILL -g $(ps -o pgid= $1 | tr -d " "); echo "$(date +%T) request -> measurement killed"
        # restore the idle profile BEFORE the service's boot script reads it as
        # "previous" (it switches only after guard v2, ~15 s into the boot)
        powerprofilesctl set power-saver
        return 1
      fi
    fi
    sleep 2
  done
  return 0
}
echo "=== pp2_when_idle start $(date -Is)"
# Step 0 (energy A/B, user order): decode tok/s of the service AS CONFIGURED
# (sleep-on-idle on), measured through the front door in an idle window. The
# request wakes the model; that is the only load this script ever causes.
if [ ! -s results/decode_svc128k_sleep.json ]; then
  wait_idle; echo "$(date +%T) energy B: decode through the service"
  /root/lh/venv/bin/python bench_decode.py --port 31651 --model qwen38-35b-a3b --max-tokens 256 \
    --seconds 15 --warmup-s 5 --label svc128k_sleep --out results/decode_svc128k_sleep.json 2>&1 | grep -E "\[A|floor"
fi
if [ ! -x llama-hip-build/bin/llama-bench ]; then
  wait_idle; echo "$(date +%T) build start"
  setsid bash pp2/build_llama_hip.sh > logs/llama_hip_outer.log 2>&1 &
  BP=$!; guard $BP stop; wait $BP; cat logs/llama_hip_outer.log
fi
for NGL in ${NGLS:-41 33 25 0}; do
  [ -s results/pp2/bench_ngl$NGL.json ] && [ -s results/pp2/coh_ngl$NGL.json ] && continue
  while true; do
    wait_idle; echo "$(date +%T) ngl=$NGL start"
    NGLS=$NGL setsid bash pp2/pp2_llama.sh > logs/pp2_ngl$NGL.log 2>&1 &
    MP=$!
    if guard $MP kill; then wait $MP; cat logs/pp2_ngl$NGL.log; break; fi
    rm -f results/pp2/bench_ngl$NGL.json results/pp2/coh_ngl$NGL.json
  done
done
powerprofilesctl set power-saver
echo "=== PP2_WHEN_IDLE DONE $(date -Is)"
