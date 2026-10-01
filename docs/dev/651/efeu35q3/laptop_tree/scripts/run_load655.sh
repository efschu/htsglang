#!/bin/bash
# #655: one instrumented load. Waits for the service to park, snapshots host
# memory, forces a cold load with a real prompt, then records the pool that
# resulted together with the available_gpu_mem the allocator saw.
set -u
LABEL="${1:-run}"
OUT=/root/651-p2/logs/kv655.tsv
SNAP=/root/651-p2/logs/kv655_snap.tsv
S() { curl -s -m 10 localhost:31651/ondemand/status; }

# 1. park. The service parks itself 60 s after the last request; nothing else
#    frees the weights, so waiting is the only way to a genuinely cold load.
for i in $(seq 1 60); do
  st=$(S | sed -n 's/.*"state": "\([a-z]*\)".*/\1/p')
  [ "$st" = "parked" ] && break
  sleep 5
done
echo "[$LABEL] state before load: ${st:-unknown}"

# 2. pre-launch snapshot, from outside (the in-script postdrop one comes later)
/root/651-p2/scripts/memsnap655.sh "pre-$LABEL" >> "$SNAP"

# 3. force the load with a prompt that also serves the correctness gate
t0=$(date +%s)
resp=$(curl -s -m 900 localhost:31651/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen36-35b-a3b","messages":[{"role":"user","content":"List the first 10 Fibonacci numbers, comma separated."}],"max_tokens":80,"temperature":0}')
t1=$(date +%s)
echo "[$LABEL] load+infer seconds: $((t1-t0))"
echo "[$LABEL] OUT1: $(echo "$resp" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["choices"][0]["message"]["content"][:300])' 2>/dev/null || echo "PARSE-FAIL: $(echo "$resp" | head -c 300)")"

# second prompt, prose rather than arithmetic -- a numerically dead path can
# still emit plausible digits, so the gate needs both shapes
resp2=$(curl -s -m 300 localhost:31651/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen36-35b-a3b","messages":[{"role":"user","content":"Describe the sea in exactly 3 sentences."}],"max_tokens":120,"temperature":0}')
echo "[$LABEL] OUT2: $(echo "$resp2" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["choices"][0]["message"]["content"][:400])' 2>/dev/null || echo "PARSE-FAIL: $(echo "$resp2" | head -c 300)")"

# 4. what the pool came out as, and what the allocator thought it had
stj=$(S)
kvt=$(echo "$stj" | sed -n 's/.*"kv_tokens": \([0-9]*\).*/\1/p')
blog=$(echo "$stj" | sed -n 's/.*"backend_log": "\([^"]*\)".*/\1/p')
agm=$(grep -h "KV Cache is allocated" "$blog" 2>/dev/null | tail -1 | grep -oE "available_gpu_mem=[0-9.]+" | tail -1)
mtn=$(grep -h "KV Cache is allocated" "$blog" 2>/dev/null | tail -1 | grep -oE "max_total_num_tokens=[0-9]+" | tail -1)
attempts=$(grep -hc "KV Cache is allocated" "$blog" 2>/dev/null || echo 0)
kvd=$(grep -hoE "kv_cache_dtype='[^']*'" "$blog" 2>/dev/null | tail -1)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
  "$(date -u +%H:%M:%S)" "$LABEL" "${kvt:-NA}" "${mtn:-NA}" "${agm:-NA}" "${attempts:-0}" "$blog" >> "$OUT"
echo "[$LABEL] RESULT kv_tokens=${kvt:-NA} $mtn $agm attempts=$attempts $kvd"
echo "[$LABEL] log=$blog"
