#!/bin/bash
# efeu-TP14: ONE small prompt-logprob request (~200 tokens, 1 new token) in a quiet
# window (service up, inflight 0, idle >= 900 s) -- exercises the lm_head
# over-cap dequant (M = prompt tokens > _MMQ_MAX_TOKENS 8) on the GPU.
# Evidence, since the tiled branch has no log line of its own:
#   * GTT in use before/after (mem_info_gtt_used): an untiled lm_head dequant
#     would leave a ~0.95 GiB block in torch's caching allocator, the tiled one
#     at most one 64 MiB tile;
#   * no OOM / traceback in the backend log;
#   * every input-token logprob finite.
set -u
cd /root/efeu35q3
OUT=results/logprob_tiling_probe.txt
exec >> $OUT 2>&1
st() { curl -s -m5 localhost:31651/ondemand/status; }
echo "=== logprob tiling probe armed $(date -Is)"
until st | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(0 if d["state"]=="up" and d["inflight"]==0 and d["idle_seconds"]>=900 else 1)' 2>/dev/null; do sleep 30; done
B=$(st | python3 -c 'import json,sys; print(json.load(sys.stdin)["backend_log"])')
G0=$(cat /sys/class/drm/card1/device/mem_info_gtt_used)
L0=$(wc -l < $B)
echo "$(date +%T) quiet; GTT used before $((G0/1048576)) MiB"
python3 - <<'PY'
import json, math, urllib.request
words = ("the system processes each request through several distinct stages before the final "
         "response is produced and returned to the caller").split()
text = " ".join(words[i % len(words)] for i in range(150))
body = {"text": text, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        "return_logprob": True, "logprob_start_len": 0}
req = urllib.request.Request("http://127.0.0.1:31651/generate", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=600) as r:
    d = json.loads(r.read())
lp = [x[0] for x in d["meta_info"]["input_token_logprobs"] if x[0] is not None]
print(f"prompt tokens {d['meta_info']['prompt_tokens']}, input logprobs {len(lp)}, "
      f"all finite: {all(math.isfinite(v) for v in lp)}, min {min(lp):.3f}, max {max(lp):.3f}")
PY
G1=$(cat /sys/class/drm/card1/device/mem_info_gtt_used)
echo "GTT used after $((G1/1048576)) MiB (delta $(( (G1-G0)/1048576 )) MiB; untiled would leave ~970 MiB)"
tail -n +$((L0+1)) $B | grep -iE "out of memory|Traceback|Error" | head -3 || true
echo "=== probe done $(date -Is)"
