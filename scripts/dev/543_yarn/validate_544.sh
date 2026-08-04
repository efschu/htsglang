#!/usr/bin/env bash
# #544 post-boot validation. Bounded everywhere; never tails the server log into
# an agent context -- only greps named markers.
set -uo pipefail
B=http://127.0.0.1:30030
LOG=${LOG:-/tmp/w544_serving.log}
HICACHE_DIR=${HICACHE_DIR:-/spinning/hicache}
PROBE_DIR=/spinning/gpu-battery-results/2026-08-04_541_thinking_ab

say() { printf '\n=== %s ===\n' "$1"; }

say "1 health / identity"
curl -s -m 10 "$B/health" >/dev/null && echo "health ok"
curl -s -m 10 "$B/get_server_info" | python3 -c '
import json,sys
d=json.load(sys.stdin)
for k in ("context_length","max_total_num_tokens","enable_hierarchical_cache",
          "hicache_storage_backend","hicache_size","hicache_write_policy",
          "speculative_algorithm","max_running_requests","reasoning_parser",
          "tool_call_parser","enable_kv_session_offload"):
    print(f"  {k} = {d.get(k,\"<absent>\")}")
print("  preserve_thinking-ish:", {k:v for k,v in d.items() if "preserve" in k or "thinking" in k})'

say "2 boot markers (grep only, bounded)"
head -c 900000 "$LOG" | grep -aE "HiCacheFile storage directory|storage backend|Hierarchical|KV Cache is allocated|Mamba Cache is allocated|max_total_num_tokens=" | head -10

say "3 short-context sanity"
curl -s -m 90 "$B/v1/chat/completions" -H 'content-type: application/json' -d '{
 "model":"Qwen3.6-27B","max_tokens":60,"temperature":0,
 "messages":[{"role":"user","content":"Name the capital of Portugal, then compute 17*23. Answer in one short line."}]}' \
 | python3 -c 'import json,sys; d=json.load(sys.stdin); m=d["choices"][0]["message"]; print("  answer:", (m.get("content") or "").strip()[:200])'

say "4 hicache disk tier active"
before=$(find "$HICACHE_DIR" -type f 2>/dev/null | wc -l)
for i in 1 2; do
  curl -s -m 120 "$B/v1/chat/completions" -H 'content-type: application/json' -d "{
   \"model\":\"Qwen3.6-27B\",\"max_tokens\":16,\"temperature\":0,
   \"messages\":[{\"role\":\"user\",\"content\":\"hicache-probe-544 nonce-7731 $(printf 'lorem ipsum dolor sit amet %.0s' $(seq 1 400)) Reply OK.\"}]}" >/dev/null
done
sleep 5
after=$(find "$HICACHE_DIR" -type f 2>/dev/null | wc -l)
echo "  files in $HICACHE_DIR: before=$before after=$after  bytes=$(du -sh "$HICACHE_DIR" 2>/dev/null | cut -f1)"
curl -s -m 10 "$B/metrics" | grep -E "^sglang:(cache_hit_rate|cached_tokens_total)\{" | head -4

say "5 preserve_thinking two-turn prefix reuse"
if [ -f "$PROBE_DIR/probe_preserve_thinking.py" ]; then
  timeout 420 python3 "$PROBE_DIR/probe_preserve_thinking.py" 2>&1 | tail -20
else
  echo "  probe missing at $PROBE_DIR"
fi

say "6 #540 thinking budget live"
curl -s -m 180 "$B/v1/chat/completions" -H 'content-type: application/json' -d '{
 "model":"Qwen3.6-27B","max_tokens":900,"temperature":0,
 "thinking_budget":128,
 "messages":[{"role":"user","content":"Prove that the square root of two is irrational."}]}' \
 | python3 -c '
import json,sys
d=json.load(sys.stdin)
u=d.get("usage",{}) or {}
det=u.get("completion_tokens_details") or {}
rt=det.get("reasoning_tokens")
print("  reasoning_tokens =",rt," completion_tokens =",u.get("completion_tokens"))
if rt is not None:
    over=rt-128
    print(f"  overshoot = {over} (must be <= draft_token_num 4)")'

say "7 VRAM corridor (100 ms sampling, 10 s)"
python3 - <<'EOF'
import subprocess,time
mins={}
end=time.time()+10
while time.time()<end:
    out=subprocess.run(["nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"],
                       capture_output=True,text=True).stdout
    for line in out.strip().splitlines():
        i,f=[x.strip() for x in line.split(",")]
        mins[i]=min(mins.get(i,10**9),int(f))
    time.sleep(0.1)
for i in sorted(mins):
    print(f"  gpu{i} min free {mins[i]} MiB  {'OK' if mins[i]>=400 else 'CORRIDOR BREACH'}")
EOF

say "8 host RAM (cgroup truth, lxcfs is distorted here)"
awk '/^anon |^file |^shmem /{printf "  %s %.1f GB\n",$1,$2/1e9}' /sys/fs/cgroup/memory.stat
printf "  memory.current %.1f GB of 98 GB host\n" "$(awk '{print $1/1e9}' /sys/fs/cgroup/memory.current)"

say "9 translator tenant untouched"
ps -p 30439 -o pid=,etime= 2>/dev/null || echo "  WARNING: translator PID 30439 gone"
curl -s -m 10 http://127.0.0.1:30800/health >/dev/null 2>&1 && echo "  tenant 30800 healthy" || echo "  tenant 30800 no /health (check manually)"
