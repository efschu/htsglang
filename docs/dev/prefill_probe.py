"""Cache-busted prefill probe: a unique salt per run so the radix cache
cannot serve any prefix, which is what made the first comparison meaningless."""
import json, sys, time, urllib.request, uuid

PARA = ("The quick brown fox jumps over the lazy dog near the riverbank "
        "while the sun sets slowly behind distant mountains. ")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
res = []
for i in range(N):
    salt = uuid.uuid4().hex
    prompt = f"Session {salt}. Summarize the following text in one word.\n\n" + PARA * 260
    body = json.dumps({"model": "Qwen3.8-27B",
                       "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 1, "temperature": 0, "seed": 735000001}).encode()
    req = urllib.request.Request("http://127.0.0.1:30030/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=900).read())
    el = time.time() - t0
    u = d["usage"]
    cached = u.get("prompt_tokens_details") or {}
    res.append(u["prompt_tokens"] / el)
    print("run%d tokens=%d wall=%.2fs tok/s=%.1f cached=%s"
          % (i, u["prompt_tokens"], el, res[-1], cached.get("cached_tokens")))
res.sort()
med = res[len(res)//2] if len(res) % 2 else (res[len(res)//2-1]+res[len(res)//2])/2
print("SUMMARY median=%.1f min=%.1f max=%.1f spread=%.1f%%"
      % (med, res[0], res[-1], (res[-1]-res[0])/res[0]*100))
