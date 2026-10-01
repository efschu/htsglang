#!/usr/bin/env python
"""efeu-TP14: greedy continuation after a ~17k-token system prompt (the omp case).

Same fixed tool-list system prompt as prefix_reuse_bench.py, then two user turns
(German and English), temperature 0, 96 tokens, thinking off. Works against
sglang (/v1/chat/completions, model name) and llama-server (same API).

    python greedy_long.py <port> <label> <out.json> [model]
"""
import json
import sys
import time
import urllib.request

sys.path.insert(0, "/root/efeu35q3")
from prefix_reuse_bench import fixed_system_prompt  # noqa: E402

port, label, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
model = sys.argv[4] if len(sys.argv) > 4 else "qwen38-35b-a3b"
S = fixed_system_prompt(17000, 1001)
QS = [
    "Kannst du auch Deutsch? Antworte in zwei Sätzen und nenne dann ein Werkzeug aus der Liste, das einen bool-Parameter hat.",
    "Which tool would you call to flush a buffer? Give its exact name and one sentence why.",
]
res = {"label": label, "runs": []}
for q in QS:
    body = {"model": model, "messages": [{"role": "system", "content": S}, {"role": "user", "content": q}],
            "temperature": 0.0, "max_tokens": 96, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read())
    txt = d["choices"][0]["message"]["content"] or ""
    res["runs"].append({"q": q, "text": txt, "prompt_tokens": (d.get("usage") or {}).get("prompt_tokens"),
                        "seconds": round(time.time() - t0, 1)})
    print(f"[{label}] {d.get('usage', {}).get('prompt_tokens')} tok, {time.time() - t0:.0f}s: {' '.join(txt.split())[:200]}")
json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
