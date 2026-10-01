#!/usr/bin/env python
"""efeu-TP14: cold-prefill determinism for a long (multi-chunk) prompt.

The same conversation N times, each with a fresh cache_salt (so every run is a
full prefill over identical tokens, no cache), greedy, 64 tokens, logprobs.
All N must be identical. Short single-chunk prompts were (probe_q38 8x); this
checks prompts that cross chunk boundaries (GDN state carried chunk->chunk in a
mamba slot).

    python cold_repeat.py <port> <sys_tokens> <n> <out.json>
"""
import json
import sys
import urllib.request
import uuid

sys.path.insert(0, "/root/efeu35q3")
from prefix_reuse_bench import fixed_system_prompt  # noqa: E402

port, sys_tokens, n, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
S = fixed_system_prompt(sys_tokens, 4242)
msgs = [{"role": "system", "content": S},
        {"role": "user", "content": "Name three tools from the list that take a str argument. Answer as a short list."}]
runs = []
for i in range(n):
    body = {"model": "qwen38-35b-a3b", "messages": msgs, "temperature": 0.0, "max_tokens": 64,
            "logprobs": True, "top_logprobs": 2, "cache_salt": f"cold-{uuid.uuid4().hex}",
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read())
    c = d["choices"][0]
    lp = [(t["token"], round(t["logprob"], 4), [(x["token"], round(x["logprob"], 4)) for x in t["top_logprobs"]])
          for t in (c.get("logprobs") or {}).get("content", [])]
    runs.append({"text": c["message"]["content"], "lp": lp})
    print(f"run {i + 1}: {' '.join((c['message']['content'] or '').split())[:160]}")
base = runs[0]
for i, r in enumerate(runs[1:], 2):
    k = next((j for j, (a, b) in enumerate(zip(base["lp"], r["lp"])) if a[0] != b[0]), None)
    if k is None:
        print(f"run {i} == run 1")
    else:
        print(f"run {i} diverges at token {k}: run1 {base['lp'][k]} vs run{i} {r['lp'][k]}")
json.dump(runs, open(out, "w"), indent=1, ensure_ascii=False)
