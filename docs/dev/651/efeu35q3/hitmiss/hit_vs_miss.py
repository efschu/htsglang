#!/usr/bin/env python
"""efeu-TP14: is garbage on cache hits a SERVER state-restore bug or the model?

Phase 1 (hits): a multi-turn chat like omp, turns sent back to back, no other
request in between: turn k+1 = turn k's prompt + its greedy answer + a new user
message, so the radix cache can match up to the end of turn k INCLUDING its
decoded tokens (the mamba/GDN anchor sits exactly there in no_buffer mode).
Phase 2 (misses): each turn's exact message list again with a fresh cache_salt
(radix and HiCache keys salted), i.e. identical tokens, computed from scratch.
With a correct restore path, hit == miss token for token (greedy). Garbage or
divergence only on the hit side isolates the server's restore path.

    python hit_vs_miss.py <port> <sys_tokens> <out.json> [turns] [seed] [think]
"""
import json
import sys
import time
import urllib.request
import uuid

sys.path.insert(0, "/root/efeu35q3")
from prefix_reuse_bench import fixed_system_prompt  # noqa: E402

port, sys_tokens, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
turns = int(sys.argv[4]) if len(sys.argv) > 4 else 3
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 9001
# "think": answers WITH thinking (as omp gets them); the next turn sends only
# the visible answer back, so the chat template drops the reasoning and the new
# prompt diverges at the start of the old assistant turn (rig finding 01.10.).
THINK = len(sys.argv) > 6 and sys.argv[6] == "think"
BASE = f"http://127.0.0.1:{port}/v1/chat/completions"


def chat(msgs, n=None, salt=None):
    n = n or (512 if THINK else 96)
    body = {"model": "qwen38-35b-a3b", "messages": msgs, "temperature": 0.0, "max_tokens": n,
            "chat_template_kwargs": {"enable_thinking": THINK}}
    if salt:
        body["cache_salt"] = salt
    t0 = time.time()
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"] or "", (d.get("usage") or {}).get("prompt_tokens"), time.time() - t0


S = fixed_system_prompt(sys_tokens, seed)
Qs = ["Name three tools from the list that take a str argument. Answer as a short list.",
      "Kannst du auch Deutsch? Antworte in zwei Sätzen.",
      "Now explain in one sentence what the first tool you named probably does.",
      "Write a one-line Python call of that tool with made-up arguments."]
hist = [{"role": "system", "content": S}]
hits = []
for i in range(turns):
    hist.append({"role": "user", "content": Qs[i % len(Qs)]})
    t, p, s = chat(hist)
    hits.append({"turn": i + 1, "msgs": json.loads(json.dumps(hist)), "hit": t, "prompt_tokens": p, "hit_s": round(s, 1)})
    print(f"hit  turn {i + 1}: prompt {p} {s:.0f}s: {' '.join(t.split())[:160]}")
    visible = t.split("</think>")[-1].strip() if THINK else t
    hist.append({"role": "assistant", "content": visible})
for h in hits:
    t, p, s = chat(h["msgs"], salt=f"miss-{uuid.uuid4().hex}")
    h.update(miss=t, miss_s=round(s, 1), identical=(t == h["hit"]))
    print(f"miss turn {h['turn']}: {'IDENTICAL' if h['identical'] else 'DIFFERENT'} ({s:.0f}s)")
    if not h["identical"]:
        print("   hit : " + " ".join(h["hit"].split())[:220])
        print("   miss: " + " ".join(t.split())[:220])
json.dump(hits, open(out, "w"), indent=1, ensure_ascii=False)
