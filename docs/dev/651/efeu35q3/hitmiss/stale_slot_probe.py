#!/usr/bin/env python
"""efeu-TP14: does a request's output depend on what ran BEFORE it?

X = fixed long prompt (fresh seed, never cached before), greedy 64 tokens.
  1  X cold, salt a
  2  Y (different long prompt) cold, salt b      -- dirties mamba slots
  3  X cold, salt c                              -- must equal 1
  4  X unsalted (first time unsalted -> cold)    -- must equal 1
  5  X unsalted again (radix hit)                -- must equal 1
  6  short unrelated request, then X unsalted    -- hit after slot churn, must equal 1
The backend log gives #cached-token per step.

    python stale_slot_probe.py <port> <sys_tokens> <seed> <out.json>
"""
import json
import sys
import urllib.request
import uuid

sys.path.insert(0, "/root/efeu35q3")
from prefix_reuse_bench import fixed_system_prompt  # noqa: E402

port, sys_tokens, seed, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
Q = "Name three tools from the list that take a str argument. Answer as a short list."


def ask(S, salt=None, n=64, q=Q):
    body = {"model": "qwen38-35b-a3b", "messages": [{"role": "system", "content": S}, {"role": "user", "content": q}],
            "temperature": 0.0, "max_tokens": n, "chat_template_kwargs": {"enable_thinking": False}}
    if salt:
        body["cache_salt"] = salt
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"] or ""


X = fixed_system_prompt(sys_tokens, seed)
Y = fixed_system_prompt(sys_tokens, seed + 1)
steps = []
steps.append(("1 X cold salt a", ask(X, f"a-{uuid.uuid4().hex}")))
steps.append(("2 Y cold salt b", ask(Y, f"b-{uuid.uuid4().hex}")))
steps.append(("3 X cold salt c", ask(X, f"c-{uuid.uuid4().hex}")))
steps.append(("4 X unsalted (cold)", ask(X)))
steps.append(("5 X unsalted (hit)", ask(X)))
ask("You are terse.", None, 8, "Say ok.")
steps.append(("6 X unsalted after churn", ask(X)))
ref = steps[0][1]
for name, t in steps:
    print(f"{'==' if t == ref else '!='} {name}: {' '.join(t.split())[:150]}")
json.dump(steps, open(out, "w"), indent=1, ensure_ascii=False)
