#!/usr/bin/env python3
"""DUAL-TP3PP3 stage 2 step 3: greedy bit-equality of the first K tokens, decode-join vs extend.

Run ONCE per boot against the front (default http://127.0.0.1:30030), sequentially (bs1,
so batch composition cannot move the numerics), temperature 0:
  boot A: dual1m image with the join code, SGLANG_WEG2_DUAL_DECODE_JOIN unset (extend path)
  boot B: same image + EXTRA_CENV="SGLANG_WEG2_DUAL_DECODE_JOIN=1" (join path)
  join_bitcheck.py run --out A.json      (on boot A)
  join_bitcheck.py run --out B.json      (on boot B)
  join_bitcheck.py compare A.json B.json [--k 64]
Each conversation has 3 turns over a fixed ~3-6k-token context (repo source text), so
every turn is a P prefill + a D hand-off; turns 2-3 also exercise follow-ups with a
cached prefix. Pure stdlib (urllib).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

QUESTIONS = [
    "Summarize in one sentence what this code does.",
    "Name the single most important function above and why.",
    "Which line would you change first to make it faster? Quote it.",
]


def _contexts(root: str, n: int):
    files = []
    for dp, _, fns in os.walk(root):
        for fn in sorted(fns):
            if fn.endswith(".py"):
                p = os.path.join(dp, fn)
                if 12_000 < os.path.getsize(p) < 40_000:
                    files.append(p)
    files.sort()
    out = []
    for p in files[:: max(1, len(files) // n)][:n]:
        with open(p, errors="replace") as f:
            out.append((os.path.relpath(p, root), f.read()[:20_000]))
    return out


def _chat(url, model, messages, k, timeout):
    body = json.dumps({"model": model, "messages": messages, "temperature": 0, "top_p": 1,
                       "max_tokens": k, "stream": False,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"].get("content") or "", d.get("usage", {}), time.time() - t0


def run(a):
    model = a.model
    if not model:
        with urllib.request.urlopen(a.url + "/v1/models", timeout=30) as r:
            model = json.loads(r.read())["data"][0]["id"]
    res = {"model": model, "k": a.k, "convs": []}
    for name, ctx in _contexts(a.root, a.n):
        msgs = [{"role": "system", "content": "Answer briefly."},
                {"role": "user", "content": f"File {name}:\n```python\n{ctx}\n```\n{QUESTIONS[0]}"}]
        turns = []
        for qi in range(len(QUESTIONS)):
            if qi:
                msgs.append({"role": "user", "content": QUESTIONS[qi]})
            text, usage, dt = _chat(a.url, model, msgs, a.k, a.timeout)
            turns.append({"text": text, "usage": usage, "s": round(dt, 2)})
            msgs.append({"role": "assistant", "content": text})
            print(f"{name} turn {qi + 1}: {len(text)} chars {dt:.1f}s prompt={usage.get('prompt_tokens')} "
                  f"cached={usage.get('prompt_tokens_details', {}) or usage.get('cached_tokens')}", flush=True)
        res["convs"].append({"file": name, "turns": turns})
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


def compare(a):
    A, B = json.load(open(a.a)), json.load(open(a.b))
    same = diff = 0
    for ca, cb in zip(A["convs"], B["convs"]):
        for i, (ta, tb) in enumerate(zip(ca["turns"], cb["turns"])):
            if ta["text"] == tb["text"]:
                same += 1
            else:
                diff += 1
                j = next((x for x in range(min(len(ta["text"]), len(tb["text"])))
                          if ta["text"][x] != tb["text"][x]), min(len(ta["text"]), len(tb["text"])))
                print(f"DIFF {ca['file']} turn {i + 1} at char {j}: A={ta['text'][max(0, j - 30):j + 40]!r} "
                      f"B={tb['text'][max(0, j - 30):j + 40]!r}")
    print(f"VERDICT {'BIT-EQUAL' if diff == 0 and same else 'DIFFERENT'}: {same} equal, {diff} different turns")
    return 0 if diff == 0 and same else 1


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--url", default="http://127.0.0.1:30030")
    r.add_argument("--model", default="")
    r.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                                    "python", "sglang", "srt", "weg2"))
    r.add_argument("--n", type=int, default=4)
    r.add_argument("--k", type=int, default=64)
    r.add_argument("--timeout", type=float, default=300)
    r.add_argument("--out", required=True)
    c = sp.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    return run(a) if a.cmd == "run" else compare(a)


if __name__ == "__main__":
    sys.exit(main() or 0)
