#!/usr/bin/env python
"""efeu-TP14: system-prompt prefix reuse -- TTFT cold vs warm vs after restart.

A coding agent resends a long system prompt (~17k tokens) on every turn. This
measures what the server makes of that, with a FIXED system prompt (byte-
identical across runs and across server restarts, so an L3 store can match it):

  cold      new system prompt never seen (unique nonce IN the system prompt)
  turn2     same conversation, second turn (prefix = turn-1 prompt + answer)
  session2  NEW conversation with the same system prompt, different user msg
            (prefix = the system prompt only)

Each request streams with max_tokens=4; TTFT = time to the first content
chunk. cached_tokens is read from the final usage block when the server
reports it. Results are appended as JSON lines with a phase label, so a run
before a restart and a run after it can be compared.

    python prefix_reuse_bench.py --port 31671 --phase before_restart --sys-tokens 17000
"""

import argparse
import json
import random
import time
import urllib.request

MAX_TOKENS = 4

WORDS = ("def class return import self value result config request handler "
         "error check index buffer stream token layer weight cache tensor "
         "shape batch page block state flush write read open close parse "
         "format string number list dict tuple file path module package").split()


def fixed_system_prompt(approx_tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    lines = ["You are a coding agent working in a local repository. The tools you may call "
             "and their exact contracts are documented below. Follow them precisely."]
    n = 0
    i = 0
    while n < approx_tokens * 0.62:
        name = f"tool_{i:04d}_{rng.choice(WORDS)}_{rng.choice(WORDS)}"
        args = ", ".join(f"{rng.choice(WORDS)}_{k}: {rng.choice(['str', 'int', 'bool', 'list[str]'])}"
                         for k in range(rng.randint(2, 5)))
        desc = " ".join(rng.choice(WORDS) for _ in range(rng.randint(12, 30)))
        lines.append(f"## {name}({args})\n{desc}.")
        n += len(lines[-1].split())
        i += 1
    return "\n".join(lines)


def stream_ttft(base, model, messages, timeout):
    body = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": MAX_TOKENS,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    usage = {}
    text = ""
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices", []):
                delta = c.get("delta", {}).get("content") or ""
                if delta and ttft is None:
                    ttft = time.perf_counter() - t0
                text += delta
    total = time.perf_counter() - t0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", usage.get("cached_tokens"))
    return {"ttft_s": ttft if ttft is not None else total, "total_s": total,
            "prompt_tokens": usage.get("prompt_tokens"), "cached_tokens": cached, "text": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=31671)
    ap.add_argument("--model", default="qwen38-35b-a3b")
    ap.add_argument("--sys-tokens", type=int, default=17000)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--phase", default="run")
    ap.add_argument("--steps", default="session1,turn2,session2")
    ap.add_argument("--out", default="prefix_reuse.jsonl")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--max-tokens", type=int, default=4)
    a = ap.parse_args()
    global MAX_TOKENS
    MAX_TOKENS = a.max_tokens
    base = f"http://127.0.0.1:{a.port}"
    S = fixed_system_prompt(a.sys_tokens, a.seed)
    q1 = "List the tools whose name contains the word cache. Answer briefly."
    q2 = "Now name one tool that takes a bool argument."
    q3 = "Which tool would you call to flush a buffer? One name."
    rows = []
    msgs1 = [{"role": "system", "content": S}, {"role": "user", "content": q1}]
    for step in a.steps.split(","):
        if step == "cold":
            m = [{"role": "system", "content": S + f"\nsession nonce {random.random()}"},
                 {"role": "user", "content": q1}]
        elif step == "session1":
            m = msgs1
        elif step == "turn2":
            prev = rows[-1]["text"] if rows else "ok"
            m = msgs1 + [{"role": "assistant", "content": prev}, {"role": "user", "content": q2}]
        elif step == "session2":
            m = [{"role": "system", "content": S}, {"role": "user", "content": q3}]
        else:
            raise SystemExit(f"unknown step {step}")
        r = stream_ttft(base, a.model, m, a.timeout)
        r.update(step=step, phase=a.phase, sys_tokens_target=a.sys_tokens, t=time.time())
        rows.append(r)
        print(f"[{a.phase}] {step:9s} prompt {r['prompt_tokens']} cached {r['cached_tokens']} "
              f"TTFT {r['ttft_s']:.2f} s  total {r['total_s']:.2f} s")
        with open(a.out, "a") as fh:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
