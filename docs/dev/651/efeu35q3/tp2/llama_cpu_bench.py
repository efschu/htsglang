#!/usr/bin/env python
"""efeu-TP14: llama.cpp CPU-only throughput on the same GGUF (upper bound for
what a CPU pipeline stage can compute per layer). Uses llama-server's own
'timings' (prompt_per_second / predicted_per_second), unique prompts, n=2 each.

    python llama_cpu_bench.py <port> <out.json>
"""
import json
import random
import sys
import urllib.request

port, out = int(sys.argv[1]), sys.argv[2]
words = ("the system processes each request through several distinct stages before the final "
         "response is produced and returned to the caller").split()


def prompt(n):
    return f"session {random.random()} " + " ".join(random.choice(words) for _ in range(int(n * 0.75)))


def run(n_prompt, n_pred):
    body = {"prompt": prompt(n_prompt), "n_predict": n_pred, "temperature": 0, "cache_prompt": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())["timings"]


res = {}
run(64, 4)  # warmup
for n in (2048, 6144):
    t = [run(n, 1) for _ in range(2)]
    res[f"prefill_{n}"] = [{"n": x["prompt_n"], "tok_s": x["prompt_per_second"]} for x in t]
    print(f"llama.cpp CPU prefill ~{n}: " + ", ".join(f"{x['prompt_n']} tok {x['prompt_per_second']:.1f} tok/s" for x in t))
t = [run(32, 128) for _ in range(2)]
res["decode"] = [{"n": x["predicted_n"], "tok_s": x["predicted_per_second"]} for x in t]
print("llama.cpp CPU decode: " + ", ".join(f"{x['predicted_per_second']:.2f} tok/s" for x in t))
json.dump(res, open(out, "w"), indent=1)
