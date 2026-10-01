#!/usr/bin/env python
"""efeu-TP14: llama.cpp CPU reference vs the sglang iGPU outputs, same GGUF file.

Queries a llama-server (CPU, -ngl 0, --jinja = the GGUF's own chat template)
with exactly the probe_q38 prompts at temperature 0 / thinking off and compares
the greedy text with what sglang produced (probe json). Reports exact matches
and, for mismatches, the common-prefix length in characters and in llama.cpp
tokens (via /tokenize), so a late near-tie flip is distinguishable from an
early divergence (= a real defect).

    python llama_ref_compare.py <llama_port> <sglang_probe.json> [--json out]
"""

import json
import sys
import urllib.request

sys.path.insert(0, "/root/efeu35q3")
from probe_q38 import PROBES  # noqa: E402  (module runs its probe on import otherwise)

PORT = int(sys.argv[1])
PROBE_JSON = sys.argv[2]
OUT = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
BASE = f"http://127.0.0.1:{PORT}"


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def chat(prompt, n):
    d = post("/v1/chat/completions", {"messages": [{"role": "user", "content": prompt}],
                                      "temperature": 0.0, "max_tokens": n,
                                      "chat_template_kwargs": {"enable_thinking": False}})
    return d["choices"][0]["message"]["content"] or ""


def ntok(text):
    return len(post("/tokenize", {"content": text})["tokens"])


sg = json.load(open(PROBE_JSON))
pairs = [(p, o) for (p, _), o in zip(PROBES, sg["outputs"])]
pairs.append(("Explain in two sentences what a hash map is.", sg["det8_text"]))
rows, exact = [], 0
for prompt, sg_text in pairs:
    n = 96 if prompt != pairs[-1][0] else 64
    ref = chat(prompt, n)
    common = 0
    for a, b in zip(ref, sg_text):
        if a != b:
            break
        common += 1
    same = ref.strip() == sg_text.strip()
    exact += same
    row = {"prompt": prompt, "exact": same, "common_chars": common,
           "common_tokens": ntok(ref[:common]) if common else 0,
           "ref_tokens": ntok(ref), "llama": ref, "sglang": sg_text}
    rows.append(row)
    print(f"[{'SAME' if same else 'DIFF'}] {prompt[:46]:46s} common {row['common_tokens']}/{row['ref_tokens']} tok")
    if not same:
        print(f"      llama : {' '.join(ref.split())[:150]}")
        print(f"      sglang: {' '.join(sg_text.split())[:150]}")
print(f"exact {exact}/{len(pairs)}")
if OUT:
    json.dump({"exact": exact, "n": len(pairs), "rows": rows}, open(OUT, "w"), indent=1)
