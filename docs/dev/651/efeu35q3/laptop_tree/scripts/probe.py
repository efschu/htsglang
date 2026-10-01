#!/usr/bin/env python
"""Content-judged coherence probe for the #651 GGUF bring-up.

HTTP 200 proves nothing here. A misfiled expert shard or an unloaded router
gate produces fluent, grammatical, WRONG text -- that is the whole failure mode
this checkpoint is exposed to (#647/#318). So every probe has a determined
answer that is checked, and the whole set is run twice at temperature 0 to
confirm greedy determinism.
"""

import json
import sys
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 30040
BASE = f"http://127.0.0.1:{PORT}"

# (prompt, list of accepted substrings -- case-insensitive, ANY may match)
PROBES = [
    ("What is the capital of France? Answer with one word.", ["paris"]),
    ("What is 14 * 3? Reply with just the number.", ["42"]),
    ("What is 31 * 7? Reply with just the number.", ["217"]),
    ("Which planet is known as the Red Planet? One word.", ["mars"]),
    ("Complete the sequence with one number: 2, 4, 8, 16, ", ["32"]),
    (
        "In one short sentence: why does ice float on water?",
        ["less dense", "lower density", "denser", "density"],
    ),
]


def generate(prompt: str) -> str:
    body = json.dumps(
        {
            "model": "Qwen3.6-35B-A3B-Q4",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 96,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"]


def run_round(label: str):
    outs, passed = [], 0
    for prompt, accepted in PROBES:
        try:
            text = generate(prompt)
        except Exception as exc:  # noqa: BLE001
            text = f"<ERROR {exc}>"
        outs.append(text)
        ok = any(a.lower() in text.lower() for a in accepted)
        passed += ok
        flat = " ".join(text.split())[:110]
        print(f"  [{'PASS' if ok else 'FAIL'}] {prompt[:44]:44s} -> {flat}")
    print(f"  {label}: {passed}/{len(PROBES)} content-correct")
    return outs, passed


print(f"=== round 1 (port {PORT}) ===")
o1, p1 = run_round("round 1")
print("=== round 2 (greedy determinism) ===")
o2, p2 = run_round("round 2")

deterministic = o1 == o2
print()
print(f"content round1 = {p1}/{len(PROBES)}   round2 = {p2}/{len(PROBES)}")
print(f"greedy deterministic (round1 == round2): {deterministic}")

verdict = p1 == len(PROBES) and p2 == len(PROBES) and deterministic
print("VERDICT:", "COHERENT" if verdict else "NOT COHERENT")
sys.exit(0 if verdict else 1)
