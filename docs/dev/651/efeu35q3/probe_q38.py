#!/usr/bin/env python
"""efeu-TP14 coherence gate for Qwen3.8-35B-A3B (probe.py-style, #651).

8 probes with determined answers, temperature 0, thinking OFF, run twice
(content 8/8 both rounds, round1 == round2), then ONE probe 8x with logprobs:
all 8 completions and their token logprob vectors must be identical.
HTTP 200 is not evidence; only checked content and byte-identical greedy runs.

    python probe_q38.py <port> [model-name] [--json out.json]
"""

import json
import sys
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 31671
MODEL = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "qwen38-35b-a3b"
OUT = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
BASE = f"http://127.0.0.1:{PORT}"

PROBES = [
    ("What is the capital of France? Answer with one word.", ["paris"]),
    ("What is 14 * 3? Reply with just the number.", ["42"]),
    ("What is 31 * 7? Reply with just the number.", ["217"]),
    ("Which planet is known as the Red Planet? One word.", ["mars"]),
    ("Complete the sequence with one number: 2, 4, 8, 16, ", ["32"]),
    ("In one short sentence: why does ice float on water?", ["less dense", "lower density", "density"]),
    ("Write a Python expression that reverses the string s. Reply with code only.", ["[::-1]", "reversed("]),
    ("What is 17 * 23? Reply with just the number.", ["391"]),
]


def chat(prompt, max_tokens=96, logprobs=False):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 1
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    c = d["choices"][0]
    lp = None
    if logprobs and c.get("logprobs"):
        lp = [(t["token"], round(t["logprob"], 6)) for t in c["logprobs"]["content"]]
    return c["message"]["content"] or "", lp


def run_round(label):
    outs, passed = [], 0
    for p, acc in PROBES:
        try:
            t, _ = chat(p)
        except Exception as e:  # noqa: BLE001
            t = f"<ERROR {e}>"
        ok = any(a.lower() in t.lower() for a in acc)
        passed += ok
        outs.append(t)
        print(f"  [{'PASS' if ok else 'FAIL'}] {p[:48]:48s} -> {' '.join(t.split())[:100]}")
    print(f"  {label}: {passed}/{len(PROBES)}")
    return outs, passed


def main():
    res = {}
    print(f"=== round 1 (port {PORT}, model {MODEL}) ===")
    o1, p1 = run_round("round 1")
    print("=== round 2 ===")
    o2, p2 = run_round("round 2")
    res.update(round1=p1, round2=p2, rounds_identical=o1 == o2, outputs=o1)
    print("=== 8x identical greedy with logprobs ===")
    runs = [chat("Explain in two sentences what a hash map is.", 64, True) for _ in range(8)]
    texts = {r[0] for r in runs}
    lps = {json.dumps(r[1]) for r in runs}
    res.update(det8_texts_distinct=len(texts), det8_logprob_sets_distinct=len(lps), det8_text=runs[0][0])
    print(f"  distinct texts {len(texts)}/8, distinct logprob vectors {len(lps)}/8")
    print(f"  text: {' '.join(runs[0][0].split())[:160]}")
    ok = p1 == len(PROBES) and p2 == len(PROBES) and o1 == o2 and len(texts) == 1 and len(lps) == 1
    res["verdict"] = "COHERENT" if ok else "NOT COHERENT"
    print("VERDICT:", res["verdict"])
    if OUT:
        json.dump(res, open(OUT, "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
