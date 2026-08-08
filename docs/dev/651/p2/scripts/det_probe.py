"""Greedy determinism probe: same prompt N times at temperature 0.

Uses the native /generate endpoint (no chat template) so the test isolates
model forward determinism from template handling. Captures the first-token
top-4 logprobs so a flip can be attributed to logit noise vs argmax ties.
"""
import json, sys, urllib.request, collections

PORT = int(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
PROMPT = "Question: What is 14 * 3? Answer with just the number.\nAnswer:"

outs = []
for i in range(N):
    body = json.dumps({
        "text": PROMPT,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 16},
        "return_logprob": True, "top_logprobs_num": 4,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    text = d["text"]
    mi = d.get("meta_info", {})
    tl = mi.get("output_top_logprobs")
    first = tl[0] if tl else None
    outs.append((text, json.dumps(first)))
    print(f"run {i}: text={text!r}")
    print(f"       first-token top4={first}")

texts = collections.Counter(t for t, _ in outs)
firsts = collections.Counter(f for _, f in outs)
print(f"\ndistinct texts: {len(texts)} of {N}")
for t, c in texts.most_common():
    print(f"  x{c}: {t!r}")
print(f"distinct first-token logprob sets: {len(firsts)} of {N}")
print("VERDICT:", "DETERMINISTIC" if len(texts) == 1 else "NON-DETERMINISTIC")
