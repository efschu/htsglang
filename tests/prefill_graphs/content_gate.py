"""Content-identity oracle for the prefill CUDA graph backend.

Drives a running sglang server with a fixed, greedy prompt set and records
the exact completion text plus token-level logprobs. Two recordings can then
be compared byte-for-byte.

The gate is only valid if the A-vs-A floor holds first: two recordings taken
from the SAME server with the SAME settings must be identical. Long prefills
are deliberately avoided -- GDN prefill is not reproducible above ~109 tokens
on this stack, which would make the oracle report noise as divergence.

Usage:
    python content_gate.py record --port 30040 --out run_a.json
    python content_gate.py compare run_a.json run_b.json
"""

import argparse
import json
import sys
import urllib.request

# Short, varied-length prompts. Each stays well under the ~109-token GDN
# reproducibility ceiling, while spanning several captured token buckets so
# that replay padding is actually exercised rather than skipped.
PROMPTS = [
    "The capital of France is",
    "Explain in one sentence why the sky appears blue.",
    "List three prime numbers greater than fifty, separated by commas.",
    "def fibonacci(n):\n    # return the nth Fibonacci number\n",
    "Translate to German: The quick brown fox jumps over the lazy dog.",
    "A train leaves at 09:00 travelling 60 km/h. After two hours it has gone",
    (
        "Summarise the following in one short sentence. A cache stores recently "
        "used values so that later reads avoid recomputing them. It trades memory "
        "for time and is only useful when the same values are read more than once."
    ),
    (
        "Answer with a single word. Which of these is a mammal: salmon, eagle, "
        "dolphin, cobra, beetle, sparrow, trout, gecko, mantis, urchin?"
    ),
]


def _post(port: int, path: str, payload: dict, timeout: float = 180.0) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def record(port: int, out_path: str, max_tokens: int) -> None:
    records = []
    for idx, prompt in enumerate(PROMPTS):
        body = _post(
            port,
            "/v1/completions",
            {
                "model": "default",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "logprobs": 1,
                "stream": False,
            },
        )
        choice = body["choices"][0]
        lp = choice.get("logprobs") or {}
        records.append(
            {
                "index": idx,
                "prompt": prompt,
                "text": choice["text"],
                "finish_reason": choice.get("finish_reason"),
                "tokens": lp.get("tokens"),
                # Full precision: repr keeps every bit of the float, so a
                # 1-ULP drift is visible instead of being rounded away.
                "token_logprobs": [
                    None if v is None else repr(float(v))
                    for v in (lp.get("token_logprobs") or [])
                ],
                "prompt_tokens": body.get("usage", {}).get("prompt_tokens"),
            }
        )
    with open(out_path, "w") as fh:
        json.dump(records, fh, indent=1)
    print(f"recorded {len(records)} completions -> {out_path}")


def compare(path_a: str, path_b: str) -> int:
    with open(path_a) as fh:
        a = json.load(fh)
    with open(path_b) as fh:
        b = json.load(fh)
    if len(a) != len(b):
        print(f"FAIL: record count {len(a)} != {len(b)}")
        return 1

    text_bad, lp_bad = [], []
    for ra, rb in zip(a, b):
        if ra["text"] != rb["text"]:
            # Report the first differing character position -- that offset is
            # the diagnostic (an early flip means the prefill itself differs,
            # a late one means drift accumulated during decode).
            pos = next(
                (
                    i
                    for i, (ca, cb) in enumerate(zip(ra["text"], rb["text"]))
                    if ca != cb
                ),
                min(len(ra["text"]), len(rb["text"])),
            )
            text_bad.append((ra["index"], pos, ra["text"][:80], rb["text"][:80]))
        elif ra["token_logprobs"] != rb["token_logprobs"]:
            lp_bad.append(ra["index"])

    for idx, pos, ta, tb in text_bad:
        print(f"TEXT DIVERGENCE prompt#{idx} at char {pos}")
        print(f"   A: {ta!r}")
        print(f"   B: {tb!r}")
    for idx in lp_bad:
        print(f"LOGPROB-ONLY DIVERGENCE prompt#{idx} (text identical)")

    if text_bad or lp_bad:
        print(
            f"RESULT: FAIL ({len(text_bad)} text, {len(lp_bad)} logprob-only "
            f"of {len(a)})"
        )
        return 1
    print(f"RESULT: PASS -- {len(a)}/{len(a)} completions byte-identical")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--port", type=int, required=True)
    rec.add_argument("--out", required=True)
    rec.add_argument("--max-tokens", type=int, default=48)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("a")
    cmp_.add_argument("b")
    args = ap.parse_args()
    if args.cmd == "record":
        record(args.port, args.out, args.max_tokens)
        return 0
    return compare(args.a, args.b)


if __name__ == "__main__":
    sys.exit(main())
