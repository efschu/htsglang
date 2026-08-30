#!/usr/bin/env python3
"""#855 — teacher-forced logprob capture + KLD, for comparing two checkpoints
served by the SAME tree and flags.

WHY TEACHER-FORCED, and not "generate and compare texts": a generation
comparison confounds the distribution with the sampling path — one different
argmax early diverges the whole continuation and produces a number that says
nothing about how close the two models are. Feeding a FIXED token sequence and
reading the per-position distribution over that same sequence removes sampling
entirely: both arms are scored at identical prefixes, position by position.

WHAT IS ACTUALLY COMPUTED, stated honestly: the server returns the top-k
logprobs per position, not the full 150k-vocab distribution. So this is a KLD
over the TRUNCATED-and-RENORMALISED top-k support of the reference arm, which
is the standard practical estimator and is a LOWER bound on the true KLD — mass
the two arms disagree about outside the reference's top-k is invisible to it.
Reported alongside it, because they do not share that blind spot:
  * mean |delta logprob| on the ACTUAL next token (full-vocab quantity, no
    truncation anywhere — this one is exact),
  * top-1 agreement rate (the argmax the greedy decoder would have taken).

Usage:
  kld_capture_855.py capture --out arm.json [--url ...] [--topk 20]
  kld_capture_855.py compare --ref armA.json --new armB.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request

# Fixed corpus. Deliberately spread across the registers the GDN layers see in
# service: prose, code, structured data, math, multilingual, and long-range
# factual recall. Held literal in this file so both arms are scored on
# byte-identical input — a corpus read from disk could drift between runs.
CORPUS = [
    "The capital of France is Paris, a city known for its art, its museums and a long history of revolution and reform.",
    "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
    "In a relational database, a B-tree index keeps keys in sorted order across balanced nodes, so a range scan walks the leaf level sequentially instead of seeking once per row.",
    "{\"name\": \"widget\", \"count\": 42, \"tags\": [\"alpha\", \"beta\"], \"nested\": {\"enabled\": true, \"ratio\": 0.75}}",
    "If a train leaves the station at 14:20 travelling at 80 kilometres per hour, and a second train leaves the same station at 15:05 travelling at 120 kilometres per hour on the same track, the second train catches the first after one and a half hours.",
    "Die Bundesrepublik Deutschland ist ein foederaler Staat mit sechzehn Laendern, deren Regierungen eigene Zustaendigkeiten in Bildung und Kultur besitzen.",
    "The mitochondrion generates most of the cell's supply of adenosine triphosphate through oxidative phosphorylation, using a proton gradient across the inner membrane.",
    "Attention mechanisms compute a weighted sum of value vectors, where the weights come from a scaled dot product between queries and keys, normalised by a softmax over the sequence.",
    "SELECT customer_id, SUM(amount) AS total FROM orders WHERE created_at >= '2026-01-01' GROUP BY customer_id HAVING SUM(amount) > 1000 ORDER BY total DESC LIMIT 25;",
    "Shakespeare wrote that all the world is a stage, and all the men and women merely players; they have their exits and their entrances, and one man in his time plays many parts.",
    "A gated delta network maintains a recurrent state that is updated multiplicatively at each step, which lets it carry information across long contexts without the quadratic cost of full attention.",
    "The Treaty of Westphalia, signed in 1648, ended the Thirty Years' War and is often cited as the origin of the modern system of sovereign states.",
]


def capture(url: str, topk: int) -> dict:
    out = []
    for i, text in enumerate(CORPUS):
        req = urllib.request.Request(
            f"{url}/generate",
            data=json.dumps(
                {
                    "text": text,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                    "return_logprob": True,
                    "logprob_start_len": 0,
                    "top_logprobs_num": topk,
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        mi = json.loads(urllib.request.urlopen(req, timeout=300).read())["meta_info"]
        out.append(
            {
                "idx": i,
                "input_token_logprobs": mi["input_token_logprobs"],
                "input_top_logprobs": mi["input_top_logprobs"],
            }
        )
        print(f"  captured {i+1}/{len(CORPUS)} ({len(mi['input_token_logprobs'])} positions)", file=sys.stderr)
    return {"topk": topk, "passages": out}


def compare(ref: dict, new: dict) -> None:
    kld_sum = kld_n = 0.0
    dlp_sum = dlp_n = 0
    top1_hit = top1_n = 0
    worst = []

    for pr, pn in zip(ref["passages"], new["passages"]):
        for pos, (tr, tn) in enumerate(zip(pr["input_top_logprobs"], pn["input_top_logprobs"])):
            if not tr or not tn:
                continue
            # top-1 agreement: the token a greedy decoder would emit here
            top1_n += 1
            top1_hit += int(tr[0][1] == tn[0][1])
            # KLD over the reference's top-k support, renormalised on both sides
            nmap = {tok: lp for lp, tok, _ in tn}
            pairs = [(lp, nmap[tok]) for lp, tok, _ in tr if tok in nmap]
            if len(pairs) < 2:
                continue
            zr = math.log(sum(math.exp(a) for a, _ in pairs))
            zn = math.log(sum(math.exp(b) for _, b in pairs))
            k = sum(math.exp(a - zr) * ((a - zr) - (b - zn)) for a, b in pairs)
            kld_sum += k
            kld_n += 1

        for pos, (lr, ln) in enumerate(zip(pr["input_token_logprobs"], pn["input_token_logprobs"])):
            if lr[0] is None or ln[0] is None:
                continue
            d = abs(lr[0] - ln[0])
            dlp_sum += d
            dlp_n += 1
            worst.append((d, pr["idx"], pos, lr[1]))

    worst.sort(reverse=True)
    print(f"positions scored (KLD) : {int(kld_n)}")
    print(f"mean KLD (top-{ref['topk']} truncated, nats) : {kld_sum/kld_n:.6f}")
    print(f"mean |delta logprob| on the actual token (exact, full-vocab) : {dlp_sum/dlp_n:.6f} nats  over {dlp_n} positions")
    print(f"top-1 argmax agreement : {top1_hit}/{top1_n} = {100.0*top1_hit/top1_n:.2f} %")
    print("worst 5 positions by |delta logprob| (delta, passage, pos, token_id):")
    for w in worst[:5]:
        print(f"  {w[0]:.4f}  passage={w[1]} pos={w[2]} token={w[3]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--url", default="http://127.0.0.1:30030")
    c.add_argument("--topk", type=int, default=20)
    c.add_argument("--out", required=True)
    m = sub.add_parser("compare")
    m.add_argument("--ref", required=True)
    m.add_argument("--new", required=True)
    a = ap.parse_args()
    if a.cmd == "capture":
        json.dump(capture(a.url, a.topk), open(a.out, "w"))
        print(f"wrote {a.out}")
    else:
        compare(json.load(open(a.ref)), json.load(open(a.new)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
