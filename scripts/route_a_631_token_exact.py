# SPDX-License-Identifier: Apache-2.0
"""#631 step-6 rung 3: token-exact equivalence of post-flip decode.

Acceptance (DESIGN_631, in-flight prefixes): decode after
PP-prefill + flip must produce the same tokens as the no-flip reference
decode of the same prompt at temperature 0, within the fork's
determinism envelope.

Usage (two runs against the SAME flip boot, or a separate reference
server):

  1. reference run (no flip armed):
     python route_a_631_token_exact.py --url http://127.0.0.1:30023 \
         --prompt-file p.txt --max-new 128 --out ref.json
  2. flip run: start the SAME prompt, arm the flip mid-request via
     --flip-after-prefill (the script waits for the prefill to be
     scheduled, POSTs /phase_flip pp_to_tp, then streams the decode):
     python route_a_631_token_exact.py --url ... --prompt-file p.txt \
         --max-new 128 --flip-after-prefill --out flip.json
  3. compare:
     python route_a_631_token_exact.py --compare ref.json flip.json

The comparison reports the FIRST divergence position and both token ids
around it -- a truncated-GDN-state bug diverges within a few tokens
(the state feeds every step), while numeric-envelope near-ties diverge
late and isolated; the report keeps the two distinguishable.
"""

import argparse
import json
import sys
import time
import urllib.request


def _post(url: str, path: str, payload: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        url + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run_generate(args) -> dict:
    prompt = open(args.prompt_file).read()
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": args.max_new,
        },
        "return_logprob": True,
        "logprob_start_len": -1,
    }
    if args.flip_after_prefill:
        # Fire the generate asynchronously-ish: issue it in a thread and
        # arm the flip once the request is in flight. The flip commits at
        # the quiescent boundary between this request's prefill and its
        # decode (requests PARKED) -- exactly the acceptance window.
        import threading

        result = {}

        def _gen():
            result["out"] = _post(args.url, "/generate", payload)

        t = threading.Thread(target=_gen)
        t.start()
        time.sleep(args.flip_delay_s)
        flip = _post(args.url, "/phase_flip", {"direction": "pp_to_tp"})
        print(f"phase_flip arm: {flip}")
        t.join()
        out = result["out"]
    else:
        out = _post(args.url, "/generate", payload)

    meta = out.get("meta_info", {})
    token_ids = [
        int(t[1]) for t in meta.get("output_token_logprobs", []) or []
    ]
    record = {
        "url": args.url,
        "flip": bool(args.flip_after_prefill),
        "text": out.get("text", ""),
        "token_ids": token_ids,
        "meta": {
            k: meta.get(k)
            for k in ("completion_tokens", "prompt_tokens", "finish_reason")
        },
    }
    json.dump(record, open(args.out, "w"), indent=1)
    print(
        f"wrote {args.out}: {len(token_ids)} decode tokens, "
        f"finish={record['meta'].get('finish_reason')}"
    )
    return record


def compare(path_a: str, path_b: str) -> int:
    a = json.load(open(path_a))
    b = json.load(open(path_b))
    ta, tb = a["token_ids"], b["token_ids"]
    if not ta or not tb:
        print("FAIL: one side has no token ids (need return_logprob)")
        return 2
    n = min(len(ta), len(tb))
    for i in range(n):
        if ta[i] != tb[i]:
            lo = max(0, i - 3)
            print(
                f"DIVERGENCE at decode position {i}/{n}: "
                f"{path_a}[{lo}:{i + 3}]={ta[lo:i + 3]} vs "
                f"{path_b}[{lo}:{i + 3}]={tb[lo:i + 3]}"
            )
            print(
                "early divergence (< ~8 tokens) smells like moved-state "
                "corruption; late isolated divergence smells like the "
                "numeric envelope -- judge against the design's "
                "determinism-envelope clause"
            )
            return 1
    if len(ta) != len(tb):
        print(
            f"LENGTH MISMATCH after {n} identical tokens: "
            f"{len(ta)} vs {len(tb)}"
        )
        return 1
    print(f"TOKEN-EXACT: {n} decode tokens identical")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30023")
    ap.add_argument("--prompt-file")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--out", default="run.json")
    ap.add_argument("--flip-after-prefill", action="store_true")
    ap.add_argument(
        "--flip-delay-s",
        type=float,
        default=0.3,
        help="delay before arming, so the prefill is scheduled first",
    )
    ap.add_argument("--compare", nargs=2, metavar=("REF", "FLIP"))
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    if not args.prompt_file:
        ap.error("--prompt-file required unless --compare")
    run_generate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
