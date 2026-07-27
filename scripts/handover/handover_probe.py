#!/usr/bin/env python3
"""Drive one arm of the #121 session-handover test.

phase A: send a prompt as raw token ids, decode for a bounded time, and write
         the exact prompt+output token ids to a json file. Token ids, not
         text: the prefix match the handover is judged on is a TOKEN prefix,
         and re-tokenizing generated text does not reliably reproduce it.
phase B: replay those ids plus a question that can only be answered from the
         imported context, and report how many prompt tokens the server
         counted as CACHED. On a freshly booted server the only possible
         source of a cached prefix is the persistent store.
"""

import argparse
import json
import sys
import time
import urllib.request


def post(url: str, payload: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def output_ids(meta: dict) -> list:
    lp = meta.get("output_token_logprobs") or []
    return [int(e[1]) for e in lp]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--phase", choices=["a", "b"], required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--state", required=True, help="json carrying the ids")
    ap.add_argument("--max-new-tokens", type=int, default=220)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    base = f"http://127.0.0.1:{args.port}"

    if args.phase == "a":
        prompt = (
            "Field report, rig inventory.\n\n"
            "The workshop runs three machines. Machine ONE is called Kestrel and "
            "carries a single large card. Machine TWO is called Marlin and carries "
            "two smaller cards. Machine THREE is called Osprey and is the spare.\n\n"
            "Write a short continuation of this report describing what each "
            "machine is used for.\n\nContinuation:"
        )
        ids = tok.encode(prompt, add_special_tokens=False)
        t0 = time.time()
        r = post(
            base + "/generate",
            {
                "input_ids": ids,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": args.max_new_tokens,
                },
                "return_logprob": True,
                "logprob_start_len": -1,
            },
        )
        out = output_ids(r["meta_info"])
        elapsed = time.time() - t0
        state = {
            "prompt_ids": ids,
            "output_ids": out,
            "text": r["text"],
            "elapsed_s": elapsed,
            "meta": {
                k: r["meta_info"].get(k)
                for k in ("prompt_tokens", "completion_tokens", "cached_tokens")
            },
        }
        with open(args.state, "w") as f:
            json.dump(state, f)
        print(
            f"[phase A] {len(ids)} prompt ids, {len(out)} output ids, "
            f"{elapsed:.1f}s, cached_tokens={state['meta']['cached_tokens']}"
        )
        print("[phase A output, first 300 chars]")
        print(r["text"][:300])
        return 0

    with open(args.state) as f:
        state = json.load(f)
    # A continuation stub, not a question turn: the raw (un-templated) prompt
    # makes the model close the turn on a "Question:/Answer:" shape, which
    # would test the chat format rather than the handover. The names can only
    # come from the imported context.
    question = (
        "\n\nAppendix A -- machine index, copied from the report above:\n"
        "- Machine ONE is named"
    )
    q_ids = tok.encode(question, add_special_tokens=False)
    carried = state["prompt_ids"] + state["output_ids"]
    r = post(
        base + "/generate",
        {
            "input_ids": carried + q_ids,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 160},
        },
    )
    meta = r["meta_info"]
    cached = meta.get("cached_tokens")
    print(
        f"[phase B] carried prefix {len(carried)} ids, sent "
        f"{len(carried) + len(q_ids)} ids, cached_tokens={cached}, "
        f"prompt_tokens={meta.get('prompt_tokens')}"
    )
    print("[phase B output, first 300 chars]")
    print(r["text"][:300])
    if not cached:
        print("VERDICT: no prefix was imported -- handover did NOT happen")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
