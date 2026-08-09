#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#631 / #543: single-session long-context needle probe past the native ceiling.

WHAT THIS PROVES, and what it deliberately does not.

The bs1 long-context leg asks for ONE session whose context reaches BEYOND
the model's native 262144 ceiling, with the content at that depth verified
-- not merely a request that fails to crash. A prompt that is accepted and
answered with a generic summary proves nothing: an attention path that
silently drops everything past position 262144 would look identical.

So the probe plants three verbatim needles at known DEPTHS:

  * ``early``  ~3 % -- control, inside every ceiling.
  * ``native`` ~55 % -- inside the native 262144 window at the target size.
  * ``deep``   ~95 % -- PAST 262144, i.e. only reachable with rope scaling
    in force. This is the one the leg turns on.

Each needle is a random code, generated per run, so a correct answer cannot
come from the prompt template, the model's priors, or a previous run's KV.

The report is a per-needle hit/miss plus the server's own
``usage.prompt_tokens``, which is the authority on how long the session
actually was -- not the client's token estimate.

The second question re-sends the SAME prefix with a different trailing
question. With the radix cache on, that is a prefix-cache hit, so the
follow-up costs a few hundred tokens of prefill rather than another full
one. It isolates "the model can retrieve the deep needle" from "the model
can retrieve three needles in one answer", which are different asks.
"""

from __future__ import annotations

import argparse
import json
import random
import string
import sys
import time

import requests

FILLER = (
    "Line {i:07d}: archive index entry {i} records a routine maintenance "
    "check on subsystem {s}, no anomalies reported, signed off by the duty "
    "operator on shift {t}."
)
SUBSYS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]


def make_code(rng: random.Random) -> str:
    letters = "".join(rng.choice(string.ascii_uppercase) for _ in range(5))
    digits = "".join(rng.choice(string.digits) for _ in range(4))
    return f"{letters}-{digits}"


def build_prompt(target_tokens: int, tok, rng: random.Random, n_lines=None):
    """Build a filler corpus of ~target_tokens with three planted needles."""
    if n_lines is None:
        sample = "\n".join(
            FILLER.format(i=i, s=SUBSYS[i % 8], t=i % 3) for i in range(200)
        )
        per_line = len(tok.encode(sample)) / 200.0
        n_lines = int(target_tokens / per_line)
    else:
        per_line = target_tokens / max(n_lines, 1)

    needles = {}
    for name, depth in (("early", 0.03), ("native", 0.55), ("deep", 0.95)):
        needles[name] = {"depth": depth, "code": make_code(rng),
                         "line": int(n_lines * depth)}

    lines = []
    at_line = {v["line"]: k for k, v in needles.items()}
    for i in range(n_lines):
        name = at_line.get(i)
        if name is not None:
            code = needles[name]["code"]
            lines.append(
                f"Line {i:07d}: PRIORITY RECORD -- the vault access code for "
                f"vault {name.upper()} is {code}. Memorize it verbatim."
            )
        else:
            lines.append(FILLER.format(i=i, s=SUBSYS[i % 8], t=i % 3))
    body = "\n".join(lines)
    return body, needles, per_line


def ask(url: str, model: str, body: str, question: str, max_tokens: int,
        timeout: float):
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": body + "\n\n" + question},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=timeout)
    dt = time.time() - t0
    r.raise_for_status()
    d = r.json()
    choice = d["choices"][0]
    msg = choice.get("message", {})
    text = (msg.get("content") or "") + "\n" + (msg.get("reasoning_content") or "")
    return {
        "seconds": round(dt, 2),
        "usage": d.get("usage", {}),
        "meta_info": d.get("meta_info", {}) or choice.get("meta_info", {}) or {},
        "text": text.strip()[:2000],
        "finish_reason": choice.get("finish_reason"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30030)
    ap.add_argument("--model", default="Qwen3.6-27B")
    ap.add_argument("--model-dir",
                    default="/spinning/llm_stuff/club-3090/models-cache/"
                            "Qwen3.6-27B-INT8-W8A8")
    ap.add_argument("--target-tokens", type=int, default=300000)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    rng = random.Random(20260809)

    # THE CEILING IS THE SERVER'S, NOT THE CLIENT'S GUESS. A request one
    # token over max_req_input_len is refused with a bare HTTP 400, which
    # looks exactly like a broken probe. The first run of this script died
    # that way: a per-line token estimate taken from the first 200 lines
    # under-counted (low line numbers tokenize shorter than high ones), the
    # corpus landed at 278976 against a limit of 277462, and the whole leg
    # reported nothing. So: ask the server for its limit, aim below it, then
    # CORRECT the line count against a real encode and re-measure.
    cap = None
    try:
        info = requests.get(
            f"http://127.0.0.1:{args.port}/get_server_info", timeout=10
        ).json()
        cap = int(info.get("max_req_input_len") or 0) or None
    except Exception:  # noqa: BLE001 - probe runs without it, just less safely
        cap = None
    target = args.target_tokens
    if cap is not None:
        target = min(target, cap - args.max_tokens - 1024)

    body, needles, per_line = build_prompt(target, tok, rng)
    client_tokens = len(tok.encode(body))
    for _ in range(3):
        if abs(client_tokens - target) <= 2000 and client_tokens <= target:
            break
        n_lines = max(1, int(len(body.split("\n")) * target / client_tokens))
        body, needles, per_line = build_prompt(target, tok, rng, n_lines=n_lines)
        client_tokens = len(tok.encode(body))
    print(f"corpus calibrated: {client_tokens} tokens (target {target}, "
          f"server cap {cap})", flush=True)

    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    report = {
        "target_tokens": args.target_tokens,
        "effective_target_tokens": target,
        "server_max_req_input_len": cap,
        "client_estimated_prompt_tokens": client_tokens,
        "tokens_per_filler_line": round(per_line, 3),
        "needles": needles,
        "rounds": [],
    }

    q_all = ("The archive above contains three PRIORITY RECORD lines, one for "
             "vault EARLY, one for vault NATIVE and one for vault DEEP. Report "
             "all three access codes exactly as written, one per line, in the "
             "form VAULT=CODE. Answer with those three lines only.")
    q_deep = ("In the archive above there is exactly one PRIORITY RECORD line "
              "for vault DEEP. Reply with only its access code, verbatim.")

    for label, question in (("all_three", q_all), ("deep_only", q_deep)):
        try:
            res = ask(url, args.model, body, question, args.max_tokens,
                      args.timeout)
        except Exception as exc:  # noqa: BLE001 - the failure IS the result
            res = {"error": f"{type(exc).__name__}: {exc}"}
        res["label"] = label
        text = res.get("text", "")
        res["hits"] = {n: (v["code"] in text) for n, v in needles.items()}
        # The server's own count is the authority on session length.
        pt = (res.get("usage") or {}).get("prompt_tokens")
        res["prompt_tokens"] = pt
        res["past_native_ceiling"] = bool(pt and pt > 262144)
        report["rounds"].append(res)
        print(json.dumps({k: res[k] for k in
                          ("label", "prompt_tokens", "past_native_ceiling",
                           "hits", "seconds") if k in res}), flush=True)

    out = json.dumps(report, indent=1)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(out)
    print(out)
    deep_ok = any(r.get("hits", {}).get("deep") for r in report["rounds"])
    past = any(r.get("past_native_ceiling") for r in report["rounds"])
    print(f"VERDICT past_native_ceiling={past} deep_needle_verified={deep_ok}")
    return 0 if (past and deep_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
