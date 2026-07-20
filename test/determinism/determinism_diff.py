#!/usr/bin/env python3
"""Reusable determinism-diff harness for byte-identity gates (#128, seeds #124).

Parameterized client-side tool against a RUNNING sglang server. Three modes:

  capture   Run a fixed greedy prompt set once, record the full decode
            trajectory (token ids + per-token output logprobs, optionally
            input/extend logprobs) to a canonical JSON file.
  diff      Compare two capture files: token-trajectory equality and the
            max |delta logprob|. Exit 0 iff trajectories match and the max
            delta <= --tol (default 0.0 = machine-zero / bit-identical).
  selfdet   Run the same prompt set N times (default 5) against one server
            and require all runs bitwise-identical (self-determinism N/N).

Typical #128 gate (same build, kill-switch A/B, SAME execution mode):
  SGLANG_DCP_COMM_OVERLAP=0 server -> capture baseline.json
  SGLANG_DCP_COMM_OVERLAP=1 server -> capture overlap.json
  determinism_diff.py diff baseline.json overlap.json --tol 0.0
  determinism_diff.py selfdet --url ... --runs 5

Typical #124 regression shape (weightless/graph vs baseline):
  capture per config, then diff with a mode-appropriate --tol
  (eager-vs-eager: 0.0; captured-vs-eager: the platform floor).

Stdlib-only (urllib); no venv dependencies. Uses the native /generate
endpoint with return_logprob so the decode trajectory is exact.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

DEFAULT_PROMPTS = [
    "The capital of France is",
    "Write a haiku about tensor parallelism.",
    "Explain, step by step, why the sky is blue.",
    "def fibonacci(n):",
    "List three prime numbers greater than 100 and justify each briefly.",
]


def _post_json(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_healthy(base_url, timeout_s):
    """Bounded health poll (anti-idle discipline: never wait unbounded)."""
    deadline = time.time() + timeout_s
    url = base_url.rstrip("/") + "/health"
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(5)
    return False


def run_prompt_set(args, prompts):
    """One pass over the prompt set; returns the canonical trajectory dict."""
    base = args.url.rstrip("/")
    results = []
    gen_tokens = 0
    gen_seconds = 0.0
    for prompt in prompts:
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": args.max_tokens,
                "top_k": 1,
            },
            "return_logprob": True,
        }
        if args.input_logprobs:
            payload["logprob_start_len"] = 0
        t0 = time.time()
        resp = _post_json(base + "/generate", payload, args.timeout)
        dt = time.time() - t0
        meta = resp.get("meta_info", {})
        out_lp = meta.get("output_token_logprobs") or []
        gen_tokens += len(out_lp)
        gen_seconds += dt
        entry = {
            "prompt": prompt,
            "output_text": resp.get("text", ""),
            "token_ids": [t[1] for t in out_lp],
            "logprobs": [t[0] for t in out_lp],
        }
        if args.input_logprobs:
            in_lp = meta.get("input_token_logprobs") or []
            entry["input_token_ids"] = [t[1] for t in in_lp]
            entry["input_logprobs"] = [
                t[0] for t in in_lp if t[0] is not None
            ]
        results.append(entry)
    tps = gen_tokens / gen_seconds if gen_seconds > 0 else 0.0
    print(
        f"pass: {gen_tokens} output tokens in {gen_seconds:.1f}s "
        f"(sequential e2e ~= {tps:.1f} tok/s incl. prefill)"
    )
    return {
        "meta": {
            "url": args.url,
            "max_tokens": args.max_tokens,
            "num_prompts": len(prompts),
            "input_logprobs": bool(args.input_logprobs),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "label": args.label,
            "gen_tokens": gen_tokens,
            "gen_seconds": round(gen_seconds, 3),
            "tok_per_s_e2e": round(tps, 2),
        },
        "results": results,
    }


def compare_captures(a, b, tol):
    """Returns (ok, report_lines). Token mismatch is always fatal; logprob
    deltas are compared against tol (0.0 = bit-identical requirement)."""
    lines = []
    ok = True
    ra, rb = a["results"], b["results"]
    if len(ra) != len(rb):
        return False, [f"FATAL: prompt-set size differs: {len(ra)} vs {len(rb)}"]
    max_delta = 0.0
    for i, (ea, eb) in enumerate(zip(ra, rb)):
        if ea["prompt"] != eb["prompt"]:
            ok = False
            lines.append(f"[{i}] FATAL: prompt text differs")
            continue
        if ea["token_ids"] != eb["token_ids"]:
            ok = False
            n = next(
                (
                    j
                    for j, (x, y) in enumerate(
                        zip(ea["token_ids"], eb["token_ids"])
                    )
                    if x != y
                ),
                min(len(ea["token_ids"]), len(eb["token_ids"])),
            )
            lines.append(
                f"[{i}] TOKEN TRAJECTORY DIVERGES at output token {n} "
                f"(lens {len(ea['token_ids'])} vs {len(eb['token_ids'])})"
            )
            continue
        deltas = [
            abs(x - y) for x, y in zip(ea["logprobs"], eb["logprobs"])
        ]
        d = max(deltas) if deltas else 0.0
        max_delta = max(max_delta, d)
        if d > tol:
            ok = False
            lines.append(
                f"[{i}] logprob max |delta| = {d:.3e} > tol {tol:.3e}"
            )
        for key in ("input_logprobs",):
            if key in ea and key in eb and ea[key] and eb[key]:
                di = max(abs(x - y) for x, y in zip(ea[key], eb[key]))
                max_delta = max(max_delta, di)
                if di > tol:
                    ok = False
                    lines.append(
                        f"[{i}] {key} max |delta| = {di:.3e} > tol {tol:.3e}"
                    )
    lines.append(
        f"max |delta logprob| across all prompts: {max_delta:.6e} "
        f"(tol {tol:.3e}) -> {'PASS' if ok else 'FAIL'}"
    )
    return ok, lines


def load_prompts(args):
    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = [ln.rstrip("\n") for ln in f if ln.strip()]
        if not prompts:
            sys.exit("prompts file is empty")
        return prompts
    return DEFAULT_PROMPTS


def cmd_capture(args):
    if not wait_healthy(args.url, args.health_timeout):
        sys.exit(f"server at {args.url} not healthy within {args.health_timeout}s")
    cap = run_prompt_set(args, load_prompts(args))
    with open(args.out, "w") as f:
        json.dump(cap, f, indent=1)
    n_tok = sum(len(e["token_ids"]) for e in cap["results"])
    print(f"captured {len(cap['results'])} prompts / {n_tok} output tokens -> {args.out}")


def cmd_diff(args):
    with open(args.file_a) as f:
        a = json.load(f)
    with open(args.file_b) as f:
        b = json.load(f)
    ok, lines = compare_captures(a, b, args.tol)
    print(f"A: {args.file_a} (label={a['meta'].get('label')})")
    print(f"B: {args.file_b} (label={b['meta'].get('label')})")
    for ln in lines:
        print(ln)
    sys.exit(0 if ok else 1)


def cmd_selfdet(args):
    if not wait_healthy(args.url, args.health_timeout):
        sys.exit(f"server at {args.url} not healthy within {args.health_timeout}s")
    prompts = load_prompts(args)
    ref = run_prompt_set(args, prompts)
    passes = 1
    all_ok = True
    for r in range(2, args.runs + 1):
        cap = run_prompt_set(args, prompts)
        ok, lines = compare_captures(ref, cap, 0.0)
        status = "identical" if ok else "DIVERGED"
        print(f"run {r}/{args.runs} vs run 1: {status}")
        if not ok:
            all_ok = False
            for ln in lines:
                print("  " + ln)
        else:
            passes += 1
    print(f"self-determinism: {passes}/{args.runs} bitwise-identical")
    sys.exit(0 if all_ok else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="mode", required=True)

    def add_server_args(sp):
        sp.add_argument("--url", default="http://127.0.0.1:30000")
        sp.add_argument("--max-tokens", type=int, default=200)
        sp.add_argument("--prompts-file", default=None)
        sp.add_argument("--input-logprobs", action="store_true",
                        help="also capture/compare prefill (extend) logprobs")
        sp.add_argument("--timeout", type=float, default=600.0)
        sp.add_argument("--health-timeout", type=float, default=60.0)
        sp.add_argument("--label", default=None,
                        help="free-form label stored in the capture meta")

    sp = sub.add_parser("capture", help="record a greedy trajectory capture")
    add_server_args(sp)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_capture)

    sp = sub.add_parser("diff", help="compare two captures")
    sp.add_argument("file_a")
    sp.add_argument("file_b")
    sp.add_argument("--tol", type=float, default=0.0,
                    help="max |delta logprob| allowed (0.0 = bit-identical)")
    sp.set_defaults(func=cmd_diff)

    sp = sub.add_parser("selfdet", help="N repeated runs must be identical")
    add_server_args(sp)
    sp.add_argument("--runs", type=int, default=5)
    sp.set_defaults(func=cmd_selfdet)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
