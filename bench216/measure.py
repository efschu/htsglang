#!/usr/bin/env python3
"""Per-boot measurement for the MLP-split campaign (task #216 follow-up).

Two quantities from one boot, both in ms so the noise floor stays at the
0.1-0.5 % level that ``tok/s`` cannot reach:

* prefill_ms(L) -- e2e latency of a ``max_new_tokens=1`` request over a
  prompt of L tokens. The radix cache is defeated by a UNIQUE random token
  prefix per request (prefix matching is left-anchored, so one differing
  first token voids every shared node). ``#cached-token: 0`` in the server
  log is the proof, checked by the caller.
* decode_step_ms(ctx) -- (e2e(N) - e2e(1)) / (N - 1) over the same prompt
  length, which cancels the prefill and the one-token constant exactly.
"""
import argparse, json, random, statistics, sys, time, urllib.request

def post(url, payload, timeout=600):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

# Random token ids give an EXACT prompt length (a text prompt's token count
# is not controllable, and overshooting trips max_prefill_tokens with a 400)
# and a first token that is different on every call, which is what actually
# voids the radix cache -- prefix matching is left-anchored.
VOCAB_LO, VOCAB_HI = 1000, 240000

def make_prompt(rng, n_tok):
    return [rng.randrange(VOCAB_LO, VOCAB_HI) for _ in range(n_tok)]

def run(base, n_tok, max_new, rng):
    payload = {
        "input_ids": make_prompt(rng, n_tok),
        "sampling_params": {"max_new_tokens": max_new, "temperature": 0.0},
    }
    t0 = time.perf_counter()
    out = post(base + "/generate", payload)
    dt = (time.perf_counter() - t0) * 1000.0
    mi = out.get("meta_info", {})
    return dt, mi.get("prompt_tokens"), mi.get("completion_tokens")

# Random input_ids make the OUTPUT degenerate, which saturates speculative
# acceptance and compresses the difference between arms (a ceiling effect).
# The decode arm therefore also runs on natural text, where acceptance sits
# in its normal range. The prompt is IDENTICAL in every arm and every rep, so
# the radix cache may hit freely -- the (N vs 1) subtraction cancels prefill
# either way, and only the decode slope is read off.
PASSAGE = (
    "The memory bandwidth of a graphics processor is only an upper bound. "
    "What a kernel actually achieves depends on the shape of the operand, "
    "the quantization format, and how many independent requests are in "
    "flight at once. A small shard of a large matrix is latency bound long "
    "before it is bandwidth bound, and a scheduler that assumes otherwise "
    "will happily concentrate work on the fastest card and then wonder why "
    "the step time went up instead of down. "
)

def natural_prompt(approx_tokens):
    reps = max(1, approx_tokens // 90)
    return ("Summarize and continue the following passage.\n\n"
            + PASSAGE * reps + "\n\nContinue:")

def run_text(base, text, max_new):
    payload = {"text": text,
               "sampling_params": {"max_new_tokens": max_new, "temperature": 0.0}}
    t0 = time.perf_counter()
    out = post(base + "/generate", payload)
    dt = (time.perf_counter() - t0) * 1000.0
    mi = out.get("meta_info", {})
    return dt, mi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--decode-new", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    rng = random.Random(a.seed)

    PREFILL_LENS = [500, 1000, 2000, 4000, 8000, 11000]
    DECODE_LENS = [400, 11000]
    res = {"arm": a.arm, "prefill": {}, "decode": {}, "raw": []}

    # warm-up: graphs, allocator, JIT
    for _ in range(2):
        run(base, 400, 8, rng)

    # ---- prefill: max_new_tokens=1, unique prefix per request ----------
    for L in PREFILL_LENS:
        samples = []
        for _ in range(a.reps):
            dt, pt, ct = run(base, L, 1, rng)
            samples.append(dt)
            res["raw"].append({"kind": "prefill", "L": L, "ms": dt,
                               "prompt_tokens": pt, "completion_tokens": ct})
        res["prefill"][str(L)] = {
            "median_ms": statistics.median(samples),
            "min_ms": min(samples), "samples": samples,
            "prompt_tokens": pt,
        }
        print(f"[{a.arm}] prefill L={L:>6} "
              f"median={statistics.median(samples):8.1f} ms  "
              f"prompt_tokens={pt}", flush=True)

    # ---- decode: step time from the (N vs 1) difference -----------------
    for L in DECODE_LENS:
        one, many = [], []
        for _ in range(a.reps):
            d1, _, _ = run(base, L, 1, rng)
            one.append(d1)
        for _ in range(a.reps + 1):
            dN, _, cN = run(base, L, a.decode_new, rng)
            many.append((dN, cN))
        m1 = statistics.median(one)
        steps = [(dN - m1) / max(cN - 1, 1) for dN, cN in many]
        res["decode"][str(L)] = {
            "step_ms": statistics.median(steps), "steps": steps,
            "one_tok_ms": m1, "many": many,
        }
        res["raw"].append({"kind": "decode", "L": L,
                           "one_tok_ms": m1, "many": many})
        print(f"[{a.arm}] decode  ctx={L:>6} "
              f"step={statistics.median(steps):7.3f} ms "
              f"(1-tok {m1:.1f} ms)", flush=True)

    # ---- decode on natural text (normal acceptance regime) -------------
    res["decode_text"] = {}
    try:
        for approx in (400, 11000):
            text = natural_prompt(approx)
            one = [run_text(base, text, 1)[0] for _ in range(a.reps)]
            m1 = statistics.median(one)
            steps, accs, pts = [], [], None
            for _ in range(a.reps + 1):
                dN, mi = run_text(base, text, a.decode_new)
                cN = mi.get("completion_tokens") or a.decode_new
                pts = mi.get("prompt_tokens")
                steps.append((dN - m1) / max(cN - 1, 1))
                if mi.get("spec_accept_length"):
                    accs.append(mi["spec_accept_length"])
            res["decode_text"][str(approx)] = {
                "step_ms": statistics.median(steps), "steps": steps,
                "one_tok_ms": m1, "prompt_tokens": pts,
                "spec_accept_length": accs,
            }
            print(f"[{a.arm}] decode-text ctx~{approx:>6} "
                  f"step={statistics.median(steps):7.3f} ms "
                  f"(prompt_tokens={pts}, accept={accs[:1]})", flush=True)
    except Exception as e:                      # never lose the rest of the run
        res["decode_text_error"] = repr(e)
        print(f"[{a.arm}] decode-text FAILED: {e!r}", flush=True)

    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[{a.arm}] wrote {a.out}", flush=True)

if __name__ == "__main__":
    main()
