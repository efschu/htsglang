"""#201 slice 2 -- decode rate, prefill TTFT and an output sample from a running server.

Time-bounded rather than token-bounded: a decode arm is asked for a large
``max_tokens`` and cut off after ``--seconds``, so a slow configuration costs the
same wall clock as a fast one and the working point is reached by prefilling to
it rather than growing into it.

Every run prints the generated text so the number can never be reported without
the output that produced it, and repeated runs of the SAME arm are what the
noise floor is read from (``--repeat``).
"""

import argparse
import json
import time
import urllib.request


def _stream(base, path, payload, timeout, deadline):
    """Stream a completion, stop at the deadline.

    Returns (first_token_s, n_tokens, elapsed_s, text). The elapsed time is
    MEASURED, not assumed to be the deadline: a greedy arm can run out of things
    to say before the window closes, and charging it the full window would
    report a rate it never ran at.
    """
    payload = {**payload, "stream": True}
    req = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first = None
    chunks = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            delta = json.loads(body)["choices"][0].get("delta", {})
            piece = delta.get("content")
            if piece is None:
                continue
            if first is None:
                first = time.perf_counter() - started
            chunks.append(piece)
            if time.perf_counter() - started > deadline:
                break
    return first, len(chunks), time.perf_counter() - started, "".join(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="http://host:port")
    ap.add_argument("--seconds", type=float, default=18.0)
    ap.add_argument("--prefill-tokens", type=int, default=8000)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--label", default="arm")
    args = ap.parse_args()

    prompt = (
        "Explain, in detailed technical prose, why a pipeline stage boundary "
        "moves less data than a tensor-parallel group, and what that means for "
        "two machines connected by a 40 gigabit link. "
    )

    for run in range(args.repeat):
        # (1) short-prompt decode: the steady-state rate, time-bounded.
        first, n, elapsed, text = _stream(
            args.base,
            "/v1/chat/completions",
            {
                "model": "default",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,
                "temperature": 0.0,
            },
            timeout=args.seconds + 120,
            deadline=args.seconds,
        )
        decode_tps = (n - 1) / max(elapsed - (first or 0.0), 1e-9)
        ms_per_token = 1000.0 / decode_tps if decode_tps else float("nan")

        # (2) long prefill: TTFT at a real working point, reached by prefilling
        # to it. A fresh nonce per run keeps the radix cache out of the number.
        nonce = f"run-{args.label}-{run}-{time.time_ns()} "
        long_prompt = nonce + (
            "pipeline parallelism stage boundary activation "
            * (args.prefill_tokens // 6)
        )
        pf_first, pf_n, _pf_elapsed, pf_text = _stream(
            args.base,
            "/v1/chat/completions",
            {
                "model": "default",
                "messages": [
                    {
                        "role": "user",
                        "content": long_prompt
                        + "\nSummarize the above in one sentence.",
                    }
                ],
                "max_tokens": 64,
                "temperature": 0.0,
            },
            timeout=300,
            deadline=120,
        )

        print(
            f"[{args.label} run {run}] decode: {n} tok in {elapsed:.1f} s -> "
            f"{decode_tps:.2f} tok/s ({ms_per_token:.2f} ms/token), TTFT {first * 1000:.0f} ms"
        )
        print(
            f"[{args.label} run {run}] prefill ~{args.prefill_tokens} tok: "
            f"TTFT {pf_first * 1000:.0f} ms, then {pf_n} tok"
        )
        print(f"[{args.label} run {run}] sample: {text[:300]!r}")
        print(f"[{args.label} run {run}] prefill sample: {pf_text[:160]!r}", flush=True)


if __name__ == "__main__":
    main()
