# SPDX-License-Identifier: Apache-2.0
"""#631 Route A acceptance driver: the numbers the user asked for.

ONE instance, THREE ranks: PP=3 prefill -> live flip -> TP=3 decode with
NEXTN speculation armed. This script drives that sequence against a
running flip boot and emits the acceptance record as json:

    reshard duration, prefill tok/s, decode tok/s, TTFT (ms),
    spec accept length

Measurement discipline (rig doctrine, DESIGN_631 section 4):

* Every draw is uncached: prompts are random input_ids inside the vocab,
  so prefix caching cannot contaminate a repeat.
* TTFT comes from a STREAMED request -- the first token's arrival, not
  the whole response's wall time, which is a decode measurement wearing
  a prefill's name.
* The spec accept length is read from ``meta_info["spec_accept_length"]``
  (completion_tokens / spec_verify_ct). The server's rolling
  ``spec_ema_accept_len`` is NOT that quantity and is not used here.
* Prefill is measured in the PP phase and decode in the TP phase --
  which is the whole point of the layout flip, so a run that reports
  both from one phase has not measured Route A.
* A warm-up draw is discarded before every timed series.

Subcommands:
  ladder     PP-phase prefill ladder (tok/s per length)
  flip       arm a flip and time it from the client side
  decode     TP-phase decode: tok/s, TTFT, spec accept length
  full       the acceptance sequence end to end, one json record
"""

import argparse
import json
import random
import time
import urllib.request


def _post(url, path, payload, timeout=1800.0):
    req = urllib.request.Request(
        url + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(url, path, timeout=30.0):
    with urllib.request.urlopen(url + path, timeout=timeout) as r:
        return r.read().decode()


def _ids(rng, n, vocab):
    return [rng.randrange(1000, vocab) for _ in range(n)]


def _phase(url):
    """The server's live phase, from the flip control plane."""
    try:
        return json.loads(_get(url, "/phase_flip_status"))
    except Exception:
        return None


# -- prefill ------------------------------------------------------------


def prefill_draw(url, n_tokens, vocab, rng):
    """One uncached prefill draw; max_new_tokens=1 isolates prefill."""
    ids = _ids(rng, n_tokens, vocab)
    t0 = time.perf_counter()
    _post(url, "/generate", {
        "input_ids": ids,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
    })
    ms = (time.perf_counter() - t0) * 1000.0
    return ms, n_tokens / (ms / 1000.0)


def cmd_ladder(args):
    rng = random.Random(args.seed)
    out = {"phase": "pp_prefill", "lengths": {}}
    for n in args.lengths:
        prefill_draw(url=args.url, n_tokens=n, vocab=args.vocab, rng=rng)  # warm-up
        draws = [prefill_draw(args.url, n, args.vocab, rng) for _ in range(args.draws)]
        ms = sorted(d[0] for d in draws)
        med = ms[len(ms) // 2]
        out["lengths"][str(n)] = {
            "ms_median": round(med, 1),
            "ms_all": [round(x, 1) for x in ms],
            "tok_s": round(n / (med / 1000.0), 1),
        }
    return out


# -- flip ---------------------------------------------------------------


def cmd_flip(args):
    """Arm a flip and time it from the client side.

    The client-side number includes the arm RPC and the wait for a
    quiescent boundary; the authoritative per-rank move cost is the
    server's PHASE-FLIP DONE record, parsed offline by
    route_a_631_step6_bench.py flip-stats.
    """
    t0 = time.perf_counter()
    resp = _post(args.url, "/phase_flip", {"direction": args.direction}, timeout=600.0)
    ms = (time.perf_counter() - t0) * 1000.0
    return {
        "direction": args.direction,
        "client_ms": round(ms, 1),
        "response": resp,
    }


# -- decode -------------------------------------------------------------


def decode_stream(url, prompt_ids, max_new, timeout=1800.0):
    """Stream one generation; return TTFT, wall time and the final meta.

    Streaming is what makes TTFT a real measurement: the first chunk's
    arrival separates prefill+first-token from the decode that follows.
    """
    payload = {
        "input_ids": prompt_ids,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": max_new},
        "stream": True,
    }
    req = urllib.request.Request(
        url + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft_ms = None
    last = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t0) * 1000.0
            try:
                last = json.loads(body)
            except Exception:
                pass
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return ttft_ms, wall_ms, last


def _decode_record(ttft_ms, wall_ms, last):
    meta = (last or {}).get("meta_info", {}) or {}
    completion = meta.get("completion_tokens")
    rec = {
        "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
        "wall_ms": round(wall_ms, 1),
        "completion_tokens": completion,
        # Decode rate excludes the first token: it is produced by the
        # prefill step, and counting it inflates a short run's tok/s.
        "decode_tok_s": (
            round((completion - 1) / ((wall_ms - ttft_ms) / 1000.0), 2)
            if completion and completion > 1 and ttft_ms is not None
            and wall_ms > ttft_ms
            else None
        ),
        "spec_accept_length": meta.get("spec_accept_length"),
        "spec_accept_rate": meta.get("spec_accept_rate"),
        "spec_verify_ct": meta.get("spec_verify_ct"),
    }
    return rec


def cmd_decode(args):
    rng = random.Random(args.seed)
    ids = _ids(rng, args.prompt_tokens, args.vocab)
    decode_stream(args.url, _ids(rng, args.prompt_tokens, args.vocab), 8)  # warm-up
    ttft, wall, last = decode_stream(args.url, ids, args.max_new)
    rec = _decode_record(ttft, wall, last)
    rec["phase"] = "tp_decode"
    rec["prompt_tokens"] = args.prompt_tokens
    return rec


# -- the acceptance sequence -------------------------------------------


def cmd_full(args):
    """PP prefill -> flip -> TP decode with speculation, in one record."""
    rng = random.Random(args.seed)
    rec = {
        "boot": {"url": args.url},
        "phase_before": _phase(args.url),
    }

    # 1. PP-phase prefill ladder.
    rec["pp_prefill"] = cmd_ladder(
        argparse.Namespace(
            url=args.url, lengths=args.lengths, draws=args.draws,
            vocab=args.vocab, seed=args.seed,
        )
    )

    # 2. The flip.
    rec["flip_pp_to_tp"] = cmd_flip(
        argparse.Namespace(url=args.url, direction="pp_to_tp")
    )
    rec["phase_after_flip"] = _phase(args.url)

    # 3. TP-phase decode with speculation live.
    rec["tp_decode"] = cmd_decode(
        argparse.Namespace(
            url=args.url, prompt_tokens=args.prompt_tokens,
            max_new=args.max_new, vocab=args.vocab, seed=args.seed + 1,
        )
    )

    # 4. Return trip, so the record covers a round trip like rung b.
    if args.round_trip:
        rec["flip_tp_to_pp"] = cmd_flip(
            argparse.Namespace(url=args.url, direction="tp_to_pp")
        )
        rec["phase_after_return"] = _phase(args.url)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30023")
    ap.add_argument("--out", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ladder")
    p.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192, 32768])
    p.add_argument("--draws", type=int, default=3)
    p.add_argument("--vocab", type=int, default=100000)
    p.add_argument("--seed", type=int, default=631)
    p.set_defaults(fn=cmd_ladder)

    p = sub.add_parser("flip")
    p.add_argument("--direction", default="pp_to_tp")
    p.set_defaults(fn=cmd_flip)

    p = sub.add_parser("decode")
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--vocab", type=int, default=100000)
    p.add_argument("--seed", type=int, default=631)
    p.set_defaults(fn=cmd_decode)

    p = sub.add_parser("full")
    p.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192])
    p.add_argument("--draws", type=int, default=3)
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--vocab", type=int, default=100000)
    p.add_argument("--seed", type=int, default=631)
    p.add_argument("--round-trip", action="store_true")
    p.set_defaults(fn=cmd_full)

    args = ap.parse_args(argv)
    out = args.fn(args)
    text = json.dumps(out, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
