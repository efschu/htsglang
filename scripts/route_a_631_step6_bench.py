# SPDX-License-Identifier: Apache-2.0
"""#631 step-6 measurement driver -- window is EXECUTION-ONLY.

Subcommands (all output json; nothing tails logs into an agent context):

* ``prefill-ladder``: uncached random-input prefill draws at the #625
  lengths (2048/8192/32768), warm-up draw discarded, N kept draws, and
  the A-vs-A same-boot floor FIRST (two identical batches back-to-back,
  spread of medians) -- no delta is meaningful before the floor exists
  (measurement doctrine, DESIGN_631 section 4).
* ``decode``: one long generation, wall tok/s (the authoritative
  ms/Verify-per-rank split comes from the server's RankPrefillLog /
  CollectiveClock lines; this is only the sanity envelope).
* ``flip-stats``: OFFLINE parse of a server log for the PHASE-FLIP DONE
  and PHASE-FLIP-GDN records -> per-phase ms per rank json. Run it on
  the log FILE after the window; keeps logs out of context.
* ``corridor``: wrapper that execs the existing rig sampler
  (/spinning/vram_corridor_sampler.py, NVML-free column, 100 ms) --
  run alongside EVERY load window; the acceptance is the per-card
  time-series minimum >= 1024 MiB.

Every draw uses input_ids (random, vocab-bounded) so prefix caching
cannot contaminate the ladder; max_new_tokens=1 isolates prefill.
"""

import argparse
import json
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.request


def _post(url, path, payload, timeout=1200.0):
    req = urllib.request.Request(
        url + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _prefill_draw(url, n_tokens, vocab, rng):
    ids = [rng.randrange(1000, vocab) for _ in range(n_tokens)]
    t0 = time.perf_counter()
    _post(url, "/generate", {
        "input_ids": ids,
        "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
    })
    return (time.perf_counter() - t0) * 1000.0


def cmd_prefill_ladder(args):
    rng = random.Random(args.seed)
    out = {"url": args.url, "lengths": {}, "floor_pct": {}}
    for n in args.lengths:
        _prefill_draw(args.url, n, args.vocab, rng)  # warm-up, discarded
        a = [_prefill_draw(args.url, n, args.vocab, rng) for _ in range(args.draws)]
        b = [_prefill_draw(args.url, n, args.vocab, rng) for _ in range(args.draws)]
        med_a, med_b = statistics.median(a), statistics.median(b)
        floor = abs(med_a - med_b) / max(med_a, med_b) * 100.0
        out["lengths"][str(n)] = {
            "draws_ms": a + b,
            "median_ms": statistics.median(a + b),
            "tok_per_s": n / (statistics.median(a + b) / 1000.0),
        }
        out["floor_pct"][str(n)] = round(floor, 3)
        print(
            f"{n} tok: median {statistics.median(a + b):.1f} ms, "
            f"A-vs-A floor {floor:.2f} %"
        )
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")
    return 0


def cmd_decode(args):
    rng = random.Random(args.seed)
    ids = [rng.randrange(1000, args.vocab) for _ in range(args.prompt_tokens)]
    t0 = time.perf_counter()
    r = _post(args.url, "/generate", {
        "input_ids": ids,
        "sampling_params": {
            "temperature": 0.7,
            "max_new_tokens": args.max_new,
            "ignore_eos": True,
        },
    })
    dt = time.perf_counter() - t0
    n = r.get("meta_info", {}).get("completion_tokens", args.max_new)
    out = {
        "completion_tokens": n,
        "wall_s": round(dt, 3),
        "tok_per_s": round(n / dt, 2),
    }
    print(json.dumps(out))
    json.dump(out, open(args.out, "w"), indent=1)
    return 0


# PHASE-FLIP DONE pp_to_tp (epoch 1) in 812.4 ms: 123 live slots, ...
# The "over N seam wave(s)" clause arrived with the waved seam (#631) and
# is OPTIONAL here on purpose: archived logs from before it are still the
# comparison baseline for every flip-time row in the bench, and a parser
# that silently skipped them would turn a format change into a missing
# regression rather than a visible one.
_FLIP_RE = re.compile(
    r"PHASE-FLIP DONE (?P<dir>\w+) \(epoch (?P<epoch>\d+)\) in "
    r"(?P<total>[\d.]+) ms(?: over (?P<waves>\d+) seam wave\(s\))?: "
    r"(?P<slots>\d+) live slots.*?"
    r"read (?P<read>[\d.]+) ms, exchange (?P<xchg>[\d.]+) ms, "
    r"write (?P<write>[\d.]+) ms",
    re.S,
)
_GDN_RE = re.compile(
    r"PHASE-FLIP-GDN moved (?P<slots>\d+) slot\(s\) (?P<dir>\w+): "
    r"sent (?P<sent>[\d.]+) MiB, received (?P<recv>[\d.]+) MiB"
)
_RANK_RE = re.compile(r"\b(?:PP|TP|DP)?(\d)\b")


def parse_flip_stats(lines):
    """Offline log parse -> list of flip records (per rank when the log
    prefix carries one; the per-rank split is the doctrine: the binding
    rank must be identifiable, never averaged away)."""
    records = []
    for line in lines:
        m = _FLIP_RE.search(line)
        if m:
            records.append(
                {
                    "kind": "kv",
                    "direction": m.group("dir"),
                    "epoch": int(m.group("epoch")),
                    "total_ms": float(m.group("total")),
                    "read_ms": float(m.group("read")),
                    "exchange_ms": float(m.group("xchg")),
                    "write_ms": float(m.group("write")),
                    "live_slots": int(m.group("slots")),
                    # None for pre-wave logs, which is a truthful "the run
                    # predates the split" rather than a fabricated 1.
                    "seam_waves": (
                        int(m.group("waves")) if m.group("waves") else None
                    ),
                    "line": line.strip()[:240],
                }
            )
            continue
        m = _GDN_RE.search(line)
        if m:
            records.append(
                {
                    "kind": "gdn",
                    "direction": m.group("dir"),
                    "slots": int(m.group("slots")),
                    "sent_mib": float(m.group("sent")),
                    "received_mib": float(m.group("recv")),
                    "line": line.strip()[:240],
                }
            )
    return records


def cmd_flip_stats(args):
    with open(args.log, errors="replace") as f:
        records = parse_flip_stats(f)
    json.dump(records, open(args.out, "w"), indent=1)
    kv = [r for r in records if r["kind"] == "kv"]
    print(
        f"parsed {len(records)} flip records ({len(kv)} kv) from "
        f"{args.log} -> {args.out}"
    )
    for r in kv:
        print(
            f"  {r['direction']} epoch {r['epoch']}: total {r['total_ms']} "
            f"ms (read {r['read_ms']} / exchange {r['exchange_ms']} / "
            f"write {r['write_ms']}), {r['live_slots']} slots"
        )
    return 0


def cmd_corridor(args):
    cmd = [
        sys.executable,
        "/spinning/vram_corridor_sampler.py",
        "--interval-ms",
        "100",
        "--duration-s",
        str(args.duration_s),
        "--out",
        args.out,
    ]
    print("exec:", " ".join(cmd))
    return subprocess.call(cmd)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prefill-ladder")
    p.add_argument("--url", default="http://127.0.0.1:30023")
    p.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192, 32768])
    p.add_argument("--draws", type=int, default=3)
    p.add_argument("--vocab", type=int, default=100000)
    p.add_argument("--seed", type=int, default=631)
    p.add_argument("--out", default="prefill_ladder.json")
    p.set_defaults(fn=cmd_prefill_ladder)

    p = sub.add_parser("decode")
    p.add_argument("--url", default="http://127.0.0.1:30023")
    p.add_argument("--prompt-tokens", type=int, default=256)
    p.add_argument("--max-new", type=int, default=512)
    p.add_argument("--vocab", type=int, default=100000)
    p.add_argument("--seed", type=int, default=631)
    p.add_argument("--out", default="decode.json")
    p.set_defaults(fn=cmd_decode)

    p = sub.add_parser("flip-stats")
    p.add_argument("--log", required=True)
    p.add_argument("--out", default="flip_stats.json")
    p.set_defaults(fn=cmd_flip_stats)

    p = sub.add_parser("corridor")
    p.add_argument("--duration-s", type=float, default=120.0)
    p.add_argument("--out", default="corridor.json")
    p.set_defaults(fn=cmd_corridor)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
