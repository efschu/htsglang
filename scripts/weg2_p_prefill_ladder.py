#!/usr/bin/env python3
"""Weg-2 variant of the #631 Route A prefill ladder (scripts/route_a_631_prefill_ladder.py
on feat/route-a-631): the SAME rungs, draws and median, but the prefill time is read
SERVER-SIDE, so the two phase flips a Weg-2 request costs never enter the number.

Why the Route A script cannot be pointed at the Weg-2 front unchanged
--------------------------------------------------------------------
* The front routes on a CHARACTER estimate of the prompt text
  (``front.request_text`` / ``price_remainder``, CHARS_PER_TOKEN=3.0). A payload that
  carries only ``input_ids`` has an empty text, prices at ~1 token and is routed
  SHORT -> D: the Route A draw would measure D's single prefill, not P.
* A LONG request is served in two legs: leg 1 on P with ``max_new_tokens`` forced to 1
  (``WeG2Front.leg1``), then -- after the P->D flip -- leg 2 on D with the client's own
  ``max_new_tokens``. P never answers the client itself, even for ``max_new_tokens=1``;
  the client wall therefore contains D->P and P->D flips and D's 1-token decode.

What this script does instead
-----------------------------
* sends TEXT prompts (unique 64-hex preamble + random long words, so no draw shares a
  prefix with another and the character estimate routes LONG), ``max_new_tokens=1``;
* sizes each rung by calibrating chars/token on the warm-up draw (P's exact
  ``prompt_tokens`` comes back in the response);
* per draw reads the rid (``meta_info.id``; the front stamps ``payload["rid"]``) and
  takes from the logs:
    - front log ``WEG2-ROUTE rid=<rid> SHORT|LONG`` -> which group served it,
    - front log ``WEG2-SERVED group=P leg=1 rid=<rid> ... wall=<s>`` -> the P leg wall
      (front POST -> P response; no flip, no D),
    - P log, per rank: the rid's chunks and the summed ``gpu-ms`` of its
      ``Prefill rank batch`` lines (compute-honest), and the ``#1463 PASS-TAIL
      ... finished=1 t=<epoch>`` stamp of the pass that finished the rid;
* reports per rung the median P wall, tok/s, per-rank gpu-ms sums and the pipeline
  efficiency ``max_rank_gpu / P wall``.

A rung whose rid was routed SHORT (served on D) is reported as such and carries no P
number -- with X=4096 that is what happens to small prompts unless their character
count prices them above X; the long-word generator exists to push a 2048-token rung
over that line, and the route line proves whether it did.

Stdlib only, like the Route A original.

Usage (the arm runs it after READY, one request at a time):
  python3 scripts/weg2_p_prefill_ladder.py --port 30030 \
      --p-log  <evidence>/boot_weg2_<TAG>_<sha>_<ts>.P.log \
      --front-log <evidence>/boot_weg2_<TAG>_<sha>_<ts>.front.log \
      --rungs 2048,8192,32768 --draws 3 --out <evidence>/ladder_<TAG>.json
"""

import argparse
import json
import os
import random
import re
import statistics
import time
import urllib.request

# Long, common words: ~6-9 characters per token on the Qwen tokenizer, i.e. 2-3x the
# front's CHARS_PER_TOKEN=3.0 estimate -- enough to price a 2048-token prompt above
# X=4096 so it takes the P route. Randomly drawn, so no two draws share a prefix.
_WORDS = (
    "understanding responsibility international development environmental "
    "information opportunity particularly organization relationship "
    "communication significant performance administration experience "
    "government investigation independent traditional professional "
    "construction application technology conversation temperature "
    "photograph individual management approximately description "
    "established recommendation throughout electricity considerable "
    "transportation neighborhood appropriate everything demonstration "
    "manufacturing agricultural competition institution requirement "
    "concentration intelligence representative philosophy mathematics "
    "consideration distribution legislation celebration accommodation "
    "architecture comprehensive contribution determination entertainment "
    "extraordinary headquarters illustration investment knowledge "
    "measurement observation population possibility preparation "
    "presentation publication registration satisfaction secretary "
    "significance supplementary television translation unfortunately "
    "vocabulary willingness wonderful yesterday advertisement"
).split()

_SERVED_RE = re.compile(
    r"WEG2-SERVED group=P leg=1 rid=(\S+) prompt_tokens=(\d+) cached_tokens=(\d+) "
    r"wall=([\d.]+)s"
)
_ROUTE_RE = re.compile(r"WEG2-ROUTE rid=(\S+) (SHORT|LONG|BATCH|CARRIER\S*)")
_RANK_RE = re.compile(r"^\[(\S+ \S+) PP(\d+)\] (.*)$")
_PRB_RE = re.compile(r"Prefill rank batch, #new-token: (\d+).*?gpu-ms: ([\d.]+)")
_TAIL_RE = re.compile(r"#1463 PASS-TAIL pp_rank=(\d+) bs=\d+ finished=(\d+) .* t=([\d.]+)")
_FRONT_TS_RE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3})\]")


def _prompt(n_words: int) -> str:
    pre = "%064x" % random.getrandbits(256)
    return pre + "\n" + " ".join(random.choice(_WORDS) for _ in range(n_words))


def _post(port: int, text: str, timeout: float) -> dict:
    body = json.dumps(
        {
            "text": text,
            "sampling_params": {"temperature": 0, "max_new_tokens": 1, "ignore_eos": True},
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        js = json.loads(r.read())
    js["_client_wall_s"] = time.perf_counter() - t0
    return js


def _read_front(path: str, rid: str):
    """(route, served) for one rid from the front log. ``served`` carries the P leg-1
    wall and the wall-clock instant the front logged it (the leg's END)."""
    route, served = None, None
    with open(path, errors="replace") as fh:
        for line in fh:
            if rid not in line:
                continue
            m = _ROUTE_RE.search(line)
            if m and m.group(1) == rid and m.group(2) in ("SHORT", "LONG"):
                route = m.group(2)
            m = _SERVED_RE.search(line)
            if m and m.group(1) == rid:
                ts = _FRONT_TS_RE.match(line)
                served = {
                    "prompt_tokens": int(m.group(2)),
                    "cached_tokens": int(m.group(3)),
                    "p_wall_s": float(m.group(4)),
                    "end_epoch": (
                        None if ts is None else
                        time.mktime(time.strptime(ts.group(1), "%Y-%m-%d %H:%M:%S"))
                        + int(ts.group(2)) / 1000.0
                    ),
                }
    return route, served


def _read_p(path: str, t0: float, t1: float) -> dict:
    """Per rank, the ``Prefill rank batch`` lines inside the P leg's own window
    [t0, t1] (front log: end instant minus leg wall). The P log stamps whole
    seconds, so the window is widened to whole seconds -- which is safe ONLY
    because the ladder sends one request at a time with ``--settle-s`` >= 2 s
    between draws and a Weg-2 flip separates consecutive P legs anyway."""
    lo, hi = int(t0), int(t1) + 1
    out = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            m = _RANK_RE.match(line)
            if not m:
                continue
            body = m.group(3)
            p = _PRB_RE.search(body)
            t = _TAIL_RE.search(body)
            if not p and not t:
                continue
            ts = time.mktime(time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
            if ts < lo or ts > hi:
                continue
            d = out.setdefault(int(m.group(2)), {"chunks": 0, "tokens": 0, "gpu_ms": 0.0,
                                                  "finished_t": None})
            if p:
                d["chunks"] += 1
                d["tokens"] += int(p.group(1))
                d["gpu_ms"] += float(p.group(2))
            elif int(t.group(2)) > 0:
                d["finished_t"] = float(t.group(3))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30030)
    ap.add_argument("--rungs", default="2048,8192,32768")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--p-log", required=True)
    ap.add_argument("--front-log", required=True)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--settle-s", type=float, default=3.0,
                    help="wait after each response before reading the logs")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # chars are what the front prices; words are what we generate. Start from a
    # guess and correct it on every warm-up draw.
    tokens_per_word = 2.0
    out = {
        "method": "text prompts (unique preamble + random long words), max_new_tokens=1, "
        "1 warm-up discarded, median of kept draws; P wall = front WEG2-SERVED leg-1 wall "
        "(no flip), gpu-ms = P log Prefill rank batch sums per rank",
        "rungs": [],
    }
    print(f"{'target':>7} {'route':>6} {'tokens':>7} {'P wall ms':>10} {'tok/s':>8} "
          f"{'gpu-ms PP0/PP1/PP2':>22} {'eff':>5}")
    for rung in [int(x) for x in args.rungs.split(",")]:
        draws = []
        for i in range(args.draws + 1):
            js = _post(args.port, _prompt(max(1, int(rung / tokens_per_word))), args.timeout)
            mi = js.get("meta_info") or {}
            rid = str(mi.get("id") or "")
            ptok = int(mi.get("prompt_tokens") or 0)
            n_words = max(1, int(rung / tokens_per_word))
            if ptok > 0:
                tokens_per_word = ptok / n_words
            time.sleep(args.settle_s)
            route, served = _read_front(args.front_log, rid) if rid else (None, None)
            per_rank = {}
            if served is not None and served.get("end_epoch"):
                per_rank = _read_p(args.p_log, served["end_epoch"] - served["p_wall_s"],
                                   served["end_epoch"])
            rec = {
                "rid": rid,
                "route": route or "?",
                "prompt_tokens": ptok,
                "client_wall_s": round(js["_client_wall_s"], 3),
                "p_wall_s": None if served is None else served["p_wall_s"],
                "cached_tokens": None if served is None else served["cached_tokens"],
                "gpu_ms": {r: round(v["gpu_ms"], 1) for r, v in sorted(per_rank.items())},
                "chunks": {r: v["chunks"] for r, v in sorted(per_rank.items())},
                "finished_t": {r: v["finished_t"] for r, v in sorted(per_rank.items())},
            }
            if i == 0:
                continue  # warm-up (also calibrates tokens_per_word), discarded
            draws.append(rec)
        p_walls = [d["p_wall_s"] for d in draws if d["p_wall_s"]]
        med = statistics.median(p_walls) if p_walls else None
        toks = statistics.median([d["prompt_tokens"] for d in draws]) if draws else 0
        gpu = {}
        for d in draws:
            for r, v in d["gpu_ms"].items():
                gpu.setdefault(r, []).append(v)
        gpu_med = {r: statistics.median(v) for r, v in sorted(gpu.items())}
        eff = (max(gpu_med.values()) / (med * 1000.0)) if (med and gpu_med) else None
        routes = sorted({d["route"] for d in draws})
        rung_rec = {
            "target_tokens": rung,
            "routes": routes,
            "prompt_tokens_median": toks,
            "p_wall_ms_median": None if med is None else round(med * 1000.0, 1),
            "p_tok_s": None if not med else round(toks / med, 1),
            "gpu_ms_median_per_rank": {str(r): round(v, 1) for r, v in gpu_med.items()},
            "pipeline_eff_max_rank_gpu_over_wall": None if eff is None else round(eff, 3),
            "draws": draws,
        }
        out["rungs"].append(rung_rec)
        gtxt = "/".join("%.0f" % gpu_med.get(r, 0.0) for r in (0, 1, 2))
        print(f"{rung:>7} {','.join(routes):>6} {toks:>7.0f} "
              f"{(med * 1000.0 if med else float('nan')):>10.1f} "
              f"{(toks / med if med else float('nan')):>8.1f} {gtxt:>22} "
              f"{(eff if eff is not None else float('nan')):>5.2f}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
