#!/usr/bin/env python3
"""s12 -- compute/wait analysis out of the server logs a run already left.

Task #293, step 1. The s12 curve measured a bar1/baseline ratio that falls from
1,143 at one session to 0,997 at eight, and the open question was WHY: a size
effect (larger, differently shaped collectives at higher concurrency) or
contention (the same collectives getting slower because several ranks push
through the same BAR1 path at once). The logs of that run carry enough to
decide it without booting a card again:

* ``Prefill rank batch`` lines carry the CollectiveClock split introduced in
  #252 -- ``gpu-ms: X (compute Y, wait Z)`` per rank. Collective time is
  charged to ``wait``, arithmetic to ``compute``, so a transport that gets
  slower shows up in ``wait`` and nowhere else.
* the same lines carry ``#new-token``, which IS the collective size: one
  all_reduce per layer collective over ``new_token x hidden x 2`` bytes.
* the ``HTCCL-BAR1: Aufbau`` line carries the slot geometry, so the round
  decomposition of a given payload is arithmetic, not a guess.

Nothing here judges. It parses, aggregates and prints; the verdict is written
by hand into docs/dev/INTEGRATION_R3_VALIDATION.md, where it can be argued
with.

Two entry points:

  --log arm:sessions:path   (repeatable) the server logs to read
  --punkte punkte.jsonl     the run's points, for the request count per point

The request count is what separates the measured window from the warmup: the
harness runs ``warmup_seconds`` of the identical load, then flushes, then runs
the point. Both phases emit the same lines, and the only honest boundary is
"the last N batches before the decode phase", with N from the point itself.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import statistics
import sys

# The lines the server really writes.
#   scheduler.py           -- "Prefill rank batch, ..." (the #252 split)
#   scheduler.py           -- "Decode batch, ..."
#   htccl_bar1.py:1751     -- "HTCCL-BAR1: Aufbau in ..."
RE_PREFILL_RANK = re.compile(
    r"\[(?P<zeit>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) TP(?P<rang>\d+)\] "
    r"Prefill rank batch, #new-token: (?P<new_token>\d+), "
    r"#cached-token: (?P<cached_token>\d+), #chunks: (?P<chunks>\d+), "
    r"gpu-ms: (?P<gpu_ms>[\d.]+) "
    r"\(compute (?P<compute_ms>[\d.]+), wait (?P<wait_ms>[\d.]+)\)"
)
RE_DECODE = re.compile(
    r"\[(?P<zeit>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) TP(?P<rang>\d+)\] "
    r"Decode batch, #running-req: (?P<running_req>\d+),.*?"
    r"accept len: (?P<accept_len>[\d.]+), accept rate: (?P<accept_rate>[\d.]+), "
    r"cuda graph: (?P<cuda_graph>\w+), "
    r"gen throughput \(token/s\): (?P<gen_tok_s>[\d.]+)"
)
RE_BAR1_SETUP = re.compile(
    r"HTCCL-BAR1: Aufbau in (?P<aufbau_ms>[\d.]+) ms, (?P<peers>\d+) Peer-Ziele, "
    r"Region (?P<region_mib>[\d.]+) MiB je Rang \((?P<schlitze>\d+) Schlitze"
    r".*?\), Schlitz (?P<schlitz_kib>\d+) KiB, "
    r"groesste Nutzlast (?P<max_nutzlast_kib>\d+) KiB"
)

# Ticks below this generation rate are the first tick of a stream, where the
# rate still carries the prefill of that request. Keeping them would drag the
# median of a 77 tok/s decode down to 30.
WARMUP_TICK_TOK_S = 20.0

# Batches at least this large are the measured prefill work. Below it are the
# chunk remainders of a prompt that did not fit one chunk (a 2055-token prompt
# at chunked_prefill_size=2048 leaves a 7-token batch) and the short prompts of
# the decode phase.
LARGE_BATCH_TOKEN = 1000


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def _numbers(m: re.Match, integers: tuple, decimals: tuple = ()) -> dict:
    d = m.groupdict()
    for k in integers:
        d[k] = int(d[k])
    for k in decimals:
        d[k] = float(d[k])
    return d


def parse_prefill_rang(lines) -> list:
    """Every ``Prefill rank batch`` line, in file order, one dict per rank."""
    out = []
    for line in lines:
        m = RE_PREFILL_RANK.search(line)
        if m:
            out.append(
                _numbers(
                    m,
                    ("rang", "new_token", "cached_token", "chunks"),
                    ("gpu_ms", "compute_ms", "wait_ms"),
                )
            )
    return out


def parse_decode(lines) -> list:
    """Every ``Decode batch`` line. One per scheduler tick, not per request."""
    out = []
    for line in lines:
        m = RE_DECODE.search(line)
        if m:
            d = _numbers(
                m,
                ("rang", "running_req"),
                ("accept_len", "accept_rate", "gen_tok_s"),
            )
            d["cuda_graph"] = d["cuda_graph"] == "True"
            out.append(d)
    return out


def parse_bar1_geometrie(lines) -> dict | None:
    """The BAR1 slot geometry of the FIRST Aufbau line, or None.

    First and not last on purpose: a boot builds one region per communicator
    group, and the group that carries the prefill all_reduce is the widest one.
    """
    for line in lines:
        m = RE_BAR1_SETUP.search(line)
        if m:
            return _numbers(
                m,
                ("peers", "schlitze", "schlitz_kib", "max_nutzlast_kib"),
                ("aufbau_ms", "region_mib"),
            )
    return None


# ---------------------------------------------------------------------------
# the collective behind a batch
# ---------------------------------------------------------------------------


def kollektiv_bytes(new_token: int, hidden: int, elem_bytes: int = 2) -> int:
    """Bytes of ONE tensor-parallel all_reduce for a batch of that size.

    The reduction after attention output projection and after the MLP down
    projection each run over the full ``[token, hidden]`` activation. Layer
    count multiplies the COUNT of collectives per batch, not the size of one,
    so it does not belong in here.
    """
    return new_token * hidden * elem_bytes


def runden(payload_bytes: int, slot_bytes: int, welt: int) -> int:
    """Rounds a payload needs, given one slot per peer.

    BAR1 splits an all_reduce into ``welt`` equal shards (reduce-scatter, then
    all-gather); the shard has to fit a slot. ``max_nutzlast = welt * slot``
    -- the field name the Aufbau line uses -- is where the shard stops
    fitting in ONE pass.

    That used to be the end of the path: above it ``handles()`` said False and
    the payload left over the base transport. Since ``ar_plan`` (the
    all_reduce/all_to_all multi-round work merged as f51d414959) it is not --
    the payload is cut into this many rounds and stays on the direct path. A
    value > 1 therefore means "more rounds", not "not on this path at all".
    """
    if slot_bytes <= 0 or welt <= 0:
        return 0
    shard = -(-payload_bytes // welt)
    return -(-shard // slot_bytes)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def punkt_fenster(
    rows: list, requests: int, min_new_token: int = LARGE_BATCH_TOKEN
) -> dict:
    """The last ``requests`` large batches per rank -- the measured point.

    The warmup runs the identical load and logs identically, so no line says
    which phase it belongs to. What IS known is how many requests the point
    counted, and at chunked_prefill_size=2048 against ~2048-token prompts one
    request is one large batch. Counting from the back is therefore exact for
    the point and drops the warmup, which is what the counting is for.
    """
    per_rank: dict = {}
    for r in rows:
        if r["new_token"] >= min_new_token:
            per_rank.setdefault(r["rang"], []).append(r)
    if requests > 0:
        per_rank = {k: v[-requests:] for k, v in per_rank.items()}
    return per_rank


def wait_aggregat(batches: list) -> dict:
    """compute / wait of one rank over one window. Medians plus the share.

    The share is built from SUMS, not from the median of the per-batch shares:
    what the curve is about is where the seconds of a point went, and a median
    of ratios does not add up to a point.
    """
    if not batches:
        return {"n": 0}
    c = [b["compute_ms"] for b in batches]
    w = [b["wait_ms"] for b in batches]
    g = [b["gpu_ms"] for b in batches]
    return {
        "n": len(batches),
        "compute_ms_median": statistics.median(c),
        "wait_ms_median": statistics.median(w),
        "gpu_ms_median": statistics.median(g),
        "wait_anteil": sum(w) / sum(g) if sum(g) else None,
        "new_token_median": statistics.median(b["new_token"] for b in batches),
        "new_token_max": max(b["new_token"] for b in batches),
    }


def im_fenster(rows: list, start: float, end: float) -> list:
    """Rows whose log timestamp falls into [start, end], both epoch seconds.

    The scheduler stamps whole LOCAL seconds, so the window is widened by one
    second on each side: a tick logged at 13:44:59,7 prints 13:44:59 and would
    otherwise fall out of a window that opened at 13:44:59,2.
    """
    a = datetime.datetime.fromtimestamp(start - 1.0)
    b = datetime.datetime.fromtimestamp(end + 1.0)
    out = []
    for r in rows:
        t = datetime.datetime.strptime(r["zeit"], "%Y-%m-%d %H:%M:%S")
        if a <= t <= b:
            out.append(r)
    return out


def belegung(batches: list) -> dict:
    """Sum of per-batch gpu-ms against the wall clock the batches span.

    Over 100 % is not a rounding error, it is the finding: batches that
    together claim more device time than the window lasted must have OVERLAPPED
    in time. Once they do, a per-batch ``gpu-ms`` is no longer that batch's
    latency but the pipeline's period, and everything the period costs lands in
    ``wait``. The log stamps whole seconds, so the span is coarse and the
    quotient is a magnitude, not a measurement.
    """
    if len(batches) < 2:
        return {"batches": len(batches)}
    t = [datetime.datetime.strptime(b["zeit"], "%Y-%m-%d %H:%M:%S") for b in batches]
    span = (max(t) - min(t)).total_seconds()
    total = sum(b["gpu_ms"] for b in batches) / 1000.0
    return {
        "batches": len(batches),
        "spanne_s": span,
        "gpu_s_summe": total,
        "belegung": (total / span) if span > 0 else None,
    }


def decode_tick_aggregat(
    ticks: list, running_req: int | None = None, min_tok_s: float = WARMUP_TICK_TOK_S
) -> dict:
    """Accept length and generation rate per TICK, not per request.

    Two things the request-level numbers cannot give:

    * ACCEPT. ``meta_info`` is opt-in on the chat endpoint, so the harness read
      None from every point of the 2026-07-30 run. The scheduler logs the
      accept length of every tick regardless of which endpoint asked.
    * THE RATE. A request-level rate divides tokens by the wall clock of the
      whole stream, which includes the prefill and the queue. The tick rate is
      what the decode loop actually turns over, and the two differ by more than
      a factor of two on this rig.
    """
    r = [t for t in ticks if running_req is None or t["running_req"] == running_req]
    warm = [t for t in r if t["gen_tok_s"] >= min_tok_s]
    if not warm:
        return {"ticks": len(r), "ticks_gewertet": 0}
    rate = statistics.median(t["gen_tok_s"] for t in warm)
    accept = statistics.median(t["accept_len"] for t in warm)
    return {
        "ticks": len(r),
        "ticks_gewertet": len(warm),
        "ticks_warmup_verworfen": len(r) - len(warm),
        "accept_len_median": accept,
        "gen_tok_s_median": rate,
        "ms_pro_token": 1000.0 / rate if rate else None,
        "ms_pro_verify": 1000.0 / rate * accept if rate and accept else None,
        "cuda_graph": all(t["cuda_graph"] for t in warm),
    }


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------


def lade_punkte(path: str) -> dict:
    """(arm, sessions) -> the point, from punkte.jsonl."""
    out: dict = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[(p.get("arm"), p.get("sessions"))] = p
    return out


def auswerten(sources: list, points: dict, hidden: int, welt: int) -> dict:
    """sources: [(arm, sessions, path)] -> one payload with every table in it."""
    lines_per_point = {}
    geo = None
    for arm, sessions, path in sources:
        with open(path, errors="replace") as f:
            lines = f.readlines()
        lines_per_point[(arm, sessions)] = lines
        geo = geo or parse_bar1_geometrie(lines)

    wait: list = []
    sizes: list = []
    decode: list = []
    for (arm, sessions), lines in sorted(lines_per_point.items()):
        point = points.get((arm, sessions)) or {}
        requests = ((point.get("prefill") or {}).get("requests")) or 0
        batches = parse_prefill_rang(lines)
        window = punkt_fenster(batches, requests)
        for rank in sorted(window):
            agg = wait_aggregat(window[rank])
            agg.update({"arm": arm, "sessions": sessions, "rang": rank})
            wait.append(agg)
        all_tp0 = [b for b in batches if b["rang"] == 0]
        large = [b for b in window.get(0, [])]
        if large:
            nt_max = max(b["new_token"] for b in large)
            sizes.append(
                {
                    "arm": arm,
                    "sessions": sessions,
                    "batches": len(large),
                    "new_token_median": statistics.median(
                        b["new_token"] for b in large
                    ),
                    "new_token_max": nt_max,
                    "chunks_max": max(b["chunks"] for b in all_tp0),
                    "kleine_batches": len(
                        [b for b in all_tp0 if b["new_token"] < 1000]
                    ),
                    "nutzlast_bytes": kollektiv_bytes(nt_max, hidden),
                }
            )
            sizes[-1].update(belegung(large))
        ticks = parse_decode(lines)
        for bs in sorted({t["running_req"] for t in ticks}):
            d = decode_tick_aggregat(ticks, bs)
            d.update({"arm": arm, "sessions": sessions, "running_req": bs})
            decode.append(d)

    slot_bytes = (geo or {}).get("schlitz_kib", 0) * 1024
    max_payload = (geo or {}).get("max_nutzlast_kib", 0) * 1024
    for g in sizes:
        g["runden"] = runden(g["nutzlast_bytes"], slot_bytes, welt)
        # "Does the direct path carry this payload at all" -- and since
        # ar_plan that question is answered by the ROUND PLAN, not by the
        # single-pass ceiling. The old predicate (`nutzlast <= max_nutzlast`)
        # was written when there was no second round, and it now reports the
        # opposite of what the runtime does: the #293 lever measurement ran a
        # 4096-token batch (41,9 MB against a 25,2 MB ceiling) whose log
        # contains not one fallback line, while this field said "nein".
        # The runtime's own honesty rule is the cross-check -- every fallback
        # is reported by `warum_nicht()`, and there were none.
        g["traegt_bar1"] = g["runden"] >= 1 if slot_bytes else None
        # Kept because it is the number the tipping point is derived from,
        # and it is what changes when the pipe takes its share of the window.
        g["einrundig"] = (
            g["nutzlast_bytes"] <= max_payload if max_payload else None
        )
    return {
        "hidden": hidden,
        "welt": welt,
        "bar1_geometrie": geo,
        "kipp_token": (max_payload // (hidden * 2)) if max_payload else None,
        "wait": wait,
        "groessen": sizes,
        "decode": decode,
    }


def _f(v, nk=1):
    return "-" if v is None else format(v, f".{nk}f")


def tabellen(payload: dict) -> str:
    # The heading is kept verbatim: test_s12_log_analyse.py asserts the
    # substring "compute/wait je Rang" against this script's stdout, and the
    # validation doc quotes the section by that name.
    lines = ["### compute/wait je Rang (measurement window of the point)", ""]
    lines += [
        "| Arm | Sessions | Rank | Batches | compute ms | wait ms | gpu-ms "
        "| wait share |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for w in payload["wait"]:
        if not w.get("n"):
            continue
        lines.append(
            f"| {w['arm']} | {w['sessions']} | TP{w['rang']} | {w['n']} "
            f"| {_f(w['compute_ms_median'])} | {_f(w['wait_ms_median'])} "
            f"| {_f(w['gpu_ms_median'])} "
            f"| {_f(w['wait_anteil'] * 100)} % |"
        )
    lines += ["", "### Chunk sizes and collective sizes", ""]
    lines += [
        "| Arm | Sessions | Large batches | #new-token median | #new-token max "
        "| #chunks max | all_reduce bytes | Rounds | on BAR1 | Occupancy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for g in payload["groessen"]:
        occ = g.get("belegung")
        lines.append(
            f"| {g['arm']} | {g['sessions']} | {g['batches']} "
            f"| {_f(g['new_token_median'], 0)} | {g['new_token_max']} "
            f"| {g['chunks_max']} | {g['nutzlast_bytes']} | {g['runden']} "
            f"| {'yes' if g['traegt_bar1'] else 'no'} "
            f"| {'-' if occ is None else format(occ * 100, '.0f') + ' %'} |"
        )
    lines += ["", "### Decode per tick (not per request)", ""]
    lines += [
        "| Arm | Sessions | bs | Ticks | counted | accept len | gen tok/s "
        "| ms/Token | ms/Verify |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for d in payload["decode"]:
        if not d.get("ticks_gewertet"):
            continue
        lines.append(
            f"| {d['arm']} | {d['sessions']} | {d['running_req']} | {d['ticks']} "
            f"| {d['ticks_gewertet']} | {_f(d['accept_len_median'], 2)} "
            f"| {_f(d['gen_tok_s_median'])} | {_f(d['ms_pro_token'], 2)} "
            f"| {_f(d['ms_pro_verify'], 2)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--log",
        action="append",
        default=[],
        metavar="ARM:SESSIONS:PFAD",
        help="server log of one point, repeatable",
    )
    ap.add_argument("--punkte", default="", help="punkte.jsonl of the run")
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--welt", type=int, default=3, help="TP ranks in the group")
    ap.add_argument("--json", default="", help="write the payload here as well")
    args = ap.parse_args()

    sources = []
    for spec in args.log:
        parts = spec.split(":", 2)
        if len(parts) != 3:
            print(f"--log needs ARM:SESSIONS:PFAD, got {spec!r}", file=sys.stderr)
            return 2
        sources.append((parts[0], int(parts[1]), parts[2]))
    if not sources:
        print("no --log given", file=sys.stderr)
        return 2

    payload = auswerten(sources, lade_punkte(args.punkte), args.hidden, args.welt)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
    print(tabellen(payload), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
