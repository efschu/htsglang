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
RE_PREFILL_RANG = re.compile(
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
RE_BAR1_AUFBAU = re.compile(
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
GROSSBATCH_TOKEN = 1000


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def _zahlen(m: re.Match, ganz: tuple, dezimal: tuple = ()) -> dict:
    d = m.groupdict()
    for k in ganz:
        d[k] = int(d[k])
    for k in dezimal:
        d[k] = float(d[k])
    return d


def parse_prefill_rang(zeilen) -> list:
    """Every ``Prefill rank batch`` line, in file order, one dict per rank."""
    out = []
    for zeile in zeilen:
        m = RE_PREFILL_RANG.search(zeile)
        if m:
            out.append(
                _zahlen(
                    m,
                    ("rang", "new_token", "cached_token", "chunks"),
                    ("gpu_ms", "compute_ms", "wait_ms"),
                )
            )
    return out


def parse_decode(zeilen) -> list:
    """Every ``Decode batch`` line. One per scheduler tick, not per request."""
    out = []
    for zeile in zeilen:
        m = RE_DECODE.search(zeile)
        if m:
            d = _zahlen(
                m,
                ("rang", "running_req"),
                ("accept_len", "accept_rate", "gen_tok_s"),
            )
            d["cuda_graph"] = d["cuda_graph"] == "True"
            out.append(d)
    return out


def parse_bar1_geometrie(zeilen) -> dict | None:
    """The BAR1 slot geometry of the FIRST Aufbau line, or None.

    First and not last on purpose: a boot builds one region per communicator
    group, and the group that carries the prefill all_reduce is the widest one.
    """
    for zeile in zeilen:
        m = RE_BAR1_AUFBAU.search(zeile)
        if m:
            return _zahlen(
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


def runden(nutzlast_bytes: int, schlitz_bytes: int, welt: int) -> int:
    """Rounds a payload needs, given one slot per peer.

    BAR1 splits an all_reduce into ``welt`` equal shards (reduce-scatter, then
    all-gather); the shard has to fit a slot. ``max_nutzlast = welt *
    schlitz`` is exactly the point where it stops fitting -- and above it
    ``handles()`` says False and the payload leaves over the base transport,
    so a value > 1 here is not "slower", it is "not on this path at all".
    """
    if schlitz_bytes <= 0 or welt <= 0:
        return 0
    scherbe = -(-nutzlast_bytes // welt)
    return -(-scherbe // schlitz_bytes)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def punkt_fenster(
    rows: list, requests: int, min_new_token: int = GROSSBATCH_TOKEN
) -> dict:
    """The last ``requests`` large batches per rank -- the measured point.

    The warmup runs the identical load and logs identically, so no line says
    which phase it belongs to. What IS known is how many requests the point
    counted, and at chunked_prefill_size=2048 against ~2048-token prompts one
    request is one large batch. Counting from the back is therefore exact for
    the point and drops the warmup, which is what the counting is for.
    """
    je_rang: dict = {}
    for r in rows:
        if r["new_token"] >= min_new_token:
            je_rang.setdefault(r["rang"], []).append(r)
    if requests > 0:
        je_rang = {k: v[-requests:] for k, v in je_rang.items()}
    return je_rang


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


def im_fenster(rows: list, von: float, bis: float) -> list:
    """Rows whose log timestamp falls into [von, bis], both epoch seconds.

    The scheduler stamps whole LOCAL seconds, so the window is widened by one
    second on each side: a tick logged at 13:44:59,7 prints 13:44:59 and would
    otherwise fall out of a window that opened at 13:44:59,2.
    """
    a = datetime.datetime.fromtimestamp(von - 1.0)
    b = datetime.datetime.fromtimestamp(bis + 1.0)
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
    spanne = (max(t) - min(t)).total_seconds()
    summe = sum(b["gpu_ms"] for b in batches) / 1000.0
    return {
        "batches": len(batches),
        "spanne_s": spanne,
        "gpu_s_summe": summe,
        "belegung": (summe / spanne) if spanne > 0 else None,
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


def lade_punkte(pfad: str) -> dict:
    """(arm, sessions) -> the point, from punkte.jsonl."""
    out: dict = {}
    if not pfad or not os.path.exists(pfad):
        return out
    with open(pfad, errors="replace") as f:
        for zeile in f:
            zeile = zeile.strip()
            if not zeile:
                continue
            try:
                p = json.loads(zeile)
            except json.JSONDecodeError:
                continue
            out[(p.get("arm"), p.get("sessions"))] = p
    return out


def auswerten(quellen: list, punkte: dict, hidden: int, welt: int) -> dict:
    """quellen: [(arm, sessions, pfad)] -> one payload with every table in it."""
    zeilen_je_punkt = {}
    geo = None
    for arm, sessions, pfad in quellen:
        with open(pfad, errors="replace") as f:
            zeilen = f.readlines()
        zeilen_je_punkt[(arm, sessions)] = zeilen
        geo = geo or parse_bar1_geometrie(zeilen)

    wait: list = []
    groessen: list = []
    decode: list = []
    for (arm, sessions), zeilen in sorted(zeilen_je_punkt.items()):
        punkt = punkte.get((arm, sessions)) or {}
        requests = ((punkt.get("prefill") or {}).get("requests")) or 0
        batches = parse_prefill_rang(zeilen)
        fenster = punkt_fenster(batches, requests)
        for rang in sorted(fenster):
            agg = wait_aggregat(fenster[rang])
            agg.update({"arm": arm, "sessions": sessions, "rang": rang})
            wait.append(agg)
        alle_tp0 = [b for b in batches if b["rang"] == 0]
        gross = [b for b in fenster.get(0, [])]
        if gross:
            nt_max = max(b["new_token"] for b in gross)
            groessen.append(
                {
                    "arm": arm,
                    "sessions": sessions,
                    "batches": len(gross),
                    "new_token_median": statistics.median(
                        b["new_token"] for b in gross
                    ),
                    "new_token_max": nt_max,
                    "chunks_max": max(b["chunks"] for b in alle_tp0),
                    "kleine_batches": len(
                        [b for b in alle_tp0 if b["new_token"] < 1000]
                    ),
                    "nutzlast_bytes": kollektiv_bytes(nt_max, hidden),
                }
            )
            groessen[-1].update(belegung(gross))
        ticks = parse_decode(zeilen)
        for bs in sorted({t["running_req"] for t in ticks}):
            d = decode_tick_aggregat(ticks, bs)
            d.update({"arm": arm, "sessions": sessions, "running_req": bs})
            decode.append(d)

    schlitz_bytes = (geo or {}).get("schlitz_kib", 0) * 1024
    max_nutzlast = (geo or {}).get("max_nutzlast_kib", 0) * 1024
    for g in groessen:
        g["runden"] = runden(g["nutzlast_bytes"], schlitz_bytes, welt)
        g["traegt_bar1"] = g["nutzlast_bytes"] <= max_nutzlast if max_nutzlast else None
    return {
        "hidden": hidden,
        "welt": welt,
        "bar1_geometrie": geo,
        "kipp_token": (max_nutzlast // (hidden * 2)) if max_nutzlast else None,
        "wait": wait,
        "groessen": groessen,
        "decode": decode,
    }


def _f(v, nk=1):
    return "-" if v is None else format(v, f".{nk}f")


def tabellen(payload: dict) -> str:
    zeilen = ["### compute/wait je Rang (Messfenster des Punktes)", ""]
    zeilen += [
        "| Arm | Sessions | Rang | Batches | compute ms | wait ms | gpu-ms | wait-Anteil |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for w in payload["wait"]:
        if not w.get("n"):
            continue
        zeilen.append(
            f"| {w['arm']} | {w['sessions']} | TP{w['rang']} | {w['n']} "
            f"| {_f(w['compute_ms_median'])} | {_f(w['wait_ms_median'])} "
            f"| {_f(w['gpu_ms_median'])} "
            f"| {_f(w['wait_anteil'] * 100)} % |"
        )
    zeilen += ["", "### Chunkgroessen und Kollektivgroessen", ""]
    zeilen += [
        "| Arm | Sessions | Grossbatches | #new-token Median | #new-token max "
        "| #chunks max | all_reduce Byte | Runden | traegt BAR1 | Belegung |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for g in payload["groessen"]:
        bel = g.get("belegung")
        zeilen.append(
            f"| {g['arm']} | {g['sessions']} | {g['batches']} "
            f"| {_f(g['new_token_median'], 0)} | {g['new_token_max']} "
            f"| {g['chunks_max']} | {g['nutzlast_bytes']} | {g['runden']} "
            f"| {'ja' if g['traegt_bar1'] else 'nein'} "
            f"| {'-' if bel is None else format(bel * 100, '.0f') + ' %'} |"
        )
    zeilen += ["", "### Decode je Tick (nicht je Anfrage)", ""]
    zeilen += [
        "| Arm | Sessions | bs | Ticks | gewertet | accept len | gen tok/s "
        "| ms/Token | ms/Verify |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for d in payload["decode"]:
        if not d.get("ticks_gewertet"):
            continue
        zeilen.append(
            f"| {d['arm']} | {d['sessions']} | {d['running_req']} | {d['ticks']} "
            f"| {d['ticks_gewertet']} | {_f(d['accept_len_median'], 2)} "
            f"| {_f(d['gen_tok_s_median'])} | {_f(d['ms_pro_token'], 2)} "
            f"| {_f(d['ms_pro_verify'], 2)} |"
        )
    return "\n".join(zeilen) + "\n"


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

    quellen = []
    for spec in args.log:
        teile = spec.split(":", 2)
        if len(teile) != 3:
            print(f"--log braucht ARM:SESSIONS:PFAD, bekam {spec!r}", file=sys.stderr)
            return 2
        quellen.append((teile[0], int(teile[1]), teile[2]))
    if not quellen:
        print("kein --log angegeben", file=sys.stderr)
        return 2

    payload = auswerten(quellen, lade_punkte(args.punkte), args.hidden, args.welt)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
    print(tabellen(payload), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
