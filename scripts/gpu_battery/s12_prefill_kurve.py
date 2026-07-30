#!/usr/bin/env python3
"""s12 -- the multi-session prefill curve, BAR1 against the baseline.

Two modes, because the measurement runs where the server runs (the PVE host,
standard library only, no venv) while the summary runs in the container next to
the rest of the battery:

  --mode messen           one point against a LIVE server: prefill throughput
                          at N sessions, plus one decode point at bs=1 and one
                          at bs=16. Appends to punkte.jsonl.
  --mode zusammenfassen   punkte.jsonl -> prefill_kurve.json + the live table.

Why the measurement looks the way it does:

* THE NUMERATOR IS WHAT THE SERVER COUNTED. prompt_tokens from the response,
  never an estimate from the prompt string. The tokenizer is the authority on
  how many tokens went in.
* EVERY PROMPT IS UNIQUE, at the FRONT. A shared prefix would be served out of
  the radix cache and the second round would measure the cache, not the
  prefill. /flush_cache is called on top.
* WARMUP BEFORE EVERY POINT. 726 us against 95 us purely from the P-state ramp
  is the measured cost of skipping it (05_FALLEN, rule 4).
* TIME-BOXED, 10-20 s per point, and the raw per-request rows are persisted
  rather than only the aggregate (rule 6).
* ONE CONTENT CLASS. The content axis belongs to the accept boots (s02-s05);
  here every arm sees byte-identical prompts, because the comparison is between
  transports and content variance would be noise in it.

What this file does NOT do: judge. Whether the curve stays flat or starts to
rise is the finding the whole exercise is for, and it is written down, not
graded.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from s11_bar1_e2e import RE_GROUP  # noqa: E402  one regex, one place
from s12_log_analyse import (  # noqa: E402  one parser, one place
    decode_tick_aggregat,
    im_fenster,
    parse_decode,
)

KIND = "bar1_prefill_kurve"
#: 2: the per-point transport evidence under `gruppen` renamed its keys from
#: German to English (`gruppe`/`angefordert`/`erreicht` ->
#: `group`/`requested`/`achieved`), in step with s11_bar1_e2e.RE_GROUP and
#: htccl.py's report_state(). A schema-1 artifact spells them in German, so
#: every point would read back as having no HTCCL group at all -- which the
#: arm check would misread as "baseline". Rejecting it by version is the
#: point: re-run the step rather than read a stale artifact.
SCHEMA_VERSION = 2

# The baseline over the host path, from MESSUNG_PREFILL_ANTEIL.md and
# 02_WAS_ERREICHT_IST.md: "1190/1097/1144/1105/1122 tok/s" over 1, 2, 4, 8 and
# 16 sessions. The three published points are 1190,7 / 1143,7 / 1122,4 at 1, 4
# and 16; 1097 and 1105 are the two that were filled in afterwards, which fixes
# the mapping of the five numbers to the five session counts.
BASELINE_KNOWN = {1: 1190.7, 2: 1097.0, 4: 1143.7, 8: 1105.0, 16: 1122.4}
BASELINE_SOURCE = (
    "MESSUNG_PREFILL_ANTEIL.md / 02_WAS_ERREICHT_IST.md (host path, NCCL, "
    "Qwen3.6-27B-FP8 tp3, noise floor 0.47 %)"
)

ARMS = ("bar1", "grundlinie")
DECODE_BATCHES = (1, 16)


def _decode_batches(args) -> tuple:
    """Decode batch sizes for this point.

    The default stays (1, 16) so every earlier step measures byte-identically;
    --decode-batches lets a caller ask for the batch sizes ITS question is
    about. s15 asks for 1 and 8, because its prefill points are 1 and 8
    sessions and a phase comparison that reads prefill at 8 and decode at 16 is
    comparing two different load points.
    """
    raw = getattr(args, "decode_batches", None)
    if not raw:
        return DECODE_BATCHES
    return tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())

# The prompt and the filler corpus stay GERMAN on purpose: they are
# measurement input, not prose. Both arms have to see byte-identical prompts,
# and every recorded output_sample of every past run was generated from
# exactly these bytes -- translating them would silently change the workload.
PROSE = (
    "Erklaere in ruhigem, sachlichem Ton, wie ein Rechner mehrere Aufgaben "
    "gleichzeitig bearbeitet, und woran man merkt, dass er dabei an eine "
    "Grenze stoesst."
)

_WORDS = (
    "die messung zeigt einen deutlichen unterschied zwischen der erwartung und "
    "dem befund auf dieser maschine denn der weg ueber den hauptspeicher kostet "
    "zeit die niemand sieht solange die karten wenig zu tun haben und erst unter "
    "last wird aus dem umweg ein deckel der sich in jeder zahl niederschlaegt"
).split()


def _post(port: int, path: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode(errors="replace"))


def _post_stream(port: int, payload: dict, timeout: float, deadline: float) -> dict:
    """Returns {"chunks": [monotonic timestamps], "text": str}."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    chunks = []
    text_parts = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            if time.monotonic() > deadline:
                break
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if not body or body == "[DONE]":
                continue
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                continue
            try:
                delta = obj["choices"][0].get("delta") or {}
            except (KeyError, IndexError, TypeError):
                continue
            piece = delta.get("content")
            if piece:
                chunks.append(time.monotonic())
                text_parts.append(piece)
    return {"chunks": chunks, "text": "".join(text_parts)}


def _flush_cache(port: int) -> str:
    for path in ("/flush_cache",):
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}{path}", method="POST"
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return f"{path}:{resp.status}"
        except (urllib.error.URLError, OSError) as exc:
            return f"{path}:{exc}"
    return "not called"


def _prompt(unique: str, target_tokens: int) -> str:
    """Unique marker FIRST, then deterministic filler.

    Deterministic so both arms see byte-identical prompts for the same
    (session, round); unique so the radix cache cannot serve any of it."""
    rng = random.Random(unique)
    words = int(target_tokens / 1.3)
    body = " ".join(rng.choice(_WORDS) for _ in range(words))
    return f"[{unique}] {body}"


# ---------------------------------------------------------------------------
# measuring one point
# ---------------------------------------------------------------------------


def measure_prefill(
    port: int, arm: str, sessions: int, seconds: float, target_tokens: int
) -> dict:
    rows: list = []
    lock = threading.Lock()
    stop_at = time.monotonic() + seconds
    errors: list = []

    def worker(slot: int) -> None:
        i = 0
        while time.monotonic() < stop_at:
            i += 1
            unique = f"{arm[:1]}{sessions}-{slot}-{i}-{int(stop_at)}"
            payload = {
                "model": "default",
                "messages": [
                    {"role": "user", "content": _prompt(unique, target_tokens)}
                ],
                "max_tokens": 1,
                "temperature": 0,
            }
            t0 = time.monotonic()
            try:
                answer = _post(port, "/v1/chat/completions", payload, timeout=180)
            except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
                with lock:
                    errors.append(f"{type(exc).__name__}: {exc}")
                return
            t1 = time.monotonic()
            usage = answer.get("usage") or {}
            with lock:
                rows.append(
                    {
                        "slot": slot,
                        "runde": i,
                        "start": t0,
                        "ende": t1,
                        "prompt_tokens": usage.get("prompt_tokens"),
                    }
                )

    threads = [threading.Thread(target=worker, args=(s,)) for s in range(sessions)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=seconds + 240)

    good = [r for r in rows if isinstance(r.get("prompt_tokens"), int)]
    if not good:
        return {
            "requests": 0,
            "prefill_tok_s": None,
            "fehler": errors[:3] or ["no answer carrying prompt_tokens"],
        }
    first = min(r["start"] for r in good)
    last = max(r["ende"] for r in good)
    wall = last - first
    tokens = sum(r["prompt_tokens"] for r in good)
    latencies = sorted((r["ende"] - r["start"]) * 1000.0 for r in good)
    return {
        "requests": len(good),
        "prompt_tokens_total": tokens,
        "prompt_tokens_mean": tokens / len(good),
        "wall_s": wall,
        "prefill_tok_s": (tokens / wall) if wall > 0 else None,
        "latenz_ms_p50": statistics.median(latencies),
        "latenz_ms_max": latencies[-1],
        "fehler": errors[:3],
        "roh": good,
    }


def ernte_ticks(server_log: str, start: float, end: float, batch: int) -> dict:
    """The scheduler's own decode ticks for the window a decode point ran in.

    Why this exists at all. The request level and the tick level answer two
    different questions and the 2026-07-30 run only ever recorded the first:

    * ACCEPT was None in every one of the eight points. `meta_info` is opt-in on
      the chat endpoint (`protocol.py` `return_meta_info: bool = False`), and
      the accept request never asked for it. The scheduler logs `accept len` on
      every tick no matter who asked.
    * THE RATE was the request level's: tokens of a stream over the wall clock
      of that stream, prefill and queue included. That reported 32 tok/s at
      bs=1 where the decode loop was turning over 77-83.

    Neither number replaces the other, so both are kept and both are labelled.
    """
    out = {"tick_quelle": server_log or None}
    if not server_log or not os.path.exists(server_log):
        out["tick_fehler"] = "kein Serverlog"
        return out
    try:
        with open(server_log, errors="replace") as f:
            ticks = parse_decode(f)
    except OSError as exc:
        out["tick_fehler"] = f"{type(exc).__name__}: {exc}"
        return out
    # Prefixed, because `ms_pro_token` exists on both levels and an unprefixed
    # merge would silently replace the request-level number with the tick one --
    # exactly the conflation this fix is about.
    agg = decode_tick_aggregat(im_fenster(ticks, start, end), batch)
    out.update({f"tick_{k}": v for k, v in agg.items()})
    return out


def measure_decode(port: int, batch: int, seconds: float) -> dict:
    results: list = []
    lock = threading.Lock()
    deadline = time.monotonic() + seconds

    def worker() -> None:
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": PROSE}],
            "max_tokens": 256,
            "temperature": 0,
            "stream": True,
        }
        try:
            out = _post_stream(port, payload, timeout=180, deadline=deadline)
        except (urllib.error.URLError, OSError) as exc:
            with lock:
                results.append({"fehler": f"{type(exc).__name__}: {exc}"})
            return
        with lock:
            results.append(out)

    threads = [threading.Thread(target=worker) for _ in range(batch)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=seconds + 240)

    good = [r for r in results if r.get("chunks")]
    if not good:
        return {
            "batch": batch,
            "decode_tok_s": None,
            "fehler": [r.get("fehler") for r in results][:3] or ["no streamed answer"],
        }
    first = min(r["chunks"][0] for r in good)
    last = max(r["chunks"][-1] for r in good)
    tokens = sum(len(r["chunks"]) - 1 for r in good)
    span = last - first
    rate = (tokens / span) if span > 0 and tokens > 0 else None

    # One non-streaming request for the accept length: meta_info rides on the
    # complete response, and spec_accept_length is the only honest accept
    # number (never spec_ema_accept_len from the log).
    accept = None
    sample = good[0]["text"][:400]
    try:
        answer = _post(
            port,
            "/v1/chat/completions",
            {
                "model": "default",
                "messages": [{"role": "user", "content": PROSE}],
                "max_tokens": 64,
                "temperature": 0,
            },
            timeout=180,
        )
        choice = (answer.get("choices") or [{}])[0]
        accept = (choice.get("meta_info") or {}).get("spec_accept_length")
        if not sample:
            sample = ((choice.get("message") or {}).get("content") or "")[:400]
    except (urllib.error.URLError, OSError, json.JSONDecodeError, IndexError) as exc:
        accept = None
        sample = sample or f"(no answer: {type(exc).__name__}: {exc})"

    return {
        "batch": batch,
        "requests": len(good),
        "tokens": tokens,
        "span_s": span,
        "decode_tok_s": rate,
        "ms_pro_token": (1000.0 / rate) if rate else None,
        "ms_pro_verify": (1000.0 / rate * accept) if rate and accept else None,
        "spec_accept_length": accept,
        "output_sample": sample,
        "fehler": [r.get("fehler") for r in results if r.get("fehler")][:3],
    }


def mode_measure(args) -> int:
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    flush = _flush_cache(args.port)
    if args.warmup_seconds > 0:
        measure_prefill(
            args.port, args.arm, args.sessions, args.warmup_seconds, args.prompt_tokens
        )
        _flush_cache(args.port)

    prefill = measure_prefill(
        args.port, args.arm, args.sessions, args.point_seconds, args.prompt_tokens
    )
    raw = prefill.pop("roh", [])

    decode = []
    if args.with_decode:
        for batch in _decode_batches(args):
            start = time.time()
            point_decode = measure_decode(args.port, batch, args.point_seconds)
            point_decode.update(ernte_ticks(args.server_log, start, time.time(), batch))
            decode.append(point_decode)

    point = {
        "folge": args.folge,
        "arm": args.arm,
        "sessions": args.sessions,
        "zeit": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "flush_cache": flush,
        "point_seconds": args.point_seconds,
        "prompt_tokens_ziel": args.prompt_tokens,
        "inhalt": "filler prose, unique head per request",
        "prefill": prefill,
        "decode": decode,
    }
    with open(os.path.join(out_dir, "punkte.jsonl"), "a") as f:
        f.write(json.dumps(point) + "\n")
    with open(os.path.join(out_dir, f"roh_{args.arm}_{args.sessions}.jsonl"), "w") as f:
        for row in raw:
            f.write(json.dumps(row) + "\n")
    print(
        f"point {args.arm}/{args.sessions}: "
        f"{prefill.get('prefill_tok_s')} tok/s from "
        f"{prefill.get('requests')} requests"
    )
    # A point without a rate is not a point. Returning 0 here would let the
    # orchestrator keep booting arms against a server that answers nothing.
    if prefill.get("prefill_tok_s") is None:
        for err in prefill.get("fehler") or []:
            print(f"  error: {err}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# summary and the live table
# ---------------------------------------------------------------------------


def load_points(step_dir: str) -> list:
    path = os.path.join(step_dir, "punkte.jsonl")
    points: list = []
    if not os.path.exists(path):
        return points
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                points.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    points.sort(key=lambda p: p.get("folge", 0))
    return points


def load_evidence(step_dir: str, folge, arm, sessions) -> dict:
    """What the boot behind a point REALLY ran.

    The transport name in the log says bar1 either way -- that cost a whole
    measurement once already. So each boot leaves the ERREICHT lines behind and
    the point carries them: a bar1 point whose dcp group fell back to gloo is
    not a bar1 point, and a baseline point with any HTCCL group is not a
    baseline point.
    """
    path = os.path.join(step_dir, "belege", f"{folge}_{arm}_{sessions}.txt")
    out: dict = {"beleg_vorhanden": os.path.exists(path), "gruppen": []}
    if not out["beleg_vorhanden"]:
        return out
    with open(path, errors="replace") as f:
        for line in f:
            m = RE_GROUP.search(line)
            if m:
                out["gruppen"].append(
                    {
                        "group": m.group("group"),
                        "requested": m.group("requested"),
                        "achieved": m.group("achieved"),
                    }
                )
    return out


def load_fatal(step_dir: str, arm, sessions) -> dict:
    """The fatal harvest of ONE boot.

    s12_prefill_kurve.sh greps every boot's host log for OOM / NCCL error /
    traceback into logs/<arm>_<n>.fatal.txt. Nothing read that file, so eight
    boots that each died in a prefill OOM could still hand in throughput
    numbers and pass -- the only step in the battery without a fatal gate.
    An EMPTY file is the healthy case: the grep found nothing.

    Lines the server QUOTED from a helper subprocess it caught and recovered
    from (the stage-0 hardware probe) are skipped: a traceback in there is
    evidence about the helper, not about this boot -- see
    check_common.QUOTED_SUBLOG_PREFIX, keep the literals in step.
    """
    quoted_prefix = "[probe-subprocess] "
    name = f"logs/{arm}_{sessions}.fatal.txt"
    path = os.path.join(step_dir, name)
    if not os.path.exists(path):
        return {"fatal_erhoben": False, "fatal": None}
    with open(path, errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            if line.strip() and quoted_prefix not in line:
                return {
                    "fatal_erhoben": True,
                    "fatal": f"{name}:{lineno}: {' '.join(line.split())[:200]}",
                }
    return {"fatal_erhoben": True, "fatal": None}


def zusammenfassen(step_dir: str, tol_pct: float, plan: list) -> dict:
    points = load_points(step_dir)
    curves: dict = {arm: {} for arm in ARMS}
    decode_points: list = []
    order = []
    for p in points:
        arm = p.get("arm")
        sessions = p.get("sessions")
        rate = (p.get("prefill") or {}).get("prefill_tok_s")
        if arm in curves and isinstance(sessions, int):
            curves[arm][sessions] = rate
        record = {
            "folge": p.get("folge"),
            "arm": arm,
            "sessions": sessions,
            "zeit": p.get("zeit"),
            "prefill_tok_s": rate,
        }
        record.update(load_evidence(step_dir, p.get("folge"), arm, sessions))
        record.update(load_fatal(step_dir, arm, sessions))
        order.append(record)
        for d in p.get("decode") or []:
            entry = dict(d)
            entry["arm"] = arm
            entry["sessions_des_boots"] = sessions
            decode_points.append(entry)

    deviation = {}
    for sessions, rate in curves.get("grundlinie", {}).items():
        known = BASELINE_KNOWN.get(sessions)
        if known and isinstance(rate, (int, float)):
            deviation[str(sessions)] = (rate - known) / known * 100.0

    ratio = {}
    for sessions, rate in curves.get("bar1", {}).items():
        base = curves.get("grundlinie", {}).get(sessions)
        if base and isinstance(rate, (int, float)) and base > 0:
            ratio[str(sessions)] = rate / base

    def _short(name: str):
        path = os.path.join(step_dir, name)
        if not os.path.exists(path):
            return None
        with open(path, errors="replace") as f:
            return " ".join(f.read().split())[:200] or "no reason given"

    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "arme": list(ARMS),
        "sessions_geplant": plan,
        "abbruch": _short("abbruch.txt"),
        # A step that never got the cards must not be diagnosed through the
        # empty artifacts it left behind.
        "blockiert": _short("blocked.txt"),
        "host_erreichbar": not os.path.exists(
            os.path.join(step_dir, "host_unreachable.txt")
        ),
        "integration_vorhanden": not os.path.exists(
            os.path.join(step_dir, "integration_missing.txt")
        ),
        "punkte": len(points),
        "reihenfolge": order,
        # One list, so the check does not have to walk the boots to find out
        # whether any of them died. Empty means every harvest came back clean.
        "fatal": [
            {
                "folge": e.get("folge"),
                "arm": e.get("arm"),
                "sessions": e.get("sessions"),
                "zeile": e["fatal"],
            }
            for e in order
            if e.get("fatal")
        ],
        "fatal_ungeprueft": [
            {
                "folge": e.get("folge"),
                "arm": e.get("arm"),
                "sessions": e.get("sessions"),
            }
            for e in order
            if not e.get("fatal_erhoben")
        ],
        "kurven": {
            arm: {str(k): v for k, v in sorted(d.items())} for arm, d in curves.items()
        },
        "decode": decode_points,
        "grundlinie_bekannt": {str(k): v for k, v in BASELINE_KNOWN.items()},
        "grundlinie_quelle": BASELINE_SOURCE,
        "grundlinie_abweichung_pct": deviation,
        "toleranz_pct": tol_pct,
        "verhaeltnis_bar1_zu_grundlinie": ratio,
        "output_samples": {
            arm: next(
                (
                    d.get("output_sample")
                    for d in decode_points
                    if d.get("arm") == arm and d.get("output_sample")
                ),
                None,
            )
            for arm in ARMS
        },
        "rohdaten": (
            sorted(f for f in os.listdir(step_dir) if f.startswith("roh_"))
            if os.path.isdir(step_dir)
            else []
        ),
    }


def _num(v, nk: int) -> str:
    return "None" if not isinstance(v, (int, float)) else format(v, f".{nk}f")


def tabelle(payload: dict) -> str:
    """The live table, rendered from the persisted points. Pure presentation:
    no marking, no comparison with an expectation, no verdict."""
    lines = [
        "| sessions | bar1 tok/s | grundlinie tok/s | bar1/grundlinie |",
        "|---:|---:|---:|---:|",
    ]
    sessions = sorted({int(s) for arm in ARMS for s in payload["kurven"].get(arm, {})})
    for s in sessions:
        a = payload["kurven"].get("bar1", {}).get(str(s))
        b = payload["kurven"].get("grundlinie", {}).get(str(s))
        r = payload["verhaeltnis_bar1_zu_grundlinie"].get(str(s))
        lines.append(
            f"| {s} | {a if a is None else format(a, '.1f')} "
            f"| {b if b is None else format(b, '.1f')} "
            f"| {r if r is None else format(r, '.3f')} |"
        )
    dec = payload.get("decode") or []
    if dec:
        # Two levels, both named. The tick columns are what the decode loop
        # turns over; the request column is tokens over the wall clock of a
        # whole stream and therefore carries prefill and queue with it. They
        # differ by more than a factor of two, so an unlabelled "decode tok/s"
        # column is not a number anybody can use.
        #
        # The column labels stay verbatim: test_s12_log_analyse.py asserts
        # "tok/s (Tick)" and "tok/s (Anfrage)", and "ms/Token" is the name
        # s12 defines for the whole battery (s14 re-reads it bit-for-bit).
        lines += [
            "",
            "| arm | bs | tok/s (Tick) | ms/Token (Tick) | accept (Tick) "
            "| Ticks | tok/s (Anfrage) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for d in dec[-8:]:
            lines.append(
                f"| {d.get('arm')} | {d.get('batch')} "
                f"| {_num(d.get('tick_gen_tok_s_median'), 1)} "
                f"| {_num(d.get('tick_ms_pro_token'), 2)} "
                f"| {_num(d.get('tick_accept_len_median'), 2)} "
                f"| {_num(d.get('tick_ticks_gewertet'), 0)} "
                f"| {_num(d.get('decode_tok_s'), 1)} |"
            )
    return "\n".join(lines) + "\n"


def mode_summarize(args) -> int:
    os.makedirs(args.step_dir, exist_ok=True)
    plan = [int(x) for x in str(args.sessions_plan).replace(",", " ").split()]
    payload = zusammenfassen(args.step_dir, args.tol_pct, plan)
    with open(os.path.join(args.step_dir, "prefill_kurve.json"), "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    text = tabelle(payload)
    with open(os.path.join(args.step_dir, "zwischentabelle.md"), "w") as f:
        f.write(text)
    print(text, end="")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("messen", "zusammenfassen"), required=True)
    ap.add_argument("--port", type=int, default=30030)
    # Free string, not `choices=ARME`. s12 itself only ever passes the two
    # names in ARME, and its summary still keys off ARME -- but the same
    # measurement driver is what #293 step 2 (s13) points at seven arms with,
    # and a closed choice list there would mean a second copy of this file.
    ap.add_argument("--arm", default="bar1")
    ap.add_argument("--sessions", type=int, default=1)
    ap.add_argument("--folge", type=int, default=0)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--step-dir", default=".")
    ap.add_argument("--point-seconds", type=float, default=15.0)
    ap.add_argument("--warmup-seconds", type=float, default=8.0)
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--with-decode", type=int, default=1)
    ap.add_argument(
        "--decode-batches",
        default="",
        help="Comma-separated decode batch sizes; empty keeps the default 1,16.",
    )
    ap.add_argument(
        "--server-log",
        default="",
        help="path of the server log ON THE HOST -- accept len and the decode "
        "tick rate come from there, neither is available at request level",
    )
    ap.add_argument(
        "--tol-pct",
        type=float,
        default=float(os.environ.get("S12_BASELINE_TOL_PCT", "5")),
    )
    ap.add_argument(
        "--sessions-plan",
        default=os.environ.get("S12_SESSIONS", "1 4 8 16"),
        help="which session counts the run is SUPPOSED to walk -- the check "
        "measures completeness against this",
    )
    args = ap.parse_args()
    if args.point_seconds > 60:
        print("point-seconds > 60 s -- against the time-box rule", file=sys.stderr)
        return 2
    if args.mode == "messen":
        return mode_measure(args)
    return mode_summarize(args)


if __name__ == "__main__":
    sys.exit(main())
