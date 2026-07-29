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

KIND = "bar1_prefill_kurve"
SCHEMA_VERSION = 1

# The baseline over the host path, from MESSUNG_PREFILL_ANTEIL.md and
# 02_WAS_ERREICHT_IST.md: "1190/1097/1144/1105/1122 tok/s" over 1, 2, 4, 8 and
# 16 sessions. The three published points are 1190,7 / 1143,7 / 1122,4 at 1, 4
# and 16; 1097 and 1105 are the two that were filled in afterwards, which fixes
# the mapping of the five numbers to the five session counts.
GRUNDLINIE_BEKANNT = {1: 1190.7, 2: 1097.0, 4: 1143.7, 8: 1105.0, 16: 1122.4}
GRUNDLINIE_QUELLE = (
    "MESSUNG_PREFILL_ANTEIL.md / 02_WAS_ERREICHT_IST.md (Host-Weg, NCCL, "
    "Qwen3.6-27B-FP8 tp3, Rauschboden 0,47 %)"
)

ARME = ("bar1", "grundlinie")
DECODE_BATCHES = (1, 16)

PROSA = (
    "Erklaere in ruhigem, sachlichem Ton, wie ein Rechner mehrere Aufgaben "
    "gleichzeitig bearbeitet, und woran man merkt, dass er dabei an eine "
    "Grenze stoesst."
)

_WORTE = (
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
    return "nicht aufgerufen"


def _prompt(unique: str, target_tokens: int) -> str:
    """Unique marker FIRST, then deterministic filler.

    Deterministic so both arms see byte-identical prompts for the same
    (session, round); unique so the radix cache cannot serve any of it."""
    rng = random.Random(unique)
    words = int(target_tokens / 1.3)
    body = " ".join(rng.choice(_WORTE) for _ in range(words))
    return f"[{unique}] {body}"


# ---------------------------------------------------------------------------
# measuring one point
# ---------------------------------------------------------------------------


def messe_prefill(
    port: int, arm: str, sessions: int, seconds: float, target_tokens: int
) -> dict:
    rows: list = []
    lock = threading.Lock()
    stop_at = time.monotonic() + seconds
    fehler: list = []

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
                    fehler.append(f"{type(exc).__name__}: {exc}")
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

    gute = [r for r in rows if isinstance(r.get("prompt_tokens"), int)]
    if not gute:
        return {
            "requests": 0,
            "prefill_tok_s": None,
            "fehler": fehler[:3] or ["keine Antwort mit prompt_tokens"],
        }
    erste = min(r["start"] for r in gute)
    letzte = max(r["ende"] for r in gute)
    wall = letzte - erste
    tokens = sum(r["prompt_tokens"] for r in gute)
    latenzen = sorted((r["ende"] - r["start"]) * 1000.0 for r in gute)
    return {
        "requests": len(gute),
        "prompt_tokens_total": tokens,
        "prompt_tokens_mean": tokens / len(gute),
        "wall_s": wall,
        "prefill_tok_s": (tokens / wall) if wall > 0 else None,
        "latenz_ms_p50": statistics.median(latenzen),
        "latenz_ms_max": latenzen[-1],
        "fehler": fehler[:3],
        "roh": gute,
    }


def messe_decode(port: int, batch: int, seconds: float) -> dict:
    results: list = []
    lock = threading.Lock()
    deadline = time.monotonic() + seconds

    def worker() -> None:
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": PROSA}],
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

    gute = [r for r in results if r.get("chunks")]
    if not gute:
        return {
            "batch": batch,
            "decode_tok_s": None,
            "fehler": [r.get("fehler") for r in results][:3] or ["keine Stromantwort"],
        }
    erster = min(r["chunks"][0] for r in gute)
    letzter = max(r["chunks"][-1] for r in gute)
    tokens = sum(len(r["chunks"]) - 1 for r in gute)
    span = letzter - erster
    rate = (tokens / span) if span > 0 and tokens > 0 else None

    # One non-streaming request for the accept length: meta_info rides on the
    # complete response, and spec_accept_length is the only honest accept
    # number (never spec_ema_accept_len from the log).
    accept = None
    sample = gute[0]["text"][:400]
    try:
        answer = _post(
            port,
            "/v1/chat/completions",
            {
                "model": "default",
                "messages": [{"role": "user", "content": PROSA}],
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
        sample = sample or f"(keine Antwort: {type(exc).__name__}: {exc})"

    return {
        "batch": batch,
        "requests": len(gute),
        "tokens": tokens,
        "span_s": span,
        "decode_tok_s": rate,
        "ms_pro_token": (1000.0 / rate) if rate else None,
        "ms_pro_verify": (1000.0 / rate * accept) if rate and accept else None,
        "spec_accept_length": accept,
        "output_sample": sample,
        "fehler": [r.get("fehler") for r in results if r.get("fehler")][:3],
    }


def modus_messen(args) -> int:
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    flush = _flush_cache(args.port)
    if args.warmup_seconds > 0:
        messe_prefill(
            args.port, args.arm, args.sessions, args.warmup_seconds, args.prompt_tokens
        )
        _flush_cache(args.port)

    prefill = messe_prefill(
        args.port, args.arm, args.sessions, args.point_seconds, args.prompt_tokens
    )
    roh = prefill.pop("roh", [])

    decode = []
    if args.with_decode:
        for batch in DECODE_BATCHES:
            decode.append(messe_decode(args.port, batch, args.point_seconds))

    punkt = {
        "folge": args.folge,
        "arm": args.arm,
        "sessions": args.sessions,
        "zeit": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "flush_cache": flush,
        "point_seconds": args.point_seconds,
        "prompt_tokens_ziel": args.prompt_tokens,
        "inhalt": "fuellprosa, je Anfrage eindeutiger Kopf",
        "prefill": prefill,
        "decode": decode,
    }
    with open(os.path.join(out_dir, "punkte.jsonl"), "a") as f:
        f.write(json.dumps(punkt) + "\n")
    with open(os.path.join(out_dir, f"roh_{args.arm}_{args.sessions}.jsonl"), "w") as f:
        for row in roh:
            f.write(json.dumps(row) + "\n")
    print(
        f"punkt {args.arm}/{args.sessions}: "
        f"{prefill.get('prefill_tok_s')} tok/s aus {prefill.get('requests')} Anfragen"
    )
    # A point without a rate is not a point. Returning 0 here would let the
    # orchestrator keep booting arms against a server that answers nothing.
    if prefill.get("prefill_tok_s") is None:
        for err in prefill.get("fehler") or []:
            print(f"  Fehler: {err}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# summary and the live table
# ---------------------------------------------------------------------------


def lade_punkte(step_dir: str) -> list:
    path = os.path.join(step_dir, "punkte.jsonl")
    punkte: list = []
    if not os.path.exists(path):
        return punkte
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                punkte.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    punkte.sort(key=lambda p: p.get("folge", 0))
    return punkte


def lade_beleg(step_dir: str, folge, arm, sessions) -> dict:
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
                        "gruppe": m.group("gruppe"),
                        "angefordert": m.group("angefordert"),
                        "erreicht": m.group("erreicht"),
                    }
                )
    return out


def zusammenfassen(step_dir: str, tol_pct: float, plan: list) -> dict:
    punkte = lade_punkte(step_dir)
    kurven: dict = {arm: {} for arm in ARME}
    decode_punkte: list = []
    reihenfolge = []
    for p in punkte:
        arm = p.get("arm")
        sessions = p.get("sessions")
        rate = (p.get("prefill") or {}).get("prefill_tok_s")
        if arm in kurven and isinstance(sessions, int):
            kurven[arm][sessions] = rate
        eintrag = {
            "folge": p.get("folge"),
            "arm": arm,
            "sessions": sessions,
            "zeit": p.get("zeit"),
            "prefill_tok_s": rate,
        }
        eintrag.update(lade_beleg(step_dir, p.get("folge"), arm, sessions))
        reihenfolge.append(eintrag)
        for d in p.get("decode") or []:
            entry = dict(d)
            entry["arm"] = arm
            entry["sessions_des_boots"] = sessions
            decode_punkte.append(entry)

    abweichung = {}
    for sessions, rate in kurven.get("grundlinie", {}).items():
        bekannt = GRUNDLINIE_BEKANNT.get(sessions)
        if bekannt and isinstance(rate, (int, float)):
            abweichung[str(sessions)] = (rate - bekannt) / bekannt * 100.0

    verhaeltnis = {}
    for sessions, rate in kurven.get("bar1", {}).items():
        base = kurven.get("grundlinie", {}).get(sessions)
        if base and isinstance(rate, (int, float)) and base > 0:
            verhaeltnis[str(sessions)] = rate / base

    def _kurz(name: str):
        pfad = os.path.join(step_dir, name)
        if not os.path.exists(pfad):
            return None
        with open(pfad, errors="replace") as f:
            return " ".join(f.read().split())[:200] or "ohne Grund"

    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "arme": list(ARME),
        "sessions_geplant": plan,
        "abbruch": _kurz("abbruch.txt"),
        # A step that never got the cards must not be diagnosed through the
        # empty artifacts it left behind.
        "blockiert": _kurz("blocked.txt"),
        "host_erreichbar": not os.path.exists(
            os.path.join(step_dir, "host_unreachable.txt")
        ),
        "integration_vorhanden": not os.path.exists(
            os.path.join(step_dir, "integration_missing.txt")
        ),
        "punkte": len(punkte),
        "reihenfolge": reihenfolge,
        "kurven": {
            arm: {str(k): v for k, v in sorted(d.items())} for arm, d in kurven.items()
        },
        "decode": decode_punkte,
        "grundlinie_bekannt": {str(k): v for k, v in GRUNDLINIE_BEKANNT.items()},
        "grundlinie_quelle": GRUNDLINIE_QUELLE,
        "grundlinie_abweichung_pct": abweichung,
        "toleranz_pct": tol_pct,
        "verhaeltnis_bar1_zu_grundlinie": verhaeltnis,
        "output_samples": {
            arm: next(
                (
                    d.get("output_sample")
                    for d in decode_punkte
                    if d.get("arm") == arm and d.get("output_sample")
                ),
                None,
            )
            for arm in ARME
        },
        "rohdaten": (
            sorted(f for f in os.listdir(step_dir) if f.startswith("roh_"))
            if os.path.isdir(step_dir)
            else []
        ),
    }


def tabelle(payload: dict) -> str:
    """The live table, rendered from the persisted points. Pure presentation:
    no marking, no comparison with an expectation, no verdict."""
    lines = [
        "| sessions | bar1 tok/s | grundlinie tok/s | bar1/grundlinie |",
        "|---:|---:|---:|---:|",
    ]
    sessions = sorted({int(s) for arm in ARME for s in payload["kurven"].get(arm, {})})
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
        lines += [
            "",
            "| arm | bs | decode tok/s | ms/Token | accept |",
            "|---|---:|---:|---:|---:|",
        ]
        for d in dec[-8:]:
            lines.append(
                f"| {d.get('arm')} | {d.get('batch')} "
                f"| {d.get('decode_tok_s') if d.get('decode_tok_s') is None else format(d['decode_tok_s'], '.1f')} "
                f"| {d.get('ms_pro_token') if d.get('ms_pro_token') is None else format(d['ms_pro_token'], '.2f')} "
                f"| {d.get('spec_accept_length')} |"
            )
    return "\n".join(lines) + "\n"


def modus_zusammenfassen(args) -> int:
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
    ap.add_argument("--arm", choices=ARME, default="bar1")
    ap.add_argument("--sessions", type=int, default=1)
    ap.add_argument("--folge", type=int, default=0)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--step-dir", default=".")
    ap.add_argument("--point-seconds", type=float, default=15.0)
    ap.add_argument("--warmup-seconds", type=float, default=8.0)
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--with-decode", type=int, default=1)
    ap.add_argument(
        "--tol-pct",
        type=float,
        default=float(os.environ.get("S12_BASELINE_TOL_PCT", "5")),
    )
    ap.add_argument(
        "--sessions-plan",
        default=os.environ.get("S12_SESSIONS", "1 4 8 16"),
        help="welche Sessionzahlen der Lauf abfahren SOLL -- der Check misst "
        "Vollstaendigkeit dagegen",
    )
    args = ap.parse_args()
    if args.point_seconds > 60:
        print("point-seconds > 60 s -- gegen die Zeitbox-Regel", file=sys.stderr)
        return 2
    if args.mode == "messen":
        return modus_messen(args)
    return modus_zusammenfassen(args)


if __name__ == "__main__":
    sys.exit(main())
