#!/usr/bin/env python3
"""Burst-Probe fuer group P (fnFL2 H37, Task #118): batcht P mehrere Requests?

Nutzer 24.09. 07:00Z: "16k-Chunking passt nicht zu kleinen Prefills, bei
Agentenlast kommen viele kleine Prefills". Bis x144 lief P nur mit EINEM
Request je Zeit (x127: 39/39 Prefill-Batches #new-seq 1). Diese Probe schickt N
Requests GLEICHZEITIG (threading.Barrier) an die Front und liest danach die
Zeilen, die P WAEHREND des Bursts geschrieben hat (Byte-Offset beim Start, nie
das ganze Log):

    [.. PP0] Prefill batch, #new-seq: k, #new-token: n, #cached-token: c, ...
    [.. PP0] Prefill rank batch, #new-token: n, ..., gpu-ms: g (compute .., wait ..)
    [.. PP0] FWD-TIMING-PREFILL forward=.. tokens=.. ... moe_fetch_ms=.. total_ms=..
                                   (nur mit SGLANG_WEG2_PREFILL_TIMING=1)

und aus dem Front-Log (optional, sonst aus dem P-Log-Pfad abgeleitet):

    WEG2-SERVED group=P leg=1 rid=.. prompt_tokens=N cached_tokens=C wall=Xs

JE REQUEST: TTFT (Senden -> erstes Text-Delta, Stream; enthaelt Warten, Flip,
P-Prefill, D-Extend), P-Wand (leg-1-Wand der Front, verbunden ueber
prompt_tokens -- die Laengen sind je Request verschieden), Route (P = leg 1 lief,
D = SHORT unter X, D prefillt selbst), Nadel MATCH/MISS (jeder Request traegt
seinen EIGENEN Code, eine vertauschte Antwort ist ein MISS).

AGGREGAT (PP0 ist der zulassende Rang, seine Zeilen zaehlen):
    batches            Prefill-Batches auf PP0 im Fenster
    multi_seq_batches  davon mit #new-seq > 1
    tok_per_fwd        #new-token-Summe / batches
    fwd_per_req        batches / P-Requests. Heutige Form (END-ANCHOR, 1 Request
                       je Forward): 2,0 (Koerper + 1-4-Token-Schwanz). Schwanz im
                       naechsten Koerper: ~1 + 1/N. ECHTES Batching: < 1,0.
    agg_tok_s          P-Tokens / P-Fenster (erste leg-1-Annahme bis letzte
                       leg-1-Antwort der Front; ohne Front-Log die Client-Wand)
    sockel_ms          RECHNUNG: Achsenabschnitt a der Geraden gpu_ms = a + b*n
                       ueber die PP0-Rangzeilen mit n >= 256 (braucht >= 2
                       verschiedene n); daneben moe_fetch_ms-Median aus
                       FWD-TIMING, wenn an.
    serial_tok_s_est   RECHNUNG: jeder Request allein (--p-bs 1): Summe ueber die
                       Raenge der Geraden + Schwanz-Median. Mit --mode serial
                       wird dasselbe GEMESSEN statt gerechnet.

Verdikt: `BURST-VERDICT batching=JA|SCHWANZ|NEIN ...` und `needles=k/N`.
Exit 0 = alle Nadeln MATCH und alle Requests 200; 2 = MISS; 3 = Fehler.

Nur Standardbibliothek: laeuft mit jedem python3 gegen die lebende Front.

    python3 nf_burst_probe.py --p-log <..>.P.log            # Burst, 8 Requests
    python3 nf_burst_probe.py --p-log <..>.P.log --mode serial --seed 38
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

#: Gemessen: 225 Saetze dieser Form + Frage = 3899 prompt_tokens (x144 KONTROLLE MID).
TOKENS_PER_SENTENCE = 17.1
PROMPT_OVERHEAD_TOKENS = 80
SENTENCE = "Satz {j}: Die Lagerhalle nummer {j} steht am Fluss."
QUESTION = ("Wie lautet der Geheimcode dieser Anfrage? Nenne zuerst nur den Code, "
            "danach in einem Satz, wovon der Text handelt.")
#: Unter diesem n ist eine Rangzeile ein END-ANCHOR-Schwanz (1-4 Token), kein Koerper.
TAIL_MAX_TOKENS = 4
FIT_MIN_TOKENS = 256


@dataclass
class BurstRequest:
    idx: int
    target_tokens: int
    code: str
    needle_pos: float
    body: dict


@dataclass
class RequestResult:
    idx: int
    target_tokens: int
    code: str
    status: int = 0
    error: str = ""
    t_send: float = 0.0
    t_first: Optional[float] = None
    t_done: Optional[float] = None
    text: str = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    # aus dem Front-Log
    route: str = "?"
    p_wall_s: Optional[float] = None
    p_cached_tokens: Optional[int] = None

    @property
    def ttft_s(self) -> Optional[float]:
        return None if self.t_first is None else self.t_first - self.t_send

    @property
    def total_s(self) -> Optional[float]:
        return None if self.t_done is None else self.t_done - self.t_send

    @property
    def needle(self) -> str:
        if self.status != 200:
            return "ERR"
        return "MATCH" if self.code in self.text else "MISS"


def build_requests(n: int, seed: int, min_tokens: int, max_tokens: int, model: str,
                   answer_tokens: int, nonce: str = "") -> List[BurstRequest]:
    """N Requests, deterministisch aus ``seed``: Laenge, Code und Nadelposition.

    Die erste Zeile traegt Seed und Index, damit kein Radix-Praefix zwischen den
    Requests greift (jeder ist ein eigener Prefill). Codes sind paarweise
    verschieden, Laengen (in Saetzen) ebenfalls -- die Front-Zeilen werden ueber
    prompt_tokens zugeordnet.
    """
    if n < 1 or min_tokens < 1 or max_tokens < min_tokens:
        raise ValueError("need n >= 1 and 1 <= min_tokens <= max_tokens")
    rng = random.Random(seed)
    out: List[BurstRequest] = []
    seen_codes = set()
    seen_sent = set()
    for i in range(n):
        target = rng.randint(min_tokens, max_tokens)
        n_sent = max(10, round((target - PROMPT_OVERHEAD_TOKENS) / TOKENS_PER_SENTENCE))
        while n_sent in seen_sent:
            n_sent += 1
        seen_sent.add(n_sent)
        while True:
            code = "%04d-%s%s" % (rng.randint(1000, 9999), chr(65 + rng.randint(0, 25)),
                                  chr(65 + rng.randint(0, 25)))
            if code not in seen_codes:
                seen_codes.add(code)
                break
        pos = round(rng.uniform(0.2, 0.8), 3)
        saetze = [SENTENCE.format(j=j) for j in range(1, n_sent)]
        saetze.insert(int(len(saetze) * pos), f"Merke dir den Geheimcode dieser Anfrage: {code}.")
        head = f"Burst {seed}{('-' + nonce) if nonce else ''} Anfrage {i + 1} von {n}. Lies genau.\n"
        content = head + " ".join(saetze) + "\n\n" + QUESTION
        body = {"model": model, "stream": True, "max_tokens": answer_tokens, "temperature": 0,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": content}]}
        out.append(BurstRequest(i, target, code, pos, body))
    return out


def _send(front: str, req: BurstRequest, res: RequestResult, timeout: float,
          barrier: Optional[threading.Barrier]) -> None:
    data = json.dumps(req.body).encode()
    http = urllib.request.Request(f"{front}/v1/chat/completions", data=data,
                                  headers={"Content-Type": "application/json"})
    if barrier is not None:
        barrier.wait()
    res.t_send = time.monotonic()
    parts: List[str] = []
    try:
        with urllib.request.urlopen(http, timeout=timeout) as resp:
            res.status = resp.status
            for raw in resp:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                if obj.get("usage"):
                    res.prompt_tokens = obj["usage"].get("prompt_tokens")
                    res.completion_tokens = obj["usage"].get("completion_tokens")
                for ch in obj.get("choices") or []:
                    d = ch.get("delta") or {}
                    txt = (d.get("content") or "") + (d.get("reasoning_content") or "")
                    if txt:
                        if res.t_first is None:
                            res.t_first = time.monotonic()
                        parts.append(txt)
    except Exception as e:  # noqa: BLE001 -- recorded per request, never swallowed
        res.error = f"{type(e).__name__}: {e}"[:300]
        if not res.status:
            res.status = getattr(e, "code", 0) or -1
    res.t_done = time.monotonic()
    res.text = "".join(parts)


def run_requests(front: str, reqs: Sequence[BurstRequest], mode: str,
                 timeout: float) -> List[RequestResult]:
    results = [RequestResult(r.idx, r.target_tokens, r.code) for r in reqs]
    if mode == "serial":
        for r, res in zip(reqs, results):
            _send(front, r, res, timeout, None)
        return results
    barrier = threading.Barrier(len(reqs))
    threads = [threading.Thread(target=_send, args=(front, r, res, timeout, barrier), daemon=True)
               for r, res in zip(reqs, results)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout + 60)
    return results


# ---------------------------------------------------------------- log parsing

_TS = r"\[(?P<ts>\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d)(?:[,.]\d+)?Z?(?: PP(?P<rank>\d+))?\]"
RE_BATCH = re.compile(_TS + r".*?Prefill batch, #new-seq: (?P<seq>\d+), #new-token: (?P<tok>\d+),"
                      r" #cached-token: (?P<cached>\d+)")
RE_RANK = re.compile(_TS + r".*?Prefill rank batch, #new-token: (?P<tok>\d+),.*?gpu-ms: (?P<gpu>[\d.]+)")
RE_FWD = re.compile(_TS + r".*?FWD-TIMING-PREFILL forward=(?P<fwd>\d+) tokens=(?P<tok>\d+) (?P<rest>.*)")
RE_MS = re.compile(r"(\w+)_ms=([\d.]+)")
RE_SERVED_P = re.compile(_TS + r".*?WEG2-SERVED group=P leg=1 rid=(?P<rid>\S+) prompt_tokens=(?P<pt>\d+)"
                         r" cached_tokens=(?P<ct>\d+) wall=(?P<wall>[\d.]+)s")
RE_SERVED_D = re.compile(_TS + r".*?WEG2-SERVED group=D leg=2 rid=(?P<rid>\S+) status=(?P<st>\d+)"
                         r" prompt_tokens=(?P<pt>\d+) cached_tokens=(?P<ct>\d+)")
RE_FLIP = re.compile(r"WEG2-FLIP begin")


def _epoch(ts: str) -> float:
    return time.mktime(time.strptime(ts.replace("T", " "), "%Y-%m-%d %H:%M:%S"))


@dataclass
class PLogWindow:
    batches: Dict[int, List[Tuple[int, int, int]]] = field(default_factory=dict)  # rank -> (seq, tok, cached)
    rank_ms: Dict[int, List[Tuple[int, float]]] = field(default_factory=dict)  # rank -> (tok, gpu_ms)
    fwd: Dict[int, List[Dict[str, float]]] = field(default_factory=dict)  # rank -> {tokens, *_ms}
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None


def parse_p_log(text: str) -> PLogWindow:
    w = PLogWindow()
    for line in text.splitlines():
        m = RE_BATCH.search(line)
        if m:
            r = int(m.group("rank") or 0)
            w.batches.setdefault(r, []).append((int(m.group("seq")), int(m.group("tok")),
                                                int(m.group("cached"))))
            ts = _epoch(m.group("ts"))
            w.first_ts = ts if w.first_ts is None else min(w.first_ts, ts)
            w.last_ts = ts if w.last_ts is None else max(w.last_ts, ts)
            continue
        m = RE_RANK.search(line)
        if m:
            w.rank_ms.setdefault(int(m.group("rank") or 0), []).append(
                (int(m.group("tok")), float(m.group("gpu"))))
            continue
        m = RE_FWD.search(line)
        if m:
            rec = {"tokens": float(m.group("tok"))}
            rec.update({k: float(v) for k, v in RE_MS.findall(m.group("rest"))})
            w.fwd.setdefault(int(m.group("rank") or 0), []).append(rec)
    return w


@dataclass
class FrontWindow:
    p_served: List[Tuple[float, int, int, float]] = field(default_factory=list)  # (end_ts, pt, ct, wall)
    d_served: List[Tuple[float, int, int]] = field(default_factory=list)  # (end_ts, pt, ct)
    flips: int = 0


def parse_front_log(text: str) -> FrontWindow:
    w = FrontWindow()
    for line in text.splitlines():
        m = RE_SERVED_P.search(line)
        if m:
            w.p_served.append((_epoch(m.group("ts")), int(m.group("pt")), int(m.group("ct")),
                               float(m.group("wall"))))
            continue
        m = RE_SERVED_D.search(line)
        if m:
            w.d_served.append((_epoch(m.group("ts")), int(m.group("pt")), int(m.group("ct"))))
            continue
        if RE_FLIP.search(line):
            w.flips += 1
    return w


def linear_fit(points: Sequence[Tuple[int, float]]) -> Optional[Tuple[float, float]]:
    """Kleinste Quadrate ``y = a + b*x``; None bei < 2 verschiedenen x."""
    pts = [(float(x), float(y)) for x, y in points]
    if len({x for x, _ in pts}) < 2:
        return None
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    b = sum((x - mx) * (y - my) for x, y in pts) / sxx
    return my - b * mx, b


def attach_front(results: Sequence[RequestResult], fw: FrontWindow) -> None:
    """Front-Zeilen den Requests zuordnen -- ueber prompt_tokens (je Request verschieden)."""
    by_pt: Dict[int, List[Tuple[float, int, int, float]]] = {}
    for rec in fw.p_served:
        by_pt.setdefault(rec[1], []).append(rec)
    d_pts = {rec[1] for rec in fw.d_served}
    for r in results:
        if r.prompt_tokens is None:
            continue
        hits = by_pt.get(int(r.prompt_tokens), [])
        if len(hits) == 1:
            r.route, r.p_wall_s, r.p_cached_tokens = "P", hits[0][3], hits[0][2]
        elif len(hits) > 1:
            r.route = "P?"  # zwei P-Zeilen mit derselben Laenge: nicht eindeutig
        elif int(r.prompt_tokens) in d_pts:
            r.route = "D"


def summarize(results: Sequence[RequestResult], pw: PLogWindow, fw: Optional[FrontWindow],
              mode: str) -> dict:
    n = len(results)
    ok = [r for r in results if r.status == 200]
    tokens = sum(int(r.prompt_tokens or 0) for r in ok)
    sends = [r.t_send for r in results if r.t_send]
    dones = [r.t_done for r in results if r.t_done]
    wall = (max(dones) - min(sends)) if sends and dones else 0.0
    pp0 = pw.batches.get(0, [])
    batches = len(pp0)
    multi = sum(1 for s, _, _ in pp0 if s > 1)
    p_new_tokens = sum(t for _, t, _ in pp0)
    # Ohne Front-Log ist die Route unbekannt: alle erfolgreichen Requests zaehlen als P.
    p_reqs = [r for r in results if r.route in ("P", "P?")] if fw is not None else list(ok)
    p_req_n = len(p_reqs)
    # P-Fenster: aus der Front (erste leg-1-Annahme bis letzte Antwort), sonst Client-Wand.
    p_window = None
    if fw is not None and fw.p_served:
        p_window = max(e for e, _, _, _ in fw.p_served) - min(e - w for e, _, _, w in fw.p_served)
    p_tokens = sum(int(r.prompt_tokens or 0) for r in p_reqs)
    agg_window = p_window if p_window and p_window > 0 else wall
    agg_tok_s = (p_tokens / agg_window) if agg_window > 0 else 0.0

    fits: Dict[int, Optional[Tuple[float, float]]] = {}
    tails: Dict[int, Optional[float]] = {}
    busy: Dict[int, float] = {}
    for rank, pts in sorted(pw.rank_ms.items()):
        fits[rank] = linear_fit([(t, g) for t, g in pts if t >= FIT_MIN_TOKENS])
        tail = [g for t, g in pts if t <= TAIL_MAX_TOKENS]
        tails[rank] = statistics.median(tail) if tail else None
        busy[rank] = sum(g for _, g in pts)
    sockel_ms = fits.get(0)[0] if fits.get(0) else None
    fetch = [f["moe_fetch"] for f in pw.fwd.get(0, []) if "moe_fetch" in f and f["tokens"] > TAIL_MAX_TOKENS]
    moe_fetch_ms = statistics.median(fetch) if fetch else None

    serial_s = None
    if fits and all(fits.get(r) for r in fits) and p_reqs:
        serial_s = 0.0
        for r in p_reqs:
            for rank, fit in fits.items():
                serial_s += (fit[0] + fit[1] * int(r.prompt_tokens or 0) + (tails.get(rank) or 0.0)) / 1000.0
    serial_tok_s_est = (p_tokens / serial_s) if serial_s else None

    fwd_per_req = (batches / p_req_n) if p_req_n else None
    if batches == 0:
        verdict = "KEINE-P-ZEILEN"
    elif fwd_per_req is not None and fwd_per_req < 1.0:
        verdict = "JA"
    elif multi > 0:
        verdict = "SCHWANZ"
    else:
        verdict = "NEIN"
    needles = sum(1 for r in results if r.needle == "MATCH")
    return {
        "mode": mode, "n": n, "ok": len(ok), "tokens": tokens, "wall_s": round(wall, 2),
        "p_requests": p_req_n, "p_tokens": p_tokens,
        "p_window_s": None if p_window is None else round(p_window, 2),
        "agg_tok_s": round(agg_tok_s, 1), "batches": batches, "multi_seq_batches": multi,
        "max_new_seq": max((s for s, _, _ in pp0), default=0),
        "p_new_tokens": p_new_tokens,
        "tok_per_fwd": round(p_new_tokens / batches, 1) if batches else 0.0,
        "fwd_per_req": None if fwd_per_req is None else round(fwd_per_req, 2),
        "sockel_ms": None if sockel_ms is None else round(sockel_ms, 1),
        "ms_per_token_pp0": None if not fits.get(0) else round(fits[0][1], 4),
        "moe_fetch_ms": None if moe_fetch_ms is None else round(moe_fetch_ms, 1),
        "tail_ms": {r: (None if v is None else round(v, 1)) for r, v in tails.items()},
        "busy_ms": {r: round(v, 1) for r, v in busy.items()},
        "serial_tok_s_est": None if serial_tok_s_est is None else round(serial_tok_s_est, 1),
        "flips": None if fw is None else fw.flips,
        "needles": needles, "verdict": verdict,
    }


def format_report(summary: dict, results: Sequence[RequestResult]) -> List[str]:
    s = summary
    lines = [
        "BURST-PROBE n=%d tokens=%d wall=%.2f agg_tok_s=%.1f batches=%d multi_seq_batches=%d "
        "tok_per_fwd=%.1f mode=%s p_requests=%d p_tokens=%d p_window_s=%s max_new_seq=%d "
        "fwd_per_req=%s sockel_ms=%s ms_per_token_pp0=%s moe_fetch_ms=%s serial_tok_s_est=%s "
        "flips=%s needles=%d/%d"
        % (s["n"], s["tokens"], s["wall_s"], s["agg_tok_s"], s["batches"], s["multi_seq_batches"],
           s["tok_per_fwd"], s["mode"], s["p_requests"], s["p_tokens"], s["p_window_s"],
           s["max_new_seq"], s["fwd_per_req"], s["sockel_ms"], s["ms_per_token_pp0"],
           s["moe_fetch_ms"], s["serial_tok_s_est"], s["flips"], s["needles"], s["n"]),
        "BURST-TABLE idx target prompt route ttft_s p_wall_s total_s p_cached needle code",
    ]

    def f(v, fmt="%.2f"):
        return "-" if v is None else fmt % v

    for r in results:
        lines.append("BURST-REQ %d %d %s %s %s %s %s %s %s %s%s" % (
            r.idx, r.target_tokens, r.prompt_tokens if r.prompt_tokens is not None else "-",
            r.route, f(r.ttft_s), f(r.p_wall_s), f(r.total_s),
            r.p_cached_tokens if r.p_cached_tokens is not None else "-", r.needle, r.code,
            (" err=" + r.error) if r.error else ""))
    why = {
        "JA": "weniger P-Forwards als P-Requests: der Sockel wird geteilt",
        "SCHWANZ": "#new-seq > 1 nur als END-ANCHOR-Schwanz + naechster Koerper; ein Request je Forward",
        "NEIN": "jeder PP0-Forward traegt genau einen Request (Koerper und Schwanz getrennt)",
        "KEINE-P-ZEILEN": "im Fenster keine PP0-'Prefill batch'-Zeile (alle SHORT an D, oder falsches --p-log)",
    }[s["verdict"]]
    lines.append("BURST-VERDICT batching=%s needles=%d/%d (%s; sockel/serial sind RECHNUNG aus den "
                 "PP0-Rangzeilen, nicht gemessen)" % (s["verdict"], s["needles"], s["n"], why))
    return lines


def _size(path: Optional[str]) -> int:
    try:
        return os.path.getsize(path) if path else 0
    except OSError:
        return 0


def _read_from(path: Optional[str], offset: int) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as fh:
        fh.seek(offset)
        return fh.read().decode(errors="replace")


def derive_front_log(p_log: Optional[str]) -> Optional[str]:
    if not p_log:
        return None
    for a, b in ((".P.log", ".front.log"), (".P", ".front")):
        if p_log.endswith(a):
            cand = p_log[: -len(a)] + b
            if os.path.exists(cand):
                return cand
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--front", default="http://127.0.0.1:30030")
    ap.add_argument("--model", default="Qwen3.8-Flash-Next")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--min-tokens", type=int, default=3000)
    ap.add_argument("--max-tokens-prompt", type=int, default=8000)
    ap.add_argument("--answer-tokens", type=int, default=48,
                    help="max_tokens je Antwort; der Code steht vorn, D dekodiert bs1 nacheinander")
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--nonce", default="", help="Zusatz in der Kopfzeile gegen Store-Treffer aus Vorlaeufen")
    ap.add_argument("--mode", choices=("burst", "serial"), default="burst")
    ap.add_argument("--p-log", required=True, help="Log von group P (boot_weg2_<tag>_..P.log)")
    ap.add_argument("--front-log", default=None, help="Default: aus --p-log abgeleitet")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--settle-s", type=float, default=2.0, help="Wartezeit fuer die letzten Logzeilen")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args(argv)

    front_log = a.front_log or derive_front_log(a.p_log)
    reqs = build_requests(a.n, a.seed, a.min_tokens, a.max_tokens_prompt, a.model, a.answer_tokens, a.nonce)
    off_p, off_f = _size(a.p_log), _size(front_log)
    results = run_requests(a.front, reqs, a.mode, a.timeout)
    time.sleep(max(0.0, a.settle_s))
    pw = parse_p_log(_read_from(a.p_log, off_p))
    fw = parse_front_log(_read_from(front_log, off_f)) if front_log else None
    if fw is not None:
        attach_front(results, fw)
    summary = summarize(results, pw, fw, a.mode)
    for line in format_report(summary, results):
        print(line, flush=True)
    if a.json_out:
        with open(a.json_out, "w") as fh:
            json.dump({"summary": summary, "requests": [
                dict(asdict(r), ttft_s=r.ttft_s, total_s=r.total_s, needle=r.needle, text=r.text[:300])
                for r in results]}, fh, indent=1)
    if any(r.status != 200 for r in results):
        return 3
    return 0 if summary["needles"] == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
