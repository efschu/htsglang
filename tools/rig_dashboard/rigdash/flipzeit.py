"""Flipzeit -- DIE eine Berechnung (Nutzer 06.10.2026, ausdrueckliche Vorgabe, einzige Definition):

    P>D = letzter P-Chunk fertig  ->  erstes Decode-Token erzeugt
    D>P = letztes Decode-Token erzeugt  ->  erster Prefill-Chunk faengt an zu rechnen
    Ausnahme: kein Flip zaehlt, wenn kein Prefill oder Decode ansteht (Leerlauf-Flip).

Jede Flipzeit-Zahl der Seite -- Kachel Ueberblick, Kachel Verlauf, Diagramm, Boot-Liste, Grafana-Punkte -- ist
``stats()`` ueber dieselben Punkte (``Point``: Flip-Beginn t, Richtung, total_ms), die ``from_marks()`` aus den
Marken ``flip_t2t`` der Historie liest (history.Recorder schreibt sie aus ipcboot.flip_views, jeder Flip einmal).
Die Kacheln unterscheiden sich nur im FENSTER, und das Fenster steht in der Beschriftung:

    Ueberblick   die letzten OVERVIEW_S (60 min) dieses Modells                 (``window``)
    Verlauf      der gewaehlte Zeitbereich bzw. der gezoomte Ausschnitt          (dieselbe Funktion)
    Boot-Liste   der ganze Boot (Beginn .. letztes Lebenszeichen)                (dieselbe Funktion)

Gezaehlt wird nur ein ABGESCHLOSSENER, GEMESSENER Flip (``counted``): kind "ok", total_ms da, nicht vorlaeufig.
Nicht gezaehlt, nirgends und auch nicht als "zuletzt": offene Flips, vorlaeufige (D>P, dessen Start im D-Log
noch nicht feststeht), Flips mit fehlendem Endpunkt, Leerlauf-Flips.  Layer/Vorlauf/Nachlauf sind nur die
ZERLEGUNG der einen Zahl (``parts``, Summe = total), nie ein eigener Kopfwert.

Rein (keine Ein-/Ausgabe), Python-Tests: tests/test_flipzeit_1006.py."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

DIRS = ("P>D", "D>P")

#: die EINE Definition (Wortlaut des Nutzers 06.10.); index.html (FLIP_DEF) und grafik.js tragen denselben Text,
#: tests/test_flipzeit_1006.py prueft die Gleichheit
DEFINITION = {
    "P>D": "last P chunk done → first decode token produced",
    "D>P": "last decode token produced → first prefill chunk starts computing (first forward on PP0)",
}
EXCEPTION = "No flip counts when no prefill or decode is pending (idle flip)."

#: Fenster der Ueberblick-Kachel (= der Bereich "1h" des Verlaufs)
OVERVIEW_S = 3600.0
#: so viele juengste Flips traegt ``recent`` (der Balkenstreifen der Boot-Karte)
RECENT_N = 24

MARK_KIND = "flip_t2t"          # Marke mit Wert: ein gezaehlter Flip
SKIP_KIND = "flip_skip"         # Marke ohne Wert: ein Leerlauf-Flip (nicht gezaehlt, nur sein Zaehler)

#: Zerlegung im Marken-Label, in dieser Reihenfolge (Summe = total, Rest explizit)
PART_KEYS = ("warmup_ms", "layer_ms", "wake_kv_dc_ms", "nachlauf_ms", "rest_ms")
VOR_KEYS = ("leer_ms", "halt_ms", "park_ms", "vor_rest_ms")



def window_label(span_s: float, zoomed: bool = False) -> str:
    """Die Beschriftung des Fensters: "last 60 min" / "last 6 h" / "last 7 d" or "zoomed section (n min)"."""
    if span_s <= 5400:
        w = "%d min" % round(span_s / 60.0)
    elif span_s < 48 * 3600:
        w = "%d h" % round(span_s / 3600.0)
    else:
        w = "%d d" % round(span_s / 86400.0)
    return "zoomed section (%s)" % w if zoomed else "last " + w


def counted(row: dict) -> bool:
    """Ein Flip der ipcboot.flip_views-Zeilen, der in JEDE Zahl eingeht (und sonst in keine)."""
    return (row.get("kind") == "ok" and row.get("total_ms") is not None and not row.get("provisional")
            and row.get("dir") in DIRS)


def mark_label(row: dict) -> str:
    """``<dir> v=.. l=.. w=.. n=.. r=.. [(leer=.. halt=.. park=.. vr=..)]`` -- die Zerlegung hinter der Marke."""
    parts = "v=%d l=%d w=%d n=%d r=%d" % tuple(int(round(row.get(k) or 0)) for k in PART_KEYS)
    if row.get("leer_ms") is not None:
        parts += " (leer=%d halt=%d park=%d vr=%d)" % tuple(int(round(row.get(k) or 0)) for k in VOR_KEYS)
    return "%s %s" % (row["dir"], parts)


def parse_label(label: str) -> Optional[dict]:
    """Umkehrung von ``mark_label``: {"dir", "parts": {warmup_ms.. rest_ms, leer_ms..}}; None bei fremdem Label.
    (Das Label ist unser eigenes Format, kein Log: Token-Schnitt ohne regulaeren Ausdruck.)"""
    toks = (label or "").replace("(", " ").replace(")", " ").split()
    if len(toks) < 6 or toks[0] not in DIRS:
        return None
    try:
        kv = {k: int(v) for k, v in (t.split("=", 1) for t in toks[1:] if "=" in t)}
        parts = {k: kv[c] for k, c in zip(PART_KEYS, ("v", "l", "w", "n", "r"))}
        if "leer" in kv:
            parts.update({k: kv[c] for k, c in zip(VOR_KEYS, ("leer", "halt", "park", "vr"))})
    except (KeyError, ValueError):
        return None
    return {"dir": toks[0], "parts": parts}


def quantile(xs: Iterable[Optional[float]], p: float) -> Optional[float]:
    """Nearest-rank (ceil), die einzige Perzentil-Rechnung der Flipzeit."""
    v = sorted(x for x in xs if x is not None)
    return v[max(0, math.ceil(p * len(v)) - 1)] if v else None


def from_marks(marks: Iterable[dict]) -> List[dict]:
    """Punkte {t, dir, ms, parts} aus den Marken ``flip_t2t`` (history.HistoryDB.marks), nach t sortiert."""
    out = []
    for m in marks:
        if m.get("kind") != MARK_KIND or m.get("v") is None:
            continue
        lab = parse_label(m.get("label") or "")
        d = lab["dir"] if lab else next((x for x in DIRS if (m.get("label") or "").startswith(x)), None)
        if d is None:
            continue
        out.append({"t": m["t"], "dir": d, "ms": float(m["v"]), "parts": lab["parts"] if lab else None})
    return sorted(out, key=lambda x: x["t"])


def skipped_from_marks(marks: Iterable[dict]) -> Dict[str, int]:
    """Leerlauf-Flips je Richtung (Marken ``flip_skip``): nicht gezaehlt, nur benannt."""
    n = {d: 0 for d in DIRS}
    for m in marks:
        if m.get("kind") == SKIP_KIND:
            d = next((x for x in DIRS if (m.get("label") or "").startswith(x)), None)
            if d:
                n[d] += 1
    return n


def from_views(views: Iterable[dict]) -> List[dict]:
    """Punkte aus ipcboot.flip_views-Zeilen -- nur ``counted`` (dieselbe Menge, die die Historie schreibt)."""
    return sorted(({"t": x["begin"], "dir": x["dir"], "ms": float(x["total_ms"]),
                    "parts": parse_label(mark_label(x))["parts"]} for x in views if counted(x)),
                  key=lambda p: p["t"])


def window(points: Iterable[dict], lo: Optional[float], hi: Optional[float]) -> List[dict]:
    """Die Punkte mit lo <= t <= hi (None = offen)."""
    return [p for p in points if (lo is None or p["t"] >= lo) and (hi is None or p["t"] <= hi)]


def stats(points: Iterable[dict], d: str) -> dict:
    """n, letzter, p50, p90, max der Richtung ``d`` -- DIE Kennzahlen der Flipzeit (Millisekunden)."""
    mine = sorted((p for p in points if p["dir"] == d), key=lambda p: p["t"])
    vals = [p["ms"] for p in mine]
    last = mine[-1] if mine else None
    return {"n": len(vals), "last_ms": last["ms"] if last else None, "last_t": last["t"] if last else None,
            "parts": last.get("parts") if last else None,
            "p50_ms": quantile(vals, 0.5), "p90_ms": quantile(vals, 0.9), "max_ms": max(vals) if vals else None}


def summary(points: Iterable[dict], skipped: Optional[Dict[str, int]] = None) -> dict:
    """{dir: stats + idle_n} fuer beide Richtungen (die Nutzlast jeder Kachel)."""
    pts = list(points)
    sk = skipped or {}
    return {d: dict(stats(pts, d), idle_n=int(sk.get(d, 0))) for d in DIRS}


def recent(points: Iterable[dict], n: int = RECENT_N) -> List[dict]:
    """Die n juengsten gezaehlten Flips (t, dir, ms) fuer den Balkenstreifen."""
    return [{"t": p["t"], "dir": p["dir"], "ms": p["ms"]} for p in sorted(points, key=lambda p: p["t"])[-n:]]


def tile(marks: List[dict], lo: Optional[float], hi: Optional[float], label: str) -> dict:
    """Die Kachel-Nutzlast eines Fensters aus den Marken: {"window": {lo, hi, label}, "P>D": .., "D>P": ..,
    "recent": [..]}.  Ueberblick, Verlauf und Boot-Liste rufen genau diese Funktion."""
    inside = [m for m in marks if (lo is None or m["t"] >= lo) and (hi is None or m["t"] <= hi)]
    pts = from_marks(inside)
    out = summary(pts, skipped_from_marks(inside))
    out["recent"] = recent(pts)
    out["window"] = {"lo": lo, "hi": hi, "label": label}
    return out
