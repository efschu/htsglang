#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""flipzeit_eval -- Nutzer-Flipzeit eines Front-Logs (SCHEDULER-SOLL v2 R10, h16u 1017).

Liest ein *.front.log und misst, ohne etwas zu rechnen, was der Scheduler daraus entscheiden koennte:

  * PDFLIP-FLIP-USERZEIT dir=<D>P|P>D> user_flipzeit_ms=<n>  -> p50 / p90 / max je Richtung (Sekunden)
  * PDFLIP DP-WAIT ... wait_s= hold_s=                       -> Hold-Anteil = sum(hold_s) / sum(wait_s) ueber alle rids
  * Flips je Stunde (USERZEIT-Zeilen je Log-Stunde, UTC) und im Mittel ueber die Log-Spanne
  * Doppelflips: zwei Flips DERSELBEN Richtung mit < 30 s Abstand (Hin-und-zurueck-Zyklus unter 30 s)
  * T_halb-Empfehlung: (p90 D>P + p90 P>D) / 2 -- SOLL R10; Abweichung von --t-half (Default 4.5 s = Flag
    FLLIPER_PDFLIP_HALF_FLIP_S) in Prozent, "abweichung" ab 30 %.

Aufruf:  flipzeit_eval.py <front.log> [--t-half 4.5] [--json]      flipzeit_eval.py --selftest
Nur Lesen. Jede Zahl nennt ihren Nenner (n) und ihr Instrument (die Logzeile).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple

RE_TS = re.compile(r"^\[(\d{4}-\d\d-\d\d)[ T](\d\d:\d\d:\d\d)")
# "PDFLIP-FLIP-USERZEIT dir=" ist der Anker; die FLIPCYCLE-Zeilen zitieren den Namen nur im Klammertext.
RE_USERZEIT = re.compile(r"PDFLIP-FLIP-USERZEIT dir=(D>P|P>D) epoch=(\d+) user_flipzeit_ms=(\d+|None)")
RE_DPWAIT = re.compile(r"PDFLIP DP-WAIT rid=(\S+) epoch=(\d+) wait_s=([\d.]+) hold_s=([\d.]+)")
RE_FLIP_BEGIN = re.compile(r"PDFLIP-FLIP begin epoch=")

DIRS = ("D>P", "P>D")
DOUBLE_FLIP_S = 30.0
DEVIATION_PCT = 30.0


def percentile(values: List[float], q: float) -> Optional[float]:
    """q in [0,100], lineare Interpolation (numpy-Default)."""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def _epoch_s(date: str, hms: str) -> float:
    return _dt.datetime.strptime(date + " " + hms, "%Y-%m-%d %H:%M:%S").replace(tzinfo=_dt.timezone.utc).timestamp()


def parse(lines: Iterable[str]) -> dict:
    flips: List[Tuple[float, str, Optional[float]]] = []   # (t, dir, user_s)
    waits: List[Tuple[float, float]] = []                  # (wait_s, hold_s)
    begins = 0
    t_first = t_last = None
    for line in lines:
        m = RE_TS.match(line)
        if not m:
            continue
        t = _epoch_s(m.group(1), m.group(2))
        if t_first is None:
            t_first = t
        t_last = t
        if "PDFLIP-FLIP" in line:
            u = RE_USERZEIT.search(line)
            if u:
                flips.append((t, u.group(1), None if u.group(3) == "None" else int(u.group(3)) / 1000.0))
                continue
            if RE_FLIP_BEGIN.search(line):
                begins += 1
                continue
        if "DP-WAIT" in line:
            w = RE_DPWAIT.search(line)
            if w:
                waits.append((float(w.group(3)), float(w.group(4))))
    return {"flips": flips, "waits": waits, "begins": begins, "t_first": t_first, "t_last": t_last}


def double_flips(flips: List[Tuple[float, str, Optional[float]]], window_s: float = DOUBLE_FLIP_S) -> List[Tuple[str, float]]:
    """Zwei aufeinanderfolgende Flips derselben Richtung mit Abstand < window_s (je Richtung in Zeitreihenfolge)."""
    out = []
    for d in DIRS:
        ts = sorted(t for t, dd, _ in flips if dd == d)
        for a, b in zip(ts, ts[1:]):
            if b - a < window_s:
                out.append((d, b - a))
    return out


def evaluate(parsed: dict, t_half_s: float = 4.5) -> dict:
    flips = parsed["flips"]
    res: dict = {"n_flips": len(flips), "n_flip_begin": parsed["begins"], "dir": {}}
    for d in DIRS:
        vals = [u for _, dd, u in flips if dd == d and u is not None]
        res["dir"][d] = {"n": len(vals), "n_missing": sum(1 for _, dd, u in flips if dd == d and u is None),
                         "p50_s": percentile(vals, 50), "p90_s": percentile(vals, 90), "max_s": max(vals) if vals else None}
    # Hold-Anteil
    sw = sum(w for w, _ in parsed["waits"])
    sh = sum(h for _, h in parsed["waits"])
    res["dp_wait"] = {"n": len(parsed["waits"]), "sum_wait_s": sw, "sum_hold_s": sh,
                      "hold_share": (sh / sw) if sw > 0 else None}
    # Flips je Stunde
    per_hour: Dict[str, int] = {}
    for t, _, _ in flips:
        h = _dt.datetime.fromtimestamp(t, _dt.timezone.utc).strftime("%Y-%m-%d %H")
        per_hour[h] = per_hour.get(h, 0) + 1
    span_h = ((parsed["t_last"] - parsed["t_first"]) / 3600.0) if parsed["t_first"] is not None else 0.0
    res["per_hour"] = per_hour
    res["span_h"] = span_h
    res["flips_per_h_mean"] = (len(flips) / span_h) if span_h > 0 else None
    dbl = double_flips(flips)
    res["double_flips"] = {"n": len(dbl), "share": (len(dbl) / len(flips)) if flips else None,
                           "window_s": DOUBLE_FLIP_S}
    # T_halb-Empfehlung (R10)
    p_dp, p_pd = res["dir"]["D>P"]["p90_s"], res["dir"]["P>D"]["p90_s"]
    rec = {"t_half_flag_s": t_half_s, "recommend_s": None, "dev_pct": None, "verdict": "keine-daten"}
    if p_dp is not None and p_pd is not None:
        rec["recommend_s"] = (p_dp + p_pd) / 2.0
        rec["dev_pct"] = 100.0 * abs(t_half_s - rec["recommend_s"]) / rec["recommend_s"] if rec["recommend_s"] > 0 else None
        rec["verdict"] = "ok" if (rec["dev_pct"] is not None and round(rec["dev_pct"], 6) <= DEVIATION_PCT) else "abweichung"
    res["t_half"] = rec
    return res


def _f(v, nd=2):
    return "-" if v is None else ("%.*f" % (nd, v))


def render(res: dict, label: str = "") -> str:
    o = ["flipzeit_eval %s" % label,
         "Instrument: PDFLIP-FLIP-USERZEIT user_flipzeit_ms (n=%d Flips, davon FLIP begin gezaehlt: %d); DP-WAIT n=%d rids"
         % (res["n_flips"], res["n_flip_begin"], res["dp_wait"]["n"])]
    for d in DIRS:
        r = res["dir"][d]
        o.append("  %s  n=%d (ohne Wert %d)  p50=%s s  p90=%s s  max=%s s" % (d, r["n"], r["n_missing"], _f(r["p50_s"]), _f(r["p90_s"]), _f(r["max_s"])))
    w = res["dp_wait"]
    o.append("  Hold-Anteil (DP-WAIT): sum hold_s=%s / sum wait_s=%s = %s %%  (n=%d rids)" % (
        _f(w["sum_hold_s"], 1), _f(w["sum_wait_s"], 1), _f(None if w["hold_share"] is None else 100 * w["hold_share"], 1), w["n"]))
    o.append("  Flips je Stunde: Mittel %s/h ueber %s h Log-Spanne; je Stunde(UTC): %s" % (
        _f(res["flips_per_h_mean"], 1), _f(res["span_h"], 2), ", ".join("%sh=%d" % (h[-2:], n) for h, n in sorted(res["per_hour"].items())) or "-"))
    dbl = res["double_flips"]
    o.append("  Doppelflips (gleiche Richtung < %.0f s): %d = %s %% der Flips" % (dbl["window_s"], dbl["n"], _f(None if dbl["share"] is None else 100 * dbl["share"], 1)))
    t = res["t_half"]
    o.append("  T_halb-Empfehlung (R10) = (p90 D>P + p90 P>D)/2 = %s s; Flag steht auf %s s; Abweichung %s %% -> %s" % (
        _f(t["recommend_s"]), _f(t["t_half_flag_s"]), _f(t["dev_pct"], 1), t["verdict"]))
    return "\n".join(o)


# ----------------------------------------------------------------------------------------------------------------
def _synthetic() -> List[str]:
    def ts(hms): return "[2026-10-16 %s,000] INFO pdflip.front: " % hms
    L = []
    # D>P: 1,2,3,4,10 s (p50=3, p90=7.6 linear), P>D: 2,2,2,2,2 s (p90=2). Zeiten: D>P 10:00:00, 10:00:20 (Doppel), 11:00:00, 11:30:00, 11:59:00
    for hms, d, ms in (("10:00:00", "D>P", 1000), ("10:00:05", "P>D", 2000), ("10:00:20", "D>P", 2000), ("10:00:40", "P>D", 2000),
                       ("11:00:00", "D>P", 3000), ("11:30:00", "P>D", 2000), ("11:45:00", "D>P", 4000), ("11:59:00", "D>P", 10000)):
        L.append(ts(hms) + "PDFLIP-FLIP begin epoch=0 sleep=D wake=P outstanding=0 queue=0")
        L.append(ts(hms) + "PDFLIP-FLIPCYCLE stage=total dir=%s epoch=1 ms=1 (see PDFLIP-FLIP-USERZEIT user_flipzeit_ms)" % d)
        L.append(ts(hms) + "PDFLIP-FLIP-USERZEIT dir=%s epoch=1 user_flipzeit_ms=%d missing=None start=x" % (d, ms))
    L.append(ts("10:00:06") + "PDFLIP DP-WAIT rid=a epoch=1 wait_s=10.0 hold_s=6.0 drain_s=0.0 flip_s=4.0 dominant=hold")
    L.append(ts("10:00:06") + "PDFLIP DP-WAIT rid=b epoch=1 wait_s=5.0 hold_s=1.0 drain_s=0.0 flip_s=4.0 dominant=flip")
    L.append("garbage line without timestamp")
    return L


def selftest() -> int:
    r = evaluate(parse(_synthetic()), t_half_s=4.5)
    fails = []
    def chk(name, got, want, tol=1e-6):
        ok = (got == want) if not isinstance(want, float) else (got is not None and abs(got - want) <= tol)
        if not ok:
            fails.append("%s: got %r want %r" % (name, got, want))
    chk("n_flips", r["n_flips"], 8)
    chk("n_flip_begin", r["n_flip_begin"], 8)          # die FLIPCYCLE-Zeile (zitiert den Namen) zaehlt NICHT als Flip
    chk("dp.n", r["dir"]["D>P"]["n"], 5)
    chk("dp.p50", r["dir"]["D>P"]["p50_s"], 3.0)
    chk("dp.p90", r["dir"]["D>P"]["p90_s"], 7.6)       # sortiert 1,2,3,4,10 -> pos 3.6 -> 4+0.6*6
    chk("dp.max", r["dir"]["D>P"]["max_s"], 10.0)
    chk("pd.p90", r["dir"]["P>D"]["p90_s"], 2.0)
    chk("hold_share", r["dp_wait"]["hold_share"], 7.0 / 15.0)
    chk("per_hour.10", r["per_hour"]["2026-10-16 10"], 4)
    chk("per_hour.11", r["per_hour"]["2026-10-16 11"], 4)
    chk("double", r["double_flips"]["n"], 1)           # nur D>P 10:00:00->10:00:20 (20 s); P>D 10:00:05->10:00:40 sind 35 s; Rest Minuten
    chk("recommend", r["t_half"]["recommend_s"], (7.6 + 2.0) / 2)
    chk("verdict", r["t_half"]["verdict"], "abweichung" if abs(4.5 - 4.8) / 4.8 * 100 > 30 else "ok")
    # Randfall: leeres Log
    e = evaluate(parse([]))
    chk("empty.verdict", e["t_half"]["verdict"], "keine-daten")
    chk("pct.single", percentile([5.0], 90), 5.0)
    if fails:
        print("SELFTEST ROT:\n  " + "\n  ".join(fails))
        return 1
    print("SELFTEST GRUEN (16 Pruefungen)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log", nargs="?")
    ap.add_argument("--t-half", type=float, default=4.5, help="Flag FLLIPER_PDFLIP_HALF_FLIP_S (Default 4.5)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.log:
        ap.error("log fehlt")
    with open(a.log, errors="replace") as f:
        res = evaluate(parse(f), a.t_half)
    print(json.dumps(res, indent=1, default=str) if a.json else render(res, a.log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
