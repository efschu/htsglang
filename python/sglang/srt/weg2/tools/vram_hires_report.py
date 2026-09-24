#!/usr/bin/env python3
"""Leser der VRAM-Sonde (fnFL2 H55): je Phase das WAHRE Maximum je Karte.

    python3 vram_hires_report.py vram_hires_TAG.csv boot_..TAG...P.log boot_..TAG...D.log \\
        [--front F] [--dry dry_TAG.log] [--dmon vram_TAG.log] [--cards 1,0,2]

Liest den ROHSTROM (``<csv ohne .csv>.raw.csv``, 5-ms-Abtastung, nur Wechsel)
und legt die Phasen des Boots darauf:

* ``P chunk i``  -- aus ``WEG2-VRAM-PEAK phase=chunk`` (ms-genau, je Rang) wenn
  vorhanden, sonst zwischen zwei ``Prefill batch``-Zeilen von PP0 (Sekunden-
  genau, das Fenster wird dann um je 1 s geweitet und heisst ``~s``).
* ``flip e P->D`` / ``D->P`` -- ``WEG2-FLIP begin`` .. ``WEG2-FLIP done`` des
  Front-Logs (ms-genau).
* ``D decode e`` -- vom Ende des Flips P->D bis zum Beginn des naechsten.
* ``Boot`` -- der ganze Rohstrom.

Je Phase und Karte:

    true_max   groesster fb-used-Wert der Karte im Fenster (MiB, 5-ms-Sonde)
    rest_min   kleinste Restluft = total - offset - true_max (= cudaMemGetInfo frei)
    at         wo im Fenster das Maximum lag (s ab Fensterbeginn)
    dmon_max   groesster Wert der 1-s-dmon-Zeilen im selben Fenster (vram_TAG.log)
    missed     true_max - dmon_max: was die 1-s-Sicht NICHT gesehen hat
    proc       groesster Prozess der Karte und sein Maximum (Rolle P:PP0 ...)
    plan       Kopfraum der Planer-Karte (dry_TAG.log: P-KARTE stage/KARTE D rang)
    plan-rest  plan - rest_min: > 0 = der Planer hat mehr Luft versprochen als da war
    verdict    UNTER near-OOM (rest_min < near-OOM-Kante) / KNAPP (< 2x) / ok

Zweite Tabelle: die In-Prozess-Fenster (``WEG2-VRAM-PEAK``) je Rang und Phase:
Maximum von peak_reserved/transient gegen die Planer-Transiente, und daneben der
NVML-Prozesswert im selben Fenster -- ``ausser_torch`` = NVML-Prozess-Max minus
peak_reserved (CUDA-Kontext, NCCL, TMS-Regionen, alles was torch nicht zaehlt).

Nur Standardbibliothek.
"""

from __future__ import annotations

import argparse
import bisect
import datetime as _dt
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

#: Rang i einer Gruppe liegt auf NVML cards[i] (boot_<tag>.json 'cards'; Form
#: dieses Rigs: P PP0/1/2 und D TP0/1/2 auf 1/0/2).
DEFAULT_RANK_CARDS = (1, 0, 2)


# --------------------------------------------------------------------------
# Rohstrom
# --------------------------------------------------------------------------


class Trace:
    """Stufenfunktionen je Reihe aus dem Rohstrom; Zeitachse unix_ms."""

    def __init__(self):
        self.t0_unix_ms = 0
        self.anchors: List[Tuple[int, int]] = []  # (t_ms, unix_ms - t_ms)
        self.cards: Dict[int, Dict[str, int]] = {}
        self.series: Dict[str, Tuple[List[int], List[int]]] = {}
        self.roles: Dict[str, str] = {}
        self.calib = ""
        self.end = ""
        self.end_t_ms: Optional[int] = None
        self.gaps = 0

    def offset_at(self, t_ms: int) -> int:
        i = bisect.bisect_right([a[0] for a in self.anchors], t_ms) - 1
        return self.anchors[max(i, 0)][1]

    def unix(self, t_ms: int) -> int:
        return t_ms + self.offset_at(t_ms)

    def window_max(self, key: str, a_ms: int, b_ms: int) -> Optional[Tuple[int, int]]:
        """(max, unix_ms of first max) over [a, b] of the step function."""
        s = self.series.get(key)
        if not s:
            return None
        ts, vs = s
        i = bisect.bisect_right(ts, a_ms) - 1
        best = None
        if i >= 0:
            best = (vs[i], a_ms)
        j = max(i + 1, 0)
        while j < len(ts) and ts[j] <= b_ms:
            if best is None or vs[j] > best[0]:
                best = (vs[j], ts[j])
            j += 1
        return best

    def free_of(self, card: int, used: int) -> Optional[int]:
        c = self.cards.get(card)
        if c is None:
            return None
        return c["total_mib"] - c["offset_mib"] - used


_RX_KV = re.compile(r"(\w+)=(\S+)")


def load_trace(raw_path: str) -> Trace:
    tr = Trace()
    unix_ms_series: Dict[str, Tuple[List[int], List[int]]] = defaultdict(lambda: ([], []))
    with open(raw_path, errors="replace") as f:
        for ln in f:
            if ln.startswith("#"):
                kv = dict(_RX_KV.findall(ln))
                if ln.startswith("# t0_unix_ms"):
                    tr.t0_unix_ms = int(kv["t0_unix_ms"])
                    tr.anchors = [(0, tr.t0_unix_ms)]
                elif ln.startswith("# card"):
                    tr.cards[int(kv["idx"])] = {"total_mib": int(kv["total_mib"]),
                                                 "offset_mib": int(kv["offset_mib"]),
                                                 "name": kv.get("name", "?")}
                elif ln.startswith("# sync"):
                    t = int(kv["t_ms"])
                    tr.anchors.append((t, int(kv["unix_ms"]) - t))
                elif ln.startswith("# proc"):
                    tr.roles[kv["key"]] = kv.get("role", "?")
                elif ln.startswith("# calib"):
                    tr.calib = ln[2:].strip()
                elif ln.startswith("# end"):
                    tr.end = ln[2:].strip()
                    tr.end_t_ms = int(kv["t_ms"]) if kv.get("t_ms", "").isdigit() else None
                elif ln.startswith("# gap"):
                    tr.gaps += 1
                continue
            parts = ln.strip().split(",")
            if len(parts) != 3 or not parts[0].lstrip("-").isdigit():
                continue
            t, key, v = int(parts[0]), parts[1], int(parts[2])
            ts, vs = unix_ms_series[key]
            ts.append(t)
            vs.append(v)
    anchors = sorted(tr.anchors)
    xs = [a[0] for a in anchors]
    for key, (ts, vs) in unix_ms_series.items():
        out = []
        for t in ts:
            i = bisect.bisect_right(xs, t) - 1
            out.append(t + anchors[max(i, 0)][1])
        tr.series[key] = (out, vs)
    return tr


# --------------------------------------------------------------------------
# Logs
# --------------------------------------------------------------------------

_TS_S = re.compile(r"^\[(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d) (\w+)\]")
_TS_MS = re.compile(r"^\[(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d),(\d{3})\]")


def _unix_ms(date: str, hms: str, ms: int = 0) -> int:
    t = _dt.datetime.strptime(f"{date} {hms}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=_dt.timezone.utc)
    return int(t.timestamp() * 1000) + ms


class Window:
    def __init__(self, name: str, a_ms: int, b_ms: int, group: str, kind: str, approx: bool = False,
                 rank: Optional[int] = None):
        self.name, self.a_ms, self.b_ms = name, a_ms, b_ms
        self.group, self.kind, self.approx, self.rank = group, kind, approx, rank


def peak_lines(path: Optional[str]) -> List[Dict[str, str]]:
    """WEG2-VRAM-PEAK-Zeilen eines Logs als dicts (+ 'rank_tag' aus dem Praefix)."""
    out = []
    if not path or not os.path.exists(path):
        return out
    with open(path, errors="replace") as f:
        for ln in f:
            if "WEG2-VRAM-PEAK " not in ln:
                continue
            m = _TS_S.match(ln)
            d = dict(_RX_KV.findall(ln.split("WEG2-VRAM-PEAK ", 1)[1]))
            d["rank_tag"] = m.group(3) if m else "?"
            out.append(d)
    return out


def p_chunk_windows(p_log: str, peaks: List[Dict[str, str]]) -> List[Window]:
    """P-Chunks: ms-genau aus WEG2-VRAM-PEAK phase=chunk (Rang 0 bestimmt das
    Fenster, die Karte jedes Rangs wird gegen dasselbe Fenster gelesen), sonst
    Sekunden aus den PP0-'Prefill batch'-Zeilen."""
    wins: List[Window] = []
    chunks = [d for d in peaks if d.get("phase") == "chunk" and d.get("t0_unix_ms", "na") != "na"
              and d.get("rank", "").isdigit()]
    if chunks:
        seen: Dict[int, int] = defaultdict(int)
        for d in chunks:
            r = int(d["rank"])
            wins.append(Window(f"P chunk {seen[r]} PP{r} rows={d.get('rows', '?')}", int(d["t0_unix_ms"]),
                               int(d["t_unix_ms"]), "P", "chunk", rank=r))
            seen[r] += 1
        return wins
    ts = []
    with open(p_log, errors="replace") as f:
        for ln in f:
            if "Prefill batch" not in ln:
                continue
            m = _TS_S.match(ln)
            if m and m.group(3) == "PP0":
                ts.append(_unix_ms(m.group(1), m.group(2)))
    for i, t in enumerate(ts):
        nxt = ts[i + 1] if i + 1 < len(ts) else t + 1000
        if nxt - t > 30_000:  # der letzte Chunk einer Anfrage: bis 5 s danach
            nxt = t + 5000
        wins.append(Window(f"P chunk {i} ~s", t - 1000, nxt + 1000, "P", "chunk", approx=True))
    return wins


def flip_windows(front_log: Optional[str]) -> List[Window]:
    if not front_log or not os.path.exists(front_log):
        return []
    begins: Dict[str, Tuple[int, str]] = {}
    wins: List[Window] = []
    with open(front_log, errors="replace") as f:
        for ln in f:
            m = _TS_MS.match(ln)
            if not m:
                continue
            t = _unix_ms(m.group(1), m.group(2), int(m.group(3)))
            if "WEG2-FLIP begin" in ln:
                mm = re.search(r"epoch=(\d+) sleep=(\w) wake=(\w)", ln)
                if mm:
                    begins[mm.group(1)] = (t, f"{mm.group(2)}->{mm.group(3)}")
            elif "WEG2-FLIP done" in ln:
                mm = re.search(r"epoch=(\d+)", ln)
                if mm and mm.group(1) in begins:
                    t0, d = begins.pop(mm.group(1))
                    wins.append(Window(f"flip {mm.group(1)} {d}", t0, t, "PD", "flip"))
    return wins


def decode_windows(flips: List[Window]) -> List[Window]:
    out = []
    for i, w in enumerate(flips):
        if w.name.endswith("P->D"):
            nxt = next((f for f in flips[i + 1:] if f.a_ms > w.b_ms), None)
            if nxt is not None:
                out.append(Window(f"D decode {w.name.split()[1]}", w.b_ms, nxt.a_ms, "D", "decode"))
    return out


_RX_PKARTE = re.compile(r"P-KARTE stage(\d+) \(nvml(\d+)\).*?-> Kopfraum (-?\d+) MiB \(near-OOM (\d+)\)")
_RX_PTRANS = re.compile(r"P-KARTE stage(\d+) \(nvml(\d+)\).*?Transiente (\d+)")
_RX_DKARTE = re.compile(r"KARTE D\(dry, expectation\) rang(\d+):.*?-> Kopfraum (-?\d+) MiB \(near-OOM (\d+)\)")


def planner_card(dry_log: Optional[str], rank_cards: Sequence[int]) -> Dict[Tuple[str, int], Dict[str, int]]:
    """{('P'|'D', nvml): {kopfraum, near_oom, transient?}} -- die LETZTE Zeile je
    Schluessel gilt (ein dry-Log kann mehrere Arme tragen)."""
    out: Dict[Tuple[str, int], Dict[str, int]] = {}
    if not dry_log or not os.path.exists(dry_log):
        return out
    with open(dry_log, errors="replace") as f:
        for ln in f:
            m = _RX_PKARTE.search(ln)
            if m:
                d = {"kopfraum": int(m.group(3)), "near_oom": int(m.group(4))}
                mt = _RX_PTRANS.search(ln)
                if mt:
                    d["transient"] = int(mt.group(3))
                out[("P", int(m.group(2)))] = d
                continue
            m = _RX_DKARTE.search(ln)
            if m:
                r = int(m.group(1))
                if r < len(rank_cards):
                    out[("D", rank_cards[r])] = {"kopfraum": int(m.group(2)), "near_oom": int(m.group(3))}
    return out


def dmon_series(dmon_log: Optional[str], day: str) -> Dict[int, List[Tuple[int, int]]]:
    """{nvml: [(unix_ms, fb_mib)]} aus ``nvidia-smi dmon -s tm -o T`` (fb = 5. Spalte)."""
    out: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    if not dmon_log or not os.path.exists(dmon_log):
        return out
    fb_col = 4
    with open(dmon_log, errors="replace") as f:
        for ln in f:
            if ln.startswith("#"):
                cols = ln[1:].split()
                if "fb" in cols:
                    fb_col = cols.index("fb")
                continue
            p = ln.split()
            if len(p) <= fb_col or not re.match(r"\d\d:\d\d:\d\d$", p[0]):
                continue
            try:
                out[int(p[1])].append((_unix_ms(day, p[0]), int(p[fb_col])))
            except ValueError:
                continue
    return out


def dmon_max(ser: List[Tuple[int, int]], a_ms: int, b_ms: int) -> Optional[int]:
    # eine dmon-Zeile HH:MM:SS steht fuer die ganze Sekunde, in der sie las
    vals = [v for t, v in ser if t + 999 >= a_ms and t <= b_ms]
    return max(vals) if vals else None


# --------------------------------------------------------------------------
# Rechnung
# --------------------------------------------------------------------------


def card_rows(tr: Trace, wins: List[Window], plan, dmon, rank_cards) -> List[Dict]:
    rows = []
    for w in wins:
        cards = sorted(tr.cards) if w.kind != "chunk" or w.rank is None else [rank_cards[w.rank]]
        for c in cards:
            mx = tr.window_max(f"c{c}", w.a_ms, w.b_ms)
            if mx is None:
                continue
            used, at = mx
            rest = tr.free_of(c, used)
            dm = dmon_max(dmon.get(c, []), w.a_ms, w.b_ms) if dmon else None
            proc = None
            for key in tr.series:
                if key.startswith(f"c{c}:") and not key.endswith(":other"):
                    pm = tr.window_max(key, w.a_ms, w.b_ms)
                    if pm is not None and (proc is None or pm[0] > proc[1]):
                        proc = (tr.roles.get(key, key), pm[0])
            grp = w.group if w.group in ("P", "D") else None
            pl = plan.get((grp, c)) if grp else None
            verdict = "-"
            if rest is not None:
                edge = pl["near_oom"] if pl else 400
                verdict = ("UNTER near-OOM" if rest < edge else "KNAPP" if rest < 2 * edge else "ok")
            rows.append({
                "phase": w.name, "win_s": (w.b_ms - w.a_ms) / 1000.0, "card": c,
                "true_max": used, "rest_min": rest, "at_s": (at - w.a_ms) / 1000.0,
                "dmon_max": dm, "missed": None if dm is None else used - dm,
                "proc": proc, "plan": pl["kopfraum"] if pl else None,
                "plan_minus_rest": (pl["kopfraum"] - rest) if (pl and rest is not None) else None,
                "verdict": verdict,
            })
    return rows


_RX_POOL_PF = re.compile(r"WEG2-GRAPH-POOL rank=(\d+) phase=\S+ .*?private_free_mib=(-?\d+)")


def private_free_by_rank(path: Optional[str]) -> Dict[str, int]:
    """H59: ``private_free_mib`` je Rang aus den WEG2-GRAPH-POOL-Zeilen eines
    Logs (der letzte Wert; im Boot konstant) -- der dritte Term des Planer-
    Kopfraums ``cap - peak - privat_frei``."""
    out: Dict[str, int] = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, errors="replace") as f:
        for ln in f:
            if "WEG2-GRAPH-POOL " not in ln:
                continue
            m = _RX_POOL_PF.search(ln)
            if m and int(m.group(2)) >= 0:
                out[m.group(1)] = int(m.group(2))
    return out


def _num(d: Dict[str, str], fld: str) -> Optional[int]:
    v = d.get(fld, "na")
    if v in ("na", ""):
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def inproc_rows(tr: Optional[Trace], peaks_by_group: Dict[str, List[Dict[str, str]]], plan,
                rank_cards, private_free: Optional[Dict[str, Dict[str, int]]] = None) -> List[Dict]:
    """Je (Gruppe, Rang, Phase) die In-Prozess-Maxima. H59: ``transient`` ist
    H55s ``peak - start``; ``tr_plan`` = ``peak - allocated NACH dem Fenster``
    ist die PLANER-Definition (``[vram-peak]``, ``PP-CUT ACTIVATION``), und
    ``persist`` = ``allocated nach - start`` ist, was das Fenster liegen laesst
    (im Planer das Chunk-WACHSTUM). ``kopf_torch`` = ``card_free + reserved -
    peak_allocated - privat_frei`` am Fensterende, das Minimum -- der Kopfraum
    in der Waehrung der P-/D-KARTE, gegen den near-OOM gilt."""
    agg: Dict[Tuple[str, str, str], Dict] = {}
    pf_all = private_free or {}
    for grp, peaks in peaks_by_group.items():
        pf = pf_all.get(grp, {})
        for d in peaks:
            ph = d.get("phase", "?") + (f":{d['leg']}" if "leg" in d else "")
            key = (grp, d.get("rank", "?"), ph)
            a = agg.setdefault(key, {"n": 0, "peak_reserved": 0, "transient": 0, "card_free_min": None,
                                     "nvml_proc_max": None, "tr_plan": None, "persist": None,
                                     "kopf_torch_min": None, "retries": None, "ooms": None})
            a["n"] += 1
            # H59: allocator retries/OOMs of the window (delta; 'na' = no start)
            for fld, dst in (("alloc_retries", "retries"), ("ooms", "ooms")):
                v = _num(d, fld)
                if v is not None:
                    a[dst] = (a[dst] or 0) + v
            for fld, dst in (("peak_reserved_mib", "peak_reserved"), ("transient_mib", "transient")):
                v = d.get(fld, "na")
                if v != "na":
                    a[dst] = max(a[dst], int(float(v)))
            pk, st, en = _num(d, "peak_allocated_mib"), _num(d, "start_allocated_mib"), _num(d, "allocated_mib")
            if pk is not None and en is not None:
                a["tr_plan"] = max(a["tr_plan"] or 0, pk - en)
                if st is not None:
                    a["persist"] = en - st if a["persist"] is None else max(a["persist"], en - st)
            cf, rs = _num(d, "card_free_mib"), _num(d, "reserved_mib")
            if pk is not None and cf is not None and rs is not None and d.get("rank", "?") in pf:
                kopf = cf + rs - pk - pf[d["rank"]]
                a["kopf_torch_min"] = kopf if a["kopf_torch_min"] is None else min(a["kopf_torch_min"], kopf)
            for fld in ("card_free_mib", "card_free_start_mib"):
                v = d.get(fld, "na")
                if v != "na":
                    a["card_free_min"] = int(float(v)) if a["card_free_min"] is None else min(a["card_free_min"], int(float(v)))
            if tr is not None and d.get("t0_unix_ms", "na") != "na":
                try:
                    card = rank_cards[int(d["rank"])]
                except (ValueError, IndexError):
                    continue
                role = f"{grp}:{'PP' if grp == 'P' else 'TP'}{d['rank']}"
                for key, r in tr.roles.items():
                    if r == role and key.startswith(f"c{card}:"):
                        pm = tr.window_max(key, int(d["t0_unix_ms"]), int(d["t_unix_ms"]))
                        if pm is not None:
                            a["nvml_proc_max"] = pm[0] if a["nvml_proc_max"] is None else max(a["nvml_proc_max"], pm[0])
    rows = []
    for (grp, rank, ph), a in sorted(agg.items()):
        try:
            card = rank_cards[int(rank)]
        except (ValueError, IndexError):
            card = None
        pl = plan.get((grp, card)) if card is not None else None
        rows.append({"group": grp, "rank": rank, "phase": ph, "card": card, **a,
                     "plan_transient": pl.get("transient") if pl else None,
                     "ausser_torch": (a["nvml_proc_max"] - a["peak_reserved"]) if a["nvml_proc_max"] is not None else None})
    return rows


def _f(v, w=7):
    if v is None:
        return "-".rjust(w)
    if isinstance(v, float):
        return f"{v:.2f}".rjust(w)
    return str(v).rjust(w)


def render(tr: Trace, crow: List[Dict], irow: List[Dict]) -> str:
    out = []
    out.append(f"VRAM-HIRES  {tr.calib}")
    if tr.end:
        out.append(f"            {tr.end}")
    out.append(f"            gaps={tr.gaps}  cards=" + ", ".join(
        f"nvml{c} {v['name']} total {v['total_mib']} offset {v['offset_mib']}" for c, v in sorted(tr.cards.items())))
    out.append("")
    out.append(f"{'phase':30} {'win_s':>7} {'card':>5} {'true_max':>8} {'rest_min':>8} {'at_s':>7} "
               f"{'dmon_max':>8} {'missed':>7} {'proc(max)':>22} {'plan':>6} {'plan-rest':>9}  verdict")
    for r in crow:
        proc = f"{r['proc'][0]} {r['proc'][1]}" if r["proc"] else "-"
        out.append(f"{r['phase'][:30]:30} {_f(r['win_s'])} {('nvml' + str(r['card'])):>5} {_f(r['true_max'], 8)} "
                   f"{_f(r['rest_min'], 8)} {_f(r['at_s'])} {_f(r['dmon_max'], 8)} {_f(r['missed'])} "
                   f"{proc:>22} {_f(r['plan'], 6)} {_f(r['plan_minus_rest'], 9)}  {r['verdict']}")
    if irow:
        out.append("")
        out.append("IN-PROZESS (WEG2-VRAM-PEAK, torch-Allokator je Fenster)")
        out.append(f"{'grp':3} {'rank':>4} {'phase':16} {'card':>5} {'n':>4} {'peak_res':>8} {'transient':>9} "
                   f"{'plan_tr':>7} {'tr_plan':>7} {'persist':>7} {'kopf_torch':>10} "
                   f"{'free_min':>8} {'nvml_proc':>9} {'ausser_torch':>12} {'retries':>7} {'ooms':>4}")
        for r in irow:
            out.append(f"{r['group']:3} {r['rank']:>4} {r['phase'][:16]:16} "
                       f"{('nvml' + str(r['card'])) if r['card'] is not None else '-':>5} {r['n']:>4} "
                       f"{_f(r['peak_reserved'], 8)} {_f(r['transient'], 9)} {_f(r['plan_transient'])} "
                       f"{_f(r.get('tr_plan'))} {_f(r.get('persist'))} {_f(r.get('kopf_torch_min'), 10)} "
                       f"{_f(r['card_free_min'], 8)} {_f(r['nvml_proc_max'], 9)} {_f(r['ausser_torch'], 12)} "
                       f"{_f(r.get('retries'))} {_f(r.get('ooms'), 4)}")
        out.append("  transient = peak - START (H55); tr_plan = peak - allocated NACH dem Fenster (die "
                   "Planer-Definition, gegen plan_tr zu lesen); persist = was das Fenster liegen laesst "
                   "(im Planer das Chunk-Wachstum); kopf_torch = card_free + reserved - peak - privat_frei "
                   "(Minimum, die Waehrung der P-/D-KARTE). rest_min oben ist die Karte NACH dem "
                   "Allokator-Cache: fnFL2x165 lief alle sechs 16k-Chunks mit rest_min 7 MiB auf PP0.")
    return "\n".join(out)


def raw_path_of(csv_path: str) -> str:
    if csv_path.endswith(".raw.csv"):
        return csv_path
    return (csv_path[:-4] if csv_path.endswith(".csv") else csv_path) + ".raw.csv"


def _sibling(p_log: str, kind: str) -> Optional[str]:
    cand = re.sub(r"\.P(\.[\w]+)?\.log$", f".{kind}\\1.log", p_log)
    return cand if cand != p_log and os.path.exists(cand) else None


def _rank_cards_from_json(tag: Optional[str]) -> Optional[List[int]]:
    if not tag:
        return None
    try:
        with open(f"/spinning/gpu-arb/weg2/boot_{tag}.json") as f:
            return [int(c["nvml_index"]) for c in json.load(f)["cards"]]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def report(csv_path: str, p_log: str, d_log: str, front: Optional[str] = None, dry: Optional[str] = None,
           dmon: Optional[str] = None, rank_cards: Optional[Sequence[int]] = None) -> str:
    tr = load_trace(raw_path_of(csv_path))
    tag = None
    m = re.search(r"vram_hires_(.+?)\.(raw\.)?csv$", os.path.basename(csv_path))
    if m:
        tag = m.group(1)
    rank_cards = list(rank_cards or _rank_cards_from_json(tag) or DEFAULT_RANK_CARDS)
    front = front or _sibling(p_log, "front")
    if dry is None and tag:
        cand = os.path.join(os.path.dirname(csv_path), f"dry_{tag}.log")
        dry = cand if os.path.exists(cand) else None
    if dmon is None and tag:
        cand = os.path.join(os.path.dirname(csv_path), f"vram_{tag}.log")
        dmon = cand if os.path.exists(cand) else None
    day = _dt.datetime.fromtimestamp(tr.t0_unix_ms / 1000, _dt.timezone.utc).strftime("%Y-%m-%d")
    p_peaks, d_peaks = peak_lines(p_log), peak_lines(d_log)
    flips = flip_windows(front)
    wins = p_chunk_windows(p_log, p_peaks) + flips + decode_windows(flips)
    if tr.series:
        a = min(s[0][0] for s in tr.series.values())
        b = max(s[0][-1] for s in tr.series.values())
        if tr.end_t_ms is not None:  # die Werte stehen bis zum Ende der Sonde
            b = max(b, tr.unix(tr.end_t_ms))
        wins.append(Window("Boot (ganzer Rohstrom)", a, b, "-", "boot"))
    wins.sort(key=lambda w: (w.kind == "boot", w.a_ms))
    plan = planner_card(dry, rank_cards)
    crow = card_rows(tr, wins, plan, dmon_series(dmon, day), rank_cards)
    irow = inproc_rows(tr, {"P": p_peaks, "D": d_peaks}, plan, rank_cards,
                       private_free={"P": private_free_by_rank(p_log), "D": private_free_by_rank(d_log)})
    return render(tr, crow, irow)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv")
    ap.add_argument("p_log")
    ap.add_argument("d_log")
    ap.add_argument("--front")
    ap.add_argument("--dry")
    ap.add_argument("--dmon")
    ap.add_argument("--cards", help="NVML je Rang, z.B. 1,0,2 (Standard: boot_<tag>.json, sonst 1,0,2)")
    ns = ap.parse_args(argv)
    rc = [int(x) for x in ns.cards.split(",")] if ns.cards else None
    print(report(ns.csv, ns.p_log, ns.d_log, front=ns.front, dry=ns.dry, dmon=ns.dmon, rank_cards=rc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
