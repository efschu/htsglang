# SPDX-License-Identifier: Apache-2.0
"""D->P-Flip je PP-Stufe: wann ist welches Band da, wer bindet, was brächte PP0-zuerst.

    python -m sglang.srt.weg2.tools.dp_stage_legs <boot>.front.log [-v]

Liest ``<boot>.front.log`` und die Geschwister ``.P.log``/``.D.log`` (zeilenweise,
nie ganz in den Speicher). Je D->P-Flip (``WEG2-FLIP begin ... sleep=D wake=P``
bis ``WEG2-FLIP done``), alle Zeiten in ms ab dem Leg-Start (kleinstes ``t0``
der ``WEG2-WAKE-TAG-TIME``-Zeilen der P-Ränge):

* je P-Stufe: Karte, Bytes und Tags ihres Bandes (``WEG2-FLIP-TAG group=P
  dir=h2d bytes>0``), ``resume_end`` (letztes ``t`` einer eigenen
  ``WEG2-WAKE-TAG-TIME``: das Mapping, NICHT die Bytes), ``collects_end``
  (Leg-Start + ``leg_collects`` der ``WEG2-WAKE-TAIL``-Zeile: die Bytes) und
  ``band_ready`` = das letzte D-Deposit eines ihrer Tags über alle D-Ränge
  (``WEG2-SLEEP-TAG-TIME t``) -- vorher kann das Band nicht vollständig sein;
  der Schritt eines D-Rangs ohne Stück dieses Tags (``WEG2-XCHG DEPOSIT ...
  pieces=0``) zählt nicht mit;
* je D-Rang: Ende seines Schlaf-Legs, Summe ``deposit_ms`` und ``pause_ms``;
* ``crit``: der D-Rang mit dem spätesten Leg-Ende (er bindet den Flip, solange
  der Wake nur auf seine Deposits wartet).

WAS-WÄRE-WENN (``what_if``): je D-Rang sind die Kosten eines Tags
(``total_ms`` seiner SLEEP-TAG-TIME-Zeile) in dieser Messung reihenfolge-
unabhängig angenommen -- das Modell rechnet die gemessene Reihenfolge nach
(``V0``, muss ``band_ready`` treffen) und zwei Umordnungen:

* ``V1``: die Bänder von PP0 zuerst, dann PP1, dann PP2, der Basis-Tag
  (``weights``: Einbettung auf PP0, Kopf auf der letzten Stufe) bleibt LETZTER
  -- der Vertrag von ``weights_family_tags`` / ``derive_waves``;
* ``V2``: wie V1, aber der Basis-Tag direkt hinter PP0s Bändern -- verletzt den
  Vertrag, nur als obere Schranke.

Der Hebel "P beginnt den ersten Prefill-Chunk, sobald PP0 sein Band hat" spart
höchstens ``legs_end - band_ready(PP0)`` der jeweiligen Variante; er trägt nur,
wenn PP1/PP2 ihr Band vor der Ankunft der ersten Aktivierung haben
(``band_ready(PP1) <= band_ready(PP0) + erster PP0-Chunk``). Die ersten
Prefill-Chunks je Stufe nach dem Flip (``Prefill rank batch ... gpu-ms``)
stehen deshalb mit im Bericht.

Nur ein Leser. Nichts hier steuert etwas.
"""

from __future__ import annotations

import datetime as _dt
import math
import re
import statistics
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence

BASE_TAG = "weights"
STAGES = ("PP0", "PP1", "PP2")

_RX_FRONT_TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d+)\]")
_RX_RANK_TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (PP\d|TP\d)\]")
_RX_SLEEP_TAG = re.compile(
    r"\[\S+ \S+ (TP\d)\] WEG2-SLEEP-TAG-TIME tag=(\S+) deposit_ms=(\d+) sync_ms=\d+ "
    r"pause_ms=(\d+) .*?total_ms=(\d+) t0=([\d.]+) t=([\d.]+)")
_RX_WAKE_TAG = re.compile(r"\[\S+ \S+ (PP\d)\] WEG2-WAKE-TAG-TIME tag=(\S+) .*?t0=([\d.]+) t=([\d.]+)")
_RX_FLIP_TAG = re.compile(
    r"\[\S+ \S+ (PP\d)\] WEG2-FLIP-TAG group=P rank=\d+ card=(\S+) dir=h2d tag=(\S+) bytes=(\d+) MiB")
_RX_WAKE_TAIL = re.compile(
    r"\[\S+ \S+ (PP\d)\] WEG2-WAKE-TAIL ms pre_leg=\d+ read_early=\d+ leg_collects=(\d+) .*?t=([\d.]+)")
_RX_NOOP_DEPOSIT = re.compile(r"\] WEG2-XCHG DEPOSIT tag=(\S+) group=D rank=(\d+) pieces=0 ")
_RX_PREFILL = re.compile(
    r"\[(\S+ \S+) (PP\d)\] Prefill rank batch, #new-token: (\d+), #cached-token: (\d+).*?gpu-ms: ([\d.]+)")


def _front_ts(line: str) -> Optional[float]:
    m = _RX_FRONT_TS.match(line)
    if m is None:
        return None
    base = _dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=_dt.timezone.utc)
    return base.timestamp() + int(m.group(2)[:3].ljust(3, "0")) / 1000.0


def _rank_ts(line: str) -> Optional[float]:
    m = _RX_RANK_TS.match(line)
    if m is None:
        return None
    base = _dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=_dt.timezone.utc)
    return base.timestamp()


def flips_from_front(lines: Iterable[str]) -> List[dict]:
    """D->P flips as ``{"begin", "done"}`` (epoch seconds, ms-genau)."""
    out: List[dict] = []
    cur: Optional[dict] = None
    for line in lines:
        if "WEG2-FLIP begin" in line:
            t = _front_ts(line)
            cur = {"begin": t, "dp": "sleep=D wake=P" in line} if t is not None else None
        elif "WEG2-FLIP done" in line and cur is not None:
            t = _front_ts(line)
            if cur["dp"] and t is not None:
                out.append({"begin": cur["begin"], "done": t})
            cur = None
    return out


def _flip_of(flips: Sequence[dict], t: float, slack: float = 0.5) -> Optional[dict]:
    for f in flips:
        if f["begin"] - slack <= t <= f["done"] + slack:
            return f
    return None


def attach_rank_logs(flips: List[dict], p_lines: Iterable[str], d_lines: Iterable[str]) -> None:
    """Fills each flip in place with the P wake and D sleep per-tag records."""
    for line in d_lines:
        if "WEG2-XCHG DEPOSIT" in line and "pieces=0 " in line:
            m = _RX_NOOP_DEPOSIT.search(line)
            t = _rank_ts(line)
            if m is not None and t is not None:
                # second-resolution rank clock: the step belongs to the flip
                # whose window holds it (one second of slack)
                f = _flip_of(flips, t, slack=1.0)
                if f is not None:
                    f.setdefault("noop", set()).add(("TP" + m.group(2), m.group(1)))
            continue
        if "WEG2-SLEEP-TAG-TIME" not in line:
            continue
        m = _RX_SLEEP_TAG.search(line)
        if m is None:
            continue
        f = _flip_of(flips, float(m.group(7)))
        if f is None:
            continue
        f.setdefault("d", defaultdict(list))[m.group(1)].append({
            "tag": m.group(2), "deposit_ms": int(m.group(3)), "pause_ms": int(m.group(4)),
            "total_ms": int(m.group(5)), "t0": float(m.group(6)), "t": float(m.group(7))})
    last_prefill_flip: Dict[str, int] = {}
    for line in p_lines:
        if "WEG2-WAKE-TAG-TIME" in line:
            m = _RX_WAKE_TAG.search(line)
            if m is None:
                continue
            f = _flip_of(flips, float(m.group(4)))
            if f is not None:
                f.setdefault("p", defaultdict(list))[m.group(1)].append(
                    {"tag": m.group(2), "t0": float(m.group(3)), "t": float(m.group(4))})
        elif "WEG2-FLIP-TAG group=P" in line and "dir=h2d" in line:
            m = _RX_FLIP_TAG.search(line)
            if m is None:
                continue
            stage = m.group(1)
            # the census lines follow their own leg: the newest flip whose leg
            # this stage ran and that has no entry for the tag yet
            for f in reversed(flips):
                if stage in f.get("p", {}):
                    own = f.setdefault("bytes", defaultdict(dict))[stage]
                    if m.group(3) not in own:
                        own[m.group(3)] = int(m.group(4))
                        f.setdefault("card", {})[stage] = m.group(2)
                    break
        elif "WEG2-WAKE-TAIL ms pre_leg=" in line:
            m = _RX_WAKE_TAIL.search(line)
            if m is None:
                continue
            f = _flip_of(flips, float(m.group(3)))
            if f is not None:
                f.setdefault("leg_collects", {})[m.group(1)] = float(m.group(2))
        elif "Prefill rank batch" in line:
            m = _RX_PREFILL.search(line)
            t = _rank_ts(line)
            if m is None or t is None:
                continue
            stage = m.group(2)
            # the first prefill of each stage after a flip's done (second-
            # resolution rank clock: one second of slack before done)
            for i, f in enumerate(flips):
                nxt = flips[i + 1]["done"] if i + 1 < len(flips) else math.inf
                if f["done"] - 1.0 <= t < nxt:
                    if last_prefill_flip.get(stage) != i:
                        last_prefill_flip[stage] = i
                        f.setdefault("first_chunk_ms", {})[stage] = float(m.group(5))
                        f.setdefault("first_chunk_tokens", {})[stage] = int(m.group(3))
                    break


def own_tags(flip: dict) -> Dict[str, set]:
    """Per P stage the tags that carried bytes into it (its BAND)."""
    return {s: {t for t, b in flip.get("bytes", {}).get(s, {}).items() if b > 0}
            for s in flip.get("p", {})}


def decompose(flip: dict) -> Optional[dict]:
    """One flip's per-stage / per-D-rank timing (ms from the leg start)."""
    p, d = flip.get("p"), flip.get("d")
    if not p or not d or not flip.get("bytes"):
        return None
    t0 = min(min(x["t0"] for x in v) for v in p.values())
    own = own_tags(flip)
    noop = flip.get("noop", set())
    # a rank's step for a tag it holds no piece of (``pieces=0``) gates no band
    d_end = {r: {x["tag"]: (x["t"] - t0) * 1000.0 for x in v if (r, x["tag"]) not in noop}
             for r, v in d.items()}
    out = {"t0": t0, "leg_start_from_begin": (t0 - flip["begin"]) * 1000.0,
           "done_from_leg_start": (flip["done"] - t0) * 1000.0,
           "stages": {}, "d": {}}
    for s, v in p.items():
        mine = own.get(s, set())
        ends = [x["t"] for x in v if x["tag"] in mine]
        ready = [e[t] for e in d_end.values() for t in mine if t in e]
        rec = {"card": flip.get("card", {}).get(s, "?"),
               "bytes_mib": sum(flip["bytes"].get(s, {}).values()),
               "tags": len(mine),
               "resume_end": (max(ends) - t0) * 1000.0 if ends else None,
               "band_ready": max(ready) if ready else None}
        lc = flip.get("leg_collects", {}).get(s)
        rec["collects_end"] = (min(x["t0"] for x in v) - t0) * 1000.0 + lc if lc is not None else None
        out["stages"][s] = rec
    for r, v in d.items():
        out["d"][r] = {"leg_end": (max(x["t"] for x in v) - t0) * 1000.0,
                       "deposit_ms": sum(x["deposit_ms"] for x in v),
                       "pause_ms": sum(x["pause_ms"] for x in v)}
    out["legs_end"] = max(x["leg_end"] for x in out["d"].values())
    out["crit"] = max(out["d"], key=lambda r: out["d"][r]["leg_end"])
    return out


def _variant_orders(order: Sequence[str], own: Dict[str, set]) -> Dict[str, List[str]]:
    base = [t for t in order if t == BASE_TAG]
    chunks = [t for t in order if t != BASE_TAG]
    groups: List[List[str]] = []
    taken: set = set()
    for s in STAGES:
        g = [t for t in chunks if t in own.get(s, set()) and t not in taken]
        taken.update(g)
        groups.append(g)
    rest = [t for t in chunks if t not in taken]
    v1 = [t for g in groups for t in g] + rest + base
    v2 = groups[0] + base + [t for g in groups[1:] for t in g] + rest
    return {"V0": list(order), "V1": v1, "V2": v2}


def what_if(flip: dict) -> Optional[Dict[str, Dict[str, float]]]:
    """band_ready per stage for V0 (measured order, the model's check), V1
    (PP0 first, base last) and V2 (PP0 first, base right behind PP0's bands).

    Per D rank each tag costs its measured ``total_ms``; the rank starts where
    it started in this flip. A band is ready when every D rank has deposited
    its last tag of the band (a ``pieces=0`` step still costs its chain time
    but gates no band)."""
    p, d = flip.get("p"), flip.get("d")
    if not p or not d or not flip.get("bytes"):
        return None
    t0 = min(min(x["t0"] for x in v) for v in p.values())
    ref = min(d, key=lambda r: min(x["t0"] for x in d[r]))
    order = [x["tag"] for x in sorted(d[ref], key=lambda x: x["t0"])]
    own = own_tags(flip)
    cost = {r: {x["tag"]: x["total_ms"] for x in v} for r, v in d.items()}
    start = {r: (min(x["t0"] for x in v) - t0) * 1000.0 for r, v in d.items()}
    noop = flip.get("noop", set())
    out: Dict[str, Dict[str, float]] = {}
    for name, o in _variant_orders(order, own).items():
        res: Dict[str, float] = {}
        for s in own:
            ends = []
            for r in cost:
                idx = [i for i, t in enumerate(o) if t in own[s] and (r, t) not in noop]
                if idx:
                    ends.append(start[r] + sum(cost[r].get(t, 0) for t in o[: max(idx) + 1]))
            if ends:
                res[s] = max(ends)
        out[name] = res
    return out


def analyze(stem: str) -> List[dict]:
    with open(stem + ".front.log", errors="replace") as fh:
        flips = flips_from_front(fh)
    with open(stem + ".P.log", errors="replace") as fp, open(stem + ".D.log", errors="replace") as fd:
        attach_rank_logs(flips, fp, fd)
    rows = []
    for f in flips:
        dec = decompose(f)
        if dec is None:
            continue
        dec["what_if"] = what_if(f)
        dec["first_chunk_ms"] = f.get("first_chunk_ms", {})
        rows.append(dec)
    return rows


def _q(xs: Sequence[float]) -> str:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return "      -/      -"
    return "%7.0f/%7.0f" % (statistics.median(xs), xs[max(1, math.ceil(0.9 * len(xs))) - 1])


def report(rows: List[dict]) -> str:
    """The table: median/p90 per stage and D rank, and the what-if."""
    out = ["DP-STAGE-LEGS n=%d (ms ab Leg-Start, median/p90)" % len(rows)]
    crit: Dict[str, int] = defaultdict(int)
    for r in rows:
        crit[r["crit"]] += 1
    out.append("  leg_start_ab_begin %s  legs_end %s  done %s  crit=%s" % (
        _q([r["leg_start_from_begin"] for r in rows]), _q([r["legs_end"] for r in rows]),
        _q([r["done_from_leg_start"] for r in rows]), dict(crit)))
    stages = sorted({s for r in rows for s in r["stages"]})
    for s in stages:
        recs = [r["stages"][s] for r in rows if s in r["stages"]]
        cards = sorted({x["card"][:12] for x in recs})
        out.append("  %s card=%s MiB=%s tags=%s resume_end %s collects_end %s band_ready %s first_chunk %s" % (
            s, ",".join(cards), _q([x["bytes_mib"] for x in recs]).split("/")[0].strip(),
            _q([x["tags"] for x in recs]).split("/")[0].strip(),
            _q([x["resume_end"] for x in recs]), _q([x["collects_end"] for x in recs]),
            _q([x["band_ready"] for x in recs]), _q([r["first_chunk_ms"].get(s) for r in rows])))
    for rk in sorted({k for r in rows for k in r["d"]}):
        recs = [r["d"][rk] for r in rows if rk in r["d"]]
        out.append("  D-%s leg_end %s deposit %s pause %s" % (
            rk, _q([x["leg_end"] for x in recs]), _q([x["deposit_ms"] for x in recs]),
            _q([x["pause_ms"] for x in recs])))
    wi = [r["what_if"] for r in rows if r.get("what_if")]
    for v in ("V0", "V1", "V2"):
        parts = []
        for s in stages:
            parts.append("%s %s" % (s, _q([w[v].get(s) for w in wi if v in w])))
        gain = [r["legs_end"] - r["what_if"][v]["PP0"] for r in rows
                if r.get("what_if") and "PP0" in r["what_if"].get(v, {})]
        out.append("  WHAT-IF %s band_ready %s  PP0-Gewinn %s" % (v, "  ".join(parts), _q(gain)))
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    paths = [a for a in args if not a.startswith("-")]
    if not paths:
        print(__doc__)
        return 2
    for path in paths:
        stem = path[: -len(".front.log")] if path.endswith(".front.log") else path
        print(report(analyze(stem)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
