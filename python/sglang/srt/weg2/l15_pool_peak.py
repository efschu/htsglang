# SPDX-License-Identifier: Apache-2.0
"""L1.5 pool, stage S1 part B: the per-card record ``P_AWAKE_PEAK_MIB``.

L15-POOL-ENTWURF-1004 section 3.4 / Q4: the pool may only include the free VRAM
of a card (the 5090 included, which holds 0 MiB today) from the MEASURED P awake
peak of that card -- never from the minimum of a sample, never from an estimate.
This module is the pure half of that record:

* :func:`build_record` reads ``WEG2-VRAM-PEAK`` lines of P group logs (H55,
  ``model_executor/vram_peak_window.py``) and keeps, per card (= P rank = budget
  ordinal), the MAXIMUM of the measured windows over every boot and every line --
  not a mean, not a percentile -- with its origin (boot, wall time, line number,
  number of lines read).  The primary basis is ``peak_reserved_mib`` (the
  allocator's reserved peak, i.e. what the P budget has to hold); the
  ``peak_allocated_mib`` maximum is carried beside it.
* :func:`planner_peaks` is the planner input behind ``SGLANG_WEG2_L15_POOL_PEAK_RECORD``
  (default OFF).  Off: the legacy peak vector goes through untouched and no line
  is printed (the planner stays byte for byte today's).  On: the pool record is
  the ONLY source of the peaks; a card without a record gets ``None`` -- which
  ``l15_plan.l15_post_mib`` prices as UNMEASURED 0, i.e. NO pool share on that
  card -- and each card prints ``L15-POOL-PEAK card=.. peak_mib=.. source=.. n=..``.

What the maximum is NOT: ``WEG2-VRAM-PEAK`` windows are closed by the P forwards
and by flip legs; the peak is exact for torch allocations inside a window, and
blind to non-torch memory (CUDA context, NCCL, cuMemMap arenas).  Those sit
outside the P budget line (carve / corridor floor), the unit ``l15_post_mib``
subtracts from.  The record is a lower bound of what any future P can need (it
is the record of what was measured): a P that exceeds it holds less at the next
sleep (F5 of the plan), it is never negative.

PURE: stdlib only; no launcher, no torch, no GPU.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

RECORD_NAME = "P_AWAKE_PEAK_MIB"
SCHEMA = 1
SWITCH_ENV = "SGLANG_WEG2_L15_POOL_PEAK_RECORD"
FILE_ENV = "SGLANG_WEG2_L15_POOL_PEAK_RECORD_FILE"
MARKER = "WEG2-VRAM-PEAK"
LINE_MARKER = "L15-POOL-PEAK"

#: windows that are P awake work (a chunk, decode/verify rounds, an idle awake
#: window); the flip legs are transitions, not awake P work
DEFAULT_PHASES = ("chunk", "round", "idle")
BASES = ("peak_reserved_mib", "peak_allocated_mib")
DEFAULT_BASIS = "peak_reserved_mib"

#: tolerance (MiB) between the card total a record was measured on and the card
#: the launch sees; a different card (or card order) is never priced from it
TOTAL_TOLERANCE_MIB = 64

_ON_VALUES = ("1", "true", "on")
_RX_LINE = re.compile(r"WEG2-VRAM-PEAK rank=(\d+) phase=([a-z_]+)\b")
_RX_KV = re.compile(r"(\w+)=(\S+)")
_RX_TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")

RECORDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile_records_data")


def switch_on(env: Mapping[str, str]) -> bool:
    """``SGLANG_WEG2_L15_POOL_PEAK_RECORD``; default OFF (anything but 1/true/on)."""
    return str(env.get(SWITCH_ENV, "") or "").strip().lower() in _ON_VALUES


def _int(v: Any) -> Optional[int]:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def parse_peak_line(line: str) -> Optional[Dict[str, Any]]:
    """One ``WEG2-VRAM-PEAK`` line -> ``{rank, phase, ts, <key>: int}`` or None.

    Fields that are ``na`` come back as None; a line without a rank or a phase
    is not a record line.
    """
    if MARKER not in line:
        return None
    m = _RX_LINE.search(line)
    if not m:
        return None
    kv = dict(_RX_KV.findall(line[m.start():]))
    ts = _RX_TS.match(line)
    out: Dict[str, Any] = {
        "rank": int(m.group(1)),
        "phase": m.group(2),
        "ts": ts.group(1) if ts else "",
    }
    for key in ("peak_allocated_mib", "peak_reserved_mib", "card_total_mib", "card_free_mib"):
        out[key] = _int(kv.get(key))
    return out


def scan_lines(boot: str, lines: Iterable[str],
               phases: Sequence[str] = DEFAULT_PHASES) -> Dict[int, Dict[str, Any]]:
    """One boot's lines -> per rank the maxima, the origin lines and the counts.

    ``n_lines`` counts the record lines of the selected phases that carry a
    reserved peak (the denominator of the maximum); ``n_skipped`` the record
    lines left out (other phase, or no peak).
    """
    want = set(phases)
    per: Dict[int, Dict[str, Any]] = {}
    for lineno, line in enumerate(lines, 1):
        if MARKER not in line:
            continue
        rec = parse_peak_line(line)
        if rec is None:
            continue
        st = per.setdefault(rec["rank"], {
            "n_lines": 0, "n_skipped": 0, "card_total_mib": None,
            "peak_reserved_mib": None, "peak_allocated_mib": None,
            "min_card_free_mib": None,
        })
        if rec["phase"] not in want or rec["peak_reserved_mib"] is None:
            st["n_skipped"] += 1
            continue
        st["n_lines"] += 1
        if rec["card_total_mib"] is not None:
            st["card_total_mib"] = rec["card_total_mib"]
        for key in ("peak_reserved_mib", "peak_allocated_mib"):
            v = rec[key]
            if v is not None and (st[key] is None or v > st[key]["mib"]):
                st[key] = {"mib": v, "boot": boot, "t": rec["ts"], "line": lineno,
                           "phase": rec["phase"]}
        cf = rec["card_free_mib"]
        if cf is not None and (st["min_card_free_mib"] is None or cf < st["min_card_free_mib"]):
            st["min_card_free_mib"] = cf
    return per


def build_record(sources: Iterable[Tuple[str, Iterable[str]]], profile: str = "qwen27b",
                 phases: Sequence[str] = DEFAULT_PHASES,
                 basis: str = DEFAULT_BASIS) -> Dict[str, Any]:
    """The record from ``(boot name, lines)`` pairs: per card the MAXIMUM of
    ``basis`` over every boot and line, with where it was seen."""
    if basis not in BASES:
        raise ValueError(f"basis {basis!r} not in {BASES}")
    boots: List[str] = []
    cards: Dict[int, Dict[str, Any]] = {}
    for boot, lines in sources:
        boots.append(boot)
        for rank, st in scan_lines(boot, lines, phases).items():
            c = cards.setdefault(rank, {
                "n": 0, "n_skipped": 0, "card_total_mib": None, "peak": None,
                "peak_allocated": None, "min_card_free_mib": None, "per_boot": {},
            })
            c["n"] += st["n_lines"]
            c["n_skipped"] += st["n_skipped"]
            if st["n_lines"] == 0:
                continue
            if st["card_total_mib"] is not None:
                c["card_total_mib"] = st["card_total_mib"]
            for dst, key in (("peak", basis), ("peak_allocated", "peak_allocated_mib")):
                v = st[key]
                if v is not None and (c[dst] is None or v["mib"] > c[dst]["mib"]):
                    c[dst] = v
            if st["min_card_free_mib"] is not None and (
                    c["min_card_free_mib"] is None or st["min_card_free_mib"] < c["min_card_free_mib"]):
                c["min_card_free_mib"] = st["min_card_free_mib"]
            c["per_boot"][boot] = {"peak_mib": st[basis]["mib"], "n": st["n_lines"]}
    out_cards: Dict[str, Any] = {}
    for rank in sorted(cards):
        c = cards[rank]
        if c["peak"] is None:
            continue  # no measured window on this card: no entry, never a guess
        p = c["peak"]
        out_cards[str(rank)] = {
            "peak_mib": int(p["mib"]),
            "peak_allocated_mib": int(c["peak_allocated"]["mib"]) if c["peak_allocated"] else None,
            "card_total_mib": c["card_total_mib"],
            "min_card_free_mib": c["min_card_free_mib"],
            "n": c["n"],
            "n_skipped": c["n_skipped"],
            "source": {"boot": p["boot"], "t": p["t"], "line": p["line"], "phase": p["phase"]},
            "per_boot": dict(sorted(c["per_boot"].items())),
        }
    return {
        "record": RECORD_NAME, "schema": SCHEMA, "profile": profile, "basis": basis,
        "aggregate": "max", "phases": list(phases), "boots": sorted(boots),
        "cards": out_cards,
    }


def record_path(profile: str, env: Optional[Mapping[str, str]] = None) -> str:
    """The record file: ``SGLANG_WEG2_L15_POOL_PEAK_RECORD_FILE`` or
    ``profile_records_data/l15_pool_peak_<profile>.json``."""
    e = os.environ if env is None else env
    override = str(e.get(FILE_ENV, "") or "").strip()
    return override or os.path.join(RECORDS_DIR, f"l15_pool_peak_{profile}.json")


def load_record(path: str) -> Optional[Dict[str, Any]]:
    """The record at ``path``; None when absent, unreadable or not a record of
    this schema (never raises: no record means no pool share)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            rec = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(rec, dict) or rec.get("record") != RECORD_NAME or rec.get("schema") != SCHEMA:
        return None
    return rec


def peak_line(card: int, peak_mib: Optional[int], source: str, n: int) -> str:
    """The boot line, one per card."""
    pm = "none" if peak_mib is None else str(int(peak_mib))
    return f"{LINE_MARKER} card={card} peak_mib={pm} source={source} n={int(n)}"


def resolve_peaks(record: Optional[Mapping[str, Any]], profile: str,
                  card_totals_mib: Sequence[Sequence[int]]
                  ) -> Tuple[List[Optional[int]], List[str]]:
    """Per launch card (budget ordinal, in order) the peak and its boot line.

    ``card_totals_mib[i]`` are the totals card ``i`` may legitimately show
    (NVML total, and total minus the driver carve: the torch-visible total the
    window lines print).  A card is priced only when the record has an entry for
    its ordinal AND that entry was measured on a card of the same total
    (+-:data:`TOTAL_TOLERANCE_MIB`); otherwise ``None`` (= no pool share) with
    the reason as ``source``.
    """
    peaks: List[Optional[int]] = []
    lines: List[str] = []
    for i, totals in enumerate(card_totals_mib):
        if record is None:
            src, peak, n = "NO-RECORD", None, 0
        elif record.get("profile") != profile:
            src, peak, n = f"PROFILE-MISMATCH({record.get('profile')})", None, 0
        else:
            ent = (record.get("cards") or {}).get(str(i))
            if not ent or ent.get("peak_mib") is None:
                src, peak, n = "NO-RECORD", None, 0
            else:
                n = int(ent.get("n") or 0)
                rec_total = ent.get("card_total_mib")
                if rec_total is not None and not any(
                        abs(int(rec_total) - int(t)) <= TOTAL_TOLERANCE_MIB for t in totals):
                    src, peak = f"CARD-MISMATCH(record_total={rec_total})", None
                else:
                    s = ent.get("source") or {}
                    peak = int(ent["peak_mib"])
                    src = (f"RECORD({record.get('basis', DEFAULT_BASIS)} max over "
                           f"{len(record.get('boots') or [])} boots; {s.get('boot', '?')}@{s.get('t', '?')}"
                           f":line{s.get('line', '?')})").replace(" ", "_")
        peaks.append(peak)
        lines.append(peak_line(i, peak, src, n))
    return peaks, lines


def planner_peaks(profile: str, cards: Sequence[Any], legacy_peaks: Sequence[Optional[int]],
                  env: Mapping[str, str],
                  load: Callable[[str], Optional[Dict[str, Any]]] = load_record
                  ) -> Tuple[List[Optional[int]], List[str]]:
    """The planner input: ``(peaks, boot lines)`` for ``l15_plan.resolve_posts``.

    Switch off: ``(list(legacy_peaks), [])`` -- nothing read, nothing printed.
    Switch on: the pool record is the only source (``legacy_peaks`` is not
    consulted); a card without a usable record gets ``None`` (UNMEASURED 0, no
    pool share).  ``cards`` carry ``total_mib`` and ``reserved_mib``.
    """
    if not switch_on(env):
        return list(legacy_peaks), []
    totals = [(int(c.total_mib), int(c.total_mib) - int(getattr(c, "reserved_mib", 0) or 0))
              for c in cards]
    return resolve_peaks(load(record_path(profile, env)), profile, totals)


def format_record(rec: Mapping[str, Any]) -> str:
    """Human-readable table of a record (CLI output)."""
    out = [f"{RECORD_NAME} profile={rec.get('profile')} basis={rec.get('basis')} "
           f"aggregate={rec.get('aggregate')} phases={','.join(rec.get('phases') or [])} "
           f"boots={len(rec.get('boots') or [])}"]
    for k in sorted((rec.get("cards") or {}), key=int):
        c = rec["cards"][k]
        s = c["source"]
        out.append(
            f"  card={k} peak_mib={c['peak_mib']} peak_allocated_mib={c['peak_allocated_mib']} "
            f"card_total_mib={c['card_total_mib']} min_card_free_mib={c['min_card_free_mib']} "
            f"n={c['n']} (skipped {c['n_skipped']}) source={s['boot']}@{s['t']} line {s['line']} "
            f"phase={s['phase']}")
        for b, v in c["per_boot"].items():
            out.append(f"      boot {b}: max {v['peak_mib']} MiB over n={v['n']}")
    return "\n".join(out)
