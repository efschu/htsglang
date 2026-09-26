"""MEASURED RECORDS of the model-profile registry (UNIFY_PLAN Schritt 9).

The registry row (:mod:`sglang.srt.weg2.form`, ``ModelProfile``) says what a
model IS; this module holds what was MEASURED on it: one JSON file per profile
under ``profile_records_data/`` (``<profile>.json``), each record carrying its
value AND the conditions it holds under:

* ``measured_on`` -- the profile whose checkpoint the value was measured on.
  A row that carries another model's measurement names it (``borrow`` in the
  profile's file): BORROWED, printed by the launcher, never mixed in silently.
* ``fmt`` -- the weight format it was measured with ("" = format-neutral).
* ``power_limit_w`` -- the board power limit per card class it was measured
  under (release table row 27, weg2/power_limit.py); ``null`` = the boot did
  not record it. A record holds only for its power limit
  (:func:`power_verdict`): a timing record under a FOREIGN limit (> 5 %) is
  not a measurement of this rig state.
* ``pp_layer_ratio`` -- the P cut a per-stage table was measured on (Agent PG,
  26.09.: a table holds only for its cut); ``[]`` = cut-independent.
* ``boots`` / ``n`` / ``confidence`` -- where it came from, how many samples.

What is NOT a measurement (a HOCHRECHNUNG, a desk choice) stays a marked
constant in the code with its provenance; a borrowed row stays until a record
of this model replaces it. A new model needs a registry row plus its records
file (calibration boots), no code.

PURE: stdlib only (the launcher, the ranks and the desk import it).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

RECORDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile_records_data")

#: Relative deviation above which a record counts as measured under a FOREIGN
#: power limit -- the same 5 % as weg2/power_limit.TOLERANCE (row 27).
POWER_TOLERANCE = 0.05

KINDS = ("memory", "timing", "census", "geometry")

POWER_MATCH = "match"
POWER_MISMATCH = "mismatch"
POWER_UNKNOWN = "unknown"

CUT_MATCH = "match"
CUT_MISMATCH = "mismatch"
CUT_UNKNOWN = "unknown"
CUT_NEUTRAL = "neutral"


class RecordError(ValueError):
    """A malformed records file or record."""


@dataclass(frozen=True)
class Record:
    profile: str
    name: str
    value: object
    provenance: str
    measured_on: str
    kind: str = "memory"
    fmt: str = ""
    #: card class -> W (e.g. {"RTX5090": 400.0, "RTX3080": 230.0}); None = not recorded
    power_limit_w: Optional[Tuple[Tuple[str, float], ...]] = None
    power_limit_source: str = ""
    pp_layer_ratio: Tuple[int, ...] = ()
    boots: Tuple[str, ...] = ()
    n: int = 0
    confidence: str = ""
    #: parts of the value taken from another record (e.g. "e_gbs <- int8"),
    #: printed with every line that uses it
    borrowed_parts: Tuple[str, ...] = ()

    @property
    def borrowed(self) -> bool:
        return self.measured_on != self.profile

    def power_limits(self) -> Optional[Dict[str, float]]:
        return None if self.power_limit_w is None else dict(self.power_limit_w)

    def describe(self) -> str:
        pl = ("power limit not recorded" if self.power_limit_w is None else
              "power " + "/".join(f"{c} {w:g} W" for c, w in self.power_limit_w))
        cut = f", cut {','.join(map(str, self.pp_layer_ratio))}" if self.pp_layer_ratio else ""
        fmt = f" fmt={self.fmt}" if self.fmt else ""
        bor = f" BORROWED from {self.measured_on}" if self.borrowed else ""
        return f"{self.name}{fmt} ({self.provenance}; {pl}{cut}){bor}"


def _to_value(v):
    """JSON lists become tuples (the registry's constants are immutable)."""
    if isinstance(v, list):
        return tuple(_to_value(x) for x in v)
    if isinstance(v, dict):
        return {k: _to_value(x) for k, x in v.items()}
    return v


def _power_field(raw, where: str) -> Optional[Tuple[Tuple[str, float], ...]]:
    if raw is None:
        return None
    if not isinstance(raw, dict) or not raw:
        raise RecordError(f"{where}: power_limit_w must be a card-class -> W object or null")
    out = []
    for k, v in sorted(raw.items()):
        try:
            w = float(v)
        except (TypeError, ValueError):
            raise RecordError(f"{where}: power_limit_w[{k!r}] = {v!r} is not a number")
        if w <= 0:
            raise RecordError(f"{where}: power_limit_w[{k!r}] must be > 0 W")
        out.append((str(k), w))
    return tuple(out)


def _cut_field(raw, where: str) -> Tuple[int, ...]:
    if raw in (None, "", []):
        return ()
    try:
        vals = tuple(int(x) for x in (raw.split(",") if isinstance(raw, str) else raw))
    except (TypeError, ValueError):
        raise RecordError(f"{where}: pp_layer_ratio {raw!r} is not a list of layer counts")
    if any(v <= 0 for v in vals):
        raise RecordError(f"{where}: pp_layer_ratio {raw!r}: every stage needs >= 1 layer")
    return vals


def parse_record(profile: str, d: Mapping, measured_on: Optional[str] = None) -> Record:
    where = f"records[{profile}] {d.get('name', '?')!r}"
    for req in ("name", "value", "provenance"):
        if req not in d:
            raise RecordError(f"{where}: field {req!r} missing")
    kind = str(d.get("kind", "memory"))
    if kind not in KINDS:
        raise RecordError(f"{where}: kind {kind!r} not one of {KINDS}")
    return Record(
        profile=profile,
        name=str(d["name"]),
        value=_to_value(d["value"]),
        provenance=str(d["provenance"]),
        measured_on=str(measured_on or d.get("measured_on") or profile),
        kind=kind,
        fmt=str(d.get("fmt", "") or ""),
        power_limit_w=_power_field(d.get("power_limit_w"), where),
        power_limit_source=str(d.get("power_limit_source", "") or ""),
        pp_layer_ratio=_cut_field(d.get("pp_layer_ratio"), where),
        boots=tuple(str(b) for b in (d.get("boots") or ())),
        n=int(d.get("n", 0) or 0),
        confidence=str(d.get("confidence", "") or ""),
        borrowed_parts=tuple(str(b) for b in (d.get("borrowed_parts") or ())),
    )


@lru_cache(maxsize=None)
def _file(profile: str, records_dir: str) -> Tuple[Tuple[Record, ...], Tuple[str, Tuple[str, ...], str]]:
    path = os.path.join(records_dir, f"{profile}.json")
    try:
        with open(path) as fh:
            data = json.load(fh)
    except OSError:
        return (), ("", (), "")
    except ValueError as exc:
        raise RecordError(f"{path}: not JSON ({exc})")
    if not isinstance(data, dict) or data.get("profile") != profile:
        raise RecordError(f"{path}: an object with \"profile\": {profile!r} expected")
    recs = tuple(parse_record(profile, d) for d in (data.get("records") or ()))
    seen = set()
    for r in recs:
        key = (r.name, r.fmt, r.power_limit_w, r.pp_layer_ratio)
        if key in seen:
            raise RecordError(f"{path}: record {r.name!r} fmt={r.fmt!r} twice for the same "
                              f"power limit and cut (which one holds?)")
        seen.add(key)
    b = data.get("borrow") or {}
    borrow = (str(b.get("from", "") or ""), tuple(str(n) for n in (b.get("names") or ())),
              str(b.get("why", "") or ""))
    if borrow[1] and not borrow[0]:
        raise RecordError(f"{path}: borrow.names without borrow.from")
    return recs, borrow


def own_records(profile: str, records_dir: str = RECORDS_DIR) -> Tuple[Record, ...]:
    """The records measured on ``profile``'s checkpoint (its own file)."""
    return _file(str(profile), records_dir)[0]


def borrow_of(profile: str, records_dir: str = RECORDS_DIR) -> Tuple[str, Tuple[str, ...], str]:
    """``(from profile, names, why)`` of the rows ``profile`` borrows."""
    return _file(str(profile), records_dir)[1]


def records(profile: str, records_dir: str = RECORDS_DIR) -> Tuple[Record, ...]:
    """Own records plus the BORROWED rows (the other profile's records under
    their names, ``measured_on`` = the profile that measured them). A name the
    profile measured itself is never borrowed."""
    own = own_records(profile, records_dir)
    src, names, _why = borrow_of(profile, records_dir)
    if not names:
        return own
    have = {r.name for r in own}
    out = list(own)
    for n in names:
        if n in have:
            raise RecordError(f"records[{profile}]: {n!r} is measured AND borrowed (drop the borrow)")
        got = [r for r in own_records(src, records_dir) if r.name == n]
        if not got:
            raise RecordError(f"records[{profile}]: borrows {n!r} from {src!r}, which has no such record")
        out.extend(Record(**{**r.__dict__, "profile": profile}) for r in got)
    return tuple(out)


def constants_of(profile: str, records_dir: str = RECORDS_DIR) -> Dict[str, Record]:
    """The format-neutral scalar rows (``fmt`` == "", one record per name) --
    what :attr:`form.ModelProfile.constants` is built from."""
    out: Dict[str, Record] = {}
    for r in records(profile, records_dir):
        if r.fmt:
            continue
        if r.name in out:
            raise RecordError(f"records[{profile}]: constant {r.name!r} has more than one "
                              f"format-neutral record; use select() with a power limit/cut")
        out[r.name] = r
    return out


# ---------------------------------------------------------------------------
# the filters: power limit (row 27) and P cut (Agent PG)


def power_verdict(record_limits: Optional[Mapping[str, float]],
                  current: Optional[Mapping[str, object]]) -> str:
    """:data:`POWER_MATCH` | :data:`POWER_MISMATCH` | :data:`POWER_UNKNOWN`.

    ``current``: card class -> running W (or a collection of W when the rig
    carries several cards of one class). A class known on both sides counts;
    one class more than :data:`POWER_TOLERANCE` apart = mismatch; no class
    comparable (record without limit, NVML without answer) = unknown."""
    if not record_limits or not current:
        return POWER_UNKNOWN
    compared = False
    for cls, cal in record_limits.items():
        cur = current.get(cls)
        if cur is None:
            continue
        vals = [cur] if isinstance(cur, (int, float)) else [v for v in cur if v is not None]
        for v in vals:
            if not cal or not v:
                continue
            compared = True
            if abs(float(v) - float(cal)) / float(cal) > POWER_TOLERANCE:
                return POWER_MISMATCH
    return POWER_MATCH if compared else POWER_UNKNOWN


def cut_verdict(record_cut: Sequence[int], boot_cut: Optional[Sequence[int]]) -> str:
    """The record's P cut against the boot's (Agent PG: a per-stage table holds
    only for its cut): neutral (cut-independent record), unknown (the boot's
    cut is not known yet), match, mismatch."""
    if not record_cut:
        return CUT_NEUTRAL
    if not boot_cut:
        return CUT_UNKNOWN
    return CUT_MATCH if tuple(int(x) for x in boot_cut) == tuple(record_cut) else CUT_MISMATCH


@dataclass(frozen=True)
class Selected:
    record: Record
    power: str
    cut: str
    #: the other candidates and why they lost (printed by callers that care)
    rejected: Tuple[str, ...] = field(default_factory=tuple)

    def tag(self) -> str:
        """A short suffix for a log line: power/cut verdicts and borrowing."""
        parts = [f"record={self.record.name}"]
        if self.record.fmt:
            parts.append(f"fmt={self.record.fmt}")
        parts.append(f"power={self.power}")
        if self.cut != CUT_NEUTRAL:
            parts.append(f"cut={self.cut}")
        if self.record.borrowed:
            parts.append(f"BORROWED({self.record.measured_on})")
        if self.record.borrowed_parts:
            parts.append("BORROWED(" + "; ".join(self.record.borrowed_parts) + ")")
        return " ".join(parts)


def select(profile: str, name: str, *, fmt: str = "",
           current_power: Optional[Mapping[str, object]] = None,
           cut: Optional[Sequence[int]] = None, strict_power: bool = False,
           records_dir: str = RECORDS_DIR) -> Optional[Selected]:
    """The record ``name`` (format ``fmt``) of ``profile`` that holds for the
    running power limit and P cut.

    Order: a power MATCH before an UNKNOWN limit; a cut mismatch never holds
    (a table holds only for its cut). A record under a FOREIGN limit is
    returned only when nothing else is left and ``strict_power`` is off --
    its ``power`` verdict says so and the caller prints it (the D speed lines
    NAME their limit, row 27); with ``strict_power`` it does not count."""
    cands = [r for r in records(profile, records_dir) if r.name == name and r.fmt == str(fmt or "")]
    if not cands:
        return None
    rejected: List[str] = []
    ranked: List[Tuple[int, Record, str, str]] = []
    for i, r in enumerate(cands):
        cv = cut_verdict(r.pp_layer_ratio, cut)
        pv = power_verdict(r.power_limits(), current_power)
        if cv == CUT_MISMATCH:
            rejected.append(f"{r.describe()}: cut mismatch against "
                            f"{','.join(map(str, cut or ()))}")
            continue
        if pv == POWER_MISMATCH and strict_power:
            rejected.append(f"{r.describe()}: FOREIGN power limit")
            continue
        rank = {POWER_MATCH: 0, POWER_UNKNOWN: 1, POWER_MISMATCH: 2}[pv]
        ranked.append((rank * 1000 + i, r, pv, cv))
    if not ranked:
        return None
    ranked.sort(key=lambda t: t[0])
    _, r, pv, cv = ranked[0]
    rejected.extend(f"{x[1].describe()}: lost to the preferred record" for x in ranked[1:])
    return Selected(r, pv, cv, tuple(rejected))


def current_power_by_class(reader=None) -> Optional[Dict[str, Tuple[float, ...]]]:
    """``{card class: (running W, ...)}`` from NVML (weg2/power_limit
    .read_current); None when NVML cannot answer. ``reader`` for tests."""
    try:
        if reader is None:
            from sglang.srt.weg2 import power_limit as _pl

            reader = _pl.read_current
        got = reader()
    except Exception:  # noqa: BLE001 - an unanswered NVML makes the filter 'unknown', never a refusal
        return None
    out: Dict[str, List[float]] = {}
    for w, cls in got.values():
        if w is not None:
            out.setdefault(str(cls), []).append(float(w))
    return {k: tuple(v) for k, v in out.items()} or None


def log_power_by_class(lines) -> Dict[str, float]:
    """``{card class: W}`` from a boot log's ``POWER-LIMIT rank`` lines (rank
    lines ``card=RTX5090 power_limit_w=400`` and the launcher's per-card
    lines alike). A line with ``power_limit_w=NA`` says nothing."""
    import re

    rx_card = re.compile(r"\bcard=(\S+)")
    rx_w = re.compile(r"\bpower_limit_w=([0-9.]+)")
    out: Dict[str, float] = {}
    for ln in lines:
        if "POWER-LIMIT rank " not in ln:
            continue
        c, w = rx_card.search(ln), rx_w.search(ln)
        if c and w:
            out[c.group(1)] = float(w.group(1))
    return out


__all__ = [
    "RECORDS_DIR", "POWER_TOLERANCE", "KINDS", "POWER_MATCH", "POWER_MISMATCH", "POWER_UNKNOWN",
    "CUT_MATCH", "CUT_MISMATCH", "CUT_UNKNOWN", "CUT_NEUTRAL", "RecordError", "Record", "Selected",
    "parse_record", "own_records", "borrow_of", "records", "constants_of", "power_verdict",
    "cut_verdict", "select", "current_power_by_class", "log_power_by_class",
]
