"""FORM MEASURES v3 -- the decode-round matrix of a rig + model, one cell per
(form, ..., bs, depth, text, temp), read by the form choice and the planner.

Feature 27B-FORM-MESSMATRIX (modell=beide), user orders 29.09.:

* "nach jedem flip von P->D man 'weiss' ja vorher schon wie viel bs kommen
  wird bei welche kontexttiefe" -- the form is chosen per (bs, depth), so the
  time table needs those axes, not just bs (``weg2.form_measures/2`` had
  ``rounds[form][cards][bs]`` only).
* "Eine Messmatrix ... die muss aus dem hardware-/modell profil kommen und auf
  wunsch komplett oder teile davon 'durchgemessen'" -- WHICH cells exist comes
  from the model profile (record ``FORM_MATRIX_AXES`` in
  ``profile_records_data/<profile>.json``); WHAT a cell measured comes from a
  calibration run (:func:`import_calib_run`), and :func:`select` names the
  cells a run should (re)measure: ``missing`` | ``stale[:Nd|:rev]`` | ``all`` |
  an explicit ``form=A,B77;bs=1..6;depth=2k,32k;text=code,prose``.

NO NEW STORE (agreed with NF 29.09.): the same file that carried v2 carries
v3; :func:`load` migrates a v2 document on read (its rounds keep their form
and bs, every other axis ``None`` -- a migrated cell answers only a question
that does not ask for depth/text, it is never stretched to one that does).

THE RULES a reader relies on:

* A cell holds only for its IDENTITY (:class:`Identity`: model, per-rank
  precision, card UUIDs in rank order, links, image rev, power limit). 27B and
  Next Flash never share a record; a lookup under another identity is a miss.
* A missing cell is MISSING (``None``), never interpolated from a neighbour.
  Depth is bucketed CONSERVATIVELY: a request of d tokens reads the smallest
  measured depth point >= d (a deeper cell is the slower one), and if that
  point has no cell the answer is ``None`` -- not the next point up.
* "unmoeglich" is a state, not a missing value: a form that cannot start on
  this image (``nicht-startbar``) or a (bs, depth) the form's KV cannot hold
  (``kapazitaet``) is recorded so, with its reason, and a planner reads it as
  "never pick", not as "unknown".

PURE: stdlib only (launcher, desk tools and tests import it).
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCHEMA_V2 = "weg2.form_measures/2"
SCHEMA_V3 = "weg2.form_measures/3"

#: Every axis a cell may carry. ``None`` = the axis does not apply / was not
#: set for this cell (e.g. ``moe_ratio`` on the dense 27B). The key of a cell
#: is the full tuple, so two cells differing only in an axis never collapse.
AXES: Tuple[str, ...] = (
    "form", "roles", "weights", "tokens", "moe_ratio", "owned_cut", "fr_d",
    "scratch_rows", "kv_stage", "precision", "spec", "transport",
    "bs", "depth", "text", "temp",
)

STATE_MEASURED = "gemessen"
STATE_UNMEASURED = "ungemessen"
STATE_IMPOSSIBLE_CAPACITY = "unmoeglich:kapazitaet"
STATE_IMPOSSIBLE_START = "unmoeglich:nicht-startbar"
STATES = (STATE_MEASURED, STATE_UNMEASURED, STATE_IMPOSSIBLE_CAPACITY, STATE_IMPOSSIBLE_START)

#: The depth points a cell is measured at (label -> tokens). A request is
#: bucketed UP to the next point (conservative, see module doc).
DEPTH_POINTS: Tuple[Tuple[str, int], ...] = (
    ("2k", 2048), ("10k", 10240), ("32k", 32768), ("97k", 99328),
    ("240k", 245760), ("257k", 262144),
)
TEXTS = ("code", "prose", "thinking")

#: Record name in profile_records_data/<profile>.json that says WHICH cells
#: the matrix of this model has on this rig (forms, bs, depths, texts,
#: capacity, startability). Kind "geometry".
MATRIX_RECORD = "FORM_MATRIX_AXES"


class FormMeasuresError(ValueError):
    """A malformed v2/v3 document, identity or selection."""


# ---------------------------------------------------------------------------
# identity, axes, cells


@dataclass(frozen=True)
class Identity:
    """What a cell holds for. Two identities are the same only if every field
    is equal; ``None`` in a field is a value too (a v2 migration without a
    model is its own identity and matches nothing a real boot asks for)."""

    model: Optional[str]
    precision: Tuple[str, ...] = ()
    cards: Tuple[str, ...] = ()
    links: Tuple[str, ...] = ()
    image_rev: Optional[str] = None
    power_limit_w: Tuple[Tuple[str, float], ...] = ()

    def key(self) -> str:
        return json.dumps(self.to_json(), sort_keys=True)

    def to_json(self) -> Dict[str, object]:
        return {"model": self.model, "precision": list(self.precision), "cards": list(self.cards),
                "links": list(self.links), "image_rev": self.image_rev,
                "power_limit_w": {k: v for k, v in self.power_limit_w}}

    @classmethod
    def from_json(cls, d: Mapping) -> "Identity":
        if not isinstance(d, Mapping):
            raise FormMeasuresError(f"identity must be an object, got {d!r}")
        pl = d.get("power_limit_w") or {}
        return cls(model=d.get("model"), precision=tuple(d.get("precision") or ()),
                   cards=tuple(d.get("cards") or ()), links=tuple(d.get("links") or ()),
                   image_rev=d.get("image_rev"),
                   power_limit_w=tuple(sorted((str(k), float(v)) for k, v in pl.items())))


def _norm_axis(name: str, v):
    if v is None:
        return None
    if name == "bs":
        return int(v)
    if name == "depth":
        return depth_label(v)
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    return str(v)


def axes_key(axes: Mapping[str, object]) -> Tuple[object, ...]:
    unknown = set(axes) - set(AXES)
    if unknown:
        raise FormMeasuresError(f"unknown axes {sorted(unknown)}; known: {AXES}")
    return tuple(_norm_axis(a, axes.get(a)) for a in AXES)


def axes_of_key(key: Sequence[object]) -> Dict[str, object]:
    return {a: v for a, v in zip(AXES, key) if v is not None}


def depth_label(v) -> str:
    """A depth point label for a label or a token count; a token count is
    bucketed UP to the next point. Deeper than every point -> error."""
    if isinstance(v, str):
        for lab, _ in DEPTH_POINTS:
            if v == lab:
                return lab
        m = re.fullmatch(r"(\d+)k", v)
        if not m:
            raise FormMeasuresError(f"depth {v!r}: not a label of {[p for p, _ in DEPTH_POINTS]} "
                                    f"nor '<n>k'")
        v = int(m.group(1)) * 1024
    tokens = int(v)
    for lab, t in DEPTH_POINTS:
        if tokens <= t:
            return lab
    raise FormMeasuresError(f"depth {tokens} tokens is deeper than every point {DEPTH_POINTS[-1]}")


@dataclass
class Cell:
    axes: Dict[str, object]
    state: str = STATE_UNMEASURED
    round_ms_median: Optional[float] = None
    n: int = 0
    p10: Optional[float] = None
    p90: Optional[float] = None
    #: per rank (index = rank), None where the split instrument was off
    compute_ms: Optional[List[float]] = None
    wait_ms: Optional[List[float]] = None
    accept_len: Optional[float] = None
    #: Next Flash: expert misses per seat per rank, measured miss cost per card
    misses_per_seat: Optional[List[float]] = None
    miss_cost_ms: Optional[Dict[str, float]] = None
    boot: Optional[str] = None
    at: Optional[str] = None
    image_rev: Optional[str] = None
    reason: str = ""
    source: str = ""

    def __post_init__(self):
        if self.state not in STATES:
            raise FormMeasuresError(f"cell state {self.state!r} not one of {STATES}")
        if self.state == STATE_MEASURED and self.round_ms_median is None:
            raise FormMeasuresError("a measured cell needs round_ms_median")
        if self.state.startswith("unmoeglich") and not self.reason:
            raise FormMeasuresError(f"state {self.state!r} needs a reason")
        self.axes = axes_of_key(axes_key(self.axes))

    def to_json(self) -> Dict[str, object]:
        return {k: v for k, v in asdict(self).items() if v not in (None, "", [])}

    @classmethod
    def from_json(cls, d: Mapping) -> "Cell":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# the store


@dataclass
class FormMeasuresV3:
    #: identity key -> (Identity, axes key -> Cell)
    records: Dict[str, Tuple[Identity, Dict[Tuple[object, ...], Cell]]] = field(default_factory=dict)
    #: v2 collective tables, carried through unchanged (graph prices per transport)
    allreduce: Dict[str, object] = field(default_factory=dict)
    dcp_exchange: Dict[str, object] = field(default_factory=dict)
    source: str = ""
    migrated_from_v2: bool = False

    # -- write ----------------------------------------------------------------
    def upsert(self, identity: Identity, cell: Cell) -> None:
        """Insert or replace the cell with the same axes under ``identity``.
        A measured cell is replaced only by a newer one (``at``); an
        "unmoeglich" cell is replaced by a measurement (the form started)."""
        _, cells = self.records.setdefault(identity.key(), (identity, {}))
        k = axes_key(cell.axes)
        old = cells.get(k)
        if old is not None and old.state == STATE_MEASURED and cell.state == STATE_MEASURED:
            if (old.at or "") > (cell.at or ""):
                return
        if old is not None and old.state != STATE_UNMEASURED and cell.state == STATE_UNMEASURED:
            # an empty run never erases what is known: a measurement, or the
            # structural "unmoeglich" (run_matrix bands are global across arms,
            # so an arm may try a bs its KV cannot hold and record 0 rounds)
            return
        cells[k] = cell

    # -- read -----------------------------------------------------------------
    def cells(self, identity: Identity) -> Dict[Tuple[object, ...], Cell]:
        rec = self.records.get(identity.key())
        return dict(rec[1]) if rec else {}

    def lookup(self, identity: Identity, **axes) -> Optional[Cell]:
        """The cell with EXACTLY these axes (depth bucketed up), or None.
        Never a neighbour, never another identity."""
        return self.cells(identity).get(axes_key(axes))

    def round_ms(self, identity: Identity, **axes) -> Optional[float]:
        c = self.lookup(identity, **axes)
        return c.round_ms_median if c is not None and c.state == STATE_MEASURED else None

    def rounds_ms_v2_view(self, identity: Identity, *, transport: str,
                          depth=None, text: Optional[str] = None,
                          temp: Optional[str] = None) -> Dict[str, Dict[str, Dict[int, float]]]:
        """The ``rounds_ms[form][cards][bs]`` shape the v2 reader
        (``rank_form.FormMeasures.rounds_ms`` / ``choose_form``) consumes:
        only MEASURED cells of this identity whose transport/depth/text/temp
        equal the arguments (``None`` = the cell must not carry that axis)."""
        want = {"transport": transport, "depth": depth, "text": text, "temp": temp}
        want = {k: _norm_axis(k, v) for k, v in want.items()}
        cards = "+".join(identity.cards)
        out: Dict[str, Dict[str, Dict[int, float]]] = {}
        for k, c in self.cells(identity).items():
            ax = dict(zip(AXES, k))
            if c.state != STATE_MEASURED or ax.get("bs") is None or ax.get("form") is None:
                continue
            if any(ax.get(a) != v for a, v in want.items()):
                continue
            out.setdefault(str(ax["form"]), {}).setdefault(cards, {})[int(ax["bs"])] = float(c.round_ms_median)
        return out

    # -- (de)serialise ----------------------------------------------------------
    def to_json(self) -> Dict[str, object]:
        recs = []
        for _, (ident, cells) in sorted(self.records.items()):
            recs.append({"identity": ident.to_json(),
                         "cells": [c.to_json() for _, c in sorted(cells.items(), key=lambda kv: repr(kv[0]))]})
        doc: Dict[str, object] = {"schema": SCHEMA_V3, "records": recs}
        if self.allreduce:
            doc["allreduce"] = self.allreduce
        if self.dcp_exchange:
            doc["dcp_exchange"] = self.dcp_exchange
        if self.migrated_from_v2:
            doc["migrated_from"] = SCHEMA_V2
        return doc

    def save(self, path: str) -> None:
        """Atomic: tmp in the same directory + fsync + os.replace."""
        d = os.path.dirname(os.path.abspath(path)) or "."
        fd, tmp = tempfile.mkstemp(prefix=".form_measures.", dir=d)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.to_json(), f, indent=1, ensure_ascii=False, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def from_doc(doc: Mapping, *, source: str = "", v2_identity: Optional[Identity] = None) -> FormMeasuresV3:
    schema = doc.get("schema")
    if schema == SCHEMA_V3:
        fm = FormMeasuresV3(source=source, allreduce=dict(doc.get("allreduce") or {}),
                            dcp_exchange=dict(doc.get("dcp_exchange") or {}),
                            migrated_from_v2=doc.get("migrated_from") == SCHEMA_V2)
        for rec in doc.get("records") or ():
            ident = Identity.from_json(rec.get("identity") or {})
            for c in rec.get("cells") or ():
                fm.upsert(ident, Cell.from_json(c))
        return fm
    if schema == SCHEMA_V2:
        return migrate_v2(doc, source=source, identity=v2_identity)
    raise FormMeasuresError(f"{source or 'document'}: schema {schema!r}, expected {SCHEMA_V3!r} or {SCHEMA_V2!r}")


def migrate_v2(doc: Mapping, *, source: str = "", identity: Optional[Identity] = None) -> FormMeasuresV3:
    """v2 ``rounds[form][cards][bs] = {median_ms, transport, n, source}`` ->
    v3 cells (form, transport, bs; every other axis None). The v2 document
    carries no model: without ``identity`` the cells land under
    ``Identity(model=None, cards=<cards>)`` and match no real boot's lookup
    -- a migration never invents a model."""
    fm = FormMeasuresV3(source=source, allreduce=dict(doc.get("allreduce") or {}),
                        dcp_exchange=dict(doc.get("dcp_exchange") or {}), migrated_from_v2=True)
    for form, per_cards in (doc.get("rounds") or {}).items():
        for cards, per_bs in (per_cards or {}).items():
            ident = identity or Identity(model=None, cards=tuple(str(cards).split("+")))
            for bs, v in (per_bs or {}).items():
                ms = (v or {}).get("median_ms")
                cell = Cell(axes={"form": form, "bs": int(bs), "transport": (v or {}).get("transport")},
                            state=STATE_MEASURED if ms is not None else STATE_UNMEASURED,
                            round_ms_median=float(ms) if ms is not None else None,
                            n=int((v or {}).get("n") or 0), source=str((v or {}).get("source") or source),
                            reason="" if ms is not None else "v2 cell without median_ms")
                fm.upsert(ident, cell)
    return fm


def load(path: Optional[str], *, v2_identity: Optional[Identity] = None) -> FormMeasuresV3:
    """Missing file = empty store (every lookup None, i.e. UNMEASURED)."""
    if not path or not os.path.exists(path):
        return FormMeasuresV3(source=path or "")
    with open(path) as f:
        try:
            doc = json.load(f)
        except json.JSONDecodeError as exc:
            raise FormMeasuresError(f"{path}: not JSON ({exc})") from None
    return from_doc(doc, source=path, v2_identity=v2_identity)


# ---------------------------------------------------------------------------
# the matrix spec from the model profile, and the selection a calibration runs


@dataclass(frozen=True)
class FormSpec:
    name: str
    arm: Optional[str]
    axes: Mapping[str, object]
    startable: bool = True
    why_not: str = ""


@dataclass(frozen=True)
class MatrixSpec:
    forms: Tuple[FormSpec, ...]
    bs: Tuple[int, ...]
    depths: Tuple[str, ...]
    texts: Tuple[str, ...]
    temps: Tuple[str, ...]
    #: form name -> depth label -> max bs the form's KV holds at that depth
    capacity: Mapping[str, Mapping[str, int]]
    provenance: str = ""

    def all_cells(self) -> List[Tuple[FormSpec, Dict[str, object]]]:
        out = []
        for fs in self.forms:
            for depth in self.depths:
                for text in self.texts:
                    for temp in self.temps:
                        for bs in self.bs:
                            ax = dict(fs.axes)
                            ax.update({"form": fs.name, "bs": bs, "depth": depth, "text": text, "temp": temp})
                            out.append((fs, ax))
        return out


def matrix_spec_from_value(value: Mapping, provenance: str = "") -> MatrixSpec:
    try:
        forms = tuple(FormSpec(name=str(n), arm=f.get("arm"), axes=dict(f.get("axes") or {}),
                               startable=bool(f.get("startable", True)), why_not=str(f.get("why_not", "")))
                      for n, f in (value.get("forms") or {}).items())
        spec = MatrixSpec(forms=forms, bs=tuple(int(b) for b in value["bs"]),
                          depths=tuple(depth_label(d) for d in value["depths"]),
                          texts=tuple(str(t) for t in value["texts"]),
                          temps=tuple(str(t) for t in value.get("temps") or ("warm",)),
                          capacity={str(f): {depth_label(d): int(b) for d, b in (m or {}).items()}
                                    for f, m in (value.get("capacity") or {}).items()},
                          provenance=provenance)
    except (KeyError, TypeError, ValueError) as exc:
        raise FormMeasuresError(f"{MATRIX_RECORD}: malformed ({exc})") from None
    for fs in spec.forms:
        axes_key(fs.axes)
        if not fs.startable and not fs.why_not:
            raise FormMeasuresError(f"{MATRIX_RECORD}: form {fs.name!r} startable=false needs why_not")
    for t in spec.texts:
        if t not in TEXTS:
            raise FormMeasuresError(f"{MATRIX_RECORD}: text {t!r} not one of {TEXTS}")
    return spec


def matrix_spec(profile: str, fmt: str, records_dir: Optional[str] = None) -> MatrixSpec:
    """The profile's ``FORM_MATRIX_AXES`` record for weight format ``fmt``
    (a format-scoped row, so it never becomes a registry constant). No record
    = refused, not a default matrix (the cells a rig has are a profile fact)."""
    from sglang.srt.weg2 import profile_records as pr

    kw = {"records_dir": records_dir} if records_dir else {}
    for r in pr.records(profile, **kw):
        if r.name == MATRIX_RECORD and r.fmt == fmt:
            return matrix_spec_from_value(r.value, provenance=f"{profile}:{r.name}[{fmt}] ({r.provenance})")
    raise FormMeasuresError(f"profile {profile!r} has no {MATRIX_RECORD} record for fmt {fmt!r}")


def structural_state(spec: MatrixSpec, fs: FormSpec, ax: Mapping[str, object]) -> Tuple[Optional[str], str]:
    """(state, reason) a cell has WITHOUT measuring it: not startable, or
    beyond the form's KV capacity; (None, "") = has to be measured."""
    if not fs.startable:
        return STATE_IMPOSSIBLE_START, fs.why_not
    cap = (spec.capacity.get(fs.name) or {}).get(str(ax["depth"]))
    if cap is not None and int(ax["bs"]) > cap:
        return STATE_IMPOSSIBLE_CAPACITY, (f"KV of form {fs.name} holds bs<={cap} at depth {ax['depth']} "
                                           f"({spec.provenance})")
    return None, ""


def _parse_range(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if ".." in part:
            a, b = part.split("..", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def select(selection: str, spec: MatrixSpec, store: FormMeasuresV3, identity: Identity, *,
           image_rev: Optional[str] = None, now: Optional[float] = None
           ) -> List[Tuple[FormSpec, Dict[str, object]]]:
    """The cells a calibration run measures (structurally impossible ones are
    never selected -- :func:`mark_structural` records them):

    * ``all`` -- every cell of the profile matrix
    * ``missing`` -- cells without a measured value for this identity
    * ``stale[:<n>d|:rev]`` -- measured cells older than n days (default 7) or,
      with ``:rev``, measured on another image rev than ``image_rev``; plus
      missing ones
    * ``key=v1,v2;key=a..b;...`` -- explicit filter over form/bs/depth/text/temp
    """
    sel = selection.strip()
    cells = [(fs, ax) for fs, ax in spec.all_cells() if structural_state(spec, fs, ax)[0] is None]
    have = store.cells(identity)

    def measured(ax):
        c = have.get(axes_key(ax))
        return c if c is not None and c.state == STATE_MEASURED else None

    if sel == "all":
        return cells
    if sel == "missing":
        return [(fs, ax) for fs, ax in cells if measured(ax) is None]
    if sel.startswith("stale"):
        arg = sel[len("stale"):].lstrip(":") or "7d"
        now = time.time() if now is None else now
        out = []
        for fs, ax in cells:
            c = measured(ax)
            if c is None:
                out.append((fs, ax))
            elif arg == "rev":
                if image_rev is None:
                    raise FormMeasuresError("stale:rev needs image_rev")
                if (c.image_rev or "") != image_rev:
                    out.append((fs, ax))
            else:
                m = re.fullmatch(r"(\d+)d", arg)
                if not m:
                    raise FormMeasuresError(f"stale:{arg!r}: expected '<n>d' or 'rev'")
                age_s = now - _parse_at(c.at)
                if age_s > int(m.group(1)) * 86400:
                    out.append((fs, ax))
        return out
    filt: Dict[str, set] = {}
    for part in sel.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise FormMeasuresError(f"selection part {part!r}: expected key=values "
                                    f"(or all|missing|stale[:Nd|:rev])")
        k, v = part.split("=", 1)
        k = k.strip()
        if k == "bs":
            filt[k] = set(_parse_range(v))
        elif k == "depth":
            filt[k] = {depth_label(x.strip()) for x in v.split(",") if x.strip()}
        elif k in ("form", "text", "temp"):
            filt[k] = {x.strip() for x in v.split(",") if x.strip()}
        else:
            raise FormMeasuresError(f"selection key {k!r}: one of form/bs/depth/text/temp")
    return [(fs, ax) for fs, ax in cells if all(ax.get(k) in vs for k, vs in filt.items())]


def mark_structural(spec: MatrixSpec, store: FormMeasuresV3, identity: Identity) -> int:
    """Record every structurally impossible cell of the matrix as such.
    Returns how many cells were (re)written."""
    n = 0
    for fs, ax in spec.all_cells():
        st, why = structural_state(spec, fs, ax)
        if st is None:
            continue
        cur = store.lookup(identity, **ax)
        if cur is not None and cur.state == STATE_MEASURED:
            continue  # it DID run -- the measurement stands over the structure
        store.upsert(identity, Cell(axes=ax, state=st, reason=why, source=spec.provenance))
        n += 1
    return n


def _parse_at(at: Optional[str]) -> float:
    if not at:
        return 0.0
    try:
        return time.mktime(time.strptime(at[:19], "%Y-%m-%dT%H:%M:%S")) - time.timezone
    except ValueError:
        return 0.0


def plan_env(cells: Sequence[Tuple[FormSpec, Mapping[str, object]]], *, points_label=None) -> Dict[str, str]:
    """The run_matrix.sh knobs for a selection: ``ARMS`` (forms with an arm)
    and ``LADDER_BANDS`` ("points:bs;points:bs", points = <text>@<depth>).
    Bands are per (depth, bs set) so a depth runs only the bs the selection
    asks for."""
    arms: List[str] = []
    bands: Dict[Tuple[int, ...], List[str]] = {}
    per_point: Dict[str, set] = {}
    for fs, ax in cells:
        if fs.arm is None:
            continue
        if fs.arm not in arms:
            arms.append(fs.arm)
        point = f"{ax['text']}@{ax['depth']}"
        per_point.setdefault(point, set()).add(int(ax["bs"]))
    for point, bss in per_point.items():
        bands.setdefault(tuple(sorted(bss)), []).append(point)
    band_s = ";".join(f"{','.join(sorted(pts))}:{','.join(str(b) for b in bss)}"
                      for bss, pts in sorted(bands.items()))
    return {"ARMS": " ".join(arms), "LADDER_BANDS": band_s}


# ---------------------------------------------------------------------------
# import a calibration run (run_matrix.sh output directory)

RE_ROUND = re.compile(
    r"Decode rank batch, rank: (?P<rank>\d+), #round: (?P<round>\d+), t: (?P<t>[0-9.]+), "
    r"bs: (?P<bs>\d+), #rows: (?P<rows>\d+), #fwd: (?P<fwd>\d+), gpu-ms: (?P<ms>[0-9.]+)"
    r"(?: \(compute (?P<compute>[0-9.]+), wait (?P<wait>[0-9.]+)\))?")
RE_ACC = re.compile(r"Decode batch.*?accept len: (?P<acc>[0-9.]+)")
#: A stream shorter than this fraction of the ladder's max_tokens ended at EOS.
SHORT_FRACTION = 0.95

RE_T_PREFIX = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def parse_rounds(log_path: str) -> Tuple[List[Dict[str, object]], List[Tuple[float, float]]]:
    """Decode rounds as ``{t, bs, rank -> (ms, compute, wait)}`` and accept
    lengths as ``(t, acc)``. The round time of a round is its slowest rank
    (the rank the round waits for), as eval_form.py counts it."""
    rounds: Dict[int, Dict[str, object]] = {}
    acc: List[Tuple[float, float]] = []
    if not os.path.exists(log_path):
        return [], []
    for line in open(log_path, errors="ignore"):
        m = RE_ROUND.search(line)
        if m:
            r = rounds.setdefault(int(m["round"]), {"t": float(m["t"]), "bs": int(m["bs"]), "ranks": {}})
            r["ranks"][int(m["rank"])] = (float(m["ms"]),
                                          float(m["compute"]) if m["compute"] else None,
                                          float(m["wait"]) if m["wait"] else None)
            continue
        a = RE_ACC.search(line)
        if a:
            p = RE_T_PREFIX.match(line)
            t = (time.mktime(time.strptime(p.group(1), "%Y-%m-%d %H:%M:%S")) - time.timezone) if p else 0.0
            acc.append((t, float(a["acc"])))
    return [rounds[k] for k in sorted(rounds)], acc


def _pct(xs: Sequence[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def import_calib_run(run_dir: str, arms: Mapping[str, Tuple[Identity, FormSpec]],
                     store: FormMeasuresV3, *, boot: Optional[str] = None,
                     image_rev: Optional[str] = None, at: Optional[str] = None,
                     min_rounds: int = 5) -> Dict[str, int]:
    """Read ``<arm>_ladder.jsonl`` + ``<arm>_server.log`` of a calibration run
    and upsert one cell per (form, bs, depth, text, temp) and arm. A group
    with a short stream (EOS before max_tokens: ``short`` / ``streams_short``,
    29.09. ladder fix) is not a measurement of that bs and is skipped; fewer
    than ``min_rounds`` rounds leave the cell UNMEASURED with the reason."""
    stats = {"measured": 0, "unmeasured": 0, "skipped_short": 0}
    for arm, (ident, fs) in arms.items():
        lad = os.path.join(run_dir, f"{arm}_ladder.jsonl")
        if not os.path.exists(lad):
            continue
        rows = [json.loads(l) for l in open(lad) if l.strip()]
        rounds, acc = parse_rounds(os.path.join(run_dir, f"{arm}_server.log"))
        groups: Dict[Tuple[object, ...], Dict[str, object]] = {}
        for r in rows:
            if r.get("role") != "stream" or r.get("bs") is None:
                continue
            key = (r.get("point"), int(r["bs"]), r.get("temp"), r.get("rep"))
            g = groups.setdefault(key, {"streams": [], "kind": r.get("kind"),
                                        "depth": r.get("target_tokens")})
            g["streams"].append(r)
        for r in rows:
            if r.get("role") == "group" and r.get("bs") is not None:
                key = (r.get("point"), int(r["bs"]), r.get("temp"), r.get("rep"))
                if key in groups and int(r.get("streams_short") or 0) > 0:
                    groups[key]["short"] = True
        # Ladders from before the 29.09. ignore_eos fix carry no short flag:
        # a stream that ended well under the run's max_tokens (the largest
        # completion of the file) stopped at EOS and is not a measurement of
        # that bs either (nvfp4form 09290630: prose bs2 streams of 28 tokens).
        ladder_max = max((int(r.get("completion_tokens") or 0) for r in rows if r.get("role") == "stream"),
                         default=0)
        cellsamples: Dict[Tuple[object, ...], Dict[str, list]] = {}
        for (point, bs, temp, _rep), g in groups.items():
            ss = g["streams"]
            too_short = any(int(s.get("completion_tokens") or 0) < SHORT_FRACTION * ladder_max for s in ss)
            if (g.get("short") or too_short or any(s.get("short") for s in ss)
                    or any(s.get("error") for s in ss)):
                stats["skipped_short"] += 1
                continue
            lo = min(float(s["t_send"]) for s in ss)
            hi = max(float(s["t_end"]) for s in ss)
            text = str(g["kind"]) if g.get("kind") else str(point).split("@")[0]
            depth = depth_label(int(g["depth"])) if g.get("depth") else depth_label(str(point).split("@")[1])
            ck = (text, depth, bs, temp)
            acc_s = cellsamples.setdefault(ck, {"round": [], "comp": [], "wait": [], "acc": []})
            for rd in rounds:
                if rd["bs"] != bs or not (lo <= rd["t"] <= hi):
                    continue
                ranks = rd["ranks"]
                acc_s["round"].append(max(v[0] for v in ranks.values()))
                if all(v[1] is not None for v in ranks.values()):
                    acc_s["comp"].append([ranks[k][1] for k in sorted(ranks)])
                    acc_s["wait"].append([ranks[k][2] for k in sorted(ranks)])
            acc_s["acc"].extend(a for t, a in acc if lo - 1 <= t <= hi + 1)
        for (text, depth, bs, temp), s in cellsamples.items():
            ax = dict(fs.axes)
            ax.update({"form": fs.name, "bs": bs, "depth": depth, "text": text, "temp": temp})
            src = f"{os.path.basename(os.path.normpath(run_dir))}/{arm}"
            if len(s["round"]) < min_rounds:
                store.upsert(ident, Cell(axes=ax, state=STATE_UNMEASURED, n=len(s["round"]), source=src,
                                         reason=f"only {len(s['round'])} rounds < {min_rounds}", boot=boot,
                                         at=at, image_rev=image_rev))
                stats["unmeasured"] += 1
                continue
            nranks = len(s["comp"][0]) if s["comp"] else 0
            comp = [round(statistics.median(x[i] for x in s["comp"]), 2) for i in range(nranks)] or None
            wait = [round(statistics.median(x[i] for x in s["wait"]), 2) for i in range(nranks)] or None
            store.upsert(ident, Cell(
                axes=ax, state=STATE_MEASURED, round_ms_median=round(statistics.median(s["round"]), 2),
                n=len(s["round"]), p10=_pct(s["round"], 0.1), p90=_pct(s["round"], 0.9),
                compute_ms=comp, wait_ms=wait,
                accept_len=round(statistics.median(s["acc"]), 3) if s["acc"] else None,
                boot=boot, at=at, image_rev=image_rev, source=src))
            stats["measured"] += 1
    return stats


# ---------------------------------------------------------------------------
# Next Flash miss terms: from RECORDS, never from log lines (NF 29.09.)

HEAT_KIND = "moe_heat"   # layers/moe/pool_heat.py RECORD_KIND (#276, 3c5ebeac97)


def heat_misses(paths: Iterable[str]) -> Dict[int, Dict[str, float]]:
    """Per D rank: ``{not_local, lanes, steps, misses_per_step}`` summed over
    the given #276 heat records (``moe_heat_<group>_tp<r>_*.json``, one per D
    sleep). ``not_local`` = routed lanes to an expert this rank does not own
    (under the owned cut: a miss). A record of another kind/version is
    refused, a record that counted no step is skipped.

    What it CANNOT give: the split per (bs, depth, text) cell -- the record
    is per PHASE (flushed at D's sleep), and a D-only calibration has no
    phase boundary inside a run. The import does not attach it to cells;
    ``misses_per_seat`` stays empty (= not measured) until the instrument
    gives a per-group snapshot (docs/weg2/FORM_MEASURES_V3_FORMAT.md §4)."""
    out: Dict[int, Dict[str, float]] = {}
    for p in paths:
        with open(p) as f:
            rec = json.load(f)
        if rec.get("kind") != HEAT_KIND or int(rec.get("version", 0)) != 1:
            raise FormMeasuresError(f"{p}: not a {HEAT_KIND} v1 record")
        r = int(rec["rank"])
        acc = out.setdefault(r, {"not_local": 0.0, "lanes": 0.0, "steps": 0.0})
        steps = max((int(L.get("steps") or 0) for L in rec.get("layers") or ()), default=0)
        if steps <= 0:
            continue
        acc["steps"] += steps
        for L in rec.get("layers") or ():
            acc["not_local"] += int(L.get("not_local") or 0)
            acc["lanes"] += sum(int(c) for c in (L.get("counts") or ())) + int(L.get("not_local") or 0)
    for acc in out.values():
        acc["misses_per_step"] = acc["not_local"] / acc["steps"] if acc["steps"] else 0.0
    return out


# ---------------------------------------------------------------------------
# CLI: plan (the --calib selection -> run_matrix.sh env), import, show


def _identity_arg(s: str) -> Identity:
    if os.path.exists(s):
        with open(s) as f:
            return Identity.from_json(json.load(f))
    return Identity.from_json(json.loads(s))


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m sglang.srt.weg2.form_measures",
                                 description="Form-Messmatrix v3: plan a calibration (--calib), import a run, show.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="--calib selection -> run_matrix.sh env (ARMS, LADDER_BANDS)")
    p.add_argument("--store", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--fmt", required=True, help="weight format of the matrix record (e.g. nvfp4, int8)")
    p.add_argument("--identity", required=True, help="JSON or path to JSON (Identity)")
    p.add_argument("--calib", required=True, help="all | missing | stale[:Nd|:rev] | key=v;...")
    p.add_argument("--image-rev")
    p.add_argument("--records-dir")
    p.add_argument("--mark-structural", action="store_true",
                   help="also write the not-startable / over-capacity cells into the store")
    i = sub.add_parser("import", help="import a run_matrix.sh output directory")
    i.add_argument("--store", required=True)
    i.add_argument("--profile", required=True)
    i.add_argument("--fmt", required=True)
    i.add_argument("--identity-map", required=True,
                   help="JSON {arm: {identity: {...}, form: <form name of the profile matrix>}}")
    i.add_argument("--run-dir", required=True)
    i.add_argument("--boot")
    i.add_argument("--image-rev")
    i.add_argument("--at", help="ISO-8601 Z of the run (default: now)")
    i.add_argument("--records-dir")
    s = sub.add_parser("show")
    s.add_argument("--store", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "show":
        fm = load(a.store)
        for _, (ident, cells) in sorted(fm.records.items()):
            by = {}
            for c in cells.values():
                by[c.state] = by.get(c.state, 0) + 1
            print(json.dumps(ident.to_json(), sort_keys=True), by)
        return 0
    spec = matrix_spec(a.profile, a.fmt, a.records_dir)
    fm = load(a.store)
    if a.cmd == "plan":
        ident = _identity_arg(a.identity)
        if a.mark_structural:
            n = mark_structural(spec, fm, ident)
            fm.save(a.store)
            print(f"# {n} structural cells (unmoeglich) written to {a.store}", file=sys.stderr)
        cells = select(a.calib, spec, fm, ident, image_rev=a.image_rev)
        env = plan_env(cells)
        print(f"# {len(cells)} cells selected by --calib {a.calib!r} ({spec.provenance})", file=sys.stderr)
        for k, v in env.items():
            print(f"{k}={json.dumps(v)}")
        return 0
    if a.cmd == "import":
        with open(a.identity_map) if os.path.exists(a.identity_map) else _StringIO(a.identity_map) as f:
            raw = json.load(f)
        forms = {fs.name: fs for fs in spec.forms}
        arms = {}
        for arm, m in raw.items():
            if m["form"] not in forms:
                raise FormMeasuresError(f"arm {arm}: form {m['form']!r} not in {sorted(forms)}")
            arms[arm] = (Identity.from_json(m["identity"]), forms[m["form"]])
        at = a.at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        st = import_calib_run(a.run_dir, arms, fm, boot=a.boot, image_rev=a.image_rev, at=at)
        fm.save(a.store)
        print(json.dumps(st))
        return 0
    return 2


class _StringIO:
    def __init__(self, s):
        import io

        self._f = io.StringIO(s)

    def __enter__(self):
        return self._f

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    sys.exit(main())
