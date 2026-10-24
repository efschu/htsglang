"""VOTE-FIXED-LIVE (24.10., desk/27b-vote-fixed-live-1024, follow-up S4 of the VOTE-LINE-SINGLE review): the fixed
part F of the 27B width-vote line, measured by the rank at run time and persisted per form and card.

WHAT F IS (unchanged from VOTE-LINE-SINGLE, budget_rest.vote_fixed_from_passes). The vote funds ``rows`` from the
driver free ``post`` it reads (after its own trim): ``rows = floor((post - line) / rate)``. What the pass then takes
from the driver above ``rate x rows`` is the admission rest the forward trim could not return plus the forward's own
fixed part::

    excess = post - end - rate x rows        end = driver free at the forward's end (and after the draft extend)

and ``line = near-OOM 400 + F`` with ``F >= excess`` leaves every such pass at ``end >= 400``. Offline that was the
max over six INT8-drei boots (238), a record that only covers the form it was measured on (review M1: int4ar /
NVFP4 duo measured 223-252) and that nobody re-measures.

WHY NOT NF'S READING. NF (VOTE-MEASURED-LINE 64bd5676a4, LEND-COST 9b6855e445) prices the TORCH cost of the extend
from AFTER the forward trim (``max(peak - start allocated, reserved growth) + draft reserved growth``). On the 27B the
line also has to carry what sits between the vote's ``post`` and the forward trim (the admission rest, ~190-250 MiB
of the 238): so the 27B reading is the DRIVER drop from the vote's post to the forward's end -- the very quantity the
offline record measured, now read by the rank itself. Same ratchet, same affine bound (``F = max_i(cost_i - rate x
rows_i)``), different instrument.

THE METER (:class:`Meter`, pure; fed by extend_trim / vram_peak_window hooks):

* ``on_vote(post, rate)``    -- every vote that read the card (also when it did not bind) opens a pending pass;
* ``on_trim()``              -- the forward trim fired for that pass (the record's denominator: trimmed passes);
* ``on_chunk(rows, end)``    -- the target extend's window closed: the pass is paired, its reading opens;
* ``on_draft(free)``         -- a draft forward after the chunk (the MTP / DFlash draft extend of the SAME pass) lowers
                                the reading's end (review VOTE-LINE-SINGLE (b): offline it fell into the next window);
* ``on_target_decode()``     -- a target decode / verify forward: commits the open reading and unpairs a pending vote
                                (decode work between vote and chunk is not the pass's);
* ``on_window(flip / idle)`` -- commits and unpairs; a ``round`` window closed right before a chunk only commits (it
                                holds decode rounds that ended BEFORE the vote -- those already unpaired it).

A committed reading counts when its forward ran after a trim and it is not the process's first paired pass (resident
first-touch growth -- the next post is lower by it; the booked rest carries it), exactly the offline basis.

THE RULES (danger direction OOM, review _080355: one boot, 481 trimmed passes, F 155 -- failed on _1022_003738 238):

1. RAISE ONLY. In the boot F is ``max(seed, own readings, group max)``; a reading above F raises the line at the next
   vote. Nothing in a boot lowers it.
2. EVERY READING COUNTS UPWARD. The store's F is the max over EVERY persisted reading of the form -- also from boots
   that are too short to count for the gate (int4ar _091424: 11 passes, 252).
3. THE GATE COUNTS BOOTS, NOT PASSES. A form's live F becomes a seed (the single line on a measured floor, and the
   budget's delta) only with :data:`MIN_BOOTS` boots of at least :data:`MIN_PASSES_PER_BOOT` trimmed passes each.
   Before that (cold start) the floor line holds (value-equal to the base), and the meter still measures and persists.
   Chronologically on INT8 drei the gate opens at the third qualifying boot with 238 = the max of all seven boots
   (incl. the out-of-sample _1024_110644, 221). Many passes of ONE boot never open it.
4. ONE F FOR THE GROUP (R11). Every rank adds ``-ceil(own F)`` to the scheduler's existing packed MIN reduce (one int,
   no new collective); the group max feeds back into each rank's line at the next vote (one pass late). Cards of one
   class share one store entry family (karten-einig); a form is one D inventory, so F never crosses inventories.
5. FORM = WHAT DECIDES THE BYTES. ``form_material``: profile, the H87 footprint key (config + quant sidecars + (name,
   dtype, shape) of every tensor -- the quant format is IN it; review: the corridor footprint digest is not), the D
   inventory, KV dtype, draft. No footprint (unreadable checkpoint) = no form = no store (fail closed).

Persistence: one JSON per (form, boot, rank, process) under ``<dir>/vote_fixed_live/<form>/``, written atomically
and MERGED with what the file holds when the rank's F rises, a reading is quarantined and at pass-count milestones
(and at exit); ``boot`` is the FULL boot name (``boot_weg2_<tag>_<rev>_<date>_<time>``) in the rank and in
``--import`` (review M1). The reading list is the Pareto front of (rows up, cost down) -- exact for every rate >= 0.
Never raises into a forward.

REVIEW ROUND (24.10., review-vote-fixed-live-1024.md):

* M2 QUARANTINE: a reading whose excess is above :data:`QUARANTINE_EXCESS_MIB` raises THIS boot's line (OOM-safe) but
  is persisted apart (``quarantined``), named loudly by the rank and the launcher and never seeds; ``--drop`` removes a
  boot. NF A5: a draft drop larger than the target forward's own drop is a foreign tail -- the chunk end stands.
* M3 IMPORT: ``--import`` counts a boot only when :func:`verify_logs` accepts its logs (the front names the digest,
  or checkpoint + every D rank's card total match the form's sidecar ``form.json`` the launcher writes); else the doc
  is ``verified: false`` and counts for neither F nor the gate.
* S1: :func:`aggregate` leaves a malformed doc out ALONE (named), never the whole store.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MARKER = "VOTE-FIXED-LIVE"
SCHEMA = "vote-fixed-live/1"

#: the switch (rank and launcher): default ON; =0 -> the seed alone, VALUE-equal to VOTE-LINE-SINGLE 9a25171a59 (line, env,
#: budgets; review S5: +1 neutral reduce slot carrying 0 and one 'store off' log line remain), no store
ENV_ON = "FLLIPER_PDFLIP_VOTE_FIXED_LIVE"
#: the form digest the launcher computed (rank: the store directory's key); unset -> the meter measures, persists nothing
ENV_FORM = "FLLIPER_PDFLIP_VOTE_FIXED_FORM"
#: the D inventory, positional by rank (rank r writes as card ENV_CARDS[r])
ENV_CARDS = "FLLIPER_PDFLIP_VOTE_FIXED_CARDS"
#: the boot tag (launcher --tag)
ENV_BOOT = "FLLIPER_PDFLIP_VOTE_FIXED_BOOT"
#: the store root (launcher resolves it once and hands it to the ranks, so both sides read one directory)
ENV_DIR = "FLLIPER_PDFLIP_VOTE_FIXED_DIR"

#: rule 3: boots of the form the gate needs ...
MIN_BOOTS = 3
#: ... each with at least this many counted (trimmed, non-first) passes over its ranks
MIN_PASSES_PER_BOOT = 20
#: persist at these counts (besides every rise of F)
PERSIST_EVERY = 50
#: review M2: the quarantine ceiling on a reading's excess (MiB above rate x rows) -- 2 x D_EXTEND_VOTE_FIXED_MIB of
#: INT8 drei (238). Every measured boot of four forms lies at 153-252 (<= 1.06 x 238); a reading far above it is
#: more likely a foreign allocation on the card (a dormant P's growth, a gpuq window with mib, lazy NCCL/cuBLAS)
#: than an extend: it still raises THIS boot's line (OOM-safe), is persisted as ``quarantined`` (named loudly, never a
#: seed) and can be dropped (CLI ``--drop``).
QUARANTINE_EXCESS_MIB = 476

SUBDIR = "vote_fixed_live"


def switch_on(environ: Optional[Mapping[str, str]] = None) -> bool:
    """:data:`ENV_ON` unless it says off (0/false/no/off); default on."""
    env = os.environ if environ is None else environ
    return str(env.get(ENV_ON, "") or "").strip().lower() not in ("0", "false", "no", "off")


def pareto(readings: Mapping[int, float]) -> List[Tuple[int, float]]:
    """The readings no other reading dominates: ``(rows_b <= rows_a and cost_b >= cost_a)`` dominates ``a`` for every
    rate >= 0 (``cost - rate x rows`` is never larger for ``a``). Sorted by rows; exact for :func:`fixed_at`."""
    out: List[Tuple[int, float]] = []
    best = -math.inf
    for r in sorted(int(x) for x in readings):
        c = float(readings[r])
        if c > best:
            out.append((r, c))
            best = c
    return out


def fixed_at(readings: Iterable[Tuple[int, float]], rate: float) -> Optional[float]:
    """``max(0, max_i(cost_i - rate x rows_i))`` (MiB), ``None`` without a reading."""
    f = None
    for r, c in readings:
        x = float(c) - float(rate) * int(r)
        f = x if f is None or x > f else f
    return None if f is None else max(0.0, f)


@dataclass
class Meter:
    """One rank's live F (pure state machine; see the module doc for the event order)."""

    rank: int = 0
    readings: Dict[int, float] = field(default_factory=dict)   # rows -> max cost (post - end), counted passes
    counted: int = 0            # trimmed, non-first passes
    paired: int = 0             # every paired pass (first included)
    untrimmed: int = 0
    worst: Optional[Tuple[int, float, float, float]] = None    # (rows, cost, post, end) of the highest excess so far
    _pending: Optional[Tuple[float, float, bool]] = None       # (post, rate, trimmed) of the open vote
    _open: Optional[List] = None             # [rows, post, end, trimmed, rate, chunk end] of the open reading
    ver: int = 0                # bumps on every reading change (quarantine included)
    persisted_f: Optional[float] = None   # F at the last store write (extend_trim persists on a rise)
    persisted_counted: int = 0            # counted at the last store write (exit flush)
    #: review M2: a reading whose excess (at its pass's rate) is above this is QUARANTINED -- it raises the line in
    #: this boot, persists apart and never seeds a later boot
    ceiling: float = 0.0
    quarantined: List[List[float]] = field(default_factory=list)   # [rows, cost, post, end, t_unix]
    tails_skipped: int = 0      # review M2 / NF A5: draft drops larger than the forward's own drop, left out
    _memo: Optional[Tuple[Tuple[int, float], Optional[float]]] = None

    def __post_init__(self):
        if not self.ceiling:
            self.ceiling = float(QUARANTINE_EXCESS_MIB)

    # -- events ------------------------------------------------------------------------------------------------
    def on_vote(self, post_mib: float, rate: float) -> Optional[str]:
        msg = self.commit()
        self._pending = (float(post_mib), float(rate), False)
        return msg

    def on_trim(self) -> None:
        if self._pending is not None:
            p, r, _ = self._pending
            self._pending = (p, r, True)

    def on_chunk(self, rows: int, end_mib: float) -> Optional[str]:
        msg = self.commit()
        v, self._pending = self._pending, None
        if v is None or int(rows) <= 0:
            return msg
        self._open = [int(rows), v[0], float(end_mib), bool(v[2]), v[1], float(end_mib)]
        return msg

    def on_draft(self, free_mib: float) -> None:
        if self._open is not None:
            self._open[2] = min(float(self._open[2]), float(free_mib))

    def on_target_decode(self) -> Optional[str]:
        msg = self.commit()
        self._pending = None
        return msg

    def on_window(self, phase: str) -> Optional[str]:
        msg = self.commit()
        if str(phase) != "round":
            self._pending = None
        return msg

    def commit(self) -> Optional[str]:
        """Close the open reading; returns a short description when it was counted (``QUARANTINED`` in it when the
        reading went to the quarantine), else ``None``."""
        o, self._open = self._open, None
        if o is None:
            return None
        rows, post, end, trimmed, rate, end_chunk = o
        self.paired += 1
        if self.paired == 1:
            return None          # the process's first pass: resident growth (offline basis), never counted
        if not trimmed:
            self.untrimmed += 1
            return None
        tail = ""
        # NF review A5: a draft "growth" larger than the target forward's own drop is not the draft extend (a foreign
        # allocation between the chunk's end and the draft's end) -- the chunk end stands, counted
        if float(end_chunk) - float(end) > float(post) - float(end_chunk):
            self.tails_skipped += 1
            tail = f" tail-skipped(draft drop {float(end_chunk) - float(end):.0f} > forward {post - end_chunk:.0f})"
            end = end_chunk
        cost = float(post) - float(end)
        self.counted += 1
        ex = cost - float(rate) * rows
        if ex > float(self.ceiling):
            self.quarantined.append([int(rows), round(cost, 1), round(float(post), 1), round(float(end), 1),
                                     int(time.time())])
            self.ver += 1
            return (f"QUARANTINED rows={rows} cost={cost:.0f} post={post:.0f} end={end:.0f} excess={ex:.0f} > ceiling "
                    f"{self.ceiling:.0f}{tail}")
        if rows not in self.readings or cost > self.readings[rows]:
            self.readings[rows] = cost
            self.ver += 1
        if self.worst is None or ex > self.worst[1] - float(rate) * self.worst[0]:
            self.worst = (int(rows), cost, float(post), float(end))
        return f"rows={rows} cost={cost:.0f} post={post:.0f} end={end:.0f} excess={ex:.0f}{tail}"

    # -- reading -----------------------------------------------------------------------------------------------
    def fixed_mib(self, rate: float) -> Optional[float]:
        """F at ``rate`` for THIS boot's line -- quarantined readings included (OOM-safe in the boot); memoised per
        (readings version, rate): the reduce reads it every scheduler iteration."""
        key = (self.ver, float(rate))
        if self._memo is not None and self._memo[0] == key:
            return self._memo[1]
        f = fixed_at(list(self.readings.items()) + [(int(q[0]), float(q[1])) for q in self.quarantined], rate)
        self._memo = (key, f)
        return f

    def clean_fixed_mib(self, rate: float) -> Optional[float]:
        """F at ``rate`` over the non-quarantined readings (what a later boot may be seeded with)."""
        return fixed_at(self.readings.items(), rate)

    def snapshot(self, rate: float, **meta) -> dict:
        f = self.clean_fixed_mib(rate)
        doc = {"schema": SCHEMA, "rank": int(self.rank), "counted": int(self.counted), "paired": int(self.paired),
               "untrimmed": int(self.untrimmed), "rate": float(rate),
               "readings": [[int(r), round(float(c), 1)] for r, c in pareto(self.readings)],
               "quarantined": [list(q) for q in self.quarantined], "tails_skipped": int(self.tails_skipped),
               "ceiling": float(self.ceiling),
               "fixed_mib": None if f is None else round(float(f), 1),
               "worst": None if self.worst is None else list(self.worst), "t_unix": int(time.time()),
               "verified": True}
        doc.update(meta)
        return doc


def feed_passes(meter: Meter, passes: Sequence) -> Meter:
    """Replay offline-paired passes (``budget_rest.ExtendPass``: post, rate, rows, end, trimmed) through the meter's
    own events -- the bridge between the log tool and the run-time reading (repro tests, :func:`import_passes`)."""
    for p in passes:
        meter.on_vote(float(p.post), float(p.rate))
        if p.trimmed:
            meter.on_trim()
        meter.on_chunk(int(p.rows), float(p.end))
    meter.commit()
    return meter


# --------------------------------------------------------------------------
# the form and the store
# --------------------------------------------------------------------------

def form_material(profile: Optional[str], footprint: Optional[str], model_key: str, inventory: Sequence[str],
                  kv_dtype: str = "", draft: str = "") -> Optional[Dict[str, object]]:
    """What decides the bytes of a D pass's fixed part, or ``None`` without a footprint (unknown = not a form)."""
    if not footprint or not inventory:
        return None
    return {"schema": SCHEMA, "profile": str(profile or ""), "footprint": str(footprint),
            "model_key": str(model_key or ""), "inventory": [str(x) for x in inventory],
            "kv_dtype": str(kv_dtype or ""), "draft": str(draft or "")}


def form_digest(material: Optional[Mapping[str, object]]) -> Optional[str]:
    if not material:
        return None
    blob = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def store_dir(root: str, form: str) -> str:
    return os.path.join(str(root), SUBDIR, str(form))


def _safe(tag: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(tag or "boot"))[:160] or "boot"


def doc_path(root: str, form: str, boot: str, rank: int, proc: str = "") -> str:
    """``<boot>.tp<rank>[.<proc>].json`` -- review M1: ``boot`` is the FULL boot name (``boot_weg2_<tag>_<rev>_<date>_
    <time>``, the same in the rank and in ``--import``), ``proc`` the writing process (pid-start ms), so a relaunch
    under the same tag or a rank restart writes its own file instead of replacing one."""
    tail = f".{_safe(proc)}" if proc else ""
    return os.path.join(store_dir(root, form), f"{_safe(boot)}.tp{int(rank)}{tail}.json")


def merge_docs(old: Mapping[str, object], new: Mapping[str, object]) -> dict:
    """Review M1: never replace -- the union of two docs of the same file: readings max per rows (Pareto front),
    quarantine union, counters max (a process's snapshot is cumulative). ``new``'s other fields win."""
    out = dict(new)
    rd: Dict[int, float] = {}
    for d in (old, new):
        for r, c in (d.get("readings") or []):
            r, c = int(r), float(c)
            if r not in rd or c > rd[r]:
                rd[r] = c
    out["readings"] = [[int(r), round(float(c), 1)] for r, c in pareto(rd)]
    q = []
    for d in (old, new):
        for item in (d.get("quarantined") or []):
            if list(item) not in q:
                q.append(list(item))
    out["quarantined"] = q
    for k in ("counted", "paired", "untrimmed", "tails_skipped"):
        out[k] = max(int(old.get(k) or 0), int(new.get(k) or 0))
    return out


def write_doc(path: str, doc: Mapping[str, object]) -> None:
    """Atomic (tmp + replace), MERGED with a readable doc already at ``path`` (review M1: a write never loses a
    reading); raises on I/O errors (the caller names them)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = dict(doc)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            old = json.load(fh)
        if isinstance(old, dict) and old.get("schema") == SCHEMA and old.get("boot") == doc.get("boot"):
            out = merge_docs(old, doc)
    except (OSError, ValueError, TypeError):
        pass
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, sort_keys=True)
    os.replace(tmp, path)


def load_docs(root: str, form: str) -> List[dict]:
    """Every readable doc of ``form`` (an unreadable file is skipped and named at debug; ``form.json`` is not a
    doc)."""
    d = store_dir(root, form)
    out: List[dict] = []
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return out
    for n in names:
        if not n.endswith(".json") or n == FORM_FILE:
            continue
        try:
            with open(os.path.join(d, n), "r", encoding="utf-8") as fh:
                doc = json.load(fh)
            if isinstance(doc, dict) and doc.get("schema") == SCHEMA:
                doc.setdefault("_file", n)
                out.append(doc)
        except Exception as exc:  # noqa: BLE001
            logger.debug("%s store doc %s skipped: %s", MARKER, n, exc)
    return out


@dataclass(frozen=True)
class StoreVerdict:
    fixed_mib: Optional[int]       # ceil of the max over every CLEAN, VERIFIED reading (rule 2), None without one
    boots: Tuple[str, ...]         # every boot with a verified doc
    qualifying: Tuple[str, ...]    # boots with >= MIN_PASSES_PER_BOOT counted passes (rule 3)
    passes: int                    # counted passes over the verified docs
    worst: str                     # where the max came from
    quarantined: Tuple[str, ...] = ()   # review M2: readings above the ceiling (never a seed), named
    skipped: Tuple[str, ...] = ()       # review S1 / M3: malformed or unverified docs, named, left out

    @property
    def gate_open(self) -> bool:
        return self.fixed_mib is not None and len(self.qualifying) >= MIN_BOOTS

    def describe(self) -> str:
        return (f"F={'-' if self.fixed_mib is None else self.fixed_mib} MiB over {self.passes} passes of "
                f"{len(self.boots)} boots, {len(self.qualifying)}/{MIN_BOOTS} qualifying (>= {MIN_PASSES_PER_BOOT} "
                f"passes) -> gate {'OPEN' if self.gate_open else 'closed'}"
                + (f"; max at {self.worst}" if self.worst else "")
                + (f"; {len(self.quarantined)} QUARANTINED reading(s) (no seed)" if self.quarantined else "")
                + (f"; {len(self.skipped)} doc(s) left out" if self.skipped else ""))


def _check_doc(d: Mapping[str, object]) -> Tuple[str, int, List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Review S1: (boot, counted, readings, quarantined) of a well-formed doc, else ``ValueError`` naming why."""
    boot = d.get("boot")
    if not isinstance(boot, str) or not boot:
        raise ValueError("no boot")
    c = d.get("counted")
    if isinstance(c, bool) or not isinstance(c, int) or c < 0:
        raise ValueError(f"counted {c!r}")

    def pairs(key, n):
        out = []
        for item in (d.get(key) or []):
            if not isinstance(item, (list, tuple)) or (len(item) != n if n else len(item) < 2):
                raise ValueError(f"{key} item {item!r}")
            r, x = item[0], item[1]
            if isinstance(r, bool) or not isinstance(r, int) or r <= 0:
                raise ValueError(f"{key} rows {r!r}")
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
                raise ValueError(f"{key} cost {x!r}")
            out.append((int(r), float(x)))
        return out

    return boot, int(c), pairs("readings", 2), pairs("quarantined", 0)


def aggregate(docs: Iterable[Mapping[str, object]], rate: float, ceiling: float = QUARANTINE_EXCESS_MIB) -> StoreVerdict:
    """Rule 2 + 3 over the persisted docs, F priced at ``rate`` (the launcher's START rate -- the vote never prices
    below it, so F at start is the larger, safe side). Review S1: a malformed doc is left out ALONE (named); review
    M3: an unverified doc (``--import`` whose logs were not this form's) counts for nothing; review M2: a reading
    whose excess at ``rate`` is above ``ceiling`` (or that the rank quarantined) never seeds, it is named."""
    per_boot: Dict[str, int] = {}
    best: Optional[float] = None
    where = ""
    n = 0
    quar: List[str] = []
    skipped: List[str] = []
    for d in docs:
        name = str(d.get("_file") or f"{d.get('boot', '?')}.tp{d.get('rank', '?')}")
        try:
            boot, c, rd, qd = _check_doc(d)
        except Exception as exc:  # noqa: BLE001 -- review S1: one bad doc, not the store
            skipped.append(f"{name}: malformed ({exc})")
            continue
        if d.get("verified") is False:
            skipped.append(f"{name}: unverified ({d.get('verify_why', 'form not checked')})")
            continue
        per_boot[boot] = per_boot.get(boot, 0) + c
        n += c
        for r, x in qd:
            quar.append(f"{boot} tp{d.get('rank', '?')} ({d.get('card', '?')}) rows={r} cost={x:.0f} (rank)")
        clean = []
        for r, x in rd:
            if x - float(rate) * r > float(ceiling):
                quar.append(f"{boot} tp{d.get('rank', '?')} ({d.get('card', '?')}) rows={r} cost={x:.0f} "
                            f"excess={x - float(rate) * r:.0f} > {float(ceiling):.0f}")
            else:
                clean.append((r, x))
        f = fixed_at(clean, rate)
        if f is not None and (best is None or f > best):
            best = f
            where = f"{boot} tp{d.get('rank', '?')} ({d.get('card', '?')})"
    qual = tuple(sorted(b for b, c in per_boot.items() if c >= MIN_PASSES_PER_BOOT))
    return StoreVerdict(None if best is None else int(math.ceil(best - 1e-9)), tuple(sorted(per_boot)), qual, n,
                        where, tuple(quar), tuple(skipped))


# --------------------------------------------------------------------------
# review M3: the form's sidecar and the import check
# --------------------------------------------------------------------------

FORM_FILE = "form.json"


def write_form(root: str, form: str, material: Mapping[str, object], totals: Sequence[int]) -> str:
    """The launcher records what the digest stands for (material + the D cards' NVML totals, positional) -- the
    yardstick ``--import`` checks old logs against."""
    path = os.path.join(store_dir(root, form), FORM_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"schema": SCHEMA, "form": form, "material": dict(material),
                   "totals": [int(t) for t in totals]}, fh, sort_keys=True)
    os.replace(tmp, path)
    return path


def read_form(root: str, form: str) -> Optional[dict]:
    try:
        with open(os.path.join(store_dir(root, form), FORM_FILE), "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc if isinstance(doc, dict) else None
    except (OSError, ValueError):
        return None


#: an NVML total in a D log may differ from the recorded one by this much and still be the same card
TOTAL_TOLERANCE_MIB = 64

_RX_MODEL = re.compile(r"--model-path\s+(\S+)")
_RX_TOTAL = re.compile(r"PDFLIP-VRAM-PEAK rank=(\d+) .*?card_total_mib=(\d+)")


def verify_logs(root: str, form: str, front_text: str, d_text: str) -> Tuple[bool, str]:
    """Review M3: is this boot of ``form``? Yes when its front log names the digest (a boot of this code line), else
    when the form's sidecar exists and the checkpoint (``--model-path`` model key) and every D rank's NVML total
    (``PDFLIP-VRAM-PEAK card_total_mib``, rank count = inventory length) match it. Pure on texts."""
    if f"{ENV_FORM}={form}" in (front_text or ""):
        return True, f"front log names {ENV_FORM}={form}"
    fd = read_form(root, form)
    if not fd:
        return False, f"no {FORM_FILE} for form {form} (a boot of the form must run first)"
    mat = fd.get("material") or {}
    m = _RX_MODEL.search(front_text or "")
    key = os.path.basename(m.group(1).rstrip("/").strip("'\"")) if m else ""
    if key != str(mat.get("model_key") or ""):
        return False, f"checkpoint {key or '?'} is not the form's {mat.get('model_key')!r}"
    totals: Dict[int, int] = {}
    for mm in _RX_TOTAL.finditer(d_text or ""):
        totals.setdefault(int(mm.group(1)), int(mm.group(2)))
    want = [int(t) for t in (fd.get("totals") or [])]
    if not want or sorted(totals) != list(range(len(want))):
        return False, f"D ranks {sorted(totals)} are not the form's {len(want)} cards"
    for r, t in totals.items():
        if abs(t - want[r]) > TOTAL_TOLERANCE_MIB:
            return False, f"D rank {r} card total {t} MiB is not the form's {want[r]} MiB"
    return True, f"checkpoint {key} and D cards {[totals[r] for r in sorted(totals)]} MiB match the form"


def boot_of_log(path: str) -> str:
    """The FULL boot name of a group log (``.../boot_weg2_<tag>_<rev>_<date>_<time>.D.log`` -> the stem) -- the
    same string the launcher hands the ranks (review M1)."""
    stem = os.path.basename(str(path))
    for suf in (".log", ".D", ".P", ".front"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    return stem


def import_passes(root: str, form: str, boot: str, passes: Mapping[int, Sequence], cards: Sequence[str] = (),
                  rate: Optional[float] = None, verified: bool = False, verify_why: str = "") -> List[str]:
    """Bootstrap the store from an old boot's D log (``budget_rest.extend_passes``) through the meter's own events;
    returns the written paths. ``verified`` (review M3) only after :func:`verify_logs` -- an unverified doc counts
    for neither the gate nor F. ``rate`` is only the doc's label; :func:`aggregate` re-prices at the launcher's start
    rate. Offline tool (CLI ``--import``), never on a rank."""
    out = []
    for r in sorted(passes):
        ps = list(passes[r])
        if not ps:
            continue
        m = feed_passes(Meter(rank=int(r)), ps)
        rt = float(rate if rate is not None else ps[0].rate)
        p = doc_path(root, form, boot, r, "import")
        write_doc(p, m.snapshot(rt, form=form, boot=boot, card=(cards[r] if r < len(cards) else "?"),
                                source="import", verified=bool(verified), verify_why=str(verify_why)))
        out.append(p)
    return out


def drop_boot(root: str, form: str, boot: str) -> List[str]:
    """Review M2: the way back -- remove every doc of ``boot`` (field or file prefix) from ``form``'s store."""
    d = store_dir(root, form)
    gone = []
    for doc in load_docs(root, form):
        n = str(doc.get("_file") or "")
        if n and (doc.get("boot") == boot or n.startswith(f"{_safe(boot)}.tp")):
            os.remove(os.path.join(d, n))
            gone.append(n)
    return gone


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``--show <root> <form> [rate]`` | ``--import <root> <form> <D.log>...`` (boot = the log's full stem; counted
    only when :func:`verify_logs` accepts the boot's front + D log) | ``--drop <root> <form> <boot>``."""
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) >= 3 and args[0] == "--show":
        rate = float(args[3]) if len(args) > 3 else 0.3091
        v = aggregate(load_docs(args[1], args[2]), rate)
        print(f"{MARKER} {args[2]}: {v.describe()}")
        for q in v.quarantined:
            print(f"{MARKER} QUARANTINED {q}")
        for k in v.skipped:
            print(f"{MARKER} left out {k}")
        return 0
    if len(args) >= 4 and args[0] == "--import":
        from flliper.srt.pdflip import budget_rest

        for path in args[3:]:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                d_text = fh.read()
            front = path[: -len(".D.log")] + ".front.log" if path.endswith(".D.log") else ""
            try:
                with open(front, "r", encoding="utf-8", errors="replace") as fh:
                    f_text = fh.read()
            except OSError:
                f_text = ""
            ok, why = verify_logs(args[1], args[2], f_text, d_text)
            boot = boot_of_log(path)
            for p in import_passes(args[1], args[2], boot, budget_rest.extend_passes(d_text), verified=ok,
                                   verify_why=why):
                print(f"{MARKER} imported {p} ({'verified' if ok else 'UNVERIFIED, counts for nothing'}: {why})")
        return 0
    if len(args) == 4 and args[0] == "--drop":
        for n in drop_boot(args[1], args[2], args[3]):
            print(f"{MARKER} dropped {n}")
        return 0
    print(main.__doc__)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
