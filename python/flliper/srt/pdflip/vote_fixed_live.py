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
   Before that (cold start) the floor line holds, byte for byte, and the meter still measures and persists.
   Chronologically on INT8 drei the gate opens at the third qualifying boot with 238 = the max of all seven boots
   (incl. the out-of-sample _1024_110644, 221). Many passes of ONE boot never open it.
4. ONE F FOR THE GROUP (R11). Every rank adds ``-ceil(own F)`` to the scheduler's existing packed MIN reduce (one int,
   no new collective); the group max feeds back into each rank's line at the next vote (one pass late). Cards of one
   class share one store entry family (karten-einig); a form is one D inventory, so F never crosses inventories.
5. FORM = WHAT DECIDES THE BYTES. ``form_material``: profile, the H87 footprint key (config + quant sidecars + (name,
   dtype, shape) of every tensor -- the quant format is IN it; review: the corridor footprint digest is not), the D
   inventory, KV dtype, draft. No footprint (unreadable checkpoint) = no form = no store (fail closed).

Persistence: one JSON per (form, boot, rank) under ``<dir>/vote_fixed_live/<form>/``, written atomically when the
rank's F rises and at pass-count milestones; the reading list is the Pareto front of (rows up, cost down) -- exact for
every rate >= 0. Never raises into a forward.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

MARKER = "VOTE-FIXED-LIVE"
SCHEMA = "vote-fixed-live/1"

#: the switch (rank and launcher): default ON; =0 -> the seed alone (VOTE-LINE-SINGLE 9a25171a59 behaviour), no store
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
    _open: Optional[List] = None                               # [rows, post, end, trimmed, rate] of the open reading
    ver: int = 0                # bumps on every reading change
    persisted_f: Optional[float] = None   # F at the last store write (extend_trim persists on a rise)
    _memo: Optional[Tuple[Tuple[int, float], Optional[float]]] = None

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
        self._open = [int(rows), v[0], float(end_mib), bool(v[2]), v[1]]
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
        """Close the open reading; returns a short description when it was counted (else ``None``)."""
        o, self._open = self._open, None
        if o is None:
            return None
        rows, post, end, trimmed, rate = o
        self.paired += 1
        if self.paired == 1:
            return None          # the process's first pass: resident growth (offline basis), never counted
        if not trimmed:
            self.untrimmed += 1
            return None
        cost = float(post) - float(end)
        self.counted += 1
        if rows not in self.readings or cost > self.readings[rows]:
            self.readings[rows] = cost
            self.ver += 1
        ex = cost - float(rate) * rows
        if self.worst is None or ex > self.worst[1] - float(rate) * self.worst[0]:
            self.worst = (int(rows), cost, float(post), float(end))
        return f"rows={rows} cost={cost:.0f} post={post:.0f} end={end:.0f} excess={ex:.0f}"

    # -- reading -----------------------------------------------------------------------------------------------
    def fixed_mib(self, rate: float) -> Optional[float]:
        """F at ``rate``, memoised per (readings version, rate): the reduce reads it every scheduler iteration."""
        key = (self.ver, float(rate))
        if self._memo is not None and self._memo[0] == key:
            return self._memo[1]
        f = fixed_at(self.readings.items(), rate)
        self._memo = (key, f)
        return f

    def snapshot(self, rate: float, **meta) -> dict:
        doc = {"schema": SCHEMA, "rank": int(self.rank), "counted": int(self.counted), "paired": int(self.paired),
               "untrimmed": int(self.untrimmed), "rate": float(rate),
               "readings": [[int(r), round(float(c), 1)] for r, c in pareto(self.readings)],
               "fixed_mib": None if self.fixed_mib(rate) is None else round(float(self.fixed_mib(rate)), 1),
               "worst": None if self.worst is None else list(self.worst), "t_unix": int(time.time())}
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


def doc_path(root: str, form: str, boot: str, rank: int) -> str:
    return os.path.join(store_dir(root, form), f"{_safe(boot)}.tp{int(rank)}.json")


def write_doc(path: str, doc: Mapping[str, object]) -> None:
    """Atomic (tmp + replace); raises on I/O errors (the caller names them)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, sort_keys=True)
    os.replace(tmp, path)


def load_docs(root: str, form: str) -> List[dict]:
    """Every readable doc of ``form`` (an unreadable file is skipped and named at debug)."""
    d = store_dir(root, form)
    out: List[dict] = []
    try:
        names = sorted(os.listdir(d))
    except OSError:
        return out
    for n in names:
        if not n.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, n), "r", encoding="utf-8") as fh:
                doc = json.load(fh)
            if isinstance(doc, dict) and doc.get("schema") == SCHEMA:
                out.append(doc)
        except Exception as exc:  # noqa: BLE001
            logger.debug("%s store doc %s skipped: %s", MARKER, n, exc)
    return out


@dataclass(frozen=True)
class StoreVerdict:
    fixed_mib: Optional[int]       # ceil of the max over EVERY reading (rule 2), None without a reading
    boots: Tuple[str, ...]         # every boot with a doc
    qualifying: Tuple[str, ...]    # boots with >= MIN_PASSES_PER_BOOT counted passes (rule 3)
    passes: int                    # counted passes over every doc
    worst: str                     # where the max came from

    @property
    def gate_open(self) -> bool:
        return self.fixed_mib is not None and len(self.qualifying) >= MIN_BOOTS

    def describe(self) -> str:
        return (f"F={'-' if self.fixed_mib is None else self.fixed_mib} MiB over {self.passes} passes of "
                f"{len(self.boots)} boots, {len(self.qualifying)}/{MIN_BOOTS} qualifying (>= {MIN_PASSES_PER_BOOT} "
                f"passes) -> gate {'OPEN' if self.gate_open else 'closed'}"
                + (f"; max at {self.worst}" if self.worst else ""))


def aggregate(docs: Iterable[Mapping[str, object]], rate: float) -> StoreVerdict:
    """Rule 2 + 3 over the persisted docs, F priced at ``rate`` (the launcher's START rate -- the vote never prices
    below it, so F at start is the larger, safe side)."""
    per_boot: Dict[str, int] = {}
    best: Optional[float] = None
    where = ""
    n = 0
    for d in docs:
        boot = str(d.get("boot") or "?")
        c = int(d.get("counted") or 0)
        per_boot[boot] = per_boot.get(boot, 0) + c
        n += c
        rd = [(int(r), float(x)) for r, x in (d.get("readings") or [])]
        f = fixed_at(rd, rate)
        if f is not None and (best is None or f > best):
            best = f
            where = f"{boot} tp{d.get('rank', '?')} ({d.get('card', '?')})"
    qual = tuple(sorted(b for b, c in per_boot.items() if c >= MIN_PASSES_PER_BOOT))
    return StoreVerdict(None if best is None else int(math.ceil(best - 1e-9)), tuple(sorted(per_boot)), qual, n,
                        where)


def import_passes(root: str, form: str, boot: str, passes: Mapping[int, Sequence], cards: Sequence[str] = (),
                  rate: Optional[float] = None) -> List[str]:
    """Bootstrap the store from an old boot's D log (``budget_rest.extend_passes``) through the meter's own events;
    returns the written paths. Offline tool (CLI ``--import``), never on a rank."""
    out = []
    for r in sorted(passes):
        ps = list(passes[r])
        if not ps:
            continue
        m = feed_passes(Meter(rank=int(r)), ps)
        rt = float(rate if rate is not None else ps[0].rate)
        p = doc_path(root, form, boot, r)
        write_doc(p, m.snapshot(rt, form=form, boot=boot, card=(cards[r] if r < len(cards) else "?"),
                                source="import"))
        out.append(p)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``--show <root> <form> [rate]`` or ``--import <root> <form> <D.log>...`` (boot = log basename)."""
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) >= 3 and args[0] == "--show":
        rate = float(args[3]) if len(args) > 3 else 0.3091
        print(f"{MARKER} {args[2]}: {aggregate(load_docs(args[1], args[2]), rate).describe()}")
        return 0
    if len(args) >= 4 and args[0] == "--import":
        from flliper.srt.pdflip import budget_rest

        for path in args[3:]:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                ps = budget_rest.extend_passes(fh.read())
            boot = os.path.basename(path).split(".")[0]
            for p in import_passes(args[1], args[2], boot, ps):
                print(f"{MARKER} imported {p}")
        return 0
    print(main.__doc__)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
