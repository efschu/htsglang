# SPDX-License-Identifier: Apache-2.0
"""H94 -- the MEASURED AWAKE OVERSHOOT per card, keyed by the checkpoint it was measured on.

WHAT THIS IS. ``launcher.budgets_from_dc`` subtracts, per card, a term the
budget line calls ``measured_awake_overshoot``: what the awake group holds on
the card OUTSIDE its ``--rank-gpu-memory-mib`` share and outside the priced
corridor (transient floor + ``D_AWAKE_OVERSHOOT_MIB``). Until H94 that term was
two module literals of the launcher, both measured on Qwen3.8-27B:

    P_OVERSHOOT_MIB = [920, 0, 512]   boot weg2ls2b2 (27B INT8, 2026-09-07)
    D_OVERSHOOT_MIB = [489, 0, 0]     boot weg2ls4b1 (27B INT8, 2026-09-07)

and they were charged on EVERY boot, whatever it loaded. The Next-Flash release
boot fnFL2x178 printed ``- measured_awake_overshoot 920 (boot weg2ls2b2)`` in
its P budget line (P argv ``--rank-gpu-memory-mib 28208,17840,17168``) and
``- measured_awake_overshoot 489 (boot weg2ls4b1)`` in its D line: a 27B
measurement shaping the NF argv, silently (UNIFY_PLAN risk (c)).

THE RULE (memory ``27b-strikt-getrennt-von-nf``, WEG2-FORM calibration
identity): a measurement speaks only for the checkpoint it was taken on. So
the term is resolved per boot, in this order, and the line says which:

1. ``RECORD``   -- the newest ``awake_overshoot`` entry of the measured-record
   sidecar for this group whose ``model`` is THIS checkpoint (and, in a real
   launch, whose boot passes ``weg2_form.same_model_sample``: same model, same
   residue axes). It must name every card of this boot by UUID.
2. ``BUILTIN``  -- a reference in :data:`REFERENCES` whose ``family`` covers
   this checkpoint. The two 27B literals live here now, unchanged, scoped to
   the ``Qwen3.8-27B`` family (every 27B checkpoint the 27B line boots keeps
   byte-identical budgets); the NF references carry NF's own derivation.
3. ``UNMEASURED-FALLBACK`` -- 0 on every card, and the line NAMES it, with the
   command that closes the gap. Never another model's number.

THE MEASUREMENT (:func:`derive_card`). One boot's own H55 windows
(``WEG2-VRAM-PEAK``, P: ``phase=chunk``; D: ``phase=round`` and ``chunk``)
give, per rank, the card's free minimum under load (``card_free_mib``, a
``cudaMemGetInfo`` read = NVML free v2, the corridor law's instrument), the
caching allocator's unallocated reserve at that same instant
(``reserved_mib - allocated_mib``) and the retries the allocator needed
(``alloc_retries``). The same boot's budget line names the charge in force and
the graded floor (``CORRIDOR-FLOOR ... verdict_floor``). The 27B derivation
(``launcher`` comment at the literals: "charged ... so the minimum lands
mid-band") is the rule: the charge that puts the measured minimum on the
graded floor. Three cases, each NAMED:

* ``SLACK``     free_min >= floor: the card had ``free_min - floor`` MiB it did
  not need; the charge shrinks by that much, never below 0. Unclaimed slack
  beyond the charge is printed, not credited (an overshoot is a charge; a
  credit would be a second budget solve, and the corridor pass owns that).
* ``SATURATED`` free_min < floor BUT the rank's own caching allocator held at
  least the shortfall as unallocated reserve at that instant, or had to retry:
  the minimum was set by the allocator filling the card, not by what the rank
  needs. Such a minimum does not move with the budget (lower the budget and
  the cache fills the freed MiB again), so deriving a raise from it would
  RATCHET the charge up boot after boot. The charge in force is kept -- the
  operating point this checkpoint ran at, ``ooms`` as printed -- and named.
* ``SHORT``     free_min < floor with no reserve to explain it: the rank really
  needed the MiB; the charge rises by the shortfall.

Pure: nothing here reads NVML, spawns or writes except :func:`append_record`.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

#: The measured-record entry kind this module writes and reads. The sidecar is
#: shared with the dormant-image and flip-ratchet entries; those carry
#: ``rss_shmem_gib`` / ``flip_ratchet_gib`` and never this key, and
#: ``host_ledger.read_measured_record`` skips entries without them, so neither
#: reader can see the other's rows.
RECORD_KIND = "awake_overshoot"
LINE_TAG = "WEG2-OVERSHOOT"

SOURCE_RECORD = "RECORD"
SOURCE_BUILTIN = "BUILTIN"
SOURCE_FALLBACK = "UNMEASURED-FALLBACK"

VERDICT_SLACK = "SLACK"
VERDICT_SATURATED = "SATURATED"
VERDICT_SHORT = "SHORT"

#: The checkpoint family the launcher's module-level 27B measurements were
#: taken on (weg2ls2b2 / weg2ls4b1 / weg2zr2 / bsscale all ran
#: ``Qwen3.8-27B-INT8-gdncov-vocabembed``). A FAMILY, not one directory: the
#: 27B line boots FP8/GGUF/NVFP4 checkpoints of the same model on the same
#: literals, and H94 must not move a 27B budget by a MiB.
QWEN38_27B_FAMILY = "Qwen3.8-27B"
#: The Next-Flash checkpoint the NF references were measured on
#: (``planner.power_limit.NF_MODEL`` names the same directory).
NF_INT4_MODEL = "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
#: The Next-Flash NVFP4 checkpoint (profile nf-nvfp4, boots fnNV4f*): ANOTHER
#: checkpoint, so it gets its own reference and never the INT4 one.
NF_NVFP4_MODEL = "Qwen3.8-Flash-Next-NVFP4-nvidia"


def model_key(model: str) -> str:
    """The calibration identity of a checkpoint: its directory name (the same
    rule as ``weg2.form.model_key``; restated so this module stays importable
    without the launcher's import chain)."""
    return os.path.basename(str(model or "").rstrip("/").strip("'\""))


def in_family(model: str, family: str) -> bool:
    """``model`` IS ``family`` or a checkpoint of it (``family-<suffix>``)."""
    key, fam = model_key(model), str(family or "")
    return bool(fam) and (key == fam or key.startswith(fam + "-"))


@dataclass(frozen=True)
class OvershootReference:
    """A built-in, model-scoped awake overshoot of one group, per CUDA ordinal
    (ordinal 0 = the 5090, then the 3080s by NVML index -- ``order_cards``)."""

    group: str
    family: str
    mib: Tuple[int, ...]
    provenance: str


#: THE ONLY PLACE A MEASURED OVERSHOOT IS WRITTEN DOWN IN CODE.
#:
#: 27B: the two former launcher literals, byte-identical, now scoped.
#:
#: NF INT4 (H94, derived by ``weg2.tools.awake_overshoot_record`` from the
#: NF boots' own H55 windows; fnFL2x177 09729d97c4, fnFL2x178 b89592806a and
#: fnFL2h91v1 39fd662d9e give the same verdict per card, h91v1 is the newest):
#:   P  ordinal 0 (5090)  free_min 179 < floor 1055, cache at min 6936, retries 6 -> SATURATED, 920 kept
#:      ordinal 1 (nvml0) free_min  20 < floor 1095, cache at min 4714, retries 2 -> SATURATED, 0 kept
#:      ordinal 2 (nvml2) free_min 2372 >= floor 858 -> SLACK 1514 -> 512 released, 1002 unclaimed
#:   D  ordinals 0/1/2: free_min 73/56/4 under round+chunk load, allocator
#:      retries 244/32/94 (x178: 77/12/25) -> SATURATED on every card, 489/0/0 kept.
#: NF NVFP4 (fnNV4f4 fd3ad2e326, the newest NVFP4 boot with P and D; f3 agrees on P):
#:   P  ord0 free_min 55 < 1055, cache 5782 -> SATURATED 920 kept; ord1 1508 >= 1095 -> SLACK, 0;
#:      ord2 1832 >= 858 -> SLACK 974 -> 512 released
#:   D  ord0 free_min 521 < 767, cache 3176 >= shortfall 246 -> SATURATED 489 kept; ord1/ord2 SLACK, 0
#: The kept 920 / 489 are therefore NF operating points (the charge the NF
#: boots ran at, ooms 0), not the 27B claim; the H55 minimum cannot grade them
#: either way while the caching allocator fills the card (see the module
#: docstring). What would: a hoard-free instrument (expandable segments, or a
#: per-rank live-need line) -- named as the open metal point, not guessed here.
REFERENCES: Tuple[OvershootReference, ...] = (
    OvershootReference(
        "P", QWEN38_27B_FAMILY, (920, 0, 512),
        "boot weg2ls2b2",
    ),
    OvershootReference(
        "D", QWEN38_27B_FAMILY, (489, 0, 0),
        "boot weg2ls4b1",
    ),
    OvershootReference(
        "P", NF_INT4_MODEL, (920, 0, 0),
        "NF boot fnFL2h91v1 H55 windows: ord0 SATURATED 920 kept, ord1 SATURATED 0, "
        "ord2 SLACK 1514 -> 512 released (x177/x178 agree)",
    ),
    OvershootReference(
        "D", NF_INT4_MODEL, (489, 0, 0),
        "NF boot fnFL2h91v1 H55 windows: all ordinals SATURATED (allocator retries), "
        "operating point 489/0/0 kept (x177/x178 agree)",
    ),
    OvershootReference(
        "P", NF_NVFP4_MODEL, (920, 0, 0),
        "NF-NVFP4 boot fnNV4f4 H55 windows: ord0 SATURATED 920 kept, ord1 SLACK 0, "
        "ord2 SLACK 974 -> 512 released (fnNV4f3 agrees)",
    ),
    OvershootReference(
        "D", NF_NVFP4_MODEL, (489, 0, 0),
        "NF-NVFP4 boot fnNV4f4 H55 windows: ord0 SATURATED 489 kept, ord1/ord2 SLACK 0",
    ),
)


def reference_for(
    group: str, model: str, refs: Sequence[OvershootReference] = REFERENCES
) -> Optional[OvershootReference]:
    """The built-in reference of ``group`` whose family covers ``model``; the
    most specific (longest family) wins."""
    hits = [r for r in refs if r.group == group and in_family(model, r.family)]
    if not hits:
        return None
    return max(hits, key=lambda r: len(r.family))


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CardMeasurement:
    """One card of one group of one boot, everything the rule reads."""

    uuid: str
    ordinal: int
    name: str
    charged_mib: int
    floor_mib: int
    free_min_mib: int
    cache_at_min_mib: int
    retries: int
    windows: int


@dataclass(frozen=True)
class CardDerivation:
    measurement: CardMeasurement
    overshoot_mib: int
    verdict: str
    unclaimed_slack_mib: int = 0

    def text(self) -> str:
        m = self.measurement
        base = (
            f"ord{m.ordinal} {m.name} free_min {m.free_min_mib} vs floor {m.floor_mib} "
            f"(charged {m.charged_mib}, cache_at_min {m.cache_at_min_mib}, "
            f"retries {m.retries}, {m.windows} windows) -> {self.verdict} {self.overshoot_mib}"
        )
        if self.unclaimed_slack_mib:
            base += f" (unclaimed slack {self.unclaimed_slack_mib}, not credited)"
        return base


def derive_card(m: CardMeasurement) -> CardDerivation:
    """The charge that puts the measured minimum on the graded floor -- see the
    module docstring for the three cases and why SATURATED never raises."""
    charged = max(0, int(m.charged_mib))
    free_min, floor = int(m.free_min_mib), int(m.floor_mib)
    if free_min >= floor:
        slack = free_min - floor
        over = max(0, charged - slack)
        return CardDerivation(m, over, VERDICT_SLACK, max(0, slack - charged))
    shortfall = floor - free_min
    if int(m.retries) > 0 or int(m.cache_at_min_mib) >= shortfall:
        return CardDerivation(m, charged, VERDICT_SATURATED)
    return CardDerivation(m, charged + shortfall, VERDICT_SHORT)


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------


def build_record(
    *,
    group: str,
    model: str,
    boot_tag: str,
    commit: str,
    at: str,
    derivations: Sequence[CardDerivation],
    form: str = "",
    source: str = "",
) -> Dict[str, object]:
    """One sidecar entry. ``model`` is stored as its key, so a reader can
    refuse a foreign entry without any evidence directory at hand."""
    return {
        "kind": RECORD_KIND,
        "group": str(group),
        "model": model_key(model),
        "form": str(form),
        "boot_tag": str(boot_tag),
        "commit": str(commit),
        "at": str(at),
        "source": str(source),
        "overshoot_by_uuid": {d.measurement.uuid: int(d.overshoot_mib) for d in derivations},
        "cards": [
            {
                "uuid": d.measurement.uuid,
                "ordinal": int(d.measurement.ordinal),
                "name": d.measurement.name,
                "charged_mib": int(d.measurement.charged_mib),
                "floor_mib": int(d.measurement.floor_mib),
                "free_min_mib": int(d.measurement.free_min_mib),
                "cache_at_min_mib": int(d.measurement.cache_at_min_mib),
                "retries": int(d.measurement.retries),
                "windows": int(d.measurement.windows),
                "verdict": d.verdict,
                "overshoot_mib": int(d.overshoot_mib),
                "unclaimed_slack_mib": int(d.unclaimed_slack_mib),
            }
            for d in derivations
        ],
    }


def read_record(
    path: Optional[str],
    group: str,
    model: str,
    accept: Optional[Callable[[dict], bool]] = None,
) -> Tuple[Optional[dict], str]:
    """The newest ``awake_overshoot`` entry of ``group`` measured on ``model``.

    Returns ``(entry, why)``; ``entry`` is None with the reason otherwise. An
    entry of another checkpoint is skipped by its OWN ``model`` field first --
    no evidence directory is needed to refuse it -- and then by ``accept``
    (``weg2_form.same_model_sample``: the boot's own front log, residue axes).
    """
    if not path:
        return None, "no measured-record sidecar path"
    try:
        with open(path) as f:
            data = json.load(f)
    except OSError:
        return None, f"sidecar {path} unreadable"
    except ValueError:
        return None, f"sidecar {path} is not JSON"
    entries = data.get("samples") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return None, f"sidecar {path} carries no samples list"
    want = model_key(model)
    best: Optional[dict] = None
    foreign = 0
    for e in entries:
        if not isinstance(e, dict) or e.get("kind") != RECORD_KIND:
            continue
        if str(e.get("group", "")) != group:
            continue
        if str(e.get("model", "")) != want:
            foreign += 1
            continue
        if accept is not None and not accept(e):
            foreign += 1
            continue
        if best is None or str(e.get("at", "")) >= str(best.get("at", "")):
            best = e
    if best is None:
        return None, (
            f"no {RECORD_KIND} record of group {group} measured on {want} in {path}"
            + (f" ({foreign} of other checkpoints/forms skipped)" if foreign else "")
        )
    return best, "ok"


def append_record(path: str, rec: Mapping[str, object]) -> None:
    """Append one entry (append-only: history is evidence). Same file shape as
    ``host_ledger.append_measured_record``; restated so the tool needs no
    launcher import chain."""
    try:
        with open(path) as f:
            data = json.load(f)
        samples = data.get("samples") if isinstance(data, dict) else None
    except (OSError, ValueError):
        samples = None
    if not isinstance(samples, list):
        samples = []
    samples.append(dict(rec))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump({"samples": samples}, f, indent=1, default=str)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# the resolution the launcher charges
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Resolution:
    group: str
    model: str
    mib: Tuple[int, ...]
    source: str
    provenance: str
    notes: Tuple[str, ...] = field(default=())

    def line(self, cards: Sequence[object]) -> str:
        per_card = ", ".join(
            f"ord{i} nvml{getattr(c, 'nvml_index', '?')}={self.mib[i]}"
            for i, c in enumerate(cards)
        )
        return (
            f"{LINE_TAG} group={self.group} model={self.model} source={self.source}: "
            f"{per_card} MiB -- {self.provenance}"
            + (f" [{'; '.join(self.notes)}]" if self.notes else "")
        )


def resolve(
    group: str,
    cards: Sequence[object],
    model: str,
    *,
    record_path: Optional[str] = None,
    accept: Optional[Callable[[dict], bool]] = None,
    refs: Sequence[OvershootReference] = REFERENCES,
) -> Resolution:
    """RECORD -> BUILTIN -> UNMEASURED-FALLBACK, never another model's number."""
    key = model_key(model)
    notes: List[str] = []
    entry, why = read_record(record_path, group, key, accept)
    if entry is not None:
        by_uuid = entry.get("overshoot_by_uuid") or {}
        missing = [str(getattr(c, "uuid", "")) for c in cards
                   if str(getattr(c, "uuid", "")) not in by_uuid]
        if not missing:
            mib = tuple(max(0, int(by_uuid[str(getattr(c, "uuid"))])) for c in cards)
            return Resolution(
                group, key, mib, SOURCE_RECORD,
                f"measured on {key}: boot {entry.get('boot_tag', '?')} @ "
                f"{entry.get('commit', '?')} at {entry.get('at', '?')}",
            )
        notes.append(
            f"record of boot {entry.get('boot_tag', '?')} names no card {','.join(missing)} "
            f"of this boot -> not used")
    else:
        notes.append(why)
    ref = reference_for(group, key, refs)
    if ref is not None:
        if len(ref.mib) != len(cards):
            notes.append(
                f"built-in {ref.family} reference has {len(ref.mib)} ordinals, this boot "
                f"{len(cards)} -> not used")
        else:
            return Resolution(
                group, key, tuple(int(v) for v in ref.mib), SOURCE_BUILTIN,
                f"{ref.provenance} (built-in, measured on {ref.family})", tuple(notes),
            )
    return Resolution(
        group, key, tuple(0 for _ in cards), SOURCE_FALLBACK,
        f"no measurement of {key}: the awake overshoot is 0 on every card and ONLY the "
        f"priced corridor stands; no other checkpoint's number is charged (27B references: "
        f"{', '.join(sorted({r.family for r in refs if r.group == group}))}). Measure it: "
        f"python -m sglang.srt.weg2.tools.awake_overshoot_record --front <this boot's "
        f"front.log> --append <measured-record sidecar>",
        tuple(notes),
    )


# ---------------------------------------------------------------------------
# log parsing (the tool's half; kept here so tests exercise one parser)
# ---------------------------------------------------------------------------

_RX_KV = re.compile(r"(\w+)=(\S+)")
_RX_BUDGET = re.compile(
    r"budget (?P<label>[PD]) group=(?P<group>[PD]) ordinal=(?P<ord>\d+) "
    r"nvml_idx=(?P<nvml>\d+) (?P<name>.+?): (?P<budget>-?\d+) MiB = .*?"
    r"(?:measured_awake_overshoot (?P<over>\d+) .*?)?MiB\s*$"
)
_RX_FLOOR = re.compile(
    r"CORRIDOR-FLOOR card=(?P<uuid>GPU-[0-9a-fA-F-]+) group=(?P<group>[PD]) "
    r"floor=(?P<floor>\d+) verdict_floor=(?P<vfloor>\d+)"
)
_RX_ORD = re.compile(
    r"argv\[\d+\] -> ordinal=(?P<ord>\d+) nvml(?P<nvml>\d+) name='(?P<name>[^']*)' "
    r"uuid=(?P<uuid>GPU-[0-9a-fA-F-]+)"
)
_RX_FORM = re.compile(r"WEG2-FORM (?P<axes>arch=\S+ .*?) profile=\S* model=(?P<model>\S+)")
_RX_BOOT = re.compile(r"=== WEG2 BOOT tag=(?P<tag>\S+) tree=\S+ @ (?P<commit>[0-9a-f]+)")
_RX_TS = re.compile(r"^\[(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\]")

#: The load windows of each group (H55 phases): P prefills, D decodes and
#: runs its own extends after a flip.
LOAD_PHASES = {"P": ("chunk",), "D": ("round", "chunk")}


@dataclass
class FrontFacts:
    tag: str = ""
    commit: str = ""
    at: str = ""
    model: str = ""
    form: str = ""
    #: ordinal -> (uuid, nvml, name)
    ordinals: Dict[int, Tuple[str, int, str]] = field(default_factory=dict)
    #: group -> ordinal -> charged overshoot (0 when the line charged none)
    charged: Dict[str, Dict[int, int]] = field(default_factory=dict)
    #: group -> uuid -> verdict floor
    floors: Dict[str, Dict[str, int]] = field(default_factory=dict)


def parse_front(lines: Iterable[str]) -> FrontFacts:
    """Tag, commit, model, card ordinals, charges and floors from a front log."""
    ff = FrontFacts()
    for ln in lines:
        if not ff.tag:
            m = _RX_BOOT.search(ln)
            if m:
                ff.tag, ff.commit = m.group("tag"), m.group("commit")
                t = _RX_TS.search(ln)
                ff.at = t.group("ts") if t else ""
                continue
        if not ff.model:
            m = _RX_FORM.search(ln)
            if m:
                ff.model, ff.form = m.group("model"), m.group("axes")
                continue
        if "USER-RESERVE PROVENANCE" in ln:
            for m in _RX_ORD.finditer(ln):
                ff.ordinals[int(m.group("ord"))] = (
                    m.group("uuid"), int(m.group("nvml")), m.group("name"))
            continue
        m = _RX_BUDGET.search(ln.rstrip())
        if m:
            g = m.group("group")
            ff.charged.setdefault(g, {})[int(m.group("ord"))] = int(m.group("over") or 0)
            continue
        m = _RX_FLOOR.search(ln)
        if m:
            ff.floors.setdefault(m.group("group"), {})[m.group("uuid")] = int(m.group("vfloor"))
    return ff


def load_minima(lines: Iterable[str], group: str) -> Dict[int, Tuple[int, int, int, int]]:
    """rank -> (free_min, cache_at_min, retries, windows) over ``group``'s load
    windows (``WEG2-VRAM-PEAK``). Rank r of a group runs on CUDA ordinal r."""
    phases = LOAD_PHASES[group]
    out: Dict[int, List[int]] = {}
    for ln in lines:
        if "WEG2-VRAM-PEAK " not in ln:
            continue
        d = dict(_RX_KV.findall(ln.split("WEG2-VRAM-PEAK ", 1)[1]))
        if d.get("phase") not in phases:
            continue
        try:
            rank = int(d["rank"])
            free = int(d["card_free_mib"])
            cache = int(d["reserved_mib"]) - int(d["allocated_mib"])
        except (KeyError, ValueError):
            continue
        try:
            retries = int(d.get("alloc_retries", "0"))
        except ValueError:
            retries = 0
        cur = out.get(rank)
        if cur is None:
            out[rank] = [free, cache, retries, 1]
            continue
        cur[2] += retries
        cur[3] += 1
        if free < cur[0]:
            cur[0], cur[1] = free, cache
    return {r: (v[0], v[1], v[2], v[3]) for r, v in out.items()}


def derive_group(
    ff: FrontFacts, group: str, minima: Mapping[int, Tuple[int, int, int, int]]
) -> Tuple[List[CardDerivation], List[str]]:
    """Every card of the boot, or a named gap per card that cannot be derived."""
    out: List[CardDerivation] = []
    gaps: List[str] = []
    for ordinal in sorted(ff.ordinals):
        uuid, nvml, name = ff.ordinals[ordinal]
        if ordinal not in minima:
            gaps.append(f"group {group} ord{ordinal}: no {'/'.join(LOAD_PHASES[group])} "
                        f"WEG2-VRAM-PEAK window")
            continue
        floor = ff.floors.get(group, {}).get(uuid)
        if floor is None:
            gaps.append(f"group {group} ord{ordinal}: no CORRIDOR-FLOOR line for {uuid}")
            continue
        charged = ff.charged.get(group, {}).get(ordinal)
        if charged is None:
            gaps.append(f"group {group} ord{ordinal}: no budget line")
            continue
        free, cache, retries, n = minima[ordinal]
        out.append(derive_card(CardMeasurement(
            uuid=uuid, ordinal=ordinal, name=name, charged_mib=charged, floor_mib=floor,
            free_min_mib=free, cache_at_min_mib=cache, retries=retries, windows=n)))
    return out, gaps
