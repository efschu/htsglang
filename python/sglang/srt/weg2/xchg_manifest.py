# SPDX-License-Identifier: Apache-2.0
"""#1330 B4n -- THE PER-RANK PLACEMENT MANIFEST, and the CROSS-GROUP JOIN.

THE DEFECT THIS CLOSES, measured on boot weg2xsn20 and verified in the tree at
``15cf96c7ab``.  All 24 injection legs read ``verdict=NO-COMPARE`` with
``W74 Weg2XchgSourceMissing ... has no source pointer ... src_resolved=0/N
dst_resolved=N/N``: the lane never assembled a single layer on metal.  The
cause is NOT a bug in the resolver.  ``weight_exchange_shadow.derive_leg_plan``
defines ``ptr_of`` (``:3336-3344``) so that it answers ONLY for this rank of
this group, and which group is asked for the SOURCE side is chosen two frames
up from the hook (``:3329-3331``): on ``hook=destination`` and
``hook=authoritative`` the source IS the peer, so ``weight_exchange.py:1811``
sets ``src_ptr=None`` on every descriptor, by construction.  A rank cannot read
another process's ``data_ptr()``, and no amount of repair inside that function
can change it.

A SECOND FINDING OF THE SAME READING, and it is the larger one: **the product
has never built a cross-group plan at all.**  The inventory is
``shard_axis=wx.REPLICATED, shard_total=0`` (``:3237``) and BOTH
``GroupLayout``s are ``tp_size=1`` (``:3332-3334``) -- the PP form on both
sides, i.e. the on-card DIAGONAL.  ``derive_leg_plan``'s own docstring says so
(``:3121``): *"It is not the full cross-group exchange plan.  That one needs
BOTH groups' shard vectors and the unsharded extent of every parameter, which
is cross-group knowledge no single rank holds."*

THIS MODULE IS THAT CROSS-GROUP KNOWLEDGE, and it is WRITTEN DOWN rather than
derived twice.  Each rank records what its OWN LOADER already decided, after
materialisation -- no second placement engine, no header arithmetic, no
duplicated repack rule.  The join then reads the six files and answers the two
questions no single rank can:

* **which P stage holds the source** of a given tensor (P's manifest names the
  holder);
* **how the tensor is cut across D** (D's three rows give the per-rank extents,
  and their SUM is the unsharded extent the plan needs).

The shard AXIS is likewise not guessed: it is READ OFF the join.  If every D
row has the same column count and the row counts sum to P's, the cut is on
ROWS; the mirror case is COLS; identical rows on every side is REPLICATED;
anything else is a shape contradiction and is refused by name.

**THE IDENTITY IS NOT A NEW ONE.**  Every piece is keyed by
``weight_exchange_shadow.manifest_entry(param_name, tensor_class, rows_full,
cols_full, itemsize)`` -- the same tuple the card manifest publishes
(``card_manifest_entries``, ``:2918``) and the same one ``seam_digest`` keys on
(``:474``).  One identity, three readers.

**NO NEW W-CODE** (plan record 2026-09-11: "W23/W39 stay free").  A tensor the
destination holds and no source covers is ``W74 Weg2XchgSourceMissing``, which
is that class's own sentence; a shape the two groups cannot both be right about
is ``W68 Weg2XchgPlanDisagree``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from sglang.srt.weg2 import weight_exchange as wx
from sglang.srt.weg2 import weight_exchange_region as xr

#: The file name, derived exactly as ``lane_coverage.dump_filename`` derives
#: the coverage dump's (#1292).  NOT a second naming scheme: P and D are
#: independently launched process groups sharing ONE dump directory, and
#: ``rank`` is unique only WITHIN a group -- #1292 already paid for that, with
#: P silently overwriting D's dump.
MANIFEST_PREFIX = "phase_manifest"

#: One schema version, written into every file and checked on read.  A manifest
#: from an older boot in the same directory is a WRONG ANSWER, not a missing
#: one, so it is refused rather than merged.
MANIFEST_VERSION = 1

JOIN_LINE_PREFIX = "WEG2-XCHG-MANIFEST"


def manifest_filename(rank: int, group: str = "") -> str:
    """``phase_manifest_{GROUP}_rank{N}.json`` -- #1292's derivation, reused."""
    tag = f"{group}_" if group else ""
    return f"{MANIFEST_PREFIX}_{tag}rank{rank}.json"


@dataclass(frozen=True)
class ManifestPiece:
    """One tensor as THIS rank's loader actually materialised it.

    ``rows_full``/``cols_full`` are this rank's LOCAL storage extents, read
    through ``ParamGeom``/``StorageGeom`` from ``stride()`` rather than
    ``shape`` -- the ``.t()``-view rule of spec section 2.4 -- so a
    column-parallel class is recorded in the space a copy primitive names.
    They are deliberately NOT the unsharded extent: a rank does not know that
    one, and writing down a number it had to infer is exactly the second
    bookkeeping this module exists to remove.  The join adds them up instead.
    """

    param_name: str
    tensor_class: str
    rows_full: int
    cols_full: int
    itemsize: int
    tag: str
    nbytes: int

    @property
    def key(self) -> Tuple[int, int, int, int, int]:
        """The PUBLISHED identity -- ``manifest_entry``, not a private tuple."""
        from sglang.srt.weg2 import weight_exchange_shadow as sh

        return sh.manifest_entry(self.param_name, self.tensor_class,
                                 self.rows_full, self.cols_full, self.itemsize)

    def as_json(self) -> Dict[str, object]:
        return {
            "param_name": self.param_name,
            "tensor_class": self.tensor_class,
            "rows_full": int(self.rows_full),
            "cols_full": int(self.cols_full),
            "itemsize": int(self.itemsize),
            "tag": self.tag,
            "nbytes": int(self.nbytes),
        }

    @classmethod
    def from_json(cls, raw: Dict[str, object]) -> ManifestPiece:
        return cls(
            param_name=str(raw["param_name"]),
            tensor_class=str(raw["tensor_class"]),
            rows_full=int(raw["rows_full"]),
            cols_full=int(raw["cols_full"]),
            itemsize=int(raw["itemsize"]),
            tag=str(raw.get("tag", "")),
            nbytes=int(raw.get("nbytes", 0)),
        )


@dataclass(frozen=True)
class RankManifest:
    """One rank's whole record, plus the provenance that makes it comparable."""

    group: str
    rank: int
    card: int
    region_tag: str
    boot_token: str
    pieces: Tuple[ManifestPiece, ...]

    @property
    def by_name(self) -> Dict[str, ManifestPiece]:
        return {p.param_name: p for p in self.pieces}

    def as_json(self) -> Dict[str, object]:
        return {
            "version": MANIFEST_VERSION,
            "group": self.group,
            "rank": int(self.rank),
            "card": int(self.card),
            "region_tag": self.region_tag,
            "boot_token": self.boot_token,
            "pieces": [p.as_json() for p in self.pieces],
        }

    @classmethod
    def from_json(cls, raw: Dict[str, object], *, path: str = "") -> RankManifest:
        version = int(raw.get("version", -1))
        if version != MANIFEST_VERSION:
            raise wx.Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree: manifest {path or '<memory>'} "
                f"carries schema version {version}, this reader is "
                f"{MANIFEST_VERSION}. A manifest from another boot in the same "
                f"directory is a WRONG answer, not a missing one, so it is "
                f"refused rather than merged into a plan."
            )
        return cls(
            group=str(raw["group"]),
            rank=int(raw["rank"]),
            card=int(raw["card"]),
            region_tag=str(raw.get("region_tag", "")),
            boot_token=str(raw.get("boot_token", "")),
            pieces=tuple(ManifestPiece.from_json(p) for p in raw.get("pieces", ())),
        )


def pieces_from_inventory(inventory: Iterable[object]) -> Tuple[ManifestPiece, ...]:
    """``ParamGeom`` records -> manifest pieces, sorted by name.

    Sorted because two processes build their lists in different orders (the
    C++ side iterates an ``unordered_map``; ``build_plan:1949`` says why), and
    the join compares SETS.  The geometries come from the caller's own
    inventory walk -- this function invents none and reads no tensor.
    """
    out: List[ManifestPiece] = []
    from sglang.srt.weg2 import weight_exchange_shadow as sh

    for geom in inventory:
        name = str(getattr(geom, "name", ""))
        rows = int(getattr(geom, "rows_full", 0))
        cols = int(getattr(geom, "cols_full", 0))
        item = int(getattr(geom, "itemsize", 0))
        out.append(
            ManifestPiece(
                param_name=name,
                tensor_class=sh.tensor_class(name),
                rows_full=rows,
                cols_full=cols,
                itemsize=item,
                tag=str(getattr(geom, "tag", "")),
                nbytes=rows * cols * item,
            )
        )
    return tuple(sorted(out, key=lambda p: p.param_name))


def write_rank_manifest(manifest: RankManifest, dump_dir: str) -> str:
    """Write one rank's manifest atomically; return the path.

    ATOMIC because the reader is another process on the same box and a
    half-written file is indistinguishable from a short one: the join would
    then refuse a tensor that IS placed, which is the false-red direction.
    """
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, manifest_filename(manifest.rank, manifest.group))
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest.as_json(), fh, sort_keys=True)
    os.replace(tmp, path)
    return path


def load_manifests(dump_dir: str, *, boot_token: str = "") -> Tuple[RankManifest, ...]:
    """Every manifest in a directory, newest-boot only.

    ``boot_token`` is a FILTER and not a hope: a stale file from a previous
    boot in the same evidence directory would join silently and the resulting
    plan would name extents no rank holds.  With a token given, a file that
    does not carry it is skipped and COUNTED by the caller's line; without
    one, everything present is read (the hermetic path).
    """
    try:
        names = sorted(f for f in os.listdir(dump_dir)
                       if f.startswith(MANIFEST_PREFIX + "_") and f.endswith(".json"))
    except OSError:
        return ()
    out: List[RankManifest] = []
    for fname in names:
        path = os.path.join(dump_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            continue
        man = RankManifest.from_json(raw, path=path)
        if boot_token and man.boot_token != boot_token:
            continue
        out.append(man)
    return tuple(out)


# ---------------------------------------------------------------------------
# THE JOIN -- the cross-group knowledge, read off the two groups' own records.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JoinedTensor:
    """One tensor, as BOTH groups together describe it.

    ``rows_full``/``cols_full`` are the UNSHARDED extents -- the sum over the
    destination's rows on the sharded axis -- which is precisely the number
    ``derive_leg_plan``'s docstring says no single rank holds.  ``dst_widths``
    is the destination's per-rank extent vector IN PLAN ORDER, which
    ``_blocks_of`` (``weight_exchange.py:1575``) consumes directly, so no
    ``ratios`` vector has to be invented for a group whose tensors are not all
    cut the same way.
    """

    param_name: str
    tensor_class: str
    tag: str
    itemsize: int
    rows_full: int
    cols_full: int
    shard_axis: int
    stage: int
    dst_widths: Tuple[int, ...]
    src_card: int

    @property
    def sharded(self) -> bool:
        return self.shard_axis != wx.REPLICATED

    def geom(self) -> wx.ParamGeom:
        """The ``ParamGeom`` the plan consumes.  No tensor is read."""
        geom = wx.ParamGeom(
            name=self.param_name,
            tag=self.tag,
            shard_axis=self.shard_axis,
            rows_full=int(self.rows_full),
            cols_full=int(self.cols_full),
            itemsize=int(self.itemsize),
            stage=int(self.stage),
            dst_widths=(tuple(int(w) for w in self.dst_widths)
                        if self.sharded else None),
        )
        geom.validate()
        return geom


@dataclass(frozen=True)
class ManifestJoin:
    """The joined placement of one direction, plus its own denominators."""

    src_group: str
    dst_group: str
    cards: Tuple[int, ...]
    tensors: Tuple[JoinedTensor, ...]
    #: Names present on the destination and on no source -- ALWAYS empty on a
    #: successful join (they are W74), kept so a caller can print the census.
    unsourced: Tuple[str, ...]
    src_ranks: int
    dst_ranks: int

    @property
    def by_name(self) -> Dict[str, JoinedTensor]:
        return {t.param_name: t for t in self.tensors}

    @property
    def n_sharded(self) -> int:
        return sum(1 for t in self.tensors if t.sharded)

    def line(self) -> str:
        """Every number with its denominator (the campaign's denominator law)."""
        classes = sorted({t.tensor_class for t in self.tensors})
        return (
            f"{JOIN_LINE_PREFIX} src={self.src_group} dst={self.dst_group} "
            f"src_ranks={self.src_ranks} dst_ranks={self.dst_ranks} "
            f"tensors={len(self.tensors)} "
            f"sharded={self.n_sharded}/{len(self.tensors)} "
            f"unsourced={len(self.unsourced)} "
            f"classes={len(classes)} "
            f"-- extents are the JOIN's (the destination's rows summed), not "
            f"any one rank's reading"
        )


def _axis_of(name: str, src: ManifestPiece,
             dst_rows: Sequence[ManifestPiece]) -> Tuple[int, int, int, Tuple[int, ...]]:
    """``(shard_axis, rows_full, cols_full, dst_widths)`` -- READ, not guessed.

    Four cases and no fifth.  The fifth would be a silent ``REPLICATED``, which
    is exactly the tree's current answer (``weight_exchange_shadow.py:3237``)
    and the reason the shard cut has been invisible since S2: a plan that calls
    every tensor replicated moves whole tensors between differently-shaped
    groups and calls it agreement.
    """
    rows = [int(p.rows_full) for p in dst_rows]
    cols = [int(p.cols_full) for p in dst_rows]
    s_rows, s_cols = int(src.rows_full), int(src.cols_full)

    same_cols = len(set(cols)) == 1 and cols[0] == s_cols
    same_rows = len(set(rows)) == 1 and rows[0] == s_rows

    if same_rows and same_cols:
        return wx.REPLICATED, s_rows, s_cols, tuple(rows)
    if same_cols and sum(rows) == s_rows:
        return wx.ROWS, s_rows, s_cols, tuple(rows)
    if same_rows and sum(cols) == s_cols:
        return wx.COLS, s_rows, s_cols, tuple(cols)
    raise wx.Weg2XchgPlanDisagree(
        f"W68 Weg2XchgPlanDisagree: {name}: the source holds "
        f"({s_rows}, {s_cols}) and the destination rows hold "
        f"{list(zip(rows, cols))}. That is neither a row cut (equal columns, "
        f"rows summing to the source's), nor a column cut, nor a replica -- "
        f"so the two groups cannot both be describing the same tensor. "
        f"Guessing REPLICATED here is what made the shard cut invisible; the "
        f"join refuses instead."
    )


def join_manifests(
    manifests: Sequence[RankManifest],
    *,
    src_group: str,
    dst_group: str,
) -> ManifestJoin:
    """Join two groups' manifests over ``param_name``.  Never a default.

    THE SOURCE SIDE IS THE JOIN'S ANSWER, which is the whole point: the
    destination rank asks "which rank of the other group holds the bytes I
    need", and the answer comes from a FILE that rank WROTE, not from a
    pointer the asking process structurally cannot read.
    """
    src = sorted((m for m in manifests if m.group == src_group),
                 key=lambda m: m.rank)
    dst = sorted((m for m in manifests if m.group == dst_group),
                 key=lambda m: m.rank)
    if not src or not dst:
        raise wx.Weg2XchgSourceMissing(
            f"W74 Weg2XchgSourceMissing: the join needs both groups' "
            f"manifests and has {len(src)} for {src_group!r} and {len(dst)} "
            f"for {dst_group!r}. An absent manifest is an absent SOURCE, and "
            f"planning over the ranks that did publish would silently narrow "
            f"the exchange to whatever happened to be on disk."
        )
    if [m.rank for m in dst] != list(range(len(dst))):
        raise wx.Weg2XchgPlanDisagree(
            f"W68 Weg2XchgPlanDisagree: group {dst_group!r} published ranks "
            f"{[m.rank for m in dst]}, which is not a contiguous 0..n-1 "
            f"range. The destination's per-rank extents are a VECTOR in rank "
            f"order; a gap in it would shift every shard boundary."
        )

    src_by_name: Dict[str, Tuple[int, ManifestPiece]] = {}
    src_card_by_rank = {m.rank: m.card for m in src}
    for man in src:
        for piece in man.pieces:
            prior = src_by_name.get(piece.param_name)
            if prior is not None and prior[1].key != piece.key:
                raise wx.Weg2XchgPlanDisagree(
                    f"W68 Weg2XchgPlanDisagree: {piece.param_name} is "
                    f"published by {src_group} ranks {prior[0]} and "
                    f"{man.rank} with DIFFERENT identities {prior[1].key} vs "
                    f"{piece.key}. Under the PP form one stage holds a tensor "
                    f"whole; two disagreeing holders leave the plan free to "
                    f"pick either."
                )
            if prior is None:
                src_by_name[piece.param_name] = (man.rank, piece)

    dst_names: List[str] = []
    seen = set()
    for man in dst:
        for piece in man.pieces:
            if piece.param_name not in seen:
                seen.add(piece.param_name)
                dst_names.append(piece.param_name)
    dst_names.sort()

    tensors: List[JoinedTensor] = []
    unsourced: List[str] = []
    for name in dst_names:
        rows = [m.by_name.get(name) for m in dst]
        if any(p is None for p in rows):
            # A tensor only SOME destination ranks hold is not a shard cut this
            # plan can name; it is reported as unsourced rather than planned
            # over the ranks that have it.
            unsourced.append(name)
            continue
        found = src_by_name.get(name)
        if found is None:
            unsourced.append(name)
            continue
        stage, src_piece = found
        if int(src_piece.itemsize) != int(rows[0].itemsize):
            raise wx.Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree: {name}: itemsize "
                f"{src_piece.itemsize} on {src_group} rank {stage} against "
                f"{rows[0].itemsize} on {dst_group}. The two groups loaded "
                f"this tensor at different element widths, so no descriptor "
                f"can name both."
            )
        axis, rows_full, cols_full, widths = _axis_of(name, src_piece, rows)
        tensors.append(
            JoinedTensor(
                param_name=name,
                tensor_class=src_piece.tensor_class,
                tag=str(rows[0].tag or src_piece.tag),
                itemsize=int(src_piece.itemsize),
                rows_full=rows_full,
                cols_full=cols_full,
                shard_axis=axis,
                stage=int(stage),
                dst_widths=widths,
                src_card=int(src_card_by_rank.get(stage, stage)),
            )
        )

    if unsourced:
        raise wx.Weg2XchgSourceMissing(
            f"W74 Weg2XchgSourceMissing: {len(unsourced)} of "
            f"{len(dst_names)} tensors held by {dst_group} have no counterpart "
            f"in {src_group}'s manifest (first: {unsourced[0]!r}). Assembly "
            f"stages a source, it does not create one -- planning the rest "
            f"would serve those slices with undefined bytes."
        )

    return ManifestJoin(
        src_group=str(src_group),
        dst_group=str(dst_group),
        cards=tuple(m.card for m in dst),
        tensors=tuple(tensors),
        unsourced=(),
        src_ranks=len(src),
        dst_ranks=len(dst),
    )


# ---------------------------------------------------------------------------
# THE PROVIDER -- the join turned into descriptors with BOTH sides resolved.
# ---------------------------------------------------------------------------


def plan_from_join(
    join: ManifestJoin,
    *,
    waves: Optional[Sequence[Sequence[str]]] = None,
    src_addr=None,
    dst_addr=None,
    dst_rank: Optional[int] = None,
) -> wx.XchgPlan:
    """Build the CROSS-GROUP plan from the join.

    THE DESTINATION IS A REAL TP GROUP, and that is the half a ``src_ptr``
    patch would have missed.  ``tp_size=len(cards)`` with the join's own
    per-tensor ``dst_widths`` makes ``_blocks_of`` cut the shards
    (``weight_exchange.py:1575``); with ``tp_size=1`` the same descriptors come
    out whole and ``src_resolved=N/N`` would be green on a plan that moves
    nothing correctly.  :func:`refuse_diagonal_layout` is the guard, and it
    runs here rather than being left to the caller.

    ``src_addr(name, rank)`` and ``dst_addr(name, rank)`` are the ADDRESS
    BOOKS and they stay the caller's: which arena a byte lives in is the
    loader's and the transport's decision, exactly as the operator ruling
    says.  What this function supplies is the IDENTITY -- which source covers
    which destination slice -- which is what the peer's manifest answers and a
    peer's ``data_ptr()`` cannot.
    """
    cards = tuple(join.cards)
    refuse_diagonal_layout(len(cards), dst_group=join.dst_group)
    src = wx.GroupLayout(name=join.src_group, cards=cards, tp_size=1, base=0)
    dst = wx.GroupLayout(name=join.dst_group, cards=cards,
                         tp_size=len(cards), base=len(cards))

    inventory = [t.geom() for t in join.tensors]
    if waves is None:
        waves = [sorted({str(g.tag) for g in inventory})]

    def ptr_of(group: str, rank: int, name: str) -> Optional[int]:
        if group == join.src_group:
            return None if src_addr is None else src_addr(name, int(rank))
        if group == join.dst_group:
            if dst_rank is not None and int(rank) != int(dst_rank):
                return None
            return None if dst_addr is None else dst_addr(name, int(rank))
        return None

    return wx.build_plan(inventory, src, dst, waves=waves, ptr_of=ptr_of)


def refuse_diagonal_layout(tp_size: int, *, dst_group: str) -> None:
    """A destination planned at ``tp_size=1`` is the diagonal, and is refused.

    Operator ruling 2026-09-12: *"src_resolved=N/N auf einem Diagonal-Plan
    zaehlt nicht"*.  This is the pin.  The tree's product path builds exactly
    that layout (``weight_exchange_shadow.py:3332-3334``), so without this
    refusal the slice's own acceptance number could be satisfied by the shape
    the slice exists to replace.
    """
    if int(tp_size) <= 1:
        raise wx.Weg2XchgPlanDisagree(
            f"W68 Weg2XchgPlanDisagree: group {dst_group!r} would be planned "
            f"at tp_size={tp_size}, which is the PP form on BOTH sides -- the "
            f"on-card diagonal, not the cross-group exchange. A plan built "
            f"that way resolves both pointer sides and still cuts no shard, "
            f"so its src_resolved=N/N would grade a plan that moves whole "
            f"tensors between two differently shaped groups."
        )


def refuse_on_materialisation_drift(piece: ManifestPiece, live) -> None:
    """The manifest against the tensor that came out of the loader.

    Sharp from commit 1 (operator ruling): a manifest that has drifted from the
    hardware is worse than no manifest, because every downstream reader trusts
    it.  Reads the storage geometry through ``StorageGeom`` -- ``stride()``,
    never ``shape`` -- so a ``.t()`` view is compared in the space the plan
    uses.  RAISES; it does not return a bool, because a bool is the swallowed
    refusal this campaign has paid for.
    """
    got = wx.StorageGeom.of(live)
    if (int(got.rows) != int(piece.rows_full)
            or int(got.cols) != int(piece.cols_full)
            or int(got.itemsize) != int(piece.itemsize)):
        raise wx.Weg2XchgPlanDisagree(
            f"W68 Weg2XchgPlanDisagree: {piece.param_name}: the manifest "
            f"records ({piece.rows_full}, {piece.cols_full}) itemsize "
            f"{piece.itemsize} and the materialised tensor holds "
            f"({got.rows}, {got.cols}) itemsize {got.itemsize}. The manifest "
            f"is what every other reader plans from, so a drifted one is a "
            f"wrong answer rather than a missing one."
        )


def n_cards_default() -> int:
    """The card count, from the module that owns it -- never a literal here."""
    return int(xr.N_CARDS)


# ---------------------------------------------------------------------------
# THE WRITE SITE -- after materialisation, on the path that is proven to run.
# ---------------------------------------------------------------------------
#
# NO SECOND DIRECTORY AUTHORITY AND NO SECOND TOKEN.  The manifests land in the
# dump directory the phase footprint and the #1292/#1348 coverage dumps already
# share (``SGLANG_PHASE_FOOTPRINT_DUMP``), and the boot token is the region's
# OWN nonce (``weight_exchange_region.ENV_REGION_BOOT``), which the launcher
# already publishes to both groups.  Inventing a third env pair for the same
# two facts is the second-bookkeeping defect this whole slice is an instance
# of removing.

#: MEASURED, not chosen by taste.  This was ``SGLANG_PHASE_FOOTPRINT_DUMP``
#: first, so that the manifests would share the #1292/#1348 dump directory and
#: no new name would exist -- and that was WRONG in the one way that matters:
#: ``weg2/launcher.py``'s ``coverage_dump_dir`` only puts that variable into a
#: rank's environment under ``--xchg-coverage-diff`` and returns ``""``
#: otherwise, so a writer keyed on it finds nothing on every boot that does not
#: also arm the coverage tracer.  It would have written no file and logged no
#: line: built, green at the desk, never executed.  One new name for a fact
#: that had none.
#:
#: THE TOKEN IS STILL NOT NEW: the boot nonce is the region's own
#: (``weight_exchange_region.ENV_REGION_BOOT``), already published to both
#: groups, so a stale manifest from a previous boot in the same evidence
#: directory is filtered rather than merged.
DIR_ENV = "SGLANG_WEG2_XCHG_MANIFEST_DIR"

#: Read as a FALLBACK only, for a process that has the coverage dump directory
#: and not this one (the hermetic tools).  Never the primary: see above.
FALLBACK_DIR_ENV = "SGLANG_PHASE_FOOTPRINT_DUMP"


def manifest_dir(default: str = "") -> str:
    """Where this boot's manifests go, or ``default``.  Two env reads."""
    return str(os.environ.get(DIR_ENV, "")
               or os.environ.get(FALLBACK_DIR_ENV, "")
               or default)


def boot_token() -> str:
    """This boot's region nonce -- already published to both groups."""
    return str(os.environ.get(xr.ENV_REGION_BOOT, "") or "")


def write_this_rank(
    inventory: Iterable[object],
    *,
    group: str,
    rank: int,
    card: int,
    region_tag: str,
    dump_dir: str = "",
    token: str = "",
) -> Optional[str]:
    """Record what THIS rank's loader decided.  ``None`` when unarmed.

    ``inventory`` is the caller's OWN ``ParamGeom`` walk -- the one it already
    built to plan and to grade coverage with.  Nothing is re-walked and no
    tensor is re-read here: the whole point of the write-along is that the
    loader's decision is written down rather than reconstructed, so a second
    walk would reintroduce the drift in the same commit that removes it.
    """
    directory = dump_dir or manifest_dir()
    if not directory or not group or int(rank) < 0:
        return None
    manifest = RankManifest(
        group=str(group),
        rank=int(rank),
        card=int(card),
        region_tag=str(region_tag),
        boot_token=str(token or boot_token()),
        pieces=pieces_from_inventory(inventory),
    )
    return write_rank_manifest(manifest, directory)


def written_line(path: str, manifest: RankManifest) -> str:
    """The acceptance line, every number with its denominator."""
    return (
        f"{JOIN_LINE_PREFIX}-WRITE group={manifest.group} rank={manifest.rank} "
        f"card={manifest.card} region_tag={manifest.region_tag} "
        f"pieces={len(manifest.pieces)} "
        f"bytes={sum(p.nbytes for p in manifest.pieces)} "
        f"boot_token={manifest.boot_token or 'unset'} path={path} "
        f"-- this rank's loader decision, written down, not reconstructed"
    )
