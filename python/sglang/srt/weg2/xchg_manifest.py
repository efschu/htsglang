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


def manifest_filename(rank: int, group: str = "", region_tag: str = "") -> str:
    """``phase_manifest_{GROUP}_rank{N}_{REGION_TAG}.json``.

    #1292's derivation PLUS the region tag, and the third component is not
    decoration.  ``arm_coverage_at_load`` is called from
    ``ModelRunner.load_model`` (``model_runner.py:2564``) with that runner's own
    ``weights_tag`` (``:2461``), and a weg2 rank runs TWO runners in one
    process: the main model and the drafter.  Boot weg2xsn20's D log proves it
    -- three ``WEG2-XCHG-RESIDENT tag=weights_draft ... rank={0,1,2}`` lines
    beside the ``weights_0`` ones.  Keyed on (group, rank) alone, the second
    write would have CLOBBERED the first and the join would have planned over
    whichever runner finished last, from a file that looked complete.

    AMENDMENT 6 is why that is a loss of real bytes and not a cosmetic one: the
    draft tag is IN the weights family and is "planned, censused, waved, covered
    and exchanged like every other layer tag".
    """
    grp = f"{group}_" if group else ""
    reg = f"_{region_tag}" if region_tag else ""
    return f"{MANIFEST_PREFIX}_{grp}rank{rank}{reg}.json"


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
    path = os.path.join(dump_dir, manifest_filename(
        manifest.rank, manifest.group, manifest.region_tag))
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

    GROUP-ORIENTED, NOT ROLE-ORIENTED, and that is the whole reason this
    dataclass names ``pp_stage``/``tp_widths`` instead of ``src``/``dst``: a
    flip goes BOTH ways, and the two directions must come out of ONE join or
    they are two derivations of one fact again.  Which side is the source is
    the DIRECTION's business (:func:`plan_from_join`), never the join's.

    ``rows_full``/``cols_full`` are the UNSHARDED extents -- the TP side's rows
    summed on the sharded axis -- which is precisely the number
    ``derive_leg_plan``'s docstring says no single rank holds (``:3121``).
    """

    param_name: str
    tensor_class: str
    tag: str
    itemsize: int
    rows_full: int
    cols_full: int
    shard_axis: int
    #: Which rank of the PP group holds this tensor WHOLE.
    pp_stage: int
    #: The TP group's per-rank extent on the sharded axis, in rank order.
    tp_widths: Tuple[int, ...]
    pp_card: int

    @property
    def sharded(self) -> bool:
        return self.shard_axis != wx.REPLICATED

    def geom(self, *, tp_is_dst: bool) -> wx.ParamGeom:
        """The ``ParamGeom`` the plan consumes.  No tensor is read.

        ``family`` IS THE PARAMETER'S OWN NAME, and that is the hinge that
        makes the mirror direction work at all.  ``_blocks_of`` honours a
        per-tensor width vector ONLY on the destination
        (``weight_exchange.py:1575``: ``if is_dst and geom.dst_widths is not
        None``); the SOURCE side with ``tp_size > 1`` goes through
        ``layout.ratios_for(geom.family)`` (``:1594``), which is a GROUP-level
        lookup.  Under ``tp_to_pp`` the TP group is the source, so a plan that
        only filled ``dst_widths`` would have fallen back to an even split on
        the very side that is unevenly cut -- silently, on both ends equally,
        which is the #1275 class.  Keying ``family_ratios`` by the parameter
        name turns that group-level lookup into a per-tensor one without a new
        field in ``ParamGeom`` and without touching ``_blocks_of``.

        THE VOCABULARY LAW IS NOT BYPASSED, IT IS SUPERSEDED BY MEASUREMENT.
        ``ratios_for`` (``:510``) exists so the vocabulary keeps the even split
        under an uneven base plan.  Here the widths are not a plan at all --
        they are what the two groups' loaders ACTUALLY DID, read back off their
        own manifests.  Whatever ``VocabParallelEmbedding`` chose for
        ``embed_tokens`` is the vector this returns for ``embed_tokens``, so
        the law is satisfied by construction rather than by a family taxonomy
        that has to be kept in step with the loader.
        """
        geom = wx.ParamGeom(
            name=self.param_name,
            tag=self.tag,
            shard_axis=self.shard_axis,
            rows_full=int(self.rows_full),
            cols_full=int(self.cols_full),
            itemsize=int(self.itemsize),
            stage=int(self.pp_stage),
            family=(None if not self.sharded else str(self.param_name)),
            dst_widths=(tuple(int(w) for w in self.tp_widths)
                        if (self.sharded and tp_is_dst) else None),
        )
        geom.validate()
        return geom


@dataclass(frozen=True)
class ManifestJoin:
    """The joined placement of ONE BOOT -- both directions, one derivation."""

    pp_group: str
    tp_group: str
    cards: Tuple[int, ...]
    tensors: Tuple[JoinedTensor, ...]
    #: ALWAYS empty on a successful join (they are W74); kept so a caller can
    #: print the census rather than infer it from an exception.
    unsourced: Tuple[str, ...]
    pp_ranks: int
    tp_ranks: int

    @property
    def by_name(self) -> Dict[str, JoinedTensor]:
        return {t.param_name: t for t in self.tensors}

    @property
    def n_sharded(self) -> int:
        return sum(1 for t in self.tensors if t.sharded)

    def family_ratios(self) -> Dict[str, Tuple[int, ...]]:
        """The TP group's per-tensor width vector, keyed by parameter name."""
        return {t.param_name: tuple(int(w) for w in t.tp_widths)
                for t in self.tensors if t.sharded}

    def line(self, direction: str = "") -> str:
        """Every number with its denominator (the denominator law)."""
        classes = sorted({t.tensor_class for t in self.tensors})
        return (
            f"{JOIN_LINE_PREFIX} pp={self.pp_group} tp={self.tp_group} "
            f"{('direction=' + direction + ' ') if direction else ''}"
            f"pp_ranks={self.pp_ranks} tp_ranks={self.tp_ranks} "
            f"tensors={len(self.tensors)} "
            f"sharded={self.n_sharded}/{len(self.tensors)} "
            f"unsourced={len(self.unsourced)} "
            f"classes={len(classes)} "
            f"-- extents are the JOIN's (the TP side's rows summed), not any "
            f"one rank's reading; the direction chooses roles, not extents"
        )


def _axis_of(name: str, whole: ManifestPiece,
             cut: Sequence[ManifestPiece],
             ) -> Tuple[int, int, int, Tuple[int, ...]]:
    """``(shard_axis, rows_full, cols_full, tp_widths)`` -- READ, not guessed.

    Four cases and no fifth.  The fifth would be a silent ``REPLICATED``, which
    is exactly the tree's current answer (``weight_exchange_shadow.py:3237``)
    and the reason the shard cut has been invisible since S2: a plan that calls
    every tensor replicated moves whole tensors between differently-shaped
    groups and calls that agreement.

    ``whole`` is the PP side (one stage holds the tensor entire) and ``cut`` the
    TP side's rows.  Both are read off manifests, so this is a comparison of
    two measurements and not an inference from either.
    """
    rows = [int(p.rows_full) for p in cut]
    cols = [int(p.cols_full) for p in cut]
    w_rows, w_cols = int(whole.rows_full), int(whole.cols_full)

    same_cols = len(set(cols)) == 1 and cols[0] == w_cols
    same_rows = len(set(rows)) == 1 and rows[0] == w_rows

    if same_rows and same_cols:
        return wx.REPLICATED, w_rows, w_cols, tuple(rows)
    if same_cols and sum(rows) == w_rows:
        return wx.ROWS, w_rows, w_cols, tuple(rows)
    if same_rows and sum(cols) == w_cols:
        return wx.COLS, w_rows, w_cols, tuple(cols)
    raise wx.Weg2XchgPlanDisagree(
        f"W68 Weg2XchgPlanDisagree: {name}: the PP side holds "
        f"({w_rows}, {w_cols}) and the TP rows hold {list(zip(rows, cols))}. "
        f"That is neither a row cut (equal columns, rows summing to the whole), "
        f"nor a column cut, nor a replica -- so the two groups cannot both be "
        f"describing the same tensor. Guessing REPLICATED here is what made the "
        f"shard cut invisible; the join refuses instead."
    )


def merge_region_tags(manifests: Iterable[RankManifest]) -> List[RankManifest]:
    """One record per RANK, unioning that rank's region-tag files.

    A rank publishes one manifest per RUNNER (main model, drafter), because
    ``arm_coverage_at_load`` fires once per runner with that runner's own region
    tag.  The join's unit is the RANK -- what bytes rank *n* holds -- so the
    files of one rank are unioned here rather than being three separate rows
    that would break the contiguous-rank check and, worse, let one runner's
    view stand for the rank's.

    A parameter NAME claimed by two region tags of one rank with DIFFERENT
    identities is refused: that is two runners disagreeing about one tensor, and
    picking either would be the rank-local derivation this module removes.
    ``region_tag`` on the merged record becomes the sorted join of the tags it
    covers, so the provenance stays readable.
    """
    by_rank: Dict[int, List[RankManifest]] = {}
    for man in manifests:
        by_rank.setdefault(int(man.rank), []).append(man)
    out: List[RankManifest] = []
    for rank in sorted(by_rank):
        rows = sorted(by_rank[rank], key=lambda m: m.region_tag)
        pieces: Dict[str, ManifestPiece] = {}
        for man in rows:
            for piece in man.pieces:
                prior = pieces.get(piece.param_name)
                if prior is not None and prior.key != piece.key:
                    raise wx.Weg2XchgPlanDisagree(
                        f"W68 Weg2XchgPlanDisagree: {piece.param_name} is "
                        f"published twice by {man.group} rank {rank} with "
                        f"DIFFERENT identities {prior.key} vs {piece.key}. Two "
                        f"runners of one rank disagree about one tensor, and "
                        f"picking either would be exactly the rank-local "
                        f"derivation the manifest replaces."
                    )
                pieces.setdefault(piece.param_name, piece)
        head = rows[0]
        out.append(RankManifest(
            group=head.group, rank=rank, card=head.card,
            region_tag="+".join(sorted({m.region_tag for m in rows if m.region_tag})),
            boot_token=head.boot_token,
            pieces=tuple(sorted(pieces.values(), key=lambda p: p.param_name)),
        ))
    return out


def join_manifests(
    manifests: Sequence[RankManifest],
    *,
    pp_group: str = "P",
    tp_group: str = "D",
) -> ManifestJoin:
    """Join the two groups' manifests over ``param_name``.  Never a default.

    ONE JOIN SERVES BOTH DIRECTIONS.  It is deliberately not parameterised by
    source and destination: a flip runs ``pp_to_tp`` and ``tp_to_pp`` and both
    must come out of the same derivation, or the campaign has two answers to
    one placement again.

    THE CROSS-GROUP ANSWER IS THE POINT: a rank asks "which rank of the other
    group holds the bytes I need, and how is this tensor cut over there", and
    the answer comes from FILES THOSE RANKS WROTE -- not from a pointer the
    asking process structurally cannot read (``weight_exchange_shadow.py:3336``).
    """
    pp = merge_region_tags(m for m in manifests if m.group == pp_group)
    tp = merge_region_tags(m for m in manifests if m.group == tp_group)
    if not pp or not tp:
        raise wx.Weg2XchgSourceMissing(
            f"W74 Weg2XchgSourceMissing: the join needs both groups' "
            f"manifests and has {len(pp)} for {pp_group!r} and {len(tp)} for "
            f"{tp_group!r}. An absent manifest is an absent SOURCE in one of "
            f"the two directions, and planning over the ranks that did publish "
            f"would silently narrow the exchange to whatever was on disk."
        )
    for label, group, mans in ((pp_group, pp_group, pp), (tp_group, tp_group, tp)):
        if [m.rank for m in mans] != list(range(len(mans))):
            raise wx.Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree: group {group!r} published ranks "
                f"{[m.rank for m in mans]}, which is not a contiguous 0..n-1 "
                f"range. The per-rank extents are a VECTOR in rank order; a "
                f"gap in it would shift every shard boundary."
            )

    pp_by_name: Dict[str, Tuple[int, ManifestPiece]] = {}
    pp_card_by_rank = {m.rank: m.card for m in pp}
    for man in pp:
        for piece in man.pieces:
            prior = pp_by_name.get(piece.param_name)
            if prior is not None and prior[1].key != piece.key:
                raise wx.Weg2XchgPlanDisagree(
                    f"W68 Weg2XchgPlanDisagree: {piece.param_name} is "
                    f"published by {pp_group} ranks {prior[0]} and {man.rank} "
                    f"with DIFFERENT identities {prior[1].key} vs {piece.key}. "
                    f"Under the PP form one stage holds a tensor whole; two "
                    f"disagreeing holders leave the plan free to pick either."
                )
            if prior is None:
                pp_by_name[piece.param_name] = (man.rank, piece)

    names: List[str] = []
    seen = set()
    for man in tp:
        for piece in man.pieces:
            if piece.param_name not in seen:
                seen.add(piece.param_name)
                names.append(piece.param_name)
    names.sort()

    tensors: List[JoinedTensor] = []
    unsourced: List[str] = []
    for name in names:
        rows = [m.by_name.get(name) for m in tp]
        if any(p is None for p in rows):
            # A tensor only SOME TP ranks hold is not a cut this plan can name;
            # it is reported rather than planned over the ranks that have it.
            unsourced.append(name)
            continue
        found = pp_by_name.get(name)
        if found is None:
            unsourced.append(name)
            continue
        stage, whole = found
        if int(whole.itemsize) != int(rows[0].itemsize):
            raise wx.Weg2XchgPlanDisagree(
                f"W68 Weg2XchgPlanDisagree: {name}: itemsize "
                f"{whole.itemsize} on {pp_group} rank {stage} against "
                f"{rows[0].itemsize} on {tp_group}. The two groups loaded this "
                f"tensor at different element widths, so no descriptor can "
                f"name both."
            )
        axis, rows_full, cols_full, widths = _axis_of(name, whole, rows)
        tensors.append(
            JoinedTensor(
                param_name=name,
                tensor_class=whole.tensor_class,
                tag=str(rows[0].tag or whole.tag),
                itemsize=int(whole.itemsize),
                rows_full=rows_full,
                cols_full=cols_full,
                shard_axis=axis,
                pp_stage=int(stage),
                tp_widths=widths,
                pp_card=int(pp_card_by_rank.get(stage, stage)),
            )
        )

    if unsourced:
        raise wx.Weg2XchgSourceMissing(
            f"W74 Weg2XchgSourceMissing: {len(unsourced)} of {len(names)} "
            f"tensors held by {tp_group} have no counterpart in {pp_group}'s "
            f"manifest (first: {unsourced[0]!r}). Assembly stages a source, it "
            f"does not create one -- planning the rest would serve those slices "
            f"with undefined bytes."
        )

    return ManifestJoin(
        pp_group=str(pp_group), tp_group=str(tp_group),
        cards=tuple(m.card for m in tp), tensors=tuple(tensors),
        unsourced=(), pp_ranks=len(pp), tp_ranks=len(tp),
    )


# ---------------------------------------------------------------------------
# THE PROVIDER -- the join turned into descriptors, in EITHER direction.
# ---------------------------------------------------------------------------


def plan_from_join(
    join: ManifestJoin,
    *,
    direction: str = "",
    waves: Optional[Sequence[Sequence[str]]] = None,
    src_addr=None,
    dst_addr=None,
) -> wx.XchgPlan:
    """Build the CROSS-GROUP plan for one direction out of the ONE join.

    BOTH DIRECTIONS ARE EQUAL CITIZENS.  ``pp_to_tp`` makes the PP group the
    source and the TP group the destination; ``tp_to_pp`` mirrors it.  Neither
    is a special case: the extents, the axis and the holder come from the same
    join, and only the ROLES change.

    THE TP SIDE IS A REAL TP GROUP IN BOTH OF THEM, which is the half a
    ``src_ptr`` patch would have missed.  ``tp_size=len(cards)`` with the
    join's own per-tensor vector makes ``_blocks_of`` cut the shards; with
    ``tp_size=1`` the same descriptors come out whole and ``src_resolved=N/N``
    would be green on a plan that moves nothing correctly.
    :func:`refuse_diagonal_layout` is the guard and it runs here.

    **THE ADDRESS CONTRACT, stated so the next wiring does not re-decide it.**
    ``src_addr(name, rank)`` and ``dst_addr(name, rank)`` are the caller's, and
    what this module supplies is the IDENTITY -- which piece of which holder
    covers which destination slice.  The intended resolution is:

    * the SOURCE hook, on the rank that holds the bytes, answers with its OWN
      DEVICE ADDRESS (it knows it; it is in its own manifest) and deposits its
      pieces into the bounce slot;
    * the DESTINATION side resolves the source to the BOUNCE SLOT OFFSET (host
      staging) -- never to a peer device address, which no process can read,
      and never to the ring.

    THE RING IS THE COUNTER-PROOF, NEVER THE SOURCE, at this stage.  Reading
    the source side out of the ring would be the ring restore through another
    door, and the goal it defeats -- zero layer bytes resident in host RAM --
    is the whole point of the exchange.  ``pointer_profile`` counts both
    resolutions the same way (it asks only whether ``src_ptr`` is set), so
    ``src_resolved`` is honest under either hook.
    """
    direction = str(direction or wx.LEGS_PP_TO_TP)
    if direction not in (wx.LEGS_PP_TO_TP, wx.LEGS_TP_TO_PP):
        raise wx.Weg2XchgPlanDisagree(
            f"W68 Weg2XchgPlanDisagree: {direction!r} is not a flip direction. "
            f"The two are {wx.LEGS_PP_TO_TP!r} and {wx.LEGS_TP_TO_PP!r} "
            f"(layers/dcp/phase_flip_plan.py), and a third spelling would "
            f"plan one of them under the other's name."
        )
    cards = tuple(join.cards)
    refuse_diagonal_layout(len(cards), tp_group=join.tp_group)
    tp_is_dst = direction == wx.LEGS_PP_TO_TP

    ratios = join.family_ratios()
    pp = wx.GroupLayout(name=join.pp_group, cards=cards, tp_size=1,
                        base=(0 if tp_is_dst else len(cards)))
    tp = wx.GroupLayout(name=join.tp_group, cards=cards, tp_size=len(cards),
                        family_ratios=ratios,
                        base=(len(cards) if tp_is_dst else 0))
    src, dst = (pp, tp) if tp_is_dst else (tp, pp)

    inventory = [t.geom(tp_is_dst=tp_is_dst) for t in join.tensors]
    if waves is None:
        waves = [sorted({str(g.tag) for g in inventory})]

    def ptr_of(group: str, rank: int, name: str) -> Optional[int]:
        if group == src.name:
            return None if src_addr is None else src_addr(name, int(rank))
        if group == dst.name:
            return None if dst_addr is None else dst_addr(name, int(rank))
        return None

    return wx.build_plan(inventory, src, dst, waves=waves, ptr_of=ptr_of)


def refuse_diagonal_layout(tp_size: int, *, tp_group: str) -> None:
    """A TP group planned at ``tp_size=1`` is the diagonal, and is refused.

    Operator ruling 2026-09-12: ``src_resolved=N/N`` on a diagonal plan does
    not count.  The tree's product path builds exactly that layout
    (``weight_exchange_shadow.py:3332-3334``, BOTH groups ``tp_size=1``), so
    without this refusal the slice's own acceptance number could be satisfied
    by the very shape the slice exists to replace.
    """
    if int(tp_size) <= 1:
        raise wx.Weg2XchgPlanDisagree(
            f"W68 Weg2XchgPlanDisagree: group {tp_group!r} would be planned at "
            f"tp_size={tp_size}, which is the PP form on BOTH sides -- the "
            f"on-card diagonal, not the cross-group exchange. A plan built that "
            f"way resolves both pointer sides and still cuts no shard, so its "
            f"src_resolved=N/N would grade a plan that moves whole tensors "
            f"between two differently shaped groups."
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


# ---------------------------------------------------------------------------
# THE PRODUCT ENTRY POINT -- one leg's plan, from the six manifests.
# ---------------------------------------------------------------------------

#: The provenance string a join-derived plan carries, so a reader can tell at a
#: glance WHICH producer answered.  The derivation's own is
#: ``weight_exchange_shadow.PLAN_SOURCE`` and names ``walk_live_tensors`` +
#: ``build_plan`` over THIS rank's model; this one names the manifests.  Two
#: producers for one leg would be the defect, so the line says which ran.
JOIN_PLAN_SOURCE = (
    "weg2/xchg_manifest.load_manifests+merge_region_tags+join_manifests"
    "(six per-rank manifests, written by each rank's own loader),"
    "weight_exchange.build_plan,weight_exchange_region.N_CARDS"
)


def refusal(reason: str, detail: str = "") -> str:
    """The one shape a refused join prints, so the log carries one spelling."""
    return f"join-{reason}" + (f": {detail}" if detail else "")


def manifests_for_boot(
    *,
    pp_group: str = "P",
    tp_group: str = "D",
    dump_dir: str = "",
    token: str = "",
) -> Tuple[Optional[Tuple[RankManifest, ...]], str]:
    """Every rank's manifest for THIS boot, or a NAMED refusal.

    **NO FALLBACK, AND THE FILE NAME IS IN THE REFUSAL.**  A missing peer
    manifest used to be answered by deriving the plan from this rank's own
    model -- which is the on-card DIAGONAL
    (``weight_exchange_shadow.py:3332-3334``, both groups ``tp_size=1``), the
    shape that produced ``src_resolved=0/N`` on all 24 legs of weg2xsn20.  It
    may never step in silently again, so this returns the EXPECTED PATH of
    what it could not find and the caller refuses on it.
    """
    directory = dump_dir or manifest_dir()
    if not directory:
        return None, refusal("no-dump-dir",
                             f"{DIR_ENV} is unset, so no rank published a "
                             f"manifest and none can be read")
    tok = token or boot_token()
    found = load_manifests(directory, boot_token=tok)
    have = {(m.group, m.rank) for m in found}
    n_cards = n_cards_default()
    # The EXPECTED PATH, with the region tag left as a glob: a rank publishes
    # one file per RUNNER (main model and drafter), so the name carries a tag
    # this reader cannot know in advance -- and printing a path that never
    # exists would send an operator hunting for the wrong file.
    missing = [
        os.path.join(directory,
                     f"{MANIFEST_PREFIX}_{g}_rank{r}_*.json")
        for g in (pp_group, tp_group)
        for r in range(n_cards)
        if (g, r) not in have
    ]
    if missing:
        return None, refusal(
            "manifest-missing",
            f"{len(missing)} of {2 * n_cards} rank manifests absent for "
            f"boot_token={tok or 'unset'} (first expected path: "
            f"{missing[0]}). Refusing: deriving this rank's own view "
            f"instead is the on-card diagonal, which resolves no source and "
            f"is what this slice replaces")
    return found, ""


def leg_plan_from_join(
    *,
    hook: str,
    group: str,
    rank: int,
    manifests: Sequence[RankManifest],
    src_addr=None,
    dst_addr=None,
    pp_group: str = "P",
    tp_group: str = "D",
    #: THIS RANK'S LIVE MODEL, for the materialisation check below.  Optional
    #: only so the hermetic callers that have no model can drive the join.
    model=None,
    log=None,
):
    """ONE leg's :class:`~weight_exchange_shadow.LegPlan`, from the manifests.

    THE DIRECTION IS DERIVED, NOT PASSED: ``weight_exchange.leg_direction``
    already answers it from ``(hook, group)`` -- ``source`` exports, every
    other hook imports -- so a second answer here could disagree with the one
    the leg knob gates on.

    THE PLAN IS NARROWED TO THIS RANK'S ROLE, and that is what makes
    ``src_resolved`` readable per leg rather than per boot: a source-hook leg
    carries the descriptors THIS rank must supply (``src_rank == rank``), a
    destination-hook leg the ones it must receive (``dst_rank == rank``).  The
    full cross-group plan is built first, because ``build_plan``'s tiling
    check (``_check_tiles``) is a statement about EVERY destination rank and
    would pass vacuously on a pre-filtered inventory.
    """
    from sglang.srt.weg2 import weight_exchange_shadow as sh
    from sglang.srt.managers import weg2_memory_saver as ms

    direction = wx.leg_direction(str(hook), str(group))
    try:
        join = join_manifests(manifests, pp_group=pp_group, tp_group=tp_group)
    except (wx.Weg2XchgSourceMissing, wx.Weg2XchgPlanDisagree) as exc:
        return None, refusal("unjoinable", f"{type(exc).__name__}: {exc}")

    try:
        plan = plan_from_join(join, direction=direction,
                              src_addr=src_addr, dst_addr=dst_addr)
    except (wx.Weg2XchgSourceMissing, wx.Weg2XchgPlanDisagree) as exc:
        return None, refusal("plan-refused", f"{type(exc).__name__}: {exc}")

    # THE MATERIALISATION CHECK, AT THE MOMENT IT IS NOT TAUTOLOGICAL.
    #
    # Re-reading the tensor at WRITE time would compare `ParamGeom.of`'s output
    # against the tensor it was just read from -- true by construction and
    # worth nothing.  Here it is a real question: the manifest was written at
    # the END OF WEIGHT LOADING and this runs at the FLIP, after pauses,
    # resumes and (once AMENDMENT 8 lands) a remap of the very pages it
    # describes.  A manifest that has drifted from the hardware is worse than
    # no manifest, because the join and every downstream reader trust it.
    #
    # Only THIS RANK'S OWN pieces, because they are the only ones whose tensor
    # this process can look at -- which is the same boundary the whole slice
    # rests on.
    if model is not None:
        mine_manifest = next(
            (m for m in manifests
             if m.group == str(group) and int(m.rank) == int(rank)), None)
        if mine_manifest is not None:
            try:
                live = {str(n): t for n, t in model.named_parameters()}
            except BaseException:  # noqa: BLE001 -- an observer never raises
                live = {}
            for piece in mine_manifest.pieces:
                tensor = live.get(piece.param_name)
                if tensor is None:
                    continue
                try:
                    refuse_on_materialisation_drift(piece, tensor)
                except wx.Weg2XchgPlanDisagree as exc:
                    return None, refusal("materialisation-drift", str(exc))

    is_source = str(hook) == sh.HOOK_SOURCE
    side = "src_rank" if is_source else "dst_rank"
    mine = tuple(d for d in plan.descs
                 if int(getattr(d, side, -1)) == int(rank))
    if not mine:
        return None, refusal(
            "no-descriptors-for-rank",
            f"hook={hook} group={group} rank={rank} direction={direction}: the "
            f"joined plan has {len(plan.descs)} descriptors and none with "
            f"{side}={rank}. A leg that moves nothing would report a flip that "
            f"moved no weights")

    # The acceptance line and the pointer profile, at the frame that holds the
    # XchgPlan -- the #1342 lesson: `WEG2-XCHG-PLAN` describes an XchgPlan and
    # the LegPlan below has no `plan_id` at all.  Wrapped, because an
    # instrument may never take a derivation down.
    emit = log if log is not None else None
    try:
        wx.emit_plan_line(plan, direction=("d2h" if is_source else "h2d"))
    except BaseException as exc:  # noqa: BLE001
        import logging as _logging

        _logging.getLogger(__name__).info(
            "WEG2-XCHG-PLAN emit failed: %s: %s", type(exc).__name__, exc)
    try:
        profile = wx.pointer_profile(mine)
        wx.record_pointer_profile(profile)
        line = wx.pointer_profile_line(profile, hook=str(hook),
                                       is_source=is_source)
        if emit is not None:
            emit(line)
        else:
            import logging as _logging

            _logging.getLogger(__name__).info("%s", line)
    except BaseException as exc:  # noqa: BLE001
        import logging as _logging

        _logging.getLogger(__name__).info(
            "WEG2-XCHG-POINTER-PROFILE emit failed: %s: %s",
            type(exc).__name__, exc)

    tags = tuple(sorted({str(d.tag) for d in mine}))
    classes = tuple(sorted({t.tensor_class for t in join.tensors}))
    chunk_layers, chunk_count = ms.weight_chunk_geometry()
    facts = sh.LegPlanFacts(
        chunk_layers=int(chunk_layers), chunk_count=int(chunk_count),
        family_tags=tuple(sorted({str(t.tag) for t in join.tensors})),
        waves=tuple(tuple(w) for w in plan.waves),
        cards=tuple(join.cards), classes=classes,
        # THE PROVENANCE IS THE MANIFESTS', and it must NOT read as the
        # derivation's: two producers for one leg is the defect, so the line a
        # reader sees says which one answered.
        source=JOIN_PLAN_SOURCE)
    leg = sh.LegPlan(
        facts=facts, descs=tuple(mine), card=int(rank), tags=tags,
        # The card digest is over the JOIN's geometry for this rank, computed
        # by the same function the derivation uses -- one hash, one owner.
        card_digest=sh.card_geometry_digest(
            [t.geom(tp_is_dst=(direction == wx.LEGS_PP_TO_TP))
             for t in join.tensors], classes),
        agreed_state=sh.MANIFEST_NOT_ASKED,
        population=len(join.tensors), planned=len(join.tensors))
    return leg, ""
