# SPDX-License-Identifier: Apache-2.0
"""#1350: THE SEAM GRADER -- a rank-local content digest over the weight seam,
keyed by the PLACEMENT IDENTITY the tree already publishes.

THE QUESTION IT ANSWERS, and it is the only one the exchange must answer:
**did the exchange bring MY bytes back?**  Taken on one rank, over that rank's
own pieces, at two moments that are both AWAKE: the last instant before the
weights family is paused, and the first instant after it has landed again.

WHY A DIGEST AND NOT A memcmp (operator ruling 2026-09-11 ~16:2xZ, recorded in
``/spinning/gpu-arb/weg2/PLAN_S6_BOUNCE_0911.md``).  RE-STAMP 11 moved the byte
comparison onto the authoritative path expecting a SECOND rank-local copy to
exist there.  There is none, and the reason is a USER LAW rather than a defect:
``tms_csrc/core.cpp`` frees the pinned host image immediately after the H2D
restore (``cudaFreeHost(metadata.cpu_backup_granules[k])`` plus ``clear()`` at
both sites; the ``cpu_backup_from_ring`` branch returns to the ring instead of
retaining), which is ``gewichtsaustausch-ziel-kein-dauer-hostram`` implemented
literally.  A byte-faithful memcmp needs two copies and the law guarantees
exactly one, so the memcmp is unfoundable on BOTH seams.

A round-trip digest is not the weaker substitute.  It is the stronger reading:
a memcmp against a second copy would have proven COPY fidelity, this proves
ROUND-TRIP fidelity, and the round trip is what the exchange performs.

THE KEY IS THE PLACEMENT, NOT A SECOND INVENTORY (operator direction 2026-09-12)
===============================================================================
The first build of this module walked ``model.named_parameters()`` itself and
keyed on its own ``(name, shape, dtype)`` triple.  That was **second
bookkeeping** beside an existing truth and is re-cut here.  Placement is a pure
function of (header, P cut, D vector, quantisation) and is decided by the LOADER
at boot; the tree already carries it as :class:`weight_exchange.ParamGeom` and
:class:`weight_exchange.XchgDesc`, produced by ``build_plan`` /
``derive_leg_plan`` / ``derive_card_manifest``.  A second inventory built from
the live model is the W80/W84/W19 family: two readings of one fact, drifting
where it is least visible.

So the piece enumeration comes from ONE producer,
``weight_exchange_shadow.card_inventory`` -- the walk extracted out of
``derive_card_manifest`` in this same commit, so the manifest the co-located
pair agrees over and the pieces this grader hashes cannot be two different sets.

THE IDENTITY FIELDS, and why each one is in the key
---------------------------------------------------
``PieceIdentity`` carries, per piece:

* ``param_name`` -- the identity ``XchgDesc`` itself uses ("never an allocation
  index, because the C++ side iterates an unordered_map whose order differs
  between the two processes").  **This is what buys the localisation**: a moved
  piece is named, not merely counted.
* ``cls`` -- ``weight_exchange_shadow.tensor_class(name)``, the spec section 2.2
  class.  In the key because the offsets that can be wrong are wrong for a
  whole class at once (R5), so a verdict that names the class is already most
  of the triage.
* ``tag`` -- ``weight_exchange.tag_of_parameter_name``, region-aware.  This is
  the piece's ARENA COORDINATE in the ring layout and it is a boot constant per
  rank; the exchange moves bytes BETWEEN tags across cards, but a rank's own
  piece keeps its tag across its own round trip.
* ``card`` -- the rank's device index.  The placement's card coordinate; it is
  in the key so a piece that arrives on a different card is a refusal rather
  than a silent content difference.
* ``rows``/``cols``/``itemsize`` -- the STORAGE extents from ``StorageGeom``,
  i.e. the slice the copy primitive names.  These are exactly the extents
  ``manifest_entry`` publishes, and the reason they ride along is the reason
  that function gives: a name alone would let two holders agree on a piece
  whose bytes they do not share.

WHAT IS DELIBERATELY **NOT** IN THE KEY, and this is condition (1) itself:
``data_ptr``, ``storage_offset``, the storage object, and the PITCH.  All four
move when the arena moves, which the exchange does BY DESIGN.  A digest keyed
on any of them would report MISMATCH on every correct flip -- Instrument-lies
class A, with the grader as the defect.  ``manifest_entry`` already excludes
the pitch for the same reason, which is why this key is that function's output
plus the two coordinates it does not carry (tag and card).

THE THREE OPERATOR CONDITIONS
=============================
(1) LOGICAL CONTENT PER PIECE, NEVER AN ARENA SPAN, KEYED BY PLACEMENT.
    :func:`piece_digest` hashes the CANONICAL ROW-MAJOR image of the piece's
    logical values; the reported ``bytes`` is ``numel * element_size`` and not
    the storage extent, so a piece living in a padded (pitched) allocation
    prices its own content and not the pad.
(2) DEFAULT OFF.  Hashing ~9.6 GiB per rank is not free, and an always-on
    grader would be a flip-cost regression, i.e. a performance defect caused by
    the instrument.  Armed only for instrument boots (the S6I shape) through
    ONE named knob, :data:`ARM_FLAG` = ``--weg2-seam-digest``, published by the
    launcher as :data:`ENV_ARM`.  There is no env rung without that flag, and
    the rung is resolved and REPORTED through the #901 knob authority
    (``knob_resolution.py``) rather than read behind its back.
(3) AN HONEST LIMIT, on the line and in the record.  **Amended by the
    re-keying, and the amendment is stated rather than assumed:** the operator's
    original condition read "a digest DETECTS, it does not LOCALIZE".  With the
    placement key that hole is closed one level: a mismatch NAMES the pieces
    (param_name, class, tag, card) whose content moved.  What remains true, and
    what :data:`LIMIT_CLAUSE` therefore says, is that it does not localise
    WITHIN a piece -- no byte offset, no row, no cause -- and that a green
    digest is still NEVER "layout verified".

AWAKE-ONLY IS STRUCTURAL, NOT A COST CHOICE
===========================================
``understand_tensor-map.md`` section 5.4 with ``weg2_memory_saver.py:123``
(``WEG2_SLEEP_TAGS = {kv_cache, weights}``): at flip time the sleeping group
holds NO weight bytes in VRAM at all -- ``pause`` releases the VMM physical
pages and the bytes survive only in the host backup.  So a "compare at the
cutover" design is not expensive, it is IMPOSSIBLE.  That is why the two
moments are the pre-pause and post-landing seams on the AWAKE side, and why
nobody should later re-open this as a gap.  Both moments lie outside the
no-return region (#875 DO-NOT-BUILD untouched): no peer, no collective, no byte
moved between cards.

WHAT IS NOT COVERED, said here so a green reading is never over-read
====================================================================
* The population is exactly ``card_inventory``'s: the parameters of the weights
  FAMILY that the plan can describe.  BUFFERS ARE NOT IN IT (the static state
  is a different mover with a different failure mode), and neither is a
  parameter ``ParamGeom.of`` refuses -- the same skip rule the plan applies, so
  the grader never claims a piece the exchange could not move.  The skipped
  count is PRINTED, because an unpriced skip is how a population shrinks
  silently.
* A digest detects and localises TO THE PIECE.  When it fires, the localisation
  within that piece has not been done -- it has been justified.

LIFECYCLE OF THE ONE PIECE OF STATE (the rule this project pays for otherwise):
``SchedulerWeightUpdaterManager.weg2_seam_before``.
  WRITER  -- ``_weg2_seam_digest_before``, at the first weights RPC of a sleep.
  READER  -- ``_weg2_seam_digest_after``, after the landing of the next wake.
  DELETER -- ``_weg2_seam_digest_after`` itself (read-and-clear), and the
             unarmed path, which clears rather than inherits.
  SEPARATING EVENT -- the pause and the whole dormancy in between.  No third
  party touches it, and no cutover clears it: that is asserted by
  ``test_the_only_call_sites_are_the_two_awake_seams``.
The epoch of each reading is RECORDED but never GATED on: a rank releases at
one flip's epoch and resumes at the next one's, so equality would be the wrong
guard and would turn every correct round trip into an UNARMED.

A PLACEMENT-BEFORE-LOAD RE-CUT (B4n) IS A SEPARATE ITEM.  This grader only has
to keep the key STABLE across it, which it does by construction: every field
above comes from the producer, so a re-cut that changes how placement is
decided changes both readings together or neither.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from sglang.srt import knob_resolution as kr

__all__ = [
    "ARM_FLAG",
    "DEFAULT_CHUNK_BYTES",
    "ENV_ARM",
    "LIMIT_CLAUSE",
    "LINE_PREFIX",
    "MAX_NAMED_PIECES",
    "POPULATION",
    "PieceIdentity",
    "PieceReading",
    "REASON_ABSENT",
    "REASON_CONTENT",
    "REASON_MATCH",
    "REASON_PLACEMENT",
    "REFUSAL_MARKER",
    "SeamReading",
    "SeamVerdict",
    "VERDICT_MATCH",
    "VERDICT_MISMATCH",
    "VERDICT_UNARMED",
    "W_CODE",
    "Weg2SeamDigestMismatch",
    "arming_line",
    "arming_resolution",
    "compare",
    "identity_of",
    "piece_digest",
    "read_pieces",
    "seam_digest_armed",
    "take_reading",
    "take_reading_if_armed",
    "unarmed_verdict",
]

# ---------------------------------------------------------------------------
# The refusal.  W90 enumerated from test_weg2_wcode_uniqueness_1263's census
# (the used set at d294bde41d tops out at W89; 5, 6, 13, 19, 24, 27, 30, 39, 73
# and 90..99 read free) AND cross-checked with a word-bounded grep over
# python/sglang/srt -- the #1332 B1b lesson: that census has been blind to a
# holder form four times, so a number is free only when two independent
# readings say so.  W19 is the standing example: the census calls it free and
# ``front.py`` holds it under a non-Weg2 name.
# ---------------------------------------------------------------------------
W_CODE = "W90"
REFUSAL_NAME = "Weg2SeamDigestMismatch"
REFUSAL_MARKER = W_CODE + " " + REFUSAL_NAME


class Weg2SeamDigestMismatch(RuntimeError):
    """The pieces that came back are not the pieces that went in."""


# ---------------------------------------------------------------------------
# The knob.  ONE named flag, published by the launcher into ONE env name, and
# resolved through the #901 authority so the boot log says which rung decided.
# ---------------------------------------------------------------------------
#: The launcher flag an operator passes for an S6I instrument boot.
ARM_FLAG = "--weg2-seam-digest"
#: What the launcher publishes to both groups' workers.  Launcher OUTPUT: it is
#: POPPED when the flag is absent (``launcher.prepare_env``), so a value
#: inherited from an operator's shell can never arm a grader this boot did not
#: ask for -- the same discipline the ring, region and on-card families follow.
ENV_ARM = "SGLANG_WEG2_SEAM_DIGEST"
#: The ONLY values that arm.  Anything else -- including an empty string and a
#: plausible typo -- leaves the grader off: the one direction an environment
#: read may not take is "arm something", and here arming costs a whole-shard
#: hash on the critical path of every flip leg.
_TRUTHY = ("1", "true", "yes", "on")

#: Bounded host buffer per hash step.  A piece is read in blocks and never
#: assembled into a second image -- re-creating a host-resident weight image
#: here would reintroduce the exact term #1273 S6-BOUNCE exists to delete.
DEFAULT_CHUNK_BYTES = 32 << 20

#: How many moved pieces the verdict line NAMES before it prints a remainder.
#: The line is read by a human; an unbounded list would be a log dump, and a
#: SILENT truncation would be a denominator defect, so the remainder is printed
#: as a count either way.
MAX_NAMED_PIECES = 8

LINE_PREFIX = "WEG2-SEAM-DIGEST"
#: Condition (3), as amended by the placement re-keying, and ON THE LINE -- a
#: boot seat reads the line, not this module.
LIMIT_CLAUSE = (
    "LIMIT: a digest DETECTS and localises TO THE PIECE it is keyed by "
    "(param_name/class/tag/card); it does NOT localise within a piece -- no "
    "byte offset, no row, no cause -- and a green digest is NEVER "
    "'layout verified'"
)
#: The denominator, on every line, per the denominator law.
POPULATION = (
    "weight_exchange_shadow.card_inventory (the weights family this rank's "
    "plan can describe); buffers, draft/MTP and plan-undescribable shapes NOT "
    "covered"
)

VERDICT_MATCH = "MATCH"
VERDICT_MISMATCH = "MISMATCH"
#: Not a third grade -- the NAMED ABSENCE OF A GRADE.  Never a pass, and it
#: refuses nothing either: an absence is not a finding in either direction.
VERDICT_UNARMED = "UNARMED"

REASON_MATCH = "content-identical"
REASON_CONTENT = "content-changed"
REASON_PLACEMENT = "placement-changed"
REASON_ABSENT = "no-before-reading"


def arming_resolution(environ: Optional[dict] = None) -> kr.Resolution:
    """Who decided whether the grader is armed -- #901 ladder, two rungs.

    Pure, like every other ``resolve_knob`` site: it logs nothing and latches
    nothing.  :func:`arming_line` is the separate announcement the boot-time
    site makes once.
    """
    environ = os.environ if environ is None else environ
    raw = kr.env_value(ENV_ARM, environ)
    return kr.resolve_knob(
        [
            kr.KnobSource(
                source=kr.env_source(ENV_ARM),
                kind=kr.KIND_ENV,
                present=kr.env_present_nonempty(ENV_ARM, environ),
                reader=lambda: (raw or "").strip().lower() in _TRUTHY,
                label=f"{ENV_ARM}={raw!r}",
                cost=(
                    "the seam grader would hash every piece on every flip leg, "
                    "which is a flip-cost regression on a serving boot"
                ),
            ),
            kr.KnobSource(
                source=kr.PROVENANCE_DEFAULT,
                kind=kr.KIND_DEFAULT,
                present=True,
                value=False,
            ),
        ]
    )


def seam_digest_armed(environ: Optional[dict] = None) -> bool:
    """Default FALSE.  The one predicate; everything else reads this."""
    return bool(arming_resolution(environ).value)


def arming_line(environ: Optional[dict] = None) -> str:
    """The one provenance line, naming the flag an operator would have to pass."""
    res = arming_resolution(environ)
    return kr.provenance_line(
        LINE_PREFIX,
        [
            kr.provenance_field("armed", res.value, res.source),
            f"flag={ARM_FLAG}",
            f"env={ENV_ARM}",
            f"population={POPULATION}",
            LIMIT_CLAUSE,
        ],
    )


# ---------------------------------------------------------------------------
# THE PLACEMENT IDENTITY.  Built from the producer's ParamGeom, never from the
# live tensor's shape -- see the module docstring for why each field is in it.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PieceIdentity:
    """One piece's placement, as the plan already names it."""

    param_name: str
    cls: str
    tag: str
    card: int
    rows: int
    cols: int
    itemsize: int
    #: ``weight_exchange_shadow.manifest_entry`` -- the tree's PUBLISHED,
    #: pointer-free, process-stable identity.  Carried verbatim so the grader's
    #: key and the co-located pair's agreed manifest are literally the same
    #: number, not two encodings of it.
    entry: Tuple[int, int, int, int, int] = ()

    @property
    def key(self) -> Tuple[Any, ...]:
        """What two readings are matched on.  No pointer, no pitch, no arena."""
        return (self.entry, self.tag, int(self.card))

    @property
    def label(self) -> str:
        """What a human reads when this piece moved."""
        return f"{self.param_name}[{self.cls}]@{self.tag}/card{self.card}"

    @property
    def bytes(self) -> int:
        return int(self.rows) * int(self.cols) * int(self.itemsize)


def identity_of(geom: Any, *, card: int) -> PieceIdentity:
    """The identity of one ``ParamGeom``, on this card.

    ``tensor_class`` and ``manifest_entry`` are imported from the shadow module
    rather than re-implemented: they ARE the published identity, and a private
    copy of either would be the second bookkeeping this re-keying removes.
    """
    from sglang.srt.weg2 import weight_exchange_shadow as shadow

    cls = shadow.tensor_class(str(geom.name))
    return PieceIdentity(
        param_name=str(geom.name),
        cls=str(cls),
        tag=str(geom.tag),
        card=int(card),
        rows=int(geom.rows_full),
        cols=int(geom.cols_full),
        itemsize=int(geom.itemsize),
        entry=shadow.manifest_entry(
            str(geom.name), cls, int(geom.rows_full), int(geom.cols_full),
            int(geom.itemsize),
        ),
    )


# ---------------------------------------------------------------------------
# The content digest.  CANONICAL ROW-MAJOR BYTES, in bounded blocks.
# ---------------------------------------------------------------------------
def _canonical_blocks(tensor: Any, chunk_bytes: int):
    """Yield the piece's logical content as canonical row-major byte blocks.

    ``contiguous()`` is what makes this a CONTENT reading rather than a span
    reading: it materialises the row-major image of the logical values, so a
    fresh arena, a non-zero storage offset and a padded pitch all produce the
    SAME bytes -- and, crucially, the pad bytes of a pitched allocation are not
    in the reading at all.  The slicing keeps the transient bounded: for an
    already contiguous piece no copy happens, and for a pitched one only one
    block is ever materialised.
    """
    if tensor.numel() == 0:
        return
    if tensor.dim() == 0:
        yield _block_bytes(tensor.reshape(1))
        return
    rows = int(tensor.shape[0])
    row_elems = max(1, tensor.numel() // max(1, rows))
    row_bytes = max(1, row_elems * tensor.element_size())
    rows_per_block = max(1, int(chunk_bytes) // row_bytes)
    for start in range(0, rows, rows_per_block):
        yield _block_bytes(tensor[start : start + rows_per_block])


def _block_bytes(block: Any):
    """One block's canonical bytes, as a host buffer hashlib can read.

    ``view(uint8)`` rather than ``numpy()`` on the tensor's own dtype, because
    this fork's weights are ``bfloat16`` and ``float8_e4m3fn``, neither of which
    numpy has a dtype for -- and a per-dtype conversion would be a second
    reading of the same bytes with its own rounding behaviour.  The byte image
    is hashed exactly as it sits in the canonical row-major layout.
    """
    import torch

    host = block.detach().contiguous()
    if host.device.type != "cpu":
        host = host.to("cpu")
    return memoryview(host.reshape(-1).view(torch.uint8).numpy())


def piece_digest(
    tensor: Any, chunk_bytes: int = DEFAULT_CHUNK_BYTES
) -> Tuple[str, int]:
    """``(digest, logical_bytes)`` of ONE piece's content.

    ``logical_bytes`` is ``numel * element_size``, i.e. the piece's own extent.
    It is NOT the storage span, and the difference is condition (1) restated one
    level down: a pitched piece lives in a larger allocation, and pricing that
    would print a number no reader could reconcile with the plan.
    """
    h = hashlib.blake2b(digest_size=16)
    for block in _canonical_blocks(tensor, chunk_bytes):
        h.update(block)
    return h.hexdigest(), int(tensor.numel()) * int(tensor.element_size())


@dataclass(frozen=True)
class PieceReading:
    identity: PieceIdentity
    digest: str
    bytes: int


def read_pieces(
    inventory: Iterable[Tuple[Any, Any]],
    *,
    card: int,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> Tuple[PieceReading, ...]:
    """Hash every ``(ParamGeom, tensor)`` pair the producer handed over.

    Sorted by key, because the walk order is a property of the process and the
    reading must be a property of the card.
    """
    out: List[PieceReading] = []
    for geom, tensor in inventory:
        identity = identity_of(geom, card=card)
        digest, nbytes = piece_digest(tensor, chunk_bytes=chunk_bytes)
        out.append(PieceReading(identity=identity, digest=digest, bytes=nbytes))
    return tuple(sorted(out, key=lambda r: (str(r.identity.key), r.identity.param_name)))


# ---------------------------------------------------------------------------
# The readings and the verdict.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SeamReading:
    """One rank's reading of its own pieces at one awake moment."""

    stage: str
    group: str
    rank: int
    card: int
    tags: Tuple[str, ...]
    epoch: Any
    pieces: Tuple[PieceReading, ...] = ()
    ms: float = 0.0

    @property
    def tag_field(self) -> str:
        return ",".join(self.tags) if self.tags else "none"

    @property
    def n_tensors(self) -> int:
        return len(self.pieces)

    @property
    def bytes(self) -> int:
        return sum(int(p.bytes) for p in self.pieces)

    @property
    def placement_key(self) -> str:
        """The PLACEMENT set's own digest -- the key the two readings must share.

        Over the identities only.  Two readings whose placement keys differ
        describe different piece sets, and grading their content would be a
        different finding wearing this one's name.
        """
        h = hashlib.blake2b(digest_size=8)
        for piece in self.pieces:
            h.update(repr(piece.identity.key).encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    @property
    def digest(self) -> str:
        """The card's content digest: the fold of ``key || piece digest``.

        The KEY is folded in beside the content on purpose, so two different
        placements can never collide into one digest by accident, while
        :func:`compare` still refuses the placement case BY NAME rather than
        letting it read as a content difference.
        """
        h = hashlib.blake2b(digest_size=16)
        for piece in self.pieces:
            h.update(repr(piece.identity.key).encode("utf-8"))
            h.update(b"\x00")
            h.update(piece.digest.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def line(self) -> str:
        return (
            f"{LINE_PREFIX} stage={self.stage} group={self.group} "
            f"rank={self.rank} card={self.card} tag={self.tag_field} "
            f"epoch={self.epoch} placement_key={self.placement_key} "
            f"n_tensors={self.n_tensors} bytes={self.bytes} "
            f"digest={self.digest} ms={self.ms:.0f} population={POPULATION}"
        )


def take_reading(
    stage: str,
    inventory: Iterable[Tuple[Any, Any]],
    *,
    group: str,
    rank: int,
    card: int,
    tags: Sequence[str],
    epoch: Any,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> SeamReading:
    import time

    t0 = time.perf_counter()
    pieces = read_pieces(inventory, card=card, chunk_bytes=chunk_bytes)
    return SeamReading(
        stage=stage,
        group=str(group),
        rank=int(rank),
        card=int(card),
        tags=tuple(str(t) for t in tags),
        epoch=epoch,
        pieces=pieces,
        ms=(time.perf_counter() - t0) * 1000.0,
    )


def take_reading_if_armed(
    stage: str,
    inventory: Iterable[Tuple[Any, Any]],
    *,
    group: str,
    rank: int,
    card: int,
    tags: Sequence[str],
    epoch: Any,
    environ: Optional[dict] = None,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> Optional[SeamReading]:
    """``None`` when unarmed, AND WITHOUT READING ONE PIECE.

    The gate is here and not at the call site so that "unarmed" can be proven to
    cost nothing: the inventory is never consumed, which is what
    ``test_default_is_off_and_not_one_byte_is_read`` asserts with an iterable
    that raises when touched.  A boolean check alone could not tell an unarmed
    hook from an armed hook whose result was discarded.
    """
    if not seam_digest_armed(environ):
        return None
    return take_reading(
        stage,
        inventory,
        group=group,
        rank=rank,
        card=card,
        tags=tags,
        epoch=epoch,
        chunk_bytes=chunk_bytes,
    )


def _named(labels: Sequence[str]) -> str:
    """A bounded, never-silently-truncated list of piece labels."""
    shown = list(labels[:MAX_NAMED_PIECES])
    rest = len(labels) - len(shown)
    body = ",".join(shown) if shown else "none"
    return body if rest <= 0 else f"{body},+{rest}-more"


@dataclass(frozen=True)
class SeamVerdict:
    verdict: str
    reason: str
    before: Optional[SeamReading]
    after: Optional[SeamReading]
    #: THE LOCALISATION the placement key buys.  Piece labels, never counts
    #: alone: a count is what the pre-re-keying design could produce and is
    #: exactly the "detects but does not localise" hole the operator closed.
    moved: Tuple[str, ...] = ()
    gone: Tuple[str, ...] = ()
    arrived: Tuple[str, ...] = ()
    #: Only used by :func:`unarmed_verdict`, where there is no reading to read
    #: the identity off.
    group: str = "?"
    rank: int = -1
    card: int = -1
    tags: Tuple[str, ...] = ()

    def _identity(self) -> Tuple[str, int, int, str]:
        ref = self.after or self.before
        if ref is None:
            return self.group, self.rank, self.card, (",".join(self.tags) or "none")
        return ref.group, ref.rank, ref.card, ref.tag_field

    def line(self) -> str:
        group, rank, card, tag = self._identity()
        before, after = self.before, self.after
        ref = after or before
        parts = [
            f"{LINE_PREFIX} stage=verdict group={group} rank={rank} "
            f"card={card} tag={tag}",
            f"placement_key_before={before.placement_key if before else 'n/a'}",
            f"placement_key_after={after.placement_key if after else 'n/a'}",
            f"n_tensors={ref.n_tensors if ref else 0}",
            f"bytes={ref.bytes if ref else 0}",
            f"digest_before={before.digest if before else 'n/a'}",
            f"digest_after={after.digest if after else 'n/a'}",
            f"epoch_before={before.epoch if before else 'n/a'}",
            f"epoch_after={after.epoch if after else 'n/a'}",
            f"ms={(before.ms if before else 0.0) + (after.ms if after else 0.0):.0f}",
            f"verdict={self.verdict}",
            f"reason={self.reason}",
            f"moved={len(self.moved)} pieces_moved={_named(self.moved)}",
            f"gone={len(self.gone)} pieces_gone={_named(self.gone)}",
            f"arrived={len(self.arrived)} pieces_arrived={_named(self.arrived)}",
            f"population={POPULATION}",
        ]
        if self.verdict == VERDICT_UNARMED:
            parts.append(
                "UNARMED is the named ABSENCE of a verdict and is never a pass"
            )
        parts.append(LIMIT_CLAUSE)
        return " ".join(parts)

    def refusal(self) -> Optional[Weg2SeamDigestMismatch]:
        """The refusal object for a MISMATCH -- ``None`` for anything else.

        A verdict object rather than a raise, so the one call site decides WHEN
        to raise and the line always reaches the log first.  What is NOT
        optional is that a MISMATCH acts at all: an instrument that produces a
        verdict nobody consumes is the class this campaign has paid for
        repeatedly, and ``_weg2_seam_digest_after`` raises this.
        """
        if self.verdict != VERDICT_MISMATCH:
            return None
        return Weg2SeamDigestMismatch(f"{REFUSAL_MARKER}: {self.line()}")


def unarmed_verdict(
    reason: str, *, group: str, rank: int, card: int = -1, tags: Sequence[str] = ()
) -> SeamVerdict:
    return SeamVerdict(
        verdict=VERDICT_UNARMED,
        reason=reason,
        before=None,
        after=None,
        group=str(group),
        rank=int(rank),
        card=int(card),
        tags=tuple(str(t) for t in tags),
    )


def compare(
    before: Optional[SeamReading], after: Optional[SeamReading]
) -> SeamVerdict:
    """The grade.  THREE outcomes, and the third is not a grade.

    THE ORDER OF THE TWO CHECKS IS LOAD-BEARING.  The PLACEMENT is asked first,
    and a placement change is reported BY ITS OWN NAME rather than falling
    through into the content comparison.  Two readings over two placements
    describe different piece sets; grading their content is not a weaker
    finding, it is a different one, and calling it ``content-changed`` would
    send a reader hunting for a corrupted byte that does not exist.  Mutant M3
    removes this check and
    ``test_a_changed_placement_refuses_the_content_compare_by_name`` is what
    dies.
    """
    if after is None or before is None:
        return SeamVerdict(
            verdict=VERDICT_UNARMED,
            reason=REASON_ABSENT,
            before=before,
            after=after,
        )
    b_map = {p.identity.key: p for p in before.pieces}
    a_map = {p.identity.key: p for p in after.pieces}
    if before.placement_key != after.placement_key:
        gone = sorted(b_map[k].identity.label for k in b_map.keys() - a_map.keys())
        arrived = sorted(a_map[k].identity.label for k in a_map.keys() - b_map.keys())
        return SeamVerdict(
            VERDICT_MISMATCH,
            REASON_PLACEMENT,
            before,
            after,
            gone=tuple(gone),
            arrived=tuple(arrived),
        )
    moved = sorted(
        a_map[k].identity.label
        for k in a_map
        if a_map[k].digest != b_map[k].digest
    )
    if moved:
        return SeamVerdict(
            VERDICT_MISMATCH, REASON_CONTENT, before, after, moved=tuple(moved)
        )
    return SeamVerdict(VERDICT_MATCH, REASON_MATCH, before, after)
