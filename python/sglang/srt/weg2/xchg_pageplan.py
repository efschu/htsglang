# SPDX-License-Identifier: Apache-2.0
"""#1352 REMAP -- THE PAGE PLAN: which physical page moves, when, and why the
schedule cannot deadlock.

WHAT THIS REPLACES, and it is a design fault rather than a bug.  Today the flip
funds the waking group's allocation out of the sleeping group's release through
a ``VramCredit`` counter (``weg2_memory_saver.py``, ``weight_updater.py``).
That counter has lied twice on metal -- boot xsn7 as a segfault, boot xsn21b as
``W85 credit=2560 requested=1906 free_mib=1370`` on the 5090 -- and the second
reading is the instructive one: the counter and NVML disagreed by 1190 MiB
while both were "correct" about their own books.  A counter is a SECOND set of
books for a fact the driver already keeps.  The remap removes the counter by
removing the allocation: the physical page is not released and re-acquired, it
changes owner.  ``cuMemUnmap`` in the source VA, ``cuMemMap`` of the SAME
handle in the destination VA (``tms_csrc/core.cpp``, ``remap_pages``).

WHAT THIS MODULE IS.  The pure, hermetic arithmetic that decides the order:
which source page is deposited when, which destination page it funds, and
whether the destination is ever asked to scatter into a page that is not backed
yet.  No CUDA, no torch, no device call -- the same reason ``xchg_manifest``
has none: a plan that can only be checked on metal is checked once a boot.

THE FIVE RULES, and where each is enforced:

1. A SOURCE PAGE MOVES ONLY AFTER EVERY CONSUMER HAS READ ITS PIECES.  Under
   the Cut-1 simplification (below) a consumer never reads the page itself --
   it reads the HOST SLOT -- so the condition collapses to "after the last
   window that deposits it".  THAT WINDOW IS GLOBAL OVER ALL DESTINATION
   CARDS, because under PP against TP a source page usually has consumers on
   more than one card, so it is :func:`plan_leg` and not :func:`plan_pages`
   that can compute it.  :func:`verify_leg` checks it independently, and it is
   what caught this module's own per-card first attempt.
2. STREAMING IN ADDRESS ORDER ON BOTH SIDES, UNDER THE BOUNCE CAPACITY.  The
   window is a run of DESTINATION pages whose feeding source pages fit in the
   slot; the planner refuses (W94) when a single destination page's feeders
   already exceed it, and says SLOT rather than plan, because that is the one
   shape no window size can rescue.
3. A FIXED PAGE FUND PER CARD FOR TENSOR BOUNDARIES AND FOR THE SIZE
   DIFFERENCE.  Allocated at BOOT, mapped at the flip, never requested.
   :func:`fund_pages` derives it and :func:`minimum_fund` sizes it against the
   SCHEDULE -- which needs more than the net difference, because the transfers
   that fund a page lag the collects that consume it.  Sufficiency is a PREFIX
   INEQUALITY at every window (the acyclicity obligation of spec 5.4, computed
   rather than argued); when it fails the refusal names the deficit and the
   fund that would have covered it.
4. AN UNEXCHANGEABLE CLASS IS NOT A REFUSAL (user ruling 2026-09-12).  Three
   routes -- exchange, repack, residual ring -- and a residual class keeps
   today's host-ring path while the RING IS SIZED to exactly those bytes.  See
   :func:`classify_routes` / :func:`residual_ring_bytes` / :func:`exchangeable_only`.
5. WHAT GETS DELETED is named in AMENDMENT 8, not here; this module only stops
   producing the numbers the credit counter consumed.

THE CUT-1 SIMPLIFICATION, priced rather than assumed.  Every piece goes through
the host slot, including the ON-CARD ones that could have gone device-to-device.
One address order, no special case, one deadlock argument.  The price is the
on-card bytes crossing PCIe twice; :func:`cut1_cost_ms` states it from the
measured per-card H2D/D2H rate rather than a nominal link figure, so the number
that justifies Cut 2 later is on the record now.

THE ONE INPUT THAT DOES NOT EXIST YET, said plainly.  A page plan needs each
piece's BYTE OFFSET inside its tag's page sequence.  The B4n manifest records
extents (``rows_full``/``cols_full``/``nbytes``) but no offset, because the
CACHING ALLOCATOR decides placement, not the loader -- so the offset is a
reading (``data_ptr() - arena_base``), available locally in each rank at arm
time and nowhere else.  :func:`extents_from_manifest_order` supplies a
prefix-sum MODEL of that layout for the desk, and it is marked as a model in
its own return value: :func:`plan_pages` refuses a plan built from a model
unless ``allow_modelled_offsets`` is passed, which production never does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

#: The remap's unit.  NOT a constant of this file's choosing: it is what
#: ``cuMemGetAllocationGranularity(CU_MEM_ALLOC_GRANULARITY_MINIMUM)`` answers
#: on the running device, read once in ``tms_csrc/core.cpp``.  It appears here
#: only as the DEFAULT for the desk, and every entry point takes it as an
#: argument so a device that answers otherwise re-plans instead of mis-slicing.
PAGE_BYTES_DEFAULT = 2 * 1024 * 1024

#: The direction names, IMPORTED rather than re-spelled where they exist
#: (``layers/dcp/phase_flip_plan.py``); duplicated here only as the literal the
#: hermetic tests compare against, and pinned by a test to those names.
PP_TO_TP = "pp_to_tp"
TP_TO_PP = "tp_to_pp"

PLAN_LINE_PREFIX = "WEG2-XCHG-PAGEPLAN"


class Weg2RemapPlanUnschedulable(RuntimeError):
    """W94: no deadlock-free page schedule exists under the given capacity.

    Raised with the WINDOW, the deficit in pages and the fund that would have
    covered it -- never as a bare "does not fit", because the caller's next
    move is either a bigger fund or a bigger slot and the message has to say
    which one is short.
    """


class Weg2RemapPageRefused(RuntimeError):
    """W95: a page-level precondition of the remap does not hold.

    The same code the C primitive writes into its ``err`` buffer
    (``tms_csrc/core.cpp``), so a refusal raised by the planner and one raised
    by the driver census under it carry ONE name in a boot log.
    """


@dataclass(frozen=True)
class PieceExtent:
    """One tensor piece as it lies in ONE side's arena, in BYTES.

    ``offset`` is measured from the start of the tag's page sequence (the
    tag's allocations concatenated in ADDRESS order -- the same order
    ``tms_remap_pages`` indexes, and it is address order on both sides for
    exactly that reason: a page index is only a stable name for a page if both
    ends derive it identically).
    """

    param_name: str
    card: int
    tag: str
    offset: int
    nbytes: int

    @property
    def end(self) -> int:
        return self.offset + self.nbytes

    def pages(self, page_bytes: int) -> range:
        """The pages this piece touches -- INCLUSIVE of a partial tail.

        A piece rarely starts or ends on a page line, which is the whole reason
        rule 3 exists: the page holding a tensor boundary is needed by BOTH
        neighbours and can belong to only one of them.
        """
        if self.nbytes <= 0:
            return range(0, 0)
        return range(self.offset // page_bytes, (self.end - 1) // page_bytes + 1)


@dataclass(frozen=True)
class SideLayout:
    """One group's byte layout on ONE card, and how it was obtained.

    ``modelled`` is not decoration.  A layout derived by prefix sum over the
    manifest order is a MODEL of what the caching allocator did, and a plan
    built on a model of placement is a plan that can be exactly wrong about
    which page holds which tensor.  Carrying the provenance in the value means
    :func:`plan_pages` can refuse it by name instead of a reader having to
    remember where the numbers came from.
    """

    group: str
    card: int
    tag: str
    pieces: Tuple[PieceExtent, ...]
    modelled: bool

    def total_bytes(self) -> int:
        return max((p.end for p in self.pieces), default=0)

    def total_pages(self, page_bytes: int) -> int:
        return (self.total_bytes() + page_bytes - 1) // page_bytes

    def boundary_pages(self, page_bytes: int) -> int:
        """Pages that carry a tensor boundary -- rule 3's own denominator.

        Counted, not estimated: a page is a boundary page when more than one
        piece touches it.  This is the number the fund must cover for the
        destination to be able to write a partial page without owning the whole
        of it, and it is why the fund is sized from the JOIN at boot rather
        than from a rule of thumb.
        """
        seen: Dict[int, int] = {}
        for piece in self.pieces:
            for pg in piece.pages(page_bytes):
                seen[pg] = seen.get(pg, 0) + 1
        return sum(1 for n in seen.values() if n > 1)


#: The three transport/ownership primitives this plan sequences.  ``deposit``
#: and ``collect`` are B4n's OWN names for the two halves of the phase-split
#: ``run_bounce_leg(phase=deposit|collect|both)`` -- deliberately not a fourth
#: vocabulary for the same two operations.  The remap replaces only WHERE the
#: destination page comes from, never how bytes are deposited and collected.
STEP_DEPOSIT = "deposit"
STEP_COLLECT = "collect"
STEP_TRANSFER = "transfer"
STEP_MAP_FUND = "map_fund"


@dataclass(frozen=True)
class Step:
    """One scheduled operation.  ``kind`` is one of:

    * ``deposit``  -- D2H of source page ``src_page`` into the host slot.  The
      SOURCE rank's half of ``run_bounce_leg``; it reads ``src_ptr`` only.
    * ``collect``  -- H2D out of the slot into destination page ``dst_page``.
      The DESTINATION rank's half; it reads ``dst_ptr`` only, and that address
      is given BY THIS PLAN rather than derived from a refill.
    * ``map_fund`` -- a boot-time fund page is mapped at ``dst_page``.  No
      allocation: the page already exists and has since the boot.
    * ``transfer`` -- ``tms_remap_pages``: the physical page changes owner from
      the source arena's VA to the destination's.  THE ONLY STEP THAT TOUCHES
      VRAM OWNERSHIP, and it allocates nothing.

    THE INVARIANT THAT BINDS THEM (B4n phase-split status, 2026-09-12): a
    ``collect`` NEVER targets a page that is not backed yet.  Its destination
    is always a page this plan has already covered by ``map_fund`` or
    ``transfer``.  Enforced constructively by :func:`plan_pages` and checked
    independently by :func:`verify_collect_invariant`, so a scheduler defect
    cannot pass by agreeing with itself.
    """

    kind: str
    window: int
    src_page: int = -1
    dst_page: int = -1
    n_pages: int = 1
    #: Which CARD's source arena ``src_page`` names.  A page index alone is
    #: ambiguous across the leg: one destination card deposits from all three
    #: source cards, and two of them will have a page 26.  The executor needs
    #: it to know which rank issues the deposit, and :func:`verify_leg` needs
    #: it to check rule 1 without confusing two cards' pages for one.
    src_card: int = -1


@dataclass(frozen=True)
class PagePlan:
    """The schedule and, beside it, every number it is accountable for."""

    card: int
    direction: str
    page_bytes: int
    steps: Tuple[Step, ...]
    #: Pages that change owner by remap.  The design's headline quantity.
    pages_moved: int
    #: Pages the destination took from the boot-time fund because the source
    #: had none left to give at that point in the stream.
    pages_from_fund: int
    #: Pages the source released that the destination never needed; they go
    #: BACK into the fund and are what makes the opposite leg free.
    residual_pool_pages: int
    #: The largest number of pages resident in the host slot at any instant.
    host_slot_peak: int
    #: Pages the destination must be able to write partially -- rule 3.
    boundary_pages: int
    #: The fund this plan was proven against.
    fund_pages: int
    windows: int
    modelled_offsets: bool

    def line(self) -> str:
        """Every number with its denominator, and its instrument named."""
        return (
            f"{PLAN_LINE_PREFIX} card={self.card} direction={self.direction} "
            f"page_bytes={self.page_bytes} windows={self.windows} "
            f"pages_moved={self.pages_moved} "
            f"pages_from_fund={self.pages_from_fund}/{self.fund_pages} "
            f"residual_pool_pages={self.residual_pool_pages} "
            f"host_slot_peak={self.host_slot_peak} "
            f"boundary_pages={self.boundary_pages} "
            f"offsets={'MODELLED' if self.modelled_offsets else 'measured'} "
            f"-- pages_moved counts REMAP transfers, not copies; the bytes still "
            f"travel through the slot (Cut 1). A page is counted once per leg."
        )


def extents_from_manifest_order(
    pieces: Sequence[object],
    *,
    group: str,
    card: int,
    tag: str,
    page_bytes: int = PAGE_BYTES_DEFAULT,
) -> SideLayout:
    """A prefix-sum MODEL of one side's arena layout, marked as a model.

    ``pieces`` are B4n ``ManifestPiece``-shaped objects (``param_name``,
    ``nbytes``); they are duck-typed rather than imported because
    ``xchg_manifest`` is not on every tree this module has to import on, and a
    hard import would make the whole planner unimportable on a tree where B4n
    has not landed.  The ONE field this reads is ``nbytes``, and a piece
    lacking it is a refusal rather than a zero.

    WHY A MODEL AND NOT A DERIVATION.  The caching allocator, not the loader,
    decides where a tensor lands in the arena, so materialisation order is a
    GUESS at placement.  The guess is good enough to size a fund and to test a
    schedule -- it is exactly wrong for deciding which page holds which tensor,
    which is why the return value carries ``modelled=True`` and
    :func:`plan_pages` refuses it unless the caller says, in the call, that it
    is planning for the desk.
    """
    out: List[PieceExtent] = []
    offset = 0
    for piece in pieces:
        name = getattr(piece, "param_name", None)
        nbytes = getattr(piece, "nbytes", None)
        if name is None or nbytes is None:
            raise Weg2RemapPageRefused(
                "W95 Weg2RemapPageRefused: a manifest piece carries no "
                f"param_name/nbytes ({piece!r}); a layout cannot be modelled "
                "from a record that does not say how big the piece is"
            )
        out.append(PieceExtent(param_name=str(name), card=card, tag=tag,
                               offset=offset, nbytes=int(nbytes)))
        offset += int(nbytes)
    return SideLayout(group=group, card=card, tag=tag, pieces=tuple(out), modelled=True)


def fund_pages(src: SideLayout, dst: SideLayout, page_bytes: int = PAGE_BYTES_DEFAULT) -> int:
    """Rule 3 + rule 4: the boot-time page fund for ONE card.

    Two terms, and they are added rather than maxed because they answer
    different questions:

    * THE SIZE DIFFERENCE (rule 4).  ``max(0, dst_pages - src_pages)``: the
      destination's image is bigger than the source's by this many pages, and
      those pages cannot come from the source because the source does not have
      them.  On the opposite leg the term is zero and the same fund sits idle,
      which is why the fund is sized per CARD as the max over both legs and
      allocated ONCE at boot (:func:`fund_for_card`).
    * THE BOUNDARY RESERVE (rule 3).  A page carrying a tensor boundary is
      wanted by two neighbours; the second one needs a page of its own to write
      its partial head into before the two are reconciled.  Counted from the
      destination layout, which is the side that does the writing.

    THE FUND COSTS NO PEAK VRAM, and that is the argument for the whole design
    rather than a convenient footnote: only one group is resident at a time, so
    the card's peak is already ``max(src_pages, dst_pages)``.  The fund is that
    same difference, named and owned instead of released and re-acquired.
    """
    src_pages = src.total_pages(page_bytes)
    dst_pages = dst.total_pages(page_bytes)
    return max(0, dst_pages - src_pages) + dst.boundary_pages(page_bytes)


def fund_for_card(
    layout_p: SideLayout, layout_d: SideLayout, page_bytes: int = PAGE_BYTES_DEFAULT
) -> int:
    """The ONE fund a card holds, covering BOTH legs.

    Allocated at boot from the seam fund, mapped at the flip, never requested
    -- so it must be the larger of what either direction needs.  A fund sized
    for one leg is the #1275 class: correct on the leg that was measured and
    silently short on its mirror.
    """
    return max(
        fund_pages(layout_p, layout_d, page_bytes),
        fund_pages(layout_d, layout_p, page_bytes),
    )


def _feeders(
    sources: Sequence[SideLayout], dst: SideLayout, page_bytes: int
) -> Tuple[Dict[int, List[Tuple[int, int]]], Dict[Tuple[int, int], int]]:
    """destination page -> the source pages feeding it, and each source page's
    LAST destination page (rule 1's release point).

    THE RELATION IS BYTE-WISE, NOT PIECE-WISE, and the difference is not
    pedantry -- it decides the slot.  A piece-wise relation ("this destination
    page needs every source page of its tensor") makes a 128 MiB tensor's every
    destination page depend on all 64 of its source pages, so the slot would
    have to hold the whole tensor and the planner refuses a schedule that is
    in fact perfectly streamable.  Measured while building: the piece-wise
    reading demanded 129 source pages for destination page 0 on the 5090
    against a 64-page slot; the byte-wise reading needs at most two.

    THE CORRESPONDENCE THIS ASSUMES, stated so a caller can violate it
    deliberately rather than by accident: within one ``param_name``, the
    destination's byte *i* comes from the source's byte *i*.  That is true for
    a contiguous RUN, which is what the transport's own ``_blocks_of``
    produces.  A tensor whose destination bytes are a STRIDED view of the
    source (a row-slice with a different row pitch) is therefore not one piece
    here -- it is several, one per contiguous run, each with its own name.  A
    caller handing this a single strided piece would get a plan that is exactly
    wrong about which page feeds which, which is why the run split belongs
    upstream, where the row map already lives.

    SOURCES ARE ALL CARDS, TRANSFERS ARE ONE CARD, and keeping the two apart is
    the structural half of this module.  A destination piece is fed by whatever
    card holds it -- under PP against TP that is usually ANOTHER card, which is
    exactly why the host bounce exists.  But a PHYSICAL PAGE CANNOT CHANGE
    CARD: ``tms_remap_pages`` refuses a cross-device move by name.  So a feeder
    is ``(card, page)`` and only the feeders on the DESTINATION'S OWN card can
    ever fund it by remap; everything else is deposit-and-collect traffic whose
    pages are released on their own card, against their own card's destination.

    Pieces present on no side are not silently dropped -- they would be a hole
    in the destination, which is W74's business at join time and must never be
    invented here.
    """
    by_name: Dict[str, Tuple[int, PieceExtent]] = {}
    for side in sources:
        for piece in side.pieces:
            if piece.param_name in by_name:
                raise Weg2RemapPageRefused(
                    f"W95 Weg2RemapPageRefused: run {piece.param_name!r} is published "
                    f"by two source cards ({by_name[piece.param_name][0]} and "
                    f"{side.card}). A run has ONE holder; two is a join defect (W68), "
                    f"and picking either would be a rank-local derivation of a "
                    f"cross-group fact."
                )
            by_name[piece.param_name] = (side.card, piece)
    feeders: Dict[int, List[Tuple[int, int]]] = {}
    last_use: Dict[Tuple[int, int], int] = {}
    for piece in dst.pieces:
        found = by_name.get(piece.param_name)
        source = found[1] if found is not None else None
        src_card = found[0] if found is not None else -1
        if source is None:
            raise Weg2RemapPageRefused(
                "W95 Weg2RemapPageRefused: destination piece "
                f"{piece.param_name!r} on card {dst.card} has no source piece of "
                "that name. Assembly does not create a source; a missing one is "
                "W74 at join time and is refused here rather than planned around."
            )
        if source.nbytes != piece.nbytes:
            raise Weg2RemapPageRefused(
                f"W95 Weg2RemapPageRefused: piece {piece.param_name!r} on card "
                f"{dst.card} is {piece.nbytes} B at the destination and "
                f"{source.nbytes} B at the source. A run must be the same bytes on "
                f"both ends; a size disagreement is a JOIN defect (W68) and is "
                f"refused rather than padded or truncated here."
            )
        if piece.nbytes <= 0:
            continue
        for dst_pg in piece.pages(page_bytes):
            # the destination bytes THIS page carries, intersected with the piece
            lo = max(piece.offset, dst_pg * page_bytes)
            hi = min(piece.end, (dst_pg + 1) * page_bytes)
            if lo >= hi:
                continue
            s_lo = source.offset + (lo - piece.offset)
            s_hi = source.offset + (hi - piece.offset)
            slot = feeders.setdefault(dst_pg, [])
            for s in range(s_lo // page_bytes, (s_hi - 1) // page_bytes + 1):
                key = (src_card, s)
                if key not in slot:
                    slot.append(key)
                if last_use.get(key, -1) < dst_pg:
                    last_use[key] = dst_pg
    return feeders, last_use


def plan_leg(
    sources: Sequence[SideLayout],
    destinations: Sequence[SideLayout],
    *,
    direction: str,
    funds: Dict[int, int],
    slot_bytes: int,
    page_bytes: int = PAGE_BYTES_DEFAULT,
    allow_modelled_offsets: bool = False,
) -> Dict[int, PagePlan]:
    """The WHOLE leg: every destination card, planned together.

    WHY THE LEG AND NOT THE CARD, and it is rule 1 rather than convenience.  A
    source page may move only once ALL ITS CONSUMERS have read it, and under PP
    against TP a source page usually has consumers on MORE THAN ONE card -- a
    replicated tensor is read by all three.  A per-card planner sees only its
    own reads, so it would release a page that another card still means to
    deposit, and the second card's deposit would then read a page that belongs
    to someone else.  Silent, and exactly the corruption direction the ordering
    exists to prevent.

    So the leg is planned in two passes: each destination is windowed
    independently (they stream in parallel, each against its own slot), and
    only then is each source page's release point taken as the MAXIMUM over
    every destination that reads it.  ``funds`` is per destination card.
    """
    sources = tuple(sources)
    destinations = tuple(destinations)
    modelled = any(s.modelled for s in sources) or any(d.modelled for d in destinations)
    _validate(sources, direction, slot_bytes, page_bytes, modelled,
              allow_modelled_offsets)
    if not destinations:
        raise Weg2RemapPageRefused(
            "W95 Weg2RemapPageRefused: a leg needs at least one destination layout"
        )
    plans: Dict[int, PagePlan] = {}
    windowing: Dict[int, Tuple[Tuple[int, int, List[Tuple[int, int]]], ...]] = {}
    feeds: Dict[int, Dict[int, List[Tuple[int, int]]]] = {}
    # ---- pass 1: window every destination, and collect the global last use ---
    global_last: Dict[Tuple[int, int], int] = {}
    for dst in destinations:
        f, _ = _feeders(sources, dst, page_bytes)
        feeds[dst.card] = f
        wins = _windows(f, dst.total_pages(page_bytes),
                        slot_bytes // page_bytes, dst.card, page_bytes)
        windowing[dst.card] = wins
        for w, (lo, hi, _feed) in enumerate(wins):
            for q in range(lo, hi):
                for key in f.get(q, ()):
                    if global_last.get(key, -1) < w:
                        global_last[key] = w
    # ---- pass 2: emit, releasing at the GLOBAL last window -------------------
    for dst in destinations:
        plans[dst.card] = _emit(
            sources, dst, direction=direction, fund=funds.get(dst.card, 0),
            windows=windowing[dst.card], feeders=feeds[dst.card],
            global_last=global_last, page_bytes=page_bytes,
            modelled=(dst.modelled or any(s.modelled for s in sources)),
            allow_modelled_offsets=allow_modelled_offsets, slot_bytes=slot_bytes,
        )
    return plans


def plan_pages(
    src,
    dst: SideLayout,
    *,
    direction: str,
    fund: int,
    slot_bytes: int,
    page_bytes: int = PAGE_BYTES_DEFAULT,
    allow_modelled_offsets: bool = False,
) -> PagePlan:
    """ONE destination card's schedule -- :func:`plan_leg` with one destination.

    Correct ONLY when this card is the leg's only destination, or when the
    caller has already established that no source page it releases is read by
    another card.  Everything else must go through :func:`plan_leg`, which is
    the one that can see rule 1's cross-card half.

    ``src`` is either ONE :class:`SideLayout` (the on-card case) or the
    sequence of every card's source layout.  Deposits may come from any of
    them; only the one whose ``card`` equals ``dst.card`` can fund the
    destination by remap, because a physical page cannot change card.
    """
    sources: Tuple[SideLayout, ...] = (
        (src,) if isinstance(src, SideLayout) else tuple(src)
    )
    modelled = dst.modelled or any(s.modelled for s in sources)
    _validate(sources, direction, slot_bytes, page_bytes, modelled,
              allow_modelled_offsets)
    if len([s for s in sources if s.card == dst.card]) > 1:
        raise Weg2RemapPageRefused(
            f"W95 Weg2RemapPageRefused: card {dst.card} publishes more than one "
            f"source layout; a card has ONE source arena per leg"
        )

    feeders, last_use = _feeders(sources, dst, page_bytes)
    windows = _windows(feeders, dst.total_pages(page_bytes),
                       slot_bytes // page_bytes, dst.card, page_bytes)
    return _emit(
        sources, dst, direction=direction, fund=fund, windows=windows,
        feeders=feeders,
        global_last={k: _window_of(windows, v) for k, v in last_use.items()},
        page_bytes=page_bytes, modelled=modelled,
        allow_modelled_offsets=allow_modelled_offsets, slot_bytes=slot_bytes,
    )


def _validate(sources, direction, slot_bytes, page_bytes, modelled, allow_modelled):
    """The preconditions both entry points share, in one place."""
    if direction not in (PP_TO_TP, TP_TO_PP):
        raise Weg2RemapPageRefused(
            f"W95 Weg2RemapPageRefused: unknown direction {direction!r}; the two "
            f"legs are {PP_TO_TP!r} and {TP_TO_PP!r} and a plan for neither is a "
            "wiring defect, not a third mode"
        )
    if not sources:
        raise Weg2RemapPageRefused(
            "W95 Weg2RemapPageRefused: a plan needs at least one source layout"
        )
    if modelled and not allow_modelled:
        raise Weg2RemapPageRefused(
            "W95 Weg2RemapPageRefused: this plan rests on MODELLED byte offsets "
            "(a prefix sum over manifest order), but the caching allocator -- not "
            "the loader -- decides placement. A production plan needs the arm-time "
            "reading (data_ptr() - arena_base) in the manifest. Pass "
            "allow_modelled_offsets=True only for desk arithmetic."
        )
    if slot_bytes < page_bytes:
        raise Weg2RemapPlanUnschedulable(
            f"W94 Weg2RemapPlanUnschedulable: the host slot holds {slot_bytes} B, "
            f"less than one page ({page_bytes} B). No window size rescues that."
        )


def _window_of(windows, dst_page: int) -> int:
    for w, (lo, hi, _feed) in enumerate(windows):
        if lo <= dst_page < hi:
            return w
    return len(windows) - 1


def _windows(feeders, dst_total: int, slot_pages: int, card: int, page_bytes: int):
    """Destination pages cut into runs whose feeders fit in the slot at once.

    The cut is greedy in ADDRESS ORDER (rule 2) and never splits a destination
    page, so a page whose feeders alone overflow the slot is a SLOT shortfall
    and is refused as one -- no window size can rescue it, and saying "the plan
    does not fit" there would send the reader to the wrong knob.
    """
    windows = []
    start = 0
    while start < dst_total:
        acc = []
        end = start
        while end < dst_total:
            here = feeders.get(end, ())
            if len(here) > slot_pages:
                raise Weg2RemapPlanUnschedulable(
                    f"W94 Weg2RemapPlanUnschedulable: destination page {end} on card "
                    f"{card} is fed by {len(here)} source pages, more than the "
                    f"slot's {slot_pages}. A window cannot be made smaller than one "
                    f"destination page, so the SLOT is short, not the schedule: it "
                    f"needs at least {len(here) * page_bytes} B."
                )
            need = [s for s in here if s not in acc]
            if acc and len(acc) + len(need) > slot_pages:
                break
            acc.extend(need)
            end += 1
        windows.append((start, end, acc))
        start = end
    return tuple(windows)


def _emit(sources, dst, *, direction, fund, windows, feeders, global_last,
          page_bytes, modelled, allow_modelled_offsets, slot_bytes) -> PagePlan:
    """Turn one destination's windows into steps, releasing at the GLOBAL last
    window of each local source page (rule 1, cross-card)."""
    local = [s for s in sources if s.card == dst.card]
    local_src = local[0] if local else None
    dst_total = dst.total_pages(page_bytes)
    steps: List[Step] = []
    #: ONE CURSOR for destination pages, and that is the fix for a defect this
    #: module's own verifier caught: the fund and the transfers each used to
    #: number their destination pages from zero, so both claimed page 0 and
    #: nobody claimed the tail.  A destination page is backed by EITHER a fund
    #: page or a transferred one, in address order, so there is one cursor.
    backed_upto = 0
    fund_left = fund
    pages_from_fund = 0
    host_slot_peak = 0
    moved = 0

    for w, (lo, hi, feed) in enumerate(windows):
        host_slot_peak = max(host_slot_peak, len(feed))
        for card, s in feed:
            steps.append(Step(kind=STEP_DEPOSIT, window=w, src_page=s, src_card=card))
        # ---- back this window's pages BEFORE collecting into them -----------
        # Pages already backed by transfers banked in earlier windows cost
        # nothing here; the rest are drawn from the fund, and that draw is the
        # MEASUREMENT of how much of the size difference the source could not
        # pay for at this point in the stream -- not a budget handed out up
        # front.  Running out is the prefix inequality failing, and it says so.
        while backed_upto < hi:
            if fund_left <= 0:
                deficit = hi - backed_upto
                raise Weg2RemapPlanUnschedulable(
                    f"W94 Weg2RemapPlanUnschedulable: window {w} on card {dst.card} "
                    f"({direction}) collects into destination pages {lo}..{hi - 1}, "
                    f"but only {backed_upto} are backed at that point (fund {fund}, "
                    f"{moved} transferred). Short by {deficit} page(s); a fund of "
                    f"{fund + deficit} would have covered it. The deficit is in "
                    f"BACKED PAGES, not in slot bytes -- a bigger slot does not help."
                )
            steps.append(Step(kind=STEP_MAP_FUND, window=w, dst_page=backed_upto))
            fund_left -= 1
            pages_from_fund += 1
            backed_upto += 1
        for q in range(lo, hi):
            steps.append(Step(kind=STEP_COLLECT, window=w, dst_page=q))
        # ---- rule 1: release only what no LATER window reads, anywhere -------
        # ONLY THE DESTINATION'S OWN CARD can donate: a physical page does not
        # change card, so a feeder page on another card is deposited, read, and
        # then released against THAT card's own destination -- never here.  The
        # window is the GLOBAL last one, computed over every destination, which
        # is why this has to come from plan_leg and not from one card's reads.
        releasable = sorted(
            page for (card, page), last_w in global_last.items()
            if card == dst.card and last_w == w
        )
        for s in releasable:
            if backed_upto >= dst_total:
                break          # nothing left to back; the rest is residual
            steps.append(Step(kind=STEP_TRANSFER, window=w, src_page=s,
                              dst_page=backed_upto, src_card=dst.card))
            backed_upto += 1
            moved += 1

    src_total = local_src.total_pages(page_bytes) if local_src is not None else 0
    residual = max(0, src_total - moved)

    return PagePlan(
        card=dst.card,
        direction=direction,
        page_bytes=page_bytes,
        steps=tuple(steps),
        pages_moved=moved,
        pages_from_fund=pages_from_fund,
        residual_pool_pages=residual,
        host_slot_peak=host_slot_peak,
        boundary_pages=dst.boundary_pages(page_bytes),
        fund_pages=fund,
        windows=len(windows),
        modelled_offsets=bool(modelled),
    )


#: THE THREE ROUTES A TENSOR CLASS CAN TAKE (user ruling 2026-09-12, verbatim:
#: *"alles was nicht byte uebertragbar ist (und nicht in sehr kurzer zeit
#: repackt werden kann) darf weiter im ringpuffer liegen. er muss nicht ganz
#: abgeschafft werden, er muss nur sehr viel kleiner werden."*).
#:
#: A class that cannot be exchanged is NOT a refusal.  It keeps today's path --
#: sleep writes it to the host ring, wake reads it back -- and the ring is
#: sized from the MANIFEST to exactly those bytes instead of to the whole
#: image.  The ring shrinks; it does not disappear.
ROUTE_EXCHANGE = "exchange"        # (a) byte-transferable: remap + copy
ROUTE_REPACK = "repack"            # (b) transferable after a NAMED repack step
ROUTE_RESIDUAL_RING = "residual"   # (c) stays in the host ring, and sizes it


def classify_routes(
    layouts: Sequence[SideLayout],
    route_of,
    page_bytes: int = PAGE_BYTES_DEFAULT,
) -> Dict[int, Dict[str, int]]:
    """Per card, the bytes on each of the three routes.

    ``route_of(param_name) -> ROUTE_*``.  Nothing here decides WHICH route a
    class takes -- that is a property of the class's storage (is a P byte a D
    byte?) and belongs beside the quantisation method, not in a scheduler.
    What this does is COUNT, so the residual ring's size is a measurement over
    the manifest and not a guess.

    Route (b) is admitted only when the repack is faster than the exchange it
    buys; that comparison needs a measured repack rate, so a caller that has
    none should classify the class as (c) and say so, rather than assume.

    A route this does not recognise is refused -- not because a class may not
    be unexchangeable, but because an unrecognised ROUTE STRING is a caller
    defect, and silently bucketing it would mis-size the ring in whichever
    direction the typo fell.
    """
    out: Dict[int, Dict[str, int]] = {}
    for side in layouts:
        cell = out.setdefault(side.card, {ROUTE_EXCHANGE: 0, ROUTE_REPACK: 0,
                                          ROUTE_RESIDUAL_RING: 0})
        for piece in side.pieces:
            route = route_of(piece.param_name)
            if route not in cell:
                raise Weg2RemapPageRefused(
                    f"W95 Weg2RemapPageRefused: run {piece.param_name!r} was "
                    f"classified as {route!r}, which is not one of "
                    f"{ROUTE_EXCHANGE!r}/{ROUTE_REPACK!r}/{ROUTE_RESIDUAL_RING!r}. "
                    f"An unexchangeable class is fine -- an unrecognised route "
                    f"string is a caller defect that would mis-size the ring."
                )
            cell[route] += piece.nbytes
    return out


def residual_ring_bytes(
    layouts: Sequence[SideLayout],
    route_of,
    page_bytes: int = PAGE_BYTES_DEFAULT,
) -> Dict[int, int]:
    """THE NEW RING SIZE, per card: the sum of the class-(c) bytes.

    This is the term that goes into the host ledger in place of the whole
    dormant image.  It is a MEASUREMENT over the manifest, so a form whose
    classes are all byte-transferable gets a ring of ZERO and a form with a
    repack-only class gets exactly that class -- neither is a constant and
    neither is a guess.

    The ring is refused by the LEDGER when this sum breaks the reap mark, never
    by the class: an unexchangeable class is a fact about storage, and refusing
    it here would be refusing the model rather than the budget.
    """
    return {card: cell[ROUTE_RESIDUAL_RING]
            for card, cell in classify_routes(layouts, route_of, page_bytes).items()}


def exchangeable_only(side: SideLayout, route_of) -> SideLayout:
    """The same layout with the class-(c) runs REMOVED, not refused.

    Their bytes stay where they are today (the host ring), so the page plan
    must not try to move them -- and must not refuse because of them either.
    Offsets are left untouched: the pages a (c) run occupies stay occupied, and
    pretending otherwise would hand the planner a compacted arena that does not
    exist.
    """
    keep = tuple(p for p in side.pieces
                 if route_of(p.param_name) != ROUTE_RESIDUAL_RING)
    return SideLayout(group=side.group, card=side.card, tag=side.tag,
                      pieces=keep, modelled=side.modelled)


def minimum_fund(
    sources: Sequence[SideLayout],
    destinations: Sequence[SideLayout],
    *,
    direction: str,
    slot_bytes: int,
    page_bytes: int = PAGE_BYTES_DEFAULT,
    allow_modelled_offsets: bool = False,
    max_rounds: int = 64,
) -> Dict[int, int]:
    """The smallest per-card fund under which this leg schedules at all.

    THE NUMBER IS NOT THE NET SIZE DIFFERENCE, and that is the finding this
    function exists to make visible.  :func:`fund_pages` gives what the two
    images differ by at the END; the schedule needs enough backed pages at
    EVERY POINT, and the transfers that fund it lag the collects that consume
    it by however long a source page's last reader takes to arrive.  A fund
    sized on the net difference is therefore short by the worst-case LEAD, and
    it would be short only sometimes -- the failure mode that looks like a
    flaky flip rather than a sizing error.

    Derived by taking the planner's own refusal at its word (it names the fund
    that would have covered the window it failed on) and re-planning, which
    converges because each round strictly increases the fund of the card that
    refused.  ``max_rounds`` bounds it so a defect cannot loop.
    """
    funds = {
        d.card: fund_for_card(
            next((s for s in sources if s.card == d.card), d), d, page_bytes)
        for d in destinations
    }
    for _ in range(max_rounds):
        try:
            plan_leg(sources, destinations, direction=direction, funds=funds,
                     slot_bytes=slot_bytes, page_bytes=page_bytes,
                     allow_modelled_offsets=allow_modelled_offsets)
            return funds
        except Weg2RemapPlanUnschedulable as exc:
            grew = False
            for card in funds:
                marker = f"on card {card} "
                if marker in str(exc) and "a fund of " in str(exc):
                    want = int(str(exc).split("a fund of ")[1].split()[0])
                    if want > funds[card]:
                        funds[card] = want
                        grew = True
            if not grew:
                raise
    raise Weg2RemapPlanUnschedulable(
        f"W94 Weg2RemapPlanUnschedulable: no fund converged in {max_rounds} rounds "
        f"for {direction}; last try {funds}. That is a planner defect, not a "
        f"sizing answer, and it is raised rather than returning the last guess."
    )


def verify_collect_invariant(plan: PagePlan) -> None:
    """ONE card's plan: no collect into a page nothing has backed yet.

    THE POINT OF A SECOND READING.  :func:`plan_pages` builds the schedule and
    believes its own arithmetic; a defect in that arithmetic agrees with itself
    perfectly.  This function knows nothing about windows or funds -- it walks
    the emitted steps with one set and asks only whether every ``collect``'s
    destination was already covered by a ``map_fund`` or a ``transfer``.  A
    collect into an unmapped page is not slow; it is a fault on a page that is
    not there (B4n phase-split invariant).

    RULE 1 IS **NOT** CHECKED HERE, and the reason is a defect this verifier
    itself found: deposits are issued by whichever card HOLDS the page, so one
    card's plan legitimately transfers a page whose deposits appear in another
    card's plan.  Checking rule 1 per card reported a correct schedule as
    broken.  It is checked across the leg instead -- :func:`verify_leg`.
    """
    backed: set = set()
    for i, step in enumerate(plan.steps):
        if step.kind in (STEP_MAP_FUND, STEP_TRANSFER):
            backed.add(step.dst_page)
        elif step.kind == STEP_COLLECT:
            if step.dst_page not in backed:
                raise Weg2RemapPageRefused(
                    f"W95 Weg2RemapPageRefused: step {i} collects into destination "
                    f"page {step.dst_page}, which no map_fund or transfer has backed "
                    f"yet. A collect must only ever target a page this plan has "
                    f"already mapped (B4n phase-split invariant)."
                )
        elif step.kind != STEP_DEPOSIT:
            raise Weg2RemapPageRefused(
                f"W95 Weg2RemapPageRefused: step {i} has unknown kind {step.kind!r}"
            )


def verify_leg(plans: Dict[int, PagePlan]) -> None:
    """THE WHOLE LEG: rule 1, across the cards that share a source page.

    The cards execute in lockstep by window -- within window ``w`` every card
    deposits, then every card collects, then every card transfers -- so this
    replays exactly that order and asks two things a per-card reading cannot:

    1. A page is never TRANSFERRED before every card that reads it has
       deposited it.  This is rule 1's real form: under PP against TP a source
       page usually has consumers on more than one card.
    2. A page is never DEPOSITED after it was transferred away.  The same rule
       seen from the reader's end, and the one that turns into silent
       corruption rather than a fault, because the VA is still mapped -- by
       somebody else.

    Also re-runs :func:`verify_collect_invariant` per card, so one call covers
    both invariants.
    """
    for plan in plans.values():
        verify_collect_invariant(plan)
    max_w = max((s.window for p in plans.values() for s in p.steps), default=-1)
    deposited: set = set()
    moved: Dict[Tuple[int, int], int] = {}
    for w in range(max_w + 1):
        for kind in (STEP_DEPOSIT, STEP_COLLECT, STEP_MAP_FUND, STEP_TRANSFER):
            for card in sorted(plans):
                for step in plans[card].steps:
                    if step.window != w or step.kind != kind:
                        continue
                    key = (step.src_card, step.src_page)
                    if kind == STEP_DEPOSIT:
                        if key in moved:
                            raise Weg2RemapPageRefused(
                                f"W95 Weg2RemapPageRefused: card {card} deposits "
                                f"source page {step.src_page} of card "
                                f"{step.src_card} in window {w}, but it was already "
                                f"transferred away in window {moved[key]}. Rule 1: a "
                                f"page moves only after its LAST reader, and this "
                                f"reader comes after the move."
                            )
                        deposited.add(key)
                    elif kind == STEP_TRANSFER:
                        if key not in deposited:
                            raise Weg2RemapPageRefused(
                                f"W95 Weg2RemapPageRefused: card {card} transfers "
                                f"source page {step.src_page} of card "
                                f"{step.src_card} away in window {w} before any card "
                                f"deposited it. Its bytes would be gone before a "
                                f"consumer read them."
                            )
                        moved[key] = w


def cut1_cost_ms(oncard_bytes: int, gbs: float) -> float:
    """Cut 1's price: the on-card bytes crossing PCIe TWICE instead of never.

    ``gbs`` is the MEASURED per-card H2D/D2H rate of this rig (13-14 GB/s on
    the WEG2-FLIP-TAG lines, #1277 section 1af), not a nominal link number, and
    it is a parameter rather than a constant so that a boot measuring otherwise
    re-prices instead of quoting this file.

    The figure exists so that Cut 2 (a device-to-device lane for the on-card
    pieces) is a decision with a number beside it rather than an intuition.
    """
    if gbs <= 0:
        raise Weg2RemapPageRefused(
            "W95 Weg2RemapPageRefused: a transfer rate of 0 GB/s cannot price "
            "anything; measure the card before asking what Cut 1 costs"
        )
    return (2.0 * oncard_bytes) / (gbs * 1e9) * 1000.0
