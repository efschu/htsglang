# SPDX-License-Identifier: Apache-2.0
"""W73: can this boot's exchange schedule live on these cards, per wave?

#1273 slice S7.  The weight exchange (``--weg2-weight-source exchange``) frees
the host ring by moving the weights card-to-card instead of through host RAM,
and it pays for that with a VRAM PEAK: for the length of one wave both groups'
bytes are resident on the same card at once.  Spec section 5 states that peak
as exact arithmetic over a measured census, and its own section 9 point 7 says
the honest thing about it -- *"no cell is separately observable"*.  W73 exists
because of that sentence: an un-observable prediction is checked BEFORE either
group starts, against this boot's own NVML totals, and REFUSES rather than
accepts risk.

THE MODEL, and it is spec section 5.0's, not a new one::

    resident(w)  = image_S - freed_S(<w) + taken_W(<=w)
    peak(card)   = max_w resident(w) + resident_tags + 2 x dormant_proc_used
    free_at_peak = nvml_total - peak

``image_S`` is the SLEEPING group's exchanged image on that card; ``taken_W`` is
the WAKING group's; the two extra terms are the ones a wave schedule never
moves -- the never-paused tags (the MTP/NEXTN draft, spec section 4.1, and
after S8 P's vision tower) and the per-rank non-tag overhead, **counted twice
because both ranks are resident during the window** (refuter finding R2-3).

WHAT IS NOT HAND-NUMBERED HERE.  Not one MiB of the model above is written in
this file.  Every byte comes from a census handed in at launch with its own
provenance string, and every total comes from live NVML.  The spec's example
table (25510 / 14305 / 15550 ...) appears in this slice exactly once, in a
TEST, as a fixture carrying boot weg2sb4's name -- which is the only place a
measured number may be quoted from.

THE DIRECTION TOKENS ARE THE TREE'S, NOT THE SPEC EXAMPLE'S.  ``d2p`` means
the bytes flow D -> P: **D sleeps, P wakes**.  That is
:attr:`ring_table.CardRing.need_d2p_mib`'s own docstring ("S = D (sleeping),
W = P (waking)") and it is also the plain reading of the arrow.  The spec's
acceptance line prints the D-WAKES triple under ``(d2p)``, i.e. the opposite
binding; following it would give this tree two contradictory meanings for one
token, which is the direction-is-a-property-of-the-reader defect exactly.  The
token order of the line is the spec's; the binding is the tree's, every
per-card line spells the binding out in words, and the drift is recorded in
WEG2_BUILD_DECISIONS_0906 section 1ai-S7 for the operator to arbitrate.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

#: The two flip directions, named by where the BYTES go.  ``d2p``: D sleeps and
#: P wakes, so D's mapped pages are the source.  ``p2d``: the transpose.
DIRECTIONS: Tuple[str, ...] = ("d2p", "p2d")

#: Which group SLEEPS (owns the authoritative bytes) in each direction, and
#: which WAKES.  One table, so no call site re-derives the arrow.
SOURCE_GROUP: Mapping[str, str] = {"d2p": "D", "p2d": "P"}
WAKING_GROUP: Mapping[str, str] = {"d2p": "P", "p2d": "D"}

#: Spelled out on every per-card line.  A token whose meaning has to be looked
#: up is a token that gets read backwards once.
DIRECTION_WORDS: Mapping[str, str] = {
    "d2p": "D sleeps, P wakes; bytes move D->P",
    "p2d": "P sleeps, D wakes; bytes move P->D",
}

#: The groups a census may name.  Two, and both are required per card: a
#: one-group census cannot state a co-residency peak.
GROUPS: Tuple[str, str] = ("P", "D")

#: WHERE THE ARMING FLOOR COMES FROM, printed beside every use of it (round-2
#: review F5).  It is ``launcher.ARMING_FLOOR_MIB``, the top of this rig's
#: measured VRAM corridor -- and that corridor was measured UNDER LOAD, while
#: the exchange peak occurs with no load running and both KV pools released
#: (spec section 5.0/5.2).  Borrowing it is deliberate and conservative, but a
#: threshold whose regime is not on the line beside it is a number the next
#: reader will re-derive or, worse, trust for the wrong reason.
FLOOR_PROVENANCE: str = (
    "the rig's VRAM corridor ceiling, measured UNDER LOAD; the exchange peak "
    "occurs unloaded with both KV pools released, so this floor is borrowed "
    "from a stricter regime and is conservative here"
)

#: Region layout terms, spec section 6/S3.  They live here ONLY so the arming
#: line can print ``region_mib`` before S3 exists; see
#: :func:`region_mib_from_layout`.
#: TODO(S7->S3): once ``weg2/weight_exchange.py`` owns the region, import its
#: layout instead of these four arguments and delete the helper.  Two modules
#: sizing one file is the second-bookkeeping shape this fork deletes on sight.


def region_mib_from_layout(
    pairs: int, slots_per_pair: int, slot_mib: int, header_mib: int
) -> int:
    """MiB of the /dev/shm staging region: header plus the static slot grid.

    Provenance of the terms the caller passes, spec section 6/S3: ``pairs`` is
    the six DIRECTED cross-card pairs of a 3-card rig, ``slots_per_pair`` = 2
    (double buffering), ``slot_mib`` = 32 (evidence E2 measured 32/64/128 MiB
    indistinguishable below 0.6 %, so the smallest of the three is taken and
    the carve-out is halved for free), ``header_mib`` = 1 (header + gate rows +
    the 6x6 matrix + the pointer directory).

    It is arithmetic over four stated inputs, never a literal 385.
    """
    for name, value in (
        ("pairs", pairs),
        ("slots_per_pair", slots_per_pair),
        ("slot_mib", slot_mib),
        ("header_mib", header_mib),
    ):
        if int(value) < 0:
            raise ValueError(f"region layout term {name} must not be negative: {value}")
    return int(header_mib) + int(pairs) * int(slots_per_pair) * int(slot_mib)


class Weg2XchgRefused(RuntimeError):
    """Every named refusal this module raises at the LAUNCH check.

    A BASE class, exactly as :class:`ring_table.Weg2RingRefused` is one, so
    ``launcher.REFUSALS`` catches the CLASS and a refusal added here later
    inherits ``cli()``'s handler -- the one named line and exit 2 -- rather
    than needing a second entry that can go out of step with the raises.

    ROUND-2 REVIEW FINDING F1 IS WHY IT EXISTS.  The refusal below shipped as a
    bare ``RuntimeError``, a subclass of none of ``REFUSALS``' four members, so
    it escaped ``cli()`` uncaught: exit **1** with a raw traceback -- which any
    wrapper keying on the exit code reads as a crash rather than as the refusal
    it is -- and ``drop_admin_key_file()`` never ran, so a boot that never
    served left its secret behind (#1275 fix 2's "both exits, or the guarantee
    is only half true").  That is boot weg2rg1's W34 defect, which
    ``ring_table.Weg2RingRefused``'s own docstring records, repeated one module
    later; the seam existed and was not used.
    """


class Weg2XchgResidencyUnarmable(Weg2XchgRefused):
    """W73: a card / direction / wave peak, or the wave-1 inequality, does not fit.

    Raised by the LAUNCHER, before either group starts, where
    :class:`ring_table.Weg2RingWeightsUnderSized` (W49) already refuses -- and
    it takes over W49's role in the exchange arm, because under ``exchange``
    there is no host ring left to under-size (spec section 0.4, finding R1-1).
    Exit 2 and a named refusal; there is no fallback that makes an
    over-committed card fit, and a ``cu_mem_create`` OOM three waves into a
    flip that has already unmapped the source is not the refusal this gate
    exists to give.
    """


@dataclass(frozen=True)
class CardCensus:
    """One card's measured bytes, from a PREVIOUS boot (or this one).

    ``tags`` is ``group -> tag -> MiB`` over BOTH groups, because a
    co-residency peak is a statement about two groups on one card.  It carries
    no NVML total on purpose: the total is this boot's own live reading, and a
    census that carried one would let a stale board size a live card.
    """

    uuid: str
    tags: Mapping[str, Mapping[str, int]]
    #: The MEASURED dormant ``proc_used`` of ONE rank on this card, counted
    #: twice by :func:`solve` because both ranks are resident during the
    #: window.  It is an instrument, not a sum of attributed rows (R2-3): it
    #: swallows the flashinfer workspace, the graph pool, the static-state
    #: clones and whatever the allocator cache was on that boot.
    dormant_proc_used_mib: int
    #: Where that number was read.  Printed; never defaulted.
    dormant_source: str = ""


@dataclass(frozen=True)
class XchgCensus:
    """The whole launch-time input: per-card bytes plus the wave partition."""

    cards: Mapping[str, CardCensus]
    #: The wave partition, in the shape the request struct carries it
    #: (spec section 3.1 ``xchg_waves: Optional[List[List[str]]]``).
    #: TODO(S7->S1): ``weight_exchange`` derives this from ``chunk_tag_cards``;
    #: until it lands the launcher has no producer and W73 refuses rather than
    #: inventing a partition.
    #: AND THE HAZARD THAT COMES WITH IT (round-2 refuter F11): the partition
    #: priced here is read by ONE process and transmitted to nobody.  Today
    #: that is harmless because no rank acts on it; from S6 on, the priced
    #: schedule and the EXECUTED schedule are two objects, which is R2-5's
    #: shape one layer up.  S1/S6 must publish this list to the ranks and have
    #: them refuse a mismatch, exactly as the front already refuses a pause
    #: order that is not its own weights tags.
    waves: Tuple[Tuple[str, ...], ...]
    provenance: str = ""
    path: str = ""


@dataclass(frozen=True)
class WaveRow:
    """One (card, direction, wave) cell of spec section 5.1 / 5.2."""

    uuid: str
    nvml_index: int
    name: str
    direction: str
    wave: int
    image_s_mib: int
    freed_s_mib: int
    taken_w_mib: int
    resident_tags_mib: int
    overhead_mib: int
    total_mib: int

    @property
    def exchanged_mib(self) -> int:
        """``image_S - freed_S(<w) + taken_W(<=w)`` -- the moving part."""
        return self.image_s_mib - self.freed_s_mib + self.taken_w_mib

    @property
    def resident_mib(self) -> int:
        """Everything on the card at this wave, including what never moves."""
        return self.exchanged_mib + self.resident_tags_mib + self.overhead_mib

    @property
    def free_mib(self) -> int:
        return self.total_mib - self.resident_mib

    def terms(self) -> str:
        return (
            f"image_S={self.image_s_mib} - freed_S={self.freed_s_mib} "
            f"+ taken_W={self.taken_w_mib} + resident_tags={self.resident_tags_mib} "
            f"+ 2x_dormant_proc_used={self.overhead_mib} "
            f"= {self.resident_mib} MiB of {self.total_mib} MiB"
        )


@dataclass
class XchgResidency:
    """The solved table, its refusals and the lines that state both."""

    rows: List[WaveRow] = field(default_factory=list)
    order: List[str] = field(default_factory=list)
    provenance: str = ""
    floor_mib: float = 0.0
    refusals: List[str] = field(default_factory=list)
    lines: List[str] = field(default_factory=list)

    @property
    def armed(self) -> bool:
        return bool(self.rows) and not self.refusals

    @property
    def waves(self) -> int:
        return max((r.wave for r in self.rows), default=0)

    def peak_row(self, direction: str, uuid: str) -> Optional[WaveRow]:
        """The wave whose residency is the peak, for one card and direction.

        The MAX over waves, never the sum: the schedule allocates as it frees,
        so summing the waves prices a shape this design does not run (and it
        prices every card as fitting, which is the mutant this returns red).
        """
        cells = [r for r in self.rows if r.direction == direction and r.uuid == uuid]
        if not cells:
            return None
        return max(cells, key=lambda r: r.resident_mib)

    def peak_mib(self, direction: str, uuid: str) -> int:
        row = self.peak_row(direction, uuid)
        return row.resident_mib if row is not None else 0

    def free_at_peak_mib(self, direction: str, uuid: str) -> int:
        row = self.peak_row(direction, uuid)
        return row.free_mib if row is not None else 0

    def wave1_rows(self) -> List[WaveRow]:
        """Spec section 5.4's acyclicity proof obligation, one row per case.

        Wave 1 is the only wave whose destination demand is funded by no prior
        release, so if it fits, ``_weg2_await_vram_credit`` takes its zero-cost
        exit at wave 1 and the credit cycle has no base edge.  It is the SAME
        quantity as this card's wave-1 residency -- named separately because it
        is the premise of the deadlock-freedom argument, not because it is a
        second number.
        """
        return [r for r in self.rows if r.wave == 1]

    def wave1_ok(self) -> Tuple[int, int]:
        rows = self.wave1_rows()
        return sum(1 for r in rows if r.resident_mib <= r.total_mib), len(rows)


def _as_int_mib(value: object, what: str) -> int:
    try:
        out = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise Weg2XchgResidencyUnarmable(
            f"W73 Weg2XchgResidencyUnarmable: {what} is not an integer MiB value: {value!r}"
        )
    if out < 0:
        raise Weg2XchgResidencyUnarmable(
            f"W73 Weg2XchgResidencyUnarmable: {what} is negative: {out}"
        )
    return out


def load_census(path: str) -> XchgCensus:
    """Read the launch-time census file, or refuse by name.

    The file is the same kind of input ``--duplex-probe`` already is: a
    measurement a PREVIOUS run produced, handed to the launcher by path so the
    number that arms the boot has a readable source.  There is no default and
    no fallback -- an absent or malformed census under ``--weg2-weight-source
    exchange`` is W73, never a guessed table.
    """
    if not path:
        raise Weg2XchgResidencyUnarmable(
            "W73 Weg2XchgResidencyUnarmable: --weg2-weight-source exchange needs a "
            "per-card census (--weg2-xchg-census <path>) and none was given.  The "
            "exchange's VRAM peak is spec section 5's arithmetic over a MEASURED "
            "census; with no census there is no peak to check, and an unchecked "
            "peak is the cu_mem_create OOM this gate exists to prevent."
        )
    if not os.path.isfile(path):
        raise Weg2XchgResidencyUnarmable(
            f"W73 Weg2XchgResidencyUnarmable: census file {path!r} does not exist"
        )
    try:
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
    except (OSError, ValueError) as exc:
        raise Weg2XchgResidencyUnarmable(
            f"W73 Weg2XchgResidencyUnarmable: census file {path!r} is unreadable: {exc}"
        )
    if not isinstance(blob, dict):
        raise Weg2XchgResidencyUnarmable(
            f"W73 Weg2XchgResidencyUnarmable: census file {path!r} is not a JSON object"
        )
    raw_waves = blob.get("waves")
    if not isinstance(raw_waves, list) or not raw_waves:
        raise Weg2XchgResidencyUnarmable(
            f"W73 Weg2XchgResidencyUnarmable: census file {path!r} carries no "
            "'waves' partition.  TODO(S7->S1): weight_exchange.derive_waves is "
            "its producer; a launcher that invented one would be checking a "
            "schedule the ranks do not run."
        )
    waves: List[Tuple[str, ...]] = []
    for k, wave in enumerate(raw_waves, start=1):
        if not isinstance(wave, list) or not all(isinstance(t, str) for t in wave):
            raise Weg2XchgResidencyUnarmable(
                f"W73 Weg2XchgResidencyUnarmable: wave {k} of {path!r} is not a "
                "list of tag names"
            )
        waves.append(tuple(wave))
    raw_cards = blob.get("cards")
    if not isinstance(raw_cards, dict) or not raw_cards:
        raise Weg2XchgResidencyUnarmable(
            f"W73 Weg2XchgResidencyUnarmable: census file {path!r} carries no 'cards'"
        )
    cards: Dict[str, CardCensus] = {}
    for uuid, entry in raw_cards.items():
        if not isinstance(entry, dict):
            raise Weg2XchgResidencyUnarmable(
                f"W73 Weg2XchgResidencyUnarmable: card {uuid} of {path!r} is not an object"
            )
        tags_raw = entry.get("tags")
        if not isinstance(tags_raw, dict) or set(tags_raw) != set(GROUPS):
            raise Weg2XchgResidencyUnarmable(
                f"W73 Weg2XchgResidencyUnarmable: card {uuid} of {path!r} must carry "
                f"'tags' for both groups {list(GROUPS)}, got "
                f"{sorted(tags_raw) if isinstance(tags_raw, dict) else type(tags_raw).__name__}. "
                "A one-group census cannot state a co-residency peak."
            )
        tags: Dict[str, Dict[str, int]] = {}
        for group in GROUPS:
            per_tag = tags_raw[group]
            if not isinstance(per_tag, dict):
                raise Weg2XchgResidencyUnarmable(
                    f"W73 Weg2XchgResidencyUnarmable: card {uuid} group {group} of "
                    f"{path!r} is not a tag->MiB object"
                )
            tags[group] = {
                str(tag): _as_int_mib(mib, f"card {uuid} group {group} tag {tag}")
                for tag, mib in per_tag.items()
            }
        cards[str(uuid)] = CardCensus(
            uuid=str(uuid),
            tags=tags,
            dormant_proc_used_mib=_as_int_mib(
                entry.get("dormant_proc_used_mib"),
                f"card {uuid} dormant_proc_used_mib",
            ),
            dormant_source=str(entry.get("dormant_source", "")),
        )
    return XchgCensus(
        cards=cards,
        waves=tuple(waves),
        provenance=str(blob.get("provenance", "")),
        path=path,
    )


def _wave_tags(census: XchgCensus) -> List[str]:
    flat: List[str] = []
    for wave in census.waves:
        flat.extend(wave)
    return flat


def check_partition(census: XchgCensus) -> List[str]:
    """The wave list must not REPEAT a tag, and must name no tag the census lacks.

    Same test the front applies to the transmitted pause order
    (``sorted(order) != sorted(self.weights_tags)``, front.py:2656-2663,
    refuter finding R2-5), applied one layer earlier to the object the launcher
    prices.  A tag in two waves is charged twice; a wave tag with no census
    bytes is an absent measurement read as a zero.  Both make the peak a number
    about a schedule nobody runs.

    WHAT THIS DOES **NOT** CHECK, and why the name above says "repeat" and not
    "partition" (round-2 review F6): the reverse gap.  A tag that the census
    carries and NO wave names is not refused -- :func:`solve` classifies it as
    permanently resident and charges BOTH groups' copies of it for the whole
    window.  That is the conservative reading (it can only over-price the peak,
    never under-price it), and it is the reading S8's resident vision tower
    needs; but it means an S1 partition that simply DROPS a real weights tag
    arms without comment, priced as if that tag never moved.  Stated here
    rather than implied by a docstring that claims more than the code does.
    """
    bad: List[str] = []
    flat = _wave_tags(census)
    dupes = sorted({t for t in flat if flat.count(t) > 1})
    if dupes:
        bad.append(
            f"the wave partition repeats {dupes} -- a tag in two waves is freed "
            "twice and taken twice, so every peak below it is fiction"
        )
    for uuid, card in sorted(census.cards.items()):
        for group in GROUPS:
            missing = sorted(set(flat) - set(card.tags[group]))
            if missing:
                bad.append(
                    f"card {uuid} group {group}: the census has no bytes for wave "
                    f"tags {missing} -- an absent tag is not a zero-byte tag, and "
                    "reading it as one under-prices the peak"
                )
    return bad


def solve(
    cards: Sequence[object],
    census: XchgCensus,
    floor_mib: float,
) -> XchgResidency:
    """Spec section 5.1/5.2 per card, per direction, per wave, plus the refusals.

    ``cards`` are this boot's live cards (anything with ``uuid``,
    ``nvml_index``, ``name`` and ``total_mib`` -- :class:`launcher.Card`).  The
    TOTALS are live NVML; the BYTES are the census; the two are joined by UUID
    and never by position, because an NVML re-enumeration between the census
    boot and this one silently swaps two rows otherwise.

    TODO(S7->S6), spec section 5.3, and it is not a nicety (round-2 review F6 /
    refuter F4): this is a LAUNCH-TIME check against the census plus the NVML
    TOTAL.  The spec asks additionally for a read of the live NVML FREE in the
    RPC preamble, because R6 -- the allocator's unbounded cache -- is the one
    term this arithmetic cannot bound, and refusing there is cheaper than
    taking a ``cu_mem_create`` OOM at a wave whose source pages are already
    unmapped.  S6 owns the preamble; until it exists W73 is launch-only and
    that is a WEAKER gate than section 5.3 specifies, not an equal one.
    """
    res = XchgResidency(provenance=census.provenance, floor_mib=float(floor_mib))
    res.order = [getattr(c, "uuid", "") for c in cards]
    # Round-2 refuter F13: every per-card figure below is keyed by UUID, so two
    # cards sharing one key (or carrying none) collapse into one row and the
    # arming line silently prices two boards as one.  Unreachable with real
    # NVML cards, which is exactly the class of assumption that stops holding
    # in a fixture, a mock rig or a driver that returns an empty string.
    blank = [
        getattr(c, "nvml_index", "?") for c in cards if not getattr(c, "uuid", "")
    ]
    if blank:
        res.refusals.append(
            f"card(s) at nvml index {blank} carry no UUID -- the census is joined "
            "by UUID and an empty key is not a join, it is a collision waiting "
            "for a second nameless card"
        )
    dupes = sorted({u for u in res.order if u and res.order.count(u) > 1})
    if dupes:
        res.refusals.append(
            f"UUID(s) {dupes} name more than one live card -- two boards under "
            "one key are priced as one board, and the peak that matters is the "
            "one that was never computed"
        )
    res.refusals.extend(check_partition(census))
    wave_tags = set(_wave_tags(census))
    for card in cards:
        uuid = getattr(card, "uuid", "")
        entry = census.cards.get(uuid)
        if entry is None:
            res.refusals.append(
                f"card {uuid} (nvml{getattr(card, 'nvml_index', '?')} "
                f"{getattr(card, 'name', '?')}) is not in the census "
                f"{census.path or '<inline>'} -- this boot would run a card whose "
                "peak nobody computed.  A census is joined by UUID, never by "
                "position, so a missing card is a missing measurement and not a "
                "re-ordering to be repaired here."
            )
            continue
        total_mib = int(getattr(card, "total_mib", 0))
        # Tags in NEITHER wave are resident for the whole window: the MTP/NEXTN
        # draft has its own tag and is never paused (spec section 4.1), and S8's
        # resident vision tower would join it here without a code change.  Both
        # groups' resident tags count, because both groups' ranks are on the
        # card.
        resident_tags_mib = sum(
            mib
            for group in GROUPS
            for tag, mib in entry.tags[group].items()
            if tag not in wave_tags
        )
        # R2-3: the per-rank non-tag term is the MEASURED dormant proc_used, and
        # it is counted TWICE because both ranks are resident during the window.
        overhead_mib = 2 * int(entry.dormant_proc_used_mib)
        for direction in DIRECTIONS:
            src, dst = SOURCE_GROUP[direction], WAKING_GROUP[direction]
            image_s = sum(entry.tags[src].get(t, 0) for t in wave_tags)
            freed = 0
            taken = 0
            for k, wave in enumerate(census.waves, start=1):
                # Peak is taken BEFORE the source frees this wave (spec section
                # 5.0), so freed_S is the waves STRICTLY before k and taken_W is
                # the waves up to and including k.
                taken += sum(entry.tags[dst].get(t, 0) for t in wave)
                res.rows.append(
                    WaveRow(
                        uuid=uuid,
                        nvml_index=int(getattr(card, "nvml_index", -1)),
                        name=str(getattr(card, "name", "")),
                        direction=direction,
                        wave=k,
                        image_s_mib=image_s,
                        freed_s_mib=freed,
                        taken_w_mib=taken,
                        resident_tags_mib=resident_tags_mib,
                        overhead_mib=overhead_mib,
                        total_mib=total_mib,
                    )
                )
                freed += sum(entry.tags[src].get(t, 0) for t in wave)
    res.refusals.extend(_peak_refusals(res))
    res.lines.extend(_check_lines(res))
    return res


def _peak_refusals(res: XchgResidency) -> List[str]:
    """One line per over-committed (card, direction), naming every term.

    Every quantity spec section 5.3 demands is on the line -- card, direction,
    wave, predicted peak, NVML total, the arming floor and each term of the sum
    -- because a refusal a reader cannot audit sends the next boot to guess at
    the same number.
    """
    bad: List[str] = []
    seen = []
    for row in res.rows:
        key = (row.direction, row.uuid)
        if key in seen:
            continue
        seen.append(key)
        peak = res.peak_row(row.direction, row.uuid)
        if peak is None:
            continue
        if peak.free_mib >= res.floor_mib:
            continue
        bad.append(
            f"card {peak.uuid} nvml{peak.nvml_index} {peak.name} dir={peak.direction} "
            f"({DIRECTION_WORDS[peak.direction]}) wave={peak.wave}: predicted peak "
            f"{peak.resident_mib} MiB against NVML total {peak.total_mib} MiB leaves "
            f"{peak.free_mib} MiB, below the arming floor {res.floor_mib:g} MiB "
            f"({FLOOR_PROVENANCE}).  Terms: {peak.terms()}"
        )
    return bad


def _check_lines(res: XchgResidency) -> List[str]:
    """The per-card, per-direction audit rows that stand under the ARMED line."""
    out: List[str] = []
    seen = []
    for row in res.rows:
        key = (row.direction, row.uuid)
        if key in seen:
            continue
        seen.append(key)
        peak = res.peak_row(row.direction, row.uuid)
        wave1 = next(
            (
                r
                for r in res.rows
                if r.uuid == row.uuid and r.direction == row.direction and r.wave == 1
            ),
            None,
        )
        if peak is None or wave1 is None:
            continue
        out.append(
            f"WEG2-XCHG-CHECK card={peak.uuid} nvml{peak.nvml_index} {peak.name} "
            f"dir={peak.direction} ({DIRECTION_WORDS[peak.direction]}) "
            f"waves={res.waves} peak_wave={peak.wave} peak_mib={peak.resident_mib} "
            f"total_mib={peak.total_mib} free_mib={peak.free_mib} "
            f"floor_mib={res.floor_mib:g} "
            f"wave1_mib={wave1.resident_mib} "
            f"wave1_fits={'yes' if wave1.resident_mib <= wave1.total_mib else 'NO'} "
            f"-- terms: {peak.terms()} "
            f"(floor: {FLOOR_PROVENANCE}.  NVML total is the full board; the "
            "driver carve-out this rig measures at 425-518 MiB/card is NOT "
            "subtracted here, exactly as the awake budget does not subtract it "
            "-- a separate open finding, named so the margin is not read as "
            "larger than it is.  Consequence, stated rather than left to the "
            "reader: free_mib above overstates the NVML-free this card will "
            "actually show by that carve-out, so the effective NVML-free floor "
            "is the printed one MINUS 425-518 MiB)"
        )
    return out


def armed_line(
    res: XchgResidency,
    epoch: object,
    region_mib: int,
    ring_h_mib: int,
    wired: bool,
    unwired_reason: str,
) -> str:
    """The spec's acceptance line, token sequence verbatim, plus ``wired=``.

    ``peak_mib``/``free_mib`` are printed in the launcher's card order (NVML
    ordinal order), ``d2p`` first.  The DIRECTION BINDING is this tree's, not
    the spec example's -- see the module docstring; every WEG2-XCHG-CHECK line
    above spells it out in words so the two cannot be confused by a reader who
    only has the log.

    ``wired`` IS NOT DECORATION (round-2 review F2).  Until S6 propagates
    ``--weg2-weight-source`` into the two groups' argv, the request structs and
    the saver's region flag, a boot launched with ``exchange`` runs the RING
    path end to end and logs this line -- and a log is the only thing anyone
    reads afterwards.  A grep that cannot tell that boot from an exchanging one
    turns this acceptance line into the same defect its own neighbour refuses:
    an unarmed gate read as a passed one.  So the line states, in its own
    tokens, whether any rank behaviour is wired to the arm it just priced.
    """
    ok, total = res.wave1_ok()
    peaks = " ".join(
        "/".join(str(res.peak_mib(d, u)) for u in res.order) + f" ({d})"
        for d in DIRECTIONS
    )
    frees = ",".join(
        "/".join(str(res.free_at_peak_mib(d, u)) for u in res.order)
        for d in DIRECTIONS
    )
    return (
        f"WEG2-XCHG-ARMED epoch={epoch} waves={res.waves} peak_mib={peaks} "
        f"free_mib={frees} wave1_ok={ok}/{total} floor_mib={res.floor_mib:g} "
        f"region_mib={region_mib} ring_H_mib={ring_h_mib} "
        f"wired={'yes' if wired else 'no'} "
        f"reason={'none' if wired else (unwired_reason or 'UNSTATED')} "
        f"-- provenance: {res.provenance or 'UNSTATED (the census named no source)'}"
    )


def refusal_head(res: XchgResidency) -> str:
    """The W73 message body: what failed, and why there is nothing to fall back to."""
    return (
        f"W73 Weg2XchgResidencyUnarmable: the exchange's predicted VRAM residency "
        f"does not fit on {len(res.refusals)} (card x direction) case(s).  This is "
        "spec section 5's arithmetic over a MEASURED census, checked at launch "
        "because no cell of it is separately observable at runtime (spec section "
        "9 point 7); it REPLACES W49's role in this arm, where there is no host "
        "ring left to under-size.  There is no fallback that makes an "
        "over-committed card fit: the boot REFUSES by name and exits 2, BEFORE "
        "either group starts, rather than take a cu_mem_create OOM at the wave "
        "whose source pages are already unmapped.  Run "
        "--weg2-weight-source ring, or re-cut the schedule."
    )
