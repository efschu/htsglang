"""WEG2 flip-cost C19/C20: the per-card ring table, SOLVED from a boot's own lines.

Nothing in this module is a constant that a human chose.  Every quantity it
returns is read out of the PREVIOUS boot's logs, carries the boot tag and the
line count it was read from, and is printed beside the arithmetic it feeds
(spec section 10.3: no hand-sized ring, no ``skew`` constant, no ``--ring-mib``
flag, no env knob a human sets).  When the lines are absent or short this
module returns ``None`` with a REASON -- it never guesses, and the launcher
prints that reason as R22.  R22 IS A REFUSAL, NOT A FALLBACK (spec Amendment
A1-3, and FIX 3 round 3 retracting the sentence that said the OLD form ran
there): no other flip form is feasible on this host budget, so the launcher
exits 2 by name -- measured on the rig, a pin matching no boot gives W20 and
rc=2, never a boot on a different form.

The four quantities, and the exact line each comes from:

``image_g(c)``
    The host bytes group ``g`` parks for card ``c`` -- spec Amendment A1-2's
    MEASURED DORMANT IMAGE, read with TWO instruments that are NOT equals:

    * THE PER-CARD CENSUS, and it is the one that is charged: ``WEG2-FLIP-TAG
      ... card=<uuid> ... bytes=N MiB`` (C16), summed over ALL BACKED-UP TAGS
      of that card in one pass and taken as the PEAK pass, never the mean -- a
      ring sized to a mean blocks on the pass that was larger.  Every tag whose
      allocation carries ``enable_cpu_backup`` emits one, so this sum IS the
      dormant image and not a proxy for it.  A boot older than C16 carries only
      ``WEG2-CHUNK-BYTES sleep ... host_image_delta=`` instead, which names the
      ``weights_*`` family alone and is therefore a LOWER BOUND, marked as such.
    * THE GROUP CROSS-CHECK: ``WEG2 DORMANT-IMAGE group=g ... rss_shmem_gib=``
      -- the sleeping group's summed per-rank ``RssShmem`` at its first sleep,
      emitted by the front (``host_ledger.format_dormant_image``, the draft-KV
      branch's fix-8 sidecar mechanism, REUSED here rather than twinned).  It is
      a WHOLE-GROUP figure with no card attribution, so it can never key a row
      by itself; what it can do is refute a census that is short, and where it
      is larger the excess is apportioned over the cards BY THEIR SHARE OF THAT
      CENSUS -- the only attribution the two instruments jointly support, and
      the line says so.  Boot weg2dk7 measured group P at 38.63 GiB against a
      28.83 GiB weight-tag census, +34 %: that gap is exactly what this
      cross-check exists to charge.

    WHAT THIS REPLACES, and why it had to go (carried review finding 1 of the
    ring fix-2 tip).  The predecessor read a "resident image" as the peak minus
    the baseline of the two absolute ``RssShmem`` readings on the
    ``WEG2-CHUNK-BYTES`` line -- but the emitter takes BOTH of those readings
    immediately before and immediately after the ``weights_*`` pause loop, so
    the span brackets that loop and nothing else, and the "resident image" was
    the weight-tag census a second time plus baseline drift.  Any tag paused
    outside that bracket was structurally invisible to it, which is precisely
    the population A1-2 is about.  Two instruments that are really one is worse
    than one, because the agreement reads as corroboration.

``dormant_D(c)`` -- A BOUND, NOT A MEASUREMENT, until D sleeps with the
    instrument on.  D's first sleep is a flip, so its ``WEG2 DORMANT-IMAGE``
    sample is INTERLEAVED and only its RssShmem term measures D at all; when no
    D sample exists the row is charged ``max(image_P(c), tags_D(c) + extra_P(c))``
    -- D's own census plus the same unattributed excess P showed -- and the
    provenance line prints the word BOUND.

``max_tag_g(c)``
    The largest single tag of that same population -- the step size of R5's
    corridor walk.

``device_credit(c, S->W)``
    ``NVML free while S is awake`` + ``the kv_cache device bytes S releases``.
    The first from the front's ``WEG2-CORRIDOR phase=S(awake) ... nvmlN:free=``
    samples, taken as the MINIMUM over the phase (the conservative end: a
    smaller credit makes the requirement larger, never smaller).  The second
    from that group's ``KV Cache is allocated ... K size: X GB, V size: Y GB``
    lines, summed over every pool the rank allocates.

    IN ALLOCATABLE FREE, ALWAYS (fix 3).  The corridor law, the arm's verdict
    and this credit are all statements about the free the driver will actually
    hand to an allocation.  Boots up to weg2rg6 sampled ``total - used``, which
    is that free PLUS the driver carve-out, so the source's number is converted
    by :func:`corridor_allocatable` before it is credited -- or the boot is
    refused by name.  The conversion, per card and with its source, is printed
    in the RING provenance line (:attr:`RingTable.credit_correction`).

Then R5, per card per direction, with ``S`` the sleeping group and ``W`` the
waking one::

    H(c)    = max_g image_g(c)
    span1(c)= image_P(c)                       # R7: registered at P's first pause
    need    = image_W(c) - credit(c) + max_tag_S(c) + max_tag_W(c)
    slack   = H(c) - need                      # negative => W32, before either
                                               #             group starts

INSTRUMENT NOTE (spec R8), stated here because this module is the last consumer
that may legitimately read it: ``host_image_delta`` is an RssShmem delta and it
DIES the moment the ring lands -- ring granules are tmpfs pages mapped by both
co-located processes, so the delta collapses to ~0.  A table solved from a
RING boot must therefore come from ``tms_tag_bytes`` (C7/C16, next slice);
this module reads whichever of the two instruments the source boot carries and
NAMES it in the provenance string, so a number can never be quoted without its
instrument.

CARD IDENTITY (FIX 2).  Every row is keyed by the card's UUID, end to end.  The
group logs are keyed by rank and the front's corridor samples by NVML index,
both of which are POSITIONS of the boot that WROTE them -- and NVML enumeration
can differ in the boot that reads them, which silently swaps two rows of equal
shape.  So the source boot's own ``NVML -> CUDA ordinal map`` line is read first
(:func:`parse_card_identity`), the ring-era ``WEG2-FLIP-TAG ... card=<uuid>``
field wins where it exists, and a boot that recorded neither is REFUSED by name
rather than paired by position.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Collection, Dict, List, Mapping, Optional, Sequence, Tuple

from sglang.srt.weg2 import host_ledger

MIB = 1024 * 1024
GIB = float(2**30)
GB = 1e9

#: ``[2026-09-07 21:10:23 TP2] WEG2-CHUNK-BYTES sleep tags=['weights_0'] host_image_delta=982 MiB``
_CHUNK_RE = re.compile(
    r"\b(?:TP|PP)(\d+)\]\s+WEG2-CHUNK-BYTES\s+sleep\s+tags=\[([^\]]*)\]\s+"
    r"host_image_delta=(-?\d+)\s+MiB"
)
#: _DORMANT_RE IS DELETED (reconciliation 2026-09-08) and the deletion is the
#: point: it scraped ``WEG2 DORMANT-IMAGE`` out of the front's log text, which
#: made this module a SECOND reader of a fact the draft-KV line already carries
#: structurally in its fix-8 JSON sidecar.  Two readers of one measurement can
#: drift, and this pair drifted in the dangerous direction -- a format change
#: yields no match, no match reads as "not measured", and the ledger silently
#: falls back to the weight-tag census that A1-2 exists to replace.  See
#: :func:`dormant_images`, which reads the sidecar through
#: :func:`host_ledger.read_measured_record`, the same call the ledger's own
#: provenance line uses.  The LOG LINE still prints; nothing parses it.
#: The RING-era instrument (C16); same shape, honest name.
#:
#: FIX 1 (round 1), carried review finding 1 reproduced in the replacement
#: instrument: the line must state the POPULATION it samples, because the
#: reader's whole question is whether the census covers every backed-up tag
#: (A1-2's second route to the dormant image) or the weights family alone (a
#: LOWER BOUND -- boot weg2dk7 measured 38.63 GiB against 28.83 of weight
#: tags).  ``population=`` is optional in the regex ON PURPOSE: a boot written
#: before this field existed carries none, and an absent claim is read as the
#: weaker one, never as the stronger.
#: ``rank`` MAY BE NEGATIVE, and the regex has to say so (FIX 3 round 3, boot
#: weg2rg5).  ``rank=(\d+)`` cannot match a minus sign, while C9's gathered legs
#: make ``weight_updater._weg2_rank()`` return ``-1`` -- the front no longer sees
#: a per-rank tag edge, so the emitter writes ``group=? rank=-1`` and names the
#: CARD instead.  The instrument built to make MEASURED possible was therefore
#: the instrument that made every ring-era boot INELIGIBLE: max_tag parsed to
#: ``{0,0,0}``, ``CardRing.complete`` is false, the boot was skipped, and weg2rg5
#: solved its table from the pre-ring weg2sc3 and printed BOUND while a ring-era
#: measurement sat unread in the evidence dir.  A parser that silently rejects
#: the emitter's current format is the Klasse-A instrument lie: the text claims
#: to read the line it cannot read.  Two halves, both needed -- the regex accepts
#: the sign, and :func:`parse_group_log` REFUSES BY NAME (W37) when a log's tag
#: lines exist and none of them parse, so the next divergence is loud.
_TAG_RE = re.compile(
    r"WEG2-FLIP-TAG\s+group=\S+\s+rank=(-?\d+)\s+card=(\S+)\s+dir=\S+\s+tag=(\S+)\s+"
    r"bytes=(\d+)\s+MiB(?:\s+population=(\S+))?"
)
#: Where the synthetic per-CARD indices start for tag lines that carry no rank.
#: Far above any real rank index so a gathered-leg record can never be mistaken
#: for rank 0's, and the card it names still resolves through ``uuid_by_rank``.
_CARD_RANK_BASE = 1000
#: The one token on a WEG2-FLIP-TAG line that licenses ``covers_all_backed_up_tags``.
TAG_POPULATION_ALL = "all-backed-up-tags"
TAG_POPULATION_WEIGHTS = "weights-family"
#: ``WEG2-HOST-LEDGER CHOSEN S=48 GB ... M=2400 MiB ...`` -- the SOURCE boot's
#: own arm.  It is what makes the non-backup host terms of that boot's RssShmem
#: computable (FIX 1, finding 3): anchors and rings are pure functions of
#: ``(S, M)`` in :mod:`host_ledger`, and the boot whose RssShmem is being read
#: is the boot whose arm must be subtracted from it.
_CHOSEN_ARM_RE = re.compile(
    r"WEG2-HOST-LEDGER CHOSEN\s+S=(\d+)\s+GB\b.*?\bM=(\d+)\s+MiB"
)
#: ``KV Cache is allocated. ... K size: 5.63 GB, V size: 5.63 GB``
_KV_RE = re.compile(
    r"\b(?:TP|PP)(\d+)\]\s+KV Cache is allocated\..*?K size:\s*([\d.]+)\s*GB,\s*"
    r"V size:\s*([\d.]+)\s*GB"
)
#: ``WEG2-CORRIDOR phase=D(awake) ... nvml0:free=2474MiB nvml1:free=1987MiB ...``
_CORRIDOR_PHASE_RE = re.compile(r"WEG2-CORRIDOR\s+phase=([A-Z])\(awake\)")
_CORRIDOR_FREE_RE = re.compile(r"nvml(\d+):free=(\d+)MiB")
#: ``instrument=nvml_v2_free,allocatable`` -- present only on boots taken after
#: the corridor instrument fix (front.CORRIDOR_INSTRUMENT).
_CORRIDOR_INSTRUMENT_RE = re.compile(r"WEG2-CORRIDOR\s+phase=[A-Z]\(awake\)[^\n]*?instrument=(\S+)")


def _corridor_sample_phase(line: str) -> Optional[str]:
    """The phase letter if this line IS a corridor SAMPLE, else ``None``.

    THE #995 PROSE TRAP, IN THIS MODULE'S OWN OUTPUT (FIX 2, finding 1).  A
    line that merely CONTAINS ``WEG2-CORRIDOR phase=P(awake)`` is not a
    sample: :func:`solve` emits exactly such a sentence into the front log as
    its own skip reason -- ``front carries no WEG2-CORRIDOR phase=P(awake)/
    D(awake) samples`` -- and it is present verbatim in
    ``boot_weg2_weg2rg6_7f88b1c75d_0908_070324.front.log`` at 07:03:46Z, 127
    seconds BEFORE that boot's first real sample.  Read as a sample it did two
    things: it made :func:`front_corridor_instrument` return early and label a
    POST-fix boot pre-fix (a printed claim about an instrument the boot did not
    use), and it made :func:`parse_front_corridor` create an EMPTY phase entry
    that satisfied the ``"P" not in corridor`` guard, turning an intended
    refusal into a silent zero credit for every card.

    What separates the two is not the phase token but the MEASUREMENT: a
    sample carries at least one ``nvmlN:free=<n>MiB`` field, and prose about
    samples carries none.  A line with no number in it can testify to nothing,
    which is why the gate is here and not at either call site.
    """
    m = _CORRIDOR_PHASE_RE.search(line)
    if not m or not _CORRIDOR_FREE_RE.search(line):
        return None
    return m.group(1)
#: What a WEG2-CORRIDOR line without an ``instrument=`` token was measuring:
#: ``total - used`` over nvidia-smi's carve-out-free ``memory.used``, i.e. free
#: PLUS the driver carve-out.  Named, never silently equated with the new one.
CORRIDOR_INSTRUMENT_PRE_FIX = "total_minus_used(carve-out-blind)"

#: The substring by which a corridor instrument token DECLARES ITS UNIT.  Both
#: post-fix tokens carry it (``front.CORRIDOR_INSTRUMENT`` =
#: ``nvml_v2_free,allocatable``, and the v2-less degradation
#: ``nvml_v1_free,allocatable(carve-out-unknown)``): both structs' ``free`` is
#: already allocatable, only the carve-out BESIDE it is unknown in the second.
#:
#: THE UNIT RULE LIVES HERE, THE TOKENS LIVE IN ``front``, and the two are
#: pinned to each other by a test rather than by a copied literal (round-2
#: finding 2 was a copied band).  A token this marker does not recognise is
#: REFUSED, never graded -- an unknown unit is the mixed-unit defect in its
#: least visible form.
CORRIDOR_ALLOCATABLE_MARKER = ",allocatable"

#: The reference rig's driver carve-outs, MiB, KEYED BY CARD UUID.  Measured
#: 2026-09-08 via NVML v2 ``reserved`` with the cards idle (record
#: ``[SECTION 1x]`` table, cross-validated against ``nvidia-smi
#: --query-gpu=memory.free`` on boots rg5 and rg6 to 0-1 MiB).  The carve-out
#: is a per-card driver constant, not load-dependent, which is what makes a
#: recorded value usable at all.
#:
#: KEYED BY UUID, NEVER BY NVML INDEX, and that is the whole point: NVML
#: enumeration is not stable across boots or driver states (the same reason
#: :data:`_ORDINAL_MAP_RE` exists), so a table keyed by index would silently
#: subtract the 5090's 518 MiB from a 3080 the day the order shifts.  A card
#: this table does not name gets NO correction and a REFUSAL by name.
RECORD_CARVE_OUT_MIB: Dict[str, int] = {
    "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7": 425,  # RTX 3080, nvml 0 on rg3/rg6
    "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d": 518,  # RTX 5090, nvml 1 on rg3/rg6
    "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4": 425,  # RTX 3080, nvml 2 on rg3/rg6
}
RECORD_CARVE_OUT_PROVENANCE = (
    "record [SECTION 1x] table, live NVML v2 `reserved` on idle cards 2026-09-08 "
    "(3080 425 / 5090 518 / 3080 425 MiB), keyed by card UUID"
)


def corridor_instrument_is_allocatable(instrument: str) -> bool:
    """True when a log's samples are ALREADY the band's unit (allocatable free).

    False for :data:`CORRIDOR_INSTRUMENT_PRE_FIX` (front units = allocatable
    free PLUS the carve-out) AND for every token this reader does not know --
    ``"unreadable"``, ``"no WEG2-CORRIDOR samples"``, or a token a future
    emitter invents.  The caller must treat False as "convert or refuse",
    never as "grade anyway".
    """
    return bool(instrument) and CORRIDOR_ALLOCATABLE_MARKER in instrument


@dataclass
class CorridorUnit:
    """One log's corridor samples brought into ONE unit -- or a named refusal.

    THE DEFECT THIS CLOSES (round-2 blocker, boots rg3/rg5/rg6).  Two readers
    -- the boot arm's band verdict and the ring's ``credit(S->W)`` -- consumed
    ``parse_front_corridor`` output directly and treated it as allocatable
    free.  On a PRE-FIX boot it is not: it is allocatable free plus that card's
    driver carve-out.  The arm therefore printed ``nvml1: min_free=859MiB
    verdict=IN`` for the 5090 that sat at 341 MiB, 478 MiB BELOW the floor, and
    the ring credited that same card 518 MiB it never had -- which UNDERSTATES
    ``need`` by 518 MiB, so an R5 case that should have been refused arms
    instead.  Both are the optimistic, i.e. unsafe, direction, and both came
    from grading a number in one unit against a rule stated in another.

    ONE CONVERTER, used by both readers.  Not two call-site patches and not a
    second carve-out table: :func:`corridor_allocatable` is the only place in
    this tree where a corridor sample changes unit.
    """

    #: The source log's instrument token, verbatim.
    instrument: str = ""
    #: ``{nvml index: ALLOCATABLE free MiB}`` -- the band's unit.  Empty when
    #: :attr:`reason` is set; a caller that grades this dict is in one unit.
    allocatable: Dict[int, int] = field(default_factory=dict)
    #: ``{nvml index: MiB subtracted}``.  All zero for an already-allocatable
    #: source, and printed rather than folded away: a correction nobody can see
    #: is indistinguishable from no correction.
    correction_mib: Dict[int, int] = field(default_factory=dict)
    #: Where each carve-out came from, named for the line that prints it.
    provenance: str = ""
    #: Non-empty = REFUSED.  Nothing in :attr:`allocatable` may be graded.
    reason: str = ""

    @property
    def ok(self) -> bool:
        return not self.reason

    @property
    def converted(self) -> bool:
        """True when a carve-out was actually subtracted from some card."""
        return any(self.correction_mib.values())

    def correction_line(self) -> str:
        """The one-line correction summary for a provenance/report line."""
        if not self.ok:
            return f"UNCONVERTIBLE ({self.reason})"
        if not self.converted:
            return f"already allocatable free ({self.instrument}), no conversion"
        per_card = ", ".join(
            f"nvml{i}-{self.correction_mib[i]}"
            for i in sorted(self.correction_mib)
            if self.correction_mib[i]
        )
        return (
            f"CONVERTED from {self.instrument} to allocatable free by subtracting "
            f"each card's driver carve-out ({per_card} MiB); {self.provenance}"
        )


def corridor_allocatable(
    samples: Mapping[int, int],
    instrument: str,
    by_nvml: Mapping[int, str],
    carve_out_by_uuid: Optional[Mapping[str, int]] = None,
) -> CorridorUnit:
    """Bring one phase's per-card samples into ALLOCATABLE free, or refuse.

    ``samples`` is ``{nvml index: MiB}`` as :func:`parse_front_corridor`
    returns it, ``instrument`` what :func:`front_corridor_instrument` says that
    same log is in, and ``by_nvml`` the SOURCE boot's own ``nvml -> UUID`` map
    (:func:`parse_card_identity`) -- because the carve-out belongs to a CARD,
    and the only thing an nvml index proves about a card is which boot printed
    it.

    ``carve_out_by_uuid`` is the caller's measured carve-outs, and it WINS over
    the recorded table: the launcher already resolved its cards through the one
    NVML reader, so ``Card.reserved_mib`` is this rig's live registry snapshot
    and needs no second read here.  Where it is absent or zero the recorded
    per-card constants stand in, and where neither names a card the whole phase
    is REFUSED by name -- a missing carve-out may never become a zero, because
    a zero correction IS the defect.
    """
    res = CorridorUnit(instrument=instrument)
    if not samples:
        res.provenance = "no samples to convert"
        return res
    if corridor_instrument_is_allocatable(instrument):
        res.allocatable = {int(i): int(v) for i, v in samples.items()}
        res.correction_mib = {int(i): 0 for i in samples}
        res.provenance = f"{instrument} is already the band's unit"
        return res
    if instrument != CORRIDOR_INSTRUMENT_PRE_FIX:
        res.reason = (
            f"corridor samples carry the instrument token {instrument!r}, which "
            "names neither the allocatable unit the corridor band is stated in "
            f"(a token containing {CORRIDOR_ALLOCATABLE_MARKER!r}) nor the known "
            f"pre-fix unit {CORRIDOR_INSTRUMENT_PRE_FIX!r} -- an unknown unit is "
            "not graded and not credited"
        )
        return res

    supplied = {u: int(m) for u, m in (carve_out_by_uuid or {}).items() if int(m) > 0}
    carve: Dict[int, int] = {}
    missing: List[int] = []
    sources = set()
    for idx in sorted(samples):
        uuid = by_nvml.get(int(idx), "")
        if uuid and uuid in supplied:
            carve[int(idx)] = supplied[uuid]
            sources.add("live NVML v2 `reserved` for that card, matched by UUID")
        elif uuid and uuid in RECORD_CARVE_OUT_MIB:
            carve[int(idx)] = RECORD_CARVE_OUT_MIB[uuid]
            sources.add(RECORD_CARVE_OUT_PROVENANCE)
        else:
            missing.append(int(idx))
    if missing:
        res.reason = (
            "pre-fix corridor samples cannot be converted to allocatable free: "
            + "; ".join(
                f"nvml{i} ("
                + (f"card {by_nvml[i]}" if by_nvml.get(i) else "no card identity in the source log")
                + ") has no measured or recorded driver carve-out"
                for i in missing
            )
            + " -- refused rather than graded in the source's own unit"
        )
        return res
    res.allocatable = {i: int(samples[i]) - carve[i] for i in carve}
    res.correction_mib = carve
    res.provenance = "carve-out source: " + " + ".join(sorted(sources))
    return res
#: ``NVML -> CUDA ordinal map: ordinal 0 = nvml 1 NVIDIA GeForce RTX 5090
#: GPU-31d7ef41-... total 32607 MiB, ordinal 1 = nvml 0 ...`` (launcher.py).
#: The SOURCE boot's own ordinal/nvml -> UUID map, and the reason this module
#: never has to assume that rank ``n`` of the reading boot is the card rank
#: ``n`` of the written boot: NVML enumeration is not stable across boots.
_ORDINAL_MAP_RE = re.compile(
    r"ordinal\s+(\d+)\s*=\s*nvml\s+(\d+)\s+.*?(GPU-[0-9a-fA-F-]+)\s+total"
)


class Weg2RingRefused(RuntimeError):
    """Every named refusal this module raises at the LAUNCH check.

    FIX 2: it exists so the launcher's ``__main__`` can catch the CLASS rather
    than an enumeration of its members.  ``Weg2RingNeedsInterleave`` was not in
    that enumeration, so an explicitly requested ``--ring-form MAP_SHARED`` on
    this rig printed a raw traceback and exited 1 -- a named refusal that
    propagates as a crash is indistinguishable from a crash to any wrapper that
    keys on the exit code, which is exactly what "refusals are NAMED and
    propagate" forbids.  A new refusal added here inherits the handler.
    """


class Weg2RingCreditRefused(Weg2RingRefused):
    """W32: the R5 inequality fails on some card/direction at the launch check."""


class Weg2FlipTagUnparsable(Weg2RingRefused):
    """W37: a source boot's ``WEG2-FLIP-TAG`` lines exist and NONE of them parse.

    FIX 3 round 3, and boot weg2rg5 is why.  ``_TAG_RE`` required ``rank=(\\d+)``
    while the gathered-leg emitter had been writing ``rank=-1`` since C9, so every
    ring-era boot silently produced ``max_tag={0,0,0}``, failed
    :attr:`CardRing.complete`, and was skipped -- and the launcher, finding an
    older boot that did parse, reported a perfectly well-formed table solved from
    a PRE-RING boot and printed BOUND.  Nothing anywhere said "I could not read
    402 lines of the newest boot's own instrument".

    That silence is the defect this refusal removes, and it is the general form:
    a parser whose text claims to read an instrument must FAIL LOUDLY when the
    emitter's format has moved, never degrade to the answer it can still compute.
    Zero parsed lines out of zero is nothing to say; zero out of N is a refusal.
    """


class Weg2RingNeedsInterleave(Weg2RingRefused):
    """W34: the ring was asked to arm under a SERIAL flip form it cannot fund.

    R5's corridor inequality is a statement about two legs CONCURRENTLY in
    flight (spec C9): W's per-tag releases are what fund S's acquires, so the
    walk of ``u`` never has to hold both whole images at once.  Under the serial
    per-tag order the tree actually ships (``front.FLIP_LEG_FORM == "serial"``)
    S must acquire before W's RPC is even sent, so the requirement is the much
    larger ``H(c) >= image_W(c) + max_tag_S(c)``.  A blocking ring sized to R5
    and armed under the serial order does not run slowly -- it wedges the FIRST
    flip and the acquire budget kills the rank (W31 -> group-fatal W4).  Raised
    at LAUNCH, before either group starts.
    """


@dataclass
class CardRing:
    uuid: str
    nvml_index: int
    name: str
    #: The CHARGED host image per group: ``max(weight-tag census, measured
    #: dormant image)`` -- see :attr:`tags_p_mib` / :attr:`dormant_p_mib`.
    image_p_mib: int = 0
    image_d_mib: int = 0
    max_tag_p_mib: int = 0
    max_tag_d_mib: int = 0
    credit_d2p_mib: int = 0
    credit_p2d_mib: int = 0
    # -- the TWO instruments behind image_*, kept apart and both printed ------
    # A1-2.  ``tags_*`` is the PER-CARD census of this boot's own per-tag lines
    # -- every backed-up tag when the source boot carried C16's WEG2-FLIP-TAG,
    # the weights family alone (a LOWER BOUND) when it carried only the older
    # WEG2-CHUNK-BYTES.  ``dormant_*`` is that card's SHARE of the group's
    # measured RssShmem dormant image, apportioned by its share of the census
    # because the measurement itself is a whole-group figure.  Zero means "this
    # boot's log did not carry that instrument", never "measured zero", and
    # ``dormant_bound_*`` says when the number is a bound rather than a reading.
    tags_p_mib: int = 0
    tags_d_mib: int = 0
    dormant_p_mib: int = 0
    dormant_d_mib: int = 0
    dormant_p_bound: bool = False
    dormant_d_bound: bool = False

    @property
    def h_mib(self) -> int:
        return max(self.image_p_mib, self.image_d_mib)

    @property
    def span1_mib(self) -> int:
        return self.image_p_mib

    @property
    def need_d2p_mib(self) -> int:
        # S = D (sleeping), W = P (waking).
        return self.image_p_mib - self.credit_d2p_mib + self.max_tag_d_mib + self.max_tag_p_mib

    @property
    def need_p2d_mib(self) -> int:
        return self.image_d_mib - self.credit_p2d_mib + self.max_tag_p_mib + self.max_tag_d_mib

    @property
    def slack_d2p_mib(self) -> int:
        return self.h_mib - self.need_d2p_mib

    @property
    def slack_p2d_mib(self) -> int:
        return self.h_mib - self.need_p2d_mib

    # -- the SERIAL form's requirement (front.FLIP_LEG_FORM == "serial") -----
    # ``need`` above is the corridor walk of C9's gathered legs, where W's
    # releases fund S's acquires.  With the legs serialised per tag, S acquires
    # while W still holds its WHOLE parked image and no device credit can be
    # spent on the host side: what has to fit is the parked image plus S's
    # first step.  A strictly larger requirement, and the one this tree runs.

    @property
    def need_serial_d2p_mib(self) -> int:
        # S = D (sleeping), W = P (waking, whole image parked on the host).
        return self.image_p_mib + self.max_tag_d_mib

    @property
    def need_serial_p2d_mib(self) -> int:
        return self.image_d_mib + self.max_tag_p_mib

    @property
    def slack_serial_d2p_mib(self) -> int:
        return self.h_mib - self.need_serial_d2p_mib

    @property
    def slack_serial_p2d_mib(self) -> int:
        return self.h_mib - self.need_serial_p2d_mib

    @property
    def complete(self) -> bool:
        return self.image_p_mib > 0 and self.image_d_mib > 0 and self.max_tag_p_mib > 0 and self.max_tag_d_mib > 0


@dataclass
class RingTable:
    boot: str
    instrument: str
    lines_read: int
    cards: List[CardRing] = field(default_factory=list)
    #: The largest SINGLE STEP of the flip, summed over cards: ``max_k Sigma_c
    #: bytes(k, c)`` over the tags the front's interleave actually walks.  This
    #: is the OLD form's in-flight term -- one tag is one RPC and its shards
    #: land on every card at once, so the sum is over cards and the max is over
    #: tags.  ``Sigma_c max_k`` (each card's own largest tag, summed) is a
    #: different and larger quantity: the maxima need not be the same tag.
    max_step_total_mib: int = 0

    @property
    def total_h_bytes(self) -> int:
        return sum(c.h_mib for c in self.cards) * MIB

    @property
    def total_span1_bytes(self) -> int:
        return sum(c.span1_mib for c in self.cards) * MIB

    @property
    def total_tags_mib(self) -> int:
        """``Sigma_c max_g tags_g(c)`` -- the weight-tag census alone."""
        return sum(max(c.tags_p_mib, c.tags_d_mib) for c in self.cards)

    @property
    def total_dormant_mib(self) -> int:
        """``Sigma_c max_g dormant_g(c)`` -- the measured sleeping-group image."""
        return sum(max(c.dormant_p_mib, c.dormant_d_mib) for c in self.cards)

    #: Which groups' dormant image is a MEASUREMENT and which is a BOUND, and
    #: what the source boot's RssShmem cross-check actually said.  Filled by
    #: :func:`solve`; printed verbatim in :meth:`provenance` so no number on
    #: the RING line can be quoted without the instrument that produced it.
    image_source: str = "no WEG2 DORMANT-IMAGE line in the source boot"
    bound_groups: Tuple[str, ...] = ()
    #: Which "free" the SOURCE BOOT's corridor samples were in -- see
    #: :func:`front_corridor_instrument`.  Printed with them, never assumed.
    credit_instrument: str = CORRIDOR_INSTRUMENT_PRE_FIX
    #: What :func:`corridor_allocatable` did to get those samples into the ONE
    #: unit every credit here is in (allocatable free), verbatim from
    #: :meth:`CorridorUnit.correction_line`.  The credits below are ALWAYS
    #: allocatable free; this line says what had to be subtracted to make them
    #: so, per card and from which source, because a correction nobody can see
    #: cannot be checked against the next boot's own ARMED line.
    credit_correction: str = "no corridor samples were converted"

    def provenance(self) -> str:
        bound = (
            f"; BOUND, NOT A MEASUREMENT, for group(s) {'/'.join(self.bound_groups)}"
            if self.bound_groups
            else "; both groups' dormant images are MEASURED"
        )
        return (
            f"boot {self.boot}, {self.lines_read} lines, instrument: {self.instrument} "
            "(every MiB below is that boot's own, none is a constant in this tree); "
            f"Sigma H {self.total_h_bytes // MIB} MiB = per card the LARGER of "
            f"[per-card tag census {self.total_tags_mib} MiB, measured dormant "
            f"image {self.total_dormant_mib or 'unmeasured'} MiB] -- A1-2: the charged image is "
            "the tag census over ALL BACKED-UP tags, cross-checked against the "
            "sleeping group's summed per-rank RssShmem from that boot's own "
            f"WEG2 DORMANT-IMAGE line ({self.image_source}){bound}"
            f"; per-card CREDIT unit: ALLOCATABLE free, source instrument "
            f"{self.credit_instrument} -- {self.credit_correction}"
        )

    def env_map(self) -> str:
        """``TMS_HOST_RING_MAP`` -- ``<uuid>=<bytes>:<span1>,...``

        A PATH-addressed map only.  There is no ``:fd=<n>`` field: an fd number
        is meaningless in the process that has to open the region, because the
        ranks are ``spawn``-started scheduler processes that inherit no
        descriptor from the launcher (``entrypoints/engine.py`` sets
        ``mp.set_start_method("spawn", force=True)``).
        """
        return ",".join(f"{c.uuid}={c.h_mib * MIB}:{c.span1_mib * MIB}" for c in self.cards)

    def format_l6(self) -> List[str]:
        """L6, one line per card, plus the refusal lines (spec section 5)."""
        out = []
        prov = self.provenance()
        for c in self.cards:
            out.append(
                f"WEG2-HOST-LEDGER RING card={c.uuid} nvml{c.nvml_index} {c.name} "
                f"image_D={c.image_d_mib}(tags {c.tags_d_mib}/dormant "
                f"{c.dormant_d_mib or 'unmeasured'}"
                f"{' BOUND' if c.dormant_d_bound else ''}) "
                f"image_P={c.image_p_mib}(tags {c.tags_p_mib}/dormant "
                f"{c.dormant_p_mib or 'unmeasured'}"
                f"{' BOUND' if c.dormant_p_bound else ''}) "
                f"H={c.h_mib} "
                f"span1={c.span1_mib} credit_d2p={c.credit_d2p_mib} "
                f"credit_p2d={c.credit_p2d_mib} need_d2p={c.need_d2p_mib} "
                f"need_p2d={c.need_p2d_mib} slack={c.slack_d2p_mib}/{c.slack_p2d_mib} MiB "
                f"| SERIAL FORM need_d2p={c.need_serial_d2p_mib} "
                f"need_p2d={c.need_serial_p2d_mib} "
                f"slack={c.slack_serial_d2p_mib}/{c.slack_serial_p2d_mib} MiB "
                f"-- provenance: {prov}"
            )
        return out

    def refusals(self) -> List[str]:
        """Every violated R5 case, with its arithmetic.  Empty = the launch check passes."""
        bad = []
        for c in self.cards:
            for direction, need, s_tag, w_tag, image_w, credit in (
                ("d2p", c.need_d2p_mib, "D", "P", c.image_p_mib, c.credit_d2p_mib),
                ("p2d", c.need_p2d_mib, "P", "D", c.image_d_mib, c.credit_p2d_mib),
            ):
                if need > c.h_mib:
                    bad.append(
                        f"RING REFUSED: need {need} > H {c.h_mib} on card {c.uuid} "
                        f"(nvml{c.nvml_index} {c.name}, {direction}): "
                        f"image_{w_tag} {image_w} - credit {credit} + max_tag_{s_tag} "
                        f"{c.max_tag_d_mib if s_tag == 'D' else c.max_tag_p_mib} + max_tag_{w_tag} "
                        f"{c.max_tag_p_mib if w_tag == 'P' else c.max_tag_d_mib} = {need} MiB"
                    )
        return bad

    def serial_refusals(self, only: Optional[Collection[str]] = None) -> List[str]:
        """Every card/direction the SERIAL flip form cannot walk.  Empty = fundable.

        Read this beside :meth:`refusals`: R5's cases can all pass while every
        one of these fails, because they price two different flip forms.  A
        blocking ring armed on the R5 numbers under the serial order wedges at
        the first acquire -- the ARMED line would then certify an inequality for
        a form this tree does not contain.

        ``only`` restricts the check to a set of card UUIDs, and that is FIX 1's
        whole point: serialisation is not a property of the FLIP, it is a
        property of a CARD.  With C9's gathered legs the two RPCs are in flight
        together, but on a card whose measured duplex ratio does not reach R17's
        gate both legs take the SAME per-card PCIe key and that key is held
        across the whole tag loop (weight_updater sleep-D2H / wake-H2D), so on
        THAT card the pair is serialised again and this -- not R5's corridor --
        is the requirement it must meet.  Boot weg2rg2 is the metal proof: nvml0
        measured 1.316 (DUPLEX-NULL), passed R5 on both directions, armed, and
        wedged 120 s into the first flip with W31 Weg2HostRingExhausted at
        free=12 MiB while its P peer sat on the un-split key.  ``None`` checks
        every card, which is the case where the FRONT itself does not gather.
        """
        bad = []
        for c in self.cards:
            if only is not None and c.uuid not in only:
                continue
            for direction, need, image_w, w_tag, s_tag, step in (
                ("d2p", c.need_serial_d2p_mib, c.image_p_mib, "P", "D", c.max_tag_d_mib),
                ("p2d", c.need_serial_p2d_mib, c.image_d_mib, "D", "P", c.max_tag_p_mib),
            ):
                if need > c.h_mib:
                    bad.append(
                        f"RING NEEDS INTERLEAVE: on card {c.uuid} (nvml{c.nvml_index} "
                        f"{c.name}, {direction}) the serial form needs image_{w_tag} "
                        f"{image_w} + max_tag_{s_tag} {step} = {need} MiB but H is "
                        f"{c.h_mib} MiB -- at flip start the host holds {w_tag}'s whole "
                        f"parked image and only {c.h_mib - image_w} MiB is free, so "
                        f"{s_tag}'s first acquire of {step} MiB blocks and no "
                        f"{w_tag}-release can be issued until it returns"
                    )
        return bad


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def _passes(records: Sequence[Tuple[Tuple[str, ...], int]]) -> List[int]:
    """Total host MiB of each complete sleep pass, in order.

    Two record shapes reach this, and conflating them is a measured 2x error:

    * a SINGLE-tag record -- one step of the front's interleave.  These group
      into passes by tag repetition: the sleep loop walks the weights family
      once per flip leg, so the second sighting of a tag is the next leg.
    * a MULTI-tag record -- the launcher's bulk ``/release_memory_occupation``
      with the whole family in one RPC (``sleep_group``).  That single line is
      already a COMPLETE pass; treating its tag tuple as one more member of the
      surrounding pass double-counts the image (measured on boot weg2dk5:
      image_P read 27 720 instead of 13 860 MiB).
    """
    out: List[int] = []
    cur = 0
    seen: set = set()
    for tags, mib in records:
        if len(tags) > 1:
            if seen:
                out.append(cur)
                cur, seen = 0, set()
            out.append(mib)
            continue
        tag = tags[0]
        if tag in seen:
            out.append(cur)
            cur, seen = 0, set()
        cur += mib
        seen.add(tag)
    if seen:
        out.append(cur)
    return out


def _family_roots(tags: Sequence[str]) -> set:
    """The tag names that name a whole FAMILY rather than one step of it.

    ``weights`` is a bulk record exactly as ``['weights_0', ..., 'weights_11']``
    in one line is: it names the family the chunked tags partition, so its delta
    is a whole pass and not a corridor step.  The multi-tag guard in
    :func:`_passes` does not catch it -- the record carries ONE tag name -- and
    counting it as a step read the 5090's D step as 2 856 instead of 1 608 MiB
    on boot weg2zr2, which is the "bulk record counted as a max_tag" shape.
    """
    names = set(tags)
    return {t for t in names if any(o != t and o.startswith(t + "_") for o in names)}


@dataclass
class GroupLog:
    """One group's log, read once.  Per RANK -- the caller resolves the card.

    FIX 2: ``uuid_by_rank`` is the identity the RING-era instrument carries in
    its own line (``WEG2-FLIP-TAG ... card=<uuid>``).  Before FIX 2 that field
    was captured and thrown away and the rank index was used as the card key,
    which is positional identity: an NVML re-enumeration between the source boot
    and the reading boot silently swapped two rows and the L6 line then printed
    a UUID beside another card's numbers with full confidence.
    """

    image: Dict[int, int] = field(default_factory=dict)
    max_tag: Dict[int, int] = field(default_factory=dict)
    kv_mib: Dict[int, int] = field(default_factory=dict)
    uuid_by_rank: Dict[int, str] = field(default_factory=dict)
    lines_read: int = 0
    instrument: str = ""
    tag_totals: Dict[str, int] = field(default_factory=dict)
    #: True when ``instrument`` names EVERY backed-up tag (C16's WEG2-FLIP-TAG),
    #: False when it names the weights family alone and the census is therefore
    #: a lower bound on the dormant image.
    covers_all_backed_up_tags: bool = False


@dataclass
class DormantImage:
    """One group's measured dormant image -- WHOLE GROUP, no card attribution.

    From the front's ``WEG2 DORMANT-IMAGE`` line (the draft-KV branch's fix-8
    sidecar mechanism; this module READS that line, it does not emit a second
    one).  ``extra_mib`` is what the group held beyond the weight-tag census it
    priced itself against -- the +9.80 GiB of boot weg2dk7 -- and it is the term
    that bounds group D until D's own sample exists.
    """

    group: str
    rss_mib: int
    weight_tags_mib: int
    extra_mib: int


def parse_group_log(path: str) -> GroupLog:
    """Everything one group's log says about host bytes, per RANK INDEX.

    The rank index is the index within its group, which is also that boot's CVD
    ordinal.  Turning it into a CARD is the CALLER's job and needs that boot's
    own ordinal map (:func:`parse_card_identity`) -- this function never guesses
    a card, and where the line names the card itself it is carried through in
    ``uuid_by_rank`` and wins.
    """
    per_rank: Dict[int, List[Tuple[Tuple[str, ...], int]]] = {}
    kv_gb: Dict[int, float] = {}
    uuid_by_rank: Dict[int, str] = {}
    card_rank: Dict[str, int] = {}
    tag_lines_seen = 0
    tag_lines_parsed = 0
    first_unparsed = ""
    lines_read = 0
    covers_all = False
    instrument = (
        "WEG2-CHUNK-BYTES sleep host_image_delta (RssShmem delta, weights_* ONLY "
        "-- a LOWER BOUND on the dormant image, and DEAD once the ring lands, R8)"
    )
    with open(path, errors="replace") as f:
        for line in f:
            if "WEG2-FLIP-TAG" in line:
                tag_lines_seen += 1
                m = _TAG_RE.search(line)
                if not m:
                    if not first_unparsed:
                        first_unparsed = line.strip()[:300]
                    continue
                if m:
                    tag_lines_parsed += 1
                    rank = int(m.group(1))
                    card = m.group(2)
                    if rank < 0:
                        # THE CARD IS THE IDENTITY WHEN THE RANK IS NOT (C9).
                        # A gathered leg has no per-rank tag edge, so the emitter
                        # writes rank=-1 and names the card -- which is the
                        # stronger identity anyway (#589: physical GPUs are
                        # resolved by UUID, never by position).  One synthetic
                        # index per distinct card keeps every downstream keyed by
                        # rank, and ``uuid_by_rank`` maps it straight back.
                        rank = card_rank.setdefault(
                            card, _CARD_RANK_BASE + len(card_rank))
                    uuid_by_rank[rank] = card
                    per_rank.setdefault(rank, []).append(((m.group(3),), int(m.group(4))))
                    lines_read += 1
                    # FIX 1 finding 2: the POPULATION is read off the line, never
                    # inferred from the line's existence.  The predecessor set
                    # this True on the first WEG2-FLIP-TAG it saw, while the
                    # emitter looped over ``weights_tags`` alone -- so the very
                    # first ring-era boot would have flipped the flag, called a
                    # weights-only census "the measured dormant image", and sized
                    # H from a lower bound while the provenance line said
                    # MEASURED.  A line that does not state its population states
                    # the weaker claim.
                    population = m.group(5) or TAG_POPULATION_WEIGHTS
                    if population == TAG_POPULATION_ALL:
                        covers_all = True
                        instrument = (
                            "WEG2-FLIP-TAG bytes (tms_tag_bytes over the saver's "
                            "OWN enable_cpu_backup metadata, population="
                            f"{TAG_POPULATION_ALL} -- A1-2's measured dormant "
                            "image per card)"
                        )
                    elif not covers_all:
                        instrument = (
                            "WEG2-FLIP-TAG bytes (tms_tag_bytes, the saver's own "
                            f"accounting, population={population} -- a LOWER "
                            "BOUND on the dormant image: boot weg2dk7 measured "
                            "38.63 GiB against 28.83 of weight tags)"
                        )
                    continue
            if "WEG2-CHUNK-BYTES sleep" in line:
                m = _CHUNK_RE.search(line)
                if m:
                    rank = int(m.group(1))
                    tags = tuple(
                        t.strip().strip("'\"") for t in m.group(2).split(",") if t.strip()
                    )
                    per_rank.setdefault(rank, []).append((tags, int(m.group(3))))
                    lines_read += 1
                continue
            if "KV Cache is allocated" in line:
                m = _KV_RE.search(line)
                if m:
                    rank = int(m.group(1))
                    kv_gb[rank] = kv_gb.get(rank, 0.0) + float(m.group(2)) + float(m.group(3))
    # W37, and it is the half that makes the regex fix a fix rather than a patch
    # (FIX 3 round 3).  Zero parsed lines out of zero says nothing; zero out of N
    # says this parser cannot read the emitter that wrote this log, and the only
    # honest answer is a named stop.  Degrading silently to the numbers it can
    # still compute is what let weg2rg5 size its ring from a PRE-RING boot while
    # 402 lines of its predecessor's own instrument sat unread.
    if tag_lines_seen and not tag_lines_parsed:
        raise Weg2FlipTagUnparsable(
            f"W37 Weg2FlipTagUnparsable {path}: {tag_lines_seen} WEG2-FLIP-TAG "
            f"line(s) and NONE parse against this tree's _TAG_RE.  The emitter's "
            "format and this parser have diverged, so every per-card byte this "
            "boot measured is invisible and the planner would silently fall back "
            "to an older boot -- which is how a ring-era boot came to be sized "
            "from a pre-ring table.  First unreadable line: " + repr(first_unparsed)
        )
    image: Dict[int, int] = {}
    max_tag: Dict[int, int] = {}
    tag_peak: Dict[str, Dict[int, int]] = {}
    for rank, records in per_rank.items():
        # PEAK pass, never the mean (a ring sized to a mean blocks on the
        # larger pass); population = the passes actually logged.
        image[rank] = max(_passes(records), default=0)
        # max_tag is the corridor STEP SIZE, so only single-tag records may feed
        # it: a bulk record's delta is a whole family, not one tag, and would
        # read as a step nothing ever takes.  A family ROOT name is bulk too.
        singles = [(tags[0], v) for tags, v in records if len(tags) == 1]
        roots = _family_roots([t for t, _ in singles])
        steps = [(t, v) for t, v in singles if t not in roots]
        max_tag[rank] = max((v for _, v in steps), default=0)
        for tag, mib in steps:
            slot = tag_peak.setdefault(tag, {})
            if mib > slot.get(rank, 0):
                slot[rank] = mib
    kv_mib = {r: int(round(gb * GB / MIB)) for r, gb in kv_gb.items()}
    # One tag is ONE RPC of the front's interleave and its shards land on every
    # card at once, so the step the host must hold in flight is the sum over
    # ranks of that tag, maximised over tags.
    tag_totals = {tag: sum(per.values()) for tag, per in tag_peak.items()}
    return GroupLog(
        image=image,
        max_tag=max_tag,
        kv_mib=kv_mib,
        uuid_by_rank=uuid_by_rank,
        lines_read=lines_read,
        instrument=instrument,
        tag_totals=tag_totals,
        covers_all_backed_up_tags=covers_all,
    )


def dormant_images(record_path: str) -> Dict[str, DormantImage]:
    """``{group: DormantImage}`` from the LINE'S OWN SIDECAR.

    ONE READER OF THE DORMANT IMAGE (reconciliation 2026-09-08).  This function
    used to be ``parse_dormant_images``: a regex over the previous boot's front
    log, scraping the very ``WEG2 DORMANT-IMAGE`` line that
    :func:`host_ledger.format_dormant_image` had just printed.  That was a
    SECOND reader of a fact the draft-KV line already carries structurally --
    :func:`host_ledger.dormant_image_sample` measures it and
    :func:`host_ledger.append_measured_record` writes it to a JSON sidecar with
    its commit, boot and timestamp -- and the two could disagree the moment the
    log line's format moved, silently, in the optimistic direction (no match ->
    "no measurement" -> charge the census).

    FLIPCOST A1-2 names the sidecar as the carrier ("D's dormant image is a
    BOUND until measured (draft-KV fix 8 sidecar 'WEG2 DORMANT-IMAGE')"), and
    ring fix 1's review item (h) required this module to REUSE that shape rather
    than re-implement it.  So the regex is deleted and both consumers --
    :func:`host_ledger.resolve_image_terms` for the provenance line and this
    module for H(c) -- now read the SAME entries through
    :func:`host_ledger.read_measured_record`.  They cannot disagree.

    The log line stays: it is the human/evidence instrument.  It is simply no
    longer parsed by anything.

    An unreadable or absent sidecar is an ABSENCE -- ``{}`` -- and the caller
    then charges the census and prints that the cross-check was unavailable; it
    is never read as a zero image.
    """
    out: Dict[str, DormantImage] = {}
    for group, rec in host_ledger.read_measured_record(record_path).items():
        rss = rec.get("rss_shmem_gib")
        if rss is None:
            continue
        tags = rec.get("weight_tags_gib") or 0.0
        extra = rec.get("extra_gib")
        if extra is None:
            extra = float(rss) - float(tags)
        out[group] = DormantImage(
            group=group,
            rss_mib=int(round(float(rss) * GIB / MIB)),
            weight_tags_mib=int(round(float(tags) * GIB / MIB)),
            extra_mib=int(round(float(extra) * GIB / MIB)),
        )
    return out


def parse_chosen_arm(front_log: str) -> Optional[Tuple[int, int]]:
    """``(S GB, M MiB)`` of the SOURCE boot's own chosen ledger arm, or None.

    FIX 1 finding 3.  The RssShmem cross-check reads the sleeping group's WHOLE
    shared-memory residency, which includes host bytes the ledger already posts
    by name for that same boot -- the mamba anchors and the HiCache rings.
    Lifting a census to that figure charges those bytes a SECOND time, and the
    arithmetic of that double book is the A1-3 state: Sigma H walks from ~32 to
    ~42 GiB and the ledger W20-refuses at every rung.  The subtrahend is a pure
    function of the arm (:func:`host_ledger.non_backup_host_bytes`), and the
    right arm is the one the boot being READ ran, not the one this boot will
    choose -- so it is parsed from that boot's own front log.
    """
    try:
        with open(front_log, errors="replace") as f:
            for line in f:
                m = _CHOSEN_ARM_RE.search(line)
                if m:
                    return int(m.group(1)), int(m.group(2))
    except OSError:
        return None
    return None


def apportion_dormant(
    group: str,
    measured: Optional[DormantImage],
    census: Dict[str, int],
    census_covers_all_tags: bool,
    *,
    fallback: Optional[DormantImage] = None,
    fallback_census: Optional[Dict[str, int]] = None,
    non_backup_mib: Optional[int] = None,
    fallback_non_backup_mib: Optional[int] = None,
) -> Tuple[Dict[str, int], str, bool]:
    """A1-2's cross-check, turned into per-card MiB -- ``(rows, source, is_bound)``.

    THE ATTRIBUTION PROBLEM, stated rather than papered over.  ``measured`` is
    ONE number for the whole group: the sum of that group's per-rank RssShmem at
    its first sleep.  The ring is per card.  There is no line anywhere that
    attributes those bytes to cards, so the only join the two instruments
    support is each card's SHARE OF THE GROUP'S OWN CENSUS -- the same
    proportions the per-card lines already state, applied to a total measured by
    a different instrument.  That is a derivation, and the returned ``source``
    string says so on the RING line.

    Three outcomes:

    * no measurement -> the census stands alone and ``source`` says the
      cross-check was unavailable.  Not a zero image: an absence.
    * a measurement no larger than the census -> the census stands (it is
      already per card and already covers every backed-up tag when C16 wrote
      it); ``source`` reports both numbers so the agreement is visible.
    * a measurement larger than the census -> every card is scaled by the same
      ratio, so ``sum(rows) == measured.rss_mib`` and no card is charged bytes
      another card's line accounts for.

    ``fallback`` is the D BOUND of A1-2: with no D sample, D is charged its own
    census plus the unattributed excess P showed -- ``tags_D(c) + extra_P(c)``,
    apportioned the same way -- and the third element of the tuple is True so
    the caller prints the word BOUND rather than letting a bound read as a
    reading.
    """
    total_census = sum(int(v) for v in census.values())
    if total_census <= 0:
        return {}, f"group {group}: no per-card census to apportion onto", False

    def scaled(target_mib: int) -> Dict[str, int]:
        return {
            uuid: int(round(int(value) * target_mib / total_census))
            for uuid, value in census.items()
        }

    covers = "all backed-up tags" if census_covers_all_tags else "weights_* only"
    if measured is not None:
        # SUBTRACT BEFORE YOU MAY RAISE (FIX 1 finding 3).  RssShmem is the
        # sleeping group's WHOLE shared-memory residency; the backup image is
        # what is left of it after the host terms the ledger posts by name for
        # the same ranks -- anchors + rings -- are taken out.  Without that
        # subtraction the same bytes are charged twice, once in the ledger's own
        # ``anchors``/``rings`` terms and once in Sigma H.
        if non_backup_mib is None:
            return (
                dict(census),
                (
                    f"group {group}: measured RssShmem {measured.rss_mib} MiB is "
                    f"REPORTED but NOT CHARGED against the per-card census "
                    f"{total_census} MiB ({covers}) -- the source boot's own chosen "
                    "ledger arm could not be read, so the non-backup host terms it "
                    "posts by name (anchors + rings) cannot be subtracted, and an "
                    "un-netted RssShmem would charge those bytes a second time.  "
                    "The census stands; route = per-card tag census"
                ),
                not census_covers_all_tags,
            )
        usable = measured.rss_mib - max(0, int(non_backup_mib))
        if usable <= total_census:
            return (
                dict(census),
                (
                    f"group {group}: measured RssShmem {measured.rss_mib} MiB minus "
                    f"the ledger's own posted non-backup host terms for this "
                    f"group's ranks {int(non_backup_mib)} MiB (anchors + rings) = "
                    f"{usable} MiB, which does NOT exceed the per-card census "
                    f"{total_census} MiB ({covers}), so the census stands and the "
                    "netted cross-check corroborates it; route = per-card tag census"
                ),
                not census_covers_all_tags,
            )
        return (
            scaled(usable),
            (
                f"group {group}: measured RssShmem {measured.rss_mib} MiB minus the "
                f"ledger's own posted non-backup host terms for this group's ranks "
                f"{int(non_backup_mib)} MiB (anchors + rings) = {usable} MiB, which "
                f"EXCEEDS the per-card census {total_census} MiB ({covers}) by "
                f"{usable - total_census} MiB; the excess is apportioned over the "
                "cards by their share of that census, which is the only attribution "
                "the two instruments jointly support (the measurement is a "
                "whole-group figure); route = netted RssShmem"
            ),
            False,
        )
    # The D BOUND borrows P's unattributed excess, so it inherits P's netting
    # problem too: an un-netted excess would charge the ledger's anchors and
    # rings into D's image as well.  No subtrahend for the lending group -> no
    # borrowing, and the census stands as the bound it already is.
    excess = (
        fallback.extra_mib - max(0, int(fallback_non_backup_mib))
        if fallback is not None and fallback_non_backup_mib is not None
        else 0
    )
    if fallback is not None and fallback_census is not None and excess > 0:
        target = total_census + excess
        return (
            scaled(target),
            (
                f"group {group}: NEVER MEASURED -- BOUND from its own census "
                f"{total_census} MiB plus the {excess} MiB group "
                f"{fallback.group} held beyond ITS census after its own "
                f"{int(fallback_non_backup_mib)} MiB of anchors + rings came out "
                "(A1-2: D's first sleep is a flip and is interleaved, so no "
                "un-confounded D sample exists yet); replace with a reading as "
                "soon as one boot logs WEG2 DORMANT-IMAGE group=D"
            ),
            True,
        )
    return (
        {},
        (
            f"group {group}: no WEG2 DORMANT-IMAGE line in this boot's front log, so "
            "the per-card census stands UNCHECKED "
            + (
                f"(it does cover every backed-up tag -- the lines say population="
                f"{TAG_POPULATION_ALL}, which C16 writes from the saver's own "
                "enable_cpu_backup metadata, so it IS A1-2's measured dormant "
                "image; route = per-card tag census)"
                if census_covers_all_tags
                else "(and it is a LOWER BOUND, not a measurement: the lines carry "
                     "no population=" + TAG_POPULATION_ALL + " claim, so weights_* "
                     "only, and boot weg2dk7 measured a real image 34 % above such "
                     "a census; route = per-card tag census, BOUND)"
            )
        ),
        # A1-2: what is charged is a MEASUREMENT only when the census itself
        # covers every backed-up tag.  A weights-only census with no cross-check
        # is a BOUND, and the line must say so or the operator reads a lower
        # bound as the image.
        not census_covers_all_tags,
    )


def parse_card_identity(front_log: str) -> Tuple[Dict[int, str], Dict[int, str]]:
    """``(uuid by CUDA ordinal, uuid by NVML index)`` of the SOURCE boot.

    Read from that boot's own ``NVML -> CUDA ordinal map`` line (launcher.py),
    which is the only place the written boot recorded which physical card each
    of its ordinals and NVML indices was.  Both directions are needed: the group
    logs are keyed by rank (= ordinal) and the front's ``WEG2-CORRIDOR`` samples
    are keyed by NVML index, and NVML enumeration can differ between the boot
    that wrote them and the boot that reads them.  Empty dicts mean the source
    boot recorded no identity at all -- a REFUSAL for the caller, never a licence
    to pair by position.
    """
    by_ordinal: Dict[int, str] = {}
    by_nvml: Dict[int, str] = {}
    try:
        with open(front_log, errors="replace") as f:
            for line in f:
                if "NVML -> CUDA ordinal map" not in line:
                    continue
                for ordinal, nvml, uuid in _ORDINAL_MAP_RE.findall(line):
                    by_ordinal[int(ordinal)] = uuid
                    by_nvml[int(nvml)] = uuid
                if by_ordinal:
                    break
    except OSError:
        return {}, {}
    return by_ordinal, by_nvml


def parse_front_corridor(path: str) -> Dict[str, Dict[int, int]]:
    """``{phase letter: {nvml index: MINIMUM free MiB observed in that phase}}``.

    The minimum, not the first sample: it is the conservative end of the credit
    (a smaller credit only ever makes R5's requirement larger), and the first
    sample of a phase is taken before the awake group has any load on it.

    Only lines that ARE samples create a phase entry (:func:`_corridor_sample_phase`).
    A phase key here is therefore a claim that measurements exist for it, which
    is what :func:`solve`'s ``"P" not in corridor`` guard reads it as.
    """
    out: Dict[str, Dict[int, int]] = {}
    with open(path, errors="replace") as f:
        for line in f:
            if "WEG2-CORRIDOR" not in line:
                continue
            letter = _corridor_sample_phase(line)
            if letter is None:
                continue
            phase = out.setdefault(letter, {})
            for idx, free in _CORRIDOR_FREE_RE.findall(line):
                i, v = int(idx), int(free)
                if i not in phase or v < phase[i]:
                    phase[i] = v
    return out


def front_corridor_instrument(path: str) -> str:
    """Which "free" the WEG2-CORRIDOR samples in this front log actually are.

    The credit below (``credit(S->W) = NVML free while S is awake + released
    kv``) is only as honest as that free, and the unit changed mid-campaign:
    boots up to weg2rg6 printed ``total - used``, which is free PLUS the driver
    carve-out (425 MiB per 3080, 518 per 5090 on the reference rig), so their
    samples over-state every card's free by that much -- the unsafe direction,
    since a larger credit makes R5's host requirement smaller.  Boots after the
    fix print ``instrument=nvml_v2_free,allocatable`` in the line itself.

    THIS FUNCTION IS STILL ONLY THE NAMER (fix 3 changed the CONSUMERS, not
    it): it classifies, it never corrects and never refuses.  What changed is
    that no consumer may now take its answer as an allocatable number -- both
    the ring credit and the boot arm pass it through
    :func:`corridor_allocatable`, which converts a pre-fix sample by
    subtracting that card's carve-out or refuses the log by name.  The pre-fix
    number is still the best (only) sample those boots carry, and the table rg6
    proved on metal is built from one of them; what must not happen is a reader
    grading or crediting it in the source's own unit.

    ONLY A SAMPLE MAY TESTIFY (FIX 2, finding 1).  The classification is made
    from lines that carry a measurement, never from a line that merely mentions
    the marker -- see :func:`_corridor_sample_phase` for the prose line, in the
    rg6 front log, that made this function label a post-fix boot pre-fix and
    :meth:`RingTable.provenance` print a warning about an instrument that boot
    never used.  Answering from the first MATCHING line is kept: it is the
    first line that actually measured something, and a boot does not change
    instrument mid-run.
    """
    try:
        with open(path, errors="replace") as f:
            for line in f:
                if "WEG2-CORRIDOR" not in line:
                    continue
                if _corridor_sample_phase(line) is None:
                    continue
                m = _CORRIDOR_INSTRUMENT_RE.search(line)
                if m:
                    return m.group(1)
                return CORRIDOR_INSTRUMENT_PRE_FIX
    except OSError:
        return "unreadable"
    return "no WEG2-CORRIDOR samples"


#: ``PROBE start mode=card ... uuid=GPU-...`` opens a card's section of the
#: step-0 probe record; ``PROBE granule tag=... ratio=1.316 verdict=DUPLEX-NULL``
#: is the measurement in the FORM C3/C4 actually issue (2 MiB granules), which
#: is why the granule row is read and the one-block row is not.
_PROBE_START_RE = re.compile(r"PROBE start mode=(\S+)\s+.*?uuid=(GPU-[0-9a-fA-F-]+)")
_PROBE_GRANULE_RE = re.compile(r"PROBE granule tag=\S+\s+.*?\bratio=([\d.]+)\s+verdict=(\S+)")


@dataclass
class DuplexTable:
    """Per-card concurrent/serial PCIe ratio, from the step-0 probe's own lines.

    Amendment A1-4: the direction split of C12/C13 carries a PER-CARD ratio and
    the flip's critical path is exactly the card that does not earn it.  A
    global ratio would open the split on the x4-linked 3080, where the metal
    measured 1.316 against R17's 1.5 gate -- the two legs there share the link
    whatever the key says, and a split key would only remove the serialisation
    that keeps them from halving each other.
    """

    path: str
    ratios: Dict[str, float] = field(default_factory=dict)
    verdicts: Dict[str, str] = field(default_factory=dict)

    def format_lines(self, cards: Sequence, gate: float) -> List[str]:
        out = []
        for card in cards:
            ratio = self.ratios.get(card.uuid)
            if ratio is None:
                out.append(
                    f"WEG2-PCIE-DUPLEX card={card.uuid} nvml{card.nvml_index} {card.name} "
                    f"ratio=UNMEASURED split=NO -- this card has no row in {self.path}, "
                    "and an unmeasured card never splits (a split that is not earned "
                    "re-creates the overlap R9 forbids)"
                )
                continue
            worth_it = ratio >= gate
            out.append(
                f"WEG2-PCIE-DUPLEX card={card.uuid} nvml{card.nvml_index} {card.name} "
                f"ratio={ratio:.3f} gate={gate:.2f} verdict={self.verdicts.get(card.uuid, '?')} "
                f"reaches_gate={'YES' if worth_it else 'NO'} -- "
                + (
                    "the split is expected to pay for itself here"
                    if worth_it
                    else "the split is not expected to pay for itself here (A1-4: this "
                         "card is the flip's critical path and gets no benefit from "
                         "it) -- but the ratio is still >1, i.e. concurrency is "
                         "FASTER than serialising even here, and whether the key "
                         "splits is decided by the WEG2-HOST-RING CHECK line, not by "
                         "this one: under gathered legs one key is a deadlock, not a "
                         "slowdown (boot weg2rg2)"
                )
                + f" -- provenance: {self.path}, 2 MiB-granule row (the form C3/C4 issue)"
            )
        return out


def solve_duplex(probe_path: str) -> Tuple[Optional[DuplexTable], str]:
    """The per-card duplex ratio, SOLVED from the step-0 probe's own lines.

    Same law as the ring table (spec section 10.3): not a constant a human
    chose, not an env knob -- a measured file, parsed, with its path printed
    beside every number it produces.  ``(None, reason)`` when the file is
    unreadable or carries no card row; the caller then splits NOTHING, which is
    the conservative direction.
    """
    current: Optional[str] = None
    table = DuplexTable(path=probe_path)
    try:
        with open(probe_path, errors="replace") as f:
            for line in f:
                m = _PROBE_START_RE.search(line)
                if m:
                    current = m.group(2)
                    continue
                if current is None:
                    continue
                g = _PROBE_GRANULE_RE.search(line)
                if g and current not in table.ratios:
                    table.ratios[current] = float(g.group(1))
                    table.verdicts[current] = g.group(2)
    except OSError as exc:
        return None, f"step-0 probe record {probe_path} unreadable: {exc}"
    if not table.ratios:
        return None, (
            f"step-0 probe record {probe_path} carries no "
            "'PROBE granule ... ratio=' row under a 'PROBE start ... uuid=' header"
        )
    return table, f"{len(table.ratios)} card row(s) from {probe_path}"


def _boot_stems(evidence_dir: str) -> List[str]:
    try:
        names = os.listdir(evidence_dir)
    except OSError:
        return []
    stems = {n[: -len(".front.log")] for n in names if n.endswith(".front.log")}
    have = [s for s in stems if f"{s}.P.log" in names and f"{s}.D.log" in names]
    have.sort(key=lambda s: os.path.getmtime(os.path.join(evidence_dir, f"{s}.front.log")), reverse=True)
    return have


def _non_backup_mib(arm: Optional[Tuple[int, int]]) -> Mapping[str, Optional[int]]:
    """``{group: MiB}`` of the source boot's posted anchors + rings, or Nones.

    The import is local because :mod:`host_ledger` imports nothing from here and
    this module is read by the launcher before either is priced; keeping the
    edge one-directional is what stops the two from becoming one circular unit.
    """
    if arm is None:
        return {"P": None, "D": None}
    from sglang.srt.weg2 import host_ledger

    s_gb, m_mib = arm
    return {
        g: int(host_ledger.non_backup_host_bytes(g, s_gb, m_mib) // MIB)
        for g in ("P", "D")
    }


def solve(
    cards: Sequence,
    evidence_dir: str,
    boot_stem: Optional[str] = None,
) -> Tuple[Optional[RingTable], str]:
    """Solve the table from the newest usable boot, or return (None, reason).

    ``cards`` is the launcher's ORDERED card list (ordinal 0 first): rank ``n``
    of either group runs on ``cards[n]``.
    """
    stems = _boot_stems(evidence_dir)
    if not stems:
        return None, f"no boot in {evidence_dir} carries all three of .front/.P/.D log"
    if boot_stem:
        # A pin is a SUBSTRING of a stem, not the stem itself: the operator
        # types the boot TAG (``weg2zr2``) while the files are named
        # ``boot_weg2_weg2zr2_<sha>_<date>_<time>.{front,P,D}.log``.  An
        # unmatched or ambiguous pin returns a REASON so the caller prints R22
        # and refuses by name (A1-3 leaves no feasible form to fall back to);
        # it never escapes as an OSError out of an unconditional open (every
        # refusal on this path is named).
        exact = [s for s in stems if s == boot_stem]
        hits = exact or [s for s in stems if boot_stem in s]
        if len(hits) != 1:
            return None, (
                f"--ring-table-boot {boot_stem!r} matches {len(hits)} of the "
                f"{len(stems)} complete boots in {evidence_dir}"
                + (f" ({', '.join(sorted(hits)[:4])})" if hits else "")
                + " -- a pin must name exactly one"
            )
        stems = hits
    reasons = []
    for stem in stems:
        p_log = os.path.join(evidence_dir, f"{stem}.P.log")
        d_log = os.path.join(evidence_dir, f"{stem}.D.log")
        f_log = os.path.join(evidence_dir, f"{stem}.front.log")
        gp = parse_group_log(p_log)
        gd = parse_group_log(d_log)
        if not gp.image or not gd.image:
            reasons.append(f"{stem}: no sleep-pass lines for {'P' if not gp.image else 'D'}")
            continue
        corridor = parse_front_corridor(f_log)
        if "P" not in corridor or "D" not in corridor:
            reasons.append(f"{stem}: front carries no WEG2-CORRIDOR phase=P(awake)/D(awake) samples")
            continue
        # FIX 2: CARD IDENTITY, never a position.  Rank -> card comes from the
        # line that names the card if the source boot has the ring-era
        # instrument, and otherwise from that boot's OWN ordinal map; the
        # corridor's nvml indices come from the same map.  Neither present means
        # the pairing would have to be assumed, and an assumed pairing is how a
        # re-enumeration between two boots swaps two 3080 rows in silence.
        by_ordinal, by_nvml = parse_card_identity(f_log)
        if not by_nvml:
            reasons.append(
                f"{stem}: no 'NVML -> CUDA ordinal map' line in the front log, so the "
                "WEG2-CORRIDOR samples' nvml indices name no card and rank->card "
                "would have to be assumed positional -- this planner never assumes "
                "card identity (physical GPUs are resolved by UUID)"
            )
            continue

        def rows(group: GroupLog, what: Dict[int, int]) -> Tuple[Dict[str, int], str]:
            """``what`` re-keyed from rank to card UUID, or a reason.

            TWO KEYS CAN NAME ONE CARD (FIX 3 round 3): a ring-era boot carries
            the pre-ring ``WEG2-CHUNK-BYTES`` records under a real rank AND the
            gathered-leg ``WEG2-FLIP-TAG`` census under that card's synthetic
            index.  Merging by MAX rather than by last-write is both
            deterministic and the safe direction for every quantity that reaches
            here: image and max_tag SIZE the ring, and a ring sized to the
            smaller of two readings is the one that hits W31 at the first sleep.
            """
            out: Dict[str, int] = {}
            for rank, value in what.items():
                uuid = group.uuid_by_rank.get(rank) or by_ordinal.get(rank, "")
                if not uuid:
                    return {}, f"rank {rank} names no card"
                out[uuid] = max(out.get(uuid, 0), value)
            return out, ""

        pairs = {}
        why = ""
        for key, group, what in (
            ("image_p", gp, gp.image), ("image_d", gd, gd.image),
            ("maxtag_p", gp, gp.max_tag), ("maxtag_d", gd, gd.max_tag),
            ("kv_p", gp, gp.kv_mib), ("kv_d", gd, gd.kv_mib),
        ):
            pairs[key], why = rows(group, what)
            if why:
                break
        if why:
            reasons.append(f"{stem}: {why} (no ordinal map entry and no WEG2-FLIP-TAG card=)")
            continue
        # A1-2: the group-level cross-check, apportioned onto the cards by their
        # share of that group's own census.  See :func:`apportion_dormant`.
        measured = dormant_images(
            os.path.join(evidence_dir, host_ledger.MEASURED_RECORD_NAME)
        )
        # FIX 1 finding 3: the subtrahend of the RssShmem cross-check is the
        # SOURCE boot's own posted non-backup host terms, computed from the arm
        # that boot chose -- not from the arm this boot is about to choose, and
        # not from a number typed here.  Unreadable -> None -> the cross-check
        # reports but never raises a census (see :func:`apportion_dormant`).
        arm = parse_chosen_arm(f_log)
        non_backup = _non_backup_mib(arm)
        dorm_p, src_p, bound_p = apportion_dormant(
            "P", measured.get("P"), pairs["image_p"], gp.covers_all_backed_up_tags,
            non_backup_mib=non_backup.get("P"),
        )
        dorm_d, src_d, bound_d = apportion_dormant(
            "D", measured.get("D"), pairs["image_d"], gd.covers_all_backed_up_tags,
            fallback=measured.get("P"), fallback_census=pairs["image_d"],
            non_backup_mib=non_backup.get("D"),
            fallback_non_backup_mib=non_backup.get("P"),
        )
        pairs["dorm_p"], pairs["dorm_d"] = dorm_p, dorm_d

        # ONE UNIT FOR ARM AND CREDIT (fix 3).  ``credit(S->W)`` is "NVML free
        # while S is awake + released kv", and the free half is only that if the
        # source boot's samples are ALLOCATABLE free.  Up to boot rg6 they were
        # ``total - used``, i.e. allocatable free PLUS the carve-out, and
        # crediting them raw over-credited every card by 425/518/425 MiB --
        # which UNDERSTATES ``need`` by the same amount, so an R5 case that
        # should have been refused arms instead (the unsafe direction, disclosed
        # by fix 1 and closed here).  The carve-outs come from the caller's own
        # cards where the launcher measured them, and are NEVER assumed by
        # position: an unnamed card refuses this boot rather than crediting it.
        credit_instrument = front_corridor_instrument(f_log)
        measured_carve_out = {
            c.uuid: int(getattr(c, "reserved_mib", 0) or 0) for c in cards
        }
        unit_d = corridor_allocatable(
            corridor["D"], credit_instrument, by_nvml, measured_carve_out
        )
        unit_p = corridor_allocatable(
            corridor["P"], credit_instrument, by_nvml, measured_carve_out
        )
        bad_unit = next((u for u in (unit_d, unit_p) if not u.ok), None)
        if bad_unit is not None:
            reasons.append(f"{stem}: {bad_unit.reason}")
            continue

        table = RingTable(
            boot=stem,
            instrument=gd.instrument or gp.instrument,
            credit_instrument=credit_instrument,
            credit_correction=" | ".join(
                f"phase {tag}: {u.correction_line()}"
                for tag, u in (("D", unit_d), ("P", unit_p))
            ),
            lines_read=gp.lines_read + gd.lines_read,
            image_source=f"P: {src_p} | D: {src_d}",
            bound_groups=tuple(
                g for g, b in (("P", bound_p), ("D", bound_d)) if b
            ),
            max_step_total_mib=max(
                max(gp.tag_totals.values(), default=0),
                max(gd.tag_totals.values(), default=0),
            ),
        )
        # The corridor sample is keyed by the SOURCE boot's nvml index, which is
        # mapped to a card by that boot's own map -- never by this boot's.  The
        # values re-keyed here are the CONVERTED ones: allocatable free, the
        # same unit the corridor band and the arm's verdict are stated in.
        free_d = {by_nvml[i]: v for i, v in unit_d.allocatable.items() if i in by_nvml}
        free_p = {by_nvml[i]: v for i, v in unit_p.allocatable.items() if i in by_nvml}
        for card in cards:
            cr = CardRing(uuid=card.uuid, nvml_index=card.nvml_index, name=card.name)
            cr.tags_p_mib = int(pairs["image_p"].get(card.uuid, 0))
            cr.tags_d_mib = int(pairs["image_d"].get(card.uuid, 0))
            cr.dormant_p_mib = int(pairs["dorm_p"].get(card.uuid, 0))
            cr.dormant_d_mib = int(pairs["dorm_d"].get(card.uuid, 0))
            cr.dormant_p_bound = bound_p
            cr.dormant_d_bound = bound_d
            # A1-2: charge the LARGER of the per-card tag census and that card's
            # apportioned share of the group's MEASURED dormant image.  When the
            # source boot carried C16 the census already names every backed-up
            # tag and the two agree; where it did not, the census is a lower
            # bound and the measurement is what stops the ring being sized ~34 %
            # short (boot weg2dk7).  A ring sized to the smaller number hits W31
            # at the first sleep, and the LAUNCH charge -- the tightest moment,
            # where LOAD_TRANSIENT sits on top -- is short by the same margin.
            cr.image_p_mib = max(cr.tags_p_mib, cr.dormant_p_mib)
            cr.image_d_mib = max(cr.tags_d_mib, cr.dormant_d_mib)
            cr.max_tag_p_mib = int(pairs["maxtag_p"].get(card.uuid, 0))
            cr.max_tag_d_mib = int(pairs["maxtag_d"].get(card.uuid, 0))
            # credit(S->W) = NVML free while S is awake + the kv bytes S releases.
            cr.credit_d2p_mib = int(free_d.get(card.uuid, 0)) + int(pairs["kv_d"].get(card.uuid, 0))
            cr.credit_p2d_mib = int(free_p.get(card.uuid, 0)) + int(pairs["kv_p"].get(card.uuid, 0))
            table.cards.append(cr)
        missing = [c.uuid for c in table.cards if not c.complete]
        if missing:
            reasons.append(f"{stem}: incomplete rows for {missing}")
            continue
        # THE SKIPPED BOOTS ARE PART OF THE ANSWER (FIX 3 round 3, boot weg2rg5).
        # ``stems`` is newest-first, so everything already in ``reasons`` is a
        # boot NEWER than the one being returned -- and the predecessor dropped
        # all of it on success, keeping it only for the case where nothing
        # worked.  That is exactly backwards: when nothing works the operator
        # gets a refusal to read, and when something works they get a table whose
        # choice they cannot check.  weg2rg5 solved from a boot five generations
        # back and its log could not say why.  One line each, through the string
        # the caller already prints; no new mechanism.
        skipped = "".join(
            f"\nWEG2-HOST-RING SKIPPED (newer than the chosen table) {r}"
            for r in reasons
        )
        return table, f"solved from {stem}" + skipped
    return None, "; ".join(reasons[:4]) or "no usable boot"
