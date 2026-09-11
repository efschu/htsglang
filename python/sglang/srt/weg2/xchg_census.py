# SPDX-License-Identifier: Apache-2.0
"""The PRODUCER of the S6 exchange census (#1273 B4c item 1).

``--weg2-weight-source exchange`` refused at launch by name and no card was
touched::

    W71 Weg2XchgResidencyUnarmable: --weg2-weight-source exchange needs a
    per-card census (--weg2-xchg-census <path>) and none was given.

That was not a defect.  :func:`xchg_residency.load_census` states what it
needs -- ``{cards: {<nvml uuid>: {tags: {group: {tag: MiB}},
dormant_proc_used_mib, dormant_source}}, waves: [[tag, ...]], provenance}`` --
and refuses to invent it, because the exchange's VRAM peak is spec section 5's
arithmetic over a MEASURED census and an unchecked peak is the
``cu_mem_create`` OOM that gate exists to prevent.  Nothing in the tree WROTE
that file.  This module writes it, and the only thing it is allowed to do is
re-state measurements a named boot already made.

WHERE EVERY FIELD COMES FROM, and not one of them is a hand number:

``cards[uuid].tags[group][tag]``
    ``ring_table.GroupLog.tag_by_rank`` of the SELECTED ring-table boot's own
    group log -- the ``WEG2-FLIP-TAG ... card=<uuid> tag=<t> bytes=<n> MiB``
    census the saver wrote from its own ``enable_cpu_backup`` metadata --
    re-keyed from rank to card by that boot's OWN identity, exactly as
    ``ring_table.solve`` re-keys ``image`` and ``max_tag``.

``cards[uuid].dormant_proc_used_mib``
    the SLEEPING rank's on-card residue: CUDA context, comm buffers, the
    flashinfer workspace, the graph pool, the static clones.  Source priority,
    and the second is named as the constant it is: (a) the front's own
    ``WEG2-DC group=<g> uuid=<u> measured=<n> MiB`` reading from the selected
    boot, PEAK per (group, card) and then the LARGER of the two groups because
    ``CardCensus`` has one field per card while P and D differ; (b) failing
    that, ``launcher.DC_MEASURED_D_*`` with its boot name.  **Never the spec
    1.6 EXPECTATIONS row** (``DC_EXPECT_*``), which is a prediction that boot
    weg2ls1b2 exceeded by 380/480 MiB.  Neither source for a card REFUSES;
    there is no default.

``waves``
    ``weight_exchange.derive_waves(family, <map>, cards)``, and WHICH map is
    the one decision in this module that is not a reading.  Both arms are
    implemented, both are named in the provenance, and the measurement that
    separates them is below.

    * ``launcher`` (default) -- the per-card ``chunk_tag_cards`` map of the
      PP form, derived through ``weg2_memory_saver.chunk_tag_cards`` (the
      owner's own function) from the REALIZED layer split the reference boot's
      launcher logged, and cross-checked against the per-tag map that same
      line published.  This is the partition ``xchg_residency``'s own TODO
      names as the producer (*"weight_exchange derives this from
      chunk_tag_cards"*) and the one spec section 5's acceptance line expects
      (``waves=3``).
    * ``ranks`` -- ``{}``, "uniform", which is what
      ``weight_exchange_shadow.build_plan`` passes today, with its own reason
      (*"a rank cannot produce a non-empty one -- the map needs the layer
      count of EVERY PP stage and a rank holds only its own"*) and its own
      label, ``wave_map=uniform-assumed``, a STATED DEVIATION carried into
      UNPROVEN.

    MEASURED against boot weg2sn5b's census on this rig's live NVML totals,
    and this is why the default is the launcher's map rather than the ranks':
    the uniform partition is ONE wave, so both groups' whole images are
    resident at once, and on the 5090 that is ``18608 + 17510 + 2x1668 =
    39454 MiB against a 32607 MiB board`` -- **W71 in both directions,
    free_at_peak -6847 MiB, unarmable by 6.7 GiB**.  The launcher's map is
    three waves and arms with 0 refusals and ``wave1_ok 6/6``, worst
    ``free_at_peak 2707 MiB`` (5090, p2d, wave 3) against the 1229 MiB floor.
    So the two arms are not a preference: one of them cannot boot.

    THE HALF THAT IS STILL OPEN, and it is stated in the census file itself
    rather than left to a reader: the priced partition and the partition the
    ranks derive are TWO OBJECTS (round-2 refuter F11's hazard, one layer up),
    and closing that is the publication ``XchgCensus.waves`` already asks for
    -- *"S1/S6 must publish this list to the ranks and have them refuse a
    mismatch"*.  On a SHADOW-inject boot the difference cannot OOM anything:
    the refill is the authority and the exchange lane does not own the flip's
    VRAM schedule.  At S6I AUTHORITATIVE it can, so the publication is a
    precondition of that boot and not of this one.

``provenance``
    the selected boot, WHY it was selected, the tool's SHA, the instrument, and
    every bound carried as a bound.

TWO THINGS THIS MODULE REFUSES THAT ARE EASY TO MISS.

**It does not select a boot.**  The selection is
:func:`ring_table.solve`'s -- the same call ``launcher.prepare_host_ring``
makes -- and the stem that comes back is the stem used.  A producer that
scanned for the newest boot itself would build a census of a boot the launcher
never reads, and the two would differ silently: measured on 2026-09-11, the
newest complete boot in the evidence tree was ``weg2xsn12`` while the
launcher's own selection for this form was ``weg2sn5b``, eleven boots older,
because #1305 item 4 EXCLUDES an xchg-shadow source (its dormant residual
carries the exchange region's host pages).  That exclusion is re-asserted here
on the selected stem, so a census built with the form gate disarmed cannot
quietly stand in for one built with it armed.

**An absent (card, tag) pair is only a zero once the enumeration is PROVEN
complete.**  ``check_partition``'s own words: *"an absent tag is not a
zero-byte tag, and reading it as one under-prices the peak"*.  Under pipeline
parallelism a chunk tag IS genuinely absent from a card -- it is a layer band,
and group P's stages hold disjoint bands -- so the census must write a zero
there or ``check_partition`` refuses a wave tag it has no bytes for.  What
turns the absence into a zero is an identity, not an argument: the card's tag
bytes must sum to that rank's OWN image pass.  Measured on boot weg2sn5b, all
six ranks, exactly: P 4988/8984/17510 and D 7136/7136/18608 MiB.  Where the
identity does not hold the absence might be a gap in the instrument, and this
module refuses rather than filling a zero.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from sglang.srt.weg2 import ring_table
from sglang.srt.weg2.xchg_residency import (
    GROUPS,
    Weg2XchgResidencyUnarmable,
)

#: The front's per-sleep NVML per-process reading, uuid-keyed (``front.py``'s
#: ``WEG2-DC group=%s uuid=%s measured=%d MiB reserve=%s``).  The launcher
#: prints a SECOND, nvml-INDEX-keyed ``WEG2-DC`` line of its own at group P's
#: budget step; this pattern matches only the uuid-keyed one, because an index
#: is not an identity across boots (#589) and the two lines do not even carry
#: the same quantity.
_DC_RE = re.compile(
    r"WEG2-DC\s+group=([A-Z])\s+uuid=(GPU-[0-9a-fA-F-]+)\s+measured=(\d+)\s+MiB"
)

#: ``WEG2-WEIGHT-CHUNKS N=8 tags (layers per chunk 8 of 64; family [...])`` --
#: the launcher's own record of the tag family that boot paused.  Read rather
#: than re-derived so the census cannot disagree with the boot it describes.
_CHUNKS_RE = re.compile(r"WEG2-WEIGHT-CHUNKS\s+N=(\d+)\s")

#: ``WEG2-HOST-RING SOURCE solved from <stem> (...)`` -- a reference boot's own
#: record of which stem ITS launcher selected.  The selection oracle.
_SOURCE_RE = re.compile(r"WEG2-HOST-RING SOURCE solved from (\S+)")

#: ``WEG2-FLIP-ORDER MAP group=P (SOLVED cut ... -> REALIZED layer split
#: [39, 13, 12] over 64 layers, 8 layers per chunk, nvml [1, 0, 2] in stage
#: order): {'weights_0': [1], ...}`` -- the launcher's OWN published wave map
#: and the three inputs it derived it from.  Both halves are read: the inputs,
#: so the map is rebuilt through ``chunk_tag_cards`` (its owner) rather than
#: parsed out of prose, and the published map itself, so the rebuild is
#: CHECKED against what that boot actually used.
_ORDER_MAP_RE = re.compile(
    r"WEG2-FLIP-ORDER MAP group=P \(.*?REALIZED layer split \[([0-9,\s]+)\] over "
    r"(\d+) layers, (\d+) layers per chunk, nvml \[([0-9,\s]+)\] in stage order\):\s*"
    r"(\{[^}]*\})"
)

#: The wave map ``build_plan`` hands ``derive_waves`` on every rank today:
#: EMPTY, read as "uniform".  A named constant so the ``ranks`` arm cannot be
#: confused with a missing argument, and so a mutant that swaps the arms has to
#: change a name rather than a literal.
RANK_WAVE_MAP: Dict[str, Tuple[int, ...]] = {}

#: The two wave-map arms.  ``launcher`` is the default because the other one
#: does not fit on this rig's 5090 -- see the module docstring's measurement.
WAVE_MAP_ARMS: Tuple[str, str] = ("launcher", "ranks")


def _refuse(what: str) -> Weg2XchgResidencyUnarmable:
    """Every refusal in this module, under the code its consumer already owns.

    W71's class, deliberately, and no new W-code: the event is the one
    ``load_census`` names -- this boot cannot arm the exchange because the
    census is not trustworthy -- caught one step earlier, at the producer,
    where the reason is still legible.  ``launcher.REFUSALS`` catches
    ``Weg2XchgRefused`` as a CLASS, so this inherits ``cli()``'s named line and
    exit 2 rather than escaping as a traceback.
    """
    return Weg2XchgResidencyUnarmable(f"W71 Weg2XchgResidencyUnarmable: {what}")


@dataclass
class CensusBuild:
    """The census, its provenance, and the audit lines that stand under it."""

    blob: dict
    stem: str
    provenance: str
    lines: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# The dormant residue.
# ---------------------------------------------------------------------------


def dormant_readings(front_log: str) -> Dict[str, Dict[str, int]]:
    """``{group: {uuid: PEAK measured MiB}}`` from the front's own WEG2-DC line.

    The PEAK over the boot's samples, never the last one: the residue is
    sampled at every sleep and a later sample can be smaller (the allocator
    cache had been trimmed), so the last reading would under-state what the
    card must hold.  ``solve`` doubles this term, so under-stating it
    under-prices the peak on BOTH ranks at once.
    """
    out: Dict[str, Dict[str, int]] = {}
    try:
        with open(front_log, errors="replace") as fh:
            for line in fh:
                if "WEG2-DC" not in line:
                    continue
                for group, uuid, mib in _DC_RE.findall(line):
                    per = out.setdefault(group, {})
                    value = int(mib)
                    if value > per.get(uuid, 0):
                        per[uuid] = value
    except OSError as exc:
        raise _refuse(f"front log {front_log!r} unreadable: {exc}")
    return out


def dormant_for_card(card: object, readings: Dict[str, Dict[str, int]], stem: str) -> Tuple[int, str]:
    """``(MiB, source)`` for one card, by the ruling's priority, or refuse.

    The LARGER of the two groups' readings, because :class:`CardCensus` carries
    one dormant field per card while P and D differ per rank (P is the PP group
    and sleeps with its stage's graphs; D is the TP group and sleeps with the
    NEXTN draft's).  Both are printed, so a reader can see which group set the
    number and by how much.
    """
    from sglang.srt.weg2 import launcher

    uuid = str(getattr(card, "uuid", ""))
    name = str(getattr(card, "name", ""))
    per_group = {g: int(readings.get(g, {}).get(uuid, 0)) for g in GROUPS}
    if any(per_group.values()):
        best = max(per_group, key=lambda g: per_group[g])
        detail = ", ".join(f"{g}={per_group[g]}" for g in GROUPS)
        return per_group[best], (
            f"READING: boot {stem} front WEG2-DC (NVML per-process at that "
            f"group's sleep), peak per group {detail} MiB; the LARGER is "
            f"charged (group {best}) because one field states both ranks"
        )
    # (b) the boot-named MEASURED constants.  The model has to be NAMED by one
    # of them: the launcher's own `"5090" in c.name else <3080>` is a
    # two-board inventory assumption, true of this rig and not a measurement
    # of a third card, so an unrecognised board refuses instead of borrowing
    # the 3080's number.
    if "5090" in name:
        return launcher.DC_MEASURED_D_5090_MIB, (
            "CONSTANT launcher.DC_MEASURED_D_5090_MIB = "
            f"{launcher.DC_MEASURED_D_5090_MIB} MiB, MEASURED on boot weg2ls1b2 "
            "(NVML per-process at group D's first sleep, NEXTN draft + TP decode "
            f"graphs resident); boot {stem} logged no uuid-keyed WEG2-DC reading"
        )
    if "3080" in name:
        return launcher.DC_MEASURED_D_3080_MIB, (
            "CONSTANT launcher.DC_MEASURED_D_3080_MIB = "
            f"{launcher.DC_MEASURED_D_3080_MIB} MiB, MEASURED on boot weg2ls1b2 "
            "(NVML per-process at group D's first sleep, NEXTN draft + TP decode "
            f"graphs resident); boot {stem} logged no uuid-keyed WEG2-DC reading"
        )
    raise _refuse(
        f"card {uuid} ({name!r}) has no dormant-residue source: boot {stem} "
        "logged no uuid-keyed WEG2-DC reading for it, and no boot-named "
        "MEASURED constant covers this board (launcher.DC_MEASURED_D_* names "
        "the 5090 and the 3080 of this rig's inventory only).  There is no "
        "default: solve() counts this term TWICE, so a guessed value moves the "
        "predicted peak by twice its error in the direction that arms."
    )


# ---------------------------------------------------------------------------
# The per-card, per-tag bytes.
# ---------------------------------------------------------------------------


def per_card_tags(
    group: str,
    glog: ring_table.GroupLog,
    by_ordinal: Dict[int, str],
    family: Sequence[str],
    stem: str,
) -> Tuple[Dict[str, Dict[str, int]], List[str]]:
    """``({uuid: {tag: MiB}}, audit lines)`` for one group, or refuse.

    Rank is re-keyed to card by the SOURCE boot's own identity -- the line's
    ``card=`` UUID first, that boot's ordinal map second -- which is
    ``ring_table.solve``'s ``rows`` rule and the reason a re-enumeration
    between the two boots cannot swap two rows here (#589: physical GPUs are
    resolved by UUID, never by position).
    """
    want = list(dict.fromkeys(str(t) for t in family))
    if not want:
        raise _refuse(
            "the tag family is empty, so the census would name no bytes and "
            "derive_waves would return no wave; the family is the launcher's "
            "own weights_family_tags(chunk_count) and must be read, never "
            "defaulted"
        )
    extra = sorted(set(glog.tag_by_rank) - set(want))
    if extra:
        raise _refuse(
            f"group {group} of boot {stem} measured tag(s) {extra} that the tag "
            f"family {want} does not name.  Dropping them would break the very "
            "identity that licenses a zero for an absent (card, tag) pair -- "
            "the card's tag bytes summing to its own image -- so the family "
            "and the instrument must be reconciled, not silently intersected"
        )
    by_rank: Dict[int, Dict[str, int]] = {}
    for tag, per in glog.tag_by_rank.items():
        for rank, mib in per.items():
            by_rank.setdefault(int(rank), {})[str(tag)] = int(mib)
    per_card: Dict[str, List[Tuple[int, int, int, Dict[str, int]]]] = {}
    for rank, tags in sorted(by_rank.items()):
        uuid = glog.uuid_by_rank.get(rank) or by_ordinal.get(rank, "")
        if not uuid:
            raise _refuse(
                f"group {group} of boot {stem}: rank {rank} carries a tag census "
                "but names no card -- neither its own WEG2-FLIP-TAG card= nor "
                "that boot's NVML -> CUDA ordinal map resolves it, and this "
                "producer never pairs a census to a card by position"
            )
        image = int(glog.image.get(rank, 0))
        per_card.setdefault(uuid, []).append((rank, sum(tags.values()), image, tags))
    out: Dict[str, Dict[str, int]] = {}
    lines: List[str] = []
    for uuid, candidates in sorted(per_card.items()):
        complete = [c for c in candidates if c[1] == c[2] and c[2] > 0]
        if not complete:
            shown = "; ".join(
                f"rank {r} sums {total} MiB over {len(t)} tag(s) against image {img} MiB"
                for r, total, img, t in candidates
            )
            raise _refuse(
                f"group {group} card {uuid} of boot {stem}: the tag census does "
                f"not account for the image on any rank of this card ({shown}).  "
                "That identity is the only thing that turns an ABSENT (card, tag) "
                "pair into a ZERO -- check_partition's own words are 'an absent "
                "tag is not a zero-byte tag, and reading it as one under-prices "
                "the peak' -- so without it the zero-fill this census needs "
                "would be exactly that under-pricing"
            )
        # More than one complete rank on one card means two instruments agreed
        # on the image; take the LARGER census, which is the safe direction for
        # a term the peak is summed from.
        rank, total, image, tags = max(complete, key=lambda c: c[1])
        out[uuid] = {t: int(tags.get(t, 0)) for t in want}
        absent = [t for t in want if t not in tags]
        lines.append(
            f"WEG2-XCHG-CENSUS group={group} card={uuid} rank_key={rank} "
            f"tags_mib={total} image_mib={image} accounted=yes "
            f"zero_filled={absent or 'none'} "
            "(a zero is written only because the sum above equals the image)"
        )
    return out, lines


# ---------------------------------------------------------------------------
# The wave map.
# ---------------------------------------------------------------------------


def wave_map_from_front(front_log: str) -> Tuple[Dict[str, Tuple[int, ...]], str]:
    """``({tag: (card ordinal, ...)}, provenance)`` of the PP form, or refuse.

    TWO READS OF ONE LINE, and the second is what makes the first checkable.
    The launcher logs the map it uses AND the three inputs it derived it from,
    so this function rebuilds the map through ``chunk_tag_cards`` -- the
    function that OWNS the derivation -- and then compares the rebuild against
    the published object.  Parsing the prose alone would make this a second
    reader of the map; rebuilding alone would make it a second producer.  Doing
    both makes it neither: one producer, one cross-check.

    THE CARD SPACE IS ORDINALS HERE AND NVML INDICES THERE.  The launcher calls
    ``chunk_tag_cards(..., card_of_stage=[c.nvml_index for c in cards])``, so
    its published map is keyed by NVML index; ``derive_waves`` is called by the
    ranks with ``cards = range(N_CARDS)``, i.e. ORDINALS.  The rebuild
    therefore uses ``chunk_tag_cards``' default ``card_of_stage`` (the stage
    ordinals, which is right because rank *n* of either group runs on
    ``cards[n]``), and the published map is translated into ordinal space
    through the stage order the same line prints.  Only the SET matters to
    ``derive_waves``' coverage test, which is why the mismatch was invisible:
    on this rig the nvml indices happen to be a permutation of {0,1,2}.
    """
    from sglang.srt.managers.weg2_memory_saver import chunk_tag_cards

    try:
        with open(front_log, errors="replace") as fh:
            found = next((m for m in (_ORDER_MAP_RE.search(ln) for ln in fh) if m), None)
    except OSError as exc:
        raise _refuse(f"front log {front_log!r} unreadable: {exc}")
    if found is None:
        raise _refuse(
            f"front log {front_log!r} carries no 'WEG2-FLIP-ORDER MAP ... REALIZED "
            "layer split' line, so the PP form's per-card wave map cannot be "
            "established.  There is no fallback to the uniform map here: on this "
            "rig the uniform partition is ONE wave, both images are resident at "
            "once, and the 5090 is over its board by 6.7 GiB -- a silent fallback "
            "would turn an unarmable schedule into an armed one"
        )
    split = [int(x) for x in found.group(1).split(",") if x.strip()]
    n_layers, per_chunk = int(found.group(2)), int(found.group(3))
    stage_nvml = [int(x) for x in found.group(4).split(",") if x.strip()]
    published_raw = found.group(5)
    if sum(split) != n_layers:
        raise _refuse(
            f"the logged layer split {split} sums to {sum(split)}, not the "
            f"{n_layers} layers of the same line -- the two halves of that "
            "record disagree and neither can be trusted as the form"
        )
    count = -(-n_layers // per_chunk) if per_chunk > 0 else 0
    rebuilt = chunk_tag_cards(split, per_chunk, count)
    # The published map, translated from NVML index to ordinal by the stage
    # order the line itself prints.  ``stage i`` ran on ``nvml stage_nvml[i]``
    # and on card ORDINAL i, so the translation is that list inverted.
    ordinal_of_nvml = {nvml: i for i, nvml in enumerate(stage_nvml)}
    published: Dict[str, Tuple[int, ...]] = {}
    for tag, ids in re.findall(r"'([^']+)':\s*\[([0-9,\s]*)\]", published_raw):
        want = []
        for one in ids.split(","):
            if not one.strip():
                continue
            nvml = int(one)
            if nvml not in ordinal_of_nvml:
                raise _refuse(
                    f"the published wave map names nvml index {nvml} for tag "
                    f"{tag}, which the same line's stage order {stage_nvml} does "
                    "not contain -- the map cannot be placed in ordinal space"
                )
            want.append(ordinal_of_nvml[nvml])
        published[tag] = tuple(sorted(want))
    if published and published != rebuilt:
        raise _refuse(
            "the wave map rebuilt through chunk_tag_cards does not match the map "
            f"the reference boot published: rebuilt {rebuilt}, published (in "
            f"ordinal space) {published}.  One of the two is not this boot's "
            "form, and pricing either would price a schedule that boot did not "
            "run"
        )
    return rebuilt, (
        f"per-card chunk_tag_cards of the PP form, rebuilt through its owner "
        f"from the reference boot's own REALIZED layer split {split} over "
        f"{n_layers} layers at {per_chunk} per chunk, and CHECKED against the "
        f"map that boot published (nvml stage order {stage_nvml} -> ordinals)"
    )


def resolve_wave_map(arm: str, front_log: str) -> Tuple[Dict[str, Tuple[int, ...]], str]:
    """The wave map of one named arm, with the arm's own provenance sentence."""
    if arm == "ranks":
        return RANK_WAVE_MAP, (
            "the UNIFORM wave map, i.e. the empty map weight_exchange_shadow."
            "build_plan passes on every rank today (its own label: "
            "wave_map=uniform-assumed, a STATED DEVIATION carried into "
            "UNPROVEN).  ONE wave, so both groups' whole images are resident at "
            "once; MEASURED on this rig that is 39454 MiB against the 5090's "
            "32607 MiB board, W71 in both directions, unarmable by 6.7 GiB"
        )
    if arm == "launcher":
        mapping, why = wave_map_from_front(front_log)
        return mapping, (
            why
            + ".  THE PRICED PARTITION AND THE PARTITION THE RANKS DERIVE ARE "
            "TWO OBJECTS: build_plan passes the EMPTY map and gets ONE wave, "
            "this census prices the per-card map and gets more.  On a "
            "SHADOW-inject boot that cannot overcommit anything -- the refill "
            "is the authority and the exchange lane does not own the flip's "
            "VRAM schedule -- but at S6I AUTHORITATIVE it can, so publishing "
            "this wave list to the ranks and having them refuse a mismatch "
            "(XchgCensus.waves' own TODO) is a PRECONDITION of that boot"
        )
    raise _refuse(
        f"unknown wave-map arm {arm!r}; the arms are {list(WAVE_MAP_ARMS)} and "
        "neither is a default for the other"
    )


# ---------------------------------------------------------------------------
# The census.
# ---------------------------------------------------------------------------


def census_from_logs(
    cards: Sequence[object],
    evidence_dir: str,
    stem: str,
    *,
    family: Sequence[str],
    n_cards: int,
    selection: str,
    tool_sha: str,
    wave_map: Optional[Dict[str, Tuple[int, ...]]] = None,
    wave_provenance: str = "",
    waves_of: Optional[Callable] = None,
) -> CensusBuild:
    """The census of ONE named boot, with the selection already made.

    Split from :func:`build_census` so the selection (which is
    ``ring_table.solve``'s and needs this boot's own group-P argv) and the
    reading (which needs only three log files) are separately testable -- and
    so the launcher can one day call this half in-process with the stem its own
    ``prepare_host_ring`` already chose.
    """
    from sglang.srt.weg2 import weight_exchange as wx

    front = os.path.join(evidence_dir, f"{stem}.front.log")
    by_ordinal, by_nvml = ring_table.parse_card_identity(front)
    if not by_ordinal:
        raise _refuse(
            f"boot {stem} logged no 'NVML -> CUDA ordinal map' line, so a rank "
            "whose tag line does not name its card would have to be paired by "
            "position -- and a census keyed by position is the silent row swap "
            "an NVML re-enumeration between two boots produces"
        )
    readings = dormant_readings(front)
    entries: Dict[str, dict] = {}
    lines: List[str] = []
    for card in cards:
        uuid = str(getattr(card, "uuid", ""))
        if not uuid:
            raise _refuse(
                f"live card at nvml index {getattr(card, 'nvml_index', '?')} "
                "carries no UUID; the census is keyed by UUID and an empty key "
                "is a collision waiting for a second nameless card"
            )
        mib, source = dormant_for_card(card, readings, stem)
        entries[uuid] = {
            "tags": {},
            "dormant_proc_used_mib": int(mib),
            "dormant_source": source,
        }
    bounds: List[str] = []
    for group in GROUPS:
        glog = ring_table.parse_group_log(os.path.join(evidence_dir, f"{stem}.{group}.log"))
        if not glog.covers_all_backed_up_tags:
            bounds.append(
                f"group {group}: the WEG2-FLIP-TAG lines carry no "
                f"population={ring_table.TAG_POPULATION_ALL} claim, so this "
                "census is the weights-family LOWER BOUND on that group's "
                "dormant image and not a measurement of it (boot weg2dk7 "
                "measured a real image 34 % above such a census)"
            )
        tags, group_lines = per_card_tags(group, glog, by_ordinal, family, stem)
        lines.extend(group_lines)
        missing = sorted(u for u in entries if u not in tags)
        if missing:
            raise _refuse(
                f"group {group} of boot {stem} has no per-tag census for live "
                f"card(s) {missing} -- that boot did not run a rank on them, or "
                "its instrument did not name them.  Writing the census without "
                "them only moves the refusal to solve(), which would then say "
                "'this boot would run a card whose peak nobody computed'"
            )
        for uuid, per_tag in tags.items():
            if uuid in entries:
                entries[uuid]["tags"][group] = per_tag
    used_map = RANK_WAVE_MAP if wave_map is None else wave_map
    # ``waves_of`` is injectable for the SAME reason ``build_plan`` makes it
    # injectable: the partition check below is a RATCHET, and a ratchet whose
    # only producer cannot violate it is a guard with no can-fail proof.
    derive = waves_of or wx.derive_waves
    waves = [list(w) for w in derive(list(family), used_map, tuple(range(int(n_cards))))]
    if not waves or not any(waves):
        raise _refuse(
            f"derive_waves({list(family)}, {used_map or 'uniform'}, {n_cards} "
            "cards) returned no wave, so there is no schedule to price"
        )
    flat = [t for w in waves for t in w]
    if sorted(flat) != sorted(dict.fromkeys(str(t) for t in family)):
        raise _refuse(
            f"the wave partition {waves} is not a partition of the family "
            f"{list(family)} -- the same check build_plan makes before it plans "
            "(stale-wave-map), here before anything is priced"
        )
    lines.append(
        f"WEG2-XCHG-CENSUS waves={len(waves)} map="
        f"{'uniform' if not used_map else 'per-card'} partition={waves}"
    )
    provenance = (
        f"ring-table boot {stem}; selection: {selection}; tool "
        f"weg2/xchg_census.py sha={tool_sha}; tags from that boot's own "
        "WEG2-FLIP-TAG census (tms_tag_bytes over the saver's enable_cpu_backup "
        "metadata), re-keyed rank -> card by that boot's card= UUID and its NVML "
        "-> CUDA ordinal map, never by position; waves from "
        f"weight_exchange.derive_waves over {wave_provenance or 'the UNIFORM wave map'}"
        "; dormant residue per card as stated in its own dormant_source"
        + ("; " + " | ".join(bounds) if bounds else "")
    )
    return CensusBuild(
        blob={"cards": entries, "waves": waves, "provenance": provenance},
        stem=stem,
        provenance=provenance,
        lines=lines,
    )


def build_census(
    cards: Sequence[object],
    evidence_dir: str,
    *,
    boot_stem: Optional[str] = None,
    p_argv: Optional[Sequence[str]] = None,
    family: Sequence[str],
    n_cards: int,
    tool_sha: str = "",
    solver: Optional[Callable] = None,
    selection_oracle: str = "",
    wave_map_arm: str = "launcher",
    wave_map_from: str = "",
) -> CensusBuild:
    """Select the ring table the launcher would select, then census that boot.

    ``solver`` defaults to :func:`ring_table.solve` and exists so a test can
    prove the delegation rather than the coincidence: the stem this function
    uses is the stem the solver returned, and nothing here scans for a boot.

    ``selection_oracle`` is a reference boot's front log.  That boot's launcher
    printed which stem IT solved from (``WEG2-HOST-RING SOURCE solved from``),
    so when the reference boot's group-P form is this boot's form the two
    selections must agree -- and a disagreement is a refusal rather than a
    note, because a census of the wrong boot is not detectable later.
    """
    solve = solver or ring_table.solve
    table, reason = solve(cards, evidence_dir, boot_stem or None, p_argv=p_argv)
    if table is None:
        raise _refuse(
            "the launcher's own ring-table selection has no usable boot, so "
            "there is no measured census to build one from: "
            f"{reason or '<no reason given>'}"
        )
    stem = str(getattr(table, "boot", ""))
    if not stem:
        raise _refuse("the ring table names no source boot")
    front = os.path.join(evidence_dir, f"{stem}.front.log")
    try:
        with open(front, errors="replace") as fh:
            marked = any(ring_table.XCHG_FORM_MARKER in line for line in fh)
    except OSError as exc:
        raise _refuse(f"front log of the selected boot {stem} is unreadable: {exc}")
    if marked:
        raise _refuse(
            f"the selected boot {stem} is an xchg shadow boot "
            f"({ring_table.XCHG_FORM_MARKER} in its front log).  #1305 item 4 "
            "EXCLUDES such a source from the launcher's own selection -- its "
            "dormant residual carries the exchange region's host pages, which "
            "the boot being armed has not allocated yet -- so a census built on "
            "it is provably not a census of the launcher's table.  Landing here "
            "means the form gate was disarmed (no p_argv), not that the ruling "
            "changed"
        )
    selection = f"ring_table.solve({'pinned ' + boot_stem if boot_stem else 'unpinned'})"
    if reason:
        selection += f"; solver note: {reason[:400]}"
    if selection_oracle:
        chose = ""
        try:
            with open(selection_oracle, errors="replace") as fh:
                for line in fh:
                    found = _SOURCE_RE.search(line)
                    if found:
                        chose = found.group(1)
                        break
        except OSError as exc:
            raise _refuse(f"selection oracle {selection_oracle!r} unreadable: {exc}")
        if not chose:
            raise _refuse(
                f"selection oracle {selection_oracle!r} carries no "
                "'WEG2-HOST-RING SOURCE solved from' line, so it cannot say "
                "which stem a launcher of this form selected"
            )
        if chose != stem:
            raise _refuse(
                f"the selection disagrees with the reference boot's own: this "
                f"producer resolved {stem}, the launcher of "
                f"{os.path.basename(selection_oracle)} solved from {chose}.  A "
                "census of the wrong boot cannot be detected downstream -- "
                "solve() joins by UUID and would price it without complaint"
            )
        selection += f"; oracle {os.path.basename(selection_oracle)} agrees ({chose})"
    # THE WAVE MAP IS THE FORM'S, NOT THE SOURCE BOOT'S.  It describes the PP
    # cut the boot being ARMED will run, so it is read from the reference log of
    # THIS form (``--form-from``) and never from the ring-table stem, which is
    # an older boot chosen for its BYTES.
    mapping, wave_why = resolve_wave_map(wave_map_arm, wave_map_from or selection_oracle)
    return census_from_logs(
        cards, evidence_dir, stem,
        family=family, n_cards=n_cards, selection=selection, tool_sha=tool_sha,
        wave_map=mapping, wave_provenance=wave_why,
    )


def write_census(build: CensusBuild, path: str) -> str:
    """The census file, pretty and sorted so two runs diff cleanly."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(build.blob, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


# ---------------------------------------------------------------------------
# The desk CLI.
# ---------------------------------------------------------------------------


def family_from_front(front_log: str) -> List[str]:
    """The tag family a named boot actually paused, from its own line.

    ``WEG2-WEIGHT-CHUNKS N=<n>`` is the launcher's record of
    ``weights_family_tags(chunk_count)``; the family is rebuilt through that
    same producer from the N it logged, so the census cannot name a family the
    boot did not run.
    """
    from sglang.srt.managers.weg2_memory_saver import weights_family_tags

    try:
        with open(front_log, errors="replace") as fh:
            for line in fh:
                found = _CHUNKS_RE.search(line)
                if found:
                    return [str(t) for t in weights_family_tags(int(found.group(1)))]
    except OSError as exc:
        raise _refuse(f"front log {front_log!r} unreadable: {exc}")
    raise _refuse(
        f"front log {front_log!r} carries no 'WEG2-WEIGHT-CHUNKS N=' line, so "
        "the tag family that boot paused cannot be established; pass "
        "--weight-chunks to state it explicitly"
    )


def p_argv_from_front(front_log: str) -> List[str]:
    """A reference boot's group-P argv, as the LAUNCHER would hand it to solve.

    ``ring_table.parse_p_form`` appends the synthetic
    :data:`ring_table.XCHG_FORM_TOKEN` when the log carries the xchg arm's
    marker, because that is the right reading of a SOURCE.  Here the argv
    stands in for THIS boot's own form, and the launcher never adds that token
    to its own (``ring_table``'s note at that name: *"THIS tree's launcher
    never arms xchg, so THIS boot never carries the token"*).  Stripping it is
    therefore mirroring the launcher, not editing the reference -- and the
    consequence is the one the same note predicts: every xchg-marked source
    stays excluded.  That the launcher SHOULD add the token now that the xchg
    slice is on the line is a real open finding, and it is reported rather than
    changed here: adding it would move every S6 boot's ring-table selection.
    """
    argv, why = ring_table.parse_p_form(front_log)
    if argv is None:
        raise _refuse(f"reference front log {front_log!r}: {why}")
    return [a for a in argv if a != ring_table.XCHG_FORM_TOKEN]


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import subprocess

    from sglang.srt.weg2 import launcher, xchg_residency

    ap = argparse.ArgumentParser(
        prog="python -m sglang.srt.weg2.xchg_census",
        description="Write the --weg2-xchg-census file from the ring table's own boot.",
    )
    ap.add_argument("--evidence-dir", default=launcher.EVIDENCE_DIR)
    ap.add_argument("--ring-table-boot", default="",
                    help="the launcher's own pin; a pin wins outright in both, "
                         "so the two selections are identical by construction")
    ap.add_argument("--form-from", default="",
                    help="a reference boot's front log: its group-P argv arms "
                         "the form gate and its own SOURCE line is the "
                         "selection oracle")
    ap.add_argument("--weight-chunks", type=int, default=0,
                    help="override the family size; default reads the "
                         "reference boot's WEG2-WEIGHT-CHUNKS line")
    ap.add_argument("--wave-map", choices=list(WAVE_MAP_ARMS), default="launcher",
                    help="which partition to price: 'launcher' = the PP form's "
                         "per-card chunk_tag_cards map (spec section 5's waves=3); "
                         "'ranks' = the empty map build_plan passes today, ONE "
                         "wave, unarmable on this rig's 5090 by 6.7 GiB")
    ap.add_argument("--out", required=True)
    ap.add_argument("--floor-mib", type=float, default=launcher.ARMING_FLOOR_MIB)
    ns = ap.parse_args(list(argv) if argv is not None else None)

    cards = launcher.order_cards(launcher.resolve_cards())
    if ns.weight_chunks > 0:
        from sglang.srt.managers.weg2_memory_saver import weights_family_tags

        family = [str(t) for t in weights_family_tags(ns.weight_chunks)]
    elif ns.form_from:
        family = family_from_front(ns.form_from)
    else:
        raise _refuse("pass --form-from or --weight-chunks: the tag family is read, never guessed")
    sha = ""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()[:10]
    except (OSError, subprocess.SubprocessError):
        sha = "unknown"
    build = build_census(
        cards, ns.evidence_dir,
        boot_stem=ns.ring_table_boot or None,
        p_argv=p_argv_from_front(ns.form_from) if ns.form_from else None,
        family=family, n_cards=len(cards), tool_sha=sha,
        selection_oracle=ns.form_from,
        wave_map_arm=ns.wave_map, wave_map_from=ns.form_from,
    )
    write_census(build, ns.out)
    for line in build.lines:
        print(line)
    print(f"WEG2-XCHG-CENSUS written {ns.out}")
    print(f"WEG2-XCHG-CENSUS provenance {build.provenance}")
    census = xchg_residency.load_census(ns.out)
    res = xchg_residency.solve(cards, census, ns.floor_mib)
    for line in res.lines:
        print(line)
    for bad in res.refusals:
        print(f"W71 REFUSAL {bad}")
    print(
        f"WEG2-XCHG-CENSUS armed={res.armed} waves={res.waves} "
        f"wave1_ok={'/'.join(str(x) for x in res.wave1_ok())} "
        f"refusals={len(res.refusals)}"
    )
    return 0 if res.armed else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
