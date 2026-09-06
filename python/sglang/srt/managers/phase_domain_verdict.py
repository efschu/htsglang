"""#1068 weg1 S0: ONE CARRIER for every rank-divergent fact at the #1203 seam.

WHY A MODULE AND NOT FOUR INLINE APPENDS. The tree already argues the case for
its own reduce, at ``scheduler.py:6840-6841``: "two reduces would be two chances
for the counts to diverge, which is the ...". The same argument applies to N
independent packings sharing ONE reduce -- four hand-rolled head indices are
four chances to get the layout wrong, and a layout defect on a MIN reduce is
silent in the one direction that matters: a slot read at the wrong index reads
as agreement.

TWO BUSES LIVE HERE, AND THEY ARE NOT ONE BUS.

* THE PACKED BUS rides ``_update_uniform_pool_budget``'s existing per-pass MIN
  ``all_reduce`` (``scheduler.py:7182-7183``) on ``tp_cpu_group``. That group is
  world size 1 at boot and through every PP phase -- ``_update_uniform_pool_budget``
  returns at ``scheduler.py:6978`` under the guard at ``:6916-6917``, and
  ``phase_flip_runtime.py:4052`` (``want_tp_size = n if tp_phase else 1``) is what
  makes it a singleton -- so EVERY verdict on this bus is read in the TP phase
  only. A PP-phase detection is not lost, it is DEFERRED to the first packed
  reduce of the next TP phase: the bound is ONE CUTOVER.
* THE BOOT BUS rides a single MIN ``all_reduce`` on
  ``scheduler.world_group.cpu_group``, taken once, in ``Scheduler.__init__``
  immediately after the host-pool build call. It exists because the packed bus's
  group is a SINGLETON at boot, and a MIN over one member is a rank-local raise
  wearing a collective's clothes.

  The two buses share this module and nothing else. Folding them would put every
  boot verdict on a singleton, which is the defect the separation exists to
  prevent.

WHAT LIVES HERE AND WHAT DOES NOT. This module DECLARES both layouts, packs
them and reads them back. It decides nothing of its own and takes no collective.
Every VALUE on either bus is produced by another slice at its own site; a later
slice supplies its term's VALUE and its unpack consumer, and it never appends a
slot. The width of each bus is DERIVED from its declared layout and is never
typed as a literal: the census offset has already moved four times, and a
hard-coded offset silently reads a count pair as a per-rank flag.

POLARITY, AND IT IS A ONE-CHARACTER DEFECT IN THE DANGER DIRECTION. The reduce
is MIN, so on an AND-slot 1-IS-GOOD makes ``group_min == 0`` mean "at least one
rank is bad" on every rank of the reduce. Every AND term is therefore stated in
the POSITIVE (``..._matches``, ``..._complete``) while the refusal it guards is
named in the NEGATIVE (``MISMATCH``, ``INCOMPLETE``). This is the polarity
``prefetch_ballot.py:71-73`` states for its own sentinel:

     71	# Sentinel verdict for a slot with no rid behind it (queue shorter than the
     72	# slot count). 1 is the MIN-neutral element for AND, so short queues never
     73	# veto anything.

A term whose producer has not landed yet reads its NEUTRAL, never the falsy
default: an AND-slot reads ``1`` (packed ``(1, -1)``), a pair reads ``(0, 0)``,
a census slot reads ``1``. Packing an unproduced AND-slot with ``0`` makes every
rank raise from the first scheduler iteration and kills every boot until that
producer's batch.

AN AND TERM RIDES AS A PAIR TOO, ``(v, -v)``. A one-slot AND term gives the
unpack a minimum and nothing else, so the STOP line printed the minimum under
both names and could not tell a unanimous refusal from a partial one -- which
is the whole question on a death path, because a partial refusal names
something rank-local as the cause. The negated half is exact at any world size;
the per-rank census below is the finer statement and is bounded by its width.

WHAT A COUNT PAIR DOES AND DOES NOT CATCH -- A NAMED LIMIT, NOT AN OMISSION.
The ``(x, -x)`` pair makes ``group_min`` and ``group_max`` visible on every rank,
so a DIVERGENCE-consumed pair stops the group on ``min != max``. A count that is
nonzero and IDENTICAL on every rank is not a divergence and does NOT stop the
group; it is carried by the rank-local ``logger.error`` line at the detecting
site. The two MAX-consumed pairs are the exception and say so in their own row:
a host->device geometry mismatch and a host backup-width mismatch are
wrong-answer conditions on ANY rank, so a uniform nonzero must stop the group
too, and their predicate is ``group_max > 0``.

``hash()`` is process-salted (PYTHONHASHSEED) and MUST NOT touch either payload;
the digest is ``zlib.crc32``, deterministic across processes.
"""

from __future__ import annotations

import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Term kinds. The KIND decides how a term is packed AND how it is consumed;
#: a builder who copies a divergence pair's consumer onto a MAX pair ships a
#: guard that is silent on the case it exists for.
AND_SLOT = "and"
DIVERGENCE_PAIR = "divergence_pair"
MAX_PAIR = "max_pair"
CENSUS_BLOCK = "census"

#: Per-rank census width. A MODULE CONSTANT, never the runtime world size, for
#: the property `build_prefetch_ballot_payload` states at
#: `prefetch_ballot.py:95-96`: "fixed width is the point: every rank appends the
#: same number of / elements whatever its queue looks like."
#:
#: 8 against this rig's TP=3 -- headroom without widening the reduce
#: meaningfully. BOUNDED LIMIT, PRINTED RATHER THAN HIDDEN: a rank whose index
#: is >= this width owns no slot, and the unpack renders those positions `?`
#: from the WORLD SIZE its caller supplies -- never from the value `1`, which a
#: healthy rank also writes. Nothing about the census is ever a STOP condition.
PHASE_DOMAIN_CENSUS_SLOTS = 8


class _Term:
    """One voted term of the packed bus. ONE ENTRY PER VOTED TERM."""

    __slots__ = ("name", "kind", "width", "neutral", "refusal", "terse_stop", "of")

    def __init__(
        self, name, kind, refusal, *, width=None, neutral=None, terse_stop=False, of=None
    ):
        self.name = name
        self.kind = kind
        self.refusal = refusal
        #: For a CENSUS_BLOCK: the AND term whose per-rank votes it carries. A
        #: census that names no term is a row any STOP line can print as its
        #: own, which is how a slot-15 death printed the loadback census under
        #: slot 15's name and contradicted the votes beside it.
        self.of = of
        if kind == AND_SLOT:
            # TWO SLOTS, `(v, -v)`, exactly as a pair. A 1-wide AND slot on a
            # MIN reduce carries no MAXIMUM, so the unpack had nothing to print
            # beside `group_min` and printed the minimum twice -- a line that
            # reads as a unanimous refusal whether the refusal was unanimous or
            # 2-of-3. The negated half is what makes "at least one rank was
            # healthy" visible on EVERY rank of the reduce, and it is exact at
            # any world size, unlike the census, which is bounded.
            self.width = 2
            self.neutral = 1
        elif kind in (DIVERGENCE_PAIR, MAX_PAIR):
            self.width = 2
            self.neutral = 0
        elif kind == CENSUS_BLOCK:
            self.width = width
            self.neutral = neutral
        else:  # pragma: no cover - a typo in the table, not a runtime state
            raise ValueError(f"unknown term kind {kind!r} for {name!r}")
        self.terse_stop = terse_stop


#: THE PACKED-BUS LAYOUT. Twenty-six scalar slots plus TWO census blocks, in
#: order. A term added later adds a ROW here and re-derives the constant below;
#: it never appends silently.
#:
#: EVERY AND TERM IS TWO SLOTS, `(v, -v)`. The scalar count was 21 while an AND
#: slot was one slot wide; the MAX half added here is what lets a STOP line say
#: whether the refusal was unanimous (see `_Term.__init__`).
#:
#: A CENSUS BLOCK NAMES THE TERM IT CENSUSES. There is one per AND term whose
#: STOP line has to say WHICH rank refused, and only those: a census is eight
#: slots, and the other three AND terms have no measured need for one, so they
#: render `per_rank=[none]` rather than borrowing a row that is not theirs.
#:
#: | slot(s) | term | produced by | the refusal it carries |
#: | 0-1     | loader_covers_own_layers    | S2-C3      | #1206 LOADER COVERAGE REFUSED |
#: | 2-3     | d_domain                    | S2-C3      | #1206 TRANSFER DOMAIN DIVERGENT |
#: | 4-5     | d_host_prov                 | S4-C5      | [#928 anchor/host] REFUSING resume |
#: | 6-7     | d_ownership                 | S5-C4c     | #924 MAMBA OWNERSHIP SPLIT AT DONATION |
#: | 8-9     | d_state_src                 | S5-C6      | #924 STATE SOURCE CONTRADICTION |
#: | 10-11   | loadback_coverage_complete  | S2-C2      | #1206 LOADBACK COVERAGE INCOMPLETE |
#: | 12-13   | draft_tier_domain_matches   | NO PRODUCER| #1206 DRAFT TIER DOMAIN MISMATCH |
#: | 14-15   | d_host_unresolvable         | S3-C9      | #1206 HOST POOL UNRESOLVABLE |
#: | 16-17   | d_slot_ownership            | S5-C4b/d   | #924 SLOT OWNERSHIP REFUSED |
#: | 18-19   | rebind_domain_within_driven | S1-C16     | #1206 REBIND DOMAIN OUTSIDE DRIVEN |
#: | 20-21   | d_geom                      | S4-C6      | the host->device geometry condition |
#: | 22-23   | host_ring_discarded         | S7-C10     | #1206 HOST RING NOT DISCARDED |
#: | 24-25   | d_backup_width              | S7 F-E(b)  | the host backup-width condition |
#: | 26..    | census_loadback[0 .. N-1]   | S2-C2      | loadback_coverage_complete's per_rank row |
#: | ..      | census_rebind[0 .. N-1]     | S1-C16     | rebind_domain_within_driven's per_rank row |
#:
#: THE SLOT NUMBERS ABOVE ARE DOCUMENTATION AND HAVE MOVED ONCE ALREADY. Every
#: reader indexes through `index_of(name)`; a consumer that types a number is
#: the defect this comment exists to make visible, not to enable.
PHASE_DOMAIN_LAYOUT: Tuple[_Term, ...] = (
    _Term("loader_covers_own_layers", AND_SLOT, "#1206 LOADER COVERAGE REFUSED"),
    _Term("d_domain", DIVERGENCE_PAIR, "#1206 TRANSFER DOMAIN DIVERGENT"),
    _Term("d_host_prov", DIVERGENCE_PAIR, "[#928 anchor/host] REFUSING resume"),
    _Term("d_ownership", DIVERGENCE_PAIR, "#924 MAMBA OWNERSHIP SPLIT AT DONATION"),
    _Term("d_state_src", DIVERGENCE_PAIR, "#924 STATE SOURCE CONTRADICTION"),
    _Term("loadback_coverage_complete", AND_SLOT, "#1206 LOADBACK COVERAGE INCOMPLETE"),
    _Term("draft_tier_domain_matches", AND_SLOT, "#1206 DRAFT TIER DOMAIN MISMATCH"),
    _Term("d_host_unresolvable", DIVERGENCE_PAIR, "#1206 HOST POOL UNRESOLVABLE"),
    _Term("d_slot_ownership", DIVERGENCE_PAIR, "#924 SLOT OWNERSHIP REFUSED"),
    _Term("rebind_domain_within_driven", AND_SLOT, "#1206 REBIND DOMAIN OUTSIDE DRIVEN"),
    _Term("d_geom", MAX_PAIR, "host-to-device geometry mismatch count"),
    _Term(
        "host_ring_discarded",
        AND_SLOT,
        "#1206 HOST RING NOT DISCARDED",
        terse_stop=True,
    ),
    _Term("d_backup_width", MAX_PAIR, "host backup width mismatch count"),
    _Term(
        "census_loadback",
        CENSUS_BLOCK,
        "the per-rank census (never a STOP condition)",
        width=PHASE_DOMAIN_CENSUS_SLOTS,
        neutral=1,
        of="loadback_coverage_complete",
    ),
    _Term(
        "census_rebind",
        CENSUS_BLOCK,
        "the per-rank census (never a STOP condition)",
        width=PHASE_DOMAIN_CENSUS_SLOTS,
        neutral=1,
        of="rebind_domain_within_driven",
    ),
)

#: The TOTAL width of the packed slice -- DERIVED from the table above and never
#: typed as a literal. `unpack_phase_domain` checks THIS number and does NOT
#: copy `prefetch_ballot.py:149`'s `+ 2`: there PREFETCH_BALLOT_SLOTS counts only
#: the per-rid verdict slots, here the constant IS the total.
PHASE_DOMAIN_SLOTS = sum(term.width for term in PHASE_DOMAIN_LAYOUT)

_PHASE_DOMAIN_INDEX: Dict[str, int] = {}
_at = 0
for _term in PHASE_DOMAIN_LAYOUT:
    _PHASE_DOMAIN_INDEX[_term.name] = _at
    _at += _term.width
del _at, _term

#: The census block belonging to each censused term, DERIVED from the table.
#: A STOP line renders THIS mapping and nothing else, so no term can print a
#: row that was voted for a different question.
_CENSUS_OF: Dict[str, _Term] = {
    _c.of: _c for _c in PHASE_DOMAIN_LAYOUT if _c.kind == CENSUS_BLOCK and _c.of
}


# ---------------------------------------------------------------------------
# THE ROUTE: where each term's rank-local value lives.
#
# Each name below is a READ, never a second record. The count lives once, in the
# object that detects it; the payload copies it onto the wire for one reduce and
# the unpack discards it. A slice that mirrors its count onto the scheduler as a
# standing attribute has created a third record.
#
# The names are the BUS CONTRACT: the detecting slice exposes its rank-local
# count or flag under exactly this name at exactly this home, and no detecting
# slice edits the payload-build site.
# ---------------------------------------------------------------------------

#: S2-C2, on `scheduler.tree_cache.cache_controller`. NEGATIVE sense (the site
#: records that coverage was INCOMPLETE); the payload INVERTS it, so an un-fired
#: detection votes the MIN-neutral 1.
LOADBACK_INCOMPLETE_ATTR = "_loadback_coverage_incomplete"

#: S4-C5, on the MAMBA tree component.
HOST_PROV_REFUSALS_ATTR = "_host_prov_refusals"

#: S3-C9, on every built tree component; SUMMED over them, because the counter
#: is declared on three component classes and a single-component read would
#: leave two of the three a detection with no vote.
HOST_UNRESOLVABLE_ATTR = "_host_unresolvable_count"

#: S5-C4c and S5-C6, on `scheduler.req_to_token_pool` (`HybridReqToTokenPool`).
OWNERSHIP_SPLIT_ATTR = "_ownership_split_count"
STATE_SRC_ATTR = "_state_src_contradictions"

#: S5-C4b/S5-C4d, on `scheduler.req_to_token_pool.mamba_allocator`; all FOUR
#: allocator transitions summed into one counter.
SLOT_OWNERSHIP_ATTR = "_slot_ownership_refusals"

#: S4-C6, on the bound group's MAMBA entry's `host_pool` (`MambaPoolHost`).
GEOM_MISMATCH_ATTR = "_geom_mismatch_count"

#: S1-C18 declares these two on `HostPoolGroup` at B1 with their neutral values;
#: S7 FILLS them at B6. They are read BY NAME and with NO DEFAULT: a
#: getattr-with-default here would turn a missing declaration into a silently
#: healthy vote on the one term that exists to STOP the group.
HOST_RING_DISCARD_OK_ATTR = "host_ring_discard_ok"
D_BACKUP_WIDTH_ATTR = "d_backup_width"

#: S0's OWN bookmark for the monotonic backup-width counter, on the Scheduler.
#: NOT a module global (a process-wide counter shared by two schedulers in one
#: process is a second record of a per-scheduler fact) and NOT a field on the
#: group (a reader that stores its bookmark on the object it reads is a third
#: record). The counter itself is MONOTONIC and is never written from here: its
#: writer is the storage backup THREAD and its reader is the scheduler loop, so
#: a read-then-zero would be a cross-thread read/modify/write and every
#: increment landing in that window would be discarded silently.
D_BACKUP_WIDTH_LAST_SEEN_ATTR = "_d_backup_width_last_seen"


def phase_domain_digest(items: Sequence[Any]) -> int:
    """Deterministic cross-process digest of an ordered item list, as a
    non-negative int64. An empty list digests to 0 on every rank, which agrees
    trivially. Copied from ``prefetch_ballot_digest``: ``hash()`` is
    process-salted and may not touch a payload every rank has to match."""
    items = list(items)
    if not items:
        return 0
    canonical = "|".join(str(item) for item in items)
    return zlib.crc32(canonical.encode("utf-8")) & 0x7FFFFFFF


def index_of(term_name: str) -> int:
    """The head index of a term inside the packed slice, DERIVED from the table.

    Callers index through this and never through a written-out number: the
    census offset has moved four times already, and a hard-coded offset reads a
    count pair as a per-rank flag without saying anything."""
    return _PHASE_DOMAIN_INDEX[term_name]


def slot_of(payload: Sequence[int], term_name: str) -> int:
    """The VOTE half of an AND term, read by name.

    An AND term is packed `(v, -v)` like a pair; this returns `v`, which is
    what a caller reading its own contribution wants. The negated half is for
    the unpack's MAX and is never read through here."""
    return int(payload[index_of(term_name)])


def pair_of(payload: Sequence[int], term_name: str) -> Tuple[int, int]:
    """Both halves of a pair term, read by name, exactly as packed."""
    at = index_of(term_name)
    return int(payload[at]), int(payload[at + 1])


def local_terms_from_payload(payload: Sequence[int]) -> Dict[str, Any]:
    """This rank's OWN values, decoded back out of the slice it packed.

    Not a second record: it is the same list read by name, so the STOP line can
    print what THIS rank contributed beside the group's min and max without the
    caller carrying a second copy of the terms across the reduce.
    """
    terms: Dict[str, Any] = {}
    for term in PHASE_DOMAIN_LAYOUT:
        at = index_of(term.name)
        if term.kind == AND_SLOT:
            terms[term.name] = int(payload[at])
        elif term.kind in (DIVERGENCE_PAIR, MAX_PAIR):
            terms[term.name] = int(payload[at])
        else:
            terms[term.name] = [int(v) for v in payload[at : at + term.width]]
    return terms


def pack_phase_domain_payload(terms: Optional[Dict[str, Any]] = None) -> List[int]:
    """This rank's contribution, packed in declared order.

    ``terms`` maps a term name to this rank's value. A term ABSENT from the
    mapping has no producer yet and contributes its NEUTRAL -- ``(1, -1)`` for
    an AND-slot, ``(0, 0)`` for a pair, 1 for each census slot. Each
    ``(x, -x)`` pair is built exactly as ``prefetch_ballot.py:99-100`` builds
    its digest pair, which is what makes the group's min AND max visible on
    every rank -- AND-slots included, since the round-3 line could not say
    whether a refusal was unanimous.
    """
    terms = {} if terms is None else terms
    payload: List[int] = []
    for term in PHASE_DOMAIN_LAYOUT:
        if term.kind == AND_SLOT:
            vote = 1 if int(terms.get(term.name, term.neutral)) else 0
            payload.append(vote)
            payload.append(-vote)
        elif term.kind in (DIVERGENCE_PAIR, MAX_PAIR):
            value = int(terms.get(term.name, term.neutral))
            payload.append(value)
            payload.append(-value)
        else:
            census = terms.get(term.name)
            if census is None:
                census = [term.neutral] * term.width
            census = list(census)[: term.width]
            census = census + [term.neutral] * (term.width - len(census))
            payload.extend(int(slot) for slot in census)
    return payload


#: Absence marker for a route read. NOT a value: a term whose home does not
#: exist yet cannot be told from one that voted well if the absence is spelled
#: as a healthy DEFAULT, which is the getattr-default-on-a-ledger-path shape
#: (#606) and, on a bus term, the shape that turns a missing declaration into a
#: silently healthy vote. Every read below is therefore ONE getattr against
#: this sentinel plus an explicit branch -- one read, not the two an
#: ``hasattr`` + ``getattr`` pair performs.
_ABSENT = object()


def _read_or_neutral(obj: Any, name: str, neutral: Any) -> Any:
    """Read a term whose producer may land in a LATER batch.

    The absence handled here is the DECLARED B1 state of a slot whose producer
    has not landed, and every call site names the batch it lands in. It is the
    reader for every term EXCEPT the two D-68 attributes, which the route table
    rejects a default for by name and which are therefore read bare.
    """
    if obj is None:
        return neutral
    value = getattr(obj, name, _ABSENT)
    return neutral if value is _ABSENT else value


def _is_pool_group(obj: Any) -> bool:
    """Is this bound object the ``HostPoolGroup`` the route names?

    ``cache_controller.mem_pool_host`` is NOT always a group: the non-hybrid
    controller binds whatever it was built with at ``cache_controller.py:630``
    ``        self.mem_pool_host = mem_pool_host``, and on that path it is a
    plain host pool. That configuration has no host-pool group and must stay
    neutral -- a supported boot may not gain a STOP it never had.

    ``entry_map`` is the group's own discriminator: ``HostPoolGroup.__init__``
    sets it at ``memory_pool_host.py:1902`` and no host pool in that module
    carries one. Tested for PRESENCE, not truth -- an empty map is still a
    group.
    """
    return getattr(obj, "entry_map", None) is not None


def _take_per_pass_count(obj: Any, name: str) -> int:
    """Read a per-pass counter AND CLEAR IT, at the payload-build site.

    NOT once per scheduler pass. The packed reduce does not run in the PP phase
    at all, so a per-pass reset erases every PP-phase detection before the next
    TP reduce can read it -- a group STOP deleted by a RESET rather than by a
    missing vote.

    The clear is written back where the producer WROTE it: a class attribute
    cleared on the instance is shadowed for the life of that object, and the
    counter would then vote 0 for ever while the class value went on climbing.
    """
    if obj is None:
        return 0
    raw = getattr(obj, name, _ABSENT)
    if raw is _ABSENT:
        return 0
    value = int(raw or 0)
    if value:
        target = obj if name in getattr(obj, "__dict__", {}) else type(obj)
        setattr(target, name, 0)
    return value


def _bound_group(scheduler: Any) -> Any:
    """The bound ``HostPoolGroup``, through the chain the Scheduler itself
    already reads at ``scheduler.py:6056``:
    ``scheduler.tree_cache`` -> ``.cache_controller`` -> ``.mem_pool_host``.

    A hop that does not exist is not a producer that voted badly -- it is a
    configuration with no host tier -- so the terms behind it read neutral."""
    controller = _bound_controller(scheduler)
    if controller is None:
        return None
    return getattr(controller, "mem_pool_host", None)


def _bound_controller(scheduler: Any) -> Any:
    tree = getattr(scheduler, "tree_cache", None)
    if tree is None:
        return None
    return getattr(tree, "cache_controller", None)


def _rebind_domain_within_driven(controller: Any, group: Any, bound_phase: Any) -> int:
    """Slot 15, COMPUTED ON READ and already POSITIVE -- ``num_layers ==
    expected domain`` is True when GOOD, so it must not be inverted again.

    TWO OBJECTS IN BOTH WINDOWS, which is the whole reason the right-hand term
    goes through the named accessor: the left term is the counter's width and
    the right term is the bound GROUP's expected domain for the phase. A
    version that compared the counter against a property derived from the same
    counter would be a guard whose two terms come from one object."""
    if controller is None or group is None:
        return 1
    counter = getattr(controller, "layer_done_counter", None)
    if counter is None:
        return 1
    accessor = getattr(group, "expected_transfer_layer_domain", None)
    if accessor is None:
        return 1
    expected = accessor(bound_phase)
    if expected is None:
        return 1
    return 1 if int(getattr(counter, "num_layers", 0) or 0) == int(expected) else 0


def _mamba_host_pool(group: Any) -> Any:
    """The bound group's MAMBA entry's host pool -- the same attribute the
    tree's own dispatcher reads at ``memory_pool_host.py:2094``."""
    if group is None:
        return None
    entry_map = getattr(group, "entry_map", None)
    if not entry_map:
        return None
    # HARD DEPENDENCY, IMPORTED WITHOUT A SWALLOW. `PoolName` is the KEY TYPE of
    # the dict on the line above, so an import that does not resolve is a route
    # that cannot be walked -- not an optional feature. Swallowing it would
    # answer the same neutral a healthy pass answers, which is a detection with
    # no vote wearing a log line. The failure is a class-shape fact, identical
    # on every rank and raised before this pass's reduce.
    from sglang.srt.mem_cache.hicache_storage import PoolName

    entry = entry_map.get(PoolName.MAMBA)
    if entry is None:
        return None
    return getattr(entry, "host_pool", None)


def _mamba_component(scheduler: Any) -> Any:
    """The MAMBA tree component -- slots 3-4's home, MAMBA-only because the
    #928 host-provenance branch is a mamba-only site."""
    tree = getattr(scheduler, "tree_cache", None)
    components = getattr(tree, "components", None) if tree is not None else None
    if not components:
        return None
    # HARD DEPENDENCY, IMPORTED WITHOUT A SWALLOW, AND FROM THE PACKAGE THAT
    # EXPORTS IT. `ComponentType` is the KEY TYPE of the dict on the line above
    # -- it is defined at `unified_cache_components/tree_component.py:31` and
    # re-exported by the package, which is where `registry.py:173` itself reads
    # it from; `mem_cache.registry` does not export it at all. An import that
    # does not resolve is a route that cannot be walked, and a swallow here
    # answers the same neutral a healthy pass answers, so slots 3-4 would carry
    # a detection with no vote for as long as nobody drove the counter.
    from sglang.srt.mem_cache.unified_cache_components import ComponentType

    return components.get(ComponentType.MAMBA)


def _bound_phase() -> Any:
    """The phase whose pools the readers currently name. Imported locally, the
    way ``scheduler.py:5515`` already imports it, so this module keeps the
    dependency-free import surface ``prefetch_ballot.py`` has.

    NO SWALLOW, for the reason its two sibling route imports have none: at B6
    this value SELECTS the phase row the slot-15 accessor answers from, so an
    import quietly resolving to ``None`` would pick a row nobody asked for.
    ``bound_phase()`` itself cannot fail -- it returns ``BindingState.phase``,
    a plain string attribute behind a property.
    """
    from sglang.srt.mem_cache.hicache_phase_binding import bound_phase

    return bound_phase()


def _own_census_slot(rank: int, vote: int) -> List[int]:
    """This rank's census row for one term: its own vote at its own position,
    the MIN-neutral 1 everywhere else, so the reduce leaves each position
    holding exactly the rank that owns it.

    A rank whose index is at or beyond the block width owns no slot and writes
    the neutral row -- the BOUNDED LIMIT the unpack prints as `?` from the
    world size rather than as the healthy `1` a real rank also writes."""
    census = [1] * PHASE_DOMAIN_CENSUS_SLOTS
    if 0 <= rank < PHASE_DOMAIN_CENSUS_SLOTS:
        census[rank] = 1 if int(vote) else 0
    return census


def read_phase_domain_terms(scheduler: Any) -> Dict[str, Any]:
    """This rank's value for every term, read at its ONE declared home.

    Per-pass counters are read AND CLEARED here, in the same block, because
    this site runs exactly as often as the reduce that consumes them.
    """
    controller = _bound_controller(scheduler)
    group = _bound_group(scheduler)
    pool = getattr(scheduler, "req_to_token_pool", None)
    allocator = getattr(pool, "mamba_allocator", None) if pool is not None else None
    terms: Dict[str, Any] = {}

    # Slots 0 and 1-2 -- S2-C3's two helpers, beside `local_seam_premise_vote`.
    # Producer lands in B2; until then both read their neutral, and the digest
    # pair's neutral is (0, 0) rather than a computed digest of an empty item
    # list, which is a value nobody has agreed on across ranks.
    from sglang.srt.managers import phase_purity

    coverage_vote = getattr(phase_purity, "local_loader_coverage_vote", None)
    if coverage_vote is not None:
        terms["loader_covers_own_layers"] = int(coverage_vote(scheduler))
    domain_digest = getattr(phase_purity, "local_transfer_domain_digest", None)
    if domain_digest is not None:
        terms["d_domain"] = int(domain_digest(scheduler))

    # Slots 3-4 -- S4-C5's per-pass host-provenance refusal count (B4).
    terms["d_host_prov"] = _take_per_pass_count(
        _mamba_component(scheduler), HOST_PROV_REFUSALS_ATTR
    )

    # Slots 5-6 and 7-8 -- S5-C4c and S5-C6, on the request-to-token pool (B5).
    terms["d_ownership"] = _take_per_pass_count(pool, OWNERSHIP_SPLIT_ATTR)
    terms["d_state_src"] = _take_per_pass_count(pool, STATE_SRC_ATTR)

    # Slot 9 and the census -- S2-C2's rank-local flag (B2), stored NEGATIVE and
    # INVERTED here, so an un-fired detection votes 1.
    incomplete = _read_or_neutral(controller, LOADBACK_INCOMPLETE_ATTR, False)
    loadback_complete = 0 if incomplete else 1
    terms["loadback_coverage_complete"] = loadback_complete
    if incomplete and controller is not None:
        setattr(controller, LOADBACK_INCOMPLETE_ATTR, False)

    rank = int(getattr(getattr(scheduler, "ps", None), "tp_rank", 0) or 0)
    terms["census_loadback"] = _own_census_slot(rank, loadback_complete)

    # Slot 10 -- NO PRODUCER, and this absence is the fix rather than an
    # omission. The term compared the DRAFT host tier's layer count against
    # the TARGET tier's `transfer_layer_domain`: 1 against 64 on every
    # speculative config, on an AND slot, i.e. a deterministic group STOP at
    # the first cutover (measured, six PhaseDomainDivergence raises in
    # boot_855_weg1b12s1_6917f46dd5_0906_115410.log).
    #
    # HAZARD THE TERM CLAIMED TO NAME, and why nothing is left to vote: "one
    # loop is driven over two domains, so the narrower one is silently
    # truncated". The draft half of that loop is bounded by the DRAFT tier's
    # own count -- `cache_controller.py` `start_loading`, `and i <
    # self.mem_pool_host_draft.layer_num` -- so the layers the draft does not
    # cover are skipped by construction, not silently. Comparing the draft
    # tier against its OWN domain instead is a tautology on a dense-from-0
    # tier and a NEW false STOP on a composite one, where `layer_num` (the
    # anchor's count) and `transfer_layer_domain` (1 + max key) legitimately
    # differ on a PP stage. So the producer is deleted rather than repaired.
    # The slot keeps its declaration and reads the MIN-neutral 1, which is the
    # same shape boot row 3 already has on this rig.

    # Slots 11-12 -- S3-C9 (B3), SUMMED over the built components.
    tree = getattr(scheduler, "tree_cache", None)
    components = getattr(tree, "components", None) if tree is not None else None
    unresolvable = 0
    if components:
        for component in components.values():
            unresolvable += _take_per_pass_count(component, HOST_UNRESOLVABLE_ATTR)
    terms["d_host_unresolvable"] = unresolvable

    # Slots 13-14 -- S5-C4b/S5-C4d (B5), all four allocator transitions summed.
    terms["d_slot_ownership"] = _take_per_pass_count(allocator, SLOT_OWNERSHIP_ATTR)

    # Slot 15 -- S1-C16 (B1), computed on read through the named accessor. Its
    # census carries the SAME value at this rank's own position; a census with
    # no producer renders the neutral row on every rank, which is what made the
    # slot-15 death line unreadable (measured: `per_rank=[1,1,1]` beside two
    # `local=0` votes).
    rebind_within_driven = _rebind_domain_within_driven(
        controller, group, _bound_phase()
    )
    terms["rebind_domain_within_driven"] = rebind_within_driven
    terms["census_rebind"] = _own_census_slot(rank, rebind_within_driven)

    # Slots 16-17 -- S4-C6 (B4), MAX-consumed.
    terms["d_geom"] = _take_per_pass_count(_mamba_host_pool(group), GEOM_MISMATCH_ATTR)

    # Slot 18 and slots 19-20 -- S7's two values (B6), read BY NAME off the
    # bound group with NO default. At B1 they answer S1-C18's class-attribute
    # neutrals, so the payload is byte-for-byte the neutral one.
    #
    # TWO ABSENCES, AND THEY ARE NOT ONE. "No bound HostPoolGroup" is a
    # configuration with no host tier and stays NEUTRAL. "The bound object IS
    # the group and does not declare the attribute" is a ROUTE DEFECT, and
    # answering the neutral there is the getattr-default-on-a-ledger-path shape
    # this route rejects by name: it would turn a missing S1-C18 declaration
    # into a silently healthy vote on the one term that exists to STOP the
    # group.
    #
    # SO THESE TWO READS CARRY NO DEFAULT -- and no refusal of this module's own
    # either. The bare attribute read IS the expression S0-C2 writes, and a
    # missing declaration dies out of the read itself, naming the holder and the
    # attribute. A named RuntimeError here would be a NEW rank-local refusal at
    # a per-pass site: D-20 admits one only for a boot-time invariant before the
    # first collective, and no numbered change owns one.
    #
    # THE SCOPE IS THESE TWO ATTRIBUTES AND NOTHING ELSE. Slots 10 and 15 read
    # terms whose neutral the layout pins (S0-C1's neutral-value list, T-46),
    # so they answer 1 when the route hands them nothing.
    if _is_pool_group(group):
        terms["host_ring_discarded"] = int(getattr(group, HOST_RING_DISCARD_OK_ATTR))
        # READ AND RESTORED TO 1, through the SAME object the read named: a
        # read through the bound group paired with a reset through anything
        # else is a flag that never clears or clears someone else's.
        setattr(group, HOST_RING_DISCARD_OK_ATTR, 1)

        current = int(getattr(group, D_BACKUP_WIDTH_ATTR) or 0)
        last_seen = int(
            _read_or_neutral(scheduler, D_BACKUP_WIDTH_LAST_SEEN_ATTR, 0) or 0
        )
        # THE DELTA, NOT A READ-AND-ZERO. The counter is MONOTONIC and its
        # writer is the storage backup THREAD, so zeroing it here would be a
        # cross-thread read/modify/write and any increment landing between the
        # read and the zero would be lost silently. The bookmark moves instead,
        # and a lost increment becomes a DEFERRED one.
        terms["d_backup_width"] = max(0, current - last_seen)
        setattr(scheduler, D_BACKUP_WIDTH_LAST_SEEN_ATTR, current)

    return terms


def build_phase_domain_payload(scheduler: Any) -> List[int]:
    """This rank's slice of the packed reduce. Reads once, packs once.

    Called from the payload-build site at ``scheduler.py:7171``, which sits
    AFTER the world-size guard's return at ``:6978`` and is therefore reached
    exactly as often as the reduce -- which is what makes reading and clearing
    the per-pass counters here correct and doing it once per scheduler pass a
    deleted STOP.
    """
    return pack_phase_domain_payload(read_phase_domain_terms(scheduler))


class PhaseDomainDivergence(RuntimeError):
    """#1206 / raenge-nie-uneins: the ranks do not agree on a fact that decides
    state reuse.

    Raised by ``unpack_phase_domain`` and ``unpack_boot_reduce`` on EVERY rank
    of the reduce in the same pass -- the ``(x, -x)`` pair makes the group's min
    and max visible to all of them, so the STOP is group-uniform by
    construction. A ``RuntimeError`` so the existing scheduler death path
    (run_scheduler_process: except Exception -> SIGQUIT -> kill_process_tree)
    takes it without a new collective or a new handler.
    """


_RAENGE = (
    "RAENGE-NIE-UNEINS: the ranks do not agree on a fact that decides state "
    "reuse. No compensation exists for this; the group stops."
)

#: THE SAME LAW, THE OTHER SHAPE, and the two must not be told in one sentence.
#: An AND slot at 0 and a MAX pair above 0 say "at least one rank refused", and
#: a UNANIMOUS refusal rendered as "the ranks do not agree" sends the reader
#: hunting for a divergence that is not there. Measured: the first cutover of
#: boot_855_weg1b12s1 printed `local=0 group_min=0 group_max=0 per_rank=[1,1,1]`
#: under the divergence sentence. Only the min != max pairs are DISAGREEMENT;
#: these are REFUSALS.
#:
#: THIS SENTENCE WAS ITSELF WRONG IN THE ROUND-3 SHAPE and is corrected here:
#: it said "a MIN carries no count, so this line does not say whether the
#: refusal was unanimous", which was true only while an AND slot was one slot
#: wide and `group_max` was a copy of `group_min`. Both halves of the AND term
#: now ride the reduce, so `group_max` is a real maximum and the line DOES
#: separate a unanimous refusal from a partial one.
#:
#: THE PROSE MUST NOT SPELL A FIELD THE LINE ALSO RENDERS. An earlier draft of
#: this sentence wrote the two cases as `group_max=1` and `group_max=0`, and
#: two mutation probes then SURVIVED because an assertion looking for the
#: rendered value found the law text instead -- a guard that cannot fail,
#: manufactured by the instrument's own explanatory sentence. The cases are
#: named in words here for that reason.
_RAENGE_REFUSED = (
    "RAENGE-NIE-UNEINS: at least one rank refused a fact that decides state "
    "reuse. A group max of one means at least one rank was HEALTHY, i.e. the "
    "refusal was partial and depends on something rank-local; a group max of "
    "zero means every rank refused. The `local=` field is this rank's own "
    "vote. No compensation exists for either shape; the group stops."
)


class PhaseDomainVerdict:
    """What the group said, per term. A message field, never a second record."""

    __slots__ = (
        "and_slots",
        "and_maxima",
        "pairs",
        "census",
        "per_rank",
        "census_width",
        "world_size",
    )

    def __init__(
        self, and_slots, and_maxima, pairs, census, per_rank, census_width, world_size
    ):
        self.and_slots = and_slots
        #: The group MAX of each AND term, from the negated half of its pair.
        #: `and_slots[name] == 0 and and_maxima[name] == 1` is the 2-of-3 shape
        #: the one-slot layout could not express.
        self.and_maxima = and_maxima
        self.pairs = pairs
        #: Per CENSUSED term name, not per bus: a census belongs to the term it
        #: was voted for.
        self.census = census
        self.per_rank = per_rank
        self.census_width = census_width
        self.world_size = world_size


#: What a STOP line prints where the failing term has no per-rank census. NOT
#: an empty list and NOT another term's row: a reader who sees a bracketed row
#: of numbers reads them as this term's votes, which is exactly how a slot-15
#: death came to print `per_rank=[1,1,1]` beside two `local=0` votes.
CENSUS_ABSENT_ROW = "none"


def _render_census(
    reduced: Sequence[int], world_size: Optional[int], term_name: str
) -> Tuple[Optional[List[Any]], str, int]:
    """THREE CASES, and the third is decided by the WORLD SIZE, never by a value.

    * ``census[r]`` reduces to exactly rank r's own flag for every r below the
      world size, because every other rank wrote the MIN-neutral 1 there. A 0 is
      unambiguous: that rank refused.
    * For r at or above the world size every rank wrote 1, so the slot is
      indistinguishable from a healthy rank ON THE PAYLOAD ALONE. The caller
      knows the world size, so those positions render ``?``.
    * The rendering is a message field. Nothing about the census is ever a STOP
      condition -- that stays the term's own AND-slot.

    KEYED BY THE CENSUSED TERM. A term with no census of its own gets no row at
    all rather than the first block in the table; the group MAX above it is the
    exact statement for those terms, and it is not bounded by the census width.
    """
    block = _CENSUS_OF.get(term_name)
    if block is None:
        return None, CENSUS_ABSENT_ROW, 0
    at = index_of(block.name)
    values: List[Any] = []
    for offset in range(block.width):
        if world_size is not None and offset >= int(world_size):
            values.append(None)
        else:
            values.append(int(reduced[at + offset]))
    rendered = ",".join("?" if v is None else str(v) for v in values)
    return values, rendered, block.width


def unpack_phase_domain(
    reduced: Sequence[int],
    *,
    rank: int = -1,
    local: Optional[Dict[str, Any]] = None,
    world_size: Optional[int] = None,
    phase: Any = None,
) -> Optional[PhaseDomainVerdict]:
    """The GROUP verdict from the reduced packed slice.

    Raises ``PhaseDomainDivergence`` when ANY of the five AND-slots reads 0,
    when a divergence pair's ``min != max``, or when a MAX pair's
    ``group_max > 0``. ``None`` is returned ONLY for a slice of the wrong width
    -- a layout defect the caller turns into its own STOP -- never for a
    disagreement, the discipline ``prefetch_ballot.py:145-147`` states.

    The width checked is ``PHASE_DOMAIN_SLOTS`` with NO ``+ 2``: that constant
    is the TOTAL here, unlike ``PREFETCH_BALLOT_SLOTS``, which counts only the
    per-rid verdict slots.
    """
    if len(reduced) != PHASE_DOMAIN_SLOTS:
        return None
    local = {} if local is None else local

    and_slots: Dict[str, int] = {}
    and_maxima: Dict[str, int] = {}
    pairs: Dict[str, Tuple[int, int]] = {}
    census: Dict[str, List[Any]] = {}
    per_rank: Dict[str, str] = {}
    for term in PHASE_DOMAIN_LAYOUT:
        at = index_of(term.name)
        if term.kind == AND_SLOT:
            and_slots[term.name] = int(reduced[at])
            and_maxima[term.name] = -int(reduced[at + 1])
        elif term.kind in (DIVERGENCE_PAIR, MAX_PAIR):
            pairs[term.name] = (int(reduced[at]), -int(reduced[at + 1]))

    for censused in _CENSUS_OF:
        values, rendered, _width = _render_census(reduced, world_size, censused)
        census[censused] = values
        per_rank[censused] = rendered

    verdict = PhaseDomainVerdict(
        and_slots,
        and_maxima,
        pairs,
        census,
        per_rank,
        PHASE_DOMAIN_CENSUS_SLOTS,
        world_size,
    )

    def _row(term_name):
        """THE ROW THIS TERM VOTED, never the first census in the table. A
        borrowed row contradicts the `local=` value printed beside it and sends
        the reader after a divergence that is not there.

        Called only on the raise path: this unpack runs once per scheduler pass
        in the TP phase, and rendering every term's row on the healthy path
        would be thirteen list builds per pass for a message nobody prints."""
        return _render_census(reduced, world_size, term_name)[1:]

    for term in PHASE_DOMAIN_LAYOUT:
        if term.kind == AND_SLOT:
            group_min = and_slots[term.name]
            if group_min == 0:
                _rendered, _width = _row(term.name)
                raise PhaseDomainDivergence(
                    _stop_message(
                        term,
                        rank=rank,
                        local=local.get(term.name),
                        group_min=group_min,
                        group_max=and_maxima[term.name],
                        phase=phase,
                        per_rank=_rendered,
                        census_width=_width,
                    )
                )
        elif term.kind == DIVERGENCE_PAIR:
            group_min, group_max = pairs[term.name]
            if group_min != group_max:
                _rendered, _width = _row(term.name)
                raise PhaseDomainDivergence(
                    _stop_message(
                        term,
                        rank=rank,
                        local=local.get(term.name),
                        group_min=group_min,
                        group_max=group_max,
                        phase=phase,
                        per_rank=_rendered,
                        census_width=_width,
                    )
                )
        elif term.kind == MAX_PAIR:
            group_min, group_max = pairs[term.name]
            # NOT the `min != max` predicate the divergence pairs use: this
            # condition is a wrong answer on ANY rank, so a nonzero count that
            # is UNIFORM must stop the group too.
            if group_max > 0:
                _rendered, _width = _row(term.name)
                raise PhaseDomainDivergence(
                    _stop_message(
                        term,
                        rank=rank,
                        local=local.get(term.name),
                        group_min=group_min,
                        group_max=group_max,
                        phase=phase,
                        per_rank=_rendered,
                        census_width=_width,
                    )
                )
    return verdict


def _stop_message(
    term, *, rank, local, group_min, group_max, phase, per_rank, census_width
) -> str:
    # WHICH LAW SENTENCE, decided by the KIND of the term and not by its
    # values: a divergence pair really did disagree (min != max), while an AND
    # slot and a MAX pair say only that at least one rank refused.
    law = _RAENGE if term.kind == DIVERGENCE_PAIR else _RAENGE_REFUSED
    if term.terse_stop:
        # The B1 consumer renders rank, group_min and the flag only -- the
        # phase term and the census are deliberately not on this line.
        return "%s STOP rank=%d group_min=%d local=%s -- %s" % (
            term.refusal,
            int(rank),
            int(group_min),
            local,
            law,
        )
    return (
        "#1206 PHASE DOMAIN DIVERGENCE STOP rank=%d term=%s refusal=%s local=%s "
        "group_min=%d group_max=%d phase=%s per_rank=[%s] census_width=%d -- %s"
        % (
            int(rank),
            term.name,
            term.refusal,
            local,
            int(group_min),
            int(group_max),
            phase,
            per_rank,
            # THIS TERM'S census width, 0 when it has none -- not the module
            # constant, which would claim eight per-rank slots on a line whose
            # `per_rank` says there are none.
            int(census_width),
            law,
        )
    )


def layout_instrument_line(head_index: int, phase: Any) -> str:
    """One INFO line at the first publication per phase, so a future layout
    change is legible instead of silent."""
    terms = ",".join(
        "%s@%d:%d" % (term.name, index_of(term.name), term.width)
        for term in PHASE_DOMAIN_LAYOUT
    )
    return (
        "#1068 PHASE-DOMAIN BUS phase=%s head=%d slots=%d census=%d layout=[%s]"
        % (phase, int(head_index), PHASE_DOMAIN_SLOTS, PHASE_DOMAIN_CENSUS_SLOTS, terms)
    )


# ---------------------------------------------------------------------------
# THE SECOND, SMALLER BUS: the BOOT reduce's layout.
#
# Taken once, at boot, on `scheduler.world_group.cpu_group` -- NOT on the packed
# bus's group, which is a singleton at boot and through every PP phase. The
# whole table is declared here in B1 even though eight of the thirteen scalars
# have no producer until B6: a bus declared four wide and appended to later
# makes the boot unpack return None on a width it does not recognise, and every
# boot verdict silently stops being read on the batch that needs it.
# ---------------------------------------------------------------------------


class _BootRow:
    __slots__ = ("name", "kind", "width", "neutral", "owner")

    def __init__(self, name, kind, owner):
        self.name = name
        self.kind = kind
        self.owner = owner
        if kind == AND_SLOT:
            self.width = 1
            self.neutral = 1
        elif kind == DIVERGENCE_PAIR:
            self.width = 2
            self.neutral = 0
        else:  # pragma: no cover - a typo in the table, not a runtime state
            raise ValueError(f"unknown boot row kind {kind!r} for {name!r}")


#: | row   | term                              | kind   | neutral | fills   | batch |
#: | 0     | owned_equals_driven               | AND    | 1       | S1-C14  | B1    |
#: | 1     | layer_mapping_non_empty           | AND    | 1       | S1-C13  | B1    |
#: | 2     | counter_index_space_known         | AND    | 1       | S1-C11  | B1    |
#: | 3     | draft_tier_domain_matches_at_boot | AND    | 1       | none on this rig |
#: | 4-5   | d_ring_format                     | digest | (0, 0)  | S7      | B6    |
#: | 6-7   | d_entry_map                       | digest | (0, 0)  | S7      | B6    |
#: | 8     | write_through_ring_ok             | AND    | 1       | S7      | B6    |
#: | 9     | one_wave_floor_ok                 | AND    | 1       | S7      | B6    |
#: | 10    | anchor_floor_ok                   | AND    | 1       | S7      | B6    |
#: | 11    | ring_arity_ok                     | AND    | 1       | S7      | B6    |
#: | 12    | host_pool_build_ok                | AND    | 1       | S1-C14  | B1    |
#:
#: Row 0 IS the transfer-domain SHORTFALL term -- one predicate, one name. A
#: separate `transfer_domain_shortfall` row beside it would be a second record
#: of one predicate, and the resulting fourteen-wide bus fails the width pin.
#:
#: Row 3 is declared with NO PRODUCER on this rig, deliberately: the day a slice
#: supplies one it supplies a VALUE, not a slot. Row 12 is its mirror -- filled
#: from B1 to B5 and unproduced from B6. Neither is a licence to vote 0 because
#: the term is unproduced.
BOOT_REDUCE_LAYOUT: Tuple[_BootRow, ...] = (
    _BootRow("owned_equals_driven", AND_SLOT, "S1-C14"),
    _BootRow("layer_mapping_non_empty", AND_SLOT, "S1-C13"),
    _BootRow("counter_index_space_known", AND_SLOT, "S1-C11"),
    _BootRow("draft_tier_domain_matches_at_boot", AND_SLOT, "none on this rig"),
    _BootRow("d_ring_format", DIVERGENCE_PAIR, "S7"),
    _BootRow("d_entry_map", DIVERGENCE_PAIR, "S7"),
    _BootRow("write_through_ring_ok", AND_SLOT, "S7"),
    _BootRow("one_wave_floor_ok", AND_SLOT, "S7"),
    _BootRow("anchor_floor_ok", AND_SLOT, "S7"),
    _BootRow("ring_arity_ok", AND_SLOT, "S7"),
    _BootRow("host_pool_build_ok", AND_SLOT, "S1-C14"),
)

#: DERIVED from the rows above and never typed as a literal, the same rule the
#: packed bus obeys: 9 AND-slots + 2x2 digest-pair scalars = 13.
PHASE_BOOT_REDUCE_SLOTS = sum(row.width for row in BOOT_REDUCE_LAYOUT)

_BOOT_INDEX: Dict[str, int] = {}
_boot_at = 0
for _row in BOOT_REDUCE_LAYOUT:
    _BOOT_INDEX[_row.name] = _boot_at
    _boot_at += _row.width
del _boot_at, _row

#: Printed by a rank whose OWN row-12 term was healthy. An int64 MIN carries no
#: string, so no rank can print a peer's reason; every rank prints its own line.
HOST_POOL_BUILD_OK_PEER_LINE = (
    "#1068 host_pool_build_ok: row 12 healthy on this rank; a peer recorded the "
    "failure, see its log"
)

#: The same statement for every OTHER boot row. Kept separate from the row-12
#: line above because that one names its row in prose and is asserted verbatim.
BOOT_PEER_REFUSED_LINE = (
    "peer refused: this row is healthy on this rank; the rank that recorded "
    "the failure prints it in its own log"
)

#: This rank refused and recorded no string. Named rather than rendered as the
#: peer line, which would claim this rank was healthy -- the instrument-text
#: hazard the row-12-only rendering already produced once.
BOOT_REFUSED_NO_REASON_LINE = (
    "this rank refused this row and recorded no reason string"
)


def boot_index_of(row_name: str) -> int:
    """The head index of a boot row, DERIVED from ``BOOT_REDUCE_LAYOUT``."""
    return _BOOT_INDEX[row_name]


def build_boot_reduce_payload(terms: Optional[Dict[str, Any]] = None) -> List[int]:
    """This rank's contribution to the boot reduce, in declared order.

    A row absent from ``terms`` has no producer in this batch and contributes
    its NEUTRAL: 1 for an AND row, ``(0, 0)`` for a digest pair.
    """
    terms = {} if terms is None else terms
    payload: List[int] = []
    for row in BOOT_REDUCE_LAYOUT:
        if row.kind == AND_SLOT:
            payload.append(1 if int(terms.get(row.name, row.neutral)) else 0)
        else:
            value = int(terms.get(row.name, row.neutral))
            payload.append(value)
            payload.append(-value)
    return payload


class BootReduceVerdict:
    __slots__ = ("and_slots", "pairs")

    def __init__(self, and_slots, pairs):
        self.and_slots = and_slots
        self.pairs = pairs


def unpack_boot_reduce(
    reduced: Sequence[int],
    *,
    rank: int = -1,
    local: Optional[Dict[str, Any]] = None,
) -> Optional[BootReduceVerdict]:
    """The GROUP verdict from the reduced boot payload.

    Same discipline as the packed bus: ``None`` for a slice of the wrong width,
    a raise for a disagreement, never the other way round.
    """
    if len(reduced) != PHASE_BOOT_REDUCE_SLOTS:
        return None
    local = {} if local is None else local

    and_slots: Dict[str, int] = {}
    pairs: Dict[str, Tuple[int, int]] = {}
    for row in BOOT_REDUCE_LAYOUT:
        at = boot_index_of(row.name)
        if row.kind == AND_SLOT:
            and_slots[row.name] = int(reduced[at])
        else:
            pairs[row.name] = (int(reduced[at]), -int(reduced[at + 1]))

    for row in BOOT_REDUCE_LAYOUT:
        if row.kind == AND_SLOT:
            if and_slots[row.name] == 0:
                raise PhaseDomainDivergence(_boot_stop_message(row, rank, local))
        else:
            group_min, group_max = pairs[row.name]
            if group_min != group_max:
                raise PhaseDomainDivergence(
                    "#1206 PHASE DOMAIN DIVERGENCE STOP (boot) rank=%d row=%s "
                    "local=%s group_min=%d group_max=%d -- %s"
                    % (int(rank), row.name, local.get(row.name), group_min, group_max, _RAENGE)
                )
    return BootReduceVerdict(and_slots, pairs)


def _boot_stop_message(row, rank, local) -> str:
    """The boot STOP line, ONE per rank, carrying THIS rank's reason.

    The rank that RECORDED the failure prints its own recorded reason; a
    healthy rank says a peer refused and points at that peer's log. A MIN
    cannot say WHICH rank voted 0, and this bus adds no second collective, so
    no rank can print a peer's string.

    HAZARD THIS SHAPE CLOSES: the reason used to be rendered for row 12 alone,
    so a routed `#1068 TP PIN CELL UNDERIVABLE` or a `#1206 TRANSFER DOMAIN
    SHORTFALL` -- both of which refuse on ROW 0, which `unpack_boot_reduce`
    reaches first -- killed all three ranks with `reason=None` and named
    neither the refusing rank nor the cause anywhere on the STOP path.
    """
    own_refusal = not int(local.get(row.name, 1) or 0)
    reason = local.get("host_pool_build_msg") if own_refusal else None
    if reason is None:
        if own_refusal:
            reason = BOOT_REFUSED_NO_REASON_LINE
        elif row.name == "host_pool_build_ok":
            reason = HOST_POOL_BUILD_OK_PEER_LINE
        else:
            reason = BOOT_PEER_REFUSED_LINE
    return (
        "#1206 PHASE DOMAIN DIVERGENCE STOP (boot) rank=%d row=%s local=%s "
        "group_min=0 reason=%s -- %s"
        % (int(rank), row.name, local.get(row.name), reason, _RAENGE_REFUSED)
    )


# THE TWO MODULE INVARIANTS THAT KEEP THE HALVES FROM DRIFTING BETWEEN TEST
# RUNS. Written as raises rather than as `assert`, which `python -O` strips --
# and a width check that can be optimised away is the one that is not there on
# the boot that needs it.
if len(pack_phase_domain_payload({})) != PHASE_DOMAIN_SLOTS:  # pragma: no cover
    raise RuntimeError(
        "#1068 PHASE-DOMAIN LAYOUT DEFECT: the packer emits %d slots and "
        "PHASE_DOMAIN_SLOTS derives to %d"
        % (len(pack_phase_domain_payload({})), PHASE_DOMAIN_SLOTS)
    )
if len(build_boot_reduce_payload()) != PHASE_BOOT_REDUCE_SLOTS:  # pragma: no cover
    raise RuntimeError(
        "#1068 BOOT-REDUCE LAYOUT DEFECT: the packer emits %d slots and "
        "PHASE_BOOT_REDUCE_SLOTS derives to %d"
        % (len(build_boot_reduce_payload()), PHASE_BOOT_REDUCE_SLOTS)
    )
