"""#631/#968: the PRODUCER-PHASE axis of a HiCache hit.

THE ONE QUESTION THIS EXISTS TO ANSWER
--------------------------------------
How many decodes served in the TP3 layout ran on context that was PRODUCED in
the PP3 layout -- ``ok > 0`` WITH ITS DENOMINATOR.

WHY THE EXISTING INSTRUMENT CANNOT ANSWER IT
--------------------------------------------
``match_refusal_census`` (#904/#913) partitions a prefix-match walk into
NOT_PRESENT / DEAD / REFUSED / HIT and refuses to invent a zero
(``NO_OBSERVATION``). That partition is correct and is NOT duplicated here.
What it has no axis for is WHERE THE ACCEPTED BYTES CAME FROM. A ``HIT`` is
compatible with two worlds that the mission must separate:

  SAME-PHASE     the bytes were written to host by THIS phase and read back by
                 THIS phase. That is the EVICTION axis. It proves the host
                 tier works; it proves nothing about the flip.
  CROSS-PHASE    the bytes were produced under an earlier binding -- a
                 different layout -- and adopted from the STORE into this one.
                 That, and only that, is the mission.

Measured basis for the distinction (desk line, 2026-09-01): MAMBA-HOST-RESUME
fires in 311 of 1338 boot logs, so host resume WORKS -- but every one of those
hits arrives through the ``BACKUP_HOST`` arm, i.e. state this same process
wrote to host. The ``PREFETCH`` arm, the only route by which a PP3-produced
anchor can enter a TP3 layout, emits NOTHING: no logger call on its success
path and none on any of the four terms of its decline conjunction. The honest
reading of the user's question today is therefore NOT "no" -- it is
NO_OBSERVATION, and that is precisely the state this module makes decidable.

THE CARRIER IS NOT REINVENTED (upstream-minimal)
------------------------------------------------
``hicache_phase_binding`` already owns phase identity: ``BindingState`` holds
the bound phase and a monotone generation, ``advance(phase, host_pool)`` mints
one generation per cutover, and ``current_generation()`` is already read on the
neighbouring prefetch path (``scheduler.py``, the #1060 store-presence key and
the #969C intake line). A second phase-identity scheme beside it would be
exactly the second bookkeeping the upstream-minimal law forbids.

But the honest statement of what that carrier does and does NOT give, because
this is the sentence a reader will otherwise get wrong:

  ``current_generation()`` returns the generation bound RIGHT NOW. Called at
  the prefetch-commit site it names the CONSUMER (TP3), never the producer.

It is the right authority; it is not, by itself, the answer. Exactly two facts
have to be added to it, and both are additive:

  1. GENERATION -> PHASE HISTORY. ``BindingState`` already records
     ``generation -> host_pool``; it does not keep ``generation -> phase``, so
     "which phase was generation 1" is unanswerable after the second cutover.
     ``note_generation`` below records it, so a producer generation can be
     rendered as a producer PHASE.
  2. THE RECORD'S WRITE GENERATION. Stamped when a page is handed to the
     storage backend, read back when that page is adopted. A process-local
     ledger is SUFFICIENT AND COMPLETE here and needs no persistence: PP3
     prefill and TP3 decode are the same process in one boot -- one NCCL
     world, one server, a layout flip in between.

WHY "DERIVE IT FROM THE CUTOVER TIMELINE + STORE mtime" IS REJECTED
-------------------------------------------------------------------
It is cheap and it does not carry. Five failures, any one of which is fatal:

  1. WRONG CLOCK JOIN. The cutover timeline is ``time.monotonic()``; a file
     mtime is wall-clock. Joining them needs a conversion that is itself an
     uncalibrated instrument -- an indicator whose own correctness is
     unproven, which is the INDIKATOR-GESETZ failure exactly.
  2. RE-WRITE MOVES THE STAMP. The file backend publishes with
     ``os.replace(tmp, path)``. A page written in PP3 and re-written in TP3
     reads as TP3-produced: a SYSTEMATIC FALSE NEGATIVE on the one number the
     mission is about, biased toward "it never happened".
  3. THREE STATES COLLAPSE TO TWO. The LRU file evictor deletes live, so a
     ``stat()`` that fails between adoption and measurement is
     indistinguishable from "not PP3-produced". The empty reading would be
     read as a zero -- the third-state defect this strand has already paid
     for once.
  4. IT MEASURES THE FILE, NOT THE BYTES. A host-tier (L2) hit has no file at
     all and would score as absent even when it is a genuine cross-phase hit.
  5. COST IN THE HOT PATH. One ``stat()`` per accepted page inside the match
     walk, to obtain a number that is wrong in case 2 anyway.

THE STAMP IS ON THE RECORD, NEVER IN THE KEY
--------------------------------------------
Explicit because it is the obvious wrong turn and it is already forbidden:
the storage key is deliberately GEOMETRY-NEUTRAL (#706/#555). A phase stamp
inside the key would re-introduce the two-geometry key -- the same page
written in PP3 and looked up in TP3 would no longer be the same page, and the
mission would become unreachable by the very instrument meant to measure it.
So provenance is keyed BY the key and never carried IN it: the ledger below
maps key -> write generation, and the key itself is untouched.

In the hermetic roundtrip test provenance is solved CONSTRUCTIVELY instead --
the store starts empty and only the PP chain writes, so anything read back is
PP-produced by construction. That argument does not survive contact with a
live boot, where both phases write into one store, which is why the live side
still needs this axis.

THE ARM LABEL IS PART OF THE ANSWER, NOT A DECORATION
-----------------------------------------------------
``AdoptionSource`` (``by_source`` on the emitted line) says whether the bytes
came through the PREFETCH arm or the BACKUP_HOST arm. Without it a green
number from BACKUP_HOST reads as evidence for a path that never ran -- the
confusion that nearly closed this whole mission as "impossible". It rides the
SAME line as ``ok``, and the #904 refusal census is armed by the SAME knob, so
the two lines cannot drift apart in a log and the arm label never has to be
re-derived from a second one.

THREE STATES, NEVER TWO
-----------------------
Every field this module reports is one of VALUE / NO_OBSERVATION / EMPTY, and
they are distinct tokens, never a bare ``0``:

  NO_OBSERVATION  the instrument was not armed, or was armed and never fed.
                  It did not measure. This is not a zero.
  EMPTY           armed and fed, and the population it was asked about was
                  genuinely empty. This IS a measurement, and its value is 0.
  VALUE           armed, fed, non-empty.

This module is a PASSIVE RECORDER. It decides nothing, evicts nothing, holds
no reference to a node or a tensor, and is off by default.
"""

from __future__ import annotations

import dataclasses
import threading
from enum import Enum

__all__ = [
    "ACCEPT_LINE_PREFIX",
    "AdoptionSource",
    "ObservationState",
    "ProducerPhase",
    "ProducerPhaseCensus",
    "adoption_source_of",
    "arrival_stats",
    "census_armed",
    "double_prefill_census",
    "emit",
    "emit_double_prefill",
    "note_double_prefill",
    "reset_double_prefill_census",
    "ledger_stats",
    "new_producer_census",
    "note_arrival",
    "note_backup_keys",
    "note_consult",
    "note_generation",
    "note_prefetch_adopted",
    "note_prefill_hit_tokens",
    "note_store_write",
    "note_walk_node",
    "reset_prefill_window",
    "payload_verdict",
    "phase_of_generation",
    "prefill_provenance_field",
    "producer_generation_of",
    "producer_phase_of",
    "reset_for_test",
    "DoublePrefillCensus",
    "DOUBLE_PREFILL_LINE_PREFIX",
    "resolve_chunk_size",
]


#: The literal that anchors every acceptance grep. Deliberately long and
#: deliberately NOT a bare ticket number: a four-digit number matches every
#: millisecond figure in a boot log (#995 / measured on boot 49).
ACCEPT_LINE_PREFIX = "#631 producer-phase"

#: The second half of the acceptance. Same anchoring discipline.
DOUBLE_PREFILL_LINE_PREFIX = "#939 double-prefill"


class ObservationState(str, Enum):
    """The three states. A reader that only knows two will misread this."""

    NO_OBSERVATION = "NO_OBSERVATION"
    EMPTY = "EMPTY"
    VALUE = "VALUE"


class ProducerPhase(str, Enum):
    """Where the accepted bytes were produced, as far as this can be known."""

    #: Produced under a binding generation strictly older than the consuming
    #: one, whose recorded phase differs from the consuming phase. THE MISSION.
    CROSS_PHASE = "cross"
    #: Produced under the generation that is consuming it. Eviction axis.
    SAME_PHASE = "same"
    #: The key was never seen by this process's write ledger, or its entry was
    #: dropped. NOT folded into ``same``: an unknown provenance that is counted
    #: as same-phase manufactures a negative result out of a gap.
    UNKNOWN = "unknown"


class AdoptionSource(str, Enum):
    """By which arm the node's host bytes were adopted.

    Kept beside the producer phase because they answer different halves and
    have been confused: ``BACKUP_HOST`` is this process writing its own device
    state to host (eviction), ``PREFETCH`` is the store handing bytes back
    (provenance). 311/1338 boot logs carry host-resume hits and all of them are
    the first kind.
    """

    PREFETCH = "prefetch"
    BACKUP_HOST = "backup_host"
    LOAD_BACK = "load_back"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------
# 1. generation -> phase history (the fact BindingState does not keep)
# --------------------------------------------------------------------------

_gen_lock = threading.Lock()
_gen_phases: dict[int, str] = {}


def note_generation(generation: int, phase: str) -> None:
    """Record that ``generation`` was bound to ``phase``.

    Called from the same ``advance()`` that mints the generation, so there is
    exactly one place where a generation and its phase are associated -- the
    same shape ``BindingState._pools`` already uses for generation -> pool.
    Idempotent: re-recording the same pair is a no-op, and a CONFLICTING
    re-record is kept as the first value and reported, because a generation
    that changed its phase under us is an instrument fault, not a datum.
    """
    if generation is None:
        return
    g = int(generation)
    p = str(phase)
    with _gen_lock:
        prior = _gen_phases.get(g)
        if prior is None:
            _gen_phases[g] = p
        elif prior != p:
            _gen_phases[g] = prior
            _gen_phases[-1] = "CONFLICT"


def phase_of_generation(generation: int | None) -> str | None:
    """The phase bound at ``generation``, or None when it was never recorded.

    None is NOT "the boot phase". An unrecorded generation is a gap in the
    instrument and has to stay visible as one.
    """
    if generation is None:
        return None
    with _gen_lock:
        return _gen_phases.get(int(generation))


def generation_history() -> tuple[tuple[int, str], ...]:
    with _gen_lock:
        return tuple(sorted((g, p) for g, p in _gen_phases.items() if g >= 0))


# --------------------------------------------------------------------------
# 2. key -> write generation (the record's producer stamp)
# --------------------------------------------------------------------------
#
# Process-local and unpersisted ON PURPOSE. The mission is one boot: PP3
# prefill and TP3 decode are the same process, so a ledger that dies with the
# process cannot miss the transition it exists to witness. Persisting it would
# add a second durable truth beside the store, with its own staleness -- the
# defect class this strand keeps paying for.

_LEDGER_MAX = 1 << 19  # 524288 keys; the measured store held 24277 pages.

_ledger_lock = threading.Lock()
_ledger: dict[str, int] = {}
_ledger_dropped = 0
_ledger_writes = 0


def note_store_write(key: str, generation: int) -> None:
    """Stamp: this key's bytes were handed to the store at ``generation``.

    Bounded, and the DROP COUNT is kept. A ledger that silently forgets turns
    a cross-phase hit into ``unknown`` with no trace, and an unknown with no
    trace is the same lie as a plausible zero.
    """
    global _ledger_dropped, _ledger_writes
    if key is None or generation is None:
        return
    k = str(key)
    g = int(generation)
    with _ledger_lock:
        _ledger_writes += 1
        if k not in _ledger and len(_ledger) >= _LEDGER_MAX:
            # FIFO on insertion order (py3.7+ dicts preserve it).
            try:
                oldest = next(iter(_ledger))
                del _ledger[oldest]
                _ledger_dropped += 1
            except StopIteration:  # pragma: no cover - empty dict at the cap
                pass
        # LAST writer wins: a re-written page IS produced by the later phase,
        # and pretending otherwise would over-report the mission's number.
        _ledger[k] = g


def producer_generation_of(key: str) -> int | None:
    if key is None:
        return None
    with _ledger_lock:
        return _ledger.get(str(key))


def producer_phase_of(key: str, consuming_generation: int) -> ProducerPhase:
    """Classify one adopted key against the generation consuming it.

    UNKNOWN whenever anything is missing -- an unstamped key, an unrecorded
    generation, a dropped ledger entry. Never guessed into ``same``.
    """
    wgen = producer_generation_of(key)
    if wgen is None or consuming_generation is None:
        return ProducerPhase.UNKNOWN
    cgen = int(consuming_generation)
    if wgen == cgen:
        return ProducerPhase.SAME_PHASE
    wphase = phase_of_generation(wgen)
    cphase = phase_of_generation(cgen)
    if wphase is None or cphase is None:
        # Different generation, but we cannot name either phase. That is a
        # weaker fact than CROSS_PHASE and must not be promoted to it.
        return ProducerPhase.UNKNOWN
    return ProducerPhase.CROSS_PHASE if wphase != cphase else ProducerPhase.SAME_PHASE


def ledger_stats() -> dict[str, int]:
    with _ledger_lock:
        return {
            "keys": len(_ledger),
            "writes": _ledger_writes,
            "dropped": _ledger_dropped,
        }


# --------------------------------------------------------------------------
# 2b. ARRIVAL ORDER -- REFUTED AS A CAUSE, RETAINED AS THE WITNESS
# --------------------------------------------------------------------------
#
# The hypothesis was that the match consults a key BEFORE its prefetch lands,
# so a real page would be missed for timing reasons. FULL-COUNT MEASUREMENT
# REFUTED IT (2026-09-01): the prefetch completes within the same second and
# the consult reads correctly. Nothing arrives late; nothing arrives at all.
#
# These counters stay, and the reason is not sentiment. A refutation that
# lives only in prose gets re-litigated; one that a running boot re-prints
# every window does not. ``late`` is expected to remain 0, and if it ever
# stops being 0 the refutation has expired and should be re-argued rather
# than assumed. No further work is owed here.

_order_lock = threading.Lock()
_missed_consults: dict[str, int] = {}  # key -> count of consults that missed
_late_arrivals = 0
_arrivals = 0
_consults = 0
_consult_misses = 0
#: #1061: keys whose bytes were adopted via the PREFETCH arm. Kept (bounded,
#: FIFO on insertion order like the write ledger) so the match walk can render
#: ``by_source`` from what actually happened instead of needing a second
#: carrier on the tree node. Value = arrival count for the key.
_arrived: dict[str, int] = {}


def note_consult(key: str, accepted: bool) -> None:
    """One page key was consulted by a match walk, and hit or missed.

    A missed key is REMEMBERED, so a later arrival for the same key can be
    scored as LATE rather than disappearing into an undifferentiated miss.
    """
    global _consults, _consult_misses
    if key is None:
        return
    k = str(key)
    with _order_lock:
        _consults += 1
        if accepted:
            return
        _consult_misses += 1
        if len(_missed_consults) < _LEDGER_MAX:
            _missed_consults[k] = _missed_consults.get(k, 0) + 1


def note_arrival(key: str) -> None:
    """A prefetched page landed and was adopted.

    If a walk had already consulted this key and missed, the page CAME TOO
    LATE for that walk. That is a real, separately actionable outcome and it
    is counted as its own term -- never as a plain miss and never as a hit.
    """
    global _late_arrivals, _arrivals
    if key is None:
        return
    k = str(key)
    with _order_lock:
        _arrivals += 1
        if _missed_consults.pop(k, 0):
            _late_arrivals += 1
        # #1061: remember the key itself (bounded), so `adoption_source_of`
        # can answer PREFETCH for it later without a per-node carrier.
        if k not in _arrived and len(_arrived) >= _LEDGER_MAX:
            try:
                oldest = next(iter(_arrived))
                del _arrived[oldest]
            except StopIteration:  # pragma: no cover - empty dict at the cap
                pass
        _arrived[k] = _arrived.get(k, 0) + 1


def arrival_stats() -> dict[str, int]:
    """The ordering partition.

    ``never`` is derived, not counted separately: keys that missed a consult
    and for which no arrival has yet been seen. It is a SNAPSHOT term -- a key
    counted as ``never`` now can become ``late`` later -- and the emission
    timestamp is what bounds it.
    """
    with _order_lock:
        return {
            "consults": _consults,
            "misses": _consult_misses,
            "arrivals": _arrivals,
            "late": _late_arrivals,
            "never_yet": len(_missed_consults),
        }


def adoption_source_of(key: str) -> AdoptionSource:
    """By which arm this key's bytes are known to this process (#1061).

    ARITHMETIC: PREFETCH if ``note_arrival`` ever recorded the key (the store
    handed its bytes back and the tree adopted them); else BACKUP_HOST if the
    write ledger stamped it (this process's own backup thread wrote it, so a
    hit on it is the eviction axis); else UNKNOWN -- never guessed. A key both
    prefetched and later re-backed-up reads PREFETCH: the adoption event is
    what put its bytes into this layout, the later copy is bookkeeping.
    """
    if key is None:
        return AdoptionSource.UNKNOWN
    k = str(key)
    with _order_lock:
        if k in _arrived:
            return AdoptionSource.PREFETCH
    if producer_generation_of(k) is not None:
        return AdoptionSource.BACKUP_HOST
    return AdoptionSource.UNKNOWN


# --------------------------------------------------------------------------
# 2c. A SUCCESS LABEL MUST MEASURE ITS PAYLOAD, NOT ITS PASSAGE
# --------------------------------------------------------------------------


def payload_verdict(
    *,
    completed_local: int = 0,
    completed_synced: int = 0,
    matched: int = 0,
    loaded: int = 0,
    refused: int = 0,
) -> str:
    """What a prefetch-completion line is allowed to call itself.

    TWO RETRACTIONS ARE RECORDED HERE, both mine, because each was already on
    its way into a verdict when it was caught.

    FIRST: "18 of 18 lines carry an all-zero payload" came from a FOUR-LINE
    SAMPLE that happened to hit only zeros. Sampling a log is not measuring
    it. The full count: 7 of 18 empty, 11 carrying ``completed_local``
    3072-8192.

    SECOND, and this is the one that matters: I then read ``matched>0`` with
    ``loaded=0`` as a defect signal and called it ``matched_not_loaded``,
    17/18. That is a FALSE POSITIVE in 14 of the 18, and the refutation is in
    the emitter's own arithmetic three lines above the log call
    (``unified_radix_cache.py``): ``matched`` is ``insert_result.prefix_len``
    -- WHAT THE TREE ALREADY HELD -- and ``loaded`` is
    ``min_completed_tokens - prefix_len``. So ``matched == completed_synced``
    forces ``loaded == 0`` as the identity x-x=0, and the emitter says so in
    as many words: "Arithmetic, not a defect."

    This is the INSTRUMENT-TEXT-LUEGT class B failure exactly -- the
    conclusion was drawn from a CORRECT line, and the counter-check is to
    read the line above it. I did not, and the axis I built on it would have
    reported a healthy tree as broken 14 times out of 18.

    THE REAL PARTITION OF THE MEASURED 18, which the corrected predicate
    reproduces: 7 storage-miss, 7 arithmetic, 3 GENUINE refusal (#841
    ``host_span_unclaimed``), 1 genuine load.

      ``delivered``     ``loaded > 0``. Bytes arrived. 1/18.
      ``refused``       ``refused > 0``: the tail WAS fetched and the tree
                        declined to adopt it (#841 contiguous-backup law).
                        THE ONLY READ-SIDE DEFECT ON THIS LINE. 3/18.
      ``storage_miss``  ``completed_synced == 0``: nothing came back from the
                        storage tier at all. Write-side or routing. 7/18.
      ``arithmetic``    completed, not refused, and the tree already held the
                        whole fetched span. ``loaded=0`` is the identity, NOT
                        a defect. 7/18 -- and never again a finding.
      ``no_completion`` the operation did not complete. Not a result at all.

    ``matched`` is deliberately NOT a discriminator anywhere above. It is the
    column that produced the false positive, and it earns no vote.
    """
    if int(loaded) > 0:
        return "delivered"
    if int(refused) > 0:
        return "refused"
    if int(completed_synced) == 0:
        if int(completed_local) > 0:
            return "storage_miss"
        return "no_completion"
    return "arithmetic"


# --------------------------------------------------------------------------
# 2d. THE FIELD THAT BELONGS ON THE PREFILL LINE
# --------------------------------------------------------------------------


def prefill_provenance_field(hit_tokens: int) -> str:
    """The producer-phase field for the ``Prefill batch`` line.

    WHY THIS IS NOT A FORMALITY -- it stopped a false win on 2026-09-01. That
    boot showed three TP-phase prefills at ``#cached-token: 16384``, which
    reads exactly like the mission succeeding. The identical 4096/4096/16384
    pattern stands in the SAME FILE in the PP phase before any flip: it is
    in-phase chunked continuation. ``phase=tp`` on that line names the phase
    that CONSUMED the tokens; nothing on it names the phase that PRODUCED
    them, so the two worlds are byte-identical to a reader. The acceptance
    clause "ok>0 WITH producer phase" is the only guard against that
    inference, and the guard has to sit on the line the reader actually
    reads.

    Same discipline as ``_active_phase_field`` beside it: silent when there is
    nothing to say, so a non-flip boot's line stays byte-identical.
    """
    if not hit_tokens:
        return ""
    if not _phase_flip_enabled():
        # With one layout the question is meaningless. SAME AUTHORITY as
        # `_active_phase_field`: the server arg the routing itself consults,
        # so the label cannot drift from the behaviour it describes.
        return ""
    if _current_generation_or_none() == "-":
        return ""
    parts = []
    for phase in (
        ProducerPhase.CROSS_PHASE,
        ProducerPhase.SAME_PHASE,
        ProducerPhase.UNKNOWN,
    ):
        n = _prefill_producer_tokens.get(phase.value, 0)
        if n:
            parts.append(f"{phase.value}:{n}")
    # #1061: the line that renders the window DRAINS it (the "drained by the
    # prefill line" contract below). Draining only on the render path -- after
    # every early return above -- means a line with zero cached tokens leaves
    # the window accumulating for the next rendering line instead of silently
    # dropping attribution that belongs to a batch not yet printed.
    _prefill_producer_tokens.clear()
    if not parts:
        # Armed but nothing attributed: say NO_OBSERVATION, never omit the
        # field and never print a zero. An absent field reads as "no cached
        # tokens"; a zero reads as "measured, none cross-phase".
        return ", #cached-producer: NO_OBSERVATION"
    return ", #cached-producer: " + ",".join(parts)


#: Per-batch attribution of ``#cached-token``, filled by the match walk and
#: drained by the prefill line. Reset per batch by ``reset_prefill_window``.
_prefill_producer_tokens: dict[str, int] = {}


def note_prefill_hit_tokens(tokens: int, producer: ProducerPhase) -> None:
    if not tokens:
        return
    k = ProducerPhase(producer).value
    _prefill_producer_tokens[k] = _prefill_producer_tokens.get(k, 0) + int(tokens)


def reset_prefill_window() -> None:
    _prefill_producer_tokens.clear()


def reset_for_test() -> None:
    """Clear all module state. Tests only."""
    global _ledger_dropped, _ledger_writes, _emitted, _suppressed
    global _late_arrivals, _arrivals, _consults, _consult_misses
    global _dpc, _dpc_emitted, _dpc_suppressed, _dpc_over_bound_seen
    global _dpc_fence_pending, _dpc_fence_seed
    _dpc = None
    _dpc_emitted = 0
    _dpc_suppressed = 0
    _dpc_over_bound_seen = 0
    _dpc_fence_pending = 0
    _dpc_fence_seed = 0
    with _gen_lock:
        _gen_phases.clear()
    with _ledger_lock:
        _ledger.clear()
        _ledger_dropped = 0
        _ledger_writes = 0
    with _order_lock:
        _missed_consults.clear()
        _arrived.clear()
        _late_arrivals = 0
        _arrivals = 0
        _consults = 0
        _consult_misses = 0
    _prefill_producer_tokens.clear()
    _emitted = 0
    _suppressed = 0


# --------------------------------------------------------------------------
# 3. the per-window census
# --------------------------------------------------------------------------


@dataclasses.dataclass
class ProducerPhaseCensus:
    """Accepted context, partitioned by where it was produced.

    Token counts are KEY tokens, the same frame ``MatchRefusalCensus`` uses, so
    the two lines can be read against each other without a unit conversion.
    """

    #: Match walks in this window that reached a verdict at all. THE
    #: DENOMINATOR, carried on the same line as the numerator so it never has
    #: to be reconstructed from a second log (#873).
    walks: int = 0
    #: Walks whose accepted prefix was non-empty (a HIT, by the #904 verdict).
    hit_walks: int = 0
    #: Of the hit walks, those carrying at least one CROSS_PHASE token.
    #: THE MISSION'S NUMERATOR.
    cross_phase_walks: int = 0
    #: Accepted key tokens by producer phase.
    tokens_by_producer: dict[str, int] = dataclasses.field(default_factory=dict)
    #: Accepted key tokens by the arm that adopted them.
    tokens_by_source: dict[str, int] = dataclasses.field(default_factory=dict)
    #: True once anything at all was recorded. Guards NO_OBSERVATION.
    observed: bool = False

    # -- recording ------------------------------------------------------

    def note_walk(self, hit: bool) -> None:
        """One completed match walk. Called for EVERY walk, hit or not.

        The denominator is recorded by the same call that records the
        numerator's opportunity, so the two cannot drift.
        """
        self.observed = True
        self.walks += 1
        if hit:
            self.hit_walks += 1

    def note_accepted_tokens(
        self,
        tokens: int,
        producer: ProducerPhase,
        source: AdoptionSource = AdoptionSource.UNKNOWN,
    ) -> None:
        self.observed = True
        t = int(tokens)
        pk = ProducerPhase(producer).value
        sk = AdoptionSource(source).value
        self.tokens_by_producer[pk] = self.tokens_by_producer.get(pk, 0) + t
        self.tokens_by_source[sk] = self.tokens_by_source.get(sk, 0) + t

    def note_cross_phase_walk(self) -> None:
        self.observed = True
        self.cross_phase_walks += 1

    # -- reading --------------------------------------------------------

    @property
    def cross_phase_tokens(self) -> int:
        return self.tokens_by_producer.get(ProducerPhase.CROSS_PHASE.value, 0)

    @property
    def accepted_tokens(self) -> int:
        return sum(self.tokens_by_producer.values())

    def state(self) -> ObservationState:
        if not self.observed:
            return ObservationState.NO_OBSERVATION
        if self.walks == 0:
            return ObservationState.EMPTY
        return ObservationState.VALUE

    def check_partition(self) -> None:
        """The parts must sum, and the numerator must fit its denominator."""
        by_producer = sum(self.tokens_by_producer.values())
        by_source = sum(self.tokens_by_source.values())
        if by_producer != by_source:
            raise ValueError(
                f"producer-phase census does not partition: by_producer="
                f"{by_producer} != by_source={by_source}"
            )
        if self.hit_walks > self.walks:
            raise ValueError(
                f"producer-phase census numerator exceeds denominator: "
                f"hit_walks={self.hit_walks} > walks={self.walks}"
            )
        if self.cross_phase_walks > self.hit_walks:
            raise ValueError(
                f"producer-phase census cross-phase exceeds hits: "
                f"cross_phase_walks={self.cross_phase_walks} > "
                f"hit_walks={self.hit_walks}"
            )

    def log_fields(self) -> dict[str, object]:
        """Flat fields for ONE line. Every number beside its denominator.

        Field names are prefixed so a bare substring grep cannot bind them to
        a neighbouring line's field: this strand has already been bitten by
        ``phase=tp`` matching ``owner_phase=tp``.
        """
        self.check_partition()
        st = self.state()
        stats = ledger_stats()
        order = arrival_stats()
        return {
            "state": st.value,
            # THE ANSWER and THE DENOMINATOR, adjacent and never separable.
            "ok": self.cross_phase_walks,
            "denom": self.walks,
            "hit_walks": self.hit_walks,
            "cross_tokens": self.cross_phase_tokens,
            "acc_tokens": self.accepted_tokens,
            "by_producer": _render(self.tokens_by_producer),
            "by_source": _render(self.tokens_by_source),
            "cur_gen": _current_generation_or_none(),
            "cur_phase": _bound_phase_or_none(),
            "gen_hist": ";".join(f"{g}:{p}" for g, p in generation_history()) or "-",
            # ORDERING: "never came" and "came too late" are two fixes.
            "late": order["late"],
            "never_yet": order["never_yet"],
            "consult_misses": order["misses"],
            "arrivals": order["arrivals"],
            "ledger_keys": stats["keys"],
            "ledger_dropped": stats["dropped"],
            "suppressed": _suppressed,
        }

    def format_line(self, prefix: str = ACCEPT_LINE_PREFIX) -> str:
        body = " ".join(f"{k}={v}" for k, v in self.log_fields().items())
        return f"[{prefix}] {body}"


def _render(d: dict[str, int]) -> str:
    if not d:
        return "-"
    return ",".join(f"{k}:{v}" for k, v in sorted(d.items()))


def _current_generation_or_none():
    try:
        from sglang.srt.mem_cache.hicache_phase_binding import current_generation

        return current_generation()
    except Exception:  # pragma: no cover - unit tests run without sglang
        return "-"


def _phase_flip_enabled() -> bool:
    """True only when the flip is armed. Fails CLOSED: an unreadable server
    arg means "do not change the line", never "assume the flip is on"."""
    try:
        from sglang.srt.runtime_context import get_server_args

        return bool(getattr(get_server_args(), "enable_phase_flip", False))
    except Exception:  # pragma: no cover - no runtime context in unit tests
        return False


def _bound_phase_or_none():
    try:
        from sglang.srt.mem_cache.hicache_phase_binding import bound_phase

        return bound_phase()
    except Exception:  # pragma: no cover - unit tests run without sglang
        return "-"


# --------------------------------------------------------------------------
# 3b. THE WIRING GLUE (#1061) -- the writers the recording sites call
# --------------------------------------------------------------------------
#
# Built 2026-08-31, wired 2026-09-01 (#1061): until these existed the module
# had ZERO production writers, so even a successful boot could not produce the
# acceptance line (built-never-wired). Each function is a no-op while the
# census is disarmed (SGLANG_MATCH_REFUSAL_CENSUS_EVERY=0, the shared #904
# knob), so the default path builds and records nothing.


def note_backup_keys(keys, generation) -> None:
    """The store-write ledger's writer, called by the backup thread.

    ARITHMETIC: one ledger entry per storage key handed to the backend;
    ``generation`` is ``operation.binding_generation`` -- the #719 stamp the
    operation was OPENED under. That stamp is the EXISTING provenance carrier
    (stamped in ``StorageOperation.__init__``); no second scheme is invented
    here. Disarmed, or an unstamped operation (generation None) -> records
    nothing; an unstamped key later classifies UNKNOWN, never same/cross.
    """
    if census_armed() <= 0 or not keys or generation is None:
        return
    for k in keys:
        note_store_write(k, generation)


def note_prefetch_adopted(keys) -> None:
    """The arrival side's writer: these keys' bytes were ADOPTED into the
    tree via the PREFETCH arm (store -> host tier -> tree).

    ARITHMETIC: one ``note_arrival`` per adopted key -- the caller passes only
    the keys the insert actually adopted (matched head excluded, declined
    insert excluded), so ``arrivals`` counts adoptions, not fetch attempts.
    Disarmed -> records nothing.
    """
    if census_armed() <= 0 or not keys:
        return
    for k in keys:
        note_arrival(k)


def note_walk_node(census, keys, key_tokens, page_size, accepted) -> bool:
    """One walked tree node, classified against the ledger. Returns True when
    any accepted token classified CROSS_PHASE, so the caller can count the
    walk as a mission hit.

    ARITHMETIC, field by field:
      * per-key tokens = min(page_size, key_tokens - i*page_size): storage
        keys are page-granular and ``key_tokens`` is this node's matched KEY
        tokens, so the last key may carry a partial page and the per-key
        tokens sum to exactly ``key_tokens``.
      * ``keys`` falsy (a device-only node that was never backuped carries
        ``hash_value=None``): there is NO storage-key carrier for this node,
        so its accepted tokens are classified UNKNOWN in one lump -- never
        guessed into same/cross (the gap stays visible as ``unknown:N``).
      * producer = ``producer_phase_of(key, current_generation())``: the
        write-ledger stamp against the #719 generation consuming right now.
      * source = ``adoption_source_of(key)``: PREFETCH / BACKUP_HOST /
        UNKNOWN, from the arrival record and the write ledger.
      * consults: every walked key is consulted (hit or miss), feeding the
        late/never ordering partition.

    Accepted tokens ALSO feed the per-batch prefill-line window
    (``note_prefill_hit_tokens``), so the ``#cached-producer`` field on the
    ``Prefill batch`` line fills from the same classification -- one
    arithmetic, two lines that cannot disagree.
    """
    if census is None:
        return False
    tokens = int(key_tokens or 0)
    if tokens <= 0:
        return False
    if not keys:
        if accepted:
            census.note_accepted_tokens(
                tokens, ProducerPhase.UNKNOWN, AdoptionSource.UNKNOWN
            )
            note_prefill_hit_tokens(tokens, ProducerPhase.UNKNOWN)
        return False
    try:
        from sglang.srt.mem_cache.hicache_phase_binding import current_generation

        cgen = int(current_generation())
    except Exception:  # pragma: no cover - unit tests run without sglang
        cgen = None
    page = max(1, int(page_size or 1))
    saw_cross = False
    for i, k in enumerate(keys):
        t = min(page, tokens - i * page)
        if t <= 0:
            break
        note_consult(k, accepted=bool(accepted))
        if not accepted:
            continue
        producer = producer_phase_of(k, cgen)
        census.note_accepted_tokens(t, producer, adoption_source_of(k))
        note_prefill_hit_tokens(t, producer)
        if producer is ProducerPhase.CROSS_PHASE:
            saw_cross = True
    return saw_cross


# --------------------------------------------------------------------------
# 4. arming and emission
# --------------------------------------------------------------------------
#
# Off by default. When off the census object is not built at all, so the
# instrumented path is not entered rather than entered and discarded.

_emitted = 0
_suppressed = 0


def census_armed() -> int:
    """0 = disarmed. N = emit the acceptance line every N walks.

    Shares the #904 knob DELIBERATELY: the producer-phase line is only
    readable beside the refusal line (a cross-phase ok=0 next to refused>0 is
    a different finding from ok=0 next to not_present>0), and two knobs would
    let the two lines drift apart in one log. One knob, both lines.
    """
    try:
        from sglang.srt.environ import envs

        return int(envs.SGLANG_MATCH_REFUSAL_CENSUS_EVERY.get())
    except Exception:  # pragma: no cover - env shape varies in unit tests
        return 0


def new_producer_census() -> ProducerPhaseCensus | None:
    return ProducerPhaseCensus() if census_armed() > 0 else None


def emit(census: ProducerPhaseCensus | None, logger) -> bool:
    """Log the census, rate-limited, and ALWAYS on a cross-phase window.

    A cross-phase hit is the finding this instrument exists for, so it is
    never sampled away. The periodic emission exists to give the log its
    denominator, which a hit-only stream would not have. The count of
    SUPPRESSED emissions rides the line, because an absence after the first N
    lines of a rate-limited emitter is not a zero (DENOMINATOR LAW, measured
    2026-08-31).

    #1061: returns True when a line actually went out (the broken-partition
    error line included -- it is an emission, just a self-indicting one), so
    the wiring can start a fresh window per emitted line and ``ok``/``denom``
    on each line describe the walks SINCE the previous line, not since boot.
    False = suppressed or disarmed; the caller keeps accumulating.
    """
    global _emitted, _suppressed
    if census is None or not census.observed:
        return False
    every = census_armed()
    if every <= 0:
        return False
    _emitted += 1
    if census.cross_phase_walks > 0 or _emitted % every == 0:
        try:
            logger.info("%s", census.format_line())
        except ValueError as exc:
            logger.error(
                "[%s] BROKEN PARTITION -- the instrument is miscounting, do "
                "not read its verdict: %s",
                ACCEPT_LINE_PREFIX,
                exc,
            )
        return True
    _suppressed += 1
    return False


# --------------------------------------------------------------------------
# 5. THE SECOND HALF OF THE ACCEPTANCE: #939, no double prefill
# --------------------------------------------------------------------------
#
# THE LAW, verbatim from the user: a double prefill is "natuerlich nicht"
# acceptable, and the admissible residual loss is AT MOST ONE HICACHE CHUNK
# SIZE. Until now that was a prose condition -- there was no line that could
# pass or break it, so a green producer-phase number could sit next to a
# silent double prefill and read as a win. That is the same false-win family
# this module already closed once.
#
# OPERATIONAL DEFINITION, on quantities the code CARRIES rather than ones
# derived here:
#
#   S_i  the span request i had ALREADY COMPUTED when the cutover retracted
#        it. The retraction stashes the live Req objects
#        (`phase_flip_runtime.py:8889`, `self._pending_seam_readmit`), so
#        this is read off the request at that instant, not reconstructed.
#   C_i  what its post-cutover re-prefill RECOVERED from cache:
#        ``len(req.prefix_indices)``, the same quantity the scheduler uses at
#        `scheduler.py:5260` (`_matched_len`) and the same one that reaches
#        the log as this request's share of ``#cached-token``.
#
#   recomputed_i = max(0, S_i - C_i)
#
# That is the double prefill, in tokens, per request: work that was done
# before the cutover and is being done again after it. The denominator is
# sum(S_i) -- the span that was already computed and therefore COULD have
# been recomputed. Without it a token count is not a finding.
#
# WHY THE POPULATION IS THE READMIT SET AND NOTHING ELSE: a request that was
# never retracted was never at risk of a double prefill, and including it
# would inflate the denominator with requests that cannot contribute to the
# numerator -- the denominator trap in its classic form.
#
# THE THRESHOLD IS READ, NEVER HARDCODED. The bound is one chunk C, and the
# code that owns the arithmetic says so at `scheduler.py:5261-5286`:
#
#       realised loss = L - floor((L-1)/C) * C   <=   C   for every L
#       worked there:  L=4618, C=4096 -> 522;  L=8192, C=4096 -> 4096
#
# The bound is ATTAINED and never exceeded, so the comparison is ``<=``, and
# a specimen that loses exactly C must PASS while C+1 must FAIL. That same
# comment warns that ``dynamic_chunked_prefill_size`` can vary C at runtime
# -- "which moves the number but not the 'at most one chunk' bound". A
# constant captured here would be the next number that was calibrated once
# and then lied; `resolve_chunk_size` reads the live value and REPORTS WHICH
# SOURCE ANSWERED, so a threshold with no stated source is visibly unusable.


def resolve_chunk_size(scheduler=None) -> tuple[int | None, str]:
    """The live chunk size C, WITH the source that produced it.

    Order: the scheduler's dynamic property (what actually chunks a prefill
    right now), then its static field, then the server arg. None means the
    threshold is unknown -- which makes the verdict NO_OBSERVATION, never a
    PASS against a guessed bound.
    """
    if scheduler is not None:
        try:
            v = scheduler.dynamic_chunked_prefill_size
            if v and int(v) > 0:
                return int(v), "scheduler.dynamic_chunked_prefill_size"
        except Exception:  # noqa: BLE001 - a missing property is not a crash
            pass
        try:
            v = scheduler.chunked_prefill_size
            if v and int(v) > 0:
                return int(v), "scheduler.chunked_prefill_size"
        except Exception:  # noqa: BLE001
            pass
    try:
        from sglang.srt.runtime_context import get_server_args

        v = getattr(get_server_args(), "chunked_prefill_size", None)
        if v and int(v) > 0:
            return int(v), "server_args.chunked_prefill_size"
    except Exception:  # noqa: BLE001 - no runtime context in unit tests
        pass
    return None, "UNRESOLVED"


@dataclasses.dataclass
class DoublePrefillCensus:
    """Re-computed tokens across a cutover, against what was already computed.

    One census per cutover. A boot with several flips produces several, and
    the acceptance is the WORST of them: the law is per-request, so a single
    request losing more than a chunk breaks it no matter how many did not.
    """

    #: Requests the cutover retracted and re-admitted. THE DENOMINATOR's
    #: population -- never all requests, only those that were at risk.
    readmitted: int = 0
    #: sum(S_i): tokens that were already computed before the cutover.
    already_computed: int = 0
    #: sum(max(0, S_i - C_i)): tokens being computed a second time.
    recomputed: int = 0
    #: max over requests of recomputed_i. THE TERM THE LAW BOUNDS.
    worst_request_tokens: int = 0
    #: Which request attained the worst, so a failure is actionable.
    worst_request_id: str = "-"
    #: Requests whose own loss exceeded one chunk.
    over_bound: int = 0
    chunk_size: int | None = None
    chunk_source: str = "UNRESOLVED"
    observed: bool = False
    #: #1068 (G9): cutovers of THIS wave that proceeded past the
    #: `_WRITEBACK_DEFER_LIMIT` with an incomplete write-back fence. Such a
    #: wave may miss the store and recompute in full for a reason that is
    #: neither a store defect nor a read-path defect; the term is carried
    #: beside the bound so `within_bound=false` stays attributable.
    fence_proceeds: int = 0

    def note_fence_proceed(self) -> None:
        self.fence_proceeds += 1

    def note_readmitted_request(
        self,
        request_id: str,
        already_computed: int,
        recovered: int,
    ) -> None:
        """One re-admitted request, with both carried quantities.

        ``recovered`` is ``len(req.prefix_indices)`` at the post-cutover
        re-prefill. It is passed in rather than read here because this object
        holds no reference to a request -- the census is a passive recorder
        and must not be able to keep a Req alive.
        """
        self.observed = True
        self.readmitted += 1
        s = max(0, int(already_computed))
        c = max(0, int(recovered))
        lost = max(0, s - c)
        self.already_computed += s
        self.recomputed += lost
        if lost > self.worst_request_tokens:
            self.worst_request_tokens = lost
            self.worst_request_id = str(request_id)
        if self.chunk_size is not None and lost > self.chunk_size:
            self.over_bound += 1

    def bind_chunk(self, scheduler=None) -> None:
        self.chunk_size, self.chunk_source = resolve_chunk_size(scheduler)

    def state(self) -> ObservationState:
        if not self.observed:
            return ObservationState.NO_OBSERVATION
        if self.chunk_size is None:
            # A bound we cannot name cannot be compared against. This is NOT
            # a pass, and it is not a failure either.
            return ObservationState.NO_OBSERVATION
        if self.readmitted == 0:
            return ObservationState.EMPTY
        return ObservationState.VALUE

    def within_bound(self) -> bool | None:
        """True/False, or None when there is nothing to decide.

        None is returned rather than a default True: a bound that was never
        evaluated must not be able to contribute a pass.
        """
        if self.state() is not ObservationState.VALUE:
            return None
        return self.worst_request_tokens <= int(self.chunk_size)

    def check_partition(self) -> None:
        if self.recomputed > self.already_computed:
            raise ValueError(
                f"#939 census does not partition: recomputed="
                f"{self.recomputed} > already_computed={self.already_computed}"
            )
        if self.worst_request_tokens > self.recomputed:
            raise ValueError(
                f"#939 census worst exceeds total: "
                f"{self.worst_request_tokens} > {self.recomputed}"
            )

    def log_fields(self) -> dict[str, object]:
        self.check_partition()
        w = self.within_bound()
        return {
            "state": self.state().value,
            "within_bound": "-" if w is None else str(bool(w)).lower(),
            # THE BOUNDED TERM and THE BOUND, adjacent and never separable.
            "worst": self.worst_request_tokens,
            "chunk": "-" if self.chunk_size is None else self.chunk_size,
            "chunk_src": self.chunk_source,
            "worst_req": self.worst_request_id,
            "over_bound": self.over_bound,
            # THE NUMERATOR and THE DENOMINATOR.
            "recomputed": self.recomputed,
            "already": self.already_computed,
            "readmitted": self.readmitted,
            # #1068 (G9, L10): the loss term that is NOT the read path's.
            "fence_proceeds": self.fence_proceeds,
        }

    def format_line(self, prefix: str = DOUBLE_PREFILL_LINE_PREFIX) -> str:
        body = " ".join(f"{k}={v}" for k, v in self.log_fields().items())
        return f"[{prefix}] {body}"


# --------------------------------------------------------------------------
# 5b. THE WIRING GLUE FOR #939 (#1047) -- the writer the recording site calls
# --------------------------------------------------------------------------
#
# Built with section 5, wired 2026-09-01 (#1047). Until this existed
# `DoublePrefillCensus` had ZERO production writers: the class, its
# arithmetic and its bound were all present and NOTHING ever called
# `note_readmitted_request`, so the second half of the acceptance ("no double
# prefill, under load, PROVABLE") could not be produced by any boot. Same
# built-never-wired shape #1061 closed for the producer-phase half.
#
# NO SECOND BOOKKEEPING. Both quantities are read off carriers that already
# exist and are already written on the seam path:
#
#   S_i = ``req.cached_prompt_tokens_at_retract``   schedule_batch.py:2607
#         Written by `Req.reset_for_retract`, whose own comment says it
#         records "WHAT WAS ALREADY COMPUTED, BEFORE THE FIELDS THAT SAY SO
#         ARE CLEARED THREE LINES DOWN". That is S_i by construction; nothing
#         else writes it and nothing clears it before the re-prefill.
#   C_i = ``pre_len``                               schedule_batch.py:4147
#         The per-chunk prefix length the extend is built against, i.e.
#         `len(req.prefix_indices)` after the read-through and after
#         `init_load_back` extended it (:4198-4199).
#
# THE POPULATION marker is likewise existing: ``phase_purity.SEAM_READMIT_ATTR``
# (`seam_readmit_epoch`), stamped ONLY by the #856 cutover retraction
# (`phase_flip_runtime.py:1973`) and by nothing else -- so an OOM-preempted
# request's ordinary re-prefill, which is real workload and NOT a double
# prefill, cannot enter the denominator. The three sibling probes (#969B,
# #969C, #1060) key on the identical attribute; a fourth definition of "the
# readmit population" would be exactly the drift this module exists to stop.

_dpc: DoublePrefillCensus | None = None
_dpc_emitted = 0
_dpc_suppressed = 0
_dpc_over_bound_seen = 0
#: #1154: has THIS wave's census emitted a line yet?
#:
#: The rate limit alone made the whole instrument unreachable. `_dpc_emitted`
#: is a process-lifetime counter and the gate was `_dpc_emitted % every == 0`
#: with `every` = the shared #904 knob, default 64 -- so the FIRST #939 line
#: of a boot needed the 64th recording call. A boot whose seam re-admits a
#: handful of requests per cutover never reaches it: measured on
#: boot_855_weg1b2 (2026-09-02), two readmit waves, ZERO
#: '[#939 double-prefill]' lines, and the acceptance read that absence as
#: "half B is not wired" rather than as a rate limit it could not clear.
#: The section-5 contract is ONE CENSUS PER CUTOVER, so the honest floor is
#: one LINE per cutover wave: the first observation after a reset always
#: emits, and the periodic sampling plus the always-on breach rule are kept
#: unchanged on top of it.
_dpc_wave_emitted = False
# #1068 (G9): fence proceeds noted since the last reset (this cutover's
# arm/preflight), and the seed the NEXT census is created with. ORDER in
# `PhaseFlipRuntime._execute_body`: fence verdict -> cutover -> reset (ends the
# previous wave's census) -> readmit (creates this wave's census). A proceed
# noted before the reset therefore belongs to the wave AFTER it, never to the
# census being closed; the reset moves pending -> seed, creation consumes the
# seed. A seed nobody consumed (a cutover that re-admitted nothing) is
# overwritten by the next reset, so it cannot leak into a later wave.
_dpc_fence_pending = 0
_dpc_fence_seed = 0


def note_fence_proceed() -> None:
    """#1068 (G9): the flip proceeded past `_WRITEBACK_DEFER_LIMIT` with an
    incomplete write-back fence. Counted for the wave this cutover re-admits."""
    global _dpc_fence_pending
    _dpc_fence_pending += 1


def note_double_prefill(
    request_id,
    already_computed,
    recovered,
    scheduler=None,
) -> None:
    """Record ONE re-admitted request's re-prefill against what it had.

    ARITHMETIC, field by field -- all of it lives in
    ``DoublePrefillCensus.note_readmitted_request``; this function adds no
    arithmetic of its own and exists only to own the census's LIFETIME:

      * ``already_computed`` -> S_i, the caller passes
        ``req.cached_prompt_tokens_at_retract``.
      * ``recovered``        -> C_i, the caller passes this chunk's
        ``pre_len``.
      * the census is created on FIRST observation and its chunk bound is
        bound THEN, from the live scheduler (`resolve_chunk_size`), so the
        threshold is the one in force at the cutover being measured rather
        than one captured at import.

    Disarmed (the shared #904 knob) -> returns immediately and builds no
    census, so the default path is byte-identical: no object, no counters, no
    line. A DISARMED run and a run with no double prefill are therefore NOT
    the same state, and `emit_double_prefill` prints which one it is.
    """
    global _dpc, _dpc_fence_seed
    if census_armed() <= 0:
        return
    if _dpc is None:
        _dpc = DoublePrefillCensus()
        _dpc.bind_chunk(scheduler)
        # #1068 (G9): the fence proceeds this wave was re-admitted under.
        _dpc.fence_proceeds = int(_dpc_fence_seed)
        _dpc_fence_seed = 0
    _dpc.note_readmitted_request(request_id, already_computed, recovered)


def emit_double_prefill(logger) -> bool:
    """Log the #939 acceptance line, rate-limited, ALWAYS on a breach.

    Mirrors `emit` deliberately, including its reasons:

      * a request that broke the one-chunk bound is THE finding this
        instrument exists for, so it is never sampled away -- the emission is
        forced whenever ``over_bound`` has grown since the last line;
      * the periodic emission supplies the denominator a breach-only stream
        would lack;
      * ``suppressed`` rides the line, because an absence after the first N
        lines of a rate-limited emitter is NOT a zero (DENOMINATOR LAW).

    Returns True when a line went out. A broken partition is reported as an
    error line and still counts as an emission -- it is self-indicting, not
    silent.

    NOT DRAINED on emission, unlike the producer-phase window: the contract in
    section 5 is ONE CENSUS PER CUTOVER and the acceptance is the WORST
    request of that cutover, so ``worst`` must be monotone across the
    cutover's whole readmit wave. `reset_double_prefill_census` is what ends
    a census, and only the cutover calls it.
    """
    global _dpc_emitted, _dpc_suppressed, _dpc_over_bound_seen, _dpc_wave_emitted
    census = _dpc
    if census is None or not census.observed:
        return False
    every = census_armed()
    if every <= 0:
        return False
    breach = census.over_bound > _dpc_over_bound_seen
    _dpc_over_bound_seen = census.over_bound
    _dpc_emitted += 1
    # FIRST OF THE WAVE ALWAYS GOES OUT (#1154). Without this the denominator
    # the periodic arm is supposed to supply never arrives on a low-volume
    # boot, and the acceptance cannot tell "nothing was recomputed" from
    # "the emitter never cleared its own rate limit".
    first_of_wave = not _dpc_wave_emitted
    if breach or first_of_wave or _dpc_emitted % every == 0:
        _dpc_wave_emitted = True
        try:
            logger.warning("%s suppressed=%d", census.format_line(), _dpc_suppressed)
        except ValueError as exc:
            logger.error(
                "[%s] BROKEN PARTITION -- the instrument is miscounting, do "
                "not read its verdict: %s",
                DOUBLE_PREFILL_LINE_PREFIX,
                exc,
            )
        return True
    _dpc_suppressed += 1
    return False


def reset_double_prefill_census() -> None:
    """End the current census. Called by the cutover, per the section-5
    contract 'one census per cutover'.

    Deliberately does NOT emit: the emission is the recording site's, so a
    cutover that retracted nothing cannot manufacture a line about a wave
    that never happened.
    """
    global _dpc, _dpc_over_bound_seen, _dpc_fence_pending, _dpc_fence_seed
    global _dpc_wave_emitted
    _dpc = None
    _dpc_over_bound_seen = 0
    # #1154: the next wave starts owing a line again. Reset here and NOT at
    # census creation, because `reset_double_prefill_census` is the only thing
    # that ends a census (the docstring above), so it is the only place where
    # "a new cutover wave begins" is actually known.
    _dpc_wave_emitted = False
    # #1068 (G9): the proceeds noted for THIS cutover seed the wave that
    # follows this reset; an unconsumed seed from a wave that never came is
    # overwritten here, never accumulated.
    _dpc_fence_seed = _dpc_fence_pending
    _dpc_fence_pending = 0


def double_prefill_census():
    """The live census, or None. Tests and callers that need to assert on the
    state read it here rather than reaching for the module global."""
    return _dpc
