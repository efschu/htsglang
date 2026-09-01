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
neighbouring prefetch path (``unified_radix_cache.py``, the ``_rehomed``
release route). A second phase-identity scheme beside it would be exactly the
second bookkeeping the upstream-minimal law forbids.

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
    "arrival_stats",
    "census_armed",
    "emit",
    "ledger_stats",
    "new_producer_census",
    "note_arrival",
    "note_consult",
    "note_generation",
    "note_store_write",
    "payload_verdict",
    "phase_of_generation",
    "prefill_provenance_field",
    "producer_generation_of",
    "producer_phase_of",
    "reset_for_test",
]


#: The literal that anchors every acceptance grep. Deliberately long and
#: deliberately NOT a bare ticket number: a four-digit number matches every
#: millisecond figure in a boot log (#995 / measured on boot 49).
ACCEPT_LINE_PREFIX = "#631 producer-phase"


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
# 2b. ARRIVAL ORDER: "it never came" and "it came too late" are two fixes
# --------------------------------------------------------------------------
#
# Measured on the booted commit: the MATCH runs ~1 s BEFORE the prefetch
# completes. A hit counter that cannot say whether the fetch had landed when
# the walk consulted the key reports one number for two different defects --
# a store that never produced the page (write-side) and a store that produced
# it after the only reader looked (ordering). Their fixes are in different
# files pointing in opposite directions, which is the #913 lesson applied to
# the time axis instead of the blame axis.

_order_lock = threading.Lock()
_missed_consults: dict[str, int] = {}  # key -> count of consults that missed
_late_arrivals = 0
_arrivals = 0
_consults = 0
_consult_misses = 0


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

    Measured on the booted commit: 18 lines reading ``HiCache prefetch
    success`` with ``completed_local=0 completed_synced=0 matched=0 loaded=0
    refused=0`` -- on exactly the rids that had been given ``verdict=issued``.
    The word "success" there describes the code REACHING the line, not
    anything having been transferred: the catalogued
    success-value-without-action class in its pure form.

    Four verdicts, because the three non-success worlds are different bugs:

      ``delivered``    payload moved: something was loaded or matched.
      ``refused``      nothing moved and a refusal was recorded -- the store
                       had an opinion. Read-side defect.
      ``empty``        nothing moved, nothing refused, and the operation did
                       complete. Nothing was there to move. Write-side.
      ``no_completion``  the operation did not even complete. Not a result at
                       all; a line claiming success here is claiming the
                       absence of a result as one.
    """
    if int(loaded) > 0 or int(matched) > 0:
        return "delivered"
    if int(refused) > 0:
        return "refused"
    if int(completed_local) > 0 or int(completed_synced) > 0:
        return "empty"
    return "no_completion"


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
    with _gen_lock:
        _gen_phases.clear()
    with _ledger_lock:
        _ledger.clear()
        _ledger_dropped = 0
        _ledger_writes = 0
    with _order_lock:
        _missed_consults.clear()
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


def emit(census: ProducerPhaseCensus | None, logger) -> None:
    """Log the census, rate-limited, and ALWAYS on a cross-phase window.

    A cross-phase hit is the finding this instrument exists for, so it is
    never sampled away. The periodic emission exists to give the log its
    denominator, which a hit-only stream would not have. The count of
    SUPPRESSED emissions rides the line, because an absence after the first N
    lines of a rate-limited emitter is not a zero (DENOMINATOR LAW, measured
    2026-08-31).
    """
    global _emitted, _suppressed
    if census is None or not census.observed:
        return
    every = census_armed()
    if every <= 0:
        return
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
    else:
        _suppressed += 1
