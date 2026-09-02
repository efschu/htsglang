"""#904 (g)/(h): why a prefix match returned nothing.

THE QUESTION THIS EXISTS TO ANSWER
----------------------------------
``#cached-token: 0`` is one number standing for three different worlds, and
two of them are defects:

  NOT_PRESENT   the walk found no stored continuation. Nothing was ever
                written, or it was written and the whole node is gone with no
                host backup to name it. Recompute is correct.
  DEAD          the walk reached a node that IS in the tree and stopped
                because the node is evicted with no backup -- the row was
                loaded, then invalidated, and the tree kept the shape without
                the bytes.
  REFUSED       the walk reached a node that HOLDS data and a component
                validator declined it. The row is resident and unmatchable:
                the load-then-invalidate half of #904, and the only one of
                the three that a capacity or bandwidth argument cannot
                explain.

#869b measured the WRITE side of this and closed it correctly for W40/W41:
``staged=0 acked=0`` on every one of 1959 flip-writeback fence lines means
the store was empty, so the read path was not reachable, let alone at fault
(``/spinning/gpu-arb/ANALYSE_869b_pp_tier_zero_hits.md``). What that
analysis names as its own residue is exactly what this module supplies: the
read path was never shown directly, only inferred from an empty store. On a
boot where the store is NOT empty, a zero hit is currently indistinguishable
between the three worlds above, and #873's "``cached_tokens == 0`` means
recomputed" stays unprovable for the same reason.

REFUSED is the discriminator. NOT_PRESENT and DEAD are both consistent with
"there was nothing to read". REFUSED is not: it says the bytes were there
and the match could not use them.

DESIGN NOTES
------------
Counts, never rates. A rate needs a denominator this object does not own,
and #873's whole finding one level up was a narrowed candidate set reading
as a decomposition. So the partition is exhaustive by construction --
``verdict()`` refuses to answer if the parts do not sum -- and the raw parts
travel with it.

The census is a passive recorder. It decides nothing, evicts nothing, and
holds no reference to a node; a walk that never feeds it produces
``NO_OBSERVATION`` rather than a plausible-looking zero. That distinction is
the #829/INDIKATOR-GESETZ rule: an instrument that cannot say "I did not
measure" is indistinguishable from one that measured nothing.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Dict, Optional, Tuple


class MatchOutcome(str, Enum):
    """Why the accepted prefix ended where it did."""

    NO_OBSERVATION = "no_observation"
    HIT = "hit"
    NOT_PRESENT = "not_present"
    DEAD = "dead"
    REFUSED = "refused"


#: Outcomes that mean "the bytes existed and the match could not use them".
#: The load-then-invalidate signature. NOT_PRESENT is deliberately absent:
#: it is the write-side null #869b already characterised.
RESIDENT_BUT_UNUSABLE = (MatchOutcome.DEAD, MatchOutcome.REFUSED)


@dataclasses.dataclass
class MatchRefusalCensus:
    """One prefix-match walk, recorded as an exhaustive partition.

    Token counts are KEY tokens, matching the frame ``cum_tokens`` uses in
    ``UnifiedRadixCache._match_prefix_helper`` -- not device rows, which
    differ for an evicted-but-backuped node and would make the parts stop
    summing.
    """

    #: Key tokens the walk actually traversed (every node it reached).
    reached_tokens: int = 0
    #: Of those, key tokens on nodes every validator accepted.
    accepted_tokens: int = 0
    #: Key tokens on nodes at least one validator declined, per component.
    refused_tokens_by_component: Dict[str, int] = dataclasses.field(
        default_factory=dict
    )
    #: The same tokens keyed by ``component:reason`` when the component can
    #: explain itself, and by ``component:unexplained`` when it cannot.
    #:
    #: #913/W42: naming the refusing COMPONENT was not enough to act on. The
    #: window census read ``refusers=MambaComponent:45`` on 671 of 675 walks,
    #: which is a component and a token count and no verdict: a node with no
    #: recurrent state (a write-side tombstone) and a node whose state sits
    #: off the checkpoint grid (a read-side determinism policy) produce the
    #: identical line, and their fixes are in different files pointing in
    #: opposite directions. An instrument that cannot separate the two is
    #: reporting the blame without the defect.
    refused_tokens_by_reason: Dict[str, int] = dataclasses.field(default_factory=dict)
    #: Key tokens the walk stopped short of because the node was evicted with
    #: no host backup. Zero when the walk ran to the end of the key.
    dead_tokens: int = 0
    #: True once anything at all was recorded. Guards NO_OBSERVATION.
    observed: bool = False

    # -- recording -------------------------------------------------------

    def note_reached(self, tokens: int) -> None:
        self.observed = True
        self.reached_tokens += int(tokens)

    def note_accepted(self, tokens: int) -> None:
        self.observed = True
        self.accepted_tokens += int(tokens)

    def note_refused(
        self, component: str, tokens: int, reason: Optional[str] = None
    ) -> None:
        """One component declined a node holding ``tokens`` key tokens.

        Several components may decline the SAME node; each is recorded, so
        the per-component numbers can exceed ``reached - accepted``. That is
        intentional -- the partition below is computed from ``reached`` and
        ``accepted``, never by summing this dict, precisely so a
        double-attributed node cannot break it.

        ``reason`` is the component's own account of WHICH of its conditions
        failed. ``None`` means the component was not asked or could not say,
        and is recorded as ``unexplained`` rather than dropped: a missing
        explanation is a fact about the instrument and has to be visible as
        one, not silently absent from a dict that otherwise reads as complete.
        """
        self.observed = True
        key = str(component)
        self.refused_tokens_by_component[key] = self.refused_tokens_by_component.get(
            key, 0
        ) + int(tokens)
        reason_key = f"{key}:{reason or 'unexplained'}"
        self.refused_tokens_by_reason[reason_key] = self.refused_tokens_by_reason.get(
            reason_key, 0
        ) + int(tokens)

    def note_dead_stop(self, tokens: int) -> None:
        """The walk hit ``evicted and not backuped`` and stopped."""
        self.observed = True
        self.dead_tokens += int(tokens)

    # -- reading ---------------------------------------------------------

    @property
    def refused_tokens(self) -> int:
        """Key tokens reached but not accepted. The DERIVED quantity.

        Not the sum of ``refused_tokens_by_component``: that dict attributes
        blame and may double-count one node across components. This is the
        partition term.
        """
        return max(0, self.reached_tokens - self.accepted_tokens)

    def verdict(self) -> MatchOutcome:
        if not self.observed:
            return MatchOutcome.NO_OBSERVATION
        if self.accepted_tokens > 0:
            return MatchOutcome.HIT
        if self.refused_tokens > 0:
            return MatchOutcome.REFUSED
        if self.dead_tokens > 0:
            return MatchOutcome.DEAD
        return MatchOutcome.NOT_PRESENT

    def is_resident_but_unusable(self) -> bool:
        """THE DISCRIMINATOR, in one call.

        True means: this zero hit is NOT explained by an empty store. Bytes
        were reachable in the tree and the match could not use them.
        """
        return self.verdict() in RESIDENT_BUT_UNUSABLE

    def check_partition(self) -> None:
        """The parts must sum. Raises rather than reporting a broken split.

        A census whose accepted + refused does not equal reached is measuring
        something other than what it names, and reporting it anyway is how a
        narrowed candidate set gets read as a decomposition (#873).
        """
        total = self.accepted_tokens + self.refused_tokens
        if total != self.reached_tokens:
            raise ValueError(
                f"match census does not partition: accepted="
                f"{self.accepted_tokens} + refused={self.refused_tokens} != "
                f"reached={self.reached_tokens}"
            )

    def top_refusers(self, limit: int = 3) -> Tuple[Tuple[str, int], ...]:
        items = sorted(
            self.refused_tokens_by_component.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
        return tuple(items[:limit])

    def top_reasons(self, limit: int = 3) -> Tuple[Tuple[str, int], ...]:
        """Same ranking over ``component:reason`` keys. See the field's note."""
        items = sorted(
            self.refused_tokens_by_reason.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
        return tuple(items[:limit])

    def log_fields(self) -> Dict[str, object]:
        """Flat fields for ONE log line. Stable keys, host-side only (#790).

        Deliberately shaped like the ``[#703 flip-writeback]`` fence line
        that made #869b decidable: every field a reader needs to classify the
        event is on the same line, so no cross-referencing of two logs is
        required to tell a write-side null from a read-side one.
        """
        self.check_partition()
        return {
            "verdict": self.verdict().value,
            "reached": self.reached_tokens,
            "accepted": self.accepted_tokens,
            "refused": self.refused_tokens,
            "dead": self.dead_tokens,
            "refusers": ",".join(f"{n}:{t}" for n, t in self.top_refusers()) or "-",
            "why": ",".join(f"{n}={t}" for n, t in self.top_reasons()) or "-",
        }

    def format_line(self, prefix: str = "#904 match-census") -> str:
        fields = self.log_fields()
        body = " ".join(f"{k}={v}" for k, v in fields.items())
        return f"[{prefix}] {body}"


#: #915: why a prefetch was not attempted, counted per reason.
#:
#: THE SECOND HALF OF THE SAME ZERO. A walk that matches nothing SHOULD fall
#: through to an L3 prefetch, so "the match refused" and "no prefetch was
#: attempted" are two different failures and only the first was instrumented.
#: The 0826 window attempted 264 prefetches against 675 sampled walks and the
#: remaining 411 left no trace at all -- not a counter, not a log line. Blame
#: with no defect, exactly as `refused_tokens_by_component` was before #914.
#:
#: Process-wide and unconditional, unlike the match census: this is one integer
#: increment on a path that already builds a RadixKey and takes a lock, so
#: there is nothing to arm and nothing to sample. A gate that only counts when
#: someone remembered to arm it cannot answer "was it ever tried".
PREFETCH_GATE_COUNTS: Dict[str, int] = {}

#: #1068 (slice 4): the decline terms `UnifiedRadixCache.prefetch_from_storage`
#: can record for ONE call, in ATTRIBUTION order -- `gate_reason_since` names
#: the first that tripped. The three gate terms come first (evaluated in that
#: sequence at the gate), then the exits behind the gate in path order:
#: host_pool_exhausted / host_alloc_failed (the non-symmetric alloc branch),
#: anchor_pool_exhausted (the #1035 site, a CAUSE counted before the exit it
#: precedes), vote_negative (the #580 group vote), alloc_failed_post_vote (the
#: alloc-failed exit after the components ran).
PREFETCH_DECLINE_ORDER = (
    "anchor",
    "too_short",
    "rate_limited",
    "host_pool_exhausted",
    "host_alloc_failed",
    "anchor_pool_exhausted",
    "vote_negative",
    "alloc_failed_post_vote",
)

#: #1068 (slice 4): the EXIT terms of `Scheduler._prefetch_kvcache`. Every
#: entry counts `intake` and every exit exactly one of these, so
#:
#:     intake == sum(PREFETCH_GATE_COUNTS[t] for t in PREFETCH_INTAKE_PARTITION)
#:
#: holds per rank at every instant on the flip boot form (tp_world_size == 1,
#: no #580 vote). Named exceptions, so the identity is never read wrong:
#: `anchor_pool_exhausted` is NOT in the sum -- it is the CAUSE counter of the
#: #1035 site and the same call then leaves through `alloc_failed_post_vote`
#: (or `vote_negative` under the vote), so
#: anchor_pool_exhausted <= alloc_failed_post_vote + vote_negative.
#: `host_pool_truncated` is not a decline and stands beside `issued`.
#: `already_in_flight` is decided after the tree ran, so the tree's own term
#: for that call (`attempted`, or a gate decline) is counted beside it. Under
#: the #580 vote (tp_world_size > 1) a rank whose own gate term declined ALSO
#: counts `vote_negative` (the group's exit), so there the sum over-counts by
#: exactly those calls. The A12.2 deferral keys (deferred, landed, ...) are a
#: separate partition of the rate_limited verdicts.
PREFETCH_INTAKE_PARTITION = (
    "issued",
    "storage_disabled",
    "store_absent",
    "anchor_no_vote",
    "unobservable",
    "already_in_flight",
    "anchor",
    "too_short",
    "rate_limited",
    "host_pool_exhausted",
    "host_alloc_failed",
    "vote_negative",
    "alloc_failed_post_vote",
    "attempted_but_unregistered",
    "unreported",
)


def note_prefetch_gate(reason: Optional[str], tokens: int = 0) -> None:
    """Record one prefetch-gate verdict. ``None`` means the prefetch ran.

    ``attempted`` is counted too, and not only the refusals. A denominator that
    has to be reconstructed from a different log is how #873's narrowed
    candidate set got read as a decomposition; the parts are kept here so the
    partition is checkable on one line.
    """
    key = "attempted" if reason is None else str(reason)
    PREFETCH_GATE_COUNTS[key] = PREFETCH_GATE_COUNTS.get(key, 0) + 1
    if tokens:
        tk = f"{key}_tokens"
        PREFETCH_GATE_COUNTS[tk] = PREFETCH_GATE_COUNTS.get(tk, 0) + int(tokens)


def format_prefetch_gate() -> str:
    """One line, stable keys, for a log-counter grep."""
    if not PREFETCH_GATE_COUNTS:
        return "[#915 prefetch-gate] no observation"
    body = " ".join(f"{k}={v}" for k, v in sorted(PREFETCH_GATE_COUNTS.items()))
    return f"[#915 prefetch-gate] {body}"


_gate_emitted = 0


def prefetch_gate_due() -> bool:
    """True on the cadence at which the #915 gate line should be logged.

    #915 SHIPPED HALF-WIRED AND STAYED THAT WAY FOR TWELVE DAYS.
    `note_prefetch_gate` has always been called (unified_radix_cache.py:2842),
    so the decline reason was RECORDED on every prefetch attempt -- but
    `format_prefetch_gate` had ZERO callers anywhere in the tree, so it was
    never REPORTED. The boot script printed that state at every window
    ("1 hit in its own file and 0 caller file(s) elsewhere -> STILL
    PRAESENT-ABER-UNVERDRAHTET") and it was read as a note rather than as the
    missing half it was.

    The cost was measured on window-946fix-0828: the #946 escape declined
    thousands of times and the reason -- anchor / too_short / rate_limited --
    was sitting in this module's counters the whole time, unreadable from the
    log. PRESENT-BUT-UNWIRED, the middle state of the three-state delivery
    rule, and the most expensive one in both directions.

    Shares `census_every()` so one env knob arms both and the two lines cannot
    drift apart in a log.
    """
    global _gate_emitted
    every = census_every()
    if every <= 0:
        return False
    _gate_emitted += 1
    return _gate_emitted % every == 0


def gate_snapshot() -> Dict[str, int]:
    """Copy of the gate counters, for a per-call DELTA.

    A running total cannot say which term declined THIS request, and the whole
    point of a decline reason is that it belongs to one request. Taken before
    the call, compared after.
    """
    return dict(PREFETCH_GATE_COUNTS)


def gate_reason_since(before: Dict[str, int]) -> str:
    """Which #915 term declined during the call bracketed by ``before``.

    Order is attribution order (`PREFETCH_DECLINE_ORDER`), matching the caller
    in `prefetch_from_storage` which evaluates the three gate terms in
    sequence and names the FIRST that fails; a request can trip several and
    summing them would double-count. #1068 (slice 4) extends the order with
    the exits BEHIND the gate -- host_pool_exhausted, host_alloc_failed,
    anchor_pool_exhausted, vote_negative, alloc_failed_post_vote -- each of
    which now counts itself at its own return.

    ``attempted_but_unregistered`` is the honest answer for "the gate let it
    through and it still never registered" and, since slice 4, means an exit
    that NO term counted: a new silent exit (the scheduler speaks it as L4).
    It is deliberately NOT called a success: the whole defect being fixed is a
    path that reported one.
    """
    for key in PREFETCH_DECLINE_ORDER:
        if PREFETCH_GATE_COUNTS.get(key, 0) > before.get(key, 0):
            return key
    if PREFETCH_GATE_COUNTS.get("attempted", 0) > before.get("attempted", 0):
        return "attempted_but_unregistered"
    return "unreported"


def classify(
    census: Optional[MatchRefusalCensus],
) -> MatchOutcome:
    """Verdict for a possibly-absent census. ``None`` is NOT a zero."""
    if census is None:
        return MatchOutcome.NO_OBSERVATION
    return census.verdict()


# --- arming -------------------------------------------------------------
#
# Off by default and the walk does not build the object at all when it is
# off, so the traced path is not entered rather than entered and discarded.

_emitted = 0


def census_every() -> int:
    """0 = disarmed. N = build a census and emit every Nth match."""
    try:
        from sglang.srt.environ import envs

        return int(envs.SGLANG_MATCH_REFUSAL_CENSUS_EVERY.get())
    except Exception:  # pragma: no cover - env shape varies in unit tests
        return 0


def new_match_census() -> Optional[MatchRefusalCensus]:
    return MatchRefusalCensus() if census_every() > 0 else None


def emit(census: Optional[MatchRefusalCensus], logger) -> None:
    """Log the census, rate-limited, and ALWAYS when it is the discriminator.

    A REFUSED verdict is the finding this instrument exists for, so it is
    never sampled away; the periodic emission exists to give the log a
    denominator, which a refusal-only stream would not have (#873: a count
    without its base reads as a decomposition).
    """
    global _emitted
    if census is None or not census.observed:
        return
    every = census_every()
    if every <= 0:
        return
    _emitted += 1
    if census.is_resident_but_unusable() or _emitted % every == 0:
        try:
            logger.info("%s", census.format_line())
        except ValueError:
            logger.error(
                "[#904 match-census] BROKEN PARTITION reached=%d accepted=%d "
                "dead=%d -- the instrument is miscounting, do not read its "
                "verdict",
                census.reached_tokens,
                census.accepted_tokens,
                census.dead_tokens,
            )
