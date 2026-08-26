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

    def note_refused(self, component: str, tokens: int) -> None:
        """One component declined a node holding ``tokens`` key tokens.

        Several components may decline the SAME node; each is recorded, so
        the per-component numbers can exceed ``reached - accepted``. That is
        intentional -- the partition below is computed from ``reached`` and
        ``accepted``, never by summing this dict, precisely so a
        double-attributed node cannot break it.
        """
        self.observed = True
        key = str(component)
        self.refused_tokens_by_component[key] = self.refused_tokens_by_component.get(
            key, 0
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
        }

    def format_line(self, prefix: str = "#904 match-census") -> str:
        fields = self.log_fields()
        body = " ".join(f"{k}={v}" for k, v in fields.items())
        return f"[{prefix}] {body}"


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
