# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""#822: ONE authority for the KV row ownership invariant.

WHY THIS MODULE EXISTS
======================

Seven crash roots in six days -- #684, #714, #717 -> #722, #744, #796, #814,
#816 -- are the same law broken in different places. Each was fixed where it
surfaced, and the next one surfaced somewhere else, because the law itself was
never written down anywhere a fourth call site could be held to it.

The law::

    exposed id space  <=  committed backing
    every claimed row  <   committed backing
    the claims partition the space: exactly one owner per row, no row unowned
    a phase cutover retires the whole old id space in ONE step

The first line already existed as :func:`exposure_over_backing`
(``kv_backing_relief.py:504``) and is reused verbatim here rather than
restated -- one definition of "over-exposed", per the #345/#352/#355 lesson.
The second line is the #722 direction, present in that module only as a
diagnostic branch. The third and fourth lines did not exist at all: the tree
had *discovered* multi-owner rows empirically (``_census_owner_probe``,
``phase_flip_runtime.py:3773``, which found ~94000 rows -- 21% of a
448698-row pool -- owned by a pool object the census never enumerated) and had
fixed one un-retired id at one site (#796, 689161de77), without ever gating
against either shape.

THE FOUR SPECIMENS THIS MODULE IS RED-FIRST AGAINST
===================================================

These are measured numbers off this rig, not illustrations. Every one is a
test case in ``test_kv_row_ownership_822.py``.

#814 -- rows owned by nobody. 340384 rows unaccounted, "73% of the PP1 pool".
    The denominator of that 73% is the EXPOSED space (466994), not the
    committed backing (124928): 340384 / 466994 = 0.7289. The unaccounted set
    grew in-boot 122 -> 85191 -> 340384 and never came back.

#816 -- ids with no page behind them. Measured
    2026-08-23 06:14:21 / 06:18:15 / 06:19:43, one clamp firing per rank::

        rank   exposed   committed   unbacked
        PP0    466994    212992      254002
        PP1    466994    124928      342066
        PP2    466994    133120      333874

    Note what the arithmetic says and the individual fixes could not: the
    exposed number is IDENTICAL on all three ranks while the backing is
    rank-local (their sum is 471040, i.e. 466994 + 4046). One quantity is
    global, the other is per-rank, and nothing in the tree related them. And
    #814's 342066-vs-340384 is the SAME population as #816's PP1 delta seen
    from the other side. Two tickets, one missing authority, counted twice.

#796 -- an id that outlived its space. PP-space id 344009 was still live
    against a TP cap of 212992 after the cutover -- and 212992 is exactly the
    PP0 committed value above. "One writer, no clearer": the fix cleared one
    site. Retirement generalizes it.

#722 -- the counter-form, and the proof that one-sided fixes cannot hold.
    The #717 fix produced backing 69054 below a highest live id of 233289.
    A clamp that only lowers exposure is blind to this shape by construction,
    which is why the law is a containment CHAIN checked in both directions and
    not a cap.

WHAT THIS MODULE DOES AND DOES NOT DO
=====================================

It is pure arithmetic over integers and row-id sets. No torch, no CUDA, no
pool object, no boot -- so every specimen above is reproducible in a hermetic
unit test, and so there is exactly one place to read the law from.

It does NOT allocate, free, clamp or move anything. Enforcement stays with the
existing actuators: :meth:`clamp_exposure_to_backing` remains the belt under
the exposure law (belt-and-suspenders, #822 item 5 -- and its firing RATE is
now a regression metric: under the authority it must have nothing left to do),
and the flip's group-unanimous commit remains the actuator for retirement.
What changes is that both now consult ONE stated law instead of each carrying
a private copy of half of it.

ATOMICITY, HONESTLY SCOPED
==========================

:meth:`RowOwnershipAuthority.retire` is atomic in the single-process sense
that matters for the #796 shape: the epoch bump and the claim drop cannot be
observed apart, because they are one assignment under one lock. It says
NOTHING about whether all ranks retire at the same wall-clock instant under
load. That is a cross-rank property of the flip's commit protocol, it is only
provable on metal, and it is carried as a named 18-lane window item rather
than asserted here.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

LOG_PREFIX = "KV-OWNERSHIP"

# Reuse, do not restate. #816 already owns the definition of "over-exposed",
# and a second definition here is precisely the failure this module exists to
# end. Imported lazily inside the function to keep this module import-cheap
# and free of a managers/ -> mem_cache/ import cycle.


def _exposure_over_backing(exposed_rows: int, backed_rows: int) -> int:
    from sglang.srt.managers.kv_backing_relief import exposure_over_backing

    return exposure_over_backing(exposed_rows, backed_rows)


class Law(str, Enum):
    """The four laws, named so a violation can be asserted on by identity.

    Tests assert on these, never on message text: a reworded log line must not
    be able to turn a red test green.
    """

    #: exposed id space must not exceed the committed backing (#816)
    EXPOSURE = "exposure"
    #: no claimed row may sit at or above the committed backing (#722)
    COVERAGE = "coverage"
    #: every committed row has exactly one owner -- none unowned (#814),
    #: none claimed twice
    EXCLUSIVITY = "exclusivity"
    #: no claim may survive the cutover that retired its id space (#796)
    RETIREMENT = "retirement"


@dataclass(frozen=True)
class Violation:
    """One broken law, carrying the numbers that broke it.

    ``law`` is what tests assert on. ``detail`` is for humans and logs; it is
    never load-bearing. ``rows`` is the size of the offending population, which
    is the quantity the specimens are stated in (340384, 342066, ...) and the
    one a regression metric can trend.
    """

    law: Law
    rows: int
    detail: str
    #: bounded sample of the offending ids -- never the whole set. The #814
    #: census printed ``sorted(leaked)[:12]``, and twelve consecutive ids at
    #: the minimum say nothing about the rest; a contiguity claim was made and
    #: withdrawn on exactly that evidence.
    sample: Tuple[int, ...] = ()

    def __str__(self) -> str:  # pragma: no cover - formatting only
        tail = f" sample={list(self.sample)}" if self.sample else ""
        return f"[{self.law.value}] {self.rows} rows: {self.detail}{tail}"


@dataclass(frozen=True)
class Claim:
    """One owner's claim on a set of row ids, stamped with an id-space epoch.

    ``owner`` must name the OBJECT that holds the rows, not the role. The
    #814 owner probe's finding was that two distinct pool objects both existed
    and only one was enumerated; "the allocator" would have named both.

    ``epoch`` is the id space the ids were minted under. A claim whose epoch is
    older than the authority's current epoch is a #796 survivor by definition,
    with no need to inspect the ids at all.
    """

    owner: str
    rows: frozenset
    epoch: int

    @property
    def count(self) -> int:
        return len(self.rows)


def _sample(rows: Iterable[int], limit: int = 8) -> Tuple[int, ...]:
    return tuple(sorted(rows)[:limit])


@dataclass
class RowSpace:
    """The immutable-per-epoch facts about one rank's KV row id space.

    Deliberately three plain integers. The #816 specimen is a disagreement
    between two of them that no single object held at once: ``exposed`` came
    from a global token budget, ``committed`` from a rank-local arena, and
    nothing in the tree ever put them side by side until a device-side assert
    did it in a crash log.
    """

    #: highest-exclusive id the allocator may hand out (``exposed_rows()``)
    exposed: int
    #: rows with physical pages actually behind them (``_current_rows()``);
    #: MEASURED, never remembered -- that is the #684 lesson.
    committed: int
    #: the id-space generation. Bumped by, and only by, a retirement.
    epoch: int = 0
    #: ids below this are structurally unowned and must not read as a leak.
    #: Row 0 is the cuda-graph padding row: ``free_pages`` is built as
    #: ``arange(1, size + 1)`` (``allocator/token.py:39``) precisely so a padded
    #: batch's default ``req_pool_indices=0`` lands somewhere harmless. Counting
    #: it as "belongs to no owner" would be the #814 defect in miniature.
    reserved: int = 1


class RowOwnershipAuthority:
    """The single place that says whether the row space is lawful.

    Usage is deliberately narrow:

    * writers ``declare`` what they own (or the auditor ``observe``s them),
    * anything that wants to know calls ``audit`` and gets a list of
      :class:`Violation`, never a bool and never a formatted string,
    * a phase cutover calls ``retire`` exactly once, and every id minted under
      the old space stops being lawful in that one step.

    ``audit`` never raises and never mutates: an authority that can itself
    crash the process it is auditing would just be an eighth crash root.
    """

    def __init__(self, space: RowSpace, *, name: str = "kv") -> None:
        self._lock = threading.RLock()
        self._space = space
        self._name = name
        self._claims: Dict[str, Claim] = {}
        #: how many times the exposure law was found broken. This is the #822
        #: item-5 regression metric: under the authority the #816 clamp must
        #: have nothing left to correct, so this must stay at its baseline.
        self.violation_counts: Dict[Law, int] = {law: 0 for law in Law}

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    @property
    def space(self) -> RowSpace:
        with self._lock:
            return RowSpace(
                exposed=self._space.exposed,
                committed=self._space.committed,
                epoch=self._space.epoch,
                reserved=self._space.reserved,
            )

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._space.epoch

    def declare(
        self, owner: str, rows: Iterable[int], *, epoch: Optional[int] = None
    ) -> None:
        """Record that ``owner`` holds ``rows``.

        ``epoch`` defaults to the CURRENT one, which is the honest default for
        a live writer. Tests and replayed census data pass an explicit epoch to
        reconstruct a #796 survivor.
        """
        with self._lock:
            stamp = self._space.epoch if epoch is None else int(epoch)
            self._claims[owner] = Claim(
                owner=owner, rows=frozenset(int(r) for r in rows), epoch=stamp
            )

    def withdraw(self, owner: str) -> None:
        with self._lock:
            self._claims.pop(owner, None)

    def owners(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._claims))

    def set_backing(
        self, *, exposed: Optional[int] = None, committed: Optional[int] = None
    ) -> None:
        """Update the two integers. Does NOT bump the epoch.

        Growing or shrinking the backing inside one id space is an ordinary
        dial move (#330); it is not a cutover and must not silently retire
        anyone's ids. Only :meth:`retire` retires.
        """
        with self._lock:
            if exposed is not None:
                self._space.exposed = int(exposed)
            if committed is not None:
                self._space.committed = int(committed)

    # ------------------------------------------------------------------
    # retirement -- the #796 leg
    # ------------------------------------------------------------------
    def retire(self, *, exposed: int, committed: int) -> int:
        """Retire the whole old id space in ONE step and open the new one.

        This is the generalization of 689161de77. That fix cleared one writer's
        leftovers at one site; the shape it was fixing -- PP-space id 344009
        still live against a TP cap of 212992 -- is possible wherever ANY
        holder of a pre-cutover id is not visited. Enumerating the holders was
        the thing that kept being incomplete, so retirement does not enumerate
        them: it invalidates the SPACE, and every id minted under it becomes
        unlawful without being touched.

        The pattern is the one already in the tree for the request-slot space
        (``ReqToTokenPool.req_generation``, ``memory_pool.py:377/409/456``,
        upstream 6cc9352dfe), which stamps a generation to defeat exactly this
        ABA shape for req_pool_idx. It was never extended to KV rows.

        Returns the number of claims dropped, so a caller can log a real number
        rather than "cleared".

        ATOMIC IN ONE PROCESS ONLY. The epoch bump and the claim drop are one
        critical section, so no reader can see a bumped epoch with stale claims
        still attached. Whether all ranks cross this point together under load
        is a property of the flip's commit protocol, not of this call, and is
        only provable on metal.
        """
        with self._lock:
            dropped = len(self._claims)
            self._claims = {}
            self._space = RowSpace(
                exposed=int(exposed),
                committed=int(committed),
                epoch=self._space.epoch + 1,
                reserved=self._space.reserved,
            )
            logger.info(
                "%s retired id space -> epoch %d: dropped %d claim(s), "
                "new exposed=%d committed=%d. Every id minted before this "
                "point is now unlawful by epoch, not by enumeration (#796).",
                LOG_PREFIX,
                self._space.epoch,
                dropped,
                self._space.exposed,
                self._space.committed,
            )
            return dropped

    # ------------------------------------------------------------------
    # the law
    # ------------------------------------------------------------------
    def audit(self, *, expect_full_coverage: bool = True) -> List[Violation]:
        """Check all four laws. Returns every violation found, never just the first.

        Returning the full list is not tidiness: #814 and #816 are the SAME
        rows seen through two laws, and a checker that short-circuits on the
        first would have reported one of them and hidden the other -- which is
        how they became two tickets.

        ``expect_full_coverage=False`` suppresses only the "unowned rows" half
        of EXCLUSIVITY, for callers that legitimately hold a partial view (a
        best-effort census mid-flip). Double ownership is checked regardless:
        a partial view can miss an owner, it can never invent one.
        """
        with self._lock:
            space = self._space
            claims = list(self._claims.values())

        out: List[Violation] = []

        # LAW 1 -- exposure <= committed (#816). Reuses the one definition.
        over = _exposure_over_backing(space.exposed, space.committed)
        if over:
            out.append(
                Violation(
                    law=Law.EXPOSURE,
                    rows=over,
                    detail=(
                        f"the allocator could hand out {space.exposed} rows while only "
                        f"{space.committed} are committed, so {over} rows have no page "
                        f"behind them; the first write above the backing is a "
                        f"device-side assert in the KV writer bound check"
                    ),
                )
            )

        # LAW 4 first among the claim laws -- a retired claim's ids are
        # meaningless in the current space, so reporting them as "out of range"
        # or "double owned" would name the wrong defect.
        current: List[Claim] = []
        for claim in claims:
            if claim.epoch < space.epoch:
                out.append(
                    Violation(
                        law=Law.RETIREMENT,
                        rows=claim.count,
                        detail=(
                            f"owner {claim.owner!r} still holds {claim.count} row id(s) "
                            f"minted under epoch {claim.epoch}, but the id space was "
                            f"retired at epoch {space.epoch}; these ids outlived the "
                            f"cutover that invalidated them"
                        ),
                        sample=_sample(claim.rows),
                    )
                )
            else:
                current.append(claim)

        # LAW 2 -- every claimed row is below the committed backing (#722).
        above: Set[int] = set()
        for claim in current:
            above |= {r for r in claim.rows if r >= space.committed}
        if above:
            out.append(
                Violation(
                    law=Law.COVERAGE,
                    rows=len(above),
                    detail=(
                        f"{len(above)} claimed row id(s) sit at or above the committed "
                        f"backing of {space.committed} (highest {max(above)}); these are "
                        f"live rows that are already unmapped -- a grow, not a cap, is "
                        f"what fixes this"
                    ),
                    sample=_sample(above),
                )
            )

        # LAW 3 -- the claims partition the space.
        seen: Dict[int, str] = {}
        doubled: Dict[int, Tuple[str, str]] = {}
        for claim in sorted(current, key=lambda c: c.owner):
            for row in claim.rows:
                prior = seen.get(row)
                if prior is None:
                    seen[row] = claim.owner
                else:
                    doubled.setdefault(row, (prior, claim.owner))
        if doubled:
            pairs = sorted({p for p in doubled.values()})
            out.append(
                Violation(
                    law=Law.EXCLUSIVITY,
                    rows=len(doubled),
                    detail=(
                        f"{len(doubled)} row id(s) are claimed by more than one owner "
                        f"{pairs}; two owners writing the same KV row is silent "
                        f"corruption, not a crash"
                    ),
                    sample=_sample(doubled),
                )
            )
        if expect_full_coverage:
            # THE #814 DEFECT, STATED AS A CHOICE OF RANGE.
            #
            # The ownership question is asked over the COMMITTED backing, never
            # over the exposed id space. The census asked it over the exposed
            # space -- `range(1, size + 1)` where `size` is the allocator's id
            # space (`phase_flip_runtime.py:3649`) -- and so every row that was
            # exposed without a page behind it fell out as "unaccounted". That
            # is why #814 read 340384 rows / 73% of the pool as a leak: the
            # denominator of that 73% is 466994, the EXPOSED space, while the
            # backing was 124928. Those rows were not unowned. They did not
            # exist. LAW.EXPOSURE is the law that governs them, and reporting
            # them here instead is how one defect became two tickets.
            #
            # Ranging over `committed` also makes the two laws non-overlapping,
            # so a violation count per law is a meaningful regression metric
            # rather than a double count.
            # ONE expression of the owned range, used by both the guard and the
            # difference. They were two expressions and a mutant proved it:
            # ranging the difference over `exposed` (the literal #814 defect)
            # left the guard still computing against `committed`, so the guard
            # skipped the check and the mutant survived the whole suite. Two
            # statements of one fact is the shape this module exists to end --
            # it does not stop being that shape because both live in one
            # function.
            lo, hi = space.reserved, space.committed
            span = max(0, hi - lo)
            # Count only IN-RANGE claims: rows above the backing are LAW.COVERAGE's
            # business and must not be able to mask a hole below it.
            in_range = sum(1 for r in seen if lo <= r < hi)
            unowned: Set[int] = set()
            if in_range < span:
                unowned = set(range(lo, hi)) - set(seen)
            if unowned:
                out.append(
                    Violation(
                        law=Law.EXCLUSIVITY,
                        rows=len(unowned),
                        detail=(
                            f"{len(unowned)} committed row id(s) of {space.committed} "
                            f"({100.0 * len(unowned) / max(1, space.committed):.1f}%) "
                            f"belong to no enumerated owner; on this stack that has "
                            f"meant an un-enumerated second pool object, not a leak"
                        ),
                        sample=_sample(unowned),
                    )
                )

        if out:
            with self._lock:
                for v in out:
                    self.violation_counts[v.law] += 1
        return out

    # ------------------------------------------------------------------
    # census bridge
    # ------------------------------------------------------------------
    def observe_census(
        self,
        *,
        free_rows: Iterable[int],
        cached_rows: Iterable[int],
        withheld_rows: Iterable[int] = (),
        resident_rows: Mapping[str, Iterable[int]] = None,
    ) -> List[Violation]:
        """Feed one ``_pool_census`` reading through the law.

        The census (``phase_flip_runtime.py:3649``) already derives its
        ``unaccounted`` figure from exactly these sets. It reports; it does not
        judge, and it names its own single allocator and single tree as the
        whole world -- which is why an un-enumerated second owner reads as a
        leak. Routed through here, the same three sets become four named law
        outcomes, and "unaccounted" stops being a number without a verdict.
        """
        self.declare("free_list", free_rows)
        self.declare("radix_cache", cached_rows)
        self.declare("cap_withheld", withheld_rows)
        for owner, rows in (resident_rows or {}).items():
            self.declare(f"resident:{owner}", rows)
        return self.audit()


def format_violations(violations: Sequence[Violation], *, why: str = "") -> str:
    """One log line per violation, with a stable prefix for log-counter tests."""
    tag = f" ({why})" if why else ""
    return "\n".join(f"{LOG_PREFIX} VIOLATION{tag} {v}" for v in violations)


# ----------------------------------------------------------------------
# call-site adapters
# ----------------------------------------------------------------------
#: attribute under which a runtime carries its authority, mirroring
#: ``phase_flip_spill.KV_BACKING_RELIEF_ATTR``'s convention.
AUTHORITY_ATTR = "kv_row_ownership_authority"


def authority_for(
    host, *, exposed: int = 0, committed: int = 0
) -> RowOwnershipAuthority:
    """Lazily attach ONE authority to ``host`` and return it.

    One per rank, not one per call site: an authority that a caller can
    construct privately is not an authority, it is a fifth copy of half the
    law -- which is the situation this module was written to end.
    """
    found = getattr(host, AUTHORITY_ATTR, None)
    if found is None:
        found = RowOwnershipAuthority(
            RowSpace(exposed=int(exposed), committed=int(committed))
        )
        setattr(host, AUTHORITY_ATTR, found)
    return found


def audit_pool_census(
    authority: RowOwnershipAuthority,
    *,
    exposed: int,
    committed: Optional[int],
    free_rows: Iterable[int],
    cached_rows: Iterable[int],
    withheld_rows: Iterable[int] = (),
    why: str = "",
) -> List[Violation]:
    """Turn one ``_pool_census`` reading into a verdict instead of an integer.

    The census (``phase_flip_runtime.py:3649``) computes::

        leaked = set(range(1, size + 1)) - free - cached - withheld

    and ``size`` there is ``alloc.size``, the EXPOSED id space. That range
    choice is the #814 defect itself: rows exposed without a page behind them
    have no owner because they do not exist, so they fall into ``leaked`` and
    read as a 340384-row leak. The #814 fix subtracted the withheld RANGE,
    which repairs the reading only while ``KvRowCap`` is engaged and publishing
    -- when the backing moved without a cap (the #816 shape) the rows are in
    neither the free lists, the tree, nor the withheld range, and the census is
    wrong again. Asking the ownership question over the COMMITTED backing
    removes the whole class rather than the instance.

    ``committed=None`` means the committed backing could not be MEASURED at
    this call. Then the exposure and coverage laws are unanswerable and this
    says so, rather than substituting ``exposed`` and reporting a clean bill of
    health -- reading ``size`` in place of the committed span is precisely what
    cost a boot on 2026-08-11 (``kv_backing_relief.py:1180``).
    """
    if committed is None:
        logger.warning(
            "%s census audit SKIPPED (%s): the committed backing could not be "
            "measured here, and substituting the %d-row id space for it would "
            "report a clean bill of health for exactly the #816 state.",
            LOG_PREFIX,
            why,
            int(exposed),
        )
        return []

    authority.set_backing(exposed=int(exposed), committed=int(committed))
    found = authority.observe_census(
        free_rows=free_rows,
        cached_rows=cached_rows,
        withheld_rows=withheld_rows,
    )
    if found:
        logger.warning(format_violations(found, why=why))
    else:
        logger.info(
            "%s census audit clean (%s): exposed=%d committed=%d epoch=%d",
            LOG_PREFIX,
            why,
            int(exposed),
            int(committed),
            authority.epoch,
        )
    return found


# ----------------------------------------------------------------------
# the #816 clamp's firing rate, as a regression metric with a parser
# ----------------------------------------------------------------------
#: the marker `clamp_exposure_to_backing` emits when it withdraws over-exposed
#: ids (``kv_backing_relief.py``, LOG_PREFIX "KV-BACKING").
CLAMP_LOG_MARKER = "KV-BACKING exposure clamp"

_CLAMP_RE = re.compile(
    r"\b(?P<rank>[A-Z]{2}\d+)\].*?"
    + re.escape(CLAMP_LOG_MARKER)
    + r".*?hand out (?P<exposed>\d+) rows while only (?P<committed>\d+) are "
    r"committed, so (?P<unbacked>\d+) rows"
)


@dataclass(frozen=True)
class ClampFiring:
    """One firing of the #816 clamp, as the log recorded it."""

    rank: str
    exposed: int
    committed: int
    unbacked: int

    @property
    def is_self_consistent(self) -> bool:
        return self.unbacked == self.exposed - self.committed


def parse_clamp_firings(lines: Iterable[str]) -> List[ClampFiring]:
    """Every #816 clamp firing in a boot log, with its numbers.

    #822 item 5. The clamp STAYS -- it is the belt under the exposure law, and
    a law with no actuator under it is a comment. But its firing rate is the
    thing that says whether the law above it is working: under the authority
    the clamp must have nothing left to correct, so a boot that still fires it
    is a boot where an id space got over-exposed without anyone noticing.

    A rate is only a metric if it has a BASELINE, and the baseline has to be
    read off a real boot rather than asserted. See
    :data:`CLAMP_BASELINE_0823` and the test that pins it against the log.
    """
    out: List[ClampFiring] = []
    for line in lines:
        m = _CLAMP_RE.search(line)
        if m:
            out.append(
                ClampFiring(
                    rank=m.group("rank"),
                    exposed=int(m.group("exposed")),
                    committed=int(m.group("committed")),
                    unbacked=int(m.group("unbacked")),
                )
            )
    return out


def clamp_firing_census(lines: Iterable[str]) -> Dict[str, int]:
    """Firings per rank. The number a later boot is compared against."""
    counts: Dict[str, int] = {}
    for firing in parse_clamp_firings(lines):
        counts[firing.rank] = counts.get(firing.rank, 0) + 1
    return counts


#: MEASURED BASELINE, boot 2026-08-23 06:08
#: (/spinning/evidence-665-f1/boot_816_core_0823_0608.log): TWELVE firings,
#: four per rank, at five distinct second-marks (06:14:21, 06:14:22, 06:18:15,
#: 06:19:43, 06:32:05).
#:
#: RECORDED BECAUSE THE BRIEF FOR THIS TASK SAID THREE. Three was the number of
#: log POSITIONS someone had cited, not the firing rate; the rate is 12. A
#: regression metric seeded from the wrong baseline would have called a boot
#: with nine firings an improvement.
CLAMP_BASELINE_0823 = {"PP0": 4, "PP1": 4, "PP2": 4}

#: And what each rank reported, IDENTICALLY on all four of its firings. The
#: exposed figure is the same on every rank while the backing is rank-local --
#: the structural signature of a global id space over a per-rank arena. It does
#: not drift across the boot, which is what tells a structural defect from a
#: leak.
CLAMP_BASELINE_ROWS_0823 = {
    "PP0": (466994, 212992, 254002),
    "PP1": (466994, 124928, 342066),
    "PP2": (466994, 133120, 333874),
}
