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
    the row STATES partition the space: exactly one state per row, none
        stateless -- and a REFERENCE to a row is not one of its states
    a phase cutover retires the whole old id space in ONE step

#916 AMENDED THE THIRD LINE, and the amendment is the whole of that ticket.
As first written it read "exactly one owner per row", which is true of the
three owners it was written for (free list, radix tree, cap band -- mutually
exclusive STATES) and false of the fourth that #822 added. A resident request
does not hold a row instead of the tree; it holds the tree's own ids, written
back into ``req_to_token`` by ``cache_unfinished_req`` after a pool ref is
taken (``radix_cache.py:509,534-537``). Under the un-amended line every cached
prefix on the stack read as "two owners writing the same KV row is silent
corruption": measured 2026-08-26 21:53:13 on all three ranks,
``cached=12281`` against ``[exclusivity] 12280 rows
[('radix_cache', 'resident:requests')]``, ``sample=[2..9]``.
The teeth did not come out with it. A reference over a row the allocator has
already FREED is a use-after-free and now has its own violation kind, instead
of being one indistinguishable line among 33.

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
from typing import (
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

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


#: LAW.EXCLUSIVITY's "claimed twice" shape -- see ``Violation.kind``.
EXCLUSIVITY_DOUBLED = "claimed_by_multiple"
#: LAW.EXCLUSIVITY's "claimed by no one" shape -- see ``Violation.kind``.
EXCLUSIVITY_UNOWNED = "unclaimed"
#: LAW.EXCLUSIVITY's third shape (#916): a REFERENCE holder still names a row
#: the allocator has already put back on its free list. Not the same fact as
#: :data:`EXCLUSIVITY_DOUBLED` and far more dangerous -- the next ``alloc``
#: hands that row to a second writer while the first still reads it, which is
#: the use-after-free the seam copy dies on.
EXCLUSIVITY_FREED_REFERENCE = "referenced_after_free"


# ----------------------------------------------------------------------
# #916: A CLAIM HAS A ROLE, BECAUSE NOT EVERY HOLDER IS AN OWNER
# ----------------------------------------------------------------------
# The law's third line -- "the claims partition the space: exactly one owner
# per row" -- is true of the three owners it was written for and FALSE of the
# fourth one #822 added. `free_list`, `radix_cache` and `cap_withheld` are
# mutually exclusive STATES of a row: free, cached, or withheld above the cap.
# `resident:requests` is not a state, it is a REFERENCE: a live request naming
# a row that some state already holds.
#
# AND THE STACK SHARES THOSE ROWS ON PURPOSE. `RadixCache.cache_unfinished_req`
# inserts the request's rows into the tree and then writes the TREE's ids back
# into `req_to_token` (`radix_cache.py:534-537`, upstream), taking one ref in
# the memory pool (`:509`). From that instant the tree and the request name the
# SAME ids, by design, refcounted by `inc_lock_ref`. Treating that as "two
# owners writing the same KV row is silent corruption" reported the ordinary
# working set as corruption: measured on this rig 2026-08-26 21:53:13, all
# three ranks, `cached=12281` against `[exclusivity] 12280 rows ...
# [('radix_cache', 'resident:requests')]` -- every cached row but one, with
# `sample=[2..9]`, the head of a shared prefix.
#
# THE ROLE IS DECLARED, NEVER INFERRED FROM THE NAME. A prefix test on
# "resident:" would be the #908 substring shape one level down; the caller that
# knows what it is declaring says so.
#: a row STATE that excludes every other state (radix_cache, cap_withheld)
ROLE_EXCLUSIVE = "exclusive"
#: the free list -- also a state, and the one a reference must never overlap
ROLE_FREE = "free"
#: a holder that NAMES rows another owner holds (resident requests)
ROLE_REFERENCE = "reference"


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
    #: structural discriminator for LAW.EXCLUSIVITY's two non-overlapping
    #: shapes -- rows claimed by more than one owner
    #: (``EXCLUSIVITY_DOUBLED``) versus rows claimed by none
    #: (``EXCLUSIVITY_UNOWNED``). #912's consumer (phase_flip_runtime.py's
    #: ownership census) needs to tell these apart to subtract only the
    #: former from a leak check; matching ``"more than one owner" in
    #: detail`` to do that is exactly the "line_gate-Substring-Defekt ->
    #: #908" shape (parsing human prose as control flow), so the
    #: discriminator is a real field here instead, set once at the point
    #: each violation is actually constructed. Empty string for the other
    #: three laws, which have only one shape each and need no split.
    kind: str = ""

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
    #: #916: what this claim IS -- a row state or a reference to one. See the
    #: ROLE_* constants. Defaults to the exclusive state, which is what every
    #: pre-#916 caller meant.
    role: str = ROLE_EXCLUSIVE

    @property
    def count(self) -> int:
        return len(self.rows)

    @property
    def is_reference(self) -> bool:
        return self.role == ROLE_REFERENCE


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

    @property
    def ownable_hi(self) -> int:
        """Exclusive upper bound of the ids that can HAVE an owner.

        TWO INDEPENDENT BOUNDS, AND THE QUESTION IS ONLY ANSWERABLE INSIDE
        BOTH. #814 established the first: a row with no page behind it does not
        exist, so ownership is asked over ``committed``, never over the exposed
        id space. The second is its mirror and was missing -- a row the
        allocator never MINTS has no owner to have, because ``free_pages`` is
        ``arange(1, size + 1)`` (``allocator/token.py:39``) and ``exposed`` is
        that ``size`` (set from ``alloc.size`` at
        ``phase_flip_runtime.py:4040``). Hence ``exposed + 1``: the minted ids
        are ``[reserved, exposed]`` INCLUSIVE.

        WHY THE MISSING BOUND WAS NOT COSMETIC. The committed figure is a
        MEASURED, page-rounded backing row count
        (``KvBackingRelief._current_rows()``), and it legitimately exceeds the
        id space. boot_window1_0823_1204 ran backing 473088 against
        ``size=471314``, so ids 471315..473087 -- exactly 1773 -- were ranged
        over and claimed by nobody, and the audit printed
        ``[exclusivity] 1773 rows ... sample=[471315..471322]`` identically on
        all three ranks: an owner enumerating rows beyond the pool, except no
        owner was. The same 1773 sat inside the 1895 printed at 12:08:33
        (1773 + the genuine 122).

        THE DIRECTIONS ARE NOT SYMMETRIC. ``exposed > committed`` is #816 --
        ids with no page behind them, a device-side assert waiting to happen,
        and LAW.EXPOSURE's whole subject. ``committed > exposed`` is slack
        backing: pages with no id in front of them, which nothing can write to
        and nothing can leak. Reporting it as an ownership violation inflated
        the per-law counter that #822 item 5 uses as its regression metric, so
        the metric trended against a page-rounding artefact.
        """
        return min(self.committed, self.exposed + 1)


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
        self,
        owner: str,
        rows: Iterable[int],
        *,
        epoch: Optional[int] = None,
        role: str = ROLE_EXCLUSIVE,
    ) -> None:
        """Record that ``owner`` holds ``rows``.

        ``epoch`` defaults to the CURRENT one, which is the honest default for
        a live writer. Tests and replayed census data pass an explicit epoch to
        reconstruct a #796 survivor.

        ``role`` (#916) says WHAT the claim is: an exclusive row state, the
        free list, or a reference to a row some state already holds. The
        default is the exclusive state, which is what every caller before #916
        meant, so an un-updated caller cannot silently become a reference.
        """
        with self._lock:
            stamp = self._space.epoch if epoch is None else int(epoch)
            self._claims[owner] = Claim(
                owner=owner,
                rows=frozenset(int(r) for r in rows),
                epoch=stamp,
                role=str(role),
            )

    def withdraw(self, owner: str) -> None:
        with self._lock:
            self._claims.pop(owner, None)

    def owners(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._claims))

    def shared_reference_rows(self) -> int:
        """Rows a REFERENCE names that a row STATE also holds (#916).

        THE LEDGER'S QUESTION, NOT THE LAW'S, and they stopped being the same
        question when the reference role landed. #912's ``double_owned`` term
        exists because a row counted in ``evictable`` (the tree) and again in
        ``session_held`` (the request holding it) is a SURPLUS against
        ``total`` -- and that arithmetic does not care whether the two holders
        are a lawful share or a corruption. The law now says the tree/request
        share is lawful; the ledger still has to subtract it, or the on-idle
        invariant reads the working set as a leak again.

        So this is deliberately NOT derived from the violation list. A
        correction term sourced from a verdict would silently change whenever
        the verdict does, which is exactly what would have happened here.

        Retired claims are excluded, on the same reasoning as :meth:`audit`:
        ids from a dead epoch are not rows.
        """
        with self._lock:
            space_epoch = self._space.epoch
            claims = [c for c in self._claims.values() if c.epoch >= space_epoch]
        state_union: Set[int] = set()
        for claim in claims:
            if not claim.is_reference:
                state_union |= claim.rows
        shared: Set[int] = set()
        for claim in claims:
            if claim.is_reference:
                shared |= claim.rows & state_union
        return len(shared)

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

        # LAW 3 -- the row STATES partition the space, and a REFERENCE is not
        # a state (#916).
        #
        # `seen` answers "does this row have any holder at all", which is the
        # unowned half's question, and a resident request is a perfectly good
        # answer to it -- that is the whole of #822 root A. `doubled` answers
        # "do two writers own this row", and only the exclusive STATES can
        # break that: the tree and the request that references it hold the same
        # ids by construction (`radix_cache.py:534-537`), so counting their
        # overlap as a violation republished the working set as corruption.
        seen: Dict[int, str] = {}
        doubled: Dict[int, Tuple[str, str]] = {}
        states = [c for c in current if not c.is_reference]
        references = [c for c in current if c.is_reference]
        for claim in sorted(states, key=lambda c: c.owner):
            for row in claim.rows:
                prior = seen.get(row)
                if prior is None:
                    seen[row] = claim.owner
                else:
                    doubled.setdefault(row, (prior, claim.owner))
        # References fill the coverage answer without ever creating a doubled
        # row: `setdefault` records the first holder and never overwrites a
        # state, so a referenced row keeps naming the state that holds it.
        for claim in sorted(references, key=lambda c: c.owner):
            for row in claim.rows:
                seen.setdefault(row, claim.owner)
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
                    kind=EXCLUSIVITY_DOUBLED,
                )
            )
        # THE SHAPE THE OLD LUMP WAS HIDING (#916). A reference over a FREE row
        # is not the benign share -- it is a use-after-free with a delay fuse:
        # the row is back in the free list, the next `alloc` hands it to a
        # second request, and the first one is still reading it. Before this it
        # was reported in the same sentence and with the same severity as the
        # tree/request share, which is why 33 violation lines in one boot could
        # not be triaged. It gets its own kind so a boot can be grepped for the
        # dangerous one alone.
        freed = {c.owner: c.rows for c in states if c.role == ROLE_FREE}
        if freed and references:
            free_union: Set[int] = set()
            for rows in freed.values():
                free_union |= rows
            for claim in sorted(references, key=lambda c: c.owner):
                hit = set(claim.rows) & free_union
                if not hit:
                    continue
                out.append(
                    Violation(
                        law=Law.EXCLUSIVITY,
                        rows=len(hit),
                        detail=(
                            f"{claim.owner!r} still references {len(hit)} row id(s) that "
                            f"the allocator has already returned to its free list "
                            f"{sorted(freed)}; the next allocation hands those rows to a "
                            f"second writer while this holder is still reading them -- a "
                            f"use-after-free, not the ordinary tree/request share"
                        ),
                        sample=_sample(hit),
                        kind=EXCLUSIVITY_FREED_REFERENCE,
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
            lo, hi = space.reserved, space.ownable_hi
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
                        kind=EXCLUSIVITY_UNOWNED,
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
        free_rows: Optional[Iterable[int]],
        cached_rows: Iterable[int],
        withheld_rows: Iterable[int] = (),
        resident_rows: Mapping[str, Iterable[int]] = None,
        expect_full_coverage: bool = True,
    ) -> List[Violation]:
        """Feed one ``_pool_census`` reading through the law.

        The census (``phase_flip_runtime.py:3649``) already derives its
        ``unaccounted`` figure from exactly these sets. It reports; it does not
        judge, and it names its own single allocator and single tree as the
        whole world -- which is why an un-enumerated second owner reads as a
        leak. Routed through here, the same three sets become four named law
        outcomes, and "unaccounted" stops being a number without a verdict.

        #832: ``free_rows=None`` means the free rows exist but their ids do not
        -- a watermark allocator. No ``free_list`` claim is declared, because a
        claim over an empty set is not the absence of a claim: it asserts that
        nothing is free, which is the one reading that turns available capacity
        into an ownership violation.
        """
        if free_rows is not None:
            self.declare("free_list", free_rows, role=ROLE_FREE)
        self.declare("radix_cache", cached_rows, role=ROLE_EXCLUSIVE)
        self.declare("cap_withheld", withheld_rows, role=ROLE_EXCLUSIVE)
        # #916: the fourth owner is a REFERENCE, and saying so here is the fix.
        # A resident request does not hold a row INSTEAD of the tree, it holds
        # the tree's own ids -- `cache_unfinished_req` writes them back into
        # `req_to_token` (`radix_cache.py:534-537`) after taking a pool ref.
        # Declared as an exclusive state, that share was reported as "two
        # owners writing the same KV row" on every census with a cached prefix.
        for owner, rows in (resident_rows or {}).items():
            self.declare(f"resident:{owner}", rows, role=ROLE_REFERENCE)
        return self.audit(expect_full_coverage=expect_full_coverage)


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


# ----------------------------------------------------------------------
# #832: how an allocator reports its FREE rows, asked instead of assumed
# ----------------------------------------------------------------------
#: Class attribute an allocator sets to declare the SHAPE of its free
#: accounting. The only declared value is :data:`FREE_WATERMARK`; an allocator
#: that says nothing is read as a page-list allocator, which is what every
#: page-list allocator in this tree already is.
#:
#: WHY A DECLARATION AND NOT AN ``isinstance`` CHECK. The two composite
#: allocators live in ``multi_ended_allocator.py``, which imports the whole
#: unified-buffer stack; importing it here to name them would make the
#: ownership law depend on the memory backend it audits. A class attribute
#: inverts that: the allocator states its own shape, this module never has to
#: know the class.
FREE_ACCOUNTING_ATTR = "census_free_accounting"

#: Free capacity is a WATERMARK COUNT, not a list of ids. ``free_pages`` /
#: ``release_pages`` on such an allocator are empty by construction and mean
#: "not applicable", never "nothing is free".
FREE_WATERMARK = "watermark"

#: The reading kinds :func:`read_free_rows` can return.
FREE_ENUMERATED = "enumerated"  # a real page list; ids are known
FREE_COUNTED = "counted"  # a declared watermark; only the COUNT is known
FREE_COUNTED_UNDECLARED = "counted-undeclared"  # page list empty, but the
#: allocator's own ``available_size()`` contradicts it -- an undeclared
#: composite, read through its watermark and named as such
FREE_UNKNOWN = "unknown"  # no page list, no declaration, no watermark


@dataclass(frozen=True)
class FreeRowReading:
    """What one allocator can honestly say about its free rows.

    #832. ``_pool_census`` used to compute::

        free = set(alloc.free_pages.tolist()) | set(alloc.release_pages.tolist())

    which ASSUMES a page-list allocator. On the two unified composite
    allocators both fields are stubbed to a permanently empty tensor at
    construction -- "we use watermark math, not free-lists"
    (``multi_ended_allocator.py:1734``) -- so ``free`` came out
    UNCONDITIONALLY EMPTY and every genuinely free row fell into
    ``unaccounted``.

    THE SIZE OF THAT ERROR IS RECORDED BUT NOT CONFIRMED, and the distinction
    is kept because this module exists to stop numbers travelling further than
    their evidence. The tree cites ~94000 rows, 21% of a 448698-row pool, FLAT
    across four censuses on a 2026-08-22 r5 flip
    (``phase_flip_runtime.py:4167``). That log was not retained. The three
    2026-08-23 window boots that WERE retained all ran ``enable_unified_memory
    =False``, so neither composite was ever constructed in them; their
    ``free=`` is nonzero and dynamic throughout (a stubbed composite can only
    ever report ``free=0``) and their ``unaccounted`` drifts rather than
    sitting flat. So the ~94000 figure is UNDECIDABLE from retained evidence.

    This fix does not rest on it. It rests on the construction fact above,
    which is checkable by reading the two ``__init__`` bodies: the fields are
    empty by design, so on those classes the old expression could not have
    returned anything but the empty set.

    That reading did not stay in the log line. ``audit_pool_census`` declares
    the same set as the ``free_list`` owner, so on a composite allocator the
    #822 audit derived one FALSE ownership violation per free row, every
    census.

    THE POINT OF THIS TYPE IS THAT "HOW MANY" AND "WHICH ONES" ARE DIFFERENT
    QUESTIONS. A watermark allocator can answer the first and genuinely cannot
    answer the second: free space above the watermark has never been minted as
    ids, so there is no set to return. Collapsing the two -- returning an empty
    set for "I cannot enumerate" -- is exactly the defect being fixed, one
    level up. ``rows is None`` therefore means UNANSWERABLE and is propagated
    as such, the same way ``resident_rows=None`` already is.
    """

    kind: str
    #: The free ids, or ``None`` when the allocator cannot enumerate them.
    rows: Optional[FrozenSet[int]]
    #: How many rows are free, or ``None`` when even the count is unknown.
    count: Optional[int]
    #: ``type(alloc).__name__``, so a census line names what it read.
    allocator: str
    detail: str

    @property
    def is_enumerable(self) -> bool:
        return self.rows is not None

    @property
    def is_answerable(self) -> bool:
        """Whether the allocator answered at all. ``False`` is the #606 state:
        a value that must be printed as UNKNOWN and never as ``0``."""
        return self.count is not None

    def __str__(self) -> str:
        n = "UNKNOWN" if self.count is None else str(self.count)
        return f"{self.kind}:{n}"


def _read_available_size(alloc) -> Optional[int]:
    """``alloc.available_size()`` as an int, or ``None`` if it cannot answer.

    Never substitutes ``0``: on this path a missing answer and "nothing is
    free" are opposite conclusions, and conflating them is the #606 getattr
    family in one line.
    """
    fn = getattr(alloc, "available_size", None)
    if not callable(fn):
        return None
    try:
        value = fn()
    except Exception:  # noqa: BLE001 -- an instrument, never a gate
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def read_free_rows(alloc) -> FreeRowReading:
    """Ask the allocator for its free rows in ITS OWN accounting shape.

    ONE authority, used by both consumers. The census line and the #822 audit
    read this function, not the allocator, so the two can never again disagree
    about what "free" meant -- the audit's false-violation storm existed
    because it re-derived the census's assumption instead of sharing its
    source.

    Dispatch order, and each step's reason:

    1. **The allocator declared its shape** (:data:`FREE_ACCOUNTING_ATTR`).
       A declaration outranks the field inspection below because the composite
       allocators DO carry ``free_pages``/``release_pages`` -- deliberately, as
       empty tensors, "for the leak checker" (``multi_ended_allocator.py:2070``).
       Inspecting the fields first would classify them as page-list allocators
       and reproduce the defect exactly.
    2. **A real page list** -- both fields present and not ``None``. Base init
       leaves them ``None`` (``allocator/base.py:47-48``) and page-list
       subclasses fill them, so this discriminates without a class check.
    3. **Neither** -- :data:`FREE_UNKNOWN`, named in the output.

    THE CORROBORATION IN STEP 2 IS THE PART THAT GENERALISES. A future
    composite that forgets to declare would fall into step 2, read ``free=0``,
    and silently rebuild the 94000-row band. So an enumerated reading of ZERO
    is checked against the allocator's own ``available_size()``, and a
    contradiction (nothing in the list, capacity on the watermark) is reported
    as :data:`FREE_COUNTED_UNDECLARED` rather than believed. The two authorities
    disagreeing is information; picking the one that happens to be empty is
    how this class of defect survives a rewrite.
    """
    name = type(alloc).__name__ if alloc is not None else "None"
    if alloc is None:
        return FreeRowReading(
            kind=FREE_UNKNOWN,
            rows=None,
            count=None,
            allocator=name,
            detail="no allocator to read",
        )

    declared = getattr(alloc, FREE_ACCOUNTING_ATTR, None)
    if declared == FREE_WATERMARK:
        available = _read_available_size(alloc)
        if available is None:
            return FreeRowReading(
                kind=FREE_UNKNOWN,
                rows=None,
                count=None,
                allocator=name,
                detail=(
                    f"{name} declares watermark free accounting but its "
                    f"available_size() did not answer; the free count is "
                    f"UNKNOWN, not zero"
                ),
            )
        return FreeRowReading(
            kind=FREE_COUNTED,
            rows=None,
            count=available,
            allocator=name,
            detail=(
                f"{name} accounts free capacity by watermark; "
                f"available_size()={available} rows free, ids not enumerable"
            ),
        )

    free_pages = getattr(alloc, "free_pages", None)
    release_pages = getattr(alloc, "release_pages", None)
    if free_pages is not None and release_pages is not None:
        try:
            rows = frozenset(free_pages.tolist()) | frozenset(release_pages.tolist())
        except Exception as exc:  # noqa: BLE001 -- an instrument, never a gate
            return FreeRowReading(
                kind=FREE_UNKNOWN,
                rows=None,
                count=None,
                allocator=name,
                detail=f"{name} page lists could not be read: {exc}",
            )
        if rows:
            return FreeRowReading(
                kind=FREE_ENUMERATED,
                rows=rows,
                count=len(rows),
                allocator=name,
                detail=f"{name} page list: {len(rows)} free id(s)",
            )
        # An EMPTY page list is the ambiguous case, and the only one worth a
        # second opinion: it is the truth on a full page-list pool and the
        # permanent state of an undeclared composite.
        available = _read_available_size(alloc)
        if available:
            return FreeRowReading(
                kind=FREE_COUNTED_UNDECLARED,
                rows=None,
                count=available,
                allocator=name,
                detail=(
                    f"{name} presents EMPTY page lists while its own "
                    f"available_size() reports {available} free row(s). Read as "
                    f"a watermark allocator that has not declared "
                    f"{FREE_ACCOUNTING_ATTR}={FREE_WATERMARK!r}; set it on that "
                    f"class. Believing the empty list here is #832 (~94000 rows, "
                    f"21% of the pool, misreported as unaccounted)"
                ),
            )
        return FreeRowReading(
            kind=FREE_ENUMERATED,
            rows=rows,
            count=0,
            allocator=name,
            detail=f"{name} page list: 0 free id(s), corroborated by available_size()",
        )

    available = _read_available_size(alloc)
    if available is not None:
        return FreeRowReading(
            kind=FREE_COUNTED_UNDECLARED,
            rows=None,
            count=available,
            allocator=name,
            detail=(
                f"{name} has no page list (free_pages/release_pages are None) "
                f"but available_size() reports {available} free row(s)"
            ),
        )
    return FreeRowReading(
        kind=FREE_UNKNOWN,
        rows=None,
        count=None,
        allocator=name,
        detail=(
            f"{name} reports free capacity in no form this census knows: no "
            f"{FREE_ACCOUNTING_ATTR} declaration, no page list, no "
            f"available_size(). The free count is UNKNOWN -- reporting it as 0 "
            f"would turn the whole pool into a phantom leak"
        ),
    )


def free_reading_of(value) -> FreeRowReading:
    """Normalise a caller's ``free`` argument into a :class:`FreeRowReading`.

    #832. A :class:`FreeRowReading` passes through. Anything else is a caller
    that HANDED OVER IDS -- so it is an enumerated reading by construction, not
    an assumption about an allocator's shape. This exists so the one call site
    that already had the ids (and the specimen tests that pass them literally)
    do not have to build a reading to say what they already said.
    """
    if isinstance(value, FreeRowReading):
        return value
    rows = frozenset(value)
    return FreeRowReading(
        kind=FREE_ENUMERATED,
        rows=rows,
        count=len(rows),
        allocator="caller-supplied",
        detail=f"caller enumerated {len(rows)} free row(s)",
    )


def audit_pool_census(
    authority: RowOwnershipAuthority,
    *,
    exposed: int,
    committed: Optional[int],
    free_rows: Optional[Iterable[int]],
    cached_rows: Iterable[int],
    withheld_rows: Iterable[int] = (),
    resident_rows: Optional[Mapping[str, Iterable[int]]] = None,
    free_count: Optional[int] = None,
    free_detail: str = "",
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

    # THE FOURTH OWNER, AND WHY ``None`` IS NOT ``{}``.
    #
    # Rows handed to an in-flight request are out of the free lists and not yet
    # in the tree. They are owned -- by that request -- and until #822 the
    # census had no term for them, so the working set read as a leak: 122 rows
    # against ``resident_reqs=1`` on the first census of
    # boot_window1_0823_1204, before anything had moved.
    #
    # ``resident_rows=None`` means the caller could not enumerate that owner.
    # Then "this row belongs to nobody" is UNANSWERABLE, not false, and the
    # unowned half is suppressed exactly as it is for any other partial view.
    # Substituting an empty mapping would instead assert that requests hold no
    # rows, which re-reports the whole working set as unowned -- the defect
    # with an extra step. Double ownership is still checked either way: a
    # partial view can miss an owner, it can never invent one.
    #
    # A CLAIM THAT IS NOT REFRESHED MUST NOT SURVIVE. ``declare`` overwrites an
    # owner but never removes one, so a ``resident:*`` claim left by an earlier
    # census would keep vouching for rows the requests have since returned, and
    # a real hole underneath it would read as owned. Withdraw before observing.
    if resident_rows is None:
        for owner in authority.owners():
            if owner.startswith("resident:"):
                authority.withdraw(owner)
    # #832: THE FREE LIST IS AN OWNER LIKE ANY OTHER, AND IT CAN ALSO BE
    # UNENUMERABLE.
    #
    # ``free_rows=None`` means the allocator accounts free capacity by
    # watermark: ``free_count`` rows are free and their ids do not exist as a
    # set. Before this, the census handed over the empty set that a composite
    # allocator's stubbed ``free_pages`` produces, and the coverage law read
    # every one of those rows as belonging to nobody -- ~94000 false violations
    # per census on a 448698-row pool, 21% of it.
    #
    # Treated exactly like an unenumerable ``resident_rows``: withdraw the
    # stale claim so it cannot keep vouching for rows, and drop full-coverage,
    # because a partial view can miss an owner but must never invent a hole.
    # Double-ownership checking is unaffected -- that law needs only the owners
    # that ARE enumerable.
    if free_rows is None:
        authority.withdraw("free_list")
        logger.info(
            "%s census audit (%s): free rows are COUNTED, not enumerable "
            "(%s free); the unowned-rows law is suppressed for this reading "
            "rather than answered from an empty set. %s",
            LOG_PREFIX,
            why,
            "UNKNOWN" if free_count is None else free_count,
            free_detail,
        )
    authority.set_backing(exposed=int(exposed), committed=int(committed))
    found = authority.observe_census(
        free_rows=free_rows,
        cached_rows=cached_rows,
        withheld_rows=withheld_rows,
        resident_rows=resident_rows,
        expect_full_coverage=resident_rows is not None and free_rows is not None,
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


# ---------------------------------------------------------------------------
# #919: WHICH OWNER WAS NOT ENUMERATED.
#
# The EXCLUSIVITY_UNOWNED line already tells the reader what it has historically
# meant -- "on this stack that has meant an un-enumerated second pool object,
# not a leak" -- and then leaves them to find that object by hand. Measured on
# the 0826 rerun, three ranks, both boots: 4096 committed ids of 228897 /
# 140961 / 148289 unowned, always `sample=[1..8]`, i.e. the same fixed block at
# the BOTTOM of the id space while the withheld set is by construction the top.
# The same shape appeared in the 2g boot at 2047-of-2048, i.e. the whole
# committed backing.
#
# #919 as filed read that line as "the tree LOSES 4096 rows without free" and
# hung #842 on it. The line says the other thing. This probe closes the gap the
# line leaves instead of arguing about the reading: it asks, at the moment of
# the violation, whether a second pool object exists and whether its id space
# covers the block nobody claimed.
#
# THE VERDICT IS THE DELIVERABLE, and it is three-valued on purpose -- the two
# "not a leak" answers and the one that sends the hunt onward are different
# conclusions and must not share a line.
# ---------------------------------------------------------------------------

#: A second pool object exists and its id space covers every sampled unowned
#: row. The census simply did not enumerate it: an ENUMERATION GAP, not a leak.
CANDIDATE_COVERS = "SECOND-POOL-COVERS"
#: A second pool object exists but its id space does not cover the sample.
#: Not the explanation; the block is still unaccounted for.
CANDIDATE_DISJOINT = "SECOND-POOL-EXISTS-BUT-DISJOINT"
#: No second pool object is reachable at all. The block is genuinely ownerless
#: and the hunt goes to the release/retirement paths (reset_tree,
#: drop_prefix_tree_returning_rows, the #920 id-space retirement neighbourhood).
CANDIDATE_ABSENT = "NO-SECOND-POOL"
#: A second pool object exists, but its id space is not NARROWER than the
#: census's own, so containment is true for every conceivable sample and the
#: test carries no information. #1050: this verdict exists because the
#: containment test WAS reported as an explanation in exactly that state.
CANDIDATE_VACUOUS = "SECOND-POOL-SPANS-EVERYTHING"


@dataclass(frozen=True)
class OwnerCandidate:
    """A pool object the census did NOT enumerate, and the ids it could own.

    ``hi`` is EXCLUSIVE. Ranges are 1-based to match the census's own
    ``range(1, size + 1)`` id space -- the off-by-one here would silently turn
    a covering candidate into a disjoint one, which is the wrong answer in the
    expensive direction (it would send the hunt to reset_tree for a block that
    was explained all along).
    """

    name: str
    lo: int
    hi: int
    #: False when this candidate's id range is not NARROWER than the census's
    #: own id space. #1050: `pp_stack_allocator owns ids [1, 578995)` against a
    #: census of size 578994 -- containment then holds for every sample that
    #: could ever be drawn, so a COVERS verdict from it is not evidence of
    #: anything. The caller sets this; the verdict below refuses to explain a
    #: block with a test that cannot fail (indicator law: a counter is a
    #: finding only once you have checked THAT it measures what it claims).
    discriminating: bool = True

    def covers(self, rows: Iterable[int]) -> bool:
        return all(self.lo <= int(r) < self.hi for r in rows)


def unenumerated_owner_verdict(
    sample: Iterable[int], candidates: Sequence[OwnerCandidate]
) -> Tuple[str, str]:
    """Three-valued: does an un-enumerated pool explain this unowned block?

    PURE. The caller resolves the candidates per access (never a construction
    reference -- that is the #927 class, and this module's own census reads its
    allocator per access for exactly that reason) and this decides.

    Judged on the SAMPLE, which is bounded and is all the violation carries.
    That is a real limit and it is stated rather than papered over: a candidate
    that covers the sample is evidence, not proof, for the whole block. It is
    still the difference between "look for a second pool" and "hunt the release
    paths", which is the decision this exists to make.
    """
    rows = [int(r) for r in sample]
    if not candidates:
        return CANDIDATE_ABSENT, "no second pool object is reachable from here"
    if not rows:
        return (
            CANDIDATE_DISJOINT,
            "the violation carried no sample, so no candidate can be tested "
            f"({len(candidates)} present)",
        )
    for cand in candidates:
        if not cand.discriminating:
            continue
        if cand.covers(rows):
            return (
                CANDIDATE_COVERS,
                f"{cand.name} owns ids [{cand.lo}, {cand.hi}) and covers every "
                f"sampled row -- the census did not enumerate it",
            )
    vacuous = [c for c in candidates if not c.discriminating]
    if vacuous and len(vacuous) == len(candidates):
        shown = ", ".join(f"{c.name}=[{c.lo}, {c.hi})" for c in vacuous[:4])
        return (
            CANDIDATE_VACUOUS,
            f"{len(vacuous)} second pool object(s) present, but every one spans "
            f"the whole census id space ({shown}), so the containment test is "
            "true for any sample and explains nothing. This block is NOT "
            "excused: treat it as unowned until a candidate can name the ids "
            "it actually holds",
        )
    shown = ", ".join(f"{c.name}=[{c.lo}, {c.hi})" for c in candidates[:4])
    return (
        CANDIDATE_DISJOINT,
        f"{len(candidates)} second pool object(s) present but none covers the "
        f"sample: {shown}",
    )
