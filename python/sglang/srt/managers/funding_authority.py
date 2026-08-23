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
# ==============================================================================
"""#770/#584/#819 -- ONE authority for "can this action be funded, and from where".

WHY THIS EXISTS, IN ONE SPECIMEN
================================

``boot_816_core_0823_0608.log``, 06:19:43 PP0, four lines from the same second::

    seam staging asked the corridor guard for 396 MiB (tp_to_pp): ok=False,
      spendable now 1625 MiB against a need of 2021 MiB
    corridor gate refused: want 2533 MiB, free 2444 -> 2444 MiB,
      reclaimed 0 MiB from [nothing]
    STAGING BUDGET CENSUS (MiB): driver free 2444, allocator cache 313
      (reclaimable), staging reserve kept free 819, seam fixed 228,
      ARENA COMMIT DUE 437, => SPENDABLE 1625
    KV rung: current=212992 rows, floor=122912, slack=90080,
      deficit=+1676 MiB -> SHRINK to 159355 (KV capacity is the funder)

The instance NAMES its funder, PRICES the exact draw -- 1676 MiB against a
396 MiB need, four times over -- and then abandons with ``[nothing]``. It then
doubles a backoff (6.4s -> 60s, stand down for good at 8) and sits in TP
indefinitely. No single component was wrong. There were three components and no
authority:

  A1  ``corridor_guard`` ladder   rank-local; sees allocator-cache + draft-weights
  A2  KV rung                     GROUP-proportional; pays BEFORE the gate,
                                  so its bytes can never appear in A1's list
  A3  staging census              subtracts 819+228+437 = 1484 MiB, then
                                  formats the result into a LOG STRING
                                  (``phase_flip_runtime.py:6287``) and returns
                                  it to no decision at all

A1 answers "what can this rank release now". A2 answers "what row cap can the
group agree on". A3 answers "what is left after the withholdings". Nobody
answers **"can this action be funded, and from which named post"**. This module
does, and only that.

WHAT IT IS NOT
--------------
It is not an allocator, not a gate, and not an admission decision. Per
``DESIGN_679`` Rule 1 the ordering is fixed -- drain gate (when) -> relief
ladder (what can be freed) -> park guard (final yes/no) -- and this module is
strictly the MIDDLE stage. It computes and explains; the caller still decides.
It never moves a byte: :func:`solve_funding` is a pure function of a snapshot.

THE FIVE LAWS, EACH BOUGHT WITH A MEASUREMENT
=============================================

1. A REFUSAL NAMES EVERY POST IT CONSIDERED, INCLUDING THE ZERO-DRAWS.
   ``corridor_guard._spend_ladder`` does ``if got <= 0: continue`` -- a provider
   that paid nothing is not recorded, so ``[nothing]`` conflates "no providers"
   with "providers paid zero" with "the funder was never in this list". Three
   different worlds, one string, and the 2026-08-16 misreading cost a morning.
   :class:`FundingVerdict` therefore carries a :class:`Draw` for EVERY post,
   with the reason a zero was zero.

2. A POST IS CREDITED BY DELIVERED DRIVER BYTES, NEVER BY PROMISED CAPACITY.
   ``corridor_guard.register``'s own docstring already states this for its
   ladder ("a provider that only returns memory to the caching allocator has
   freed nothing the law can see") and ``_spend_ladder`` re-probes rather than
   trusting provider arithmetic. The KV rung pays OUTSIDE that ladder and is
   accounted on its promise: 06:18:41 claims ``APPLIED shrink to 209555 rows
   (107 MiB returned)`` for a draw the pool reports as ``claimed=0``. A post
   whose last draw underdelivered is DERATED here (:attr:`Post.derate_num` /
   :attr:`Post.derate_den`), not re-trusted at face value. DESIGN_679's reject
   list calls the undetected form of this an "accounting lie".

3. GRANULARITY IS DECLARED, AND A SUB-GRANULE ASK IS A NAMED REFUSAL.
   06:19:43: ``asked 3437 rows against a release granularity of 8192 rows`` ->
   released NOTHING, silently. A post that cannot pay below a granule says so
   (:data:`CAUSE_GRANULARITY`) and offers the round-UP it could pay instead;
   one granule here is 256 MiB against a 396 MiB shortfall.

4. THE USER RESERVE IS NEVER A FUNDABLE POST.
   ``DESIGN_584`` R2, carried from #582: ``card_total = user_reserve +
   computed_demand + kv_pool``, and nothing internal is ever funded from the
   reserve (default 1024 MiB/card). :func:`declare_post` REFUSES to register a
   post flagged :attr:`Post.is_user_reserve`. Note the trap this guards: the
   specimen's ``staging reserve kept free 819`` is the SEAM staging reserve
   (``phase_flip_seam_reserve``), a different object which IS declarable. Two
   things called "reserve"; only one of them is the user's.

5. REDUCE THE VALUE THAT IS BRANCHED ON, NOT THE POOL THAT HOLDS IT.
   DESIGN_679's contract stands unweakened: every quantity feeding a branch or
   a loop bound must be the group-reduced one (#583 and #603 are the two hangs
   that bought it). But it does not license reducing the POOL. Under uneven DCP
   this fork already runs pools that differ by construction and branches safely
   on a reduced floor. The specimen instead reduces a PROPORTION OF EACH RANK'S
   OWN CAP across caps that are unequal by design::

       PP0  cap 212992  floor 128549  slack 84443   -> wants shrink to 159102
       PP1  cap 124928  floor 128549  slack     0   -> floor EXCEEDS its cap
       PP2  cap 133120  floor 128549  slack  4571

       verdict: "DECLINED by a PEER FLOOR ... A rank under no pressure can veto
                 the shrink a pressed rank needs."

   PP1's floor is 102.9% of PP1's own cap, so PP1 vetoes at the no-change
   point, and PP0's 84443 rows -- 1684 MiB, on the very rank that refused, for
   a 396 MiB need -- are unreachable. Per the standing ``kein-bindender-rang``
   law, KV heads are replicated and the token axis is continuous, so per-rank
   capacity is a FREE variable and a reduction against designed inequality is a
   DEFECT, never physics. This module classifies that outcome as
   :data:`CAUSE_PEER_VETO` and refuses to call it scarcity.

   Concretely: a post declared :attr:`Post.group_scope` GROUP_UNIFORM_CAP is
   reduced; a post declared GROUP_NONE (a rank-local draw that changes no
   branch input) is not.

WHY THE CAUSE FIELD MATTERS MORE THAN THE BOOLEAN
-------------------------------------------------
The abandon backoff is keyed on the COUNT of refusals, not on whether the
refusal's cause could have changed (18f §8.5: 6.4s doubled per abandon, stand
down for good at 8, and draining the load did not lift it because the hold
outlives the drain). That is right for scarcity and wrong for a veto over
slack that is sitting right there. :attr:`FundingVerdict.retry_is_pointless`
makes the distinction available to a caller for the first time.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

MIB = 1024 * 1024

# -- relief tiers -------------------------------------------------------------
# Mirrors corridor_guard's ordering so a post can be routed to that ladder
# without a translation table. Tier outranks cost, exactly as there.
RELIEF_LOCAL = 0
RELIEF_REBALANCE = 1
RELIEF_PARK = 2
RELIEF_HOST = 3

_TIER_NAMES = {
    RELIEF_LOCAL: "local",
    RELIEF_REBALANCE: "rebalance",
    RELIEF_PARK: "park",
    RELIEF_HOST: "host",
}

# -- group scope --------------------------------------------------------------
#: The draw changes nothing any rank branches on -- a rank-local staging buffer
#: funded from this rank's own slack. No reduction. (Law 5.)
GROUP_NONE = "none"
#: The draw moves a quantity every rank branches on (a KV row cap changes
#: ``available_size()`` and therefore admission). The AGREED ABSOLUTE is
#: reduced -- never a proportion of a per-rank cap.
GROUP_UNIFORM_CAP = "uniform_cap"

# -- refusal causes -----------------------------------------------------------
#: Every post is genuinely empty. Backing off is correct.
CAUSE_SCARCITY = "scarcity"
#: Slack exists on this rank and a cross-rank reduction refused it. Law 5.
#: NOT scarcity; retrying the same price is pointless but RE-PRICING is not.
CAUSE_PEER_VETO = "peer_veto"
#: The ask is below a post's release granule, so it rounds to zero. Law 3.
CAUSE_GRANULARITY = "granularity"
#: A post promised capacity its last draw did not deliver. Law 2.
CAUSE_PHANTOM = "phantom_capacity"
#: Funded.
CAUSE_FUNDED = "funded"
#: The gate's own watermark sits ABOVE the corridor band it defends, so no
#: amount of funding can ever satisfy it. A CONFIGURATION defect, and the only
#: cause here whose answer is "change a constant", not "find MiB". See
#: :func:`diagnose_floor_band`.
CAUSE_UNSATISFIABLE_FLOOR = "unsatisfiable_floor"


class FundingError(ValueError):
    """A funding question that cannot be answered as posed."""


@dataclasses.dataclass(frozen=True)
class Post:
    """One named place MiB can come from.

    ``available_bytes`` is what the post CLAIMS it could supply right now. What
    it is CREDITED with is that figure after derating (law 2) and after the
    granule (law 3) -- see :meth:`creditable`.
    """

    name: str
    available_bytes: int
    tier: int = RELIEF_REBALANCE
    cost: int = 50
    #: Smallest draw this post can actually deliver. 0 means "any size".
    #: The KV arena's is ``ceil(8 MiB * 32 / 32 KiB) = 8192 rows`` = 256 MiB.
    granule_bytes: int = 0
    group_scope: str = GROUP_NONE
    #: Law 2. Held as a fraction rather than a float so a verdict is exactly
    #: reproducible from a log line: last draw delivered ``num`` of ``den``
    #: promised. ``den == 0`` means no draw has been observed yet -- trust it
    #: once, and the next verdict will know better.
    derate_num: int = 1
    derate_den: int = 1
    #: Law 4. A post so flagged can never be registered.
    is_user_reserve: bool = False
    #: Free text for a refusal line when this post cannot pay.
    unavailable_reason: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise FundingError("a post must be named")
        if self.tier not in _TIER_NAMES:
            raise FundingError(
                f"post {self.name!r}: unknown relief tier {self.tier!r}, "
                f"expected one of {sorted(_TIER_NAMES)}"
            )
        if self.group_scope not in (GROUP_NONE, GROUP_UNIFORM_CAP):
            raise FundingError(
                f"post {self.name!r}: unknown group scope {self.group_scope!r}"
            )
        if self.available_bytes < 0:
            raise FundingError(f"post {self.name!r}: negative availability")
        if self.derate_den < 0 or self.derate_num < 0:
            raise FundingError(f"post {self.name!r}: negative derate")
        if self.derate_den and self.derate_num > self.derate_den:
            raise FundingError(
                f"post {self.name!r}: derate {self.derate_num}/{self.derate_den} "
                "exceeds 1 -- a post cannot deliver more than it promised, and "
                "a ratio above 1 means the measurement, not the post, is wrong"
            )

    @property
    def derated_bytes(self) -> int:
        """Availability after law 2. An unobserved post is trusted once."""
        if self.derate_den == 0:
            return int(self.available_bytes)
        return int(self.available_bytes) * int(self.derate_num) // int(self.derate_den)

    def creditable(self, want: int) -> Tuple[int, str]:
        """Bytes this post can actually deliver toward ``want``, and why not.

        Returns ``(bytes, reason)``; ``reason`` is non-empty only when the
        figure is zero, and it is the text that lands in the refusal line.
        """
        if self.unavailable_reason:
            return 0, self.unavailable_reason
        have = self.derated_bytes
        if have <= 0:
            if self.available_bytes > 0:
                return 0, (
                    f"derated to zero: last draw delivered "
                    f"{self.derate_num}/{self.derate_den} of its promise"
                )
            return 0, "empty"
        draw = min(int(want), have)
        gran = int(self.granule_bytes)
        if gran > 0:
            # Law 3: round UP to a whole granule, because a sub-granule draw
            # delivers ZERO, not a partial. Only if the post cannot cover the
            # rounded-up figure is this a refusal.
            rounded = ((draw + gran - 1) // gran) * gran
            if rounded > have:
                return 0, (
                    f"ask of {draw / MIB:.0f} MiB rounds up to one granule of "
                    f"{gran / MIB:.0f} MiB, which exceeds the "
                    f"{have / MIB:.0f} MiB this post holds"
                )
            draw = rounded
        return int(draw), ""


@dataclasses.dataclass(frozen=True)
class Draw:
    """What one post was asked for and what it gave. Zero-draws included (law 1)."""

    post: str
    tier: int
    asked_bytes: int
    drawn_bytes: int
    reason: str = ""

    def describe(self) -> str:
        if self.drawn_bytes > 0:
            return f"{self.post}[{_TIER_NAMES[self.tier]}] {self.drawn_bytes / MIB:.0f} MiB"
        return f"{self.post}[{_TIER_NAMES[self.tier]}] 0 MiB ({self.reason or 'no reason given'})"


@dataclasses.dataclass(frozen=True)
class FundingVerdict:
    """The answer. ``ok`` is the least interesting field on it."""

    ok: bool
    want_bytes: int
    covered_bytes: int
    draws: Tuple[Draw, ...]
    cause: str
    #: True only when every post is structurally dry, i.e. backing off is the
    #: correct response. A peer veto or a granularity round-down is NOT
    #: pointless to retry -- it is pointless to retry AT THE SAME PRICE.
    retry_is_pointless: bool
    #: Set when law 5 was violated upstream: slack that exists on this rank was
    #: refused by a cross-rank reduction. Names the vetoing rank if known.
    veto_detail: str = ""

    @property
    def shortfall_bytes(self) -> int:
        return max(0, int(self.want_bytes) - int(self.covered_bytes))

    def describe(self) -> str:
        """The line that replaces ``reclaimed 0 MiB from [nothing]``.

        Every post considered appears here, including the ones that paid zero
        and the reason each did. That is the whole point of law 1.
        """
        head = (
            f"want {self.want_bytes / MIB:.0f} MiB, covered "
            f"{self.covered_bytes / MIB:.0f} MiB"
        )
        if not self.ok:
            head += f", SHORT {self.shortfall_bytes / MIB:.0f} MiB"
        body = "; ".join(d.describe() for d in self.draws) or "no posts declared"
        tail = f"cause={self.cause}"
        if self.veto_detail:
            tail += f" ({self.veto_detail})"
        tail += f", retry_is_pointless={self.retry_is_pointless}"
        return f"{head} from [{body}], {tail}"


class FundingAuthority:
    """The registry the posts declare into, and the one place they are solved.

    Deliberately NOT a subclass of, or a wrapper around, ``CorridorGuard``. The
    guard spends; this decides and explains. Keeping them separate is what
    lets a verdict be computed in a test with no CUDA, no NVML and no pool --
    which is the only reason the specimen's arithmetic is reproducible at a
    desk at all.
    """

    def __init__(self, *, rank: int = 0) -> None:
        self.rank = int(rank)
        self._posts: List[Post] = []

    # -- declaration ---------------------------------------------------------

    def declare_post(self, post: Post) -> None:
        """Register a funding post. Law 4 is enforced HERE, not at solve time.

        A rejected user reserve must fail loudly at declaration, because a post
        that silently vanishes between declaration and solve is exactly the
        failure shape #813's empty registry had: something that looks wired,
        answers nothing, and is indistinguishable in a log from a component
        that was simply never needed.
        """
        if post.is_user_reserve:
            raise FundingError(
                f"post {post.name!r} is the USER RESERVE and can never fund an "
                "internal action (DESIGN_584 R2, carried from #582: card_total "
                "= user_reserve + computed_demand + kv_pool). If this is the "
                "seam staging reserve, it is a different object -- declare it "
                "without is_user_reserve."
            )
        if any(p.name == post.name for p in self._posts):
            raise FundingError(f"post {post.name!r} is already declared")
        self._posts.append(post)
        self._posts.sort(key=lambda p: (p.tier, p.cost, p.name))

    @property
    def posts(self) -> Tuple[str, ...]:
        return tuple(p.name for p in self._posts)

    # -- the one question ----------------------------------------------------

    def can_fund(
        self,
        want_bytes: int,
        *,
        group_floor_bytes: Optional[Dict[str, int]] = None,
        local_slack_bytes: Optional[Dict[str, int]] = None,
    ) -> FundingVerdict:
        """Can ``want_bytes`` be funded right now, and from which named posts.

        ``group_floor_bytes`` maps a GROUP_UNIFORM_CAP post name to the
        group-reduced absolute the ranks agreed on -- law 5's legitimate half.
        ``local_slack_bytes`` maps the same names to what THIS rank actually
        holds above its own floor. When the two disagree, the difference is a
        peer veto and is reported as one rather than folded into scarcity.
        """
        want = max(0, int(want_bytes))
        group_floor_bytes = dict(group_floor_bytes or {})
        local_slack_bytes = dict(local_slack_bytes or {})

        draws: List[Draw] = []
        covered = 0
        vetoed_bytes = 0
        veto_names: List[str] = []
        saw_granularity = False
        saw_phantom = False

        for post in self._posts:
            remaining = want - covered
            if remaining <= 0:
                # Law 1 still applies: an untouched post is reported, so a
                # reader can tell "not needed" from "had nothing".
                draws.append(Draw(post.name, post.tier, 0, 0, "not needed"))
                continue

            effective = post
            if post.group_scope == GROUP_UNIFORM_CAP and post.name in group_floor_bytes:
                agreed = max(0, int(group_floor_bytes[post.name]))
                mine = int(local_slack_bytes.get(post.name, post.available_bytes))
                if agreed < mine:
                    # LAW 5, THE MEASURED CASE. This rank holds `mine` and the
                    # group reduction released `agreed`. The difference is not
                    # gone -- it was refused, on this rank, by a rank that does
                    # not need it. Record it as a veto and keep the amount, so
                    # the caller can see exactly what a correct reduction would
                    # have made available.
                    vetoed_bytes += mine - agreed
                    veto_names.append(post.name)
                effective = dataclasses.replace(post, available_bytes=agreed)

            drawn, reason = effective.creditable(remaining)
            if drawn <= 0 and "granule" in reason:
                saw_granularity = True
            if drawn <= 0 and "derated" in reason:
                saw_phantom = True
            draws.append(Draw(post.name, post.tier, remaining, drawn, reason))
            covered += drawn

        ok = covered >= want
        if ok:
            cause = CAUSE_FUNDED
        elif vetoed_bytes > 0:
            cause = CAUSE_PEER_VETO
        elif saw_granularity:
            cause = CAUSE_GRANULARITY
        elif saw_phantom:
            cause = CAUSE_PHANTOM
        else:
            cause = CAUSE_SCARCITY

        veto_detail = ""
        if vetoed_bytes > 0:
            veto_detail = (
                f"{vetoed_bytes / MIB:.0f} MiB of this rank's own slack in "
                f"[{', '.join(veto_names)}] was withheld by a cross-rank "
                f"reduction while this rank is the one that needs it"
            )

        return FundingVerdict(
            ok=ok,
            want_bytes=want,
            covered_bytes=covered,
            draws=tuple(draws),
            cause=cause,
            # Only genuine scarcity justifies the exponential stand-down.
            retry_is_pointless=(not ok and cause == CAUSE_SCARCITY),
            veto_detail=veto_detail,
        )


# -- law 5's reduction, done correctly ---------------------------------------


def uniform_absolute_floor(
    per_rank_caps: Sequence[int],
    per_rank_floors: Sequence[int],
) -> int:
    """The group-agreed ABSOLUTE a uniform pool may shrink to.

    This is the reduction DESIGN_679 actually requires: agree on the absolute
    that every rank's live set clears, so any quantity branched on is uniform.

    It is NOT the reduction the specimen performs. That one reduces a
    PROPORTION of each rank's own cap, and with caps unequal by design
    (212992 / 124928 / 133120) a rank whose floor exceeds its own cap clamps
    the proportion to the no-change point and vetoes the whole group. Compare::

        proportion-of-own-cap:  PP1 floor 128549 / cap 124928 = 102.9% -> 100%
                                -> group grants nothing, PP0's 84443 rows lost
        absolute (this):        max(floors) = 128549 -> PP0 may shrink
                                212992 -> 128549, releasing 84443 rows

    The absolute form cannot be vetoed by a roomy peer, because a peer's floor
    only ever raises the common target -- it never forces a rank that HAS slack
    to keep it.
    """
    if len(per_rank_caps) != len(per_rank_floors):
        raise FundingError(
            f"cap/floor vectors disagree in length: "
            f"{len(per_rank_caps)} vs {len(per_rank_floors)}"
        )
    if not per_rank_caps:
        raise FundingError("no ranks given")
    return max(int(f) for f in per_rank_floors)


def authority_from_seam_snapshot(
    *,
    allocator_cache_bytes: int = 0,
    kv_slack_rows: int = 0,
    row_bytes: int = 0,
    kv_granule_rows: int = 0,
    draft_available_bytes: int = 0,
    draft_reason: str = "",
    rank: int = 0,
) -> FundingAuthority:
    """Build an authority from the numbers a seam refusal ALREADY has in hand.

    Every argument is a plain integer the caller has already computed for its
    own census, so this introspects nothing and can be called from a refusal
    path without reaching into a scheduler. That is deliberate: the census at
    ``phase_flip_runtime._staging_budget_census`` gathers exactly these figures
    and then formats them into a log string, and the whole defect this module
    addresses is that no decision ever receives them.

    A post with nothing to give is still DECLARED, with its reason, because law
    1 requires a refusal to name what it considered.
    """
    auth = FundingAuthority(rank=rank)
    auth.declare_post(
        Post(
            "allocator-cache",
            max(0, int(allocator_cache_bytes)),
            tier=RELIEF_LOCAL,
            cost=10,
            unavailable_reason=(
                "" if allocator_cache_bytes > 0 else "torch cache already returned"
            ),
        )
    )
    auth.declare_post(
        Post(
            "draft-weights",
            max(0, int(draft_available_bytes)),
            tier=RELIEF_REBALANCE,
            cost=20,
            unavailable_reason=(
                draft_reason
                if draft_reason
                else ("" if draft_available_bytes > 0 else "no draft payload resident")
            ),
        )
    )
    # THE POST THE GUARD LADDER CANNOT SEE. It is declared here precisely
    # because it is absent there: the rung pays before the gate, so its bytes
    # never enter the ladder's provider list and a refusal has never once been
    # able to name it. See the module docstring's specimen.
    auth.declare_post(
        Post(
            "kv-slack",
            max(0, int(kv_slack_rows)) * max(0, int(row_bytes)),
            tier=RELIEF_REBALANCE,
            cost=30,
            granule_bytes=max(0, int(kv_granule_rows)) * max(0, int(row_bytes)),
            group_scope=GROUP_UNIFORM_CAP,
            unavailable_reason=(
                "" if kv_slack_rows > 0 else "pool is at or below its rung floor"
            ),
        )
    )
    return auth


@dataclasses.dataclass(frozen=True)
class FloorBandDiagnosis:
    """Whether the gate's watermark is reachable inside the corridor band."""

    satisfiable: bool
    arming_floor_mib: int
    band_ceiling_mib: int
    margin_mib: int
    detail: str

    @property
    def overshoot_mib(self) -> int:
        return max(0, self.arming_floor_mib + self.margin_mib - self.band_ceiling_mib)


def diagnose_floor_band(
    arming_floor_mib: int,
    band_ceiling_mib: int,
    arming_margin_mib: int = 0,
) -> FloorBandDiagnosis:
    """Is the arming floor reachable without leaving the corridor band?

    THIS IS ``BUG_planner_corridor_capacity.md`` DEFECT A, AND IT IS AN
    ARITHMETIC CONTRADICTION BETWEEN TWO MODULE-LEVEL CONSTANTS. Computed from
    the shipped defaults, hermetically::

        corridor_law_mib()          1024
        CORRIDOR_BAND_FRACTION      0.20
        corridor_band_floor_mib()    819   = 1024 - 1024*0.20
        corridor_band_ceiling_mib() 1229
        DEFAULT_SEAM_ENTRY_RESERVE   512
        arming_floor_mib()          1331   = 819 + 512      <- 102 above the ceiling
        DEFAULT_ARMING_MARGIN_MIB    192
        floor + margin              1523   <- 294 above the ceiling

    The standing ``vram-korridor-regel`` says free memory per card under load
    must sit in 819-1229 MiB, and ABOVE that band the boot acceptance has
    FAILED (VRAM buying no tokens). So a rig filled exactly as the corridor
    law requires cannot reach the watermark the seam gate demands before it
    will arm. **A correctly-filled instance is structurally unable to flip.**

    That is why the specimen's refusals never recover no matter how long the
    load drains: the abandon message advises "the flip is retried when
    occupancy drops", but occupancy dropping far enough to clear 1331 MiB
    means leaving the acceptance band in the other direction. The retry
    advice describes a state the corridor law forbids.

    A funding authority must not report this as scarcity. No post can fix it;
    only changing one of the two constants can.
    """
    floor = int(arming_floor_mib)
    ceiling = int(band_ceiling_mib)
    margin = max(0, int(arming_margin_mib))
    need = floor + margin
    if need <= ceiling:
        return FloorBandDiagnosis(
            True,
            floor,
            ceiling,
            margin,
            f"arming floor {floor} MiB (+{margin} margin) fits under the "
            f"corridor band ceiling {ceiling} MiB",
        )
    return FloorBandDiagnosis(
        False,
        floor,
        ceiling,
        margin,
        f"UNSATISFIABLE: the gate arms at {floor} MiB (+{margin} MiB margin = "
        f"{need} MiB) but the corridor band tops out at {ceiling} MiB, so a "
        f"card filled inside its own acceptance band is {need - ceiling} MiB "
        f"short of the watermark BY CONSTRUCTION. No funding post can close "
        f"this; it is a contradiction between the law and the gate derived "
        f"from it. Retrying when occupancy drops cannot help -- clearing the "
        f"watermark means leaving the acceptance band from above."
    )


@dataclasses.dataclass(frozen=True)
class ArmingFloorSolution:
    """A gate watermark that is reachable inside the corridor band, or a refusal."""

    satisfiable: bool
    arming_floor_mib: int
    seam_entry_reserve_mib: int
    max_seam_entry_reserve_mib: int
    detail: str


def solve_arming_floor(
    band_floor_mib: int,
    band_ceiling_mib: int,
    requested_seam_entry_reserve_mib: int,
    arming_margin_mib: int = 0,
) -> ArmingFloorSolution:
    """Solve the arming floor as the FREE VARIABLE it is.

    TWO LAWS MEET HERE AND ONLY ONE OF THEM MAY MOVE.

    The corridor band (819-1229 MiB free per card under load) is a HARD user
    rule: below it is a breach, above it the boot acceptance has FAILED because
    VRAM is buying no tokens. It is not negotiable and this function never
    touches it.

    The arming floor is DERIVED -- ``band_floor + seam_entry_reserve`` -- and
    the seam entry reserve is an allowance, not a law. So when the two are
    inconsistent it is the FLOOR that must give, and the only honest answers
    are (a) a floor that fits under the ceiling, or (b) a refusal that says so
    BY NAME at boot.

    On the shipped defaults there is no (a)::

        band floor 819 + seam reserve 512          = 1331   arming floor
        arming floor 1331 + arming margin 192      = 1523
        band ceiling                                 1229   <- 294 MiB short

    The largest reserve that fits is ``ceiling - floor - margin`` = 218 MiB
    against the 512 shipped, so the shipped configuration cannot arm from
    inside its own acceptance band, on any card, ever.

    WHY THIS MUST NOT BE SILENTLY AUTO-CORRECTED. Cutting the reserve from 512
    to 218 changes what the seam is allowed to spend while it runs; that is a
    behaviour change to the flip and it needs metal, not a desk. So this
    function SOLVES and REPORTS -- it returns the reachable floor and the
    reserve it implies -- and the caller decides whether to adopt it or refuse
    at boot. What must never happen again is the third option the tree shipped:
    neither adopting nor refusing, and instead advising at runtime that the
    flip "is retried when occupancy drops", which describes a state the
    corridor law forbids.
    """
    floor_base = max(0, int(band_floor_mib))
    ceiling = max(0, int(band_ceiling_mib))
    margin = max(0, int(arming_margin_mib))
    requested = max(0, int(requested_seam_entry_reserve_mib))

    headroom = ceiling - floor_base - margin
    max_reserve = max(0, headroom)
    if requested <= max_reserve:
        return ArmingFloorSolution(
            True,
            floor_base + requested,
            requested,
            max_reserve,
            f"arming floor {floor_base + requested} MiB (band floor "
            f"{floor_base} + seam reserve {requested}) plus margin {margin} "
            f"fits under the band ceiling {ceiling} MiB",
        )
    return ArmingFloorSolution(
        False,
        floor_base + requested,
        requested,
        max_reserve,
        f"UNSATISFIABLE ARMING FLOOR: the gate would arm at "
        f"{floor_base + requested} MiB (band floor {floor_base} + seam entry "
        f"reserve {requested} MiB) and with the {margin} MiB arming margin "
        f"needs {floor_base + requested + margin} MiB free, but the corridor "
        f"band tops out at {ceiling} MiB -- a card filled INSIDE its own "
        f"acceptance band is {floor_base + requested + margin - ceiling} MiB "
        f"short of the watermark by construction. The corridor band is a hard "
        f"user rule and does not move; the seam entry reserve is the free "
        f"variable and must be at most {max_reserve} MiB for the gate to be "
        f"reachable at all. Until one of the two is changed the flip cannot "
        f"arm from a correctly filled card, and waiting for occupancy to drop "
        f"CANNOT help: clearing the watermark means leaving the acceptance "
        f"band from above.",
    )


# -- #819: where the break-even inputs actually come from ---------------------

#: A value measured at runtime and self-correcting.
PROV_MEASURED = "measured"
#: A value an operator supplied for this rig.
PROV_ENV = "env"
#: A value frozen in source from another rig's benchmark. The one that decides
#: silently.
PROV_FROZEN = "frozen-literal"


@dataclasses.dataclass(frozen=True)
class BreakEvenProvenance:
    """Where each of the three break-even inputs came from, per #819.

    THE BRIEFING'S FRAMING WAS THAT 7004 IS A FROZEN SEED. It is not a literal
    anywhere: ``phase_policy.break_even_tokens`` computes
    ``N = C / (1/X - 1/P)`` and 7004 is what the shipped inputs happen to
    produce. The staleness is one level down, and it is UNEVEN across the
    three -- which is the finding:

      C  flip cost      SELF-CORRECTING. ``FlipCostEstimator.observe()`` is fed
                        real cutover durations, seeded from
                        ``DEFAULT_FLIP_COST_S = 3.2`` and overridable via
                        ``SGLANG_PHASE_POLICY_FLIP_COST_S``.
      X  TP prefill     ``DEFAULT_TP_PREFILL_TOK_S = 1681.0``, env-overridable
                        (``SGLANG_PHASE_POLICY_TP_TOK_S``) but NEVER measured
                        at runtime.
      P  PP prefill     ``DEFAULT_PP_PREFILL_TOK_S = 7245.5``, same shape.

    So one input self-corrects and two cannot. A rig whose prefill ladder
    differs from the #631 mainrig silently solves N against another machine's
    hardware unless a human sets two env vars -- and nothing tells them to.
    This type makes that visible instead of leaving it implicit.
    """

    flip_cost_s: float
    flip_cost_provenance: str
    tp_tok_s: float
    tp_provenance: str
    pp_tok_s: float
    pp_provenance: str

    @property
    def frozen_inputs(self) -> Tuple[str, ...]:
        out = []
        if self.flip_cost_provenance == PROV_FROZEN:
            out.append("flip_cost_s")
        if self.tp_provenance == PROV_FROZEN:
            out.append("tp_tok_s")
        if self.pp_provenance == PROV_FROZEN:
            out.append("pp_tok_s")
        return tuple(out)

    @property
    def is_fully_solved(self) -> bool:
        return not self.frozen_inputs

    def describe(self) -> str:
        head = (
            f"break-even inputs: C={self.flip_cost_s:g}s "
            f"[{self.flip_cost_provenance}], X={self.tp_tok_s:g} tok/s "
            f"[{self.tp_provenance}], P={self.pp_tok_s:g} tok/s "
            f"[{self.pp_provenance}]"
        )
        if self.is_fully_solved:
            return head + " -- every input is rig-local"
        return (
            head + f" -- STILL FROZEN: {', '.join(self.frozen_inputs)}. These "
            "carry another rig's benchmark and no runtime measurement can "
            "correct them, so the flip threshold is solved against hardware "
            "this instance may not have."
        )


def slack_above_uniform_floor(cap: int, uniform_floor: int) -> int:
    """What one rank may give up once the group has agreed an absolute floor.

    Zero, never negative: a rank whose floor already exceeds its cap (PP1 in
    the specimen, 128549 against 124928) simply has nothing to give. Under the
    ``kein-bindender-rang`` law that is a fact about ONE rank and must not
    propagate into a constraint on the others.
    """
    return max(0, int(cap) - int(uniform_floor))
