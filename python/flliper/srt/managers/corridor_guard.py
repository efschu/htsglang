"""#656 spec item 15a/15b: SPILL BEFORE THE ALLOCATION, not after the alarm.

WHY THIS IS NOT ``kv_pressure_runtime``
---------------------------------------
``KvPressureRuntime`` (#287) already drives a spill ladder, and it already
takes a ``spill_fn``. It is nonetheless the wrong shape for item 15a, and the
difference is the whole point of the user's order:

    "SPILL-BEFORE-ALLOC -- check AT the allocation (free-X >= 1024, otherwise
     spill synchronously first), not reactive threshold watching"

(Translated from the user's German; the original wording is in the commit that
introduced this module.)

``on_round`` observes occupancy at a ROUND BOUNDARY and reacts. Between two
boundaries an allocation can take the card below the floor, and the corridor
law is a CONTINUOUS minimum -- it is broken by the trough, not by the average,
and a sampler at 100 ms will see a dip that a per-round controller never
notices. Worse, the allocation that most needs guarding is the seam's
``commit_range``, which does not happen on a round boundary at all: it happens
inside the flip's no-return region, and it has killed this instance before
(``cuMemCreate failed: CUDA_ERROR_OUT_OF_MEMORY``, 2026-08-09).

So this module is a GATE AT THE CALL SITE, synchronous, returning only once
the headroom exists or once it is provably unobtainable. It does not replace
the pressure ladder; the ladder still handles slow, planned residency changes.
This handles the instant a specific allocation is about to happen.

TWO WATERMARKS (item 15b), and why one is not enough
----------------------------------------------------
Freeing exactly enough to clear the floor guarantees that the NEXT allocation
of any size spills again. That is thrashing: every allocation pays a spill,
and the spill/restore pair costs far more than the allocation it enabled. So
the gate arms at the FLOOR and frees up to ``floor + delta``:

    arm at   free - want <  floor           (the corridor law's own bound)
    free to  free - want >= floor + delta   (headroom for the next few asks)

``delta`` is deliberately a byte figure and not a percentage: the corridor law
is stated in absolute MiB, and a percentage of a 20 GiB card and a 32 GiB card
would arm at different absolute headrooms on hardware whose floor is the same.

WHAT A PROVIDER IS
------------------
A payload class that can give memory back synchronously: the draft-weight
carrier, idle GDN/mamba slots at bs<4, the inactive layout's arena tail,
session KV via kvso. Each registers a callable that frees "up to N bytes and
returns how many it actually freed", plus a COST RANK. The gate spends the
cheapest first, which is the reclaim-ordering law this chain already follows
elsewhere: coldest and cheapest-to-restore goes first, hot data last.

ITEM 15c IS A PROVIDER, NOT A SPECIAL CASE. "If everything resident is hot,
keep computing over the host tier (kvso) -- the price is tempo, never a
corridor breach" is expressed here as the most expensive provider in the
order. When the cheap ones are exhausted the gate reaches it, pays the
latency, and still does not breach. The gate NEVER returns "allocate anyway".

FAILURE IS A REFUSAL, NOT A BREACH
----------------------------------
If every provider is exhausted and the headroom still is not there, the gate
says so and the caller must not allocate. At the seam that means abandoning
the flip -- which the affordability verdict already knows how to do
unanimously and for free -- rather than dying inside the no-return region.
A gate that shrugs and lets the allocation proceed would be worse than no
gate, because it would launder a corridor breach as a check that passed.

ITEM 16: THE CARDS FILL EVENLY, AND HOST RAM IS LAST
----------------------------------------------------
    "CARDS MUST FILL EVENLY BEFORE ANY HOST SPILL. Host spill happens ONLY
     when ALL three cards are at the floor -- never while one card binds and
     the others have headroom."

"Even" is defined on FREE HEADROOM, not on bytes held: this rig's cards are
32/20/20 GiB, so equal bytes would mean permanently unequal pressure and the
20 GiB cards would bind forever. The objective is therefore a water-filling
one over the per-card FREE column, and the metric that says whether it has
been achieved is the SPREAD of that column (:func:`free_spread_mib`), which
belongs in every corridor CSV so that "evenly filled" is provable rather
than asserted.

This splits the provider order into a TIER above the cost:

    RELIEF_LOCAL      give back memory NO payload owns. Torch's caching
                      allocator holds blocks no tensor uses; returning them
                      moves NVML's free column, moves no payload anywhere,
                      and costs only the re-``cudaMalloc`` of whatever asks
                      next. It levels nothing, so it is not a rebalance; it
                      spills nothing, so it is not a host tier. It is simply
                      free money and must be spent before anything that
                      moves a real payload.
    RELIEF_REBALANCE  make the free column more level. In TP this is the
                      uneven-DCP token vector (``distributed.corridor_vector``
                      already solves it against per-card corridor capacity):
                      steer tokens to the freest card, physical backing
                      follows the vector, VA stays geometric. In PP the KV is
                      layer-bound and cannot be token-steered, so levelling
                      means evacuating everything NOT layer-bound from the
                      heavy card -- the drafter (done), idle mamba slots,
                      arena slack, graph pools.
    RELIEF_PARK       park a cold payload in ANOTHER card's surplus free
                      space. Still VRAM, still fast, and it levels.
    RELIEF_HOST       system RAM. Gated: see below.

**Tier outranks cost, and that is the point.** Cost still orders providers
WITHIN a tier -- item 15's cheapest-first law is untouched -- but it may not
promote a host spill ahead of a rebalance merely because the host spill is
cheaper to execute. Those are answers to different questions.

**The host gate is a FLEET predicate.** A guard that sees only its own card
cannot distinguish "everything is full" from "I am the only one that is
full", and those two states have opposite correct answers. So the guard reads
the whole free column and admits the host tier only when
:func:`fleet_is_level` holds. Without a fleet probe the host tier stays shut:
item 16 is a permission that must be proven, not assumed.

**Why the drafter is a REBALANCE and not a host spill, even though its bytes
land in host RAM.** The tier names what the action does to the free column,
not where the bytes go. Evacuating a non-layer-bound payload from the binding
card is precisely the PP levelling move the user prescribes ("drafter done;
mamba slots, arena slack, graph pools next"), and it is not the KV/cold spill
class item 16 gates. If a successor finds the user meant the destination
rather than the effect, the change is one ``tier=`` argument at the
registration site -- deliberately, so that it is a one-line decision and not
a refactor.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

LOG_PREFIX = "CORRIDOR-GUARD"

_MIB = 1024 * 1024

#: The user's corridor law, in the unit the law is stated in. THE ONE
#: DECLARATION. Every other module that needs the law imports it from here
#: rather than repeating the literal -- ``corridor_trace.summary`` used to
#: default to its own ``1024`` and ``phase_flip_seam_census`` to another, so
#: three modules each held a private copy of a number the operator states
#: once.
CORRIDOR_LAW_MIB = 1024
#: Historical name, kept because it is the ``floor_mib`` default of the guard
#: and callers pass it positionally.
#:
#: #1257c: it is the FALLBACK, not the law. A caller that knows which card it
#: is guarding passes ``corridor_floor_mib(uuid, group=...).verdict_floor_mib``
#: instead; this default is what a caller with no card identity can honestly
#: use, and it is the same number the derivation returns under
#: ``UNMEASURED-FALLBACK``, so the two can never disagree.
DEFAULT_FLOOR_MIB = CORRIDOR_LAW_MIB

#: How far above the LAW the gate starts working. The arming floor is not a
#: second policy number: it is the law plus what a seam is expected to draw
#: while it runs, and it must be DECLARED WITH the law or the two drift.
#:
#: #656 acceptance (2026-08-13) is what that drift costs. The gate armed at
#: 1536 MiB while the verdict was read against 1024, so a 512 MiB entry
#: allowance was being asked to cover a draw that the corridor sampler
#: measured at 1814-1852 MiB on GPU0 -- 3.6x. Five cutovers entered cleanly
#: through a gate that had no objection and took the card to 886 MiB, 138
#: below the law, and NOTHING IN THE PROCESS NOTICED: the runtime's own
#: verdict is read at 1024 and its own gate arms at 1536, so the breach fell
#: in the gap between the two numbers.
#:
#: The default keeps the shipped 512 MiB allowance, so a boot that does not
#: measure its draw behaves exactly as before. What changes is that the pair
#: is now derived and logged together, and an arming floor BELOW the law --
#: which would let the gate bless an allocation the law forbids -- is
#: refused instead of silently accepted.
DEFAULT_SEAM_ENTRY_RESERVE_MIB = 512


#: Overrides the law. Read HERE and nowhere else: three modules used to read
#: it with their own ``"1024"`` fallback (``kv_vmm_backing``,
#: ``phase_flip_seam_census``, and ``corridor_trace``'s default argument), so
#: the law could be moved for one of them and not the others -- a divergence
#: with no symptom until a breach is judged twice and answered differently.
LAW_ENV = "FLLIPER_CORRIDOR_LAW_FLOOR_MIB"


def corridor_law_mib() -> int:
    """The law in force, in MiB. THE reader of :data:`LAW_ENV`.

    Read per call, not frozen at import: a rank can be told the law late
    (the ``kv_vmm_backing`` preempt path is reached long after import), and
    a value captured at import cannot be corrected by a boot that sets the
    variable afterwards.
    """
    raw = os.environ.get(LAW_ENV)
    if raw is None:
        return CORRIDOR_LAW_MIB
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return CORRIDOR_LAW_MIB


def corridor_law_bytes() -> int:
    """The law in force, in bytes."""
    return corridor_law_mib() * _MIB


#: THE CORRIDOR IS A BAND, NOT A POINT (user relaxation, 2026-08-15).
#:
#: The law was stated as "~1024 MiB free per card, best-filled", and both
#: halves were being read as an exact target: a sample at 1000 MiB counted as
#: a breach and a card resting at 2500 MiB counted as over-filled. Neither
#: reading is what the number is for. The operator's relaxation makes the
#: tolerance explicit at +-20 %, which is what lets the boot-time gate and the
#: planner solve be simple: a solve that has to land on a point has no
#: feasible region, and one that has to land in a band does.
#:
#: The centre stays the target -- the self-correcting margin pulls back to it,
#: because a mechanism that aims at the edge of its own tolerance has none.
#: The FLOOR is the verdict: below it is a breach, and nothing inside the band
#: is. Measured consequence on this rig: the cutover transient at the
#: `weights_refill` stage bottomed at 895-935 MiB on gpu0, which is inside the
#: band, so that class stops being chased.
CORRIDOR_BAND_FRACTION = 0.20


def corridor_band_floor_mib() -> int:
    """Below this is a breach. The law minus its tolerance."""
    law = corridor_law_mib()
    return int(law - law * CORRIDOR_BAND_FRACTION)


def corridor_band_ceiling_mib() -> int:
    """Above this at REST is over-filled: VRAM buying no tokens.

    The second half of "best-filled", and the one a boot-time gate needs in
    order to refuse a configuration that leaves gibibytes idle.

    WHAT "BUYING NO TOKENS" MEANS, because the flip made it ambiguous (#784).
    This ceiling grades the USER-FREE column, not the raw NVML free field. An
    armed phase-flip boot holds an arming floor above the band floor on every
    rank; those MiB are working capital the guard actively defends so a flip
    can arm, so they buy the flip and are not idle. Grade
    :func:`net_free_mib`, never the raw reading -- see that function for why
    the raw comparison is unsatisfiable by construction.

    ROUNDED, NOT TRUNCATED. ``int()`` gave 1228 here while the acceptance
    verdict (``corridor_verdict_774.sh``) computed ``int(round(...))`` = 1229,
    so the code and the gate that decides pass/fail disagreed by 1 MiB on the
    number under test. Rounding is also the symmetric choice: the floor's
    ``int(819.2)`` and ``round(819.2)`` are the same 819, so only this side
    ever moved.
    """
    law = corridor_law_mib()
    return int(round(law + law * CORRIDOR_BAND_FRACTION))


def committed_arming_mib(arming_mib: int) -> int:
    """The part of an arming floor that is CHARGED, not free.

    An arming floor is ``corridor_band_floor_mib() + seam draw + margin``. The
    band floor half belongs to the user -- it is the external headroom the
    corridor law promises. Everything ABOVE it is internal demand: working
    capital the flip spends while it runs, which the guard reclaims and spills
    to defend. Under the reserve semantics a reserve is the user's external
    free space and internal demand is booked exactly, in the ledger, so this
    part is a ledger post and must not be graded as free VRAM.

    Returns 0 for any floor at or below the band floor: such a rank holds
    nothing on the flip's behalf. Never negative.
    """
    return max(0, int(arming_mib) - corridor_band_floor_mib())


def net_free_mib(free_mib: int, arming_mib: int) -> int:
    """The quantity the corridor band actually grades: free MINUS the credit.

    THE DEFECT THIS EXISTS FOR (#784). The shipped constants make the raw
    comparison unsatisfiable. The smallest arming floor any rig can ask for is
    ``819 + 512 + 192 = 1523`` MiB -- the band floor, the shipped seam entry
    allowance and the arming margin -- while the ceiling a boot is graded
    against is 1229 MiB. ``1523 > 1229`` on every rank, unconditionally, so an
    armed phase-flip boot was graded against a threshold its own arming rule
    forbade it to reach, and no gate said so: this module's only pair check
    refused an arming floor BELOW the law and never looked at the ceiling
    side, and :func:`corridor_band_ceiling_mib` had no production consumer.

    Crediting the committed part resolves the pair rather than widening the
    band: a rank resting exactly at its arming floor reads exactly the band
    floor and is in band, whatever its measured seam draw. This is the
    Option-A formula the acceptance verdict has documented since 2026-07-22
    (``net = free_min - sum(registered posts)``) and never implemented.

    IT CANNOT MANUFACTURE A PASS, which is the whole risk of the change. The
    credit is a rank's own floor and nothing more, so a rank resting far above
    that floor keeps every stranded MiB visible: boot instr9's ranks 0 and 1
    rest at 5229 / 6612 MiB against a 1523 MiB floor and still read 4525 /
    5908 MiB net, still far over the ceiling, still a failed acceptance. The
    credit makes the grading honest; the capacity defect stays the planner's
    to solve.
    """
    return int(free_mib) - committed_arming_mib(arming_mib)


def corridor_band_mib():
    """``(floor, centre, ceiling)`` -- the whole band in one read."""
    return corridor_band_floor_mib(), corridor_law_mib(), corridor_band_ceiling_mib()


# ---------------------------------------------------------------------------
# #1257c -- THE CORRIDOR FLOOR IS DERIVED, NOT DECLARED (user decision,
# 2026-09-09).
#
# The operator's own words for why 1024 existed: "die 1024er grenze von mir
# existiert ja nur weil du den wahren vram verbrauch nicht bepreisen konntest
# UND weil ich manchmal noch vram fuer andere prozesse brauche. wenn du jetzt
# korrekt bepreisen kannst, dann kann die default 1024er grenze auch weg (das
# feature muss aber erhalten bleiben, eben weil ich noch andere prozesse
# manchmal nebenher habe die vram brauchen)".
#
# So the number splits into the two things it was standing in for, and each
# half is now named:
#
#     corridor_floor(card) = measured_peak(group awake on that card)
#                          + user_reserve(card)
#
# The first term is a MEASUREMENT -- the S3 activation probe's
# ``activation_delta_bytes`` for this (hardware fingerprint, activation
# profile, card), read through ``mem_ledger.activation.resolve_phase_footprint``
# and through nothing else. The second is the knob the user kept, and it is
# the ONLY policy number left in the floor.
#
# 1024 SURVIVES AS A STRING, NOT AS A NUMBER. Where the transient is not
# measured the floor still reads 1024, but it is stamped
# ``UNMEASURED-FALLBACK`` and it is VERDICT-ONLY: it may print, it may grade,
# it may refuse -- it may never actuate a budget cut. That asymmetry is the
# whole point of the decision. A cut priced off a number nobody measured is
# exactly what #1257's own module docstring refuses ("a boot must never
# silently get a budget that was priced off numbers nobody measured"), and the
# fallback IS such a number.
#
# THE MEASURED FLOOR HAS NO +-20 % TOLERANCE, and the fallback keeps it. The
# band exists because a hand-stated target needs slack; a measured transient
# peak is a physical requirement and slack below it is a breach with extra
# steps. So ``verdict_floor_mib`` is the derived floor exactly when the
# transient is measured, and the familiar band floor (``floor * 0.8`` = 819 at
# 1024) when it is not -- which keeps an unmeasured rig at reserve 0 grading
# byte-identically to the shipped tree.
#
# THE UPPER EDGE IS A FINDING, NEVER A FAIL (decision 5). ``ceiling_mib``
# still exists and ``unmobilised_free_mib`` is still worth printing, but no
# consumer may turn "above the ceiling" into a failed verdict on its own.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# REFUTER FIX 5 (2026-09-09): THE FLOOR'S MEASURED TERM AND ITS VERDICT ARE IN
# DIFFERENT UNITS, AND THE LINE NOW SAYS SO.
#
# The measured transient is ``activation_delta_bytes`` -- a TORCH CACHING
# ALLOCATOR high-water delta across the probe bracket (``activation_probe``).
# The verdict (``free_mib < verdict_floor``) and the budget cut are both
# graded in NVML FREE bytes. A true allocator peak catches every spike inside
# its bracket, so this is NOT sampler-cadence blindness -- but it is blind to
# everything the allocator never sees: the CUDA context, BAR1/mapping windows,
# NCCL and cuBLAS workspaces, and any non-torch allocation on the card.
#
# CONSEQUENCE, AND IT IS THE DANGEROUS DIRECTION: a floor built from allocator
# bytes and enforced on NVML bytes is a LOWER BOUND on the real requirement,
# systematically low by an unmeasured amount. Too low a floor buys too small a
# cut, which leaves LESS free VRAM than the transient actually needs. The tree
# already knows the gap exists and has never priced it: ``D_AWAKE_OVERSHOOT_MIB
# = 404`` in the Weg-2 launcher is a cross-boot constant that exists precisely
# because that consumption "sits outside the fraction".
#
# NOT COMPENSATED HERE. Adding a margin would be a hand number, which this
# slice exists to remove. It is DECLARED instead: every floor line carries
# ``basis=``, and the gap between the two units is an open UNPROVEN item, not
# a silent assumption.
#: The transient came from a torch allocator high-water delta and is enforced
#: on NVML free. A LOWER BOUND -- see the block above.
BASIS_ALLOC_DELTA = "torch-alloc-delta(NVML-lower-bound)"
#: The transient is a number an operator typed. No unit question, no measurement.
BASIS_HAND_SET = "hand-set"
#: The transient is the stated law nobody measured. Verdict-only by construction.
BASIS_STATED_LAW = "stated-law"

#: The transient term came from the env override (:data:`LAW_ENV`). Actuating:
#: an operator who moves the law by hand has priced it by hand.
FLOOR_SOURCE_ENV = "ENV-OVERRIDE"
#: The transient term is 1024 and nobody measured it. VERDICT-ONLY.
FLOOR_SOURCE_FALLBACK = "UNMEASURED-FALLBACK"

#: The dump layer's name for the other half of the same ambiguity: one
#: filename, two digests. Quoted (never redeclared) by
#: :func:`digest_group_collision`, which refuses the mirror -- one digest, two
#: groups -- so an operator grepping either half finds both.
PHASE_FOOTPRINT_COLLISION_NAME = "W18 PdFlipPhaseFootprintCollision"


def corridor_ceiling_for_floor_mib(floor_mib: int) -> int:
    """The unmobilised-free edge above a given floor. A FINDING threshold.

    #1257c. Split out so a consumer that has only a NUMBER (a floor read back
    off a log line, say) derives the edge the same way
    :attr:`CorridorFloor.ceiling_mib` does, instead of re-typing ``* 1.2``.
    """
    f = max(0, int(floor_mib))
    return int(round(f + f * CORRIDOR_BAND_FRACTION))


def unmobilised_above_ceiling_mib(free_mib: int, ceiling_mib: int) -> int:
    """MiB resting above a KNOWN ceiling, or 0. Never a failure by itself.

    The ceiling-first form of :func:`unmobilised_free_mib`, for a consumer that
    already holds the edge (a pre-#1257c log grades against the band's OWN
    ceiling, which is not derivable from the band floor). It exists so that
    ``corridor_arm.arm_report`` does not do the subtraction itself: #656's
    one-converter gate forbids arithmetic there BY NAME, because a second
    subtraction in that function is exactly how the two readers came to
    disagree about the unit.
    """
    return max(0, int(free_mib) - int(ceiling_mib))


def unmobilised_free_mib(free_mib: int, floor_mib: int) -> int:
    """MiB resting ABOVE the finding edge, or 0. Never a failure by itself.

    User decision 2026-09-09, consequence 5: "the upper band edge stays a
    FINDING ('unmobilised free') never a FAIL by itself".
    """
    return max(0, int(free_mib) - corridor_ceiling_for_floor_mib(floor_mib))


#: Every measured source token starts with this. Declared so the two readers
#: of "is this measured" (:attr:`CorridorFloor.measured` and
#: :func:`verdict_floor_for_mib`, which grades a floor recovered from a LOG
#: LINE and has no object) test the same prefix instead of two literals.
MEASURED_SOURCE_PREFIX = "MEASURED-"

#: The only group tags a floor may be keyed by. Weg-2 has exactly two phases
#: and the store has exactly two digests; anything else cannot resolve.
CANONICAL_GROUPS = ("P", "D")

#: What :func:`normalise_group` returns for a tag it will not accept. No
#: digest is ever published under it, so it can only ever reach the fallback.
GROUP_UNKNOWN = "?"


def normalise_group(group) -> str:
    """The CANONICAL group tag. ``"D(dry, expectation)"`` -> ``"D"``.

    REFUTER FIX 2 (2026-09-09). The group was a KEY in one place
    (:func:`_published_digest` looks the digest up by it) and a free-form
    LABEL in another: ``launcher.budgets_from_dc`` passes its ``label``
    straight through, and one call site passes ``"D(dry, expectation)"`` while
    the other passes ``"D"``. Both resolved to the fallback today, so the two
    budgets agreed BY ACCIDENT; the day D is measured the dry pass would read
    ``UNMEASURED-FALLBACK`` and the real pass ``MEASURED-D``, and the launcher
    would hold two different budgets for one group with nothing saying so.

    The rule is deliberately narrow: the LEADING RUN OF LETTERS, upper-cased,
    and only if that run is a canonical group. A decorated label
    (``"D(dry, expectation)"``, ``"P-warm"``) keys the same floor as its bare
    group; anything that is not a group (``"decode"``, ``""``, ``None``)
    becomes :data:`GROUP_UNKNOWN` rather than being upper-cased and passed on
    to a lookup that would silently miss.
    """
    raw = str(group or "").strip().upper()
    head = ""
    for ch in raw:
        if not ch.isalpha():
            break
        head += ch
    return head if head in CANONICAL_GROUPS else GROUP_UNKNOWN


_WARNED_UNKNOWN_GROUPS: set = set()


def _warn_unknown_group(asked) -> None:
    """Say ONCE that a tag will never resolve a measured floor."""
    key = str(asked)
    if key in _WARNED_UNKNOWN_GROUPS:
        return
    _WARNED_UNKNOWN_GROUPS.add(key)
    logger.warning(
        "CORRIDOR-FLOOR group tag %r is not one of %s, so no measured "
        "transient can be attributed to it and this card falls back to the "
        "stated law. A decorated label (%r) keys its bare group; a tag that "
        "is not a group does not.",
        asked,
        list(CANONICAL_GROUPS),
        "D(dry, expectation)",
    )


def measured_floor_source(group: str) -> str:
    """``MEASURED-P`` / ``MEASURED-D`` -- the source token for a measured peak.

    The group travels IN the token because P and D are two different
    quantities: P's peak is a prefill activation peak and D's is whatever the
    decode/verify tree peaks at. A reader that cannot tell them apart cannot
    tell a P-derived floor graded against D-awake free -- a scope error -- from
    a correct pairing.
    """
    return f"{MEASURED_SOURCE_PREFIX}{normalise_group(group)}"


def verdict_floor_for_mib(floor_mib: int, source: str) -> int:
    """The BELOW threshold for a floor known only as ``(number, source)``.

    REFUTER FIX 1 (2026-09-09). :attr:`CorridorFloor.verdict_floor_mib` is the
    same rule with an object in hand; this is the rule for a consumer that
    recovered a floor from a LOG LINE (``corridor_arm`` reading a front log).
    ONE rule, two entry points -- the pre-fix tree had the arm grade against
    the FLOOR while the front printed its verdict against the VERDICT FLOOR,
    so a card at 852 MiB under an unmeasured 1024 read ``verdict=IN`` on one
    line and "BELOW its corridor floor" on the next.
    """
    f = max(0, int(floor_mib))
    if str(source or "").startswith(MEASURED_SOURCE_PREFIX):
        return f
    return int(f - f * CORRIDOR_BAND_FRACTION)


#: Pointer file: ``{"hw_fingerprint": str, "groups": {"P": digest, ...}}``.
#:
#: NOT A SECOND LEDGER, and the distinction is load-bearing. It carries
#: DIGESTS and never a MiB: every number still comes from the one activation
#: store under ``mem_ledger.activation``. It exists because the process that
#: OWNS the ``ActivationProfile`` (the boot, which has a real ServerArgs) is
#: not the process that needs the floor (the front and the launcher), and
#: ``launcher.p_activation_reserve_provenance`` refuses BY NAME to restate
#: ``profile_from_server_args`` -- "a restatement that is wrong by one field
#: does not fail loudly: it returns None, which reads as uncalibrated, which
#: would refuse a launch that is perfectly fine". Publishing the digest the
#: boot actually used removes the restatement instead of making it twice.
FLOOR_DIGEST_FILE_ENV = "FLLIPER_CORRIDOR_FLOOR_DIGEST_FILE"
FLOOR_DIGEST_FILE_DEFAULT = "corridor_floor_digest.json"

#: #233 (24.10., fund #232): THE POINTER IS KEYED BY MODEL IDENTITY, NOT BY
#: GROUP ALONE. The NF boot read the 27B INT8 corridor floors (767/701/700 MiB
#: on D, stamped MEASURED-D, actuates=yes) because the pointer said only
#: ``{"groups": {"D": digest}}`` and the NF cache directory was a ``cp -a``
#: copy of the 27B one: nothing on the read path compared the model. The
#: pointer now carries, next to the legacy ``groups`` map (left untouched for
#: older trees), a ``records`` map keyed by :func:`record_identity_key`::
#:
#:     {"hw_fingerprint": "...", "groups": {...legacy...},
#:      "records": {"<checkpoint>|<arch>|<recipe>": {
#:          "identity": {"checkpoint": ..., "arch": ..., "profile": ...,
#:                       "recipe": ..., (review MUSS 2, :data:`RECIPE_ENV`)
#:                       "architectures": [...HF config architectures...]},
#:          "groups": {"D": digest, "P": digest}}}}
#:
#: The identity terms are the CALIBRATION IDENTITY the launcher already uses
#: (``pdflip_form.calibration_identity``: checkpoint = ``model_key`` + form),
#: read from the ONE rank contract ``FLLIPER_PDFLIP_FORM`` -- so the launcher,
#: the front and every rank derive the same key without a new channel. A
#: record of another identity is refused BY NAME (:data:`RECORD_FOREIGN_TAG`),
#: a legacy entry without identity is "unknown origin"
#: (:data:`RECORD_UNKNOWN_TAG`) unless the footprint it names states this
#: boot's architectures in its own content (:data:`RECORD_LEGACY_TAG`, the
#: migration rule). Either refusal falls back to the stated law (1024, or
#: ``FLLIPER_CORRIDOR_LAW_FLOOR_MIB``), named in the provenance. DANGER
#: DIRECTION: a foreign record with a LOWER floor must never price this boot.
RECORD_FOREIGN_TAG = "#233 CORRIDOR-RECORD-FOREIGN"
RECORD_UNKNOWN_TAG = "#233 CORRIDOR-RECORD-UNKNOWN-ORIGIN"
RECORD_LEGACY_TAG = "#233 CORRIDOR-RECORD-LEGACY-ADOPTED"
RECORD_OWN_TAG = "#233 CORRIDOR-RECORD-OWN"
RECORD_PUBLISH_TAG = "#233 CORRIDOR-RECORD-PUBLISH"
#: The identity check is ON by default (user order 14.10.: new features ship
#: on). ``0``/``off`` restores the pre-#233 group-keyed read -- only on an
#: explicit user word, because off is the OOM direction #232 measured.
RECORD_IDENTITY_ENV = "FLLIPER_CORRIDOR_RECORD_IDENTITY"

#: #233 review MUSS 1 (24.10.): ONE POINTER STATE PER BOOT. Since #233 every
#: group rank publishes in ``alloc_memory_pool``, so the pointer file changes
#: WHILE the boot runs. The launcher's budget, its lend floor
#: (``FLLIPER_PDFLIP_D_LEND_FLOOR_MIB``) and EXTEND_TRIM were fixed before the
#: ranks started; every rank-local reader (credit floor, extend line, preempt,
#: vram_dial, front) asked the file per call -- a rank with TWO floors at once,
#: the state #232's digest section 4 forbids ("nicht tun"). The launcher now
#: reads the pointer ONCE, before any floor of the boot is read, resolves it
#: for both groups (:func:`freeze_boot_pointer`) and hands the resolution to
#: every process of the boot in this variable (the front's env starts from
#: ``os.environ``; ``launch_group`` arms each group's env with it -- ``build_env``
#: pops it, so the dry-run's PDFLIP-GROUP-ENV plan line carries no box state). Every
#: reader takes the frozen resolution; the writer keeps writing for the NEXT
#: boot. Chosen over "readers ignore records of their own boot" (review
#: option b) because (b) still lets a reader see a record a CONCURRENT boot of
#: another container wrote mid-boot, needs a boot id in every record, and keeps
#: the per-call file read on the hot path; the frozen value is one parse.
POINTER_FROZEN_ENV = "FLLIPER_CORRIDOR_POINTER_FROZEN"
POINTER_FROZEN_TAG = "#233 CORRIDOR-POINTER-FROZEN"
POINTER_READ_TAG = "#233 CORRIDOR-POINTER-READ"
#: The ``source=`` of a pointer read: the launcher's frozen record (a digest),
#: the frozen state says "no record" (the law), the frozen state is a named
#: refusal (foreign / unknown origin / other recipe -> the law), or no frozen
#: state at all (a process the launcher did not start: desk, planner CLI).
POINTER_SOURCE_FROZEN = "FROZEN"
POINTER_SOURCE_LAW = "LAW"
POINTER_SOURCE_FOREIGN = "FOREIGN"
POINTER_SOURCE_LIVE = "LIVE"

#: #233 review MUSS 2: THE RECIPE IS PART OF THE RECORD KEY. ``checkpoint|arch``
#: separates models, not recipes, and the footprint store is keyed by the
#: activation-profile digest -- so a pointer that names the LAST boot's digest
#: hands the next boot of another recipe that recipe's measured footprint
#: (store evidence: a 122-token P footprint, chunk 512, 39/20/20 MiB). The key
#: is ``checkpoint|arch|recipe``; a boot of another recipe finds no record of
#: its own and takes the law, named (:data:`RECORD_OTHER_RECIPE_TAG`).
#:
#: ``recipe`` hashes the FOOTPRINT-RELEVANT flags of the group argv the
#: launcher builds (:data:`RECIPE_FLAGS`), never ``profile_from_server_args``
#: restated in the launcher (the #1257c refusal stands). This variable carries
#: the launcher's recipe of BOTH groups, ``{"P": {"hash", "items"}, "D": ...}``,
#: computed before the first floor read; :data:`RECIPE_SELF_ENV` carries the
#: hash of the REAL argv of the group a rank belongs to (the writer's key).
RECIPE_ENV = "FLLIPER_CORRIDOR_RECIPE"
RECIPE_SELF_ENV = "FLLIPER_CORRIDOR_RECIPE_SELF"
RECORD_OTHER_RECIPE_TAG = "#233 CORRIDOR-RECORD-OTHER-RECIPE"
RECIPE_DRIFT_TAG = "#233 CORRIDOR-RECIPE-DRIFT"
RECIPE_TAG = "#233 CORRIDOR-RECIPE"
#: The flags whose value moves the prefill transient a footprint measures, read
#: from the code that builds the footprint's key and sizes the forward:
#:
#: * ``mem_ledger.activation.profile_from_server_args`` -- the ActivationProfile
#:   fields: ``--chunked-prefill-size`` (tokens per forward), ``--tp-size`` /
#:   ``--pp-size`` (shard width, stage depth), ``--kv-cache-dtype``,
#:   ``--speculative-num-draft-tokens``, and the decode graph's ``max_bs``
#:   (``--cuda-graph-max-bs[-decode]``; unstated, server_args derives it from a
#:   memory tier and ``--max-running-requests``, server_args.py
#:   ``cuda_graph_config.decode.max_bs`` / ``max_running_requests or ...``);
#: * the draft forward that runs inside a prefill: ``--speculative-algorithm``
#:   (DFLASH vs NEXTN/MTP are different forwards), ``--speculative-num-steps``,
#:   ``--speculative-eagle-topk`` (they derive the draft token count when it is
#:   not stated);
#: * ``--max-running-requests`` (the order names it; it bounds the batch and,
#:   unstated graph bs, the decode capture);
#: * ``--enable-mixed-chunk`` (decode tokens ride a prefill forward) and
#:   ``--disable-cuda-graph`` (no decode capture at all).
#:
#: DELIBERATELY OUT: everything the floor itself PRICES (``--rank-gpu-memory-mib``,
#: ``--mem-fraction-static``, ``--max-total-tokens``) -- the floor feeds the
#: budget, a budget term in the key would change the key with the floor; the
#: per-boot SOLVED layout (``--pp-stage-ratio``/``--pp-attn-stage-ratio``,
#: ``--rank-tp-ratio``), solved from records each boot, would never repeat; and
#: ports, paths, seeds. The checkpoint (dtype, quantization) is the key's
#: ``checkpoint`` term already.
RECIPE_FLAGS: Tuple[str, ...] = (
    "--chunked-prefill-size",
    "--tp-size",
    "--tensor-parallel-size",
    "--pp-size",
    "--pipeline-parallel-size",
    "--kv-cache-dtype",
    "--speculative-algorithm",
    "--speculative-num-steps",
    "--speculative-eagle-topk",
    "--speculative-num-draft-tokens",
    "--cuda-graph-max-bs",
    "--cuda-graph-max-bs-decode",
    "--max-running-requests",
)
#: Store-true flags of the recipe (present = ``"1"``).
RECIPE_BOOL_FLAGS: Tuple[str, ...] = ("--enable-mixed-chunk", "--disable-cuda-graph")
#: Spellings folded onto one item, so ``--tensor-parallel-size 3`` and
#: ``--tp-size 3`` are one recipe.
_RECIPE_ALIASES = {
    "--tensor-parallel-size": "--tp-size",
    "--pipeline-parallel-size": "--pp-size",
}

#: #233 review MUSS 2 (second half): MINIMUM EVIDENCE OF A MEASURED FOOTPRINT.
#: ``probe_activation.py`` takes ``allocated_bytes.all.peak`` minus the floor
#: over ONE prefill; a prefill shorter than one chunk never ran a full-width
#: forward, so its peak is not the recipe's transient (store evidence: digest
#: 1a4c9a1cdba1, chunk 512, "prefill of 122 tokens", 39/20/20 MiB). The
#: minimum is defined from quantities the footprint already records: the
#: prefill token count its ``source`` states (``prefill of N tokens``, written
#: by the ingest) must be at least the ``chunked_prefill_size`` of its own
#: profile. Below that -- or "?" tokens, or a profile with no chunk -- the
#: footprint is refused as a corridor floor BY NAME and the law holds; it is
#: never MEASURED-* with actuates=yes. A footprint that states no prefill at all
#: (hand-written, a desk fixture, the shipped reference bounds) is not judged
#: here: there is no count to compare.
RECORD_THIN_TAG = "#233 CORRIDOR-RECORD-THIN-EVIDENCE"

#: The group this process is, when it knows. Set by the Weg-2 launcher for
#: both groups; absent outside Weg-2, which is why ``group`` is an explicit
#: argument everywhere and this is only the last resort.
GROUP_ENV = "FLLIPER_PDFLIP_GROUP"

#: ``{card_uuid: MiB}`` as JSON -- the operator's external headroom, handed to
#: a child process that has no argv for it (the Weg-2 front, and since #236
#: every rank of group P and group D through the group env). DECLARED HERE,
#: with the rest of the floor, so the producer (the launcher) and the consumers
#: (front, ranks) cannot drift apart through two copies of a string.
USER_RESERVE_ENV = "FLLIPER_PDFLIP_USER_RESERVE_MIB_BY_CARD"

#: #236 review MUSS-1: ``{card_uuid: stated MiB}`` of the cards whose operator
#: reserve the launcher did NOT book for this group because that card's awake
#: rest is a measured, booked record (``budgets_from_dc``: "the reserve was
#: never free VRAM -- the awake D consumed it"). The value in
#: :data:`USER_RESERVE_ENV` for such a card is the EFFECTIVE one (0); this
#: variable only lets the rank NAME why (log source ``masked-by-booked-rest``),
#: it never feeds a floor.
USER_RESERVE_MASKED_ENV = "FLLIPER_PDFLIP_USER_RESERVE_MASKED_MIB_BY_CARD"


#: #236: raw values of :data:`USER_RESERVE_ENV` already reported by this process
#: (``None`` = the variable was absent). One line per distinct value, not one
#: per call -- the floor readers run on every credit wait and every dial.
_RESERVE_READ_NOTED: set = set()


def user_reserve_masked_by_card() -> Dict[str, int]:
    """``{card_uuid: stated MiB}`` the launcher masked to 0 for this group (booked
    awake rest); ``{}`` when absent or unreadable. Diagnostic only."""
    raw = os.environ.get(USER_RESERVE_MASKED_ENV)
    if not raw:
        return {}
    try:
        import json

        return {str(k): max(0, int(v)) for k, v in json.loads(raw).items()}
    except Exception:  # noqa: BLE001 - a malformed hint is no hint
        return {}


def reserve_for_card(uuid: str, fallback_reserve_mib: int = 0) -> Tuple[int, str]:
    """THE rank-side choice of one card's reserve: ``(MiB, source)``.

    The published per-card mapping first (``env-by-card``, or
    ``masked-by-booked-rest`` when the launcher booked the card's awake rest and
    therefore charged no reserve), the caller's scalar only where the mapping has
    no entry (``fallback-scalar``), else ``none``. ONE function for every rank
    reader of the floor (``corridor_floor_for_current_device``,
    ``weight_updater._pdflip_corridor_floor_bytes``), so two readers cannot
    resolve the same card differently (#236 review SOLL-2).
    """
    reserve = user_reserve_by_card().get(uuid)
    if reserve is not None:
        masked = user_reserve_masked_by_card()
        if uuid in masked:
            return int(reserve), "masked-by-booked-rest"
        return int(reserve), "env-by-card"
    scalar = max(0, int(fallback_reserve_mib or 0))
    return scalar, ("fallback-scalar" if scalar else "none")


def _note_reserve_read(raw: Optional[str], parsed: Dict[str, int]) -> None:
    """#236: name, once per process, WHERE this process's reserve came from.

    The rank logs ``source=env:<NAME>`` with the per-card mapping it read, or
    ``source=none`` when the launcher published nothing -- so a boot log shows
    at a glance whether a rank priced its floor with the operator's reserve
    (the front and the launcher print the same mapping on their lines). Never
    a number the reader invented: ``value`` is the parsed operator input.
    """
    if raw in _RESERVE_READ_NOTED:
        return
    _RESERVE_READ_NOTED.add(raw)
    if raw is None:
        logger.info(
            "PDFLIP-USER-RESERVE-READ source=none (%s unset: this process prices "
            "its floor with the --rank-user-reserve-mib scalar fallback or 0) "
            "value={} pid=%d",
            USER_RESERVE_ENV,
            os.getpid(),
        )
        return
    masked = user_reserve_masked_by_card()
    logger.info(
        "PDFLIP-USER-RESERVE-READ source=env:%s value=%s (effective, as the launcher "
        "booked it for this group) masked_by_booked_rest=%s pid=%d",
        USER_RESERVE_ENV,
        dict(sorted(parsed.items())),
        dict(sorted(masked.items())),
        os.getpid(),
    )


def user_reserve_by_card() -> Dict[str, int]:
    """``{card_uuid: MiB}`` of operator-reserved headroom. THE reader.

    REFUTER FINDING 6 (2026-09-09). ``pdflip.front`` had the only reader of
    :data:`USER_RESERVE_ENV`, so a second consumer of the derived floor
    (``vram_dial``) had no way to see the reserve without restating the
    parse. Declared beside the variable it reads: a malformed value is NO
    reserve and says so, never a guessed number.
    """
    raw = os.environ.get(USER_RESERVE_ENV)
    if not raw:
        _note_reserve_read(None, {})
        return {}
    try:
        import json

        parsed = {str(k): max(0, int(v)) for k, v in json.loads(raw).items()}
        _note_reserve_read(raw, parsed)
        return parsed
    except Exception:  # noqa: BLE001 - a malformed hint is no reserve, loudly
        logger.warning(
            "PDFLIP-CORRIDOR %s=%r is unreadable; reserve read as 0",
            USER_RESERVE_ENV,
            raw,
        )
        return {}


def _floor_digest_path() -> str:
    override = os.environ.get(FLOOR_DIGEST_FILE_ENV)
    if override:
        return override
    try:
        from flliper.srt.rigmon.card_probe import CACHE_DIR
    except Exception:  # pragma: no cover - rigmon import shape
        CACHE_DIR = os.path.expanduser("~/.cache/flliper")
    return os.path.join(CACHE_DIR, FLOOR_DIGEST_FILE_DEFAULT)


def record_identity_enabled() -> bool:
    """#233: is the identity-keyed pointer read/write ON? Default ON."""
    raw = os.environ.get(RECORD_IDENTITY_ENV)
    if raw is None:
        return True
    return str(raw).strip().lower() not in ("0", "off", "false", "no")


def boot_record_identity(environ=None) -> Optional[Dict[str, str]]:
    """This boot's calibration identity for the corridor pointer, or ``None``.

    #233. NOT A NEW IDENTITY: the terms are the ones
    ``pdflip_form.calibration_identity`` already accepts measured sources by
    -- the checkpoint (``model_key``, the directory name) and the form's
    ``arch`` -- plus the profile row id for the reader's benefit. They come
    from ``FLLIPER_PDFLIP_FORM``, the ONE variable the launcher publishes into
    its own environment before any floor is read and that ``build_env`` and
    the front hand to every rank, so the writer (a group rank) and every
    reader (launcher, front, ranks) compute the SAME key. ``None`` = no form
    published (a desk test, a non-pdflip process): the caller keeps the
    pre-#233 group-keyed read and says that the identity is unverified.
    """
    try:
        from flliper.srt.pdflip import form as pdflip_form

        form = pdflip_form.current_form(environ)
    except Exception as exc:  # pragma: no cover - import shape
        logger.debug("pdflip form unreadable for the corridor identity: %s", exc)
        return None
    if form is None or not str(form.model or "").strip():
        return None
    return {
        "checkpoint": str(form.model).strip(),
        "arch": str(form.arch or ""),
        "profile": str(form.profile or ""),
    }


def record_identity_prefix(identity: Mapping[str, object]) -> str:
    """``"<checkpoint>|<arch>"`` -- the model part of a ``records`` key."""
    return f"{identity.get('checkpoint', '')}|{identity.get('arch', '')}"


def record_identity_key(identity: Mapping[str, object], recipe: str = "") -> str:
    """``"<checkpoint>|<arch>|<recipe>"`` -- the ``records`` key of one identity.

    #233 review MUSS 2: the recipe is IN the key. An empty recipe (a process
    the launcher did not arm) yields a key no armed reader ever looks up.
    """
    return f"{record_identity_prefix(identity)}|{recipe or ''}"


def recipe_relevant_tokens(tokens: Sequence[str]) -> List[str]:
    """Only the :data:`RECIPE_FLAGS` / :data:`RECIPE_BOOL_FLAGS` of ``tokens``.

    The launcher's recipe probe hands the operator's ``--extra-*`` through the
    group argv builders with SENTINEL budgets; the budget-raising refusal (W100)
    and every other ``extra`` rule must not see the operator's non-recipe flags
    there, and the recipe items are the same either way (they are read off these
    flags only).
    """
    out: List[str] = []
    toks = [str(t) for t in tokens]
    i = 0
    while i < len(toks):
        t = toks[i]
        name = t.split("=", 1)[0]
        if name in RECIPE_BOOL_FLAGS:
            out.append(t)
        elif name in RECIPE_FLAGS:
            out.append(t)
            if "=" not in t and i + 1 < len(toks):
                out.append(toks[i + 1])
                i += 1
        i += 1
    return out


def recipe_from_argv(argv: Sequence[str]) -> Dict[str, object]:
    """``{"hash": str, "items": {flag: value}}`` of a group argv's recipe.

    argparse semantics: the LAST value of a flag wins, ``--flag v`` and
    ``--flag=v`` alike; a store-true flag present is ``"1"``. Aliases fold onto
    one item. The hash is over the sorted items only -- never the group (the
    record names the group), never a path.
    """
    import hashlib
    import json

    items: Dict[str, str] = {}
    toks = [str(t) for t in argv]
    for i, t in enumerate(toks):
        name, eq, val = t.partition("=")
        if name in RECIPE_BOOL_FLAGS:
            items[name] = "1"
        elif name in RECIPE_FLAGS:
            if not eq:
                if i + 1 >= len(toks):
                    continue
                val = toks[i + 1]
            items[_RECIPE_ALIASES.get(name, name)] = str(val)
    blob = json.dumps(sorted(items.items()), separators=(",", ":"))
    return {"hash": hashlib.sha1(blob.encode()).hexdigest()[:12], "items": items}


def boot_recipes(environ=None) -> Dict[str, Dict[str, object]]:
    """``{group: {"hash", "items"}}`` the launcher published (:data:`RECIPE_ENV`).

    ``{}`` when absent or unreadable -- an unarmed process has no recipe, and
    without one no own record can be found (the safe direction).
    """
    env = os.environ if environ is None else environ
    raw = env.get(RECIPE_ENV)
    if not raw:
        return {}
    try:
        import json

        got = json.loads(raw)
    except ValueError:
        logger.warning("%s %s=%r is unreadable; no recipe for this boot", RECIPE_TAG, RECIPE_ENV, raw)
        return {}
    out: Dict[str, Dict[str, object]] = {}
    for g, v in (got or {}).items():
        if isinstance(v, dict) and v.get("hash"):
            out[normalise_group(g)] = {
                "hash": str(v["hash"]),
                "items": dict(v.get("items") or {}),
            }
    return out


def arm_group_recipe_env(group: str, argv: Sequence[str], env) -> List[str]:
    """``launch_group``'s half of MUSS 1 + 2: arm one group's environment.

    * :data:`RECIPE_SELF_ENV` = the recipe of the group's REAL argv -- the key
      the group's ranks publish under, for the NEXT boot;
    * :data:`POINTER_FROZEN_ENV` / :data:`RECIPE_ENV` copied from the
      launcher's own environment (launcher OUTPUT, the R19 discipline: an
      ``--env-*`` value or a shell leftover never describes this boot;
      ``build_env`` pops both, so this is the ONE place a group gets them);
    * the launcher's pre-budget recipe probe is compared with the real argv. A
      difference is :data:`RECIPE_DRIFT_TAG`, named per flag: the probe missed
      a footprint-relevant input, so the record this boot writes (under the
      REAL recipe) is one the probe of the same command never finds -> the next
      boot reads the law, the safe direction, and this line says why.

    Returns the log lines; never raises.
    """
    lines: List[str] = []
    tag = normalise_group(group)
    try:
        real = recipe_from_argv(argv)
    except Exception as exc:  # pragma: no cover - argv shape
        env.pop(RECIPE_SELF_ENV, None)
        return [f"{RECIPE_DRIFT_TAG} group={tag}: real argv recipe unreadable ({exc}); this group publishes no record"]
    env[RECIPE_SELF_ENV] = str(real["hash"])
    for k in (POINTER_FROZEN_ENV, RECIPE_ENV):
        if os.environ.get(k):
            env[k] = os.environ[k]
        else:
            env.pop(k, None)
    probe = boot_recipes().get(tag)
    if probe is None:
        lines.append(
            f"{RECIPE_DRIFT_TAG} group={tag} recipe={real['hash']}: the launcher published "
            f"no pre-budget recipe for this group, so no reader of this boot found an "
            f"own record; the group publishes under its real recipe for the next boot"
        )
    elif probe["hash"] != real["hash"]:
        pi, ri = dict(probe.get("items") or {}), dict(real["items"])
        diff = {
            k: (pi.get(k), ri.get(k))
            for k in sorted(set(pi) | set(ri))
            if pi.get(k) != ri.get(k)
        }
        lines.append(
            f"{RECIPE_DRIFT_TAG} group={tag} probe={probe['hash']} argv={real['hash']} "
            f"differs (probe, argv): {diff} -- the pre-budget recipe probe missed a "
            f"footprint-relevant input; this boot's record goes under the argv recipe, "
            f"which the probe never finds, so the next boot of this command reads the law"
        )
    else:
        lines.append(
            f"{RECIPE_TAG} group={tag} recipe={real['hash']} items={real['items']} "
            f"(the pre-budget probe and the real group argv agree)"
        )
    return lines


#: ``{(profile, checkpoint): (architectures, source)}`` -- one config read per
#: process; the checkpoint does not change under a boot, and the rank-local
#: readers (``kv_vmm_backing`` and friends) ask per call.
_ARCH_CACHE: Dict[Tuple[str, str], Tuple[Tuple[str, ...], str]] = {}


def boot_architectures(
    identity: Optional[Mapping[str, object]],
) -> Tuple[Tuple[str, ...], str]:
    """HF ``architectures`` of this boot's checkpoint, sorted; ``((), why)``.

    ``identity["architectures"]`` when the caller states it (the writer, which
    holds the real ``ActivationProfile``). Otherwise the registry's path of the
    checkpoint's format (``pdflip_form.format_of``; a listed derivative maps to
    its base, whose config.json is the derivative's by definition) -- "the rig
    mounts every checkpoint under the registry's path". Unreadable is NOT a
    mismatch: it is reported as such and the caller treats an unprovable
    origin as not-own.
    """
    if not identity:
        return (), "no model identity"
    given = identity.get("architectures")
    if given:
        return tuple(sorted(str(a) for a in given)), "stated by the writer"
    key = (str(identity.get("profile") or ""), str(identity.get("checkpoint") or ""))
    if key in _ARCH_CACHE:
        return _ARCH_CACHE[key]
    out: Tuple[Tuple[str, ...], str]
    try:
        import json

        from flliper.srt.pdflip import form as pdflip_form

        fmt = pdflip_form.format_of(key[0], key[1])
        row = pdflip_form.profile_row(key[0])
        wf = row.formats.get(fmt) if (row is not None and fmt) else None
        ck = str(getattr(wf, "checkpoint", "") or "")
        if not ck:
            out = ((), f"checkpoint {key[1]!r} is not a registry format of profile {key[0]!r}")
        else:
            cfg = os.path.join(ck, "config.json")
            with open(cfg, "r", encoding="utf-8") as fh:
                archs = (json.load(fh) or {}).get("architectures") or ()
            out = (
                (tuple(sorted(str(a) for a in archs)), cfg)
                if archs
                else ((), f"{cfg} names no architectures")
            )
    except (OSError, ValueError) as exc:
        out = ((), f"checkpoint config unreadable ({exc})")
    except Exception as exc:  # pragma: no cover - registry shape
        out = ((), f"model registry unreadable ({exc})")
    _ARCH_CACHE[key] = out
    return out


def _footprint_architectures(
    hw_fingerprint: str, digest: str, cache_dir: Optional[str] = None
) -> Tuple[Tuple[str, ...], str]:
    """The ``architectures`` a footprint file states IN ITS OWN CONTENT.

    #233 migration rule: a legacy pointer entry carries no identity, but the
    footprint it names does (``profile[0]`` of the canonical profile). That is
    the only identity a legacy record can prove.
    """
    try:
        import json

        from flliper.srt.mem_ledger import activation as _act

        path = _act.footprint_cache_path(hw_fingerprint, digest, cache_dir)
        with open(path, "r", encoding="utf-8") as fh:
            prof = (json.load(fh) or {}).get("profile") or []
        archs = prof[0] if prof and isinstance(prof[0], (list, tuple)) else ()
        if not archs:
            return (), f"{path} states no architectures"
        return tuple(sorted(str(a) for a in archs)), path
    except (OSError, ValueError) as exc:
        return (), f"footprint content unreadable ({exc})"
    except Exception as exc:  # pragma: no cover - import shape
        return (), f"footprint content unreadable ({exc})"


_RECORD_NOTED: set = set()


def _note_record_once(line: str) -> None:
    """Every #233 verdict line once per process, at WARNING (it moves a floor)."""
    if line in _RECORD_NOTED:
        return
    _RECORD_NOTED.add(line)
    logger.warning("%s", line)


class _PointerLock:
    """Advisory ``flock`` around the pointer's read-modify-write.

    #233 (a) makes every group rank publish (P and D boot side by side), so two
    writers can now meet; without the lock one group's entry is lost to the
    other's ``os.replace``. Best effort: no ``fcntl`` = no lock, same as before.
    """

    def __init__(self, path: str):
        self._path = path + ".lock"
        self._fh = None

    def __enter__(self):
        try:
            import fcntl

            self._fh = open(self._path, "a")
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        except Exception:  # pragma: no cover - platform
            self._fh = None
        return self

    def __exit__(self, *exc):
        if self._fh is not None:
            try:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            except Exception:  # pragma: no cover
                pass
            self._fh.close()
        return False


#: Sentinel: "take this boot's identity from the published form".
_IDENTITY_FROM_FORM = object()

#: SOLL 1 (review #233): the self-correction hangs on the publish now, so a
#: silent write failure (read-only mount, rights) would renew the #232 class
#: "pointer stands forever" invisibly. One WARNING per process and reason.
_PUBLISH_FAILED_NOTED: set = set()


def _warn_publish_failed_once(group: str, why: str) -> None:
    key = (group, why.split(":", 1)[0])
    if key in _PUBLISH_FAILED_NOTED:
        return
    _PUBLISH_FAILED_NOTED.add(key)
    logger.warning(
        "%s group=%s FAILED: %s -- the corridor pointer is NOT updated by this "
        "process, so the next boot reads whatever stands there now (the #232 "
        "class 'pointer stands forever' if nobody else writes it)",
        RECORD_PUBLISH_TAG, group, why,
    )


def publish_floor_profile(
    group: str,
    profile,
    hw_fingerprint: Optional[str],
    identity=_IDENTITY_FROM_FORM,
    recipe: Optional[str] = None,
) -> Optional[str]:
    """Record WHICH activation profile the awake group booted with.

    Writes a digest and a fingerprint; never a measurement. Returns the path,
    or ``None`` when there was nothing to publish -- publication is best effort
    by construction, because a rig that cannot write it simply reads
    UNMEASURED-FALLBACK, which is verdict-only and therefore safe. A failure
    is a named WARNING once per process (SOLL 1).

    #233: with a model identity (``FLLIPER_PDFLIP_FORM`` published, or
    ``identity`` passed) the digest goes under
    ``records["<checkpoint>|<arch>|<recipe>"]`` (review MUSS 2: the recipe of
    this group's real argv, :data:`RECIPE_SELF_ENV` unless ``recipe`` is
    passed), beside the identity terms and the profile's HF architectures; the
    legacy ``groups`` map is left exactly as it was. Without an identity the
    pre-#233 shape is written, because there is nothing to key by. The write is
    for the NEXT boot: this boot's readers hold the launcher's frozen state
    (MUSS 1).
    """
    tag = normalise_group(group)
    if tag == GROUP_UNKNOWN or not hw_fingerprint:
        return None
    ident = boot_record_identity() if identity is _IDENTITY_FROM_FORM else identity
    if not record_identity_enabled():
        ident = None
    rcp = os.environ.get(RECIPE_SELF_ENV, "") if recipe is None else str(recipe)
    try:
        import json

        from flliper.srt.mem_ledger.activation import profile_key

        digest = profile_key(profile)
        path = _floor_digest_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with _PointerLock(path):
            cur = {}
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        cur = json.load(fh) or {}
                except (OSError, ValueError):
                    cur = {}
            if cur.get("hw_fingerprint") != hw_fingerprint:
                cur = {"hw_fingerprint": hw_fingerprint, "groups": {}}
            if ident is None:
                groups = dict(cur.get("groups") or {})
                groups[tag] = digest
                cur["groups"] = groups
                key = "(no identity: legacy groups)"
            else:
                key = record_identity_key(ident, rcp)
                records = dict(cur.get("records") or {})
                rec = dict(records.get(key) or {})
                rec["identity"] = {
                    "checkpoint": str(ident.get("checkpoint") or ""),
                    "arch": str(ident.get("arch") or ""),
                    "profile": str(ident.get("profile") or ""),
                    "recipe": rcp,
                    "architectures": sorted(
                        str(a) for a in (getattr(profile, "architectures", ()) or ())
                    ),
                }
                rec_groups = dict(rec.get("groups") or {})
                rec_groups[tag] = digest
                rec["groups"] = rec_groups
                records[key] = rec
                cur["records"] = records
            cur["hw_fingerprint"] = hw_fingerprint
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(cur, fh, indent=1)
            os.replace(tmp, path)
        logger.info(
            "%s group=%s identity=%s digest=%s hw_fingerprint=%s path=%s",
            RECORD_PUBLISH_TAG, tag, key, digest, hw_fingerprint, path,
        )
        return path
    except Exception as exc:  # best effort by design, but never silent (SOLL 1)
        _warn_publish_failed_once(tag, f"write: {exc}")
        return None


#: ``{group: digest}`` this process already published (SOLL 5: ONE write per
#: group and process; a second, different digest of the same group is refused
#: by name instead of overwriting the first).
_BOOT_PUBLISHED: Dict[str, str] = {}


def publish_boot_floor_profile(
    server_args, *, profile=None, footprint_hit: Optional[bool] = None
) -> Optional[str]:
    """#233 (a): PUBLISH THIS GROUP'S PROFILE DIGEST, HIT OR MISS.

    The pre-#233 comment promised "every boot republishes its own digest, so
    the pointer self-corrects from the second boot". It never did: the only
    writer sat behind ``if footprint is not None`` in
    ``ServerArgs.activation_reserve_mb`` -- a boot WITHOUT a record of its own
    (NF: architecture ``Qwen4Exp...`` has no footprint) never wrote, so the
    27B pointer copied into the NF cache stood forever. Measured 24.10.: that
    method is not even reached on either line's group ranks (0 heuristic lines
    in the NF and the 27B D.log), so the pointer on both rigs was the 09.09.
    hand seed. This is the writer every group rank passes
    (``ModelRunner.alloc_memory_pool``), and it publishes on a miss too: the
    boot's own digest then resolves to "no phase footprint" -> the named law,
    never to another model's record.

    The fingerprint is the RIG's (``rig_hardware_fingerprint``, NVML, CVD-
    blind), the same one the readers compare against; the CVD-masked
    ``live_fingerprint`` of a pinned rank would be refused by every reader.

    The record is filed under this group's REAL argv recipe
    (:data:`RECIPE_SELF_ENV`, armed by ``launch_group``) and read by the NEXT
    boot; this boot's readers hold the launcher's frozen state (MUSS 1).
    """
    group = os.environ.get(GROUP_ENV, "")
    tag = normalise_group(group)
    if tag == GROUP_UNKNOWN:
        return None
    try:
        if profile is None:
            from flliper.srt.mem_ledger.activation import profile_from_server_args
            from flliper.srt.mem_ledger.engine import _model_architectures

            profile = profile_from_server_args(
                server_args, _model_architectures(server_args)
            )
        from flliper.srt.mem_ledger.activation import profile_key

        digest = profile_key(profile)
    except Exception as exc:  # config availability -- named, once (SOLL 1)
        _warn_publish_failed_once(tag, f"profile not derivable: {exc}")
        return None
    prior = _BOOT_PUBLISHED.get(tag)
    if prior == digest:
        return None
    if prior is not None:
        logger.warning(
            "%s group=%s SECOND DIGEST %s REFUSED: this process already published "
            "%s for group %s (one write per group and process; a second profile "
            "in one process would overwrite the first record)",
            RECORD_PUBLISH_TAG, tag, digest, prior, tag,
        )
        return None
    fp, why = rig_hardware_fingerprint()
    if not fp:
        logger.info("%s group=%s SKIPPED: %s", RECORD_PUBLISH_TAG, tag, why)
        return None
    path = publish_floor_profile(group, profile, fp)
    if path:
        _BOOT_PUBLISHED[tag] = digest
        logger.info(
            "%s group=%s digest=%s recipe=%s footprint=%s (published on %s, for "
            "the next boot; this boot reads the launcher's frozen pointer)",
            RECORD_PUBLISH_TAG, tag, digest, os.environ.get(RECIPE_SELF_ENV) or "(none)",
            {True: "hit", False: "miss", None: "not-resolved-here"}[footprint_hit],
            "a hit" if footprint_hit else "a miss too (#233 a)",
        )
    return path


def _read_pointer() -> dict:
    """The pointer file as a dict, ``{}`` when absent or unreadable."""
    try:
        import json

        path = _floor_digest_path()
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh) or {}
        return raw if isinstance(raw, dict) else {}
    except Exception as exc:  # pragma: no cover
        logger.debug("corridor floor digest unreadable: %s", exc)
        return {}


def _published_groups() -> Tuple[Optional[str], Dict[str, str]]:
    """``(hw_fingerprint, {group: digest})`` from the LEGACY pointer map."""
    raw = _read_pointer()
    fp = raw.get("hw_fingerprint")
    groups = {str(k): str(v) for k, v in (raw.get("groups") or {}).items() if v}
    return (str(fp) if fp else None), groups


def _frozen_snapshot() -> Optional[dict]:
    """The launcher's frozen pointer resolution (MUSS 1), or ``None``.

    ``None`` = no frozen state: this process was not started by a launcher of
    this tree (desk, planner CLI) and reads the file live, saying so. A value
    that is present but unreadable is NOT a reason to read live -- a live read
    is exactly the second floor MUSS 1 removes -- so it resolves every group to
    the law, named (``{"broken": why}``).
    """
    raw = os.environ.get(POINTER_FROZEN_ENV)
    if not raw:
        return None
    try:
        import json

        snap = json.loads(raw)
        if not isinstance(snap, dict) or snap.get("v") != 1:
            raise ValueError(f"unknown snapshot shape {type(snap).__name__} v={getattr(snap, 'get', lambda k: None)('v')!r}")
    except ValueError as exc:
        return {"v": 1, "broken": f"{POINTER_FROZEN_ENV} unreadable ({exc})"}
    if str(snap.get("pointer") or "") != _floor_digest_path():
        # Frozen for ANOTHER pointer file (a leftover of an earlier main() in
        # this process, e.g. a desk test that ran the launcher): it does not
        # describe this pointer, so it is not this process's boot state. A
        # real boot never moves its pointer path between freeze and read.
        return None
    return snap


def _group_digest_pairs(raw: dict, ident, recipes) -> List[Tuple[str, str]]:
    """Every ``(group, digest)`` a boot of ``ident`` with ``recipes`` can reach.

    The legacy map plus -- #233 -- the records under this boot's own keys (one
    per group recipe). Another identity's or another recipe's record is not in
    it: its digests are never read by this boot, so they cannot be ambiguous
    for it.
    """
    pairs = [(str(k), str(v)) for k, v in (raw.get("groups") or {}).items() if v]
    if ident is not None:
        records = raw.get("records") if isinstance(raw.get("records"), dict) else {}
        for key in sorted({record_identity_key(ident, r["hash"]) for r in (recipes or {}).values() if r.get("hash")}):
            rec = records.get(key) or {}
            if isinstance(rec, dict):
                pairs.extend(
                    (str(k), str(v)) for k, v in (rec.get("groups") or {}).items() if v
                )
    return sorted(set(pairs))


def _published_group_digests() -> List[Tuple[str, str]]:
    """Every ``(group, digest)`` this boot's reads can reach -- FROZEN when the
    launcher froze the pointer (MUSS 1: the collision is judged on the same
    state the floors were), live otherwise."""
    snap = _frozen_snapshot()
    if snap is not None:
        if snap.get("broken"):
            return []
        return [(str(g), str(d)) for g, d in (snap.get("pairs") or [])]
    ident = boot_record_identity() if record_identity_enabled() else None
    return _group_digest_pairs(_read_pointer(), ident, boot_recipes())


def digest_group_collision(digest: str, tag: str) -> Optional[str]:
    """The OTHER canonical groups that published this same digest, or ``None``.

    REFUTER FIX 3 (2026-09-09). THE STORE IS KEYED BY DIGEST AND THE GROUP
    PICKS THE DIGEST -- after that the group plays no part.
    ``profile_from_server_args`` carries no phase field, so if P and D ever
    agree on (arch, chunked_prefill_size, tp, pp, kv_dtype, spec_draft_tokens,
    decode_max_bs) they compute the SAME digest, both write the SAME cache
    file, and whoever ingested last owns the bytes. D's floor would then be
    P's prefill peak stamped ``MEASURED-D``, ``actuates=yes``, and it would
    CUT a budget -- the exact scope error ``measured_floor_source`` says a
    reader must be able to see.

    The dump layer already refuses the mirror of this by name
    (``activation_probe`` / ``W18 PdFlipPhaseFootprintCollision``); this is the
    same refusal at the cache/floor layer. An ambiguous digest resolves to
    UNMEASURED-FALLBACK for EVERY group sharing it -- not just the second one
    -- because the file cannot say whose bytes it holds, and verdict-only is
    the safe direction.
    """
    if not digest:
        return None
    others = sorted(
        {
            normalise_group(g)
            for g, d in _published_group_digests()
            if d == digest and normalise_group(g) != normalise_group(tag)
        }
    )
    return ",".join(others) if others else None


def _published_digest(group: str) -> Tuple[Optional[str], Optional[str]]:
    """``(hw_fingerprint, profile_digest)`` for ``group`` from the LEGACY map.

    Pre-#233 accessor, kept for callers that want the raw group-keyed entry;
    the floor derivation reads :func:`_published_digest_for_boot`.
    """
    fp, groups = _published_groups()
    tag = normalise_group(group)
    digest = groups.get(tag)
    if not digest or not fp:
        return None, None
    return str(fp), str(digest)


def _legacy_recipe_verdict(
    hw_fingerprint: str,
    digest: str,
    cache_dir: Optional[str],
    items: Optional[Mapping[str, object]],
) -> Tuple[str, str]:
    """``("match"|"mismatch"|"undeterminable", detail)`` for a legacy entry.

    #233 review MUSS 2, the Altbestand half: a legacy pointer entry has no
    recipe, but the footprint it names records its PROFILE in its own content
    (``[architectures, chunk, tp, pp, kv_dtype, spec_draft_tokens,
    decode_max_bs]``). Those terms are compared with the flags this boot's
    recipe states. ``decode_max_bs`` is NOT compared, by reason: it prices the
    decode graph capture (``capture_mib``), the corridor floor reads only the
    prefill transient (``activation_mib``), and the group argv does not state it
    (server_args derives it). A term the argv does not state is undeterminable,
    never assumed equal.
    """
    if items is None:
        return "undeterminable", "this boot published no recipe (no launcher armed it)"
    try:
        import json

        from flliper.srt.mem_ledger import activation as _act

        path = _act.footprint_cache_path(hw_fingerprint, digest, cache_dir)
        with open(path, "r", encoding="utf-8") as fh:
            prof = (json.load(fh) or {}).get("profile") or []
    except Exception as exc:
        return "undeterminable", f"legacy footprint unreadable ({exc})"
    if not isinstance(prof, (list, tuple)) or len(prof) < 6:
        return "undeterminable", "the legacy footprint states no full profile"
    want = {
        "--chunked-prefill-size": prof[1],
        "--tp-size": prof[2],
        "--pp-size": prof[3],
        "--kv-cache-dtype": prof[4],
    }
    missing = [k for k in want if k not in items]
    if missing:
        return "undeterminable", f"this boot's group argv states no {missing}"
    diffs = {
        k: (want[k], items[k]) for k in want if str(items[k]) != str(want[k])
    }
    if "--speculative-num-draft-tokens" in items:
        spec_boot = items["--speculative-num-draft-tokens"]
    elif "--speculative-algorithm" not in items:
        spec_boot = 0
    else:
        return "undeterminable", (
            f"this boot runs --speculative-algorithm {items['--speculative-algorithm']} "
            f"without stating --speculative-num-draft-tokens"
        )
    if str(spec_boot) != str(prof[5]):
        diffs["--speculative-num-draft-tokens"] = (prof[5], spec_boot)
    if diffs:
        return "mismatch", f"(legacy footprint, this boot): {diffs}"
    return "match", (
        f"chunk {prof[1]}, tp {prof[2]}, pp {prof[3]}, kv {prof[4]}, spec {prof[5]} "
        f"equal to this boot's argv; decode_max_bs {prof[6] if len(prof) > 6 else '?'} "
        f"not compared (decode capture, not the prefill transient; unstated by the argv)"
    )


#: SOLL 2 (review #233): the identity does NOT reach the footprint store.
#: ``ActivationProfile`` has no checkpoint or quant field and the file is named
#: ``phase_footprint-<hw>-<digest>.json``, so INT8, NVFP4, FP8 and bf16 of one
#: architecture and recipe share ONE file even with records of their own. The
#: record says WHICH digest this model+recipe booted; the bytes behind it are
#: per architecture+recipe. Named in every OWN/LEGACY provenance.
_STORE_SHARED_NOTE = (
    "footprint store keyed by activation profile only (no checkpoint/quant term): "
    "every checkpoint of these architectures and this recipe reads the same file"
)


def _resolve_pointer(
    raw: dict,
    group: str,
    ident,
    recipe: Optional[Mapping[str, object]],
    cache_dir: Optional[str],
    path: str,
    enabled: bool,
) -> Tuple[Optional[str], Optional[str], str, Optional[str]]:
    """``(hw_fingerprint, digest, note, refusal)`` for ``group`` from ``raw``.

    PURE: one pointer state in, one resolution out -- the launcher calls it
    once per group to FREEZE the boot's state (MUSS 1), a process without a
    frozen state calls it on a live read. Precedence, strictly:

    1. no model identity (no form published, or the user switched the check
       off): the pre-#233 group-keyed read (the switch-off is named);
    2. this boot's own ``records["<checkpoint>|<arch>|<recipe>"]`` entry ->
       OWN (refused as FOREIGN if the record's stated architectures contradict
       this boot's);
    3. a legacy ``groups`` entry -> adopted only when the footprint it names
       states THIS boot's architectures AND this boot's recipe terms in its own
       content (the migration rule); a different architecture -> FOREIGN, a
       different recipe -> OTHER RECIPE, unprovable -> UNKNOWN ORIGIN;
    4. records of this model under other recipes only -> OTHER RECIPE;
    5. only other identities' records -> FOREIGN, named;
    6. nothing -> ``(None, None, "", None)``, the caller's "first boot" fallback.
    """
    tag = normalise_group(group)
    fp = str(raw.get("hw_fingerprint") or "") or None
    legacy = {str(k): str(v) for k, v in (raw.get("groups") or {}).items() if v}
    records = raw.get("records") if isinstance(raw.get("records"), dict) else {}
    if ident is None:
        d = legacy.get(tag)
        if not d or not fp:
            return None, None, "", None
        # No form published (a desk test, a non-pdflip process): the pre-#233
        # read, byte-identical -- such a process has no identity to compare.
        # Switched off by the user: the same read, but it SAYS so, because
        # off is the direction #232 measured as OOM.
        note = (
            ""
            if enabled
            else f"; #233 record identity check switched off ({RECORD_IDENTITY_ENV}=0)"
        )
        return fp, d, note, None
    rcp = str((recipe or {}).get("hash") or "")
    items = dict((recipe or {}).get("items") or {}) if rcp else None
    prefix = record_identity_prefix(ident)
    key = record_identity_key(ident, rcp)
    own = records.get(key) if (rcp and isinstance(records.get(key), dict)) else {}
    d = str((own.get("groups") or {}).get(tag) or "")
    if d and fp:
        rec_archs = tuple(
            sorted(
                str(a)
                for a in ((own.get("identity") or {}).get("architectures") or ())
            )
        )
        boot_archs, why = boot_architectures(ident)
        if rec_archs and boot_archs and rec_archs != boot_archs:
            line = (
                f"{RECORD_FOREIGN_TAG} group={tag} identity={key} digest={d} "
                f"pointer={path!r}: the record under this boot's key names "
                f"architectures {list(rec_archs)} and this boot's checkpoint "
                f"config names {list(boot_archs)} ({why}) -- another model's "
                f"corridor floor is never used"
            )
            return fp, None, "", line
        return fp, d, f"; {RECORD_OWN_TAG} identity={key} ({_STORE_SHARED_NOTE})", None

    def _holds(r) -> bool:
        return isinstance(r, dict) and bool((r.get("groups") or {}).get(tag))

    same_model = sorted(
        k for k, r in records.items()
        if k != key and (k == prefix or k.startswith(prefix + "|")) and _holds(r)
    )
    others = sorted(
        k for k, r in records.items()
        if k != key and k not in same_model and _holds(r)
    )
    ld = legacy.get(tag)
    if ld and fp:
        rec_archs, rec_why = _footprint_architectures(fp, ld, cache_dir)
        boot_archs, boot_why = boot_architectures(ident)
        if rec_archs and boot_archs and rec_archs == boot_archs:
            verdict, detail = _legacy_recipe_verdict(fp, ld, cache_dir, items)
            if verdict == "match":
                line = (
                    f"{RECORD_LEGACY_TAG} group={tag} identity={key} digest={ld}: "
                    f"legacy pointer entry (no identity) adopted -- the footprint "
                    f"it names states architectures {list(rec_archs)} in its own "
                    f"content ({rec_why}), the same as this boot's ({boot_why}), "
                    f"and its recipe: {detail}; {_STORE_SHARED_NOTE}; this boot "
                    f"republishes its own digest under its identity"
                )
                return fp, ld, f"; {line}", None
            if verdict == "mismatch":
                line = (
                    f"{RECORD_OTHER_RECIPE_TAG} group={tag} identity={key} digest={ld} "
                    f"pointer={path!r}: the legacy pointer entry names a footprint of "
                    f"this boot's architectures {list(rec_archs)} measured under "
                    f"another recipe {detail} -- another recipe's transient is never "
                    f"this boot's floor"
                )
            else:
                line = (
                    f"{RECORD_UNKNOWN_TAG} group={tag} identity={key} digest={ld} "
                    f"pointer={path!r}: the legacy pointer entry names a footprint of "
                    f"this boot's architectures {list(rec_archs)}, but its recipe "
                    f"cannot be matched ({detail}) -- an unknown recipe is not this "
                    f"boot's own record"
                )
            return fp, None, "", line
        if rec_archs and boot_archs:
            line = (
                f"{RECORD_FOREIGN_TAG} group={tag} identity={key} digest={ld} "
                f"pointer={path!r}: the legacy pointer entry (no identity) names "
                f"a footprint of architectures {list(rec_archs)} ({rec_why}) and "
                f"this boot is {list(boot_archs)} ({boot_why}) -- another "
                f"model's corridor floor is never used"
            )
        else:
            line = (
                f"{RECORD_UNKNOWN_TAG} group={tag} identity={key} digest={ld} "
                f"pointer={path!r}: the legacy pointer entry carries no model "
                f"identity and its origin cannot be proven (record: "
                f"{list(rec_archs) if rec_archs else rec_why}; this boot: "
                f"{list(boot_archs) if boot_archs else boot_why}) -- an "
                f"unknown origin is not this boot's own record"
            )
        return fp, None, "", line
    if same_model:
        line = (
            f"{RECORD_OTHER_RECIPE_TAG} group={tag} identity={key} pointer={path!r}: "
            f"the pointer holds group {tag} records of this model only under "
            f"{same_model}, none for this boot's recipe "
            f"{rcp or '(none published)'} -- another recipe's footprint is never "
            f"this boot's floor"
        )
        return fp, None, "", line
    if others:
        line = (
            f"{RECORD_FOREIGN_TAG} group={tag} identity={key} pointer={path!r}: "
            f"the pointer holds group {tag} records only for {others}, none "
            f"for this boot -- another model's corridor floor is never used"
        )
        return fp, None, "", line
    return None, None, "", None


def _pointer_source(digest: Optional[str], refusal: Optional[str]) -> str:
    if refusal:
        return POINTER_SOURCE_FOREIGN
    return POINTER_SOURCE_FROZEN if digest else POINTER_SOURCE_LAW


def clear_boot_pointer_env() -> None:
    """Drop a frozen state / recipe this process inherited (a shell, an earlier
    ``main()`` in the same process): launcher OUTPUT never describes a boot it
    did not resolve (R19)."""
    for k in (POINTER_FROZEN_ENV, RECIPE_ENV, RECIPE_SELF_ENV):
        os.environ.pop(k, None)
    _POINTER_READ_NOTED.clear()


def freeze_boot_pointer(
    groups: Sequence[str] = CANONICAL_GROUPS, cache_dir: Optional[str] = None
) -> List[str]:
    """MUSS 1: read the pointer ONCE, resolve it for ``groups``, freeze it.

    Called by the launcher before the first floor of the boot is read (the
    budget pass), after the form and the recipe are published. The resolution
    goes into :data:`POINTER_FROZEN_ENV` of THIS process, so the launcher's own
    budget and lend floor, the front (``dict(os.environ)``) and every rank
    (``launch_group`` -> :func:`arm_group_recipe_env`) read the SAME state. The ranks keep writing the file -- for the next boot.
    Returns the log lines (one per group, tag :data:`POINTER_FROZEN_TAG`).
    """
    import json
    import time

    raw = _read_pointer()
    path = _floor_digest_path()
    enabled = record_identity_enabled()
    ident = boot_record_identity() if enabled else None
    recipes = boot_recipes()
    snap = {
        "v": 1,
        "pointer": path,
        "hw_fingerprint": str(raw.get("hw_fingerprint") or ""),
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": os.getpid(),
        "groups": {},
        "pairs": [list(p) for p in _group_digest_pairs(raw, ident, recipes)],
    }
    lines: List[str] = []
    for g in groups:
        tag = normalise_group(g)
        fp, d, note, refusal = _resolve_pointer(
            raw, tag, ident, recipes.get(tag), cache_dir, path, enabled
        )
        rcp = str((recipes.get(tag) or {}).get("hash") or "")
        snap["groups"][tag] = {
            "fp": fp, "digest": d, "note": note, "refusal": refusal, "recipe": rcp,
        }
        what = refusal or note.lstrip("; ") or (
            "no record for this boot's identity and recipe (first boot of it: the law)"
        )
        lines.append(
            f"{POINTER_FROZEN_TAG} group={tag} source={_pointer_source(d, refusal)} "
            f"digest={d or '-'} recipe={rcp or '(none)'} identity="
            f"{record_identity_key(ident, rcp) if ident else '(none)'} pointer={path!r}: "
            f"{what} -- every reader of this boot (launcher budget and lend floor, "
            f"front, ranks) reads THIS state; the ranks' publish is for the next boot"
        )
    os.environ[POINTER_FROZEN_ENV] = json.dumps(snap, sort_keys=True)
    _POINTER_READ_NOTED.clear()
    return lines


_POINTER_READ_NOTED: set = set()


def _note_pointer_read_once(tag: str, source: str, digest: Optional[str], detail: str) -> None:
    """``#233 CORRIDOR-POINTER-READ group= source=FROZEN|LAW|FOREIGN|LIVE`` --
    once per process and (group, source, digest): the named line MUSS 1 asks
    for, so a rank log shows WHICH pointer state its floors came from."""
    k = (tag, source, digest or "")
    if k in _POINTER_READ_NOTED:
        return
    _POINTER_READ_NOTED.add(k)
    logger.info(
        "%s group=%s source=%s digest=%s pid=%d: %s",
        POINTER_READ_TAG, tag, source, digest or "-", os.getpid(), detail,
    )


def _published_digest_for_boot(
    group: str, cache_dir: Optional[str] = None
) -> Tuple[Optional[str], Optional[str], str, Optional[str]]:
    """``(hw_fingerprint, digest, note, refusal)`` for ``group`` and THIS boot.

    MUSS 1: the launcher's FROZEN resolution when it froze one -- the same
    state its budget and lend floor were priced from, no matter what the ranks
    have published since. Without a frozen state (a process the launcher did
    not start) the file is read live and resolved by the same pure function
    (:func:`_resolve_pointer`, precedence there); the read line says LIVE.

    A ``refusal`` is a full provenance sentence with a stable tag; the caller
    falls back to the stated law with it. It is never a measurement. With a
    refusal the pointer's fingerprint is still returned, so the caller's
    REFUTER FIX 4 hardware check runs FIRST: "another rig" stays the verdict
    where it already was, and the refusal is logged only where it decides.
    """
    tag = normalise_group(group)
    snap = _frozen_snapshot()
    if snap is not None:
        if snap.get("broken"):
            line = (
                f"{RECORD_UNKNOWN_TAG} group={tag}: {snap['broken']} -- a live read "
                f"would be a second floor in this boot (MUSS 1), so the law holds"
            )
            _note_pointer_read_once(tag, POINTER_SOURCE_LAW, None, line)
            return None, None, "", line
        ent = (snap.get("groups") or {}).get(tag)
        if not isinstance(ent, dict):
            _note_pointer_read_once(
                tag, POINTER_SOURCE_LAW, None,
                f"group {tag} is not in the launcher's frozen pointer state (frozen: "
                f"{sorted((snap.get('groups') or {}))}) -- the law",
            )
            return None, None, "", None
        fp = ent.get("fp") or None
        d = ent.get("digest") or None
        note = str(ent.get("note") or "")
        refusal = ent.get("refusal") or None
        _note_pointer_read_once(
            tag, _pointer_source(d, refusal), d,
            f"launcher's frozen state ({snap.get('frozen_at')}, pid {snap.get('pid')}, "
            f"recipe {ent.get('recipe') or '(none)'}): "
            f"{refusal or note.lstrip('; ') or 'no record (the law)'}",
        )
        return fp, d, note, refusal
    raw = _read_pointer()
    enabled = record_identity_enabled()
    ident = boot_record_identity() if enabled else None
    recipe = boot_recipes().get(tag)
    out = _resolve_pointer(
        raw, tag, ident, recipe, cache_dir, _floor_digest_path(), enabled
    )
    _note_pointer_read_once(
        tag, POINTER_SOURCE_LIVE, out[1],
        f"no frozen pointer state in this process ({POINTER_FROZEN_ENV} unset: not "
        f"started by a launcher of this tree); read live -> "
        f"{_pointer_source(out[1], out[3])}",
    )
    return out


#: Cached ``(fingerprint, why_not)`` of the WHOLE rig. One NVML inventory per
#: process: :func:`corridor_floors_for_cards` asks per card, and the rig does
#: not change under a boot.
_RIG_FP_CACHE: List[Optional[str]] = []


def rig_hardware_fingerprint() -> Tuple[Optional[str], Optional[str]]:
    """``(fingerprint, why_not)`` for the rig this process runs on.

    ``calibration.rig_fingerprint``, NOT ``live_fingerprint``: the latter is
    built from ``torch.cuda`` and therefore describes the CVD-masked slice the
    calling process can see. Each window-5 rank fingerprinted only its own
    card (``fad5762191c9`` / ``9ce0ed6a79fc`` / ``b0c936f23170``) against the
    rig's ``a191a0712717`` (#589) -- so inside a pinned Weg-2 rank the masked
    form can never match a stored footprint, and outside one it would happily
    match the wrong rig. NVML ignores CVD, so its device list IS the rig.

    Returns ``(None, why)`` when no inventory can be read at all (no NVML, a
    hermetic test box). That is NOT a mismatch and must never be read as one:
    a check that cannot run fails OPEN and says so in the provenance.
    """
    if _RIG_FP_CACHE:
        return _RIG_FP_CACHE[0], _RIG_FP_CACHE[1]
    fp: Optional[str] = None
    why: Optional[str] = None
    try:
        from flliper.srt.mem_ledger.calibration import rig_fingerprint

        rig = rig_fingerprint()
        if rig and rig[0]:
            fp = str(rig[0])
        else:
            why = "no NVML rig inventory"
    except Exception as exc:  # pragma: no cover - import/NVML shape
        why = f"rig fingerprint unavailable ({exc})"
    _RIG_FP_CACHE.extend([fp, why])
    return fp, why


@dataclass(frozen=True)
class CorridorFloor:
    """One card's corridor floor, with the provenance of both of its terms.

    ``mib`` is what the floor IS. ``verdict_floor_mib`` is what BELOW is graded
    against -- the same number when the transient is measured, the familiar
    -20 % band floor when it is not. ``actuates`` is the one predicate that
    decides whether a budget may be cut for this floor's sake.
    """

    card_uuid: str
    group: str
    transient_mib: int
    reserve_mib: int
    source: str
    provenance: str = ""
    basis: str = BASIS_STATED_LAW

    @property
    def mib(self) -> int:
        return int(self.transient_mib) + int(self.reserve_mib)

    @property
    def measured(self) -> bool:
        """Did somebody MEASURE this card's transient?

        The env override does NOT count, and the distinction is not pedantry:
        an operator who sets ``FLLIPER_CORRIDOR_LAW_FLOOR_MIB`` is moving the
        STATED law, which is a target and carries the +-20 % band exactly as
        the shipped 1024 does. Only a footprint the probe measured is a
        physical requirement with no slack under it.
        """
        return self.source.startswith("MEASURED-")

    @property
    def actuates(self) -> bool:
        """May a budget be CUT to buy this floor?

        Only when somebody priced it: a measured transient, an operator who
        moved the law by hand, or a user reserve that was actually asked for.
        An unmeasured 1024 with no reserve prints, grades and refuses -- it
        never spends KV pool.
        """
        return (
            self.measured
            or self.source == FLOOR_SOURCE_ENV
            or int(self.reserve_mib) > 0
        )

    @property
    def reason(self) -> str:
        """Why a cut for this floor is legitimate: the actuating term."""
        if self.measured and int(self.reserve_mib) > 0:
            return "measured-peak+user-reserve"
        if self.measured:
            return "measured-peak"
        if self.source == FLOOR_SOURCE_ENV:
            return "hand-set-law"
        if int(self.reserve_mib) > 0:
            return "user-reserve"
        return "unmeasured-fallback"

    @property
    def verdict_floor_mib(self) -> int:
        """Below this is a breach. No tolerance under a MEASURED peak.

        Delegates so that this object and a floor recovered from a log line
        (:func:`verdict_floor_for_mib`) can never grade differently.
        """
        return verdict_floor_for_mib(self.mib, self.source)

    @property
    def ceiling_mib(self) -> int:
        """Above this at rest is unmobilised free. A FINDING, never a FAIL."""
        return corridor_ceiling_for_floor_mib(self.mib)

    @property
    def line(self) -> str:
        """The grep-able provenance string every consumer prints verbatim."""
        return (
            f"CORRIDOR-FLOOR card={self.card_uuid} group={self.group} "
            f"floor={self.mib} verdict_floor={self.verdict_floor_mib} "
            f"ceiling={self.ceiling_mib} transient={self.transient_mib} "
            f"source={self.source} basis={self.basis} "
            f"reserve={self.reserve_mib} "
            f"actuates={'yes' if self.actuates else 'no'} "
            f"reason={self.reason} provenance={self.provenance or '(none)'}"
        )


def footprint_prefill_tokens(source: str) -> Optional[str]:
    """The ``prefill of N tokens`` the ingest wrote into a footprint's
    ``source`` (``N`` as written, possibly ``"?"``), or ``None`` if the source
    states no prefill at all."""
    import re

    m = re.search(r"prefill of (\S+) tokens", str(source or ""))
    return m.group(1) if m else None


def _thin_evidence_refusal(hit, path: str, card_uuid: str) -> Optional[str]:
    """:data:`RECORD_THIN_TAG` sentence when ``hit`` lacks the minimum evidence.

    Minimum (defined from what the footprint already records): a MEASURED
    footprint's stated prefill token count >= the ``chunked_prefill_size`` of
    the profile in the same file. Bounds and footprints that state no prefill
    are not judged (no count to compare).
    """
    prov = getattr(getattr(hit, "provenance", None), "value", getattr(hit, "provenance", ""))
    if str(prov) != "measured_peak":
        return None
    stated = footprint_prefill_tokens(getattr(hit, "source", ""))
    if stated is None:
        return None
    chunk = 0
    try:
        import json

        with open(path, "r", encoding="utf-8") as fh:
            prof = (json.load(fh) or {}).get("profile") or []
        chunk = int(prof[1]) if isinstance(prof, (list, tuple)) and len(prof) > 1 else 0
    except Exception:  # noqa: BLE001 - an unreadable profile has no minimum
        chunk = 0
    try:
        tokens = int(stated)
    except ValueError:
        tokens = -1
    if chunk > 0 and tokens >= chunk:
        return None
    why = (
        f"its profile states no chunk, so the minimum is unknown"
        if chunk <= 0
        else (
            f"it states 'prefill of {stated} tokens', no count"
            if tokens < 0
            else f"prefill of {tokens} tokens < one chunk ({chunk})"
        )
    )
    return (
        f"{RECORD_THIN_TAG} card={card_uuid} footprint={path}: activation "
        f"{int(getattr(hit, 'activation_mib', 0))} MiB was measured over too little "
        f"load ({why}) -- a peak that never ran a full-width forward is not this "
        f"recipe's transient and is never a MEASURED floor"
    )


def _resolve_transient_mib(
    card_uuid: str,
    group: str,
    hw_fingerprint: Optional[str],
    profile,
    profile_digest: Optional[str],
    cache_dir: Optional[str],
) -> Tuple[Optional[int], str]:
    """``(mib, provenance)`` from the ONE activation store, or ``(None, why)``.

    Exactly one read. Lazily imported for the same reason
    ``corridor_trace.corridor_law_mib`` lazily imports this module in the
    opposite direction: ``mem_ledger`` and ``managers`` must not tie their
    import graphs together.
    """
    try:
        from flliper.srt.mem_ledger import activation as _act
    except Exception as exc:
        return None, f"mem_ledger.activation not importable ({exc})"
    fp = hw_fingerprint
    digest = profile_digest
    unverified_fp = ""
    ident_note = ""
    if profile is None and digest is None:
        # #233: the pointer entry of THIS BOOT'S MODEL IDENTITY, never another
        # model's. A refusal (foreign record, legacy entry of unknown origin)
        # is a fallback to the stated law with the refusal as provenance.
        fp2, digest, ident_note, refusal = _published_digest_for_boot(
            group, cache_dir
        )
        # REFUTER FIX 4 (serve-next4, 2026-09-09). THE POINTER SUPPLIED BOTH
        # TERMS AND NOTHING CHECKED THE FIRST ONE. On this path the caller
        # passes no profile and no digest, so a pointer file written on OTHER
        # HARDWARE -- or before a card was swapped -- handed us its own
        # fingerprint, which then opened its own footprint file, which was
        # stamped MEASURED-* with actuates=yes and CUT a budget derived on a
        # rig this boot is not running on. The digest half of exactly this
        # hazard is refused by name below (``digest_group_collision``); this
        # is its fingerprint half.
        #
        # The reference is the CALLER's fingerprint when it passed one (free),
        # else the rig's (one NVML inventory). A check that cannot run --
        # no NVML on this box -- does not refuse; it says so in the
        # provenance, because "unverifiable" and "mismatched" are different
        # facts and only the second one is a reason to drop to fallback.
        if fp2:
            ref = fp
            if not ref:
                ref, why = rig_hardware_fingerprint()
                if not ref:
                    unverified_fp = (
                        f"; hardware fingerprint {fp2} taken from the pointer "
                        f"UNVERIFIED ({why})"
                    )
            if ref and str(ref) != str(fp2):
                return None, (
                    f"the published floor pointer at {_floor_digest_path()!r} "
                    f"was written on hardware fingerprint {fp2} and this rig "
                    f"is {ref} -- refusing to price a corridor floor off "
                    f"another rig's measurement (a mismatched pointer is a "
                    f"fallback, not a measurement)"
                )
        if refusal:
            _note_record_once(refusal)
            return None, refusal
        if RECORD_LEGACY_TAG in ident_note:
            _note_record_once(ident_note.lstrip("; "))
        fp = fp or fp2
        if digest is None:
            # THE FIRST BOOT IS ALWAYS UNMEASURED, and an operator reading this
            # should not have to work that out. Only a GROUP PROCESS can write
            # this pointer (``ServerArgs.activation_reserve_mb`` ->
            # :func:`publish_floor_profile`, with ``FLLIPER_PDFLIP_GROUP`` set),
            # and the LAUNCHER needs the floor BEFORE it starts that process --
            # so on a rig where this recipe has never booted, on a cleared
            # cache, or after any change to the activation profile, the floor
            # is honestly unmeasured and the pass is verdict-only. It measures
            # from the NEXT boot of the same recipe.
            #
            # STATE ON THIS RIG, 2026-09-09 (serve-next4 train), read from the
            # files rather than from the earlier note that stood here: the
            # pointer NOW EXISTS. It was seeded from boot weg2sb5h's own two
            # dumps through :func:`publish_floor_profile`, so the FIRST boot of
            # this recipe reads MEASURED floors:
            #   P -> 055c2e4b0867 (chunk 4096, tp 1, pp 3, fp8_e4m3, spec 0,
            #        decode_max_bs 24): 1055 / 1095 / 858 MiB
            #   D -> cc0ac5caf91e (chunk 4096, tp 3, pp 1, fp8_e4m3, spec 3,
            #        decode_max_bs 24):  767 /  700 / 701 MiB
            # The earlier note in this comment read "1055/1097/858"; the file
            # it named says 1095 on GPU-5c648f96, and the FILE is the
            # instrument. THE SEED IS ONLY AS GOOD AS THE RECIPE MATCHING:
            # this pointer is keyed by GROUP, not by profile, so a boot whose
            # P or D profile differs from the tuple above would read another
            # recipe's transient stamped MEASURED-*. Every boot republishes its
            # own digest, so the pointer self-corrects from the second boot.
            #
            # #233 (24.10.): THAT LAST SENTENCE WAS FALSE UNTIL #233. The only
            # writer published on a footprint HIT, and the NF (no record of its
            # own) never hit, so the 27B pointer copied into its cache stood
            # forever and priced the NF corridor at 767/701/700. Since #233 the
            # pointer is keyed by model identity (:func:`_published_digest_for_boot`)
            # and every group rank publishes hit or miss
            # (:func:`publish_boot_floor_profile`).
            return None, (
                f"no activation profile published for group {group!r} at "
                f"{_floor_digest_path()!r} (only a boot of that group writes "
                f"it, and the launcher reads the floor before that process "
                f"exists -- expected on the first boot of a recipe)"
            )
    if digest is None and profile is not None:
        try:
            digest = _act.profile_key(profile)
        except Exception:  # pragma: no cover - profile shape
            digest = None
    # REFUTER FIX 3: the digest must belong to exactly ONE group, on BOTH
    # paths -- the boot owns a real profile and still reads a cache file the
    # other group may have written under the same key. An ambiguous digest is
    # refused rather than attributed, and the refusal names the sharers.
    shared = digest_group_collision(digest, group) if digest else None
    if shared:
        return None, (
            f"profile digest {digest} is published by groups {shared} as well "
            f"as {normalise_group(group)}, so the one activation store cannot "
            f"attribute its bytes to a group (the same ambiguity "
            f"{PHASE_FOOTPRINT_COLLISION_NAME} refuses at the dump layer); "
            f"refusing to stamp another group's transient as this one's"
        )
    if not fp:
        # REFUTER FIX 4, same site: this used to call ``live_fingerprint``,
        # which is CVD-masked -- inside a pinned rank it fingerprints one card
        # and matches nothing (#589). It failed safe (a miss becomes the
        # fallback) but it could never measure. ``rig_hardware_fingerprint``
        # is the unmasked form.
        fp, why = rig_hardware_fingerprint()
        if not fp:
            return None, f"rig hardware fingerprint unavailable ({why})"
    try:
        if profile is not None:
            hit = _act.resolve_phase_footprint(
                card_uuid, hw_fingerprint=fp, profile=profile, cache_dir=cache_dir
            )
        else:
            hit = _act.load_footprints_by_digest(
                hw_fingerprint=fp, profile_digest=digest, cache_dir=cache_dir
            ).get(card_uuid)
    except Exception as exc:
        return None, f"phase footprint unreadable ({exc})"
    if hit is None:
        return None, (
            f"no phase footprint for card {card_uuid} at fingerprint {fp} "
            f"digest {digest or '(from profile)'}{ident_note}"
        )
    path = _act.footprint_cache_path(
        fp, digest or _act.profile_key(profile), cache_dir
    )
    # #233 review MUSS 2: a footprint measured over less than one chunk of
    # prefill is not the recipe's transient -- never MEASURED, never actuating.
    thin = _thin_evidence_refusal(hit, path, card_uuid)
    if thin:
        _note_record_once(thin)
        return None, thin
    return int(hit.activation_mib), (
        f"{path}#cards.{card_uuid}.activation_mib={int(hit.activation_mib)}MiB "
        f"({getattr(hit.provenance, 'value', hit.provenance)}){unverified_fp}"
        f"{ident_note}"
    )


def corridor_floor_mib(
    card_uuid: str,
    *,
    group: Optional[str] = None,
    hw_fingerprint: Optional[str] = None,
    profile=None,
    profile_digest: Optional[str] = None,
    user_reserve_mib: int = 0,
    cache_dir: Optional[str] = None,
) -> CorridorFloor:
    """THE derivation. Every consumer of a corridor floor calls this one.

    ``group`` is the group AWAKE on this card, because the floor is the
    transient of whoever is running -- not of whoever is asleep. It defaults to
    :data:`GROUP_ENV` and then to ``"?"``, which can only ever resolve to the
    fallback (no digest is published under ``"?"``).

    Pass ``profile`` when the caller owns a real ``ActivationProfile`` (the
    boot). Pass neither and the digest published by the boot is used, which is
    how the front and the launcher read the same measurement without restating
    the profile.
    """
    # REFUTER FIX 2: canonical, so a decorated label ("D(dry, expectation)")
    # and its bare group ("D") key ONE floor. A tag that is not a group
    # becomes GROUP_UNKNOWN, under which no digest is ever published, so it
    # reaches the fallback instead of silently missing a lookup.
    asked = group if group is not None else os.environ.get(GROUP_ENV)
    tag = normalise_group(asked)
    if tag == GROUP_UNKNOWN and str(asked or "").strip():
        # LOUD, ONCE PER TAG. A tag that is not a group cannot resolve a
        # measured floor and would otherwise fail as a silent fallback --
        # indistinguishable from "nobody has measured this rig yet".
        _warn_unknown_group(asked)
    reserve = max(0, int(user_reserve_mib or 0))

    raw = os.environ.get(LAW_ENV)
    if raw is not None:
        try:
            return CorridorFloor(
                card_uuid=card_uuid,
                group=tag,
                transient_mib=max(0, int(raw)),
                reserve_mib=reserve,
                source=FLOOR_SOURCE_ENV,
                provenance=f"{LAW_ENV}={raw}",
                basis=BASIS_HAND_SET,
            )
        except (TypeError, ValueError):
            pass  # a malformed override is the fallback, exactly as before

    mib, why = _resolve_transient_mib(
        card_uuid, tag, hw_fingerprint, profile, profile_digest, cache_dir
    )
    if mib is not None:
        return CorridorFloor(
            card_uuid=card_uuid,
            group=tag,
            transient_mib=int(mib),
            reserve_mib=reserve,
            source=measured_floor_source(tag),
            provenance=why,
            basis=BASIS_ALLOC_DELTA,
        )
    return CorridorFloor(
        card_uuid=card_uuid,
        group=tag,
        transient_mib=CORRIDOR_LAW_MIB,
        reserve_mib=reserve,
        source=FLOOR_SOURCE_FALLBACK,
        provenance=f"{why}; falling back to the stated law {CORRIDOR_LAW_MIB} MiB",
        basis=BASIS_STATED_LAW,
    )


def corridor_floors_for_cards(
    card_uuids: Sequence[str],
    *,
    group: Optional[str] = None,
    user_reserve_mib=None,
    **kw,
) -> Dict[str, CorridorFloor]:
    """The floor for several cards in one call, keyed by uuid.

    ``user_reserve_mib`` is a scalar or a ``{uuid: MiB}`` mapping.

    ONE READER, GROUP-WIDE (``raenge-nie-uneins``). Every rank co-located on a
    card gets the same floor by construction: the transient is stored per CARD
    and the reserve is already collapsed per card by
    ``ServerArgs.user_reserve_mib_per_gpu``. There is deliberately no per-rank
    entry point, so two ranks on one card cannot disagree about their floor.
    """
    out: Dict[str, CorridorFloor] = {}
    for uuid in card_uuids:
        if isinstance(user_reserve_mib, dict):
            reserve = int(user_reserve_mib.get(uuid, 0) or 0)
        else:
            reserve = int(user_reserve_mib or 0)
        out[uuid] = corridor_floor_mib(
            uuid, group=group, user_reserve_mib=reserve, **kw
        )
    return out


def corridor_floor_for_current_device(
    group: Optional[str] = None,
    fallback_reserve_mib: int = 0,
) -> CorridorFloor:
    """The derived floor for the card THIS process is on. ONE rank-local door.

    REFUTER FINDINGS 6 AND 9 (2026-09-09). Three rank-local consumers
    (``kv_vmm_backing``'s allocator preempt, ``phase_flip_seam_reserve``,
    ``phase_flip_runtime``) each needed "the floor for my card" and each did
    something different about it: two imported the flat ``DEFAULT_FLOOR_MIB``
    and the third resolved its own uuid inline. That is three readers of one
    law, which is the defect ``vram_dial``'s own docstring names.

    RANK-UNIFORM BY CONSTRUCTION, and this is the part worth stating. The
    floor is a pure function of (card uuid, group, reserve, store, env), and
    every one of those is a CARD- or GROUP-level quantity: two ranks
    co-located on one card resolve the same uuid, read the same
    :data:`GROUP_ENV`, read the same :data:`USER_RESERVE_ENV`, and open the
    same store file. There is no rank term anywhere in the derivation and no
    entry point that takes one. What this does NOT do is DETECT a disagreement
    that arose anyway (a store file half-written under one rank, say) -- that
    needs a group collective and a boot, and it is named in UNPROVEN rather
    than faked here.
    """
    uuid = ""
    try:
        from flliper.srt.registry import nvml as registry_nvml

        uuid = registry_nvml.current_device_uuid() or ""
    except Exception as exc:  # pragma: no cover - NVML availability
        logger.debug("current device uuid unavailable for the floor: %s", exc)
    # THE PUBLISHED PER-CARD MAPPING FIRST, the caller's scalar only where
    # there is none. A rank booted WITHOUT the Weg-2 launcher has no
    # USER_RESERVE_ENV at all, and reading 0 there would silently drop an
    # operator's ``--rank-user-reserve-mib`` -- the one knob the user
    # explicitly kept ("das feature muss aber erhalten bleiben"). The fallback
    # is still rank-uniform: it is one CLI value shared by every rank.
    # ``reserve_for_card`` is the one choice: ``user_reserve_by_card()`` first, the
    # named ``fallback_reserve_mib`` scalar only where the mapping has no entry.
    reserve, reserve_source = reserve_for_card(uuid, fallback_reserve_mib)
    # #236: the card this rank resolved and the reserve it will add, once per
    # (card, value, source) -- the rank-side half of the launcher's
    # PDFLIP-USER-RESERVE PROVENANCE line.
    _note_key = ("card", uuid, int(reserve), reserve_source)
    if _note_key not in _RESERVE_READ_NOTED:
        _RESERVE_READ_NOTED.add(_note_key)
        logger.info(
            "PDFLIP-USER-RESERVE-READ card=%s reserve_mib=%d source=%s%s group=%s pid=%d",
            uuid or "?",
            int(reserve),
            reserve_source,
            (f" stated_mib={user_reserve_masked_by_card().get(uuid)}"
             if reserve_source == "masked-by-booked-rest" else ""),
            group,
            os.getpid(),
        )
    return corridor_floor_mib(uuid, group=group, user_reserve_mib=reserve)


#: #826: OPT-IN ADOPTION OF THE SOLVED ARMING FLOOR.
#:
#: #770 shipped `solve_arming_floor` as an OBSERVER -- its only consumer turns
#: it into advice text and sets no floor -- so the shipped reserve stayed at
#: 512 and `arming_floor_mib()` stayed at 1331 against a band ceiling of 1229.
#: 294 MiB short by construction, on every card, on every boot. The solver's
#: own docstring names the missing half: "the caller decides whether to adopt
#: it or refuse at boot. What must never happen again is the third option the
#: tree shipped: neither adopting nor refusing".
#:
#: This is that caller, and it is OPT-IN because cutting the reserve from 512
#: to 218 changes what the seam may spend WHILE IT RUNS -- a behaviour change
#: to the flip that needs metal, which is exactly why it is a flag and not a
#: new default.
#:
#: SCOPE, held deliberately narrow: this adopts the floor VALUE. It does not
#: revive the withdrawn floor clamp (a cap under live rows, withdrawn in
#: 3b2bbde3ad; `test_residency_cap_flip_levelling_792` is that watchdog) and
#: it takes no kv-slack draws.
SOLVED_FLOOR_ENV = "FLLIPER_ARMING_FLOOR_SOLVED"

_ARMING_FLOOR_PROVENANCE_LOGGED = False


class ArmingFloorUnsatisfiable(RuntimeError):
    """No seam-entry reserve fits under the corridor band ceiling.

    Raised AT BOOT, by name. The alternative the tree shipped was to advise at
    runtime that the flip "is retried when occupancy drops" -- which describes
    a state the corridor law forbids, and which 18f measured directly:
    draining the load did not lift the lock. A configuration that cannot arm
    from inside its own acceptance band must say so before it serves, not
    livelock after.
    """


def _reset_arming_floor_provenance() -> None:
    """Test seam: the provenance line is logged once per process."""
    global _ARMING_FLOOR_PROVENANCE_LOGGED
    _ARMING_FLOOR_PROVENANCE_LOGGED = False


def _arming_margin_mib_for_solver() -> int:
    """The margin the arming gate wants on top of the floor.

    Indirected through a function so the solve can be driven in tests without
    monkeypatching another module's constant, and so this module does not take
    an import-time dependency on the seam reserve module.
    """
    from flliper.srt.managers.phase_flip_seam_reserve import DEFAULT_ARMING_MARGIN_MIB

    return int(DEFAULT_ARMING_MARGIN_MIB)


def seam_entry_reserve_mib_resolved() -> int:
    """The seam-entry allowance the arming floor is built on.

    Default: the shipped 512, byte-identical to the pre-#826 path.
    With ``FLLIPER_ARMING_FLOOR_SOLVED=1``: the largest reserve that fits under
    the band ceiling, as solved by `funding_authority.solve_arming_floor`.

    Raises `ArmingFloorUnsatisfiable` when the flag is set and NOTHING fits.
    That is the whole point of the flag: adopt, or refuse by name.
    """
    global _ARMING_FLOOR_PROVENANCE_LOGGED
    raw = os.environ.get(SOLVED_FLOOR_ENV, "")
    if raw.strip().lower() not in ("1", "true", "yes", "on"):
        return int(DEFAULT_SEAM_ENTRY_RESERVE_MIB)

    from flliper.srt.managers.funding_authority import solve_arming_floor

    floor = corridor_band_floor_mib()
    ceiling = corridor_band_ceiling_mib()
    margin = _arming_margin_mib_for_solver()
    sol = solve_arming_floor(
        floor, ceiling, int(DEFAULT_SEAM_ENTRY_RESERVE_MIB), margin
    )
    # CONTRACT, and it is easy to misread: `satisfiable` means the REQUESTED
    # reserve (512) fits under the ceiling -- on the shipped numbers it never
    # does. The SOLVED answer is `max_seam_entry_reserve_mib`, the largest
    # reserve that does fit (218 here). Refusal is reserved for the case where
    # nothing fits at all. Reading `satisfiable` as "a solution exists" would
    # make this actuator refuse every boot it was built to enable.
    reserve = (
        int(sol.seam_entry_reserve_mib)
        if sol.satisfiable
        else int(sol.max_seam_entry_reserve_mib)
    )
    if reserve <= 0:
        raise ArmingFloorUnsatisfiable(
            f"{LOG_PREFIX} #826 arming floor is unsatisfiable and this boot "
            f"asked for the solved floor ({SOLVED_FLOOR_ENV}=1): band floor "
            f"{floor} MiB, band ceiling {ceiling} MiB, arming margin {margin} "
            f"MiB, requested seam entry reserve "
            f"{int(DEFAULT_SEAM_ENTRY_RESERVE_MIB)} MiB. {sol.detail} "
            f"Refusing at boot rather than arming a gate that cannot be "
            f"reached from inside the corridor band."
        )
    if not _ARMING_FLOOR_PROVENANCE_LOGGED:
        _ARMING_FLOOR_PROVENANCE_LOGGED = True
        logger.warning(
            "%s #826 arming floor %d MiB, solver-derived, corridor ceiling "
            "%d MiB (band floor %d + seam entry reserve %d, arming margin "
            "%d; shipped reserve %d would have put the floor at %d, which is "
            "%d MiB past the ceiling once the margin is added)",
            LOG_PREFIX,
            floor + reserve,
            ceiling,
            floor,
            reserve,
            margin,
            int(DEFAULT_SEAM_ENTRY_RESERVE_MIB),
            floor + int(DEFAULT_SEAM_ENTRY_RESERVE_MIB),
            floor + int(DEFAULT_SEAM_ENTRY_RESERVE_MIB) + margin - ceiling,
        )
    return reserve


seam_entry_reserve_mib = seam_entry_reserve_mib_resolved


def arming_floor_mib(
    seam_entry_reserve_mib: Optional[int] = None,
    law_mib: int = CORRIDOR_LAW_MIB,
) -> int:
    """The gate's watermark, derived from the law it protects.

    ``seam_entry_reserve_mib`` is the MEASURED draw a seam makes while it
    runs, where a measurement exists; the default is the shipped allowance.
    Deriving it means an operator cannot raise one number and leave the
    other behind, which is the failure this function exists to make
    impossible.

    BUILT ON THE BAND FLOOR, not the centre. The gate exists to keep the
    worst instant out of breach, and breach is now defined at
    :func:`corridor_band_floor_mib`. Arming from the centre instead would
    reserve the band's whole tolerance on top of the seam's draw on every
    card, for every boot -- roughly 205 MiB per rank here -- which is pool
    given up to protect a threshold that is not the verdict. The centre is
    what the self-correcting margin aims at; the floor is what the gate
    defends.
    """
    # #826: resolved at CALL time, not bound as a default at def time -- a
    # default argument would freeze the shipped 512 into the signature and
    # make the flag inert, which is the shape of the bug this closes.
    if seam_entry_reserve_mib is None:
        seam_entry_reserve_mib = seam_entry_reserve_mib_resolved()
    return corridor_band_floor_mib() + max(0, int(seam_entry_reserve_mib))


#: The one machine-readable line a boot emits so its acceptance verdict can
#: credit what the flip has committed. THE FORMAT LIVES HERE AND NOWHERE ELSE:
#: ``corridor_verdict_774.sh`` imports :func:`parse_corridor_posts` rather than
#: re-deriving the field names in awk, which is how the verdict and the runtime
#: came to disagree about the band ceiling by 1 MiB in the first place.
CORRIDOR_POST_PREFIX = "CORRIDOR-POST"


def format_corridor_post(
    rank: int, gpu_id: int, arming_mib: int, uuid: Optional[str] = None
) -> str:
    """One rank's committed arming reserve, as the verdict reads it.

    ``gpu_id`` is the PHYSICAL device index, not the process-local one: with
    per-rank ``CUDA_VISIBLE_DEVICES`` isolation every rank sees itself as
    ``cuda:0``, while the verdict enumerates NVML in physical order. The UUID
    rides along because NVML's enumeration order can shift between boots and
    driver states, so a reader that doubts the index has the identity.
    """
    line = (
        f"{CORRIDOR_POST_PREFIX} rank={int(rank)} gpu={int(gpu_id)} "
        f"arming_floor_mib={int(arming_mib)} "
        f"band_floor_mib={corridor_band_floor_mib()} "
        f"committed_mib={committed_arming_mib(arming_mib)}"
    )
    if uuid:
        line += f" uuid={uuid}"
    return line


def parse_corridor_posts(text: str) -> dict:
    """``{physical_gpu_id: committed MiB}`` from a boot log.

    TWO RULES, BOTH OF THEM LOAD-BEARING.

    *Co-resident ranks ADD.* Two ranks sharing a physical GPU each hold their
    own arming floor, so the card's committed column is the sum.

    *A rank is counted ONCE, at its latest reading.* A phase-flip boot
    resolves its arming floor in the PP phase and again when the TP stack is
    built, so the same rank emits more than once in a single log. Summing the
    LINES would charge the card twice and hand the verdict a pass it did not
    earn, which is the one failure mode a credit must not have. So the last
    line per rank wins -- including its card, so a rank that moved is not
    counted on both.

    A malformed line is skipped rather than raising: this runs inside an
    acceptance verdict, where dying on a truncated log would turn a gradeable
    boot into no verdict at all.
    """
    latest = {}
    for raw in str(text).splitlines():
        line = raw.strip()
        if CORRIDOR_POST_PREFIX not in line:
            continue
        fields = {}
        for token in line[line.index(CORRIDOR_POST_PREFIX) :].split():
            if "=" in token:
                key, _, value = token.partition("=")
                fields[key] = value
        try:
            rank = int(fields["rank"])
            gpu = int(fields["gpu"])
            committed = int(fields["committed_mib"])
        except (KeyError, TypeError, ValueError):
            continue
        latest[rank] = (gpu, max(0, committed))
    posts: dict = {}
    for gpu, committed in latest.values():
        posts[gpu] = posts.get(gpu, 0) + committed
    return posts


#: The layout that BINDS the KV pool, and therefore the only one whose free
#: column grades a boot's sizing. Under Route A the PP prefill layout is where
#: the id space is min-reduced across ranks and where the binding rank's
#: bracket runs down to its arming floor; the TP decode layout inherits that id
#: space through the canonical KV page and sizes nothing of its own.
SIZING_PHASE = "pp"

#: Below this a card is one allocation from death, in any phase.
NEAR_OOM_MIB = 400


def phase_corridor_verdict(net_by_phase: dict, sizing_phase: str = SIZING_PHASE):
    """``(verdict, findings)`` from per-phase, per-card NET free minima.

    THE DEFECT (#784). The acceptance verdict had no phase detection: it
    sampled one window and called that the verdict. On 2026-08-20 the window
    fell entirely inside the TP phase, where the arena tail has been released
    and the free column reads gibibytes higher than the phase that actually
    sizes the pool. Rank 2 was graded at 5475 MiB while at rest in PP it sat at
    3232 MiB against a 3226 MiB arming floor -- six MiB of slack. The boot was
    graded on a layout that sizes nothing, 2243 MiB away from the number that
    binds.

    So the sizing phase decides, and a window that never observed it is not a
    pass but an absent measurement. Other phases are still reported: a card
    resting high in TP is the layout doing what it should, not a sizing defect,
    and demoting that to a finding is what stops the instrument from failing a
    boot for releasing its own arena tail.

    The band is asymmetric by the operator's ruling. Over-band in the sizing
    phase is a failed acceptance -- VRAM that buys no tokens. Under-band is a
    finding and a planner task, never a stopper. A near-OOM reading fails in
    ANY phase, because that one is about survival rather than efficiency.
    """
    findings: List[str] = []
    floor, ceiling = corridor_band_floor_mib(), corridor_band_ceiling_mib()
    fail = False
    warn = False

    for phase in sorted(net_by_phase):
        for gpu in sorted(net_by_phase[phase]):
            net = int(net_by_phase[phase][gpu])
            if net < NEAR_OOM_MIB:
                fail = True
                findings.append(
                    f"GPU{gpu} phase={phase}: net {net} MiB is below the "
                    f"{NEAR_OOM_MIB} MiB near-OOM floor -- one allocation from "
                    f"death, and that is a stopper in any phase"
                )
            elif net > ceiling:
                if phase == sizing_phase:
                    fail = True
                    findings.append(
                        f"GPU{gpu} phase={phase}: net {net} MiB > {ceiling} MiB "
                        f"-- {net - ceiling} MiB of KV pool given away in the "
                        f"phase that sizes it"
                    )
                else:
                    findings.append(
                        f"note: GPU{gpu} phase={phase}: net {net} MiB > "
                        f"{ceiling} MiB, reported only -- this layout inherits "
                        f"its id space and sizes nothing, and it has released "
                        f"its arena tail, so a high reading here is the layout "
                        f"working, not a sizing defect"
                    )
            elif net < floor and phase == sizing_phase:
                warn = True
                findings.append(
                    f"GPU{gpu} phase={phase}: net {net} MiB < {floor} MiB -- "
                    f"file as a planner task, NOT a stopper"
                )

    if sizing_phase not in net_by_phase:
        findings.append(
            f"the sizing phase {sizing_phase!r} was never sampled (saw: "
            f"{sorted(net_by_phase) or 'nothing'}). The pool is bound in "
            f"{sizing_phase!r}, so this window grades a layout that sizes "
            f"nothing. An absent measurement is not a pass."
        )
        return "NO-VERDICT", findings

    if fail:
        return "FAIL", findings
    return ("WARN" if warn else "PASS"), findings


def check_threshold_pair(
    arming_mib: int,
    law_mib: int = CORRIDOR_LAW_MIB,
    band_floor_mib: Optional[int] = None,
    band_ceiling_mib: Optional[int] = None,
) -> None:
    """Refuse a pair that cannot mean what it says.

    An arming floor below the law lets the gate return "no reclaim needed"
    for an allocation that ends under the corridor -- the guard would be
    laundering a breach as a passed check, which is the one thing its own
    refusal message says it must never do.

    THE CEILING SIDE (#784). The floor check alone let a second unsatisfiable
    pair through in silence: a band whose floor sits above its own ceiling has
    no feasible region at all, so every boot graded under it fails an
    acceptance no configuration could pass. The band is inverted exactly when
    the fraction exceeds 1.0, which no shipped value does -- but the pair was
    accepted without a word when it happened, and "accepted in silence" is the
    same failure mode the floor side exists to prevent.

    The arming floor itself is deliberately NOT compared against the ceiling.
    It exceeds it by construction on every rig (1523 > 1229 with the shipped
    constants) and that is not an error: the excess is charged working
    capital, credited by :func:`net_free_mib`, not idle VRAM. Refusing it here
    would refuse every armed flip boot.

    ``band_floor_mib`` / ``band_ceiling_mib`` override the derived band for
    tests that need to construct an inverted one; production passes neither.
    """
    floor = corridor_band_floor_mib() if band_floor_mib is None else int(band_floor_mib)
    ceiling = (
        corridor_band_ceiling_mib()
        if band_ceiling_mib is None
        else int(band_ceiling_mib)
    )
    if floor > ceiling:
        raise ValueError(
            f"corridor band is INVERTED: floor {floor} MiB is above ceiling "
            f"{ceiling} MiB, so the band has no feasible region and no boot "
            f"can ever be graded against it. The band is the law "
            f"{law_mib} MiB plus and minus {CORRIDOR_BAND_FRACTION:.0%}; a "
            f"fraction above 100 % inverts it."
        )
    if int(arming_mib) < floor:
        raise ValueError(
            f"corridor arming floor {arming_mib} MiB is BELOW the corridor "
            f"law {law_mib} MiB. The gate would clear allocations the law "
            f"forbids and no refusal would ever fire. The arming floor is "
            f"the law PLUS the seam's expected draw, never less than it."
        )


#: How far ABOVE the floor a reclaim frees, so the next few allocations do not
#: each pay a spill. 256 MiB is one seam's worth of slack on this rig's
#: measured seam trough (1196 MiB free at the tightest instant, floor 1024).
DEFAULT_DELTA_MIB = 256

#: Relief tiers (item 16). Lower is tried first, unconditionally: the tier
#: outranks the cost, so a cheap host spill can never overtake an expensive
#: rebalance. Cost still orders providers within one tier.
RELIEF_LOCAL = -1  # give back memory NO payload owns (torch's allocator cache)
RELIEF_REBALANCE = 0  # level the free column: token vector, evacuate the heavy card
RELIEF_PARK = 1  # park a cold payload in another card's surplus VRAM
RELIEF_HOST = 2  # system RAM -- only once every card is at the floor

_TIER_NAMES = {
    RELIEF_LOCAL: "local",
    RELIEF_REBALANCE: "rebalance",
    RELIEF_PARK: "park",
    RELIEF_HOST: "host",
}


def free_spread_mib(free_column) -> int:
    """max - min over the per-card free column, in MiB. The levelness metric.

    This is the number that goes in the corridor CSV. A run whose spread
    stays small was evenly filled; a run whose spread is thousands of MiB was
    not, whatever its minimum says. The corridor law's minimum and item 16's
    spread are independent axes and a CSV needs both.
    """
    col = [int(f) for f in free_column]
    if not col:
        return 0
    return int((max(col) - min(col)) // _MIB)


def water_fill_targets(free_column):
    """The equal-free-headroom targets for a fleet, in bytes.

    The objective is EQUAL FREE, so every card's target is the mean of the
    column regardless of its total size. Returned as a vector (rather than a
    scalar) because the caller subtracts each card's current free to get the
    signed transfer it should aim for: negative means "this card should give
    bytes up", positive means "this card can take them".
    """
    col = [int(f) for f in free_column]
    if not col:
        return []
    mean = sum(col) // len(col)
    return [mean] * len(col)


def water_fill_transfers(free_column):
    """The signed PAYLOAD move each card needs to reach equal free, in bytes.

    Positive means **this card should SHED that many bytes of payload**;
    negative means it can ABSORB them. The vector sums to zero, because a
    transfer is conserved.

    WHY THIS EXISTS ALONGSIDE :func:`water_fill_targets` RATHER THAN INSTEAD
    OF IT. The targets function is the objective and this is one subtraction
    away from it -- computed FROM it here, so there is one derivation and not
    two. What it adds is a sign convention a reader can act on without a
    decoder ring: ``water_fill_targets`` states its sign in terms of FREE
    bytes, and "this card should give bytes up" reads backwards to anyone
    thinking about where the payload goes, because the card that should give
    FREE bytes up is the card that should TAKE payload.

    IT IS NOW BOTH AN INSTRUMENT AND AN OBJECTIVE, AND THE SPLIT MATTERS.
    Item 16's first relief stage is "redistribute onto the card with the most
    headroom", and that has two halves with different fates:

    * the SHED half -- get the bytes off the card that binds -- is actuated,
      by :meth:`CorridorGuard.lend_to_level` driven from
      ``corridor_rebalance.RebalanceLender``. This function supplies its
      bound: the lender frees up to the transfer named here and no further.
    * the ABSORB half -- put those same bytes on the peer card -- is still
      missing, and structurally so for KV: the DCP owner rule makes a token's
      row a pure function of the token vector (``layers/dcp/owner.py:348``),
      the vector is boot-constant (``phase_flip_runtime.py:866``), and the
      one actuator that may change it needs a fully idle instance
      (``kv_reshard.py:781``). In the PP phase KV is layer-bound anyway.

    So a positive transfer here is a move that WILL be attempted from the
    tight card, and a negative one is capacity that stays unusable to its
    peers. Before HANDOFF_679 ``water_fill_targets`` had exactly one caller
    in the tree and it was a test: the objective was computed by nothing.

    A spread figure alone does not tell a successor whether the actuator is
    worth building; a per-card "shed 900 MiB / absorb 900 MiB" does.
    """
    col = [int(f) for f in free_column]
    if not col:
        return []
    return [t - f for t, f in zip(water_fill_targets(col), col)]


def describe_water_fill(free_column) -> str:
    """One human-readable clause naming the levelling move, or ''.

    Empty when the fleet is unknown or already level, so a log line does not
    grow a clause that says nothing.
    """
    transfers = water_fill_transfers(free_column)
    if not transfers or all(t == 0 for t in transfers):
        return ""
    shed = max(range(len(transfers)), key=lambda i: transfers[i])
    absorb = min(range(len(transfers)), key=lambda i: transfers[i])
    if transfers[shed] <= 0:
        return ""
    return (
        f"item 16 water-fill wants card {shed} to SHED "
        f"{transfers[shed] / _MIB:.0f} MiB onto card {absorb} "
        f"(which can absorb {-transfers[absorb] / _MIB:.0f} MiB). The SHED "
        "half is actuated by the rebalance lender, bounded by this figure; "
        "the ABSORB half has no actuator (the DCP owner rule fixes KV "
        "ownership to a boot-constant token vector), so any residue is "
        "levelling NOT performed"
    )


def fleet_is_level(free_column, floor_mib: int, delta_mib: int) -> bool:
    """True when EVERY card sits at the floor, i.e. host RAM is permitted.

    "At the floor" is a band and not a point -- a card is never exactly at
    1024 MiB, so a point predicate would be unreachable and the host tier
    would be dead code. The band is the same ``floor + delta`` upper
    watermark the gate frees to, which keeps one number in charge of both.

    An EMPTY column is not level. No evidence is not evidence.
    """
    col = [int(f) for f in free_column]
    if not col:
        return False
    ceiling = (int(floor_mib) + int(delta_mib)) * _MIB
    return all(f <= ceiling for f in col)


class CorridorBreachRefused(RuntimeError):
    """The allocation cannot be made without breaking the corridor law.

    Raised only when the caller asked for ``raise_on_refusal``; the default
    is a returned verdict, because the seam's caller wants to abandon the
    flip cleanly rather than unwind an exception from inside a cutover.
    """


@dataclass(order=True)
class _Provider:
    # Ordered by TIER first (item 16), then by cost (item 15). The two-key
    # sort is the whole enforcement: a cheap host spill sorts after every
    # rebalance no matter how expensive the rebalance is.
    tier: int
    cost: int
    name: str = field(compare=False)
    free_up_to: Callable[[int], int] = field(compare=False)


@dataclass
class ProviderDelivery:
    """What one provider PROMISED and what the driver actually gave back (#852).

    THE CLASS #852 LEFT STANDING. :meth:`CorridorGuard.register` states the law
    in its own docstring -- ``free_up_to`` "must ... return the bytes it
    actually gave back to the DRIVER -- not to torch's cache" -- and nothing
    measured it, so no provider could be asked whether it obeyed. #852 answered
    that question for ONE funder by hand (the allocator cache, priced at what
    an ``empty_cache()`` draw can return, then narrowed again by R3's
    graph-pool term). This makes it answerable for EVERY provider, including
    the ones nobody has suspected yet, which is the only version of the fix
    that survives the next funder.

    ``claimed_bytes`` is the sum of the providers' own return values.
    ``delivered_bytes`` is the sum of the DRIVER's free-column delta across
    each call -- the corridor law's own unit, and already sampled on every
    ladder iteration before this existed.

    ``observations`` is the number of times the provider was actually CALLED.
    Zero means never measured, which is why :attr:`delivery_ratio` is ``None``
    there rather than ``0.0``: "no probe has recorded this" and "recorded and
    delivered nothing" are the two readings this whole ticket family exists to
    keep apart, and a default of ``0`` is exactly how the first silently reads
    as the second.
    """

    name: str
    claimed_bytes: int = 0
    delivered_bytes: int = 0
    observations: int = 0
    #: Passes in which the provider claimed bytes and the driver saw none.
    #: One is an anecdote; W24's phantom ran 43 times in 21.6 minutes.
    phantom_passes: int = 0
    #: Law 2's memory, and it is the LAST observation, not the history. A
    #: provider that comes good must be able to earn its credit back -- a
    #: derate that could only ratchet down would strand a funder that started
    #: paying, which is the seam-stickiness defect with the sign flipped.
    last_claimed_bytes: int = 0
    last_delivered_bytes: int = 0

    @property
    def delivery_ratio(self) -> Optional[float]:
        """Delivered over claimed across the record, or ``None`` if unobserved.

        A provider that claimed nothing but was called has nothing to be wrong
        about, so it reads 1.0 rather than dividing by zero.
        """
        if self.observations == 0:
            return None
        if self.claimed_bytes <= 0:
            return 1.0
        return self.delivered_bytes / self.claimed_bytes

    @property
    def is_phantom(self) -> bool:
        """The standing verdict, from the LAST observation only.

        Deliberately not a rate over the history: the question a caller asks is
        "should I believe this provider's next claim", and the only evidence
        about the next claim is the most recent one.
        """
        return self.observations > 0 and (
            self.last_claimed_bytes > 0 and self.last_delivered_bytes <= 0
        )

    def describe(self) -> str:
        if self.observations == 0:
            return f"{self.name}: never measured"
        return (
            f"{self.name}: claimed {self.claimed_bytes / _MIB:.0f} MiB, "
            f"delivered {self.delivered_bytes / _MIB:.0f} MiB over "
            f"{self.observations} pass(es)"
            + (
                f", {self.phantom_passes} of them phantom"
                if self.phantom_passes
                else ""
            )
        )


@dataclass
class GuardResult:
    """What the gate did, in bytes, so a caller can log or account for it."""

    ok: bool
    free_before: int
    free_after: int
    want: int
    reclaimed: int
    used_providers: Tuple[str, ...]
    detail: str = ""
    #: The verdict HELD but the residual sits under the corridor law. Carried
    #: so the caller can WARN (user decision 2026-08-16: the law is advisory
    #: at seam entry, and a dip has to be sayable to be warned about).
    law_breached: bool = False

    @property
    def reclaimed_mib(self) -> float:
        return self.reclaimed / _MIB


class CorridorGuard:
    """Synchronous spill-before-alloc gate for ONE device.

    One guard per rank per device. The rank's own device is named
    explicitly rather than read from ``current_device()``: under
    ``--rank-gpu-id`` each worker sees exactly one card, but an absolute
    memory figure should still say which card it is about.
    """

    def __init__(
        self,
        device_index: int,
        *,
        floor_mib: int = DEFAULT_FLOOR_MIB,
        delta_mib: int = DEFAULT_DELTA_MIB,
        probe: Optional[Callable[[], int]] = None,
        fleet_probe: Optional[Callable[[], Sequence[int]]] = None,
        law_floor_mib: Optional[int] = None,
    ) -> None:
        self.device_index = int(device_index)
        self.floor_mib = int(floor_mib)
        # THE ARMING WATERMARK AND THE LAW ARE NOT THE SAME NUMBER, and
        # conflating them wedged this instance on 2026-08-10.
        #
        # ``floor_mib`` is a POLICY target: where the gate starts working and
        # how far it frees. ``law_floor_mib`` is the user's corridor law, the
        # only thing a REFUSAL may be justified by. When they are equal (the
        # default) nothing changes. When the policy floor is raised -- for a
        # proof run, or by a future per-card policy -- a shared threshold
        # makes the gate refuse allocations that the law permits perfectly
        # well, and on the pp->tp leg that is not a conservative choice: it
        # is a DEADLOCK. Strict purity forbids decode in PP, so a permanently
        # refused pp->tp flip means decode never runs again, and nothing in
        # the PP phase can free the memory that would end the refusal.
        # Measured: 411 abandons, 0 requests completed in 6 minutes, /health
        # 503 while every rank was alive and logging normally.
        self.law_floor_mib = int(
            law_floor_mib if law_floor_mib is not None else floor_mib
        )
        self.law_floor_bytes = self.law_floor_mib * _MIB
        self.delta_mib = int(delta_mib)
        self.floor_bytes = int(floor_mib) * _MIB
        self.delta_bytes = int(delta_mib) * _MIB
        self._probe = probe
        # #851 F4: THE FUNDER THIS LADDER MAY NOT SPEND, so that a refusal can
        # still NAME it. Optional and None by default -- an unset view leaves
        # every existing refusal line byte-identical.
        #
        # A post can be declared to the FundingAuthority and be unreachable
        # from here on purpose: the KV rung pays BEFORE this gate, and its cap
        # is a GROUP decision while this ladder is rank-local
        # (phase_flip_runtime.py:8290-8294). Registering it as a provider would
        # spend a group quantity from a rank-local ladder, twice. So the fix is
        # not to reach it -- it is to stop reporting "[nothing]" when the true
        # answer is "there was money, one bookkeeper earlier, and this gate
        # could not draw it".
        self._offledger_funder: Optional[Callable[[int], Sequence]] = None
        # Item 16's fleet predicate. Deliberately a per-card NVML read rather
        # than a collective: this gate runs inside the flip's no-return
        # region, and a collective there is a deadlock waiting for the one
        # rank that took a different branch. NVML sees every card regardless
        # of CUDA_VISIBLE_DEVICES, so no rank has to be asked.
        self._fleet_probe = fleet_probe
        self._providers: List[_Provider] = []
        #: #852 CLASS: promised-vs-delivered, per provider, for the life of the
        #: guard. Keyed by name and created at REGISTRATION rather than at
        #: first spend, so a provider that has never been called reads "never
        #: measured" instead of being absent -- an absent key and a zero
        #: delivery are the two readings this family keeps confusing.
        self._delivery: Dict[str, ProviderDelivery] = {}
        self.arm_count = 0
        self.refuse_count = 0
        self.host_blocked_count = 0
        #: Times the host tier was admitted onto an UNLEVEL fleet because
        #: refusing would have deadlocked the caller. Every one of these is a
        #: levelling failure that item 16 wanted avoided, so the counter is
        #: the honest measure of how much the missing rebalance tier costs.
        self.host_forced_count = 0
        self.reclaimed_total = 0
        #: Item 16's REBALANCE lender (see :meth:`lend_to_level`). Counted
        #: apart from ``arm_count`` on purpose: an arm is an allocation that
        #: would have breached, a lend is relief taken BEFORE any allocation
        #: asked, and averaging the two would hide which one moved the trough.
        self.lend_count = 0
        self.lent_total = 0

    # -- registration ----------------------------------------------------

    def register(
        self,
        name: str,
        cost: int,
        free_up_to: Callable[[int], int],
        tier: int = RELIEF_REBALANCE,
    ) -> None:
        """Add a payload class the gate may spend.

        ``free_up_to(nbytes)`` must free AT MOST ``nbytes``, synchronously,
        and return the bytes it actually gave back to the DRIVER -- not to
        torch's cache. The corridor law is stated in NVML's free column, and
        a provider that only returns memory to the caching allocator has
        freed nothing the law can see.

        THAT LAW IS NOW MEASURED RATHER THAN TRUSTED (#852). It stood here as
        prose for the whole life of this module, and a provider that broke it
        was credited in full: the ladder summed return values while the driver
        delta it sampled on every iteration was discarded. Each call is now
        recorded in :meth:`delivery_report` -- what the provider claimed
        against what the free column actually did -- so a provider that breaks
        the contract is NAMED on the first pass instead of after a GPU window,
        and no judgement rests on its own account of itself. Breaking the law
        is still not an error: the ledger reports, it does not refuse. A
        provider that cannot measure its own delta may simply return its best
        figure and let the guard find the truth.

        ``cost`` orders the spend: lower is cheaper to give up and to get
        back. Ties are resolved by registration order.

        ``tier`` (item 16) outranks ``cost`` entirely -- see the module
        docstring. It defaults to ``RELIEF_REBALANCE`` so that a provider
        written before tiers existed keeps working: defaulting to
        ``RELIEF_HOST`` would silently switch off relief that already works,
        while this direction forces the gated class to be declared on
        purpose, which is the way an omission should fall.
        """
        if not callable(free_up_to):
            raise TypeError(f"{LOG_PREFIX} provider {name!r} is not callable")
        if tier not in _TIER_NAMES:
            raise ValueError(
                f"{LOG_PREFIX} provider {name!r}: unknown relief tier {tier!r}, "
                f"expected one of {sorted(_TIER_NAMES)}"
            )
        self._providers.append(_Provider(int(tier), int(cost), str(name), free_up_to))
        self._providers.sort(key=lambda p: (p.tier, p.cost))
        # #852: open the ledger now, so "registered but never called" is a
        # readable state rather than a missing key. setdefault, because a
        # re-registration under the same name must not erase what that name
        # has already been measured doing.
        self._delivery.setdefault(str(name), ProviderDelivery(str(name)))
        logger.info(
            "%s registered provider %r in tier %s at cost %d (device %d); "
            "spend order is now: %s",
            LOG_PREFIX,
            name,
            _TIER_NAMES[tier],
            cost,
            self.device_index,
            ", ".join(f"{p.name}[{_TIER_NAMES[p.tier]}]" for p in self._providers),
        )

    @property
    def providers(self) -> Tuple[str, ...]:
        return tuple(p.name for p in self._providers)

    # -- the gate --------------------------------------------------------

    # #1054e: `device_index` IS PROCESS-LOCAL, and the log lines that print it
    # read as if it were physical. Verified on the live boot 25 (2026-08-31):
    # every rank process carries CUDA_VISIBLE_DEVICES set to a single GPU UUID
    # (PP0 -> GPU-31d7ef41 = the 5090, PP1 -> GPU-5c648f96, PP2 -> GPU-62dbbae1,
    # cross-checked against nvidia-smi's per-PID compute-apps list), so torch
    # sees exactly one device per process and index 0 is that process's own
    # card. "cleared on device 0" therefore appears on PP0 AND PP2 and is
    # correct in both -- they are different physical cards.
    #
    # RULED OUT BY THIS, explicitly, because it was a live hypothesis: the
    # guard is NOT reading a foreign card for a rank. The #392/#406/#589
    # device-order family does not apply here; isolation is at the process
    # level, which is the canonical form, not an in-process index map. The
    # 774 MiB -> 21.69 MiB gap in boot 24 is therefore a REAL transient spike
    # between the two readings, not a wrong-card artifact.
    def free_bytes(self) -> int:
        if self._probe is not None:
            return int(self._probe())
        import torch

        return int(torch.cuda.mem_get_info(self.device_index)[0])

    def fleet_free(self) -> List[int]:
        """The per-card free column, in bytes. Empty when unknown."""
        if self._fleet_probe is None:
            return []
        try:
            return [int(f) for f in self._fleet_probe()]
        except Exception as e:
            # An unreadable fleet is an UNPROVEN fleet, and item 16's host
            # permission must be proven. Degrading to "empty" therefore
            # closes the host tier rather than opening it.
            logger.warning("%s fleet probe failed: %s", LOG_PREFIX, e)
            return []

    def _host_tier_permitted(self, column: Sequence[int]) -> bool:
        return fleet_is_level(column, self.floor_mib, self.delta_mib)

    def declare_offledger_funder(
        self, view: Optional[Callable[[int], Sequence]]
    ) -> None:
        """Name a funder this ladder may NOT spend, so refusals can cite it.

        ``view(want_bytes)`` returns an iterable of ``(name, credit_bytes,
        reason)``. It is consulted ONLY on a refusal and ONLY for text: it
        never frees a byte, never changes the verdict, and is wrapped so a
        broken view cannot turn a refusal into a crash. A diagnostic that can
        alter the thing it reports is worse than none -- the same rule
        ``explain_kv_target`` states for the group verdict.

        Unset by default, which keeps every existing refusal line unchanged.
        """
        self._offledger_funder = view

    def _offledger_detail(self, want: int) -> str:
        """One clause naming declared funders this gate could not draw on."""
        view = getattr(self, "_offledger_funder", None)
        if view is None:
            return ""
        try:
            posts = list(view(int(want)))
        except Exception as exc:  # noqa: BLE001 - a diagnostic may never raise
            logger.debug("%s off-ledger funder view failed: %s", LOG_PREFIX, exc)
            return ""
        held, empty = [], []
        for entry in posts:
            try:
                name, credit, why = entry
            except Exception:  # noqa: BLE001 - tolerate a malformed row
                continue
            if int(credit) > 0:
                held.append(f"{name} holds {int(credit) / _MIB:.0f} MiB")
            elif why:
                empty.append(f"{name}: {why}")
        if not held and not empty:
            return ""
        parts = []
        if held:
            parts.append(
                "this gate may not draw on "
                + ", ".join(held)
                + " -- that funder pays before the gate and its cap is a group "
                "decision, so it can never appear in the list above (#813)"
            )
        if empty:
            parts.append("declared but empty: " + "; ".join(empty))
        return "; ".join(parts)

    def ensure_headroom(
        self,
        want_bytes: int,
        *,
        reason: str = "",
        raise_on_refusal: bool = False,
        refusal_is_fatal: bool = False,
        must_reclaim: bool = False,
    ) -> GuardResult:
        """Make ``want_bytes`` allocatable without breaching the floor.

        Returns a verdict. ``ok=False`` means DO NOT ALLOCATE -- the caller
        must take its own refusal path (at the seam: abandon the flip).

        ``refusal_is_fatal`` tells the gate that the caller has NO survivable
        refusal path, and it opens the host tier even on an unlevel fleet.
        Item 16 is a preference, not a suicide pact: withholding host RAM
        while a peer has headroom is right when the caller can wait, and
        catastrophic when it cannot. On the pp->tp leg it cannot -- strict
        purity forbids decode in PP, so a refused pp->tp starves decode and
        nothing in PP can free the memory that would end the refusal.
        Item 15c already authorises this in the user's terms: the price of the
        host tier is tempo, never a corridor breach. Refusing forever does not
        protect the corridor, which is fine in that state; it kills serving.

        It opens the tier; it does NOT reorder the ladder. Rebalance and park
        are still spent first, so the escape is only reached when nothing
        cheaper exists.
        """
        want = max(0, int(want_bytes))
        free_before = self.free_bytes()
        # The corridor law's own bound. Note it is checked against the
        # allocation that is ABOUT to happen, which is the entire difference
        # between this and a threshold observer: after the fact, a breach has
        # already been recorded by a 100 ms sampler and cannot be undone.
        # #689: THIS BRANCH IS WHERE THE FALSE SUCCESS RETURNED. Telling a
        # caller "no reclaim needed" is correct for an allocator about to
        # allocate `want`, and misleading for one that already accounted the
        # free column and needs this ladder to RELEASE more -- the 12:29 seam
        # asks returned here three times, ok=True, having freed nothing, while
        # spendable sat at 609 against a need of 788. Under must_reclaim the
        # ladder actually runs and the verdict is the measured delta.
        if free_before - want >= self.floor_bytes and not must_reclaim:
            return GuardResult(
                True, free_before, free_before, want, 0, (), "no reclaim needed"
            )

        self.arm_count += 1
        # Free to the UPPER watermark, not merely to the floor, so the next
        # few allocations do not each pay a spill.
        target = self.floor_bytes + self.delta_bytes + want
        if must_reclaim:
            # #689 THE TARGET IS RELATIVE UNDER must_reclaim. The ladder spends
            # only against a DEFICIT (`deficit = target - free_now`), so with
            # 1428 MiB free and a 178 MiB ask there is no deficit and it
            # correctly spends nothing -- which is why the 12:29 asks freed
            # zero. A caller that already accounted the free column is asking
            # for `want` MORE bytes, so the target has to be measured from
            # where the column stands now, not from the floor.
            target = max(target, free_before + want)
        # Item 16: read the fleet ONCE, before spending. Re-reading it after
        # each provider would let a rebalance that just filled a peer card
        # close the host gate mid-ladder on the strength of its own effect,
        # which is a feedback loop, not a policy.
        column = self.fleet_free()
        fleet_level = self._host_tier_permitted(column)
        host_ok = fleet_level or bool(refusal_is_fatal)
        host_forced = bool(refusal_is_fatal) and not fleet_level
        (
            reclaimed,
            claimed,
            free_now,
            used,
            used_host,
            host_blocked,
        ) = self._spend_ladder(
            target=target,
            free_now=free_before,
            column=column,
            host_ok=host_ok,
            host_forced=host_forced,
            max_tier=RELIEF_HOST,
            reason=reason,
        )

        # #852: ``reclaimed`` IS THE DRIVER'S DELTA NOW, not the sum of what
        # the providers said. The lender was given this treatment when it was
        # written -- "this chain has three times credited bytes that went to an
        # allocator free-list instead of to the driver" -- and the gate was
        # exempted on the argument that "its verdict is re-probed anyway".
        # That argument holds for ``ok`` and fails for ``must_reclaim``, which
        # is a SECOND verdict, decided on this number, and whose own refusal
        # text says it judges "the DELTA and nothing else". W24 is the shape:
        # a funder promising ~320 MiB against a free column that never moved.
        self.reclaimed_total += reclaimed
        # USER DECISION 2026-08-16: THE LAW IS ADVISORY HERE, OOM IS NOT.
        #
        # This line used to read `ok = (free_now - want) >= law_floor_bytes`,
        # and that single comparison produced the 06:47:48 wedge: PP1's want
        # of 2163 MiB FIT inside 2456 MiB free, but the 293 MiB residual sat
        # under the law, so the seam was refused 76 times in a row while
        # 727004 tokens waited on an idle GPU. It protected a few hundred MiB
        # of headroom by stopping the machine.
        #
        # The ~1024 line exists because the planner was not filling VRAM well
        # enough. It is a FILL-QUALITY target and it remains the planner's
        # job; it was never a safety device, and it may not block, delay or
        # refuse anything on its own. What it still does is SPEAK: a dip is
        # carried out on the verdict and warned about by the caller.
        #
        # WHAT STAYS HARD is the only thing that was ever unsurvivable -- an
        # allocation larger than free. That is not a corridor dip, it is an
        # OOM, and softening it would trade a warning for a dead worker.
        ok = free_now >= want
        law_breached = ok and (free_now - want) < self.law_floor_bytes
        if not ok:
            self.refuse_count += 1
        # COUNTED WHEN IT MATTERED, and "mattered" is no longer the same as
        # "refused". Since the law stopped gating (2026-08-16), a withheld
        # host tier usually ends in a verdict that HOLDS but dips under the
        # law -- item 16's decision is exactly as consequential as before, so
        # a counter keyed on refusal alone would silently stop recording it.
        if host_blocked and (not ok or law_breached):
            self.host_blocked_count += 1
        detail = (
            f"want {want / _MIB:.0f} MiB, free {free_before / _MIB:.0f} -> "
            f"{free_now / _MIB:.0f} MiB, reclaimed {reclaimed / _MIB:.0f} MiB "
            f"from [{', '.join(used) or 'nothing'}], arming floor "
            f"{self.floor_bytes / _MIB:.0f} MiB, corridor law "
            f"{self.law_floor_mib} MiB" + (f" ({reason})" if reason else "")
        )
        # #851 F4: SAY WHAT THIS GATE COULD NOT DRAW, instead of leaving
        # "[nothing]" to mean two different things.
        #
        # W22 printed "reclaimed 0 MiB from [nothing]" 39 times while the
        # kv-slack post held 2776 MiB. "[nothing]" is the LADDER's provider
        # list, and it is honest about the ladder -- but read as a sentence it
        # says "this rig had no memory to give", which sends the next reader to
        # capacity planning. The true statement is "this rig had a funder this
        # gate may not spend". One window was partly spent on that difference.
        #
        # Appended, never substituted: the ladder's own list stays exactly as
        # it was, because it is the truthful record of what the gate actually
        # spent. Only failures carry the suffix -- a successful ask does not
        # need to explain money it did not need.
        # #853(ii): THE CLAUSE IS APPLIED BELOW, AFTER EVERY MUTATION OF
        # `detail`, NOT HERE. Applying it at this point covered only the
        # refusals that were already decided: the `must_reclaim` branch below
        # flips `ok` AFTER this line and REBUILDS `detail` from scratch, so a
        # clause added here was discarded, and the CANNOT-FULLY-HOLD report is
        # emitted with `ok` still True and never reached this branch at all.
        # W24 counted the result: 67 unclaused "from [nothing]" lines, 66
        # REFUSED and 1 CANNOT-FULLY-HOLD, every one of them a standalone
        # corridor-guard emission.
        # #689 A RECLAIM ASK IS JUDGED BY WHAT MOVED.
        #
        # ``free_now >= want`` asks "is want allocatable", which is exactly
        # right for an allocator about to allocate it -- every existing caller
        # is one, so their verdict is untouched and must_reclaim defaults off.
        #
        # It is the WRONG question for a caller that already accounted the
        # memory and needs this ladder to FREE more. Measured 2026-08-16
        # 12:29, three consecutive seam asks on the binding rank:
        #     asked the corridor guard for 178 MiB (pp_to_tp): ok=True,
        #     spendable now 609 MiB against a need of 788 MiB
        # 609 never moved. With 1428 MiB of driver-free, "are 178 MiB free"
        # was trivially true while the ladder reclaimed nothing, so the seam
        # was told it was funded and abandoned anyway. A success that is true
        # of the world before the call is not a report about the call.
        #
        # It also propagates: fundable_width's pre-arm picture is built from
        # what this returns, so an optimistic guard forms a window the seam
        # cannot carry and moves the failure later, into an abandon.
        if must_reclaim and ok and reclaimed < want:
            ok = False
            # THE MESSAGE MUST NOT QUOTE TERMS IT DID NOT JUDGE. The default
            # verdict weighs free against the floor; this one does not weigh
            # them at all, and a refusal that recites "free 2518, arming floor
            # 1536, corridor law 1024" next to "want 6" reads as an inversion
            # -- it was reported as one. Under must_reclaim the ONLY quantities
            # in the verdict are asked-vs-reclaimed, so those are the only ones
            # stated, with the reason the free column is irrelevant here.
            detail = (
                f"want {want / _MIB:.0f} MiB INCREMENTAL, reclaimed "
                f"{reclaimed / _MIB:.0f} MiB from "
                f"[{', '.join(used) if used else 'nothing'}] "
                f"({reason}). REFUSED under must_reclaim, which judges the "
                f"DELTA and nothing else: the caller has already accounted the "
                f"free column and is asking this ladder to RELEASE more, so "
                f"free memory it did not release cannot fund the ask and is "
                f"not weighed. Providers available: "
                f"{', '.join(self.providers) or 'none'}"
            )
        if host_forced and used_host:
            detail += (
                "; host tier admitted on an UNLEVEL fleet because refusal "
                f"would deadlock (free column {[int(f // _MIB) for f in column]} "
                f"MiB, spread {free_spread_mib(column)} MiB)"
            )
            clause = describe_water_fill(column)
            if clause:
                detail += f". {clause}"
        if host_blocked:
            detail += (
                "; host tier withheld -- fleet is not level"
                f" (free column {[int(f // _MIB) for f in column]} MiB, spread "
                f"{free_spread_mib(column)} MiB): item 16 spends host RAM only "
                "once every card is at the floor"
            )
            # NAME THE MOVE, not just the unevenness. "spread 879 MiB" does
            # not say which card to fix or by how much, and the decision a
            # successor faces -- is the missing rebalance tier worth building
            # -- turns on exactly that number.
            clause = describe_water_fill(column)
            if clause:
                detail += f". {clause}"
        # #851 F4, COMPLETED BY #853(ii): SAY WHAT THIS GATE COULD NOT DRAW,
        # instead of leaving "[nothing]" to mean two different things.
        #
        # W22 printed "reclaimed 0 MiB from [nothing]" 39 times while the
        # kv-slack post held 2776 MiB. "[nothing]" is the LADDER's provider
        # list, and it is honest about the ladder -- but read as a sentence it
        # says "this rig had no memory to give", which sends the next reader to
        # capacity planning. The true statement is "this rig had a funder this
        # gate may not spend". One window was partly spent on that difference.
        #
        # HERE, at the last possible point, because every earlier position is
        # reachable by only some of the emissions -- that was the #853(ii)
        # defect exactly. Every path that ends in a REFUSED or a
        # CANNOT-FULLY-HOLD line passes through this statement.
        #
        # THE LAW BREACH IS INCLUDED EVEN THOUGH IT SUCCEEDS. It is a report
        # about memory pressure, which is precisely when a reader needs to know
        # a funder was in the room and unreachable.
        #
        # Appended, never substituted: the ladder's own list stays exactly as
        # it was, because it is the truthful record of what the gate actually
        # spent. And a clean success still says nothing -- an ask that needed
        # no money does not have to explain the money it did not need.
        # #852 CLASS: NAME THE DIVERGENCE ON THE FIRST PASS IT HAPPENS.
        #
        # Discovering that a funder promised ~320 MiB and delivered 0 cost W24
        # a whole GPU window and a ticket, and the evidence was in hand at the
        # instant of the draw. Placed HERE, at the same last-possible point as
        # the F4 clause and for the same #853(ii) reason: every path that ends
        # in a REFUSED, a CLEARED or a CANNOT-FULLY-HOLD line passes through
        # this statement, and the ``must_reclaim`` branch above rebuilds
        # ``detail`` from scratch, so anything appended earlier is lost on
        # exactly the emission that needs it most.
        #
        # UNCONDITIONAL, unlike F4's clause. A CLEARED pass that spent a
        # phantom provider is the case that HIDES this defect -- W25 cleared
        # its way to 8 phantom_capacity readings and a TP-sticky instance --
        # so a report gated on refusal would go quiet precisely where the
        # ladder is being lied to and getting away with it. It costs nothing
        # on an honest ladder: the clause is empty unless a provider this pass
        # claimed bytes the driver did not see.
        #
        # Appended, never substituted: the ladder's own list stays the truthful
        # record of what the gate spent.
        detail += self._phantom_clause(used)
        if not ok or law_breached:
            named = self._offledger_detail(want)
            if named:
                detail = f"{detail}; {named}"
        if ok and law_breached:
            self.law_dip_count = getattr(self, "law_dip_count", 0) + 1
            logger.warning(
                "%s CANNOT FULLY HOLD THE CORRIDOR FLOOR through this seam "
                "entry on device %d: predicted trough %d MiB below the %d MiB "
                "law. PROCEEDING -- the law is a fill-quality target, not a "
                "gate (user decision 2026-08-16), and THE ALLOCATION THIS GATE "
                "WAS ASKED ABOUT fits. That is the whole of the claim: the "
                "trough is PREDICTED from this pass's providers and does NOT "
                "bound what a later kernel allocates on the same device. "
                "REFUTED ONCE, and the sentence that used to stand here said "
                "'this is a dip and not an OOM' -- boot 24 (2026-08-31) "
                "emitted this line three times on PP0 and took a CUDA OOM in "
                "the third one's own second, at fla/chunk_o.py:146 "
                "torch.zeros_like(v) inside the GDN extend kernel, a "
                "transient this predictor has no term for. Read this as a "
                "WARNING WITH AN UNMEASURED RESIDUAL, never as an all-clear. "
                "(dip %d of this process.) %s",
                LOG_PREFIX,
                self.device_index,
                (self.law_floor_bytes - (free_now - want)) / _MIB,
                self.law_floor_mib,
                self.law_dip_count,
                detail,
            )
            # #1054: capture the allocator on the FIRST dip of the process.
            # This line is the last honest warning before the class of death
            # boot 24 died of, and until now nothing recorded who held the
            # memory when it fired -- the log carried a predicted trough and no
            # attribution for the 31 GiB behind it. `dump_trace` is a no-op
            # unless the flight recorder was armed, so an ordinary boot pays
            # one attribute lookup. First dip only: the interesting state is
            # the onset, and a dump per dip would write gigabytes under load.
            if self.law_dip_count == 1:
                try:
                    from flliper.srt.mem_ledger import flight_recorder

                    flight_recorder.dump_trace(
                        "corridor_first_dip", rank=self.device_index
                    )
                except Exception:  # noqa: BLE001 - a probe may never break a seam
                    logger.debug("#1054: corridor-dip snapshot unavailable")
        elif ok:
            logger.info(
                "%s cleared on device %d: %s", LOG_PREFIX, self.device_index, detail
            )
        else:
            logger.error(
                "%s REFUSED on device %d: %s. Every provider is exhausted, so "
                "this allocation cannot be made without breaking the corridor "
                "law. The caller must take its refusal path -- allocating "
                "anyway would launder a breach as a passed check.",
                LOG_PREFIX,
                self.device_index,
                detail,
            )
            if raise_on_refusal:
                raise CorridorBreachRefused(f"{LOG_PREFIX} {detail}")
        return GuardResult(
            ok,
            free_before,
            free_now,
            want,
            reclaimed,
            tuple(used),
            detail,
            law_breached=law_breached,
        )

    # -- the ladder, shared by the gate and the lender ---------------------

    def _spend_ladder(
        self,
        *,
        target: int,
        free_now: int,
        column: Sequence[int],
        host_ok: bool,
        host_forced: bool,
        max_tier: int,
        reason: str,
    ):
        """Spend providers, cheapest tier first, until ``target`` free or dry.

        ONE ladder, two callers. :meth:`ensure_headroom` runs it with
        ``max_tier=RELIEF_HOST`` (spend anything the fleet permits);
        :meth:`lend_to_level` runs it with ``max_tier=RELIEF_REBALANCE``, so
        item 16's first relief stage physically cannot reach a park or a host
        spill. Extracting it was the alternative to a second spend loop, which
        in this module is how two policies drift apart while both look right.

        ``max_tier`` may terminate the loop rather than skip, because
        ``_providers`` is kept sorted by ``(tier, cost)``.

        RETURNS DELIVERED AND CLAIMED, SEPARATELY (#852). The loop already
        re-probed the driver after every provider -- to decide when to stop --
        and then discarded the delta, which is the one number the corridor law
        is written in. Summing the providers' own return values instead is how
        this ladder credited bytes that went to an allocator free-list, and the
        ``must_reclaim`` branch DECIDES on that sum. Both figures now leave
        here: the delivered one for every judgement, the claimed one so a
        divergence can be NAMED rather than silently absorbed.
        """
        used: List[str] = []
        used_host = False
        host_blocked = False
        reclaimed = 0
        claimed_total = 0
        for provider in self._providers:
            if free_now >= target:
                break
            if provider.tier > max_tier:
                break
            if provider.tier == RELIEF_HOST and host_forced and not used_host:
                used_host = True
                self.host_forced_count += 1
                logger.warning(
                    "%s spending HOST RAM on device %d while the fleet is NOT "
                    "level (free column %s MiB, spread %d MiB), because "
                    "refusing here would deadlock the caller. Item 16 wanted "
                    "these bytes rebalanced onto a peer card instead -- this "
                    "counter is the cost of the missing rebalance tier, not a "
                    "licence. (%s)",
                    LOG_PREFIX,
                    self.device_index,
                    [int(f // _MIB) for f in column],
                    free_spread_mib(column),
                    reason or "no reason given",
                )
            if provider.tier == RELIEF_HOST and not host_ok:
                # Never while a peer still has headroom: the bytes belong on
                # that card, not in RAM.
                host_blocked = True
                continue
            deficit = target - free_now
            try:
                got = int(provider.free_up_to(deficit))
            except Exception as e:
                # A provider that fails must not take the allocation down;
                # the gate simply has less to spend and may still refuse.
                logger.warning(
                    "%s provider %r raised while freeing: %s",
                    LOG_PREFIX,
                    provider.name,
                    e,
                )
                continue
            if got <= 0:
                # A provider that declined was still ASKED. Recording the pass
                # with a zero claim keeps "asked and had nothing" apart from
                # "never asked", and only the second is an unmeasured post.
                self._record_delivery(provider.name, claimed=0, delivered=0)
                continue
            claimed_total += got
            used.append(provider.name)
            # Re-probe rather than trusting the provider's arithmetic: the
            # law is what the DRIVER reports, and a provider that returns
            # its payload size while the pages went to torch's cache has
            # freed nothing the corridor can see.
            free_before_provider = free_now
            free_now = self.free_bytes()
            # #852: THE DELTA WAS ALWAYS HERE, AND WAS ALWAYS THROWN AWAY.
            # A fall between the probes -- another process taking memory
            # mid-ladder -- reads as 0 rather than as a negative, matching the
            # rule ``lend_to_level`` already applies to its own figure; a
            # negative credit would make the running total lie in the other
            # direction. The converse caveat is the lender's too and is stated
            # there: the delta credits this provider with anything another
            # process happened to release in the same interval, so it
            # over-reports in the PERMISSIVE direction, never the suppressive
            # one. That asymmetry is deliberate -- an under-reporting ladder
            # would refuse relief that was really available and make the flip
            # stickier, which is the defect #852 exists to remove.
            delivered = max(0, free_now - free_before_provider)
            reclaimed += delivered
            self._record_delivery(provider.name, claimed=got, delivered=delivered)

        return reclaimed, claimed_total, free_now, used, used_host, host_blocked

    def _record_delivery(self, name: str, *, claimed: int, delivered: int) -> None:
        """One observation into the promised-vs-delivered ledger (#852)."""
        rec = self._delivery.setdefault(name, ProviderDelivery(name))
        rec.observations += 1
        rec.claimed_bytes += int(claimed)
        rec.delivered_bytes += int(delivered)
        rec.last_claimed_bytes = int(claimed)
        rec.last_delivered_bytes = int(delivered)
        if claimed > 0 and delivered <= 0:
            rec.phantom_passes += 1

    def delivery_report(self) -> Dict[str, ProviderDelivery]:
        """What every registered provider PROMISED against what it DELIVERED.

        The answer #852 could give for one funder only, made askable of all of
        them. Includes providers that have never been called -- their record
        reads ``observations == 0`` and ``delivery_ratio is None``, which is
        the honest "never measured", not a zero.
        """
        return dict(self._delivery)

    def _phantom_clause(self, used: Sequence[str]) -> str:
        """The sentence W24 cost a GPU window to discover, priced at nothing.

        EDGE-TRIGGERED ON DIVERGENCE, and scoped to the providers this pass
        actually spent. A clause that fires on every pass is a clause nobody
        reads, and the honest ladder is the common case.
        """
        parts = []
        for name in used:
            rec = self._delivery.get(name)
            if rec is None or not rec.is_phantom:
                continue
            parts.append(
                f"{name} claimed {rec.last_claimed_bytes / _MIB:.0f} MiB and "
                f"delivered only {rec.last_delivered_bytes / _MIB:.0f} MiB to "
                f"the driver (phantom on {rec.phantom_passes} of "
                f"{rec.observations} passes)"
            )
        if not parts:
            return ""
        return (
            "; PHANTOM CAPACITY: "
            + "; ".join(parts)
            + ". The corridor law is stated in the driver's free column, so a "
            "release that only reached torch's cache funds nothing this gate "
            "can spend"
        )

    # -- item 16's first relief stage --------------------------------------

    def lend_to_level(
        self,
        bound_bytes: int,
        *,
        column: Sequence[int],
        max_tier: int = RELIEF_REBALANCE,
        reason: str = "",
    ) -> GuardResult:
        """Give up to ``bound_bytes`` back on THIS card because the fleet is
        unlevel -- before any allocation has asked for them.

        WHAT MAKES THIS THE REBALANCE TIER AND NOT A SECOND GATE. The gate is
        reactive by construction: it arms on an allocation that would breach,
        which means the trough has already been reached by the time relief is
        spent. s34's green window measured that trough at 19 MiB of margin on
        the binding card while a peer held 3280 MiB free -- a PLACEMENT
        problem, not a capacity one. The lender is the same ladder, spent on
        the schedule the water-fill dictates instead of the schedule the
        allocator dictates.

        THE BOUND IS THE OBJECTIVE, NOT A DEFICIT. ``bound_bytes`` comes from
        :func:`water_fill_transfers` -- the payload this card must shed to sit
        level with its peers. Freeing more than that would evacuate a card
        that is no longer the tightest, which is the same unevenness with the
        sign flipped, and it would pay restore costs for nothing.

        IT PHYSICALLY CANNOT REACH HOST RAM. ``max_tier`` defaults to
        ``RELIEF_REBALANCE`` and the ladder terminates there, so the lender
        can never do what item 16 forbids -- spill to RAM while a peer has
        headroom -- no matter what a caller passes as a bound. Host RAM stays
        exactly where item 15c put it: the last stage, reached only through
        :meth:`ensure_headroom`, only once every card is at the floor.

        It never touches the KV rung either: that one is collective (it moves
        ``available_size()`` and therefore admission), and a rank-local lender
        that shrank the pool would be "a smaller pool as the fix", which the
        standing rule forbids and which this method is the alternative to.
        """
        bound = max(0, int(bound_bytes))
        free_before = self.free_bytes()
        if bound == 0:
            return GuardResult(
                True, free_before, free_before, 0, 0, (), "nothing to lend"
            )
        target = free_before + bound
        # HARD CEILING, not a default a caller can lift. Park is legitimate
        # here -- parking a cold payload in a peer card's surplus IS the
        # redistribution item 16 asks for -- but host RAM is the last stage by
        # user order, and the lender runs precisely when the fleet is UNLEVEL,
        # which is the one state in which host RAM is forbidden.
        ceiling = min(int(max_tier), RELIEF_PARK)
        _delivered, claimed, free_now, used, _used_host, _blocked = self._spend_ladder(
            target=target,
            free_now=free_before,
            column=list(column),
            # The fleet is unlevel by construction -- that is why the lender
            # was called -- so the host tier is shut and cannot be forced.
            host_ok=False,
            host_forced=False,
            max_tier=ceiling,
            reason=reason,
        )
        # THE MEASURED DELTA, NOT THE SUM OF THE PROVIDERS' CLAIMS. The gate
        # can afford to credit claims because its verdict is re-probed anyway;
        # the lender's whole output IS the number, and this chain has three
        # times credited bytes that went to an allocator free-list instead of
        # to the driver. A fall in free between the probes (another process
        # taking memory) reads as 0 rather than as a negative.
        measured = max(0, free_now - free_before)
        self.lend_count += 1
        self.lent_total += measured
        # DELIBERATELY NOT added to ``reclaimed_total``. That counter answers
        # "what did the GATE spend", and ``reclaimed_total / arm_count`` is a
        # figure a successor will compute; folding lends into it pollutes both
        # and sets up a double count for anyone who later sums the two.
        # ``lent_total`` carries the lender's own bytes.
        #
        # One caveat this figure carries, named because the negative case is
        # already named and the positive one was not: the delta credits the
        # lend with any memory ANOTHER process released between the two
        # probes. It over-reports in the permissive direction.
        detail = (
            f"lent {measured / _MIB:.0f} MiB of a {bound / _MIB:.0f} MiB "
            f"water-fill bound, free {free_before / _MIB:.0f} -> "
            f"{free_now / _MIB:.0f} MiB, from [{', '.join(used) or 'nothing'}], "
            f"column {[int(f // _MIB) for f in column]} MiB, spread "
            f"{free_spread_mib(column)} MiB"
            + (
                f" (providers claimed {claimed / _MIB:.0f} MiB)"
                if claimed != measured
                else ""
            )
            + (f" ({reason})" if reason else "")
        )
        return GuardResult(
            True, free_before, free_now, 0, measured, tuple(used), detail
        )


def nvml_fleet_probe() -> Callable[[], List[int]]:
    """A fleet free-column probe over NVML, for item 16's host gate.

    NVML, deliberately, and not a collective. This gate runs inside the
    flip's no-return region; a collective there deadlocks the moment one rank
    takes a different branch, and the whole reason the seam is guarded is
    that ranks can disagree about affordability. NVML sidesteps the question:
    it sees every physical card regardless of ``CUDA_VISIBLE_DEVICES``, so a
    rank pinned to one card can still read the other two without asking
    anyone.

    The column is in NVML index order and includes cards this instance does
    not use. That is correct for the predicate being asked -- "is there
    headroom anywhere else" -- and it is conservative in the safe direction:
    a foreign card with free memory keeps the host tier SHUT, which costs
    tempo and never costs the corridor.

    Only called when the gate arms, which is rare, so no cache: a cached free
    column can only be wrong in the direction that opens the host gate.
    """

    def probe() -> List[int]:
        from flliper.srt.registry import nvml as registry_nvml

        with registry_nvml.nvml_session() as pynvml:
            out: List[int] = []
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                out.append(int(pynvml.nvmlDeviceGetMemoryInfo(handle).free))
            return out

    return probe


def allocator_cache_provider(
    probe: Callable[[], int],
    *,
    empty_cache: Optional[Callable[[], None]] = None,
) -> Callable[[int], int]:
    """Return torch's unused cached blocks to the driver. The cheapest relief.

    WHY THIS IS A PROVIDER AND NOT AN IMPLEMENTATION DETAIL. The seam already
    reclaims the allocator cache -- but inside ``_staging_affordable``, which
    runs AFTER the corridor gate. So the gate formed its verdict against a
    free column that understated the truth by the size of the hoard, and on
    this rig the hoard is 1028-1426 MiB per card at idle (register: flush
    contamination). The gate could therefore REFUSE a ``pp->tp`` flip -- the
    leg that starves decode outright under strict purity -- or force the host
    tier onto an unlevel fleet, while a gibibyte of nobody's memory sat on the
    card. Registering it makes the gate spend it FIRST, before any payload
    moves anywhere.

    THE RETURN VALUE IS A MEASURED DELTA, NOT ``memory_reserved() -
    memory_allocated()``. That difference is the size of the hoard, not the
    size of the release: the allocator keeps whole segments it is still
    carving from, so the two numbers routinely disagree. Crediting the hoard
    would be the same class of error as crediting a free-list push -- bytes
    the corridor law cannot see -- which this chain has now made three times.
    So the provider probes, flushes, probes again, and reports the difference.

    A fall in driver-free between the two probes (another process taking
    memory) is reported as 0 rather than as a negative number: the guard sums
    these into ``reclaimed_total`` and a negative would make the accounting
    lie in the permissive direction.

    ``nbytes`` is ignored, honestly rather than silently: ``empty_cache`` is
    all-or-nothing and cannot free a bounded amount. Over-delivering is not a
    failure -- the guard re-probes the driver and stops asking once the target
    is met.
    """

    def free_up_to(_nbytes: int) -> int:
        try:
            before = int(probe())
            if empty_cache is not None:
                empty_cache()
            else:
                import torch

                torch.cuda.empty_cache()
            after = int(probe())
        except Exception as e:
            logger.warning("%s allocator cache reclaim failed: %s", LOG_PREFIX, e)
            return 0
        return max(0, after - before)

    return free_up_to


def draft_carrier_provider(carrier) -> Callable[[int], int]:
    """Adapt a :class:`VmmDraftWeightCarrier` to the provider protocol.

    Cheapest real provider on this rig and the natural first registration:
    the payload is already proven to return its pages to the driver, the
    restore is priced into the seam's affordability verdict, and under strict
    purity the drafter is idle for the whole PP phase.

    It is ALL-OR-NOTHING -- ``decommit_range`` releases whole extents and the
    drafter is either resident or not -- so a request for fewer bytes than
    the payload still frees the whole payload. That is reported honestly
    rather than clipped, because the guard re-probes the driver anyway and
    over-delivering is not a failure.
    """

    def free_up_to(_nbytes: int) -> int:
        if carrier is None or carrier.spilled:
            return 0
        return int(carrier.spill() * _MIB)

    return free_up_to
