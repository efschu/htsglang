"""#1059 ARM A': the group-visible pass geometry is a DECISION, not an observation.

THE ONE PLACE, as established by the convergence proof
(`/spinning/gpu-arb/KONVERGENZ_BEWEIS_968.md`): `get_next_batch_to_run` +
admission run as a COPY on every rank -- proven for the TP phase by py-spy on a
live PP1 and for the PP phase by `scheduler_pp_mixin.py:4402` plus metal
(`#969N ADMIT` on PP0 and PP1 in the same pass). 14 of 16 rank-divergence
blockers converge there. Each blocker is one divergent INPUT into that one
replicated function; this module removes the divergence at the output instead
of chasing the inputs one at a time.

WHERE CONGRUENCE IS ACTUALLY LOST, sharply. `pp_chunked_local_match`'s docstring
argues the pass geometry is "CONGRUENT BY CONSTRUCTION": every rank advances a
chunked request through the same `add_chunked_req` sequence, so every rank's
`extend_range.end` is the same number. That argument is sound for a request
advancing IN PHASE -- and it has no coverage at a POST-CUTOVER RE-ADMISSION,
where `prefix_indices` is reseeded from each rank's OWN HiCache host hit
(`#969B READMIT-MATCH ... host_hit=15932`, schedule_batch.py:1758). Boots 26 and
27 both died within seconds of a cutover, which is exactly that hole:

    boot 27, same slot, same fwd_ct, same rids, different width
    [PP0] #969N ADMIT slot=0 fwd_ct=77 bs=2 extend=4096
    [PP1] #969N ADMIT slot=0 fwd_ct=77 bs=2 extend=2596

THE RULE, in one sentence: the told geometry is the DECISION and is
rank-uniform; whether a rank READS its prefix bytes from its tier or RECOMPUTES
them is EXECUTION and is rank-local. A rank that holds less than it was told
does more work; it does not present a different batch.

WHY THERE IS NO REFUSAL AND NO VOID IN THIS MODULE, structurally rather than by
discipline. Both were tried and both killed a boot:

  * per-rank refusal on an uncoverable told value -- BOOT 15 (#1048): "PP0
    published an extent of 4618 token(s) and this rank's load-back yielded only
    0", 1448 refusals on ONE rid, every rank every pass, until the ring wedged
    (#789) and the schedulers died. #1048's own fix DELETED that raise.
  * #995 v1: a refusal on the live path with no way onward, 175 refusals on one
    rid and a dead window.

So an overshoot is made HARMLESS rather than IMPOSSIBLE. That is also why this
needs no coverage feed: the ordered form of Arm A wanted a MIN over per-rank
coverage facts returned on the ring lap, and those facts never come home --
`observed_local` has exactly ONE writer, the dataclass default
(`PPAdmissionEntry:288`), because its only real writer
(`reconcile_pp_admission_decision`) has 0 call sites and is deliberately dark.
Recomputation buys the same uniformity without lighting that second mechanism,
at the standing #939 price of at most one HiCache chunk.

ABSENT FACT = NO ADOPTION, NEVER A LOCAL SUBSTITUTE. A rank that was told
nothing keeps exactly its pre-#1059 behaviour, so an older sender, a stand-in,
or a pass PP0 did not name is byte-identical to before.
"""

from typing import Callable, Iterable, List, NamedTuple, Optional

#: #1061: a read-through accessor to the ONE authority for the cutover epoch
#: (`Scheduler._pp_flip_epoch` -> `PhaseFlipRuntime._epoch`). Registered once by
#: the scheduler; called LIVE at every apply.
#:
#: A CALLABLE AND NOT A CACHED INT, deliberately. Boot 30 died inside
#: `_release_residents_for_cutover` -- i.e. the apply that needed the epoch ran
#: DURING the cutover, after the tree was dropped. A value refreshed at the top
#: of the pass would have been one cutover stale exactly there, which is the
#: staleness this whole mechanism exists to detect. Reading through to the
#: authority on every call cannot be stale, and it is not a second bookkeeping:
#: nothing is stored.
_EPOCH_SOURCE: List[Optional[Callable[[], Optional[int]]]] = [None]


def set_epoch_source(fn: Optional[Callable[[], Optional[int]]]) -> None:
    """Register the live cutover-epoch accessor (idempotent, never raises)."""
    _EPOCH_SOURCE[0] = fn


def current_epoch() -> Optional[int]:
    """The rank's CURRENT cutover epoch, or None if there is no epoch namespace.

    None means "this deployment has no cutovers to tell apart" (no flip
    runtime, a stand-in, an unreadable runtime) -- never "epoch zero".
    """
    fn = _EPOCH_SOURCE[0]
    if fn is None:
        return None
    try:
        v = fn()
    except Exception:  # noqa: BLE001 - an unreadable epoch names no epoch
        return None
    return None if v is None else int(v)


def epoch_admits_row(row_epoch: Optional[int], now_epoch: Optional[int]) -> bool:
    """May a row decided under ``row_epoch`` be applied under ``now_epoch``?

    THE VERDICT IS A FUNCTION OF TWO GROUP QUANTITIES AND NOTHING ELSE. No rank
    consults its own span, its own tier, or its own request here -- which is the
    entire point, and the trap the obvious formulation falls into. "Void when
    the told value exceeds MY pinned span" would be a per-rank decision, i.e.
    the 26th divergent input into the one replicated decision and precisely what
    `UniformWidthPromiseBroken`'s own message forbids: *"compensating locally
    here would move the batch and reappear as #631 on a peer"*. Because the
    epoch advances once per COMPLETED cutover on every rank (and the runtime's
    consensus reduction raises DESYNC if two ranks disagree), every rank
    computes the SAME answer from the SAME two numbers.

    THE FOUR CASES, each with its reason:

    * both None -- no epoch namespace exists (a boot without the phase flip, a
      stand-in). There are no cutovers, so there is no staleness to detect and
      the gate is VACUOUS: admit, which is byte-identical to the pre-#1061 tree.
    * equal -- same generation, admit.
    * different -- the row was decided before a cutover this rank has already
      completed. STALE: refuse uniformly. This is boot 30 (told=12288 from
      before the cutover, applied to a request the cutover had just re-admitted
      with a span of 0).
    * exactly one is None -- the row cannot be placed in this rank's epoch
      namespace, so it is UNVERIFIABLE. Refused, by the module's standing law
      "ABSENT FACT = NO ADOPTION, NEVER A LOCAL SUBSTITUTE": an unverifiable
      fact is an absent one, and the no-adopt path is the contract's own
      already-proven behaviour (boot 29 ran it for its whole life, divergence
      free).

    Refusal here is NOT the boot-15 shape. Boot 15 was a per-rank REFUSAL OF THE
    PASS on a reachable condition -- 1448 refusals on one rid until the ring
    wedged. This returns False, which routes into `uniform_pass_geometry`'s
    already-existing `told_prefix is None` branch: the rank runs its own
    geometry and the request proceeds. It is a way ONWARD, not a refusal.
    """
    if row_epoch is None and now_epoch is None:
        return True
    if row_epoch is None or now_epoch is None:
        return False
    return int(row_epoch) == int(now_epoch)


class UniformWidthPromiseBroken(RuntimeError):
    """A rank was told a prefix its PINNED span does not cover.

    THIS IS AN INVARIANT VIOLATION, NOT FLOW CONTROL, and the distinction is the
    whole reason it may raise at all. `report_local_coverage` pins the span it
    reports (`cache_protected_len`, which `mem_cache/common.py:82` already
    honours as an eviction floor: `evict_floor = max(req.cache_protected_len,
    req.swa_evict_floor)`), and PP0 publishes the MIN over those promises. So
    `told <= pinned` holds BY CONSTRUCTION and this exception is unreachable
    unless the pin itself failed.

    Raising here is `raenge-nie-uneins` applied literally: a detected broken
    promise is a group crash, never a per-rank compensation. It is NOT the
    boot-15 shape -- that was a per-pass refusal on a reachable condition, which
    re-fired 1448 times and wedged the ring. This condition is unreachable while
    the pin holds, fires once, and names both numbers.

    #1061: IT WAS REACHABLE, AND IT KILLED BOOT 30 -- but not because the pin
    failed. `#1059c` gave the row a sender for the first time; a told prefix of
    12288 decided BEFORE a cutover was applied to a request that same cutover had
    just re-admitted, after the tree was dropped, so the pin was legitimately 0.
    The premise "told <= pinned holds by construction" is true only WITHIN one
    cutover epoch, and nothing enforced that.

    THE GATE MOVED, THE GUARD STAYED. `epoch_admits_row` now refuses a
    cross-epoch row uniformly on every rank BEFORE this function sees it, which
    makes the cross-epoch path unreachable here. The guard is deliberately NOT
    softened into a void: it is the in-epoch invariant's backstop, and if it ever
    fires again the pin itself really has failed -- which is a group crash by
    `raenge-nie-uneins`, exactly as written."""


def report_local_coverage(local_prefix: int) -> int:
    """What this rank promises for the NEXT lap: reporting IS pinning.

    The eviction-between-laps gap, closed by construction. `observed_local` is
    reported on lap N and applied on lap N+1; if the tier evicted the span in
    between, `told > local` would hold despite the MIN -- and a prefix
    shortfall is NOT absorbable (see DESIGN_968 5f: prefix is a START POSITION,
    not a read amount). So a rank that reports a span also pins it, and the
    caller must set `cache_protected_len >= ` this value until the apply
    releases it or the lap expires.

    Reuses the existing eviction floor rather than adding a second protection
    mechanism -- upstream-minimal at this seam.
    """
    return max(0, int(local_prefix))


def min_told(reported: Iterable[Optional[int]]) -> Optional[int]:
    """PP0's published value: the MIN over the promises that came home.

    A rank that reported nothing (None) is NOT counted as zero -- that would let
    one silent rank collapse the group's prefix to 0 and recompute everything.
    It is skipped, and if NOBODY reported, the answer is None = no fact = no
    adoption, never a local substitute.
    """
    vals = [int(v) for v in reported if v is not None]
    return min(vals) if vals else None


class PassGeometry(NamedTuple):
    """What this rank will present to the pipeline for one request.

    ``prefix`` and ``extend`` are the GROUP-VISIBLE numbers and must be equal on
    every rank of the group. ``shortfall`` is execution-local bookkeeping: how
    many leading tokens this rank must recompute because its own tier holds
    fewer than it was told. It is reported so the cost is counted, and it must
    never feed back into ``prefix`` or ``extend``.
    """

    prefix: int
    extend: Optional[int]
    shortfall: int
    adopted: bool


def uniform_pass_geometry(
    told_prefix: Optional[int],
    told_extend: Optional[int],
    local_prefix: int,
    pinned_prefix: Optional[int] = None,
) -> PassGeometry:
    """The pass geometry every rank of the group runs, told-first.

    THE INVARIANT THIS FUNCTION EXISTS TO MAKE UNBREAKABLE: when a told value is
    present, the returned ``prefix``/``extend`` are functions of the TOLD values
    ALONE. ``local_prefix`` may only influence ``shortfall``. Any edit that lets
    the local tier reach the geometry reintroduces the 26th divergent input into
    the one replicated decision, and `test_pp_uniform_width_1059` fails.

    There is deliberately no error return, no sentinel and no exception: every
    input maps to a geometry the caller can run. A branch that refuses here
    would be the boot-15 shape, and the way onward for a request PP0 does not
    name already exists one layer up (`_note_skip("pp_not_named", ...)` leaves
    it in `waiting_queue` for a later pass -- "the same requeue-for-free
    mechanism this loop already relies on").
    """
    if told_prefix is None:
        # No fact. Pre-#1059 behaviour, exactly.
        return PassGeometry(
            prefix=local_prefix, extend=told_extend, shortfall=0, adopted=False
        )

    told_prefix = int(told_prefix)

    # THE EVICTION-BETWEEN-LAPS CASE, given its defined answer instead of a
    # silent one. `told` was MINned over promises made on the previous lap; the
    # pin (`cache_protected_len`) is what makes those promises still true now.
    # A silent `told > local` must not be constructible here -- that is exactly
    # the shape that cannot be absorbed (5f) and would surface as #631 three
    # ranks later instead of at its cause.
    effective_pin = local_prefix if pinned_prefix is None else int(pinned_prefix)
    if told_prefix > effective_pin:
        raise UniformWidthPromiseBroken(
            f"told prefix {told_prefix} exceeds this rank's pinned span "
            f"{effective_pin} (live local {int(local_prefix)}). The pin is what "
            "makes the MIN realizable, so this is a broken promise, not a "
            "capacity event: either the span was released before its lap "
            "expired or it was evicted despite cache_protected_len. Crashing "
            "the group is raenge-nie-uneins; compensating locally here would "
            "move the batch and reappear as #631 on a peer."
        )

    # EXECUTION, not decision: the pin guarantees the tokens are still there, so
    # a rank whose LIVE match came back short simply re-reads the pinned span.
    # Surplus cache is likewise execution -- it is not used this pass.
    shortfall = told_prefix - int(local_prefix)
    if shortfall < 0:
        shortfall = 0

    return PassGeometry(
        prefix=told_prefix,
        extend=None if told_extend is None else int(told_extend),
        shortfall=shortfall,
        adopted=True,
    )
