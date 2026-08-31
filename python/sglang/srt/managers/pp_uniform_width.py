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

from typing import NamedTuple, Optional


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
    # EXECUTION, not decision: a rank holding fewer leading tokens than it was
    # told recomputes the gap. Negative differences are surplus cache the rank
    # simply does not use this pass -- also execution, also invisible here.
    shortfall = told_prefix - int(local_prefix)
    if shortfall < 0:
        shortfall = 0

    return PassGeometry(
        prefix=told_prefix,
        extend=None if told_extend is None else int(told_extend),
        shortfall=shortfall,
        adopted=True,
    )
