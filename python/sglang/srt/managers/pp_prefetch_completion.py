"""#1175: the group's per-rid store-prefetch completion, formed WITHOUT a
collective on the admission path.

THE DEFECT THIS CLOSES (boot_855_weg1b5_cd5bb69607_0903_115008, rid
0c34259f, 11:55:13-11:55:30). After the `tp_to_pp` cutover of generation 4
all three ranks issued a re-admission prefetch for the same rid and the same
key set (`#1028B FETCH CAP ... keys=13224` byte-identical on PP0/PP1/PP2).
PP0's completed in ~3 s, printed `completed_local=12288
completed_synced=12288`, and ADMITted `prefix_lens=12288`. PP1's and PP2's
had not terminated. PP1 then received the forwarded row and entered
`execute_scheduled_prefix`, whose length-priced bound expired 14 s later as
`#968 PREFIX MATERIALISATION SHORTFALL` -- the designed group STOP for
detected rank divergence.

THE NUMBER PP0 ADMITTED ON WAS RANK-LOCAL BY CONSTRUCTION. `completed_synced`
is `min_completed_tokens`, which `UnifiedRadixCache.check_prefetch_progress`
overwrites from a packed MIN all_reduce ONLY under `if self.tp_world_size >
1:`. `tp_world_size` is the ATTENTION-TP cache group; this boot runs
`--tp-size 1 --pp-size 3`, so the reduce never executed -- measured,
`attn_reduce_world=1` on 307/307 `#1028 HICACHE-ROUND` lines of the whole
log, both phases. The field named "synced" synced nothing (Instrument-Text-
luegt Klasse A), and the group that actually diverges under PP is the PP
group, which that reduce does not span.

WHY NOT SIMPLY REDUCE OVER `pp_group`. The PP0-authority order
(`memory/pp0-autoritativer-umbau-order.md`) records a collective on the
admission path as fatal, and `_drain_prefetch_progress` -> the admission
loop IS that path. So the fact travels the way every other per-rid fact in
this feature travels: it RIDES THE RING LAP the followers already send
upstream, exactly as `#968`'s parked-continuation table does
(`scheduler_pp_mixin.pp_parked_continuation_stamp` / `..._note_parked_...`).
No new message, no new collective, no new blocking point.

WHAT A FOLLOWER REPORTS, and why it is that quantity: the number of prefix
tokens its own storage prefetch has COMPLETED for that rid
(`UnifiedRadixCache.completed_prefetch_tokens`), plus `PENDING` while an
operation is still registered and unterminated. That is precisely the
quantity whose asymmetry killed the boot -- PP0 12288, the followers
nothing -- and it is rank-local, cheap, and already computed.

WHAT PP0 DOES WITH IT (`group_completion_verdict`):

  * `floor` is the MIN over the peers' reports. A peer that has not
    reported at all, or reports PENDING, contributes NO NUMBER: it makes the
    verdict INCOMPLETE, never a zero. "I have not spoken" and "I have zero"
    are different facts and a floor built from the first would collapse every
    admission to 0 (the same rule `note_observed_coverage` holds for #1059).
  * ADMIT when the group floor is at least what PP0 wants to schedule.
  * Otherwise DEFER -- named, bounded, and only for THIS rid; other work in
    the same pass proceeds.
  * When the deferral outlives the length-priced bound
    (`store_read_bound_s`, the same pricing the follower's own wait used),
    PP0 stops deferring and CLAMPS: it publishes the group floor as the told
    prefix through the #1059 channel that already exists
    (`PPAdmissionCongruenceGuard.note_observed_coverage`), so the decision
    names a prefix every rank can actually hold. This is the bulletin's own
    prescription, printed on every under-coverage line since #631: "PP0
    defers its raise to the group floor". It never over-tells, never wedges,
    and never kills the group -- the follower's #968 STOP becomes
    unreachable for this cause because `told <= floor <= every rank's
    coverage`.

BYTE-IDENTICAL WHEN NOTHING WAS FETCHED. `want <= 0` (PP0 has no completed
store span for the rid) admits unconditionally, so an ordinary boot with no
storage hit takes exactly the pre-#1175 path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

#: A follower that has a prefetch registered but not yet terminated reports
#: this instead of a token count. NOT -1-as-a-length: every reader below
#: branches on it before any arithmetic, so it can never be min()'d into a
#: floor. (The #944 lesson: "I looked and found N" and "I am still looking"
#: are different facts and must not share a spelling.)
PENDING = "pending"

#: #1176 (review B3): a follower whose STORE WITNESS contradicts its own retract
#: stamp reports THIS instead of a token count. It is not a length either, and
#: every reader branches on it before any arithmetic, for the same reason
#: :data:`PENDING` does.
#:
#: WHY IT RIDES THIS CARRIER AND NOT A NEW ONE. The contradiction is exactly the
#: fact this table already exists to move: "my own store prefetch did not
#: deliver what the group is about to admit on". Under #968 a follower holds no
#: admission verdict, so it may not turn its reading into a group STOP by
#: raising -- but it may not silently withhold the seam premise either, because
#: `prefill_blocked_here` gates the WHOLE prefill-batch build and a rank that
#: builds none while its peers do is mismatched collectives
#: (raenge-nie-uneins). So the follower counts the candidate, states the
#: divergence, and lets the ONE verdict happen where it belongs: PP0 raises on
#: the report, loudly, once, naming the peer.
CONTRADICTION = "contradiction"


def peers_reporting_contradiction(
    table: Mapping[Tuple[str, int], object],
    rid: str,
    pp_size: int,
    own_rank: int = 0,
) -> Tuple[int, ...]:
    """#1176 (review B3): PURE. Which peers reported a store-witness
    contradiction for ``rid``.

    Read BEFORE the ``want <= 0`` early-out of the admission gate: a follower
    can contradict on a rid for which PP0 has no completed store span of its
    own, and that combination is precisely the divergence the report exists to
    surface. Empty tuple on any world without peers.
    """
    peers = _peer_ranks(int(pp_size), int(own_rank))
    if not peers:
        return ()
    return tuple(
        peer
        for peer in peers
        if table.get((str(rid), int(peer))) == CONTRADICTION
    )


def group_completion_enabled() -> bool:
    """#1175 KILL SWITCH, not an opt-in.

    The gate ships ON: the behaviour it replaces is a measured group STOP,
    so the dangerous direction is having it off. ``SGLANG_PP_GROUP_
    COMPLETION=0`` restores the pre-#1175 admission exactly (PP0 admits on
    its own completion alone), which is what an A/B on metal needs.
    """
    import os

    return os.environ.get("SGLANG_PP_GROUP_COMPLETION", "1") != "0"


@dataclass(frozen=True)
class GroupCompletionVerdict:
    """PP0's reading of the group's store-read coverage for ONE rid."""

    admit: bool
    """True when the decision may name `want` (or the clamped floor)."""

    floor: Optional[int]
    """MIN over the peers that reported a NUMBER; None when none did."""

    clamp_to: Optional[int]
    """When set, PP0 must publish THIS as the told prefix instead of `want`
    -- the bound expired and the group floor is the honest number."""

    missing: Tuple[int, ...]
    """Peer pp_ranks that reported nothing at all this lap."""

    pending: Tuple[int, ...]
    """Peer pp_ranks whose prefetch is registered and still running."""

    short: Tuple[Tuple[int, int], ...]
    """(pp_rank, completed) for peers that reported a number below `want`."""

    reason: str
    """Machine-stable term for the log line and the skip census."""

    reports: Tuple[Tuple[int, str], ...] = ()
    """EVERY peer's reading, in rank order, including the silent ones
    (`absent`) and the running ones (`pending`). #1175/E3: a follower that
    says nothing must be VISIBLE as saying nothing -- the specimen's PP2
    printed not one line about rid 0c34259f, and this tuple is what makes
    that a printed fact at the decider instead of an absence nobody sees."""


def _peer_ranks(pp_size: int, own_rank: int) -> Tuple[int, ...]:
    return tuple(r for r in range(max(1, int(pp_size))) if r != int(own_rank))


def group_completion_verdict(
    table: Mapping[Tuple[str, int], object],
    rid: str,
    want: int,
    pp_size: int,
    own_rank: int = 0,
    deadline_expired: bool = False,
) -> GroupCompletionVerdict:
    """PURE. No I/O, no collectives, no clock -- the caller owns the clock.

    ``table`` is PP0's absorbed ring table: ``(rid, pp_rank) -> completed``
    where ``completed`` is an int or :data:`PENDING`.
    """
    want = int(want)
    peers = _peer_ranks(pp_size, own_rank)
    if want <= 0 or not peers:
        # Nothing was fetched (or there is no group): the pre-#1175 path.
        return GroupCompletionVerdict(
            admit=True,
            floor=None,
            clamp_to=None,
            missing=(),
            pending=(),
            short=(),
            reason="no_store_span" if want <= 0 else "no_peers",
        )

    missing = []
    pending = []
    short = []
    reports = []
    floor: Optional[int] = None
    for peer in peers:
        entry = table.get((str(rid), int(peer)))
        if entry is None:
            missing.append(peer)
            reports.append((peer, "absent"))
            continue
        if entry == PENDING:
            pending.append(peer)
            reports.append((peer, "pending"))
            continue
        if entry == CONTRADICTION:
            # #1176 (review B3): NOT A NUMBER, and never min()'d into a floor.
            # The caller raises on this before it ever reaches the verdict
            # (`peers_reporting_contradiction`, read before the `want <= 0`
            # early-out); this branch exists so that a kill-switched or
            # stand-in caller degrades to "this peer produced no usable
            # reading" instead of dying in `int("contradiction")`.
            missing.append(peer)
            reports.append((peer, CONTRADICTION))
            continue
        value = int(entry)  # type: ignore[arg-type]
        floor = value if floor is None else min(floor, value)
        reports.append((peer, str(value)))
        if value < want:
            short.append((peer, value))
    reports_t = tuple(reports)

    complete = not missing and not pending
    if complete and floor is not None and floor >= want:
        return GroupCompletionVerdict(
            admit=True,
            floor=floor,
            clamp_to=None,
            missing=(),
            pending=(),
            short=(),
            reason="group_floor_covers",
            reports=reports_t,
        )

    if deadline_expired:
        # THE BOUND IS SPENT. Admitting at `want` here is exactly the
        # divergence that killed weg1b5; refusing for ever is a wedge. The
        # third option is the one the #631 bulletin has been printing as the
        # remedy all along: tell the group floor. `floor is None` means not
        # one peer produced a number, so the only honest floor is 0.
        return GroupCompletionVerdict(
            admit=True,
            floor=floor,
            clamp_to=int(floor) if floor is not None else 0,
            missing=tuple(missing),
            pending=tuple(pending),
            short=tuple(short),
            reason="bound_expired_clamped_to_group_floor",
            reports=reports_t,
        )

    if pending and not missing:
        reason = "peer_prefetch_pending"
    elif missing and not pending:
        reason = "peer_report_absent"
    elif missing or pending:
        reason = "peer_report_absent_and_pending"
    else:
        reason = "peer_coverage_short"
    return GroupCompletionVerdict(
        admit=False,
        floor=floor,
        clamp_to=None,
        missing=tuple(missing),
        pending=tuple(pending),
        short=tuple(short),
        reason=reason,
        reports=reports_t,
    )


def format_group_fact(
    rid: str,
    want: int,
    verdict: GroupCompletionVerdict,
    own_rank: int = 0,
) -> str:
    """The GROUP FACT line: every rank's reading named, including silence."""
    parts = [f"r{own_rank}={want}"]
    for peer, text in verdict.reports:
        parts.append(f"r{peer}={text}")
    floor_txt = "none" if verdict.floor is None else str(verdict.floor)
    clamp_txt = "-" if verdict.clamp_to is None else str(verdict.clamp_to)
    return (
        f"rid={rid[:8]} want={want} floor={floor_txt} clamp={clamp_txt} "
        f"reports=[{' '.join(parts)}] reason={verdict.reason}"
    )
