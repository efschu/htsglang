"""#1268 fix 1c: the group idle vote, as a control object that rides the ring.

WHAT THIS REPLACES, AND WHY THE TWO PREVIOUS SHAPES DIED ON METAL.

Fix 1 took a ``torch.distributed.all_reduce`` inside the ``/flush_cache`` RPC
handler on PP0.  Boot weg2sb2 deadlocked on the second flip; fix 1b committed
PP0's owed async request-chain forward before joining (#631 clause (ii)) and
boot weg2sb3 deadlocked IDENTICALLY -- only line numbers moved.  The py-spy
topology of both (``BOOT_weg2sb3_killer_context.txt``, six stacks) names the
reason and it is not the one fix 1b assumed::

    PP0        bounded_wait <- group_idle_verdict <- flush_cache
    PP1/PP2    _join <- advance <- receive <- recv_object
               <- _recv_tensor_dict_metadata <- recv_tensor_dict
               <- recv_typed_tensor_dict <- _pp_recv_typed_dict
               <- _pp_recv_dict_from_prev_stage

The followers are NOT parked in the request-chain receive waiting for the
forward fix 1b committed.  They are parked in the HIDDEN-STATES / proxy
tensor-dict receive, one channel over, waiting for the proxy send PP0 has not
issued yet -- because PP0 is inside the reduce, and the proxy send comes LATER
in its own pass.  That is a cycle between adjacent stages, and committing the
request forward cannot open it: the followers were never waiting on that
message.  #631's own docstring already named this as ``variant B`` --
"deadlocks against the HIDDEN-STATES exchange, because a peer need not be in
the chain recv at all".

SO PP0 MUST NOT BLOCK ON ANYTHING while it holds the RPC.  It stamps, it
returns "pending", it finishes its pass (which releases the proxy the
followers are waiting for), and the answer arrives on a later poll.  The front
already polls ``/flush_cache`` every 0.5 s to ``QUIESCE_DEADLINE_S``
(``weg2/front.py``, ``Front.quiesce``), so "pending" needs no front change --
it is simply a non-200 that keeps the poll going, and the LAST body is what
the W3 refusal prints.

THE CARRIER, and why it is not a second bookkeeping.  A fact that must reach
PP0 from its followers already has an established form in this tree: the
#797 / #968 / #1175 "return trip", where each relay unions its own reading
into the payload the last rank sends home and PP0 absorbs it
(``pp_output_payload_with_return_trip``, ``pp_note_prefetch_completion``).
This module is that pattern, per slot, with one difference forced by the
state it has to observe: the #1175 carrier rides the last rank's OUTPUT
message, which is gated on ``_pp_output_exchange_due(mbs[slot])`` and is
therefore SILENT ON AN IDLE PASS -- and an idle pass is exactly when an idle
vote is needed.  The downward arc has no such problem: the request-chain
forward at ``scheduler_pp_mixin.py`` (``if not self.pp_group.is_last_rank:``)
is unconditional per pass, armed or not (#969 §W3), so the vote reaches PP1
and PP2 on the arc that is already running.  Only the last hop home is
missing, and this module adds exactly that hop and nothing else.

THE HOME HOP CARRIES ITS OWN TAG, and that is load-bearing.  The
(last rank -> PP0) pair on tag 0 is ALREADY IN USE: ``pp_typed_channel``
demultiplexes ``proxy`` and ``output`` messages IN BAND, by a ``__msg_type__``
entry, on the default tag.  A standing receive posted on tag 0 would sooner or
later swallow an output tensor-dict and misframe the stream -- the precise
hazard ``pp_object_recv``'s module docstring exists to prevent.  ``WEG2_VOTE_TAG``
gives the vote its own stream, on which nothing else ever travels.

AND THE HOME RECEIVE IS NEVER JOINED.  ``ObjectRecvFrame.advance`` with a small
step budget returns False with the receive STILL POSTED and RESUMABLE, so PP0
absorbs the lap when it lands and never blocks a pass on it.  The two obvious
alternatives are both measured dead on this build and must not be
reintroduced: ``Work.is_completed()`` never reports True even seconds after
the payload has landed ("corpse F",
``test_pp_chain_receiver.test_measured_gloo_does_not_progress_a_posted_irecv_by_polling``),
and ``Work.wait(timeout=...)`` fires on time but CLOSES THE GLOO PAIR while
doing it, taking the group down on both sides
(``pp_object_recv``'s own docstring, hermetic 2-process measurement
2026-08-23).  #1029b's "the home hop must not block: a lap nested in a round
is a cycle" is the same lesson from the other end.

WHAT THIS MODULE IS NOT.  It decides nothing about what a rank DOES.  Each
rank keeps flushing its own pools on its own ``is_fully_idle()`` exactly as
upstream does; that action is rank-local and always was.  What #1268 is about
is the fact that LEAVES the group toward the front, on which the front then
commands ``sleep(P, kv_cache)``.  That one fact is now reduced over every
rank before it is answered, and it is answered in ONE place (PP0).

A LANDED LAP ANSWERS ONLY FOR THE STATE IT WITNESSED (fnFL2 H77).  The lap is a
snapshot: each slot says "this rank was idle when the object passed".  It used
to be kept in ``_weg2_vote_verdict`` with no round binding and no expiry, and
the first reader took it -- whoever that was, whenever that was.  On boots
fnFL2x166/x169 that reader was the NEXT quiesce on every P flip: the sleep
leg's own flush (``release_memory_occupation`` -> ``flush_cache``) wanted a
lap, the lap was stamped with P already dormant, came home 3/3 idle and
waited; the first /flush_cache poll of the next quiesce read it and answered
200 in 36 ms (x169: stamped 22:00:34, read 22:04:26, while PP1/PP2 held
``hicache_backup(5)`` and PP2 was still in the last prefill pass).  PP0 now
keeps a :class:`Weg2LapWitness` per stamp -- never sent -- and reads a landed
lap only while (i) it is the lap of PP0's latest stamp, (ii) PP0 neither slept
nor was busy since that stamp (work enters the group through PP0, so a busy
entrypoint means work may have entered behind the lap), (iii) it is younger
than ``SGLANG_WEG2_IDLE_VOTE_TTL_S``, and (iv) PP0 itself is idle at the read.
Anything else drops the lap by name (``#1268 IDLE-ROUND stale ... dropped``)
and wants a new one; the front keeps polling as it does for ``pending``.  All
of it is PP0-local bookkeeping: no collective, nothing on the wire, and the
followers' side of the lap is unchanged.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: The vote's own point-to-point stream.  NOT tag 0: that pair already carries
#: the ``pp_typed_channel`` proxy/output messages, which are demultiplexed in
#: band rather than by tag, so a standing frame there would misframe them.
WEG2_VOTE_TAG = 1268

#: One bounded step of the home receive per pass.  Small on purpose: this is a
#: poll that must never become a wait (see the module docstring on why a
#: blocking home hop closes a cycle).
VOTE_HOME_STEP_BUDGET_S = 0.002


@dataclass
class Weg2IdleVoteReq:
    """One idle vote, travelling PP0 -> PP1 -> ... -> last -> PP0.

    ``slots`` is append-only per rank and is the whole of the vote's payload:
    a rank that has not attached is INDISTINGUISHABLE from a rank that never
    saw the object, and both are reported as missing rather than assumed idle.
    That asymmetry is deliberate -- it is the sb1 defect (an absent peer read
    as a consenting one) expressed as a data-structure invariant.
    """

    epoch: int
    origin: int
    world: int
    #: (rank, idle as 0/1, this rank's blocker clause names)
    slots: List[Tuple[int, int, str]] = field(default_factory=list)

    def has_slot(self, rank: int) -> bool:
        return any(int(r) == int(rank) for r, _i, _b in self.slots)


@dataclass(frozen=True)
class Weg2VoteTally:
    n_idle: int
    n_present: int
    world: int
    mask: int
    blocking_ranks: Tuple[int, ...]
    missing_ranks: Tuple[int, ...]

    @property
    def complete(self) -> bool:
        return self.n_present == self.world

    @property
    def idle(self) -> bool:
        """Idle only when EVERY rank is present AND every present rank agreed.

        Never ``n_idle == n_present``: that is "all present voted idle"
        presented as a statement about the whole group, which is #1268 itself
        in a new costume.
        """
        return self.n_present == self.world and self.n_idle == self.world


def attach_slot(
    vote: Weg2IdleVoteReq, rank: int, idle: bool, blockers: str
) -> bool:
    """Attach THIS rank's own answer.  True iff it was newly attached.

    Rank-local by construction: no collective, no peer state read.  A second
    attach by the same rank is refused rather than overwritten -- a rank sees
    the object once per lap, so a duplicate means the lap wrapped further than
    it should have and silently replacing the earlier answer would hide it.
    """
    if vote.has_slot(rank):
        return False
    vote.slots.append((int(rank), 1 if idle else 0, str(blockers)))
    return True


def tally(vote: Weg2IdleVoteReq) -> Weg2VoteTally:
    """SUM-style counting over the attached slots.

    SUM rather than MIN because MIN cannot COUNT (three ranks each
    contributing -1 reduce to -1, not to -3), and participation is the fact
    this whole shape exists to carry.  The mask's bits are disjoint by
    construction -- one per rank -- so its set bits name every blocking rank
    and its lowest set bit the first.
    """
    world = int(vote.world)
    n_idle = 0
    mask = 0
    present: set = set()
    for rank, idle, _blockers in vote.slots:
        rank = int(rank)
        present.add(rank)
        if int(idle):
            n_idle += 1
        else:
            mask |= 1 << max(rank, 0)
    blocking = tuple(r for r in range(world) if mask & (1 << r))
    missing = tuple(r for r in range(world) if r not in present)
    return Weg2VoteTally(
        n_idle=n_idle,
        n_present=len(present),
        world=world,
        mask=mask,
        blocking_ranks=blocking,
        missing_ranks=missing,
    )


def blockers_of(vote: Weg2IdleVoteReq, rank: int) -> str:
    for r, _i, blockers in vote.slots:
        if int(r) == int(rank):
            return blockers
    return "none"


def log_verdict(vote: Weg2IdleVoteReq, rank: int) -> str:
    """The one line every rank prints for its own slot, same epoch.

    #1268 fix 1b (3) kept: an instrument that deadlocks silently is not an
    instrument, so a rank that reached the vote is visible as a LINE as well
    as a participation count.  ``epoch`` is now a REAL NUMBER: the previous
    shape read ``self.weg2_flip_epoch``, an attribute nothing in the tree ever
    assigned, so every verdict line on every boot printed the literal
    ``epoch=?``.  The epoch is minted by PP0 when it stamps the object and
    travels ON the object, so a follower's line and PP0's line for one lap
    always carry the same number.
    """
    t = tally(vote)
    line = (
        "WEG2-P-IDLE-VERDICT epoch=%d idle=%s blocking_rank=%s blockers=[%s] "
        "participation=%d/%d"
    ) % (
        int(vote.epoch),
        bool(t.idle),
        str(t.blocking_ranks[0]) if t.blocking_ranks else "none",
        blockers_of(vote, rank),
        t.n_present,
        t.world,
    )
    logger.info(line)
    return line


def refusal_detail(vote: Weg2IdleVoteReq, t: Weg2VoteTally, note: str) -> str:
    """A refusal NAMES THE MISSING RANKS.  Never a bare TimeoutError.

    sb3 stopped the front on the literal string ``'TimeoutError: '`` -- gloo
    timeouts carry an empty message, so the only thing the operator learned
    from a dead group was the exception's class name.  Everything knowable
    from this side is stated, and what is not knowable is named as such.
    """
    return (
        f"{note} epoch={int(vote.epoch)} participation={t.n_present}/{t.world}; "
        f"ranks whose slot is MISSING from this lap: "
        f"{list(t.missing_ranks) or 'none'}; ranks that voted NOT idle: "
        f"{list(t.blocking_ranks) or 'none'}. A missing slot is not a consenting "
        f"rank -- it is a rank this lap has no statement from, and answering "
        f"for it is the #1268 defect (boot weg2sb1)."
    )


# --------------------------------------------------------------------------
# fnFL2 H77: how long a landed lap may answer (PP0-local, never sent)
# --------------------------------------------------------------------------


@dataclass
class Weg2LapWitness:
    """PP0's own record of what ONE lap witnesses.

    Minted with the stamp and never put on the wire: the lap carries the
    slots, this carries the conditions under which those slots still describe
    the group.  ``taint`` is the first reason the witnessed state stopped
    holding, as PP0 saw it; empty while it holds.
    """

    epoch: int
    stamped_at: float
    taint: str = ""

    def spoil(self, reason: str) -> bool:
        """Record ``reason`` unless an earlier one is already recorded."""
        if reason and not self.taint:
            self.taint = str(reason)
            return True
        return False


def lap_clock() -> float:
    """The one clock every stamp and every read uses (CLOCK_MONOTONIC)."""
    return time.monotonic()


def entrypoint_taint(*, dormant: bool, idle: bool) -> str:
    """Why PP0's own state since the stamp voids the lap ('' = it does not).

    Work enters the group only through PP0, so a PP0 that was busy after the
    stamp may have put work on the chain BEHIND the lap -- the followers'
    slots were taken ahead of it.  A sleep voids every slot taken before it.
    """
    if dormant:
        return (
            "the entrypoint went to sleep with the lap on the ring (a sleep "
            "voids every slot taken before it)"
        )
    if not idle:
        return (
            "the entrypoint was busy after the stamp (work may have entered "
            "the group behind the lap)"
        )
    return ""


def expiry_reason(witness: Weg2LapWitness, *, now: float, ttl_s: float) -> str:
    age = float(now) - float(witness.stamped_at)
    if ttl_s > 0 and age > ttl_s:
        return f"expired (age={age:.3f}s > ttl={float(ttl_s):.3f}s)"
    return ""


def stale_reason(
    vote: Weg2IdleVoteReq,
    witness: Optional[Weg2LapWitness],
    *,
    latest_epoch: int,
    now: float,
    ttl_s: float,
    own_idle: bool,
    own_blockers: str,
) -> str:
    """'' iff the landed lap may answer for the group NOW; else why not.

    Round id first (a lap PP0 has no current stamp record of answers for
    nothing), then what PP0 saw since the stamp, then the expiry, then the
    entrypoint's own state at the read -- the lap's PP0 slot is from the
    stamp, and the read is later.
    """
    if witness is None:
        return f"no stamp record for the lap of epoch={int(vote.epoch)}"
    if int(witness.epoch) != int(vote.epoch) or int(vote.epoch) != int(latest_epoch):
        return (
            f"round id mismatch (lap epoch={int(vote.epoch)}, stamp record "
            f"epoch={int(witness.epoch)}, latest stamp epoch={int(latest_epoch)})"
        )
    if witness.taint:
        return witness.taint
    expired = expiry_reason(witness, now=now, ttl_s=ttl_s)
    if expired:
        return expired
    if not own_idle:
        return f"the entrypoint is not idle at the read (blockers=[{own_blockers}])"
    return ""


def _age_ms(witness: Optional[Weg2LapWitness], now: float) -> float:
    if witness is None:
        return -1.0
    return (float(now) - float(witness.stamped_at)) * 1000.0


def log_stale_drop(
    vote: Weg2IdleVoteReq,
    witness: Optional[Weg2LapWitness],
    why: str,
    *,
    where: str,
    now: float,
) -> str:
    """The metal marker: a landed lap was dropped, never read as the verdict."""
    t = tally(vote)
    line = (
        "#1268 IDLE-ROUND stale epoch=%d age_ms=%.0f where=%s lap_idle=%s "
        "participation=%d/%d blocking=%s reason=%s -- dropped, never read as "
        "the group's verdict; the next poll wants a new lap"
    ) % (
        int(vote.epoch),
        _age_ms(witness, now),
        where,
        bool(t.idle),
        t.n_present,
        t.world,
        list(t.blocking_ranks) or "none",
        why,
    )
    logger.info(line)
    return line


def log_fresh_read(
    vote: Weg2IdleVoteReq, witness: Optional[Weg2LapWitness], *, now: float
) -> str:
    t = tally(vote)
    line = (
        "#1268 IDLE-ROUND fresh epoch=%d age_ms=%.0f participation=%d/%d "
        "entrypoint idle -- read as the group's verdict"
    ) % (int(vote.epoch), _age_ms(witness, now), t.n_present, t.world)
    logger.info(line)
    return line


def stale_detail(vote: Weg2IdleVoteReq, why: str, own_blockers: str) -> str:
    """The refusal body the front's poll gets for a dropped lap."""
    t = tally(vote)
    return (
        f"GROUP VERDICT STALE: the landed lap of epoch={int(vote.epoch)} "
        f"(idle={bool(t.idle)}, participation={t.n_present}/{t.world}, "
        f"blocking={list(t.blocking_ranks) or 'none'}) was dropped: {why}. A "
        f"lap answers only for the state it witnessed; a new one is wanted and "
        f"the front keeps polling. This rank 0 blockers=[{own_blockers}]."
    )


# --------------------------------------------------------------------------
# the wire: one dedicated stream, last rank -> PP0
# --------------------------------------------------------------------------


def home_ranks(pp_rank: int, pp_size: int, tp_size: int, dp_offset: int) -> Tuple[int, int]:
    """(src_global, dst_global) of the home hop, in the ring's own arithmetic.

    Identical to what ``_pp_send_pyobj_to_next_stage`` computes -- destination
    ``((pp_rank + 1) % pp_size) * tp_size + dp_offset`` -- evaluated at the
    LAST rank, where the modulo wraps to PP0.  Stated here so the sender and
    the receiver derive the pair from ONE expression rather than two that
    agree by inspection (the ``_pp_output_exchange_due`` lesson: send and
    receive must be the same question asked of the same thing).
    """
    last = (int(pp_size) - 1) * int(tp_size) + int(dp_offset)
    home = 0 * int(tp_size) + int(dp_offset)
    return last, home


def send_home(vote: Weg2IdleVoteReq, group: Any, src_global: int, dst_global: int) -> List[Any]:
    """Post the completed lap back to PP0, asynchronously.

    Mirrors ``point_to_point_pyobj``'s size-then-payload protocol byte for
    byte -- a ``torch.long`` size then a ``uint8`` payload -- because the
    receiver is ``ObjectRecvFrame``, which mirrors ``recv_object``'s half of
    that same protocol.  It differs in exactly one respect, the tag, and
    ``point_to_point_pyobj`` has no tag parameter to pass.
    """
    import pickle

    import numpy as np
    import torch
    import torch.distributed as dist

    from sglang.srt.distributed.parallel_state import P2PWork

    payload = pickle.dumps(vote)
    size_t = torch.tensor([len(payload)], dtype=torch.long)
    data_t = torch.ByteTensor(np.frombuffer(payload, dtype=np.uint8).copy())
    # P2PWork carries the tensor beside the handle for exactly one reason: the
    # buffer must outlive the isend, and the work list is what keeps it
    # referenced until the caller commits it.
    return [
        P2PWork(dist.isend(size_t, dst_global, group=group, tag=WEG2_VOTE_TAG), size_t),
        P2PWork(dist.isend(data_t, dst_global, group=group, tag=WEG2_VOTE_TAG), data_t),
    ]
