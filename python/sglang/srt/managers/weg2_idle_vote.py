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
"""

from __future__ import annotations

import logging
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
