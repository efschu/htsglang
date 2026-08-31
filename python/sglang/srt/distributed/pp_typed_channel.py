# Copyright 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0
"""#753: one demultiplexer for the PP group's tensor-dict channel.

WHY THIS EXISTS. Before a gapped layer set there were two kinds of tensor dict
on the PP group -- ``proxy`` (stage-boundary hidden states) and ``output``
(sampled tokens, ringed from the last rank) -- and exactly one consumer loop,
``Scheduler._pp_recv_typed_dict``, which owned the inbox that holds a message
of the wrong kind until its real consumer asks. The crossing wire adds a THIRD
kind and, critically, a SECOND consumer. Two consumers with two private inboxes
on one channel is not a demultiplexer; it is a race with a stash on each side.

THE COLLISION IS CERTAIN, NOT HYPOTHETICAL. ``recv_tensor_dict(src=None)``
resolves to ``(rank_in_group - 1) % world_size`` -- the previous rank in the
ring -- so on the #735 gapped cut PP0's ``output`` receive and PP0's crossing
receive from PP2 are the SAME directed p2p pair. A raw crossing receive would
take the output dict off the wire and hand a token tensor to a decoder layer as
hidden states: silent wrong output, which is the #753 defect wearing a
different hat.

THE KEY IS ``(src, kind)`` AND NOT ``kind``. Keying by kind alone is safe only
while every kind arrives from one source. PP0 receives crossings from BOTH PP1
(after layer 31) and PP2 (after layer 35), so a crossing stashed under "the
crossing queue" could be handed to a receive that named the other peer, pairing
one stage's activations with another's schedule slot. The source is part of the
message's identity, so it is part of the key.

ONE INBOX PER GROUP, created lazily on the group itself. Not a module global:
a process may hold more than one coordinator, and a global would merge their
channels. Not a constructor field either -- that would force every construction
site to be found and stamped, which is exactly the "never-stamped group
attribute" failure this module's own lineage already paid for.
"""

from __future__ import annotations

import collections
from typing import Any, Callable, Deque, Dict, Optional, Tuple

__all__ = [
    "MSG_TYPE_KEY",
    "CROSSING_KIND",
    "resolve_src",
    "typed_inbox",
    "stash_typed",
    "take_typed",
    "recv_typed_tensor_dict",
    "send_typed_tensor_dict",
]

#: The entry that carries a message's kind. A non-tensor entry travelling in
#: this dict is established practice on this channel, not a new risk.
MSG_TYPE_KEY = "__msg_type__"

#: The crossing wire's kind. Distinct from "proxy" and "output" so that a
#: mid-loop activation can never be mistaken for a stage-boundary one.
CROSSING_KIND = "crossing"

_INBOX_ATTR = "_pp_typed_dict_inbox"


def resolve_src(group, src: Optional[int]) -> int:
    """The rank a receive will actually match.

    ``None`` means "the previous rank in the ring", which is what
    ``recv_tensor_dict`` itself does. Resolving it here means a message stashed
    by a ``src=None`` receive and a message taken by an explicit ``src=N``
    receive land in the same queue when they are, in fact, the same pair.
    """
    if src is not None:
        return int(src)
    return (int(group.rank_in_group) - 1) % int(group.world_size)


def typed_inbox(group) -> Dict[Tuple[int, str], Deque[Any]]:
    """The group's one inbox, created on first use."""
    inbox = getattr(group, _INBOX_ATTR, None)
    if inbox is None:
        inbox = collections.defaultdict(collections.deque)
        setattr(group, _INBOX_ATTR, inbox)
    return inbox


def stash_typed(group, src: Optional[int], kind: str, message: Any) -> None:
    """Hold a message for the consumer that actually asked for it.

    Never a discard. A message of the wrong kind has still LEFT THE WIRE, and
    some peer's blocking send is already satisfied by that fact; destroying it
    turns a demultiplex into a lost payload.
    """
    typed_inbox(group)[(resolve_src(group, src), str(kind))].append(message)


def take_typed(
    group,
    src: Optional[int],
    kind: str,
    accept: Optional[Callable[[Any], bool]] = None,
    on_reject: Optional[Callable[[Any], None]] = None,
) -> Optional[Any]:
    """A previously stashed message for this exact ``(src, kind)``, or None.

    #1057: ``accept`` IS THE ONLY PLACE FIFO IS NOT ALREADY DOING THIS JOB.
    A message that comes off the wire arrives in send order, so the wire path
    cannot hand a receiver an OLDER pass than one it has already consumed.
    The inbox can, and it is served BEFORE the wire (that is the point of the
    stash), so this queue is the one door an out-of-order pass can walk
    through. That is why the identity filter lives here and not on the wire.

    When ``accept`` rejects the head, the entry is DROPPED, not put back:
    putting it back would make this a busy loop over a message nobody will
    ever want, and leaving it in place is exactly the outliving #800 exists
    to prevent. ``on_reject`` is called for each dropped entry so the loss is
    counted and named rather than silent -- the #800 rule ("a retired message
    is a LOST payload if any consumer did in fact owe it") applies verbatim.

    The loop then continues to the NEXT stashed entry and, when the queue
    empties, returns None so the caller falls through to the wire. There is no
    branch here that refuses without a way onward.
    """
    queue = typed_inbox(group).get((resolve_src(group, src), str(kind)))
    while queue:
        message = queue.popleft()
        if accept is None or accept(message):
            return message
        if on_reject is not None:
            on_reject(message)
    return None


def send_typed_tensor_dict(
    group,
    tensor_dict: Dict[str, Any],
    dst: Optional[int],
    kind: str,
    async_send: bool = False,
    all_gather_group=None,
):
    """Send with its kind attached, so the far side can demultiplex it."""
    tensor_dict[MSG_TYPE_KEY] = str(kind)
    return group.send_tensor_dict(
        tensor_dict,
        dst,
        all_gather_group=all_gather_group,
        async_send=async_send,
    )


def recv_typed_tensor_dict(
    group,
    expected_kind: str,
    src: Optional[int] = None,
    all_gather_group=None,
    on_message: Optional[Callable[[Dict[str, Any]], None]] = None,
    accept: Optional[Callable[[Dict[str, Any]], bool]] = None,
    on_reject: Optional[Callable[[Dict[str, Any]], None]] = None,
):
    """Receive the next message of ``expected_kind`` from ``src``.

    Anything else that arrives on that pair is stashed under its own
    ``(src, kind)`` and the receive continues. ``on_message`` is called for
    every message that comes OFF THE WIRE -- before the demultiplex, never for
    one served from the inbox -- because the flip's quiescence counters and the
    boundary stats are counting wire traffic, and a stashed message has already
    crossed it.

    #1057: ``accept`` GUARDS THE INBOX ONLY, AND THAT IS DELIBERATE, NOT AN
    OVERSIGHT. The wire delivers in send order, so a message taken off it can
    never be an older pass than one already consumed from the same sender --
    the identity filter would be dead weight there. It would also be the
    boot-killing direction: rejecting a wire message means dropping it and
    blocking for the next, and if the predicate is ever wrong that is a wedge
    with no way onward (#995 v1: 175 refusals on one rid and a dead window).
    The inbox is the only source that can hand back an out-of-order pass, it
    is bounded, and draining it always terminates in "fall through to the
    wire". So the guard sits exactly where the hazard is and nowhere else.
    """
    served = take_typed(group, src, expected_kind, accept=accept, on_reject=on_reject)
    if served is not None:
        return served

    while True:
        message = group.recv_tensor_dict(src, all_gather_group=all_gather_group)
        if on_message is not None:
            on_message(message)
        kind = (
            message.get(MSG_TYPE_KEY, "default")
            if isinstance(message, dict)
            else "default"
        )
        if kind == expected_kind:
            return message
        stash_typed(group, src, kind, message)
