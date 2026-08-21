# Copyright 2023-2024 SGLang Team
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
"""#631 defect G: monotone per-channel message counters in /dev/shm.

WHY THIS EXISTS -- THE ONE THING THE TRANSPORT CANNOT TELL US
-------------------------------------------------------------
The armed service loop has to answer exactly one question, over and over:
*is there a message on this stream for me to take?* On this build the
transport cannot answer it. Both halves of the obvious answer are measured
dead (corpse F, and the transport facts in ``phase_flip_presence``):

  * a posted ``irecv`` never completes by polling ``is_completed()``, so
    "peek and give up" absorbs nothing;
  * an ``isend`` never completes by polling either, so a sender cannot
    learn that its message was taken.

Only ``wait()`` progresses a transfer here, and ``wait()`` is blocking --
precisely what an armed rank may not do speculatively. So the readiness
signal has to come from OUTSIDE the transport, on a channel that is
POLLABLE without any peer's cooperation. That is this module: the same
``/dev/shm`` discipline the presence gate already uses (#615 precedent,
single-node by V1 scope), carrying counts instead of flags.

With a count in hand the blocking call becomes safe and BOUNDED: a
receiver calls ``recv()`` only once the sender's published count exceeds
its own consumed count, so a message is provably in flight and the wait is
bounded by TRANSFER TIME, not by peer scheduling. That turns the one
transport behaviour with positive evidence -- the recv side's ``wait()``
drives the transfer, which is how arms propagated across boots 14-18 --
into a mechanism instead of an accident.

THE ORDERING CONSTRAINT, and it is the whole safety argument
------------------------------------------------------------
    PUBLISH THE COUNTER STRICTLY AFTER THE ISEND IS POSTED.

Then the only possible skew between the counter and the wire is
COUNTER-LAGS-SEND: a real message may be seen late, and is consumed on the
next poll. That direction is harmless.

Publishing FIRST inverts the skew into SEND-LAGS-COUNTER, and a receiver
that believes a phantom message calls a blocking ``recv()`` for something
nobody posted -- an unbounded block, i.e. the wedge class this entire
feature exists to remove. There is no third option and no "mostly fine"
here: the ordering is the design.
Pinned by ``test_can_fail_publishing_before_the_post_wedges_the_receiver``.

THE QUESTION THAT RULE CANNOT ANSWER, AND THE SECOND COUNTER (#789)
------------------------------------------------------------------
Because ``sent`` means "on the wire", it says nothing during the window in
which the sender is INSIDE the send call. That window is empty for an
isend -- until the send is a RENDEZVOUS, which the lazy creation of a
torch NCCL 2-rank p2p communicator is: it does not return until the peer
enters the matching receive. A receiver gating on ``sent`` alone then
becomes one arc of a cycle it is itself waiting to break. Two boots died
there (2026-08-21). ``bump_attempted`` / ``attempted`` answer the other
question -- "has the upstream irrevocably entered a send for me?" -- and
are published BEFORE the post for exactly that reason. The rule above is
unchanged and still governs ``sent``; see ``bump_attempted`` for why the
second counter is not the phantom-message hazard wearing a new name, and
why only the gate that would otherwise RAISE may read it.

WHY A REWRITTEN FILE AND NOT A WRITE-ONCE MARKER PER MESSAGE
------------------------------------------------------------
Presence markers are write-once because they answer a yes/no question that
is asked once per (epoch, round). A counter answers "how many so far",
which is asked on a hot path for the lifetime of the boot; one file per
message would grow without bound and turn every poll into a directory
scan. So each counter is ONE file per (channel, kind, rank), rewritten by
its single writer via a temp file and an atomic ``os.replace``.

That keeps every property the flags were built for:
  * SINGLE WRITER -- only the owning rank ever writes its own counter;
  * ATOMIC -- a reader observes the old value or the new one, never a torn
    one, because ``os.replace`` swaps a fully written inode;
  * MONOTONE -- the writer only ever increments, so a reader can never see
    a count go backwards, which is what makes a stale read safe rather
    than merely unlikely.
A reader that loses the race simply reads a slightly old value and polls
again -- counter-lags-send, the safe direction, once more.

CHANNELS
--------
A "channel" here is one directed WIRE between adjacent PP ranks, named by
what frames it, and the counter counts LOGICAL messages on it:

  ``req``   the request chain, ``point_to_point_pyobj`` (a size message
            then a payload message -- one logical message, counted once).
            Sender k, receiver k+1.
  ``dict``  the tensor-dict stream, ``send_tensor_dict`` /
            ``recv_tensor_dict``. Sender k, receiver (k+1) % pp_size.
            NOTE: ``proxy`` (hidden states) and ``output`` messages share
            this ONE wire and are demultiplexed by ``__msg_type__`` after
            the fact, so they must share ONE counter. Counting them
            separately would let a rank believe a wire was empty while a
            message of the other kind sat on it.
"""

from __future__ import annotations

import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)

LOG_PREFIX = "PHASE-FLIP-COUNTERS"

#: The request chain (point_to_point_pyobj), rank k -> k+1.
CHAN_REQ = "req"
#: The tensor-dict wire (proxy AND output share it), rank k -> k+1 mod n.
CHAN_DICT = "dict"
#: NOT A WIRE. The number of pp_loop_size slot iterations a rank has run
#: SINCE IT ARMED -- the instrument for #631 defect Q (the armed window has
#: no pass clock). It rides the counter machinery because that machinery is
#: already a cross-rank readable side channel with the right lifetime, so
#: one rank can print all three ranks' pass counts on one line instead of
#: three log streams having to be correlated by hand.
#:
#: Published ONLY while a flip is armed, and reset at each arm. A boot
#: without the flip, and every unarmed pass of a boot with it, writes
#: nothing at all.
CHAN_PASS = "pass"
#: NOT A WIRE. The MICROBATCH SLOT INDEX this rank is currently on, published
#: while a flip is armed -- the answer to defect Q rather than another
#: measurement of it.
#:
#: WHY THE SLOT INDEX IS THE QUANTITY THAT MATTERS. ``CHAN_PASS`` above
#: counts armed iterations and MEASURED the divergence (spreads of ~10787
#: iterations over a 5 s armed window, 2026-08-09 07:19:23Z). But the pass
#: count is not what the pipeline pairs on: ``mb_id`` is. Two ranks may spin
#: any number of parked iterations and stay correct so long as they RESUME
#: the pass loop on the same slot, because the proxy stamp, the ``mbs``
#: occupancy and the output pairing are all indexed by that slot and by
#: nothing else. So this gauge is what the falling-edge check reads, and
#: agreement on it is the invariant the armed window must preserve.
CHAN_SLOT = "slot"

_SENT = "s"
_CONSUMED = "c"
#: "This rank has ENTERED the send call for one more message on this
#: channel", published BEFORE the post. See ``bump_attempted``: this is a
#: second, differently-timed counter, NOT a relaxation of the ordering rule
#: above, which governs ``_SENT`` and is unchanged.
_ATTEMPTED = "a"


class PhaseFlipCounters:
    """Monotone per-(channel, kind, rank) message counters on ``/dev/shm``.

    Shares the presence gate's directory and instance tag on purpose: they
    are one rendezvous with one lifetime, swept together at boot, and a
    second tag would be a second thing to keep identical across ranks.
    """

    def __init__(
        self,
        n_ranks: int,
        rank: int,
        directory: str,
        instance: str,
    ):
        self.n_ranks = int(n_ranks)
        self.rank = int(rank)
        self.directory = directory
        self.instance = instance
        os.makedirs(self.directory, exist_ok=True)
        #: This rank's own counts. The files are the PUBLISHED view for
        #: peers; these are the authority for this rank, so a failed write
        #: can never make a rank forget what it already did.
        self._local: Dict[str, int] = {}

    # -- naming ---------------------------------------------------------

    def _path(self, chan: str, kind: str, rank: int) -> str:
        return os.path.join(
            self.directory,
            f"{self.instance}.ctr.{chan}.{kind}{int(rank)}",
        )

    # -- writing (single writer: always this rank) ----------------------

    def _publish(self, chan: str, kind: str, value: int) -> None:
        path = self._path(chan, kind, self.rank)
        tmp = f"{path}.tmp.{os.getpid()}"
        try:
            with open(tmp, "w") as fh:
                fh.write(f"{int(value)}\n")
            os.replace(tmp, path)
        except OSError as exc:  # pragma: no cover - disk-full etc.
            logger.error("%s could not publish %s: %s", LOG_PREFIX, path, exc)
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def bump_sent(self, chan: str) -> int:
        """Record one more message POSTED on ``chan`` by this rank.

        MUST be called strictly AFTER the isend is posted -- see the module
        docstring. Every caller is on the far side of the send call, and
        that placement is the safety property, not a style choice.
        """
        key = (chan, _SENT)
        n = self._local.get(key, 0) + 1
        self._local[key] = n
        self._publish(chan, _SENT, n)
        return n

    def bump_attempted(self, chan: str) -> int:
        """Record that this rank has ENTERED the send call for one message.

        PUBLISHED BEFORE THE POST -- the opposite timing to ``bump_sent``,
        and the reason a second counter exists at all instead of moving
        the first one.

        WHY THE ORDERING RULE IN THE MODULE DOCSTRING IS NOT WEAKENED.
        That rule governs ``_SENT`` and still does, unchanged: ``sent``
        means "on the wire", and nothing may publish it early. This
        counter answers a DIFFERENT question -- "has the upstream
        irrevocably committed to sending me a message?" -- which the
        ``sent`` counter structurally cannot answer, because the very
        window in which the answer matters is the window BEFORE the post
        completes.

        THE FAILURE THAT MADE THIS NECESSARY (#789, boots instr7 + instr8,
        2026-08-21, both identical). ``_pp_wait_for_proxy_readiness`` refuses
        to enter the blocking proxy receive until ``sent`` exceeds this
        rank's ``consumed``. Its docstring argued a stuck SENDER is
        "covered identically" to a sender that never intended to send.
        That is true only if the sender's progress is INDEPENDENT of the
        receiver. It is not: the first point-to-point op on a torch NCCL
        process group creates the 2-rank communicator lazily, and that
        creation is a RENDEZVOUS -- ``isend`` does not return until the
        peer enters the matching receive. So on the first real prefill of
        every boot:

            PP0  isend (distributed_c10d.py:2552)
                 send_tensor_dict (parallel_state.py:2384)
                 _pp_send_dict_to_next_stage (scheduler_pp_mixin.py:3196)
            PP1  _pp_wait_for_proxy_readiness, polling counters
            PP2  the same, one hop behind

        PP0 cannot bump ``sent`` until the isend returns; the isend cannot
        return until PP1 enters the receive; PP1 will not enter the
        receive until ``sent`` bumps. A three-arc cycle in which the
        readiness gate is itself one of the arcs. 30 s later the gate
        raised "#789 PROXY READINESS TIMEOUT ... posted 1681, consumed
        1681" and killed the boot -- a message that was NOT absent, it was
        being posted, waiting for the very rank that refused to collect
        it.

        WHY THIS IS NOT THE PHANTOM-MESSAGE HAZARD IN REVERSE. The hazard
        the ordering rule prevents is a receiver blocking for a message
        NOBODY DECIDED TO SEND -- unbounded, because it waits on peer
        scheduling. Here the decision is already made and unconditional:
        this counter is bumped on the line immediately before the send
        call, past every branch that could still skip it, so a receiver
        acting on it waits only for a post that a peer is already inside.
        That is bounded by transfer and rendezvous time, which is exactly
        the bound the module docstring asks for. The one way the post can
        still never happen is the sender RAISING inside it -- in which
        case that rank dies and distributed teardown ends the receiver's
        wait, the same disposal every dead-upstream case already relies
        on.

        Consequently only the gate that would otherwise RAISE reads this
        counter. The drain loops keep reading ``sent``: they must take off
        the wire only what is provably already on it, and are correct to
        stop when nothing is.
        """
        key = (chan, _ATTEMPTED)
        n = self._local.get(key, 0) + 1
        self._local[key] = n
        self._publish(chan, _ATTEMPTED, n)
        return n

    def bump_consumed(self, chan: str) -> int:
        """Record one more message fully TAKEN OFF ``chan`` by this rank.

        Published so the upstream can learn that the message it posted is
        gone from the wire, which is what makes its own blocking commit
        bounded rather than speculative.
        """
        key = (chan, _CONSUMED)
        n = self._local.get(key, 0) + 1
        self._local[key] = n
        self._publish(chan, _CONSUMED, n)
        return n

    def publish_gauge(self, chan: str, value: int) -> None:
        """Publish an ABSOLUTE value for a NON-WIRE channel (``CHAN_PASS``).

        Deliberately not monotone, and that is safe ONLY because nothing
        blocks on a gauge. The monotonicity of ``bump_sent`` exists so a
        stale read can never send a receiver into an unbounded wait; a
        gauge is read for DIAGNOSIS only, so a stale read costs a slightly
        old number in a log line and nothing else. Do not route a wire
        through here.
        """
        self._local[(chan, _SENT)] = int(value)
        self._publish(chan, _SENT, int(value))

    # -- reading --------------------------------------------------------

    def _read(self, chan: str, kind: str, rank: int) -> int:
        if int(rank) == self.rank:
            # Our own counts come from memory, never from the file: a
            # failed publish must not make this rank forget its own state.
            return self._local.get((chan, kind), 0)
        try:
            with open(self._path(chan, kind, rank)) as fh:
                return int(fh.read().strip() or 0)
        except (OSError, ValueError):
            # Absent means "nothing sent/consumed yet", which is the truth
            # before the first publish. A malformed read is the same
            # answer one poll early -- counter-lags-send, the safe skew.
            return 0

    def sent(self, chan: str, rank: int) -> int:
        return self._read(chan, _SENT, rank)

    def consumed(self, chan: str, rank: int) -> int:
        return self._read(chan, _CONSUMED, rank)

    def attempted(self, chan: str, rank: int) -> int:
        """How many sends ``rank`` has ENTERED on ``chan``. See bump_attempted.

        Always >= ``sent(chan, rank)`` for the same rank, since every post
        is entered before it completes. A reader that gates on this one is
        therefore strictly more permissive than one gating on ``sent``,
        which is the entire point: the difference between the two IS the
        set of messages currently being posted.
        """
        return self._read(chan, _ATTEMPTED, rank)

    def local_sent(self, chan: str) -> int:
        return self._local.get((chan, _SENT), 0)

    def local_consumed(self, chan: str) -> int:
        return self._local.get((chan, _CONSUMED), 0)

    def sweep(self) -> int:
        """Drop THIS RANK's own counter files. Boot housekeeping only.

        Own-rank only, deliberately. Ranks construct their counters
        independently at boot with no barrier between them, so a sweep
        that removed peers' files could delete a count a faster rank had
        already published -- and a count read as 0 when it is really 3 is
        exactly the phantom-message hazard the ordering rule exists to
        prevent, arriving by another door.

        Cross-boot hygiene needs no help from here: the instance tag is
        unique per boot and ``PhaseFlipPresence.sweep_foreign_instances``
        removes every file of every earlier tag, counters included, before
        anything can be read.
        """
        removed = 0
        prefix = f"{self.instance}.ctr."
        suffixes = (
            f".{_SENT}{self.rank}",
            f".{_CONSUMED}{self.rank}",
            f".{_ATTEMPTED}{self.rank}",
        )
        try:
            names = os.listdir(self.directory)
        except OSError:
            return 0
        for name in names:
            if not name.startswith(prefix) or not name.endswith(suffixes):
                continue
            try:
                os.unlink(os.path.join(self.directory, name))
                removed += 1
            except OSError:
                pass
        return removed


__all__ = [
    "PhaseFlipCounters",
    "CHAN_REQ",
    "CHAN_DICT",
    "CHAN_PASS",
    "CHAN_SLOT",
    "LOG_PREFIX",
]
