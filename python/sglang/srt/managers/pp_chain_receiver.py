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
"""#631: the single owner of the PP request-chain receive stream.

WHY THIS EXISTS
---------------
The flip's design law is that no rank may block on any channel while a
peer may be in a different blocking channel. Satisfying it needs an ARMED
rank to keep CONSUMING the request chain without blocking on it -- boot 18
is the measured proof: rank 2 armed and stopped consuming, so rank 1's
ordinary top-of-pass commit of the previous pass's forward blocked in
``work.wait()`` before rank 1 could announce, the gate never assembled,
and rank 0 sat in the reduction alone.

``point_to_point_pyobj`` cannot be consumed by halves. It is a TWO-STEP
protocol -- an ``irecv`` of a size, then an ``irecv`` of that many bytes --
and the two steps are not independent: once the size has been received,
the payload is already on the wire and MUST be received, or every later
message on that stream is misframed. So a non-blocking consumer cannot
simply "peek and give up"; it has to be a state machine that remembers it
is mid-message.

That is also why this class is the SINGLE OWNER of the stream. A design
where a non-blocking drainer and a blocking ``point_to_point_pyobj`` both
post their own ``irecv`` would race for the same messages and misframe the
stream the moment both were in flight. Every consumer therefore goes
through one instance, sharing one inbox and one in-flight message.

WHICH CONSUMER TO USE, and why ``poll()`` is not it (#631 G)
------------------------------------------------------------
``consume_up_to(sent_count)`` is the ARMED path and ``recv()`` the
ordinary one. ``poll()`` is neither: it rests on ``is_completed()``
progressing a posted ``irecv``, which is MEASURED FALSE on this build
(corpse F), so it absorbs nothing and always has. It is kept only as the
pinned record of that measurement. The armed path instead learns that a
message exists from the sender's published counter
(``phase_flip_counters``) and then blocks deliberately, bounded by
transfer time.

BUFFERING IS NOT DROPPING
-------------------------
Messages consumed while a flip is armed are queued in ``inbox`` and handed
to the scheduler by the ordinary ``recv()`` once the flip is over. The
armed rank keeps its upstream unblocked without acting on the requests
mid-flip, which is exactly what the flip's quiescence requirement wants.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from collections import deque
from typing import Any, Callable, Deque, List, Optional

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.mem_cache.hicache_collective import ParkedWait

logger = logging.getLogger(__name__)

LOG_PREFIX = "PP-CHAIN-RECV"

#: #824 W4: hard abort deadline for a single blocking chain receive, in
#: seconds. DEFAULT 0 = never abort, which is the pre-#824 behaviour and
#: is deliberate: an idle PP rank legitimately waits on this channel for
#: as long as no request arrives, so a wall-clock abort that is on by
#: default would fire on a perfectly healthy idle server. Set it only where
#: something else establishes that a stall here is a WEDGE and not an idle
#: wait -- or, better, inject ``abort_check`` and decide from real state.
STALL_ENV = "SGLANG_PP_CHAIN_RECV_STALL_S"

#: How often a blocking receive surfaces to stamp the blocked-recv marker
#: and consult ``abort_check``. This is a YIELD interval, not a bound: the
#: underlying wait is never interrupted (see ParkedWait), so a short value
#: costs one Event wakeup and buys watchdog resolution.
YIELD_INTERVAL_S = 1.0


class PpChainRecvStalled(RuntimeError):
    """#824 W4: a blocking chain receive gave up by a NAMED branch.

    Raised INSTEAD of parking for ever when the upstream cannot feed this
    channel -- the boot_827 wedge, where PP0 sat in an admission send on
    the typed-dict channel while PP1 and PP2 sat in this receive on the
    request-relay channel and the ring went silent for 31 s.

    THE STREAM IS STILL FRAMED when this is raised. The posted receive was
    never cancelled and the parked wait is still running, so calling
    ``recv()`` again resumes the SAME receive and a late message still
    arrives intact. Handle this by servicing whatever the peer is actually
    waiting on, then come back -- do NOT rebuild the receiver, and do not
    post another receive on this stream.
    """


def _default_stall_timeout_s() -> float:
    raw = os.environ.get(STALL_ENV, "0")
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "%s ignoring unparsable %s=%r; blocking receives stay unbounded",
            LOG_PREFIX,
            STALL_ENV,
            raw,
        )
        return 0.0


#: State machine positions. IDLE means no message is half-received and the
#: stream may be abandoned safely; the other two mean it may NOT be.
_IDLE = "idle"
_AWAITING_SIZE = "awaiting_size"
_AWAITING_DATA = "awaiting_data"


class PpChainReceiver:
    """Non-blocking-capable receiver for one PP request-chain stream.

    Mirrors the receive half of ``point_to_point_pyobj`` exactly -- same
    two tensors, same dtypes, same order -- so it is wire-compatible with
    the unmodified send half on the upstream rank. That compatibility is
    the point: only the RECEIVER changes, and only on ranks that run with
    the phase flip enabled.
    """

    def __init__(
        self,
        group,
        src: int,
        dst: int,
        on_consumed=None,
        stall_timeout_s: Optional[float] = None,
        abort_check: Optional[Callable[[str, float], Optional[str]]] = None,
        on_blocked: Optional[Callable[[Optional[str], Optional[float]], None]] = None,
    ):
        self._group = group
        self._src = int(src)
        self._dst = int(dst)
        #: #824 W4. See STALL_ENV: 0 keeps the pre-#824 unbounded block.
        self._stall_timeout_s = (
            _default_stall_timeout_s()
            if stall_timeout_s is None
            else float(stall_timeout_s)
        )
        #: #824 W4: called as ``abort_check(arm, waited_s)`` every yield
        #: while a blocking receive is stalled. Return a reason string to
        #: abort by the named branch, or None to keep waiting. This is the
        #: seam for deciding from REAL state (is a peer owed a message on
        #: another channel?) rather than from a wall clock.
        self._abort_check = abort_check
        #: #824 W5: called as ``on_blocked(arm, since_monotonic)`` when a
        #: receive first goes overdue and as ``on_blocked(None, None)``
        #: when it clears. This is the funnel #821's
        #: ``_pp_blocked_recv_since`` marker never had on this path -- the
        #: very path two of three ranks wedged in on boot_827.
        self._on_blocked = on_blocked
        #: Diagnostics mirror of the marker, readable without the hook.
        self.blocked_recv_arm: Optional[str] = None
        self.blocked_recv_since: Optional[float] = None
        self.stall_aborts = 0
        #: #631 G: called with the new consumed count each time a whole
        #: message leaves the wire, so the upstream can learn that its
        #: send is gone and stop treating its own commit as speculative.
        #: Fired from ONE place (``_note_consumed``) for every path that
        #: completes a message, because a count that misses a path is
        #: worse than no count at all -- it would make a non-empty wire
        #: look empty at the flip's entry.
        self._on_consumed = on_consumed
        #: Whole messages taken off the WIRE by this receiver. Not the
        #: same as messages handed to the scheduler: the inbox sits
        #: between them.
        self.consumed = 0
        #: Publishes that failed. Nonzero means the upstream is blind to
        #: this rank's progress, which stalls every flip.
        self.publish_failures = 0
        self._state = _IDLE
        self._size_tensor: Optional[torch.Tensor] = None
        self._size_work = None
        self._size_park: Optional[ParkedWait] = None
        self._data_tensor: Optional[torch.Tensor] = None
        self._data_work = None
        self._data_park: Optional[ParkedWait] = None
        #: Fully received, not yet handed to the scheduler.
        self.inbox: Deque[List[Any]] = deque()
        #: Diagnostics: how many messages were absorbed by a non-blocking
        #: poll rather than by an ordinary blocking receive.
        self.polled_messages = 0

    # -- state ---------------------------------------------------------

    @property
    def mid_message(self) -> bool:
        """True when a message is HALF-RECEIVED: its size has landed and
        its payload has not.

        Only this state is dangerous. An outstanding size recv that has not
        completed has consumed nothing -- the stream is still framed at a
        message boundary and resuming or blocking on it is equally safe.
        Once the size is in, the payload is committed: giving up there and
        posting a fresh size recv would read the payload AS a size and
        misframe every later message.
        """
        return self._state == _AWAITING_DATA

    @property
    def has_outstanding_recv(self) -> bool:
        """True when an irecv is posted on this stream. Not a hazard on its
        own -- this class is the stream's only consumer -- but it is why
        that single-owner rule exists."""
        return self._state != _IDLE

    def pending(self) -> int:
        return len(self.inbox)

    # -- the machine ---------------------------------------------------

    def _post_size_recv(self) -> None:
        self._size_tensor = torch.tensor([0], dtype=torch.long)
        self._size_work = dist.irecv(
            self._size_tensor, src=self._src, group=self._group
        )
        self._size_park = None
        self._state = _AWAITING_SIZE

    def _post_data_recv(self, size: int) -> None:
        self._data_tensor = torch.empty(size, dtype=torch.uint8)
        self._data_work = dist.irecv(
            self._data_tensor, src=self._src, group=self._group
        )
        self._data_park = None
        self._state = _AWAITING_DATA

    # -- #824 W4/W5: the bounded, named, instrumented block -------------

    def _mark_blocked(self, arm: str, since: float) -> None:
        """#824 W5: publish that THIS arm is overdue, by name.

        The watchdog's job on boot_827 was to say which wait had stopped
        the ring; it could not, because #821's marker only covers
        ``_pp_recv_typed_dict`` and both wedged ranks were in here.
        """
        if self.blocked_recv_arm == arm:
            return
        self.blocked_recv_arm = arm
        self.blocked_recv_since = since
        if self._on_blocked is not None:
            try:
                self._on_blocked(arm, since)
            except Exception as exc:  # noqa: BLE001 - diagnostics never raise
                logger.error("%s blocked-recv marker hook failed: %s", LOG_PREFIX, exc)

    def _clear_blocked(self) -> None:
        if self.blocked_recv_arm is None:
            return
        self.blocked_recv_arm = None
        self.blocked_recv_since = None
        if self._on_blocked is not None:
            try:
                self._on_blocked(None, None)
            except Exception as exc:  # noqa: BLE001 - diagnostics never raise
                logger.error("%s blocked-recv marker hook failed: %s", LOG_PREFIX, exc)

    def _ensure_park(self, arm: str) -> ParkedWait:
        """Park the blocking wait for ``arm``, creating it on FIRST BLOCK.

        Deliberately lazy. Parking at post time would call ``wait()`` on
        every receive this class posts, including the ones ``poll()``
        posts -- and ``poll()``'s whole contract is that it never waits on
        a peer. Creating the waiter only when a caller actually asks to
        block keeps that contract exact (and costs no thread on the
        non-blocking path).
        """
        if arm == "size":
            if self._size_park is None:
                self._size_park = ParkedWait(
                    self._size_work, f"pp-chain-size<-{self._src}"
                )
            return self._size_park
        if self._data_park is None:
            self._data_park = ParkedWait(self._data_work, f"pp-chain-data<-{self._src}")
        return self._data_park

    def _block_on(self, park: ParkedWait, arm: str) -> None:
        """Block until ``park`` completes, an abort_check fires, or the
        stall deadline expires.

        Every exit that is not completion raises ``PpChainRecvStalled`` and
        leaves the receive POSTED and the state machine untouched, so the
        caller may service the peer and come back.
        """
        since = time.monotonic()
        deadline = self._stall_timeout_s
        yield_s = YIELD_INTERVAL_S
        if deadline and deadline > 0:
            yield_s = min(yield_s, max(deadline / 3.0, 0.01))
        while True:
            if park.join(yield_s):
                self._clear_blocked()
                return
            waited = park.waited_s
            self._mark_blocked(arm, since)
            if self._abort_check is not None:
                try:
                    reason = self._abort_check(arm, waited)
                except Exception as exc:  # noqa: BLE001
                    logger.error("%s abort_check failed: %s", LOG_PREFIX, exc)
                    reason = None
                if reason:
                    self.stall_aborts += 1
                    raise PpChainRecvStalled(
                        f"{LOG_PREFIX} arm={arm} src={self._src} dst={self._dst} "
                        f"waited={waited:.1f}s state={self._state} "
                        f"consumed={self.consumed}: {reason}. The receive is "
                        f"still posted; service the peer, then receive again."
                    )
            if deadline and deadline > 0 and waited >= deadline:
                self.stall_aborts += 1
                raise PpChainRecvStalled(
                    f"{LOG_PREFIX} arm={arm} src={self._src} dst={self._dst} "
                    f"waited={waited:.1f}s state={self._state} "
                    f"consumed={self.consumed}: exceeded the {deadline:g}s stall "
                    f"bound ({STALL_ENV}). The upstream is not feeding this "
                    f"channel -- on boot_827 it was blocked in an admission "
                    f"send on the typed-dict channel. The receive is still "
                    f"posted; service the peer, then receive again."
                )

    def _note_consumed(self) -> None:
        """One whole message has left the wire. The single accounting point.

        Publishing the count is best effort by contract -- a failed publish
        costs the upstream one more poll (counter-lags-send, the safe
        skew) -- but it must never propagate, because the message is
        already consumed and unwinding here would misframe the stream.
        """
        self.consumed += 1
        if self._on_consumed is None:
            return
        try:
            self._on_consumed(self.consumed)
        except Exception as exc:  # noqa: BLE001
            # THROTTLED, because this fires per message and a permanent
            # failure would otherwise bury the log it needs to be found in.
            # It must still be loud on the FIRST occurrence: a consumed
            # count that never gets published means the upstream can never
            # reap its send, so it withholds presence for ever and every
            # flip abandons at the presence deadline. That is exactly how
            # the first metal run of this design failed (a NameError in the
            # publish callback, boot 2026-08-09 01:00Z) -- and this line is
            # what identified it in two minutes.
            self.publish_failures += 1
            if self.publish_failures == 1 or self.publish_failures % 1000 == 0:
                logger.error(
                    "%s consumed-counter publish failed (%d so far): %s. The "
                    "upstream cannot learn its send was taken, so it will "
                    "withhold presence and every flip will abandon.",
                    LOG_PREFIX,
                    self.publish_failures,
                    exc,
                )

    def _complete_data(self) -> None:
        serialized = bytes(self._data_tensor.cpu().numpy())
        self._data_tensor = None
        self._data_work = None
        self._data_park = None
        self._state = _IDLE
        self.inbox.append(pickle.loads(serialized))
        self._note_consumed()

    def _advance(self, block: bool) -> bool:
        """Advance the machine by at most one whole message.

        Returns True when a complete message landed in the inbox. With
        ``block=False`` this NEVER waits on a peer: it returns False the
        moment the outstanding ``irecv`` is not yet complete, leaving the
        machine mid-message so the next call resumes exactly there.
        """
        if self._state == _IDLE:
            self._post_size_recv()

        if self._state == _AWAITING_SIZE:
            if block:
                # #824 W4: THE wedge site. py-spy caught PP1 and PP2 both
                # here (this file:218 pre-fix) on boot_827, parked for ever
                # on a size that PP0 could not send because PP0 was itself
                # blocked in a send on the admission channel. Bounded and
                # named now -- and still resumable, which is the property
                # that lets the caller break the ring and come back.
                self._block_on(self._ensure_park("size"), "size")
            elif not self._size_work.is_completed():
                return False
            size = int(self._size_tensor.item())
            self._size_tensor = None
            self._size_work = None
            self._size_park = None
            if size == 0:
                # An empty forward. It still carries the pass, so it is a
                # message like any other and must be queued rather than
                # swallowed: the scheduler counts on one receive per
                # upstream send.
                self._state = _IDLE
                self.inbox.append([])
                self._note_consumed()
                return True
            self._post_data_recv(size)

        if self._state == _AWAITING_DATA:
            # The payload is already on the wire behind a size that has
            # been received, so this completes without any further
            # cooperation from the peer -- blocking here cannot deadlock,
            # and not blocking here cannot lose the message.
            if block:
                # Bounded too, but for INSTRUMENTATION rather than for
                # recovery: this wait is bounded by transfer time (the
                # payload is already on the wire behind a received size),
                # so it should never go overdue. If it ever does, the
                # marker names it instead of leaving another silent arm.
                self._block_on(self._ensure_park("data"), "data")
            elif not self._data_work.is_completed():
                return False
            self._complete_data()
            return True

        return False

    # -- the two consumers ---------------------------------------------

    def poll(self, max_messages: int = 16) -> int:
        """Consume whatever has already arrived. NEVER blocks.

        This is the armed path. ``max_messages`` bounds the work done in
        one call so a fast upstream cannot hold an armed rank in here
        indefinitely -- the rank has a gate to get to.
        """
        absorbed = 0
        while absorbed < max_messages:
            try:
                if not self._advance(block=False):
                    break
            except Exception as exc:  # noqa: BLE001
                # A poll is best effort by contract, but a failure here
                # leaves the stream mid-message, so it must be loud.
                logger.error(
                    "%s non-blocking poll failed in state %s: %s",
                    LOG_PREFIX,
                    self._state,
                    exc,
                )
                break
            absorbed += 1
        self.polled_messages += absorbed
        return absorbed

    def consume_up_to(self, sent_count: int, max_messages: int = 64) -> int:
        """#631 G: take every message the SENDER says it has posted.

        This is the armed path, and it is what ``poll()`` could never be.
        ``poll()`` asks the transport "has anything arrived?" via
        ``is_completed()``, which on this build never says yes (corpse F);
        it therefore absorbs nothing. This asks a POLLABLE SIDE CHANNEL
        instead -- the sender's published counter -- and only once that
        counter proves a message is in flight does it call the BLOCKING
        advance. The block is then bounded by TRANSFER TIME, not by peer
        scheduling, because the message provably exists and the recv
        side's ``wait()`` is what drives it across (the one transport
        behaviour with positive evidence).

        GREEDY BY CONTRACT: it consumes every message the counter accounts
        for and never leaves one behind. That is what separates it from
        the bounded-recv corpse, whose failure driver was completing
        iterations WITHOUT consuming while the upstream kept sending, so
        unmatched sends piled up and the senders blocked. Here the
        upstream is armed and issues no new forwards, and this loop
        finishes the ones already posted.

        ``max_messages`` is a runaway guard, not a policy: it can only be
        hit if the counter is wrong, and stopping is then better than
        spinning. It is logged loudly if so.
        """
        absorbed = 0
        while self.consumed < int(sent_count):
            if absorbed >= max_messages:
                logger.error(
                    "%s consume_up_to hit its %d-message guard with "
                    "consumed=%d < sent=%d; the counter and the wire "
                    "disagree",
                    LOG_PREFIX,
                    max_messages,
                    self.consumed,
                    int(sent_count),
                )
                break
            # Blocking, and safe: the counter says this message is posted.
            self._advance(block=True)
            absorbed += 1
        return absorbed

    def recv(self, max_messages: int = 64) -> List[Any]:
        """Return the next chain message, blocking until one exists.

        Inbox first: messages absorbed while a flip was armed are handed
        over here, in arrival order, before anything new is taken off the
        wire.

        #1071 (A-ii): BOUNDED THE SAME WAY ITS SIBLING ALREADY IS. Until
        here this loop was a bare ``while not self.inbox:
        self._advance(block=True)`` -- no counter evidence, no runaway
        guard -- while ``consume_up_to`` twenty lines above takes neither
        step without both. The asymmetry was not a design: it is where PP1
        sat for five minutes in 1068cap (07:34:09 until the deadman) after
        a one-shot armed a wait for a hop nobody owed. Two bounds, and
        neither of them invents a policy:

        * ``self._stall_timeout_s`` (``STALL_ENV``) already bounds each
          blocking ``_advance`` and raises ``PpChainRecvStalled``. Nothing
          new is added here; the launcher now sets it, because the default
          of 0 kept the pre-#824 unbounded block.
        * ``max_messages`` is the runaway guard, verbatim the sibling's:
          it can only be reached if this loop absorbs messages that never
          land in the inbox, and stopping loudly is then better than
          spinning. ``self.consumed`` is reported with it so the counter
          and the wire can be compared at the point they disagreed.
        """
        absorbed = 0
        while not self.inbox:
            if absorbed >= int(max_messages):
                raise PpChainRecvStalled(
                    f"{LOG_PREFIX} recv absorbed {absorbed} messages "
                    f"(consumed={self.consumed}) without one reaching the "
                    f"inbox; the counter and the wire disagree. Blocking on "
                    f"further advances here is the 1068cap park (PP1, "
                    f"07:34:09), so this stops instead."
                )
            self._advance(block=True)
            absorbed += 1
        return self.inbox.popleft()


__all__ = ["PpChainReceiver", "PpChainRecvStalled", "LOG_PREFIX", "STALL_ENV"]
