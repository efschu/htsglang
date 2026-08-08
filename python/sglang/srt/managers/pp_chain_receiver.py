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
through one instance: ``poll()`` for the armed non-blocking path and
``recv()`` for the ordinary blocking one, sharing one inbox and one
in-flight message.

BUFFERING IS NOT DROPPING
-------------------------
Messages consumed while a flip is armed are queued in ``inbox`` and handed
to the scheduler by the ordinary ``recv()`` once the flip is over. The
armed rank keeps its upstream unblocked without acting on the requests
mid-flip, which is exactly what the flip's quiescence requirement wants.
"""

from __future__ import annotations

import logging
import pickle
from collections import deque
from typing import Any, Deque, List, Optional

import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

LOG_PREFIX = "PP-CHAIN-RECV"

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

    def __init__(self, group, src: int, dst: int):
        self._group = group
        self._src = int(src)
        self._dst = int(dst)
        self._state = _IDLE
        self._size_tensor: Optional[torch.Tensor] = None
        self._size_work = None
        self._data_tensor: Optional[torch.Tensor] = None
        self._data_work = None
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
        self._state = _AWAITING_SIZE

    def _post_data_recv(self, size: int) -> None:
        self._data_tensor = torch.empty(size, dtype=torch.uint8)
        self._data_work = dist.irecv(
            self._data_tensor, src=self._src, group=self._group
        )
        self._state = _AWAITING_DATA

    def _complete_data(self) -> None:
        serialized = bytes(self._data_tensor.cpu().numpy())
        self._data_tensor = None
        self._data_work = None
        self._state = _IDLE
        self.inbox.append(pickle.loads(serialized))

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
                self._size_work.wait()
            elif not self._size_work.is_completed():
                return False
            size = int(self._size_tensor.item())
            self._size_tensor = None
            self._size_work = None
            if size == 0:
                # An empty forward. It still carries the pass, so it is a
                # message like any other and must be queued rather than
                # swallowed: the scheduler counts on one receive per
                # upstream send.
                self._state = _IDLE
                self.inbox.append([])
                return True
            self._post_data_recv(size)

        if self._state == _AWAITING_DATA:
            # The payload is already on the wire behind a size that has
            # been received, so this completes without any further
            # cooperation from the peer -- blocking here cannot deadlock,
            # and not blocking here cannot lose the message.
            if block:
                self._data_work.wait()
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

    def recv(self) -> List[Any]:
        """Return the next chain message, blocking until one exists.

        Inbox first: messages absorbed by ``poll()`` during a flip are
        handed over here, in arrival order, before anything new is taken
        off the wire.
        """
        while not self.inbox:
            self._advance(block=True)
        return self.inbox.popleft()


__all__ = ["PpChainReceiver", "LOG_PREFIX"]
