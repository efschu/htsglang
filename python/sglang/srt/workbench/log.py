# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The workbench's event channel (DESIGN #347 W1).

The same pattern as :class:`~sglang.srt.training.store.JobStore`'s event log --
monotonic sequence numbers, cursor pagination, asyncio subscribers, a bounded
ring -- with one difference that is the whole reason it is a separate class:
training events belong to a *job* and are read back through the OpenAI
fine-tuning protocol, while workbench events belong to the *rig* and have no
protocol to be spec-shaped for. Forcing them into a job-shaped log would mean
inventing a synthetic job id for "the tuner ran a shape", which is a lie a
client could act on.

Subscribers exist so the frontend can stream this later (ANALYSE #347 item 5:
one observation pattern across classes). M1 wires only the cursor-paginated
GET; nothing is lost by having the hook and not using it yet.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)


@dataclass
class WorkLogEntry:
    """One line in the rig's idle-work log."""

    seq: int
    created_at: float
    tenant: str
    level: str
    message: str
    type: str = "message"
    data: Optional[dict[str, Any]] = None

    def to_json(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "created_at": self.created_at,
            "tenant": self.tenant,
            "level": self.level,
            "message": self.message,
            "type": self.type,
            "data": self.data,
        }


@dataclass
class WorkLog:
    """A bounded, cursor-addressable log with optional live subscribers."""

    max_entries: int = 5000
    _entries: list[WorkLogEntry] = field(default_factory=list)
    _subscribers: list[asyncio.Queue] = field(default_factory=list)
    _seq: int = 0

    def append(
        self,
        tenant: str,
        level: str,
        message: str,
        *,
        data: Optional[Mapping[str, Any]] = None,
        event_type: str = "message",
    ) -> WorkLogEntry:
        self._seq += 1
        entry = WorkLogEntry(
            seq=self._seq,
            created_at=time.time(),
            tenant=tenant,
            level=level,
            message=message,
            type=event_type,
            data=dict(data) if data is not None else None,
        )
        self._entries.append(entry)
        if len(self._entries) > self.max_entries:
            # The head is dropped rather than the tail: a consumer that just
            # attached wants the recent end. A cursor into a dropped range is
            # answered from the oldest entry still held, see :meth:`after`.
            del self._entries[: len(self._entries) - self.max_entries]
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(entry)
            except asyncio.QueueFull:
                logger.debug("workbench: dropped an event for a slow subscriber")
        return entry

    # -- reading ------------------------------------------------------------

    def after(
        self, cursor: int = 0, *, limit: int = 200
    ) -> tuple[list[WorkLogEntry], bool]:
        """Entries with ``seq > cursor``, oldest first, plus ``has_more``.

        A cursor older than everything still held is not an error: the caller
        gets the oldest entries there are and can see from their ``seq`` that
        it missed a range. Raising would turn a slow reader into a broken one.
        """
        window = [e for e in self._entries if e.seq > int(cursor)]
        head = window[: max(1, int(limit))]
        return head, len(head) < len(window)

    def tail(self, count: int = 20) -> list[WorkLogEntry]:
        return self._entries[-max(0, int(count)) :]

    @property
    def last_seq(self) -> int:
        return self._seq

    def __len__(self) -> int:
        return len(self._entries)

    # -- streaming ----------------------------------------------------------

    def subscribe(self, *, maxsize: int = 512) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


__all__ = ["WorkLog", "WorkLogEntry"]
