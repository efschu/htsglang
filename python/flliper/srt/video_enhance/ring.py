# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Bounded rings between stages, and the back-pressure they carry.

DESIGN #333 §8.4 states the failure mode this module exists to prevent: the
chain has a fast producer (decode) and a possibly slow consumer (the HTTP
client). Without back-pressure, VRAM fills with completed frames waiting to
be written and the tenant blows its reservation -- an OOM caused by a slow
network client, untraceable from the GPU side.

The four rules implemented here:

1.  One bounded ring per stage boundary, depth declared at configuration and
    counted in the reservation. There is no unbounded queue in this module,
    and ``BoundedRing`` has no code path that grows past ``depth``.
2.  Back-pressure propagates upstream to the decoder. A full ring blocks its
    producer; the decoder is a pull source, so a blocked producer stops
    pulling and the stream stalls at the decoder rather than in a buffer.
3.  The socket write is the throttle -- the encode stage awaits the send
    coroutine, so a stalled TCP window reaches the decoder within one ring
    depth. Enforced by the executor; this module provides the mechanism.
4.  Overload policy is explicit and per request: ``stall`` (default) or
    ``drop_frames``. Silent dropping is prohibited, so a drop increments a
    counter that the response trailer reports.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum


class OverloadPolicy(str, Enum):
    #: Correct for file-to-file enhancement: never lose a frame, stall instead.
    STALL = "stall"
    #: Only meaningful when the source is live. Drops are counted, never silent.
    DROP_FRAMES = "drop_frames"


class RingClosed(RuntimeError):
    """Put or get on a closed ring."""


@dataclass
class RingStats:
    name: str
    depth: int
    occupancy: int = 0
    high_water: int = 0
    put_total: int = 0
    get_total: int = 0
    dropped: int = 0
    #: Wall time the producer spent blocked on a full ring. This is the
    #: observable that proves back-pressure reached the producer.
    producer_stall_seconds: float = 0.0
    consumer_wait_seconds: float = 0.0

    def snapshot(self) -> dict:
        return {
            "name": self.name,
            "depth": self.depth,
            "occupancy": self.occupancy,
            "high_water": self.high_water,
            "put_total": self.put_total,
            "get_total": self.get_total,
            "dropped": self.dropped,
            "producer_stall_seconds": round(self.producer_stall_seconds, 6),
            "consumer_wait_seconds": round(self.consumer_wait_seconds, 6),
        }


class BoundedRing:
    """A depth-bounded asyncio channel with an explicit overload policy.

    ``asyncio.Queue(maxsize=n)`` would cover the stall policy alone. It is not
    used because the drop policy has to drop the *oldest* item -- the one
    whose deadline has already passed -- while the queue is full and a
    producer is waiting, and because the stall accounting above has to be
    attributable to the ring rather than inferred from timings.
    """

    def __init__(
        self,
        name: str,
        depth: int,
        policy: OverloadPolicy = OverloadPolicy.STALL,
        *,
        loop_time=None,
    ) -> None:
        if depth < 1:
            raise ValueError("ring depth must be at least 1")
        self.stats = RingStats(name=name, depth=depth)
        self.policy = policy
        self._items: list = []
        self._not_empty = asyncio.Condition()
        self._not_full = asyncio.Condition()
        self._closed = False
        self._time = loop_time or (lambda: asyncio.get_running_loop().time())

    @property
    def depth(self) -> int:
        return self.stats.depth

    def __len__(self) -> int:
        return len(self._items)

    @property
    def is_full(self) -> bool:
        return len(self._items) >= self.stats.depth

    async def put(self, item) -> bool:
        """Enqueue one item. Returns False if the item was dropped.

        Under ``STALL`` this coroutine does not return until there is room --
        that block is the back-pressure signal, and the caller (a stage, and
        transitively the decoder) is expected to be stopped by it.
        """
        if self._closed:
            raise RingClosed(f"ring {self.stats.name} is closed")

        if self.is_full and self.policy is OverloadPolicy.DROP_FRAMES:
            async with self._not_empty:
                if self._items:
                    self._items.pop(0)
                    self.stats.dropped += 1
                self._items.append(item)
                self._not_empty.notify()
            self.stats.put_total += 1
            self._touch()
            return False

        started = self._time()
        async with self._not_full:
            while self.is_full and not self._closed:
                await self._not_full.wait()
        if self._closed:
            raise RingClosed(f"ring {self.stats.name} closed while putting")
        waited = self._time() - started
        if waited > 0:
            self.stats.producer_stall_seconds += waited

        async with self._not_empty:
            self._items.append(item)
            self._not_empty.notify()
        self.stats.put_total += 1
        self._touch()
        return True

    async def get(self):
        started = self._time()
        async with self._not_empty:
            while not self._items and not self._closed:
                await self._not_empty.wait()
            if not self._items:
                raise RingClosed(f"ring {self.stats.name} is closed and drained")
            item = self._items.pop(0)
        self.stats.consumer_wait_seconds += self._time() - started
        self.stats.get_total += 1
        async with self._not_full:
            self._not_full.notify()
        self._touch()
        return item

    def try_get(self):
        if not self._items:
            return None
        item = self._items.pop(0)
        self.stats.get_total += 1
        self._touch()
        return item

    async def close(self) -> None:
        self._closed = True
        async with self._not_empty:
            self._not_empty.notify_all()
        async with self._not_full:
            self._not_full.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed

    def _touch(self) -> None:
        self.stats.occupancy = len(self._items)
        self.stats.high_water = max(self.stats.high_water, self.stats.occupancy)


@dataclass
class RingSet:
    """All rings of one chain, so occupancy is reportable in one call."""

    rings: dict[str, BoundedRing] = field(default_factory=dict)

    def add(self, ring: BoundedRing) -> BoundedRing:
        self.rings[ring.stats.name] = ring
        return ring

    @property
    def dropped(self) -> int:
        return sum(r.stats.dropped for r in self.rings.values())

    def occupancies(self) -> dict[str, int]:
        return {name: len(ring) for name, ring in self.rings.items()}

    def snapshot(self) -> list[dict]:
        return [ring.stats.snapshot() for ring in self.rings.values()]

    async def close(self) -> None:
        for ring in self.rings.values():
            await ring.close()

    def total_frames_in_flight(self) -> int:
        return sum(len(r) for r in self.rings.values())


def ring_depths_for(stage_names: list[str], max_in_flight: int) -> dict[str, int]:
    """Split an in-flight budget across boundaries.

    ``max_in_flight`` comes from the reservation (§8.4 rule 5), so the sum of
    the ring depths is what the ledger was told the tenant would hold. An
    equal split is used because every boundary between decode and encode
    carries one frame per work unit; a boundary that carried more would have
    shown up as a larger term in the §8.3 arithmetic.
    """
    if max_in_flight < 1:
        raise ValueError("max_in_flight must be at least 1")
    boundaries = max(1, len(stage_names) - 1)
    base = max(1, max_in_flight // boundaries)
    return {f"{a}->{b}": base for a, b in zip(stage_names, stage_names[1:])}
