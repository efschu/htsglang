"""#720: a bounded, REGISTERED reusable read buffer for the file backend.

THE SPIKE. Every storage read takes its target from
``host_pool.get_dummy_flat_data_page()``, which allocates a fresh tensor with
``pin_memory=self.pin_memory`` -- on both pool families (``pool_host/mha.py``,
``memory_pool_host.py``). So a prefetch-heavy path pins one page per read,
transiently, and the joint pinned budget cannot see any of it: the registry
accounts POOLS declared at attach, and this is neither declared nor a pool. It
scales with concurrent reads rather than with tier size, which is precisely the
shape a budget built around steady-state posts is blind to.

THE FIX, and its bound. A small ring of buffers, allocated ONCE and reused,
declared to the registry like every other pinned consumer. Bounded on purpose:
the point is to make the pinned footprint of the read path a NUMBER the budget
can refuse, not to make it fast. When the ring is exhausted the borrow falls
back to a fresh allocation -- today's behaviour, counted -- rather than
blocking, because stalling the prefetch worker to save memory would trade a
bounded spike for an unbounded latency.

DEFAULT OFF. A registered pool is real memory charged at attach, and this box
refuses over-commitment; turning it on changes what fits. ``0`` keeps the
per-read allocation exactly as it is today.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

LOG_PREFIX = "[#720 read-buffers]"


class ReadBufferPool:
    """A fixed ring of reusable read targets, charged to the pinned budget.

    Not a cache and not an allocator: buffers are borrowed for the duration of
    one read and returned. The only thing it changes about the read path is who
    owns the memory.
    """

    def __init__(
        self,
        *,
        name: str,
        flag: str,
        capacity: int,
        page_bytes: int,
        factory: Callable[[], Any],
        register: bool = True,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"{LOG_PREFIX} capacity must be > 0, got {capacity}")
        self.name = name
        self.capacity = int(capacity)
        self.page_bytes = int(page_bytes)
        self._factory = factory
        self._lock = threading.Lock()
        self._free: list = []
        self._registered = False
        self.overflow_allocations = 0

        if register:
            from sglang.srt.mem_cache.pinned_host_budget import (
                check_and_register_pinned_post,
            )

            # Declared BEFORE the buffers exist, so an over-commitment is
            # refused instead of discovered: the registry's whole job is to
            # fail at the declaration rather than at the allocation.
            check_and_register_pinned_post(
                name=name,
                flag=flag,
                requested_bytes=self.capacity * self.page_bytes,
            )
            self._registered = True
        # SAME WINDOW as weights_arena's image post, closed the same way. The
        # declaration above is deliberately BEFORE the buffers exist; if making
        # them then fails, the post would describe bytes that never existed and
        # #706's credit-back would subtract them from the next admission. Only
        # the failure path is new -- the original error is re-raised untouched.
        try:
            self._free = [factory() for _ in range(self.capacity)]
        except BaseException:
            if self._registered:
                from sglang.srt.mem_cache.pinned_host_budget import (
                    unregister_pinned_post,
                )

                try:
                    unregister_pinned_post(self.name)
                except Exception:  # noqa: BLE001 -- cleanup never masks
                    pass
                self._registered = False
            raise

    def acquire(self):
        """Borrow a buffer. Falls back to a fresh one when the ring is dry."""
        with self._lock:
            if self._free:
                return self._free.pop()
            self.overflow_allocations += 1
        return self._factory()

    @property
    def available(self) -> int:
        """Slots currently in the ring, i.e. not borrowed.

        A consumer that must return the ring to full between uses (#809's
        rotation is one: a slot leaked per flip degrades the ring into
        unbounded allocation by the third one) needs to be able to ASK,
        rather than read the free list through the class's back.
        """
        with self._lock:
            return len(self._free)

    def release(self, buffer: Any) -> None:
        """Return a buffer. Extras beyond the ring are simply dropped."""
        with self._lock:
            if len(self._free) < self.capacity:
                self._free.append(buffer)

    def close(self) -> None:
        """Drop the ring and stop charging the budget for it."""
        with self._lock:
            self._free.clear()
        if self._registered:
            from sglang.srt.mem_cache.pinned_host_budget import (
                unregister_pinned_post,
            )

            unregister_pinned_post(self.name)
            self._registered = False
        if self.overflow_allocations:
            logger.info(
                "%s %s: %d read(s) exceeded the ring of %d and fell back to a "
                "fresh allocation.",
                LOG_PREFIX,
                self.name,
                self.overflow_allocations,
                self.capacity,
            )

    def __enter__(self) -> ReadBufferPool:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class borrowed:
    """Context manager around one borrow, so a raised read still returns it."""

    def __init__(self, pool: Optional[ReadBufferPool], fallback: Callable[[], Any]):
        self._pool = pool
        self._fallback = fallback
        self._buffer = None

    def __enter__(self):
        self._buffer = (
            self._pool.acquire() if self._pool is not None else self._fallback()
        )
        return self._buffer

    def __exit__(self, *exc) -> None:
        if self._pool is not None and self._buffer is not None:
            self._pool.release(self._buffer)
        self._buffer = None
