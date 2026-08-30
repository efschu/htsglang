# SPDX-License-Identifier: Apache-2.0
"""#810: a COUNTED bound on write-through pages awaiting their storage drain.

WHAT OCCUPIES A STAGING TIER, precisely. A write-through backup pins host
capacity for longer than the device->host copy: ``write_backup`` allocates the
host slots, ``write_backup_storage`` calls ``node.protect_host()`` when the
page is handed to the storage backend, and the protection is only dropped when
the backup ACKS (``ongoing_backup.pop(...)`` -> ``entry.release_host()``). So
the resident set is the bandwidth-delay product of the DRAIN, which is exactly
the quantity ``planner/hicache_staging.write_staging_bytes`` sizes the tier
from. Nothing at runtime bounded that set; the tier itself did, by running out.

WHY RUNNING OUT IS NOT A BOUND. Under ``--hicache-host-role retention`` the
tier is large and saturation is a hit-rate question. Under ``staging`` the tier
is deliberately small, and an unbounded write-through consumer takes ALL of it:

  * The read consumer starves. A prefetch takes its landing slot from
    ``mem_pool_host.alloc()`` BEFORE issuing the storage read, so with the tier
    full of undrained write-through pages the prefetch evicts, retries,
    truncates and abandons. That is a correct miss and an invisible one -- no
    capacity metric shows a serialised prefetch, only latency does.
  * The refusal is silent. ``write_backup`` learns about exhaustion by reading
    a ``None`` back out of an allocation that already happened, and then, with
    no rank-uniform floor published, takes ``evict_host()`` -- a RANK-LOCAL
    tree edit, which is the #645 divergence.

So the bound belongs BEFORE the allocation, and it has to be counted.

WHY NOT #720's ``ReadBufferPool``. That class solves the shaped-alike problem
for READS and makes the opposite exhaustion choice on purpose: its
``acquire()`` (``read_buffer_pool.py:99-105``) allocates a fresh, UNCOUNTED
pinned buffer and increments ``overflow_allocations`` rather than blocking,
because "stalling the prefetch worker to save memory would trade a bounded
spike for an unbounded latency". Correct there, and precisely wrong here: the
footprint that overflow re-inflates is the one a staging tier exists to cap.
Reusing that class verbatim on the write path would restore, page by page, the
pinned bytes ``--hicache-host-role staging`` was set to remove. This ring
therefore NEVER allocates. Its only two outcomes are "admitted" and "refused,
counted" -- backpressure at the producer, in the direction
``planner/hicache_staging`` names when :func:`~sglang.srt.planner.
hicache_staging.sustainable` returns False: no finite size holds an ingest
rate above the drain rate, so a finite size is only meaningful if the producer
can be made to wait.

WHERE THE CAPACITY COMES FROM, and why it is not a new number. The read
consumer is already bounded at runtime: ``cache_controller.prefetch_capacity_
limit`` is ``int(0.5 * mem_pool_host.size)``. The two consumers share one
tier, so the write consumer's bound is that number's COMPLEMENT. Deriving it
keeps this module from becoming a second sizing authority next to the planner
(#584/#785) -- it introduces no constant of its own.

NO HYSTERESIS, deliberately. A refused backup is not lost work that has to be
re-driven: the node stays in the tree and the next insert that reaches it
offers it again, so there is no thrash cost for a watermark to damp. A resume
fraction would be an invented number damping a cost that does not exist.

DEFAULT OFF. The ring is constructed only under ``--hicache-host-role
staging``. Under ``retention`` -- the default -- the attribute is ``None`` and
every call site is a single ``is None`` test, so that path is unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Hashable, Optional

logger = logging.getLogger(__name__)

LOG_PREFIX = "[#810 staging-ring]"


class StagingWriteRing:
    """A counted, non-allocating bound on undrained write-through pages.

    Occupancy is tracked per KEY rather than as a bare counter so that a
    release can never subtract bytes twice and a release for something that
    was never admitted is a no-op. That matters because the release edges are
    the same ones that retire ``ongoing_backup``, including the force-release
    path taken on detach -- a bookkeeping leak there would shrink the ring
    permanently and stop write-through altogether, which is a worse failure
    than the overshoot it would be protecting against.
    """

    def __init__(self, *, capacity_tokens: int, name: str = "hicache staging") -> None:
        if capacity_tokens <= 0:
            raise ValueError(
                f"{LOG_PREFIX} capacity_tokens must be > 0, got {capacity_tokens}. "
                "A ring that can admit nothing does not throttle write-through, "
                "it disables it."
            )
        self.name = name
        self.capacity_tokens = int(capacity_tokens)
        self._inflight: Dict[Hashable, int] = {}
        self._occupied_tokens = 0
        # Counters, not gauges: the point of a bound is that its refusals are
        # a NUMBER rather than an invisible slowdown.
        self.admitted = 0
        self.admitted_tokens = 0
        self.refused = 0
        self.refused_tokens = 0
        self.readmitted = 0
        self.occupied = 0
        self.aborted = 0
        self.released = 0
        self.peak_occupied_tokens = 0

    @property
    def occupied_tokens(self) -> int:
        """Tokens admitted and not yet drained."""
        return self._occupied_tokens

    @property
    def available_tokens(self) -> int:
        return max(0, self.capacity_tokens - self._occupied_tokens)

    def admit(self, key: Hashable, tokens: int) -> bool:
        """Admit ``tokens`` under ``key``, or refuse. NEVER allocates.

        Re-admitting a key that is still in flight is idempotent and does not
        charge twice: the caller's retry loops (a node offered again by a
        later insert before its first backup drained) must not be able to
        inflate the occupancy of a page that is already counted.
        """
        if tokens <= 0:
            return True
        if key in self._inflight:
            self.readmitted += 1
            return True
        if self._occupied_tokens + tokens > self.capacity_tokens:
            self.refused += 1
            self.refused_tokens += int(tokens)
            # #1035 W12: THE COUNTER NOBODY READS.
            # `refused`/`refused_tokens` have been maintained here all along,
            # but `stats()` -- their only exposure -- has ZERO callers in the
            # tree, so a write-through page refused for want of staging room
            # left no trace anywhere. Nothing can be READ back that was never
            # WRITTEN, which makes a silent write-side refusal indistinguishable
            # from a read-side miss for whoever debugs the read path afterwards.
            # Emit at the refusal itself rather than waiting for a reporting
            # site to be invented; rate-limited so a full ring cannot flood.
            if self.refused <= 40 or self.refused % 256 == 0:
                logger.warning(
                    "#1035 W12 STAGING RING REFUSED n=%d ring=%s: %d token(s) "
                    "rejected, occupied=%d/%d, refused_tokens=%d. This page is "
                    "NOT written through -- it can never be read back.",
                    self.refused,
                    self.name,
                    int(tokens),
                    self._occupied_tokens,
                    self.capacity_tokens,
                    self.refused_tokens,
                )
            return False
        self._inflight[key] = int(tokens)
        self._occupied_tokens += int(tokens)
        self.admitted += 1
        self.admitted_tokens += int(tokens)
        if self._occupied_tokens > self.peak_occupied_tokens:
            self.peak_occupied_tokens = self._occupied_tokens
        return True

    def occupy(self, key: Hashable, tokens: int) -> None:
        """Charge a page that is ALREADY resident. Cannot be refused.

        The residency has two phases and only the first one can be refused.
        Phase one is the device->host copy, admitted by :meth:`admit` before
        any host slot is taken -- that is where a refusal is actionable,
        because nothing has been allocated yet. Phase two is the storage
        drain: ``write_backup_storage`` hands the page to the backend and
        ``node.protect_host()`` keeps it resident until the backup acks.
        Refusing there would not free anything, it would only stop the page
        from being counted, and the drain window is the whole quantity
        ``planner/hicache_staging.write_staging_bytes`` sizes the tier from.

        Keying phase two by the STORAGE OPERATION id rather than by the node
        is what makes the accounting split-safe: one write-through ack can
        fan out into several storage backups after a node split, each with its
        own id and its own ack, and each releases exactly its own charge.

        Occupancy may exceed the capacity here. That is not an error, it is
        the bound doing its job in the only way left: the next :meth:`admit`
        refuses until the drain catches up.
        """
        if tokens <= 0 or key in self._inflight:
            return
        self._inflight[key] = int(tokens)
        self._occupied_tokens += int(tokens)
        self.occupied += 1
        if self._occupied_tokens > self.peak_occupied_tokens:
            self.peak_occupied_tokens = self._occupied_tokens

    def abort(self, key: Hashable) -> None:
        """Give back an admission whose write never started."""
        if self._pop(key):
            self.aborted += 1

    def release(self, key: Hashable) -> None:
        """Give back an admission whose storage write has drained."""
        if self._pop(key):
            self.released += 1

    def release_all(self) -> None:
        """Drop every admission (detach / shutdown)."""
        self._inflight.clear()
        self._occupied_tokens = 0

    def _pop(self, key: Hashable) -> bool:
        tokens = self._inflight.pop(key, None)
        if tokens is None:
            return False
        self._occupied_tokens -= tokens
        if self._occupied_tokens < 0:
            # Unreachable by construction (occupancy is the sum of the dict).
            # Clamped rather than asserted because a bookkeeping slip must not
            # take down a serving process.
            self._occupied_tokens = 0
        return True

    def stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "capacity_tokens": self.capacity_tokens,
            "occupied_tokens": self._occupied_tokens,
            "peak_occupied_tokens": self.peak_occupied_tokens,
            "inflight_pages": len(self._inflight),
            "admitted": self.admitted,
            "admitted_tokens": self.admitted_tokens,
            "refused": self.refused,
            "refused_tokens": self.refused_tokens,
            "readmitted": self.readmitted,
            "occupied": self.occupied,
            "aborted": self.aborted,
            "released": self.released,
        }


def build_staging_write_ring(
    server_args: Any, cache_controller: Any
) -> Optional[StagingWriteRing]:
    """The ring for a STAGING host tier, or ``None`` for every other case.

    ``None`` is the default and the whole of the retention path: the role flag
    defaults to ``retention``, and callers test the result for ``None`` before
    doing anything at all, so a retention boot executes the same instructions
    it did before this module existed.

    The capacity is the complement of the read consumer's existing bound
    (``prefetch_capacity_limit``), read AFTER the caller has symmetrized that
    limit across ranks -- so on an uneven rig the write bound is derived from
    the same group-agreed number the read bound uses, and stays rank-uniform
    with it. A rank-dependent admission bound on this path is the #645 defect.
    """
    if getattr(server_args, "hicache_host_role", "retention") != "staging":
        return None
    if cache_controller is None:
        return None
    mem_pool_host = getattr(cache_controller, "mem_pool_host", None)
    if mem_pool_host is None:
        return None
    tier_tokens = int(getattr(mem_pool_host, "size", 0) or 0)
    read_reserved = int(getattr(cache_controller, "prefetch_capacity_limit", 0) or 0)
    capacity = tier_tokens - read_reserved
    if capacity <= 0:
        # The read bound already claims the whole tier. Refusing to build a
        # ring is not "no bound": the tier is what it always was, and a ring
        # of zero would stop write-through completely rather than throttle it.
        logger.warning(
            "%s host tier of %d tokens leaves nothing after the prefetch "
            "reservation of %d; write-through stays unbounded.",
            LOG_PREFIX,
            tier_tokens,
            read_reserved,
        )
        return None
    ring = StagingWriteRing(capacity_tokens=capacity, name="hicache write-through")
    logger.info(
        "%s write-through bounded to %d of %d host tokens (prefetch holds %d).",
        LOG_PREFIX,
        capacity,
        tier_tokens,
        read_reserved,
    )
    return ring
