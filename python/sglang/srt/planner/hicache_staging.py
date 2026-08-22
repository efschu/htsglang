# SPDX-License-Identifier: Apache-2.0
"""#810: derive the size of a STAGING host tier, so the number is not a hand-pin.

`--hicache-host-role staging` declares that the pinned host tier is a buffer
in front of the storage tier rather than an L2 retention cache. It refuses
`--hicache-ratio` and requires an explicit `--hicache-size`, because a
staging area is sized from drain bandwidth x latency and a ratio of the
device pool does not express that. This module is where that size comes from.

It lives in the planner because the planner is the sole authority over the
memory budget (#584/#785). A hand-picked `--hicache-size` would be exactly
the pool pin that authority exists to prevent, and the flag's own refusal
would then be protecting a number nobody derived.

TWO CONSUMERS, NOT ONE
----------------------
The tier is a landing area for BOTH directions, and sizing for only the
write side silently bounds the read side:

* WRITE STAGING. Write-through hands a page to the storage backend and the
  page must stay resident until that write drains. The bytes in flight are
  ``drain_bytes_per_s * drain_latency_s``, times a margin for burstiness.
* READ LANDING. A prefetch takes its destination slot from
  ``mem_pool_host.alloc()`` BEFORE the storage read is issued
  (``hiradix_cache.py``, the `_prefetch` path). If the alloc fails the
  prefetch evicts, retries, truncates, and finally abandons -- a correct
  miss, never a wrong hit, so this direction is safe but not free: the tier
  size is what bounds how many reads can be in flight at once.

The tier must hold the larger of the two. Sizing from the write side alone
produces a tier that is correct and quietly serialises prefetch.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------
It does not decide whether the ingest rate is sustainable. If write-through
produces bytes faster than the backend drains them, NO finite staging size
holds: the residency time of a page is set by the drain, and a faster
producer grows the in-flight set without bound. A ring can bound a BURST; it
cannot bound a sustained imbalance. That case needs backpressure at the
producer -- write-through must slow down when the ring is dry, with a counted
fallback -- which is a runtime mechanism and not a size. :func:`sustainable`
answers the question so the caller can refuse rather than emit a number that
cannot be correct; it does not invent a size for the unsustainable case.

This distinction matters because #720's `ReadBufferPool` made the opposite
choice for READS on purpose: on exhaustion it allocates a fresh uncounted
buffer rather than blocking, trading "a bounded spike for an unbounded
latency". That is right for a prefetch worker and wrong for a write path
under a RAM budget, where re-inflating the pinned footprint is the specific
failure the budget exists to prevent.
"""

from __future__ import annotations

import math
from typing import Optional

#: Burst margin over the steady-state in-flight bytes. Write-through arrives
#: in bursts (a prefill completing publishes many pages at once) while the
#: drain is roughly steady, so the ring must absorb more than the average.
#: A multiplier rather than a constant: the burst scales with the drain
#: window it is measured against, not with an absolute number of bytes.
DEFAULT_BURST_MARGIN = 2.0

#: `--hicache-size` is an integer count of GB, so anything below 1 rounds to
#: a tier that cannot hold a single page. The floor is the flag's own
#: granularity, not a safety opinion.
MIN_STAGING_GB = 1

#: GB as the flag means it: `pool_host/base.py` computes
#: `host_size * 1e9 // size_per_token`, i.e. decimal GB, not GiB. Matching
#: that exactly keeps the derived number and the allocated tier in step.
BYTES_PER_GB = 1e9


def write_staging_bytes(
    drain_bytes_per_s: float,
    drain_latency_s: float,
    burst_margin: float = DEFAULT_BURST_MARGIN,
) -> float:
    """Bytes that must stay resident while write-through drains.

    The classic bandwidth-delay product: a page is pinned from the moment it
    is handed to the backend until the write acks, so the resident set is the
    drain rate times that residency, widened by the burst margin.
    """
    if drain_bytes_per_s <= 0 or drain_latency_s <= 0:
        raise ValueError(
            "write staging needs a positive drain rate and latency; got "
            f"drain_bytes_per_s={drain_bytes_per_s}, "
            f"drain_latency_s={drain_latency_s}. A zero drain is not a small "
            "staging area, it is an unbounded one."
        )
    if burst_margin < 1.0:
        raise ValueError(
            f"burst_margin must be >= 1.0 (got {burst_margin}): a margin "
            "below the steady-state in-flight bytes would size the tier "
            "under its own drain."
        )
    return drain_bytes_per_s * drain_latency_s * burst_margin


def read_landing_bytes(max_concurrent_prefetch: int, page_bytes: int) -> float:
    """Bytes needed so `max_concurrent_prefetch` reads can hold a slot.

    Each in-flight prefetch owns its destination in the host tier for the
    whole read, so this direction scales with CONCURRENCY rather than with
    bandwidth.
    """
    if max_concurrent_prefetch < 0 or page_bytes < 0:
        raise ValueError(
            "read landing needs non-negative concurrency and page size; got "
            f"max_concurrent_prefetch={max_concurrent_prefetch}, "
            f"page_bytes={page_bytes}."
        )
    return float(max_concurrent_prefetch) * float(page_bytes)


def sustainable(ingest_bytes_per_s: float, drain_bytes_per_s: float) -> bool:
    """Can ANY finite staging size hold this configuration?

    False when the producer outruns the backend. The honest answer then is
    not a bigger number -- the in-flight set grows without bound at any size
    -- but backpressure at the producer. Callers should refuse rather than
    emit.
    """
    return drain_bytes_per_s >= ingest_bytes_per_s


def staging_size_gb(
    drain_bytes_per_s: float,
    drain_latency_s: float,
    max_concurrent_prefetch: int = 0,
    page_bytes: int = 0,
    burst_margin: float = DEFAULT_BURST_MARGIN,
) -> int:
    """The `--hicache-size` value (GB) for a staging tier.

    The larger of the two consumers, rounded UP to the flag's granularity:
    rounding down would emit a tier smaller than the derivation that
    justifies it.
    """
    need = max(
        write_staging_bytes(drain_bytes_per_s, drain_latency_s, burst_margin),
        read_landing_bytes(max_concurrent_prefetch, page_bytes),
    )
    return max(MIN_STAGING_GB, int(math.ceil(need / BYTES_PER_GB)))


def describe_staging_size(
    drain_bytes_per_s: float,
    drain_latency_s: float,
    max_concurrent_prefetch: int = 0,
    page_bytes: int = 0,
    burst_margin: float = DEFAULT_BURST_MARGIN,
) -> str:
    """Human-readable derivation, for the planner's info line and boot logs.

    Names WHICH consumer bound the result, because the two have different
    remedies: a write-bound tier shrinks by draining faster, a read-bound one
    only by admitting fewer concurrent prefetches.
    """
    write_b = write_staging_bytes(drain_bytes_per_s, drain_latency_s, burst_margin)
    read_b = read_landing_bytes(max_concurrent_prefetch, page_bytes)
    gb = staging_size_gb(
        drain_bytes_per_s,
        drain_latency_s,
        max_concurrent_prefetch,
        page_bytes,
        burst_margin,
    )
    bound = "write staging" if write_b >= read_b else "read landing slots"
    return (
        f"--hicache-size {gb} GB "
        f"(write staging {write_b / BYTES_PER_GB:.2f} GB = "
        f"{drain_bytes_per_s / BYTES_PER_GB:.2f} GB/s x {drain_latency_s:g} s "
        f"x {burst_margin:g} margin; read landing "
        f"{read_b / BYTES_PER_GB:.2f} GB = {max_concurrent_prefetch} "
        f"concurrent prefetch x {page_bytes} B/page; bound by {bound})"
    )


def fits_pinned_host_budget(
    size_gb: int,
    name: str = "HiCache staging host tier",
    flag: str = "--hicache-size (--hicache-host-role staging)",
) -> Optional[str]:
    """Ask the #729 authority whether this tier may be pinned.

    Returns None when it fits, or the refusal text when it does not. This
    REGISTERS the post, exactly as every other pinned producer does -- the
    budget sums posts and never caps, so a tier that does not fit has to be
    refused where it is derived rather than discovered at allocation time.
    """
    from sglang.srt.mem_cache.pinned_host_budget import check_and_register_pinned_post

    try:
        check_and_register_pinned_post(
            name=name,
            flag=flag,
            requested_bytes=int(size_gb * BYTES_PER_GB),
        )
    except ValueError as exc:
        return str(exc)
    return None
