"""Bounded waits for the HiCache cross-rank control collectives (#259, #630).

The control collectives of the hierarchical caches run on the gloo ``cpu_group``s
built by ``GroupCoordinator.__init__``, whose ``gloo_timeout`` defaults to
``timedelta(seconds=120 * 60)`` -- two hours (``parallel_state.py:619``). Nothing
on the HiCache path shortens it. A peer that dies without closing its socket, or
a PP rank that never reaches the matching send, therefore parks the survivor for
two hours and then surfaces a generic ``RuntimeError`` from inside gloo that
names no call site.

Every helper here turns that into a named, bounded failure: poll the ``Work``
against a deadline and raise ``HiCacheCollectiveTimeoutError`` carrying the
collective's label, the rank that gave up, and how long it actually waited.

The healthy path is unchanged. ``COLLECTIVE_POLL_SPINS`` bare loop iterations run
before the first ``time.sleep``, which covers the latency of a CPU collective
between local ranks by a wide margin, so a completed ``isend`` or ``all_reduce``
costs a few hundred predicted-branch iterations and no syscall. The sleep path is
only ever reached once something is already wrong.

This module holds the mechanism so that ``unified_radix_cache`` and
``hiradix_cache`` share ONE implementation rather than two drifting copies --
the drift is exactly what #630 was: ``hiradix_cache`` never received the #259
work and kept raw blocking calls on the same collectives.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import torch


class HiCacheCollectiveError(RuntimeError):
    """A HiCache control collective cannot be issued rank-uniformly."""


class HiCacheCollectiveDesyncError(HiCacheCollectiveError):
    """A HiCache control collective reduced a payload it did not post.

    Raised when a self-identifying control vector comes back carrying a
    foreign tag, i.e. the ranks were not all inside the same collective.
    gloo only aborts the process when the mismatched buffers differ in size
    (``op.preamble.length <= op.nbytes``); equal-sized traffic from another
    site is accepted silently and would corrupt the reduced values instead.
    This turns that second case into a named error on every rank.
    """


class HiCacheCollectiveTimeoutError(HiCacheCollectiveError):
    """A HiCache control collective did not complete within its bound.

    Raised on the surviving rank when a peer is dead or wedged. The alternative
    is the failure this exists to prevent: the gloo ``cpu_group`` these control
    collectives run on carries a two-hour default timeout, so without a bound
    the survivor parks in ``all_reduce`` for hours after its peer dies of OOM,
    with nothing in the log naming the reason.
    """


# Poll schedule for bounded_wait. The spin window covers the latency of a
# healthy CPU collective between local ranks, so the sleep path is only ever
# reached once something is actually wrong.
COLLECTIVE_POLL_SPINS = 512
COLLECTIVE_POLL_MIN_S = 0.0005
COLLECTIVE_POLL_MAX_S = 0.05


def collective_rank_desc(holder: Any) -> str:
    """Describe the rank that is about to wait, for timeout messages.

    Defensive throughout: this runs on the failure path of a process that is
    already in trouble, and a missing attribute or an uninitialized default
    process group must never mask the timeout it is trying to report.
    """
    parts = []
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            parts.append(f"global_rank={torch.distributed.get_rank()}")
    except Exception:  # pragma: no cover - diagnostics must not raise
        pass
    pp_rank = getattr(holder, "pp_rank", None)
    pp_size = getattr(holder, "pp_size", None)
    if pp_rank is not None:
        parts.append(f"pp_rank={pp_rank}" + (f"/{pp_size}" if pp_size else ""))
    tp_world_size = getattr(holder, "tp_world_size", None)
    if tp_world_size is not None:
        parts.append(f"tp_world_size={tp_world_size}")
    return " ".join(parts) or "rank=unknown"


def bounded_wait(
    work: Optional[torch.distributed.Work],
    label: str,
    timeout_s: float,
    rank_desc: str = "",
) -> None:
    """Wait for ``work`` with a deadline, or raise a named error.

    ``timeout_s <= 0`` restores the raw blocking ``work.wait()``, which is the
    documented escape hatch of SGLANG_HICACHE_COLLECTIVE_TIMEOUT_S.

    ``work`` may be ``None``: single-rank process groups make torch.distributed
    p2p helpers return ``None`` rather than a ``Work``, and that is a completed
    no-op, not something to wait on.
    """
    if work is None:
        return
    if not timeout_s or timeout_s <= 0:
        work.wait()
        return

    started = time.monotonic()
    deadline = started + timeout_s
    spins = 0
    sleep_s = COLLECTIVE_POLL_MIN_S
    while not work.is_completed():
        spins += 1
        if spins <= COLLECTIVE_POLL_SPINS:
            continue
        now = time.monotonic()
        if now >= deadline:
            raise HiCacheCollectiveTimeoutError(
                f"HiCache control collective '{label}' did not complete "
                f"within {timeout_s:g}s (waited {now - started:.1f}s) on "
                f"[{rank_desc or collective_rank_desc(None)}]. A peer rank is "
                "dead or wedged; this rank aborts instead of blocking in the "
                "collective until the process-group timeout (2h) expires. "
                "Adjust the bound with SGLANG_HICACHE_COLLECTIVE_TIMEOUT_S."
            )
        time.sleep(sleep_s)
        sleep_s = min(sleep_s * 2, COLLECTIVE_POLL_MAX_S)
    work.wait()


def bounded_recv(
    tensor: torch.Tensor,
    *,
    group,
    group_src: int,
    tag: int,
    label: str,
    timeout_s: float,
    rank_desc: str = "",
) -> None:
    """Blocking receive with a deadline.

    ``torch.distributed.recv`` has no async form and no timeout, so it cannot be
    bounded in place. ``irecv`` posts the identical operation and returns a
    ``Work`` that ``bounded_wait`` can poll; its signature carries the same
    ``group_src``/``tag``/``group`` arguments, and ``recv`` is itself implemented
    as a posted receive that is immediately waited on. Message matching is by
    (source, tag, group) in posting order, and this function posts exactly one
    receive and does not return until it has completed, so the substitution
    preserves the ordering semantics of the ``recv`` it replaces.

    On timeout the receive is still outstanding and ``tensor`` remains
    registered with the backend as its destination buffer. That is only
    acceptable because the raised error is fatal: the scheduler does not catch
    ``HiCacheCollectiveError`` and the process is on its way down. Do not
    "recover" from this exception and reuse ``tensor``.
    """
    work = torch.distributed.irecv(
        tensor,
        group_src=group_src,
        group=group,
        tag=tag,
    )
    bounded_wait(work, label, timeout_s, rank_desc)
