"""#703: push warm prefixes to the canonical store BEFORE a phase flip.

THE GAP THIS CLOSES, traced rather than assumed.

Nothing in the flip path touches HiCache. The cutover
(``phase_flip_runtime.py``) rebuilds topology, groups, loop state and the
worker stack; it never backs up, flushes, detaches or resets the cache. That
is deliberate -- device rows survive the flip, because the live row set (radix
tree values UNION parked requests' rows) is relocated between the two phase
pools BY ROW ID. So the device tier rides through.

The HOST tier does not, and the reason is structural rather than a missing
call. There are two device KV pools: the boot PP stack's and the flip's TP
stack's (``phase_flip_runtime`` builds both, and the TP stack builds with
``pp_size = 1``, so it spans ALL attention layers while a PP stage spans only
its own 7/5/4). ``HiRadixCache`` binds its host pool to whichever device pool
existed at construction -- the boot one -- and the scheduler's allocator is
likewise assigned once and never reassigned. In the phase that did not build
it, therefore, the host tier mirrors a pool that is not the live one.

So a prefix's only way across the flip is the geometry-free STORE (#706): the
disk tier, whose keys carry content alone and whose pages are cut at read time
for whichever geometry asks. Getting a prefix there is what this module does.

WHY A HOOK IS NEEDED AT ALL -- the write-back timing gap:

* ``write_through``            device->host on the FIRST insert (eager);
* ``write_through_selective``  on the second hit;
* ``write_back``               ONLY under eviction pressure.

and host->storage is a separate asynchronous stage: the backup thread picks
entries off ``backup_queue`` and acknowledges on ``ack_backup_queue``. Under
``write_back`` -- and under either write-through policy for anything below
threshold or still in flight -- a warm prefix can sit in device memory,
never staged, indefinitely. A flip does not change that: it neither forces
the staging nor waits for the queue to drain.

There is no existing API to lean on. ``writing_check`` only completes writes
already issued. ``detach_storage_backend`` DROPS pending backups rather than
finishing them. ``flush_cache`` is a wipe, not a write-back, despite the name.
``HiCacheController.reset`` clears every queue outright, so pending backup work
is lost, not drained.

This hook therefore does the one thing none of them does: it stages the tree's
un-backed prefixes to host, lets the storage stage run, and WAITS for the
acknowledgements -- under a deadline, because an unbounded wait at the flip
seam is the #630 wedge shape and this runs with requests parked.

It refuses loudly without the canonical store. Writing pages named by this
phase's geometry, at the seam of a phase change, would buy exactly nothing:
the other phase cannot name them. Better to say so than to spend the IO.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

LOG_PREFIX = "[#703 flip-writeback]"

# Poll interval while waiting for storage acknowledgements.
_POLL_S = 0.005


class FlipWritebackRefused(RuntimeError):
    """The writeback cannot do anything useful in this configuration."""


@dataclasses.dataclass(frozen=True)
class FlipWritebackReport:
    """What the hook actually achieved. Counts, never a verdict."""

    eligible: int  # tree nodes carrying a hash, i.e. persistable
    staged: int  # nodes this call staged device -> host
    already_staged: int  # nodes the normal policy had already staged
    acknowledged: int  # storage backups confirmed before the deadline
    outstanding: int  # storage backups still in flight when time ran out
    elapsed_s: float
    deadline_s: float

    @property
    def complete(self) -> bool:
        return self.outstanding == 0

    @property
    def persisted_nothing(self) -> bool:
        """#783: nothing of this fence is retrievable after the tree is dropped.

        NOT the same question as `complete`, which is `outstanding == 0` and is
        therefore TRUE for a fence that had nothing to do. In W37-G 33 of 39
        cutovers fenced `eligible=0 staged=0 already_staged=0 acked=0
        outstanding=0 elapsed=0.000s` -- complete, cheap, and empty -- and every
        instrument in the tree read them as healthy. `#cached-token: 0` on all
        209 prefill batch lines is the consequence.

        The seam's own guard could not see it either: it warns when
        `_writeback_fence_ms(...) is None`, and an empty fence returns 0.0. A
        cheap fence and a healthy fence are indistinguishable BY COST, because
        the healthy case is also cheap. So this asks about retrievability, which
        is the property the read-through actually depends on, using counts the
        report already carried and nothing read.

        DELIBERATELY NARROW: true only when NEITHER route survives -- no storage
        acknowledgement AND no pre-existing host copy. A fence that acked
        nothing but left `already_staged` nodes behind is NOT claimed here; that
        host copy may still serve the read-through, and that case already draws
        the "deadline reached with backups still in flight" warning. Widening
        this to cover it would double-report one condition and make a
        crying-wolf gate out of the instrument built to replace one.
        """
        return self.acknowledged == 0 and self.already_staged == 0

    def as_log(self) -> str:
        return (
            f"eligible={self.eligible} staged={self.staged} "
            f"already_staged={self.already_staged} acked={self.acknowledged} "
            f"outstanding={self.outstanding} "
            f"elapsed={self.elapsed_s:.3f}s/{self.deadline_s:.3f}s"
        )


def canonical_store_of(tree_cache: Any):
    """The canonical page window a storage backend is running with, or None."""
    controller = getattr(tree_cache, "cache_controller", None)
    backend = getattr(controller, "storage_backend", None)
    return getattr(backend, "canonical_kv_page", None)


def require_canonical_store(tree_cache: Any) -> None:
    """Refuse unless a flip-crossing store is actually configured.

    Three separate ways to have nothing to write to, each named, because each
    has a different fix and a generic "not supported" would hide which one is
    in force.
    """
    if not getattr(tree_cache, "enable_storage", False):
        raise FlipWritebackRefused(
            f"{LOG_PREFIX} no storage tier is attached, so there is nowhere "
            "for a prefix to survive the flip: the host tier mirrors the pool "
            "that built it, which in the other phase is not the live pool. "
            "Attach the file backend (--hicache-storage-backend file)."
        )
    if canonical_store_of(tree_cache) is None:
        raise FlipWritebackRefused(
            f"{LOG_PREFIX} the storage tier is not running the #706 canonical "
            "page format, so its keys carry this phase's geometry "
            "(_{tp_rank}_{tp_size} and _{pp_size}_{pp_rank}). Pages written at "
            "the flip seam would be unreadable in the phase they were written "
            "for. Enable --phase-flip-canonical-kv-page, or leave the "
            "writeback off; spending the IO for keys the other phase cannot "
            "name is the one outcome worth refusing."
        )


def _hashed_nodes(tree_cache: Any) -> list:
    """Persistable nodes, PARENT FIRST.

    Parent order is not cosmetic: the backup invariant is that backed-up nodes
    form a contiguous prefix from the root, and a child staged before its
    parent is a gap the normal path would have skipped.

    #841: parent order alone was never enough, and the walk below shows why --
    a node WITHOUT a ``hash_value`` is skipped from the list while the walk
    still descends into its children. So an unhashed node's children reach the
    staging loop with their parent absent from it entirely, and the loop's
    ``write_back=True`` also disarms ``write_backup``'s own parent gate. Order
    is preserved here; the law itself is enforced at the call site.
    """
    root = getattr(tree_cache, "root_node", None)
    if root is None:
        return []
    out: list = []

    def _walk(node) -> None:
        for child in list(getattr(node, "children", {}).values()):
            if getattr(child, "hash_value", None):
                out.append(child)
            _walk(child)

    _walk(root)
    return out


def flip_writeback(
    tree_cache: Any,
    *,
    deadline_s: float = 2.0,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> FlipWritebackReport:
    """Stage the tree's warm prefixes to the canonical store, bounded.

    Called at the flip's quiescent seam, with requests parked, BEFORE the
    cutover -- while the pool this cache is bound to is still the live one.
    That ordering is the whole point: after the cutover the same copy would
    read a pool the model is no longer writing into.

    The deadline is a hard bound, not a target. Whatever is not acknowledged in
    time stays in flight; the report says so and the caller decides. Nothing
    here aborts the flip -- a prefix that misses the store is a cache miss
    later, which is a cost, while a flip that stalls behind an unbounded wait
    is a wedge.
    """
    require_canonical_store(tree_cache)

    started = now()
    deadline = started + float(deadline_s)

    nodes = _hashed_nodes(tree_cache)
    root = getattr(tree_cache, "root_node", None)
    staged = 0
    already = 0
    skipped_unbacked_parent = 0
    for node in nodes:
        if getattr(node, "backuped", False):
            already += 1
            continue
        # #841: THE CONTIGUOUS-BACKUP LAW. This loop used to reason that
        # "the parent is in this same list, earlier, so the invariant holds
        # by construction", and staged the child regardless with
        # write_back=True -- which ALSO disarms write_backup's own parent gate
        # at unified_radix_cache.py:1943-1948. Both halves of that reasoning
        # fail: an unhashed parent is never in the list at all (see
        # _hashed_nodes), and a parent that IS in the list can still have had
        # its own write_backup refused a moment earlier -- the mamba pin
        # budget, the rank-uniform host floor and the staging ring each return
        # 0 without raising.
        #
        # A child staged over that gap is not merely a lost page. Under any
        # write policy other than `write_back` the tree's own idle-path
        # sanity check enforces the law and ABORTS every rank on it, and an
        # un-backed parent above a backed child can be deleted by a device
        # eviction, orphaning the subtree in the host ledger. That is the
        # window-5 crash. The store being content-addressed per page does not
        # make the tree's ledger consistent.
        parent = getattr(node, "parent", None)
        if (
            parent is not None
            and parent is not root
            and not getattr(parent, "backuped", False)
        ):
            skipped_unbacked_parent += 1
            continue
        try:
            written = tree_cache.write_backup(node, write_back=True)
        except Exception as e:  # an instrument at a seam, never a gate
            logger.warning("%s staging node failed: %s", LOG_PREFIX, e)
            continue
        if written:
            staged += 1
    if skipped_unbacked_parent:
        logger.info(
            "%s skipped %d node(s) whose parent carries no host copy; staging "
            "them would break the contiguous-backup law the tree asserts on "
            "the idle path.",
            LOG_PREFIX,
            skipped_unbacked_parent,
        )

    # Complete the device->host copies. This is also what triggers the
    # host->storage stage: the write ack path calls write_backup_storage.
    try:
        tree_cache.writing_check(write_back=True)
    except Exception as e:
        logger.warning("%s writing_check failed: %s", LOG_PREFIX, e)

    acknowledged, outstanding = _await_storage_acks(
        tree_cache, deadline=deadline, now=now, sleep=sleep
    )

    report = FlipWritebackReport(
        eligible=len(nodes),
        staged=staged,
        already_staged=already,
        acknowledged=acknowledged,
        outstanding=outstanding,
        elapsed_s=now() - started,
        deadline_s=float(deadline_s),
    )
    if report.complete:
        logger.info("%s %s", LOG_PREFIX, report.as_log())
    else:
        # Not an error: the bound did its job. Loud enough to be attributable
        # if post-flip hit rates disappoint.
        logger.warning(
            "%s deadline reached with backups still in flight; those prefixes "
            "will miss after the flip. %s",
            LOG_PREFIX,
            report.as_log(),
        )
    return report


def _await_storage_acks(
    tree_cache: Any,
    *,
    deadline: float,
    now: Callable[[], float],
    sleep: Callable[[float], None],
) -> tuple[int, int]:
    """Drain backup acknowledgements until the tree is quiet or time is up.

    Drains LOCALLY. The cross-rank variant all-reduces the queue sizes, and a
    collective issued inside the flip seam is the #630 wedge shape; every rank
    runs this hook at the same seam anyway, so a rank draining its own acks
    needs no agreement with its peers.

    #872: BOTH LOOKUPS BELOW ARE LOUD ON A MISS, and that is the point of the
    ticket rather than a courtesy. These are duck-typed capability probes, and
    each one's miss path used to return exactly what a healthy no-op returns --
    "nothing was outstanding", "nothing was acknowledged" -- so a cache class
    that simply does not carry the name was INDISTINGUISHABLE from a cache with
    an empty queue. ``UnifiedRadixCache`` was such a class for the whole of its
    life on this path: it has ``_drain_storage_control_queues_impl`` and had no
    ``_local`` wrapper, so the fence skipped its wait entirely and reported
    ``acked=0`` on every fence of every boot while the guards above and the
    #871 streak alarm all read healthy.

    Naming the method on one more class fixes today's cache and leaves the next
    one to fail the same invisible way, so the probe itself now reports. The
    fence still does not raise -- a missed prefix is a later miss, never a
    wrong answer, and a flip may not die on an instrument -- but the condition
    is no longer silent, and it is no longer reported as completeness.
    """
    ongoing = getattr(tree_cache, "ongoing_backup", None)
    if ongoing is None:
        logger.error(
            "%s #872 UNSERVABLE CACHE: %s carries no `ongoing_backup`, so this "
            "fence cannot see whether ANY backup is in flight and reports the "
            "seam complete without having looked. Every prefix it was supposed "
            "to persist may be lost when the tree is dropped at the cutover, "
            "and the loss appears only as a post-flip cache miss.",
            LOG_PREFIX,
            type(tree_cache).__name__,
        )
        return 0, 0
    before = len(ongoing)
    drain = getattr(tree_cache, "_drain_storage_control_queues_local", None)
    if not callable(drain):
        # Reported as fully OUTSTANDING, never as acknowledged: nothing was
        # drained, so claiming otherwise would hand `persisted_nothing` and
        # `complete` a value neither of them earned. This is what makes the
        # #871 streak alarm able to see the condition at all.
        logger.error(
            "%s #872 UNSERVABLE CACHE: %s has no callable "
            "`_drain_storage_control_queues_local`, so the writeback fence "
            "cannot wait for storage acknowledgements and did not wait: %d "
            "backup(s) are left in flight and the tree is dropped at the "
            "cutover regardless. Add the local-drain wrapper to that class "
            "(all it needs is `_drain_storage_control_queues_impl` with None "
            "limits, as UnifiedRadixCache/HiRadixCache/HiMambaRadixCache do) "
            "-- a fence that never waits is retention that never happens.",
            LOG_PREFIX,
            type(tree_cache).__name__,
            before,
        )
        return 0, before
    while True:
        try:
            drain()
        except Exception as e:
            logger.warning("%s draining backup acks failed: %s", LOG_PREFIX, e)
            break
        if not ongoing:
            break
        if now() >= deadline:
            break
        sleep(_POLL_S)
    outstanding = len(ongoing)
    return max(0, before - outstanding), outstanding


def maybe_flip_writeback(
    scheduler: Any, *, deadline_s: Optional[float] = None
) -> Optional[FlipWritebackReport]:
    """Flag-gated entry point for the flip seam.

    Returns None when the feature is off -- the default, and byte-identical to
    not calling it. When it is ON and the store cannot serve the flip, the
    refusal propagates: an operator who asked for retention across the flip and
    silently got none is the failure mode worth being loud about.
    """
    server_args = getattr(scheduler, "server_args", None)
    if not getattr(server_args, "phase_flip_writeback", False):
        return None
    tree_cache = getattr(scheduler, "tree_cache", None)
    if tree_cache is None:
        raise FlipWritebackRefused(
            f"{LOG_PREFIX} --phase-flip-writeback is set but this scheduler has "
            "no tree cache to write back."
        )
    if deadline_s is None:
        deadline_s = float(
            getattr(server_args, "phase_flip_writeback_deadline_s", None) or 2.0
        )
    return flip_writeback(tree_cache, deadline_s=deadline_s)
