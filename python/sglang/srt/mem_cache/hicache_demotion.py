"""#703: persist a warm prefix ON THE WAY OUT, instead of losing it.

WHAT TODAY'S LINE ACTUALLY DOES, at the code, because the answer is nuanced
and the nuance is the whole ticket:

* Under the DEFAULT ``write_through`` policy, a node is staged device->host at
  insert (``_inc_hit_count`` -> ``write_backup``), and the storage write rides
  the write-through ACK: ``_finish_write_through_ack`` calls
  ``write_backup_storage`` when storage is enabled
  (``hiradix_cache.py``). So disk writes are INSERT-driven, not eviction-driven.
* Device eviction (``_evict_write_through``) then has two outcomes:
  ``_evict_backuped`` DEMOTES a node that already has a host copy, and
  ``_evict_regular`` DROPS one that does not -- freed, unlinked, gone. Nothing
  in that path writes to storage.
* Host eviction (``evict_host``) frees the host rows and unlinks the node.
  Nothing in that path writes to storage either.
* Under ``write_back`` staging happens only at eviction, and when the host pool
  is full ``write_backup`` returns 0 and the subtree is dropped outright.

So the honest premise: there IS a path to disk, but it is the insert-time ack
path. Eviction is where prefixes are LOST -- specifically (a) a node that never
reached the host tier (below threshold, or host full) and (b) any node whose
host copy is evicted while it is not already on disk (storage attached late, an
earlier storage write refused by the evictor's cap, a backend that was detached
and reattached). For those, eviction is pure loss today.

THIS MODULE closes exactly that gap: at the two eviction seams, a prefix that
is about to disappear is written to the geometry-neutral disk tier first, so it
can come back through the #706 read-cut under the same content key.

THREE PROPERTIES IT MUST HAVE, and they are the design:

1. IT NEVER BLOCKS THE SERVING PATH. Demotion enqueues onto the controller's
   existing backup queue -- the same asynchronous stage insert-time writes use
   -- and returns. Eviction runs while the scheduler is trying to admit work;
   a synchronous disk write there would turn memory pressure into latency.
2. IT IS BOUNDED, AND DROPS LOUDLY. Beyond a cap on outstanding demotions the
   request is DROPPED and counted, never queued without limit. A dropped
   demotion is a lost cache entry -- a later miss -- and never corruption,
   because the page it would have written is content-addressed and simply
   absent. Unbounded queueing under eviction pressure is how memory pressure
   becomes an OOM.
3. IT IS OFF BY DEFAULT. The default path stays byte-identical; a store that
   was not asked for this does not start taking eviction-time writes.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

LOG_PREFIX = "[#703 demote]"


@dataclasses.dataclass
class DemotionStats:
    """Counters. ``dropped_backpressure`` is the one to watch."""

    demoted: int = 0
    dropped_backpressure: int = 0
    skipped_not_persistable: int = 0
    skipped_no_storage: int = 0
    failed: int = 0

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


_STATS = DemotionStats()
_LOCK = threading.Lock()
_WARNED_BACKPRESSURE = False


def stats() -> DemotionStats:
    return _STATS


def reset_stats() -> None:
    global _WARNED_BACKPRESSURE
    with _LOCK:
        for field in dataclasses.fields(DemotionStats):
            setattr(_STATS, field.name, 0)
        _WARNED_BACKPRESSURE = False


def enabled() -> bool:
    """Off unless asked for. See property 3 in the module docstring."""
    from sglang.srt.environ import envs

    return int(envs.SGLANG_HICACHE_DEMOTE_ON_EVICT.get() or 0) > 0


def _outstanding_cap() -> int:
    from sglang.srt.environ import envs

    return int(envs.SGLANG_HICACHE_DEMOTE_ON_EVICT.get() or 0)


def is_persistable(node: Any) -> bool:
    """A node can only be demoted if it has bytes AND a content key.

    ``host_value`` is what the storage write reads and ``hash_value`` is what
    names it. A node missing either is not a candidate -- not a failure.
    """
    return bool(getattr(node, "host_value", None) is not None) and bool(
        getattr(node, "hash_value", None)
    )


def demote(tree_cache: Any, node: Any, *, reason: str) -> bool:
    """Write an about-to-be-lost prefix to the disk tier. Never raises.

    Returns True when the write was ENQUEUED (not when it landed -- the
    controller's backup thread owns that, and waiting for it here is exactly
    the blocking this must not do).
    """
    if not enabled():
        return False
    if not getattr(tree_cache, "enable_storage", False):
        with _LOCK:
            _STATS.skipped_no_storage += 1
        return False
    if not is_persistable(node):
        with _LOCK:
            _STATS.skipped_not_persistable += 1
        return False

    ongoing = getattr(tree_cache, "ongoing_backup", None)
    cap = _outstanding_cap()
    if ongoing is not None and len(ongoing) >= cap:
        _count_drop(reason, len(ongoing), cap)
        return False

    try:
        tree_cache.write_backup_storage(node)
    except Exception as e:
        # Eviction must proceed regardless: this runs inside the allocator's
        # pressure path, and a raise here would turn "the cache lost an entry"
        # into "the request failed".
        with _LOCK:
            _STATS.failed += 1
        logger.warning("%s demotion failed (%s): %s", LOG_PREFIX, reason, e)
        return False

    with _LOCK:
        _STATS.demoted += 1
    return True


def _count_drop(reason: str, outstanding: int, cap: int) -> None:
    global _WARNED_BACKPRESSURE
    with _LOCK:
        _STATS.dropped_backpressure += 1
        first = not _WARNED_BACKPRESSURE
        _WARNED_BACKPRESSURE = True
        total = _STATS.dropped_backpressure
    if first:
        # Once, loudly, then counted: this fires under pressure, where a line
        # per eviction would bury the log it is trying to inform.
        logger.warning(
            "%s dropping demotions: %d already outstanding at the cap of %d "
            "(%s). A dropped demotion is a LOST CACHE ENTRY -- a later miss, "
            "never corruption, since the page it would have written is "
            "content-addressed and simply absent. Raise "
            "SGLANG_HICACHE_DEMOTE_ON_EVICT if the disk tier can keep up.",
            LOG_PREFIX,
            outstanding,
            cap,
            reason,
        )
    elif total % 1000 == 0:
        logger.warning("%s %d demotions dropped so far.", LOG_PREFIX, total)


def demote_before_host_evict(tree_cache: Any, node: Any) -> bool:
    """Last chance: the host copy is about to be freed and the node unlinked.

    Called BEFORE ``cache_controller.evict_host``, because that frees the very
    bytes the storage write reads.
    """
    return demote(tree_cache, node, reason="host-evict")


def demote_on_device_evict(tree_cache: Any, node: Any) -> bool:
    """A node demoted device->host that may never have reached disk.

    The insert-time ack writes storage once; a node whose storage write was
    refused then (evictor cap, backend detached, storage attached later) is
    never retried, and this is where that is noticed.
    """
    return demote(tree_cache, node, reason="device-evict")
