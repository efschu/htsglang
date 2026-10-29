from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union

import msgspec

if TYPE_CHECKING:
    from flliper.srt.mem_cache.radix_cache import TreeNode

logger = logging.getLogger(__name__)


class EvictionStrategy(ABC):
    @abstractmethod
    def get_priority(self, node: TreeNode) -> Union[float, Tuple]:
        pass

    def note_waiting_queue(self, waiting_queue: Sequence[Any]) -> None:
        """Scheduler hint once per round: the current waiting queue (order = admission order).
        Only queue-aware strategies use it; the others ignore it."""

    def note_eviction_run(self) -> None:
        """Called once at the start of every eviction pass (device or host), before the heap is built."""


class LRUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return node.last_access_time


class LFUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        return (node.hit_count, node.last_access_time)


class FIFOStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return node.creation_time


class MRUStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return -node.last_access_time


class FILOStrategy(EvictionStrategy):
    def get_priority(self, node: TreeNode) -> float:
        return -node.creation_time


class PriorityStrategy(EvictionStrategy):
    """Priority-aware eviction: lower priority values evicted first, then LRU within same priority."""

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        # Return (priority, last_access_time) so lower priority nodes are evicted first
        return (node.priority, node.last_access_time)


class SLRUStrategy(EvictionStrategy):
    def __init__(self, protected_threshold: int = 2):
        self.protected_threshold = protected_threshold

    def get_priority(self, node: TreeNode) -> Tuple[int, float]:
        # Priority Logic:
        # Smaller value = Evicted earlier.
        #
        # Segment 0 (Probationary): hit_count < threshold
        # Segment 1 (Protected): hit_count >= threshold
        #
        # Tuple comparison: (segment, last_access_time)
        # Nodes in segment 0 will always be evicted before segment 1.
        # Inside the same segment, older nodes (smaller time) are evicted first.

        is_protected = 1 if node.hit_count >= self.protected_threshold else 0
        return (is_protected, node.last_access_time)


# --------------------------------------------------------------------------------------------------
# QUEUE-AWARE (Belady over the waiting queue) -- 27b-queue-evict-1029
#
# Key (smaller = evicted first):
#   (0, 0, last_access_time)      no waiting request has the node on its matched path
#   (1, -pos, last_access_time)   pos = waiting-queue position of the EARLIEST waiting user
# i.e. first everything no waiter needs (LRU among them), then the waiters' prefixes, the one the
# queue needs LATEST first; equal keys fall back to LRU. With an empty queue the order is LRU's order
# exactly. Only WHICH entries leave changes -- nothing that is computed.
#
# The waiting-queue view is NOT walked per eviction: the scheduler hands the queue over once per
# round (note_waiting_queue), and the marks are rebuilt at the start of an eviction pass only when the
# queue changed since the last build. A build walks each waiting request's matched path from its
# deepest matched node (best_match_node, else last_node) towards the root and stops at the first node
# an earlier waiter already marked, so its cost is the size of the UNION of the waiting paths, capped.
# Rank uniformity: the queue order is group-identical (#823 W9) and the walk reads the rank-local tree,
# exactly as LRU's last_access_time does -- no new divergence source, no collective.
# --------------------------------------------------------------------------------------------------

QUEUE_AWARE_LOG_EVERY_RUNS = 256
QUEUE_AWARE_MAX_MARK_VISITS = 1 << 16


class QueueAwareStats(msgspec.Struct):
    runs: int = 0
    # passes in which at least one candidate carried a waiter key: the ONLY passes whose order can
    # differ from LRU (without a marked candidate the key is monotone in last_access_time)
    runs_protecting: int = 0
    # passes in which a protected candidate is OLDER than an unprotected one: LRU would have taken the
    # protected one first, queue_aware takes the unprotected one first -- the order provably differs
    runs_inverted: int = 0
    protected_keys: int = 0
    plain_keys: int = 0
    rebuilds: int = 0
    marked_nodes: int = 0
    capped_rebuilds: int = 0


class QueueMarks(msgspec.Struct, frozen=True):
    first_use: Dict[int, int]
    visits: int
    capped: bool


def queue_signature(waiting_queue: Sequence[Any]) -> Tuple[int, int, int, int]:
    """O(1) change detector: identity of the list, its length and its two ends."""
    if not waiting_queue:
        return (id(waiting_queue), 0, 0, 0)
    return (id(waiting_queue), len(waiting_queue), id(waiting_queue[0]), id(waiting_queue[-1]))


def _deepest_matched_node(req: Any) -> Any:
    if req.best_match_node is not None:
        return req.best_match_node
    return req.last_node


def build_queue_marks(*, waiting_queue: Sequence[Any], max_visits: int) -> QueueMarks:
    """node.id -> position of the earliest waiting request whose matched path contains the node.

    Requests are visited in queue order, so the first mark a node receives is its minimum; a walk
    stops at the first already-marked node (its ancestors carry an equal or smaller position)."""
    first_use: Dict[int, int] = {}
    visits = 0
    for pos, req in enumerate(waiting_queue):
        node = _deepest_matched_node(req)
        while node is not None and node.parent is not None:
            if node.id in first_use:
                break
            if visits >= max_visits:
                return QueueMarks(first_use=first_use, visits=visits, capped=True)
            first_use[node.id] = pos
            visits += 1
            node = node.parent
    return QueueMarks(first_use=first_use, visits=visits, capped=False)


class QueueAwareStrategy(EvictionStrategy):
    """Belady over the waiting queue, LRU everywhere else (see the block comment above)."""

    def __init__(
        self,
        *,
        log_every_runs: int = QUEUE_AWARE_LOG_EVERY_RUNS,
        max_mark_visits: int = QUEUE_AWARE_MAX_MARK_VISITS,
    ):
        self._log_every_runs = log_every_runs
        self._max_mark_visits = max_mark_visits
        self._queue: Sequence[Any] = ()
        self._dirty = False
        self._built_sig: Optional[Tuple[int, int, int, int]] = None
        self._first_use: Dict[int, int] = {}
        self._run_open = False
        self._run_protected = 0
        self._run_min_protected_t: Optional[float] = None
        self._run_max_plain_t: Optional[float] = None
        self.stats = QueueAwareStats()

    def note_waiting_queue(self, waiting_queue: Sequence[Any]) -> None:
        self._queue = waiting_queue
        self._dirty = True

    def note_eviction_run(self) -> None:
        self._close_run()
        sig = queue_signature(self._queue)
        if self._dirty or sig != self._built_sig:
            self._rebuild(sig=sig)
        self._run_open = True

    def get_priority(self, node: TreeNode) -> Tuple[int, int, float]:
        t = node.last_access_time
        pos = self._first_use.get(node.id)
        if pos is None:
            self.stats.plain_keys += 1
            if self._run_max_plain_t is None or t > self._run_max_plain_t:
                self._run_max_plain_t = t
            return (0, 0, t)
        self.stats.protected_keys += 1
        self._run_protected += 1
        if self._run_min_protected_t is None or t < self._run_min_protected_t:
            self._run_min_protected_t = t
        return (1, -pos, t)

    def _rebuild(self, *, sig: Tuple[int, int, int, int]) -> None:
        marks = build_queue_marks(waiting_queue=self._queue, max_visits=self._max_mark_visits)
        self._first_use = marks.first_use
        self._built_sig = sig
        self._dirty = False
        self.stats.rebuilds += 1
        self.stats.marked_nodes = len(marks.first_use)
        self.stats.capped_rebuilds += int(marks.capped)

    def _close_run(self) -> None:
        if not self._run_open:
            return
        self._run_open = False
        self.stats.runs += 1
        self.stats.runs_protecting += int(self._run_protected > 0)
        self.stats.runs_inverted += int(
            self._run_min_protected_t is not None
            and self._run_max_plain_t is not None
            and self._run_min_protected_t < self._run_max_plain_t
        )
        self._run_protected = 0
        self._run_min_protected_t = None
        self._run_max_plain_t = None
        if self._log_every_runs > 0 and self.stats.runs % self._log_every_runs == 0:
            _log_queue_aware_stats(self.stats)


def _log_queue_aware_stats(stats: QueueAwareStats) -> None:
    share = 100.0 * stats.runs_inverted / max(1, stats.runs)
    logger.info(
        "EVICT-QUEUE-AWARE runs=%d lru_order_inverted=%d (%.1f%%) protecting=%d protected_keys=%d "
        "plain_keys=%d rebuilds=%d marked_nodes=%d capped_rebuilds=%d",
        stats.runs,
        stats.runs_inverted,
        share,
        stats.runs_protecting,
        stats.protected_keys,
        stats.plain_keys,
        stats.rebuilds,
        stats.marked_nodes,
        stats.capped_rebuilds,
    )
