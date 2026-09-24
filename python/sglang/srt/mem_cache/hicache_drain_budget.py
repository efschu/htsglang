"""Bounded HiCache storage-queue drain (27B, 24.09., default OFF).

User order 2026-09-24: HiCache work "darf nie den laufenden prefill oder decode
beeintraechtigen"; what has to wait for synchronisation or copying goes through
a small extra buffer, and host work never blocks the round.

THE MEASURED CAUSE (boot xsn429, 682cae209e). `check_hicache_events` drains the
storage control queues on the scheduler thread once per round. The spikes of
its drain term (P PP0: 171.96 / 171.40 / 401.11 ms, 52-60 ms on the 32k ladder;
D: 8-14 ms) follow every terminated or revoked store prefetch, linearly in the
span it gives back: ~1.6-1.8 us per token on metal (8189 -> 13.3 ms, 32766 ->
59.8 ms, 94795 -> 171.9 ms, 255726 -> 401.1 ms; D 4314 -> 8.3-8.8 ms). The span
is released through `append_host_mem_release`, and the hybrid controller splits
it by the host pool's page size -- 1 on the arena host pool -- so a 95k-token
span becomes 95k queue entries. `_drain_release` then pays a Queue.get_nowait,
a list append and a len() per entry (~1.5 us) plus a torch.cat over 95k one-row
tensors, to free rows `HostPoolGroup.free` drops anyway (placeholders and arena
ids are outside the staging range; #718 index axis). Desk, real objects, 94795
tokens: pop loop 143-156 ms, cat 32-42 ms, unique 2-4 ms, free 0.5-6 ms; the
producer (the prefetch thread on metal, GIL held) 142-184 ms.

THE FIX, behind SGLANG_HICACHE_DRAIN_BUDGET=<tokens per round> (0/unset = off,
the unchanged path):

1. A release is ONE queue entry (the whole index tensor), not one per page:
   the producer and the drain's per-entry cost disappear.
2. The steady-state drain takes at most <tokens> rows (and at most
   ENTRY_BUDGET entries) per release queue and round. An entry larger than the
   remaining allowance is cut on a page boundary; the rest goes back to the
   FRONT of its queue. The queue itself is the extra buffer: every path that
   settles, clears or fence-drains the queue (rebind settle, reset, the local
   drain, detach) sees the rest exactly as before -- no second holder exists.
3. At most SGLANG_HICACHE_DRAIN_ACK_BUDGET storage acks per round (default 32
   while the budget is on, 0 = all); the rest stays queued and drains in the
   next rounds.

RANK-UNIFORM BY CONSTRUCTION: the allowances are constants, applied to the
group-MIN entry counts the agreement already produced, in queue order -- no
wall clock enters a decision, so a TP group drains the same rows and acks on
the same rounds. Nothing is dropped: a cut entry's rest is re-queued before the
head is freed, and acks beyond the allowance never leave their queue. Host rows
are recycled no earlier than before (a release is only ever delayed).
"""

from __future__ import annotations

import os
from queue import Empty
from typing import Optional

ENV_BUDGET = "SGLANG_HICACHE_DRAIN_BUDGET"
ENV_ACK_BUDGET = "SGLANG_HICACHE_DRAIN_ACK_BUDGET"
#: acks per round while the budget is on and ENV_ACK_BUDGET is unset
DEFAULT_ACK_BUDGET = 32
#: queue entries per release queue and round -- bounds the per-entry pop/cat
#: cost for producers that still enqueue page-sized entries
ENTRY_BUDGET = 512


def _int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, str(default)) or default))
    except ValueError:
        return default


def drain_budget_tokens() -> int:
    """Rows per release queue and round; 0 = the unbounded (unchanged) drain."""
    return _int_env(ENV_BUDGET, 0)


def coalesce_host_releases() -> bool:
    """One queue entry per release instead of one per page (switch on)."""
    return drain_budget_tokens() > 0


def drain_ack_budget() -> int:
    """Storage acks per round while the budget is on; 0 = all agreed acks."""
    if drain_budget_tokens() <= 0:
        return 0
    return _int_env(ENV_ACK_BUDGET, DEFAULT_ACK_BUDGET)


def requeue_front(q, item) -> bool:
    """Put ``item`` back at the FRONT of a ``queue.Queue`` (put() semantics:
    the task counter and the not-empty wake-up). False when ``q`` is not a
    ``queue.Queue`` -- the caller then keeps the entry whole."""
    mutex = getattr(q, "mutex", None)
    dq = getattr(q, "queue", None)
    if mutex is None or not hasattr(dq, "appendleft"):
        return False
    with mutex:
        dq.appendleft(item)
        q.unfinished_tasks += 1
        q.not_empty.notify()
    return True


class ReleaseBudget:
    """The per-round allowance of one steady-state drain pass."""

    __slots__ = ("tokens", "entries")

    def __init__(self, tokens: int, entries: int = ENTRY_BUDGET):
        self.tokens = int(tokens)
        self.entries = int(entries)

    def take(self, q, limit: Optional[int], page_size: int = 1):
        """Take up to ``limit`` entries (the agreed count; None = all) of ``q``,
        at most ``self.tokens`` rows and ``self.entries`` entries. Returns
        (entries_taken_as_tensors, rows). A cut entry's rest is re-queued at
        the front BEFORE its head is handed out, so no row is ever held
        outside the queue by this call."""
        page = max(1, int(page_size or 1))
        items = []
        rows = 0
        whole = 0
        tokens = self.tokens
        while tokens > 0 and whole < self.entries and (limit is None or whole < limit):
            try:
                item = q.get_nowait()
            except Empty:
                break
            k = int(item.numel())
            if k > tokens:
                cut = tokens - tokens % page
                if cut > 0 and requeue_front(q, item[cut:]):
                    items.append(item[:cut])
                    rows += cut
                elif not requeue_front(q, item):
                    # not a queue.Queue: nothing can be put back -- take it whole
                    items.append(item)
                    rows += k
                break
            items.append(item)
            rows += k
            tokens -= k
            whole += 1
        return items, rows
