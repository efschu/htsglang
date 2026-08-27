"""
Copyright 2026 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Slot allocator for the Mamba state pool.

Mamba caches one whole state tensor per request, so the allocator hands out
fixed-size slots (1 per request) rather than paged token KV indices.  The
underlying tensor storage lives in ``MambaPool``; this class owns only the
free-slot bookkeeping.
"""

from __future__ import annotations

import logging
import traceback
from typing import Iterator, Optional

import torch

logger = logging.getLogger(__name__)


class MambaSlotDoubleFree(RuntimeError):
    """A Mamba state slot was returned to the free list twice.

    #924. Named rather than generic because the free list is a bare
    ``torch.cat``: a duplicate is REPRESENTABLE, and the next ``alloc`` hands
    the same state slot to two requests. Two requests sharing a Mamba state is
    a wrong answer that never crashes -- the class the on-idle ledger only ever
    saw minutes later, as an unattributable ``available=23`` on a 20-slot pool,
    with the duplicate already collapsed by the diagnosis's own ``set()``.
    """


class MambaSlotAllocator:
    """Manages the free-list of Mamba pool slot indices.

    Unlike ``BaseTokenToKVPoolAllocator`` which is designed for per-token KV
    pages, Mamba slots are request-level (typically 1 slot per request).
    We keep the interface minimal and do NOT inherit the KV base class.
    """

    def __init__(self, size: int, device: str):
        self.size = size
        self.device = device
        # Active preallocated batch for `alloc_group_begin` / `alloc_group_end`.
        # When non-None, `alloc(1)` consumes the next slot from this iterator
        # instead of calling `_do_alloc(1)` per request. Reset to None outside
        # a group window so `alloc` falls through to the per-call path.
        self._alloc_iter: Optional[Iterator] = None
        self.clear()

    def available_size(self) -> int:
        return len(self.free_slots)

    def schedulable_available_size(self) -> int:
        """Planner-facing free count. Identity to ``available_size`` for the
        static pool (slot-count and byte-coordinated views coincide); the shared
        ``UnifiedMambaSlotAllocator`` overrides it with the byte-coordinated view.
        Lets ``alloc_req_slots`` call it uniformly without a getattr fallback."""
        return self.available_size()

    def alloc_group_begin(self, num_reqs: int):
        """Pre-allocate a batch of slots for match_prefix to amortize overhead."""
        self._alloc_iter = None
        if num_reqs > 0:
            result = self._do_alloc(num_reqs)
            if result is not None:
                self._alloc_iter = iter(result.split(1))

    def alloc_group_end(self):
        """Return any unused pre-allocated slots from the current group."""
        if self._alloc_iter is not None:
            remaining = list(self._alloc_iter)
            if remaining:
                self.free(torch.cat(remaining))
        self._alloc_iter = None

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if self._alloc_iter is not None and need_size == 1:
            slot = next(self._alloc_iter, None)
            if slot is not None:
                return slot
        return self._do_alloc(need_size)

    def _do_alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size > len(self.free_slots):
            return None
        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        self.slot_used[select_index] = True
        return select_index

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return
        # Ids outside the ledger's span (the ``-1`` sentinel the ping-pong
        # track buffer carries, filtered at memory_pool.py:2213 but not
        # everywhere) would WRAP a bool tensor and answer the membership
        # question about the wrong slot. They are excluded from the ledger
        # rather than answered wrongly; the free list itself is unchanged.
        in_ledger = (free_index >= 0) & (free_index < self.slot_used.numel())
        self._refuse_double_free(free_index[in_ledger])
        self.slot_used[free_index[in_ledger]] = False
        self.free_slots = torch.cat((self.free_slots, free_index))

    def _refuse_double_free(self, free_index: torch.Tensor) -> None:
        """#924: say WHO returned a slot that is already free, then raise.

        WHY NOT A SILENT DEDUP. Dropping the duplicate here would keep the
        free list sound and leave the second releaser in place -- the surplus
        would stop being observable while the ownership defect that produced it
        went on running. That is symptom treatment, and the on-idle ledger,
        which is the only thing that ever noticed, would go quiet.

        WHY THE CALLER'S STACK. There are twenty-odd call sites that reach
        ``mamba_allocator.free`` (regular finish, radix eviction, the flip's
        slot union, the HiCache ack drain, abort/retraction, the streaming
        session). The 2026-08-27 specimen inflated by ONE slot at a time and
        the death came minutes later at ``on_idle``, by which point nothing in
        the process still knew which of those it was. The first offender's
        stack is recorded, so ONE boot attributes it instead of three.

        WHY IT RAISES. The sibling with the same job -- ``HostKVCache.free``
        (``pool_host/base.py:359-380``, the #905 hardening) -- logs the same
        diagnosis and then asserts. This one is worse if allowed to continue:
        a doubly-freed KV row is caught by the row-ownership authority, while a
        doubly-freed Mamba slot is handed to two requests and answers both.
        """
        if free_index.numel() == 0:
            return
        already_free = self.slot_used[free_index].logical_not()
        if not bool(already_free.any()):
            return
        offenders = free_index[already_free].tolist()
        n = getattr(self, "_double_free_count", 0) + 1
        self._double_free_count = n
        trace = "".join(traceback.format_stack(limit=12)[:-2])
        if getattr(self, "_first_double_free_trace", None) is None:
            self._first_double_free_trace = trace
        logger.error(
            "#924 MAMBA SLOT DOUBLE FREE: slot(s) %s are already on the free "
            "list and were returned again. The free list is a bare torch.cat, "
            "so a second release is representable and inflates "
            "available_size() by exactly the duplicate count -- which is how "
            "this surfaced, as available=%d on a %d-slot pool at an on_idle "
            "check minutes later, with the duplicate already collapsed by the "
            "diagnosis's own set(). (%d so far.) Releasing caller:\n%s",
            offenders,
            len(self.free_slots) + int(already_free.sum()),
            self.size,
            n,
            trace,
        )
        raise MambaSlotDoubleFree(
            f"mamba slot(s) {offenders} were already free; a second release "
            f"would let alloc() hand one state slot to two requests"
        )

    def clear(self):
        # Slot 0 is reserved as a dummy write target for padded tokens.
        self.free_slots = torch.arange(
            1, self.size + 1, dtype=torch.int64, device=self.device
        )
        # #924 ownership ledger, the shape ``HostKVCache`` already carries:
        # ``free_slots`` alone cannot answer "is this slot already free?"
        # without an O(n) membership scan, and the answer is what separates a
        # legitimate release from a double free. Sized ``size + 1`` because
        # slot 0 is reserved and never handed out; it stays False forever.
        self.slot_used = torch.zeros(
            self.size + 1, dtype=torch.bool, device=self.device
        )
