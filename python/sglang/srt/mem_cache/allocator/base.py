"""
Copyright 2025 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


class BaseTokenToKVPoolAllocator(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        self.size = size
        self.page_size = page_size
        self.dtype = dtype
        self.device = device
        self._kvcache = kvcache
        self.need_sort = need_sort

        self.free_pages = None
        self.release_pages = None
        self.is_not_in_free_group = True
        self.free_group = []
        # Free/clear listeners (T156 task D: the DFLASH small solo draft
        # pool mirrors global slot LIFETIMES through these). Subclasses that
        # implement free()/clear() should call _notify_free/_notify_clear;
        # consumers must verify their allocator class does (the token
        # allocator is wired; see register_free_listener callers).
        self._free_listeners = []
        #: #656 item 16: the DCP owner class new allocations should prefer,
        #: or None. Set through :meth:`set_owner_bias` only.
        self._owner_bias = None

    # -- #657 item 16, the REBALANCE tier: steering, not moving ------------
    #
    # ON THE BASE, NOT ON THE PAGED SUBCLASS, and the reason is a measured
    # one. The scheduler holds ONE allocator for process life -- the PP
    # stack's -- and the PP layout has ``dcp_size == 1``, so with
    # ``page_size == 1`` the chooser builds a plain ``TokenToKVPoolAllocator``
    # (``model_runner_kv_cache_mixin.py:4083``). A bias implemented on the
    # paged class alone armed, logged, and steered NOTHING on this rig: the
    # first boot reported "the active allocator has no owner bias" nine
    # times. Both classes take the HEAD of ``free_pages``, so both can be
    # steered by the same reordering, and the shared base is where that
    # belongs.

    def set_owner_bias(self, bias) -> int:
        """Prefer free slots of one DCP owner class at the head of the list.

        ``bias`` is ``(mod, lo, hi)`` -- the weighted owner rule's
        ``(cp_S, cp_lo, cp_hi)`` for the rank that should ABSORB new KV --
        or ``None`` to steer nothing. Returns how many free slots of that
        class were promoted, so a caller can log a number instead of a
        claim.

        THIS PLACES BYTES; IT DOES NOT MOVE OR FREE ANY. Every slot in the
        free list is already a legal placement, and this only reorders which
        legal one the next allocation takes. Nothing is copied, nothing is
        decommitted, and ``available_size()`` is unchanged -- which is why
        the measurement law that falsified the cache-dump lender (a
        free-memory metric cannot validate a mechanism whose action is
        freeing memory) does not apply to it: the quantity to measure here
        is PLACEMENT.

        WHY REORDERING IS THE WHOLE MECHANISM. Every allocation path in
        every subclass consumes the HEAD of ``free_pages`` -- ``alloc``, and
        on the paged class the two Triton kernels, which index it from 0 --
        so the head is the only choice point any of them has. A stable
        partition steers all of them at once without touching a kernel, and
        it degrades to a no-op the moment the preferred class runs out: the
        tail is still the whole rest of the free list, so no allocation can
        fail because of a bias.

        DETERMINISM IS A CORRECTNESS REQUIREMENT, NOT A NICETY. The free
        list is replicated scheduler state: every rank hands out the same
        global slot ids for the same tokens, and the owner rule turns an id
        into a row on exactly one rank. Two ranks that ordered their free
        lists differently would write one token's KV to two different rows.
        The partition is therefore a pure function of ``free_pages`` and
        ``bias`` (stable, so ties keep the sorted order), and the CHOICE of
        bias must be group-uniform -- see ``managers/corridor_steering.py``,
        which reduces it and then checks the resulting order agrees across
        ranks.
        """
        if bias is not None:
            mod, lo, hi = (int(v) for v in bias)
            if mod <= 0 or not (0 <= lo < hi <= mod):
                raise ValueError(
                    f"owner bias {bias!r} is not a class of a {mod}-slot block"
                )
            if self.page_size != 1:
                # With page_size > 1 a page spans several residues, so a page
                # id does not name an owner at all. Same precondition as
                # ``alloc_owner_matched_classes``.
                raise ValueError(
                    "owner bias requires page_size == 1 (uneven-DCP keeps the "
                    f"natural page size); this allocator has {self.page_size}"
                )
            bias = (mod, lo, hi)
        self._owner_bias = bias
        return self._apply_owner_bias()

    def _apply_owner_bias(self) -> int:
        """Stable-partition ``free_pages`` so the biased class leads."""
        bias = getattr(self, "_owner_bias", None)
        if bias is None:
            return 0
        if not self.is_not_in_free_group:
            # A free group is mid-flight; its pages are not in the list yet
            # and reordering now would be an incomplete answer taken as a
            # complete one. The round clock will come back.
            return 0
        if self.release_pages is not None and len(self.release_pages) > 0:
            self._merge_and_sort_free_unbiased()
        pages = self.free_pages
        if pages is None or pages.numel() == 0:
            return 0
        mod, lo, hi = bias
        res = pages % mod
        m = (res >= lo) & (res < hi)
        n = int(m.sum())
        if n == 0 or n == pages.numel():
            # Nothing to promote, or nothing to promote it over. Leaving the
            # list untouched keeps the sorted order a later merge assumes.
            return n
        self.free_pages = torch.cat((pages[m], pages[~m]))
        return n

    def _merge_and_sort_free_unbiased(self):
        if len(self.release_pages) > 0:
            self.free_pages = torch.cat((self.free_pages, self.release_pages))
            self.free_pages, _ = torch.sort(self.free_pages)
            self.release_pages = torch.empty(
                (0,), dtype=self.release_pages.dtype, device=self.device
            )

    def register_free_listener(self, on_free, on_clear=None) -> None:
        """Subscribe to slot lifetime events: ``on_free(free_index)`` after
        indices return to the free set, ``on_clear()`` on a full reset.
        Listeners must be cheap and thread-tolerant (free runs on the
        scheduler thread)."""
        self._free_listeners.append((on_free, on_clear))

    def _notify_free(self, free_index) -> None:
        # getattr: subclasses may clear() from __init__ before the base
        # __init__ ran (defensive; the token allocator initializes in order).
        for on_free, _on_clear in getattr(self, "_free_listeners", ()):
            on_free(free_index)

    def _notify_clear(self) -> None:
        for _on_free, on_clear in getattr(self, "_free_listeners", ()):
            if on_clear is not None:
                on_clear()

    @property
    def size_full(self):
        return self.size

    def debug_print(self) -> str:
        return ""

    def available_size(self):
        return (len(self.free_pages) + len(self.release_pages)) * self.page_size

    def get_kvcache(self):
        return self._kvcache

    def restore_state(self, state):
        self.free_pages, self.release_pages = state

    def backup_state(self):
        return (self.free_pages, self.release_pages)

    def free_group_begin(self):
        self.is_not_in_free_group = False
        self.free_group = []

    def free_group_end(self):
        self.is_not_in_free_group = True
        if self.free_group:
            self.free(torch.cat(self.free_group))

    def merge_and_sort_free(self):
        self._merge_and_sort_free_unbiased()
        # The sort undoes the partition, and it runs from inside the alloc
        # paths, so re-applying it here is what keeps a steer alive across a
        # refill instead of only until the first one.
        if getattr(self, "_owner_bias", None) is not None:
            self._apply_owner_bias()

    def get_cpu_copy(self, indices, mamba_indices=None):
        # FIXME: reuse the get_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        # FIXME: reuse the load_cpu_copy after paged allocator is implemented
        raise NotImplementedError()

    def alloc_extend(self, *args, **kwargs):
        raise NotImplementedError("alloc_extend is only for paged allocator")

    def alloc_decode(self, *args, **kwargs):
        raise NotImplementedError("alloc_decode is only for paged allocator")

    def resize(self, config) -> None:
        self.size = config.max_total_num_tokens
        # The paged allocator keeps num_pages even at page_size == 1 (the
        # uneven-DCP natural-page lane); the old `page_size > 1` condition
        # left it stale there and clear() would rebuild the free list at the
        # OLD size (#330 finding).
        if hasattr(self, "num_pages"):
            self.num_pages = config.max_total_num_tokens // self.page_size
        self.clear()

    def grow_size(self, new_size: int) -> None:
        """#330 capacity re-raise: raise the slot-id ceiling at runtime while
        KEEPING every live allocation — the new ids ``(size, new_size]`` join
        the free list, nothing is cleared. Valid for the arange(1..N) free-id
        convention shared by the token and paged allocators; subclasses with
        extra size-derived state must override."""
        new_size = int(new_size)
        if new_size < self.size:
            raise ValueError(
                f"grow_size({new_size}) below current size {self.size}; "
                f"shrinking requires the flush + resize path"
            )
        if new_size == self.size:
            return
        if new_size % self.page_size != 0:
            raise ValueError(
                f"grow_size({new_size}) not a multiple of page_size "
                f"{self.page_size}"
            )
        old_pages = self.size // self.page_size
        new_pages = new_size // self.page_size
        fresh = torch.arange(
            old_pages + 1, new_pages + 1, dtype=torch.int64, device=self.device
        )
        self.free_pages = torch.cat((self.free_pages, fresh))
        self.size = new_size
        if hasattr(self, "num_pages"):
            self.num_pages = new_pages

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def alloc(self, need_size: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def free(self, free_index: torch.Tensor):
        raise NotImplementedError()
