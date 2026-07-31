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
        if len(self.release_pages) > 0:
            self.free_pages = torch.cat((self.free_pages, self.release_pages))
            self.free_pages, _ = torch.sort(self.free_pages)
            self.release_pages = torch.empty(
                (0,), dtype=self.release_pages.dtype, device=self.device
            )

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
        if self.page_size > 1:
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
        if self.page_size > 1:
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
