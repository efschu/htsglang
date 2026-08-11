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

"""
Page-aligned memory pool.
"""


from typing import TYPE_CHECKING

import torch

from sglang.kernels.ops.memory.allocator import (
    alloc_decode_kernel,
    alloc_extend_kernel,
)
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.utils import (
    get_bool_env_var,
    get_num_new_pages,
    is_hip,
    next_power_of_2,
)

_is_hip = is_hip()

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache


def alloc_extend_naive(
    prefix_lens,
    seq_lens,
    last_loc,
    free_pages,
    out_indices,
    page_size,
    device,
):
    extend_lens = seq_lens - prefix_lens
    end_pos = torch.cumsum(extend_lens, 0)
    start_pos = end_pos - extend_lens
    num_new_pages = (seq_lens + page_size - 1) // page_size - (
        prefix_lens + page_size - 1
    ) // page_size
    num_full_new_pages = (seq_lens) // page_size - (
        prefix_lens + page_size - 1
    ) // page_size
    need_page = num_new_pages - num_full_new_pages
    end_new_pages = torch.cumsum(num_new_pages, 0)
    start_new_pages = end_new_pages - num_new_pages
    pos_in_page = torch.arange(page_size, device=device, dtype=torch.int32)
    for i in range(len(prefix_lens)):
        num1 = (
            min(
                seq_lens[i],
                (prefix_lens[i] + page_size - 1) // page_size * page_size,
            )
            - prefix_lens[i]
        )
        if num1:
            out_indices[start_pos[i] : start_pos[i] + num1] = (
                last_loc[i] + 1 + pos_in_page[:num1].view(-1)
            )

        if prefix_lens[i] + num1 == seq_lens[i]:
            continue

        num2 = (
            seq_lens[i] // page_size - (prefix_lens[i] + page_size - 1) // page_size
        ) * page_size
        if num2:
            pages = (
                free_pages[start_new_pages[i] : end_new_pages[i] - need_page[i]]
                * page_size
            )
            out_indices[start_pos[i] + num1 : start_pos[i] + num1 + num2] = (
                pages.view(-1, 1) + pos_in_page.view(1, -1)
            ).view(-1)

        if prefix_lens[i] + num1 + num2 == seq_lens[i]:
            continue

        num3 = seq_lens[i] - seq_lens[i] // page_size * page_size
        if num3:
            out_indices[end_pos[i] - num3 : end_pos[i]] = (
                free_pages[end_new_pages[i] - 1] * page_size + pos_in_page[:num3]
            ).view(-1)


class PagedTokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    """
    An allocator managing the indices to kv cache data.

    This class has the same interface as `TokenToKVPoolAllocator` but the output
    of one request is always page-aligned.

    TODO: fuse last_loc into the kernel.
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        kvcache: KVCache,
        need_sort: bool,
    ):
        super().__init__(size, page_size, dtype, device, kvcache, need_sort)
        self.num_pages = size // page_size
        self.debug_mode = get_bool_env_var("SGLANG_DEBUG_MEMORY_POOL")
        #: #656 item 16: the DCP owner class new allocations should prefer,
        #: or None. Set through :meth:`set_owner_bias` only.
        self._owner_bias = None

        # Pre-warm the torch.unique HIP kernel used in free(). When a request
        # finishes with a prompt that already exists in the radix tree (e.g.
        # bench_serving sending the same warmup+measured prompt), the radix
        # cache's _insert_helper frees the duplicate KV indices via
        # token_to_kv_pool_allocator.free(value[start:prefix_len]). That call
        # path runs `torch.unique(free_index // self.page_size)` on a
        # ~prompt_len-sized int64 tensor. The first such call on AMD ROCm
        # JIT-compiles rocPRIM sort/unique kernels and costs ~200ms, which
        # shows up as a mysterious "second-request slow" (Run 1) for
        # repeated-prompt benchmarks. Running it once at init time moves
        # that JIT cost to startup. This is a ROCm-only JIT cost, so the
        # warm-up is gated on _is_hip and skipped on other platforms.
        if _is_hip and torch.cuda.is_available():
            try:
                _warmup = torch.arange(1024, dtype=torch.int64, device=device)
                _ = torch.unique(_warmup // page_size)
                torch.cuda.synchronize()
            except Exception:
                pass
        self.clear()

    def alloc(self, need_size: int):
        # page-aligned allocation, returning contiguous indices of pages
        if self.debug_mode:
            assert (
                need_size % self.page_size == 0
            ), "The allocation size should be page-aligned"

        num_pages = need_size // self.page_size
        if self.need_sort and num_pages > len(self.free_pages):
            self.merge_and_sort_free()
        if num_pages > len(self.free_pages):
            return None

        out_pages = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]

        out_indices = (
            out_pages[:, None] * self.page_size
            + torch.arange(self.page_size, device=self.device)
        ).reshape(-1)

        return out_indices

    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        num_new_pages: int = None,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 1) % self.page_size == prefix_lens % self.page_size
            )

        bs = len(prefix_lens)
        if self.need_sort and extend_num_tokens // self.page_size + bs + 1 > len(
            self.free_pages
        ):
            self.merge_and_sort_free()

        out_indices = torch.empty(
            (extend_num_tokens,), dtype=torch.int64, device=self.device
        )

        alloc_extend_kernel[(bs,)](
            prefix_lens,
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            next_power_of_2(bs),
            self.page_size,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        if num_new_pages is None:
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu,
                page_size=self.page_size,
                prefix_lens=prefix_lens_cpu,
            )
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 2) % self.page_size == seq_lens % self.page_size
            )

        bs = len(seq_lens)
        if self.need_sort and bs > len(self.free_pages):
            self.merge_and_sort_free()

        out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)
        alloc_decode_kernel[(bs,)](
            seq_lens,
            last_loc,
            self.free_pages,
            out_indices,
            next_power_of_2(bs),
            self.page_size,
        )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=self.page_size,
            decode=True,
        )
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def alloc_owner_matched_classes(self, mod, class_bounds, class_needs):
        """kv-session-offload (S1) restore: allocate slots whose residue
        ``slot % mod`` falls into per-class bounds ``[lo, hi)`` -- one class
        per DCP rank under the weighted owner rule, so each restored token
        lands on the rank that holds its host-backup row.

        page_size must be 1 (the uneven-DCP allocator keeps the natural page
        size, so page id == slot id). Deterministic: picks the FIRST n free
        slots of each class in current free-list order, identical on every
        rank (the free list is replicated scheduler state). Returns a list
        of int64 slot tensors (one per class, ascending pick order) and
        removes them from the free list, or None (state unchanged) when any
        class has too few free slots. Never called on the default path."""
        assert self.page_size == 1, (
            "alloc_owner_matched_classes requires page_size == 1"
        )
        if len(self.release_pages) > 0:
            self.merge_and_sort_free()
        pages = self.free_pages
        res = pages % mod
        take_mask = torch.zeros_like(pages, dtype=torch.bool)
        picks = []
        for (lo, hi), need in zip(class_bounds, class_needs):
            m = (res >= lo) & (res < hi)
            idxs = m.nonzero(as_tuple=False).flatten()[:need]
            if idxs.numel() < need:
                return None
            take_mask[idxs] = True
            picks.append(pages[idxs].to(torch.int64))
        self.free_pages = pages[~take_mask]
        return picks

    # -- #656 item 16, the REBALANCE tier: steering, not moving ------------

    def set_owner_bias(self, bias) -> int:
        """Prefer free slots of one DCP owner class at the head of the list.

        ``bias`` is ``(mod, lo, hi)`` -- the weighted owner rule's
        ``(cp_S, cp_lo, cp_hi)`` for the rank that should ABSORB new KV --
        or ``None`` to steer nothing. Returns how many free slots of that
        class were promoted, so a caller can log a number instead of a
        claim.

        THIS PLACES BYTES; IT DOES NOT MOVE OR FREE ANY. Every slot in the
        free list is already a legal placement on its own rank, and this
        only reorders which legal one the next allocation takes. Nothing is
        copied, nothing is decommitted, and ``available_size()`` is
        unchanged -- which is why the measurement law that falsified the
        cache-dump lender (a free-memory metric cannot validate a mechanism
        whose action is freeing memory) does not apply to it: the quantity
        to measure here is PLACEMENT.

        WHY REORDERING IS THE WHOLE MECHANISM. ``alloc``, ``alloc_extend``
        and ``alloc_decode`` all consume the HEAD of ``free_pages``
        (``:162``, ``:219``, ``:258``, and the two Triton kernels index it
        from 0), so the head is the only choice point any of them has. A
        stable partition therefore steers all three paths at once without
        touching a kernel, and it degrades to a no-op the moment the
        preferred class runs out -- the tail is still the whole rest of the
        free list, so no allocation can fail because of a bias.

        DETERMINISM IS A CORRECTNESS REQUIREMENT, NOT A NICETY. The free
        list is replicated scheduler state: every rank hands out the same
        global slot ids for the same tokens, and the owner rule turns an id
        into a row on exactly one rank. Two ranks that ordered their free
        lists differently would write one token's KV to two different
        slots. The partition is therefore a pure function of
        ``free_pages`` and ``bias`` (stable, so ties keep the sorted order),
        and the CHOICE of bias must be group-uniform -- see
        ``managers/corridor_steering.py``, which reduces it and then checks
        the resulting order agrees across ranks.
        """
        if bias is not None:
            mod, lo, hi = (int(v) for v in bias)
            if mod <= 0 or not (0 <= lo < hi <= mod):
                raise ValueError(
                    f"owner bias {bias!r} is not a class of a {mod}-slot block"
                )
            if self.page_size != 1:
                # Same precondition as alloc_owner_matched_classes: with
                # page_size > 1 a page spans several residues, so a page id
                # does not name an owner at all.
                raise ValueError(
                    "owner bias requires page_size == 1 (uneven-DCP keeps the "
                    f"natural page size); this allocator has {self.page_size}"
                )
            bias = (mod, lo, hi)
        self._owner_bias = bias
        return self._apply_owner_bias()

    def _apply_owner_bias(self) -> int:
        """Stable-partition ``free_pages`` so the biased class leads. """
        bias = getattr(self, "_owner_bias", None)
        if bias is None:
            return 0
        if not self.is_not_in_free_group:
            # A free group is mid-flight; its pages are not in the list yet
            # and reordering now would be an incomplete answer taken as a
            # complete one. The round clock will come back.
            return 0
        if len(self.release_pages) > 0:
            # The BASE merge, not this class's override: the override calls
            # back into here, and a partition that re-enters itself would
            # pay for the same pass twice on every refill.
            super().merge_and_sort_free()
        pages = self.free_pages
        if pages.numel() == 0:
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

    def merge_and_sort_free(self):
        super().merge_and_sort_free()
        # The sort undoes the partition, and it runs from inside the alloc
        # paths, so re-applying it here is what keeps the steer alive across
        # a refill instead of only until the first one.
        if getattr(self, "_owner_bias", None) is not None:
            self._apply_owner_bias()

    def free(self, free_index: torch.Tensor):
        if free_index.numel() == 0:
            return

        if self.is_not_in_free_group:
            free_page_indices = torch.unique(free_index // self.page_size)
            if self.need_sort:
                self.release_pages = torch.cat((free_page_indices, self.release_pages))
            else:
                self.free_pages = torch.cat((free_page_indices, self.free_pages))
            # Token-index semantics for listeners (grouped frees are
            # notified once, via free_group_end re-entering here).
            self._notify_free(free_index)
        else:
            self.free_group.append(free_index)

        if self.debug_mode:
            assert len(torch.unique(self.free_pages)) == len(self.free_pages)

    def clear(self):
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.free_pages = torch.arange(
            1, self.num_pages + 1, dtype=torch.int64, device=self.device
        )
        self.is_not_in_free_group = True
        self.free_group = []
        self.release_pages = torch.empty((0,), dtype=torch.int64, device=self.device)
        self._notify_clear()

    def get_cpu_copy(self, indices, mamba_indices=None):
        return self._kvcache.get_cpu_copy(indices, mamba_indices=mamba_indices)

    def load_cpu_copy(self, kv_cache_cpu, indices, mamba_indices=None):
        return self._kvcache.load_cpu_copy(
            kv_cache_cpu, indices, mamba_indices=mamba_indices
        )
