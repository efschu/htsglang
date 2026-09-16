"""#1424 Stufe 3: the host tier IS the shared arena.

User order 2026-09-16 ("EINEN L2 pool, nicht pro rang, nicht pro phase"): a
per-rank host pool that has to hold a whole request is a second pool, not a
transit buffer. Here the host rows of the radix tree are arena slots:

* READ (prefetch): a page that is COMPLETE in the arena is addressed in place
  -- key -> slot (arena_find_slots), a reader reference (arena_ref_slots), no
  copy. The load to the card runs the same per-layer transfer kernel the
  page_first layout uses, over strided views of the pinned arena data region
  (`data_off + slot * page_bytes`, this rank's layers at `k_off + l * cell`).
* WRITE (backup): a small staging ring (rows [0, S)) as before; the store
  write puts the page into the arena. (Rebinding the node's rows to the slot
  after the ack is the next step.)

Host index space: [0, S) staging rows, [S, S+A) arena slot `id - S`,
[S+A, S+A+P) read placeholders handed out at prefetch registration and
replaced in place by the resolution. The draft pool (MTP) shares the KV
pool's index space, so a draft row for an arena id maps through
`row_slot` to the DRAFT arena's slot (or to a zero row on a miss).
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import torch

from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

logger = logging.getLogger(__name__)

ENV_ARENA_HOST = "SGLANG_HICACHE_ARENA_HOST"
ENV_STAGING_GB = "SGLANG_HICACHE_ARENA_STAGING_GB"
PLACEHOLDERS = 1 << 22
_CUDA_HOST_REGISTER_FLAGS = 3  # Portable | Mapped
_CUDA_ERROR_ALREADY_REGISTERED = 712


def arena_host_enabled() -> bool:
    return os.environ.get(ENV_ARENA_HOST, "0") == "1"


class ArenaMHAHostPool(MHATokenToKVPoolHost):
    """MHA host pool whose rows beyond the staging ring are arena slots."""

    arena_read = True

    def __init__(self, device_pool, host_to_device_ratio, host_size, page_size, layout,
                 *args, **kwargs):
        staging_gb = int(os.environ.get(ENV_STAGING_GB, "1") or 1)
        if host_to_device_ratio and not host_size:
            # the draft pool follows the anchor's size by ratio; keep that
            super().__init__(device_pool, host_to_device_ratio, host_size, page_size, layout,
                             *args, **kwargs)
        else:
            super().__init__(device_pool, 0.0, staging_gb, page_size, layout, *args, **kwargs)
        self._arena_init_fields()

    def _arena_init_fields(self) -> None:
        self.staging_rows = int(self.size)
        self.arena = None
        self.arena_slots = 0
        self.arena_k_refs: Optional[list] = None
        self.arena_v_refs: Optional[list] = None
        self.row_slot: Optional[dict] = None  # draft role: kv-row -> draft slot (-1 miss)
        self._zero_k = None
        self._zero_v = None
        self._read_ph_next = 0
        self.id_space = self.staging_rows
        self.prefetch_capacity_tokens = None

    # -- binding -----------------------------------------------------------
    def ensure_bound(self, storage_backend, role: str = "kv") -> bool:
        if self.arena is not None:
            return True
        try:
            window = (storage_backend._canonical_kv_extents if role == "kv"
                      else storage_backend.canonical_draft_page)
            if window is None:
                return False
            arena = storage_backend._arena_for(int(window.total_bytes))
            if arena is None:
                return False
            self.bind(arena, window, role=role)
            return True
        except Exception as exc:  # noqa: BLE001 - loud, never silent
            logger.error("#1424 arena host pool bind failed (role=%s): %r", role, exc)
            return False

    def bind(self, arena, window, role: str = "kv", pin: bool = True) -> None:
        if int(self.page_size) != 1:
            raise ValueError("#1424 the arena host pool needs page_size 1 (one slot per token)")
        ext = [(int(o), int(l)) for o, l in window.extents]
        if len(ext) != 2:
            raise ValueError(f"#1424 expected a K and a V extent, got {ext}")
        (k_off, k_len), (v_off, v_len) = ext
        L = int(self.layer_num)
        e = int(self.dtype.itemsize)
        cell = int(self.head_num) * int(self.head_dim) * e
        if k_len != v_len or k_len != L * cell:
            raise ValueError(f"#1424 window extents {ext} do not match {L} layers x {cell} B")
        page_bytes = int(window.total_bytes)
        A = int(arena.slots)
        data_off = int(arena.data_offset())
        buf = torch.frombuffer(arena._mm, dtype=torch.uint8)
        region = buf[data_off:data_off + A * page_bytes]
        typed = region.view(self.dtype)
        tok = page_bytes // e
        H, D = int(self.head_num), int(self.head_dim)
        base_off = int(typed.storage_offset())  # the data region's own offset in the mapping
        self.arena_k_refs = [
            typed.as_strided((A, H, D), (tok, D, 1), storage_offset=base_off + (k_off + l * cell) // e)
            for l in range(L)
        ]
        self.arena_v_refs = [
            typed.as_strided((A, H, D), (tok, D, 1), storage_offset=base_off + (v_off + l * cell) // e)
            for l in range(L)
        ]
        if pin and torch.cuda.is_available() and not getattr(arena, "_pinned", False):
            rc = torch.cuda.cudart().cudaHostRegister(
                buf.data_ptr(), buf.numel(), _CUDA_HOST_REGISTER_FLAGS)
            if int(rc) not in (0, _CUDA_ERROR_ALREADY_REGISTERED):
                raise RuntimeError(f"#1424 cudaHostRegister({buf.numel()} B) failed: {int(rc)}")
            arena._pinned = True
        self.arena = arena
        self.arena_slots = A
        self.id_space = self.staging_rows + A + PLACEHOLDERS
        self.prefetch_capacity_tokens = A
        if role == "draft":
            self.row_slot = {}
            self._zero_k = torch.zeros((1, H, D), dtype=self.dtype, pin_memory=self.pin_memory)
            self._zero_v = torch.zeros((1, H, D), dtype=self.dtype, pin_memory=self.pin_memory)
        logger.info("#1424 arena host pool bound role=%s staging=%d slots=%d layers=%d k_off=%d v_off=%d",
                    role, self.staging_rows, A, L, k_off, v_off)

    # -- id space ------------------------------------------------------------
    def is_arena_id(self, i: int) -> bool:
        return self.staging_rows <= int(i) < self.staging_rows + self.arena_slots

    def is_placeholder(self, i: int) -> bool:
        return int(i) >= self.staging_rows + self.arena_slots

    def alloc_read(self, n: int) -> torch.Tensor:
        """Placeholders for a prefetch registration; resolved in place later."""
        base = self.staging_rows + self.arena_slots
        start = self._read_ph_next
        self._read_ph_next = (start + n) % PLACEHOLDERS
        return (torch.arange(start, start + n, dtype=torch.int64) % PLACEHOLDERS) + base

    def resolve_rows(self, host_indices: torch.Tensor, slots: Sequence[int]) -> None:
        """KV role: write arena ids over the placeholders, in place."""
        n = len(slots)
        vals = torch.tensor([self.staging_rows + int(s) for s in slots], dtype=host_indices.dtype)
        host_indices[:n] = vals.to(host_indices.device)

    def resolve_draft_rows(self, rows: Sequence[int], slots: Sequence[int]) -> None:
        """Draft role: remember the DRAFT arena slot behind each KV row (-1 = miss)."""
        if self.row_slot is None:
            raise RuntimeError("#1424 resolve_draft_rows on a pool not bound as draft")
        for r, s in zip(rows, slots):
            self.row_slot[int(r)] = int(s)

    # -- transfers ------------------------------------------------------------
    def load_to_device_per_layer(self, device_pool, host_indices, device_indices, layer_id, io_backend):
        if self.arena is None or host_indices.numel() == 0:
            return super().load_to_device_per_layer(device_pool, host_indices, device_indices, layer_id, io_backend)
        S, A = self.staging_rows, self.arena_slots
        is_arena = (host_indices >= S) & (host_indices < S + A)
        if not bool(is_arena.any()):
            return super().load_to_device_per_layer(device_pool, host_indices, device_indices, layer_id, io_backend)
        if bool(is_arena.all()):
            return self._load_arena(device_pool, host_indices - S, device_indices, layer_id)
        sel = is_arena.nonzero(as_tuple=True)[0]
        rest = (~is_arena).nonzero(as_tuple=True)[0]
        self._load_arena(device_pool, host_indices[sel] - S, device_indices[sel], layer_id)
        super().load_to_device_per_layer(device_pool, host_indices[rest], device_indices[rest], layer_id, io_backend)

    def _load_arena(self, device_pool, rows, device_indices, layer_id) -> None:
        if self.row_slot is not None:
            rl = rows.tolist()
            slots = torch.tensor([self.row_slot.get(int(r), -1) for r in rl], dtype=rows.dtype, device=rows.device)
            miss = slots < 0
            if bool(miss.any()):
                z = torch.zeros((int(miss.sum()),), dtype=rows.dtype, device=rows.device)
                self._transfer(device_pool, self._zero_k, self._zero_v, z, device_indices[miss], layer_id)
            hit = ~miss
            if bool(hit.any()):
                self._transfer(device_pool, self.arena_k_refs[layer_id], self.arena_v_refs[layer_id],
                               slots[hit], device_indices[hit], layer_id)
            return
        self._transfer(device_pool, self.arena_k_refs[layer_id], self.arena_v_refs[layer_id],
                       rows, device_indices, layer_id)

    def _transfer(self, device_pool, k_src, v_src, src_idx, dst_idx, layer_id) -> None:
        if not getattr(self, "can_use_jit", False):
            raise RuntimeError("#1424 the arena host pool needs the JIT hicache transfer kernel")
        from sglang.jit_kernel.hicache import transfer_hicache_one_layer
        transfer_hicache_one_layer(
            k_cache_dst=device_pool.k_buffer[layer_id],
            v_cache_dst=device_pool.v_buffer[layer_id],
            k_cache_src=k_src,
            v_cache_src=v_src,
            indices_dst=dst_idx,
            indices_src=src_idx,
            element_dim=self.element_dim,
        )

    def backup_from_device_all_layer(self, device_pool, host_indices, device_indices, io_backend):
        if self.arena is not None and host_indices.numel() and int(host_indices.max()) >= self.staging_rows:
            raise RuntimeError("#1424 backup targets must be staging rows, not arena ids")
        return super().backup_from_device_all_layer(device_pool, host_indices, device_indices, io_backend)

    # -- page accessors ---------------------------------------------------------
    def get_data_page(self, index, flat: bool = True):
        if self.arena is not None and int(index) >= self.staging_rows:
            raise RuntimeError(f"#1424 host row {int(index)} is an arena page; it is already in the store")
        return super().get_data_page(index, flat)

    def set_from_flat_data_page(self, index: int, data_page) -> None:
        if self.arena is not None and int(index) >= self.staging_rows:
            raise RuntimeError(f"#1424 host row {int(index)} is an arena page; writes go to the staging ring")
        return super().set_from_flat_data_page(index, data_page)

    # -- allocator ----------------------------------------------------------------
    def free(self, indices: torch.Tensor) -> int:
        if self.arena is None:
            return super().free(indices)
        idx = indices.cpu() if torch.is_tensor(indices) else torch.as_tensor(indices)
        S, A = self.staging_rows, self.arena_slots
        staging = idx[idx < S]
        arena_rows = idx[(idx >= S) & (idx < S + A)]
        freed = 0
        if staging.numel():
            freed += int(super().free(staging))
        if arena_rows.numel():
            rows = (arena_rows - S).tolist()
            if self.row_slot is not None:
                slots = [self.row_slot.pop(int(r), -1) for r in rows]
                slots = [s for s in slots if s >= 0]
            else:
                slots = rows
            if slots:
                self.arena.ref_slots(slots, -1)
            freed += len(rows)
        return freed
