"""#1427 Stufe 4b: the mamba (GDN) anchor host pool whose rows beyond the
staging slots ARE the mamba arena's slots.

Stufe 3/4 gave the KV pages a home in the shared arena and a direct write
into it; the recurrent states still travelled through a 13-slot host anchor
pool (--hicache-mamba-host-mib 600) into the store thread and out again --
the wall the park test died on (xsn187: "host anchor pool exhausted",
controller.write None). Here every arena slot holds one canonical GDN blob
(all layers, all heads; ``MambaBlobSpec``), and this rank owns the byte
ranges of its layers and heads inside it (``temporal_extents`` /
``conv_extents``, the same cut the store reads and writes). The pool keeps
strided views on those ranges, one per local layer and conv segment, so a
backup is a device->host index copy into the view and a load is the
reverse -- no serialisation, no payload copy, no anchor pool.

Ids: [0, S) staging (the old anchor slots, kept for nodes without hashes),
[S, S+A) arena slot id-S, [S+A, S+A+2^22) read placeholders resolved in
place at prefetch time.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import torch

from sglang.srt.mem_cache.memory_pool_host import MambaPoolHost
from sglang.srt.mem_cache.pool_host.arena_pool import PLACEHOLDERS, ArenaMHAHostPool

logger = logging.getLogger(__name__)


def _contig_strides(shape):
    strides = []
    acc = 1
    for d in reversed(tuple(shape)):
        strides.append(acc)
        acc *= int(d)
    return tuple(reversed(strides))


class ArenaMambaPoolHost(MambaPoolHost):
    arena_read = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._arena_init_fields()

    def _arena_init_fields(self) -> None:
        self.staging_rows = int(self.size)
        self.arena = None
        self.arena_slots = 0
        self.row_slot = None  # never a draft role; keeps the shared helpers on the row path
        self._read_ph_next = 0
        self.id_space = self.staging_rows
        self._backend = None
        self._weg2_parts = None  # (spec, ratios, rank, layer_lo, layer_hi) from the controller
        self._own_extents: Optional[list] = None
        self._page_bytes = 0
        self._pending: dict = {}
        self._pin = False
        self._pin_base = 0
        self._pin_bytes = 0
        self._pinned = None
        self._t_views: Optional[list] = None      # per local layer: (A, *temporal_shape)
        self._c_views: Optional[list] = None      # per conv buffer: per local layer: [(ch0, n, view)]

    # -- binding -------------------------------------------------------------------
    def ensure_bound(self, storage_backend, role: str = "mamba") -> bool:
        if self.arena is not None:
            return True
        parts = self._weg2_parts
        blob = getattr(storage_backend, "canonical_mamba_blob", None)
        if parts is None or blob is None:
            return False
        try:
            arena = storage_backend._arena_for(int(blob.total_bytes))
            if arena is None:
                return False
            self.bind(arena, int(blob.total_bytes), parts)
            self._backend = storage_backend
            return True
        except Exception as exc:  # noqa: BLE001 - loud, never silent
            logger.error("#1427 mamba arena host pool bind failed: %r", exc)
            return False

    def bind(self, arena, total_bytes: int, parts, pin: bool = True) -> None:
        from sglang.srt.mem_cache.hicache_migrate import conv_extents, temporal_extents

        spec, ratios, rank, lo, hi = parts
        L = int(self.num_mamba_layers)
        if hi - lo != L:
            raise ValueError(f"#1427 mamba window covers {hi - lo} layers, this pool has {L}")
        if len(self.conv_state_shapes) != 1:
            raise ValueError("#1427 the mamba arena views expect exactly one conv buffer")
        t_ext = temporal_extents(spec, list(ratios), int(rank))[lo:hi]
        c_ext = conv_extents(spec, list(ratios), int(rank))[3 * lo: 3 * hi]
        A = int(arena.slots)
        slot_bytes = int(arena.slot_bytes)
        if slot_bytes != int(total_bytes):
            raise ValueError(f"#1427 arena slot {slot_bytes} B != mamba blob {total_bytes} B")
        data_off = int(arena.data_offset())
        e_t = int(self.temporal_dtype.itemsize)
        e_c = int(self.conv_dtype.itemsize)
        if slot_bytes % e_t or slot_bytes % e_c:
            raise ValueError("#1427 mamba blob bytes are not a multiple of the state itemsizes")
        t_shape = tuple(int(d) for d in self.temporal_state_shape)
        t_row = int(self.temporal_state_elem_size) * e_t
        conv_shape = tuple(int(d) for d in self.conv_state_shapes[0])
        width = conv_shape[-1]
        per_ch = width * e_c

        def _view(dtype, e, off, shape):
            if off % e:
                raise ValueError(f"#1427 extent offset {off} not aligned to {e}")
            rows_elems = 1
            for d in shape:
                rows_elems *= d
            count = (A - 1) * (slot_bytes // e) + rows_elems
            t = torch.frombuffer(arena._mm, dtype=dtype, count=count, offset=data_off + off)
            return t.as_strided((A,) + tuple(shape), (slot_bytes // e,) + _contig_strides(shape))

        t_views, c_views, extents = [], [], []
        for l in range(L):
            off, ln = int(t_ext[l][0]), int(t_ext[l][1])
            if ln != t_row:
                raise ValueError(f"#1427 temporal extent {ln} B != this pool's row {t_row} B")
            t_views.append(_view(self.temporal_dtype, e_t, off, t_shape))
            extents.append((off, ln))
            segs, ch0 = [], 0
            for j in range(3):
                off_j, ln_j = int(c_ext[3 * l + j][0]), int(c_ext[3 * l + j][1])
                if ln_j % per_ch:
                    raise ValueError(f"#1427 conv extent {ln_j} B is not whole channels of {per_ch} B")
                n_j = ln_j // per_ch
                segs.append((ch0, n_j, _view(self.conv_dtype, e_c, off_j, (n_j, width))))
                extents.append((off_j, ln_j))
                ch0 += n_j
            if ch0 != conv_shape[0]:
                raise ValueError(f"#1427 conv segments cover {ch0} channels, the row has {conv_shape[0]}")
            c_views.append(segs)
        self._t_views = t_views
        self._c_views = [c_views]
        self._own_extents = extents
        self._page_bytes = slot_bytes
        self._pin = bool(pin and torch.cuda.is_available())
        buf = torch.frombuffer(arena._mm, dtype=torch.uint8)
        self._pin_base = int(buf.data_ptr()) + data_off
        self._pin_bytes = slot_bytes
        self._pinned = getattr(arena, "_pinned_slots", None)
        if self._pinned is None:
            self._pinned = arena._pinned_slots = torch.zeros(A, dtype=torch.bool)
        self.arena = arena
        self.arena_slots = A
        self.id_space = self.staging_rows + A + PLACEHOLDERS
        logger.info("#1427 mamba arena host pool bound staging=%d slots=%d layers=[%d,%d) extents=%d",
                    self.staging_rows, A, lo, hi, len(extents))

    # -- shared arena helpers (row path, no draft mapping) -------------------------
    pin_slots = ArenaMHAHostPool.pin_slots
    _claim = ArenaMHAHostPool._claim
    complete_write = ArenaMHAHostPool.complete_write
    abort_write = ArenaMHAHostPool.abort_write
    _slots_of = ArenaMHAHostPool._slots_of

    def _stems(self, hashes, suffix: str = ""):
        from sglang.srt.mem_cache.hicache_storage import PoolName
        return [self._backend._get_suffixed_key(self._backend._log_key(PoolName.MAMBA, h)) for h in hashes]

    def alloc_write(self, hashes) -> Optional[torch.Tensor]:
        if self.arena is None or self._backend is None or not hashes:
            return None
        slots = self._claim(self._stems(hashes))
        if slots is None:
            return None
        return torch.tensor([self.staging_rows + s for s in slots], dtype=torch.int64)

    def is_arena_id(self, i: int) -> bool:
        return self.staging_rows <= int(i) < self.staging_rows + self.arena_slots

    def is_placeholder(self, i: int) -> bool:
        return int(i) >= self.staging_rows + self.arena_slots

    def alloc_read(self, n: int) -> Optional[torch.Tensor]:
        # #1427c (xsn189, D): an UNBOUND pool must not hand out placeholders --
        # nothing could resolve them, the copy path wrote past the staging
        # buffer and the load_back crashed on the stale id. Unbound = the old
        # anchor-slot path.
        if self.arena is None:
            return None  # #1430: unbound = no read target, not an anchor slot
        base = self.staging_rows + self.arena_slots
        start = self._read_ph_next
        self._read_ph_next = (start + n) % PLACEHOLDERS
        return (torch.arange(start, start + n, dtype=torch.int64) % PLACEHOLDERS) + base

    def arena_resolve_reads(self, backend, host_indices: torch.Tensor, storage_keys: Sequence[str]):
        """Prefetch: COMPLETE blobs are addressed in place -- slot, reader
        reference, id written over the placeholder. Returns a flag per key:
        True resolved, None = not in the arena (the caller reads the disk)."""
        if not self.ensure_bound(backend):
            return [False] * len(storage_keys)  # #1430: unbound = miss, never the copy path
        stems = [backend._get_suffixed_key(k) for k in storage_keys]
        found = self.arena.find_slots(stems)
        out = []
        for i, (slot, state) in enumerate(found):
            if slot >= 0 and state == 2 and self.arena.ref_slots([slot], +1) == 1:
                host_indices[i] = self.staging_rows + slot
                out.append(True)
            else:
                # #1427d (xsn190): NOT in the arena = an honest miss, exactly
                # as the KV pages (#1424 `_arena_page_get`): the arena is the
                # L2, the disk stays the cold tier, and a disk read would have
                # no row to land in (the placeholder is not a buffer). The
                # blob is usually just not COMPLETE yet (another layer shard
                # still writing) -- the prefix is recomputed, nothing crashes.
                out.append(False)
        return out

    # -- transfers -------------------------------------------------------------------
    def _split(self, host_indices: torch.Tensor):
        hi = host_indices.cpu()
        S, A = self.staging_rows, self.arena_slots
        is_arena = (hi >= S) & (hi < S + A)
        return hi, is_arena

    def backup_from_device_all_layer(self, device_pool, host_indices, device_indices, io_backend="direct"):
        if self.arena is None or host_indices.numel() == 0:
            return super().backup_from_device_all_layer(device_pool, host_indices, device_indices, io_backend)
        hi, is_arena = self._split(host_indices)
        if not bool(is_arena.any()):
            return super().backup_from_device_all_layer(device_pool, host_indices, device_indices, io_backend)
        sel = is_arena.nonzero(as_tuple=True)[0]
        pairs = [(int(r), i) for r, i in zip((hi[sel] - self.staging_rows).tolist(), sel.tolist())
                 if int(r) in self._pending]
        if pairs:
            slots = torch.tensor([s for s, _ in pairs], dtype=torch.int64)
            didx = device_indices.cpu()[torch.tensor([i for _, i in pairs], dtype=torch.int64)]
            self.pin_slots(slots)
            dev = device_pool.mamba_cache.temporal.device
            didx_d = didx.to(dev)
            for l in range(int(self.num_mamba_layers)):
                src = device_pool.mamba_cache.temporal[l].index_select(0, didx_d).to("cpu")
                self._t_views[l].index_copy_(0, slots, src)
                srcc = device_pool.mamba_cache.conv[0][l].index_select(0, didx_d).to("cpu")
                for ch0, n, view in self._c_views[0][l]:
                    view.index_copy_(0, slots, srcc[:, ch0:ch0 + n].contiguous())
        rest = (~is_arena).nonzero(as_tuple=True)[0]
        if rest.numel():
            super().backup_from_device_all_layer(
                device_pool, host_indices[rest.to(host_indices.device)],
                device_indices[rest.to(device_indices.device)], io_backend)

    def load_to_device_per_layer(self, device_pool, host_indices, device_indices, layer_id, io_backend="direct"):
        if self.arena is None or host_indices.numel() == 0:
            return super().load_to_device_per_layer(device_pool, host_indices, device_indices, layer_id, io_backend)
        hi, is_arena = self._split(host_indices)
        if not bool(is_arena.any()):
            return super().load_to_device_per_layer(device_pool, host_indices, device_indices, layer_id, io_backend)
        sel = is_arena.nonzero(as_tuple=True)[0]
        slots = hi[sel] - self.staging_rows
        dst_t = device_pool.mamba_cache.temporal[layer_id]
        dev = dst_t.device
        didx = device_indices.cpu()[sel].to(dev)
        dst_t.index_copy_(0, didx, self._t_views[layer_id].index_select(0, slots).to(dev))
        dst_c = device_pool.mamba_cache.conv[0][layer_id]
        row = torch.empty((int(slots.numel()),) + tuple(dst_c.shape[1:]), dtype=dst_c.dtype)
        for ch0, n, view in self._c_views[0][layer_id]:
            row[:, ch0:ch0 + n] = view.index_select(0, slots)
        dst_c.index_copy_(0, didx, row.to(dev))
        rest = (~is_arena).nonzero(as_tuple=True)[0]
        if rest.numel():
            super().load_to_device_per_layer(
                device_pool, host_indices[rest.to(host_indices.device)],
                device_indices[rest.to(device_indices.device)], layer_id, io_backend)

    # -- page accessors: an arena blob is never serialised through this pool ------
    def _iter_page_tensors(self, index: int):
        if self.arena is not None and int(index) >= self.staging_rows:
            raise RuntimeError(f"#1427 mamba host row {int(index)} is an arena slot; it is already in the store")
        return super()._iter_page_tensors(index)

    def set_from_flat_data_page(self, index: int, data_page) -> None:
        if self.arena is not None and int(index) >= self.staging_rows:
            raise RuntimeError(f"#1427 mamba host row {int(index)} is an arena slot; reads resolve in place")
        return super().set_from_flat_data_page(index, data_page)

    # -- allocator -------------------------------------------------------------------
    def free(self, indices: torch.Tensor) -> int:
        if self.arena is None:
            return super().free(indices)
        idx = indices.cpu() if torch.is_tensor(indices) else torch.as_tensor(indices)
        S, A = self.staging_rows, self.arena_slots
        staging = idx[idx < S]
        rows = (idx[(idx >= S) & (idx < S + A)] - S).tolist()
        freed = 0
        if staging.numel():
            freed += int(super().free(staging))
        if rows:
            pend = [s for s in rows if s in self._pending]
            fresh = [s for s in pend if self._pending.pop(s)[1]]
            if fresh:
                self.arena.free_slots(fresh)
            keep = [s for s in rows if s not in pend]
            if keep:
                self.arena.ref_slots(keep, -1)
            freed += len(rows)
        return freed
