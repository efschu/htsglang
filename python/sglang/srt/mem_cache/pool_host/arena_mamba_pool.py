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
import os
from typing import Optional, Sequence

import torch

from sglang.srt.mem_cache.memory_pool_host import MambaPoolHost
from sglang.srt.weg2 import ple_state
from sglang.srt.mem_cache.pool_host.arena_pool import (
    PLACEHOLDERS,
    ArenaMHAHostPool,
    _arena_page_load_mode,
    _select_rows_async,
    index_to_device_async,
    load_index_async,
)

logger = logging.getLogger(__name__)


_STATE_LOAD_N = 0


def _arena_state_load_block_bytes() -> int:
    try:
        return max(1 << 20, int(os.environ.get("SGLANG_WEG2_ARENA_STATE_LOAD_BLOCK_BYTES", str(256 << 20))))
    except ValueError:
        return 256 << 20


def merge_state_extents(comp):
    """H12: the compact extents ``(cursor, arena_off, len)`` merged wherever
    the next one continues the previous one in the arena slot too. The
    compact side is contiguous by construction, so a merged run is ONE copy
    from the slot into the stage row."""
    runs = []
    for cur, off, ln in comp:
        if runs and runs[-1][1] + runs[-1][2] == off and runs[-1][0] + runs[-1][2] == cur:
            runs[-1] = (runs[-1][0], runs[-1][1], runs[-1][2] + ln)
        else:
            runs.append((cur, off, ln))
    return runs


def _state_block_dma(slot_view, dev_stage, slots, runs) -> int:
    """H12: this rank's extents of each slot copied straight from the
    registered mamba arena into the device stage row (no pinned CPU gather).
    Both the pre-pin (pieces of whole slots from slot 0) and the lazy pin
    (runs of whole slots) register whole slots, so an extent -- which never
    leaves its slot -- lies inside one registration. Returns the copies
    issued."""
    for i, slot in enumerate(slots):
        src = slot_view[slot]
        for cur, off, ln in runs:
            dev_stage[i, cur:cur + ln].copy_(src[off:off + ln], non_blocking=True)
    return len(slots) * len(runs)


def _arena_state_load_on() -> bool:
    # xsn337/338: 571-706 ms vs 432 per-layer -- opt-in back then. 19.09. xsn397 (compact
    # stage, only this rank's extents): the wake pass 1039 ms vs 1368-1392 with the
    # per-layer path (48 layers x synchronous pageable copies, the first one waiting
    # out the whole page DMA), flip 3.36 s vs 3.66, needle MATCH -> DEFAULT ON.
    return str(os.environ.get("SGLANG_WEG2_ARENA_STATE_LOAD", "1")).strip().lower() not in ("0", "false", "no", "off")


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
        # xsn332: the whole slot as ONE (A, slot_bytes) uint8 view -- the
        # all-layers loader gathers slots into a pinned stage and splits the
        # layers on the device (the KV page loader's shape, 5.9-12 GB/s).
        _buf_all = torch.frombuffer(arena._mm, dtype=torch.uint8)
        self._slot_view = _buf_all[data_off:data_off + A * slot_bytes].view(A, slot_bytes)
        self._state_stage = None
        self._state_loaded_key = None
        self._layout = {"L": L, "t_shape": t_shape, "conv_shape": conv_shape, "width": width,
                        "t_ext": [(int(t_ext[l][0]), int(t_ext[l][1])) for l in range(L)],
                        "c_ext": [[(int(c_ext[3 * l + j][0]), int(c_ext[3 * l + j][1])) for j in range(3)] for l in range(L)]}
        self._pin = bool(pin and torch.cuda.is_available())
        buf = torch.frombuffer(arena._mm, dtype=torch.uint8)
        self._pin_base = int(buf.data_ptr()) + data_off
        self._pin_bytes = slot_bytes
        self._pinned = getattr(arena, "_pinned_slots", None)
        if self._pinned is None:
            self._pinned = arena._pinned_slots = torch.zeros(A, dtype=torch.bool)
        if self._pin and os.environ.get("SGLANG_HICACHE_ARENA_PREPIN", "1") == "1" and not bool(self._pinned.all()):
            # #1436: the whole mamba arena registered once at bind (see arena_pool.bind)
            import time as _time
            from sglang.srt.mem_cache.pool_host.arena_pool import _CUDA_HOST_REGISTER_FLAGS, _CUDA_ERROR_ALREADY_REGISTERED
            t0 = _time.perf_counter()
            cudart = torch.cuda.cudart()
            total = A * slot_bytes
            step = max(slot_bytes, ((1 << 30) // slot_bytes) * slot_bytes)
            off = 0
            while off < total:
                n = min(step, total - off)
                rc = cudart.cudaHostRegister(self._pin_base + off, n, _CUDA_HOST_REGISTER_FLAGS)
                if int(rc) not in (0, _CUDA_ERROR_ALREADY_REGISTERED):
                    raise RuntimeError(f"#1436 cudaHostRegister(mamba arena {off}+{n}) failed: {int(rc)}")
                off += n
            self._pinned[:] = True
            logger.info("#1436 mamba arena pre-pinned: %.2f GiB in %.1f s", total / (1 << 30), _time.perf_counter() - t0)
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
    # xsn356: the borrowed _claim/complete_write/abort_write call these too
    _pend_mark = ArenaMHAHostPool._pend_mark
    _pend_pop = ArenaMHAHostPool._pend_pop
    _pending_mask = None   # the mamba pool keeps the dict only (1 state per node)

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

    # -- fnFL2 H19: anchor displacement -----------------------------------------------
    def settled_anchor_slots(self, host_value: Optional[torch.Tensor]) -> Optional[list]:
        """The arena slots of a node's mamba host value when this rank's write
        into them has landed (acked, no pending claim) -- the only anchors a
        displacement may release. None = no arena anchor, or still in flight."""
        if self.arena is None or host_value is None or host_value.numel() == 0:
            return None
        S, A = self.staging_rows, self.arena_slots
        rows = [int(i) - S for i in host_value.cpu().tolist()]
        if any(r < 0 or r >= A for r in rows):
            return None
        if any(r in self._pending for r in rows):
            return None
        return rows

    def drop_unreferenced(self, slots: Sequence[int]) -> int:
        """Free the displaced anchors' slots once no rank references them."""
        if self.arena is None or not slots:
            return 0
        return self.arena.drop_unreferenced(list(slots))

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
            if slot < 0 or state != 2:
                # #1433: the L3 return path -- a blob on disk is read straight
                # into a fresh slot and completed.
                fill = getattr(backend, "arena_fill_from_disk", None)
                f = fill(self.arena, [stems[i]], int(self._page_bytes))[0] if callable(fill) else None
                if f is not None:
                    slot, state = f, 2
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
            sel = torch.tensor([i for _, i in pairs], dtype=torch.int64)
            self.pin_slots(slots)
            dev = device_pool.mamba_cache.temporal.device
            if device_indices.device.type == "cuda":
                didx_d = device_indices.index_select(0, sel.pin_memory().to(dev, non_blocking=True))
            else:
                didx_d = device_indices[sel].to(dev)
            if ple_state.enabled():
                # H63c: the PLE side states beside the GDN blob, on the same
                # (write) stream, so the write's ack covers them too
                ple_state.side_write(self.arena, device_pool, slots.tolist(), didx_d)
            if self._mamba_write_kernel(device_pool, slots, didx_d, dev):
                return self._backup_rest(device_pool, host_indices, device_indices, io_backend, is_arena)
            didx = didx_d.cpu()
            for l in range(int(self.num_mamba_layers)):
                src = device_pool.mamba_cache.temporal[l].index_select(0, didx_d).to("cpu")
                self._t_views[l].index_copy_(0, slots, src)
                srcc = device_pool.mamba_cache.conv[0][l].index_select(0, didx_d).to("cpu")
                for ch0, n, view in self._c_views[0][l]:
                    view.index_copy_(0, slots, srcc[:, ch0:ch0 + n].contiguous())
        return self._backup_rest(device_pool, host_indices, device_indices, io_backend, is_arena)

    def _mamba_write_kernel(self, device_pool, slots, didx_d, dev) -> bool:
        """xsn351 (py-spy PP0): the per-layer `.to("cpu")` + index_copy_ of the
        node's mamba state ran SYNCHRONOUSLY in the scheduler thread (~30 ms
        per chunk). The pointer/stride kernel writes every piece (temporal row,
        conv channel segments) straight from the card into the pinned slot on
        the write stream; the write op's finish event covers it and the
        complete happens at the ack, as for the KV pages. True = done here."""
        from sglang.srt.weg2 import arena_write as _aw
        if dev.type != "cuda" or getattr(self, "_mamba_kernel_off", False) or _aw.mamba_write_mode() != "kernel":
            return False
        try:
            from sglang.jit_kernel.hicache import transfer_hicache_all_layer_mla
            temporal = device_pool.mamba_cache.temporal
            conv = device_pool.mamba_cache.conv[0]
            # xsn354: the pseudo-layer pointer arrays (52k entries on PP0) do not
            # depend on the slot or the device row -- the kernel adds
            # indices * stride -- so they are built ONCE per pool/device layout
            # and kept on the card; per node only slots and didx cross.
            key = (int(temporal.data_ptr()), int(conv.data_ptr()), str(dev))
            plan = getattr(self, "_mamba_kern_plan", None)
            if plan is None or plan[0] != key:
                L = int(self.num_mamba_layers)
                e_t = int(self.temporal_dtype.itemsize)
                e_c = int(self.conv_dtype.itemsize)
                pieces = []
                for l in range(L):
                    t_src = temporal[l]
                    pieces.append((self._t_views[l][0].numel() * e_t, self._t_views[l].data_ptr(),
                                   t_src.data_ptr(), t_src.stride(0) * e_t))
                    c_src = conv[l]
                    per_ch = int(c_src.shape[-1]) * e_c
                    for ch0, n, view in self._c_views[0][l]:
                        pieces.append((n * per_ch, view.data_ptr(),
                                       c_src.data_ptr() + ch0 * per_ch, c_src.stride(0) * e_c))
                # xsn353: fixed element size (1024 B, the KV cell -- its JIT module
                # is cached); pieces that are not whole elements take the copy path.
                kern, rest = _aw.split_pieces(pieces, _aw.mamba_element_bytes())
                if rest:
                    if not getattr(self, "_mamba_rest_said", False):
                        self._mamba_rest_said = True
                        logger.info("WEG2-MAMBA-WRITE %d piece(s) are not whole %d-B elements (e.g. %d B): copy mode",
                                    len(rest), _aw.mamba_element_bytes(), int(rest[0][0]))
                    return False
                groups = []
                for (es, ss), (dps, sps) in _aw.group_pieces(kern).items():
                    groups.append((int(es), int(ss),
                                   torch.tensor(dps, dtype=torch.uint64).to(dev),
                                   torch.tensor(sps, dtype=torch.uint64).to(dev)))
                torch.cuda.synchronize(dev)
                plan = self._mamba_kern_plan = (key, groups, len(pieces))
            _, groups, n_pieces = plan
            pieces = [None] * n_pieces
            slots_d = slots.to(dtype=torch.int64).pin_memory().to(dev, non_blocking=True)
            didx_d = didx_d.to(dtype=torch.int64)
            for es, ss, ptr_dst, ptr_src in groups:
                transfer_hicache_all_layer_mla(
                    ptr_dst=ptr_dst,
                    indices_dst=slots_d,
                    ptr_src=ptr_src,
                    indices_src=didx_d,
                    cache_src_stride_bytes=int(ss),
                    cache_dst_stride_bytes=int(self.arena.slot_bytes),
                    element_size=int(es),
                    block_quota=_aw.write_block_quota(),
                )
            n = getattr(type(self), "_mamba_kernel_n", 0) + 1
            type(self)._mamba_kernel_n = n
            if n <= 8 or n % 256 == 0:
                logger.info("WEG2-MAMBA-WRITE n=%d states=%d pieces=%d launches=%d mode=kernel (plan cached)",
                            n, int(slots.numel()), len(pieces), len(groups))
            return True
        except Exception as exc:  # noqa: BLE001 -- one named fallback, then the copy path
            logger.warning("WEG2-MAMBA-WRITE kernel mode failed (%s: %s); copy mode from now on",
                           type(exc).__name__, exc)
            self._mamba_kernel_off = True
            return False

    def _backup_rest(self, device_pool, host_indices, device_indices, io_backend, is_arena):
        rest = (~is_arena).nonzero(as_tuple=True)[0]
        if rest.numel():
            super().backup_from_device_all_layer(
                device_pool, host_indices[rest.to(host_indices.device)],
                device_indices[rest.to(device_indices.device)], io_backend)

    def backup_accepts_device_indices(self, host_indices, device_indices) -> str:
        """H2: all-arena host ids are written by the kernel (xsn351) or copy
        path above, both of which select the device rows on the card; the
        staging rows' base branch wants host indices."""
        if self.arena is None:
            return "mamba:arena_unbound"
        if host_indices.is_cuda:
            return "mamba:host_ids_on_card"
        _, is_arena = self._split(host_indices)
        if not bool(is_arena.all()):
            return "mamba:staging_rows"
        return ""

    def backup_from_device_indices(self, device_pool, host_indices, device_indices):
        # all arena rows (accepted above): _backup_rest is empty, the backend
        # argument is never read
        self.backup_from_device_all_layer(device_pool, host_indices, device_indices, "direct")

    def _load_states_all_layers(self, device_pool, slots, didx) -> None:
        """xsn332/337: this rank's extents of the requested slots (temporal +
        conv, all layers) gathered COMPACTLY into a pinned stage, one H2D per
        block, split on the device. xsn337: copying the whole 78-MiB canonical
        blob per state (every rank's extents) made the pass slower (706 ms);
        the compact stage carries only this rank's bytes.

        H12: in "dma" mode (``_arena_page_load_mode`` of the slot size, the
        default for the 78-MiB blob) the extents go straight from the
        registered arena into the device stage -- fnFL2x104 TP0 spent
        components_ms mamba=91 on two pinned CPU gathers of 58.8 MB."""
        n = int(slots.numel())
        if n == 0:
            return
        lay = self._layout
        dev = device_pool.mamba_cache.temporal[0].device
        _pin = bool(dev.type == "cuda")
        # compact layout: (cursor, off, ln) per extent in layer order: t, c0, c1, c2
        if getattr(self, "_compact", None) is None:
            cur, comp = 0, []
            for l in range(int(lay["L"])):
                off, ln = lay["t_ext"][l]
                comp.append((cur, off, ln)); cur += ln
                for (off_j, ln_j) in lay["c_ext"][l]:
                    comp.append((cur, off_j, ln_j)); cur += ln_j
            self._compact = (comp, cur)
        comp, row_bytes = self._compact
        B = max(1, _arena_state_load_block_bytes() // max(1, row_bytes))
        bb = min(B, n)
        dma = _arena_page_load_mode(self._page_bytes) == "dma"
        if dma:
            runs = merge_state_extents(comp)
        elif self._state_stage is None or self._state_stage.shape[0] < bb or self._state_stage.shape[1] != row_bytes:
            self._state_stage = torch.empty((bb, row_bytes), dtype=torch.uint8, pin_memory=_pin)
        if getattr(self, "_state_dev_stage", None) is None or self._state_dev_stage.shape[0] < bb \
                or self._state_dev_stage.shape[1] != row_bytes or self._state_dev_stage.device != dev:
            self._state_dev_stage = torch.empty((bb, row_bytes), dtype=torch.uint8, device=dev)
        slots_cpu = slots.to("cpu", dtype=torch.int64)
        # SGLANG_HICACHE_LOAD_ASYNC_INDEX: pinned + non_blocking instead of a
        # pageable `.to(dev)` (a host wait on the forward-fenced load stream).
        didx = index_to_device_async(didx, dev) if load_index_async() else didx.to(dev)
        e_c = int(self.conv_dtype.itemsize)
        t_shape = tuple(lay["t_shape"]); conv_shape = tuple(lay["conv_shape"]); width = int(lay["width"])
        L = int(lay["L"])
        for start in range(0, n, B):
            b = min(B, n - start)
            sl = slots_cpu[start:start + b]
            dev_stage = self._state_dev_stage[:b]
            if dma:
                _state_block_dma(self._slot_view, dev_stage, sl.tolist(), runs)
            else:
                stage = self._state_stage[:b]
                for (cur, off, ln) in comp:
                    torch.index_select(self._slot_view[:, off:off + ln], 0, sl, out=stage[:, cur:cur + ln])
                dev_stage.copy_(stage, non_blocking=_pin)
            d_b = didx[start:start + b]
            k = 0
            for l in range(L):
                cur, _off, ln = comp[k]; k += 1
                src = dev_stage[:, cur:cur + ln].contiguous().view(self.temporal_dtype).view((b,) + t_shape)
                device_pool.mamba_cache.temporal[l].index_copy_(0, d_b, src)
                dst_c = device_pool.mamba_cache.conv[0][l]
                row = torch.empty((b,) + conv_shape, dtype=dst_c.dtype, device=dev)
                ch0 = 0
                for _j in range(3):
                    cur_j, _off_j, ln_j = comp[k]; k += 1
                    n_j = ln_j // (width * e_c)
                    row[:, ch0:ch0 + n_j] = dev_stage[:, cur_j:cur_j + ln_j].contiguous().view(self.conv_dtype).view(b, n_j, width)
                    ch0 += n_j
                dst_c.index_copy_(0, d_b, row)
            if _pin and not dma and start + b < n:
                torch.cuda.current_stream(dev).synchronize()  # the stage is reused by the next block
        global _STATE_LOAD_N
        _STATE_LOAD_N += 1
        if _STATE_LOAD_N <= 8 or _STATE_LOAD_N % 64 == 0:
            logger.info("WEG2-ARENA-STATE-LOAD n=%d slots=%d own_bytes=%d block=%d mode=%s (this rank's extents, %s, split on device)",
                        _STATE_LOAD_N, n, n * row_bytes, B, "dma" if dma else "cpu",
                        f"{len(runs)} copies per slot from the arena" if dma else "pinned gather + one H2D per block")

    def load_to_device_per_layer(self, device_pool, host_indices, device_indices, layer_id, io_backend="direct"):
        if self.arena is None or host_indices.numel() == 0:
            return super().load_to_device_per_layer(device_pool, host_indices, device_indices, layer_id, io_backend)
        hi, is_arena = self._split(host_indices)
        if not bool(is_arena.any()):
            return super().load_to_device_per_layer(device_pool, host_indices, device_indices, layer_id, io_backend)
        sel = is_arena.nonzero(as_tuple=True)[0]
        slots = hi[sel] - self.staging_rows
        if _arena_state_load_on() and getattr(self, "_slot_view", None) is not None:
            key = (id(host_indices), id(device_indices), int(host_indices.numel()), int(device_indices.numel()))
            if layer_id != 0 and self._state_loaded_key == key:
                rest = (~is_arena).nonzero(as_tuple=True)[0]
                if rest.numel():
                    super().load_to_device_per_layer(
                        device_pool, host_indices[rest.to(host_indices.device)],
                        device_indices[rest.to(device_indices.device)], layer_id, io_backend)
                return  # every layer came with the state load at layer 0
            if layer_id == 0:
                try:
                    if load_index_async():
                        # SGLANG_HICACHE_LOAD_ASYNC_INDEX: the device rows stay on
                        # the card (selected there when not every row is an arena
                        # row) -- `.cpu()` of the mamba slot tensor is a host wait
                        # on the forward-fenced load stream. Same rows, same order.
                        _didx = (device_indices if bool(is_arena.all())
                                 else _select_rows_async(device_indices, is_arena))
                    else:
                        _didx = device_indices.cpu()[sel]
                    self._load_states_all_layers(device_pool, slots, _didx)
                    self._state_loaded_key = key
                    if ple_state.enabled():
                        # H63c: the anchors' PLE side states into the same
                        # targets, on the load stream (before the PLE read's join);
                        # the rows are selected on the card, never `.cpu()`
                        ple_state.side_read(self.arena, device_pool, slots.tolist(), device_indices, sel)
                    rest = (~is_arena).nonzero(as_tuple=True)[0]
                    if rest.numel():
                        super().load_to_device_per_layer(
                            device_pool, host_indices[rest.to(host_indices.device)],
                            device_indices[rest.to(device_indices.device)], layer_id, io_backend)
                    return
                except Exception as exc:  # noqa: BLE001 -- one named fallback to the per-layer path
                    logger.warning("WEG2-ARENA-STATE-LOAD failed (%s: %s); per-layer path", type(exc).__name__, exc)
                    self._state_loaded_key = None
        dst_t = device_pool.mamba_cache.temporal[layer_id]
        dev = dst_t.device
        didx = device_indices.cpu()[sel].to(dev)
        dst_t.index_copy_(0, didx, self._t_views[layer_id].index_select(0, slots).to(dev))
        dst_c = device_pool.mamba_cache.conv[0][layer_id]
        row = torch.empty((int(slots.numel()),) + tuple(dst_c.shape[1:]), dtype=dst_c.dtype)
        for ch0, n, view in self._c_views[0][layer_id]:
            row[:, ch0:ch0 + n] = view.index_select(0, slots)
        dst_c.index_copy_(0, didx, row.to(dev))
        if layer_id == 0 and ple_state.enabled():
            # H63c: the per-layer path's layer 0 carries the PLE side states
            ple_state.side_read(self.arena, device_pool, slots.tolist(), didx)
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
