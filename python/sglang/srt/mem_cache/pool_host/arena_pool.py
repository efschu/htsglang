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
import time

from sglang.srt.managers.weg2_pass_timer import timed as _pass_timed
from typing import Optional, Sequence

import torch

from sglang.jit_kernel.hicache import (
    transfer_hicache_all_layer as jit_transfer_hicache_all_layer,
)

from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

logger = logging.getLogger(__name__)

ENV_ARENA_HOST = "SGLANG_HICACHE_ARENA_HOST"
ENV_STAGING_GB = "SGLANG_HICACHE_ARENA_STAGING_GB"
PLACEHOLDERS = 1 << 22
_CUDA_HOST_REGISTER_FLAGS = 3  # Portable | Mapped
_CUDA_ERROR_ALREADY_REGISTERED = 712


ARENA_PAGE_LOAD_ENV = "SGLANG_WEG2_ARENA_PAGE_LOAD"
ARENA_PAGE_LOAD_BLOCK_ENV = "SGLANG_WEG2_ARENA_PAGE_LOAD_BLOCK"


_PAGE_LOAD_N = 0


class _NoEvent:
    """Stand-in for torch.cuda.Event on a CPU-only desk (tests)."""
    def record(self, *a, **k):
        pass

    def synchronize(self):
        pass


def _arena_page_load_on() -> bool:
    """Posten 2 (18.09.): whole-page loadback, standard on; =0 for an A/B."""
    return str(os.environ.get(ARENA_PAGE_LOAD_ENV, "1")).strip().lower() not in ("0", "false", "no", "off")


def _arena_page_load_mode() -> str:
    m = str(os.environ.get("SGLANG_WEG2_ARENA_PAGE_LOAD_MODE", "kernel")).strip().lower()
    return "cpu" if m == "cpu" else "kernel"


def _arena_page_load_timing() -> bool:
    return str(os.environ.get("SGLANG_WEG2_ARENA_PAGE_LOAD_TIMING", "0")).strip() not in ("", "0")


def _arena_page_load_block() -> int:
    try:
        return max(64, int(os.environ.get(ARENA_PAGE_LOAD_BLOCK_ENV, "2048")))
    except ValueError:
        return 2048


def _arena_load_block_quota():
    """Task #3 (17.09.): the JIT gather's block quota for the ARENA -> device
    load. The kernel default (2 blocks = 64 warps in flight) is tuned for
    interference with a running forward; the re-admission after a wake
    runs on an idle card and is bounded by outstanding PCIe reads of 1-KiB
    cells (~1 GB/s per rank measured on xsn246). None keeps the default."""
    v = os.environ.get("SGLANG_HICACHE_ARENA_LOAD_BLOCK_QUOTA", "").strip()
    try:
        return int(v) if v else None
    except ValueError:
        return None


def arena_host_enabled() -> bool:
    return os.environ.get(ENV_ARENA_HOST, "0") == "1"


class ArenaMHAHostPool(MHATokenToKVPoolHost):
    """MHA host pool whose rows beyond the staging ring are arena slots."""

    arena_read = True

    def __init__(self, device_pool, host_to_device_ratio, host_size, page_size, layout,
                 *args, **kwargs):
        # #1430: the staging rows are a FALLBACK id range only (nodes without
        # hashes on a non-arena boot); every hashed page is claimed and
        # written in the arena. 0.05 GB (~1.5k rows) instead of the 1 GB
        # ring the transit needed -- ~6 GB of pinned host RAM on P.
        staging_gb = float(os.environ.get(ENV_STAGING_GB, "0.05") or 0.05)
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
        # #1427 Stufe 4 (direct writes): the backend that names the keys, this
        # rank's extents inside a page, the slots this pool still has to
        # complete (slot -> (generation, fresh)), and the per-layer device
        # pointers into the arena data region for the transfer kernel.
        self._backend = None
        self._own_extents: Optional[list] = None
        self._page_bytes = 0
        self._data_base = 0
        self._k_off = 0
        self._v_off = 0
        self._cell = 0
        self._k_offs: list = []
        self._v_offs: list = []
        self._pending: dict = {}
        self.arena_k_ptrs = None
        self.arena_v_ptrs = None

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
            self._backend = storage_backend
            return True
        except Exception as exc:  # noqa: BLE001 - loud, never silent
            logger.error("#1424 arena host pool bind failed (role=%s): %r", role, exc)
            return False

    def bind(self, arena, window, role: str = "kv", pin: bool = True) -> None:
        if int(self.page_size) != 1:
            raise ValueError("#1424 the arena host pool needs page_size 1 (one slot per token)")
        ext = [(int(o), int(l)) for o, l in window.extents]
        if len(ext) == 1 and ext[0][0] == 0 and ext[0][1] == int(window.total_bytes):
            # the whole page (D's DCP ranks hold every layer and head): the
            # canonical page is K-major, [K all slots][V all slots]
            half = int(window.total_bytes) // 2
            ext = [(0, half), (half, half)]
        L = int(self.layer_num)
        e = int(self.dtype.itemsize)
        cell = int(self.head_num) * int(self.head_dim) * e
        if len(ext) == 2:
            (k_off, k_len), (v_off, v_len) = ext
            if k_len != v_len or k_len != L * cell:
                raise ValueError(f"#1424 window extents {ext} do not match {L} layers x {cell} B")
            k_offs = [k_off + l * cell for l in range(L)]
            v_offs = [v_off + l * cell for l in range(L)]
        elif len(ext) == 2 * L:
            # xsn267 (17.09., DFlash draft on D): a HEAD-SHARDED window names
            # its K and V extent PER LAYER -- the canonical draft page is
            # [K L0 | V L0 | K L1 | ...] and this rank owns a head slice of
            # each (TP0 5 of 8 heads: (0,640),(1024,640),...; TP1/TP2 2 heads:
            # (640,256),(1664,256),...). Even extents are K, odd V; every
            # extent is exactly this pool's own cell. The 2-extent form above
            # is the whole-page special case of this one.
            bad = [x for x in ext if x[1] != cell]
            if bad:
                raise ValueError(f"#1424 per-layer window extents {ext} carry lengths other "
                                 f"than this pool's cell {cell} B: {bad[:3]}")
            k_offs = [ext[2 * l][0] for l in range(L)]
            v_offs = [ext[2 * l + 1][0] for l in range(L)]
            k_off, v_off = k_offs[0], v_offs[0]
        else:
            raise ValueError(f"#1424 expected a K and a V extent (or {2 * L} per-layer "
                             f"K/V extents), got {ext}")
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
            typed.as_strided((A, H, D), (tok, D, 1), storage_offset=base_off + k_offs[l] // e)
            for l in range(L)
        ]
        self.arena_v_refs = [
            typed.as_strided((A, H, D), (tok, D, 1), storage_offset=base_off + v_offs[l] // e)
            for l in range(L)
        ]
        # Posten 2 (18.09.): WHOLE-PAGE loadback. One arena slot is one page
        # holding K and V of EVERY layer; the per-layer gather kernel read it
        # as 2*L separate 1-KiB rows over PCIe (0,6-1,5 GB/s, xsn303). The
        # page view lets the load fetch 32-KiB pages once and split layers on
        # the device (`_load_pages_all_layers`).
        self._page_view = region.view(A, page_bytes)
        self._page_bytes = int(page_bytes)
        self._k_offs_b = [int(k_offs[l]) for l in range(L)]
        self._v_offs_b = [int(v_offs[l]) for l in range(L)]
        self._page_loaded_key = None
        self._page_stages = None
        # #1424e (boot xsn177/179): registering the WHOLE mapping materialised
        # every untouched tmpfs slot (27 GiB resident at once); the ledger
        # measured it as a 37.95 GiB flip ratchet and refused every later
        # boot. Pages are pinned LAZILY, per slot, at first use instead -- the
        # arena stays sparse and residency equals what was actually written.
        self._pin = bool(pin and torch.cuda.is_available())
        self._pin_base = int(buf.data_ptr()) + data_off
        self._pin_bytes = page_bytes
        self._own_extents = list(ext)
        self._k_offs, self._v_offs = list(k_offs), list(v_offs)
        self._page_bytes = page_bytes
        self._data_base = self._pin_base
        self._k_off, self._v_off, self._cell = k_off, v_off, cell
        self._pinned = getattr(arena, "_pinned_slots", None)
        if self._pinned is None:
            self._pinned = arena._pinned_slots = torch.zeros(A, dtype=torch.bool)
        self._all_pinned = bool(self._pinned.all())
        # #1436 (xsn196): pinning slot runs on first use cost 3.4 % of P's
        # prefill loop and sat on D's re-admission of every 100k prompt.
        # Register the whole data region once at bind, in 1 GiB pieces; the
        # bytes are priced in full by the ledger (#1432 arena_gib), so the
        # #1424e residency argument no longer applies. PREPIN=0 keeps lazy.
        if self._pin and os.environ.get("SGLANG_HICACHE_ARENA_PREPIN", "1") == "1" and not bool(self._pinned.all()):
            t0 = time.perf_counter()
            cudart = torch.cuda.cudart()
            total = A * page_bytes
            step = max(page_bytes, ((1 << 30) // page_bytes) * page_bytes)
            off = 0
            while off < total:
                n = min(step, total - off)
                rc = cudart.cudaHostRegister(self._pin_base + off, n, _CUDA_HOST_REGISTER_FLAGS)
                if int(rc) not in (0, _CUDA_ERROR_ALREADY_REGISTERED):
                    raise RuntimeError(f"#1436 cudaHostRegister(arena {off}+{n}) failed: {int(rc)}")
                off += n
            self._pinned[:] = True
            self._all_pinned = True
            logger.info("#1436 arena pre-pinned: %.2f GiB in %.1f s (role=%s)", total / (1 << 30),
                        time.perf_counter() - t0, role)
        # Task #3: build the load's JIT variant NOW (boot), not inside the
        # first re-admission after a wake (a cold JIT build there is seconds
        # on the flip tail's critical path).
        try:
            _q = _arena_load_block_quota()
            if _q and torch.cuda.is_available():
                from sglang.jit_kernel.hicache import can_use_hicache_jit_kernel
                can_use_hicache_jit_kernel(element_size=int(cell), block_quota=int(_q))
        except Exception as exc:  # noqa: BLE001 -- a warm-up never refuses a bind
            logger.info("#task3 arena load JIT warm-up skipped: %r", exc)
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

    @_pass_timed("_1474_alloc_ms")  # #1474
    def alloc_read(self, n: int) -> torch.Tensor:
        """Placeholders for a prefetch registration; resolved in place later."""
        base = self.staging_rows + self.arena_slots
        start = self._read_ph_next
        self._read_ph_next = (start + n) % PLACEHOLDERS
        return (torch.arange(start, start + n, dtype=torch.int64) % PLACEHOLDERS) + base

    def resolve_rows(self, host_indices: torch.Tensor, slots: Sequence[int]) -> None:
        """KV role: write arena ids over the placeholders, in place."""
        n = len(slots)
        self.pin_slots(slots)
        vals = torch.as_tensor(list(slots) if not torch.is_tensor(slots) else slots, dtype=host_indices.dtype) + self.staging_rows
        host_indices[:n] = vals.to(host_indices.device)

    def resolve_draft_rows(self, rows: Sequence[int], slots: Sequence[int]) -> None:
        """Draft role: remember the DRAFT arena slot behind each KV row (-1 = miss)."""
        if self.row_slot is None:
            raise RuntimeError("#1424 resolve_draft_rows on a pool not bound as draft")
        self.row_slot.update(zip(map(int, rows), map(int, slots)))
        self.pin_slots([int(s) for s in slots if int(s) >= 0])

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

    def _arena_load_guard(self, device_pool, slots, device_indices, layer_id, *, nrows: int, nmiss: int) -> None:
        """weg2xsn277 (18.09.): the first 98k-token arena->device load of a
        boot died with 'CUDA error: an illegal memory access' reported
        asynchronously at `_transfer` (the 4k smoke loaded fine). The kernel
        takes raw slot/row indices and 64-bit strides; a slot outside the
        arena or a destination row outside the device pool is exactly the
        fault that surfaces one kernel later with no name. Refuse BY NAME
        before the launch, and print the load's terms once per load (layer
        0) so the next log carries the numbers."""
        A = int(self.arena_slots)
        try:
            dst_rows = int(device_pool.k_buffer[layer_id].shape[0])
        except Exception:  # noqa: BLE001 -- a desk double without buffers
            dst_rows = -1
        s_min = int(slots.min()) if slots.numel() else 0
        s_max = int(slots.max()) if slots.numel() else -1
        d_min = int(device_indices.min()) if device_indices.numel() else 0
        d_max = int(device_indices.max()) if device_indices.numel() else -1
        pinned = "n/a"
        bm = getattr(self, "_pinned", None)
        if bm is not None and slots.numel():
            try:
                pinned = "all" if bool(bm[slots.to("cpu")].all()) else "PARTIAL"
            except Exception:  # noqa: BLE001
                pinned = "?"
        if int(layer_id) == 0:
            logger.info(
                "WEG2-ARENA-LOAD rows=%d hits=%d miss=%d slot=[%d,%d] of %d dst=[%d,%d] of %d "
                "pinned=%s layer=%d",
                nrows, int(slots.numel()), nmiss, s_min, s_max, A, d_min, d_max, dst_rows,
                pinned, int(layer_id),
            )
        if slots.numel() and (s_min < 0 or s_max >= A):
            raise RuntimeError(
                f"#1424 WEG2-ARENA-LOAD REFUSED: slot range [{s_min},{s_max}] outside the "
                f"arena of {A} slots (layer {layer_id}, {nrows} rows) -- the kernel would "
                f"read past the arena mapping (xsn277: illegal memory access)")
        if dst_rows > 0 and device_indices.numel() and (d_min < 0 or d_max >= dst_rows):
            raise RuntimeError(
                f"#1424 WEG2-ARENA-LOAD REFUSED: device rows [{d_min},{d_max}] outside the "
                f"pool of {dst_rows} rows (layer {layer_id}, {nrows} rows)")
        if pinned == "PARTIAL":
            raise RuntimeError(
                f"#1424 WEG2-ARENA-LOAD REFUSED: {int(slots.numel())} slot(s) of layer "
                f"{layer_id} are not all registered (pinned bitmap PARTIAL) -- a device "
                f"read of an unregistered host page is an illegal address")

    def _load_arena(self, device_pool, rows, device_indices, layer_id) -> None:
        if _arena_page_load_on() and getattr(self, "_page_view", None) is not None:
            key = (id(rows), id(device_indices), int(rows.numel()), int(device_indices.numel()))
            if layer_id != 0 and self._page_loaded_key == key:
                return  # every layer came with the page load at layer 0
            if layer_id == 0:
                slots = rows
                if self.row_slot is not None:
                    rl = rows.tolist()
                    slots = torch.tensor([self.row_slot.get(int(r), -1) for r in rl], dtype=rows.dtype, device=rows.device)
                if slots.numel() and bool((slots >= 0).all()):
                    self.pin_slots(slots)
                    self._arena_load_guard(device_pool, slots, device_indices, 0, nrows=int(rows.numel()), nmiss=0)
                    self._load_pages_all_layers(device_pool, slots, device_indices)
                    self._page_loaded_key = key
                    return
                # misses (row_slot without a slot) keep the per-layer path below
            self._page_loaded_key = None
        if self.row_slot is not None:
            rl = rows.tolist()
            slots = torch.tensor([self.row_slot.get(int(r), -1) for r in rl], dtype=rows.dtype, device=rows.device)
            miss = slots < 0
            if bool(miss.any()):
                z = torch.zeros((int(miss.sum()),), dtype=rows.dtype, device=rows.device)
                self._transfer(device_pool, self._zero_k, self._zero_v, z, device_indices[miss], layer_id)
            hit = ~miss
            if bool(hit.any()):
                self._arena_load_guard(device_pool, slots[hit], device_indices[hit], layer_id,
                                       nrows=int(rows.numel()), nmiss=int(miss.sum()))
                self._transfer(device_pool, self.arena_k_refs[layer_id], self.arena_v_refs[layer_id],
                               slots[hit], device_indices[hit], layer_id)
            return
        self.pin_slots(rows)
        self._arena_load_guard(device_pool, rows, device_indices, layer_id, nrows=int(rows.numel()), nmiss=0)
        self._transfer(device_pool, self.arena_k_refs[layer_id], self.arena_v_refs[layer_id],
                       rows, device_indices, layer_id)

    def _load_pages_all_layers(self, device_pool, slots, device_indices) -> None:
        """Fetch whole 32-KiB pages (all layers of a token) in blocks through
        a pinned host stage and one H2D copy per block, then scatter each
        layer on the device. Two pinned stages alternate so the CPU gather of
        block i+1 overlaps the DMA of block i."""
        B = _arena_page_load_block()
        n = int(slots.numel())
        if n == 0:
            return
        dev = device_pool.k_buffer[0].device
        pb = self._page_bytes
        H, D = int(self.head_num), int(self.head_dim)
        e = self.dtype.itemsize
        cell = H * D * e
        if self._page_stages is None or self._page_stages[0].shape[0] < B:
            _pin = bool(dev.type == "cuda")
            self._page_stages = (
                torch.empty((B, pb), dtype=torch.uint8, pin_memory=_pin),
                torch.empty((B, pb), dtype=torch.uint8, pin_memory=_pin),
            )
            self._page_events = ((torch.cuda.Event(), torch.cuda.Event()) if _pin
                                 else (_NoEvent(), _NoEvent()))
        cpu_slots = slots.to("cpu", dtype=torch.int64)
        dev_stage = torch.empty((min(B, n), pb), dtype=torch.uint8, device=dev)
        L = len(self._k_offs_b)
        # xsn305: the CPU gather of three ranks at once (3 x 3 GB memcpy) did
        # not beat the per-layer kernel (0,5-1,95 GB/s). Mode "kernel" lets
        # the GPU gather WHOLE PAGES straight from the mapped arena (the MLA
        # one-buffer kernel with element_dim = page bytes); "cpu" is the
        # pinned-stage form; the first kernel failure falls back to cpu.
        mode = getattr(self, "_page_mode", None) or _arena_page_load_mode()
        if mode == "kernel" and dev.type != "cuda":
            mode = "cpu"
        _timing = _arena_page_load_timing()
        _t0 = time.perf_counter() if _timing else 0.0
        _gather_ms = 0.0
        for bi, start in enumerate(range(0, n, B)):
            b = min(B, n - start)
            k = bi % 2
            if mode == "kernel":
                try:
                    from sglang.jit_kernel.hicache import transfer_hicache_one_layer_mla
                    src_idx = slots[start:start + b].to(device=dev, dtype=torch.int64)
                    stage_idx = torch.arange(b, device=dev, dtype=torch.int64)
                    transfer_hicache_one_layer_mla(
                        cache_dst=dev_stage[:b], indices_dst=stage_idx,
                        cache_src=self._page_view, indices_src=src_idx,
                        element_dim=pb, block_quota=_arena_load_block_quota())
                except Exception as exc:  # noqa: BLE001 -- one named fallback, then cpu
                    logger.warning("WEG2-ARENA-PAGE-LOAD kernel mode failed (%s: %s); cpu mode from now on",
                                   type(exc).__name__, exc)
                    self._page_mode = mode = "cpu"
            if mode == "cpu":
                self._page_events[k].synchronize()  # the DMA that last read this stage is done
                host_stage = self._page_stages[k]
                _g0 = time.perf_counter()
                torch.index_select(self._page_view, 0, cpu_slots[start:start + b], out=host_stage[:b])
                _gather_ms += (time.perf_counter() - _g0) * 1000.0
                dev_stage[:b].copy_(host_stage[:b], non_blocking=True)
                self._page_events[k].record()
            dst = device_indices[start:start + b].to(device=dev, dtype=torch.int64)
            for l in range(L):
                ko, vo = self._k_offs_b[l], self._v_offs_b[l]
                device_pool.k_buffer[l].index_copy_(
                    0, dst, dev_stage[:b, ko:ko + cell].reshape(-1).view(self.dtype).view(b, H, D))
                device_pool.v_buffer[l].index_copy_(
                    0, dst, dev_stage[:b, vo:vo + cell].reshape(-1).view(self.dtype).view(b, H, D))
        global _PAGE_LOAD_N
        _PAGE_LOAD_N += 1
        n_log = _PAGE_LOAD_N
        _wall = ""
        if _timing and dev.type == "cuda":
            torch.cuda.current_stream(dev).synchronize()
            _ms = (time.perf_counter() - _t0) * 1000.0
            _wall = f" wall_ms={_ms:.0f} GB/s={(n * pb) / max(_ms, 1e-3) / 1e6:.2f} gather_ms={_gather_ms:.0f}"
        if n_log <= 8 or n_log % 64 == 0 or _timing:
            logger.info("WEG2-ARENA-PAGE-LOAD n=%d rows=%d pages=%d block=%d bytes=%d mode=%s%s (whole pages, layers split on device)",
                        n_log, n, n, B, n * pb, mode, _wall)

    def pin_slots(self, slots) -> int:
        """Register the slots' pages (contiguous runs in one call each) that
        are not registered yet in this process. Returns the number of slots
        newly pinned."""
        if not getattr(self, "_pin", False):
            return 0
        # Task #3 (17.09.): with SGLANG_HICACHE_ARENA_PREPIN=1 (the default)
        # every slot is registered at bind; the list/unique/index walk per
        # 128-page batch below was 2.7 s of xsn246's 262k-page re-admission
        # (ARENA-GET resolve+pin_ms) for an answer that is always 0.
        if getattr(self, "_all_pinned", False):
            return 0
        idx = torch.as_tensor(list(slots) if not torch.is_tensor(slots) else slots.cpu(), dtype=torch.int64)
        if idx.numel() == 0:
            return 0
        idx = torch.unique(idx)
        idx = idx[~self._pinned[idx]]
        if idx.numel() == 0:
            return 0
        cudart = torch.cuda.cudart()
        vals = idx.tolist()
        n = 0
        start = prev = vals[0]
        def _reg(a, b):
            rc = cudart.cudaHostRegister(self._pin_base + a * self._pin_bytes,
                                         (b - a + 1) * self._pin_bytes, _CUDA_HOST_REGISTER_FLAGS)
            if int(rc) not in (0, _CUDA_ERROR_ALREADY_REGISTERED):
                raise RuntimeError(f"#1424 cudaHostRegister(slots {a}..{b}) failed: {int(rc)}")
        for v in vals[1:]:
            if v == prev + 1:
                prev = v
                continue
            _reg(start, prev); n += prev - start + 1
            start = prev = v
        _reg(start, prev); n += prev - start + 1
        self._pinned[idx] = True
        return n

    def _transfer(self, device_pool, k_src, v_src, src_idx, dst_idx, layer_id) -> None:
        if not getattr(self, "can_use_jit", False):
            raise RuntimeError("#1424 the arena host pool needs the JIT hicache transfer kernel")
        from sglang.jit_kernel.hicache import transfer_hicache_one_layer
        dst_k = device_pool.k_buffer[layer_id]
        # xsn177: the kernel type-checks its index tensors -- "Tensor match
        # failed for Tensor<4313>[dtype=int64, device=cpu]" -- both live on
        # the card, like the ordinary path's indices.
        dev = dst_k.device
        src_idx = src_idx.to(device=dev, dtype=torch.int64, non_blocking=False)
        dst_idx = dst_idx.to(device=dev, dtype=torch.int64, non_blocking=False)
        transfer_hicache_one_layer(
            k_cache_dst=dst_k,
            v_cache_dst=device_pool.v_buffer[layer_id],
            k_cache_src=k_src,
            v_cache_src=v_src,
            indices_dst=dst_idx,
            indices_src=src_idx,
            element_dim=self.element_dim,
            block_quota=_arena_load_block_quota(),
        )

    # -- #1427 Stufe 4: direct writes, card -> arena slot -------------------------
    def _stems(self, hashes, suffix: str = ""):
        if suffix:
            return [self._backend._get_suffixed_key(f"{h}.{suffix}") for h in hashes]
        return [self._backend._get_suffixed_key(h) for h in hashes]

    def _claim(self, stems):
        """Claim (or join, or find complete) one slot per stem. Returns the
        slot list, or None when a slot could not be had even after one
        arena-to-disk eviction round; fresh claims of a failed batch are
        freed again so the key is not poisoned."""
        arena = self.arena
        totals = [self._page_bytes] * len(stems)
        got = arena.claim_slots(stems, totals)
        if any(st in (3, 4) for _, st, _ in got):
            ev = getattr(self._backend, "_arena_evict_to_disk", None)
            if callable(ev) and any(st == 4 for _, st, _ in got):
                try:
                    ev(arena, max(256, len(stems)))
                except Exception as exc:  # noqa: BLE001 - loud, the claim below decides
                    logger.warning("#1427 arena evict-to-disk failed: %r", exc)
                redo = [i for i, (_, st, _) in enumerate(got) if st == 4]
                again = arena.claim_slots([stems[i] for i in redo], [self._page_bytes] * len(redo))
                for i, g in zip(redo, again):
                    got[i] = g
            if any(st in (3, 4) for _, st, _ in got):
                fresh = [s for s, st, _ in got if st == 0]
                if fresh:
                    arena.free_slots(fresh)
                k = getattr(ArenaMHAHostPool, "_1427_full_n", 0) + 1
                ArenaMHAHostPool._1427_full_n = k
                if k <= 8 or k % 256 == 0:
                    logger.warning("#1427 ARENA-CLAIM REFUSED n=%d pages=%d statuses=%s (4 = no free slot)",
                                   k, len(stems), sorted({st for _, st, _ in got}))
                return None
        slots = []
        for slot, st, gen in got:
            if st == 2:
                arena.ref_slots([slot], +1)      # complete already: reader reference only
            else:
                self._pending[slot] = (gen, st == 0)
            slots.append(slot)
        return slots

    def alloc_write(self, hashes) -> Optional[torch.Tensor]:
        """KV role: one arena slot per page hash, claimed for a direct write.
        Returns the host ids (staging_rows + slot), or None."""
        if self.arena is None or self._backend is None or not hashes:
            return None
        slots = self._claim(self._stems(hashes))
        if slots is None:
            return None
        return torch.tensor([self.staging_rows + s for s in slots], dtype=torch.int64)

    def alloc_write_draft(self, kv_host_indices: torch.Tensor, hashes, comp: str) -> bool:
        """Draft role: claim the draft slot behind each KV arena row."""
        if self.arena is None or self._backend is None or self.row_slot is None:
            return False
        slots = self._claim(self._stems(hashes, comp))
        if slots is None:
            return False
        rows = (kv_host_indices.cpu() - self.staging_rows).tolist()
        for r, s in zip(rows, slots):
            self.row_slot[int(r)] = int(s)
        return True

    def _arena_ptrs(self, device):
        if self.arena_k_ptrs is None or self.arena_k_ptrs.device != device:
            L = int(self.layer_num)
            self.arena_k_ptrs = torch.tensor(
                [self._data_base + int(self._k_offs[l]) for l in range(L)],
                dtype=torch.uint64, device=device)
            self.arena_v_ptrs = torch.tensor(
                [self._data_base + int(self._v_offs[l]) for l in range(L)],
                dtype=torch.uint64, device=device)
        return self.arena_k_ptrs, self.arena_v_ptrs

    def publish_direct(self, hashes, comp: str, device_pool, device_indices: torch.Tensor,
                       storage_backend) -> int:
        """Draft role, PRODUCER path (Weg 2 group P, DFlash): claim one draft
        slot per page hash, copy the rows straight from ``device_pool`` at
        ``device_indices`` into the slots, complete them. No target host row
        is involved and no reader reference is taken: the producer keeps
        nothing, the page lives in the arena (or on disk after an
        evict-to-disk round) until a consumer references it. Returns how
        many of the pages are complete in the arena after this call --
        freshly written, joined, or already there."""
        if not hashes:
            return 0
        if not self.ensure_bound(storage_backend, role="draft"):
            raise RuntimeError(
                "draft arena publish: the draft host pool is not bound to an "
                "arena (no canonical draft page window or no arena dir)"
            )
        if self.row_slot is None:
            raise RuntimeError("draft arena publish: pool is not bound in the draft role")
        slots = self._claim(self._stems(hashes, comp))
        if slots is None:
            return 0
        pending_idx = [i for i, s in enumerate(slots) if s in self._pending]
        complete_idx = [i for i, s in enumerate(slots) if s not in self._pending]
        if pending_idx:
            sl = torch.tensor([slots[i] for i in pending_idx], dtype=torch.int64)
            di = device_indices.reshape(-1)[torch.tensor(pending_idx, device=device_indices.device)]
            self._backup_arena(device_pool, sl, di)
            # the all-layer copy runs on the current stream into pinned arena
            # memory; the completion below publishes the bytes to every rank
            torch.cuda.current_stream().synchronize()
            gens = [self._pending[s][0] for s in sl.tolist()]
            st = self.arena.complete_slots(sl.tolist(), gens, self._own_extents)
            lost = 0
            for s, r in zip(sl.tolist(), st):
                self._pending.pop(s, None)
                lost += int(r == 3)
            if lost:
                logger.warning("#1427 ARENA-COMPLETE LOST %d draft page(s) under the producer", lost)
        if complete_idx:
            # _claim took a reader reference on already-complete pages; the
            # producer holds none.
            self.arena.ref_slots([slots[i] for i in complete_idx], -1)
        return len(slots)

    def _backup_arena(self, device_pool, slots: torch.Tensor, device_indices: torch.Tensor) -> None:
        """This rank's K and V extents of every page go straight from the
        card into the slot (one all-layer kernel, the arena as the host
        buffer: per-layer pointers, page stride)."""
        if slots.numel() == 0:
            return
        self.pin_slots(slots)
        dev = device_pool.k_buffer[0].device
        k_ptrs, v_ptrs = self._arena_ptrs(dev)
        jit_transfer_hicache_all_layer(
            k_ptr_dst=k_ptrs,
            v_ptr_dst=v_ptrs,
            indices_dst=slots.to(device=dev, dtype=torch.int64),
            k_ptr_src=device_pool.k_data_ptrs,
            v_ptr_src=device_pool.v_data_ptrs,
            indices_src=device_indices.to(device=dev, dtype=torch.int64),
            kv_cache_dst_stride_bytes=self._page_bytes,
            kv_cache_src_stride_bytes=self.token_stride_size,
            element_size=self.element_dim * self.dtype.itemsize,
        )

    def backup_from_device_all_layer(self, device_pool, host_indices, device_indices, io_backend):
        if self.arena is None and host_indices.numel():
            # #1427g: an unbound pool handed ids beyond its staging rows (a
            # sibling pool's arena ids) has nothing to write there -- drop
            # them, named, instead of indexing past the staging buffer.
            hi = host_indices.cpu()
            keep = hi < self.staging_rows
            if not bool(keep.all()):
                k = getattr(ArenaMHAHostPool, "_1427g_n", 0) + 1
                ArenaMHAHostPool._1427g_n = k
                if k <= 8 or k % 256 == 0:
                    logger.warning("#1427g unbound arena pool: %d of %d backup ids are not staging rows -- skipped (n=%d)",
                                   int((~keep).sum()), int(hi.numel()), k)
                sel = keep.nonzero(as_tuple=True)[0]
                host_indices = host_indices[sel.to(host_indices.device)]
                device_indices = device_indices[sel.to(device_indices.device)]
        if self.arena is None or host_indices.numel() == 0:
            return super().backup_from_device_all_layer(device_pool, host_indices, device_indices, io_backend)
        hi = host_indices.cpu()
        S, A = self.staging_rows, self.arena_slots
        is_arena = (hi >= S) & (hi < S + A)
        if not bool(is_arena.any()):
            return super().backup_from_device_all_layer(device_pool, host_indices, device_indices, io_backend)
        sel = is_arena.nonzero(as_tuple=True)[0]
        rows = (hi[sel] - S).tolist()
        if self.row_slot is not None:   # draft role: KV row -> draft slot
            pairs = [(self.row_slot.get(int(r), -1), i) for r, i in zip(rows, sel.tolist())]
        else:
            pairs = [(int(r), i) for r, i in zip(rows, sel.tolist())]
        todo = [(s, i) for s, i in pairs if s >= 0 and s in self._pending]
        if todo:
            slots = torch.tensor([s for s, _ in todo], dtype=torch.int64)
            didx = device_indices.cpu()[torch.tensor([i for _, i in todo], dtype=torch.int64)]
            self._backup_arena(device_pool, slots, didx)
        rest = (~is_arena).nonzero(as_tuple=True)[0]
        if rest.numel():
            super().backup_from_device_all_layer(
                device_pool, host_indices[rest.to(host_indices.device)],
                device_indices[rest.to(device_indices.device)], io_backend)

    def _slots_of(self, host_indices: torch.Tensor):
        hi = host_indices.cpu()
        S, A = self.staging_rows, self.arena_slots
        rows = hi[(hi >= S) & (hi < S + A)] - S
        if self.row_slot is not None:
            return [self.row_slot.get(int(r), -1) for r in rows.tolist()]
        return rows.tolist()

    def complete_write(self, host_indices: torch.Tensor) -> int:
        """The copy landed (ack): merge this rank's extents into the pages'
        coverage, take the node's reader reference. Returns how many pages
        this call completed for everyone."""
        if self.arena is None:
            return 0
        slots = [s for s in self._slots_of(host_indices) if s >= 0 and s in self._pending]
        if not slots:
            return 0
        gens = [self._pending[s][0] for s in slots]
        st = self.arena.complete_slots(slots, gens, self._own_extents)
        done = 0
        for s, r in zip(slots, st):
            self._pending.pop(s, None)
            if r == 3:
                k = getattr(ArenaMHAHostPool, "_1427_lost_n", 0) + 1
                ArenaMHAHostPool._1427_lost_n = k
                if k <= 8 or k % 256 == 0:
                    logger.warning("#1427 ARENA-COMPLETE LOST slot=%d (recycled under the writer) n=%d", s, k)
                continue
            done += int(r == 1)
        self.arena.ref_slots([s for s, r in zip(slots, st) if r != 3], +1)
        return done

    def abort_write(self, host_indices: torch.Tensor) -> None:
        """The write never happened: free fresh claims, drop joins, release
        references taken on complete pages."""
        if self.arena is None:
            return
        fresh, refd = [], []
        for s in self._slots_of(host_indices):
            if s < 0:
                continue
            p = self._pending.pop(s, None)
            if p is None:
                refd.append(s)
            elif p[1]:
                fresh.append(s)
        if fresh:
            self.arena.free_slots(fresh)
        if refd:
            self.arena.ref_slots(refd, -1)
        if self.row_slot is not None:
            for r in (host_indices.cpu() - self.staging_rows).tolist():
                self.row_slot.pop(int(r), None)

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
    def available_size(self):
        """#1440 (xsn200): the front's D-seat gate reads this as 'rows D can
        still prefetch into' (WEG2 D-SEAT-WAIT available=1526 -- the staging
        fallback after #1430) and so admitted the six parked 100k prompts one
        at a time although d_bs=6. Bound to the arena, the answer is the
        arena's free slot count; the staging fallback is `staging_free`."""
        if self.arena is None:
            return len(self.free_slots)
        # #1440b (xsn202): 'complete' pages must NOT be subtracted -- a read of
        # a page that is in the arena consumes no row at all (it is resolved
        # in place), so with six 100k prompts COMPLETE in the arena the gate
        # saw 46k 'free' and held the fourth seat again. What a read can
        # still need is a slot per page NOT in the arena (the L3 fill): the
        # slots not held by a writer in flight.
        try:
            st = self.arena.stats()
            return max(0, int(st["slots"]) - int(st["claimed"]))
        except Exception:  # noqa: BLE001
            return len(self.free_slots)

    def staging_free(self) -> int:
        return len(self.free_slots)

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        """Staging rows only (the fallback range); bounded by the staging free
        list, never by the arena answer available_size() gives the front."""
        if need_size > len(self.free_slots):
            return None
        return super().alloc(need_size)

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
            pend = [s for s in slots if s in self._pending]
            if pend:
                fresh = [s for s in pend if self._pending.pop(s)[1]]
                if fresh:
                    self.arena.free_slots(fresh)
                slots = [s for s in slots if s not in pend]
            if slots:
                self.arena.ref_slots(slots, -1)
            freed += len(rows)
        return freed
