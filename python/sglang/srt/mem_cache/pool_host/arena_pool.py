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
    transfer_hicache_all_layer_mla as jit_transfer_hicache_all_layer_mla,
    transfer_hicache_all_layer as jit_transfer_hicache_all_layer,
)

from sglang.srt.mem_cache.pool_host.base import NO_KV_RANK_TOKENS
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

logger = logging.getLogger(__name__)

ENV_ARENA_HOST = "SGLANG_HICACHE_ARENA_HOST"
ENV_STAGING_GB = "SGLANG_HICACHE_ARENA_STAGING_GB"
PLACEHOLDERS = 1 << 22


def _psz(pool) -> int:
    """x59: tokens per arena slot. Set by ``ArenaMHAHostPool.bind`` from the
    pool's page size; 1 is the 27B form, the mamba arena pool (which borrows
    the slot bookkeeping below and carries a page size of its own meaning)
    and every hermetic fixture that never binds a paged pool. A module
    function, not a method: the fixtures call the pool's methods on bare
    namespaces, and the mamba pool borrows them as plain functions."""
    return int(getattr(pool, "_arena_page_tokens", 1) or 1)


def _arena_mask(pool, hi: torch.Tensor) -> torch.Tensor:
    S = int(pool.staging_rows)
    return (hi >= S) & (hi < S + _atok(pool))


def _slots_of_rows(pool, rows: torch.Tensor) -> torch.Tensor:
    """Arena ROWS (host id - S) -> the slots they belong to, one per page, in
    order, duplicates of one page folded (a page's ids are consecutive)."""
    P = _psz(pool)
    if P == 1:
        return rows.to(torch.int64)
    return torch.unique_consecutive(rows.to(torch.int64) // P)


def _atok(pool) -> int:
    """x59: the arena's id range in TOKENS (A * P); a pool bound by an older
    fixture carries only the slot count."""
    t = getattr(pool, "arena_tokens", None)
    return int(t) if t else int(getattr(pool, "arena_slots", 0)) * _psz(pool)


def _page_slots(pool, rows: torch.Tensor) -> torch.Tensor:
    """x59: token rows of whole pages -> one slot per page (the P consecutive
    ids of each page, in order; anything else is a caller handing tokens of
    a page out of order and is refused). P == 1: the rows are the slots."""
    P = _psz(pool)
    if P == 1:
        return rows
    n = int(rows.numel())
    if n % P:
        raise RuntimeError(f"#1424 paged arena load: {n} token rows are not whole pages of {P}")
    pages = rows.view(-1, P)
    first = pages[:, 0]
    lane = torch.arange(P, device=rows.device, dtype=rows.dtype)[None, :]
    if bool((first % P).any()) or bool((pages != first[:, None] + lane).any()):
        raise RuntimeError("#1424 paged arena load: a page's token rows are not consecutive from its first id")
    return first // P
_CUDA_HOST_REGISTER_FLAGS = 3  # Portable | Mapped
_CUDA_ERROR_ALREADY_REGISTERED = 712


ARENA_PAGE_LOAD_ENV = "SGLANG_WEG2_ARENA_PAGE_LOAD"
ARENA_PAGE_LOAD_BLOCK_ENV = "SGLANG_WEG2_ARENA_PAGE_LOAD_BLOCK"


_PAGE_LOAD_N = 0
_ARENA_WRITE_N = 0


class _NoEvent:
    """Stand-in for torch.cuda.Event on a CPU-only desk (tests)."""
    def record(self, *a, **k):
        pass

    def synchronize(self):
        pass


def _arena_page_load_on() -> bool:
    """Posten 2 (18.09.): whole-page loadback, standard on; =0 for an A/B."""
    return str(os.environ.get(ARENA_PAGE_LOAD_ENV, "1")).strip().lower() not in ("0", "false", "no", "off")


ARENA_PAGE_LOAD_MODES = ("cpu", "kernel", "dma")


def _arena_page_load_mode(page_bytes: int = 0) -> str:
    """"kernel" = the MLA one-buffer JIT gather with element_dim = page bytes;
    "cpu" = the pinned-stage index_select; "dma" = one H2D copy per run of
    consecutive slots straight out of the registered arena (no CPU gather, no
    JIT, no pinned stage).

    fnFL2x66 (23.09.): the JIT module is instantiated PER ELEMENT SIZE. The
    27B's 32-KiB page is a warm build; a Next Flash page (786432 B) is a new
    instantiation, and the first load after the wake built it with ninja in
    the scheduler thread (py-spy: transfer_hicache_one_layer_mla -> load_jit
    -> build_ninja) for > 150 s while the expert workers waited in the
    all_reduce and died at the BAR1 cycle deadline.

    H12 (fnFL2x104, 90k needle): the cpu stage that replaced it is a memcpy in
    the scheduler thread -- 1528 pages = 1.2 GB in 347 ms on D TP0 (3.5 GB/s,
    ``components_ms=[kv=347``), 78 % of the first pass's init_new 448 ms, while
    the expert workers wait in the extend's first all_reduce. A Next Flash page
    is 768 KiB and the pages of one request lie in consecutive slots
    (x104: slot=[72,1599] for 1528 pages), so the DMA engine reads them from
    the registered arena at link rate without any gather. Unset, a page above
    32 KiB takes "dma"; the 27B's 32-KiB page keeps "kernel"; an explicit env
    value wins for both sizes."""
    m = str(os.environ.get("SGLANG_WEG2_ARENA_PAGE_LOAD_MODE", "")).strip().lower()
    if m in ARENA_PAGE_LOAD_MODES:
        return m
    return "dma" if int(page_bytes or 0) > 32768 else "kernel"


def page_dma_runs(slots: Sequence[int], piece_pages: int) -> list:
    """Runs of consecutive arena slots, as ``(stage_row, first_slot, count)``.

    One run is one H2D copy of ``count`` whole pages from the arena into the
    device stage rows ``[stage_row, stage_row + count)``. A run never crosses a
    multiple of ``piece_pages``: the arena is registered with cudaHostRegister
    in pieces of that many pages (``bind``'s pre-pin; 1 for the lazy per-slot
    pin), and a single copy must stay inside ONE registration to be a DMA from
    page-locked memory. The stage keeps the caller's page order, so the
    per-layer scatter that follows is the same one the cpu stage feeds."""
    piece = max(1, int(piece_pages))
    runs = []
    row = 0
    n = len(slots)
    while row < n:
        first = int(slots[row])
        count = 1
        while (
            row + count < n
            and int(slots[row + count]) == first + count
            and (first + count) % piece != 0
        ):
            count += 1
        runs.append((row, first, count))
        row += count
    return runs


def _ensure_page_stages(pool, B: int, pb: int, dev) -> None:
    """The cpu mode's two alternating pinned host stages (2 x 256 MiB);
    "dma" and "kernel" never allocate them. A module function like ``_psz``:
    the fixtures drive the page load on bare namespaces."""
    if pool._page_stages is None or pool._page_stages[0].shape[0] < B:
        _pin = bool(dev.type == "cuda")
        pool._page_stages = (
            torch.empty((B, pb), dtype=torch.uint8, pin_memory=_pin),
            torch.empty((B, pb), dtype=torch.uint8, pin_memory=_pin),
        )
        pool._page_events = ((torch.cuda.Event(), torch.cuda.Event()) if _pin
                             else (_NoEvent(), _NoEvent()))


def _page_block_dma(pool, dev_stage, block_slots) -> int:
    """H12: one H2D copy per run of consecutive slots, straight from the
    registered arena into the device stage rows (page order kept, see
    ``page_dma_runs``). Returns the number of copies issued. The arena bytes
    stay valid until the copy lands for the same reason they do in "kernel"
    mode: the slots are referenced by the load that reads them."""
    runs = page_dma_runs(block_slots.tolist(), pool._dma_piece_pages)
    view = pool._page_view
    for row, first, count in runs:
        dev_stage[row:row + count].copy_(view[first:first + count], non_blocking=True)
    return len(runs)


def _arena_page_load_timing() -> bool:
    return str(os.environ.get("SGLANG_WEG2_ARENA_PAGE_LOAD_TIMING", "0")).strip() not in ("", "0")


#: The pinned stage the default block was tuned for: 8192 pages of the 27B's
#: 32-KiB page (xsn325: 10.4-12.5 GB/s vs 5.9-11 at 2048) = 256 MiB per stage.
_PAGE_LOAD_STAGE_BYTES = 8192 * 32768


def _arena_page_load_block(page_bytes: int = 0) -> int:
    """Pages per pinned stage of the all-layer page load.

    fnFL2x65 (23.09.): the default was a PAGE COUNT tuned for the 27B's 32-KiB
    page. A Next Flash page is 786432 B (12 layers x 64 tokens x 1 KiB), so the
    same count pinned 2 x 6 GiB (+ the device stage) on D TP0 at the wake --
    shmem +15 GiB in 4 s, the kernel OOM-killed the rank (x63/x64/x65, py-spy in
    ``_load_pages_all_layers``). Unset, the block is 256 MiB worth of pages of
    THIS pool's page; an explicit env value stays a page count."""
    v = os.environ.get(ARENA_PAGE_LOAD_BLOCK_ENV, "").strip()
    if v:
        try:
            return max(64, int(v))
        except ValueError:
            pass
    if int(page_bytes or 0) > 32768:
        return max(64, _PAGE_LOAD_STAGE_BYTES // int(page_bytes))
    return 8192


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


ENV_ARENA_KV_PAGE_BYTES = "SGLANG_HICACHE_ARENA_KV_PAGE_BYTES"


def planned_arena_slots(kv_page_bytes: int) -> int:
    """The slot count the storage backend will give the KV arena
    (``HiCacheFile._arena_for``: ``max(1024, GIB * 2^30 // page_bytes)``),
    computed BEFORE the bind from the same environment -- so a sidecar pool
    that is addressed by the KV pool's ids (QSA index, paged draft) can be
    sized to the whole id space at construction. ``kv_page_bytes`` is the
    CANONICAL page (every attention layer of the model); a PP stage passes
    ``SGLANG_HICACHE_ARENA_KV_PAGE_BYTES`` from the launcher when its local
    layers are fewer than the model's, or over-estimates with its own."""
    try:
        gib = float(os.environ.get("SGLANG_HICACHE_ARENA_GIB", "8"))
    except ValueError:
        gib = 8.0
    env_pb = os.environ.get(ENV_ARENA_KV_PAGE_BYTES)
    if env_pb:
        try:
            kv_page_bytes = int(env_pb)
        except ValueError:
            pass
    return max(1024, int(gib * (1 << 30)) // max(1, int(kv_page_bytes)))


def planned_id_space_tokens(staging_tokens: int, page_size: int, kv_page_bytes: int) -> int:
    """Tokens a sidecar pool must be able to address: the staging ring plus
    P tokens per planned arena slot (placeholders never carry bytes)."""
    return int(staging_tokens) + planned_arena_slots(kv_page_bytes) * int(page_size)


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

    def _P(self) -> int:
        """Tokens per arena slot (the pool's page size); 1 = the 27B form.
        Read defensively: the hermetic fixtures build the pool without
        ``__init__`` and never set a page size."""
        return int(getattr(self, "page_size", 1) or 1)

    def _arena_tok(self) -> int:
        """Arena ids in TOKENS (A * P); falls back to the slot count for a
        pool bound by an older fixture."""
        t = getattr(self, "arena_tokens", None)
        return int(t) if t else int(self.arena_slots) * _psz(self)

    def _carrier_capacity_bid(self, size: int) -> int:
        """fnFL2x62: this pool hands out ids up to staging + A*P, not up to its
        ``size`` (the staging ring), so a no-KV peer must accept the whole id
        space. Planned from the environment exactly as ``bind()`` will size
        the arena (``planned_arena_slots``), because this runs in ``__init__``
        before the arena is bound."""
        P = int(getattr(self, "page_size", 1) or 1)
        spt = int(getattr(self, "size_per_token", 0) or 0)
        if spt <= 0:
            return NO_KV_RANK_TOKENS
        return int(size) + int(planned_arena_slots(spt * P)) * P

    def _arena_init_fields(self) -> None:
        self.staging_rows = int(self.size)
        self.arena = None
        self.arena_slots = 0
        self.arena_tokens = 0  # x59: A * page_size once bound
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
        self._pending_mask = None   # xsn355: bool[A], mirrors _pending's keys (vectorised membership)
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
        # x59 (23.09., Task #107): ONE arena slot per PAGE of ``page_size``
        # tokens. Next Flash pages by 64 (QSA groups, GDN anchors); the
        # token-paged 27B form is the special case P == 1 and stays
        # byte-identical. A page's flat layout is the one
        # ``MHATokenToKVPoolHost.get_data_page`` produces for ``layer_first``:
        # K/V-major, layer-major inside each half, TOKEN-MINOR -- so one
        # layer's K block of a page is ``P * cell`` contiguous bytes.
        P = int(getattr(self, "page_size", 1) or 1)  # the pool's own page size (bind sets _arena_page_tokens)
        if P < 1:
            raise ValueError(f"#1424 the arena host pool needs a positive page size, got {P}")
        if role == "draft" and P != 1:
            raise ValueError(
                "#1424 the draft role of the arena host pool is token-paged only; a paged "
                "draft page rides the KV page as a per-key sidecar (kv_cache_builder)"
            )
        ext = [(int(o), int(l)) for o, l in window.extents]
        if len(ext) == 1 and ext[0][0] == 0 and ext[0][1] == int(window.total_bytes):
            # the whole page (D's DCP ranks hold every layer and head): the
            # canonical page is K-major, [K all slots][V all slots]
            half = int(window.total_bytes) // 2
            ext = [(0, half), (half, half)]
        L = int(self.layer_num)
        e = int(self.dtype.itemsize)
        cell = int(self.head_num) * int(self.head_dim) * e
        block = P * cell  # one layer's K (or V) bytes of one page
        if len(ext) == 2:
            (k_off, k_len), (v_off, v_len) = ext
            if k_len != v_len or k_len != L * block:
                raise ValueError(
                    f"#1424 window extents {ext} do not match {L} layers x {P} tokens x {cell} B"
                )
            k_offs = [k_off + l * block for l in range(L)]
            v_offs = [v_off + l * block for l in range(L)]
        elif len(ext) == 2 * L and P == 1:
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
        if P == 1:
            self.arena_k_refs = [
                typed.as_strided((A, H, D), (tok, D, 1), storage_offset=base_off + k_offs[l] // e)
                for l in range(L)
            ]
            self.arena_v_refs = [
                typed.as_strided((A, H, D), (tok, D, 1), storage_offset=base_off + v_offs[l] // e)
                for l in range(L)
            ]
        else:
            # (slot, token-in-page, H, D): a page's tokens are contiguous cells
            # inside one layer block, the slots are page_bytes apart.
            self.arena_k_refs = [
                typed.as_strided((A, P, H, D), (tok, H * D, D, 1), storage_offset=base_off + k_offs[l] // e)
                for l in range(L)
            ]
            self.arena_v_refs = [
                typed.as_strided((A, P, H, D), (tok, H * D, D, 1), storage_offset=base_off + v_offs[l] // e)
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
        # H12: pages per cudaHostRegister piece -- the "dma" page load keeps
        # each copy inside one registration. 1 (a page) is always inside one:
        # the lazy pin registers whole slots, and a piece of the pre-pin below
        # is a whole number of pages. Raised to the pre-pin's piece only when
        # THIS bind registered the region, so the piece size is known.
        self._dma_piece_pages = 1
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
            self._dma_piece_pages = step // page_bytes
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
        self._pending_mask = torch.zeros(int(A), dtype=torch.bool)
        self._pending_gen = torch.zeros(int(A), dtype=torch.int64)     # xsn359: generation per pending slot
        self._pending_fresh = torch.zeros(int(A), dtype=torch.bool)   # xsn359: fresh claim (free on abort)
        # the id space counts TOKENS: P ids per slot, [S, S + A*P)
        self._arena_page_tokens = P
        self.arena_tokens = A * P
        self.id_space = self.staging_rows + A * P + PLACEHOLDERS
        self.prefetch_capacity_tokens = A * P
        if role == "draft":
            self.row_slot = {}
            self._zero_k = torch.zeros((1, H, D), dtype=self.dtype, pin_memory=self.pin_memory)
            self._zero_v = torch.zeros((1, H, D), dtype=self.dtype, pin_memory=self.pin_memory)
        logger.info("#1424 arena host pool bound role=%s staging=%d slots=%d layers=%d k_off=%d v_off=%d",
                    role, self.staging_rows, A, L, k_off, v_off)

    # -- id space ------------------------------------------------------------
    # x59 (Task #107): host ids are TOKENS. [0, S) staging rows, [S, S + A*P)
    # arena ids -- id S + slot*P + t is token t of arena slot `slot` -- and
    # placeholders above. P == 1 is the token-paged 27B form unchanged.
    def is_arena_id(self, i: int) -> bool:
        return self.staging_rows <= int(i) < self.staging_rows + _atok(self)

    def is_placeholder(self, i: int) -> bool:
        return int(i) >= self.staging_rows + _atok(self)

    def arena_ids(self, slots) -> torch.Tensor:
        """The host ids of whole slots: P consecutive ids per slot, in slot order."""
        s = torch.as_tensor(list(slots) if not torch.is_tensor(slots) else slots, dtype=torch.int64)
        P = _psz(self)
        if P == 1:
            return s + self.staging_rows
        return (s[:, None] * P + torch.arange(P, dtype=torch.int64)[None, :]).reshape(-1) + self.staging_rows

    @_pass_timed("_1474_alloc_ms")  # #1474
    def alloc_read(self, n: int) -> torch.Tensor:
        """Placeholders for a prefetch registration; resolved in place later."""
        base = self.staging_rows + _atok(self)
        start = self._read_ph_next
        self._read_ph_next = (start + n) % PLACEHOLDERS
        return (torch.arange(start, start + n, dtype=torch.int64) % PLACEHOLDERS) + base

    def resolve_rows(self, host_indices: torch.Tensor, slots: Sequence[int]) -> None:
        """KV role: write arena ids over the placeholders, in place -- P ids
        per slot, so a resolved run covers len(slots) * P tokens."""
        self.pin_slots(slots)
        vals = self.arena_ids(slots).to(dtype=host_indices.dtype)
        n = int(vals.numel())
        if n > int(host_indices.numel()):
            raise ValueError(
                f"#1424 resolve_rows: {len(slots)} slots x {_psz(self)} tokens "
                f"= {n} ids do not fit the {int(host_indices.numel())}-token registration"
            )
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
        S = self.staging_rows
        is_arena = _arena_mask(self,host_indices)
        if not bool(is_arena.any()):
            return super().load_to_device_per_layer(device_pool, host_indices, device_indices, layer_id, io_backend)
        # 19.09. (xsn398, Task #3): the page load's "already loaded at layer 0"
        # key is taken from the CALLER's index objects, which are the same
        # objects for every layer of one start_loading. Keying on the
        # per-layer `host_indices - S` temporaries relied on CPython reusing
        # the freed tensor's id -- when it did not (TP0, xsn398), layers
        # 1..47 also ran the per-layer kernel: a second full copy (+321 ms).
        self._page_key_hint = (id(host_indices), id(device_indices),
                               int(host_indices.numel()), int(device_indices.numel()))
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

    def _page_slots_of_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """x59: token rows of whole pages -> one slot per page (rows must be
        the P consecutive ids of each page, in order; anything else is a
        caller handing tokens of a page out of order and is refused)."""
        P = _psz(self)
        if P == 1:
            return rows
        n = int(rows.numel())
        if n % P:
            raise RuntimeError(
                f"#1424 paged arena load: {n} token rows are not whole pages of {P}")
        pages = rows.view(-1, P)
        first = pages[:, 0]
        if bool((first % P).any()) or bool((pages != first[:, None] + torch.arange(P, device=rows.device, dtype=rows.dtype)[None, :]).any()):
            raise RuntimeError("#1424 paged arena load: a page's token rows are not consecutive from its first id")
        return first // P

    def _load_arena(self, device_pool, rows, device_indices, layer_id) -> None:
        P = _psz(self)
        if _arena_page_load_on() and getattr(self, "_page_view", None) is not None:
            key = getattr(self, "_page_key_hint", None) or (
                id(rows), id(device_indices), int(rows.numel()), int(device_indices.numel()))
            if layer_id != 0 and self._page_loaded_key == key:
                return  # every layer came with the page load at layer 0
            if layer_id == 0:
                slots = _page_slots(self,rows)
                _tm0 = time.perf_counter()
                if self.row_slot is not None:
                    rl = rows.tolist()
                    slots = torch.tensor([self.row_slot.get(int(r), -1) for r in rl], dtype=rows.dtype, device=rows.device)
                self._last_load_map_ms = (time.perf_counter() - _tm0) * 1000.0
                if slots.numel() and bool((slots >= 0).all()):
                    _tp0 = time.perf_counter()
                    _newly = self.pin_slots(slots)
                    self._last_load_pin_ms = (time.perf_counter() - _tp0) * 1000.0
                    self._last_load_pinned_n = int(_newly or 0)
                    self._arena_load_guard(device_pool, slots, device_indices, 0, nrows=int(rows.numel()), nmiss=0)
                    self._load_pages_all_layers(device_pool, slots, device_indices)
                    self._page_loaded_key = key
                    return
                # misses (row_slot without a slot) keep the per-layer path below
            self._page_loaded_key = None
        if P != 1:
            # per-layer path of a paged pool: the per-token kernel cannot
            # address (slot, token) through one stride; a torch gather can.
            slots = rows // P
            self.pin_slots(torch.unique(slots))
            self._arena_load_guard(device_pool, slots, device_indices, layer_id, nrows=int(rows.numel()), nmiss=0)
            self._transfer_paged(device_pool, rows, device_indices, layer_id)
            return
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
        """Fetch whole pages (all layers of a token) in blocks into a device
        stage, then scatter each layer on the device. How a block reaches the
        stage is the mode (``_arena_page_load_mode``): "dma" copies runs of
        consecutive slots straight out of the registered arena, "kernel" lets
        the GPU gather the pages through the mapped arena, "cpu" gathers into
        two alternating pinned stages so the CPU gather of block i+1 overlaps
        the DMA of block i."""
        n = int(slots.numel())
        if n == 0:
            return
        dev = device_pool.k_buffer[0].device
        pb = self._page_bytes
        B = _arena_page_load_block(pb)  # x65: a 256-MiB stage, not 8192 pages of any size
        H, D = int(self.head_num), int(self.head_dim)
        e = self.dtype.itemsize
        cell = H * D * e
        cpu_slots = slots.to("cpu", dtype=torch.int64)
        dev_stage = torch.empty((min(B, n), pb), dtype=torch.uint8, device=dev)
        # xsn338: the index tensors cross to the device ONCE from pinned
        # memory; a pageable `.to(dev)` per block was a host sync per block,
        # so the scheduler thread waited out the whole DMA (~120-150 ms for
        # 1.8 GB). The per-block slices below are device views.
        _async_idx = bool(dev.type == "cuda")
        if _async_idx:
            _slots_dev = cpu_slots.pin_memory().to(dev, non_blocking=True)
            _dst_all = device_indices.to("cpu", dtype=torch.int64).pin_memory().to(dev, non_blocking=True) \
                if device_indices.device.type != "cuda" else device_indices.to(dtype=torch.int64)
        L = len(self._k_offs_b)
        # xsn305: the CPU gather of three ranks at once (3 x 3 GB memcpy) did
        # not beat the per-layer kernel (0,5-1,95 GB/s). Mode "kernel" lets
        # the GPU gather WHOLE PAGES straight from the mapped arena (the MLA
        # one-buffer kernel with element_dim = page bytes); "cpu" is the
        # pinned-stage form; the first kernel failure falls back to cpu.
        mode = getattr(self, "_page_mode", None) or _arena_page_load_mode(pb)  # x66: no JIT build at the wake
        if mode == "kernel" and dev.type != "cuda":
            mode = "cpu"
        if mode == "cpu":
            _ensure_page_stages(self, B, pb, dev)
        _timing = _arena_page_load_timing()
        _t0 = time.perf_counter() if _timing else 0.0
        _gather_ms = 0.0
        _runs = 0
        # 19.09. (Task #3, xsn394): under the timing env every block records
        # three events (before gather, after gather, after the per-layer
        # scatter) so the device time splits into GATHER (host->stage over
        # PCIe) and SCATTER (stage->pools), next to the CPU launch time.
        _ev = [] if (_timing and dev.type == "cuda") else None
        for bi, start in enumerate(range(0, n, B)):
            b = min(B, n - start)
            k = bi % 2
            if _ev is not None:
                _e0 = torch.cuda.Event(enable_timing=True); _e0.record()
            if mode == "dma":
                try:
                    _runs += _page_block_dma(self, dev_stage, cpu_slots[start:start + b])
                except Exception as exc:  # noqa: BLE001 -- one named fallback, then cpu
                    logger.warning("WEG2-ARENA-PAGE-LOAD dma mode failed (%s: %s); cpu mode from now on",
                                   type(exc).__name__, exc)
                    self._page_mode = mode = "cpu"
                    _ensure_page_stages(self, B, pb, dev)
            if mode == "kernel":
                try:
                    from sglang.jit_kernel.hicache import transfer_hicache_one_layer_mla
                    src_idx = _slots_dev[start:start + b] if _async_idx else slots[start:start + b].to(device=dev, dtype=torch.int64)
                    stage_idx = torch.arange(b, device=dev, dtype=torch.int64)
                    transfer_hicache_one_layer_mla(
                        cache_dst=dev_stage[:b], indices_dst=stage_idx,
                        cache_src=self._page_view, indices_src=src_idx,
                        element_dim=pb, block_quota=_arena_load_block_quota())
                except Exception as exc:  # noqa: BLE001 -- one named fallback, then cpu
                    logger.warning("WEG2-ARENA-PAGE-LOAD kernel mode failed (%s: %s); cpu mode from now on",
                                   type(exc).__name__, exc)
                    self._page_mode = mode = "cpu"
                    _ensure_page_stages(self, B, pb, dev)
            if mode == "cpu":
                self._page_events[k].synchronize()  # the DMA that last read this stage is done
                host_stage = self._page_stages[k]
                _g0 = time.perf_counter()
                torch.index_select(self._page_view, 0, cpu_slots[start:start + b], out=host_stage[:b])
                _gather_ms += (time.perf_counter() - _g0) * 1000.0
                dev_stage[:b].copy_(host_stage[:b], non_blocking=True)
                self._page_events[k].record()
            if _ev is not None:
                _e1 = torch.cuda.Event(enable_timing=True); _e1.record()
            # x59: P device tokens per page, in page order; one layer's K block
            # of a page is P contiguous cells, so the stage slice scatters as
            # (b * P) rows.
            P = _psz(self)
            if _async_idx:
                dst = _dst_all[start * P:(start + b) * P]
            else:
                dst = device_indices[start * P:(start + b) * P].to(device=dev, dtype=torch.int64)
            for l in range(L):
                ko, vo = self._k_offs_b[l], self._v_offs_b[l]
                device_pool.k_buffer[l].index_copy_(
                    0, dst, dev_stage[:b, ko:ko + P * cell].reshape(-1).view(self.dtype).view(b * P, H, D))
                device_pool.v_buffer[l].index_copy_(
                    0, dst, dev_stage[:b, vo:vo + P * cell].reshape(-1).view(self.dtype).view(b * P, H, D))
            if _ev is not None:
                _e2 = torch.cuda.Event(enable_timing=True); _e2.record()
                _ev.append((_e0, _e1, _e2))
        _cpu_ms = (time.perf_counter() - _t0) * 1000.0 if _timing else 0.0
        global _PAGE_LOAD_N
        _PAGE_LOAD_N += 1
        n_log = _PAGE_LOAD_N
        _wall = ""
        if _timing and dev.type == "cuda":
            torch.cuda.current_stream(dev).synchronize()
            _ms = (time.perf_counter() - _t0) * 1000.0
            _gd = sum(e0.elapsed_time(e1) for e0, e1, _e2 in _ev) if _ev else 0.0
            _sd = sum(e1.elapsed_time(e2) for _e0, e1, e2 in _ev) if _ev else 0.0
            _wall = (f" wall_ms={_ms:.0f} GB/s={(n * pb) / max(_ms, 1e-3) / 1e6:.2f} cpu_launch_ms={_cpu_ms:.0f}"
                     f" dev_gather_ms={_gd:.0f} dev_scatter_ms={_sd:.0f}"
                     f" gather_GB/s={(n * pb) / max(_gd, 1e-3) / 1e6:.2f} cpu_gather_ms={_gather_ms:.0f}"
                     f" map_ms={getattr(self, '_last_load_map_ms', 0.0):.0f} pin_ms={getattr(self, '_last_load_pin_ms', 0.0):.0f}"
                     f" pinned_new={getattr(self, '_last_load_pinned_n', 0)}")
        if mode == "dma":
            _wall = f" runs={_runs} piece={self._dma_piece_pages}" + _wall
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

    # -- x59 (Task #107): the paged pool's torch paths ---------------------------
    def _transfer_paged(self, device_pool, rows, device_indices, layer_id) -> None:
        """Per-layer load of a PAGED pool (fallback beside the whole-page
        load): token rows -> (slot, token) into the strided arena views, one
        gather per K and V, then an indexed copy onto the card."""
        P = _psz(self)
        rows = rows.to("cpu", dtype=torch.int64)
        slot, tok = rows // P, rows % P
        k_src = self.arena_k_refs[layer_id][slot, tok]
        v_src = self.arena_v_refs[layer_id][slot, tok]
        dst_k = device_pool.k_buffer[layer_id]
        dev = dst_k.device
        didx = device_indices.to(device=dev, dtype=torch.int64)
        dst_k.index_copy_(0, didx, k_src.to(device=dev, non_blocking=False))
        device_pool.v_buffer[layer_id].index_copy_(0, didx, v_src.to(device=dev, non_blocking=False))

    def _backup_paged_copy(self, device_pool, slots: torch.Tensor, device_indices: torch.Tensor) -> None:
        """Direct write of a PAGED pool without the pointer/stride kernel:
        this rank's K and V blocks of every page go card -> host stage ->
        the slots' layer blocks (one contiguous run per layer per page)."""
        P = _psz(self)
        b = int(slots.numel())
        if b == 0:
            return
        self.pin_slots(slots)
        L = len(self._k_offs_b)
        cell = int(self.element_dim) * int(self.dtype.itemsize)
        block = P * cell
        slots_cpu = slots.to("cpu", dtype=torch.int64)
        didx = device_indices.to(device=device_pool.k_buffer[0].device, dtype=torch.int64)
        for l in range(L):
            kv = device_pool.k_buffer[l]
            vv = device_pool.v_buffer[l]
            k_rows = kv.view(torch.uint8).reshape(kv.shape[0], -1)[didx].reshape(b, block).to("cpu")
            v_rows = vv.view(torch.uint8).reshape(vv.shape[0], -1)[didx].reshape(b, block).to("cpu")
            ko, vo = self._k_offs_b[l], self._v_offs_b[l]
            self._page_view[slots_cpu, ko:ko + block] = k_rows
            self._page_view[slots_cpu, vo:vo + block] = v_rows
        global _ARENA_WRITE_N
        if _ARENA_WRITE_N <= 8 or _ARENA_WRITE_N % 256 == 0:
            logger.info("WEG2-ARENA-WRITE n=%d pages=%d bytes=%d mode=paged-copy block=%dB layers=%d",
                        _ARENA_WRITE_N, b, b * 2 * L * block, block, L)

    # -- #1427 Stufe 4: direct writes, card -> arena slot -------------------------
    def _claim_np(self, stems, totals):
        """xsn359: the KV/draft pool's claim on numpy arrays -- one C call, the
        pending state as mask/gen/fresh tensors, no per-slot Python."""
        import numpy as np
        arena = self.arena
        slots, st, gens = arena.claim_slots_np(stems, totals)
        _cn = getattr(ArenaMHAHostPool, "_1427_claim_n", 0) + 1
        ArenaMHAHostPool._1427_claim_n = _cn
        if _cn <= 12 or _cn % 512 == 0:
            logger.info("#1427 ARENA-CLAIM n=%d stems=%d first=%s last=%s statuses=%s arena=%s",
                        _cn, len(stems), stems[0] if stems else "-", stems[-1] if stems else "-",
                        sorted(set(st.tolist())), getattr(arena, "path", "?"))
        bad = (st == 3) | (st == 4)
        if bool(bad.any()):
            ev = getattr(self._backend, "_arena_evict_to_disk", None)
            if callable(ev) and bool((st == 4).any()):
                try:
                    ev(arena, max(256, len(stems)))
                except Exception as exc:  # noqa: BLE001 - loud, the claim below decides
                    logger.warning("#1427 arena evict-to-disk failed: %r", exc)
                redo = np.nonzero(st == 4)[0]
                s2, st2, g2 = arena.claim_slots_np([stems[int(i)] for i in redo], [self._page_bytes] * int(redo.size))
                slots[redo] = s2; st[redo] = st2; gens[redo] = g2
                bad = (st == 3) | (st == 4)
            if bool(bad.any()):
                fresh = slots[st == 0]
                if fresh.size:
                    arena.free_slots(fresh.tolist())
                k = getattr(ArenaMHAHostPool, "_1427_full_n", 0) + 1
                ArenaMHAHostPool._1427_full_n = k
                if k <= 8 or k % 256 == 0:
                    logger.warning("#1427 ARENA-CLAIM REFUSED n=%d pages=%d statuses=%s (4 = no free slot)",
                                   k, len(stems), sorted(set(st.tolist())))
                return None
        pend = st != 2
        if bool(pend.any()):
            idx = torch.from_numpy(slots[pend])
            self._pending_mask[idx] = True
            self._pending_gen[idx] = torch.from_numpy(gens[pend])
            self._pending_fresh[idx] = torch.from_numpy(st[pend] == 0)
        complete = slots[st == 2]
        if complete.size:
            arena.ref_slots(complete.tolist(), +1)        # complete already: reader reference only
        return slots.tolist()

    def _pend_take(self, slots_t: torch.Tensor):
        """xsn359: (pending-mask over slots_t, gens, fresh) and CLEAR them -- the
        vectorised form of one _pend_pop per slot."""
        m = self._pending_mask[slots_t]
        sel = slots_t[m]
        gens = self._pending_gen[sel].clone()
        fresh = self._pending_fresh[sel].clone()
        self._pending_mask[sel] = False
        return m, sel, gens, fresh

    def _pend_mark(self, slots, value: bool) -> None:
        m = self._pending_mask
        if m is not None and slots:
            m[torch.as_tensor(list(slots), dtype=torch.int64)] = value

    def _pend_has(self, slots):
        """pending? per slot (list of bool), mask or dict."""
        if self._pending_mask is not None:
            t = torch.as_tensor(list(slots), dtype=torch.int64)
            return self._pending_mask[t].tolist() if t.numel() else []
        return [s in self._pending for s in slots]

    def _pend_pop(self, s):
        p = self._pending.pop(s, None)
        m = self._pending_mask
        if m is not None and p is not None:
            m[int(s)] = False
        return p

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
        if self._pending_mask is not None:
            return self._claim_np(stems, totals)
        got = arena.claim_slots(stems, totals)
        # xsn327: D's dormant re-reads never find P's pages -- name what P claims
        # (full stem incl. suffix) so the reader's stem can be compared by eye.
        _cn = getattr(ArenaMHAHostPool, "_1427_claim_n", 0) + 1
        ArenaMHAHostPool._1427_claim_n = _cn
        if _cn <= 12 or _cn % 512 == 0:
            logger.info("#1427 ARENA-CLAIM n=%d stems=%d first=%s last=%s statuses=%s arena=%s",
                        _cn, len(stems), stems[0] if stems else "-", stems[-1] if stems else "-",
                        sorted({st for _, st, _ in got}), getattr(arena, "path", "?"))
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
        # xsn352 (py-spy PP0): the per-slot loop was ~27 ms per 4096-page node
        # in the scheduler thread -- batched: one dict update, one ref call.
        self._pending.update((slot, (gen, st == 0)) for slot, st, gen in got if st != 2)
        self._pend_mark([slot for slot, st, _ in got if st != 2], True)
        complete = [slot for slot, st, _ in got if st == 2]
        if complete:
            arena.ref_slots(complete, +1)        # complete already: reader reference only
        return [slot for slot, _, _ in got]

    def alloc_write(self, hashes) -> Optional[torch.Tensor]:
        """KV role: one arena slot per page hash, claimed for a direct write.
        Returns the host ids (staging_rows + slot), or None."""
        if self.arena is None or self._backend is None or not hashes:
            return None
        slots = self._claim(self._stems(hashes))
        if slots is None:
            return None
        return self.arena_ids(slots)  # x59: P ids per page hash

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
        _isp = self._pend_has(slots)
        pending_idx = [i for i, p in enumerate(_isp) if p]
        complete_idx = [i for i, p in enumerate(_isp) if not p]
        if pending_idx:
            sl = torch.tensor([slots[i] for i in pending_idx], dtype=torch.int64)
            di = device_indices.reshape(-1)[torch.tensor(pending_idx, device=device_indices.device)]
            self._backup_arena(device_pool, sl, di)
            # the all-layer copy runs on the current stream into pinned arena
            # memory; the completion below publishes the bytes to every rank
            torch.cuda.current_stream().synchronize()
            if self._pending_mask is not None:
                # xsn360: the producer path on the tensors too (KeyError: 0 on
                # PP2 -- the dict is empty once the mask carries the state)
                _m, sel, gens_t, _f = self._pend_take(sl)
                st_np = self.arena.complete_slots_np(sel.numpy(), gens_t.numpy(), self._own_extents)
                lost = int((st_np == 3).sum())
            else:
                gens = [self._pending[s][0] for s in sl.tolist()]
                st = self.arena.complete_slots(sl.tolist(), gens, self._own_extents)
                lost = 0
                for s, r in zip(sl.tolist(), st):
                    self._pend_pop(s)
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
        from sglang.srt.weg2 import arena_write as _aw
        global _ARENA_WRITE_N
        _ARENA_WRITE_N += 1
        _n_log = _ARENA_WRITE_N
        b = int(slots.numel())
        cell = int(self.element_dim) * int(self.dtype.itemsize)
        P = _psz(self)
        block = P * cell  # x59: one layer's K (or V) block of a page
        if int(device_indices.numel()) != b * P:
            raise RuntimeError(
                f"#1424 arena write: {b} slots need {b * P} device tokens, got {int(device_indices.numel())}")
        _quota = _aw.write_block_quota()
        mode = getattr(self, "_write_mode", None) or _aw.write_mode()
        runs = _aw.contiguous_runs(self._k_offs, self._v_offs, block) if mode == "run" else None
        if runs is None and P != 1:
            # the per-token cell kernel cannot address (slot, token); the
            # paged pool writes through a host stage instead (torch copies)
            self._backup_paged_copy(device_pool, slots, device_indices)
            return
        if runs is not None and dev.type == "cuda":
            # xsn345 RUN MODE: gather this rank's layers into a run-shaped stage
            # on the card, then two contiguous runs per page (K, V) go to the
            # slot with the pointer/stride kernel -- L KiB per transfer instead
            # of 1 KiB, and the loader's block quota instead of the default 2.
            try:
                k_off, v_off, run = runs
                L = len(self._k_offs)
                didx = device_indices.to(device=dev, dtype=torch.int64)
                stage = torch.empty((b, 2 * run), dtype=torch.uint8, device=dev)
                for l in range(L):
                    kv = device_pool.k_buffer[l]
                    vv = device_pool.v_buffer[l]
                    stage[:, l * block:(l + 1) * block].copy_(
                        kv.view(torch.uint8).reshape(kv.shape[0], -1)[didx].reshape(b, block))
                    stage[:, run + l * block:run + (l + 1) * block].copy_(
                        vv.view(torch.uint8).reshape(vv.shape[0], -1)[didx].reshape(b, block))
                dst_ptrs, src_ptrs, src_stride = _aw.run_pointers(
                    self._data_base, k_off, v_off, run, stage.data_ptr())
                _nb = lambda t: t.pin_memory().to(dev, non_blocking=True)   # noqa: E731 -- no stream sync
                jit_transfer_hicache_all_layer_mla(
                    ptr_dst=_nb(torch.tensor(dst_ptrs, dtype=torch.uint64)),
                    indices_dst=_nb(slots.to(dtype=torch.int64)),
                    ptr_src=_nb(torch.tensor(src_ptrs, dtype=torch.uint64)),
                    indices_src=torch.arange(b, device=dev, dtype=torch.int64),
                    cache_src_stride_bytes=src_stride,
                    cache_dst_stride_bytes=self._page_bytes,
                    element_size=run,
                    block_quota=_quota,
                )
                self._write_stage_keep = stage   # alive until the write stream is done with it
                if _n_log <= 8 or _n_log % 256 == 0:
                    logger.info("WEG2-ARENA-WRITE n=%d pages=%d bytes=%d mode=run runs=2x%dB quota=%s",
                                _n_log, b, b * 2 * run, run, _quota)
                return
            except Exception as exc:  # noqa: BLE001 -- one named fallback, then the cell kernel
                logger.warning("WEG2-ARENA-WRITE run mode failed (%s: %s); cell mode from now on",
                               type(exc).__name__, exc)
                self._write_mode = mode = "cell"
                if P != 1:
                    self._backup_paged_copy(device_pool, slots, device_indices)
                    return
        if P != 1:
            self._backup_paged_copy(device_pool, slots, device_indices)
            return
        k_ptrs, v_ptrs = self._arena_ptrs(dev)
        _slots_dev = (slots.to(dtype=torch.int64).pin_memory().to(dev, non_blocking=True)
                      if dev.type == "cuda" else slots.to(device=dev, dtype=torch.int64))
        jit_transfer_hicache_all_layer(
            k_ptr_dst=k_ptrs,
            v_ptr_dst=v_ptrs,
            indices_dst=_slots_dev,
            k_ptr_src=device_pool.k_data_ptrs,
            v_ptr_src=device_pool.v_data_ptrs,
            indices_src=device_indices.to(device=dev, dtype=torch.int64),
            kv_cache_dst_stride_bytes=self._page_bytes,
            kv_cache_src_stride_bytes=self.token_stride_size,
            element_size=cell,
            block_quota=_quota,
        )
        if _n_log <= 8 or _n_log % 256 == 0:
            logger.info("WEG2-ARENA-WRITE n=%d pages=%d bytes=%d mode=cell cell=%dB quota=%s",
                        _n_log, b, b * 2 * len(self._k_offs) * cell, cell, _quota)

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
        S = self.staging_rows
        is_arena = _arena_mask(self,hi)
        if not bool(is_arena.any()):
            return super().backup_from_device_all_layer(device_pool, host_indices, device_indices, io_backend)
        sel = is_arena.nonzero(as_tuple=True)[0]
        if self.row_slot is None and self._pending_mask is not None:
            # xsn355 (py-spy PP0): the per-row Python pairs/todo lists were ~30 ms
            # per 4096-page node; KV role membership comes from the mask.
            P = _psz(self)
            rows_t = (hi[sel] - S).to(torch.int64)
            if P == 1:
                keep = self._pending_mask[rows_t]
                slots = rows_t[keep]
                sel = sel[keep]
            else:
                # x59: one slot per page; the P token ids of a page travel together
                page_slots = _page_slots(self,rows_t)
                keep_p = self._pending_mask[page_slots]
                slots = page_slots[keep_p]
                sel = sel.view(-1, P)[keep_p].reshape(-1)
            todo = bool(slots.numel())
        else:
            rows = (hi[sel] - S).tolist()
            pairs = [(self.row_slot.get(int(r), -1), i) for r, i in zip(rows, sel.tolist())]
            _isp = self._pend_has([s for s, _ in pairs])   # xsn359: mask or dict
            todo = [(s, i) for (s, i), p in zip(pairs, _isp) if s >= 0 and p]
            if todo:
                slots = torch.tensor([s for s, _ in todo], dtype=torch.int64)
                sel = torch.tensor([i for _, i in todo], dtype=torch.int64)
        if todo:
            if device_indices.device.type == "cuda":
                # xsn349: `device_indices.cpu()` here synchronised the scheduler
                # thread with the write stream (the previous chunk's copy still
                # in flight): CHUNK-PUBLISH 109 ms per 4096-page node. Select on
                # the device instead; the index crosses pinned + non-blocking.
                didx = device_indices.index_select(
                    0, sel.pin_memory().to(device_indices.device, non_blocking=True))
            else:
                didx = device_indices[sel]
            self._backup_arena(device_pool, slots, didx)
        rest = (~is_arena).nonzero(as_tuple=True)[0]
        if rest.numel():
            super().backup_from_device_all_layer(
                device_pool, host_indices[rest.to(host_indices.device)],
                device_indices[rest.to(device_indices.device)], io_backend)

    def backup_accepts_device_indices(self, host_indices, device_indices) -> str:
        """H2: an op whose host ids are ALL arena rows (host side, a direct
        write's claim) is written by the arena path above, which selects on
        the card (xsn349) and never reads the io backend. A staging row would
        take the base pool's backend branch, which wants host indices."""
        if self.arena is None:
            return "kv:arena_unbound"
        if host_indices.is_cuda:
            return "kv:host_ids_on_card"
        if not bool(_arena_mask(self, host_indices).all()):
            return "kv:staging_rows"
        return ""

    def backup_from_device_indices(self, device_pool, host_indices, device_indices):
        # every row is an arena row (accepted above): the backend argument is
        # never read on this path, the rest-branch that would read it is empty
        self.backup_from_device_all_layer(device_pool, host_indices, device_indices, "direct")

    def _slots_of(self, host_indices: torch.Tensor):
        hi = host_indices.cpu()
        rows = hi[_arena_mask(self,hi)] - self.staging_rows
        if self.row_slot is not None:
            return [self.row_slot.get(int(r), -1) for r in rows.tolist()]
        return _slots_of_rows(self,rows).tolist()

    def complete_write(self, host_indices: torch.Tensor) -> int:
        """The copy landed (ack): merge this rank's extents into the pages'
        coverage, take the node's reader reference. Returns how many pages
        this call completed for everyone."""
        if self.arena is None:
            return 0
        if self._pending_mask is not None:
            # xsn359: vectorised -- mask/gen/fresh tensors, one C call, no per-slot loop
            import numpy as np
            all_t = torch.as_tensor([s for s in self._slots_of(host_indices) if s >= 0], dtype=torch.int64)
            if all_t.numel() == 0:
                return 0
            _, sel, gens, _fresh = self._pend_take(all_t)
            if sel.numel() == 0:
                return 0
            st = self.arena.complete_slots_np(sel.numpy(), gens.numpy(), self._own_extents)
            lost = int((st == 3).sum())
            if lost:
                k = getattr(ArenaMHAHostPool, "_1427_lost_n", 0) + lost
                ArenaMHAHostPool._1427_lost_n = k
                if k <= 8 or k % 256 < lost:
                    logger.warning("#1427 ARENA-COMPLETE LOST slots=%s (recycled under the writer) n=%d",
                                   sel.numpy()[st == 3][:4].tolist(), k)
            keep = sel.numpy()[st != 3]
            if keep.size:
                self.arena.ref_slots(keep.tolist(), +1)
            return int((st == 1).sum())
        slots = [s for s in self._slots_of(host_indices) if s >= 0 and s in self._pending]
        if not slots:
            return 0
        gens = [self._pending[s][0] for s in slots]
        st = self.arena.complete_slots(slots, gens, self._own_extents)
        done = 0
        for s, r in zip(slots, st):
            self._pend_pop(s)
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
        if self._pending_mask is not None:
            all_t = torch.as_tensor([s for s in self._slots_of(host_indices) if s >= 0], dtype=torch.int64)
            if all_t.numel():
                m, sel, _g, fr = self._pend_take(all_t)
                refd = all_t[~m].tolist()
                fresh = sel[fr].tolist()
        else:
            for s in self._slots_of(host_indices):
                if s < 0:
                    continue
                p = self._pend_pop(s)
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
            # x59: the answer is TOKENS (P per slot), like the staging count
            return max(0, int(st["slots"]) - int(st["claimed"])) * _psz(self)
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
        S = self.staging_rows
        staging = idx[idx < S]
        arena_rows = idx[_arena_mask(self,idx)]
        freed = 0
        if staging.numel():
            freed += int(super().free(staging))
        if arena_rows.numel():
            rows = (arena_rows - S).tolist()
            if self.row_slot is not None:
                slots = [self.row_slot.pop(int(r), -1) for r in rows]
                slots = [s for s in slots if s >= 0]
            else:
                # x59: P token ids per slot -> one release per slot
                slots = _slots_of_rows(self,arena_rows - S).tolist()
            if self._pending_mask is not None:
                st_ = torch.as_tensor(slots, dtype=torch.int64)
                m, sel, _g, fr = self._pend_take(st_)
                fresh = sel[fr].tolist()
                if fresh:
                    self.arena.free_slots(fresh)
                slots = st_[~m].tolist()
            else:
                pend = [s for s in slots if s in self._pending]
                if pend:
                    fresh = [s for s in pend if self._pend_pop(s)[1]]
                    if fresh:
                        self.arena.free_slots(fresh)
                    slots = [s for s in slots if s not in pend]
            if slots:
                self.arena.ref_slots(slots, -1)
            freed += len(rows)
        return freed
