from __future__ import annotations

import ctypes
import logging
import os
import tempfile
from math import prod
from typing import TYPE_CHECKING, List, Optional, Sequence

import torch
import torch.utils.cpp_extension
from torch.cuda.memory import CUDAPluggableAllocator

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KvBufferDesc

logger = logging.getLogger(__name__)

_drv = None


def _driver():
    global _drv
    if _drv is None:
        from cuda.bindings import driver

        _drv = driver
    return _drv


def _check(result, label: str):
    drv = _driver()
    err = result[0] if isinstance(result, tuple) else result
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{label} failed: {err}")
    return result[1] if isinstance(result, tuple) and len(result) > 1 else None


def _mem_create_reclaiming(
    step: int,
    prop,
    label: str = "cuMemCreate",
    reclaim: Optional[callable] = None,
):
    """``cuMemCreate``, and on OUT_OF_MEMORY reclaim torch's cache and retry.

    THE COMPETITION THIS RESOLVES, measured on metal (#631, 2026-08-09
    12:47:45, rank 1 on a 3080). The arena allocates RAW DRIVER memory:
    ``cuMemCreate`` can only be served by physical pages the driver holds
    free. torch's caching allocator, by design, does NOT return freed
    blocks to the driver -- it keeps them reserved for its own reuse. A
    long prefill grows that reserve and never gives it back, so the free
    physical memory the arena can see SHRINKS over the life of the
    process even though nothing is actually using the pages.

    The phase flip is where that bites. ``_build_kv_backing_swap``
    releases the source pool's pages and immediately commits the
    destination's, and its comment asserted the restore "cannot fail for
    want of memory" because boot sized the budget for max(PP, TP). That
    reasoning holds only against a static allocator. Under the acceptance
    load it was false: the flip died with

        RuntimeError: cuMemCreate failed: CUDA_ERROR_OUT_OF_MEMORY

    inside restore_backing, taking the whole server down via SIGQUIT --
    a crash whose root is that torch was SITTING on the pages, not that
    the card lacked them.

    ``empty_cache`` returns exactly those unused-but-reserved blocks to
    the driver, which is the one reclaim that makes an arena commit
    possible. It costs nothing on the happy path (only reached after a
    real OOM) and costs a re-warm of torch's cache when it does fire --
    which is strictly better than losing the instance. If the retry still
    fails, the error is genuine and propagates unchanged.
    """
    drv = _driver()
    result = drv.cuMemCreate(step, prop, 0)
    err = result[0] if isinstance(result, tuple) else result
    if err == drv.CUresult.CUDA_ERROR_OUT_OF_MEMORY:
        logger.warning(
            "%s: %d bytes refused by the driver; releasing torch's cached "
            "blocks and retrying once. torch reserved %.2f GiB / allocated "
            "%.2f GiB before the reclaim.",
            label,
            int(step),
            torch.cuda.memory_reserved() / (1 << 30),
            torch.cuda.memory_allocated() / (1 << 30),
        )
        torch.cuda.empty_cache()
        if reclaim is not None:
            # #631: the arena's own parked handles are the other thing that
            # can be sitting on the pages this create needs. Dropping them
            # costs the zero-allocation property until the next release
            # re-parks -- strictly better than failing the flip.
            freed = reclaim()
            if freed:
                logger.warning(
                    "%s: released %.2f GiB of parked handles as well.",
                    label,
                    freed / (1 << 30),
                )
        result = drv.cuMemCreate(step, prop, 0)
    return _check(result, label)


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def query_granularity(device_id: int) -> int:
    """Minimum CUDA virtual-memory allocation granularity (bytes) for ``device_id``."""
    drv = _driver()
    prop = drv.CUmemAllocationProp()
    prop.type = drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = int(device_id)
    return int(
        _check(
            drv.cuMemGetAllocationGranularity(
                prop,
                drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            ),
            "cuMemGetAllocationGranularity",
        )
    )


# Bump allocator: hands back base+cursor, bounded by the RESERVED size (not the
# committed watermark) so upper-bound tensors can be allocated before physical
# commit. Allocations are granularity-aligned so each pointer can be committed at
# its own VA range (cuMemMap requires it; GB300 rejects partial-handle maps).
# Symbols are SUFFIXED per (process, arena instance) and each instance loads its
# own .so, so neither multiple arenas per process (hybrid-SWA: full + swa) nor
# co-located engine processes sharing the tempdir clobber each other.
def _stub_source(sfx: str) -> str:
    return f"""
#include <cstddef>
#include <cstdint>
#include <mutex>
extern "C" {{
static uintptr_t g_base = 0;
static size_t g_cursor = 0;
static size_t g_reserved = 0;
static size_t g_align = 512;
static std::mutex g_mu;
static size_t align_up(size_t v, size_t a){{ return (v + a - 1) / a * a; }}
void kvarena_set_base_{sfx}(uintptr_t b){{ std::lock_guard<std::mutex> lk(g_mu); g_base=b; g_cursor=0; }}
void kvarena_set_reserved_{sfx}(size_t r){{ std::lock_guard<std::mutex> lk(g_mu); g_reserved=r; }}
void kvarena_set_align_{sfx}(size_t a){{ std::lock_guard<std::mutex> lk(g_mu); if (a) g_align=a; }}
size_t kvarena_cursor_{sfx}(void){{ std::lock_guard<std::mutex> lk(g_mu); return g_cursor; }}
void* kvarena_malloc_{sfx}(size_t size, int device, void* stream){{
  std::lock_guard<std::mutex> lk(g_mu);
  size_t need = g_cursor + align_up(size, g_align);
  if (need > g_reserved) return 0;   // never exceed the reserved VA range
  void* p = reinterpret_cast<void*>(g_base + g_cursor);
  g_cursor = need;
  return p;
}}
void kvarena_free_{sfx}(void* ptr, size_t size, int device, void* stream){{}}
}}
"""


_DEFAULT_RESERVE_BYTES = 256 * (1024**3)  # 256 GiB virtual; free until committed


class KvVmmArena:
    """One device's CUDA virtual-memory reservation exposed as a ``torch.cuda.MemPool``."""

    # Per-instance suffix source -> isolated allocator symbols/state (see _stub_source).
    _instance_count = 0

    def __init__(
        self,
        device_id: int,
        reserve_bytes: int = _DEFAULT_RESERVE_BYTES,
        commit_chunk_bytes: Optional[int] = None,
        retain_handles: bool = False,
    ):
        self.device_id = int(device_id)
        # Unique per (process, arena instance): the stub .so lives in a host-shared
        # tempdir, so co-located engine processes must not build the same-named .so
        # (they race and one loads a half-relinked copy -> undefined symbol crash).
        self._sfx = f"{os.getpid()}_{KvVmmArena._instance_count}"
        KvVmmArena._instance_count += 1
        drv = _driver()
        with torch.cuda.device(self.device_id):
            _check(drv.cuInit(0), "cuInit")
            self._prop = drv.CUmemAllocationProp()
            self._prop.type = drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
            self._prop.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            self._prop.location.id = self.device_id
            self.granularity = query_granularity(self.device_id)
            self._access = drv.CUmemAccessDesc()
            self._access.location.type = (
                drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            )
            self._access.location.id = self.device_id
            self._access.flags = (
                drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            )

            self.reserved = self._align(reserve_bytes)
            # Align the base to granularity so base + (granularity-aligned cursor) is
            # always a valid cuMemMap address for per-buffer commit_range().
            self.base = int(
                _check(
                    drv.cuMemAddressReserve(self.reserved, self.granularity, 0, 0),
                    "cuMemAddressReserve",
                )
            )
            # commit_range bookkeeping, per bump-allocator offset: an ordered
            # list of physically-mapped extents [(rel_start, size, handle)]
            # plus the contiguous committed watermark. Extents are the unit of
            # release: cuMemUnmap operates on whole mappings only, so
            # decommit_range can only give back whole extents (#330).
            self._extents_by_offset = {}
            self._committed_by_offset = {}
            self._range_backed = 0
            self._closed = False
            # #330 dial: cap each physical handle at commit_chunk_bytes so a
            # later decommit_range can release the tail at chunk granularity.
            # None keeps the pre-#330 behavior: one handle per extension.
            self._chunk = (
                None if commit_chunk_bytes is None else self._align(commit_chunk_bytes)
            )
            # #631 ZERO-ALLOCATION SEAM. With retention on, decommit_range
            # UNMAPS an extent but does not cuMemRelease its physical
            # handle: the handle is parked here, keyed by its exact byte
            # size, and the next commit of that size re-maps it instead of
            # asking the driver for pages.
            #
            # WHY THE SIZE KEY IS EXACT AND WHY THAT NEEDS A COMMIT CHUNK.
            # A CUDA physical handle is created at a fixed size; there is
            # no API to split or merge one, so a parked handle can only
            # serve a request of the SAME size. With commit_chunk_bytes
            # unset the arena makes ONE monolithic handle per buffer, so
            # the PP span and the TP span produce differently-sized
            # handles and nothing is ever reusable -- retention alone
            # would park memory and still allocate. Chunked commits make
            # every handle the same granule, and only then are the pages
            # fungible between the two layouts. The two settings are one
            # feature; enabling retention without a chunk is a no-op that
            # costs memory, which is why _back_spans logs the pairing.
            self._retain_handles = bool(retain_handles)
            self._retained = {}
            self._retained_bytes = 0

        self._lib = self._build_stub()
        self._fn_set_base(ctypes.c_void_p(self.base))
        self._fn_set_reserved(ctypes.c_size_t(self.reserved))
        self._fn_set_align(ctypes.c_size_t(self.granularity))
        self._allocator = CUDAPluggableAllocator(
            self._so_path, f"kvarena_malloc_{self._sfx}", f"kvarena_free_{self._sfx}"
        ).allocator()
        # no_split so the caching allocator hands our bump pointers back verbatim.
        self.pool = torch.cuda.MemPool(self._allocator, no_split=True)
        logger.info(
            "KvVmmArena[%s] ready: device=%d reserved=%.1f GiB granularity=%d KiB",
            self._sfx,
            self.device_id,
            self.reserved / (1024**3),
            self.granularity // 1024,
        )

    def _align(self, v: int) -> int:
        return align_up(v, self.granularity)

    def _build_stub(self) -> ctypes.CDLL:
        # Per-arena build dir: load_inline writes every caller's source to the same
        # main.cpp inside build_directory, so any sharing (across co-located engine
        # processes under the host tempdir, or across arenas within one process)
        # can compile another arena's source and link a .so missing this arena's
        # symbols. One dir per stub means no shared ninja scratch or .so, ever.
        out_dir = os.path.join(tempfile.gettempdir(), "sgl_kv_vmm_arena", self._sfx)
        os.makedirs(out_dir, exist_ok=True)
        libname = f"sgl_kv_vmm_arena_stub_{self._sfx}"
        # A build killed before it released torch's `lock` here would stall
        # every later arena with the same suffix; the guard bounds that wait.
        # See sglang.jit_kernel.baton_health.
        from sglang.jit_kernel.baton_health import jit_build_guard

        with jit_build_guard(libname, build_directory=out_dir):
            torch.utils.cpp_extension.load_inline(
                name=libname,
                cpp_sources=_stub_source(self._sfx),
                with_cuda=False,  # pure arithmetic — no nvcc, no CUDA headers
                is_python_module=False,
                verbose=False,
                build_directory=out_dir,
            )
        self._so_path = f"{out_dir}/{libname}.so"
        lib = ctypes.CDLL(self._so_path)
        self._fn_set_base = getattr(lib, f"kvarena_set_base_{self._sfx}")
        self._fn_set_base.argtypes = [ctypes.c_void_p]
        self._fn_set_base.restype = None
        self._fn_set_reserved = getattr(lib, f"kvarena_set_reserved_{self._sfx}")
        self._fn_set_reserved.argtypes = [ctypes.c_size_t]
        self._fn_set_reserved.restype = None
        self._fn_set_align = getattr(lib, f"kvarena_set_align_{self._sfx}")
        self._fn_set_align.argtypes = [ctypes.c_size_t]
        self._fn_set_align.restype = None
        self._fn_cursor = getattr(lib, f"kvarena_cursor_{self._sfx}")
        self._fn_cursor.argtypes = []
        self._fn_cursor.restype = ctypes.c_size_t
        return lib

    def commit_range(self, offset: int, want_bytes: int) -> None:
        """Back ``[base+offset, base+offset+want_bytes)`` (monotonic per offset).
        ``offset`` must be granularity-aligned (the bump allocator guarantees it).
        Maps one full handle per extension -- GB300 rejects partial-handle maps."""
        if self._closed:
            raise RuntimeError("KvVmmArena.commit_range after close")
        if offset % self.granularity != 0:
            raise ValueError(
                f"commit_range offset {offset} not granularity-aligned "
                f"({self.granularity})"
            )
        want = self._align(int(want_bytes))
        prev = self._committed_by_offset.get(offset, 0)
        if want <= prev:
            return
        if offset + want > self.reserved:
            raise RuntimeError(
                f"commit_range [{offset}, {offset + want}) exceeds reservation "
                f"{self.reserved}"
            )
        drv = _driver()
        extents = self._extents_by_offset.setdefault(offset, [])
        pos = prev
        with torch.cuda.device(self.device_id):
            while pos < want:
                step = want - pos
                if self._chunk is not None:
                    step = min(step, self._chunk)
                addr = self.base + offset + pos
                handle = self._take_retained(step)
                reused = handle is not None
                if handle is None:
                    # The park is checked FIRST and the driver only after,
                    # so the steady state after one round trip is zero
                    # driver allocations. On a genuine miss the park itself
                    # may be what is holding the pages this create needs
                    # (sizes changed), so it is offered as a reclaim source
                    # alongside torch's cache.
                    handle = _mem_create_reclaiming(
                        step, self._prop, reclaim=self.drop_retained
                    )
                try:
                    _check(drv.cuMemMap(addr, step, 0, handle, 0), "cuMemMap")
                    _check(
                        drv.cuMemSetAccess(addr, step, [self._access], 1),
                        "cuMemSetAccess",
                    )
                except Exception:
                    # Roll back this failed extent; keep the extents mapped so
                    # far (the watermark below reflects them truthfully).
                    unmap = drv.cuMemUnmap(addr, step)
                    unmap = unmap[0] if isinstance(unmap, tuple) else unmap
                    # A REUSED handle goes back to the park, not to the
                    # driver. Releasing it here would silently shrink the
                    # retained pool on every failed map, so a transient
                    # mapping error would degrade the seam to allocating
                    # again -- with no log line saying so.
                    if reused:
                        self._park_retained(step, handle)
                    else:
                        rel = drv.cuMemRelease(handle)
                        rel = rel[0] if isinstance(rel, tuple) else rel
                    raise
                extents.append((pos, step, handle))
                pos += step
                self._range_backed += step
                self._committed_by_offset[offset] = pos

    def decommit_range(self, offset: int, keep_bytes: int) -> int:
        """Release every whole extent of ``offset`` at or above ``keep_bytes``
        back to the driver (#330 dial). The keep point is rounded UP to the
        next extent boundary so bytes below it are never lost; with chunked
        commits the rounding error is bounded by one chunk. Returns the bytes
        actually released. Caller must ensure the released tail is quiescent
        (idle boundary); a defensive synchronize still precedes the unmap.
        """
        if self._closed:
            raise RuntimeError("KvVmmArena.decommit_range after close")
        keep = align_up(max(int(keep_bytes), 0), self.granularity)
        extents = self._extents_by_offset.get(offset, [])
        if not extents or self._committed_by_offset.get(offset, 0) <= keep:
            return 0
        drv = _driver()
        kept = []
        released = 0
        with torch.cuda.device(self.device_id):
            torch.cuda.synchronize()
            for rel, size, handle in extents:
                if rel >= keep:
                    _check(
                        drv.cuMemUnmap(self.base + offset + rel, size), "cuMemUnmap"
                    )
                    if self._retain_handles:
                        self._park_retained(size, handle)
                    else:
                        _check(drv.cuMemRelease(handle), "cuMemRelease")
                    released += size
                    self._range_backed -= size
                else:
                    kept.append((rel, size, handle))
        self._extents_by_offset[offset] = kept
        self._committed_by_offset[offset] = max(
            (rel + size for rel, size, _ in kept), default=0
        )
        return released

    def _park_retained(self, size: int, handle) -> None:
        self._retained.setdefault(int(size), []).append(handle)
        self._retained_bytes += int(size)

    def _take_retained(self, size: int):
        """A parked handle of EXACTLY ``size`` bytes, or None."""
        if not self._retain_handles:
            return None
        bucket = self._retained.get(int(size))
        if not bucket:
            return None
        handle = bucket.pop()
        self._retained_bytes -= int(size)
        return handle

    @property
    def retained_bytes(self) -> int:
        """Physical bytes parked: unmapped, still owned, NOT driver-free.

        This is the honest name for what retention costs. ``decommit_range``
        keeps returning bytes UNMAPPED, which is what its callers log, and
        with retention on those bytes are no longer the same thing as bytes
        handed back to the driver -- so a caller that reports "released N
        MiB" while this is non-zero is reporting address space, not free
        memory. Read both or neither.
        """
        return int(self._retained_bytes)

    def drop_retained(self) -> int:
        """Release every parked handle to the driver; return bytes freed.

        The escape hatch for the one case retention cannot serve: a commit
        whose size never matches anything parked (a layout resize), where
        the park would otherwise hold pages the commit needs.
        """
        drv = _driver()
        freed = 0
        for size, bucket in self._retained.items():
            for handle in bucket:
                err = drv.cuMemRelease(handle)
                err = err[0] if isinstance(err, tuple) else err
                if err != drv.CUresult.CUDA_SUCCESS:  # pragma: no cover
                    logger.warning("cuMemRelease(retained) -> %s", err)
                else:
                    freed += int(size)
        self._retained.clear()
        self._retained_bytes = 0
        return freed

    def committed_bytes(self, offset: int) -> int:
        """The contiguous physically-committed watermark at ``offset``."""
        return self._committed_by_offset.get(offset, 0)

    @property
    def backed_bytes(self) -> int:
        """Total physically-backed bytes (sum of scattered per-buffer ranges)."""
        return self._range_backed

    @property
    def cursor_bytes(self) -> int:
        return int(self._fn_cursor())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        drv = _driver()
        try:
            torch.cuda.synchronize()
        except Exception as e:  # pragma: no cover
            logger.warning("KvVmmArena.close synchronize failed: %s", e)
        for offset, extents in self._extents_by_offset.items():
            for rel, size, handle in extents:
                err = drv.cuMemUnmap(self.base + offset + rel, size)
                err = err[0] if isinstance(err, tuple) else err
                if err != drv.CUresult.CUDA_SUCCESS:
                    logger.warning("cuMemUnmap range -> %s", err)
                err = drv.cuMemRelease(handle)
                err = err[0] if isinstance(err, tuple) else err
                if err != drv.CUresult.CUDA_SUCCESS:
                    logger.warning("cuMemRelease range -> %s", err)
        self._extents_by_offset.clear()
        # Parked handles are owned physical memory; skipping them here
        # leaks the entire retention pool for the life of the process.
        self.drop_retained()
        err = drv.cuMemAddressFree(self.base, self.reserved)
        err = err[0] if isinstance(err, tuple) else err
        if err != drv.CUresult.CUDA_SUCCESS:
            logger.warning("cuMemAddressFree -> %s", err)


# torch's caching allocator hands the pluggable allocator whole large-pool segments
# (rounded up to >= ~20 MiB) per tensor, so reserve slack beyond the tight tensor sum.
# VA is free until committed, so this costs only address space, not GPU memory.
_PER_BUFFER_VA_SLACK = 32 << 20


class _BufferSpec:
    """Per-buffer placement + backing state inside the shared VA reservation."""

    __slots__ = ("desc", "offset", "reserved_span", "aligned_reserved", "backed_to")

    def __init__(
        self,
        desc: KvBufferDesc,
        offset: int,
        reserved_span: int,
        aligned_reserved: int,
    ):
        self.desc = desc
        self.offset = offset  # granularity-aligned arena offset of this buffer
        self.reserved_span = reserved_span  # logical (unaligned) tensor bytes
        self.aligned_reserved = aligned_reserved  # reserved span rounded to granularity
        self.backed_to = 0  # bytes from offset currently backed


class KvVmmBufferOwner:
    """Owns one ``KvVmmArena`` plus its incrementally-backed KV buffers.

    ``buffer_descs`` is an ordered list of ``KvBufferDesc``; the created ``torch.empty``
    tensors are exposed in the same order as ``self.tensors``.
    """

    def __init__(
        self,
        *,
        device: str,
        device_id: int,
        store_dtype: torch.dtype,
        page_size: int,
        reserved_num_tokens: int,
        buffer_descs: Sequence[KvBufferDesc],
        commit_chunk_bytes: Optional[int] = None,
        retain_handles: bool = False,
    ):
        self.device = device
        self.device_id = int(device_id)
        self.store_dtype = store_dtype
        self.page_size = int(page_size)
        self._reserved_num_tokens = int(reserved_num_tokens)
        self._final_num_tokens: Optional[int] = None
        self._arena: Optional[KvVmmArena] = None
        self._specs: List[_BufferSpec] = []
        self.tensors: List[torch.Tensor] = []

        itemsize = store_dtype.itemsize
        with torch.cuda.device(self.device_id):
            gran = query_granularity(self.device_id)
            reserved_spans = [d.reserved_span_bytes(itemsize) for d in buffer_descs]
            aligned = [align_up(s, gran) for s in reserved_spans]
            reserve_bytes = sum(a + _PER_BUFFER_VA_SLACK for a in aligned) + gran
            if retain_handles and commit_chunk_bytes is None:
                # Named loudly rather than silently tolerated: retention
                # with one monolithic handle per buffer parks memory that
                # the other layout's differently-sized commit can never
                # reuse, so it is pure cost. See KvVmmArena.__init__.
                logger.warning(
                    "KvVmmBufferOwner: retain_handles was requested without a "
                    "commit chunk. Handles are then per-buffer monoliths whose "
                    "sizes differ between the PP and TP layouts, so nothing is "
                    "reusable and the park is pure cost. Retention is DISABLED "
                    "for this owner; set a commit chunk to enable it."
                )
                retain_handles = False
            self._arena = KvVmmArena(
                self.device_id,
                reserve_bytes=reserve_bytes,
                commit_chunk_bytes=commit_chunk_bytes,
                retain_handles=retain_handles,
            )
            assert self._arena.granularity == gran, (self._arena.granularity, gran)

            # NORMAL torch tensors through the arena MemPool; torch.empty never touches
            # the unbacked tail.
            with torch.cuda.use_mem_pool(self._arena.pool):
                self.tensors = [
                    torch.empty(d.shape, dtype=store_dtype, device=self.device)
                    for d in buffer_descs
                ]

            specs: List[_BufferSpec] = []
            for desc, tensor, reserved_span, aligned_reserved in zip(
                buffer_descs, self.tensors, reserved_spans, aligned
            ):
                if prod(tensor.shape) * itemsize != reserved_span:
                    raise RuntimeError(
                        f"buffer {desc.name!r} tensor bytes "
                        f"{prod(tensor.shape) * itemsize} != reserved span {reserved_span}"
                    )
                offset = tensor.data_ptr() - self._arena.base
                if offset < 0 or offset % gran != 0:
                    raise RuntimeError(
                        f"buffer {desc.name!r} arena offset {offset} not "
                        f"granularity-aligned ({gran})"
                    )
                if offset + aligned_reserved > self._arena.reserved:
                    raise RuntimeError(
                        f"buffer {desc.name!r} [{offset}, {offset + aligned_reserved}) "
                        f"exceeds reservation {self._arena.reserved}"
                    )
                specs.append(_BufferSpec(desc, offset, reserved_span, aligned_reserved))
            self._specs = specs

            # Back one page so slot 0 is resident before capture: capture routes every
            # dummy KV write to slot 0 (out_cache_loc is zeros). finalize() backs the rest.
            self.ensure_prefix(self.page_size)

        for t in self.tensors:
            assert (
                t.is_cuda and t.device.index == self.device_id
            ), f"post-capture KV buffer landed on {t.device}, expected cuda:{self.device_id}"

    # -- backing --------------------------------------------------------------

    @staticmethod
    def _check_span(spec: _BufferSpec, span: int) -> int:
        """Return ``span`` if it fits ``[0, reserved_span]``; raise otherwise."""
        span = int(span)
        if not (0 <= span <= spec.reserved_span):
            raise ValueError(
                f"buffer {spec.desc.name!r}: span {span} outside "
                f"[0, {spec.reserved_span}] (reserved tensor bytes)"
            )
        return span

    def _back_spans(self, span_bytes: Sequence[int]) -> None:
        """Back each buffer to (at least) ``span_bytes[i]``. An out-of-reservation
        span is a descriptor bug: raise before committing anything, never clamp."""
        if self._arena is None:
            raise RuntimeError("backing after close / before construction")
        for spec, span in zip(self._specs, span_bytes):
            self._check_span(spec, span)
        gran = self._arena.granularity
        for spec, span in zip(self._specs, span_bytes):
            want = align_up(
                int(span), gran
            )  # <= aligned_reserved since span <= reserved
            if want > spec.backed_to:
                self._arena.commit_range(spec.offset, want)
                spec.backed_to = want

    def ensure_prefix(self, num_tokens: int) -> None:
        """Ensure the first ``num_tokens`` slots of every buffer are physically backed."""
        self._back_spans(
            [s.desc.prefix_span_bytes(num_tokens, self.page_size) for s in self._specs]
        )

    def finalize(self, final_num_tokens: int) -> None:
        """Back each buffer's final advertised span; set the final serving capacity."""
        final = int(final_num_tokens)
        if not (self.page_size <= final <= self._reserved_num_tokens):
            raise ValueError(
                f"final_num_tokens={final} must satisfy page_size="
                f"{self.page_size} <= final <= reserved={self._reserved_num_tokens}"
            )
        self._back_spans(
            [s.desc.final_span_bytes(final, self.page_size) for s in self._specs]
        )
        self._final_num_tokens = final

    def shrink(self, final_num_tokens: int) -> int:
        """#330 dial: decommit every buffer's backing above the span of
        ``final_num_tokens`` and return the bytes actually released to the
        driver. Release is extent-granular (see ``decommit_range``), so the
        remaining backing may exceed the exact span by < 1 commit chunk per
        buffer — ``backed_bytes`` stays the truthful number. Caller must hold
        an idle boundary: rows above the new span must be dead."""
        if self._arena is None:
            raise RuntimeError("shrink after close / before construction")
        final = int(final_num_tokens)
        if not (self.page_size <= final <= self._reserved_num_tokens):
            raise ValueError(
                f"shrink final_num_tokens={final} must satisfy page_size="
                f"{self.page_size} <= final <= reserved={self._reserved_num_tokens}"
            )
        released = 0
        for spec in self._specs:
            keep = spec.desc.final_span_bytes(final, self.page_size)
            released += self._arena.decommit_range(spec.offset, keep)
            spec.backed_to = self._arena.committed_bytes(spec.offset)
        self._final_num_tokens = final
        return released

    # -- accessors / teardown -------------------------------------------------

    @property
    def backed_bytes(self) -> int:
        return self._arena.backed_bytes if self._arena is not None else 0

    def close(self) -> None:
        self.tensors = []
        self._specs = []
        if self._arena is not None:
            self._arena.close()
            self._arena = None
