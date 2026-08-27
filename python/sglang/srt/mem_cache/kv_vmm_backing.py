from __future__ import annotations

import ctypes
import dataclasses
import logging
import os
import tempfile
import weakref
from math import prod
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

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


#: Below this much torch slack the preemption is not worth its cost: spending
#: ``empty_cache`` to recover a few MiB re-warms the allocator for nothing, and
#: on a walk that is genuinely short it would fire on every commit step.
_CORRIDOR_MIN_SLACK_BYTES = 64 * 1024 * 1024


def _corridor_law_floor_bytes() -> int:
    """The corridor law floor in bytes, or 0 when the preemption is disabled.

    Read per call rather than cached at import: the arena is built long before
    the first flip, and a value frozen at import time cannot be corrected by a
    boot that sets the variable later. The read is a dict lookup.

    This is the LAW floor (1024 MiB), deliberately NOT
    ``SGLANG_CORRIDOR_FLOOR_MIB`` (1536), which is the admission gate's ARMING
    floor and carries a margin on top of the law. Preempting on the arming
    floor here would spend the allocator cache on a walk that was never going
    to break the law.
    """
    # 656: read through the ONE declaration rather than re-implementing the
    # env read with a private "1024" fallback here. Three modules had their
    # own copy, so the law could be moved for one and not the others -- a
    # divergence with no symptom until a breach is judged twice and answered
    # differently. Still read per call, for the reason above.
    from sglang.srt.managers.corridor_guard import corridor_law_bytes

    return corridor_law_bytes()


def _corridor_preempt(step: int, label: str, reclaim: Optional[callable]) -> None:
    """Spend torch's cache BEFORE a commit that would cross the corridor law.

    THE DEFECT THIS CLOSES, measured on metal (#656, 2026-08-12 01:18:26-29,
    PP1 on a 3080). ``_mem_create_reclaiming`` below already knows the exact
    remedy -- torch sits on reserved-but-unused blocks that the arena needs as
    RAW driver pages, and ``empty_cache`` returns them. Its trigger, however,
    is ``CUDA_ERROR_OUT_OF_MEMORY``: it fires when free memory reaches ZERO.
    The corridor law floor is 1024 MiB ABOVE zero, so a restore walk marching
    down in 8-24 MiB commit steps crosses the law long before anything refuses,
    and the remedy never runs.

    That is not hypothetical. A ``pp_to_tp`` cutover entered on the law with
    3006 MiB free, drew a 2066 MiB transient through this very path, and sat at
    **940 MiB free for 1.5 s** -- 84 MiB under the law. The seam census recorded
    ``slack=1054`` at the trough: torch was holding 1054 MiB of cached blocks
    throughout, more than enough to have kept the walk legal, and nothing asked
    for them because nothing had failed yet.

    So the trigger moves from "the driver refused" to "the next commit would
    cross the law", and the remedy is unchanged. Properties that matter:

    * **Rank-local.** No collective is entered and no group verdict is
      consumed, so this cannot split a group decision (register laws 14, 15).
      Every rank makes the same KIND of decision on its own card's numbers.
    * **It does not shrink the pool.** The bytes come from torch's cache, which
      by definition nothing is using; the KV pool's capacity is untouched
      (register law 13 -- a smaller pool is never the fix).
    * **The common path is one driver read.** ``mem_get_info`` is the same call
      the seam census already makes at every stage boundary.
    * **It preempts rather than recovers**, which is the whole point: a walk
      that has already crossed the law cannot be un-crossed by a later reclaim.

    Silent by design when it does nothing. When it acts it says so once per
    commit, because a mechanism that cannot be shown to have fired is
    indistinguishable from one that is never reached.
    """
    floor = _corridor_law_floor_bytes()
    if floor <= 0:
        return
    try:
        free, _total = torch.cuda.mem_get_info()
    except Exception:  # noqa: BLE001 -- a probe must never break the walk
        return
    if free - int(step) >= floor:
        return
    try:
        reserved = int(torch.cuda.memory_reserved())
        allocated = int(torch.cuda.memory_allocated())
    except Exception:  # noqa: BLE001
        return
    slack = reserved - allocated
    if slack < _CORRIDOR_MIN_SLACK_BYTES:
        # Nothing worth spending. The walk proceeds and the census records
        # where it went: an unfundable walk is a REAL finding about the
        # configuration, and hiding it behind pointless cache churn would
        # remove the evidence that the budget is too tight.
        return
    logger.warning(
        "%s: committing %d bytes would leave %d MiB free, below the %d MiB "
        "corridor law floor. Releasing torch's cached blocks FIRST (%d MiB of "
        "slack held, reserved %.2f GiB / allocated %.2f GiB). This is the "
        "same reclaim the OUT_OF_MEMORY path takes, moved ahead of the "
        "crossing instead of after the refusal.",
        label,
        int(step),
        (free - int(step)) // (1024 * 1024),
        floor // (1024 * 1024),
        slack // (1024 * 1024),
        reserved / (1 << 30),
        allocated / (1 << 30),
    )
    try:
        torch.cuda.empty_cache()
        if reclaim is not None:
            freed = reclaim()
            if freed:
                logger.warning(
                    "%s: released %.2f GiB of parked handles as well, ahead of "
                    "the corridor crossing.",
                    label,
                    freed / (1 << 30),
                )
    except Exception as err:  # noqa: BLE001 -- the walk owns the no-return path
        logger.warning("%s: corridor preemption failed: %s", label, err)


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
    # PREEMPT THE CORRIDOR CROSSING BEFORE THE COMMIT, not after a refusal.
    # The OUT_OF_MEMORY branch below is the same remedy at a trigger 1024 MiB
    # too late: it waits for the driver to refuse, and the corridor law is
    # broken long before anything is refused. See ``_corridor_preempt``.
    _corridor_preempt(step, label, reclaim)
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

#: Every live arena, for READ-ONLY census by the VRAM flight recorder (#605).
#:
#: WHY THIS EXISTS. torch's ``reserved`` counts what this allocator handed out
#: -- the pool's LOGICAL size -- while NVML counts the pages actually mapped.
#: On the ship config the two differ by 4.6-8.3 GiB per rank, and with no way
#: to read the arena's own committed total that difference could only be
#: ARGUED. The census makes "commit watermark -> free VRAM -> corridor" a
#: measured chain.
#:
#: A ``WeakValueDictionary`` and not a ``WeakSet``: an arena must not be kept
#: alive by being observed, and the WeakSet form of this pattern was already
#: fixed once in the attention workspace registry for exactly that reason.
#: Keyed by instance counter, which is unique per process.
_LIVE_ARENAS: weakref.WeakValueDictionary[int, KvVmmArena] = (
    weakref.WeakValueDictionary()
)


def arena_census() -> Dict[int, Dict[str, int]]:
    """``{device_id: {reserved, backed, retained, arenas}}`` over live arenas.

    Read-only and allocation-free: it reads counters the arena already keeps.
    Never raises -- a census that can fail a boot is not an instrument.

    ``backed`` is mapped physical memory. ``retained`` is physical memory this
    arena still OWNS but has unmapped (parked handles); NVML charges the
    process for it while ``backed`` does not, so reporting only ``backed``
    would leave a real, resident post invisible.
    """
    out: Dict[int, Dict[str, int]] = {}
    try:
        arenas = list(_LIVE_ARENAS.values())
    except Exception:  # pragma: no cover - registry races are not worth a boot
        return out
    for arena in arenas:
        try:
            if getattr(arena, "_closed", True):
                continue
            row = out.setdefault(
                int(arena.device_id),
                {"reserved": 0, "backed": 0, "retained": 0, "arenas": 0},
            )
            row["reserved"] += int(getattr(arena, "reserved", 0) or 0)
            row["backed"] += int(getattr(arena, "_range_backed", 0) or 0)
            row["retained"] += int(getattr(arena, "_retained_bytes", 0) or 0)
            row["arenas"] += 1
        except Exception:  # pragma: no cover
            continue
    return out


#: The lever that makes #464 reachable from a boot. See
#: :func:`resolve_coalesce_resume`.
COALESCE_RESUME_ENV = "SGLANG_VMM_COALESCE_RESUME"


def resolve_coalesce_resume(explicit: Optional[bool]) -> bool:
    """Decide whether ``commit_range`` coalesces, honouring an explicit caller.

    WHY THIS EXISTS. The coalescer shipped with a constructor flag and the
    stated purpose "so the measurement can be taken". Nothing passed it: both
    carrier arenas (``phase_flip_spill.py:597``, ``:927``) and the KV seam
    owner (``memory_pool.py:2488``) construct without the argument, so the flag
    was False in every real boot and the measurement it exists for could not be
    taken. A switch whose actuator is unreachable is not a dark feature, it is
    an absent one.

    Resolved HERE, at the single place the flag is stored, rather than threaded
    through every construction site: a site that has no opinion needs no edit
    and cannot drift out of sync with the others.

    An explicit argument WINS over the environment -- ambient state must not
    overrule a caller that has decided. Only ``None`` (no opinion) consults the
    lever.

    Parsed locally rather than through ``utils.common.get_bool_env_var``
    because this module deliberately imports nothing from ``sglang``; the
    truthy set is kept identical to it.
    """
    if explicit is not None:
        return bool(explicit)
    return os.environ.get(COALESCE_RESUME_ENV, "").strip().lower() in ("1", "true")


class VmmCoalesceRefused(RuntimeError):
    """A coalesce was REQUIRED by the caller and the region could not honour it."""


@dataclasses.dataclass(frozen=True)
class CoalesceReport:
    """What the coalescer did, so a caller can log it rather than guess."""

    coalesced: bool
    extents_before: int
    extents_after: int
    bytes_total: int
    reason: str

    @property
    def driver_calls_saved(self) -> int:
        """One map + one setAccess per extent removed."""
        return max(0, (self.extents_before - self.extents_after) * 2)


def coalesce_commit_plan(plan, *, enabled: bool, require_contiguous: bool = False):
    """#464: one physical handle per CONTIGUOUS VA run, not one per chunk.

    ``KvVmmArena.commit_range`` splits a gap into ``self._chunk``-sized extents
    (the #330 dial), so a ~1 GiB resume becomes ~500 x 2 MiB extents and hence
    ~500 map + ~500 setAccess driver calls. When the run is contiguous those
    extents describe ONE VA region and can be backed by one handle, taking the
    resume to ~3 calls (map, setAccess, memset).

    **Contiguity is a precondition, not an assumption.** ``decommit_span`` can
    leave an interior HOLE -- see the #631 note inside ``commit_range`` -- so a
    non-contiguous plan is a legitimate state, not a bug. This function then
    REFUSES TO COALESCE and says why, rather than merging across the hole
    (which would map pages nobody asked for) or splitting into some other shape
    silently (which the caller cannot predict).

    #286 asset classes are respected by construction: a plan is built inside a
    single ``commit_range`` call, which addresses ONE ``offset`` -- one buffer
    -- so no coalesced run can span two independently parked assets.

    DEFAULT OFF. The 40-85 ms band is unmeasured; the flag exists so the
    measurement can be taken, not because the win is assumed.
    """
    plan = list(plan)
    total = sum(step for _, step in plan)
    if not enabled:
        return plan, CoalesceReport(
            False, len(plan), len(plan), total, "coalescing disabled (default)"
        )
    if len(plan) <= 1:
        return plan, CoalesceReport(
            False, len(plan), len(plan), total, "nothing to coalesce"
        )

    gaps = []
    for (pos_a, step_a), (pos_b, _) in zip(plan, plan[1:]):
        if pos_a + step_a != pos_b:
            gaps.append((pos_a + step_a, pos_b))
    if gaps:
        reason = (
            f"region is not contiguous: {len(gaps)} hole(s), first at "
            f"[{gaps[0][0]}, {gaps[0][1]}). Refusing to coalesce -- merging "
            "across a hole would map pages nobody asked for, and splitting "
            "silently would give a shape the caller cannot predict."
        )
        if require_contiguous:
            raise VmmCoalesceRefused(reason)
        return plan, CoalesceReport(False, len(plan), len(plan), total, reason)

    merged = [(plan[0][0], total)]
    return merged, CoalesceReport(
        True,
        len(plan),
        1,
        total,
        f"coalesced {len(plan)} contiguous extents into one {total}-byte handle",
    )


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
        coalesce_resume: Optional[bool] = None,
    ):
        # #464, DEFAULT OFF: the 40-85 ms band is unmeasured, so the flag
        # exists to make the measurement possible, not because the win is
        # assumed. Off reproduces the per-#330-chunk plan byte-for-byte.
        self._coalesce_resume = resolve_coalesce_resume(coalesce_resume)
        self.device_id = int(device_id)
        # Unique per (process, arena instance): the stub .so lives in a host-shared
        # tempdir, so co-located engine processes must not build the same-named .so
        # (they race and one loads a half-relinked copy -> undefined symbol crash).
        self._sfx = f"{os.getpid()}_{KvVmmArena._instance_count}"
        # Read-only census hook (#605). Registered before any mapping happens so
        # a crash during construction still leaves a countable arena.
        _LIVE_ARENAS[KvVmmArena._instance_count] = self
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
            # fungible AT ALL. Enabling retention without a chunk is a
            # no-op that costs memory, which is why _back_spans logs the
            # pairing.
            #
            # BUT FUNGIBLE WITHIN ONE ARENA ONLY, and the phase flip's two
            # layouts are two arenas -- one per KV pool, each with its own
            # ``_retained``. The PP arena parks exactly the pages the TP
            # arena is about to ask the driver for, and ``_take_retained``
            # on the TP side can never see them, so both layouts stay
            # resident and exclusive backing is defeated. Retention is
            # therefore OFF by default even when a chunk is set; see
            # ``memory_pool.seam_chunk_and_retention``. It stays reachable
            # because a single-layout arena that grows and shrinks in place
            # (the #330 dial) does recycle its own handles.
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

    @property
    def commit_chunk_bytes(self) -> int:
        """The commit granule, or 0 when the arena maps monolithic extents.

        Read by ``KvVmmBufferOwner.has_commit_chunk``, which the phase
        flip's seam gate consults before choosing the span-granular path.
        """
        return int(self._chunk or 0)

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
        # #631: fill only the UNMAPPED parts of [prev, want).
        #
        # This used to walk straight from the watermark to `want`, which is
        # correct only while coverage is contiguous from zero. decommit_span
        # can leave an interior HOLE, and the watermark then reports the
        # contiguous prefix BELOW that hole -- so a later whole-pool commit
        # (finalize -> restore_backing, the streamed seam's completion step)
        # would re-map extents that are still mapped, double-counting
        # _range_backed and issuing cuMemMap over live mappings. With no
        # holes there is exactly one gap, [prev, want), so the legacy path
        # is byte-identical.
        plan = []
        for gap_lo, gap_hi in self._gaps_in(offset, prev, want):
            p = gap_lo
            while p < gap_hi:
                step = gap_hi - p
                if self._chunk is not None:
                    step = min(step, self._chunk)
                plan.append((p, step))
                p += step
        # #464 (flag-gated, DEFAULT OFF): back one CONTIGUOUS run with one
        # handle instead of one per #330 chunk. Refuses on a hole rather than
        # merging across it; see coalesce_commit_plan.
        plan, coalesce_report = coalesce_commit_plan(
            plan, enabled=self._coalesce_resume
        )
        if coalesce_report.coalesced:
            logger.info(
                "kv-vmm: #464 %s (%d driver calls saved)",
                coalesce_report.reason,
                coalesce_report.driver_calls_saved,
            )
        with torch.cuda.device(self.device_id):
            for pos, step in plan:
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
                    extents.sort()
                    self._refresh_watermark(offset)
                    raise
                extents.append((pos, step, handle))
                self._range_backed += step
        extents.sort()
        self._refresh_watermark(offset)

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
                    _check(drv.cuMemUnmap(self.base + offset + rel, size), "cuMemUnmap")
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

    # -- #631 arbitrary-extent backing, for the STREAMED seam ----------------
    #
    # commit_range/decommit_range are both PREFIX operations: one grows a
    # contiguous watermark up from zero, the other drops the tail above a
    # keep point. The flip's seam cannot be streamed with only those,
    # because reading consumes from one end while writing fills the other:
    # walking rows ascending lets the destination grow as a prefix but
    # leaves the source owing a suffix it cannot release, and walking them
    # descending does the mirror. Either way the two layouts peak at twice
    # one layout. The source and destination row lists are index-aligned
    # and both ascending, so the two directions are LOCKED and no
    # scheduling choice escapes it -- the arena has to be able to back a
    # range that does not start at zero.
    #
    # ROUNDING IS ASYMMETRIC ON PURPOSE. commit rounds OUTWARD so every
    # byte asked for is covered; decommit rounds INWARD so a chunk that is
    # only partly inside the range -- and therefore still holds live rows
    # -- is never unmapped. Reversing either one produces a fault or
    # silent KV corruption at a chunk boundary.

    @staticmethod
    def span_bounds(lo_bytes: int, hi_bytes: int, granularity: int, outward: bool):
        """Normalise a byte span to the MAPPING unit, which is granularity.

        THE CHUNK IS A HANDLE SIZE, NOT AN ALIGNMENT, and conflating the two
        took all three ranks down on the first boot where the streamed seam
        actually engaged (cuMemMap -> CUDA_ERROR_INVALID_VALUE).

        ``commit_span`` used to round outward to the CHUNK while
        ``commit_range`` rounds to the granularity. Buffer VA extents are
        laid out granularity-aligned, so a chunk-rounded ``hi`` overshoots
        the end of its own buffer -- by up to chunk-1 bytes -- and asks the
        driver to map over the NEXT buffer's live mapping. cuMemMap answers
        INVALID_VALUE, the exception climbs out of the seam inside the
        no-return region, and the instance dies.

        It hid because the legacy whole-pool path never produced a span
        ending anywhere but at a buffer's own end, and because the span
        unit tests do not run against a real arena with a neighbour to
        collide with.

        ``outward`` covers every byte asked for (commit); inward keeps a
        chunk that still holds live rows out of the range (decommit), where
        over-releasing is silent KV corruption rather than a fault.
        """
        g = int(granularity)
        lo = max(int(lo_bytes), 0)
        hi = max(int(hi_bytes), 0)
        if outward:
            return (lo // g) * g, align_up(hi, g)
        return align_up(lo, g), (hi // g) * g

    def _require_chunk(self, op: str) -> int:
        if self._chunk is None:
            raise RuntimeError(
                f"KvVmmArena.{op} requires a commit chunk. Without "
                f"commit_chunk_bytes the arena maps ONE monolithic extent "
                f"per buffer, and cuMemUnmap only takes whole mappings, so "
                f"an extent-range op could only release everything or "
                f"nothing. Construct the pool with a commit chunk."
            )
        return self._chunk

    def _extent_intervals(self, offset: int):
        """Mapped [start, end) intervals at ``offset``, ascending."""
        return sorted(
            (rel, rel + size)
            for rel, size, _h in self._extents_by_offset.get(offset, [])
        )

    def _gaps_in(self, offset: int, lo: int, hi: int):
        """The UNMAPPED sub-ranges of ``[lo, hi)`` at ``offset``, ascending.

        Shared by ``commit_range`` and ``commit_span`` so the two cannot
        disagree about what is already backed. When coverage is contiguous
        and reaches ``lo`` there is exactly one gap, ``[lo, hi)``, which is
        what makes the legacy prefix path byte-identical through here.
        """
        gaps = []
        cursor = lo
        for start, stop in self._extent_intervals(offset):
            if stop <= lo or start >= hi:
                continue
            if start > cursor:
                gaps.append((cursor, min(start, hi)))
            cursor = max(cursor, stop)
            if cursor >= hi:
                break
        if cursor < hi:
            gaps.append((cursor, hi))
        return gaps

    def _refresh_watermark(self, offset: int) -> None:
        """``_committed_by_offset`` is the CONTIGUOUS-from-zero watermark.

        Keeping it contiguous rather than "highest mapped byte" is what
        lets the legacy prefix path stay correct while a span op has left
        a hole: a watermark that counted a suffix would make
        ``commit_range`` skip the gap below it and hand back a pool with
        unbacked rows in the middle.
        """
        end = 0
        for start, stop in self._extent_intervals(offset):
            if start > end:
                break
            end = max(end, stop)
        self._committed_by_offset[offset] = end

    def commit_span(self, offset: int, lo_bytes: int, hi_bytes: int) -> int:
        """Back ``[offset+lo, offset+hi)``, skipping chunks already mapped.

        Returns the bytes newly committed. Idempotent: re-committing a
        covered span allocates nothing, which is what keeps a streamed
        seam from allocating once per row block.
        """
        if self._closed:
            raise RuntimeError("KvVmmArena.commit_span after close")
        chunk = self._require_chunk("commit_span")
        if offset % self.granularity != 0:
            raise ValueError(
                f"commit_span offset {offset} not granularity-aligned "
                f"({self.granularity})"
            )
        # Granularity, NOT chunk: the chunk caps the HANDLE size below, but
        # the mapping unit -- and the alignment every buffer's VA extent was
        # laid out on -- is the granularity. See span_bounds.
        lo, hi = self.span_bounds(lo_bytes, hi_bytes, self.granularity, True)
        if hi <= lo:
            return 0
        if offset + hi > self.reserved:
            raise RuntimeError(
                f"commit_span [{offset + lo}, {offset + hi}) exceeds "
                f"reservation {self.reserved}"
            )
        # Fill only the GAPS, computed from the real extent list rather
        # than a watermark -- the boot page commit can leave a first
        # extent smaller than a chunk, so chunk-aligned arithmetic alone
        # would mis-detect coverage.
        gaps = self._gaps_in(offset, lo, hi)

        drv = _driver()
        extents = self._extents_by_offset.setdefault(offset, [])
        committed = 0
        with torch.cuda.device(self.device_id):
            for gap_lo, gap_hi in gaps:
                pos = gap_lo
                while pos < gap_hi:
                    step = min(chunk, gap_hi - pos)
                    addr = self.base + offset + pos
                    handle = self._take_retained(step)
                    reused = handle is not None
                    if handle is None:
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
                        unmap = drv.cuMemUnmap(addr, step)
                        unmap = unmap[0] if isinstance(unmap, tuple) else unmap
                        if reused:
                            self._park_retained(step, handle)
                        else:
                            rel = drv.cuMemRelease(handle)
                            rel = rel[0] if isinstance(rel, tuple) else rel
                        self._refresh_watermark(offset)
                        raise
                    extents.append((pos, step, handle))
                    self._range_backed += step
                    committed += step
                    pos += step
        extents.sort()
        self._refresh_watermark(offset)
        return committed

    def decommit_span(self, offset: int, lo_bytes: int, hi_bytes: int) -> int:
        """Release every extent lying WHOLLY inside ``[lo, hi)``.

        Returns the bytes released. Caller must hold an idle boundary for
        the released rows; a defensive synchronize precedes the unmap.
        """
        if self._closed:
            raise RuntimeError("KvVmmArena.decommit_span after close")
        # Kept for its REFUSAL, not its value: on an unchunked arena the
        # extents are per-buffer monoliths and a range release could only
        # give back everything or nothing.
        self._require_chunk("decommit_span")
        lo, hi = self.span_bounds(lo_bytes, hi_bytes, self.granularity, False)
        extents = self._extents_by_offset.get(offset, [])
        if not extents or hi <= lo:
            return 0
        drv = _driver()
        kept = []
        released = 0
        with torch.cuda.device(self.device_id):
            torch.cuda.synchronize()
            for rel, size, handle in extents:
                if rel >= lo and rel + size <= hi:
                    _check(drv.cuMemUnmap(self.base + offset + rel, size), "cuMemUnmap")
                    if self._retain_handles:
                        self._park_retained(size, handle)
                    else:
                        _check(drv.cuMemRelease(handle), "cuMemRelease")
                    released += size
                    self._range_backed -= size
                else:
                    kept.append((rel, size, handle))
        self._extents_by_offset[offset] = kept
        self._refresh_watermark(offset)
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


def arena_reserve_bytes(reserved_spans: Sequence[int], granularity: int) -> int:
    """VA bytes ``KvVmmBufferOwner`` reserves for a set of buffer spans.

    EXTRACTED SO IT CAN BE ASKED WITHOUT A GPU. The arena's size is a pure
    function of the per-buffer spans and the driver granularity, but it used to
    be spelled inline inside ``__init__`` between two CUDA calls, so the only
    way to learn what a configuration would reserve was to boot and read the
    log. That is how the gapped boot spent six windows discovering a number
    that is arithmetic:

        PP0, 30 buffers (15 layers x k+v) x 754020 slots x 1024 B
          -> 22.56 GiB, logged as "reserved=22.6 GiB", on a 32.6 GiB card
             already holding 27.1 GiB -> cuMemCreate refused at 49.70 GiB
        PP1/PP2, 16 buffers (8 layers) at the same token count
          -> 12.03 GiB, logged as "reserved=12.0 GiB"

    The 15 against the 8 is the defect (see ``model_runner_kv_cache_mixin``'s
    owned-attention-layer resolution): PP0 owns ZERO full-attention layers
    under the gapped set and was sized for fifteen. With ownership resolved
    correctly the same expression returns ``granularity`` -- one 2 MiB page of
    address space, which is the floor, not a 22.6 GiB reservation.

    Note it is VA, not physical memory. The reservation itself is not what the
    driver refused; it is the torch-visible ceiling this arena's MemPool then
    commits against, which is why an over-sized reservation ends in
    ``cuMemCreate: ... refused`` rather than in a clean sizing error.
    """
    gran = int(granularity)
    if gran <= 0:
        raise ValueError(f"granularity must be positive, got {granularity!r}")
    total = gran
    for span in reserved_spans:
        total += align_up(int(span), gran) + _PER_BUFFER_VA_SLACK
    return total


def serviceable_reserved_tokens(
    buffer_descs: Sequence[KvBufferDesc], itemsize: int, page_size: int
) -> int:
    """The largest ``final_num_tokens`` these buffers' BYTES can actually serve.

    #918, AND IT IS THE OTHER HALF OF A NUMBER THAT WAS ONLY EVER RAISED ON ONE
    SIDE. ``KvVmmBufferOwner`` carries two independent derivations of "the
    reservation" and enforces them in two different places:

    * ``_check_final`` bounds a request by ``_reserved_num_tokens`` -- a TOKEN
      COUNT the caller passes in, since #851 F2 (``e62b1fae26``)
      ``lawful_reservation_rows(size, admission_reserve, 0)`` = ``size + 1 +
      reserve``;
    * ``_check_span`` bounds the resulting BYTES by ``spec.reserved_span`` =
      ``prod(desc.shape) * itemsize`` -- the tensor, whose leading dimension
      came from ``size + page_size``.

    Nothing ever tied the two together. #851 F2 raised the token ceiling by
    ``1 + reserve`` rows and left the byte reservation exactly where it was, so
    every value in ``(shape rows - page_size, reserved_num_tokens]`` passes the
    first check and fails the second. Measured, boot_rerun0826 21:59:55Z, PP1::

        reserved tensor bytes 118305792 = (115532 + 1) rows x 1024 B
        _reserved_num_tokens  131917    = 115532 + 1 + 16384
        the dial exposed       124127   -> span (124127 + 1) x 1024
                                        =  127107072  > 118305792   BOOM

    Pure, so the divergence is constructible without a card, a driver or a
    boot -- which is what it took to find it, because both numbers are correct
    in isolation and only their RELATION is wrong.

    ``final_span_bytes(n)`` is ``ceil((n + page_size) / tokens_per_row) *
    row_bytes``, monotone in ``n``, so the inverse is exact: a buffer holding
    ``R = reserved_span // row_bytes`` whole rows serves at most
    ``R * tokens_per_row - page_size`` tokens. The minimum across buffers is
    the owner's real ceiling.
    """
    page = int(page_size)
    best: Optional[int] = None
    for desc in buffer_descs:
        row_bytes = int(desc.row_bytes)
        if row_bytes <= 0:
            return 0
        rows = int(desc.reserved_span_bytes(int(itemsize))) // row_bytes
        per_row = max(1, int(desc.tokens_per_row))
        serviceable = rows * per_row - page
        if best is None or serviceable < best:
            best = serviceable
    return max(0, int(best)) if best is not None else 0


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

        # #918: THE TWO DERIVATIONS OF "THE RESERVATION" MUST AGREE, AND THE
        # ONLY PLACE THAT CAN SEE BOTH IS HERE. ``reserved_num_tokens`` arrives
        # as a token count from the caller; ``buffer_descs`` arrive as shapes.
        # ``_check_final`` enforces the first and ``_check_span`` the second,
        # 13 seconds and one phase flip apart, so a caller that raises one
        # without the other ships a ceiling that is pure fiction and finds out
        # from a dead rank. Refuse at construction instead, naming BOTH numbers
        # and the row count each implies -- a boot-time refusal is recoverable,
        # an unbackable exposed id space is not.
        serviceable = serviceable_reserved_tokens(
            buffer_descs, itemsize, self.page_size
        )
        if self._reserved_num_tokens > serviceable:
            raise ValueError(
                f"KvVmmBufferOwner: reserved_num_tokens={self._reserved_num_tokens} "
                f"exceeds the {serviceable} tokens these buffer shapes can hold "
                f"(smallest buffer's reserved tensor bytes / row_bytes, minus the "
                f"padded page). The token ceiling and the byte reservation are two "
                f"derivations of ONE quantity: raising only the first publishes a "
                f"ceiling `_check_final` accepts and `_check_span` then refuses, "
                f"which is #918. Size the buffer_descs for the reservation "
                f"(MHATokenToKVPool._alloc_post_capture_buffers passes the same "
                f"_lawful_reserved_tokens() to both), or lower the ceiling."
            )

        with torch.cuda.device(self.device_id):
            gran = query_granularity(self.device_id)
            reserved_spans = [d.reserved_span_bytes(itemsize) for d in buffer_descs]
            aligned = [align_up(s, gran) for s in reserved_spans]
            reserve_bytes = arena_reserve_bytes(reserved_spans, gran)
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
            assert t.is_cuda and t.device.index == self.device_id, (
                f"post-capture KV buffer landed on {t.device}, expected cuda:{self.device_id}"
            )

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

    def _back_subset(self, pairs: Sequence[Tuple[int, int]]) -> None:
        """``_back_spans`` for an explicit ``(buffer index, span)`` list."""
        if self._arena is None:
            raise RuntimeError("backing after close / before construction")
        for idx, span in pairs:
            self._check_span(self._specs[idx], span)
        gran = self._arena.granularity
        for idx, span in pairs:
            spec = self._specs[idx]
            want = align_up(int(span), gran)
            if want > spec.backed_to:
                self._arena.commit_range(spec.offset, want)
                spec.backed_to = want

    def ensure_prefix(self, num_tokens: int) -> None:
        """Ensure the first ``num_tokens`` slots of every buffer are physically backed."""
        self._back_spans(
            [s.desc.prefix_span_bytes(num_tokens, self.page_size) for s in self._specs]
        )

    def _check_final(self, final_num_tokens: int) -> int:
        final = int(final_num_tokens)
        if not (self.page_size <= final <= self._reserved_num_tokens):
            raise ValueError(
                f"final_num_tokens={final} must satisfy page_size="
                f"{self.page_size} <= final <= reserved={self._reserved_num_tokens}"
            )
        return final

    def _resolve_indices(self, buffer_indices: Optional[Sequence[int]]) -> List[int]:
        """Validate a buffer subset, or return every index when None.

        An out-of-range index is a caller bug and is raised BEFORE anything
        commits or decommits: the #631 seam calls these per wave from
        inside the flip's no-return region, where a half-applied subset
        would leave one layout's layers backed and the other's not.
        """
        if buffer_indices is None:
            return list(range(len(self._specs)))
        out: List[int] = []
        for i in buffer_indices:
            idx = int(i)
            if not (0 <= idx < len(self._specs)):
                raise ValueError(
                    f"buffer index {idx} outside [0, {len(self._specs)}) -- "
                    f"the owner holds {len(self._specs)} buffers"
                )
            out.append(idx)
        return out

    def finalize(
        self,
        final_num_tokens: int,
        buffer_indices: Optional[Sequence[int]] = None,
    ) -> None:
        """Back each buffer's final advertised span; set the final serving capacity.

        ``buffer_indices`` restricts the commit to a SUBSET of the buffers
        (#631 waved seam): the phase flip commits the destination layout
        one layer wave at a time so that only one wave's worth of pages is
        ever held on top of the resting layout. None = every buffer, which
        is the whole-pool behaviour this call has always had.
        """
        final = self._check_final(final_num_tokens)
        indices = self._resolve_indices(buffer_indices)
        self._back_subset(
            [
                (i, self._specs[i].desc.final_span_bytes(final, self.page_size))
                for i in indices
            ]
        )
        if buffer_indices is None:
            self._final_num_tokens = final

    def shrink(
        self,
        final_num_tokens: int,
        buffer_indices: Optional[Sequence[int]] = None,
    ) -> int:
        """#330 dial: decommit every buffer's backing above the span of
        ``final_num_tokens`` and return the bytes actually released to the
        driver. Release is extent-granular (see ``decommit_range``), so the
        remaining backing may exceed the exact span by < 1 commit chunk per
        buffer — ``backed_bytes`` stays the truthful number. Caller must hold
        an idle boundary: rows above the new span must be dead.

        ``buffer_indices`` restricts the release to a SUBSET (#631 waved
        seam), which is what lets the flip hand back one layer wave at a
        time instead of the whole source layout at once."""
        if self._arena is None:
            raise RuntimeError("shrink after close / before construction")
        final = self._check_final(final_num_tokens)
        released = 0
        for idx in self._resolve_indices(buffer_indices):
            spec = self._specs[idx]
            keep = spec.desc.final_span_bytes(final, self.page_size)
            released += self._arena.decommit_range(spec.offset, keep)
            spec.backed_to = self._arena.committed_bytes(spec.offset)
        if buffer_indices is None:
            self._final_num_tokens = final
        return released

    # -- #631 row-range backing, the streamed seam's unit ---------------------
    #
    # ROUNDING IS ASYMMETRIC AND DELIBERATE. A commit must cover every row
    # that will be WRITTEN, so its span runs from the row's own byte offset
    # up to the PADDED end of the top row. A release must never drop a row
    # that will still be READ, so its span starts past the padded end of
    # the bottom row and stops at the plain offset of the top one. Folding
    # these two into one helper -- they look like the same arithmetic --
    # makes the seam unmap live rows at a chunk boundary, which surfaces as
    # data-dependent KV corruption rather than a fault.

    def back_token_span(
        self,
        lo_tokens: int,
        hi_tokens: int,
        buffer_indices: Optional[Sequence[int]] = None,
    ) -> int:
        """Back tokens ``[lo, hi)`` of a buffer subset. Rounds OUTWARD."""
        if self._arena is None:
            raise RuntimeError("back_token_span after close / before construction")
        committed = 0
        for idx in self._resolve_indices(buffer_indices):
            spec = self._specs[idx]
            lo_b = spec.desc.prefix_span_bytes(lo_tokens, self.page_size)
            hi_b = spec.desc.final_span_bytes(hi_tokens, self.page_size)
            hi_b = min(hi_b, spec.reserved_span)
            committed += self._arena.commit_span(spec.offset, lo_b, hi_b)
            spec.backed_to = self._arena.committed_bytes(spec.offset)
        return committed

    def release_token_span(
        self,
        lo_tokens: int,
        hi_tokens: int,
        buffer_indices: Optional[Sequence[int]] = None,
    ) -> int:
        """Release tokens ``[lo, hi)`` of a buffer subset. Rounds INWARD."""
        if self._arena is None:
            raise RuntimeError("release_token_span after close / before construction")
        released = 0
        for idx in self._resolve_indices(buffer_indices):
            spec = self._specs[idx]
            lo_b = spec.desc.final_span_bytes(lo_tokens, self.page_size)
            hi_b = spec.desc.prefix_span_bytes(hi_tokens, self.page_size)
            released += self._arena.decommit_span(spec.offset, lo_b, hi_b)
            spec.backed_to = self._arena.committed_bytes(spec.offset)
        return released

    # -- accessors / teardown -------------------------------------------------

    @property
    def has_commit_chunk(self) -> bool:
        """Whether the span calls can run at all (see ``_require_chunk``)."""
        return bool(self._arena is not None and self._arena.commit_chunk_bytes)

    @property
    def backed_bytes(self) -> int:
        return self._arena.backed_bytes if self._arena is not None else 0

    @property
    def reserved_rows(self) -> int:
        """The VA reservation in rows -- the CEILING a grow can ever reach.

        IMMUTABLE, and that is why it has to be readable (#684). It is fixed
        once at construction and never assigned again, while ``size`` itself
        is mutable at runtime -- the #330 dial writes it on every step. So a
        caller that derives a grow target from a remembered or configured row
        count can aim ABOVE this number, and ``_check_final`` will refuse it
        every single time.

        WHAT THE CONSTRUCTION VALUE IS, and do not re-read the old answer.
        Until #851 F2 it was ``reserved_num_tokens=self.size`` -- the size at
        that instant -- and THAT is the #848 wall: the dial moved ``size``
        past the boot reservation and no grow could ever be accepted again.
        Since e62b1fae26 the caller passes
        ``MHATokenToKVPool._lawful_reserved_tokens()``, i.e.
        ``lawful_reservation_rows(size, admission_reserve, 0)`` =
        ``size + 1 + reserve`` (memory_pool.py, ``KvVmmBufferOwner(...)``), so
        the reservation covers the largest floor the rung can demand AT THE
        BOOT SIZE. #848 is closed and measured 0 from W24 onward.

        AND UNTIL #918 THIS NUMBER WAS NOT BACKED BY BYTES. F2 raised the
        token ceiling and left ``buffer_descs`` sized for ``self.size``, so
        ``reserved_rows`` ran ``1 + reserve`` rows ahead of the tensors it
        claims to describe: ``_check_final`` accepted targets in that gap and
        ``_check_span`` refused them, one phase flip later, on a live rank.
        ``__init__`` now refuses that pair outright and the caller passes ONE
        ``_lawful_reserved_tokens()`` to both sides, so this property and
        ``MHATokenToKVPool.store_bound_rows`` describe the same extent again.

        WHAT IS STILL NOT COVERED, named rather than implied: the law is not
        a FIXED POINT. ``size`` may lawfully climb to this reservation, and at
        ``size == reserved_rows`` the rung's floor is ``reserved_rows + 1 +
        reserve`` -- above the ceiling again. See
        ``test_reservation_fixed_point_848.py`` for the arithmetic and the
        reason it is pinned rather than patched.

        Measured 2026-08-16, 02:15:24 to 02:35:26, 59 times on three ranks:
        ``recovery to 270646 rows failed: ... <= reserved=190596``. Recovery
        is what LIFTS the backing cap, so 59 refusals meant the cap never
        lifted and the corridor guard's only rung above the allocator cache
        stayed dead for the whole boot.
        """
        return int(self._reserved_num_tokens)

    @property
    def uniform_backed_tokens(self) -> int:
        """Tokens backed in EVERY buffer -- the depth a shrink can act on.

        ``backed_bytes`` is a SUM across buffers, so dividing it by the
        all-buffers per-row size yields an AVERAGE depth. That is only the
        real depth when the backing is uniform, and it is not: the waved seam
        releases and restores a layer at a time, and ``decommit_range`` frees
        only extents lying wholly above the keep point PER BUFFER. So a caller
        that trusts the average computes a keep point above the shallowest
        buffer's watermark, and the shrink returns nothing while looking like
        a large one.

        Measured 2026-08-15 on the 2048-chunk boot: the rung read 591872 rows
        from the average, asked to shrink to 320217 and 352067, and got 0 MiB
        nine times; the shrinks that DID pay were the ones whose target was
        below every buffer (73345 from 149504). This is that minimum.

        Zero when the arena is gone, which reads as "nothing to give" -- the
        safe direction for a number that decides how deep to cut.
        """
        if self._arena is None or not self._specs:
            return 0
        depths = []
        for spec in self._specs:
            committed = self._arena.committed_bytes(spec.offset)
            row_bytes = int(getattr(spec.desc, "row_bytes", 0) or 0)
            if row_bytes <= 0:
                return 0
            per_row = max(1, int(getattr(spec.desc, "tokens_per_row", 1) or 1))
            depths.append((committed // row_bytes) * per_row)
        return int(min(depths)) if depths else 0

    def close(self) -> None:
        self.tensors = []
        self._specs = []
        if self._arena is not None:
            self._arena.close()
            self._arena = None
