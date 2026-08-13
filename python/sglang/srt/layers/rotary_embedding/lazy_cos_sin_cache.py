# SPDX-License-Identifier: Apache-2.0
"""#656 T1: a RoPE cos/sin cache that costs its ceiling only when it is reached.

WHY THIS EXISTS
---------------
A context ceiling is a promise, not a consumption: raising it from 393216 to
1048576 should cost nothing until a session actually reaches those positions.
It did not. The cos/sin cache is materialized in full at layer construction
(``_compute_cos_sin_cache`` builds ``max_position_embeddings * scaling_factor``
rows) and then pre-expanded again before CUDA graph capture
(``reserve_rope_cache_for_long_sequences``). Measured on this rig: 440 MiB per
rank at 1048576 with nothing using the context, which the KV sizer then pays
for out of the pool -- 9.0% of it (register 69).

WHY IT CANNOT SIMPLY GROW
-------------------------
``_ensure_cos_sin_cache_length`` grows by ``torch.cat``, i.e. by REALLOCATION.
That is safe today only because the one runtime caller runs before CUDA graph
capture. A captured graph bakes in the ADDRESS of ``cos_sin_cache``; a cache
that reallocates after capture leaves every replayed decode graph reading a
freed buffer. So a lazy cache needs growth with a STABLE POINTER.

HOW
---
Reserve the full ceiling as CUDA *managed* memory (``cuMemAllocManaged``) and
fill it in chunks. Reserving commits no physical pages -- measured 0 MiB for a
256 MiB reservation on this rig -- and pages become resident when the fill
kernel writes them, so the cost tracks the positions actually reached. The
address never moves, so graph replay stays valid.

THE MiB ARE STILL PRICED, JUST NOT PREPAID: ``have`` is a physical-free
reading, so the sizer sees resident pages as they appear. The reachable
commitment is bounded by the pool -- a position beyond the pool's token count
cannot be occupied -- not by the ceiling.

The rows this module writes go through ``_build_cos_sin_rows``, i.e. through
the same ``_cos_sin_cache_inv_freq`` / ``_cos_sin_cache_row_scale`` hooks that
register 47 established for the growth path. A lazy cache that computed its
own frequencies would reintroduce exactly that bug, silently.
"""

from __future__ import annotations

import ctypes
import logging
import weakref
from typing import TYPE_CHECKING, Optional, Set

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding

logger = logging.getLogger(__name__)

CU_MEM_ATTACH_GLOBAL = 1

# Only classes whose growth hooks are known to reproduce their own
# constructor are eligible. An unknown subclass keeps the eager cache: the
# failure mode of guessing wrong here is silent wrong attention past the
# initial chunk, which is precisely register 47's bug.
_LAZY_SAFE_CLASSES: Set[type] = set()

# Every installed lazy cache, so one runtime hook can serve every model stack
# in the process (the phase-flip instance holds two, plus the draft model).
_LAZY_MODULES: "weakref.WeakSet[RotaryEmbedding]" = weakref.WeakSet()

# Cheapest possible fast path for the per-batch hook: the smallest filled
# count over all installed caches. While a batch stays under it, the hook is
# an int comparison.
_MIN_FILLED: int = 1 << 62


def register_lazy_safe(cls: type) -> type:
    """Mark a RotaryEmbedding subclass as eligible for the lazy cache."""
    _LAZY_SAFE_CLASSES.add(cls)
    return cls


def lazy_cache_enabled() -> bool:
    return bool(envs.SGLANG_ROPE_LAZY_CACHE.get())


def lazy_chunk_rows() -> int:
    return max(1, int(envs.SGLANG_ROPE_LAZY_CHUNK_ROWS.get()))


def lazy_min_rows() -> int:
    return int(envs.SGLANG_ROPE_LAZY_MIN_ROWS.get())


class ManagedRowBuffer:
    """A cos/sin cache buffer whose pages are committed on write.

    Owns a ``cuMemAllocManaged`` reservation and hands out a torch view of it.
    The view is built through ``__cuda_array_interface__`` so the tensor shares
    the reservation rather than copying it -- ``tensor.data_ptr()`` is the
    reservation address and stays constant for the life of this object.
    """

    def __init__(self, rows: int, cols: int, device: torch.device):
        if device.type != "cuda":
            raise RuntimeError(
                f"managed cos/sin buffer needs a CUDA device, got {device}"
            )
        # A context must exist before the driver call, and it must be the
        # context of the device this buffer will be read from.
        torch.cuda.init()
        nbytes = int(rows) * int(cols) * 4  # float32 only, see install()
        self._lib = ctypes.CDLL("libcuda.so.1")
        ptr = ctypes.c_void_p()
        with torch.cuda.device(device):
            rc = self._lib.cuMemAllocManaged(
                ctypes.byref(ptr),
                ctypes.c_size_t(nbytes),
                ctypes.c_uint(CU_MEM_ATTACH_GLOBAL),
            )
        if rc != 0 or not ptr.value:
            raise RuntimeError(f"cuMemAllocManaged({nbytes} B) failed with rc={rc}")
        self._ptr = int(ptr.value)
        self._nbytes = nbytes

        class _Interface:
            __cuda_array_interface__ = {
                "shape": (int(rows), int(cols)),
                "typestr": "<f4",
                "data": (self._ptr, False),
                "version": 3,
                "strides": None,
            }

        self.tensor = torch.as_tensor(_Interface(), device=device)
        if self.tensor.data_ptr() != self._ptr:
            raise RuntimeError("torch did not wrap the managed reservation in place")

    @property
    def ptr(self) -> int:
        return self._ptr

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __del__(self):
        try:
            if getattr(self, "_ptr", 0):
                self._lib.cuMemFree(ctypes.c_void_p(self._ptr))
                self._ptr = 0
        except Exception:  # interpreter teardown
            pass


class LazyCosSinState:
    """Bookkeeping for one installed lazy cache."""

    def __init__(self, capacity: int, filled: int, ptr: int, backing):
        self.capacity = int(capacity)
        self.filled = int(filled)
        self.ptr = int(ptr)
        self.backing = backing  # ManagedRowBuffer, or a plain tensor (fallback)
        self.managed = isinstance(backing, ManagedRowBuffer)


def _refresh_min_filled() -> None:
    global _MIN_FILLED
    smallest = 1 << 62
    for module in list(_LAZY_MODULES):
        state = getattr(module, "_lazy_cos_sin", None)
        if state is not None:
            smallest = min(smallest, state.filled)
    _MIN_FILLED = smallest


def install(
    module: "RotaryEmbedding",
    capacity_rows: int,
    cols: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Reserve ``capacity_rows`` and fill the first chunk. None means no.

    Returning None is never a failure of the caller's request: it means this
    layer keeps the eager cache it would have had anyway, so the decision is
    always safe to ignore.
    """
    if not lazy_cache_enabled():
        return None
    if type(module) not in _LAZY_SAFE_CLASSES:
        logger.info(
            "RoPE lazy cache declined for %s: growth hooks not verified for this class",
            type(module).__name__,
        )
        return None
    if capacity_rows < lazy_min_rows():
        return None
    if dtype != torch.float32:
        # The managed view is typed through __cuda_array_interface__, which has
        # no bf16 typestr, and a non-fp32 cache is the CPU/npu path anyway.
        logger.info("RoPE lazy cache declined: cache dtype %s is not float32", dtype)
        return None

    return reserve(module, capacity_rows, cols, device, dtype)


def reserve(
    module: "RotaryEmbedding",
    capacity_rows: int,
    cols: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Take (or retake) the reservation. Eligibility is the caller's business.

    Split from install() because ENLARGING a reservation must not re-ask
    whether this layer was ever allowed to be lazy: it already is, and a
    re-check that reads an env would silently drop a live lazy cache back to
    eager on the enlarge path.
    """
    filled = min(capacity_rows, lazy_chunk_rows())
    backing = None
    if device.type == "cuda":
        try:
            backing = ManagedRowBuffer(capacity_rows, cols, device)
            cache = backing.tensor
        except Exception as exc:  # driver refused, no UVM, ...
            logger.warning("RoPE lazy cache declined: %s", exc)
            return None
    else:
        # Correctness-identical fallback for CPU tests. It saves NOTHING --
        # the whole capacity is allocated -- and exists so the lazy code path
        # is exercisable without a GPU.
        cache = torch.empty((capacity_rows, cols), dtype=dtype, device=device)
        backing = cache

    cache[:filled] = module._build_cos_sin_rows(0, filled, device=device, dtype=dtype)
    module._lazy_cos_sin = LazyCosSinState(
        capacity_rows, filled, cache.data_ptr(), backing
    )
    _LAZY_MODULES.add(module)
    _refresh_min_filled()
    logger.info(
        "RoPE lazy cos/sin cache: reserved %d rows (%.1f MiB %s), filled %d, "
        "the rest is committed only when positions reach it",
        capacity_rows,
        capacity_rows * cols * 4 / 2**20,
        "managed" if isinstance(backing, ManagedRowBuffer) else "eager-fallback",
        filled,
    )
    return cache


def note_growth(module: "RotaryEmbedding") -> None:
    _refresh_min_filled()


def drop(module: "RotaryEmbedding") -> None:
    """Forget a cache that stopped being lazy (see base._relinquish_lazy)."""
    _LAZY_MODULES.discard(module)
    _refresh_min_filled()


def any_installed() -> bool:
    """True only when some layer actually took the lazy path."""
    return bool(_LAZY_MODULES)


def verify_positions_are_filled(max_position: int, where: str = "") -> None:
    """Raise if a position would read reservation that was never written.

    The failure this guards is silent by nature -- unwritten managed memory
    reads as whatever was there, not as an error -- so the guard has to be
    explicit, and it has to be able to fail. Costs a device sync at the call
    site, so it is behind SGLANG_ROPE_LAZY_VERIFY.
    """
    for module in list(_LAZY_MODULES):
        state = getattr(module, "_lazy_cos_sin", None)
        if state is not None and max_position >= state.filled:
            raise AssertionError(
                f"RoPE lazy cache read past the fill at {where}: position "
                f"{max_position} >= filled {state.filled} (capacity {state.capacity})"
            )


def ensure_capacity_for_position(max_position: int) -> None:
    """Grow every installed lazy cache so ``max_position`` is filled.

    Called once per batch from the model runner with a host-side position, so
    it costs an int comparison on the overwhelming majority of batches and
    never synchronizes with the device.
    """
    if max_position < _MIN_FILLED:
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        # Never write into a cache while a graph is being recorded: the fill
        # would be baked into the graph and replayed forever. Capture runs on
        # dummy positions well inside the initial chunk, so there is nothing
        # to grow for.
        return
    for module in list(_LAZY_MODULES):
        module._ensure_cos_sin_cache_length(max_position)
