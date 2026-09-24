"""Sleep-Residuum H15: the local-memory reservation of a sleeping rank's context.

WHAT THIS POST IS. The CUDA driver backs every thread's stack (local arrays,
register spills, call frames) with ONE device allocation per context, sized

    per-thread stack bytes x MULTIPROCESSOR_COUNT x MAX_THREADS_PER_MULTIPROCESSOR

and grows it when a launch needs more, but never shrinks it by itself
(``CU_CTX_LMEM_RESIZE_TO_MAX`` is the default and "cannot be disabled"; the
documented way down is ``cuCtxSetLimit(CU_LIMIT_STACK_SIZE, smaller)``). It is
not in torch's caching allocator, so it sits inside the ``other
(context+driver+communicator+non-torch)`` post of WEG2-DC-BREAKDOWN and in
``outside_torch`` of WEG2-SLEEP-RESIDUE, and no tag pause touches it.

SIZE, derived (the NVML delta on the log line is the measurement): the driver
default is 1024 B per thread, and no sm_120 kernel this stack runs asks for
more (desk survey 24.09. with ``cuobjdump -res-usage``: sgl_kernel max 512 B,
flashinfer JIT cache max 664 B, the Triton cache of the last 20 days max
1000 B, libtorch_cuda above 1 KiB only in the IGamma kernels). Hence

    RTX 5090: 1024 B x 170 SM x 1536 threads = 255 MiB per process
    RTX 3080: 1024 B x  68 SM x 1536 threads = 102 MiB per process

and on the 5090 both phases hold one: PP0 while D decodes, TP0 while P
prefills.

THE PARK. At the end of a complete sleep the limit goes to the lowest rung of
:data:`PARK_STACK_LADDER` the driver accepts; the driver frees the backing
allocation. A kernel the dormant process still launches (a barlink collective
of the dormant hold, a lane copy) grows it back to exactly what that kernel
needs -- that is the driver's normal path, not an error.

THE RESTORE. At the wake, before the kv_cache fit check reads the card, the
limit goes back to the value saved at the park, so the woken rank runs its
first graph replay with the reservation it had before the sleep. The cost is
one ``cuCtxSetLimit`` (a device-idle wait plus one allocation), printed in ms.

Everything here is fail-soft: a refused park leaves the context as it was, a
refused restore leaves the driver's on-demand growth, and both say so on the
log line. Nothing here raises into the sleep or the wake.
"""

from __future__ import annotations

import ctypes
import time
from typing import Callable, Optional, Protocol, Sequence, Tuple

import msgspec

#: ``CUlimit`` value of ``CU_LIMIT_STACK_SIZE`` (cuda.h).
CU_LIMIT_STACK_SIZE = 0x00
#: Rungs tried at the park, lowest first; the first one the driver accepts wins.
PARK_STACK_LADDER: Tuple[int, ...] = (0, 16, 256)
_MIB = 1 << 20

NvmlReader = Callable[[], Optional[int]]


class StackLimitDriver(Protocol):
    """The two driver calls the park needs, on the calling thread's context."""

    def get_stack_bytes(self) -> int: ...

    def set_stack_bytes(self, value: int) -> None: ...


class CudaDriverStackLimit:
    """``cuCtxGetLimit`` / ``cuCtxSetLimit(CU_LIMIT_STACK_SIZE)`` from libcuda.so.1.

    Acts on the CURRENT context of the calling thread -- the primary context
    torch made current on the scheduler thread. No current context is a
    ``CUDA_ERROR_INVALID_CONTEXT`` from the driver, which raises here and is
    reported as a refused park, never as a zero.
    """

    def __init__(self, lib=None) -> None:
        self._lib = lib if lib is not None else ctypes.CDLL("libcuda.so.1")

    def get_stack_bytes(self) -> int:
        value = ctypes.c_size_t(0)
        rc = self._lib.cuCtxGetLimit(ctypes.byref(value), ctypes.c_int(CU_LIMIT_STACK_SIZE))
        _check(rc, "cuCtxGetLimit(STACK_SIZE)")
        return int(value.value)

    def set_stack_bytes(self, value: int) -> None:
        rc = self._lib.cuCtxSetLimit(ctypes.c_int(CU_LIMIT_STACK_SIZE), ctypes.c_size_t(int(value)))
        _check(rc, f"cuCtxSetLimit(STACK_SIZE, {int(value)})")


def _check(rc: int, what: str) -> None:
    if int(rc) != 0:
        raise RuntimeError(f"{what} returned CUresult {int(rc)}")


def lmem_mib(*, stack_bytes: int, threads: int) -> float:
    """The driver's reservation for a per-thread stack (derived, not read)."""
    return max(0, int(stack_bytes)) * max(0, int(threads)) / _MIB


def _delta_mib(before: Optional[int], after: Optional[int]) -> Optional[float]:
    if before is None or after is None:
        return None
    return (int(before) - int(after)) / _MIB


class LmemPark(msgspec.Struct, frozen=True, kw_only=True):
    """What the park did. ``refused`` empty = the limit was lowered."""

    saved_stack_bytes: int
    parked_stack_bytes: int
    threads: int
    nvml_before: Optional[int] = None
    nvml_after: Optional[int] = None
    ms: float = 0.0
    refused: str = ""

    def released_nvml_mib(self) -> Optional[float]:
        return _delta_mib(self.nvml_before, self.nvml_after)

    def format_post(self) -> str:
        before = lmem_mib(stack_bytes=self.saved_stack_bytes, threads=self.threads)
        if self.refused:
            return f"lmem {before:.0f}->{before:.0f} MiB KEPT ({self.refused})"
        after = lmem_mib(stack_bytes=self.parked_stack_bytes, threads=self.threads)
        nv = self.released_nvml_mib()
        return (
            f"lmem {before:.0f}->{after:.0f} MiB released (stack {self.saved_stack_bytes}->"
            f"{self.parked_stack_bytes} B x {self.threads} threads, derived; NVML "
            f"{'n/a' if nv is None else f'-{nv:.0f}'} MiB measured; {self.ms:.1f} ms)"
        )


class LmemRestore(msgspec.Struct, frozen=True, kw_only=True):
    """What the wake did. ``found`` above ``parked`` = regrown while dormant."""

    found_stack_bytes: int
    restored_stack_bytes: int
    threads: int
    nvml_before: Optional[int] = None
    nvml_after: Optional[int] = None
    ms: float = 0.0
    refused: str = ""

    def format_line(self, *, park: LmemPark) -> str:
        grown = self.found_stack_bytes - park.parked_stack_bytes
        nv = _delta_mib(self.nvml_after, self.nvml_before)
        head = (
            f"WEG2-WAKE-LMEM stack {self.found_stack_bytes}->{self.restored_stack_bytes} B "
            f"(parked {park.parked_stack_bytes}, saved {park.saved_stack_bytes}; "
            f"regrown_while_dormant={max(0, grown)} B) lmem "
            f"{lmem_mib(stack_bytes=self.found_stack_bytes, threads=self.threads):.0f}->"
            f"{lmem_mib(stack_bytes=self.restored_stack_bytes, threads=self.threads):.0f} MiB "
            f"NVML {'n/a' if nv is None else f'+{nv:.0f}'} MiB ms={self.ms:.1f}"
        )
        if self.refused:
            return head + f" REFUSED ({self.refused}) -- the driver grows it on demand at the next launch"
        return head


def park_lmem(
    *,
    driver: StackLimitDriver,
    threads: int,
    nvml_bytes: NvmlReader,
    ladder: Sequence[int] = PARK_STACK_LADDER,
) -> LmemPark:
    """Lower the stack limit to the first rung the driver accepts. Never raises."""
    t0 = time.perf_counter()
    try:
        saved = driver.get_stack_bytes()
    except Exception as exc:  # noqa: BLE001 -- a refused park leaves the context as it was
        return LmemPark(saved_stack_bytes=0, parked_stack_bytes=0, threads=threads,
                        refused=f"get: {type(exc).__name__}: {exc}")
    before = nvml_bytes()
    parked, errors = _first_accepted_rung(driver=driver, saved=saved, ladder=ladder)
    after = nvml_bytes()
    ms = (time.perf_counter() - t0) * 1000.0
    if parked is None:
        why = "; ".join(errors) if errors else f"no rung below the saved {saved} B"
        return LmemPark(saved_stack_bytes=saved, parked_stack_bytes=saved, threads=threads,
                        nvml_before=before, nvml_after=after, ms=ms, refused=why)
    return LmemPark(saved_stack_bytes=saved, parked_stack_bytes=parked, threads=threads,
                    nvml_before=before, nvml_after=after, ms=ms)


def _first_accepted_rung(
    *, driver: StackLimitDriver, saved: int, ladder: Sequence[int]
) -> Tuple[Optional[int], list]:
    errors: list = []
    for rung in ladder:
        if int(rung) >= int(saved):
            break
        try:
            driver.set_stack_bytes(int(rung))
            return driver.get_stack_bytes(), errors
        except Exception as exc:  # noqa: BLE001 -- try the next rung
            errors.append(f"{rung} B: {type(exc).__name__}: {exc}")
    return None, errors


def restore_lmem(
    *, driver: StackLimitDriver, park: LmemPark, nvml_bytes: NvmlReader
) -> LmemRestore:
    """Put the saved limit back unless the context already holds at least that."""
    t0 = time.perf_counter()
    try:
        found = driver.get_stack_bytes()
    except Exception as exc:  # noqa: BLE001
        return LmemRestore(found_stack_bytes=park.parked_stack_bytes,
                           restored_stack_bytes=park.parked_stack_bytes, threads=park.threads,
                           refused=f"get: {type(exc).__name__}: {exc}")
    if found >= park.saved_stack_bytes:
        return LmemRestore(found_stack_bytes=found, restored_stack_bytes=found,
                           threads=park.threads, ms=(time.perf_counter() - t0) * 1000.0)
    before = nvml_bytes()
    try:
        driver.set_stack_bytes(park.saved_stack_bytes)
        restored = driver.get_stack_bytes()
        refused = ""
    except Exception as exc:  # noqa: BLE001
        restored, refused = found, f"set: {type(exc).__name__}: {exc}"
    return LmemRestore(found_stack_bytes=found, restored_stack_bytes=restored,
                       threads=park.threads, nvml_before=before, nvml_after=nvml_bytes(),
                       ms=(time.perf_counter() - t0) * 1000.0, refused=refused)


def format_residue_posts(
    *,
    park: Optional[LmemPark],
    group_windows: Sequence[Tuple[str, int]],
    lane_windows: Sequence[Tuple[str, int]],
) -> str:
    """The per-post split WEG2-SLEEP-RESIDUE carries: MiB before->after the sleep.

    ``group_windows`` = the barlink BAR1 ledger of this card (payload+flags per
    group, device VMM memory); ``lane_windows`` = this rank's own receiver
    windows of the flip lanes (borrowed ones carry no VRAM of their own and are
    left out by the caller). Both stay: the dormant rank's fence and hold run
    collectives over the group windows, and the next flip's depositor writes
    into the lane windows while this rank is still asleep.
    """
    parts = ["lmem n/a (park off)" if park is None else park.format_post()]
    gw = sum(int(b) for _, b in group_windows) / _MIB
    parts.append(
        f"barlink_group_windows {gw:.0f}->{gw:.0f} MiB KEPT "
        f"({', '.join(f'{g}:{int(b) / _MIB:.0f}' for g, b in group_windows) or '-'})"
    )
    lw = sum(int(b) for _, b in lane_windows) / _MIB
    parts.append(
        f"bar1_lane_windows {lw:.0f}->{lw:.0f} MiB KEPT "
        f"({', '.join(f'{k}:{int(b) / _MIB:.0f}' for k, b in lane_windows) or '-'})"
    )
    return "posts=[" + " | ".join(parts) + "]"
