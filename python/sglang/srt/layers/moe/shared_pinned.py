"""Shared (cross-process) pinned host buffers for the expert host store.

Task #47 (19.09.): a P/D flip for Next Flash needs the expert host store
ONCE in RAM (61 GB for 512 x 48 experts) while two layouts -- P as PP3
stages, D as uneven TP3 shards -- read it from different processes. Today's
``pinned_exact_empty`` maps ``MAP_PRIVATE|MAP_ANONYMOUS`` per process, so a
second group would need a second copy (RAM mark 88 GiB).

``shared_pinned_empty(path, shape, dtype)`` maps a tmpfs file
(``/dev/shm/...``) ``MAP_SHARED``, sized exactly, and page-locks the mapping
in THIS process with ``cudaHostRegister`` (tmpfs pages are anonymous shared
memory, which the driver pins like any other). Every process that maps the
same path sees the same bytes; the first opener creates and sizes the file,
later openers attach. Nothing here decides who WRITES which rows -- that is
the loader's protocol (each rank writes the expert ids it owns; overlapping
writers write identical bytes).

Lifetime: the mapping and the registration are released when the returned
tensor's map object is garbage-collected (weakref.finalize); the FILE is
left in place on purpose -- the other group still needs it -- and is removed
by ``unlink_store(dir)`` at teardown by whoever owns the epoch.
"""
from __future__ import annotations

import logging
import mmap as _mmap
import os
import time
import weakref
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_CUDA_HOST_REGISTER_MAPPED = 2

#: PIN-REGISTER (30.09., NF y3z/y4a): the per-file registration clock. A slow
#: D load was 6-8 layers whose presplit took 4-17 s instead of 0.3 s (the rest
#: normal), inside this call -- the LOAD-PROFILE could only say "shared_pinned
#: 71 %", not WHICH file waited how long. Every call that registers logs one
#: line up to ``_REG_LOG_CAP`` per process; a call at or above
#: ``_REG_SLOW_MS`` is always logged (the stall a later reader correlates with
#: the host's compaction counters in memts' vmstat companion).
REG_MARK = "WEG2-PIN-REGISTER"
_REG_LOG_CAP = 512
_REG_SLOW_MS = 1000.0
_REG_N = {"n": 0}


class Weg2StoreMlockRefused(RuntimeError):
    """STORE-MLOCK: ``mlock`` of a store mapping failed -- named, never swallowed."""


def _libc_mlock(ptr: int, nbytes: int) -> int:
    """``mlock(2)`` on the mapping; returns 0 or the errno (a seam for the desk)."""
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mlock.restype = ctypes.c_int
    rc = int(libc.mlock(ctypes.c_void_p(int(ptr)), ctypes.c_size_t(int(nbytes))))
    return 0 if rc == 0 else (ctypes.get_errno() or -1)


def store_mlock_on() -> bool:
    """``SGLANG_WEG2_STORE_MLOCK`` (default off, see environ.py)."""
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_WEG2_STORE_MLOCK.get())


def _note_register(path: str, nbytes: int, created: bool, register_ms: float,
                   mlock_ms: Optional[float] = None) -> None:
    n = _REG_N["n"] + 1
    _REG_N["n"] = n
    if n > _REG_LOG_CAP and register_ms < _REG_SLOW_MS:
        return
    logger.info(
        "%s n=%d file=%s bytes=%d created=%s mlock_ms=%s register_ms=%.1f%s",
        REG_MARK, n, os.path.basename(path), int(nbytes), "yes" if created else "no",
        "-" if mlock_ms is None else "%.1f" % mlock_ms, register_ms,
        " SLOW" if register_ms >= _REG_SLOW_MS else "")


class SharedMap(_mmap.mmap):
    """A ``mmap.mmap`` subclass (plain mmaps cannot be weakly referenced)."""

    def __new__(cls, fd: int, nbytes: int):
        return super().__new__(cls, fd, nbytes, flags=_mmap.MAP_SHARED)


def _nbytes(shape: Tuple[int, ...], dtype: torch.dtype) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n * torch.empty((), dtype=dtype).element_size()


def open_shared_file(path: str, nbytes: int) -> Tuple[int, bool]:
    """Open-or-create ``path`` with exactly ``nbytes``; returns (fd, created).
    A file that already exists with a DIFFERENT size is refused loudly: two
    layouts disagreeing on a store's shape must never silently alias."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
    except FileExistsError:
        fd = os.open(path, os.O_RDWR)
        created = False
    size = os.fstat(fd).st_size
    if created or size == 0:
        os.ftruncate(fd, nbytes)
    elif size != nbytes:
        os.close(fd)
        raise ValueError(
            f"shared store {path} has {size} bytes, this layout wants {nbytes}"
        )
    return fd, created


def shared_pinned_empty(
    path: str, shape, dtype: torch.dtype, register: Optional[bool] = None
):
    """A CPU tensor of ``shape``/``dtype`` backed by the MAP_SHARED tmpfs file
    ``path``, page-locked in this process when CUDA is available (or when
    ``register`` is True). Returns (tensor, created)."""
    shape = tuple(int(d) for d in shape)
    nbytes = _nbytes(shape, dtype)
    if nbytes == 0:
        return torch.empty(shape, dtype=dtype, device="cpu"), False
    fd, created = open_shared_file(path, nbytes)
    try:
        mm = SharedMap(fd, nbytes)
    finally:
        os.close(fd)  # the mapping keeps the pages; the fd is not needed
    flat = torch.frombuffer(mm, dtype=torch.uint8)
    ptr = flat.data_ptr()
    do_register = torch.cuda.is_available() if register is None else register
    registered = False
    mlock_ms = None
    if store_mlock_on():
        # STORE-MLOCK (30.09., NF y3z/y4a): the store's 4K shmem pages, pinned
        # below, stay on the movable LRU; the host's direct compaction isolates
        # them (the pinned-page precheck in mm/compaction.c covers anonymous
        # pages only) and fails to migrate them over and over -- measured
        # 03:19Z ~313k failed migrations/s, 16 direct compactions/s at 99 %
        # failure -- and the registration of the next file stalls 4-17 s
        # behind it. mlock moves the pages to the unevictable LRU, which the
        # compaction skips once the host sets vm.compact_unevictable_allowed=0
        # (host config, not this process). BEFORE the registration: the pin
        # then finds every page already resident and locked.
        t_ml = time.perf_counter()
        rc = _libc_mlock(ptr, nbytes)
        mlock_ms = (time.perf_counter() - t_ml) * 1000.0
        if rc != 0:
            mm.close()
            raise Weg2StoreMlockRefused(
                f"WEG2-STORE-MLOCK REFUSED: mlock({nbytes} bytes of {path}) failed with "
                f"errno {rc} ({os.strerror(rc) if rc > 0 else 'unknown'}) -- "
                "SGLANG_WEG2_STORE_MLOCK=1 asks for every store page locked; check "
                "RLIMIT_MEMLOCK (docker --ulimit memlock=-1) or turn the switch off")
    if do_register:
        t_reg = time.perf_counter()
        err = torch.cuda.cudart().cudaHostRegister(ptr, nbytes, _CUDA_HOST_REGISTER_MAPPED)
        if int(err) != 0:
            mm.close()
            raise RuntimeError(
                f"cudaHostRegister({nbytes} bytes of {path}) failed with cudaError {int(err)}"
            )
        registered = True
        _note_register(path, nbytes, created, (time.perf_counter() - t_reg) * 1000.0,
                       mlock_ms)

    def _release(m=mm, p=ptr, reg=registered):
        if reg:
            try:
                torch.cuda.cudart().cudaHostUnregister(p)
            except Exception:  # noqa: BLE001 -- teardown never raises
                pass
        try:
            m.close()
        except Exception:  # noqa: BLE001
            pass

    weakref.finalize(mm, _release)
    t = flat.view(dtype).view(shape)
    t._shared_map = mm  # keeps the map alive as long as the tensor lives
    return t, created


def unlink_store(directory: str) -> int:
    """Remove every file of a store directory (epoch teardown). Returns the count."""
    n = 0
    if not os.path.isdir(directory):
        return 0
    for name in os.listdir(directory):
        try:
            os.unlink(os.path.join(directory, name))
            n += 1
        except FileNotFoundError:
            pass
    try:
        os.rmdir(directory)
    except OSError:
        pass
    return n
