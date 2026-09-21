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

import mmap as _mmap
import os
import weakref
from typing import Optional, Tuple

import torch

_CUDA_HOST_REGISTER_MAPPED = 2


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


def _page_align(ranges, nbytes: int):
    """Byte-Bereiche auf Seitengrenzen runden und verschmelzen.

    `cudaHostRegister` verlangt seitenausgerichtete Adressen. NACH AUSSEN
    runden ist die sichere Richtung: ein paar Byte zu viel zu pinnen kostet
    eine Seite, ein paar zu wenig laesst die GPU auf ungepinnte Seiten
    zeigen. Ueberlappende oder benachbarte Bereiche werden verschmolzen,
    damit aus 450 kalten Zeilen nicht 450 Registrierungen werden.
    """
    page = 4096
    out = []
    for lo, hi in sorted((int(a), int(b)) for a, b in ranges if int(b) > int(a)):
        lo = max(0, (lo // page) * page)
        hi = min(int(nbytes), -(-hi // page) * page)
        if hi <= lo:
            continue
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def shared_pinned_empty(
    path: str, shape, dtype: torch.dtype, register: Optional[bool] = None,
    pin_ranges=None,
):
    """A CPU tensor of ``shape``/``dtype`` backed by the MAP_SHARED tmpfs file
    ``path``, page-locked in this process when CUDA is available (or when
    ``register`` is True). Returns (tensor, created).

    ``pin_ranges`` (Byte-Paare) registriert NUR DIESE BEREICHE statt der
    ganzen Datei -- der Hebel gegen den Host-RAM, und zwar der einzige, der
    ohne Umnummerierung auskommt.

    WARUM DAS UEBERHAUPT ETWAS SPART (gemessen fnFL2w11, 21.09.): die Datei
    ist nominal 58,01 GiB gross und belegt 58,01 GiB -- keine einzige
    Nullseite, obwohl nur die KALTEN Zeilen je geschrieben werden
    (`spill_ids` schliesst die Residenten aus, expert_offload.py:1529).
    Der Grund ist genau diese Registrierung: `cudaHostRegister` ueber die
    ganze Datei faultet JEDE Seite ein, auch die, in die nie jemand
    schreibt. Die residenten Experten haben damit eine Host-Kopie, die
    niemand schreibt und niemand liest -- sie existiert nur, weil das
    Pinning sie anfasst. Bei P (0.12/0.3/0.3 auf 171/171/170) sind das 123
    von 512 Zeilen = 24 % der Datei.

    Die Datei behaelt ihre VOLLE Groesse, damit die globale Experten-Id der
    Zeilenindex bleibt und beide Ranggruppen dieselbe Adresse rechnen. Nur
    der Inhalt schrumpft. Das ist der Unterschied zur Umnummerierung
    (#72-Slot-Pool), die eine gemeinsame Residenzmenge beider Gruppen
    gebraucht haette und deshalb am Metall 0 GiB brachte."""
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
    spans = _page_align(pin_ranges, nbytes) if pin_ranges else [(0, nbytes)]
    pinned_ptrs = []
    if do_register:
        for lo, hi in spans:
            err = torch.cuda.cudart().cudaHostRegister(
                ptr + lo, hi - lo, _CUDA_HOST_REGISTER_MAPPED
            )
            if int(err) != 0:
                for q in pinned_ptrs:
                    try:
                        torch.cuda.cudart().cudaHostUnregister(q)
                    except Exception:  # noqa: BLE001
                        pass
                mm.close()
                raise RuntimeError(
                    f"cudaHostRegister({hi - lo} bytes at +{lo} of {path}) "
                    f"failed with cudaError {int(err)}"
                )
            pinned_ptrs.append(ptr + lo)

    def _release(m=mm, ps=tuple(pinned_ptrs)):
        for q in ps:
            try:
                torch.cuda.cudart().cudaHostUnregister(q)
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
