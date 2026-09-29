"""ONE HiCache for all six ranks: the shared-memory page arena (arena.c).

BAUPLAN_SHM_ARENA_0916 (user order 2026-09-16: "ich moechte EINEN L2 haben ...
jeder rank schreibt und liest in den gleichen hicache"). One file per slot
class under /dev/shm, mapped by every rank process of both groups; a canonical
page is written into it by the ranks that own its extents (PP stages, TP
shards) and read out of it by any rank that needs its own extents. The disk
store (HiCacheFile's directory) stays the cold tier behind it: a miss reads
from disk into the arena, eviction writes complete pages to disk first.

This module is the ctypes/mmap side; the protocol lives in arena.c. Nothing
here needs CUDA: the arena is a plain shared mapping until stage 3 (direct
DMA) pins it with cudaHostRegister.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import logging
import mmap
import os
import subprocess
import tempfile
import threading
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arena.c")
_lock = threading.Lock()
_lib: Optional[ctypes.CDLL] = None
_failed = False

#: every extent boundary must sit on this granule (arena.c GRANULE)
GRANULE = 1  # coverage is an interval list; any byte range counts

#: 27B 24.09. (arena reader refs out of the release queue).
#: "1": (a) every ShmArena of this process keeps a LEDGER of the reader
#: references this process holds per slot, and a release (-1) never takes more
#: than that -- a rank can never drop another rank's reference, nor drop one
#: it does not hold; (b) HostPoolGroup.free hands arena ids from the host
#: release queue back to the arena (one reference per row) instead of dropping
#: them as #718 strays. See HostPoolGroup._free_arena_rows.
#:
#: HX (NF, 26.09.): DEFAULT ON. The NF-RC2 form (b89592806a, x178 + f3, and
#: every docker profile since: nf.env NF_ENV_RC2_FORM) sets it to 1; the native
#: H91 arms descend from the pre-RC2 x177 arm and never carried it, so h91v1,
#: h91bb1 and h91bb2 ran the leaking default again ("HICACHE-INDEX REFUSED" 1
#: line per group, 0 "ARENA-QUEUE-REFS" lines; x178: the reverse). Off, every
#: resolved-but-unadopted arena page on the queue keeps its reader reference
#: forever (docker agent boots measured up to 1528 such pages per boot on D,
#: a 5461-slot arena) -- the pin that ends in ARENA-CLAIM REFUSED. A form that
#: exists only in an arm's env is lost with the arm; "0" still opts out.
ENV_QUEUE_REFS = "SGLANG_HICACHE_ARENA_QUEUE_REFS"


def arena_queue_refs_on() -> bool:
    return os.environ.get(ENV_QUEUE_REFS, "1").strip().lower() in ("1", "true", "yes", "on")


class RefLedger:
    """The reader references THIS process holds, per slot of one arena file.

    The C refcount is shared by every rank process mapping the file; it cannot
    say whose a reference is. This count can: it is raised by every +1 this
    process takes through ShmArena and consumed by every -1, and a -1 beyond
    it is refused (counted in `refused`). One ledger per file and process --
    two ShmArena objects on the same path share it."""

    def __init__(self, slots: int):
        import torch

        self.held = torch.zeros(int(slots), dtype=torch.int32)
        self.lock = threading.Lock()
        self.refused = 0

    def took(self, slots) -> None:
        import torch

        t = torch.as_tensor(slots, dtype=torch.int64).reshape(-1)
        t = t[t >= 0]
        if t.numel():
            with self.lock:
                self.held.index_add_(0, t, torch.ones(t.numel(), dtype=torch.int32))

    def allow_release(self, slots):
        """The multiset of `slots` this process may release (each at most as
        often as it holds a reference there); the ledger is debited for it."""
        import torch

        t = torch.as_tensor(slots, dtype=torch.int64).reshape(-1)
        t = t[t >= 0]
        if t.numel() == 0:
            return t
        with self.lock:
            uq, cnt = torch.unique(t, return_counts=True)
            have = self.held[uq]
            allow = torch.minimum(cnt.to(torch.int32), have)
            self.held[uq] = have - allow
            self.refused += int((cnt - allow).sum())
        return torch.repeat_interleave(uq, allow.to(torch.int64))


_LEDGERS: dict = {}

#: Instrument (default 0 = off): every N seconds a daemon thread logs the
#: arena's reference census (ShmArena.ref_census) -- how many COMPLETE slots a
#: reader reference pins, over the whole boot. Off the round path: the strided
#: header read of a 720k-slot arena is ~15 ms (desk), numpy releases the GIL.
ENV_REF_CENSUS_S = "SGLANG_HICACHE_ARENA_REF_CENSUS_S"
_CENSUS_THREADS: dict = {}
#: #1424e: the reference HOLDERS of this process, per class, next to the census
#: (weak methods: a provider never keeps its tree alive). A provider is called
#: with the arena and returns one line of holder classes, or None when it holds
#: nothing on that arena.
_HOLDER_PROVIDERS: list = []


def register_holder_census(fn) -> None:
    """#1424e: add a holder provider (a bound method, held weakly) to the
    ARENA-REF-CENSUS thread -- one ARENA-REF-HOLDERS line per provider and
    census tick, never on a round path."""
    import weakref
    ref = weakref.WeakMethod(fn) if hasattr(fn, "__self__") else (lambda f=fn: f)
    with _lock:
        _HOLDER_PROVIDERS[:] = [r for r in _HOLDER_PROVIDERS if r() is not None]
        _HOLDER_PROVIDERS.append(ref)


def _holder_lines(arena) -> list:
    with _lock:
        fns = [r() for r in _HOLDER_PROVIDERS]
    out = []
    for fn in fns:
        if fn is None:
            continue
        try:
            line = fn(arena)
        except Exception as exc:  # noqa: BLE001 - an instrument never raises
            line = f"failed={exc!r}"
        if line:
            out.append(line)
    return out


def _start_ref_census(arena: "ShmArena") -> None:
    try:
        every = float(os.environ.get(ENV_REF_CENSUS_S, "0") or 0)
    except ValueError:
        every = 0.0
    if every <= 0:
        return
    key = os.path.realpath(arena.path)
    with _lock:
        if key in _CENSUS_THREADS:
            return
        stop = threading.Event()

        def _run():
            n = 0
            while not stop.wait(every):
                n += 1
                try:
                    pinned, refs, complete = arena.ref_census()
                    led = arena._ledger
                    logger.info(
                        "ARENA-REF-CENSUS n=%d path=%s slots=%d complete=%d pinned=%d refs=%d "
                        "own_held=%s own_refused=%s (pinned = COMPLETE slots a reader "
                        "reference keeps from eviction, all ranks; own_* = this process's "
                        "ledger, SGLANG_HICACHE_ARENA_QUEUE_REFS)",
                        n, arena.path, arena.slots, complete, pinned, refs,
                        int(led.held.sum()) if led is not None else "-",
                        led.refused if led is not None else "-",
                    )
                except Exception as exc:  # noqa: BLE001 - an instrument never raises
                    logger.info("ARENA-REF-CENSUS n=%d failed: %r", n, exc)
                for line in _holder_lines(arena):
                    logger.info("ARENA-REF-HOLDERS n=%d path=%s %s", n, arena.path, line)

        th = threading.Thread(target=_run, name="arena-ref-census", daemon=True)
        _CENSUS_THREADS[key] = stop
        th.start()


# #1427s: ARENA-FREE per reason -> [calls, slots] (process-wide, for the throttle)
_FREE_BY_REASON: dict = {}


def _note_free(reason: str, slots) -> None:
    """#1427s (operator order after z30p): EVERY slot free names its reason
    -- 'who frees the slot' took a boot to answer because ARENA-FREE only
    logged its first 16 calls with a stack. One line per reason, throttled
    per reason (the first 8 calls, then every power of two), with the
    reason's running call and slot totals; an ``unnamed`` caller still gets
    the xsn328 stack so the missing name can be found."""
    n = len(slots)
    cnt = _FREE_BY_REASON.setdefault(reason, [0, 0])
    cnt[0] += 1
    cnt[1] += n
    k = cnt[0]
    if k <= 8 or (k & (k - 1)) == 0:
        caller = ""
        if reason == "unnamed":
            import traceback as _tb
            caller = " caller=" + "".join(_tb.format_stack(limit=5)[:-2]).strip().replace("\n", " | ")[-300:]
        logger.info("ARENA-FREE reason=%s slots=%d first=%s calls=%d slots_total=%d%s",
                    reason, n, (list(slots)[:4] if n else []), k, cnt[1], caller)


def free_named(arena, slots, reason: str) -> None:
    """#1427s: ``arena.free_slots(slots, reason=reason)`` for every caller; a
    hermetic fake arena without the keyword frees as before."""
    try:
        arena.free_slots(slots, reason=reason)
    except TypeError:
        arena.free_slots(slots)


def _ledger_for(path: str, slots: int) -> RefLedger:
    key = (os.path.realpath(path), int(slots))
    with _lock:
        led = _LEDGERS.get(key)
        if led is None:
            led = _LEDGERS[key] = RefLedger(slots)
        return led


def _load_lib() -> Optional[ctypes.CDLL]:
    global _lib, _failed
    if _lib is not None:
        return _lib
    if _failed:
        return None
    with _lock:
        if _lib is not None:
            return _lib
        if _failed:
            return None
        try:
            with open(_SRC, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()[:16]
            so = os.path.join(
                tempfile.gettempdir(), f"sglang_hicache_arena_{digest}_{os.getuid()}.so"
            )
            if not os.path.exists(so):
                tmp = f"{so}.{os.getpid()}.tmp"
                subprocess.run(
                    ["gcc", "-O2", "-shared", "-fPIC", "-o", tmp, _SRC],
                    check=True, capture_output=True, timeout=120,
                )
                os.replace(tmp, so)
            lib = ctypes.CDLL(so)
            i64 = ctypes.c_int64
            p_i64 = ctypes.POINTER(i64)
            p_u64 = ctypes.POINTER(ctypes.c_uint64)
            p_i8 = ctypes.POINTER(ctypes.c_int8)
            p_u8 = ctypes.c_void_p
            lib.arena_layout.restype = i64
            lib.arena_layout.argtypes = [i64, i64, p_i64]
            lib.arena_init.restype = ctypes.c_int
            lib.arena_init.argtypes = [p_u8, i64, i64]
            lib.arena_write.restype = i64
            lib.arena_write.argtypes = [p_u8, i64, p_u64, p_u64, p_i64, p_i64, p_i64, p_i64,
                                        ctypes.POINTER(ctypes.c_void_p),
                                        ctypes.POINTER(ctypes.c_char_p), p_i8]
            lib.arena_slot_stem.restype = ctypes.c_char_p
            lib.arena_slot_stem.argtypes = [p_u8, i64]
            lib.arena_read.restype = i64
            lib.arena_read.argtypes = [p_u8, i64, p_u64, p_u64, p_i64, p_i64, p_i64, p_i64,
                                       ctypes.POINTER(ctypes.c_void_p), p_i8]
            lib.arena_lookup.restype = i64
            lib.arena_lookup.argtypes = [p_u8, i64, p_u64, p_u64, p_i8]
            lib.arena_evict_candidates.restype = i64
            lib.arena_evict_candidates.argtypes = [p_u8, i64, p_i64, p_u64, p_u64, p_i64, p_u64, i64]
            lib.arena_stem_keys.restype = None
            lib.arena_stem_keys.argtypes = [i64, ctypes.POINTER(ctypes.c_char_p), p_u64]
            lib.arena_slot_ptr.restype = ctypes.c_void_p
            lib.arena_slot_ptr.argtypes = [p_u8, i64]
            lib.arena_free_slots.restype = None
            lib.arena_free_slots.argtypes = [p_u8, i64, p_i64]
            lib.arena_drop_unreferenced.restype = i64
            lib.arena_drop_unreferenced.argtypes = [p_u8, i64, p_i64, p_i8]
            lib.arena_reap_stale.restype = i64
            lib.arena_reap_stale.argtypes = [p_u8]
            lib.arena_reap_partial.restype = i64
            lib.arena_reap_partial.argtypes = [p_u8, i64, p_i64, i64]
            lib.arena_unclaim.restype = i64
            lib.arena_unclaim.argtypes = [p_u8, i64, p_i64, p_i64]
            lib.arena_release_claims.restype = i64
            lib.arena_release_claims.argtypes = [p_u8, i64, p_i64, p_i64, p_i8]
            lib.arena_ival_cap.restype = i64
            lib.arena_ival_cap.argtypes = [p_u8]
            lib.arena_stats.restype = None
            lib.arena_stats.argtypes = [p_u8, p_i64]
            lib.arena_find_slots.restype = i64
            lib.arena_find_slots.argtypes = [p_u8, i64, p_u64, p_u64, p_i64, p_i8]
            lib.arena_find_stems.restype = i64
            lib.arena_find_stems.argtypes = [p_u8, i64, ctypes.POINTER(ctypes.c_char_p), p_i64, p_i8]
            lib.arena_claim.restype = i64
            lib.arena_claim.argtypes = [p_u8, i64, p_u64, p_u64, p_i64,
                                        ctypes.POINTER(ctypes.c_char_p), p_i64, p_i64, p_i8]
            lib.arena_lookup_stems.restype = i64
            lib.arena_lookup_stems.argtypes = [p_u8, i64, ctypes.POINTER(ctypes.c_char_p), p_i8]
            lib.arena_claim_stems.restype = i64
            lib.arena_claim_stems.argtypes = [p_u8, i64, ctypes.POINTER(ctypes.c_char_p), p_i64, p_i64, p_i64, p_i8]
            lib.arena_complete.restype = i64
            lib.arena_complete.argtypes = [p_u8, i64, p_i64, p_i64, p_i64, p_i64, p_i64, p_i8]
            lib.arena_ref_slots.restype = i64
            lib.arena_ref_slots.argtypes = [p_u8, i64, p_i64, ctypes.c_int32]
            lib.arena_data_offset.restype = i64
            lib.arena_data_offset.argtypes = [p_u8]
            lib.arena_complete_census.restype = i64
            lib.arena_complete_census.argtypes = [p_u8, i64, p_i64, p_i64, p_u64, p_u64]
            lib.arena_pin_complete.restype = i64
            lib.arena_pin_complete.argtypes = [p_u8, i64, p_i64, p_u64, p_u64, p_i8]
            lib.arena_ref_slots_mask.restype = i64
            lib.arena_ref_slots_mask.argtypes = [p_u8, i64, p_i64, p_i8]
            _lib = lib
            return lib
        except Exception as e:  # noqa: BLE001 - the arena is optional
            _failed = True
            logger.warning("[arena] helper unavailable (%s: %s); no shared arena.",
                           type(e).__name__, str(e)[:200])
            return None


@functools.lru_cache(maxsize=1 << 20)  # #1438: the same stems are hashed for find/ref/draft; cached
def key128(stem: str) -> tuple[int, int]:
    """The 128-bit key of a store stem; low word never 0 or ~0 (index sentinels)."""
    d = hashlib.blake2b(stem.encode("utf-8"), digest_size=16).digest()
    lo = int.from_bytes(d[:8], "little")
    hi = int.from_bytes(d[8:], "little")
    if lo in (0, (1 << 64) - 1):
        lo = 1
    return lo, hi


def stem_keys_lo(stems: Sequence[str]):
    """#243: key128(stem)[0] for every stem as a numpy uint64 array, hashed in
    C (arena.c arena_stem_keys); Python's key128 when the helper is missing."""
    import numpy as np
    n = len(stems)
    lib = _load_lib()
    if n == 0:
        return np.zeros(0, dtype=np.uint64)
    if lib is None or not hasattr(lib, "arena_stem_keys"):
        return np.fromiter((key128(s)[0] for s in stems), dtype=np.uint64, count=n)
    out = np.empty(n, dtype=np.uint64)
    c_stems = (ctypes.c_char_p * n)(*[s.encode("utf-8") for s in stems])
    lib.arena_stem_keys(n, c_stems, out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)))
    return out


class ShmArena:
    """One arena file: `slots` slots of `slot_bytes` each.

    `path` is created (sparse, sized by arena_layout) when absent; every
    process that opens the same path shares the same pages. Statuses follow
    arena.c: write 0 partial / 1 completed / 2 already / 3 refused / 4 full;
    read 0 ok / 1 absent / 2 width / 3 refused.
    """

    #: SGLANG_HICACHE_ARENA_QUEUE_REFS: this process's RefLedger (None = off)
    _ledger = None

    def __init__(self, path: str, slot_bytes: int, slots: int):
        lib = _load_lib()
        if lib is None:
            raise RuntimeError("arena helper unavailable")
        self._lib = lib
        self.path = path
        self.slot_bytes = int(slot_bytes)
        self.slots = int(slots)
        out = (ctypes.c_int64 * 6)()
        total = lib.arena_layout(self.slots, self.slot_bytes, out)
        self.file_bytes = int(total)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            st = os.fstat(fd)
            if st.st_size < self.file_bytes:
                os.ftruncate(fd, self.file_bytes)
            self._mm = mmap.mmap(fd, self.file_bytes, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)
        self._base = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(self._mm)))
        rc = lib.arena_init(self._base, self.slots, self.slot_bytes)
        if rc < 0:
            raise RuntimeError(
                f"{path} holds an arena of another geometry (not {slots} x {slot_bytes})"
            )
        self.fresh = rc == 0
        # SGLANG_HICACHE_ARENA_QUEUE_REFS: this process's reader references
        # (None = off, the unchanged path).
        self._ledger = _ledger_for(path, self.slots) if arena_queue_refs_on() else None
        _start_ref_census(self)

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _keys(stems: Sequence[str]):
        n = len(stems)
        lo = (ctypes.c_uint64 * n)()
        hi = (ctypes.c_uint64 * n)()
        for i, s in enumerate(stems):
            a, b = key128(s)
            lo[i] = a
            hi[i] = b
        return lo, hi

    @staticmethod
    def _extents(extents):
        n = len(extents)
        n_ext = (ctypes.c_int64 * n)(*[len(e) for e in extents])
        flat_off = [int(o) for e in extents for o, _ in e]
        flat_len = [int(l) for e in extents for _, l in e]
        m = max(1, len(flat_off))
        return n_ext, (ctypes.c_int64 * m)(*flat_off), (ctypes.c_int64 * m)(*flat_len)

    # -- protocol ----------------------------------------------------------
    def lookup(self, stems: Sequence[str]) -> list[bool]:
        n = len(stems)
        if n == 0:
            return []
        st = (ctypes.c_int8 * n)()
        try:  # xsn357: hashed in C (arena_lookup_stems); Python key128 was the prefetch thread's cost
            c_stems = (ctypes.c_char_p * n)(*[s.encode("utf-8") for s in stems])
            self._lib.arena_lookup_stems(self._base, n, c_stems, st)
        except Exception:  # noqa: BLE001 -- the key path stays as the fallback
            lo, hi = self._keys(stems)
            self._lib.arena_lookup(self._base, n, lo, hi, st)
        return [bool(x) for x in st]

    def write(self, stems, totals, extents, payload_ptrs) -> list[int]:
        n = len(stems)
        if n == 0:
            return []
        lo, hi = self._keys(stems)
        n_ext, c_off, c_len = self._extents(extents)
        c_tot = (ctypes.c_int64 * n)(*[int(t) for t in totals])
        c_pay = (ctypes.c_void_p * n)(*[int(p) for p in payload_ptrs])
        c_stems = (ctypes.c_char_p * n)(*[s.encode("utf-8") for s in stems])
        st = (ctypes.c_int8 * n)()
        self._lib.arena_write(self._base, n, lo, hi, c_tot, n_ext, c_off, c_len, c_pay, c_stems, st)
        return list(st)

    def read(self, stems, totals, extents, out_ptrs) -> list[int]:
        n = len(stems)
        if n == 0:
            return []
        lo, hi = self._keys(stems)
        n_ext, c_off, c_len = self._extents(extents)
        c_tot = (ctypes.c_int64 * n)(*[int(t) for t in totals])
        c_out = (ctypes.c_void_p * n)(*[int(p) for p in out_ptrs])
        st = (ctypes.c_int8 * n)()
        self._lib.arena_read(self._base, n, lo, hi, c_tot, n_ext, c_off, c_len, c_out, st)
        return list(st)

    def evict_candidates(self, want: int, keep_stems: Sequence[str] = (), keep_lo=None):
        """(slot, key_lo, key_hi, total) of up to `want` slots now EVICTING.
        Slots whose key is in `keep_stems` or `keep_lo` (the keys' low words,
        #243: a pending hand-off's pages) are passed over; arena.c searches the
        keep list by bisection, so it goes in sorted."""
        import numpy as np
        want = int(want)
        if want <= 0:
            return []
        slots = (ctypes.c_int64 * want)()
        lo = (ctypes.c_uint64 * want)()
        hi = (ctypes.c_uint64 * want)()
        tot = (ctypes.c_int64 * want)()
        if keep_stems or keep_lo is None:
            keep = np.fromiter((key128(s)[0] for s in keep_stems), dtype=np.uint64)
            if keep_lo is not None and len(keep_lo):
                keep = np.concatenate([keep, np.asarray(keep_lo, dtype=np.uint64)])
            keep = np.ascontiguousarray(np.sort(keep))
        else:
            # `keep_lo` alone comes sorted (handoff_pending.Keep.keys): no
            # per-claim sort of a thousands-long list
            keep = np.ascontiguousarray(keep_lo, dtype=np.uint64)
        n_keep = int(keep.shape[0])
        c_keep = (keep.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)) if n_keep
                  else (ctypes.c_uint64 * 1)())
        got = self._lib.arena_evict_candidates(self._base, want, slots, lo, hi, tot, c_keep, n_keep)
        return [(int(slots[i]), int(lo[i]), int(hi[i]), int(tot[i])) for i in range(got)]

    # -- Stufe 3 (#1424): address pages in place --------------------------
    def find_slots(self, stems: Sequence[str]) -> list[tuple[int, int]]:
        """(slot, state) per stem; slot -1 when absent. state 2 = COMPLETE.
        #1439: hashed in C (arena_find_stems) -- one call for a 100k prefix."""
        n = len(stems)
        if n == 0:
            return []
        c_stems = (ctypes.c_char_p * n)(*[s.encode("utf-8") for s in stems])
        slots = (ctypes.c_int64 * n)()
        st = (ctypes.c_int8 * n)()
        self._lib.arena_find_stems(self._base, n, c_stems, slots, st)
        return list(zip(slots, st))

    def find_slots_np(self, stems: Sequence[str]):
        """Posten 2 (18.09.): (slots int64[n], states int8[n]) as numpy views --
        no 520k-tuple Python list (xsn306: zip 95 ms + lead-list 26 ms + the
        per-element loops downstream)."""
        import numpy as np
        n = len(stems)
        if n == 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int8)
        c_stems = (ctypes.c_char_p * n)(*[s.encode("utf-8") for s in stems])
        slots = np.empty((n,), dtype=np.int64)
        st = np.empty((n,), dtype=np.int8)
        self._lib.arena_find_stems(self._base, n,
                                   c_stems,
                                   slots.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                                   st.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)))
        return slots, st

    def ref_slots_np(self, slots, delta: int) -> int:
        """ref_slots over a numpy int64 array (zero-copy pointer)."""
        import numpy as np
        a = np.ascontiguousarray(np.asarray(slots, dtype=np.int64))
        n = int(a.shape[0])
        if n == 0:
            return 0
        if self._ledger is not None:
            return self._ref_ledgered(a, int(delta))
        return int(self._lib.arena_ref_slots(self._base, n,
                                             a.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                                             int(delta)))

    def ref_slots_mask_np(self, slots):
        """PB: +1 on every slot of ``slots`` in ONE call; returns the int8
        per-slot verdict (1 = took the reference). Through this process's
        ledger when one is armed -- exactly the slots that took it are
        recorded, so each one can be released by name later."""
        import numpy as np
        a = np.ascontiguousarray(np.asarray(slots, dtype=np.int64))
        n = int(a.shape[0])
        ok = np.zeros((n,), dtype=np.int8)
        if n == 0:
            return ok
        self._lib.arena_ref_slots_mask(self._base, n,
                                       a.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                                       ok.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)))
        if self._ledger is not None:
            took = a[ok == 1]
            if took.shape[0]:
                self._ledger.took(took)
        return ok

    def _ref_ledgered(self, a, delta: int) -> int:
        """SGLANG_HICACHE_ARENA_QUEUE_REFS: a reference change through this
        process's ledger. -1: only what this process holds is released (the
        rest is refused and counted, never applied). +1: recorded when every
        valid slot took it; a partial batch (a slot left COMPLETE/CLAIMED
        between find and ref) is NOT recorded -- a reference this process
        cannot name is leaked rather than ever released twice."""
        import numpy as np
        if delta < 0:
            a = self._ledger.allow_release(a).numpy()
            if a.shape[0] == 0:
                return 0
        n = int(a.shape[0])
        done = int(self._lib.arena_ref_slots(self._base, n,
                                             a.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                                             int(delta)))
        if delta > 0 and done == int(np.count_nonzero(a >= 0)):
            self._ledger.took(a)
        return done

    def ref_census(self):
        """Read-only census of the slot headers: (COMPLETE slots with a reader
        reference, sum of references, COMPLETE slots). Strided read of every
        header -- a diagnostic, never on a round path."""
        import numpy as np
        out = (ctypes.c_int64 * 6)()
        self._lib.arena_layout(self.slots, self.slot_bytes, out)
        hb, hoff = int(out[0]), int(out[3])
        u32 = np.frombuffer(self._mm, dtype=np.uint32, count=self.slots * hb // 4, offset=hoff)
        hdr = u32.reshape(self.slots, hb // 4)
        state, ref = hdr[:, 0], hdr[:, 1]
        return (int(((state == 2) & (ref > 0)).sum()), int(ref.sum(dtype=np.int64)),
                int((state == 2).sum()))

    def slot_refs(self, slots) -> list[int]:
        """#1424e: the reader references of these slots (the C header, all
        ranks) -- one strided header read, for the rare re-point path."""
        import numpy as np
        out = (ctypes.c_int64 * 6)()
        self._lib.arena_layout(self.slots, self.slot_bytes, out)
        hb, hoff = int(out[0]), int(out[3])
        u32 = np.frombuffer(self._mm, dtype=np.uint32, count=self.slots * hb // 4, offset=hoff)
        ref = u32.reshape(self.slots, hb // 4)[:, 1]
        return [int(ref[int(s)]) if 0 <= int(s) < self.slots else 0 for s in slots]

    def find_states(self, stems: Sequence[str]) -> list[int]:
        """#1439: the states only (0 absent/free, 1 claimed, 2 complete), by stem, hashed in C."""
        n = len(stems)
        if n == 0:
            return []
        c_stems = (ctypes.c_char_p * n)(*[s.encode("utf-8") for s in stems])
        slots = (ctypes.c_int64 * n)()
        st = (ctypes.c_int8 * n)()
        self._lib.arena_find_stems(self._base, n, c_stems, slots, st)
        return list(st)

    def claim_slots(self, stems: Sequence[str], totals: Sequence[int]) -> list[tuple[int, int, int]]:
        """#1427 direct writes: (slot, status, generation) per stem. status
        0 = fresh claim, 1 = join an earlier writer's claim, 2 = already
        COMPLETE, 3 = too large, 4 = no free slot (slot -1)."""
        n = len(stems)
        if n == 0:
            return []
        c_tot = (ctypes.c_int64 * n)(*[int(t) for t in totals])
        c_stems = (ctypes.c_char_p * n)(*[s.encode("utf-8") for s in stems])
        slots = (ctypes.c_int64 * n)()
        gens = (ctypes.c_int64 * n)()
        st = (ctypes.c_int8 * n)()
        # xsn350: keys hashed in C (arena_claim_stems); the Python key128 per
        # stem was ~30 ms per 4096-page node in the scheduler thread.
        rc = self._lib.arena_claim_stems(self._base, n, c_stems, c_tot, slots, gens, st)
        if rc < 0:
            lo, hi = self._keys(stems)
            self._lib.arena_claim(self._base, n, lo, hi, c_tot, c_stems, slots, gens, st)
        return [(int(slots[i]), int(st[i]), int(gens[i])) for i in range(n)]

    def claim_slots_np(self, stems: Sequence[str], totals: Sequence[int]):
        """xsn359: claim_slots as numpy arrays (slots int64, status int8,
        generation int64) -- no 3 x n ctypes reads into Python tuples."""
        import numpy as np
        n = len(stems)
        if n == 0:
            return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int8), np.zeros(0, dtype=np.int64))
        c_tot = (ctypes.c_int64 * n)(*[int(t) for t in totals])
        c_stems = (ctypes.c_char_p * n)(*[s.encode("utf-8") for s in stems])
        slots = np.zeros(n, dtype=np.int64); gens = np.zeros(n, dtype=np.int64); st = np.zeros(n, dtype=np.int8)
        p_slots = slots.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        p_gens = gens.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        p_st = st.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        rc = self._lib.arena_claim_stems(self._base, n, c_stems, c_tot, p_slots, p_gens, p_st)
        if rc < 0:
            lo, hi = self._keys(stems)
            self._lib.arena_claim(self._base, n, lo, hi, c_tot, c_stems, p_slots, p_gens, p_st)
        return slots, st, gens

    def complete_slots_np(self, slots, gens, extents):
        """xsn359: complete_slots on numpy int64 arrays; returns status int8."""
        import numpy as np
        n = int(len(slots))
        if n == 0:
            return np.zeros(0, dtype=np.int8)
        ext = [tuple(extents)] * n
        n_ext, c_off, c_len = self._extents(ext)
        s_np = np.ascontiguousarray(slots, dtype=np.int64); g_np = np.ascontiguousarray(gens, dtype=np.int64)
        st = np.zeros(n, dtype=np.int8)
        self._lib.arena_complete(self._base, n, s_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                                 g_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), n_ext, c_off, c_len,
                                 st.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)))
        return st

    def complete_slots(self, slots: Sequence[int], gens: Sequence[int], extents) -> list[int]:
        """#1427: merge this writer's extents (same shape for every slot) into
        the coverage and flip COMPLETE when the page is full. status per slot:
        1 completed now, 0 merged but not full, 2 already complete, 3 lost."""
        n = len(slots)
        if n == 0:
            return []
        ext = [tuple(extents)] * n
        n_ext, c_off, c_len = self._extents(ext)
        c_slots = (ctypes.c_int64 * n)(*[int(s) for s in slots])
        c_gens = (ctypes.c_int64 * n)(*[int(g) for g in gens])
        st = (ctypes.c_int8 * n)()
        self._lib.arena_complete(self._base, n, c_slots, c_gens, n_ext, c_off, c_len, st)
        return list(st)

    def ref_slots(self, slots: Sequence[int], delta: int) -> int:
        """Reader references: +1 pins COMPLETE slots against eviction, -1 releases.
        Returns how many slots took the delta."""
        n = len(slots)
        if n == 0:
            return 0
        if self._ledger is not None:
            import numpy as np
            return self._ref_ledgered(np.asarray([int(s) for s in slots], dtype=np.int64), int(delta))
        c = (ctypes.c_int64 * n)(*[int(s) for s in slots])
        return int(self._lib.arena_ref_slots(self._base, n, c, int(delta)))

    def complete_census(self):
        """L3-REUSE 0928: ``(slots, gens, key_lo, key_hi)`` of every COMPLETE
        slot as numpy arrays, from ONE C call (arena_complete_census)."""
        import numpy as np

        n = int(self.slots)
        slots = np.empty(n, dtype=np.int64)
        gens = np.empty(n, dtype=np.int64)
        klo = np.empty(n, dtype=np.uint64)
        khi = np.empty(n, dtype=np.uint64)
        P64 = ctypes.POINTER(ctypes.c_int64)
        PU64 = ctypes.POINTER(ctypes.c_uint64)
        got = int(self._lib.arena_complete_census(
            self._base, n, slots.ctypes.data_as(P64), gens.ctypes.data_as(P64),
            klo.ctypes.data_as(PU64), khi.ctypes.data_as(PU64)))
        return slots[:got], gens[:got], klo[:got], khi[:got]

    def pin_complete(self, slots, klo, khi):
        """L3-REUSE 0928: pin each slot still COMPLETE under that key (one C
        call); returns a numpy bool mask of the pinned ones. The caller unpins
        exactly those with :meth:`unpin`."""
        import numpy as np

        s = np.ascontiguousarray(slots, dtype=np.int64)
        lo = np.ascontiguousarray(klo, dtype=np.uint64)
        hi = np.ascontiguousarray(khi, dtype=np.uint64)
        n = int(s.shape[0])
        ok = np.zeros(max(1, n), dtype=np.int8)
        if n:
            self._lib.arena_pin_complete(
                self._base, n, s.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                lo.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                hi.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                ok.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)))
        return ok[:n].astype(bool)

    def unpin(self, slots) -> int:
        """Give back the pins of :meth:`pin_complete` (raw, past the ledger:
        the write-behind is no holder)."""
        import numpy as np

        s = np.ascontiguousarray(slots, dtype=np.int64)
        n = int(s.shape[0])
        if not n:
            return 0
        return int(self._lib.arena_ref_slots(
            self._base, n, s.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), -1))

    def data_offset(self) -> int:
        """Byte offset of slot 0's data inside the mapping."""
        return int(self._lib.arena_data_offset(self._base))

    def slot_stem(self, slot: int) -> str:
        """The store stem recorded in the slot header (any rank may evict it)."""
        raw = self._lib.arena_slot_stem(self._base, int(slot))
        return (raw or b"").decode("utf-8", "replace")

    def slot_ptr(self, slot: int) -> int:
        """Address of slot data inside the mapping (for a C read straight into the slot)."""
        return int(self._lib.arena_slot_ptr(self._base, int(slot)))

    def slot_view(self, slot: int, nbytes: int) -> memoryview:
        off = int(self._lib.arena_slot_ptr(self._base, int(slot))) - int(self._base.value)
        return memoryview(self._mm)[off:off + int(nbytes)]

    def free_slots(self, slots: Sequence[int], reason: str = "unnamed") -> None:
        """Free ``slots`` (any state) -- the generation moves on, a writer
        still holding the old one gets status 3 at its completion. Every
        free names its ``reason`` (#1427s, see _note_free)."""
        n = len(slots)
        _note_free(reason, slots)
        if n == 0:
            return
        c = (ctypes.c_int64 * n)(*[int(s) for s in slots])
        self._lib.arena_free_slots(self._base, n, c)

    @property
    def ival_cap(self) -> int:
        """#1427s: coverage intervals one slot can hold before its completion
        overflows (arena.c ival_cap_for)."""
        return int(self._lib.arena_ival_cap(self._base))

    def drop_unreferenced(self, slots: Sequence[int]) -> int:
        """fnFL2 H19: free the COMPLETE slots nobody references any more --
        a displaced anchor, no disk write (arena.c arena_drop_unreferenced).
        Returns how many were dropped; a referenced/claimed slot stays."""
        n = len(slots)
        if n == 0:
            return 0
        c = (ctypes.c_int64 * n)(*[int(s) for s in slots])
        out = (ctypes.c_int8 * n)()
        return int(self._lib.arena_drop_unreferenced(self._base, n, c, out))

    def reap_stale(self) -> int:
        return int(self._lib.arena_reap_stale(self._base))

    def reap_partial(self, min_age_s: float, cap: int = 64) -> list[int]:
        """#231: free CLAIMED direct-write slots no writer can still come to
        -- every claimer merged or gave up (no open writer), nobody holds a
        reference, untouched for ``min_age_s`` (arena.c arena_reap_partial);
        returns the freed slot ids (at most ``cap`` listed)."""
        out = (ctypes.c_int64 * max(1, int(cap)))()
        n = int(self._lib.arena_reap_partial(self._base, int(float(min_age_s) * 1e3), out, int(cap)))
        return [int(out[i]) for i in range(min(n, int(cap)))] + [-1] * max(0, n - int(cap))

    def unclaim(self, slots: Sequence[int], gens: Sequence[int]) -> int:
        """#231: a direct writer gives its JOINED claim up without merging."""
        n = len(slots)
        if n == 0:
            return 0
        cs = (ctypes.c_int64 * n)(*[int(s) for s in slots])
        cg = (ctypes.c_int64 * n)(*[int(g) for g in gens])
        return int(self._lib.arena_unclaim(self._base, n, cs, cg))

    def release_claims(self, slots: Sequence[int], gens: Sequence[int],
                       reason: str = "release") -> list[int]:
        """#1427r: a direct writer gives up claims it took FRESH. A slot is
        freed only when this writer was its sole claimant (status 0); a slot
        other writers joined -- or already completed -- stays theirs and only
        this writer's claim is resolved (1); a moved generation is skipped (2).
        See arena.c arena_release_claims."""
        n = len(slots)
        if n == 0:
            return []
        cs = (ctypes.c_int64 * n)(*[int(s) for s in slots])
        cg = (ctypes.c_int64 * n)(*[int(g) for g in gens])
        st = (ctypes.c_int8 * n)()
        self._lib.arena_release_claims(self._base, n, cs, cg, st)
        out = list(st)
        freed = [int(slots[i]) for i, x in enumerate(out) if x == 0]
        if freed:
            _note_free(reason, freed)
        return out

    def stats(self) -> dict:
        out = (ctypes.c_int64 * 4)()
        self._lib.arena_stats(self._base, out)
        return {"slots": int(out[0]), "complete": int(out[1]), "claimed": int(out[2]),
                "slot_bytes": int(out[3]), "file_bytes": self.file_bytes}

    def close(self) -> None:
        try:
            self._mm.close()
        except Exception:  # noqa: BLE001
            pass
