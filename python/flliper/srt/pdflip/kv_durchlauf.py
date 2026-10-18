"""S36 (18.10.) KV-DURCHLAUF: a KV store read that finds no arena slot runs THROUGH -- never waits for L2 room.

USER ORDER (18.10., repeated): "wenn im L2 zu wenig platz ist, warum wird dann nicht L3 genommen?" and "wozu ein
puffer? die daten fliessen doch schon". Memory ruege-null-vermeidbare-token-1015: "volle Arena ist keine Ausrede
(L3 lesen)".

THE DEFECT (fLLiper rc9 restart, D TP0 10:09:06-10:10:14Z, 68 s without a D round at 18 open requests; the
09:53:19Z death on the same ground): a KV page read from L3 landed ONLY through an arena slot
(``HiCacheFile.arena_fill_from_disk`` -> ``_await_fill_room`` ROOM-WAIT). The arena was full of pages without an L3
copy (``slots=6485 complete=6485 pinned=6485``, ``ARENA-DROP ... written:0 freed=0``), a slot frees only with an L3
copy, the L3 copy is the write-behind, and the write-behind yielded to that very read. The read waited, was cut
(``READ-ROOM SHORT``), re-read, waited again.

THE RULE (one place -- :func:`durchlauf_stems`): a KV stem that the fill could not claim because the arena is FULL
(after its immediate eviction attempt) and that IS on disk is delivered as a DISK PAGE. It takes no arena slot,
gets no bounce ring of its own and never waits: the read resolves it to a DISK ID (:data:`DISK_BASE` and up, outside
every arena / staging / placeholder / orphan id) whose registry entry HOLDS the L3 file open (an fd keeps the inode
alive even if the L3 LRU unlinks the path; writers publish by rename, so the bytes stay the page the key names).
The load-back reads the file (O_DIRECT on XFS where the target is aligned, the buffered descriptor of the same inode
otherwise) into the transfer target and copies the cells straight into the device KV rows
(``ArenaMHAHostPool._load_disk_rows``). Nothing stays in L2. The Mamba precedent is ANCHOR-BYPASS
(``arena_mamba_pool.arena_resolve_reads``, 14.10.).

GROUP-UNIFORM (SCHEDULER-SOLL R11, raenge-nie-uneins): the rule depends only on the read's stems and the L3 store --
both shared by every rank of the group -- never on this rank's slot luck in time: every rank delivers every on-disk
page of the read (arena slot or disk page, never a wait), so the delivered count every rank feeds into the
existing ``check_prefetch_progress`` MIN is the same. Before, the rank that found the arena full delivered SHORT
(or nothing after the 5 s room wait) and cut the whole group. On the Form A group D only the attention host reads
the KV bytes that matter for the forward; the workers' owner lanes take the same rule.

GPUDirect Storage: on this rig only compatibility mode (S33, ``gdscheck -p``: nvidia_fs not loaded, GeForce BAR1) --
cuFile would be the same pread into host memory plus a copy. Not used.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

ENV = "FLLIPER_PDFLIP_KV_DURCHLAUF"
ENV_DIRECT = "FLLIPER_PDFLIP_KV_DURCHLAUF_DIRECT"
ENV_BLOCK_MIB = "FLLIPER_PDFLIP_KV_DURCHLAUF_BLOCK_MIB"

#: the first disk id: far above staging + arena tokens + 2**22 placeholders + the A2 orphan id of any pool
#: (NF: 4096 + 6485*64 + 2**22 ~ 4.6e6), so no existing range test can read a disk id as one of its own
DISK_BASE = 1 << 40


class _Durchlauf:
    """The fill's answer for a stem delivered as a disk page (not a slot, never an int)."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "DURCHLAUF"


DURCHLAUF = _Durchlauf()


class DiskRef:
    """One disk page in a read's slot list (``_arena_page_get`` -> ``resolve_rows``)."""

    __slots__ = ("d",)

    def __init__(self, d: int):
        self.d = int(d)

    def __repr__(self) -> str:
        return f"DiskRef({self.d})"


def on() -> bool:
    """New feature, default ON (user order 14.10.); ``FLLIPER_PDFLIP_KV_DURCHLAUF=0`` only with a user word."""
    return os.environ.get(ENV, "1").strip() not in ("0", "false", "False", "")


def direct_on() -> bool:
    return os.environ.get(ENV_DIRECT, "1").strip() not in ("0", "false", "False", "")


def durchlauf_stems(still_full, on_disk) -> List[int]:
    """THE decision, one place for every rank: the indices of the fill's ``still_full`` stems (``(index, stem)``:
    the arena was FULL for them after the immediate eviction) that are delivered as disk pages -- every one that is
    on disk. Depends on the read's stems and the shared L3 store only."""
    if not on():
        return []
    return [int(i) for i, st in still_full if st in on_disk]


def disk_base(P: int) -> int:
    P = max(1, int(P))
    return ((DISK_BASE + P - 1) // P) * P


def is_disk_id(i) -> bool:
    return int(i) >= DISK_BASE


def disk_mask(host_indices: torch.Tensor) -> torch.Tensor:
    return host_indices >= DISK_BASE


def any_disk(host_indices) -> bool:
    """Cheap first: no disk page EVER allocated in this process -> no disk id anywhere (no device sync). Once one
    was, every call looks -- a released page's stale id must reach the named refusal, never the staging path."""
    if host_indices is None or REGISTRY.stats["alloc"] == 0:
        return False
    t = torch.as_tensor(host_indices)
    if t.numel() == 0:
        return False
    return bool((t >= DISK_BASE).any())


def disk_ids(d: int, P: int) -> torch.Tensor:
    """The P host ids of disk page ``d``."""
    P = max(1, int(P))
    return torch.arange(P, dtype=torch.int64) + disk_base(P) + int(d) * P


def pages_of_rows(rows: torch.Tensor, P: int):
    """``(disk page per row, lane per row)`` of disk ids (int64, CPU)."""
    P = max(1, int(P))
    r = rows.to("cpu", dtype=torch.int64) - disk_base(P)
    return r // P, r % P


def _raise_nofile() -> None:
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard == resource.RLIM_INFINITY or soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except Exception:  # noqa: BLE001 - a limit we cannot raise is reported by the EMFILE below
        pass


class _Entry:
    __slots__ = ("key", "stem", "path", "fd", "direct", "total", "qsa_fd", "qsa_mode")

    def __init__(self, key, stem, path, fd, direct, total):
        self.key = key
        self.stem = stem
        self.path = path
        self.fd = fd
        self.direct = direct
        self.total = int(total)
        self.qsa_fd = -1
        self.qsa_mode = None   # None: not asked yet, "fd": file held, "store": the arena/L3 at load, "miss"


class DiskPages:
    """The disk pages of this process: disk page index -> (key, stem, held fd). Allocated by the store read's
    resolve (prefetch io thread), released with the host rows that name them (``free`` / ``release_queued_rows``,
    scheduler thread) -- one lock."""

    def __init__(self):
        self._lock = threading.Lock()
        self._e: dict = {}
        # S38 review (18.10.): disk page ids are NEVER reused (monotonic). A LIFO free list handed a released id to
        # the next read at once: the QSA mirror's per-load stage cache (``QSAPagedHostPool._s36_disk_stage``, keyed
        # by the disk page tuple) then served the OLD page's QSA index for the new page (wrong sparse selection,
        # silent), and a stale host row would have loaded another page's bytes instead of reaching the named
        # ``LOAD REFUSED``. The id space is 2**40 + d*P in int64 -- it does not run out.
        self._next = 0
        self.backend = None
        self.stats = {"alloc": 0, "released": 0, "open_failed": 0, "direct_fds": 0,
                      "read_pages": 0, "read_direct": 0, "read_failed": 0, "qsa_fd": 0, "qsa_store": 0,
                      "qsa_miss": 0}
        self._raised = False

    def live(self) -> int:
        return len(self._e)

    def _open(self, path: str):
        if not self._raised:
            _raise_nofile()
            self._raised = True
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if direct_on() and hasattr(os, "O_DIRECT"):
            try:
                return os.open(path, flags | os.O_DIRECT), True
            except OSError as exc:
                if exc.errno not in (22, 95):   # EINVAL / EOPNOTSUPP: no O_DIRECT here -> buffered
                    raise
        return os.open(path, flags), False

    def alloc(self, keys: Sequence[str], stems: Sequence[str], paths: Sequence[str], total: int) -> List[Optional[int]]:
        """One disk page per stem, or None where the L3 file cannot be opened (gone since the stat: the read's
        prefix ends there, an honest miss as before -- never a wait)."""
        out: List[Optional[int]] = []
        for k, st, p in zip(keys, stems, paths):
            try:
                fd, direct = self._open(p)
            except OSError as exc:
                self.stats["open_failed"] += 1
                if self.stats["open_failed"] <= 8 or self.stats["open_failed"] % 256 == 0:
                    logger.warning("PDFLIP KV-DURCHLAUF OPEN-FAILED n=%d stem=%s errno=%s -- the page is an honest miss "
                                   "(the read's prefix ends here)", self.stats["open_failed"], str(st)[:24], exc.errno)
                out.append(None)
                continue
            with self._lock:
                d = self._next
                self._next += 1
                self._e[d] = _Entry(k, st, p, fd, direct, total)
                self.stats["alloc"] += 1
                if direct:
                    self.stats["direct_fds"] += 1
            out.append(d)
        return out

    def entry(self, d: int) -> Optional[_Entry]:
        return self._e.get(int(d))

    def release(self, ds) -> int:
        n = 0
        for d in ds:
            with self._lock:
                e = self._e.pop(int(d), None)
            if e is None:
                continue
            n += 1
            for fd in (e.fd, e.qsa_fd):
                if fd is not None and fd >= 0:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
        self.stats["released"] += n
        return n

    def release_rows(self, host_indices, P: int) -> int:
        """Release the disk pages named by these host rows (one per page; rows of other ranges are ignored)."""
        if host_indices is None or self.live() == 0:
            return 0
        t = torch.as_tensor(host_indices).reshape(-1)
        if t.numel() == 0:
            return 0
        t = t.to("cpu", dtype=torch.int64)
        m = t >= DISK_BASE
        if not bool(m.any()):
            return 0
        pg, _ln = pages_of_rows(t[m], P)
        return self.release(torch.unique(pg).tolist())

    def stem_of(self, d: int) -> Optional[str]:
        e = self._e.get(int(d))
        return None if e is None else e.stem

    # -- the QSA index sidecar of a disk page (Next Flash) ----------------------------------------------------
    def attach_qsa(self, d: int, path: Optional[str], in_arena: bool) -> bool:
        """At the store read: the page's QSA index file is held open (``fd``), or it is in the QSA arena (``store``:
        read through the backend at the load), or it is a miss (the read's QSA hit pages end here, exactly as a
        QSA miss of an arena page)."""
        e = self._e.get(int(d))
        if e is None:
            return False
        if e.qsa_mode in ("fd", "store"):
            return True
        if path:
            try:
                e.qsa_fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
                e.qsa_mode = "fd"
                self.stats["qsa_fd"] += 1
                return True
            except OSError:
                pass
        if in_arena:
            e.qsa_mode = "store"
            self.stats["qsa_store"] += 1
            return True
        e.qsa_mode = "miss"
        self.stats["qsa_miss"] += 1
        return False


REGISTRY = DiskPages()


def block_pages(page_bytes: int) -> int:
    try:
        mib = float(os.environ.get(ENV_BLOCK_MIB, "") or 64)
    except ValueError:
        mib = 64.0
    return max(1, int(mib * (1 << 20)) // max(1, int(page_bytes)))


def read_pages(entries, ptrs, total: int) -> List[int]:
    """Whole L3 pages of ``entries`` into ``ptrs`` (one page each), from the HELD descriptors, in parallel over the
    L3 read threads; status per page (0 ok). The read has priority over the L2 -> L3 write-behind
    (``l3_write_behind.read_enter``)."""
    from flliper.srt.mem_cache import l3_write_behind as _wb
    from flliper.srt.mem_cache.hicache_storage import _l3_executor, l3_read_threads
    from flliper.srt.mem_cache.storage.file.pageio import load as _load_pageio

    n = len(entries)
    if n == 0:
        return []
    pio = _load_pageio()
    st = [4] * n
    dr = [0] * n
    _wb.read_enter()
    try:
        if pio is not None and getattr(pio, "_has_fds", False):
            k = max(1, min(l3_read_threads(), n))
            groups = [list(range(g, n, k)) for g in range(k)]

            def one(idx):
                return pio.read_fds([entries[i].fd for i in idx], [0] * len(idx), [int(total)] * len(idx),
                                    [int(ptrs[i]) for i in idx])

            if k == 1:
                res = [one(groups[0])]
            else:
                res = list(_l3_executor(k).map(one, groups))
            for idx, (s_, d_) in zip(groups, res):
                for i, s1, d1 in zip(idx, s_, d_):
                    st[i] = int(s1)
                    dr[i] = int(d1)
        else:
            import ctypes

            for i, e in enumerate(entries):
                try:
                    buf = (ctypes.c_char * int(total)).from_address(int(ptrs[i]))
                    got = os.preadv(e.fd, [memoryview(buf).cast("B")], 0)
                    st[i] = 0 if got == int(total) else 3
                except OSError:
                    try:
                        bfd = os.open(f"/proc/self/fd/{e.fd}", os.O_RDONLY)
                        try:
                            buf = (ctypes.c_char * int(total)).from_address(int(ptrs[i]))
                            got = os.preadv(bfd, [memoryview(buf).cast("B")], 0)
                            st[i] = 0 if got == int(total) else 3
                        finally:
                            os.close(bfd)
                    except OSError:
                        st[i] = 4
    finally:
        _wb.read_exit()
    REGISTRY.stats["read_pages"] += n
    REGISTRY.stats["read_direct"] += sum(dr)
    REGISTRY.stats["read_failed"] += sum(1 for s in st if s != 0)
    return st


_N = [0]


def note_read(rid, pages: int, disk: int) -> None:
    """One line per read that delivered disk pages (rate limited)."""
    if disk <= 0:
        return
    _N[0] += 1
    k = _N[0]
    if k <= 16 or (k & (k - 1)) == 0:
        s = REGISTRY.stats
        logger.info("PDFLIP KV-DURCHLAUF n=%d rid=%s pages=%d disk=%d live=%d alloc=%d released=%d direct_fds=%d "
                    "read_pages=%d read_direct=%d read_failed=%d qsa=fd:%d,store:%d,miss:%d (the arena was FULL "
                    "for these pages: read THROUGH from L3 into the device at the load, no L2 slot, no wait)",
                    k, rid, pages, disk, REGISTRY.live(), s["alloc"], s["released"], s["direct_fds"],
                    s["read_pages"], s["read_direct"], s["read_failed"], s["qsa_fd"], s["qsa_store"], s["qsa_miss"])


def _reset_for_tests() -> None:
    global REGISTRY
    try:
        REGISTRY.release(list(REGISTRY._e))
    except Exception:  # noqa: BLE001
        pass
    REGISTRY = DiskPages()
