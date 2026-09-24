"""Host-side storage for the offloaded Qwen4-Exp PLE n-gram table.

``--ple-offload-embedding`` keeps the PLE table (47.7 GiB in fp8 for
Qwen3.8-Flash-Next) out of device memory and lets the Triton gather kernel read
rows straight from a host pointer. Two backends provide that pointer:

``pinned`` (default)
    ``torch.empty(..., pin_memory=True)``. On a discrete GPU this frees VRAM.

``file``
    A file-backed, shared ``mmap`` of a sparse file under
    ``--ple-offload-dir``. Meant for unified-memory parts (GB10 / DGX Spark and
    similar), where pinned host memory comes out of the *same* pool as the
    model weights and ``pinned`` therefore frees nothing: Qwen3.8-Flash-Next is
    126.0 GiB of weights on a 121.63 GiB box and does not boot with ``pinned``.
    The kernel dereferences the pageable pointer directly, which only works on
    devices that report ``cudaDevAttrPageableMemoryAccessUsesHostPageTables``;
    rows are paged in from storage on demand, the file is sparse, deterministic
    in name and reused across restarts, and gathers of prefill size hint the
    page cache (``posix_fadvise(WILLNEED)``) so page faults are served
    concurrently instead of one at a time. A background trimmer keeps the
    mapping's resident set under a budget, because faulting rows in maps whole
    page-cache folios and the table would otherwise creep towards full
    residency (see ``PleFileRssTrimmer``).

This module has no Triton or CUDA-kernel imports so that its allocator and
prefetcher can be unit-tested on CPU.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import functools
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers import host_contention
from sglang.srt.layers.prefill_timing import (
    log_ple_gather,
    log_ple_gather_host,
    timing_on,
)

logger = logging.getLogger(__name__)

_LIBC: Optional[ctypes.CDLL] = None
_SMAPS_HEADER = re.compile(r"^([0-9a-f]+)-([0-9a-f]+) ")
_SMAPS_RSS = re.compile(r"^Rss:\s+(\d+) kB")

PLE_OFFLOAD_BACKENDS = ("pinned", "file", "checkpoint")

# cudaDeviceAttr enum values (cuda_runtime_api.h).
_CUDA_DEV_ATTR_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100
_MADV_RANDOM = 1
_MADV_DONTNEED = 4
_PAGE_SHIFT = 12
# One MADV_DONTNEED call takes mmap_lock for its whole range; over the full
# 47.7 GiB table that is ~3.5 s during which every fault in the process --
# including the ones the gather kernel takes -- stalls. Trim in slices.
PLE_FILE_RSS_TRIM_CHUNK_BYTES = 1 << 30
# Below this many rows a gather is decode-sized (16 rows per token): the page
# faults are cheap and the host-side hint would cost more than it saves.
PLE_FILE_PREFETCH_MIN_ROWS = 2048


class PleFilePrefetcher:
    """Hint the page cache about the rows a prefill-sized gather is about to read.

    With the table on storage, a cold prefill chunk faults tens of thousands of
    4 KiB pages one at a time from inside the gather kernel. Advising them
    first (``posix_fadvise(WILLNEED)`` per distinct page, on one background
    thread) lets the block layer serve them concurrently. Measured on a GB10 /
    NVMe: cold prefill 650-750 tok/s -> 1,000-2,100 tok/s (warm: ~2,200-2,600).
    Decode-sized gathers are skipped; nothing runs during CUDA-graph capture.
    """

    def __init__(
        self,
        path: str,
        row_bytes: int,
        min_rows: int = PLE_FILE_PREFETCH_MIN_ROWS,
    ) -> None:
        self._fd = os.open(path, os.O_RDONLY)
        self._row_bytes = int(row_bytes)
        self._min_rows = int(min_rows)
        self._pool = ThreadPoolExecutor(max_workers=1)

    @staticmethod
    def pages_for_rows(row_ids: torch.Tensor, row_bytes: int) -> list[int]:
        start = row_ids.to(torch.int64) * row_bytes
        end = start + (row_bytes - 1)
        return (
            torch.cat([start >> _PAGE_SHIFT, end >> _PAGE_SHIFT])
            .unique(sorted=True)
            .tolist()
        )

    def _advise(self, pages: list[int]) -> None:
        for p in pages:
            try:
                os.posix_fadvise(
                    self._fd, p << _PAGE_SHIFT, 1 << _PAGE_SHIFT, os.POSIX_FADV_WILLNEED
                )
            except OSError:
                return

    def enqueue(
        self,
        flat_ids: torch.Tensor,
        *,
        vocab_start: int = 0,
        vocab_end: Optional[int] = None,
    ) -> bool:
        """Queue the hint for ``flat_ids``. Returns whether anything was queued."""
        if flat_ids.numel() < self._min_rows:
            return False
        if flat_ids.is_cuda and torch.cuda.is_current_stream_capturing():
            return False
        # The .cpu() syncs the stream; acceptable for prefill chunks (~1 s) and
        # it is what lets the page set be computed without touching the kernel.
        row_ids = flat_ids.detach().cpu()
        if vocab_end is not None:
            # The file contains only this rank's vocabulary shard.
            row_ids = row_ids[(row_ids >= vocab_start) & (row_ids < vocab_end)]
        row_ids = row_ids - vocab_start
        if row_ids.numel() == 0:
            return False
        pages = self.pages_for_rows(row_ids, self._row_bytes)
        self._pool.submit(self._advise, pages)
        return True

    def close(self) -> None:
        self._pool.shutdown(wait=False)
        try:
            os.close(self._fd)
        except OSError:
            pass


class PleCheckpointPrefetcher:
    """WP1b: the ``checkpoint`` backend's page-cache warmer.

    The gather kernel reads the table through HMM out of read-only mmaps of
    the checkpoint's own safetensors files, and every cold row is one 4 KiB
    page fault served from storage inside the kernel, one at a time. Measured
    on the 5090 (2026-09-16, cold ZFS): 262144 random rows in 99 s, ~2.6k
    rows/s -- a 9k-token prefill (~72k rows) spends ~30 s in one PLE layer.
    This warms the pages first: ``posix_fadvise(WILLNEED)`` per distinct
    page and, because not every file system honours the hint, a ``pread`` of
    each page on a small thread pool, so the block layer serves them
    concurrently and the faults find them in the page cache. Nothing is
    pinned and nothing is kept: the page cache stays the only copy (PLE law:
    the table lives on disk). Decode-sized gathers are skipped (min_rows);
    nothing runs during CUDA-graph capture.
    """

    def __init__(
        self,
        table: "CheckpointMappedPleTable",
        min_rows: int = PLE_FILE_PREFETCH_MIN_ROWS,
        workers: int = 8,
    ) -> None:
        if not table.shard_files or len(table.shard_files) != len(table.bases):
            raise ValueError("checkpoint PLE table carries no per-shard file map")
        self._table = table
        self._row_bytes = int(table.row_bytes)
        self._min_rows = int(min_rows)
        self._fds: dict = {}
        for path in dict.fromkeys(table.shard_files):
            self._fds[path] = os.open(path, os.O_RDONLY)
        self._shard_fd = [self._fds[p] for p in table.shard_files]
        self._pool = ThreadPoolExecutor(max_workers=max(1, int(workers)))
        self._workers = max(1, int(workers))
        self.stats = {"enqueued": 0, "rows": 0, "pages": 0}

    def pages_by_fd(self, row_ids: torch.Tensor) -> dict:
        """Global row ids (host tensor) -> {fd: sorted unique page numbers}."""
        row_ids = row_ids.to(torch.int64)
        shard = torch.div(row_ids, self._table.shard_rows, rounding_mode="floor")
        row = row_ids - shard * self._table.shard_rows
        offsets = torch.tensor(self._table.shard_offsets, dtype=torch.int64)
        start = offsets[shard] + row * self._row_bytes
        end = start + (self._row_bytes - 1)
        out: dict = {}
        for s in torch.unique(shard).tolist():
            sel = shard == s
            pages = torch.cat([start[sel] >> _PAGE_SHIFT, end[sel] >> _PAGE_SHIFT])
            fd = self._shard_fd[int(s)]
            out.setdefault(fd, []).append(pages)
        return {fd: torch.cat(v).unique(sorted=True).tolist() for fd, v in out.items()}

    @staticmethod
    def _touch(fd: int, pages: list) -> None:
        for p in pages:
            try:
                os.posix_fadvise(fd, p << _PAGE_SHIFT, 1 << _PAGE_SHIFT, os.POSIX_FADV_WILLNEED)
            except OSError:
                break
        for p in pages:
            try:
                os.pread(fd, 1 << _PAGE_SHIFT, p << _PAGE_SHIFT)
            except OSError:
                return

    def enqueue(
        self,
        flat_ids: torch.Tensor,
        *,
        vocab_start: int = 0,
        vocab_end: Optional[int] = None,
    ) -> bool:
        """Queue the warm-up for ``flat_ids``. Returns whether anything was queued."""
        if flat_ids.numel() < self._min_rows:
            return False
        if flat_ids.is_cuda and torch.cuda.is_current_stream_capturing():
            return False
        row_ids = flat_ids.detach().cpu().to(torch.int64)
        if vocab_end is not None:
            row_ids = row_ids[(row_ids >= vocab_start) & (row_ids < vocab_end)]
        row_ids = row_ids[(row_ids >= 0) & (row_ids < self._table.total_rows)]
        if row_ids.numel() == 0:
            return False
        note_ple_prefill_rows(row_ids)  # fnFL2 H29a (hmm gather on P)
        by_fd = self.pages_by_fd(row_ids)
        n_pages = 0
        for fd, pages in by_fd.items():
            n_pages += len(pages)
            chunk = max(1, (len(pages) + self._workers - 1) // self._workers)
            for i in range(0, len(pages), chunk):
                self._pool.submit(self._touch, fd, pages[i : i + chunk])
        self.stats["enqueued"] += 1
        self.stats["rows"] += int(row_ids.numel())
        self.stats["pages"] += n_pages
        return True

    def close(self) -> None:
        self._pool.shutdown(wait=False)
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass


PLE_CKPT_GATHER_ENV = "SGLANG_QWEN4_PLE_CKPT_GATHER"
PLE_CKPT_PREAD_WORKERS_ENV = "SGLANG_QWEN4_PLE_PREAD_WORKERS"
PLE_CKPT_PREAD_WORKERS = 32


class PleCheckpointPreadGather:
    """Task #55 (fnFL2 v26, 21.09.): prefill-sized gathers of the ``checkpoint``
    backend read on the CPU instead of through HMM.

    Measured on this rig's ZFS (zfs_mmap_probe.py / zfs_lock_probe.py,
    21.09.): the gather kernel's HMM faults serve ~1.6k cold pages/s, one at a
    time (an 8192-token chunk = 393k rows = 393k random 4 KiB pages = 4 min);
    ``pread`` does NOT fill the page cache for the mmap (mincore 0/20000 after
    pread), so the pread+WILLNEED warmer only warms the ARC; and a CPU touch of
    the mmap serialises at ~6.5k pages/s globally (one file = four files).
    ``pread`` itself runs at 36-55k IOPS on 8-64 threads. So a gather of
    ``>= min_rows`` rows is done here: every row is one ``preadv`` straight
    into a (pinned) staging buffer on a thread pool, then ONE H2D copy. Same
    bytes as the mmap, same in-range rule as the kernel (rows outside
    ``[vocab_start, vocab_end)`` are zero), output bf16. Decode-sized gathers
    keep the kernel (their rows are warm). Off unless
    ``SGLANG_QWEN4_PLE_CKPT_GATHER=pread``.
    """

    def __init__(
        self,
        table: "CheckpointMappedPleTable",
        *,
        min_rows: int = PLE_FILE_PREFETCH_MIN_ROWS,
        workers: int = PLE_CKPT_PREAD_WORKERS,
    ) -> None:
        if not table.shard_files or len(table.shard_files) != len(table.bases):
            raise ValueError("checkpoint PLE table carries no per-shard file map")
        if len(table.shard_offsets) != len(table.bases):
            raise ValueError("checkpoint PLE table carries no per-shard offsets")
        self._table = table
        self._row_bytes = int(table.row_bytes)
        self._min_rows = int(min_rows)
        self._workers = max(1, int(workers))
        self._fds: dict = {}
        for path in dict.fromkeys(table.shard_files):
            self._fds[path] = os.open(path, os.O_RDONLY)
        self._shard_fd = [self._fds[p] for p in table.shard_files]
        self._pool = ThreadPoolExecutor(max_workers=self._workers)
        self._staging: Optional[torch.Tensor] = None
        self.stats = {"gathers": 0, "rows": 0, "zero_rows": 0, "seconds": 0.0}

    @property
    def min_rows(self) -> int:
        return self._min_rows

    def wants(self, flat_ids: torch.Tensor) -> bool:
        if flat_ids.numel() < self._min_rows:
            return False
        if flat_ids.is_cuda and torch.cuda.is_current_stream_capturing():
            return False
        return True

    def _staging_for(self, n: int) -> torch.Tensor:
        st = self._staging
        if st is None or st.shape[0] < n:
            cap = max(n, 1 << 15)
            pin = torch.cuda.is_available()
            st = torch.empty((cap, self._table.embedding_dim), dtype=self._table.dtype, pin_memory=pin)
            self._staging = st
        return st[:n]

    @staticmethod
    def _read_rows(buf: memoryview, rb: int, rows: list, fds: list, offs: list) -> None:
        """``rows[j]`` (staging row) <- ``preadv(fds[j], offs[j])``."""
        for j in range(len(rows)):
            i = rows[j]
            os.preadv(fds[j], [buf[i * rb : (i + 1) * rb]], offs[j])

    def gather_into(
        self,
        flat_ids: torch.Tensor,
        out: torch.Tensor,
        *,
        vocab_start: int = 0,
        vocab_end: Optional[int] = None,
    ) -> torch.Tensor:
        """Fill ``out`` ([n, dim] bf16) with the rows of ``flat_ids``."""
        import time as _time

        # fnFL2 H38: the host split (PLE-GATHER-HOST) rides the timing switch
        timed = timing_on()
        host0 = host_contention.sample() if timed else None
        t0 = _time.monotonic()
        t_read0 = t_read1 = t0
        split = host_contention.ThreadSplit()
        ids = flat_ids.detach().reshape(-1).cpu().to(torch.int64)
        n = int(ids.numel())
        dim = self._table.embedding_dim
        if out.numel() != n * dim or (out.dim() >= 1 and out.shape[-1] != dim):
            raise ValueError(
                f"PLE pread gather: output {tuple(out.shape)} does not match {n} ids x {dim}"
            )
        if n == 0:
            return out
        # the model hands (*ids.shape, dim); fill it through a flat [n, dim] view
        flat_out = out.reshape(n, dim)
        if vocab_end is None:
            vocab_end = self._table.total_rows
        in_range = (ids >= vocab_start) & (ids < vocab_end) & (ids < self._table.total_rows)
        staging = self._staging_for(n)
        valid_idx = torch.nonzero(in_range).flatten()
        nv = int(valid_idx.numel())
        # fnFL2 H29a: P hands the rows it just read to D's decode warm
        note_ple_prefill_rows(ids[in_range])
        if nv != n:
            # same rule as the kernel: out-of-range rows are 0.0
            staging.zero_()
        if nv:
            vid = ids[valid_idx]
            shard = torch.div(vid, self._table.shard_rows, rounding_mode="floor")
            local = vid - shard * self._table.shard_rows
            offs = torch.tensor(self._table.shard_offsets, dtype=torch.int64)[shard] + local * self._row_bytes
            fds = torch.tensor(self._shard_fd, dtype=torch.int64)[shard]
            # read in (fd, offset) order -- the block layer likes locality --
            # but land every row at its own staging index (no scatter pass)
            order = torch.argsort(fds * (1 << 48) + offs)
            rows = valid_idx[order].tolist()
            fds_l = fds[order].tolist()
            offs_l = offs[order].tolist()
            buf = memoryview(staging.view(torch.uint8).numpy().reshape(-1))
            chunk = max(256, (nv + self._workers - 1) // self._workers)
            # timed: every task also returns its (wall, on-CPU, run-queue) ns
            task = functools.partial(host_contention.timed_task, self._read_rows) if timed else self._read_rows
            t_read0 = _time.monotonic()
            futs = [
                self._pool.submit(
                    task, buf, self._row_bytes, rows[lo : lo + chunk], fds_l[lo : lo + chunk], offs_l[lo : lo + chunk]
                )
                for lo in range(0, nv, chunk)
            ]
            for f in futs:
                r = f.result()
                if timed:
                    split.add(*r)
            t_read1 = _time.monotonic()
        else:
            t_read0 = t_read1 = _time.monotonic()
        flat_out.copy_(staging, non_blocking=out.is_cuda)
        if flat_out.data_ptr() != out.data_ptr():
            out.copy_(flat_out.reshape(out.shape))
        t_end = _time.monotonic()
        dt = t_end - t0
        self.stats["gathers"] += 1
        self.stats["rows"] += n
        self.stats["zero_rows"] += n - nv
        self.stats["seconds"] += dt
        log_ple_gather(n, n - nv, dt, self._workers)
        if timed:
            log_ple_gather_host(
                n,
                dt * 1000.0,
                (t_read0 - t0) * 1000.0,
                (t_read1 - t_read0) * 1000.0,
                (t_end - t_read1) * 1000.0,
                split,
                host_contention.delta(host0, host_contention.sample()),
            )
        return out

    def close(self) -> None:
        self._pool.shutdown(wait=False)
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass


def make_ple_checkpoint_pread_gather(
    table: "CheckpointMappedPleTable",
) -> Optional["PleCheckpointPreadGather"]:
    """``SGLANG_QWEN4_PLE_CKPT_GATHER=pread`` routes prefill-sized gathers of
    the checkpoint backend through :class:`PleCheckpointPreadGather`; the
    default (``hmm``) keeps the kernel for every gather."""
    mode = os.environ.get(PLE_CKPT_GATHER_ENV, "hmm").strip().lower()
    if mode in ("", "hmm", "0", "off"):
        return None
    if mode != "pread":
        raise ValueError(f"{PLE_CKPT_GATHER_ENV} must be 'hmm' or 'pread', got {mode!r}")
    min_rows = int(
        os.environ.get("SGLANG_QWEN4_PLE_PREFETCH_MIN_ROWS", str(PLE_FILE_PREFETCH_MIN_ROWS))
    )
    workers = int(os.environ.get(PLE_CKPT_PREAD_WORKERS_ENV, str(PLE_CKPT_PREAD_WORKERS)))
    g = PleCheckpointPreadGather(table, min_rows=min_rows, workers=workers)
    logger.info(
        "PLE table: prefill gathers of >= %d rows read on the CPU (preadv per row, "
        "%d threads, pinned staging, one H2D copy) -- the HMM kernel serves only "
        "smaller gathers (#55: cold pages cost 4 min per 8k chunk through HMM on ZFS)",
        min_rows,
        workers,
    )
    return g


def make_ple_checkpoint_prefetcher(
    table: "CheckpointMappedPleTable",
) -> Optional["PleCheckpointPrefetcher"]:
    """The warmer for a ``checkpoint``-backend table; the file backend's
    SGLANG_QWEN4_PLE_FILE_PREFETCH switch governs both."""
    if not envs.SGLANG_QWEN4_PLE_FILE_PREFETCH.get():
        return None
    # 20.09. (fn8ad, Task #54): the decode round's own gather (48 rows) was
    # below the threshold, so every decode round paid its cold page faults
    # inside the kernel -- 8.6 ms of a 34.5 ms round on every rank. The
    # threshold is an env now (SGLANG_QWEN4_PLE_PREFETCH_MIN_ROWS, default
    # unchanged) so a boot can warm decode-sized gathers too.
    min_rows = int(
        os.environ.get("SGLANG_QWEN4_PLE_PREFETCH_MIN_ROWS", str(PLE_FILE_PREFETCH_MIN_ROWS))
    )
    prefetcher = PleCheckpointPrefetcher(table, min_rows=min_rows)
    logger.info(
        "PLE table: checkpoint page warm-up on for gathers of >= %d rows "
        "(row = %d B, %d files, %d threads)",
        min_rows,
        table.row_bytes,
        len(prefetcher._fds),
        prefetcher._workers,
    )
    return prefetcher


class PleFileRssTrimmer:
    """Keep the mapped table's resident set under a budget.

    Every random row fault maps in a whole page-cache folio, so with large
    folios (Linux 6.x) the mapping's Rss climbs towards the table's full size
    while a generated token only reads a few KB of it: measured ~45 KB of Rss
    growth per token on a GB10. On a unified-memory part that is not a slow
    leak, it is a countdown -- the free-memory readings that size the KV pool
    come from the same pool the folios are accumulating in.

    ``MADV_RANDOM`` does not prevent it (it limits readahead I/O, not the
    mapping-in of folios already in cache) and ``posix_fadvise(DONTNEED)`` does
    not release them either. ``MADV_DONTNEED`` over the mapping does: the page
    table entries go, the pages stay in the page cache, and hot rows come back
    at minor-fault cost.

    Dropping entries under a running gather is the state this backend already
    handles: the file starts out entirely unfaulted and every cold row is
    faulted in from inside the kernel through the same host page tables. What
    must not happen is one ``madvise`` call over the whole table, so the trim
    is chunked (see ``PLE_FILE_RSS_TRIM_CHUNK_BYTES``) and runs on its own
    daemon thread -- decode replays a CUDA graph and executes no Python, so a
    hook in the gather would never fire in the phase that grows the table.
    """

    def __init__(
        self,
        addr: int,
        nbytes: int,
        budget_bytes: int,
        interval_s: float,
        chunk_bytes: int = PLE_FILE_RSS_TRIM_CHUNK_BYTES,
    ) -> None:
        self._addr = int(addr)
        self._nbytes = int(nbytes)
        self._budget = int(budget_bytes)
        self._interval = float(interval_s)
        self._chunk = int(chunk_bytes)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="ple-file-rss-trim", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def mapping_rss_bytes(self) -> Optional[int]:
        """Resident bytes of the VMAs backing the table, or None off Linux."""
        return _mapping_rss_bytes(self._addr, self._nbytes)

    def trim_once(self) -> int:
        """Drop the mapping's resident pages if over budget. Returns bytes freed."""
        before = self.mapping_rss_bytes()
        if before is None or before <= self._budget:
            return 0
        for offset in range(0, self._nbytes, self._chunk):
            if self._stop.is_set():
                break
            length = min(self._chunk, self._nbytes - offset)
            if not _madvise(self._addr + offset, length, _MADV_DONTNEED):
                return 0
            # Let the faults that queued behind mmap_lock through.
            self._stop.wait(0.005)
        after = self.mapping_rss_bytes()
        freed = before - after if after is not None else 0
        logger.info(
            "PLE table: trimmed resident set %.1f -> %.1f GiB (budget %.1f GiB)",
            before / 2**30,
            (after if after is not None else 0) / 2**30,
            self._budget / 2**30,
        )
        return max(freed, 0)

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.trim_once()
            except Exception as exc:  # advisory only; never fail a request
                logger.warning("PLE table: resident-set trim skipped (%s)", exc)

    def close(self) -> None:
        self._stop.set()


def allocate_ple_host_table(
    shape: Sequence[int],
    dtype: torch.dtype,
    backend: str = "pinned",
    table_dir: Optional[str] = None,
    tag: Optional[str] = None,
) -> torch.Tensor:
    """Return a host tensor of ``shape``/``dtype`` for the PLE table.

    For the file backend, ``table_dir`` should be private to one checkpoint
    (the server defaults it to ``$SGLANG_CACHE_DIR/ple/<model path>``): the
    file name only encodes shape, dtype and ``tag``, and every boot rewrites
    the whole table through the weight loader.
    """
    if backend not in PLE_OFFLOAD_BACKENDS:
        raise ValueError(
            f"unknown PLE offload backend {backend!r}; choose from {PLE_OFFLOAD_BACKENDS}"
        )
    if backend == "pinned":
        return torch.empty(tuple(shape), dtype=dtype, device="cpu", pin_memory=True)

    numel = 1
    for d in shape:
        numel *= int(d)
    nbytes = numel * torch.empty(0, dtype=dtype).element_size()
    table_dir = os.path.expanduser(table_dir or envs.SGLANG_QWEN4_PLE_FILE_DIR.get())
    os.makedirs(table_dir, exist_ok=True)
    path = os.path.join(table_dir, ple_table_file_name(shape, dtype, tag))
    if not os.path.exists(path) or os.path.getsize(path) != nbytes:
        # Sparse: only pages that get written take disk space.
        with open(path, "wb") as f:
            f.truncate(nbytes)
    logger.info(
        "PLE table: file-backed mmap %s (%.1f GiB, %s)", path, nbytes / 2**30, dtype
    )
    storage = torch.from_file(path, shared=True, size=nbytes, dtype=torch.uint8)
    _madvise_random(storage, nbytes)
    table = storage.view(dtype).view(*[int(d) for d in shape])
    table._sglang_ple_file_path = path  # consumed by PleFilePrefetcher
    return table


def make_ple_file_prefetcher(table: torch.Tensor) -> Optional[PleFilePrefetcher]:
    """A prefetcher for a table returned by ``allocate_ple_host_table(..., "file")``."""
    path = getattr(table, "_sglang_ple_file_path", None)
    if path is None or not envs.SGLANG_QWEN4_PLE_FILE_PREFETCH.get():
        return None
    row_bytes = (
        int(table.shape[-1]) * table.element_size()
        if table.dim() >= 2
        else table.element_size()
    )
    prefetcher = PleFilePrefetcher(path=path, row_bytes=row_bytes)
    logger.info(
        "PLE table: WILLNEED prefetch on for gathers of >= %d rows (row = %d B)",
        PLE_FILE_PREFETCH_MIN_ROWS,
        row_bytes,
    )
    return prefetcher


def make_ple_file_rss_trimmer(table: torch.Tensor) -> Optional[PleFileRssTrimmer]:
    """A started trimmer for a table from ``allocate_ple_host_table(..., "file")``.

    ``SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB=0`` turns it off; it is also absent
    where the resident set cannot be read (no ``/proc/self/smaps``).
    """
    path = getattr(table, "_sglang_ple_file_path", None)
    if path is None:
        return None
    budget_gb = float(envs.SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB.get())
    if budget_gb <= 0:
        return None
    nbytes = table.numel() * table.element_size()
    if _mapping_rss_bytes(table.data_ptr(), nbytes) is None:
        logger.warning(
            "PLE table: resident-set trim off, /proc/self/smaps is not readable; "
            "the mapping will creep towards %.1f GiB resident",
            nbytes / 2**30,
        )
        return None
    trimmer = PleFileRssTrimmer(
        addr=table.data_ptr(),
        nbytes=nbytes,
        budget_bytes=int(budget_gb * 2**30),
        interval_s=float(envs.SGLANG_QWEN4_PLE_FILE_RSS_INTERVAL_S.get()),
    )
    trimmer.start()
    logger.info(
        "PLE table: resident set capped at %.1f GiB, checked every %.0f s",
        budget_gb,
        float(envs.SGLANG_QWEN4_PLE_FILE_RSS_INTERVAL_S.get()),
    )
    return trimmer


def check_file_backend_supported(device_index: int = 0) -> None:
    """Fail fast at load time instead of silently reading garbage in the kernel."""
    if envs.SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK.get():
        logger.warning(
            "PLE table: file backend device check skipped by "
            "SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK"
        )
        return
    supported = device_uses_host_page_tables(device_index)
    if supported is None:
        raise RuntimeError(
            "--ple-offload-backend file: could not query "
            "cudaDevAttrPageableMemoryAccessUsesHostPageTables. Set "
            "SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK=1 only if you know the "
            "device reads pageable host memory through the host page tables."
        )
    if not supported:
        raise ValueError(
            "--ple-offload-backend file needs a device whose pageable host "
            "memory accesses go through the host page tables (unified-memory "
            "parts such as GB10). This device reports it does not; use "
            "--ple-offload-backend pinned."
        )


def default_ple_table_dir(model_path: str) -> str:
    """``$SGLANG_QWEN4_PLE_FILE_DIR/<model path>``, one directory per checkpoint."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_path).rstrip("/")).strip("_")
    return os.path.join(envs.SGLANG_QWEN4_PLE_FILE_DIR.get(), safe or "model")


def ple_table_file_name(
    shape: Sequence[int], dtype: torch.dtype, tag: Optional[str] = None
) -> str:
    """Deterministic file name so the sparse table is reused across restarts.

    ``tag`` distinguishes tables of the same shape that must not share a file,
    e.g. the vocabulary shards of different tensor-parallel ranks.
    """
    numel = 1
    for d in shape:
        numel *= int(d)
    elem = torch.empty(0, dtype=dtype).element_size()
    dims = "x".join(str(int(d)) for d in shape)
    suffix = f"_{tag}" if tag else ""
    return f"ple_table_{dims}_{str(dtype).replace('torch.', '')}_{numel * elem}B{suffix}.bin"


def device_uses_host_page_tables(device_index: int = 0) -> Optional[bool]:
    """Whether pageable host memory is directly addressable by the GPU.

    Returns None when the CUDA runtime library cannot be queried.
    """
    candidates = [ctypes.util.find_library("cudart")]
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.isdir(torch_lib):
        candidates += sorted(
            os.path.join(torch_lib, f)
            for f in os.listdir(torch_lib)
            if f.startswith("libcudart.so")
        )
    try:
        import nvidia.cuda_runtime  # type: ignore

        nv_lib = os.path.join(os.path.dirname(nvidia.cuda_runtime.__file__), "lib")
        if os.path.isdir(nv_lib):
            candidates += sorted(
                os.path.join(nv_lib, f)
                for f in os.listdir(nv_lib)
                if f.startswith("libcudart.so")
            )
    except Exception:
        pass
    for name in [c for c in candidates if c]:
        try:
            cudart = ctypes.CDLL(name)
            value = ctypes.c_int()
            rc = cudart.cudaDeviceGetAttribute(
                ctypes.byref(value),
                ctypes.c_int(
                    _CUDA_DEV_ATTR_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES
                ),
                ctypes.c_int(device_index),
            )
            if rc == 0:
                return bool(value.value)
        except OSError:
            continue
    return None


def _madvise_random(storage: torch.Tensor, nbytes: int) -> None:
    """The table is pure random access (16 rows of 160 B per token). Without
    this the kernel's readahead pulls its whole window: measured 1.4 MB of disk
    per token, ~560x the bytes actually used.

    It bounds readahead I/O only. Folios that are already in the page cache are
    still mapped in whole on a fault, which is what ``PleFileRssTrimmer``
    exists for."""
    if not _madvise(storage.data_ptr(), nbytes, _MADV_RANDOM):
        logger.warning("PLE table: madvise(MADV_RANDOM) not applied")


def _libc() -> Optional[ctypes.CDLL]:
    global _LIBC
    if _LIBC is None:
        try:
            _LIBC = ctypes.CDLL(
                ctypes.util.find_library("c") or "libc.so.6", use_errno=True
            )
        except OSError:
            return None
    return _LIBC


def _madvise(addr: int, length: int, advice: int) -> bool:
    """``madvise(2)`` on our own mapping. Advisory: never affects correctness."""
    libc = _libc()
    if libc is None:
        return False
    try:
        rc = libc.madvise(
            ctypes.c_void_p(addr), ctypes.c_size_t(length), ctypes.c_int(advice)
        )
    except Exception:
        return False
    if rc != 0:
        logger.warning(
            "PLE table: madvise(advice=%d) failed (errno %d)",
            advice,
            ctypes.get_errno(),
        )
        return False
    return True


def _mapping_rss_bytes(
    addr: int, nbytes: int, smaps_path: str = "/proc/self/smaps"
) -> Optional[int]:
    """Resident bytes of the VMAs overlapping ``[addr, addr + nbytes)``.

    Summed per mapping rather than taken from ``statm``/``smaps_rollup``: only
    the table's own residency should drive the trim, and on a unified-memory
    box the process RSS is dominated by everything else.
    """
    lo, hi = int(addr), int(addr) + int(nbytes)
    total = 0
    overlapping = False
    try:
        with open(smaps_path, "r") as f:
            for line in f:
                header = _SMAPS_HEADER.match(line)
                if header is not None:
                    start = int(header.group(1), 16)
                    end = int(header.group(2), 16)
                    overlapping = start < hi and end > lo
                elif overlapping:
                    rss = _SMAPS_RSS.match(line)
                    if rss is not None:
                        total += int(rss.group(1)) * 1024
    except OSError:
        return None
    return total


# ---------------------------------------------------------------------------
# ``checkpoint`` backend (this line, user order 2026-09-16 "das PLE bleibt auf
# disk ... ja genau so machen"): no copy of the table anywhere. The n-gram
# table already lies in the checkpoint as ``<prefix>.ngram_embedding.shard_N.weight``
# tensors (Qwen3.8-Flash-Next: 130 x [2500012, 160] bf16, 800 MB each, spread
# over 22 safetensors files). Each file is mapped read-only, whole, and the
# gather kernel gets one base pointer per shard; a global row resolves to
# (row // rows_per_shard, row % rows_per_shard). The GPU reads the pageable,
# file-backed pages through HMM -- measured on this rig 2026-09-16 (probe
# hmm_ple_probe.py: bit-exact against a CPU gather, 5.2 GB/s warm).
# ---------------------------------------------------------------------------

import json as _json
import struct as _struct


class CheckpointMappedPleTable:
    """The PLE table as base pointers into read-only mmaps of the checkpoint.

    ``shard_rows`` is the row count of every shard but the last (the loader's
    ``ceil(vocab / split_ngram_parts)``); ``bases`` holds one host address per
    shard, in shard order. The numpy memmaps are kept alive here.
    """

    def __init__(
        self,
        bases: Sequence[int],
        shard_rows: int,
        total_rows: int,
        dtype: torch.dtype,
        embedding_dim: int,
        keepalive: Sequence[object],
        files: Sequence[str],
        shard_files: Sequence[str] = (),
        shard_offsets: Sequence[int] = (),
    ) -> None:
        self.bases = tuple(int(b) for b in bases)
        self.shard_rows = int(shard_rows)
        self.total_rows = int(total_rows)
        self.dtype = dtype
        self.embedding_dim = int(embedding_dim)
        self._keepalive = tuple(keepalive)
        self.files = tuple(files)
        # Per shard: the file it lives in and the byte offset of its first
        # row in that file (the prefetcher's page arithmetic).
        self.shard_files = tuple(shard_files)
        self.shard_offsets = tuple(int(o) for o in shard_offsets)
        self._device_bases: dict = {}

    @property
    def row_bytes(self) -> int:
        return self.embedding_dim * torch.empty(0, dtype=self.dtype).element_size()

    def bases_on(self, device) -> torch.Tensor:
        key = str(device)
        t = self._device_bases.get(key)
        if t is None:
            t = torch.tensor(self.bases, dtype=torch.int64, device=device)
            self._device_bases[key] = t
        return t

    def row_ptr(self, row: int) -> int:
        """Host address of one row (CPU-side check / tests)."""
        shard, off = divmod(int(row), self.shard_rows)
        return self.bases[shard] + off * self.row_bytes


def _safetensors_header(path: str) -> Tuple[dict, int]:
    with open(path, "rb") as f:
        n = _struct.unpack("<Q", f.read(8))[0]
        header = _json.loads(f.read(n))
    return header, 8 + n


def map_ple_table_from_checkpoint(
    model_path: str,
    shard_name_pattern: str,
    *,
    shard_rows: int,
    total_rows: int,
    embedding_dim: int,
) -> CheckpointMappedPleTable:
    """Map every ``<shard_name_pattern>.shard_N.weight`` of the checkpoint.

    ``shard_name_pattern`` is the checkpoint-side prefix up to (excluding)
    ``.shard_``; the index file names the file of each shard. Row counts are
    taken from the headers and checked against ``shard_rows`` / ``total_rows``
    so a mismatched checkpoint refuses instead of gathering garbage.
    """
    import numpy as np

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        weight_map = _json.load(open(index_path))["weight_map"]
    else:
        single = os.path.join(model_path, "model.safetensors")
        header, _ = _safetensors_header(single)
        weight_map = {k: "model.safetensors" for k in header if k != "__metadata__"}
    pat = re.compile(re.escape(shard_name_pattern) + r"\.shard_(\d+)\.weight$")
    shards = sorted(
        ((int(m.group(1)), name) for name in weight_map if (m := pat.search(name))),
        key=lambda t: t[0],
    )
    if not shards:
        raise ValueError(
            f"no '{shard_name_pattern}.shard_N.weight' tensors in {index_path}"
        )
    if [i for i, _ in shards] != list(range(len(shards))):
        raise ValueError(f"PLE shards are not contiguous: {[i for i, _ in shards]}")
    mmaps: dict = {}
    headers: dict = {}
    bases = []
    shard_files: list = []
    shard_offsets: list = []
    dtype = None
    rows_seen = 0
    for idx, name in shards:
        fname = weight_map[name]
        path = os.path.join(model_path, fname)
        if path not in headers:
            headers[path] = _safetensors_header(path)
            mm = np.memmap(path, dtype=np.uint8, mode="r")
            mmaps[path] = mm
            _madvise(int(mm.ctypes.data), int(mm.shape[0]), _MADV_RANDOM)
        header, data_start = headers[path]
        info = header[name]
        st_dtype = {"BF16": torch.bfloat16, "F8_E4M3": torch.float8_e4m3fn}.get(
            info["dtype"]
        )
        if st_dtype is None:
            raise ValueError(f"PLE shard {name}: unsupported dtype {info['dtype']}")
        dtype = dtype or st_dtype
        if st_dtype != dtype:
            raise ValueError(f"PLE shards mix dtypes ({dtype} vs {st_dtype})")
        rows, dim = info["shape"]
        if int(dim) != int(embedding_dim):
            raise ValueError(f"PLE shard {name}: dim {dim} != {embedding_dim}")
        last = idx == len(shards) - 1
        if (not last and int(rows) != int(shard_rows)) or int(rows) > int(shard_rows):
            raise ValueError(
                f"PLE shard {name}: {rows} rows, loader expects {shard_rows} "
                f"per shard (split_ngram_parts)"
            )
        rows_seen += int(rows)
        off0, off1 = info["data_offsets"]
        if off1 - off0 != int(rows) * int(dim) * torch.empty(0, dtype=dtype).element_size():
            raise ValueError(f"PLE shard {name}: byte span does not match its shape")
        bases.append(int(mmaps[path].ctypes.data) + data_start + int(off0))
        shard_files.append(path)
        shard_offsets.append(data_start + int(off0))
    if rows_seen != int(total_rows):
        raise ValueError(
            f"PLE shards cover {rows_seen} rows, embedding expects {total_rows}"
        )
    logger.info(
        "PLE table: mapped %d checkpoint shards in %d files read-only "
        "(%.1f GiB, %s, %d rows/shard) -- no copy",
        len(shards),
        len(mmaps),
        rows_seen * embedding_dim * torch.empty(0, dtype=dtype).element_size() / 2**30,
        dtype,
        shard_rows,
    )
    return CheckpointMappedPleTable(
        bases=bases,
        shard_rows=shard_rows,
        total_rows=rows_seen,
        dtype=dtype,
        embedding_dim=embedding_dim,
        keepalive=list(mmaps.values()),
        files=list(mmaps),
        shard_files=shard_files,
        shard_offsets=shard_offsets,
    )


# ---- fnFL2 H29a: D's decode PLE rows warm from P's last prefill gather -------
#
# WHERE THE DECODE GATHER RUNS. ``Qwen4ExpPinnedHostEmbedding.gather`` launches
# ``_gather_ple_embedding_from_shards_kernel`` -- inside the captured verify
# graph, on the PLE layer (``ple_layer_ids``, one layer on Qwen3.8-Flash-Next),
# on the rank that owns the dense side (Form A: TP0 only, the expert workers
# never build the PLE). It dereferences the checkpoint mmap through HMM; a row
# whose page is not in THIS process's page cache mapping faults inside the
# kernel. No Python runs per replay, so a per-round host prefetch of the draft
# candidates cannot run before it (and the draft tokens exist only on the
# device until the verify has been launched).
#
# WHAT IS DIFFERENT ON D AFTER A FLIP. On a one-group boot the decoding process
# prefilled the prompt itself: the prefill gathers faulted every prompt n-gram
# row into its own mapping, and a decode that repeats prompt n-grams (code)
# reads them warm. Under Weg 2, P prefilled -- with the pread gather (#55),
# which on ZFS fills the ARC but NOT the mmap page cache -- and D never saw
# the prompt. Here D faults P's published rows (repeated rows first, then the
# last tokens' rows) into its own mapping on one background thread, bounded by
# SGLANG_WEG2_PLE_DECODE_PREFETCH_PAGES, and counts page-cache residency
# (mincore) before it touches: ``warm`` pages were in the page cache already,
# ``cold`` ones were faulted in by the touch.

_PLE_PUBLISH_POOL: Optional[ThreadPoolExecutor] = None
_PLE_WARM_LAST_MTIME: Optional[float] = None
_PLE_WARM_DEADLINE_S = 30.0


def _ple_publish_armed() -> bool:
    """P side: the switch is on and this process is not group D."""
    if not envs.SGLANG_WEG2_PLE_DECODE_PREFETCH.get():
        return False
    return os.environ.get("SGLANG_WEG2_GROUP", "").strip() != "D"


def note_ple_prefill_rows(row_ids: torch.Tensor) -> bool:
    """P side, from a prefill gather that already holds its row ids on the
    host: publish the rows worth warming (weg2/decode_warm_handoff) on one
    background thread. Returns whether anything was queued."""
    global _PLE_PUBLISH_POOL
    if not _ple_publish_armed() or row_ids.numel() == 0 or row_ids.is_cuda:
        return False
    ids = row_ids.detach().to(torch.int64).numpy().copy()
    if _PLE_PUBLISH_POOL is None:
        _PLE_PUBLISH_POOL = ThreadPoolExecutor(max_workers=1)

    def _publish() -> None:
        from sglang.srt.weg2.decode_warm_handoff import publish_ple_rows

        try:
            publish_ple_rows(ids)
        except Exception as exc:  # noqa: BLE001 -- a warm hint never fails P
            logger.debug("[H29a] PLE row publish skipped: %s", exc)

    _PLE_PUBLISH_POOL.submit(_publish)
    return True


def _mincore_page(addr: int, vec) -> int:
    """1 = in the page cache, 0 = not, -1 = not mapped (or no libc)."""
    libc = _libc()
    if libc is None:
        return -1
    rc = libc.mincore(
        ctypes.c_void_p(addr), ctypes.c_size_t(1 << _PAGE_SHIFT), vec
    )
    if rc != 0:
        return -1
    return int(vec[0]) & 1


def ple_row_pages(table: "CheckpointMappedPleTable", rows) -> list:
    """Global PLE rows -> distinct page addresses of THIS mapping, in the
    rows' order (a row that straddles a page boundary names both pages)."""
    rows = torch.as_tensor(rows, dtype=torch.int64).reshape(-1)
    rows = rows[(rows >= 0) & (rows < table.total_rows)]
    if rows.numel() == 0 or not table.bases:
        return []
    shard = torch.div(rows, table.shard_rows, rounding_mode="floor")
    keep = shard < len(table.bases)
    rows, shard = rows[keep], shard[keep]
    bases = torch.tensor(table.bases, dtype=torch.int64)
    start = bases[shard] + (rows - shard * table.shard_rows) * table.row_bytes
    end = start + (table.row_bytes - 1)
    pages = torch.stack([start >> _PAGE_SHIFT, end >> _PAGE_SHIFT], dim=1).reshape(-1)
    uniq, inverse = torch.unique(pages, return_inverse=True)
    first = torch.full((uniq.numel(),), pages.numel(), dtype=torch.int64)
    first.scatter_reduce_(0, inverse, torch.arange(pages.numel()), reduce="amin")
    ordered = uniq[torch.argsort(first)]
    return [int(p) << _PAGE_SHIFT for p in ordered.tolist()]


def warm_ple_rows(table: "CheckpointMappedPleTable", rows, max_pages: int,
                  deadline_s: float = _PLE_WARM_DEADLINE_S) -> dict:
    """Fault the pages of ``rows`` into this process's mapping. Every page is
    checked with mincore first: an unmapped address is never touched (no
    segfault can come out of a stale row id), a resident one is only
    counted. Returns the census."""
    import time as _time

    t0 = _time.monotonic()
    pages = ple_row_pages(table, rows)
    if max_pages > 0:
        pages = pages[: int(max_pages)]
    vec = (ctypes.c_ubyte * 1)()
    warm = cold = unmapped = touched = 0
    for addr in pages:
        state = _mincore_page(addr, vec)
        if state < 0:
            unmapped += 1
            continue
        if state == 1:
            warm += 1
            continue
        cold += 1
        if _time.monotonic() - t0 > deadline_s:
            continue
        ctypes.string_at(addr, 1)  # the fault: page cache + this mapping's PTE
        touched += 1
    return {
        "rows": int(torch.as_tensor(rows).numel()),
        "pages": len(pages),
        "warm": warm,
        "cold": cold,
        "unmapped": unmapped,
        "touched": touched,
        "ms": (_time.monotonic() - t0) * 1000.0,
    }


def _ple_tables_of(model) -> list:
    out = []
    for module in model.modules():
        table = getattr(module, "_ckpt_table", None)
        if isinstance(table, CheckpointMappedPleTable) and table not in out:
            out.append(table)
    return out


def start_ple_decode_warm(model, directory: Optional[str] = None,
                          background: bool = True) -> Optional[threading.Thread]:
    """D side, after the wake: warm P's published rows in every mapped PLE
    table of ``model`` on one daemon thread (``background=False`` runs it
    inline, for tests). A file already warmed (same mtime) is not warmed
    again. Logs ``PLE-DECODE-PREFETCH``."""
    global _PLE_WARM_LAST_MTIME
    from sglang.srt.weg2.decode_warm_handoff import load_ple_rows

    tables = _ple_tables_of(model)
    if not tables:
        return None
    rows, mtime = load_ple_rows(directory)
    if rows is None:
        logger.info("PLE-DECODE-PREFETCH rows=0 (no fresh ple_rows file from P)")
        return None
    if mtime == _PLE_WARM_LAST_MTIME:
        return None
    _PLE_WARM_LAST_MTIME = mtime
    max_pages = int(envs.SGLANG_WEG2_PLE_DECODE_PREFETCH_PAGES.get())

    def _run() -> None:
        for table in tables:
            try:
                c = warm_ple_rows(table, rows, max_pages)
            except Exception as exc:  # noqa: BLE001 -- a warm hint never kills D
                logger.warning("PLE-DECODE-PREFETCH failed: %s", exc)
                return
            logger.info(
                "PLE-DECODE-PREFETCH rows=%d pages=%d warm=%d cold=%d unmapped=%d "
                "touched=%d ms=%.1f (fnFL2 H29a: P's last prefill rows faulted into "
                "D's mapping; warm/cold = mincore page-cache census before the touch)",
                c["rows"], c["pages"], c["warm"], c["cold"], c["unmapped"],
                c["touched"], c["ms"],
            )

    if not background:
        _run()
        return None
    th = threading.Thread(target=_run, name="ple-decode-warm", daemon=True)
    th.start()
    return th
