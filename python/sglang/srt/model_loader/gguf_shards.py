# SPDX-License-Identifier: Apache-2.0
"""Split-GGUF shard-set resolution (#391 blocker 2).

llama.cpp's ``gguf-split`` writes a large export as ``<stem>-00001-of-000NN.gguf``
and records ``split.no`` / ``split.count`` / ``split.tensors.count`` in EVERY
part. The parts are not equivalent:

* the FIRST part carries the full KV block (architecture, geometry, tokenizer) --
  and, for an unsloth DeepSeek V4 export, **zero tensors**;
* every later part carries tensors and a six-entry KV block that does not even
  include ``general.architecture``.

sglang's GGUF path was written for one file. Pointed at part 1 of such an export
it built a correct model skeleton and loaded nothing into it -- no error, no
warning, a fluent-nonsense server. Pointed at a later part it could not find the
architecture at all.

This module is the single place that turns "the path the user passed" into "the
ordered set of files that make up this checkpoint", so that every reader in the
loader agrees on the same answer:

* :func:`resolve_gguf_shard_paths` -- the ordered set, validated;
* :func:`gguf_metadata_path` -- the part whose KV block is authoritative;
* :func:`iter_gguf_tensors` -- one tensor stream over the whole set;
* :class:`ConsumedPageDropper` -- release the page cache behind the stream.

A file that is not part of a split set resolves to a one-element list, and every
call site then does exactly what it did before this module existed.
"""

from __future__ import annotations

import logging
import mmap as _mmap
import os
import re
import sys
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

logger = logging.getLogger(__name__)

#: ``gguf-split``'s filename convention. The index is 1-based on disk while
#: ``split.no`` in the KV block is 0-based.
_SHARD_FILENAME_RE = re.compile(
    r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$"
)

#: abspath -> resolved ordered shard list. A checkpoint does not change under a
#: running server, and the loader asks four to six times per process (name map,
#: extra-tensor probe, unquantized prefixes, weight iterator, plus the config
#: peek and the sibling reconciliation).
_RESOLVED_CACHE: Dict[str, Tuple[str, ...]] = {}


def _read_split_kv(path: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """``(split.no, split.count, split.tensors.count)`` of one file, as ints.

    Any field the file does not carry comes back as None; a plain unsplit GGUF
    carries none of them.
    """
    import gguf

    reader = gguf.GGUFReader(path, "r")

    def field(key: str) -> Optional[int]:
        entry = reader.fields.get(key)
        if entry is None:
            return None
        try:
            return int(entry.contents())
        except (TypeError, ValueError):
            return None

    return field("split.no"), field("split.count"), field("split.tensors.count")


def resolve_gguf_shard_paths(gguf_file: str) -> List[str]:
    """Every file of ``gguf_file``'s split set, in shard order.

    Returns ``[gguf_file]`` unchanged for a file that is not split -- that is
    the overwhelmingly common case and it costs one header read.

    ``gguf_file`` may be ANY part of the set, not just the first: ``split.count``
    is recorded in all of them, so a user who points at part 3 gets the same
    answer as one who points at part 1.

    Raises ``RuntimeError`` rather than returning a partial set: a split export
    that silently loses a shard loads an incomplete model, which is the failure
    this whole module exists to prevent.
    """
    abspath = os.path.abspath(gguf_file)
    cached = _RESOLVED_CACHE.get(abspath)
    if cached is not None:
        return list(cached)

    _split_no, count, _tensors = _read_split_kv(abspath)
    if count is None or count <= 1:
        _RESOLVED_CACHE[abspath] = (abspath,)
        return [abspath]

    directory, filename = os.path.split(abspath)
    match = _SHARD_FILENAME_RE.match(filename)
    if match is None:
        raise RuntimeError(
            f"GGUF {abspath} declares split.count={count}, so it is one part of "
            "a split export, but its name does not follow llama.cpp's "
            "'<stem>-00001-of-000NN.gguf' convention and the sibling parts "
            "cannot be located. Restore the original filenames, or merge the "
            "export with 'llama-gguf-split --merge'."
        )

    total_from_name = int(match.group("total"))
    if total_from_name != count:
        raise RuntimeError(
            f"GGUF {abspath} is inconsistent with itself: the filename says it "
            f"is one of {total_from_name} parts, the KV block says "
            f"split.count={count}. Refusing to guess which is right."
        )

    stem = match.group("stem")
    paths = [
        os.path.join(directory, f"{stem}-{i + 1:05d}-of-{count:05d}.gguf")
        for i in range(count)
    ]

    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise RuntimeError(
            f"GGUF split export {stem} declares {count} parts but "
            f"{len(missing)} are missing from {directory}: "
            f"{[os.path.basename(p) for p in missing]}. Loading the parts that "
            "are present would produce a model with silently missing weights."
        )

    _validate_shard_set(paths, count)
    _RESOLVED_CACHE[abspath] = tuple(paths)
    logger.info(
        "GGUF split export: resolved %d parts for %s", count, os.path.basename(abspath)
    )
    return paths


def _validate_shard_set(paths: Sequence[str], count: int) -> None:
    """Every resolved part must agree that it is part ``i`` of ``count``.

    Catches a directory that holds two different exports whose names happen to
    collide, and a part that was replaced by one from another quantization.
    """
    declared_tensor_counts = set()
    for index, path in enumerate(paths):
        split_no, split_count, tensors_total = _read_split_kv(path)
        if split_count != count or split_no != index:
            raise RuntimeError(
                f"GGUF split export: {os.path.basename(path)} should be part "
                f"{index + 1} of {count} but its KV block says part "
                f"{(split_no + 1) if split_no is not None else '?'} of "
                f"{split_count if split_count is not None else '?'}. The files "
                "in this directory are not one consistent export."
            )
        if tensors_total is not None:
            declared_tensor_counts.add(tensors_total)
    if len(declared_tensor_counts) > 1:
        raise RuntimeError(
            "GGUF split export: the parts disagree about the total tensor "
            f"count ({sorted(declared_tensor_counts)}). The files in this "
            "directory are not one consistent export."
        )


def declared_tensor_count(gguf_file: str) -> Optional[int]:
    """``split.tensors.count`` -- how many tensors the whole set should hold, or
    None for a file that does not declare it (every unsplit GGUF)."""
    _split_no, _count, tensors_total = _read_split_kv(gguf_file)
    return tensors_total


def gguf_metadata_path(gguf_file: str) -> str:
    """The part of ``gguf_file``'s set whose KV block is authoritative.

    Only the first part of a split export carries the architecture, geometry and
    tokenizer; the later parts carry six ``split.*`` entries and nothing else. A
    caller that wants metadata must read this path, whichever part it was handed.
    """
    return resolve_gguf_shard_paths(gguf_file)[0]


def iter_gguf_tensors(paths: Iterable[str]) -> Iterator:
    """One ``ReaderTensor`` stream over an ordered set of GGUF files.

    The readers are held for the life of the generator on purpose: a
    ``ReaderTensor``'s ``.data`` is a view into its reader's memory map, and the
    consumer reads it lazily.
    """
    import gguf

    readers = [gguf.GGUFReader(path, "r") for path in paths]
    for reader in readers:
        yield from reader.tensors


def gguf_tensor_names(gguf_file: str) -> set:
    """Union of the tensor names across ``gguf_file``'s whole split set."""
    return {
        str(tensor.name)
        for tensor in iter_gguf_tensors(resolve_gguf_shard_paths(gguf_file))
    }


# ---------------------------------------------------------------------------
# Releasing the page cache behind the weight stream (#391 boot-9 blocker)
# ---------------------------------------------------------------------------
#
# ``gguf.GGUFReader`` (gguf-py 0.19) maps the whole part with
# ``np.memmap(path, mode="r")`` and hands every ``ReaderTensor.data`` out as a
# view into that map. Reading the stream therefore faults the entire checkpoint
# into the page cache and NOTHING ever asks for it back: a 119 GiB export leaves
# ~48+ GiB of clean file pages behind while the loader's own anonymous memory
# (repacked payloads, the pinned host pool) is still growing.
#
# Boot attempt 8 (2026-08-01, /spinning/gpu-battery-results/2026-08-01_391_
# dsv4flash8) is the shape this exists to remove: ``memory.current`` pinned at
# 98.3 of 98.5 GiB swapless for seven minutes while the kernel traded file pages
# for anon one for one (file 81.6 -> 63.8 GiB, anon 15.1 -> 32.9 GiB), until a
# 2.4 GiB anon burst inside one 15 s window outran reclaim and the OOM killer
# fired. Steady-state anon extrapolated to ~55-57 GiB -- the load fits, the
# cache was the only thing in the way.
#
# WHICH SYSCALL -- measured on this rig, not assumed. The obvious candidate
# (``posix_fadvise(POSIX_FADV_DONTNEED)`` on the fd) is the wrong one, twice
# over:
#
# * ``invalidate_mapping_pages()`` skips a page that is still mapped into a live
#   page table, so fadvise over a range a live mmap covers frees close to
#   nothing. ``test_a2_fadvise_alone_does_not_free_a_mapped_range`` pins that.
# * On ZFS -- which is what ``/spinning`` is, and where the checkpoints live --
#   fadvise does not free the pages even after the mapping is gone. Measured:
#   64 MiB file, read through a memmap, then ``madvise(MADV_DONTNEED)`` +
#   ``posix_fadvise(DONTNEED)`` + unmap + ``posix_fadvise(DONTNEED)`` again ->
#   16384 of 16384 pages still resident by ``mincore(2)``. The same fadvise on
#   the same file BEFORE it was ever mapped drops it to zero, so the call works
#   and the mmap'd state is what defeats it.
#
# ``madvise(MADV_PAGEOUT)`` (Linux 5.4+) does work: it does not invalidate, it
# runs the range through the ordinary reclaim path -- the same path that WAS
# visibly working during boot 8, just far too late and only under pressure.
# Same measurement, same file: 16384 -> 0 resident pages. So the ladder is
#
#   1. ``madvise(MADV_PAGEOUT)`` -- the one that works here;
#   2. ``madvise(MADV_DONTNEED)`` + ``posix_fadvise(POSIX_FADV_DONTNEED)`` --
#      the classic recipe, kept for kernels older than 5.4 and filesystems
#      where the invalidate path does the job.
#
# The mapping stays valid throughout: it is a read-only ``MAP_SHARED`` mapping
# of an unmodified file, so a later access to an advised range simply re-faults
# the same bytes from disk. Nothing the stream produces can change.
#
# ``MADV_PAGEOUT`` reclaims through the rmap, so a page another TP rank still
# maps can be taken from under it. That costs the slower rank a re-read; it
# cannot cost it correctness, and the ranks walk the same tensor order within
# seconds of each other.

#: How many newly droppable bytes to accumulate before issuing the syscalls.
#: Per-tensor syscalls would be affordable (the repack runs at ~0.7 GiB/s), but
#: the advice is over a coalesced range, so batching costs nothing in coverage
#: and keeps the syscall count in the low thousands rather than the low
#: hundreds of thousands.
_DEFAULT_DROP_BATCH_BYTES = 64 << 20

#: How many released bytes between two in-stream progress lines.
#:
#: ``close()`` speaks only for a stream that reached its end. Boot 9
#: (2026-08-01) died mid-load and its log therefore said nothing at all about
#: whether the dropper had run -- ``mincore(2)`` had to answer that from
#: outside the process, against the live mapping. A periodic line puts the
#: evidence in the log BEFORE the step that fails, which is the only place it
#: is worth anything on a boot that does not finish.
_DEFAULT_DROP_PROGRESS_BYTES = 8 << 30

_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096

#: ``asm-generic/mman-common.h``. Identical on every Linux architecture sglang
#: builds for. CPython only exposes the ``MADV_*`` constants its own build host
#: had headers for, and ``MADV_PAGEOUT`` is missing from common builds, so the
#: numbers are spelled out rather than depended on.
_MADV_DONTNEED = getattr(_mmap, "MADV_DONTNEED", 4)
_MADV_PAGEOUT = getattr(_mmap, "MADV_PAGEOUT", 21)

_LIBC: Any = None
_LIBC_LOADED = False


def _libc() -> Any:
    """``libc`` with ``madvise`` bound, or None where it is not reachable."""
    global _LIBC, _LIBC_LOADED
    if _LIBC_LOADED:
        return _LIBC
    _LIBC_LOADED = True
    if not sys.platform.startswith("linux"):
        return None
    try:
        import ctypes
        import ctypes.util

        name = ctypes.util.find_library("c")
        if name is None:
            return None
        libc = ctypes.CDLL(name, use_errno=True)
        libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        libc.madvise.restype = ctypes.c_int
        _LIBC = libc
    except Exception as err:  # pragma: no cover - platform dependent
        logger.debug("GGUF stream: libc madvise unavailable: %s", err)
        _LIBC = None
    return _LIBC


def stream_drop_cache_enabled() -> bool:
    """Whether the streaming loader releases the page cache behind itself.

    ``SGLANG_GGUF_STREAM_DROP_CACHE=0`` restores the pre-#391 behaviour: the
    whole checkpoint accumulates in the page cache for the life of the load.
    """
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_GGUF_STREAM_DROP_CACHE.get())


def _page_cache_advice_available() -> bool:
    return _libc() is not None


def _madvise_addr(
    address: Optional[int],
    map_len: Optional[int],
    start: int,
    length: int,
    advice: int,
    *,
    log_path: str = "<mapping>",
) -> bool:
    """One ``madvise(address+start, length, advice)`` call. False if it did
    not run (no libc, no live mapping, or a zero-length clamp) or the kernel
    rejected it.

    Shared between ``ConsumedPageDropper`` (a long-lived mapping it already
    holds, one shard at a time) and any single-shot caller that owns a
    mapping just for the length of one call (``drop_page_cache_range``
    below).
    """
    libc = _libc()
    if libc is None or address is None:
        return False
    if map_len and start + length > map_len:
        length = max(0, map_len - start)
    if length <= 0:
        return False
    # madvise() requires a page-aligned address. GGUF's own call sites always
    # pass one already (the watermark only ever advises at page boundaries,
    # see ConsumedPageDropper._flush_shard), so this is a no-op for them; a
    # caller working from an arbitrary tensor's own data pointer (the
    # safetensors path in weight_utils.py) will not have one, so round down
    # to the enclosing page and extend the length to match.
    target = address + start
    aligned = (target // _PAGE_SIZE) * _PAGE_SIZE
    length += target - aligned
    target = aligned
    import ctypes

    rc = libc.madvise(ctypes.c_void_p(target), ctypes.c_size_t(length), advice)
    if rc != 0:
        logger.debug(
            "page cache drop: madvise(%d) on %s [%d,+%d) failed: errno %d",
            advice,
            log_path,
            start,
            length,
            ctypes.get_errno(),
        )
        return False
    return True


def _fadvise_dontneed(
    fd: int, path: str, start: int, length: int, *, tag: str = "page cache drop"
) -> None:
    """``posix_fadvise(fd, start, length, POSIX_FADV_DONTNEED)``, logged on failure.

    Classic recipe half of the fallback ladder. On its own -- over a range a
    live mapping still covers, or on this rig's ZFS pool even after that
    mapping is gone -- this is a measured no-op; it is kept only as the
    fallback for kernels without ``MADV_PAGEOUT`` (pre-5.4) and filesystems
    where ``invalidate_mapping_pages()`` does the job it advertises.
    """
    try:
        os.posix_fadvise(fd, start, length, os.POSIX_FADV_DONTNEED)
    except OSError as err:  # pragma: no cover - kernel dependent
        logger.debug(
            "%s: posix_fadvise(DONTNEED) on %s [%d,%d) failed: %s",
            tag,
            path,
            start,
            start + length,
            err,
        )


def drop_page_cache_range(
    path: str,
    start: int,
    length: int,
    *,
    address: Optional[int] = None,
    map_len: Optional[int] = None,
) -> None:
    """Single-shot release of ``[start, start+length)`` of ``path``'s page cache.

    Same ladder as ``ConsumedPageDropper._advise`` (see the module comment
    above it for the measurements this is based on), but for a caller that
    has no long-lived dropper object to amortise an fd across many calls --
    it opens one for the fadvise fallback and closes it again before
    returning.

    ``address``/``map_len`` are the base and byte length of a mapping the
    caller already has faulted PTEs in for ``[start, start+length)`` --
    ``MADV_PAGEOUT`` reclaims through the page table, so an address with no
    live, faulted mapping over the range makes step 1 a no-op and this falls
    straight through to the fadvise-only fallback (a measured no-op on this
    rig's ZFS pool; see ``weight_utils._drop_file_cache_after_load`` for how
    the safetensors path gets itself a faulted mapping to avoid exactly that).
    """
    if _madvise_addr(address, map_len, start, length, _MADV_PAGEOUT, log_path=path):
        return
    _madvise_addr(address, map_len, start, length, _MADV_DONTNEED, log_path=path)
    if not hasattr(os, "posix_fadvise"):
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as err:  # pragma: no cover - path validated by the caller
        logger.debug("page cache drop: cannot open %s for fadvise: %s", path, err)
        return
    try:
        _fadvise_dontneed(fd, path, start, length)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


_CGROUP = "/sys/fs/cgroup"


class ProgressCoupledTrim:
    """Reclaim page cache when the STREAM advances, not when a clock ticks.

    ``cachetrim.sh`` samples ``memory.current`` every few seconds and reclaims
    when it is over a watermark. That works until the load outruns the sample
    interval: window 3 of #417 saw ``memory.current`` go 88.2 -> 102.5 GiB
    inside one 15 s window, tripping a guard that a 5 s sampler had been
    holding fine on the previous boot with the same recipe. Whether a boot
    survived came down to where a noisy peak landed between two samples.

    The pressure is produced by the loader reading, so the trim is driven from
    the loader reading: :class:`ConsumedPageDropper` calls :meth:`maybe_trim`
    after every advice batch it issues. A load that streams faster therefore
    trims more often, automatically, with no interval to tune.

    Off unless ``SGLANG_GGUF_STREAM_TRIM_SOFT_GIB`` is set; when off, nothing
    in the load path changes.

    The safety argument is ``cachetrim.sh``'s and is checked, not assumed:
    with no swap, cgroup reclaim cannot evict anonymous pages, so the pinned
    host expert pool, the CUDA host allocations and the Python heap are all
    structurally out of reach and only page cache can be taken -- which the
    loader re-reads from disk if it needs it again. Worst case is a slower
    load, not a wrong one. WITH swap configured that argument fails, so this
    refuses to act.
    """

    def __init__(self) -> None:
        from sglang.srt.environ import envs

        self.soft_bytes = int(envs.SGLANG_GGUF_STREAM_TRIM_SOFT_GIB.get() * (1 << 30))
        target_gib = envs.SGLANG_GGUF_STREAM_TRIM_TARGET_GIB.get()
        self.target_bytes = int(target_gib * (1 << 30)) if target_gib else 0
        self.enabled = self.soft_bytes > 0
        self.trims = 0
        self.bytes_reclaimed = 0
        if self.enabled and not self._swapless():
            logger.warning(
                "GGUF stream: SGLANG_GGUF_STREAM_TRIM_SOFT_GIB is set but this "
                "host has swap configured, where cgroup reclaim can page out "
                "the loader's own anonymous memory and turn a fast load into a "
                "thrashing one. Progress-coupled trim disabled."
            )
            self.enabled = False
        if self.enabled and self.target_bytes <= 0:
            # A target below the soft mark is required, or every call reclaims.
            self.target_bytes = int(self.soft_bytes * 0.9)

    @staticmethod
    def _swapless() -> bool:
        try:
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("SwapTotal:"):
                        return int(line.split()[1]) == 0
        except OSError:
            return False
        return False

    @staticmethod
    def _current() -> Optional[int]:
        try:
            with open(f"{_CGROUP}/memory.current") as fh:
                return int(fh.read().strip())
        except (OSError, ValueError):
            return None

    def maybe_trim(self) -> None:
        if not self.enabled:
            return
        current = self._current()
        if current is None or current <= self.soft_bytes:
            return
        ask = current - self.target_bytes
        if ask <= 0:
            return
        try:
            with open(f"{_CGROUP}/memory.reclaim", "w") as fh:
                fh.write(str(ask))
        except OSError as err:
            # A kernel without memory.reclaim, or a cgroup that refuses: this
            # is an optimisation, never a requirement. Say so once and stop.
            logger.warning(
                "GGUF stream: progress-coupled trim could not reclaim (%s); "
                "disabling it for this load.",
                err,
            )
            self.enabled = False
            return
        after = self._current()
        if after is not None:
            self.bytes_reclaimed += max(current - after, 0)
        self.trims += 1


class ConsumedPageDropper:
    """Release the page cache of GGUF shard regions the stream has passed.

    Bookkeeping contract -- the whole point of the class, and what
    ``test_gguf_stream_page_cache.py`` pins:

    * a region is advised only once EVERY tensor that owns a byte in it has
      been marked consumed;
    * the advised range is cut at the page boundary BELOW the first
      not-yet-consumed tensor's start, so a page shared between a consumed and
      an unconsumed tensor is left alone until the second one is consumed too;
    * consumption order does not have to match file order. Registered extents
      are sorted by offset and a watermark walks them; an out-of-order or
      never-consumed tensor stalls the watermark, which loses cache release and
      cannot lose correctness.
    """

    def __init__(
        self,
        shard_paths: Sequence[str],
        mappings: Sequence[Optional[Any]],
        *,
        batch_bytes: Optional[int] = None,
        progress_bytes: Optional[int] = None,
    ) -> None:
        if batch_bytes is None:
            # Read at call time, not bound at def time, so the module global
            # stays the single knob (and the tests can turn it down).
            batch_bytes = _DEFAULT_DROP_BATCH_BYTES
        if progress_bytes is None:
            progress_bytes = _DEFAULT_DROP_PROGRESS_BYTES
        if len(shard_paths) != len(mappings):
            raise ValueError("shard_paths and mappings must have the same length")
        self._trim = ProgressCoupledTrim()
        self._paths = list(shard_paths)
        # Base address + byte length of each part's mapping. ``madvise`` needs an
        # address, not a file offset, and only the numpy view knows it: numpy
        # closes the fd right after mapping and maps from offset 0, so the
        # array's data pointer IS the mapping base.
        self._addrs: List[Optional[int]] = []
        self._map_len: List[int] = []
        for mapping in mappings:
            pointer = getattr(getattr(mapping, "ctypes", None), "data", None)
            self._addrs.append(int(pointer) if pointer else None)
            self._map_len.append(int(getattr(mapping, "nbytes", 0) or 0))
        self._batch_bytes = int(batch_bytes)
        self._fds: List[Optional[int]] = [None] * len(self._paths)
        # Per shard: extents in registration order, the registration indices
        # sorted by start offset, a consumed flag per extent, the watermark into
        # the sorted order, and how far the file has already been advised.
        self._extents: List[List[Tuple[int, int]]] = [[] for _ in self._paths]
        self._order: List[List[int]] = [[] for _ in self._paths]
        self._consumed: List[List[bool]] = [[] for _ in self._paths]
        self._cursor: List[int] = [0] * len(self._paths)
        self._advised_to: List[int] = [0] * len(self._paths)
        self._pending: List[int] = [0] * len(self._paths)
        self._sizes: List[int] = []
        for path in self._paths:
            try:
                self._sizes.append(os.path.getsize(path))
            except OSError:
                self._sizes.append(0)
        self.bytes_advised = 0
        self.calls = 0
        #: Extents marked consumed so far / registered in total, reported by
        #: the progress line as the stream's own position.
        self.tensors_consumed = 0
        self.tensors_registered = 0
        self._progress_bytes = int(progress_bytes)
        self._progress_milestone = 0

    @classmethod
    def for_readers(
        cls,
        shard_paths: Sequence[str],
        readers: Sequence[Any],
        **kwargs: Any,
    ) -> ConsumedPageDropper:
        """Build a dropper for ``gguf.GGUFReader`` objects, extents included.

        ``reader.data`` is the ``np.memmap`` of the whole part. A reader that
        does not expose one (a future gguf-py that reads instead of maps) still
        gets the ``fadvise`` half, which is the correct advice for a
        read()-based reader.
        """
        mappings = [getattr(r, "data", None) for r in readers]
        dropper = cls(shard_paths, mappings, **kwargs)
        for shard_index, reader in enumerate(readers):
            dropper.register(
                shard_index,
                [
                    (int(tensor.data_offset), int(tensor.n_bytes))
                    for tensor in reader.tensors
                ],
            )
        return dropper

    def register(self, shard_index: int, extents: Iterable[Tuple[int, int]]) -> None:
        """Declare ``(offset, nbytes)`` for every tensor of one shard."""
        stored = [(int(offset), int(nbytes)) for offset, nbytes in extents]
        previous = len(self._extents[shard_index])
        self._extents[shard_index] = stored
        self._consumed[shard_index] = [False] * len(stored)
        self._order[shard_index] = sorted(
            range(len(stored)), key=lambda i: stored[i][0]
        )
        self._cursor[shard_index] = 0
        self.tensors_registered += len(stored) - previous

    def consume(self, shard_index: int, tensor_index: int) -> None:
        """Mark one registered tensor as fully read by the consumer."""
        consumed = self._consumed[shard_index]
        if tensor_index < 0 or tensor_index >= len(consumed):
            raise IndexError(
                f"tensor index {tensor_index} is outside shard {shard_index}'s "
                f"{len(consumed)} registered extents"
            )
        if consumed[tensor_index]:
            return
        consumed[tensor_index] = True
        self.tensors_consumed += 1
        self._pending[shard_index] += self._extents[shard_index][tensor_index][1]
        if self._pending[shard_index] >= self._batch_bytes:
            self._flush_shard(shard_index)

    def flush(self) -> None:
        """Advise every region that is droppable right now, batch or not."""
        for shard_index in range(len(self._paths)):
            self._flush_shard(shard_index, force=True)

    def close(self) -> None:
        """Final flush, then release the file descriptors."""
        try:
            self.flush()
        finally:
            for index, fd in enumerate(self._fds):
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                    self._fds[index] = None
            if self.bytes_advised:
                logger.info(
                    "GGUF stream: released %.2f GiB of checkpoint page cache "
                    "behind the consumer in %d advice call(s). Set "
                    "SGLANG_GGUF_STREAM_DROP_CACHE=0 to keep it instead.",
                    self.bytes_advised / float(1 << 30),
                    self.calls,
                )

    def __enter__(self) -> ConsumedPageDropper:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # -- internals ---------------------------------------------------------

    def _first_unconsumed_start(self, shard_index: int) -> int:
        """Start offset of the lowest-offset tensor still unconsumed.

        The whole remaining file when every tensor is consumed. This is the
        ceiling the advised range is never allowed to cross.
        """
        order = self._order[shard_index]
        consumed = self._consumed[shard_index]
        cursor = self._cursor[shard_index]
        while cursor < len(order) and consumed[order[cursor]]:
            cursor += 1
        self._cursor[shard_index] = cursor
        if cursor >= len(order):
            return self._sizes[shard_index]
        return self._extents[shard_index][order[cursor]][0]

    def _flush_shard(self, shard_index: int, force: bool = False) -> None:
        ceiling = self._first_unconsumed_start(shard_index)
        # Cut at the page boundary BELOW the first unconsumed byte: GGUF packs
        # tensors to a 32-byte alignment, so consecutive tensors routinely share
        # a page and advising the partial page would drop bytes the consumer has
        # not read yet.
        end = (ceiling // _PAGE_SIZE) * _PAGE_SIZE
        start = self._advised_to[shard_index]
        if end <= start:
            self._pending[shard_index] = 0
            return
        if not force and (end - start) < self._batch_bytes:
            return
        self._advise(shard_index, start, end)
        self._advised_to[shard_index] = end
        self._pending[shard_index] = 0
        # Progress-coupled, not clock-coupled: the batch that just advanced the
        # stream is also what grew the cache ahead of it (#391 / #417 w3).
        self._trim.maybe_trim()
        self._log_progress()

    def _log_progress(self) -> None:
        """One INFO line per ``_progress_bytes`` released, in the stream.

        Reports what has been released and where the consumer stands, so a
        load that is killed before ``close()`` still leaves the evidence of
        this feature's operation in the log ahead of the failing step. Purely
        a log: the advice calls, their ranges and their order are untouched,
        so the streamed bytes are identical with the line and without it.
        """
        if self._progress_bytes <= 0:
            return
        milestone = self.bytes_advised // self._progress_bytes
        if milestone <= self._progress_milestone:
            return
        self._progress_milestone = milestone
        logger.info(
            "GGUF stream: released %.2f GiB of checkpoint page cache so far "
            "in %d advice call(s); consumer at tensor %d/%d.",
            self.bytes_advised / float(1 << 30),
            self.calls,
            self.tensors_consumed,
            self.tensors_registered,
        )

    def _madvise(self, shard_index: int, start: int, length: int, advice: int) -> bool:
        return _madvise_addr(
            self._addrs[shard_index],
            self._map_len[shard_index],
            start,
            length,
            advice,
            log_path=self._paths[shard_index],
        )

    def _advise(self, shard_index: int, start: int, end: int) -> None:
        """Drop ``[start, end)`` of one shard. Overridden by the contract test."""
        length = end - start
        self.calls += 1
        self.bytes_advised += length
        if self._madvise(shard_index, start, length, _MADV_PAGEOUT):
            return
        # Fallback ladder (see the module comment): unmap first, because
        # fadvise cannot free a page that is still mapped, then invalidate.
        self._madvise(shard_index, start, length, _MADV_DONTNEED)
        if not hasattr(os, "posix_fadvise"):
            return
        fd = self._fds[shard_index]
        if fd is None:
            try:
                fd = os.open(self._paths[shard_index], os.O_RDONLY)
            except OSError as err:  # pragma: no cover - path validated earlier
                logger.debug(
                    "GGUF stream: cannot open %s for fadvise: %s",
                    self._paths[shard_index],
                    err,
                )
                return
            self._fds[shard_index] = fd
        _fadvise_dontneed(
            fd, self._paths[shard_index], start, length, tag="GGUF stream"
        )


class _NullPageDropper:
    """No-op stand-in so the call sites stay branch-free."""

    bytes_advised = 0
    calls = 0
    tensors_consumed = 0
    tensors_registered = 0

    def consume(self, shard_index: int, tensor_index: int) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def make_stream_page_dropper(
    shard_paths: Sequence[str],
    readers: Sequence[Any],
) -> Any:
    """A dropper for this stream, or a no-op when the feature is switched off."""
    if not stream_drop_cache_enabled():
        logger.info(
            "GGUF stream: SGLANG_GGUF_STREAM_DROP_CACHE=0 -- the checkpoint's "
            "pages stay in the page cache for the whole load (pre-#391 "
            "behaviour)."
        )
        return _NullPageDropper()
    if not _page_cache_advice_available():
        logger.info(
            "GGUF stream: madvise is not reachable on this platform, the page "
            "cache is left to the kernel."
        )
        return _NullPageDropper()
    return ConsumedPageDropper.for_readers(shard_paths, readers)
