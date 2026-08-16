"""#706: the PERSISTED canonical page -- partial writes, completeness, read cut.

``canonical_kv_page`` defines the page (Option A: every attention layer of one
token, layer-major, ``page_size == 1``). This module is the storage protocol
built on it: how three PP stages, each holding only its own layers, deposit
their slots into ONE suffix-free page file, and how any later geometry reads
its own slice back out of it.

Why this exists at all. Today a KV page key carries ``_{pp_size}_{pp_rank}``
(``hicache_storage.py``) and it carries it HONESTLY: under PP a stage's page
holds only that stage's layers, so the bytes really are stage-specific. The key
rule is not "keys always carry geometry" but "**the key carries exactly the
geometry the bytes still depend on**". Make the page complete across stages and
the suffix has nothing left to name -- which is the same argument
``dcp_owner_mode`` already used to drop the tp suffix on the token axis.

The protocol, in one paragraph. A page file is full width from the moment it is
created (``spec.page_bytes``, sparse until filled). A writer holds an exclusive
lock on the page, ``pwrite``s its own slot window at the window's GLOBAL byte
offset, and sets those bits in a small sidecar marker. The writer that sets the
last bit fsyncs the data and ``os.replace``s the page to its final ``.bin``
name, deleting the sidecars. Readers only ever open ``.bin``: an incomplete
page is not "readable with holes", it is INVISIBLE, so a half-written page can
never be served. That is the "slot bitmap + rename-on-complete" marker
``DESIGN_706_mechanism.md`` says must be built, and both halves are load
bearing -- the bitmap is how a writer knows the page is done, the rename is how
a reader is prevented from finding out too early.

Two traps this file is shaped around:

**The rank-local index trap.** A slot is addressed by the GLOBAL attention-layer
index. A stage that addresses slots with its rank-LOCAL layer ordering writes
the right NUMBER of bytes into the WRONG slots, silently -- the same failure
shape as cutting a mamba blob as one flat range. ``window_for_layers`` takes the
model's own attention-layer id list and the stage's GLOBAL layer ids, so the
index is global by construction. And when the trap is planted anyway, the
marker converts it from corruption into a permanent MISS: rank-local windows
from different stages overlap, the high slots are never written, the page never
completes, and nothing ever reads it.

**Silent partial pages.** An unwritten slot reads exactly like a legitimately
zero one. Nothing in the byte stream distinguishes them, which is why
completeness is tracked outside the bytes rather than inferred from them.

The marker deliberately lives in a SIDECAR rather than in a page header. A page
file must stay pure bytes: ``HiCacheFile.get`` reads a whole file into an
exactly-sized tensor, the LRU evictor accounts raw file bytes, and
``hicache_migrate`` parses the store as bare pages. A header would corrupt all
three at once, and quietly.
"""

from __future__ import annotations

import dataclasses
import errno
import fcntl
import logging
import os
import struct
from collections.abc import Sequence
from typing import Optional

import torch

from sglang.srt.mem_cache.canonical_kv_page import (
    CanonicalPageError,
    CanonicalPageSpec,
    PageCompleteness,
    attn_layer_index,
)

logger = logging.getLogger(__name__)

# Sidecar suffixes. ``.bin`` (the complete page) is the only name a reader looks
# for, so neither of these can ever be mistaken for a servable page.
PART_SUFFIX = ".part706"
MARKER_SUFFIX = ".slots706"
LOCK_SUFFIX = ".lock706"

_MARKER_MAGIC = b"SGL706\x01"
# magic | num_attn_layers (u16) | cell bytes (u32) | bitmap
_MARKER_HEADER = struct.Struct("<HI")


@dataclasses.dataclass(frozen=True)
class CanonicalPageWindow:
    """The contiguous run of GLOBAL page slots one rank writes and reads.

    A PP stage owns a contiguous run of model layers, so its attention layers
    are a contiguous run of page slots. The TP decode phase owns all of them
    (``first_slot=0, num_slots=spec.num_attn_layers``) and therefore uses the
    very same code path -- there is no second reader.
    """

    spec: CanonicalPageSpec
    first_slot: int
    num_slots: int

    def __post_init__(self) -> None:
        total = int(self.spec.num_attn_layers)
        if int(self.num_slots) <= 0:
            raise CanonicalPageError(
                f"a window must cover at least one slot, got {self.num_slots}. "
                "A rank with no attention layers has nothing to persist and "
                "must not be given a page window at all."
            )
        if not 0 <= int(self.first_slot) < total:
            raise CanonicalPageError(
                f"window starts at slot {self.first_slot}, outside the {total} "
                "slots this page carries."
            )
        if int(self.first_slot) + int(self.num_slots) > total:
            raise CanonicalPageError(
                f"window [{self.first_slot}, "
                f"{int(self.first_slot) + int(self.num_slots)}) runs past the "
                f"{total} slots this page carries."
            )

    @property
    def cell_bytes(self) -> int:
        return int(self.spec.kv_bytes_per_token_per_attn_layer)

    @property
    def byte_offset(self) -> int:
        return int(self.first_slot) * self.cell_bytes

    @property
    def byte_length(self) -> int:
        return int(self.num_slots) * self.cell_bytes

    @property
    def slots(self) -> tuple[int, ...]:
        return tuple(range(int(self.first_slot), int(self.first_slot + self.num_slots)))

    @property
    def is_whole_page(self) -> bool:
        return int(self.num_slots) == int(self.spec.num_attn_layers)


def window_for_layers(
    spec: CanonicalPageSpec,
    attn_layer_ids: Sequence[int],
    local_layer_ids: Sequence[int],
) -> CanonicalPageWindow:
    """Page window of a rank that holds ``local_layer_ids``.

    ``attn_layer_ids`` is the MODEL's full attention-layer id list (global, in
    model order); ``local_layer_ids`` are this rank's attention layers, given by
    their GLOBAL ids -- e.g. the keys of
    ``HybridLinearKVPool.full_attention_layer_id_mapping``, which are global
    because the list was filtered by the runner's global stage bounds, not
    renumbered.

    Passing rank-local ordinals here is THE trap of this design: on a hybrid
    checkpoint some local ordinals are themselves valid attention-layer ids, so
    the lookup succeeds and returns a window of the right size at the wrong
    offset. Nothing downstream can tell the difference -- see the module
    docstring for why the completeness marker still turns that into a miss
    rather than a wrong answer.
    """
    ids = [int(i) for i in local_layer_ids]
    if not ids:
        raise CanonicalPageError(
            "a rank with no attention layers has no page window; it must not "
            "take part in the canonical page protocol at all."
        )
    if ids != sorted(ids):
        raise CanonicalPageError(
            f"local attention layers {ids} are not in model order; a page "
            "window is a run of slots and the caller's order decides which "
            "bytes land in which slot."
        )
    slots = [attn_layer_index(i, attn_layer_ids) for i in ids]
    expected = list(range(slots[0], slots[0] + len(slots)))
    if slots != expected:
        raise CanonicalPageError(
            f"attention layers {ids} map to page slots {slots}, which is not a "
            "contiguous run. A PP stage owns a contiguous layer range, so a "
            "gap here means the layer list is not this stage's range -- "
            "refusing rather than writing a page nobody can cut."
        )
    if len(attn_layer_ids) != int(spec.num_attn_layers):
        raise CanonicalPageError(
            f"the model has {len(attn_layer_ids)} attention layers but the "
            f"page spec describes {spec.num_attn_layers} slots. The spec IS "
            "the layout contract: one key must name one set of bytes for every "
            "geometry, so a mismatch here is a different page format."
        )
    return CanonicalPageWindow(spec=spec, first_slot=slots[0], num_slots=len(slots))


def local_attention_layer_ids(device_pool, attn_layer_ids: Sequence[int]) -> list[int]:
    """This rank's attention layers, by their GLOBAL model ids.

    The one question the whole protocol turns on, so it is answered from the
    pool that KNOWS the answer rather than reconstructed from a layer count:

    1. ``full_attention_layer_id_mapping`` (``HybridLinearKVPool``) is a map
       from GLOBAL attention-layer id to the rank's local index. Its keys are
       global because the runner filtered the model's list with its own stage
       bounds -- ``model_runner_kv_cache_mixin`` -- and never renumbered them.
    2. ``start_layer``/``end_layer`` bounds, but ONLY when they cannot be a
       zero-based local renumbering: either they start above zero, or they span
       the whole model.
    3. A pool that holds ALL of the model's attention layers (no PP split, or
       the TP decode phase) owns the whole list.

    Anything else RAISES, and rule 2's restriction is the reason. A KV pool
    counts ATTENTION layers, so a middle PP stage's local pool reports
    ``start_layer=0, end_layer=5`` while its global range is ``[28, 48)``.
    Filtering the model's attention layers by ``[0, 5)`` yields layer 3 -- one
    plausible layer, at the wrong offset, silently. The two cases are
    indistinguishable from the pool alone, so a zero-based range that does not
    span the model is REFUSED rather than resolved. That refusal also catches
    the honest-but-unanswerable case (a dense model's PP stage 0, whose global
    range genuinely does start at zero): loud, with the reason, instead of a
    page whose slots are off by a stage.
    """
    ids = [int(i) for i in attn_layer_ids]
    mapping = getattr(device_pool, "full_attention_layer_id_mapping", None)
    if mapping is None:
        inner = getattr(device_pool, "full_kv_pool", None)
        mapping = getattr(inner, "full_attention_layer_id_mapping", None)
    if mapping:
        return sorted(int(i) for i in mapping.keys())

    start = getattr(device_pool, "start_layer", None)
    end = getattr(device_pool, "end_layer", None)
    if start is not None and end is not None and int(end) > int(start):
        spans_model = int(end) >= ids[-1] + 1
        if int(start) > 0 or spans_model:
            within = [i for i in ids if int(start) <= i < int(end)]
            if within:
                return within

    layer_num = getattr(device_pool, "layer_num", None)
    if layer_num is not None and int(layer_num) == len(ids):
        return ids

    raise CanonicalPageError(
        f"cannot establish which of the model's {len(ids)} attention layers "
        f"{type(device_pool).__name__} holds (start_layer={start}, "
        f"end_layer={end}, layer_num={layer_num}). The canonical page addresses "
        "slots by GLOBAL attention-layer index, and a zero-based range that "
        "does not span the model is indistinguishable from a rank-local "
        "renumbering -- which would deposit the right number of bytes in the "
        "wrong slots. Refusing to guess."
    )


def host_page_bytes(host_pool) -> int:
    """Bytes of the flat page this host pool reads and writes.

    Taken from ``get_dummy_flat_data_page`` -- the very object the storage path
    hands to the backend -- rather than from ``get_size_per_token``, whose
    meaning differs between pool classes (per token across layers for the MHA
    host pool, per layer for the V4 paged one). The page is the contract; a
    derived number that disagrees with it would size the window against
    something nobody writes.
    """
    page = host_pool.get_dummy_flat_data_page()
    return int(page.numel()) * int(page.element_size())


def build_page_window(
    attn_layer_ids: Sequence[int], device_pool, host_pool
) -> CanonicalPageWindow:
    """This rank's window in the canonical page, from the live pools.

    Also the place where a rank whose per-layer KV size is not the canonical
    one is caught: the flat page it writes must be exactly its slot count times
    the page's cell size, or its bytes are not the canonical form and the
    suffix-free key would name something else.
    """
    ids = [int(i) for i in attn_layer_ids]
    local = local_attention_layer_ids(device_pool, ids)
    page_bytes = host_page_bytes(host_pool)
    if page_bytes % len(local):
        raise CanonicalPageError(
            f"this rank's flat KV page is {page_bytes} bytes over {len(local)} "
            "attention layers, which does not divide evenly. The canonical page "
            "is layer-major with one cell per layer, so an indivisible page is "
            "a different format."
        )
    cell = page_bytes // len(local)
    spec = CanonicalPageSpec(
        num_attn_layers=len(ids), kv_bytes_per_token_per_attn_layer=cell
    )
    window = window_for_layers(spec, ids, local)
    if window.byte_length != page_bytes:
        raise CanonicalPageError(
            f"window [{window.first_slot}, {window.first_slot + window.num_slots}) "
            f"is {window.byte_length} bytes but this rank's page is {page_bytes}."
        )
    return window


@dataclasses.dataclass(frozen=True)
class SliceWriteResult:
    """Outcome of depositing one window into a page."""

    completed: bool  # this write set the last missing slot
    already_complete: bool  # the page was whole before this call
    missing: tuple[int, ...]  # slots still absent afterwards


def part_path(final_path: str) -> str:
    return final_path + PART_SUFFIX


def marker_path(final_path: str) -> str:
    return final_path + MARKER_SUFFIX


def lock_path(final_path: str) -> str:
    return final_path + LOCK_SUFFIX


def _bitmap_bytes(num_slots: int) -> int:
    return (int(num_slots) + 7) // 8


def _encode_marker(spec: CanonicalPageSpec, written: set[int]) -> bytes:
    bitmap = bytearray(_bitmap_bytes(spec.num_attn_layers))
    for slot in written:
        bitmap[int(slot) // 8] |= 1 << (int(slot) % 8)
    return (
        _MARKER_MAGIC
        + _MARKER_HEADER.pack(
            int(spec.num_attn_layers), int(spec.kv_bytes_per_token_per_attn_layer)
        )
        + bytes(bitmap)
    )


def _decode_marker(blob: bytes, spec: CanonicalPageSpec) -> Optional[set[int]]:
    """Slots recorded as present, or ``None`` when the marker is unusable.

    ``None`` covers a truncated marker and a marker written under a DIFFERENT
    page geometry. Both mean the partial page on disk cannot be reasoned about,
    and since a page is only ever a cache entry the answer is to discard it and
    start again -- loudly, never by reinterpreting foreign bytes.
    """
    head = len(_MARKER_MAGIC) + _MARKER_HEADER.size
    if len(blob) < head or blob[: len(_MARKER_MAGIC)] != _MARKER_MAGIC:
        return None
    num_slots, cell = _MARKER_HEADER.unpack(blob[len(_MARKER_MAGIC) : head])
    if num_slots != int(spec.num_attn_layers) or cell != int(
        spec.kv_bytes_per_token_per_attn_layer
    ):
        return None
    bitmap = blob[head:]
    if len(bitmap) != _bitmap_bytes(num_slots):
        return None
    return {slot for slot in range(num_slots) if bitmap[slot // 8] & (1 << (slot % 8))}


def _as_bytes(payload: torch.Tensor) -> memoryview:
    if payload.dtype != torch.uint8:
        payload = payload.view(torch.uint8)
    return memoryview(payload.contiguous().numpy()).cast("B")


class _PageLock:
    """Exclusive lock over one page, held across processes.

    The three PP stages are three PROCESSES writing disjoint ranges of one file
    plus a shared marker; the read-modify-write of that marker is what has to be
    serialised, and a lock file is the only primitive all three share. The lock
    is per PAGE, so writers of different pages never wait on each other.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._fd = -1

    def __enter__(self) -> _PageLock:
        self._fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc) -> None:
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = -1


def _unlink_quiet(path: str) -> None:
    try:
        os.unlink(path)
    except OSError as e:
        if e.errno != errno.ENOENT:
            logger.warning("Could not remove %s: %s", path, e)


def write_slice(
    final_path: str,
    window: CanonicalPageWindow,
    payload: torch.Tensor,
    *,
    fsync: bool = True,
) -> SliceWriteResult:
    """Deposit one window's bytes into the canonical page at ``final_path``.

    Returns without touching anything when the page is already complete: pages
    are content-addressed, so a complete page already holds exactly these bytes
    and rewriting them would only risk tearing a page a reader is using.
    """
    spec = window.spec
    data = _as_bytes(payload)
    if data.nbytes != window.byte_length:
        raise CanonicalPageError(
            f"window [{window.first_slot}, {window.first_slot + window.num_slots}) "
            f"expects {window.byte_length} bytes, got {data.nbytes}. A rank whose "
            "per-layer KV size differs from the page spec is not writing the "
            "canonical form -- refusing rather than padding."
        )

    if os.path.exists(final_path):
        return SliceWriteResult(completed=False, already_complete=True, missing=())

    part = part_path(final_path)
    marker = marker_path(final_path)
    with _PageLock(lock_path(final_path)):
        # Re-check under the lock: another stage may have completed the page
        # between the check above and the lock being granted.
        if os.path.exists(final_path):
            return SliceWriteResult(completed=False, already_complete=True, missing=())

        written: set[int] = set()
        if os.path.exists(part):
            try:
                with open(marker, "rb") as f:
                    decoded = _decode_marker(f.read(), spec)
            except FileNotFoundError:
                decoded = None
            if decoded is None:
                logger.warning(
                    "Discarding partial canonical page %s: its completeness "
                    "marker is missing or describes a different page geometry.",
                    os.path.basename(part),
                )
                _unlink_quiet(part)
            else:
                written = decoded

        fd = os.open(part, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            if os.fstat(fd).st_size != spec.page_bytes:
                # Full width from creation: every writer addresses absolute slot
                # offsets, so the file cannot grow into its own layout.
                os.ftruncate(fd, spec.page_bytes)
            os.pwrite(fd, data, window.byte_offset)

            completeness = PageCompleteness(spec)
            for slot in written:
                completeness.mark(slot)
            for slot in window.slots:
                # Re-writing a slot this rank already wrote is legitimate (a
                # crashed run resumes; the bytes are content-addressed and
                # identical). ``PageCompleteness`` treats a repeat as a layout
                # bug, which is the right rule INSIDE one process and the wrong
                # one across a restart, so the persisted marker is idempotent
                # and the strict object is only fed slots it has not seen.
                if slot not in written:
                    completeness.mark(slot)
            now_written = set(written) | set(window.slots)
            missing = completeness.missing()

            if not completeness.is_complete():
                with open(marker, "wb") as f:
                    f.write(_encode_marker(spec, now_written))
                return SliceWriteResult(
                    completed=False, already_complete=False, missing=missing
                )

            if fsync:
                # The page becomes visible by rename, so the DATA must reach the
                # medium before the name does; otherwise a crash can publish a
                # complete-looking page with unwritten holes. The directory
                # entry itself is deliberately not synced: losing the rename
                # costs a cache miss, which is free.
                os.fsync(fd)
        finally:
            os.close(fd)

        os.replace(part, final_path)
        _unlink_quiet(marker)
        return SliceWriteResult(completed=True, already_complete=False, missing=())


def read_slice(final_path: str, window: CanonicalPageWindow, out: torch.Tensor) -> bool:
    """Read this window's slots out of a COMPLETE canonical page.

    False means "not served": the page does not exist, or exists only as an
    incomplete ``.part706``. There is no third answer -- a partial page is never
    handed back, whatever slots the caller happens to need, because a page that
    is missing another stage's layers is a prefix nobody can continue from.
    """
    expected = window.byte_length
    buf = _as_bytes(out)
    if buf.nbytes != expected:
        raise CanonicalPageError(
            f"read target holds {buf.nbytes} bytes but window "
            f"[{window.first_slot}, {window.first_slot + window.num_slots}) is "
            f"{expected} bytes."
        )
    try:
        fd = os.open(final_path, os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        size = os.fstat(fd).st_size
        if size != window.spec.page_bytes:
            logger.warning(
                "Canonical page %s is %d bytes, expected %d; refusing to cut a "
                "slice out of a page of the wrong width.",
                os.path.basename(final_path),
                size,
                window.spec.page_bytes,
            )
            return False
        got = 0
        while got < expected:
            chunk = os.pread(fd, expected - got, window.byte_offset + got)
            if not chunk:
                break
            buf[got : got + len(chunk)] = chunk
            got += len(chunk)
    finally:
        os.close(fd)
    if got != expected:
        logger.warning(
            "Short read of canonical page %s: %d of %d bytes.",
            os.path.basename(final_path),
            got,
            expected,
        )
        return False
    return True


def page_is_complete(final_path: str) -> bool:
    """True when a servable page exists. Incomplete pages are invisible."""
    return os.path.exists(final_path)


def missing_slots(final_path: str, spec: CanonicalPageSpec) -> tuple[int, ...]:
    """Slots still absent from a page, for diagnostics and tests.

    ``()`` for a complete page; every slot for one that does not exist at all.
    """
    if os.path.exists(final_path):
        return ()
    try:
        with open(marker_path(final_path), "rb") as f:
            decoded = _decode_marker(f.read(), spec)
    except FileNotFoundError:
        decoded = None
    if decoded is None:
        return tuple(range(int(spec.num_attn_layers)))
    completeness = PageCompleteness(spec)
    for slot in decoded:
        completeness.mark(slot)
    return completeness.missing()
