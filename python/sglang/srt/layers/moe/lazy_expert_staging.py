"""#396(a): on-demand (page-fault style) materialization of COLD experts.

The load-time staging in :mod:`sglang.srt.layers.moe.expert_offload` fills two
tiers: a ``[buffer_slots, ...]`` device buffer for the resident set, and a
pinned host pool holding every cold expert. Filling that host pool means
reading and copying every cold expert's bytes at LOAD time, and on the models
this offload exists for the cold set is the overwhelming majority of the
checkpoint. A boot therefore pays the full read before it serves the first
token, even though a routed-MoE forward touches only the experts the router
picks.

This module is the on-demand alternative: the host pool is ALLOCATED at load
time (so every byte figure, ledger entry and capacity check stays exactly what
it was) but its rows are left unwritten, each row carrying a
:class:`ExpertFileRef` describing where on disk that expert's bytes live. The
row is read the first time somebody indexes it -- the fetch path's
``spill[row]`` -- and never again.

Three properties are load-bearing, and each has a hermetic falsifier:

1. **Transparency.** :class:`LazySpillPool` is indexed with ``pool[row]`` and
   answers ``numel()`` / ``element_size()`` / ``shape`` / ``dtype`` /
   ``is_pinned()`` like the tensor it replaces. The #125 prefetch/double-buffer
   path and the #394 link-proportional cold shard consume the pool through
   exactly those calls, and neither gains a branch: materialization happens
   BEHIND the accessor, not in front of it. There is deliberately no
   ``if lazy:`` in any consumer.

2. **Once per expert.** The weight loaders run on a thread pool and the fetch
   path is re-entered per wave, so several first touches of one row can race.
   Each row has its own latch: the winner reads, the losers wait on the same
   latch and then observe the finished row. Exactly one read per expert, and
   the pool is never a torn half-read.

3. **Loud on absence.** A ref whose file has vanished, shrunk, or been
   replaced raises :class:`LazyExpertUnavailable` naming the expert, the path
   and the byte range. It never leaves the row at its allocation value, which
   for a fresh ``torch.empty`` is arbitrary and for a zeroed pool would be a
   silently wrong -- and plausible-looking -- expert.

GATE: ``SGLANG_EXPERT_LAZY_STAGING`` (``environ.py``), default **False**. With
it off, :func:`lazy_expert_staging_enabled` is the only thing this module
contributes to a boot and the staging loop is the one it always was.
"""

from __future__ import annotations

import dataclasses
import os
import threading
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

__all__ = [
    "ExpertFileRef",
    "LazyExpertUnavailable",
    "LazySpillPool",
    "expert_refs_from_expert_major_tensor",
    "lazy_expert_staging_enabled",
]


class LazyExpertUnavailable(RuntimeError):
    """A cold expert's backing bytes could not be read on first touch.

    Deliberately a hard error rather than a fallback. The alternatives -- a
    zero row or an uninitialized row -- both produce a model that answers, and
    a MoE that answers with one dead expert is a quality regression nobody
    traces back to a missing file three hours later.
    """


def lazy_expert_staging_enabled() -> bool:
    """The ``SGLANG_EXPERT_LAZY_STAGING`` gate, read at call time.

    Read through :mod:`sglang.srt.environ` when it is importable (the
    production path) and from the raw environment otherwise, so this module
    stays usable from a desk test with no sglang runtime constructed.
    """
    try:
        from sglang.srt.environ import envs

        return bool(envs.SGLANG_EXPERT_LAZY_STAGING.get())
    except Exception:  # noqa: BLE001 - desk fallback, never a silent default
        raw = os.environ.get("SGLANG_EXPERT_LAZY_STAGING", "")
        return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclasses.dataclass(frozen=True)
class ExpertFileRef:
    """Where one expert's bytes live: a file, an offset, and a length.

    ``shape`` and ``dtype`` describe the row this range decodes to, and are
    checked against the destination row on every read -- a checkpoint swapped
    under a running process is exactly the case the loud-failure rule exists
    for, and a shape mismatch is the cheapest signal of it.

    Frozen and comparable, so a test can assert the ref SET a staging pass
    built without caring about the order it built them in.
    """

    path: str
    offset: int
    nbytes: int
    shape: Tuple[int, ...]
    dtype: Any

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError(f"expert ref offset {self.offset} is negative")
        if self.nbytes <= 0:
            raise ValueError(f"expert ref nbytes {self.nbytes} is not positive")

    def read_bytes(self) -> bytes:
        """The raw range, or :class:`LazyExpertUnavailable` naming the defect.

        ``pread`` rather than seek+read: the same fd would otherwise need a
        lock across the whole read to be thread-safe, and the per-row latch
        already covers the only sharing that matters. A short read is an
        error, not a partial row -- the caller has no way to tell a truncated
        file from a fast one.
        """
        try:
            fd = os.open(self.path, os.O_RDONLY)
        except OSError as exc:
            raise LazyExpertUnavailable(
                f"cold expert bytes at {self.path}:{self.offset}"
                f"+{self.nbytes} could not be opened ({exc}); the checkpoint "
                f"backing SGLANG_EXPERT_LAZY_STAGING must stay readable for "
                f"the life of the process"
            ) from exc
        try:
            blob = os.pread(fd, self.nbytes, self.offset)
        except OSError as exc:
            raise LazyExpertUnavailable(
                f"cold expert bytes at {self.path}:{self.offset}"
                f"+{self.nbytes} could not be read ({exc})"
            ) from exc
        finally:
            os.close(fd)
        if len(blob) != self.nbytes:
            raise LazyExpertUnavailable(
                f"cold expert bytes at {self.path}:{self.offset} are short: "
                f"wanted {self.nbytes} B, got {len(blob)} B -- the file was "
                f"truncated or replaced after load"
            )
        return blob

    def read_into(self, row) -> None:
        """Decode this range into ``row`` (a tensor view), checking geometry."""
        import torch

        if tuple(row.shape) != tuple(self.shape):
            raise LazyExpertUnavailable(
                f"cold expert row shape {tuple(row.shape)} does not match the "
                f"ref's {tuple(self.shape)} for {self.path}:{self.offset}"
            )
        if row.dtype != self.dtype:
            raise LazyExpertUnavailable(
                f"cold expert row dtype {row.dtype} does not match the ref's "
                f"{self.dtype} for {self.path}:{self.offset}"
            )
        expect = int(row.numel()) * int(row.element_size())
        if expect != self.nbytes:
            raise LazyExpertUnavailable(
                f"cold expert row is {expect} B but the ref claims "
                f"{self.nbytes} B for {self.path}:{self.offset}"
            )
        blob = self.read_bytes()
        # frombuffer over a bytes object gives a read-only-backed tensor; the
        # copy_ below is what puts the bytes in the pinned pool, so the
        # temporary's writability never matters.
        flat = torch.frombuffer(bytearray(blob), dtype=self.dtype)
        row.copy_(flat.view(self.shape))


def expert_refs_from_expert_major_tensor(
    path: str,
    data_offset: int,
    nbytes: int,
    num_experts: int,
    row_shape: Sequence[int],
    dtype: Any,
    expert_ids: Optional[Iterable[int]] = None,
) -> Dict[int, ExpertFileRef]:
    """Per-expert refs for one EXPERT-MAJOR tensor stored contiguously.

    This is the layout every door in the offload already depends on:
    ``stage_experts_into_tiers`` copies "an expert's bytes whole" precisely
    because the expert axis is the outermost axis and the one axis with no
    quantization-block structure on it (a GGUF Q4_K row is 144 B per 256
    values, so any other split cuts a block in half). The same fact is what
    makes expert ``i``'s byte range a pure function of the tensor's own
    offset: ``data_offset + i * (nbytes // num_experts)``.

    Refusing a non-divisible tensor is the point of the check below -- a
    tensor whose expert stride is not exact is not expert-major in the sense
    this function means, and computing a ref for it would hand back a range
    that straddles two experts.
    """
    E = int(num_experts)
    if E <= 0:
        raise ValueError(f"num_experts must be positive, got {E}")
    total = int(nbytes)
    if total % E:
        raise ValueError(
            f"expert-major tensor of {total} B does not divide into {E} "
            f"experts; its expert stride is not exact, so per-expert byte "
            f"ranges cannot be derived from the tensor offset alone"
        )
    stride = total // E
    shape = tuple(int(d) for d in row_shape)
    ids = range(E) if expert_ids is None else expert_ids
    refs: Dict[int, ExpertFileRef] = {}
    for e in ids:
        e = int(e)
        if e < 0 or e >= E:
            raise ValueError(f"expert id {e} out of range [0,{E})")
        refs[e] = ExpertFileRef(
            path=path,
            offset=int(data_offset) + e * stride,
            nbytes=stride,
            shape=shape,
            dtype=dtype,
        )
    return refs


class _RowLatch:
    """One expert's once-latch: the winner reads, everybody else waits."""

    __slots__ = ("event", "error", "claimed")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.error: Optional[BaseException] = None
        self.claimed = False


class LazySpillPool:
    """A pinned host cold pool whose rows materialize on first touch.

    Substitutable for the ``[len(spill_ids), ...]`` tensor
    ``stage_experts_into_tiers`` returns: the consumers index it (``pool[row]``
    in the fetch path), size it (``numel`` / ``element_size`` for the released
    -bytes ledger) and describe it (``shape`` / ``dtype``). All of those answer
    from the underlying storage, which is allocated -- and therefore already
    counted -- the moment this object exists. Only the CONTENT is deferred.

    What is DELIBERATELY not proxied is the address-taking surface --
    ``data_ptr()``, ``is_contiguous()``, and the tensor type itself. A consumer
    that wants the pool's raw address wants to read it without asking, which
    for a lazy tier means reading rows nobody has materialized. Leaving those
    absent turns such a consumer into an ``AttributeError`` at wiring time
    instead of uninitialized bytes at serving time; the one real case,
    ``device_view_of_pinned``'s CUDA-graph view, is refused by name at
    ``expert_offload.py``'s copy of that check.

    Row indices are POOL rows, the same ``[0, len(spill_ids))`` space the eager
    pool uses; ``spill_ids[row]`` is the expert that row holds, which is what
    the ref table is keyed by. Keeping the ref table expert-keyed rather than
    row-keyed means a #394 cold shard (which changes WHICH experts a rank's
    pool holds, not the row order) needs no re-keying.
    """

    def __init__(
        self, storage, spill_ids: Sequence[int], refs: Dict[int, ExpertFileRef]
    ):
        ids = tuple(int(e) for e in spill_ids)
        if int(storage.shape[0]) != len(ids):
            raise ValueError(
                f"lazy spill pool storage has {int(storage.shape[0])} rows but "
                f"{len(ids)} spill ids were planned"
            )
        missing = [e for e in ids if e not in refs]
        if missing:
            raise ValueError(
                f"lazy staging has no file ref for cold experts {missing}; a "
                f"row with no ref could never be materialized, which is a load"
                f"-time defect and must not become a first-token failure"
            )
        self._storage = storage
        self._spill_ids = ids
        self._refs = dict(refs)
        self._latches = [_RowLatch() for _ in ids]
        self._lock = threading.Lock()
        #: Instrumentation, and the falsifier's only observation point: how
        #: many DISK reads this pool has performed. Under the once-latch it
        #: must equal the number of distinct rows ever touched, no matter how
        #: many threads touched them.
        self.disk_reads = 0

    # -- tensor-shaped surface (what the consumers actually call) ----------

    @property
    def shape(self):
        return self._storage.shape

    @property
    def dtype(self):
        return self._storage.dtype

    @property
    def device(self):
        return self._storage.device

    def numel(self) -> int:
        return self._storage.numel()

    def element_size(self) -> int:
        return self._storage.element_size()

    def is_pinned(self) -> bool:
        return bool(self._storage.is_pinned())

    def __len__(self) -> int:
        return len(self._spill_ids)

    def __getitem__(self, row):
        """The accessor. Materializes ``row`` if this is its first touch."""
        if isinstance(row, slice):
            for r in range(*row.indices(len(self._spill_ids))):
                self._materialize(r)
            return self._storage[row]
        self._materialize(int(row))
        return self._storage[int(row)]

    # -- introspection for tests and the offload ledger --------------------

    @property
    def spill_ids(self) -> Tuple[int, ...]:
        return self._spill_ids

    @property
    def storage(self):
        """The backing tensor. For code that must NOT trigger a read.

        The only legitimate users are accounting (bytes) and teardown. Reading
        content through here is reading uninitialized memory.
        """
        return self._storage

    def materialized_rows(self) -> Tuple[int, ...]:
        return tuple(r for r in range(len(self._latches)) if self.is_materialized(r))

    def is_materialized(self, row: int) -> bool:
        """True only for rows that HOLD their expert's bytes.

        A row whose read raised is settled but not materialized: its latch is
        set (so the waiters get the error rather than hanging) while its
        storage still holds allocation garbage. Reporting it as materialized
        would be the exact "silent zeros" answer this module refuses.
        """
        latch = self._latches[int(row)]
        return latch.event.is_set() and latch.error is None

    # -- the latch --------------------------------------------------------

    def _materialize(self, row: int) -> None:
        latch = self._latches[row]
        if latch.event.is_set():
            if latch.error is not None:
                raise latch.error
            return
        with self._lock:
            mine = not latch.claimed
            latch.claimed = True
        if not mine:
            # Somebody else is reading this row. Wait for THEIR result rather
            # than reading it again: a second read would be correct but would
            # break the one-read-per-expert contract the NVMe tier is sized
            # against, and would double a cold-start stall.
            latch.event.wait()
            if latch.error is not None:
                raise latch.error
            return
        try:
            ref = self._refs[self._spill_ids[row]]
            ref.read_into(self._storage[row])
        except BaseException as exc:  # noqa: BLE001 - re-raised, never swallowed
            latch.error = exc
            latch.event.set()
            raise
        with self._lock:
            self.disk_reads += 1
        latch.event.set()
