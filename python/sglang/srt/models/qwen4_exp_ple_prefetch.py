"""fnFL2 H32: the PLE pread gather of a chunked prefill, one chunk ahead.

THE COST. With ``SGLANG_QWEN4_PLE_CKPT_GATHER=pread`` (Task #55) every
prefill-sized PLE gather is a host gather: one ``preadv`` per row out of the
checkpoint shards into a staging buffer, then one H2D copy. The forward waits
on it at the (single) PLE layer, with the GPU idle. x135 (24.09., PP0 = 5090,
16384-token chunks): ``PLE-GATHER-PREFILL rows=262144 ... ms=1154-1558`` per
chunk, ``ple_ms`` 1177-1582 of a ~5.6 s forward; PP1/PP2 carry no PLE layer.

THE FIX. The token ids of the whole request are known from its admission, so
the rows of chunk n+1 can be read while chunk n computes:

* the scheduler publishes, per extend batch, the NEXT chunk's tokens of the
  batch's chunked request (:func:`publish_ple_next_chunk`, one call in
  ``Scheduler._run_batch_forward``);
* the PLE gather serves its own chunk (joining the prefetch that was started
  for it one forward earlier; rows the prediction got wrong are read on the
  spot), copies it to the device, then hashes the published tokens with the
  layer's own n-gram hash (:func:`ple_ngram_lookup_ids`, the model's
  arithmetic on the host) and starts reading those rows into the OTHER slot of
  a two-slot ring;
* the reads run in worker PROCESSES (``qwen4_exp_ple_pread_worker.py``): the
  gather is one Python call per row, and done on threads of the scheduler
  process it starves the scheduler thread of the GIL (desk 24.09.: a 20000-op
  torch loop 0.054 s alone, 1.495 s beside the threaded gather, 0.060 s beside
  the same gather in a child process). The gather that is NOT hidden (a
  request's first chunk) stays where it was: 262144 rows read in 1.36-1.45 s
  on 4 processes against 1.41-1.59 s on 32 in-process threads (desk, ARC-warm
  checkpoint file) -- it is bound by ZFS, not by the GIL.

Correctness does not rest on the prediction: every row of the served chunk is
compared with the predicted id at its position, and only equal rows are taken
from the prefetch; everything else is read for this chunk. Bytes are those of
the serial gather (same shard/offset rule, out-of-range rows zero).

``SGLANG_QWEN4_PLE_PREFETCH=0`` keeps the in-process serial gather unchanged.
Host RAM: two memfd slots of one chunk's rows each (16384 tokens x 16 rows x
320 B = 80 MiB per slot, page-locked), which replace the serial gather's own
80 MiB staging buffer (never allocated while this runs): +80 MiB, plus the
workers' interpreter RSS (stdlib only, no torch).
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import select
import struct
import subprocess
import sys
import time
from array import array
from typing import Callable, List, NamedTuple, Optional, Sequence

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

PLE_PREFETCH_DEPTH = 2
PLE_PREFETCH_LEAD_TOKENS = 8
_KEY_SHIFT = 48
_REQ = struct.Struct("<IQIIQ")
_REP = struct.Struct("<QiId")
_HELLO = 0x504C4531
_KIND_GATHER = 1
_KIND_MAP = 2
_KIND_QUIT = 3
_F_SETPIPE_SZ = 1031
_PIPE_BYTES = 1 << 20
_CUDA_HOST_REGISTER_PORTABLE = 1
_WORKER_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "qwen4_exp_ple_pread_worker.py"
)


# --------------------------------------------------------------------------
# The layer's n-gram hash, on the host (mirror of Qwen4ExpNGramEmbedding).
# --------------------------------------------------------------------------


class PleHashParams(NamedTuple):
    layer_multipliers: torch.Tensor
    head_vocab_sizes: torch.Tensor
    head_offsets: torch.Tensor
    heads_per_ngram: int
    ngram_size: int
    eos_token_id: int

    @classmethod
    def of(cls, emb) -> "PleHashParams":
        """Host copies of a ``Qwen4ExpNGramEmbedding``'s hash constants."""
        return cls(
            emb.layer_multipliers.detach().to("cpu", torch.long).clone(),
            emb.ngram_heads_vocab_sizes.detach().to("cpu", torch.long).clone(),
            emb.ngram_heads_offsets.detach().to("cpu", torch.long).clone(),
            int(emb.heads_per_ngram),
            int(emb.ngram_size),
            int(emb.eos_token_id),
        )


def _shift_right_ignore_eos(tensor: torch.Tensor, n: int, eos: int) -> torch.Tensor:
    """``Qwen4ExpNGramEmbedding._shift_right_ignore_eos``, op for op."""
    if n == 0:
        return tensor
    batch_size, seq_len = tensor.shape
    idx = torch.arange(seq_len, device=tensor.device, dtype=torch.long)
    eos_mask = tensor == eos
    eos_pos = torch.where(eos_mask, idx, -1)
    prev_eos_inclusive = torch.cummax(eos_pos, dim=1).values
    prev_eos = torch.cat(
        [eos_pos.new_full((batch_size, 1), -1), prev_eos_inclusive[:, :-1]], dim=1
    )
    segment_start = prev_eos + 1
    pos_in_segment = idx.unsqueeze(0) - segment_start
    src_idx = idx - n
    gather_idx = torch.clamp(src_idx, min=0).unsqueeze(0).expand(batch_size, -1)
    shifted = tensor.gather(dim=1, index=gather_idx)
    valid_mask = (pos_in_segment >= n) & (src_idx.unsqueeze(0) >= 0)
    return torch.where(valid_mask, shifted, tensor.new_full((), eos))


def ple_ngram_lookup_ids(contexts: torch.Tensor, p: PleHashParams) -> torch.Tensor:
    """[L, ngram_size] token windows -> [L, ngram_heads] lookup ids: the
    non-fused branch of ``Qwen4ExpNGramEmbedding._hash_contexts``, op for op
    (int64 wrap-around and ``remainder`` agree between host and device)."""
    contexts = contexts.to(torch.long)
    shifted = [contexts]
    for shift in range(1, p.ngram_size):
        shifted.append(_shift_right_ignore_eos(contexts, shift, p.eos_token_id))
    blocks = []
    for ngram in range(2, p.ngram_size + 1):
        start_idx = (ngram - 2) * p.heads_per_ngram
        end_idx = start_idx + p.heads_per_ngram
        mix = shifted[0] * p.layer_multipliers[0]
        for pos in range(1, ngram):
            mix = torch.bitwise_xor(mix, shifted[pos] * p.layer_multipliers[pos])
        sizes = p.head_vocab_sizes[start_idx:end_idx]
        offsets = p.head_offsets[start_idx:end_idx]
        ids = torch.remainder(mix[:, -1:].unsqueeze(-1), sizes.view(1, 1, -1))
        blocks.append((ids + offsets.view(1, 1, -1))[:, 0])
    return torch.cat(blocks, dim=-1)


def ple_chunk_windows(
    tokens: torch.Tensor, lead: int, ngram_size: int, eos_token_id: int
) -> torch.Tensor:
    """``lead`` preceding tokens + one chunk's tokens -> the chunk's [L,
    ngram_size] windows, as ``_prepare_ple_batch`` builds them from the
    request's n-gram history (the last ngram_size-1 tokens before the chunk;
    EOS before the request's first token)."""
    need = ngram_size - 1
    tokens = tokens.to(torch.long).reshape(-1)
    if lead > need:
        tokens = tokens[lead - need :]
    elif lead < need:
        tokens = torch.cat([tokens.new_full((need - lead,), eos_token_id), tokens])
    if tokens.numel() < ngram_size:
        return tokens.new_empty((0, ngram_size))
    return tokens.unfold(0, ngram_size, 1)


def ple_next_chunk_hasher(emb) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """The hasher a ``Qwen4ExpPinnedHostEmbedding`` hands its prefetcher: the
    owning n-gram embedding's constants, copied to the host on first use (the
    weights are loaded by then)."""
    params: List[PleHashParams] = []

    def _hash(tokens: torch.Tensor, lead: int) -> torch.Tensor:
        if not params:
            params.append(PleHashParams.of(emb))
        p = params[0]
        windows = ple_chunk_windows(tokens, lead, p.ngram_size, p.eos_token_id)
        return ple_ngram_lookup_ids(windows, p).reshape(-1)

    # fnFL2 H43: the admission hashes outside a forward, possibly while the
    # weights are paused -- it asks first whether the constants are on the
    # host, and the gather copies them there inside a forward.
    def _warm() -> None:
        if not params:
            params.append(PleHashParams.of(emb))

    _hash.ple_hash_ready = lambda: bool(params)
    _hash.ple_hash_warm = _warm
    return _hash


# --------------------------------------------------------------------------
# Scheduler -> model: the next chunk's tokens.
# --------------------------------------------------------------------------


class PleChunkHint(NamedTuple):
    gen: int
    cur_start: int  # token position of the chunk this forward runs (-1: unknown)
    chunk_size: int
    next_start: int  # token position of the next chunk (-1: none)
    next_tokens: Optional[torch.Tensor]  # int64, lead + next chunk
    lead: int


_CONSUMERS = 0
_HINT: Optional[PleChunkHint] = None
_GEN = 0


def register_ple_prefetch_consumer() -> None:
    global _CONSUMERS
    _CONSUMERS += 1


def unregister_ple_prefetch_consumer() -> None:
    global _CONSUMERS, _HINT
    _CONSUMERS = max(0, _CONSUMERS - 1)
    if not _CONSUMERS:
        _HINT = None


def _as_int64(seq) -> torch.Tensor:
    if isinstance(seq, array) and seq.typecode == "q" and len(seq):
        return torch.frombuffer(seq, dtype=torch.int64).clone()
    return torch.tensor(list(seq), dtype=torch.int64)


def publish_ple_next_chunk(reqs: Sequence, chunk_size: Optional[int]) -> Optional[PleChunkHint]:
    """Scheduler, before an extend batch's forward: publish the next chunk of
    the batch's chunked request (the last request whose fill does not reach
    the end of its ids). No-op unless a PLE prefetcher lives in this process."""
    global _HINT, _GEN
    if not _CONSUMERS:
        return None
    pick = None
    for req in reversed(reqs):
        rng = getattr(req, "extend_range", None)
        fill = getattr(req, "full_untruncated_fill_ids", None)
        if rng is None or fill is None:
            continue
        if pick is None:
            pick = (rng, fill)
        if rng.end < len(fill):
            pick = (rng, fill)
            break
    _GEN += 1
    size = int(chunk_size) if chunk_size and int(chunk_size) > 0 else 0
    if pick is None:
        _HINT = PleChunkHint(_GEN, -1, size, -1, None, 0)
        return _HINT
    rng, fill = pick
    if size <= 0:
        size = int(rng.end - rng.start)
    nxt, lead, nstart = None, 0, -1
    if rng.end < len(fill) and size > 0:
        nstart = int(rng.end)
        lead = min(nstart, PLE_PREFETCH_LEAD_TOKENS)
        nxt = _as_int64(fill[nstart - lead : min(len(fill), nstart + size)])
    _HINT = PleChunkHint(_GEN, int(rng.start), size, nstart, nxt, lead)
    return _HINT


def take_ple_next_chunk(last_gen: int) -> Optional[PleChunkHint]:
    h = _HINT
    if h is None or h.gen == last_gen:
        return None
    return h


# --------------------------------------------------------------------------
# The worker processes and their shared slots.
# --------------------------------------------------------------------------


class PleWorkerLost(RuntimeError):
    """A pread worker died, answered out of order, or failed a read."""


def _write_all(fd: int, data) -> None:
    view = memoryview(data).cast("B")
    while view:
        k = os.write(fd, view)
        view = view[k:]


def _read_exact(fd: int, n: int) -> bytes:
    parts = []
    while n:
        b = os.read(fd, n)
        if not b:
            raise PleWorkerLost("PLE pread worker closed its pipe")
        parts.append(b)
        n -= len(b)
    return b"".join(parts)


class PlePreadProcs:
    """``procs`` worker processes that read rows into ``depth`` shared slots."""

    def __init__(
        self,
        files: Sequence[str],
        row_bytes: int,
        *,
        depth: int = PLE_PREFETCH_DEPTH,
        procs: int = 4,
        threads: int = 4,
        delay_s: float = 0.0,
        slot_fds: Optional[Sequence[int]] = None,
    ) -> None:
        self.row_bytes = int(row_bytes)
        if slot_fds is not None:
            # fnFL2 H40: the owner's own memfds (dup'ed: close() closes only
            # these copies) -- the decode stage must exist, at a fixed
            # address, before any worker does
            self.slot_fds = [os.dup(int(fd)) for fd in slot_fds]
            self.depth = len(self.slot_fds)
        else:
            self.depth = int(depth)
            self.slot_fds = [os.memfd_create(f"ple-prefetch-{i}", 0) for i in range(self.depth)]
        self.slot_bytes = [0] * self.depth
        # fnFL2 H40: per worker, the destination rows of the gather in flight
        self.parts: dict = {}
        cfg = json.dumps(
            {
                "files": list(files),
                "slot_fds": self.slot_fds,
                "row_bytes": self.row_bytes,
                "threads": int(threads),
                "delay_s": float(delay_s),
            }
        ).encode()
        self._procs: List[subprocess.Popen] = []
        self._seq = 0
        self._inflight: Optional[tuple] = None  # (seq, [proc indices])
        try:
            for _ in range(max(1, int(procs))):
                p = subprocess.Popen(
                    [sys.executable, "-I", "-S", _WORKER_SCRIPT],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    pass_fds=self.slot_fds,
                    close_fds=True,
                )
                try:
                    fcntl.fcntl(p.stdin.fileno(), _F_SETPIPE_SZ, _PIPE_BYTES)
                except OSError:
                    pass
                self._procs.append(p)
                _write_all(p.stdin.fileno(), struct.pack("<I", len(cfg)) + cfg)
            for p in self._procs:
                if struct.unpack("<I", _read_exact(p.stdout.fileno(), 4))[0] != _HELLO:
                    raise PleWorkerLost("PLE pread worker sent no hello")
        except BaseException:
            self.close()
            raise

    @property
    def n_procs(self) -> int:
        return len(self._procs)

    def pids(self) -> List[int]:
        return [p.pid for p in self._procs]

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _send(self, i: int, kind: int, seq: int, slot: int, n: int, arg: int, payload=None) -> None:
        fd = self._procs[i].stdin.fileno()
        try:
            _write_all(fd, _REQ.pack(kind, seq, slot, n, arg))
            if payload is not None:
                _write_all(fd, payload)
        except (BrokenPipeError, OSError) as exc:
            raise PleWorkerLost(f"PLE pread worker {i} is gone ({exc})") from exc

    def _recv(self, i: int, seq: int) -> float:
        rseq, status, _, seconds = _REP.unpack(_read_exact(self._procs[i].stdout.fileno(), _REP.size))
        if rseq != seq:
            raise PleWorkerLost(f"PLE pread worker {i} answered seq {rseq}, expected {seq}")
        if status:
            raise PleWorkerLost(f"PLE pread worker {i} failed seq {seq} (status {status})")
        return seconds

    def map_slot(self, slot: int, nbytes: int) -> None:
        """Size slot ``slot`` to ``nbytes`` (content below the old size kept)
        and have every worker map it. Nothing may be in flight."""
        if self._inflight is not None:
            raise RuntimeError("PLE prefetch: slot remap with a gather in flight")
        os.ftruncate(self.slot_fds[slot], int(nbytes))
        seq = self._next_seq()
        for i in range(self.n_procs):
            self._send(i, _KIND_MAP, seq, slot, 0, int(nbytes))
        for i in range(self.n_procs):
            self._recv(i, seq)
        self.slot_bytes[slot] = int(nbytes)

    def submit(self, slot: int, dest: torch.Tensor, keys: torch.Tensor, step: int = 0) -> int:
        """Start reading row ``dest[j]`` of ``slot`` from ``keys[j]`` ((file
        index << 48) | offset, < 0 = zero row). Returns the sequence number.
        ``step`` > 0: rows per worker-thread task (fnFL2 H40)."""
        if self._inflight is not None:
            raise RuntimeError("PLE prefetch: a second gather submitted while one is in flight")
        n = int(dest.numel())
        # read in (file, offset) order, split into contiguous key ranges: on
        # ZFS a row read decompresses its whole record, and neighbouring rows
        # of one record then come from the dbuf cache (desk 24.09., 262144
        # rows: unsorted 6.4 s, sorted ~1.4 s on the same workers)
        keys, order = torch.sort(keys.to(torch.int64))
        dest = dest.to(torch.int64)[order]
        seq = self._next_seq()
        per = (n + self.n_procs - 1) // self.n_procs if n else 0
        used = []
        self.parts = {}
        for i in range(self.n_procs):
            lo, hi = i * per, min(n, (i + 1) * per)
            if hi <= lo:
                continue
            payload = torch.cat([dest[lo:hi], keys[lo:hi]]).to(torch.int64).contiguous().numpy()
            self._send(i, _KIND_GATHER, seq, slot, hi - lo, max(0, int(step)), payload)
            used.append(i)
            self.parts[i] = dest[lo:hi]
        self._inflight = (seq, used)
        return seq

    def collect(self, seq: int, timeout_s: float) -> tuple:
        """fnFL2 H40: wait at most ``timeout_s`` for gather ``seq``. Returns
        (worker indices that answered, slowest read time). Workers that did
        not answer stay in flight -- ``join`` drains them later; until then
        their destination rows (``parts``) may still be written."""
        if self._inflight is None or self._inflight[0] != seq:
            return [], 0.0
        pending = {self._procs[i].stdout.fileno(): i for i in self._inflight[1]}
        done: List[int] = []
        seconds = 0.0
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while pending:
            left = deadline - time.monotonic()
            readable, _, _ = select.select(list(pending), [], [], max(0.0, left))
            if not readable:
                break
            for fd in readable:
                i = pending.pop(fd)
                seconds = max(seconds, self._recv(i, seq))
                done.append(i)
        self._inflight = (seq, sorted(pending.values())) if pending else None
        return done, seconds

    def ready(self, seq: int) -> bool:
        """Whether gather ``seq`` has already finished (no wait)."""
        if self._inflight is None or self._inflight[0] != seq:
            return True
        fds = [self._procs[i].stdout.fileno() for i in self._inflight[1]]
        if not fds:
            return True
        readable, _, _ = select.select(fds, [], [], 0)
        return len(readable) == len(fds)

    def join(self, seq: int) -> float:
        """Wait for gather ``seq``; returns its read time (slowest worker)."""
        if self._inflight is None or self._inflight[0] != seq:
            return 0.0
        _, used = self._inflight
        self._inflight = None
        return max((self._recv(i, seq) for i in used), default=0.0)

    def close(self) -> None:
        for p in self._procs:
            try:
                p.stdin.close()
            except OSError:
                pass
        for p in self._procs:
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                p.kill()
            try:
                p.stdout.close()
            except OSError:
                pass
        self._procs = []
        for fd in self.slot_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self.slot_fds = []


class _Slot:
    """The owner's view of one shared slot: host tensor, pin state, last H2D."""

    def __init__(self) -> None:
        self.rows = 0
        self.bytes: Optional[torch.Tensor] = None
        self.pinned = False
        self.event = None

    def wait_copy(self) -> None:
        if self.event is not None:
            self.event.synchronize()
            self.event = None

    def unmap(self) -> None:
        self.wait_copy()
        if self.pinned and self.bytes is not None:
            torch.cuda.cudart().cudaHostUnregister(self.bytes.data_ptr())
        self.pinned = False
        self.bytes = None
        self.rows = 0


class _Pending(NamedTuple):
    seq: int
    slot: int
    ids: torch.Tensor
    vocab: tuple


def ple_row_keys(
    ids: torch.Tensor,
    in_range: torch.Tensor,
    table,
    file_index: Sequence[int],
) -> torch.Tensor:
    """Global row ids -> (file index << 48) | byte offset; -1 for rows outside
    this rank's vocabulary (the kernel's zero rows). Same shard arithmetic as
    ``PleCheckpointPreadGather.gather_into``."""
    keys = torch.full_like(ids, -1)
    if bool(in_range.any()):
        vid = ids[in_range]
        shard = torch.div(vid, table.shard_rows, rounding_mode="floor")
        local = vid - shard * table.shard_rows
        offs = torch.tensor(table.shard_offsets, dtype=torch.int64)[shard] + local * int(table.row_bytes)
        fidx = torch.tensor(file_index, dtype=torch.int64)[shard]
        keys[in_range] = (fidx << _KEY_SHIFT) | offs
    return keys


class PlePrefetchGather:
    """Drop-in for :class:`PleCheckpointPreadGather` (``wants``/``gather_into``)
    that reads in worker processes and one chunk ahead (module docstring)."""

    def __init__(
        self,
        base,
        table,
        hasher: Callable[[torch.Tensor, int], torch.Tensor],
        *,
        procs: int = 4,
        threads: int = 4,
        delay_s: float = 0.0,
    ) -> None:
        self._base = base
        self._table = table
        self._hasher = hasher
        self._n_procs = int(procs)
        self._threads = int(threads)
        self._delay_s = float(delay_s)
        self._files = list(dict.fromkeys(table.shard_files))
        pos = {p: i for i, p in enumerate(self._files)}
        self._file_index = [pos[p] for p in table.shard_files]
        self._workers: Optional[PlePreadProcs] = None
        self._slots = [_Slot() for _ in range(PLE_PREFETCH_DEPTH)]
        self._pending: Optional[_Pending] = None
        self._last_gen = -1
        self._gathers = 0
        self._disabled = False
        self.stats = {"gathers": 0, "rows": 0, "hit_rows": 0, "read_rows": 0,
                      "prefetches": 0, "wait_s": 0.0}
        register_ple_prefetch_consumer()

    # -- the PleCheckpointPreadGather interface ------------------------------
    @property
    def min_rows(self) -> int:
        return self._base.min_rows

    def wants(self, flat_ids: torch.Tensor) -> bool:
        return self._base.wants(flat_ids)

    def gather_into(self, flat_ids, out, *, vocab_start: int = 0, vocab_end: Optional[int] = None):
        if self._disabled:
            return self._base.gather_into(flat_ids, out, vocab_start=vocab_start, vocab_end=vocab_end)
        try:
            return self._gather_into(flat_ids, out, vocab_start, vocab_end)
        except (PleWorkerLost, OSError) as exc:
            logger.error(
                "PLE-PREFETCH disabled: %s -- this process gathers serially from now on", exc
            )
            self._disable()
            return self._base.gather_into(flat_ids, out, vocab_start=vocab_start, vocab_end=vocab_end)

    def close(self) -> None:
        self._disable()
        self._base.close()

    # -- internals -------------------------------------------------------------
    def _disable(self) -> None:
        if not self._disabled:
            unregister_ple_prefetch_consumer()
        self._disabled = True
        self._pending = None
        for s in self._slots:
            try:
                s.unmap()
            except Exception:  # noqa: BLE001 -- teardown of a dead prefetcher
                pass
        if self._workers is not None:
            self._workers.close()
            self._workers = None

    def _ensure_workers(self) -> PlePreadProcs:
        if self._workers is None:
            self._workers = PlePreadProcs(
                self._files,
                self._table.row_bytes,
                depth=PLE_PREFETCH_DEPTH,
                procs=self._n_procs,
                threads=self._threads,
                delay_s=self._delay_s,
            )
            logger.info(
                "PLE-PREFETCH on: %d pread worker processes x %d threads (pids %s), "
                "%d shared slots, next chunk read during the current forward",
                self._workers.n_procs, self._threads, self._workers.pids(), PLE_PREFETCH_DEPTH,
            )
        return self._workers

    def _ensure_rows(self, k: int, rows: int) -> _Slot:
        """Slot ``k`` holds at least ``rows`` rows (content kept on growth)."""
        slot = self._slots[k]
        if slot.rows >= rows:
            return slot
        w = self._ensure_workers()
        cap = max(int(rows), 1 << 15)
        nbytes = cap * self._table.row_bytes
        slot.unmap()
        w.map_slot(k, nbytes)
        t = torch.from_file(f"/proc/self/fd/{w.slot_fds[k]}", shared=True, size=nbytes, dtype=torch.uint8)
        pinned = False
        if torch.cuda.is_available():
            rc = int(torch.cuda.cudart().cudaHostRegister(t.data_ptr(), nbytes, _CUDA_HOST_REGISTER_PORTABLE))
            pinned = rc == 0
            if not pinned:
                logger.warning("PLE-PREFETCH: cudaHostRegister(slot %d, %d B) failed (%d); pageable copy", k, nbytes, rc)
        slot.bytes, slot.rows, slot.pinned = t, cap, pinned
        return slot

    def _rows_view(self, slot: _Slot, n: int) -> torch.Tensor:
        dim = self._table.embedding_dim
        return slot.bytes[: n * self._table.row_bytes].view(self._table.dtype).view(n, dim)

    def _gather_into(self, flat_ids, out, vocab_start, vocab_end):
        from sglang.srt.layers.prefill_timing import log_ple_gather
        from sglang.srt.models.qwen4_exp_ple_table import note_ple_prefill_rows

        t0 = time.monotonic()
        hint = take_ple_next_chunk(self._last_gen)
        if hint is not None:
            self._last_gen = hint.gen
        ids = flat_ids.detach().reshape(-1).cpu().to(torch.int64)
        n = int(ids.numel())
        dim = self._table.embedding_dim
        if out.numel() != n * dim or (out.dim() >= 1 and out.shape[-1] != dim):
            raise ValueError(f"PLE pread gather: output {tuple(out.shape)} does not match {n} ids x {dim}")
        if n == 0:
            return out
        if vocab_end is None:
            vocab_end = self._table.total_rows
        vocab = (int(vocab_start), int(vocab_end))
        in_range = (ids >= vocab_start) & (ids < vocab_end) & (ids < self._table.total_rows)
        note_ple_prefill_rows(ids[in_range])  # fnFL2 H29a, as the serial gather
        w = self._ensure_workers()

        wait_s = 0.0
        gather_s = 0.0
        hit = 0
        ready = "none"
        pend, self._pending = self._pending, None
        if pend is not None:
            was_ready = w.ready(pend.seq)
            tj = time.monotonic()
            gather_s += w.join(pend.seq)
            wait_s += time.monotonic() - tj
            k = pend.slot
            if pend.vocab == vocab:
                m = min(n, int(pend.ids.numel()))
                eq = ids[:m] == pend.ids[:m]
                hit = int(eq.sum())
                if hit:
                    ready = "yes" if was_ready else "no"
                    miss = torch.cat([torch.nonzero(~eq).flatten(), torch.arange(m, n)])
                else:
                    miss = torch.arange(n)
            else:
                miss = torch.arange(n)
        else:
            k = 0
            miss = torch.arange(n)
        slot = self._ensure_rows(k, n)
        if miss.numel():
            keys = ple_row_keys(ids[miss], in_range[miss], self._table, self._file_index)
            tr = time.monotonic()
            seq = w.submit(k, miss, keys)
            gather_s += w.join(seq)
            wait_s += time.monotonic() - tr
        flat_out = out.reshape(n, dim)
        flat_out.copy_(self._rows_view(slot, n), non_blocking=out.is_cuda and slot.pinned)
        if flat_out.data_ptr() != out.data_ptr():
            out.copy_(flat_out.reshape(out.shape))
        if out.is_cuda:
            slot.event = torch.cuda.Event()
            slot.event.record()

        # the next chunk, into the other slot, while this forward runs
        nxt = "none"
        if hint is not None and hint.next_tokens is not None:
            nids = self._hasher(hint.next_tokens, hint.lead).to(torch.int64).reshape(-1)
            if int(nids.numel()) >= self.min_rows:
                k2 = (k + 1) % PLE_PREFETCH_DEPTH
                other = self._slots[k2]
                other.wait_copy()
                self._ensure_rows(k2, int(nids.numel()))
                n_in = (nids >= vocab[0]) & (nids < vocab[1]) & (nids < self._table.total_rows)
                seq = w.submit(k2, torch.arange(nids.numel()), ple_row_keys(nids, n_in, self._table, self._file_index))
                self._pending = _Pending(seq, k2, nids, vocab)
                self.stats["prefetches"] += 1
                nxt = f"queued:{int(nids.numel())}"

        wall = time.monotonic() - t0
        chunk = self._gathers
        if hint is not None and hint.cur_start >= 0 and hint.chunk_size > 0:
            chunk = hint.cur_start // hint.chunk_size
        self._gathers += 1
        zero_rows = n - int(in_range.sum())
        self.stats["gathers"] += 1
        self.stats["rows"] += n
        self.stats["hit_rows"] += hit
        self.stats["read_rows"] += int(miss.numel())
        self.stats["wait_s"] += wait_s
        logger.info(
            "PLE-PREFETCH chunk=%d rows=%d ready=%s wait_ms=%.1f gather_ms=%.1f "
            "hit_rows=%d read_rows=%d host_ms=%.1f next=%s procs=%d",
            chunk, n, ready, wait_s * 1000.0, gather_s * 1000.0, hit,
            int(miss.numel()), wall * 1000.0, nxt, w.n_procs,
        )
        log_ple_gather(n, zero_rows, wall, w.n_procs * self._threads)
        return out


def make_ple_prefetch_gather(base, table, hasher, *, delay_s: float = 0.0):
    """Wrap the pread gather (``SGLANG_QWEN4_PLE_PREFETCH``, default on);
    returns ``base`` itself when the switch is off or there is nothing to wrap."""
    if base is None or hasher is None or not envs.SGLANG_QWEN4_PLE_PREFETCH.get():
        return base
    # fnFL2 H43: plus the first chunk's read from the request's admission
    from sglang.srt.models import qwen4_exp_ple_admit as _admit

    cls = _admit.PleAdmitPrefetchGather if _admit.ple_admission_wanted() else PlePrefetchGather
    return cls(
        base,
        table,
        hasher,
        procs=max(1, int(envs.SGLANG_QWEN4_PLE_PREFETCH_PROCS.get())),
        threads=max(1, int(envs.SGLANG_QWEN4_PLE_PREFETCH_THREADS.get())),
        delay_s=delay_s,
    )
