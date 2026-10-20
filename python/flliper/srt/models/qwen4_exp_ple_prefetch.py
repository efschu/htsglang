"""fnFL2 H32: the PLE pread gather of a chunked prefill, one chunk ahead.

THE COST. With ``FLLIPER_QWEN4_PLE_CKPT_GATHER=pread`` (Task #55) every
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

``FLLIPER_QWEN4_PLE_PREFETCH=0`` keeps the in-process serial gather unchanged.
Host RAM: two memfd slots of one chunk's rows each (16384 tokens x 16 rows x
320 B = 80 MiB per slot, page-locked), which replace the serial gather's own
80 MiB staging buffer (never allocated while this runs): +80 MiB, plus the
workers' interpreter RSS (stdlib only, no torch).
"""

from __future__ import annotations

import collections
import fcntl
import json
import math
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

from flliper.srt.environ import envs
from flliper.srt.pdflip import rank_timing as _rank_timing  # RANK-TIMING: rankstats ple.prefill

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
# fnFL2 H73: the autonomous decode stage (qwen4_exp_ple_pread_worker.AutoStage)
_KIND_AUTO = 4
_KIND_ROUND = 5
_F_SETPIPE_SZ = 1031
_PIPE_BYTES = 1 << 20
_CUDA_HOST_REGISTER_PORTABLE = 1
# NF-PLE-SMALL (20.10.): the slots are also read by the device (the staged
# gather of a small extend, below), so they are mapped as well -- the decode
# stage's flags (qwen4_exp_ple_decode_pread._CUDA_HOST_REGISTER_PORTABLE_MAPPED)
_CUDA_HOST_REGISTER_PORTABLE_MAPPED = 3
#: NF-PLE-SMALL: the scheduler publishes the CURRENT chunk's tokens too when the
#: chunk is at most this long, so a small extend with no prefetch behind it is
#: still predicted on the host (its rows: tokens x 16 <= 8192, the pread
#: gather's own range starts at 2048 rows = 128 tokens)
PLE_SMALL_CUR_TOKENS = 512
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
    #: NF-PLE-SMALL: THIS batch's tokens as the forward lays them out -- one
    #: (lead + the request's extend range, lead) per request in batch order
    #: (a mixed batch: the extend requests, then one row per running decode
    #: request) -- None unless they are at most PLE_SMALL_CUR_TOKENS together
    cur_parts: Optional[tuple] = None
    #: NF-PLE-SMALL review A4: how many of those tokens are running decode
    #: rows of a MIXED batch -- under the overlap schedule + spec decode their
    #: newest token is not on the host yet (schedule_batch mix_with_running),
    #: so their prediction may be one token old; counted so the kernel_hit line
    #: names them
    cur_decode_tokens: int = 0


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


#: H125: the token PLE sees at image positions (``image_token_id``), set by
#: the model when its group tokenizes images; None = ids pass untouched (every
#: boot without image tokenization).
_MM_PAD_TOKEN: Optional[int] = None


def set_mm_pad_token(token_id: Optional[int]) -> None:
    global _MM_PAD_TOKEN
    _MM_PAD_TOKEN = None if token_id is None else int(token_id)


def _as_int64(seq) -> torch.Tensor:
    if isinstance(seq, array) and seq.typecode == "q" and len(seq):
        out = torch.frombuffer(seq, dtype=torch.int64).clone()
    else:
        out = torch.tensor(list(seq), dtype=torch.int64)
    if _MM_PAD_TOKEN is not None and out.numel():
        # H125: the host mirror predicts the rows the forward will read, so
        # it maps image pad ids exactly as ple_ids_for_images does.
        from flliper.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE

        out = torch.where(out >= MM_PAD_SHIFT_VALUE, out.new_tensor(_MM_PAD_TOKEN), out)
    return out


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
    parts, n_dec = _cur_parts(reqs)
    _HINT = PleChunkHint(_GEN, int(rng.start), size, nstart, nxt, lead, parts, n_dec)
    return _HINT


def _cur_parts(reqs: Sequence) -> tuple:
    """NF-PLE-SMALL: a small batch's own tokens, laid out as the forward
    gathers them (request order, each request's extend range with its n-gram
    lead), so its PLE rows can be predicted on the host without reading the
    device's lookup ids back. None when a request's range is unknown or the
    batch is longer than PLE_SMALL_CUR_TOKENS (the pread range serves it).
    Returns (parts or None, decode tokens): a one-token range of a request
    that already has output is a running decode row of a MIXED batch."""
    parts, total, n_dec = [], 0, 0
    for req in reqs:
        rng = getattr(req, "extend_range", None)
        fill = getattr(req, "full_untruncated_fill_ids", None)
        if rng is None or fill is None:
            return None, 0
        c0, c1 = int(rng.start), min(int(rng.end), len(fill))
        if c0 < 0 or c1 <= c0:
            return None, 0
        total += c1 - c0
        if total > PLE_SMALL_CUR_TOKENS:
            return None, 0
        if c1 - c0 == 1 and len(getattr(req, "output_ids", None) or ()) > 0:
            n_dec += 1
        lead = min(c0, PLE_PREFETCH_LEAD_TOKENS)
        parts.append((_as_int64(fill[c0 - lead : c1]), lead))
    return (tuple(parts) if parts else None), n_dec


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


#: S89: what poll() reports as "this pipe has something for us" -- data, or the
#: end of it (a dead worker: HUP/ERR; a closed fd: NVAL). select() reported all of
#: these as readable; the reader then gets EOF and raises PleWorkerLost, as before.
_POLL_READY = select.POLLIN | select.POLLPRI | select.POLLHUP | select.POLLERR | select.POLLNVAL


def _poll_readable(fds: Sequence[int], timeout_s: float) -> List[int]:
    """S89: the fds of ``fds`` that are readable within ``timeout_s`` seconds.

    poll(), never select(): select() takes an fd_set of FD_SETSIZE (1024) bits and
    raises ``ValueError: filedescriptor out of range in select()`` for any fd >= 1024.
    A D rank holds thousands of fds BY DESIGN -- S36 KV-DURCHLAUF keeps one L3 file
    open per live disk page (rc19 NF D TP0 07:30:42Z: live 4070, peak 5465) and raises
    RLIMIT_NOFILE for it -- so a pipe made after that point is numbered above 1023.
    NF rc19 boot 1020_072658: the PLE-PREFETCH workers started 07:30:50Z, 8 s after
    the first DURCHLAUF burst; the first ``ready()`` with a gather in flight
    (07:43:23Z, a vision extend) killed D TP0. poll() does not care about the number."""
    fds = list(fds)
    if not fds:
        return []
    p = select.poll()
    for fd in fds:
        p.register(fd, select.POLLIN | select.POLLPRI)
    ms = int(math.ceil(max(0.0, float(timeout_s)) * 1000.0))
    return [fd for fd, ev in p.poll(ms) if ev & _POLL_READY]


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

    def configure_auto(self, slot: int, cfgs: Sequence[dict]) -> None:
        """fnFL2 H73: worker ``i`` fills its share of slot ``slot`` on its own,
        as ``cfgs[i]`` describes (``AutoStage`` of the worker script). The
        slot must be mapped; nothing may be in flight."""
        if self._inflight is not None:
            raise RuntimeError("PLE pread: auto-stage setup with a gather in flight")
        if len(cfgs) != self.n_procs:
            raise ValueError(f"auto stage: {len(cfgs)} configs for {self.n_procs} workers")
        seq = self._next_seq()
        for i, cfg in enumerate(cfgs):
            payload = json.dumps(cfg).encode()
            self._send(i, _KIND_AUTO, seq, slot, len(payload), 0, payload)
        for i in range(self.n_procs):
            self._recv(i, seq)

    def ping(self, slot: int, part: int, seq: int, bs: int, cols: int, width: int = 0) -> None:
        """fnFL2 H73: announce round ``seq`` of part ``part`` of the auto stage
        in ``slot`` (``bs`` requests of ``cols`` tokens in the mailbox, verify
        width ``width``). No reply is ever read: the workers poll the round's
        flag on their own."""
        arg = ((int(part) & 0xFF) << 32) | ((int(width) & 0xFFFF) << 16) | (int(cols) & 0xFFFF)
        for i in range(self.n_procs):
            self._send(i, _KIND_ROUND, int(seq), slot, int(bs), arg)

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
            readable = _poll_readable(list(pending), max(0.0, left))
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
        readable = _poll_readable(fds, 0.0)
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


def _small_device(t: torch.Tensor) -> bool:
    """NF-PLE-SMALL: the staged kernel runs on CUDA only (tests patch this)."""
    return bool(getattr(t, "is_cuda", False))


_SMALL_ADDRS_MAX = 8


def _slot_free(slot) -> bool:
    """No kernel / copy reads the slot any more (``event.query()``, no wait)."""
    ev = slot.event
    if ev is None:
        return True
    q = getattr(ev, "query", None)
    if q is not None and q():
        slot.event = None
        return True
    return False


def _capturing(t: torch.Tensor) -> bool:
    """Under graph capture the decode stage / kernel keep the gather."""
    return bool(getattr(t, "is_cuda", False)) and torch.cuda.is_current_stream_capturing()


def _small_needs_pin(t: torch.Tensor) -> bool:
    """The device reads the slot directly: it must be page-locked AND mapped
    (an unregistered slot keeps the old gather). Tests on CPU patch this."""
    return bool(getattr(t, "is_cuda", False))


class _Slot:
    """The owner's view of one shared slot: host tensor, pin state, last H2D."""

    def __init__(self) -> None:
        self.rows = 0
        self.bytes: Optional[torch.Tensor] = None
        self.pinned = False
        self.event = None
        # NF-PLE-SMALL: the predicted id of every staged row (page-locked
        # int64, read by the staged kernel) and the device copies of
        # [ids address, rows address] per (device, row offset)
        self.stage_ids: Optional[torch.Tensor] = None
        self.addrs: dict = {}

    def stage_ids_for(self, n: int) -> torch.Tensor:
        """At least ``n`` page-locked int64 ids (content not kept on growth)."""
        if self.stage_ids is None or self.stage_ids.numel() < n:
            cap = max(int(n), 1 << 13)
            self.stage_ids = torch.empty(cap, dtype=torch.int64, pin_memory=torch.cuda.is_available())
            self.addrs = {}
        return self.stage_ids

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
        self.addrs = {}


class _Pending(NamedTuple):
    seq: int
    slot: int
    ids: torch.Tensor
    vocab: tuple
    #: H43 re-key (1001): the prediction's rows begin at this row of the slot
    #: (an admission whose read window started before the chunk does)
    off: int = 0


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

    def wants_small(self, flat_ids: torch.Tensor) -> bool:
        """NF-PLE-SMALL (20.10.): an EXTEND gather below the pread range.

        Measured (rc28 D ``..._1020_174205``, TP0 = the only PLE rank of Form
        A): every extend whose PLE gather had fewer than ``min_rows`` rows went
        through the HMM kernel -- cold checkpoint pages faulted in one at a
        time on ZFS -- and paid 0.11-1.42 s outside the expert span (T=95:
        gpu-ms 1875.7 against em_ms 443.2, TP1/TP2 waiting 1472/1516 ms in
        tp.all_reduce); every gather of >= 2048 rows (pread) paid 25-145 ms.
        Taken only for a gather the scheduler published a NEW extend hint for
        (a decode / verify gather never publishes one, so it keeps the decode
        stage and the kernel exactly as before), never under graph capture,
        never on a device the staged kernel cannot read (``_small_device``)."""
        n = int(flat_ids.numel())
        if self._disabled or n == 0 or n >= self.min_rows:
            return False
        if not _small_device(flat_ids) or _capturing(flat_ids):
            return False
        h = _HINT
        return h is not None and h.gen != self._last_gen

    def gather_small(self, flat_ids, out, *, vocab_start: int = 0, vocab_end: Optional[int] = None,
                     launch: Callable) -> bool:
        """NF-PLE-SMALL: serve a small extend gather from a host slot WITHOUT
        reading the device's lookup ids back (no sync): the rows are those of
        the prefetch the previous chunk queued for this one, or -- none queued
        -- of this chunk's own tokens hashed on the host and read now on the
        worker processes. ``launch(flat_ids, out, addrs, counters)`` runs the
        staged gather kernel (``_gather_ple_embedding_staged_kernel``): row
        ``r`` from the slot when its predicted id equals the device's id, else
        from the table (HMM) -- a wrong prediction costs a cold read, never
        bytes. False = not served (the caller gathers as before)."""
        if self._disabled:
            return False
        try:
            return self._gather_into(flat_ids, out, vocab_start, vocab_end, small_launch=launch) is not None
        except (PleWorkerLost, OSError) as exc:
            logger.error(
                "PLE-PREFETCH disabled: %s -- this process gathers serially from now on", exc
            )
            self._disable()
            return False

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
            rc = int(torch.cuda.cudart().cudaHostRegister(t.data_ptr(), nbytes, _CUDA_HOST_REGISTER_PORTABLE_MAPPED))
            pinned = rc == 0
            if not pinned:
                logger.warning("PLE-PREFETCH: cudaHostRegister(slot %d, %d B) failed (%d); pageable copy", k, nbytes, rc)
        slot.bytes, slot.rows, slot.pinned = t, cap, pinned
        return slot

    def _rows_view(self, slot: _Slot, n: int) -> torch.Tensor:
        dim = self._table.embedding_dim
        return slot.bytes[: n * self._table.row_bytes].view(self._table.dtype).view(n, dim)

    def _queue_next(self, hint, k: int, vocab: tuple, w) -> str:
        """The next chunk, into the other slot, while this forward runs.

        NF-PLE-SMALL: queued whatever its size -- a small tail chunk (rc28 D
        17:51:03, pdflip-2-50: 95 tokens = 1520 rows behind a 2812-token
        chunk) is exactly the gather the HMM kernel made slow; its read now
        overlaps the chunk before it and its gather (``gather_small``) takes
        the rows from the slot."""
        if hint is None or hint.next_tokens is None:
            return "none"
        nids = self._hasher(hint.next_tokens, hint.lead).to(torch.int64).reshape(-1)
        if int(nids.numel()) <= 0:
            return "none"
        k2 = (k + 1) % PLE_PREFETCH_DEPTH
        other = self._slots[k2]
        if not _slot_free(other):
            # review A2: a kernel of an earlier forward still reads that slot;
            # never wait for it on the host -- the next chunk is then read on
            # its own gather (the cur path), no device sync either way
            self.stats["next_skipped_busy"] = self.stats.get("next_skipped_busy", 0) + 1
            return "skip:busy"
        self._ensure_rows(k2, int(nids.numel()))
        n_in = (nids >= vocab[0]) & (nids < vocab[1]) & (nids < self._table.total_rows)
        seq = w.submit(k2, torch.arange(nids.numel()), ple_row_keys(nids, n_in, self._table, self._file_index))
        self._pending = _Pending(seq, k2, nids, vocab)
        self.stats["prefetches"] += 1
        return f"queued:{int(nids.numel())}"

    def _small_addrs(self, slot: _Slot, off: int, device) -> torch.Tensor:
        """[stage ids address, stage rows address (+ ``off`` rows)] on
        ``device``, cached; made from page-locked memory with a non-blocking
        copy (no host sync)."""
        key = (str(device), int(off))
        a = slot.addrs.get(key)
        if a is None and len(slot.addrs) >= _SMALL_ADDRS_MAX:
            slot.addrs.clear()  # review A6: one entry per (device, row offset), bounded
        if a is None:
            host = torch.tensor(
                [slot.stage_ids.data_ptr(), slot.bytes.data_ptr() + int(off) * int(self._table.row_bytes)],
                dtype=torch.int64,
            )
            if getattr(device, "type", str(device)) != "cpu" and torch.cuda.is_available():
                host = host.pin_memory()
                a = host.to(device, non_blocking=True)
            else:
                a = host
            slot.addrs[key] = a
        return a

    def _small_launch(self, launch, flat_ids, out, addrs, meta: dict) -> None:
        """Launch the staged kernel with this gather's own device counters
        [in-range rows, rows taken from the slot] and queue their read-back
        WITHOUT a sync (review A1): a non-blocking copy into page-locked host
        memory behind an event, read by :meth:`_harvest_small` once
        ``event.query()`` says it landed (the next gather, or now on CPU)."""
        dev = flat_ids.device
        ctr = torch.zeros(2, dtype=torch.int32, device=dev)
        launch(flat_ids, out, addrs, ctr)
        pool = self.__dict__.setdefault("_small_host_pool", [])
        host = pool.pop() if pool else torch.zeros(2, dtype=torch.int32, pin_memory=bool(ctr.is_cuda))
        host.copy_(ctr, non_blocking=bool(ctr.is_cuda))
        ev = None
        if ctr.is_cuda:
            ev = torch.cuda.Event()
            ev.record()
        q = self.__dict__.setdefault("_small_hits", collections.deque())
        q.append((ev, host, ctr, meta))
        if ev is None:
            self._harvest_small()

    def _harvest_small(self) -> int:
        """Log the device's own count for every small gather whose counters
        have landed (``event.query()``, never ``synchronize``)."""
        q = self.__dict__.get("_small_hits")
        if not q:
            return 0
        done = 0
        while q and (q[0][0] is None or q[0][0].query()):
            _ev, host, _ctr, meta = q.popleft()
            in_range, hit = (int(x) for x in host.tolist())  # page-locked host memory, already landed
            self.__dict__.setdefault("_small_host_pool", []).append(host)
            self.stats["small_kernel_hit"] = self.stats.get("small_kernel_hit", 0) + hit
            self.stats["small_staged"] = self.stats.get("small_staged", 0) + int(meta["staged"])
            logger.info(
                "PLE-PREFETCH-SMALL-HIT gather=%d rows=%d staged=%d kernel_hit=%d in_range=%d src=%s "
                "decode_tokens=%d (kernel_hit = rows the staged kernel took from the slot, device count "
                "read without a sync; staged - kernel_hit rows went to the table through HMM)",
                meta["gather"], meta["rows"], meta["staged"], hit, in_range, meta["src"], meta["decode"],
            )
            done += 1
        return done

    def _hash_parts(self, parts) -> torch.Tensor:
        return torch.cat([self._hasher(t, lead).to(torch.int64).reshape(-1) for t, lead in parts])

    def _pick_small_slot(self, avoid: Optional[int]) -> Optional[int]:
        """Review A2: a ring slot no kernel reads any more (``event.query()``),
        starting after the last small gather's slot; None when every one is
        busy (never a host wait)."""
        last = self.__dict__.get("_small_last_k", -1)
        for i in range(1, PLE_PREFETCH_DEPTH + 1):
            k = (last + i) % PLE_PREFETCH_DEPTH
            if k != avoid and _slot_free(self._slots[k]):
                return k
        if avoid is not None and _slot_free(self._slots[avoid]):
            return avoid
        return None

    def _gather_small_into(self, flat_ids, out, vocab_start, vocab_end, hint, launch, t0):
        """NF-PLE-SMALL body (see ``gather_small``). Returns ``out`` when
        served, None when there was nothing to predict (the caller gathers as
        before; the next chunk is queued either way)."""
        n = int(flat_ids.numel())
        dim = self._table.embedding_dim
        if out.numel() != n * dim or (out.dim() >= 1 and out.shape[-1] != dim):
            raise ValueError(f"PLE staged gather: output {tuple(out.shape)} does not match {n} ids x {dim}")
        if vocab_end is None:
            vocab_end = self._table.total_rows
        vocab = (int(vocab_start), int(vocab_end))
        w = self._ensure_workers()
        self._harvest_small()
        wait_s = 0.0
        gather_s = 0.0
        ready = "none"
        src = "none"
        why = ""
        pred = None
        read_rows = 0
        k, off = 0, 0
        cp = None
        if hint is not None and hint.cur_parts:
            cp = self._hash_parts(hint.cur_parts)[:n]
            if int(cp.numel()) == 0:
                cp = None
        pend, self._pending = self._pending, None
        pend_k = None
        if pend is not None:
            was_ready = w.ready(pend.seq)
            tj = time.monotonic()
            gather_s += w.join(pend.seq)
            wait_s += time.monotonic() - tj
            pend_k = pend.slot
            if pend.vocab == vocab:
                pred, k, off = pend.ids, pend.slot, max(0, int(pend.off))
                ready = "yes" if was_ready else "no"
                src = "next"
                slot = self._slots[k]
                if cp is not None and slot.bytes is not None:
                    # review A3: rows the batch's own tokens know but the
                    # prefetch does not (a second request, a different
                    # request) are read now into the same slot rows
                    L = min(n, int(cp.numel()), max(0, slot.rows - off))
                    P = int(pred.numel())
                    base = torch.full((L,), -1, dtype=torch.int64)
                    base[: min(P, L)] = pred[: min(P, L)]
                    diff = torch.nonzero(base != cp[:L]).flatten()
                    if diff.numel():
                        d_ids = cp[diff]
                        d_in = (d_ids >= vocab[0]) & (d_ids < vocab[1]) & (d_ids < self._table.total_rows)
                        tr = time.monotonic()
                        seq = w.submit(k, diff + off, ple_row_keys(d_ids, d_in, self._table, self._file_index))
                        gather_s += w.join(seq)
                        wait_s += time.monotonic() - tr
                        merged = torch.full((max(P, L),), -1, dtype=torch.int64)
                        merged[:P] = pred
                        merged[diff] = d_ids
                        pred = merged
                        read_rows += int(diff.numel())
                        src = "next+cur"
        if pred is None and cp is not None:
            # nothing queued for this batch: its own tokens, hashed here, read now
            kk = self._pick_small_slot(avoid=None)
            if kk is None:
                why = " reason=slots_busy"
                self.stats["small_slots_busy"] = self.stats.get("small_slots_busy", 0) + 1
            else:
                k = kk
                pred = cp
                self._ensure_rows(k, int(pred.numel()))
                p_in = (pred >= vocab[0]) & (pred < vocab[1]) & (pred < self._table.total_rows)
                tr = time.monotonic()
                seq = w.submit(k, torch.arange(pred.numel()), ple_row_keys(pred, p_in, self._table, self._file_index))
                gather_s += w.join(seq)
                wait_s += time.monotonic() - tr
                off = 0
                read_rows += int(pred.numel())
                src = "cur"
        elif pred is None and pend_k is not None:
            k = pend_k
        slot = self._slots[k]
        served = pred is not None and slot.bytes is not None and (slot.pinned or not _small_needs_pin(flat_ids))
        m = 0
        gather_no = self.__dict__.get("_small_gather_no", 0) + 1
        self._small_gather_no = gather_no
        n_dec = int(getattr(hint, "cur_decode_tokens", 0) or 0) if hint is not None else 0
        if served:
            m = min(n, int(pred.numel()), max(0, slot.rows - off))
            ids_buf = slot.stage_ids_for(n)
            ids_buf[:m].copy_(pred[:m])
            if m < n:
                ids_buf[m:n].fill_(-1)  # no prediction: the kernel reads these rows from the table
            self._small_launch(launch, flat_ids, out, self._small_addrs(slot, off, flat_ids.device),
                               {"gather": gather_no, "rows": n, "staged": m, "src": src, "decode": n_dec})
            if out.is_cuda:
                slot.event = torch.cuda.Event()
                slot.event.record()
            self._small_last_k = k
        nxt = self._queue_next(hint, k, vocab, w)
        wall = time.monotonic() - t0
        self.stats["small_gathers"] = self.stats.get("small_gathers", 0) + (1 if served else 0)
        self.stats["small_rows"] = self.stats.get("small_rows", 0) + (m if served else 0)
        self.stats["wait_s"] += wait_s
        # review A6: the bytes this gather read now (cur rows + A3 rows)
        _rank_timing.note_ple("prefill", wall * 1000.0, hit=m, miss=n - m, wait_ms=wait_s * 1000.0,
                              nbytes=read_rows * int(self._table.row_bytes))
        logger.info(
            "PLE-PREFETCH-SMALL gather=%d rows=%d staged=%d src=%s ready=%s served=%s%s read_rows=%d "
            "decode_tokens=%d wait_ms=%.1f gather_ms=%.1f host_ms=%.1f next=%s (extend below the pread "
            "range: rows from the slot, a row whose id the device does not confirm from the table; no "
            "device read-back; the device's own count follows as PLE-PREFETCH-SMALL-HIT)",
            gather_no, n, m, src, ready, "yes" if served else "no", why, read_rows, n_dec,
            wait_s * 1000.0, gather_s * 1000.0, wall * 1000.0, nxt,
        )
        return out if served else None

    def _gather_into(self, flat_ids, out, vocab_start, vocab_end, small_launch=None):
        from flliper.srt.layers.prefill_timing import log_ple_gather
        from flliper.srt.models.qwen4_exp_ple_table import note_ple_prefill_rows

        t0 = time.monotonic()
        hint = take_ple_next_chunk(self._last_gen)
        if hint is not None:
            self._last_gen = hint.gen
        if small_launch is not None:
            return self._gather_small_into(flat_ids, out, vocab_start, vocab_end, hint, small_launch, t0)
        self._harvest_small()
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
        off = 0
        if pend is not None:
            was_ready = w.ready(pend.seq)
            tj = time.monotonic()
            gather_s += w.join(pend.seq)
            wait_s += time.monotonic() - tj
            k = pend.slot
            off = max(0, int(pend.off))
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
        # review A5: the slot's last reader (a small gather's kernel) is done
        # before the workers write it -- explicit, not by the .cpu() above
        self._slots[k].wait_copy()
        slot = self._ensure_rows(k, off + n)
        if miss.numel():
            keys = ple_row_keys(ids[miss], in_range[miss], self._table, self._file_index)
            tr = time.monotonic()
            seq = w.submit(k, miss + off, keys)
            gather_s += w.join(seq)
            wait_s += time.monotonic() - tr
        flat_out = out.reshape(n, dim)
        flat_out.copy_(self._rows_view(slot, off + n)[off:], non_blocking=out.is_cuda and slot.pinned)
        if flat_out.data_ptr() != out.data_ptr():
            out.copy_(flat_out.reshape(out.shape))
        if out.is_cuda:
            slot.event = torch.cuda.Event()
            slot.event.record()

        # the next chunk, into the other slot, while this forward runs
        nxt = self._queue_next(hint, k, vocab, w)

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
        # RANK-TIMING: the chunk's PLE (host wall; hit = rows the prefetch had ready)
        _rank_timing.note_ple("prefill", wall * 1000.0, hit=hit, miss=int(miss.numel()),
                              wait_ms=wait_s * 1000.0,
                              nbytes=int(miss.numel()) * int(self._table.row_bytes))
        logger.info(
            "PLE-PREFETCH chunk=%d rows=%d ready=%s wait_ms=%.1f gather_ms=%.1f "
            "hit_rows=%d read_rows=%d host_ms=%.1f next=%s procs=%d",
            chunk, n, ready, wait_s * 1000.0, gather_s * 1000.0, hit,
            int(miss.numel()), wall * 1000.0, nxt, w.n_procs,
        )
        log_ple_gather(n, zero_rows, wall, w.n_procs * self._threads)
        return out


def make_ple_prefetch_gather(base, table, hasher, *, delay_s: float = 0.0):
    """Wrap the pread gather (``FLLIPER_QWEN4_PLE_PREFETCH``, default on);
    returns ``base`` itself when the switch is off or there is nothing to wrap."""
    if base is None or hasher is None or not envs.FLLIPER_QWEN4_PLE_PREFETCH.get():
        return base
    # fnFL2 H43: plus the first chunk's read from the request's admission
    from flliper.srt.models import qwen4_exp_ple_admit as _admit

    cls = _admit.PleAdmitPrefetchGather if _admit.ple_admission_wanted() else PlePrefetchGather
    return cls(
        base,
        table,
        hasher,
        procs=max(1, int(envs.FLLIPER_QWEN4_PLE_PREFETCH_PROCS.get())),
        threads=max(1, int(envs.FLLIPER_QWEN4_PLE_PREFETCH_THREADS.get())),
        delay_s=delay_s,
    )
