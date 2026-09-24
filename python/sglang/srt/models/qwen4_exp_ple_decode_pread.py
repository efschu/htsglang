"""fnFL2 H40: the verify round's PLE rows, read by pread workers before the replay.

THE COST (H35, 24.09.). On D TP0 (5090, Form A and flip form alike) the verify
compute per decode round has a floor of 9.9-10.0 ms and, on text generated for
the FIRST time in the process, a bump of +3..+10 ms (code 6.4, prose ~7,
thinking ~3.4 ms per round); the same answer generated again sits on the floor.
The PLE table (95.4 GiB, one PLE layer, row 320 B) stays on disk (ZFS, ARC
5 GiB); the gather kernel inside the captured verify graph reads it through
HMM, 16 rows per token (8 bigram + 8 trigram heads), and every row not yet
mapped in this process is a cold ZFS page, served one at a time (~1.6k
pages/s, probe 21.09.).

THE FIX. The round's tokens are known before the verify: the verify row of a
request is ``[bonus, d1 .. dk]`` (``verify_input.draft_token``) and its n-gram
history is the pool's context of the request (two tokens for trigrams). So:

* :func:`begin_ple_verify_stage` (top of ``EAGLEWorkerV2.verify``, the draft
  is launched): one small D2H of ``[history | verify row]`` per request, an
  event behind it -- nothing waits yet;
* :func:`finish_ple_verify_stage` (right before the verify forward, after
  the CPU-side verify preparation): waits for that event, hashes the windows
  on the host with the layer's own arithmetic (:func:`ple_verify_row_ids`,
  bit-equal to ``Qwen4ExpNGramEmbedding._hash_contexts``), and has the H32
  pread worker PROCESSES (``python -I -S``, never the scheduler thread; the
  H29a CPU touch of the mmap held the GIL up to 30 s) read the rows straight
  into a page-locked host STAGE -- sorted by (file, offset), spread over
  procs x threads. Rows read within ``SGLANG_QWEN4_PLE_DECODE_PREAD_BUDGET_MS``
  get their id written next to them; the rest keep id -1;
* the gather kernel of decode/verify-sized gathers
  (:func:`_gather_ple_embedding_staged_kernel`, captured once into the graph
  in place of the plain HMM kernel) takes row ``r`` from the stage when
  ``stage_id[r] == id[r]``, otherwise reads the table through HMM exactly as
  before. The stage is host memory at a FIXED address (memfd, mapped
  portable+mapped): no H2D copy, no re-capture, nothing the GPU memory saver
  can drop; the workers write the very bytes the kernel reads.

Correctness never rests on the stage: a staged row is taken only under its
own id, and a stage row holds the file bytes of the id next to it (the same
file and offset the mmap maps) -- stale entries are still true pairs. A lost
worker or a failed host region switches the stage off (every id -1, the
kernel reads everything through HMM).

COST. Host RAM: the stage (4096 rows x (row + 8 B) = 1.3 MiB page-locked for
bf16) plus the workers' interpreters (~9 MiB anon each, 4 by default). Per
round on the critical path: the host gap between the draft's end and the
verify launch (hash, pread, bookkeeping; ``wait_ms`` on the log line).
Device: two int32 counters. Proof line every ``..._LOG_EVERY`` rounds::

    PLE-DECODE-PREAD rounds=32 rows=2048 hit=2048 late=0 kernel_rows=2048
        kernel_hit=2048 wait_ms=0.41 wait_max_ms=1.20 read_ms=0.30
        sync_ms=2.10 procs=4

``hit`` = rows staged in time (host), ``kernel_hit`` = rows the kernel took
from the stage (device counter, counts every staged-kernel gather since the
last line) -- the second one proves the host hash mirrors the model's.
``SGLANG_QWEN4_PLE_DECODE_PREAD=0`` keeps the plain kernel in the graph.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, List, NamedTuple, Optional, Sequence

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.decode_host_split import (
    note_span as _h58_span,
)
from sglang.srt.models.qwen4_exp_ple_prefetch import (
    PleHashParams,
    PlePreadProcs,
    PleWorkerLost,
    ple_row_keys,
)

logger = logging.getLogger(__name__)

#: Rows the stage holds: bs 64 x 4 verify tokens x 16 heads. A larger gather
#: (a prefill chunk) keeps the plain kernel.
PLE_DECODE_STAGE_ROWS = 4096
_CUDA_HOST_REGISTER_PORTABLE_MAPPED = 3
_MASK64 = (1 << 64) - 1
_SIGN64 = 1 << 63
_WRAP32 = 1 << 32

__all__ = [
    "PLE_DECODE_STAGE_ROWS",
    "PleDecodeStager",
    "PleHashParamsPy",
    "begin_ple_verify_stage",
    "finish_ple_verify_stage",
    "make_ple_decode_stager",
    "ple_verify_row_ids",
]


# --------------------------------------------------------------------------
# The layer's n-gram hash for a handful of windows, in plain Python ints.
# --------------------------------------------------------------------------


class PleHashParamsPy(NamedTuple):
    multipliers: tuple
    head_vocab_sizes: tuple
    head_offsets: tuple
    heads_per_ngram: int
    ngram_size: int
    eos_token_id: int

    @classmethod
    def of(cls, p: PleHashParams) -> "PleHashParamsPy":
        return cls(
            tuple(int(x) for x in p.layer_multipliers.tolist()),
            tuple(int(x) for x in p.head_vocab_sizes.tolist()),
            tuple(int(x) for x in p.head_offsets.tolist()),
            int(p.heads_per_ngram),
            int(p.ngram_size),
            int(p.eos_token_id),
        )


def _wrap64(x: int) -> int:
    """int64 two's-complement wrap-around (torch's int64 multiply)."""
    x &= _MASK64
    return x - (1 << 64) if x & _SIGN64 else x


def ple_verify_row_ids(ctx_rows: Sequence[Sequence[int]], p: PleHashParamsPy) -> List[int]:
    """``[history (ngram_size-1) | verify row]`` per request -> the gather's
    row ids, token-major, ``ngram_heads`` per token (bigram heads, then
    trigram heads ...): the model's ``cat([history, padded]).unfold`` windows
    through ``Qwen4ExpNGramEmbedding._hash_contexts``, value for value.

    Only the window's LAST position is hashed (the model keeps
    ``mix[:, -1:]``); a shift by ``s`` reads ``w[i-s]`` unless an EOS stands in
    ``w[i-s .. i-1]`` (``_shift_right_ignore_eos``), then EOS."""
    n = p.ngram_size
    eos = p.eos_token_id
    mult = p.multipliers
    hpn = p.heads_per_ngram
    out: List[int] = []
    for row in ctx_rows:
        row = [int(t) for t in row]
        for j in range(len(row) - n + 1):
            i = j + n - 1
            shifted = [row[i]]
            broken = False
            for s in range(1, n):
                v = row[i - s]
                if broken or v == eos:
                    broken = True
                    v = eos
                shifted.append(v)
            for ngram in range(2, n + 1):
                mix = _wrap64(shifted[0] * mult[0])
                for pos in range(1, ngram):
                    mix ^= _wrap64(shifted[pos] * mult[pos])
                start = (ngram - 2) * hpn
                for h in range(start, start + hpn):
                    out.append(mix % p.head_vocab_sizes[h] + p.head_offsets[h])
    return out


# --------------------------------------------------------------------------
# The gather kernel with a host stage in front of the table.
# --------------------------------------------------------------------------


@triton.jit
def _gather_ple_embedding_staged_kernel(
    bases_ptr,
    shard_rows,
    ids_ptr,
    stage_addrs_ptr,
    counters_ptr,
    output_ptr,
    embedding_dim,
    tp_vocab_start,
    tp_vocab_end,
    is_fp8: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """``_gather_ple_embedding_from_shards_kernel`` with a stage in front:
    row ``r`` comes from ``stage_rows[r]`` when ``stage_ids[r]`` is its id,
    else from the table (HMM). ``stage_addrs_ptr`` = [host address of the
    stage ids, host address of the stage rows] (fixed for the process, like
    the table's shard bases), ``counters_ptr`` = [in-range rows, rows taken
    from the stage]."""
    row_id = tl.program_id(0)
    global_idx = tl.load(ids_ptr + row_id).to(tl.int64)
    in_range = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
    stage_ids_addr = tl.load(stage_addrs_ptr)
    stage_rows_addr = tl.load(stage_addrs_ptr + 1)
    staged_idx = tl.load(stage_ids_addr.to(tl.pointer_type(tl.int64)) + row_id)
    hit = in_range & (staged_idx == global_idx)
    safe_idx = tl.where(in_range, global_idx, 0)
    shard = safe_idx // shard_rows
    local = safe_idx - shard * shard_rows
    base = tl.load(bases_ptr + shard)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < embedding_dim
    if is_fp8:
        weight_ptr = base.to(tl.pointer_type(tl.float8e4nv))
        stage_ptr = stage_rows_addr.to(tl.pointer_type(tl.float8e4nv))
    else:
        weight_ptr = base.to(tl.pointer_type(tl.bfloat16))
        stage_ptr = stage_rows_addr.to(tl.pointer_type(tl.bfloat16))
    from_table = tl.load(
        weight_ptr + local * embedding_dim + offsets,
        mask=mask & in_range & (~hit),
        other=0.0,
    ).to(tl.bfloat16)
    from_stage = tl.load(
        stage_ptr + row_id * embedding_dim + offsets,
        mask=mask & hit,
        other=0.0,
    ).to(tl.bfloat16)
    values = tl.where(hit, from_stage, from_table)
    tl.store(
        output_ptr + row_id * embedding_dim + offsets,
        tl.where(in_range, values, 0.0),
        mask=mask,
    )
    tl.atomic_add(counters_ptr, in_range.to(tl.int32))
    tl.atomic_add(counters_ptr + 1, hit.to(tl.int32))


# --------------------------------------------------------------------------
# The stage and its workers.
# --------------------------------------------------------------------------

_LIVE_STAGERS = 0


class PleDecodeStager:
    """One PLE table's host stage, its pread workers and its staged kernel."""

    def __init__(
        self,
        table,
        params_fn: Callable[[], PleHashParams],
        *,
        vocab_start: int,
        vocab_end: int,
        procs: int = 4,
        threads: int = 4,
        budget_s: float = 0.008,
        log_every: int = 32,
        delay_s: float = 0.0,
        capacity: int = PLE_DECODE_STAGE_ROWS,
        device: Optional[torch.device] = None,
    ) -> None:
        global _LIVE_STAGERS
        if not table.shard_files or len(table.shard_offsets) != len(table.bases):
            raise ValueError("checkpoint PLE table carries no per-shard file map")
        self._table = table
        self._params_fn = params_fn
        self._params: Optional[PleHashParamsPy] = None
        self._vocab = (int(vocab_start), int(min(int(vocab_end), table.total_rows)))
        self._files = list(dict.fromkeys(table.shard_files))
        pos = {p: i for i, p in enumerate(self._files)}
        self._file_index = [pos[p] for p in table.shard_files]
        self._n_procs = max(1, int(procs))
        self._threads = max(1, int(threads))
        self._budget_s = max(0.0, float(budget_s))
        self._log_every = max(1, int(log_every))
        self._delay_s = float(delay_s)
        self.capacity = int(capacity)
        self._rb = int(table.row_bytes)
        self._nbytes = self.capacity * (self._rb + 8)
        self._fd: Optional[int] = None
        self._region: Optional[torch.Tensor] = None
        self._stage_ids: Optional[torch.Tensor] = None
        self._pinned = False
        self._host_failed = False
        self._workers: Optional[PlePreadProcs] = None
        self._late_seq: Optional[int] = None
        self._n_prev = 0
        self._disabled = False
        if device is None:
            device = (
                torch.device("cuda", torch.cuda.current_device())
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        # per device: (stage addresses, kernel counters), made by the first
        # NON-capturing launch -- the graph's warm-up forward, the same call
        # that makes the table's ``bases_on`` tensor, so both live in the same
        # memory (a capture must not copy host lists to the device; load time
        # would put them into the weights' region)
        self._dev: dict = {}
        self._ctr_device = str(device)
        self._ensure_host()
        pin = torch.cuda.is_available()
        self._ctr_host = torch.zeros(2, dtype=torch.int32, pin_memory=pin)
        self._ctr_seen: Optional[tuple] = None
        self._win = self._new_window()
        self.stats = {"rounds": 0, "rows": 0, "hit": 0, "late": 0, "wait_s": 0.0}
        _LIVE_STAGERS += 1

    # -- state ------------------------------------------------------------
    @property
    def active(self) -> bool:
        return not self._disabled and not self._host_failed

    @property
    def stage_ids(self) -> Optional[torch.Tensor]:
        return self._stage_ids

    @property
    def stage_rows(self) -> Optional[torch.Tensor]:
        """The stage as ``[capacity, dim]`` of the table dtype (host)."""
        if self._region is None:
            return None
        dim = self._table.embedding_dim
        return self._region[: self.capacity * self._rb].view(self._table.dtype).view(self.capacity, dim)

    @property
    def counters(self) -> Optional[torch.Tensor]:
        st = self._dev.get(self._ctr_device)
        return None if st is None else st[1]

    def _device_state(self, device: torch.device, *, capturing: bool) -> Optional[tuple]:
        key = str(device)
        st = self._dev.get(key)
        if st is None:
            if capturing:
                return None
            addrs = torch.tensor(
                [self._stage_ids.data_ptr(), self._region.data_ptr()],
                dtype=torch.int64,
                device=device,
            )
            st = (addrs, torch.zeros(2, dtype=torch.int32, device=device))
            self._dev[key] = st
            if self._ctr_device not in self._dev:
                self._ctr_device = key
        return st

    def params(self) -> PleHashParamsPy:
        if self._params is None:
            self._params = PleHashParamsPy.of(self._params_fn())
        return self._params

    @staticmethod
    def _new_window() -> dict:
        return {"rounds": 0, "rows": 0, "hit": 0, "late": 0, "wait_s": 0.0,
                "wait_max_s": 0.0, "read_s": 0.0, "sync_s": 0.0}

    # -- the host stage -------------------------------------------------------
    def _ensure_host(self) -> bool:
        if self._region is not None:
            return True
        if self._host_failed:
            return False
        fd = None
        try:
            fd = os.memfd_create("ple-decode-stage", 0)
            os.ftruncate(fd, self._nbytes)
            region = torch.from_file(
                f"/proc/self/fd/{fd}", shared=True, size=self._nbytes, dtype=torch.uint8
            )
            ids = region[self.capacity * self._rb :].view(torch.int64)
            ids.fill_(-1)
            if torch.cuda.is_available():
                rc = int(
                    torch.cuda.cudart().cudaHostRegister(
                        region.data_ptr(), self._nbytes, _CUDA_HOST_REGISTER_PORTABLE_MAPPED
                    )
                )
                self._pinned = rc == 0
                if not self._pinned:
                    logger.warning(
                        "PLE-DECODE-PREAD: cudaHostRegister(stage, %d B) failed (%d); "
                        "the kernel reaches the stage through HMM", self._nbytes, rc,
                    )
        except (OSError, RuntimeError) as exc:
            logger.error("PLE-DECODE-PREAD off: no host stage (%s)", exc)
            self._host_failed = True
            if fd is not None:
                os.close(fd)
            return False
        self._fd, self._region, self._stage_ids = fd, region, ids
        return True

    def _ensure_workers(self) -> Optional[PlePreadProcs]:
        if self._workers is not None or self._disabled:
            return self._workers
        if not self._ensure_host():
            return None
        try:
            w = PlePreadProcs(
                self._files, self._rb, procs=self._n_procs, threads=self._threads,
                delay_s=self._delay_s, slot_fds=[self._fd],
            )
            try:
                w.map_slot(0, self._nbytes)
            except BaseException:
                w.close()
                raise
        except (PleWorkerLost, OSError) as exc:
            self._disable(exc)
            return None
        self._workers = w
        logger.info(
            "PLE-DECODE-PREAD on: %d pread worker processes x %d threads (pids %s), "
            "stage %d rows at %#x (%s), budget %.1f ms per round",
            w.n_procs, self._threads, w.pids(), self.capacity, self._region.data_ptr(),
            "page-locked" if self._pinned else "pageable", self._budget_s * 1000.0,
        )
        return w

    def _disable(self, why) -> None:
        if not self._disabled:
            logger.error(
                "PLE-DECODE-PREAD disabled: %s -- every decode PLE row is read "
                "through HMM from now on", why,
            )
        self._disabled = True
        if self._stage_ids is not None:
            self._stage_ids.fill_(-1)
        if self._workers is not None:
            self._workers.close()
            self._workers = None
        self._late_seq = None

    def retire(self) -> None:
        """Replaced by a new stage (the table was attached again): every id
        -1 and the workers gone, but the host stage stays mapped -- a graph
        captured against it may still be replayed, and then reads the table
        through HMM."""
        self._disable("replaced by a new stage")

    def close(self) -> None:
        global _LIVE_STAGERS
        self._disable("closed")
        if self._region is not None:
            if self._pinned:
                torch.cuda.cudart().cudaHostUnregister(self._region.data_ptr())
            self._region = None
            self._stage_ids = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        _LIVE_STAGERS = max(0, _LIVE_STAGERS - 1)

    # -- the kernel -------------------------------------------------------------
    def launch(
        self,
        flat_ids: torch.Tensor,
        output: torch.Tensor,
        *,
        vocab_start: int,
        vocab_end: int,
        block_d: int,
    ) -> bool:
        """Launch the staged gather for ``flat_ids`` (eager or under capture).
        False = not served here (too many rows, no host stage): the caller
        launches the plain kernel."""
        n = int(flat_ids.numel())
        if n == 0 or n > self.capacity or not self._ensure_host():
            return False
        capturing = flat_ids.is_cuda and torch.cuda.is_current_stream_capturing()
        st = self._device_state(flat_ids.device, capturing=capturing)
        if st is None:
            return False
        addrs, counters = st
        if capturing:
            # the graph is built before the first round: start the workers
            # now, not inside the first decode round after a flip
            self._ensure_workers()
        table = self._table
        _gather_ple_embedding_staged_kernel[(n,)](
            table.bases_on(flat_ids.device),
            table.shard_rows,
            flat_ids,
            addrs,
            counters,
            output,
            embedding_dim=table.embedding_dim,
            tp_vocab_start=vocab_start,
            tp_vocab_end=vocab_end,
            is_fp8=table.dtype == torch.float8_e4m3fn,
            BLOCK_D=block_d,
        )
        return True

    # -- the round ---------------------------------------------------------------
    def snapshot_counters(self) -> None:
        """Stream-ordered copy of the kernel counters (read after the event)."""
        c = self.counters
        if c is not None:
            self._ctr_host.copy_(c, non_blocking=c.is_cuda)

    def stage(self, ctx_rows: Sequence[Sequence[int]], *, sync_s: float, t_ready: float) -> None:
        """Fill the stage with this round's rows. Must run while no kernel
        reads the stage: after the event behind the draft (the previous
        verify has finished) and before the verify is launched."""
        if not self.active:
            return
        try:
            self._stage(ctx_rows, sync_s, t_ready)
        except (PleWorkerLost, OSError) as exc:
            self._disable(exc)

    def _stage(self, ctx_rows, sync_s: float, t_ready: float) -> None:
        w = self._ensure_workers()
        if w is None:
            return
        ids = ple_verify_row_ids(ctx_rows, self.params())
        n = len(ids)
        if self._late_seq is not None:
            # a worker of the last round is still writing stage rows
            w.join(self._late_seq)
            self._late_seq = None
        stage_ids = self._stage_ids
        clear = min(self.capacity, max(n, self._n_prev))
        if clear:
            stage_ids[:clear] = -1
        hit = 0
        late = 0
        read_s = 0.0
        if 0 < n <= self.capacity:
            ids_t = torch.tensor(ids, dtype=torch.int64)
            lo, hi = self._vocab
            dest = torch.nonzero((ids_t >= lo) & (ids_t < hi)).flatten()
            if dest.numel():
                keys = ple_row_keys(
                    ids_t[dest], torch.ones(dest.numel(), dtype=torch.bool),
                    self._table, self._file_index,
                )
                per_proc = -(-int(dest.numel()) // w.n_procs)
                step = max(1, -(-per_proc // self._threads))
                seq = w.submit(0, dest, keys, step=step)
                sent = dict(w.parts)
                done, read_s = w.collect(seq, self._budget_s)
                if len(done) < len(sent):
                    self._late_seq = seq
                    late = sum(int(sent[i].numel()) for i in sent if i not in done)
                if done:
                    got = torch.cat([sent[i] for i in done])
                    # the id goes in AFTER its bytes: a row is staged only
                    # once its worker has answered for it
                    stage_ids[got] = ids_t[got]
                    hit = int(got.numel())
        self._n_prev = n if n <= self.capacity else 0
        wait_s = time.monotonic() - t_ready
        win = self._win
        win["rounds"] += 1
        win["rows"] += n
        win["hit"] += hit
        win["late"] += late
        win["wait_s"] += wait_s
        win["wait_max_s"] = max(win["wait_max_s"], wait_s)
        win["read_s"] += read_s
        win["sync_s"] += sync_s
        self.stats["rounds"] += 1
        self.stats["rows"] += n
        self.stats["hit"] += hit
        self.stats["late"] += late
        self.stats["wait_s"] += wait_s
        if win["rounds"] >= self._log_every:
            self._log_window(w.n_procs)

    def _kernel_delta(self) -> tuple:
        now = tuple(int(x) for x in self._ctr_host.tolist())
        prev, self._ctr_seen = self._ctr_seen, now
        if prev is None:
            return now
        return tuple((a - b) % _WRAP32 for a, b in zip(now, prev))

    def _log_window(self, procs: int) -> None:
        win = self._win
        k_rows, k_hit = self._kernel_delta()
        r = max(1, win["rounds"])
        logger.info(
            "PLE-DECODE-PREAD rounds=%d rows=%d hit=%d late=%d kernel_rows=%d "
            "kernel_hit=%d wait_ms=%.2f wait_max_ms=%.2f read_ms=%.2f sync_ms=%.2f procs=%d",
            win["rounds"], win["rows"], win["hit"], win["late"], k_rows, k_hit,
            win["wait_s"] * 1000.0 / r, win["wait_max_s"] * 1000.0,
            win["read_s"] * 1000.0 / r, win["sync_s"] * 1000.0 / r, procs,
        )
        self._win = self._new_window()


def make_ple_decode_stager(
    table,
    params_fn: Optional[Callable[[], PleHashParams]],
    *,
    vocab_start: int,
    vocab_end: int,
    device: Optional[torch.device] = None,
) -> Optional[PleDecodeStager]:
    """The stage of a checkpoint-backend PLE table (``SGLANG_QWEN4_PLE_DECODE_PREAD``,
    default on); None when the switch is off or the layer has no host hash."""
    if table is None or params_fn is None or not envs.SGLANG_QWEN4_PLE_DECODE_PREAD.get():
        return None
    return PleDecodeStager(
        table,
        params_fn,
        vocab_start=vocab_start,
        vocab_end=vocab_end,
        procs=envs.SGLANG_QWEN4_PLE_DECODE_PREAD_PROCS.get(),
        threads=envs.SGLANG_QWEN4_PLE_DECODE_PREAD_THREADS.get(),
        budget_s=envs.SGLANG_QWEN4_PLE_DECODE_PREAD_BUDGET_MS.get() / 1000.0,
        log_every=envs.SGLANG_QWEN4_PLE_DECODE_PREAD_LOG_EVERY.get(),
        device=device,
    )


# --------------------------------------------------------------------------
# The verify round: begin after the draft, finish before the forward.
# --------------------------------------------------------------------------


class PleVerifyStage(NamedTuple):
    stagers: tuple
    ctx_host: torch.Tensor
    event: Optional[object]


_MODEL_EMBEDDINGS: dict = {}
_CTX_HOST: dict = {}


def _stagers_of(model) -> List[PleDecodeStager]:
    """The live stagers of ``model``'s PLE tables (module scan once per model)."""
    key = id(model)
    mods = _MODEL_EMBEDDINGS.get(key)
    if mods is None:
        modules = getattr(model, "modules", None)
        mods = [m for m in modules() if hasattr(m, "_decode_stager")] if callable(modules) else []
        _MODEL_EMBEDDINGS[key] = mods
    out = []
    for m in mods:
        st = m._decode_stager
        if st is not None and st.active:
            out.append(st)
    return out


def _ctx_host(shape, pin: bool) -> torch.Tensor:
    key = (tuple(shape), pin)
    t = _CTX_HOST.get(key)
    if t is None:
        t = torch.empty(tuple(shape), dtype=torch.int64, pin_memory=pin)
        _CTX_HOST[key] = t
    return t


def begin_ple_verify_stage(model, pool, batch, verify_input) -> Optional[PleVerifyStage]:
    """After the draft was launched: queue the D2H of ``[history | verify
    row]`` per request behind it. No wait. None = nothing to stage (no PLE
    stage in this process / model, an idle batch, a shape this does not
    know) -- the round then runs exactly as before."""
    if not _LIVE_STAGERS or model is None or pool is None:
        return None
    stagers = _stagers_of(model)
    if not stagers:
        return None
    if batch.forward_mode.is_idle():
        return None
    if not hasattr(pool, "get_ngram_context") or not hasattr(pool, "get_mamba_indices"):
        return None
    tokens = verify_input.draft_token
    width = int(verify_input.draft_token_num)
    req = batch.req_pool_indices
    bs = int(req.shape[0])
    if tokens is None or bs <= 0 or width <= 0 or tokens.numel() != bs * width:
        return None
    if tokens.is_cuda and torch.cuda.is_current_stream_capturing():
        return None
    history = pool.get_ngram_context(pool.get_mamba_indices(req).long())
    ctx = torch.cat(
        [history.to(torch.int64), tokens.reshape(bs, width).to(torch.int64)], dim=1
    )
    host = _ctx_host(ctx.shape, ctx.is_cuda)
    host.copy_(ctx, non_blocking=ctx.is_cuda)
    for st in stagers:
        st.snapshot_counters()
    event = None
    if ctx.is_cuda:
        event = torch.cuda.Event()
        event.record()
    return PleVerifyStage(tuple(stagers), host, event)


def finish_ple_verify_stage(stage: Optional[PleVerifyStage]) -> None:
    """Right before the verify forward: wait for the draft's tokens, stage
    their PLE rows (bounded by the budget)."""
    if stage is None:
        return
    t0 = time.monotonic()
    # fnFL2 H58: ple_sync (the wait for the draft) and ple_stage (hash + pread
    # while the stream idles) of DECODE-HOST-SPLIT; timing only.
    h58_t = time.perf_counter()
    if stage.event is not None:
        stage.event.synchronize()
    t_ready = time.monotonic()
    h58_t = _h58_span("ple_sync_ms", h58_t)
    rows = stage.ctx_host.tolist()
    for st in stage.stagers:
        st.stage(rows, sync_s=t_ready - t0, t_ready=t_ready)
    _h58_span("ple_stage_ms", h58_t)
