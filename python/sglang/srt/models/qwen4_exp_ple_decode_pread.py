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

fnFL2 H69 -- THE STAGE BEHIND THE REPLAY (``SGLANG_WEG2_PLE_STAGE_BEHIND_REPLAY``).
x168 (DECODE-HOST-SPLIT, TP0, 64-round medians): the device round tiles as
draft 1.9 + gap_ple 1.7 + verify 21.2 + accept 0.1 + dext 0.9 ms with
gap_round 0.0 -- the ONLY idle stretch of the round is the one this module
opens: the host waits for the draft's tokens, hashes, preads and only then
launches the verify, and the workers wait for it in the verify's first
all-reduce. Every other host wait of the round (ctl_wait 8-10 ms, the
workers' seq_wait/recv) is slack: the device is busy while it lasts.

The rows are read by layer 1 (``ple_layer_ids == [2]``) through the prefetch
stream that forks at layer 0, so they are not needed when the verify STARTS,
only one decoder layer later. With the switch on, a graphed verify round
runs in this order instead:

* ``arm_ple_verify_gate`` (before the forward): the round gets a number
  ``seq`` and a stream-ordered ``expect <- seq`` lands on the device;
* the verify graph is launched while the draft may still be running; its
  prefetch stream runs :func:`_ple_stage_gate_kernel` (one warp) that spins
  on the host's ``done`` word -- page-locked, in the stage's own memfd --
  until ``done >= expect``, bounded; decoder layer 0 runs meanwhile on the
  forward stream;
* the host stages the round exactly as before (``finish_ple_verify_stage``,
  called from the model runner's post-replay hook right after the graph
  launch) and then publishes ``done <- seq``;
* :func:`_gather_ple_embedding_gated_kernel` takes stage rows only when the
  gate passed; a gate that timed out sends every row through HMM (bytes
  unchanged, it only costs time), and the kernel never touches a stage the
  host may still be writing;
* ``disarm_ple_verify_gate`` (after the forward): ``expect <- -1``, so any
  later gather that is not an armed verify reads through HMM as well.

Eager verify rounds keep the H40 order (stage, publish, then launch) -- the
gate is passed before the kernel starts. ``gate_pass``/``gate_timeout`` on
the proof line count the gate's outcomes.

fnFL2 H73 -- THE STAGE WITHOUT THE HOST (``SGLANG_WEG2_PLE_STAGE_AUTONOMOUS``).
x172 (H69b: 64 rows per round instead of 21, H69 on): TP0 ``ple_stage``
2.8-3.0 ms on the scheduler thread, and on the device ``spec_verify:ple.wait``
3.2-3.8 ms per round -- the gate holds layer 1 until the host, which starts
only once the verify REPLAY has returned (~1.2 ms after the graph began),
has hashed and read 64 cold rows (``read_ms`` 2.0-2.5: ZFS serves ~20 sync
reads at a time, a cold 320-B row costs a whole 128-KiB record). The rows
depend on nothing the host knows before the draft's end; the host only
relays. With the switch on, nobody relays:

* ``begin_ple_verify_stage`` (top of verify, the draft launched) numbers the
  round, arms the gate and POSTS it: the windows ``[history | verify row]``
  go into a mailbox inside the stage's own memfd (D2H copy behind the
  draft), then -- a second copy on the same stream, so strictly after the
  first -- the round number into the mailbox's ``flag`` word; one 28-byte
  ROUND message per worker announces it (``PlePreadProcs.ping``, no reply);
* each pread worker (``AutoStage`` in the worker script) polls ``flag``,
  hashes the windows itself (stdlib mirror of :func:`ple_verify_row_ids`),
  reads the rows of ITS stage positions (``r % procs``: a position always
  belongs to the same worker) into the stage -- id -1, bytes, id -- and
  publishes its own ``done`` word;
* :func:`_ple_stage_gate_all_kernel` passes when EVERY worker's ``done``
  reached the round (bounded as H69; timed out: every row through HMM);
* ``finish_ple_verify_stage`` does nothing, no post-replay hook is parked.

The reads now start ~0.1 ms after the draft's end (flag + poll + hash)
instead of after the replay returned, and the scheduler thread's share is
the post (a few us plus one pipe write per worker).
``SGLANG_WEG2_PLE_STAGE_BONUS_EARLY`` adds a second mailbox: after the accept
(end of verify) the device posts ``[committed history | bonus]`` per
request -- the first token of the NEXT verify row -- and the workers read
those rows (16 per request) into the positions the next round will ask for,
while the draft extend and the next draft run; the next verify round keeps
every position whose id already names the wanted row (another batch order
or width only costs the read). Proof line: ``auto=1 stage_max_ms poll_ms
read_rows kept missed errors bonus_rows`` appended; ``hit``/``late`` are
the kernel's (rows taken from the stage / in-range rows read via HMM),
``wait_ms`` is the scheduler thread's post, ``read_ms`` the workers' mean
flag-to-done time. The kernel counters are read without a sync and may lag
one round.
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
from sglang.srt.models.qwen4_exp_ple_pread_worker import (
    PART_BONUS,
    PART_VERIFY,
    ST_BONUS_READ,
    ST_ERRORS,
    ST_MISSED,
    ST_POLL_NS,
    ST_READ,
    ST_REUSED,
    ST_ROUNDS,
    ST_STAGE_MAX_NS,
    ST_STAGE_NS,
    STATS_WORDS,
)

logger = logging.getLogger(__name__)

#: Rows the stage holds: bs 64 x 4 verify tokens x 16 heads. A larger gather
#: (a prefill chunk) keeps the plain kernel.
PLE_DECODE_STAGE_ROWS = 4096
_CUDA_HOST_REGISTER_PORTABLE_MAPPED = 3
_MASK64 = (1 << 64) - 1
_SIGN64 = 1 << 63
_WRAP32 = 1 << 32
#: fnFL2 H69: the gate block behind the stage ids (the ``done`` word, padded
#: to a cache line); only a gated stage has it.
_GATE_BYTES = 64
#: fnFL2 H69: the gate's poll bound per microsecond of host budget, and its
#: floor (see ``ple_gate_spins``).
_GATE_SPINS_PER_US = 4
_GATE_SPINS_MIN = 4096
#: fnFL2 H73: every block of the autonomous stage starts on its own cache line
_LINE = 64
#: fnFL2 H73: the workers' poll interval between the announcement and the
#: device's post -- short for the verify round (its latency is the gate's), long
#: for the bonus rows (posted after the accept, read long before the next
#: verify) -- and how long a worker waits for a post before it gives the round up
_AUTO_POLL_US = 20
_AUTO_POLL_BONUS_US = 200
_AUTO_DEADLINE_MS = 500.0

__all__ = [
    "PLE_DECODE_STAGE_ROWS",
    "PleDecodeStager",
    "PleHashParamsPy",
    "arm_ple_verify_gate",
    "begin_ple_verify_stage",
    "disarm_ple_verify_gate",
    "finish_ple_verify_stage",
    "make_ple_decode_stager",
    "ple_gate_spins",
    "ple_stage_is_gated",
    "ple_verify_row_ids",
    "post_ple_bonus_stage",
]


def _lines(nbytes: int) -> int:
    """``nbytes`` rounded up to whole cache lines."""
    return -(-int(nbytes) // _LINE) * _LINE


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


@triton.jit
def _ple_stage_gate_kernel(
    expect_ptr,
    done_addr_ptr,
    go_ptr,
    gate_counters_ptr,
    MAX_SPINS: tl.constexpr,
):
    """fnFL2 H69: one program (launch it with ``num_warps=1``) in front of
    :func:`_gather_ple_embedding_gated_kernel` on the same stream.

    ``expect_ptr`` = the round the verify was armed for (device int64, -1 =
    not armed), ``done_addr_ptr`` = [host address of the stage's ``done``
    word] (int64, page-locked, written by the host after the round's rows and
    ids). Spins on ``done`` until it reaches ``expect`` or ``MAX_SPINS`` polls
    passed, then writes ``go_ptr`` (int32: 1 = the stage holds this round) and
    counts [passed, timed out] in ``gate_counters_ptr``. Not armed: no poll,
    ``go`` 0, nothing counted."""
    expect = tl.load(expect_ptr)
    armed = expect >= 0
    done_ptr = tl.load(done_addr_ptr).to(tl.pointer_type(tl.int64))
    done = tl.load(done_ptr, volatile=True)
    spins = (expect * 0).to(tl.int32)  # a scalar tensor: the loop carries it
    while armed & (done < expect) & (spins < MAX_SPINS):
        done = tl.load(done_ptr, volatile=True)
        spins += 1
    go = armed & (done >= expect)
    tl.store(go_ptr, go.to(tl.int32))
    tl.atomic_add(gate_counters_ptr, go.to(tl.int32))
    tl.atomic_add(gate_counters_ptr + 1, (armed & (done < expect)).to(tl.int32))


@triton.jit
def _ple_stage_gate_all_kernel(
    expect_ptr,
    done_addr_ptr,
    go_ptr,
    gate_counters_ptr,
    N_DONE: tl.constexpr,
    MAX_SPINS: tl.constexpr,
):
    """fnFL2 H73: :func:`_ple_stage_gate_kernel` over ``N_DONE`` words -- one
    per pread worker of an autonomous stage, each written by its worker once
    its share of the round is in. Passes when the SMALLEST reached
    ``expect``. A poll issues the ``N_DONE`` independent 8-byte loads of one
    cache line back to back (unrolled). Same bound, same outputs and
    counters as the one-word gate."""
    expect = tl.load(expect_ptr)
    armed = expect >= 0
    done_ptr = tl.load(done_addr_ptr).to(tl.pointer_type(tl.int64))
    done = tl.load(done_ptr, volatile=True)
    for i in tl.static_range(1, N_DONE):
        done = tl.minimum(done, tl.load(done_ptr + i, volatile=True))
    spins = (expect * 0).to(tl.int32)  # a scalar tensor: the loop carries it
    while armed & (done < expect) & (spins < MAX_SPINS):
        done = tl.load(done_ptr, volatile=True)
        for i in tl.static_range(1, N_DONE):
            done = tl.minimum(done, tl.load(done_ptr + i, volatile=True))
        spins += 1
    go = armed & (done >= expect)
    tl.store(go_ptr, go.to(tl.int32))
    tl.atomic_add(gate_counters_ptr, go.to(tl.int32))
    tl.atomic_add(gate_counters_ptr + 1, (armed & (done < expect)).to(tl.int32))


@triton.jit
def _gather_ple_embedding_gated_kernel(
    bases_ptr,
    shard_rows,
    ids_ptr,
    stage_addrs_ptr,
    go_ptr,
    counters_ptr,
    output_ptr,
    embedding_dim,
    tp_vocab_start,
    tp_vocab_end,
    is_fp8: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """fnFL2 H69: :func:`_gather_ple_embedding_staged_kernel` behind the gate.
    The stage is read only when ``go_ptr`` (written by
    :func:`_ple_stage_gate_kernel` just before, same stream) says the host
    finished this round; otherwise every row comes from the table (HMM) and
    the stage -- which the host may still be writing -- is not touched."""
    row_id = tl.program_id(0)
    global_idx = tl.load(ids_ptr + row_id).to(tl.int64)
    in_range = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
    go = tl.load(go_ptr) != 0
    stage_ids_addr = tl.load(stage_addrs_ptr)
    stage_rows_addr = tl.load(stage_addrs_ptr + 1)
    staged_idx = tl.load(
        stage_ids_addr.to(tl.pointer_type(tl.int64)) + row_id, mask=go, other=-1
    )
    hit = in_range & go & (staged_idx == global_idx)
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
#: fnFL2 H73: live stagers that take the bonus rows early (the verify path's
#: one-read no-op test while the switch is off)
_LIVE_BONUS = 0


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
        gated: bool = False,
        gate_spins: int = _GATE_SPINS_MIN,
        autonomous: bool = False,
        bonus_early: bool = False,
    ) -> None:
        global _LIVE_STAGERS, _LIVE_BONUS
        if not table.shard_files or len(table.shard_offsets) != len(table.bases):
            raise ValueError("checkpoint PLE table carries no per-shard file map")
        # fnFL2 H73: the workers fill the stage on their own (module
        # docstring); it is always gated, with one ``done`` word per worker
        self._auto = bool(autonomous)
        self._bonus = self._auto and bool(bonus_early)
        # fnFL2 H69: a gated stage carries the ``done`` word and launches the
        # gate + gated kernel pair (module docstring); set once, never flipped
        self._gated = bool(gated) or self._auto
        self._gate_spins = max(1, int(gate_spins))
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
        # rows | ids | gate block (H69: one done word; H73: one per worker) |
        # H73: verify mailbox (flag line, windows) | bonus mailbox | stats
        self._n_done = self._n_procs if self._auto else 1
        self._ids_end = self.capacity * (self._rb + 8)
        gate_bytes = _lines(8 * self._n_done) if self._gated else 0
        self._auto_off = self._ids_end + gate_bytes
        self._ctx_cap = self.capacity
        auto_bytes = (
            2 * (_LINE + 8 * self._ctx_cap) + 8 * STATS_WORDS * self._n_procs
            if self._auto
            else 0
        )
        self._nbytes = self._ids_end + gate_bytes + auto_bytes
        self._fd: Optional[int] = None
        self._region: Optional[torch.Tensor] = None
        self._stage_ids: Optional[torch.Tensor] = None
        #: fnFL2 H69: the page-locked ``done`` word (int64 view), gated only;
        #: fnFL2 H73: the workers' ``done`` words
        self._done: Optional[torch.Tensor] = None
        #: fnFL2 H69: per device (expect, go, gate counters, [done address])
        self._gate_dev: dict = {}
        #: fnFL2 H73: per part (flag int64[1], windows int64[ctx_cap]) and the
        #: workers' stats [procs, STATS_WORDS] (host views of the stage)
        self._mbox: Optional[tuple] = None
        self._wstats: Optional[torch.Tensor] = None
        self._wstats_seen: Optional[list] = None
        #: fnFL2 H73: per (device, part) the round number the post copies
        self._seq_dev: dict = {}
        self._auto_ready = False
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
        self._gate_ctr_host: Optional[torch.Tensor] = (
            torch.zeros(2, dtype=torch.int32, pin_memory=pin) if self._gated else None
        )
        self._gate_ctr_seen: Optional[tuple] = None
        self._win = self._new_window()
        self.stats = {"rounds": 0, "rows": 0, "hit": 0, "late": 0, "wait_s": 0.0}
        _LIVE_STAGERS += 1
        if self._bonus:
            _LIVE_BONUS += 1

    # -- state ------------------------------------------------------------
    @property
    def active(self) -> bool:
        return not self._disabled and not self._host_failed

    @property
    def gated(self) -> bool:
        """fnFL2 H69: this stage is filled behind the verify replay."""
        return self._gated

    @property
    def autonomous(self) -> bool:
        """fnFL2 H73: the workers fill this stage on their own."""
        return self._auto

    @property
    def bonus_early(self) -> bool:
        """fnFL2 H73: the next round's bonus rows are posted after the accept."""
        return self._bonus

    @property
    def done_word(self) -> Optional[torch.Tensor]:
        """fnFL2 H69: the page-locked ``done`` word the gate polls (gated)."""
        return self._done

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

    def _gate_state(self, device: torch.device, *, capturing: bool) -> Optional[tuple]:
        """fnFL2 H69: (expect int64[1], go int32[1], gate counters int32[2],
        [done address] int64[1]) on ``device``, made like ``_device_state``
        by the first NON-capturing launch; ``expect`` starts at -1 (a gather
        before the first armed verify reads through HMM)."""
        key = str(device)
        st = self._gate_dev.get(key)
        if st is None:
            if capturing or self._done is None:
                return None
            st = (
                torch.full((1,), -1, dtype=torch.int64, device=device),
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.zeros(2, dtype=torch.int32, device=device),
                torch.tensor([self._done.data_ptr()], dtype=torch.int64, device=device),
            )
            self._gate_dev[key] = st
            if self._auto:
                # fnFL2 H73: the post's round numbers, made with the gate
                # state (same moment, same memory)
                for part in (PART_VERIFY, PART_BONUS):
                    self._seq_on(device, part)
        return st

    def _seq_on(self, device: torch.device, part: int) -> torch.Tensor:
        """fnFL2 H73: the int64 the post fills with the round number and copies
        into the part's ``flag`` word, on ``device``."""
        key = (str(device), int(part))
        t = self._seq_dev.get(key)
        if t is None:
            t = torch.zeros(1, dtype=torch.int64, device=device)
            self._seq_dev[key] = t
        return t

    def params(self) -> PleHashParamsPy:
        if self._params is None:
            self._params = PleHashParamsPy.of(self._params_fn())
        return self._params

    @staticmethod
    def _new_window() -> dict:
        return {"rounds": 0, "rows": 0, "hit": 0, "late": 0, "wait_s": 0.0,
                "wait_max_s": 0.0, "read_s": 0.0, "sync_s": 0.0}

    def _auto_layout(self) -> dict:
        """fnFL2 H73: byte offsets of the autonomous stage's blocks in the
        region (the workers get the same numbers)."""
        a, cap = self._auto_off, self._ctx_cap
        return {
            "flag_off": a,
            "ctx_off": a + _LINE,
            "flag_bonus_off": a + _LINE + 8 * cap,
            "ctx_bonus_off": a + 2 * _LINE + 8 * cap,
            "stats_off": a + 2 * (_LINE + 8 * cap),
        }

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
            ids_end = self._ids_end
            ids = region[self.capacity * self._rb : ids_end].view(torch.int64)
            ids.fill_(-1)
            if self._gated:
                # fnFL2 H69: no round published yet (seqs start at 1);
                # fnFL2 H73: one word per worker
                self._done = region[ids_end : ids_end + 8 * self._n_done].view(torch.int64)
                self._done.fill_(0)
            if self._auto:
                lay = self._auto_layout()

                def _q(off: int, count: int) -> torch.Tensor:
                    return region[off : off + 8 * count].view(torch.int64)

                self._mbox = (
                    (_q(lay["flag_off"], 1), _q(lay["ctx_off"], self._ctx_cap)),
                    (_q(lay["flag_bonus_off"], 1), _q(lay["ctx_bonus_off"], self._ctx_cap)),
                )
                for flag, ctx in self._mbox:
                    flag.fill_(0)
                    ctx.fill_(0)
                self._wstats = _q(lay["stats_off"], STATS_WORDS * self._n_procs).view(
                    self._n_procs, STATS_WORDS
                )
                self._wstats.fill_(0)
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

    def _auto_cfg(self, index: int, n_procs: int) -> dict:
        """fnFL2 H73: what worker ``index`` needs to fill its share alone."""
        p = self.params()
        ids_off = self.capacity * self._rb
        return {
            "index": int(index),
            "procs": int(n_procs),
            "capacity": self.capacity,
            "ctx_cap": self._ctx_cap,
            "ids_off": ids_off,
            "done_off": self._ids_end,
            **self._auto_layout(),
            "vocab": [int(self._vocab[0]), int(self._vocab[1])],
            "shard_rows": int(self._table.shard_rows),
            "shard_offsets": [int(o) for o in self._table.shard_offsets],
            "shard_file": [int(i) for i in self._file_index],
            "hash": {
                "mult": [int(x) for x in p.multipliers],
                "sizes": [int(x) for x in p.head_vocab_sizes],
                "offsets": [int(x) for x in p.head_offsets],
                "hpn": int(p.heads_per_ngram),
                "ngram": int(p.ngram_size),
                "eos": int(p.eos_token_id),
            },
            "poll_us": _AUTO_POLL_US,
            "poll_bonus_us": _AUTO_POLL_BONUS_US,
            "deadline_ms": _AUTO_DEADLINE_MS,
        }

    def _ensure_auto(self) -> bool:
        """fnFL2 H73: hand every worker its share of the stage, once. Never
        under a capture: the hash constants are copied from the device."""
        if self._auto_ready:
            return True
        if not self._auto or not self.active:
            return False
        w = self._ensure_workers()
        if w is None:
            return False
        if w.n_procs != self._n_done:
            self._disable(f"{w.n_procs} workers for {self._n_done} done words")
            return False
        try:
            w.configure_auto(0, [self._auto_cfg(i, w.n_procs) for i in range(w.n_procs)])
        except (PleWorkerLost, OSError) as exc:
            self._disable(exc)
            return False
        self._auto_ready = True
        self._wstats_seen = self._wstats.tolist()
        logger.info(
            "PLE-DECODE-PREAD autonomous: the %d workers stage every verify round "
            "from the device's post (poll %d us, bonus rows early: %s); the "
            "scheduler thread only posts",
            w.n_procs, _AUTO_POLL_US, "on" if self._bonus else "off",
        )
        return True

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
        global _LIVE_STAGERS, _LIVE_BONUS
        if self._bonus and self._region is not None:
            _LIVE_BONUS = max(0, _LIVE_BONUS - 1)
        self._disable("closed")
        if self._region is not None:
            if self._pinned:
                torch.cuda.cudart().cudaHostUnregister(self._region.data_ptr())
            self._region = None
            self._stage_ids = None
            self._done = None
            self._mbox = None
            self._wstats = None
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
        gate = None
        if self._gated:
            gate = self._gate_state(flat_ids.device, capturing=capturing)
            if gate is None:
                return False
        if capturing:
            # the graph is built before the first round: start the workers
            # now, not inside the first decode round after a flip
            self._ensure_workers()
        elif self._auto and not self._auto_ready:
            # fnFL2 H73: the warm-up forward (never a capture) configures the
            # workers -- the hash constants come from the device
            self._ensure_auto()
        table = self._table
        if gate is not None:
            self._launch_gated(
                flat_ids, output, addrs, counters, gate,
                vocab_start=vocab_start, vocab_end=vocab_end, block_d=block_d,
            )
            return True
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

    def _launch_gated(
        self, flat_ids, output, addrs, counters, gate, *, vocab_start, vocab_end, block_d
    ) -> None:
        """fnFL2 H69: the gate (one warp) then the gated gather, one stream."""
        expect, go, gate_counters, done_addr = gate
        table = self._table
        if self._auto:
            # fnFL2 H73: one done word per worker
            _ple_stage_gate_all_kernel[(1,)](
                expect, done_addr, go, gate_counters,
                N_DONE=self._n_done, MAX_SPINS=self._gate_spins, num_warps=1,
            )
        else:
            _ple_stage_gate_kernel[(1,)](
                expect, done_addr, go, gate_counters, MAX_SPINS=self._gate_spins, num_warps=1,
            )
        _gather_ple_embedding_gated_kernel[(int(flat_ids.numel()),)](
            table.bases_on(flat_ids.device),
            table.shard_rows,
            flat_ids,
            addrs,
            go,
            counters,
            output,
            embedding_dim=table.embedding_dim,
            tp_vocab_start=vocab_start,
            tp_vocab_end=vocab_end,
            is_fp8=table.dtype == torch.float8_e4m3fn,
            BLOCK_D=block_d,
        )

    # -- the gate (fnFL2 H69) ----------------------------------------------------
    def arm_gate(self, seq: int) -> None:
        """Stream-ordered ``expect <- seq`` before the verify is launched: its
        gate waits for ``done >= seq``."""
        for expect, _, _, _ in self._gate_dev.values():
            expect.fill_(int(seq))

    def publish(self, seq: int) -> None:
        """Host side, after the round's rows and ids are written: ``done <- seq``
        (the gate's release). A plain int64 store into page-locked memory.
        fnFL2 H73: never for an autonomous stage -- its done words are the
        workers' own."""
        if self._done is not None and not self._auto:
            self._done[0] = int(seq)

    def disarm_gate(self) -> None:
        """Stream-ordered ``expect <- -1`` after the verify: a later gather that
        is not an armed verify reads through HMM."""
        for expect, _, _, _ in self._gate_dev.values():
            expect.fill_(-1)

    # -- the round ---------------------------------------------------------------
    def snapshot_counters(self) -> None:
        """Stream-ordered copy of the kernel counters (read after the event)."""
        c = self.counters
        if c is not None:
            self._ctr_host.copy_(c, non_blocking=c.is_cuda)
        g = self._gate_dev.get(self._ctr_device)
        if g is not None and self._gate_ctr_host is not None:
            self._gate_ctr_host.copy_(g[2], non_blocking=g[2].is_cuda)

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

    def post(self, ctx: torch.Tensor, seq: int, *, part: int = PART_VERIFY, width: int = 0) -> bool:
        """fnFL2 H73: hand round ``seq`` to the workers, stream-ordered behind
        whatever produced ``ctx`` ([requests, history + tokens], int64): the
        windows into the part's mailbox, THEN the round number into its flag
        (a second copy on the same stream), then one ROUND message per worker.
        Nothing waits. False = not posted (not autonomous, off, a shape the
        mailbox does not hold); the caller then does not arm the gate."""
        if not self._auto or not self.active:
            return False
        if part == PART_BONUS and not self._bonus:
            return False
        if ctx.dim() != 2:
            return False
        bs, cols = int(ctx.shape[0]), int(ctx.shape[1])
        if bs <= 0 or bs * cols > self._ctx_cap or cols > 0xFFFF or not 0 <= int(width) <= 0xFFFF:
            return False
        if not self._ensure_auto():
            return False
        p = self.params()
        n = bs * max(0, cols - p.ngram_size + 1) * (p.ngram_size - 1) * p.heads_per_ngram
        if part == PART_VERIFY and not 0 < n <= self.capacity:
            return False  # the gather will not be the staged one (launch)
        t0 = time.monotonic()
        flag, windows = self._mbox[part]
        seq_t = self._seq_on(ctx.device, part)
        seq_t.fill_(int(seq))
        async_copy = ctx.is_cuda
        windows[: bs * cols].copy_(ctx.reshape(-1), non_blocking=async_copy)
        flag.copy_(seq_t, non_blocking=async_copy)
        log_due = part == PART_VERIFY and self._win["rounds"] + 1 >= self._log_every
        # the workers' stats BEFORE the announcement: every earlier round, none
        # of this one (a worker starts a round only once it is announced)
        workers = self._worker_delta() if log_due else None
        try:
            self._workers.ping(0, part, int(seq), bs, cols, int(width))
        except (PleWorkerLost, OSError) as exc:
            self._disable(exc)
            return False
        if part == PART_VERIFY:
            post_s = time.monotonic() - t0
            win = self._win
            win["rounds"] += 1
            win["rows"] += n
            win["wait_s"] += post_s
            win["wait_max_s"] = max(win["wait_max_s"], post_s)
            self.stats["rounds"] += 1
            self.stats["rows"] += n
            if log_due:
                self._log_window_auto(workers)
        return True

    def _worker_delta(self) -> dict:
        """fnFL2 H73: the workers' stats since the last proof line (the max
        fields are taken and reset: a racing update is lost, never invented)."""
        now = self._wstats.tolist()
        prev = self._wstats_seen or [[0] * STATS_WORDS for _ in now]
        self._wstats_seen = now
        d = {"rounds": 0, "read": 0, "stage_ns": 0, "poll_ns": 0, "kept": 0, "missed": 0,
             "errors": 0, "bonus_read": 0, "stage_max_ns": 0}
        fields = (("rounds", ST_ROUNDS), ("read", ST_READ), ("stage_ns", ST_STAGE_NS),
                  ("poll_ns", ST_POLL_NS), ("kept", ST_REUSED), ("missed", ST_MISSED),
                  ("errors", ST_ERRORS), ("bonus_read", ST_BONUS_READ))
        for i, (a, b) in enumerate(zip(now, prev)):
            for name, k in fields:
                d[name] += a[k] - b[k]
            d["stage_max_ns"] = max(d["stage_max_ns"], a[ST_STAGE_MAX_NS])
            self._wstats[i, ST_STAGE_MAX_NS] = 0
        return d

    def _log_window_auto(self, w: dict) -> None:
        win = self._win
        k_rows, k_hit = self._kernel_delta()
        g_pass, g_timeout = self._gate_delta()
        r = max(1, win["rounds"])
        wr = max(1, w["rounds"])
        logger.info(
            "PLE-DECODE-PREAD rounds=%d rows=%d hit=%d late=%d kernel_rows=%d "
            "kernel_hit=%d wait_ms=%.2f wait_max_ms=%.2f read_ms=%.2f sync_ms=%.2f procs=%d "
            "gate_pass=%d gate_timeout=%d auto=1 stage_max_ms=%.2f poll_ms=%.2f "
            "read_rows=%d kept=%d missed=%d errors=%d bonus_rows=%d",
            win["rounds"], win["rows"], k_hit, max(0, k_rows - k_hit), k_rows, k_hit,
            win["wait_s"] * 1000.0 / r, win["wait_max_s"] * 1000.0,
            w["stage_ns"] / 1e6 / wr, 0.0, self._n_procs, g_pass, g_timeout,
            w["stage_max_ns"] / 1e6, w["poll_ns"] / 1e6 / wr,
            w["read"], w["kept"], w["missed"], w["errors"], w["bonus_read"],
        )
        self._win = self._new_window()

    def _kernel_delta(self) -> tuple:
        now = tuple(int(x) for x in self._ctr_host.tolist())
        prev, self._ctr_seen = self._ctr_seen, now
        if prev is None:
            return now
        return tuple((a - b) % _WRAP32 for a, b in zip(now, prev))

    def _gate_delta(self) -> tuple:
        now = tuple(int(x) for x in self._gate_ctr_host.tolist())
        prev, self._gate_ctr_seen = self._gate_ctr_seen, now
        if prev is None:
            return now
        return tuple((a - b) % _WRAP32 for a, b in zip(now, prev))

    def _log_window(self, procs: int) -> None:
        win = self._win
        k_rows, k_hit = self._kernel_delta()
        r = max(1, win["rounds"])
        gate = ""
        if self._gated:
            # fnFL2 H69: the gates since the last line that found the round
            # staged / gave up (their gather read every row through HMM)
            g_pass, g_timeout = self._gate_delta()
            gate = " gate_pass=%d gate_timeout=%d" % (g_pass, g_timeout)
        logger.info(
            "PLE-DECODE-PREAD rounds=%d rows=%d hit=%d late=%d kernel_rows=%d "
            "kernel_hit=%d wait_ms=%.2f wait_max_ms=%.2f read_ms=%.2f sync_ms=%.2f procs=%d%s",
            win["rounds"], win["rows"], win["hit"], win["late"], k_rows, k_hit,
            win["wait_s"] * 1000.0 / r, win["wait_max_s"] * 1000.0,
            win["read_s"] * 1000.0 / r, win["sync_s"] * 1000.0 / r, procs, gate,
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
    budget_ms = envs.SGLANG_QWEN4_PLE_DECODE_PREAD_BUDGET_MS.get()
    return PleDecodeStager(
        table,
        params_fn,
        vocab_start=vocab_start,
        vocab_end=vocab_end,
        procs=envs.SGLANG_QWEN4_PLE_DECODE_PREAD_PROCS.get(),
        threads=envs.SGLANG_QWEN4_PLE_DECODE_PREAD_THREADS.get(),
        budget_s=budget_ms / 1000.0,
        log_every=envs.SGLANG_QWEN4_PLE_DECODE_PREAD_LOG_EVERY.get(),
        device=device,
        gated=envs.SGLANG_WEG2_PLE_STAGE_BEHIND_REPLAY.get(),
        gate_spins=ple_gate_spins(budget_ms),
        autonomous=envs.SGLANG_WEG2_PLE_STAGE_AUTONOMOUS.get(),
        bonus_early=envs.SGLANG_WEG2_PLE_STAGE_BONUS_EARLY.get(),
    )


def ple_gate_spins(budget_ms: float) -> int:
    """fnFL2 H69: the gate's poll bound for a host budget of ``budget_ms``.
    A poll of page-locked host memory over PCIe takes about a microsecond, so
    the gate waits up to ~``_GATE_SPINS_PER_US`` x the budget -- the host's own
    bound on the stage -- before its gather falls back to HMM."""
    return max(_GATE_SPINS_MIN, int(float(budget_ms) * 1000.0 * _GATE_SPINS_PER_US))


# --------------------------------------------------------------------------
# The verify round: begin after the draft, finish before the forward.
# --------------------------------------------------------------------------


class PleVerifyStage(NamedTuple):
    stagers: tuple
    ctx_host: torch.Tensor
    event: Optional[object]
    #: fnFL2 H69: the round number the gates were armed for (-1 = not armed)
    seq: int = -1
    #: fnFL2 H73: posted to the workers and armed at begin -- nothing for the
    #: host to finish, no hook
    autonomous: bool = False


_MODEL_EMBEDDINGS: dict = {}
_CTX_HOST: dict = {}
#: fnFL2 H69: the last armed round (process-wide, monotone; ``done`` starts at 0)
_GATE_SEQ = 0
#: fnFL2 H73: the last posted bonus part (its own flag, its own numbers)
_BONUS_SEQ = 0


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
    if getattr(stagers[0], "autonomous", False):
        # fnFL2 H73: one switch for the process -- the workers take it from here
        return _post_verify_round(stagers, ctx)
    host = _ctx_host(ctx.shape, ctx.is_cuda)
    host.copy_(ctx, non_blocking=ctx.is_cuda)
    for st in stagers:
        st.snapshot_counters()
    event = None
    if ctx.is_cuda:
        event = torch.cuda.Event()
        event.record()
    return PleVerifyStage(tuple(stagers), host, event)


def _post_verify_round(stagers, ctx: torch.Tensor) -> Optional[PleVerifyStage]:
    """fnFL2 H73: number the round, post it to every autonomous stage and arm
    the gates of those that took it (stream-ordered: counters snapshot, post,
    ``expect``, all before the verify). None when none took it."""
    global _GATE_SEQ
    seq = _GATE_SEQ + 1
    posted = []
    for st in stagers:
        if not getattr(st, "autonomous", False):
            continue
        st.snapshot_counters()
        if st.post(ctx, seq):
            st.arm_gate(seq)
            posted.append(st)
    if not posted:
        return None
    _GATE_SEQ = seq
    return PleVerifyStage(tuple(posted), None, None, seq, True)


def post_ple_bonus_stage(model, pool, batch, bonus_tokens, width: int) -> bool:
    """fnFL2 H73 (``SGLANG_WEG2_PLE_STAGE_BONUS_EARLY``), end of verify, after
    the accept committed the n-gram history: the NEXT verify row starts with
    ``bonus_tokens`` -- post ``[committed history | bonus]`` per request, so
    the workers read those rows (16 per request) into the next round's first
    positions while the draft extend and the next draft run. No wait. False =
    nothing posted (switch off, no autonomous stage in this process, an idle
    batch, a capture, a shape the mailbox does not hold)."""
    global _BONUS_SEQ
    if not _LIVE_BONUS or model is None or pool is None or bonus_tokens is None:
        return False
    stagers = [st for st in _stagers_of(model) if getattr(st, "bonus_early", False)]
    if not stagers:
        return False
    if batch.forward_mode.is_idle():
        return False
    if not hasattr(pool, "get_ngram_context") or not hasattr(pool, "get_mamba_indices"):
        return False
    req = batch.req_pool_indices
    bs = int(req.shape[0])
    if bs <= 0 or int(width) <= 0 or bonus_tokens.numel() != bs:
        return False
    if bonus_tokens.is_cuda and torch.cuda.is_current_stream_capturing():
        return False
    history = pool.get_ngram_context(pool.get_mamba_indices(req).long())
    ctx = torch.cat(
        [history.to(torch.int64), bonus_tokens.reshape(bs, 1).to(torch.int64)], dim=1
    )
    seq = _BONUS_SEQ + 1
    posted = False
    for st in stagers:
        posted = st.post(ctx, seq, part=PART_BONUS, width=int(width)) or posted
    if posted:
        _BONUS_SEQ = seq
    return posted


def finish_ple_verify_stage(stage: Optional[PleVerifyStage]) -> float:
    """Right before the verify forward (fnFL2 H69, gated and graphed: right
    after its launch): wait for the draft's tokens, stage their PLE rows
    (bounded by the budget); an armed round then publishes ``done``. Returns
    the host seconds it took (0.0 without a stage). fnFL2 H73: an autonomous
    round is the workers' -- nothing to do here, nothing published."""
    if stage is None or stage.autonomous:
        return 0.0
    t0 = time.monotonic()
    # fnFL2 H58: ple_sync (the wait for the draft) and ple_stage (hash + pread
    # while the stream idles) of DECODE-HOST-SPLIT; timing only.
    h58_t = time.perf_counter()
    try:
        if stage.event is not None:
            stage.event.synchronize()
        t_ready = time.monotonic()
        h58_t = _h58_span("ple_sync_ms", h58_t)
        rows = stage.ctx_host.tolist()
        for st in stage.stagers:
            st.stage(rows, sync_s=t_ready - t0, t_ready=t_ready)
        _h58_span("ple_stage_ms", h58_t)
    finally:
        if stage.seq >= 0:
            # fnFL2 H69: released even when staging failed -- the ids say
            # -1 for what is not there, so the gather is right either way,
            # and a gate never waits out its bound for a round that is over
            for st in stage.stagers:
                st.publish(stage.seq)
    return time.monotonic() - t0


def ple_stage_is_gated(stage: Optional[PleVerifyStage]) -> bool:
    """fnFL2 H69: whether this round's stage sits behind a device gate that the
    HOST releases (arm, then stage from the post-replay hook). fnFL2 H73: an
    autonomous round is gated too, but armed at begin and released by the
    workers -- False here, so nothing is armed twice or parked on the hook."""
    return (
        stage is not None
        and not stage.autonomous
        and any(st.gated for st in stage.stagers)
    )


def arm_ple_verify_gate(stage: Optional[PleVerifyStage]) -> Optional[PleVerifyStage]:
    """fnFL2 H69, before the verify forward is launched: number the round and
    put ``expect <- seq`` on the stream, so the verify's gate waits for this
    round's ``done``. Returns the stage carrying its ``seq`` (unchanged when
    it is not gated: no stage, switch off)."""
    global _GATE_SEQ
    if not ple_stage_is_gated(stage):
        return stage
    _GATE_SEQ += 1
    seq = _GATE_SEQ
    for st in stage.stagers:
        if st.gated:
            st.arm_gate(seq)
    return stage._replace(seq=seq)


def disarm_ple_verify_gate(stage: Optional[PleVerifyStage]) -> None:
    """fnFL2 H69, after the verify forward was launched: ``expect <- -1`` on
    the stream, behind the verify."""
    if stage is None or stage.seq < 0:
        return
    for st in stage.stagers:
        if st.gated:
            st.disarm_gate()
