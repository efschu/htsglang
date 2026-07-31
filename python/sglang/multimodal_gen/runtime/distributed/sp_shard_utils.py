# SPDX-License-Identifier: Apache-2.0
"""Unified sequence-parallel shard / pad / gather helpers.

Layout invariant: padding always sits at the end of the LAST rank's local
chunk, so the ulysses-gathered sequence carries one contiguous pad block at its
global tail. `tail_attn_meta` then lets attention skip that block for free
(the pad becomes its own varlen segment - no repacking, no mask compute).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_sp_parallel_rank,
    get_sp_world_size,
)

# Text shorter than this stays replicated instead of SP-sharded (see
# plan_text_strategy). 0 = always shard when legal; H100 bench showed sharding
# wins from trivial lengths on, so the knob exists only as an escape hatch.
_TEXT_SHARD_MIN = int(os.environ.get("SGLANG_SP_TEXT_SHARD_MIN", "0"))

# Per-rank capacity weights for the heterogeneous (uneven) sequence-parallel
# split. Comma-separated, one positive float per SP rank, e.g. "1.0,0.46,0.46"
# for a 5090 + 2x3080 rig. The registry's Class-2 adapter fills this from the
# same per-card GEMM rates the K1 uneven-TP planner uses
# (``sglang.srt.uneven_perf.rank_gemm_scores`` -> ``gemm_tflops``), so the
# faster card is handed a proportionally longer slice of the sequence. Absent
# or malformed, the split stays the classic equal-and-tail-padded one and every
# existing call is byte-identical. This is the ONLY entry point for uneven SP:
# nothing else in this module reaches into the registry, so the file stays
# hermetic and CPU-testable.
_CAPACITY_WEIGHTS_ENV = "SGLANG_SP_CAPACITY_WEIGHTS"


@dataclass(frozen=True)
class SpShard:
    """Facts of one sequence shard, shared by tensors of that stream.

    Two shard schemes live behind one type:

    * The classic **equal** scheme: every rank gets the same ``local_len`` by
      ceil-division and the remainder is padded onto the LAST rank's tail, so
      the ulysses-gathered sequence carries one contiguous pad block at its
      global tail. ``uneven`` is False and ``offsets``/``lens`` are empty; the
      geometry is fully described by ``local_len``/``num_pad`` exactly as
      before.
    * The **uneven** (capacity-weighted) scheme: each rank owns a contiguous,
      differently sized real slice ``[offsets[r], offsets[r] + lens[r])`` with
      no padding at all (``num_pad == 0``). ``sum(lens) == orig_len`` exactly,
      so the split covers the sequence with no gap and no overlap; the faster
      card simply owns a longer slice. Padding is unnecessary because the real
      lengths already sum to the whole, and dropping it is what removes the
      dead compute the equal scheme spends on its ceil-division slack.
    """

    orig_len: int  # real tokens (global)
    local_len: int  # THIS rank's chunk length (equal scheme: same on every rank)
    num_pad: int  # pad tokens, all at the last rank's local tail (0 when uneven)
    sp_size: int
    sp_rank: int
    #: Uneven scheme only: per-rank real slice offset and length. Empty for the
    #: equal scheme, which needs no table because the stride is uniform.
    offsets: tuple[int, ...] = field(default_factory=tuple)
    lens: tuple[int, ...] = field(default_factory=tuple)
    uneven: bool = False

    @property
    def local_pad(self) -> int:
        """Pad rows inside THIS rank's chunk (tail rows of the last rank)."""
        if self.uneven:
            return 0
        return self.num_pad if self.sp_rank == self.sp_size - 1 else 0

    @property
    def local_real_len(self) -> int:
        return self.local_len - self.local_pad

    @property
    def local_offset(self) -> int:
        """Start of THIS rank's real slice in the global sequence."""
        if self.uneven:
            return self.offsets[self.sp_rank]
        return self.sp_rank * self.local_len


def _apportion(total: int, weights: Sequence[float]) -> list[int]:
    """Split ``total`` into per-rank integer counts proportional to ``weights``.

    Largest-remainder (Hamilton) apportionment: floor every ideal share, then
    hand the leftover units to the ranks with the largest fractional parts. The
    result sums to ``total`` exactly, which is the coverage invariant the whole
    uneven split rests on -- no unit is created or lost.

    Every rank is guaranteed at least one token when ``total >= len(weights)``:
    a rank with an empty sequence slice has no work and no attention rows, which
    is a degenerate SP configuration, not a split. Below that floor the request
    is too short to spread over this many ranks and the caller falls back to the
    equal scheme.
    """
    n = len(weights)
    if any(w <= 0 for w in weights):
        raise ValueError(f"capacity weights must be positive; got {list(weights)}")
    total_w = math.fsum(weights)
    ideal = [total * w / total_w for w in weights]
    counts = [int(math.floor(x)) for x in ideal]
    remainder = total - sum(counts)
    order = sorted(range(n), key=lambda i: ideal[i] - counts[i], reverse=True)
    for k in range(remainder):
        counts[order[k]] += 1
    # Guarantee >= 1 per rank by moving units off the largest holders.
    if total >= n:
        donors = sorted(range(n), key=lambda i: counts[i], reverse=True)
        d = 0
        for i in range(n):
            if counts[i] == 0:
                while counts[donors[d]] <= 1:
                    d += 1
                counts[donors[d]] -= 1
                counts[i] = 1
    return counts


def capacity_weights_from_env(sp_size: int) -> tuple[float, ...] | None:
    """Read the per-rank capacity weights, or None for the equal split.

    Returns None -- keeping the classic equal split -- unless the environment
    declares exactly ``sp_size`` positive weights. A malformed or wrong-length
    value is a configuration error the caller should see, so it raises rather
    than silently falling back: a rig that meant to run uneven and typed the
    vector wrong must not quietly run equal and blame the hardware.
    """
    raw = os.environ.get(_CAPACITY_WEIGHTS_ENV)
    if not raw:
        return None
    try:
        weights = tuple(float(p) for p in raw.split(",") if p.strip() != "")
    except ValueError as exc:
        raise ValueError(
            f"{_CAPACITY_WEIGHTS_ENV}={raw!r} is not a comma-separated float list"
        ) from exc
    if len(weights) != sp_size:
        raise ValueError(
            f"{_CAPACITY_WEIGHTS_ENV} has {len(weights)} weights but SP world size "
            f"is {sp_size}; declare exactly one weight per rank"
        )
    if any(w <= 0 for w in weights):
        raise ValueError(f"{_CAPACITY_WEIGHTS_ENV} weights must be positive: {raw!r}")
    return weights


def build_shard_plan(
    seq_len: int, weights: Sequence[float] | None = None
) -> SpShard:
    """Shard math only; tensors are sliced separately via `shard_like`.

    ``weights`` selects the scheme. ``None`` (the default) reads the capacity
    vector from the environment; if that too is unset -- the overwhelmingly
    common path -- the split is the classic equal-and-tail-padded one and this
    function is byte-identical to its pre-uneven form. A weight vector (explicit
    or from the environment) switches to the capacity-weighted split.
    """
    sp_size = get_sp_world_size()
    if sp_size <= 1:
        return SpShard(seq_len, seq_len, 0, 1, 0)
    if weights is None:
        weights = capacity_weights_from_env(sp_size)
    if weights is not None and seq_len >= sp_size:
        return _build_uneven_plan(seq_len, sp_size, weights)
    # Equal scheme: identical to the original implementation.
    local_len = (seq_len + sp_size - 1) // sp_size
    return SpShard(
        orig_len=seq_len,
        local_len=local_len,
        num_pad=local_len * sp_size - seq_len,
        sp_size=sp_size,
        sp_rank=get_sp_parallel_rank(),
    )


def _build_uneven_plan(
    seq_len: int, sp_size: int, weights: Sequence[float]
) -> SpShard:
    if len(weights) != sp_size:
        raise ValueError(
            f"capacity weights has {len(weights)} entries but SP world size is "
            f"{sp_size}"
        )
    lens = _apportion(seq_len, weights)
    offsets: list[int] = []
    running = 0
    for length in lens:
        offsets.append(running)
        running += length
    sp_rank = get_sp_parallel_rank()
    return SpShard(
        orig_len=seq_len,
        local_len=lens[sp_rank],
        num_pad=0,
        sp_size=sp_size,
        sp_rank=sp_rank,
        offsets=tuple(offsets),
        lens=tuple(lens),
        uneven=True,
    )


def shard_like(
    x: torch.Tensor, shard: SpShard, dim: int = 1, pad_mode: str = "zeros"
) -> torch.Tensor:
    """Apply a planned shard to one tensor (RoPE caches use the same plan as
    hidden states so their chunks stay aligned)."""
    if shard.sp_size <= 1:
        return x
    if shard.uneven:
        # Contiguous, differently sized real slice; no padding to add or skip.
        return x.narrow(dim, shard.local_offset, shard.local_len)
    if shard.num_pad > 0:
        if pad_mode == "repeat_last":
            pad = x.narrow(dim, x.shape[dim] - 1, 1)
            pad = pad.expand(
                *[shard.num_pad if i == dim else -1 for i in range(x.dim())]
            )
            x = torch.cat([x, pad], dim=dim)
        else:
            # F.pad pads dims last-to-first: (left, right) pairs from dim -1.
            pads = [0, 0] * (x.dim() - 1 - dim) + [0, shard.num_pad]
            x = F.pad(x, pads)
    return x.narrow(dim, shard.sp_rank * shard.local_len, shard.local_len)


def shard_seq(
    x: torch.Tensor, dim: int = 1, pad_mode: str = "zeros"
) -> tuple[torch.Tensor, SpShard]:
    """
    mode:
        zeroes: pad with zeroes at tail
        repeat_last: repeat the last token, only for rotary embedding
    """
    shard = build_shard_plan(x.shape[dim])
    return shard_like(x, shard, dim=dim, pad_mode=pad_mode), shard


def gather_seq(
    local: torch.Tensor,
    orig_len: int,
    dim: int = 1,
    shard: SpShard | None = None,
) -> torch.Tensor:
    """All-gather an SP-sharded sequence and trim padding.

    Equal scheme (``shard`` is None or not uneven): every rank contributed an
    identical ``local_len`` chunk, so a plain all-gather reassembles the
    sequence and the single global-tail pad block is trimmed to ``orig_len``.

    Uneven scheme: ranks contributed differently sized chunks. NCCL's
    fixed-size all-gather cannot carry those directly, so each rank's chunk is
    padded to the common maximum, gathered, and each rank's real slice
    (``shard.lens[r]``) is cut back out and concatenated in rank order. The
    result is the original sequence exactly -- padding costs bandwidth on the
    shorter ranks but never appears in the output. The zero-copy variable-length
    all-gather that would avoid that bandwidth is the M4 collective change
    (see DESIGN_333_M3 §3); it does not change this result, only its cost.
    """
    if get_sp_world_size() <= 1:
        return local
    if shard is not None and shard.uneven:
        max_len = max(shard.lens)
        pad = max_len - local.shape[dim]
        if pad > 0:
            pads = [0, 0] * (local.dim() - 1 - dim) + [0, pad]
            local = F.pad(local, pads)
        gathered = sequence_model_parallel_all_gather(local.contiguous(), dim=dim)
        pieces = [
            gathered.narrow(dim, rank * max_len, shard.lens[rank])
            for rank in range(shard.sp_size)
        ]
        return torch.cat(pieces, dim=dim)
    full = sequence_model_parallel_all_gather(local.contiguous(), dim=dim)
    if full.shape[dim] > orig_len:
        full = full.narrow(dim, 0, orig_len)
    return full


def shard_seq_prefix(
    x: torch.Tensor, prefix_len: int, shard: SpShard, dim: int = 0
) -> torch.Tensor:
    """Shard only the leading ``prefix_len`` rows (e.g. the text segment of a
    joint RoPE cache) with an existing plan; the remainder is kept as-is."""
    rest = x.shape[dim] - prefix_len
    return torch.cat(
        [
            shard_like(x.narrow(dim, 0, prefix_len), shard, dim=dim),
            x.narrow(dim, prefix_len, rest),
        ],
        dim=dim,
    )


def join_seqs(
    prefix: torch.Tensor, body: torch.Tensor, local_pad: int, dim: int = 1
) -> torch.Tensor:
    """Concatenate local sharded ``[prefix (txt tokens, padding tokens), body (img tokens)]`` for joint attention, while relocating the
     prefix's ``local_pad`` tail rows behind the body.

    Why leave the padding at tail: the shard pads the *text* chunk, but the local joint layout is
    [text, image].

    In naive implementation, after the ulysses all-to-all, that pad would sit mid-sequence (of last rank)
    ([... txt_last, PAD, img_last]), which required further mem copy (for the padding tokens), inefficient in this case

    With the pad relocated behind the image, the padding forms one global-tail block that the zero-copy varlen
    path (tail_attn_meta, implemented in USPAttention.forward) skips for free
    """
    if local_pad > 0:
        real = prefix.shape[dim] - local_pad
        return torch.cat(
            [
                # txt tokens
                prefix.narrow(dim, 0, real),
                body,
                # leave the padding at global-tail
                prefix.narrow(dim, real, local_pad),
            ],
            dim=dim,
        )
    return torch.cat([prefix, body], dim=dim)


def split_seqs(
    joint: torch.Tensor, prefix_len: int, local_pad: int, dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of ``join_seqs``: recover ``(prefix, body)`` from the joint output, with the pad rows rejoining the prefix tail so the residual text
    stream keeps its per-rank shape.

     ([... txt_last, PAD, img_last]) -> prefix (txt + pad), body (img)
    """
    total = joint.shape[dim]
    if local_pad > 0:
        real = prefix_len - local_pad
        body_end = total - local_pad
        prefix = torch.cat(
            [joint.narrow(dim, 0, real), joint.narrow(dim, body_end, local_pad)],
            dim=dim,
        )
        return prefix, joint.narrow(dim, real, body_end - real)
    return (
        joint.narrow(dim, 0, prefix_len),
        joint.narrow(dim, prefix_len, total - prefix_len),
    )


def should_shard_text(txt_len: int) -> bool:
    """True when the joint-attention text stream should be SP-sharded here
    (see plan_text_strategy for the policy)."""
    return get_sp_world_size() > 1 and plan_text_strategy(txt_len) == "shard"


def tail_attn_meta(
    shard: SpShard,
    batch_size: int,
    device: torch.device,
    image_seq_len: int = 0,
) -> dict | None:
    """Per-request attention meta for a tail-padded shard: `cu_seqlens_tail`
    splits each batch row into [valid | pad] varlen segments over the gathered
    layout, so USPAttention runs varlen FA on the padded q/k/v with zero
    repacking. Built once per request, reused by every block."""
    if shard.sp_size <= 1 or shard.num_pad == 0:
        return None
    seq = shard.sp_size * (shard.local_len + image_seq_len)
    valid = seq - shard.num_pad
    row = torch.tensor([valid, shard.num_pad], dtype=torch.int32, device=device)
    seglens = row.repeat(batch_size)
    cu_seqlens = torch.zeros(2 * batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(seglens, dim=0)
    return {
        "pad_start": valid,
        "pad_end": seq,
        "local_pad": shard.local_pad,
        "cu_seqlens_tail": cu_seqlens,
        "max_seqlen_tail": max(valid, shard.num_pad),
    }


def plan_text_strategy(txt_len: int) -> str:
    """Choose "shard" or "replicate" for the joint-attention text stream.

    Prefer "shard" by default. for small sequence (shorter than SGLANG_SP_TEXT_SHARD_MIN), choose "replicate" for better performance

    """
    sp_size = get_sp_world_size()
    if sp_size <= 1:
        return "replicate"
    if txt_len % sp_size != 0 and get_ring_parallel_world_size() > 1:
        return "replicate"
    if txt_len < _TEXT_SHARD_MIN:
        return "replicate"
    return "shard"
