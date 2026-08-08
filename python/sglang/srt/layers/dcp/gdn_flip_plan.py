# SPDX-License-Identifier: Apache-2.0
"""Pure GDN/mamba state arithmetic for the #631 PP<->TP phase flip.

The paged full-attention KV moves by (layer, token) ownership
(``phase_flip_plan.py``). GDN conv/ssm state is per REQUEST, not per
token, so its flip has no owner rule at all:

* PP layout: stage ``s`` holds, for each of ITS linear layers, the FULL
  state of every live request (all heads, all conv channels).
* TP layout: every rank holds, for ALL linear layers, its HEAD-SHARD of
  every live request -- and the conv shard is NOT a contiguous slice:
  the GDN conv1d operates on [query | key | value] concatenated along
  conv_dim, and each sub-block is sharded INDEPENDENTLY (the
  ``get_conv_subblock_spec`` finding; a flat slice delivers the wrong
  channels -- the PD mamba transfer bug class).

A (stage ``s``, rank ``r``) pair therefore moves: ``s``'s linear layers
x ``r``'s slices, for every live request. Layer routing reuses the same
per-stage layer-map convention as the KV flip.

This module is pure: the TP sharding arrives as EXPLICIT per-rank shard
vectors (``GdnShardSpec``), derived once at boot by the integration
layer from the process-global partition plan (``tp_partition_sizes``)
and validated here. Everything is a function of replicated inputs.
"""

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch

from sglang.srt.layers.dcp.reshard_plan import KvReshardError


@dataclass(frozen=True)
class GdnShardSpec:
    """Replicated description of the TP-side GDN state sharding.

    ``sub_block_sizes``: full channel count per conv sub-block (GDN:
    ``(key_dim, key_dim, value_dim)``). ``sub_block_shards[b][r]``: rank
    ``r``'s channel count of sub-block ``b`` (each sub-block partitions
    independently). ``head_shards[r]``: rank ``r``'s temporal-state head
    count; ``num_heads`` the unsharded total."""

    sub_block_sizes: Tuple[int, ...]
    sub_block_shards: Tuple[Tuple[int, ...], ...]
    head_shards: Tuple[int, ...]
    num_heads: int

    def __post_init__(self):
        if len(self.sub_block_sizes) != len(self.sub_block_shards):
            raise KvReshardError(
                f"GDN spec: {len(self.sub_block_sizes)} sub-blocks but "
                f"{len(self.sub_block_shards)} shard vectors"
            )
        n_ranks = len(self.head_shards)
        for b, (size, shards) in enumerate(
            zip(self.sub_block_sizes, self.sub_block_shards)
        ):
            if len(shards) != n_ranks:
                raise KvReshardError(
                    f"GDN spec: sub-block {b} has {len(shards)} shard "
                    f"entries for {n_ranks} ranks"
                )
            if sum(shards) != size or any(s < 0 for s in shards):
                raise KvReshardError(
                    f"GDN spec: sub-block {b} shards {shards} do not "
                    f"partition its {size} channels -- a hole or overlap "
                    f"here is the mis-sliced-conv-state bug class"
                )
        if sum(self.head_shards) != self.num_heads or any(
            h < 0 for h in self.head_shards
        ):
            raise KvReshardError(
                f"GDN spec: head shards {self.head_shards} do not "
                f"partition {self.num_heads} heads"
            )

    @property
    def n_ranks(self) -> int:
        return len(self.head_shards)

    @property
    def conv_dim(self) -> int:
        return sum(self.sub_block_sizes)

    def conv_segments(self, rank: int) -> Tuple[Tuple[int, int], ...]:
        """(src offset in the FULL conv dim, length) per sub-block for
        ``rank``'s shard. Destination (compact) offsets are the prefix
        sums of the lengths, in sub-block order."""
        segs = []
        base = 0
        for size, shards in zip(self.sub_block_sizes, self.sub_block_shards):
            off = base + sum(shards[:rank])
            segs.append((off, shards[rank]))
            base += size
        return tuple(segs)

    def conv_shard_channels(self, rank: int) -> int:
        return sum(shards[rank] for shards in self.sub_block_shards)

    def head_range(self, rank: int) -> Tuple[int, int]:
        return sum(self.head_shards[:rank]), self.head_shards[rank]


# -- slice / scatter (single request, single layer) -------------------------


def conv_slice(full: torch.Tensor, spec: GdnShardSpec, rank: int) -> torch.Tensor:
    """Rank's COMPACT conv shard ([q|k|v] sub-shards concatenated) from
    the FULL conv state ``[conv_dim, width]``."""
    if int(full.shape[0]) != spec.conv_dim:
        raise KvReshardError(
            f"conv state has {int(full.shape[0])} channels, spec expects "
            f"{spec.conv_dim}"
        )
    parts = [full[off : off + n] for off, n in spec.conv_segments(rank)]
    return torch.cat(parts, dim=0)


def conv_scatter(
    full: torch.Tensor, shard: torch.Tensor, spec: GdnShardSpec, rank: int
) -> None:
    """Inverse of :func:`conv_slice`: write a rank's compact shard back
    into the FULL conv state at the sub-block source offsets."""
    if int(shard.shape[0]) != spec.conv_shard_channels(rank):
        raise KvReshardError(
            f"conv shard has {int(shard.shape[0])} channels, rank {rank} "
            f"owns {spec.conv_shard_channels(rank)}"
        )
    dst = 0
    for off, n in spec.conv_segments(rank):
        full[off : off + n] = shard[dst : dst + n]
        dst += n


def temporal_slice(
    full: torch.Tensor, spec: GdnShardSpec, rank: int
) -> torch.Tensor:
    """Rank's head shard of the FULL temporal (ssm) state
    ``[num_heads, head_dim, state_size]``."""
    if int(full.shape[0]) != spec.num_heads:
        raise KvReshardError(
            f"temporal state has {int(full.shape[0])} heads, spec expects "
            f"{spec.num_heads}"
        )
    off, n = spec.head_range(rank)
    return full[off : off + n]


def temporal_scatter(
    full: torch.Tensor, shard: torch.Tensor, spec: GdnShardSpec, rank: int
) -> None:
    off, n = spec.head_range(rank)
    if int(shard.shape[0]) != n:
        raise KvReshardError(
            f"temporal shard has {int(shard.shape[0])} heads, rank {rank} "
            f"owns {n}"
        )
    full[off : off + n] = shard


# -- pair payloads (layers ascending, conv then temporal per layer) ---------


def pack_gdn_pair_payload(
    conv_full: Sequence[torch.Tensor],
    temporal_full: Sequence[torch.Tensor],
    spec: GdnShardSpec,
    dst_rank: int,
) -> torch.Tensor:
    """PP->TP sender side: one request's payload for ``dst_rank`` over the
    sender stage's linear layers (``conv_full``/``temporal_full`` in
    ascending layer order). 1-D uint8; per layer: compact conv shard
    bytes, then temporal head-shard bytes."""
    if len(conv_full) != len(temporal_full):
        raise KvReshardError(
            f"{len(conv_full)} conv layers vs {len(temporal_full)} "
            f"temporal layers"
        )
    parts = []
    for conv, temp in zip(conv_full, temporal_full):
        parts.append(
            conv_slice(conv, spec, dst_rank).contiguous().view(torch.uint8).reshape(-1)
        )
        parts.append(
            temporal_slice(temp, spec, dst_rank)
            .contiguous()
            .view(torch.uint8)
            .reshape(-1)
        )
    if not parts:
        return torch.empty(0, dtype=torch.uint8)
    return torch.cat(parts)


def unpack_gdn_pair_payload(
    payload: torch.Tensor,
    conv_full: Sequence[torch.Tensor],
    temporal_full: Sequence[torch.Tensor],
    spec: GdnShardSpec,
    my_rank: int,
) -> None:
    """TP->PP receiver side: scatter a peer rank's payload (built by
    :func:`pack_gdn_pair_payload` semantics from ITS shard) back into
    this stage's FULL per-layer states. ``my_rank`` here is the SENDING
    rank whose shard the payload carries."""
    offset = 0
    for conv, temp in zip(conv_full, temporal_full):
        c_channels = spec.conv_shard_channels(my_rank)
        c_bytes = c_channels * int(conv.shape[1]) * conv.element_size()
        chunk = payload[offset : offset + c_bytes]
        if chunk.numel() != c_bytes:
            raise KvReshardError(
                f"GDN payload truncated: wanted {c_bytes} conv bytes, got "
                f"{chunk.numel()}"
            )
        shard = chunk.view(conv.dtype).view(c_channels, int(conv.shape[1]))
        conv_scatter(conv, shard, spec, my_rank)
        offset += c_bytes
        h_off, h_n = spec.head_range(my_rank)
        t_bytes = (
            h_n * int(temp.shape[1]) * int(temp.shape[2]) * temp.element_size()
        )
        chunk = payload[offset : offset + t_bytes]
        if chunk.numel() != t_bytes:
            raise KvReshardError(
                f"GDN payload truncated: wanted {t_bytes} temporal bytes, "
                f"got {chunk.numel()}"
            )
        shard = chunk.view(temp.dtype).view(
            h_n, int(temp.shape[1]), int(temp.shape[2])
        )
        temporal_scatter(temp, shard, spec, my_rank)
        offset += t_bytes
    if offset != payload.numel():
        raise KvReshardError(
            f"GDN payload has {payload.numel() - offset} trailing bytes -- "
            f"layer count or shapes diverged between sender and receiver"
        )
