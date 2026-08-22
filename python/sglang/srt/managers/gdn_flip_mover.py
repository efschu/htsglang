# SPDX-License-Identifier: Apache-2.0
"""Production GDN/mamba state mover for the #631 phase flip (slice 5.3b).

Moves linear-attention conv/ssm state between the PP layout (stage owns
its linear layers, FULL heads/channels) and the TP layout (every rank
owns ALL linear layers, head-sharded) at the flip boundary, using the
pure primitives of ``layers/dcp/gdn_flip_plan.py``. Runs as a
``pre_cutover_fn`` of PhaseFlipRuntime, inside the no-return region,
BEFORE the weights refill.

HONESTY CONTRACT (operator hold, 2026-08-08): the 5.3 placeholder's loud
refusal stays REACHABLE. ``gdn_flip_preconditions`` re-validates every
structural assumption (single-conv GDN layout, derivable sub-block spec,
shard vectors matching the TP pool's ACTUAL tensor shapes, equal slot
spaces, no replayssm ring buffers, no int8 mamba checkpoint pool, no
speculative scratch expectations) on EVERY flip, and any violation is a
loud KvReshardError before a byte moves -- a future regression degrades
to refusal, never to a silent no-move (#212 Store-Route lesson).

Payload convention per (stage s, shard rank r) pair, both directions:
slots ascending; per slot, s's linear layers ascending; per layer,
compact conv shard bytes then temporal head-shard bytes
(``pack_gdn_pair_payload`` semantics); one 8-byte checksum trailer per
pair. Receivers derive expected byte counts from replicated inputs only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.layers.dcp.gdn_flip_plan import (
    GdnShardSpec,
    conv_slice,
    conv_scatter,
    temporal_slice,
    temporal_scatter,
)
from sglang.srt.layers.dcp.phase_flip_plan import PP_TO_TP, TP_TO_PP
from sglang.srt.layers.dcp.reshard_plan import KvReshardError
from sglang.srt.model_executor.weights_arena import (
    checksum_is_representable,
    uint8_checksum,
)
from sglang.srt.managers.kv_reshard import _CHECKSUM_BYTES, _checksum

logger = logging.getLogger(__name__)

LOG_PREFIX = "PHASE-FLIP-GDN"


def derive_pp_linear_layer_map(
    linear_layer_ids: Sequence[int], num_hidden_layers: int, pp_size: int
) -> Tuple[Tuple[int, ...], ...]:
    """Per-stage GLOBAL linear-layer ids -- the GDN sibling of
    derive_pp_full_attn_layer_map (same get_pp_indices bounds, but global
    ids because pool addressing goes through each pool's mamba_map)."""
    from sglang.srt.distributed.utils import get_pp_indices

    ids = [int(x) for x in linear_layer_ids]
    if ids != sorted(set(ids)):
        raise KvReshardError(f"linear layer ids must be strictly ascending, got {ids}")
    bounds = [get_pp_indices(num_hidden_layers, r, pp_size) for r in range(pp_size)]
    if bounds[0][0] != 0 or bounds[-1][1] != num_hidden_layers:
        raise KvReshardError(
            f"PP stage bounds {bounds} do not span [0, {num_hidden_layers})"
        )
    stage_ids = tuple(
        tuple(gid for gid in ids if start <= gid < end) for start, end in bounds
    )
    covered = sorted(g for stage in stage_ids for g in stage)
    if covered != ids:
        raise KvReshardError(
            f"stage map {stage_ids} does not cover the linear ids exactly "
            f"once (bounds {bounds})"
        )
    return stage_ids


@dataclass(frozen=True)
class GdnFlipPools:
    """Validated handles into both layouts' mamba pools.

    PP side: FULL-width per-layer state for THIS stage's linear layers.
    TP side: this rank's COMPACT shard for ALL linear layers. ``*_index``
    maps a GLOBAL linear layer id to the pool's layer axis index."""

    spec: GdnShardSpec
    pp_conv: torch.Tensor  # [L_stage, S, conv_dim_full, W]
    pp_temporal: torch.Tensor  # [L_stage, S, H_full, D, N]
    pp_index: Dict[int, int]
    tp_conv: torch.Tensor  # [L_all, S, conv_shard, W]
    tp_temporal: torch.Tensor  # [L_all, S, H_shard, D, N]
    tp_index: Dict[int, int]
    n_slots: int


def gdn_flip_preconditions(
    pp_req_pool, tp_req_pool, n_ranks: int, rank: int, gdn_geometry=None
) -> GdnFlipPools:
    """Validate every structural assumption; refuse loudly otherwise.

    THE reachable-refusal seam: called on every flip, before any byte
    moves. Takes the two HybridReqToTokenPools (which own mamba_pool and
    mamba_map). Returns the validated pool handles + the shard spec
    derived from the process-global plan and CHECKED against the TP
    pool's actual tensor shapes -- a spec/pool disagreement is the
    mis-sliced-conv-state bug class and dies here.

    ``gdn_geometry`` is the model's hybrid HF text config (or any object
    with ``linear_num_key_heads``, ``linear_num_value_heads`` and the
    model-stashed ``gdn_tp_units``). When given, the shard derivation
    replicates the MODEL's own head split -- tp_partition_size over HEAD
    counts in gdn_tp_units, exactly the qwen3_5 local_num_*_heads calls.
    Splitting the channel totals instead (the legacy fallback below)
    rounds differently and produced head shards (22, 12, 12) for 48 heads
    on the first real-metal flip (2026-08-08): the uneven-TP-head-geometry
    class -- per-rank geometry comes from the model, never recomputed."""
    from sglang.srt.distributed.utils import tp_partition_size

    reasons: List[str] = []

    def _fail():
        raise KvReshardError(
            f"{LOG_PREFIX} refusing to move GDN state: "
            f"{'; '.join(reasons)}. A flip must never proceed past a "
            f"broken linear-state assumption (silent truncation, #212)."
        )

    pp_pool = pp_req_pool.mamba_pool
    tp_pool = tp_req_pool.mamba_pool
    for name, req_pool, pool in (
        ("PP", pp_req_pool, pp_pool),
        ("TP", tp_req_pool, tp_pool),
    ):
        if getattr(pool, "enable_linear_replayssm", False) or (
            getattr(pool.mamba_cache, "replayssm_d", None) is not None
        ):
            reasons.append(
                f"{name} pool has ReplaySSM ring buffers (mover does not "
                f"carry them yet -- V1 scope)"
            )
        if getattr(req_pool, "mamba_ckpt_pool", None) is not None:
            reasons.append(
                f"{name} side has an int8 mamba checkpoint pool (cached "
                f"prefix states would not be moved -- V1 scope)"
            )
    if reasons:
        _fail()

    pp_conv_list = pp_pool.mamba_cache.conv
    tp_conv_list = tp_pool.mamba_cache.conv
    if len(pp_conv_list) != 1 or len(tp_conv_list) != 1:
        reasons.append(
            f"expected the single-conv GDN layout, got "
            f"{len(pp_conv_list)}/{len(tp_conv_list)} conv tensors"
        )
        _fail()
    pp_conv, tp_conv = pp_conv_list[0], tp_conv_list[0]
    pp_temporal, tp_temporal = (
        pp_pool.mamba_cache.temporal,
        tp_pool.mamba_cache.temporal,
    )

    spec_tuple = pp_pool.get_conv_subblock_spec()
    if spec_tuple is None:
        reasons.append(
            "PP pool does not expose the GDN [q|k|v] conv structure "
            "(get_conv_subblock_spec returned None)"
        )
        _fail()
    sub_sizes, units, conv_dim = spec_tuple
    if int(pp_conv.shape[2]) != conv_dim:
        reasons.append(
            f"PP conv tensor has {int(pp_conv.shape[2])} channels, spec "
            f"says {conv_dim} -- the PP pool must hold the FULL width"
        )
        _fail()

    num_heads = int(pp_temporal.shape[2])
    value_dim = sub_sizes[-1]
    if value_dim % num_heads != 0:
        reasons.append(
            f"value sub-block ({value_dim}) not divisible by temporal "
            f"heads ({num_heads})"
        )
        _fail()
    if gdn_geometry is not None:
        # Model-identical head split (see docstring).
        num_k_heads = int(gdn_geometry.linear_num_key_heads)
        num_v_heads = int(gdn_geometry.linear_num_value_heads)
        gdn_units = getattr(gdn_geometry, "gdn_tp_units", None)
        if num_v_heads != num_heads:
            reasons.append(
                f"config says {num_v_heads} linear value heads but the PP "
                f"temporal pool holds {num_heads}"
            )
            _fail()
        if sub_sizes[0] % num_k_heads != 0:
            reasons.append(
                f"key sub-block ({sub_sizes[0]}) not divisible by "
                f"{num_k_heads} key heads"
            )
            _fail()
        head_k_dim = sub_sizes[0] // num_k_heads
        head_v_dim = value_dim // num_v_heads
        k_shards = tuple(
            tp_partition_size(num_k_heads, n_ranks, r, gdn_units)
            for r in range(n_ranks)
        )
        head_shards = tuple(
            tp_partition_size(num_v_heads, n_ranks, r, gdn_units)
            for r in range(n_ranks)
        )
        sub_shards = (
            tuple(k * head_k_dim for k in k_shards),
            tuple(k * head_k_dim for k in k_shards),
            tuple(v * head_v_dim for v in head_shards),
        )
    else:
        sub_shards = tuple(
            tuple(
                tp_partition_size(sub_full, n_ranks, r, units) for r in range(n_ranks)
            )
            for sub_full in sub_sizes
        )
        head_dim_units = value_dim // num_heads
        head_shards = tuple(s // head_dim_units for s in sub_shards[-1])
    spec = GdnShardSpec(
        sub_block_sizes=tuple(sub_sizes),
        sub_block_shards=sub_shards,
        head_shards=head_shards,
        num_heads=num_heads,
    )

    # The derived spec must match the TP pool's ACTUAL shapes.
    if int(tp_conv.shape[2]) != spec.conv_shard_channels(rank):
        reasons.append(
            f"TP conv tensor has {int(tp_conv.shape[2])} channels but the "
            f"plan gives rank {rank} {spec.conv_shard_channels(rank)}"
        )
    if int(tp_temporal.shape[2]) != spec.head_shards[rank]:
        reasons.append(
            f"TP temporal tensor has {int(tp_temporal.shape[2])} heads but "
            f"the plan gives rank {rank} {spec.head_shards[rank]}"
        )
    if int(pp_conv.shape[1]) != int(tp_conv.shape[1]):
        reasons.append(
            f"slot spaces differ (PP {int(pp_conv.shape[1])} vs TP "
            f"{int(tp_conv.shape[1])}) -- the flip relies on slot-id "
            f"identity across layouts"
        )
    if pp_conv.dtype != tp_conv.dtype or pp_temporal.dtype != tp_temporal.dtype:
        reasons.append("state dtypes differ between layouts")
    if reasons:
        _fail()

    return GdnFlipPools(
        spec=spec,
        pp_conv=pp_conv,
        pp_temporal=pp_temporal,
        pp_index=dict(pp_req_pool.mamba_map),
        tp_conv=tp_conv,
        tp_temporal=tp_temporal,
        tp_index=dict(tp_req_pool.mamba_map),
        n_slots=int(pp_conv.shape[1]),
    )


def _pair_nbytes(
    spec: GdnShardSpec,
    shard_rank: int,
    n_layers: int,
    n_slots: int,
    conv_width: int,
    conv_elem: int,
    temporal_tail: int,
    temporal_elem: int,
) -> int:
    conv_b = spec.conv_shard_channels(shard_rank) * conv_width * conv_elem
    temp_b = spec.head_shards[shard_rank] * temporal_tail * temporal_elem
    return (conv_b + temp_b) * n_layers * n_slots


class GdnFlipMover:
    """One rank's GDN leg of the flip. ``exchange`` has the KV runtime's
    channel contract (outgoing payloads -> received payloads by peer)."""

    def __init__(
        self,
        *,
        n_ranks: int,
        rank: int,
        stage_layer_ids: Sequence[Sequence[int]],
        pools_fn: Callable[[], GdnFlipPools],
        slots_fn: Callable[[], torch.Tensor],
        exchange: Callable[
            [Dict[int, torch.Tensor], Dict[int, int]], Dict[int, torch.Tensor]
        ],
    ):
        self._n = int(n_ranks)
        self._rank = int(rank)
        self._stage_ids = tuple(tuple(int(g) for g in s) for s in stage_layer_ids)
        if len(self._stage_ids) != self._n:
            raise KvReshardError(f"{len(self._stage_ids)} stages for {self._n} ranks")
        self._pools_fn = pools_fn
        self._slots_fn = slots_fn
        self._exchange = exchange
        self.last_stats: Optional[dict] = None

    # -- helpers -------------------------------------------------------------

    def _slot_list(self) -> List[int]:
        slots = self._slots_fn()
        slots = torch.unique(slots.detach().to("cpu", torch.int64))
        return [int(s) for s in slots.tolist()]

    def _pack_pp_side(
        self, pools: GdnFlipPools, slots: List[int], dst: int
    ) -> torch.Tensor:
        """My stage's layers x dst's shard, slots ascending (PP->TP)."""
        parts = []
        for s in slots:
            for gid in self._stage_ids[self._rank]:
                li = pools.pp_index[gid]
                parts.append(
                    conv_slice(pools.pp_conv[li, s], pools.spec, dst)
                    .contiguous()
                    .view(torch.uint8)
                    .reshape(-1)
                )
                parts.append(
                    temporal_slice(pools.pp_temporal[li, s], pools.spec, dst)
                    .contiguous()
                    .view(torch.uint8)
                    .reshape(-1)
                )
        flat = torch.cat(parts) if parts else torch.empty(0, dtype=torch.uint8)
        return torch.cat([flat, _checksum(flat)])

    def _pack_tp_side(
        self, pools: GdnFlipPools, slots: List[int], dst_stage: int
    ) -> torch.Tensor:
        """dst_stage's layers x MY shard, from my compact tensors (TP->PP)."""
        parts = []
        for s in slots:
            for gid in self._stage_ids[dst_stage]:
                li = pools.tp_index[gid]
                parts.append(
                    pools.tp_conv[li, s].contiguous().view(torch.uint8).reshape(-1)
                )
                parts.append(
                    pools.tp_temporal[li, s].contiguous().view(torch.uint8).reshape(-1)
                )
        flat = torch.cat(parts) if parts else torch.empty(0, dtype=torch.uint8)
        return torch.cat([flat, _checksum(flat)])

    def _verify(self, payload: torch.Tensor, want: int, peer: int) -> torch.Tensor:
        if payload is None or payload.numel() != want:
            got = 0 if payload is None else payload.numel()
            raise KvReshardError(
                f"{LOG_PREFIX} peer {peer} sent {got} bytes, expected "
                f"{want} -- layer map, shard spec or slot set diverged"
            )
        data = payload[:-_CHECKSUM_BYTES]
        stored = int(payload[-_CHECKSUM_BYTES:].clone().view(torch.int64).item())
        # #802: IS THE TRAILER A CHECKSUM AT ALL? Ask before comparing.
        #
        # `uint8_checksum` is an exact int64 sum of UNSIGNED bytes, so its
        # value always lies in [0, 255 * nbytes] -- it can never be negative
        # and can never exceed that ceiling. A trailer outside that range was
        # therefore never written by a sender at all: the bytes read as a
        # checksum are something else, which means the payload was not framed
        # the way this reader expected. The receive buffer in
        # `kv_reshard._dist_exchange` is `torch.empty` and is never
        # pre-zeroed, so an under-filled or mis-framed receive leaves the
        # ORIGINAL GARBAGE exactly where this trailer is read from -- and the
        # length check above cannot see it, because an under-filled buffer
        # still has the allocated `numel`.
        #
        # WHY THIS IS ITS OWN ERROR AND NOT A MISMATCH. The two states need
        # different readers and different next steps:
        #   in range, different  -> the two ends framed the payload the same
        #                           way and got different sums. The DATA is
        #                           wrong. That is what this guard is for.
        #   out of range         -> nobody wrote a checksum there. The data is
        #                           not the thing that is wrong; the transport
        #                           or the geometry is.
        # Reporting the second as the first kills an instance for a corruption
        # that did not happen. `checksum_is_representable` was written for
        # exactly this discrimination (see its docstring: #656 register C22 is
        # the same misreport, with a negative "sender" checksum) and this call
        # site never asked it. Measured again 2026-08-22 17:22:44: PP0 died on
        # `stored=-4664535355886616603` -- negative, impossible for any sum of
        # unsigned bytes -- reported as a data corruption. Specimen
        # /spinning/evidence-800/crash_1722_gdn_checksum/.
        if not checksum_is_representable(stored, data.numel()):
            raise KvReshardError(
                f"{LOG_PREFIX} payload from peer {peer} carries NO CHECKSUM in "
                f"its trailer: the last {_CHECKSUM_BYTES} bytes read as "
                f"{stored}, which is outside [0, {255 * data.numel()}] and so "
                f"cannot be uint8_checksum of ANY {data.numel()}-byte payload. "
                "The trailer was never written, so this rank was handed a "
                "buffer that was not filled the way it expected -- an "
                "under-filled receive (the recv buffer is torch.empty and is "
                "never pre-zeroed, and the length check passes on an allocated "
                "but unfilled buffer) or a framing/geometry divergence, whose "
                "byte counts are derived independently per rank and never "
                "handshaked. THIS IS NOT EVIDENCE OF DATA CORRUPTION: no "
                "comparison of contents has been made. Refusing to scatter."
            )
        # Bounded-transient checksum (the weights_arena host-OOM family,
        # 2026-08-08): GDN payloads are GB-scale, a converted copy is 8x.
        have = uint8_checksum(data)
        if stored != have:
            raise KvReshardError(
                f"{LOG_PREFIX} payload checksum mismatch from peer {peer} "
                f"(stored {stored}, computed {have}); refusing to scatter. "
                f"Both values are representable for a {data.numel()}-byte "
                "payload, so the two ends framed it the same way and summed "
                "different bytes: the DATA differs (#802)."
            )
        return data

    def _write_tp_side(
        self,
        pools: GdnFlipPools,
        slots: List[int],
        src_stage: int,
        data: torch.Tensor,
    ) -> None:
        """Scatter a stage's payload (my shard) into my compact TP pool."""
        spec = pools.spec
        conv_w = int(pools.tp_conv.shape[3])
        conv_c = spec.conv_shard_channels(self._rank)
        conv_b = conv_c * conv_w * pools.tp_conv.element_size()
        h = spec.head_shards[self._rank]
        d, n = int(pools.tp_temporal.shape[3]), int(pools.tp_temporal.shape[4])
        temp_b = h * d * n * pools.tp_temporal.element_size()
        off = 0
        for s in slots:
            for gid in self._stage_ids[src_stage]:
                li = pools.tp_index[gid]
                chunk = data[off : off + conv_b]
                pools.tp_conv[li, s] = chunk.view(pools.tp_conv.dtype).view(
                    conv_c, conv_w
                )
                off += conv_b
                chunk = data[off : off + temp_b]
                pools.tp_temporal[li, s] = chunk.view(pools.tp_temporal.dtype).view(
                    h, d, n
                )
                off += temp_b
        if off != data.numel():
            raise KvReshardError(
                f"{LOG_PREFIX} stage {src_stage} payload has trailing bytes"
            )

    def _write_pp_side(
        self,
        pools: GdnFlipPools,
        slots: List[int],
        src_rank: int,
        data: torch.Tensor,
    ) -> None:
        """Scatter shard rank's payload into my stage's FULL tensors."""
        spec = pools.spec
        conv_w = int(pools.pp_conv.shape[3])
        conv_c = spec.conv_shard_channels(src_rank)
        conv_b = conv_c * conv_w * pools.pp_conv.element_size()
        h = spec.head_shards[src_rank]
        d, n = int(pools.pp_temporal.shape[3]), int(pools.pp_temporal.shape[4])
        temp_b = h * d * n * pools.pp_temporal.element_size()
        off = 0
        for s in slots:
            for gid in self._stage_ids[self._rank]:
                li = pools.pp_index[gid]
                chunk = data[off : off + conv_b]
                conv_scatter(
                    pools.pp_conv[li, s],
                    chunk.view(pools.pp_conv.dtype).view(conv_c, conv_w),
                    spec,
                    src_rank,
                )
                off += conv_b
                chunk = data[off : off + temp_b]
                temporal_scatter(
                    pools.pp_temporal[li, s],
                    chunk.view(pools.pp_temporal.dtype).view(h, d, n),
                    spec,
                    src_rank,
                )
                off += temp_b
        if off != data.numel():
            raise KvReshardError(
                f"{LOG_PREFIX} shard {src_rank} payload has trailing bytes"
            )

    # -- the move ------------------------------------------------------------

    def move(self, direction: str) -> dict:
        pools = self._pools_fn()  # re-validates preconditions EVERY flip
        slots = self._slot_list()
        spec = pools.spec
        conv_w = int(pools.pp_conv.shape[3])
        conv_e = pools.pp_conv.element_size()
        d, n = int(pools.pp_temporal.shape[3]), int(pools.pp_temporal.shape[4])
        temp_tail = d * n
        temp_e = pools.pp_temporal.element_size()

        if direction == PP_TO_TP:
            outgoing = {
                r: self._pack_pp_side(pools, slots, r)
                for r in range(self._n)
                if r != self._rank
            }
            incoming = {
                s: _pair_nbytes(
                    spec,
                    self._rank,
                    len(self._stage_ids[s]),
                    len(slots),
                    conv_w,
                    conv_e,
                    temp_tail,
                    temp_e,
                )
                + _CHECKSUM_BYTES
                for s in range(self._n)
                if s != self._rank
            }
        elif direction == TP_TO_PP:
            outgoing = {
                s: self._pack_tp_side(pools, slots, s)
                for s in range(self._n)
                if s != self._rank
            }
            incoming = {
                r: _pair_nbytes(
                    spec,
                    r,
                    len(self._stage_ids[self._rank]),
                    len(slots),
                    conv_w,
                    conv_e,
                    temp_tail,
                    temp_e,
                )
                + _CHECKSUM_BYTES
                for r in range(self._n)
                if r != self._rank
            }
        else:
            raise KvReshardError(f"unknown GDN flip direction {direction!r}")

        received = self._exchange(outgoing, incoming)
        verified = {
            peer: self._verify(received.get(peer), incoming[peer], peer)
            for peer in incoming
        }

        # Local leg (my stage <-> my shard), pools disjoint per direction.
        if direction == PP_TO_TP:
            local = self._pack_pp_side(pools, slots, self._rank)
            self._write_tp_side(pools, slots, self._rank, local[:-_CHECKSUM_BYTES])
            for peer, data in verified.items():
                self._write_tp_side(pools, slots, peer, data)
        else:
            local = self._pack_tp_side(pools, slots, self._rank)
            self._write_pp_side(pools, slots, self._rank, local[:-_CHECKSUM_BYTES])
            for peer, data in verified.items():
                self._write_pp_side(pools, slots, peer, data)

        stats = {
            "direction": direction,
            "slots": len(slots),
            "sent_bytes": sum(int(t.numel()) for t in outgoing.values()),
            "received_bytes": sum(incoming.values()),
        }
        self.last_stats = stats
        logger.info(
            "%s moved %d slot(s) %s: sent %.2f MiB, received %.2f MiB",
            LOG_PREFIX,
            len(slots),
            direction,
            stats["sent_bytes"] / 1048576.0,
            stats["received_bytes"] / 1048576.0,
        )
        return stats


def resident_mamba_slots(scheduler) -> "torch.Tensor":
    """The mamba slots the flip must move: THE RESIDENT SET's, not one slot's.

    #631 J.1, SECOND OCCURRENCE, and the reason this is a named function
    with a pin rather than a closure. This used to read
    ``scheduler.running_batch``. Under ``event_loop_pp`` that attribute is
    rebound to ``running_mbs[mb_id]`` at the top of every slot iteration,
    so it names ONE microbatch slot -- and the flip's hook fires at the end
    of an arbitrary one. The GDN leg therefore moved the conv/ssm state of
    whichever slot happened to be current and LEFT BEHIND the linear state
    of every request resident in another slot.

    THAT FAILURE IS SILENT, which is why it is called out rather than
    quietly corrected: since J.1 the KV move carries the request's
    attention cache correctly, so the request would decode on -- with its
    linear-attention state truncated at the flip point. #212's exact
    shape, and nothing raises. ``_live_reqs`` is the one authority for
    "who is resident" and the KV enumeration already uses it; so does this.
    """
    from sglang.srt.managers.phase_flip_runtime import _live_reqs

    vals = []
    for req in _live_reqs(scheduler):
        idx = getattr(req, "mamba_pool_idx", None)
        if idx is None:
            raise KvReshardError(
                f"{LOG_PREFIX} live request {getattr(req, 'rid', '?')} "
                f"has no mamba slot -- refusing to flip past unmoved "
                f"linear state"
            )
        vals.append(int(idx.item() if hasattr(idx, "item") else idx))
    return torch.tensor(sorted(set(vals)), dtype=torch.int64)


def flip_mamba_slots(scheduler) -> torch.Tensor:
    """Resident requests' mamba slots UNION the radix tree's checkpoints.

    THE ASYMMETRY THAT POISONED THE CACHE (#767 residual, measured
    2026-08-19). The KV leg enumerates "radix tree values UNION resident
    requests' rows" (build_flip_live_slots_fn); this leg enumerated only
    the resident set, so every mamba checkpoint the tree held -- cached
    prefix states, #745/#755 anchor donors -- kept its slot NUMBER across
    the cutover while its slot CONTENT was never translated into the new
    layout's pool. A later prefix hit COWed that untranslated slot into a
    fresh request's state (model_runner._maybe_execute_deferred_mamba_cow
    _and_clear), which is how salted probes looped after one sentence and
    one probe answered with a FOREIGN request's topic while the flip-off
    arm stayed clean. The host tier then archived the same stale bytes
    (PP-phase write-backs read the PP pool), which is why a device-side
    cache flush did not cure it.

    A tree cache that cannot report its mamba values is refused loudly:
    under a flip build a silent no-op here IS the corruption above, and
    every mamba-carrying tree in this repo exposes the API
    (MambaRadixCache.all_mamba_values_flatten, unified_radix_cache's
    override; the KV leg refuses the same way for all_values_flatten).
    """
    resident = resident_mamba_slots(scheduler)
    tree = getattr(scheduler, "tree_cache", None)
    values_fn = getattr(tree, "all_mamba_values_flatten", None)
    if values_fn is None:
        raise KvReshardError(
            f"{LOG_PREFIX} tree cache {type(tree).__name__} does not expose "
            f"all_mamba_values_flatten -- refusing to flip past mamba "
            f"checkpoints that could not be enumerated (a tree-held slot "
            f"left behind is served to a later prefix hit as garbage)"
        )
    tree_vals = values_fn()
    if tree_vals is None or tree_vals.numel() == 0:
        return resident
    tree_vals = tree_vals.detach().to("cpu", torch.int64).reshape(-1)
    return torch.unique(torch.cat([resident, tree_vals]))


def install_phase_aware_mamba_state_pool(scheduler) -> None:
    """Point the tree cache's mamba STATE-byte operations at the pool of
    the phase that is actually computing (#767 residual).

    Slot bookkeeping stays single-authority on the primary pool; only the
    byte copies (checkpoint donation copy, int8 store) follow the active
    stack. ``phase_flip_active_stack`` is unset until the first cutover,
    during which the primary stack computes -- the fallback is correct by
    construction. See mem_cache/mamba_state_pool.py for the read side.
    """
    primary_pool = scheduler.req_to_token_pool.mamba_pool
    tp_pool = (
        scheduler.phase_flip_stacks.tp_worker.model_runner.req_to_token_pool.mamba_pool
    )

    def _active_mamba_pool():
        from sglang.srt.managers.phase_flip_runtime import PHASE_TP

        if getattr(scheduler, "phase_flip_active_stack", None) == PHASE_TP:
            return tp_pool
        return primary_pool

    scheduler.tree_cache.phase_active_mamba_pool = _active_mamba_pool


def build_gdn_flip_mover(scheduler) -> Callable[[str], None]:
    """Production pre_cutover_fn: preconditions re-validated per flip,
    slots from the replicated running batch, exchange over the flip
    device group (a second closure of the KV runtime's channel)."""
    from sglang.srt.distributed.parallel_state import (
        get_phase_flip_group,
        get_world_group,
    )
    from sglang.srt.managers.kv_reshard import _dist_exchange

    stacks = scheduler.phase_flip_stacks
    world = get_world_group()
    n, rank = world.world_size, world.rank_in_group
    pp_req_pool = scheduler.tp_worker.model_runner.req_to_token_pool
    tp_req_pool = stacks.tp_worker.model_runner.req_to_token_pool
    tp_model_config = stacks.tp_worker.model_config
    linear_ids = sorted(pp_pool_ids_union(tp_req_pool))
    stage_ids = derive_pp_linear_layer_map(
        linear_ids,
        int(tp_model_config.num_hidden_layers),
        int(scheduler.server_args.pp_size),
    )

    def _pools() -> GdnFlipPools:
        return gdn_flip_preconditions(
            pp_req_pool,
            tp_req_pool,
            n,
            rank,
            # The hybrid HF text config: linear_num_*_heads plus the
            # model-stashed gdn_tp_units -- the mover replicates the
            # model's own head split (uneven-TP-head-geometry rule).
            gdn_geometry=tp_model_config.hf_text_config,
        )

    def _slots() -> torch.Tensor:
        return flip_mamba_slots(scheduler)

    install_phase_aware_mamba_state_pool(scheduler)

    flip_tp = get_phase_flip_group("tp")
    device = pp_req_pool.mamba_pool.mamba_cache.temporal.device
    mover = GdnFlipMover(
        n_ranks=n,
        rank=rank,
        stage_layer_ids=stage_ids,
        pools_fn=_pools,
        slots_fn=_slots,
        exchange=_dist_exchange(flip_tp.device_group, device),
    )
    scheduler.phase_flip_gdn_mover = mover

    def _leg(direction: str) -> None:
        mover.move(direction)

    return _leg


def pp_pool_ids_union(tp_req_pool) -> List[int]:
    """ALL linear layer ids, read from the TP req pool's mamba_map (the
    TP stack holds every linear layer; the PP pool holds only its
    stage)."""
    return [int(gid) for gid in tp_req_pool.mamba_map]
