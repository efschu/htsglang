# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Group accessors, LSE-merge and all-gather collectives for decode CP (DCP).

The two LSE-merge variants kept separate (bodies are backend-forced, see
PR #25090 vs #14194):
  - cp_lse_ag_out_rs_mha: torch / natural-log logsumexp / all-reduce + head slice
  - cp_lse_ag_out_rs_mla: Triton correction / reduce-scatter. The correction
    kernel's log base follows the attention backend that produced the LSE
    (base-2 for FlashInfer MLA, natural log for FlashMLA -- upstream #33064);
    it is not a property of this collective.
"""

import warnings
from typing import Optional

import logging

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.distributed.utils import get_tp_partition_ratios, weightless_kv_active
from sglang.srt.layers.dcp.collective_guard import guard_dcp_step
from sglang.srt.layers.dcp.kernels import CPTritonContext, correct_attn_out
from sglang.srt.runtime_context import get_parallel

#: Attention backends whose softmax LSE is a NATURAL log. Everything else in
#: this tree emits base-2 (FlashInfer MLA and the Triton/aiter MLA paths), so
#: the set is small and stated positively -- a backend that is not named here
#: keeps the pre-#426 base-2 correction. Upstream #33064.
NATURAL_LOG_LSE_ATTENTION_BACKENDS = frozenset({"flashmla"})


def lse_is_base_e(attention_backend: Optional[str]) -> bool:
    """True when ``attention_backend`` emits natural-log LSE.

    One named predicate instead of a string comparison at the call site: the
    log base is a property of the producing backend, and the DCP reduction is
    only one of its consumers.
    """
    return attention_backend in NATURAL_LOG_LSE_ATTENTION_BACKENDS


def _reject_uneven_tp_mla(fn: str, detail: str) -> None:
    """Fail fast: the MLA DCP collectives have NO uneven-TP variant.

    ``cp_lse_ag_out_rs_mla`` and ``all_gather_q_for_mla_decode`` both assume an
    EQUAL per-rank head count. Under a non-uniform --rank-tp-ratio plan they do
    not error on their own -- they either compute a silently wrong result or
    deadlock, because torch's reduce_scatter/all_gather require every rank to
    agree on the shape. That is the worst possible failure mode, so reject it
    here instead.

    Contrast the MHA path, which HAS an uneven variant and encodes the design
    decision in its name: ``cp_lse_ag_out_AR_mha_uneven`` uses all-**r**educe
    plus a head-bounds slice, precisely because ``_RS_`` (reduce-**s**catter)
    cannot express an uneven split.

    RANK-UNIFORMITY (this is the point of the guard, more than the message):
    the predicate is ``get_tp_partition_ratios()``, process-global state
    installed once per scheduler process from ``server_args.rank_tp_ratio``
    before the model is built -- a deterministic function of the CLI args, so
    every rank computes the SAME answer. Combined with SPMD (all ranks reach
    the same call site in the same order) the raise happens on every rank or on
    none. A guard that fired on a subset of ranks would itself be a hang
    source, which is exactly what this is meant to prevent.
    """
    ratios = get_tp_partition_ratios()
    if not ratios or len(set(ratios)) == 1:
        return
    raise NotImplementedError(
        f"{fn} does not support uneven tensor parallelism "
        f"(--rank-tp-ratio={list(ratios)}). {detail} There is no uneven-MLA "
        "variant: the uneven-MHA combine (cp_lse_ag_out_ar_mha_uneven) had to "
        "trade reduce-scatter for all-reduce-plus-slice to express unequal "
        "head shards, and that rework was never done for MLA. Running anyway "
        "would silently produce wrong output or deadlock, so it is rejected "
        "here. Use an even --rank-tp-ratio (or omit it) for MLA models, or an "
        "MHA model for uneven TP."
    )


def _warn_deprecated_dcp_accessor(name: str, replacement: str) -> None:
    warnings.warn(
        f"{name} is deprecated; use {replacement} instead.",
        DeprecationWarning,
        stacklevel=2,
    )


def dcp_enabled() -> bool:
    """Deprecated: use ``get_parallel().dcp_enabled``."""
    _warn_deprecated_dcp_accessor("dcp_enabled()", "get_parallel().dcp_enabled")
    return get_parallel().dcp_enabled


def get_attention_dcp_world_size() -> int:
    """Deprecated: use ``get_parallel().attn_dcp_size``."""
    _warn_deprecated_dcp_accessor(
        "get_attention_dcp_world_size()", "get_parallel().attn_dcp_size"
    )
    return get_parallel().attn_dcp_size


def get_attention_dcp_rank() -> int:
    """Deprecated: use ``get_parallel().attn_dcp_rank``."""
    _warn_deprecated_dcp_accessor(
        "get_attention_dcp_rank()", "get_parallel().attn_dcp_rank"
    )
    return get_parallel().attn_dcp_rank


def _ag_lse(cp_attn_lse: torch.Tensor, cp_group: GroupCoordinator) -> torch.Tensor:
    """All-gather each rank's LSE into a ``[world_size, *lse.shape]`` stack.

    Shared prologue of both ``cp_lse_ag_out_rs_{mha,mla}``. Callers do their own
    pre-processing (``contiguous()`` for MHA, fp32 cast for MLA) before calling.
    """
    return cp_group.all_gather(cp_attn_lse, dim=0).view(
        (cp_group.world_size,) + cp_attn_lse.shape
    )


def cp_lse_ag_out_rs_mha(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    return_lse: bool = False,
):
    """Merge DCP partial attention outputs using natural-log LSE (PR #25090)."""
    if cp_group.world_size == 1:
        return (cp_attn_out, cp_attn_lse) if return_lse else cp_attn_out

    cp_attn_lse = cp_attn_lse.contiguous()
    lses = _ag_lse(cp_attn_lse, cp_group)
    global_lse = torch.logsumexp(lses, dim=0)
    scale = torch.exp(cp_attn_lse - global_lse).unsqueeze(-1)
    scale = torch.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)

    out = torch.nan_to_num(cp_attn_out, nan=0.0, posinf=0.0, neginf=0.0) * scale
    out = cp_group.all_reduce(out)

    cp_num_heads = global_lse.shape[1] // cp_group.world_size
    cp_rank = cp_group.rank_in_group
    head_start = cp_num_heads * cp_rank
    head_end = cp_num_heads * (cp_rank + 1)
    out = out[:, head_start:head_end, :].contiguous()
    if return_lse:
        return out, global_lse[:, head_start:head_end].contiguous()
    return out


def cp_all_gather_heads_uneven(
    x: torch.Tensor,
    cp_group: GroupCoordinator,
    head_counts: list,
) -> torch.Tensor:
    """All-gather along the head dim (dim=1) when per-rank head counts differ
    (uneven TP / --rank-tp-ratio).

    torch's all_gather requires EQUAL shapes on every rank, so an uneven
    ``[12,6,6]`` q-head split would deadlock the collective (ranks disagree on
    the gathered size). The sync trick (ported from the vLLM uneven-DCP fork,
    ``ops/common.cp_all_gather_heads_uneven``): pad every rank's head dim to
    ``max(head_counts)``, all-gather the equal-shaped padded tensor, then slice
    out each rank's TRUE head count and re-concatenate in rank order (= global
    head order, since head shards are contiguous prefix-sum slices). Shapes are
    static per rank, so this is cuda-graph friendly; the pad garbage is sliced
    away before any compute.

    x: [ ..., local_heads, D ] with local_heads == head_counts[this_rank].
    Returns [ ..., sum(head_counts), D ].
    """
    if weightless_kv_active():
        # Anti-hang guard: the Q/K/V head all-gather is a per-attention-layer
        # DCP step. Verify the head rank and the weightless workers agree on the
        # step's identity before the real (uneven-shape) collective runs.
        guard_dcp_step(f"ag_heads:{sum(head_counts)}", cp_group)
    counts = list(head_counts)
    local_heads = x.shape[1]
    rank = cp_group.rank_in_group
    assert counts[rank] == local_heads, (
        f"cp_all_gather_heads_uneven: head_counts[{rank}]={counts[rank]} != "
        f"local head dim {local_heads}"
    )
    max_heads = max(counts)
    if all(c == max_heads for c in counts):
        # Equal split (e.g. even TP DCP, #107 Variant A): the padded path is a
        # no-op; use the plain collective. all_gather_into_tensor requires a
        # CONTIGUOUS input, but the k/v tensor handed in from the attention
        # forward is typically a non-contiguous head-sliced view -- the uneven
        # branch below hides this because it copies into a fresh padded buffer.
        # Force contiguity here so the equal-split fast path matches.
        return cp_group.all_gather(x.contiguous(), dim=1)
    pad_shape = list(x.shape)
    pad_shape[1] = max_heads
    padded = x.new_zeros(pad_shape)
    padded[:, :local_heads].copy_(x)
    gathered = cp_group.all_gather(padded, dim=1)
    parts = [
        gathered[:, r * max_heads : r * max_heads + counts[r]]
        for r in range(cp_group.world_size)
    ]
    return torch.cat(parts, dim=1).contiguous()


_KVQ_FUSE = {"on": None, "max_rows": None}


def dcp_fuse_max_rows() -> int:
    """SGLANG_DCP_FUSE_MAX_ROWS (default 256): both DCP fusions only for
    forwards of at most this many rows (decode / verify / short tails). The
    latency they save is per collective, so a wide prefill-with-prefix gains
    nothing -- and it would pay the fused buffers' extra transient at up to
    4096 rows on D TP0, the regime of the RC7 merge OOM. Row count is equal on
    every rank (the q/LSE gathers require it), so the gate is rank-uniform."""
    if _KVQ_FUSE["max_rows"] is None:
        import os

        try:
            _KVQ_FUSE["max_rows"] = max(0, int(os.environ.get("SGLANG_DCP_FUSE_MAX_ROWS", "256")))
        except ValueError:
            _KVQ_FUSE["max_rows"] = 256
    return int(_KVQ_FUSE["max_rows"])


def dcp_fuse_kvq_gather() -> bool:
    """SGLANG_DCP_FUSE_KVQ_GATHER=1 (default off): the per-layer KV write
    gather (A) and the q-head gather (B) of the uneven-DCP attention become ONE
    all-gather (``cp_all_gather_kvq_heads_uneven``).

    Measured motive (27B rc9meas i8h/i8r, 26.09., D TP3 bar1, bs1): the verify
    graph carries 193 collectives per round, 64 of them DCP (16 layers x
    {A 32 KiB, B 96 KiB, LSE 1.5 KiB all-gathers, a2a merge}); the dcp
    all-gathers alone wait 1.5 ms (TP1) / 2.3 ms (TP0) per round, ~31/48 us
    each, of which the bytes on TP1's x4 link are a minority. Fusing A and B
    keeps the bytes and drops one rendezvous per full-attention layer.

    Read once; must be set identically on every rank of the DCP group (the
    collective sequence changes from A,B,C,D to AB,C,D)."""
    if _KVQ_FUSE["on"] is None:
        import os

        v = str(os.environ.get("SGLANG_DCP_FUSE_KVQ_GATHER", "0")).strip().lower()
        _KVQ_FUSE["on"] = v in ("1", "true", "yes", "on")
        if _KVQ_FUSE["on"]:
            # the A/B needs proof that the arm bound (default off logs nothing)
            logging.getLogger(__name__).info(
                "DCP-FUSE kvq_gather=on (SGLANG_DCP_FUSE_KVQ_GATHER): KV write "
                "gather + q gather = one all-gather per full-attention layer"
            )
    return bool(_KVQ_FUSE["on"])


def cp_all_gather_kvq_heads_uneven(
    k: torch.Tensor,
    v: torch.Tensor,
    q: torch.Tensor,
    cp_group: GroupCoordinator,
    kv_head_counts: list,
    q_head_counts: list,
):
    """ONE uneven head all-gather for the KV write (k, v) AND the q heads.

    Equivalent, element for element, to the two gathers it replaces --
    ``cp_all_gather_heads_uneven(cat((k, v), 0), kv_counts)`` split at T, and
    ``cp_all_gather_heads_uneven(q, q_counts)`` -- because an all-gather is
    pure data movement: every rank contributes ``cat((k, v, q), dim=1)``
    ([T, 2*kv_r + q_r, D]), the padded gather puts the blocks in rank order,
    and each output is re-assembled from its per-rank slice of those blocks.
    No arithmetic touches a value, so the result is BIT-identical.

    Bytes per rank: max_r(2*kv_r + q_r) heads x T x D, never more than the
    two padded gathers' max_r(kv_r)*2 + max_r(q_r) (for [2,1,1]/[12,6,6]:
    16 = 4 + 12). One rendezvous instead of two.

    Returns ``(k_full, v_full, q_full)``: [T, sum(kv), D], [T, sum(kv), D],
    [T, sum(q), D], each contiguous.
    """
    T, kv_local, D = k.shape
    assert v.shape == k.shape, f"k {tuple(k.shape)} != v {tuple(v.shape)}"
    assert q.shape[0] == T and q.shape[2] == D, (
        f"q {tuple(q.shape)} does not share T/D with k {tuple(k.shape)}"
    )
    assert q.dtype == k.dtype == v.dtype, (q.dtype, k.dtype, v.dtype)
    kv = [int(c) for c in kv_head_counts]
    qc = [int(c) for c in q_head_counts]
    world = cp_group.world_size
    assert len(kv) == len(qc) == world
    fused_counts = [2 * a + b for a, b in zip(kv, qc)]
    x = torch.cat((k, v, q), dim=1)
    g = cp_all_gather_heads_uneven(x, cp_group, fused_counts)
    ks, vs, qs = [], [], []
    off = 0
    for r in range(world):
        ks.append(g[:, off : off + kv[r]])
        vs.append(g[:, off + kv[r] : off + 2 * kv[r]])
        qs.append(g[:, off + 2 * kv[r] : off + fused_counts[r]])
        off += fused_counts[r]
    return (
        torch.cat(ks, dim=1).contiguous(),
        torch.cat(vs, dim=1).contiguous(),
        torch.cat(qs, dim=1).contiguous(),
    )


def cp_local_head_bounds(cp_group: GroupCoordinator, head_counts: list) -> tuple:
    """(start, stop) of this rank's head slice in the gathered full head set
    under uneven DCP (head_counts = per-rank q-head partition, e.g. [12,6,6])."""
    rank = cp_group.rank_in_group
    start = sum(head_counts[:rank])
    return start, start + head_counts[rank]


logger = logging.getLogger(__name__)

_LSE_MERGE = {"dtype": None, "mode": None, "fused": None}


def _ng(stage: str, t: torch.Tensor, cp_group, allow_neg_inf: bool = False) -> None:
    """Task #49 (19.09.): SGLANG_NAN_GUARD=1 names the merge stage in which
    the numbers die (fn7y: layer 15, all ranks, the same 2608 rows). -inf
    is a legal LSE (a query without rows); anything else non-finite is not."""
    try:
        from sglang.srt.layers.nan_guard import nan_guard_on
    except Exception:  # noqa: BLE001
        return
    if not nan_guard_on() or t is None or t.numel() == 0:
        return
    try:
        if t.is_cuda and torch.cuda.is_current_stream_capturing():
            return
        tf = t.float()
        bad = ~torch.isfinite(tf)
        if allow_neg_inf:
            bad &= ~(tf == float("-inf"))
        n = int(bad.sum().item())
        if n == 0:
            return
        rows = torch.nonzero(bad.reshape(bad.shape[0], -1).any(dim=1) if t.dim() > 1 else bad).reshape(-1)
        logger.error(
            "[nan-guard] merge stage %s rank %d: %d non-finite of %d, shape %s, first rows %s (n_rows %d)",
            stage, int(getattr(cp_group, "rank_in_group", -1)), n, t.numel(), tuple(t.shape),
            rows[:8].tolist(), int(rows.numel()),
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("[nan-guard] merge check skipped: %s", exc)


def lse_merge_reduce_dtype() -> str:
    """SGLANG_DCP_LSE_MERGE_DTYPE: 'fp32' (default) or 'bf16' for the reduction
    of the scaled attention partials in the uneven LSE merge."""
    if _LSE_MERGE["dtype"] is None:
        import os

        v = str(os.environ.get("SGLANG_DCP_LSE_MERGE_DTYPE", "fp32")).strip().lower()
        _LSE_MERGE["dtype"] = "bf16" if v in ("bf16", "bfloat16", "half") else "fp32"
    return _LSE_MERGE["dtype"]


def lse_merge_mode() -> str:
    """SGLANG_DCP_LSE_MERGE: 'ar' (default; all-reduce of the full head range,
    then slice) or 'a2a' (reduce-scatter built from one all_to_all_single of
    per-rank head blocks plus a local sum -- each rank ships only the heads the
    others own)."""
    if _LSE_MERGE["mode"] is None:
        import os

        v = str(os.environ.get("SGLANG_DCP_LSE_MERGE", "ar")).strip().lower()
        _LSE_MERGE["mode"] = "a2a" if v in ("a2a", "rs", "reduce_scatter") else "ar"
    return _LSE_MERGE["mode"]


def lse_merge_fused() -> bool:
    """SGLANG_DCP_LSE_MERGE_FUSED=1 (default off; only with
    SGLANG_DCP_LSE_MERGE=a2a and the fp32 wire): the LSE all-gather of the a2a
    merge rides INSIDE the head all_to_all -- every rank ships the UNSCALED
    partials of the peer-owned heads plus their LSE (one extra fp32 column,
    padded to four so a head row stays 16-byte aligned), and the receiver does
    the logsumexp / scale / sum for its own heads. One rendezvous per merge
    instead of two (see ``_cp_lse_a2a_fused_body``).

    Read once; rank-uniform like SGLANG_DCP_LSE_MERGE (the collective sequence
    changes from C,D to CD)."""
    if _LSE_MERGE["fused"] is None:
        import os

        v = str(os.environ.get("SGLANG_DCP_LSE_MERGE_FUSED", "0")).strip().lower()
        _LSE_MERGE["fused"] = v in ("1", "true", "yes", "on")
        if _LSE_MERGE["fused"]:
            logger.info(
                "DCP-FUSE lse_merge=on (SGLANG_DCP_LSE_MERGE_FUSED): the LSE "
                "all-gather rides in the a2a (fp32 wire only; bf16 keeps two)"
            )
    return bool(_LSE_MERGE["fused"])


#: Columns the fused a2a appends to each [head, token] row: the LSE in the
#: first, zeros after it. Four fp32 = 16 bytes, so a head row of D fp32 plus
#: this tail stays a multiple of 16 bytes (the bar1 a2a kernel's packet).
LSE_FUSED_TAIL = 4


def _cp_lse_a2a_fused_body(cp_attn_out, cp_attn_lse, cp_group, counts, return_lse):
    """The a2a merge with the LSE all-gather folded into the all_to_all.

    Same math as the two-collective body, moved to the receiver: the sender
    ships nan_to_num(o) (fp32) and its lse for the heads rank ``s`` owns; rank
    ``s`` stacks the W received LSEs of ITS heads, takes the same logsumexp over
    the rank axis, the same nan_to_num(exp(lse - global)) scale, multiplies in
    fp32 exactly as the sender did, and sums over the rank axis in the same
    rank order. Every value that enters the arithmetic is transferred exactly
    (fp32 wire), so on one device class the result is bit-identical to the
    two-collective body; across device classes it can differ only where the
    same elementwise exp/logsumexp kernel rounds differently per arch."""
    world = cp_group.world_size
    tokens, _h, dim = cp_attn_out.shape
    rank = cp_group.rank_in_group
    mine = counts[rank]
    tail = LSE_FUSED_TAIL
    o32 = torch.nan_to_num(cp_attn_out, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)
    send = o32.new_zeros((_h, tokens, dim + tail))  # [H_total, tokens, D + tail]
    send[:, :, :dim].copy_(o32.transpose(0, 1))
    send[:, :, dim].copy_(cp_attn_lse.to(torch.float32).transpose(0, 1))
    recv = torch.empty((world * mine, tokens, dim + tail), dtype=torch.float32,
                       device=cp_attn_out.device)
    cp_group.all_to_all_single_v(
        recv, send, output_split_sizes=[mine] * world, input_split_sizes=counts
    )
    _ng("merge.a2a_send", send, cp_group, allow_neg_inf=True)
    _ng("merge.a2a_recv", recv, cp_group, allow_neg_inf=True)
    r4 = recv.view(world, mine, tokens, dim + tail)
    lses = r4[..., dim]  # [W, mine, tokens]
    global_lse = torch.logsumexp(lses, dim=0)  # [mine, tokens]
    scale = torch.exp(lses - global_lse).unsqueeze(-1)
    scale = torch.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)
    merged = (r4[..., :dim] * scale).sum(dim=0)  # [mine, tokens, D]
    _ng("merge.result", merged, cp_group)
    merged = merged.transpose(0, 1).contiguous()  # [tokens, H_local, D]
    if return_lse:
        return merged, global_lse.transpose(0, 1).contiguous()
    return merged


def cp_lse_ag_out_a2a_mha_uneven(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    head_counts: list,
    return_lse: bool = False,
):
    """Uneven-DCP MHA combine as a REDUCE-SCATTER: the same LSE math as
    ``cp_lse_ag_out_ar_mha_uneven``, but instead of all-reducing the whole
    [tokens, H_total, D] and slicing, every rank sends rank ``s`` only the
    scaled partial of ``s``'s heads (one uneven ``all_to_all_single_v`` --
    bar1 serves unequal blocks natively, no padding) and sums the blocks it
    receives. Bytes per rank: (W-1)/W x full instead of the ring's
    2(W-1)/W x full -- half the wire -- and half of that again with
    SGLANG_DCP_LSE_MERGE_DTYPE=bf16.

    cp_attn_out: [ tokens, H_total, D ]; cp_attn_lse: [ tokens, H_total ].
    Returns this rank's [ tokens, H_local, D ] in fp32 (and its global lse).
    """
    if weightless_kv_active():
        guard_dcp_step("lse_merge_a2a", cp_group)
    if cp_group.world_size == 1:
        return (cp_attn_out, cp_attn_lse) if return_lse else cp_attn_out
    world = cp_group.world_size
    counts = [int(c) for c in head_counts]
    assert len(counts) == world and sum(counts) == cp_attn_out.shape[1]
    cp_attn_lse = cp_attn_lse.contiguous()
    _ng("merge.local_out", cp_attn_out, cp_group)
    _ng("merge.local_lse", cp_attn_lse, cp_group, allow_neg_inf=True)
    if (
        lse_merge_fused()
        and lse_merge_reduce_dtype() == "fp32"
        and cp_attn_out.shape[0] <= dcp_fuse_max_rows()
    ):
        # SGLANG_DCP_LSE_MERGE_FUSED: one collective instead of two. The bf16
        # wire keeps the two-collective body -- the LSE must travel in fp32.
        return _cp_lse_a2a_fused_body(
            cp_attn_out, cp_attn_lse, cp_group, counts, return_lse
        )
    lses = _ag_lse(cp_attn_lse, cp_group)
    _ng("merge.gathered_lse", lses, cp_group, allow_neg_inf=True)
    global_lse = torch.logsumexp(lses, dim=0)
    scale = torch.exp(cp_attn_lse - global_lse).unsqueeze(-1)
    scale = torch.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)
    out = torch.nan_to_num(cp_attn_out, nan=0.0, posinf=0.0, neginf=0.0) * scale
    wire_dtype = torch.bfloat16 if lse_merge_reduce_dtype() == "bf16" else torch.float32
    tokens, _h, dim = out.shape
    rank = cp_group.rank_in_group
    mine = counts[rank]
    # head-major so that axis 0 (the only axis the a2a splits) is the head axis
    send = out.transpose(0, 1).contiguous().to(wire_dtype)  # [H_total, tokens, D]
    recv = torch.empty((world * mine, tokens, dim), dtype=wire_dtype, device=out.device)
    cp_group.all_to_all_single_v(
        recv, send, output_split_sizes=[mine] * world, input_split_sizes=counts
    )
    _ng("merge.a2a_send", send, cp_group)
    _ng("merge.a2a_recv", recv, cp_group)
    merged = recv.view(world, mine, tokens, dim).to(torch.float32).sum(dim=0)
    _ng("merge.result", merged, cp_group)
    merged = merged.transpose(0, 1).contiguous()  # [tokens, H_local, D]
    if return_lse:
        start, stop = cp_local_head_bounds(cp_group, counts)
        return merged, global_lse[:, start:stop].contiguous()
    return merged


def cp_lse_ag_out_ar_mha_uneven(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    head_counts: list,
    return_lse: bool = False,
):
    """Uneven-DCP MHA combine: like ``cp_lse_ag_out_rs_mha`` but per-rank head
    shards differ (--rank-tp-ratio), so the equal ``H // world_size`` slice is
    replaced by this rank's prefix-sum head bounds (from ``head_counts``).

    cp_attn_out: [ tokens, H_total, D ]  (H_total = sum(head_counts))
    cp_attn_lse: [ tokens, H_total ]
    """
    if weightless_kv_active():
        # Anti-hang guard: the LSE merge is the per-attention-layer combine
        # step. Verify sequence agreement before the all-gather/all-reduce.
        guard_dcp_step("lse_merge", cp_group)
    if cp_group.world_size == 1:
        return (cp_attn_out, cp_attn_lse) if return_lse else cp_attn_out

    cp_attn_lse = cp_attn_lse.contiguous()
    lses = _ag_lse(cp_attn_lse, cp_group)
    global_lse = torch.logsumexp(lses, dim=0)
    scale = torch.exp(cp_attn_lse - global_lse).unsqueeze(-1)
    scale = torch.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)

    out = torch.nan_to_num(cp_attn_out, nan=0.0, posinf=0.0, neginf=0.0) * scale
    # fn7g (19.09.): this all-reduce is 12 x 100 MB fp32 per 8k prefill chunk
    # (0.67 s of 7.66 s) and reduce_scatterv is not served by barlink. With
    # SGLANG_DCP_LSE_MERGE_DTYPE=bf16 the scaled partials are reduced in the
    # query dtype (half the bytes); the scale itself stays fp32.
    if lse_merge_reduce_dtype() == "bf16" and out.dtype == torch.float32:
        # the QSA rows kernel hands its partials over in fp32 (fn7j): the wire
        # dtype has to be named, not inherited from the input
        out = cp_group.all_reduce(out.to(torch.bfloat16)).to(torch.float32)
    else:
        out = cp_group.all_reduce(out)

    start, stop = cp_local_head_bounds(cp_group, head_counts)
    out = out[:, start:stop, :].contiguous()
    if return_lse:
        return out, global_lse[:, start:stop].contiguous()
    return out


# ---------------------------------------------------------------------------
# 27B RC7 (25.09.): THE UNEVEN LSE MERGE IN TOKEN BLOCKS.
#
# Measured, boot weg2rc7_186c022f80 D TP0 (5090), 09:22:19: `Tried to
# allocate 144.00 MiB` in cp_lse_ag_out_a2a_mha_uneven's `recv` =
# (W*H_local, T, D) fp32 = (3*12, 4096, 256) * 4 B -- T = 4096, the chunk
# width (`#969 EXTENT ... (4096, 8192, 4096, 4096)`: prefix 4096, extend 4096).
# The width did NOT grow past the chunk; what the X ceiling (and, equally, a
# multi-turn follow-up with <= X uncached tokens over a D-radix prefix) made
# possible is a PREFIX-BEARING forward at full chunk width. Every 27B D boot
# with X <= 4096 topped out at 229-231 prefix-bearing tokens, so the one-shot
# merge's working set at 4096 (~483 MiB on TP0 in a2a/fp32: out, send, recv,
# merged, merged^T) was never exercised and sits in no budget.
#
# THE BOUND. Blocking along the token axis is exact: every step of both merge
# bodies is per (token, head) -- the LSE all-gather, the logsumexp over the
# RANK axis, the scale, the head all_to_all / all_reduce and the local sum --
# so a block is today's call on a token slice. The block width is derived, not
# pinned: the working set of one block is at most ONE full-head prefix partial
# (the paged read's own output, [budget_tokens, H_total, D] in the attention
# dtype) at the budgeted width. The merge then never holds more than the
# partial it redistributes, whatever the forward width.
#
# RANK-UNIFORM BY CONSTRUCTION: the width is a function of budget_tokens,
# sum/max/len of head_counts, head_dim, the attention dtype and the merge mode
# (all replicated), never of this rank's own head count; the split of T is a
# function of T (the gathered-q token count, equal on every rank -- the q/LSE
# all-gathers require it) and the width. Every rank therefore runs the same
# number of blocks and issues the same collectives in the same order.
# ---------------------------------------------------------------------------

#: The merge's per-token working set, per mode. Every full-width tensor the
#: one-shot body allocates is counted (the SUM, an upper bound on the peak --
#: independent of when the allocator frees what, and of whether the transport's
#: all_reduce is in-place), so the per-block bound holds on any transport.
def lse_merge_bytes_per_token(
    mode: str,
    wire_bytes: int,
    total_heads: int,
    max_local_heads: int,
    world: int,
    head_dim: int,
    out_itemsize: int,
) -> int:
    H, m, W, D = int(total_heads), int(max_local_heads), int(world), int(head_dim)
    f32 = 4
    # prologue, both modes (fp32 [H] per token each): lse.contiguous(), the
    # gathered [W, H] lses, logsumexp's own temporaries (amax, |amax|, the
    # W-wide sub and exp, the sum), then (lse - global_lse), exp and
    # nan_to_num(scale) -- 3W + 8 in all; then nan_to_num(o) in the attention
    # dtype and `* scale` in fp32 (`out`).
    b = f32 * H * (3 * W + 8) + H * D * int(out_itemsize) + f32 * H * D
    if mode == "a2a":
        b += f32 * H * D  # out.transpose(0, 1).contiguous()
        if wire_bytes != f32:
            b += wire_bytes * H * D  # .to(wire)
        b += wire_bytes * W * m * D  # recv
        if wire_bytes != f32:
            b += f32 * W * m * D  # recv.to(fp32)
        b += f32 * m * D  # .sum(0)
        b += f32 * m * D  # merged.transpose(0, 1).contiguous()
    else:
        if wire_bytes != f32:
            b += wire_bytes * H * D  # out.to(bf16)
        b += wire_bytes * H * D  # all_reduce, out-of-place (barlink)
        if wire_bytes != f32:
            b += f32 * H * D  # .to(fp32)
        b += f32 * m * D  # out[:, start:stop].contiguous()
    b += f32 * m  # global_lse[:, start:stop].contiguous() (return_lse)
    return int(b)


def lse_merge_effective_mode() -> str:
    """The merge the flashinfer head sites actually run: a2a only when asked
    for AND not weightless (the weightless workers' sites stay on the
    all_reduce, so a weightless boot does too -- see _dcp_uneven_merge)."""
    return "a2a" if (lse_merge_mode() == "a2a" and not weightless_kv_active()) else "ar"


def lse_merge_block_tokens(
    budget_tokens: int,
    head_counts: list,
    head_dim: int,
    out_itemsize: int,
    mode: Optional[str] = None,
) -> int:
    """Derived token-block width of the uneven LSE merge (0 = one block).

    The largest width whose per-block working set
    (:func:`lse_merge_bytes_per_token`) fits in ONE full-head prefix partial at
    the budgeted width, ``budget_tokens * H_total * head_dim * out_itemsize``
    bytes. ``max(head_counts)`` stands in for the local head count so the
    answer is the same on every rank (the largest shard binds)."""
    counts = [int(c) for c in head_counts]
    W = len(counts)
    # head_dim <= 0 = no head geometry known: the bound (one full-head partial)
    # is then 0 bytes and the old max(1, 0 // per_tok) returned width 1 --
    # 4096 one-row blocks per layer at a 4096 chunk. No geometry -> no derived
    # blocking (0 = one block); the resolver's DCP-MERGE-BLOCK line says so.
    if W <= 1 or int(budget_tokens) <= 0 or sum(counts) <= 0 or int(head_dim) <= 0:
        return 0
    mode = mode or lse_merge_effective_mode()
    wire = 2 if lse_merge_reduce_dtype() == "bf16" else 4
    per_tok = lse_merge_bytes_per_token(
        mode, wire, sum(counts), max(counts), W, head_dim, out_itemsize
    )
    if per_tok <= 0:
        return 0
    ref = int(budget_tokens) * sum(counts) * int(head_dim) * int(out_itemsize)
    return max(1, ref // per_tok)


def lse_merge_token_spans(tokens: int, block_tokens: int) -> list:
    """[(start, stop), ...] covering [0, tokens): ONE span (today's call) when
    blocking is off or ``tokens <= block_tokens``, else ceil(tokens/block)
    near-equal spans, each <= block_tokens (no ragged straggler block)."""
    T, b = int(tokens), int(block_tokens or 0)
    if b <= 0 or T <= b:
        return [(0, T)]
    n = -(-T // b)
    base, extra = divmod(T, n)
    spans, s = [], 0
    for i in range(n):
        e = s + base + (1 if i < extra else 0)
        spans.append((s, e))
        s = e
    return spans


def lse_merge_is_blocked(tokens: int, block_tokens: int, world_size: int) -> bool:
    """True when the merge of ``tokens`` rows runs in more than one block.
    world_size 1 never blocks: the merge is the identity there."""
    return int(world_size) > 1 and len(lse_merge_token_spans(tokens, block_tokens)) > 1


_BLOCKED_MERGE_N = [0]
_BLOCKED_MERGE_LAST_T = [-1]


def cp_lse_merge_token_blocks(
    merge_fn,
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    head_counts: list,
    return_lse: bool = False,
    block_tokens: int = 0,
):
    """``merge_fn`` (cp_lse_ag_out_{ar,a2a}_mha_uneven) over token blocks of
    at most ``block_tokens`` rows, results written into one [T, H_local, D]
    fp32 output (and [T, H_local] lse). One block -- blocking off, T <= width,
    or a single-rank group -- is EXACTLY today's call and returns its result
    object unchanged. Each block is also exactly today's call, on a token
    slice, so the collectives per block (LSE all-gather, a2a/all-reduce) and
    the weightless guard step are those of one merge; see the module note on
    rank uniformity."""
    T = int(cp_attn_out.shape[0])
    world = int(cp_group.world_size)
    if not lse_merge_is_blocked(T, block_tokens, world):
        return merge_fn(
            cp_attn_out, cp_attn_lse, cp_group, head_counts, return_lse=return_lse
        )
    spans = lse_merge_token_spans(T, block_tokens)
    out = lse_out = None
    for s, e in spans:
        res = merge_fn(
            cp_attn_out[s:e], cp_attn_lse[s:e], cp_group, head_counts,
            return_lse=return_lse,
        )
        o_b, l_b = res if return_lse else (res, None)
        if out is None:
            out = o_b.new_empty((T,) + tuple(o_b.shape[1:]))
            if return_lse:
                lse_out = l_b.new_empty((T,) + tuple(l_b.shape[1:]))
        out[s:e].copy_(o_b)
        if return_lse:
            lse_out[s:e].copy_(l_b)
        del res, o_b, l_b
    _BLOCKED_MERGE_N[0] += 1
    n = _BLOCKED_MERGE_N[0]
    # one line per forward width in practice: the 16 full-attention layers of
    # a forward merge the same T back to back, so "T changed" fires on the
    # forward's first layer; plus calls 1-3 and every 64th.
    if n <= 3 or T != _BLOCKED_MERGE_LAST_T[0] or n % 64 == 0:
        _BLOCKED_MERGE_LAST_T[0] = T
        logger.info(
            "DCP-MERGE-BLOCKED n=%d fn=%s T=%d blocks=%d block_tokens=%d "
            "max_block=%d world=%d heads=%s return_lse=%s (each block is one "
            "whole merge: LSE all-gather + head exchange; logged on calls 1-3, "
            "on every change of T and every 64th blocked call)",
            n, getattr(merge_fn, "__name__", "?"), T, len(spans),
            int(block_tokens), max(e - s for s, e in spans), world,
            list(head_counts), bool(return_lse),
        )
    return (out, lse_out) if return_lse else out


def cp_lse_ag_out_rs_mla(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: Optional[CPTritonContext] = None,
    is_lse_base_on_e: bool = False,
):
    """Merge DCP partial attention outputs via Triton correction (PR #14194).

    cp_attn_out: [ B, H, D ]
    cp_attn_lse: [ B, H ]
    is_lse_base_on_e: True when the attention backend emits natural-log LSE
        (FlashMLA), False for base-2 LSE (FlashInfer MLA). The caller decides,
        because only it knows which backend produced ``cp_attn_lse``; see the
        LOG BASE note on ``correct_attn_out`` (upstream #33064).
    """
    if cp_group.world_size == 1:
        return cp_attn_out

    # reduce_scatter_along_dim(dim=0) splits dim 0 into world_size EQUAL
    # chunks -- it cannot express an uneven head split. See _reject_uneven_tp_mla.
    _reject_uneven_tp_mla(
        "cp_lse_ag_out_rs_mla",
        "Its reduce_scatter_along_dim(dim=0) splits the head dim into "
        "world_size EQUAL chunks.",
    )

    if ctx is None:
        ctx = CPTritonContext()

    with use_symmetric_memory(cp_group):
        # cp_attn_out is [B,H,D], we want to transpose it to [H,B,D] for the kernel, and then transpose back after correction.
        new_output = cp_attn_out.new_empty(
            cp_attn_out.transpose(0, 1).shape, dtype=torch.float32
        )
        cp_attn_lse = cp_attn_lse.to(torch.float32)
    lses = _ag_lse(cp_attn_lse, cp_group)
    out, _ = correct_attn_out(
        cp_attn_out,
        lses,
        cp_group.rank_in_group,
        ctx,
        new_output,
        is_lse_base_on_e=is_lse_base_on_e,
    )
    out = cp_group.reduce_scatter_along_dim(out, dim=0)
    return out.to(cp_attn_out.dtype)


def _all_gather_dcp_kv_cache(kv_a: torch.Tensor):
    parallel = get_parallel()
    dcp_world_size = parallel.dcp_size
    # not use symmetric_memory unless torch mem_pool updated, see https://github.com/pytorch/pytorch/issues/178138
    gathered_kv_a = kv_a.new_empty(
        (kv_a.shape[0] * dcp_world_size, *kv_a.shape[1:]),
    )
    parallel.dcp_group.all_gather_into_tensor(gathered_kv_a, kv_a)
    gathered_kv_a = (
        gathered_kv_a.reshape((dcp_world_size,) + kv_a.shape)
        .transpose(0, 1)
        .reshape(-1, *kv_a.shape[1:])
    )
    return gathered_kv_a


def all_gather_kv_cache_for_mha_chunk_extend(
    kv_a: torch.Tensor,
    k_pe: torch.Tensor,
    prefix_kv_lens_cpu: torch.Tensor,
    prefix_starts_cpu: torch.Tensor = None,
):
    if get_parallel().dcp_enabled:
        kv_a = kv_a.unsqueeze(1)
        gathered_kv = all_gather_kv_cache_for_dcp(
            kv_a,
            k_pe,
            prefix_kv_lens_cpu,
            prefix_starts_cpu,
        )
        kv_a, k_pe = gathered_kv.split([kv_a.shape[-1], k_pe.shape[-1]], dim=-1)
        kv_a = kv_a.squeeze(1)
    return kv_a.contiguous(), k_pe.contiguous()


def all_gather_kv_cache_for_mha_extend(
    token_to_kv_pool,
    attn_mqa,
    dcp_local_prefix_kv_indices,
    seq_lens,
    extend_prefix_lens,
    extend_prefix_lens_cpu: list[int],
    extend_seq_lens,
    kv_a: torch.Tensor,
    k_pe: torch.Tensor,
):
    prefix_kv_a, prefix_k_pe = token_to_kv_pool.get_mla_kv_buffer(
        attn_mqa, dcp_local_prefix_kv_indices
    )
    extend_prefix_lens_cpu = torch.tensor(extend_prefix_lens_cpu)
    gathered_kv_cache = all_gather_kv_cache_for_dcp(
        prefix_kv_a,
        prefix_k_pe,
        extend_prefix_lens_cpu,
    )
    prefix_kv_a, prefix_k_pe = gathered_kv_cache.split(
        [kv_a.shape[-1], k_pe.shape[-1]], dim=-1
    )
    prefix_kv_a = prefix_kv_a.squeeze(1)
    # re-organize kv with query orders
    prefix_lens_cu = torch.zeros(
        len(seq_lens) + 1,
        dtype=torch.int32,
        device=kv_a.device,
    )
    extend_lens_cu = torch.zeros_like(prefix_lens_cu)
    prefix_lens_cu[1:] = torch.cumsum(extend_prefix_lens, dim=0)
    extend_lens_cu[1:] = torch.cumsum(extend_seq_lens, dim=0)
    kv_a_tuple = ()
    k_pe_tuple = ()
    for i in range(len(seq_lens)):
        kv_a_tuple += (
            prefix_kv_a[prefix_lens_cu[i] : prefix_lens_cu[i + 1]],
            kv_a[extend_lens_cu[i] : extend_lens_cu[i + 1]],
        )
        k_pe_tuple += (
            prefix_k_pe[prefix_lens_cu[i] : prefix_lens_cu[i + 1]],
            k_pe[extend_lens_cu[i] : extend_lens_cu[i + 1]],
        )
    kv_a = torch.cat(kv_a_tuple, dim=0)
    k_pe = torch.cat(k_pe_tuple, dim=0)
    return kv_a.contiguous(), k_pe.contiguous()


def all_gather_q_for_mla_decode(
    q_nope_out: torch.Tensor,
    q_pe: torch.Tensor,
):
    group = get_parallel().dcp_group
    # The all_gather below is over dim 0 of a [H, B, L] tensor, where H is THIS
    # rank's head count -- unequal across ranks under uneven TP, which torch's
    # all_gather cannot express. See _reject_uneven_tp_mla.
    _reject_uneven_tp_mla(
        "all_gather_q_for_mla_decode",
        "It all-gathers a [H, B, L] tensor over dim 0, where H is this rank's "
        "head count and therefore differs between ranks.",
    )
    with use_symmetric_memory(group):
        # transpose q_pe and q_nope_out from [B, H, L] to [H, B, L]
        combined = torch.cat([q_pe.transpose(0, 1), q_nope_out.transpose(0, 1)], dim=-1)
    gathered = group.all_gather(combined, dim=0)
    d_pe = q_pe.size(-1)
    d_nope = q_nope_out.size(-1)
    q_pe, q_nope_out = gathered.split([d_pe, d_nope], dim=-1)
    q_pe = q_pe.transpose(0, 1)
    q_nope_out = q_nope_out.transpose(0, 1)
    return q_nope_out, q_pe


def all_gather_kv_cache_for_mla_extend(
    token_to_kv_pool,
    attn_mqa,
    extend_prefix_lens_cpu: list[int],
    dcp_local_prefix_kv_indices,
    dcp_extend_prefix_lens_sum,
    dcp_kv_buffer,
    kv_lora_rank,
    k_nope,
    k_pe,
):
    cache_k_nope, cache_k_rope = token_to_kv_pool.get_mla_kv_buffer(
        attn_mqa,
        dcp_local_prefix_kv_indices,
    )
    extend_prefix_lens_cpu = torch.tensor(extend_prefix_lens_cpu)
    # all gather kv cache into forward_batch.attn_dcp_metadata.dcp_kv_buffer
    gathered_kv = all_gather_kv_cache_for_dcp(
        cache_k_nope,
        cache_k_rope,
        extend_prefix_lens_cpu,
        prefix_starts_cpu=torch.zeros_like(extend_prefix_lens_cpu),
    )
    dcp_kv_buffer[:dcp_extend_prefix_lens_sum] = gathered_kv

    # copy local kv cache into forward_batch.attn_dcp_metadata.dcp_kv_buffer
    dcp_kv_buffer[
        dcp_extend_prefix_lens_sum:,
        ...,
        :kv_lora_rank,
    ] = k_nope
    dcp_kv_buffer[
        dcp_extend_prefix_lens_sum:,
        ...,
        kv_lora_rank:,
    ] = k_pe


# all gather kv cache and re-org to query orders
def all_gather_kv_cache_for_dcp(
    prefix_kv_a: torch.Tensor,
    prefix_k_pe: torch.Tensor,
    prefix_kv_lens_cpu: torch.Tensor,
    prefix_starts_cpu: torch.Tensor = None,
):
    """
    prefix_kv_a and prefix_k_pe should have same shape, expect for last dim
    """
    parallel = get_parallel()
    if not parallel.dcp_enabled:
        return torch.cat([prefix_kv_a, prefix_k_pe], dim=-1)
    # 1. compute max kv_lens for each seq
    dcp_world_size = parallel.dcp_size
    dcp_rank = parallel.dcp_rank

    if prefix_starts_cpu is None:
        prefix_starts_cpu = torch.zeros_like(prefix_kv_lens_cpu)

    left_pads = prefix_starts_cpu % dcp_world_size > dcp_rank
    left_pads = left_pads.to(torch.int32)
    right_pads = (
        prefix_starts_cpu + prefix_kv_lens_cpu - 1
    ) % dcp_world_size < dcp_rank
    right_pads = right_pads.to(torch.int32)
    padded_lens = (
        prefix_kv_lens_cpu + (prefix_starts_cpu % dcp_world_size) + dcp_world_size - 1
    ) // dcp_world_size

    local_kv_lens = padded_lens - left_pads - right_pads
    local_kv_lens_cu = torch.zeros(
        len(prefix_kv_lens_cpu) + 1,
        dtype=torch.int32,
    )
    local_kv_lens_cu[1:] = torch.cumsum(local_kv_lens, dim=0)

    padded_kv_cache_arr = []
    prefix_kv_cache = torch.cat([prefix_kv_a, prefix_k_pe], dim=-1)
    for req_idx in range(len(prefix_kv_lens_cpu)):
        padded_tensor = prefix_kv_cache.new_empty(
            (padded_lens[req_idx].item(),) + prefix_kv_cache.size()[1:]
        )
        padded_tensor[
            left_pads[req_idx] : left_pads[req_idx] + local_kv_lens[req_idx]
        ] = prefix_kv_cache[local_kv_lens_cu[req_idx] : local_kv_lens_cu[req_idx + 1]]
        padded_kv_cache_arr.append(padded_tensor)

    padded_kv_cache = torch.cat(padded_kv_cache_arr, dim=0)

    gatherd_kv_cache = _all_gather_dcp_kv_cache(padded_kv_cache)

    # 2. re-org kv cache to query orders
    padded_lens_cu = torch.zeros(
        len(prefix_kv_lens_cpu) + 1,
        dtype=torch.int32,
    )
    padded_lens_cu[1:] = torch.cumsum(padded_lens, dim=0)
    kv_cache_tuple = ()
    for req_idx in range(len(prefix_kv_lens_cpu)):
        kv_cache_tuple += (
            gatherd_kv_cache[
                padded_lens_cu[req_idx] * dcp_world_size
                + (prefix_starts_cpu[req_idx] % dcp_world_size) :
            ][: prefix_kv_lens_cpu[req_idx]],
        )
    gatherd_kv_cache = torch.cat(kv_cache_tuple, dim=0)

    return gatherd_kv_cache
