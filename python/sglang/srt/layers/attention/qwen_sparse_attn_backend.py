"""Sparse-attention backend for Qwen4-Exp models with an indexer.

The backend is installed as the full-attention side of Qwen4-Exp's hybrid backend;
linear-attention layers continue to use GDN.
"""

from __future__ import annotations

import logging
import math
import os
from copy import copy
from functools import lru_cache
from typing import Dict, Optional, Tuple

import msgspec
import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.qsa.config import (
    is_qwen_qsa,
    parse_qsa_profile,
)
from sglang.srt.layers.attention.qsa.kernel import qsa_sparse_attention
from sglang.srt.layers.attention.qsa.metadata import (
    QSAIndexerMetadata,
    build_group_ring_slots,
    build_pending_ring_slots,
    build_rope_position_matrix,
    compressed_decode_view,
)
from sglang.srt.layers.attention.qsa.sparse_attn import (
    sparse_attn_rows_triton,
    qwen_sparse_fa2_cu_seqlens_triton,
    qwen_sparse_kv_extraction_compact_triton,
    qwen_sparse_valid_counts_triton,
    sparse_gqa_fwd_interface_triton,
    sparse_gqa_fwd_interface_triton_ck,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode

logger = logging.getLogger(__name__)


_TRTLLM_SPARSE_PAGE_SIZE = 64


@lru_cache(maxsize=1)
def _resolve_trtllm_sparse_decode():
    """FlashInfer paged decode for the post-gather sparse attention;
    the FA4 varlen fallback runs a prefill-shaped kernel at decode row counts."""
    from sglang.srt.utils import is_sm100_supported, is_sm120

    # This path is numerically validated on SM100 and SM120. Do not widen it
    # to every SM12x device: it silently corrupts long-context decode on
    # SM121/GB10.
    if not (is_sm100_supported() or is_sm120()):
        return None
    try:
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache
    except ImportError:
        return None
    return trtllm_batch_decode_with_kv_cache


_QSA_ROWS_COMPACT = {"on": None}


_QSA_ROWS_FUSED = {"on": None}


def _qsa_rows_fused_on() -> bool:
    """SGLANG_QSA_ROWS_FUSED (default 1): graph-path rows resolve + owner
    compaction in one Triton launch (qsa/rows_resolve.py); 0 = the torch
    chain + compact_owned_rows (the fn8ab form)."""
    if _QSA_ROWS_FUSED["on"] is None:
        _QSA_ROWS_FUSED["on"] = os.environ.get("SGLANG_QSA_ROWS_FUSED", "1").strip().lower() not in ("0", "false", "off", "")
    return _QSA_ROWS_FUSED["on"]


def _qsa_rows_fused_route(metadata, topk_indices, req_to_token, eager: bool) -> bool:
    """Whether _rows_and_counts resolves through the fused Triton launch
    (qsa/rows_resolve.py): always on the graph path (Task #53), and on an
    EAGER forward only with SGLANG_WEG2_QSA_ROWS_FUSED_EAGER (fnFL2 H65).

    H65: the eager torch chain (_logical_to_physical -> _local_rows) of a
    16k P prefix chunk holds, per full-attention layer, the int64 copy of the
    [16384, 2051] top-k (269 MB), the gathered slots, a full_like and the
    where result (134 MB each) plus the masks -- ~0.57 GB of transient above
    the 134-MB rows it returns; the fused kernel writes the rows (and the
    counts) directly. Same rows for the valid-first top-k QSA hands over;
    the counts bound the kernel loop, whose trailing all -1 blocks were exact
    no-ops -- the attention output is bit-identical."""
    return bool(
        _qsa_rows_fused_on()
        and (getattr(metadata, "is_cuda_graph", False) or eager)
        and topk_indices.is_cuda
        and req_to_token is not None
        and metadata.row_req_pool_indices is not None
    )


def _qsa_rows_compact_on() -> bool:
    """SGLANG_QSA_ROWS_COMPACT (default 1): under DCP, bound every query's
    sparse-attention loop to the rows this rank owns instead of masking the
    foreign lanes (Task #42)."""
    if _QSA_ROWS_COMPACT["on"] is None:
        import os

        _QSA_ROWS_COMPACT["on"] = str(os.environ.get("SGLANG_QSA_ROWS_COMPACT", "1")).strip() not in ("", "0")
    return bool(_QSA_ROWS_COMPACT["on"])


@lru_cache(maxsize=1)
def _qsa_rows_path_armed() -> bool:
    """fn5e 2026-09-16 (PP=3, a 3080 stage): the packed varlen fallback of
    the paged decode path is FA4 cute on this venv (no FA2), and cute's MLIR
    refused to build on sm86 ('Operation creation failed', pack_gqa).  The
    DCP rows kernel (WP3b) already handles dcp_size <= 1 (rows = slots) and
    is the proven decode path of the TP=3 boots, so it is the default on
    every rank; SGLANG_QSA_ROWS_PATH=0 restores the paged FA path."""
    import os
    return os.environ.get("SGLANG_QSA_ROWS_PATH", "1") != "0"


_QSA_DENSE_CHECK_LOGS = [0]


def _qsa_dense_check_mode() -> str:
    """fn5i 2026-09-19: SGLANG_QSA_DENSE_CHECK=1 compares every single-request,
    prefix-free extend (target prefill AND the draft's extend after prefill)
    against a dense causal reference over the same q/k/v and logs the
    deviation; =subst additionally returns the reference (the draft extend
    then runs on exact attention, so SPEC-TF tells whether the rest of the
    draft forward is sound)."""
    import os

    value = os.environ.get("SGLANG_QSA_DENSE_CHECK", "").strip().lower()
    if value in ("", "0", "off", "false"):
        return ""
    if value in ("subst", "subst_global"):
        return value
    return "check"


def _qsa_global_kv_map(
    head_offset: int, local_heads: int, total_heads: int, total_kv_heads: int
) -> torch.Tensor:
    """kv head index of every LOCAL q head under the model's GLOBAL GQA
    grouping: global q head h belongs to kv head ``h // (H // KV)``. Under the
    REPLICATED-KV geometry (kv < tp) a rank holds ALL kv heads but only a
    slice of the q heads, so the kernels' local grouping
    ``q_local // (hq_local // hkv)`` names the wrong kv head for every rank
    whose slice does not start at a kv-group boundary AND span whole groups
    (fn5i 19.09.: rank 0 = heads 0..11 = all of kv group 0, kernel pairs
    heads 6..11 with kv head 1)."""
    if total_heads % total_kv_heads:
        raise ValueError(f"heads {total_heads} not a multiple of kv heads {total_kv_heads}")
    group = total_heads // total_kv_heads
    return (torch.arange(local_heads) + int(head_offset)) // group


def _qsa_dense_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scaling: float,
    kv_map: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dense causal GQA attention in fp32 over one prefix-free sequence.

    ``q`` [n, hq, d], ``k``/``v`` [n, hkv, d]. Without ``kv_map`` kv head j
    serves q heads ``j*g .. (j+1)*g-1`` with ``g = hq // hkv`` (the sparse
    kernels' LOCAL grouping); with ``kv_map`` ([hq] kv index per q head) the
    given assignment is used (see _qsa_global_kv_map). Returns [n, hq*d] in
    q's dtype."""
    n, hq, d = q.shape
    hkv = k.shape[1]
    qf = q.float().transpose(0, 1)
    if kv_map is None:
        if hq % hkv:
            raise ValueError(f"q heads {hq} not a multiple of kv heads {hkv}")
        g = hq // hkv
        kf = k[:n].float().transpose(0, 1).repeat_interleave(g, 0)
        vf = v[:n].float().transpose(0, 1).repeat_interleave(g, 0)
    else:
        idx = kv_map.to(device=q.device, dtype=torch.long)
        kf = k[:n].float().transpose(0, 1).index_select(0, idx)
        vf = v[:n].float().transpose(0, 1).index_select(0, idx)
    scores = torch.matmul(qf, kf.transpose(1, 2)) * float(scaling)
    causal = torch.ones(n, n, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, vf)
    return out.transpose(0, 1).reshape(n, hq * d).to(q.dtype)


def _qsa_dense_check_eligible(forward_batch, rows: int, k: torch.Tensor) -> bool:
    seq_lens = getattr(forward_batch, "seq_lens_cpu", None)
    ext_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if seq_lens is None or ext_lens is None or len(seq_lens) != 1 or len(ext_lens) != 1:
        return False
    if int(seq_lens[0]) != int(ext_lens[0]) or int(ext_lens[0]) != rows:
        return False
    return k.dim() == 3 and k.shape[0] >= rows and 1 < rows <= 2048


def _speculative_rows_route(
    dcp_size: int, rows_armed: bool, q_is_cuda: bool, fa_available: bool
) -> bool:
    """Whether a speculative-paged forward (target verify / draft extend)
    takes the WP3b rows kernel: always under DCP, on every CUDA rank while the
    rows path is armed (SGLANG_QSA_ROWS_PATH, default on -- the same rule as
    the decode path since fn5e), and whenever no FA varlen kernel exists."""
    return bool(dcp_size > 1 or (rows_armed and q_is_cuda) or not fa_available)


def _resolve_flash_attn_varlen_func_or_none():
    try:
        return _resolve_flash_attn_varlen_func()
    except (ImportError, AttributeError):
        return None


def _resolve_flash_attn_varlen_func():
    from sglang.srt.utils import is_sm121

    if is_sm121():
        from sglang.kernels.ops.attention import (
            qwen38_qsa_sm121_varlen,
        )

        return qwen38_qsa_sm121_varlen
    try:
        from flash_attn import flash_attn_varlen_func

        return flash_attn_varlen_func
    except ImportError:
        pass
    try:
        from flash_attn.cute.interface import flash_attn_varlen_func as cute_varlen_func

        def flash_attn_varlen_func(*args, **kwargs):
            output = cute_varlen_func(*args, **kwargs)
            # The cute interface returns (out, lse); lse is None here.
            return output[0] if isinstance(output, tuple) else output

        return flash_attn_varlen_func
    except (ImportError, AttributeError) as exc:
        # AttributeError: an FA4 build that does not match the installed CuTe DSL (e.g. FA4 b15 with
        # cutlass-dsl 4.7.1) fails inside its own import; treat it like a missing FA4.
        raise ImportError(
            "QSA decode requires flash_attn (FA2) or flash-attn-4 "
            "(FA4 cute) for its packed varlen fallback."
        ) from exc


class QwenSparseAttnMetadata(msgspec.Struct, frozen=True):
    """Per-forward metadata consumed by core sparse attention."""

    sequence_lengths: torch.Tensor
    token_to_batch_idx: torch.Tensor
    token_slot_table: torch.Tensor
    indexer_metadata: QSAIndexerMetadata
    row_req_pool_indices: Optional[torch.Tensor] = None
    is_cuda_graph: bool = False
    fa2_valid_counts: Optional[torch.Tensor] = None
    fa2_cu_seqlens_k: Optional[torch.Tensor] = None
    fa2_cu_seqlens_q: Optional[torch.Tensor] = None


class QSAMTPSharedSparseIndices:
    """Draft-extend top-k reused by one MTP iteration's decode steps,
    causally valid because a draft moves at most speculative_num_steps positions.
    Row ``num_requests`` is the trash row for padded/degenerate requests."""

    def __init__(
        self, *, layer_ids, num_requests, token_topk, tail_width, device
    ) -> None:
        self.layer_slots = {int(l): i for i, l in enumerate(sorted(layer_ids))}
        self.tail_width = tail_width
        self.trash_row = num_requests
        # Logical index 0 keeps never-captured rows (graph warmup dummies)
        # attending exactly the first token instead of an empty/invalid set.
        self.indices = torch.zeros(
            (len(self.layer_slots), num_requests + 1, token_topk + tail_width),
            dtype=torch.int32,
            device=device,
        )
        self.captured_len = torch.ones(
            (len(self.layer_slots), num_requests + 1),
            dtype=torch.int32,
            device=device,
        )
        self._tail_offsets = torch.arange(tail_width, device=device)

    def capture(
        self,
        topk_indices: torch.Tensor,
        req_pool_indices: torch.Tensor,
        captured_lens: torch.Tensor,
        layer_id: int,
    ) -> None:
        slot = self.layer_slots[int(layer_id)]
        rows = req_pool_indices.to(torch.long)
        self.indices[slot, :, : topk_indices.shape[1]].index_copy_(
            0, rows, topk_indices.to(self.indices.dtype)
        )
        self.captured_len[slot].index_copy_(0, rows, captured_lens.to(torch.int32))

    def lookup(
        self,
        req_pool_indices: torch.Tensor,
        current_positions: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Frozen selection plus a tail of positions drafted since the capture;
        tail slots past current_position hold -1, which downstream drops."""
        slot = self.layer_slots[int(layer_id)]
        rows = req_pool_indices.to(torch.long)
        out = self.indices[slot, rows]
        base = self.captured_len[slot, rows].to(torch.int64)
        tail = base.unsqueeze(1) + self._tail_offsets.unsqueeze(0)
        valid = tail <= current_positions.to(torch.int64).unsqueeze(1)
        out[:, out.shape[1] - self.tail_width :] = torch.where(valid, tail, -1).to(
            out.dtype
        )
        return out


class QwenSparseAttnBackend(AttentionBackend):
    """QSA backend using trtllm-gen decode with a packed FA2/FA4 fallback."""

    # Every seq_lens_cpu read here has a device fallback (one readback);
    # the graphed decode path never reads it, so opting out is safe.
    needs_cpu_seq_lens: bool = False

    def __init__(self, runner=None) -> None:
        self.runner = runner
        self.token_to_kv_pool = getattr(runner, "token_to_kv_pool", None)
        self.device = getattr(runner, "device", None)
        model_config = getattr(runner, "model_config", None)
        config = getattr(model_config, "hf_text_config", None)
        if config is None:
            config = getattr(model_config, "hf_config", None)
        self.qsa_profile = parse_qsa_profile(config)
        self.max_context_len = int(getattr(model_config, "context_len", 0))
        self.compress_ratio = (
            self.qsa_profile.compress_ratio
            if self.qsa_profile is not None
            else int(getattr(config, "indexer_compress_ratio", 4))
        )
        req_pool = getattr(runner, "req_to_token_pool", None)
        self.req_to_token = getattr(req_pool, "req_to_token", None)
        self.req_to_token_pool = req_pool
        self.forward_metadata: Optional[QwenSparseAttnMetadata] = None
        self._init_dcp(runner, model_config)
        self._cuda_graph_metadata: Dict[
            Tuple[ForwardMode, int], QwenSparseAttnMetadata
        ] = {}
        self._cuda_graph_max_tokens = 0
        self._fa2_scratch: Dict[
            Tuple[int, int, torch.dtype, torch.device],
            Tuple[torch.Tensor, torch.Tensor],
        ] = {}
        self._graph_seq_lens = None
        self._graph_token_to_batch = None
        self._graph_cu_seqlens_q = None
        self._graph_fa2_valid_counts = None
        self._graph_fa2_cu_seqlens_k = None
        self._graph_write_locs = None
        self._graph_compressed_page_table = None
        self._graph_compressed_lengths = None
        self._graph_prefix_lengths = None
        self._graph_dummy_token_slot_table = None
        self._graph_dummy_out_cache_loc = None
        self._graph_row_req_pool_indices = None
        self._trtllm_sparse_tables = {}
        self._mtp_shared_sparse_indices = None
        self._trtllm_workspace = None
        self._graph_extend_lens = None
        self._graph_extend_lens_pin = None

    # ------------------------------------------------------------------
    # WP3b: uneven DCP (this line's token-sharded KV pool) for the sparse
    # attention. Upstream's QSA backend reads and writes the pool with the
    # GLOBAL req_to_token slots; under --rank-tp-ratio every rank stores only
    # the rows it owns (layers/dcp/owner.py), so the write is masked to the
    # owned rows and the read runs over this rank's OWNED subset of every
    # query's top-k, on ALL q heads (gathered across the DCP group), and the
    # partials are LSE-merged exactly as the dense Triton/FlashInfer paths do.
    # fn1u boot 2026-09-16: without this TP0 (4100 of 32772 rows) asserted
    # out of bounds in the indexer's compressed cache; the sparse attention
    # would have been next.
    # ------------------------------------------------------------------
    def _init_dcp(self, runner, model_config) -> None:
        self.dcp_size = 1
        self.dcp_rank = 0
        self.uneven_dcp = False
        self.uneven_dcp_weighted = False
        self.cp_S = self.cp_lo = self.cp_hi = self.cp_ratio = 0
        self.dcp_kv_replicated_heads = True
        self.dcp_model_config = model_config
        self.is_draft_worker = bool(getattr(runner, "is_draft_worker", False))
        if runner is None or model_config is None:
            return
        try:
            from sglang.srt.runtime_context import get_parallel

            parallel = get_parallel()
            dcp_size = int(parallel.attn_dcp_size)
            dcp_rank = int(parallel.attn_dcp_rank)
            attn_tp_size = int(parallel.attn_tp_size)
        except Exception:
            return
        if dcp_size <= 1:
            return
        from sglang.srt.distributed.utils import (
            attn_kv_replicated,
            uneven_dcp_active,
            uneven_dcp_kv_replicated,
        )
        from sglang.srt.layers.dcp.owner import (
            dcp_weighted_owner_bounds,
            register_owner_bounds_consumer,
        )

        uneven_plan = uneven_dcp_kv_replicated(dcp_size)
        if self.is_draft_worker:
            # Same exclusion as the dense backends: the draft's pool keeps the
            # full token context, so DCP is off for this instance.
            return
        self.dcp_size = dcp_size
        self.dcp_rank = dcp_rank
        self.uneven_dcp = bool(uneven_plan)
        self.uneven_dcp_weighted = self.uneven_dcp and uneven_dcp_active(dcp_size)
        total_kv = int(model_config.get_total_num_kv_heads())
        self.dcp_kv_replicated_heads = (
            attn_kv_replicated(attn_tp_size, total_kv) if self.uneven_dcp else False
        )
        if self.uneven_dcp and not self.dcp_kv_replicated_heads:
            raise ValueError(
                "QSA sparse attention under uneven DCP needs replicated kv heads "
                f"(kv heads {total_kv} >= attention TP {attn_tp_size}); the "
                "head-sharded kv gather is not wired for this backend."
            )
        if self.uneven_dcp_weighted:
            self.cp_S, self.cp_lo, self.cp_hi, self.cp_ratio = dcp_weighted_owner_bounds(
                dcp_size, dcp_rank
            )
            register_owner_bounds_consumer(self)
        logger.info(
            "[qsa-dcp] sparse attention on DCP rank %d/%d (%s owner rule%s)",
            dcp_rank,
            dcp_size,
            "weighted" if self.uneven_dcp_weighted else "even",
            f", cp_S={self.cp_S} [{self.cp_lo},{self.cp_hi})" if self.uneven_dcp_weighted else "",
        )

    def refresh_dcp_owner_bounds(self) -> None:
        """#297 cutover hook (the owner-bounds registry requires it of every
        consumer): re-derive the cached weighted owner bounds from the freshly
        installed token vector. Idle-boundary only, never mid-forward. fn1v
        boot 2026-09-16: registering without it refused at backend init."""
        if not self.uneven_dcp_weighted:
            return
        from sglang.srt.layers.dcp.owner import dcp_weighted_owner_bounds

        self.cp_S, self.cp_lo, self.cp_hi, self.cp_ratio = dcp_weighted_owner_bounds(
            self.dcp_size, self.dcp_rank
        )

    def _dcp_group_q_head_counts(self, local_heads: int) -> list:
        from sglang.srt.layers.attention.triton_backend import (
            _plan_aware_dcp_group_q_head_counts,
        )

        return _plan_aware_dcp_group_q_head_counts(
            self.dcp_model_config, self.dcp_size, local_heads
        )

    def _set_kv_buffer(self, forward_batch, layer, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store this forward's k/v: every row on a non-DCP pool, only the
        owned rows (at their compact slot) under DCP -- the dense backends'
        write rule, stated through the same shared helpers."""
        loc = forward_batch.out_cache_loc
        if self.dcp_size <= 1:
            self.token_to_kv_pool.set_kv_buffer(layer, loc, k, v)
            return
        if self.uneven_dcp_weighted:
            from sglang.srt.layers.dcp.owner import dcp_weighted_write_slots

            loc, mask = dcp_weighted_write_slots(
                loc, self.cp_S, self.cp_lo, self.cp_hi, self.cp_ratio
            )
        else:
            from sglang.srt.layers.dcp.owner import dcp_even_write_mask

            mask = dcp_even_write_mask(
                forward_batch.positions,
                loc.numel(),
                self.dcp_size,
                self.dcp_rank,
                getattr(forward_batch, "dcp_kv_mask", None),
            )
            loc = loc // self.dcp_size
        self.token_to_kv_pool.set_kv_buffer(layer, loc, k, v, dcp_kv_mask=mask)

    def _local_rows(self, slots: torch.Tensor) -> torch.Tensor:
        """Global KV slots (-1 = none) -> this rank's pool rows, -1 where the
        slot is invalid or owned by another rank."""
        if self.dcp_size <= 1:
            return slots.to(torch.int32)
        valid = slots >= 0
        if self.uneven_dcp_weighted:
            from sglang.srt.layers.dcp.owner import dcp_weighted_read_slots

            compact, owned = dcp_weighted_read_slots(
                slots.clamp(min=0), self.cp_S, self.cp_lo, self.cp_hi, self.cp_ratio
            )
        else:
            safe = slots.clamp(min=0)
            owned = (safe % self.dcp_size) == self.dcp_rank
            compact = (safe // self.dcp_size).to(torch.int32)
        return torch.where(valid & owned, compact, torch.full_like(compact, -1))

    def _topk_rows(self, topk_indices: torch.Tensor, metadata) -> torch.Tensor:
        """Per-query top-k (logical positions) -> this rank's pool rows."""
        if getattr(metadata, "is_cuda_graph", False):
            slots = self._logical_to_physical_graph(topk_indices, metadata)
        else:
            slots = self._logical_to_physical(topk_indices, metadata)
        return self._local_rows(slots)

    def _rows_and_counts(self, topk_indices: torch.Tensor, metadata):
        """(rows, counts-or-None): the graph path resolves and compacts in ONE
        Triton launch (qsa/rows_resolve.py, Task #53 Sitz 5) when
        SGLANG_QSA_ROWS_FUSED is on (default); every other path keeps the
        torch chain and lets _attend_rows compact -- unless
        SGLANG_WEG2_QSA_ROWS_FUSED_EAGER routes the eager forwards (the P
        prefix chunks) through the same launch (_qsa_rows_fused_route, H65)."""
        from sglang.srt.environ import envs

        eager = bool(envs.SGLANG_WEG2_QSA_ROWS_FUSED_EAGER.get())
        if _qsa_rows_fused_route(metadata, topk_indices, self.req_to_token, eager):
            from sglang.srt.layers.attention.qsa.rows_resolve import (
                MODE_EVEN,
                MODE_NONE,
                MODE_WEIGHTED,
                qsa_rows_resolve,
            )

            if self.dcp_size <= 1:
                mode, kw = MODE_NONE, {}
            elif self.uneven_dcp_weighted:
                mode, kw = MODE_WEIGHTED, dict(
                    cp_S=self.cp_S, cp_lo=self.cp_lo, cp_hi=self.cp_hi, cp_ratio=self.cp_ratio
                )
            else:
                mode, kw = MODE_EVEN, dict(world=self.dcp_size, rank=self.dcp_rank)
            if not getattr(self, "_rows_fused_logged", False):
                self._rows_fused_logged = True
                logger.info(
                    "[qsa-rows] fused resolve+compaction armed (mode %d, dcp %d, K %d)",
                    mode, self.dcp_size, int(topk_indices.shape[1]),
                )
            if not getattr(metadata, "is_cuda_graph", False) and not getattr(
                self, "_rows_fused_eager_logged", False
            ):
                self._rows_fused_eager_logged = True
                logger.info(
                    "[qsa-rows] EAGER forwards resolve through the fused launch too "
                    "(SGLANG_WEG2_QSA_ROWS_FUSED_EAGER, H65; first: %d rows x K %d, mode %d)",
                    int(topk_indices.shape[0]), int(topk_indices.shape[1]), mode,
                )
            return qsa_rows_resolve(
                topk_indices,
                metadata.token_to_batch_idx,
                metadata.sequence_lengths,
                metadata.row_req_pool_indices,
                self.req_to_token,
                mode=mode,
                **kw,
            )
        return self._topk_rows(topk_indices, metadata), None

    def _logical_to_physical_graph(
        self, logical_indices: torch.Tensor, metadata
    ) -> torch.Tensor:
        """The rows path under a CUDA graph (fn3q, 19.09.): the capture-time
        ``token_slot_table`` is the ``(rows, 1)`` dummy of
        ``_capture_cuda_graph_metadata`` and NO replay refresh rewrites it (the
        replay-prep rebuilds the paged-path buffers only), so
        ``_logical_to_physical`` clamped every top-k position to column 0 and
        every graph decode attended KV slot 0 -- coherent text, no context
        (needle MISS at 351 and 10k tokens alike; the #452 B2 divergence at
        character 5). Resolve through ``req_to_token`` instead: a stable
        address, indexed by ``row_req_pool_indices`` -- a graph buffer the
        replay-prep does refresh -- which is exactly what the eager table
        (``self.req_to_token[req_pool_indices]``) holds. Pure device ops, so
        the capture records it and every replay reads the live table."""
        req_to_token = self.req_to_token
        if req_to_token is None or metadata.row_req_pool_indices is None:
            raise RuntimeError(
                "QSA graph rows path needs req_to_token and row_req_pool_indices"
            )
        sequence_ids = metadata.token_to_batch_idx.long()
        if sequence_ids.numel() != logical_indices.shape[0]:
            raise ValueError("QSA top-k rows do not match query rows")
        row_lengths = metadata.sequence_lengths.to(torch.int32).index_select(
            0, sequence_ids
        )
        valid = (logical_indices >= 0) & (logical_indices < row_lengths.unsqueeze(1))
        req_rows = metadata.row_req_pool_indices.long().index_select(0, sequence_ids)
        safe = logical_indices.clamp(min=0, max=req_to_token.shape[1] - 1).long()
        slots = req_to_token[req_rows[:, None], safe]
        return torch.where(valid, slots, torch.full_like(slots, -1)).to(torch.int32)

    def _attend_rows(
        self, q: torch.Tensor, layer, rows: torch.Tensor, row_counts=None
    ) -> torch.Tensor:
        """Sparse attention over ``rows`` for this rank's q heads; under DCP the
        group's heads are gathered, the owned partial computed, and the
        partials LSE-merged back to this rank's heads."""
        pool = self.token_to_kv_pool
        k_pool = pool.get_key_buffer(layer.layer_id)
        v_pool = pool.get_value_buffer(layer.layer_id)
        if self.dcp_size <= 1:
            out, _ = sparse_attn_rows_triton(
                q.contiguous(), k_pool, v_pool, rows, layer.scaling, row_counts=row_counts
            )
            return out
        from sglang.srt.layers.dcp.comm import (
            cp_all_gather_heads_uneven,
            cp_lse_ag_out_a2a_mha_uneven,
            cp_lse_ag_out_ar_mha_uneven,
            lse_merge_mode,
        )
        from sglang.srt.runtime_context import get_parallel

        group = get_parallel().dcp_group
        counts = self._dcp_group_q_head_counts(q.shape[1])
        q_all = cp_all_gather_heads_uneven(q.contiguous(), group, counts)
        if row_counts is None and _qsa_rows_compact_on():
            # Task #42: loop only over the rows this rank owns (see
            # sparse_attn.compact_owned_rows); the foreign lanes are -1 here.
            from sglang.srt.layers.attention.qsa.sparse_attn import compact_owned_rows

            rows, row_counts = compact_owned_rows(rows)
        if row_counts is not None:
            out, lse = sparse_attn_rows_triton(
                q_all, k_pool, v_pool, rows, layer.scaling, row_counts=row_counts
            )
        else:
            out, lse = sparse_attn_rows_triton(q_all, k_pool, v_pool, rows, layer.scaling)
        # The merge scales in fp32 and hands back fp32 (fn1x 2026-09-16: every
        # rank died in o_proj with 'float != BFloat16'); the layer's output
        # projection expects the query dtype.
        merge = cp_lse_ag_out_a2a_mha_uneven if lse_merge_mode() == "a2a" else cp_lse_ag_out_ar_mha_uneven
        return merge(out, lse, group, counts).to(q.dtype)

    @staticmethod
    def _is_speculative_paged_mode(forward_mode) -> bool:
        if forward_mode is None:
            return False
        return forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2()

    def _require_chain_speculation(self, forward_mode, spec_info) -> None:
        if forward_mode is None or not forward_mode.is_target_verify():
            return
        if int(getattr(spec_info, "topk", 1) or 1) != 1:
            raise NotImplementedError(
                "Qwen QSA target verification supports only speculative_eagle_topk=1"
            )
        draft_tokens = int(getattr(spec_info, "draft_token_num", 0) or 0)
        if draft_tokens > self.compress_ratio:
            # The pending-group ring keys state by position % ratio; a verify
            # window wider than the ratio would collide within one forward.
            raise NotImplementedError(
                "Qwen QSA requires speculative_num_draft_tokens <= the QSA "
                f"compress ratio ({self.compress_ratio}): the pending "
                f"index-key ring holds one group; got {draft_tokens}"
            )

    @staticmethod
    def _speculative_max_row_length(forward_batch, sequence_lengths) -> int:
        """Sync-free host-side over-bound for the token-slot gather width:
        request length plus the draft window covers every speculative row."""
        seq_lens_cpu = forward_batch.seq_lens_cpu
        if seq_lens_cpu is None or seq_lens_cpu.numel() == 0:
            logger.warning_once(
                "QSA speculative metadata without CPU request lengths: the "
                "token-slot bound reads them back once per forward"
            )
            return max(1, int(sequence_lengths.max()))
        spec_info = forward_batch.spec_info
        # Target verify exposes ``draft_token_num`` while draft-extend exposes
        # ``num_tokens_per_req`` (upstream form; EagleDraftExtendInput has no
        # draft_token_num). Both modes use this gather-width bound.
        draft_window = int(
            getattr(spec_info, "draft_token_num", getattr(spec_info, "num_tokens_per_req", 0))
            or 0
        )
        return max(1, int(seq_lens_cpu.max()) + draft_window)

    @staticmethod
    def _speculative_row_to_request(forward_batch, num_rows: int) -> torch.Tensor:
        batch_size = int(forward_batch.req_pool_indices.numel())
        if batch_size == 0:
            if num_rows == 0:
                return torch.zeros(
                    0,
                    dtype=torch.long,
                    device=forward_batch.req_pool_indices.device,
                )
            raise ValueError(
                "QSA speculative query rows cannot be mapped to an empty batch: "
                f"rows={num_rows}"
            )
        extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        if extend_seq_lens is not None:
            # DP token padding rows belong to no request; alias them to request row 0,
            # as the CUDA-graph layout does, so gathers stay in bounds.
            repeats = extend_seq_lens[:batch_size].to(dtype=torch.long)
            real_rows = int(repeats.sum().item())
            if real_rows > num_rows:
                raise ValueError(
                    "QSA speculative query rows are fewer than the extend "
                    f"mapping: rows={num_rows}, mapped={real_rows}"
                )
            row_to_request = torch.repeat_interleave(
                torch.arange(
                    batch_size,
                    dtype=torch.long,
                    device=forward_batch.req_pool_indices.device,
                ),
                repeats,
            )
            padding = num_rows - real_rows
            if padding:
                row_to_request = torch.cat(
                    [
                        row_to_request,
                        row_to_request.new_zeros(padding),
                    ]
                )
            return row_to_request
        if num_rows % batch_size != 0:
            raise ValueError(
                "QSA speculative query rows cannot be mapped to requests: "
                f"rows={num_rows}, batch={batch_size}"
            )
        return torch.arange(
            batch_size,
            dtype=torch.long,
            device=forward_batch.req_pool_indices.device,
        ).repeat_interleave(num_rows // batch_size)

    @staticmethod
    def _as_cpu_int_tensor(values, size: int) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            return values[:size].detach().cpu().to(torch.int32)
        return torch.tensor(values[:size], dtype=torch.int32)

    @classmethod
    def _graph_speculative_layout(
        cls,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens_cpu,
        forward_mode,
        spec_info,
        num_padding: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_lengths = cls._as_cpu_int_tensor(seq_lens_cpu, bs)
        if forward_mode.is_target_verify():
            extend_len = int(spec_info.draft_token_num)
            extend_lengths = torch.full((bs,), extend_len, dtype=torch.int32)
        else:
            extend_lengths_cpu = getattr(spec_info, "extend_seq_lens_cpu", None)
            if extend_lengths_cpu is not None:
                extend_lengths = cls._as_cpu_int_tensor(extend_lengths_cpu, bs)
                if extend_lengths.numel() < bs:
                    extend_lengths = torch.cat(
                        [
                            extend_lengths,
                            torch.zeros(bs - extend_lengths.numel(), dtype=torch.int32),
                        ]
                    )
            else:
                extend_lengths = torch.full(
                    (bs,), num_tokens // max(bs, 1), dtype=torch.int32
                )
        num_padding = max(0, min(int(num_padding), bs))
        if num_padding:
            extend_lengths = extend_lengths.clone()
            extend_lengths[bs - num_padding :] = 0
        effective_lengths = (
            base_lengths + extend_lengths
            if forward_mode.is_target_verify()
            else base_lengths
        )

        row_lengths = []
        row_prefix_lengths = []
        for seq_len, extend_len in zip(
            effective_lengths.tolist(), extend_lengths.tolist()
        ):
            prefix_len = max(int(seq_len) - int(extend_len), 0)
            row_lengths.append(
                torch.arange(
                    prefix_len + 1,
                    prefix_len + int(extend_len) + 1,
                    dtype=torch.int32,
                ).clamp_max(int(seq_len))
            )
            row_prefix_lengths.append(
                torch.full((int(extend_len),), prefix_len, dtype=torch.int32)
            )
        row_lengths = (
            torch.cat(row_lengths) if row_lengths else torch.empty(0, dtype=torch.int32)
        )
        row_prefix_lengths = (
            torch.cat(row_prefix_lengths)
            if row_prefix_lengths
            else torch.empty(0, dtype=torch.int32)
        )
        actual_rows = row_lengths.numel()
        if actual_rows > num_tokens:
            raise ValueError(
                "QSA CUDA graph speculative layout has inconsistent token count: "
                f"capacity={num_tokens}, actual={actual_rows}"
            )
        repeats = extend_lengths.to(device=req_pool_indices.device, dtype=torch.long)
        row_req_pool_indices = torch.repeat_interleave(req_pool_indices[:bs], repeats)
        # A draft-extend replay may hold fewer accepted rows than the captured shape;
        # the runner packs real rows first, so give the tail inert metadata.
        padding = num_tokens - actual_rows
        if padding:
            row_lengths = torch.cat(
                [row_lengths, torch.ones(padding, dtype=torch.int32)]
            )
            row_prefix_lengths = torch.cat(
                [row_prefix_lengths, torch.zeros(padding, dtype=torch.int32)]
            )
            dummy_req = (
                req_pool_indices[0]
                if req_pool_indices.numel()
                else torch.zeros((), dtype=torch.int32, device=req_pool_indices.device)
            )
            row_req_pool_indices = torch.cat(
                [
                    row_req_pool_indices,
                    dummy_req.to(row_req_pool_indices.dtype).expand(padding),
                ]
            )
        return row_lengths, row_req_pool_indices, row_prefix_lengths

    def _empty_metadata(self, forward_batch) -> QwenSparseAttnMetadata:
        """Well-formed zero-row metadata for IDLE/zero-request forwards."""

        device = forward_batch.seq_lens.device
        sequence_lengths = torch.zeros(0, dtype=torch.int32, device=device)
        token_to_batch_idx = torch.zeros(0, dtype=torch.int32, device=device)
        token_slot_table = torch.zeros((0, 1), dtype=torch.int32, device=device)
        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        if out_cache_loc is None:
            out_cache_loc = torch.zeros(0, dtype=torch.int64, device=device)
        indexer_metadata = QSAIndexerMetadata(
            sequence_lengths=sequence_lengths,
            token_to_batch_idx=token_to_batch_idx,
            token_slot_table=token_slot_table,
            out_cache_loc=out_cache_loc,
            token_to_kv_pool=self.token_to_kv_pool,
            compress_ratio=self.token_to_kv_pool.qsa_compress_ratio,
            block_topk=self.token_to_kv_pool.qsa_block_topk,
        )
        return QwenSparseAttnMetadata(
            sequence_lengths=sequence_lengths,
            token_to_batch_idx=token_to_batch_idx,
            token_slot_table=token_slot_table,
            indexer_metadata=indexer_metadata,
            row_req_pool_indices=torch.zeros(0, dtype=torch.int32, device=device),
        )

    @staticmethod
    def _qsa_write_plan(
        *,
        token_slot_table,
        start_blocks,
        end_blocks,
        capacity,
        compress_ratio,
        row_token_starts=None,
        prefix_lens=None,
    ):
        """Compact per-row block ranges into ``capacity`` write entries,
        a shape-derived bound (no sync); padding writes the inert reserved slot 0."""
        device = token_slot_table.device
        # The table width is a host-side bound;
        # assert on device so a short table fails loudly without a sync.
        torch._assert_async(
            (end_blocks * compress_ratio <= token_slot_table.shape[1]).all()
        )
        counts = (end_blocks - start_blocks).clamp_min(0)
        ends = torch.cumsum(counts, 0)
        starts = ends - counts
        entries = torch.arange(capacity, dtype=torch.long, device=device)
        # Which row owns each compacted entry, and its ordinal within the row.
        rows = torch.searchsorted(ends, entries, right=True)
        valid = entries < ends[-1] if counts.numel() else entries < 0
        rows = torch.where(valid, rows.clamp_max(max(counts.numel() - 1, 0)), 0)
        blocks = torch.where(valid, start_blocks[rows] + entries - starts[rows], 0)
        write_locs = torch.where(
            valid,
            token_slot_table[rows, blocks * compress_ratio].long() // compress_ratio,
            torch.zeros_like(blocks),
        ).to(torch.int32)
        group_end_positions = blocks * compress_ratio + (compress_ratio - 1)
        member_rows = None
        if row_token_starts is not None:
            # member_rows index this forward's packed token rows;
            # the chunk is group-aligned, so a group starts at block * ratio - prefix.
            member_rows = torch.where(
                valid,
                row_token_starts[rows] + blocks * compress_ratio - prefix_lens[rows],
                torch.zeros_like(blocks),
            )
        return write_locs, group_end_positions, rows, member_rows

    def _qsa_build_write_plan(
        self,
        *,
        forward_batch,
        speculative_paged: bool,
        token_slot_table,
        sequence_lengths,
    ):
        ratio = self.token_to_kv_pool.qsa_compress_ratio
        lengths = sequence_lengths.long()
        end_blocks = lengths // ratio
        if forward_batch.forward_mode.is_decode() or speculative_paged:
            # A paged row compresses exactly the block its length completes;
            # its members come from the per-request pending ring.
            start_blocks = end_blocks - (lengths % ratio == 0).long()
            return self._qsa_write_plan(
                token_slot_table=token_slot_table,
                start_blocks=start_blocks,
                end_blocks=end_blocks,
                capacity=int(lengths.numel()),
                compress_ratio=ratio,
            )
        extend_lens = forward_batch.extend_seq_lens
        if extend_lens is None:
            raise ValueError("QSA extend write plan requires extend_seq_lens")
        extend_lens = extend_lens.long()[: lengths.numel()]
        prefix_lens = (lengths - extend_lens).clamp_min(0)
        # Prefix sharing is page-granular and the page is a ratio
        # multiple, so a matched prefix always covers whole groups. A
        # misaligned prefix would leave a shared group half-written.
        torch._assert_async((prefix_lens % ratio == 0).all())
        # Each row spans at most ceil(extend_len / ratio) blocks, so the
        # token count and row count bound the plan without a sync.
        capacity = int(forward_batch.input_ids.numel()) // ratio + int(lengths.numel())
        row_token_starts = torch.cumsum(extend_lens, 0) - extend_lens
        return self._qsa_write_plan(
            token_slot_table=token_slot_table,
            start_blocks=prefix_lens // ratio,
            end_blocks=end_blocks,
            capacity=capacity,
            compress_ratio=ratio,
            row_token_starts=row_token_starts,
            prefix_lens=prefix_lens,
        )

    def _metadata_from_forward_batch(self, forward_batch) -> QwenSparseAttnMetadata:
        self._require_chain_speculation(
            forward_batch.forward_mode,
            getattr(forward_batch, "spec_info", None),
        )
        if self.device is None:
            self.device = forward_batch.seq_lens.device
        if not self.max_context_len:
            self.max_context_len = self.req_to_token.shape[1]
        if forward_batch.forward_mode.is_idle() or forward_batch.seq_lens.numel() == 0:
            # DP idle forwards reach metadata init even though layers skip attention;
            # so do the zero-row DECODE steps the MTP wrapper makes of them.
            return self._empty_metadata(forward_batch)
        original_mode = getattr(forward_batch, "_original_forward_mode", None)
        if original_mode is not None and self._is_speculative_paged_mode(original_mode):
            # DP MAX_LEN pseudo-extend sets extend_seq_lens == 1 per request,
            # losing the draft fan-out this mapping needs.
            raise ValueError(
                "QSA cannot build metadata for DP MAX_LEN pseudo-extend of "
                f"speculative mode {original_mode}: token rows would be "
                "mis-mapped to requests"
            )
        speculative_paged = self._is_speculative_paged_mode(forward_batch.forward_mode)
        if speculative_paged:
            logical_positions = forward_batch.positions
            if logical_positions.ndim == 2:
                logical_positions = logical_positions[0]
            logical_positions = logical_positions.flatten()
            sequence_lengths = (logical_positions + 1).to(torch.int32)
            row_to_request = self._speculative_row_to_request(
                forward_batch, logical_positions.numel()
            )
            row_req_pool_indices = forward_batch.req_pool_indices.index_select(
                0, row_to_request
            )
            max_length = self._speculative_max_row_length(
                forward_batch, sequence_lengths
            )
            token_slot_table = self.req_to_token[
                row_req_pool_indices.long(), :max_length
            ].to(torch.int32)
            token_to_batch_idx = torch.arange(
                sequence_lengths.numel(),
                device=sequence_lengths.device,
                dtype=torch.int32,
            )
        else:
            sequence_lengths = forward_batch.seq_lens.to(torch.int32)
            batch_size = sequence_lengths.numel()
            if forward_batch.seq_lens_cpu is not None:
                max_length = int(forward_batch.seq_lens_cpu[:batch_size].max())
            else:
                max_length = int(sequence_lengths.max())
            row_req_pool_indices = forward_batch.req_pool_indices[:batch_size]
            token_slot_table = self.req_to_token[
                row_req_pool_indices.long(), :max_length
            ].to(torch.int32)
            positions = forward_batch.positions
            num_position_tokens = (
                positions.shape[-1] if positions.ndim == 2 else positions.numel()
            )
            if forward_batch.forward_mode.is_decode():
                if num_position_tokens != batch_size:
                    raise ValueError(
                        "QSA decode requires exactly one query row per request: "
                        f"rows={num_position_tokens}, batch={batch_size}"
                    )
                token_to_batch_idx = torch.arange(
                    batch_size,
                    device=sequence_lengths.device,
                    dtype=torch.int32,
                )
            else:
                extend_seq_lens = forward_batch.extend_seq_lens
                if extend_seq_lens is None:
                    raise ValueError("QSA extend metadata requires extend_seq_lens")
                token_to_batch_idx = torch.repeat_interleave(
                    torch.arange(
                        batch_size,
                        device=sequence_lengths.device,
                        dtype=torch.int32,
                    ),
                    extend_seq_lens.to(torch.long),
                )
                # More mapped rows than physical would index past them;
                # fewer is fine (DP MAX_LEN padding is trimmed downstream).
                if token_to_batch_idx.numel() > num_position_tokens:
                    raise ValueError(
                        "QSA extend request mapping exceeds token rows: "
                        f"mapping={token_to_batch_idx.numel()}, "
                        f"positions={num_position_tokens}"
                    )
        decode_page_table = None
        decode_lengths = None
        decode_logical_positions = None
        pending_ring_slots = None
        compress_group_ring_locs = None
        extend_rope_matrix = None
        write_locs, group_positions, group_sequence_ids, group_member_rows = (
            self._qsa_build_write_plan(
                forward_batch=forward_batch,
                speculative_paged=speculative_paged,
                token_slot_table=token_slot_table,
                sequence_lengths=sequence_lengths,
            )
        )
        decode_like = speculative_paged or forward_batch.forward_mode.is_decode()
        if decode_like:
            decode_logical_positions = (
                logical_positions.to(torch.int32)
                if speculative_paged
                else sequence_lengths - 1
            )
            ring_logical_positions = decode_logical_positions
        else:
            extend_positions = forward_batch.positions
            if extend_positions.ndim == 2:
                extend_positions = extend_positions[0]
            ring_logical_positions = extend_positions.flatten()[
                : token_to_batch_idx.numel()
            ]
        if not self.should_reuse_mtp_sparse_indices(forward_batch):
            if decode_like:
                pool = self.token_to_kv_pool
                decode_page_table, decode_lengths = compressed_decode_view(
                    compressed_page_size=pool.qsa_compressed_page_size,
                    compress_ratio=pool.qsa_compress_ratio,
                    sequence_lengths=sequence_lengths,
                    token_slot_table=token_slot_table,
                )
            pending_ring_slots = build_pending_ring_slots(
                token_to_batch_idx=token_to_batch_idx,
                req_pool_indices=row_req_pool_indices,
                sequence_lengths=sequence_lengths,
                logical_positions=ring_logical_positions,
                compress_ratio=self.compress_ratio,
                is_extend=group_member_rows is not None,
            )
            if write_locs.numel():
                if group_member_rows is not None:
                    rope_source = (
                        forward_batch.mrope_positions
                        if forward_batch.mrope_positions is not None
                        else forward_batch.positions
                    )
                    extend_rope_matrix = build_rope_position_matrix(
                        rope_source, token_to_batch_idx.numel()
                    )
                else:
                    compress_group_ring_locs = build_group_ring_slots(
                        req_pool_indices=row_req_pool_indices,
                        group_end_positions=group_positions.long(),
                        sequence_ids=group_sequence_ids.long(),
                        compress_ratio=self.compress_ratio,
                    )
        indexer_metadata = QSAIndexerMetadata(
            sequence_lengths=sequence_lengths,
            token_to_batch_idx=token_to_batch_idx,
            token_slot_table=token_slot_table,
            out_cache_loc=forward_batch.out_cache_loc,
            token_to_kv_pool=self.token_to_kv_pool,
            compress_ratio=self.token_to_kv_pool.qsa_compress_ratio,
            block_topk=self.token_to_kv_pool.qsa_block_topk,
            req_pool_indices=row_req_pool_indices,
            write_locs=write_locs,
            compress_group_positions=group_positions,
            compress_sequence_ids=group_sequence_ids,
            compress_member_rows=group_member_rows,
            decode_page_table=decode_page_table,
            decode_lengths=decode_lengths,
            decode_logical_positions=decode_logical_positions,
            pending_ring_slots=pending_ring_slots,
            compress_group_ring_locs=compress_group_ring_locs,
            extend_rope_matrix=extend_rope_matrix,
        )
        return QwenSparseAttnMetadata(
            sequence_lengths=sequence_lengths,
            token_to_batch_idx=token_to_batch_idx,
            token_slot_table=token_slot_table,
            indexer_metadata=indexer_metadata,
            row_req_pool_indices=row_req_pool_indices,
        )

    def init_forward_metadata(self, forward_batch):
        if forward_batch.forward_mode.is_idle():
            self.forward_metadata = None
            return
        self.forward_metadata = self._metadata_from_forward_batch(forward_batch)

    def init_forward_metadata_out_graph(
        self,
        forward_batch,
        in_capture: bool = False,
    ):
        if in_capture:
            num_tokens = (
                forward_batch.input_ids.shape[0]
                if self._is_speculative_paged_mode(forward_batch.forward_mode)
                else forward_batch.batch_size
            )
            self._capture_cuda_graph_metadata(
                bs=forward_batch.batch_size,
                num_tokens=num_tokens,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                forward_mode=forward_batch.forward_mode,
                spec_info=forward_batch.spec_info,
            )
        else:
            num_padding = getattr(forward_batch, "num_padding", None)
            self._replay_cuda_graph_metadata(
                bs=forward_batch.batch_size,
                req_pool_indices=forward_batch.req_pool_indices,
                seq_lens=forward_batch.seq_lens,
                forward_mode=forward_batch.forward_mode,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
                num_padding=num_padding if num_padding is not None else 0,
            )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        if self.device is None:
            raise RuntimeError(
                "QSA backend requires a ModelRunner to initialize CUDA graph state"
            )
        self._cuda_graph_max_tokens = max_num_tokens
        max_blocks = math.ceil(self.max_context_len / self.compress_ratio)
        max_pages = max(
            1,
            math.ceil(max_blocks / self.token_to_kv_pool.qsa_compressed_page_size),
        )
        self._graph_seq_lens = torch.zeros(
            max_num_tokens, dtype=torch.int32, device=self.device
        )
        self._graph_token_to_batch = torch.arange(
            max_num_tokens, dtype=torch.int32, device=self.device
        )
        self._graph_cu_seqlens_q = torch.arange(
            max_num_tokens + 1, dtype=torch.int32, device=self.device
        )
        self._graph_fa2_valid_counts = torch.zeros(
            max_num_tokens, dtype=torch.int32, device=self.device
        )
        self._graph_fa2_cu_seqlens_k = torch.zeros(
            max_num_tokens + 1, dtype=torch.int32, device=self.device
        )
        self._graph_write_locs = torch.zeros(
            max_num_tokens, dtype=torch.int32, device=self.device
        )
        self._graph_compressed_page_table = torch.zeros(
            (max_num_tokens, max_pages), dtype=torch.int32, device=self.device
        )
        self._graph_compressed_lengths = torch.zeros(
            max_num_tokens, dtype=torch.int32, device=self.device
        )
        self._graph_prefix_lengths = torch.zeros(
            max_num_tokens, dtype=torch.int32, device=self.device
        )
        self._graph_dummy_token_slot_table = torch.zeros(
            (max_num_tokens, 1), dtype=torch.int32, device=self.device
        )
        self._graph_dummy_out_cache_loc = torch.zeros(
            max_num_tokens, dtype=torch.int64, device=self.device
        )
        self._graph_row_req_pool_indices = torch.zeros(
            max_num_tokens, dtype=torch.int32, device=self.device
        )
        self._graph_logical_positions = torch.zeros(
            max_num_tokens, dtype=torch.int32, device=self.device
        )
        self._graph_state_slots = torch.zeros(
            max_num_tokens, dtype=torch.int64, device=self.device
        )
        self._graph_ring_group_locs = torch.zeros(
            (max_num_tokens, self.compress_ratio),
            dtype=torch.int32,
            device=self.device,
        )
        # Two pinned halves: when a replay refills the pinned buffer,
        # the previous async HtoD copy from it may still be queued.
        self._graph_extend_lens = torch.zeros(
            max_bs, dtype=torch.int32, device=self.device
        )
        self._graph_extend_lens_pin = [
            torch.zeros(max_bs, dtype=torch.int32, pin_memory=True) for _ in range(2)
        ]
        self._extend_lens_pin_idx = 0

    def _capture_cuda_graph_metadata(
        self,
        *,
        bs,
        num_tokens,
        req_pool_indices,
        seq_lens,
        forward_mode,
        spec_info,
    ) -> None:
        self._require_chain_speculation(forward_mode, spec_info)
        if self.token_to_kv_pool is None:
            self.token_to_kv_pool = getattr(self.runner, "token_to_kv_pool", None)
        if self.req_to_token is None:
            req_pool = getattr(self.runner, "req_to_token_pool", None)
            self.req_to_token = getattr(req_pool, "req_to_token", None)
            self.req_to_token_pool = req_pool
        if self._graph_seq_lens is None or self.token_to_kv_pool is None:
            raise RuntimeError("QSA CUDA graph state is not initialized")
        pool = self.token_to_kv_pool
        speculative_paged = self._is_speculative_paged_mode(forward_mode)
        metadata_rows = num_tokens if speculative_paged else bs
        if speculative_paged:
            row_lengths, row_req_pool_indices, row_prefix_lengths = (
                self._graph_speculative_layout(
                    bs,
                    num_tokens,
                    req_pool_indices,
                    seq_lens[:bs].detach().cpu(),
                    forward_mode,
                    spec_info,
                    num_padding=bs,
                )
            )
            self._graph_prefix_lengths[:metadata_rows].copy_(
                row_prefix_lengths.to(device=self.device)
            )
            self._graph_seq_lens[:metadata_rows].copy_(
                row_lengths.to(device=self.device)
            )
            self._graph_row_req_pool_indices[:metadata_rows].copy_(
                row_req_pool_indices.to(torch.int32)
            )
        else:
            self._graph_seq_lens[:bs].copy_(seq_lens[:bs].to(torch.int32))
            self._graph_row_req_pool_indices[:bs].copy_(
                req_pool_indices[:bs].to(torch.int32)
            )
            self._graph_prefix_lengths[:bs].copy_(
                (seq_lens[:bs] - 1).clamp_min(0).to(torch.int32)
            )
        indexer_metadata = QSAIndexerMetadata(
            sequence_lengths=self._graph_seq_lens[:metadata_rows],
            token_to_batch_idx=self._graph_token_to_batch[:metadata_rows],
            token_slot_table=self._graph_dummy_token_slot_table[:metadata_rows],
            out_cache_loc=self._graph_dummy_out_cache_loc[:metadata_rows],
            token_to_kv_pool=pool,
            compress_ratio=pool.qsa_compress_ratio,
            block_topk=pool.qsa_block_topk,
            req_pool_indices=self._graph_row_req_pool_indices[:metadata_rows],
            is_cuda_graph=True,
            graph_write_locs=self._graph_write_locs[:metadata_rows],
            graph_compressed_page_table=self._graph_compressed_page_table[
                :metadata_rows
            ],
            graph_compressed_lengths=self._graph_compressed_lengths[:metadata_rows],
            graph_prefix_lengths=self._graph_prefix_lengths[:metadata_rows],
            decode_logical_positions=self._graph_logical_positions[:metadata_rows],
            pending_ring_slots=self._graph_state_slots[:metadata_rows],
            graph_ring_group_locs=self._graph_ring_group_locs[:metadata_rows],
        )
        metadata = QwenSparseAttnMetadata(
            sequence_lengths=self._graph_seq_lens[:metadata_rows],
            token_to_batch_idx=self._graph_token_to_batch[:metadata_rows],
            token_slot_table=self._graph_dummy_token_slot_table[:metadata_rows],
            indexer_metadata=indexer_metadata,
            row_req_pool_indices=self._graph_row_req_pool_indices[:metadata_rows],
            is_cuda_graph=True,
            fa2_valid_counts=self._graph_fa2_valid_counts[:metadata_rows],
            fa2_cu_seqlens_k=self._graph_fa2_cu_seqlens_k[: metadata_rows + 1],
            fa2_cu_seqlens_q=self._graph_cu_seqlens_q[: metadata_rows + 1],
        )
        self._cuda_graph_metadata[(forward_mode, bs)] = metadata
        self.forward_metadata = metadata

    def _replay_cuda_graph_metadata(
        self,
        *,
        bs,
        req_pool_indices,
        seq_lens,
        forward_mode,
        spec_info,
        seq_lens_cpu,
        num_padding: int = 0,
    ) -> None:
        self._require_chain_speculation(forward_mode, spec_info)
        metadata = self._cuda_graph_metadata[(forward_mode, bs)]
        # Compressed addressing is arithmetic over req_to_token; pure reads below.
        if self._can_replay_with_gpu_kernels(metadata, seq_lens):
            self._replay_cuda_graph_metadata_gpu(
                metadata,
                bs=bs,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                forward_mode=forward_mode,
                spec_info=spec_info,
                seq_lens_cpu=seq_lens_cpu,
                num_padding=num_padding,
            )
            self.forward_metadata = metadata
            return
        if seq_lens_cpu is None:
            # Host-fallback staging only (non-CUDA pools): an explicit
            # slow-path readback, never the serving path.
            seq_lens_cpu = seq_lens[:bs].cpu()
        if self._is_speculative_paged_mode(forward_mode):
            row_lengths, row_req_pool_indices, row_prefix_lengths = (
                self._graph_speculative_layout(
                    bs,
                    metadata.sequence_lengths.numel(),
                    req_pool_indices,
                    seq_lens_cpu,
                    forward_mode,
                    spec_info,
                    num_padding=num_padding or 0,
                )
            )
            metadata.sequence_lengths.copy_(row_lengths.to(device=self.device))
            metadata.row_req_pool_indices.copy_(row_req_pool_indices.to(torch.int32))
            metadata.indexer_metadata.graph_prefix_lengths.copy_(
                row_prefix_lengths.to(device=self.device)
            )
        else:
            metadata.sequence_lengths.copy_(seq_lens[:bs].to(torch.int32))
            metadata.row_req_pool_indices.copy_(req_pool_indices[:bs].to(torch.int32))
            metadata.indexer_metadata.graph_prefix_lengths.copy_(
                (seq_lens[:bs] - 1).clamp_min(0).to(torch.int32)
            )
        self._update_qsa_cuda_graph_metadata(
            metadata.indexer_metadata, metadata.row_req_pool_indices
        )
        self.forward_metadata = metadata

    def _can_replay_with_gpu_kernels(self, metadata, seq_lens) -> bool:
        if self.req_to_token is None:
            return False
        from sglang.srt.layers.attention.qsa.graph_metadata import (
            supports_graph_metadata_kernels,
        )

        return seq_lens.is_cuda and supports_graph_metadata_kernels(
            metadata.indexer_metadata.token_to_kv_pool, seq_lens.device
        )

    def _stage_extend_lens(self, spec_info, bs: int, num_tokens: int):

        extend_lens = getattr(spec_info, "extend_seq_lens_tensor", None)
        if extend_lens is not None and extend_lens.numel() >= bs:
            return extend_lens[:bs]
        pin = self._graph_extend_lens_pin[self._extend_lens_pin_idx]
        self._extend_lens_pin_idx = 1 - self._extend_lens_pin_idx
        values = getattr(spec_info, "extend_seq_lens_cpu", None)
        if values is None:
            # Mirrors the legacy fallback: an even per-request share of the
            # captured token capacity.
            pin[:bs].fill_(num_tokens // max(bs, 1))
        elif isinstance(values, torch.Tensor):
            pin[:bs].copy_(values[:bs].to(torch.int32))
        else:
            pin[:bs].copy_(torch.tensor(values[:bs], dtype=torch.int32))
        staged = self._graph_extend_lens
        staged[:bs].copy_(pin[:bs], non_blocking=True)
        return staged[:bs]

    def _replay_cuda_graph_metadata_gpu(
        self,
        metadata,
        *,
        bs,
        req_pool_indices,
        seq_lens,
        forward_mode,
        spec_info,
        seq_lens_cpu,
        num_padding: int = 0,
    ) -> None:
        """Sync-free replay refresh: nothing here may read back to the host,
        since launch_graph_metadata rebuilds every per-row graph buffer on device."""

        from sglang.srt.layers.attention.qsa.graph_metadata import (
            launch_graph_metadata,
        )

        indexer = metadata.indexer_metadata
        pool = indexer.token_to_kv_pool
        speculative = self._is_speculative_paged_mode(forward_mode)
        num_rows = metadata.sequence_lengths.numel() if speculative else bs
        num_padding = max(0, min(int(num_padding or 0), bs))
        if bs <= 0 or num_rows <= 0:
            return

        if speculative:
            if forward_mode.is_target_verify():
                mode = 1
                extend_lens = None
                extend_len = int(spec_info.draft_token_num)
            else:
                mode = 2
                extend_len = 0
                extend_lens = self._stage_extend_lens(spec_info, bs, num_rows)
        else:
            mode = 0
            extend_lens = None
            extend_len = 0
        launch_graph_metadata(
            mode=mode,
            bs=bs,
            num_rows=num_rows,
            seq_lens=seq_lens,
            req_pool_indices=req_pool_indices,
            extend_lens=extend_lens,
            extend_len=extend_len,
            num_padding=num_padding,
            metadata=metadata,
            req_to_token=self.req_to_token,
            pool=pool,
        )

    def _update_qsa_cuda_graph_metadata(
        self, metadata: QSAIndexerMetadata, req_pool_indices: torch.Tensor
    ) -> None:
        """Replay refresh for pools/devices without the graph_metadata kernels."""
        if self.req_to_token is None:
            raise RuntimeError("QSA req_to_token table is not initialized")
        pool = metadata.token_to_kv_pool
        lengths = metadata.sequence_lengths.long()
        req_indices = req_pool_indices.long()
        ratio = metadata.compress_ratio
        full_page = pool.qsa_compressed_page_size * ratio

        current_positions = (lengths - 1).clamp_min(0)
        compressed_lengths = torch.div(lengths, ratio, rounding_mode="floor")
        metadata.graph_compressed_lengths.copy_(compressed_lengths.to(torch.int32))

        # Boundary rows write their group's slot (last raw slot // ratio);
        # every other row keeps the inert reserved slot 0.
        boundary = (lengths % ratio == 0) & (lengths > 0)
        last_locs = self.req_to_token[req_indices, current_positions].long()
        write_locs = torch.where(
            boundary,
            last_locs // ratio,
            torch.zeros_like(lengths),
        )
        metadata.graph_write_locs.copy_(write_locs.to(torch.int32))

        metadata.decode_logical_positions.copy_(current_positions.to(torch.int32))
        metadata.pending_ring_slots.copy_(
            build_pending_ring_slots(
                token_to_batch_idx=metadata.token_to_batch_idx,
                req_pool_indices=req_pool_indices,
                sequence_lengths=metadata.sequence_lengths,
                logical_positions=current_positions,
                compress_ratio=ratio,
                is_extend=False,
            )
        )
        metadata.graph_ring_group_locs.copy_(
            build_group_ring_slots(
                req_pool_indices=req_pool_indices,
                group_end_positions=current_positions,
                sequence_ids=metadata.token_to_batch_idx.long(),
                compress_ratio=ratio,
            ).to(torch.int32)
        )

        # Page-table entries are full-KV page ids from the token-slot rows.
        page_table = metadata.graph_compressed_page_table
        max_pages = page_table.shape[1]
        row_width_pages = self.req_to_token.shape[1] // full_page
        num_pages = min(max_pages, row_width_pages)
        table = (
            self.req_to_token[req_indices, : num_pages * full_page : full_page].long()
            // full_page
        ).clamp_min(0)
        page_table[:, :num_pages].copy_(table.to(torch.int32))

    def get_indexer_metadata(self, layer_id: int, forward_batch):
        if self.forward_metadata is None:
            self.init_forward_metadata(forward_batch)
        assert self.forward_metadata is not None
        return self.forward_metadata.indexer_metadata

    def set_mtp_shared_sparse_indices(self, state) -> None:
        self._mtp_shared_sparse_indices = state

    def should_reuse_mtp_sparse_indices(self, forward_batch) -> bool:
        """Draft decode steps reuse the draft-extend selection."""
        return (
            self._mtp_shared_sparse_indices is not None
            and forward_batch.forward_mode.is_decode()
        )

    def should_capture_mtp_sparse_indices(self, forward_batch) -> bool:
        """Capture on both draft-extend flavors (a DP MAX_LEN rewrite is neither):
        DRAFT_EXTEND_V2, and the draft runner's plain post-prefill EXTEND."""
        if self._mtp_shared_sparse_indices is None:
            return False
        if forward_batch._original_forward_mode is not None:
            return False
        mode = forward_batch.forward_mode
        return mode.is_draft_extend_v2() or mode.is_extend_without_speculative()

    def capture_mtp_sparse_indices(
        self, topk_indices: torch.Tensor, forward_batch, layer_id: int, metadata=None
    ) -> None:
        """Store each request's final accepted row as the iteration seed."""
        if topk_indices.shape[0] == 0:
            return
        if metadata is None:
            metadata = self.get_indexer_metadata(layer_id, forward_batch)
        if metadata.is_cuda_graph or forward_batch.forward_mode.is_draft_extend_v2():
            self._capture_mtp_sparse_indices_from_extend_lens(
                topk_indices, forward_batch, metadata, layer_id
            )
            return
        if metadata.req_pool_indices is None or metadata.req_pool_indices.numel() == 0:
            return
        state = self._mtp_shared_sparse_indices
        row_to_req = metadata.get_token_to_batch_idx().long()
        row_req_pool_indices = metadata.req_pool_indices[
            row_to_req[: topk_indices.shape[0]]
        ]
        is_last = torch.ones_like(row_req_pool_indices, dtype=torch.bool)
        if row_req_pool_indices.numel() > 1:
            is_last[:-1] = row_req_pool_indices[:-1] != row_req_pool_indices[1:]
        anchor_rows = is_last.nonzero().flatten()
        req_rows = row_req_pool_indices[anchor_rows]
        captured_lens = metadata.get_seqlens_expanded()[anchor_rows]
        state.capture(topk_indices[anchor_rows], req_rows, captured_lens, layer_id)

    @staticmethod
    def _capture_extend_seq_lens(forward_batch) -> torch.Tensor:
        """Same resolution order as _stage_extend_lens,
        so the capture and in-graph layout read the same replay-refreshed buffer."""
        spec_info = getattr(forward_batch, "spec_info", None)
        extend_seq_lens = getattr(spec_info, "extend_seq_lens_tensor", None)
        if extend_seq_lens is None:
            extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        if extend_seq_lens is None:
            raise RuntimeError(
                "QSA draft-extend capture requires GPU extend lengths "
                "(spec_info.extend_seq_lens_tensor or "
                "forward_batch.extend_seq_lens)"
            )
        return extend_seq_lens

    def _capture_mtp_sparse_indices_from_extend_lens(
        self, topk_indices, forward_batch, metadata, layer_id: int
    ) -> None:
        """DRAFT_EXTEND_V2 packs each request as [front rows][draft-window rows],
        so its last accepted row is block_end - (window - front - num_accept);
        rows with no real request (zero extend, padding slot 0) go to the trash row."""

        state = self._mtp_shared_sparse_indices
        bs = int(forward_batch.batch_size)
        if bs == 0 or forward_batch.req_pool_indices.numel() < bs:
            return
        request_extend_lens = self._capture_extend_seq_lens(forward_batch)[:bs].long()
        block_ends = request_extend_lens.cumsum(0) - 1
        spec_info = forward_batch.spec_info
        num_accept_tokens = getattr(spec_info, "num_accept_tokens", None)
        if num_accept_tokens is not None and num_accept_tokens.numel() >= bs:
            # Only EagleDraftExtendInput carries accept counts;
            # prefill-side EXTEND has no draft window, so its rows end at the seed.
            front_tokens = int(spec_info.num_front_tokens)
            anchor_rows = block_ends - (
                request_extend_lens - front_tokens - num_accept_tokens[:bs].long()
            )
        else:
            anchor_rows = block_ends
        anchor_rows = anchor_rows.clamp(min=0, max=topk_indices.shape[0] - 1)
        req_pool_rows = forward_batch.req_pool_indices[:bs].long()
        req_rows = torch.where(
            (request_extend_lens > 0) & (req_pool_rows > 0),
            req_pool_rows,
            state.trash_row,
        )
        captured_lens = metadata.get_seqlens_expanded()[anchor_rows]
        state.capture(
            topk_indices.index_select(0, anchor_rows),
            req_rows,
            captured_lens,
            layer_id,
        )

    def lookup_mtp_sparse_indices(self, forward_batch, layer_id: int) -> torch.Tensor:
        metadata = self.get_indexer_metadata(layer_id, forward_batch)
        logical_positions = metadata.decode_logical_positions
        if logical_positions is None:
            logical_positions = metadata.get_seqlens_expanded() - 1
        return self._mtp_shared_sparse_indices.lookup(
            forward_batch.req_pool_indices,
            logical_positions,
            layer_id,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def _resolve_metadata(self, forward_batch) -> QwenSparseAttnMetadata:
        if self.forward_metadata is None:
            self.init_forward_metadata(forward_batch)
        assert self.forward_metadata is not None
        return self.forward_metadata

    @staticmethod
    def _logical_to_physical(
        logical_indices: torch.Tensor, metadata: QwenSparseAttnMetadata
    ) -> torch.Tensor:
        sequence_ids = metadata.token_to_batch_idx.long()
        if sequence_ids.numel() != logical_indices.shape[0]:
            raise ValueError("QSA top-k rows do not match query rows")
        row_lengths = metadata.sequence_lengths.to(torch.int32).index_select(
            0, sequence_ids
        )
        valid = (logical_indices >= 0) & (logical_indices < row_lengths.unsqueeze(1))
        safe = logical_indices.clamp(
            min=0, max=metadata.token_slot_table.shape[1] - 1
        ).long()
        slots = metadata.token_slot_table[sequence_ids[:, None], safe]
        return torch.where(valid, slots, torch.full_like(slots, -1)).to(torch.int32)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch,
        save_kv_cache: bool = True,
        topk_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        mode = _qsa_dense_check_mode()
        if not mode or topk_indices is None:
            return self._forward_extend_impl(
                q, k, v, layer, forward_batch, save_kv_cache, topk_indices, **kwargs
            )
        rows = int(topk_indices.shape[0])
        q3 = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
        eligible = _qsa_dense_check_eligible(forward_batch, rows, k) and q3.shape[0] >= rows
        ref = ref_global = None
        if eligible:
            ref = _qsa_dense_reference(q3[:rows], k, v, layer.scaling)
            kv_map = self._qsa_global_kv_map_for_layer(layer, k.shape[1])
            if kv_map is not None:
                ref_global = _qsa_dense_reference(
                    q3[:rows], k, v, layer.scaling, kv_map=kv_map
                )
        output = self._forward_extend_impl(
            q, k, v, layer, forward_batch, save_kv_cache, topk_indices, **kwargs
        )
        if ref is None:
            return output
        self._qsa_dense_check_log(
            mode, layer, forward_batch, output, ref, rows, ref_global=ref_global
        )
        chosen = ref_global if mode == "subst_global" else ref if mode == "subst" else None
        if chosen is not None and output.shape[0] >= rows:
            output = output.clone()
            output[:rows] = chosen.reshape(rows, -1).to(output.dtype)
        return output

    def _rank_local_kv_head(self, layer) -> Optional[int]:
        """kv head this rank's q heads attend to when the layer holds ALL kv
        heads under REPLICATED-KV (attn_replicated_kv_local_head), else None
        (single-head layer, no plan, or the geometry does not apply)."""
        cached = getattr(layer, "_qsa_rank_local_kv_head", "unset")
        if cached != "unset":
            return cached
        head = None
        try:
            from sglang.srt.distributed.utils import attn_replicated_kv_local_head
            from sglang.srt.runtime_context import get_parallel

            cfg = self.dcp_model_config
            total_kv = int(cfg.get_total_num_kv_heads()) if cfg is not None else 0
            if cfg is not None and int(layer.tp_k_head_num) == total_kv and total_kv > 1:
                parallel = get_parallel()
                total_q = int(cfg.hf_text_config.num_attention_heads)
                head = attn_replicated_kv_local_head(
                    total_q, total_kv, int(parallel.attn_tp_size), int(parallel.attn_tp_rank)
                )
        except Exception as exc:
            logger.warning("QSA rank-local kv head unresolved (all heads kept): %r", exc)
            head = None
        layer._qsa_rank_local_kv_head = head
        return head

    def _rank_local_kv_inputs(self, layer, k: torch.Tensor, v: torch.Tensor):
        """fn5m 19.09.: the prefix-free extend attends rank-locally with the
        forward's own k/v; under REPLICATED-KV only this rank's kv head may
        feed the kernel (its local grouping pairs the wrong head otherwise)."""
        head = self._rank_local_kv_head(layer)
        if head is None or k.dim() != 3 or k.shape[1] <= 1:
            return k, v
        return k.narrow(1, head, 1), v.narrow(1, head, 1)

    def _qsa_global_kv_map_for_layer(self, layer, total_kv_heads: int):
        """[hq_local] kv index per local q head under the GLOBAL grouping, or
        None when no shard plan is installed (then local == global)."""
        try:
            from sglang.srt.distributed.utils import tp_plan_active
            from sglang.srt.layers.attention.triton_backend import (
                _plan_aware_dcp_group_q_head_counts,
            )
            from sglang.srt.runtime_context import get_parallel

            parallel = get_parallel()
            tp_size = int(parallel.attn_tp_size)
            if tp_size <= 1 or not tp_plan_active(tp_size):
                return None
            rank = int(parallel.attn_tp_rank)
            hq = int(layer.tp_q_head_num)
            counts = _plan_aware_dcp_group_q_head_counts(
                self.dcp_model_config, tp_size, hq
            )
            if len(counts) != tp_size or int(counts[rank]) != hq:
                return None
            return _qsa_global_kv_map(
                sum(int(c) for c in counts[:rank]), hq, sum(int(c) for c in counts), total_kv_heads
            )
        except Exception as exc:  # diagnostics never kill the forward
            logger.warning("QSA-DENSE-CHECK global map unavailable: %r", exc)
            return None

    def _qsa_dense_check_log(
        self, mode, layer, forward_batch, output, ref, rows, ref_global=None
    ):
        if _QSA_DENSE_CHECK_LOGS[0] >= 240:
            return
        _QSA_DENSE_CHECK_LOGS[0] += 1
        try:
            a = output[:rows].float().reshape(rows, -1)
            b = ref.float().reshape(rows, -1)
            diff = (a - b).abs()
            cos = F.cosine_similarity(a, b, dim=-1)
            gtxt = ""
            if ref_global is not None:
                c = ref_global.float().reshape(rows, -1)
                dg = (a - c).abs()
                cg = F.cosine_similarity(a, c, dim=-1)
                gtxt = (
                    f" GLOBAL: max_abs={float(dg.max()):.4g} mean_abs={float(dg.mean()):.4g}"
                    f" cos_mean={float(cg.mean()):.4f} cos_min={float(cg.min()):.4f}"
                    f" local_vs_global_max_abs={float((b - c).abs().max()):.4g}"
                )
            pool = self.token_to_kv_pool
            loc = getattr(forward_batch, "out_cache_loc", None)
            loc_min = int(loc.min()) if loc is not None and loc.numel() else -1
            loc_max = int(loc.max()) if loc is not None and loc.numel() else -1
            logger.warning(
                "QSA-DENSE-CHECK mode=%s draft=%s layer=%d fmode=%s rows=%d hq=%d hkv=%d "
                "max_abs=%.4g mean_abs=%.4g ref_mean_abs=%.4g cos_mean=%.4f cos_min=%.4f "
                "cos_first8=%s pool_size=%s loc_min=%d loc_max=%d dcp=%d%s",
                mode,
                self.is_draft_worker,
                int(layer.layer_id),
                getattr(forward_batch.forward_mode, "name", str(forward_batch.forward_mode)),
                rows,
                int(layer.tp_q_head_num),
                int(layer.tp_k_head_num),
                float(diff.max()),
                float(diff.mean()),
                float(b.abs().mean()),
                float(cos.mean()),
                float(cos.min()),
                [round(float(x), 3) for x in cos[:8].tolist()],
                getattr(pool, "size", None),
                loc_min,
                loc_max,
                int(self.dcp_size),
                gtxt,
            )
        except Exception as exc:  # diagnostics never kill the forward
            logger.warning("QSA-DENSE-CHECK skipped: %r", exc)

    def _forward_extend_impl(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch,
        save_kv_cache: bool = True,
        topk_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if topk_indices is None:
            raise ValueError("QSA sparse attention requires topk_indices")
        if save_kv_cache:
            self._set_kv_buffer(forward_batch, layer, k, v)
        q = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
        num_output_rows = q.shape[0]
        num_valid_rows = topk_indices.shape[0]
        if num_valid_rows > num_output_rows:
            raise ValueError(
                "QSA top-k rows exceed query rows: "
                f"topk={num_valid_rows}, query={num_output_rows}"
            )
        # DP attention may pad q beyond the indexer's query rows.
        # The kernels see only valid rows; the output is zero-padded to q's row count.
        q = q[:num_valid_rows]
        if self._is_speculative_paged_mode(forward_batch.forward_mode):
            output = self._forward_paged_attention(
                q, layer, forward_batch, topk_indices
            )
            return self._pad_extend_output(output, num_output_rows)
        if not q.is_cuda:
            metadata = self._resolve_metadata(forward_batch)
            slots = self._logical_to_physical(topk_indices, metadata)
            pool = self.token_to_kv_pool
            output = qsa_sparse_attention(
                q,
                pool.get_key_buffer(layer.layer_id),
                pool.get_value_buffer(layer.layer_id),
                slots,
                layer.scaling,
            )
            return self._pad_extend_output(output, num_output_rows)

        topk_indices = topk_indices.to(torch.int32).contiguous()
        extend_lens = [int(x) for x in forward_batch.extend_seq_lens_cpu]
        sequence_lens = [int(x) for x in forward_batch.seq_lens_cpu]
        prefix_lens = [
            sequence_lens[i] - extend_lens[i] for i in range(len(extend_lens))
        ]
        cu_seqlens_q = F.pad(
            forward_batch.extend_seq_lens.to(q.device, dtype=torch.int32).cumsum(0),
            (1, 0),
        ).contiguous()
        if not any(prefix_lens):
            k_loc, v_loc = self._rank_local_kv_inputs(
                layer, k[:num_valid_rows], v[:num_valid_rows]
            )
            output = sparse_gqa_fwd_interface_triton(
                q.contiguous(),
                k_loc.contiguous(),
                v_loc.contiguous(),
                max(sequence_lens, default=1),
                topk_indices,
                cu_seqlens_q,
                layer.scaling,
            )
            return self._pad_extend_output(output, num_output_rows)

        if self.dcp_size > 1 or (_qsa_rows_path_armed() and q.is_cuda):
            # WP3b: the prefix rows live on their owner ranks; attend the owned
            # subset on the gathered heads and LSE-merge (see _attend_rows).
            # fn5e: also the non-DCP branch -- see _qsa_rows_path_armed.
            metadata = self._resolve_metadata(forward_batch)
            rows, counts = self._rows_and_counts(topk_indices, metadata)
            output = self._attend_rows(q, layer, rows, counts)
            return self._pad_extend_output(output, num_output_rows)

        # The validated chunk-prefill kernel consumes tightly packed full-context
        # K/V. Current-chunk K/V has already been committed to the cache above.
        pool = self.token_to_kv_pool
        k_buffer = pool.get_key_buffer(layer.layer_id)
        v_buffer = pool.get_value_buffer(layer.layer_id)
        req_to_token = self.req_to_token_pool.req_to_token
        req_indices = forward_batch.req_pool_indices.tolist()
        k_parts = [
            k_buffer.index_select(
                0, req_to_token[req_indices[i], : sequence_lens[i]].long()
            )
            for i in range(len(sequence_lens))
        ]
        v_parts = [
            v_buffer.index_select(
                0, req_to_token[req_indices[i], : sequence_lens[i]].long()
            )
            for i in range(len(sequence_lens))
        ]
        sequence_lens_tensor = torch.tensor(
            sequence_lens, dtype=torch.int32, device=q.device
        )
        cu_seqlens_k = F.pad(sequence_lens_tensor.cumsum(0), (1, 0)).contiguous()
        output = sparse_gqa_fwd_interface_triton_ck(
            q.contiguous(),
            torch.cat(k_parts),
            torch.cat(v_parts),
            topk_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            sequence_lens_tensor,
            layer.scaling,
        )
        return self._pad_extend_output(output, num_output_rows)

    @staticmethod
    def _pad_extend_output(output: torch.Tensor, num_rows: int) -> torch.Tensor:
        output = output.reshape(output.shape[0], -1)
        if output.shape[0] == num_rows:
            return output
        if output.shape[0] > num_rows:
            raise ValueError(
                "QSA attention output rows exceed padded query rows: "
                f"output={output.shape[0]}, query={num_rows}"
            )
        padded = output.new_zeros((num_rows, output.shape[1]))
        padded[: output.shape[0]].copy_(output)
        return padded

    def _get_fa2_scratch(
        self,
        capacity: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (num_kv_heads, head_dim, dtype, device)
        buffers = self._fa2_scratch.get(key)
        if buffers is None or buffers[0].shape[0] < capacity:
            shape = (capacity, num_kv_heads, head_dim)
            buffers = (
                torch.empty(shape, dtype=dtype, device=device),
                torch.empty(shape, dtype=dtype, device=device),
            )
            self._fa2_scratch[key] = buffers
        return buffers[0][:capacity], buffers[1][:capacity]

    def _get_trtllm_sparse_tables(self, batch, pages_per_row, page, device):
        key = (batch, pages_per_row, device)
        cached = self._trtllm_sparse_tables.get(key)
        if cached is None:
            stride = pages_per_row * page
            cu = torch.arange(batch + 1, dtype=torch.int32, device=device) * stride
            block_tables = (
                torch.arange(batch, dtype=torch.int32, device=device)[:, None]
                * pages_per_row
                + torch.arange(pages_per_row, dtype=torch.int32, device=device)[None, :]
            ).contiguous()
            cached = (cu, block_tables)
            self._trtllm_sparse_tables[key] = cached
        return cached

    def _forward_trtllm_sparse(
        self,
        q: torch.Tensor,
        k_buffer: torch.Tensor,
        v_buffer: torch.Tensor,
        layer,
        forward_batch,
        metadata,
        topk_indices: torch.Tensor,
        trtllm_decode,
    ) -> torch.Tensor:
        """Pack selected KV at page-aligned row strides for FlashInfer's paged decode,
        driven by a static arange block table and the per-row valid counts."""
        batch, topk = topk_indices.shape
        page = _TRTLLM_SPARSE_PAGE_SIZE
        pages_per_row = (topk + page - 1) // page
        stride = pages_per_row * page
        device = q.device
        sequence_lens = metadata.sequence_lengths
        if metadata.is_cuda_graph:
            valid_counts = metadata.fa2_valid_counts
            if valid_counts is None:
                raise RuntimeError("QSA CUDA graph metadata is incomplete")
        else:
            valid_counts = torch.empty(batch, dtype=torch.int32, device=device)
        qwen_sparse_valid_counts_triton(
            sequence_lens, topk_indices, valid_counts, batch, topk
        )
        cu_strided, block_tables = self._get_trtllm_sparse_tables(
            batch, pages_per_row, page, device
        )
        capacity_rows = self._cuda_graph_max_tokens if metadata.is_cuda_graph else batch
        # Gather into the query dtype: an FP8 pool is dequantized on the way in, so the
        # paged kernel always runs the bf16 q + bf16 KV path.
        packed_k, packed_v = self._get_fa2_scratch(
            max(capacity_rows, batch) * stride,
            k_buffer.shape[1],
            k_buffer.shape[2],
            q.dtype,
            k_buffer.device,
        )
        qwen_sparse_kv_extraction_compact_triton(
            k_buffer,
            v_buffer,
            self.req_to_token_pool.req_to_token,
            (
                metadata.row_req_pool_indices
                if metadata.row_req_pool_indices is not None
                else forward_batch.req_pool_indices
            ),
            topk_indices,
            sequence_lens,
            cu_strided,
            packed_k,
            packed_v,
            batch,
            topk,
            zero_fill_cols=stride,
        )
        num_kv_heads = k_buffer.shape[1]
        head_dim = k_buffer.shape[2]
        kc = (
            packed_k[: batch * stride]
            .view(-1, page, num_kv_heads, head_dim)
            .permute(0, 2, 1, 3)
        )
        vc = (
            packed_v[: batch * stride]
            .view(-1, page, num_kv_heads, head_dim)
            .permute(0, 2, 1, 3)
        )
        if self._trtllm_workspace is None:
            self._trtllm_workspace = torch.zeros(
                128 * 1024 * 1024, dtype=torch.uint8, device=device
            )
        output = trtllm_decode(
            query=q.contiguous(),
            kv_cache=(kc, vc),
            workspace_buffer=self._trtllm_workspace,
            block_tables=block_tables,
            seq_lens=valid_counts,
            max_seq_len=stride,
            bmm1_scale=layer.scaling,
            bmm2_scale=1.0,
        )
        return output.reshape(q.shape[0], -1)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch,
        save_kv_cache: bool = True,
        topk_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if topk_indices is None:
            raise ValueError("QSA sparse attention requires topk_indices")
        if save_kv_cache:
            self._set_kv_buffer(forward_batch, layer, k, v)
        q = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
        if self.dcp_size <= 1 and _qsa_rows_path_armed() and q.is_cuda:
            # fn5e: rows kernel on every rank (see _qsa_rows_path_armed)
            metadata = self._resolve_metadata(forward_batch)
            rows, counts = self._rows_and_counts(topk_indices, metadata)
            return self._attend_rows(q, layer, rows, counts).reshape(q.shape[0], -1)
        return self._forward_paged_attention(q, layer, forward_batch, topk_indices)

    def _forward_paged_attention(
        self,
        q: torch.Tensor,
        layer,
        forward_batch,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        pool = self.token_to_kv_pool
        k_buffer = pool.get_key_buffer(layer.layer_id)
        v_buffer = pool.get_value_buffer(layer.layer_id)
        if not q.is_cuda:
            metadata = self._resolve_metadata(forward_batch)
            slots = self._logical_to_physical(topk_indices, metadata)
            output = qsa_sparse_attention(q, k_buffer, v_buffer, slots, layer.scaling)
            return output.reshape(q.shape[0], -1)

        metadata = self._resolve_metadata(forward_batch)
        topk_indices = topk_indices.to(torch.int32).contiguous()
        trtllm_decode = _resolve_trtllm_sparse_decode() if self.dcp_size <= 1 else None
        if trtllm_decode is None and _speculative_rows_route(
            self.dcp_size,
            _qsa_rows_path_armed(),
            q.is_cuda,
            _resolve_flash_attn_varlen_func_or_none() is not None,
        ):
            # WP3b rows path: the Triton rows kernel over the pool (owned rows
            # under DCP, merged across the group); also the only decode route
            # on a host without FA2/FA4 (this rig's serving venv has neither).
            # fn4o 19.09.: ARMED here like the decode path (fn5e) -- the draft
            # backend runs dcp_size 1 and the packed varlen fallback is FA4
            # cute, whose MLIR refuses sm86 ('Operation creation failed',
            # pack_gqa) in the target-verify / draft-extend capture.
            rows, counts = self._rows_and_counts(topk_indices, metadata)
            output = self._attend_rows(q, layer, rows, counts)
            return output.reshape(q.shape[0], -1)
        if trtllm_decode is not None:
            return self._forward_trtllm_sparse(
                q,
                k_buffer,
                v_buffer,
                layer,
                forward_batch,
                metadata,
                topk_indices,
                trtllm_decode,
            )

        flash_attn_varlen_func = _resolve_flash_attn_varlen_func()
        batch, topk = topk_indices.shape
        sequence_lens = metadata.sequence_lengths
        if metadata.is_cuda_graph:
            valid_counts = metadata.fa2_valid_counts
            cu_seqlens_k = metadata.fa2_cu_seqlens_k
            cu_seqlens_q = metadata.fa2_cu_seqlens_q
            if valid_counts is None or cu_seqlens_k is None or cu_seqlens_q is None:
                raise RuntimeError("QSA CUDA graph FA2 metadata is incomplete")
        else:
            valid_counts = torch.empty(batch, dtype=torch.int32, device=q.device)
            cu_seqlens_k = torch.empty(batch + 1, dtype=torch.int32, device=q.device)
            cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device=q.device)
        qwen_sparse_fa2_cu_seqlens_triton(
            sequence_lens,
            topk_indices,
            valid_counts,
            cu_seqlens_k,
            batch,
            topk,
        )
        scratch_capacity = (
            self._cuda_graph_max_tokens * topk
            if metadata.is_cuda_graph
            else batch * topk
        )
        packed_k, packed_v = self._get_fa2_scratch(
            scratch_capacity,
            k_buffer.shape[1],
            k_buffer.shape[2],
            q.dtype,
            k_buffer.device,
        )
        qwen_sparse_kv_extraction_compact_triton(
            k_buffer,
            v_buffer,
            self.req_to_token_pool.req_to_token,
            (
                metadata.row_req_pool_indices
                if metadata.row_req_pool_indices is not None
                else forward_batch.req_pool_indices
            ),
            topk_indices,
            sequence_lens,
            cu_seqlens_k,
            packed_k,
            packed_v,
            batch,
            topk,
        )
        output = flash_attn_varlen_func(
            q=q,
            k=packed_k,
            v=packed_v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=1,
            max_seqlen_k=topk,
            softmax_scale=layer.scaling,
            causal=True,
        )
        return output.reshape(q.shape[0], -1)


class QwenSparseMultiStepDraftBackend:
    """Per-step QSA metadata for consecutive Qwen4-Exp MTP draft decoding."""

    needs_cpu_seq_lens: bool = False

    def __init__(self, model_runner, topk: int, speculative_num_steps: int):
        if topk != 1:
            raise NotImplementedError(
                "Qwen4-Exp QSA MTP currently supports speculative_eagle_topk=1"
            )
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends = [
            QwenSparseAttnBackend(model_runner)
            for _ in range(speculative_num_steps - 1)
        ]

    @staticmethod
    def _as_cpu_lengths(seq_lens_cpu, seq_lens: torch.Tensor) -> torch.Tensor:
        if seq_lens_cpu is None:
            return seq_lens.detach().cpu().to(torch.int32)
        if isinstance(seq_lens_cpu, torch.Tensor):
            return seq_lens_cpu.to(dtype=torch.int32)
        return torch.tensor(seq_lens_cpu, dtype=torch.int32)

    def _step_out_cache_loc(self, forward_batch, step: int):
        """Mirror EagleDraftWorker.draft_forward's per-step out_cache_loc slice;
        a mismatch makes every QSA step write into the first ``batch_size`` slots."""

        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        steps = self.speculative_num_steps
        batch_rows = int(
            getattr(
                forward_batch,
                "batch_size",
                forward_batch.seq_lens.numel(),
            )
        )
        if out_cache_loc is None or steps <= 1:
            return out_cache_loc
        if out_cache_loc.numel() != batch_rows * self.topk * steps:
            # Idle/zero-row and non-draft batches do not use the interleaved layout.
            return out_cache_loc
        # Same expression chain as EagleDraftWorker.draft_forward (views only).
        return (
            out_cache_loc.reshape(batch_rows, self.topk, steps)
            .permute(2, 0, 1)
            .reshape(steps, -1)[step]
        )

    def _make_step_forward_batch(self, forward_batch, step: int, num_padding: int = 0):
        step_forward_batch = copy(forward_batch)
        step_forward_batch.forward_mode = ForwardMode.DECODE
        step_forward_batch.seq_lens = (forward_batch.seq_lens + step + 1).to(
            torch.int32
        )
        if forward_batch.seq_lens_cpu is None:
            # GPU-only serving: downstream metadata paths derive host bounds
            # from batch shape; do not force a per-step D2H here.
            step_forward_batch.seq_lens_cpu = None
        else:
            step_forward_batch.seq_lens_cpu = self._as_cpu_lengths(
                forward_batch.seq_lens_cpu, forward_batch.seq_lens
            ) + (step + 1)
        num_padding = max(
            0,
            min(int(num_padding), int(step_forward_batch.seq_lens.numel())),
        )
        if num_padding:
            step_forward_batch.seq_lens = step_forward_batch.seq_lens.clone()
            step_forward_batch.seq_lens[-num_padding:] = 1
            if step_forward_batch.seq_lens_cpu is not None:
                step_forward_batch.seq_lens_cpu = (
                    step_forward_batch.seq_lens_cpu.clone()
                )
                step_forward_batch.seq_lens_cpu[-num_padding:] = 1
        step_forward_batch.batch_size = int(step_forward_batch.seq_lens.numel())
        step_forward_batch.out_cache_loc = self._step_out_cache_loc(forward_batch, step)
        return step_forward_batch

    def set_mtp_shared_sparse_indices(self, state) -> None:
        for backend in self.attn_backends:
            backend.set_mtp_shared_sparse_indices(state)

    def init_forward_metadata(self, forward_batch):
        for step, backend in enumerate(self.attn_backends):
            backend.init_forward_metadata(
                self._make_step_forward_batch(forward_batch, step)
            )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for backend in self.attn_backends:
            backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False):
        if in_capture:
            for step, backend in enumerate(self.attn_backends):
                # Warmup must not write via shared req_pool_idx 0;
                # pad every capture row to stay below the first compression boundary.
                step_batch = self._make_step_forward_batch(
                    forward_batch,
                    step,
                    num_padding=forward_batch.batch_size,
                )
                backend._capture_cuda_graph_metadata(
                    bs=step_batch.batch_size,
                    num_tokens=step_batch.batch_size,
                    req_pool_indices=step_batch.req_pool_indices,
                    seq_lens=step_batch.seq_lens,
                    forward_mode=ForwardMode.DECODE,
                    spec_info=step_batch.spec_info,
                )
            return

        num_padding = getattr(forward_batch, "num_padding", None)
        num_padding = num_padding if num_padding is not None else 0
        for step, backend in enumerate(self.attn_backends):
            step_batch = self._make_step_forward_batch(
                forward_batch, step, num_padding=num_padding
            )
            backend._replay_cuda_graph_metadata(
                bs=step_batch.batch_size,
                req_pool_indices=step_batch.req_pool_indices,
                seq_lens=step_batch.seq_lens,
                forward_mode=ForwardMode.DECODE,
                spec_info=step_batch.spec_info,
                seq_lens_cpu=step_batch.seq_lens_cpu,
                num_padding=num_padding,
            )

    def init_forward_metadata_in_graph(self, forward_batch) -> None:
        pass


__all__ = [
    "is_qwen_qsa",
    "QwenSparseAttnMetadata",
    "QwenSparseAttnBackend",
    "QwenSparseMultiStepDraftBackend",
]
