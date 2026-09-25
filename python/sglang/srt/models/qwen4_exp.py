"""Inference-only Qwen4-Exp (text + VL) on the Qwen3.5 backbone."""

import os
import re
import math
import contextlib
import functools
from contextlib import nullcontext
from typing import Any, Iterable, Optional, Set, Tuple

import msgspec
import sympy
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn

from sglang.srt.layers.elementwise import fused_sigmoid_mul
from sglang.srt.layers.fwd_timeline import (
    begin_if_timed,
    fwd_abort,
    fwd_end,
    fwd_mark,
    fwd_timing_on,
)
from sglang.srt.configs.qwen4_exp import Qwen4ExpConfig, Qwen4ExpTextConfig
from sglang.srt.distributed import get_tp_group, tensor_model_parallel_all_reduce
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather,
    attn_tp_all_reduce,
    dp_gather_replicate,
    dp_scatter,
    get_attention_dp_size,
    get_dp_global_num_tokens,
    get_global_dp_buffer,
    get_local_dp_buffer,
    is_allocation_symmetric,
    is_dp_attention_enabled,
)
from sglang.srt.layers.hyperconnection import (
    GatedResidual,
    HyperConnectionConfig,
)
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe import get_moe_a2a_backend, should_use_dp_reduce_scatterv
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.nan_guard import check as _nan_check
from sglang.srt.layers.nan_guard import nan_guard_on as _nan_guard_on
from sglang.srt.layers.ple_wait_span import ple_wait_scope, wait_for_ple_prefetch


def _nan_discriminate(layer, mlp_in, mlp_out, forward_batch) -> None:
    """Task #49 (19.09.): when a layer's MoE output stops being finite, say
    WHICH of three things happened, on the spot: (1) the MoE input was already
    non-finite (the fault is upstream: attention / residual / norm); (2) the
    same input recomputed through the same MoE is finite again (TRANSIENT: a
    fetch/compute race or a slot read in flight); (3) it is non-finite again
    (PERSISTENT: corrupted device slot bytes or a deterministic kernel fault)
    -- then the offload cache's float buffers (scales) are scanned for
    non-finite slots. The recompute runs the same collectives on every rank,
    and every rank sees the same post-all-reduce NaN, so the group stays in
    lockstep. fn8g run 2 (19.09.): layer 0, TP0's partial NaN in 1648 rows,
    finite again on layer 1 -- this instrument names why."""
    try:
        import logging

        import torch

        log = logging.getLogger(__name__)
        lid = getattr(layer, "layer_id", "?")
        if mlp_in is None:
            return
        in_bad = ~torch.isfinite(mlp_in)
        in_rows_t = torch.nonzero(in_bad.reshape(mlp_in.shape[0], -1).any(dim=1)).reshape(-1)
        out_bad = ~torch.isfinite(mlp_out)
        out_rows_t = torch.nonzero(out_bad.reshape(mlp_out.shape[0], -1).any(dim=1)).reshape(-1)
        if in_rows_t.numel():
            log.error(
                "[nan-guard] DISCRIMINATE layer %s: MoE INPUT already non-finite in %d rows "
                "(first %s) -> upstream of the experts",
                lid, int(in_rows_t.numel()), in_rows_t[:8].tolist(),
            )
            return

        # Task #49 (20.09.): the second discriminator. 'TRANSIENT' is where the
        # three-way verdict below stops; this group says WHICH of the five
        # candidate classes produced it. Everything that has to be read BEFORE
        # the recompute (the recompute re-resolves and RE-FETCHES the slot) is
        # collected here; the group is emitted as one record afterwards, so the
        # before/after fingerprints sit side by side instead of in two lines a
        # boot log would separate. First hit of the process only.
        disc2 = None
        try:
            from sglang.srt.layers.moe import nan_disc2

            if nan_disc2.disc2_on() and nan_disc2.arm_once():
                experts_mod, cache = nan_disc2.find_cache(layer)
                if cache is not None:
                    n_rows = int(mlp_out.shape[0])
                    bad_set = set(out_rows_t.tolist())
                    good = [r for r in range(n_rows) if r not in bad_set][:512]
                    snap = nan_disc2.snapshot(
                        experts_mod, cache, out_rows_t.tolist(), good
                    )
                    if snap is not None:
                        disc2 = (experts_mod, cache, snap)
                else:
                    log.error(
                        "[nan-disc2] layer %s: no expert-offload cache on the MoE block "
                        "-- nothing to fingerprint", lid,
                    )
        except Exception as exc:  # noqa: BLE001
            log.error("[nan-disc2] pre-recompute collection failed: %r", exc)

        again = layer.mlp(mlp_in.clone(), forward_batch)
        again_bad = ~torch.isfinite(again)
        again_rows_t = torch.nonzero(again_bad.reshape(again.shape[0], -1).any(dim=1)).reshape(-1)
        if again_rows_t.numel() == 0:
            verdict = "TRANSIENT (recompute finite)"
        elif again_rows_t.numel() == out_rows_t.numel() and bool((again_rows_t == out_rows_t).all().item()):
            verdict = "PERSISTENT same rows"
        else:
            verdict = "PERSISTENT different rows"
        log.error(
            "[nan-guard] DISCRIMINATE layer %s: input finite, output %d bad rows (first %s); "
            "recompute on the same input: %d bad rows (first %s) -> %s",
            lid, int(out_rows_t.numel()), out_rows_t[:8].tolist(),
            int(again_rows_t.numel()), again_rows_t[:8].tolist(), verdict,
        )
        if disc2 is not None:
            try:
                from sglang.srt.layers.moe import nan_disc2

                experts_mod, cache, snap = disc2
                snap = nan_disc2.finish(
                    experts_mod, cache, snap, int(again_rows_t.numel())
                )
                log.error("%s", nan_disc2.render(lid, snap))
            except Exception as exc:  # noqa: BLE001
                log.error("[nan-disc2] post-recompute report failed: %r", exc)
        seen = 0
        for obj in vars(getattr(layer.mlp, "experts", layer.mlp)).values():
            resident = getattr(obj, "_resident", None)
            if not isinstance(resident, dict):
                continue
            seen += 1
            for attr, buf in resident.items():
                if not torch.is_tensor(buf) or not buf.is_floating_point():
                    continue
                flat = buf.reshape(buf.shape[0], -1)
                bad_slots = torch.nonzero(~torch.isfinite(flat).all(dim=1)).reshape(-1)
                log.error(
                    "[nan-guard] DISCRIMINATE layer %s: device buffer %s (%d slots) -> %d non-finite slot(s) %s",
                    lid, attr, int(buf.shape[0]), int(bad_slots.numel()), bad_slots[:16].tolist(),
                )
        if not seen:
            log.error("[nan-guard] DISCRIMINATE layer %s: no offload cache on the MoE block", lid)
    except Exception as exc:  # noqa: BLE001
        import logging

        logging.getLogger(__name__).error("[nan-guard] DISCRIMINATE failed: %r", exc)
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.quantization.compressed_tensors.ct_embedding import (
    vocab_named_in_targets,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (
    dequantize_pack_quantized_weight,
)
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptMixedPrecisionConfig,
)
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_req_to_token_pool,
)
from sglang.srt.model_executor.runner import get_is_capture_mode
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_5 import (
    Qwen3_5AttentionDecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5GatedDeltaNet,
    Qwen3_5LinearDecoderLayer,
)
from sglang.srt.models.mtp_vocab_share import MtpEmbedDeferred
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.models.qwen4_exp_ple_table import (
    allocate_ple_host_table,
    make_ple_file_prefetcher,
    make_ple_checkpoint_prefetcher,
    make_ple_checkpoint_pread_gather,
    make_ple_file_rss_trimmer,
)
from sglang.srt.models.qwen4_exp_ple_prefetch import (
    PleHashParams,
    make_ple_prefetch_gather,
    ple_next_chunk_hasher,
)
from sglang.srt.models.qwen4_exp_ple_decode_pread import make_ple_decode_stager
from sglang.srt.models.qwen4_exp_ple_fp8 import ple_fp8_bytes_to_bf16, ple_fp8_decode_arg
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix, logger

# Decode/verify-sized batches only: at prefill sizes both chains are compute
# bound and serializing them on one stream is faster than contending.
_QSA_INDEXER_OVERLAP_TOKEN_THRESHOLD = 1024


def _ple_table_is_fp8(
    config: Qwen4ExpTextConfig,
    quant_config: Optional[QuantizationConfig],
    prefix: str,
) -> bool:
    """fp8 PLE shards: declared by config, an fp8 checkpoint, or a ModelOpt
    MIXED_PRECISION entry for the ngram table (nvidia/*-Flash-Next-NVFP4)."""
    if config.ple_embedding_dtype == "float8_e4m3fn":
        return True
    if quant_config is None:
        return False
    if quant_config.get_name() == "fp8":
        return True
    if isinstance(quant_config, ModelOptMixedPrecisionConfig):
        return quant_config.resolve_quant_algo(prefix) == "FP8"
    return False


def _get_ple_forward_mode(forward_batch: ForwardBatch) -> ForwardMode:
    if forward_batch._original_forward_mode is not None:
        return forward_batch._original_forward_mode
    return forward_batch.forward_mode


def _get_processed_token_count(
    forward_batch: ForwardBatch, physical_tokens: int
) -> int:
    # Upstream keeps a DP-global non-padded count on the batch; this line
    # carries the per-rank one (attention DP is 1 here), and neither is set
    # on a forward without padding (fn1p, 2026-09-16: the first forward
    # after 'ready' died on the missing attribute). The chain falls through
    # to the extend lengths and finally to the physical count.
    processed_tokens = getattr(forward_batch, "global_num_token_non_padded_cpu", None)
    if processed_tokens is None:
        processed_tokens = getattr(forward_batch, "num_token_non_padded_cpu", None)
    if processed_tokens is None and forward_batch.extend_seq_lens_cpu is not None:
        processed_tokens = sum(forward_batch.extend_seq_lens_cpu)
    if processed_tokens is None:
        return physical_tokens
    processed_tokens = int(processed_tokens)
    if not 0 <= processed_tokens <= physical_tokens:
        raise RuntimeError(
            f"invalid PLE token counts: {processed_tokens=}, {physical_tokens=}"
        )
    return processed_tokens


class _PLEBatch(msgspec.Struct, frozen=True):
    mode: ForwardMode
    use_decode_fast_path: bool
    physical_tokens: int
    processed_tokens: int
    lengths: torch.Tensor
    row_width: int
    req_indices: torch.Tensor
    token_offsets: torch.Tensor
    valid_tokens: torch.Tensor
    state_indices: torch.Tensor
    ngram_context: Optional[torch.Tensor]
    ngram_eos_token_id: Optional[int]


def _prepare_ple_batch(
    input_ids: torch.Tensor,
    forward_batch: ForwardBatch,
    *,
    ngram_size: Optional[int],
    ngram_eos_token_id: Optional[int],
) -> Optional[_PLEBatch]:
    """Prepare the token layout and the shared N-gram history once per forward."""

    if forward_batch.tbo_parent_token_range is not None:
        raise NotImplementedError("Qwen4 PLE is not compatible with two-batch overlap")
    spec_algorithm = forward_batch.spec_algorithm
    if spec_algorithm is not None and spec_algorithm.is_ngram():
        raise NotImplementedError("Qwen4 PLE does not support NGRAM speculation")
    if (
        forward_batch.spec_info is not None
        and getattr(forward_batch.spec_info, "topk", 1) != 1
    ):
        raise NotImplementedError("Qwen4 PLE speculative decoding supports only topk=1")

    mode = _get_ple_forward_mode(forward_batch)
    get_req_to_token_pool().ple_window_cache = None
    if mode.is_idle():
        return None
    use_decode_fast_path = (
        envs.SGLANG_ENABLE_QWEN4_PLE_FUSION.get() and mode.is_decode()
    )

    if input_ids.dim() > 1:
        input_ids = input_ids.reshape(-1)
    physical_tokens = input_ids.shape[0]
    processed_tokens = _get_processed_token_count(forward_batch, physical_tokens)
    tokens = input_ids[:processed_tokens]
    positions = torch.arange(processed_tokens, device=tokens.device, dtype=torch.long)

    if mode.is_target_verify():
        assert forward_batch.spec_info is not None
        row_width = int(forward_batch.spec_info.draft_token_num)
        if row_width <= 0 or processed_tokens % row_width != 0:
            raise RuntimeError(
                "target verify rows must contain complete draft strides: "
                f"{processed_tokens=} {row_width=}"
            )
        sequence_count = processed_tokens // row_width
        # Eager verify rows can be shorter than the row stride;
        # ignore the synthetic one-token lengths from DP attention's verify-as-EXTEND.
        lengths = (
            forward_batch.extend_seq_lens[:sequence_count].long()
            if forward_batch.forward_mode.is_target_verify()
            and forward_batch.extend_seq_lens is not None
            else torch.full(
                (sequence_count,),
                row_width,
                dtype=torch.long,
                device=tokens.device,
            )
        )
        if lengths.shape[0] != sequence_count:
            raise RuntimeError(
                "target verify length metadata does not match its fixed rows: "
                f"{lengths.shape[0]=} {sequence_count=}"
            )
        req_indices = torch.div(positions, row_width, rounding_mode="floor")
        token_offsets = positions - req_indices * row_width
    elif mode.is_decode():
        lengths = torch.ones(processed_tokens, dtype=torch.long, device=tokens.device)
        row_width = 1
        req_indices = positions
        token_offsets = torch.zeros_like(positions)
    else:
        if forward_batch.extend_seq_lens is None:
            raise RuntimeError(f"PLE requires sequence lengths in {mode!r}")
        lengths = forward_batch.extend_seq_lens.long()
        extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
        row_width = (
            max(extend_seq_lens_cpu, default=0)
            if extend_seq_lens_cpu is not None
            else processed_tokens
        )
        query_start_loc = torch.cat(
            [lengths.new_zeros(1), torch.cumsum(lengths, dim=0)]
        )
        sequence_count = lengths.shape[0]
        req_indices = torch.searchsorted(query_start_loc, positions, right=True) - 1
        if processed_tokens:
            req_indices = req_indices.clamp(min=0, max=sequence_count - 1)
        token_offsets = positions - query_start_loc.index_select(0, req_indices)

    sequence_count = lengths.shape[0]
    if use_decode_fast_path:
        # One token per decode row: every offset is valid, no index-select needed.
        valid_tokens = torch.ones(
            processed_tokens, dtype=torch.bool, device=tokens.device
        )
    else:
        valid_tokens = token_offsets < lengths.index_select(0, req_indices)

    state_indices = (
        get_req_to_token_pool()
        .get_mamba_indices(forward_batch.req_pool_indices[:sequence_count])
        .long()
    )

    # CUDA graph padding uses request slot 0, which may belong to a real request.
    # Map padded sequences to the state pools' reserved dummy slot instead.
    out_cache_loc = forward_batch.out_cache_loc
    if use_decode_fast_path:
        if out_cache_loc is not None:
            state_indices = torch.where(
                out_cache_loc[:sequence_count].ne(0),
                state_indices,
                torch.zeros_like(state_indices),
            )
    else:
        valid = lengths.ne(0)
        if out_cache_loc is not None and mode.is_decode():
            valid = valid & out_cache_loc[:sequence_count].ne(0)
        elif out_cache_loc is not None and mode.is_target_verify():
            valid = valid & out_cache_loc[:processed_tokens].reshape(
                sequence_count, row_width
            ).ne(0).any(dim=1)
        state_indices = torch.where(
            valid, state_indices, torch.zeros_like(state_indices)
        )

    ngram_context = None
    if ngram_size is not None:
        assert ngram_eos_token_id is not None
        if use_decode_fast_path:
            # One token per decode row:
            # this view is the padded tensor the general path would materialize.
            padded = tokens.unsqueeze(1)
        else:
            padded = tokens.new_full((sequence_count, row_width), ngram_eos_token_id)
            if processed_tokens:
                padded[req_indices, token_offsets] = torch.where(
                    valid_tokens,
                    tokens,
                    tokens.new_full((), ngram_eos_token_id),
                )
        history = get_req_to_token_pool().get_ngram_context(state_indices)
        if history.shape[1] != ngram_size - 1:
            raise RuntimeError(
                "Qwen4 PLE N-gram cache has the wrong context width: "
                f"{history.shape[1]=} {ngram_size=}"
            )
        ngram_context = torch.cat([history, padded], dim=1)

    return _PLEBatch(
        mode=mode,
        use_decode_fast_path=use_decode_fast_path,
        physical_tokens=physical_tokens,
        processed_tokens=processed_tokens,
        lengths=lengths,
        row_width=row_width,
        req_indices=req_indices,
        token_offsets=token_offsets,
        valid_tokens=valid_tokens,
        state_indices=state_indices,
        ngram_context=ngram_context,
        ngram_eos_token_id=ngram_eos_token_id,
    )


def _commit_ple_batch(batch: Optional[_PLEBatch], forward_batch: ForwardBatch) -> None:
    """Commit the shared N-gram history after every PLE layer consumed it."""

    if batch is None or batch.ngram_context is None or not batch.processed_tokens:
        return

    pool = get_req_to_token_pool()
    context = batch.ngram_context
    context_len = context.shape[1] - batch.row_width
    if batch.mode.is_target_verify():
        step_contexts = context.unfold(1, context_len, 1)[:, 1:]
        valid_steps = batch.valid_tokens.reshape(
            batch.lengths.shape[0], batch.row_width
        )
        pool.set_ngram_intermediate_context(
            torch.where(
                valid_steps.unsqueeze(-1),
                step_contexts,
                torch.full_like(step_contexts, batch.ngram_eos_token_id),
            )
        )
        return

    if batch.use_decode_fast_path:
        # Decode advances every two-token history by exactly one column.  Slicing
        # preserves the int64 values while avoiding arange + gather launches.
        next_context = context[:, batch.row_width :]
        pool.set_ngram_context(batch.state_indices, next_context)
        track = _ple_track_targets(forward_batch, batch)
        if track is not None:
            track_indices, _ = track
            pool.set_ngram_context(track_indices, next_context)
        return

    context_cols = torch.arange(context_len, device=context.device, dtype=torch.long)
    next_context = context.gather(
        1, batch.lengths.unsqueeze(1) + context_cols.unsqueeze(0)
    )
    pool.set_ngram_context(batch.state_indices, next_context)

    track = _ple_track_targets(forward_batch, batch)
    if track is not None:
        track_indices, track_offsets = track
        pool.set_ngram_context(
            track_indices,
            context.gather(1, track_offsets.unsqueeze(1) + context_cols.unsqueeze(0)),
        )


def _ple_track_targets(
    forward_batch: ForwardBatch, batch: _PLEBatch
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Destination slots and gather offsets for the extra-buffer track snapshot.

    With extra_buffer the radix tree caches the ping-pong track slot,
    not the working slot.
    Both PLE side states are laid out [incoming_state | chunk tokens],
    so the boundary value is the state's own gather at a smaller offset;
    callers differ only in tensor rank, hence offsets rather than a gather.
    Masked-off rows route to reserved slot 0, so the shape stays graph-capturable.
    None when tracking is inactive or its metadata is absent.
    """
    track_indices = forward_batch.mamba_track_indices
    track_mask = forward_batch.mamba_track_mask
    if track_indices is None or track_mask is None:
        return None

    rows = batch.lengths.shape[0]
    track_indices = track_indices[:rows]
    dst = torch.where(track_mask[:rows], track_indices, torch.zeros_like(track_indices))

    if batch.mode.is_decode():
        # One token per step, so the boundary offset is the current one. Decode never
        # carries mamba_track_seqlens, so this path must not consult it.
        return dst, batch.lengths

    aligned = forward_batch.mamba_track_aligned_lens()
    if aligned is None:
        return None

    return dst, aligned[:rows].clamp(min=0).minimum(batch.lengths)


def _pad_token_rows(x: torch.Tensor, total_tokens: int) -> torch.Tensor:
    if x.shape[0] == total_tokens:
        return x
    out = x.new_zeros((total_tokens, *x.shape[1:]))
    out[: x.shape[0]] = x
    return out


def _use_attn_tp_ngram() -> bool:
    return is_dp_attention_enabled() and envs.SGLANG_USE_ATTN_TP_NGRAM.get()


def ple_ngram_vocab_tp_kwargs(config, use_attn_tp_ngram: bool) -> dict:
    """fnFL2 H69b: the vocab-parallel layout of the PLE n-gram table.

    FORM A (F13, the seam's missing sibling): F13 builds the host's
    ``embed_tokens`` with ``enable_tp=False`` because ``tp_vocab_ratios`` keeps
    every vocab dimension EVEN under ``--rank-tp-ratio 1,0,0`` -- the host
    would hold one third of the rows. The n-gram table is a vocab dimension
    too and was left on the default: on D's host (TP=3) its shard is
    ``[0, V/3)`` of the n-gram id space, i.e. bigram heads 0-4 and a third of
    head 5, while ``Qwen4ExpPinnedHostEmbedding.reduce`` (F12) skips the
    all-reduce that would have added the workers' shards -- which a Form A
    worker never builds. The rest of head 5, bigram heads 6-7 and all eight
    trigram heads read as zero rows on D; x168 counted it: kernel_rows 685 of
    rows 2048 per 32 rounds (33.4 %). P (PP3, tp_size 1 per stage) is
    unaffected, so D decodes a different model than P prefilled.

    ``SGLANG_WEG2_FORM_A_PLE_FULL_VOCAB=1`` gives the host the full table, like
    F13 does for ``embed_tokens``: tp_size 1, full range, no mask, no
    collective. Only with the ``checkpoint`` offload backend, which maps the
    whole table anyway; a copying backend would materialize all of it in host
    memory, so there the switch is refused by name and the layout stays.
    Off, or outside Form A: the pre-H69b layout, byte-identical."""
    if not (
        envs.SGLANG_WEG2_FORM_A_PLE_FULL_VOCAB.get()
        and form_a_dense_is_unsharded()
        and not use_attn_tp_ngram
    ):
        return {}
    if not (
        getattr(config, "ple_offload_embedding", False)
        and getattr(config, "ple_offload_backend", None) == "checkpoint"
    ):
        logger.warning(
            "SGLANG_WEG2_FORM_A_PLE_FULL_VOCAB refused: the full n-gram table "
            "is only taken with the 'checkpoint' PLE offload backend (got %r); "
            "the host keeps its even TP shard",
            getattr(config, "ple_offload_backend", None),
        )
        return {}
    return {"enable_tp": False}


class Qwen4ExpPLEGroupedNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        group_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if group_size is not None and hidden_size % group_size != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by group_size ({group_size})"
            )
        self.eps = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        # The JIT kernel requires group_size to be a multiple of 512; this is
        # init-static, so resolve it once here (device/dtype stay per-call).
        effective_group_size = group_size if group_size is not None else hidden_size
        self._jit_group_size = (
            effective_group_size if effective_group_size % 512 == 0 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self._jit_group_size is not None
            and x.is_cuda
            and x.dtype in (torch.bfloat16, torch.float16)
        ):
            from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm import (
                grouped_gemma_rmsnorm,
            )

            return grouped_gemma_rmsnorm(x, self.weight, self._jit_group_size, self.eps)
        compute_dtype = x.dtype
        x_float = x.float()
        if self.group_size is None:
            variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        else:
            group_shape = x_float.shape[:-1] + (-1, self.group_size)
            variance = x_float.reshape(group_shape).pow(2).mean(dim=-1, keepdim=True)
            variance = variance.expand(group_shape).reshape_as(x_float)
        x_norm = x_float * torch.rsqrt(variance + self.eps)
        weight = self.weight.float() + 1.0
        return (x_norm * weight).to(compute_dtype)


class Qwen4ExpNGramEmbedding(nn.Module):
    _MASK64 = (1 << 64) - 1
    _SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
    _SPLITMIX_M1 = 0xBF58476D1CE4E5B9
    _SPLITMIX_M2 = 0x94D049BB133111EB
    _PRIME_1 = 10007

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        embedding_dim: int,
        ple_layer_index: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.ngram_embed_dim = int(embedding_dim)
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.ple_layer_index = int(ple_layer_index)
        self.unigram_vocab_size = int(config.vocab_size)
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be > 0, got {self.heads_per_ngram}")
        if self.ngram_embed_dim % self.ngram_heads != 0:
            raise ValueError(
                "ple_embed_dim must be divisible by total ngram heads: "
                f"{self.ngram_embed_dim} % {self.ngram_heads} != 0"
            )
        self.ngram_vocab_size_base = int(config.ngram_vocab_size_base)
        if self.ngram_vocab_size_base <= 0:
            raise ValueError("ngram_vocab_size_base must be > 0")
        self.make_ngram_vocab_size_divisible_by = int(
            config.make_ngram_vocab_size_divisible_by
        )
        self.head_dim_per_ngram = self.ngram_embed_dim // self.ngram_heads
        self.eos_token_id = int(config.eos_token_id)
        self.enable_ple_fusion = envs.SGLANG_ENABLE_QWEN4_PLE_FUSION.get()

        self.register_buffer(
            "layer_multipliers",
            self._build_layer_multipliers(self.ngram_size),
            persistent=True,
        )
        head_vocab_sizes, head_offsets, total_vocab_size = (
            self._build_head_vocab_and_offsets()
        )
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(head_vocab_sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(head_offsets, dtype=torch.long),
            persistent=True,
        )
        padded_vocab_size = (
            (total_vocab_size + self.make_ngram_vocab_size_divisible_by - 1)
            // self.make_ngram_vocab_size_divisible_by
        ) * self.make_ngram_vocab_size_divisible_by
        self.use_attn_tp_ngram = _use_attn_tp_ngram()
        self.gather_dp_tokens = (
            is_dp_attention_enabled()
            and get_attention_dp_size() > 1
            and not self.use_attn_tp_ngram
        )
        ngram_prefix = f"{prefix}.ngram_embedding" if prefix else "ngram_embedding"
        # Line note (Qwen3.8-Flash-Next on 20/32 GB cards): with the PLE host
        # offload on, the device table is only a SHAPE/DTYPE carrier --
        # Qwen4ExpPinnedHostEmbedding reads those, copies the weight's
        # attributes and deletes it. Upstream still allocated the full
        # [vocab/tp, dim] shard on the card first (32 GiB per rank of the
        # 95 GiB table), which a 20 GB card cannot hold even transiently.
        # Build it on ``meta`` in that case; nothing downstream touches its
        # bytes before the wrapper replaces it.
        _table_device = (
            torch.device("meta")
            if getattr(config, "ple_offload_embedding", False)
            else contextlib.nullcontext()
        )
        with _table_device:
            self.ngram_embedding = VocabParallelEmbedding(
                padded_vocab_size,
                self.head_dim_per_ngram,
                params_dtype=(
                    torch.float8_e4m3fn
                    if _ple_table_is_fp8(config, quant_config, ngram_prefix)
                    else torch.bfloat16
                ),
                output_dtype=torch.bfloat16,
                use_attn_tp_group=self.use_attn_tp_ngram,
                **ple_ngram_vocab_tp_kwargs(config, self.use_attn_tp_ngram),
            )
        self.ngram_embedding.register_buffer(
            "weight_scale", torch.ones(1, dtype=torch.bfloat16), persistent=True
        )

    @classmethod
    def _splitmix64(cls, x: int) -> int:
        x = (x + cls._SPLITMIX_GAMMA) & cls._MASK64
        x = ((x ^ (x >> 30)) * cls._SPLITMIX_M1) & cls._MASK64
        x = ((x ^ (x >> 27)) * cls._SPLITMIX_M2) & cls._MASK64
        return (x ^ (x >> 31)) & cls._MASK64

    def _build_layer_multipliers(self, size: int) -> torch.Tensor:
        seed = int(getattr(self.config, "seed", 1234))
        max_long = (1 << 63) - 1
        m_max = max_long // max(self.unigram_vocab_size, 1)
        half_bound = max(1, m_max // 2)
        values = []
        base_seed = seed + self._PRIME_1 * self.ple_layer_index
        for idx in range(size):
            x0 = (base_seed + self._SPLITMIX_GAMMA * (idx + 1)) & self._MASK64
            mixed = self._splitmix64(x0)
            values.append(int(2 * (mixed % half_bound) + 1))
        return torch.tensor(values, dtype=torch.long)

    @staticmethod
    def _find_nth_prime_after(start: int, n: int) -> int:
        prime = int(start)
        for _ in range(n):
            prime = int(sympy.nextprime(prime))
        return prime

    def _build_head_vocab_and_offsets(self):
        sizes = []
        offsets = []
        total = 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = self.ple_layer_index * self.ngram_heads + head_idx
            size = self._find_nth_prime_after(
                self.ngram_vocab_size_base - 1, global_head_idx + 1
            )
            sizes.append(size)
            offsets.append(total)
            total += size
        return sizes, offsets, total

    def _embed_ngram_ids(
        self,
        ngram_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        physical_tokens: int,
    ) -> torch.Tensor:
        lookup_ids, semantic_tokens = self._prepare_embedding_lookup(
            ngram_ids, forward_batch, physical_tokens
        )
        embeddings = self.ngram_embedding(lookup_ids)
        embeddings = embeddings * self.ngram_embedding.weight_scale
        return self._finish_embedding_lookup(
            embeddings, semantic_tokens, forward_batch, physical_tokens
        )

    def _prepare_embedding_lookup(
        self,
        ngram_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        physical_tokens: int,
    ) -> Tuple[torch.Tensor, int]:
        semantic_tokens = ngram_ids.shape[0]
        if not self.gather_dp_tokens:
            return ngram_ids, semantic_tokens

        padded_ngram_ids = _pad_token_rows(ngram_ids, physical_tokens)
        global_tokens = forward_batch.global_dp_buffer_len
        if global_tokens is None:
            raise RuntimeError(
                "global-TP Qwen4 N-gram lookup under DP attention requires a "
                "DP token layout; set SGLANG_USE_ATTN_TP_NGRAM=1 to shard the "
                "table within each attention-TP group"
            )

        global_ngram_ids = ngram_ids.new_empty((global_tokens, *ngram_ids.shape[1:]))
        dp_gather_replicate(
            global_ngram_ids, padded_ngram_ids.contiguous(), forward_batch
        )
        return global_ngram_ids, semantic_tokens

    def _finish_embedding_lookup(
        self,
        embeddings: torch.Tensor,
        semantic_tokens: int,
        forward_batch: ForwardBatch,
        physical_tokens: int,
    ) -> torch.Tensor:
        if not self.gather_dp_tokens:
            return embeddings
        local_embeddings = embeddings.new_empty(
            (physical_tokens, *embeddings.shape[1:])
        )
        dp_scatter(local_embeddings, embeddings.contiguous(), forward_batch)
        return local_embeddings[:semantic_tokens]

    def _hash_contexts(
        self, contexts: torch.Tensor, *, decode_sized: bool = False
    ) -> torch.Tensor:
        contexts = contexts.to(torch.long)
        if self.enable_ple_fusion and decode_sized:
            from sglang.kernels.ops.qwen4_ple import (
                can_fuse_qwen4_ngram_hash,
                fused_qwen4_ngram_hash,
            )

            if can_fuse_qwen4_ngram_hash(
                contexts,
                self.layer_multipliers,
                self.ngram_heads_vocab_sizes,
                self.ngram_heads_offsets,
            ):
                return fused_qwen4_ngram_hash(
                    contexts,
                    self.layer_multipliers,
                    self.ngram_heads_vocab_sizes,
                    self.ngram_heads_offsets,
                    self.eos_token_id,
                )

        pool = get_req_to_token_pool()
        cached = pool.ple_window_cache
        if cached is not None and cached[1] is contexts and cached[2] is not None:
            shifted_tokens = cached[2]
            assert len(shifted_tokens) == self.ngram_size
        else:
            shifted_tokens = [contexts]
            for shift in range(1, self.ngram_size):
                shifted_tokens.append(self._shift_right_ignore_eos(contexts, shift))
            if cached is not None and cached[1] is contexts:
                pool.ple_window_cache = (cached[0], contexts, shifted_tokens)

        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            ngram_idx = ngram - 2
            start_idx = ngram_idx * self.heads_per_ngram
            end_idx = start_idx + self.heads_per_ngram
            mix = shifted_tokens[0] * self.layer_multipliers[0]
            for pos in range(1, ngram):
                mix = torch.bitwise_xor(
                    mix, shifted_tokens[pos] * self.layer_multipliers[pos]
                )
            head_vocab_sizes = self.ngram_heads_vocab_sizes[start_idx:end_idx]
            head_offsets = self.ngram_heads_offsets[start_idx:end_idx]
            ngram_ids = torch.remainder(
                mix[:, -1:].unsqueeze(-1), head_vocab_sizes.view(1, 1, -1)
            )
            ngram_ids = ngram_ids + head_offsets.view(1, 1, -1)
            blocks.append(ngram_ids[:, 0])
        return torch.cat(blocks, dim=-1)

    def _shift_right_ignore_eos(self, tensor: torch.Tensor, n: int) -> torch.Tensor:
        if n == 0:
            return tensor
        batch_size, seq_len = tensor.shape
        idx = torch.arange(seq_len, device=tensor.device, dtype=torch.long)
        eos_mask = tensor == self.eos_token_id
        eos_pos = torch.where(eos_mask, idx, -1)
        prev_eos_inclusive = torch.cummax(eos_pos, dim=1).values
        prev_eos = torch.cat(
            [eos_pos.new_full((batch_size, 1), -1), prev_eos_inclusive[:, :-1]],
            dim=1,
        )
        segment_start = prev_eos + 1
        pos_in_segment = idx.unsqueeze(0) - segment_start
        src_idx = idx - n
        gather_idx = torch.clamp(src_idx, min=0).unsqueeze(0).expand(batch_size, -1)
        shifted = tensor.gather(dim=1, index=gather_idx)
        valid_mask = (pos_in_segment >= n) & (src_idx.unsqueeze(0) >= 0)
        return torch.where(valid_mask, shifted, tensor.new_full((), self.eos_token_id))

    def forward_idle(self, forward_batch: ForwardBatch) -> None:
        if not self.gather_dp_tokens:
            return
        input_ids = forward_batch.input_ids.reshape(-1)
        dummy_ids = input_ids.new_zeros((input_ids.shape[0], self.ngram_heads))
        self._embed_ngram_ids(dummy_ids, forward_batch, input_ids.shape[0])

    def forward(
        self,
        batch: _PLEBatch,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        ngram_ids = self.compute_ngram_ids(batch)
        embeddings = self._embed_ngram_ids(
            ngram_ids, forward_batch, batch.physical_tokens
        )
        return embeddings.flatten(start_dim=-2)

    def compute_ngram_ids(self, batch: _PLEBatch) -> torch.Tensor:
        assert batch.ngram_context is not None
        pool = get_req_to_token_pool()
        cached = pool.ple_window_cache
        if cached is not None and cached[0] is batch:
            contexts = cached[1]
        else:
            if batch.use_decode_fast_path:
                contexts = batch.ngram_context
            else:
                contexts = batch.ngram_context.unfold(1, self.ngram_size, 1)[
                    batch.req_indices, batch.token_offsets
                ]
            contexts = contexts.to(torch.long)
            pool.ple_window_cache = (batch, contexts, None)
        return self._hash_contexts(
            contexts,
            decode_sized=batch.mode.is_decode() or batch.mode.is_target_verify(),
        )


@triton.jit
def _gather_ple_embedding_from_pinned_kernel(
    weight_ptr,
    ids_ptr,
    output_ptr,
    embedding_dim,
    tp_vocab_start,
    tp_vocab_end,
    is_fp8: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FP8_DECODE: tl.constexpr = 3,
):
    # H68d: FP8_DECODE (qwen4_exp_ple_fp8.py) -- 3 = the native fp8e4nv
    # pointer (sm90+; this branch and the bf16 one are the pre-H68d code),
    # 0..2 = uint8 bytes decoded in the kernel (sm86: no fp8e4nv there)
    row_id = tl.program_id(0)
    global_idx = tl.load(ids_ptr + row_id)
    in_range = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
    local_idx = tl.where(in_range, global_idx - tp_vocab_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < embedding_dim
    if is_fp8:
        if FP8_DECODE == 3:
            weight_ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
        else:
            weight_ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
    else:
        weight_ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.bfloat16))
    if is_fp8 and FP8_DECODE != 3:
        values = ple_fp8_bytes_to_bf16(
            tl.load(weight_ptr + local_idx * embedding_dim + offsets, mask=mask, other=0),
            FP8_DECODE,
        )
    else:
        values = tl.load(
            weight_ptr + local_idx * embedding_dim + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.bfloat16)
    tl.store(
        output_ptr + row_id * embedding_dim + offsets,
        tl.where(in_range, values, 0.0),
        mask=mask,
    )


@triton.jit
def _gather_ple_embedding_from_shards_kernel(
    bases_ptr,
    shard_rows,
    ids_ptr,
    output_ptr,
    embedding_dim,
    tp_vocab_start,
    tp_vocab_end,
    is_fp8: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FP8_DECODE: tl.constexpr = 3,
):
    """Same gather as the pinned kernel, but the table is a list of shard
    base pointers (read-only mmaps of the checkpoint files): global row ->
    (row // shard_rows, row % shard_rows). FP8_DECODE as in the pinned
    kernel (H68d)."""
    row_id = tl.program_id(0)
    global_idx = tl.load(ids_ptr + row_id).to(tl.int64)
    in_range = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
    safe_idx = tl.where(in_range, global_idx, 0)
    shard = safe_idx // shard_rows
    local = safe_idx - shard * shard_rows
    base = tl.load(bases_ptr + shard)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < embedding_dim
    if is_fp8:
        if FP8_DECODE == 3:
            weight_ptr = base.to(tl.pointer_type(tl.float8e4nv))
        else:
            weight_ptr = base.to(tl.pointer_type(tl.uint8))
    else:
        weight_ptr = base.to(tl.pointer_type(tl.bfloat16))
    if is_fp8 and FP8_DECODE != 3:
        values = ple_fp8_bytes_to_bf16(
            tl.load(weight_ptr + local * embedding_dim + offsets, mask=mask, other=0),
            FP8_DECODE,
        )
    else:
        values = tl.load(
            weight_ptr + local * embedding_dim + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.bfloat16)
    tl.store(
        output_ptr + row_id * embedding_dim + offsets,
        tl.where(in_range, values, 0.0),
        mask=mask,
    )


class Qwen4ExpPinnedHostEmbedding(VocabParallelEmbedding):
    """PLE table read directly from host memory (pinned, or a file-backed mmap).

    The table stays in its checkpoint storage dtype (fp8 with a per-tensor
    weight_scale for fp8 checkpoints, bf16 otherwise); gathers emit bf16.
    """

    _COPIED_ATTRIBUTES = (
        "quant_config",
        "enable_tp",
        "use_attn_tp_group",
        "tp_size",
        "num_embeddings",
        "org_vocab_size",
        "padding_size",
        "num_added_embeddings",
        "use_presharded_weights",
        "org_vocab_size_padded",
        "num_embeddings_padded",
        "shard_indices",
        "embedding_dim",
        "num_embeddings_per_partition",
        "num_org_embeddings_per_partition",
        "num_added_embeddings_per_partition",
    )

    def __init__(
        self,
        embedding: VocabParallelEmbedding,
        *,
        backend: str = "pinned",
        table_dir: Optional[str] = None,
    ) -> None:
        nn.Module.__init__(self)
        if not isinstance(embedding.quant_method, UnquantizedEmbeddingMethod):
            raise NotImplementedError(
                "PLE embedding offload requires an unquantized embedding table"
            )
        if embedding.weight.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise TypeError(
                "PLE embedding offload requires bfloat16 or fp8 weights, got "
                f"{embedding.weight.dtype}"
            )
        if embedding.num_added_embeddings:
            raise NotImplementedError(
                "PLE embedding offload does not support added vocabulary rows"
            )
        for name in self._COPIED_ATTRIBUTES:
            setattr(self, name, getattr(embedding, name))
        # The unquantized CUDA post-load hook is a no-op. Exclude this CPU-only
        # table so the generic loader does not stage it back to GPU unnecessarily.
        self.quant_method = None

        source_weight = embedding.weight
        # ``checkpoint`` (this line): the table is never copied; the checkpoint
        # files themselves are mapped (see qwen4_exp_ple_table) and
        # ``self.weight`` is a 0-row carrier of dtype/attributes. The mapping
        # is created by the model's load_weights once it knows the checkpoint
        # tensor names (``attach_checkpoint_table``); a gather before that
        # refuses by name.
        self._ckpt_table = None
        self._ckpt_pread = None
        # fnFL2 H32: the owning n-gram embedding's hash, for the next-chunk
        # prefetch (set by Qwen4ExpPLELayer; None = no prefetch)
        self.next_chunk_hasher = None
        # fnFL2 H40: the owning n-gram embedding's hash constants for the
        # decode stage (set by Qwen4ExpPLELayer; None = no stage), and the
        # stage itself (attach_checkpoint_table)
        self.decode_stage_params = None
        self._decode_stager = None
        self._retired_decode_stagers = []
        self._ckpt_backend = backend == "checkpoint"
        if self._ckpt_backend:
            host_table = torch.empty(
                (0, source_weight.shape[1]), dtype=source_weight.dtype, device="cpu"
            )
        else:
            host_table = allocate_ple_host_table(
                shape=source_weight.shape,
                dtype=source_weight.dtype,
                backend=backend,
                table_dir=table_dir,
                # Each TP rank holds a different vocabulary shard of the same shape.
                tag=(
                    f"rows{self.shard_indices.org_vocab_start_index}"
                    f"-{self.shard_indices.org_vocab_end_index}"
                ),
            )
        # Only the file backend has anything to prefetch (rows live on storage).
        self._file_prefetcher = make_ple_file_prefetcher(host_table)
        # ... and only it needs its resident set bounded: a fault maps a whole
        # folio, so the mapping would otherwise creep towards the full table.
        self._file_rss_trimmer = make_ple_file_rss_trimmer(host_table)
        cpu_weight = nn.Parameter(host_table, requires_grad=False)
        for name, value in vars(source_weight).items():
            setattr(cpu_weight, name, value)
        cpu_weight.weight_loader = self.weight_loader
        self.register_parameter("weight", cpu_weight)
        # The scale is tiny; keep it with the model instead of offloading it
        # with the table.
        self.register_buffer("weight_scale", embedding.weight_scale, persistent=True)
        del embedding.weight
        self._block_d = triton.next_power_of_2(self.embedding_dim)

    def attach_checkpoint_table(self, table) -> None:
        """``checkpoint`` backend: adopt the mapped shards (model load_weights)."""
        if not self._ckpt_backend:
            raise RuntimeError("attach_checkpoint_table on a non-checkpoint PLE table")
        if table.embedding_dim != self.embedding_dim or table.dtype != self.weight.dtype:
            raise ValueError(
                f"mapped PLE table {table.dtype}x{table.embedding_dim} does not "
                f"match the embedding {self.weight.dtype}x{self.embedding_dim}"
            )
        self._ckpt_table = table
        self._ckpt_prefetcher = make_ple_checkpoint_prefetcher(table)
        self._ckpt_pread = make_ple_prefetch_gather(
            make_ple_checkpoint_pread_gather(table), table, self.next_chunk_hasher
        )
        if self._decode_stager is not None:
            # a graph captured against the old stage may still replay: retire
            # it (every id -1) but keep its host memory alive
            self._decode_stager.retire()
            self._retired_decode_stagers.append(self._decode_stager)
        self._decode_stager = make_ple_decode_stager(
            table,
            self.decode_stage_params,
            vocab_start=self.shard_indices.org_vocab_start_index,
            vocab_end=self.shard_indices.org_vocab_end_index,
        )

    def allocate_output(
        self, shape: Tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        allocation_context = nullcontext()
        if self.tp_size > 1:
            allocation_context = use_symmetric_memory(
                get_tp_group(), disabled=not is_allocation_symmetric()
            )
        with allocation_context, torch.inference_mode(False):
            # The gather kernel emits bf16 rows regardless of the table dtype.
            return torch.empty(shape, dtype=torch.bfloat16, device=device)

    def gather(
        self, input_ids: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        expected_shape = (*input_ids.shape, self.embedding_dim)
        if out is None:
            output = self.allocate_output(expected_shape, input_ids.device)
        else:
            if tuple(out.shape) != expected_shape:
                raise ValueError(
                    f"invalid PLE prefetch output shape: {tuple(out.shape)} != "
                    f"{expected_shape}"
                )
            if out.dtype != torch.bfloat16 or out.device != input_ids.device:
                raise ValueError(
                    "PLE prefetch output must be bfloat16 on the id device"
                )
            output = out

        flat_ids = input_ids.reshape(-1).long()
        if flat_ids.numel() and self._ckpt_backend:
            table = self._ckpt_table
            if table is None:
                raise RuntimeError(
                    "PLE checkpoint table was never attached (load_weights did "
                    "not see the ngram_embedding shards)"
                )
            pread = getattr(self, "_ckpt_pread", None)
            if pread is not None and pread.wants(flat_ids):
                return pread.gather_into(
                    flat_ids,
                    output,
                    vocab_start=self.shard_indices.org_vocab_start_index,
                    vocab_end=self.shard_indices.org_vocab_end_index,
                )
            prefetcher = getattr(self, "_ckpt_prefetcher", None)
            if prefetcher is not None:
                prefetcher.enqueue(
                    flat_ids,
                    vocab_start=self.shard_indices.org_vocab_start_index,
                    vocab_end=self.shard_indices.org_vocab_end_index,
                )
            # fnFL2 H40: decode/verify-sized gathers read their staged rows
            # from the host stage the pread workers filled before the replay
            stager = self._decode_stager
            if stager is not None and stager.launch(
                flat_ids,
                output,
                vocab_start=self.shard_indices.org_vocab_start_index,
                vocab_end=self.shard_indices.org_vocab_end_index,
                block_d=self._block_d,
            ):
                return output
            _gather_ple_embedding_from_shards_kernel[(flat_ids.numel(),)](
                table.bases_on(flat_ids.device),
                table.shard_rows,
                flat_ids,
                output,
                embedding_dim=self.embedding_dim,
                tp_vocab_start=self.shard_indices.org_vocab_start_index,
                tp_vocab_end=self.shard_indices.org_vocab_end_index,
                is_fp8=table.dtype == torch.float8_e4m3fn,
                BLOCK_D=self._block_d,
                FP8_DECODE=ple_fp8_decode_arg(table.dtype, flat_ids.device),
            )
            return output
        if flat_ids.numel():
            if self._file_prefetcher is not None:
                self._file_prefetcher.enqueue(
                    flat_ids,
                    vocab_start=self.shard_indices.org_vocab_start_index,
                    vocab_end=self.shard_indices.org_vocab_end_index,
                )
            _gather_ple_embedding_from_pinned_kernel[(flat_ids.numel(),)](
                self.weight.data_ptr(),
                flat_ids,
                output,
                embedding_dim=self.embedding_dim,
                tp_vocab_start=self.shard_indices.org_vocab_start_index,
                tp_vocab_end=self.shard_indices.org_vocab_end_index,
                is_fp8=self.weight.dtype == torch.float8_e4m3fn,
                BLOCK_D=self._block_d,
                FP8_DECODE=ple_fp8_decode_arg(self.weight.dtype, flat_ids.device),
            )
        return output

    def reduce(self, output: torch.Tensor) -> torch.Tensor:
        # FORM A (F12): the dense side is not split across ranks -- this
        # rank owns all of it -- so there is no partial sum to reduce and
        # no second participant to reduce with. Issuing the collective
        # would block on ranks that never arrive.
        if form_a_dense_is_unsharded():
            return output
        if self.tp_size > 1 and not get_attn_tp_context().input_scattered:
            if self.use_attn_tp_group:
                return attn_tp_all_reduce(output)
            return tensor_model_parallel_all_reduce(output)
        return output

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.reduce(self.gather(input_ids))


class Qwen4ExpPLELayer(nn.Module):
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        layer_id: Optional[int] = None,
        ple_layer_index: int = 0,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.ple_embed_dim = config.ple_embed_dim
        self.conv_kernel_size = config.ple_conv_kernel_size
        self.hc_count = config.hc_count
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.ple_embedding = Qwen4ExpNGramEmbedding(
            config,
            self.ple_embed_dim,
            ple_layer_index=ple_layer_index,
            quant_config=quant_config,
            prefix=f"{prefix}.ple_embedding" if prefix else "ple_embedding",
        )
        if config.ple_offload_embedding:
            self.ple_embedding.ngram_embedding = Qwen4ExpPinnedHostEmbedding(
                self.ple_embedding.ngram_embedding,
                backend=getattr(config, "ple_offload_backend", "pinned"),
                table_dir=getattr(config, "ple_offload_dir", None),
            )
            self.ple_embedding.ngram_embedding.next_chunk_hasher = (
                ple_next_chunk_hasher(self.ple_embedding)
            )
            # fnFL2 H40: the decode stage hashes the verify windows on the
            # host; the DP-gathered layout (gather_dp_tokens) is not staged
            if not self.ple_embedding.gather_dp_tokens:
                self.ple_embedding.ngram_embedding.decode_stage_params = (
                    functools.partial(PleHashParams.of, self.ple_embedding)
                )
        self.short_conv_dilation = self.ple_embedding.ngram_size
        self.short_conv_state_len = (
            self.conv_kernel_size - 1
        ) * self.short_conv_dilation
        self.conv_channels = self.hc_hidden_size
        self.key_proj = ReplicatedLinear(
            self.ple_embed_dim,
            self.conv_channels,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.key_proj",
        )
        self.value_proj = ReplicatedLinear(
            self.ple_embed_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.value_proj",
        )
        norm_hidden = self.hc_hidden_size
        norm_group = self.hidden_size
        self.norm_key = Qwen4ExpPLEGroupedNorm(
            norm_hidden,
            eps=config.rms_norm_eps,
            group_size=norm_group,
        )
        self.norm_query = Qwen4ExpPLEGroupedNorm(
            norm_hidden,
            eps=config.rms_norm_eps,
            group_size=norm_group,
        )
        self.norm_conv = Qwen4ExpPLEGroupedNorm(
            norm_hidden,
            eps=config.rms_norm_eps,
            group_size=norm_group,
        )
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_channels,
            out_channels=self.conv_channels,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_channels,
            padding=(self.conv_kernel_size - 1) * self.short_conv_dilation,
            dilation=self.short_conv_dilation,
            bias=False,
        )
        nn.init.zeros_(self.conv1d.weight)
        self._prefetch_stream = (
            torch.cuda.Stream() if config.ple_offload_embedding else None
        )
        self._graph_prefetch_buffers = {}
        self._eager_prefetch_buffer = None
        self._prefetch_state = None

    def _apply_ple_norm(self, norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
        y = norm(x.flatten(-2, -1))
        return y.unflatten(-1, (self.hc_count, self.hidden_size))

    def _short_conv(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        batch: _PLEBatch,
    ) -> torch.Tensor:
        if x.shape[0] == 0:
            return x
        pool = get_req_to_token_pool()
        conv_state = pool.short_conv_layer_cache(self.layer_id)

        if batch.use_decode_fast_path:
            # With row_width=1 the padded/transpose path is x.unsqueeze(-1),
            # and each state boundary is a one-column shift; conv and SiLU stay native.
            from sglang.kernels.ops.qwen4_ple import (
                can_fuse_qwen4_short_conv_state,
                fused_qwen4_short_conv_state,
            )

            fused_state = can_fuse_qwen4_short_conv_state(
                conv_state, batch.state_indices, x
            )
            if fused_state:
                conv_input = fused_qwen4_short_conv_state(
                    conv_state, batch.state_indices, x
                )
            else:
                state = conv_state.index_select(0, batch.state_indices).to(
                    dtype=x.dtype
                )
                conv_input = torch.cat([state, x.unsqueeze(-1)], dim=-1)
            conv_output = F.conv1d(
                conv_input,
                self.conv1d.weight.to(dtype=x.dtype),
                bias=None,
                dilation=self.short_conv_dilation,
                groups=self.conv_channels,
            ).squeeze(-1)
            next_state = conv_input[:, :, batch.row_width :]
            if not fused_state:
                conv_state[batch.state_indices] = next_state.to(dtype=conv_state.dtype)

            track = _ple_track_targets(forward_batch, batch)
            if track is not None:
                track_indices, _ = track
                conv_state[track_indices] = next_state.to(dtype=conv_state.dtype)
            return F.silu(conv_output)

        state = conv_state.index_select(0, batch.state_indices).to(dtype=x.dtype)
        padded_seq = x.new_zeros(
            (batch.lengths.shape[0], batch.row_width, self.conv_channels)
        )
        padded_seq[batch.req_indices, batch.token_offsets] = x
        conv_input = torch.cat([state, padded_seq.transpose(1, 2)], dim=-1)
        conv_output = F.conv1d(
            conv_input,
            self.conv1d.weight.to(dtype=x.dtype),
            bias=None,
            dilation=self.short_conv_dilation,
            groups=self.conv_channels,
        ).transpose(1, 2)

        if batch.mode.is_target_verify():
            intermediate_cache = pool.short_conv_layer_intermediate_cache(self.layer_id)
            if intermediate_cache is not None:
                if self.short_conv_state_len:
                    intermediate_state = (
                        conv_input.unfold(2, self.short_conv_state_len, 1)[
                            :, :, 1 : batch.row_width + 1
                        ]
                        .permute(0, 2, 1, 3)
                        .contiguous()
                    )
                else:
                    intermediate_state = x.new_empty(
                        (
                            batch.lengths.shape[0],
                            batch.row_width,
                            self.conv_channels,
                            0,
                        )
                    )
                valid_steps = batch.valid_tokens.reshape(
                    batch.lengths.shape[0], batch.row_width, 1, 1
                )
                intermediate_state = torch.where(
                    valid_steps,
                    intermediate_state,
                    torch.zeros_like(intermediate_state),
                )
                intermediate_cache[: batch.lengths.shape[0], : batch.row_width].copy_(
                    intermediate_state.to(dtype=intermediate_cache.dtype)
                )
        else:
            state_cols = torch.arange(
                self.short_conv_state_len, device=x.device, dtype=torch.long
            )

            def _gather_at(offsets: torch.Tensor) -> torch.Tensor:
                return conv_input.gather(
                    2,
                    (offsets.unsqueeze(1) + state_cols.unsqueeze(0))
                    .unsqueeze(1)
                    .expand(-1, self.conv_channels, -1),
                )

            next_state = _gather_at(batch.lengths)
            conv_state[batch.state_indices] = next_state.to(dtype=conv_state.dtype)

            # Same boundary mamba uses, into the slot the radix tree reads.
            track = _ple_track_targets(forward_batch, batch)
            if track is not None:
                track_indices, track_offsets = track
                conv_state[track_indices] = _gather_at(track_offsets).to(
                    dtype=conv_state.dtype
                )

        return F.silu(conv_output[batch.req_indices, batch.token_offsets])

    def forward_idle(self, forward_batch: ForwardBatch) -> None:
        if self._prefetch_state is not None:
            self._consume_prefetched_embeddings(forward_batch)
        else:
            self.ple_embedding.forward_idle(forward_batch)

    def _allocate_prefetch_buffer(
        self, lookup_tokens: int, lookup_ids: torch.Tensor
    ) -> torch.Tensor:
        offloaded_embedding = self.ple_embedding.ngram_embedding
        return offloaded_embedding.allocate_output(
            (lookup_tokens, self.ple_embed_dim), lookup_ids.device
        )

    def _get_prefetch_buffer(
        self, lookup_tokens: int, lookup_ids: torch.Tensor
    ) -> torch.Tensor:
        if get_is_capture_mode():
            buffer = self._graph_prefetch_buffers.get(lookup_tokens)
            if buffer is None:
                buffer = self._allocate_prefetch_buffer(lookup_tokens, lookup_ids)
                self._graph_prefetch_buffers[lookup_tokens] = buffer
            return buffer

        buffer = self._eager_prefetch_buffer
        if buffer is None or buffer.shape[0] < lookup_tokens:
            buffer = self._allocate_prefetch_buffer(lookup_tokens, lookup_ids)
            self._eager_prefetch_buffer = buffer
        return buffer[:lookup_tokens]

    def start_prefetch(
        self,
        batch: Optional[_PLEBatch],
        forward_batch: ForwardBatch,
    ) -> None:
        """Gather PLE rows via UVA while the preceding decoder layer runs."""
        if self._prefetch_stream is None:
            return
        if self._prefetch_state is not None:
            raise RuntimeError("PLE prefetch state was not consumed before reuse")
        if batch is None:
            if not self.ple_embedding.gather_dp_tokens:
                return
            physical_tokens = forward_batch.input_ids.numel()
            ngram_ids = forward_batch.input_ids.new_zeros(
                (physical_tokens, self.ple_embedding.ngram_heads)
            )
        else:
            physical_tokens = batch.physical_tokens
            ngram_ids = self.ple_embedding.compute_ngram_ids(batch)

        lookup_ids, semantic_tokens = self.ple_embedding._prepare_embedding_lookup(
            ngram_ids, forward_batch, physical_tokens
        )
        lookup_tokens = lookup_ids.shape[0]
        if lookup_tokens == 0:
            return
        prefetched = self._get_prefetch_buffer(lookup_tokens, lookup_ids)
        output_view = prefetched.view(lookup_tokens, self.ple_embedding.ngram_heads, -1)
        offloaded_embedding = self.ple_embedding.ngram_embedding

        stream = self._prefetch_stream
        stream.wait_stream(torch.cuda.current_stream())
        lookup_ids.record_stream(stream)
        with torch.cuda.stream(stream):
            offloaded_embedding.gather(lookup_ids, out=output_view)
        self._prefetch_state = prefetched, semantic_tokens, physical_tokens

    def _consume_prefetched_embeddings(
        self, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        if self._prefetch_state is None:
            raise RuntimeError("PLE prefetch state is missing")
        embeddings, semantic_tokens, physical_tokens = self._prefetch_state
        # fnFL2 H35: the join is timed as ``ple.wait`` (SGLANG_DEBUG_DECODE_PLE_WAIT).
        wait_for_ple_prefetch(self._prefetch_stream)
        embeddings = self.ple_embedding.ngram_embedding.reduce(embeddings)
        embeddings = embeddings * self.ple_embedding.ngram_embedding.weight_scale
        embeddings = self.ple_embedding._finish_embedding_lookup(
            embeddings,
            semantic_tokens,
            forward_batch,
            physical_tokens,
        )
        self._prefetch_state = None
        return embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        batch: _PLEBatch,
    ) -> torch.Tensor:
        hidden_states = hidden_states[: batch.processed_tokens]
        if self._prefetch_state is not None:
            embeddings = self._consume_prefetched_embeddings(forward_batch)
        else:
            with ple_wait_scope():
                embeddings = self.ple_embedding(batch, forward_batch)
        key, _ = self.key_proj(embeddings)
        value, _ = self.value_proj(embeddings)
        token_count = hidden_states.shape[0]
        hidden_size = self.hidden_size
        hc_count = self.hc_count
        if hidden_states.shape[-1] != hc_count * hidden_size:
            raise RuntimeError(
                "PLE hidden size does not match its hyper-connection layout: "
                f"expected {hc_count * hidden_size}, got {hidden_states.shape[-1]}"
            )
        key = key.reshape(token_count, hc_count, hidden_size)
        query = hidden_states.reshape(token_count, hc_count, hidden_size)
        key_normed = self._apply_ple_norm(self.norm_key, key)
        query_normed = self._apply_ple_norm(self.norm_query, query)
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True)
        gate = gate / math.sqrt(hidden_size)
        fused_gate_value = False
        if batch.use_decode_fast_path:
            from sglang.kernels.ops.qwen4_ple import (
                can_fuse_qwen4_gate_value,
                fused_qwen4_gate_value,
            )

            fused_gate_value = can_fuse_qwen4_gate_value(gate, value)
        if fused_gate_value:
            gated_value = fused_qwen4_gate_value(gate, value)
        else:
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = torch.sigmoid(gate)
            gated_value = gate * value.unsqueeze(-2)
        gated_value_normed = self._apply_ple_norm(self.norm_conv, gated_value)
        gated_value = gated_value.flatten(-2)
        gated_value_normed = gated_value_normed.flatten(-2)
        conv_output = self._short_conv(
            gated_value_normed,
            forward_batch,
            batch,
        )
        output = gated_value + conv_output
        if not batch.use_decode_fast_path:
            output = torch.where(
                batch.valid_tokens.unsqueeze(-1),
                output,
                torch.zeros_like(output),
            )
        return _pad_token_rows(output, batch.physical_tokens)


class Qwen4ExpLayerExtensionMixin:
    def _init_qwen4_exp_layer_extensions(
        self,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.ple = None

        for attr_name in (
            "input_layernorm",
            "post_attention_layernorm",
            "layer_communicator",
        ):
            if hasattr(self, attr_name):
                delattr(self, attr_name)

        if (layer_id + 1) in config.ple_layer_ids:
            ple_layer_ids_sorted = sorted(set(config.ple_layer_ids))
            ple_layer_index = {
                abs_id: index for index, abs_id in enumerate(ple_layer_ids_sorted)
            }[layer_id + 1]
            # Strip the block-type segment like the dense mlp (PLE is attn's sibling);
            # else the quant prefix misses the ckpt skip-list -> NaN.
            ple_prefix = prefix.replace(".linear_attn", "").replace(".self_attn", "")
            # FORM A (F3, construction half): a worker holds experts and
            # nothing else, so the PLE is never built there -- which is what
            # actually keeps its parameters off the card. Inert on a classic
            # boot (skip_on_worker returns None when no role plan is
            # installed), so the default path constructs exactly as before.
            _ple_ph = skip_on_worker("ple", ple_prefix)
            self.ple = _ple_ph if _ple_ph is not None else Qwen4ExpPLELayer(
                config,
                quant_config=quant_config,
                prefix=f"{ple_prefix}.ple" if ple_prefix else "ple",
                layer_id=layer_id,
                ple_layer_index=ple_layer_index,
            )

        hc_config = HyperConnectionConfig(
            hc_count=self.hc_count,
            hidden_size=self.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        # Task #46 (19.09.): the mixers may stay INT8 (SGLANG_HC_MIXER_INT8);
        # the prefix is the checkpoint name, which the CT config lists verbatim.
        hc_prefix = prefix.replace(".linear_attn", "").replace(".self_attn", "")
        # FORM A (F3): the two hyper-connection mixers are the largest
        # single post a worker carries for nothing -- they are NOT sharded
        # (layers/hyperconnection.py builds plain nn.Linear), so every rank
        # holds them in full: 0.63 GiB per rank in INT8, 1.19 in BF16
        # (measured, boot fn8ah). Skipping their construction is where that
        # VRAM actually comes back.
        _attn_hc_ph = skip_on_worker("hyper_connection", f"{hc_prefix}.attn")
        self.attn_hyper_connection = _attn_hc_ph if _attn_hc_ph is not None else GatedResidual(
            hc_config,
            use_mix=True,
            use_combine=True,
            quant_config=quant_config,
            prefix=f"{hc_prefix}.attn_hyper_connection" if hc_prefix else "attn_hyper_connection",
        )
        _mlp_hc_ph = skip_on_worker("hyper_connection", f"{hc_prefix}.mlp")
        self.mlp_hyper_connection = _mlp_hc_ph if _mlp_hc_ph is not None else GatedResidual(
            hc_config,
            use_mix=True,
            use_combine=True,
            quant_config=quant_config,
            prefix=f"{hc_prefix}.mlp_hyper_connection" if hc_prefix else "mlp_hyper_connection",
        )

    def _prepare_qwen4_exp_attn(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        *,
        ple_batch: Optional[_PLEBatch],
    ):
        hc_dim = self.hc_count * self.hidden_size
        if hidden_states.shape[-1] != hc_dim:
            assert hidden_states.shape[-1] == self.hidden_size
            hidden_states = torch.cat(
                [hidden_states for _ in range(self.hc_count)], dim=-1
            )

        if self.ple is not None:
            fwd_mark("other")
            if ple_batch is None:
                if not _get_ple_forward_mode(forward_batch).is_idle():
                    raise RuntimeError(
                        "non-idle Qwen4 PLE forward is missing its batch"
                    )
                self.ple.forward_idle(forward_batch)
            else:
                ple_query = (
                    hidden_states if residual is None else hidden_states + residual
                )
                hidden_states = hidden_states + self.ple(
                    ple_query, forward_batch, ple_batch
                )
            fwd_mark("ple")

        hidden_states, residual = self.attn_hyper_connection.mix(hidden_states)
        fwd_mark("hc")
        return hidden_states, residual

    def _prepare_qwen4_exp_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ):
        # fnFL2 H20: the segment ending here is the attention block's output
        # side -- o_proj / GDN norm + out_proj, the output gate.
        fwd_mark("dense")
        # FORM A (F12): see LinearBase.reduce -- unsharded dense means no
        # attn-TP partial to all-reduce.
        if not forward_batch.forward_mode.is_idle() and not form_a_dense_is_unsharded():
            hidden_states = attn_tp_all_reduce(hidden_states)
        hidden_states = self.attn_hyper_connection.combine(hidden_states, residual)
        hidden_states, residual = self.mlp_hyper_connection.mix(hidden_states)
        fwd_mark("hc")
        return hidden_states, residual

    def _qwen4_exp_use_dp_moe_gather(self) -> bool:
        return get_attention_dp_size() > 1 and get_moe_a2a_backend().is_none()

    def _qwen4_exp_use_attn_tp_a2a_scatter(self) -> bool:
        return get_parallel().attn_tp_size > 1 and not get_moe_a2a_backend().is_none()

    def _run_qwen4_exp_mlp(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if not self.config.num_experts:
            return self.mlp(hidden_states)

        use_dp_moe_gather = self._qwen4_exp_use_dp_moe_gather()
        use_attn_tp_a2a_scatter = self._qwen4_exp_use_attn_tp_a2a_scatter()

        if use_dp_moe_gather:
            hidden_states, local_hidden_states = (
                get_global_dp_buffer(get_tp_group()),
                hidden_states,
            )
            dp_gather_replicate(hidden_states, local_hidden_states, forward_batch)
        elif hidden_states.shape[0] == 0 and get_moe_a2a_backend().is_none():
            # Only safe to short-circuit an empty batch when the MoE holds no collective;
            # under deepep an idle DP rank must still join dispatch/combine or peers hang.
            return hidden_states

        attn_tp_chunks = None
        if use_attn_tp_a2a_scatter:
            attn_tp_size = get_parallel().attn_tp_size
            attn_tp_chunks = list(hidden_states.tensor_split(attn_tp_size))
            hidden_states = attn_tp_chunks[get_parallel().attn_tp_rank].contiguous()

        # FORM A (slice 6a): this is where the host hands the MoE input to
        # the ranks that have no dense chain to compute it with. On a
        # classic boot every rank produced the identical value itself (the
        # dense path ends in an all-reduce), so no carrier is needed and
        # this call is skipped entirely. Under Form A the value exists on
        # ONE card and the workers contribute zeros to the same collective,
        # which makes it a broadcast without a new transport.
        # form_a_worker_forward's docstring carries the count correction
        # this implies (96 collectives per round, not 48).
        # fnFA11 (20.09.): NOT for the solo MTP draft. The draft is the
        # host's alone (--speculative-draft-placement solo) with every expert
        # resident there (SGLANG_MOE_OFFLOAD_EXCLUDE_DRAFT); the workers run
        # no draft forward at all, so a carrier collective here has no
        # second participant -- fnFA11 died in exactly that all-reduce.
        if form_a_dense_is_unsharded() and not getattr(self, "is_nextn", False):
            hidden_states = publish_moe_input(hidden_states)

        mlp_in = hidden_states if _nan_guard_on() else None
        hidden_states = self.mlp(hidden_states, forward_batch)
        # fnFL2 H20: the routed experts' tail (top-k combine, shared add).
        fwd_mark("moe_apply")

        # Task #49 (19.09.): SGLANG_NAN_GUARD=1 names the first layer whose
        # MoE output stops being finite (the '!!!' = token-0 answers at 259k);
        # on a hit the discriminator says input / transient / persistent.
        if _nan_guard_on():
            ok = _nan_check("mlp_out", hidden_states, getattr(self, "layer_id", None), forward_batch)
            if not ok:
                _nan_discriminate(self, mlp_in, hidden_states, forward_batch)

        if use_dp_moe_gather:
            hidden_states, global_hidden_states = (
                get_local_dp_buffer(get_tp_group()),
                hidden_states,
            )
            if should_use_dp_reduce_scatterv():
                get_tp_group().reduce_scatterv(
                    global_hidden_states,
                    output=hidden_states,
                    sizes=get_dp_global_num_tokens(),
                )
            else:
                dp_scatter(hidden_states, global_hidden_states, forward_batch)
        elif use_attn_tp_a2a_scatter and not form_a_dense_is_unsharded():
            # FORM A (F12): nothing to gather -- this rank produced every
            # q packet itself.
            assert attn_tp_chunks is not None
            gathered = [torch.empty_like(t) for t in attn_tp_chunks]
            attn_tp_all_gather(gathered, hidden_states.contiguous())
            hidden_states = torch.cat(gathered)

        return hidden_states

    def _postprocess_qwen4_exp_layer(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ):
        hidden_states = self.mlp_hyper_connection.combine(hidden_states, residual)
        fwd_mark("hc")
        return hidden_states, None


class Qwen4ExpLinearDecoderLayer(
    Qwen4ExpLayerExtensionMixin, Qwen3_5LinearDecoderLayer
):
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ) -> None:
        super().__init__(config, layer_id, quant_config, prefix, alt_stream, is_nextn)
        self.is_nextn = bool(is_nextn)
        self._init_qwen4_exp_layer_extensions(config, layer_id, quant_config, prefix)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        **kwargs,
    ):
        forward_batch = kwargs.get("forward_batch", None)

        hidden_states, residual = self._prepare_qwen4_exp_attn(
            hidden_states,
            residual,
            forward_batch,
            ple_batch=kwargs.get("ple_batch"),
        )

        if not forward_batch.forward_mode.is_idle():
            hidden_states = self.linear_attn(hidden_states, forward_batch)

        hidden_states, residual = self._prepare_qwen4_exp_mlp(
            hidden_states, residual, forward_batch
        )
        hidden_states = self._run_qwen4_exp_mlp(hidden_states, forward_batch)
        return self._postprocess_qwen4_exp_layer(hidden_states, residual, forward_batch)


class Qwen4ExpAttentionDecoderLayer(
    Qwen4ExpLayerExtensionMixin, Qwen3_5AttentionDecoderLayer
):
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ) -> None:
        config.attn_output_gate = True
        super().__init__(config, layer_id, quant_config, prefix, alt_stream, is_nextn)
        self.is_nextn = bool(is_nextn)
        from sglang.srt.layers.attention.qsa.config import is_qwen_qsa
        from sglang.srt.layers.attention.qsa.glue import build_qsa_indexer

        self.is_qsa = is_qwen_qsa(config)
        # FORM A (F3): the QSA indexer is part of the attention host's
        # attention -- it selects the sparse rows for it. A worker runs no
        # attention at all, so it builds none.
        _idx_ph = skip_on_worker("self_attn", f"{prefix}.indexer")
        if self.is_qsa and _idx_ph is not None:
            self.indexer = _idx_ph
        elif self.is_qsa:
            self.indexer = build_qsa_indexer(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=f"{prefix}.indexer" if prefix else "indexer",
                rotary_emb=self.rotary_emb,
            )
        self._init_qwen4_exp_layer_extensions(config, layer_id, quant_config, prefix)

    def _compute_qsa_topk_indices(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.qsa.glue import (
            get_qsa_indexer_metadata,
            resolve_qsa_sparse_backend,
        )

        # fnFL2 H20: q/k/v projection, q/k norm and rope end here.
        fwd_mark("dense")
        backend = get_attn_backend()
        sparse_backend = resolve_qsa_sparse_backend(backend)
        should_reuse = getattr(sparse_backend, "should_reuse_mtp_sparse_indices", None)
        if should_reuse is not None and should_reuse(forward_batch):
            # MTP decode steps reuse the draft-extend's target-aligned
            # selection; the indexer never runs inside the decode graph.
            return sparse_backend.lookup_mtp_sparse_indices(
                forward_batch, self.layer_id
            )
        indexer_metadata = get_qsa_indexer_metadata(
            backend, self.layer_id, forward_batch
        )
        topk_indices = self.indexer(
            hidden_states,
            positions,
            forward_batch,
            indexer_metadata,
        )
        should_capture = getattr(
            sparse_backend, "should_capture_mtp_sparse_indices", None
        )
        if should_capture is not None and should_capture(forward_batch):
            sparse_backend.capture_mtp_sparse_indices(
                topk_indices, forward_batch, self.layer_id, metadata=indexer_metadata
            )
        fwd_mark("qsa_idx")
        return topk_indices

    def self_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        overlap_indexer = (
            self.is_qsa
            and self.alt_stream is not None
            and get_is_capture_mode()
            and hidden_states.shape[0] < _QSA_INDEXER_OVERLAP_TOKEN_THRESHOLD
        )
        attention_kwargs = {}
        if overlap_indexer:
            # Safe to overlap: the indexer reads only hidden_states/positions,
            # and writes QSA-private pool buffers.
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)

        q, k, v, gate = self._prepare_qkv_gate(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        if overlap_indexer:
            with torch.cuda.stream(self.alt_stream):
                topk_indices = self._compute_qsa_topk_indices(
                    hidden_states, positions, forward_batch
                )
            current_stream.wait_stream(self.alt_stream)
            # Allocated on alt_stream, consumed by attention on the current
            # stream; tell the caching allocator before alt_stream is reused.
            topk_indices.record_stream(current_stream)
            attention_kwargs["topk_indices"] = topk_indices
        elif self.is_qsa:
            attention_kwargs["topk_indices"] = self._compute_qsa_topk_indices(
                hidden_states, positions, forward_batch
            )

        attn_output = self.attn(q, k, v, forward_batch, **attention_kwargs)
        if gate is not None:
            if attn_output.is_cuda:
                # The strided 3D gate view feeds the kernel directly, so the
                # gate reshape copy disappears along with the sigmoid + mul.
                attn_output = fused_sigmoid_mul(attn_output, gate, inplace=True)
            else:
                gate = gate.reshape(gate.shape[0], -1) if gate.ndim == 3 else gate
                attn_output = attn_output * torch.sigmoid(gate)
        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ):
        hidden_states, residual = self._prepare_qwen4_exp_attn(
            hidden_states,
            residual,
            forward_batch,
            ple_batch=kwargs.get("ple_batch"),
        )

        if not forward_batch.forward_mode.is_idle():
            hidden_states = self.self_attention(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        hidden_states, residual = self._prepare_qwen4_exp_mlp(
            hidden_states, residual, forward_batch
        )
        hidden_states = self._run_qwen4_exp_mlp(hidden_states, forward_batch)
        return self._postprocess_qwen4_exp_layer(hidden_states, residual, forward_batch)


ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen4ExpAttentionDecoderLayer,
    "full_attention": Qwen4ExpAttentionDecoderLayer,
    "linear_attention": Qwen4ExpLinearDecoderLayer,
}


class Qwen4ExpModel(Qwen3_5ForCausalLM):
    decoder_layer_types = ALL_DECODER_LAYER_TYPES

    def _build_embed_tokens(
        self,
        config: Qwen4ExpTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> nn.Module:
        # Minachist's AutoRound export packs embed_tokens (INT8 g128, group_3
        # `re:.*embed_tokens`) -- the vocab gets the quant_config only when a
        # config group NAMES it. cyankiwi (targets [Linear], embedding dense
        # and, being no Linear, absent from the ignore list) keeps the dense
        # embedding it always had; the ignore-only vocab rule would have
        # called it quantized.
        if not self.pp_group.is_first_rank:
            # WP5: like the base backbone -- only the first pipeline stage
            # embeds (the meta probe built a full embedding on every stage).
            return PPMissingLayer()
        name = add_prefix("embed_tokens", prefix)
        # FORM A (F3 / F13): the vocabulary is the host's. A worker never
        # embeds anything -- it enters the model at the MoE input of layer 0
        # and leaves at the last MoE combine.
        _emb_ph = skip_on_worker("embed_tokens", name)
        if _emb_ph is not None:
            return _emb_ph
        # fnFL2 H1b: the D drafter shares the TARGET's (INT8-packed) table;
        # its own BF16 one was never loaded and cost 1212.5 MiB of the
        # weights_draft tag (mtp_vocab_share.py). Decided in the base __init__.
        if self._defer_embed:
            return MtpEmbedDeferred()
        # #66: an MTP build hands its UNMODIFIED config in here (see
        # __init__); every other caller passes none and keeps its own.
        vocab_config = getattr(self, "_embed_quant_config", None) or quant_config
        raw = getattr(vocab_config, "config", None)
        vocab_quant = (
            vocab_config
            if isinstance(raw, dict) and vocab_named_in_targets(raw, name)
            else None
        )
        return VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            quant_config=vocab_quant,
            prefix=name,
            use_attn_tp_group=is_dp_attention_enabled(),
            # FORM A (F13): the vocab family deliberately does NOT inherit
            # the base ratio vector (distributed/utils.py:1734
            # tp_vocab_ratios, "vocab always even"), so under a Form A plan
            # the host would still hold one third of the rows and
            # all-reduce the embedding with two ranks that hold none. The
            # sharding has to fall the same way the dense sharding does
            # (F12): enable_tp=False gives tp_size=1 here -- full vocab, no
            # mask, no collective.
            enable_tp=not form_a_dense_is_unsharded(),
        )

    def _build_hyper_connection_mixer(self, hc_config) -> nn.Module:
        # fnFL2x21 (2026-09-23): like the vocab (first stage only), the
        # model-level mixer belongs to the LAST pipeline stage -- forward()
        # hands every other stage's stream on before `mix`. Built everywhere it
        # was a dead replica on PP0/PP1, and the flip join, which takes the
        # first stage that publishes a replicated name as its holder, moved the
        # bytes to PP0: the one stage that mixes (PP2) got nothing at a D->P
        # wake, and PP1's copy had no source at the P->D sleep (W106).
        if not self.pp_group.is_last_rank:
            return PPMissingLayer()
        # FORM A (F3): the model-level mixer, like the two per-layer ones,
        # is not sharded at all -- every rank held it in full. A worker
        # enters the model at a MoE input and never touches it.
        _mix_ph = skip_on_worker("hyper_connection", "hyper_connection_mixer")
        if _mix_ph is not None:
            return _mix_ph
        return GatedResidual(hc_config, use_combine=False)

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        is_nextn: bool = False,
        embed_quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        # #66 (fnFL2v67): a caller may override the config the VOCAB is built
        # under. The draft-KV producer needs that: under placement A the
        # vocab rows come from the TARGET checkpoint, and target and draft
        # disagree in opposite directions -- Minachist packs the vocab
        # (AutoRound group_3 `re:.*embed_tokens`) while the albucino MTP
        # checkpoint lists it under `ignore`. Building the draft's table from
        # the DRAFT's config therefore produced a bf16 table that no
        # `weight_packed` fits into. The producer sets this from the target
        # model before it loads (draft_kv_producer.load_resident_embedding);
        # the MTP constructor does NOT -- its own config is the draft's and
        # would be exactly the wrong one.
        # Set BEFORE super().__init__, which is what calls the builder.
        self._embed_quant_config = embed_quant_config
        super().__init__(config, quant_config, prefix, is_nextn)
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.has_ple = bool(config.ple_layer_ids)
        self.ple_ngram_size = int(config.ngram_size) if self.has_ple else None
        self.ple_ngram_eos_token_id = (
            int(config.eos_token_id) if self.ple_ngram_size is not None else None
        )
        if hasattr(self, "norm"):
            delattr(self, "norm")
        hc_config = HyperConnectionConfig(
            hc_count=self.hc_count,
            hidden_size=self.hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.hyper_connection_mixer = self._build_hyper_connection_mixer(hc_config)
        # WP5 (PP=3 prefill): the PLE batch (n-gram hashing, prefetch, commit)
        # is only built on the stage that owns a PLE layer.
        self._stage_has_ple = self.has_ple and any(
            getattr(self.layers[i], "ple", None) is not None
            for i in range(self.start_layer, self.end_layer)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        # WP5: under pipeline parallelism the activation crossing a stage
        # boundary is the hyper-connection stream (hc_count x hidden), which
        # the decoder layers pass along as ``hidden_states`` with a None
        # residual (see _postprocess_qwen4_exp_layer). The first stage embeds;
        # every later stage takes the stream out of the proxy.
        if self.pp_group.is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
        elif pp_proxy_tensors is not None:
            hidden_states = pp_proxy_tensors["hidden_states"]
        else:
            raise AssertionError(
                "a non-first PP stage of Qwen4-Exp entered the forward without "
                "pp_proxy_tensors: the hyper-connection stream of the previous "
                "stage did not arrive."
            )
        fwd_mark("embed")

        ple_batch = (
            _prepare_ple_batch(
                input_ids,
                forward_batch,
                ngram_size=self.ple_ngram_size,
                ngram_eos_token_id=self.ple_ngram_eos_token_id,
            )
            if self._stage_has_ple
            else None
        )
        fwd_mark("ple")
        residual = None
        aux_hidden_states = []
        # H13: SGLANG_DEBUG_HOST_ANON_PROBE -- one pass per model forward, a
        # RssAnon checkpoint before every decoder layer (no-op when off).
        _hap.pass_begin(
            getattr(forward_batch.forward_mode, "name", str(forward_batch.forward_mode)),
            int(hidden_states.shape[0]),
        )
        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            _hap.checkpoint("layer", layer=i)
            if i + 1 < self.end_layer:
                next_ple = getattr(self.layers[i + 1], "ple", None)
                if next_ple is not None:
                    next_ple.start_prefetch(ple_batch, forward_batch)
                    # the pread gather blocks the host here (PLE-GATHER-PREFILL)
                    fwd_mark("ple")
            with get_global_expert_distribution_recorder().with_current_layer(i):
                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                    forward_batch=forward_batch,
                    ple_batch=ple_batch,
                    captured_last_layer_outputs=(
                        aux_hidden_states
                        if getattr(layer, "_is_layer_to_capture", False)
                        else None
                    ),
                )

        if ple_batch is not None:
            _commit_ple_batch(ple_batch, forward_batch)
            fwd_mark("ple")
        _hap.pass_end()

        if not self.pp_group.is_last_rank:
            # Hand the hyper-connection stream to the next stage. The residual
            # is None between Qwen4-Exp layers, so it is not part of the proxy.
            # Flat [tokens, hc_count * hidden]: the runner sizes the proxy
            # buffer by hc_hidden_size (mHC form, as DeepSeek-V4 does).
            return PPProxyTensors({"hidden_states": hidden_states.flatten(1)})

        hc_hidden_states = hidden_states
        hidden_states, _ = self.hyper_connection_mixer.mix(hidden_states)
        if not forward_batch.forward_mode.is_idle():
            return hidden_states, hc_hidden_states

        if len(aux_hidden_states) == 0:
            return hidden_states
        return hidden_states, aux_hidden_states


class Qwen4ExpVLModel(Qwen4ExpModel):
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        self.last_hc_hidden_states = None

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[Any] = None,
        input_deepstack_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.last_hc_hidden_states = None
        # mm routine passes input_ids=None; PLE needs the real ids.
        if input_ids is None:
            input_ids = forward_batch.input_ids
        model_output = super().forward(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            inputs_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        if isinstance(model_output, tuple):
            hidden_states, self.last_hc_hidden_states = model_output
            return hidden_states
        return model_output  # a tensor, or the PPProxyTensors of a non-last stage


_LAYER_ID_RE = re.compile(r"\.layers\.(\d+)\.")

# Form A (F3): the loader veto in weight_name_needed. Imported at module
# level rather than inside the method because it runs once per checkpoint
# tensor -- 225.300 of them for this model.
from sglang.srt.form_a_construction import skip_on_worker  # noqa: E402
from sglang.srt.form_a_worker_forward import publish_moe_input  # noqa: E402
from sglang.srt.debug_utils import host_anon_probe as _hap  # noqa: E402
from sglang.srt.rank_role import (  # noqa: E402
    form_a_dense_is_unsharded,
    this_rank_is_form_a_worker,
    worker_keeps_parameter,
)


def weight_layer_is_owned(name: str, start_layer: int, end_layer: int) -> bool:
    """WP5 (pipeline parallelism): does this stage own the decoder layer a
    checkpoint tensor belongs to? Names without a ``layers.N.`` component
    (embedding, lm_head, mixers) are owned by whoever has the module -- the
    per-tensor lookups decide those. Skipping foreign layers here keeps the
    expert loader's KeyError and the generic branch's per-tensor warning
    (thousands under PP) off the load path."""
    m = _LAYER_ID_RE.search(name)
    if m is None:
        return True
    layer_id = int(m.group(1))
    return start_layer <= layer_id < end_layer


def mixer_is_foreign(model, name: str) -> bool:
    """fnFL2x21: the model-level mixer exists on the last pipeline stage only
    (``Qwen4ExpModel._build_hyper_connection_mixer``). On every other stage its
    checkpoint tensors have no module; they are skipped before
    ``load_packed_hc_linear`` would refuse them as a module-less KeyError."""
    return name.startswith("model.hyper_connection_mixer.") and isinstance(
        model.hyper_connection_mixer, PPMissingLayer
    )


_HC_PACKED_SUFFIXES = (".weight_packed", ".weight_scale", ".weight_shape")


def load_packed_hc_linear(
    pending: dict, name: str, loaded_weight: torch.Tensor, params_dict: dict
) -> bool:
    """Minachist's AutoRound export quantizes the hyper-connection mixers
    (``*_hyper_connection.input_mix_weight_down/up``, INT8 g64) which this
    line keeps as plain ``nn.Linear`` (3 MB each). The packed payload and its
    scale are collected per module in ``pending`` and, once both are there,
    widened into the dense parameter. Returns True when ``name`` was one of
    those tensors. The ``weight_shape`` tensor is implied by the module."""
    if "hyper_connection" not in name or ".input_mix_weight_" not in name:
        return False
    suffix = next((s for s in _HC_PACKED_SUFFIXES if name.endswith(s)), None)
    if suffix is None:
        return False
    module_name = name[: -len(suffix)]
    if module_name + ".weight_packed" in params_dict:
        # Task #46: the mixer is a quantized ReplicatedLinear -- its own
        # params take the packed tensors as they are, nothing is widened.
        param = params_dict.get(name)
        if param is None:
            return True  # weight_shape is implied by the module
        loader = getattr(param, "weight_loader", None)
        if loader is None:
            param.data.copy_(loaded_weight.to(param.dtype))
        else:
            loader(param, loaded_weight)
        return True
    if suffix == ".weight_shape":
        return True
    param_name = module_name + ".weight"
    if param_name not in params_dict:
        raise KeyError(f"packed hyper-connection weight without a module: {name}")
    slot = pending.setdefault(module_name, {})
    slot[suffix[1:]] = loaded_weight
    if "weight_packed" in slot and "weight_scale" in slot:
        param = params_dict[param_name]
        dense = dequantize_pack_quantized_weight(
            slot["weight_packed"], slot["weight_scale"], param.shape
        )
        param.data.copy_(dense.to(param.dtype))
        del pending[module_name]
    return True


def _transpose_in_worker() -> bool:
    """ONE switch, read by BOTH sides of the seam (#66).

    `Qwen4ExpForConditionalGeneration.weight_post_load` transposes the expert
    shards in the loader thread iff this is on; `_weight_loader_impl` skips
    its own transpose iff this is on. Read fresh each call -- the load happens
    once per process and a cached verdict would only hide a mis-set env.
    """
    return os.environ.get("SGLANG_LOAD_TRANSPOSE_IN_WORKER") == "1"


#: The checkpoint tensors the compressed-tensors WNA16 MoE path transposes:
#: the packed weights and everything that shares their [out, in/pack]
#: orientation. Names come from the checkpoint, before any mapping.
_CT_EXPERT_SUFFIXES = (
    "weight_packed",
    "weight_scale",
    "weight_zero_point",
)


#: #68f: WIE OFT hat der Lade-Worker wirklich gedreht -- je Prozess, nicht
#: je Rang-Log-Zeile. Ohne diese Zahl ist "der Schalter brachte nichts" und
#: "der Schalter griff nie" am Ende eines Boots NICHT unterscheidbar; genau
#: daran haben w58 (Env kam nicht an) und w59 (Praedikat sagte immer nein)
#: je einen Boot gekostet. Nur zaehlen, nichts halten: kein Tensor, keine
#: Referenz, kein Speicher -- ein Instrument darf das Gemessene nicht
#: festhalten.
_WORKER_TRANSPOSED = [0, 0]  # [gedreht, angeboten]


def worker_transpose_counts():
    """(gedreht, angeboten) seit Prozessstart."""
    return tuple(_WORKER_TRANSPOSED)


def _ct_expert_layer_to_transpose(name: str, model):
    """Das FusedMoE-Modul, dessen Shards dieser Worker transponieren darf --
    oder None. Traegt die Entscheidung UND ihren Adressaten, damit die
    Quittung (#68e) genau dort landet, wo der Verbraucher sie liest."""
    if not _is_ct_wna16_expert_shard(name, model):
        return None
    return _expert_layer_for_name(name, model)


def _is_ct_wna16_expert_shard(name: str, model) -> bool:
    """Is this checkpoint tensor one the WNA16 MoE path would transpose?

    Deliberately narrow: an expert tensor of a compressed-tensors checkpoint.
    A name this returns False for keeps the consumer's own transpose, so a
    miss costs speed and never correctness -- the asymmetry a switch like this
    must have.
    """
    if ".experts." not in name:
        return False
    if not name.endswith(_CT_EXPERT_SUFFIXES):
        return False
    # #68a: DIE METHODE DES LAYERS, NICHT DER TYP DES MODELLS.
    #
    # Hier stand `"CompressedTensors" in type(model.quant_config).__name__`
    # -- und genau daran starb fnFL2v87 ("The size of tensor a (2560) must
    # match the size of tensor b (80)"): der VERBRAUCHER transponiert nur
    # unter drei ganz bestimmten Layer-Methoden, dieses Praedikat traf
    # jedes compressed-tensors-Schema. Breiter als der Verbraucher heisst:
    # der Worker transponiert Tensoren, die der Verbraucher nie anfasst,
    # und die Shapes passen danach nicht mehr zusammen.
    #
    # Jetzt fragt diese Funktion DIESELBE Quelle wie der Verbraucher --
    # `ct_method_transposes` in fused_moe_triton/layer.py -- und zwar an
    # der Methode DES LAYERS, zu dem der Tensor gehoert. Findet sie den
    # Layer nicht, ist die Antwort False: ein Fehltreffer kostet dann
    # Geschwindigkeit (der Verbraucher transponiert selbst), nie
    # Korrektheit -- die Asymmetrie, die ein solcher Schalter haben muss.
    from sglang.srt.layers.moe.fused_moe_triton.layer import (
        ct_effective_method,
        ct_method_transposes,
    )

    layer = _expert_layer_for_name(name, model)
    if layer is None:
        return False
    # #68d: `ct_effective_method` statt `layer.quant_method`. Die Namen in
    # `_CT_TRANSPOSING_METHODS` sind Schema-Namen; `quant_method` allein
    # traf nie einen davon, also transponierte der Worker NIE -- und der
    # Verbraucher sprang wegen der Env trotzdem ueber seine eigene
    # Transposition (fnFL2w59, "2560 vs 80").
    return ct_method_transposes(ct_effective_method(layer))


def _expert_layer_for_name(name: str, model):
    """Das FusedMoE-Modul, zu dem dieser Checkpoint-Tensor gehoert (#68a).

    ``model.language_model.layers.7.mlp.experts.3.down_proj.weight_packed``
    -> das ``experts``-Modul von Layer 7. Ueber ``get_submodule`` auf dem
    Praefix VOR ``.experts.``, damit die Namensform an genau einer Stelle
    steht und nicht als zweite Regel neben dem Loader lebt.
    """
    marke = ".experts."
    if marke not in name:
        return None
    praefix = name.split(marke, 1)[0] + ".experts"
    for kandidat in (
        praefix.replace("model.language_model.", "model."),
        praefix,
        praefix.replace("model.", "model.language_model.", 1),
    ):
        try:
            return model.get_submodule(kandidat)
        except AttributeError:
            continue
    return None


class Qwen4ExpForConditionalGeneration(Qwen3VLForConditionalGeneration):
    packed_modules_mapping = Qwen3_5ForCausalLM.packed_modules_mapping
    hf_to_sglang_mapper = None

    @staticmethod
    def shared_experts_fusion_disable_reason(hf_config, quant_config):
        return Qwen4ExpVLModel.shared_experts_fusion_disable_reason(
            hf_config, quant_config
        )

    def __init__(
        self,
        config: Qwen4ExpConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        language_model_cls=Qwen4ExpVLModel,
    ) -> None:
        super().__init__(config, quant_config, prefix, language_model_cls)
        rope_config = getattr(self.config, "rope_parameters", None) or getattr(
            self.config, "rope_scaling", {}
        )
        self.is_mrope_enabled = (
            "mrope_section" in rope_config and not self.language_model_only
        )
        self.deepstack_visual_indexes = (
            self.visual.deepstack_visual_indexes if self.visual is not None else []
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        # fn5c 2026-09-16: ModelRunner decides PP support by
        # `"pp_proxy_tensors" in inspect.signature(model.forward).parameters`;
        # the former `*args, **kwargs` wrapper hid the base signature and PP=3
        # died at init ("Pipeline Parallel is not compatible with this model").
        # fnFL2 H20: FWD-TIMING-PREFILL spans exactly this call -- the body
        # the rank's gpu-ms event pair wraps (eager_runner -> model.forward).
        # Switch off: one env read, nothing else evaluated.
        timed = fwd_timing_on() and begin_if_timed(
            forward_mode=forward_batch.forward_mode,
            tokens=int(forward_batch.input_ids.shape[0]),
            layers=self.model.end_layer - self.model.start_layer,
        )
        try:
            output = super().forward(
                input_ids, positions, forward_batch,
                get_embedding=get_embedding, pp_proxy_tensors=pp_proxy_tensors,
            )
        except BaseException:
            if timed:
                fwd_abort()
            raise
        if timed:
            fwd_end()
        hc_hidden_states = self.model.last_hc_hidden_states
        if hc_hidden_states is not None and isinstance(output, LogitsProcessorOutput):
            output.hidden_states = hc_hidden_states
        return output

    _EXPERT_ID_RE = re.compile(r"\.experts\.(\d+)\.")

    def weight_post_load(self, name: str, tensor):
        """Transform applied IN THE LOADER THREAD, before load_weights sees
        the tensor (#66).

        WHAT AND WHY. The compressed-tensors WNA16 MoE path transposes every
        expert shard -- fused_moe_triton/layer.py, `loaded_weight.t()
        .contiguous()` -- because the checkpoint stores [out, in/pack] while
        create_weights lays the parameter out as [E, in/pack, out]. The
        transpose is NECESSARY (checked against the two layouts) and it is
        expensive in the WRONG PLACE: measured on fnFL2v84/v85 it is 43-50 %
        of the whole load, serial, on the consumer thread, while the eight
        file workers sit in threading.wait with a full buffer and the NVMe
        reports 40-60 % read load at queue depth 0,64.

        Doing it HERE costs the same microseconds on a thread that had
        nothing to do, and takes them off the critical path.

        ONE SWITCH, READ IN BOTH PLACES. The consumer skips its own transpose
        under the same env, so the work happens exactly once. A per-tensor
        marker was the alternative and is refused: a marker that gets lost on
        the way (a narrow, a clone, a dict round-trip) transposes twice and
        loads SILENTLY WRONG, which is the one failure this code must not
        have. A global switch can be read by both sides and checked by a
        test.

        DEFEKT UND VERRIEGELT (fnFL2v87, 21.09. 13:54:15Z). Eingeschaltet
        starb der Boot mit

            RuntimeError: The size of tensor a (2560) must match the size of
            tensor b (80) at non-singleton dimension 1

        also genau an dem Shape-Konflikt, den diese Naht nicht haben darf. Die
        Ursache ist das Praedikat, nicht die Idee: der Verbraucher transponiert
        NICHT jeden Tensor, den `_is_ct_wna16_expert_shard` trifft -- er hat
        Zweige, die vorher abbiegen (`_load_single_value`, `_load_w13`,
        `_load_w2`), und die Bedingung dort haengt an der METHODE DES LAYERS,
        nicht an der quant_config des Modells. Mein Praedikat ist breiter als
        seins, also transponiert der Worker Tensoren, die der Verbraucher nie
        angefasst haette.

        UND ES WAR AUCH NICHT SCHNELLER: gemessen wanderte die Zeile zwar aus
        dem Profil (layer.py:1973 verschwindet, der Verbraucher steht zu
        34,2 % in threading.wait), aber die Ladezeit STIEG -- PP2 36,7 -> 47,5
        s, PP1 50,3 -> 57,0 s. Grund: `post_load` laeuft je Datei SERIELL in
        einem Worker, die acht Worker parallelisieren nur ueber Dateien, und
        die Sliding-Window-Pipeline hat dadurch weniger Vorlauf. Der Engpass
        ist umgezogen, nicht kleiner geworden.

        Deshalb WIRFT dieser Pfad, statt still zu laden: ein Schalter, der ein
        Modell falsch laedt, muss laut sein. Wer ihn wieder einschaltet,
        braucht zuerst (1) ein Praedikat, das exakt die Tensoren trifft, die
        der Verbraucher transponiert, und (2) einen Pool, der die Keys EINER
        Datei aufteilt -- dann greifen die gemessenen 1,55-1,96x, ohne den
        Verbraucher zu blockieren (#68).
        """
        if not _transpose_in_worker():
            return tensor
        # #68a DIE VERRIEGELUNG FAELLT, WEIL IHR GRUND GEFALLEN IST.
        #
        # Sie stand hier, weil das Praedikat dieses Workers BREITER war als
        # das des Verbrauchers -- daran starb fnFL2v87 ("size of tensor a
        # (2560) must match tensor b (80)"). Seit df11022316 fragen BEIDE
        # dieselbe Funktion (`ct_method_transposes`, an der Methode DES
        # LAYERS), und ein Test stellt sie gegen dieselbe Namensliste.
        #
        # DER ZWEITE GRUND BLEIBT UND IST HIER NICHT GELOEST: "post_load
        # runs serially per file" -- diese Arbeit laeuft im Datei-Worker,
        # aber je Datei in EINEM Thread. Gemessen wurde deshalb PP2 36,7 ->
        # 47,5 s. Wer diesen Schalter einschaltet, MUSS die Ladezeit gegen
        # den ausgeschalteten Lauf messen; er ist kein Selbstlaeufer.
        #
        # DEFAULT BLEIBT AUS (`SGLANG_LOAD_TRANSPOSE_IN_WORKER` ungesetzt),
        # und der Verbraucher ueberspringt seinen eigenen Transpose unter
        # DERSELBEN Env -- die Arbeit passiert genau einmal.
        _WORKER_TRANSPOSED[1] += 1
        layer = _ct_expert_layer_to_transpose(name, self)
        if layer is None:
            return tensor
        _WORKER_TRANSPOSED[0] += 1
        # #68e DIE QUITTUNG STEHT AM LAYER, NICHT AN EINER ENV.
        #
        # Der Verbraucher ueberspringt seine eigene Transposition nur noch
        # fuer Layer, die HIER wirklich transponiert wurden. Sagt dieser
        # Worker fuer einen Tensor nein -- Layer nicht aufloesbar, Methode
        # nicht in der Liste -- dann transponiert der Verbraucher selbst.
        # Vorher las er eine prozessweite Env und uebersprang auch dann:
        # niemand transponierte, und der Boot starb an "2560 vs 80".
        from sglang.srt.layers.moe.fused_moe_triton.layer import (
            CT_WORKER_TRANSPOSED_ATTR,
        )

        setattr(layer, CT_WORKER_TRANSPOSED_ATTR, True)
        return tensor.t()

    def weight_name_needed(self, name: str):
        """Loader veto BEFORE a checkpoint tensor is read (weight_utils
        pread_safetensors_file): True = read, False = skip, "meta" = hand
        load_weights a meta tensor of the right shape (the name is all it
        needs). fn1v/fn1w 2026-09-16: the mmap loader materialised 95 GB of
        PLE shards per rank that the checkpoint backend only maps by name,
        plus every expert of every layer on every rank."""
        if "rotary_emb.inv_freq" in name or "mtp" in name:
            return False  # load_weights drops these on the floor anyway
        if "visual" in name and self.language_model_only:
            return False
        # FORM A (F3): a worker rank holds the ROUTED experts and nothing
        # else -- no attention, no GDN, no mixer, no embeddings, no lm_head,
        # no PLE, no router, no shared expert. Vetoing by NAME here rather
        # than skipping later in load_weights is the whole point: the tensor
        # is never read, so the 1.84 GiB of dense weights per worker never
        # touch the card. Inert on a classic boot -- no role plan installed
        # means this is False for every rank.
        if this_rank_is_form_a_worker() and not worker_keeps_parameter(
            name, self._num_routed_experts_for_form_a()
        ):
            return False
        local = name.replace("model.language_model.", "model.")
        lm = self.model
        start = int(getattr(lm, "start_layer", 0))
        end = int(getattr(lm, "end_layer", getattr(self.config, "num_hidden_layers", 1 << 30)))
        if not weight_layer_is_owned(local, start, end):
            return False
        if ".ngram_embedding.shard_" in name:
            emb = self._ple_ngram_embedding()
            if emb is not None and getattr(emb, "_ckpt_backend", False):
                return "meta"
            return True
        m = self._EXPERT_ID_RE.search(local)
        if m is not None:
            rng = self._owned_expert_range()
            if rng is not None:
                lo, hi = rng
                return lo <= int(m.group(1)) < hi
        return True

    def _num_routed_experts_for_form_a(self):
        """`config.num_experts`, or None when it cannot be read.

        Used only to tell a ROUTED expert from the FUSED shared expert,
        which arrives under the same ``mlp.experts.<id>.`` name with
        id == num_experts (models/qwen3_5.py:2448-2452). None means "do not
        try", which keeps the fused shared expert on a worker -- wrong, but
        visible in the census, whereas guessing a count would be wrong and
        invisible.
        """
        n = getattr(self.config, "num_experts", None)
        return int(n) if isinstance(n, int) and n > 0 else None

    def _ple_ngram_embedding(self):
        for layer in getattr(self.model, "layers", ()):
            ple = getattr(layer, "ple", None)
            if ple is not None:
                return getattr(getattr(ple, "ple_embedding", None), "ngram_embedding", None)
        return None

    def _owned_expert_range(self):
        """(lo, hi) of the experts this rank holds under the expert-index shard
        (the same on every MoE layer), None when experts are not index-sharded."""
        cached = getattr(self, "_owned_expert_range_cache", "unset")
        if cached != "unset":
            return cached
        rng = None
        for layer in getattr(self.model, "layers", ()):
            experts = getattr(getattr(layer, "mlp", None), "experts", None)
            if experts is None:
                continue
            if getattr(experts, "_gguf_expert_shard", False):
                rng = tuple(int(x) for x in experts._gguf_expert_range)
            break
        self._owned_expert_range_cache = rng
        return rng

    def _load_qwen4_exp_ple_buffer(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        buffers: dict,
        loaded_buffers: Set[str],
    ) -> bool:
        if ".ple.ple_embedding." not in name:
            return False
        buffer_name = name.rsplit(".", 1)[-1]
        if buffer_name.startswith("hashstats_"):
            return True
        if buffer_name == "token_lookup":
            return True
        if buffer_name not in {
            "layer_multipliers",
            "ngram_heads_offsets",
            "ngram_heads_vocab_sizes",
            "weight_scale",
        }:
            return False
        buffer = buffers.get(name)
        if buffer is None:
            return False
        if buffer.shape != loaded_weight.shape:
            raise ValueError(
                f"Shape mismatch for {name}: expected {tuple(buffer.shape)}, "
                f"got {tuple(loaded_weight.shape)}"
            )
        buffer.copy_(loaded_weight.to(device=buffer.device, dtype=buffer.dtype))
        loaded_buffers.add(name)
        return True

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # Checkpoints use the qwen3.5 head-first in_proj layout,
            # matching Qwen3_5GatedDeltaNet's forward, not qwen3-next's group-first.
            ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
            ("in_proj_qkvz.", "in_proj_z.", 3),
            ("in_proj_ba.", "in_proj_b.", 0),
            ("in_proj_ba.", "in_proj_a.", 1),
        ]

        num_experts = getattr(self.config, "num_experts", None)
        expert_params_mapping = (
            FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=num_experts,
            )
            if num_experts is not None
            else []
        )
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]
        from sglang.srt.model_loader.load_consumer import (
            ExpertLoadPool,
            consumer_threads,
            current_device_index,
        )

        expert_pool = ExpertLoadPool(
            consumer_threads(), device_index=current_device_index()
        )
        with expert_pool:
            loaded_params = self._load_weights_with_pool(
                weights,
                stacked_params_mapping=stacked_params_mapping,
                expert_params_mapping=expert_params_mapping,
                fused_expert_params_mapping=fused_expert_params_mapping,
                num_experts=num_experts,
                expert_pool=expert_pool,
            )
        logger.info(
            "Ladezeit-2 EXPERT-CONSUMER threads=%d submitted=%d completed=%d",
            expert_pool.threads,
            expert_pool.submitted,
            expert_pool.completed,
        )
        return loaded_params

    def _load_weights_with_pool(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
        *,
        stacked_params_mapping,
        expert_params_mapping,
        fused_expert_params_mapping,
        num_experts,
        expert_pool,
    ):
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale_inv",
            "_weight_scale_inv",
            ".input_scale_inv",
            "_input_scale_inv",
            "_weight_scale",
            "_input_scale",
        )

        def load_fused_expert_weights(
            name: str,
            params_dict: dict,
            loaded_weight: torch.Tensor,
            shard_id: str,
            num_experts: int,
        ) -> bool:
            if name not in params_dict:
                return False
            param = params_dict[name]
            weight_loader = param.weight_loader
            for expert_id in range(num_experts):
                weight_loader(
                    param,
                    loaded_weight[expert_id],
                    name,
                    shard_id,
                    expert_id,
                )
            return True

        def copy_ple_rows_to_tp_embedding(
            emb, loaded_weight: torch.Tensor, row_start: int, row_end: int
        ) -> None:
            tp_start = emb.shard_indices.org_vocab_start_index
            tp_end = emb.shard_indices.org_vocab_end_index
            ov_start = max(row_start, tp_start)
            ov_end = min(row_end, tp_end)
            if ov_start < ov_end:
                local_start = ov_start - tp_start
                src_start = ov_start - row_start
                n_rows = ov_end - ov_start
                emb.weight.data[local_start : local_start + n_rows].copy_(
                    loaded_weight[src_start : src_start + n_rows].to(
                        device=emb.weight.device, dtype=emb.weight.dtype
                    )
                )

        def load_qwen4_exp_ple_shard(name: str, loaded_weight: torch.Tensor) -> bool:
            if ".ngram_embedding.shard_" not in name:
                return False
            import re

            match = re.search(r"\.ngram_embedding\.shard_(\d+)\.weight$", name)
            if not match:
                return False
            shard_idx = int(match.group(1))
            mod_prefix = name[: name.index(".ngram_embedding.shard_")]
            ple_mod = ple_modules.get(mod_prefix)
            if ple_mod is None:
                return False
            emb = ple_mod.ngram_embedding
            if getattr(emb, "_ckpt_backend", False):
                # checkpoint backend: nothing is copied; map the shards once.
                if emb._ckpt_table is None:
                    from sglang.srt.models.qwen4_exp_ple_table import (
                        map_ple_table_from_checkpoint,
                    )
                    from sglang.srt.runtime_context import get_server_args

                    shard_size = (
                        emb.org_vocab_size + ple_num_sync_shards - 1
                    ) // ple_num_sync_shards
                    emb.attach_checkpoint_table(
                        map_ple_table_from_checkpoint(
                            get_server_args().model_path,
                            name[: name.index(".shard_")],
                            shard_rows=shard_size,
                            total_rows=emb.org_vocab_size,
                            embedding_dim=emb.embedding_dim,
                        )
                    )
                loaded_shard_params.add(f"{mod_prefix}.ngram_embedding.weight")
                return True
            if (
                loaded_weight.dtype == torch.float8_e4m3fn
                and emb.weight.dtype != torch.float8_e4m3fn
            ):
                if isinstance(emb, Qwen4ExpPinnedHostEmbedding):
                    # offload gathers from pinned host memory; a swapped-in
                    # pageable tensor would fault in the Triton kernel.
                    raise ValueError(
                        "fp8 PLE auto-switch is unsupported with "
                        "ple_offload_embedding; set "
                        'text_config.ple_embedding_dtype="float8_e4m3fn" instead'
                    )
                logger.info(
                    "PLE embedding switched to fp8 storage: %s (%s)",
                    mod_prefix,
                    tuple(emb.weight.data.shape),
                )
                old_weight_data = emb.weight.data
                # StartupWeightLoadManager enforces tensor identity/dtype; this
                # swap breaks that contract if the model is ever enrolled.
                emb.weight = torch.nn.Parameter(
                    torch.empty_like(old_weight_data, dtype=torch.float8_e4m3fn),
                    requires_grad=False,
                )
                del old_weight_data
                # params_dict was snapshotted before the loop; drop the stale
                # entry or it pins the old bf16 storage until load end.
                params_dict.pop(f"{mod_prefix}.ngram_embedding.weight", None)
                torch.cuda.empty_cache()
            if (
                emb.weight.dtype == torch.float8_e4m3fn
                and loaded_weight.dtype != torch.float8_e4m3fn
            ):
                if not getattr(load_qwen4_exp_ple_shard, "_warned_downcast", False):
                    load_qwen4_exp_ple_shard._warned_downcast = True
                    logger.warning(
                        "PLE checkpoint shards are %s but the embedding storage "
                        "is fp8 (ple_embedding_dtype / fp8 quant config); "
                        "downcasting is lossy",
                        loaded_weight.dtype,
                    )
            shard_size = (
                emb.org_vocab_size + ple_num_sync_shards - 1
            ) // ple_num_sync_shards
            shard_start = shard_idx * shard_size
            actual_rows = loaded_weight.shape[0]
            shard_end = shard_start + actual_rows
            copy_ple_rows_to_tp_embedding(emb, loaded_weight, shard_start, shard_end)
            loaded_shard_params.add(f"{mod_prefix}.ngram_embedding.weight")
            return True

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        buffers = dict(self.named_buffers())

        ple_modules = {
            mod_name: mod
            for mod_name, mod in self.named_modules()
            if isinstance(mod, Qwen4ExpNGramEmbedding)
        }
        text_config = getattr(self.config, "text_config", self.config)
        ple_num_sync_shards = int(
            getattr(
                text_config,
                "split_ngram_parts",
                getattr(self.config, "split_ngram_parts", 512),
            )
        )
        loaded_params: Set[str] = set()
        loaded_buffers: Set[str] = set()
        loaded_shard_params: Set[str] = set()
        skipped_visual_count = 0
        hc_packed_pending: dict = {}
        # WP5: this stage's decoder layer range (whole model when PP is off).
        pp_start_layer = int(getattr(self.model, "start_layer", 0))
        pp_end_layer = int(
            getattr(self.model, "end_layer", getattr(self.config, "num_hidden_layers", 1 << 30))
        )
        skipped_foreign_layer_count = 0

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mtp" in name:
                continue
            if "visual" in name and self.language_model_only:
                skipped_visual_count += 1
                continue
            if "language_model" in name:
                name = name.replace("model.language_model.", "model.")
            if not weight_layer_is_owned(name, pp_start_layer, pp_end_layer):
                skipped_foreign_layer_count += 1
                continue
            if mixer_is_foreign(self.model, name):
                continue
            if ".self_attn." in name:
                name = name.replace(".self_attn", "")
            if name.endswith(".k_proj.k_scale"):
                name = name.replace(".k_proj.k_scale", ".attn.k_scale")
            elif name.endswith(".v_proj.v_scale"):
                name = name.replace(".v_proj.v_scale", ".attn.v_scale")

            if load_packed_hc_linear(hc_packed_pending, name, loaded_weight, params_dict):
                continue
            if self._load_qwen4_exp_ple_buffer(
                name, loaded_weight, buffers, loaded_buffers
            ):
                continue
            if load_qwen4_exp_ple_shard(name, loaded_weight):
                continue
            if ".ple.ple_embedding.ngram_embedding." in name and name.endswith(
                ".weight"
            ):
                raise ValueError(
                    f"unsupported PLE weight layout (expected shard_N shards): {name}"
                )

            if (
                self.config.tie_word_embeddings
                and self.pp_group.is_last_rank
                and "model.embed_tokens.weight" in name
                and "lm_head.weight" in params_dict
            ):
                lm_head_param = params_dict["lm_head.weight"]
                weight_loader = getattr(
                    lm_head_param, "weight_loader", default_weight_loader
                )
                weight_loader(lm_head_param, loaded_weight)

            layer_id = get_layer_id(name)
            if layer_id is not None and (
                layer_id < self.start_layer or layer_id >= self.end_layer
            ):
                continue

            is_fused_expert = (
                "experts.gate_up_proj" in name or "experts.down_proj" in name
            )

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "visual" in name or "mlp.experts" in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if (
                    mapped_name.endswith(ignore_suffixes)
                    and mapped_name not in params_dict
                ):
                    continue
                if mapped_name not in params_dict:
                    continue
                param = params_dict[mapped_name]
                param.weight_loader(param, loaded_weight, shard_id)
                name = mapped_name
                break
            else:
                is_expert_weight = False
                current_expert_params_mapping = (
                    fused_expert_params_mapping
                    if is_fused_expert
                    else expert_params_mapping
                )
                for mapping in current_expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    if "visual" in name or self.config.encoder_only:
                        continue
                    is_expert_weight = True
                    mapped_name = name.replace(weight_name, param_name)
                    if is_fused_expert:
                        if "experts.gate_up_proj" in name:
                            gate_weight, up_weight = loaded_weight.chunk(2, dim=-2)
                            if not load_fused_expert_weights(
                                mapped_name,
                                params_dict,
                                gate_weight,
                                "w1",
                                num_experts,
                            ):
                                raise KeyError(f"Parameter {mapped_name} not found")
                            if not load_fused_expert_weights(
                                mapped_name,
                                params_dict,
                                up_weight,
                                "w3",
                                num_experts,
                            ):
                                raise KeyError(f"Parameter {mapped_name} not found")
                        else:
                            if not load_fused_expert_weights(
                                mapped_name,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            ):
                                raise KeyError(f"Parameter {mapped_name} not found")
                    else:
                        if (
                            mapped_name.endswith(ignore_suffixes)
                            and mapped_name not in params_dict
                        ):
                            continue
                        param = params_dict[mapped_name]
                        # Ladezeit 2: the expert shards land through the
                        # consumer pool (load_consumer.py); serial when
                        # SGLANG_LOAD_CONSUMER_THREADS=0
                        expert_pool.submit(
                            param.weight_loader,
                            param,
                            loaded_weight,
                            mapped_name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    name = mapped_name
                    break
                else:
                    if is_expert_weight:
                        continue
                    if "visual" in name:
                        name = name.replace("attn.qkv.", "attn.qkv_proj.")
                        name = name.replace("model.visual.", "visual.")
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue
                    if name.endswith("_scale") and name not in params_dict:
                        # fn7r (19.09., PP=3): a stage that does not own the
                        # module sees its GROUP scales too (lm_head.weight_scale,
                        # 4.97 M elements on PP1) -- only a scalar scale is the
                        # "must be 1.0" case; a foreign module's scale is skipped.
                        if loaded_weight.numel() == 1:
                            assert abs(loaded_weight.item() - 1.0) < 1e-6, (
                                f"Expected 1.0, got {loaded_weight.item()} in skipped {name}"
                            )
                        continue
                    if name in params_dict:
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(
                            "Parameter %s not found while loading Qwen4-Exp VL weights",
                            name,
                        )
                        continue
            loaded_params.add(name)

        loaded_params.update(loaded_buffers)
        loaded_params.update(loaded_shard_params)
        if hc_packed_pending:
            raise ValueError(
                "packed hyper-connection weights without their scale (or vice versa): "
                f"{sorted(hc_packed_pending)}"
            )

        if skipped_foreign_layer_count > 0:
            logger.info(
                "[pp] Qwen4 load_weights: skipped %d tensors of layers outside "
                "this stage [%d, %d)",
                skipped_foreign_layer_count,
                pp_start_layer,
                pp_end_layer,
            )
        if skipped_visual_count > 0:
            logger.info(
                f"[language_model_only] Qwen4 load_weights: skipped "
                f"{skipped_visual_count} visual weights"
            )

        # H79: the fusion's census, one line per rank. On a Weg-2 flip rank
        # every layer that would fuse reads `reason=weg2-flip` (the flip form
        # keeps qkvz/ba in their own band storages, no 80 MiB hole per layer);
        # outside Weg-2 `tags=` names the band of each fused block and
        # `unbanded` the ones fused outside any band.
        _fuse_status = [
            module.finalize_fused_in_proj()
            for module in self.modules()
            if isinstance(module, Qwen3_5GatedDeltaNet)
        ]
        if _fuse_status:
            logger.info(
                "%s", Qwen3_5GatedDeltaNet.fused_in_proj_census_line(_fuse_status)
            )

        # #68f: der Schalter sagt, was er sollte -- diese Zahl sagt, was er
        # TAT. gedreht=0 bei eingeschaltetem Schalter heisst: das Praedikat
        # hat fuer jeden Tensor nein gesagt (w59) oder die Env kam nicht an
        # (w58); die Ladezeit misst dann die ALTE Form, nicht die neue.
        _gedreht, _angeboten = worker_transpose_counts()
        logger.info(
            "#68 WORKER-TRANSPOSE: gedreht=%d von angeboten=%d (Schalter %s)",
            _gedreht,
            _angeboten,
            "an" if _transpose_in_worker() else "aus",
        )

        return loaded_params

    def precompile_kernels_after_loading(self) -> None:
        from sglang.srt.layers.quantization.unquant import precompile_splitk_tactics

        if precompile_splitk_tactics():
            logger.info("Precompiled BF16 split-K GEMM tactics for Qwen4-Exp")

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        text_config = getattr(config, "text_config", config)
        if getattr(text_config, "num_experts", None) is None:
            return None
        return ModelConfigForExpertLocation(
            num_layers=text_config.num_hidden_layers,
            num_logical_experts=text_config.num_experts,
            num_groups=None,
        )


EntryClass = [Qwen4ExpForConditionalGeneration]
