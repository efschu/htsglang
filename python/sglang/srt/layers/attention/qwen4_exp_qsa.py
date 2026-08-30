"""Qwen Sparse Attention (QSA) block for Qwen4-Exp, with a dense reference.

Upstream (``qwen4-main-squashed``) already ships the whole sparse stack --
``qwen_sparse_attn_backend.QwenSparseAttnBackend`` plus ``layers/attention/qsa/``
(indexer, MQA logits, block top-k, Triton sparse GQA).  Nothing here
reimplements any of it.  This module supplies the two things the fork needs on
top of it:

1.  **The seam.**  Upstream's ``Qwen4ExpAttentionDecoderLayer.self_attention``
    calls ``self._prepare_qkv_gate(...)`` and then passes ``topk_indices`` into
    ``self.attn``.  The fork's ``models/qwen3_5.Qwen3_5AttentionDecoderLayer``
    has neither: it exposes ``forward_prepare_{cuda_fused,fused_gate,native,
    npu}`` and its ``self_attention`` calls ``self.attn(q, k, v, forward_batch)``
    with no kwargs, so it can never hand the backend a selection.  This module
    is a self-contained attention block that closes that gap.

2.  **A dense correctness reference.**  ``SGLANG_QSA_MODE=dense`` (the default)
    attends to *everything* through the tree's ordinary attention backend and
    discards the indexer's selection.  Dense is a strict numerical SUPERSET of
    sparse -- attending to all blocks rather than the selected 512 -- so any
    end-to-end discrepancy between the two modes is a defect in the sparse
    lane, never in the model.  Dense is the trustworthy mode; sparse is the
    fast one.  Dense still runs the indexer projections, so every checkpoint
    tensor is loaded and exercised and a mode flip is a pure A/B.

Checkpoint contract (measured from the real safetensors header of
``model-00023-of-00038.safetensors``, ``layers.3.self_attn.*``, all BF16)::

    q_proj.weight                   (12288, 2560)   24 heads x 256 x (q, gate)
    k_proj.weight                   (  512, 2560)    2 kv heads x 256
    v_proj.weight                   (  512, 2560)
    o_proj.weight                   ( 2560, 6144)   in = 24 x 256
    q_norm.weight                   (  256,)
    k_norm.weight                   (  256,)
    indexer.index_qk_proj.weight    (  640, 2560)   (4 q + 1 kv) x 128
    indexer.q_layernorm.weight      (  128,)
    indexer.k_layernorm.weight      (  128,)

q/k/v are consumed through the usual fused ``qkv_proj`` via
:attr:`Qwen4ExpSparseAttention.stacked_params_mapping`; the other six load
name-for-name.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, NamedTuple, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa.glue import (
    build_qsa_indexer,
    get_qsa_indexer_metadata,
    resolve_qsa_sparse_backend,
)
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------

#: Attend to the full causal context through the tree's ordinary attention
#: backend and DISCARD the indexer's block selection.  Numerically a superset
#: of :data:`QSA_MODE_SPARSE`; this is the correctness reference.
QSA_MODE_DENSE = "dense"

#: Attend only to the indexer-selected micro-blocks, through upstream's
#: ``QwenSparseAttnBackend``.  Faster, and the mode the checkpoint was trained
#: for, but it depends on kernels whose arch coverage and long-context
#: correctness are audited in :func:`resolve_qsa_route`.
QSA_MODE_SPARSE = "sparse"

_QSA_MODES = (QSA_MODE_DENSE, QSA_MODE_SPARSE)

#: ``q_proj`` row layout.  The 12288 rows are 24 groups of 512, each group
#: being that head's 256 query rows followed by its 256 sigmoid-gate rows --
#: NOT [all 24 q heads][all 24 gate heads].
#:
#: DESK-PROVEN twice, independently:
#:   * From checkpoint bytes.  Per-256-row-block RMS over the real
#:     ``layers.3.self_attn.q_proj.weight`` separates the 48 blocks into two
#:     populations by PARITY (Welch t = 8.98, df 31.5) and not at all by half
#:     (t = 0.45).  Which parity is Q is settled by ``partial_rotary_factor``:
#:     only the first 64 of each head's 256 query dims are rotated, so a Q
#:     block has a structural boundary at index 64 and a gate block does not --
#:     the rope-slice / non-rope-slice row-RMS ratio has 129x more variance
#:     across EVEN blocks (sd 0.171) than ODD ones (sd 0.0150, pinned at
#:     1.000).  Even = Q, odd = gate.
#:   * From code.  ``models/qwen3_5.py:1172-1175`` does
#:     ``q_gate.view(*shape, num_heads, -1)`` then
#:     ``q, gate = torch.chunk(q_gate, 2, dim=-1)`` -- the same layout.
Q_GATE_SPLIT_IS_PER_HEAD = True


class QSARoute(NamedTuple):
    """Which kernels this process can actually reach for each QSA stage."""

    #: FlashInfer trtllm-gen paged decode, gated on SM100/SM120 upstream.
    trtllm_decode: bool
    #: Classic FA2 ``flash_attn_varlen_func`` -- the packed sparse-decode
    #: fallback for everything the trtllm gate rejects (notably SM86).
    flash_attn_2: bool
    #: ``sglang.srt.utils.is_sm120`` exists.  Upstream's
    #: ``_resolve_trtllm_sparse_decode`` imports it by name; the fork only has
    #: the major-only ``is_sm120_supported``, so without it the sparse decode
    #: path raises ImportError at the first decode -- after boot.
    exact_sm120_gate: bool
    #: Human-readable reason, logged once.
    summary: str

    @property
    def sparse_decode_reachable(self) -> bool:
        if not self.exact_sm120_gate:
            return False
        return self.trtllm_decode or self.flash_attn_2


def _probe_route() -> QSARoute:
    """Answer the routability question without touching the GPU.

    Every probe here is an import or an attribute lookup.  Nothing queries a
    device, so this is safe to run on a CPU/meta construction and safe to run
    before the CUDA context exists.
    """
    import importlib.util

    def _has(module: str, attr: Optional[str] = None) -> bool:
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            return False
        if spec is None:
            return False
        if attr is None:
            return True
        try:
            return hasattr(__import__(module, fromlist=[attr]), attr)
        except Exception:
            return False

    trtllm = _has("flashinfer.decode", "trtllm_batch_decode_with_kv_cache")
    fa2 = _has("flash_attn", "flash_attn_varlen_func")

    from sglang.srt import utils as _srt_utils

    exact_gate = hasattr(_srt_utils, "is_sm120")

    parts = [
        f"trtllm_batch_decode_with_kv_cache={'yes' if trtllm else 'no'}",
        f"flash_attn(FA2).flash_attn_varlen_func={'yes' if fa2 else 'no'}",
        f"sglang.srt.utils.is_sm120={'yes' if exact_gate else 'MISSING'}",
    ]
    if not exact_gate:
        parts.append(
            "-- qwen_sparse_attn_backend._resolve_trtllm_sparse_decode does "
            "`from sglang.srt.utils import is_sm100_supported, is_sm120`; the "
            "fork has only the major-only is_sm120_supported (which also "
            "admits SM121, where upstream documents silent long-context "
            "corruption), so sparse decode dies at the first decode"
        )
    if not fa2:
        parts.append(
            "-- without classic FA2 the packed fallback resolves to the FA4 "
            "cute interface, which has no SM86 path and fails MLIR "
            "construction on consumer Blackwell (upstream #37089/#36558)"
        )
    return QSARoute(trtllm, fa2, exact_gate, "; ".join(parts))


_route_cache: Optional[QSARoute] = None
_route_logged = False


def resolve_qsa_route() -> QSARoute:
    """Cached :func:`_probe_route`, logged once per process at INFO."""
    global _route_cache, _route_logged
    if _route_cache is None:
        _route_cache = _probe_route()
    if not _route_logged:
        _route_logged = True
        logger.info("QSA kernel route: %s", _route_cache.summary)
    return _route_cache


_mode_cache: Optional[str] = None


def resolve_qsa_mode() -> str:
    """Resolve ``SGLANG_QSA_MODE``, logging the choice once with its reason."""
    global _mode_cache
    if _mode_cache is not None:
        return _mode_cache

    requested = str(envs.SGLANG_QSA_MODE.get()).strip().lower()
    explicit = envs.SGLANG_QSA_MODE.is_set()
    if requested not in _QSA_MODES:
        raise ValueError(
            f"SGLANG_QSA_MODE must be one of {_QSA_MODES}, got {requested!r}"
        )

    route = resolve_qsa_route()
    if requested == QSA_MODE_SPARSE:
        reason = (
            "requested explicitly; block-selected attention through "
            "QwenSparseAttnBackend"
        )
        if not route.sparse_decode_reachable:
            raise RuntimeError(
                "SGLANG_QSA_MODE=sparse was requested but this process cannot "
                f"reach a QSA decode kernel. {route.summary}. Either install "
                "the missing dependency or run the dense reference "
                "(SGLANG_QSA_MODE=dense)."
            )
    elif explicit:
        reason = "requested explicitly; full causal attention, selection discarded"
    else:
        reason = (
            "default; dense is a numerical superset of sparse and is the "
            "correctness reference. Sparse depends on kernels with known "
            "gaps on this class of hardware -- see resolve_qsa_route()"
        )

    logger.info("QSA mode: %s (%s)", requested, reason)
    _mode_cache = requested
    return requested


# --------------------------------------------------------------------------
# Cache accounting -- one source of truth, so the planner and the pool agree
# --------------------------------------------------------------------------


def qsa_kv_bytes_per_token(
    config: Any, num_full_attention_layers: int, dtype: torch.dtype
) -> int:
    """Bytes of ordinary K+V cache one token costs across the QSA layers.

    This is what ``self.attn`` makes the tree's KV pool allocate: it is the
    plain GQA cost, identical in both modes.  For this checkpoint
    (``num_key_value_heads`` 2, ``head_dim`` 256, 12 full-attention layers)
    that is 2 x 256 x 2 (K and V) x 2 B x 12 = **24576 B/token = 24.0 KiB**
    at BF16, 12.0 KiB at FP8 -> 6.00 / 3.00 GiB at 262,144 tokens.
    """
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(config.head_dim)
    return kv_heads * head_dim * 2 * dtype.itemsize * int(num_full_attention_layers)


def qsa_index_bytes_per_token(config: Any, num_full_attention_layers: int) -> int:
    """Bytes of INDEXER cache one token costs, in sparse mode only.

    Delegates to upstream's own
    ``QSATokenToKVPool.qsa_bytes_per_token`` rather than restating the
    formula, so the planner's number cannot drift from the pool's allocation.

    For this checkpoint (``indexer_kv_heads`` 1, ``indexer_head_dim`` 128,
    ``indexer_compress_ratio`` 4, BF16 index state, 12 layers) that is
    1 x 128 x 2 // 4 x 12 = **768 B/token = 0.1875 GiB at 262,144 tokens**.

    In :data:`QSA_MODE_DENSE` the QSA pool is never constructed and this cost
    is **zero** -- the indexer projections still run, but nothing is cached.

    Not priced here, because they are per-REQUEST rather than per-token:
    the pre-compression key ring (``num_request_slots x compress_ratio x 1 x
    128 x 2 B`` per layer = ``num_request_slots x 12288 B`` over 12 layers)
    and the rope-position buffer (``num_request_slots x 96 B``).
    """
    from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool

    return QSATokenToKVPool.qsa_bytes_per_token(
        kv_heads=int(config.indexer_kv_heads),
        head_dim=int(config.indexer_head_dim),
        compress_ratio=int(config.indexer_compress_ratio),
        num_layers=int(num_full_attention_layers),
    )


# --------------------------------------------------------------------------
# The attention block
# --------------------------------------------------------------------------


class Qwen4ExpSparseAttention(nn.Module):
    """One Qwen4-Exp ``full_attention`` layer's attention block.

    Gated GQA (24 q heads / 2 kv heads / head_dim 256, sigmoid output gate)
    with partial interleaved mrope, plus upstream's QSA indexer.  The mode
    decides only whether the indexer's selection reaches the attention
    backend; everything else is identical between the two paths, which is what
    makes them A/B-comparable.
    """

    #: Checkpoint name -> fused parameter, in the loader's usual form.  Kept
    #: here so the model file and the load-contract check read the SAME table.
    stacked_params_mapping: Tuple[Tuple[str, str, str], ...] = (
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
    )

    def __init__(
        self,
        config: Any,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = getattr(config, "text_config", config)
        self.config = config
        self.layer_id = layer_id
        self.mode = resolve_qsa_mode()

        tp_size = get_tensor_model_parallel_world_size()
        self.hidden_size = int(config.hidden_size)
        self.total_num_heads = int(config.num_attention_heads)
        self.total_num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(
            getattr(config, "head_dim", None) or self.hidden_size // self.total_num_heads
        )
        if self.total_num_heads % tp_size != 0:
            raise ValueError(
                f"Qwen4-Exp QSA needs num_attention_heads ({self.total_num_heads}) "
                f"divisible by tp_size ({tp_size})"
            )
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size != 0:
                raise ValueError(
                    f"Qwen4-Exp QSA needs num_key_value_heads "
                    f"({self.total_num_kv_heads}) divisible by tp_size ({tp_size})"
                )
        elif tp_size % self.total_num_kv_heads != 0:
            # QKVParallelLinear replicates KV heads only when tp_size is a
            # multiple of the head count; anything else mis-shards silently.
            raise ValueError(
                f"Qwen4-Exp QSA needs tp_size ({tp_size}) to be a multiple of "
                f"num_key_value_heads ({self.total_num_kv_heads}) when "
                f"replicating KV heads"
            )
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # `output_gate_type: sigmoid` in the checkpoint config; q_proj's 12288
        # rows are 2x the 6144 a plain Q would need, which is the gate.
        self.attn_output_gate = str(
            getattr(config, "output_gate_type", "sigmoid")
        ).lower() in ("sigmoid", "true")
        if not self.attn_output_gate:
            raise ValueError(
                "Qwen4-Exp ships a sigmoid output gate; q_proj is "
                "(num_heads x head_dim x 2, hidden) and cannot be loaded "
                "without it"
            )

        rope_scaling = getattr(config, "rope_parameters", None) or getattr(
            config, "rope_scaling", None
        )
        self.partial_rotary_factor = float(
            getattr(config, "partial_rotary_factor", 1.0)
        )
        # head_dim 256 x 0.25 = 64 rotary dims = 32 pairs, which is exactly
        # sum(mrope_section) = 11 + 11 + 10.  That equality is the check that
        # the partial factor and the mrope sections belong to each other;
        # MRotaryEmbedding rescales the sections if it does not hold, which
        # would silently change the rope.
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=int(getattr(config, "max_position_embeddings", 262144)),
            base=int(
                (rope_scaling or {}).get("rope_theta", None)
                or getattr(config, "rope_theta", 10000)
            ),
            rope_scaling=rope_scaling,
            partial_rotary_factor=self.partial_rotary_factor,
            is_neox_style=True,
            dtype=torch.get_default_dtype(),
        )

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            # The gate rides in the q block, so the q slot count doubles.
            self.total_num_heads * 2,
            self.total_num_kv_heads,
            bias=bool(getattr(config, "attention_bias", False)),
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=bool(getattr(config, "attention_bias", False)),
            quant_config=quant_config,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        # ZERO-CENTERED RMSNorm, i.e. `x * (1 + weight)`.  DESK-PROVEN from
        # the checkpoint: q_norm.weight has mean +0.283 with values down to
        # -0.346, and indexer.q_layernorm.weight has mean -0.037 with max
        # +0.024.  A plain `x * weight` would annihilate and sign-flip the
        # signal.  The fork's fused kernel documents the same convention
        # ("raw GemmaRMSNorm weights (kernel adds +1.0)",
        # layers/fused_qk_rmsnorm_rope_gate.py:162).
        eps = float(getattr(config, "rms_norm_eps", 1e-6))
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=eps)

        # Upstream's indexer, verbatim -- it owns index_qk_proj / q_layernorm /
        # k_layernorm and reuses THIS layer's mrope (it asserts
        # 0 < rotary_dim <= indexer_head_dim, i.e. 0 < 64 <= 128).
        self.indexer = build_qsa_indexer(
            config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("indexer", prefix),
            rotary_emb=self.rotary_emb,
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

        self._backend_checked = False

    # -- q/k/v/gate ---------------------------------------------------------

    def _prepare_qkv_gate(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv, _ = self.qkv_proj(hidden_states)
        q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)

        # Per-head [q, gate]; see Q_GATE_SPLIT_IS_PER_HEAD for the evidence.
        orig_shape = q_gate.shape[:-1]
        q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        q = q.reshape(*orig_shape, -1)

        q_by_head = self.q_norm(q.reshape(-1, self.head_dim)).view(q.shape)
        k_by_head = self.k_norm(k.reshape(-1, self.head_dim)).view(k.shape)
        q, k = self.rotary_emb(positions, q_by_head, k_by_head)
        return q, k, v, gate

    # -- indexer ------------------------------------------------------------

    def _run_indexer_projections(self, hidden_states: torch.Tensor) -> None:
        """Dense mode: exercise every indexer tensor, discard the result.

        The full indexer needs metadata that only ``QwenSparseAttnBackend``
        produces, so dense mode runs the projection half -- index_qk_proj plus
        both layernorms, i.e. all three checkpoint tensors.  This keeps weight
        loading, shapes and dtypes identical between the modes so a flip is a
        pure A/B, and it is cheap: one (T, 2560) x (2560, 640) GEMM per QSA
        layer against ~9.2 GiB of weight reads per decode token.
        """
        indexer = self.indexer
        qk, _ = indexer.index_qk_proj(hidden_states)
        split = indexer.index_n_heads * indexer.index_head_dim
        indexer.q_layernorm(qk[:, :split].reshape(-1, indexer.index_head_dim))
        indexer.k_layernorm(qk[:, split:].reshape(-1, indexer.index_head_dim))

    def _topk_indices(
        self, hidden_states: torch.Tensor, positions: torch.Tensor, forward_batch: Any
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.attention_registry import (  # noqa: F401
            attn_backend_wrapper,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch  # noqa

        backend = forward_batch.attn_backend
        sparse_backend = resolve_qsa_sparse_backend(backend)

        should_reuse = getattr(sparse_backend, "should_reuse_mtp_sparse_indices", None)
        if should_reuse is not None and should_reuse(forward_batch):
            # MTP decode steps reuse the draft-extend's target-aligned
            # selection; the indexer never runs inside the decode graph.
            return sparse_backend.lookup_mtp_sparse_indices(forward_batch, self.layer_id)

        metadata = get_qsa_indexer_metadata(backend, self.layer_id, forward_batch)
        topk_indices = self.indexer(hidden_states, positions, forward_batch, metadata)

        should_capture = getattr(
            sparse_backend, "should_capture_mtp_sparse_indices", None
        )
        if should_capture is not None and should_capture(forward_batch):
            sparse_backend.capture_mtp_sparse_indices(
                topk_indices, forward_batch, self.layer_id, metadata=metadata
            )
        return topk_indices

    def _check_backend_matches_mode(self, forward_batch: Any) -> None:
        """Refuse a mode/backend mismatch loudly, once, at the first forward.

        ``QwenSparseAttnBackend`` raises ``ValueError`` when it is handed no
        ``topk_indices``; the dense mode deliberately hands it none.  Catching
        the mismatch here names the actual cause instead of letting the
        backend report a missing argument.
        """
        if self._backend_checked:
            return
        self._backend_checked = True
        backend = resolve_qsa_sparse_backend(forward_batch.attn_backend)
        is_qsa_backend = hasattr(backend, "get_indexer_metadata") and hasattr(
            backend, "should_reuse_mtp_sparse_indices"
        )
        if self.mode == QSA_MODE_DENSE and is_qsa_backend:
            raise RuntimeError(
                "SGLANG_QSA_MODE=dense but the full-attention backend is "
                f"{type(backend).__name__}, which requires a block selection. "
                "Dense mode must run on an ordinary dense attention backend "
                "(e.g. --attention-backend fa3/triton); select the sparse "
                "backend only with SGLANG_QSA_MODE=sparse."
            )
        if self.mode == QSA_MODE_SPARSE and not is_qsa_backend:
            raise RuntimeError(
                "SGLANG_QSA_MODE=sparse but the full-attention backend is "
                f"{type(backend).__name__}, which cannot consume a block "
                "selection. Install QwenSparseAttnBackend or run dense."
            )

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: Any,
    ) -> torch.Tensor:
        self._check_backend_matches_mode(forward_batch)

        q, k, v, gate = self._prepare_qkv_gate(positions, hidden_states)

        attention_kwargs: Dict[str, Any] = {}
        if self.mode == QSA_MODE_SPARSE:
            attention_kwargs["topk_indices"] = self._topk_indices(
                hidden_states, positions, forward_batch
            )
        else:
            self._run_indexer_projections(hidden_states)

        attn_output = self.attn(q, k, v, forward_batch, **attention_kwargs)

        gate = gate.reshape(gate.shape[0], -1) if gate.ndim == 3 else gate
        attn_output = attn_output * torch.sigmoid(gate)

        output, _ = self.o_proj(attn_output)
        return output


__all__ = [
    "QSA_MODE_DENSE",
    "QSA_MODE_SPARSE",
    "QSARoute",
    "Q_GATE_SPLIT_IS_PER_HEAD",
    "Qwen4ExpSparseAttention",
    "qsa_index_bytes_per_token",
    "qsa_kv_bytes_per_token",
    "resolve_qsa_mode",
    "resolve_qsa_route",
]
