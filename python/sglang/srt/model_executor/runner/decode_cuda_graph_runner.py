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
"""DecodeCudaGraphRunner — runs DECODE / TARGET_VERIFY / DLLM_EXTEND under
a pluggable backend.

Backend selection comes from cuda_graph_config.decode:
  - "full"      — default, FullCudaGraphBackend: one
                      torch.cuda.CUDAGraph per shape.
  - "breakable" — experimental, BreakableCudaGraphBackend:
                      segmented capture (no torch.compile).
  - "tc_piecewise"     — not implemented for decode; logs a one-shot warning
                      and falls back to "full".
"""

from __future__ import annotations

import contextlib
import inspect
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Callable, Optional, Union

import torch
import tqdm
from torch.profiler import ProfilerActivity, profile

from sglang.srt.compilation import torch_compile_decoration
from sglang.srt.compilation.torch_compile_decoration import set_torch_compile_config
from sglang.srt.distributed.parallel_state import (
    graph_capture,
    set_pdmux_status,
)
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.utils.cp_utils import is_mla_prefill_cp_enabled
from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    CudaGraphBufferRegistry,
    build_decode_registry,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
    compute_local_num_token_non_padded,
    enable_num_token_non_padded,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
    BaseCudaGraphRunner,
    freeze_gc,
    get_batch_sizes_to_capture,
)
from sglang.srt.model_executor.runner.flashinfer_autotune import (
    maybe_flashinfer_autotune_speculative_draft,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.utils import resolve_decode_backend
from sglang.srt.model_executor.runner_backend_utils import (
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
)
from sglang.srt.model_executor.runner_utils.buffers import (
    DecodeInputBuffers,
)
from sglang.srt.model_executor.runner_utils.capture_mode import (
    _set_capture_lora_variant,
    model_capture_mode,
)
from sglang.srt.model_executor.runner_utils.deepep_adapter import (
    DeepEPCudaGraphRunnerAdapter,
)
from sglang.srt.multiplex.pdmux_context import get_current_stream_idx, get_stream_groups
from sglang.srt.runtime_context import get_flags, get_parallel
from sglang.srt.speculative.ragged_verify import resolve_ragged_verify_layout
from sglang.srt.utils import (
    empty_context,
    get_available_gpu_memory,
    require_attn_tp_gather,
    require_mlp_tp_gather,
)
from sglang.srt.utils.profile_utils import export_cuda_graph_capture_trace

try:
    from kt_kernel import KTMoEWrapper

    KTRANSFORMERS_AVAILABLE = True
except ImportError:
    KTRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def _register_gguf_decode_buckets(capture_bs, num_tokens_per_bs: int) -> None:
    """#163: hand the decode token-count buckets to the GGUF dispatch.

    Imported lazily so a non-GGUF deployment never pulls the module in, and
    tolerant of an sgl_kernel build without the GGUF ops (the import is the
    only thing that can fail; a failure just leaves the dispatch on raw token
    counts, which is what it used before).
    """
    try:
        from sglang.srt.layers.quantization.gguf import set_decode_token_buckets

        set_decode_token_buckets(bs * num_tokens_per_bs for bs in capture_bs)
    except Exception as e:  # pragma: no cover - optional dependency path
        logger.debug("GGUF decode-bucket registration skipped: %s", e)


def ragged_verify_compact_graphs_enabled(spec_algorithm: SpeculativeAlgorithm) -> bool:
    if not spec_algorithm.supports_ragged_verify():
        return False
    from sglang.srt.speculative.ragged_verify import ragged_verify_compact_enabled

    return ragged_verify_compact_enabled()


def build_replay_fb_view(
    forward_batch: ForwardBatch,
    buffers: DecodeInputBuffers,
    bs: int,
    raw_bs: int,
    num_tokens: int,
    seq_len_fill_value: int,
    capture_forward_mode: ForwardMode,
    is_encoder_decoder: bool,
) -> SimpleNamespace:
    """Construct a ForwardBatch-like view for backend replay-side init.

    Combines the original forward_batch (for unpadded / per-iter
    fields like spec_info, out_cache_loc, and the runtime
    actual_forward_mode) with the padded capture-time buffers from
    buffers (for req_pool_indices, seq_lens, seq_lens_cpu,
    positions, encoder_lens).

    forward_mode is the capture-time mode (used by backends for
    bucket / dispatch decisions); actual_forward_mode is the
    runtime mode (may be IDLE while the captured graph targets DECODE
    — DSV4's replay metadata prep uses this for IDLE substitution).

    Subsumes the _replay_forward_batch side channel that DSV4 used to
    read out-of-band before the init_forward_metadata 3-method ABC.
    """
    # kv-session-offload S5 spill-tick graph replay: the spill tick is a PLAIN
    # bs=1 DECODE batch (spec_algorithm=NONE) that replays the sessblk{rung}
    # DECODE graph. It MUST reach the backend as a spill tick with its REAL
    # (decode) forward_mode. Two things break otherwise, but ONLY under a
    # speculative server (the deep-offload reference runs --speculative-config
    # mtp): (1) the marker attribute kv_session_spill_tick was dropped by this
    # view, so the backend's spill-tick early-return
    # (FlashInferAttnBackend.init_forward_metadata_out_graph) never fired at
    # replay; (2) forward_mode was overwritten with capture_forward_mode
    # (=TARGET_VERIFY under MTP), so the tick was mis-dispatched into the
    # uneven-DCP target-verify path with spec_info=None -> "assert prefix_lens
    # is not None". Carry the marker and keep the tick's real forward_mode.
    # (Non-spill batches are unchanged: _spill_tick is False -> byte-identical.)
    _spill_tick = bool(getattr(forward_batch, "kv_session_spill_tick", False))
    return SimpleNamespace(
        batch_size=bs,
        forward_mode=(
            forward_batch.forward_mode if _spill_tick else capture_forward_mode
        ),
        actual_forward_mode=forward_batch.forward_mode,
        kv_session_spill_tick=_spill_tick,
        input_ids=buffers.input_ids[:num_tokens],
        positions=buffers.positions[:num_tokens],
        req_pool_indices=buffers.req_pool_indices[:bs],
        seq_lens=buffers.seq_lens[:bs],
        seq_lens_sum=(
            None
            if forward_batch.seq_lens_sum is None
            else forward_batch.seq_lens_sum + (bs - raw_bs) * seq_len_fill_value
        ),
        seq_lens_cpu=buffers.seq_lens_cpu[:bs],
        num_padding=bs - raw_bs,
        encoder_lens=buffers.encoder_lens[:bs] if is_encoder_decoder else None,
        out_cache_loc=getattr(forward_batch, "out_cache_loc", None),
        out_cache_loc_dsv4=getattr(forward_batch, "out_cache_loc_dsv4", None),
        # The mamba-track registry slot (VIRTUAL ids) is the v2p translate SOURCE
        # for the backend, which copies the result into its own static buffer and
        # reads THAT in the decode track-save — this slot is never mutated. None
        # when mamba-track is disabled.
        mamba_track_indices=getattr(buffers, "mamba_track_indices", None),
        spec_info=forward_batch.spec_info,
    )


class DecodeCudaGraphRunner(BaseCudaGraphRunner):
    """Decode-phase CUDA graph runner.

    Owns: static input buffers (DecodeInputBuffers), capture-bs list,
    attention backend, two-batch-overlap plugin, DeepEP adapter, and the
    pluggable self.backend that handles the actual capture/replay.
    """

    # #274 round 6, as CLASS attributes rather than instance ones: subclasses
    # (EAGLEDraftCudaGraphRunner and its relatives) reuse this class's
    # ``capture`` / ``_capture_one_stream`` / ``_wl_variant_label`` while
    # running their OWN __init__, so an instance-only field is missing on
    # exactly the paths that read it -- measured, a boot-killing AttributeError
    # in the serving group's draft capture. Those subclasses set the older
    # _wl_block_graph / _sess_block_graph flags by hand for the same reason;
    # defaults on the class are the version of that which cannot be forgotten
    # by the next subclass.
    # Round 7a turns the single verify entry into a LADDER: a sorted tuple of
    # candidate-row counts (K+1 per rung), one captured graph each. The active
    # field holds the rung currently in scope (an int, never 0) rather than a
    # bool, so the variant label can name it; None means no lane verify is in
    # scope. ``_lane_verify_captured`` is the SET of rungs that were actually
    # recorded -- a rung whose capture was thinned away must not be replayed.
    _lane_verify_tokens: Optional[tuple] = None
    _lane_verify_active: Optional[int] = None
    _lane_verify_captured: frozenset = frozenset()
    LANE_VERIFY_VARIANT = "lanetv"
    # Round 7a: the lane's NEXTN HEAD. Its every decode forward is an MTP
    # forward, so unlike the verify there is no second shape to protect on
    # this runner -- the flag is set for the runner's whole life and the one
    # captured entry IS the draft entry.
    _lane_draft_capture: bool = False
    _lane_draft_captured: bool = False
    _lane_draft_hidden: Optional[torch.Tensor] = None
    LANE_DRAFT_VARIANT = "lanedraft"

    def __init__(
        self,
        model_runner: ModelRunner,
        *,
        attn_backend=None,
        speculative_num_steps: Optional[int] = None,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        super().__init__(model_runner)
        # --- core state ------------------------------------------------
        self.enable_torch_compile = get_flags().capture.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.is_encoder_decoder = model_runner.model_config.is_encoder_decoder
        self.require_mlp_tp_gather = require_mlp_tp_gather(
            model_runner.server_args
        ) and not self._forward_is_dp_local(model_runner)
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)
        # Composite predicates derive from the instance values so the dp-local
        # draft exemption above stays consistent (require_gathered_buffer ==
        # mlp_tp_gather or attn_tp_gather; require_mlp_sync adds dp attention).
        self.require_gathered_buffer = (
            self.require_mlp_tp_gather or self.require_attn_tp_gather
        )
        self.require_mlp_sync = (
            model_runner.server_args.enable_dp_attention or self.require_gathered_buffer
        )
        self.enable_two_batch_overlap = (
            model_runner.server_args.enable_two_batch_overlap
        )
        self.use_ngram_embedding = model_runner.use_ngram_embedding
        if self.use_ngram_embedding:
            hf_config = model_runner.model_config.hf_config
            self.ngram_embedding_n = hf_config.ngram_embedding_n
            self.ngram_embedding_k = hf_config.ngram_embedding_k
        self.speculative_algorithm = model_runner.server_args.speculative_algorithm
        self.enable_profile_cuda_graph = (
            model_runner.server_args.enable_profile_cuda_graph
        )

        self.attn_tp_size = get_parallel().attn_tp_size
        self.attn_tp_rank = get_parallel().attn_tp_rank
        # True if a DSACPLayerCommunicator-style prefill-CP flavor is active
        # (DSA or MLA). These flavors feed a zigzag-split rank-local layout
        # into the runner; MHA-arch prefill CP (Qwen3/Qwen2 MoE via PR
        # #18233) uses the plain LayerCommunicator with an attn_tp-replicated
        # layout and is intentionally excluded so the attn_tp-local
        # num_token_non_padded adjustment still runs for it.
        self.enable_prefill_cp = (
            is_dsa_enable_prefill_cp() or is_mla_prefill_cp_enabled()
        )

        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()

        self.dllm_config = DllmConfig.from_server_args(model_runner.server_args)
        self.is_dllm = self.dllm_config is not None
        self.attn_backend = attn_backend or model_runner.attn_backend
        self.speculative_num_steps = (
            model_runner.server_args.speculative_num_steps
            if speculative_num_steps is None
            else speculative_num_steps
        )
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
            if speculative_num_draft_tokens is None
            else speculative_num_draft_tokens
        )

        # --- capture mode + tokens-per-bs ------------------------------
        self.capture_forward_mode = ForwardMode.DECODE
        self.capture_hidden_mode = CaptureHiddenMode.NULL
        self.num_tokens_per_bs = model_runner.decode_num_tokens_per_bs(
            num_draft_tokens=self.speculative_num_draft_tokens
        )
        if model_runner.spec_algorithm.is_speculative():
            if self.model_runner.is_draft_model_runner:
                # Draft workers can use TARGET_VERIFY mode.
                #
                # THROUGHOUT THIS FILE (and the sibling runners) the
                # draft/target question is asked as is_draft_MODEL_runner,
                # never is_draft_worker. The latter is a CONSTRUCTION gate
                # with three producers -- a speculative draft worker, the
                # #274 dual-group lane, and the #631 phase-flip TP stack --
                # and only the first holds draft weights. This code means
                # "am I the draft model or the target", which for the flip
                # stack is unambiguously the target: it holds the full
                # target model and captures ordinary target TARGET_VERIFY
                # graphs. Asking the construction gate sent it down the
                # draft branch and every rank died on "This should not
                # happen" -- true, and the runner was not a draft one
                # (measured, boots 16 and 17, 2026-08-08). The two other
                # producers are unaffected: for them the two flags are
                # equal by definition.
                if (
                    not self.model_runner.spec_algorithm.supports_target_verify_for_draft()
                ):
                    raise RuntimeError("This should not happen")
            self.capture_forward_mode = ForwardMode.TARGET_VERIFY
        elif self.is_dllm:
            self.capture_forward_mode = ForwardMode.DLLM_EXTEND

        # #274 round 6: the dual-group lane's chain VERIFY as an ADDITIONAL
        # capture entry, beside the lane's plain decode graphs rather than
        # instead of them.
        #
        # The lane runs its own speculation but its args view clears
        # speculative_algorithm, so the block above leaves this runner in the
        # plain-decode shape (DECODE, 1 token/bs, hidden mode NULL) and the
        # lane's TARGET_VERIFY forward misses every graph. Un-clearing the
        # algorithm would fix the verify by DELETING the plain decode entry
        # (num_tokens_per_bs becomes num_draft+1 and capture_forward_mode
        # becomes TARGET_VERIFY for the whole runner) -- and that entry is the
        # lane's no-spec path, byte-green over five rounds. So the opening is
        # targeted instead: everything above stays as it was, and the verify
        # gets its own capture pass under a temporary shape swap (the S5
        # spill-tick pattern below, run in the other direction).
        _lane_verify_rungs = getattr(
            model_runner, "dual_group_lane_verify_tokens", None
        )
        self._lane_verify_tokens: Optional[tuple] = (
            tuple(sorted({int(n) for n in _lane_verify_rungs}))
            if _lane_verify_rungs
            else None
        )
        # Set only inside a verify capture pass and inside the lane's replay
        # scope, and then to the RUNG (candidate-row count) in scope; every
        # predicate that has to tell the entries apart reads it. Default None
        # -> byte-inert for every other deployment.
        self._lane_verify_active = None
        self._lane_verify_captured = frozenset()
        # Round 7a: this runner is the lane's NEXTN head and its decode entry
        # has to carry a real EagleDraftInput. Byte-inert everywhere else
        # (False), which is every runner except the lane head's.
        self._lane_draft_capture = bool(
            getattr(model_runner, "dual_group_lane_draft_capture", False)
        )
        self._lane_draft_captured = False
        self._lane_draft_hidden = None

        # --- bucket sizes ---------------------------------------------
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(
            model_runner, self.num_tokens_per_bs
        )
        # Stage-3 MoE-offload capturable decode: the captured scratch gather
        # serves at most C spill experts per step (bs * top_k <= C invariant),
        # so only small decode buckets are capture-eligible. The env caps the
        # captured bucket list; larger decode batches fall back to the eager
        # offload path (run_waves) via the normal can_run_graph bs check.
        _moe_offload_graph_bs = envs.SGLANG_MOE_OFFLOAD_MAX_GRAPH_BS.get()
        if _moe_offload_graph_bs > 0:
            capped = [bs for bs in self.capture_bs if bs <= _moe_offload_graph_bs]
            if not capped:
                raise ValueError(
                    f"SGLANG_MOE_OFFLOAD_MAX_GRAPH_BS={_moe_offload_graph_bs} "
                    f"filters out every decode capture bucket {self.capture_bs}"
                )
            self.capture_bs = capped
            self.compile_bs = [
                bs for bs in self.compile_bs if bs <= _moe_offload_graph_bs
            ]
        if KTRANSFORMERS_AVAILABLE:
            KTMoEWrapper.set_capture_batch_sizes(self.capture_bs)

        # Weightless-KV streaming block-decode graphs (#136a): when the lane's
        # B0/B1 block loop is active, decode graphs are captured as a bucketed
        # LADDER over the block count (see FlashInferAttnBackend
        # wl_build_graph_ladder). Cap the bs buckets (each carries a persistent
        # block-wrapper pool; the host-spill path only supports bs=1) and build
        # the ladder the capture loop + replay admission share. Larger batches
        # / over-ladder seq lens fall back to the eager block loop via
        # can_run_graph (guard-free for decode under graphs-enabled, #133).
        self._wl_block_graph = False
        _wl_ab = getattr(self.attn_backend, "full_attn_backend", self.attn_backend)
        if (
            (model_runner.is_weightless_head or model_runner.is_weightless_worker)
            and not model_runner.is_draft_model_runner
            and getattr(_wl_ab, "_wl_chunk_block_size", 0)
        ):
            self._wl_block_graph = True
            self._wl_attn = _wl_ab
            _wl_max_bs = envs.SGLANG_WL_GRAPH_MAX_BS.get()
            if getattr(_wl_ab, "_wl_spill_active", False):
                # The captured H2D staging template is bs=1-only.
                _wl_max_bs = 1
            capped = [bs for bs in self.capture_bs if bs <= _wl_max_bs]
            if not capped:
                raise ValueError(
                    f"SGLANG_WL_GRAPH_MAX_BS={_wl_max_bs} filters out every "
                    f"decode capture bucket {self.capture_bs} for the "
                    "weightless block-decode graph ladder."
                )
            self.capture_bs = capped
            self.compile_bs = [bs for bs in self.compile_bs if bs <= _wl_max_bs]
            _wl_ab.wl_build_graph_ladder()

        # #163: publish the decode TOKEN-count buckets (bs * num_tokens_per_bs)
        # to the GGUF dispatch, so its opt-in MMVQ->MMQ threshold decides per
        # bucket instead of per raw token count. Without this coupling a
        # captured graph could replay a kernel other than the one it was
        # captured with. Registration is unconditional and side-effect free
        # (the consumer ignores it while --gguf-mmq-decode-threshold is off);
        # target and draft runners both register and the union is kept, which
        # preserves the invariant that every replayed token count is itself a
        # bucket.
        #
        # ORDERING IS LOAD-BEARING: this must come AFTER every mutation of
        # self.capture_bs (the MoE-offload cap and the weightless-KV cap above).
        # Publishing earlier happens to stay correct only while the later edits
        # SHRINK the list -- a superset still contains every replayed bucket, so
        # the rounding stays the identity on replay. The moment something ADDS a
        # bucket after publication, a captured graph could replay a kernel other
        # than the one it was captured with, silently. Registering last makes
        # the published set exactly the captured set, which is correct by
        # construction rather than by the direction of the edits above.
        _register_gguf_decode_buckets(self.capture_bs, self.num_tokens_per_bs)
        if self._lane_verify_tokens:
            # The lane verify replays with K+1 tokens, which is a token count
            # this runner otherwise never publishes. Register it too, for the
            # reason stated directly above: the published set must be exactly
            # the captured set, or the GGUF dispatch can pick a different
            # kernel on replay than the capture recorded. (Round 5's defect was
            # in that same <= 8-row MMVQ window, so this is not hypothetical.)
            # Round 7a: every rung of the ladder, for the same reason.
            _register_gguf_decode_buckets(self._lane_verify_tokens, 1)

        # S5 spill-tick graph: UNLIKE the weightless block-decode (which IS the
        # decode and reshapes every capture bucket to bs=1), the spill tick is a
        # SEPARATE bs=1 batch type -- normal decode keeps its own graphs. The
        # spill-tick graphs are an ADDITIONAL per-rung bs=1 capture pass. Gated
        # by the backend's _sess_graph_enabled (flag OFF -> this stays False and
        # the runner is byte-identical). GPU-JUSTIFICATION: the additional
        # capture pass needs a live-ish spilled-session synthetic batch, so it
        # is DRIVEN ON GPU by the messagent; here we only wire the admission +
        # replay-variant so the mechanism is ready.
        self._sess_block_graph = False
        # S0 (deep-offload): set True only inside _sess_capture_one_spill_rung to
        # force plain-decode shaping for the spill-rung capture (read by
        # get_spec_info + capture_prepare). Default False -> byte-inert.
        self._sess_force_plain_decode = False
        _sess_ab = getattr(self.attn_backend, "full_attn_backend", self.attn_backend)
        if (
            getattr(_sess_ab, "_sess_graph_enabled", False)
            and not model_runner.is_draft_model_runner
        ):
            self._sess_block_graph = True
            self._sess_attn = _sess_ab

        self.ragged_verify_mode = (
            ragged_verify_compact_graphs_enabled(self.model_runner.spec_algorithm)
            and (self.capture_forward_mode == ForwardMode.TARGET_VERIFY)
            and not self.model_runner.is_draft_model_runner
        )
        self.capture_num_tokens: Optional[list[int]] = (
            self._build_ragged_verify_token_buckets()
            if self.ragged_verify_mode
            else None
        )
        self._ragged_graph_size = 0
        if self.ragged_verify_mode and (
            self.enable_two_batch_overlap
            or model_runner.server_args.enable_lora
            or self.disable_padding
        ):
            raise ValueError(
                "Compact ragged verify does not support two-batch-overlap, "
                "LoRA, or disable-cuda-graph-padding (bs pads to the captured "
                "tier); disable SGLANG_RAGGED_VERIFY_MODE or the conflicting "
                "feature."
            )

        # If returning hidden states is enabled, set initial capture hidden mode to full to avoid double-capture on startup
        if self.enable_return_hidden_states:
            self.capture_hidden_mode = CaptureHiddenMode.FULL

        # Attention backend
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs
        if self._lane_verify_tokens:
            # One sizing for both entries. The static input buffers, the
            # attention backend's graph state and the shared logits buffer are
            # all cut once, before anything is captured, so the widest entry
            # sets the width: the plain decode graphs are recorded afterwards
            # against the very buffers they will replay with, and a buffer that
            # is longer than their one token changes nothing they read.
            # Re-cutting them per entry is what would break -- the decode
            # graphs would hold pointers into state the backend no longer
            # writes to.
            #
            # It is max_bs * K+1 and not max(max_bs, K+1) on purpose: the mamba
            # backend derives its per-slot verify width from this pair as
            # ``max_num_tokens // max_bs`` (MambaAttnBackendBase.
            # init_cuda_graph_state), so anything else hands the GDN verify a
            # query-start-loc ladder of the wrong step.
            # Round 7a: the WIDEST rung of the ladder sets the width, for
            # exactly the reason above -- the buffers are cut once and every
            # rung is recorded against them.
            self.max_num_token = self.max_bs * max(
                self.num_tokens_per_bs, *self._lane_verify_tokens
            )
        self.attn_backend.init_cuda_graph_state(self.max_bs, self.max_num_token)

        # Init PDMux if needed
        self.maybe_init_pdmux()
        self.seq_len_fill_value = (
            self.attn_backend.get_cuda_graph_seq_len_fill_value()
            if self.dllm_config is None
            else self.dllm_config.block_size
        )

        # Non-zero encoder length ensures cross-attention kernels are captured in the graph.
        self.encoder_len_fill_value = (
            getattr(model_runner.model_config.hf_config, "max_source_positions", 0)
            if self.is_encoder_decoder
            else 0
        )

        if self.enable_torch_compile:
            set_torch_compile_config()

        if self.model_runner.server_args.enable_lora:
            # Phase 2 of LoRA CUDA graph init: dense LoRA batch metadata.
            # Phase 1 (MoE buffers) was handled earlier in ModelRunner via
            # lora_manager.init_cuda_graph_moe_buffers().
            self.model_runner.lora_manager.init_cuda_graph_batch_info(
                max_bs_in_cuda_graph=self.max_bs,
                num_tokens_per_bs=self.num_tokens_per_bs,
            )

        enable_mamba_track = (
            self.model_runner.server_args.enable_mamba_extra_buffer()
            and self.model_runner.spec_algorithm.is_none()
        )

        if self.require_gathered_buffer:
            assert self.require_mlp_tp_gather or self.require_attn_tp_gather

        # --- buffers ---------------------------------------------------
        self.buffers: DecodeInputBuffers = DecodeInputBuffers.create(
            device=self.device,
            max_bs=self.max_bs,
            max_num_token=self.max_num_token,
            hidden_size=self.model_runner.model_config.hidden_size,
            next_token_logits_buffer=self.model_runner.graph_shared_output.get_logits_buffer(
                self.model_runner.model_config.vocab_size, rows=self.max_num_token
            ),
            dtype=self.model_runner.model_config.dtype,
            dp_size=self.dp_size,
            pp_size=self.pp_size,
            is_encoder_decoder=self.is_encoder_decoder,
            require_mlp_tp_gather=self.require_mlp_tp_gather,
            seq_len_fill_value=self.seq_len_fill_value,
            encoder_len_fill_value=self.encoder_len_fill_value,
            num_tokens_per_bs=self.num_tokens_per_bs,
            cache_loc_dtype=self._cache_loc_dtype(),
            enable_mamba_track=enable_mamba_track,
            ne_token_table=(
                model_runner.token_table if self.use_ngram_embedding else None
            ),
            hc_hidden_size=getattr(
                self.model_runner.model_config, "hc_hidden_size", None
            ),
            pp_proxy_topk_size=self.model_runner.get_pp_proxy_topk_size(),
        )
        self.buffers.share_buffers()
        if self._lane_draft_capture:
            # #274 round 7a: the ONE static input the generic decode buffers do
            # not carry, because no non-speculative decode has it -- the MTP
            # head reads its previous-layer hidden states off
            # ``spec_info.hidden_states`` and concatenates them with the token
            # embedding. It is a graph INPUT, so it has to be a fixed address
            # that the round copies into, exactly like input_ids.
            #
            # Allocated OUTSIDE share_buffers() on purpose: the process-wide
            # coalescing pool is keyed by (lane, name) since slice D2/D3, but a
            # private tensor cannot be aliased by anything at all, and this is
            # the third member of a bug family (#274 D2/D3) about returned
            # buffers. Nothing else wants this shape, so there is no saving to
            # give up.
            self._lane_draft_hidden = torch.zeros(
                (self.max_num_token, self.model_runner.model_config.hidden_size),
                dtype=self.model_runner.model_config.dtype,
                device=self.device,
            )
        # FB-shared slot registry adopting DecodeInputBuffers storage (same
        # physical tensors, stable data_ptr for capture vs replay). Provides
        # the unified fill_from / slot access surface, replacing
        # populate_from_forward_batch on capture/replay paths.
        self.buffer_registry: CudaGraphBufferRegistry = build_decode_registry(
            device=self.device,
            max_bs=self.max_bs,
            max_num_token=self.max_num_token,
            seq_len_fill_value=self.seq_len_fill_value,
            cache_loc_dtype=self._cache_loc_dtype(),
            enable_mamba_track=enable_mamba_track,
            is_encoder_decoder=self.is_encoder_decoder,
            encoder_len_fill_value=self.encoder_len_fill_value,
            enable_num_token_non_padded=enable_num_token_non_padded(),
            require_gathered_buffer=self.require_gathered_buffer,
            enable_prefill_cp=self.enable_prefill_cp,
            require_mlp_tp_gather=self.require_mlp_tp_gather,
            dp_size=self.dp_size,
            source=self.buffers,
        )

        # --- backend ---------------------------------------------------
        self.backend = resolve_decode_backend(self)

        # --- capture --------------------------------------------------
        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture cuda graph failed: {e}\n" f"{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

    def _build_ragged_verify_token_buckets(self) -> list[int]:
        buckets = sorted({bs * self.num_tokens_per_bs for bs in self.capture_bs})
        assert buckets and buckets[0] > 0, f"{buckets=}"
        return buckets

    def _autotune_buffers(self):
        """Reuse these static decode buffers (sized to max_bs) for the warmup
        flashinfer-autotune dummy forward instead of allocating a throwaway set
        — see BaseRunner._autotune_buffers / BaseRunner._dummy_run.

        The dummy forward derives its shape from max_bs and must match these
        buffers exactly; _dummy_run asserts that. Every autotune-reachable
        decode shape (plain decode, spec target-verify) matches. DLLM would not
        (its buffers hold block_size tokens/bs while the dummy run derives 1),
        but DLLM does not use a flashinfer MoE backend, so autotune never runs
        for it and this is never reached there.
        """
        return self.buffers, self.max_bs

    def maybe_init_pdmux(self):
        if self.enable_pdmux:
            self.stream_groups = get_stream_groups()
            for attn_backend in self.model_runner.decode_attn_backend_group:
                attn_backend.init_cuda_graph_state(self.max_bs, self.max_num_token)

    def _cache_loc_dtype(self):
        return torch.int64

    def _make_graph_key(self, size, stream_idx=None, variant_label=None):
        return ShapeKey(
            size=size,
            stream_idx=stream_idx,
            variant_label=variant_label,
        )

    def _capture_graph_size(self, *, bs: int, num_tokens: int) -> int:
        return num_tokens if self.ragged_verify_mode else bs

    def _resolve_lora_variant(self, forward_batch: ForwardBatch):
        if not getattr(self, "record_nolora_graph", False):
            return None
        if forward_batch.lora_ids is not None and any(
            uid is not None for uid in forward_batch.lora_ids
        ):
            return "lora"
        return "nolora"

    @staticmethod
    def _forward_is_dp_local(model_runner) -> bool:
        """The DSpark dense draft runs attn-TP-local (draft_tp_context): each
        DP rank drafts independently with no cross-DP collective, so its
        hand-built batches carry no dp-global metadata and must key graphs by
        local batch size. Everything else keeps the dp-global padding path."""
        if not model_runner.is_draft_model_runner:
            return False
        if not model_runner.spec_algorithm.is_dspark():
            return False
        from sglang.srt.speculative.dspark_components.dspark_config import (
            draft_is_deepseek_v4,
        )

        return not draft_is_deepseek_v4(server_args=model_runner.server_args)

    def _ragged_capture_slots(self, num_tokens: int) -> int:
        if envs.SGLANG_TEST_RAGGED_VERIFY_FORCE_UNIFORM_CAPTURE.get():
            return num_tokens // self.num_tokens_per_bs
        return min(num_tokens, self.max_bs)

    def _capture_ragged_verify_layout(self, num_tokens: int):
        if not self.ragged_verify_mode:
            return None
        if envs.SGLANG_TEST_RAGGED_VERIFY_FORCE_UNIFORM_CAPTURE.get():
            return None
        from sglang.srt.speculative.ragged_verify import (
            RaggedVerifyLayout,
            build_capture_verify_lens,
        )

        verify_lens_cpu = build_capture_verify_lens(
            num_tokens=num_tokens,
            num_slots=self._ragged_capture_slots(num_tokens),
            num_draft_tokens=self.num_tokens_per_bs,
        )
        return RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=verify_lens_cpu,
            device=self.device,
            grid=self.capture_num_tokens,
        )

    def _wl_variant_label(self, default):
        """#136a: replay graph-key variant for the weightless block-decode
        ladder -- the rung chosen by wl_graph_can_replay (which every replay
        passes through via can_run_graph before load_batch)."""
        if self._lane_verify_active:
            # #274 round 6: the lane verify shares ShapeKey.size with the
            # plain bs=1 decode graph, so the variant is the only thing that
            # keeps the two entries apart. Only set while the lane holds the
            # shape scope. Round 7a: and the RUNGS share it with each other,
            # so the rung's row count is part of the label -- without it the
            # ladder would be one key and the last capture would win.
            return f"{self.LANE_VERIFY_VARIANT}{self._lane_verify_active}"
        if self._lane_draft_capture:
            # #274 round 7a: the lane head's only entry. A distinct label
            # rather than the default None costs nothing and makes a head
            # graph impossible to confuse with a target decode graph in a
            # log or a dump.
            return self.LANE_DRAFT_VARIANT
        if self._wl_block_graph:
            rung = self._wl_attn._wl_graph_replay_blocks
            assert rung is not None, (
                "weightless block-decode graph replay without a rung "
                "(can_run_graph admission must precede load_batch)"
            )
            return f"wlblk{rung}"
        if self._sess_block_graph and self._sess_attn._sess_graph_replay_blocks:
            # S5: spill-tick replay keys on its captured rung.
            return f"sessblk{self._sess_attn._sess_graph_replay_blocks}"
        return default

    def can_run_graph(self, forward_batch: ForwardBatch):
        # #1007: THE TOKEN AXIS, WHICH THIS PREDICATE NEVER CHECKED. Every
        # verdict below reasons about BATCH SIZE (`cuda_graph_bs =
        # forward_batch.batch_size`, `cuda_graph_bs <= self.max_bs`); none of
        # them asks how many tokens each sequence carries. The captured graph
        # fixes that separately as `num_tokens_per_bs`, and `load_batch`
        # derives every buffer width from it (`raw_num_token = raw_bs *
        # self.num_tokens_per_bs`). So a batch with the right bs and the wrong
        # tokens-per-sequence passes here and dies one frame later in the
        # copy.
        #
        # MEASURED, boot 66: a 3-token first-pass prefill (rid 6f9d8c6e,
        # prefix_lens=0 fill_lens=3 out_lens=0, no retraction, no chunk --
        # literally a "Say OK." probe) reached this runner with raw_bs=1 and
        # was accepted, then failed as
        # `input_ids: dst(1,) <- src(3,)`. Every short prompt does this,
        # health checks included, which is why every boot of this window died
        # on its first real request.
        #
        # THE LINE IS NOT OPTIONAL (#967): a silent `return False` here would
        # make it unmeasurable whether this ever fires, and the mode it prints
        # is the half of the root that is still unread -- `is_cuda_graph()`
        # admits DECODE, TARGET_VERIFY, IDLE and DLLM_EXTEND, and which of
        # those this batch claims to be cannot be settled from the boot-66 log.
        # So the guard reports it, and the next boot names it instead of me
        # guessing it.
        _ids = getattr(forward_batch, "input_ids", None)
        if _ids is not None and self.num_tokens_per_bs:
            _have = int(_ids.shape[0])
            _want = int(forward_batch.batch_size) * int(self.num_tokens_per_bs)
            if _have != _want:
                logger.warning(
                    "#1007 DECODE GRAPH REFUSED (token geometry): "
                    "input_ids=%d but batch_size=%d x num_tokens_per_bs=%d=%d "
                    "-- forward_mode=%s. The captured graph fixes tokens per "
                    "sequence; this batch does not match it, so it runs eager "
                    "instead of failing in the buffer copy. A non-zero count "
                    "with forward_mode=DECODE means the classification is "
                    "wrong upstream; with an EXTEND mode it means "
                    "is_cuda_graph() admits a mode this runner cannot serve.",
                    _have,
                    int(forward_batch.batch_size),
                    int(self.num_tokens_per_bs),
                    _want,
                    getattr(forward_batch, "forward_mode", "?"),
                )
                return False
        # Disable for token embedding overrides (dynamic per-request)
        if forward_batch.replace_embeds is not None:
            return False

        # #274 round 6: the lane's verify entry answers for itself and returns
        # EARLY, before the shared gates below. Two of them would otherwise
        # decide wrongly: the bs gate would pad a K+1-token chain into the
        # 1-token decode bucket, and capture_hidden_mode_matches would refuse
        # FULL -- which is worse than it sounds, because the refusal is
        # followed on the next admitted batch by recapture_if_needed tearing
        # down and re-recording EVERY plain decode graph in FULL mode. The
        # lane's no-spec entry has to stay exactly the graph it was.
        if self._lane_verify_active:
            return self.lane_verify_can_replay(forward_batch)

        # #274 round 7a: the lane's NEXTN head answers for itself for the
        # mirror-image reason -- its batch carries an EagleDraftInput whose
        # hidden states are the graph's second input, and a batch without one
        # (there is none on this runner today, but a future caller is not
        # bound by that) must stay eager rather than replay a graph against a
        # stale hidden buffer.
        if self._lane_draft_capture:
            return self.lane_draft_can_replay(forward_batch)

        # kv-session-offload: the spill tick streams host-resident KV
        # blockwise. S5: run it as a captured graph when the spill-graph flag
        # is on, admission succeeds (bs=1 + a covering rung -- rank-uniform),
        # AND that rung was actually captured (the additional per-rung capture
        # pass is GPU-wired; until then _sess_graph_captured is empty -> eager).
        # Flag OFF or over-ladder -> eager, byte-identical. Replicated inputs
        # -> every rank decides identically (no collective-count divergence).
        # BUGFIX: _sess_graph_replay_blocks is a sticky backend field that
        # _variant_label reads to pick the sessblk{R} graph key. It is set only
        # when a SPILL TICK is admitted (via _sess_graph_can_replay below). A
        # DEVICE batch never calls that, so a stale value would leak into the
        # device batch's variant label -> at bs>1 a KeyError (no size=N
        # sessblk graph), at bs=1 a SILENT replay of the wrong (spill) graph.
        # Clear it on EVERY admission; the spill-tick branch re-sets it.
        if self._sess_block_graph:
            self._sess_attn._sess_graph_replay_blocks = None
        if getattr(forward_batch, "kv_session_spill_tick", False):
            if (
                self._sess_block_graph
                # S0 (deep-offload): the spill tick is a PLAIN bs=1 DECODE with
                # ONE new token (spec_algorithm=NONE, spill_decode_alloc gives one
                # sentinel slot/tick). This decode-graph runner buckets bs by
                # num_tokens_per_bs -- and under a SPECULATIVE server that is
                # num_draft+1 (TARGET_VERIFY shape). The replay buffer/graph-size
                # bookkeeping (raw_num_token = raw_bs * num_tokens_per_bs, the
                # padded req_pool_indices slot) then assumes the multi-token
                # verify shape and mismatches the 1-token tick (stale capture
                # req_pool row -> KeyError, wrong token count). The eager block
                # loop handles deep multi-block tails correctly regardless, so
                # route the spill tick to EAGER whenever the runner is not in the
                # plain 1-token/bs decode shape. Non-spec servers (num_tokens_per
                # _bs == 1) keep the captured spill-tick graph. Capacity (reachable
                # depth) is unaffected -- only this tick's per-step graph speedup
                # is deferred under MTP. Rank-uniform: num_tokens_per_bs is a
                # replicated server config, so every DCP rank decides identically.
                and self.num_tokens_per_bs == 1
                and self._sess_attn._sess_graph_can_replay(forward_batch)
                and self._sess_attn._sess_graph_captured(
                    self._sess_attn._sess_graph_replay_blocks
                )
            ):
                return True
            return False

        # Weightless streaming block-decode (#136a): rung availability +
        # (under host spill) linear-slot-layout admission. Rank-uniform
        # inputs -> head and workers decide identically.
        if self._wl_block_graph and not self._wl_attn.wl_graph_can_replay(
            forward_batch
        ):
            return False

        # A VERIFY OF A DIFFERENT WIDTH IS A DIFFERENT GRAPH (#631).
        #
        # These graphs are recorded at a fixed tokens-per-sequence
        # (``num_tokens_per_bs`` = num_draft_tokens under speculation), and
        # every gate below reasons in SEQUENCES: the bs gate divides token
        # counts by that constant and the buffers were sized by it. A
        # verify input carrying a DIFFERENT ``draft_token_num`` therefore
        # replays a graph whose per-sequence token stride does not match
        # its own -- not a padding question; the tokens land in the wrong
        # slots.
        #
        # Nothing produced such a batch until #631's phase-flip bootstrap
        # round, which runs a 1-node trivial verify on an instance whose
        # graphs were captured at 4. It gets eager execution -- correct and
        # merely slower, for one round per flip -- instead of a silently
        # misaligned replay.
        #
        # THE CONDITION IS DELIBERATELY THE NARROWEST ONE THAT COVERS IT,
        # and a plain ``draft_token_num != num_tokens_per_bs`` was written
        # first and rejected: ``num_tokens_per_bs`` is fixed at capture time
        # from ``speculative_num_draft_tokens``, while adaptive speculative
        # decoding varies the ACTIVE width at runtime, so the inequality
        # would have refused graphs across ordinary adaptive operation --
        # a silent throughput collapse dressed as a safety check. The
        # 1-node trivial verify is identified by what only it sets:
        # ``spec_steps == 0`` (no drafting happened) with a width of 1, on
        # a runner whose graphs are wider than 1. The zero-step config,
        # where this shape is the NORMAL one, captures at
        # num_tokens_per_bs == 1 and is excluded by the last clause.
        spec_info = forward_batch.spec_info
        spec_width = getattr(spec_info, "draft_token_num", None)
        if (
            spec_width is not None
            and int(spec_width) == 1
            and getattr(spec_info, "spec_steps", None) == 0
            and int(self.num_tokens_per_bs) > 1
        ):
            return False

        ragged_layout = (
            resolve_ragged_verify_layout(forward_batch)
            if self.ragged_verify_mode
            else None
        )
        if ragged_layout is not None:
            return self._can_run_ragged_verify_graph(forward_batch, ragged_layout)
        if self.ragged_verify_mode and forward_batch.forward_mode.is_target_verify():
            return False

        if self.require_mlp_tp_gather:
            cuda_graph_bs = (
                max(forward_batch.global_num_tokens_cpu) // self.num_tokens_per_bs
                if self.model_runner.spec_algorithm.is_eagle()
                or self.model_runner.spec_algorithm.is_standalone()
                or self.model_runner.spec_algorithm.is_dflash_family()
                else max(forward_batch.global_num_tokens_cpu)
            )
        else:
            cuda_graph_bs = forward_batch.batch_size

        graph_key = cuda_graph_bs
        if self.enable_pdmux:
            graph_key = f"{get_current_stream_idx()}_{cuda_graph_bs}"

        is_bs_supported = (
            self.backend.can_run(forward_batch, graph_key)
            if self.disable_padding
            else cuda_graph_bs <= self.max_bs
        )

        if self.require_mlp_sync:
            is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph

        # NOTE: cuda graph cannot handle mixed batch (encoder_len = 0)
        # If mixed batch cannot be supported, then encoder_lens can be removed in cuda graph
        # because the full_text_row_masked_out_mask tensor will always be ones
        is_encoder_lens_supported = (
            torch.all(forward_batch.encoder_lens > 0)
            if self.is_encoder_decoder
            else True
        )

        requested_capture_hidden_mode = max(
            forward_batch.capture_hidden_mode,
            (
                forward_batch.spec_info.capture_hidden_mode
                if getattr(forward_batch.spec_info, "capture_hidden_mode", None)
                is not None
                else CaptureHiddenMode.NULL
            ),
        )
        capture_hidden_mode_matches = (
            requested_capture_hidden_mode == CaptureHiddenMode.NULL
            or requested_capture_hidden_mode == self.capture_hidden_mode
        )
        is_tbo_supported = (
            forward_batch.can_run_tbo if self.enable_two_batch_overlap else True
        )

        is_ngram_supported = (
            (
                forward_batch.batch_size * self.num_tokens_per_bs
                == forward_batch.input_ids.numel()
            )
            if self.model_runner.spec_algorithm.is_ngram()
            else True
        )

        return (
            is_bs_supported
            and is_encoder_lens_supported
            and is_tbo_supported
            and capture_hidden_mode_matches
            and is_ngram_supported
        )

    def _can_run_ragged_verify_graph(self, forward_batch: ForwardBatch, ragged_layout):
        if not self.attn_backend.supports_ragged_verify_graph:
            return False

        admission_tokens = ragged_layout.graph_num_tokens
        is_tokens_supported = admission_tokens <= self.capture_num_tokens[
            -1
        ] and forward_batch.batch_size <= self._ragged_capture_slots(admission_tokens)

        is_dp_supported = (
            forward_batch.can_run_dp_cuda_graph if self.require_mlp_sync else True
        )

        is_encoder_lens_supported = (
            torch.all(forward_batch.encoder_lens > 0)
            if self.is_encoder_decoder
            else True
        )

        requested_capture_hidden_mode = max(
            forward_batch.capture_hidden_mode,
            (
                forward_batch.spec_info.capture_hidden_mode
                if getattr(forward_batch.spec_info, "capture_hidden_mode", None)
                is not None
                else CaptureHiddenMode.NULL
            ),
        )
        capture_hidden_mode_matches = (
            requested_capture_hidden_mode == CaptureHiddenMode.NULL
            or requested_capture_hidden_mode == self.capture_hidden_mode
        )

        return (
            is_tokens_supported
            and is_dp_supported
            and is_encoder_lens_supported
            and capture_hidden_mode_matches
        )

    def _init_profile_context_and_memory_record(self):
        profile_context = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        )
        torch.cuda.memory._record_memory_history()
        return profile_context

    def _post_process_after_profile(self, prof_context):
        torch.cuda.memory._dump_snapshot("cuda_graph_runner_memory_usage.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        log_message = (
            "Sorted by CUDA Time:\n"
            + prof_context.key_averages(group_by_input_shape=True).table(
                sort_by="cuda_time_total", row_limit=10
            )
            + "\n\nSorted by CPU Time:\n"
            + prof_context.key_averages(group_by_input_shape=True).table(
                sort_by="cpu_time_total", row_limit=10
            )
            + "\n\nMemory Usage is saved to cuda_graph_runner_memory_usage.pickle\n"
        )
        logger.info(log_message)

        # Optionally persist the shaped capture trace (record_shapes=True) for
        # offline per-kernel analysis -- opt-in via
        # SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE; the in-log tables above are
        # unchanged.
        export_cuda_graph_capture_trace(
            prof_context,
            runner_name=type(self).__name__,
            tp_rank=get_parallel().tp_rank,
        )

    def capture_prepare(
        self,
        size: int,
        stream_idx: Optional[int] = None,
        num_tokens: Optional[int] = None,
    ):
        """Build the dummy decode ForwardBatch for capture at size (=bs),
        populate static input buffers, choose the active attn backend, and
        optionally build pp_proxy_tensors.

        num_tokens defaults to the uniform bs * num_tokens_per_bs; ragged
        verify capture passes the decoupled (slots, tier tokens) pair.

        Returns (forward_batch, attn_backend, pp_proxy_tensors);
        pp_proxy_tensors is None unless pp_size > 1.
        """
        bs = size
        buffers: DecodeInputBuffers = self.buffers
        if num_tokens is None:
            num_tokens = bs * self.num_tokens_per_bs

        # Registry-owned FB-shared slots come through the registry (which
        # shares physical storage with self.buffers via source=...); the rest
        # still come off buffers directly.
        registry = self.buffer_registry

        def _slot(name):
            return registry.get_slot(name).slice_for(bs, num_tokens)

        input_ids = _slot("input_ids")
        req_pool_indices = _slot("req_pool_indices")
        seq_lens = _slot("seq_lens")
        seq_lens_cpu = _slot("seq_lens_cpu")
        out_cache_loc = _slot("out_cache_loc")
        positions = _slot("positions")
        encoder_lens = (
            _slot("encoder_lens") if registry.has_slot("encoder_lens") else None
        )
        mrope_positions = _slot("mrope_positions")
        next_token_logits_buffer = buffers.next_token_logits_buffer[:num_tokens]
        rids_int = buffers.rids_int[:bs] if buffers.rids_int is not None else None
        bootstrap_room_ids_int = (
            buffers.bootstrap_room_ids_int[:bs]
            if buffers.bootstrap_room_ids_int is not None
            else None
        )

        # Adjust for attention TP if needed (matching replay path in
        # populate_from_forward_batch).
        buffers.num_token_non_padded[...] = num_tokens
        if (
            enable_num_token_non_padded()
            and self.require_gathered_buffer
            and not self.enable_prefill_cp
        ):
            local = compute_local_num_token_non_padded(
                global_num_token_non_padded=buffers.num_token_non_padded,
                num_tokens_per_dp=num_tokens,
            )
            buffers.num_token_non_padded.copy_(local)

        pp_proxy_tensors = None
        # pipeline parallelism
        if self.pp_size > 1:
            pp_proxy_tensors = PPProxyTensors(
                {k: v[:num_tokens] for k, v in buffers.pp_proxy_tensors.items()}
            )

        if self.require_mlp_tp_gather:
            global_num_tokens_cpu = [num_tokens] * self.dp_size
        elif self.require_attn_tp_gather:
            global_num_tokens_cpu = [num_tokens]
        else:
            global_num_tokens_cpu = None

        if global_num_tokens_cpu is not None:
            global_dp_buffer_len = sum(global_num_tokens_cpu)
            num_tokens_tensor = torch.tensor(
                global_num_tokens_cpu, dtype=torch.int32, device=input_ids.device
            )
            buffers.global_num_tokens_gpu.copy_(num_tokens_tensor)
            buffers.global_num_tokens_for_logprob_gpu.copy_(num_tokens_tensor)
        else:
            global_dp_buffer_len = None

        spec_info = self.get_spec_info(num_tokens)
        if self.capture_hidden_mode != CaptureHiddenMode.FULL:
            self.capture_hidden_mode = (
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            )

        if self.model_runner.server_args.enable_lora:
            # It is safe to capture CUDA graph using empty LoRA id, as the LoRA kernels will always be launched whenever
            # `--enable-lora` is set to True (and return immediately if the LoRA id is empty for perf optimization).
            lora_ids = [None] * bs
        else:
            lora_ids = None

        # mamba state tracking (registry-owned when enabled)
        mamba_track_indices = (
            _slot("mamba_track_indices")
            if registry.has_slot("mamba_track_indices")
            else None
        )
        mamba_track_mask = (
            _slot("mamba_track_mask") if registry.has_slot("mamba_track_mask") else None
        )

        if stream_idx is None:
            attn_backend = self.attn_backend
        else:
            assert self.enable_pdmux
            attn_backend = self.model_runner.decode_attn_backend_group[stream_idx]

        forward_batch = ForwardBatch(
            forward_mode=self.capture_forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            next_token_logits_buffer=next_token_logits_buffer,
            orig_seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            mamba_track_indices=mamba_track_indices,
            mamba_track_mask=mamba_track_mask,
            mamba_track_seqlens=None,
            encoder_lens=encoder_lens,
            return_logprob=False,
            positions=positions,
            global_num_tokens_gpu=buffers.global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=buffers.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            global_num_tokens_cpu=global_num_tokens_cpu,
            mrope_positions=mrope_positions,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=self.capture_hidden_mode,
            num_token_non_padded=buffers.num_token_non_padded,
            global_forward_mode=self.capture_forward_mode,
            lora_ids=lora_ids,
            rids_int=rids_int,
            bootstrap_room_ids_int=bootstrap_room_ids_int,
        )

        # Trip the coordinator so the hisparse code path is captured into the
        # graph; backends read it from self.model_runner.hisparse_coordinator.
        forward_batch.hisparse_coordinator = self.model_runner.hisparse_coordinator
        if forward_batch.hisparse_coordinator is not None:
            forward_batch.hisparse_coordinator.num_real_reqs.fill_(bs)

        if buffers.ngram_embedding_info is not None:
            forward_batch.ngram_embedding_info = buffers.ngram_embedding_info.slice(bs)

        return forward_batch, attn_backend, pp_proxy_tensors

    def capture(self) -> None:
        # Warm up + autotune kernels once before capture (run-once across the
        # decode + prefill runners; see BaseRunner.warmup).
        self.warmup()
        # warmup() may disable torch.compile for a model whose _can_torch_compile
        # is False; recompute the compile bucket so capture matches.
        if self.enable_torch_compile and not (get_flags().capture.enable_torch_compile):
            self.enable_torch_compile = False
            _, self.compile_bs = get_batch_sizes_to_capture(
                self.model_runner, self.num_tokens_per_bs
            )
        profile_context = empty_context()
        if self.enable_profile_cuda_graph:
            profile_context = self._init_profile_context_and_memory_record()

        # share_buffers() coalesces seq_lens / seq_lens_cpu through the process-
        # wide pool, so they may alias a buffer seeded by an earlier runner (the
        # eager registry fills them with 0). The capture-time attention-metadata
        # plan reads these as the per-request KV length, and the prefill wrapper
        # (DLLM_EXTEND) asserts kv_len >= qo_len, so restore the fill value the
        # captured graph needs before capturing.
        self.buffers.seq_lens.fill_(self.seq_len_fill_value)
        self.buffers.seq_lens_cpu.fill_(self.seq_len_fill_value)

        # Weightless-KV fast lane (#133): the anti-hang guard's per-step gloo
        # handshake is a host-sync that CANNOT be recorded into a CUDA graph.
        # Head + worker capture a SYMMETRIC DCP-collective sequence here (the
        # head's model.forward and the worker's stripped per-layer dispatch), so
        # the captured sequence is fixed by construction and the guard is
        # unnecessary inside the captured region. Disable it for the whole
        # capture (the 2 eager warmup forwards + the recorded pass); it is
        # re-enabled for eager prefill/extend. Rank-uniform: every rank in the
        # lane (head + weightless workers) builds a decode graph runner and hits
        # this toggle at the same point.
        _wl_lane = (
            self.model_runner.is_weightless_head
            or self.model_runner.is_weightless_worker
        )
        # S5 spill-tick graph capture is ALSO guard-free (its per-layer DCP
        # cp_lse collectives are captured symmetrically on every rank; the
        # guard's per-step gloo handshake is a host-sync that cannot be
        # recorded). Rank-uniform: every DCP rank builds this runner and hits
        # the toggle at the same point (#133 lockstep). GPU-JUSTIFICATION:
        # confirm the disable window covers exactly the spill capture region.
        _guard_off = _wl_lane or self._sess_block_graph
        if _guard_off:
            from sglang.srt.layers.dcp.collective_guard import set_guard_enabled

            set_guard_enabled(False)

        try:
            # Trigger CUDA graph capture for specific shapes.
            # Capture the large shapes first so that the smaller shapes
            # can reuse the memory pool allocated for the large shapes.
            with freeze_gc(self.model_runner.server_args.enable_cudagraph_gc):
                if not self.enable_pdmux:
                    with graph_capture() as graph_capture_context, profile_context as prof:
                        self.stream = graph_capture_context.stream
                        with self.backend.capture_session(self.stream):
                            self._capture_one_stream()
                else:
                    set_pdmux_status(False)
                    for i, sg in enumerate(self.stream_groups):
                        with (
                            graph_capture(stream=sg[1]) as graph_capture_context,
                            profile_context as prof,
                        ):
                            self.stream = graph_capture_context.stream
                            with self.backend.capture_session(self.stream):
                                self._capture_one_stream(i)

            if self.enable_profile_cuda_graph:
                self._post_process_after_profile(prof)
        finally:
            if _guard_off:
                # Restore the guard for the eager prefill/extend path.
                set_guard_enabled(True)

        # No pool-side pin to clear: the captured full-physical write loc rides the
        # backend's `ForwardMetadata.out_cache_loc_full_physical` (-> KVWriteLoc.full_loc).

        # S5: ENV-gated per-rung graph==eager selftest (KVSO_GRAPH_SELFTEST),
        # run OUTSIDE the capture context now that all rungs are recorded.
        # Default OFF -> no-op. Covers rungs 2-28 (unreachable under real load).
        if self._sess_block_graph:
            sess_selftest = getattr(self._sess_attn, "_sess_graph_selftest", None)
            if sess_selftest is not None:
                try:
                    sess_selftest(self.model_runner)
                except Exception as _sess_e:
                    # Diagnostic only -- never fail the boot on the selftest.
                    logger.warning(
                        "kvso spill-graph selftest raised (ignored): %r", _sess_e
                    )

        # C4 (spec-in-spill): ENV-gated (KVSO_ATTN_SELFTEST) target-verify
        # host-prefix attention correctness proof. Independent of the spill
        # graph (resolves the target full-attn backend directly), gated by the
        # env inside the method -> default OFF is byte-inert. Run once here now
        # that the model + KV pool + staging buffers are fully initialized.
        _sess_ab = getattr(self.attn_backend, "full_attn_backend", self.attn_backend)
        attn_selftest = getattr(_sess_ab, "_sess_attn_selftest", None)
        if (
            attn_selftest is not None
            and not self.model_runner.is_draft_model_runner
            and getattr(_sess_ab, "_sess_enabled", False)
        ):
            try:
                attn_selftest(self.model_runner)
            except Exception as _sess_e:
                # Diagnostic only -- never fail the boot on the selftest.
                logger.warning(
                    "kvso C4 attn selftest raised (ignored): %r", _sess_e
                )

    def _capture_one_stream(self, stream_idx: Optional[int] = None) -> None:
        avail_mem = get_available_gpu_memory(
            self.model_runner.device,
            self.model_runner.gpu_id,
            empty_cache=False,
        )
        # Reverse so cuda graphs share memory better.
        capture_range = (
            tqdm.tqdm(list(reversed(self.capture_bs)))
            if get_parallel().tp_rank == 0
            else reversed(self.capture_bs)
        )
        lora_variants = (
            [("lora", True), ("nolora", False)]
            if getattr(self, "record_nolora_graph", False)
            else [(None, None)]
        )
        for bs in capture_range:
            if get_parallel().tp_rank == 0:
                avail_mem = get_available_gpu_memory(
                    self.model_runner.device,
                    self.model_runner.gpu_id,
                    empty_cache=False,
                )
                capture_range.set_description(
                    f"Capturing batches ({bs=} {avail_mem=:.2f} GB)"
                )

            if self._wl_block_graph:
                # #136a rung ladder: one graph per block-count rung, captured
                # largest-first (memory-pool reuse, same rationale as the bs
                # order). Head + every weightless worker iterate the IDENTICAL
                # (bs, rung) sequence -- the capture-time DCP collectives of
                # each graph require symmetric co-participation (#133).
                try:
                    for rung in sorted(self._wl_attn._wl_graph_ladder, reverse=True):
                        self._wl_attn._wl_graph_capture_blocks = rung
                        with torch_compile_decoration.patch_model(
                            self.model_runner.model,
                            bs in self.compile_bs,
                            num_tokens=bs * self.num_tokens_per_bs,
                            tp_group=self.model_runner.tp_group,
                        ) as forward:
                            self.capture_one_shape(
                                bs, forward, stream_idx, f"wlblk{rung}"
                            )
                finally:
                    self._wl_attn._wl_graph_capture_blocks = None
                continue

            for variant_label, _variant_has_lora in lora_variants:
                _set_capture_lora_variant(variant_label)
                # #274 round 7a: the two sides of the graph key have to be
                # computed by the SAME rule. Replay derives its label from
                # ``_wl_variant_label``; capture takes whatever is passed in
                # here, which for the plain loop is the LoRA variant (None on
                # every deployment without LoRA). The lane head's replay label
                # is ``lanedraft``, so a capture under None records a key the
                # replay can never find -- measured, boot 2 of round 7a, a
                # ``KeyError: ShapeKey(size=1, variant_label='lanedraft')`` on
                # the first head forward. Resolving through the same helper
                # closes that by construction; for every runner without a lane
                # entry the label is the LoRA variant exactly as before, so
                # every other deployment is byte-inert.
                capture_label = (
                    self.LANE_DRAFT_VARIANT
                    if self._lane_draft_capture
                    else variant_label
                )
                with torch_compile_decoration.patch_model(
                    self.model_runner.model,
                    bs in self.compile_bs,
                    num_tokens=bs * self.num_tokens_per_bs,
                    tp_group=self.model_runner.tp_group,
                ) as forward:
                    self.capture_one_shape(bs, forward, stream_idx, capture_label)

            # S5 spill-tick graph: an ADDITIONAL bs=1 capture pass, one graph
            # per rung (largest-first for memory-pool reuse). PORT of the
            # #136a rung loop above; the spill tick is a SEPARATE batch, so it
            # is captured beside the normal decode graphs (not instead of
            # them). Rank-uniform SYMMETRIC capture: every DCP rank builds this
            # runner and iterates the IDENTICAL rung sequence, so the capture-
            # time per-layer cp_lse collectives pair up (#133 lockstep). Only
            # bs==1 (the spill tick is always bs=1).
            if self._sess_block_graph and bs == 1:
                try:
                    for rung in sorted(
                        self._sess_attn._sess_graph_ladder, reverse=True
                    ):
                        self._sess_attn._sess_graph_capture_blocks = rung
                        with torch_compile_decoration.patch_model(
                            self.model_runner.model,
                            bs in self.compile_bs,
                            num_tokens=bs * self.num_tokens_per_bs,
                            tp_group=self.model_runner.tp_group,
                        ) as forward:
                            self._sess_capture_one_spill_rung(
                                bs, forward, stream_idx, rung
                            )
                        self._sess_attn._sess_graph_captured_rungs.add(rung)
                finally:
                    self._sess_attn._sess_graph_capture_blocks = None

            # #274 round 6: the lane's chain-verify entry. One graph, bs 1,
            # K+1 tokens, TARGET_VERIFY, hidden mode FULL -- an ADDITIONAL
            # pass, captured after the plain decode graphs of this bs so those
            # are recorded exactly as they were before this existed.
            # Round 7a: one such pass PER RUNG of the K ladder, widest first
            # (memory-pool reuse, the same ordering argument as the bs loop).
            # All rungs are captured up front so a rung change at runtime is a
            # key flip and never a re-capture.
            if self._lane_verify_tokens and bs == 1:
                captured = set(self._lane_verify_captured)
                for rung in sorted(self._lane_verify_tokens, reverse=True):
                    self._lane_capture_verify(bs, stream_idx, rung)
                    captured.add(rung)
                self._lane_verify_captured = frozenset(captured)

        if self._lane_draft_capture:
            # #274 round 7a: the head's entries are the ordinary decode
            # captures of the loop above (this runner has no second shape);
            # what the flag records is that they exist, so a forward before
            # capture -- the eager warmup -- cannot admit itself to a graph
            # that is not there yet.
            self._lane_draft_captured = True
            logger.info(
                "dual-group lane: NEXTN head graph captured (bs %s, 1 token, "
                "DECODE, hidden LAST, EagleDraftInput).",
                self.capture_bs,
            )

    def _sess_capture_one_spill_rung(
        self, bs, forward, stream_idx, rung
    ) -> None:
        """Capture one spill-tick decode graph for ``rung`` (PORT of the
        weightless head's capture_one_shape rung recording). Reuses the normal
        capture harness (capture_prepare + model.forward record) but marks the
        synthetic batch as a spill tick at the rung worst case, so every full-
        attention layer routes to _sess_blockwise_decode_return_lse_graph and
        the whole tick forward is recorded.

        GPU-JUSTIFICATION (the messagent wires + iterates these; the STRUCTURE
        hugs the head capture_one_shape so tuning is minimal):
          * _sess_install_capture_state builds a SYNTHETIC spilled session
            (backend region slot + sentinel req_to_token row + host-pool rows)
            so _sess_prepare_step yields a rung-max plan (block FULL, head at
            max) -- capture-time state install/teardown boundaries;
          * the exact seq_lens / req_pool_indices the synthetic tick needs;
          * capture-region boundaries + event ordering."""
        # Install synthetic spilled-session state for the rung worst case.
        install = getattr(self._sess_attn, "_sess_install_capture_state", None)
        ctx = (
            install(bs, rung)
            if install is not None
            else contextlib.nullcontext()
        )
        # S0 (deep-offload block>512 fix): the LIVE spill tick is a PLAIN bs=1
        # DECODE batch -- _build_spill_batch forces spec_algorithm=NONE, so the
        # session decodes host-streamed one token per tick (kv_session_offload.py
        # _build_spill_batch). The capture MUST match that shape. Under a
        # speculative server (--speculative-config mtp, the deep-offload
        # reference) the runner's default capture shaping is TARGET_VERIFY with
        # num_tokens_per_bs = num_draft+1 and a spec_info; feeding that to the
        # spill-rung capture routes _sess_prepare_step into the C4 is_verify
        # early-return -> forward_extend -> _sess_blockwise_prefix_return_lse
        # (the verify twin), which (a) never records the multi-block DECODE body
        # the live plain tick replays, and (b) builds CPU tensors mid-capture
        # (torch.tensor([0, Q]) at flashinfer_backend.py:3493) -> hard
        # "Cannot copy between CPU and CUDA tensors during CUDA graph capture"
        # for any rung>=2 (deep host tail > 1 block). Force plain-decode capture
        # shaping (DECODE mode, 1 token/bs, no spec_info, no ragged verify) for
        # the spill rung so it records the capture-safe
        # _sess_blockwise_decode_return_lse_graph body, then restore. This is
        # the spill-graph analog of _build_spill_batch's spec=NONE erasure; the
        # C4 verify twin path is a SEPARATE (spec-in-tick, default OFF) concern
        # and is untouched.
        saved_mode = self.capture_forward_mode
        saved_ntpb = self.num_tokens_per_bs
        saved_ragged = self.ragged_verify_mode
        saved_hidden = self.capture_hidden_mode
        self.capture_forward_mode = ForwardMode.DECODE
        self.num_tokens_per_bs = 1
        self.ragged_verify_mode = False
        self.capture_hidden_mode = CaptureHiddenMode.NULL
        self._sess_force_plain_decode = True
        try:
            with ctx:
                # Reuse the head capture path; capture_prepare's batch is tagged
                # as a spill tick by the backend's install context (it sets the
                # flag + synthetic state that _sess_prepare_step reads).
                self.capture_one_shape(bs, forward, stream_idx, f"sessblk{rung}")
        finally:
            self.capture_forward_mode = saved_mode
            self.num_tokens_per_bs = saved_ntpb
            self.ragged_verify_mode = saved_ragged
            self.capture_hidden_mode = saved_hidden
            self._sess_force_plain_decode = False

    # -- #274 round 6: the dual-group lane's verify entry -------------------

    @contextlib.contextmanager
    def lane_verify_shape(self, num_tokens: Optional[int] = None):
        """Put this runner into the lane's VERIFY shape for the duration.

        Capture and replay have to agree on four things -- forward mode, tokens
        per slot, hidden mode and the graph-key variant -- and every one of them
        is read from ``self`` by code that has no idea a second entry exists.
        Swapping them around both sides is what makes the second entry work
        without a second runner: a second runner would have to call
        ``init_cuda_graph_state`` again on the SAME attention backend, which
        re-cuts the buffers the already-captured decode graphs point into.

        Restores unconditionally, so an exception inside a verify forward
        cannot leave the plain decode path looking like a verify.

        Mutable runner state under a CONCURRENT lane, deliberately: the object
        being mutated is the LANE's own decode graph runner, which the serving
        group never touches (it has its own), and within the lane the tick is
        serial. What this scope must not become is something the serving
        group's runner can also be put into.
        """
        assert self._lane_verify_tokens, "no lane verify entry on this runner"
        # Round 7a: which RUNG. Defaulting to the widest keeps a caller that
        # predates the ladder (and every single-rung deployment) on exactly the
        # entry it had.
        if num_tokens is None:
            num_tokens = max(self._lane_verify_tokens)
        assert num_tokens in self._lane_verify_tokens, (
            f"lane verify rung {num_tokens} is not on this runner's ladder "
            f"{self._lane_verify_tokens}"
        )
        saved = (
            self.capture_forward_mode,
            self.num_tokens_per_bs,
            self.capture_hidden_mode,
            self._lane_verify_active,
        )
        self.capture_forward_mode = ForwardMode.TARGET_VERIFY
        self.num_tokens_per_bs = num_tokens
        self.capture_hidden_mode = CaptureHiddenMode.FULL
        self._lane_verify_active = num_tokens
        try:
            yield
        finally:
            (
                self.capture_forward_mode,
                self.num_tokens_per_bs,
                self.capture_hidden_mode,
                self._lane_verify_active,
            ) = saved

    def lane_verify_can_replay(self, forward_batch: ForwardBatch) -> bool:
        """Is THIS batch the entry that was captured? Shape-exact, no padding.

        The lane's chain is a fixed K+1 tokens at bs 1; a batch of any other
        shape has no graph here and must stay eager rather than be padded into
        one (padding a verify would change which candidate rows exist).

        Round 7a: the rung in scope must be one that was actually RECORDED.
        A ladder that was thinned (or a rung whose capture raised) leaves the
        entry missing, and replaying the neighbouring rung's graph would be a
        silent wrong answer rather than a slow one.
        """
        return (
            self._lane_verify_active in self._lane_verify_captured
            and forward_batch.batch_size == 1
            and forward_batch.forward_mode.is_target_verify()
            and forward_batch.input_ids.numel() == self._lane_verify_active
        )

    def _lane_capture_verify(
        self, bs: int, stream_idx: Optional[int], num_tokens: Optional[int] = None
    ) -> None:
        """Record one rung of the lane's verify ladder (bs 1, TARGET_VERIFY).

        Mirrors ``_sess_capture_one_spill_rung`` and runs its swap the other
        way round: that pass forces a SPECULATIVE runner down to plain decode
        shaping for one extra graph, this one lifts a PLAIN runner to verify
        shaping for one extra graph. Both restore what they found.
        """
        if num_tokens is None:
            num_tokens = max(self._lane_verify_tokens)
        with self.lane_verify_shape(num_tokens):
            with torch_compile_decoration.patch_model(
                self.model_runner.model,
                bs in self.compile_bs,
                num_tokens=num_tokens,
                tp_group=self.model_runner.tp_group,
            ) as forward:
                self.capture_one_shape(
                    bs, forward, stream_idx, f"{self.LANE_VERIFY_VARIANT}{num_tokens}"
                )
        logger.info(
            "dual-group lane: verify graph captured (bs 1, %d tokens, "
            "TARGET_VERIFY, hidden FULL) beside the lane's decode graphs.",
            num_tokens,
        )

    # -- #274 round 7a: the dual-group lane's NEXTN head entry --------------

    def lane_draft_can_replay(self, forward_batch: ForwardBatch) -> bool:
        """Is THIS batch the lane head's captured draft forward?

        The head's live batch is a plain bs<=max_bs DECODE carrying an
        ``EagleDraftInput``. The hidden states are a graph INPUT copied into a
        fixed address by ``_lane_draft_load_hidden``; a batch that carries none
        has nothing to copy, so it must stay eager rather than replay against
        whatever the last round left in the buffer.
        """
        return (
            self._lane_draft_captured
            and forward_batch.forward_mode.is_decode()
            and forward_batch.batch_size <= self.max_bs
            and getattr(forward_batch.spec_info, "hidden_states", None) is not None
        )

    def _lane_draft_spec_info(self, num_tokens: int):
        """The capture-time stand-in for the lane head's ``EagleDraftInput``.

        This is the whole of the named gap round 6 left open: the generic
        decode capture builds ``spec_info=None`` and an MTP forward
        dereferences it (``forward_batch.spec_info.hidden_states``), so the
        head could not be captured at all. It carries only what the MTP forward
        reads -- the previous hidden states and the hidden mode -- because
        everything else on ``EagleDraftInput`` belongs to the tree/topk
        machinery the lane's chain deliberately does not have.
        """
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        return EagleDraftInput(
            hidden_states=self._lane_draft_hidden[:num_tokens],
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )

    def _lane_draft_load_hidden(self, forward_batch: ForwardBatch) -> None:
        """Copy the round's hidden states into the captured graph's input.

        The replay counterpart of ``_lane_draft_spec_info``. Only the REAL
        rows are written; padded slots are not read by the head any more than
        padded input_ids are.
        """
        hidden = getattr(forward_batch.spec_info, "hidden_states", None)
        if hidden is None:
            return
        rows = hidden.shape[0]
        self._lane_draft_hidden[:rows].copy_(hidden)

    def _lane_verify_spec_info(self, num_tokens: int):
        """The capture-time stand-in for the lane's ``EagleVerifyInput``.

        Shaped by ``build_lane_chain_verify_input`` -- the same builder the live
        round uses -- because the wrapper the capture creates is decided by what
        this object CARRIES, not by what it says: flashinfer picks its mask mode
        from the presence of ``custom_mask_buf`` (see ``_create_prefill_wrappers``),
        so a capture with no custom mask records a causal kernel that the live
        chain-masked round would then replay. The committed prefix length is the
        capture-time ``seq_len_fill_value``, which is what the padded seq_lens
        buffer holds at capture; the live mask is copied into the backend's
        fixed cuda-graph mask buffer per round, out of the graph.
        """
        from sglang.srt.model_executor.dual_group_lane import (
            build_lane_chain_verify_input,
        )

        return build_lane_chain_verify_input(
            [0] * num_tokens,
            int(self.seq_len_fill_value),
            device=self.device,
        )

    def capture_one_shape(
        self,
        size: int,
        forward: Callable,
        stream_idx: Optional[int] = None,
        variant_label: Optional[str] = None,
    ):
        # Weightless-KV WORKER (#133): the worker holds a meta model (zero
        # weights) and must NOT run model.forward. It captures the stripped
        # per-full-attention-layer DCP dispatch as its own decode graph so that
        # head + worker BOTH replay graphs, keeping their capture-time DCP
        # collectives in lockstep. Head-only / normal paths fall through
        # unchanged.
        if self.model_runner.is_weightless_worker:
            return self._capture_one_shape_weightless(size, stream_idx, variant_label)

        num_tokens = size * self.num_tokens_per_bs
        bs = self._ragged_capture_slots(num_tokens) if self.ragged_verify_mode else size

        # Sanity-check: --debug-cuda-graph requires breakable backend.
        if self.model_runner.server_args.debug_cuda_graph:
            assert isinstance(
                self.backend, BreakableCudaGraphBackend
            ), "Breakable CUDA graph is required for --debug-cuda-graph"

        forward_batch, attn_backend, pp_proxy_tensors = self.capture_prepare(
            bs, stream_idx=stream_idx, num_tokens=num_tokens
        )

        # All setup hooks below read get_attn_backend() (TboForwardBatchPreparer,
        # DeepEP adapter, …) so they must run inside the same ForwardContext
        # that wraps the warmup/capture forward.
        with forward_context(ForwardContext(attn_backend=attn_backend)):
            self.tbo_plugin.capture_one_batch_size(forward_batch, num_tokens=num_tokens)

            if forward_batch.lora_ids is not None:
                self.model_runner.lora_manager.prepare_lora_batch(forward_batch)

            attn_backend.init_forward_metadata_out_graph(forward_batch, in_capture=True)

            def run_once():
                # Graph-recordable metadata-prep hook. The unified memory pool
                # records ZERO translate nodes here: all its read/write translates
                # run eagerly in `init_forward_metadata_out_graph` (replay-prep), so
                # the captured graph reads already-physical locs. Base no-op for triton.
                attn_backend.init_forward_metadata_in_graph(forward_batch)

                # No invalidate_loc_cache() here: the unified pool translates its
                # locs in `init_forward_metadata_out_graph`, so no cache to invalidate.

                forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = (
                    None
                )
                set_dp_buffer_len(
                    forward_batch.global_dp_buffer_len,
                    num_tokens,
                    forward_batch.dp_padding_mode.is_max_len(),
                    forward_batch.global_num_tokens_cpu,
                )
                set_is_extend_in_batch(False)

                kwargs = {}
                if (
                    self.pp_size > 1
                    and "pp_proxy_tensors" in inspect.signature(forward).parameters
                ):
                    kwargs["pp_proxy_tensors"] = PPProxyTensors(
                        {k: v.clone() for k, v in pp_proxy_tensors.tensors.items()}
                    )
                if (
                    self.model_runner.spec_algorithm.is_dflash_family()
                    and self.model_runner.is_draft_model_runner
                    and "input_embeds" in inspect.signature(forward).parameters
                    and not hasattr(self.model_runner.model, "forward_embed")
                ):
                    kwargs["input_embeds"] = self.buffers.input_embeds[:num_tokens]

                out = forward(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                    **kwargs,
                )
                for capture_hook in self.model_runner.capture_tail_hooks:
                    capture_hook(self, out, forward_batch, num_tokens)
                return out

            self.deepep_adapter.capture(is_extend_in_batch=False)
            canary_ctx = (
                c.with_active_single_forward_manager(0)
                if (c := self.model_runner.canary_manager) is not None
                else contextlib.nullcontext()
            )
            # Full-physical write loc lives in the attention metadata (the backend's
            # `out_cache_loc_full_physical` -> KVWriteLoc.full_loc), so the runner
            # wires no buffer here. (SWA write loc rides the `swa_out_cache_loc` rail.)

            with canary_ctx:
                shape_key = self._make_graph_key(
                    self._capture_graph_size(bs=bs, num_tokens=num_tokens),
                    stream_idx,
                    variant_label,
                )
                post_warmup_hook = getattr(
                    self.model_runner.attn_backend,
                    "on_after_cuda_graph_warmup",
                    None,
                )
                maybe_flashinfer_autotune_speculative_draft(
                    self,
                    run_once,
                    post_warmup_hook=post_warmup_hook,
                    skip_logits=False,
                )
                self.backend.capture_one(
                    shape_key,
                    run_once,
                    dummies=None,
                    post_warmup_hook=post_warmup_hook,
                )

    def _capture_one_shape_weightless(
        self,
        size: int,
        stream_idx: Optional[int] = None,
        variant_label: Optional[str] = None,
    ):
        """Weightless-KV WORKER decode/verify-graph capture (#133, #143).

        Symmetric counterpart of the head's capture_one_shape: instead of
        recording model.forward (impossible — the worker's model is on the meta
        device), record the stripped per-full-attention-layer DCP dispatch
        (``ModelRunner._forward_weightless_worker``'s loop). The recorded
        graph issues the IDENTICAL DCP-group collective sequence as the head's
        graph (fused K+V all-gather, Q all-gather, LSE-merge per layer), so
        the two graphs' NCCL ops pair up on replay.

        #143: the recorded BODY follows ``capture_forward_mode``, exactly as the
        eager ``_forward_weightless_worker`` follows ``forward_batch.forward_mode``.
        With chain speculation on, this runner captures TARGET_VERIFY (the target
        never runs a plain DECODE step again) with ``num_tokens_per_bs ==
        num_draft_tokens``, and the worker must record the EXTEND dispatch — it
        reads ``forward_metadata.prefill_wrappers`` where the decode dispatch
        reads ``decode_wrappers``. ``capture_forward_mode`` is derived from
        server_args, so head and worker pick the same body without communicating.
        The two bodies emit the same op-tag tuple either way (see
        ``layers/dcp/lockstep.weightless_layer_op_tags``); what differs is which
        flashinfer wrapper the baked kernels came from, and getting THAT wrong is
        a wrong-answer bug rather than a hang.

        Attention metadata is prepped OUT of the graph
        (``init_forward_metadata_out_graph(in_capture=True)`` installs the decode
        cuda-graph wrappers onto ``forward_metadata``); the recorded body only
        runs the capturable per-layer dispatch. No logits, no sampling, no
        capture-tail hooks, no autotune (the lane forbids speculative decode).
        The guard is already disabled for the whole capture (see ``capture``)."""
        num_tokens = size * self.num_tokens_per_bs
        bs = size

        forward_batch, attn_backend, _pp_proxy_tensors = self.capture_prepare(
            bs, stream_idx=stream_idx, num_tokens=num_tokens
        )
        model_runner = self.model_runner
        # Hybrid-GDN models nest the flashinfer FULL-attention backend inside a
        # HybridLinearAttnBackend; the weightless_worker dispatch lives on the
        # flashinfer backend (mirrors _forward_weightless_worker).
        wl_attn = getattr(attn_backend, "full_attn_backend", attn_backend)
        # #143: pick the body by capture mode, mirroring the eager dispatch in
        # ModelRunner._forward_weightless_worker. Rank-uniform (capture_forward_mode
        # comes from server_args), so the head's captured body and this one always
        # describe the same step.
        if self.capture_forward_mode == ForwardMode.TARGET_VERIFY:
            wl_dispatch = wl_attn.forward_extend_weightless_worker
        elif self.capture_forward_mode.is_decode_or_idle():
            wl_dispatch = wl_attn.forward_decode_weightless_worker
        else:
            raise NotImplementedError(
                "weightless-KV worker graph capture: unsupported capture mode "
                f"{self.capture_forward_mode} (decode and target-verify only)."
            )

        with forward_context(ForwardContext(attn_backend=attn_backend)):
            # Out-of-graph metadata prep: installs the decode cuda-graph wrappers
            # onto attn_backend.forward_metadata (read by the dispatch below).
            attn_backend.init_forward_metadata_out_graph(forward_batch, in_capture=True)

            def run_once():
                # Graph-recordable metadata-prep hook (no-op for flashinfer decode
                # — everything static was prepped out-of-graph above).
                attn_backend.init_forward_metadata_in_graph(forward_batch)
                forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = (
                    None
                )
                set_dp_buffer_len(
                    forward_batch.global_dp_buffer_len,
                    num_tokens,
                    forward_batch.dp_padding_mode.is_max_len(),
                    forward_batch.global_num_tokens_cpu,
                )
                set_is_extend_in_batch(False)
                for layer in model_runner._weightless_attn_layers():
                    wl_dispatch(layer, forward_batch)
                # Sentinel output: the worker produces no logits. The backend
                # stores this (None) and returns it from replay; execute() maps
                # a None replay output to a logits-free ModelRunnerOutput.
                return None

            shape_key = self._make_graph_key(
                self._capture_graph_size(bs=bs, num_tokens=num_tokens),
                stream_idx,
                variant_label,
            )
            self.backend.capture_one(
                shape_key,
                run_once,
                dummies=None,
                post_warmup_hook=None,
            )

    def recapture_if_needed(self, forward_batch: ForwardBatch):

        # If the required capture_hidden_mode changes, we need to recapture the graph

        # These are the different factors that can influence the capture_hidden_mode
        capture_hidden_mode_required_by_forward_batch = (
            forward_batch.capture_hidden_mode
        )
        capture_hidden_mode_required_by_spec_info = (
            getattr(forward_batch.spec_info, "capture_hidden_mode", None)
            or CaptureHiddenMode.NULL
        )
        capture_hidden_mode_required_for_returning_hidden_states = (
            CaptureHiddenMode.FULL
            if self.enable_return_hidden_states
            else CaptureHiddenMode.NULL
        )

        # Determine the highest capture_hidden_mode required
        # (If we have FULL, we can emulate LAST or NULL)
        # (If we have LAST, we can emulate NULL)
        required_capture_hidden_mode = max(
            capture_hidden_mode_required_by_forward_batch,
            capture_hidden_mode_required_by_spec_info,
            capture_hidden_mode_required_for_returning_hidden_states,
        )

        # If the current hidden mode is no longer aligned with the required hidden mode, we need to set it to what is required and re-capture
        if self.capture_hidden_mode != required_capture_hidden_mode:
            self.capture_hidden_mode = required_capture_hidden_mode
            self.backend.cleanup()
            self.capture()

    def load_batch(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        # #274 round 7a: the lane head's second graph INPUT. Before anything
        # else, and before the pre-planned early return below, because both
        # paths replay the same graph and both need the hidden states at the
        # captured address.
        if self._lane_draft_capture:
            self._lane_draft_load_hidden(forward_batch)

        ragged_layout = (
            resolve_ragged_verify_layout(forward_batch)
            if self.ragged_verify_mode
            else None
        )
        is_ragged = ragged_layout is not None

        self.deepep_adapter.replay()

        if not forward_batch.needs_forward_metadata_init():
            # Pre-planned (plan-stream load_batch already ran).
            # In speculative decoding, these two fields are still needed.
            graph_size_key = (
                self._ragged_graph_size
                if is_ragged
                else self._capture_graph_size(
                    bs=self.bs, num_tokens=self.bs * self.num_tokens_per_bs
                )
            )
            if is_ragged:
                assert self.raw_num_token == ragged_layout.graph_num_tokens, (
                    f"stale ragged raw_num_token {self.raw_num_token} != "
                    f"{ragged_layout.graph_num_tokens}"
                )
            self.buffers.input_ids[: self.raw_num_token].copy_(forward_batch.input_ids)
            self.buffers.positions[: self.raw_num_token].copy_(forward_batch.positions)
            if (
                not is_ragged
                and self.model_runner.spec_algorithm.is_dflash_family()
                and self.model_runner.is_draft_model_runner
                and forward_batch.input_embeds is not None
            ):
                self.buffers.input_embeds[: self.raw_num_token].copy_(
                    forward_batch.input_embeds
                )
            variant_label = self._wl_variant_label(
                self._resolve_lora_variant(forward_batch)
            )
            stream_idx = get_current_stream_idx() if self.enable_pdmux else None
            self._replay_graph_key = self._make_graph_key(
                graph_size_key, stream_idx, variant_label
            )
            return

        buffers = self.buffers
        self.recapture_if_needed(forward_batch)

        raw_bs = forward_batch.batch_size

        if is_ragged:
            raw_num_token = ragged_layout.graph_num_tokens
            graph_size_key = self._ragged_graph_num_tokens(raw_num_token)
            assert graph_size_key == ragged_layout.graph_num_tokens, (
                f"ragged verify tier mismatch: runner tier {graph_size_key} != "
                f"layout graph_num_tokens {ragged_layout.graph_num_tokens}"
            )
            bs = self._ragged_capture_slots(graph_size_key)
            assert bs >= raw_bs, (
                f"ragged capture slots {bs} (tier {graph_size_key}) < raw_bs "
                f"{raw_bs}; the planner must reject this batch before replay"
            )
            padded_num_tokens = graph_size_key
        else:
            raw_num_token = raw_bs * self.num_tokens_per_bs
            if self.require_mlp_tp_gather:
                max_num_tokens = max(forward_batch.global_num_tokens_cpu)
                max_batch_size = (
                    max_num_tokens / self.num_tokens_per_bs
                    if self.model_runner.spec_algorithm.is_eagle()
                    or self.model_runner.spec_algorithm.is_standalone()
                    or self.model_runner.spec_algorithm.is_dflash_family()
                    else max_num_tokens
                )
                bs = self._pad_to_bucket(int(max_batch_size), self.capture_bs)
            else:
                bs = self._pad_to_bucket(raw_bs, self.capture_bs)
            padded_num_tokens = bs * self.num_tokens_per_bs
            graph_size_key = self._capture_graph_size(
                bs=bs, num_tokens=padded_num_tokens
            )

        self.buffer_registry.fill_from(
            forward_batch,
            raw_bs=raw_bs,
            padded_bs=bs,
            raw_num_tokens=raw_num_token,
            padded_num_tokens=padded_num_tokens,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if (
            not is_ragged
            and self.model_runner.spec_algorithm.is_dflash_family()
            and self.model_runner.is_draft_model_runner
            and forward_batch.input_embeds is not None
        ):
            buffers.input_embeds[:raw_num_token].copy_(forward_batch.input_embeds)
        # Padded tokens aren't read, so skip zeroing. Ragged input_ids arrive
        # from the planner already padded to the tier, invalid slots zeroed.
        if self.enable_two_batch_overlap:
            self.tbo_plugin.replay_prepare(
                forward_mode=self.capture_forward_mode,
                bs=bs,
                num_token_non_padded=len(forward_batch.input_ids),
                spec_info=forward_batch.spec_info,
            )
        if (
            not is_ragged
            and forward_batch.forward_mode.is_idle()
            and forward_batch.spec_info is not None
        ):
            forward_batch.spec_info.custom_mask = buffers.custom_mask
        if self.enable_pdmux:
            stream_idx = get_current_stream_idx()
            attn_backend = self.model_runner.decode_attn_backend_group[stream_idx]
        else:
            attn_backend = self.attn_backend
        fb_view = build_replay_fb_view(
            forward_batch=forward_batch,
            buffers=buffers,
            bs=bs,
            raw_bs=raw_bs,
            num_tokens=padded_num_tokens,
            seq_len_fill_value=self.seq_len_fill_value,
            capture_forward_mode=self.capture_forward_mode,
            is_encoder_decoder=self.is_encoder_decoder,
        )
        attn_backend.init_forward_metadata_out_graph(fb_view)

        self.raw_bs = raw_bs
        self.raw_num_token = raw_num_token
        self.bs = bs
        if is_ragged:
            self._ragged_graph_size = graph_size_key

        if self.model_runner.hisparse_coordinator is not None:
            self.model_runner.hisparse_coordinator.num_real_reqs.fill_(raw_bs)

        variant_label = self._wl_variant_label(
            self._resolve_lora_variant(forward_batch)
        )
        stream_idx = get_current_stream_idx() if self.enable_pdmux else None
        self._replay_graph_key = self._make_graph_key(
            graph_size_key, stream_idx, variant_label
        )

    def _ragged_graph_num_tokens(self, total_verify_tokens: int) -> int:
        from sglang.srt.speculative.ragged_verify import round_up_grid

        return round_up_grid(total_verify_tokens, self.capture_num_tokens)

    def execute(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        timer_ctx = (
            self.model_runner.device_timer.wrap(
                metadata={"category": forward_batch.forward_mode.name.lower()}
            )
            if self.model_runner.device_timer
            else contextlib.nullcontext()
        )
        # Publish a read-done event for the WAR barrier: a cuda-graph forward
        # finishes its shared req_to_token / SWA reads at this pre-replay
        # snapshot, so plain DECODE and block-draft TARGET_VERIFY qualify.
        publish_read_done = forward_batch.forward_mode.is_decode() or (
            forward_batch.forward_mode.is_target_verify()
            and self.model_runner.spec_algorithm.is_dflash_family()
        )
        # Exception: breakable-graph verify replays (captured forward metadata)
        # re-read req_to_token *during* replay, so the pre-replay snapshot is
        # too early -- record the event after replay instead.
        read_done_post_replay = (
            publish_read_done
            and forward_batch.forward_mode.is_target_verify()
            and self.attn_backend.use_captured_forward_metadata_for_breakable_cuda_graph
        )
        with timer_ctx, self.backend.replay_session():
            self.load_batch(forward_batch, pp_proxy_tensors)
            if envs.SGLANG_LOG_DECODE_GRAPH_KEY.get():
                logger.info(
                    "Decode graph replay: worker=%s key_size=%s (%s) mode=%s raw_bs=%d%s",
                    "draft" if self.model_runner.is_draft_model_runner else "target",
                    self._replay_graph_key.size,
                    "num_tokens" if self.ragged_verify_mode else "bs",
                    forward_batch.forward_mode.name,
                    forward_batch.batch_size,
                    (
                        f" slots={self._ragged_capture_slots(self._replay_graph_key.size)}"
                        if self.ragged_verify_mode
                        else ""
                    ),
                )
            if publish_read_done and not read_done_post_replay:
                read_done = self.device_module.Event()
                read_done.record()
                self.model_runner.war_fastpath_read_done_event = read_done
            output = self.backend.replay(self._replay_graph_key, forward_batch)
            if read_done_post_replay:
                read_done = self.device_module.Event()
                read_done.record()
                self.model_runner.war_fastpath_read_done_event = read_done

        if output is None:
            # Weightless-KV WORKER decode graph (#133): the captured dispatch
            # produces no logits — the stored/replayed sentinel is None. Nothing
            # to slice; the caller wraps a logits-free ModelRunnerOutput.
            return None

        if isinstance(output, LogitsProcessorOutput):
            if self.is_dllm:
                next_token_logits = None
                full_logits = (
                    output.full_logits[: self.raw_num_token]
                    if output.full_logits is not None
                    else None
                )
            else:
                full_logits = None
                next_token_logits = (
                    output.next_token_logits[: self.raw_num_token]
                    if output.next_token_logits is not None
                    else None
                )

            return LogitsProcessorOutput(
                next_token_logits=next_token_logits,
                full_logits=full_logits,
                hidden_states=(
                    output.hidden_states[: self.raw_num_token]
                    if output.hidden_states is not None
                    else None
                ),
                # T156 stage 3 dual capture: aux concat captured alongside the
                # final hidden states (cross-algorithm schedule mode only).
                cross_aux_hidden_states=(
                    output.cross_aux_hidden_states[: self.raw_num_token]
                    if output.cross_aux_hidden_states is not None
                    else None
                ),
                customized_info=output.customized_info,
            )
        else:
            assert isinstance(output, PPProxyTensors)
            return PPProxyTensors({k: v[: self.bs] for k, v in output.tensors.items()})

    def get_spec_info(self, num_tokens: int):
        spec_info = None
        # S0 (deep-offload): during the spill-rung capture the batch is forced
        # to a PLAIN bs=1 DECODE shape (see _sess_capture_one_spill_rung), which
        # must carry no spec_info -- exactly like the live plain spill tick
        # (_build_spill_batch spec_algorithm=NONE). Return None regardless of the
        # server's speculative config.
        if getattr(self, "_sess_force_plain_decode", False):
            return None
        # #274 round 6: the lane's verify entry. The lane's runner is not
        # speculative (its args view clears the algorithm), so without this the
        # branches below all fall through to None and the capture would record
        # a maskless causal kernel for a chain-masked forward.
        if self._lane_verify_active:
            return self._lane_verify_spec_info(num_tokens)
        # #274 round 7a: the lane's NEXTN head, and the reason its capture was
        # a named gap until now -- this branch is what stops the fall-through
        # to None that an MTP forward dereferences.
        if self._lane_draft_capture:
            return self._lane_draft_spec_info(num_tokens)
        if (
            self.model_runner.spec_algorithm.is_eagle()
            or self.model_runner.spec_algorithm.is_standalone()
        ):
            from sglang.srt.speculative.eagle_info import EagleVerifyInput

            if self.model_runner.is_draft_model_runner:
                raise RuntimeError("This should not happen.")
            else:

                capture_mode = (
                    CaptureHiddenMode.NULL
                    if self.model_runner.spec_algorithm.is_standalone()
                    else CaptureHiddenMode.FULL
                )
                spec_info = EagleVerifyInput(
                    draft_token=None,
                    custom_mask=self.buffers.custom_mask,
                    positions=None,
                    retrieve_index=None,
                    retrieve_next_token=None,
                    retrieve_next_sibling=None,
                    retrieve_cum_len=None,
                    spec_steps=self.speculative_num_steps,
                    topk=self.model_runner.server_args.speculative_eagle_topk,
                    draft_token_num=self.speculative_num_draft_tokens,
                    capture_hidden_mode=capture_mode,
                    seq_lens_sum=None,
                    seq_lens_cpu=None,
                )
                # MTP models (e.g. deepseek_nextn) read spec_info.hidden_states
                spec_info.hidden_states = torch.zeros(
                    (num_tokens, self.model_runner.model_config.hidden_size),
                    dtype=self.model_runner.dtype,
                    device=self.model_runner.device,
                )
        elif self.model_runner.spec_algorithm.is_dflash_family():
            from sglang.srt.speculative.dflash_info import DFlashVerifyInput
            from sglang.srt.speculative.dflash_utils import (
                resolve_dflash_verify_mask_policy,
            )

            # Avoid enabling custom-mask modes during graph capture for backends that
            # can express DFLASH verify via their built-in causal path.
            _, build_custom_mask = resolve_dflash_verify_mask_policy(
                self.model_runner.attn_backend
            )
            spec_info = DFlashVerifyInput(
                draft_token=None,
                positions=None,
                draft_token_num=self.num_tokens_per_bs,
                custom_mask=(
                    None
                    if (self.model_runner.is_draft_model_runner or not build_custom_mask)
                    else self.buffers.custom_mask
                ),
                capture_hidden_mode=(
                    CaptureHiddenMode.NULL
                    if self.model_runner.is_draft_model_runner
                    else CaptureHiddenMode.FULL
                ),
                ragged_verify_layout=self._capture_ragged_verify_layout(num_tokens),
            )

        elif self.model_runner.spec_algorithm.is_ngram():
            from sglang.srt.speculative.ngram_info import NgramVerifyInput

            spec_info = NgramVerifyInput(
                draft_token=None,
                custom_mask=self.buffers.custom_mask,
                positions=None,
                retrieve_index=None,
                retrieve_next_token=None,
                retrieve_next_sibling=None,
                draft_token_num=self.num_tokens_per_bs,
            )
            spec_info.capture_hidden_mode = CaptureHiddenMode.NULL

        return spec_info
