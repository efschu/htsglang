from typing import Optional, Tuple, Union

import torch

from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnKernelBackend,
    get_linear_attn_decode_backend,
    get_linear_attn_prefill_backend,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    PAD_SLOT_ID,
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.utils import is_cpu, is_cuda, is_hip, is_npu
from sglang.srt.utils.common import rank0_log

if not is_cpu():
    from sglang.srt.layers.attention.fla.chunk_delta_h import (
        CHUNK_SIZE as FLA_CHUNK_SIZE,
    )

if is_cuda() or is_hip():
    from sglang.jit_kernel.triton.gdn_fused_proj import fused_qkv_split_gdn_prefill

MAX_FUSED_QKV_SPLIT_DIM = 8192

if is_cuda():
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn as causal_conv1d_fn_cuda,
    )

    causal_conv1d_fn = causal_conv1d_fn_cuda
elif is_npu():
    from sgl_kernel_npu.fla.fused_gdn_gating import fused_gdn_gating_npu
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu,
        causal_conv1d_update_npu,
    )

    fused_gdn_gating = fused_gdn_gating_npu
    causal_conv1d_fn = causal_conv1d_fn_npu
    causal_conv1d_update = causal_conv1d_update_npu
elif is_cpu():
    from sgl_kernel.mamba import causal_conv1d_fn_cpu, causal_conv1d_update_cpu

    causal_conv1d_fn = causal_conv1d_fn_cpu
    causal_conv1d_update = causal_conv1d_update_cpu
    fused_gdn_gating = torch.ops.sgl_kernel.fused_gdn_gating_cpu


def maybe_set_default_flashinfer_gdn_prefill(model_runner: ModelRunner) -> None:
    """Use FlashInfer for the narrow SM100 GDN prefill domain we validated."""
    args = model_runner.server_args
    if (
        args.linear_attn_prefill_backend is not None
        or args.linear_attn_backend != "triton"
        or args.enable_page_major_kv_layout
        or not is_cuda()
        or torch.cuda.get_device_capability()[0] != 10
    ):
        return

    # Extra-buffer strategies need intermediate state checkpoints.
    if args.uses_mamba_radix_cache and args.mamba_radix_cache_strategy != "no_buffer":
        return

    cuda_version = torch.version.cuda
    chunk_size = args.chunked_prefill_size
    config = model_runner.hybrid_gdn_config
    if (
        cuda_version is None
        or int(cuda_version.split(".", 1)[0]) < 13
        or args.enable_dynamic_chunking
        or chunk_size is None
        or not 1 <= chunk_size <= 8192
        or getattr(config, "linear_key_head_dim", None) != 128
        or getattr(config, "linear_value_head_dim", None) != 128
        or model_runner.req_to_token_pool.mamba_pool.mamba_cache.temporal.dtype
        != torch.bfloat16
    ):
        return

    from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
        is_flashinfer_gdn_prefill_available,
    )

    if is_flashinfer_gdn_prefill_available():
        args.linear_attn_prefill_backend = "flashinfer"
        rank0_log("Defaulting SM100 GDN prefill backend to FlashInfer.")


class GDNKernelDispatcher:
    """Dispatches GDN kernel calls to the appropriate backend per mode."""

    def __init__(
        self,
        decode_backend: LinearAttnKernelBackend,
        prefill_backend: LinearAttnKernelBackend,
    ):
        triton_kernel = TritonGDNKernel()
        self.tree_verify_kernel = triton_kernel

        cutedsl_kernel = None
        if decode_backend.is_triton():
            self.decode_kernel = triton_kernel
        elif decode_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("GDN CuTe DSL backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
                CuteDSLGDNKernel,
            )

            cutedsl_kernel = CuteDSLGDNKernel()
            self.decode_kernel = cutedsl_kernel
        elif decode_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer GDN backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                FlashInferGDNKernel,
            )

            flashinfer_kernel = FlashInferGDNKernel()
            self.decode_kernel = flashinfer_kernel
        else:
            raise ValueError(f"Unsupported GDN decode backend: {decode_backend}")

        if prefill_backend.is_triton():
            self.extend_kernel = triton_kernel
        elif prefill_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("GDN CuTe DSL backend requires CUDA")
            # Reuse the CuteDSL kernel if already created for decode
            if cutedsl_kernel is None:
                from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
                    CuteDSLGDNKernel,
                )

                cutedsl_kernel = CuteDSLGDNKernel()
            # The CuteDSL prefill kernel is validated on SM100/SM103 only.
            # Everywhere else -- SM90 (Hopper) below it, consumer Blackwell
            # (SM12x) above it -- fall back to Triton so users can pick
            # `cutedsl` uniformly across hardware.
            if cutedsl_kernel.supports_prefill:
                self.extend_kernel = cutedsl_kernel
            else:
                rank0_log(
                    "CuTe DSL GDN prefill is not supported on this GPU "
                    "(requires SM100/SM103). Falling back to Triton for prefill."
                )
                self.extend_kernel = triton_kernel
        elif prefill_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer GDN backend requires CUDA")
            # Reuse the FlashInfer kernel if already created for decode
            if decode_backend.is_flashinfer():
                self.extend_kernel = flashinfer_kernel
            else:
                from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                    FlashInferGDNKernel,
                )

                flashinfer_kernel = FlashInferGDNKernel()
                self.extend_kernel = flashinfer_kernel
        else:
            raise ValueError(f"Unsupported GDN prefill backend: {prefill_backend}")

        # Verify kernel: use FlashInfer when the selected FlashInfer kernel
        # supports MTP verify. SM90 uses the fp32-state path; SM100 uses the
        # bf16-state adapter in FlashInferGDNKernel.
        if (
            decode_backend.is_flashinfer() or prefill_backend.is_flashinfer()
        ) and flashinfer_kernel.supports_target_verify:
            self.verify_kernel = flashinfer_kernel
        else:
            self.verify_kernel = triton_kernel

        self.supports_packed_decode = getattr(
            self.decode_kernel, "supports_packed_decode", False
        )

        rank0_log(
            f"GDN kernel dispatcher: decode={self.decode_kernel.__class__.__name__}, "
            f"extend={self.extend_kernel.__class__.__name__}, "
            f"verify={self.verify_kernel.__class__.__name__} "
            f"packed_decode={self.supports_packed_decode}"
        )

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Attempt packed decode. Returns output tensor or None if
        the decode kernel does not support packed decode."""
        if not self.supports_packed_decode:
            return None
        return self.decode_kernel.packed_decode(
            mixed_qkv,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            **kwargs,
        )

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.decode_kernel.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> tuple:
        return self.extend_kernel.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # FlashInfer verify supports a linear MTP chain. Tree-shaped drafts
        # carry parent indices and must use Triton even when decode/prefill use
        # FlashInfer.
        verify_kernel = (
            self.tree_verify_kernel
            if kwargs.get("retrieve_parent_token") is not None
            else self.verify_kernel
        )
        return verify_kernel.target_verify(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )


class GDNAttnBackend(MambaAttnBackendBase):
    """Attention backend for GDN (Gated Delta Network) linear attention."""

    needs_cpu_seq_lens: bool = False

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
        )
        if not is_cpu() and not is_npu():
            assert (
                self.conv_states_shape[-1] < FLA_CHUNK_SIZE
            ), f"{self.conv_states_shape[-1]=} should be less than {FLA_CHUNK_SIZE}"

        decode_backend = get_linear_attn_decode_backend()
        prefill_backend = get_linear_attn_prefill_backend()
        self.kernel_dispatcher = GDNKernelDispatcher(decode_backend, prefill_backend)
        self.verify_intermediate_state_indices = torch.arange(
            self.req_to_token_pool.size, dtype=torch.int32, device=model_runner.device
        )
        # #444: request-private conv window for TARGET_VERIFY. See
        # `_target_verify_conv` for why the verify conv must not run on the
        # persistent pool row. Rows are the rows the intermediate caches
        # already use, so the buffer is bounded by the same spec state size and
        # carries no new indexing rule. Allocated once at backend construction
        # (never on the hot path, never inside a graph capture) and shared
        # across layers: within one forward the layers run in sequence on one
        # stream.
        #
        # #450b correction: those rows are BATCH POSITIONS, not slots, so they
        # are not self-owning -- two batches both start at row 0. What keeps
        # concurrent verifies apart is that this buffer and the intermediate
        # caches are per-runner: the #274 lane is its own ModelRunner with its
        # own req_to_token_pool and its own attention backend. That is a
        # precondition, pinned in
        # test/registered/unit/spec/test_verify_intermediate_row_ownership_450.py.
        self._verify_conv_scratch: Optional[torch.Tensor] = None
        mamba_cache = model_runner.req_to_token_pool.mamba_pool.mamba_cache
        if isinstance(mamba_cache, MambaPool.SpeculativeState):
            conv = mamba_cache.conv[0]
            intermediate = mamba_cache.intermediate_conv_window[0]
            self._verify_conv_scratch = torch.empty(
                (intermediate.shape[1], conv.shape[-2], conv.shape[-1]),
                dtype=conv.dtype,
                device=conv.device,
            )

    def _target_verify_conv(
        self,
        layer: RadixLinearAttention,
        mixed_qkv_reshaped: torch.Tensor,
        conv_states: torch.Tensor,
        cache_indices: torch.Tensor,
        intermediate_conv_window_cache: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
        retrieve_next_token: Optional[torch.Tensor],
        retrieve_next_sibling: Optional[torch.Tensor],
        retrieve_parent_token: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run the TARGET_VERIFY conv on a request-private copy of the window.

        The conv kernel is intentionally in-place: STEP 2 of
        ``_causal_conv1d_update_kernel``
        (``mamba/causal_conv1d_triton.py:752``) stores the shifted window back
        into whatever ``conv_state`` it was handed, and that store is not gated
        on ``SAVE_INTERMEDIATE``. Handing it the persistent pool row therefore
        advanced that row over EVERY drafted step, accepted or not, and the
        pool only became correct again once
        ``update_mamba_state_after_mtp_verify``
        (``hybrid_linear_attn_backend.py:1098``) scattered the accepted step
        back out of the intermediate cache.

        Between those two points the pool row held the last candidate's
        window: a state that is neither the pre-verify state nor the committed
        one. Single-stream that window is unobservable — nothing between the
        verify forward and the commit reads the recurrent pool. It is
        observable for anything that reads the same pool concurrently, and
        mutate-then-repair is only ever as correct as the absence of such a
        reader. (#450b: the #274 dual-group lane is NOT that reader -- it is a
        second ModelRunner with its own pool, `_build_lane_under_scope`. The
        window is real; the concurrent reader named in #444 was not.)

        Running on a private copy removes the window instead of timing it. The
        persistent row is now written exactly once per verify, by the commit,
        from the intermediate cache — which already carries the full
        ``dim x (K-1)`` window of every step, so no information is lost: the
        commit's scatter overwrites every element of the row it touches
        (``mamba/mamba_state_scatter_triton.py:_fused_conv_window_scatter_with_mask_kernel``),
        and the row count it covers is the verify batch
        (``speculative/spec_utils.py:commit_mamba_states_after_verify``).
        The committed bytes are therefore unchanged; only the intermediate
        window stops being published.
        """
        scratch = self._verify_conv_scratch
        assert scratch is not None, (
            "TARGET_VERIFY reached GDNAttnBackend without a speculative mamba "
            "cache: the private conv window was never allocated."
        )
        batch_size = cache_indices.shape[0]
        rows = intermediate_state_indices[:batch_size]
        # #444b: a verify replayed into a larger captured graph carries
        # PAD_SLOT_ID (-1) in the tail of `cache_indices`
        # (`hybrid_linear_attn_backend.py:378`/`:459` fill the per-bs buffers
        # with it, `:555` re-stamps it over the padding requests on every
        # replay, `:97-99` does the same eagerly). The conv kernel tolerates
        # that by construction -- USE_PAD_SLOT returns before the padded lane
        # reads or writes anything (`causal_conv1d_triton.py:659-662`) -- but
        # `index_select` / `index_copy_` bounds-check, and a -1 row is the
        # device-side assert `indexSelectSmallIndex: srcIndex <
        # srcSelectDimSize`. Both sides therefore have to be masked, and the
        # mask has to stay on the device: this runs per layer per verify, so
        # `.item()` / `.nonzero()` would put a D2H sync on the hot path and
        # would not be capturable in a CUDA graph at all.
        pad_lanes = cache_indices < 0
        # Gather side: clamp the padded rows onto row 0 so the read is in
        # bounds. Their seeded values are never consumed -- the kernel skips
        # exactly those lanes below -- so which valid row is read is arbitrary.
        # Seed the private rows with the persistent window the kernel needs as
        # its prior state. `rows` is the same index space the intermediate
        # caches use, so the private buffer inherits their ownership rule
        # instead of introducing a second one (#450b: that rule is per-runner
        # isolation, not per-row identity).
        scratch.index_copy_(
            0,
            rows.to(torch.int64),
            conv_states.index_select(
                0, cache_indices.masked_fill(pad_lanes, 0).to(torch.int64)
            ),
        )
        # Scatter side: hand the kernel PAD_SLOT_ID for the padded lanes so it
        # still skips them. Masking only the gather would leave every padded
        # lane pointing at a real scratch row, and the kernel would run the
        # conv on it instead of not running it -- writing a private row and an
        # intermediate window the pre-#444a path never wrote.
        kernel_rows = rows.masked_fill(pad_lanes, PAD_SLOT_ID)
        return causal_conv1d_update(
            mixed_qkv_reshaped,
            scratch,
            layer.conv_weights,
            layer.bias,
            layer.activation,
            conv_state_indices=kernel_rows,
            intermediate_conv_window=intermediate_conv_window_cache,
            intermediate_state_indices=rows,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_parent_token=retrieve_parent_token,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        if self.forward_metadata.has_mamba_track_mask:
            self.forward_metadata.mamba_track_mask_indices = (
                forward_batch.mamba_track_mask.nonzero(as_tuple=True)[0]
            )
            self.forward_metadata.conv_states_mask_indices = (
                forward_batch.mamba_track_indices[
                    self.forward_metadata.mamba_track_mask_indices
                ]
            )

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices
        # GDN ReplaySSM (slice 1a): per-layer ring slices + the once-per-forward
        # per-row write cursor. All None unless --enable-linear-replayssm, so the
        # legacy dispatch below is byte-identical when the flag is off.
        replayssm_write_pos = self.forward_metadata.replayssm_write_pos
        # GDN ReplaySSM (slice 2b): per-row force-flush at radix track
        # boundaries (None unless --enable-linear-replayssm). When present the
        # kernel folds the ring into temporal[slot] on the snapshot steps.
        replayssm_force_flush = self.forward_metadata.replayssm_force_flush
        replayssm_d = layer_cache.replayssm_d
        replayssm_k = layer_cache.replayssm_k
        replayssm_g = layer_cache.replayssm_g

        assert isinstance(mixed_qkv, torch.Tensor)
        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            layer.conv_weights,
            layer.bias,
            layer.activation,
            conv_state_indices=cache_indices,
        )

        # Skip split + reshape + separate gating kernel by consuming
        # the packed mixed_qkv directly in a single fused Triton kernel.
        if self.kernel_dispatcher.supports_packed_decode:
            core_attn_out = self.kernel_dispatcher.packed_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                scale=layer.head_k_dim**-0.5,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                num_v_heads=layer.num_v_heads,
                head_v_dim=layer.head_v_dim,
                replayssm_d=replayssm_d,
                replayssm_k=replayssm_k,
                replayssm_g=replayssm_g,
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
            )
            self._track_mamba_state_decode(
                forward_batch, conv_states, ssm_states, cache_indices
            )
            return core_attn_out

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        # Reshape from [bs, h*d] to [1, bs, h, d]
        bs = forward_batch.batch_size
        query = query.view(1, bs, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, bs, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, bs, layer.num_v_heads, layer.head_v_dim)

        core_attn_out = self.kernel_dispatcher.decode(
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices
        )

        return core_attn_out

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        assert isinstance(mixed_qkv, torch.Tensor)
        seq_len = mixed_qkv.shape[0]

        is_target_verify = forward_batch.forward_mode.is_target_verify()
        forward_metadata = self.forward_metadata

        query_start_loc = forward_metadata.query_start_loc
        cache_indices = forward_metadata.mamba_cache_indices
        retrieve_next_token = forward_metadata.retrieve_next_token
        retrieve_next_sibling = forward_metadata.retrieve_next_sibling
        retrieve_parent_token = forward_metadata.retrieve_parent_token

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal
        if is_target_verify:
            assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
            intermediate_state_cache = mamba_cache_params.intermediate_ssm
            intermediate_conv_window_cache = (
                mamba_cache_params.intermediate_conv_window[0]
            )
            intermediate_state_indices = self.verify_intermediate_state_indices
        else:
            has_initial_states = forward_batch.extend_prefix_lens > 0

        # Page-major envelope: the prefill kernels (CUDA causal_conv1d_fwd,
        # chunk_gated_delta_rule) write state back in place assuming a contiguous
        # slot layout, so they silently drop the write to the strided envelope
        # pool. Run them on contiguous per-sequence copies (identity-indexed) and
        # scatter the result back. No-op for the default contiguous pool.
        # TODO(ch-wan): drop these .contiguous() copies by making the prefill conv
        # and chunk_gated_delta_rule kernels honor the pool's real slot stride +
        # int64 indexing, like packed_decode / causal_conv1d_update already do.
        needs_state_gather = (not is_target_verify) and (
            not conv_states.is_contiguous() or not ssm_states.is_contiguous()
        )
        if needs_state_gather:
            conv_states_contig = conv_states[cache_indices].contiguous()
            ssm_states_contig = ssm_states[cache_indices].contiguous()
            state_cache_indices = torch.arange(
                cache_indices.shape[0],
                device=cache_indices.device,
                dtype=cache_indices.dtype,
            )
        else:
            conv_states_contig = conv_states
            ssm_states_contig = ssm_states
            state_cache_indices = cache_indices

        if is_target_verify:
            batch_size = seq_len // forward_batch.spec_info.draft_token_num
            draft_token_num = forward_batch.spec_info.draft_token_num
            mixed_qkv_reshaped = mixed_qkv.view(
                batch_size, draft_token_num, -1
            ).transpose(1, 2)
            mixed_qkv_processed = self._target_verify_conv(
                layer,
                mixed_qkv_reshaped,
                conv_states,
                cache_indices[:batch_size],
                intermediate_conv_window_cache,
                intermediate_state_indices,
                retrieve_next_token,
                retrieve_next_sibling,
                retrieve_parent_token,
            )
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
        else:
            mixed_qkv = mixed_qkv.transpose(0, 1)
            if forward_metadata.has_mamba_track_mask:
                mixed_qkv_to_track = mixed_qkv[
                    :, forward_metadata.track_conv_indices
                ].transpose(0, 1)
                conv_states[forward_metadata.conv_states_mask_indices] = (
                    mixed_qkv_to_track
                )

            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                layer.conv_weights,
                layer.bias,
                activation=layer.activation,
                conv_states=conv_states_contig,
                has_initial_state=has_initial_states,
                cache_indices=state_cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            ).transpose(0, 1)[:seq_len]

        actual_seq_len = mixed_qkv.shape[0]
        qkv_dim = layer.q_dim + layer.k_dim + layer.v_dim
        if (is_cuda() or is_hip()) and qkv_dim <= MAX_FUSED_QKV_SPLIT_DIM:
            query, key, value = fused_qkv_split_gdn_prefill(
                mixed_qkv,
                layer.num_q_heads,
                layer.num_k_heads,
                layer.num_v_heads,
                layer.head_q_dim,
                layer.head_k_dim,
                layer.head_v_dim,
            )
        else:
            query, key, value = torch.split(
                mixed_qkv,
                [layer.q_dim, layer.k_dim, layer.v_dim],
                dim=-1,
            )
            query = query.view(1, actual_seq_len, layer.num_q_heads, layer.head_q_dim)
            key = key.view(1, actual_seq_len, layer.num_k_heads, layer.head_k_dim)
            value = value.view(1, actual_seq_len, layer.num_v_heads, layer.head_v_dim)

        if is_target_verify:
            core_attn_out = self.kernel_dispatcher.target_verify(
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                q=query,
                k=key,
                v=value,
                a=a,
                b=b,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                intermediate_states_buffer=intermediate_state_cache,
                intermediate_state_indices=intermediate_state_indices,
                cache_steps=forward_batch.spec_info.draft_token_num,
                retrieve_parent_token=retrieve_parent_token,
            )
        else:
            g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
            core_attn_out, last_recurrent_state, h = self.kernel_dispatcher.extend(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                ssm_states=ssm_states_contig,
                cache_indices=state_cache_indices,
                query_start_loc=query_start_loc,
            )

            if is_npu() and last_recurrent_state is not None:
                last_recurrent_state = last_recurrent_state.to(
                    ssm_states.dtype, copy=False
                )
                ssm_states[cache_indices] = last_recurrent_state

            if needs_state_gather:
                # Scatter the in-place-updated contiguous copies back to the
                # strided envelope pool (advanced indexing handles the strides).
                conv_states[cache_indices] = conv_states_contig
                ssm_states[cache_indices] = ssm_states_contig

            if h is not None:
                self._track_mamba_state_extend(
                    forward_batch, h, ssm_states, forward_metadata
                )

        return core_attn_out
