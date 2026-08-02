"""CuTe DSL kernels for GDN (Gated Delta Network) linear attention.

Decode path uses the existing ``cutedsl_fused_sigmoid_gating_delta_rule_update``
(works on SM90+).

Prefill (extend) path uses the ported vLLM SM100 chunkwise kernel
(``chunk_gated_delta_rule_cutedsl``). Requires SM100/SM103 and ``head_k_dim == 128``.
"""

import logging
from typing import Optional

import torch

from sglang.jit_kernel.cutedsl_gdn import cutedsl_fused_sigmoid_gating_delta_rule_update
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.utils import cuda_sm_in_range, get_cuda_sm

logger = logging.getLogger(__name__)


def _supports_cutedsl_prefill() -> bool:
    """True iff this is an SM10x part, the family this prefill kernel targets.

    Two separate namespace traps meet here.

    Vendor (#171): "Blackwell" is an NVIDIA statement, so it is asked in the
    NVIDIA namespace. The bare ``major >= 10`` this replaces was vendor-blind
    and the namespaces collide: gfx1030 reports ``(10, 3)`` -- the same
    integer as a B300 -- and gfx1100 reports ``(11, 0)``, so RDNA2/RDNA3 cards
    identified as Blackwell and were routed into a tcgen05/TMA kernel that
    cannot exist on them.

    Family: within NVIDIA, "SM100 or newer" is still too wide. The ported
    chunk kernel is validated on the datacenter SM100/SM103 parts only.
    Consumer Blackwell reports ``(12, 0)``, which is >= 10 but is a different
    architecture -- an open upper bound silently routes an RTX 50-series card
    into the same tcgen05 path. Hence a half-open range, not a floor.
    """
    return cuda_sm_in_range((10, 0), (11, 0))


class CuteDSLGDNKernel(LinearAttnKernelBase):
    """CuTe DSL kernel for GDN.

    Decode: ``cutedsl_fused_sigmoid_gating_delta_rule_update`` (SM90+).
    Extend (prefill): chunkwise ``chunk_gated_delta_rule_cutedsl``
    (SM100/SM103 only, ``head_k_dim`` must be 128). On every other
    architecture the prefill path is unsupported; callers should query
    :attr:`supports_prefill` and fall back to another backend (e.g. Triton).
    """

    def __init__(self):
        # The SM10x extend kernel uses tcgen05/TMA-bulk-swizzle features that
        # neither SM90 nor consumer Blackwell (SM12x) provides in this form.
        # The decode kernel does work on SM90+.
        self.supports_prefill = _supports_cutedsl_prefill()

        # Heavy CuteDSL imports are deferred to extend() so SM90 boxes can
        # still construct the kernel just for decode.
        self._extend_fn: Optional[callable] = None
        self._prepare_meta_fn: Optional[callable] = None
        self._l2norm_fn: Optional[callable] = None

    def _ensure_extend_loaded(self, head_k_dim: int) -> None:
        if self._extend_fn is not None:
            return
        if not self.supports_prefill:
            sm = get_cuda_sm()
            raise RuntimeError(
                "CuTe DSL GDN prefill requires SM100/SM103; got "
                + (f"SM{sm}." if sm is not None else "a non-NVIDIA device.")
            )
        if head_k_dim != 128:
            raise RuntimeError(
                f"CuTe DSL GDN prefill requires head_k_dim=128, got {head_k_dim}."
            )
        from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd
        from sglang.srt.layers.attention.linear.kernels.gdn_blackwell import (
            chunk_gated_delta_rule_cutedsl,
            prepare_metadata_cutedsl,
        )

        self._extend_fn = chunk_gated_delta_rule_cutedsl
        self._prepare_meta_fn = prepare_metadata_cutedsl
        self._l2norm_fn = l2norm_fwd
        logger.info("Using CuTe DSL GDN prefill (SM10x)")

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
        return cutedsl_fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
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
        head_k_dim = k.shape[-1]
        self._ensure_extend_loaded(head_k_dim)

        total_seq_len = q.shape[1]
        num_v_heads = v.shape[2]
        head_v_dim = v.shape[3]

        # L2 norm Q/K outside the kernel (same as flashinfer path).
        q_norm = self._l2norm_fn(q[0].contiguous()).unsqueeze(0)
        k_norm = self._l2norm_fn(k[0].contiguous()).unsqueeze(0)
        v_in = v[0].contiguous().unsqueeze(0)
        # Kernel expects log-space float32 gate per (token, v-head).
        g_in = g[0].to(torch.float32).unsqueeze(0)
        beta_in = beta[0].to(torch.float32).unsqueeze(0)

        cu_seqlens = query_start_loc.to(torch.int32)

        # Pool gather: remap padding (-1) to the last (sentinel) slot.
        ssm_cache_indices = torch.where(
            cache_indices >= 0,
            cache_indices,
            ssm_states.shape[0] - 1,
        ).to(torch.long)
        initial_state = ssm_states[ssm_cache_indices].contiguous()

        chunk_indices, chunk_offsets = self._prepare_meta_fn(
            cu_seqlens, total_seq_len, chunk_size=64
        )

        output, final_state = self._extend_fn(
            q=q_norm,
            k=k_norm,
            v=v_in,
            g=g_in,
            beta=beta_in,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
        )

        ssm_states.index_copy_(
            0,
            ssm_cache_indices,
            final_state.to(ssm_states.dtype),
        )

        # Match Triton extend interface: (output, last_recurrent_state, h).
        # We've already written state back, so no need to return it.
        return output, None, None

    def target_verify(self, *args, **kwargs):
        raise NotImplementedError("CuteDSLGDNKernel does not support target_verify")
