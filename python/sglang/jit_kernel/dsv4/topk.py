from __future__ import annotations

from typing import Optional

import torch

from sglang.jit_kernel.utils import (
    cache_once_per_arch,
    is_arch_support_pdl,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)

from .utils import make_name


@cache_once_per_arch
def _jit_topk_v1_module(topk: int):
    args = make_cpp_args(is_arch_support_pdl())
    assert topk in (512, 1024), "Only support topk=512 or 1024"
    return load_jit(
        make_name(f"topk_v1_{topk}"),
        *args,
        cuda_files=["deepseek_v4/topk_v1.cuh"],
        cuda_wrappers=[("topk_transform", f"TopKKernel<{args}>::transform")],
        extra_cuda_cflags=[f"-DSGL_TOPK={topk}"],
    )


@cache_once_per_arch
def _jit_topk_v2_module():
    # v2 is universal: topk (<= 2048) is a runtime argument, not a compile-time
    # constant, so a single module serves every k.
    return load_jit(
        make_name("topk_v2"),
        cuda_files=["deepseek_v4/topk_v2.cuh"],
        cuda_wrappers=[
            ("topk_transform", "TopKKernel::transform"),
            ("topk_plan", "TopKKernel::plan"),
        ],
    )


class NegativeSeqLenError(ValueError):
    """A top-k kernel was handed a negative sequence length.

    Raised by the wrapper, before the launch, because the kernels themselves
    cannot report it: they read the length as ``uint32_t`` and simply run off
    the end of the score buffer.
    """


def _check_seq_lens_non_negative(seq_lens: torch.Tensor, where: str) -> None:
    """Env-gated precondition check; see SGLANG_DSV4_CHECK_TOPK_SEQ_LENS.

    Off by default: reading the minimum back from the device costs a sync on
    every decode step. Turn it on when bringing up a new producer of
    ``seq_lens``, which is where the negative rows come from.
    """
    from sglang.srt.environ import envs

    if not envs.SGLANG_DSV4_CHECK_TOPK_SEQ_LENS.get():
        return
    if seq_lens.numel() == 0:
        return
    min_len = int(seq_lens.min().item())
    if min_len < 0:
        raise NegativeSeqLenError(
            f"{where}: seq_lens must be non-negative, got min {min_len} over "
            f"{seq_lens.numel()} rows. The device kernel reads this int32 "
            f"buffer as uint32_t, so {min_len} becomes {min_len & 0xFFFFFFFF} "
            f"and the top-k walks off the score buffer (illegal memory "
            f"access). Clamp padded / idle rows to 0, which is the valid way "
            f"to say 'no tokens'."
        )


def topk_transform_512(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    """Fused top-k + page-table transform (DeepSeek-V4 top-k v1 kernel).

    IMPORTANT: every entry of ``seq_lens`` must be NON-NEGATIVE. Both kernels
    behind this wrapper read the int32 buffer as ``uint32_t`` -- the JIT one
    at ``deepseek_v4/topk_v1.cuh`` (``const uint32_t seq_len =
    seq_lens[work_id]``) and the HIP one at
    ``elementwise/deepseek_v4_topk.cu`` (``static_cast<uint32_t>(length)`` in
    the naive path) -- so a negative length (e.g. -4 from a DP-padded / idle
    companion row) reinterprets as ~4e9, the row scans far past the end of its
    score buffer, and the launch dies with an illegal memory access. A length
    of 0 is the valid way to express "no tokens".

    This is the same precondition :func:`plan_topk_v2` and
    :func:`topk_transform_512_v2` document. Set
    ``SGLANG_DSV4_CHECK_TOPK_SEQ_LENS=1`` to have it verified here (raising
    :class:`NegativeSeqLenError`) instead of surfacing as a device fault; the
    check is off by default because it forces a device sync per call.
    """
    _check_seq_lens_non_negative(seq_lens, "topk_transform_512")
    if is_hip_runtime():
        torch.ops.sgl_kernel.deepseek_v4_topk_transform_512(
            scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices
        )
    else:
        module = _jit_topk_v1_module(out_page_indices.shape[1])
        module.topk_transform(
            scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices
        )


# metadata is (batch+1, 2) int32: row 0 = {cluster_threshold, num_cluster_items};
# rows 1..N = {batch_id, seq_len} of items routed to the persistent cluster pool.
_PLAN_METADATA_INTS_PER_BATCH = 2


def plan_topk_v2(seq_lens: torch.Tensor, static_threshold: int = 0) -> torch.Tensor:
    """Preprocess the per-batch routing plan for :func:`topk_transform_512_v2`.

    IMPORTANT: every entry of ``seq_lens`` must be NON-NEGATIVE. The device
    kernel reads the int32 buffer as ``uint32_t``, so a negative length (e.g.
    -4 from a DP-padded / idle-companion row) reinterprets as ~4e9, poisons
    the plan, and drives the transform kernel into an illegal memory access.
    Producers of padded rows must clamp their lengths to 0 (0 selects the
    trivial all-(-1) output path, which is safe). Set
    ``SGLANG_DSV4_CHECK_TOPK_SEQ_LENS=1`` to have that verified here.
    """
    _check_seq_lens_non_negative(seq_lens, "plan_topk_v2")
    module = _jit_topk_v2_module()
    bs = seq_lens.shape[0]
    metadata = seq_lens.new_empty(bs + 1, _PLAN_METADATA_INTS_PER_BATCH)
    module.topk_plan(seq_lens, metadata, static_threshold)
    return metadata


def topk_transform_512_v2(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    metadata: torch.Tensor,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    """Fused top-k + page-table transform (DeepSeek-V4 top-k v2 kernel).

    IMPORTANT: every entry of ``seq_lens`` must be NON-NEGATIVE, and
    ``metadata`` must come from :func:`plan_topk_v2` over the same ``seq_lens``
    values. The kernel reads lengths as ``uint32_t``: a negative entry
    reinterprets as a ~4e9-token sequence, sending the row down the cluster
    path over garbage scores and crashing with an illegal memory access
    (GLM 5.2 MTP DP-idle companion rows hit exactly this). A length of 0 is
    the valid way to express "no tokens": the row takes the trivial path and
    the output is all -1. Set ``SGLANG_DSV4_CHECK_TOPK_SEQ_LENS=1`` to have
    that verified here.
    """
    _check_seq_lens_non_negative(seq_lens, "topk_transform_512_v2")
    module = _jit_topk_v2_module()
    module.topk_transform(
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        page_size,
        metadata,
        out_raw_indices,
    )
