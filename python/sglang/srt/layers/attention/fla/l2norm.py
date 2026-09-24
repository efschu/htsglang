# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/l2norm.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os
from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl

from sglang.srt.layers.attention.fla.utils import input_guard

BT_LIST = [8, 16, 32, 64, 128]


# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8, 16, 32]
#     ],
#     key=["D"],
# )
@triton.jit
def l2norm_fwd_kernel1(
    x,
    y,
    D,
    BD: tl.constexpr,
    eps,
):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    # Compute mean and variance
    cols = tl.arange(0, BD)
    mask = cols < D
    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=0)
    b_rstd = 1 / tl.sqrt(b_var + eps)
    # tl.store(Rstd + i_t, rstd)
    # Normalize and apply linear transformation
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)


# @triton.autotune(
#     configs=[
#         triton.Config({"BT": BT}, num_warps=num_warps)
#         for num_warps in [1, 2, 4, 8, 16]
#         for BT in BT_LIST
#     ],
#     key=["D", "NB"],
# )
@triton.jit
def l2norm_fwd_kernel(
    x,
    y,
    eps,
    NB: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


# ---------------------------------------------------------------------------
# RUNTIME-T VARIANT (SGLANG_FLA_L2NORM_RUNTIME_T=1; set for group P by the weg2
# launcher's --p-host-overlap). ``l2norm_fwd_kernel`` above takes the row count
# ``T`` (tokens x heads) and ``NB = cdiv(T, 2048)`` as ``tl.constexpr``, so
# every new token count is a new Triton kernel: compiled or read from the disk
# cache and loaded into the context, on the scheduler thread, inside the
# launch. MEASURED weg2xsn423 P: 63 cold loads of l2norm_fwd_kernel, 3.20 s
# of load windows (every other Triton kernel loaded once per rank and
# variant); py-spy PP0 134 of 2263 samples in its _init_handles -- short and
# last chunks, whose length changes per prompt. Upstream FLA passes T at run
# time (``@triton.jit(do_not_specialize=["T"])``, fla/modules/l2norm.py); this
# is the same kernel body with T as a run-time bound: one compile per (D, BD).
# ---------------------------------------------------------------------------
L2NORM_RUNTIME_T_ENV = "SGLANG_FLA_L2NORM_RUNTIME_T"


def l2norm_runtime_t_on() -> bool:
    return os.environ.get(L2NORM_RUNTIME_T_ENV, "") == "1"


@triton.jit(do_not_specialize=["T"])
def l2norm_fwd_kernel_rt(
    x,
    y,
    eps,
    T,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


def l2norm_fwd(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None
):
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    # allocate output
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    # rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    if D <= 512 and l2norm_runtime_t_on():
        # same kernel body, T a run-time bound: no new kernel per token count
        l2norm_fwd_kernel_rt[(triton.cdiv(T, 16),)](
            x,
            y,
            eps,
            T,
            D=D,
            BT=16,
            BD=BD,
            num_warps=8,
            num_stages=3,
        )
    elif D <= 512:
        NB = triton.cdiv(T, 2048)

        def grid(meta):
            return (triton.cdiv(T, meta["BT"]),)

        l2norm_fwd_kernel[grid](
            x,
            y,
            eps,
            NB=NB,
            T=T,
            D=D,
            BD=BD,
            BT=16,
            num_warps=8,
            num_stages=3,
        )
    else:
        l2norm_fwd_kernel1[(T,)](
            x,
            y,
            eps=eps,
            D=D,
            BD=BD,
            num_warps=8,
            num_stages=3,
        )

    return y.view(x_shape_og)


class L2NormFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(ctx, x, eps=1e-6, output_dtype=None):
        return l2norm_fwd(x, eps, output_dtype)


def l2norm(
    x: torch.Tensor, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    return L2NormFunction.apply(x, eps, output_dtype)


l2_norm = l2norm


class L2Norm(nn.Module):

    def __init__(self, eps: float = 1e-6, output_dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.eps = eps
        self.output_dtype = output_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2norm(x, self.eps, self.output_dtype)
