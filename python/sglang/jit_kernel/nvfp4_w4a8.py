"""NVFP4 W4A8-g16 linear for sm_86 on the INT8 tensor cores, reading the NATIVE NVFP4 layout.

Backlog #38 (user order 2026-09-25): the RTX 5090 computes NVFP4 natively (W4A4, FP4 tensor cores), the
RTX 3080s compute the SAME bytes on their INT8 tensor cores (W4A8). One byte layout on every card, nothing
reshaped at the flip.

Inputs are exactly what ``ModelOptFp4LinearMethod.process_weights_after_loading`` produces on its native
(sm_120, backend ``cutlass``) branch:

* ``weight``            uint8 [N_p32, K_p32/2], E2M1 with element 2j in the LOW nibble of byte j
                        (``pad_nvfp4_weight``);
* ``weight_scale``      float8_e4m3fn [ceil128(N), ceil4(K/16)], 128x4-swizzled
                        (the ``weight_scale_interleaved`` binding, same bytes as ``utils.swizzle_blockscale``);
                        a K-shard may instead be passed as its tile view [N_pad/128, Ks*128] (any row stride);
* ``weight_scale_2``    fp32, the global weight scale (``layer.weight_scale_2.max()``). ``alpha`` and
                        ``input_scale_inv`` belong to the FP4 activation of the 5090 and are not used here.

Arithmetic (per 16-element block b): INT8 activations (per token, symmetric) x (2 * E2M1) on
``mma.m16n8k16.s8`` -> exact INT32 block sum -> exact FP32 via the magic-number accumulator init ->
``s_b * S_b`` exact in one FFMA -> FP32 accumulation -> x s_x[m] * weight_scale_2 / 2 -> bf16.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.jit_kernel.utils import cache_once_per_arch, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

#: Tile variants of the GEMM, selected by the token count M (see config_for_m in the .cuh).
# fmt: off
_BM_BY_CONFIG = (8, 16, 32, 48, 64, 128, 128, 64,   # 0-7   FP32 epilogue (EPI 0)
                 8, 16, 32, 48, 64, 128, 64,        # 8-14  exact INT64 epilogue (EPI 1)
                 8, 16,                             # 15-16 EPI 0, 4-stage decode tiles
                 128, 64,                           # 17-18 EPI 2 (one I2F per mma)
                 256, 128,                          # 19-20 EPI 0, 64x64 warp tiles
                 128, 256)                          # 21-22 g32 RATE PROBE (not valid for NVFP4)
# fmt: on
#: configs 8..14 use the exact INT64 epilogue (8-byte split-K partials)
EXACT_CONFIGS = frozenset(range(8, 15))
_BN = 128
_BK = 64


@cache_once_per_arch
def _jit_nvfp4_w4a8_module() -> Module:
    return load_jit(
        "nvfp4_w4a8_sm86",
        cuda_files=["gemm/nvfp4_w4a8_sm86.cuh"],
        cuda_wrappers=[
            ("gemm", "nvfp4_w4a8_gemm"),
            ("quant_act", "nvfp4_w4a8_quant_act"),
        ],
        # The FP32 epilogue relies on IEEE denormals (E4M3 scales enter as value * 2^-120) and on correctly
        # rounded division in the activation quantiser; pin both instead of inheriting a default.
        extra_cuda_cflags=[
            "-ftz=false",
            "-prec-div=true",
            "-prec-sqrt=true",
            "-lineinfo",
        ],
    )


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def config_for_m(m: int) -> int:
    if m <= 8:
        return 0
    if m <= 16:
        return 1
    if m <= 32:
        return 2
    if m <= 48:
        return 3
    if m <= 64:
        return 4
    return 5


def exact_config_for_m(m: int) -> int:
    """The exact INT64-epilogue tile variant for M tokens."""
    if m <= 8:
        return 8
    if m <= 16:
        return 9
    if m <= 32:
        return 10
    if m <= 48:
        return 11
    return 12


def choose_split_k(
    m: int, n_rows: int, k_padded: int, sm_count: int, config: int = -1
) -> int:
    """Split K only when the output grid cannot fill the card (decode on narrow layers, e.g. down_proj)."""
    cfg = config_for_m(m) if config < 0 else config
    grid = -(-n_rows // _BN) * -(-m // _BM_BY_CONFIG[cfg])
    num_kt = -(-k_padded // _BK)
    target = 2 * sm_count
    if grid >= target or num_kt < 16:
        return 1
    splits = min(-(-target // grid), num_kt // 8, 8)
    return max(splits, 1)


def nvfp4_w4a8_quantize_activation(
    x: torch.Tensor, k_padded: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """bf16 [M, K] -> (int8 [M, Kp], fp32 [M]); same arithmetic as ``per_token_quant_int8``,
    columns K..Kp zero-filled (the weight's K padding)."""
    assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.stride(-1) == 1
    m, k = x.shape
    kp = k if k_padded is None else int(k_padded)
    xq = torch.empty((m, kp), dtype=torch.int8, device=x.device)
    xs = torch.empty((m,), dtype=torch.float32, device=x.device)
    _jit_nvfp4_w4a8_module().quant_act(x, xq, xs)
    return xq, xs


def nvfp4_w4a8_gemm(
    xq: torch.Tensor,
    xs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    out_features: int,
    *,
    split_k: int = 0,
    config: int = -1,
    workspace: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """int8 [M, Kp] x NVFP4 native weight -> bf16 [M, out_features].

    ``split_k=0`` chooses automatically; the split partials use ``workspace`` (fp32, >= splits*M*N floats),
    allocated here when not given. ``weight_global_scale`` is a 1-element fp32 device tensor.
    """
    m = xq.shape[0]
    kp = xq.shape[1]
    if split_k <= 0:
        split_k = choose_split_k(
            m,
            weight.shape[0],
            kp,
            _sm_count(
                xq.device.index
                if xq.device.index is not None
                else torch.cuda.current_device()
            ),
            config,
        )
    if out is None:
        out = torch.empty((m, out_features), dtype=torch.bfloat16, device=xq.device)
    num_kt = -(-kp // _BK)
    split_k = max(1, min(split_k, num_kt))
    if split_k > 1:
        exact = config in EXACT_CONFIGS
        need = split_k * m * out_features * (2 if exact else 1)
        if workspace is None or workspace.numel() * workspace.element_size() < need * 4:
            workspace = torch.empty((need,), dtype=torch.float32, device=xq.device)
    elif workspace is None:
        workspace = torch.empty((1,), dtype=torch.float32, device=xq.device)
    gs = weight_global_scale.reshape(1).to(torch.float32)
    ws = (
        weight_scale.view(torch.uint8)
        if weight_scale.dtype != torch.uint8
        else weight_scale
    )
    _jit_nvfp4_w4a8_module().gemm(
        out,
        xq,
        xs,
        weight,
        ws,
        gs,
        workspace,
        int(out_features),
        int(split_k),
        int(config),
    )
    return out


def nvfp4_w4a8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    out_features: int,
    bias: Optional[torch.Tensor] = None,
    *,
    split_k: int = 0,
) -> torch.Tensor:
    """bf16 [..., K] -> bf16 [..., out_features]. The drop-in for ``ModelOptFp4LinearMethod.apply`` on sm_86."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    xq, xs = nvfp4_w4a8_quantize_activation(x2, k_padded=weight.shape[1] * 2)
    y = nvfp4_w4a8_gemm(
        xq, xs, weight, weight_scale, weight_global_scale, out_features, split_k=split_k
    )
    if bias is not None:
        y = y + bias
    return y.view(*shape[:-1], out_features)
