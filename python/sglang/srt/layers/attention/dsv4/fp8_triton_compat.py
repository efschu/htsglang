"""Architecture-portable FP8 E4M3 decoding for the DeepSeek-V4 Triton kernels.

The V4 KV cache stores its nope half as FP8 E4M3 *by architecture* (448 fp8
nope + 64 bf16 rope + 7 ue8m0 scales per token, all inside a ``uint8`` pool);
``--kv-cache-dtype`` does not change that. The Triton kernels that read it
therefore need an E4M3 decode on every architecture the fork serves.

Triton's NVIDIA backend only admits the ``fp8e4nv`` element type from compute
capability 8.9 (Ada) upward. On SM80/SM86 (Ampere) it refuses with::

    ValueError: type fp8e4nv not supported in this architecture.
                The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')

and it refuses at *compile* time on the ``def`` line: handing a
``torch.float8_e4m3fn`` tensor to a ``@triton.jit`` kernel is enough to fail,
even when the pointer is never dereferenced. A branch inside the kernel does
not help. The argument itself has to be a ``uint8`` view, and the decode has
to be done from the raw byte.

This module supplies both halves of that:

* :func:`triton_fp8e4nv_supported` -- the per-device gate (see #343: one
  process can hold parts of one model on two cards of different
  architecture, so this is a question about a *device*, never about the
  process);
* :func:`e4m3fn_bits_to_f32` -- a ``@triton.jit`` device function that decodes
  one E4M3-FN byte to float32 with pure integer/FP arithmetic, no fp8 element
  type involved.

Callers pass the nope pointer as the ``uint8`` view when the gate is off and
as the ``fp8`` view when it is on, and select the branch with a
``tl.constexpr`` flag, so each architecture compiles only the form it can.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.utils.common import (
    get_device_capability_no_init,
    is_cuda,
    per_device_gate,
)

# Triton's NVIDIA backend gained `fp8e4nv` with Ada. Ampere (8.0/8.6) has only
# `fp8e4b15` and `fp8e5`, neither of which is the E4M3-FN encoding the V4 pool
# writes, so those cards take the manual decode.
TRITON_FP8E4NV_MIN_CAPABILITY: Tuple[int, int] = (8, 9)


@per_device_gate
def triton_fp8e4nv_supported(device_id: int) -> bool:
    """Whether Triton can type a pointer as ``fp8e4nv`` on this device.

    Non-CUDA vendors are left on their native path: ROCm's V4 pool stores
    E4M3-FNUZ (bias 8, a single NaN code at 0x80), which is a different
    encoding from the E4M3-FN this module decodes, and Triton's AMD backend
    accepts it natively.
    """
    if not is_cuda():
        return True
    return get_device_capability_no_init(device_id) >= TRITON_FP8E4NV_MIN_CAPABILITY


def nope_cache_view(
    cache_uint8: torch.Tensor, fp8_dtype: torch.dtype
) -> Tuple[torch.Tensor, bool]:
    """Pick the tensor to hand a kernel for the FP8 nope region.

    Returns ``(tensor, fp8_native)``. When ``fp8_native`` is False the tensor
    is the ``uint8`` view and the kernel must decode with
    :func:`e4m3fn_bits_to_f32`; passing the fp8 view instead would fail to
    compile before the kernel body is ever considered.
    """
    native = triton_fp8e4nv_supported(_device_id_of(cache_uint8))
    return (cache_uint8.view(fp8_dtype) if native else cache_uint8), native


def _device_id_of(tensor: torch.Tensor) -> Optional[int]:
    return tensor.device.index if tensor.device.type == "cuda" else None


@triton.jit
def e4m3fn_bits_to_f32(raw):
    """Decode raw E4M3-FN bytes to float32 without the ``fp8e4nv`` type.

    ``raw`` is the value of a ``uint8`` load; the result is bit-identical to
    ``torch.uint8.view(torch.float8_e4m3fn).to(torch.float32)`` for all 254
    finite codes -- including the signed zero at 0x80 -- and NaN for the two
    NaN codes 0x7F/0xFF, whose payload torch does not preserve either.

    E4M3-FN: 1 sign, 4 exponent (bias 7), 3 mantissa, no infinities.
      exponent 0      -> subnormal, value = mantissa * 2**-9
      exponent 15,m=7 -> NaN (the only NaN codes)
      otherwise       -> (1 + mantissa/8) * 2**(exponent - 7)
    """
    bits = raw.to(tl.int32)
    exponent = (bits >> 3) & 15
    mantissa = (bits & 7).to(tl.float32)

    subnormal = mantissa * 0.001953125  # 2**-9
    normal = (1.0 + mantissa * 0.125) * tl.exp2((exponent - 7).to(tl.float32))

    magnitude = tl.where(exponent == 0, subnormal, normal)
    magnitude = tl.where((exponent == 15) & ((bits & 7) == 7), float("nan"), magnitude)

    # Multiply rather than negate so that 0x80 yields -0.0 like torch does.
    sign = tl.where((bits >> 7) == 1, -1.0, 1.0)
    return magnitude * sign
