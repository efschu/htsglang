"""A tensor that shares another tensor's memory without owning it.

CUDA-graph capture uses this to keep a handle on an address while letting the
caching allocator reclaim the block: the graph pool keeps the memory pinned for
as long as any captured graph is alive, so the Python-side reference is not the
thing that must keep it valid. Holding a normal reference instead is *correct*
but pins every per-layer intermediate for the process's lifetime, which for a
breakable prefill capture is one live buffer set per layer per token bucket.

sgl-kernel provides this as `at::from_blob(data_ptr, sizes, strides, options)`.
Its wheel is cubin-only with a gencode floor of sm_80 and has no ROCm build
below gfx942, so a Turing (sm75) or gfx900 rank does not have it -- and the
module-scope import used to raise from inside a graph break point, where it
surfaced as an unrelated state-machine assertion. The fallback below is the
same construction expressed through the CUDA array interface, which needs no
extension build. CAPABILITY, NOT VENDOR: what decides is whether sgl-kernel is
importable here, not which vendor this is. The op launches no kernel, so
importability -- not the sm_80 run floor -- is the right level to ask at.
"""

import logging
from typing import Any, Callable, Union

import torch

from sglang.srt.utils.common import (
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    is_xpu,
    sgl_kernel_importable,
)

logger = logging.getLogger(__name__)

# Array-interface type codes for the dtypes the interface can express.
# bfloat16 and the fp8 formats have no code at all; they must be refused rather
# than mapped onto a same-width code, which would reinterpret the bytes.
_CAI_TYPESTR = {
    torch.bool: "|b1",
    torch.uint8: "|u1",
    torch.int8: "|i1",
    torch.int16: "<i2",
    torch.int32: "<i4",
    torch.int64: "<i8",
    torch.float16: "<f2",
    torch.float32: "<f4",
    torch.float64: "<f8",
}

# dtypes already reported as degrading to a strong reference.
_identity_warned: set = set()


def _cuda_array_interface(tensor: torch.Tensor) -> dict:
    """Describe ``tensor``'s memory for the CUDA array interface.

    Strides are in BYTES there and in elements in torch; a contiguous tensor
    passes ``None``, which the interface defines as C-contiguous.
    """
    typestr = _CAI_TYPESTR.get(tensor.dtype)
    if typestr is None:
        raise TypeError(f"the CUDA array interface has no type code for {tensor.dtype}")
    itemsize = tensor.element_size()
    strides = (
        None if tensor.is_contiguous() else tuple(s * itemsize for s in tensor.stride())
    )
    return {
        "data": (tensor.data_ptr(), False),
        "shape": tuple(tensor.shape),
        "strides": strides,
        "typestr": typestr,
        "version": 2,
    }


class _ArrayInterfaceView:
    """Carrier object for the interface dict.

    It deliberately holds no reference to the source tensor: torch keeps THIS
    object alive for the lifetime of the tensor it builds, so a reference here
    would pin the storage and defeat the whole point.
    """

    __slots__ = ("__cuda_array_interface__",)

    def __init__(self, interface: dict):
        self.__cuda_array_interface__ = interface


def _is_device_tensor(tensor: torch.Tensor) -> bool:
    return tensor.device.type != "cpu"


def _warn_identity_once(tensor: torch.Tensor, reason: str) -> None:
    if tensor.dtype in _identity_warned:
        return
    _identity_warned.add(tensor.dtype)
    logger.warning(
        "weak_ref_tensor falls back to a strong reference for %s (%s). That is "
        "correct but keeps every captured intermediate in the graph memory "
        "pool, so CUDA-graph capture uses noticeably more memory; disable "
        "prefill CUDA graphs if the rank runs short.",
        tensor.dtype,
        reason,
    )


def _native_weak_ref_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """sgl-kernel-free equivalent, for ranks the wheel does not cover.

    Builds a non-owning tensor over the same memory through the CUDA array
    interface. Anything it cannot express degrades to the source tensor itself
    -- a strong reference, which is safe (strictly more stable than a weak one)
    but keeps the block allocated.
    """
    if tensor.numel() == 0 or not _is_device_tensor(tensor):
        # Nothing to reclaim, and no device memory to describe.
        return tensor
    try:
        interface = _cuda_array_interface(tensor)
    except TypeError as exc:
        _warn_identity_once(tensor, str(exc))
        return tensor
    try:
        return torch.as_tensor(_ArrayInterfaceView(interface), device=tensor.device)
    except Exception as exc:  # a platform that does not ingest the interface
        _warn_identity_once(tensor, f"torch refused the array interface: {exc}")
        return tensor


def _select_weak_ref_tensor() -> Callable[[torch.Tensor], torch.Tensor]:
    if is_cuda() or is_hip() or is_musa() or is_xpu():
        if sgl_kernel_importable():
            from sgl_kernel import weak_ref_tensor as impl

            return impl
        return _native_weak_ref_tensor
    if is_npu():
        from torch_npu._C import _weak_ref_tensor as impl

        return impl
    raise NotImplementedError(
        "weak_ref_tensor is implemented only for CUDA, XPU, and NPU."
    )


_impl: Callable[[torch.Tensor], torch.Tensor] | None = None


def weak_ref_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Non-owning view of ``tensor``'s memory.

    The implementation is chosen on FIRST USE, not at import: the choice reads
    the local backend, and a module that refuses to import on a CPU-only host
    cannot be unit-tested there, nor imported by a caller that never uses it.
    """
    global _impl
    if _impl is None:
        _impl = _select_weak_ref_tensor()
    return _impl(tensor)


def weak_ref_tensors(
    tensors: Union[torch.Tensor, list[torch.Tensor], tuple[torch.Tensor]],
) -> Union[torch.Tensor, list[Any], tuple[Any], Any]:
    """
    Convenience function to create weak references to tensors,
    for single tensor, list of tensors or tuple of tensors.
    """
    if isinstance(tensors, torch.Tensor):
        return weak_ref_tensor(tensors)
    if isinstance(tensors, list):
        return [weak_ref_tensor(t) for t in tensors]
    if isinstance(tensors, tuple):
        return tuple(weak_ref_tensor(t) for t in tensors)
    raise ValueError("Invalid type for tensors")
