"""Which FlashMLA kernel exists on which device, for the DeepSeek-V4 backend.

The DSV4 backend can reach three FlashMLA implementations, each with a
different hardware domain:

* ``sgl_kernel.flash_mla.flash_mla_with_kvcache`` and
  ``flash_mla_sparse_fwd`` -- hand-written CUDA, Hopper and datacenter
  Blackwell only. Both state their own domain in the error they raise:
  ``Sparse Attention Forward Kernel is only supported on SM90a and SM100f
  architectures`` / ``Unsupported architecture for sparse decode fwd``.
* ``flash_mla_sm120.flash_mla_with_kvcache_sm120`` -- a portable dense entry
  point over the same packed V4 KV layout, with a flashinfer, a Triton and a
  pure-torch implementation behind it. Written for consumer Blackwell, but
  nothing in it is SM120-specific; Ampere is in exactly the same position
  (no FlashMLA CUDA kernel) and uses it too. The name is kept for continuity
  with upstream.

The gates below are phrased as "does this kernel exist on this device",
never as a list of architecture names, so a new consumer architecture lands
on the portable path by default instead of on a kernel that will refuse it.

Every gate takes a device id and is cached per device (#343): one process
can hold parts of one model on two cards of different architecture, so a
process-wide answer is wrong on a heterogeneous rig.
"""

import logging
from typing import Optional

from sglang.srt.environ import envs
from sglang.srt.utils.common import (
    get_device_capability_no_init,
    is_cuda,
    per_device_gate,
)

logger = logging.getLogger(__name__)

# Compute-capability majors that ship a FlashMLA CUDA kernel in sgl-kernel:
# 9 = Hopper (SM90a), 10 = datacenter Blackwell (SM100f/SM103).
_FLASH_MLA_CUDA_MAJORS = (9, 10)

# flashinfer's sparse-MLA kernels carry @supported_compute_capability([120, 121]).
_SM12X_MAJOR = 12


@per_device_gate
def flash_mla_cuda_kernel_supported(device_id: int) -> bool:
    """Whether ``sgl_kernel.flash_mla.flash_mla_with_kvcache`` runs here.

    False on Ampere, Ada and consumer Blackwell, which all take
    ``flash_mla_with_kvcache_sm120`` instead.
    """
    if not is_cuda():
        return True
    return get_device_capability_no_init(device_id)[0] in _FLASH_MLA_CUDA_MAJORS


@per_device_gate
def flash_mla_sparse_fwd_supported(device_id: int) -> bool:
    """Whether ``sgl_kernel.flash_mla.flash_mla_sparse_fwd`` runs here.

    Same domain as the dense CUDA kernel. There is no portable counterpart
    for the sparse-prefill kernel, so a device outside this domain takes the
    dense branch instead -- which consumes the same indexer top-k selection
    and is therefore a change of kernel, not of attention support.
    """
    if not is_cuda():
        return True
    return get_device_capability_no_init(device_id)[0] in _FLASH_MLA_CUDA_MAJORS


@per_device_gate
def _is_sm12x(device_id: int) -> bool:
    if not is_cuda():
        return False
    return get_device_capability_no_init(device_id)[0] == _SM12X_MAJOR


def resolve_flashmla_fallback_backend(device_id: Optional[int] = None) -> str:
    """Which implementation ``flash_mla_with_kvcache_sm120`` should use here.

    ``SGLANG_SM120_FLASHMLA_BACKEND`` defaults to ``flashinfer``, whose
    sparse-MLA kernels are gated to compute capability 120/121. That default
    is therefore not selectable on an Ampere rank, and reading it once into a
    module global -- as this dispatch used to -- gives every rank of a
    heterogeneous group the first rank's answer (#343).

    An explicitly set value is honoured everywhere: that is a statement about
    the launch, not a probe. It is still downgraded off SM12x when it names
    ``flashinfer``, because that kernel cannot run there at all; the downgrade
    is logged rather than silent.
    """
    requested = envs.SGLANG_SM120_FLASHMLA_BACKEND.get()
    if requested != "flashinfer" or not is_cuda() or _is_sm12x(device_id):
        return requested

    if envs.SGLANG_SM120_FLASHMLA_BACKEND.is_set():
        logger.warning(
            "SGLANG_SM120_FLASHMLA_BACKEND=flashinfer was requested, but "
            "flashinfer's sparse-MLA kernels are restricted to compute "
            "capability 120/121 and device %s is not one. Using the Triton "
            "implementation instead.",
            device_id,
        )
    return "triton"
