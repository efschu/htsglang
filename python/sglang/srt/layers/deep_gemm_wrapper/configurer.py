import logging

from sglang.srt.environ import envs
from sglang.srt.utils import (
    get_device_sm,
    is_cuda,
    is_musa,
    is_sm100_supported,
    min_visible_cuda_capability_no_init,
)

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_musa = is_musa()


def _compute_enable_deep_gemm():
    # The FLOOR over every visible card rather than device 0 (#343). DeepGEMM
    # is switched on for the WHOLE process -- ``ENABLE_JIT_DEEPGEMM`` is read
    # as a module constant in a dozen MoE call sites -- so it cannot be
    # re-asked per device, and device 0's answer is wrong in the dangerous
    # direction: a Hopper card 0 next to an sm86 card would arm DeepGEMM for
    # experts the second card cannot run. The floor refuses instead.
    capability = min_visible_cuda_capability_no_init()
    if capability is None:
        sm_version = get_device_sm()
    else:
        sm_version = capability[0] * 10 + capability[1]
    if (_is_cuda and sm_version < 90) or (_is_musa and sm_version < 31):
        return False
    # DeepGEMM requires TMEM/tcgen05 (SM100+datacenter), not available on SM120
    if sm_version == 120:
        return False
    if not (_is_cuda or _is_musa):
        return False

    try:
        import deep_gemm  # noqa: F401
    except ImportError:
        return False

    return envs.SGLANG_ENABLE_JIT_DEEPGEMM.get()


ENABLE_JIT_DEEPGEMM = _compute_enable_deep_gemm()

DEEPGEMM_BLACKWELL = ENABLE_JIT_DEEPGEMM and is_sm100_supported()
DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL
DEEPGEMM_NEED_TMA_ALIGNED_SCALES = not (DEEPGEMM_SCALE_UE8M0 or _is_musa)
