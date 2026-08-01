"""Which paged-MQA-logits implementation the DSV4 indexer gets, per device.

The DSpark layers of DeepSeek-V4 (any layer with ``compress_ratio in (4, 128)``
-- layers 40-42 of DeepSeek-V4-Flash) run an indexer whose paged-MQA-logits
step has four implementations, with four different hardware domains:

* **DeepGEMM** (``deep_gemm.fp8_paged_mqa_logits``, the default) -- Hopper and
  datacenter Blackwell. It asserts ``Unsupported architecture`` in compiled C++
  elsewhere, and DeepGEMM upstream declined SM12x support (deepgemm PR #318),
  so consumer Blackwell is out as well as Ampere.
* **TileLang** and **AITER** -- opt-in, their own domains (AITER is ROCm).
* **torch** (``fp8_paged_mqa_logits_torch_sm120``) -- a vectorized,
  CUDA-graph-safe pure-torch implementation. It was written for SM120 but
  nothing in it is SM120-specific: it dequantises the fp8 index cache with
  ``Tensor.to``, which is a software conversion available on every
  architecture, and accumulates in float32 with ``torch.bmm``.

Until now the torch implementation was reachable only by setting
``SGLANG_FP8_PAGED_MQA_LOGITS_TORCH``, so a device without DeepGEMM went to
DeepGEMM anyway and died there. This module makes the choice a question about
the device -- "does this kernel exist here" -- with the explicit environment
selections still winning, and every answer cached per device (#343: one
process can hold parts of one model on two cards of different architecture).

Adapted from the shape upstream PR #31480 used for the sibling ``dsa/``
package. The parts that PR had to build -- a torch implementation and its
metadata bypass -- already exist here (#24692); what was missing was the
capability-driven selection, so this module is the dispatch half only.
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

# Compute-capability majors DeepGEMM's paged-MQA-logits kernel supports:
# 9 = Hopper, 10 = datacenter Blackwell.
_DEEPGEMM_MAJORS = (9, 10)

# Backend names this module resolves to.
BACKEND_DEEPGEMM = "deepgemm"
BACKEND_TILELANG = "tilelang"
BACKEND_AITER = "aiter"
BACKEND_TORCH = "torch"

_warned_devices: set = set()


@per_device_gate
def deepgemm_indexer_supported(device_id: int) -> bool:
    """Whether DeepGEMM's paged-MQA-logits kernel runs on this device.

    False on Ampere, Ada and consumer Blackwell. Non-CUDA vendors are left
    alone: they reach this code only through their own opt-in backend.
    """
    if not is_cuda():
        return True
    return get_device_capability_no_init(device_id)[0] in _DEEPGEMM_MAJORS


def resolve_paged_mqa_logits_backend(device_id: Optional[int] = None) -> str:
    """Pick the paged-MQA-logits implementation for this device.

    Explicit environment selections win -- they are statements about the
    launch, not probes (#343's rule). Only the otherwise-default DeepGEMM
    choice is overridden, and only where DeepGEMM cannot run at all.
    """
    if envs.SGLANG_OPT_USE_TILELANG_INDEXER.get():
        return BACKEND_TILELANG
    if envs.SGLANG_OPT_USE_AITER_INDEXER.get():
        return BACKEND_AITER
    if envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get():
        return BACKEND_TORCH
    if not deepgemm_indexer_supported(device_id):
        return BACKEND_TORCH
    return BACKEND_DEEPGEMM


def deepgemm_indexer_metadata_needed(device_id: Optional[int] = None) -> bool:
    """Whether ``PagedIndexerMetadata`` should build the DeepGEMM schedule.

    The schedule is a tiling plan for DeepGEMM's own kernel; every other
    implementation takes it as an argument and ignores it. Building it calls
    into DeepGEMM, which is itself unavailable on the architectures this
    module routes away from, so the metadata has to be skipped in exactly the
    cases where the kernel is not selected -- the two decisions must not be
    able to disagree, or a rank ends up with a DeepGEMM schedule and no
    DeepGEMM, or the reverse.
    """
    if envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get():
        return False
    if envs.SGLANG_OPT_USE_AITER_INDEXER.get():
        return False
    return deepgemm_indexer_supported(device_id)


def warn_torch_indexer_substitution_once(device_id: Optional[int] = None) -> None:
    """Name the substitution the first time a device makes it.

    This is not a drop-in equivalence and must not be silent. DeepGEMM
    computes the indexer logits as an FP8 tensor-core GEMM; the torch
    implementation dequantises to float32 first and accumulates with
    ``torch.bmm``. The mathematical quantity is the same, the arithmetic is
    not: the two disagree in the low bits, and because these logits are only
    ever consumed by a top-k selection, a near-tie between two KV positions
    can be broken differently. Output can therefore differ from a Hopper run
    of the same checkpoint -- not by an approximation of the attention itself,
    but by which tokens the indexer selects at the margin.
    """
    key = device_id if device_id is not None else -1
    if key in _warned_devices:
        return
    _warned_devices.add(key)
    logger.warning(
        "DSV4 indexer on device %s: DeepGEMM's paged-MQA-logits kernel does "
        "not exist on this architecture (compute capability %s); falling back "
        "to the float32 torch implementation. This is NOT bit-identical to a "
        "Hopper run: the logits are accumulated in float32 instead of by an "
        "FP8 tensor-core GEMM, and the top-k selection they feed can break a "
        "near-tie between KV positions differently. Expect correct output, "
        "not identical output.",
        device_id,
        capability_for_log(device_id),
    )


def reset_substitution_warnings() -> None:
    """For tests. Nothing in the serving path calls it."""
    _warned_devices.clear()


def capability_for_log(device_id: Optional[int]) -> str:
    try:
        return "%d.%d" % get_device_capability_no_init(device_id)
    except Exception:  # pragma: no cover - logging must never be the failure
        return "unknown"
