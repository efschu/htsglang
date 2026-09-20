from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

import logging
import os

from sglang.jit_kernel.marlin_switches import (
    arch_override_applies,
    arch_override_cuda_cflags,
    arch_override_from_env,
    format_switch_log,
    switch_census,
)
from sglang.jit_kernel.utils import (
    cache_once_per_arch,
    get_jit_cuda_arch,
    load_jit,
    make_cpp_args,
    override_jit_cuda_arch,
)
from sglang.kernel_api_logging import debug_kernel_api

if TYPE_CHECKING:
    from sgl_kernel.scalar_type import ScalarType
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

# Constants matching device::marlin_moe:: in marlin.cuh
_MAX_THREAD_N = 256

_SWITCHES_LOGGED = {"done": False}


def _log_marlin_switches(device) -> None:
    """Task #49: ONE line per rank naming every Marlin switch this boot ran with.

    Without it a clean A/B arm cannot be told from an arm whose environment
    never reached the worker -- the same null-result trap the #49 workspace and
    reduce-branch logs already close. Emitted at error level on purpose: it must
    survive the serving log level, and it is one line per process."""
    if _SWITCHES_LOGGED["done"]:
        return
    _SWITCHES_LOGGED["done"] = True
    try:
        props = torch.cuda.get_device_properties(device)
        sms = props.multi_processor_count
        smem = getattr(props, "shared_memory_per_block_optin", None)
        cap = (props.major, props.minor)
        arch = get_jit_cuda_arch().target_name
        logger.error(
            "%s",
            format_switch_log(
                arch, switch_census(sms, smem_optin=smem, device_cap=cap)
            ),
        )
    except Exception as exc:  # noqa: BLE001 -- an instrument never kills a GEMM
        logger.debug("[nan-49c] marlin switch log skipped: %s", exc)


@cache_once_per_arch
def _jit_moe_wna16_marlin_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(dtype)
    override = arch_override_from_env()

    # fn8c4 (2026-09-20) died here: the env is scoped per BOOT, not per RANK, so
    # both sm_86 ranks also built the module for compute_90 and hit "no kernel
    # image is available for execution on the device" at the first launch. The
    # decision is a property of THIS process's card, never of the environment
    # alone -- so resolve the real capability and let arch_override_applies say
    # yes or no, with a reason that lands in the per-rank log line.
    dev_arch = get_jit_cuda_arch()
    applies, reason = arch_override_applies(override, dev_arch.major, dev_arch.minor)
    if override is not None and not applies:
        logger.error(
            "[nan-49c] marlin arch override NOT applied on this rank: "
            "requested=%s device=%s reason=%s -- building natively instead",
            override.target_name, dev_arch.target_name, reason,
        )
        override = None

    def _build():
        return load_jit(
            "moe_wna16_marlin",
            *args,
            cuda_files=["gemm/marlin_moe/moe_wna16_marlin.cuh"],
            cuda_wrappers=[
                (
                    "moe_wna16_marlin_gemm",
                    f"moe_wna16_marlin_gemm<{args}>",
                )
            ],
            extra_cuda_cflags=arch_override_cuda_cflags(override),
        )

    if override is None:
        return _build()

    # Task #49: build this ONE module for a different virtual architecture and
    # let the driver JIT the embedded PTX for the real device.
    #
    # `override_jit_cuda_arch` changes three things at once, and all three are
    # required together: the `-DSGL_CUDA_ARCH` macro the kernel headers
    # static_assert `__CUDA_ARCH__` against, the `TVM_FFI_CUDA_ARCH_LIST` from
    # which tvm-ffi derives its `-gencode`, and the `arch` that goes into both
    # the JIT build hash and the build directory NAME. The last one is what
    # makes the cache key disjoint from the native build: an overridden module
    # lands in `..._arch_9.0__...`, the native one in `..._arch_12.0__...`, and
    # `_compat_ok` re-checks the recorded provenance before reusing either, so
    # the two can never be mistaken for one another.
    #
    # Scope: this module only. Every other JIT kernel in the process keeps
    # building for the real device, which is what keeps the A/B pointed at
    # Marlin instead of at the whole runtime.
    with override_jit_cuda_arch(override.major, override.minor, override.suffix):
        module = _build()
    _log_override_artefact(override, args)
    return module


def _log_override_artefact(override, args=None) -> None:
    """Prove the module this rank LOADED actually carries PTX the driver can JIT.

    The coordinator's question after fn8c4 -- 'is a cubin or a fatbin with PTX
    built and loaded?' -- had to be answered by hand with cuobjdump against the
    cache directory. Answered once, it stays answered only if the boot answers
    it itself, so the check is an instrument now. Best effort: a missing
    cuobjdump reports 'unknown', never a false 'yes'."""
    import glob
    import subprocess

    try:
        cache = os.environ.get("TVM_FFI_CACHE_DIR", "~/.cache/tvm-ffi")
        hits = glob.glob(os.path.join(os.path.expanduser(cache), "*arch_%s__*" % override.target_name, "*.so"))
        so = max(hits, key=os.path.getmtime) if hits else None
        ptx = "unknown"
        if so:
            try:
                out = subprocess.run(
                    ["cuobjdump", "-lptx", so], capture_output=True, timeout=30
                ).stdout.decode("utf-8", "replace")
                ptx = [l.split(":")[-1].strip() for l in out.splitlines() if "PTX file" in l] or "NONE"
            except Exception:  # noqa: BLE001 -- cuobjdump may not be on PATH
                ptx = "unknown"
        logger.error(
            "[nan-49c] marlin arch override APPLIED: target=%s so=%s ptx_sections=%s "
            "-- 'NONE' means the fatbin has no PTX and the driver cannot JIT it; "
            "that is the fn8c4 death mode and the arm is invalid",
            override.target_name, so, ptx,
        )
    except Exception as exc:  # noqa: BLE001 -- an instrument never kills a build
        logger.debug("[nan-49c] arch override artefact log skipped: %s", exc)


def _or_empty(
    t: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return t if t is not None else torch.empty(0, device=device, dtype=dtype)


@debug_kernel_api
def moe_wna16_marlin_gemm(
    a: torch.Tensor,
    c_or_none: Optional[torch.Tensor],
    b_q_weight: torch.Tensor,
    b_bias_or_none: Optional[torch.Tensor],
    b_scales: torch.Tensor,
    global_scale_or_none: Optional[torch.Tensor],
    b_zeros_or_none: Optional[torch.Tensor],
    g_idx_or_none: Optional[torch.Tensor],
    perm_or_none: Optional[torch.Tensor],
    workspace: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    moe_block_size: int,
    top_k: int,
    mul_topk_weights: bool,
    is_ep: bool,
    b_q_type: ScalarType,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
) -> torch.Tensor:
    device = a.device

    # Allocate output if not provided
    if c_or_none is not None:
        c = c_or_none
    else:
        c = torch.empty((size_m * top_k, size_n), dtype=a.dtype, device=device)

    # Early return for zero-size M
    if size_m == 0:
        return c

    # Determine activation ordering
    has_act_order = (
        g_idx_or_none is not None
        and perm_or_none is not None
        and g_idx_or_none.numel() > 0
        and perm_or_none.numel() > 0
        and g_idx_or_none.size(-1) > 0
        and perm_or_none.size(-1) > 0
    )

    # Determine has_zp
    has_zp = b_zeros_or_none is not None and b_zeros_or_none.numel() > 0

    # Determine has_bias
    has_bias = b_bias_or_none is not None

    # Derive num_groups and group_size from b_scales
    num_groups = b_scales.size(1)
    if has_act_order:
        if is_k_full:
            group_size = size_k // num_groups
        else:
            group_size = 0
    else:
        if num_groups > 1:
            group_size = size_k // num_groups
        else:
            group_size = -1

    # Allocate a_tmp for act_order column permutation
    if has_act_order:
        a_tmp = torch.empty((size_m * top_k, size_k), dtype=a.dtype, device=device)
    else:
        a_tmp = torch.empty(0, dtype=a.dtype, device=device)

    # Allocate c_tmp for fp32 reduce
    if use_fp32_reduce and not use_atomic_add:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        # max num of threadblocks is sms * 4
        max_c_tmp_size = min(
            size_n * sorted_token_ids.size(0),
            sms * 4 * moe_block_size * _MAX_THREAD_N,
        )
        if moe_block_size == 8:
            max_c_tmp_size *= 2
        c_tmp = torch.empty(max_c_tmp_size, dtype=torch.float32, device=device)
    else:
        c_tmp = torch.empty(0, dtype=torch.float32, device=device)

    # Convert Optional tensors to empty tensors
    g_idx_t = _or_empty(g_idx_or_none, device, torch.int32)
    perm_t = _or_empty(perm_or_none, device, torch.int32)
    b_zeros_t = _or_empty(b_zeros_or_none, device, a.dtype)
    b_bias_t = _or_empty(b_bias_or_none, device, a.dtype)
    global_scale_t = _or_empty(global_scale_or_none, device, a.dtype)

    _log_marlin_switches(device)
    module = _jit_moe_wna16_marlin_module(a.dtype)
    module.moe_wna16_marlin_gemm(
        a,
        c,
        b_q_weight,
        b_bias_t,
        b_scales,
        global_scale_t,
        b_zeros_t,
        g_idx_t,
        perm_t,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        a_tmp,
        c_tmp,
        moe_block_size,
        top_k,
        mul_topk_weights,
        is_ep,
        b_q_type.id,
        size_m,
        size_n,
        size_k,
        has_act_order,
        has_bias,
        is_k_full,
        has_zp,
        num_groups,
        group_size,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    )

    return c
