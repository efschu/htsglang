from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.utils.common import (
    get_device_capability,
    is_cuda,
    is_sm100_supported,
    is_sm120_supported,
)
from sglang.srt.utils.custom_op import register_custom_op_from_extern

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


fp4_quantize = None
try:
    from flashinfer import fp4_quantize as _flashinfer_fp4_quantize

    _flashinfer_fp4_quantize_backend = "cute-dsl" if is_sm100_supported() else "cuda"

    def _round_up(x: int, y: int) -> int:
        return ((x + y - 1) // y) * y

    def _flashinfer_fp4_quantize_impl(
        input: torch.Tensor,
        global_scale: Optional[torch.Tensor] = None,
        sf_vec_size: int = 16,
        sf_use_ue8m0: bool = False,
        is_sf_swizzled_layout: bool = True,
        is_sf_8x4_layout: bool = False,
        enable_pdl: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _flashinfer_fp4_quantize(
            input=input,
            global_scale=global_scale,
            sf_vec_size=sf_vec_size,
            sf_use_ue8m0=sf_use_ue8m0,
            is_sf_swizzled_layout=is_sf_swizzled_layout,
            is_sf_8x4_layout=is_sf_8x4_layout,
            enable_pdl=enable_pdl,
            backend=_flashinfer_fp4_quantize_backend,
        )

    def _flashinfer_fp4_quantize_fake(
        input: torch.Tensor,
        global_scale: Optional[torch.Tensor] = None,
        sf_vec_size: int = 16,
        sf_use_ue8m0: bool = False,
        is_sf_swizzled_layout: bool = True,
        is_sf_8x4_layout: bool = False,
        enable_pdl: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        is_column_major = input.stride(-2) == 1
        if is_column_major:
            m = input.shape[-1]
            K = input.shape[-2]
        else:
            m = input.numel() // input.shape[-1]
            K = input.shape[-1]
        if is_column_major:
            x_q = input.new_empty((*input.shape[:-2], K // 2, m), dtype=torch.uint8)
        else:
            x_q = input.new_empty((*input.shape[:-1], K // 2), dtype=torch.uint8)
        if is_sf_swizzled_layout:
            row_size = 8 if is_sf_8x4_layout else 128
            sf_rows = _round_up(m, row_size)
            sf_cols = _round_up(K // sf_vec_size, 4)
        else:
            sf_rows = m
            sf_cols = K // sf_vec_size
        if is_column_major:
            sf = input.new_empty((sf_cols, sf_rows), dtype=torch.uint8)
        else:
            sf = input.new_empty((sf_rows, sf_cols), dtype=torch.uint8)
        return x_q, sf

    fp4_quantize = register_custom_op_from_extern(
        _flashinfer_fp4_quantize_impl,
        op_name="flashinfer_fp4_quantize",
        fake_impl=_flashinfer_fp4_quantize_fake,
    )
except ImportError:
    fp4_quantize = None


class Fp4GemmRunnerBackend(Enum):
    """Enum for FP4 GEMM runner backend selection."""

    AUTO = "auto"
    CUTLASS = "cutlass"
    FLASHINFER_CUDNN = "flashinfer_cudnn"
    FLASHINFER_CUTEDSL = "flashinfer_cutedsl"
    FLASHINFER_CUTLASS = "flashinfer_cutlass"
    FLASHINFER_TRTLLM = "flashinfer_trtllm"
    MARLIN = "marlin"

    def is_auto(self) -> bool:
        return self == Fp4GemmRunnerBackend.AUTO

    def is_cutlass(self) -> bool:
        return self == Fp4GemmRunnerBackend.CUTLASS

    def is_flashinfer_cudnn(self) -> bool:
        return self == Fp4GemmRunnerBackend.FLASHINFER_CUDNN

    def is_flashinfer_cutlass(self) -> bool:
        return self == Fp4GemmRunnerBackend.FLASHINFER_CUTLASS

    def is_flashinfer_trtllm(self) -> bool:
        return self == Fp4GemmRunnerBackend.FLASHINFER_TRTLLM

    def is_flashinfer_cutedsl(self) -> bool:
        return self == Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL

    def is_marlin(self) -> bool:
        return self == Fp4GemmRunnerBackend.MARLIN

    def is_flashinfer(self) -> bool:
        return self.value.startswith("flashinfer_")

    def get_flashinfer_backend(self) -> str:
        """Get the backend string to pass to FlashInfer's mm_fp4 API.

        This remaps SGLang's user-facing backend names to FlashInfer's API names.
        Examples:
            'flashinfer_trtllm' -> 'trtllm'
            'flashinfer_cutlass' -> 'cutlass'
            'flashinfer_cudnn' -> 'cudnn'
            'flashinfer_cutedsl' -> 'cute-dsl'
        """
        if self == Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL:
            return "cute-dsl"
        if self.value.startswith("flashinfer_"):
            return self.value.removeprefix("flashinfer_")
        else:
            return self.value


FP4_GEMM_RUNNER_BACKEND: Fp4GemmRunnerBackend | None = None


def has_fork_nvfp4_cutlass_kernel() -> bool:
    """Is the fork's own NVFP4 CUTLASS GEMM reachable on THIS device?

    Two independent conditions, both required (#323c):

    1. The entry point imports. ``sglang.jit_kernel.nvfp4`` is a JIT tree, so
       importing it only proves the Python wrapper and its sources exist -- it
       compiles nothing.
    2. The device is one the JIT tree has an architecture image for. The
       kernels are built per architecture with an ``sm_*a`` target
       (``_nvfp4_arch_env`` refuses ``major < 10``) and the dispatch in
       ``nvfp4_scaled_mm_kernels.cuh`` has exactly two arms: ``>= 120`` and
       Blackwell datacenter below it. Claiming the kernel on an architecture
       outside that set is the #304 failure -- a module built for a foreign
       architecture is loaded from cache and dies at launch with "no kernel
       image is available for execution on the device". So the architecture is
       checked HERE, before the backend is chosen, not at first forward.

    Answered functionally, per device, never from a hardcoded arch table
    alone: the capability integer is only meaningful together with the vendor
    (``is_sm100_supported`` / ``is_sm120_supported`` are the NVIDIA-namespaced
    predicates and answer False off NVIDIA).
    """
    if not (is_sm100_supported() or is_sm120_supported()):
        return False
    try:
        from sglang.jit_kernel.nvfp4 import cutlass_scaled_fp4_mm  # noqa: F401
    except Exception:
        return False
    return True


def initialize_fp4_gemm_config(server_args: ServerArgs) -> None:
    """Initialize FP4 GEMM configuration from server args.

    Called once per scheduler process, i.e. once per rank, so a mixed-arch rig
    resolves a different backend per rank without any extra code.
    """
    global FP4_GEMM_RUNNER_BACKEND

    backend = server_args.fp4_gemm_runner_backend
    if backend == "auto":
        if is_sm100_supported():
            backend = "flashinfer_cutedsl"
        elif is_sm120_supported():
            # sm_120 consumer Blackwell. The fork ships its own block-scaled
            # NVFP4 CUTLASS GEMM with an sm_120a arm
            # (nvfp4_scaled_mm_sm120.cuh), but the previous dispatch could
            # never select it: `is_sm100_supported()` is False here (major 12
            # is not 10) and `(10, 0) > (12, 0)` is False too, so `auto` fell
            # through to `flashinfer_cutlass` -- an external dependency. The
            # fork's kernel was then only ever *defaulted into* by
            # `fp4_gemm`'s "flashinfer missing" branch, never chosen. Choose it
            # explicitly when it is genuinely available for this device, and
            # say so in the log when it is not.
            if has_fork_nvfp4_cutlass_kernel():
                backend = "cutlass"
            else:
                backend = "flashinfer_cutlass"
                logger.warning(
                    "NVFP4 auto backend on sm_120: the fork's sm_120a CUTLASS "
                    "GEMM (sglang.jit_kernel.nvfp4.cutlass_scaled_fp4_mm) is "
                    "not importable, falling back to flashinfer_cutlass. Pass "
                    "--fp4-gemm-backend cutlass to make this a hard error "
                    "instead of a fallback."
                )
        elif is_cuda() and (10, 0) > get_device_capability() >= (8, 0):
            # sm_80-sm_89: no native FP4 tensor path. Marlin dequantises E2M1
            # in-kernel and multiplies on bf16 tensor cores -- W4A16, and on
            # these cards the best quantised lane available, not a degraded
            # fallback.
            backend = "marlin"
        else:
            backend = "flashinfer_cutlass"

    FP4_GEMM_RUNNER_BACKEND = Fp4GemmRunnerBackend(backend)


def get_fp4_gemm_runner_backend() -> Fp4GemmRunnerBackend:
    """Get the current FP4 GEMM runner backend."""
    global FP4_GEMM_RUNNER_BACKEND
    if FP4_GEMM_RUNNER_BACKEND is None:
        FP4_GEMM_RUNNER_BACKEND = Fp4GemmRunnerBackend.AUTO
    return FP4_GEMM_RUNNER_BACKEND
