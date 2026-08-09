# SPDX-License-Identifier: Apache-2.0
"""
KT Expert Parallelism Wrapper for MoE layers.

This module provides a generic wrapper that enables CPU-GPU expert parallelism
for any MoE quantization method. It coordinates parallel execution of GPU experts
(using any quantization method) and CPU experts (using AMX/AVX instructions).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_compiler_backend

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.server_args import ServerArgs

try:
    from kt_kernel import KTMoEWrapper

    KTRANSFORMERS_AVAILABLE = True
except ImportError:
    KTRANSFORMERS_AVAILABLE = False

import logging
import os

logger = logging.getLogger(__name__)

# kt_kernel's submit_with_cuda_stream / sync_with_cuda_stream order the CPU
# expert work against the GPU stream through cudaLaunchHostFunc, which the
# prebuilt kt_kernel wheel resolves by symbol at runtime. On a ROCm build there
# is no such symbol in the process (torch exports the hip* names), so both calls
# degrade to a no-op: the MoE task is never handed to the worker pool and
# sync_forward hands back an untouched zero buffer. That is silent -- the server
# starts, every kt call "succeeds", and the CPU experts contribute exactly
# nothing to the logits. Measured on gfx1103: absmean 0.0 out, 0.024 ms/layer.
#
# On ROCm we therefore drive the same MOE object through cpu_infer.submit /
# cpu_infer.sync, the stream-free pair kt itself uses for load_weights. Those
# are still asynchronous with respect to each other -- submit hands the task to
# the worker pool and returns -- so CPU experts still overlap the GPU expert
# GEMM. What is lost is stream ORDERING, so the D2H copy of the hidden states
# has to be forced to completion with an explicit stream sync before the CPU
# may read the pinned buffer. One sync per MoE layer.
_KT_IS_ROCM = bool(getattr(torch.version, "hip", None))
_KT_STREAM_MODE = os.environ.get("KT_STREAM_MODE", "auto")
if _KT_STREAM_MODE == "auto":
    _KT_USE_CUDA_STREAM = not _KT_IS_ROCM
else:
    _KT_USE_CUDA_STREAM = _KT_STREAM_MODE == "cuda"


@dataclass
class KTConfig:
    """Configuration for KTransformers heterogeneous computing CPU part.

    Args:
        layer_idx: Layer index in the model
        num_gpu_experts: Number of experts to run on GPU
        cpuinfer_threads: Number of CPU inference threads
        threadpool_count: Number of thread pools for CPU computation
        weight_path: Path to CPU quantized weights
        chunked_prefill_size: Chunk size for prefill computation
        method: CPU computation method (e.g., "int4")
        num_layers: Total number of layers in the model (optional)
    """

    layer_idx: int
    num_gpu_experts: int
    cpuinfer_threads: int
    threadpool_count: int
    weight_path: str
    chunked_prefill_size: int
    max_deferred_experts_per_token: int
    method: str
    num_layers: Optional[int] = None


def create_kt_config_from_server_args(
    server_args: "ServerArgs", layer_idx: int
) -> Optional[KTConfig]:
    """Create KTConfig from ServerArgs if KT is configured.

    Args:
        server_args: Global server arguments
        layer_idx: Layer index in the model

    Returns:
        KTConfig if KT is configured, None otherwise
    """
    if server_args.kt_weight_path is None:
        return None

    # Try to get num_layers from model config
    num_layers = None
    try:
        hf_config = server_args.get_hf_config()
        num_layers = getattr(hf_config, "num_hidden_layers", None)
    except Exception:
        # If we can't get the config, num_layers will be None
        pass

    return KTConfig(
        layer_idx=layer_idx,
        num_gpu_experts=server_args.kt_num_gpu_experts,
        cpuinfer_threads=server_args.kt_cpuinfer,
        threadpool_count=server_args.kt_threadpool_count,
        weight_path=server_args.kt_weight_path,
        chunked_prefill_size=server_args.chunked_prefill_size,
        method=server_args.kt_method,
        max_deferred_experts_per_token=server_args.kt_max_deferred_experts_per_token,
        num_layers=num_layers,
    )


@torch.compile(dynamic=True, backend=get_compiler_backend())
def mask_cpu_expert_ids(topk_ids: torch.Tensor, num_gpu_experts: int) -> torch.Tensor:
    """Mask CPU expert IDs by setting them to -1.

    This function masks expert IDs that should be computed on CPU (IDs >= num_gpu_experts)
    so they won't be computed on GPU. The masked IDs are set to -1, which causes the
    GPU MoE kernel to skip those experts.

    Args:
        topk_ids: Tensor of shape [num_tokens, top_k] containing expert IDs
        num_gpu_experts: Number of experts that should run on GPU (experts 0 to num_gpu_experts-1)

    Returns:
        Modified topk_ids tensor with CPU expert IDs masked as -1
    """
    topk_ids[topk_ids >= num_gpu_experts] = -1
    return topk_ids


class KTEPWrapperMethod(FusedMoEMethodBase):
    """Wrapper for any MoE quantization method to enable CPU-GPU expert parallelism.

    This wrapper coordinates parallel execution of:
    - GPU experts (0 to num_gpu_experts-1) using any quantization method
    - CPU experts (num_gpu_experts to total_experts-1) using AMX/AVX instructions

    The wrapper implements the submit-compute-sync pattern:
    1. Submit CPU expert computation (non-blocking)
    2. Execute GPU expert computation in parallel
    3. Synchronize and merge CPU+GPU results

    Example:
        # Wrap any GPU method with AMX/AVX CPU expert support
        gpu_method = CompressedTensorsWNA16MoE(quant_config, prefix)
        kt_config = KTConfig(layer_idx=0, num_gpu_experts=4, ...)
        method = KTEPWrapperMethod(gpu_method, kt_config)
    """

    def __init__(
        self,
        gpu_method: FusedMoEMethodBase,
        kt_config: KTConfig,
    ):
        """Initialize the KT EP wrapper.

        Args:
            gpu_method: The quantization method to use for GPU experts
            kt_config: Configuration for KT CPU expert computation
        """
        if not KTRANSFORMERS_AVAILABLE:
            raise ImportError(
                "kt_kernel is not installed. To use KTransformers EP wrapper, please install kt_kernel."
            )

        self.gpu_method = gpu_method
        self.kt_config = kt_config

        # num_gpu_experts == 0 means "every expert runs on the CPU", and on the
        # GGUF path that is the ONLY split that works. mask_cpu_expert_ids sets
        # every CPU-bound routed id to -1, and the GGUF decode kernel
        # (ggml_moe_a8_vec) indexes its expert table with those ids without any
        # skip semantics -- a -1 is read as an address, and the load dies with
        # hipErrorIllegalAddress during decode-graph capture. Only the MMQ
        # prefill branch sanitizes. So a partial GPU/CPU split needs a kernel
        # that can skip an expert, which GGUF here does not have.
        #
        # All-CPU is handled by never calling the GPU expert kernel at all,
        # which is also the only variant that actually removes expert GEMM work
        # from the GPU: remapping CPU experts onto a zero-pad expert (the
        # uneven-TP trick) would keep every top_k GEMM on the card and buy
        # nothing but correctness.
        self.kt_all_cpu = (kt_config.num_gpu_experts or 0) == 0
        # ... but the checkpoint loader still needs somewhere to put an expert:
        # FusedMoE.weight_loader gates on quant_method.num_gpu_experts, and a 0
        # there drops every expert tensor on the floor, leaving the GGUF
        # parameters uninitialized and process_weights_after_loading with an
        # empty data_container. Keep one resident expert (~1/256 of the expert
        # bytes) so the GGUF parameter machinery sees a well-formed layer. It is
        # never read: apply() returns the CPU result directly.
        self.num_gpu_experts = 1 if self.kt_all_cpu else kt_config.num_gpu_experts
        if not self.kt_all_cpu and type(gpu_method).__name__ == "GGUFMoEMethod":
            raise ValueError(
                "--kt-num-gpu-experts must be 0 with --quantization gguf: the "
                "GGUF decode kernel ggml_moe_a8_vec has no skip semantics for "
                "the -1 ids this wrapper uses to route an expert to the CPU, "
                f"so a partial split (got {kt_config.num_gpu_experts}) dies "
                "with an illegal memory access on the first decode. Use 0 to "
                "run every expert on the CPU."
            )

        self.override_num_local_experts = True
        self.gpu_method.num_gpu_experts = self.num_gpu_experts
        self.tp_rank = get_parallel().tp_rank

        # KT wrapper will be initialized in create_weights
        self.wrapper: Optional[KTMoEWrapper] = None

        # Store parameters needed for KT initialization
        self._layer_params = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for both GPU and CPU experts.

        Args:
            layer: The MoE layer module
            num_experts: Total number of experts (GPU + CPU)
            hidden_size: Hidden dimension size
            intermediate_size_per_partition: Intermediate size per TP partition
            params_dtype: Data type for parameters
            **extra_weight_attrs: Additional weight attributes
        """
        self.global_num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size_per_partition

        # Get required parameters from layer object
        # top_k: number of experts selected per token
        num_experts_per_tok = layer.top_k

        # intermediate_size_full: full intermediate size before TP partitioning
        intermediate_size_full = (
            layer.intermediate_size_per_partition * layer.moe_tp_size
        )

        layer_max_deferred = self.kt_config.max_deferred_experts_per_token or 0
        if (
            self.kt_config.max_deferred_experts_per_token is not None
            and self.kt_config.num_layers is not None
            and self.kt_config.layer_idx == self.kt_config.num_layers - 1
        ):
            layer_max_deferred = 0

        # 1. Create weights for GPU experts using the wrapped method
        # GPU experts: 0 to num_gpu_experts-1
        self.gpu_method.create_weights(
            layer=layer,
            num_experts=self.num_gpu_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

        # 2. Initialize KT wrapper for CPU experts
        # CPU experts: num_gpu_experts to num_experts-1
        if self.tp_rank == 0:
            # kt_kernel >= 0.6 takes a per-expert boolean mask instead of the
            # scalar num_gpu_experts this call site was written against; the
            # scalar survives only on the SFT path. This wrapper's contract is
            # positional -- experts [0, num_gpu_experts) run on GPU, the rest on
            # CPU (see mask_cpu_expert_ids) -- so the mask is that same split
            # expressed per expert. Older kt_kernel is still accepted.
            if layer_max_deferred > 0 and not _KT_USE_CUDA_STREAM:
                # Deferral relies on sync_with_cuda_stream(stream, allow_pending)
                # to leave the second task outstanding across a layer boundary.
                # cpu_infer.sync() has no allow_pending argument and waits for
                # every submitted task, so on the stream-free path deferral
                # would cost the extra submit and buy nothing while still
                # perturbing the numerics. Refuse rather than pretend.
                raise ValueError(
                    "--kt-max-deferred-experts-per-token is not supported on the "
                    "stream-free (ROCm) kt path; pass 0."
                )

            gpu_experts_mask = torch.zeros(num_experts, dtype=torch.bool)
            if not self.kt_all_cpu:
                gpu_experts_mask[: self.num_gpu_experts] = True

            kt_kwargs = dict(
                layer_idx=self.kt_config.layer_idx,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                hidden_size=hidden_size,
                moe_intermediate_size=intermediate_size_full,
                cpuinfer_threads=self.kt_config.cpuinfer_threads,
                threadpool_count=self.kt_config.threadpool_count,
                weight_path=self.kt_config.weight_path,
                chunked_prefill_size=self.kt_config.chunked_prefill_size,
                method=self.kt_config.method,
                max_deferred_experts_per_token=layer_max_deferred,
            )
            try:
                self.wrapper = KTMoEWrapper(
                    gpu_experts_mask=gpu_experts_mask, **kt_kwargs
                )
            except TypeError:
                self.wrapper = KTMoEWrapper(
                    num_gpu_experts=self.num_gpu_experts, **kt_kwargs
                )
            if self.kt_config.layer_idx == 0:
                logger.info(
                    "[KT] engaged: method=%s experts=%d gpu_experts=%d cpu_experts=%d "
                    "cpuinfer=%d threadpool=%d deferred=%s backend=%s stream_mode=%s "
                    "weight_path=%s",
                    self.kt_config.method,
                    num_experts,
                    0 if self.kt_all_cpu else self.num_gpu_experts,
                    num_experts if self.kt_all_cpu else num_experts - self.num_gpu_experts,
                    self.kt_config.cpuinfer_threads,
                    self.kt_config.threadpool_count,
                    layer_max_deferred,
                    type(self.wrapper).__name__,
                    "cuda_stream" if _KT_USE_CUDA_STREAM else "rocm_stream_free",
                    self.kt_config.weight_path,
                )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process weights after loading from checkpoint.

        Args:
            layer: The MoE layer module
        """
        # 1. Process GPU weights
        if hasattr(self.gpu_method, "process_weights_after_loading"):
            self.gpu_method.process_weights_after_loading(layer)

        # 2. Load CPU weights using KT wrapper
        if self.tp_rank == 0 and self.wrapper is not None:
            torch.cuda.synchronize()

            # Get expert location metadata for CPU expert mapping
            from sglang.srt.eplb.expert_location_dispatch import (
                get_global_expert_location_metadata,
            )

            physical_to_logical_map_cpu = (
                get_global_expert_location_metadata()
                .physical_to_logical_map_cpu[self.kt_config.layer_idx]
                .contiguous()
            )
            self.wrapper.load_weights(physical_to_logical_map_cpu)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        """Create MoE runner for computation.

        Args:
            layer: The MoE layer module
            moe_runner_config: Configuration for MoE runner
        """
        self.moe_runner_config = moe_runner_config
        if self.override_num_local_experts:
            moe_runner_config.num_local_experts = self.num_gpu_experts
        # Delegate to GPU method to create its runner
        self.gpu_method.create_moe_runner(layer, moe_runner_config)

    def submit(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> None:
        """Submit CPU expert computation asynchronously (non-blocking).

        This method submits the CPU expert computation to AMX/AVX without waiting
        for completion, allowing GPU computation to proceed in parallel.

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information
        """
        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        if self.tp_rank != 0 or self.wrapper is None:
            return

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        if not _KT_USE_CUDA_STREAM:
            self._submit_stream_free(x, topk_ids, topk_weights)
            return

        # Submit forward task to CPU (non-blocking)
        self.wrapper.submit_forward(
            x, topk_ids, topk_weights, torch.cuda.current_stream(x.device).cuda_stream
        )

    def _kt_buffers(self, x: torch.Tensor):
        from kt_kernel.experts_base import KExpertsCPUBuffer

        flat = x.view(-1, x.shape[-1])
        buffers = KExpertsCPUBuffer.get_buffer(flat, self.wrapper.num_experts_per_tok)
        slot = self.wrapper.layer_idx % KExpertsCPUBuffer.buffer_depth
        return flat, buffers, slot

    def _submit_stream_free(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        """submit_forward without the cudaLaunchHostFunc bridge (ROCm).

        Stream ordering is replaced by one explicit stream sync: the pinned
        staging buffers are filled by D2H copies on the current stream, and the
        CPU worker pool reads them directly, so the copies must have LANDED
        before the task is submitted. After the submit this returns immediately
        and the GPU expert GEMM runs concurrently with the CPU experts, which is
        the overlap the stream path was there to provide.
        """
        flat, (inp, imm, _defr, wts, out_cpu, bsz, _out_gpu), slot = self._kt_buffers(x)

        inp[slot].copy_(flat, non_blocking=True)
        wts[slot].copy_(topk_weights.to(torch.float32), non_blocking=True)
        imm[slot].copy_(topk_ids.to(torch.long), non_blocking=True)
        bsz[slot].fill_(flat.shape[0])
        torch.cuda.current_stream(x.device).synchronize()

        self.wrapper.cpu_infer.submit(
            self.wrapper.moe.forward_task(
                bsz[slot].data_ptr(),
                imm[slot].size(-1),
                imm[slot].data_ptr(),
                wts[slot].data_ptr(),
                inp[slot].data_ptr(),
                out_cpu[slot].data_ptr(),
                False,
            )
        )

    def _sync_stream_free(self, x: torch.Tensor) -> torch.Tensor:
        flat, (_i, _im, _d, _w, out_cpu, _b, out_gpu), slot = self._kt_buffers(x)
        self.wrapper.cpu_infer.sync()
        out_gpu[slot].copy_(out_cpu[slot], non_blocking=True)
        return out_gpu[slot].view(x.shape)

    def sync(self, x: torch.Tensor) -> torch.Tensor:
        """Synchronize and retrieve CPU expert computation results.

        This method waits for the CPU computation to complete and returns the results.

        Args:
            x: Reference tensor for shape and device information

        Returns:
            CPU expert computation results
        """
        if self.tp_rank != 0 or self.wrapper is None:
            return torch.zeros_like(x)

        if not _KT_USE_CUDA_STREAM:
            return self._sync_stream_free(x)

        # Wait for CPU computation and retrieve results
        return self.wrapper.sync_forward(
            x, torch.cuda.current_stream(x.device).cuda_stream
        )

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        """Execute hybrid CPU+GPU MoE forward pass with parallelism.

        This is the main computation method that coordinates:
        1. Submit CPU expert computation (non-blocking)
        2. Execute GPU expert computation in parallel
        3. Synchronize CPU results and merge with GPU results

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information

        Returns:
            Combined computation results from CPU and GPU experts
        """
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        # Step 1: Submit CPU expert computation (non-blocking)
        if self.tp_rank == 0:
            self.submit(layer, dispatch_output)

        if self.kt_all_cpu:
            # No expert runs on the GPU, so there is no GPU expert kernel to
            # launch and nothing to merge with -- the CPU result IS the layer
            # output. Skipping the kernel is the entire point: it is the expert
            # GEMM work that moves off the card.
            if self.tp_rank == 0:
                return StandardCombineInput(hidden_states=self.sync(x))
            return StandardCombineInput(hidden_states=torch.zeros_like(x))

        # Step 2: Prepare GPU computation by masking CPU expert IDs
        # CPU expert IDs (>= num_gpu_experts) are set to -1 so GPU kernel skips them
        topk_ids = topk_output.topk_ids
        masked_topk_ids = mask_cpu_expert_ids(topk_ids, self.num_gpu_experts)

        # Create modified dispatch output for GPU computation
        masked_topk_output = topk_output._replace(topk_ids=masked_topk_ids)
        masked_dispatch_output = dispatch_output._replace(
            topk_output=masked_topk_output
        )

        # Step 3: Execute GPU expert computation (any quantization method)
        # This runs in parallel with CPU computation
        gpu_combine_input = self.gpu_method.apply(layer, masked_dispatch_output)

        # Step 4: Synchronize CPU results and merge with GPU results
        output = gpu_combine_input.hidden_states
        if self.tp_rank == 0:
            cpu_output = self.sync(x)
            output = output + cpu_output

        return StandardCombineInput(hidden_states=output)

    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped GPU method.

        This allows the wrapper to transparently expose attributes and methods
        from the wrapped GPU quantization method.

        Args:
            name: Attribute name

        Returns:
            Attribute value from gpu_method
        """
        # Avoid infinite recursion for internal attributes
        if name in ("gpu_method", "wrapper", "kt_config"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        return getattr(self.gpu_method, name)
