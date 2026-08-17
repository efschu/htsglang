# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ModelRunner runs the forward passes of the models."""

from __future__ import annotations

import contextlib
import datetime
import gc
import inspect
import logging
import os
import socket
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from sglang.srt.configs import (
    BailingHybridConfig,
    FalconH1Config,
    GraniteMoeHybridConfig,
    InternS2PreviewConfig,
    JetNemotronConfig,
    JetVLMConfig,
    KimiLinearConfig,
    Lfm2Config,
    Lfm2MoeConfig,
    Lfm2VlConfig,
    NemotronH_Nano_VL_V2_Config,
    NemotronHConfig,
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3NextConfig,
    ZayaConfig,
)
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.linear_attn_model_registry import get_linear_attn_config
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.configs.model_config import (
    AttentionArch,
    ModelConfig,
    ModelImpl,
    dsa_layer_skips_topk,
    get_num_indexer_layers,
    is_deepseek_dsa,
)
from sglang.srt.configs.update_config import adjust_config_with_unaligned_cpu_tp
from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.debug_utils.dumper import dumper
from sglang.srt.debug_utils.tensor_dump_forward_hook import (
    register_forward_hook_for_model,
)
from sglang.srt.distributed import (
    get_default_distributed_backend,
    get_pp_group,
    get_tp_group,
    get_world_group,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
    set_mscclpp_all_reduce,
    set_torch_symm_mem_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.distributed.parallel_state import monkey_patch_vllm_parallel_state
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.elastic_ep.elastic_ep import (
    ElasticEPStateManager,
    join_process_groups,
    try_recover_ranks,
)
from sglang.srt.elastic_ep.expert_backup_client import ExpertBackupClient
from sglang.srt.environ import envs
from sglang.srt.eplb.eplb_manager import EPLBManager
from sglang.srt.eplb.expert_distribution import (
    ExpertDistributionMetrics,
    ExpertDistributionRecorder,
    get_global_expert_distribution_recorder,
    set_global_expert_distribution_recorder,
)
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    broadcast_global_expert_location_metadata,
    compute_initial_expert_location_metadata,
    format_expert_location_layout,
    get_global_expert_location_metadata,
    set_global_expert_location_metadata,
)
from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
from sglang.srt.eplb.lplb_solver import (
    LPLBSolver,
    assert_lplb_supported_model,
    clear_global_lplb_solvers,
    set_global_lplb_solver,
)
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.hardware_backend.xpu.graph_runner.xpu_graph_runner import XPUGraphRunner
from sglang.srt.kv_canary.api import install_canary
from sglang.srt.kv_canary.runner.canary_manager import context_tuple
from sglang.srt.kv_canary.token_oracle.install import install_token_oracle_from_env
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.attention.attention_registry import (
    ATTENTION_BACKENDS,
    attn_backend_wrapper,
)
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.layers.cp.utils import (
    get_cp_strategy,
)
from sglang.srt.layers.dp_attention import (
    initialize_dp_attention,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.hash_topk import HashTopK
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
from sglang.srt.layers.sampler import create_sampler
from sglang.srt.layers.torchao_utils import apply_torchao_config_to_model
from sglang.srt.layers.utils.cp_utils import is_mla_prefill_cp_enabled
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.schedule_batch import sanity_check_mm_pad_shift_value
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
    cuda_graph_fully_disabled,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    has_forward_context,
)
from sglang.srt.model_executor.graph_shared_output import GraphSharedOutput
from sglang.srt.model_executor.hook_manager import register_forward_hooks
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.srt.model_executor.ngram_token_table import (
    update_ngram_token_table_after_sampling,
)
from sglang.srt.model_executor.offload_register import (
    configure_global_register_from_server_args,
)
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.model_executor.runner import (
    EagerRunner,
    PrefillCudaGraphRunner,
    get_batch_sizes_to_capture,
)
from sglang.srt.model_loader.loader import DefaultModelLoader, get_model_loader
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    RemoteInstanceWeightLoaderBackend,
    register_memory_region,
    trigger_init_weights_send_group_for_remote_instance_request,
)
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.observability.startup_func_log_and_timer import time_startup_latency
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_flags, get_parallel, get_server_args
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import (  # noqa: F401  (re-export)
    CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS,
    ServerArgs,
    add_chunked_prefix_cache_attention_backend,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.state_capturer.base import TopkCaptureOutput
from sglang.srt.state_capturer.indexer_topk import (
    create_indexer_capturer,
    get_global_indexer_capturer,
    set_global_indexer_capturer,
)
from sglang.srt.state_capturer.routed_experts import (
    RoutedExpertsCapturer,
    disable_routed_experts_capture_for_draft,
    get_global_experts_capturer,
    set_global_experts_capturer,
)
from sglang.srt.utils import (
    MultiprocessingSerializer,
    broadcast_pyobj,
    cpu_has_amx_support,
    dynamic_import,
    enable_show_time_cost,
    get_available_gpu_memory,
    get_bool_env_var,
    get_cpu_ids_by_node,
    get_device_capability,
    get_hip_arch,
    init_custom_process_group,
    is_hip,
    is_host_cpu_arm64,
    is_npu,
    log_info_on_rank0,
    monkey_patch_p2p_access_check,
    require_gathered_buffer,
    reserve_rope_cache_for_long_sequences,
    set_cuda_arch,
    slow_rank_detector,
)
from sglang.srt.utils.device_timer import DeviceTimer
from sglang.srt.utils.network import NetworkAddress, get_local_ip_auto
from sglang.srt.utils.nvtx_pytorch_hooks import PytHooks
from sglang.srt.utils.nvtx_utils import profile_range
from sglang.srt.utils.offloader import (
    create_offloader_from_server_args,
    get_offloader,
    set_offloader,
)
from sglang.srt.utils.patch_torch import (
    monkey_patch_torch_reductions,
    register_sgl_tp_rank,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils.weight_checker import WeightChecker
from sglang.srt.weight_sync.tensor_bucket import (
    FlattenedTensorBucket,
    FlattenedTensorMetadata,
)

_is_hip = is_hip()
_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu_arm64 = is_host_cpu_arm64()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _is_npu:
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend

    init_npu_backend()
elif current_platform.is_out_of_tree():
    current_platform.init_backend()

MLA_ATTENTION_BACKENDS = [
    "aiter",
    "flashinfer",
    "fa3",
    "fa4",
    "triton",
    "flashmla",
    "cutedsl_mla",
    "cutlass_mla",
    "trtllm_mla",
    "tokenspeed_mla",
    "ascend",
    "dsa",
    "nsa",  # Deprecated alias for "dsa"
    "intel_xpu",
]


TORCH_DTYPE_TO_KV_CACHE_STR = {
    torch.float8_e4m3fn: "fp8_e4m3",
    torch.float8_e4m3fnuz: "fp8_e4m3",
    torch.float8_e5m2: "fp8_e5m2",
    torch.bfloat16: "bf16",
}


def add_mla_attention_backend(backend_name):
    if backend_name not in MLA_ATTENTION_BACKENDS:
        MLA_ATTENTION_BACKENDS.append(backend_name)
        logger.info(f"Added {backend_name} to MLA_ATTENTION_BACKENDS.")


# Detect stragger ranks in model loading
UNBALANCED_MODEL_LOADING_TIMEOUT_S = 480  # leave more time for post data processing


logger = logging.getLogger(__name__)

_UNSET: Any = object()


# GCN5/Vega-class AMD parts have no bfloat16 units. ROCm still reports
# is_bf16_supported() == True for them because the DTYPE works (by emulation),
# so the arch family is the only honest signal. Measured on gfx900: bf16 matmul
# 2.885 ms vs fp16 1.785 ms, i.e. 62% slower. Only families measured or
# documented to lack bf16 are listed -- unknown/newer archs are deliberately
# absent so a working card cannot regress on a guess.
_ROCM_ARCHS_WITHOUT_BF16 = frozenset(
    {"gfx900", "gfx902", "gfx904", "gfx906", "gfx909", "gfx90c"}
)


def _needs_float16_fallback(device_id: Optional[int] = None) -> bool:
    """Does this device lack bfloat16, so the model must fall back to fp16?

    ``device_id`` is the card this rank was placed on. ``None`` means the
    current device, which is 0 only while nothing has been placed elsewhere
    -- the previous ``get_device_properties(0)`` asked about card 0 no matter
    where the rank actually ran (#343).

    Asked vendor-first, because the old test -- ``get_device_capability()[0] <
    8`` -- reads a number whose namespace depends on the torch build, while the
    threshold 8 is an NVIDIA compute capability. The two COLLIDE: gfx900
    reports ``(9, 0)``, the same integer as Hopper, so ``9 < 8`` is False and
    the fallback never fires on a Vega 64 -- a card with no bfloat16 at all.
    That is the DANGEROUS direction: not a loud refusal but a silently wrong
    dtype, on the load path of every rank. (It is why the second host has to
    pass ``--dtype float16`` by hand.)

    Outside the NVIDIA namespace the question is asked FUNCTIONALLY instead --
    does torch report bf16 support on this device? -- which keeps MI300-class
    parts on bf16 and puts gfx900 on fp16 without either being a magic number.

    On CUDA this is exactly the previous comparison, so NVIDIA behaviour does
    not move; that is the regression criterion.
    """
    if torch.version.hip is not None:
        # NOT torch.cuda.is_bf16_supported(): MEASURED on a Radeon RX Vega 64
        # (gfx900) it returns True, while a bf16 matmul is 2.885 ms against
        # 1.785 ms for fp16 -- i.e. ROCm reports the DTYPE as usable (it is,
        # by emulation) and says nothing about hardware acceleration. Trusting
        # it kept bf16 on a card with no bf16 units, which is the exact bug
        # this function exists to prevent, and it is why the second host had to
        # pass --dtype float16 by hand.
        #
        # So the question is asked in the AMD namespace instead, by gfx family.
        # Only families MEASURED to lack bf16 are listed; anything unknown or
        # newer is left alone, so no working card can regress on a guess.
        arch = get_hip_arch(device_id)
        if arch is None:
            logger.warning(
                "Could not read gcnArchName; leaving the model dtype "
                "unchanged. If this card has no bf16, pass --dtype float16."
            )
            return False
        if arch in _ROCM_ARCHS_WITHOUT_BF16:
            logger.info(
                "%s has no bfloat16 hardware (torch reports is_bf16_supported()"
                " = True, but that is emulation); selecting float16.",
                arch,
            )
            return True
        return False
    return get_device_capability(device_id)[0] < 8


def resolve_language_model(model: nn.Module) -> nn.Module:
    model_cls_name = model.__class__.__name__
    if model_cls_name == "Qwen3OmniMoeForConditionalGeneration":
        return model.thinker.model
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "language_model"):
        return model.language_model
    return model.model


def compute_draft_solo_role(server_args, is_draft_worker, tp_rank):
    """Draft-solo placement (--speculative-draft-placement solo) role of one
    ModelRunner: returns ``(is_draft_solo_host, is_draft_solo_shadow)``.

    * host: the DRAFT runner on the solo rank -- loads the draft UNSHARDED
      (built under a weight-TP=1 override, like the weightless-KV head).
    * shadow: the DRAFT runner on every other rank -- builds the draft on the
      ``meta`` device (no weights, no draft KV, no draft graphs) and never
      runs a draft forward; it waits on the solo rank's token-id broadcast.

    TARGET runners are never host nor shadow: the target path is unchanged.
    Split placement (the default) returns (False, False) everywhere.
    """
    if not is_draft_worker:
        return False, False
    if getattr(server_args, "speculative_draft_placement", "split") != "solo":
        return False, False
    solo_rank = server_args.speculative_draft_solo_rank()
    return tp_rank == solo_rank, tp_rank != solo_rank


class RankZeroFilter(logging.Filter):
    """Filter that only allows INFO level logs from rank 0, but allows all other levels from any rank."""

    def __init__(self, is_rank_zero):
        super().__init__()
        self.is_rank_zero = is_rank_zero

    def filter(self, record):
        if record.levelno == logging.INFO:
            return self.is_rank_zero
        return True


@dataclass
class ModelRunnerOutput:
    logits_output: Union[LogitsProcessorOutput, PPProxyTensors]
    can_run_graph: bool
    expert_distribution_metrics: Optional[ExpertDistributionMetrics] = None
    routed_experts_output: Optional[TopkCaptureOutput] = None
    indexer_topk_output: Optional[TopkCaptureOutput] = None


class ModelRunner(ModelRunnerKVCacheMixin):
    """ModelRunner runs the forward passes of the models."""

    #: #605 flight recorder: whether the first-extend mark has been written.
    #: A CLASS attribute rather than an ``__init__`` assignment plus a
    #: ``getattr`` default at the read site, so a runner built through a path
    #: that skips ``initialize()`` still reads False instead of raising -- and
    #: so the default lives in one place a reader can find.
    _flight_first_forward_marked = False

    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        moe_ep_rank: int,
        moe_ep_size: int,
        pp_rank: int,
        pp_size: int,
        nccl_port: int,
        server_args: ServerArgs,
        dp_rank: Optional[int] = None,
        attn_cp_rank: Optional[int] = None,
        moe_dp_rank: Optional[int] = None,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
        memory_pool_config: Optional[MemoryPoolConfig] = None,
        draft_model_idx: Optional[int] = None,
        is_dual_group_lane: bool = False,
        is_dual_group_lane_draft: bool = False,
        dual_group_host_runner: Optional["ModelRunner"] = None,
        dual_group_plan: Optional[Any] = None,
        dual_group_lane_id: int = 0,
        dual_group_lane_target_model: Optional[Any] = None,
        is_phase_flip_tp_stack: bool = False,
    ):
        # Multi-group runtime (#274): this runner is an in-process LANE over
        # the host runner's weight bytes (dual_group_lane.py). Rank-local by
        # contract: a lane runner exists in ONE serving rank's process only,
        # so nothing on its init path may enter a group collective. It is
        # constructed with is_draft_worker=True so every existing
        # "secondary runner in this process" gate (distributed init,
        # process-global installs) applies; lane-specific deviations are
        # gated on is_dual_group_lane.
        self.is_dual_group_lane = is_dual_group_lane
        # The lane's NEXTN head runner (#274 slice C): same lane, same plan,
        # different model -- it shares the lane target's vocab shells.
        self.is_dual_group_lane_draft = is_dual_group_lane_draft
        self.dual_group_lane_target_model = dual_group_lane_target_model
        self.dual_group_host_runner = dual_group_host_runner
        self.dual_group_plan = dual_group_plan
        self.dual_group_lane_id = dual_group_lane_id
        self.dual_group_lane_weight_added_mib = 0
        self.dual_group_part_models: List[Any] = []
        if is_dual_group_lane and not is_draft_worker:
            raise ValueError(
                "A dual-group lane runner must be constructed with "
                "is_draft_worker=True (the existing secondary-runner gates)."
            )
        # #631 Route A: this runner is the SECONDARY (TP decode) stack of the
        # phase flip -- the full target model rebuilt under the flip group
        # set's geometry, beside the primary PP stack. Like the lane, it
        # rides the is_draft_worker secondary-runner gates (no distributed
        # re-init, no process-global installs); flip-specific deviations are
        # gated on is_phase_flip_tp_stack.
        self.is_phase_flip_tp_stack = is_phase_flip_tp_stack
        if is_phase_flip_tp_stack and not is_draft_worker:
            raise ValueError(
                "A phase-flip TP-stack runner must be constructed with "
                "is_draft_worker=True (the existing secondary-runner gates)."
            )
        # POOL GEOMETRY effective draft flag: the flip's TP stack rides the
        # draft construction gates (distributed init, process globals) but
        # its POOLS take the identical target-model treatment -- full
        # full-attention layer set, replicated-KV uneven-DCP geometry, its
        # own TP-shaped mamba pool. The pool-shape decision sites consult
        # THIS flag, never is_draft_worker directly (a draft-shaped [0]-
        # layer, head-sharded pool here would be the #345 corruption class,
        # caught only by the pin-3 boot assert).
        self.is_draft_pool_worker = is_draft_worker and not is_phase_flip_tp_stack
        # The same distinction, stated once for the sites that mean
        # draft-NESS rather than pool geometry. ``is_draft_worker`` is a
        # CONSTRUCTION gate with three producers -- a speculative draft
        # worker, the #274 dual-group lane, and the #631 phase-flip TP
        # stack -- and only the first actually holds draft weights. A site
        # that asks the construction gate when it means "is this a draft
        # model" gets the wrong answer for the other two producers; that
        # has now cost five separate boot failures (#631 window 3).
        self.is_draft_model_runner = is_draft_worker and not is_phase_flip_tp_stack

        # Parse args
        self.mem_fraction_static = mem_fraction_static
        # Set on target by `_resolve_memory_pool_config`; passed in for draft
        # workers so they reuse target's resolved sizes (replaces legacy
        # `server_args._draft_pool_config` mutation hack).
        self.memory_pool_config = memory_pool_config
        self.device = server_args.device
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        # Weightless-KV fast lane (Variant C Stage 1, Option-B). Derived from
        # server_args (source of truth, independent of the module-global install
        # ordering). The head rank holds the full weights and runs the model
        # TP=1 + the attention dispatch; a weightless worker holds ONLY a KV
        # token-shard, runs the stripped attention-only forward, and
        # materializes no layer weights. These flags are COMPUTED here (B0) and
        # consumed by the worker forward (B1) and the weightless loader (B2);
        # they are no-ops on the default path (fast lane off).
        #
        # NEVER set on a DRAFT runner (#143). The flags name a role in the
        # TARGET forward: "holds the full model" vs "holds only a KV token
        # shard and mirrors the attention dispatch". A draft runner has neither
        # role -- its pool is not token-sharded and its attention backend
        # already excludes itself from the lane
        # (`flashinfer_backend.py`: `weightless_kv = ... and not
        # is_draft_worker`). Leaving them set on a draft runner crosses the two:
        # `_forward_raw` would route a draft forward into the stripped
        # weightless dispatch whose first line asserts a `weightless_kv`
        # backend. The draft's own asymmetry is expressed by the draft-solo
        # roles computed just below, and every load-path site that reads the
        # weightless flags for a draft runner already has an
        # `is_draft_solo_host` / `is_draft_solo_shadow` sibling.
        _wl_fastlane = getattr(server_args, "weightless_kv_fastlane", False)
        _wl_head = getattr(server_args, "weightless_kv_head_rank", 0)
        _wl_target = _wl_fastlane and not is_draft_worker
        self.is_weightless_head = _wl_target and self.tp_rank == _wl_head
        self.is_weightless_worker = _wl_target and self.tp_rank != _wl_head
        if self.is_weightless_worker:
            logger.info(
                "Weightless-KV fast lane: rank %d is a WEIGHTLESS KV worker "
                "(KV-token-shard only; no layer weights).",
                self.tp_rank,
            )
        elif self.is_weightless_head:
            logger.info(
                "Weightless-KV fast lane: rank %d is the HEAD rank (full "
                "weights, TP=1 + attention dispatch).",
                self.tp_rank,
            )
        # Draft-solo placement (--speculative-draft-placement solo): computed
        # here for the same reason as the weightless flags above (single init
        # path, consumed by load_model). No-ops on the default 'split' path
        # and for every TARGET runner.
        self.is_draft_solo_host, self.is_draft_solo_shadow = compute_draft_solo_role(
            server_args, is_draft_worker, tp_rank
        )
        if self.is_draft_solo_host:
            logger.info(
                "Draft-solo placement: rank %d HOSTS the unsharded solo "
                "draft (weight-TP=1 build).",
                self.tp_rank,
            )
        elif self.is_draft_solo_shadow:
            logger.info(
                "Draft-solo placement: rank %d is a draft SHADOW (meta-device "
                "draft, no draft weights/KV/graphs; waits on the solo rank's "
                "draft-token broadcast).",
                self.tp_rank,
            )
        self.dcp_size = server_args.dcp_size
        self.dcp_rank = self.tp_rank % self.dcp_size
        self.moe_ep_rank = moe_ep_rank
        self.moe_ep_size = moe_ep_size
        self.dp_rank = dp_rank
        self.dp_size = server_args.dp_size if server_args.enable_dp_attention else 1
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.attn_cp_rank = attn_cp_rank
        self.attn_cp_size = server_args.attn_cp_size
        self.moe_dp_rank = moe_dp_rank
        self.moe_dp_size = server_args.moe_dp_size
        self.model_config = model_config
        self.dist_port = nccl_port
        self.server_args = server_args
        # #656 T1: how far past the batch's deepest sequence length the lazy
        # RoPE fill has to reach. Same terms as
        # reserve_rope_cache_for_long_sequences, so the two agree on what a
        # speculative step can ask for.
        self._rope_lazy_batch_margin = int(
            getattr(server_args, "speculative_num_steps", 0) or 0
        ) * int(getattr(server_args, "speculative_num_draft_tokens", 0) or 0) * int(
            envs.SGLANG_SPEC_EXPANSION_SAFETY_FACTOR.get()
        ) + int(
            envs.SGLANG_ROPE_CACHE_SAFETY_MARGIN.get()
        )
        # #421 F2: build the #286 offload register from the operator's
        # --lane-offload-* flags. Must run BEFORE the first adapter read: the
        # pools, input buffers and lane workspaces created further down all go
        # through get_global_register(), whose fallback would otherwise build a
        # bare latency-profile register and silently discard the flags. No-op
        # (and no side effect) unless SGLANG_OFFLOAD_REGISTER=1; once per
        # process, so a draft runner or a #274 lane runner does not rebuild the
        # register out from under the items the first runner booked.
        configure_global_register_from_server_args(server_args)
        self.is_draft_worker = is_draft_worker
        self.is_generation = model_config.is_generation
        self.device_timer = None
        # Per-rank prefill timer (metrics_reporter installs it on the TARGET
        # runner only). Separate from device_timer: it must pair 1:1 with
        # plain prefill forwards (see ForwardMode.is_plain_prefill), while
        # device_timer is shared with the draft runners and covers every
        # forward category.
        self.prefill_rank_timer: Optional[DeviceTimer] = None
        self.is_multimodal = model_config.is_multimodal
        self.is_multimodal_chunked_prefill_supported = (
            model_config.is_multimodal_chunked_prefill_supported
        )
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.capture_tail_hooks = []
        self.page_size = server_args.page_size
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.is_hybrid_swa = model_config.is_hybrid_swa
        self.is_hybrid_swa_compress = getattr(
            model_config, "is_hybrid_swa_compress", False
        )
        self.use_mla_backend = self.model_config.attention_arch == AttentionArch.MLA
        self.attention_chunk_size = model_config.attention_chunk_size
        rope_scaling = getattr(
            model_config.hf_text_config, "rope_parameters", None
        ) or getattr(model_config.hf_text_config, "rope_scaling", {})
        self.model_is_mrope = (
            rope_scaling is not None and "mrope_section" in rope_scaling
        )
        self.enable_elastic_ep = server_args.elastic_ep_backend is not None
        self.forward_pass_id = 0
        self.init_new_workspace = False
        self.draft_model_idx = draft_model_idx
        self.enable_hisparse = server_args.enable_hisparse

        self.remote_instance_transfer_engine = None
        self.remote_instance_transfer_engine_session_id = ""
        self.remote_instance_transfer_engine_weight_info = None

        self.msprobe_debugger = None
        if server_args.msprobe_dump_config is not None:
            self.init_msprobe()

        # auxiliary hidden capture mode. TODO: expose this to server args?
        self.eagle_use_aux_hidden_state = False
        self.eagle_draft_num_layers = None
        self.dflash_family_use_aux_hidden_state = False
        self.dflash_family_target_layer_ids = None
        self.dflash_family_draft_num_layers = None
        if (
            (self.spec_algorithm.is_eagle() or self.spec_algorithm.is_standalone())
            and not self.is_draft_worker
            and server_args.speculative_draft_model_path
        ):
            # Load draft config to get layer count for KV cache sizing
            draft_model_config = self._build_model_config(
                server_args,
                model_path=server_args.speculative_draft_model_path,
                model_revision=server_args.speculative_draft_model_revision,
                is_draft_model=True,
            )
            num_nextn_predict_layers = draft_model_config.num_nextn_predict_layers
            if num_nextn_predict_layers is not None:
                self.eagle_draft_num_layers = int(num_nextn_predict_layers)
            else:
                self.eagle_draft_num_layers = int(
                    max(
                        draft_model_config.num_hidden_layers,
                        draft_model_config.num_attention_layers,
                    )
                )

            if self.spec_algorithm.is_eagle3():
                self.eagle_use_aux_hidden_state = True
                try:
                    eagle_config = getattr(
                        draft_model_config.hf_config, "eagle_config", None
                    )
                    self.eagle_use_aux_hidden_state = eagle_config.get(
                        "use_aux_hidden_state", True
                    )
                    self.eagle_aux_hidden_state_layer_ids = eagle_config[
                        "eagle_aux_hidden_state_layer_ids"
                    ]
                except:
                    # if there is no aux layer, set to None
                    self.eagle_aux_hidden_state_layer_ids = None

        # T156 stage 2 (--speculative-cross-algorithm): when the FORCED rung is
        # NEXTN, spec_algorithm is EAGLE but a DFLASH rung is co-resident; the
        # target still needs the DFLASH planning fields (draft layer count for
        # KV-cell sizing, capture layer ids). Its draft path lives in the
        # stashed dflash shape, not in speculative_draft_model_path (that one
        # is None, matching the plain NEXTN server). Whether the target model
        # ACTUALLY captures aux hidden states at runtime is toggled by
        # CrossAlgoWorker per forced rung (NEXTN needs the final hidden state,
        # DFLASH the concatenated aux layers).
        _cross_dflash_shape = None
        if (
            not self.is_draft_worker
            and getattr(server_args, "speculative_cross_algorithm", False)
            and not self.spec_algorithm.is_dflash_family()
        ):
            from sglang.srt.speculative.cross_algo_utils import get_cross_shapes

            _cross_dflash_shape = get_cross_shapes(server_args).get("dflash")

        if (
            self.spec_algorithm.is_dflash_family() or _cross_dflash_shape is not None
        ) and not self.is_draft_worker:
            from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config

            # Select target layers to capture for building draft context features.
            draft_model_config = self._build_model_config(
                server_args,
                model_path=(
                    _cross_dflash_shape["speculative_draft_model_path"]
                    if _cross_dflash_shape is not None
                    else server_args.speculative_draft_model_path
                ),
                model_revision=(
                    _cross_dflash_shape["speculative_draft_model_revision"]
                    if _cross_dflash_shape is not None
                    else server_args.speculative_draft_model_revision
                ),
                is_draft_model=True,
            )
            dflash_draft_config = parse_dflash_draft_config(
                draft_hf_config=draft_model_config.hf_config
            )
            draft_num_layers = dflash_draft_config.require_num_layers()
            trained_target_layers = dflash_draft_config.num_target_layers

            target_num_layers = getattr(
                self.model_config.hf_text_config, "num_hidden_layers", None
            )
            if target_num_layers is None:
                raise ValueError(
                    "Block-draft-with-target-kv spec requires target num_hidden_layers "
                    f"in config. Got target={target_num_layers}."
                )
            target_num_layers = int(target_num_layers)

            if (
                trained_target_layers is not None
                and trained_target_layers != target_num_layers
            ):
                logger.warning(
                    "Draft config num_target_layers=%s differs from runtime target num_hidden_layers=%s; "
                    "selecting capture layers based on the runtime target model.",
                    trained_target_layers,
                    target_num_layers,
                )

            target_layer_ids = dflash_draft_config.resolve_target_layer_ids(
                target_num_layers=int(target_num_layers),
                draft_num_layers=int(draft_num_layers),
            )

            if self.spec_algorithm.is_dspark():
                from sglang.srt.speculative.dspark_components.dspark_config import (
                    parse_dspark_draft_config,
                )

                dspark_draft_config = parse_dspark_draft_config(
                    draft_hf_config=draft_model_config.hf_config
                )
                if not dspark_draft_config.require_markov():
                    raise ValueError(
                        "DSPARK requires markov_rank > 0 in the draft config, "
                        f"got markov_rank={dspark_draft_config.markov_rank}."
                    )
                if dspark_draft_config.target_layer_ids is not None:
                    target_layer_ids = list(dspark_draft_config.target_layer_ids)

            self.dflash_family_use_aux_hidden_state = True
            self.dflash_family_draft_num_layers = int(draft_num_layers)
            self.dflash_family_target_layer_ids = target_layer_ids

        # Apply the rank zero filter to logger
        if server_args.show_time_cost:
            enable_show_time_cost()

        # Chunked prefix caching requires an MLA model on a backend whose
        # kernels read that layout. This is a load-time gate, not a
        # resolution-time one: out-of-tree platforms register their supported
        # backends in init_backend(), which runs when this module is imported
        # — after ServerArgs.__post_init__. Target runner only: a draft
        # model's (often non-MLA) config must not flip the shared setting.
        if not self.is_draft_worker and (
            not self.use_mla_backend
            or server_args.attention_backend
            not in CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS
        ):
            if not server_args.disable_chunked_prefix_cache:
                server_args.override(
                    "model_runner.chunked_prefix_cache_gate",
                    disable_chunked_prefix_cache=True,
                )
        if not self.is_draft_worker and not server_args.disable_chunked_prefix_cache:
            logger.info("Chunked prefix cache is turned on.")

        # Set the global server_args in the scheduler process (target worker
        # only, so a draft init cannot clobber target-derived global state).
        if not self.is_draft_worker:
            set_global_server_args_for_scheduler(server_args)

        # Init OpenMP threads binding for CPU
        if self.device == "cpu":
            self.init_threads_binding()

        # Set float32 matmul precision
        if get_server_args().enable_tf32_matmul:
            torch.set_float32_matmul_precision("high")

        # Get available memory before model loading.
        # Stored for later use by alloc_memory_pool().
        self.pre_model_load_memory = self.init_torch_distributed()

        # Initialize MooncakeTransferEngine
        self.init_shared_mooncake_transfer_engine()

        # Init forward stream for overlap schedule
        self.forward_stream = torch.get_device_module(self.device).Stream()

        # WAR fast-path: a decode-graph forward publishes a fresh event here after
        # load_batch; the scheduler's WAR barrier waits on it (then clears it)
        # instead of the whole-forward wait_stream. None -> whole-forward fallback.
        self.war_fastpath_read_done_event: Optional[torch.cuda.Event] = None

        # CPU offload. set_offloader is a PROCESS-GLOBAL install (the known
        # unguarded global of this family, DESIGN_121 §1.4): a lane runner
        # constructing after the serving rank would silently replace the
        # resident offloader mid-flight. The lane computes with the resident
        # weights and needs no offloader of its own. Same guard for the #631
        # phase-flip TP stack: it builds AFTER the primary PP stack in the
        # same process and must not replace the primary's offloader.
        if not self.is_dual_group_lane and not self.is_phase_flip_tp_stack:
            set_offloader(
                create_offloader_from_server_args(server_args, dp_rank=dp_rank)
            )

        self._weight_checker = WeightChecker(model_runner=self)

        if envs.SGLANG_DETECT_SLOW_RANK.get():
            slow_rank_detector.execute()

        # Init mindspore running environment when model impl is "mindspore"
        self.init_mindspore_runner()

        # Update deep gemm configure
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            deep_gemm_wrapper.update_deep_gemm_config(gpu_id, server_args)

        # For hisparse (must be set before initialize() so CUDA graph capture can see it)
        self.hisparse_coordinator = None

        self._linear_attn_registry_cache: Any = _UNSET

        # Load model weights and configure
        self.initialize()
        self.check_quantized_moe_compatibility()

        if (
            self.server_args.elastic_ep_backend is not None
            and self.server_args.elastic_ep_rejoin
        ):
            join_process_groups()
            broadcast_global_expert_location_metadata(
                src_rank=self._get_healthy_expert_location_src_rank(
                    invoked_in_elastic_ep_rejoin_path=True
                )
            )
            ElasticEPStateManager.instance().reset()

        if self.is_multimodal:
            sanity_check_mm_pad_shift_value(self.model_config.vocab_size)

        # Temporary cached values
        self.support_pp = (
            "pp_proxy_tensors" in inspect.signature(self.model.forward).parameters
        )

        if self.pp_size > 1:
            assert (
                self.support_pp
            ), "Pipeline Parallel is not compatible with this model."

        # For weight updates
        self._model_update_group = {}
        self._weights_send_group = {}

    def _build_model_config(
        self, server_args, model_path=None, model_revision=None, is_draft_model=False
    ):
        return ModelConfig.from_server_args(
            server_args,
            model_path=model_path,
            model_revision=model_revision,
            is_draft_model=is_draft_model,
        )

    def init_msprobe(self):
        # Init the msprobe
        try:
            from msprobe.pytorch import PrecisionDebugger, seed_all
        except ImportError:
            logger.warning(
                "Please install msprobe for tensor data dump: pip install mindstudio-probe --pre, "
                "see https://gitcode.com/Ascend/msprobe for details."
            )
            return
        seed_all(mode=True)
        self.msprobe_debugger = PrecisionDebugger(
            config_path=self.server_args.msprobe_dump_config
        )

    def init_mindspore_runner(self):
        # Init the mindspore runner
        # for now, there is only some communication initialization work
        if self.server_args.model_impl.lower() == ModelImpl.MINDSPORE and _is_npu:
            from sglang.srt.model_executor.mindspore_runner import init_ms_distributed

            init_ms_distributed(
                world_size=self.tp_size * self.pp_size,
                rank=self.tp_size * self.pp_rank + self.tp_rank,
                local_rank=self.gpu_id,
                server_args=self.server_args,
                port=self.dist_port,
            )

    def _check_bar1_fp8_uneven_dcp_combination(self):
        """Warn about the slow first boot of the #431 combination (#431).

        This was a hard refusal. The #424 battery saw barlink BAR1 x uneven
        weighted DCP x an fp8 checkpoint wedge the prefill path in both
        layouts, and this method refused the combination before a weight was
        loaded. The 2026-08-02 re-check on the tip carrying the BAR1 deadline
        fix ran the same arm to completion at the stock-NCCL twin's speed, so
        what #424 recorded as a wedge is a slow first boot -- roughly 190 s
        per rank inside the initial JIT cold-build window on a cold kernel
        cache. The refusal is a warning now, and it is deliberately verbose:
        the failure mode it prevents is an operator killing a boot that is
        working. Evidence and the full history are in the predicate's
        docstring and in docs/dev/ANALYSE_431_fp8_bar1_dcp_deadlock.md.

        An operator who wants the old behaviour still has it -- see
        SGLANG_BARLINK_REFUSE_FP8_UNEVEN_DCP_BAR1 and the legacy
        SGLANG_BARLINK_ALLOW_FP8_UNEVEN_DCP_BAR1=0 -- in which case this
        raises exactly as before.

        The predicate lives in `barlink_uniformity` as a pure function so it
        can be falsified without a GPU; this method only supplies the values.
        """
        from sglang.srt.distributed.device_communicators.barlink_uniformity import (
            bar1_fp8_uneven_dcp_notice,
        )

        notice = bar1_fp8_uneven_dcp_notice(
            barlink_enabled=envs.SGLANG_BARLINK.get(),
            transport=envs.SGLANG_BARLINK_TRANSPORT.get(),
            uneven_weighted_dcp=(
                self.server_args.uneven_weighted_dcp_enabled() and self.dcp_size > 1
            ),
            quantization=self.model_config.quantization,
        )
        if notice is None:
            return
        if notice.refuse:
            raise ValueError(notice.message)
        logger.warning(notice.message)

    def initialize(self):
        server_args = self.server_args

        # #431: warn about barlink BAR1 x uneven weighted DCP x an fp8
        # checkpoint before a single weight is loaded, so the slow-first-boot
        # notice is on screen BEFORE the operator starts waiting. HERE and not
        # in ServerArgs because the predicate needs the RESOLVED quantization:
        # the #424 arms passed no --quantization at all and the method comes
        # out of the checkpoint's own quant_config in
        # ModelConfig._verify_quantization. Rank-uniform by construction
        # (server args + model config are replicated), so the notice fires on
        # every rank or on none -- which also means the opt-in refusal raises
        # on every rank or on none.
        self._check_bar1_fp8_uneven_dcp_combination()

        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.server_args.enable_memory_saver
        )

        if self.server_args.remote_instance_weight_loader_use_transfer_engine():
            self.remote_instance_init_transfer_engine()

        if not self.is_draft_worker:
            set_global_expert_location_metadata(
                compute_initial_expert_location_metadata(
                    server_args=server_args,
                    model_config=self.model_config,
                    moe_ep_rank=self.moe_ep_rank,
                )
            )
            if self.tp_rank == 0 and envs.SGLANG_LOG_EXPERT_LOCATION_METADATA.get():
                logger.info(
                    "Initial expert_location_metadata:\n%s",
                    format_expert_location_layout(
                        get_global_expert_location_metadata()
                    ),
                )

            set_global_expert_distribution_recorder(
                ExpertDistributionRecorder.init_new(
                    server_args,
                    get_global_expert_location_metadata(),
                    rank=self.tp_rank,
                )
            )

        if self.server_args.ep_dispatch_algorithm == "lp" and not self.is_draft_worker:
            self._init_lplb_solvers()

        # Expert parallelism
        self.eplb_manager = (
            EPLBManager(self)
            if self.server_args.enable_eplb and (not self.is_draft_worker)
            else None
        )
        self.expert_location_updater = ExpertLocationUpdater()

        if self.server_args.elastic_ep_backend:
            ElasticEPStateManager.init(self.server_args)
        self._token_oracle_manager = install_token_oracle_from_env(
            server_args=server_args,
            vocab_size=self.model_config.vocab_size,
        )
        # Load the model
        self.sampler = create_sampler()
        # #605 phase marks. Each pair of consecutive marks IS one measured
        # post; no-ops unless SGLANG_VRAM_FLIGHT_DIR is set.
        from sglang.srt.mem_ledger import flight_recorder

        flight_recorder.mark(
            "pre_weight_load",
            rank=self.tp_rank,
            extra={"draft_worker": bool(self.is_draft_worker)},
        )
        self.load_model()
        flight_recorder.mark(
            "weights_loaded",
            rank=self.tp_rank,
            extra={"draft_worker": bool(self.is_draft_worker)},
        )
        self._attach_layer_fingerprint()
        self._prepare_moe_topk()

        # Must run before backend/graph init so no draft graph records a
        # routed-experts capture-write kernel.
        if self.is_draft_worker:
            disable_routed_experts_capture_for_draft(self.model)

        # Load the expert backup client
        self.expert_backup_client = (
            ExpertBackupClient(self.server_args, self)
            if (
                self.server_args.enable_elastic_expert_backup
                and self.server_args.elastic_ep_backend is not None
            )
            else None
        )

        if (
            self.server_args.remote_instance_weight_loader_use_transfer_engine()
            # ModelExpress owns TransferEngine memory registration and metadata
            # publishing for backend=modelexpress. Re-registering here would
            # overlap the same weight buffers.
            and self.server_args.remote_instance_weight_loader_backend
            != RemoteInstanceWeightLoaderBackend.MODELEXPRESS
            and self.remote_instance_transfer_engine is not None
            and self.remote_instance_transfer_engine_weight_info is None
        ):
            # Register memory and upstream the transfer engine info to the bootstrap server
            self.remote_instance_transfer_engine_weight_info = register_memory_region(
                self.model, self.remote_instance_transfer_engine
            )
            self._register_to_engine_info_bootstrap()

        # For MTP models like DeepSeek-V3 or GLM-4.5, the MTP layer(s) are used separately as draft
        # models for speculative decoding. In those cases, `num_nextn_predict_layers` is used to
        # determine the number of layers.
        # Some EAGLE3 drafts (e.g. nvidia/Kimi-K2.5-Thinking-Eagle3) carry the full DeepSeek-V3
        # config schema and explicitly set `num_nextn_predict_layers: 0`. Treat that the same as
        # the field being absent — otherwise the draft worker takes the MTP branch below with
        # model_num_layers=0, sizing the draft KV pool to zero and producing an IndexError on
        # the first forward (`set_mla_kv_buffer` -> `self.kv_buffer[layer_id - self.start_layer]`).
        _nnpl = self.model_config.num_nextn_predict_layers
        model_has_mtp_layers = _nnpl is not None and _nnpl > 0
        if self.is_draft_worker and model_has_mtp_layers:
            model_num_layers = getattr(
                self.model, "num_stages", self.model_config.num_nextn_predict_layers
            )
        else:
            model_num_layers = max(
                self.model_config.num_hidden_layers,
                self.model_config.num_attention_layers,
            )
        if self.model_config.hf_config.architectures[0] == "MiMoV2MTP":
            model_num_layers = 1
        elif self.model_config.hf_config.architectures[0] == "Step3p5MTP":
            model_num_layers = 1
        self.start_layer = getattr(self.model, "start_layer", 0)
        self.end_layer = getattr(self.model, "end_layer", model_num_layers)
        # #pp-layer-set: end - start is the SPAN, which equals the COUNT only
        # while a stage is a contiguous interval. Under SGLANG_PP_LAYER_SET a
        # stage owns an arbitrary set -- e.g. the 8 full-attention layers of a
        # 64-layer hybrid, spanning [3, 64) -- and the span would over-count
        # them 61 to 8. This number feeds `layer_num=` for KV pool allocation
        # (model_runner_kv_cache_mixin.py), so the span would size the pool
        # 7.6x too large and the boot would fail on memory it never needed.
        # Read from the SAME parser that built the layer list, rather than
        # asking the model to propagate an attribute -- that would need an edit
        # in every model file, and two places deriving ownership is how they
        # start disagreeing.
        from sglang.srt.distributed.utils import get_pp_layer_set

        owned = get_pp_layer_set(model_num_layers, self.pp_rank, self.pp_size)
        if owned is not None:
            self.num_effective_layers = len(owned)
        else:
            self.num_effective_layers = self.end_layer - self.start_layer

        self.adjust_hybrid_swa_layers_for_pp()

        # For LoopCoder models, each loop has its own layer_id, so we need to multiply by loop_num
        loop_num = getattr(self.model_config.hf_config, "loop_num", 1)
        if loop_num > 1:
            self.num_effective_layers = self.num_effective_layers * loop_num

        assert (
            (not model_has_mtp_layers)
            or (self.spec_algorithm.is_none())
            or (
                (not self.spec_algorithm.is_none())
                and (self.num_effective_layers == model_num_layers)
            )
        ), "PP is not compatible with MTP models."

        # BOOT GATE tier 2 (#108): --draft-kv-layout dcp is a one-layer
        # contract, and this is the first point where the resolved draft
        # checkpoint's KV layer count is actually known. ServerArgs bounded
        # the algorithm, the topk and --enable-multi-layer-eagle from the CLI;
        # a draft CHECKPOINT carrying several attention layers is only visible
        # here. Fires before the KV pool is built, so the failure is a
        # sentence rather than a shape mismatch in the draft capture.
        if self.is_draft_worker:
            from sglang.srt.layers.dcp.owner import (
                draft_kv_layout_is_dcp,
                reject_multi_layer_draft_kv_dcp,
            )

            if draft_kv_layout_is_dcp(self.server_args):
                reject_multi_layer_draft_kv_dcp(
                    self.num_effective_layers,
                    self.model_config.hf_config.architectures[0],
                )

        # Apply torchao quantization
        torchao_applied = getattr(self.model, "torchao_applied", False)
        # In layered loading, torchao may have been applied
        if not torchao_applied:
            apply_torchao_config_to_model(self.model, get_server_args().torchao_config)

        # Apply torch TP if the model supports it
        supports_torch_tp = getattr(self.model, "supports_torch_tp", False)
        if self.tp_size > 1 and supports_torch_tp:
            self.apply_torch_tp()

        # Init lora
        if server_args.enable_lora:
            self.init_lora_manager()
            if not cuda_graph_fully_disabled():
                # Phase 1 of LoRA CUDA graph init: pre-allocate large MoE
                # intermediate buffers before init_memory_pool() so memory
                # profiling accounts for them. The buffers are reused by
                # any captured graph (decode today; widen here so any
                # future prefill capture path also picks them up).
                self._init_lora_cuda_graph_moe_buffers()

        # Enable batch invariant mode
        if server_args.enable_deterministic_inference:
            from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

            enable_batch_invariant_mode()

        # Deduce KV cache dtype
        self.configure_kv_cache_dtype()

    def get_pp_proxy_topk_size(self) -> Optional[int]:
        hf_config = self.model_config.hf_text_config
        if (
            self.pp_size <= 1
            or self.pp_rank == 0
            or not is_deepseek_dsa(hf_config)
            or not dsa_layer_skips_topk(hf_config, self.start_layer)
        ):
            return None
        return getattr(hf_config, "index_topk", None)

    def decode_num_tokens_per_bs(
        self, *, num_draft_tokens: Optional[int] = None
    ) -> int:
        """Logits rows per decode batch slot."""
        if self.spec_algorithm.is_speculative():
            if num_draft_tokens is None:
                num_draft_tokens = self.server_args.speculative_num_draft_tokens
            return self.spec_algorithm.get_num_tokens_per_bs_for_target_verify(
                num_draft_tokens, self.is_draft_worker
            )
        dllm_config = DllmConfig.from_server_args(self.server_args)
        return dllm_config.block_size if dllm_config is not None else 1

    def max_decode_logits_rows(self) -> int:
        """Rows the shared logits buffer needs.

        Size with ``max_speculative_num_draft_tokens``: under adaptive
        speculative decoding the per-step runtime states capture verify /
        draft-extend graphs up to ``max(candidate_steps) + 1`` draft tokens,
        which exceeds the static ``speculative_num_draft_tokens``. Non-adaptive
        (and non-speculative) configs resolve to the same value as before.
        """
        num_tokens_per_bs = self.decode_num_tokens_per_bs(
            num_draft_tokens=self.server_args.max_speculative_num_draft_tokens
        )
        capture_bs, _ = get_batch_sizes_to_capture(self, num_tokens_per_bs)
        rows = max(capture_bs) * num_tokens_per_bs
        if self.dual_group_lane_verify_tokens:
            # The lane's chain-verify entry emits one logits row per candidate
            # (#274 round 6). Its runner is non-speculative, so the line above
            # sizes for one row per slot and the capture would trip the shared
            # buffer's row assert. Lane-keyed buffer: a concurrent lane grows
            # its own, the serving group's is untouched.
            # Round 7a: the widest rung of the ladder decides.
            rows = max(capture_bs) * max(
                num_tokens_per_bs, *self.dual_group_lane_verify_tokens
            )
        return rows

    @time_startup_latency(name="memory_pool_init")
    def alloc_memory_pool(self, memory_pool_config: Optional[MemoryPoolConfig] = None):
        """Allocate KV cache memory pools only (no backends or cuda graphs)."""
        if memory_pool_config is not None:
            self.memory_pool_config = memory_pool_config

        self.init_memory_pool(self.pre_model_load_memory)

        from sglang.srt.mem_ledger import flight_recorder

        flight_recorder.mark(
            "kv_pool_sized",
            rank=self.tp_rank,
            extra={"draft_worker": bool(self.is_draft_worker)},
        )

        # Must be called AFTER init_memory_pool so the pool object exists for
        # canary to monkey-patch, and BEFORE init_decode_cuda_graph so warmup
        # forwards captured into the graph see the patched pool methods.
        self.canary_manager = install_canary(
            server_args=self.server_args,
            model_runner=self,
            token_oracle_manager=self._token_oracle_manager,
        )

        # Init ngram embedding token table
        self.maybe_init_ngram_embedding()

        if self.enable_hisparse:
            from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
            from sglang.srt.mem_cache.sparsity import parse_hisparse_config

            hisparse_cfg = parse_hisparse_config(self.server_args)
            hisparse_top_k = getattr(
                self.model_config.hf_text_config, "index_topk", hisparse_cfg.top_k
            )
            self.hisparse_coordinator = HiSparseCoordinator(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                top_k=hisparse_top_k,
                device_buffer_size=hisparse_cfg.device_buffer_size,
                device=self.device,
                tp_group=(
                    self.attention_tp_group.cpu_group
                    if self.server_args.enable_dp_attention
                    else self.tp_group.cpu_group
                ),
                host_to_device_ratio=hisparse_cfg.host_to_device_ratio,
                swap_in_block_size=hisparse_cfg.swap_in_block_size,
            )

        self.init_routed_experts_capturer()
        self.init_indexer_capturer()

        self.attn_backend = None
        self.decode_attn_backend = None
        self.decode_attn_backend_group = []
        self.decode_cuda_graph_runner = None
        self.graph_mem_usage = 0
        self.prefill_cuda_graph_runner = None
        self.graph_shared_output = None
        # #274 round 6: K+1 when this runner is a dual-group lane TARGET that
        # speculates, so its decode graph runner adds a chain-verify entry
        # beside the plain decode ones. Written by the lane bring-up between
        # construction and init_cuda_graphs(), like
        # spec_solo_rank_local_graphs. None everywhere else.
        #
        # Round 7a makes it the K LADDER: a tuple of candidate-row counts
        # (K+1 per rung), one captured entry each, so a rung change at runtime
        # is a graph-key flip rather than a re-capture.
        self.dual_group_lane_verify_tokens: Optional[Tuple[int, ...]] = None
        # #274 round 7a: True when this runner is the dual-group lane's NEXTN
        # HEAD. Its decode capture then builds a real EagleDraftInput instead
        # of the None an MTP forward dereferences. False everywhere else.
        self.dual_group_lane_draft_capture: bool = False

    @time_startup_latency(name="attention_backend_init")
    def init_attention_backends(self):
        """Initialize attention backends only (no cuda graph capture)."""
        # TODO: Refactor device-specific init branches into platform interface (separate PR).
        # Must be called BEFORE init_decode_cuda_graph() so CUDA graph capture
        # runs with aux hidden state capture enabled.
        self.init_aux_hidden_state_capture()

        if self.device == "cuda" or self.device == "musa":
            self.init_cublas()
            self.init_attention_backend()
        elif self.device in ["cpu", "xpu"]:
            self.init_attention_backend()
        elif self.device == "npu":
            self.init_attention_backend()
            # lazy init for zbal with mix mode (before graph capture when enable_cuda_graph)
            if envs.SGLANG_ZBAL_LOCAL_MEM_SIZE.get() > 0 and not self.is_draft_worker:
                from sglang.srt.hardware_backend.npu.utils import lazy_init_zbal_gva_mem

                lazy_init_zbal_gva_mem(
                    self.device,
                    self.gpu_id,
                    get_world_group().rank_in_group,
                    get_world_group().world_size,
                    get_world_group().cpu_group,
                )
        else:
            self.init_attention_backend()

    def _harmonize_cuda_graph_plan(self):
        """Make the resolved cuda-graph plan identical on every rank.

        `cuda_graph_config` is resolved per rank from that rank's own device,
        model and memory, through an auto-disable cascade. On a homogeneous
        group every rank reaches the same answer and this function is a no-op.
        On a MIXED group it silently produces different plans: measured on an
        RTX 2080 Ti + Radeon RX Vega 64 pair, rank 0 began prefill capture
        (backend=breakable) while rank 1's cascade disabled prefill and went
        straight to decode. Nobody selected that.

        A captured rank replays a FIXED collective sequence while an eager rank
        re-derives it per batch, so a phase captured on one rank and not the
        other is a rank-divergent execution path -- the same class of defect as
        a kernel chosen by platform instead of availability, and the reason the
        weightless-KV path already routes BOTH sides to the EagerRunner rather
        than letting one side capture (see init_prefill_cuda_graph).

        Rule: gather each rank's per-phase backend; if they do not all agree,
        DISABLE that phase for everyone. Deliberately not "pick the most
        capable" or an ordering over backends -- two ranks capturing different
        STRUCTURES is exactly what must not happen, and eager is the only
        option guaranteed available everywhere. Logged on every rank, because a
        silently downgraded plan is how this class of bug hides.

        On a homogeneous group the gathered values are equal by construction,
        so nothing changes -- that is the NVIDIA regression criterion.
        """
        import torch.distributed as dist

        from sglang.srt.runtime_context import get_server_args

        try:
            server_args = get_server_args()
        except ValueError:
            return  # no published ServerArgs (unit tests, early startup)
        cfg = getattr(server_args, "cuda_graph_config", None)
        if cfg is None:
            return

        # DRAFT-SOLO: this runner's graphs are RANK-LOCAL, so there is no group
        # to harmonise with -- and harmonising would hang.
        #
        # Under --speculative-draft-placement solo only the solo rank owns a
        # real draft runner; the shadow ranks get a stub
        # (install_shadow_draft_runner_surface) and never call
        # init_cuda_graphs on it. The all_gather_object below is group-wide, so
        # the solo rank would enter a collective that no other rank ever joins.
        #
        # MEASURED, Nordstern L0 S4 (TP=5 over two hosts, --disable-cuda-graph):
        # ranks 1-4 reached "Tree cache initialized" and their health checks
        # while rank 0 sat here for 8 minutes until the launcher killed it.
        # py-spy on the solo rank:
        #     all_gather_object -> _harmonize_cuda_graph_plan (model_runner)
        #     -> init_cuda_graphs -> eagle_worker_v2.init_cuda_graphs
        #
        # `spec_solo_rank_local_graphs` already declares exactly this property
        # (it suppresses the capture backends' per-warmup TP barrier for this
        # runner); the harmonisation is a second group collective in the same
        # category and needs the same exemption. Note it also MUTATES the
        # shared cuda_graph_config, so letting the draft runner harmonise could
        # disable a phase for the target runner too.
        if getattr(self, "spec_solo_rank_local_graphs", False):
            return

        group = (
            self.attention_tp_group.cpu_group
            if self.server_args.enable_dp_attention
            else self.tp_group.cpu_group
        )
        if group is None or dist.get_world_size(group) <= 1:
            return

        local = {phase: getattr(cfg, phase).backend for phase in Phase.ALL}
        gathered: list = [None] * dist.get_world_size(group)
        dist.all_gather_object(gathered, local, group=group)

        for phase in Phase.ALL:
            seen = {g[phase] for g in gathered if g is not None}
            if len(seen) <= 1:
                continue
            logger.warning(
                "CUDA graph plan differs across the group for the %s phase "
                "(ranks resolved %s); disabling it on every rank so the "
                "collective sequence stays rank-uniform. Set "
                "--cuda-graph-backend-%s explicitly to make this deliberate.",
                phase,
                ", ".join(sorted(seen)),
                phase,
            )
            getattr(cfg, phase).backend = Backend.DISABLED

    @property
    def is_phase_flip_pp_stack(self) -> bool:
        """#631: True for the PRIMARY (PP prefill) stack of a phase-flip
        boot. False on every non-flip boot (default path untouched), on the
        flip's own TP stack, and on any draft/lane secondary runner."""
        return (
            getattr(self.server_args, "enable_phase_flip", False)
            and not self.is_draft_worker
            and not self.is_phase_flip_tp_stack
        )

    def _install_eager_only_runners(self, reason: str) -> None:
        """The complete no-capture terminal state for this runner.

        Shared by every branch that declines to capture, because the state
        is four coupled assignments and a divergence between two copies of
        them is invisible until a forward pass reaches the one that was
        missed. ``eager_runner`` in particular is what the prefill and
        decode runners are aliased to; leaving it unset does not fall back
        to an eager path, it raises AttributeError at the first forward.
        """
        self.graph_shared_output = GraphSharedOutput.create_for_model_runner(self)
        self.eager_runner = EagerRunner(self)
        self.prefill_cuda_graph_runner = self.eager_runner
        self.decode_cuda_graph_runner = self.eager_runner
        self.graph_mem_usage = 0
        logger.info(reason)

    @time_startup_latency(name="cuda_graph_capture")
    def init_cuda_graphs(self, capture_decode_cuda_graph: bool = True):
        """Capture cuda graphs. Requires init_attention_backends() to have run.

        Spec draft runners pass capture_decode_cuda_graph=False
        because they capture their own decode-style graphs separately.
        """

        # #631 operator pin 2, STRUCTURAL: the phase flip's PP stack captures
        # NO graphs at all -- it runs eager prefill exactly like the measured
        # #625 recipe, and PP decode never runs in steady state (3.6x worse
        # than TP decode; the whole point of the flip). The carve sits BEFORE
        # the plan harmonization collective and is rank-uniform (the flag is
        # boot-arg-uniform), so no rank ever waits in a collective the others
        # skipped. The capture entry points below additionally REFUSE for
        # this stack, making a PP decode-graph set unreachable by
        # construction, not merely configured off.
        if self.is_phase_flip_pp_stack:
            self._install_eager_only_runners(
                "PHASE-FLIP PP stack: no CUDA graphs captured by "
                "construction (eager prefill per the #625 recipe; decode "
                "graphs belong exclusively to the TP stack)."
            )
            return

        # #656 spec item 8: --disable-draft-cuda-graph. THE REFUSAL BELONGS
        # HERE, not around this call. The first attempt skipped
        # init_cuda_graphs wholesale from the draft worker and all three
        # ranks died at the first draft extend with
        #   AttributeError: 'ModelRunner' object has no attribute 'eager_runner'
        # because this method builds the EAGER runner as well as the graph
        # runners -- the eager path is not the absence of the graph path, it
        # is a thing that has to be constructed. Refusing inside, where that
        # construction lives, gives the caller the eager draft runner it
        # needs and no captures.
        # Imported locally: base_spec_worker reaches back into this module's
        # neighbourhood, and a module-level import here closes the cycle.
        from sglang.srt.speculative.base_spec_worker import (
            should_capture_draft_graphs,
        )

        if self.is_draft_worker and not should_capture_draft_graphs(self.server_args):
            self._install_eager_only_runners(
                "Draft CUDA graphs DISABLED by --disable-draft-cuda-graph; "
                "this draft runner is eager. Target-model graphs unaffected."
            )
            return

        # The graph plan must be a GROUP decision, not a per-rank one.
        self._harmonize_cuda_graph_plan()

        # Phase-footprint probe: baseline here, with the KV pool sized and
        # nothing captured yet. Inert unless SGLANG_PHASE_FOOTPRINT_DUMP is set.
        from sglang.srt.mem_ledger import activation_probe, flight_recorder

        activation_probe.note_capture_begin()
        # Marked separately from the probe above rather than folded into it:
        # note_capture_begin returns early unless its OWN env var is armed, so
        # a mark placed inside it would be silently conditional on an
        # unrelated instrument.
        flight_recorder.mark(
            "capture_begin",
            rank=self.tp_rank,
            extra={"draft_worker": bool(self.is_draft_worker)},
        )

        self.graph_shared_output = GraphSharedOutput.create_for_model_runner(self)

        # The eager (no-cuda-graph) phase runner, built AFTER the attention
        # backend so its __init__ can warm up kernels (run-once) and allocate the
        # fixed-max static buffer — both before the cuda-graph runners, so that
        # buffer is canonical in the shared pool and the cg runners coalesce onto
        # it. Always built: it serves both the fully-disabled case (decode/prefill
        # runners point at it) and the eager fallback when a cg runner can't run a
        # batch.
        self.eager_runner = EagerRunner(self)

        # cuda-graph capture: prefill before decode, so both coalesce onto the
        # eager buffer allocated above. (init_prefill_cuda_graph routes prefill
        # to the eager runner when the prefill graph is disabled.)
        self.init_prefill_cuda_graph()

        self.decode_cuda_graph_runner = None
        self.graph_mem_usage = 0

        if capture_decode_cuda_graph:
            if self.device in ("cuda", "musa", "cpu", "npu", "xpu"):
                self.init_decode_cuda_graph()
            elif (
                current_platform.is_out_of_tree()
                and current_platform.support_cuda_graph()
            ):
                self.init_decode_cuda_graph()
        else:
            self.decode_cuda_graph_runner = self.eager_runner

        # Register forward hooks AFTER cuda-graph capture so their tensor ops are
        # not traced into any captured graph — capture stays hook-free and hooks
        # fire only on the eager forward path (capture replay never runs Python
        # hooks anyway).
        if self.server_args.forward_hooks:
            register_forward_hooks(self.model, self.server_args.forward_hooks)

        self.prealloc_symmetric_memory_pool()

        # Phase-footprint probe: charge what the captured graphs KEEP, then
        # re-baseline so the prefill peak is measured from the steady state.
        # Called once per runner, so a speculative process folds the target's
        # and the draft's capture cost together.
        activation_probe.note_capture_end()
        # Tagged with the runner, because a speculative process runs this twice
        # (target and NEXTN draft) and their capture costs ADD. The snapshot is
        # NOT dumped here for the same reason: the first runner to finish would
        # write a picture that is missing the second one's captures. It is
        # dumped once per process, after every runner is up, in
        # run_scheduler_process.
        flight_recorder.mark(
            "capture_end",
            rank=self.tp_rank,
            extra={"draft_worker": bool(self.is_draft_worker)},
        )

        if self.canary_manager is not None and not self.is_draft_worker:
            self.canary_manager.mark_init_finished()

    def adjust_hybrid_swa_layers_for_pp(self):
        if not self.is_hybrid_swa:
            return

        if self.model_config.is_deepseek_v4_arch:
            return

        full_attention_layer_ids = [
            layer_idx
            for layer_idx in range(self.start_layer, self.end_layer + 1)
            if hasattr(self.model_config, "full_attention_layer_ids")
            and layer_idx in self.model_config.full_attention_layer_ids
        ]
        swa_attention_layer_ids = [
            layer_idx
            for layer_idx in range(self.start_layer, self.end_layer + 1)
            if hasattr(self.model_config, "swa_attention_layer_ids")
            and layer_idx in self.model_config.swa_attention_layer_ids
        ]
        self.model_config.swa_attention_layer_ids = swa_attention_layer_ids
        self.model_config.full_attention_layer_ids = full_attention_layer_ids

    def init_routed_experts_capturer(self):
        if self.is_draft_worker:
            # Capture is target-only. The draft worker runs in the same process
            # as its target and inits after it, so installing a capturer here
            # would overwrite the target's process-global one.
            return

        if not self.server_args.disable_shared_experts_fusion and hasattr(
            self.model, "num_fused_shared_experts"
        ):
            num_fused_shared_experts = self.model.num_fused_shared_experts
        else:
            num_fused_shared_experts = 0

        set_global_experts_capturer(
            RoutedExpertsCapturer.create(
                enable=get_server_args().enable_return_routed_experts,
                model_config=self.model_config,
                num_fused_shared_experts=num_fused_shared_experts,
                num_tokens=self.max_total_num_tokens + self.page_size,
                max_running_requests=self.max_running_requests,
                device=self.device,
            )
        )

    def init_indexer_capturer(self):
        enable = get_server_args().enable_return_indexer_topk
        # Producer wiring is CUDA-only (Indexer.forward_cuda + MLA skip_topk
        # path); other backends would create a capturer but never feed it.
        if enable and self.device != "cuda":
            logger.warning(
                "indexer-topk capture is CUDA-only; %s backend not yet wired. "
                "Disabling capturer.",
                self.device,
            )
            set_global_indexer_capturer(None)
            return

        hf_text_config = self.model_config.hf_text_config
        num_indexer_layers = get_num_indexer_layers(hf_text_config)
        index_topk = getattr(hf_text_config, "index_topk", 0)
        set_global_indexer_capturer(
            create_indexer_capturer(
                enable=enable,
                num_indexer_layers=num_indexer_layers,
                index_topk=index_topk,
                num_tokens=self.max_total_num_tokens + self.page_size,
                max_running_requests=self.max_running_requests,
                device=self.device,
            )
        )

    def init_aux_hidden_state_capture(self):
        """Configure auxiliary hidden state capture for speculative decoding.

        Must be called before CUDA graph capture so the captured graphs
        include aux hidden state output paths.
        """
        if self.eagle_use_aux_hidden_state:
            self.model.set_eagle3_layers_to_capture(
                self.eagle_aux_hidden_state_layer_ids
            )
        if self.dflash_family_use_aux_hidden_state:
            if self.spec_algorithm.is_dspark() and hasattr(
                self.model, "set_dspark_layers_to_capture"
            ):
                self.model.set_dspark_layers_to_capture(
                    self.dflash_family_target_layer_ids
                )
            elif hasattr(self.model, "set_dflash_layers_to_capture"):
                self.model.set_dflash_layers_to_capture(
                    self.dflash_family_target_layer_ids
                )
            else:
                raise ValueError(
                    f"Model {self.model.__class__.__name__} implements neither "
                    "set_dspark_layers_to_capture nor set_dflash_layers_to_capture, "
                    "one of which is required for DFLASH/DSPARK."
                )

    def remote_instance_init_transfer_engine(self):
        try:
            from mooncake.engine import TransferEngine
        except ImportError:
            logger.warning(
                "Please install mooncake for using remote instance transfer engine: pip install mooncake-transfer-engine"
            )
            return
        self.remote_instance_transfer_engine = TransferEngine()
        local_ip = get_local_ip_auto()
        self.remote_instance_transfer_engine.initialize(
            local_ip,
            "P2PHANDSHAKE",
            envs.MOONCAKE_PROTOCOL.get(),
            envs.MOONCAKE_DEVICE.get(),
        )
        self.remote_instance_transfer_engine_session_id = NetworkAddress(
            local_ip, self.remote_instance_transfer_engine.get_rpc_port()
        ).to_host_port_str()

    def _register_to_engine_info_bootstrap(self):
        """Register transfer engine info with the EngineInfoBootstrapServer via HTTP PUT.

        The bootstrap server runs on node_rank==0. For multi-node setups, the
        host is derived from dist_init_addr. For single-node, use 127.0.0.1.
        """
        import requests as http_requests

        if self.server_args.dist_init_addr:
            # Multi-node: bootstrap server is on the head node (node_rank==0).
            # Derive host from dist_init_addr (shared across all nodes).
            bootstrap_host = (
                NetworkAddress.parse(self.server_args.dist_init_addr).resolved().host
            )
        else:
            bootstrap_host = "127.0.0.1"

        bootstrap_port = self.server_args.engine_info_bootstrap_port
        bootstrap_na = NetworkAddress(bootstrap_host, bootstrap_port)
        url = f"{bootstrap_na.to_url()}/register_transfer_engine_info"

        payload = {
            "tp_rank": self.tp_rank,
            "transfer_engine_info": {
                "session_id": self.remote_instance_transfer_engine_session_id,
                "weights_info_dict": self.remote_instance_transfer_engine_weight_info,
            },
        }

        try:
            resp = http_requests.put(url, json=payload, timeout=5)
            if resp.status_code == 200:
                logger.info(
                    f"Registered transfer engine info for tp_rank={self.tp_rank} "
                    f"with bootstrap server at {bootstrap_na}"
                )
            else:
                logger.error(
                    f"Failed to register transfer engine info for tp_rank={self.tp_rank}: "
                    f"{resp.status_code}, {resp.text}"
                )
        except Exception as e:
            logger.error(
                f"Failed to register transfer engine info for tp_rank={self.tp_rank}: {e}"
            )

    def check_quantized_moe_compatibility(self):
        if (
            quantization_config := getattr(
                self.model_config.hf_config, "quantization_config", None
            )
        ) is not None and (
            weight_block_size := quantization_config.get("weight_block_size", None)
        ) is not None:
            weight_block_size_n = weight_block_size[0]

            if self.tp_size % self.moe_ep_size != 0:
                raise ValueError(
                    f"tp_size {self.tp_size} must be divisible by ep_size {self.moe_ep_size}"
                )
            moe_tp_size = self.tp_size // self.moe_ep_size // self.moe_dp_size

            moe_intermediate_size = getattr(
                self.model_config.hf_text_config, "moe_intermediate_size", None
            )
            if moe_intermediate_size is None:
                return

            from sglang.srt.distributed.utils import (
                get_tp_partition_ratios,
                partition_sizes,
                tp_plan_active,
            )

            _plan = get_tp_partition_ratios()
            if tp_plan_active(moe_tp_size) and _plan and len(_plan) == moe_tp_size:
                # Uneven TP: the expert intermediate is partitioned in
                # quant-block units by the shard plan — validate that
                # every rank's share is a whole multiple of the quant
                # block instead of requiring even divisibility.
                _units = moe_intermediate_size
                if weight_block_size_n and (
                    moe_intermediate_size % weight_block_size_n == 0
                ):
                    _units = moe_intermediate_size // weight_block_size_n
                partition_sizes(moe_intermediate_size, _plan, _units)
            elif moe_intermediate_size % moe_tp_size != 0:
                raise ValueError(
                    f"moe_intermediate_size {moe_intermediate_size} must be divisible by moe_tp_size ({moe_tp_size}) which is tp_size ({self.tp_size}) divided by moe_ep_size ({self.moe_ep_size})."
                )

            if (
                not envs.SGLANG_SHARED_EXPERT_TP1.get()
                and not (
                    tp_plan_active(moe_tp_size) and _plan and len(_plan) == moe_tp_size
                )
                and (moe_intermediate_size // moe_tp_size) % weight_block_size_n != 0
                and not _use_aiter
            ):
                raise ValueError(
                    f"For quantized MoE models, please make sure ({moe_intermediate_size=} / {moe_tp_size=}) % {weight_block_size_n=} == 0 "
                    f"where moe_tp_size is equal to tp_size ({self.tp_size}) divided by ep_size ({self.moe_ep_size}). "
                    f"You can fix this by setting arguments `--tp` and `--ep` correctly."
                )

    def init_torch_distributed(self):
        tic = time.perf_counter()
        logger.info("Init torch distributed begin.")

        try:
            torch.get_device_module(self.device).set_device(self.gpu_id)
        except Exception:
            logger.warning(
                f"Context: {self.device=} {self.gpu_id=} {os.environ.get('CUDA_VISIBLE_DEVICES')=} {self.tp_rank=} {self.tp_size=}"
            )
            raise

        # Mixed-architecture rigs (bug #208): nvidia-cutlass-dsl picks its
        # compile target ONCE per process from driver device 0, not from the
        # device this rank just selected, and that same string keys its JIT
        # cache. Without --rank-gpu-id every rank sees every GPU, so a rank
        # not owning device 0 compiles (and caches) for the wrong arch and
        # dies in the first CuTe kernel -- e.g. flashinfer rmsnorm_cute --
        # with cudaErrorNoKernelImageForDevice. Retarget it to this rank's
        # own device; a no-op when all cards share one architecture.
        if self.device == "cuda":
            from sglang.srt.utils.cute_dsl_arch import align_cute_dsl_arch

            arch = align_cute_dsl_arch(self.gpu_id)
            if arch is not None:
                logger.debug(
                    "CuTe DSL compile target for TP rank %s (GPU %s): %s",
                    self.tp_rank,
                    self.gpu_id,
                    arch,
                )

        backend = get_default_distributed_backend(self.device)
        if self.device == "cuda" and self.server_args.elastic_ep_backend == "mooncake":
            backend = "mooncake"
            if self.server_args.mooncake_ib_device:
                from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
                    get_ib_devices_for_gpu,
                )

                ib_device_for_gpu = get_ib_devices_for_gpu(
                    self.server_args.mooncake_ib_device, self.gpu_id
                )
                mooncake_ib_device = (
                    ib_device_for_gpu.split(",") if ib_device_for_gpu else []
                )
                try:
                    from mooncake import ep as mooncake_ep

                    mooncake_ep.set_device_filter(mooncake_ib_device)
                except:
                    pass  # A warning will be raised in `init_distributed_environment`

        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        if not self.server_args.enable_p2p_check:
            monkey_patch_p2p_access_check()

        # Allow external orchestrators (e.g. trainpi) to override the distributed
        # init method.  When set to "env://", torch uses MASTER_ADDR/MASTER_PORT
        # env-vars and an externally-created TCPStore, completely avoiding port
        # conflicts with intra-host collocation.
        dist_init_method_override = envs.SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE.get()
        if dist_init_method_override:
            dist_init_method = dist_init_method_override
        elif self.server_args.dist_init_addr:
            na = NetworkAddress.parse(self.server_args.dist_init_addr)
            dist_init_method = na.to_tcp()
        else:
            dist_init_method = NetworkAddress(
                self.server_args.host or "127.0.0.1", self.dist_port
            ).to_tcp()
        set_custom_all_reduce(not self.server_args.disable_custom_all_reduce)
        set_mscclpp_all_reduce(self.server_args.enable_mscclpp)
        set_torch_symm_mem_all_reduce(self.server_args.enable_torch_symm_mem)

        if not self.is_draft_worker:
            if self.device == "cpu":
                if _is_cpu_amx_available or _is_cpu_arm64:
                    # Bind OpenMP threads to CPU cores
                    torch.ops.sgl_kernel.init_cpu_threads_env(self.local_omp_cpuid)

                    # Set local size to hint SGLang to use shared memory based AllReduce
                    os.environ["LOCAL_SIZE"] = str(self.tp_size)
                    torch.ops.sgl_kernel.initialize(self.tp_size, self.tp_rank)

                else:
                    logger.warning(
                        "init_cpu_threads_env and shared memory based AllReduce is disabled, only intel amx backend and arm64 are supported"
                    )

            # #605 NCCL boundary. Everything between this mark and
            # nccl_init_end builds process groups, and communicator buffers
            # are what that costs. The ledger has carried TERM_NCCL_BUFFERS
            # since #595 with no boundary to measure it against, so the term
            # read UNMEASURED on every boot ever recorded. No-op unless
            # SGLANG_VRAM_FLIGHT_DIR is set.
            from sglang.srt.mem_ledger import flight_recorder as _flight_recorder

            _flight_recorder.mark(
                "nccl_init_begin",
                rank=self.tp_rank,
                extra={"draft_worker": bool(self.is_draft_worker)},
            )

            # Only initialize the distributed environment on the target model worker.
            init_distributed_environment(
                backend=backend,
                world_size=self.tp_size * self.pp_size,
                rank=self.tp_size * self.pp_rank + self.tp_rank,
                local_rank=self.gpu_id,
                distributed_init_method=dist_init_method,
                timeout=self.server_args.dist_timeout,
                moe_a2a_backend=self.server_args.moe_a2a_backend,
                recovered_rank=self.server_args.elastic_ep_rejoin,
            )
            initialize_model_parallel(
                tensor_model_parallel_size=self.tp_size,
                attention_data_parallel_size=self.dp_size,
                pipeline_model_parallel_size=self.pp_size,
                expert_model_parallel_size=self.moe_ep_size,
                attention_context_model_parallel_size=self.attn_cp_size,
                moe_data_model_parallel_size=self.moe_dp_size,
                decode_context_parallel_size=self.dcp_size,
                duplicate_tp_group=self.server_args.enable_pdmux,
                enable_symm_mem=self.server_args.enable_symm_mem,
                recovered_rank=self.server_args.elastic_ep_rejoin,
            )
            initialize_dp_attention(
                server_args=self.server_args,
                model_config=self.model_config,
            )
            # #631 Route A: build the SECONDARY (flip-target) group set NOW,
            # right after the primary topology and before ANY attention
            # backend constructor caches dcp state (the attn_dcp_size
            # silent-1 hazard, DESIGN_631 3.2). Eager on every rank -- the
            # create is a collective; the flag is env-uniform, so reaching
            # this line is rank-uniform. The verify-then-create manifest
            # inside dies loudly on any divergence before creating anything.
            if self.server_args.enable_phase_flip:
                from sglang.srt.distributed.parallel_state import (
                    initialize_phase_flip_secondary_groups,
                    phase_flip_groups_initialized,
                )

                if not phase_flip_groups_initialized():
                    flip_vec = [
                        int(x) for x in self.server_args.phase_flip_tp_vector.split(",")
                    ]
                    initialize_phase_flip_secondary_groups(
                        tp_size=len(flip_vec),
                        pp_size=1,
                        dcp_size=len(flip_vec),
                    )
            # #605: placed AFTER the phase-flip secondary groups, not after
            # initialize_model_parallel. Those groups are communicators too,
            # and a boundary drawn before them would price a launch's NCCL
            # buffers at the fraction the primary topology happens to build.
            _flight_recorder.mark(
                "nccl_init_end",
                rank=self.tp_rank,
                extra={"draft_worker": bool(self.is_draft_worker)},
            )

            if is_npu():
                register_sgl_tp_rank(self.gpu_id)

            # Pre-warm NCCL/RCCL/HCCL to eliminate cold-start latency in first request
            # Controlled by --pre-warm-nccl flag (default: enabled on AMD GPUs)
            if self.server_args.pre_warm_nccl and (
                self.tp_size > 1 or self.pp_size > 1 or self.moe_ep_size > 1
            ):
                warmup_start = time.perf_counter()
                tp_group_handle = get_tp_group().device_group

                # Single warmup all_reduce to initialize NCCL/RCCL/HCCL communicator
                warmup_tensor = torch.zeros(1, device=torch.cuda.current_device())
                dist.all_reduce(warmup_tensor, group=tp_group_handle)
                current_platform.synchronize()

                warmup_elapsed = time.perf_counter() - warmup_start
                logger.info(
                    f"NCCL/RCCL/HCCL warmup completed in {warmup_elapsed:.3f}s "
                    f"(tp_size={self.tp_size}, pp_size={self.pp_size}, ep_size={self.moe_ep_size})"
                )

        # --rank-gpu-memory-mib: budgets are deliberately per-rank, so the
        # free-byte measurement must stay LOCAL (no min-reduce across
        # ranks); consistency across ranks is restored later by
        # min-reducing the derived token capacities.
        uneven_memory = self.server_args.uneven_memory_budgets_active()
        pre_model_load_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            # A dual-group lane runner is rank-local: the other serving ranks
            # are not constructing one, so the distributed min-reduce would
            # hang the group (rank-local-before-collective rule).
            distributed=get_world_group().world_size > 1
            and not uneven_memory
            and not self.is_dual_group_lane,
            cpu_group=get_world_group().cpu_group,
        )
        self.tp_group = get_tp_group()
        self.pp_group = get_pp_group()
        self.attention_tp_group = get_parallel().attn_tp_group

        # Check memory for tensor parallelism (definitionally unbalanced
        # under per-rank memory budgets — skip in that mode).
        local_gpu_memory = get_available_gpu_memory(self.device, self.gpu_id)
        if self.tp_size > 1 and not self.is_draft_worker and not uneven_memory:
            if pre_model_load_memory < local_gpu_memory * 0.9:
                msg = "The memory capacity is unbalanced. Some GPUs may be occupied by other processes. "
                msg += f"{pre_model_load_memory=}, {local_gpu_memory=}, {local_gpu_memory * 0.9=}"
                if envs.SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK.get():
                    raise RuntimeError(msg)
                else:
                    logger.warning(msg)

        # #583: agree the all-reduce FUSION ARCH term across the TP group,
        # once, here -- every rank reaches this line exactly once, the TP
        # group exists, and no CUDA graph is capturing. Deciding it per rank
        # would let a mixed group fuse on some ranks only, which removes a
        # collective from those ranks' layers and desyncs the group (silently
        # first, see decide_ar_fusion_arch).
        from sglang.srt.layers.communicator import decide_ar_fusion_arch

        decide_ar_fusion_arch(self.tp_group)

        logger.info(
            f"Init torch distributed ends. elapsed={time.perf_counter() - tic:.2f} s, "
            f"mem usage={(before_avail_memory - local_gpu_memory):.2f} GB"
        )
        return pre_model_load_memory

    def init_shared_mooncake_transfer_engine(self):
        """
        Need MooncakeTransferEngine when:
        1) PD disaggregation uses mooncake for KV transfer (prefill/decode)
        2) HiCache uses mooncake storage backend
        3) Encoder disaggregation uses mooncake
        """
        use_mooncake_te = (
            (
                self.server_args.disaggregation_mode != "null"
                and self.server_args.disaggregation_transfer_backend == "mooncake"
            )
            or (
                self.server_args.enable_hierarchical_cache
                and self.server_args.hicache_storage_backend == "mooncake"
                and envs.SGLANG_HICACHE_MOONCAKE_REUSE_TE.get()
            )
            or (
                self.server_args.encoder_only
                and self.server_args.encoder_transfer_backend == "mooncake"
            )
            or (
                self.server_args.language_only
                and self.server_args.encoder_transfer_backend == "mooncake"
            )
            or (
                self.server_args.enable_elastic_expert_backup
                and self.server_args.elastic_ep_backend is not None
            )
        )

        if use_mooncake_te:
            from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
                init_mooncake_transfer_engine,
            )

            init_mooncake_transfer_engine(
                hostname=get_local_ip_auto(),
                gpu_id=self.gpu_id,
                ib_device=(
                    self.server_args.disaggregation_ib_device
                    or self.server_args.mooncake_ib_device
                ),
            )

    def _install_moe_expert_offload_eager(self):
        """Variant-C B2b: eagerly install the per-expert MoE offload cache on
        every FusedMoE layer right after weight load, so the resident experts
        are on GPU before KV/mamba pool sizing. Gated: no-op unless the layer
        has _moe_offload_enabled (SGLANG_MOE_RESIDENT_EXPERT_FRACTION<1.0). Each
        layer self-guards and falls back to fully-resident on any failure."""
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        n = 0
        for module in self.model.modules():
            if (
                isinstance(module, FusedMoE)
                and getattr(module, "_moe_offload_enabled", False)
                and getattr(module, "_expert_offload", None) is None
                and not getattr(module, "_expert_offload_install_failed", False)
            ):
                module._install_expert_offload()
                if module._expert_offload is not None:
                    n += 1
        if n:
            logger.info(
                "Variant-C B2b: eager MoE expert-offload installed on %d FusedMoE "
                "layers (resident experts materialized before pool sizing).",
                n,
            )

    def _build_weightless_worker_meta_model(self):
        """B2a: build a weightless KV worker's model on the `meta` device with
        NO weight materialization (the VRAM win).

        A weightless worker rank holds ONLY a DCP token-shard of the KV cache;
        it never runs embed / GDN / FFN / proj / lm_head / sample. Its forward
        (``_forward_weightless_worker``) loops the model's full-attention layer
        ids and issues only the dcp-group attention-dispatch collectives, so the
        model's weight Parameters are never read. We therefore build the module
        graph on ``meta`` (0-byte weights) and skip weight loading entirely; the
        real memory the worker needs -- the KV pool + flashinfer attention
        backend -- is built later from ``model_config`` (not from weights).

        Mirrors the meta-build in ``LayeredModelLoader`` (loader.py): create on
        ``meta`` under the model dtype, do NOT call ``load_weights`` and never
        ``.to(device)``. Runs inside the same weight-TP=1 override context as the
        head build so partition geometry is consistent.

        Also reused by a draft-solo SHADOW rank (--speculative-draft-placement
        solo): its draft model exists only so config-derived reads
        (hidden-state spec, layer counts) keep working; the shadow never runs
        a draft forward and never touches these meta parameters.
        """
        from sglang.srt.model_loader.loader import (
            _get_quantization_config,
            _initialize_model,
            set_default_torch_dtype,
        )

        quant_config = _get_quantization_config(self.model_config, self.load_config)
        with set_default_torch_dtype(self.model_config.dtype):
            with torch.device("meta"):
                model = _initialize_model(
                    self.model_config,
                    self.load_config,
                    quant_config,
                )
        return model

    def load_model(self):
        tic_total = time.perf_counter()
        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Load weight begin. avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        # --- Weightless-KV fast lane, loader interception point (Option-B) ---
        # B2a (ACTIVE): a weightless KV worker rank builds the model on the
        # `meta` device and materializes ONLY the KV pool + attention backend
        # (no layer weights) -- the VRAM win. This is safe because the B1 worker
        # forward (_forward_weightless_worker) never calls any model layer (it
        # loops full_attention_layer_ids issuing only dcp-group attention
        # collectives), so the meta weights are never touched. The meta-build
        # itself happens at the loader call below; here we just log. Weight-
        # touching post-load steps (offloader.post_init, kv-cache scales, RoPE
        # cache pre-expansion) are gated off for the worker further down. The
        # load-time tp-group monitored_barrier at the end of load_model is NOT
        # gated -- the meta-worker still participates. No-op on the default path.
        if self.is_weightless_worker:
            logger.info(
                "Weightless-KV worker rank %d: B2a meta-model -- building on "
                "`meta`, skipping weight materialization (KV-pool-only).",
                self.tp_rank,
            )

        # This can reduce thread conflicts and speed up weight loading.
        if self.device != "cpu":
            torch.set_num_threads(1)
        if self.device == "cuda":
            if _needs_float16_fallback(self.gpu_id):
                logger.info(
                    "Compute capability below sm80. Use float16 due to lack of bfloat16 support."
                )
                from sglang.srt.arg_groups.overrides import (
                    declare_load_time_override,
                )

                declare_load_time_override(
                    "ModelRunner._sm80_dtype_fallback", {"dtype": "float16"}
                )
                # This is the LAST word on the dtype and it lands after
                # ModelConfig.__init__, so the HF config still carries the
                # checkpoint's bfloat16. Writing through ModelConfig.dtype
                # re-pins the HF config(s) (see its setter): consumers that
                # derive their own dtype from the config -- the mamba conv-state
                # cache above all -- must follow the fallback, or a hybrid GDN
                # model writes float16 activations into a bfloat16 cache.
                self.model_config.dtype = torch.float16
                # "sm75 and above" is a statement about NVIDIA parts, so the
                # minor number may only be read in the NVIDIA namespace.
                if (
                    torch.version.hip is None
                    and get_device_capability(self.gpu_id)[1] < 5
                ):
                    raise RuntimeError("SGLang only supports sm75 and above.")

        set_cuda_arch()

        # Prepare the model config
        from sglang.srt.configs.modelopt_config import ModelOptConfig

        modelopt_config = ModelOptConfig(
            quant=self.server_args.modelopt_quant,
            checkpoint_restore_path=self.server_args.modelopt_checkpoint_restore_path,
            checkpoint_save_path=self.server_args.modelopt_checkpoint_save_path,
            export_path=self.server_args.modelopt_export_path,
            quantize_and_serve=self.server_args.quantize_and_serve,
        )

        self.load_config = LoadConfig(
            load_format=self.server_args.load_format,
            download_dir=self.server_args.download_dir,
            model_loader_extra_config=self.server_args.model_loader_extra_config,
            tp_rank=self.tp_rank,
            remote_instance_weight_loader_seed_instance_ip=self.server_args.remote_instance_weight_loader_seed_instance_ip,
            remote_instance_weight_loader_seed_instance_service_port=self.server_args.remote_instance_weight_loader_seed_instance_service_port,
            remote_instance_weight_loader_send_weights_group_ports=self.server_args.remote_instance_weight_loader_send_weights_group_ports,
            remote_instance_weight_loader_backend=self.server_args.remote_instance_weight_loader_backend,
            remote_instance_weight_loader_transfer_engine=self.remote_instance_transfer_engine,
            remote_instance_weight_loader_transfer_engine_session_id=self.remote_instance_transfer_engine_session_id,
            modelexpress_url=self.server_args.modelexpress_url,
            modelexpress_transport=self.server_args.modelexpress_transport,
            modelopt_config=modelopt_config,
            rl_quant_profile=self.server_args.rl_quant_profile,
            draft_model_idx=self.draft_model_idx,
            hibernate_dir=self.server_args.hibernate_dir,
        )
        if self.device == "cpu":
            self.model_config = adjust_config_with_unaligned_cpu_tp(
                self.model_config, self.load_config, self.tp_size
            )

        if (
            self.server_args.load_format == LoadFormat.REMOTE_INSTANCE
            and self.server_args.remote_instance_weight_loader_backend
            == RemoteInstanceWeightLoaderBackend.NCCL
        ):
            if self.tp_rank == 0:
                instance_ip = NetworkAddress.resolve_host(socket.gethostname())
                t = threading.Thread(
                    target=trigger_init_weights_send_group_for_remote_instance_request,
                    args=(
                        self.server_args.remote_instance_weight_loader_seed_instance_ip,
                        self.server_args.remote_instance_weight_loader_seed_instance_service_port,
                        self.server_args.remote_instance_weight_loader_send_weights_group_ports,
                        instance_ip,
                    ),
                )
                t.start()

        # Load the model
        # Remove monkey_patch when linear.py quant remove dependencies with vllm
        monkey_patch_vllm_parallel_state()

        enable_cpu_backup = self.server_args.enable_weights_cpu_backup or (
            self.is_draft_worker and self.server_args.enable_draft_weights_cpu_backup
        )
        with self.memory_saver_adapter.region(
            GPU_MEMORY_TYPE_WEIGHTS,
            enable_cpu_backup=enable_cpu_backup,
        ):
            from sglang.srt.observability.startup_func_log_and_timer import (
                startup_timer,
            )

            with startup_timer("weight_loading"):
                self.loader = get_model_loader(
                    load_config=self.load_config,
                    model_config=self.model_config,
                )
                # Weightless-KV fast lane (Option-B, B1): build + load the model as
                # weight-TP=1 (every family full, no sharding, no FFN/residual
                # all-reduce). The head rank runs this full model TP=1; only the
                # attention op dispatches across the dcp group (size 3) via the
                # [H,0,0] weightless path. Linears cache self.tp_size at construction
                # and RowParallelLinear gates its all-reduce on that cached value
                # (linear.py:2103), so a construction-time override sticks through
                # forward. attn_tp_size is already 1 under dcp==tp, so attention proj
                # is full; the override makes FFN/embed/lm_head/GDN full too. The dcp
                # group is NOT overridden, so the attention dispatch still spans 3
                # ranks. No-op on the default path.
                from sglang.srt.runtime_context import get_parallel

                # Also pin the per-family RANKS to 0: at tp_size==1 the only shard
                # is rank 0, but the real tp_rank is still 1/2 on the workers. Any
                # rank-indexed partition (e.g. the VL vision tower's
                # tp_partition_size(total, tp_size, rank)[rank]) would IndexError on
                # the now length-1 partition list without this. The dcp geometry is
                # set later at attention-backend init (NOT under this override), so
                # it still uses the real rank for the KV token-shard.
                # Draft-solo placement reuses the exact same construction-time
                # override: BOTH the solo host and the shadows build the draft
                # module graph as weight-TP=1 (full families, no sharding, no
                # draft-internal all-reduce -- linears cache tp_size at
                # construction), and rank pinned to 0 so rank-indexed partition
                # lists stay valid. The host then LOADS the full weights; a
                # shadow builds on `meta` (no weights) exactly like a weightless
                # KV worker -- its draft forward is never called.
                _wl_build_ctx = (
                    get_parallel().override(
                        tp_size=1,
                        tp_rank=0,
                        moe_tp_size=1,
                        moe_tp_rank=0,
                        attn_tp_size=1,
                        attn_tp_rank=0,
                    )
                    if (
                        self.is_weightless_head
                        or self.is_weightless_worker
                        or self.is_draft_solo_host
                        or self.is_draft_solo_shadow
                    )
                    else contextlib.nullcontext()
                )
                with _wl_build_ctx:
                    if self.is_dual_group_lane:
                        # Multi-group runtime (#274): the lane's model is not
                        # loaded, it is ASSEMBLED -- complement shards via the
                        # stock loader under the lane's own scoped geometry, a
                        # full-width hull, and shells that replace the lane
                        # group's collectives with local ops. Rank-local.
                        #
                        # The NEXTN head takes the SAME treatment: it is one more
                        # decoder layer in the serving group's geometry, so the
                        # assembly is the assembly -- only the vocab is special
                        # (it points at the lane target's shells).
                        from sglang.srt.model_executor.dual_group_lane import (
                            build_lane_draft_model,
                            build_lane_model,
                        )

                        if self.is_dual_group_lane_draft:
                            self.model = build_lane_draft_model(self)
                        else:
                            self.model = build_lane_model(self)
                    elif self.is_weightless_worker or self.is_draft_solo_shadow:
                        # B2a / draft-solo shadow: meta-model, NO weight load.
                        self.model = self._build_weightless_worker_meta_model()
                    else:
                        self.model = self.loader.load_model(
                            model_config=self.model_config,
                            device_config=DeviceConfig(self.device, self.gpu_id),
                        )
                if hasattr(self.loader, "remote_instance_transfer_engine_weight_info"):
                    self.remote_instance_transfer_engine_weight_info = (
                        self.loader.remote_instance_transfer_engine_weight_info
                    )
        # Cache needs to be cleared after loading model weights (in the self.loader.load_model function).
        # To avoid conflict with memory_saver_adapter.region, empty_cache operation is now moved here.
        if _is_npu:
            torch.npu.empty_cache()
        monkey_patch_vllm_parallel_state(reverse=True)

        if not self.is_draft_worker and not self.is_weightless_worker:
            # B2a: a weightless worker holds no offloadable weights (meta model).
            get_offloader().post_init()

        # Per-expert MoE offload (Variant-C B2b): install the resident/spill
        # cache EAGERLY here, right after load, so the n_slots resident experts
        # are materialized on GPU BEFORE the KV/mamba memory pool is sized. The
        # #77 cache otherwise installs lazily on the first forward, which is
        # AFTER pool sizing -> the pool would over-allocate and the resident
        # experts would OOM. No-op unless SGLANG_MOE_RESIDENT_EXPERT_FRACTION<1.
        # Draft-solo shadow: meta draft holds no expert weights to offload.
        if not self.is_draft_solo_shadow:
            self._install_moe_expert_offload_eager()

        # Register model for layerwise NVTX profiling if enabled
        if self.server_args.enable_layerwise_nvtx_marker:
            pyt_hooks = PytHooks()
            pyt_hooks.register_hooks(self.model, module_prefix="model")

        if (
            self.server_args.kv_cache_dtype == "fp8_e4m3"
            and not self.is_weightless_worker
            # Draft-solo shadow: meta model, no weights to load scales into.
            and not self.is_draft_solo_shadow
        ):
            # B2a: KV-cache scales load into model weights (meta on the worker);
            # the worker's KV pool gets its scales via the head's dispatch path.
            if self.server_args.quantization_param_path is not None:
                if callable(getattr(self.model, "load_kv_cache_scales", None)):
                    self.model.load_kv_cache_scales(
                        self.server_args.quantization_param_path
                    )
                    logger.info(
                        "Loaded KV cache scaling factors from %s",
                        self.server_args.quantization_param_path,
                    )
                else:
                    raise RuntimeError(
                        "Using FP8 KV cache and scaling factors provided but "
                        "model %s does not support loading scaling factors.",
                        self.model.__class__,
                    )
            else:
                logger.warning(
                    "Using FP8 KV cache but no scaling factors "
                    "provided. Defaulting to scaling factors of 1.0. "
                    "This may lead to less accurate results!"
                )

        # Parse other args
        self.sliding_window_size = None
        if hasattr(self.model, "get_attention_sliding_window_size"):
            self.sliding_window_size = self.model.get_attention_sliding_window_size()
        elif (
            self.model_config.is_hybrid_swa
            and self.model_config.sliding_window_size is not None
        ):
            # sliding window field in model config may have different meaning for different kinds of models (e.g., dllm), here we only consider the sliding window in SWA model
            self.sliding_window_size = self.model_config.sliding_window_size
        elif self.model_config.attention_chunk_size is not None:
            self.sliding_window_size = self.model_config.attention_chunk_size
            logger.info(
                f"Setting sliding_window_size to be attention_chunk_size: {self.sliding_window_size}"
            )

        self.prefill_aware_swa = (
            hasattr(self.model, "is_prefill_aware_swa")
            and self.model.is_prefill_aware_swa()
        )

        self.dtype = self.model_config.dtype

        after_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        self.weight_load_mem_usage = before_avail_memory - after_avail_memory
        # Get quantization config from ModelConfig
        # This handles both config.json (standard) and hf_quant_config.json (ModelOpt)
        quant_str = self.model_config.get_quantization_config_log_str()

        logger.info(
            f"Load weight end. "
            f"elapsed={time.perf_counter() - tic_total:.2f} s, "
            f"type={type(self.model).__name__}, "
            f"{quant_str + ', ' if quant_str else ''}"
            f"avail mem={after_avail_memory:.2f} GB, "
            f"mem usage={self.weight_load_mem_usage:.2f} GB."
        )

        # #644: end-of-load host-anon discriminator. Off unless
        # SGLANG_644_DISCRIMINATOR is set; the answer it produces (references
        # genuinely persist vs glibc arenas holding freed pages) cannot be read
        # off an RSS sampler, and gdb is not installed on this rig. Placed here
        # because "after load" is exactly the point the ~16 GB residue was
        # measured at, and because a failure inside a diagnostic must never
        # cost a boot.
        try:
            from sglang.srt.mem_ledger.host_anon_644 import run_discriminator

            run_discriminator(
                self.model,
                tag=f"pp{self.pp_rank}tp{self.tp_rank}"
                + ("-draft" if self.is_draft_worker else ""),
            )
        except Exception as exc:  # noqa: BLE001 -- diagnosis never kills a boot
            logger.debug("#644 discriminator skipped: %s", exc)

        # TODO: Make sure all models have `quant_config` attribute, and all online quantization methods register which layers they actually quantize.
        # TODO: Move this online-quantization reporting out of ModelRunner.
        quantized_layers = getattr(
            getattr(self.model, "quant_config", None), "quantized_layers", None
        )
        if (
            self.server_args.quantization is not None
            and isinstance(quantized_layers, tuple)
            and len(quantized_layers) == 2
        ):
            layer_types, quantized_layers_count = quantized_layers
            logger.info(
                f"Online {self.server_args.quantization} quantization: quantized {quantized_layers_count} layers of types: {layer_types}"
            )

        if self.server_args.debug_tensor_dump_output_folder is not None:
            dump_folder = self.server_args.debug_tensor_dump_output_folder
            if self.spec_algorithm.is_eagle():
                role = "draft" if self.is_draft_worker else "target"
                dump_folder = os.path.join(dump_folder, role)
            register_forward_hook_for_model(
                self.model,
                dump_folder,
                self.server_args.debug_tensor_dump_layers,
                self.tp_size,
                self.tp_rank,
                self.pp_rank,
            )

        if dumper.may_enable:
            dumper.apply_source_patches()
            dumper.register_non_intrusive_dumper(self.model)

        # Pre-expand RoPE cache before CUDA Graph capture.
        # B2a: skip on a weightless worker -- its model layers are meta (RoPE
        # cos/sin caches are meta buffers) and the worker forward never runs
        # RoPE (it only issues the attention dispatch collectives). Same for
        # a draft-solo shadow (meta draft, forward never called).
        if not self.is_weightless_worker and not self.is_draft_solo_shadow:
            reserve_rope_cache_for_long_sequences(
                self.model,
                self.server_args,
                self.model_config,
                logger,
            )

        if self.is_dual_group_lane:
            # Rank-local lane assembly: the other serving ranks are not in a
            # model load, so the group barrier below would hang the group
            # (rank-local-before-collective rule). Nothing to synchronize --
            # the lane exists in this process only.
            pass
        elif self.server_args.elastic_ep_backend == "mooncake":
            # Mooncake does not support `monitored_barrier`
            dist.barrier(group=get_tp_group().cpu_group)
        else:
            # Handle the case where some ranks do not finish loading.
            try:
                dist.monitored_barrier(
                    group=get_tp_group().cpu_group,
                    timeout=datetime.timedelta(
                        seconds=UNBALANCED_MODEL_LOADING_TIMEOUT_S
                    ),
                    wait_all_ranks=True,
                )
            except RuntimeError as e:
                # #386 sibling: `from None` used to drop the barrier's own
                # message, which is the only text that NAMES the ranks that
                # never checked in -- the generic sentence below is a guess
                # ("likely ... OOM or a slow node") and was all the operator
                # got. The verdict stays a ValueError with the same wording;
                # the evidence is appended and the chain reconnected.
                raise ValueError(
                    f"TP rank {self.tp_rank} could finish the model loading, but there are other ranks that didn't finish loading. It is likely due to unexpected failures (e.g., OOM) or a slow node. "
                    f"The barrier reported: {e}"
                ) from e

    def _prepare_moe_topk(self):
        balancer_cls = None
        num_prepared = 0
        num_routed_experts = None
        for module in self.model.modules():
            if not isinstance(module, (TopK, HashTopK)):
                continue
            if (
                not module.enable_deepep_waterfill
                or module.deepep_waterfill_balancer is not None
            ):
                continue
            if num_routed_experts is None:
                num_routed_experts = getattr(
                    self.model_config.hf_config, "n_routed_experts", None
                )
                if num_routed_experts is None:
                    raise ValueError(
                        "DeepEP waterfill requires model config n_routed_experts."
                    )
            if balancer_cls is None:
                from sglang.srt.layers.moe.deepep_waterfill import (
                    DeepEPWaterfillBalancer,
                )

                balancer_cls = DeepEPWaterfillBalancer
            # Static EPLB remaps TopK ids to physical expert ids before Waterfill.
            # Redundant experts therefore need to be included in the per-rank
            # expert count used for Waterfill's shared-expert slot remapping.
            num_physical_routed_experts = (
                num_routed_experts + self.server_args.ep_num_redundant_experts
            )
            if isinstance(module, TopK):
                routed_scaling_factor = module.topk_config.routed_scaling_factor
            else:
                routed_scaling_factor = module.routed_scaling_factor
            module.deepep_waterfill_balancer = balancer_cls(
                num_routed_experts=num_physical_routed_experts,
                world_size=self.moe_ep_size,
                rank=self.moe_ep_rank,
                layer_id=module.layer_id,
                routed_scaling_factor=(
                    routed_scaling_factor if routed_scaling_factor is not None else 1.0
                ),
            )
            num_prepared += 1
        if num_prepared:
            log_info_on_rank0(
                logger, f"Prepared {num_prepared} DeepEP waterfill TopK modules."
            )

    def _init_lplb_solvers(self):
        """Initialize per-layer LPLB solvers from current expert location metadata."""
        from sglang.srt.distributed import get_moe_ep_group

        # Gate: refuse LP for non-DeepSeek MoE families whose empty-token paths
        # don't participate in the EP all-reduce (would deadlock under DP-
        # attention). Failure here happens before any forward pass.
        architectures = getattr(self.model_config.hf_config, "architectures", None)
        if architectures:
            assert_lplb_supported_model(architectures[0])

        metadata = get_global_expert_location_metadata()
        if metadata is None:
            return
        clear_global_lplb_solvers()
        ep_group = get_moe_ep_group()
        for lid in range(metadata.num_layers):
            solver = LPLBSolver(
                phy2log=metadata.physical_to_logical_map[lid],
                log2phy=metadata.logical_to_all_physical_map[lid],
                num_gpus=metadata.ep_size,
                ep_group=ep_group,
                logical_to_all_physical_map_num_valid=(
                    metadata.logical_to_all_physical_map_num_valid[lid]
                ),
            )
            set_global_lplb_solver(lid, solver)
        logger.info(f"Initialized LPLB solvers for {metadata.num_layers} layers")

    def update_expert_location(
        self,
        new_expert_location_metadata: ExpertLocationMetadata,
        update_layer_ids: List[int],
    ):
        p2p_missing_logical_experts = self.expert_location_updater.update(
            self.model.routed_experts_weights_of_layer,
            new_expert_location_metadata,
            update_layer_ids=update_layer_ids,
            nnodes=self.server_args.nnodes,
            rank=self.tp_rank,
        )

        if len(p2p_missing_logical_experts) > 0:
            # Load the missing expert weights from disk
            if callable(getattr(self.model, "generate_weight_name_filter", None)):
                # Filter and load only missing expert weights
                weight_name_filter = self.model.generate_weight_name_filter(
                    p2p_missing_logical_experts
                )
            else:
                # Do a full reload from disk/DRAM
                logger.info(
                    "[Elastic EP] Model does not implement generate_weight_name_filter. "
                    "Performing full weight reload."
                )
                weight_name_filter = None

            if (
                self.expert_backup_client is not None
                and self.expert_backup_client.use_backup
            ):
                # Load the missing weights from the DRAM backup
                self.expert_backup_client.update_weights(weight_name_filter)
            else:
                # Load the missing weights from disk
                self.update_weights_from_disk(
                    get_server_args().model_path,
                    get_server_args().load_format,
                    weight_name_filter=weight_name_filter,
                )

        # Re-init LPLB solvers after expert location update
        if self.server_args.ep_dispatch_algorithm == "lp":
            self._init_lplb_solvers()

    def maybe_recover_ep_ranks(self):
        # TODO(perf): `active_ranks.all()` on a CUDA tensor triggers host-device
        # synchronization, and this function is on the forward-path.
        # This check only runs when `--elastic-ep-backend` is enabled, so the
        # synchronization overhead does not propagate to other configs.
        # Leave for future optimization of the elastic EP path.
        if self.tp_group.active_ranks.all() and self.tp_group.active_ranks_cpu.all():
            return

        tp_active_ranks = self.tp_group.active_ranks.detach().cpu().numpy()
        tp_active_ranks_cpu = self.tp_group.active_ranks_cpu.detach().numpy()
        tp_active_ranks &= tp_active_ranks_cpu
        # NOTE: `ranks_to_recover` uses indices in `tp_group`. For the current
        # Mooncake elastic EP implementation we assume `--pp-size=1`, so the
        # tp-group index is the same as the global rank index.
        ranks_to_recover = [
            i for i in range(len(tp_active_ranks)) if not tp_active_ranks[i]
        ]

        # try_recover_ranks polls peer state via Mooncake EP backend.
        # Mooncake's internal semantics guarantee that all ranks observe
        # consistent peer readiness state, so collective operations below
        # are safe even though polling appears local.
        if ranks_to_recover and try_recover_ranks(ranks_to_recover):
            self.forward_pass_id = 0
            self.eplb_manager.reset_generator()
            broadcast_global_expert_location_metadata(
                src_rank=self._get_healthy_expert_location_src_rank(
                    invoked_in_elastic_ep_rejoin_path=False
                )
            )
            ElasticEPStateManager.instance().reset()

            broadcast_pyobj(
                [self.server_args.random_seed],
                get_world_group().rank,
                get_world_group().cpu_group,
                src=get_world_group().ranks[0],
            )
            logger.info(f"recover ranks {ranks_to_recover} done")

    def _get_healthy_expert_location_src_rank(
        self, invoked_in_elastic_ep_rejoin_path: bool
    ) -> int:
        world_group = get_world_group()
        # NOTE: do not key off `self.server_args.elastic_ep_rejoin` here.
        # A rank that was started as a rejoin rank may later act as a healthy
        # rank in a subsequent recovery cycle.
        local_rejoin_flag = bool(invoked_in_elastic_ep_rejoin_path)
        gathered_rejoin_flags = world_group.all_gather_object(local_rejoin_flag)

        for rank_in_group, is_rejoin_rank in enumerate(gathered_rejoin_flags):
            if not is_rejoin_rank:
                return world_group.ranks[rank_in_group]

        raise RuntimeError(
            "No healthy rank found for broadcasting expert location metadata. "
            "All ranks are marked as elastic_ep_rejoin."
        )

    def update_weights_from_disk(
        self,
        model_path: str,
        load_format: str,
        weight_name_filter: Optional[Callable[[str], bool]] = None,
        recapture_cuda_graph: bool = False,
    ) -> tuple[bool, str]:
        """Update engine weights in-place from the disk."""
        logger.info(
            f"Update engine weights online from disk begin. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id, empty_cache=False):.2f} GB"
        )

        target_device = torch.device(self.device)
        self.model_config.model_path = model_path
        load_config = LoadConfig(load_format=load_format)

        # Only support DefaultModelLoader for now
        loader = get_model_loader(load_config, self.model_config)
        if not isinstance(loader, DefaultModelLoader):
            message = f"Failed to get model loader: {loader}."
            return False, message

        def get_weight_iter(config):
            iter = loader._get_weights_iterator(
                DefaultModelLoader.Source.init_new(config, self.model)
            )
            if weight_name_filter is not None:
                iter = (
                    (name, weight) for name, weight in iter if weight_name_filter(name)
                )

            return iter

        def model_load_weights(model, iter):
            loader.load_weights_and_postprocess(model, iter, target_device)
            return model

        with set_default_torch_dtype(self.model_config.dtype):
            try:
                iter = get_weight_iter(self.model_config)
            except Exception as e:
                message = f"Failed to get weights iterator: {e}."
                return False, message
            try:
                model = model_load_weights(self.model, iter)
            except Exception as e:
                message = (
                    f"Failed to update weights: {e}.\nRolling back to original weights."
                )
                del iter
                gc.collect()
                iter = get_weight_iter(self.model_config)
                self.model = model_load_weights(self.model, iter)
                return False, message

        self.model = model
        self.server_args.override(
            "model_runner.update_weights",
            model_path=model_path,
            load_format=load_format,
        )
        self.load_config = load_config

        if recapture_cuda_graph and (
            self.device == "cuda"
            or self.device == "musa"
            or (
                current_platform.is_out_of_tree()
                and current_platform.support_cuda_graph()
            )
        ):
            self.init_decode_cuda_graph()

        logger.info("Update weights end.")
        return True, "Succeeded to update model weights."

    def init_weights_send_group_for_remote_instance(
        self,
        master_address,
        ports,
        group_rank,
        world_size,
        group_name,
        backend="nccl",
    ):
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        ports_list = ports.split(",")
        assert (
            len(ports_list) == self.tp_size
        ), f"Expected {self.tp_size} ports, but got {len(ports_list)} ports."
        group_port = ports_list[self.tp_rank]
        group_name = f"{group_name}_{group_port}_{self.tp_rank}"

        logger.info(
            f"init custom process group: tp_rank={self.tp_rank}, gpu_id={self.gpu_id}, master_address={master_address}, master_port={group_port}, "
            f"group_rank={group_rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        current_platform.empty_cache()
        success = False
        message = ""
        try:
            na = NetworkAddress(master_address, group_port)
            self._weights_send_group[group_name] = init_custom_process_group(
                backend=backend,
                init_method=na.to_tcp(),
                world_size=world_size,
                rank=group_rank,
                group_name=group_name,
                device_id=torch.device("cuda", self.gpu_id),
            )
            dist.barrier(group=self._weights_send_group[group_name])
            success = True
            message = f"Succeeded to init group through {na.to_host_port_str()} group."
        except Exception as e:
            message = f"Failed to init group: {e}."
            logger.error(message)

        current_platform.empty_cache()
        return success, message

    def send_weights_to_remote_instance(
        self,
        master_address,
        ports,
        group_name,
    ):
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        ports_list = ports.split(",")
        assert (
            len(ports_list) == self.tp_size
        ), f"Expected {self.tp_size} ports, but got {len(ports_list)} ports."
        group_port = ports_list[self.tp_rank]
        group_name = f"{group_name}_{group_port}_{self.tp_rank}"

        if self._weights_send_group[group_name] is not None:
            send_group = self._weights_send_group[group_name]
        else:
            message = f"Group {group_name} not in _weights_send_group list. Please call `init_weights_send_group_for_remote_instance` first."
            logger.error(message)
            return False, message

        current_platform.empty_cache()
        success = False
        na = NetworkAddress(master_address, group_port)
        message = ""
        try:
            for _, weights in self.model.named_parameters():
                torch.distributed.broadcast(
                    weights,
                    src=0,
                    group=send_group,
                )
            success = True
            message = f"Succeeded to send weights through {na.to_host_port_str()} {group_name}."
        except Exception as e:
            message = f"Failed to send weights: {e}."
            logger.error(message)

        # destroy the process group after sending weights
        del self._weights_send_group[group_name]
        torch.distributed.distributed_c10d.destroy_process_group(send_group)
        current_platform.empty_cache()
        return success, message

    def init_weights_update_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
    ):
        """Initialize the Torch process group for model parameter updates.

        `_model_update_group` is used in the RLHF workflow, where rank
        0 is the actor model in the training engine, and the other ranks are
        the inference engine, which is used for rollout.

        In the RLHF workflow, the training engine updates the model
        weights/parameters online, and broadcasts them to the inference
        engine through the `_model_update_group` process group.
        """
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        rank = rank_offset + self.tp_rank

        logger.info(
            f"init custom process group: master_address={master_address}, master_port={master_port}, "
            f"rank_offset={rank_offset}, rank={rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        try:
            na = NetworkAddress(master_address, master_port)
            self._model_update_group[group_name] = init_custom_process_group(
                backend=backend,
                init_method=na.to_tcp(),
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )
            return True, "Succeeded to initialize custom process group."
        except Exception as e:
            message = f"Failed to initialize custom process group: {e}."
            logger.error(message)
            return False, message

    def destroy_weights_update_group(self, group_name):
        try:
            if group_name in self._model_update_group:
                pg = self._model_update_group.pop(group_name)
                torch.distributed.destroy_process_group(pg)
                return True, "Succeeded to destroy custom process group."
            else:
                return False, "The group to be destroyed does not exist."
        except Exception as e:
            message = f"Failed to destroy custom process group: {e}."
            logger.error(message)
            return False, message

    def update_weights_from_distributed(
        self,
        names,
        dtypes,
        shapes,
        group_name,
        load_format: Optional[str] = None,
    ):
        """
        Update specific parameter in the model weights online
        through `_model_update_group` process group.

        Args:
            name: the name of the parameter to be updated.
            dtype: the data type of the parameter to be updated.
            shape: the shape of the parameter to be updated.
        """

        assert group_name in self._model_update_group, (
            f"Group {group_name} not in {list(self._model_update_group.keys())}. "
            "Please call `init_weights_update_group` first."
        )

        if load_format == "flattened_bucket":
            return self._update_bucketed_weights_from_distributed(
                names, dtypes, shapes, group_name
            )
        try:
            weights = []
            handles = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                weight = torch.empty(shape, dtype=target_dtype, device=self.device)
                handles.append(
                    torch.distributed.broadcast(
                        weight,
                        src=0,
                        group=self._model_update_group[group_name],
                        async_op=True,
                    )
                )
                weights.append((name, weight))
            for handle in handles:
                handle.wait()

            self.model.load_weights(weights)
            return True, "Succeeded to update parameter online."

        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    def _update_bucketed_weights_from_distributed(
        self, names, dtypes, shapes, group_name
    ):
        try:
            named_tensors = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                named_tensors.append(
                    (name, torch.empty(shape, dtype=target_dtype, device=self.device))
                )
            bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            flattened_tensor = bucket.get_flattened_tensor()
            torch.distributed.broadcast(
                flattened_tensor,
                src=0,
                group=self._model_update_group[group_name],
            )
            reconstructed_tensors = bucket.reconstruct_tensors()
            self.model.load_weights(reconstructed_tensors)
            return True, f"Succeeded to update parameter online."
        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, Union[torch.Tensor, LocalSerializedTensor]]],
        load_format: Optional[str] = None,
    ):
        monkey_patch_torch_reductions()
        if load_format == "flattened_bucket":
            # Handle flattened bucket format
            return self._update_weights_from_flattened_bucket(
                flattened_tensor_bucket_dict=named_tensors
            )

        # We need to get device after patch otherwise the device would be wrong
        device_module = torch.get_device_module(self.device)
        infered_device = device_module.current_device()

        named_tensors = [
            (name, _unwrap_tensor(tensor, tp_rank=self.tp_rank, device=infered_device))
            for name, tensor in named_tensors
        ]
        if load_format == "direct":
            _model_load_weights_direct(self.model, named_tensors)
        elif load_format in self.server_args.custom_weight_loader:
            custom_loader = dynamic_import(load_format)
            custom_loader(self.model, named_tensors)
        elif load_format is None:
            self.model.load_weights(named_tensors)
        else:
            raise NotImplementedError(f"Unknown load_format={load_format}")
        return True, "Success"

    def _update_weights_from_flattened_bucket(
        self,
        flattened_tensor_bucket_dict,
    ):
        """Handle flattened bucket format for weight updates"""
        flattened_tensor = flattened_tensor_bucket_dict["flattened_tensor"]
        metadata = flattened_tensor_bucket_dict["metadata"]

        # Convert metadata dict to our format
        converted_metadata = []
        for meta in metadata:
            converted_meta = FlattenedTensorMetadata(
                name=meta.name,
                shape=meta.shape,
                dtype=meta.dtype,
                start_idx=meta.start_idx,
                end_idx=meta.end_idx,
                numel=meta.numel,
            )
            converted_metadata.append(converted_meta)

        # Create bucket and reconstruct tensors
        bucket = FlattenedTensorBucket(
            flattened_tensor=flattened_tensor, metadata=converted_metadata
        )
        reconstructed_tensors = bucket.reconstruct_tensors()

        # Load the reconstructed tensors using the standard method
        self.model.load_weights(reconstructed_tensors)

        return True, "Success"

    def get_weights_by_name(
        self, name: str, truncate_size: int = 100
    ) -> Optional[torch.Tensor]:
        """Get the weights of the parameter by its name. Similar to `get_parameter` in Hugging Face.

        Only used for unit test with an unoptimized performance.
        For optimized performance, please use torch.save and torch.load.
        """
        # TODO: (chenyang) Add support for Qwen models.
        try:
            return self.model.get_weights_by_name(
                name, truncate_size, tp_size=self.tp_size
            )
        except Exception as e:
            logger.error(f"Error when getting parameter {name}: {e}")
            return None

    def init_lora_manager(self):
        self.lora_manager = LoRAManager(
            base_model=self.model,
            base_hf_config=self.model_config.hf_config,
            max_loras_per_batch=self.server_args.max_loras_per_batch,
            load_config=self.load_config,
            dtype=self.dtype,
            server_args=self.server_args,
            lora_backend=self.server_args.lora_backend,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            max_lora_rank=self.server_args.max_lora_rank,
            target_modules=self.server_args.lora_target_modules,
            lora_paths=self.server_args.lora_paths,
        )

    def _init_lora_cuda_graph_moe_buffers(self):
        """Phase 1 of LoRA CUDA graph init: pre-allocate MoE intermediate buffers.

        Must be called before init_memory_pool() so that memory profiling
        sees the reduced available memory and sizes KV cache correctly.
        All MoE LoRA layers share one set of buffers (managed by the
        lora_backend) since they execute sequentially during forward.

        Phase 2 (dense LoRA batch metadata) is handled later in
        CudaGraphRunner.__init__() via lora_manager.init_cuda_graph_batch_info(),
        because it needs capture-time parameters (max_bs, num_tokens_per_bs)
        that are only available at that stage.
        """
        from sglang.srt.lora.layers import FusedMoEWithLoRA

        max_bs = self.server_args.cuda_graph_config.decode.max_bs
        max_loras = self.server_args.max_loras_per_batch
        for module in self.model.modules():
            if isinstance(module, FusedMoEWithLoRA):
                self.lora_manager.init_cuda_graph_moe_buffers(
                    max_bs, max_loras, self.dtype, module
                )
                logger.info(
                    f"Pre-allocated shared MoE LoRA CUDA graph buffers "
                    f"(max_bs={max_bs}, max_loras={max_loras})"
                )
                break

    def load_lora_adapter(self, lora_ref: LoRARef):
        """Load a new lora adapter from disk or huggingface."""

        logger.info(
            f"LoRA adapter loading starts: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        result = self.lora_manager.load_lora_adapter(lora_ref)

        logger.info(
            f"LoRA adapter loading completes: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        return result

    def load_lora_adapter_from_tensors(
        self, lora_ref: LoRARef, tensors, config_dict, added_tokens_config=None
    ):
        logger.info(f"LoRA adapter loading from tensors starts: {lora_ref}.")
        result = self.lora_manager.load_lora_adapter_from_tensors(
            lora_ref, tensors, config_dict, added_tokens_config
        )
        logger.info(f"LoRA adapter loading from tensors completes: {lora_ref}.")
        return result

    def unload_lora_adapter(self, lora_ref: LoRARef):
        """Unload a lora adapter that was previously loaded during initialization or dynamic loading."""

        logger.info(
            f"LoRA adapter unloading starts: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        result = self.lora_manager.unload_lora_adapter(lora_ref)

        logger.info(
            f"LoRA adapter unloading completes: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        return result

    @property
    def is_dual_group_lane_target(self) -> bool:
        """True only for the lane's TARGET runner (#274).

        Both lane runners carry ``is_draft_worker=True`` for the
        secondary-runner gates, but they want OPPOSITE answers wherever the
        code asks "is this a draft model?":

        * the lane TARGET is a draft worker in name only -- it runs the full
          model and must keep the real per-layer routing, KV pool and graphs;
        * the lane HEAD is a draft model in the ordinary sense -- one MTP
          layer, full attention, one layer's KV pool.

        The exemptions were first written as ``is_draft_worker and not
        is_dual_group_lane``, which is true of the target AND the head, so the
        head silently inherited the target's 64-layer geometry. That produced
        two boot failures in a row from one root: its full-attention call
        dispatched into the GDN backend, and then its KV pool had an empty
        full-attention layer map. Asking this property instead of respelling
        the condition keeps the sites from drifting apart again.
        """
        return self.is_dual_group_lane and not self.is_dual_group_lane_draft

    @property
    def qwen3_next_config(self):
        config = self.model_config.hf_config
        if isinstance(config, Qwen3NextConfig):
            return config
        return None

    @property
    def hybrid_lightning_config(self):
        config = self.model_config.hf_config
        if isinstance(config, BailingHybridConfig):
            return config
        return None

    @property
    def hybrid_gdn_config(self):
        config = self.model_config.hf_config.get_text_config()
        if isinstance(
            config,
            Qwen3NextConfig
            | Qwen3_5Config
            | Qwen3_5MoeConfig
            | InternS2PreviewConfig
            | JetNemotronConfig
            | JetVLMConfig,
        ):
            return config
        return None

    @property
    def mamba2_config(self):
        config = self.model_config.hf_config
        if isinstance(config, NemotronHConfig) and self.is_draft_worker:
            # NemotronH MTP draft models have no Mamba layers (pattern like "*E")
            # so they shouldn't use HybridLinearAttnBackend
            pattern = getattr(config, "mtp_hybrid_override_pattern", None)
            if pattern is not None and "M" not in pattern:
                return None
        if isinstance(
            config,
            FalconH1Config
            | NemotronHConfig
            | Lfm2Config
            | Lfm2MoeConfig
            | Lfm2VlConfig
            | ZayaConfig,
        ):
            return config
        if isinstance(config, NemotronH_Nano_VL_V2_Config):
            return config.llm_config

        if isinstance(config, GraniteMoeHybridConfig):
            has_mamba = any(
                layer_type == "mamba"
                for layer_type in getattr(config, "layer_types", [])
            )
            if not has_mamba:
                return None
            else:
                return config

        return None

    @property
    def max_token_pool_size(self):
        """Return the max token pool size considering hybrid swa settings."""
        # Weightless-KV host spill (B2): a single sequence is DCP-token-sharded,
        # so it can span the GLOBAL tiered capacity (per-rank pool x dcp_size),
        # not just this rank's pool. Expose that so max_req_len admits an
        # over-VRAM sequence up to context_len (see the per-rank pool shrink in
        # _init_pools). Only set on the spill lane; None/absent everywhere else.
        _wl_global = getattr(self, "_wl_spill_global_capacity", 0)
        if _wl_global:
            # Already the global span (per-rank pool x dcp_size), computed in
            # _init_pools; do not apply the lane factor a second time.
            return _wl_global
        if self.is_hybrid_swa:
            pool_tokens = (
                self.full_max_total_num_tokens or self.swa_max_total_num_tokens
            )
        else:
            pool_tokens = self.max_total_num_tokens
        # #346: this number becomes max_req_len, i.e. the length at which a
        # request is REFUSED, and a request length is a GLOBAL token count.
        # Under the even-modulo owner rule max_total_num_tokens is only this
        # rank's shard of that span, so the ceiling came out dcp_size too
        # small and the group turned away context its pool could hold. The
        # weighted rule and every non-lane path are the identity here.
        from sglang.srt.layers.dcp.owner import dcp_global_context_slots

        return dcp_global_context_slots(pool_tokens, self.dcp_size)

    @property
    def kimi_linear_config(self):
        config = self.model_config.hf_config
        if isinstance(config, KimiLinearConfig):
            return config
        return None

    def _get_linear_attn_registry_result(self):
        if self._linear_attn_registry_cache is _UNSET:
            self._linear_attn_registry_cache = get_linear_attn_config(
                self.model_config.hf_config
            )
        return self._linear_attn_registry_cache

    @property
    def linear_attn_model_spec(self):
        result = self._get_linear_attn_registry_result()
        return result[0] if result else None

    @property
    def mambaish_config(self):
        existing = (
            self.mamba2_config
            or self.hybrid_gdn_config
            or self.kimi_linear_config
            or self.hybrid_lightning_config
        )
        if existing:
            return existing
        result = self._get_linear_attn_registry_result()
        return result[1] if result else None

    def _record_kv_cache_dtype(self, resolved: str) -> None:
        # Load-time resolution transition: the weight-resolved kv-cache dtype
        # is declared into the flags tier; the dual-apply inside the helper
        # replaces the legacy in-place write. Mock runners whose server_args
        # is not the published object keep the plain write.
        from sglang.srt.runtime_context import get_context

        if get_context()._server_args is self.server_args:
            from sglang.srt.arg_groups.overrides import declare_load_time_override

            declare_load_time_override(
                "ModelRunner.configure_kv_cache_dtype",
                {"kv_cache_dtype": resolved},
            )
        else:
            self.server_args.override(
                "ModelRunner.configure_kv_cache_dtype", kv_cache_dtype=resolved
            )

    def configure_kv_cache_dtype(self):
        # #127 per-ROLE KV precision (weightless-KV lane): a weightless WORKER
        # rank may store its KV token-shard at a different precision than the
        # head rank, because the two roles spend their VRAM on completely
        # different things (worker = KV only, head = weights + KV). This
        # resolves to a SPEC STRING that is fed through the one existing
        # branch chain below -- there is no second resolver that could drift
        # from it. Unset / "auto" / non-lane / head rank -> identity, so the
        # default path and every existing lane boot are byte-identical.
        from sglang.srt.layers.dcp.role_kv_dtype import effective_kv_cache_dtype_spec

        kv_spec = effective_kv_cache_dtype_spec(
            self.server_args.kv_cache_dtype,
            getattr(self.server_args, "weightless_kv_worker_cache_dtype", None),
            bool(getattr(self, "is_weightless_worker", False)),
        )
        if kv_spec != self.server_args.kv_cache_dtype:
            logger.info(
                "Weightless-KV worker rank %d: per-role KV storage dtype "
                "'%s' (group-wide --kv-cache-dtype is '%s'). This rank's KV "
                "pool, its per-token cell size and any host overflow tier all "
                "follow the role dtype; the cross-rank K/V broadcast is "
                "unaffected (it travels in the model compute dtype).",
                self.tp_rank,
                kv_spec,
                self.server_args.kv_cache_dtype,
            )

        if kv_spec == "auto":
            quant_config = getattr(self.model, "quant_config", None)
            kv_cache_quant_algo = getattr(quant_config, "kv_cache_quant_algo", None)
            if (
                isinstance(kv_cache_quant_algo, str)
                and kv_cache_quant_algo.upper() == "FP8"
            ):
                self.kv_cache_dtype = fp8_dtype if _is_hip else torch.float8_e4m3fn
                self._record_kv_cache_dtype(
                    TORCH_DTYPE_TO_KV_CACHE_STR[self.kv_cache_dtype]
                )
            else:
                self.kv_cache_dtype = self.dtype
        elif kv_spec == "fp8_e5m2":
            # e5m2 keeps 2 mantissa bits against e4m3's 3, and k/v scales are
            # only ever loaded from a checkpoint -- there is no calibration
            # step that would recover the lost precision, so an uncalibrated
            # run stores at scale 1.0. Checkpoint scales, when present, were
            # calibrated against e4m3's 448 maximum and leave most of e5m2's
            # range unused. Store and load stay symmetric in either case
            # (set_kv_buffer divides by the same scale the attention backend
            # multiplies back), so this is a precision note, not a correctness
            # one; e4m3 is the better default for a KV cache on every backend
            # this stack supports.
            logger.warning(
                "--kv-cache-dtype fp8_e5m2 selected. e5m2 carries one mantissa "
                "bit less than fp8_e4m3 and is used here without any KV "
                "calibration step; prefer fp8_e4m3 unless e5m2 is required for "
                "compatibility with an external consumer of the KV cache."
            )
            if _is_hip:  # Using natively supported format
                self.kv_cache_dtype = fp8_dtype
            else:
                self.kv_cache_dtype = torch.float8_e5m2
        elif kv_spec == "fp8_e4m3":
            if _is_hip:  # Using natively supported format
                self.kv_cache_dtype = fp8_dtype
            else:
                self.kv_cache_dtype = torch.float8_e4m3fn
        elif kv_spec in ("bf16", "bfloat16"):
            self.kv_cache_dtype = torch.bfloat16
        elif kv_spec == "fp4_e2m1":
            if hasattr(torch, "float4_e2m1fn_x2"):
                self.kv_cache_dtype = torch.float4_e2m1fn_x2
                logger.warning(f"FP4 (E2M1) KV Cache might lead to a accuracy drop!")
            else:
                logger.warning(
                    f"--kv-cache-dtype falls back to 'auto' because this torch version does not support torch.float4_e2m1fn_x2"
                )
                self.kv_cache_dtype = self.dtype
        else:
            raise ValueError(f"Unsupported kv_cache_dtype: {kv_spec}.")

        # DFLASH: fa4 draft attention can't read the target's fp8 KV (needs K.dtype == Q.dtype),
        # so give the fa4 draft its own compute-dtype KV. fp8-capable backends keep the target dtype.
        if (
            self.is_draft_worker
            and self.spec_algorithm.is_dflash()
            and self.server_args.speculative_draft_attention_backend == "fa4"
            and self.kv_cache_dtype != self.dtype
        ):
            logger.info(
                "DFLASH fa4 draft: overriding KV cache dtype %s -> %s "
                "(fa4 needs K.dtype == Q.dtype; cannot read the target's quantized KV).",
                self.kv_cache_dtype,
                self.dtype,
            )
            self.kv_cache_dtype = self.dtype

    def init_cublas(self):
        """We need to run a small matmul to init cublas. Otherwise, it will raise some errors later."""
        dtype = torch.float16
        device = "cuda"
        a = torch.ones((16, 16), dtype=dtype, device=device)
        b = torch.ones((16, 16), dtype=dtype, device=device)
        c = a @ b
        return c

    def init_attention_backend(self):
        """Init attention kernel backend."""
        if self.server_args.enable_pdmux:
            self.attn_backend = self._get_attention_backend(init_new_workspace=True)
            self.decode_attn_backend_group = []
            for _ in range(self.server_args.sm_group_num):
                self.decode_attn_backend_group.append(self._get_attention_backend())
            self.decode_attn_backend = self.decode_attn_backend_group[0]
        elif self.server_args.enable_two_batch_overlap and not self.is_draft_worker:
            self.attn_backend = TboAttnBackend.init_new(self._get_attention_backend)
        else:
            self.attn_backend = self._get_attention_backend()

        # Record resolved per-mode backends on the backend for model dispatch.
        self.attn_backend.prefill_attention_backend_str = (
            self.prefill_attention_backend_str
        )
        self.attn_backend.decode_attention_backend_str = (
            self.decode_attention_backend_str
        )

    def _get_attention_backend(self, init_new_workspace: bool = False):
        """Init attention kernel backend."""
        draft_attn_backend = self.server_args.speculative_draft_attention_backend
        if self.is_draft_worker and draft_attn_backend:
            logger.warning(
                f"Overriding draft attention backend to {draft_attn_backend}."
            )
            # Single backend for all draft modes (no prefill/decode split).
            self.prefill_attention_backend_str = draft_attn_backend
            self.decode_attention_backend_str = draft_attn_backend
            return self._get_attention_backend_from_str(
                draft_attn_backend,
                init_new_workspace=init_new_workspace,
            )

        (
            self.prefill_attention_backend_str,
            self.decode_attention_backend_str,
        ) = self.server_args.get_attention_backends()

        if self.decode_attention_backend_str != self.prefill_attention_backend_str:
            from sglang.srt.layers.attention.hybrid_attn_backend import (
                HybridAttnBackend,
            )

            attn_backend = HybridAttnBackend(
                self,
                decode_backend=self._get_attention_backend_from_str(
                    self.decode_attention_backend_str,
                    init_new_workspace=init_new_workspace,
                ),
                prefill_backend=self._get_attention_backend_from_str(
                    self.prefill_attention_backend_str,
                    init_new_workspace=init_new_workspace,
                ),
            )
            logger.info(
                f"Using hybrid attention backend for decode and prefill: "
                f"decode_backend={self.decode_attention_backend_str}, "
                f"prefill_backend={self.prefill_attention_backend_str}."
            )
            logger.warning(
                "Warning: Attention backend specified by --attention-backend or default backend might be overridden."
                "The feature of hybrid attention backend is experimental and unstable. Please raise an issue if you encounter any problem."
            )
        else:
            attn_backend = self._get_attention_backend_from_str(
                self.server_args.attention_backend,
                init_new_workspace=init_new_workspace,
            )

        return attn_backend

    def _get_attention_backend_from_str(
        self, backend_str: str, init_new_workspace: bool = False
    ):
        if backend_str not in ATTENTION_BACKENDS:
            raise ValueError(f"Invalid attention backend: {backend_str}")
        self.init_new_workspace = init_new_workspace
        full_attention_backend = ATTENTION_BACKENDS[backend_str](self)
        return attn_backend_wrapper(self, full_attention_backend)

    def maybe_init_ngram_embedding(self):
        self.use_ngram_embedding = self.model_config.use_ngram_embedding
        if self.use_ngram_embedding:
            from sglang.srt.layers.n_gram_embedding import NgramEmbedding

            # Sized to mirror req_to_token (indexed by req_pool_idx).
            self.token_table = torch.empty(
                self.req_to_token_pool.req_to_token.shape[0],
                self.model_config.context_len,
                dtype=torch.int32,
                device=self.device,
            )
            chunked_prefill_size = self.server_args.chunked_prefill_size
            assert (
                chunked_prefill_size is not None and chunked_prefill_size > 0
            ), "Ngram embedding requires chunked prefill to be enabled (chunked_prefill_size > 0)"
            for module in self.model.modules():
                if isinstance(module, NgramEmbedding):
                    module.init_buffers(
                        self.max_running_requests, chunked_prefill_size, self.device
                    )

    def maybe_update_ngram_token_table(
        self,
        next_token_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        """Update the ngram embedding token table after sampling."""
        ngram_embedding_info = forward_batch.ngram_embedding_info
        if ngram_embedding_info is None:
            return
        update_ngram_token_table_after_sampling(
            ngram_embedding_info=ngram_embedding_info,
            next_token_ids=next_token_ids,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            batch_size=forward_batch.batch_size,
        )

    def init_decode_cuda_graph(self):
        """Capture device graphs."""
        if self.is_phase_flip_pp_stack:
            raise RuntimeError(
                "#631 pin 2 (structural): the phase-flip PP stack must not "
                "capture a decode CUDA graph set; only the TP stack captures "
                "decode graphs (once, against its fixed arena addresses). "
                "Reaching this line means a new code path bypassed the "
                "init_cuda_graphs carve."
            )
        self.decode_cuda_graph_runner = None
        self.graph_mem_usage = 0

        if not self.is_generation:
            # TODO: Currently, cuda graph only captures decode steps, which only exists for generation models
            return

        if self.server_args.model_impl.lower() == ModelImpl.MINDSPORE:
            return

        if self.device != "cpu" and check_cuda_graph_backend(
            Phase.DECODE, Backend.DISABLED
        ):
            return

        if self.device == "cpu" and not get_flags().capture.enable_torch_compile:
            return

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        graph_backend = defaultdict(
            lambda: f"{current_platform.device_name} graph",
            {
                "cuda": "CUDA graph",
                "musa": "CUDA graph",
                "cpu": "CPU graph",
                "npu": "NPU graph",
                "xpu": "XPU graph",
            },
        )
        role = "draft" if self.is_draft_worker else "target"
        if self.spec_algorithm.is_speculative():
            capture_name = f"{role} verify"
            num_tokens_per_bs = (
                self.spec_algorithm.get_num_tokens_per_bs_for_target_verify(
                    self.server_args.speculative_num_draft_tokens,
                    self.is_draft_worker,
                )
            )
        else:
            capture_name = f"{role} decode"
            num_tokens_per_bs = 1
        capture_bs, _ = get_batch_sizes_to_capture(self, num_tokens_per_bs)
        decode_backend = self.server_args.cuda_graph_config.decode.backend
        logger.info(
            f"Capture {capture_name} {graph_backend[self.device]} begin. "
            f"backend={decode_backend}, num_tokens_per_bs={num_tokens_per_bs}, "
            f"bs={capture_bs}, avail mem={before_mem:.2f} GB"
        )

        if current_platform.is_out_of_tree():
            GraphRunnerCls = current_platform.get_graph_runner_cls()
            self.decode_cuda_graph_runner = GraphRunnerCls(self)
        else:
            from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
                DecodeCudaGraphRunner,
            )

            graph_runners = defaultdict(
                lambda: DecodeCudaGraphRunner,
                {
                    "cpu": CPUGraphRunner,
                    "npu": NPUGraphRunner,
                    "xpu": XPUGraphRunner,
                },
            )
            self.decode_cuda_graph_runner = graph_runners[self.device](self)

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        self.graph_mem_usage = before_mem - after_mem
        logger.info(
            f"Capture {capture_name} {graph_backend[self.device]} end. "
            f"elapsed={time.perf_counter() - tic:.2f} s, "
            f"mem usage={self.graph_mem_usage:.2f} GB, avail mem={after_mem:.2f} GB."
        )

    def init_prefill_cuda_graph(self, force_for_draft_worker: bool = False):
        """Initialize prefill CUDA graph runner."""
        if self.is_phase_flip_pp_stack:
            raise RuntimeError(
                "#631 pin 2 (structural): the phase-flip PP stack must not "
                "capture ANY CUDA graph; init_cuda_graphs routes it to the "
                "EagerRunner before this method can be reached. Reaching "
                "this line means a new code path bypassed that carve."
            )
        self.prefill_cuda_graph_runner = None

        # Weightless-KV fast lane (#133): only the DECODE phase is captured, as
        # symmetric head+worker decode graphs. PREFILL/EXTEND stays EAGER (with
        # the anti-hang guard ON) — the weightless workers hold a meta model and
        # cannot record model.forward, and the extend collective sequence is
        # data-dependent (has_prefix) and validated eagerly by the guard. Both
        # the head and the workers route prefill to the EagerRunner so the two
        # sides stay rank-uniform (both eager on extend).
        if self.is_weightless_head or self.is_weightless_worker:
            if not self.is_draft_worker:
                self.prefill_cuda_graph_runner = self.eager_runner
            return

        if check_cuda_graph_backend(Phase.PREFILL, Backend.DISABLED):
            logger.info(
                "Disable prefill CUDA graph because cuda_graph_config "
                "resolved prefill.backend='disabled' (e.g. via "
                "--cuda-graph-backend-prefill=disabled or auto-disable rules)."
            )
            # Prefill cuda graph disabled: route eager prefill through the
            # EagerRunner (its can_run_graph returns False, so _forward_raw's
            # extend branch falls through to the eager path).
            if not self.is_draft_worker:
                self.prefill_cuda_graph_runner = self.eager_runner
            return

        # Draft models skip here during __init__; the eagle worker calls
        # this method explicitly (force_for_draft_worker=True) after
        # init_lm_head so graphs capture the final embedding weights.
        # A dual-group lane TARGET (#274) is constructed as a draft worker
        # (secondary-runner gates) but is a PREFILL lane -- its prefill
        # graphs are the point, so it does not take the draft skip. The
        # lane's NEXTN HEAD does take it: it is a draft model in the ordinary
        # sense (see ModelRunner.is_dual_group_lane_target).
        if (
            self.is_draft_worker
            and not force_for_draft_worker
            and not self.is_dual_group_lane_target
        ):
            return

        # Skip prefill CG for EAGLE target on tc_piecewise: that backend
        # captures CaptureHiddenMode.NULL while runtime requests FULL, so
        # the captured graph is dead, and capturing it perturbs FP4 /
        # TRTLLM-MoE state and corrupts decode replay (see #28386). BCG
        # captures FULL for EAGLE target in PrefillCudaGraphRunner.__init__
        # (restored from #25795), so it does NOT need this skip.
        if (
            self.spec_algorithm.is_eagle()
            and not self.is_draft_worker
            and not self.server_args.enable_return_hidden_states
            and not check_cuda_graph_backend(Phase.PREFILL, Backend.BREAKABLE)
        ):
            logger.info(
                "Disable prefill CUDA graph for EAGLE target on tc_piecewise "
                "to avoid FP4/MoE decode-replay corruption (#28386)."
            )
            self.prefill_cuda_graph_runner = self.eager_runner
            return

        # Disable prefill CUDA graph for non-language models
        if not hasattr(self.model, "model"):
            logger.warning(
                "Disable prefill CUDA graph because the model is not a language model"
            )
            return

        # Disable prefill CUDA graph for non capture size
        if not self.server_args.cuda_graph_config.prefill.bs:
            logger.warning(
                "Disable prefill CUDA graph because the capture size is not set"
            )
            return

        # Collect attention layers and moe layers from the model
        self.model.model = resolve_language_model(self.model)
        language_model = getattr(self.model, "language_model", self.model)

        # Find the module that owns the decoder `layers`. Models wrap it at
        # varying depths: a direct text model exposes `.layers`, a CausalLM
        # wraps it as `.model.layers`, and some multimodal models add another
        # level (e.g. DeepSeek-OCR: OCR wrapper -> Deepseek*ForCausalLM ->
        # text model -> `.layers`). Descend the `.model` chain until we find it.
        layer_model = language_model
        while not hasattr(layer_model, "layers") and hasattr(layer_model, "model"):
            layer_model = layer_model.model

        if not hasattr(layer_model, "layers"):
            logger.warning(
                "Disable prefill CUDA graph because the model does not have a 'layers' attribute"
            )
            return

        self.attention_layers = []
        self.moe_layers = []
        self.moe_fusions = []
        self.dsa_indexers = []
        for layer in layer_model.layers:
            attn_layer = None
            if hasattr(layer, "self_attn"):
                if hasattr(layer.self_attn, "attn"):
                    attn_layer = layer.self_attn.attn
                elif hasattr(layer.self_attn, "attn_mqa"):
                    # For DeepSeek model
                    attn_layer = layer.self_attn.attn_mqa
                    if _is_hip and hasattr(layer.self_attn, "attn_mha"):
                        attn_layer._pcg_mha_companion = layer.self_attn.attn_mha
            # For hybrid model
            elif hasattr(layer, "attn"):
                attn_layer = layer.attn
            elif hasattr(layer, "linear_attn"):
                if hasattr(layer.linear_attn, "attn"):
                    attn_layer = layer.linear_attn.attn
                else:
                    attn_layer = layer.linear_attn
            # For InternVL model
            elif hasattr(layer, "attention"):
                if hasattr(layer.attention, "attn"):
                    attn_layer = layer.attention.attn
            # For NemotronH and similar hybrid models using 'mixer' attribute
            elif hasattr(layer, "mixer"):
                if hasattr(layer.mixer, "attn"):
                    attn_layer = layer.mixer.attn
                elif hasattr(layer, "_forward_mamba"):
                    # Mamba layer with split op support - store the layer itself
                    attn_layer = layer

            if attn_layer is not None:
                self.attention_layers.append(attn_layer)
            elif hasattr(layer, "mixer"):
                self.attention_layers.append(None)

            moe_block = None
            moe_fusion = None
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                moe_block = layer.mlp.experts
                moe_fusion = layer.mlp
            if hasattr(layer, "block_sparse_moe") and hasattr(
                layer.block_sparse_moe, "experts"
            ):
                moe_block = layer.block_sparse_moe.experts
                moe_fusion = layer.block_sparse_moe
            if hasattr(layer, "moe") and hasattr(layer.moe, "experts"):
                moe_block = layer.moe.experts
                moe_fusion = layer.moe
            # For NemotronH MoE layers using 'mixer' attribute
            if hasattr(layer, "mixer") and hasattr(layer.mixer, "experts"):
                moe_block = layer.mixer.experts
                moe_fusion = layer.mixer
            self.moe_layers.append(moe_block)
            self.moe_fusions.append(moe_fusion)
            # NSA indexers (None for layers without NSA)
            dsa_indexer = None
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "indexer"):
                dsa_indexer = layer.self_attn.indexer
            self.dsa_indexers.append(dsa_indexer)

        if len(self.attention_layers) < self.model_config.num_hidden_layers:
            # TODO(yuwei): support Non-Standard GQA
            log_info_on_rank0(
                logger,
                "Disable prefill CUDA graph because some layers do not apply Standard GQA",
            )
            return

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        prefill_backend = self.server_args.cuda_graph_config.prefill.backend
        role = "draft" if self.is_draft_worker else "target"
        capture_name = f"{role} prefill"
        capture_num_tokens = sorted(self.server_args.cuda_graph_config.prefill.bs)
        logger.info(
            f"Capture {capture_name} CUDA graph begin. "
            f"backend={prefill_backend}, num_tokens={capture_num_tokens}, "
            f"avail mem={before_mem:.2f} GB"
        )

        self.prefill_cuda_graph_runner = PrefillCudaGraphRunner(self)

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        mem_usage = before_mem - after_mem
        logger.info(
            f"Capture {capture_name} CUDA graph end. "
            f"elapsed={time.perf_counter() - tic:.2f} s, "
            f"mem usage={mem_usage:.2f} GB, avail mem={after_mem:.2f} GB."
        )

    def init_threads_binding(self):
        omp_cpuids = os.environ.get("SGLANG_CPU_OMP_THREADS_BIND", "all")
        cpu_ids_by_node = get_cpu_ids_by_node()
        n_numa_node = len(cpu_ids_by_node)
        if omp_cpuids == "all":
            assert self.tp_size <= n_numa_node, (
                f"SGLANG_CPU_OMP_THREADS_BIND is not set, in this case, "
                f"tp_size {self.tp_size} should be smaller than or equal to number of numa node on the machine {n_numa_node}. "
                f"If you need tp_size to be larger than number of numa node, please set the CPU cores for each tp rank via SGLANG_CPU_OMP_THREADS_BIND explicitly. "
                f"For example, on a machine with 2 numa nodes, where core 0-31 are on numa node 0 and core 32-63 are on numa node 1, "
                f"it is suggested to use -tp 2 and bind tp rank 0 to core 0-31 and tp rank 1 to core 32-63. "
                f"This is the default behavior if SGLANG_CPU_OMP_THREADS_BIND is not set and it is the same as setting SGLANG_CPU_OMP_THREADS_BIND=0-31|32-63. "
                f"If you do need tp_size to be larger than the number of numa nodes, you could set SGLANG_CPU_OMP_THREADS_BIND explicitly for example SGLANG_CPU_OMP_THREADS_BIND=0-15|16-31|32-47|48-63 and run with -tp 4. "
                f"If you don't want each tp rank to use all the cores on one numa node, you could set for example SGLANG_CPU_OMP_THREADS_BIND=0-15|32-47 and run with -tp 2."
            )
            if self.tp_size < n_numa_node:
                logger.warning(
                    f"Detected the current machine has {n_numa_node} numa nodes available, but tp_size is set to {self.tp_size}, so only {self.tp_size} numa nodes are used."
                )
            self.local_omp_cpuid = cpu_ids_by_node[self.tp_rank]
        else:
            threads_bind_list = omp_cpuids.split("|")
            assert self.tp_size == len(threads_bind_list), (
                f"SGLANG_CPU_OMP_THREADS_BIND setting must be aligned with TP size parameter ({self.tp_size}). "
                f"Please double check your settings."
            )
            self.local_omp_cpuid = threads_bind_list[self.tp_rank]
            if self.tp_size > n_numa_node:
                logger.warning(
                    f"TP size ({self.tp_size})is larger than numa node number ({n_numa_node}), "
                    f"in this case the available memory amount of each rank cannot be determined in prior. "
                    f"Please set proper `--max-total-tokens` to avoid the out-of-memory error."
                )

    def apply_torch_tp(self):
        logger.info(f"Enabling torch tensor parallelism on {self.tp_size} devices.")
        from sglang.srt.layers.model_parallel import tensor_parallel

        device_mesh = torch.distributed.init_device_mesh(self.device, (self.tp_size,))
        tensor_parallel(self.model, device_mesh)

    def update_decode_attn_backend(self, stream_idx: int):
        self.decode_attn_backend = self.decode_attn_backend_group[stream_idx]

    def _prepare_eager_forward_batch(self, forward_batch: ForwardBatch) -> None:
        """Pad / normalize a batch for the eager (non-cuda-graph) forward.

        Runs the DP/MLP-sync padding, the attn-tp num_token_non_padded
        normalization, and the hisparse-coordinator refresh that the eager
        forward path needs — the cuda-graph path does the equivalent inside the
        runner's capture/replay, so this is skipped there.
        """
        # For MLP sync
        if forward_batch.global_num_tokens_cpu is not None:
            forward_batch.prepare_mlp_sync_batch(self)
        else:
            forward_batch.prepare_attn_tp_scatter_input(self)

        # Normalize num_token_non_padded to be local to this attention TP rank if needed.
        # The skip is scoped to DSACPLayerCommunicator-style CP (DSA, MLA): those
        # flavors already feed a zigzag-split rank-local layout whose token count
        # should not be further divided by attn_tp_size. MHA-arch prefill CP
        # (Qwen3/Qwen2 MoE) keeps the attn_tp-replicated layout and wants the
        # adjustment to run — see docs/design/prefill-cp-mla.md §Phase 5.
        if (
            forward_batch.num_token_non_padded is not None
            and forward_batch.global_num_tokens_gpu is not None
            and require_gathered_buffer(self.server_args)
            and not is_dsa_enable_prefill_cp()
            and not is_mla_prefill_cp_enabled()
        ):
            forward_batch.adjust_num_token_non_padded_for_attn_tp(
                server_args=self.server_args,
            )

        # Hisparse coordinator — backends now read it from self.model_runner.
        if self.hisparse_coordinator is not None:
            self.hisparse_coordinator.num_real_reqs.fill_(forward_batch.batch_size)

    def _pp_kwargs(self, pp_proxy_tensors) -> dict:
        """Build the pp_proxy_tensors forward kwarg, in one place.

        Pipeline-parallel proxy tensors are threaded into model.forward only
        when the model accepts them (``support_pp``).
        """
        return {"pp_proxy_tensors": pp_proxy_tensors} if self.support_pp else {}

    def _extend_forward_kwargs(
        self, forward_batch: ForwardBatch, pp_proxy_tensors
    ) -> dict:
        """Build the extend/prefill model.forward kwargs (pp_proxy_tensors +
        input_embeds / replace_embeds overrides + get_embedding), shared by the
        prefill cuda-graph path and the EagerRunner's eager extend path."""
        kwargs = self._pp_kwargs(pp_proxy_tensors)
        if forward_batch.input_embeds is not None:
            kwargs["input_embeds"] = forward_batch.input_embeds.bfloat16()
        if (
            forward_batch.replace_embeds is not None
            and forward_batch.replace_positions is not None
        ):
            # Token embedding overrides: get base embeddings, scatter replacements
            if "input_embeds" not in kwargs:
                embed_layer = self.model.get_input_embeddings()
                kwargs["input_embeds"] = embed_layer(forward_batch.input_ids)
            kwargs["input_embeds"][forward_batch.replace_positions] = (
                forward_batch.replace_embeds.to(kwargs["input_embeds"].dtype)
            )
        if not self.is_generation:
            kwargs["get_embedding"] = True
        return kwargs

    def forward_split_prefill(
        self,
        forward_batch: ForwardBatch,
        reinit_attn_backend: bool = False,
        forward_count: int = 1,
    ) -> LogitsProcessorOutput:
        if forward_batch.split_index == 0 or reinit_attn_backend:
            self.attn_backend.init_forward_metadata(forward_batch)
        next_split_index = min(
            forward_batch.split_index + forward_count,
            self.model_config.num_hidden_layers,
        )
        ctx = (
            self.device_timer.wrap(metadata={"category": "split_prefill"})
            if self.device_timer
            else contextlib.nullcontext()
        )
        with ctx:
            ret = self.model.forward_split_prefill(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                (forward_batch.split_index, next_split_index),
            )
        forward_batch.split_index = next_split_index
        return ret

    def _ensure_rope_cache_for_batch(self, forward_batch: ForwardBatch) -> None:
        """Fill the lazy RoPE reserve up to this batch's deepest position.

        The whole point of the lazy reserve is that a 1M ceiling costs nothing
        until it is used, so the cost of using it has to be paid HERE, before
        the forward that reads those rows -- including before a captured decode
        graph is replayed, which is why this sits in forward() and not in the
        model. The fill writes in place, so the address the graph captured does
        not move.
        """
        from sglang.srt.layers.rotary_embedding import lazy_cos_sin_cache

        if not lazy_cos_sin_cache.any_installed():
            return

        # CAPTURE IS NOT A BATCH. The eagle draft worker enters this same
        # forward() from inside torch.cuda.graph capture
        # (eagle_draft_cuda_graph_runner.run_once -> draft_runner.forward), and
        # while a stream is capturing, ANY device read is
        # cudaErrorStreamCaptureUnsupported -- which is how the first lazy boot
        # died, in capture_end rather than at the read. Capture runs dummy
        # positions inside the initial chunk, and every real batch enters
        # forward() again outside capture, so there is nothing to grow here.
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return

        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if seq_lens_cpu is None or seq_lens_cpu.numel() == 0:
            return
        needed = int(seq_lens_cpu.max()) + self._rope_lazy_batch_margin

        # MULTIMODAL BATCHES BREAK THE seq_lens BOUND. Text positions are
        # bounded by the sequence length, but an M-RoPE batch carries its own
        # position ids (temporal/height/width sections), which are NOT that
        # bound -- a video's temporal ids run past the token count. The
        # allowlisted lazy class on this rig IS the M-RoPE YaRN variant, so
        # this is not hypothetical. Only these batches pay the device read.
        mrope_positions = getattr(forward_batch, "mrope_positions", None)
        if mrope_positions is not None and mrope_positions.numel():
            needed = max(
                needed, int(mrope_positions.max()) + self._rope_lazy_batch_margin
            )

        lazy_cos_sin_cache.ensure_capacity_for_position(needed)

        if envs.SGLANG_ROPE_LAZY_VERIFY.get():
            # Can-fail proof that the fill really precedes every position that
            # gets read. Costs a device sync, so it is never on in serving.
            positions = getattr(forward_batch, "positions", None)
            if positions is not None and positions.numel():
                lazy_cos_sin_cache.verify_positions_are_filled(
                    int(positions.max()), where="ModelRunner.forward"
                )

    def forward(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: Optional[bool] = None,  # deprecated
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> ModelRunnerOutput:
        # Deprecated kwarg: pre-planners mark the batch themselves now.
        forward_batch.apply_deprecated_skip_attn_backend_init(skip_attn_backend_init)

        # #631: DOES THIS HIDDEN STATE BELONG TO THIS BATCH?
        #
        # Under PP the pairing of a received proxy tensor to a local batch is
        # PURELY POSITIONAL -- "whatever came off the wire this slot iteration
        # belongs to whatever batch I have this slot iteration". PPProxyTensors
        # carries no mb_id, no sequence number and no length, the receive
        # demultiplexes only on __msg_type__, and the model asserts the
        # proxy's PRESENCE but never its SHAPE. So a single stranded message
        # puts every later receive off by one, silently and permanently.
        #
        # It can strand: the recv is guarded by THIS rank's ``cur_batch`` while
        # the upstream's send is guarded by the UPSTREAM's, and nothing
        # enforces that the two agree -- only the unasserted "every rank builds
        # the same batches" property. A commit hides any strand because the
        # cutover re-enters event_loop_pp and init_pp_loop_state resets every
        # buffer including _pp_tensor_dict_inbox; an ABANDON resets nothing.
        #
        # MEASURED COST (2026-08-09 05:21:16Z, specimen
        # /spinning/evidence-631/oom_and_abandon_20260809T0521Z): a decode
        # batch of one request was computed on a 2048-row chunked-prefill
        # hidden state, and the mismatched pair reached causal_conv1d_update
        # -- an out-of-bounds write into another request's conv state.
        #
        # Checked here because this is the one funnel every PP stage's forward
        # passes through, so one check covers every model. Two host-side shape
        # reads, no synchronisation. It names mb-level facts the old failure
        # could not: the previous symptom appeared 30 layers deep in a GDN
        # kernel and pointed at the wrong subsystem entirely.
        if pp_proxy_tensors is not None:
            _hs = pp_proxy_tensors.tensors.get("hidden_states")
            _want = forward_batch.input_ids.shape[0]
            if _hs is not None and _hs.shape[0] != _want:
                raise ValueError(
                    f"#631 PP proxy/batch mismatch: received hidden_states with "
                    f"{_hs.shape[0]} row(s) for a {forward_batch.forward_mode} "
                    f"batch of {_want} token(s) "
                    f"(bs={forward_batch.batch_size}). The hidden states of one "
                    f"microbatch have been paired with another microbatch's "
                    f"metadata -- the PP stages are out of step. Computing on "
                    f"this pair corrupts memory rather than merely failing: the "
                    f"cache-index tensors are sized for THIS batch."
                )

        self.forward_pass_id += 1

        # Try msprob debugger
        if self.msprobe_debugger is not None:
            rank_id = (
                self.gpu_id if self.dp_size is not None and self.dp_size > 1 else None
            )
            self.msprobe_debugger.start(model=self.model, rank_id=rank_id)

        # Step span
        step_span_ctx = profile_range(_build_step_span_name(forward_batch))

        canary_ctx = (
            context_tuple(
                c.with_ops_outside_graph(
                    single_forward_indices=[0],
                    maybe_inaccurate_forward_batch=forward_batch,
                ),
                c.with_active_single_forward_manager(0),
            )
            if not self.is_draft_worker and ((c := self.canary_manager) is not None)
            else contextlib.nullcontext()
        )

        # #656 T1: a lazy RoPE reserve has written only the rows positions have
        # actually reached, so the rows THIS batch needs may still be
        # unwritten reservation. Host-side only: seq_lens_cpu is already a CPU
        # tensor, and the speculative path has bumped it by its draft tokens
        # before this point, so no synchronisation is added on any path. With
        # no lazy cache installed -- the shipped default -- this is one bool.
        self._ensure_rope_cache_for_batch(forward_batch)

        with (
            canary_ctx,
            step_span_ctx,
            get_global_expert_distribution_recorder().with_forward_pass(
                self.forward_pass_id,
                forward_batch,
            ) as recorder_outputs,
        ):
            output = self._forward_raw(
                forward_batch,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
            if self.enable_elastic_ep:
                output = self._maybe_rebalance_after_rank_fault(
                    output,
                    forward_batch,
                    pp_proxy_tensors,
                    reinit_attn_backend,
                    split_forward_count,
                )
        output.expert_distribution_metrics = recorder_outputs.get("metrics")

        # Phase-footprint probe: the prefill activation peak. EXTEND only --
        # decode runs inside captured graphs against a far smaller working set,
        # so folding it in could only lower the high-water mark the ledger
        # reserves for. Inert unless SGLANG_PHASE_FOOTPRINT_DUMP is set.
        if forward_batch.forward_mode.is_extend():
            from sglang.srt.mem_ledger import activation_probe

            activation_probe.record_prefill_peak(
                self, int(forward_batch.extend_num_tokens or 0)
            )
            # #605: one mark on the FIRST extend only. The boundary that
            # matters is "boot finished -> real work has run once"; marking
            # every prefill would turn a boot record into a serving log and put
            # an NVML call on the hot path.
            from sglang.srt.mem_ledger import flight_recorder

            if not self._flight_first_forward_marked:
                self._flight_first_forward_marked = True
                flight_recorder.mark(
                    "first_forward",
                    rank=self.tp_rank,
                    extra={"extend_tokens": int(forward_batch.extend_num_tokens or 0)},
                )

        no_copy_to_cpu = not self.server_args.disable_overlap_schedule
        if (
            not self.is_draft_worker
            and (experts_capturer := get_global_experts_capturer()) is not None
        ):
            output.routed_experts_output = experts_capturer.on_forward_end(
                forward_batch=forward_batch,
                can_run_graph=output.can_run_graph,
                cuda_graph_batch=getattr(self.decode_cuda_graph_runner, "bs", None),
                no_copy_to_cpu=no_copy_to_cpu,
            )

        if (indexer_capturer := get_global_indexer_capturer()) is not None:
            output.indexer_topk_output = indexer_capturer.on_forward_end(
                forward_batch=forward_batch,
                can_run_graph=output.can_run_graph,
                cuda_graph_batch=getattr(self.decode_cuda_graph_runner, "bs", None),
                no_copy_to_cpu=no_copy_to_cpu,
            )

        if self.eplb_manager is not None:
            self.eplb_manager.on_forward_pass_end()

        if dumper.may_enable:
            dumper.step()

        if self.msprobe_debugger is not None:
            self.msprobe_debugger.stop()
            self.msprobe_debugger.step()

        if self.enable_elastic_ep:
            self.maybe_recover_ep_ranks()

        return output

    def _maybe_execute_deferred_mamba_cow_and_clear(
        self, forward_batch: ForwardBatch
    ) -> None:
        """Run deferred clear/COW on the forward stream, before the mamba layers
        read the pool, so the copies don't race the scheduler copy stream.

        No-op unless this is an extend forward on a mamba model's target worker;
        COW/clear only happen at prefix match on extend.
        """
        pool = self.req_to_token_pool
        if (
            not isinstance(pool, HybridReqToTokenPool)
            or self.is_draft_worker
            or not forward_batch.forward_mode.is_extend()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            return
        if (
            forward_batch.mamba_clear_indices is not None
            and len(forward_batch.mamba_clear_indices) > 0
        ):
            # mamba_pool is a pure PHYSICAL store; translate before zeroing or
            # clear_slots zeroes the wrong physical slots.
            pool.mamba_pool.clear_slots(
                pool.translate_mamba_indices(forward_batch.mamba_clear_indices)
            )
        if (
            forward_batch.mamba_cow_src_indices is not None
            and len(forward_batch.mamba_cow_src_indices) > 0
        ):
            if pool.mamba_ckpt_pool is not None:
                # int8 checkpoints: dequantize src int8 ckpt slot into the active bf16 dst.
                pool.mamba_ckpt_pool.load_to_active(
                    pool.mamba_pool,
                    forward_batch.mamba_cow_src_indices,
                    forward_batch.mamba_cow_dst_indices,
                )
            else:
                # mamba_pool is a pure PHYSICAL store; translate both COW slot ids.
                pool.mamba_pool.copy_from(
                    pool.translate_mamba_indices(forward_batch.mamba_cow_src_indices),
                    pool.translate_mamba_indices(forward_batch.mamba_cow_dst_indices),
                )
        forward_batch.mamba_clear_indices = None
        forward_batch.mamba_cow_src_indices = None
        forward_batch.mamba_cow_dst_indices = None

    def _weightless_attn_layers(self):
        """The model's full-attention RadixAttention modules in layer order.
        The weightless worker loops these (only) to issue the per-layer dcp
        dispatch; GDN/linear-attention layers carry no KV and are skipped."""
        cached = getattr(self, "_wl_attn_layers_cache", None)
        if cached is not None:
            return cached
        from sglang.srt.layers.radix_attention import RadixAttention

        layers = [m for m in self.model.modules() if isinstance(m, RadixAttention)]
        layers.sort(key=lambda a: a.layer_id)
        self._wl_verify_role_kv_scales(layers)
        self._wl_attn_layers_cache = layers
        return layers

    def _wl_verify_role_kv_scales(self, layers):
        """#127 GRENZ-ASSERTION for per-ROLE KV precision (once per process).

        A worker quantizes its KV shard with the scales carried by its OWN
        (meta-device) attention modules, while the head quantizes with the
        scales carried by the real weights. Those are the same number only as
        long as both are the untrained default (None / 1.0). If a checkpoint
        or a scale file ever put a non-unit scale on one role's layers, the
        two roles' shards would sit on different scales and the LSE merge
        would silently combine incomparable partial attentions -- no shape,
        dtype or NCCL check would notice. server_args refuses the known
        source of that (--quantization-param-path); this catches any other
        route at the point where the layers first exist.

        Only runs when a role split is actually configured; on the inherited
        path (default) it is a no-op.
        """
        from sglang.srt.layers.dcp.role_kv_dtype import worker_dtype_is_role_split

        if not worker_dtype_is_role_split(
            self.server_args.kv_cache_dtype,
            getattr(self.server_args, "weightless_kv_worker_cache_dtype", None),
        ):
            return
        for layer in layers:
            for name in ("k_scale_float", "v_scale_float"):
                scale = getattr(layer, name, None)
                if scale is None or float(scale) == 1.0:
                    continue
                raise ValueError(
                    "--weightless-kv-worker-cache-dtype is active, but "
                    f"attention layer {layer.layer_id} on this weightless "
                    f"worker carries a non-unit {name}={float(scale)}. The "
                    "head rank scales its own KV shard independently, so the "
                    "two roles would quantize against different scales and "
                    "the cross-rank LSE merge would combine incomparable "
                    "partials. Drop the per-role KV dtype and use the "
                    "group-wide --kv-cache-dtype for this checkpoint."
                )

    def _forward_weightless_worker(
        self, forward_batch: ForwardBatch
    ) -> ModelRunnerOutput:
        """Stripped attention-only forward for a weightless KV worker rank
        (Option-B B1). Runs NO model layers; per full-attention layer it
        participates in the head rank's dcp dispatch with an empty [T,0,D] head
        slice + its KV token-shard (KV-broadcast+owner-write, Q-broadcast,
        attend local shard, LSE-merge), then returns a sentinel output (no
        logits). Sampling is skipped on workers (tp_worker gates on
        is_weightless_worker)."""
        self._prepare_eager_forward_batch(forward_batch)
        # IDLE (padding) forward: the model's decoder layers skip attention when
        # forward_mode.is_idle(), so the head issues NO DCP collective -> the
        # worker must also issue none. Return without metadata/dispatch.
        if forward_batch.forward_mode.is_idle():
            return ModelRunnerOutput(logits_output=None, can_run_graph=False)
        # Hybrid-GDN models wrap the flashinfer FULL-attention backend inside a
        # HybridLinearAttnBackend (full_attn_backend + a GDN linear backend). The
        # weightless worker only handles the full-attention KV dispatch (it never
        # runs GDN), and the weightless_worker methods live on the flashinfer
        # backend -> resolve to full_attn_backend when present.
        attn = getattr(self.attn_backend, "full_attn_backend", self.attn_backend)
        attn.init_forward_metadata(forward_batch)
        if forward_batch.forward_mode.is_decode():
            worker_dispatch = attn.forward_decode_weightless_worker
        elif forward_batch.forward_mode.is_extend():
            # PREFILL/EXTEND: mirror the head's _forward_extend_dcp per-layer
            # dispatch (data-dependent 1-or-4 DCP collectives on the rank-uniform
            # has_prefix). The worker writes the current chunk's KV to its owned
            # token-shard and reads its owned prefix slots; it skips the
            # head-local current-chunk ragged attention.
            #
            # TARGET_VERIFY lands here too (ForwardMode.is_extend() includes it),
            # and that is the intended route for chain speculation (#143), not an
            # accident: a verify step IS an extend over bs*(k+1) rows whose
            # committed prefix is read owner-sharded while the draft block is
            # attended out of the head's local k/v. has_prefix is forced True for
            # verify on both sides (layers/dcp/lockstep.weightless_has_prefix), so
            # the collective sequence is the same 4-op tuple decode issues.
            worker_dispatch = attn.forward_extend_weightless_worker
        else:
            raise NotImplementedError(
                "weightless-KV worker forward: unsupported forward mode "
                f"{forward_batch.forward_mode} (decode, extend and target-verify "
                "are supported). Draft-side modes (DRAFT_EXTEND_V2) must never "
                "reach a weightless worker: the lane requires "
                "--speculative-draft-placement solo, under which non-host ranks "
                "run no draft forward at all."
            )
        for layer in self._weightless_attn_layers():
            worker_dispatch(layer, forward_batch)
        return ModelRunnerOutput(logits_output=None, can_run_graph=False)

    def _forward_raw(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors],
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> ModelRunnerOutput:
        # Task #343 layer tap. Driven from here rather than from a forward
        # pre-hook on the model, because the model is entered as
        # ``model.forward(...)`` -- torch runs no hook on a direct .forward()
        # call, only on __call__, so a root pre-hook would never fire and every
        # submodule hook would record against an unstarted step.
        if getattr(self, "_layer_fingerprint", None) is not None:
            self._layer_fingerprint.begin_step(forward_batch)
        # Forward-peak probe (#417 w3 follow-up). Same reasoning as the #343 tap
        # above for living here rather than in a module hook. Off by default;
        # `maybe_create` returns None unless SGLANG_FORWARD_PEAK_PATH is set.
        peak_probe = getattr(self, "_forward_peak", "unset")
        if peak_probe == "unset":
            from sglang.srt.model_executor.forward_peak import maybe_create

            peak_probe = maybe_create(
                f"tp{getattr(self, 'tp_rank', 0)}", getattr(self, "device_id", None)
            )
            self._forward_peak = peak_probe
        if peak_probe is not None:
            peak_probe.begin(
                "decode" if forward_batch.forward_mode.is_decode() else "extend",
                (
                    int(getattr(forward_batch, "input_ids").shape[0])
                    if getattr(forward_batch, "input_ids", None) is not None
                    else 0
                ),
            )
        if has_forward_context():
            ctx_mgr = contextlib.nullcontext()
        else:
            ctx_mgr = forward_context(ForwardContext(attn_backend=self.attn_backend))
        from sglang.srt.model_executor.forward_peak import peak_scope

        with ctx_mgr, peak_scope(peak_probe):
            # Weightless-KV fast lane (Option-B, B1): reset the per-forward
            # anti-hang guard step counter on BOTH the head rank and the
            # weightless workers at the same point, so their dcp-collective step
            # counters stay aligned across the asymmetric forward. The worker
            # then runs the stripped attention-only forward and returns early
            # (no model layers, no logits); the head continues to model.forward.
            if self.is_weightless_head or self.is_weightless_worker:
                from sglang.srt.layers.dcp.collective_guard import (
                    reset_forward_guard,
                    set_guard_enabled,
                )

                reset_forward_guard()
                # Rank-uniform guard rule for the graph regime (#133): a captured
                # decode graph replays its DCP collectives with NO gloo handshake
                # (the guard's per-step host-sync is not capturable and was baked
                # out at capture time). So when decode CUDA graphs are enabled,
                # the DCP guard must be OFF for DECODE on BOTH the head and the
                # workers — including any eager-decode bucket-miss fallback —
                # otherwise one side runs a guard-free graph while the other
                # blocks on the gloo handshake (a desync/hang). PREFILL/EXTEND
                # stays eager and GUARDED. `disable_cuda_graph` and the forward
                # mode are rank-uniform, so head and workers decide identically.
                #
                # #143: TARGET_VERIFY joins DECODE/IDLE here. Under chain spec
                # every generation step IS a verify, captured by the same
                # DecodeCudaGraphRunner (its capture_forward_mode flips to
                # TARGET_VERIFY), so a verify replay is just as guard-free as a
                # decode replay -- and leaving the guard ON for verify would put
                # a gloo handshake inside a captured region. The rule to keep is
                # "guard OFF whenever a captured region may be entered", i.e.
                # exactly the modes ForwardMode.is_cuda_graph() admits that the
                # lane can reach. PREFILL/EXTEND stays eager and GUARDED.
                _wl_mode = forward_batch.forward_mode
                _wl_decode_graphs = not self.server_args.disable_cuda_graph and (
                    _wl_mode.is_decode_or_idle() or _wl_mode.is_target_verify()
                )
                set_guard_enabled(not _wl_decode_graphs)
            if self.is_weightless_worker:
                # Worker decode-graph replay (#133): mirror the head's decode
                # graph. When the batch is graph-eligible, replay the captured
                # weightless-worker decode graph — its recorded DCP collectives
                # pair up with the head's decode-graph replay in lockstep.
                # Otherwise fall back to the eager per-layer dispatch (guard-free
                # for decode per the rule above; guarded for extend).
                mode_check = (
                    forward_batch.forward_mode.is_cpu_graph
                    if self.device == "cpu"
                    else forward_batch.forward_mode.is_cuda_graph
                )
                worker_can_run_graph = bool(
                    mode_check()
                    and self.decode_cuda_graph_runner
                    and self.decode_cuda_graph_runner.can_run_graph(forward_batch)
                )
                if worker_can_run_graph:
                    self.decode_cuda_graph_runner.execute(
                        forward_batch,
                        pp_proxy_tensors=pp_proxy_tensors,
                    )
                    return ModelRunnerOutput(logits_output=None, can_run_graph=True)
                return self._forward_weightless_worker(forward_batch)

            mode_check = (
                forward_batch.forward_mode.is_cpu_graph
                if self.device == "cpu"
                else forward_batch.forward_mode.is_cuda_graph
            )
            can_run_graph = bool(
                mode_check()
                and self.decode_cuda_graph_runner
                and self.decode_cuda_graph_runner.can_run_graph(forward_batch)
            )

            if (
                forward_batch.forward_mode.is_decode()
                and self.hisparse_coordinator is not None
            ):
                forward_batch.hisparse_coordinator = self.hisparse_coordinator
                self.hisparse_coordinator.wait_for_pending_backup()
                self.hisparse_coordinator.num_real_reqs.fill_(forward_batch.batch_size)

            # Replay cuda graph if applicable
            if can_run_graph:
                ret = self.decode_cuda_graph_runner.execute(
                    forward_batch,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
                return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)

            # DP / MLP-sync padding + attn-tp normalization. Only the decode
            # cuda-graph path above pre-pads its static buffers and returns
            # early; split prefill, the prefill cuda graph, and the eager
            # forward all run the live batch and need this first — it sets
            # global_dp_buffer_len / padded token counts that graph eligibility
            # and the collectives depend on.
            self._prepare_eager_forward_batch(forward_batch)

            # Deferred mamba COW/clear on the forward stream, before the extend
            # dispatch below reads the pool.
            self._maybe_execute_deferred_mamba_cow_and_clear(forward_batch)

            if forward_batch.forward_mode.is_split_prefill():
                # Layer-split mode; stays on ModelRunner, not the eager runner.
                ret = self.forward_split_prefill(
                    forward_batch,
                    reinit_attn_backend=reinit_attn_backend,
                    forward_count=split_forward_count,
                )
            elif (
                forward_batch.forward_mode.is_extend(include_draft_extend_v2=True)
                and not isinstance(self.prefill_cuda_graph_runner, EagerRunner)
                and self.prefill_cuda_graph_runner is not None
                and self.prefill_cuda_graph_runner.can_run_graph(forward_batch)
                and get_cp_strategy() is None
            ):
                category = (
                    "target_verify"
                    if forward_batch.forward_mode.is_target_verify()
                    else "extend"
                )
                # Prefill cuda graph (piecewise).
                kwargs = self._extend_forward_kwargs(forward_batch, pp_proxy_tensors)
                # TODO: device_timer.wrap is too broad here — it also includes
                # load_batch time. Move timing into the prefill cuda graph runner
                # to capture only the model.forward part.
                ctx = (
                    self.device_timer.wrap(metadata={"category": category})
                    if self.device_timer
                    else contextlib.nullcontext()
                )
                rank_ctx = (
                    self.prefill_rank_timer.wrap(metadata={"category": category})
                    if self.prefill_rank_timer
                    and forward_batch.forward_mode.is_plain_prefill()
                    else contextlib.nullcontext()
                )
                with ctx, rank_ctx:
                    ret = self.prefill_cuda_graph_runner.execute(
                        forward_batch, **kwargs
                    )
                can_run_graph = True
            else:
                # Eager: decode / extend / idle dispatched inside the runner.
                ret = self.eager_runner.execute(
                    forward_batch, pp_proxy_tensors=pp_proxy_tensors
                )

            if (
                forward_batch.global_num_tokens_cpu is not None
                and self.pp_group.is_last_rank
            ):
                forward_batch.post_forward_mlp_sync_batch(ret)

            return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)

    def _preprocess_logits(
        self, logits_output: LogitsProcessorOutput, sampling_info: SamplingBatchInfo
    ):
        # NOTE: In overlap mode, the function update_regex_vocab_mask (in sample)
        #       was executed after we processed last batch's results.

        # Calculate logits bias and apply it to next_token_logits.
        sampling_info.update_regex_vocab_mask()
        sampling_info.apply_logits_bias(logits_output.next_token_logits)

        # Release the vocab_mask GPU tensor immediately after it has been applied
        # to the logits. In overlap scheduling, the sampling_info (and its
        # vocab_mask) can be kept alive by the delay_sample_func closure and
        # batch_record_buf until the next iteration, causing a steady VRAM leak
        # when structured output (grammar) is used.
        sampling_info.vocab_mask = None

    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Sample and compute logprobs and update logits_output.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output

        Returns:
            A list of next_token_ids
        """
        if self.server_args.determinism_logits_dump_dir is not None:
            self._determinism_dump_logits(logits_output, forward_batch)
        self._preprocess_logits(logits_output, forward_batch.sampling_info)

        # Sample the next tokens
        next_token_ids = self.sampler(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
            # For prefill, we only use the position of the last token.
            (
                forward_batch.positions
                if forward_batch.forward_mode.is_decode()
                else forward_batch.seq_lens - 1
            ),
        )
        self.maybe_update_ngram_token_table(next_token_ids, forward_batch)
        return next_token_ids

    def _attach_layer_fingerprint(self) -> None:
        """Task #343: per-layer forward tap for the determinism harness.

        Two gates, both off by default, so neither the serving path nor the
        logits-only #124 harness changes: ``--determinism-logits-dump-dir``
        must be set AND ``SGLANG_DETERMINISM_LAYER_FINGERPRINT=1`` must be in
        the environment. See ``layer_fingerprint`` for what is recorded.
        """
        from sglang.srt.model_executor.layer_fingerprint import maybe_attach

        self._layer_fingerprint = maybe_attach(
            self.model, self.server_args.determinism_logits_dump_dir, self.tp_rank
        )

    def _determinism_dump_logits(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> None:
        """DEBUG surface for the #124 determinism harness (tests/determinism).

        Dumps this rank's next-token logits row for single-sequence batches as
        a sequentially numbered ``torch.save`` file, with the dtype exactly as
        served by the model / CUDA-graph output and captured BEFORE
        ``_preprocess_logits`` (the machine-zero byte-identity classes are
        dtype-strict). Guarded by ``--determinism-logits-dump-dir`` (default
        off, so the default serving path is untouched). Files are written
        atomically (tmp + rename) so a reader never sees a partial row.
        """
        logits = logits_output.next_token_logits
        if logits is None or logits.shape[0] != 1:
            return
        dump_dir = self.server_args.determinism_logits_dump_dir
        os.makedirs(dump_dir, exist_ok=True)
        step = getattr(self, "_determinism_dump_counter", 0)
        self._determinism_dump_counter = step + 1
        mode = "decode" if forward_batch.forward_mode.is_decode() else "extend"
        path = os.path.join(dump_dir, f"rank{self.tp_rank}_step{step:07d}.pt")
        tmp_path = path + ".tmp"
        torch.save(
            {
                "step": step,
                "mode": mode,
                "logits": logits[0].detach().to("cpu", copy=True),
            },
            tmp_path,
        )
        os.replace(tmp_path, path)

    def compute_logprobs_only(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> None:
        """
        Compute token_ids_logprobs without performing sampling.

        Optimized path for prefill-only requests that need token_ids_logprobs but don't
        require next token generation. Skips expensive sampling operations
        while still providing requested probability information.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output
        """
        if not forward_batch.token_ids_logprobs:
            return

        # Preprocess logits (same as in sample method)
        self._preprocess_logits(logits_output, forward_batch.sampling_info)

        # Delegate to sampler for logprob-only computation
        # This populates logits_output with requested token probabilities
        self.sampler.compute_logprobs_only(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
        )

    def save_remote_model(self, url: str):
        from sglang.srt.model_loader.loader import RemoteModelLoader

        logger.info(f"Saving model to {url}")
        RemoteModelLoader.save_model(self.model, self.model_config.model_path, url)

    def save_sharded_model(
        self, path: str, pattern: Optional[str] = None, max_size: Optional[int] = None
    ):
        from sglang.srt.model_loader.loader import ShardedStateLoader

        logger.info(
            f"Save sharded model to {path} with pattern {pattern} and max_size {max_size}"
        )
        ShardedStateLoader.save_model(self.model, path, pattern, max_size)

    def check_weights(self, action: str, allow_quant_error: bool = False):
        return self._weight_checker.handle(
            action=action, allow_quant_error=allow_quant_error
        )

    def update_weights_from_ipc(self, recv_req):
        """Update weights from IPC for checkpoint-engine integration."""
        try:
            from sglang.srt.checkpoint_engine.checkpoint_engine_worker import (
                SGLangCheckpointEngineWorkerExtensionImpl,
            )

            # Create a worker extension that integrates with SGLang's model
            worker = SGLangCheckpointEngineWorkerExtensionImpl(self)
            worker.update_weights_from_ipc(recv_req.zmq_handles)
            return True, "IPC weight update completed successfully"
        except ImportError as e:
            return False, f"IPC weight update failed: ImportError {e}"
        except Exception as e:
            logger.error(f"IPC weight update failed: {e}")
            return False, str(e)

    def prealloc_symmetric_memory_pool(self):
        # PyTorch mempools never de-fragment memory in OOM scenarios, so we need to pre-allocate a large chunk of memory to limit fragmentation.
        if (
            self.is_draft_worker
            or not self.server_args.enable_symm_mem
            or envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get() <= 0
        ):
            return

        # Memory allocation is tied to a cuda stream, use the forward stream
        with torch.get_device_module(self.device).stream(self.forward_stream):
            logger.info(
                f"Pre-allocating symmetric memory pool with {envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get()} GiB"
            )
            with use_symmetric_memory(get_tp_group()):
                torch.empty(
                    (envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get() * 1024 * 1024 * 1024,),
                    dtype=torch.uint8,
                    device=self.device,
                )

    def _maybe_rebalance_after_rank_fault(
        self,
        output: ModelRunnerOutput,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors],
        reinit_attn_backend: bool,
        split_forward_count: int,
    ) -> ModelRunnerOutput:
        elastic_ep_state = ElasticEPStateManager.instance()
        if elastic_ep_state is not None and not elastic_ep_state.is_active_equal_last():
            elastic_ep_state.snapshot_active_to_last()
            elastic_ep_state.sync_active_to_cpu()
            logging.info("EPLB due to rank faults")
            gen = self.eplb_manager.rebalance()
            while True:
                try:
                    next(gen)
                except StopIteration:
                    break
            output = self._forward_raw(
                forward_batch,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
        return output


def _model_load_weights_direct(model, named_tensors: List[Tuple[str, torch.Tensor]]):
    params_dict = dict(model.named_parameters())
    for name, tensor in named_tensors:
        default_weight_loader(params_dict[name], tensor)


def _unwrap_tensor(tensor, tp_rank, device):
    if isinstance(tensor, LocalSerializedTensor):
        tensor = tensor.get(tp_rank)
    return tensor.to(device)


def _build_step_span_name(forward_batch: ForwardBatch) -> str:
    """Build a profile-trace span name for one forward step."""
    mode = forward_batch.forward_mode
    bs = forward_batch.batch_size
    if mode == ForwardMode.EXTEND:
        ext_toks = forward_batch.extend_num_tokens or 0
        return f"step[EXTEND bs={bs} toks={ext_toks}]"
    return f"step[{mode.name} bs={bs}]"


@dataclass
class LocalSerializedTensor:
    """torch.Tensor that gets serialized by MultiprocessingSerializer (which only serializes a pointer and not the data).
    The i-th element in the list corresponds to i-th rank's GPU."""

    values: List[bytes]

    def get(self, rank: int):
        return MultiprocessingSerializer.deserialize(self.values[rank])
