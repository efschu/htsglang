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
"""A scheduler that manages a tensor parallel GPU worker."""

import dataclasses
import faulthandler
import logging
import os
import re
import signal
import sys
import time
from array import array
from collections import deque
from contextlib import contextmanager, nullcontext
from functools import partial
from http import HTTPStatus
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

from sglang.srt.utils.common import suppress_noisy_warnings  # isort: skip

suppress_noisy_warnings()

import psutil  # isort: skip
import setproctitle
import torch
import torch.distributed
from torch.cuda import Stream as CudaStream
from torch.distributed import barrier

from sglang.jit_kernel.ngram_embedding import update_token_table
from sglang.srt.configs.model_config import ModelConfig, ModelImpl, is_minimax_sparse
from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.srt.debug_utils import index_race_guard
from sglang.srt.debug_utils.pr_fix_toggle import maybe_revert_pr_fix
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.srt.disaggregation.encode_receiver import create_mm_receiver
from sglang.srt.disaggregation.prefill import (
    PrefillBootstrapQueue,
    SchedulerDisaggregationPrefillMixin,
    maybe_release_metadata_buffer,
)
from sglang.srt.disaggregation.utils import (
    DisaggregationMode,
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
    prepare_abort,
)
from sglang.srt.distributed import get_pp_group, get_world_group
from sglang.srt.distributed.collective_census import (  # noqa: E402
    census,
    census_enabled,
    census_heartbeat,
    census_interval,
)
from sglang.srt.distributed.device_communicators.barlink_capture_census import (  # noqa: E402
    capture_census,
    capture_census_enabled,
)
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.attention.mamba.ops import (
    initialize_mamba_selective_state_update_backend,
)
from sglang.srt.layers.dp_attention import (
    compute_dp_attention_world_info,
)
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.layers.quantization.unquant import initialize_bf16_gemm_config
from sglang.srt.lora.lora_drainer import LoRADrainer
from sglang.srt.lora.lora_overlap_loader import LoRAOverlapLoader
from sglang.srt.managers.admission_limiter import (
    AdmissionLimiter,
    AdmissionLimitError,
    replicated_pool_usage,
    resolve_admission_start,
    set_admission_limiter,
    throttle_before_retract,
)
from sglang.srt.managers.corridor_admission import (
    get_prefill_admission_gate,
    guard_prefill_admission,
)
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.wedge_recovery import drain_recovery_request
from sglang.srt.managers.io_struct import (
    AbortReq,
    ActiveRanksOutput,
    AddExternalCorpusReqInput,
    AddExternalCorpusReqOutput,
    AttachHiCacheStorageReqInput,
    AttachHiCacheStorageReqOutput,
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
    CheckWeightsReqInput,
    ClearHiCacheReqInput,
    ClearHiCacheReqOutput,
    CloseSessionReqInput,
    ConfigureLoggingReq,
    ContinueGenerationReqInput,
    DestroyWeightsUpdateGroupReqInput,
    DetachHiCacheStorageReqInput,
    DetachHiCacheStorageReqOutput,
    DumperControlReqInput,
    DumperControlReqOutput,
    ExpertDistributionReq,
    ExpertDistributionReqOutput,
    ExpertDistributionReqType,
    FlushCacheReqInput,
    FreezeGCReq,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    GetWeightsByNameReqInput,
    HealthCheckOutput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsSendGroupForRemoteInstanceReqOutput,
    InitWeightsUpdateGroupReqInput,
    KvReshardReqInput,
    KvReshardReqOutput,
    PhaseFlipReqInput,
    PhaseFlipReqOutput,
    SessionHandoverReqInput,
    SessionHandoverReqOutput,
    ListExternalCorporaReqInput,
    ListExternalCorporaReqOutput,
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterFromTensorsReqOutput,
    LoadLoRAAdapterReqInput,
    LoadLoRAAdapterReqOutput,
    OpenSessionReqInput,
    PauseGenerationReqInput,
    ProfileReq,
    ReleaseMemoryOccupationReqInput,
    RemoveExternalCorpusReqInput,
    RemoveExternalCorpusReqOutput,
    ResizeHiCacheStorageReqInput,
    ResizeHiCacheStorageReqOutput,
    ResumeMemoryOccupationReqInput,
    RpcReqInput,
    RpcReqOutput,
    SendWeightsToRemoteInstanceReqInput,
    SendWeightsToRemoteInstanceReqOutput,
    SessionCheckpointReqInput,
    SessionCheckpointReqOutput,
    SessionHandoverReqInput,
    SessionHandoverReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    ShutdownReq,
    SlowDownReqInput,
    SlowDownReqOutput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    UnloadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
    VramBudgetReqInput,
    VramBudgetReqOutput,
    sock_send,
)
from sglang.srt.managers.load_snapshot import create_load_snapshot_writer
from sglang.srt.managers.log_cycle_collapse import CycleCollapse
from sglang.srt.managers.min_free_slots_delayer import (
    MinFreeSlotsDelayer,
    resolve_min_free_slots,
)
from sglang.srt.managers.multimodal_processor import get_mm_processor, import_processors
from sglang.srt.managers.overlap_utils import (
    RelayPayload,
    decide_needs_confidence_relay,
    decide_needs_cpu_seq_lens,
    resolve_forward_inputs,
)
from sglang.srt.managers.phase_purity import (
    decode_blocked_here as phase_decode_blocked_here,
)
from sglang.srt.managers.phase_purity import (
    prefill_blocked_here as phase_prefill_blocked_here,
)

# #791b: imported AS A MODULE, not from-imported: the admission loop resolves
# `prefetch_ballot.prefetch_done_under_ballot` through the module's globals,
# which is what lets a test neuter exactly that one function in a child
# process without touching anything else (the instr-boot can-fail
# discipline).
from sglang.srt.managers import prefetch_ballot
from sglang.srt.managers.pp_admission_congruence import (
    PP_ADMISSION_VACUOUS_ROLLUP_EVERY,
    PPAdmissionCongruenceGuard,
    PPScheduleRefused,
    build_pp_admission_decision,
    pp_admission_verdict_is_vacuous,
    void_pp_admission_decision,
)
from sglang.srt.managers.prefill_delayer import (
    PrefillDelayer,
    PrefillDelayerSinglePassExecutor,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    MultimodalInputs,
    NextBatchPlan,
    Req,
    ScheduleBatch,
)
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    SchedulePolicy,
    truncation_align_admission_error,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.dp_attn import SchedulerDPAttnAdapter
from sglang.srt.managers.scheduler_components.flush_wrapper import SchedulerFlushWrapper
from sglang.srt.managers.scheduler_components.idle_sleeper import IdleSleeper
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
    create_admission_wedge_watchdog,
    create_scheduler_watchdog,
)
from sglang.srt.managers.scheduler_components.ipc_channels import SchedulerIpcChannels
from sglang.srt.managers.scheduler_components.kv_events_publisher import (
    SchedulerKvEventsPublisher,
)
from sglang.srt.managers.scheduler_components.load_inquirer import SchedulerLoadInquirer
from sglang.srt.managers.scheduler_components.logprob_result_processor import (
    SchedulerLogprobResultProcessor,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    RECORD_STEP_TIME,
    PrefillStats,
    SchedulerMetricsReporter,
)
from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import (
    NewTokenRatioTracker,
)
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    SchedulerPoolStatsObserver,
)
from sglang.srt.managers.scheduler_components.profiler_manager import (
    SchedulerProfilerManager,
)
from sglang.srt.managers.scheduler_components.request_receiver import (
    SchedulerRequestReceiver,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.managers.scheduler_input_blocker import SchedulerInputBlocker
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.managers.scheduler_recv_skipper import SchedulerRecvSkipper
from sglang.srt.managers.utils import (
    EmbeddingBatchResult,
    GenerationBatchResult,
    is_health_check_generate_req,
    validate_input_length,
)
from sglang.srt.mem_cache import kv_cache_builder
from sglang.srt.planner import transient_census as _transient_census
from sglang.srt.mem_cache.common import (
    evict_from_tree_cache,
    maybe_cache_unfinished_req,
    release_kv_cache,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.model_loader.utils import get_resolved_model_impl
from sglang.srt.multiplex.multiplexing_mixin import SchedulerMultiplexMixin
from sglang.srt.observability.metrics_collector import SchedulerMetricsCollector
from sglang.srt.observability.req_time_stats import (
    set_schedule_time_batch,
    set_time_batch,
)
from sglang.srt.observability.trace import process_tracing_init, trace_set_thread_info
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.platforms import current_platform
from sglang.srt.plugins import load_plugins
from sglang.srt.runtime_context import get_parallel, get_server_args
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.session.session_controller import SessionController
from sglang.srt.speculative.dflash_utils import validate_dflash_request
from sglang.srt.speculative.eagle_utils import get_draft_recurrent_hidden_state_spec
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    DynamicGradMode,
    configure_gc_logger,
    configure_logger,
    freeze_gc,
    get_available_gpu_memory,
    get_bool_env_var,
    get_int_env_var,
    is_cuda,
    is_hip,
    is_mps,
    kill_itself_when_parent_died,
    require_mlp_sync,
    set_gpu_proc_affinity,
    set_random_seed,
    suppress_other_loggers,
)
from sglang.srt.utils.common import (
    is_npu,
    kill_process_tree,
    process_group_is_confined_to_tree,
)
from sglang.srt.utils.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
)
from sglang.srt.utils.msgspec_utils import msgspec_to_builtins
from sglang.srt.utils.numa_utils import get_numa_node_if_available, numa_bind_to_node
from sglang.srt.utils.nvtx_utils import scheduler_nvtx_method
from sglang.srt.utils.tensor_bridge import use_mlx
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

if is_mps():
    CudaStreamContext = nullcontext
    from sglang.srt.hardware_backend.mlx.scheduler_mixin import SchedulerMlxOverlapMixin
else:
    from torch.cuda import StreamContext as CudaStreamContext

    class SchedulerMlxOverlapMixin:
        pass


logger = logging.getLogger(__name__)

#: #631: the phase policy's hold reasons carry live numbers, so the log
#: throttle keys on the reason with its digits removed -- the SHAPE of the
#: hold rather than its instantaneous value.
_POLICY_REASON_DIGITS = re.compile(r"[0-9]+(?:\.[0-9]+)?")

# Runtime HiCache resize requests are expressed in GiB.
_GIB = 1024**3

# #583 collective census: resolved once at import so the per-iteration
# tick is a bool read, not an environment lookup.
_CENSUS = census()
_CENSUS_ON = census_enabled()
_CENSUS_INTERVAL = census_interval()
_CENSUS_HEARTBEAT = census_heartbeat()

# Test retract decode for debugging purposes
TEST_RETRACT = envs.SGLANG_TEST_RETRACT.get()
TEST_RETRACT_INTERVAL = envs.SGLANG_TEST_RETRACT_INTERVAL.get()
TEST_RETRACT_NO_PREFILL_BS = envs.SGLANG_TEST_RETRACT_NO_PREFILL_BS.get()

_is_npu = is_npu()
_is_hip = is_hip()


def derive_enable_hicache_storage(server_args: ServerArgs) -> bool:
    """Whether the scheduler may drive the HiCache storage-prefetch path.

    Both halves are required, and the second one used to be missing.

    The prefetch path (``Scheduler._prefetch_kvcache``) reaches into
    ``tree_cache.hicache_storage_pass_prefix_keys`` and
    ``tree_cache.prefetch_from_storage``. Those attributes exist on the
    HiCache tree caches only (``hiradix_cache``, ``hi_mamba_radix_cache``,
    ``unified_radix_cache`` after ``init_hicache``) -- and which tree cache
    gets built is decided by ``enable_hierarchical_cache`` alone
    (``mem_cache/registry.py``), never by the storage backend.

    Keying the flag off ``hicache_storage_backend is not None`` ALONE
    therefore armed the prefetch path against a plain ``RadixCache`` /
    ``MambaRadixCache`` whenever a storage backend was configured while
    ``--enable-hierarchical-cache`` was off. The result was an
    ``AttributeError`` inside the first request of every scheduler process --
    all of them, at once, with a traceback that pointed at the cache instead
    of at the configuration.
    """
    return (
        server_args.hicache_storage_backend is not None
        and server_args.enable_hierarchical_cache
    )


def default_pp_micro_batch_size(
    *, max_running_requests: int, pp_size: int, enable_phase_flip: bool
) -> int:
    """The auto-computed ``pp_max_micro_batch_size``.

    Classic PP divides the concurrency cap by ``pp_size`` because the stages
    run micro-batches of one batch, so each stage may only hold its share.

    UNDER THE PHASE FLIP THAT DIVISION IS WRONG, and it is the binding cap on
    this deployment. Decode does not run in the PP layout at all -- it runs in
    the TP layout, which has no pipeline to divide by. Dividing anyway throttles
    the DECODE phase with a bound belonging to the PREFILL phase.

    Measured 2026-08-16 on max_running_requests=4, pp_size=3: the default was
    max(4 // 3, 1) = 1, and under a sustained depth-5 load decode concurrency
    never exceeded 2 -- 973 prefill rounds with 0 requests running, 792 with 1,
    48 with 2, against a configured ceiling of 4. Nothing was deadlocked; the
    scheduler was obeying a cap of one.

    The flip branch returns the full ``max_running_requests``. It is not
    unbounded: ``get_num_allocatable_reqs`` still mins this against the
    admission limiter, the request-slot pool, and the mamba/GDN state headroom,
    so the state pool remains the real ceiling -- this only stops a
    prefill-layout divisor from pre-empting all three.
    """
    if max_running_requests <= 0:
        return 1
    if enable_phase_flip:
        return max(int(max_running_requests), 1)
    return max(int(max_running_requests) // max(int(pp_size), 1), 1)


def _arriving_prefill_tokens(inflight, _already_queued=None) -> int:
    """#713: prompt tokens that have ARRIVED but are not yet on the queue.

    ``inflight`` is the raw ``recv_reqs`` list. It is heterogeneous -- abort
    messages, the policy's own flip arm, control traffic -- so only items that
    actually carry a prompt are counted. Counting a control message as work
    would arm a flip on nothing, which is the opposite defect to the one this
    fixes and would be harder to see.

    A just-received request is a ``TokenizedGenerateReqInput`` and carries
    ``input_ids`` (``io_struct.py:798``); the ``Req`` the scheduler builds
    later carries ``origin_input_ids``. Both are accepted because this helper
    runs on the boundary between them and reading only one field would make
    the count depend on where in the round it was called.

    Every access is guarded. This runs inside the admission path, and a probe
    that faults there kills the round it exists to inform -- the #715 lesson,
    where a diagnostic died inside the crash it was written to explain.
    """
    if not inflight:
        return 0
    total = 0
    for item in inflight:
        # #731 hardening: the docstring above asserts these have "ARRIVED but
        # are not yet on the queue", and nothing enforced it. The only
        # inflight-bearing call site runs pre-queue, so the invariant holds
        # today -- but an asserted-never-checked invariant is exactly how the
        # resident-vs-queued double count stayed silent, so it is checked now
        # rather than trusted. A request already queued is priced by the queue
        # term and must not be counted again here.
        if _already_queued is not None and id(item) in _already_queued:
            continue
        for field in ("input_ids", "origin_input_ids"):
            try:
                ids = getattr(item, field, None)
                if ids:
                    total += len(ids)
                    break
            except Exception:  # noqa: BLE001 - a probe must not break intake
                continue
    return total


class Scheduler(
    SchedulerDisaggregationDecodeMixin,
    SchedulerDisaggregationPrefillMixin,
    SchedulerMultiplexMixin,
    SchedulerPPMixin,
    SchedulerDllmMixin,
    SchedulerMlxOverlapMixin,
):
    """A scheduler that manages a tensor parallel GPU worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        dp_rank: Optional[int],
    ):
        self.is_initializing = True
        # init_soft_watchdog starts a daemon thread that reads these on its first tick.
        self.forward_ct: int = 0
        self.cur_batch_for_debug: Optional[ScheduleBatch] = None
        # kv-session-offload defaults BEFORE any watchdog thread can touch
        # is_fully_idle(); the manager itself is created late in __init__.
        self.kv_session_offload = None
        # #287 KV pressure ladder runtime: None on every default path; built
        # lazily on the first scheduler iteration when the flag is set.
        self.kv_pressure_runtime = None
        # #364 GDN resident-slot ladder executor: None on every default path;
        # built lazily on the first scheduler iteration when
        # --gdn-resident-state-slots is set.
        self.gdn_slot_executor = None
        # #363 regime observer (OBSERVE-ONLY, actuates nothing): None on every
        # default path. The mode is resolved once on the first iteration and
        # cached, so the default path costs one attribute compare per round.
        self.regime_observer = None
        self._regime_observer_mode = None
        # #363 phase 3: the boot stage table. Built once, lazily, next to the
        # observer -- the planner-solved set the runtime may SELECT from and
        # never adds to. None = no table, which is a real state (no probe, no
        # declared vectors) and is reported as such rather than as an empty
        # table pretending to be a full one.
        self.regime_stage_table = None
        # #297 phase-boundary KV reshard runtime: None on every default path;
        # built lazily on the first scheduler iteration when
        # --kv-reshard-vectors is set.
        self.kv_reshard_runtime = None
        # #631 phase flip: runtime built lazily on the first scheduler
        # iteration when --enable-phase-flip is set (the boot builder must
        # have installed phase_flip_stacks by then). The abort deferral
        # window exists from boot so abort routing never races the lazy
        # build; it only defers while ACTIVE (armed flip). active_stack
        # tracks the serving phase for the event-loop re-dispatch.
        self.phase_flip_runtime = None
        self.phase_flip_abort_window = None
        self.phase_flip_active_stack = "pp"
        if getattr(server_args, "enable_phase_flip", False):
            from sglang.srt.managers.phase_flip_runtime import (
                AbortDeferralWindow,
            )

            self.phase_flip_abort_window = AbortDeferralWindow()
        # #631 automatic phase policy: the thing that decides WHEN to flip.
        # Built from boot config on every rank (so the state objects exist
        # and the code path is uniform), but only the request-origin rank
        # ever evaluates it -- see maybe_arm_phase_policy. Config is built
        # even when disabled so a bad env value is found at boot rather
        # than on the first busy round.
        from sglang.srt.managers.phase_policy import config_from_env

        self.phase_policy_cfg = config_from_env(
            chunk_tokens=int(getattr(server_args, "chunked_prefill_size", 0) or 0),
            # #689: a decode window should open at the width the pools were
            # built for, not at whatever single carrier happened to finish
            # first. max_running_requests IS that width.
            formation_target=int(getattr(server_args, "max_running_requests", 0) or 0),
            enabled=(
                getattr(server_args, "enable_phase_flip", False)
                and getattr(server_args, "phase_flip_policy", "manual") == "auto"
            ),
            # #781: hand the whole ServerArgs over so the tuning knobs resolve
            # from FLAGS first and fall back to their deprecated env vars only
            # when a flag is unset. ServerArgs is pickled into this subprocess
            # (entrypoints/engine.py), so the flag value is already here -- no
            # env round-trip is needed to carry it across the fork.
            server_args=server_args,
        )
        self.phase_policy_state = None
        if self.phase_policy_cfg.enabled:
            from sglang.srt.managers.phase_policy import PhasePolicyState

            self.phase_policy_state = PhasePolicyState()
        # #631 STRICT PHASE PURITY. Resolved at boot, like the policy config
        # and for the same reason: an unusable value must be found here, not
        # on the first busy round. The pair check refuses the one
        # combination that deadlocks (purity enforced with no bounded PP
        # window to break a PP phase that may not decode and cannot admit).
        self._phase_purity = None
        if getattr(server_args, "enable_phase_flip", False):
            from sglang.srt.managers.phase_purity import (
                purity_from_server_args,
                validate_purity_policy_pair,
            )

            self._phase_purity = purity_from_server_args(server_args)
            validate_purity_policy_pair(self._phase_purity, self.phase_policy_cfg)
            # Tell the policy that the TP layout cannot prefill, so its
            # break-even N collapses to 0 in that direction. Without this
            # the two features contradict: purity refuses the prefill and
            # the policy refuses to leave TP for anything below N, so a
            # prompt smaller than N never runs at all (metal, 21:39:50Z --
            # a one-token health check wedged an otherwise idle server).
            if not self._phase_purity.prefill_allowed_in_tp():
                self.phase_policy_cfg = dataclasses.replace(
                    self.phase_policy_cfg, prefill_runs_in_tp=False
                )
            # The PP phase is drained when less than one chunk is left, and
            # the chunk size is a runtime fact, not a policy guess. Only fill
            # it in when the operator has not pinned one.
            if self.phase_policy_cfg.pp_exit_tokens <= 0:
                self.phase_policy_cfg = dataclasses.replace(
                    self.phase_policy_cfg,
                    pp_exit_tokens=int(
                        getattr(server_args, "chunked_prefill_size", 0) or 0
                    ),
                )
        # #261 live session handover runtime: None on every default path;
        # built lazily on the first /session_handover control request. The
        # admission hook in handle_generate_request is a no-op while this
        # is None or while no handover is active.
        self.session_handover_runtime = None
        # #410 session checkpoint runtime: None on every default path;
        # built lazily on the first /session/... control request, and
        # only when --enable-session-checkpoints is set.
        self.session_checkpoint_runtime = None
        # #330 VRAM dial / KV capacity runtime: None on every default path;
        # built lazily on the first scheduler iteration when
        # --enable-vram-dial is set.
        self.kv_capacity_runtime = None
        # colocated-congruent PD lane (#107): None on every default path.
        self.congruent_prefill_lane = None
        # Multi-group runtime (#274): in-process lanes over shared bytes.
        # Empty on every default path and on every non-shared rank.
        self.dual_group_lanes = []
        self.lane_share_meter = None
        self._lane_share_next_t = 0.0
        # Pairing objective (#274 slice D): None on every default path; the
        # serving grain publish in run_batch gates on it with one check.
        self.lane_pairing_signal = None
        self._kv_arrival_ct = 0
        self.init_soft_watchdog(server_args)

        # Parse args
        self.server_args = server_args
        self.nccl_port = port_args.nccl_port
        self.schedule_policy = server_args.schedule_policy
        self.enable_priority_scheduling = server_args.enable_priority_scheduling
        self.abort_on_priority_when_disabled = (
            server_args.abort_on_priority_when_disabled
        )
        self.schedule_low_priority_values_first = (
            server_args.schedule_low_priority_values_first
        )
        self.priority_scheduling_preemption_threshold = (
            server_args.priority_scheduling_preemption_threshold
        )
        self.enable_lora = server_args.enable_lora
        self.enable_lora_overlap_loading = server_args.enable_lora_overlap_loading
        self.max_loras_per_batch = server_args.max_loras_per_batch
        self.enable_overlap = not server_args.disable_overlap_schedule and not use_mlx()
        self.enable_overlap_mlx = not server_args.disable_overlap_schedule and use_mlx()
        self.enable_pdmux = server_args.enable_pdmux
        self.skip_tokenizer_init = server_args.skip_tokenizer_init
        self.stream_interval = server_args.stream_interval
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        # #631 Route A: speculation is a property of the TP DECODE phase.
        # A phase-flip instance boots in the PP prefill phase, where no
        # draft worker exists -- the draft workers take no pp_rank and
        # there is no PP-shaped draft stack -- so the boot-phase algorithm
        # is NONE and every spec-keyed branch in this class takes exactly
        # the path it takes on an instance without speculation. The
        # configured algorithm is kept aside here and swapped in at the
        # cutover, together with the draft worker that phase_flip_boot
        # builds on the flip's TP stack.
        self.flip_spec_algorithm = SpeculativeAlgorithm.from_string(None)
        if server_args.enable_phase_flip and not self.spec_algorithm.is_none():
            self.flip_spec_algorithm = self.spec_algorithm
            self.spec_algorithm = SpeculativeAlgorithm.from_string(None)
            logger.info(
                "#631 phase flip: speculation (%s) is armed for the TP "
                "decode phase; the PP prefill phase runs without a draft "
                "worker",
                server_args.speculative_algorithm,
            )
        # T156 stage 3/4: per-batch cross-algorithm switching (schedule and
        # auto/bandit modes). True => decode batches consult the meta-worker's
        # switch hook before prepare_for_decode, and DFLASH request validation
        # applies (any request may hit a DFLASH segment).
        from sglang.srt.speculative.cross_algo_utils import cross_switching_active

        self._cross_schedule_mode = cross_switching_active(server_args)
        self.page_size = server_args.page_size
        self.enable_hierarchical_cache = server_args.enable_hierarchical_cache
        self.enable_hicache_storage = derive_enable_hicache_storage(server_args)
        self.enable_decode_hicache = (
            server_args.disaggregation_decode_enable_radix_cache
            and self.enable_hierarchical_cache
        )
        self.max_recv_per_poll = envs.SGLANG_SCHEDULER_MAX_RECV_PER_POLL.get()
        self.enable_hisparse = server_args.enable_hisparse
        self.enable_dp_attention = server_args.enable_dp_attention
        self.enable_unified_memory = server_args.enable_unified_memory

        # Distributed rank info
        attn_tp_rank, attn_tp_size, attn_dp_rank, attn_dp_size = (
            compute_dp_attention_world_info(
                server_args.enable_dp_attention,
                tp_rank,
                server_args.tp_size,
                server_args.dp_size,
                server_args.attn_cp_size,
            )
        )
        self.ps = ParallelState(
            tp_rank=tp_rank,
            tp_size=server_args.tp_size,
            pp_rank=pp_rank,
            pp_size=server_args.pp_size,
            dp_rank=dp_rank,
            dp_size=server_args.dp_size,
            attn_tp_rank=attn_tp_rank,
            attn_tp_size=attn_tp_size,
            attn_cp_rank=attn_cp_rank,
            attn_cp_size=server_args.attn_cp_size,
            attn_dp_rank=attn_dp_rank,
            attn_dp_size=attn_dp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=server_args.ep_size,
            moe_dp_rank=moe_dp_rank,
            moe_dp_size=server_args.moe_dp_size,
            gpu_id=gpu_id,
        )

        # Init model configs
        self.init_model_config()

        # Init metrics stats
        self.init_metrics_collector(tp_rank, pp_rank, dp_rank)

        # Init inter-process communication
        self.init_ipc_channels(port_args)
        self.init_idle_sleeper()

        # Init ZBAL, switch allocator should before any torch alloc action
        self.init_zbal_on_npu()

        # Init PD-multiplexing context
        if self.enable_pdmux:
            self.init_pdmux()

        # Init tokenizer
        self.init_tokenizer()

        # Init moe config and GEMM config (FP8 GEMM, etc.)
        self.init_moe_gemm_config()

        # Init mamba backend
        self.init_mamba_backend()

        # Must precede init_model_worker: revert targets like _init_pools run during it,
        # so patching them afterwards is a no-op.
        maybe_revert_pr_fix()

        # Launch a model worker and draft model worker if using speculative decoding
        self.init_model_worker()

        if (t := envs.SGLANG_TEST_STUCK_SCHEDULER_INIT.get()) > 0:
            time.sleep(t)

        # Init cache and memory pool
        result = kv_cache_builder.build_kv_cache(
            server_args=self.server_args,
            model_config=self.model_config,
            tp_worker=self.tp_worker,
            page_size=self.page_size,
            spec_algorithm=self.spec_algorithm,
            attn_tp_cpu_group=self.attn_tp_cpu_group,
            tp_cpu_group=self.tp_cpu_group,
            attn_cp_cpu_group=self.attn_cp_cpu_group,
            enable_metrics=self.server_args.enable_metrics,
            enable_kv_cache_events=bool(
                self.server_args.kv_events_config
                and self.ps.pp_rank == 0
                and self.ps.attn_tp_rank == 0
                and self.ps.attn_cp_rank == 0
            ),
            ps=self.ps,
            tp_group=self.tp_group,
            pp_group=self.pp_group,
            enable_hierarchical_cache=self.enable_hierarchical_cache,
        )
        self.is_hybrid_swa = result.is_hybrid_swa
        self.is_hybrid_ssm = result.is_hybrid_ssm
        self.sliding_window_size = result.sliding_window_size
        self.full_tokens_per_layer = result.full_tokens_per_layer
        self.swa_tokens_per_layer = result.swa_tokens_per_layer
        self.req_to_token_pool = result.req_to_token_pool
        self.token_to_kv_pool_allocator = result.token_to_kv_pool_allocator
        self.disable_radix_cache = result.disable_radix_cache
        self.tree_cache = result.tree_cache

        # #677 PHASE 1: HERE, AND NOT BESIDE init_admission_limiter.
        # It reads the GDN slot pool off req_to_token_pool.mamba_allocator,
        # and that attribute is assigned four lines up -- AFTER
        # init_model_worker() has already returned. Sitting next to the
        # admission limiter (inside init_model_worker) put it before its own
        # input existed and killed every rank on the first boot that got far
        # enough to reach it:
        #     line 1420, in init_model_worker -> self.init_parked_decode_set()
        #     AttributeError: 'Scheduler' object has no attribute
        #     'req_to_token_pool'                     (metal, 2026-08-16 09:00)
        # The two earlier boots died at build_flip_draft_worker before ever
        # reaching the call, which is exactly why a stand-in unit test could
        # not have caught this: the ordering bug lives in the constructor, and
        # the tests bind the methods to an object that already has the fields.
        self.init_parked_decode_set()

        if (c := self.tp_worker.model_runner.canary_manager) is not None:
            c.attach_radix_cache(self.tree_cache)

        self.init_hisparse_coordinator()

        if (
            self.server_args.disaggregation_mode == "decode"
            and self.server_args.disaggregation_decode_enable_offload_kvcache
        ):
            self.decode_offload_manager = DecodeKVCacheOffloadManager(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                tp_group=(
                    self.attn_tp_cpu_group
                    if self.enable_dp_attention
                    else self.tp_cpu_group
                ),
                tree_cache=self.tree_cache,
                server_args=self.server_args,
            )
        else:
            self.decode_offload_manager = None

        # Register draft KV pool (when spec + HiCache co-enabled).
        kv_cache_builder.maybe_register_hicache_draft(
            tree_cache=self.tree_cache,
            draft_worker=self.draft_worker,
            spec_algorithm=self.spec_algorithm,
            server_args=self.server_args,
            enable_hierarchical_cache=self.enable_hierarchical_cache,
            page_size=self.page_size,
        )

        # Init running status
        self.init_running_status()

        # Init chunked prefill
        self.init_chunked_prefill()

        # Init diffusion LLM
        self.init_diffusion_llm()

        self.init_metrics_reporter(tp_rank, pp_rank, dp_rank)

        # Init schedule policy and new token estimation
        self.init_schedule_policy()

        # Init watchdog, memory saver, input blocker and recv skipper
        self.init_watch_dog_memory_saver_input_blocker()

        # Init profiler
        self.init_profiler()

        # Init prefill-decodedisaggregation
        self.init_disaggregation()

        # Init overlap schedule
        self.init_overlap()

        # Init Ngram Embedding
        self.maybe_init_ngram_embedding()

        # Init prefill kv split size when deterministic inference is enabled with various attention backends
        self.init_deterministic_inference_config()

        self.init_weight_updater()

        # Init request dispatcher
        self.init_request_dispatcher()

        # Init LoRA drainer for fair scheduling
        self.init_lora_drainer()

        # Init LoRA overlap loader
        self.init_lora_overlap_loader()

        # Init the grammar backend for constrained generation
        self.init_grammar_manager()

        self.maybe_init_scripted_scheduler_hook()

        self.init_request_receiver()

        self.init_dp_attn_adapter()

        self.init_pool_stats_observer()

        self.init_invariant_checker()

        self.init_kv_events_publisher()

        self.init_load_inquirer()

        self.init_output_streamer()

        # colocated-congruent PD topology (#107): the prefill lane rides
        # these decode processes with DECODE priority, computing with the
        # resident weight shards. Bind the model so the weight-sharing
        # invariant (one copy per card) is verified at the first lane tick
        # instead of assumed.
        if self.server_args.disaggregation_topology == "colocated-congruent":
            from sglang.srt.disaggregation.congruent_lane import (
                CongruentPrefillLane,
            )

            self.congruent_prefill_lane = CongruentPrefillLane(
                self.server_args.disaggregation_prefill_lane_interval
            )
            self.congruent_prefill_lane.bind_model(self.tp_worker.model_runner.model)

        # Multi-group runtime (#274) slice B: build the configured in-process
        # lanes (one PD lane on the shared rank). AFTER the serving group's
        # own init so the rank-local bring-up (complement load, lane pools,
        # rank-local graph capture) cannot interleave with any group
        # collective; the other ranks are already event-loop-ready and only
        # wait longer for the first broadcast.
        if getattr(self.server_args, "dual_group_lane", False):
            from sglang.srt.model_executor.dual_group_lane import (
                build_dual_group_lanes,
            )

            self.dual_group_lanes = build_dual_group_lanes(self)

            # Pairing objective of the two-class scheduler (#274 slice D).
            # Built whenever concurrent lanes exist so the runtime A/B toggle
            # (set_internal_state) has an object to flip; ACTIVE only when
            # --dual-group-lane-pairing is set. Inactive, the lane's job pick
            # stays the FIFO pop(0) -- pick() returns 0 unconditionally --
            # and the only cost on the serving path is one None-check per
            # run_batch.
            if self.dual_group_lanes and any(
                lane.concurrent for lane in self.dual_group_lanes
            ):
                from sglang.srt.model_executor.lane_pairing import (
                    PairingPolicy,
                    ServingGrainSignal,
                )

                sa = self.server_args
                self.lane_pairing_signal = ServingGrainSignal(
                    stale_ms=float(
                        getattr(sa, "dual_group_lane_pairing_stale_ms", 100.0)
                    ),
                    ms_per_row=float(
                        getattr(
                            sa,
                            "dual_group_lane_pairing_prefill_ms_per_row",
                            1.0,
                        )
                    ),
                )
                # Rows per sequence of a non-extend serving forward: a
                # target-verify runs num_draft_tokens rows per sequence, a
                # plain decode runs 1. Folded in here once so the launch-path
                # publish stays integer arithmetic.
                self._lane_pairing_rows_per_seq = (
                    int(getattr(sa, "speculative_num_draft_tokens", None) or 1)
                    if sa.speculative_algorithm is not None
                    else 1
                )
                self._lane_pairing_sat_rows = int(
                    getattr(sa, "dual_group_lane_pairing_sat_rows", 64)
                )
                for lane in self.dual_group_lanes:
                    if lane.concurrent:
                        lane.pairing_policy = PairingPolicy(
                            sat_rows=self._lane_pairing_sat_rows,
                            decode_step_rows=int(
                                getattr(
                                    sa,
                                    "dual_group_lane_pairing_decode_step_rows",
                                    12,
                                )
                            ),
                            max_defer_ms=float(
                                getattr(
                                    sa,
                                    "dual_group_lane_pairing_max_defer_ms",
                                    500.0,
                                )
                            ),
                            spec_steps=lane.spec_steps,
                            enabled=bool(getattr(sa, "dual_group_lane_pairing", False)),
                            signal=self.lane_pairing_signal,
                        )

            # Online card-equivalent estimator (#274 slice D, S1). Built only
            # where lanes exist, so the default path never allocates it and
            # never samples anything.
            from sglang.srt.model_executor.lane_share import LaneShareMeter

            share_window = float(
                getattr(self.server_args, "dual_group_lane_share_window_s", 1.0)
            )
            # 0 turns the instrument off completely: no sampling, no snapshot
            # in get_internal_state, no gauges. An instrument that cannot be
            # switched off cannot be ruled out as the cause of anything.
            if share_window > 0:
                self.lane_share_meter = LaneShareMeter(
                    window_s=share_window,
                    ema_s=float(
                        getattr(self.server_args, "dual_group_lane_share_ema_s", 1.0)
                    ),
                )
                # Standing gate (#284), one per lane, reporting only. Attached
                # here rather than built into the meter so that a boot without
                # a threshold carries no gate at all and the readout says so
                # by being empty.
                min_share = getattr(self.server_args, "dual_group_lane_share_min", None)
                if min_share is not None:
                    from sglang.srt.model_executor.lane_share import LaneShareGate

                    for lane in self.dual_group_lanes:
                        self.lane_share_meter.attach_gate(
                            LaneShareGate(
                                f"lane{lane.lane_id}",
                                float(min_share),
                                load=str(
                                    getattr(
                                        self.server_args,
                                        "dual_group_lane_share_load",
                                        "unspecified",
                                    )
                                ),
                                min_windows=int(
                                    getattr(
                                        self.server_args,
                                        "dual_group_lane_share_min_windows",
                                        5,
                                    )
                                ),
                            )
                        )

        # kv-session-offload (S1): FCFS host-spill of the youngest session
        # under KV pressure. Must init before the batch-result processor
        # (frozen dataclass takes the manager ref at construction).
        if self.server_args.enable_kv_session_offload:
            from sglang.srt.managers.kv_session_offload import (
                KVSessionOffloadManager,
            )

            self.kv_session_offload = KVSessionOffloadManager(self)

        self.init_batch_result_processor()

        self.is_initializing = False

    def init_zbal_on_npu(self):
        if _is_npu:
            from sglang.srt.hardware_backend.npu.utils import init_zbal

            if self.ps.pp_size > 1:
                logger.error("only zbal mix mode support pp_size > 1!")
            init_zbal(
                self.ps.tp_size, self.ps.gpu_id, self.ps.tp_rank
            )  # only switch allocator if is mix mode

    def init_model_config(self):
        self.model_config = ModelConfig.from_server_args(self.server_args)
        if _is_npu:
            # make sure the page size is not larger than block_size and chunked_prefill_size on NPU backend
            # the npu backend request the defined page size to be no larger than block_size and chunked_prefill_size
            from sglang.srt.dllm.config import DllmConfig

            self.dllm_config = (  # For diffusion LLM
                DllmConfig.from_server_args(self.server_args)
                if self.server_args.dllm_algorithm is not None
                else None
            )

    def init_metrics_collector(
        self, tp_rank: int, pp_rank: int, dp_rank: Optional[int]
    ) -> None:
        self.metrics_collector_context = SchedulerMetricsCollector.init_new(
            server_args=self.server_args,
            ps=self.ps,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank,
            enable_priority_scheduling=self.enable_priority_scheduling,
            enable_lora=self.enable_lora,
            enable_hierarchical_cache=self.enable_hierarchical_cache,
        )
        self.metrics_collector = self.metrics_collector_context.collector

    def init_ipc_channels(self, port_args: PortArgs):
        is_rank_zero = (
            self.ps.pp_rank == 0
            and self.ps.attn_tp_rank == 0
            and self.ps.attn_cp_rank == 0
        )
        self.ipc_channels = SchedulerIpcChannels.create(
            port_args=port_args,
            is_rank_zero=is_rank_zero,
            skip_tokenizer_init=self.server_args.skip_tokenizer_init,
            metrics_enabled=self.server_args.enable_metrics
            and (
                self.ps.attn_tp_rank == 0
                or self.server_args.enable_metrics_for_all_schedulers
            ),
            enable_scripted_runtime=envs.SGLANG_TEST_SCRIPTED_RUNTIME.get(),
        )

        self.load_snapshot_writer = None
        if not is_rank_zero:
            return

        dp_rank = self.ps.dp_rank if self.ps.dp_rank is not None else 0
        try:
            self.load_snapshot_writer = create_load_snapshot_writer(
                self.server_args,
                port_args,
                self.ps.dp_size,
                dp_rank,
                publish_interval=self.server_args.load_snapshot_publish_interval,
            )
        except Exception as e:
            logger.warning("load snapshot writer init failed: %s", e)

    def init_idle_sleeper(self) -> None:
        if (
            self.ps.pp_rank == 0
            and self.ps.attn_tp_rank == 0
            and self.ps.attn_cp_rank == 0
            and (
                self.server_args.sleep_on_idle
                # #547: same mechanism, reachable without the server arg.
                or envs.SGLANG_IDLE_BLOCKING_POLL.get()
            )
        ):
            self.idle_sleeper = IdleSleeper(
                sockets=[
                    self.ipc_channels.recv_from_tokenizer,
                    self.ipc_channels.recv_from_rpc,
                ],
            )
        else:
            self.idle_sleeper = None

    def publish_load_snapshot(self, force: bool = False):
        writer = self.load_snapshot_writer
        if writer is None:
            return
        if not force:
            writer.publish_counter += 1
            if writer.publish_counter < writer.publish_interval:
                return
        writer.publish_counter = 0
        try:
            writer.write(self.load_inquirer.get_loads())
        except Exception as e:
            logger.warning("load snapshot publish failed: %s", e)

    def init_tokenizer(self):
        server_args = self.server_args
        self.is_generation = self.model_config.is_generation

        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                    use_fast=not server_args.disable_fast_image_processor,
                    tokenizer_backend=server_args.tokenizer_backend,
                    model_name=server_args.model_path,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                    tokenizer_backend=server_args.tokenizer_backend,
                )

        # Load multimodal processor for M-RoPE fallback computation.
        self._mm_processor = None
        if self.model_config.is_multimodal and self.processor is not None:
            try:
                import_processors("sglang.srt.multimodal.processors")
                self._mm_processor = get_mm_processor(
                    self.model_config.hf_config,
                    server_args,
                    self.processor,
                    "default",
                    skip_mm_pool=True,
                )
            except Exception:
                logger.warning(
                    "Failed to load multimodal processor in scheduler; "
                    "M-RoPE fallback will not be available."
                )

        # Set reasoning_parser and think_end_id if --reasoning_parser is enabled
        if self.server_args.reasoning_parser and self.tokenizer:
            reasoning_parser = ReasoningParser(
                model_type=self.server_args.reasoning_parser,
                stream_reasoning=False,
                tokenizer=self.tokenizer,
            )
            self.model_config.think_end_id = self.tokenizer.encode(
                reasoning_parser.detector.think_end_token, add_special_tokens=False
            )[0]

    def init_mamba_backend(self) -> None:
        initialize_mamba_selective_state_update_backend(self.server_args)

    def init_moe_gemm_config(self):
        # For the MM models, check the text_config for MoE settings
        config_to_check = getattr(
            self.model_config.hf_config, "text_config", self.model_config.hf_config
        )

        # Different MoE architectures expose the per-token expert count under
        # different attribute names (e.g. Gemma4 uses ``top_k_experts``).
        moe_topk_attrs = (
            "num_experts_per_tok",
            "num_experts_per_token",
            "top_k_experts",
            "moe_top_k",
        )
        if any(hasattr(config_to_check, attr) for attr in moe_topk_attrs):
            initialize_moe_config(self.server_args)

        # Initialize GEMM-related configuration for FP8 and FP4 backends.
        initialize_fp8_gemm_config(self.server_args)
        initialize_fp4_gemm_config(self.server_args)
        initialize_bf16_gemm_config(self.server_args)

        # This must be called after initialize_moe_config
        self.require_mlp_sync = require_mlp_sync(self.server_args)

    def init_tp_model_worker(self):
        worker_kwargs = dict(
            server_args=self.server_args,
            gpu_id=self.ps.gpu_id,
            tp_rank=self.ps.tp_rank,
            moe_ep_rank=self.ps.moe_ep_rank,
            pp_rank=self.ps.pp_rank,
            attn_cp_rank=self.ps.attn_cp_rank,
            moe_dp_rank=self.ps.moe_dp_rank,
            dp_rank=self.ps.dp_rank,
            nccl_port=self.nccl_port,
        )

        # FIXME: move tp worker's init logic outside of the scheduler.
        if use_mlx():
            from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker

            self.tp_worker = MlxTpModelWorker(**worker_kwargs)
        else:
            from sglang.srt.managers.tp_worker import TpModelWorker

            self.tp_worker = TpModelWorker(**worker_kwargs)

    def maybe_init_draft_worker(self):
        if self.spec_algorithm.is_none():
            self.draft_worker = None
            self.external_corpus_manager = None
            return

        # Launch a draft worker for speculative decoding
        draft_worker_kwargs = dict(
            server_args=self.server_args,
            gpu_id=self.ps.gpu_id,
            tp_rank=self.ps.tp_rank,
            moe_ep_rank=self.ps.moe_ep_rank,
            nccl_port=self.nccl_port,
            target_worker=self.tp_worker,
            dp_rank=self.ps.dp_rank,
            attn_cp_rank=self.ps.attn_cp_rank,
            moe_dp_rank=self.ps.moe_dp_rank,
        )

        if self.server_args.speculative_draft_load_format is not None:
            self.server_args.override(
                "scheduler.draft_load_format",
                load_format=self.server_args.speculative_draft_load_format,
            )
            logger.info(
                f"Using draft model load_format: '{self.server_args.speculative_draft_load_format}'"
            )

        DraftWorkerClass = self.spec_algorithm.create_worker(self.server_args)
        self.draft_worker = DraftWorkerClass(**draft_worker_kwargs)

        if self.spec_algorithm.is_ngram():
            from sglang.srt.speculative.external_corpus_manager import (
                ExternalCorpusManager,
            )

            self.external_corpus_manager = ExternalCorpusManager(
                self.draft_worker,
                self.ipc_channels.send_to_tokenizer.send_output,
            )
        else:
            self.external_corpus_manager = None

    def init_target_memory_pool(self):
        """Allocate target KV cache pools if they have not been allocated yet."""
        if (
            self.tp_worker.model_runner.memory_pool_config is not None
            and self.tp_worker.model_runner.req_to_token_pool is not None
            and self.tp_worker.model_runner.token_to_kv_pool_allocator is not None
        ):
            return
        self.tp_worker.alloc_memory_pool()

    def init_memory_pools(self):
        """Allocate KV cache pools for target and draft workers."""
        self.init_target_memory_pool()
        if self.draft_worker is not None:
            pool, allocator = self.tp_worker.get_memory_pool()
            self.draft_worker.alloc_memory_pool(
                memory_pool_config=self.tp_worker.model_runner.memory_pool_config,
                req_to_token_pool=pool,
                token_to_kv_pool_allocator=allocator,
            )

    def _solo_draft_kv_pool_bytes(self) -> int:
        """Measured bytes of solo-resident draft KV pools on THIS rank.

        Walks the draft worker (including the cross-algorithm meta-worker's
        sub-workers) for draft model runners flagged as the solo host and
        sums their private KV pools. 0 on shadow ranks, on split placement,
        and without speculative decoding. Best-effort: sizing the balance
        post must never break boot."""
        total = 0
        try:
            roots = []
            dw = self.draft_worker
            if dw is not None:
                for attr in ("_primary", "_secondary"):
                    sub = getattr(dw, attr, None)
                    if sub is not None:
                        roots.append(sub)
                if not roots:
                    roots.append(dw)
            seen = set()
            for w in roots:
                dmr = getattr(w, "draft_model_runner", None)
                if (
                    dmr is None
                    or id(dmr) in seen
                    or not getattr(dmr, "is_draft_solo_host", False)
                ):
                    continue
                seen.add(id(dmr))
                pool = getattr(dmr, "token_to_kv_pool", None)
                if pool is None:
                    continue
                try:
                    v = pool.get_kv_size_bytes()
                    total += int(sum(v)) if isinstance(v, (tuple, list)) else int(v)
                except Exception:
                    continue
        except Exception:
            return 0
        return total

    def init_all_attention_backends(self):
        """Initialize attention backends for all workers."""
        # #631: the last moment before a backend caches attn_dcp_size /
        # attn_dcp_rank. If the DCP process group is not built yet the
        # backend caches dcp_size=1 without failing, which silently
        # disables the owner rule for the whole run. No-op whenever the
        # group matches the boot recipe -- see dcp_group_guard.
        from sglang.srt.distributed.dcp_group_guard import (
            _worker_page_size,
            assert_dcp_group_formed,
            assert_pd_decode_dcp_supported,
        )

        assert_dcp_group_formed(
            self.server_args, where="Scheduler.init_all_attention_backends"
        )
        assert_pd_decode_dcp_supported(
            self.server_args,
            # Resolved, not the CLI value: --page-size defaults to None and
            # is filled in per backend. Read from the WORKER's allocator,
            # which is the object disaggregation/decode.py later reads.
            # self.token_to_kv_pool_allocator is deliberately NOT used: it
            # is assigned after init_model_worker() returns, so at this
            # point in the boot it does not exist yet and reading it would
            # make this check silently unreachable.
            page_size=_worker_page_size(self.tp_worker),
        )
        self.tp_worker.init_attention_backends()
        if self.draft_worker is not None:
            self.draft_worker.init_attention_backends()

    def init_all_cuda_graphs(self):
        """Capture cuda graphs for all workers."""
        self.tp_worker.init_cuda_graphs()
        if self.draft_worker is not None:
            self.draft_worker.init_cuda_graphs()

    def init_model_worker(self):
        # Load model weights.
        self.init_tp_model_worker()
        self.maybe_init_draft_worker()

        # Prepare KV cache pools for all workers
        self.init_memory_pools()

        self.init_all_attention_backends()
        self.init_all_cuda_graphs()

        # #631 Route A: build the phase flip's SECONDARY (TP decode) stack
        # beside the fully-constructed primary PP stack -- weights arena,
        # TP pools, and the flip's ONLY decode-graph set (pin 2). Placed
        # BEFORE the post-capture pool resize below so the resize sees the
        # TP stack's VRAM as taken, never as free to grow into. Default
        # path: flag off, no import, nothing built.
        self.phase_flip_stacks = None
        if self.server_args.enable_phase_flip:
            from sglang.srt.managers.phase_flip_boot import (
                build_phase_flip_tp_stack,
            )

            self.phase_flip_stacks = build_phase_flip_tp_stack(self)

        # #797: hold a SEED vector to its claim, here and not earlier. Every
        # stack that can size a KV pool is built by this point -- the PP stack
        # from init_memory_pools() above and, under --enable-phase-flip, the TP
        # decode stack just above -- so "no measured vector ever superseded the
        # estimate" is a finished fact rather than a not-yet. Under the flip it
        # is the TP stack that carries the DCP layout, so a check placed after
        # init_memory_pools() alone would refuse every flip boot.
        from sglang.srt.distributed.utils import assert_seed_superseded

        assert_seed_superseded()

        model_runner = self.tp_worker.model_runner
        # post_capture_kv_active gate: the #330 vram-dial lane also sets the
        # pool's post_capture_active (its buffers are VMM-backed), but its
        # sizing is the boot fitted ceiling + runtime capacity commits, NOT
        # the dcp=1 post-capture free-memory resize below.
        if (
            model_runner.token_to_kv_pool.post_capture_active
            and model_runner.post_capture_kv_active
        ):
            model_runner.post_capture_resize_kv_pool()
        # Measured KV-budget correction (two-boot convergence, env-gated):
        # everything permanent is resident here (weights, pools, graphs,
        # workspaces; paused offload tags hold no pages) — measure the real
        # leftover and persist it for the next boot's budget. The scheduler
        # owns the draft worker, so it contributes the solo draft-KV pool
        # size (the cross gate's DFLASH pool on rank 0) as its own registry
        # post — the weight planner models it as a per-global-token cell,
        # not as fixed residency.
        model_runner.note_post_capture_leftover(
            draft_solo_pool_bytes=self._solo_draft_kv_pool_bytes()
        )
        # #485 residency census (env-gated, read-only): the same point, seen
        # from the CUT's side. note_post_capture_leftover above answers "how
        # much is left"; this answers "what is here, and who owns it", which
        # is what the cut gate has to price. Off by default and byte-identical
        # when off, so it can ride along on a corridor-measuring boot.
        from sglang.srt.planner.residency_census import log_residency_census

        log_residency_census(model_runner)

        # #695 host-shmem census. The residency census above answers "what is
        # on this CARD"; this answers the question that actually killed the
        # 2026-08-12 boot, which no ledger asked: what is this rank holding in
        # PAGE-LOCKED HOST memory. cgroup v2 files that memory under `file`,
        # so it never appears in an `anon` figure, and with no swap it cannot
        # be reclaimed -- 75 GiB of it across three ranks, nine cgroup OOM
        # kills, one of them presenting as a silent rank death. Unlike the two
        # censuses above this one is NOT env-gated: it is the line that has to
        # already be in the log when a rank dies with exit code -9.
        # The rank comes off the model_runner with getattr, exactly as the
        # residency census above does, and the whole call is guarded. The
        # first version read `self.tp_rank`, which the Scheduler does not
        # have: the AttributeError killed the scheduler, the launcher
        # SIGKILLed the process group, and a read-only instrument took the
        # boot down with it. A try/except INSIDE the census was not enough,
        # because the argument is evaluated out here.
        try:
            from sglang.srt.mem_ledger.host_shmem import log_host_shmem_census

            log_host_shmem_census(rank=getattr(model_runner, "tp_rank", None))
        except Exception as host_shmem_exc:  # noqa: BLE001
            logger.warning("#695 host-shmem census skipped: %s", host_shmem_exc)

        # #485 transient census (env-gated, read-only): the residency census
        # above is a snapshot AT REST, and a cut gate calibrated on at-rest
        # bytes alone certifies configurations that cannot serve. This arms
        # the per-load-state measurement of how far below at-rest each load
        # state actually pulls this rank. Armed HERE because this point is
        # the at-rest baseline: capture is done, no request has run.
        from sglang.srt.planner import transient_census

        if transient_census.census_enabled():
            try:
                import torch

                _free_at_rest, _ = torch.cuda.mem_get_info()
                transient_census.begin(
                    pp_rank=int(getattr(model_runner, "pp_rank", -1)),
                    gpu_name=torch.cuda.get_device_name(),
                    baseline_free_bytes=int(_free_at_rest),
                )
            except Exception as exc:  # pragma: no cover - instrument only
                logger.warning("transient census could not be armed: %s", exc)

        # Dispatch the model worker
        if self.spec_algorithm.is_none():
            self.model_worker = self.tp_worker
        else:
            self.model_worker = self.draft_worker

        # Get token and memory info from the model worker
        (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_queued_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            self.forward_stream,
            _,
            _,
            _,
        ) = self.tp_worker.get_worker_info()
        # #287: self.max_running_requests is now the CEILING the pools were
        # built for (ServerArgs rewrote the field; the resolver may have cut
        # it further for KV capacity). The limit that admission honours
        # floats below it and lives in the limiter.
        self.init_admission_limiter()
        # DFlash auto-enables the legacy formula; other workloads opt in via
        # --min-free-slots-delay. Built independently of the prefill delayer.
        self.min_free_slots_delayer: Optional[MinFreeSlotsDelayer] = None
        min_free_slots = resolve_min_free_slots(
            self.server_args.min_free_slots_delay,
            self.max_running_requests,
            is_dflash_family=self.spec_algorithm.is_dflash_family(),
        )
        if min_free_slots is not None:
            self.min_free_slots_delayer = MinFreeSlotsDelayer(
                min_free_slots=min_free_slots
            )
        if not get_server_args().pp_max_micro_batch_size:
            get_server_args().override(
                "scheduler.pp_max_micro_batch_size_default",
                pp_max_micro_batch_size=default_pp_micro_batch_size(
                    max_running_requests=self.max_running_requests,
                    pp_size=self.ps.pp_size,
                    enable_phase_flip=bool(
                        getattr(get_server_args(), "enable_phase_flip", False)
                    ),
                ),
            )

        self.tp_group = get_tp_group()
        self.tp_cpu_group = self.tp_group.cpu_group
        self.attn_tp_group = get_parallel().attn_tp_group
        self.attn_tp_cpu_group = self.attn_tp_group.cpu_group
        self.attn_cp_group = get_parallel().attn_cp_group
        self.attn_cp_cpu_group = self.attn_cp_group.cpu_group
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()

        # #791 PP ADMISSION UNIFORMITY. PP0-side memory of downstream
        # admission shortfalls (#630's learned-floor guard) -- PROCESS
        # LIFETIME state, deliberately not reset by init_pp_loop_state
        # (which can re-run mid-session, e.g. the phase-flip topology
        # swap): a floor learned from an earlier retract must survive a
        # loop re-init or it would just be re-taught the same way again.
        # None (not merely idle) when PP is not in use, so the guard costs
        # nothing on the non-PP default path. See
        # pp_admission_congruence.py's PPAdmissionCongruenceGuard docstring
        # for the well-founded strictly-decreasing termination argument.
        self._pp_admission_guard: Optional[PPAdmissionCongruenceGuard] = (
            PPAdmissionCongruenceGuard() if self.ps.pp_size > 1 else None
        )
        # Filled by the admission loop in _get_new_batch_prefill_raw below
        # (PP0 only) and drained every iteration by
        # scheduler_pp_mixin.py's _event_loop_pp_body, which stamps in the
        # real mb_id (unknown to this method) via dataclasses.replace.
        self._pp_admission_last_built_decision = None

        # NOTE: dp_tp_* are request/data-plane coordination groups (not tensor collectives).
        # When DP attention is enabled, scope to the attention-TP group; otherwise use
        # the base TP group. Entry rank is the local rank 0 in that group.
        # Use the CPU (gloo) group to broadcast VLM Python objects and avoid CUDA
        # stream/device coupling (#11910).
        self.dp_tp_group = (
            self.attn_tp_group if self.enable_dp_attention else self.tp_group
        )
        self.dp_tp_cpu_group = self.dp_tp_group.cpu_group

        # TODO(Jialin): Migrate pad_input_ids implementations to return array.
        self.pad_input_ids_func = self.tp_worker.get_pad_input_ids_func()
        set_random_seed(self.random_seed)

        # Print debug info
        avail_mem = get_available_gpu_memory(
            self.device, self.ps.gpu_id, empty_cache=False
        )
        if self.ps.tp_rank == 0:
            logger.info(
                f"max_total_num_tokens={self.max_total_num_tokens}, "
                f"chunked_prefill_size={self.server_args.chunked_prefill_size}, "
                f"max_prefill_tokens={self.max_prefill_tokens}, "
                f"max_running_requests={self.max_running_requests}, "
                f"context_len={self.model_config.context_len}, "
                f"{'available_cpu_mem' if self.device == 'cpu' else 'available_gpu_mem'}={avail_mem:.2f} GB"
            )

        if self.server_args.enable_metrics:
            self.metrics_collector.emit_constants(
                max_total_num_tokens=self.max_total_num_tokens,
                # TODO: max_running_requests_under_SLO has no setter — dead chain.
                max_running_requests_under_SLO=getattr(
                    self, "max_running_requests_under_SLO", None
                ),
                engine_startup_time=0.0,
                engine_load_weights_time=0.0,
                page_size=self.page_size,
                num_pages=self.max_total_num_tokens // self.page_size,
                context_len=self.model_config.context_len,
                startup_available_gpu_memory_gb=avail_mem,
            )

        # #603b: LAST in this method, after every worker, pool, backend and
        # graph exists. The warmup ends in a group barrier, so it must sit at a
        # point every rank reaches exactly once with the model fully built.
        self.warm_sampling_backend()

    def warm_sampling_backend(self):
        """#603b: make the sampling JIT kernels resident BEFORE serving starts.

        The flashinfer sampling module is built lazily on its first call, which
        lands inside ``Sampler.forward`` -- i.e. inside a serving forward, on a
        rank its peers are waiting for. On a cold cache that is a 60-90 s nvcc
        compile, and it produced seven ``Bar1CollectiveAborted`` crashes on
        2026-08-06 (py-spy caught two ranks in ``run_ninja`` and one on the
        build's ``FileLock``). See ``layers/sampler_warmup.py`` for the full
        mechanism, including why the heterogeneous arch-keyed cache makes the
        two same-arch ranks serialise against each other.

        Placed here rather than inside the Sampler because this point is
        rank-uniform and unconditional: every rank reaches ``init_model_worker``
        exactly once, after capture and before the event loop, so the barrier
        inside the warmup pairs up by construction.
        """
        from sglang.srt.layers.sampler_warmup import warm_sampling_backend_kernels

        # NOT `self.tp_group`: that attribute is assigned later in __init__ than
        # `init_model_worker()` runs, so reading it here raised
        # `AttributeError: 'Scheduler' object has no attribute 'tp_group'` and
        # killed the boot (observed on-card, window boot 438456). The group
        # itself exists by now -- the model worker is built and its collectives
        # have run -- so resolve it from the registry the same way __init__
        # later does, instead of depending on attribute-assignment order.
        try:
            group = get_tp_group()
        except Exception as exc:  # noqa: BLE001 - warmup must not kill a boot
            logger.warning(
                "sampling-backend JIT warmup: no TP group available (%s: %s); "
                "warming without the rendezvous barrier.",
                type(exc).__name__,
                exc,
            )
            group = None

        status = warm_sampling_backend_kernels(
            self.server_args.sampling_backend,
            self.tp_worker.device,
            tp_group=group,
        )
        logger.info("sampling-backend JIT warmup: %s", status)

    def init_hisparse_coordinator(self) -> None:
        self.hisparse_coordinator: Optional[HiSparseCoordinator] = None
        if not self.enable_hisparse:
            return

        # Coordinator was created inside ModelRunner.initialize() before CUDA graph capture.
        self.hisparse_coordinator = self.tp_worker.model_runner.hisparse_coordinator
        self.hisparse_coordinator.set_decode_producer_stream(self.forward_stream)

    def init_running_status(self):
        # Set by the ShutdownReq handler to break the event loop for graceful shutdown.
        self.gracefully_exit = False
        self.waiting_queue: List[Req] = []
        # The running decoding batch for continuous batching
        self.running_batch: ScheduleBatch = ScheduleBatch(reqs=[], batch_is_full=False)
        # The current forward batch
        self.cur_batch_for_debug: Optional[ScheduleBatch] = None
        # The last forward batch
        self.last_batch: Optional[ScheduleBatch] = None
        self.forward_ct = 0
        # #699: the admission-wedge clock. Seeded to "now" so an idle box at
        # boot reads as "no progress yet since start" rather than a false
        # infinite age; stamped again only when a request's first output
        # token is committed (see note_first_token_progress /
        # SchedulerBatchResultProcessor.process_batch_result_prefill), never
        # on a forward pass -- that is the exact signal #699 proved blind.
        self.last_first_token_progress_time: float = time.perf_counter()
        # #739: the SECOND progress signal. The first-token clock alone cannot
        # separate a wedge from a mega-prefill -- a 500k-token backlog chunking
        # at chunked_prefill_size produces no first token for minutes while
        # prefill runs the whole time, with the same queued>0/running==0 shape.
        # Stamped when a chunked request retires a middle chunk, so it is an
        # EVENT clock, not a delta of the pending counter (#731 shows that
        # counter double-billed; a detector keyed to it inherits the noise).
        self.last_prefill_progress_time: float = time.perf_counter()
        self.return_health_check_ipcs: Deque[Optional[str]] = deque()
        self.flush_wrapper = SchedulerFlushWrapper(
            flush_cache=self.flush_cache,
            is_fully_idle=self.is_fully_idle,
            ipc_channels=self.ipc_channels,
        )
        self.session_controller = SessionController(self.tree_cache)
        self.forward_sleep_time = None
        self._engine_paused = False

    def init_chunked_prefill(self):
        self.chunked_prefill_size = self.server_args.chunked_prefill_size
        uses_transformers_backend = (
            get_resolved_model_impl(self.model_config) == ModelImpl.TRANSFORMERS
        )
        if (
            self.chunked_prefill_size is not None
            and self.chunked_prefill_size > 0
            and self.model_config.is_multimodal
            and uses_transformers_backend
        ):
            logger.warning(
                "Chunked prefill is disabled for multimodal models with the "
                "Transformers backend to avoid partial multimodal chunk mismatches."
            )
            self.chunked_prefill_size = None
        elif self.chunked_prefill_size is not None and self.chunked_prefill_size <= 0:
            self.chunked_prefill_size = None
        self.chunked_req = None
        self._pending_chunked_abort_req = None
        self.is_mixed_chunk = (
            self.chunked_prefill_size is not None
            and self.server_args.enable_mixed_chunk
        )

        # Init the dynamic chunking predictor for PP
        self.enable_dynamic_chunking = (
            self.server_args.enable_dynamic_chunking and self.ps.pp_size > 1
        )
        if self.enable_dynamic_chunking:
            try:
                self.profile_and_init_predictor()
            except Exception as e:
                logger.warning(
                    f"[PP Dynamic Chunk] Failed to profile prefill latency: {e}. "
                    "Dynamic chunking will be disabled."
                )
                self.enable_dynamic_chunking = False

    def init_metrics_reporter(
        self, tp_rank: int, pp_rank: int, dp_rank: Optional[int]
    ) -> None:
        # Override point for deployments that need a specialized reporter.
        self.metrics_reporter = SchedulerMetricsReporter(
            scheduler=self,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank,
            metrics_collector_context=self.metrics_collector_context,
            metrics_collector=self.metrics_collector,
        )

    def init_schedule_policy(self):
        # Init schedule policy and new token estimation
        self.policy = SchedulePolicy(
            self.schedule_policy,
            self.tree_cache,
            self.enable_hierarchical_cache,
            self.enable_priority_scheduling,
            self.schedule_low_priority_values_first,
            enable_fast_lane=self.server_args.enable_fast_lane,
            fast_lane_priority=self.server_args.fast_lane_priority,
            fast_lane_heavy_aging_ms=self.server_args.fast_lane_heavy_aging_ms,
        )
        self.prefill_delayer: Optional[PrefillDelayer] = None
        self.max_prefill_bs: int = 0
        if self.server_args.enable_prefill_delayer:
            if self.server_args.disaggregation_mode == "decode":
                logger.info(
                    "Ignoring --enable-prefill-delayer on decode engine "
                    "(no prefill scheduling path; delayer would be a no-op)."
                )
            else:
                self.prefill_delayer = PrefillDelayer(
                    dp_size=self.ps.dp_size,
                    attn_tp_size=self.ps.attn_tp_size,
                    cpu_group=self.tp_cpu_group,
                    device_group=self.tp_group.device_group,
                    server_args=self.server_args,
                    metrics_collector=(
                        self.metrics_collector
                        if self.metrics_reporter.enable_metrics
                        else None
                    ),
                    max_delay_passes=self.server_args.prefill_delayer_max_delay_passes,
                    token_usage_low_watermark=self.server_args.prefill_delayer_token_usage_low_watermark,
                    device=self.tp_group.device,
                )

        # NOTE: preemption is enabled by default for priority scheduling.
        self.enable_priority_preemption = (
            self.enable_priority_scheduling
            and not self.server_args.disable_priority_preemption
        )

        self.new_token_ratio_tracker = NewTokenRatioTracker.from_server_args(
            self.server_args
        )

    def init_soft_watchdog(self, server_args: ServerArgs):
        if (x := server_args.soft_watchdog_timeout) is not None:
            self.soft_watchdog = create_scheduler_watchdog(
                self, watchdog_timeout=x, soft=True
            )

    def note_first_token_progress(self, ts: Optional[float] = None) -> None:
        """#699: stamp the moment ANY request reached its first output token.

        This is the clock the admission-wedge detector reads (queue age vs
        progress), never forward_ct: chunked prefill can advance forward_ct
        for tens of seconds while zero requests progress, which is exactly
        the wedge shape #699 exists to catch. Call this ONLY at the instant a
        request's first output token is committed
        (SchedulerBatchResultProcessor.process_batch_result_prefill), never
        from a forward pass alone.
        """
        self.last_first_token_progress_time = (
            ts if ts is not None else time.perf_counter()
        )

    def note_prefill_progress(self, ts: Optional[float] = None) -> None:
        """Stamp the #739 prefill-progress clock.

        Call this when a chunked request retires a middle chunk -- real
        forward progress on a prompt that has not reached a first token yet.
        Never call it from a bare forward pass: that is the signal #699 proved
        blind, and reusing it here would re-import the blindness.
        """
        self.last_prefill_progress_time = ts if ts is not None else time.perf_counter()

    def init_watch_dog_memory_saver_input_blocker(self):
        # Start watchdog thread
        self.watchdog = create_scheduler_watchdog(
            self, watchdog_timeout=self.server_args.watchdog_timeout
        )

        # #699: log-only admission-wedge watchdog, wired to the real
        # first-token-progress clock above (queue age vs progress -- see
        # invariant_checker.create_admission_wedge_watchdog for why this is
        # not the same signal as the forward_ct watchdog just started).
        self.admission_wedge_watchdog = create_admission_wedge_watchdog(self)

        # Init memory saver, profiler and metric stats
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.server_args.enable_memory_saver
        )

        # Init recv skipper and input blocker
        self.recv_skipper = SchedulerRecvSkipper.maybe_create(self.server_args)
        self.input_blocker = (
            SchedulerInputBlocker(noop=self.ps.attn_tp_rank != 0)
            if get_bool_env_var("SGLANG_ENABLE_COLOCATED_BATCH_GEN")
            else None
        )

        # Configure GC logger
        if envs.SGLANG_LOG_GC.get():
            configure_gc_logger()

    def init_disaggregation(self):
        self.mm_receiver = None
        self.disagg_prefill_bootstrap_queue = None
        self.disagg_prefill_inflight_queue = None
        self.disagg_decode_prealloc_queue = None
        self.disagg_decode_transfer_queue = None

        self.disaggregation_mode = DisaggregationMode(
            self.server_args.disaggregation_mode
        )
        self.transfer_backend = TransferBackend(
            self.server_args.disaggregation_transfer_backend
        )

        # todo: should we fix this when enabling mtp or it doesn't matter since we only enable mtp in decode node thus we don't transfer draft kvs between P and D?
        draft_token_to_kv_pool = kv_cache_builder.get_draft_kv_pool(
            draft_worker=self.draft_worker,
            spec_algorithm=self.spec_algorithm,
            server_args=self.server_args,
        )

        if self.spec_algorithm.carries_draft_hidden_states():
            # `draft_runner` aliases `draft_runner_list[0]` in the multi-layer
            # worker, so a single accessor covers both shapes.
            draft_runner = self.draft_worker.draft_worker.draft_runner
            disagg_hidden_size, disagg_hidden_states_dtype = (
                get_draft_recurrent_hidden_state_spec(draft_runner)
            )
        else:
            disagg_hidden_size = 16  # minimal padding size for RDMA
            disagg_hidden_states_dtype = torch.float32

        if (
            self.disaggregation_mode == DisaggregationMode.DECODE
        ):  # *8 headroom for MiniMax-M3; *2 for other models.
            buffer_multiplier = (
                8 if is_minimax_sparse(self.model_config.hf_config) else 2
            )
            buffer_size = (self.req_to_token_pool.size) * buffer_multiplier
            self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                buffer_size
            )
            self.disagg_metadata_buffers = MetadataBuffers(
                buffer_size,
                hidden_size=disagg_hidden_size,
                hidden_states_dtype=disagg_hidden_states_dtype,
                custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
            )

            # The decode requests polling kv cache
            self.disagg_decode_transfer_queue = DecodeTransferQueue(
                gloo_group=self.attn_tp_cpu_group,
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                tp_rank=self.ps.tp_rank,
                metadata_buffers=self.disagg_metadata_buffers,
                scheduler=self,
                tree_cache=self.tree_cache,
            )

            # The decode requests pending for pre-allocation
            self.disagg_decode_prealloc_queue = DecodePreallocQueue(
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                draft_token_to_kv_pool=draft_token_to_kv_pool,
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                metadata_buffers=self.disagg_metadata_buffers,
                scheduler=self,
                transfer_queue=self.disagg_decode_transfer_queue,
                tree_cache=self.tree_cache,
                gloo_group=self.attn_tp_cpu_group,
                tp_rank=self.ps.tp_rank,
                tp_size=self.ps.tp_size,
                dp_size=self.server_args.dp_size,
                gpu_id=self.ps.gpu_id,
                bootstrap_port=self.server_args.disaggregation_bootstrap_port,
                # Used for _check_if_req_exceed_kv_capacity, which compares a
                # request LENGTH against it -- the global span, same as the
                # prefill queue reads off max_token_pool_size (#346).
                max_total_num_tokens=self._global_kv_capacity_tokens(),
                pp_rank=self.ps.pp_rank,
                num_reserved_decode_tokens=self.server_args.num_reserved_decode_tokens,
                transfer_backend=self.transfer_backend,
            )

        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            # *2 for the headroom.
            buffer_size = self.max_running_requests * 2
            self.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(
                buffer_size
            )
            self.disagg_metadata_buffers = MetadataBuffers(
                buffer_size,
                hidden_size=disagg_hidden_size,
                hidden_states_dtype=disagg_hidden_states_dtype,
                custom_mem_pool=self.token_to_kv_pool_allocator.get_kvcache().maybe_get_custom_mem_pool(),
            )

            self.disagg_prefill_bootstrap_queue = PrefillBootstrapQueue(
                token_to_kv_pool=self.token_to_kv_pool_allocator.get_kvcache(),
                draft_token_to_kv_pool=draft_token_to_kv_pool,
                req_to_metadata_buffer_idx_allocator=self.req_to_metadata_buffer_idx_allocator,
                metadata_buffers=self.disagg_metadata_buffers,
                tp_rank=self.ps.tp_rank,
                tp_size=self.ps.tp_size,
                gpu_id=self.ps.gpu_id,
                bootstrap_port=self.server_args.disaggregation_bootstrap_port,
                gloo_group=self.attn_tp_cpu_group,
                max_total_num_tokens=self.max_total_num_tokens,
                scheduler=self,
                pp_rank=self.ps.pp_rank,
                pp_size=self.ps.pp_size,
                transfer_backend=self.transfer_backend,
            )
            # The prefill requests that are in the middle of kv sending
            self.disagg_prefill_inflight_queue: List[Req] = []

            self.enable_staging = envs.SGLANG_DISAGG_STAGING_BUFFER.get()

        # Init mm receiver for EPD disaggregation mode
        if (
            self.server_args.language_only
            and self.server_args.encoder_transfer_backend
            in ["zmq_to_scheduler", "mooncake"]
        ):
            self.mm_receiver = create_mm_receiver(
                self.server_args,
                dtype=self.model_config.dtype,
                hf_config=self.model_config.hf_config,
                pp_rank=self.ps.pp_rank,
                tp_rank=self.ps.tp_rank,
                tp_group=self.tp_group,
                scheduler=self,
            )

    def init_overlap(self):
        self.device_module = torch.get_device_module(self.device)

        # FutureMap is always-on: input_ids relay used in both modes.
        # Workers without the spec_v2_attn_backends override fall back to
        # target-only so the helper still produces a safe decision (no
        # accidental opt-out for unaudited shapes).
        if self.draft_worker is not None:
            attn_backends = getattr(
                self.draft_worker,
                "spec_v2_attn_backends",
                (self.tp_worker.model_runner.attn_backend,),
            )
        else:
            attn_backends = (self.tp_worker.model_runner.attn_backend,)
        needs_cpu_seq_lens = decide_needs_cpu_seq_lens(self.server_args, attn_backends)
        needs_confidence_relay = decide_needs_confidence_relay(self.server_args)
        self.future_map = self.spec_algorithm.create_future_map(
            self.device,
            self.req_to_token_pool,
            needs_cpu_seq_lens=needs_cpu_seq_lens,
            needs_confidence_relay=needs_confidence_relay,
        )

        self._confidence_budget_prepare = None
        if (
            needs_confidence_relay
            and self.enable_overlap
            and self.draft_worker is not None
        ):
            self._confidence_budget_prepare = (
                self.draft_worker.get_confidence_budget_prepare()
            )

        if use_mlx():
            # MLX uses its own overlap loop and does not create CUDA streams,
            # but the normal non-overlap scheduler path still relays decode
            # input IDs through FutureMap.
            self.result_queue: Deque = deque()
            return

        # forward_stream_ctx / copy_stream are also used by PP (non-overlap)
        # via scheduler_pp_mixin; init unconditionally to match main.
        self.forward_stream_ctx: CudaStreamContext = self.device_module.stream(
            self.forward_stream
        )
        self.copy_stream: CudaStream = self.device_module.Stream()
        self.copy_stream_ctx: CudaStreamContext = self.device_module.stream(
            self.copy_stream
        )

        # DECOUPLE S4b (kv-session-offload, gated SGLANG_KVSO_DECOUPLE, default
        # OFF -> byte-identical): a SECOND forward stream for the concurrent
        # spill lane. The device decode forward keeps forward_stream (comm A);
        # a due spill tick is issued on spill_stream (comm B, routed via the
        # spill batch's kv_session_spill_tick flag inside the attention
        # backend). Both descend from schedule_stream each iteration so their
        # GPU kernels overlap while the SM-idle H2D of the spill lane hides
        # behind the device compute. The stash + the spill lane's own depth-1
        # overlap result queue live here so the two result streams never
        # cross-contaminate (device tokens -> device reqs, spill tokens ->
        # spilled reqs). Allocated once at init (never inside graph capture),
        # mirroring the _sess_copy_stream lease. None/empty when the flag is
        # OFF so nothing extra is reserved and the default path is untouched.
        self._pending_spill_batch: Optional[ScheduleBatch] = None
        self._spill_result_queue: Deque = deque()
        self.spill_stream: Optional[CudaStream] = None
        self.spill_stream_ctx: Optional[CudaStreamContext] = None
        from sglang.srt.managers.kv_session_offload import spill_decouple_enabled

        if spill_decouple_enabled():
            self.spill_stream = self.device_module.Stream()
            self.spill_stream_ctx = self.device_module.stream(self.spill_stream)
            logger.info(
                "kv-session-offload DECOUPLE: second forward stream "
                "'spill_stream' leased for the concurrent spill lane (S4b)."
            )

        if not self.enable_overlap:
            return

        self.batch_record_buf = [None] * 2
        self.batch_record_ct = 0
        # DECOUPLE S4b: the 2-slot keep-alive ring assumes ONE forward per
        # iteration (2 slots cover the 1-iteration result-processing offset).
        # The concurrent spill lane adds a SECOND forward per iteration, which
        # would advance the shared ring twice and evict the device batch's
        # snapshot a full iteration early (GPU-tensor use-after-free during
        # result processing). Give the spill lane its OWN ring, swapped in
        # around the spill run_batch (_dispatch_concurrent_spill), so the two
        # lanes never share this mutable resource. Allocated only under overlap
        # (the ring is an overlap-only asset); harmless when decoupling is off.
        self._spill_record_buf = [None] * 2
        self._spill_record_ct = 0

    def maybe_init_ngram_embedding(self):
        self.use_ngram_embedding = self.tp_worker.model_config.use_ngram_embedding
        if self.use_ngram_embedding:
            self.token_table = self.tp_worker.model_runner.token_table
            hf_config = self.tp_worker.model_config.hf_config
            self.ngram_embedding_n = hf_config.ngram_embedding_n
            self.ngram_embedding_k = hf_config.ngram_embedding_k

    def _maybe_prepare_ngram_embedding(
        self, batch: Optional[ScheduleBatch]
    ) -> Optional[ScheduleBatch]:
        """Fill the token table for ngram embedding before a forward pass."""
        if batch is None or not self.use_ngram_embedding:
            return batch
        batch.ne_token_table = self.token_table
        if batch.forward_mode == ForwardMode.EXTEND:
            all_tokens = []
            column_starts = []
            request_lengths = []
            for req in batch.reqs:
                start = len(req.prefix_indices)
                end = start + req.extend_range.length
                fill_ids = req.origin_input_ids + req.output_ids
                if start == 0:
                    tokens = fill_ids[start:end]
                    column_starts.append(0)
                elif start < self.ngram_embedding_n:
                    tokens = fill_ids[0:end]
                    column_starts.append(0)
                else:
                    # Prepend n-1 tokens before prefix_len for n-gram context
                    tokens = fill_ids[start - self.ngram_embedding_n + 1 : end]
                    column_starts.append(start - self.ngram_embedding_n + 1)
                all_tokens.extend(tokens)
                request_lengths.append(len(tokens))
            dtype = self.token_table.dtype
            device = self.token_table.device
            update_token_table(
                ne_token_table=self.token_table,
                tokens=torch.tensor(all_tokens, dtype=dtype, device=device),
                row_indices=batch.req_pool_indices,
                column_starts=torch.tensor(
                    column_starts, dtype=torch.int32, device=device
                ),
                req_lens=torch.tensor(
                    request_lengths, dtype=torch.int32, device=device
                ),
                ignore_tokens=None,
            )
            # Mark the chunked (not-yet-finished) prefill request so sample()
            # skips writing its pseudo next-token into the ngram token table.
            # Use self.chunked_req identity (not req.is_chunked) to avoid
            # overlap-scheduling timing issues.
            if self.chunked_req is not None:
                skip_token_table_update = [
                    req is self.chunked_req for req in batch.reqs
                ]
                batch.ne_skip_token_table_update = (
                    torch.tensor(
                        skip_token_table_update, dtype=torch.bool, device=device
                    )
                    if any(skip_token_table_update)
                    else None
                )
        return batch

    def init_deterministic_inference_config(self):
        """Initialize deterministic inference configuration for different attention backends."""
        self.truncation_align_size = None
        if self.server_args.enable_deterministic_inference:
            backend_sizes = {
                "flashinfer": ("SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE", 4096),
                "triton": ("SGLANG_TRITON_PREFILL_TRUNCATION_ALIGN_SIZE", 4096),
            }
            env_var, default_size = backend_sizes.get(
                self.server_args.attention_backend, (None, None)
            )
            self.truncation_align_size = (
                get_int_env_var(env_var, default_size) if env_var else None
            )

        # --mamba-checkpoint-interval: while the interval fits inside the
        # chunk budget, chunked prefill steps are clipped to end on the
        # checkpoint grid (a resumed prefix is on the grid, so aligning
        # every chunk keeps all snapshot positions absolute). #750: a
        # SPARSE grid (interval an exact multiple of the chunk budget,
        # validated) is NOT folded in -- full chunk ends land on the grid
        # every (interval // chunk)-th step by divisibility, the retention
        # rule drops the ends between, and the chunk budget stays what the
        # user set. One shared rule decides which case this boot is
        # (mamba_ckpt_utils.checkpoint_truncation_align), so the
        # validation's promise and this fold cannot drift apart.
        from sglang.srt.mem_cache.mamba_ckpt_utils import (
            checkpoint_truncation_align,
        )

        ckpt_interval = self.server_args.mamba_checkpoint_interval
        sources = []
        if self.truncation_align_size is not None:
            sources.append(
                f"--enable-deterministic-inference on the "
                f"{self.server_args.attention_backend} backend"
            )
        self.truncation_align_size, _ckpt_folded = checkpoint_truncation_align(
            self.truncation_align_size,
            ckpt_interval,
            self.server_args.chunked_prefill_size,
        )
        if _ckpt_folded:
            # Part of the alignment -> part of the C30 sources list. A
            # sparse interval is deliberately absent here: it contributes
            # nothing to the alignment, and naming it would make a C30
            # refusal blame a flag that did not cause it.
            sources.append(f"--mamba-checkpoint-interval={ckpt_interval}")
        elif ckpt_interval is not None:
            logger.info(
                "mamba checkpoint interval %d exceeds the chunk budget %s "
                "(#750 sparse grid): every %d-th full chunk end lands on "
                "the grid and is anchored; ends between are not cached. "
                "The chunk budget and truncation alignment are unchanged.",
                ckpt_interval,
                self.server_args.chunked_prefill_size,
                ckpt_interval // self.server_args.chunked_prefill_size,
            )

        # C30: refuse a chunk budget that can never satisfy the alignment just
        # derived. Checked HERE because this is the only point where BOTH
        # contributors (deterministic inference and the mamba checkpoint grid)
        # have been folded into the final align size -- server_args cannot see
        # the lcm without restating it. Reading chunked_prefill_size off
        # server_args rather than self: init_chunked_prefill() runs later in
        # __init__, and its only transform is to map <= 0 to None, which this
        # helper already treats as "chunked prefill off".
        _align_err, _align_warn = truncation_align_admission_error(
            self.server_args.chunked_prefill_size,
            self.server_args.page_size,
            self.truncation_align_size,
            sources,
            dynamic_chunking=bool(
                getattr(self.server_args, "enable_dynamic_chunking", False)
            ),
        )
        if _align_err is not None:
            raise ValueError(_align_err)
        if _align_warn is not None:
            # ERROR level: the failure it describes is a total admission stall
            # with no other symptom, so this line is the only warning anyone
            # would ever get.
            logger.error("kv/prefill admission: %s", _align_warn)

    def init_request_dispatcher(self):
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.handle_generate_request),
                (TokenizedEmbeddingReqInput, self.handle_embedding_request),
                (BatchTokenizedGenerateReqInput, self.handle_batch_generate_request),
                (BatchTokenizedEmbeddingReqInput, self.handle_batch_embedding_request),
                (FlushCacheReqInput, self.flush_wrapper.handle),
                (KvReshardReqInput, self.handle_kv_reshard),
                (PhaseFlipReqInput, self.handle_phase_flip),
                (SessionHandoverReqInput, self.handle_session_handover),
                (SessionCheckpointReqInput, self.handle_session_checkpoint),
                (VramBudgetReqInput, self.handle_vram_budget),
                (ClearHiCacheReqInput, self.clear_hicache_storage_wrapped),
                (AttachHiCacheStorageReqInput, self.attach_hicache_storage_wrapped),
                (DetachHiCacheStorageReqInput, self.detach_hicache_storage_wrapped),
                (ResizeHiCacheStorageReqInput, self.resize_hicache_storage_wrapped),
                (AbortReq, self.abort_request),
                (OpenSessionReqInput, self.open_session),
                (CloseSessionReqInput, self.close_session),
                (
                    UpdateWeightFromDiskReqInput,
                    self.weight_updater.update_weights_from_disk,
                ),
                (
                    InitWeightsUpdateGroupReqInput,
                    self.weight_updater.init_weights_update_group,
                ),
                (
                    DestroyWeightsUpdateGroupReqInput,
                    self.weight_updater.destroy_weights_update_group,
                ),
                (
                    InitWeightsSendGroupForRemoteInstanceReqInput,
                    self.init_weights_send_group_for_remote_instance,
                ),
                (
                    SendWeightsToRemoteInstanceReqInput,
                    self.send_weights_to_remote_instance,
                ),
                (
                    UpdateWeightsFromDistributedReqInput,
                    self.weight_updater.update_weights_from_distributed,
                ),
                (
                    UpdateWeightsFromTensorReqInput,
                    self.weight_updater.update_weights_from_tensor,
                ),
                (
                    UpdateWeightsFromIPCReqInput,
                    self.weight_updater.update_weights_from_ipc,
                ),
                (
                    GetWeightsByNameReqInput,
                    self.weight_updater.get_weights_by_name,
                ),
                (
                    ReleaseMemoryOccupationReqInput,
                    self.weight_updater.release_memory_occupation,
                ),
                (
                    ResumeMemoryOccupationReqInput,
                    self.weight_updater.resume_memory_occupation,
                ),
                (
                    CheckWeightsReqInput,
                    self.weight_updater.check_weights,
                ),
                (SlowDownReqInput, self.slow_down),
                (
                    ProfileReq,
                    lambda req: self.profiler_manager._profile(req),
                ),
                (FreezeGCReq, self.handle_freeze_gc),
                (ShutdownReq, self.handle_shutdown),
                (GetInternalStateReq, self.get_internal_state),
                (SetInternalStateReq, self.set_internal_state),
                (RpcReqInput, self.handle_rpc_request),
                (ExpertDistributionReq, self.expert_distribution_handle),
                (LoadLoRAAdapterReqInput, self.load_lora_adapter),
                (
                    LoadLoRAAdapterFromTensorsReqInput,
                    self.load_lora_adapter_from_tensors,
                ),
                (UnloadLoRAAdapterReqInput, self.unload_lora_adapter),
                (PauseGenerationReqInput, self.pause_generation),
                (ContinueGenerationReqInput, self.continue_generation),
                (ConfigureLoggingReq, self.configure_logging),
                (DumperControlReqInput, self.handle_dumper_control),
                (AddExternalCorpusReqInput, self.add_external_corpus),
                (
                    RemoveExternalCorpusReqInput,
                    self.remove_external_corpus,
                ),
                (
                    ListExternalCorporaReqInput,
                    self.list_external_corpora,
                ),
            ]
        )

    def _uniform_timeout_ballot(self, local_verdicts: List[bool]) -> List[bool]:
        """MAX-reduce a positional wall-clock timeout verdict over the TP group.

        #610, the rank-local-test-before-a-group-collective family. A timeout
        verdict is built from TWO rank-local quantities: the entry timestamp
        (``req_time_stats.set_wait_queue_entry_time`` / ``set_forward_entry_time``
        stamp ``time.perf_counter()`` on the rank that processes the request)
        and ``time.perf_counter()`` read at THIS rank's own point in its own
        scheduler iteration. Two ranks straddling the deadline therefore
        disagree, and both callers act on that disagreement in ways that split
        the group -- see their own comments.

        The fix is the established family pattern (#580/#607-E): the rank-local
        verdict does not decide, it only casts a ballot, and participation in
        the ballot is unconditional. MAX (logical OR), not MIN: the timeout is a
        LIVENESS promise to the client, so the first rank to observe the
        deadline decides for the group. MAX is also monotone -- once a request
        is voted out it cannot be voted back in by a slower rank on a later
        iteration, which a MIN ballot would allow and which would make the
        abort flap.

        COST: one MAX ``all_reduce`` of a ``len(local_verdicts)``-element int64
        CPU tensor on ``tp_cpu_group`` (gloo), i.e. NOT the device group and not
        BAR1. Both callers are gated on ``SGLANG_REQ_*_TIMEOUT > 0``, which
        defaults to -1, so the default path takes NO collective at all and is
        byte-identical.

        PAYLOAD LENGTH is rank-uniform by construction: both callers index a
        collection that is already replicated across the group (the waiting
        queue is fed by the rank-0 broadcast in ``_broadcast_reqs_across_ranks``
        and drained by decisions that are themselves rank-uniform; the running
        batch is built from a rank-uniform plan). A length that nonetheless
        diverged would be a pre-existing composition split, and gloo reports it
        as the named ``op.preamble.length`` mismatch rather than hanging.
        """
        if not local_verdicts:
            return local_verdicts
        grp = getattr(self, "tp_cpu_group", None)
        if grp is None or torch.distributed.get_world_size(grp) <= 1:
            return local_verdicts
        t = torch.tensor([int(v) for v in local_verdicts], dtype=torch.int64)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX, group=grp)
        return [bool(v) for v in t.tolist()]

    def _abort_on_running_timeout(self, running_batch: ScheduleBatch):
        # NOTE: this should be called before a batch is launched.
        timeout_s = envs.SGLANG_REQ_RUNNING_TIMEOUT.get()
        if timeout_s <= 0:
            return
        if running_batch.is_empty():
            return

        # #610: the wall-clock verdict below is RANK-LOCAL (see
        # `_uniform_timeout_ballot`). Acting on it directly sets `to_finish` on
        # a rank-dependent subset of the running batch; `check_finished`
        # promotes that to `finished_reason` (schedule_batch.py:1516) and the
        # request leaves the batch on some ranks only. The next iteration then
        # builds decode batches of DIFFERENT composition per rank, and the
        # per-layer TP collectives inside the forward are entered with
        # mismatched shapes. Vote first, act on the group's verdict.
        deadline = time.perf_counter() - timeout_s
        timed_out = self._uniform_timeout_ballot(
            [
                0 < req.time_stats.forward_entry_time < deadline
                for req in running_batch.reqs
            ]
        )
        for req, expired in zip(running_batch.reqs, timed_out):
            if expired and not req.finished():
                req.to_finish = FINISH_ABORT(
                    "Request running timeout reached.", HTTPStatus.SERVICE_UNAVAILABLE
                )

    def get_init_info(self) -> Dict[str, Any]:
        """Return scheduler initialization info for handshake.

        This method provides the initialization info needed by the tokenizer manager
        and other components to verify the scheduler is ready.
        """
        result_dict = {
            "status": "ready",
            "max_total_num_tokens": self.max_total_num_tokens,
            "max_req_input_len": self.max_req_input_len,
        }

        return result_dict

    def release_host_resources(self) -> None:
        # Release pinned host buffers in userspace on graceful shutdown; see
        # HostKVCache.destroy. Called from run_scheduler_process's finally.
        if self.hisparse_coordinator is not None:
            self.hisparse_coordinator.destroy()

    def run_event_loop(self) -> None:
        """Run the scheduler's event loop.

        Sets up the schedule stream and dispatches to the appropriate event loop.
        The event loop blocks until shutdown.
        """
        if use_mlx():
            # MLX overlap uses mx.async_eval for CPU/GPU overlap,
            # not PyTorch MPS streams.
            dispatch_event_loop(self)
            return

        self.schedule_stream = self.device_module.Stream(priority=0)
        if self.device == "cpu":
            self.schedule_stream.synchronize = lambda: None  # No-op for CPU
        # The global WAR barrier fences the scheduler's next shared-buffer write
        # on the previous forward's read of the unified memory pool.
        self._war_barrier_enabled = is_cuda() or envs.SGLANG_ENABLE_WAR_BARRIER.get()
        with self.device_module.StreamContext(self.schedule_stream):
            dispatch_event_loop(self)

    def _apply_war_barrier(self):
        # Wait for the prev forward to finish reading the shared buffers this
        # iter's schedule will overwrite. Fast path: wait on the read-done event
        # the forward published after its snapshot (non-spec: decode graph;
        # spec: draft_extend), then clear it. Else fall back to whole-forward
        # wait_stream.
        if not self._war_barrier_enabled:
            return
        # #616 bisection arm: SGLANG_WAR_BARRIER_FASTPATH=0 forces the
        # CONSERVATIVE barrier (wait on the whole forward stream) instead of the
        # read-done event. If the crash survives the fast path but disappears
        # here, the event is published before the forward's last read of the
        # shared pool and the scheduler's next write races that read.
        runner = self.model_worker.war_fastpath_runner
        ev = runner.war_fastpath_read_done_event
        if ev is not None and not envs.SGLANG_WAR_BARRIER_FASTPATH.get():
            runner.war_fastpath_read_done_event = None
            ev = None
        if ev is not None:
            self.schedule_stream.wait_event(ev)
            runner.war_fastpath_read_done_event = None
        else:
            self.schedule_stream.wait_stream(self.forward_stream)

    @DynamicGradMode()
    def event_loop_normal(self):
        """A normal scheduler loop."""
        while True:
            if self.gracefully_exit:
                break

            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue

            # Multi-group runtime (#274): one lane tick per loop iteration,
            # BEFORE the serving batch (PD priority: in a conflict the lane
            # wins, the serving group is the work-conserving scavenger).
            # Rank-local, no collectives -- the other ranks simply see this
            # rank join the next collective later.
            self._dual_group_lane_tick()

            # Get the next batch to run
            plan = self.get_next_batch_to_run(
                running_batch=self.running_batch, last_batch=self.last_batch
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            self.cur_batch_for_debug = batch

            # Launch the current batch
            if batch:
                # #547: a batch ran this iteration -> the idle poll ladder goes
                # back to its zero-poll rung, so the first iterations after a
                # loaded phase never block. One None-test per forward.
                if self.idle_sleeper is not None:
                    self.idle_sleeper.reset()
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # When the server is idle, do self-check and re-init some states.
                self.on_idle()

            # DECOUPLE S4b: dispatch a due spill tick concurrently on
            # spill_stream / comm B (non-overlap: run + process synchronously).
            # No-op unless decoupling is on and a tick is due.
            if self._pending_spill_batch is not None:
                spill_batch = self._pending_spill_batch
                self._pending_spill_batch = None
                self._dispatch_concurrent_spill(spill_batch)

            # Update last_batch
            self.last_batch = batch
            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.invariant_checker.self_check_during_busy()

    @DynamicGradMode()
    def event_loop_overlap(self):
        """A scheduler loop that overlaps the CPU processing and GPU computation."""
        self.result_queue: Deque[
            Tuple[ScheduleBatch, Union[GenerationBatchResult, EmbeddingBatchResult]]
        ] = deque()

        def pop_and_process():
            # Process the results of the last batch
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        while True:
            if self.gracefully_exit:
                break

            # Receive requests
            recv_reqs = self.request_receiver.recv_requests()
            self.process_input_requests(recv_reqs)
            if self._engine_paused:
                continue

            # Multi-group runtime (#274): serial lane tick, PD priority
            # (see event_loop_normal). Runs on the default stream before the
            # overlap machinery touches forward_stream this iteration.
            self._dual_group_lane_tick()

            # #616 instrument: consume/stage the index-race counters. Sync-free
            # (staged D2H + event query), no-op unless the guard is armed.
            index_race_guard.poll()

            self._apply_war_barrier()

            # Get the next batch to run
            plan = self.get_next_batch_to_run(
                running_batch=self.running_batch, last_batch=self.last_batch
            )
            self.running_batch = plan.running_batch
            batch = plan.batch_to_run
            self.cur_batch_for_debug = batch
            disable_overlap_for_batch = self.is_disable_overlap_for_batch(
                batch, last_batch=self.last_batch
            )

            # If we do not need to overlap the current batch with the last batch,
            # we can process the last batch immediately.
            if disable_overlap_for_batch:
                pop_and_process()
                # Opportunistic flush at the disable_overlap sync boundary:
                # forward_stream is idle (prev forward drained, next not launched),
                # so `_flush`'s non-urgent guard compacts freely. Sync-free, best-effort.
                if self.enable_unified_memory:
                    try:
                        self.token_to_kv_pool_allocator.flush_opportunistic()
                    except Exception:
                        pass

            # Launch the current batch
            if batch:
                # #547: see event_loop_normal -- a loaded iteration resets the
                # idle poll ladder to its zero-poll rung.
                if self.idle_sleeper is not None:
                    self.idle_sleeper.reset()
                batch_result = self.run_batch(batch)
                self.result_queue.append((batch.copy(), batch_result))
            else:
                batch_result = None

            # DECOUPLE S4b: dispatch a due spill tick CONCURRENTLY on
            # spill_stream / comm B, overlapping the device forward just
            # enqueued on forward_stream / comm A. No-op (stash is None) unless
            # decoupling is on and a tick is due, so the default path is
            # untouched. Issued after the device forward so each communicator
            # sees a rank-uniform, ordered op stream (device ops then spill ops).
            if self._pending_spill_batch is not None:
                spill_batch = self._pending_spill_batch
                self._pending_spill_batch = None
                self._dispatch_concurrent_spill(spill_batch)

            # Process the last batch
            if self.last_batch:
                if not disable_overlap_for_batch:
                    pop_and_process()
            elif batch is None:
                # When the server is idle, do self-check and re-init some states
                self.on_idle()

            # Run sample of the current batch
            # It depends on the result of the last batch (e.g., grammar), so we run it after the last batch is processed.
            if self.is_generation:
                self.launch_batch_sample_if_needed(batch_result, batch)

            # Update last_batch
            self.last_batch = batch

            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():
                self.invariant_checker.self_check_during_busy()

    def is_disable_overlap_for_batch(
        self, batch: ScheduleBatch, last_batch: Optional[ScheduleBatch]
    ) -> bool:
        # For two consecutive prefill batches, we disable overlap to improve the TTFT of the first batch.
        # This might slightly hurt the throughput, so we use an environment variable to control it.
        # In DP attention mode, use the globally synchronized is_extend_in_batch
        # so all DP ranks make the same overlap decision (avoiding deadlock).
        # In non-DP mode, use the local forward_mode directly.
        if self.require_mlp_sync:
            is_extend = lambda b: b and b.is_extend_in_batch
        else:
            is_extend = lambda b: b and b.forward_mode.is_extend()

        batch_is_extend = is_extend(batch)
        last_batch_is_extend = is_extend(last_batch)

        disable_overlap_for_batch = (
            envs.SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP.get()
            and batch_is_extend
            and last_batch_is_extend
        )

        # We do not support overlap + spec + grammar yet,
        # so we need to turn off overlap for this batch.
        # TODO(lsyin): support overlap + spec + grammar
        need_grammar_sync = (
            batch
            and not batch.spec_algorithm.is_none()
            and batch.has_grammar
            and batch.forward_mode.is_decode()
            and len(self.result_queue) > 0
        )

        return disable_overlap_for_batch or need_grammar_sync

    @scheduler_nvtx_method("scheduler.process_input_requests")
    def process_input_requests(self, recv_reqs: List):
        now = time.monotonic()
        # #800: run any wedge-recovery request the watchdog thread posted, on
        # THIS thread. This function is the one place every loop family reaches
        # once per iteration -- event_loop_normal, event_loop_overlap and the
        # three PP loops via _pp_forward_and_process_input_requests -- which is
        # what a phase-flip boot needs, since it re-dispatches between
        # event_loop_pp and event_loop_normal per phase and a drain point in
        # only one of them would be inert in the other. Costs one attribute
        # read and one int compare on a boot that has never wedged; see
        # managers/wedge_recovery.py for why the actuator must not run on the
        # watchdog thread (it silenced PP0's detector on 2026-08-22).
        drain_recovery_request(self)
        self.session_controller.maybe_reap(now)
        for recv_req in recv_reqs:
            # Skip health check when server is busy — ongoing requests already carry health info.
            if is_health_check_generate_req(recv_req) and not self.is_fully_idle(
                for_health_check=True
            ):
                self.return_health_check_ipcs.append(
                    getattr(recv_req, "http_worker_ipc", None)
                )
                continue

            output = self._request_dispatcher(recv_req)
            if output is not None:
                if not isinstance(output, RpcReqOutput):
                    self.ipc_channels.send_to_tokenizer.send_output(output, recv_req)
                else:
                    if self.ipc_channels.recv_from_rpc is not None:
                        sock_send(self.ipc_channels.recv_from_rpc, output)

        self.flush_wrapper.check_pending()
        if self.external_corpus_manager is not None:
            self.external_corpus_manager.check_pending_load()

    def init_profiler(self) -> None:
        self.profiler_manager = SchedulerProfilerManager(
            ps=self.ps,
            dp_tp_cpu_group=self.dp_tp_cpu_group,
            get_forward_ct=lambda: self.forward_ct,
        )

    def init_weight_updater(self) -> None:
        self.weight_updater = SchedulerWeightUpdaterManager(
            tp_worker=self.tp_worker,
            draft_worker=self.draft_worker,
            tp_cpu_group=self.tp_cpu_group,
            memory_saver_adapter=self.memory_saver_adapter,
            flush_cache=self.flush_cache,
            is_fully_idle=self.is_fully_idle,
            scheduler=self,
            metrics_collector=self.metrics_collector,
        )

    def init_lora_drainer(self) -> None:
        if self.server_args.lora_drain_wait_threshold > 0.0:
            self.lora_drainer = LoRADrainer(
                self.server_args.max_loras_per_batch,
                self.server_args.lora_drain_wait_threshold,
            )
        else:
            self.lora_drainer = None

    def init_lora_overlap_loader(self) -> None:
        if self.enable_lora_overlap_loading:
            self.lora_overlap_loader = LoRAOverlapLoader(
                self.tp_worker.model_runner.lora_manager
            )

    def init_grammar_manager(self) -> None:
        self.grammar_manager = GrammarManager(self)

    def maybe_init_scripted_scheduler_hook(self) -> None:
        if envs.SGLANG_TEST_SCRIPTED_RUNTIME.get():
            from sglang.test.scripted_runtime.scheduler_hook import (
                ScriptedSchedulerHook,
            )

            self.scripted_scheduler_hook = ScriptedSchedulerHook(
                scheduler=self,
                tokenizer_recv_proxy=self.ipc_channels.recv_from_tokenizer,
            )
        else:
            self.scripted_scheduler_hook = None

    def _build_pp_chain_receiver(self):
        """#631: the single owner of this rank's request-chain receive
        stream, or None to keep the unmodified upstream path.

        Built ONLY with the phase flip on and only on a PP stage that has
        an upstream, because it exists to serve one requirement: an armed
        rank must keep consuming the chain WITHOUT blocking on it, so its
        upstream never blocks committing a forward to it (boot 18). A boot
        without the flip has no armed state, so it keeps the direct
        point_to_point_pyobj call and is byte-for-byte unchanged.

        It must be built ONCE per rank and used by BOTH the blocking and
        the non-blocking consumer: two consumers posting their own irecv
        on one stream would misframe it the moment a message was split
        across them.

        ON when the flip is enabled (#631 G). It was parked for one
        design generation because the only consumer it offered was
        ``poll()``, which rests on progressing a posted ``irecv`` by
        polling ``is_completed()`` -- MEASURED FALSE on this build and
        pinned in
        test_pp_chain_receiver.test_measured_gloo_does_not_progress_a_posted_irecv_by_polling.
        Wired live back then it would have absorbed nothing while the
        matching announce-when-flushed clause withheld presence for ever,
        so every flip would have abandoned at the presence deadline:
        strictly worse than the defect it was written to fix.

        What changed is the READINESS SIGNAL, not the transport. The armed
        path no longer asks the transport whether a message arrived; it
        reads the sender's published counter (``phase_flip_counters``) and
        only then makes a deliberate BLOCKING receive, bounded by transfer
        time. This class's two-step size-then-payload state machine is
        exactly what that path needs, and it is why it was kept rather
        than deleted.

        ``SGLANG_PP_CHAIN_RECEIVER=0`` disables it as a kill switch --
        which also disables the armed intake rule on this rank, so it is a
        diagnostic, not a supported serving mode.
        """
        if os.environ.get("SGLANG_PP_CHAIN_RECEIVER", "1") != "1":
            return None
        if not self.server_args.enable_phase_flip:
            return None
        if self.ps.pp_size <= 1 or self.ps.pp_rank == 0:
            return None
        if self.ps.attn_tp_rank != 0 or self.ps.attn_cp_rank != 0:
            return None
        from sglang.srt.managers.phase_flip_counters import CHAN_REQ
        from sglang.srt.managers.pp_chain_receiver import PpChainReceiver

        dp_offset = self.ps.attn_dp_rank * self.ps.attn_cp_size * self.ps.attn_tp_size
        counters = self.pp_flip_counters
        return PpChainReceiver(
            group=self.world_group.cpu_group,
            src=(self.ps.pp_rank - 1) * self.ps.tp_size + dp_offset,
            dst=self.ps.pp_rank * self.ps.tp_size + dp_offset,
            # Publish the consumed count as each message leaves the wire,
            # so the upstream learns its send is gone and can reap it with
            # a bounded blocking commit instead of a speculative one.
            on_consumed=(
                (lambda _n: counters.bump_consumed(CHAN_REQ))
                if counters is not None
                else None
            ),
        )

    def _build_pp_flip_counters(self):
        """#631 G: the pollable message-count channel, or None.

        Built on EVERY rank of a flip-enabled PP boot -- including rank 0,
        which has no upstream chain receiver but does have sends to reap
        and is the rank whose starvation defined corpse G.
        """
        if not self.server_args.enable_phase_flip:
            return None
        if self.ps.pp_size <= 1:
            return None
        if self.ps.attn_tp_rank != 0 or self.ps.attn_cp_rank != 0:
            return None
        from sglang.srt.managers.phase_flip_counters import PhaseFlipCounters
        from sglang.srt.managers.phase_flip_presence import (
            DEFAULT_PRESENCE_DIR,
            resolve_instance_tag,
        )

        counters = PhaseFlipCounters(
            n_ranks=self.ps.pp_size,
            rank=self.ps.pp_rank,
            directory=DEFAULT_PRESENCE_DIR,
            instance=resolve_instance_tag(),
        )
        # A previous boot's counts on this instance tag would be read as
        # messages in flight and send this rank into a blocking recv for
        # nothing. Same hazard the presence sweep exists for (boot 15).
        counters.sweep()
        return counters

    def phase_flip_is_armed(self) -> bool:
        """#631: is a flip armed on this rank right now?

        Read straight off the runtime rather than mirrored into a flag:
        the runtime's ``_pending`` is the one authority for arming, and a
        second copy would be a state to keep in sync. Absent runtime (not
        yet lazily built) means not armed.
        """
        runtime = getattr(self, "phase_flip_runtime", None)
        return runtime is not None and runtime.is_armed()

    def init_request_receiver(self) -> None:
        # #631 G: counters BEFORE the receiver -- the receiver publishes
        # its consumed count through them.
        self.pp_flip_counters = self._build_pp_flip_counters()
        self.pp_chain_receiver = self._build_pp_chain_receiver()
        self.request_receiver = SchedulerRequestReceiver(
            recv_from_tokenizer=self.ipc_channels.recv_from_tokenizer,
            recv_from_rpc=self.ipc_channels.recv_from_rpc,
            recv_skipper=self.recv_skipper,
            input_blocker=self.input_blocker,
            mm_receiver=self.mm_receiver,
            ps=self.ps,
            tp_group=self.tp_group,
            tp_cpu_group=self.tp_cpu_group,
            attn_tp_group=self.attn_tp_group,
            attn_tp_cpu_group=self.attn_tp_cpu_group,
            attn_cp_group=self.attn_cp_group,
            attn_cp_cpu_group=self.attn_cp_cpu_group,
            world_group=self.world_group,
            server_args=self.server_args,
            model_config=self.model_config,
            max_recv_per_poll=self.max_recv_per_poll,
            stream_output=lambda *a, **kw: self.output_streamer.stream_output(*a, **kw),
            get_last_forward_mode=lambda: (
                self.last_batch.forward_mode if self.last_batch is not None else None
            ),
            scripted_scheduler_hook=self.scripted_scheduler_hook,
            # #631: wired only when the policy is on; every other boot
            # carries None here and the receiver path is untouched.
            phase_policy_hook=(
                self.maybe_arm_phase_policy
                if (
                    getattr(self, "phase_policy_cfg", None) is not None
                    and self.phase_policy_cfg.enabled
                )
                else None
            ),
            # #631: both None unless the flip is enabled, so the default
            # intake path is unchanged.
            chain_receiver=self.pp_chain_receiver,
            # #631 G: gated on the FLIP, not on the receiver. The receiver
            # exists only on ranks with an upstream, so gating on it left
            # the armed intake rule off on rank 0 -- the intake rank, the
            # one that must stop admitting work for the group to reach a
            # quiescent boundary at all, and the rank whose starvation
            # defined corpse G.
            phase_flip_armed_hook=(
                self.phase_flip_is_armed if self.server_args.enable_phase_flip else None
            ),
            # #631 G: one service turn -- consume every inbound message
            # the upstream's counter accounts for, then reap this rank's
            # own sends that the downstream's counter proves consumed.
            phase_flip_service_hook=(
                self.pp_flip_service if self.server_args.enable_phase_flip else None
            ),
        )

    def init_dp_attn_adapter(self) -> None:
        self.dp_attn_adapter = SchedulerDPAttnAdapter(
            tp_group=self.tp_group,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            offload_tags=self.weight_updater.offload_tags,
            ps=self.ps,
            server_args=self.server_args,
            model_config=self.model_config,
            enable_overlap=self.enable_overlap,
            spec_algorithm=self.spec_algorithm,
            get_require_mlp_sync=lambda: self.require_mlp_sync,
        )

    def _pool_stats_dcp_factor(self) -> int:
        """Multiplier turning per-rank max_total_num_tokens into the GLOBAL DCP
        token capacity for pool-utilization stats. Even (modulo) DCP: dcp_size
        (max_total is this rank's physical pool). WEIGHTED DCP: 1, because
        max_total_num_tokens is ALREADY the global context C (the physical pool
        is the smaller per-rank ratio_r/S share, sized separately)."""
        from sglang.srt.distributed.utils import uneven_dcp_active

        if uneven_dcp_active(self.server_args.dcp_size):
            return 1
        return self.server_args.dcp_size

    def _global_kv_capacity_tokens(self) -> int:
        """This group's KV capacity as a GLOBAL token count (#346).

        Every scheduler-side comparison against a request's length needs this
        number, because a length is global on every rank while
        ``max_total_num_tokens`` is only global under the WEIGHTED owner rule;
        under the even-modulo rule it is this rank's physical shard and the
        allocator carries ``x cp_token_split_factor(dcp_size)`` slot ids.
        ``dcp_global_context_slots`` states that rule once for the runner-side
        ceiling (``max_token_pool_size``) and for here.

        The weightless spill lane pre-computes its own global span in
        ``_init_pools`` (device + host tiers, already multiplied); it wins,
        exactly as it does in ``max_token_pool_size``.

        Rank-uniform: ``max_total_num_tokens`` is min-reduced and the split
        factor is group-wide, so no collective and no divergent verdicts.
        Read live rather than cached at init, because the #330 VRAM dial
        raises ``max_total_num_tokens`` in place while the server runs.
        """
        wl_global = getattr(
            getattr(getattr(self, "tp_worker", None), "model_runner", None),
            "_wl_spill_global_capacity",
            0,
        )
        if wl_global:
            return int(wl_global)
        from sglang.srt.layers.dcp.owner import dcp_global_context_slots

        return dcp_global_context_slots(
            self.max_total_num_tokens, getattr(self.server_args, "dcp_size", 1)
        )

    def init_pool_stats_observer(self) -> None:
        self.pool_stats_observer = SchedulerPoolStatsObserver(
            tree_cache=self.tree_cache,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            req_to_token_pool=self.req_to_token_pool,
            session_controller=self.session_controller,
            hisparse_coordinator=self.hisparse_coordinator,
            is_hybrid_swa=self.is_hybrid_swa,
            is_hybrid_ssm=self.is_hybrid_ssm,
            enable_hisparse=self.enable_hisparse,
            full_tokens_per_layer=self.full_tokens_per_layer,
            swa_tokens_per_layer=self.swa_tokens_per_layer,
            max_total_num_tokens=self.max_total_num_tokens
            * self._pool_stats_dcp_factor(),
            get_last_batch=lambda: self.last_batch,
            get_running_batch=lambda: self.running_batch,
        )

    def init_invariant_checker(self) -> None:
        self.invariant_checker = SchedulerInvariantChecker(
            is_hybrid_swa=self.is_hybrid_swa,
            is_hybrid_ssm=self.is_hybrid_ssm,
            disaggregation_mode=self.disaggregation_mode,
            page_size=self.page_size,
            full_tokens_per_layer=self.full_tokens_per_layer,
            swa_tokens_per_layer=self.swa_tokens_per_layer,
            max_total_num_tokens=self.max_total_num_tokens,
            server_args=self.server_args,
            tree_cache=self.tree_cache,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            req_to_token_pool=self.req_to_token_pool,
            pool_stats_observer=self.pool_stats_observer,
            get_last_batch=lambda: self.last_batch,
            get_running_batch=lambda: self.running_batch,
        )

    def init_parked_decode_set(self) -> None:
        """#677 phase 1: stop charging undecodable carriers to the cap.

        ARMED ONLY WHERE THE DEFECT EXISTS. The wedge needs a phase that
        forbids decode while prefill keeps arriving, which is the phase
        flip under an enforcing purity mode. Without the flip there is no
        phase that forbids decode, every resident request is decodable,
        the discount would be zero on every round, and arming would only
        add a branch to the hottest gate in the scheduler.

        THE SLOT POOL IS READ FROM THE ALLOCATOR, NEVER FROM A FLAG. It is
        the mamba/GDN state pool -- the bound that actually refuses, late,
        inside alloc_req_slots -- and on a non-hybrid model there is no
        such pool and no such bound, so parking stays off there too rather
        than inventing a ceiling out of max_running_requests.
        """
        from sglang.srt.managers.parked_decode_set import (
            LOG_PREFIX as PARKED_DECODE_LOG_PREFIX,
        )
        from sglang.srt.managers.parked_decode_set import ParkedDecodeSet

        mamba_allocator = getattr(self.req_to_token_pool, "mamba_allocator", None)
        # Plain os.environ, matching its siblings (SGLANG_FLIP_SEAM_CHUNK_MIB,
        # SGLANG_PHASE_POLICY_DRAIN_MODE) rather than the envs registry, so
        # every phase-flip knob is set and read the same way.
        want = bool(
            int(os.environ.get("SGLANG_PHASE_PARK_CARRIERS", "1") or 0)
            and getattr(self.server_args, "enable_phase_flip", False)
            and mamba_allocator is not None
        )
        slot_pool = int(getattr(mamba_allocator, "size", 0) or 0)
        if want and slot_pool <= 0:
            # A pool that reports no slots cannot bound anything, and a
            # zero ceiling would refuse ALL admission. Refuse the feature,
            # not the traffic.
            logger.warning(
                "%s parking disarmed: the mamba allocator reports %d slots, "
                "so there is no state-pool ceiling to admit against.",
                PARKED_DECODE_LOG_PREFIX,
                slot_pool,
            )
            want = False
        self.parked_decode_set = ParkedDecodeSet(
            slot_pool=slot_pool,
            max_running=int(self.max_running_requests or 0),
            enabled=want,
        )
        #: The purity verdict the decode branch last reached, with the phase
        #: it was reached in. The gate cannot re-evaluate it: the predicate
        #: (`decode_blocked_here`) advances the starvation clock, so asking
        #: twice per round would double-tick it. Recording the phase is what
        #: makes a stale verdict safe -- a verdict from the other layout is
        #: discarded rather than trusted.
        self._parked_decode_verdict: Tuple[Optional[str], bool] = (None, False)
        logger.info(
            "%s parking %s: GDN slot pool %d, concurrency cap %d, phase flip "
            "%s. Armed, a carrier this phase forbids to decode stops counting "
            "against the cap and the slot pool becomes the admission ceiling.",
            PARKED_DECODE_LOG_PREFIX,
            "ARMED" if want else "off",
            slot_pool,
            int(self.max_running_requests or 0),
            "on" if getattr(self.server_args, "enable_phase_flip", False) else "off",
        )

    #: Rounds the target layout may take to build its first batch after a
    #: policy-armed flip before the arm is declared wrong. Small, because the
    #: thing being caught is a loop that re-armed every 3-4 seconds; large
    #: enough that one empty round during the post-cutover settle is not an
    #: accusation.
    ARM_VERDICT_ROUNDS = 8

    #: Minimum seconds between BOTH-BLOCKED evict attempts. The decline is
    #: evaluated every round, and an unbounded evict call there would be a
    #: tight loop over the whole radix tree while the instance is already
    #: wedged.
    BOTH_BLOCKED_EVICT_INTERVAL_S = 5.0

    def _uniform_kv_available(self):
        """Rank-uniform KV rows available, or None (#708).

        The BOTH-BLOCKED decline names its binding resource from this. It must
        be the GROUP MIN, not the local pool: every PhasePolicyInputs field is
        replicated by contract, and under uneven DCP the local availability
        differs per rank, so a local value would make the decline text -- and
        anything later keyed on it -- rank-dependent. That is the #616g
        divergence class this codebase already pays to avoid.
        ``uniform_avail_for_evict`` is the existing accessor: it returns the
        published group-min floor when the pools are uneven and the live local
        value when they agree.

        Returns None rather than a guess when it cannot be read, so the policy
        can say "not measured" instead of asserting.
        """
        try:
            tree = getattr(self, "tree_cache", None)
            if tree is None:
                return None
            allocator = getattr(tree, "token_to_kv_pool_allocator", None)
            if allocator is None:
                return None
            from sglang.srt.mem_cache.common import uniform_avail_for_evict

            return int(uniform_avail_for_evict(tree, allocator))
        except Exception:  # noqa: BLE001 - a diagnosis must never break a round
            return None

    def _apply_both_blocked_relief(self, decision, inp) -> None:
        """Actually run the eviction the BOTH-BLOCKED receipt promises.

        #698 THE INVARIANT WAS COMMENT-ONLY, and that is what a 54-minute
        outage was made of. phase_policy's branch says:

            "Declining here is what routes the caller to the evict rung
             instead of to a cutover."

        The caller did no such thing. Every decline was handled identically --
        one throttled log line, return None -- so the system printed "this is
        an evict trigger and NOT a flip" 350 times over 54 minutes while no
        evict was ever attempted. Health returned 200 throughout, three GPUs
        sat at 0%, and 10.5M tokens queued behind a pool nothing would free.
        That is the fourth counter-vs-actuator member found in one day, and
        the first where the actuator was described in prose and never written.

        BOUNDED, AND HONEST ABOUT DELIVERING NOTHING. The decline is evaluated
        every round, so the call is rate-limited; and it REPORTS what eviction
        actually returned, because "the remedy ran and freed 0" and "the remedy
        never ran" are the two states this outage could not distinguish.

        A ZERO HERE IS A FINDING, NOT A FAILURE OF THIS CODE. At the 16:23
        specimen the pool was held by an in-flight CHUNKED request's protected
        prefix -- a chunked request is resident but sits in no batch (#631
        defect O), which is why the scheduler read `#running-req: 0` while its
        prefix was locked. Eviction cannot free a locked chain, so on that
        specimen this routing would have delivered 0 and said so. Naming that
        in one line is what turns a silent wedge into a diagnosis.
        """
        try:
            from sglang.srt.managers.phase_policy import BOTH_BLOCKED

            if (decision.reason or "").find(BOTH_BLOCKED) != 0:
                return
            now = float(inp.now)
            last = getattr(self, "_both_blocked_evict_at", 0.0)
            if last and (now - last) < self.BOTH_BLOCKED_EVICT_INTERVAL_S:
                return
            self._both_blocked_evict_at = now
            tree = getattr(self, "tree_cache", None)
            if tree is None:
                return
            from sglang.srt.mem_cache.common import (
                evict_from_tree_cache,
                uniform_avail_for_evict,
            )

            want = int(getattr(self.server_args, "chunked_prefill_size", 0) or 0) or 512

            # ZERO IS AMBIGUOUS, AND THE FIRST VERSION OF THIS LOG READ IT
            # WRONG. `evict_from_tree_cache` returns 0 in TWO different states:
            # it ran and could not reach anything, or it was SKIPPED because
            # `avail >= num_tokens` already (see its `return 0` tail). Reporting
            # the alarming reading unconditionally sends the next reader hunting
            # a phantom locked chain -- which is the counter-vs-actuator mistake
            # this whole routine exists to catch, committed by the instrument
            # itself. Measure `avail` on the SAME side of the call the actuator
            # decides from, so the two zeros are told apart by evidence.
            avail_before = None
            try:
                allocator = getattr(tree, "token_to_kv_pool_allocator", None)
                if allocator is not None:
                    avail_before = int(uniform_avail_for_evict(tree, allocator))
            except Exception:  # noqa: BLE001 - diagnosis must not break relief
                avail_before = None

            freed = int(evict_from_tree_cache(tree, want) or 0)
            rows = self._post_evict_rows()

            if freed > 0:
                verdict = "The remedy the receipt names has now actually run."
            elif avail_before is not None and avail_before >= want:
                # Eviction was skipped: `want` was already available. But `want`
                # is chunked_prefill_size, an arbitrary chunk, NOT what the
                # blocked work actually needs -- so "avail >= 512" does NOT
                # license "KV is not binding". The 22:22:33 specimen made that
                # concrete: 19004 rows available, 512 wanted, and 97922 tokens
                # of prefill pending. The first version of this branch concluded
                # KV was not the binding resource from the 512 test alone, which
                # is the same over-claim from a partial instrument that this
                # routine exists to stop. Compare against the real demand.
                pending = None
                try:
                    pending = int(getattr(inp, "pending_prefill_tokens", 0) or 0)
                except Exception:  # noqa: BLE001
                    pending = None
                if pending and avail_before < pending:
                    verdict = (
                        f"Eviction was SKIPPED, not defeated: {avail_before} "
                        f"rows were already available against the {want} asked "
                        f"for. But {pending} tokens of prefill are pending, so "
                        f"KV may well be binding for the REAL demand -- this "
                        f"asked for a chunk, not for what the blocked work "
                        f"needs. Do not read this as 'KV is fine'."
                    )
                else:
                    verdict = (
                        f"Eviction was SKIPPED, not defeated: {avail_before} "
                        f"rows were already available against {want} wanted"
                        + (f" and {pending} pending" if pending else "")
                        + ", so the actuator had nothing to do. KV is not the "
                        "binding resource here -- look at the state-slot bound "
                        "(mamba/GDN slots) before blaming the pool."
                    )
            else:
                verdict = (
                    "Eviction RAN and delivered nothing"
                    + (
                        f" ({avail_before} rows available, {want} wanted)"
                        if avail_before is not None
                        else ""
                    )
                    + " -- the pool is held by something the frontier cannot "
                    "reach (an in-flight chunked request's protected prefix is "
                    "the known case). This is the state to escalate, not to "
                    "retry."
                )

            logger.warning(
                "PHASE-POLICY BOTH-BLOCKED RELIEF: asked the tree cache for "
                "%d rows, it freed %d; %d rows now reachable. %s",
                want,
                freed,
                rows,
                verdict,
            )
        except Exception as exc:  # noqa: BLE001 - relief must never break a round
            logger.warning(
                "PHASE-POLICY BOTH-BLOCKED RELIEF could not run (%r); the "
                "decline stands and nothing was freed.",
                exc,
            )

    def _note_round_build_outcome(self, ret, running_batch) -> None:
        """Record that this round built nothing while both classes had work.

        AN OBSERVATION, NOT A PREDICTION, and that distinction is the whole
        fix. ``get_next_batch_to_run`` has just finished trying to build a
        batch and produced none; whether this layout can run anything is
        therefore already ANSWERED at this line. The old escape from that
        state was the 180 s decode-stall cap -- a timer re-deriving by
        waiting what the round already knew (metal 2026-08-16 09:42:39-45:
        six seconds of zero GPU, zero PCIe, 572715 tok queued, four carriers
        ready, py-spy showing all three ranks spinning this very function).

        BOTH CLASSES MUST HAVE WORK AND BOTH MUST HAVE FAILED. An empty
        instance also builds no batch, and flipping an idle server is thrash
        with no benefit -- so "no batch" alone is deliberately not the
        trigger.
        """
        watch = getattr(self, "_arm_watch", None)
        if watch is not None:
            if ret is not None:
                # The target ran. The verdict is vindicated; stop watching.
                self._arm_watch = None
            else:
                watch["rounds"] += 1
                if watch["rounds"] == self.ARM_VERDICT_ROUNDS:
                    committed = (
                        getattr(self, "phase_flip_active_stack", None)
                        != watch["phase_at_arm"]
                    )
                    if not committed:
                        # ARM-UNFUNDED, NOT ARM-VERDICT-WRONG, and the split
                        # matters because the first version accused the wrong
                        # component. Measured 2026-08-16 11:05: three
                        # ARM-VERDICT-WRONG against twelve
                        # "FLIP ABANDONED (pool too small for the live set)",
                        # eight of them "This rank: fits (a peer did not)".
                        # The target layout never became active, so the
                        # admissibility verdict was never tested -- the SEAM
                        # could not pay. A falsifier that fires on funding
                        # failures stops being a falsifier for verdicts.
                        logger.warning(
                            "PHASE-POLICY ARM-UNFUNDED: armed %s (%s) and the "
                            "cutover has not committed after %d rounds -- the "
                            "instance is still in the %s layout. This is a "
                            "SEAM FUNDING failure, not a wrong admissibility "
                            "verdict: the target was never entered, so the "
                            "verdict was never tested. Look for FLIP ABANDONED "
                            "on the binding rank, not at the arm.",
                            watch["direction"],
                            watch["reason"],
                            watch["rounds"],
                            watch["phase_at_arm"],
                        )
                    else:
                        logger.warning(
                            "PHASE-POLICY ARM-VERDICT-WRONG: armed %s (%s), the "
                            "cutover COMMITTED into the target layout, and it "
                            "still built no batch in %d rounds. The "
                            "admissibility inputs that produced this arm were "
                            "running_bs=%d pending=%d nothing_can_run=%s "
                            "target_can_admit=%s ready_carriers=%d. The verdict "
                            "was tested and was wrong; if this repeats in "
                            "alternating directions it is the 2026-08-16 10:24 "
                            "ping-pong and the target term is lying again.",
                            watch["direction"],
                            watch["reason"],
                            watch["rounds"],
                            watch["running_bs"],
                            watch["pending"],
                            watch["nothing_can_run"],
                            watch["target_can_admit"],
                            watch["ready_carriers"],
                        )
        if ret is not None:
            self._round_built_nothing = False
            return
        resident = len(getattr(running_batch, "reqs", None) or [])
        try:
            pending = int(self._pending_prefill_tokens() or 0)
        except Exception:  # noqa: BLE001 - an observation must not break a round
            pending = 0
        self._round_built_nothing = bool(resident > 0 or pending > 0)

    def _post_evict_rows(self) -> int:
        """KV rows this rank could hand out if the cache gave up everything.

        POST-EVICT, because the cache is not a claim on the pool -- it is a
        cache. The 10:30:50 boot died with ``full_available_size=0`` and
        ``full_evictable_size=151040`` in the same message: an allocator that
        looked empty while 151040 rows sat there evictable. An admissibility
        answer computed from ``available`` alone would call that layout
        unusable and be wrong by 151040 rows.
        """
        alloc = getattr(self, "token_to_kv_pool_allocator", None)
        tree = getattr(self, "tree_cache", None)
        try:
            avail = int(alloc.available_size()) if alloc is not None else 0
        except Exception:  # noqa: BLE001 - a probe must not break the round
            avail = 0
        # #698: ASK FOR THE FULL-ATTENTION COUNT FIRST, and fall back to the
        # flat accessor only for the classes that have one.
        #
        # MambaRadixCache.evictable_size() RAISES NotImplementedError -- it
        # splits the count in two and says so ("use full_evictable_size() and
        # mamba_evictable_size() instead"). This swallowed that exception and
        # used 0, so on the class this rig actually runs the probe returned
        # `available` ALONE -- exactly the error the docstring above warns
        # about, committed three lines below it.
        #
        # At usage 1.00 that reads ~0, so every admissibility question answered
        # "no": pp could not admit, tp had nothing resident to decode, and the
        # #688 BOTH BLOCKED branch declined the flip. That branch returns
        # BEFORE alloc_token_slots, so the allocator was never reached, so
        # eviction never ran, so a pool that was 100% UNLOCKED CACHE with zero
        # resident requests was never freed. Serving stopped for 54 minutes on
        # 2026-08-16 with health returning 200 throughout, three GPUs at 0%,
        # and 10.5M tokens queued behind a cache nothing would evict.
        #
        # The identical trap is documented at mem_cache/common.py:411-425 for
        # the same two classes. Resolution order is copied from there rather
        # than re-derived, because two spellings of one rule is how this
        # returns.
        #
        # A SWALLOWED EXCEPTION THAT YIELDS A PLAUSIBLE NUMBER is the shape to
        # avoid: zero is a legal row count, so nothing downstream could
        # distinguish "the cache holds nothing" from "the cache was never
        # asked". Each accessor is tried in turn and only a genuine absence of
        # all of them yields zero.
        evictable = 0
        for name in ("full_evictable_size", "evictable_size"):
            getter = getattr(tree, name, None) if tree is not None else None
            if getter is None:
                continue
            try:
                evictable = int(getter())
                break
            except Exception:  # noqa: BLE001 - try the next accessor
                continue
        return max(0, avail) + max(0, evictable)

    def _layout_admits(self, phase: str, running_bs: int, pending_tokens: int) -> bool:
        """Can the layout ``phase`` build a batch of the class it is allowed?

        ONE SIMULATION, USED FOR BOTH SIDES. The current layout and the target
        layout are the same question asked of two phases, so they must not be
        answered by two different mechanisms -- that asymmetry is exactly what
        produced the 10:24 ping-pong (target simulated, current inferred) and
        then the 10:47 premature arm (current inferred from a single round).

        PP may only prefill: it needs a CHUNK of rows and a free GDN state
        slot for the incoming request. TP may only decode under drain purity:
        it needs the pool to back one decode step for the residents, which
        under NEXTN is a draft window per carrier rather than one row for the
        batch. Rows are counted POST-EVICT, because the cache is not a claim
        on the pool.

        #748 REFAIL: WHICH CLASS A LAYOUT MAY RUN IS THE PURITY RULE'S ANSWER,
        NOT A CONSTANT. The two sentences above were written under ``strict``
        purity and then hardcoded, so this simulation kept answering "TP may
        only decode" on a server booted ``--phase-flip-purity prefill_in_tp``.
        Measured on boot_735_nohc.log, 2026-08-18 08:37:40-08:54:42: 160 armings
        in 1022 s (9.4/min), 80 of them ``tp_to_pp`` with 0 req resident and
        163-817 tok pending. On those rounds ``running_bs <= 0`` returned False
        below, the policy read ``nothing_can_run=True``, and the #688 escape
        fired -- while the SAME log carries 76 lines of the policy's own verdict
        ``holding in tp: pending prefill 163 tok <= N=7004, running it in tp``
        and 42 executed ``Prefill batch phase=tp``. The simulation contradicted
        both the config and the observed batches, seconds apart.

        This is why neither #748 (1cc0d24ae7 / 256fe09fab) nor #759
        (72c1ed9c18) closed it: both fixed how the POLICY prices the escape.
        The escape was not mispriced, it was fed a false premise from here.

        THE ORACLE IS THE ONE THE BATCH BUILDER USES, not a second opinion.
        ``prefill_suppressed_in_tp`` is what ``phase_purity.prefill_blocked_here``
        consults before a TP prefill batch is built, and at ``running_bs == 0``
        it lifts drain-mode suppression outright (phase_policy.py, "with
        running_bs == 0 the bundle is finished") -- which is exactly the
        specimen's state. Asking a different question here is how the two
        answers diverged in the first place.

        ``flip_unavailable`` is passed False deliberately: resolving it calls
        ``flip_unavailable_reason``, which reads live seam state and logs, and
        this must stay a pure probe. False is also the SAFE direction -- it
        makes suppression more likely, so the simulation is more likely to say
        "TP cannot prefill" and leave the escape armed. An error here delays
        nothing and can never wedge.

        ABSENT PURITY KEEPS TODAY'S ANSWER. A scheduler with no ``_phase_purity``
        (flip disabled, or a unit fixture) is not evidence that the other class
        is allowed, so the historical decode-only / prefill-only shape stands.
        """
        rows = self._post_evict_rows()
        if phase == "pp":
            if self._layout_admits_prefill(rows, pending_tokens):
                return True
            # Decode in PP is forbidden under strict and prefill_in_tp, allowed
            # under off, and bounded under threshold:<n> -- the rule's own
            # answer, not this function's assumption.
            if int(running_bs) > 0 and self._purity_allows("decode_in_pp", running_bs):
                return self._layout_admits_decode(rows, running_bs)
            return False
        if phase == "tp":
            if int(pending_tokens) > 0 and self._purity_allows(
                "prefill_in_tp", running_bs
            ):
                if self._layout_admits_prefill(rows, pending_tokens):
                    return True
            if int(running_bs) <= 0:
                return False
            return self._layout_admits_decode(rows, running_bs)
        return False

    def _layout_admits_prefill(self, rows: int, pending_tokens: int) -> bool:
        """Rows and a state slot for one chunk of the pending backlog.

        Factored out of ``_layout_admits`` so both layouts ask the prefill
        question with ONE arithmetic, which is the property that docstring's
        "ONE SIMULATION" claim rests on. It was true across the two PHASES and
        false across the two CLASSES: TP had no prefill arm at all.
        """
        if int(pending_tokens) <= 0:
            return False
        chunk = int(getattr(self.server_args, "chunked_prefill_size", 0) or 0) or 512
        need = min(chunk, int(pending_tokens))
        mamba = getattr(
            getattr(self, "req_to_token_pool", None), "mamba_allocator", None
        )
        try:
            slots = int(mamba.available_size()) if mamba is not None else 1
        except Exception:  # noqa: BLE001 - a probe must not break the round
            slots = 0
        return rows >= need and slots >= 1

    def _layout_admits_decode(self, rows: int, running_bs: int) -> bool:
        """One decode step for the residents; under NEXTN a draft window each."""
        if int(running_bs) <= 0:
            return False
        per_req = max(
            1, int(getattr(self.server_args, "speculative_num_draft_tokens", 1) or 1)
        )
        return rows >= int(running_bs) * per_req

    def _purity_allows(self, what: str, running_bs: int) -> bool:
        """Does the BOOT'S purity rule permit ``what`` in the off-class layout?

        Returns False when no purity rule is resolved, so a scheduler without
        one behaves exactly as before this fix.
        """
        purity = getattr(self, "_phase_purity", None)
        if purity is None:
            return False
        try:
            if what == "prefill_in_tp":
                if not purity.prefill_allowed_in_tp():
                    return False
                from sglang.srt.managers.phase_policy import (
                    PHASE_TP,
                    prefill_suppressed_in_tp,
                )

                cfg = getattr(self, "phase_policy_cfg", None)
                if cfg is None:
                    return True
                return not prefill_suppressed_in_tp(
                    cfg, PHASE_TP, flip_unavailable=False, running_bs=running_bs
                )
            if what == "decode_in_pp":
                return bool(purity.decode_allowed_in_pp(running_bs))
        except Exception:  # noqa: BLE001 - a probe must not break the round
            return False
        return False

    def _idle_locked_inputs(self, running_bs: int, pending_tokens: int):
        """``(nothing_can_run, target_admissible)`` for the policy.

        BOTH TERMS ARE SIMULATED NOW. Each was a proxy once and each cost a
        live defect, in the same shape:

        * ``target_can_admit`` was "work of that class exists". On 2026-08-16
          10:24 that was permanently true on both sides while NEITHER layout
          could run, and the policy ping-ponged every 3-4 seconds.
        * ``nothing_can_run`` was "the round happened to build nothing". On
          2026-08-16 10:47:42 that fired on a single transient empty round
          with the pool 5% used, 6 of 12 GDN slots free and a request still
          queued -- PP could plainly have admitted more. The arm bypassed
          window formation (correctly, #688 outranks #689) and the decode
          window opened at ONE carrier, which is the bs=1 defect #689 exists
          to remove.

        A ROUND THAT BUILT NOTHING IS NECESSARY BUT NOT SUFFICIENT. It is kept
        as the trigger -- the check is worthless if it fires while batches are
        being built -- but the verdict is now whether the layout CAN build
        one, not whether it just did.
        """
        if not bool(getattr(self, "_round_built_nothing", False)):
            return False, False
        phase = getattr(self, "phase_flip_active_stack", None)
        if phase not in ("pp", "tp"):
            # No flip enabled, or a phase this rule says nothing about.
            return False, False
        other = "tp" if phase == "pp" else "pp"
        here = self._layout_admits(phase, running_bs, pending_tokens)
        there = self._layout_admits(other, running_bs, pending_tokens)
        # #713: NAME THE TERMS WHEN THE VERDICT IS THE EXPENSIVE ONE.
        #
        # Measured 2026-08-17 03:0x: a TEN-token prompt waited 31.64 s on an
        # idle box -- 0 running, 1 queued, 3 mamba slots free, 72033 KV rows
        # free -- because this returned target_can_admit=False and the policy
        # declined to flip. But replaying _layout_admits with exactly those
        # numbers returns pp=True/tp=False, i.e. the simulation is RIGHT for
        # that state and would have armed. So the inputs it reads in-process
        # differ from what /metrics reports, and no external sampling can show
        # which -- the terms have to be printed where they are computed.
        #
        # Emitted only on the refusal (nothing here, nothing there), and rate
        # limited, because the whole point is to catch a state that persists
        # for tens of seconds rather than to narrate healthy rounds.
        if (not here) and (not there):
            now = time.perf_counter()
            last = getattr(self, "_idle_locked_diag_at", 0.0)
            if now - last >= 5.0:
                self._idle_locked_diag_at = now
                mamba = getattr(
                    getattr(self, "req_to_token_pool", None), "mamba_allocator", None
                )
                try:
                    slots = int(mamba.available_size()) if mamba is not None else -1
                except Exception as exc:  # noqa: BLE001 - a probe must not break
                    slots = f"RAISED {type(exc).__name__}"
                alloc = getattr(self, "token_to_kv_pool_allocator", None)
                try:
                    avail = int(alloc.available_size()) if alloc is not None else -1
                except Exception as exc:  # noqa: BLE001
                    avail = f"RAISED {type(exc).__name__}"
                # THE THIRD PROBE GETS THE SAME ARMOUR, and it is not
                # decoration: called bare inside the logger arguments, a raise
                # here would kill the scheduler round this line exists to
                # OBSERVE -- which is exactly how #715's RADIX SHAPE walk died
                # inside the crash it was written to explain. "Probably safe
                # because _layout_admits just evaluated it" is the reasoning
                # the RAISED pattern exists to replace.
                try:
                    rows_seen = self._post_evict_rows()
                except Exception as exc:  # noqa: BLE001
                    rows_seen = f"RAISED {type(exc).__name__}"
                logger.warning(
                    "PHASE-POLICY IDLE-LOCKED TERMS phase=%s running_bs=%s "
                    "pending_tokens=%s | here(%s)=%s there(%s)=%s | "
                    "post_evict_rows=%s allocator_avail=%s mamba_slots=%s "
                    "chunk=%s -- both layouts refused; these are the numbers "
                    "the simulation actually read.",
                    phase,
                    running_bs,
                    pending_tokens,
                    phase,
                    here,
                    other,
                    there,
                    rows_seen,
                    avail,
                    slots,
                    getattr(self.server_args, "chunked_prefill_size", None),
                )
        return (not here), there

    def _note_parked_carriers(self, running_batch, decode_blocked: bool) -> None:
        """Record this round's purity verdict and reconcile the parked set.

        Called from the ONE site that already evaluates the verdict, so the
        starvation clock still ticks exactly once per round.
        """
        if not self.parked_decode_set.enabled:
            return
        phase = getattr(self, "phase_flip_active_stack", None)
        self._parked_decode_verdict = (phase, bool(decode_blocked))
        reqs = (
            list(getattr(running_batch, "reqs", None) or []) if decode_blocked else []
        )
        self.parked_decode_set.sync_carriers(
            [getattr(r, "rid", "") for r in reqs],
            len(getattr(running_batch, "reqs", None) or []),
        )

    def _parked_carrier_discount(self, running_bs: int) -> int:
        """Carriers the concurrency cap must not count, this round.

        TWO CLAMPS, EACH FOR A DIFFERENT WAY THE RECORD CAN BE STALE.

        The verdict is recorded by the decode branch, which does not run on
        every round -- a round that selects a prefill batch never reaches
        it. So the record can outlive the layout it was made in, and a
        verdict from the OTHER phase is discarded outright: PP forbids
        decode and TP does not, so trusting a PP verdict inside TP would
        discount carriers that are actively decoding.

        The id set can also outlive the requests in it, for the same
        reason -- a carrier that finished on a round the decode branch did
        not reach is still listed. Clamping the discount to the resident
        count means a stale id can never credit more than is there, so the
        worst case degrades to the pre-change gate rather than to
        over-admission.
        """
        phase, blocked = getattr(self, "_parked_decode_verdict", (None, False))
        if not blocked or phase != getattr(self, "phase_flip_active_stack", None):
            return 0
        return min(self.parked_decode_set.carrier_discount(), max(0, int(running_bs)))

    def init_admission_limiter(self) -> None:
        """Build this group's floating admission limit (#287).

        ``self.max_running_requests`` at this point is the RESOLVED ceiling
        (per dp worker): the value the pools, the mamba slot table and the
        decode capture set were dimensioned for. Without
        --max-running-requests-ceiling the limiter is a passive holder of
        exactly that value, so ``get_num_allocatable_reqs`` and the prefill
        adder see the same number they see today.

        The limiter is per group/lane and is published into a context
        variable (#274 Slice C1 idiom) so distant readers -- the #236 spill
        budget today, the #242 latency classes later -- resolve to their own
        lane's value instead of a shared singleton.
        """
        sa = self.server_args
        ceiling = int(self.max_running_requests)
        start = resolve_admission_start(
            ceiling,
            getattr(sa, "max_running_requests_start", None),
            dp_size=sa.dp_size if sa.enable_dp_attention else 1,
            floor=sa.admission_floor,
        )
        self.admission_limiter = AdmissionLimiter(
            ceiling,
            start,
            floor=min(sa.admission_floor, ceiling),
            throttle_high=sa.admission_throttle_high,
            release_low=sa.admission_release_low,
            release_hysteresis=sa.admission_release_hysteresis,
            auto=sa.max_running_requests_ceiling is not None,
            lane_id=None,
        )
        set_admission_limiter(self.admission_limiter)
        if self.admission_limiter.auto and self.ps.tp_rank == 0:
            # #307: the ceiling is a REQUEST, and the pools it dimensions are
            # fitted to the budget when the cards cannot hold it (a hybrid
            # model's per-request state is not elastic). The limiter floats
            # below the FITTED ceiling, so say when the two differ -- silently
            # serving 18 where 64 was asked for is the kind of gap that gets
            # read as a throttling bug. Uniform across ranks by construction:
            # every input to `ceiling` is min-reduced before it gets here.
            per_worker = sa.dp_size if sa.enable_dp_attention else 1
            requested = max(1, int(sa.max_running_requests_ceiling) // per_worker)
            if ceiling < requested:
                logger.warning(
                    "Dynamic admission limit: the requested ceiling %d (per "
                    "worker) does not fit the memory budget; the state pools "
                    "and the float were fitted to %d. Raise the per-rank "
                    "budget (--rank-gpu-memory-mib / --rank-auto-reserve-mib) "
                    "or lower the ceiling to make the request honest.",
                    requested,
                    ceiling,
                )
            logger.info(
                "Dynamic admission limit: ceiling=%d, start=%d, floor=%d "
                "(throttle>=%.2f, release<=%.2f x%d). State pools are "
                "dimensioned for the ceiling; the limit floats below it.",
                self.admission_limiter.ceiling,
                self.admission_limiter.current,
                self.admission_limiter.floor,
                self.admission_limiter.throttle_high,
                self.admission_limiter.release_low,
                self.admission_limiter.release_hysteresis,
            )

    def init_kv_events_publisher(self) -> None:
        self.kv_events_publisher = SchedulerKvEventsPublisher(
            kv_events_config=self.server_args.kv_events_config,
            ps=self.ps,
            attn_tp_rank=self.ps.attn_tp_rank,
            attn_cp_rank=self.ps.attn_cp_rank,
            attn_dp_rank=self.ps.attn_dp_rank,
            dp_rank=self.ps.dp_rank,
            tree_cache=self.tree_cache,
            send_metrics_from_scheduler=self.ipc_channels.send_metrics_from_scheduler,
            max_running_requests=self.max_running_requests,
            max_total_num_tokens=self.max_total_num_tokens,
            get_stats=lambda: self.metrics_reporter.stats,
        )

    def init_load_inquirer(self) -> None:
        self.load_inquirer = SchedulerLoadInquirer(
            disaggregation_mode=self.disaggregation_mode,
            ps=self.ps,
            server_args=self.server_args,
            max_total_num_tokens=self.max_total_num_tokens,
            max_running_requests=self.max_running_requests,
            pool_stats_observer=self.pool_stats_observer,
            tp_worker=self.tp_worker,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            spec_algorithm=self.spec_algorithm,
            get_running_batch=lambda: self.running_batch,
            get_waiting_queue=lambda: self.waiting_queue,
            get_stats=lambda: self.metrics_reporter.stats,
            get_chunked_req=lambda: self.chunked_req,
            get_disagg_prefill_bootstrap_queue=lambda: (
                self.disagg_prefill_bootstrap_queue
            ),
            get_disagg_prefill_inflight_queue=lambda: (
                self.disagg_prefill_inflight_queue
            ),
            get_disagg_decode_prealloc_queue=lambda: self.disagg_decode_prealloc_queue,
            get_disagg_decode_transfer_queue=lambda: self.disagg_decode_transfer_queue,
            get_spec_total_num_accept_tokens=lambda: (
                self.metrics_reporter.spec_total_num_accept_tokens
            ),
            get_spec_total_num_forward_ct=lambda: (
                self.metrics_reporter.spec_total_num_forward_ct
            ),
        )

    def init_output_streamer(self) -> None:
        self.output_streamer = SchedulerOutputStreamer(
            send_to_detokenizer=self.ipc_channels.send_to_detokenizer,
            tree_cache=self.tree_cache,
            ps=self.ps,
            server_args=self.server_args,
            is_generation=self.is_generation,
            spec_algorithm=self.spec_algorithm,
            disaggregation_mode=self.disaggregation_mode,
            enable_hicache_storage=lambda: self.enable_hicache_storage,
        )

    def init_batch_result_processor(self) -> None:
        self.batch_result_processor = SchedulerBatchResultProcessor(
            is_generation=self.is_generation,
            disaggregation_mode=self.disaggregation_mode,
            enable_overlap=self.enable_overlap,
            enable_overlap_mlx=self.enable_overlap_mlx,
            server_args=self.server_args,
            model_config=self.model_config,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            hisparse_coordinator=self.hisparse_coordinator,
            req_to_token_pool=self.req_to_token_pool,
            decode_offload_manager=self.decode_offload_manager,
            metrics_collector=self.metrics_collector,
            metrics_reporter=self.metrics_reporter,
            draft_worker=self.draft_worker,
            model_worker=self.model_worker,
            logprob_result_processor=SchedulerLogprobResultProcessor(
                server_args=self.server_args, model_config=self.model_config
            ),
            output_streamer=self.output_streamer,
            abort_request=self.abort_request,
            kv_session_offload=self.kv_session_offload,
            record_first_token_progress=self.note_first_token_progress,
            record_prefill_progress=self.note_prefill_progress,
        )

    def init_req_max_new_tokens(self, req):
        input_len = len(req.origin_input_ids)
        # Keep this bound consistent with PrefillAdder's admission budget:
        # ceil_page(input_len) + max_new_tokens + page_size must be strictly
        # smaller than max_total_num_tokens. Otherwise a request can be accepted
        # into the waiting queue but can never be scheduled, blocking the queue
        # and eventually making health checks fail.
        paged_input_len = -(-input_len // self.page_size) * self.page_size
        # input_len is a GLOBAL token count, but a DCP-token-sharded sequence
        # only consumes input_len // S slots PER RANK while the allocator
        # index space is the GLOBAL capacity. Subtracting a global length from
        # the per-rank pool clamps the generation budget to 0 for every input
        # above that pool even though it fits the group (an over-VRAM 259k
        # sequence on a 104k per-rank pool; #346 for the general even-modulo
        # lane, previously handled for the weightless spill tier alone).
        # It is also what keeps this bound consistent with the PrefillAdder
        # budget it is documented to track: `rem_total_tokens` counts the
        # ALLOCATOR's available slots, i.e. the global space, so the per-rank
        # number was the tighter of two different quantities rather than the
        # same one. Off the lane and under the weighted rule this is the same
        # number as before, byte-identical.
        _pool_cap = self._global_kv_capacity_tokens()
        req.sampling_params.max_new_tokens = max(
            0,
            min(
                (
                    req.sampling_params.max_new_tokens
                    if req.sampling_params.max_new_tokens is not None
                    else 1 << 30
                ),
                self.max_req_len - input_len - 1,
                _pool_cap - paged_input_len - self.page_size - 1,
            ),
        )

    def _process_and_broadcast_mm_inputs(
        self,
        raw_mm_inputs,
    ):
        """Materialize MultimodalInputs once on the entry rank and broadcast to others.

        Entry rank:
        - constructs MultimodalInputs.from_processor_output() once
        - broadcasts to other ranks in self.cpu_group (if world_size > 1)

        Non-entry ranks:
        - receive the object via broadcast (if world_size > 1)
        - otherwise (single-rank / no group) fall back to local from_processor_output

        Returns:
            MultimodalInputs | None
        """
        if raw_mm_inputs is None:
            return None

        group_world_size = 1
        try:
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and self.dp_tp_cpu_group is not None
            ):
                group_world_size = torch.distributed.get_world_size(
                    group=self.dp_tp_cpu_group
                )
        except Exception as e:
            logger.warning(
                f"Failed to get world size in mm_inputs handling with {e}, fallback to 1."
            )

        # In case tp size > 1, all the Scheduler TP ranks runs the duplicated computing
        # process in CPU which occupies the main thread CPU cycle. This computing logic
        # merely needs to be run on TP0 and be broadcast to other TP ranks.
        # Since the Scheduler is single-threaded, any large CPU cost will impact
        # handling of other messages. For example, CPU hits 99.9% can significantly
        # increase the CUDA kernel launch time.
        if self.dp_tp_group.rank_in_group == 0:
            # Only the entry rank materializes once from dict.
            image_inputs = MultimodalInputs.from_processor_output(raw_mm_inputs)
            # Broadcast to other TP ranks (use src=0 within the group).
            if group_world_size > 1:
                obj_list = [image_inputs]
                torch.distributed.broadcast_object_list(
                    obj_list,
                    src=self.dp_tp_group.first_rank,
                    group=self.dp_tp_cpu_group,
                )
                image_inputs = obj_list[0]
        else:
            # Non-entry ranks: receive if group size > 1; otherwise materialize locally.
            if group_world_size > 1:
                obj_list = [None]
                torch.distributed.broadcast_object_list(
                    obj_list,
                    src=self.dp_tp_group.first_rank,
                    group=self.dp_tp_cpu_group,
                )
                image_inputs = obj_list[0]
            else:
                image_inputs = MultimodalInputs.from_processor_output(raw_mm_inputs)

        return image_inputs

    def _get_multimodal_inputs(self, mm_inputs_dict):
        if self.server_args.enable_broadcast_mm_inputs_process:
            return self._process_and_broadcast_mm_inputs(mm_inputs_dict)
        else:
            return MultimodalInputs.from_processor_output(mm_inputs_dict)

    @staticmethod
    def _try_apply_padded_mm_input_ids(recv_req, req, image_inputs) -> bool:
        """setup origin_input_ids with trying to reuse existing MultimodalInputs.padded_input_ids first,
        if absent, call pad_input_ids_func"""
        padded_input_ids = image_inputs.padded_input_ids
        if padded_input_ids is None or recv_req.input_ids is None:
            return False

        recv_input_len = len(recv_req.input_ids)
        if len(padded_input_ids) != recv_input_len:
            return False

        prefix_len = len(req.origin_input_ids) - recv_input_len
        if prefix_len < 0:
            return False

        padded_input_ids = array("q", padded_input_ids)
        if prefix_len == 0:
            req.origin_input_ids = padded_input_ids
        else:
            req.origin_input_ids = req.origin_input_ids[:prefix_len] + padded_input_ids
        return True

    def _maybe_compute_mrope_positions(self, req) -> None:
        """Compute M-RoPE positions when they are missing (e.g. gRPC preprocessed path)."""
        if self._mm_processor is None:
            return
        mm = req.multimodal_inputs
        if mm is None or mm.mrope_positions is not None:
            return

        mrope_positions, mrope_position_delta = (
            self._mm_processor.compute_mrope_positions(
                req.origin_input_ids, mm.mm_items
            )
        )
        if mrope_positions is not None:
            mm.mrope_positions = mrope_positions
            mm.mrope_position_delta = mrope_position_delta

    def _maybe_clear_mm_inputs(self, batch: ScheduleBatch) -> None:
        for req in batch.reqs:
            if not req.finished() or not (mm_inputs := req.multimodal_inputs):
                continue
            # For session requests, keep mm_inputs for the next request
            if req.session:
                continue
            # For non-session requests, clear features and mm_inputs
            mm_inputs.release_features()
            req.multimodal_inputs = None

    def handle_generate_request(
        self,
        recv_req: TokenizedGenerateReqInput,
    ):
        # #261 live handover: a prefix parked for handover must not be
        # extended while the destination may still commit. No-op on every
        # default path (runtime is None until the first handover request,
        # and returns immediately while no handover is active).
        if self.session_handover_runtime is not None and recv_req.input_ids is not None:
            parked_msg = self.session_handover_runtime.parked_conflict(
                recv_req.input_ids
            )
            if parked_msg is not None:
                req = Req(
                    recv_req.rid,
                    recv_req.input_text,
                    recv_req.input_ids,
                    recv_req.sampling_params,
                    vocab_size=self.model_config.vocab_size,
                    http_worker_ipc=recv_req.http_worker_ipc,
                )
                req.tokenizer = self.tokenizer
                req.set_finish_with_abort(parked_msg)
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

        # Route: normal request / session request / session-not-found
        session_id = (
            recv_req.session_params.id if recv_req.session_params is not None else None
        )
        # Radix-native sessions use only the top-level session_id.
        radix_native_session = (
            recv_req.session_id is not None
            and self.server_args.enable_session_radix_cache
        )

        if session_id is None or radix_native_session:
            # Normal non-session request, or a radix-native session request
            if recv_req.input_embeds is not None:
                # Generate fake input_ids based on the length of input_embeds
                seq_length = len(recv_req.input_embeds)
                recv_req.input_ids = array("q", [1]) * seq_length

            if recv_req.bootstrap_port is None:
                # Use default bootstrap port
                recv_req.bootstrap_port = self.server_args.disaggregation_bootstrap_port

            req = Req(
                recv_req.rid,
                recv_req.input_text,
                recv_req.input_ids,
                recv_req.sampling_params,
                return_logprob=recv_req.return_logprob,
                top_logprobs_num=recv_req.top_logprobs_num,
                token_ids_logprob=recv_req.token_ids_logprob,
                stream=recv_req.stream,
                lora_id=recv_req.lora_id,
                session_id=recv_req.session_id,
                input_embeds=recv_req.input_embeds,
                positional_embed_overrides=recv_req.positional_embed_overrides,
                token_type_ids=recv_req.token_type_ids,
                custom_logit_processor=recv_req.custom_logit_processor,
                require_reasoning=recv_req.require_reasoning,
                return_hidden_states=recv_req.return_hidden_states,
                return_routed_experts=recv_req.return_routed_experts,
                routed_experts_start_len=recv_req.routed_experts_start_len,
                return_indexer_topk=recv_req.return_indexer_topk,
                eos_token_ids=self.model_config.hf_eos_token_id,
                bootstrap_host=recv_req.bootstrap_host,
                bootstrap_port=recv_req.bootstrap_port,
                bootstrap_room=recv_req.bootstrap_room,
                disagg_mode=self.disaggregation_mode,
                routed_dp_rank=recv_req.routed_dp_rank,
                disagg_prefill_dp_rank=recv_req.disagg_prefill_dp_rank,
                vocab_size=self.model_config.vocab_size,
                priority=recv_req.priority,
                metrics_collector=(
                    self.metrics_collector
                    if self.metrics_reporter.enable_metrics
                    else None
                ),
                routing_key=recv_req.routing_key,
                extra_key=recv_req.extra_key,
                http_worker_ipc=recv_req.http_worker_ipc,
                dllm_config=self.dllm_config,
                time_stats=recv_req.time_stats,
                multi_item_delimiter_indices=recv_req.multi_item_delimiter_indices,
            )
            req.tokenizer = self.tokenizer
            # Fast lane (Variant C Stage 0): tag the request's lane so the
            # anti-starvation reserved-heavy-slots floor can distinguish fast
            # from heavy requests during preemption. Only meaningful when the
            # server was launched with --enable-fast-lane.
            req.is_fast_lane = getattr(recv_req, "lane", None) == "fast"
            # kv-session-offload: carry the per-session spill (latency) class
            # onto the Req. The tokenizer manager already validated it and
            # applied the server default; a None here means an internal path
            # that never carried the field -> keep the "normal" default set in
            # Req.__init__ (stock FCFS order).
            _spill_class = getattr(recv_req, "spill_class", None)
            if _spill_class is not None:
                req.spill_class = _spill_class

            if self.disaggregation_mode != DisaggregationMode.NULL:
                # Invalid request for disaggregated mode
                if (
                    recv_req.bootstrap_room is None
                    and self.transfer_backend != TransferBackend.FAKE
                ):
                    error_msg = (
                        f"Invalid request: Disaggregated request received without "
                        f"bootstrap room id. {req.rid=}"
                    )
                    logger.error(error_msg)
                    recv_req.time_stats.trace_ctx.abort(
                        abort_info={"reason": error_msg}
                    )
                    prepare_abort(req, error_msg, status_code=HTTPStatus.BAD_REQUEST)
                    self.output_streamer.stream_output([req], req.return_logprob)
                    return

        elif (
            session_id in self.session_controller
            and not self.session_controller.get(session_id).close_on_finish
        ):
            # Session exists and is not closing: create request from session
            session = self.session_controller.get(session_id)
            req = session.create_req(
                recv_req,
                self.tokenizer,
                self.model_config.vocab_size,
                eos_token_ids=self.model_config.hf_eos_token_id,
            )
            # TODO: set trace context
            if self.metrics_reporter.enable_metrics:
                req.time_stats.set_metrics_collector(self.metrics_collector)
            if isinstance(req.finished_reason, FINISH_ABORT):
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

        else:
            # Session not found, or session is closing
            if session_id in self.session_controller:
                error_msg = (
                    f"Invalid request: close was requested for session {session_id}"
                )
            else:
                error_msg = f"Invalid request: session id {session_id} does not exist"
            req = Req(
                recv_req.rid,
                recv_req.input_text,
                recv_req.input_ids,
                recv_req.sampling_params,
                vocab_size=self.model_config.vocab_size,
                http_worker_ipc=recv_req.http_worker_ipc,
            )
            req.tokenizer = self.tokenizer
            req.set_finish_with_abort(error_msg)
            self.init_req_max_new_tokens(req)
            self._add_request_to_queue(req)
            return

        if self.spec_algorithm.is_dflash_family() or self._cross_schedule_mode:
            error_msg = validate_dflash_request(req, self.enable_overlap)
            if error_msg is not None:
                req.set_finish_with_abort(error_msg)
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return
        # Handle multimodal inputs
        if recv_req.mm_inputs is not None:
            image_inputs = self._get_multimodal_inputs(recv_req.mm_inputs)

            SessionController.adjust_mm_offsets(recv_req, req, image_inputs)

            # The following steps are already fast, execute locally on each rank.
            # Expand a single image token into multiple dummy tokens for receiving image embeddings.
            # The pad function is model-specific and can be None for some backends.
            if (
                not self._try_apply_padded_mm_input_ids(recv_req, req, image_inputs)
                and self.pad_input_ids_func
            ):
                req.origin_input_ids = array(
                    "q", self.pad_input_ids_func(req.origin_input_ids, image_inputs)
                )
            req.extend_image_inputs(image_inputs)
            self._maybe_compute_mrope_positions(req)

            if len(req.origin_input_ids) >= self.max_req_input_len:
                req.set_finish_with_abort(
                    error_msg=(
                        "Multimodal prompt is too long after expanding multimodal tokens. "
                        f"After expanding {len(req.origin_input_ids_unpadded)=} => {len(req.origin_input_ids)} >= {self.max_req_input_len}."
                    )
                )
                self.init_req_max_new_tokens(req)
                self._add_request_to_queue(req)
                return

        # initialize before returning
        self.init_req_max_new_tokens(req)

        # Validate prompt length
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            self.server_args.allow_auto_truncate,
        )
        if error_msg:
            req.set_finish_with_abort(error_msg)
            self._add_request_to_queue(req)
            return

        if not recv_req.return_logprob and recv_req.logprob_start_len != -1:
            # When return_logprob is False, logprob_start_len should be ignored
            recv_req.logprob_start_len = -1

        if recv_req.logprob_start_len == -1:
            if recv_req.return_logprob and recv_req.token_ids_logprob is None:
                # If logprob is required but neither token_ids_logprob nor logprob_start_len is
                # set, return the logprobs for output tokens by default
                req.logprob_start_len = len(req.origin_input_ids)
            elif req.is_prefill_only:
                # For prefill-only requests with logprob_start_len == -1, set logprob_start_len
                # beyond input sequence to skip input logprob computation entirely
                req.logprob_start_len = len(req.origin_input_ids)
            else:
                # If return_logprob is False, only the last token requires logprob computation
                req.logprob_start_len = -1
        else:
            req.logprob_start_len = recv_req.logprob_start_len

        if req.logprob_start_len > len(req.origin_input_ids):
            error_msg = f"{req.logprob_start_len=} is higher than the number of input tokens {len(req.origin_input_ids)=}. Please use a smaller logprob_start_len."
            req.logprob_start_len = -1
            req.set_finish_with_abort(error_msg)
            self._add_request_to_queue(req)
            return

        if recv_req.return_routed_experts:
            error_msg = None
            if recv_req.routed_experts_start_len < 0:
                error_msg = (
                    f"{recv_req.routed_experts_start_len=} is lower than 0. "
                    "Please use a non-negative routed_experts_start_len."
                )

            if recv_req.routed_experts_start_len > len(req.origin_input_ids):
                error_msg = (
                    f"{recv_req.routed_experts_start_len=} is higher than the "
                    f"number of input tokens {len(req.origin_input_ids)=}. Please "
                    f"use a smaller routed_experts_start_len."
                )

            if error_msg is not None:
                req.routed_experts_start_len = 0
                req.set_finish_with_abort(error_msg)
                self._add_request_to_queue(req)
                return

        added_to_grammar_queue = self.grammar_manager.process_req_with_grammar(req)
        if not added_to_grammar_queue:
            self._add_request_to_queue(req)

    def handle_batch_generate_request(
        self,
        recv_req: BatchTokenizedGenerateReqInput,
    ):
        """Handle optimized batch generate request."""
        logger.debug(f"Processing batch generate request with {len(recv_req)} requests")

        # Process each request in the batch
        for tokenized_req in recv_req:
            self.handle_generate_request(tokenized_req)

    def _prefetch_kvcache(self, req: Req):
        if not self.enable_hicache_storage:
            return
        req.init_next_round_input(self.tree_cache, cow_mamba=False)
        last_host_node = req.last_host_node
        # RANK-LOCAL: `backuped` means "full KV present in THIS rank's host
        # pool", and uneven DCP gives the ranks host pools of different sizes,
        # so it can be true here and false on a peer for the same node.
        locally_eligible = (
            last_host_node.backuped or last_host_node is self.tree_cache.root_node
        )
        # #580: when the tree cache decides prefetch participation by group
        # vote, skipping the call here would leave the peers alone in that
        # collective. Call unconditionally and let the vote decide; the local
        # verdict rides along as `locally_eligible`. Tree caches without the
        # predicate (and even-TP HiCache, where it is False) keep the plain
        # local gate and the byte-identical call below.
        group_decides = getattr(
            self.tree_cache, "prefetch_participation_is_collective", None
        )
        group_decides = bool(group_decides is not None and group_decides())
        if not locally_eligible and not group_decides:
            return

        if locally_eligible:
            last_hash = last_host_node.get_last_hash_value()
            matched_len = len(req.prefix_indices) + req.host_hit_length
            match_end = req._compute_max_prefix_len(len(req.full_untruncated_fill_ids))
            new_input_tokens = req.full_untruncated_fill_ids[matched_len:match_end]

            prefix_keys = (
                last_host_node.get_prefix_hash_values(last_host_node.parent)
                if self.tree_cache.hicache_storage_pass_prefix_keys
                else None
            )
        else:
            # Ineligible here: enter the vote carrying nothing. The empty token
            # list keeps every derived length at 0 on this rank, and the vote
            # will be negative, so no rank registers a prefetch.
            last_hash, new_input_tokens, prefix_keys = None, [], None

        if group_decides:
            self.tree_cache.prefetch_from_storage(
                req.rid,
                last_host_node,
                new_input_tokens,
                last_hash,
                prefix_keys,
                locally_eligible=locally_eligible,
            )
        else:
            self.tree_cache.prefetch_from_storage(
                req.rid,
                last_host_node,
                new_input_tokens,
                last_hash,
                prefix_keys,
            )

    def _add_request_to_queue(self, req: Req, is_retracted: bool = False):
        if not self._set_or_validate_priority(req):
            return
        # kv-session-offload: FCFS arrival order. Assigned once (a retracted
        # re-queue keeps its original arrival position). The admission order
        # is identical on every TP rank, so the counter is rank-uniform.
        if req.kv_arrival_seq is None:
            req.kv_arrival_seq = self._kv_arrival_ct
            self._kv_arrival_ct += 1
        if self.disaggregation_mode == DisaggregationMode.NULL:
            if self._abort_on_queued_limit(req):
                return
            self._prefetch_kvcache(req)
            self.waiting_queue.append(req)
            req.time_stats.set_wait_queue_entry_time()
        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            self._prefetch_kvcache(req)
            self.disagg_prefill_bootstrap_queue.add(
                req, self.model_config.num_key_value_heads
            )
            req.time_stats.set_prefill_bootstrap_queue_entry_time()
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.disagg_decode_prealloc_queue.add(req, is_retracted=is_retracted)
            if not is_retracted:
                req.time_stats.set_decode_prealloc_queue_entry_time()
            else:
                req.time_stats.set_retract_time()
        else:
            raise ValueError(f"Invalid {self.disaggregation_mode=}")

    def _set_or_validate_priority(self, req: Req) -> bool:
        """Set the default priority value, or abort the request based on the priority scheduling mode."""
        if self.enable_priority_scheduling and req.priority is None:
            if self.schedule_low_priority_values_first:
                req.priority = sys.maxsize
            else:
                req.priority = -sys.maxsize - 1
        elif (
            not self.enable_priority_scheduling
            and req.priority is not None
            and self.abort_on_priority_when_disabled
        ):
            abort_req = AbortReq(
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                    "message": "Using priority is disabled for this server. Please send a new request without a priority.",
                },
                rid=req.rid,
            )
            req.time_stats.trace_ctx.abort(abort_info=abort_req.finished_reason)
            self.ipc_channels.send_to_tokenizer.send_output(abort_req, req)
            return False
        return True

    def _abort_on_queued_limit(self, recv_req: Req) -> bool:
        """Abort an incoming or existing request if the waiting queue is full. Returns True if the incoming request is aborted."""
        if (
            self.max_queued_requests is None
            or len(self.waiting_queue) + 1 <= self.max_queued_requests
        ):
            return False

        # Reject the incoming request by default.
        req_to_abort = recv_req
        message = "The request queue is full."
        if self.enable_priority_scheduling:
            # With priority scheduling, consider aboritng an existing request based on the priority.
            # direction = 1  => smaller number = higher priority; -1 => larger number = higher priority.
            # max(...) + (direction * priority, queue_time_start) picks the least-preferred request.
            # Tie: later queue_time_start (newer) is evicted first. Preempt only if strictly better.
            direction = 1 if self.schedule_low_priority_values_first else -1
            key_fn = lambda item: (
                direction * item[1].priority,
                item[1].time_stats.wait_queue_entry_time,
            )
            idx, candidate_req = max(enumerate(self.waiting_queue), key=key_fn)
            abort_existing_req = (
                direction * recv_req.priority < direction * candidate_req.priority
            )
            if abort_existing_req:
                if self.enable_hicache_storage:
                    # Release prefetch events associated with the request
                    self.tree_cache.release_aborted_request(candidate_req.rid)
                elif self.enable_hierarchical_cache:
                    self.tree_cache.terminate_prefetch(candidate_req.rid)
                self.waiting_queue.pop(idx)
                req_to_abort = candidate_req
                message = "The request is aborted by a higher priority request."

        self.ipc_channels.send_to_tokenizer.send_output(
            AbortReq(
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                    "message": message,
                },
                rid=req_to_abort.rid,
            ),
            req_to_abort,
        )
        req_to_abort.time_stats.trace_ctx.abort(abort_info={"reason": message})
        return req_to_abort.rid == recv_req.rid

    def _abort_on_waiting_timeout(self):
        if (timeout_s := envs.SGLANG_REQ_WAITING_TIMEOUT.get()) <= 0:
            return

        # #610: the wall-clock verdict below is RANK-LOCAL (see
        # `_uniform_timeout_ballot`) and it FENCES A GROUP COLLECTIVE. The abort
        # body calls `tree_cache.release_aborted_request`, which enters a TP
        # barrier on every hierarchical cache class -- `_barrier_attn_groups` at
        # unified_radix_cache.py:2480 and hiradix_cache.py:1921,
        # `torch.distributed.barrier` at hi_mamba_radix_cache.py:2296. A rank
        # whose clock has crossed the deadline enters that barrier while its
        # peers, still a scheduler-loop jitter short of it, never call the
        # function at all: the aborting rank blocks against a partner that is
        # not coming. With hicache storage off the same divergence is still a
        # defect, one step milder -- the request leaves `waiting_queue` on a
        # subset of ranks and the next prefill batch is composed differently
        # per rank.
        #
        # This function is reached from `get_next_batch_to_run` (:3398), which
        # every rank runs once per iteration, so voting here keeps the
        # collective count rank-uniform.
        deleted_reqs = set()
        deadline = time.perf_counter() - timeout_s
        timed_out = self._uniform_timeout_ballot(
            [
                0 < req.time_stats.wait_queue_entry_time < deadline
                for req in self.waiting_queue
            ]
        )
        for req, expired in zip(self.waiting_queue, timed_out):
            if expired:
                if self.enable_hicache_storage:
                    # Release prefetch events associated with the request
                    self.tree_cache.release_aborted_request(req.rid)
                self.ipc_channels.send_to_tokenizer.send_output(
                    AbortReq(
                        finished_reason={
                            "type": "abort",
                            "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                            "message": "Request waiting timeout reached.",
                        },
                        rid=req.rid,
                    ),
                    req,
                )
                deleted_reqs.add(req)

        if deleted_reqs:
            self.waiting_queue = [
                req for req in self.waiting_queue if req not in deleted_reqs
            ]

    def handle_embedding_request(
        self,
        recv_req: TokenizedEmbeddingReqInput,
    ):
        req = Req(
            recv_req.rid,
            recv_req.input_text,
            recv_req.input_ids,
            recv_req.sampling_params,
            positional_embed_overrides=recv_req.positional_embed_overrides,
            token_type_ids=recv_req.token_type_ids,
            routed_dp_rank=recv_req.routed_dp_rank,
            priority=recv_req.priority,
            dimensions=recv_req.dimensions,
            lora_id=recv_req.lora_id,
            http_worker_ipc=recv_req.http_worker_ipc,
            time_stats=recv_req.time_stats,
            return_pooled_hidden_states=recv_req.return_pooled_hidden_states,
            multi_item_delimiter_indices=recv_req.multi_item_delimiter_indices,
        )
        req.tokenizer = self.tokenizer

        # Handle multimodal inputs
        if recv_req.mm_inputs is not None:
            image_inputs = self._get_multimodal_inputs(recv_req.mm_inputs)
            # Expand a single image token into multiple dummy tokens for receiving image embeddings
            # The `pad_input_ids_func` is model-specific and may be None for
            # embedding models or models not requiring special padding.
            # If None, `req.origin_input_ids` is expected to be correctly populated already.
            if (
                not self._try_apply_padded_mm_input_ids(recv_req, req, image_inputs)
                and self.pad_input_ids_func
            ):
                # See companion call site above for the array.array wrap rationale.
                req.origin_input_ids = array(
                    "q", self.pad_input_ids_func(req.origin_input_ids, image_inputs)
                )

            req.extend_image_inputs(image_inputs)
            self._maybe_compute_mrope_positions(req)

            if len(req.origin_input_ids) >= self.max_req_input_len:
                req.set_finish_with_abort(
                    error_msg=(
                        "Multimodal prompt is too long after expanding multimodal tokens. "
                        f"After expanding {len(req.origin_input_ids_unpadded)=} => {len(req.origin_input_ids)} >= {self.max_req_input_len}."
                    )
                )
                self._add_request_to_queue(req)
                return

        # Validate prompts length
        error_msg = validate_input_length(
            req,
            self.max_req_input_len,
            self.server_args.allow_auto_truncate,
        )
        if error_msg:
            self._add_request_to_queue(req)
            return

        # Copy more attributes
        req.logprob_start_len = -1
        self._add_request_to_queue(req)

    def handle_batch_embedding_request(
        self,
        recv_req: BatchTokenizedEmbeddingReqInput,
    ):
        """Handle optimized batch embedding request."""
        logger.debug(
            f"Processing batch embedding request with {len(recv_req)} requests"
        )

        # Process each request in the batch
        for tokenized_req in recv_req:
            self.handle_embedding_request(tokenized_req)

    def stash_chunked_request(self, req: Req):
        maybe_cache_unfinished_req(req, self.tree_cache, chunked=True)

    def process_pending_chunked_abort(self) -> None:
        """Abort an in-flight chunked-prefill request once it is safe to do so.

        ``abort_request`` only records the target in ``_pending_chunked_abort_req``
        (tearing it down mid-iteration is unsafe). Clearing ``chunked_req`` here at
        the top of the scheduling step stops the next chunk from launching; the
        chunk already launched is drained when its result is resolved. Under overlap
        the result lands a step later, so the batch-result processors keep
        ``inflight_middle_chunks`` accounting intact and skip the aborted chunk:
        ``process_batch_result_disagg_prefill`` via its ``is_aborted`` drop, and
        ``process_batch_result_prefill`` via its chunked branch (the finished req
        is excluded from streaming and its logprob offset is still accounted).
        Mirrors ``handle_bootstrap_failure``.
        """
        req = self._pending_chunked_abort_req
        if req is None:
            return
        if self.chunked_req is not req:
            # Already past chunked prefill; the running-batch abort path handles
            # it. Drop the marker once the request is actually gone.
            if req.finished() or req.req_pool_idx is None:
                self._pending_chunked_abort_req = None
            return

        prepare_abort(req, "Aborted")
        req.time_stats.trace_ctx.abort(abort_info={"reason": "Aborted"})
        req.to_finish = None
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            req.disagg_kv_sender.abort()
            maybe_release_metadata_buffer(
                req, self.req_to_metadata_buffer_idx_allocator
            )
            req.pending_bootstrap = False
        if self.enable_hicache_storage:
            self.tree_cache.release_aborted_request(req.rid)
        if (
            req.req_pool_idx is not None or self.tree_cache.supports_mamba()
        ) and not req.kv_committed_freed:
            release_kv_cache(req, self.tree_cache, is_insert=False)

        self.chunked_req = None
        self._pending_chunked_abort_req = None
        self.ipc_channels.send_to_tokenizer.send_output(AbortReq(rid=req.rid), req)
        logger.debug(f"Abort chunked prefill request. {req.rid=}")

    def _build_hisparse_decode_batch(self, reqs):
        """Build a ScheduleBatch for hisparse requests transitioning from staging to decode."""
        device = self.device

        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=self.enable_overlap,
            spec_algorithm=self.spec_algorithm,
        )

        req_pool_indices = [r.req_pool_idx for r in reqs]
        batch.req_pool_indices = torch.tensor(
            req_pool_indices, dtype=torch.int64, device=device
        )
        batch.req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)
        seq_lens = [len(r.origin_input_ids) + len(r.output_ids) - 1 for r in reqs]
        batch.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=device)
        batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        batch.orig_seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        batch.seq_lens_sum = sum(seq_lens)
        # Stash last token into relay; resolve_forward_inputs will gather.
        last_tokens = torch.tensor(
            [r.output_ids[-1] for r in reqs], dtype=torch.int64, device=device
        )
        self.future_map.stash(
            batch.req_pool_indices, RelayPayload(bonus_tokens=last_tokens)
        )
        batch.input_ids = None

        if batch.return_logprob:
            batch.top_logprobs_nums = [r.logprob.top_logprobs_num for r in reqs]
            batch.token_ids_logprobs = [list(r.origin_input_ids) for r in reqs]

        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, self.model_config.vocab_size
        )
        # todo hisparse, maybe other info to contain for the new batch
        return batch

    @scheduler_nvtx_method("scheduler.get_next_batch_to_run")
    def _update_uniform_pool_budget(self) -> None:
        """MIN-reduce this iteration's pool headroom across the TP group.

        The rank-uniform counterpart to ``token_to_kv_pool_allocator.
        available_size()``, which under uneven DCP/TP differs per rank
        because the pools do (weighted ownership). Anything that decides a
        BRANCH from the local value can split the group across branches that
        carry different collectives -- the rank-local-test-before-a-group-
        collective family.

        COST, named exactly: ONE MIN ``all_reduce`` of a ONE-element int64
        CPU tensor -- 8 bytes -- on ``tp_cpu_group``, i.e. the gloo CPU group,
        NOT the device group and not BAR1. It therefore neither touches the
        BAR1 aperture nor serialises with the forward stream; it is a host
        synchronisation point once per scheduler iteration. World size 1 takes
        no collective at all and is byte-identical to the previous path.

        (The offload variant packs two elements, because it also reduces
        available+evictable for the prefill admission budget. This path needs
        only the decode-mem quantity, so it reduces one.)

        When the offload manager is active it has ALREADY done exactly this
        reduce for this iteration (``update_dcp_admission_state``), so the
        value is read from there rather than reduced a second time -- two
        reduces would be two chances for the counts to diverge, which is the
        failure this is closing.

        Valid for the whole iteration, on the same argument the offload
        version documents: between here and the decode-mem check the
        scheduler only builds batches from offsets and defers real allocation
        to the forward, so the snapshot still holds.
        """
        kvso = self.kv_session_offload
        if kvso is not None:
            self._uniform_min_avail = int(kvso.dcp_min_avail())
            # The offload manager only exists under uneven DCP, so the pools
            # ARE uneven here; publish unconditionally (no max to compare).
            self._publish_uniform_evict_floor(self._uniform_min_avail)
            # #639, A NAMED GAP rather than a silent one. The host floor does
            # NOT ride the offload manager's reduce, and this branch's whole
            # contract is that it takes NO reduce of its own -- "a second
            # reduce here would be a second chance for the collective counts
            # to diverge", pinned by
            # test_uniform_decode_mem_603.test_the_offload_manager_is_not_reduced_twice.
            # Adding one to carry the host term would trade a divergence this
            # branch has closed for one it has not, on a path that cannot be
            # exercised without --enable-kv-session-offload.
            #
            # So under kv-session-offload the host floor stays OFF and
            # `write_backup` keeps its pre-#639 rank-local gate. The right
            # close is to widen `update_dcp_admission_state`'s existing packed
            # reduce by one pair, in kv_session_offload.py, where the reduce
            # already lives -- not here.
            self._publish_uniform_host_floor(None)
            # #639b: the mamba floor is off here for the same reason and with
            # the same named gap -- this branch's contract is that it takes NO
            # reduce of its own, and the offload manager's existing reduce
            # does not carry a mamba term. Under --enable-kv-session-offload
            # the mamba eviction keeps its pre-#639b rank-local gate. The right
            # close is to widen `update_dcp_admission_state`'s packed reduce in
            # kv_session_offload.py, where the reduce already lives, not to add
            # a second one here.
            self._publish_uniform_mamba_floor(None)
            # The offload manager reduced the admission quantity itself in
            # `update_dcp_admission_state`; `get_new_batch_prefill` reads it
            # from there, so nothing is stored here.
            self._uniform_budget_deficit = 0
            # #791b: NAMED GAP, same shape as the #639 host floor above --
            # this branch's contract is that it takes no reduce of its own,
            # so the prefetch ballot stays OFF under kv-session-offload and
            # admission keeps its rank-local prefetch verdict there. The
            # right close is to widen `update_dcp_admission_state`'s packed
            # reduce, where the reduce already lives.
            self._uniform_prefetch_ballot = None
            # #794: NAMED GAP, same shape as the #639 host floor above. This
            # branch takes no reduce of its own, so the corridor width stays
            # unreduced and the consumer declines to narrow rather than narrow
            # on a rank-local reading. The right close is one more term in
            # `update_dcp_admission_state`'s packed reduce, where it already
            # lives.
            self._uniform_corridor_width = None
            return

        alloc = self.token_to_kv_pool_allocator
        local_avail = int(alloc.available_size())
        grp = getattr(self, "tp_cpu_group", None)
        if grp is None or torch.distributed.get_world_size(grp) <= 1:
            # #788: SAY IT ONCE, LOUDLY, THAT THE FLOORS ARE OFF.
            #
            # The comment below ("nothing to diverge from") is true for TP and
            # FALSE for PP. This reduce group is tp_cpu_group, so on a
            # TP=1/PP=3 boot it has one member on every rank and all three
            # uniformity floors switch off -- while three PP ranks go on
            # deriving admission from purely local availability and prefix
            # matches. That is the measured cause of a pipeline deadlock, so
            # the condition should be readable from the boot log rather than
            # inferred from source months later.
            if not getattr(self, "_uniform_floor_scope_logged", False):
                self._uniform_floor_scope_logged = True
                logger.info(
                    "#788 UNIFORM-FLOOR SCOPE: tp_cpu_group world=%d -> floors OFF "
                    "(evict/host/mamba). pp_size=%d tp_size=%d. With pp_size>1 the "
                    "ranks that must agree are NOT in this reduce group.",
                    0 if grp is None else int(torch.distributed.get_world_size(grp)),
                    int(getattr(self.server_args, "pp_size", 1) or 1),
                    int(getattr(self.server_args, "tp_size", 1) or 1),
                )
            self._uniform_min_avail = local_avail
            self._uniform_budget_deficit = 0
            # One rank: nothing to diverge from, so the floor stays OFF and
            # every cache-mutation trigger keeps reading its live local value
            # -- byte-identical to the pre-#616g path.
            self._publish_uniform_evict_floor(None)
            self._publish_uniform_host_floor(None)
            self._publish_uniform_mamba_floor(None)
            # #791b: one rank -- nothing to diverge from, ballot off, the
            # local prefetch verdict is already the group verdict.
            self._uniform_prefetch_ballot = None
            # #794: one rank in this group -- the local corridor decision IS
            # the group decision, taken at the call site.
            self._uniform_corridor_width = None
            return
        # #610: PREFILL ADMISSION rides on this same reduce. `PrefillAdder`'s
        # `rem_total_tokens` / `cur_rem_tokens` (schedule_policy.py:681/719) read
        # `available_size() + evictable_size()` -- rank-LOCAL under uneven DCP,
        # where the ranks own weighted shares of the token axis. `budget_state`
        # (:808) turns that straight into NO_TOKEN, so the binding rank stops
        # admitting while the slack ranks continue: `can_run_list` differs, the
        # prefill batch is composed differently per rank, and the per-layer TP
        # collectives in the forward are entered with mismatched shapes. The
        # kv-session-offload path has been pinned against exactly this since
        # `update_dcp_admission_state` (kv_session_offload.py:2741), but that
        # pin only exists when the offload manager is constructed -- with
        # `--enable-kv-session-offload` off, which is the default, prefill
        # admission was still deciding rank-locally. Same defect, same fix, one
        # branch further out.
        #
        # MIN-ballot, not gate removal: the budget every rank admits against is
        # the BINDING rank's, so no rank is asked to hold a request it cannot.
        # The per-rank surplus is exported as a non-negative deficit the adder
        # subtracts, which is the representation the offload variant already
        # uses and the adder already consumes (`dcp_avail_deficit`).
        #
        # PACK SIZE is decided by `uneven_dcp_active(dcp_size)`, a replicated
        # config-derived predicate (the token-ratio vector is installed
        # identically on every rank), so the payload width is rank-uniform. Even
        # TP keeps the one-element payload and a structurally zero deficit --
        # byte-identical, no behaviour change on the default path.
        from sglang.srt.distributed.utils import uneven_dcp_active

        pin_admission = uneven_dcp_active(self.server_args.dcp_size)
        local_admission = 0
        if pin_admission:
            tree = self.tree_cache
            fe = getattr(tree, "full_evictable_size", None)
            # Mirrors `update_dcp_admission_state`'s quantity exactly, including
            # its known simplification: the all-SWA adder reads
            # `swa_evictable_size` instead. SWA models route to
            # UnifiedRadixCache and are out of this path's reach.
            local_evict = int(fe() if fe is not None else tree.evictable_size())
            local_admission = local_avail + local_evict
        # #616g: `-local_avail` rides the SAME reduce so one MIN yields both
        # the group minimum and (negated) the group MAXIMUM. The pair is what
        # decides whether the pools are uneven AT ALL this iteration, which is
        # the activation predicate for the eviction floor below -- derived from
        # a collective, so it is rank-uniform by construction and needs no
        # server-arg coupling. Width stays rank-uniform: the element is added
        # unconditionally, and `pin_admission` (the only other width term) is a
        # replicated config predicate.
        vals = (
            [local_avail, local_admission, -local_avail]
            if pin_admission
            else [local_avail, -local_avail]
        )
        # #639: the HOST tier's availability rides the SAME reduce, as a
        # (x, -x) pair so one MIN yields its group min and max -- the same
        # trick and the same activation predicate the device floor above
        # uses. Appended UNCONDITIONALLY: a boot with no host tier still
        # contributes the pair, carrying `_HOST_AVAIL_ABSENT`, so the payload
        # width never depends on a per-rank capability. (Whether a host tier
        # exists is `--enable-hierarchical-cache`, a replicated server arg, so
        # the sentinel is all-or-nothing in practice; contributing it
        # unconditionally is what makes that a property of the code rather
        # than of the flagset.)
        local_host_avail = self._local_host_avail()
        vals = vals + [local_host_avail, -local_host_avail]
        # #639b: the MAMBA slot pool's availability rides the SAME reduce, as
        # a third (x, -x) pair, on the identical argument the host pair above
        # documents -- one MIN yields its group min and max, and the
        # min-vs-max comparison is the activation predicate. Appended
        # UNCONDITIONALLY, carrying `_MAMBA_AVAIL_ABSENT` on a boot with no
        # mamba pool, so the payload width never depends on a per-rank
        # capability. Whether a mamba pool exists is a property of the model
        # (hybrid SSM or not), replicated across ranks, so the sentinel is
        # all-or-nothing in practice; contributing it unconditionally is what
        # makes the uniform width a property of the code rather than of the
        # checkpoint.
        local_mamba_avail = self._local_mamba_avail()
        vals = vals + [local_mamba_avail, -local_mamba_avail]
        # #794: THE CORRIDOR WIDTH CEILING rides this reduce too, and it has to.
        #
        # The flip changes the topology under this scheduler AT RUNTIME. In the
        # PP phase this group has world 1 and PP0 decides the pass geometry for
        # the ring; in the TP phase the SAME process reads pp_size=1 and
        # tp_cpu_group world=3, every rank builds its own prefill batch, and a
        # rank-local width cut would split the group. The first metal run of the
        # actuator printed exactly that, from all three ranks:
        #     #794 CORRIDOR WIDTH ACTUATOR OFF on this topology:
        #     tp_cpu_group world=3 with pp_size=1
        # -- i.e. the actuator switched itself off in the phase where this fork
        # runs prefill (--phase-flip-purity prefill_in_tp) and where the
        # 2026-08-21 17:33 OOM actually happened, inside the first flip. A gap
        # that swallows the whole use case is not a gap, it is an outage.
        #
        # ONE MORE INT ON A REDUCE THAT ALREADY RUNS, which is what the gap note
        # itself named as the right close. Contributed UNCONDITIONALLY so the
        # payload width never depends on a per-rank capability: a rank that
        # cannot price, or has no guard, contributes the configured width and is
        # therefore never the binding vote. MIN, because the group may take on
        # only what its tightest card can fund.
        #
        # Placed BETWEEN the mamba pair and the ballot on purpose: everything
        # above is indexed from the head and the ballot is indexed from the
        # TAIL, so inserting here leaves both readings intact.
        _corridor_at = len(vals)
        vals = vals + [self._local_corridor_width_ceiling()]
        # #791b: THE PREFETCH BALLOT rides the SAME reduce (instr22/instr23:
        # the admission loop's rank-local prefetch verdict split the TP
        # replicas -- one rank admitted a batch its peers declined, three
        # lockstep families crossed, barlink aborted the group 150 s later).
        # The drain that feeds the local verdicts is pulled forward to HERE
        # -- rank-local, no collective, and this site runs exactly once per
        # TP-loop iteration -- and memoised for `_get_new_batch_prefill_raw`
        # to consume once, so the pass still drains exactly once. Appended
        # AFTER the mamba pair, so every existing index below keeps its
        # meaning. See prefetch_ballot.py for the layout and the MIN==AND
        # argument.
        _ballot_verdicts = self._drain_prefetch_progress()
        self._pass_prefetch_verdicts = _ballot_verdicts
        _ballot_rids = [
            req.rid
            for req in self.waiting_queue[: prefetch_ballot.PREFETCH_BALLOT_SLOTS]
        ]
        vals = vals + prefetch_ballot.build_prefetch_ballot_payload(
            _ballot_rids, _ballot_verdicts
        )
        t = torch.tensor(vals, dtype=torch.int64)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=grp)
        # #794: the group's tightest corridor width, read back before the
        # ballot so a later change to the ballot layout cannot silently move it.
        self._uniform_corridor_width = int(t[_corridor_at])
        _ballot_at = len(vals) - (prefetch_ballot.PREFETCH_BALLOT_SLOTS + 2)
        self._uniform_prefetch_ballot = prefetch_ballot.unpack_prefetch_ballot(
            t[_ballot_at:].tolist(), _ballot_rids
        )
        if self._uniform_prefetch_ballot is None and not getattr(
            self, "_prefetch_ballot_mismatch_logged", False
        ):
            # A divergent queue HEAD is a deeper breakage than a divergent
            # prefetch verdict; say so once and fall back to the local
            # verdict (the status quo ante) rather than papering over it.
            self._prefetch_ballot_mismatch_logged = True
            logger.warning(
                "#791b PREFETCH-BALLOT digest mismatch: the TP ranks disagree "
                "about the first %d rids of waiting_queue (this rank digest=%d, "
                "group min=%d max=%d). Ballot void for this pass; admission "
                "falls back to the rank-local prefetch verdict.",
                prefetch_ballot.PREFETCH_BALLOT_SLOTS,
                prefetch_ballot.prefetch_ballot_digest(_ballot_rids),
                int(t[_ballot_at].item()),
                -int(t[_ballot_at + 1].item()),
            )
        self._uniform_min_avail = int(t[0].item())
        # >= 0: the local budget can only exceed the group minimum.
        self._uniform_budget_deficit = (
            local_admission - int(t[1].item()) if pin_admission else 0
        )
        # Indices are explicit rather than negative, and now ALL of them are.
        # #639b appended the mamba pair after the host pair, so the `t[-2]` /
        # `t[-1]` the host floor used to read would have silently started
        # reading MAMBA availability -- the exact hazard the previous revision
        # of this comment named. Layout, in payload order:
        #   [avail, (admission,) -avail, host, -host, mamba, -mamba]
        max_avail_at = 2 if pin_admission else 1
        host_at = 3 if pin_admission else 2
        mamba_at = 5 if pin_admission else 4
        self._publish_uniform_evict_floor(
            int(t[0].item()), max_avail=-int(t[max_avail_at].item())
        )
        self._publish_uniform_host_floor(
            int(t[host_at].item()), max_host_avail=-int(t[host_at + 1].item())
        )
        self._publish_uniform_mamba_floor(
            int(t[mamba_at].item()), max_mamba_avail=-int(t[mamba_at + 1].item())
        )

    def _publish_uniform_evict_floor(
        self, min_avail: Optional[int], max_avail: Optional[int] = None
    ) -> None:
        """Publish this iteration's rank-uniform availability floor on the tree
        cache, for the CACHE-MUTATION triggers to decide from (#616g).

        WHY, measured: #603 and #610 made the decode-mem and prefill-admission
        DECISIONS rank-uniform, but both left eviction as an explicitly local
        side effect ("Still evict locally for the space"). Under uneven pools
        that side effect is itself a divergence source, one level below the
        decisions:

          `evict_from_tree_cache` fires on `available_size() < num_tokens`,
          where the DEMAND is replicated (`batch.extend_num_tokens`) but the
          AVAILABILITY is this rank's own shard -- 179825 / 143860 / 136667
          tokens on the 2026-08-06 boot. The rank with the roomiest pool
          declines to evict while the tight ranks evict, so the radix trees
          stop being replicas. `match_prefix` then returns a rank-dependent
          prefix, `prepare_for_extend` computes
          `extend_num_tokens = sum(seq_len - len(prefix_indices))` from it,
          and every per-layer TP all_reduce of that forward is entered with a
          rank-dependent token count. That is the 21:52:25 wedge exactly:
          all three ranks parked in the SAME collective at the SAME layer_idx
          with rank 0 reducing 1690 tokens against 1818 on its peers.

        The floor is the MIN over ranks of the same quantity, taken from the
        reduce that already runs once per iteration, pre-branch -- NO new
        collective and no new synchronisation point.

        DIRECTION IS SAFE BY CONSTRUCTION: min <= local, so
        `floor < num_tokens` is true whenever the local test was, and
        sometimes when it was not. Every rank therefore evicts at least as
        often as before the fix; under-eviction (which would turn into an
        allocator OOM) is arithmetically impossible. The price is that slack
        ranks drop cache they did not personally need, which is what keeping
        the replicas identical costs.

        ACTIVATION is itself collective-derived: the floor is published only
        when the group's min and max availability DIFFER, i.e. when the pools
        are actually uneven this iteration. On even TP the two are equal, the
        floor stays None, and every trigger reads its live local value exactly
        as before -- byte-identical, no behaviour change on the default path.
        """
        tree = getattr(self, "tree_cache", None)
        if tree is None:
            return
        if min_avail is None or (max_avail is not None and min_avail >= max_avail):
            tree.uniform_avail_floor = None
            # #694: the ledger corrects a floor; with no floor there is nothing
            # to correct, and a value left over from a previous generation
            # would be charged against the NEXT one.
            tree.uniform_admitted_since_floor = 0
            return
        tree.uniform_avail_floor = int(min_avail)
        # #694: RESET IN THE SAME CALL THAT PUBLISHES, so the ledger never
        # outlives the number it corrects. The floor is a snapshot of this
        # instant; allocations charged against the PREVIOUS snapshot have
        # already been reflected in this new MIN, and charging them twice would
        # drive the predicate to zero and evict on every allocation.
        tree.uniform_admitted_since_floor = 0

    #: What a rank with no host tier contributes to the host pair, so the
    #: reduce payload width never depends on a per-rank capability. Large
    #: enough that a MIN against any real availability always loses; small
    #: enough to stay well inside int64 after negation.
    _HOST_AVAIL_ABSENT = 1 << 62

    def _local_host_avail(self) -> int:
        """This rank's free HOST-tier slots, or ``_HOST_AVAIL_ABSENT`` (#639).

        Read through the tree cache's controller because that is the object
        ``write_backup`` reads it from -- the pin and the gate must sample the
        same pool or the floor would describe a different tier than the one it
        governs.
        """
        tree = getattr(self, "tree_cache", None)
        controller = getattr(tree, "cache_controller", None) if tree else None
        pool = getattr(controller, "mem_pool_host", None) if controller else None
        if pool is None:
            return self._HOST_AVAIL_ABSENT
        try:
            return int(pool.available_size())
        except Exception:  # noqa: BLE001
            # A diagnostic must never be what turns a scheduler iteration into
            # a crash; an unreadable pool means "no floor", i.e. the pre-#639
            # local behaviour.
            return self._HOST_AVAIL_ABSENT

    def _publish_uniform_host_floor(
        self,
        min_host_avail: Optional[int],
        max_host_avail: Optional[int] = None,
    ) -> None:
        """Publish this iteration's rank-uniform HOST availability floor on
        the tree cache, for ``write_backup`` to decide from (#639).

        WHY, measured. #616g pinned the DEVICE-side mutation triggers and
        explicitly left the host tier alone, on the reading that under
        ``write_through`` "the eviction path gates on write_policy ==
        'write_back', not on backup state, so the chain to the device tree is
        NOT established". ``UnifiedRadixCache._evict_device_leaf`` gates on
        ``node.backuped`` FIRST and consults ``write_policy`` only inside that
        branch -- and its write_through arm is the one that removes the node
        from the tree entirely. So the chain IS established, and it exists
        only on the policy this deployment runs:

          ``write_backup`` refuses when this rank's own host pool cannot hold
          the node (359652 / 287722 / 273336 slots on the crashing boot,
          against a REPLICATED node length). The roomy rank backs the node up
          and demotes it on eviction -- it stays in the tree, matchable, and
          ``load_back`` can restore it. The tight ranks refuse and DELETE it.
          ``match_prefix`` then returns a rank-dependent ``prefix_indices``,
          ``prepare_for_extend`` turns that into a rank-dependent
          ``extend_num_tokens``, and every per-layer TP all_reduce of that
          forward is entered with a shape that cannot pair. Four specimens,
          all with the roomiest rank LOW: 1690 vs 1818, 828 vs 2048, 912 vs
          2048, 914 vs 2048.

        The floor is the group MIN of the host ``available_size()``, taken
        from the reduce that already runs once per iteration, pre-branch --
        NO new collective on the default path.

        DIRECTION IS SAFE BY CONSTRUCTION: min <= local, so the pinned gate
        refuses whenever the local gate refused and sometimes when it did not.
        Over-admission into the host pool -- the only way this pin could
        itself become a fault -- is arithmetically impossible. The price is
        that the rank with the roomiest host tier caches less than it could,
        which is what keeping the replicas identical costs.

        ACTIVATION is collective-derived, exactly as the device floor's is:
        published only when the group's min and max host availability differ.
        When they agree, the local test compares the same left side against a
        replicated right side on every rank and is already uniform, so there
        is nothing to fix and the path stays byte-identical.
        """
        tree = getattr(self, "tree_cache", None)
        if tree is None:
            return
        absent = self._HOST_AVAIL_ABSENT
        if (
            min_host_avail is None
            or min_host_avail >= absent
            or (max_host_avail is not None and min_host_avail >= max_host_avail)
        ):
            tree.uniform_host_avail_floor = None
            tree.uniform_host_admitted_since_floor = 0
            tree.uniform_host_refusals_since_floor = 0
            return
        tree.uniform_host_avail_floor = int(min_host_avail)
        # #645: the floor above is a snapshot taken at the TOP of the
        # iteration, but the backups that read it run at the END of it
        # (`process_batch_result_prefill` -> `cache_unfinished_req` ->
        # `insert` -> `write_backup`). Without a ledger it keeps counting as
        # free every slot this iteration's own backups have already taken, so
        # the rank whose pool IS the floor runs out for real while its peers,
        # reading the same optimistic number, sail on -- and the tight rank
        # then takes a rank-LOCAL host-eviction branch its peers do not.
        # Reset here, in the same call that publishes the number it corrects,
        # so the ledger can never outlive its floor. Every path through
        # `_update_uniform_pool_budget` reaches this publisher exactly once
        # per iteration, so the reset inherits that cadence for free.
        tree.uniform_host_admitted_since_floor = 0
        # Same cadence for the refusal counter, which only exists to keep the
        # #645 warning to one line per published floor instead of one per
        # insert.
        tree.uniform_host_refusals_since_floor = 0

    #: What a rank with no mamba pool contributes to the mamba pair, so the
    #: reduce payload width never depends on a per-rank capability. Same role
    #: and same magnitude argument as ``_HOST_AVAIL_ABSENT``.
    _MAMBA_AVAIL_ABSENT = 1 << 62

    def _local_mamba_avail(self) -> int:
        """This rank's free MAMBA slots, or ``_MAMBA_AVAIL_ABSENT`` (#639b).

        Read off ``req_to_token_pool.mamba_allocator`` because that is the
        allocator every gated call site allocates from -- the pin and the
        gates must sample the same pool or the floor would describe a
        different pool than the one it governs.

        ``available_size()`` and not ``schedulable_available_size()``: the
        raw free-slot count is defined identically on both allocator classes,
        while the schedulable view is the LARGER of the two on the shared
        ``UnifiedMambaSlotAllocator``. Reducing the smaller, universally
        defined quantity is what lets ``uniform_mamba_avail_for_evict`` stay
        direction-safe for callers reading either one.
        """
        pool = getattr(self, "req_to_token_pool", None)
        alloc = getattr(pool, "mamba_allocator", None) if pool else None
        if alloc is None:
            return self._MAMBA_AVAIL_ABSENT
        try:
            return int(alloc.available_size())
        except Exception:  # noqa: BLE001
            # A diagnostic must never be what turns a scheduler iteration into
            # a crash; an unreadable pool means "no floor", i.e. the pre-#639b
            # local behaviour.
            return self._MAMBA_AVAIL_ABSENT

    def _publish_uniform_mamba_floor(
        self,
        min_mamba_avail: Optional[int],
        max_mamba_avail: Optional[int] = None,
    ) -> None:
        """Publish this iteration's rank-uniform MAMBA availability floor on
        the tree cache, for the mamba eviction triggers to decide from (#639b).

        WHY, measured. The two floors above pin the KV token axis on the
        device and on the host. Neither reaches the mamba slot pool, and the
        2026-08-07 07:45 and 10:04 crashes are that gap:

            rank 0: n=1 sum=19711 has_prefix=True [19711]
            rank 1: n=1 sum=16957 has_prefix=True [16957]

        ``alloc_req_slots`` evicts on
        ``mamba_available_size < num_reqs * factor`` -- a REPLICATED demand
        against this rank's OWN occupancy -- and then evicts
        ``mamba_state_needed - mamba_available_size`` slots, a magnitude that
        is itself rank-local. ``MambaComponent.evict_component`` tombstones
        the mamba component of a node and LEAVES ITS KV, so the node stays in
        the tree on both ranks but stops satisfying the mamba validator on
        the rank that evicted. ``_match_prefix_helper`` advances only while
        every component is resident, so that rank's match stops short, its
        extend covers more tokens, it takes more slots, and it evicts more
        next round -- which is how the two ranks got 2754 tokens apart.

        A pre-existing comment in ``MambaRadixCache._alloc_mamba_slot``
        concluded this path was "rank-uniform without a collective" because
        ``max_mamba_cache_size`` is min-reduced at startup. That reasoning
        does not hold and is corrected in this change: a uniform pool SIZE is
        not a uniform eviction OUTCOME. Which node is tombstoned comes from
        ``drive_eviction``'s rank-local LRU walk, and occupancy diverges from
        lock_ref history and from the rank-local "skip this cache insert"
        degradation that an exhausted pool triggers.

        The floor is the group MIN of the mamba allocator's
        ``available_size()``, taken from the reduce that already runs once per
        iteration, pre-branch -- NO new collective.

        DIRECTION IS SAFE BY CONSTRUCTION: min <= local, so every rank evicts
        at least as often, and at least as much, as before. Under-eviction --
        which would turn into mamba slot starvation -- is arithmetically
        impossible. The price is that the rank with the roomiest mamba pool
        drops checkpoints it did not personally need.

        ACTIVATION is collective-derived, exactly as both siblings are:
        published only when the group's min and max mamba availability
        DIFFER. When they agree, every rank's local test already compares the
        same two numbers and the path stays byte-identical.
        """
        tree = getattr(self, "tree_cache", None)
        if tree is None:
            return
        absent = self._MAMBA_AVAIL_ABSENT
        if (
            min_mamba_avail is None
            or min_mamba_avail >= absent
            or (max_mamba_avail is not None and min_mamba_avail >= max_mamba_avail)
        ):
            tree.uniform_mamba_avail_floor = None
            return
        tree.uniform_mamba_avail_floor = int(min_mamba_avail)

    def uniform_budget_deficit(self) -> int:
        """This iteration's non-negative prefill-admission surplus over the
        group minimum, for `PrefillAdder` to subtract (#610).

        0 when the pin is not active (even TP, world size 1) and 0 when the
        per-iteration value was never computed -- the same fallback shape
        `KVSessionOffload.dcp_budget_deficit` documents, so an entry point that
        reaches prefill without passing `_update_uniform_pool_budget` (the PP
        mixin's own loop) keeps today's behaviour rather than reading a stale
        reduce from a differently-shaped path.
        """
        return int(getattr(self, "_uniform_budget_deficit", 0))

    #: #605 corridor sampler state. Class-level defaults rather than two more
    #: lines in ``__init__``: a scheduler that never arms the sampler then
    #: carries no per-instance attribute for it at all, which is the strictest
    #: reading of "byte-identical when off".
    _corridor_trace = None
    _corridor_trace_armed = False
    #: Monotonic deadline for the next breach read, and the deepest floor
    #: already reported. See _corridor_trace_tick.
    _corridor_trace_next_check = 0.0
    _corridor_trace_reported_floor = None

    #: How often the armed trace is asked whether the law still holds. The
    #: SAMPLING is 100 ms and unchanged; this is only how often the verdict
    #: is read, and it is deliberately coarse -- the report is for an
    #: operator, and a breach that has happened cannot be un-happened by
    #: reading it sooner.
    _CORRIDOR_BREACH_CHECK_S = 10.0

    def _flight_serving_tick(self) -> None:
        """One paced VRAM flight sample while serving (#684).

        WHY, MEASURED. The recorder's last boot post is ``first_forward``. On
        2026-08-16 an instance died 36 minutes later with ``76.38 MiB is free
        ... Process 1920108 has 4.29 GiB memory in use``, and naming that
        process took hours -- log archaeology plus a pid-clock interpolation
        across two boots' ``boot_id`` fields. Every fact needed to answer it in
        one line was already computed by the recorder's own NVML view, which
        includes the full pid->bytes map of everyone on the card. Nothing was
        marking after boot.

        NOT A DUPLICATE OF ``_corridor_trace_tick``. That one arms a 100 ms
        sampler for the corridor LAW, keeping a fixed-size RAM ring -- which
        dies with the process that crashes, and whose ``Sample`` discards the
        per-pid map it reads. This appends to a FILE, so it survives the
        crash; the surviving boot marks are what made that pid clock
        calibratable at all.

        HERE, AND ONCE PER ITERATION, for the reason the two lines above this
        call site give: every rank reaches it exactly once per round, so the
        cadence is replicated and the per-rank files line up round for round,
        which is what makes them comparable across ranks. Unlike its
        neighbours this needs no collective and takes no branch -- it is
        write-only -- so it cannot make two ranks disagree about anything.

        The pacing lives in the recorder (default 30 s), so the cost here is
        one monotonic clock read per iteration, and exactly zero when the
        recorder is not armed.

        THE IMPORT IS OUTSIDE THE GUARD, DELIBERATELY. ``flight_recorder`` is
        imported inside ``run_scheduler_process``, not at module scope, so a
        module-level reference here is a ``NameError`` -- and the first version
        of this method had one, inside a bare ``except Exception`` that turned
        it into an instrument which silently never ran. That is the exact
        failure this task exists to prevent, so the import may fail loudly and
        only the CALL is guarded.
        """
        from sglang.srt.mem_ledger import flight_recorder

        try:
            flight_recorder.mark_serving(rank=self.ps.tp_rank)
        except Exception as e:  # noqa: BLE001 - a probe never breaks serving
            if not getattr(self, "_flight_serving_warned", False):
                # ONCE, at WARNING: a probe must not spam a serving loop, and
                # it must not be silent either. Silence is what cost the hours.
                self._flight_serving_warned = True
                logger.warning(
                    "VRAM flight serving mark failed and will be retried "
                    "quietly from here on: %s",
                    e,
                )

    def _corridor_trace_tick(self) -> None:
        """Arm the continuous corridor sampler, then AUDIT it on a cadence.

        #605 R2 open item 1: ``mem_ledger.corridor_trace`` was tested, ready
        and callerless. The boot marks are snapshots and the corridor law is a
        continuous minimum, so the sampler is the only instrument in this tree
        that can answer the law's own question -- but a module nothing calls
        measures nothing.

        WHY THE TICK, AND WHY ONCE. The sampler is a background thread with
        its own cadence; it does not need a per-iteration call and must not
        get one, because arming per iteration would be one NVML thread per
        scheduler round. The tick is used only as the first moment at which
        serving is definitely up, which is what makes this the natural home
        rather than a boot hook that fires before the cards are loaded.

        OFF BY DEFAULT, in two independent ways: ``corridor_trace.start()``
        returns None unless :data:`corridor_trace.TRACE_ENV` is set, and the
        attempt is made exactly once either way. A serving process that does
        not opt in starts no thread, touches no NVML on a timer, and is
        byte-identical.

        Never raises, and never retries: an instrument that cannot arm must
        not take serving down with it, and a failure that re-fires every
        iteration is the 2661-line boot log the census already learned from.
        """
        if not self._corridor_trace_armed:
            self._corridor_trace_armed = True
            try:
                from sglang.srt.mem_ledger import corridor_trace

                self._corridor_trace = corridor_trace.start()
            except Exception as exc:  # noqa: BLE001 - instrument must not raise
                logger.warning("corridor trace could not be armed: %s", exc)
            return

        # #656: AND THEN READ IT. Arming alone made the process able to
        # measure its own corridor and still unable to SAY anything about
        # it. The acceptance run sampled 63 minutes at 100 ms, went 138 MiB
        # under the law on GPU0 in five episodes, and no line in the serving
        # log mentions it -- the breach was found afterwards, by an external
        # sampler, in a CSV. A law the runtime cannot see is a law it cannot
        # be held to.
        trace = getattr(self, "_corridor_trace", None)
        if trace is None:
            return
        # getattr WITH DEFAULTS, not attribute access. The class carries these
        # as class attributes, but this method is called on whatever object
        # holds it, and `test_it_is_armed_ONCE_however_many_ticks_run` drives
        # it against a minimal stub -- correctly, because an instrument that
        # needs its host to have grown particular attributes is an instrument
        # that raises on the scheduler's hot path the first time someone
        # refactors around it. Same reason the sampler itself never raises.
        now = time.monotonic()
        if now < getattr(self, "_corridor_trace_next_check", 0.0):
            return
        self._corridor_trace_next_check = now + getattr(
            self, "_CORRIDOR_BREACH_CHECK_S", 10.0
        )
        try:
            summary = trace.summary()
            floor_raw = summary.get("free_min_mib")
            if not summary.get("breach") or floor_raw is None:
                return
            floor = int(floor_raw)
        except Exception:  # noqa: BLE001 - instrument must not raise
            return
        # The ring holds the whole window, so once a breach is in it the
        # verdict stays true forever. Report only when it gets WORSE, which
        # makes each line a new deepest instant rather than a repeat.
        prior = getattr(self, "_corridor_trace_reported_floor", None)
        if prior is not None and floor >= prior:
            return
        self._corridor_trace_reported_floor = floor
        logger.error(
            "CORRIDOR LAW BREACHED on this rank's card: the continuous "
            "minimum over the last %s samples (%.0f s at %s ms) is %d MiB, "
            "%d MiB BELOW the %d MiB law. This is the time-series minimum, "
            "not a snapshot -- the binding instant is a transient, typically "
            "inside a flip cutover, and it is already over by the time this "
            "line is written. It is reported because the alternative is what "
            "the #656 acceptance did: breach for 63 minutes and find out "
            "from a CSV afterwards. If this fires on a flip boot, the seam's "
            "measured DRAW exceeds the gate's seam-entry reserve and the "
            "arming floor is the number to re-derive (corridor_guard."
            "arming_floor_mib).",
            summary.get("n"),
            float(summary.get("span_s") or 0.0),
            summary.get("period_ms"),
            floor,
            int(summary.get("corridor_mib", 0)) - floor,
            summary.get("corridor_mib"),
        )
        # AND WRITE IT DOWN, so the next boot of this configuration sizes
        # itself against a measured shortfall instead of an assumed one.
        # Guarded and never raises: an instrument must not take serving down,
        # and a rank with no seam record (a cold boot) simply has nothing to
        # append to.
        try:
            from sglang.srt.managers.phase_flip_seam_reserve import (
                record_corridor_shortfall,
            )

            # THE FLIP RUNTIME'S RANK, not self.ps.tp_rank. The seam record
            # is keyed on the rank write_seam_reserve used (runtime._rank),
            # and under --tp-size 1 --pp-size 3 the TP rank is 0 in ALL
            # THREE processes -- register C605-3, where exactly that
            # substitution filed three cards under one rank. No runtime, no
            # record to append to, so nothing is written.
            runtime = getattr(self, "phase_flip_runtime", None)
            rank = getattr(runtime, "_rank", None)
            if rank is not None:
                depth_mib = int(summary.get("corridor_mib", 0)) - floor
                record_corridor_shortfall(self.server_args, int(rank), depth_mib << 20)
        except Exception:  # noqa: BLE001 - instrument must not raise
            pass

    def _census_tick(self) -> None:
        """Advance the census round and, on cadence, diff counts across ranks.

        #583. The collective COUNT is the quantity every desync in this
        family disagrees about; comparing it is what turns "rank 2 was
        somewhere else" into "rank 2 skipped its Nth tp.all_reduce". The
        detector fires while the ranks are still healthy -- typically within
        one tick, i.e. ~30 s before the spin deadline would abort.

        Instrument discipline: it never raises, and a failure to compare is
        logged and dropped rather than escalated (see
        ``CollectiveCensus.compare_across_ranks``).
        """
        if not _CENSUS_ON:
            return
        try:
            c = _CENSUS
            c.next_round()
            # `self.ps.tp_rank`, NOT `self.tp_rank`: the Scheduler keeps its
            # parallel identity on the ParallelState wrapper. The first cut
            # of this used `self.tp_rank`, which does not exist -- every
            # tick raised AttributeError, the census counted nothing, and a
            # desk test that stubbed the attribute never noticed. Pinned by
            # test_census_attribute_surface_583.py against the REAL class.
            rank = self.ps.tp_rank
            if c.due(_CENSUS_HEARTBEAT):
                c.heartbeat(rank)
            if c.due(_CENSUS_INTERVAL):
                grp = getattr(self, "tp_cpu_group", None)
                if grp is not None:
                    c.compare_across_ranks(
                        grp, torch.distributed.get_world_size(grp), rank
                    )
                    # #603b: the CAPTURE census, once. Graph capture is over
                    # by the time the scheduler loop runs, so the recorded
                    # sequences are final and one comparison settles them for
                    # the whole boot -- a per-tick repeat would re-ship the
                    # same answer forever. It rides this point rather than the
                    # end of capture because this is a proven rank-uniform
                    # cadence on the gloo group: every rank reaches it in the
                    # same round or none do.
                    self._capture_census_once(grp, rank)
            # Announced only AFTER a tick has completed without raising, so
            # a broken census can never look armed. The arming line is the
            # only evidence a reader has that the instrument is live; if it
            # printed before the work, it would certify the very failure it
            # exists to expose.
            c.announce_armed_once(rank, _CENSUS_INTERVAL, _CENSUS_HEARTBEAT)
        except Exception as exc:  # noqa: BLE001 - instrument must not raise
            # ONCE, not per tick: this fired 2661 times in one boot and
            # still told the reader nothing the first line had not.
            _CENSUS.warn_skipped_once(exc)

    def _capture_census_once(self, group, rank: int) -> None:
        """Diff the CAPTURED collective sequences across ranks. Exactly once.

        #603b. The #583 census above compares host-side collective COUNTS,
        and a captured collective is a host-side call exactly once in a boot
        -- at capture. Everything after that is replay, which makes no host
        call at all, so the count census is structurally blind to what a
        replayed graph does. That is where this crash family lives: the ranks
        are wedged inside a replayed decode graph, waiting on peer flags.

        A graph's behaviour is fixed when it is recorded. So if the ranks
        recorded different sequences, the mismatch is already present and
        readable at boot, minutes before the first wedge -- no reproduction
        needed. This asks that question once and logs the answer either way.

        Instrument discipline, as everywhere in this family: warn-never-raise.
        """
        if not capture_census_enabled():
            return
        cc = capture_census()
        if cc.compared:
            return
        try:
            path = cc.dump_to_file(rank)
            cc.compare_across_ranks(
                group, torch.distributed.get_world_size(group), rank
            )
            if path:
                logger.info("barlink capture census: per-rank record at %s", path)
        except Exception as exc:  # noqa: BLE001 - instrument must not raise
            # Mark it done regardless: a comparison that cannot run will not
            # start working on the next tick, and retrying it every interval
            # would turn one warning into a per-tick stream.
            cc.compared = True
            logger.warning(
                "barlink capture census: one-shot comparison unavailable (%s: %s)",
                type(exc).__name__,
                exc,
            )

    def uniform_min_avail(self) -> int:
        """This iteration's rank-uniform available_size.

        Falls back to the live local value ONLY on a single rank, where
        there is nothing to diverge from. On a group a missing reduce is
        refused loudly instead -- a decision must never read a value that
        silently means something else, and "this rank's own pool" is exactly
        that when the caller asked for the group's.
        """
        v = getattr(self, "_uniform_min_avail", None)
        if v is not None:
            return int(v)
        # #583 follow-up: refuse rather than silently reinstate the defect.
        #
        # The old fallback returned this rank's local available_size()
        # whenever the reduce had not run. On a single rank that is exactly
        # right. On a GROUP it is the rank-local predicate #603 removed,
        # restored silently by any path that reaches a decode-mem decision
        # before `_update_uniform_pool_budget` -- and a getattr default is
        # precisely the shape that hides such a path (#606): the guard reads
        # as present in the source while being absent at runtime.
        if self.ps.tp_size > 1:
            raise RuntimeError(
                "uniform_min_avail() was reached before "
                "_update_uniform_pool_budget() ran on a multi-rank boot. "
                "Returning this rank's local available_size() here would "
                "restore the rank-local decode-mem predicate #603 removed "
                "and split the ranks across branches carrying different "
                "collectives. The reduce is unconditional and pre-branch in "
                "get_next_batch_to_run; a caller reaching this line means a "
                "decode-mem decision escaped that ordering."
            )
        return int(self.token_to_kv_pool_allocator.available_size())

    def _phase_flip_on_round(self, require_armed_and_parked: bool = False):
        """One phase-flip runtime round (#631): lazy-build, bounded
        consensus, loop exit on commit.

        Two call sites, one per loop family: get_next_batch_to_run for
        event_loop_normal (lockstep TP rounds, periodic consensus), and
        the END of the event_loop_pp microbatch iteration with
        require_armed_and_parked=True -- under PP the ranks' local round
        counters diverge absolutely, so the reduction is entered only from
        an armed AND locally-parked state, where this rank owes no
        pipeline send (measured wedges 2026-08-08, boots 9+10; see
        PhaseFlipRuntime.on_round)."""
        if self.phase_flip_runtime is None:
            from sglang.srt.managers.phase_flip_runtime import (
                build_phase_flip_runtime,
            )

            self.phase_flip_runtime = build_phase_flip_runtime(self)
            # #656: the first round is the earliest point at which both
            # layouts, the arena carrier and the drafter all exist, so it is
            # the earliest point at which the seam's at-rest cost is a
            # measurement rather than a prediction. Measure it once, leave
            # the record for the next boot's sizer, and say out loud whether
            # THIS boot can fund its own flip.
            from sglang.srt.managers.phase_flip_seam_reserve import (
                measure_and_record,
            )

            measure_and_record(self, self.phase_flip_runtime)
        # #631: the per-rank output_ids clock, once per pass. Both loop
        # families reach this line exactly once per pass, which is what
        # makes the three ranks' lines comparable; the site name records
        # WHERE in the pass the sample was taken, because the PP hook sits
        # at the END of the iteration (after that pass's result was
        # processed) and the TP one at the top (before this pass ran).
        from sglang.srt.managers.phase_flip_output_trace import trace_tick

        trace_tick(self, "pp_end" if require_armed_and_parked else "tp_top")
        # SPEC ITEM 16: consult the rebalance lender on the one clock that
        # ticks in every phase, including the seam and the rounds no gate
        # prices. Rank-local, no collective (that is a hard requirement at
        # this cadence -- see PhaseFlipRuntime.on_round), rate-limited to a
        # monotonic clock read on the common path, and it never raises.
        from sglang.srt.managers.corridor_rebalance import lend_on_round

        lend_on_round(self)
        # #657 item 16: re-apply the standing allocation steer. The DECISION
        # was taken and reduced at the seam; this only keeps the free list's
        # order matching it, because frees return pages to the head of the
        # list and wash the partition out. Rate-limited, rank-local, and a
        # pure reordering -- it places nothing new and frees nothing.
        from sglang.srt.managers.corridor_steering import steer_on_round

        steer_on_round(self)
        flip_stats = self.phase_flip_runtime.on_round(
            require_armed_and_parked=require_armed_and_parked
        )
        _drain_seam_abandons_into_policy(self)
        if flip_stats is not None:
            from sglang.srt.managers.phase_flip_runtime import (
                PhaseFlipLoopExit,
            )

            # #656: bytes moved. This -- not the arm -- is what retires the
            # outstanding attempt and clears any refusal backoff.
            state = getattr(self, "phase_policy_state", None)
            if state is not None:
                try:
                    from sglang.srt.managers.phase_policy import note_flip_completed

                    note_flip_completed(
                        self.phase_policy_cfg,
                        state,
                        flip_stats["direction"],
                        time.perf_counter(),
                    )
                except Exception as e:  # pragma: no cover - bookkeeping only
                    logger.warning("PHASE-POLICY completion not recorded: %s", e)
            raise PhaseFlipLoopExit(flip_stats["direction"])

    def get_next_batch_to_run(
        self, running_batch: ScheduleBatch, last_batch: Optional[ScheduleBatch]
    ) -> NextBatchPlan:
        self.process_pending_chunked_abort()

        # #581: release lock_ref on completed write-through nodes ONCE PER
        # SCHEDULER ITERATION, before any batch is built. This used to sit in
        # `update_running_batch`, i.e. only on iterations that had a running
        # decode batch; a prefill-dominated window therefore took hicache
        # write-through pins (one per inserted checkpoint) with no release
        # path running, and the mamba state pool ratcheted to full with
        # `evict_mamba` finding nothing evictable. Here it is unconditional
        # and rank-symmetric -- every rank reaches this line exactly once per
        # iteration, which is required because `writing_check` min-reduces the
        # ack count across the TP group.
        if self.enable_hierarchical_cache:
            self.tree_cache.flush_write_through_acks()

        if self.enable_fpm:
            self._fpm_batch_t0 = time.monotonic()
        self._abort_on_waiting_timeout()
        self._abort_on_running_timeout(running_batch)
        if self.dllm_config is not None:
            self.dllm_manager.filter_finished_reqs()

        # Merge the prefill batch into the running batch
        chunked_req_to_exclude = set()

        if self.dllm_config is not None and self.dllm_manager.any_staging_reqs():
            chunked_req_to_exclude.update(self.dllm_manager.staging_queue)
            for req in self.dllm_manager.staging_queue:
                if self.dllm_config.first_done_first_out_mode:
                    if not req.dllm_incomplete_ids:
                        self.stash_chunked_request(req)
                    self.req_to_token_pool.free(req)
                else:
                    self.stash_chunked_request(req)

        if self.chunked_req is not None:
            # Move the chunked request out of the batch so that we can merge
            # only finished requests to running_batch.
            chunked_req_to_exclude.add(self.chunked_req)

            # Stash (cache) the previous chunk only when it produced new KV
            # beyond what is already cached. A parked chunk (add_chunked_req
            # hybrid-SWA early-return) leaves extend_range.end ==
            # len(prefix_indices), so there is nothing new to cache and
            # stashing would be a no-op.
            if self.chunked_req.extend_range.end > len(self.chunked_req.prefix_indices):
                self.stash_chunked_request(self.chunked_req)

        # HiSparse has its own prefill-to-decode transition; skip last_batch merge.
        if self.enable_hisparse:
            ready_reqs = self.hisparse_coordinator.collect_ready_reqs()
            if len(ready_reqs) > 0:
                new_batch = self._build_hisparse_decode_batch(ready_reqs)
                if running_batch.is_empty():
                    running_batch = new_batch
                else:
                    running_batch.merge_batch(new_batch)
                running_batch.hisparse_coordinator = self.hisparse_coordinator
            # Reset batch_is_full so the scheduler can schedule more prefills.
            running_batch.batch_is_full = False

        if (
            not self.enable_hisparse
            and last_batch
            and last_batch.forward_mode.is_extend()
        ):
            if last_batch.chunked_req is not None:
                # In the context pipeline parallelism, after the last chunk, the current microbatch still track outdated chunked_req.
                # We need to discard it.
                chunked_req_to_exclude.add(last_batch.chunked_req)

            if self.dllm_config is not None and last_batch.reqs:
                chunked_req_to_exclude.update(last_batch.reqs)

            # Filter batch
            last_bs = last_batch.batch_size()
            last_batch.filter_batch(chunked_req_to_exclude=list(chunked_req_to_exclude))
            if last_batch.batch_size() < last_bs:
                running_batch.batch_is_full = False

            # Merge the new batch into the running batch.
            if not last_batch.is_empty():
                if running_batch.is_empty():
                    # NOTE this ALIASES the two names onto one object. Under
                    # PP the scheduler then stores that object into BOTH
                    # running_mbs[mb_id] and (via mbs[mb_id]) last_mbs[mb_id],
                    # so the next visit to the same slot rebinds
                    # running_batch and last_batch to the SAME batch -- which
                    # is what the identity guard below exists to catch.
                    running_batch = last_batch
                elif last_batch is running_batch:
                    # SELF-MERGE, and it is fatal rather than wasteful: every
                    # per-request field would be torch.cat'ed with itself, so
                    # the batch DOUBLES on each visit to the slot. Measured
                    # 2026-08-09 23:42:45Z under --phase-flip-purity strict:
                    # the resident count walked 2^23 -> 2^24 -> 2^25 in three
                    # seconds and all three ranks died in
                    # sampling_batch_info.merge_batch's torch.cat, asking for
                    # 256 MiB with 138 MiB free.
                    #
                    # Skipping is not a mitigation, it is the correct answer:
                    # when the two names are one object its requests are
                    # ALREADY in running_batch, so merging could only
                    # double-count them. Strict purity is what made this
                    # reachable -- it confines prefill to the PP layout, and
                    # the merge branch above only runs for an extend
                    # last_batch, so before purity the aliased slot was
                    # rarely revisited while still carrying an extend batch.
                    #
                    # Logged, not silent: scheduler.py's flip-policy guard
                    # already DETECTED this state ("the resident set is
                    # corrupted") and correctly refused to arm a flip, but a
                    # detector that only declines to act cannot stop a
                    # doubling -- the instance still died. An observable line
                    # here is what makes the condition attributable.
                    logger.warning(
                        "SELF-MERGE REFUSED: last_batch is running_batch "
                        "(bs=%d). Its requests are already resident; merging "
                        "would double the batch. See #631/#656.",
                        running_batch.batch_size(),
                    )
                else:
                    # Merge running_batch with prefill batch
                    running_batch.merge_batch(last_batch)

        # For prefill-only batch, filter out finished requests since they
        # won't go through the decode step. This keeps running_batch accurate
        # for load reporting (num_running_reqs via /v1/loads).
        # Runs outside the last_batch block so stale requests are cleaned
        # even when no new batches arrive (e.g. traffic stops).
        if running_batch.is_prefill_only:
            running_batch.filter_batch()
            if running_batch.is_empty():
                running_batch.batch_is_full = False

        # kv-session-offload (S1): cleanup of a finished spilled session +
        # FIFO restore with hysteresis (merges the restored session back
        # into running_batch BEFORE batch selection / prepare_for_decode).
        if self.kv_session_offload is not None:
            # #656 LAYOUT PIN, re-applied every round and BEFORE pre_schedule
            # picks a tick. A host image belongs to the phase it was captured
            # in; while the other phase is live, that session must not run.
            # `suppress_tick` is a one-shot the picker clears, so this is a
            # per-round re-assert and not a latch -- a latch would release
            # itself on the first tick it suppressed, which is the tick that
            # matters. No-op when the flip is off (no phase to pin against).
            if getattr(self, "phase_flip_active_stack", None) is not None:
                from sglang.srt.managers.kvso_flip_contract import (
                    pin_spills_to_phase,
                )

                pin_spills_to_phase(
                    self.kv_session_offload, self.phase_flip_active_stack
                )
            running_batch = self.kv_session_offload.pre_schedule(
                running_batch, last_batch
            )
            # RANK-UNIFORM admission under uneven DCP: recompute the min-reduced
            # available / (available+evictable) budget for this iteration with a
            # SINGLE collective, HERE -- before the prefill/decode branch below,
            # reached unconditionally by every rank every iteration. This is the
            # only cross-rank reduce on the offload scheduling path; both the
            # prefill admission (PrefillAdder deficit) and the decode-spill
            # trigger read the stored result, so a divergent local pool can never
            # split the ranks across branches with mismatched collective counts.
            # See update_dcp_admission_state for the full rationale.
            self.kv_session_offload.update_dcp_admission_state()

        # #603: the SAME reduce for the path without the offload manager.
        #
        # The block above has always been correct, and it was correct for a
        # reason that has nothing to do with session offload: under uneven
        # DCP/TP the per-rank pools differ, so any decision read off a LOCAL
        # available_size can flip on the binding rank while the others still
        # fit -- and the two groups then take branches carrying different
        # collectives. The guard `kv_session_offload is not None` was
        # orthogonal to that; it merely happened to be the feature whose
        # development surfaced the divergence.
        #
        # With `--enable-kv-session-offload` OFF (the production default) the
        # protection was therefore absent and `check_decode_mem` decided
        # rank-locally. That is the 2026-08-05 19:41 abort: ranks 0/1 entered
        # the decode collective inside run_batch while rank 2, having decided
        # to retract, went round to recv_requests -- the BAR1 spin kernels on
        # 0/1 then waited ~30 s for a peer that was never coming and took
        # their abort path. Exactly the shape update_dcp_admission_state's
        # docstring predicts ("observed in recv_requests one iteration
        # later").
        #
        # Unconditional and pre-branch, like the call above: every rank
        # reaches this line exactly once per iteration, so the collective
        # count stays rank-uniform no matter which branch is taken later.
        self._update_uniform_pool_budget()

        # #583 collective census. Same placement argument as the reduce above:
        # every rank reaches this line exactly once per iteration, so the
        # round counter it advances is REPLICATED and the cadence gate inside
        # `due` opens on the same round for all of them. The comparison runs
        # on tp_cpu_group (gloo), never the device/BAR1 path, so a wedged
        # device group cannot silence the instrument meant to explain it.
        # Warn-never-raise lives inside compare_across_ranks.
        self._census_tick()

        # #605: the corridor sampler's production call site. Arms once and
        # returns immediately on every later iteration; no-op unless
        # SGLANG_CORRIDOR_TRACE_MS is set.
        self._corridor_trace_tick()

        # #684: the DURABLE serving series, beside the sampler above and for a
        # different job -- the ring dies with the process, this survives it.
        # Paced in the recorder; no-op unless SGLANG_VRAM_FLIGHT_DIR is set.
        self._flight_serving_tick()

        # #297 phase-boundary KV resharding: same lazy-build and cadence
        # discipline as the #287 block below. Built FIRST so the pressure
        # runtime's dcp_ratio actuator can bind to it in the same iteration.
        # on_round enters its bounded consensus reduction only every
        # consensus_interval-th round, gated by the replicated round counter;
        # the byte move itself runs only from a group-agreed armed+idle
        # state. Flag unset = attribute stays None, no collective,
        # byte-identical to today.
        if self.server_args.kv_reshard_vectors is not None:
            if self.kv_reshard_runtime is None:
                from sglang.srt.managers.kv_reshard import build_kv_reshard_runtime

                self.kv_reshard_runtime = build_kv_reshard_runtime(self)
            self.kv_reshard_runtime.on_round()

        # #631 phase flip: same lazy-build and cadence discipline as the
        # #297 block above (bounded consensus every consensus_interval-th
        # round; the move itself only from a group-agreed armed+quiescent
        # state). When on_round returns commit stats, THIS ROUND flipped
        # the topology: exit the current event loop to the re-dispatching
        # wrapper (dispatch picks its loop once from pp_size) -- raised
        # here, after the runtime's epoch/phase bookkeeping completed, at
        # a quiescent boundary by construction. Flag unset = attribute
        # stays None, no collective, byte-identical to today.
        # PP-LOOP ORDERING (first real-metal flip boot, 2026-08-08): under
        # event_loop_pp this hook is DEFERRED to the end of the microbatch
        # iteration (scheduler_pp_mixin sets _defer_flip_round_to_pp_loop).
        # get_next_batch_to_run sits at the TOP of the pp iteration, before
        # this rank's sends are issued; entering the bounded world-reduction
        # there closes a cycle with the pipeline's p2p chain (measured
        # wedge: PP0+PP1 in the consensus, PP2 in recv-from-PP1, barlink
        # liveness killed the tree after 120 s). The invariant is the
        # rank-local-state-feeds-collective family rule in PP form: a
        # blocking group reduction may only be entered once every send a
        # peer needs to reach ITS reduction of the same round is flushed --
        # i.e. as the LAST blocking op of the iteration.
        if self.server_args.enable_phase_flip and not getattr(
            self, "_defer_flip_round_to_pp_loop", False
        ):
            self._phase_flip_on_round()

        # #330 VRAM dial / KV capacity runtime: same lazy-build and cadence
        # discipline. Runs AFTER the reshard block so a #297 cutover in this
        # round is already visible to the capacity math (the C re-raise arms
        # at the next boundary from the new vector). Flag unset = attribute
        # stays None, no collective, byte-identical to today.
        if self.server_args.enable_vram_dial:
            if self.kv_capacity_runtime is None:
                from sglang.srt.managers.vram_dial import build_kv_capacity_runtime

                self.kv_capacity_runtime = build_kv_capacity_runtime(self)
            self.kv_capacity_runtime.on_round()

        # #287 KV pressure ladder: one occupancy sample per iteration, ladder
        # transitions only at the rank-uniform consensus boundary inside
        # on_round. Constructed lazily on the first iteration (every
        # dependency -- limiter, offload manager, tp_cpu_group -- exists by
        # then, still before any request is admitted). Flag unset = the
        # attribute stays None and this block is two predictable branches:
        # no sample, no collective, byte-identical to today.
        if self.server_args.kv_pressure_ladder is not None:
            if self.kv_pressure_runtime is None:
                from sglang.srt.managers.kv_pressure_runtime import (
                    build_kv_pressure_runtime,
                )

                self.kv_pressure_runtime = build_kv_pressure_runtime(self)
            if self.kv_pressure_runtime is not None:
                # Every input is REPLICATED (held tokens of live requests,
                # the group-agreed capacity, batch size, the derived phase)
                # -- the same uniformity argument as the admission limiter's
                # sample below at update_running_batch.
                decode_active = (
                    not running_batch.is_empty() and not running_batch.is_prefill_only
                )
                self.kv_pressure_runtime.on_round(
                    held_tokens=sum(req.seqlen for req in running_batch.reqs),
                    # held_tokens are GLOBAL sequence lengths, so the capacity
                    # they are weighed against has to be the global span too
                    # (#346). Identity off the token-sharded lane.
                    capacity_tokens=self._global_kv_capacity_tokens(),
                    running_bs=running_batch.batch_size(),
                    phase="decode" if decode_active else "prefill",
                )

        # #364 GDN resident-slot ladder. Same between-tick boundary and the
        # same lazy-build discipline as the #287 block above, for the same
        # reason: the previous batch is retired here and the next one is not
        # selected yet, so no captured graph can replay while the executor
        # rewrites a state slot (#52/#53). The executor REFUSES to run outside
        # the window it is given here, so this placement is enforced, not just
        # documented. Flag unset = the attribute stays None and this is one
        # predictable branch, byte-identical to today.
        if self.server_args.gdn_resident_state_slots is not None:
            if self.gdn_slot_executor is None:
                from sglang.srt.managers.gdn_slot_runtime import (
                    build_gdn_slot_executor,
                )

                self.gdn_slot_executor = build_gdn_slot_executor(self)
            if self.gdn_slot_executor is not None:
                self.gdn_slot_executor.on_round(running_batch, self.waiting_queue)

        # #363 regime observer, OBSERVE-ONLY. Same between-tick boundary and
        # the same lazy-build discipline as the two blocks above, and for one
        # more reason of its own: this is where the previous batch's forward
        # is already retired, so the per-rank device timing it reads is a
        # completed measurement rather than a half-recorded one.
        #
        # It classifies and logs; it calls no actuator (DESIGN_363 section 6
        # ships v1 observe-only). Every argument below is REPLICATED across
        # the TP group -- the same uniformity argument the #287 block above
        # makes -- except rank_forward_ms, which is this rank's own number and
        # is released only through the observer's consensus reduction.
        #
        # Unset env = the attribute stays None and this is one predictable
        # branch per round, byte-identical to today.
        if self._regime_observer_mode is None:
            from sglang.srt.managers.regime_runtime import resolve_mode

            self._regime_observer_mode = resolve_mode(self.server_args)
        if self._regime_observer_mode != "off":
            if self.regime_observer is None:
                from sglang.srt.managers.regime_runtime import (
                    build_regime_observer,
                    build_regime_stage_table,
                )

                # The table first: the observer reads it, and both are built
                # once on the first iteration where the pools and the runtimes
                # this server actually wired already exist.
                self.regime_stage_table = build_regime_stage_table(self)
                self.regime_observer = build_regime_observer(self)
            if self.regime_observer is not None:
                from sglang.srt.managers.regime_runtime import (
                    phase_of_last_batch,
                    rank_forward_ms_from,
                    rank_split_ms_from,
                )

                # THREE-way (prefill / decode / idle), read off LAST_BATCH --
                # the forward that just retired into this boundary -- and not
                # off running_batch.
                #
                # running_batch cannot answer it. By the time control reaches
                # here the top of this method has already merged a finished
                # extend batch INTO running_batch, so during a prefill burst
                # running_batch is the decode batch; and the next prefill
                # batch is not built until get_new_batch_prefill below. The
                # flag the hook used to read, is_prefill_only, is a REQUEST
                # KIND (max_new_tokens == 0, forced False under spec), never
                # an execution phase -- so prefill_share read 0.000 on all
                # 34 954 boundaries of the 2026-08-01 window and on both
                # gate-3 arms. #388.
                #
                # last_batch is the honest source at this point in the loop
                # and it is also the CONSISTENT one: rank_forward_ms below is
                # that same retired forward's device time, so phase and
                # timing on one record describe one event. The attribution
                # rules, including why one boundary of lag does not move a
                # share, are in phase_of_last_batch.
                # One read, two terms: calling the accessor twice would ask
                # the same retired forward for its split twice.
                rank_split = rank_split_ms_from(self)
                self.regime_observer.on_round(
                    phase=phase_of_last_batch(last_batch),
                    held_tokens=sum(req.seqlen for req in running_batch.reqs),
                    capacity_tokens=self._global_kv_capacity_tokens(),
                    running_bs=running_batch.batch_size(),
                    queued_reqs=len(self.waiting_queue),
                    queued_prompt_tokens=sum(
                        len(req.origin_input_ids) for req in self.waiting_queue
                    ),
                    max_queued_prompt_tokens=max(
                        (len(req.origin_input_ids) for req in self.waiting_queue),
                        default=0,
                    ),
                    rank_forward_ms=rank_forward_ms_from(self),
                    # #363 intra-phase axis. The SAME retired forward, split
                    # into its two terms: the stage axes move the wait term
                    # and not the compute term, so a total would hide the only
                    # part of the round they can be credited against.
                    #
                    # Both are None for a forward whose collectives ran inside
                    # a captured graph, and the observer's packed sentinel
                    # carries that absence through the group reduction rather
                    # than letting a blind rank read as fast. Read that
                    # narrowly: it is a statement about the COLLECTIVES, not
                    # about every forward on a rig that captures graphs. This
                    # comment previously said "None on a graph-covered
                    # forward", and the R14 window read it as "every forward
                    # on this rig", which sent a whole window looking for a
                    # device-timing fix. The rig it was written on captures
                    # DECODE graphs and runs prefill eager, so the split was
                    # being measured on ~2 574 forwards in the very boot the
                    # window concluded had none. What was actually missing
                    # was the CLOCK (build_regime_observer, defect 8a).
                    # Accumulated unconditionally and read only by the
                    # flag-gated clock, so an off boot pays two float adds.
                    rank_compute_ms=rank_split[0],
                    rank_wait_ms=rank_split[1],
                    # The sample's identity, not a second measurement (#363
                    # defect 8b). The accessor carries its last reading
                    # forward, so without this the observer counts one
                    # retired forward once per boundary and calls the result
                    # a mean.
                    rank_split_seq=rank_split[2],
                )

        # #631 PARKING: an ARMED flip withholds all new work -- no prefill
        # batch is built and the decode batch is not launched -- so the
        # in-flight state drains and ready_fn's quiescence becomes
        # reachable BETWEEN a request's prefill and its decode (the
        # design's core promise; without this, an armed flip waited for
        # every stream to FINISH -- measured on the first rung-c attempt,
        # 2026-08-08). Rank-uniform: pending arrives via the broadcast RPC
        # in the same round on every rank, and chunked_req is replicated
        # batch state. The park is BOUNDED by the runtime's park deadline
        # (group-agreed abort of the FLIP, never of the requests).
        #
        # THE CHUNK EXEMPTION, NARROWED 2026-08-09 -- defect O's other half.
        # This condition used to be `self.chunked_req is None`, on the
        # reasoning that "a half-written chunk is exempt: its continuation
        # must complete or ready_fn could never go true". Defect O retired
        # that premise IN THE SAME SESSION and this site was never
        # revisited: ready_fn no longer blocks on a chunked request at all,
        # only on one that has no pool row yet (mid-admission), because
        # BETWEEN CHUNKS IS A SETTLED BOUNDARY -- committed KV, a fully
        # accounted extend_range, exactly the state the carry moves.
        #
        # Left wide, the exemption defeated the very flip it was meant to
        # protect. MEASURED, production, 2026-08-09 20:31:38-48Z: tp_to_pp
        # armed BECAUSE "pending prefill 12747 tok > N=7004", and then the
        # scheduler kept building the next 2048-token chunk every round for
        # ten seconds until all 12747 tokens had been prefilled in the TP
        # layout at ~1500 tok/s (PP does the same work at ~4200). The
        # cutover committed into PP with 459 tokens left, whereupon the
        # policy immediately armed pp_to_tp again ("prefill down to 459
        # tok"). Two cutovers paid to move work already done in the slow
        # layout -- and those back-to-back epochs 6/7/8 are the interleaving
        # that exposed corpse I and killed PP0 at 20:31:48.
        #
        # The exemption now covers exactly what ready_fn still blocks on
        # and nothing more. Once the request holds a pool row the park
        # applies, the boundary arrives within a round or two, and the
        # pending prefill lands in PP -- which is the entire point of
        # arming tp_to_pp.
        # The predicate lives next to the quiescence rule it must agree
        # with (chunk_blocks_quiescence), so the two cannot drift apart
        # again the way they did between defect O and 20:31:48.
        from sglang.srt.managers.phase_flip_runtime import (
            chunk_blocks_quiescence,
        )

        if (
            self.server_args.enable_phase_flip
            and self.phase_flip_runtime is not None
            and self.phase_flip_runtime.pending is not None
            and not chunk_blocks_quiescence(self.chunked_req)
        ):
            # A round withheld for a PENDING FLIP is not a round that could
            # not build a batch -- it is one that deliberately did not try. It
            # must not leave the previous round's verdict standing, or the
            # arming gate reads a stale "nothing can run" from before the flip
            # was armed.
            self._round_built_nothing = False
            return NextBatchPlan(batch_to_run=None, running_batch=running_batch)

        # #797: A ROUND WITHHELD FOR A VOIDED PP PASS, on the same argument
        # and in the same shape as the pending-flip branch above -- and it has
        # to be the WHOLE round, not just the prefill admission.
        #
        # `_pp_void_retracted_pass` (scheduler_pp_mixin.py) empties this pass's
        # `effective`, which stops every prefill rid being admitted; the
        # admission loop below reads a rid it does not name as "not this
        # pass". That alone is not enough. With nothing to prefill this method
        # falls through to its decode branch and returns the RUNNING batch --
        # so the rank would run a decode batch for a slot whose upstream is
        # running a prefill batch, which is the same mispairing one forward
        # further on. The retraction voids the pass, so the pass must build
        # nothing at all.
        #
        # Uniform across the ranks that matter by construction, which is the
        # property every other branch here has to argue: the flag is not
        # re-derived per rank, it is set by the one rank that retracted and
        # carried to every rank after it on the admission decision
        # (`_PP_PASS_VOIDED_KEY`). Ranks BEFORE it never see it, and they are
        # exactly the ranks that already launched -- the void is what their
        # output ring absorbs (#791b), not something they can join.
        #
        # Nothing is lost and nothing leaks: no batch was built, so no KV,
        # mamba slot or req-pool row was allocated to drop, and the requests
        # stay in `self.waiting_queue` for the next pass.
        #
        # Scoped by `ps.pp_size > 1` exactly as `_pp_admission_incoming_
        # effective` is at this method's other consumer
        # (`_get_new_batch_prefill_raw`): the flag is written only by the PP
        # event loop, and a phase flip into the TP layout must not be able to
        # inherit a void decided in the PP window. `init_pp_loop_state` clears
        # it at every cutover for the same reason.
        if self.ps.pp_size > 1 and getattr(self, "_pp_admission_pass_voided", False):
            self._round_built_nothing = False
            return NextBatchPlan(batch_to_run=None, running_batch=running_batch)

        if self.dllm_config is not None:
            new_batch = self.get_new_batch_dllm(running_batch)
        elif phase_prefill_blocked_here(
            self,
            running_bs=(
                len(running_batch.reqs)
                if running_batch is not None and running_batch.reqs is not None
                else 0
            ),
        ):
            # #631 STRICT PHASE PURITY: not a single token is prefilled in
            # the TP layout. The prefill batch is not BUILT (nothing is
            # allocated and then dropped) -- the work stays queued and is
            # executed, batched, in the next PP window. TP prefill measures
            # 1681 tok/s against PP's 7245 on this rig, so this defers work
            # rather than losing it.
            #
            # Same rank-uniformity argument as the congruent lane below:
            # every input (static purity config, the replicated active
            # layout) is identical on every rank, so the group never splits
            # across branches with mismatched collectives.
            new_batch = None
        elif (
            self.congruent_prefill_lane is not None
            and not self.congruent_prefill_lane.allow_prefill(
                device_has_decode_work=not running_batch.is_empty()
                and not running_batch.is_prefill_only
            )
        ):
            # colocated-congruent PD lane (#107): DECODE priority. The stock
            # policy runs prefill first whenever one can be built; under the
            # lane, prefill is cadence-gated so the decode lane keeps its
            # rate — the property PD exists for, delivered in-process. The
            # prefill batch is not built at all this iteration (nothing is
            # allocated and then dropped). Rank-uniform: the gate reads only
            # replicated state (batch emptiness, the deterministic tick
            # counter), so every rank takes the same branch — the same
            # argument the spill tick makes; a rank-local input here would
            # split the ranks across mismatched collective counts.
            new_batch = None
        else:
            prefill_plan = self.get_new_batch_prefill(running_batch)
            new_batch = prefill_plan.batch_to_run
            running_batch = prefill_plan.running_batch
            if (
                self.congruent_prefill_lane is not None
                and new_batch is not None
                and new_batch.forward_mode.is_extend()
            ):
                # A prefill batch was actually selected: reset the cadence
                # and verify (once) that it computes with the decode ranks'
                # resident weights — the shared-bytes invariant of this
                # topology, checked instead of assumed.
                self.congruent_prefill_lane.note_prefill_ran(
                    self.tp_worker.model_runner.model
                )

        need_mlp_sync = self.require_mlp_sync
        if (
            need_mlp_sync
            and not self.spec_algorithm.is_none()
            and not self.server_args.speculative_skip_dp_mlp_sync
        ):
            # NOTE: This branch makes sure prefill and decode batches will not be mixed when spec and dp-attn is enabled.
            # Before merging the new batch into running batch:
            # 1. All new batches are none -> need_mlp_sync remains true (sync is needed for decode batch).
            # 2. All new batches are some (prefill / idle) -> we do not need prepare mlp sync one more time.
            new_batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(new_batch)
            need_mlp_sync = new_batch is None

        # DECOUPLE S4b: default the concurrent-spill stash to empty every
        # iteration (rank-uniform; set only in the decouple branch below).
        self._pending_spill_batch = None

        if new_batch is not None:
            # Run prefill first if possible
            ret = new_batch
        elif self.kv_session_offload is not None and self.kv_session_offload.decouple:
            # kv-session-offload DECOUPLE S4b: run the DEVICE decode batch AND a
            # due spill tick CONCURRENTLY this iteration (not instead-of).
            # Select the device decode batch exactly as the default branch
            # does, then STASH a due spill tick for concurrent dispatch on
            # spill_stream / comm B in the event loop. The spill tick no longer
            # steals the device iteration -> the device lane stops waiting on
            # the spill's PCIe H2D. Rank-uniform: both the device-batch
            # emptiness and the tick decision derive from replicated state, so
            # every rank either dispatches the same pair or neither. Only
            # reached when the flag is ON, so the default path is byte-identical.
            if not running_batch.is_empty() and not running_batch.is_prefill_only:
                running_batch = self.update_running_batch(running_batch)
                ret = running_batch if not running_batch.is_empty() else None
            else:
                ret = None
            # maybe_take_tick keeps its cadence gate, now read as PCIe
            # backpressure (how often the spill lane advances) rather than
            # device protection. Pass the post-update running_batch so
            # device_has_work reflects the actual device dispatch.
            self._pending_spill_batch = self.kv_session_offload.maybe_take_tick(
                running_batch
            )
        elif (
            self.kv_session_offload is not None
            and (
                spill_tick_batch := self.kv_session_offload.maybe_take_tick(
                    running_batch
                )
            )
            is not None
        ):
            # kv-session-offload (S1, flag OFF / serial): run the spilled
            # session's eager decode tick INSTEAD of the device decode batch
            # this iteration (cadence-gated; the device batch was not prepared,
            # so nothing is lost). Never mixed into the device batch -> the
            # device graph lockstep stays untouched.
            ret = spill_tick_batch
        else:
            # Run decode (skip for prefill-only batches)
            if not running_batch.is_empty() and not running_batch.is_prefill_only:
                # #677 PHASE 1: the ONE evaluation of the purity verdict per
                # round happens on the next line, and the admission gate reads
                # what it records. It cannot ask the predicate itself --
                # decode_blocked_here advances the decode starvation clock, so
                # a second call per round would double-tick it and relax
                # purity early.
                _decode_blocked = phase_decode_blocked_here(
                    self, running_batch.batch_size()
                )
                self._note_parked_carriers(running_batch, _decode_blocked)
                if _decode_blocked:
                    # #631 STRICT PHASE PURITY: no decode step executes in
                    # the PP layout. The requests stay RESIDENT and are
                    # carried across the next cutover by the resident-carry
                    # and draft-bootstrap machinery, then resume batched in
                    # the TP window with graphs and speculation live.
                    #
                    # Returning None here leaves the round empty, which is
                    # the intended signal: with prefill drained and decode
                    # forbidden the PP phase has no work, and the policy's
                    # ``pending <= N and running_bs > 0`` rule arms PP->TP
                    # on the very next decision. When prefill is NOT
                    # drained but cannot be admitted either, the policy's
                    # bounded PP window is the exit -- see the deadlock
                    # section of phase_purity.
                    ret = None
                else:
                    running_batch = self.update_running_batch(running_batch)
                    ret = running_batch if not running_batch.is_empty() else None
            else:
                ret = None

        # Handle DP attention and log stats
        ret = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(
            ret, need_sync=need_mlp_sync
        )

        # Handle ngram embedding
        ret = self._maybe_prepare_ngram_embedding(ret)

        if ret:
            set_schedule_time_batch(ret)
            if self.enable_fpm:
                ret.fpm_start_time = self._fpm_batch_t0

        self._note_round_build_outcome(ret, running_batch)
        return NextBatchPlan(batch_to_run=ret, running_batch=running_batch)

    def get_num_allocatable_reqs(self, running_bs):
        # #287: the floating admission limit joins the existing bounds as one
        # more min(). Without --max-running-requests-ceiling the limiter holds
        # the same max_running_requests that pp_max_micro_batch_size was
        # derived from (max_running_requests // pp_size <= it), so the min is
        # inert and this is the stock expression.
        limit = min(
            get_server_args().pp_max_micro_batch_size,
            self.admission_limiter.current,
        )
        # #677 PHASE 1: a carrier the phase FORBIDS to decode stops being
        # charged to the concurrency cap. `limit` bounds how much decode may
        # run at once; a request PP may not decode is running nothing, so
        # counting it there reserves concurrency nobody can spend. Measured
        # 2026-08-16 06:04: four such carriers held the whole cap of four and
        # 403779 tokens of prefill could not be admitted behind them.
        #
        # ZERO UNLESS PARKING IS ARMED AND THE PHASE FORBIDS DECODE, so the
        # default path below is the pre-change expression unchanged.
        parked = self._parked_carrier_discount(running_bs)
        res = limit - max(0, running_bs - parked)
        res = min(res, self.req_to_token_pool.available_size())
        # THE SECOND BOUND EXISTS BECAUSE THE FIRST ONE STOPPED BINDING.
        # available_size() above is the REQUEST-slot count -- HybridReqToToken
        # Pool does not override it -- so nothing in this expression has ever
        # seen the mamba/GDN state pool; alloc_req_slots consults it later and
        # refuses there. While the concurrency cap bound admission that late
        # refusal was unreachable. Discounting carriers removes that cap, so
        # the state pool becomes the real ceiling and is asserted HERE, early
        # and by name, instead of as a late refusal in the allocator.
        res = min(res, self.parked_decode_set.admission_headroom(running_bs, res))
        return res

    def dynamic_chunked_prefill_size(self) -> int:
        """This batch's chunked-prefill budget, dynamic when available.

        #656: THE FIRST CHUNK IS SIZED TOO, and it did not used to be. The
        gate here was ``self.chunked_req is not None``, so only an already
        in-flight partial prefill got a predicted size and every prefill's
        opening chunk took the static one.

        That skipped the feature exactly where it pays. A prompt short
        enough to finish in one chunk never becomes a ``chunked_req``, so
        it was never sized dynamically at all -- the whole class of
        requests the predictor could serve end to end was excluded. And
        for long prompts the first chunk is the one that sets the
        pipeline's opening bubble, so it is the worst one to take blind.

        Nothing in the predictor needed the in-flight request: it takes a
        ``history_len``, and for a prefill that has not started that value
        is 0 -- known, not assumed. So the only real change is which
        history is passed.

        Still refusable, and that matters: the predictor returns None
        until it has profiled (and ``enable_dynamic_chunking`` self-clears
        when profiling raises at init), so a None must fall back to the
        static size rather than be handed on as a chunk width.
        """
        if not self.enable_dynamic_chunking:
            return self.chunked_prefill_size
        history_len = (
            len(self.chunked_req.prefix_indices) if self.chunked_req is not None else 0
        )
        dynamic_size = self.predict_next_chunk_size(history_len)
        if dynamic_size is None:
            return self.chunked_prefill_size
        self._log_dynamic_chunk_engagement(dynamic_size, history_len)
        return dynamic_size

    def _log_dynamic_chunk_engagement(self, dynamic_size: int, history_len: int):
        """ENGAGEMENT PROOF for the dynamic-chunking arm, at INFO.

        #656: the only line that reported a predicted chunk size was
        DEBUG-level (``scheduler_pp_mixin.predict_next_chunk_size``), and
        nothing at INFO or above fired when the width actually deviated from
        ``--chunked-prefill-size``. So an A/B of the arm produced a throughput
        number with NO evidence that the mechanism ever engaged -- and a
        throughput delta with no engagement proof is not a measurement of the
        feature, it is a measurement of the run.

        EDGE-TRIGGERED, not per-iteration. This is called once per scheduling
        iteration (``scheduler.py`` in ``get_new_batch_prefill``'s budget
        line), which is thousands of times a minute; an unconditional INFO
        there would be a log flood and would itself perturb the thing being
        measured. It therefore fires only when the width CHANGES to a value
        not last reported, which is exactly the event the proof needs: the
        first deviation from the static size is the engagement, and every
        subsequent distinct width is the feature moving at runtime.
        """
        if dynamic_size == getattr(self, "_dyn_chunk_last_logged", None):
            return
        self._dyn_chunk_last_logged = int(dynamic_size)
        static = int(self.chunked_prefill_size)
        logger.info(
            "[PP Dynamic Chunk] ENGAGED [PP%s]: chunk width %d (static "
            "--chunked-prefill-size is %d, delta %+d, history_len=%d). This "
            "line is edge-triggered on a change of width and is the runtime "
            "engagement proof for the dynamic-chunking arm.",
            getattr(getattr(self, "ps", None), "pp_rank", "?"),
            int(dynamic_size),
            static,
            int(dynamic_size) - static,
            int(history_len),
        )

    def _local_corridor_width_ceiling(self) -> int:
        """#794: the widest prefill chunk THIS card can fund, for the reduce.

        Contributed unconditionally and computed from the CONFIGURED width, not
        from this pass's dynamic one: the reduce runs before the pass picks a
        width, and a payload whose meaning varied per rank would be worse than
        no payload at all. The consumer takes ``min(requested, reduced)``, so a
        dynamic width narrower than the configured one still wins.

        NEVER RAISES AND NEVER RETURNS SOMETHING SMALLER THAN IT MEANS. Every
        failure path -- no gate, no guard, no price, an exception -- returns the
        configured width, which makes this rank a non-binding vote rather than
        one that silently narrows the whole group.
        """
        configured = int(getattr(self, "chunked_prefill_size", 0) or 0)
        if configured <= 0:
            return 1 << 30
        try:
            gate = get_prefill_admission_gate(self)
            if gate is None:
                return configured
            granted = int(gate.granted_width(configured))
        except Exception as e:  # noqa: BLE001
            if not getattr(self, "_corridor_width_probe_failed", False):
                self._corridor_width_probe_failed = True
                logger.error("#794 corridor width ceiling failed to price: %s", e)
            return configured
        if granted <= 0 or granted > configured:
            return configured
        return granted

    def uniform_corridor_width(self):
        """The group's tightest corridor width this iteration, or None.

        None means the reduce did not run (world size 1, or the offload branch),
        in which case the local decision IS the group decision.
        """
        return getattr(self, "_uniform_corridor_width", None)

    def _corridor_granted_prefill_width(self, chunked_prefill_size: int) -> int:
        """#794: narrow this pass's chunk to what this card's corridor funds.

        WHO IS ENTITLED TO NARROW, and why the answer is not "every rank".

        The width must end up rank-uniform or the group splits, and this fork
        already has exactly one place per topology where a uniform width is
        decided. This function does not add a second one, and above all it
        does not add a collective to the admission path -- the #791 corpus
        exists because collectives here deadlocked this instance twice
        (instr22 5m19s, instr23 5m12s, identical signature).

        * pp_size > 1: PP0 DECIDES AND THE RING CARRIES IT. The decision is
          built on rank 0 only (`build_pp_admission_decision`, called at
          `_get_new_batch_prefill_raw` under `pp_rank == 0`) and every other
          rank must reproduce the forwarded `(rid, prefix_len, extend_len)`
          set exactly or raise `PPScheduleRefused` and void the pass. So a cut
          taken on PP0 is uniform by construction and costs nothing, while a
          cut taken DOWNSTREAM would produce a narrower local batch than the
          one it was handed -- a refusal, then a void, then the same request
          again: a livelock, not a safety measure. Downstream ranks therefore
          never narrow, and that is not a gap in their protection: the guard's
          relief ladder still runs on every rank, and the binding card on this
          rig is PP0 (the 5090, both OOM specimens).

        * pp_size == 1 and tp_cpu_group world > 1: NAMED GAP, deliberately.
          Under pure TP every rank builds its own batch, so a uniform width
          needs the MIN reduce -- and the right close is to widen
          `_update_uniform_pool_budget`'s existing packed reduce by one term,
          where the reduce already lives, rather than to open a second one
          here ("two reduces would be two chances for the counts to diverge").
          Until then the width is not narrowed on that topology and this says
          so once, at WARNING, rather than being silently inert.

        * otherwise (world size 1): the local decision IS the group decision.
        """
        requested = int(chunked_prefill_size or 0)
        if requested <= 0:
            return chunked_prefill_size
        pp_size = int(getattr(getattr(self, "ps", None), "pp_size", 1) or 1)
        if pp_size > 1:
            if int(getattr(self.ps, "pp_rank", 0) or 0) != 0:
                return chunked_prefill_size
        else:
            grp = getattr(self, "tp_cpu_group", None)
            try:
                world = 0 if grp is None else int(torch.distributed.get_world_size(grp))
            except Exception:  # noqa: BLE001
                world = 0
            if world > 1:
                # THE GROUP'S WIDTH, NOT THIS RANK'S. Every rank in this group
                # builds its own prefill batch, so the width must be one number
                # for all of them: the MIN reduced once this iteration in
                # `_update_uniform_pool_budget`, pre-branch, no collective here.
                reduced = self.uniform_corridor_width()
                if reduced is None:
                    # The reduce did not run this iteration (offload branch).
                    # Do not narrow on a rank-local reading in a group that
                    # would not narrow with us.
                    return chunked_prefill_size
                reduced = int(reduced)
                if reduced <= 0 or reduced >= requested:
                    return chunked_prefill_size
                if reduced != getattr(self, "_corridor_width_group_logged", None):
                    self._corridor_width_group_logged = reduced
                    logger.warning(
                        "#794 GROUP-NARROWED this prefill chunk from %d to %d "
                        "tokens: %d ranks share this admission decision and "
                        "the tightest card's corridor funds %d. Every rank "
                        "cuts to the same number, so no rank admits work its "
                        "peers declined.",
                        requested,
                        reduced,
                        world,
                        reduced,
                    )
                return reduced
        gate = get_prefill_admission_gate(self)
        if gate is None:
            return chunked_prefill_size
        try:
            granted = int(gate.granted_width(requested))
        except Exception as e:  # noqa: BLE001
            # An actuator that cannot compute a width must leave the width
            # alone. It is a safety net; a net that tears must not take down
            # the thing it was protecting.
            logger.error("#794 corridor width actuator failed: %s", e)
            return chunked_prefill_size
        if granted <= 0 or granted > requested:
            return chunked_prefill_size
        return granted

    def get_new_batch_prefill(self, running_batch: ScheduleBatch) -> NextBatchPlan:
        prefill_delayer_single_pass = None
        if self.prefill_delayer:
            # Get max usage across all pools for prefill delay decision
            max_pool_usage = (
                self.pool_stats_observer.get_pool_stats().get_max_pool_usage()
            )
            prefill_delayer_single_pass = PrefillDelayerSinglePassExecutor(
                self.prefill_delayer, token_usage=max_pool_usage
            )

        try:
            ret, running_batch = self._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=prefill_delayer_single_pass,
                running_batch=running_batch,
            )
        except PPScheduleRefused as refusal:
            # #791 CORE: THE LOUD REFUSAL, and the ONLY thing it may do.
            # It may not narrow the batch, retry with different numbers, or
            # fall through to a decode batch -- every one of those pairs this
            # rank's metadata with the upstream's hidden states for a
            # different pass. It voids the pass, through #797's existing
            # machinery and with no new mechanism: `_pp_admission_pass_voided`
            # is what `_event_loop_pp_body` reads to forward `pass_voided`
            # and, with no batch built, to drain the upstream's orphaned
            # proxy.
            ret = self._pp_refuse_forwarded_schedule(refusal)

        if self.prefill_delayer:
            prefill_delayer_single_pass.finalize(actual_prefill=ret is not None)

        if envs.SGLANG_PP_ADMISSION_TRACE.get():
            self._trace_pp_admission_verdict(ret)

        return NextBatchPlan(batch_to_run=ret, running_batch=running_batch)

    def _pp_scheduled_extents(self) -> Optional[Dict[str, Tuple[int, int]]]:
        """#791 CORE: the forwarded pass geometry this rank must EXECUTE.

        `None` -- and therefore the pre-#791 local arithmetic, unentered --
        on every rank that owns its own admission truth: the first PP rank,
        which BUILDS the decision, and every boot with `pp_size <= 1`, which
        has no upstream to be told by. Downstream, it is the mapping received
        earlier in this same pass; `None` there too on a pass that received
        nothing, which the admission loop's own membership gate already
        refuses to admit anything on.
        """
        ps = getattr(self, "ps", None)
        if ps is None or ps.pp_size <= 1 or ps.pp_rank == 0:
            return None
        return getattr(self, "_pp_admission_incoming_schedule", None)

    def _pp_refuse_forwarded_schedule(self, refusal: Exception) -> None:
        """#791 CORE: turn an unexecutable forwarded geometry into a void.

        QUOTING THE DECISION IS THE POINT. `PPScheduleRefused` carries the
        forwarded numbers and this rank's own, formatted by
        `pp_admission_congruence.schedule_refusal_reason`, so the log line
        names what the upstream committed rather than only what this rank
        found -- the failure mode this replaces was a rank quietly building a
        different batch and saying nothing at all.

        NO NEW MECHANISM. The pass is voided exactly as a #797 retraction
        voids it: `_pp_admission_pass_voided` is set so `_event_loop_pp_body`
        forwards `pass_voided=True` and drains the upstream's orphaned proxy,
        and the decision this rank forwards is emptied by
        `void_pp_admission_decision` so every rank after it makes the same
        membership decision. Re-noted against the slot, because the
        expectation was recorded before this refusal could be known.
        """
        self._pp_pass_schedule_refusals = (
            getattr(self, "_pp_pass_schedule_refusals", 0) + 1
        )
        logger.error(
            "#791 PP-ADMISSION forwarded schedule REFUSED on rank %s: %s "
            "Voiding the whole pass rather than building a batch of a shape "
            "the upstream did not decide.",
            getattr(getattr(self, "ps", None), "pp_rank", None),
            refusal,
        )
        self._pp_admission_pass_voided = True
        self._pp_admission_incoming_effective = {}
        self._pp_admission_incoming_schedule = {}
        amended = getattr(self, "_pp_admission_amended_to_forward", None)
        if amended is not None:
            amended = void_pp_admission_decision(amended)
            self._pp_admission_amended_to_forward = amended
            note = getattr(self, "_pp_note_output_expectation", None)
            if note is not None:
                note(
                    amended.mb_id,
                    bool(getattr(self, "_pp_output_expected_incoming", False)),
                    amended,
                )
        return None

    def _trace_pp_admission_verdict(self, ret: Optional[ScheduleBatch]) -> None:
        """#788: record THIS rank's admission verdict and the inputs behind it.

        WHY THIS EXISTS. Under PP the ranks are N independent schedulers that
        agree only by determinism: the request is chain-forwarded to every
        stage unconditionally, but each stage re-derives admission from its
        OWN queue and radix state, and the proxy send to the next stage is
        gated on this rank's own batch. So one rank declining while another
        admits is a silent, unbounded pipeline deadlock -- the admitting rank
        blocks for a proxy nobody will ever send. We have that deadlock
        measured, and a mechanism proof for its cause (#616g's uniformity
        floors are scoped to tp_cpu_group and switch OFF entirely when that
        group has one member, which is every rank of a TP=1/PP=3 boot), but
        no captured value showing the ranks actually disagreeing. One
        instrumented boot with this on turns that into evidence -- or
        falsifies it honestly, which is just as useful.

        HOST-SIDE VALUES ONLY, and that is not a style preference. #790 was
        exactly this shape: a diagnostic passed a CUDA tensor as a logging
        argument, the %s formatting forced a D2H copy and a stream sync inside
        logging.emit, and the scheduler sat there for 25 minutes. So every
        value below is an int already resident on the host. len() on a tensor
        reads its shape and does NOT synchronize, which is why prefix length
        is taken that way rather than by reading the tensor.

        VACUOUS VERDICTS ARE SUPPRESSED (#788, after boot instr11's 5.9 GB
        idle log). A pass with nothing admitted, queued, running or chunked
        is dropped and counted; a roll-up line naming the count is emitted
        every ``PP_ADMISSION_VACUOUS_ROLLUP_EVERY`` such passes and in front
        of the next informative verdict. Both the predicate and the cadence
        read only rank-congruent payload fields, which is what keeps the
        cross-rank payload diff usable -- see
        ``pp_admission_verdict_is_vacuous``.
        """
        try:
            alloc = self.token_to_kv_pool_allocator
            n_reqs = 0 if ret is None else len(ret.reqs)
            queue = len(self.waiting_queue)
            running = len(self.running_batch.reqs) if self.running_batch else 0
            chunked = 1 if self.chunked_req is not None else 0

            # #788: drop the VACUOUS verdicts, keep every informative one.
            # Boot instr11 ran this instrument for three hours against an
            # idle server and wrote 5.9 GB; instr10's census shows the
            # payload -- 146023 lines per rank, overwhelmingly
            # "DECLINE n_reqs=0 ... queue=0 running=0 chunked=0". A verdict
            # taken over an empty scheduler cannot show two ranks
            # disagreeing, because there is nothing for them to disagree
            # about, so the flood buried exactly the lines the instrument
            # exists to produce.
            #
            # `ret is None` is the DECLINE half of the `verdict` payload
            # field, so this whole condition -- like the predicate it calls,
            # whose docstring explains why this is load-bearing rather than
            # tidy -- is a pure function of RANK-CONGRUENT payload data. Every
            # rank therefore decides to speak or stay silent identically, and
            # the acceptance gate's cross-rank payload diff keeps meaning what
            # it says.
            vacuous = ret is None and pp_admission_verdict_is_vacuous(
                n_reqs, queue, running, chunked
            )
            run = getattr(self, "_pp_admission_vacuous_run", 0)
            if vacuous:
                run += 1
                # The roll-up cadence is a COUNT of passes, not a duration:
                # an iteration counter advances identically on every rank,
                # a clock does not.
                if run >= PP_ADMISSION_VACUOUS_ROLLUP_EVERY:
                    rollup, run = run, 0
                else:
                    rollup = 0
            else:
                rollup, run = run, 0
            self._pp_admission_vacuous_run = run
            if rollup:
                # Silence must never be ambiguous. Without this line a reader
                # cannot tell "nothing was happening" from "the instrument
                # died", which is the one thing an instrument may not leave
                # open. The count is congruent too, so it diffs across ranks
                # exactly like the verdict lines. Deliberately NOT spelled
                # "verdict=": the gate greps that token to collect payloads.
                logger.info(
                    "#788 PP-ADMISSION suppressed=%d vacuous verdicts "
                    "(n_reqs=0 queue=0 running=0 chunked=0) "
                    "since the last emitted line",
                    rollup,
                )
            if vacuous:
                return

            if ret is None:
                verdict = "DECLINE"
                rids = ""
                prefix_lens = ""
            else:
                verdict = "ADMIT"
                # Truncated on purpose: this is a divergence signal, not a
                # batch dump. A flood here has cost this feature a self-kill
                # before (see the origin guard in request_receiver).
                rids = ",".join(r.rid for r in ret.reqs[:4])
                # #796: NO `or []` HERE. `prefix_indices` is a tensor, and
                # `x or []` asks `bool(x)`, which torch refuses for an empty
                # tensor ("Boolean value of Tensor with no values is
                # ambiguous") and for a multi-element one. This branch runs
                # only on an ADMIT, so the effect was that EVERY admitting
                # pass lost its trace line to the except below while the idle
                # DECLINE passes logged fine -- boot instr6 showed all three
                # ranks reporting a bare "RuntimeError" at the exact pass the
                # first real request arrived, and the verdict lines that
                # mattered were the ones that never got printed.
                # The docstring above already said len() is the right
                # spelling; the `or []` slipped in anyway.
                prefix_lens = ",".join(
                    str(
                        0
                        if getattr(r, "prefix_indices", None) is None
                        else len(r.prefix_indices)
                    )
                    for r in ret.reqs[:4]
                )

            # #788 (boot instr14): drop passes that REPEAT A CYCLE already
            # printed in full. The vacuous predicate above only fires on an
            # empty scheduler, so under burst load every pass counted as
            # informative and printed: instr14 wrote 141513 admission lines
            # in ten minutes, 20367381 bytes, at 13.03 MB/min peak.
            #
            # A plain "same as last line" test recovers NOTHING here -- the
            # measured stream alternates (queue=1 running=2 / queue=1
            # running=4 / ...), so consecutive lines are almost never equal
            # and the run lengths are 1. See log_cycle_collapse for the
            # replay that establishes this.
            #
            # THE KEY IS THE CONGRUENT PAYLOAD AND NOTHING ELSE. `avail` and
            # `evictable` are deliberately absent: they are this rank's own
            # pool accounting and legitimately differ between ranks, so
            # keying on them would let rank 0 suppress a line rank 1 emits
            # and make verdict_790.sh's cross-rank payload diff report a
            # divergence that never happened. Same law as
            # pp_admission_verdict_is_vacuous, whose docstring carries the
            # long form: no wall-clock time, no per-rank counter, no log
            # volume, no sampling.
            # Created on first use, like `_pp_admission_vacuous_run` above,
            # so the instrument owns all of its own state and adds nothing
            # to __init__ that the scheduler would carry when the trace is
            # off.
            collapser = getattr(self, "_pp_admission_repeat_collapse", None)
            if collapser is None:
                collapser = CycleCollapse()
                self._pp_admission_repeat_collapse = collapser
            collapse = collapser.observe(
                (verdict, n_reqs, rids, prefix_lens, queue, running, chunked)
            )
            if collapse.rollup:
                (
                    last_verdict,
                    last_n_reqs,
                    _last_rids,
                    _last_prefix_lens,
                    last_queue,
                    last_running,
                    last_chunked,
                ) = collapse.last
                # Silence must never be ambiguous: without this line a reader
                # cannot tell "the scheduler kept doing the same thing" from
                # "the instrument died". The count and the period are
                # congruent too, so they diff across ranks exactly like the
                # verdict lines. Deliberately NOT spelled "verdict=": the
                # gate greps that token to collect payloads.
                logger.info(
                    "#788 PP-ADMISSION suppressed=%d passes repeating a "
                    "%d-pass cycle (last: decision=%s n_reqs=%d queue=%d "
                    "running=%d chunked=%d) since the last emitted line",
                    collapse.rollup,
                    collapse.period,
                    last_verdict,
                    last_n_reqs,
                    last_queue,
                    last_running,
                    last_chunked,
                )
            if not collapse.emit:
                return

            logger.info(
                "#788 PP-ADMISSION verdict=%s n_reqs=%d rids=%s prefix_lens=%s "
                "avail=%d evictable=%d queue=%d running=%d chunked=%d reason=%s",
                verdict,
                n_reqs,
                rids,
                prefix_lens,
                int(alloc.available_size()),
                int(self.tree_cache.evictable_size()),
                queue,
                running,
                chunked,
                # #791b-instr22: WHICH gate declined, set by
                # _get_new_batch_prefill_raw. "-" on ADMIT lines and on
                # paths that never reached the prefill raw body (e.g. a
                # decode-only pass), so the field never lies by omission.
                getattr(self, "_admission_decline_note", None) or "-",
            )
        except Exception as e:  # noqa: BLE001
            # An instrument must never be able to kill the scheduler it is
            # measuring -- that is the #790 lesson generalised.
            # #796: the MESSAGE, not just the type. Boot instr6 logged three
            # ranks reporting a bare "RuntimeError" at the exact pass a real
            # request first appeared, and a bare type name is a diagnostic
            # dead end -- it cost a whole boot to learn nothing. Still
            # swallowed rather than raised: an instrument must never be able
            # to kill the scheduler it is measuring (the #790 lesson
            # generalised); it just has to say what went wrong.
            logger.warning(
                "#788 PP-ADMISSION trace unavailable: %s: %s",
                type(e).__name__,
                e,
            )

    def _drain_prefetch_progress(self) -> Dict[str, bool]:
        """Advance EVERY queued request's HiCache storage prefetch, and return
        the per-request verdict the admission loop then reads.

        #580, the rank-local-condition-before-a-group-collective family
        (#94/#194/#312/#431). ``UnifiedRadixCache.check_prefetch_progress``
        carries two collectives on the attn/TP groups -- the MAX reduce inside
        ``can_terminate_prefetch`` and the ``check_prefetch_progress``
        MIN reduce over completed tokens + sidecar hit pages. It used to be
        called from INSIDE the waiting-queue admission loop, i.e. only for the
        prefix of the queue that the loop reached before one of its exits:

          * ``running_batch.batch_is_full`` -- a PERSISTENT flag carried on
            running_batch, set from ``get_num_allocatable_reqs`` and from the
            adder's NO_TOKEN verdict, which reads
            ``token_to_kv_pool_allocator.available_size()``;
          * the ``AddReqResult != CONTINUE`` break, same source;
          * and, before the loop is even entered, the early returns on
            ``batch_is_full``, on ``min_free_slots_delayer.should_delay`` and
            on ``get_num_allocatable_reqs(...) <= 0``.

        Every one of those reads THIS rank's pool. Under uneven TP/DCP the
        pools differ by construction, so near a pool boundary one rank stops
        at request N while a peer walks on to N+1 and enters the collectives
        alone. The peer then meets the stopping rank's next collective instead
        of its own -- two different ops of two different widths on one gloo
        group, which is the observed ``op.preamble.length <= op.nbytes. 16 vs
        4`` abort. The existing #580 vote
        (``prefetch_from_storage``/``prefetch_participation_vote``) makes
        REGISTRATION rank-uniform; it says nothing about who later walks into
        the progress check, which is this defect.

        The fix is structural, not a mask: the entry decision is moved off the
        rank-local admission state entirely. The drain runs over
        ``self.waiting_queue``, which is replicated across the TP ranks (same
        request stream, same admission order -- see ``_add_request_to_queue``),
        under the rank-uniform ``enable_hicache_storage`` config flag, and it
        visits ALL of the queue, so the number and order of collectives no
        longer depend on any per-rank pool value. All ranks enter for a given
        request, or none do.

        Placement: called from ``_get_new_batch_prefill_raw`` AFTER the
        grammar queue has drained into the waiting queue (those requests
        register prefetches in ``_add_request_to_queue``, so an earlier call
        would miss them) and BEFORE the first rank-local admission predicate.
        That is the same point ``check_hicache_events`` already occupies, so
        no new assumption about which ranks reach this line is introduced.

        Cost: unchanged in the common case. A request with no registered
        prefetch returns at the tree cache's ``req_id not in
        ongoing_prefetch`` guard without any collective, and
        ``ongoing_prefetch`` is exactly the set the participation vote
        already keeps rank-uniform. (That set lives on the TREE CACHE, not
        on the scheduler -- the 610 drift guard reads ``self.``-qualified
        names in this method's source as scheduler surface.) Off HiCache storage
        this returns an empty map and touches nothing -- byte-identical.

        Behavioural note, deliberately named: under
        ``--hicache-storage-prefetch-policy best_effort`` (NOT the default,
        which is ``timeout``) ``can_terminate_prefetch`` returns True
        immediately, so a queued request's prefetch is now cut on the first
        iteration after registration rather than whenever admission happened
        to look at it. "Whenever admission looked at it" is precisely the
        rank-local quantity this removes; under a group the only rank-uniform
        answer available without adding a collective is "every iteration".
        """
        if not self.enable_hicache_storage:
            return {}
        return {
            req.rid: self.tree_cache.check_prefetch_progress(req.rid)
            for req in self.waiting_queue
        }

    def _prefetch_done_for(self, req: Req, drained: Dict[str, bool]) -> bool:
        """Read the drained verdict for ``req``; refuse to answer locally.

        A miss means the waiting queue changed between the drain and the
        admission loop, i.e. the set of requests is no longer the replicated
        one the drain was computed over. Falling back to a live
        ``check_prefetch_progress`` call here would re-open #580 exactly:
        this rank would enter a collective its peers are not in. On a group
        that is named and refused; on a single rank there is nothing to
        diverge from, so the live call is correct and kept.
        """
        verdict = drained.get(req.rid)
        if verdict is not None:
            return verdict
        if self.ps.tp_size > 1:
            raise RuntimeError(
                f"request {req.rid} reached the prefill admission loop without "
                "a drained HiCache prefetch verdict on a multi-rank boot. The "
                "drain runs over the replicated waiting queue before any "
                "rank-local admission predicate; a miss means the queue was "
                "mutated in between, so calling check_prefetch_progress() here "
                "would enter a collective the peer ranks are not in -- the "
                "#580 failure."
            )
        return self.tree_cache.check_prefetch_progress(req.rid)

    @property
    def chunked_commitment_ledger(self):
        """#701 defect (b): the cross-pass reservation, owned HERE.

        A ``PrefillAdder`` is rebuilt on every pass (see the construction in
        ``_get_new_batch_prefill_raw``), so a ledger the adder made would forget
        a resident chunked request's outstanding prefill exactly when the next
        pass needs to see it -- which IS the defect. The scheduler outlives the
        passes, so it is the right owner.

        Not per-request-set either: commitments are keyed by request id and
        released on finish/abort/retract, so the ledger self-cleans; tying it to
        a set that turns over would drop live commitments instead.

        LAZY on purpose. This must not depend on anything ``__init__`` builds --
        it holds no pool, no allocator and no config, only integers keyed by
        request id -- and constructing it on first read is what lets the
        ownership be tested without standing up a scheduler.
        """
        from sglang.srt.planner.chunked_admission import ChunkedCommitmentLedger

        ledger = getattr(self, "_chunked_commitment_ledger", None)
        if ledger is None:
            ledger = ChunkedCommitmentLedger()
            self._chunked_commitment_ledger = ledger
        return ledger

    def _get_new_batch_prefill_raw(
        self,
        prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor],
        running_batch: ScheduleBatch,
    ) -> Tuple[Optional[ScheduleBatch], ScheduleBatch]:
        # #791b-instr22: WHY a pass admitted nothing, named. instr22 died on
        # two TP replicas declining the same queued request a third one
        # admitted (verdict=DECLINE vs verdict=ADMIT, 11:16:26), and the
        # trace could show THAT they diverged but not WHICH of the half-dozen
        # rank-local gates split. Every decline path below therefore names
        # itself here, and _trace_pp_admission_verdict prints it on DECLINE
        # lines. Host-side strings only (#790).
        self._admission_decline_note = None

        # Check if the grammar is ready in the grammar queue
        if self.grammar_manager.has_waiting_grammars():
            ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
            for req in ready_grammar_requests:
                self._add_request_to_queue(req)

        if self.enable_hierarchical_cache or self.server_args.enable_flexkv:
            self.tree_cache.check_hicache_events()

        # #580: rank-uniform entry into the prefetch-progress collectives.
        # MUST stay above every early return and every loop exit below -- all
        # of them read this rank's own pool. See _drain_prefetch_progress.
        #
        # #737 -- WHAT THIS GUARANTEE IS NOT. It secures uniform entry WITHIN
        # this function once a rank calls it. It says nothing about whether all
        # ranks CALL this function on the same tick, and under pipeline
        # parallelism they provably do not: `_get_new_batch_prefill_raw` runs in
        # the per-microbatch loop body, whose very next step
        # (`_pp_recv_proxy_tensors`) blocks a stage on its upstream, leaving
        # stages at different microbatch offsets by design.
        #
        # A collective placed here therefore requires an alignment this position
        # cannot supply. The HiCache ack-count reduction leaned on this comment
        # and deadlocked on 2026-08-17 (PP0/PP1 in the drain, PP2 in the
        # pipeline recv); it is now rank-local. Anything added above that needs
        # GROUP agreement needs a pipeline-aligned point, not this one.
        # #791b: the TP loop's uniform-budget site already ran this drain
        # (once per iteration, before packing the prefetch ballot) and
        # memoised it -- consume it ONCE here so the pass drains exactly
        # once. The pop is what keeps a PP-loop pass (which never runs the
        # budget site) from reading a stale TP memo: absent memo means this
        # path drains for itself, byte-identical to before. The ballot is
        # popped with the same consume-once discipline for the same reason.
        prefetch_verdicts = self.__dict__.pop("_pass_prefetch_verdicts", None)
        if prefetch_verdicts is None:
            prefetch_verdicts = self._drain_prefetch_progress()
        _prefetch_ballot = self.__dict__.pop("_uniform_prefetch_ballot", None)

        if self.enable_priority_preemption or self.is_hybrid_swa:
            # Reset batch_is_full to try preemption with a prefill adder.
            running_batch.batch_is_full = False

        if (
            running_batch.batch_is_full or len(self.waiting_queue) == 0
        ) and self.chunked_req is None:
            self._admission_decline_note = (
                f"gate=batch_full_or_empty_queue(batch_is_full="
                f"{int(running_batch.batch_is_full)},queue={len(self.waiting_queue)})"
            )
            return None, running_batch

        running_bs = len(running_batch.reqs)
        # Skipped during a chunked prefill: that pass must proceed regardless.
        if (
            self.min_free_slots_delayer is not None
            and self.chunked_req is None
            and self.min_free_slots_delayer.should_delay(
                running_bs=running_bs,
                num_allocatable_reqs=self.get_num_allocatable_reqs(running_bs),
            )
        ):
            self._admission_decline_note = (
                f"gate=min_free_slots_delay(running_bs={running_bs},"
                f"allocatable={self.get_num_allocatable_reqs(running_bs)})"
            )
            return None, running_batch

        # Ignore the check if self.chunked_req is not None.
        # In the non-PP case, when self.chunked_req is not None, num_allocatable_reqs should always be greater than 0,
        # as the space for the chunked requests has just been released.
        # In PP case, chunked requests (or dllm requests) can start in one microbatch and end in another microbatch, so the max_running_requests per microbatch should not be strict.
        # Instead, we should always allow chunked requests to be added, otherwise, there will be a memory leak.
        if (
            self.get_num_allocatable_reqs(running_bs) <= 0
            and self.chunked_req is None
            and not self.enable_priority_preemption
        ):
            running_batch.batch_is_full = True
            self._admission_decline_note = (
                f"gate=no_allocatable_reqs(running_bs={running_bs})"
            )
            return None, running_batch

        # Get priority queue
        self.policy.calc_priority(self.waiting_queue, running_batch)

        if TEST_RETRACT and running_bs > TEST_RETRACT_NO_PREFILL_BS:
            # If we are testing retraction and the running batch size exceeds
            # TEST_RETRACT_NO_PREFILL_BS, we skip the prefill to keep the requests
            # in the waiting queue.
            self._admission_decline_note = "gate=test_retract"
            return None, running_batch

        # Determine chunked_prefill_size for this batch
        chunked_prefill_size = self.dynamic_chunked_prefill_size()

        # #656 item 15a, AT THE PREFILL ALLOCATION SITE (register C17).
        #
        # The corridor law used to be enforced at exactly one allocation site,
        # the flip seam, and successor 33's acceptance breached at this one
        # instead: a 272k-token bs1 prefill walked the binding card down to
        # 1001 MiB and held it there for 1.6 s, until the next seam armed the
        # very same gate and reclaimed 964 MiB in one call. The gate worked.
        # It was not called from here.
        #
        # IT SPILLS; IT NEVER REFUSES, and that is a correctness requirement
        # rather than caution. This gate reads THIS RANK'S free column, while
        # prefill admission has to stay rank-uniform -- exactly the property
        # the DCP note below is about. A rank-local refusal here would let one
        # rank admit work its peers declined, which is the capacity desync
        # that previously left a scheduler not heartbeating with every rank
        # alive. So the verdict is logged, counted and ignored for the
        # admission decision. See managers/corridor_admission.py.
        guard_prefill_admission(self, chunked_prefill_size)

        # #794 THE VERDICT NOW ACTUATES -- AS A WIDTH, NEVER AS A REFUSAL.
        #
        # The comment above is still true and is the reason this is a separate
        # line: a rank-local REFUSAL splits the group's admission decision and
        # hangs it. A rank-local NARROWING does not, provided it is taken where
        # the pass geometry is decided, and it is the only remedy that matches
        # the physics -- the GDN prefill transient is first-order in the chunk
        # width, so a chunk the card cannot fund always has a prefix it can.
        #
        # TWICE MEASURED, TWICE FATAL, before this line existed. 2026-08-21
        # 17:33 the gate reported "corridor shortfall of 981 MiB for rank 0 /
        # the corridor cannot be restored ahead of this chunk" and the chunk
        # ran, dying in in_proj_qkvz (256 MiB asked, 131.69 MiB free). 18:01,
        # after the width was halved to 4096 by hand, the same rank died one
        # allocation further down the same layer -- causal_conv1d, 80 MiB
        # asked, 5.69 MiB free. A static width cannot answer a free column
        # that moves; only a width derived FROM it can.
        chunked_prefill_size = self._corridor_granted_prefill_width(
            chunked_prefill_size
        )

        # RANK-UNIFORM prefill admission budget (uneven DCP): the deficit was
        # min-reduced once this iteration in update_dcp_admission_state (single
        # collective, pre-branch). Read the stored value -- NO collective here,
        # so admission stays uniform without adding a branch-local reduce. 0 on
        # the default path (offload manager absent) -> byte-identical.
        dcp_avail_deficit = 0
        prefill_spill_regions = 0
        prefill_spill_region_tokens = 0
        prefill_spill_deep = False
        # #610: off the offload path the same pin now comes from
        # `_update_uniform_pool_budget`, which ran unconditionally and
        # pre-branch at the top of this iteration. 0 unless uneven DCP is
        # active, so the default path is unchanged.
        dcp_avail_deficit = self.uniform_budget_deficit()
        if self.kv_session_offload is not None:
            dcp_avail_deficit = self.kv_session_offload.dcp_budget_deficit()
            # PS2 (deep prefill-spill): replicated region capacity + master
            # gate. Both 0/False when --kv-session-offload-prefill is off, so
            # the deep branch in the adder is unreachable (byte-identical).
            prefill_spill_region_tokens = int(
                self.kv_session_offload.region_tokens
                if self.kv_session_offload.prefill_spill
                else 0
            )
            # PS2 master gate: the feature flag AND no spec configuration that
            # would later READ the prompt's draft KV. Under plain speculative
            # decoding the draft extend of a born-spilled prefill is skipped
            # (nothing reads it) and PS2 runs -- see
            # prefill_spill_deep_reject_reason for the full argument and for
            # the three conditions that still block it.
            from sglang.srt.managers.kv_session_offload import (
                prefill_spill_deep_gate,
                prefill_spill_deep_reject_reason,
                resume_under_spec_enabled,
            )

            _spec_active = not self.spec_algorithm.is_none()
            # BOOT-time DFLASH predicate: under cross-algorithm switching the
            # DFLASH prefill append runs on every prefill regardless of which
            # rung is active, so keying this to the active rung would miss it.
            _dflash_prefill = (
                bool(getattr(self.server_args, "speculative_cross_algorithm", False))
                or self.spec_algorithm.is_dflash_family()
            )
            _spec_in_tick = bool(
                getattr(self.kv_session_offload, "spec_in_tick_ready", False)
            )
            _resume_spec = resume_under_spec_enabled()
            # C26: PS2's sentinel out_cache_loc is only diverted on the DCP
            # lane. Admitting it on plain TP sends host row ids into
            # store_kvcache and asserts device-side. Replicated boot config,
            # so every rank computes the same verdict without a collective.
            _backend_hook = bool(
                getattr(
                    self.kv_session_offload,
                    "prefill_spill_deep_backend_ok",
                    False,
                )
            )
            prefill_spill_deep = prefill_spill_deep_gate(
                self.kv_session_offload.prefill_spill,
                _spec_active,
                spec_in_tick_ready=_spec_in_tick,
                resume_under_spec=_resume_spec,
                dflash_prefill_append=_dflash_prefill,
                backend_write_hook=_backend_hook,
            )
            if self.kv_session_offload.prefill_spill and not prefill_spill_deep:
                _reason = prefill_spill_deep_reject_reason(
                    _spec_active,
                    _spec_in_tick,
                    _resume_spec,
                    _dflash_prefill,
                    _backend_hook,
                )
                if _reason is not None and _reason != getattr(
                    self, "_ps2_spec_declined_reason", None
                ):
                    self._ps2_spec_declined_reason = _reason
                    logger.info(
                        "kv-session-offload prefill-spill (PS2): DEEP "
                        "born-spilled admission is declined -- %s "
                        "(spec=%s). PS1 born-spilled admission is unaffected.",
                        _reason,
                        self.spec_algorithm,
                    )
            # Prefill-Spill (PS1-V1a): replicated free-region count (0 when the
            # feature is off -> the adder relaxation is inert). No collective
            # here (rank-uniform by construction, see prefill_spill_free_regions).
            prefill_spill_regions = self.kv_session_offload.prefill_spill_free_regions()

        # Prefill policy
        from sglang.srt.mem_cache.common import published_fundable_floor

        adder = PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            running_batch,
            self.new_token_ratio_tracker.current,
            self.max_prefill_tokens,
            chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
            self.priority_scheduling_preemption_threshold,
            max_prefill_bs=self.max_prefill_bs,
            # #287: the adder is an ADMISSION consumer, so it gets the
            # floating limit, not the ceiling the pools were built for.
            max_running_requests=self.admission_limiter.current,
            prefill_max_requests=self.server_args.prefill_max_requests,
            prefill_delayer_single_pass=prefill_delayer_single_pass,
            dllm_config=self.dllm_config,
            waiting_queue_len=len(self.waiting_queue),
            dcp_avail_deficit=dcp_avail_deficit,
            prefill_spill_regions=prefill_spill_regions,
            prefill_spill_region_tokens=prefill_spill_region_tokens,
            prefill_spill_deep=prefill_spill_deep,
            # #681: the NEW-request half of #679's park. `add_chunked_req`
            # already refuses to schedule a chunk the pool cannot fund, on the
            # group-published floor; without this the sibling gate for fresh
            # requests still branches on THIS rank's pool and admits batches
            # `alloc_for_extend` then dies on. Read once per iteration, from
            # the same helper the chunked gate uses, so both gates agree by
            # construction rather than by two copies of the arithmetic.
            #
            # `published_fundable_floor` and not `fundable_extend_tokens`
            # directly: as a CEILING a mis-read 0 would admit nothing forever,
            # so the cap is applied only where a group floor was actually
            # published. See that helper for why the two gates read the same
            # number through different doors.
            fundable_extend_floor=published_fundable_floor(self.tree_cache),
            # #701 defect (b): pass the SCHEDULER-owned ledger into the adder
            # this pass builds. Constructing it here instead would reset the
            # outstanding commitments every pass, which is the hole the
            # chokepoint subtraction cannot close on its own.
            commitment_ledger=self.chunked_commitment_ledger,
            # #791 CORE: the pass geometry the first rank already decided, or
            # None on the rank that decides it and on every non-PP boot. This
            # is what turns the adder from a second scheduler into an
            # executor -- see `PrefillAdder._add_scheduled_req`.
            scheduled_extents=self._pp_scheduled_extents(),
        )

        if self.chunked_req is not None:
            self.chunked_req.init_next_round_input()
            # #679 rung 1-3: SPEND RELIEF BEFORE THE PARK, not instead of it.
            #
            # The ladder runs here and nowhere else: this is the last point at
            # which the pool can still be topped up before add_chunked_req
            # decides, and it is downstream of the pre-branch reduce (this
            # iteration's uniform_min_avail is already published), so no rung
            # takes a collective of its own.
            #
            # ORDER IS THE CONTRACT (DESIGN_679 §4, rule 1): the ladder changes
            # what there is to decide from; add_chunked_req still decides. A
            # ladder that admitted work itself would be a second admission
            # authority, and the park guard would no longer be final.
            self._maybe_spend_admission_relief(running_batch)
            self.chunked_req = adder.add_chunked_req(self.chunked_req)

        if self.enable_lora:
            running_loras = {
                req.lora_id for req in running_batch.reqs if not req.finished()
            }
            # Account for LoRAs that are already loaded in the adder, such as chunked requests
            running_loras.update(req.lora_id for req in adder.can_run_list)

            if self.lora_drainer:
                self.lora_drainer.update_draining_state(
                    self.waiting_queue,
                    running_batch.reqs,
                )

        mamba_allocator = getattr(self.req_to_token_pool, "mamba_allocator", None)
        if mamba_allocator is not None:
            mamba_allocator.alloc_group_begin(len(self.waiting_queue))
        # #791 CORE: a refusal raised inside the loop is CARRIED, not thrown,
        # so `alloc_group_end` below still runs on its own line. Leaking an
        # open mamba alloc group would replace one corruption with another.
        schedule_refusal: Optional[PPScheduleRefused] = None
        # #791b-instr22: per-category skip census for this pass's loop, and
        # the FIRST skipped rid per category -- the divergence instrument's
        # payload. Ints and short strings only (#790).
        _skips: Dict[str, int] = {}
        _skip_first_rid: Dict[str, str] = {}

        def _note_skip(kind: str, rid) -> None:
            _skips[kind] = _skips.get(kind, 0) + 1
            _skip_first_rid.setdefault(kind, str(rid))

        # Get requests from the waiting queue to a new prefill batch
        for req in self.waiting_queue:
            if self.enable_lora and not self._can_schedule_lora_req(req, running_loras):
                _note_skip("lora", req.rid)
                continue

            running_bs = len(running_batch.reqs)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                running_batch.batch_is_full = True
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                # In prefill mode, prealloc queue and transfer queue can also take memory,
                # so we need to check if the available size for the actual available size.
                if len(adder.can_run_list) >= self.req_to_token_pool.available_size():
                    running_batch.batch_is_full = True

            if running_batch.batch_is_full:
                if (
                    not self.enable_priority_preemption
                    or not adder.preempt_to_schedule(req, self.server_args)
                ):
                    _note_skip("batch_full_break", req.rid)
                    break

            if self.enable_hicache_storage:
                # #580: READ the verdict drained before the rank-local
                # predicates above; do NOT call into the tree cache here. The
                # call carries collectives and this point is only reachable on
                # the ranks whose own pool let them get this far.
                _local_prefetch_done = self._prefetch_done_for(req, prefetch_verdicts)
                # #791b: THE GROUP VERDICT, not the local one. instr22/23:
                # the storage backend finishes each rank's load at that
                # rank's own speed, so the local verdict split the TP
                # replicas -- one rank admitted, its peers declined, and the
                # crossed collectives took the group down. MIN==AND: this
                # can only ever DELAY an admission (group-done implies
                # locally done), never force one on an unfinished rank.
                # Resolved through the module so the can-fail test can
                # neuter exactly this application in a child process.
                prefetch_done = prefetch_ballot.prefetch_done_under_ballot(
                    _local_prefetch_done, req.rid, _prefetch_ballot
                )
                if not prefetch_done:
                    # skip staging requests that are ongoing prefetch
                    if _prefetch_ballot is not None and req.rid not in _prefetch_ballot:
                        _note_skip("prefetch_ballot_uncovered", req.rid)
                    elif _local_prefetch_done:
                        _note_skip("prefetch_pending_group", req.rid)
                    else:
                        _note_skip("prefetch_pending", req.rid)
                    continue
                # Pop the number of tokens loaded from storage (L3 hits)
                loaded_tokens = self.tree_cache.pop_prefetch_loaded_tokens(req.rid)
                if loaded_tokens > 0:
                    req.storage_hit_length = loaded_tokens

            req.init_next_round_input(self.tree_cache)

            # #791 PP ADMISSION UNIFORMITY. Every PP stage independently
            # re-derives its own admission verdict from its own local radix
            # state; nothing forwards the DECISION alongside the
            # chain-forwarded requests, so two stages can disagree about
            # which requests are admitted or how much prefix each one
            # reuses -- and since `prepare_for_extend` (below, via
            # ScheduleBatch.init_new) sizes the cross-stage tensor directly
            # off `len(req.prefix_indices)`, a length disagreement is a
            # SHAPE disagreement (pp_admission_congruence.py's module
            # docstring: WHAT CROSSES THE WIRE). Applied HERE, strictly
            # BEFORE `adder.add_one_req` commits `extend_range` against
            # whatever `req.prefix_indices` currently is -- `prepare_for_
            # extend` is not safe to call a second time (fresh KV
            # allocation, asserted invariants each call), so there is no
            # later point at which a mismatch could still be corrected
            # without a second pass over an already-committed batch.
            #
            # NO COLLECTIVE: `told` below is either this rank's own
            # guard-clamped candidate (PP0, via the #630 learned floor) or a
            # value that already crossed the wire earlier THIS pass
            # (downstream, via scheduler_pp_mixin.py's pre-loop reconcile) --
            # never a new blocking op introduced on this path (see the
            # 2026-08-17 deadlock family this must not repeat, referenced
            # throughout this method).
            if self.ps.pp_size > 1:
                if self.ps.pp_rank == 0:
                    told = self._pp_admission_guard.prefix_len_for(
                        req.rid, len(req.prefix_indices)
                    )
                    if told < len(req.prefix_indices):
                        req.prefix_indices = req.prefix_indices[:told]
                elif self._pp_admission_incoming_effective is not None:
                    told = self._pp_admission_incoming_effective.get(req.rid)
                    if told is None:
                        # Not named by PP0's decision this pass: excluded by
                        # PP0's own verdict, retracted by an earlier rank's
                        # reconcile, or simply not visible to PP0 yet.
                        # Uniform membership means this rank must not admit
                        # it either. Left unmutated in self.waiting_queue
                        # (never added to can_run_set below), so it is
                        # reconsidered on a later pass -- the same
                        # requeue-for-free mechanism this loop already
                        # relies on for a capacity-driven rejection.
                        _note_skip("pp_not_named", req.rid)
                        continue
                    # reconcile_pp_admission_decision's own contract
                    # guarantees told <= this rank's local match, i.e.
                    # len(req.prefix_indices) >= told always holds here.
                    if len(req.prefix_indices) > told:
                        req.prefix_indices = req.prefix_indices[:told]

            try:
                res = adder.add_one_req(
                    req,
                    has_chunked_req=(self.chunked_req is not None),
                    truncation_align_size=self.truncation_align_size,
                )
            except PPScheduleRefused as exc:
                # NAMED, NOT FOLDED IN (the #797 practice for a sibling with a
                # different root): requests admitted EARLIER in this same loop
                # have already taken a persistent `inc_lock_ref` via
                # `_req_inc_lock_ref`, released when the batch completes -- and
                # a refused batch never completes, so those refs leak for the
                # rest of the pass's life. Undoing them needs the exact
                # `IncLockRefResult` each `inc_lock_ref` returned (SWA/Mamba
                # tombstone params; see `_lock_node`), which the adder does not
                # keep, so a release written here would be a guess at the one
                # thing a mismatched release makes worse. Bounded: it takes a
                # genuinely unexecutable geometry to reach this line at all,
                # and reaching it kills the pass rather than continuing on the
                # leaked state. Filed at the site.
                schedule_refusal = exc
                break

            if self.enable_lora:
                running_loras.add(req.lora_id)

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    if self.enable_hierarchical_cache:
                        # Set batch_is_full after making sure there are requests that can be served
                        running_batch.batch_is_full = len(adder.can_run_list) > 0 or (
                            not running_batch.is_empty()
                        )
                    else:
                        running_batch.batch_is_full = True
                # revert matched mamba idx to avoid memory leak, if req is not added.
                # Only free if the slot was freshly allocated in this batch (not
                # pre-existing from a session). Session-held slots have their own
                # lifecycle and freeing them here causes double-free.
                added = len(adder.can_run_list) > 0 and req is adder.can_run_list[-1]
                if not added:
                    # init_next_round_input() may stage deferred Mamba COW/clear
                    # metadata before add_one_req() rejects the request.
                    req.mamba_cow_src_index = None
                    req.mamba_needs_clear = False
                    if req.mamba_pool_idx is not None and not getattr(
                        req, "session", None
                    ):
                        self.tree_cache.req_to_token_pool.mamba_allocator.free(
                            req.mamba_pool_idx.unsqueeze(-1)
                        )
                        req.mamba_pool_idx = None
                _note_skip(f"add_result_{res.name}", req.rid)
                break

        if mamba_allocator is not None:
            mamba_allocator.alloc_group_end()

        if schedule_refusal is not None:
            raise schedule_refusal

        # Update waiting queue
        can_run_list: List[Req] = adder.can_run_list

        # #791 CORE: EVERY NAMED REQUEST, OR NONE OF THEM.
        #
        # The loop above still holds several rank-local vetoes that run before
        # `add_one_req` is ever reached and therefore before the executor can
        # refuse: `batch_is_full` (:6957, off this rank's own
        # `get_num_allocatable_reqs`), the HiCache `prefetch_done` skip
        # (:6969 -- the very race that produced boot instr20), and the LoRA
        # gate (:6945). Each of them is a `continue`/`break`, i.e. a SILENT
        # LOCAL NARROWING of a pass the upstream has already launched. They
        # are correct as rank-local admission control and wrong as an answer
        # to a forwarded schedule, so on that path they become a refusal
        # instead of a smaller batch. Checked here, once, where the pass's
        # final membership is first knowable -- and before the
        # `len(can_run_list) == 0` early return below, which would otherwise
        # swallow the total-narrowing case.
        scheduled_extents = self._pp_scheduled_extents()
        if scheduled_extents:
            admitted_rids = {req.rid for req in can_run_list}
            missing = [rid for rid in scheduled_extents if rid not in admitted_rids]
            if missing:
                raise PPScheduleRefused(
                    f"#791 FORWARDED SCHEDULE UNEXECUTABLE: the decision names "
                    f"{len(scheduled_extents)} request(s) and this rank's "
                    f"admission loop reached only {len(admitted_rids)}; "
                    f"missing rid(s)={','.join(sorted(missing))}. A rank "
                    f"executing a forwarded schedule may not drop a named "
                    f"request -- the upstream's hidden states for it are "
                    f"already on the wire."
                )
            extra = [rid for rid in admitted_rids if rid not in scheduled_extents]
            if extra:
                raise PPScheduleRefused(
                    f"#791 FORWARDED SCHEDULE UNEXECUTABLE: this rank admitted "
                    f"rid(s)={','.join(sorted(extra))}, which the decision does "
                    f"not name. A rank executing a forwarded schedule may not "
                    f"add a request either -- the upstream computed no hidden "
                    f"states for it."
                )

            # #791 CORE: ORDER IS GEOMETRY TOO -- the one divergence every
            # width check on this branch is blind to. Membership is proven
            # identical by the two refusals above, so the permutation this
            # applies is total. See `order_batch_by_schedule`.
            can_run_list = self._pp_order_batch_by_schedule(
                can_run_list, scheduled_extents
            )
            adder.can_run_list = can_run_list

        if len(can_run_list) == 0:
            # #791b-instr22: an empty loop names its skips -- the silent
            # local narrowing, made loud. "loop=clean" means the loop saw
            # every queued request and admitted none WITHOUT any skip
            # firing, which (queue > 0) should be impossible and is itself
            # a finding.
            if _skips:
                self._admission_decline_note = (
                    "loop_skips("
                    + ",".join(
                        f"{k}={v}(first={_skip_first_rid[k][:16]})"
                        for k, v in sorted(_skips.items())
                    )
                    + ")"
                )
            else:
                self._admission_decline_note = "loop=clean"
            return None, running_batch

        can_run_set = set(can_run_list)
        self.waiting_queue = [x for x in self.waiting_queue if x not in can_run_set]

        # #791 PP ADMISSION UNIFORMITY: PP0 publishes this pass's admission
        # decision here; scheduler_pp_mixin.py's _event_loop_pp_body drains
        # `self._pp_admission_last_built_decision`, stamps in the real
        # mb_id (unknown to this method -- get_next_batch_to_run takes no
        # mb_id parameter), and sends it. Built from `can_run_list` AFTER
        # the loop above has already applied any guard clamp, so
        # `len(req.prefix_indices)` here is already the value this rank is
        # really using: `build_pp_admission_decision`'s own guard
        # application is therefore an idempotent re-confirmation, not a
        # second clamp (`prefix_len_for` on an already-clamped candidate
        # returns that same candidate).
        if self.ps.pp_size > 1 and self.ps.pp_rank == 0:
            self._pp_admission_last_built_decision = build_pp_admission_decision(
                0,  # placeholder mb_id; stamped with the real one downstream
                can_run_list,
                pp_size=self.ps.pp_size,
                guard=self._pp_admission_guard,
                # #791 CORE: the ONE production call site, and the only one
                # that must never fall back. A `can_run_list` member with no
                # `extend_range` is a torn-down request, not a missing
                # optimisation -- refuse and name it.
                require_executed_geometry=True,
            )
        if adder.preempt_list:
            for req in adder.preempt_list:
                self._add_request_to_queue(req)

        if adder.new_chunked_req is not None:
            # Update chunked prefill
            assert self.chunked_req is None
            self.chunked_req = adder.new_chunked_req

        if self.chunked_req is not None:
            self.chunked_req.inflight_middle_chunks += 1

        set_time_batch(can_run_list, "set_forward_entry_time")

        # Create a new batch
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            chunked_req=self.chunked_req,
        )

        new_batch.contains_last_prefill_chunk = (
            self.chunked_req is None or len(can_run_list) != 1
        )

        self.max_prefill_bs = max(self.max_prefill_bs, len(can_run_list))
        if self.enable_hierarchical_cache:
            # todo (zhiqiang): disable cuda graph execution if hicache loading triggered
            new_batch.hicache_consumer_index = (
                self.tree_cache.ready_to_load_host_cache()
            )

        new_batch.prepare_for_extend()

        if self.tp_worker.model_runner.prefill_aware_swa:
            for req in can_run_list:
                req.swa_evict_floor = req.extend_range.end

        # Record prefill stats for logging after forward.
        new_batch.prefill_stats = PrefillStats.from_adder(
            adder,
            running_batch.reqs,
            self.enable_priority_scheduling,
            num_pending_tokens=self.load_inquirer._get_num_pending_tokens(
                chunk_deduct=(
                    self.chunked_req.extend_range.length
                    if self.chunked_req is not None
                    else 0
                ),
            ),
        )

        # Mixed-style chunked prefill
        if (
            self.is_mixed_chunk
            and not running_batch.is_empty()
            and not (new_batch.return_logprob or running_batch.return_logprob)
            # mix_with_running cats input_ids but not input_embeds — shapes would mismatch
            and new_batch.input_embeds is None
        ):
            # TODO (lianmin): support return_logprob + mixed chunked prefill
            running_batch.filter_batch()
            if not running_batch.is_empty():
                if self._cross_schedule_mode:
                    self.draft_worker.maybe_switch_rung(running_batch)
                running_batch.prepare_for_decode()
                new_batch.mix_with_running(running_batch)
                new_batch.decoding_reqs = running_batch.reqs
            running_batch = ScheduleBatch(
                reqs=[], batch_is_full=running_batch.batch_is_full
            )
        else:
            new_batch.decoding_reqs = None

        return new_batch, running_batch

    def _can_schedule_lora_req(
        self, req: Req, running_loras: set[Optional[str]]
    ) -> bool:
        """
        Check if a LoRA request can be scheduled.

        This method checks two conditions:
        1. The drainer allows scheduling (based on draining state)
        2. The LoRA adapter can be loaded (either already running or can be added)
        """
        if self.lora_drainer and not self.lora_drainer.can_schedule(req):
            return False

        if req.lora_id in running_loras:
            return True

        if self.enable_lora_overlap_loading:
            # For overlapping loading of LoRA weights with computation, we will load each
            # adapter one at a time, as opposed to loading them in one batch
            return self.lora_overlap_loader.try_overlap_load_lora(
                req.lora_id, running_loras
            )
        else:
            new_lora_set = {req.lora_id} | running_loras
            return self.tp_worker.model_runner.lora_manager.validate_lora_batch(
                new_lora_set
            )

    #: One prefix so a boot log can be grepped for the whole ladder at once.
    _LADDER_PREFIX = "KV-ADMISSION-LADDER"

    def _maybe_spend_admission_relief(self, running_batch: ScheduleBatch) -> int:
        """Decide, group-uniformly, whether the chunked prefill needs relief.

        #679. Split from the ladder so the TRIGGER and the ACTUATORS can be
        tested apart, and so the trigger's one job is visible: read the agreed
        number, compare it against what the next chunk would ask for, and spend
        nothing at all when the pool is comfortable. The comfortable case is
        every iteration on a healthy instance, so it must cost one comparison.

        THE SHORTFALL IS SIZED FROM THE REDUCED VALUE, which makes every rank
        ask its rungs for the same number of tokens. Sizing it from this rank's
        own availability would have the binding rank retract more victims than
        its peers -- #583, one layer up from where #583 was found.
        """
        from sglang.srt.mem_cache.common import admission_relief_ladder_enabled

        if not admission_relief_ladder_enabled():
            return 0
        req = self.chunked_req
        if req is None:
            return 0
        try:
            want = int(self.server_args.chunked_prefill_size or 0)
            if want <= 0:
                return 0
            avail = int(self.uniform_min_avail())
            if avail >= want:
                # The common case, and it must stay cheap: one reduced read,
                # one comparison, no rung entered.
                return 0
            return self._admission_relief_ladder(running_batch, want - avail)
        except Exception as e:  # noqa: BLE001 - relief must never fail a boot
            logger.warning(
                "%s trigger failed (%s); admission proceeds and the park guard "
                "decides unaided",
                self._LADDER_PREFIX,
                e,
            )
            return 0

    def _admission_relief_ladder(
        self, running_batch: ScheduleBatch, need_tokens: int
    ) -> int:
        """Spend relief so an admission need not PARK. Returns tokens freed.

        #679 rung 1-3, built to DESIGN_679_admission_relief_ladder.md. The park
        guard remains the floor of this ladder and the final authority: this
        function only changes how much the pool can fund BEFORE that guard
        decides. It never admits anything and never refuses anything.

        THE ORDER, and why it is this order (see the design note for the
        costing of each rung):

          rung 0  radix eviction -- already spent by the caller, and by
                  alloc_token_slots after it. Not repeated here.
          rung 1  kvso.try_spill -- a BOUNDED, CHOSEN amount (the victim's
                  block-aligned tail overhang) at the cost of host bandwidth
                  and no request's progress. Best rung available.
          rung 2  throttle_before_retract -- frees NOTHING now; lowers inflow
                  so rung 3 does not repeat next round. Placed between the two
                  for exactly the reason the decode-OOM branch places it there.
          rung 3  retract_decode -- the most tokens per call and the loudest:
                  the victim loses all decode progress and re-prefills.

        EXHAUSTION IS A RUNG OUTCOME, NEVER AN ERROR. ``try_spill`` returns
        False when no host region is free -- a reachable state whose bound has
        never been measured under the 5-lane load that produced the crash -- and
        the ladder simply falls through to the next rung. Same for a rung that
        frees less than asked: the ladder continues, the caller re-reads the
        pool, and the park guard has the last word. Nothing here raises.

        GROUP UNIFORMITY. Every decision reads ``uniform_min_avail()``, the
        value the pre-branch reduce published at the top of this iteration
        (scheduler.py's _update_uniform_pool_budget, unconditional and once per
        rank), so no rung takes a collective of its own and no rung can split
        the group. ``need`` is sized from that same reduced value, so every
        rank spills and retracts for the same shortfall. This is the property
        #603 and #583 were paid for; it is not re-derived here, it is reused.

        OFF BY DEFAULT. Without the env flag this returns 0 immediately and the
        caller behaves exactly as c4b88e1923 did.
        """
        from sglang.srt.mem_cache.common import (
            admission_relief_ladder_enabled,
            admission_retraction_enabled,
        )

        if not admission_relief_ladder_enabled():
            return 0
        if running_batch is None or running_batch.is_empty():
            # Nothing to spill and nothing to retract: every rung below acts on
            # the RUNNING batch. An empty one means the pressure is not coming
            # from resident work, so there is nothing this ladder can take.
            return 0

        need = max(0, int(need_tokens))
        if need <= 0:
            return 0

        before = int(self.uniform_min_avail())
        freed_by = []

        # -- rung 1: spill a session's tail to host ---------------------------
        if self.kv_session_offload is not None:
            try:
                if self.kv_session_offload.try_spill(running_batch, need=need):
                    freed_by.append("kvso_spill")
            except Exception as e:  # noqa: BLE001 - a rung must not kill a boot
                logger.warning(
                    "%s rung 1 (kvso spill) failed: %s", self._LADDER_PREFIX, e
                )
            if int(self.uniform_min_avail()) - before >= need:
                self._log_ladder(need, before, freed_by)
                return int(self.uniform_min_avail()) - before

        # -- rung 2: lower inflow so rung 3 does not repeat -------------------
        # Frees nothing. Deliberately spent even when rung 3 is skipped below:
        # the pressure that got us here is inflow, and the throttle is the only
        # rung that addresses that rather than its symptom.
        try:
            if throttle_before_retract(
                self.admission_limiter, running_batch.batch_size()
            ):
                freed_by.append("throttle")
        except Exception as e:  # noqa: BLE001
            logger.warning("%s rung 2 (throttle) failed: %s", self._LADDER_PREFIX, e)

        # -- rung 3: retract decode victims -----------------------------------
        # THE PRECONDITION IS NOT OPTIONAL. retract_decode's loop bound and its
        # last-survivor test both read uniform_avail_floor; handing it a
        # rank-local value is #583 exactly -- ranks enter together and pop
        # DIFFERENT numbers of victims.
        if admission_retraction_enabled():
            try:
                running_batch.uniform_avail_floor = self.uniform_min_avail()
                gained = self._retract_decode_and_requeue(
                    running_batch, kv_full_retract_flag=True
                )
                if gained:
                    freed_by.append(f"retract({gained})")
            except Exception as e:  # noqa: BLE001
                logger.warning("%s rung 3 (retract) failed: %s", self._LADDER_PREFIX, e)

        freed = max(0, int(self.uniform_min_avail()) - before)
        self._log_ladder(need, before, freed_by)
        return freed

    def _log_ladder(self, need: int, before: int, freed_by: list) -> None:
        after = int(self.uniform_min_avail())
        logger.warning(
            "%s asked for %d tokens: group-uniform available %d -> %d (%+d) "
            "via %s. The park guard still decides; this only changed what "
            "there is to decide from.",
            self._LADDER_PREFIX,
            need,
            before,
            after,
            after - before,
            ", ".join(freed_by) if freed_by else "nothing (every rung was spent)",
        )

    def _retract_decode_and_requeue(
        self, batch: ScheduleBatch, *, kv_full_retract_flag: bool
    ) -> int:
        """Retract decode victims and put every one of them back. Returns the
        tokens the pool gained.

        #679 rung 3: EXTRACTED VERBATIM from update_running_batch so the
        admission ladder can reach this actuator without owning a second copy
        of it. That matters more than it looks. The retraction itself is one
        call; what surrounds it is the part that must not drift -- the metrics,
        the new_token_ratio handover, the abort dispatch for requests that
        could not be kept, and above all

            for req in retracted_reqs:
                self._add_request_to_queue(req, is_retracted=True)

        A second implementation that forgot that line would LEAK every victim
        it retracted, which is a worse failure than the crash this ladder
        exists to prevent. One implementation, two call sites, no drift.

        THE CALLER OWNS THE PRECONDITIONS, and they are not optional:
          * ``batch.uniform_avail_floor`` must already be the reduced value --
            it bounds the retraction loop and the last-survivor test, and #583
            is exactly the case where the entry decision was uniform and the
            loop bound was not, so ranks popped DIFFERENT numbers of victims;
          * the decision to call at all must be group-uniform for the same
            reason.
        """
        old_available_tokens = self.token_to_kv_pool_allocator.available_size()
        old_ratio = self.new_token_ratio_tracker.current
        mamba_allocator = getattr(
            self.tree_cache.req_to_token_pool, "mamba_allocator", None
        )
        old_mamba_available = (
            mamba_allocator.available_size() if mamba_allocator is not None else None
        )
        retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(
            self.server_args
        )
        new_available_tokens = self.token_to_kv_pool_allocator.available_size()
        new_token_gained = new_available_tokens - old_available_tokens
        mamba_num_gained = (
            mamba_allocator.available_size() - old_mamba_available
            if mamba_allocator is not None
            else None
        )

        self.metrics_reporter.num_retracted_reqs = len(retracted_reqs)
        if self.metrics_reporter.enable_metrics and len(retracted_reqs) > 0:
            self.metrics_reporter.metrics_collector.increment_retracted_reqs(
                num_retracted_reqs=len(retracted_reqs),
                num_retracted_input_tokens=sum(
                    len(r.origin_input_ids) for r in retracted_reqs
                ),
                num_retracted_output_tokens=sum(
                    len(r.output_ids) for r in retracted_reqs
                ),
            )
        self.new_token_ratio_tracker.current = new_token_ratio
        for req in reqs_to_abort:
            abort_reason: FINISH_ABORT = req.to_finish
            self.ipc_channels.send_to_tokenizer.send_output(
                AbortReq(
                    finished_reason=abort_reason.to_json(),
                    rid=req.rid,
                ),
                req,
            )

        msg_prefix = (
            "KV cache pool is full. Retract requests. "
            if kv_full_retract_flag
            else "Testing retraction. "
        )
        msg_details = f"#retracted_reqs: {len(retracted_reqs)}, #new_tokens_gained: {new_token_gained}"
        if mamba_num_gained is not None:
            msg_details += f", #mamba_num_gained: {mamba_num_gained}"
        if kv_full_retract_flag:
            msg_details += (
                f", #new_token_ratio: {old_ratio:.4f} -> {new_token_ratio:.4f}"
            )
        logger.warning(msg_prefix + msg_details)

        for req in retracted_reqs:
            self._add_request_to_queue(req, is_retracted=True)
        return new_token_gained

    def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch]:
        """Update the current running decoding batch."""
        initial_bs = batch.batch_size()

        batch.filter_batch()
        if batch.is_empty():
            batch.batch_is_full = False
            return batch

        # NOTE(#581): the write-through ack drain moved to the top of
        # `get_next_batch_to_run`, so it runs on every scheduler iteration
        # rather than only on iterations with a running decode batch. Calling
        # it again here would issue a second TP collective per iteration for
        # no additional headroom.

        # #287: one pressure sample per decode round. This is the EARLY half
        # of the throttle -- lowering here, at the water mark, is what keeps
        # the pre-retraction throttle below from ever being reached in the
        # common case; it is also the only place the limit is raised again,
        # and only after --admission-release-hysteresis consecutive samples
        # of genuinely free headroom. Not sampled at all unless the auto
        # controller is armed, so the default path is untouched.
        #
        # RANK-UNIFORM by construction. pool_stats' token_usage would be the
        # obvious source and is the wrong one: it derives from THIS rank's
        # physical available_size, and under uneven DCP the ranks' pools
        # differ, so a per-rank usage gives per-rank verdicts -> divergent
        # admission -> divergent batches -> collective desync. The running
        # batch's held tokens and max_total_num_tokens are both replicated,
        # so this sample is identical on every rank with no collective. It
        # also measures the right thing: tokens held by live requests are the
        # non-reclaimable occupancy, while the radix cache's evictable
        # remainder is not pressure.
        if self.admission_limiter.auto:
            self.admission_limiter.observe(
                replicated_pool_usage(
                    sum(req.seqlen for req in batch.reqs),
                    # Both sides GLOBAL: seqlen is a global length, so the
                    # denominator is the global span, not this rank's shard
                    # of it (#346). Still replicated -- the span is the
                    # min-reduced capacity times a group-wide factor -- so the
                    # no-collective argument above is unaffected.
                    self._global_kv_capacity_tokens(),
                ),
                batch.batch_size(),
            )

        # Check if decode out of memory.
        # kv-session-offload (S1): on OOM, first try to SPILL a session to
        # host (FCFS, youngest-sufficient victim) -- it keeps decoding from
        # host instead of being discarded + re-prefilled -- then re-check.
        # Zero-overhead invariant: in the fits-case this is exactly ONE
        # check_decode_mem call, identical to the stock path; the extra
        # re-check runs only on an actual spill event. If one spill is not
        # enough (or the slot is taken), the stock retraction runs below.
        if self.kv_session_offload is not None:
            # RANK-UNIFORM decode-OOM decision (uneven DCP). check_decode_mem
            # would read the LOCAL per-rank available_size, which differs per
            # rank; near the pool boundary the binding rank flips to OOM while
            # others still fit -> divergent spill/retract -> divergent batch ->
            # collective desync (hang). Instead decide on the iteration's
            # min-reduced available (dcp_min_avail, from the single pre-branch
            # reduce) vs the replicated token demand -> identical on every rank,
            # with NO collective here. Still evict locally for the space (side
            # effect); the decision uses the reduced pre-evict value (in the
            # full-pool spill regime eviction frees ~nothing, so this is exact
            # and always uniform).
            num_tokens_next = batch.new_tokens_required_next_decode()
            evict_from_tree_cache(self.tree_cache, num_tokens_next)
            kv_full_retract_flag = (
                self.kv_session_offload.dcp_min_avail() < num_tokens_next
            )
            if kv_full_retract_flag and self.kv_session_offload.try_spill(batch):
                if batch.is_empty():
                    batch.batch_is_full = False
                    return batch
                # try_spill freed the binding rank's shortfall (need sized from
                # the same reduced available); treat the OOM as resolved. If one
                # spill was not enough the NEXT iteration re-evaluates uniformly
                # (max_spills gates further victims) -> no second collective, no
                # branch-count divergence.
                kv_full_retract_flag = False
        else:
            # #603: the same RANK-UNIFORM decision as the branch above, on the
            # path that runs when session offload is off. `check_decode_mem`
            # compares the LOCAL available_size against the replicated token
            # demand; under uneven DCP the local side differs per rank, so
            # near the pool boundary the binding rank retracts while the
            # others do not -> divergent batch -> divergent branch -> the
            # ranks stop agreeing on which collectives run. The eviction side
            # effect is kept (it is what frees the space); only the COMPARISON
            # moves to the reduced value.
            #
            # Pre-evict reduced value on purpose, exactly as the offload
            # branch documents: in the regime where this decision is close,
            # eviction frees ~nothing, so it is both exact and uniform.
            num_tokens_next = batch.new_tokens_required_next_decode()
            evict_from_tree_cache(self.tree_cache, num_tokens_next)
            kv_full_retract_flag = self.uniform_min_avail() < num_tokens_next
        # #797, EXAMINED AND DELIBERATELY NOT CHANGED. This decision and the
        # loop bound below are RANK-LOCAL on a TP=1/PP=3 boot -- not by
        # oversight, but because `_update_uniform_pool_budget` reduces on
        # `tp_cpu_group`, which has one member per rank there, and says so in
        # its own boot log line: "#788 UNIFORM-FLOOR SCOPE ... With pp_size>1
        # the ranks that must agree are NOT in this reduce group". The stages
        # own different layer counts and therefore different pools, so two
        # stages CAN pop different numbers of decode victims and end a pass
        # with different membership. The same holds for the offload branch
        # above and for `_admission_relief_ladder`'s rung 3, which reaches
        # this same actuator through the same `_retract_decode_and_requeue`.
        #
        # WHAT THEY CANNOT DO is produce #791c's SILENT divergence, and that
        # is why they are not folded into #797:
        #   * `ScheduleBatch._get_decode_retraction_order` sorts on
        #     `len(req.output_ids)`, `-len(req.origin_input_ids)` and
        #     optionally `req.priority` -- all replicated per-request state --
        #     under replicated server args (`retraction_policy`,
        #     `schedule_low_priority_values_first`, `spec_algorithm`), so
        #     every stage computes the SAME preference order over the SAME
        #     request list;
        #   * `retract_decode` pops only from the END of that order, so the
        #     stages' victim sets are NESTED, never disjoint: for a given
        #     victim COUNT the victims are identical;
        #   * a decode batch's row count is a strict function of its request
        #     count (one token per request, or `1 + num_speculative_tokens`),
        #     so different membership always means a different
        #     `forward_batch.input_ids.shape[0]`;
        #   * and `retract_decode` entered with more than one request always
        #     retracts at least one (its `first_iter` do-while), so "entered
        #     on one stage only" is never a zero-victim reorder.
        # A divergence here is therefore ALWAYS a width divergence, which
        # `model_runner.forward`'s `_hs.shape[0] != _want` raises on every
        # time. Loud, attributable, and not the same-width class that made
        # instr15/16/17's ~4000 narrowings compute in silence.
        #
        # It is still a real defect -- a crash, and a running batch that stays
        # divergent afterwards -- and #797's remedy does not fit it: voiding a
        # pass repairs per-pass membership, while a retracted decode victim is
        # a change to LONG-LIVED state that one voided pass cannot undo. The
        # root is the reduce group at `_update_uniform_pool_budget`, and
        # closing it means either widening that group (a new cross-stage
        # synchronisation point every iteration, against the law this whole
        # feature is built on) or giving the decode retraction the same
        # learn-and-carry shape `PPAdmissionCongruenceGuard` has. Either is
        # its own change.
        #
        # #583 (desync site 2): hand the SAME reduced value to the batch, so
        # `retract_decode`'s loop bound and its last-survivor test decide from
        # it too. #603 made the decision to ENTER retraction rank-uniform;
        # without this line the ranks still enter together and then pop
        # DIFFERENT numbers of victims from their local pools -- and a rank
        # that pops down to empty returns `ret = None`, skips `run_batch`
        # (the `if batch:` at the top of the event loop) and goes round to
        # `recv_requests` while the others enter the decode collective. That
        # is the observed 2026-08-05 21:10 stack pair exactly.
        batch.uniform_avail_floor = self.uniform_min_avail()
        if kv_full_retract_flag:
            # #287 THROTTLE BEFORE RETRACT. Retraction still runs -- it is the
            # only thing that frees tokens for the step that is about to
            # start. What the throttle prevents is the loop after it: the
            # slots retraction just freed are handed straight back to the
            # waiting queue on the next prefill pass, and the same pressure
            # then discards the next victim. Lowering the inflow first turns a
            # repeated discard into a single one. A no-op unless the auto
            # controller is armed by --max-running-requests-ceiling.
            throttle_before_retract(self.admission_limiter, batch.batch_size())
        if kv_full_retract_flag or (
            TEST_RETRACT and self.forward_ct % TEST_RETRACT_INTERVAL == 0
        ):
            self._retract_decode_and_requeue(
                batch, kv_full_retract_flag=kv_full_retract_flag
            )
        else:
            self.new_token_ratio_tracker.decay_step()

        if batch.batch_size() < initial_bs:
            batch.batch_is_full = False

        if batch.is_empty():
            return batch

        # T156 stage 3 (cross-algorithm schedule mode): decide the active rung
        # BEFORE prepare_for_decode -- the prep dispatches on
        # batch.spec_algorithm and reserves the incoming rung's KV headroom,
        # and switches are legal only here, at the round boundary after the
        # previous round's commit.
        if self._cross_schedule_mode:
            self.draft_worker.maybe_switch_rung(batch)

        # Update batch tensors
        batch.prepare_for_decode()
        return batch

    def record_batch_in_overlap(self, batch: ScheduleBatch):
        # FIXME(lsyin): hacky way to keep a reference to avoid GPU tensors being freed by torch GC
        # NOTE: More Reliable: record all tensors into the forward stream
        # NOTE: - for all future tensors, we shall always read from future map
        #       - for all non-future tensors (produced only by schedule stream),
        #       we shall keep its reference not being release during all the forwarding pass
        # Snapshot all fields: spec V2 rebinds seq_lens / spec_info mid-forward.
        attr_snapshot = [
            getattr(batch, f.name, None) for f in dataclasses.fields(batch)
        ]
        self.batch_record_ct = (self.batch_record_ct + 1) % 2
        # List (not tuple) so that workers can register additional refs via
        # GenerationBatchResult.extra_keep_alive_refs after forward returns.
        self.batch_record_buf[self.batch_record_ct] = [batch, attr_snapshot]

    @contextmanager
    def _forward_isolation(self, batch: ScheduleBatch, *, overlap: bool):
        """Make SB transactional across one forward (overlap and non-overlap).

        1. Snapshot SB fields so V2's mid-forward mutations (forward_mode /
           input_ids / seq_lens / spec_info / ...) can be undone. V1 / non-spec
           only need sampling_info restored - V1 carries spec_info forward as
           next-iter draft input.
        2. Substitute sampling_info with a forward-only copy (orchestrator=None,
           shares the pre-accumulated penalty buffer) so V2's multiple init_new
           calls don't double-accumulate penalties.
        3. (overlap=True only) Pin (batch, snapshot) into batch_record_buf
           for 2 iters so GPU tensors in the snapshot survive the caching
           allocator past the forward stream. Must run AFTER the sampling_info
           swap so the forward-only copy gets pinned. The non-overlap (sync) path
           runs on a single stream and doesn't allocate batch_record_buf, so it
           passes overlap=False.
        """
        # 1. snapshot
        snapshot_v2_full = not batch.spec_algorithm.is_none()
        sched_snapshot = (
            {f.name: getattr(batch, f.name) for f in dataclasses.fields(batch)}
            if snapshot_v2_full
            else None
        )
        sched_sampling_info = batch.sampling_info

        # 2. sampling_info substitute
        if sched_sampling_info is not None:
            batch.sampling_info = sched_sampling_info.copy_for_forward()

        # 3. pin for 2-iter tensor lifetime (overlap path only)
        if overlap:
            self.record_batch_in_overlap(batch)

        try:
            yield
        finally:
            if snapshot_v2_full:
                for name, value in sched_snapshot.items():
                    setattr(batch, name, value)
            else:
                batch.sampling_info = sched_sampling_info

    @scheduler_nvtx_method("scheduler.run_batch")
    def run_batch(
        self,
        batch: ScheduleBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[GenerationBatchResult, EmbeddingBatchResult]:
        """Run a batch."""
        self.forward_ct += 1
        batch.forward_iter = self.forward_ct

        # Pairing objective (#274 slice D): publish this batch's grain shape
        # for the lane's pairing policy. Read-only for the policy, one tuple
        # store here; None on every default path. Publishing must not alter
        # the batch or its timing beyond this constant cost -- the policy's
        # regression gate is that scheduling with the policy off is
        # byte-identical.
        if self.lane_pairing_signal is not None:
            from sglang.srt.model_executor.lane_pairing import serving_batch_grain

            # is_plain_prefill: EXTEND/MIXED, the scheduler-level prefill
            # batches whose extend_num_tokens is the row count. Everything
            # else at this level is decode-shaped and gets bs * rows_per_seq
            # (the spec worker's verify runs num_draft_tokens rows per
            # sequence below this grain).
            self.lane_pairing_signal.publish(
                serving_batch_grain(
                    batch.forward_mode.is_plain_prefill(),
                    batch.extend_num_tokens,
                    batch.batch_size(),
                    self._lane_pairing_rows_per_seq,
                    self._lane_pairing_sat_rows,
                )
            )

        if self.scripted_scheduler_hook is not None:
            self.scripted_scheduler_hook.on_run_batch(batch)

        # Whether to run the profiler
        self.profiler_manager._profile_batch_predicate(batch)
        if self.forward_sleep_time is not None:
            logger.info(f"Scheduler.run_batch sleep {self.forward_sleep_time}s")
            time.sleep(self.forward_sleep_time)

        # Place holder handling for pd-disagg decode event loop
        if batch.forward_mode.is_prebuilt():
            return self._run_batch_prebuilt(batch)

        # PD prefill: early-send cached prefix KV, overlapping the suffix forward.
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            for req in batch.reqs:
                self.maybe_send_cached_prefix_chunk(req)

        # Run forward
        if self.is_generation:
            if self.enable_overlap:
                # Self-gates on batch.spec_info.future_indices; non-spec_v2
                # no-ops (ForwardBatch.init_new lazily computes the sum).
                self.future_map.resolve_seq_lens_cpu(batch)
                if self._confidence_budget_prepare is not None:
                    self._confidence_budget_prepare(batch, self.future_map)

                with self.forward_stream_ctx:
                    self.forward_stream.wait_stream(self.schedule_stream)
                    # resolve consumes SB staging (prefill_input_ids_cpu /
                    # mix_running_indices). Run OUTSIDE isolation so the
                    # snapshot captures the post-consume state — restoring
                    # post-forward must not un-consume staging.
                    resolve_forward_inputs(batch, self.future_map)

                    with self._forward_isolation(batch, overlap=True):
                        future_indices = batch.req_pool_indices

                        # Spec_v2 fires on_publish mid-worker (between verify and
                        # draft_extend) so schedule prep can overlap with draft_extend.
                        # Non-spec has no later work — scheduler publishes after return.
                        fwd_kwargs = (
                            {
                                "on_publish": partial(
                                    self.future_map.publish, future_indices
                                )
                            }
                            if not batch.spec_algorithm.is_none()
                            else {}
                        )

                        # FIXME: pp is not compatible with overlap
                        batch_result = self.model_worker.forward_batch_generation(
                            batch, **fwd_kwargs
                        )
                        if batch.spec_algorithm.is_none():
                            self.future_map.publish(future_indices, batch.seq_lens + 1)
                        # Park any refs the worker wants kept alive 2 iters
                        # (cross-stream tensor lifetime; pinned in the same
                        # ring slot as the SB attr snapshot).
                        if batch_result.extra_keep_alive_refs:
                            self.batch_record_buf[self.batch_record_ct].extend(
                                batch_result.extra_keep_alive_refs
                            )
                        if self.enable_unified_memory:
                            # Record a `forward_done` event after the forward (before
                            # copy_to_cpu); lazy-compaction `_flush` gates src reuse on
                            # it. Only the unified pool's allocator exposes these hooks.
                            allocator = self.token_to_kv_pool_allocator
                            forward_done = self.device_module.Event()
                            forward_done.record(stream=self.forward_stream)
                            allocator.set_latest_forward_done_event(forward_done)
                            # Write-set classification: hand the allocator this
                            # forward's virtual out_cache_loc as a tensor ref (no GPU work).
                            allocator.set_inflight_forward(
                                forward_done,
                                batch.out_cache_loc,
                            )
                        # FIXME(lsyin): maybe move this to forward_batch_generation
                        batch_result.copy_done = self.device_module.Event()
                        if batch_result.delay_sample_func is None:
                            self._relay_forward_payload(future_indices, batch_result)
                            if _is_hip:
                                # Cross-stream sync costs more than the tiny D2H it
                                # overlaps.
                                batch_result.copy_to_cpu(
                                    return_logprob=batch.return_logprob,
                                    return_hidden_states=batch.return_hidden_states,
                                )
                            else:
                                # Result D2H on copy_stream overlaps the next forward
                                # instead of serializing on forward_stream; it's a leaf
                                # gated by copy_done, so nothing on forward_stream waits.
                                self.copy_stream.wait_stream(self.forward_stream)
                                with self.copy_stream_ctx:
                                    batch_result.copy_to_cpu(
                                        return_logprob=batch.return_logprob,
                                        return_hidden_states=batch.return_hidden_states,
                                    )
                        else:
                            batch_result.future_indices = future_indices

                # Next-iter input_ids relayed via future_map.
                batch.input_ids = None

                if not batch.spec_algorithm.is_none():
                    batch.spec_info = batch_result.next_draft_input
                    batch.spec_info.future_indices = future_indices
            elif self.enable_pdmux and batch.forward_mode.is_split_prefill():
                resolve_forward_inputs(batch, self.future_map)
                batch_result = self.tp_worker.forward_batch_split_prefill(batch)
                self._relay_forward_payload(batch.req_pool_indices, batch_result)
                batch.input_ids = None
            elif not batch.spec_algorithm.is_none():
                # Non-overlap: drive the V2 worker synchronously (no
                # future_map relay / on_publish).
                resolve_forward_inputs(batch, self.future_map)
                with self._forward_isolation(batch, overlap=False):
                    batch_result = self.model_worker.forward_batch_generation(batch)
                # The isolation restore reverted the worker's in-forward SB edits;
                # re-apply what must carry to the next iter.
                batch.spec_info = batch_result.next_draft_input
                if batch_result.new_seq_lens is not None:
                    batch.seq_lens = batch_result.new_seq_lens
                    if batch.seq_lens_cpu is not None:
                        batch.seq_lens_cpu = batch_result.new_seq_lens.to("cpu")
                        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
                batch.input_ids = None  # rebuilt next iter from draft_token
                self.update_cache_from_scheduler(batch, batch_result)
                # Sync D2H so the result processor can read CPU tensors.
                batch_result.copy_done = self.device_module.Event()
                batch_result.copy_to_cpu(
                    return_logprob=batch.return_logprob,
                    return_hidden_states=batch.return_hidden_states,
                )
            else:
                kwargs = (
                    {"pp_proxy_tensors": pp_proxy_tensors}
                    if self.spec_algorithm.is_none()
                    else {}
                )
                resolve_forward_inputs(batch, self.future_map)
                batch_result = self.model_worker.forward_batch_generation(
                    batch, **kwargs
                )
                if batch_result.has_sampled_token_ids:
                    # Non-spec: relay via future_map, gathered next iter.
                    self._relay_forward_payload(batch.req_pool_indices, batch_result)
                    batch.input_ids = None
                self.update_cache_from_scheduler(batch, batch_result)

            # These 2 values are needed for processing the output, but the values can be
            # modified by overlap schedule. So we have to copy them here so that
            # we can use the correct values in output processing.
            if batch.return_logprob:
                batch_result.extend_input_len_per_req = [
                    req.extend_range.length if req.extend_range is not None else 0
                    for req in batch.reqs
                ]
                batch_result.extend_logprob_start_len_per_req = (
                    batch.extend_logprob_start_lens
                )
            else:
                batch_result.extend_input_len_per_req = None
                batch_result.extend_logprob_start_len_per_req = None

            ret = batch_result
        else:  # embedding or reward model
            if self.enable_overlap:
                self.record_batch_in_overlap(batch)
                with self.forward_stream_ctx:
                    self.forward_stream.wait_stream(self.schedule_stream)
                    resolve_forward_inputs(batch, self.future_map)
                    pooler_output = self.tp_worker.forward_batch_embedding(batch)
                    ret = EmbeddingBatchResult(
                        embeddings=pooler_output.embeddings,
                        pooled_hidden_states=pooler_output.pooled_hidden_states,
                    )
                    ret.copy_to_cpu()
            else:
                resolve_forward_inputs(batch, self.future_map)
                pooler_output = self.tp_worker.forward_batch_embedding(batch)
                ret = EmbeddingBatchResult(
                    embeddings=pooler_output.embeddings,
                    pooled_hidden_states=pooler_output.pooled_hidden_states,
                )

        self._maybe_report_active_ranks()

        return ret

    def _dispatch_concurrent_spill(self, spill_batch: ScheduleBatch) -> None:
        """DECOUPLE S4b: issue a due spill tick CONCURRENTLY with the device
        forward that was just enqueued on forward_stream (comm A).

        The device forward is already in flight (run_batch returns after an
        async launch), so issuing the spill forward now on a SECOND stream
        makes their GPU kernels overlap: the device compute fills the SMs while
        the spill lane's PCIe H2D (on _sess_copy_stream, a child of
        spill_stream) drains -- the whole point of decoupling.

        Isolation (verified disjoint, no cross-lane mutable sharing):
          * stream: run_batch is routed onto spill_stream by a temporary swap of
            forward_stream / forward_stream_ctx; `with self.forward_stream_ctx`
            then enqueues the entire spill forward -- and, via current_stream(),
            the _sess_copy_stream fork -- on spill_stream. Restored in finally so
            an exception can never leak the swap into the device lane.
          * communicator: the spill batch carries kv_session_spill_tick=True, so
            the attention backend routes its DCP collectives to comm B (S3);
            the device forward stays on comm A. (Only the DCP collective is
            duplicated today -- the model's TP / MoE all-reduces still share
            their communicators; that shared-comm concurrency is the S4b
            hang-risk the boot measurement exists to falsify.)
          * keep-alive ring: swap in the spill lane's private
            batch_record_buf/ct so the shared 2-slot device ring is not advanced
            twice per iteration (would evict the device snapshot a full
            iteration early).
          * result: committed to the SPILLED request via a depth-1 overlap
            queue -- device tokens -> device reqs, spill tokens -> spilled reqs,
            never mixed.

        Python issues device-then-spill sequentially, so each communicator sees
        a rank-uniform, ordered op stream; every rank runs this identically
        (the stash is set from replicated state)."""
        if spill_batch is None:
            return

        prev_fs = self.forward_stream
        prev_ctx = self.forward_stream_ctx
        self.forward_stream = self.spill_stream
        self.forward_stream_ctx = self.spill_stream_ctx
        # The keep-alive ring is an overlap-only asset (record_batch_in_overlap
        # runs only under overlap). Swap it in only then.
        swap_ring = self.enable_overlap
        if swap_ring:
            prev_brb = self.batch_record_buf
            prev_brc = self.batch_record_ct
            self.batch_record_buf = self._spill_record_buf
            self.batch_record_ct = self._spill_record_ct
        try:
            spill_result = self.run_batch(spill_batch)
            # Delay-sample (spec V2) must run on the spill stream too; a no-op
            # for non-spec (delay_sample_func is None). Kept inside the swap so
            # forward_stream_ctx is still spill_stream.
            if self.is_generation:
                self.launch_batch_sample_if_needed(spill_result, spill_batch)
        finally:
            if swap_ring:
                # Persist the spill ring's advance; restore the device ring.
                self._spill_record_buf = self.batch_record_buf
                self._spill_record_ct = self.batch_record_ct
                self.batch_record_buf = prev_brb
                self.batch_record_ct = prev_brc
            self.forward_stream = prev_fs
            self.forward_stream_ctx = prev_ctx

        if self.enable_overlap:
            # Depth-1 overlap mirror of the device result_queue: process the
            # PREVIOUS spill result while this one runs, so token commit stays
            # off the scheduler's critical path without ever mixing lanes.
            self._spill_result_queue.append((spill_batch.copy(), spill_result))
            if len(self._spill_result_queue) > 1:
                b, r = self._spill_result_queue.popleft()
                self.process_batch_result(b, r)
        else:
            self.process_batch_result(spill_batch, spill_result)

    def _maybe_report_active_ranks(self) -> None:
        if not (
            self.enable_dp_attention and self.server_args.elastic_ep_backend is not None
        ):
            return
        # Get the tensors indicating rank activeness
        tp_active_ranks = self.tp_group.active_ranks.detach().cpu().numpy()
        tp_active_ranks_cpu = self.tp_group.active_ranks_cpu.detach().numpy()
        tp_active_ranks &= tp_active_ranks_cpu
        dp_active_ranks = tp_active_ranks.reshape(self.ps.dp_size, -1).prod(axis=1)
        self.ipc_channels.send_to_tokenizer.send_output(
            ActiveRanksOutput(status=dp_active_ranks.tolist())
        )

    def _relay_forward_payload(
        self, future_indices: torch.Tensor, batch_result: GenerationBatchResult
    ) -> None:
        """Stash this iter's relay payload for next iter's resolve_forward_inputs.
        ngram is skipped: it relays its draft via batch.spec_info, not the FutureMap."""
        if self.spec_algorithm.is_ngram():
            return
        if batch_result.next_draft_input is not None:
            payload = RelayPayload.from_draft_input(batch_result.next_draft_input)
        elif batch_result.has_sampled_token_ids:
            payload = RelayPayload(bonus_tokens=batch_result.next_token_ids)
        else:
            return
        self.future_map.stash(future_indices, payload)

    def launch_batch_sample_if_needed(
        self, batch_result: GenerationBatchResult, cur_batch: ScheduleBatch
    ) -> Union[GenerationBatchResult]:
        # TODO(lsyin): make the delayed sample a default behavior after
        # unifying the forward_batch_generation interface (related to spec V2).
        if batch_result is None or batch_result.delay_sample_func is None:
            return

        with self.forward_stream_ctx:
            self.forward_stream.wait_stream(self.schedule_stream)
            _batch_result = batch_result.delay_sample_func()
            assert _batch_result is batch_result
            # Delay-sample is non-spec only; relays the sampled bonus tokens.
            self._relay_forward_payload(batch_result.future_indices, batch_result)
            batch_result.copy_to_cpu(
                return_logprob=cur_batch.return_logprob,
                return_hidden_states=cur_batch.return_hidden_states,
            )

        # Release the closure and large GPU tensors that are no longer needed.
        # The delay_sample_func closure captures forward_batch (which holds
        # sampling_info with vocab_mask) and logits_output (which holds
        # next_token_logits). Without clearing these, they stay alive via
        # batch_result in result_queue and batch_record_buf until the next
        # iteration, causing a steady VRAM leak with structured output.
        batch_result.delay_sample_func = None
        if batch_result.logits_output is not None:
            batch_result.logits_output.next_token_logits = None

    @scheduler_nvtx_method("scheduler.process_batch_result")
    def process_batch_result(
        self,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
    ):
        self.publish_load_snapshot(force=batch.forward_mode.is_extend())

        # #485 transient census: one strided, read-only driver query, labelled
        # with the load state that produced it. On every default boot this is
        # a module-global boolean test and nothing more.
        if _transient_census.ARMED:
            _transient_census.note(batch.forward_mode.name)

        if batch.forward_mode.is_decode():
            self.batch_result_processor.process_batch_result_decode(batch, result)
        elif batch.forward_mode.is_extend():
            if batch.is_dllm():
                self.process_batch_result_dllm(batch, result)
            elif self.disaggregation_mode == DisaggregationMode.PREFILL:
                self.process_batch_result_disagg_prefill(batch, result)
            else:
                self.batch_result_processor.process_batch_result_prefill(batch, result)
        elif batch.forward_mode.is_prebuilt():
            self.batch_result_processor.process_batch_result_prebuilt(batch)
        elif batch.forward_mode.is_idle():
            self.batch_result_processor.process_batch_result_idle(batch, result)

        self.metrics_reporter.log_batch_result_stats(batch, result)

        # Emit forward pass metrics (every iteration when enabled)
        if self.enable_fpm:
            self.metrics_reporter._emit_forward_pass_metrics(batch, result)

        self._maybe_clear_mm_inputs(batch)
        self.maybe_send_health_check_signal()
        self.metrics_reporter.update_device_timer()

        # #50 campaign debug levers: hash-dump all persistent state and/or
        # hard-reset selected state families after each finished request, so
        # deterministic cross-request state evolution can be diffed and its
        # carrier isolated. No-op unless one of the envs is set. Runs BEFORE
        # the production workspace zeroing below so dumps see the state the
        # request actually left behind.
        if envs.SGLANG_SPEC_STATE_HASH.get() or envs.SGLANG_SPEC_RESET_PROBE.get():
            from sglang.srt.debug_utils.spec_state_hash import (
                maybe_dump_on_request_finish,
            )

            maybe_dump_on_request_finish(self, batch)

        # #50 root fix: restore the flashinfer float-workspace boot contract
        # (first-touch zeros) at request boundaries — the fa2 split-KV
        # kernels read workspace regions the current forward did not write,
        # so residue of the previous request otherwise perturbs the next
        # request's logits in the last bits (request-ordinal-dependent
        # outputs; degenerate attractor under cuda graphs). Proven by the
        # round-11 GPU bisection: zeroing exactly _float_workspace_buffer
        # per finished request flattens the ordinal sequence at the natural
        # run-1 value. Enqueued on the worker forward stream when one exists
        # (overlap scheduling): stream order makes the memset land between
        # forwards, never inside one; without a forward stream the scheduler
        # thread's current stream IS the forward stream.
        if envs.SGLANG_FLASHINFER_ZERO_WORKSPACE_PER_REQUEST.get() and any(
            req.finished() for req in batch.reqs
        ):
            # Only when the flashinfer backend module is actually loaded
            # (otherwise no workspaces exist, and importing it would pull in
            # the flashinfer package on boots that do not use it).
            fi_mod = sys.modules.get("sglang.srt.layers.attention.flashinfer_backend")
            if fi_mod is not None:
                forward_stream = getattr(self.tp_worker, "forward_stream", None)
                if forward_stream is not None:
                    with torch.cuda.stream(forward_stream):
                        fi_mod.zero_flashinfer_workspaces()
                else:
                    fi_mod.zero_flashinfer_workspaces()

    def maybe_send_health_check_signal(self):
        if self.return_health_check_ipcs:
            # Return some signal for the health check.
            # This is used to prevent the health check signal being blocked by long context prefill.
            # However, one minor issue is that this code path does not check the status of detokenizer manager.
            self.ipc_channels.send_to_tokenizer.send_output(
                HealthCheckOutput(
                    http_worker_ipc=self.return_health_check_ipcs.popleft()
                )
            )

    def add_external_corpus(
        self, recv_req: AddExternalCorpusReqInput
    ) -> Optional[AddExternalCorpusReqOutput]:
        if self.external_corpus_manager is None:
            return AddExternalCorpusReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        return self.external_corpus_manager.add(recv_req)

    def remove_external_corpus(
        self, recv_req: RemoveExternalCorpusReqInput
    ) -> RemoveExternalCorpusReqOutput:
        if self.external_corpus_manager is None:
            return RemoveExternalCorpusReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        return self.external_corpus_manager.remove(recv_req)

    def list_external_corpora(
        self, recv_req: ListExternalCorporaReqInput
    ) -> ListExternalCorporaReqOutput:
        if self.external_corpus_manager is None:
            return ListExternalCorporaReqOutput(
                success=False,
                message="Ngram speculative decoding is not enabled.",
            )
        return self.external_corpus_manager.list(recv_req)

    def clear_hicache_storage_wrapped(self, recv_req: ClearHiCacheReqInput):
        if self.enable_hierarchical_cache:
            self.tree_cache.clear_storage_backend()
            logger.info("Hierarchical cache cleared successfully!")
            if_success = True
        else:
            logging.warning("Hierarchical cache is not enabled.")
            if_success = False
        return ClearHiCacheReqOutput(success=if_success)

    def on_idle(self):
        """Idle housekeeping: guard, check, metrics, reset, sleep."""
        if not self.is_fully_idle():
            # #547: no batch to run, but work is queued somewhere (waiting
            # queue, grammar, disagg, hicache drain). That is the loaded path
            # as far as the idle poll is concerned -- back to the zero-poll rung.
            if self.idle_sleeper is not None:
                self.idle_sleeper.reset()
            return

        if self.enable_unified_memory:
            try:
                self.token_to_kv_pool_allocator.flush_opportunistic()
            except Exception:
                pass

        # memory leak check (skipped for hisparse — pool counters intentionally
        # diverge during host-backup, see _get_swa_token_info clamp).
        if not self.enable_hisparse:
            has_leak, messages = self.invariant_checker._check_all_pools(
                self.pool_stats_observer.get_pool_stats(),
            )
            if has_leak:
                self.invariant_checker._report_leak("pool", "\n".join(messages))
            self.invariant_checker._check_req_pool()

        # tree cache sanity check
        self.invariant_checker._check_tree_cache()

        # metrics every 30s
        self.metrics_reporter._maybe_log_idle_metrics()

        # kv event publishing
        self.kv_events_publisher.publish_kv_events()

        # reset token ratio
        self.new_token_ratio_tracker.reset()

        # reset device timer window so idle time isn't counted
        self.metrics_reporter.reset_device_timer_window()

        # Publish the idle state so /get_loads and DP balancing do not see stale load.
        self.publish_load_snapshot(force=True)

        # sleep until next event
        self.maybe_sleep_on_idle()

    def is_fully_idle(self, for_health_check=False) -> bool:
        # Health check piggybacks on running requests in process_output.
        # Only running_batch + waiting_queue guarantee active GPU processing;
        # disagg queues (bootstrap/prealloc/transfer) may have items without
        # any request actually running on GPU — e.g. stuck handshake, full
        # KV cache, or stalled transfer — so they can't carry health info.
        # Batch running status
        idle = (
            self.running_batch.is_empty()
            and self.chunked_req is None
            and not self.dllm_manager.any_staging_reqs()
            and (self.last_batch is None or self.last_batch.is_empty())
            and (not self.enable_overlap or len(self.result_queue) == 0)
            and self._pp_microbatches_drained()
            # kv-session-offload: a host-spilled session is still running
            # (its req slot + mamba state are live; pool checks would flag
            # a "leak" and health would look idle mid-decode).
            and (
                self.kv_session_offload is None
                or not self.kv_session_offload.has_spilled()
            )
        )

        # Waiting queues: waiting + bootstrapping + preallocation + kv transfer (decode)
        idle &= len(self.waiting_queue) == 0

        if not for_health_check:
            # Grammar queue and prefill inflight queue may not produce batch
            # results instantly, but they still indicate the server is not idle.
            idle &= len(self.grammar_manager.grammar_queue) == 0
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                idle &= len(self.disagg_prefill_inflight_queue) == 0
                idle &= len(self.disagg_prefill_bootstrap_queue.queue) == 0

            if self.disaggregation_mode == DisaggregationMode.DECODE:
                idle &= len(self.disagg_decode_prealloc_queue.queue) == 0
                idle &= len(self.disagg_decode_prealloc_queue.retracted_queue) == 0
                idle &= len(self.disagg_decode_transfer_queue.queue) == 0
                if self.decode_offload_manager is not None:
                    idle &= len(self.decode_offload_manager.ongoing_offload) == 0

            # HiSparse: staging requests transitioning prefill -> decode
            if self.enable_hisparse:
                idle &= not self.hisparse_coordinator.has_ongoing_staging()

            # HiCache: in-flight async ops (GPU↔Host↔L3) must drain before
            # destructive operations like attach/detach/flush_cache.
            if self.enable_hierarchical_cache:
                tc = self.tree_cache
                idle &= len(tc.ongoing_write_through) == 0
                idle &= len(tc.ongoing_load_back) == 0
                if tc.enable_storage:
                    idle &= len(tc.ongoing_prefetch) == 0
                    idle &= len(tc.ongoing_backup) == 0

        return idle

    def _pp_microbatches_drained(self) -> bool:
        if self.ps.pp_size == 1:
            return True
        return all(x.is_empty() for x in self.running_mbs) and all(
            mb is None or mb.is_empty() for mb in self.mbs
        )

    def _admin_world_rank(self) -> int:
        """Flat world rank of this scheduler process, for admin replies.

        ``self.ps.tp_rank``, NOT ``self.tp_rank``: the Scheduler keeps its
        parallel identity on the ParallelState wrapper, and an earlier cut of
        another feature used the non-existent attribute and raised on every
        tick. Reduces to tp_rank without a pipeline.
        """
        try:
            return int(self.ps.pp_rank) * int(self.ps.tp_size) + int(self.ps.tp_rank)
        except Exception:  # pragma: no cover - an admin reply may not raise
            return -1

    def attach_hicache_storage_wrapped(
        self, recv_req: AttachHiCacheStorageReqInput
    ) -> AttachHiCacheStorageReqOutput:
        # #545: stamp the rank on EVERY return path. Done by wrapping rather
        # than at each `return`, because there are several and a new one added
        # later would silently ship unstamped (-1) and break the rollback's
        # ability to name stranded ranks.
        out = self._attach_hicache_storage_impl(recv_req)
        out.rank = self._admin_world_rank()
        return out

    def _attach_hicache_storage_impl(
        self, recv_req: AttachHiCacheStorageReqInput
    ) -> AttachHiCacheStorageReqOutput:
        if not self.enable_hierarchical_cache:
            return AttachHiCacheStorageReqOutput(
                success=False, message="Hierarchical cache is not enabled."
            )

        if not self.is_fully_idle():
            return AttachHiCacheStorageReqOutput(
                success=False,
                message=(
                    "Reject attach: scheduler is not idle. "
                    f"#queue-req={len(self.waiting_queue)} "
                    f"#running-req={len(self.running_batch.reqs)}"
                ),
            )

        if not hasattr(self.tree_cache, "attach_storage_backend"):
            return AttachHiCacheStorageReqOutput(
                success=False,
                message="Current tree_cache implementation does not support dynamic attach.",
            )

        try:
            ok, msg = self.tree_cache.attach_storage_backend(
                storage_backend=recv_req.hicache_storage_backend,
                storage_backend_extra_config_json=recv_req.hicache_storage_backend_extra_config_json,
                served_model_name=self.server_args.served_model_name,
                hicache_storage_prefetch_policy=recv_req.hicache_storage_prefetch_policy,
                hicache_write_policy=recv_req.hicache_write_policy,
            )
        except Exception as e:
            logger.exception("Attach HiCache storage backend failed with exception.")
            return AttachHiCacheStorageReqOutput(success=False, message=str(e))
        if ok:
            self.enable_hicache_storage = True
            hicache_fields = {
                "hicache_storage_backend": recv_req.hicache_storage_backend
            }
            if recv_req.hicache_storage_backend_extra_config_json is not None:
                hicache_fields["hicache_storage_backend_extra_config"] = (
                    recv_req.hicache_storage_backend_extra_config_json
                )
            if recv_req.hicache_storage_prefetch_policy is not None:
                hicache_fields["hicache_storage_prefetch_policy"] = (
                    recv_req.hicache_storage_prefetch_policy
                )
            if recv_req.hicache_write_policy is not None:
                hicache_fields["hicache_write_policy"] = recv_req.hicache_write_policy
            self.server_args.override("scheduler.attach_hicache", **hicache_fields)
            logger.info(
                f"Attached HiCache storage backend: {recv_req.hicache_storage_backend}"
            )
        return AttachHiCacheStorageReqOutput(success=ok, message=msg)

    def detach_hicache_storage_wrapped(
        self, recv_req: DetachHiCacheStorageReqInput
    ) -> DetachHiCacheStorageReqOutput:
        out = self._detach_hicache_storage_impl(recv_req)
        out.rank = self._admin_world_rank()
        return out

    def _detach_hicache_storage_impl(
        self, recv_req: DetachHiCacheStorageReqInput
    ) -> DetachHiCacheStorageReqOutput:
        if not self.enable_hierarchical_cache:
            return DetachHiCacheStorageReqOutput(
                success=False, message="Hierarchical cache is not enabled."
            )

        if not self.is_fully_idle():
            return DetachHiCacheStorageReqOutput(
                success=False,
                message=(
                    "Reject detach: scheduler is not idle. "
                    f"#queue-req={len(self.waiting_queue)} "
                    f"#running-req={len(self.running_batch.reqs)}"
                ),
            )

        if not hasattr(self.tree_cache, "detach_storage_backend"):
            return DetachHiCacheStorageReqOutput(
                success=False,
                message="Current tree_cache implementation does not support dynamic detach.",
            )

        # Idempotent detach: even if scheduler thinks storage is disabled, we still
        # attempt best-effort cleanup in tree_cache (it may have leftover state).
        try:
            ok, msg = self.tree_cache.detach_storage_backend()
        except Exception as e:
            logger.exception("Detach HiCache storage backend failed with exception.")
            return DetachHiCacheStorageReqOutput(success=False, message=str(e))

        if ok or (not self.enable_hicache_storage):
            # Treat "already disabled / nothing to do" as success for idempotence.
            self.enable_hicache_storage = False
            self.server_args.override(
                "scheduler.detach_hicache",
                hicache_storage_backend=None,
                hicache_storage_backend_extra_config=None,
            )
            logger.info("Detached HiCache storage backend.")
            return DetachHiCacheStorageReqOutput(
                success=True, message=msg or "HiCache storage backend is detached."
            )

        return DetachHiCacheStorageReqOutput(success=False, message=msg)

    def resize_hicache_storage_wrapped(
        self, recv_req: ResizeHiCacheStorageReqInput
    ) -> ResizeHiCacheStorageReqOutput:
        if not self.enable_hierarchical_cache:
            return ResizeHiCacheStorageReqOutput(
                success=False, message="Hierarchical cache is not enabled."
            )

        if recv_req.max_size_gb is None and recv_req.min_free_gb is None:
            return ResizeHiCacheStorageReqOutput(
                success=False,
                message="Nothing to resize: pass max_size_gb and/or min_free_gb.",
            )
        if recv_req.max_size_gb is not None and recv_req.max_size_gb <= 0:
            return ResizeHiCacheStorageReqOutput(
                success=False,
                message=(
                    f"max_size_gb must be > 0 (got {recv_req.max_size_gb}). "
                    f"Use DELETE /hicache/storage-backend to remove the tier."
                ),
            )
        if recv_req.min_free_gb is not None and recv_req.min_free_gb < 0:
            return ResizeHiCacheStorageReqOutput(
                success=False,
                message=f"min_free_gb must be >= 0 (got {recv_req.min_free_gb}).",
            )

        if not hasattr(self.tree_cache, "resize_storage_backend"):
            return ResizeHiCacheStorageReqOutput(
                success=False,
                message="Current tree_cache implementation does not support resize.",
            )

        # Unlike attach/detach, resize starts no threads and rebuilds no pools: it
        # only re-caps the backend's own evictor, which locks against the backup and
        # prefetch threads itself. So it deliberately does NOT demand an idle
        # scheduler -- an operator can re-cap a busy server. A shrink still runs its
        # unlinks inline here, delaying the next batch by that much.
        max_size_bytes = (
            None if recv_req.max_size_gb is None else int(recv_req.max_size_gb * _GIB)
        )
        min_free_bytes = (
            None if recv_req.min_free_gb is None else int(recv_req.min_free_gb * _GIB)
        )

        try:
            ok, msg, stats = self.tree_cache.resize_storage_backend(
                max_size_bytes=max_size_bytes, min_free_bytes=min_free_bytes
            )
        except Exception as e:
            logger.exception("Resize HiCache storage backend failed with exception.")
            return ResizeHiCacheStorageReqOutput(success=False, message=str(e))

        if ok:
            logger.info(
                f"Resized HiCache storage backend: max_size_gb={recv_req.max_size_gb} "
                f"min_free_gb={recv_req.min_free_gb} -> {stats}"
            )
        return ResizeHiCacheStorageReqOutput(success=ok, message=msg, stats=stats)

    def flush_cache(self, empty_cache: bool = True):
        """Flush memory pools (e.g., KV cache, Mamba cache) and optionally empty device allocator cache."""
        if self.is_fully_idle():
            self.cur_batch_for_debug = None
            self.last_batch = None
            self.tree_cache.reset()
            self.req_to_token_pool.clear()
            self.token_to_kv_pool_allocator.clear()
            if envs.SGLANG_FLUSH_ZERO_KV.get():
                # Default part of the flush (opt-out env): the post-flush
                # state must equal a fresh boot, whose pools are torch.zeros.
                self._flush_zero_kv_buffers()
            self.grammar_manager.clear()
            self.metrics_reporter.reset_metrics()

            if self.draft_worker:
                self.draft_worker.clear_cache_pool()

            if empty_cache:
                current_platform.empty_cache()
            if envs.SGLANG_FLUSH_SCRUB_FREE_MEMORY.get():
                self._flush_scrub_free_memory()
            # Per-DP-group leader logs once: ranks within a DP group are
            # state-synchronous, but DP groups may diverge.
            if self.metrics_reporter.is_stats_logging_rank:
                logger.info("Cache flushed successfully!")
            success = True
        else:
            # NAME THE CLAUSE, not just the verdict (#631/#656). The wedged
            # instance of 2026-08-10 refused this flush while the metrics
            # reported 0 running and 0 queued, and the two counters printed
            # here could not say why -- so the remedy the flip's own abandon
            # message advertises looked broken for no visible reason, and
            # only a reboot recovered. is_fully_idle() is a conjunction of
            # nine clauses; which one is false is the whole diagnosis.
            logging.warning(
                f"Cache not flushed because there are pending requests. "
                f"#queue-req: {len(self.waiting_queue)}, "
                f"#running-req: {len(self.running_batch.reqs)}, "
                f"not-idle because: {', '.join(self.idle_blockers()) or 'unknown'}"
            )
            success = False
        return success

    def idle_blockers(self) -> List[str]:
        """Which clauses of :meth:`is_fully_idle` are currently false.

        Diagnostic only -- no caller changes behaviour on it. It exists
        because "not idle" with every visible counter at zero is a state
        this server can reach and could not previously explain.
        """
        blockers: List[str] = []

        def _check(name: str, ok: bool) -> None:
            if not ok:
                blockers.append(name)

        _check("running_batch", self.running_batch.is_empty())
        _check("chunked_req", self.chunked_req is None)
        _check("dllm_staging", not self.dllm_manager.any_staging_reqs())
        _check(
            "last_batch",
            self.last_batch is None or self.last_batch.is_empty(),
        )
        _check(
            "overlap_result_queue",
            not self.enable_overlap or len(self.result_queue) == 0,
        )
        _check("pp_microbatches", self._pp_microbatches_drained())
        _check(
            "kv_session_offload_spilled",
            self.kv_session_offload is None
            or not self.kv_session_offload.has_spilled(),
        )
        _check("waiting_queue", len(self.waiting_queue) == 0)
        _check("grammar_queue", len(self.grammar_manager.grammar_queue) == 0)
        return blockers

    def handle_session_handover(
        self, recv_req: SessionHandoverReqInput
    ) -> SessionHandoverReqOutput:
        """#261 control plane: live session handover (export / commit /
        abort on the source, verify_import on the destination). Runs on the
        scheduler thread between iterations, so the snapshot is atomic with
        respect to the radix tree; no collective is issued anywhere."""
        if self.session_handover_runtime is None:
            from sglang.srt.managers.session_handover import SessionHandoverRuntime

            self.session_handover_runtime = SessionHandoverRuntime(self)
        return self.session_handover_runtime.handle(recv_req)

    def handle_session_checkpoint(
        self, recv_req: SessionCheckpointReqInput
    ) -> SessionCheckpointReqOutput:
        """#410 control plane: session checkpoint / branch / rewind.

        Runs on the scheduler thread between iterations, so the snapshot,
        the lock reference and the session splice are all atomic with
        respect to the radix tree. Rank-local; no collective is issued.
        """
        if self.session_checkpoint_runtime is None:
            from sglang.srt.managers.session_checkpoint import (
                SessionCheckpointRuntime,
            )

            self.session_checkpoint_runtime = SessionCheckpointRuntime(self)
        return self.session_checkpoint_runtime.handle(recv_req)

    def handle_kv_reshard(self, recv_req: KvReshardReqInput) -> KvReshardReqOutput:
        """#297 control plane: arm a phase-boundary KV reshard.

        Arming is a replicated call (the request reaches every TP scheduler
        through the broadcast pipe) and does no collective work itself -- the
        move commits later, at a consensus boundary where every rank is armed
        and fully idle. Delivery skew across ranks is legal and absorbed by
        the runtime's MIN-semantics on the armed flag.
        """
        if self.server_args.kv_reshard_vectors is None:
            return KvReshardReqOutput(
                success=False,
                message=(
                    "--kv-reshard-vectors is not set; the pool has no fitted "
                    "ceiling to reshard within. Declare the vector set at "
                    "boot."
                ),
            )
        try:
            if self.kv_reshard_runtime is None:
                from sglang.srt.managers.kv_reshard import build_kv_reshard_runtime

                self.kv_reshard_runtime = build_kv_reshard_runtime(self)
            ok, msg = self.kv_reshard_runtime.arm(
                tuple(recv_req.target_vector), source="rpc"
            )
        except Exception as e:
            # Arming performs no collective and moves no byte, so reporting
            # the failure instead of crashing is safe -- and every rank
            # computes the same verdict from the same replicated input.
            logger.warning("KV-RESHARD arm failed: %s", e)
            return KvReshardReqOutput(success=False, message=str(e))
        return KvReshardReqOutput(success=ok, message=msg)

    def handle_phase_flip(self, recv_req: PhaseFlipReqInput) -> PhaseFlipReqOutput:
        """#631 control plane: arm a phase flip (mirror of handle_kv_reshard).

        Replicated call through the broadcast pipe; arming does no
        collective work -- the flip commits at a consensus boundary where
        every rank is armed and quiescent. Delivery skew is absorbed by
        the runtime's MIN-semantics on the armed flag. Routed through
        arm_phase_flip so the abort deferral window activates atomically
        with the arm (pin 4)."""
        # An INTERNALLY generated request (the automatic phase policy) is
        # answered with None, never with a PhaseFlipReqOutput. The reply
        # path ends at _Communicator.handle_recv, which appends to
        # _result_values -- an attribute that only exists while a caller
        # is awaiting that RPC. The policy is not a caller: it synthesised
        # this request inside the scheduler, so there is no awaiting
        # future, _result_values is None, and answering raises
        # AttributeError in the TokenizerManager's handle_loop. That
        # exception is fatal to the tokenizer and takes the whole server
        # with it (measured 2026-08-08: three consecutive boots died
        # seconds after health, each one immediately after the policy's
        # first arm). The outcome is logged by arm_phase_flip either way,
        # so suppressing the reply loses nothing.
        internal = bool(getattr(recv_req, "internal", False))
        if not self.server_args.enable_phase_flip:
            if internal:
                return None
            return PhaseFlipReqOutput(
                success=False,
                message=(
                    "--enable-phase-flip is not set; this server has no "
                    "secondary stack to flip to. Enable it at boot."
                ),
            )
        try:
            ok, msg = self.arm_phase_flip(
                recv_req.direction, source=recv_req.source or "rpc"
            )
        except Exception as e:
            # Arming performs no collective and moves no byte; every rank
            # computes the same verdict from the same replicated input.
            logger.warning("PHASE-FLIP arm failed: %s", e)
            _note_policy_arm_outcome(self, recv_req.direction, False, str(e))
            if internal:
                return None
            return PhaseFlipReqOutput(success=False, message=str(e))
        _note_policy_arm_outcome(self, recv_req.direction, ok, msg)
        if internal:
            return None
        return PhaseFlipReqOutput(success=ok, message=msg)

    def handle_vram_budget(self, recv_req: VramBudgetReqInput) -> VramBudgetReqOutput:
        """#330 control plane: dial a card's VRAM budget or query the state.

        The request reaches every TP scheduler through the broadcast pipe;
        budget mutation is a replicated, collective-free call (the physical
        commit happens later, at a consensus boundary where every rank is
        idle). Below-floor requests are rejected here with exact numbers.
        """
        if not self.server_args.enable_vram_dial:
            return VramBudgetReqOutput(
                success=False,
                message=(
                    "--enable-vram-dial is not set; the KV pools are not "
                    "VMM-backed, so there is no releasable tail. Boot with "
                    "the flag to use the dial."
                ),
            )
        try:
            if self.kv_capacity_runtime is None:
                from sglang.srt.managers.vram_dial import build_kv_capacity_runtime

                self.kv_capacity_runtime = build_kv_capacity_runtime(self)
            rt = self.kv_capacity_runtime
            if recv_req.query:
                return VramBudgetReqOutput(success=True, message="", state=rt.status())
            ok, msg = rt.apply_budget_request(
                device=recv_req.device,
                budget_mib=recv_req.budget_mib,
                release_mib=recv_req.release_mib,
                release_fraction=recv_req.release_fraction,
            )
        except Exception as e:
            logger.warning("VRAM-DIAL request failed: %s", e)
            return VramBudgetReqOutput(success=False, message=str(e))
        return VramBudgetReqOutput(
            success=ok, message=msg, state=self.kv_capacity_runtime.status()
        )

    def _flush_zero_kv_buffers(self):
        """Zero the attention KV data buffers during an idle flush (default,
        SGLANG_FLUSH_ZERO_KV=0 opts out), so a flushed server matches a
        freshly booted one bit-for-bit even if some kernel folds residual
        bytes beyond the valid sequence region into its result.

        #631: with the phase flip there are TWO KV layouts and, since the
        backing became exclusive, at most one of them holds physical pages.
        Two consequences, both load-bearing:

          * an UNBACKED pool must be skipped. Writing into it is a write to
            unmapped VA -- cudaErrorIllegalAddress, which kills every rank.
            Measured: a flush_cache issued in the TP decode phase died here,
            because this method zeroed the scheduler's pool and the
            scheduler's pool is the PP stack's, released while TP serves.
          * the pool that IS backed must still be zeroed, and in the TP
            phase that is the flip stack's, which the scheduler's allocator
            does not reach. Zeroing only what the allocator can see would
            leave the active layout un-zeroed and quietly break the
            bit-for-bit property this exists for.

        So the set is "every KV pool that currently holds pages", not "the
        scheduler's pool".
        """
        from sglang.srt.mem_cache.memory_pool import zero_kv_data_buffers

        zeroed = 0
        skipped = 0
        for pool in self._kv_pools_for_flush():
            if not getattr(pool, "backing_is_resident", True):
                skipped += 1
                continue
            zeroed += zero_kv_data_buffers(pool)
        current_platform.synchronize()
        logger.info(
            "flush: zeroed %d KV data buffers (%d unbacked layout(s) skipped)",
            zeroed,
            skipped,
        )

    def _kv_pools_for_flush(self):
        """The KV pools a flush may touch: the scheduler's, plus the phase
        flip's TP stack when one was built. Residency is checked by the
        caller -- this only enumerates."""
        pools = [self.token_to_kv_pool_allocator.get_kvcache()]
        stacks = getattr(self, "phase_flip_stacks", None)
        if stacks is not None and getattr(stacks, "tp_worker", None) is not None:
            flip_pool = stacks.tp_worker.model_runner.token_to_kv_pool
            if flip_pool is not None and all(flip_pool is not p for p in pools):
                pools.append(flip_pool)
        return pools

    def _flush_scrub_free_memory(self):
        """SGLANG_FLUSH_SCRUB_FREE_MEMORY debug lever: after empty_cache,
        claim as much of the free device memory as possible, zero it and
        release it again. Allocator-recycled pages then read as zeros — the
        same content a freshly booted process sees on its first-touch pages
        — which discriminates (and works around) kernels whose results
        depend on residual bytes in uninitialized activation scratch.
        Best effort; idle-time only, cost irrelevant.
        """
        if not current_platform.is_cuda():
            return
        device = self.tp_worker.model_runner.device
        scrubbed = 0
        buffers = []
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
            chunk = int(free_bytes * 0.95)
            min_chunk = 256 << 20
            while chunk >= min_chunk:
                try:
                    # torch.zeros allocates AND memsets the pages.
                    buffers.append(torch.zeros(chunk, dtype=torch.uint8, device=device))
                    scrubbed += chunk
                except torch.OutOfMemoryError:
                    chunk //= 2
        finally:
            buffers.clear()
            current_platform.empty_cache()
            current_platform.synchronize()
        logger.info(
            "SGLANG_FLUSH_SCRUB_FREE_MEMORY: scrubbed %.2f GiB of free memory",
            scrubbed / (1 << 30),
        )

    def _dual_group_lane_tick(self) -> None:
        """Multi-group runtime (#274): the two-class scheduler's grain
        boundary.  Rank-local by contract (lanes have no communicator); a
        no-op on every default path and on every rank without lanes.

        SERIAL mode (slice B): run one lane step inline. The serving group
        pays the whole step in wall time -- that is the +50 % price slice C
        exists to undercut.

        CONCURRENT mode (slice C): the lane runs on its own thread and its
        own high-priority stream, so this method never executes a forward.
        It is the point where the two CLASSES meet, and it does three things
        at the natural grain (an iteration boundary is a decode-step grain;
        chunked prefill hits it once per chunk):

        1. PROTECTED CLASS FIRST: if the lane has work that has not been
           submitted yet, yield briefly so the lane's kernels are enqueued
           before the scavenger's. Bounded by
           --dual-group-lane-admission-ms, measured per occurrence.
        2. SCAVENGER GETS THE IDLE BYTES: if the lane has been idle past the
           amortization threshold, lend it its configured segment.
        3. Never block on the lane's completion -- the serving group is
           work-conserving, so it goes on to build its own batch either way.
        """
        if not self.dual_group_lanes:
            return
        self._lane_share_sample()
        for lane in self.dual_group_lanes:
            if not lane.concurrent:
                if lane.has_work:
                    try:
                        lane.tick()
                    except Exception:
                        logger.exception(
                            "dual-group lane %d tick failed; dropping the active job.",
                            lane.lane_id,
                        )
                        # Releases the pool slots too: with one request slot
                        # per lane runner, dropping without freeing makes
                        # every later job fail in alloc_req_slots.
                        lane.drop_active()
                continue
            self._dual_group_admit(lane)

    def _lane_share_sample(self) -> None:
        """Feed one sample to the online card-equivalent estimator.

        Called at the same grain boundary as the two-class scheduler, which
        is the only place that sees BOTH classes' counters on one clock.

        The rung id is the controller state a window was measured under.
        Slice D1 builds no controller, so it is constant here -- the argument
        exists because DESIGN_201 addendum 12 (4) requires the bookkeeping to
        be in place BEFORE the first measurement, not retrofitted after one.
        """
        meter = self.lane_share_meter
        if meter is None:
            return
        # One float compare on the overwhelming majority of iterations. The
        # meter would return None anyway until its window is up, but BUILDING
        # the samples for it would not be free, and this call sits directly in
        # front of the serving group's batch launch -- everything spent here
        # is spent by the serving group (DESIGN_121 §12.7).
        now = time.perf_counter()
        if now < self._lane_share_next_t:
            return
        self._lane_share_next_t = now + meter.window_s * 0.25
        from sglang.srt.model_executor.lane_share import ClassSample

        samples = [
            ClassSample(
                "serving",
                {
                    "decode_tokens": self.metrics_reporter.gen_tokens_total,
                    "prefill_tokens": self.metrics_reporter.prefill_tokens_total,
                },
            )
        ]
        for lane in self.dual_group_lanes:
            # The device counters ride along with the work counters, from the
            # same instant: reading them from two different calls would put a
            # window boundary between the numerator and the denominator of the
            # occupancy this window reports (#284).
            clock = getattr(lane, "device_clock", None)
            samples.append(
                ClassSample(
                    f"lane{lane.lane_id}",
                    dict(lane.work_total),
                    device=None if clock is None else clock.snapshot().to_counters(),
                )
            )
        try:
            win = meter.observe(now, samples, rung=self._lane_rung())
        except Exception:
            logger.exception("lane share meter failed; disabling it for this boot.")
            self.lane_share_meter = None
            return
        if win is not None:
            self.metrics_reporter.log_lane_share(win)

    def _lane_rung(self) -> str:
        """Identity of the controller state the current window runs under."""
        return "static"

    def _dual_group_admit(self, lane) -> None:
        """One grain-boundary admission decision for a concurrent lane."""
        budget_ms = float(
            getattr(self.server_args, "dual_group_lane_admission_ms", 0.0) or 0.0
        )
        if lane.has_work and budget_ms > 0 and not lane._submitted.is_set():
            t0 = time.perf_counter()
            # Waiting on the event RELEASES the GIL, which is the actual
            # mechanism: the lane thread needs the interpreter to issue its
            # launches, and the scheduler thread holding it is the only way
            # the scavenger could starve the protected class.
            lane._submitted.wait(timeout=budget_ms / 1000.0)
            lane.note_admission((time.perf_counter() - t0) * 1000.0)
        if lane.lending is not None:
            lane.lending.maybe_lend()

    def get_internal_state(self, recv_req: GetInternalStateReq):
        ret = dict(vars(get_server_args()))  # vars returns a ref to obj.__dict__
        ret["last_gen_throughput"] = self.metrics_reporter.last_gen_throughput
        ret["memory_usage"] = {
            "weight": round(self.tp_worker.model_runner.weight_load_mem_usage, 2),
            "kvcache": round(
                self.token_to_kv_pool_allocator.get_kvcache().mem_usage, 2
            ),
            "token_capacity": int(self.max_total_num_tokens),
            "graph": round(self.tp_worker.model_runner.graph_mem_usage, 2),
        }
        # #287: the effective figure is the limiter's floating value. Without
        # a ceiling the limiter holds max_running_requests, so this key keeps
        # reporting exactly what it reported before.
        ret["effective_max_running_requests_per_dp"] = self.admission_limiter.current
        ret["admission_limiter"] = self.admission_limiter.snapshot()

        if (
            not self.spec_algorithm.is_none()
            and self.metrics_reporter.spec_total_num_forward_ct > 0
        ):
            ret["avg_spec_accept_length"] = (
                self.metrics_reporter.spec_total_num_accept_tokens
                / self.metrics_reporter.spec_total_num_forward_ct
            )

        # Round 7b posten 0: the serving group's per-position acceptance curve,
        # in the same shape the lane's policy reports, so the two can be read
        # next to each other. Present only when the probe env is set.
        from sglang.srt.speculative import accept_position_probe

        if accept_position_probe.probe_enabled():
            ret["spec_accept_positions"] = accept_position_probe.snapshot()

        if RECORD_STEP_TIME:
            ret["step_time_dict"] = self.metrics_reporter.step_time_dict

        if self.spec_algorithm.is_dspark() and self.draft_worker is not None:
            info_record = self.draft_worker.dump_info_records()
            if info_record is not None:
                ret["dspark_info_record"] = info_record

        # Multi-group runtime (#274): lane state + timings (rank 0 carries
        # the lanes; other ranks report an empty list).
        if self.dual_group_lanes:
            ret["dual_group_lanes"] = [lane.stats() for lane in self.dual_group_lanes]
        if self.lane_share_meter is not None:
            ret["lane_share"] = self.lane_share_meter.snapshot()
            # The estimator's own inputs, so an external instrument can
            # difference the SAME counters over its own window and the two
            # can only differ in the windowing -- which is the thing under
            # test, and would be unfalsifiable if each side counted its own way.
            ret["lane_share_counters"] = {
                "serving": {
                    "decode_tokens": self.metrics_reporter.gen_tokens_total,
                    "prefill_tokens": self.metrics_reporter.prefill_tokens_total,
                },
                "t": time.perf_counter(),
            }

        # This field is not serializable.
        ret.pop("model_config", None)

        return GetInternalStateReqOutput(internal_state=msgspec_to_builtins(ret))

    def set_internal_state(self, recv_req: SetInternalStateReq):
        server_args_dict = recv_req.server_args
        args_allow_update = set(
            [
                "pp_max_micro_batch_size",
                "speculative_accept_threshold_single",
                "speculative_accept_threshold_acc",
                "dspark_force_budget_frac",
                "dspark_clear_info_records",
                # Multi-group runtime (#274): enqueue a lane generation job
                # ({"lane_id": 0, "input_ids": [...], "max_new_tokens": N,
                # "repeat": K}). A command, not a server arg; ranks without
                # the addressed lane treat it as a no-op success.
                "dual_group_lane_prefill",
                # Pairing objective (#274 slice D): flip the lane's pairing
                # policy at runtime, so a policy-on and a policy-off arm come
                # from ONE boot (same floors, same captures) instead of
                # carrying boot-to-boot variance. A command on the lane-local
                # policy object; ranks without lanes no-op successfully.
                "dual_group_lane_pairing",
                # #287: move THIS group's floating admission limit. A command
                # on the lane-local limiter, not a server arg -- the ceiling
                # the pools were built for stays where it is.
                "effective_max_running_requests",
                # #665-F1: the measured decode-contention fraction the flip
                # threshold is solved against. Runtime-settable for the same
                # reason as dual_group_lane_pairing above -- it lets the
                # one-sided and the measured threshold be compared from ONE
                # boot, on the same memory vector, the same KV token vector
                # and the same corridor, instead of carrying boot-to-boot
                # variance into the comparison. It changes only how a
                # threshold is COMPUTED; it moves no memory and reshapes no
                # pool, which is why it is safe to move at runtime while the
                # budgets around it are not.
                "phase_policy_decode_contention",
            ]
        )

        if_success = True
        for k, v in server_args_dict.items():
            if k not in args_allow_update:
                logging.warning(f"Updating {k} is not supported.")
                if_success = False
                break
            elif k == "effective_max_running_requests":
                try:
                    self.admission_limiter.set_limit(v)
                except AdmissionLimitError as e:
                    logging.warning(f"Updating {k} to {v} is rejected: {e}")
                    if_success = False
                    break
            elif k == "pp_max_micro_batch_size" and (
                v > self.max_running_requests // self.ps.pp_size or v < 1
            ):
                logging.warning(
                    f"Updating {k} to {v} is rejected because it is out of the valid range [1, {self.max_running_requests // self.ps.pp_size}]."
                )
                if_success = False
                break
            elif k == "dspark_force_budget_frac":
                if not self.spec_algorithm.is_dspark() or not hasattr(
                    self.draft_worker, "set_dspark_forced_budget_frac"
                ):
                    logging.warning(
                        "dspark_force_budget_frac requires a DSpark draft worker."
                    )
                    if_success = False
                    break
                if v is not None and not (0.0 < float(v) <= 1.0):
                    logging.warning(
                        f"dspark_force_budget_frac must be in (0, 1] or null, got {v}."
                    )
                    if_success = False
                    break
            elif k == "dspark_clear_info_records":
                if not self.spec_algorithm.is_dspark() or not hasattr(
                    self.draft_worker, "clear_info_records"
                ):
                    logging.warning(
                        "dspark_clear_info_records requires a DSpark draft worker."
                    )
                    if_success = False
                    break
            elif k == "phase_policy_decode_contention":
                if self.phase_policy_state is None:
                    logging.warning(
                        "phase_policy_decode_contention requires the phase "
                        "policy to be enabled (--phase-flip-policy auto)."
                    )
                    if_success = False
                    break
                from sglang.srt.managers.phase_policy import (
                    PhasePolicyError,
                    with_decode_contention,
                )

                try:
                    with_decode_contention(self.phase_policy_cfg, v)
                except PhasePolicyError as e:
                    logging.warning(f"Updating {k} to {v!r} is rejected: {e}")
                    if_success = False
                    break

        if if_success:
            if (
                not self.spec_algorithm.is_none()
                and self.metrics_reporter.spec_total_num_forward_ct > 0
            ):
                avg_spec_accept_length = (
                    self.metrics_reporter.spec_total_num_accept_tokens
                    / self.metrics_reporter.spec_total_num_forward_ct
                )
                logger.info(f"{avg_spec_accept_length=}")
            self.metrics_reporter.spec_total_num_accept_tokens = (
                self.metrics_reporter.spec_total_num_forward_ct
            ) = 0
            # DSpark control keys are worker commands, not server args; route
            # them to the draft worker and keep them out of the override.
            remaining = dict(server_args_dict)
            frac = remaining.pop("dspark_force_budget_frac", None)
            if "dspark_force_budget_frac" in server_args_dict:
                self.draft_worker.set_dspark_forced_budget_frac(
                    None if frac is None else float(frac)
                )
            if remaining.pop("dspark_clear_info_records", None):
                self.draft_worker.clear_info_records()
            # #287: already applied to the lane-local limiter in the
            # validation loop above (that is where it can still be rejected);
            # keep it out of the server-args override.
            if remaining.pop("effective_max_running_requests", None) is not None:
                logger.info(
                    "Admission limit set to %d (ceiling %d).",
                    self.admission_limiter.current,
                    self.admission_limiter.ceiling,
                )
            # #665-F1: recompute the threshold, not a pool. Logged with the
            # whole ladder because the failure this guards against is a TOP
            # rung no prompt can reach, which is invisible if only one rung
            # is printed.
            if "phase_policy_decode_contention" in server_args_dict:
                from sglang.srt.managers.phase_policy import (
                    effective_flip_threshold,
                    with_decode_contention,
                )

                self.phase_policy_cfg = with_decode_contention(
                    self.phase_policy_cfg,
                    remaining.pop("phase_policy_decode_contention"),
                )
                logger.warning(
                    "PHASE-POLICY decode contention set to %g; N ladder by "
                    "decoding reqs now %s",
                    self.phase_policy_cfg.decode_contention,
                    [
                        effective_flip_threshold(self.phase_policy_cfg, b)
                        for b in range(5)
                    ],
                )
            # Multi-group runtime (#274): lane job -- a command, not a server
            # arg. Only the rank carrying the addressed lane enqueues; every
            # other rank no-ops successfully (the control message is
            # broadcast to all TP schedulers).
            # Pairing objective (#274 slice D): runtime A/B flip of the
            # pairing policy. Popped before the generic override so it never
            # lands in get_server_args() as a fake server arg.
            if "dual_group_lane_pairing" in remaining:
                pairing_on = bool(remaining.pop("dual_group_lane_pairing"))
                flipped = 0
                for lane in self.dual_group_lanes:
                    if lane.pairing_policy is not None:
                        lane.pairing_policy.enabled = pairing_on
                        flipped += 1
                if flipped:
                    logger.info(
                        "dual-group lane pairing policy set to %s on %d lane(s).",
                        pairing_on,
                        flipped,
                    )
            lane_job = remaining.pop("dual_group_lane_prefill", None)
            if lane_job:
                lane_id = int(lane_job.get("lane_id", 0))
                for lane in self.dual_group_lanes:
                    if lane.lane_id == lane_id:
                        for _ in range(int(lane_job.get("repeat", 1))):
                            lane.enqueue(lane_job)
                        logger.info(
                            "dual-group lane %d: %d job(s) enqueued "
                            "(input_len=%d, max_new_tokens=%s).",
                            lane_id,
                            int(lane_job.get("repeat", 1)),
                            len(lane_job.get("input_ids") or []),
                            lane_job.get("max_new_tokens"),
                        )
            if remaining:
                get_server_args().override(source="update_server_args", **remaining)
            logger.info(f"Global server args updated! {get_server_args()=}")

        server_args = dict(vars(get_server_args()))
        # This field is not serializable.
        server_args.pop("model_config", None)
        return SetInternalStateReqOutput(
            updated=if_success,
            server_args=msgspec_to_builtins(server_args),
        )

    def save_remote_model(self, **kwargs):
        self.weight_updater.save_remote_model(kwargs)

    def save_sharded_model(self, **kwargs):
        self.weight_updater.save_sharded_model(kwargs)

    def handle_rpc_request(self, recv_req: RpcReqInput):
        # Handle RPC requests
        logger.info(
            f"handle_rpc_request: {recv_req.method}, param: {recv_req.parameters}"
        )

        success = True
        exec = None
        try:
            func = getattr(self, recv_req.method)
            if recv_req.parameters is not None:
                func(**recv_req.parameters)
            else:
                func()
        except Exception as e:
            success = False
            exec = e
            logger.error(f"Failed to call rpc {recv_req.method}: {str(e)}")

        barrier()
        return RpcReqOutput(success=success, message="" if not exec else str(exec))

    def _pending_prefill_tokens(self, inflight=None) -> int:
        """Prompt tokens ADMITTED BUT NOT YET COMPUTED (#631 defect N).

        ``inflight`` (#713) is the batch of requests that have just been pulled
        off the wire and have NOT yet reached ``waiting_queue``. Passing it is
        what the flip policy must do; every other caller leaves it None and
        gets the pre-#713 number unchanged, because that number is the #363
        observer's quantity and the denominator the break-even N is expressed
        in, and moving it would move a different rule.

        WHY THIS PARAMETER EXISTS. ``recv_requests`` evaluates the phase policy
        BEFORE the requests it just received are queued
        (``scheduler_components/request_receiver.py:104-129``, then
        ``scheduler.py:4089``). So on an idle box the policy asked "is there
        prefill work?" of a queue that had not been told yet, read 0, and
        ``_layout_admits("pp")`` early-falsed on its first line -- refusing a
        flip whose other two terms both held. Measured cost: 31.64 s to first
        token for a ten-token prompt. The rule was right; its input was stale.

        This used to be ``sum(len(req.origin_input_ids) for req in
        self.waiting_queue)`` -- the NOT-YET-ADMITTED queue -- while the
        comment at its use site said "admitted but not yet computed". The
        two disagreed, and the disagreement pinned the instance in the
        wrong layout.

        A long prompt under CHUNKED PREFILL does not sit in the waiting
        queue: it hangs off ``self.chunked_req`` while it is filled a
        chunk per round. So for the whole duration of exactly the work PP
        exists to do, the policy read 0 pending prefill and the TP->PP
        rule (``pending > N``) could not fire. Measured 2026-08-09
        03:36-03:39Z, POLICY=auto with 8k/32k prompts arriving every 5s:
        the acceptance ran its ENTIRE 186 s mixed phase in the TP layout,
        one single policy decision in the whole run, and the layout only
        corrected itself on the idle return.

        The fill boundary is ``extend_range.end``, which is what the
        scheduler's own chunked-prefill code uses
        (``_compute_chunked_req_next_prompt_token``); the remainder behind
        it is admitted, uncomputed, and is precisely the work that would
        be repaid by being in the prefill layout.

        Evaluated on the request-origin rank only (see
        maybe_arm_phase_policy), so this needs no cross-rank replication.
        """
        queued = list(self.waiting_queue)
        pending = sum(len(req.origin_input_ids) for req in queued)
        # #731: THE TERMS BELOW MUST NOT RE-BILL WHAT THE QUEUE ALREADY DID.
        #
        # The resident term further down and this one are two different sets,
        # and nothing kept a request out of both. A cutover could leave one
        # request resident AND queued (the carry re-homed it without consuming
        # the queue entry), and the same prompt was then counted twice:
        # measured 2026-08-17, 51,369 -> 102,307 tokens across one cutover,
        # within rounding of exactly 2x. The inflated backlog drove the flip
        # policy past its threshold -- six cutovers, nothing served.
        #
        # The state fix is in the carry (it now consumes the queue entry). This
        # is the counter's half: even if some other path re-introduces the
        # overlap, the number stays honest.
        #
        # DE-DUPLICATION IS AT THE INTERSECTION ONLY, deliberately. A blanket
        # "count each rid once anywhere" would also swallow a FUTURE legitimate
        # double-booking -- a request genuinely holding budget in two places is
        # a real state that a future reader may need to see, and hiding it
        # behind a global dedup would make that class silent the way this one
        # was. So exactly one overlap is excluded, and only this one.
        _queued_ids = {id(req) for req in queued}
        chunked = getattr(self, "chunked_req", None)
        if chunked is not None:
            rng = getattr(chunked, "extend_range", None)
            filled = int(rng.end) if rng is not None else 0
            pending += max(0, len(chunked.origin_input_ids) - filled)
        pending += _arriving_prefill_tokens(inflight, _queued_ids)
        # #713 (a): RESIDENT-BUT-UNPREFILLED. The three terms above see a
        # request in the waiting queue, in the chunked slot, or in the recv
        # batch -- and NOWHERE ELSE. A request that has been ADMITTED has left
        # the waiting queue, and if it is not the chunked_req it is invisible,
        # so the policy reads 0 prefill pending while holding exactly that
        # work. The arm states the contradiction itself:
        #
        #   IDLE-LOCKED: ... (1 REQ RESIDENT, 0 TOK PREFILL PENDING)
        #
        # Measured 2026-08-17 06:53:59.344 (specimen D2): admitted inside the
        # :59->:01 PP window with ~1.7 s of PP left, invisible at the :01 arm,
        # first token not until :04.879 -- a full cycle late.
        #
        # SAME EXTENT LOGIC AS THE CHUNKED TERM, deliberately: origin_input_ids
        # minus the filled prefix. No new notion of progress is introduced.
        #
        # UNKNOWN PROGRESS COUNTS AS ZERO, and that direction is chosen. An
        # over-count would keep pending above 0 forever, pull and hold the
        # policy toward PP permanently and starve decode -- bounded only by the
        # SLO cap. Under-counting merely restores today's behaviour for that
        # request. So a missing extend_range is treated as fully prefilled.
        try:
            running = getattr(self, "running_batch", None)
            for req in list(getattr(running, "reqs", None) or ()):
                if chunked is not None and req is chunked:
                    continue  # already priced by the chunked term above
                if id(req) in _queued_ids:
                    continue  # #731: the waiting-queue term already billed it
                rng = getattr(req, "extend_range", None)
                if rng is None:
                    continue
                total = len(getattr(req, "origin_input_ids", ()) or ())
                pending += max(0, total - int(rng.end))
        except Exception:  # noqa: BLE001 - an observation must not break a round
            pass
        return pending

    def maybe_arm_phase_policy(self, inflight_reqs=None):
        """#631: evaluate the automatic phase policy on the intake rank.

        ``inflight_reqs`` (#713) is the ``recv_reqs`` batch this evaluation is
        riding in. It is REQUIRED for a correct verdict on an idle box: this
        hook runs before those requests are queued, so without it the policy
        reads an empty queue and refuses to flip toward the very work that
        just woke it. Optional in the signature only so a caller that does not
        have the batch degrades to the pre-#713 reading rather than failing.

        Returns a ``PhaseFlipReqInput`` to put on the request stream, or
        None. The AUTOMATIC path is then byte-for-byte the manual one --
        same request type, same chain, same wake, same drain-park-meet --
        and the only difference is who originates the arm: this policy on
        rank 0, rather than an HTTP caller. That equivalence with a
        mechanism proven on metal (~1.2 s cutovers, reproduced all
        session) is the strongest argument the design has.

        WHY A MESSAGE, when a message caused two deadlocks. Because THE
        MESSAGE IS THE WAKEUP. Ranks 1..n-1 spend idle time BLOCKED in
        the chain recv; forwarding the arm is precisely what unblocks
        them. A message-free variant was built and measured (boot 10):
        every rank reached an identical verdict and armed itself with
        zero messages -- and it wedged solid, because rank 0 was idle,
        therefore instantly parked, and entered the BLOCKING flip
        reduction, which stops it forwarding the batch its peers are
        blocked on. arms=3, cutovers=0, 0 % GPU, /generate dead at 45 s.
        Removing the channel removed the synchronisation.

        The hazard was never the channel; it was the ORDER. See
        DELIVERY-BEFORE-BLOCK in scheduler_pp_mixin.

        Only the request-ORIGIN rank evaluates, so exactly one arm enters
        the chain. Consulting every rank was measured to give 1/2/3 arms
        on PP0/PP1/PP2 (each stage injected its own and forwarded it on),
        a 12765-line census flood and a self-kill -- see the origin guard
        in request_receiver.
        """
        cfg = getattr(self, "phase_policy_cfg", None)
        if cfg is None or not cfg.enabled:
            return None
        runtime = self.phase_flip_runtime
        if runtime is None:
            # No round has run yet, so there is no layout to flip from.
            return None
        if runtime.pending is not None:
            # A flip is already armed and waiting for its consensus
            # boundary; re-arming it would only restart the park clock.
            return None

        from sglang.srt.layers.dcp.phase_flip_plan import PP_TO_TP
        from sglang.srt.managers.phase_policy import (
            IDLE_LOCKED as POLICY_IDLE_LOCKED,
        )
        from sglang.srt.managers.phase_policy import (
            PhasePolicyDecision,
            PhasePolicyInputs,
            decide,
            note_flip_armed,
            observe_idle,
        )

        # THE RESIDENT SET, NOT self.running_batch (#631 J.1, THIRD
        # occurrence -- found by the audit this feature's own handoff
        # commissioned). This hook runs inside recv_requests(), i.e. once
        # per MICROBATCH SLOT under event_loop_pp, immediately after that
        # slot's rebind of running_batch/last_batch. Reading running_batch
        # here therefore counts whichever slot happens to be bound, not
        # the rank's resident decode set.
        #
        # It is load-bearing for the decision, not decoration: the PP->TP
        # rule is "pending <= N AND running_bs > 0", so a request decoding
        # in slot 1 while the hook fires for an empty slot 0 reads
        # running_bs=0 and the flip is not armed. That is the same
        # arming-condition-cannot-hold shape as the quiescence defects,
        # arriving from the other side, and it makes the flip depend on
        # WHICH SLOT the hook samples rather than on the load.
        from sglang.srt.managers.phase_flip_resident_carry import (
            harvest_resident_batches,
        )

        # DEFECT M containment. ``harvest_resident_batches`` now refuses a
        # resident set it cannot identify (a ``reqs`` that is not a request
        # list, or a length above max_running_requests) instead of
        # returning a number the rest of this function would act on.
        #
        # It is caught HERE, and only here, because this call site is a
        # POLICY OBSERVATION: its output decides whether to arm a flip, and
        # declining to arm is always a safe answer. The instance keeps
        # serving in its current layout. The same refusal raised on the
        # CUTOVER path is deliberately not caught -- there, proceeding
        # would allocate against the corrupted set, which is the thing
        # that killed the 21:47Z run.
        #
        # Loud, once per occurrence, with the exception text carrying the
        # offending object's type: defect M appeared once in one boot, so
        # the log line is the only instrument that will catch it again.
        from sglang.srt.managers.phase_flip_resident_carry import (
            ResidentCarryError,
            describe_resident_slots,
            repair_duplicate_resident_reqs,
        )

        def _running_bs() -> int:
            return sum(
                len(getattr(b, "reqs", []) or [])
                for b in harvest_resident_batches(self)
            )

        try:
            running_bs = _running_bs()
        except ResidentCarryError as exc:
            # #631 DEFECT R. Refusing was NOT containment, it was the
            # deadlock: under strict purity only a flip to TP drains the
            # resident set, so declining to evaluate the flip policy
            # blocked the one action that clears the condition being
            # detected -- 1115 flips before, zero after, forever.
            #
            # So repair once, then re-ask. The repair only removes Req
            # entries that are duplicated INSIDE one batch, which is
            # unambiguously wrong at any count; if the set is corrupt some
            # other way the retry raises again and we still decline, but
            # now with the full slot row in the log instead of a bare
            # count, because two boots were spent attributing that count.
            logger.error(
                "PHASE-POLICY resident set is corrupted (%s). Slots: %s",
                exc,
                describe_resident_slots(self),
            )
            try:
                repaired = repair_duplicate_resident_reqs(self)
                running_bs = _running_bs()
            except ResidentCarryError as exc2:
                logger.error(
                    "PHASE-POLICY refusing to evaluate the flip policy this "
                    "round: the resident set is still corrupted after repair "
                    "(%s). Not arming; the instance keeps serving in its "
                    "current phase.",
                    exc2,
                )
                return
            logger.error(
                "PHASE-POLICY resident set repaired (%d duplicate entrie(s) "
                "removed); evaluating the flip policy with running_bs=%d.",
                repaired,
                running_bs,
            )
        # #713: ONE reading, used by the verdict AND by the message that
        # reports it. Calling the accessor twice inside one constructor is how
        # a refusal could name a number the simulation never saw.
        _pending_now = self._pending_prefill_tokens(inflight_reqs)
        inp = PhasePolicyInputs(
            phase=runtime.phase,
            # The same quantity the #363 observer reads, and the one the
            # break-even N is denominated in: prompt tokens admitted but
            # not yet computed.
            pending_prefill_tokens=_pending_now,
            running_bs=int(running_bs or 0),
            now=time.perf_counter(),
            # getattr, because this gate is driven in tests by scheduler
            # STAND-INS that carry only the fields the policy reads. A
            # stand-in without the observation has not observed anything, and
            # "not observed" must mean "do not arm on it" -- the pre-change
            # behaviour -- rather than an AttributeError in the arming path.
            # #689 FORMATION INPUTS. ready_carriers is the PARKED count, not
            # running_bs: with #677 phase-1 parking live, carriers PP cannot
            # decode are discounted from the admission cap, so running_bs
            # reads 0 exactly when the window is fullest. Both terms are
            # replicated -- the parked set is reconciled from the same
            # resident batch on every rank, and the queue is the replicated
            # waiting queue.
            ready_carriers=int(
                getattr(getattr(self, "parked_decode_set", None), "resident_count", 0)
                or 0
            ),
            queue_nonempty=bool(len(getattr(self, "waiting_queue", ()) or ())),
            # #708: the RANK-UNIFORM availability, so the BOTH-BLOCKED decline
            # names its binding resource from a measurement. Group MIN via the
            # existing accessor, never this rank's local pool -- every field on
            # PhasePolicyInputs is replicated by contract, and a local value
            # would make the decline rank-dependent (#616g). None when it
            # cannot be read, which the policy reports as "not measured"
            # instead of guessing.
            # getattr, because this gate is driven in tests by scheduler
            # STAND-INS that carry only the fields the policy reads -- the
            # same trap that broke _idle_locked_inputs and the both-blocked
            # relief earlier in this series. A stand-in without the probe has
            # measured nothing, and 'not measured' is a state the policy
            # already reports honestly.
            kv_available_tokens=getattr(self, "_uniform_kv_available", lambda: None)(),
            **dict(
                zip(
                    ("nothing_can_run", "target_can_admit"),
                    getattr(self, "_idle_locked_inputs", lambda *_: (False, False))(
                        int(running_bs or 0), _pending_now
                    ),
                )
            ),
        )
        state = self.phase_policy_state
        observe_idle(state, inp)
        decision = decide(cfg, state, inp)
        # #631: DO NOT ARM a flip that cannot become ready.
        #
        # This used to decline EVERY pp_to_tp flip with anything resident,
        # because a carried request had no draft state: the readiness
        # predicate then held the flip until nothing was resident, and
        # under sustained decode something always is, so the flip parked
        # for the full deadline and abandoned EVERY time (05:21:16Z, and
        # the abandon path itself faulted). That pinned the instance in PP
        # at 16.8 tok/s against the 113 tok/s TP+MTP does.
        #
        # The draft-state bootstrap removes the cause
        # (managers/phase_flip_draft_bootstrap.py), so a resident request
        # is no longer a reason to decline. What survives is the structural
        # residue: if the armed draft worker exposes no KV pool, the
        # cutover cannot bootstrap anything and the old park-and-abandon
        # would be back -- so keep declining in exactly that case.
        #
        # Still decided rank-locally, and still safe for the same reason:
        # only the REQUEST-ORIGIN rank evaluates the policy and the arm is
        # broadcast from it. A refusal inside PhaseFlipRuntime.arm would be
        # a different thing and would risk diverging epochs, corpse H.
        from sglang.srt.managers.phase_flip_runtime import (
            _flip_can_bootstrap_draft,
        )

        if (
            decision.wants_flip
            and decision.direction == PP_TO_TP
            and not self.flip_spec_algorithm.is_none()
            and inp.running_bs > 0
            and not _flip_can_bootstrap_draft(self)
        ):
            decision = PhasePolicyDecision(
                None,
                f"speculating TP phase and {inp.running_bs} request(s) "
                f"resident, and the armed draft worker exposes no KV pool "
                f"to bootstrap them into: arming would park until the "
                f"deadline and abandon",
            )
        if not decision.wants_flip:
            # #631 defect N: a policy that DECLINES used to be silent, so
            # "the layout is wrong under load" was indistinguishable from
            # "the hook never ran". One throttled line makes the standing
            # reason readable from the same log as everything else -- the
            # same bet as the withhold reason and the quiescence reason,
            # both of which named a defect in a single boot.
            #
            # THROTTLE ON A NUMBER-FREE KEY. The reason string carries live
            # quantities ("min dwell: 13.4s since last flip"), so keying on
            # the string itself made every call look like a NEW reason and
            # the throttle never engaged -- three identical lines inside one
            # second, measured on the very boot that introduced it. A
            # 12765-line log flood has already cost this feature a self-kill
            # once (see the origin guard in request_receiver), so the key is
            # the reason with its digits removed: the SHAPE of the hold, not
            # its instantaneous value.
            now_mono = inp.now
            last = getattr(self, "_phase_policy_last_log", 0.0)
            prev = getattr(self, "_phase_policy_last_reason", None)
            key = _POLICY_REASON_DIGITS.sub("#", decision.reason)
            if key != prev or now_mono - last >= 10.0:
                self._phase_policy_last_log = now_mono
                self._phase_policy_last_reason = key
                logger.info(
                    "PHASE-POLICY holding in %s: %s (pending prefill %d tok, "
                    "running bs %d)",
                    inp.phase,
                    decision.reason,
                    inp.pending_prefill_tokens,
                    inp.running_bs,
                )
            # getattr, for the third time in this file and for the same
            # reason: the policy gate is driven in tests by scheduler
            # STAND-INS carrying only the fields the policy reads. A stand-in
            # without the relief hook must decline exactly as before, not raise
            # AttributeError inside the arming path.
            getattr(self, "_apply_both_blocked_relief", lambda *_: None)(decision, inp)
            return None
        # #688 FUNDING COMPOSITION. An idle-locked arm that then cannot fund
        # its seam has moved the zero-GPU window one stage right instead of
        # removing it (live specimen 09:43:11Z: staging 1706 MiB needed
        # against 1635 spendable -- 71 MiB short, with 364884 cached rows
        # sitting there). Recorded on the runtime so the funding path can see
        # WHY this flip was armed; cleared by the runtime once it is read.
        rt_for_funding = getattr(self, "phase_flip_runtime", None)
        if rt_for_funding is not None:
            rt_for_funding.armed_idle_locked = bool(
                (decision.reason or "").startswith(POLICY_IDLE_LOCKED)
            )
        # #688 ANTI-OSCILLATION AS A RUNTIME INVARIANT, not a test premise.
        #
        # The hermetic test asserted that the target runs after the flip by
        # FEEDING that assumption in by hand, so it could not have caught the
        # 10:24 ping-pong -- the assumption was the bug. The property has to
        # be checked where it can actually be false: on metal, after the flip,
        # against what the target layout then does. If the target does not
        # build a batch within a few rounds, the admissibility verdict that
        # armed this flip was WRONG, and the inputs that produced it are what
        # a reader needs.
        self._arm_watch = {
            "direction": decision.direction,
            "reason": (decision.reason or "")[:120],
            "running_bs": int(inp.running_bs),
            "pending": int(inp.pending_prefill_tokens),
            "nothing_can_run": bool(getattr(inp, "nothing_can_run", False)),
            "target_can_admit": bool(getattr(inp, "target_can_admit", False)),
            "ready_carriers": int(getattr(inp, "ready_carriers", 0) or 0),
            "rounds": 0,
            # THE DISCRIMINATOR. An arm can only be judged once the layout it
            # asked for actually arrived; until the phase changes, the cutover
            # has not committed and any silence belongs to the FUNDING, not to
            # the verdict.
            "phase_at_arm": getattr(self, "phase_flip_active_stack", None),
        }
        note_flip_armed(state, decision, inp.now)
        logger.warning(
            "PHASE-POLICY arming %s: %s", decision.direction, decision.reason
        )
        # internal=True: nobody awaits a reply, and answering would kill
        # the TokenizerManager (see handle_phase_flip).
        return PhaseFlipReqInput(
            direction=decision.direction, source="policy", internal=True
        )

    def arm_phase_flip(self, direction: str, source: str):
        """#631: replicated arming entry (RPC / regime gate). Activates the
        abort deferral window BEFORE arming so no abort can slip between
        the arm and the first consensus round (pin 4); the window drains
        at cutover, or deactivates here if arming is refused."""
        if self.phase_flip_runtime is None:
            return False, (
                "phase-flip runtime not built yet (no scheduler round has "
                "run); retry after the first round"
            )
        window = self.phase_flip_abort_window
        if window is not None:
            window.activate()
        ok, msg = self.phase_flip_runtime.arm(direction, source)
        if not ok and window is not None:
            window.deactivate_and_drain()
        return ok, msg

    def abort_request(self, recv_req: AbortReq):
        # #631 pin 4: while a flip is armed/executing, abort work is
        # DEFERRED -- an abort applied on one rank before its peers
        # diverges the replicated live set mid-flip. The queue preserves
        # order and drains in the first post-cutover round (or on refused
        # arming). Inactive window (or no flip boot) = direct call,
        # byte-identical to today.
        window = self.phase_flip_abort_window
        if window is not None and window.active:
            window.submit(lambda: self._abort_request_now(recv_req))
            return
        self._abort_request_now(recv_req)

    def _abort_request_now(self, recv_req: AbortReq):
        if (chunked_req := self.chunked_req) is not None:
            if recv_req.abort_all or chunked_req.rid.startswith(recv_req.rid):
                self._pending_chunked_abort_req = chunked_req

        # todo hisparse, release resources for abort requests in hisparse coordinator
        # Delete requests in the waiting queue
        to_del = []
        for i, req in enumerate(self.waiting_queue):
            if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                to_del.append(i)

        # Sort in reverse order to avoid index issues when deleting
        for i in reversed(to_del):
            # Abort method 1: directly pop from the queue
            # This only works for requests that have not started anything.
            # We still need to send something back to TokenizerManager to clean up the state.
            req = self.waiting_queue.pop(i)
            if self.enable_hicache_storage:
                # to release prefetch events associated with the request
                self.tree_cache.release_aborted_request(req.rid)
            self.ipc_channels.send_to_tokenizer.send_output(AbortReq(rid=req.rid), req)
            # For disaggregation decode mode, the request in the waiting queue has KV cache allocated.
            if self.disaggregation_mode == DisaggregationMode.DECODE:
                release_kv_cache(req, self.tree_cache)
            # For disaggregation prefill mode, free the metadata buffer index
            if self.disaggregation_mode == DisaggregationMode.PREFILL:
                bootstrap_pending = req.pending_bootstrap
                maybe_release_metadata_buffer(
                    req, self.req_to_metadata_buffer_idx_allocator
                )
                if (
                    bootstrap_pending
                    and hasattr(req, "disagg_kv_sender")
                    and req.disagg_kv_sender is not None
                ):
                    if hasattr(req.disagg_kv_sender, "abort"):
                        req.disagg_kv_sender.abort()

            # For mamba radix cache
            if (
                req.mamba_pool_idx is not None
                and self.disaggregation_mode != DisaggregationMode.DECODE
            ):
                release_kv_cache(req, self.tree_cache, is_insert=False)
            logger.debug(f"Abort queued request. {req.rid=}")

        # Delete the requests in the grammar queue
        # Abort method 2: call `set_finish_with_abort`
        # The request will still run one prefill forward pass.
        # In this case, we change the input_ids to be only one token to make this prefill cheap.
        self.grammar_manager.abort_requests(recv_req)

        # Delete requests not in the waiting queue when PD disaggregation is enabled
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            # Abort requests that have not yet been bootstrapped
            for req in self.disagg_prefill_bootstrap_queue.queue:
                if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort bootstrap queue request. {req.rid=}")
                    if self.enable_hicache_storage:
                        self.tree_cache.release_aborted_request(req.rid)

                    if hasattr(req.disagg_kv_sender, "abort"):
                        req.disagg_kv_sender.abort()

            # Abort in-flight requests
            for req in self.disagg_prefill_inflight_queue:
                if recv_req.abort_all or req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort inflight queue request. {req.rid=}")
                    if hasattr(req.disagg_kv_sender, "abort"):
                        req.disagg_kv_sender.abort()

        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            # Abort requests that have not yet finished preallocation
            for decode_req in self.disagg_decode_prealloc_queue.queue:
                if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort prealloc queue request. {decode_req.req.rid=}")
                    decode_req.kv_receiver.abort()

            # Abort requests waiting for kvcache to release tree cache
            for decode_req in self.disagg_decode_transfer_queue.queue:
                if recv_req.abort_all or decode_req.req.rid.startswith(recv_req.rid):
                    logger.debug(f"Abort transfer queue request. {decode_req.req.rid=}")
                    decode_req.kv_receiver.abort()

            # Abort requests already retracted to CPU cache
            if self.disagg_decode_prealloc_queue.retracted_queue:
                remaining_retracted = []
                for decode_req in self.disagg_decode_prealloc_queue.retracted_queue:
                    if recv_req.abort_all or decode_req.rid.startswith(recv_req.rid):
                        assert hasattr(decode_req, "kv_cache_cpu")
                        del decode_req.kv_cache_cpu
                        self.ipc_channels.send_to_tokenizer.send_output(
                            AbortReq(rid=decode_req.rid), decode_req
                        )
                    else:
                        remaining_retracted.append(decode_req)
                self.disagg_decode_prealloc_queue.retracted_queue = remaining_retracted

        # Delete requests in the running batch
        if self.ps.pp_size == 1:
            inflight_batches = [self.running_batch, self.last_batch]
        else:
            inflight_batches = [*self.running_mbs, *self.mbs]
        if self.kv_session_offload is not None:
            # A host-spilled session is running too (it decodes via the
            # spill tick, outside running_batch) -- abort must reach it.
            inflight_batches.extend(self.kv_session_offload.inflight_batches())

        inflight_reqs = {r for b in inflight_batches if b is not None for r in b.reqs}
        for req in inflight_reqs:
            if not req.finished() and (
                recv_req.abort_all or req.rid.startswith(recv_req.rid)
            ):
                # Abort method 3: set `to_finish`
                # The request will still run one decode forward pass.
                # Then we reuse all existing code to clean up the KV cache allocation.
                logger.debug(f"Abort running request. {req.rid=}")
                req.to_finish = FINISH_ABORT()

    def _pause_engine(self) -> Tuple[List[Req], int]:
        raise NotImplementedError()

    def pause_generation(self, recv_req: PauseGenerationReqInput):
        self._engine_paused = True

        if recv_req.mode == "in_place":
            # In-place pause: just set the flag and return immediately.
            # All scheduler state (running_batch, last_batch, chunked_req,
            # result_queue) is left untouched. On resume, the normal event
            # loop (get_next_batch_to_run) handles last_batch merge,
            # chunked_req cleanup, and overlap result processing through
            # the standard code paths. This avoids duplicating batch
            # manipulation logic and the accounting bugs that come with it.
            return

        if self.enable_overlap and self.last_batch:
            # Process the results of the last batch
            tmp_batch, tmp_result = self.result_queue.popleft()
            self.process_batch_result(tmp_batch, tmp_result)

        # DECOUPLE S4b: drain the concurrent spill lane's depth-1 result queue
        # so a paused spilled request's last token is committed (never left
        # orphaned across the pause). No-op when decoupling is off.
        spill_rq = getattr(self, "_spill_result_queue", None)
        while spill_rq:
            b, r = spill_rq.popleft()
            self.process_batch_result(b, r)

        if self.last_batch and self.last_batch.forward_mode.is_extend():
            chunked_req_to_exclude = set()
            self.last_batch.filter_batch(
                chunked_req_to_exclude=list(chunked_req_to_exclude)
            )
            # Skip merge for disagg prefill: completed prefill requests are
            # already in disagg_prefill_inflight_queue. Merging them into
            # running_batch leaks them, since the prefill event loop never
            # calls update_running_batch to clean them up.
            if (
                not self.last_batch.is_empty()
                and self.disaggregation_mode != DisaggregationMode.PREFILL
            ):
                if self.running_batch.is_empty():
                    self.running_batch = self.last_batch
                else:
                    self.running_batch.merge_batch(self.last_batch)

        self.last_batch = None
        self.cur_batch_for_debug = None

        if recv_req.mode == "retract" and not self.running_batch.is_empty():
            self.running_batch.filter_batch()
            if len(self.running_batch.reqs) != 0:
                # Decode-side retract always rebootstraps (recomputes the KV from
                # the prefill), so skip the device->host KV offload that release_req
                # would otherwise do; the offloaded copy would be immediately
                # discarded. Non-decode modes ignore offload_kv (they never offload).
                retracted_reqs = self.running_batch.retract_all(
                    self.server_args, offload_kv=False
                )
                for req in retracted_reqs:
                    if self.disaggregation_mode == DisaggregationMode.DECODE:
                        if req.output_ids:
                            req.pd_rebootstrap_forced_output_id = req.output_ids.pop()
                        req.pd_rebootstrap_in_progress = True
                        req.time_stats.set_retract_time()
                        self.disagg_decode_prealloc_queue.hold_rebootstrap(req)
                    else:
                        self._add_request_to_queue(req)

            self.running_batch.batch_is_full = False
            self.chunked_req = None

        # Surface the paused state to dashboards immediately. The scheduler
        # event loop short-circuits before reaching ``on_idle`` while paused,
        # so without this hop ``gen_throughput`` retains its last non-zero
        # value and KV events are not flushed for the entire pause window
        # (e.g. across a weight update). Zero the gauge, force a one-shot
        # idle log by resetting the rate-limit timestamp, and flush pending
        # KV events.
        self.metrics_reporter.last_gen_throughput = 0.0
        if self.metrics_reporter.current_scheduler_metrics_enabled:
            self.metrics_reporter.metrics_collector.last_log_time = 0.0
            self.metrics_reporter._maybe_log_idle_metrics()
        self.kv_events_publisher.publish_kv_events()

    def continue_generation(self, recv_req: ContinueGenerationReqInput):
        if recv_req.torch_empty_cache:
            before_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            torch.cuda.empty_cache()
            after_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            logger.info(
                f"[continue_generation] torch.cuda.empty_cache() called: "
                f"reserved {before_mb:.1f} MB -> {after_mb:.1f} MB "
                f"(freed {before_mb - after_mb:.1f} MB)"
            )
        # Enqueue any rebootstrap requests that were staged during a
        # retract-mode pause. Deferring until resume keeps the preallocation
        # queue empty during the pause window (so an intervening weight update
        # can flush the cache) and recomputes the prefix KV under the updated
        # weights.
        if (
            self.disaggregation_mode == DisaggregationMode.DECODE
            and self.disagg_decode_prealloc_queue is not None
        ):
            self.disagg_decode_prealloc_queue.enqueue_held_rebootstrap()
        self._engine_paused = False

    def load_lora_adapter(
        self, recv_req: LoadLoRAAdapterReqInput
    ) -> LoadLoRAAdapterReqOutput:
        """In-place loading a new lora adapter from disk or huggingface."""

        result = self.tp_worker.load_lora_adapter(recv_req)
        return result

    def load_lora_adapter_from_tensors(
        self, recv_req: LoadLoRAAdapterFromTensorsReqInput
    ) -> LoadLoRAAdapterFromTensorsReqOutput:
        """In-place loading a new lora adapter from serialized tensors."""

        result = self.tp_worker.load_lora_adapter_from_tensors(recv_req)
        return result

    def unload_lora_adapter(
        self, recv_req: UnloadLoRAAdapterReqInput
    ) -> UnloadLoRAAdapterReqOutput:
        """Unload the lora adapter."""

        result = self.tp_worker.unload_lora_adapter(recv_req)
        return result

    def init_weights_send_group_for_remote_instance(
        self, recv_req: InitWeightsSendGroupForRemoteInstanceReqInput
    ):
        """Init the seed and client instance communication group."""
        success, message = self.tp_worker.init_weights_send_group_for_remote_instance(
            recv_req
        )
        return InitWeightsSendGroupForRemoteInstanceReqOutput(
            success=success, message=message
        )

    def send_weights_to_remote_instance(
        self, recv_req: SendWeightsToRemoteInstanceReqInput
    ):
        """Send the seed instance weights to the destination instance."""
        success, message = self.tp_worker.send_weights_to_remote_instance(recv_req)
        return SendWeightsToRemoteInstanceReqOutput(success=success, message=message)

    def slow_down(self, recv_req: SlowDownReqInput):
        t = recv_req.forward_sleep_time
        if t is not None and t <= 0:
            t = None
        self.forward_sleep_time = t
        return SlowDownReqOutput()

    def expert_distribution_handle(self, recv_req: ExpertDistributionReq):
        action = recv_req.action
        if action == ExpertDistributionReqType.START_RECORD:
            get_global_expert_distribution_recorder().start_record()
        elif action == ExpertDistributionReqType.STOP_RECORD:
            get_global_expert_distribution_recorder().stop_record()
        elif action == ExpertDistributionReqType.DUMP_RECORD:
            get_global_expert_distribution_recorder().dump_record()
        else:
            raise ValueError(f"Unrecognized ExpertDistributionReq value: {recv_req=}")
        return ExpertDistributionReqOutput()

    def open_session(self, recv_req: OpenSessionReqInput):
        output = self.session_controller.open(recv_req)
        if self.ps.pp_rank == 0 and self.ps.tp_rank == 0 and self.ps.attn_cp_rank == 0:
            return output
        return None

    def close_session(self, recv_req: CloseSessionReqInput):
        if self.server_args.enable_session_radix_cache:
            self.tree_cache.release_radix_session(recv_req.session_id)
        if recv_req.session_id in self.session_controller or not (
            self.server_args.enable_session_radix_cache
        ):
            self.session_controller.close(recv_req)

    def maybe_sleep_on_idle(self):
        # #297: an armed reshard commits at an IDLE consensus boundary, so
        # the loop must keep ticking rounds until the commit -- the sleeper
        # parks rank 0 in a zmq poll and the request broadcast parks every
        # other rank behind it, and a parked loop never reaches a boundary
        # (observed on the first card run: armed on all ranks, zero
        # boundaries). The pending target is replicated (broadcast RPC or
        # consensus-committed ladder flip), so every rank skips the sleep in
        # the same rounds; the busy spin is bounded by the consensus
        # interval plus the move itself.
        if (
            self.kv_reshard_runtime is not None
            and self.kv_reshard_runtime.pending is not None
        ):
            return
        # #631: identical rationale for an armed phase flip -- the commit
        # fires at a consensus boundary and a parked loop never reaches
        # one. pending is replicated (armed via broadcast RPC or the
        # regime gate's replicated decision), so every rank skips the
        # sleep in the same rounds.
        if (
            self.phase_flip_runtime is not None
            and self.phase_flip_runtime.pending is not None
        ):
            return
        # #330: same rationale for a pending capacity change -- the commit
        # needs consensus boundaries, and a parked loop never reaches one.
        # pending_work() is a pure function of replicated state, so every
        # rank skips the sleep in the same rounds.
        if (
            self.kv_capacity_runtime is not None
            and self.kv_capacity_runtime.pending_work()
        ):
            return
        # #274/#547: the multi-group lane runtime is driven from this loop --
        # a SERIAL lane's forward runs inside _dual_group_lane_tick, and a
        # CONCURRENT lane's admission/lending decision is taken there too. The
        # serving group being idle says nothing about the lane, so parking the
        # loop would starve the protected class. Scope the idle poll to the
        # classic single-group path and leave the lane runtime alone.
        if self.dual_group_lanes:
            return
        if self.idle_sleeper is not None:
            self.idle_sleeper.maybe_sleep()

    def handle_freeze_gc(self, recv_req: FreezeGCReq):
        """Handle freeze_gc request: freeze scheduler's GC and forward to detokenizer."""
        freeze_gc("Scheduler")
        self.ipc_channels.send_to_detokenizer.send_output(recv_req, recv_req)
        return None

    def handle_shutdown(self, recv_req: ShutdownReq):
        # Break the event loop; the finally in run_scheduler_process releases resources.
        self.gracefully_exit = True
        return None

    def configure_logging(self, recv_req: ConfigureLoggingReq):
        if recv_req.log_level is not None:
            logging.getLogger().setLevel(recv_req.log_level.upper())
        self.ipc_channels.send_to_detokenizer.send_output(recv_req, recv_req)

    def handle_dumper_control(self, recv_req: DumperControlReqInput):
        from sglang.srt.debug_utils.dumper import dumper

        try:
            response: list = []
            if (
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ):
                response = dumper._http_manager.handle_request(
                    method=recv_req.method, body=recv_req.body
                )
            self.ipc_channels.send_to_tokenizer.send_output(
                DumperControlReqOutput(success=True, response=response), recv_req
            )
        except Exception as e:
            print(f"[Scheduler] handle_dumper_control error: {e}", flush=True)
            self.ipc_channels.send_to_tokenizer.send_output(
                DumperControlReqOutput(success=False, response=[], error=str(e)),
                recv_req,
            )

    # placeholder for override
    def update_cache_from_scheduler(
        self, schedule_batch: ScheduleBatch, batch_result: GenerationBatchResult
    ):
        pass


def run_phase_flip_event_loops(scheduler: Scheduler):
    """#631: the re-dispatching wrapper around the event loops.

    dispatch_event_loop picks its loop ONCE from server_args.pp_size, so a
    flipped topology has no representation there -- this wrapper wraps
    (never patches) that decision: each phase runs its own loop, and a
    committed flip exits the current loop via PhaseFlipLoopExit (raised by
    the on_round hook at a quiescent boundary, after all cutover rebuilds)
    to be re-dispatched here under the new phase. A normal loop return
    (shutdown) returns from the wrapper."""
    from sglang.srt.managers.phase_flip_runtime import (
        PHASE_TP,
        PhaseFlipLoopExit,
    )

    assert (
        scheduler.disaggregation_mode == DisaggregationMode.NULL
    ), "phase flip x PD disaggregation is refused at argument time"
    assert not scheduler.enable_pdmux, "phase flip x pdmux is out of scope"
    while True:
        try:
            if scheduler.phase_flip_active_stack == PHASE_TP:
                # TP decode phase: the non-PP production loop family.
                if scheduler.enable_overlap:
                    scheduler.event_loop_overlap()
                else:
                    scheduler.event_loop_normal()
            else:
                # PP prefill phase: the boot topology's loop.
                scheduler.event_loop_pp()
            return
        except PhaseFlipLoopExit as e:
            logger.warning(
                "PHASE-FLIP event loop re-dispatch after %s (active stack now %s)",
                e.direction,
                scheduler.phase_flip_active_stack,
            )
            continue


def dispatch_event_loop(scheduler: Scheduler):
    # Dispatch to the appropriate event loop based on the disaggregation mode
    # #631: a phase-flip boot re-dispatches per phase; wrapper, not patch.
    if getattr(scheduler.server_args, "enable_phase_flip", False):
        run_phase_flip_event_loops(scheduler)
        return
    server_args = scheduler.server_args
    disaggregation_mode: DisaggregationMode = scheduler.disaggregation_mode
    if disaggregation_mode == DisaggregationMode.NULL:
        if scheduler.enable_pdmux:
            scheduler.event_loop_pdmux()
        elif server_args.pp_size > 1:
            scheduler.event_loop_pp()
        elif scheduler.enable_overlap_mlx:
            scheduler.event_loop_overlap_mlx()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap()
        else:
            scheduler.event_loop_normal()
    elif disaggregation_mode == DisaggregationMode.PREFILL:
        if server_args.pp_size > 1:
            scheduler.event_loop_pp_disagg_prefill()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap_disagg_prefill()
        else:
            scheduler.event_loop_normal_disagg_prefill()
    elif disaggregation_mode == DisaggregationMode.DECODE:
        if server_args.pp_size > 1:
            scheduler.event_loop_pp_disagg_decode()
        elif scheduler.enable_overlap:
            scheduler.event_loop_overlap_disagg_decode()
        else:
            scheduler.event_loop_normal_disagg_decode()


def uneven_family_plans(server_args) -> dict:
    """The per-family shard vectors this worker will install, or ``{}``.

    "vocab" (--rank-vocab-ratio / SGLANG_UNEVEN_VOCAB_VECTOR) is read
    exclusively through ``tp_vocab_ratios()``, which does NOT fall back to the
    base plan: without the flag, vocab sharding stays even. It is still listed
    here so a symbolic value in that family gets the same treatment.

    A SYMBOLIC value that reaches a worker was never resolved by the launcher
    (#394 slice 3), and falling through to the base plan is precisely the
    failure the symbol exists to avoid: the boot would serve the baseline under
    the treatment's name, and only a #390 dump read a week later would show it.
    Its own function so the refusal can be exercised without spawning a
    scheduler.
    """
    family_plans = {}
    for family, field in (
        ("mlp", "rank_mlp_ratio"),
        ("moe", "rank_moe_ratio"),
        ("vocab", "rank_vocab_ratio"),
    ):
        vector = getattr(server_args, field, None)
        if isinstance(vector, str):
            raise ValueError(
                f"{field}={vector!r} is a symbolic placement that must be "
                "resolved before the workers are spawned. It reached this "
                "scheduler unresolved, which means the launcher path did not "
                "run resolve_moe_compute_placement_flag()."
            )
        if isinstance(vector, list):
            family_plans[family] = vector
    return family_plans


def configure_scheduler_process(
    server_args: ServerArgs,
    gpu_id: int,
    tp_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
) -> Optional[int]:
    """Configure scheduler worker: logging, process title, etc.

    Returns:
        dp_rank
    """
    kill_itself_when_parent_died()

    # Generate the logger prefix
    if dp_rank is None and "SGLANG_DP_RANK" in os.environ:
        # [For Router] if env var "SGLANG_DP_RANK" exist, set dp_rank to the value of the env var
        dp_rank = int(os.environ["SGLANG_DP_RANK"])

    prefix = ""
    if dp_rank is not None:
        prefix += f" DP{dp_rank}"
    if server_args.pp_size > 1:
        prefix += f" PP{pp_rank}"
    if server_args.attn_cp_size > 1:
        prefix += f" ATTN_CP{attn_cp_rank}"
    if server_args.moe_dp_size > 1:
        prefix += f" MOE_DP{moe_dp_rank}"
    if server_args.tp_size > 1:
        prefix += f" TP{tp_rank}"
    if server_args.ep_size > 1:
        prefix += f" EP{moe_ep_rank}"

    # Install the uneven-TP shard plan (--rank-tp-ratio) for this worker
    # process before any model code runs. None (the default) keeps the
    # classic even split everywhere. The optional family vectors
    # ("mlp" = --rank-mlp-ratio / SGLANG_UNEVEN_MLP_VECTOR, "moe" =
    # --rank-moe-ratio / SGLANG_UNEVEN_MOE_VECTOR, validated in
    # server_args) re-balance only their weight family; families
    # without a vector fall back to the base plan.
    from sglang.srt.distributed.utils import set_tp_partition_ratios

    base_plan = (
        server_args.rank_tp_ratio
        if isinstance(server_args.rank_tp_ratio, list)
        else None
    )
    set_tp_partition_ratios(
        base_plan, families=uneven_family_plans(server_args) or None
    )

    # Uneven DCP (M1): install the token-axis split vector so the KV pool
    # pinning and the weighted owner rule agree across ranks. None keeps the
    # even modulo DCP path. Gated behind dcp_size>1 (auto-set for a
    # non-uniform --rank-tp-ratio) so the default path is untouched.
    from sglang.srt.distributed.utils import (
        resolve_cp_token_ratios,
        set_cp_token_ratios,
    )

    # The weighted owner rule (SGLANG_UNEVEN_DCP_WEIGHTED=1, or any
    # non-'coupled' --rank-kv-ratio, which implies it without the env pair)
    # installs the token vector so cp_token_split_factor()=sum(ratios);
    # without it, DCP uses the even modulo owner rule
    # (split_factor=dcp_size). Staged so the allocator size factor always
    # matches the divisor the DCP kernels use. Under --rank-kv-ratio
    # capacity this pre-boot vector is the phase-1 ESTIMATE; the measured
    # capacity-optimal vector replaces it after the post-weight-load
    # profiling, before any pool/backend snapshots it (see
    # _maybe_suggest_dcp_token_vector).
    _dcp_size = getattr(server_args, "dcp_size", 1)
    if (
        _dcp_size > 1
        and base_plan is not None
        and server_args.uneven_weighted_dcp_enabled()
    ):
        cp_vector = resolve_cp_token_ratios(server_args)
        set_cp_token_ratios(cp_vector)
        if cp_vector is not None:
            logger.info("Uneven DCP token vector installed: %s", cp_vector)
    elif _dcp_size > 1 and envs.SGLANG_UNEVEN_TOKEN_VECTOR.get():
        # GUARD REACHABILITY, not a second installer.
        #
        # resolve_cp_token_ratios owns the honesty guard for "a token vector
        # is set but nothing can engage it" -- and the gate above asked for
        # `base_plan is not None`, i.e. for the presence of the very plan
        # whose ABSENCE that guard reports. So from a real boot the resolver
        # was never called in the one case it exists for: the vector was
        # silently ignored and the server came up looking configured while
        # running plain even DCP (measured: Qwen3.5-2B TP=2/DCP=2 TOKVEC=2,1,
        # token-identical to TP=1, zero uneven-machinery log lines). Tie the
        # call to the token-vector machinery's OWN state instead: a vector is
        # set, so its resolver runs, and is allowed to refuse.
        #
        # It raises for every shape reachable here (no plan, or a uniform
        # one). Should it ever return a vector instead, we deliberately do
        # NOT install it -- the installing lane above is the uneven-DCP lane
        # and stays exactly as it was, so the validated --rank-tp-ratio boots
        # cannot move. The one remaining inert combination (a non-uniform
        # plan plus a vector while the weighted owner rule is switched off)
        # is left inert on purpose: rejecting it would change that lane.
        resolve_cp_token_ratios(server_args)

    # Weightless-KV fast lane (Variant C Stage 1): name the head rank that
    # holds all weights/heads. The flashinfer backend then forces the DCP
    # decode path with the [total,0,...,0] head vector (Q all-gather ->
    # broadcast from head rank; O merge -> sliced back to head rank only),
    # independently of --rank-tp-ratio. None keeps every other path
    # byte-identical.
    if getattr(server_args, "weightless_kv_fastlane", False):
        from sglang.srt.distributed.utils import set_weightless_kv_head_rank

        set_weightless_kv_head_rank(server_args.weightless_kv_head_rank)
        logger.info(
            "Weightless-KV fast lane active: head rank = %d (holds all "
            "weights/heads); other ranks weightless (KV-token-shard only).",
            server_args.weightless_kv_head_rank,
        )

    # Apply this rank's --rank-gpu-memory-mib budget as its
    # mem_fraction_static (MiB -> fraction against the rank's physical
    # GPU NVML total, derived once at resolution time). The MiB value is
    # the rank's ENTIRE budget — it is applied unmodified, without any
    # additional utilization ceiling. No-op on the default path.
    world_rank = server_args.world_rank(pp_rank, tp_rank)
    rank_fraction = server_args.apply_rank_memory_budget(world_rank)
    if rank_fraction is not None:
        logger.info(
            "Uneven TP: rank %d (PP%d TP%d) uses mem_fraction_static=%.4f "
            "(--rank-gpu-memory-mib budget on GPU %s).",
            world_rank,
            pp_rank,
            tp_rank,
            rank_fraction,
            server_args.rank_gpu_id[world_rank],
        )

    # Config the process
    setproctitle.setproctitle(f"sglang::scheduler{prefix.replace(' ', '_')}")
    faulthandler.enable()

    # Configure the logger
    configure_logger(server_args, prefix=prefix)
    suppress_other_loggers()

    # Set cpu affinity to this gpu process
    if envs.SGLANG_SET_CPU_AFFINITY.get():
        set_gpu_proc_affinity(
            server_args.pp_size, server_args.tp_size, server_args.nnodes, gpu_id
        )
    if not envs.SGLANG_NUMA_BIND_V2.get():
        numa_node = get_numa_node_if_available(server_args, gpu_id)
        if numa_node is not None:
            numa_bind_to_node(numa_node)

    return dp_rank


def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
):
    # Load plugins so hooks can override Scheduler and its dependencies.
    load_plugins()
    dp_rank = configure_scheduler_process(
        server_args,
        gpu_id,
        tp_rank,
        attn_cp_rank,
        moe_dp_rank,
        moe_ep_rank,
        pp_rank,
        dp_rank,
    )
    parent_process = psutil.Process().parent()

    # VRAM flight recorder (#605). Armed here and nowhere later: this is the
    # last point in a rank's life at which no CUDA allocation has happened yet,
    # and a recording started after the first one cannot attribute a single
    # boot-resident post -- the #602 capture proved that with 3 MiB of 25142
    # MiB. Both calls are no-ops unless their env var is set, and neither
    # initialises CUDA.
    from sglang.srt.mem_ledger import flight_recorder

    flight_recorder.arm_process_trace(rank=tp_rank)
    flight_recorder.mark("process_start", rank=tp_rank)

    # Startup phase timers (#560). The boot phases that dominate time to first
    # token -- weight loading, KV pool sizing, attention backend init, cuda
    # graph capture -- all run in THIS process, not in the parent that calls
    # launch_server. Without this call the phase timers here would log but
    # never reach the gauge, which is the half-wired state #560 inherited.
    #
    # Gauge on rank 0 only. STARTUP_LATENCY_SECONDS carries a single "context"
    # label and no rank label, and its multiprocess_mode is "mostrecent", so
    # arming every rank would make the exported value the last rank to finish
    # a phase rather than any well-defined rank. The per-rank breakdown is not
    # lost: startup_timer logs its INFO line on every rank regardless of this
    # gate, which is the form #539 boot attribution actually reads.
    if server_args.enable_metrics and tp_rank == 0:
        from sglang.srt.observability.startup_func_log_and_timer import (
            enable_startup_timer,
        )

        enable_startup_timer()

    # Set up tracing
    if server_args.enable_trace:
        process_tracing_init(
            server_args.otlp_traces_endpoint,
            "sglang",
            trace_modules=server_args.trace_modules,
        )
        thread_label = "Scheduler"
        if server_args.disaggregation_mode == "prefill":
            thread_label = "Prefill Scheduler"
        elif server_args.disaggregation_mode == "decode":
            thread_label = "Decode Scheduler"
        trace_set_thread_info(thread_label, tp_rank, dp_rank, pp_rank)

    # Create a scheduler and run the event loop
    scheduler = None
    try:
        scheduler = Scheduler(
            server_args,
            port_args,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            attn_cp_rank,
            moe_dp_rank,
            dp_rank,
        )

        # #605: every runner in this process is now up, so this is the first
        # moment a snapshot shows the WHOLE boot -- under speculative decoding
        # the target and the NEXTN draft each capture graphs, and a snapshot
        # taken at either runner's capture_end is missing the other's.
        flight_recorder.mark("boot_complete", rank=tp_rank)
        flight_recorder.dump_trace("boot_complete", rank=tp_rank)
        flight_recorder.disarm_process_trace()

        # Send initialization info back to the parent process
        pipe_writer.send(scheduler.get_init_info())

        # Run the event loop (blocks until a ShutdownReq sets gracefully_exit)
        scheduler.run_event_loop()

    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"Scheduler hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
        # Opt-in: SIGKILL the pgroup so sibling ranks don't spew thousands
        # of NCCL/TCPStore tracebacks before they finally die.
        if envs.SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION.get():
            try:
                # A process group is inherited, so it is only ours if nothing
                # else joined it. Two servers launched from the same shell share
                # one pgid, and killpg would then SIGKILL the co-tenant's ranks
                # too -- an uncatchable signal, so the victim dies with no
                # traceback and no indication of who killed it. Blast radius
                # must be verified, not assumed.
                pgid = os.getpgrp()
                root_pid = parent_process.pid
                if process_group_is_confined_to_tree(root_pid, pgid):
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    logger.error(
                        "SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION: process group %d "
                        "holds processes outside this server's tree (root pid "
                        "%d); killing only this server's tree so a co-tenant "
                        "server is not taken down with it.",
                        pgid,
                        root_pid,
                    )
                    kill_process_tree(root_pid)
            except Exception:
                pass
    finally:
        if scheduler is not None:
            # #363: the verdict trace's summary line. THIS block is the one
            # that runs on both stop paths that unwind -- the graceful one (a
            # ShutdownReq broke the event loop above) and the exception one,
            # KeyboardInterrupt included, which `except Exception` does not
            # catch but `finally` still honours. SIGTERM unwinds through
            # neither and is covered by the handler the observer installs at
            # build time. Idempotent, never raises.
            from sglang.srt.managers.regime_runtime import close_regime_trace

            close_regime_trace(scheduler)
            # FPM has a background ZMQ publisher thread that needs explicit
            # teardown to flush queued metrics and close the socket cleanly.
            scheduler.metrics_reporter._shutdown_fpm()
            # Graceful path only: on the exception path the GPU may be wedged
            # and the synchronize() in destroy() could itself hang.
            if scheduler.gracefully_exit:
                scheduler.release_host_resources()
            # #673: destroy the collectives before the interpreter does. Their
            # C++ watchdog and heartbeat threads are joined by the process
            # group's DESTRUCTOR; with the group never destroyed those
            # std::threads are still joinable at teardown, and destroying a
            # joinable std::thread calls std::terminate -- "terminate called
            # without an active exception", after a clean drain, which is the
            # #673 signature. Graceful path only and flag-gated: the destroy
            # path runs barlink's close(), which is #722's machinery.
            # #673 TEARDOWN STACK -- ONE ORDERED SEQUENCE, and the order is
            # the content. Four background threads and the collectives they
            # run on must come down in a direction that never leaves a thread
            # using something that is already gone, and never blinds the
            # machinery that reports on the ones still dying.
            #
            #   1. kvso-dest-io      -- a host IO thread whose body is
            #      cudaEventSynchronize. It runs no collective and nothing
            #      reports through it, so it goes first: the earlier it stops,
            #      the smaller the window in which it can be inside a CUDA call
            #      while anything below tears down.
            #   2. dual-group lanes  -- CUDA stream workers that launch
            #      kernels. Stopped before the abort readers, because a lane
            #      dying badly is exactly the event those readers exist to
            #      report.
            #   3. barlink watchdog  -- an abort READER (#650/#653 family). It
            #      stays alive through 1 and 2 on purpose, so their deaths are
            #      still observed; stopping it re-arms the transports' in-line
            #      reads, so the guard survives losing its reader.
            #   4. lockstep sentinel -- LAST of the threads, because the other
            #      stops report divergence through its 0.5 s gloo gather.
            #   5. release_distributed -- destroys the groups, which also closes
            #      barlink_comm. Everything above must already be down.
            #
            # Steps 3 and 4 are ALSO called inside release_distributed, which is
            # deliberate belt-and-braces on the ORDERING only: they are
            # idempotent, and repeating them there means the destroy cannot be
            # reached with either thread still running even if this block is
            # later reordered by a careless edit. It is not a second stop.
            from sglang.srt.managers.scheduler_teardown import (
                release_barlink_watchdog,
                release_distributed,
                release_dual_group_lanes,
                release_kv_session_offload_io,
                release_lockstep_sentinel,
            )

            release_kv_session_offload_io(scheduler, graceful=scheduler.gracefully_exit)
            release_dual_group_lanes(scheduler, graceful=scheduler.gracefully_exit)
            release_barlink_watchdog(scheduler, graceful=scheduler.gracefully_exit)
            release_lockstep_sentinel(scheduler, graceful=scheduler.gracefully_exit)
            release_distributed(scheduler, graceful=scheduler.gracefully_exit)


# ---------------------------------------------------------------------------
# #656: the phase policy's control loop, at MODULE level and not on the class.
#
# Both are reached from handle_phase_flip and the round hook, and both of
# those are exercised by stub schedulers (SimpleNamespace and _StubScheduler
# in test_phase_flip_protocol) that bind the real method and carry none of the
# scheduler's fields. As methods these turned "this stub has no policy" into
# an AttributeError inside the RPC handler -- a wiring hole reported as a
# protocol failure. As functions taking the scheduler, every field access is
# already a guarded getattr and a stub simply has no policy to inform.
# ---------------------------------------------------------------------------
def _drain_seam_abandons_into_policy(scheduler) -> None:
    """Report a seam ABANDON to the policy as a refused arm (#656).

    An arm that returns True and then dies at the seam is, from the
    policy's point of view, indistinguishable from an arm that was
    refused outright: no layout changed, and the work the other layout
    owes is still undone. Reporting only the second kind means the first
    ``seam_abandon_cap()`` attempts of every unfundable configuration are
    invisible to the policy -- which is exactly the window boot E spent
    re-arming at the dwell interval.

    Edge-triggered on the runtime's own sequence number so a rank that
    reads it twice in one round counts one abandon.
    """
    rt = getattr(scheduler, "phase_flip_runtime", None)
    if rt is None or getattr(scheduler, "phase_policy_state", None) is None:
        return
    seq = getattr(rt, "seam_abandon_seq", 0)
    if seq == getattr(scheduler, "_last_seam_abandon_seq", 0):
        return
    scheduler._last_seam_abandon_seq = seq
    last = getattr(rt, "last_seam_abandon", None)
    if not last:
        return
    direction, detail = last
    _note_policy_arm_outcome(scheduler, direction, False, f"seam abandoned: {detail}")


def _note_policy_arm_outcome(scheduler, direction: str, ok: bool, msg: str) -> None:
    """Close the policy's control loop with the arm verdict (#656).

    REPORTED FOR EVERY SOURCE, not just the policy's own arms. A manual
    POST /phase_flip that the runtime refuses proves the same thing about
    the seam as a refused internal arm does, and a manual one that
    SUCCEEDS really did reset the dwell -- so the policy has to see both
    or its clock drifts from the layout it is steering.

    Never raises. This runs on the request-handling path of a replicated
    broadcast; an exception here would take down a scheduler over
    bookkeeping.
    """
    state = getattr(scheduler, "phase_policy_state", None)
    if state is None:
        return
    try:
        from sglang.srt.managers.phase_policy import note_flip_outcome

        note_flip_outcome(
            scheduler.phase_policy_cfg,
            state,
            direction,
            bool(ok),
            msg or "",
            # THE SAME CLOCK the policy inputs are stamped with. A
            # perf_counter hold compared against a monotonic stamp is
            # not a shorter or longer hold, it is an arbitrary one.
            time.perf_counter(),
        )
    except Exception as e:  # pragma: no cover - bookkeeping must not kill
        logger.warning("PHASE-POLICY arm outcome not recorded: %s", e)
