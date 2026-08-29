# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/parallel_state.py

# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
"""Distributed state.
It takes over the control of the distributed environment from PyTorch.
The typical workflow is:

- call `init_distributed_environment` to initialize the distributed environment.
- call `initialize_model_parallel` or `ensure_model_parallel_initialized` to
 initialize the model parallel groups.

- any code dealing with the distributed stuff

- call `destroy_model_parallel` to destroy the model parallel groups.
- call `destroy_distributed_environment` to destroy the distributed environment.

If you only need to use the distributed environment without model/pipeline
 parallelism, you can skip the model parallel initialization and destruction
 steps.
"""

import contextlib
import gc
import logging
import os
import pickle
import sys
import weakref
from collections import namedtuple
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing import shared_memory
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from unittest.mock import patch

import torch
import torch.distributed
from torch.distributed import Backend, ProcessGroup

from sglang.srt import platforms
from sglang.srt.compilation.compilation_config import register_split_op

# Imported for its import-time side effect: it raises if a retired
# SGLANG_HTCCL* variable is set (task #358). It sits HERE rather than only in
# the barlink modules because a stale launch script sets SGLANG_HTCCL=1 and
# then nothing imports barlink at all -- the run would come up over NCCL and
# quietly measure the wrong transport, which is exactly the failure the guard
# exists to prevent. parallel_state is imported by every rank of every
# distributed run, and the guard itself is stdlib-only.
from sglang.srt.distributed.device_communicators import (  # noqa: F401
    barlink_env_guard,
)
from sglang.srt.distributed.pp_object_recv import (
    get_or_create_frame as get_or_create_object_recv_frame,
)
from sglang.srt.distributed.pp_object_recv import (
    recv_object_abort_after_s,
    recv_object_step_budget_s,
)
from sglang.srt.distributed.utils import set_global_tcp_store
from sglang.srt.environ import envs
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.platforms.device_mixin import _DEVICE_TO_DISTRIBUTED_BACKEND
from sglang.srt.utils import (
    get_current_device_stream_fast,
    get_int_env_var,
    is_cpu,
    is_cuda,
    is_cuda_alike,
    is_hip,
    is_musa,
    is_npu,
    is_shm_available,
    is_xpu,
)
from sglang.srt.distributed.collective_census import census, census_enabled
from sglang.srt.utils.collective_clock import collective_clock
from sglang.srt.utils.custom_op import register_custom_op
from sglang.srt.utils.network import get_local_ip_auto
from sglang.srt.utils.stale_shm_cleanup import make_shm_name

_is_npu = is_npu()
_is_cpu = is_cpu()
_is_xpu = is_xpu()
_is_musa = is_musa()

TensorMetadata = namedtuple("TensorMetadata", ["device", "dtype", "size"])

# Per-rank compute/wait split for the prefill log. Disarmed for every forward
# except the ones the per-rank prefill line already times, so the cost on all
# other collectives is the `armed` attribute read below.
_COLLECTIVE_CLOCK = collective_clock()

# #583 collective census: resolved once at import so the hot-path guard is a
# module-global bool read, not an environment lookup per collective.
_CENSUS = census()
_CENSUS_ON = census_enabled()


def collective_clock_families(
    group_name: str,
) -> Tuple[str, str, str, str, str, str]:
    """Per-group collective-clock family names, in dispatch-site order.

    ``(all_reduce, all_gather, all_gatherv, reduce_scatterv, broadcast,
    all_to_all)``.

    The GROUP is the axis the wait decomposition needs. "tp.all_reduce" is
    the per-layer tensor reduction whose payload scales with new tokens;
    "dcp.all_gather" is context-parallel attention traffic. They have the
    same units and completely different fixes, which is precisely why one
    summed ``wait`` number cannot arbitrate between them.

    CENSUS-ONLY TAIL. The first four names have a ``_COLLECTIVE_CLOCK.span``
    at their dispatch site and are therefore also wait-decomposed in the
    per-rank prefill line. ``broadcast`` and ``all_to_all`` are counted by the
    census but carry NO clock span: this function is the one place that names
    a group's families, and both instruments read it, but arming a span is a
    separate decision with its own cost and its own log-line surface. Naming
    a family here is not a claim that it is timed.

    ``broadcast`` and ``all_to_all`` were added by #583. The 2026-08-06
    05:53:59 crash was a rank-arrival desync in which all FOUR censused
    families agreed exactly across ranks (10176/2864/580/36824 on ranks 0 and
    1), so the family that diverged was necessarily one nobody counted --
    and the abort named an 8-byte collective served by the a2a kernel. A
    census that cannot see the two families the failure hid in reports
    agreement and calls it health.
    """
    return (
        f"{group_name}.all_reduce",
        f"{group_name}.all_gather",
        f"{group_name}.all_gatherv",
        f"{group_name}.reduce_scatterv",
        f"{group_name}.broadcast",
        f"{group_name}.all_to_all",
    )


# use int value instead of ReduceOp.SUM to support torch compile
REDUCE_OP_SUM = int(torch.distributed.ReduceOp.SUM)

# Reuse the user-provided distributed timeout for model-parallel subgroup
# creation so runtime collectives do not silently fall back to backend defaults.
_MODEL_PARALLEL_GROUP_TIMEOUT: Optional[timedelta] = None

#: Fully qualified name of the peer-liveness module. Looked up in
#: ``sys.modules`` rather than imported: a boot without an barlink transport
#: never loads it, and ``GroupCoordinator.barrier`` must not be the thing
#: that changes that.
_BARLINK_LIVENESS_MODULE = (
    "sglang.srt.distributed.device_communicators.barlink_liveness"
)

#: Mirrors ``barlink_liveness.ENV_ENABLE``. Duplicated because reading the
#: switch must not require importing the module the switch controls; the
#: module's own value is preferred whenever it is already loaded.
_PEER_LIVENESS_ENV = "SGLANG_BARLINK_PEER_LIVENESS"
_PEER_LIVENESS_OFF = ("", "0", "false", "no", "off")


def _peer_liveness_forced(module=None) -> bool:
    """Did the operator name the feature explicitly, rather than default it?

    The module defaults the feature ON, which is right for a process that
    already runs barlink. It is not enough to make a process that does not
    import the module load it, so the barrier below asks for an explicit
    opt-in instead.
    """
    name = getattr(module, "ENV_ENABLE", _PEER_LIVENESS_ENV)
    return os.environ.get(name, "").strip().lower() not in _PEER_LIVENESS_OFF


def _peer_liveness_for_barrier():
    """The liveness module if this process's barriers should be bounded.

    ``None`` on a plain boot, and reaching that answer costs one dict
    lookup and one ``os.environ`` read: no import, no collective, and the
    caller's original ``torch.distributed.barrier`` unchanged.
    """
    module = sys.modules.get(_BARLINK_LIVENESS_MODULE)
    if module is None:
        # Never imported => no barlink transport in this process. Only an
        # explicit opt-in justifies importing it just for the barrier.
        if not _peer_liveness_forced():
            return None
        from sglang.srt.distributed.device_communicators import barlink_liveness

        return barlink_liveness
    # An barlink transport lives here. A registered table means the identity
    # exchange succeeded and a dead peer can be named by rank, host and pid.
    if module.registered_tables():
        return module
    return module if _peer_liveness_forced(module) else None


def get_torch_distributed_pg_options(group_name=None):
    if not _is_npu:
        return None

    # Only create HCCL options for default group or MoE-related groups
    if group_name is not None and "moe" not in group_name:
        return None

    import torch_npu

    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    hccl_buffer_size = int(
        os.environ.get("DEEPEP_HCCL_BUFFSIZE") or os.environ.get("HCCL_BUFFSIZE") or 200
    )
    options.hccl_config = {"hccl_buffer_size": hccl_buffer_size}
    return options


@dataclass
class GraphCaptureContext:
    stream: torch.get_device_module().Stream


@dataclass
class P2PWork:
    work: Optional[torch.distributed.Work]
    payload: Optional[torch.Tensor]


def _split_tensor_dict(
    tensor_dict: Dict[str, Union[torch.Tensor, Any]],
) -> Tuple[List[Tuple[str, Any]], List[torch.Tensor]]:
    """Split the tensor dictionary into two parts:
    1. A list of (key, value) pairs. If the value is a tensor, it is replaced
         by its metadata.
    2. A list of tensors.
    """
    metadata_list: List[Tuple[str, Any]] = []
    tensor_list: List[torch.Tensor] = []
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            # Note: we cannot use `value.device` here,
            # because it contains not only the device type but also the device
            # index (e.g. "cuda:0"). We only need the device type.
            # receiving side will set the device index.
            device = value.device.type
            metadata_list.append(
                (key, TensorMetadata(device, value.dtype, value.size()))
            )
            tensor_list.append(value)
        else:
            metadata_list.append((key, value))
    return metadata_list, tensor_list


_group_name_counter: Dict[str, int] = {}


def _get_unique_name(name: str) -> str:
    """Get a unique name for the group.
    Example:
    _get_unique_name("tp") -> "tp:0"
    _get_unique_name("tp") -> "tp:1"
    """
    if name not in _group_name_counter:
        _group_name_counter[name] = 0
    newname = f"{name}:{_group_name_counter[name]}"
    _group_name_counter[name] += 1
    return newname


_groups: Dict[str, Callable[[], Optional["GroupCoordinator"]]] = {}


def _register_group(group: "GroupCoordinator") -> None:
    _groups[group.unique_name] = weakref.ref(group)


@register_custom_op(mutates_args=["tensor"])
@register_split_op()
def inplace_all_reduce(tensor: torch.Tensor, group_name: str) -> None:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    group._all_reduce_in_place(tensor)


@register_custom_op(out_shape="tensor")
def outplace_all_reduce(
    tensor: torch.Tensor, group_name: str, outplace_all_reduce_method: str
) -> torch.Tensor:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    return group._all_reduce_out_place(tensor, outplace_all_reduce_method)


@register_custom_op(mutates_args=["output"])
def reg_all_gather_into_tensor(
    output: torch.Tensor, input: torch.Tensor, group_name: str
) -> None:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    group._all_gather_into_tensor(output, input)


@register_custom_op(mutates_args=["output"])
def reg_reduce_scatter_tensor(
    output: torch.Tensor, input: torch.Tensor, group_name: str
) -> None:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    group._reduce_scatter_tensor(output, input)


@register_custom_op(mutates_args=["output"])
def reg_all_to_all_single(
    output: torch.Tensor, input: torch.Tensor, group_name: str
) -> None:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    group._all_to_all_single(output, input)


# The transports whose collectives run entirely ON the GPU and are therefore
# legal inside a CUDA-graph capture. An ALLOWLIST on purpose: every other
# value host-stages its collectives -- shm and gloo over CPU memory, ucx over
# pinned host buffers for RDMA (FEATURES_VS_UPSTREAM.md, ucx transport table:
# "`--enforce-eager` required ... Only `device` is capturable"), and any
# unknown name silently maps to the inline gloo plane in
# barlink._build_transport. The previous denylist ({"shm", "gloo"}) did not
# know "ucx", so a ucx boot with CUDA graphs enabled passed this guard and
# then either crashed mid-capture or captured only rank-local regions --
# silently, which left the graph regime of a cross-rig measurement unknown.
#
# "host" is on the list for the same reason "device" is, and NOT because its
# bytes sit in host memory -- that is precisely what the other three do too.
# What decides membership is who drives: barlink_host stages and reduces from
# two stream-ordered kernels that spin on flags in the pinned segment, never
# calls a synchronize, and keeps its per-op sequence number in DEVICE memory
# so a graph replay advances it exactly as the first run did.
CAPTURABLE_BARLINK_TRANSPORTS = frozenset({"device", "host"})

#: The GPU-driven transports whose graph capture is carried by the release
#: switch below rather than by the proven base set. They are NOT in the list
#: above -- they join it through the switch, and only through it.
GRAPH_ENABLE_TRANSPORTS = frozenset({"bar1", "matrix"})

#: The switch, RELEASED on 2026-08-01 (task #369): the default is now on.
#:
#: Why a switch rather than two more names in the list: what stood between
#: bar1 and a capture had been fixed in code (the fallback bolt in
#: `barlink._select`, the kernel choice in `BarlinkBar1Transport._kernel`, the
#: direct mode in `_res_slot`) -- but "fixed" was a statement about the code,
#: not about the hardware. The one point that could not be settled without
#: cards was whether the driver accepts `cudaLaunchCooperativeKernel` from
#: inside a stream capture.
#:
#: It does. `benchmark/bar1_graph_check.py` passed 10/10 on three cards
#: (5090 + 2x 3080), all nine gate cases plus the informational `grid` case,
#: which is exactly the cooperative-launch question. Evidence:
#: `/spinning/gpu-battery-results/2026-08-01_369_bar1_graph_gate/`
#: `gate_PASS_docker.log`. It had to run inside the htsglang Docker image on
#: the Proxmox host, because that is the only place on this rig that has both
#: `/dev/dmabuf_holder` and a toolchain able to build the JIT extension --
#: see `docs/rig-runbook.md` section 4.15.
#:
#: The switch stays as the off-ramp: `SGLANG_BARLINK_GRAPH_ENABLE=0` restores
#: the pre-release behaviour, under which a captured run falls back to the
#: 1blk variant. Note that reaching bar1 at all already requires a patched
#: driver and the dma-buf holder, so this default cannot surprise a rig that
#: has not deliberately set that up: without them the transport declines and
#: never becomes capturable in the first place.
_GRAPH_ENABLE_ENV = "SGLANG_BARLINK_GRAPH_ENABLE"

#: Environment values that count as "off". Word for word the same tuple as
#: ``barlink_bar1._OFF`` -- both decide the same thing and must never read
#: differently.
_OFF_VALUES = ("0", "no", "off", "false", "")


def graph_enable_set() -> bool:
    """Whether the release switch for the GPU-driven transports is on."""
    import os

    # This function re-reads os.environ live on every call (by design --
    # callers may set the env var at runtime, not just before import), so the
    # retired-name check has to run on every call too. An import-time-only
    # check would miss a retired name exported AFTER this module was first
    # imported, and that name would then be silently ignored.
    barlink_env_guard.check_retired_env_vars()

    return os.environ.get(_GRAPH_ENABLE_ENV, "1") not in _OFF_VALUES


def capturable_transports() -> frozenset:
    """The set that counts as capturable RIGHT NOW.

    ``CAPTURABLE_BARLINK_TRANSPORTS`` is the proven base set and stays that
    way; this function is the one place that adds the release switch on top.
    Whoever asks "is this capturable?" asks here -- not the constant, or the
    switch would take effect in one place and not in the other.
    """
    if graph_enable_set():
        return CAPTURABLE_BARLINK_TRANSPORTS | GRAPH_ENABLE_TRANSPORTS
    return CAPTURABLE_BARLINK_TRANSPORTS


def _enforce_cpu_transport_needs_eager(transport: str) -> None:
    """Reject a host-staged barlink transport while CUDA graphs are enabled.

    Not a style check: a host-staged collective inside a capture raises
    `cudaErrorStreamCaptureUnsupported` from whichever kernel happens to be
    capturing, which reads as an unrelated CUDA fault. Checked here, where the
    transport is known, and only when barlink is actually on -- flag off, this
    function is never called.
    """
    if transport in capturable_transports():
        return
    try:
        from sglang.srt.runtime_context import get_server_args

        server_args = get_server_args()
    except Exception:
        return  # no published ServerArgs yet -> nothing to validate against
    if server_args is None or getattr(server_args, "disable_cuda_graph", False):
        return
    if transport in GRAPH_ENABLE_TRANSPORTS:
        # Reachable only when somebody turned the release OFF again, since
        # #369 flipped its default on. Before that this branch said "capture
        # is UNMEASURED" -- true then, false now, so it must not keep saying
        # it. These two are not host-staged: their payload never touches host
        # memory and their round counter lives in device memory precisely so
        # a graph replay advances it instead of reusing a stale value. The
        # one open question, whether the driver accepts
        # cudaLaunchCooperativeKernel from inside a stream capture, was
        # answered by bar1_graph_check.py's `grid` case on real cards.
        raise ValueError(
            f"SGLANG_BARLINK_TRANSPORT={transport!r} is capturable, but "
            f"{_GRAPH_ENABLE_ENV} is set to an off value in this "
            "environment, which takes it back out of the capturable set. "
            "Its capture was proven on 2026-08-01 (task #369, "
            "benchmark/bar1_graph_check.py, 10/10 including the cooperative "
            "`grid` case), so the release is on by default and this is a "
            f"deliberate opt-out. Either unset {_GRAPH_ENABLE_ENV} to let "
            "this transport be captured, or keep it off and pass "
            "--disable-cuda-graph to run eagerly -- the combination of an "
            "off switch and enabled graphs is the one thing that cannot be "
            "served."
        )
    raise ValueError(
        f"SGLANG_BARLINK_TRANSPORT={transport!r} is a host-staged transport "
        "(shm/gloo stage over CPU memory, ucx stages over pinned host "
        "buffers for RDMA; an unknown name falls back to the gloo plane): "
        "every collective synchronizes with the host, which is illegal "
        "inside a CUDA-graph capture. Pass --disable-cuda-graph, or use "
        "SGLANG_BARLINK_TRANSPORT=device (or =host), which run the collectives "
        "on the GPU and ARE capturable."
    )


def should_build_barlink(world_size: int) -> bool:
    """Whether GroupCoordinator should CONSTRUCT a barlink communicator for a
    group of this size, i.e. whether barlink OWNS that group's transport.

    Module-level for the same reason as the two predicates below: one
    definition. `GroupCoordinator.__init__` calls it, and so does the VRAM
    ledger's NCCL term (`sglang.srt.mem_ledger.nccl_transport`), which has to
    answer "does this launch build an NCCL communicator at all?" at ARGUMENT
    PARSE time, long before any GroupCoordinator exists. A ledger that
    re-implemented `envs.SGLANG_BARLINK.get() and world_size > 1` would keep
    pricing an NCCL term at 0 after this condition changed -- an under-charge
    that surfaces as an OOM, which is the exact failure class the ledger
    exists to remove.

    `barlink_comm is not None` is equivalent to this predicate rather than
    merely implied by it: the constructor below is not wrapped in a
    try/except, so a failure aborts the boot instead of silently leaving the
    attribute None.

    RANK-UNIFORM: both inputs (an env var that must be identical on every rank,
    and a group size every rank derives from the same CLI) are launch-uniform.
    """
    return bool(envs.SGLANG_BARLINK.get()) and world_size > 1


def should_build_pynccl(
    use_pynccl: bool, world_size: int, barlink_active: bool
) -> bool:
    """Whether GroupCoordinator should CONSTRUCT a PyNccl communicator.

    Module-level and importable ON PURPOSE: it is the single definition of this
    decision, used by __init__ and asserted on directly by the tests. A test
    that re-implemented the condition would keep passing after the real one was
    reverted -- which is exactly what happened while developing this, and is the
    same "checks something adjacent to the thing under test" failure this
    codebase keeps producing.

    `barlink_active` -> do not build. NCCL cannot span vendors: on a mixed group
    ncclCommInitRank segfaults inside the C library trying to form a world with
    a peer that has no NCCL. barlink reroutes the collectives but never prevented
    the CONSTRUCTION, because `use_pynccl` is independent of SGLANG_BARLINK.

    RANK-UNIFORM: every input is derived identically on every rank from the same
    CLI/env. A rank-divergent answer here would produce a hang rather than a
    crash -- quieter and worse.

    Flag OFF reduces exactly to the original `use_pynccl and world_size > 1`,
    so same-vendor rigs keep pynccl, which is their faster path.
    """
    return use_pynccl and world_size > 1 and not barlink_active


def should_build_custom_allreduce(
    use_custom_allreduce: bool, world_size: int, barlink_active: bool
) -> bool:
    """Whether GroupCoordinator should CONSTRUCT a CustomAllreduce.

    Same shape and same reason as `should_build_pynccl`, and module-level for
    the same reason: one definition, imported by the test rather than copied.

    `barlink_active` -> do not build. CustomAllreduce is a CUDA-IPC intra-node
    fast path; it cannot serve a group that spans vendors or hosts, which is
    precisely the group barlink exists for.

    THE FAILURE IT PREVENTS IS A HANG, NOT A CRASH, AND IT WAS MEASURED.
    `CustomAllreduce.__init__` returns EARLY, before any collective, when
    `ops.IS_CUSTOM_AR_AVAILABLE` is false -- i.e. wherever sgl_kernel's custom
    AR ops are absent (an sm75 build, a ROCm build). Where they ARE present the
    constructor calls `can_use_custom_all_reduce_with_nvlink`, which runs
    `in_the_same_node_as` -> `broadcast_object_list` over the WHOLE group.
    A group in which only some ranks have the ops therefore has some ranks
    inside a collective and some already past it: deadlock.

    Nordstern L0, TP=5 over two hosts, per-rank py-spy at the stall:

        ranks 0,1,2 (sm120/sm86, sgl_kernel present)
            broadcast_object_list -> in_the_same_node_as
            -> can_use_custom_all_reduce_with_nvlink
            -> CustomAllreduce.__init__ -> GroupCoordinator.__init__
        rank 3 (sm75) and rank 4 (gfx900), ops absent, already past it:
            all_reduce -> get_available_gpu_memory -> init_torch_distributed

    Note the trap in the existing code: the constructor is wrapped in
    `try/except` with a warning, which handles a rank that FAILS but not one
    that never arrives. An exception is rank-local; a missing participant is
    not.

    RANK-UNIFORM: every input is derived identically on every rank from the
    same CLI/env. `IS_CUSTOM_AR_AVAILABLE` -- the thing that actually diverged
    -- is rank-LOCAL and must not decide a collective.

    Flag OFF reduces exactly to the original `use_custom_allreduce and
    world_size > 1`, so same-vendor rigs keep custom all-reduce.
    """
    return use_custom_allreduce and world_size > 1 and not barlink_active


class GroupCoordinator:
    """
    PyTorch ProcessGroup wrapper for a group of processes.
    PyTorch ProcessGroup is bound to one specific communication backend,
        e.g. NCCL, Gloo, MPI, etc.
    GroupCoordinator takes charge of all the communication operations among
        the processes in the group. It can route the communication to
        a specific implementation (e.g. switch allreduce implementation
        based on the tensor size and cuda graph mode).
    """

    # available attributes:
    rank: int  # global rank
    ranks: List[int]  # global ranks in the group
    world_size: int  # size of the group
    # difference between `local_rank` and `rank_in_group`:
    # if we have a group of size 4 across two nodes:
    # Process | Node | Rank | Local Rank | Rank in Group
    #   0     |   0  |  0   |     0      |       0
    #   1     |   0  |  1   |     1      |       1
    #   2     |   1  |  2   |     0      |       2
    #   3     |   1  |  3   |     1      |       3
    local_rank: int  # local rank used to assign devices
    rank_in_group: int  # rank inside the group
    cpu_group: ProcessGroup  # group for CPU communication
    device_group: ProcessGroup  # group for device communication
    use_pynccl: bool  # a hint of whether to use PyNccl
    use_pymscclpp: bool  # a hint of whether to use PyMsccl
    use_custom_allreduce: bool  # a hint of whether to use CustomAllreduce
    use_torch_symm_mem_all_reduce: (
        bool  # a hint of whether to use TorchSymmMemAllReduce
    )
    use_message_queue_broadcaster: (
        bool  # a hint of whether to use message queue broadcaster
    )
    # communicators are only created for world size > 1
    pynccl_comm: Optional[Any]  # PyNccl communicator
    ca_comm: Optional[Any]  # Custom allreduce communicator
    torch_symm_mem_comm: Optional[Any]  # Torch symm mem communicator
    mq_broadcaster: Optional[Any]  # shared memory broadcaster

    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
        use_pynccl: bool,
        use_pymscclpp: bool,
        use_custom_allreduce: bool,
        use_torch_symm_mem_all_reduce: bool,
        use_hpu_communicator: bool,
        use_xpu_communicator: bool,
        use_npu_communicator: bool,
        use_message_queue_broadcaster: bool = False,
        group_name: Optional[str] = None,
        gloo_timeout: timedelta = timedelta(seconds=120 * 60),
        recovered_rank: bool = False,
    ):
        # Set group info
        group_name = group_name or "anonymous"
        self.unique_name = _get_unique_name(group_name)
        _register_group(self)

        # Collective-clock families for this group's dispatch sites (#588).
        # Built ONCE here, so an armed span costs an attribute read and no
        # string work on the hot path.
        _clock_families = collective_clock_families(group_name)
        (
            self._clock_family_all_reduce,
            self._clock_family_all_gather,
            self._clock_family_all_gatherv,
            self._clock_family_reduce_scatterv,
            self._clock_family_broadcast,
            self._clock_family_all_to_all,
        ) = _clock_families

        # Set rank info
        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.device_group = None
        self.cpu_group = None
        self.local_size = get_int_env_var("LOCAL_SIZE", 0)

        if is_cuda_alike():
            device_id = (
                0 if envs.SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS.get() else local_rank
            )
            self.device = torch.device(f"cuda:{device_id}")
        elif _is_npu:
            self.device = torch.device(f"npu:{local_rank}")
        elif _is_xpu:
            self.device = torch.device(f"xpu:{local_rank}")
        elif _is_musa:
            self.device = torch.device(f"musa:{local_rank}")
        else:
            self.device = torch.device("cpu")
        self.device_module = torch.get_device_module(self.device)

        for ranks in group_ranks:
            active_ranks = torch.ones(len(ranks), dtype=torch.int32, device=self.device)
            active_ranks_cpu = torch.ones(len(ranks), dtype=torch.int32)
            subgroup_timeout = _MODEL_PARALLEL_GROUP_TIMEOUT
            if "mooncake" in torch_distributed_backend:
                from mooncake.ep import MooncakeBackendOptions

                device_group = torch.distributed.new_group(
                    ranks,
                    backend="mooncake",
                    pg_options=MooncakeBackendOptions(active_ranks, recovered_rank),
                    timeout=subgroup_timeout,
                )
                cpu_group = torch.distributed.new_group(
                    ranks,
                    backend="mooncake-cpu",
                    pg_options=MooncakeBackendOptions(active_ranks_cpu, recovered_rank),
                    timeout=subgroup_timeout,
                )
            else:
                pg_options = get_torch_distributed_pg_options(group_name)
                device_group = torch.distributed.new_group(
                    ranks,
                    backend=torch_distributed_backend,
                    pg_options=pg_options,
                    timeout=subgroup_timeout,
                )
                # a group with `gloo` backend, to allow direct coordination
                # between processes through the CPU.
                #
                # The timeout follows `--dist-timeout` when one is set, the
                # same as the device group above. It used to be pinned to
                # the `gloo_timeout` default no matter what the operator
                # configured, and that is the defect behind #312: every
                # barlink handshake, `GroupCoordinator.barrier` and therefore
                # the CUDA-graph capture barrier run on THIS group, so a
                # rank killed during capture left the survivors waiting out
                # a hardcoded 7200 s that nothing could shorten. The same
                # divergence produced the hicache stall fixed in #259.
                # Unset keeps the previous 7200 s default exactly.
                cpu_group = torch.distributed.new_group(
                    ranks,
                    backend="gloo",
                    timeout=(
                        subgroup_timeout
                        if subgroup_timeout is not None
                        else gloo_timeout
                    ),
                )
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self.device_group = device_group
                self.cpu_group = cpu_group
                self.active_ranks = active_ranks
                self.active_ranks_cpu = active_ranks_cpu

        assert self.cpu_group is not None
        assert self.device_group is not None

        # #631: does this group have a WIRE at all? The census counts
        # collectives so the counts can be compared ACROSS ranks; a
        # world_size==1 group short-circuits every collective below
        # (``if self.world_size == 1: return input_``) and never touches
        # the wire, so its counts measure this rank's local work and
        # nothing that could pair with a peer. Comparing them across ranks
        # is therefore a category error, and it produced a real false
        # positive: under PP=3/TP=1 the size-1 "tp" group reported
        # ``tp.all_gather: counts [536, 1096, 1096]`` -- the stage ratio
        # 2,1,1 giving rank 0 a different local layer count, exactly as it
        # should (measured, window-2 boot 13, 2026-08-08).
        #
        # This EXCLUDES non-wire events from a wire census; it does not
        # weaken the check on any group that can actually desync. Computed
        # once, here, because world_size is only known after the loop
        # above -- an earlier read raised AttributeError inside __init__
        # and turned every group construction into a retry storm.
        self._census_wire = _CENSUS_ON and self.world_size > 1

        # #583: declare this group's families to the census from REPLICATED
        # config, at construction, before any of them can fire. Every rank
        # builds the same groups under the same names, so the census payload
        # width is fixed by configuration rather than by which collectives a
        # rank has happened to reach (#610).
        if self._census_wire:
            _CENSUS.declare_families(_clock_families)

        # Import communicators
        self.use_pynccl = use_pynccl
        self.use_pymscclpp = use_pymscclpp
        self.use_custom_allreduce = use_custom_allreduce
        self.use_torch_symm_mem_all_reduce = use_torch_symm_mem_all_reduce
        self.use_hpu_communicator = use_hpu_communicator
        self.use_xpu_communicator = use_xpu_communicator
        self.use_npu_communicator = use_npu_communicator
        self.use_message_queue_broadcaster = use_message_queue_broadcaster

        # Lazy import to avoid documentation build error
        from sglang.srt.distributed.device_communicators.custom_all_reduce import (
            dispatch_custom_allreduce,
        )
        from sglang.srt.distributed.device_communicators.pymscclpp import (
            PyMscclppCommunicator,
        )
        from sglang.srt.distributed.device_communicators.pynccl import (
            PyNcclCommunicator,
        )
        from sglang.srt.distributed.device_communicators.pynccl_allocator import (
            debug_check_symmetric_mempool,
            is_symmetric_memory_enabled,
            use_symmetric_memory,
        )
        from sglang.srt.distributed.device_communicators.torch_symm_mem import (
            TorchSymmMemCommunicator,
        )
        from sglang.srt.layers.dp_attention import is_allocation_symmetric

        self.is_symmetric_memory_enabled = is_symmetric_memory_enabled
        self.use_symmetric_memory = use_symmetric_memory
        self.is_allocation_symmetric = is_allocation_symmetric
        self.debug_check_symmetric_mempool = debug_check_symmetric_mempool
        if is_hip():
            from sglang.srt.distributed.device_communicators.quick_all_reduce import (
                QuickAllReduce,
                qr_rocm_arch_available,
            )

        # barlink (task #117): vendor-neutral host-staged collectives, for TP
        # groups that span GPUs without a common device collective library
        # (NCCL and RCCL cannot form a joint communicator). Constructed
        # BEFORE pynccl and consulted FIRST in every dispatch, so that when
        # the flag is on no collective reaches NCCL.
        #
        # Flag OFF (the default) leaves barlink_comm as None and every dispatch
        # seam falls through a single `is not None` check -- behavior is
        # byte-identical to stock sglang.
        #
        # It attaches to self.cpu_group (gloo), which sglang already builds
        # for every group, so this adds no process group and no new
        # collective beyond barlink's own startup calibration.
        self.barlink_comm: Optional[Any] = None
        if should_build_barlink(self.world_size):
            from sglang.srt.distributed.device_communicators.barlink import (
                BarlinkCommunicator,
            )

            # The CPU transports host-stage every collective: shm calls
            # cudaStreamSynchronize twice per op and spins on shm counters,
            # the gloo plane calls cudaEventSynchronize plus a gloo CPU
            # collective per chunk. All of that is ILLEGAL inside a CUDA-graph
            # capture. Until now that constraint lived only in the log line
            # below, so violating it surfaced as a bare
            # `cudaErrorStreamCaptureUnsupported` in the middle of capture --
            # the exact shape of the arm-E crash. Fail at startup, naming the
            # cause, instead.
            _enforce_cpu_transport_needs_eager(envs.SGLANG_BARLINK_TRANSPORT.get())
            # Through a LOCAL variable, and so is the state query further
            # down: `self.barlink_comm` may only be touched behind an
            # `is not None` (test_dispatch_seams_are_all_none_guarded pins
            # that, and rightly so -- it is what keeps the flag-off path
            # byte-identical).
            _comm = BarlinkCommunicator(
                cpu_group=self.cpu_group,
                device=self.device,
                group=self.unique_name,
            )
            self.barlink_comm = _comm
            # What was REQUESTED and what was ACHIEVED -- kept apart.
            #
            # Until now this line named the requested transport, whatever had
            # actually come of it. On the real model with SGLANG_UNEVEN_DCP
            # that meant: 'tp' brought the direct path up in 27 ms, 'dcp'
            # failed at the holder with ENOMEM and fell back to gloo -- and
            # both lines said "transport=bar1". Half of the number won from
            # that run was not a bar1 number at all. So the failure is now a
            # WARNING carrying the group name and the reason, and the success
            # says explicitly what is really running.
            requested = envs.SGLANG_BARLINK_TRANSPORT.get()
            state = getattr(_comm, "state", {}) or {}
            achieved = state.get("achieved", requested)
            if state.get("direct", True):
                logger.info(
                    "barlink enabled for group '%s': requested=%s, "
                    "ACHIEVED=%s. Every SGLANG_BARLINK* env must be identical "
                    "on all ranks; the host-staged transports (shm/gloo/ucx) "
                    "additionally require --disable-cuda-graph.",
                    self.unique_name,
                    requested,
                    achieved,
                )
            else:
                logger.warning(
                    "barlink group '%s': requested=%s, ACHIEVED=%s (%s: %s). "
                    "This group does NOT run over %s. A measurement from "
                    "this run is mixed and must not be reported as a "
                    "%s value.",
                    self.unique_name,
                    requested,
                    achieved,
                    state.get("stage", "?"),
                    state.get("reason", "?"),
                    requested,
                    requested,
                )

        # When barlink is active the pynccl communicator is NOT CONSTRUCTED --
        # not merely left unused.
        #
        # `use_pynccl` is independent of SGLANG_BARLINK: enabling barlink reroutes
        # the COLLECTIVES but never stopped PyNcclCommunicator from being built
        # first. On a same-vendor rig that construction is harmless. On a
        # vendor-mixed group it is fatal: ncclCommInitRank tries to form an
        # NCCL world with a peer that has no NCCL at all, and SEGFAULTS inside
        # the C library. Measured on the cross-vendor host -- both ranks logged
        # "barlink enabled ... (transport=gloo)" and rank 0 then died at
        #   pynccl_wrapper.py:404 in ncclCommInitRank
        #   parallel_state.py:420 in __init__
        # before the model was even loaded.
        #
        # RANK-UNIFORM by construction: the condition reads only
        # `envs.SGLANG_BARLINK` and `self.world_size`, both of which every rank
        # derives identically from the same CLI/env. That matters more than the
        # crash it prevents -- if one rank built pynccl and another did not,
        # the result would not be a segfault but a HANG, which is quieter and
        # worse.
        #
        # FLAG-OFF IS BYTE-IDENTICAL: with SGLANG_BARLINK unset, `_barlink_active`
        # is False and the condition reduces to the original
        # `use_pynccl and self.world_size > 1`. pynccl remains the faster path
        # on same-vendor rigs and is not given up -- the rule is
        # "barlink active -> no pynccl", not "no pynccl".
        #
        # Every consumer of self.pynccl_comm was checked to tolerate None:
        # the two sites that dereference it without a guard
        # (_all_reduce_out_place's assert, and the reduce_scatter path) are
        # both preceded by an `if self.barlink_comm is not None:` early return,
        # so they are unreachable when this is None. Otherwise the fix would
        # only relocate the crash.
        _barlink_active = self.barlink_comm is not None
        self.pynccl_comm: Optional[PyNcclCommunicator] = None
        if should_build_pynccl(use_pynccl, self.world_size, _barlink_active):
            # The ledger's NCCL term cannot be derived -- libnccl sizes these
            # buffers itself, outside the torch allocator -- so the only way it
            # ever reaches "priced" is a measurement taken right here, around
            # the one constructor that allocates them. Unarmed, the bracket is
            # two no-op calls. See mem_ledger/nccl_probe.py.
            from sglang.srt.mem_ledger.nccl_probe import measure_communicator_init

            with measure_communicator_init(self.unique_name, self.device.index):
                self.pynccl_comm = PyNcclCommunicator(
                    group=self.cpu_group,
                    device=self.device,
                )
        elif use_pynccl and self.world_size > 1:
            logger.info(
                "barlink is active for group '%s': skipping PyNccl "
                "communicator construction. NCCL cannot span vendors, and "
                "constructing it would abort in ncclCommInitRank on a "
                "vendor-mixed group.",
                self.unique_name,
            )

        self.pymscclpp_comm: Optional[PyMscclppCommunicator] = None
        if use_pymscclpp and self.world_size > 1:
            self.pymscclpp_comm = PyMscclppCommunicator(
                group=self.cpu_group,
                device=self.device,
            )

        self.ca_comm: Optional[Any] = None
        self.qr_comm: Optional[QuickAllReduce] = None
        if should_build_custom_allreduce(
            use_custom_allreduce, self.world_size, _barlink_active
        ):
            # Initialize a custom fast all-reduce implementation.
            try:
                CAClass = dispatch_custom_allreduce(
                    group=self.cpu_group,
                    device=self.device,
                )
                self.ca_comm = CAClass(
                    group=self.cpu_group,
                    device=self.device,
                )
            except Exception as e:
                logger.warning(
                    f"Setup Custom allreduce failed with {e}. To silence this "
                    "warning, specify --disable-custom-all-reduce explicitly."
                )

            if is_hip():
                try:
                    # Initialize a custom quick all-reduce implementation for AMD
                    # when rocm >= gfx942. Quick reduce is designed as a
                    # complement to custom allreduce.
                    # Based on quickreduce (https://github.com/mk1-project/quickreduce).
                    if qr_rocm_arch_available():
                        self.qr_comm = QuickAllReduce(
                            group=self.cpu_group, device=self.device
                        )
                except Exception as e:
                    logger.warning(f"Failed to initialize QuickAllReduce: {e}")

            # #195 (collective family): custom-allreduce enablement must be
            # rank-uniform, because it later gates group collectives (the
            # graph-buffer registration in ca_comm.capture()'s exit, and the
            # dispatch decision in all_reduce). The block we are in is entered
            # on a rank-uniform condition (should_build_custom_allreduce reads
            # only constructor args and env that must be launch-uniform), so
            # every rank of the group arrives here and the agreement below is
            # balanced.
            self._harmonize_ca_comm_enablement()
        elif use_custom_allreduce and self.world_size > 1:
            # barlink active. Skipping is the point: the constructor's
            # nvlink probe is a COLLECTIVE that only the ranks with
            # sgl_kernel's custom-AR ops would enter (measured: L0 TP=5
            # deadlocked with ranks 0-2 inside broadcast_object_list and
            # ranks 3-4 already past it).
            logger.info(
                "barlink is active for group '%s': skipping CustomAllreduce "
                "construction. Its NVLink probe is a group-wide collective "
                "that ranks without sgl_kernel's custom-AR ops never enter, "
                "so a vendor-mixed or multi-host group deadlocks there.",
                self.unique_name,
            )
        elif self.world_size > 1 and is_hip():
            logger.info("[AR] All-reduce call path: NCCL (custom AR disabled)")

        self.torch_symm_mem_comm: Optional[TorchSymmMemCommunicator] = None
        if self.use_torch_symm_mem_all_reduce and self.world_size > 1:
            self.torch_symm_mem_comm = TorchSymmMemCommunicator(
                group=self.cpu_group,
                device=self.device,
            )

        # Create communicator for other hardware backends
        from sglang.srt.distributed.device_communicators.hpu_communicator import (
            HpuCommunicator,
        )
        from sglang.srt.distributed.device_communicators.npu_communicator import (
            NpuCommunicator,
        )
        from sglang.srt.distributed.device_communicators.xpu_communicator import (
            XpuCommunicator,
        )

        self.hpu_communicator: Optional[HpuCommunicator] = None
        if use_hpu_communicator and self.world_size > 1:
            self.hpu_communicator = HpuCommunicator(group=self.device_group)

        self.xpu_communicator: Optional[XpuCommunicator] = None
        if use_xpu_communicator and self.world_size > 1:
            self.xpu_communicator = XpuCommunicator(group=self.device_group)

        self.npu_communicator: Optional[NpuCommunicator] = None
        if use_npu_communicator and self.world_size > 1:
            self.npu_communicator = NpuCommunicator(group=self.device_group)

        # Create message queue
        from sglang.srt.distributed.device_communicators.shm_broadcast import (
            MessageQueue,
        )

        self.mq_broadcaster: Optional[MessageQueue] = None
        if use_message_queue_broadcaster and self.world_size > 1 and not recovered_rank:
            # Recovered ranks create their mq_broadcaster in elastic_ep.py
            self.mq_broadcaster = MessageQueue.create_from_process_group(
                self.cpu_group, 1 << 22, 6
            )

    def __repr__(self):
        return (
            f"ranks={self.ranks} rank={self.rank} local_rank={self.local_rank} use_pynccl={self.use_pynccl} "
            f"device_group={self.device_group} cpu_group={self.cpu_group} unique_name={self.unique_name} "
            f"world_size={self.world_size} rank_in_group={self.rank_in_group}"
        )

    @property
    def first_rank(self):
        """Return the global rank of the first process in the group"""
        return self.ranks[0]

    @property
    def last_rank(self):
        """Return the global rank of the last process in the group"""
        return self.ranks[-1]

    @property
    def is_first_rank(self):
        """Return whether the caller is the first process in the group"""
        return self.rank == self.first_rank

    @property
    def is_last_rank(self):
        """Return whether the caller is the last process in the group"""
        return self.rank == self.last_rank

    @property
    def next_rank(self):
        """Return the global rank of the process that follows the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(rank_in_group + 1) % world_size]

    @property
    def prev_rank(self):
        """Return the global rank of the process that precedes the caller"""
        rank_in_group = self.rank_in_group
        world_size = self.world_size
        return self.ranks[(rank_in_group - 1) % world_size]

    def _harmonize_ca_comm_enablement(self):
        """#195 (collective family, sibling of #194/#94): force custom-allreduce
        enablement to the GROUP consensus, so no later collective is gated on a
        rank-local state.

        `ca_comm.disabled` (and `ca_comm is None` after a construction
        exception) is rank-local: sgl_kernel import success, the cached can_p2p
        verdict, and any constructor exception are all per-process facts. Yet
        that state later decides whether a rank enters GROUP collectives -- the
        graph-buffer registration in ca_comm.capture()'s exit and the custom-AR
        arm of all_reduce dispatch. If it diverges, the enabled ranks wait in a
        broadcast the disabled ranks never join (the #194 measurement: both
        GPUs 0% util, logs frozen, py-spy shows one rank inside
        broadcast_object_list) -- or worse, an enabled rank captures a custom-AR
        kernel that spins on peer flags a disabled peer never sets.

        Rule (same as _harmonize_cuda_graph_plan for graph plans): gather each
        rank's verdict once, at construction time -- the one point where every
        rank of the group is provably present -- and on divergence DISABLE for
        everyone, loudly. On a homogeneous group the gathered values are equal
        and nothing changes.

        Deliberately NOT freeing the disabled communicator's buffers here:
        v2's obj.free() takes the group and may itself collect, and only the
        ranks with a live object could enter it -- the same unbalanced shape
        this fix removes. Holding a few MB on a misconfigured rank is the
        balanced choice; the warning names the ranks so the misconfiguration
        gets fixed instead.
        """
        local_ok = self.ca_comm is not None and not getattr(
            self.ca_comm, "disabled", True
        )
        gathered: List[Optional[bool]] = [None] * self.world_size
        torch.distributed.all_gather_object(
            gathered, bool(local_ok), group=self.cpu_group
        )
        if all(gathered) or not any(gathered):
            return
        enabled_ranks = [r for r, ok in zip(self.ranks, gathered) if ok]
        disabled_ranks = [r for r, ok in zip(self.ranks, gathered) if not ok]
        logger.warning(
            "Custom allreduce enablement diverges across group '%s' "
            "(enabled on ranks %s, unavailable on ranks %s); disabling it on "
            "every rank so no collective is gated on a rank-local state. "
            "Fix the unavailable ranks (sgl_kernel build, P2P probe cache) or "
            "pass --disable-custom-all-reduce to make this deliberate.",
            self.unique_name,
            enabled_ranks,
            disabled_ranks,
        )
        if self.ca_comm is not None:
            self.ca_comm.disabled = True
            if hasattr(self.ca_comm, "original_disabled"):
                self.ca_comm.original_disabled = True

    @contextmanager
    def graph_capture(
        self,
        graph_capture_context: Optional[GraphCaptureContext] = None,
        stream=None,
    ):
        if graph_capture_context is None:
            if stream is None:
                stream = self.device_module.Stream()
            graph_capture_context = GraphCaptureContext(stream)
        else:
            stream = graph_capture_context.stream
        # We don't need the context of custom quick allreduce because the ipc access
        # is already collected in init() and we can capture the quick allreduce directly.
        ca_comm = self.ca_comm
        maybe_ca_context = nullcontext() if ca_comm is None else ca_comm.capture()

        # ensure all initialization operations complete before attempting to
        # capture the graph on another stream
        curr_stream = get_current_device_stream_fast()
        if curr_stream != stream:
            stream.wait_stream(curr_stream)

        with self.device_module.stream(stream), maybe_ca_context:
            # In graph mode, we have to be very careful about the collective
            # operations. The current status is:
            #     allreduce \ Mode   |  Eager  |  Graph  |
            # --------------------------------------------
            # quick allreduce        | enabled | enabled |
            # custom allreduce       | enabled | enabled |
            # PyNccl                 | disabled| enabled |
            # PyMscclpp              | disabled| enabled |
            # TorchSymmMem           | disabled| enabled |
            # torch.distributed      | enabled | disabled|
            #
            # Note: When custom quick allreduce is enabled, a runtime check
            #  will be performed. If the tensor size is too small, it will
            #  automatically fall back to the next available option.
            # Note that custom allreduce will have a runtime check, if the
            #  tensor size is too large, it will fallback to the next
            #  available option.
            # Note that the PyMsccl needs to register the tensor in ahead,
            #  which will introduce large overhead in the eager case,
            #  therefore it is only supported in the graph case.
            # In summary: We select the appropriate allreduce method for
            #  each mode based on the algorithm order in the table and
            #  their usage conditions.
            pynccl_comm = self.pynccl_comm
            maybe_pynccl_context: Any
            if not pynccl_comm:
                maybe_pynccl_context = nullcontext()
            else:
                maybe_pynccl_context = pynccl_comm.change_state(enable=True)

            pymscclpp_comm = self.pymscclpp_comm
            maybe_pymscclpp_context: Any
            if not pymscclpp_comm:
                maybe_pymscclpp_context = nullcontext()
            else:
                maybe_pymscclpp_context = pymscclpp_comm.change_state(enable=True)
            with maybe_pynccl_context, maybe_pymscclpp_context:
                yield graph_capture_context

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """
        User-facing all-reduce function before we actually call the
        all-reduce operation.

        We need this because Dynamo does not support passing an arbitrary
        object (`self` in this case) to a custom op. We need to pass the
         group name as a string, and then look up the group coordinator from
         the group name, dispatch the all-reduce operation to the group
         coordinator.

        In addition, PyTorch custom ops do not support mutation or returning
        a new tensor in the same op. So we need to figure out if the op is
        in-place or out-of-place ahead of time.
        """
        if _COLLECTIVE_CLOCK.armed:
            # Time this collective for the per-rank compute/wait split, then
            # re-enter with the clock disarmed: the body runs exactly once,
            # and a collective built out of other collectives is counted once
            # rather than once per level.
            with _COLLECTIVE_CLOCK.span(self._clock_family_all_reduce):
                return self.all_reduce(input_)

        # #583 collective census. Placed AFTER the clock's re-entry guard so
        # every collective is counted exactly once: the armed path re-enters
        # with the clock disarmed and falls through to here. Gated on
        # _census_wire because the size-1 bypass sits immediately below:
        # counting a call that returns without touching the wire would put
        # a purely local event into a cross-rank comparison (#631).
        if self._census_wire:
            _CENSUS.bump(self._clock_family_all_reduce)

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_

        if input_.is_cpu:
            if is_shm_available(input_.dtype, self.world_size, self.local_size):
                torch.ops.sgl_kernel.shm_allreduce(input_, REDUCE_OP_SUM)
            else:
                torch.distributed.all_reduce(input_, group=self.device_group)
            return input_

        # barlink must return BEFORE the Dynamo custom-op machinery below
        # (inplace_all_reduce / outplace_all_reduce). Those ops exist to keep
        # the NCCL call opaque to torch.compile; letting barlink be decomposed
        # through them would split the host-staging schedule across graph
        # segments. Out-of-place, matching the outplace contract.
        if self.barlink_comm is not None:
            return self.barlink_comm.all_reduce(input_)

        if self.hpu_communicator is not None and not self.hpu_communicator.disabled:
            return self.hpu_communicator.all_reduce(input_)

        if self.xpu_communicator is not None and not self.xpu_communicator.disabled:
            # Route through inplace_all_reduce custom op so Dynamo treats this as
            # an opaque call and does not decompose it into _c10d_functional primitives
            # (which invoke sycl_event.wait() and break XPU graph capture).
            # Keeps the operation in-place; the all-reduce is performed by
            # _all_reduce_in_place, which for XPU falls through to
            # torch.distributed.all_reduce on self.device_group (the same group
            # used by xpu_communicator).
            inplace_all_reduce(input_, group_name=self.unique_name)
            return input_

        if self.npu_communicator is not None and not self.npu_communicator.disabled:
            return self.npu_communicator.all_reduce(input_)

        should_use_pymscclpp_allreduce = (
            self.pymscclpp_comm is not None
            and self.pymscclpp_comm.should_mscclpp_allreduce(input_)
        )
        if (
            self.pynccl_comm is not None
            and self.is_symmetric_memory_enabled()
            and not should_use_pymscclpp_allreduce
        ):
            self.debug_check_symmetric_mempool(self, {"input": input_}, "all_reduce")
            with self.pynccl_comm.change_state(enable=True):
                self.pynccl_comm.all_reduce(input_)
                return input_

        outplace_all_reduce_method = None
        if (
            self.ca_comm is not None
            and not self.ca_comm.disabled
            and not should_use_pymscclpp_allreduce
            and self.ca_comm.should_custom_ar(input_)
        ):
            outplace_all_reduce_method = "ca"
        elif (
            self.qr_comm is not None
            and not self.qr_comm.disabled
            and self.qr_comm.should_quick_allreduce(input_)
        ):
            outplace_all_reduce_method = "qr"
        elif self.pymscclpp_comm is not None and should_use_pymscclpp_allreduce:
            outplace_all_reduce_method = "pymscclpp"
        elif (
            self.torch_symm_mem_comm is not None
            and not self.torch_symm_mem_comm.disabled
            and self.torch_symm_mem_comm.should_torch_symm_mem_allreduce(input_)
        ):
            outplace_all_reduce_method = "torch_symm_mem"
        elif is_in_tc_piecewise_cuda_graph() and self.pynccl_comm is not None:
            # For piecewise cuda graph, we use pynccl outplace allreduce
            outplace_all_reduce_method = "pynccl"
        if outplace_all_reduce_method is not None:
            return outplace_all_reduce(
                input_,
                group_name=self.unique_name,
                outplace_all_reduce_method=outplace_all_reduce_method,
            )
        else:
            inplace_all_reduce(input_, group_name=self.unique_name)
            return input_

    def quant_all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """
        User-facing quant-all-reduce function similar to all-reduce. (NPU support only)
        """
        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_

        if self.npu_communicator is not None and not self.npu_communicator.disabled:
            return self.npu_communicator.quant_all_reduce(input_)
        else:
            inplace_all_reduce(input_, group_name=self.unique_name)
            return input_

    def fused_allreduce_rmsnorm(
        self,
        input_: torch.Tensor,
        residual_inp_: torch.Tensor,
        weight_: torch.Tensor,
        eps: float,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Attempt fused all-reduce + RMSNorm via custom all-reduce communicator. ROCm/HIP Only"""
        ca_comm = self.ca_comm
        if ca_comm is None or getattr(ca_comm, "disabled", True):
            return None

        # Prefer communicator-native fused API when provided.
        if hasattr(ca_comm, "fused_allreduce_rmsnorm"):
            try:
                return ca_comm.fused_allreduce_rmsnorm(
                    input_, residual_inp_, weight_, eps
                )
            except Exception:
                # Fall back to custom_fused_ar_rms path below.
                pass

        if not hasattr(ca_comm, "custom_fused_ar_rms"):
            return None

        # 1-stage vs 2-stage selection for fused AR+RMSNorm:
        # The 1-stage kernel launches one block per token and is capped at
        # 80 tokens (kMaxBlocks).  Guard with a byte threshold so large
        # prefill batches fall through to the 2-stage kernel instead of
        # hitting a runtime error.  AITER's C++ dispatch already gates
        # which hidden_dims have valid 1-stage support.
        if envs.SGLANG_USE_1STAGE_ALLREDUCE.is_set():
            use_1stage_ar = envs.SGLANG_USE_1STAGE_ALLREDUCE.get()
        else:
            total_bytes = input_.numel() * input_.element_size()
            use_1stage_ar = total_bytes <= 128 * 1024

        if (
            getattr(ca_comm, "_IS_CAPTURING", False)
            and not torch.cuda.is_current_stream_capturing()
            and is_in_tc_piecewise_cuda_graph()
        ):
            if not hasattr(ca_comm, "fused_ar_rms"):
                return None
            return ca_comm.fused_ar_rms(
                input_,
                residual_inp_,
                w=weight_,
                eps=eps,
                registered=False,
                use_1stage=use_1stage_ar,
            )
        fused_outputs = ca_comm.custom_fused_ar_rms(
            input_,
            residual_inp_,
            weight_,
            eps,
            use_1stage_ar,
        )
        return fused_outputs

    def _all_reduce_out_place(
        self, input_: torch.Tensor, outplace_all_reduce_method: str
    ) -> torch.Tensor:
        ca_comm = self.ca_comm
        qr_comm = self.qr_comm
        pymscclpp_comm = self.pymscclpp_comm
        torch_symm_mem_comm = self.torch_symm_mem_comm
        pynccl_comm = self.pynccl_comm
        assert any([qr_comm, ca_comm, pymscclpp_comm, torch_symm_mem_comm, pynccl_comm])
        if outplace_all_reduce_method == "ca":
            assert not ca_comm.disabled
            out = ca_comm.custom_all_reduce(input_)
        elif outplace_all_reduce_method == "qr":
            assert not qr_comm.disabled
            out = qr_comm.quick_all_reduce(input_)
        elif outplace_all_reduce_method == "torch_symm_mem":
            assert not torch_symm_mem_comm.disabled
            out = torch_symm_mem_comm.all_reduce(input_)
        elif outplace_all_reduce_method == "pymscclpp":
            assert not pymscclpp_comm.disabled
            out = pymscclpp_comm.all_reduce(input_)
        elif outplace_all_reduce_method == "pynccl":
            with pynccl_comm.change_state(enable=True):
                out = pynccl_comm.outplace_all_reduce(input_)
        assert out is not None
        return out

    def _all_reduce_in_place(self, input_: torch.Tensor) -> None:
        pynccl_comm = self.pynccl_comm
        torch_symm_mem_comm = self.torch_symm_mem_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.all_reduce(input_)
        elif (
            torch_symm_mem_comm is not None
            and not torch_symm_mem_comm.disabled
            and torch_symm_mem_comm.should_torch_symm_mem_allreduce(input_)
        ):
            torch_symm_mem_comm.all_reduce(input_, out=input_)
        else:
            torch.distributed.all_reduce(input_, group=self.device_group)

    def reduce_scatter_along_dim(
        self, input_: torch.Tensor, dim: int = -1
    ) -> torch.Tensor:
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )

        # barlink handles the dim/movedim bookkeeping itself and must not go
        # through the symmetric-memory allocator below (which is a NCCL/
        # pynccl mechanism).
        if self.barlink_comm is not None:
            return self.barlink_comm.reduce_scatter(input_, dim)

        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        with self.use_symmetric_memory(self):
            # TODO: make sure whether tensor layout affects nccl reduce_scatter
            # Note: This will produce an incorrect answer if we don't make
            # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
            input_tensor = input_.movedim(dim, 0).contiguous()

        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        with self.use_symmetric_memory(self):
            output_tensor = torch.empty(
                output_shape,
                dtype=input_tensor.dtype,
                device=input_tensor.device,
            )

        self.reduce_scatter_tensor(output_tensor, input_tensor)

        # Reshape before returning
        return output_tensor.movedim(0, dim)

    def _reduce_scatter_tensor(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
    ) -> torch.Tensor:
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and (
            not pynccl_comm.disabled or self.is_symmetric_memory_enabled()
        ):
            self.debug_check_symmetric_mempool(
                self, {"output": output, "input": input}, "reduce_scatter_tensor"
            )
            with pynccl_comm.change_state(enable=True):
                pynccl_comm.reduce_scatter(output, input)
        else:
            torch.distributed.reduce_scatter_tensor(
                output, input, group=self.device_group
            )
        return output

    def _barlink_unsupported(self, op: str) -> None:
        """Fail fast on a collective barlink does not implement.

        On a mixed-vendor group there is no NCCL fallback to silently take:
        NCCL and RCCL cannot form a joint communicator, so letting the call
        through does not degrade performance, it deadlocks (or asserts deep
        inside pynccl). Raising here names the op and the caller instead of
        surfacing as a hang minutes later.

        Rank-uniform by construction: all ranks issue the same collective
        sequence (SPMD), so every rank raises on the same call.
        """
        raise NotImplementedError(
            f"barlink does not implement {op!r}. SGLANG_BARLINK routes TP "
            f"collectives over the vendor-neutral host-staged path, which "
            f"currently covers all_reduce, all_gather(_into_tensor), "
            f"reduce_scatter(_tensor), all_to_all_single (equal split and "
            f"the split-size form via all_to_all_single_v) and broadcast. "
            f"{op!r} would fall back "
            f"to NCCL, which cannot span a mixed-vendor group -- that is a "
            f"deadlock, not a slowdown. Either avoid the feature that calls "
            f"{op!r} (uneven/variable-size collectives are the usual "
            f"source) or extend BarlinkCommunicator."
        )

    def reduce_scatter_tensor(self, output: torch.Tensor, input: torch.Tensor):
        if self.barlink_comm is not None:
            return self.barlink_comm.reduce_scatter_tensor(output, input)
        if _is_npu:
            self._reduce_scatter_tensor(output, input)
        elif self._maybe_aiter_reduce_scatter(output, input):
            return
        else:
            reg_reduce_scatter_tensor(output, input, group_name=self.unique_name)

    def _has_aiter_custom_reduce_scatter(self) -> bool:
        ca_comm = self.ca_comm
        return (
            ca_comm is not None
            and not getattr(ca_comm, "disabled", True)
            and hasattr(ca_comm, "should_custom_ar")
            and hasattr(ca_comm, "reduce_scatter")
        )

    def _maybe_aiter_reduce_scatter(
        self, output: torch.Tensor, input: torch.Tensor
    ) -> bool:
        # Aiter custom reduce-scatter (ROCm). Mirrors `_all_gather_into_tensor`'s
        # custom all-gather path: an equal-chunk (no variable sizes) reduce-scatter
        # using the registered symmetric-memory buffers, which is faster than the
        # generic RCCL kernel for the small, latency-bound decode collective.
        # Gated by SGLANG_DP_USE_REDUCE_SCATTER. Falls back (returns False)
        # for non-ROCm / unsupported shape/size/topology so the caller uses RCCL.
        if not (
            is_hip()
            and envs.SGLANG_DP_USE_REDUCE_SCATTER.get()
            and self._has_aiter_custom_reduce_scatter()
            and input.is_contiguous()
            and output.is_contiguous()
            and input.dtype in (torch.float32, torch.float16, torch.bfloat16)
        ):
            return False
        ca_comm = self.ca_comm
        # input is the full (pre-reduce) buffer; should_custom_ar bounds its size.
        if not ca_comm.should_custom_ar(input):
            return False
        # Equal-chunk only: input rows must split evenly into world_size chunks
        # matching the per-rank output rows.
        if input.shape[0] != output.shape[0] * self.world_size:
            return False
        if getattr(ca_comm, "_IS_CAPTURING", False):
            if torch.cuda.is_current_stream_capturing():
                ca_comm.reduce_scatter(input, output, registered=True)
            elif is_in_tc_piecewise_cuda_graph():
                ca_comm.reduce_scatter(input, output, registered=False)
            else:
                # True CUDA graph warmup: avoid a different host collective.
                output.zero_()
            return True
        ca_comm.reduce_scatter(input, output, registered=False)
        return True

    def _all_to_all_single(self, output: torch.Tensor, input: torch.Tensor) -> None:
        # barlink first. Without this branch, all_to_all_single went
        # straight to self.device_group while SGLANG_BARLINK was on -- that
        # is, to exactly the NCCL that _barlink_unsupported above names as
        # the deadlock cause on a group spanning two vendors. It had no
        # consequence so far because the method has no caller; no
        # consequence is not the same as correct, and a hang does not
        # announce itself with an error message.
        if self.barlink_comm is not None:
            self.barlink_comm.all_to_all_single(output, input)
            return
        torch.distributed.all_to_all_single(output, input, group=self.device_group)

    def all_to_all_single(self, output: torch.Tensor, input: torch.Tensor):
        if self._census_wire:  # #583/#631: see all_reduce for the placement rule.
            _CENSUS.bump(self._clock_family_all_to_all)
        if self.world_size == 1:
            output.copy_(input)
            return
        # As with all_reduce: barlink returns BEFORE the Dynamo custom-op
        # machinery. The branch in _all_to_all_single stays anyway -- it
        # catches the route through the registered op.
        if self.barlink_comm is not None:
            self.barlink_comm.all_to_all_single(output, input)
            return
        reg_all_to_all_single(output, input, group_name=self.unique_name)

    def all_to_all_single_v(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        output_split_sizes: Optional[List[int]] = None,
        input_split_sizes: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """all_to_all_single with uneven split sizes (rows, not bytes).

        The shape MoE needs: the token count per expert varies, so the block
        size per destination rank varies with it. ``output_split_sizes=None``
        derives the receive counts from the send counts -- an all_gather of
        the counts, the way DeepEP does it before the dispatch
        (``get_dispatch_layout`` -> ``num_tokens_per_rank``).

        DELIBERATELY NOT routed through ``reg_all_to_all_single``: the
        registered custom op has a fixed signature without split sizes, and
        extending it by ``Optional[List[int]]`` would have changed the
        existing op's schema contract. This form is therefore not opaque to
        Dynamo; whoever needs it under torch.compile has to graph-break
        around it.
        """
        # Same family as the even form: what the census compares is how many
        # times each rank entered this wire family, and a rank that skips an
        # uneven a2a has skipped an a2a.
        if self._census_wire:  # #583/#631: see all_reduce for the placement rule.
            _CENSUS.bump(self._clock_family_all_to_all)
        if self.world_size == 1:
            output.copy_(input)
            return output
        if self.barlink_comm is not None:
            return self.barlink_comm.all_to_all_single(
                output, input, output_split_sizes, input_split_sizes
            )
        torch.distributed.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=self.device_group,
        )
        return output

    def reduce_scatter(
        self,
        output: torch.Tensor,
        input_list: List[torch.Tensor],
    ) -> None:
        if self.barlink_comm is not None:
            self._barlink_unsupported("reduce_scatter(output, input_list)")
        # TODO(ch-wan): support other backends
        torch.distributed.reduce_scatter(output, input_list, group=self.device_group)
        return output

    def reduce_scatterv(
        self,
        input_: torch.Tensor,
        output: Optional[torch.Tensor] = None,
        sizes: Optional[List[int]] = None,
    ) -> torch.Tensor:
        if _COLLECTIVE_CLOCK.armed:
            # See all_reduce for why this re-enters.
            with _COLLECTIVE_CLOCK.span(self._clock_family_reduce_scatterv):
                return self.reduce_scatterv(input_, output=output, sizes=sizes)
        if self._census_wire:  # #583/#631: see all_reduce for the placement rule.
            _CENSUS.bump(self._clock_family_reduce_scatterv)
        if self.barlink_comm is not None:
            self._barlink_unsupported("reduce_scatterv")
        world_size = self.world_size
        pynccl_comm = self.pynccl_comm

        with pynccl_comm.change_state(enable=True):
            assert pynccl_comm is not None and not pynccl_comm.disabled, (
                "pynccl is required for reduce_scatterv"
            )

            if sizes is not None:
                assert len(sizes) == world_size
                assert input_.shape[0] == sum(sizes)
                chunk_size = sizes[self.rank_in_group]
            else:
                assert input_.shape[0] % world_size == 0
                chunk_size = input_.shape[0] // world_size
            output_shape = (chunk_size,) + input_.shape[1:]

            if output is None:
                output = torch.empty(
                    output_shape, dtype=input_.dtype, device=input_.device
                )
            else:
                assert output.shape == output_shape

            pynccl_comm.reduce_scatter(output, input_, sizes=sizes)
            return output

    def _all_gather_into_tensor(self, output: torch.Tensor, input: torch.Tensor):
        # Aiter custom all-gather (ROCm). Set SGLANG_USE_AITER_AG=0 to disable.
        # Aiter's should_custom_ag still owns shape/layout validation:
        # 16B alignment, weak-contiguous, supported topology, and per-rank
        # size <= max_size/(world*2).
        # On a hit, writes directly into the caller's pre-allocated `output` via
        # all_gather_reg during CUDA-graph capture, and all_gather_unreg
        # under torch_memory_saver and other paths.
        ca_comm = self.ca_comm
        if (
            is_hip()
            and envs.SGLANG_USE_AITER_AG.get()
            and self._has_aiter_custom_all_gather()
            and input.is_contiguous()
            and output.is_contiguous()
            and input.dtype in (torch.float32, torch.float16, torch.bfloat16)
            and ca_comm.should_custom_ag(input)
        ):
            if getattr(ca_comm, "_IS_CAPTURING", False):
                if torch.cuda.is_current_stream_capturing():
                    if envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get():
                        ca_comm.all_gather_unreg(input, out=output, dim=0)
                    else:
                        ca_comm.all_gather_reg(input, out=output, dim=0)
                elif is_in_tc_piecewise_cuda_graph():
                    ca_comm.all_gather_unreg(input, out=output, dim=0)
                else:
                    # True CUDA graph warmup: avoid a different host collective.
                    output.zero_()
                return
            else:
                ca_comm.all_gather_unreg(input, out=output, dim=0)
                return

        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and (
            not pynccl_comm.disabled or self.is_symmetric_memory_enabled()
        ):
            self.debug_check_symmetric_mempool(
                self, {"output": output}, "all_gather_into_tensor"
            )
            with pynccl_comm.change_state(enable=True):
                pynccl_comm.all_gather(output, input)
        else:
            torch.distributed.all_gather_into_tensor(
                output, input, group=self.device_group
            )

    def _has_aiter_custom_all_gather(self) -> bool:
        if self._deterministic_collectives_enabled():
            return False
        ca_comm = self.ca_comm
        return (
            ca_comm is not None
            and not getattr(ca_comm, "disabled", True)
            and hasattr(ca_comm, "should_custom_ag")
            and hasattr(ca_comm, "all_gather_reg")
            and hasattr(ca_comm, "all_gather_unreg")
        )

    @staticmethod
    def _deterministic_collectives_enabled() -> bool:
        if envs.SGLANG_USE_1STAGE_ALLREDUCE.is_set():
            return envs.SGLANG_USE_1STAGE_ALLREDUCE.get()
        return envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()

    def all_gather_into_tensor(self, output: torch.Tensor, input: torch.Tensor):
        if self.barlink_comm is not None:
            return self.barlink_comm.all_gather_into_tensor(output, input)
        if _is_npu:
            self._all_gather_into_tensor(output, input)
        else:
            # XPU and CUDA both go through reg_all_gather_into_tensor (custom_op) to
            # stay opaque to Dynamo. Calling torch.distributed.all_gather_into_tensor
            # directly causes Dynamo to rewrite it as _c10d_functional.all_gather_into_tensor
            # + wait_tensor, which invokes sycl_event.wait() and breaks XPU graph capture.
            reg_all_gather_into_tensor(output, input, group_name=self.unique_name)

    def cp_all_gather_into_tensor_async(
        self, output: torch.Tensor, input: torch.Tensor, stream: torch.cuda.Stream
    ):
        """
        Implement an asynchronous `allgather` operation on a specified stream.
        (the default `torch.distributed.all_gather_into_tensor` will trigger event synchronization),
        eliminating the CPU-side launch-kernel blocking issue caused by synchronization problems.
        The specific implementation uses the interface provided by pynccl to remove the synchronization logic of events.
        """
        pynccl_comm = self.pynccl_comm
        # Under barlink the pynccl fast path must not be taken even when a
        # pynccl communicator happens to exist: route to the (barlink-aware)
        # synchronous form instead. The async-stream optimization is a NCCL
        # feature and has no host-staged equivalent.
        if self.barlink_comm is not None or pynccl_comm is None or pynccl_comm.disabled:
            self.all_gather_into_tensor(output, input)
        else:
            pynccl_comm.cp_all_gather_into_tensor(output, input, stream=stream)

    def all_gather(
        self,
        input_: torch.Tensor,
        dim: int = -1,
        output_tensor_list: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if _COLLECTIVE_CLOCK.armed:
            # See all_reduce for why this re-enters.
            with _COLLECTIVE_CLOCK.span(self._clock_family_all_gather):
                return self.all_gather(
                    input_, dim=dim, output_tensor_list=output_tensor_list
                )
        if self._census_wire:  # #583/#631: see all_reduce for the placement rule.
            _CENSUS.bump(self._clock_family_all_gather)
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            if output_tensor_list is not None:
                logger.warning(
                    "Performing in-place all-gather with a group size of 1. "
                    "This may be unnecessary; consider bypassing it for better efficiency."
                )
                output_tensor_list[0].copy_(input_)
                return None
            else:
                return input_

        if output_tensor_list is not None:
            if self.barlink_comm is not None:
                self._barlink_unsupported("all_gather(output_tensor_list=...)")
            # TODO(ch-wan): support other backends
            return torch.distributed.all_gather(
                output_tensor_list, input_, group=self.device_group
            )

        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )

        # barlink: vendor-neutral path, ahead of every device-specific
        # communicator. The output_tensor_list form above is left on NCCL in
        # v1 (out-parameter variant, see PLAN_barlink_port.md section 5).
        if self.barlink_comm is not None:
            return self.barlink_comm.all_gather(input_, dim)

        # For HPUs, use HPU communicator.
        hpu_comm = self.hpu_communicator
        if hpu_comm is not None and not hpu_comm.disabled:
            return hpu_comm.all_gather(input_, dim)

        # For NPUs, use NPU communicator.
        npu_comm = self.npu_communicator
        if npu_comm is not None and not npu_comm.disabled:
            return npu_comm.all_gather(input_, dim)

        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        input_size = input_.size()
        # NOTE: we have to use concat-style all-gather here,
        # stack-style all-gather has compatibility issues with
        # torch.compile . see https://github.com/pytorch/pytorch/issues/138795
        output_size = (input_size[0] * world_size,) + input_size[1:]
        # Allocate output tensor.
        with self.use_symmetric_memory(
            self, disabled=not self.is_allocation_symmetric()
        ):
            output_tensor = torch.empty(
                output_size, dtype=input_.dtype, device=input_.device
            )

        # All-gather.
        if input_.is_cpu:
            if is_shm_available(input_.dtype, self.world_size, self.local_size):
                return torch.ops.sgl_kernel.shm_allgather(input_, dim)
            else:
                torch.distributed.all_gather_into_tensor(
                    output_tensor, input_, group=self.device_group
                )
        else:
            self.all_gather_into_tensor(output_tensor, input_)

        # Reshape
        output_tensor = output_tensor.reshape((world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim] + (world_size * input_size[dim],) + input_size[dim + 1 :]
        )
        return output_tensor

    def all_gatherv(
        self,
        input_: Union[torch.Tensor, List[torch.Tensor]],
        sizes: Optional[List[int]] = None,
        output: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Supports varying sizes per rank and input tensor list.
        `sizes`: a list of len(world_size) with the number of items per rank to gather.
        `output`: optional pre-allocated destination buffer (single-tensor input only).
            When given, NCCL writes the gathered result directly into it, avoiding an
            extra output allocation + caller-side copy.
        """
        if _COLLECTIVE_CLOCK.armed:
            # See all_reduce for why this re-enters.
            with _COLLECTIVE_CLOCK.span(self._clock_family_all_gatherv):
                return self.all_gatherv(input_, sizes=sizes, output=output)
        if self._census_wire:  # #583/#631: see all_reduce for the placement rule.
            _CENSUS.bump(self._clock_family_all_gatherv)
        if self.barlink_comm is not None:
            self._barlink_unsupported("all_gatherv")
        world_size = self.world_size
        pynccl_comm = self.pynccl_comm

        with pynccl_comm.change_state(enable=True):
            assert pynccl_comm is not None and not pynccl_comm.disabled, (
                "pynccl is required for all_gatherv"
            )

            def _all_gather_allocate_output(
                input_: torch.Tensor,
                sizes: Optional[List[int]] = None,
                output: Optional[torch.Tensor] = None,
            ):
                input_size = input_.size()
                if sizes is not None:
                    assert len(sizes) == world_size
                    assert input_.shape[0] == sizes[self.rank_in_group]
                    output_size = (sum(sizes),) + input_size[1:]
                    # 'sizes' is not needed if all inputs in the same group have the same shape
                    if all(s == sizes[0] for s in sizes):
                        sizes = None
                else:
                    output_size = (input_size[0] * world_size,) + input_size[1:]
                if output is not None:
                    assert tuple(output.shape) == tuple(output_size), (
                        f"all_gatherv output buffer shape {tuple(output.shape)} "
                        f"!= expected {tuple(output_size)}"
                    )
                    return output, sizes
                # Allocate output tensor.
                with self.use_symmetric_memory(self, disabled=sizes is not None):
                    output_tensor = torch.empty(
                        output_size, dtype=input_.dtype, device=input_.device
                    )
                return output_tensor, sizes

            single_input = isinstance(input_, torch.Tensor)
            if single_input:
                input_ = [input_]
            elif output is not None:
                raise ValueError("all_gatherv `output` requires a single-tensor input")

            output_list = []
            size_list = []
            for inp in input_:
                output_tensor, s = _all_gather_allocate_output(
                    inp, sizes=sizes, output=output
                )
                output_list.append(output_tensor)
                size_list.append(s)

            pynccl_comm.group_start()
            for i, inp in enumerate(input_):
                pynccl_comm.all_gather(output_list[i], inp, sizes=size_list[i])
            pynccl_comm.group_end()

            return output_list

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> Optional[torch.Tensor]:
        """
        NOTE: We assume that the input tensor is on the same device across
        all the ranks.
        NOTE: `dst` is the local rank of the destination rank.
        """
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        if self.xpu_communicator is not None and not self.xpu_communicator.disabled:
            return self.xpu_communicator.gather(input_, self.rank_in_group, dst, dim)
        # Allocate output tensor.
        if self.rank_in_group == dst:
            gather_list = [torch.empty_like(input_) for _ in range(world_size)]
        else:
            gather_list = None
        # Gather.
        torch.distributed.gather(
            input_, gather_list, dst=self.ranks[dst], group=self.device_group
        )
        if self.rank_in_group == dst:
            output_tensor = torch.cat(gather_list, dim=dim)
        else:
            output_tensor = None
        return output_tensor

    def broadcast(self, input_: torch.Tensor, src: int = 0):
        """Broadcast the input tensor.
        NOTE: `src` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        if self._census_wire:  # #583/#631: see all_reduce for the placement rule.
            _CENSUS.bump(self._clock_family_broadcast)

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_
        # barlink: broadcast over the gloo cpu_group instead of the NCCL
        # device_group. In-place, like the stock path.
        if self.barlink_comm is not None:
            return self.barlink_comm.broadcast(input_, src)
        # Broadcast.
        torch.distributed.broadcast(
            input_, src=self.ranks[src], group=self.device_group
        )
        return input_

    def broadcast_object(self, obj: Optional[Any] = None, src: int = 0):
        """Broadcast the input object.
        NOTE: `src` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return obj
        if self.mq_broadcaster is not None:
            assert src == 0, "Message queue broadcaster only supports src=0"
            return self.mq_broadcaster.broadcast_object(obj)
        if self.rank_in_group == src:
            torch.distributed.broadcast_object_list(
                [obj], src=self.ranks[src], group=self.cpu_group
            )
            return obj
        else:
            recv = [None]
            torch.distributed.broadcast_object_list(
                recv, src=self.ranks[src], group=self.cpu_group
            )
            return recv[0]

    def broadcast_object_list(
        self, obj_list: List[Any], src: int = 0, group: Optional[ProcessGroup] = None
    ):
        """Broadcast the input object list.
        NOTE: `src` is the local rank of the source rank.
        """
        assert src < self.world_size, f"Invalid src rank ({src})"

        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return obj_list
        # Broadcast.
        torch.distributed.broadcast_object_list(
            obj_list, src=self.ranks[src], group=self.device_group
        )
        return obj_list

    def all_gather_object(self, obj: Any) -> List[Any]:
        objs = [None] * self.world_size
        torch.distributed.all_gather_object(objs, obj, group=self.cpu_group)
        return objs

    def send_object(
        self,
        obj: Any,
        dst: int,
        async_send: bool = False,
        tag: int = 0,
    ) -> List[P2PWork]:
        """
        Send the input object list to the destination rank.
        This function uses the CPU group for all communications.

        TODO: If you want to use GPU communication, please add a new argument (e.g., data_group, group),
        use other functions (e.g., send), or implement a new function (e.g., send_object_device).

        NOTE: `dst` is the local rank of the destination rank.
        """

        assert dst < self.world_size, f"Invalid dst rank ({dst})"
        assert dst != self.rank_in_group, (
            "Invalid destination rank. Destination rank is the same "
            "as the current rank."
        )
        send_func = torch.distributed.isend if async_send else torch.distributed.send

        # Serialize object to tensor and get the size as well
        object_tensor = torch.frombuffer(pickle.dumps(obj), dtype=torch.uint8)
        size_tensor = torch.tensor(
            [object_tensor.numel()], dtype=torch.long, device="cpu"
        )

        # Send object size
        p2p_work = []
        size_work = send_func(
            size_tensor,
            self.ranks[dst],
            group=self.cpu_group,
            tag=tag,
        )
        if async_send:
            p2p_work.append(P2PWork(size_work, size_tensor))

        object_work = send_func(
            object_tensor,
            self.ranks[dst],
            group=self.cpu_group,
            tag=tag,
        )
        if async_send:
            p2p_work.append(P2PWork(object_work, object_tensor))

        return p2p_work

    def recv_object(
        self,
        src: int,
        tag: int = 0,
    ) -> Any:
        """Receive the input object list from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""

        assert src < self.world_size, f"Invalid src rank ({src})"
        assert src != self.rank_in_group, (
            "Invalid source rank. Source rank is the same as the current rank."
        )

        # #980: the two naked ``work.wait()`` calls that used to stand here --
        # one on the size header, one on the payload -- are the site where PP1
        # was caught LIVE and SILENT in boot 7 of the flip window. They are now
        # driven by a RESUMABLE frame that holds the protocol position
        # explicitly, because this stream cannot survive a terminal timeout:
        # once the size has been received the payload is already on the wire,
        # and a receiver that gives up mid-frame and re-posts later reads a
        # payload AS a size and misframes every later message.
        #
        # The wait itself is unchanged -- still the unbounded ``wait()`` that is
        # the only call with positive evidence that it DRIVES gloo -- but it is
        # parked on its own thread and the deadline sits on the join. So an
        # expired step never closes the pair (#829) and never abandons a
        # half-received message (#630 rules out is_completed() polling as the
        # alternative). The DEFAULT abort deadline is 0 == never abort, so this
        # path still blocks exactly as long as it did before; what changed is
        # that it now names the stall, per frame state, instead of going silent.
        #
        # The frame is kept per (src, tag) ON THE COORDINATOR, not per call:
        # that is what makes an expired or aborted step resumable, and a
        # per-call frame would post a second receive on a stream whose first is
        # still parked. Lazily attached in the local style of
        # ``_shape_cache_recv`` below rather than in __init__.
        frames = getattr(self, "_object_recv_frames", None)
        if frames is None:
            frames = self._object_recv_frames = {}
        frame = get_or_create_object_recv_frame(
            frames,
            (src, tag),
            group=self.cpu_group,
            src_global=self.ranks[src],
            tag=tag,
            site=f"{self.unique_name}/recv_object[src={src},tag={tag}]",
            rank_desc=f"rank_in_group={self.rank_in_group}/{self.world_size}",
        )
        return frame.receive(
            step_budget_s=recv_object_step_budget_s(),
            abort_after_s=recv_object_abort_after_s(),
        )

    def broadcast_tensor_dict(
        self,
        tensor_dict: Optional[Dict[str, Union[torch.Tensor, Any]]] = None,
        src: int = 0,
        group: Optional[ProcessGroup] = None,
        metadata_group: Optional[ProcessGroup] = None,
    ) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:
        """Broadcast the input tensor dictionary.
        NOTE: `src` is the local rank of the source rank.
        """
        # Bypass the function if we are using only 1 GPU.
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return tensor_dict

        group = self.device_group
        metadata_group = self.cpu_group
        assert src < self.world_size, f"Invalid src rank ({src})"

        rank_in_group = self.rank_in_group
        if rank_in_group == src:
            metadata_list: List[Tuple[Any, Any]] = []
            assert isinstance(tensor_dict, dict), (
                f"Expecting a dictionary, got {type(tensor_dict)}"
            )
            metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
            # `metadata_list` lives in CPU memory.
            # `broadcast_object_list` has serialization & deserialization,
            # all happening on CPU. Therefore, we can use the CPU group.
            self.broadcast_object(metadata_list, src=src)
            async_handles = []
            for tensor in tensor_list:
                if tensor.numel() == 0:
                    # Skip broadcasting empty tensors.
                    continue
                if tensor.is_cpu:
                    # use metadata_group for CPU tensors
                    handle = torch.distributed.broadcast(
                        tensor, src=self.ranks[src], group=metadata_group, async_op=True
                    )
                else:
                    # use group for GPU tensors
                    handle = torch.distributed.broadcast(
                        tensor, src=self.ranks[src], group=group, async_op=True
                    )
                async_handles.append(handle)
            for async_handle in async_handles:
                async_handle.wait()

        else:
            metadata_list = self.broadcast_object(None, src=src)
            tensor_dict = {}
            async_handles = []
            for key, value in metadata_list:
                if isinstance(value, TensorMetadata):
                    tensor = torch.empty(
                        value.size, dtype=value.dtype, device=value.device
                    )
                    if tensor.numel() == 0:
                        # Skip broadcasting empty tensors.
                        tensor_dict[key] = tensor
                        continue
                    if tensor.is_cpu:
                        # use metadata_group for CPU tensors
                        handle = torch.distributed.broadcast(
                            tensor,
                            src=self.ranks[src],
                            group=metadata_group,
                            async_op=True,
                        )
                    else:
                        # use group for GPU tensors
                        handle = torch.distributed.broadcast(
                            tensor, src=self.ranks[src], group=group, async_op=True
                        )
                    async_handles.append(handle)
                    tensor_dict[key] = tensor
                else:
                    tensor_dict[key] = value
            for async_handle in async_handles:
                async_handle.wait()
        return tensor_dict

    # ------------------------------------------------------------------
    # #201 slice 3: tensor-dict metadata shape cache (SGLANG_PP_SHAPE_CACHE).
    #
    # At the pipeline stage boundary the pickled metadata costs MORE than
    # the bs=1 hidden-state payload itself (measured slice 2 on the 40G
    # link: 249 us metadata vs 142 us payload one-way, 64% of the
    # crossing), and the metadata is a pure function of the batch geometry
    # -- static across decode rounds. With the cache on, a repeat crossing
    # sends a 16-byte reference header instead of size + pickle.
    #
    # Protocol (per (peer, direction) channel, over the same FIFO gloo
    # p2p ordering the stock size+pickle pair uses):
    #   header [code, size] (2 x int64) --
    #     code -k  : reuse mirrored cache entry k-1, no payload follows;
    #     code  1  : `size` pickle bytes follow, BOTH ends append to their
    #                mirror (ids assigned by arrival order, so they agree
    #                without negotiation);
    #     code  0  : `size` pickle bytes follow, uncached (sender mirror
    #                full) -- the receiver must not append either.
    # The env flag must be uniform across the world (it is inherited from
    # the launcher; the cross-rig rank script exports it on both nodes) --
    # a mixed setting would desynchronize the wire format.
    # ------------------------------------------------------------------

    #: Upper bound on mirrored shape-cache entries per peer. Distinct
    #: metadata blobs are one per batch geometry (a handful for decode,
    #: dozens for chunked prefill). Beyond the cap new blobs travel
    #: uncached (code 0), so the mirrors stay in lockstep without any
    #: eviction coordination.
    SHAPE_CACHE_MAX_ENTRIES = 1024

    @property
    def _pp_shape_cache_enabled(self) -> bool:
        cached = getattr(self, "_pp_shape_cache_flag", None)
        if cached is None:
            from sglang.srt.environ import envs

            cached = bool(envs.SGLANG_PP_SHAPE_CACHE.get())
            self._pp_shape_cache_flag = cached
        return cached

    def _send_tensor_dict_metadata(
        self, metadata_list: List[Any], dst: int, async_send: bool
    ) -> List[P2PWork]:
        """Send a tensor dict's metadata list, through the shape cache when
        SGLANG_PP_SHAPE_CACHE is on; stock send_object otherwise."""
        if not self._pp_shape_cache_enabled:
            return self.send_object(metadata_list, dst=dst, async_send=async_send)
        send_func = torch.distributed.isend if async_send else torch.distributed.send
        caches = getattr(self, "_shape_cache_send", None)
        if caches is None:
            caches = self._shape_cache_send = {}
        cache: Dict[bytes, int] = caches.setdefault(dst, {})
        blob = pickle.dumps(metadata_list)
        p2p_works: List[P2PWork] = []
        entry = cache.get(blob)
        if entry is not None:
            header = torch.tensor([-(entry + 1), 0], dtype=torch.long, device="cpu")
            work = send_func(header, self.ranks[dst], group=self.cpu_group, tag=0)
            if async_send:
                p2p_works.append(P2PWork(work, header))
            return p2p_works
        code = 0
        if len(cache) < self.SHAPE_CACHE_MAX_ENTRIES:
            cache[blob] = len(cache)
            code = 1
        object_tensor = torch.frombuffer(blob, dtype=torch.uint8)
        header = torch.tensor(
            [code, object_tensor.numel()], dtype=torch.long, device="cpu"
        )
        work = send_func(header, self.ranks[dst], group=self.cpu_group, tag=0)
        if async_send:
            p2p_works.append(P2PWork(work, header))
        work = send_func(object_tensor, self.ranks[dst], group=self.cpu_group, tag=0)
        if async_send:
            p2p_works.append(P2PWork(work, object_tensor))
        return p2p_works

    def _recv_tensor_dict_metadata(self, src: int) -> List[Any]:
        """Receive what _send_tensor_dict_metadata sent (mirror side)."""
        if not self._pp_shape_cache_enabled:
            return self.recv_object(src=src)
        caches = getattr(self, "_shape_cache_recv", None)
        if caches is None:
            caches = self._shape_cache_recv = {}
        cache: List[Any] = caches.setdefault(src, [])
        header = torch.empty(2, dtype=torch.long, device="cpu")
        torch.distributed.irecv(
            header, src=self.ranks[src], group=self.cpu_group, tag=0
        ).wait()
        code, size = int(header[0].item()), int(header[1].item())
        if code < 0:
            return cache[-code - 1]
        object_tensor = torch.empty(size, dtype=torch.uint8, device="cpu")
        torch.distributed.irecv(
            object_tensor, src=self.ranks[src], group=self.cpu_group, tag=0
        ).wait()
        obj = pickle.loads(object_tensor.numpy())
        if code == 1:
            cache.append(obj)
        return obj

    def send_tensor_dict(
        self,
        tensor_dict: Dict[str, Union[torch.Tensor, Any]],
        dst: Optional[int] = None,
        all_gather_group: Optional["GroupCoordinator"] = None,
        async_send: bool = False,
    ) -> Optional[List[P2PWork]]:
        """Send the input tensor dictionary.
        NOTE: `dst` is the local rank of the source rank.
        """
        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return tensor_dict

        all_gather_size = 1 if all_gather_group is None else all_gather_group.world_size
        all_gather_rank = (
            0 if all_gather_group is None else all_gather_group.rank_in_group
        )

        group = self.device_group
        metadata_group = self.cpu_group

        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        assert dst < self.world_size, f"Invalid dst rank ({dst})"

        assert isinstance(tensor_dict, dict), (
            f"Expecting a dictionary, got {type(tensor_dict)}"
        )
        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        # Note: While switching to Device-to-Device (D2D) would introduce an extra
        # Device-to-Host (D2H) memory copy overhead for serialization, our benchmarks
        # show better overall transmission performance with D2D due to:
        # 1. Superior D2D transfer bandwidth
        # 2. Ability to overlap send and recv operations
        # Thus the net performance gain justifies this approach.

        send_func = torch.distributed.isend if async_send else torch.distributed.send
        p2p_works = self._send_tensor_dict_metadata(metadata_list, dst, async_send)

        for tensor in tensor_list:
            if tensor.numel() == 0:
                # Skip sending empty tensors.
                continue

            # send-allgather: send only a slice, then do allgather.
            if all_gather_group is not None and tensor.numel() % all_gather_size == 0:
                tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]

            comm_group = metadata_group if tensor.is_cpu else group
            work = send_func(tensor, self.ranks[dst], group=comm_group)
            if async_send:
                p2p_works.append(P2PWork(work, tensor))
        return p2p_works

    def warmup_p2p_pairs(self) -> None:
        """Establish this ring's p2p pairs BEFORE anyone needs them.

        MEASURED, boot 59 of window-flip-0828, all three ranks in one window:
        PP2 sat in ``torch.distributed.isend`` under ``send_tensor_dict`` --
        the last rank's output wrap, which addresses ``self.ranks[dst]`` with
        ``dst = (rank_in_group + 1) % world_size`` and so wraps 2 -> 0. That
        pair had never been used, and a torch p2p pair is built lazily and
        needs BOTH ends. PP0 was by then already parked in the admission
        ring-commit (``_pp_commit_admission_send_work``) and PP1 with it, so
        nobody joined the build. Closed three-arc cycle, and the ranks were at
        two different lines of ONE loop body -- PP2 at :4454, PP0/PP1 at :4514.

        So the first use of a pair must not be the one that has to succeed
        while the peers are elsewhere. This walks the same neighbours over the
        same groups the real send uses -- ``device_group`` for device tensors,
        ``cpu_group`` for the metadata half -- so the pair the loop later needs
        is the pair this warmed.

        Both directions in one shot: every rank sends to its successor and
        receives from its predecessor, which is exactly the ring the loop
        drives, and no rank waits on a peer that is not simultaneously here.
        Cheap (one byte per direction per group) and idempotent.

        NOT gated on the flip. The cycle reproduces with enable_phase_flip
        off (#990), so gating it would leave the plain PP path holding the
        defect. Same channel-asymmetry family as #801.
        """
        if self.world_size < 2:
            return
        import time as _time

        t0 = _time.perf_counter()
        nxt = self.ranks[(self.rank_in_group + 1) % self.world_size]
        prv = self.ranks[(self.rank_in_group - 1) % self.world_size]
        pairs = []
        for name, grp, dev in (
            ("device", self.device_group, self.device),
            ("cpu", self.cpu_group, torch.device("cpu")),
        ):
            if grp is None:
                continue
            try:
                out = torch.zeros(1, dtype=torch.uint8, device=dev)
                inp = torch.zeros(1, dtype=torch.uint8, device=dev)
                works = [
                    torch.distributed.isend(out, nxt, group=grp),
                    torch.distributed.irecv(inp, prv, group=grp),
                ]
                for w in works:
                    w.wait()
                pairs.append(f"{name}:{self.rank}->{nxt},{prv}->{self.rank}")
            except Exception as exc:  # noqa: BLE001
                # A warmup may not be the thing that stops a boot; a pair that
                # refuses here would have refused later, and the named line is
                # what a reader needs either way.
                pairs.append(f"{name}:FAILED({type(exc).__name__})")
        logger.warning(
            "PP-P2P-WARMUP pairs=%s done in %.1fms (group=%s world=%d). "
            "Lazy pair construction needs both ends; boot 59 deadlocked "
            "because the first use of 2->0 fell after the peers had entered "
            "the admission ring-commit.",
            ";".join(pairs) or "-",
            (_time.perf_counter() - t0) * 1e3,
            getattr(self, "group_name", "?"),
            self.world_size,
        )

    def recv_tensor_dict(
        self,
        src: Optional[int] = None,
        all_gather_group: Optional["GroupCoordinator"] = None,
    ) -> Optional[Dict[str, Union[torch.Tensor, Any]]]:
        """Recv the input tensor dictionary.
        NOTE: `src` is the local rank of the source rank.
        """
        # Bypass the function if we are using only 1 GPU.
        if not torch.distributed.is_initialized() or self.world_size == 1:
            return None

        all_gather_size = 1 if all_gather_group is None else all_gather_group.world_size
        all_gather_rank = (
            0 if all_gather_group is None else all_gather_group.rank_in_group
        )

        group = self.device_group
        metadata_group = self.cpu_group

        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        assert src < self.world_size, f"Invalid src rank ({src})"

        recv_metadata_list = self._recv_tensor_dict_metadata(src)
        tensor_dict: Dict[str, Any] = {}
        for key, value in recv_metadata_list:
            if isinstance(value, TensorMetadata):
                tensor = torch.empty(value.size, dtype=value.dtype, device=value.device)
                if tensor.numel() == 0:
                    # Skip broadcasting empty tensors.
                    tensor_dict[key] = tensor
                    continue

                # send-allgather: send only a slice, then do allgather.
                use_all_gather = (
                    all_gather_group is not None
                    and tensor.numel() % all_gather_size == 0
                )

                if use_all_gather:
                    orig_shape = tensor.shape
                    tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]

                # We have to use irecv here to make it work for both isend and send.
                comm_group = metadata_group if tensor.is_cpu else group
                work = torch.distributed.irecv(
                    tensor, src=self.ranks[src], group=comm_group
                )
                work.wait()

                if use_all_gather:
                    tensor = all_gather_group.all_gather(tensor, dim=0)
                    tensor = tensor.reshape(orig_shape)

                tensor_dict[key] = tensor
            else:
                tensor_dict[key] = value
        return tensor_dict

    def barrier(self):
        """Barrier synchronization among the group.
        NOTE: don't use `device_group` here! `barrier` in NCCL is
        terrible because it is internally a broadcast operation with
        secretly created GPU tensors. It is easy to mess up the current
        device. Use the CPU group instead.

        This is the site of the 17-minute stall measured in #194: it is
        what `enter_capture_group_barrier` and `run_capture_warmups` call
        (see model_executor/runner/base_runner.py), so a rank SIGKILLed
        during CUDA-graph capture parks every survivor here for the whole
        gloo process-group timeout. When an barlink transport has published a
        peer table, the wait goes through `bounded_barrier` and ends with
        the dead rank named instead. A boot without barlink takes the plain
        call below, unchanged and without importing anything.
        """
        liveness = _peer_liveness_for_barrier()
        if liveness is not None:
            liveness.bounded_barrier(
                self.cpu_group, f"group barrier ({self.unique_name})"
            )
            return
        torch.distributed.barrier(group=self.cpu_group)

    def send(self, tensor: torch.Tensor, dst: Optional[int] = None) -> None:
        """Sends a tensor to the destination rank in a non-blocking way"""
        """NOTE: `dst` is the local rank of the destination rank."""
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size

        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.send(tensor, dst)
        else:
            torch.distributed.send(tensor, self.ranks[dst], self.device_group)

    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: Optional[int] = None
    ) -> torch.Tensor:
        """Receives a tensor from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size

        tensor = torch.empty(size, dtype=dtype, device=self.device)
        pynccl_comm = self.pynccl_comm
        if pynccl_comm is not None and not pynccl_comm.disabled:
            pynccl_comm.recv(tensor, src)
        else:
            torch.distributed.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    def destroy(self):
        # Before the process groups go away: barlink owns a POSIX shm segment
        # that must be unlinked, and its close() uses no collectives.
        if getattr(self, "barlink_comm", None) is not None:
            self.barlink_comm.close()
            self.barlink_comm = None
        if self.device_group is not None:
            torch.distributed.destroy_process_group(self.device_group)
            self.device_group = None
        if self.cpu_group is not None:
            torch.distributed.destroy_process_group(self.cpu_group)
            self.cpu_group = None
        if self.pynccl_comm is not None:
            self.pynccl_comm = None
        if self.pymscclpp_comm is not None:
            self.pymscclpp_comm.destroy()
        if self.ca_comm is not None:
            self.ca_comm = None
        if self.mq_broadcaster is not None:
            self.mq_broadcaster = None


_WORLD: Optional[GroupCoordinator] = None


def get_world_group() -> GroupCoordinator:
    assert _WORLD is not None, "world group is not initialized"
    return _WORLD


def init_world_group(
    ranks: List[int], local_rank: int, backend: str, recovered_rank: bool = False
) -> GroupCoordinator:
    return GroupCoordinator(
        group_ranks=[ranks],
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_pynccl=False,
        use_pymscclpp=False,
        use_custom_allreduce=False,
        use_torch_symm_mem_all_reduce=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        use_npu_communicator=False,
        group_name="world",
        recovered_rank=recovered_rank,
    )


def init_model_parallel_group(
    group_ranks: List[List[int]],
    local_rank: int,
    backend: str,
    use_pynccl: Optional[bool] = None,
    use_custom_allreduce: Optional[bool] = None,
    use_message_queue_broadcaster: bool = False,
    group_name: Optional[str] = None,
    use_mscclpp_allreduce: Optional[bool] = None,
    use_torch_symm_mem_allreduce: Optional[bool] = None,
    recovered_rank: bool = False,
) -> GroupCoordinator:
    if use_custom_allreduce is None:
        use_custom_allreduce = _ENABLE_CUSTOM_ALL_REDUCE
    if use_mscclpp_allreduce is None:
        use_mscclpp_allreduce = _ENABLE_MSCCLPP_ALL_REDUCE
    if use_torch_symm_mem_allreduce is None:
        use_torch_symm_mem_allreduce = _ENABLE_TORCH_SYMM_MEM_ALL_REDUCE
    return GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_pynccl=(
            not (_is_npu or _is_xpu or backend == "mooncake")
            if use_pynccl is None
            else use_pynccl
        ),
        use_pymscclpp=use_mscclpp_allreduce,
        use_custom_allreduce=use_custom_allreduce,
        use_torch_symm_mem_all_reduce=use_torch_symm_mem_allreduce,
        use_hpu_communicator=True,
        use_xpu_communicator=True,
        use_npu_communicator=True,
        use_message_queue_broadcaster=use_message_queue_broadcaster,
        group_name=group_name,
        recovered_rank=recovered_rank,
    )


_TP: Optional[GroupCoordinator] = None
_ATTN_TP: Optional[GroupCoordinator] = None
_ATTN_CP: Optional[GroupCoordinator] = None
_DCP: Optional[GroupCoordinator] = None

# kv-session-offload decoupling: a SECOND DCP communicator over the SAME ranks
# (PDMUX-style duplicate group), so the spill forward's DCP collectives run on
# their own NCCL comm -- disjoint from the device forward's comm -- a
# prerequisite for running the two lanes concurrently (NCCL pairs collectives
# by per-communicator enqueue order; one comm across two lanes = hang). Built
# at init only when SGLANG_KVSO_DECOUPLE=1 and DCP>1; None otherwise (so
# get_dcp_group is byte-identical). Routing is a serial per-forward flag: the
# spill forward sets _DCP_SPILL_ACTIVE, so get_dcp_group returns _DCP_SPILL for
# the whole spill forward and _DCP for everything else. Rank-uniform: every
# rank builds the spill batch from replicated state and toggles the flag
# identically, so comm A sees only device ops and comm B only spill ops, both
# in a rank-uniform order.
_DCP_SPILL: Optional[GroupCoordinator] = None
_DCP_SPILL_ACTIVE: bool = False

# #704b B1: the DECOUPLED-KV group -- attention KV token-sharded across the
# ranks of one PP pipeline, so a layer's KV no longer lives only on the rank
# that owns the layer.
#
# It gets its OWN global, getter and routing flag rather than reusing
# ``dcp_size``. Overloading dcp_size would make ``ParallelContext.dcp_enabled``
# (runtime_context.py:331-337) report True on PP prefill ranks, which
# contradicts the invariant ``dcp_group_guard.py:38-42`` states in prose -- and
# the guard would still PASS, so nothing would announce the contradiction.
# Keeping a separate flag leaves that invariant TRUE and needs no consumer
# audit. There are no "typed group" classes in this tree: a type IS a global,
# a named getter and a routing flag (#616 survey).
_DECOUPLED_KV: Optional[GroupCoordinator] = None
_DECOUPLED_KV_ACTIVE: bool = False

# duplicate GroupCoordinator for prefill in PD-Multiplexing
_PDMUX_PREFILL_TP_GROUP: Optional[GroupCoordinator] = None

# #631 Route A phase flip: the SECONDARY (flip-target) group set, built
# eagerly at init over the same world (the dcp_spill/pdmux duplicate-group
# precedent -- a group CREATE is itself a collective, never lazy). All None
# until initialize_phase_flip_secondary_groups runs; the default path never
# builds them, keeping today's behavior byte-identical.
_FLIP_TP: Optional[GroupCoordinator] = None
_FLIP_DCP: Optional[GroupCoordinator] = None
_FLIP_PP: Optional[GroupCoordinator] = None

# #631: when True, the module-level group getters below route to the flip
# set (_FLIP_*). This is the _DCP_SPILL_ACTIVE/_ENABLE_PDMUX_P_TP routing
# precedent, needed because forward-time collectives reach groups through
# these getters (tensor_model_parallel_all_reduce -> get_tp_group()), NOT
# through the runtime_context contextvar -- an override alone would shard
# weights correctly and then all-reduce on the WRONG (primary tp=1) group,
# which is a silent no-op, i.e. silent corruption. Toggled rank-uniformly:
# at boot around the TP-stack build, and at cutover by the flip protocol.
_PHASE_FLIP_TP_ACTIVE: bool = False

_ENABLE_PDMUX_P_TP: bool = False


def set_pdmux_status(enable_prefill_multiplexing: bool):
    global _ENABLE_PDMUX_P_TP
    _ENABLE_PDMUX_P_TP = enable_prefill_multiplexing


def get_tp_group() -> GroupCoordinator:
    if _PHASE_FLIP_TP_ACTIVE:
        assert _FLIP_TP is not None, (
            "phase-flip TP routing is active but the flip group set is not initialized"
        )
        return _FLIP_TP
    if _ENABLE_PDMUX_P_TP:
        assert _PDMUX_PREFILL_TP_GROUP is not None, (
            "tensor model parallel group for PD-Multiplexing Prefill is not initialized"
        )
        return _PDMUX_PREFILL_TP_GROUP
    assert _TP is not None, "tensor model parallel group is not initialized"
    return _TP


def set_phase_flip_tp_active(active: bool) -> None:
    """#631: route the module-level group getters to the flip set.

    Unlike set_dcp_spill_active this REFUSES (instead of silently no-oping)
    when the flip groups were never built: activating the route without them
    would leave every later collective on the primary (tp=1) groups -- a
    silent no-op all-reduce, the exact corruption class this routing exists
    to prevent. Rank-uniform by contract: callers are the boot-time TP-stack
    build scope and the flip cutover, both of which run on every rank in the
    same round."""
    global _PHASE_FLIP_TP_ACTIVE
    if active and _FLIP_TP is None:
        raise RuntimeError(
            "set_phase_flip_tp_active(True) without initialized phase-flip "
            "secondary groups (initialize_phase_flip_secondary_groups was "
            "never called)"
        )
    _PHASE_FLIP_TP_ACTIVE = bool(active)


def phase_flip_tp_routing_active() -> bool:
    return _PHASE_FLIP_TP_ACTIVE


def get_attn_tp_group() -> GroupCoordinator:
    if _PHASE_FLIP_TP_ACTIVE:
        # The flip TP phase is pure TP (no dp-attention, guarded in
        # server_args): attn_tp == tp, served by the same flip group.
        assert _FLIP_TP is not None, (
            "phase-flip TP routing is active but the flip group set is not initialized"
        )
        return _FLIP_TP
    assert _ATTN_TP is not None, (
        "attention tensor model parallel group is not initialized"
    )
    return _ATTN_TP


def get_attn_cp_group() -> GroupCoordinator:
    assert _ATTN_CP is not None, (
        "attention context model parallel group is not initialized"
    )
    return _ATTN_CP


def get_dcp_group_no_assert() -> Optional[GroupCoordinator]:
    return _DCP


def get_dcp_group() -> GroupCoordinator:
    # #631: the flip TP phase owns tokens under the flip DCP group. Checked
    # before the spill route -- kv-session-offload is a flip arming guard,
    # so the two can never be legitimately active together.
    if _PHASE_FLIP_TP_ACTIVE:
        assert _FLIP_DCP is not None, (
            "phase-flip TP routing is active but the flip dcp group is not initialized"
        )
        return _FLIP_DCP
    # kv-session-offload decoupling: route the (serial, per-forward) spill
    # forward to its own communicator. Falls back to _DCP when decoupling is
    # off (_DCP_SPILL is None) or outside a spill forward -> byte-identical.
    if _DCP_SPILL_ACTIVE and _DCP_SPILL is not None:
        return _DCP_SPILL
    # #704b B1, deliberately LAST before the primary fall-through.
    #
    # It claims only what would otherwise reach ``_DCP``, so every existing
    # route keeps precedence: the flip owns the TP decode phase (B1 is a PP
    # PREFILL mechanism and cannot be legitimately active there), and a spill
    # forward keeps its dedicated serial communicator. Placing B1 earlier would
    # silently capture those forwards -- first-match routing is the failure
    # mode, so the position is pinned by a test rather than left to reading
    # order.
    if _DECOUPLED_KV_ACTIVE and _DECOUPLED_KV is not None:
        return _DECOUPLED_KV
    assert _DCP is not None, "decode context parallel group is not initialized"
    return _DCP


def get_decoupled_kv_group_no_assert() -> Optional[GroupCoordinator]:
    return _DECOUPLED_KV


def get_decoupled_kv_group() -> GroupCoordinator:
    assert _DECOUPLED_KV is not None, (
        "the #704b decoupled-KV group is not initialized; call "
        "initialize_decoupled_kv_group before routing through it"
    )
    return _DECOUPLED_KV


def set_decoupled_kv_active(active: bool) -> None:
    """Arm/disarm B1 routing. Off is byte-identical to the pre-#704b path.

    REFUSES to arm without an initialized group, following
    ``set_phase_flip_tp_active`` above rather than the silently-no-oping
    ``set_dcp_spill_active``. The reason is the same one stated there: arming
    a route whose group was never built leaves every later collective on the
    primary group, "a silent no-op all-reduce, the exact corruption class this
    routing exists to prevent". The fall-through in ``get_dcp_group`` stays as
    a second line of defence -- it protects a state this setter now makes
    unreachable, and defence in depth is cheap here.

    Disarming is unconditional: it returns to the pre-#704b route, which is
    valid regardless of what was or was not built.
    """
    global _DECOUPLED_KV_ACTIVE
    if active and _DECOUPLED_KV is None:
        raise RuntimeError(
            "set_decoupled_kv_active(True) without an initialized #704b "
            "decoupled-KV group (initialize_decoupled_kv_group was never "
            "called). Arming now would route the attention merge to the "
            "primary DCP group while the pool is sized for decoupled "
            "ownership -- silently wrong output, not a crash."
        )
    _DECOUPLED_KV_ACTIVE = bool(active)


def get_dcp_spill_group_no_assert() -> Optional[GroupCoordinator]:
    return _DCP_SPILL


def set_dcp_spill_active(active: bool) -> None:
    """Route get_dcp_group() to the spill communicator (_DCP_SPILL) for the
    duration of one spill forward. No-op unless the second comm was built
    (decoupling on). Serial: SGLang runs one forward at a time per rank, so
    this whole-forward flag cleanly scopes the spill forward's collectives to
    comm B and leaves device forwards on comm A. Rank-uniform (set from
    replicated kv_session_spill_tick)."""
    global _DCP_SPILL_ACTIVE
    _DCP_SPILL_ACTIVE = bool(active) and _DCP_SPILL is not None


_MOE_DP: Optional[GroupCoordinator] = None
_MOE_EP: Optional[GroupCoordinator] = None
_MOE_TP: Optional[GroupCoordinator] = None


def get_moe_dp_group() -> GroupCoordinator:
    assert _MOE_DP is not None, "moe data parallel group is not initialized"
    return _MOE_DP


def get_moe_ep_group() -> GroupCoordinator:
    assert _MOE_EP is not None, "expert model parallel group is not initialized"
    return _MOE_EP


def get_moe_tp_group() -> GroupCoordinator:
    assert _MOE_TP is not None, "expert model parallel group is not initialized"
    return _MOE_TP


# kept for backward compatibility
get_tensor_model_parallel_group = get_tp_group

_PP: Optional[GroupCoordinator] = None


def get_pp_group() -> GroupCoordinator:
    # #631: under flip TP routing the pp axis is trivial (pp_size=1); the
    # flip pp group keeps is_first_rank/is_last_rank and any send/recv
    # bookkeeping consistent with that geometry instead of the primary
    # 3-stage pipeline's.
    if _PHASE_FLIP_TP_ACTIVE:
        assert _FLIP_PP is not None, (
            "phase-flip TP routing is active but the flip pp group is not initialized"
        )
        return _FLIP_PP
    assert _PP is not None, "pipeline model parallel group is not initialized"
    return _PP


# kept for backward compatibility
get_pipeline_model_parallel_group = get_pp_group


def get_mooncake_transfer_engine():
    """
    Return the shared MooncakeTransferEngine if initialized in device_communicators,
    else None. Used by disaggregation mooncake backend and mem_cache mooncake_store.
    """
    from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
        get_mooncake_transfer_engine as _get_engine,
    )

    return _get_engine()


@contextmanager
def graph_capture(stream=None):
    """
    `graph_capture` is a context manager which should surround the code that
    is capturing the CUDA graph. Its main purpose is to ensure that the
    some operations will be run after the graph is captured, before the graph
    is replayed. It returns a `GraphCaptureContext` object which contains the
    necessary data for the graph capture. Currently, it only contains the
    stream that the graph capture is running on. This stream is set to the
    current CUDA stream when the context manager is entered and reset to the
    default stream when the context manager is exited. This is to ensure that
    the graph capture is running on a separate stream from the default stream,
    in order to explicitly distinguish the kernels to capture
    from other kernels possibly launched on background in the default stream.
    """
    # #517 phase 2: exclude the barlink watchdog's abort-word poll for the
    # duration of the capture. Torch captures in cudaStreamCaptureModeGlobal,
    # under which a synchronizing CUDA call in ANY thread of the process
    # invalidates the capture -- so the poll may not merely check a flag and
    # hope, it has to be locked out. This is the one context manager that
    # surrounds every capture in the process, which is why the exclusion is
    # stated here and not guessed per backend.
    # (barlink.graph_capture_running() cannot serve: it answers for the
    # CALLING thread's stream, and the watchdog is a different thread.)
    from sglang.srt.distributed.device_communicators import barlink_abort_gate

    with (
        barlink_abort_gate.pause_polling(),
        get_tp_group().graph_capture(stream=stream) as context,
        get_pp_group().graph_capture(context),
    ):
        with contextlib.ExitStack() as stack:
            seen = {id(_TP), id(_PP)}
            for group in (_DCP, _MOE_EP, _MOE_TP):
                if group is not None and id(group) not in seen:
                    seen.add(id(group))
                    stack.enter_context(group.graph_capture(context))
            yield context


logger = logging.getLogger(__name__)

_ENABLE_CUSTOM_ALL_REDUCE = True
_ENABLE_MSCCLPP_ALL_REDUCE = False
_ENABLE_TORCH_SYMM_MEM_ALL_REDUCE = False


def set_custom_all_reduce(enable: bool):
    global _ENABLE_CUSTOM_ALL_REDUCE
    _ENABLE_CUSTOM_ALL_REDUCE = enable


def set_mscclpp_all_reduce(enable: bool):
    global _ENABLE_MSCCLPP_ALL_REDUCE
    _ENABLE_MSCCLPP_ALL_REDUCE = enable


def set_torch_symm_mem_all_reduce(enable: bool):
    global _ENABLE_TORCH_SYMM_MEM_ALL_REDUCE
    _ENABLE_TORCH_SYMM_MEM_ALL_REDUCE = enable


# TODO: refactor in-tree platforms to get rid of this wrapper
def get_default_distributed_backend(device: str) -> str:
    # We deliberately go through ``platforms.current_platform`` (rather than
    # ``from ... import current_platform``) so each call resolves through the
    # platforms package's lazy ``__getattr__`` and picks up runtime overrides
    # of ``_current_platform`` (e.g. in tests).
    if device == platforms.current_platform.device_type:
        return platforms.current_platform.get_torch_distributed_backend_str()
    return _DEVICE_TO_DISTRIBUTED_BACKEND.get(device, "gloo")


def _create_global_tcp_store(rank: int, world_size: int) -> None:
    """Create a global TCPStore for coordination across ranks.

    This function creates a TCPStore that all ranks can use for coordination
    (e.g., for NIXL buffer setup).
    """
    from torch.distributed import TCPStore

    master_ip = os.environ.get("MASTER_ADDR")

    if not master_ip:
        logger.warning(
            "Could not determine master IP for global TCPStore. "
            "Broadcasting from rank 0 to all ranks."
        )

    base_store_port = envs.SGLANG_TCP_STORE_PORT.get()

    # Rank 0 gets its local IP and broadcasts it to all ranks
    # Use broadcast_object_list which works with any backend (handles CPU/GPU automatically)
    if not master_ip:
        if rank == 0:
            master_ip = get_local_ip_auto()
            ip_list = [master_ip]
        else:
            ip_list = [None]

        torch.distributed.broadcast_object_list(ip_list, src=0)
        master_ip = ip_list[0]

    try:
        tcp_store = TCPStore(
            host_name=master_ip,
            port=base_store_port,
            world_size=world_size,
            is_master=(rank == 0),
        )
        set_global_tcp_store(tcp_store)
        logger.info(
            "Created global TCPStore at %s:%d (rank=%d, world_size=%d)",
            master_ip,
            base_store_port,
            rank,
            world_size,
        )
    except Exception as e:
        logger.warning(
            "Failed to create global TCPStore at %s:%d: %s. "
            "Components requiring TCPStore (like NIXL) may not work.",
            master_ip,
            base_store_port,
            e,
        )


def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "nccl",
    timeout: Optional[int] = None,
    moe_a2a_backend: Optional[str] = None,
    recovered_rank: bool = False,
):
    logger.debug(
        "world_size=%d rank=%d local_rank=%d distributed_init_method=%s backend=%s",
        world_size,
        rank,
        local_rank,
        distributed_init_method,
        backend,
    )
    if "mooncake" in backend:
        try:
            from mooncake import ep as mooncake_ep
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run SGLang with Mooncake Backend."
            ) from e
        mooncake_ep.set_host_ip(get_local_ip_auto())

    if not torch.distributed.is_initialized():
        global _MODEL_PARALLEL_GROUP_TIMEOUT
        assert distributed_init_method is not None, (
            "distributed_init_method must be provided when initializing "
            "distributed environment"
        )
        if timeout is not None:
            assert isinstance(timeout, (int)), "timeout must be a number"
            assert timeout > 0, "timeout must be positive"
            timeout = timedelta(seconds=timeout)

        _MODEL_PARALLEL_GROUP_TIMEOUT = timeout

        if backend == "mooncake":
            from mooncake.ep import MooncakeBackendOptions

            # Setting "cuda" as device here is safe, as it is guarded under the mooncake case
            active_ranks = torch.ones(world_size, dtype=torch.int32, device="cuda")
            pg_options = MooncakeBackendOptions(active_ranks, recovered_rank)
        else:
            pg_options = get_torch_distributed_pg_options()

        # this backend is used for WORLD
        torch.distributed.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank,
            timeout=timeout,
            pg_options=pg_options,
        )

        # Create a global TCPStore for coordination (used by NIXL)
        if moe_a2a_backend == "nixl":
            _create_global_tcp_store(rank, world_size)

    # set the local rank
    # local_rank is not available in torch ProcessGroup,
    # see https://github.com/pytorch/pytorch/issues/122816
    if local_rank == -1:
        # local rank not set, this usually happens in single-node
        # setting, where we can use rank as local rank
        if distributed_init_method == "env://":
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        else:
            local_rank = rank
    global _WORLD
    if _WORLD is None:
        ranks = list(range(torch.distributed.get_world_size()))
        _WORLD = init_world_group(
            ranks, local_rank, backend, recovered_rank=recovered_rank
        )
    else:
        assert _WORLD.world_size == torch.distributed.get_world_size(), (
            "world group already initialized with a different world size"
        )


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    attention_data_parallel_size: int = 1,
    attention_context_model_parallel_size: int = 1,
    moe_data_model_parallel_size: int = 1,
    decode_context_parallel_size: int = 1,
    backend: Optional[str] = None,
    duplicate_tp_group: bool = False,
    enable_symm_mem: bool = False,
    recovered_rank: bool = False,
) -> None:
    """
    Initialize model parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model
            parallelism.
        expert_model_parallel_size: number of GPUs used for expert model
            parallelism.
        pipeline_model_parallel_size: number of GPUs used for pipeline model
            parallelism.
        attention_data_parallel_size: number of GPUs used for attention data
            parallelism.
        attention_context_model_parallel_size: number of GPUs used for attention context
            parallelism.
        moe_data_model_parallel_size: number of GPUs used for moe data
            parallelism.
        decode_context_parallel_size: number of GPUs used for decode context
            parallelism, which splits the KV cache across GPUs within each
            tensor-parallel group during decoding. Must be a divisor of
            tensor_model_parallel_size and is currently only supported on the
            AMD HIP platform.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 4 tensor model-parallel groups and 2 pipeline model-parallel groups:
        4 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 pipeline model-parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]

    Let's say we use 2 GPUs for attention context parallelism (attn_cp_size=2) and 4 GPUs for
    attention tensor parallelism (attn_tp_size=4). As for MoE part, we use 2 GPUs for moe data
    parallelism (moe_dp_size=2) and 4 GPUs for moe expert parallelism (moe_ep_size=4). The present
    function will create the following groups:
        2 tensor model-parallel groups:
            [g0, g1, g2, g3], [g4, g5, g6, g7]
        4 attention context-parallel groups:
            [g0, g4], [g1, g5], [g2, g6], [g3, g7]
        2 moe expert-parallel groups:
            [g0, g1, g2, g3], [g4, g5, g6, g7]
        4 moe data-parallel groups:
            [g0, g4], [g1, g5], [g2, g6], [g3, g7]

    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    if world_size != tensor_model_parallel_size * pipeline_model_parallel_size:
        raise RuntimeError(
            f"world_size ({world_size}) is not equal to "
            f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
            f"pipeline_model_parallel_size ({pipeline_model_parallel_size})"
        )
    if decode_context_parallel_size < 1:
        raise RuntimeError(
            f"decode_context_parallel_size ({decode_context_parallel_size}) must be >= 1"
        )
    if decode_context_parallel_size > 1 and not (is_hip() or is_cuda()):
        raise RuntimeError(
            "Decode context parallel (decode_context_parallel_size > 1) is "
            "currently only supported on the AMD HIP platform or CUDA platform, but got "
            f"decode_context_parallel_size ({decode_context_parallel_size}) "
            "on a non-HIP or non-CUDA platform."
        )
    if tensor_model_parallel_size % decode_context_parallel_size != 0:
        raise RuntimeError(
            f"tensor_model_parallel_size ({tensor_model_parallel_size}) must be divisible by "
            f"decode_context_parallel_size ({decode_context_parallel_size})"
        )

    # Build the tensor model-parallel groups.
    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    global _TP
    assert _TP is None, "tensor model parallel group is already initialized"
    group_ranks = []
    for tp_group_idx in range(num_tensor_model_parallel_groups):
        ranks = list(
            range(
                tp_group_idx * tensor_model_parallel_size,
                (tp_group_idx + 1) * tensor_model_parallel_size,
            )
        )
        group_ranks.append(ranks)

    # message queue broadcaster is only used in tensor model parallel group
    _TP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=envs.SGLANG_USE_MESSAGE_QUEUE_BROADCASTER.get(),
        group_name="tp",
        recovered_rank=recovered_rank,
    )

    if duplicate_tp_group:
        global _PDMUX_PREFILL_TP_GROUP
        assert _PDMUX_PREFILL_TP_GROUP is None, (
            "tensor model parallel group for PD-Multiplexing Prefill is already initialized"
        )
        _PDMUX_PREFILL_TP_GROUP = init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            use_message_queue_broadcaster=envs.SGLANG_USE_MESSAGE_QUEUE_BROADCASTER.get(),
            group_name="pdmux_prefill_tp",
            recovered_rank=recovered_rank,
        )
        if _TP.pynccl_comm:
            _TP.pynccl_comm.disabled = False
            _PDMUX_PREFILL_TP_GROUP.pynccl_comm.disabled = False

    # #616 labeled gap, closed here so it fires BEFORE any group is created:
    # pp>1 with dcp>1 would make dcp_enabled True on PP prefill ranks against
    # the invariant dcp_group_guard.py:38-42 documents, and the guard would
    # still pass. Token-sharded KV under PP goes through the #704b
    # decoupled-KV group, which carries its own flag.
    refuse_pp_dcp_combination(
        pipeline_model_parallel_size, decode_context_parallel_size
    )

    # Build decode context-parallel groups inside each TP group only when DCP is enabled.
    global _DCP
    assert _DCP is None, "decode context parallel group is already initialized"
    if decode_context_parallel_size > 1:
        dcp_group_ranks = []
        for tp_group in group_ranks:
            for start in range(0, len(tp_group), decode_context_parallel_size):
                dcp_group_ranks.append(
                    tp_group[start : start + decode_context_parallel_size]
                )
        _DCP = init_model_parallel_group(
            dcp_group_ranks,
            get_world_group().local_rank,
            backend,
            use_message_queue_broadcaster=envs.SGLANG_USE_MESSAGE_QUEUE_BROADCASTER.get(),
            group_name="dcp",
            recovered_rank=recovered_rank,
        )
        if get_tensor_model_parallel_rank() == 0:
            logger.info(
                f"DCP enabled, dcp_size={decode_context_parallel_size}, tp_size={tensor_model_parallel_size}"
            )

        # kv-session-offload decoupling: build the SECOND DCP communicator over
        # the SAME dcp_group_ranks (PDMUX duplicate-group pattern). A group
        # CREATE is itself a collective, so build it now on ALL ranks at init --
        # never lazily on first spill (a lazy create would race the device
        # lane). Gated by SGLANG_KVSO_DECOUPLE so the default path is
        # byte-identical (comm B not created, get_dcp_group stays on _DCP).
        # Enable both pynccl comms explicitly (PDMUX precedent).
        import os as _os

        if _os.environ.get("SGLANG_KVSO_DECOUPLE", "0") == "1":
            global _DCP_SPILL
            assert _DCP_SPILL is None, "spill DCP group already initialized"
            _DCP_SPILL = init_model_parallel_group(
                dcp_group_ranks,
                get_world_group().local_rank,
                backend,
                use_message_queue_broadcaster=envs.SGLANG_USE_MESSAGE_QUEUE_BROADCASTER.get(),
                group_name="dcp_spill",
                recovered_rank=recovered_rank,
            )
            if _DCP.pynccl_comm:
                _DCP.pynccl_comm.disabled = False
            if _DCP_SPILL.pynccl_comm:
                _DCP_SPILL.pynccl_comm.disabled = False
            if get_tensor_model_parallel_rank() == 0:
                logger.info(
                    "kv-session-offload DECOUPLE: second DCP communicator "
                    "'dcp_spill' built over the same ranks (comm B)."
                )

    attn_dp_size = attention_data_parallel_size
    attn_cp_size = attention_context_model_parallel_size
    attn_tp_size = tensor_model_parallel_size // attn_cp_size // attn_dp_size

    global _ATTN_CP
    assert _ATTN_CP is None, (
        "attention context model parallel group is already initialized"
    )
    if attn_cp_size == tensor_model_parallel_size:
        _ATTN_CP = _TP
    else:
        group_ranks = []
        for tp_group_idx in range(num_tensor_model_parallel_groups):
            for dp_idx in range(attn_dp_size):
                for attn_tp_idx in range(attn_tp_size):
                    st = (
                        tp_group_idx * tensor_model_parallel_size
                        + dp_idx * attn_tp_size * attn_cp_size
                        + attn_tp_idx
                    )
                    en = (
                        tp_group_idx * tensor_model_parallel_size
                        + (dp_idx + 1) * attn_tp_size * attn_cp_size
                        + attn_tp_idx
                    )
                    ranks = list(range(st, en, attn_tp_size))
                    group_ranks.append(ranks)
        _ATTN_CP = init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            use_message_queue_broadcaster=envs.SGLANG_USE_MESSAGE_QUEUE_BROADCASTER.get(),
            group_name="attn_cp",
            recovered_rank=recovered_rank,
        )

    from sglang.srt.layers.sampler import SYNC_TOKEN_IDS_ACROSS_TP

    global _ATTN_TP
    assert _ATTN_TP is None, (
        "attention tensor model parallel group is already initialized"
    )
    if attn_tp_size == tensor_model_parallel_size:
        _ATTN_TP = _TP
    else:
        group_ranks = []
        for tp_group_idx in range(num_tensor_model_parallel_groups):
            for cp_dp_combined_idx in range(attn_cp_size * attn_dp_size):
                st = (
                    tp_group_idx * tensor_model_parallel_size
                    + cp_dp_combined_idx * attn_tp_size
                )
                en = (
                    tp_group_idx * tensor_model_parallel_size
                    + (cp_dp_combined_idx + 1) * attn_tp_size
                )
                ranks = list(range(st, en))
                group_ranks.append(ranks)

        _ATTN_TP = init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            use_pynccl=SYNC_TOKEN_IDS_ACROSS_TP or enable_symm_mem,
            use_mscclpp_allreduce=False,
            use_custom_allreduce=False,
            use_torch_symm_mem_allreduce=False,
            use_message_queue_broadcaster=envs.SGLANG_USE_MESSAGE_QUEUE_BROADCASTER.get(),
            group_name="attention_tp",
            recovered_rank=recovered_rank,
        )

    moe_ep_size = expert_model_parallel_size
    moe_dp_size = moe_data_model_parallel_size
    moe_tp_size = tensor_model_parallel_size // moe_ep_size // moe_dp_size

    global _MOE_DP
    assert _MOE_DP is None, "moe data parallel group is already initialized"
    if attn_cp_size > moe_dp_size:
        # When moe_dp_size < attn_cp_size, CP ranks must share tokens before MoE.
        # The MOE_DP group includes these CP partners, so the existing DP
        # allgather/scatter handles the token sharing.
        _MOE_DP = _ATTN_CP
    elif moe_dp_size == tensor_model_parallel_size:
        _MOE_DP = _TP
    else:
        group_ranks = []
        for tp_group_idx in range(num_tensor_model_parallel_groups):
            for tp_ep_combined_idx in range(moe_tp_size * moe_ep_size):
                st = tp_group_idx * tensor_model_parallel_size + tp_ep_combined_idx
                en = (
                    tp_group_idx + 1
                ) * tensor_model_parallel_size + tp_ep_combined_idx
                ranks = list(range(st, en, moe_tp_size * moe_ep_size))
                group_ranks.append(ranks)
        _MOE_DP = init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            group_name="moe_dp",
            recovered_rank=recovered_rank,
        )

    global _MOE_EP
    assert _MOE_EP is None, "expert model parallel group is already initialized"
    # NPU requires a standalone group for MOE expert parallelism
    if moe_ep_size == tensor_model_parallel_size and not _is_npu:
        _MOE_EP = _TP
    else:
        group_ranks = []
        for tp_group_idx in range(num_tensor_model_parallel_groups):
            for moe_dp_idx in range(moe_dp_size):
                for moe_tp_idx in range(moe_tp_size):
                    st = (
                        tp_group_idx * tensor_model_parallel_size
                        + moe_dp_idx * moe_ep_size * moe_tp_size
                        + moe_tp_idx
                    )
                    en = st + moe_ep_size * moe_tp_size
                    ranks = list(range(st, en, moe_tp_size))
                    group_ranks.append(ranks)
        _MOE_EP = init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            use_pynccl=False,
            use_custom_allreduce=False,
            group_name="moe_ep",
            recovered_rank=recovered_rank,
        )

    global _MOE_TP
    assert _MOE_TP is None, "expert model parallel group is already initialized"
    if moe_tp_size == tensor_model_parallel_size:
        _MOE_TP = _TP
    else:
        group_ranks = []
        for tp_group_idx in range(num_tensor_model_parallel_groups):
            for ep_dp_combined_idx in range(moe_ep_size * moe_dp_size):
                st = (
                    tp_group_idx * tensor_model_parallel_size
                    + ep_dp_combined_idx * moe_tp_size
                )
                en = (
                    tp_group_idx * tensor_model_parallel_size
                    + (ep_dp_combined_idx + 1) * moe_tp_size
                )
                ranks = list(range(st, en))
                group_ranks.append(ranks)
        _MOE_TP = init_model_parallel_group(
            group_ranks,
            get_world_group().local_rank,
            backend,
            use_pynccl=False,
            use_custom_allreduce=False,
            group_name="moe_tp",
            recovered_rank=recovered_rank,
        )

    # Build the pipeline model-parallel groups.
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    global _PP
    assert _PP is None, "pipeline model parallel group is already initialized"
    group_ranks = []
    for pp_group_idx in range(num_pipeline_model_parallel_groups):
        ranks = list(
            range(pp_group_idx, world_size, num_pipeline_model_parallel_groups)
        )
        group_ranks.append(ranks)
    # pipeline parallel does not need custom allreduce
    _PP = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_custom_allreduce=False,
        group_name="pp",
        recovered_rank=recovered_rank,
    )

    # #622: lockstep sentinel (env-gated, off by default). Its sidecar needs
    # a DEDICATED gloo group: the census's cpu_group is used by the scheduler
    # thread, and two threads issuing collectives on one gloo group corrupt
    # each other's op ordering. Created here because every rank passes
    # through this function, making the new_group call collective. V1 scope:
    # single-node pure TP (world == tp size) — the positional merge of the
    # world/tp/dcp streams is only comparable when every group spans the
    # same ranks.
    from sglang.srt.distributed.device_communicators import lockstep_sentinel

    if lockstep_sentinel.enabled():
        _world = torch.distributed.get_world_size()
        if _world == tensor_model_parallel_size and pipeline_model_parallel_size == 1:
            _sentinel_pg = torch.distributed.new_group(
                ranks=list(range(_world)),
                backend="gloo",
                timeout=timedelta(seconds=60),
            )
            lockstep_sentinel.install(
                torch.distributed.get_rank(), _world, _sentinel_pg
            )
        else:
            logger.warning(
                "lockstep sentinel requested but world_size (%d) != "
                "tensor_parallel_size (%d) or PP > 1 — NOT armed (the "
                "positional stream merge requires identical group membership)",
                torch.distributed.get_world_size(),
                tensor_model_parallel_size,
            )


def phase_flip_groups_initialized() -> bool:
    return _FLIP_TP is not None


def get_phase_flip_group(name: str) -> GroupCoordinator:
    """Secondary-set accessor for #631 Route A: ``tp``, ``dcp`` or ``pp``."""
    group = {"tp": _FLIP_TP, "dcp": _FLIP_DCP, "pp": _FLIP_PP}.get(name)
    if name not in ("tp", "dcp", "pp"):
        raise ValueError(f"unknown phase-flip group {name!r}")
    assert group is not None, (
        f"phase-flip secondary group {name!r} is not initialized "
        f"(initialize_phase_flip_secondary_groups was not called, or dcp "
        f"was not requested)"
    )
    return group


def plan_decoupled_kv_ranks(
    world_size: int, tp_size: int, pp_size: int
) -> List[List[int]]:
    """#704b B1 rank sets: one group per TP position, spanning its PP stages.

    KV is token-sharded across the ranks of ONE pipeline, so the membership is
    exactly the PP group's -- ``range(idx, world_size, world_size // pp_size)``.

    Computed INLINE rather than read from ``_PP``, and that is load-bearing:
    ``_DCP`` is created at ``:3152`` and ``_PP`` only at ``:3365``, so at the
    point this runs ``_PP`` does not exist yet. Reading it would be the
    ordering bug the #616 survey called out.
    """
    if world_size <= 0 or tp_size <= 0 or pp_size <= 0:
        raise ValueError("world/tp/pp sizes must be positive.")
    if tp_size * pp_size != world_size:
        raise ValueError(
            f"tp_size {tp_size} x pp_size {pp_size} != world_size {world_size}; "
            "the decoupled-KV membership is derived from the pipeline layout "
            "and cannot be guessed from a world that does not factor."
        )
    num_pp_groups = world_size // pp_size
    return [list(range(idx, world_size, num_pp_groups)) for idx in range(num_pp_groups)]


def decoupled_kv_manifest(planned: object, salt: int = 0) -> str:
    """The string every rank must agree on BEFORE any group is created."""
    return repr((planned, int(salt)))


def check_manifest_agreement(
    manifest: str, gathered: Sequence[Optional[str]], label: str
) -> None:
    """Refuse to create anything when the ranks do not agree.

    Same contract as the phase-flip precedent (``:3484-3497``): a divergent
    create order is the #431/#616B/#645 rank-divergent-collective family, and
    dying here is the cheap failure -- the expensive one is a half-formed
    communicator that hangs a later collective with no attribution.
    """
    if any(m != manifest for m in gathered):
        raise RuntimeError(
            f"{label} group-creation manifest DIVERGES across ranks -- "
            "refusing to create any group. Per-rank manifests: "
            f"{dict(enumerate(gathered))}"
        )


def refuse_pp_dcp_combination(pp_size: int, dcp_size: int) -> None:
    """#616 labeled gap, closed loudly instead of left latent.

    ``pp>1`` with ``dcp>1`` is not refused anywhere in the group path today.
    It is unreachable on our configuration, so the contradiction it would
    create -- ``dcp_enabled`` True on PP prefill ranks against the invariant
    ``dcp_group_guard.py:38-42`` documents -- stays latent. B1 does NOT reuse
    ``dcp_size`` precisely so that invariant keeps holding, and this refusal
    makes the gap loud rather than merely unexercised.
    """
    if int(pp_size) > 1 and int(dcp_size) > 1:
        raise RuntimeError(
            f"pipeline_parallel_size={pp_size} with "
            f"decode_context_parallel_size={dcp_size} is not supported: "
            "ParallelContext.dcp_enabled would report True on PP prefill "
            "ranks, contradicting the invariant dcp_group_guard.py:38-42 "
            "documents ('a PP prefill group runs with dcp_size == 1 and no "
            "DCP group'). The guard itself would still PASS, so nothing would "
            "announce it. For token-sharded KV under PP use the #704b "
            "decoupled-KV group, which carries its own flag."
        )


def initialize_decoupled_kv_group(
    world_size: int,
    tp_size: int,
    pp_size: int,
    backend: Optional[str] = None,
    _manifest_salt: int = 0,
) -> None:
    """Build the #704b B1 group, reusing the phase-flip creation pattern.

    Fixed plan -> world-wide manifest all_gather -> equality check -> create.
    Nothing is created before every rank has agreed on the same plan.

    ``_manifest_salt`` exists solely so a test can prove the check can FAIL,
    exactly as the phase-flip precedent uses it.
    """
    global _DECOUPLED_KV
    if _DECOUPLED_KV is not None:
        raise RuntimeError("the decoupled-KV group is already initialized")

    planned = plan_decoupled_kv_ranks(world_size, tp_size, pp_size)
    manifest = decoupled_kv_manifest(planned, _manifest_salt)
    gathered: List[Optional[str]] = [None] * world_size
    torch.distributed.all_gather_object(
        gathered, manifest, group=get_world_group().cpu_group
    )
    check_manifest_agreement(manifest, gathered, "DECOUPLED-KV")

    backend = backend or torch.distributed.get_backend(get_world_group().device_group)
    _DECOUPLED_KV = init_model_parallel_group(
        planned,
        get_world_group().local_rank,
        backend,
        group_name="decoupled_kv",
    )
    logger.info("#704b decoupled-KV group built over pipeline ranks %s", planned)


def initialize_phase_flip_secondary_groups(
    *,
    tp_size: int,
    pp_size: int,
    dcp_size: int = 1,
    backend: Optional[str] = None,
    _manifest_salt: int = 0,
) -> None:
    """Build the #631 SECONDARY (flip-target) group set over the same world.

    The primary topology (from ``initialize_model_parallel``) serves one
    phase; this set serves the other (Route A: primary tp=1/pp=3 for the
    PP prefill phase, secondary tp=N/dcp=N/pp=1 for the TP decode phase).
    Runtime code routes between the sets via ``get_parallel().override``
    scopes -- no global here is mutated at flip time.

    DISCIPLINE (operator pin 1, the #431/#616B/#645 rank-divergent
    collective family): group creation is itself a collective, so the
    INTENDED creation manifest -- group names, rank lists, in the one
    fixed creation order tp -> dcp -> pp -- is exchanged and
    equality-checked across the WORLD **before** the first create. Ranks
    that disagree die loudly here, never inside a half-built communicator.
    ``_manifest_salt`` exists solely so the pin test can prove the check
    can fail; production callers never pass it.
    """
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)
    if tp_size * pp_size != world_size:
        raise RuntimeError(
            f"phase-flip secondary set: tp_size ({tp_size}) x pp_size "
            f"({pp_size}) != world_size ({world_size})"
        )
    if dcp_size > 1 and tp_size % dcp_size != 0:
        raise RuntimeError(
            f"phase-flip secondary set: tp_size ({tp_size}) not divisible "
            f"by dcp_size ({dcp_size})"
        )
    global _FLIP_TP, _FLIP_DCP, _FLIP_PP
    assert _FLIP_TP is None and _FLIP_DCP is None and _FLIP_PP is None, (
        "phase-flip secondary groups are already initialized"
    )

    # Plan first (pure), in THE fixed creation order.
    tp_ranks = [
        list(range(i * tp_size, (i + 1) * tp_size))
        for i in range(world_size // tp_size)
    ]
    planned = [("flip_tp", tp_ranks)]
    dcp_ranks = None
    if dcp_size > 1:
        dcp_ranks = []
        for group in tp_ranks:
            for start in range(0, len(group), dcp_size):
                dcp_ranks.append(group[start : start + dcp_size])
        planned.append(("flip_dcp", dcp_ranks))
    num_pp_groups = world_size // pp_size
    pp_ranks = [
        list(range(idx, world_size, num_pp_groups)) for idx in range(num_pp_groups)
    ]
    planned.append(("flip_pp", pp_ranks))

    # Verify the manifest group-wide BEFORE creating anything.
    manifest = repr((planned, int(_manifest_salt)))
    gathered: List[Optional[str]] = [None] * world_size
    torch.distributed.all_gather_object(
        gathered, manifest, group=get_world_group().cpu_group
    )
    if any(m != manifest for m in gathered):
        raise RuntimeError(
            f"{'PHASE-FLIP'} group-creation manifest DIVERGES across ranks "
            f"-- refusing to create any group (a divergent create order is "
            f"the #431/#616B/#645 rank-divergent-collective family; dying "
            f"here is the cheap failure). Per-rank manifests: "
            f"{dict(enumerate(gathered))}"
        )

    # Create, in exactly the planned order.
    local_rank = get_world_group().local_rank
    _FLIP_TP = init_model_parallel_group(
        tp_ranks, local_rank, backend, group_name="flip_tp"
    )
    if dcp_ranks is not None:
        _FLIP_DCP = init_model_parallel_group(
            dcp_ranks, local_rank, backend, group_name="flip_dcp"
        )
    _FLIP_PP = init_model_parallel_group(
        pp_ranks,
        local_rank,
        backend,
        use_custom_allreduce=False,
        group_name="flip_pp",
    )
    logger.info(
        "phase-flip secondary groups built: tp %s, dcp %s, pp %s",
        tp_ranks,
        dcp_ranks,
        pp_ranks,
    )


def _object_exchange_group() -> Optional[torch.distributed.ProcessGroup]:
    """The CPU process group the rank-config exchange must run on.

    Upstream sgl-project/sglang#32751: ``all_gather_object`` with no ``group=``
    resolves to WORLD, and torch then picks the staging device for the
    size-exchange tensor via ``_get_object_coll_device(group)``. When WORLD has
    several registered backends and none of them is CPU, that helper takes its
    documented "no cpu in the backend list, randomly pick the first backend"
    branch -- so a function whose entire purpose is to build a ``gloo`` group
    stages its metadata on whichever accelerator backend happens to be listed
    first. On MetaX/MACA that surfaced as an intermittent
    ``CUDA error: invalid argument`` roughly one launch in a few; on stock
    NVIDIA it is silent and merely non-deterministic.

    The world group already owns a gloo ``cpu_group`` (built for exactly this
    class of host-side coordination), so name it. That removes the device
    question instead of answering it -- the same rule the card-identity work
    (#397, #406, #394) applies to ordinals: never let an implicit enumeration
    decide which device a collective touches.

    Returns ``None`` (= torch's default, the previous behavior) only when the
    world group has not been built yet, which cannot happen at the production
    call site (HiCache prefetch-sync groups are created during scheduler
    init, long after parallel state) but keeps this usable from tests and
    from a bare ``torch.distributed`` setup.
    """
    if _WORLD is None:
        return None
    return _WORLD.cpu_group


def create_custom_parallel_group(
    group_ranks: List[int], backend: str = "gloo"
) -> Optional[torch.distributed.ProcessGroup]:
    """
    Create a custom parallel group based on the provided ranks.

    Args:
        group_ranks: The list of ranks that the CURRENT process wants to join.
                     (e.g., Rank 0 passes [0...7], Rank 8 passes [8...15])
        backend: The communication backend (default: "gloo").

    Returns:
        The ProcessGroup if the current rank is in group_ranks, else None.
    """
    assert torch.distributed.is_initialized()

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    local_config = sorted(list(set(group_ranks)))
    gathered_configs = [None for _ in range(world_size)]

    torch.distributed.all_gather_object(
        gathered_configs, local_config, group=_object_exchange_group()
    )

    unique_groups = []
    seen_signatures = set()

    for config in gathered_configs:
        config_tuple = tuple(config)
        if config_tuple not in seen_signatures:
            seen_signatures.add(config_tuple)
            unique_groups.append(list(config_tuple))

    unique_groups.sort(key=lambda x: x[0])

    my_new_group = None

    for g_ranks in unique_groups:
        group = torch.distributed.new_group(ranks=g_ranks, backend=backend)

        if set(g_ranks) == set(local_config):
            my_new_group = group
            logger.debug(
                f"Rank {rank} successfully created/joined custom group: {g_ranks}"
            )

    return my_new_group


def ensure_model_parallel_initialized(
    tensor_model_parallel_size: int,
    expert_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    decode_context_parallel_size: int = 1,
    backend: Optional[str] = None,
) -> None:
    """Helper to initialize model parallel groups if they are not initialized,
    or ensure tensor-parallel and pipeline-parallel sizes are equal to expected
    values if the model parallel groups are initialized.
    """
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)
    if not model_parallel_is_initialized():
        initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            decode_context_parallel_size=decode_context_parallel_size,
            backend=backend,
        )
        return

    assert get_tensor_model_parallel_world_size() == tensor_model_parallel_size, (
        "tensor parallel group already initialized, but of unexpected size: "
        f"{get_tensor_model_parallel_world_size()=} vs. "
        f"{tensor_model_parallel_size=}"
    )
    pp_world_size = get_pp_group().world_size
    assert pp_world_size == pipeline_model_parallel_size, (
        "pipeline parallel group already initialized, but of unexpected size: "
        f"{pp_world_size=} vs. "
        f"{pipeline_model_parallel_size=}"
    )
    if decode_context_parallel_size > 1:
        dcp_world_size = get_dcp_group().world_size
        assert dcp_world_size == decode_context_parallel_size, (
            f"decode context parallel group already initialized, but of unexpected size: {dcp_world_size=} {decode_context_parallel_size=}"
        )


def model_parallel_is_initialized():
    """Check if tensor and pipeline parallel groups are initialized."""
    return _TP is not None and _PP is not None


_TP_STATE_PATCHED = False


@contextmanager
def patch_tensor_parallel_group(tp_group: GroupCoordinator):
    """Patch the tp group temporarily until this function ends.

    This method is for draft workers of speculative decoding to run draft model
    with different tp degree from that of target model workers.

    Args:
        tp_group (GroupCoordinator): the tp group coordinator
    """
    global _TP_STATE_PATCHED
    assert not _TP_STATE_PATCHED, "Should not call when it's already patched"

    _TP_STATE_PATCHED = True
    old_tp_group = get_tp_group()
    global _TP
    _TP = tp_group
    try:
        yield
    finally:
        # restore the original state
        _TP_STATE_PATCHED = False
        _TP = old_tp_group


def get_world_size():
    """Return world size for the world group."""
    return get_world_group().world_size


def get_world_rank():
    """Return my rank for the world group."""
    return get_world_group().rank_in_group


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return get_tp_group().world_size


def get_dcp_world_size():
    return get_dcp_group().world_size


def get_dcp_rank():
    return get_dcp_group().rank_in_group


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    return get_tp_group().rank_in_group


# ATTN_TP
def get_attn_tensor_model_parallel_world_size():
    """Return world size for the attention tensor model parallel group."""
    return get_attn_tp_group().world_size


def get_attn_tensor_model_parallel_rank():
    """Return my rank for the attention tensor model parallel group."""
    return get_attn_tp_group().rank_in_group


# ATTN_CP
def get_attn_context_model_parallel_world_size():
    """Return world size for the attention context model parallel group."""
    return get_attn_cp_group().world_size


def get_attn_context_model_parallel_rank():
    """Return my rank for the attention context model parallel group."""
    return get_attn_cp_group().rank_in_group


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    return get_pp_group().world_size


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    return get_pp_group().rank_in_group


# MOE_DP
def get_moe_data_parallel_world_size():
    """Return world size for the moe data parallel group."""
    return get_moe_dp_group().world_size


def get_moe_data_parallel_rank():
    """Return my rank for the moe data parallel group."""
    return get_moe_dp_group().rank_in_group


# MOE_EP
def get_moe_expert_parallel_world_size():
    """Return world size for the moe expert parallel group."""
    return get_moe_ep_group().world_size


def get_moe_expert_parallel_rank():
    """Return my rank for the moe expert parallel group."""
    return get_moe_ep_group().rank_in_group


# MOE_TP
def get_moe_tensor_parallel_world_size():
    """Return world size for the moe tensor parallel group."""
    return get_moe_tp_group().world_size


def get_moe_tensor_parallel_rank():
    """Return my rank for the moe tensor parallel group."""
    return get_moe_tp_group().rank_in_group


def destroy_model_parallel():
    """Set the groups to none and destroy them."""
    global _TP
    if _TP:
        _TP.destroy()
    _TP = None

    global _PP
    if _PP:
        _PP.destroy()
    _PP = None

    global _DCP
    if _DCP:
        _DCP.destroy()
    _DCP = None

    global _DCP_SPILL, _DCP_SPILL_ACTIVE
    if _DCP_SPILL:
        _DCP_SPILL.destroy()
    _DCP_SPILL = None
    _DCP_SPILL_ACTIVE = False

    global _FLIP_TP, _FLIP_DCP, _FLIP_PP, _PHASE_FLIP_TP_ACTIVE
    _PHASE_FLIP_TP_ACTIVE = False
    if _FLIP_TP:
        _FLIP_TP.destroy()
    _FLIP_TP = None
    if _FLIP_DCP:
        _FLIP_DCP.destroy()
    _FLIP_DCP = None
    if _FLIP_PP:
        _FLIP_PP.destroy()
    _FLIP_PP = None

    global _MOE_EP
    if _MOE_EP:
        _MOE_EP.destroy()
    _MOE_EP = None

    global _MOE_TP
    if _MOE_TP:
        _MOE_TP.destroy()
    _MOE_TP = None

    global _ATTN_CP
    global _MOE_DP
    # Destroy _MOE_DP before _ATTN_CP since it may alias _ATTN_CP.
    # Only destroy if not aliasing another group.
    if _MOE_DP and _MOE_DP is not _ATTN_CP and _MOE_DP is not _TP:
        _MOE_DP.destroy()
    _MOE_DP = None
    if _ATTN_CP:
        _ATTN_CP.destroy()
    _ATTN_CP = None

    global _ATTN_TP
    if _ATTN_TP:
        _ATTN_TP.destroy()
    _ATTN_TP = None

    global _PDMUX_PREFILL_TP_GROUP
    if _PDMUX_PREFILL_TP_GROUP:  # type: ignore[union-attr]
        _PDMUX_PREFILL_TP_GROUP.destroy()
    _PDMUX_PREFILL_TP_GROUP = None


def destroy_distributed_environment():
    global _WORLD, _MODEL_PARALLEL_GROUP_TIMEOUT
    if _WORLD:
        _WORLD.destroy()
    _WORLD = None
    _MODEL_PARALLEL_GROUP_TIMEOUT = None
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def cleanup_dist_env_and_memory(shutdown_ray: bool = False):
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    if shutdown_ray:
        import ray  # Lazy import Ray

        ray.shutdown()
    gc.collect()
    if not _is_cpu:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch._C, "_host_emptyCache"):
                torch._C._host_emptyCache()
            else:
                logger.warning(
                    "torch._C._host_emptyCache() only available in Pytorch >=2.5"
                )
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
        elif hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.empty_cache()
        elif hasattr(torch, "musa") and torch.musa.is_available():
            torch.musa.empty_cache()


def in_the_same_node_as(pg: ProcessGroup, source_rank: int = 0) -> List[bool]:
    """
    This is a collective operation that returns if each rank is in the same node
    as the source rank. It tests if processes are attached to the same
    memory system (shared access to shared memory).
    """
    assert torch.distributed.get_backend(pg) != torch.distributed.Backend.NCCL, (
        "in_the_same_node_as should be tested with a non-NCCL group."
    )
    # local rank inside the group
    rank = torch.distributed.get_rank(group=pg)
    world_size = torch.distributed.get_world_size(group=pg)

    # local tensor in each process to store the result
    is_in_the_same_node = torch.tensor([0] * world_size, dtype=torch.int32)

    # global ranks of the processes in the group
    ranks = torch.distributed.get_process_group_ranks(pg)

    magic_message = b"magic_message"
    shm = None

    try:
        with contextlib.suppress(OSError):
            if rank == source_rank:
                # create a shared memory segment
                shm = shared_memory.SharedMemory(
                    create=True, size=128, name=make_shm_name("nodecheck")
                )
                shm.buf[: len(magic_message)] = magic_message
                torch.distributed.broadcast_object_list(
                    [shm.name], src=ranks[source_rank], group=pg
                )
                is_in_the_same_node[rank] = 1
            else:
                # try to open the shared memory segment
                recv = [None]
                torch.distributed.broadcast_object_list(
                    recv, src=ranks[source_rank], group=pg
                )
                name = recv[0]
                # fix to https://stackoverflow.com/q/62748654/9191338
                # Python incorrectly tracks shared memory even if it is not
                # created by the process. The following patch is a workaround.
                with patch(
                    "multiprocessing.resource_tracker.register",
                    lambda *args, **kwargs: None,
                ):
                    shm = shared_memory.SharedMemory(name=name)
                if shm.buf[: len(magic_message)] == magic_message:
                    is_in_the_same_node[rank] = 1
    except Exception as e:
        logger.error("Error ignored in is_in_the_same_node: %s", e)
    finally:
        if shm:
            shm.close()

    torch.distributed.barrier(group=pg)

    # clean up the shared memory segment
    with contextlib.suppress(OSError):
        if rank == source_rank and shm:
            shm.unlink()
    torch.distributed.all_reduce(is_in_the_same_node, group=pg)

    return [x == 1 for x in is_in_the_same_node.tolist()]


vllm_get_pp_group = None
vllm_get_tp_group = None
vllm_get_world_group = None


def monkey_patch_vllm_parallel_state(reverse: bool = False):
    try:
        import vllm.distributed.parallel_state as vllm_parallel_state
    except ImportError:
        return

    global vllm_get_pp_group, vllm_get_tp_group, vllm_get_world_group
    if vllm_get_pp_group is None:
        vllm_get_pp_group = vllm_parallel_state.get_pp_group
        vllm_get_tp_group = vllm_parallel_state.get_tp_group
        vllm_get_world_group = vllm_parallel_state.get_world_group
    if reverse:
        setattr(vllm_parallel_state, "get_pp_group", vllm_get_pp_group)
        setattr(vllm_parallel_state, "get_tp_group", vllm_get_tp_group)
        setattr(vllm_parallel_state, "get_world_group", vllm_get_world_group)
    else:
        setattr(vllm_parallel_state, "get_pp_group", get_pp_group)
        setattr(vllm_parallel_state, "get_tp_group", get_tp_group)
        setattr(vllm_parallel_state, "get_world_group", get_world_group)
