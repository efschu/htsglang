# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/utils.py

# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/utils.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import dataclasses
import logging
import os
import pickle
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Sequence, Tuple

import torch
from torch.distributed import TCPStore

logger = logging.getLogger(__name__)


def set_global_tcp_store(store: TCPStore) -> None:
    """Install the shared TCPStore created during distributed initialization;
    the handle lives on ``ctx.resources``."""
    from sglang.srt.runtime_context import get_resources

    get_resources().tcp_store = store
    logger.info("Global TCPStore has been set")


def get_global_tcp_store() -> Optional[TCPStore]:
    """Get the existing global TCPStore.

    This function provides access to the shared TCPStore instance that was
    created during distributed initialization. All components (like NIXL buffers)
    should use this same store for coordination.

    Returns:
        The global TCPStore instance, or None if not initialized yet.
    """
    from sglang.srt.runtime_context import get_resources

    store = get_resources().tcp_store
    if store is None:
        logger.warning(
            "Global TCPStore not found. Make sure init_distributed_environment "
            "was called with a tcp:// init method."
        )
    return store


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


# ---------------------------------------------------------------------------
# Uneven tensor-parallel partitioning (--rank-tp-ratio).
#
# When a ratio vector like (2, 1, 1) is active, TP rank r owns
# total * ratio[r] / sum(ratio) of every sharded dimension instead of
# total / tp_size. Offsets become prefix sums. The ratio vector is
# process-global state installed once per scheduler process (from
# server_args.rank_tp_ratio) before the model is built; when unset, all
# helpers reproduce the classic even split (divide) exactly, so the
# default path stays unchanged.
#
# NAMED FAMILY PLANS (--rank-mlp-ratio / SGLANG_UNEVEN_MLP_VECTOR): on top
# of the base vector, individual weight FAMILIES (today: "mlp", the dense
# MLP intermediate dimension) may carry their own weight vector. Layers
# opt in by passing tp_family=<name>; families without an installed
# vector fall back to the base plan, so passing a family is always safe
# and the base behavior is unchanged. This is the calibration lever for
# maximizing the KV pool: shifting MLP units between ranks re-balances
# the per-rank weight bytes without touching attention/KV-head splits.
# ---------------------------------------------------------------------------

_TP_PARTITION_RATIOS: Optional[list] = None
_TP_PARTITION_FAMILIES: dict = {}


def set_tp_partition_ratios(
    ratios: Optional[Sequence[int]],
    families: Optional[dict] = None,
) -> None:
    """Install the uneven-TP ratio vector for this process (or None).

    `families` optionally maps family names (e.g. "mlp") to their own
    weight vectors, overriding the base vector for layers constructed
    with a matching tp_family. Families are only valid together with a
    base vector and must have the same length; empty/None entries are
    ignored. Every call replaces the complete plan (base + families)."""
    global _TP_PARTITION_RATIOS, _TP_PARTITION_FAMILIES
    base = list(ratios) if ratios else None
    fams: dict = {}
    if families:
        for name, vec in families.items():
            vec = list(vec) if vec else None
            if not vec:
                continue
            if base is None:
                raise ValueError(
                    f"Family shard plan {name!r} requires an active base "
                    "plan (--rank-tp-ratio)."
                )
            if len(vec) != len(base):
                raise ValueError(
                    f"Family shard plan {name!r} has {len(vec)} entries "
                    f"but the base plan has {len(base)} "
                    f"({vec} vs {base})."
                )
            if any(not isinstance(w, int) or w <= 0 for w in vec):
                raise ValueError(
                    f"Family shard plan {name!r} entries must be positive "
                    f"integers, got {vec}."
                )
            fams[name] = vec
    _TP_PARTITION_RATIOS = base
    _TP_PARTITION_FAMILIES = fams


def get_tp_partition_ratios(family: Optional[str] = None) -> Optional[list]:
    """The installed weight vector: the family's own vector when one is
    installed under `family`, otherwise the base vector (or None)."""
    if family is not None:
        vec = _TP_PARTITION_FAMILIES.get(family)
        if vec is not None:
            return vec
    return _TP_PARTITION_RATIOS


def partition_units(units: int, weights: Sequence[int]) -> list:
    """Split `units` indivisible units over ranks proportionally to
    `weights` (largest-remainder rounding, every rank gets >= 1 unit).

    Deterministic pure function of (units, weights) so every process
    computes the identical partition. Ties in the fractional parts are
    broken toward the lower rank index.
    """
    n = len(weights)
    if units < n:
        raise ValueError(
            f"Cannot give each of {n} ranks at least one of {units} units."
        )
    total_w = sum(weights)
    quotas = [units * w / total_w for w in weights]
    sizes = [int(q) for q in quotas]
    # Reserve a minimum of one unit per rank before distributing the rest.
    sizes = [max(s, 1) for s in sizes]
    remaining = units - sum(sizes)
    if remaining < 0:
        # Minimum-1 bumping overshot: take back from the largest shares.
        for _ in range(-remaining):
            i = max(range(n), key=lambda r: (sizes[r], -r))
            sizes[i] -= 1
        remaining = 0
    order = sorted(
        range(n), key=lambda r: (quotas[r] - int(quotas[r]), -r), reverse=True
    )
    for k in range(remaining):
        sizes[order[k % n]] += 1
    assert sum(sizes) == units and all(s >= 1 for s in sizes)
    return sizes


def partition_sizes(
    total: int, weights: Sequence[int], units: Optional[int] = None
) -> list:
    """Per-rank sizes of a sharded dimension of `total` elements under the
    weight vector `weights`.

    With `units`, the dimension is treated as `units` indivisible units of
    `total // units` elements each (e.g. attention heads): the units are
    distributed by largest-remainder rounding (every rank >= 1 unit) and
    scaled back to elements, so any positive weights work. `total` must be
    a multiple of `units`.

    Without `units`, per-rank sizes must be exact: `total` must be
    divisible by sum(weights); otherwise this raises, naming the offending
    dimension size.
    """
    if units is not None:
        if total % units != 0:
            raise ValueError(
                f"Dimension of size {total} is not a multiple of its "
                f"unit count {units}."
            )
        scale = total // units
        return [s * scale for s in partition_units(units, weights)]
    denom = sum(weights)
    if total % denom != 0:
        raise ValueError(
            f"Cannot partition dimension of size {total} with weight "
            f"vector {list(weights)}: {total} is not divisible by "
            f"sum(weights)={denom}. Choose weights whose sum divides every "
            "sharded dimension, or pass the dimension's unit count."
        )
    unit = total // denom
    return [unit * w for w in weights]


def partition_offsets(
    total: int, weights: Sequence[int], rank: int, units: Optional[int] = None
) -> Tuple[int, int]:
    """(start offset, size) of `rank` in a sharded dimension of `total`
    elements: the prefix sum over partition_sizes and this rank's share."""
    sizes = partition_sizes(total, weights, units)
    return sum(sizes[:rank]), sizes[rank]


def tp_partition_sizes(
    total: int,
    tp_size: int,
    units: Optional[int] = None,
    family: Optional[str] = None,
) -> list:
    """Per-rank sizes of a sharded dimension under the process-global
    shard plan. Without an installed ratio vector (or when this layer runs
    with its own tp_size, e.g. disable_tp layers use tp_size=1), this is
    the classic even split via divide(). `family` selects a named family
    plan (e.g. "mlp") and falls back to the base vector when that family
    has no own vector installed."""
    ratios = get_tp_partition_ratios(family)
    if not ratios or len(ratios) != tp_size:
        ensure_divisibility(total, tp_size)
        return [total // tp_size] * tp_size
    return partition_sizes(total, ratios, units)


def tp_partition_size(
    total: int,
    tp_size: int,
    rank: int,
    units: Optional[int] = None,
    family: Optional[str] = None,
) -> int:
    """This rank's size of a sharded dimension under the global plan."""
    return tp_partition_sizes(total, tp_size, units, family)[rank]


def tp_partition_offset(
    total: int,
    tp_size: int,
    rank: int,
    units: Optional[int] = None,
    family: Optional[str] = None,
) -> int:
    """This rank's start offset (prefix sum) in a sharded dimension."""
    return sum(tp_partition_sizes(total, tp_size, units, family)[:rank])


def tp_plan_active(tp_size: int, family: Optional[str] = None) -> bool:
    """True when an uneven-TP ratio plan is installed AND applies to a
    layer/group of the given tp_size (disable_tp layers with tp_size=1 and
    groups of a different size keep the classic even split)."""
    ratios = get_tp_partition_ratios(family)
    return bool(ratios) and len(ratios) == tp_size


def tp_loaded_shard_start(
    loaded_full: int,
    tp_size: Optional[int],
    rank: int,
    shard_size: int,
    units: Optional[int] = None,
    family: Optional[str] = None,
) -> int:
    """Start offset when narrowing a full checkpoint dimension of
    `loaded_full` elements down to this rank's shard of `shard_size`.

    Even TP (no ratio plan installed, or the plan does not match
    `tp_size`): `rank * shard_size` -- the classic formula, bit-for-bit
    the previous behavior. Uneven TP (--rank-tp-ratio): the prefix sum of
    the per-rank partition sizes (with `units` = the dimension's
    indivisible unit count, e.g. kv heads); the given `shard_size` must
    match this rank's partition, which cross-checks the parameter shape
    against the plan.

    `tp_size=None` means "derive from the plan" (callers such as the
    parameter-class loaders that only know the rank); with no plan
    installed this still degrades to `rank * shard_size`. `family`
    selects a named family plan (falling back to the base vector) and
    must match the family the owning layer partitioned with.
    """
    ratios = get_tp_partition_ratios(family)
    if not ratios or (tp_size is not None and len(ratios) != tp_size):
        return rank * shard_size
    if shard_size == loaded_full:
        # Fully replicated component: every rank loads the whole
        # checkpoint dimension.
        return 0
    sizes = partition_sizes(loaded_full, ratios, units)
    if sizes[rank] != shard_size:
        raise ValueError(
            f"uneven-TP shard mismatch: expected size {sizes[rank]} for "
            f"rank {rank} of dimension {loaded_full} under weight vector "
            f"{list(ratios)} (units={units}), but the parameter shard "
            f"has {shard_size}."
        )
    return sum(sizes[:rank])


def solve_unit_rebalance(
    free_bytes: Sequence[float],
    bytes_per_token: Sequence[float],
    units: Sequence[int],
    bytes_per_unit: Sequence[float],
    min_units: int = 1,
) -> Tuple[list, int]:
    """Maximin solver for the uneven-TP self-calibration: find the
    per-rank MLP unit counts that maximize the MINIMUM token capacity.

    Model: rank r currently owns units[r] MLP units; its profiled KV byte
    budget is free_bytes[r] and one KV token costs bytes_per_token[r]
    (both rank-local — per-token bytes scale with the rank's kv-head
    share). Every MLP unit a rank sheds frees bytes_per_unit[r] weight
    bytes (measured empirically as the rank's mlp-family parameter bytes
    divided by its unit count), which become KV budget:

        capacity_r(u) = (free[r] + (units[r] - u[r]) * bytes_per_unit[r])
                        / bytes_per_token[r]

    Since the scheduler's single max_total_num_tokens is the MIN over
    ranks, the objective is maximin. Greedy over single units suffices:
    the minimum only ever rises when the pinned (capacity-poorest) rank
    sheds a unit, so iteratively move one unit from the pinned rank to
    the rank that stays capacity-richest after receiving it, as long as
    the minimum strictly increases. Every rank keeps >= min_units units
    (partition_units requires >= 1 per rank).

    Returns (new_units, projected_min_tokens).
    """
    n = len(units)
    if not (n == len(free_bytes) == len(bytes_per_token) == len(bytes_per_unit)):
        raise ValueError(
            "solve_unit_rebalance: input vectors must have equal length, "
            f"got {len(free_bytes)}/{len(bytes_per_token)}/{len(units)}/"
            f"{len(bytes_per_unit)}."
        )
    if any(b <= 0 for b in bytes_per_token) or any(u < min_units for u in units):
        raise ValueError(
            "solve_unit_rebalance: bytes_per_token must be positive and "
            f"every rank must own >= {min_units} unit(s); got "
            f"bytes_per_token={list(bytes_per_token)}, units={list(units)}."
        )

    u = list(units)

    def capacity(r: int, u_r: int) -> float:
        return (
            free_bytes[r] + (units[r] - u_r) * bytes_per_unit[r]
        ) / bytes_per_token[r]

    while True:
        caps = [capacity(r, u[r]) for r in range(n)]
        cur_min = min(caps)
        donor = min(range(n), key=lambda r: (caps[r], r))
        if u[donor] <= min_units:
            break
        # Receiver: the rank that remains capacity-richest after taking
        # on the unit's weight bytes (hurts the maximin objective least).
        receiver = max(
            (r for r in range(n) if r != donor),
            key=lambda r: (capacity(r, u[r] + 1), -r),
        )
        u_try = list(u)
        u_try[donor] -= 1
        u_try[receiver] += 1
        new_min = min(capacity(r, u_try[r]) for r in range(n))
        if new_min <= cur_min:
            break
        u = u_try

    projected = int(min(capacity(r, u[r]) for r in range(n)))
    return u, projected


def suggest_unit_rebalance(
    free_bytes: Sequence[float],
    bytes_per_token: Sequence[float],
    units: Sequence[int],
    family_bytes: Sequence[float],
    imbalance_threshold: float = 1.10,
) -> Optional[Tuple[list, int, int]]:
    """Decide whether shifting MLP units between uneven-TP ranks would
    meaningfully raise the (MIN-synced) KV token capacity.

    Inputs are the per-rank values gathered after rank-local KV
    profiling: profiled free bytes, bytes per KV token, current mlp
    family partition in units and the mlp family's total parameter bytes
    (bytes_per_unit is derived as family_bytes / units per rank).

    Returns None when the capacities are already balanced (max/min <=
    imbalance_threshold), when the inputs cannot be calibrated, or when
    the solver finds no strictly better partition. Otherwise returns
    (new_units, current_min_tokens, projected_min_tokens).
    """
    n = len(units)
    if n < 2:
        return None
    if any(u <= 0 for u in units) or any(b <= 0 for b in family_bytes):
        return None
    if any(f < 0 for f in free_bytes) or any(b <= 0 for b in bytes_per_token):
        return None
    capacities = [free_bytes[r] / bytes_per_token[r] for r in range(n)]
    cur_min = min(capacities)
    if cur_min <= 0:
        return None
    if max(capacities) / cur_min <= imbalance_threshold:
        return None
    bytes_per_unit = [family_bytes[r] / units[r] for r in range(n)]
    new_units, projected = solve_unit_rebalance(
        free_bytes, bytes_per_token, units, bytes_per_unit
    )
    if new_units == list(units) or projected <= int(cur_min):
        return None
    return new_units, int(cur_min), projected


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> Sequence[torch.Tensor]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # NOTE: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def get_pp_indices(
    num_hidden_layers: int, pp_rank: int, pp_size: int
) -> Tuple[int, int]:
    """Try to evenly distribute layers across partitions.
    If the number of layers is not divisible by the number of partitions,
    the last N partitions will have one extra layer, where N = remainder.
    """
    # partition_list_str can be set to None in sglang
    partition_list_str = os.getenv("SGLANG_PP_LAYER_PARTITION", None)
    if partition_list_str is not None:
        try:
            partitions = [int(layer) for layer in partition_list_str.split(",")]
        except ValueError as err:
            raise ValueError(
                "Invalid partition string: {}".format(partition_list_str)
            ) from err
        if len(partitions) != pp_size:
            raise ValueError(f"{len(partitions)=} does not match {pp_size=}.")
        if sum(partitions) != num_hidden_layers:
            raise ValueError(f"{sum(partitions)=} does not match {num_hidden_layers=}.")
        start_layer = sum(partitions[:pp_rank])
        end_layer = start_layer + partitions[pp_rank]
    else:
        base_layers = num_hidden_layers // pp_size
        remainder = num_hidden_layers % pp_size
        # Distribute the extra layers to the last 'remainder' partitions
        if pp_rank >= pp_size - remainder:
            partitions_without_extra_layer = pp_size - remainder
            # This partition gets one extra layer
            start_layer = pp_rank * (base_layers + 1) - partitions_without_extra_layer
            end_layer = start_layer + (base_layers + 1)
        else:
            # This partition gets only base layers
            start_layer = pp_rank * base_layers
            end_layer = start_layer + base_layers

    return (start_layer, end_layer)


@dataclasses.dataclass
class StatelessProcessGroup:
    """A dataclass to hold a metadata store, and the rank, world_size of the
    group. Only use it to communicate metadata between processes.
    For data-plane communication, create NCCL-related objects.
    """

    rank: int
    world_size: int
    store: torch._C._distributed_c10d.Store
    data_expiration_seconds: int = 3600  # 1 hour

    # dst rank -> counter
    send_dst_counter: Dict[int, int] = dataclasses.field(default_factory=dict)
    # src rank -> counter
    recv_src_counter: Dict[int, int] = dataclasses.field(default_factory=dict)
    broadcast_send_counter: int = 0
    broadcast_recv_src_counter: Dict[int, int] = dataclasses.field(default_factory=dict)

    # A deque to store the data entries, with key and timestamp.
    entries: Deque[Tuple[str, float]] = dataclasses.field(default_factory=deque)

    def __post_init__(self):
        assert self.rank < self.world_size
        self.send_dst_counter = {i: 0 for i in range(self.world_size)}
        self.recv_src_counter = {i: 0 for i in range(self.world_size)}
        self.broadcast_recv_src_counter = {i: 0 for i in range(self.world_size)}

    def send_obj(self, obj: Any, dst: int):
        """Send an object to a destination rank."""
        self.expire_data()
        key = f"send_to/{dst}/{self.send_dst_counter[dst]}"
        self.store.set(key, pickle.dumps(obj))
        self.send_dst_counter[dst] += 1
        self.entries.append((key, time.perf_counter()))

    def expire_data(self):
        """Expire data that is older than `data_expiration_seconds` seconds."""
        while self.entries:
            # check the oldest entry
            key, timestamp = self.entries[0]
            if time.perf_counter() - timestamp > self.data_expiration_seconds:
                self.store.delete_key(key)
                self.entries.popleft()
            else:
                break

    def recv_obj(self, src: int) -> Any:
        """Receive an object from a source rank."""
        obj = pickle.loads(
            self.store.get(f"send_to/{self.rank}/{self.recv_src_counter[src]}")
        )
        self.recv_src_counter[src] += 1
        return obj

    def broadcast_obj(self, obj: Optional[Any], src: int) -> Any:
        """Broadcast an object from a source rank to all other ranks.
        It does not clean up after all ranks have received the object.
        Use it for limited times, e.g., for initialization.
        """
        if self.rank == src:
            self.expire_data()
            key = f"broadcast_from/{src}/" f"{self.broadcast_send_counter}"
            self.store.set(key, pickle.dumps(obj))
            self.broadcast_send_counter += 1
            self.entries.append((key, time.perf_counter()))
            return obj
        else:
            key = f"broadcast_from/{src}/" f"{self.broadcast_recv_src_counter[src]}"
            recv_obj = pickle.loads(self.store.get(key))
            self.broadcast_recv_src_counter[src] += 1
            return recv_obj

    def all_gather_obj(self, obj: Any) -> list[Any]:
        """All gather an object from all ranks."""
        gathered_objs = []
        for i in range(self.world_size):
            if i == self.rank:
                gathered_objs.append(obj)
                self.broadcast_obj(obj, src=self.rank)
            else:
                recv_obj = self.broadcast_obj(None, src=i)
                gathered_objs.append(recv_obj)
        return gathered_objs

    def barrier(self):
        """A barrier to synchronize all ranks."""
        for i in range(self.world_size):
            if i == self.rank:
                self.broadcast_obj(None, src=self.rank)
            else:
                self.broadcast_obj(None, src=i)

    @staticmethod
    def create(
        host: str,
        port: int,
        rank: int,
        world_size: int,
        data_expiration_seconds: int = 3600,
    ) -> "StatelessProcessGroup":
        """A replacement for `torch.distributed.init_process_group` that does not
        pollute the global state.

        If we have process A and process B called `torch.distributed.init_process_group`
        to form a group, and then we want to form another group with process A, B, C,
        D, it is not possible in PyTorch, because process A and process B have already
        formed a group, and process C and process D cannot join that group. This
        function is a workaround for this issue.

        `torch.distributed.init_process_group` is a global call, while this function
        is a stateless call. It will return a `StatelessProcessGroup` object that can be
        used for exchanging metadata. With this function, process A and process B
        can call `StatelessProcessGroup.create` to form a group, and then process A, B,
        C, and D can call `StatelessProcessGroup.create` to form another group.
        """  # noqa
        store = TCPStore(
            host_name=host,
            port=port,
            world_size=world_size,
            is_master=(rank == 0),
        )

        return StatelessProcessGroup(
            rank=rank,
            world_size=world_size,
            store=store,
            data_expiration_seconds=data_expiration_seconds,
        )
