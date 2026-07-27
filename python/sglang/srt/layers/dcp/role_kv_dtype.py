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
"""Per-ROLE KV-cache storage precision for the weightless-KV fast lane (#127).

The weightless lane splits one TP/DCP group into two structurally different
roles: the HEAD rank carries the model weights and its KV token-shard, every
WEIGHTLESS worker carries a KV token-shard and nothing else. Their VRAM
budgets are therefore spent on completely different things, and there is no
reason the two roles must store KV at the same precision.

This module holds the PURE decision functions for that split -- no torch
tensors, no CUDA, no distributed state -- so the whole contract is unit
testable on CPU:

* :func:`effective_kv_cache_dtype_spec` -- which ``--kv-cache-dtype`` SPEC
  STRING a given rank resolves. The string is then fed through the existing,
  unmodified ``ModelRunner.configure_kv_cache_dtype()`` branch chain, so a
  per-role dtype is a change of INPUT to that resolver, never a second
  resolver that could drift from it.
* :func:`even_modulo_global_capacity` -- what the per-rank token capacities
  actually BUY under the lane's even-modulo owner rule. This is the honest
  arithmetic that decides whether a per-role precision change gains capacity
  at all (see the module docstring section below).
* :func:`host_tier_stride_mismatch` -- the byte-stride agreement between a
  device pool and a host overflow tier, as a message-or-``None`` check.

WHY THE CAPACITY FUNCTION LIVES HERE
------------------------------------
Under the lane's even-modulo DCP owner rule token L is owned by rank
``L % dcp_size`` and lands at compacted slot ``L // dcp_size`` -- the SAME
slot index on every rank. The slot space is therefore rank-uniform, and
``ModelRunner._apply_token_constraints`` MIN-reduces the per-rank token
capacities into one number. Consequence:

    global_capacity = dcp_size * min_r(per_rank_capacity_r)

Halving the per-token bytes on the WORKERS raises ``per_rank_capacity`` on
the workers only. If a worker is the binding (minimum) rank, the whole group
gains. If the HEAD is the binding rank -- the common case whenever the head's
weights eat most of its card -- worker-side precision buys exactly ZERO extra
tokens and only costs accuracy. That is not a reason to omit the feature (the
binding rank is a property of the deployment, not of the design), but it IS a
reason to compute and REPORT the binding rank at boot instead of assuming the
gain. :func:`even_modulo_global_capacity` is that computation.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

#: ``--kv-cache-dtype`` spec strings a weightless WORKER may be given.
#:
#: This is the global ``--kv-cache-dtype`` choice list MINUS ``fp4_e2m1``.
#: fp4 is excluded on purpose: an fp4 pool carries an extra per-block scale
#: buffer (``MemoryPoolConfigurator._compute_cell_size`` adds a second term
#: for it) and needs the ``MHATokenToKVPoolFP4`` variant, none of which has
#: ever been exercised on this lane. Refusing it is a scope statement, not a
#: capability claim.
WORKER_KV_CACHE_DTYPE_CHOICES: Tuple[str, ...] = (
    "auto",
    "fp8_e5m2",
    "fp8_e4m3",
    "bf16",
    "bfloat16",
)

#: Spec strings that store KV LOSSILY relative to the compute dtype. Used for
#: the opt-in/lossy-class labelling in logs and docs (quality-last rule): the
#: lane's default must never land here implicitly.
LOSSY_KV_CACHE_DTYPE_SPECS: Tuple[str, ...] = ("fp8_e5m2", "fp8_e4m3")


def effective_kv_cache_dtype_spec(
    global_spec: str,
    worker_spec: Optional[str],
    is_weightless_worker: bool,
) -> str:
    """The ``--kv-cache-dtype`` spec string THIS rank must resolve.

    ``worker_spec`` is ``--weightless-kv-worker-cache-dtype``; ``None`` (the
    default) and ``"auto"`` both mean "inherit the group-wide spec", which
    makes this function the identity and keeps every existing path -- default
    and lane alike -- byte-identical.

    The override applies ONLY on a weightless worker rank. The head rank and
    every non-lane rank always keep ``global_spec``; there is deliberately no
    way to reach the worker branch from the default path, so the new
    behaviour is gated by a single condition rather than interleaved.

    Note that this returns a SPEC STRING, not a torch dtype: the caller feeds
    it to the one existing resolver so the per-role path can never disagree
    with the global path about what ``fp8_e5m2`` means (HIP remaps it, "auto"
    consults the checkpoint, ...).
    """
    if not is_weightless_worker:
        return global_spec
    if worker_spec is None or worker_spec == "auto":
        return global_spec
    return worker_spec


def worker_dtype_is_role_split(
    global_spec: str,
    worker_spec: Optional[str],
) -> bool:
    """True when the worker spec genuinely differs from the group-wide spec.

    ``auto``/``None`` inherit, and an explicit worker spec equal to the global
    one is a no-op -- neither is a role split. Used to decide whether the
    extra boot diagnostics / grenz-assertions are worth emitting at all.
    """
    if worker_spec is None or worker_spec == "auto":
        return False
    if worker_spec == global_spec:
        return False
    # bf16/bfloat16 are the same spec spelled two ways.
    if {worker_spec, global_spec} == {"bf16", "bfloat16"}:
        return False
    return True


def even_modulo_global_capacity(
    per_rank_capacity: Sequence[int],
    dcp_size: int,
) -> Tuple[int, int, List[int]]:
    """Global addressable token capacity under the EVEN-MODULO DCP owner rule.

    Returns ``(global_capacity, binding_rank, stranded_per_rank)``:

    * ``global_capacity`` -- ``dcp_size * min_r(per_rank_capacity_r)``. This
      mirrors what ``ModelRunner._apply_token_constraints`` produces via its
      MIN all-reduce plus the ``max_total_num_tokens * dcp_size`` logical
      slot space.
    * ``binding_rank`` -- the LOWEST-indexed rank achieving that minimum, i.e.
      the rank whose budget actually caps the deployment. Raising any other
      rank's capacity (for example by giving it an fp8 KV pool) changes
      nothing until this rank is raised too.
    * ``stranded_per_rank`` -- per rank, how many of its own token slots the
      min-reduce throws away. A large stranded count on the workers is the
      signature of "worker-side precision cannot help here".

    Pure arithmetic over a replicated vector; no collectives.
    """
    if dcp_size <= 0:
        raise ValueError(f"dcp_size must be >= 1, got {dcp_size}")
    if not per_rank_capacity:
        raise ValueError("per_rank_capacity must not be empty")
    if len(per_rank_capacity) != dcp_size:
        raise ValueError(
            "per_rank_capacity must have one entry per DCP rank: got "
            f"{len(per_rank_capacity)} entries for dcp_size={dcp_size}"
        )
    caps = [int(c) for c in per_rank_capacity]
    unit = min(caps)
    binding_rank = caps.index(unit)
    stranded = [c - unit for c in caps]
    return unit * dcp_size, binding_rank, stranded


def host_tier_stride_mismatch(
    *,
    device_store_itemsize: int,
    device_head_num: int,
    device_head_dim: int,
    host_itemsize: int,
    host_token_stride_size: int,
) -> Optional[str]:
    """Grenz-check between a device KV pool and its pinned-host overflow tier.

    Every host<->device move on the weightless spill tier and on the
    kv-session-offload tier is a RAW BYTE copy driven by the host pool's
    ``token_stride_size``. Nothing in that path re-derives the element type,
    so a device pool and a host tier built from different dtypes would not
    fail -- they would silently reinterpret bytes at the wrong stride and
    return plausible garbage. With per-role precision in the picture (one
    role's pool is fp8/uint8-backed, another's is bf16) that stops being a
    theoretical seam.

    Returns ``None`` when the two agree, otherwise a message naming both
    sides, ready to raise.
    """
    expected_stride = device_head_num * device_head_dim * device_store_itemsize
    if host_itemsize != device_store_itemsize:
        return (
            "host KV tier element size does not match its device pool: host "
            f"itemsize={host_itemsize} B vs device store itemsize="
            f"{device_store_itemsize} B. The host<->device transfers are raw "
            "byte copies at the host tier's stride, so a mismatch would "
            "reinterpret KV bytes instead of failing."
        )
    if host_token_stride_size != expected_stride:
        return (
            "host KV tier token stride does not match its device pool: host "
            f"token_stride_size={host_token_stride_size} B vs expected "
            f"{expected_stride} B (head_num={device_head_num} x "
            f"head_dim={device_head_dim} x itemsize={device_store_itemsize}). "
            "The host<->device transfers are raw byte copies at the host "
            "tier's stride, so a mismatch would reinterpret KV bytes instead "
            "of failing."
        )
    return None
