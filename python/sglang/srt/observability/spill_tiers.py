# Copyright 2026 SGLang Team
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
"""Live occupancy of the spill / offload tiers, read off the bookkeeping that
already exists -- for the ``sglang:spill_tier_*_bytes`` gauges.

The rule this module is written to: **read the consumer's own ledger, never a
proxy.** Each tier below names the object it reads and the unit that object
speaks; where a consumer has no current-occupancy figure, this module returns
NOTHING for it rather than a derived guess, and the dashboard renders that tier
as absent (#218 provenance: measured / estimate / absent, no "probably" tier).

What is deliberately NOT read, and why -- each of these was checked in the
tree and found to have no live occupancy source, so publishing a number for
them would be inventing one:

* the expert-offload ``StreamingStagingLedger``
  (``layers/moe/expert_offload.py``) -- every field on it is CUMULATIVE
  (``record()`` only ever adds; there is no decrement anywhere) and frozen
  after weight load, so ``pinned_bytes`` is "bytes ever staged", not "bytes
  held". The pinned pool's true current size is the sum over the per-layer
  ``MoEExpertOffloadCache._pinned`` tensors, which is what
  :func:`expert_host_bytes` reads instead.
* ``ExpertOffloadRelease.host_bytes`` -- likewise cumulative-since-load.
* the #286 short-term offload register -- its ``CapacityLedger`` does track
  current bytes, but no production path ever constructs the
  ``RealMovementBackend`` that owns one; ``OffloadRegister`` falls back to
  ``CpuFakeMovementBackend``, which counts item ids and no bytes.
* the #407 memtier ``TierRegistry`` -- a capability/profile description with
  no consumer wired to it; ``TierCapacity.reserved`` is never populated from a
  live source.
* hibernate staging images -- written once at boot / on ``POST /hibernate``,
  not a varying occupancy; a size-on-disk read is a different question from a
  live tier gauge.

Cost: every read here is a plain Python attribute/len/arithmetic read on
objects the scheduler thread owns. No CUDA sync, no lock, no syscall. That
matters because the caller runs at prefill-batch cadence and every
``decode_log_interval``-th decode step.

Every source is wrapped: a metrics read must never be able to take the
scheduler down, so a raising source is dropped from the result (and the tier
renders absent) instead of propagating.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "TIER_EXPERT_HOST",
    "TIER_KV_SESSION_HOST",
    "TIER_HICACHE_FILE",
    "TIER_PARK_PREFIX",
    "expert_host_bytes",
    "kv_session_host_bytes",
    "park_bytes_by_tier",
    "hicache_file_bytes",
    "collect_spill_tiers",
]

#: Tier ids, shared verbatim with the dashboard's ``tier_occupancy.py`` so a
#: gauge label and a rendered row cannot drift apart.
TIER_EXPERT_HOST = "expert_host_ram"
TIER_KV_SESSION_HOST = "kv_session_host_ram"
TIER_HICACHE_FILE = "hicache_file_disk"
#: Park destinations (#224) are one tier each: ``park:<backend-name>``.
TIER_PARK_PREFIX = "park:"

_WARNED: set = set()


def _warn_once(key: str, exc: BaseException) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning("spill-tier metric %r unavailable: %r", key, exc)


def expert_host_bytes(model_runner: Any) -> Optional[int]:
    """Bytes held by the expert-offload pinned HOST pool, right now.

    Summed over the per-layer ``MoEExpertOffloadCache._pinned`` tensors -- the
    actual pinned CPU tensors backing the pool. ``numel() * element_size()`` is
    a CPU-side attribute read; the dict is populated at install and only
    re-pointed by the hot-set rearrange, both on the forward thread.

    None when no layer has an offload cache installed, i.e. the tier does not
    exist in this configuration. That is a different statement from 0 bytes,
    and the caller keeps the difference.
    """
    model = getattr(model_runner, "model", None)
    if model is None or not hasattr(model, "modules"):
        return None
    total = 0
    found = False
    for module in model.modules():
        cache = getattr(module, "_expert_offload", None)
        if cache is None:
            continue
        pinned = getattr(cache, "_pinned", None)
        if not pinned:
            continue
        found = True
        for tensor in pinned.values():
            try:
                total += int(tensor.numel()) * int(tensor.element_size())
            except Exception:  # a non-tensor slot is not a byte figure
                continue
    return total if found else None


def kv_session_host_bytes(manager: Any) -> Optional[Tuple[int, int]]:
    """``(used, total)`` host-RAM bytes of the KV session spill pool (kvso).

    The pool is partitioned into ``max_spills`` regions of ``region_tokens``
    rows; a spilled session reserves a whole region, so occupancy is quantized
    to region granularity -- that is the pool's own allocation unit, not a
    rounding this function invents. Bytes per token come from the host pool's
    ``get_size_per_token()``, the same figure the manager logs its own pool
    size with.

    None when session offload is off (no manager, or no host pool).
    """
    if manager is None:
        return None
    host_pool = getattr(manager, "host_pool", None)
    if host_pool is None:
        return None
    per_token = int(host_pool.get_size_per_token())
    region_tokens = int(getattr(manager, "region_tokens", 0))
    max_spills = int(getattr(manager, "max_spills", 0))
    occupied = len(getattr(manager, "spills", ()) or ())
    region_bytes = region_tokens * per_token
    return occupied * region_bytes, max_spills * region_bytes


def park_bytes_by_tier(manager: Any) -> Dict[str, int]:
    """Bytes currently PARKED on each #224 destination tier.

    ``SpillDestinationController.parked`` is the live set of parked sessions;
    each carries ``rows`` (this rank's owned host-pool rows of the parked
    tail), so bytes = rows x the same per-token size the local pool uses. The
    controller's ``counters`` are deliberately not used here: ``park_bytes_out``
    is bytes ever moved, which is not occupancy.

    The mapping key is ``park:<tier-name>`` (``file`` = local persistent
    storage, ``mooncake`` = remote host RAM on the paired rig, ``dynamic`` =
    operator-supplied). Empty dict when no destinations are configured.
    """
    out: Dict[str, int] = {}
    if manager is None:
        return out
    ctl = getattr(manager, "_dest", None)
    if ctl is None:
        return out
    host_pool = getattr(manager, "host_pool", None)
    if host_pool is None:
        return out
    per_token = int(host_pool.get_size_per_token())
    tiers = list(getattr(ctl, "tiers", ()) or ())
    # Every configured tier is reported, including the ones holding nothing:
    # "configured and empty" and "not configured" are different states and the
    # dashboard draws them differently.
    for tier in tiers:
        out[TIER_PARK_PREFIX + str(getattr(tier, "name", "?"))] = 0
    for parked in list((getattr(ctl, "parked", None) or {}).values()):
        idx = int(getattr(parked, "tier_index", 0))
        name = (str(getattr(tiers[idx], "name", "?"))
                if 0 <= idx < len(tiers) else "?")
        key = TIER_PARK_PREFIX + name
        out[key] = out.get(key, 0) + int(getattr(parked, "rows", 0)) * per_token
    return out


def park_capacity_by_tier(manager: Any) -> Dict[str, int]:
    """Measured CAPACITY of each #224 park tier, from its #407 entry (#659).

    Until the park tier became a registered entry there was no capacity to
    report: the dashboard drew occupancy against no denominator, so a park tier
    at 2 GB looked the same whether its budget was 4 GB or 400. The number here
    is the one measured at registration -- ``min(configured budget, free space
    - df headroom)`` -- and it is reported as a total, never as a promise: a
    shared volume can lose free space to something else entirely.

    A tier whose capacity could not be measured contributes NO key, so the
    dashboard keeps drawing "configured, denominator unknown" rather than
    inventing a zero. Empty dict when no destinations are configured.
    """
    out: Dict[str, int] = {}
    if manager is None:
        return out
    ctl = getattr(manager, "_dest", None)
    if ctl is None:
        return out
    tiers = list(getattr(ctl, "tiers", ()) or ())
    descriptors = list(getattr(ctl, "park_descriptors", ()) or ())
    if len(descriptors) != len(tiers):
        return out
    for tier, descriptor in zip(tiers, descriptors):
        total = descriptor.capacity.total
        if total.is_absent:
            continue
        out[TIER_PARK_PREFIX + str(getattr(tier, "name", "?"))] = int(
            total.require("total")
        )
    return out


def hicache_file_bytes(tree_cache: Any) -> Optional[int]:
    """On-disk bytes held by the HiCache file backend (the L3 rung).

    Read from the backend's LRU evictor, which is the only place this tree
    tracks the figure. IMPORTANT and the reason for the ``_eviction_enabled``
    check: without a configured max size / min free space the evictor returns
    early from its own ``__init__`` and ``_total_bytes`` stays 0 forever while
    files DO accumulate on disk. Publishing that 0 would be silently wrong, so
    this returns None in that configuration and the tier renders absent with
    its reason rather than as an empty disk.
    """
    if tree_cache is None:
        return None
    controller = getattr(tree_cache, "cache_controller", None)
    backend = getattr(controller, "storage_backend", None)
    evictor = getattr(backend, "_evictor", None)
    if evictor is None:
        return None
    if not getattr(evictor, "_eviction_enabled", False):
        return None
    total = getattr(evictor, "_total_bytes", None)
    return None if total is None else int(total)


def collect_spill_tiers(scheduler: Any) -> Tuple[Dict[str, int], Dict[str, int]]:
    """``(used_by_tier, total_by_tier)`` for every ACTIVE spill tier.

    A tier that is not configured contributes no key at all -- absence in the
    scrape is how the dashboard learns a tier is absent, which is why nothing
    here emits a placeholder zero for an inactive consumer.
    """
    used: Dict[str, int] = {}
    total: Dict[str, int] = {}

    try:
        runner = getattr(getattr(scheduler, "tp_worker", None), "model_runner", None)
        experts = expert_host_bytes(runner)
        if experts is not None:
            used[TIER_EXPERT_HOST] = experts
    except Exception as e:  # pragma: no cover - defensive, never break metrics
        _warn_once(TIER_EXPERT_HOST, e)

    kvso = getattr(scheduler, "kv_session_offload", None)
    try:
        pair = kv_session_host_bytes(kvso)
        if pair is not None:
            used[TIER_KV_SESSION_HOST], total[TIER_KV_SESSION_HOST] = pair
    except Exception as e:  # pragma: no cover - defensive
        _warn_once(TIER_KV_SESSION_HOST, e)

    try:
        used.update(park_bytes_by_tier(kvso))
    except Exception as e:  # pragma: no cover - defensive
        _warn_once(TIER_PARK_PREFIX, e)

    try:
        total.update(park_capacity_by_tier(kvso))
    except Exception as e:  # pragma: no cover - defensive
        _warn_once(TIER_PARK_PREFIX + "capacity", e)

    try:
        disk = hicache_file_bytes(getattr(scheduler, "tree_cache", None))
        if disk is not None:
            used[TIER_HICACHE_FILE] = disk
    except Exception as e:  # pragma: no cover - defensive
        _warn_once(TIER_HICACHE_FILE, e)

    return used, total
