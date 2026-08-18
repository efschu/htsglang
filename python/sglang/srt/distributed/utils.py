# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/utils.py

# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/utils.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import contextvars
import dataclasses
import logging
import math
import os
import pickle
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Deque, Dict, FrozenSet, List, Optional, Sequence, Tuple

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

# CONTEXT-LOCAL OVERLAY over the two process globals above (#274 slice C).
#
# The globals stay the process's ONE installed plan (written once at start-up
# by set_tp_partition_ratios, inherited by every thread). A second group in
# the same process needs its own vector while it builds, loads or forwards --
# and once lanes run CONCURRENTLY, a swap of the globals would be read by the
# other lane's forward. The overlay is a context variable, so it is per-thread
# by construction: a lane worker thread installs its vector once, and the
# serving group's thread keeps reading the installed plan.
#
# Sentinel-based rather than None-based: None is a MEANINGFUL plan value
# ("even split"), so "no overlay" needs its own marker.
_NO_OVERLAY = object()
_TP_PARTITION_OVERLAY: contextvars.ContextVar = contextvars.ContextVar(
    "distributed.tp_partition_overlay", default=_NO_OVERLAY
)


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
    _TP_PARTITION_RATIOS, _TP_PARTITION_FAMILIES = _normalize_partition_plan(
        ratios, families
    )


def _normalize_partition_plan(
    ratios: Optional[Sequence[int]],
    families: Optional[dict] = None,
) -> tuple:
    """Validate and normalize a (base, families) shard plan; shared by the
    process-wide setter and the context-local scope."""
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
    return base, fams


def get_tp_partition_ratios(family: Optional[str] = None) -> Optional[list]:
    """The active weight vector: the family's own vector when one is
    installed under `family`, otherwise the base vector (or None).

    Reads the context-local overlay first (a lane's own plan, #274), then
    the process-installed plan."""
    overlay = _TP_PARTITION_OVERLAY.get()
    if overlay is not _NO_OVERLAY:
        base, fams = overlay
    else:
        base, fams = _TP_PARTITION_RATIOS, _TP_PARTITION_FAMILIES
    if family is not None:
        vec = fams.get(family)
        if vec is not None:
            return vec
    return base


@contextmanager
def scoped_tp_partition_ratios(
    ratios: Optional[Sequence[int]],
    families: Optional[dict] = None,
):
    """Install a shard plan for the duration of a block, then restore.

    The plan above is a process global with a bare setter, which is right for
    the one plan a scheduler process installs at startup. A SECOND group in the
    same process (the dual-group runtime, #121: a PD lane whose FAST group is
    nested in the serving group's split) has to build and load its complement
    shard under ITS OWN vector, and hand the process back unchanged afterwards.

    Doing that without a scope is not merely untidy, it is silently wrong: the
    only discriminator that decides whether a plan applies is
    `len(ratios) == tp_size` (see `tp_partition_sizes`, `tp_plan_active`). A
    2-rank group built while a 3-entry vector is installed does not raise -- it
    falls back to the EVEN split and loads the wrong units.

    Restores both the base vector and the family vectors, and is nesting-safe.

    Slice C (#274): the scope writes a CONTEXT-LOCAL overlay, not the process
    globals. Within one thread that is the same observable behavior as before
    (install, restore); across threads it is the difference between correct
    and silently wrong -- a lane loading its complement under a 2-entry vector
    must not make the serving group's concurrent forward read that vector.
    """
    token = _TP_PARTITION_OVERLAY.set(_normalize_partition_plan(ratios, families))
    try:
        yield
    finally:
        _TP_PARTITION_OVERLAY.reset(token)


# ---------------------------------------------------------------------------
# Uneven decode context parallel (uneven DCP) token-axis split.
#
# When --rank-tp-ratio is non-uniform AND dcp_size == tp_size, the KV cache of
# the full-attention layers is split along the TOKEN axis instead of the head
# axis: every rank stores the FULL (replicated) kv-heads but only the context
# tokens it owns. Rank r owns ratio[r] contiguous slots of every virtual block
# of sum(ratio) tokens (weighted prefix-range owner rule, generalizing the even
# modulo rule owner==pos%N). The scheduler pool is pinned in VIRTUAL blocks
# (min over ranks of local_tokens/ratio[r], times sum(ratio)), so the total
# max_total_num_tokens can far exceed any single rank's local capacity.
#
# The token vector is SEPARATE from the weight (head) vector: q/kv heads follow
# --rank-tp-ratio, but the token split follows this vector (derived from each
# rank's free KV budget so every card fills up). When all-equal, this collapses
# to the classic even DCP (modulo) fast path, keeping that path bit-identical.
# ---------------------------------------------------------------------------

_CP_TOKEN_RATIOS: Optional[list] = None


def set_cp_token_ratios(ratios: Optional[Sequence[int]]) -> None:
    """Install the uneven-DCP token-axis split vector for this process (or
    None to disable, restoring the even modulo path)."""
    global _CP_TOKEN_RATIOS
    _CP_TOKEN_RATIOS = list(ratios) if ratios else None


def get_cp_token_ratios() -> Optional[list]:
    """The installed uneven-DCP token-axis split vector (or None)."""
    return _CP_TOKEN_RATIOS


# ---------------------------------------------------------------------------
# Weightless-KV fast lane (Variant C Stage 1).
#
# The fast lane DECOUPLES the head/weight partition from the token/KV (DCP)
# partition entirely. One rank (the "head rank", the fast weight-bearing card,
# e.g. the 5090 at rank 0) holds ALL attention heads and runs Q/O-proj + FFN +
# GDN as pure TP=1 (collective-free). The other ranks are WEIGHTLESS: they hold
# ONLY a token-shard of the KV cache (via the existing _CP_TOKEN_RATIOS token
# vector) and compute attention over it, contributing ZERO heads.
#
# So the per-rank HEAD-count vector is [total, 0, 0, ...] with `total` on the
# head rank and 0 everywhere else. partition_units() cannot express this (it
# forces every rank >= 1 unit, the correct rule when every rank bears weights),
# so the weightless head plan is set DIRECTLY here, independently of
# --rank-tp-ratio. The token vector stays free to be big on the weightless
# cards. This separation is the whole point of the fast lane; see
# variantC_architecture / the Stage-1 design.
# ---------------------------------------------------------------------------

_WEIGHTLESS_KV_HEAD_RANK: Optional[int] = None


def set_weightless_kv_head_rank(head_rank: Optional[int]) -> None:
    """Enable the weightless-KV fast lane for this process by naming the rank
    that holds ALL attention heads / all weights (or None to disable, keeping
    every other path byte-identical). All other DCP ranks are weightless (0
    heads, KV-token-shard only)."""
    global _WEIGHTLESS_KV_HEAD_RANK
    _WEIGHTLESS_KV_HEAD_RANK = head_rank


def get_weightless_kv_head_rank() -> Optional[int]:
    """The rank holding all heads/weights under the weightless-KV fast lane, or
    None when the fast lane is not active."""
    return _WEIGHTLESS_KV_HEAD_RANK


def weightless_kv_active() -> bool:
    """True when the weightless-KV fast lane is installed for this process."""
    return _WEIGHTLESS_KV_HEAD_RANK is not None


def is_weightless_head_rank(rank: int) -> bool:
    """True when the weightless-KV fast lane is active AND `rank` is the head
    rank (holds the full weights, runs the model TP=1 + the attention dispatch).
    False on the default path (fast lane off)."""
    head_rank = _WEIGHTLESS_KV_HEAD_RANK
    return head_rank is not None and rank == head_rank


def weightless_worker_rank(rank: int) -> bool:
    """True when the weightless-KV fast lane is active AND `rank` is a WEIGHTLESS
    KV worker (holds ONLY a KV token-shard; runs the stripped attention-only
    forward; materializes NO layer weights). False on the default path."""
    head_rank = _WEIGHTLESS_KV_HEAD_RANK
    return head_rank is not None and rank != head_rank


def weightless_head_counts(total: int, world_size: int) -> list:
    """Per-rank head-count vector for the weightless-KV fast lane: `total` on
    the head rank, 0 on every weightless rank. E.g. total=24 heads, world=3,
    head_rank=0 -> [24, 0, 0]. The uneven-DCP collectives already tolerate a
    0-head shard (cp_all_gather_heads_uneven pads to max(counts); a 0-count
    rank contributes an empty slice, so the Q all-gather becomes a broadcast
    from the head rank and the O merge slices the merged output back to the
    head rank only)."""
    head_rank = _WEIGHTLESS_KV_HEAD_RANK
    assert head_rank is not None, "weightless_head_counts() with fast lane off"
    assert (
        0 <= head_rank < world_size
    ), f"weightless head_rank {head_rank} out of range for world {world_size}"
    return [total if r == head_rank else 0 for r in range(world_size)]


def uneven_dcp_active(dcp_size: Optional[int] = None) -> bool:
    """True when the uneven-DCP token-axis split is in force: a non-uniform
    token vector is installed. When `dcp_size` is given, the vector must also
    match it (guards against a stale vector on a differently-sized group).
    All-equal vectors are NOT uneven -- they use the even modulo fast path."""
    ratios = _CP_TOKEN_RATIOS
    if not ratios or len(set(ratios)) == 1:
        return False
    if dcp_size is not None and len(ratios) != dcp_size:
        return False
    return True


def cp_token_split_factor(dcp_size: int) -> int:
    """The number of token slots per virtual block along the DCP axis.

    Uneven DCP: sum(token ratios) -- the virtual block that the weighted
    prefix-range owner rule cycles over, and the factor the KV pool / page
    size is inflated by. Even DCP (uniform or no vector installed): dcp_size,
    reproducing the classic modulo layout exactly."""
    if uneven_dcp_active(dcp_size):
        return sum(_CP_TOKEN_RATIOS)
    return dcp_size


def uneven_dcp_kv_replicated(dcp_size: int) -> bool:
    """True when the uneven-TP + DCP KV-replication path is in force: DCP spans
    the whole TP group (dcp_size>1) AND a --rank-tp-ratio base plan is installed
    (so kv-heads are split UNEVENLY and cannot be head-sharded across the DCP
    group). Under this path every rank stores the FULL (replicated) kv-heads but
    only its owned token slots. Covers BOTH the even-modulo (no token vector) and
    the weighted (token vector installed) owner rules. False -> stock behavior
    (even head-sharded DCP, or no DCP), keeping those paths bit-identical."""
    return dcp_size > 1 and get_tp_partition_ratios() is not None


def plan_uneven_dcp_kv_replicated(flags, base_plan) -> bool:
    """PLAN-TIME mirror of :func:`uneven_dcp_kv_replicated` (#503).

    The runtime predicate one function above reads PROCESS state -- the
    installed ``--rank-tp-ratio`` (``get_tp_partition_ratios()``) and the
    resolved ``dcp_size``. A planner runs before any of that exists, so it has
    to answer the same question from the flags it is about to emit. This lives
    next to the runtime gate deliberately: it is the same statement twice, and
    two files apart is how the two spellings drift.

    It answers one question only -- **is the KV POOL replicated-heads +
    token-sharded**. It does NOT say whether the k/v PROJECTIONS are
    replicated; that is :func:`attn_kv_replicated` (``kv < tp``, strictly),
    and the two are independent. At 4 kv heads over 3 ranks with uneven DCP
    the pool is replicated and the projections are not: "the attention write
    gathers this rank's uneven projection shard up to
    ``get_total_num_kv_heads()``"
    (``model_executor/model_runner_kv_cache_mixin.py:2721``). Conflating them
    is what audit #500-B1 did, and pricing attention WEIGHTS on this predicate
    would model a layout the #105 ragged-kernel guard refuses at the first
    forward. Use it for the token axis and the core term; never for the
    weight split.

    Term for term against ``uneven_dcp_kv_replicated``:

    * ``get_tp_partition_ratios() is not None`` -- a NON-UNIFORM rank ratio
      plan is installed. An all-equal plan takes the even fast path, so
      ``len(set(base_plan)) > 1`` is the plan-time spelling.
    * ``dcp_size > 1`` -- DCP spans the TP group. At plan time that is either
      an explicit ``dcp_size``, or an explicit KV token vector (which is what
      ``--rank-kv-ratio <vector>`` installs, and a non-``coupled``
      ``--rank-kv-ratio`` is exactly what auto-sets ``dcp_size = tp_size``:
      ``server_args.py:9845-9853``, ``uneven_kv_flag_active()`` at
      ``server_args.py:8433``).

    ``flags`` is anything carrying those two attributes -- ``ServerArgs``,
    ``PlacementFlags`` or ``PlanInputs``.
    """
    non_uniform = base_plan is not None and len(set(base_plan)) > 1
    if not non_uniform:
        return False
    if flags is None:
        return False
    if (getattr(flags, "dcp_size", None) or 0) > 1:
        return True
    if getattr(flags, "kv_token_vector", None):
        return True
    return False


def cp_token_prefix(dcp_size: int) -> list:
    """Prefix sums of the token ratios: cp_token_prefix()[r] is the first slot
    index (within a virtual block of cp_token_split_factor tokens) owned by
    rank r; entry [dcp_size] == the block size. Even DCP -> [0,1,2,...,N]."""
    if uneven_dcp_active(dcp_size):
        ratios = _CP_TOKEN_RATIOS
    else:
        ratios = [1] * dcp_size
    out = [0]
    for r in ratios:
        out.append(out[-1] + r)
    return out


def uneven_dcp_owner_bounds() -> Optional[tuple]:
    """(S, lo, hi) of this rank's DCP owner range under the uneven-TP
    KV-replication path, or None when that path is not in force.

    Under the owner rule a GLOBAL allocator slot L is owned by this rank iff
    (L % S) in [lo, hi); its physical slot in this rank's COMPACT per-rank KV
    pool is (L // S) * (hi - lo) + (L % S - lo). Weighted DCP: S = sum(token
    ratios) with the prefix-range bounds; even-modulo (no token vector):
    S = dcp_size, [lo, hi) = [rank, rank+1) -- the classic L // dcp_size
    compaction. HiCache must use exactly this mapping for its device<->host
    KV transfers: the radix tree stores GLOBAL indices, while the device pool
    only holds this rank's compact owned slots (see FlashInferAttnBackend
    uneven_dcp / uneven_dcp_weighted). Indexing the compact pool with raw
    global indices captures rows that belong to OTHER (usually later) tokens
    -- time-dependent content, all-zero before those rows are first written
    (task #60 L3 zero-page corruption)."""
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    dcp_size = parallel.attn_dcp_size
    if not uneven_dcp_kv_replicated(dcp_size):
        return None
    prefix = cp_token_prefix(dcp_size)
    lo = prefix[parallel.attn_dcp_rank]
    hi = prefix[parallel.attn_dcp_rank + 1]
    return prefix[-1], lo, hi


def _shards_missing_against_index(model_path: str, present: set) -> tuple:
    """``(missing_shard_names, declared_total_bytes)`` from the safetensors index.

    ``([], 0)`` when there is no index to check against -- a single-file
    checkpoint has no manifest and therefore cannot be short of one.
    """
    import json

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return [], 0
    try:
        with open(index_path) as f:
            index = json.load(f)
    except Exception:
        return [], 0
    named = set((index.get("weight_map") or {}).values())
    if not named:
        return [], 0
    declared = int((index.get("metadata") or {}).get("total_size") or 0)
    return sorted(named - present), declared


def _checkpoint_size_mib(model_path: Optional[str]) -> int:
    """Total on-disk checkpoint size (MiB), 0 if unknown. Deterministic in
    every process so the derived token vector is identical everywhere.

    Summing whatever ``*.safetensors`` happen to be present is only the size
    of the checkpoint when those files ARE the checkpoint. A directory that is
    still downloading answers this question with a smaller model, confidently
    and without complaint, and the planner anchors every quantized family's
    bytes/param on the answer.

    Measured 2026-08-14: planning Qwen3.8-27B-INT8 with 4 of 18 shards on disk
    reported it as SMALLER than the Qwen3.6 checkpoint it replaces, when the
    complete checkpoint is 4.76 GiB LARGER. Nothing was wrong with any single
    step; the input was simply never checked.

    So the index -- the checkpoint's own statement of what it consists of --
    gates the measurement. Complete: measure, as before. Incomplete: say so
    loudly, and fall back to the declared total, or to 0 ("unknown", which
    callers already treat as "use the config-derived estimate") when the index
    declares no total. Never a partial sum presented as a whole.
    """
    import glob

    if not model_path:
        return 0
    if os.path.isfile(model_path):
        return os.path.getsize(model_path) // 2**20
    if not os.path.isdir(model_path):
        return 0
    shards = glob.glob(os.path.join(model_path, "*.safetensors"))
    total = sum(os.path.getsize(f) for f in shards)
    if total == 0:
        total = sum(
            os.path.getsize(f) for f in glob.glob(os.path.join(model_path, "*.gguf"))
        )
        return total // 2**20

    missing, declared = _shards_missing_against_index(
        model_path, {os.path.basename(f) for f in shards}
    )
    if missing:
        logger.warning(
            "Checkpoint at %s is INCOMPLETE: %d of the %d shards named by "
            "model.safetensors.index.json are absent (%s%s). The %d MiB "
            "actually on disk is NOT this model's size; using the index's "
            "declared total (%d MiB) instead%s. Any plan or token vector "
            "derived from a partial checkpoint describes a model that does "
            "not exist.",
            model_path,
            len(missing),
            len(missing) + len(shards),
            ", ".join(missing[:5]),
            ", ..." if len(missing) > 5 else "",
            total // 2**20,
            declared // 2**20,
            "" if declared else " (index declares none -> reporting 0/unknown)",
        )
        return declared // 2**20

    return total // 2**20


def resolve_cp_token_ratios(
    server_args, checkpoint_size_mib: Optional[int] = None
) -> Optional[list]:
    """Token-axis split vector for uneven DCP, derived from the server args.

    Returns None (use even modulo) when no uneven plan applies. Otherwise a
    small positive-integer vector (gcd-reduced) proportional to each rank's
    FREE KV budget, so heterogeneous cards each fill up:

        avail[r] = rank_gpu_memory_mib[r]
                   - checkpoint_size_mib * weight[r] / sum(weights)  (weight bytes)
                   - _CP_TOKEN_OVERHEAD_MIB                          (context/frag)

    integerized to _CP_TOKEN_UNITS units (>=1 per rank), gcd-reduced so the
    virtual scheduler block (page_size * sum(ratios)) stays as fine as
    possible. When no per-rank byte budgets are available, falls back to the
    gcd-reduced --rank-tp-ratio weights (a simple weights-based split).

    Precedence: SGLANG_UNEVEN_TOKEN_VECTOR (env) > --rank-kv-ratio a,b,c
    (explicit pin) > rank_kv_capacity_seed (the planner's predicted match)
    > budget estimate > weights fallback. --rank-kv-ratio capacity keeps the
    estimate here (phase 1) and installs the MEASURED optimal vector after
    the post-weight-load profiling instead (phase 2, see
    ModelRunnerKVCacheMixin._maybe_suggest_dcp_token_vector).

    Deterministic pure function of the args so every rank computes the same
    vector (the pool pinning and owner rule must agree across ranks)."""
    weights = getattr(server_args, "rank_tp_ratio", None)
    dcp_size = getattr(server_args, "dcp_size", 1)
    if not weights or dcp_size <= 1 or len(set(weights)) == 1:
        # HONESTY GUARD (measured): this bail-out runs BEFORE the env vector
        # is read, so SGLANG_UNEVEN_TOKEN_VECTOR without a non-uniform
        # --rank-tp-ratio plan was SILENTLY IGNORED -- the server booted
        # green, flashinfer's even-DCP no-op served plain TP output, and the
        # requested token ownership never existed. Every engagement point of
        # the token-sharded pool (uneven_dcp_kv_replicated, the dcp auto-set,
        # this resolver) keys on the base plan today, so decoupled token
        # ownership is NOT reachable through this door; say so instead of
        # ignoring the ask.
        #
        # REACHABILITY (#182): this branch is worth nothing unless a real boot
        # gets here. configure_scheduler_process used to call the resolver
        # only when a base plan was installed -- i.e. only when this guard
        # could not fire -- so the guard held on a direct call and never on a
        # server. The boot gate now keys on the token vector's own presence;
        # see the "GUARD REACHABILITY" branch there before narrowing it again.
        from sglang.srt.environ import envs as _envs

        _env_vec = _envs.SGLANG_UNEVEN_TOKEN_VECTOR.get()
        if _env_vec and dcp_size > 1:
            raise ValueError(
                f"SGLANG_UNEVEN_TOKEN_VECTOR={_env_vec!r} is set, but no "
                "non-uniform --rank-tp-ratio plan is installed. The uneven "
                "token-ownership machinery only engages under a base shard "
                "plan (uneven_dcp_kv_replicated keys on it); without one the "
                "vector would be silently ignored and the server would run "
                "plain even DCP while looking configured. Install a "
                "non-uniform --rank-tp-ratio, or unset the vector."
            )
        return None
    if len(weights) != dcp_size:
        return None

    # Self-calibration override: SGLANG_UNEVEN_TOKEN_VECTOR wins over the
    # budget-estimate derivation below. The KV-pool calibration emits it as a
    # restart hint (measured optimal from the actual per-rank profiled token
    # capacity); feeding it back here converges the pools to that optimum on
    # the next boot. Model-type-agnostic (dtype-independent measured capacity).
    from sglang.srt.environ import envs

    env_vec = envs.SGLANG_UNEVEN_TOKEN_VECTOR.get()
    if env_vec:
        parsed = [int(x) for x in env_vec.split(",") if x.strip() != ""]
        if len(parsed) != dcp_size or any(v <= 0 for v in parsed):
            raise ValueError(
                f"SGLANG_UNEVEN_TOKEN_VECTOR must be {dcp_size} positive "
                f"integers (one per DCP rank), got {env_vec!r}."
            )
        if len(set(parsed)) == 1:
            return None
        g = math.gcd(*parsed)
        return [v // g for v in parsed]

    # Explicit pin via --rank-kv-ratio a,b,c (task #88): the decoupled
    # KV-token ownership vector, below the env override (family-flag
    # convention) and above the derivations. Validated + gcd-reduced in
    # ServerArgs._handle_uneven_tp; an all-equal pin means uniform token
    # ownership = the even-modulo owner rule (return None).
    kv_flag = getattr(server_args, "rank_kv_ratio", None)
    if isinstance(kv_flag, list) and len(kv_flag) == dcp_size:
        if len(set(kv_flag)) == 1:
            return None
        g = math.gcd(*kv_flag)
        return [v // g for v in kv_flag]

    # Planner phase-1 SEED: the predicted per-rank capacity vector, parked by
    # apply_auto_performance. Below the explicit pin and the env override,
    # above the budget estimate -- which the two writers of this field both
    # have a reason to distrust. Under draft-solo placement (--rank-kv-ratio
    # capacity) it does not model the host's unsharded draft weights +
    # globally-sized draft KV pool; under the phase-optimal arms (#435) the
    # solved MLP vector has moved weight mass off the budget proportion the
    # estimate assumes, so the boot would size its pool for a vector the plan
    # never gated. Purely a starting vector where a phase-2 measured install
    # exists (_maybe_suggest_dcp_token_vector replaces it after profiling in
    # the derived modes); under 'coupled' it is the boot vector.
    seed = getattr(server_args, "rank_kv_capacity_seed", None)
    if isinstance(seed, list) and len(seed) == dcp_size and all(v > 0 for v in seed):
        if len(set(seed)) == 1:
            return None
        g = math.gcd(*seed)
        return [v // g for v in seed]

    if checkpoint_size_mib is None:
        checkpoint_size_mib = _checkpoint_size_mib(
            getattr(server_args, "model_path", None)
        )

    budgets = getattr(server_args, "rank_gpu_memory_mib", None)
    if (
        isinstance(budgets, list)
        and len(budgets) == len(weights)
        and checkpoint_size_mib > 0
    ):
        total_w = sum(weights)
        avail = [
            max(b - checkpoint_size_mib * w / total_w - _CP_TOKEN_OVERHEAD_MIB, 1.0)
            for b, w in zip(budgets, weights)
        ]
        vector = partition_units(_CP_TOKEN_UNITS, [max(int(a), 1) for a in avail])
        g = math.gcd(*vector)
        return [v // g for v in vector]

    g = math.gcd(*weights)
    return [w // g for w in weights]


#: Token-vector resolution granularity (units) and the assumed
#: weight-independent per-rank overhead (CUDA context, fragmentation,
#: attention scratch) subtracted before the free-memory split.
_CP_TOKEN_UNITS = 64
_CP_TOKEN_OVERHEAD_MIB = 1536


def _partition_units_raw(units: int, weights: Sequence[int]) -> list:
    """Largest-remainder split of `units` over ranks proportional to
    `weights` (the classic behavior; every rank gets >= 1 unit)."""
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


def _balanced_group_lengths(weights: Sequence[int], groups: int, max_len: int) -> list:
    """Partition the `len(weights)` ranks (in order) into `groups`
    contiguous non-empty segments, each of length in [1, max_len],
    minimizing imbalance of per-segment weight sums (each kv-group has
    the SAME q capacity, so balancing weight sums keeps every rank's
    q-share ~ its weight). Deterministic: ties break toward the flatter
    then the lexicographically smaller length vector."""
    n = len(weights)
    prefix = [0]
    for w in weights:
        prefix.append(prefix[-1] + w)
    best = [None]  # (key, lengths)

    def rec(start: int, g_left: int, lengths: list) -> None:
        remaining = n - start
        if g_left == 0:
            if remaining == 0:
                sums, idx = [], 0
                for L in lengths:
                    sums.append(prefix[idx + L] - prefix[idx])
                    idx += L
                key = (max(sums), tuple(sorted(sums, reverse=True)), tuple(lengths))
                if best[0] is None or key < best[0][0]:
                    best[0] = (key, list(lengths))
            return
        # Leave >= 1 rank per remaining group and <= max_len each.
        hi = min(max_len, remaining - (g_left - 1))
        for L in range(1, hi + 1):
            if remaining - L > (g_left - 1) * max_len:
                continue
            rec(start + L, g_left - 1, lengths + [L])

    rec(0, groups, [])
    if best[0] is None:
        raise ValueError(
            f"kv-aligned split infeasible: {n} ranks into {groups} groups of "
            f"<= {max_len} units each."
        )
    return best[0][1]


def _partition_units_kv_aligned(
    units: int, weights: Sequence[int], groups: int
) -> list:
    """kv-boundary-aware q-head split (task #116).

    Under REPLICATED-KV geometry (TP > num_kv_heads) the q heads split in
    `units` indivisible packets, and the global kv-head groups fall at unit
    positions that are multiples of `units // groups`. The memory-proportional
    auto planner (--rank-tp-ratio auto) can otherwise produce a raw split whose
    per-rank q packets STRADDLE such a boundary, which the #105 current-chunk
    ragged kernel cannot represent (it fails fast in
    _replicated_kv_ragged_reindex). This constrains the split so every rank's
    q packets map cleanly into a single kv-head group.

    Returns the raw largest-remainder split UNCHANGED whenever it is already
    boundary-aligned (so even splits, kv >= tp, and explicit kv-aligned ratios
    stay byte-identical); otherwise repairs it by assigning contiguous rank
    segments to whole kv-groups. When `groups` cannot tile `units` into equal
    whole-unit blocks (kv**2 does not divide q), alignment is impossible and
    the raw split is returned (the #105 guard then correctly rejects it)."""
    sizes = _partition_units_raw(units, weights)
    n = len(weights)
    # Alignment only meaningful when the units tile into `groups` equal
    # whole-unit blocks AND ranks outnumber groups (each rank fits in one
    # group). Otherwise fall back to the raw split.
    if groups < 2 or groups >= n or units % groups != 0:
        return sizes
    per = units // groups  # units per kv-group (== global GQA group / kv_total)
    boundaries = [per * k for k in range(1, groups)]
    seen, c = set(), 0
    for s in sizes[:-1]:
        c += s
        seen.add(c)
    if all(b in seen for b in boundaries):
        return sizes  # already aligned -> byte-identical to the raw split
    lengths = _balanced_group_lengths(weights, groups, per)
    out, idx = [], 0
    for L in lengths:
        out.extend(_partition_units_raw(per, weights[idx : idx + L]))
        idx += L
    assert sum(out) == units and all(s >= 1 for s in out)
    return out


def cp_token_context_budget(vector: Sequence[int], capacities: Sequence[int]) -> int:
    """Global max_total_num_tokens under the weighted owner rule.

    Rank r owns vector[r] of every sum(vector) context tokens, so the unit that
    every rank can fund is min_r(capacities[r] // vector[r]) and the global
    budget is that unit times sum(vector). Maximised when the vector is
    proportional to the capacities -- which is exactly what --rank-kv-ratio
    capacity installs."""
    n = len(vector)
    assert len(capacities) == n and all(v > 0 for v in vector)
    return min(capacities[r] // vector[r] for r in range(n)) * sum(vector)


def cp_token_speed_vector(
    capacities: Sequence[int],
    bandwidth_weights: Sequence[int],
    loose_ctx_percent: float,
    grain: int = 64,
    hard_cap: Optional[int] = None,
) -> tuple:
    """KV-token ownership vector for --rank-kv-ratio speed.

    THE TRADE-OFF. Two vectors are of interest and they pull in opposite
    directions on a heterogeneous rig:

      * proportional to CAPACITY  -> maximises max_total_num_tokens, and hands
        the biggest token share to whichever ranks have the most free VRAM
        after weights -- typically the WEAK cards.
      * proportional to BANDWIDTH -> minimises the deep-context part of the
        decode step, because under DCP each rank runs attention over the tokens
        it owns and at bs=1 the group waits on the slowest rank.

    This walks the straight line between the two share vectors and returns the
    most bandwidth-shifted point that still funds the allowed context, i.e. the
    largest t in [0, 1] with

        budget(partition(t*bw_share + (1-t)*cap_share)) >= floor

    where floor = best_effective_budget * (1 - loose_ctx_percent/100).

    `hard_cap` is the other ceiling on max_total_num_tokens (the hybrid
    mamba/SWA cap = max_running_requests x context_len, which on hybrid models
    routinely binds far below the KV-derived budget). It matters because the
    user-visible quantity is min(kv_budget, hard_cap): while the cap binds, a
    bandwidth shift costs literally nothing and should be taken in full even at
    the default loose_ctx_percent=0. Without this the mode would refuse every
    free gain on exactly the models where the gain is largest.

    Deterministic pure function of its arguments -- every rank derives the same
    vector from the same all-gathered capacities, which is the invariant the
    phase-2 install depends on.

    Returns (vector, budget, t) with the vector gcd-reduced."""
    n = len(capacities)
    assert len(bandwidth_weights) == n and n > 0
    assert all(c > 0 for c in capacities) and all(w > 0 for w in bandwidth_weights)

    def effective(vec):
        b = cp_token_context_budget(vec, capacities)
        return min(b, hard_cap) if hard_cap else b

    cap_vec = partition_units(grain, list(capacities))
    floor = int(effective(cap_vec) * (1.0 - float(loose_ctx_percent) / 100.0))

    cap_sum = float(sum(capacities))
    bw_sum = float(sum(bandwidth_weights))
    cap_share = [c / cap_sum for c in capacities]
    bw_share = [w / bw_sum for w in bandwidth_weights]

    best = None
    for i in range(grain, -1, -1):
        t = i / float(grain)
        blend = [t * bw_share[r] + (1.0 - t) * cap_share[r] for r in range(n)]
        # partition_units takes integer weights; scale the blend up so the
        # rounding granularity does not swallow small differences.
        vec = partition_units(grain, [max(1, int(round(x * 10**6))) for x in blend])
        if any(v <= 0 for v in vec):
            continue
        if effective(vec) >= floor:
            best = (vec, t)
            break
    if best is None:  # cap_vec itself always meets its own floor
        best = (cap_vec, 0.0)
    vec, t = best
    g = math.gcd(*vec)
    vec = [v // g for v in vec]
    return vec, cp_token_context_budget(vec, capacities), t


def partition_units(
    units: int, weights: Sequence[int], groups: Optional[int] = None
) -> list:
    """Split `units` indivisible units over ranks proportionally to
    `weights` (largest-remainder rounding, every rank gets >= 1 unit).

    Deterministic pure function of (units, weights, groups) so every
    process computes the identical partition. Ties in the fractional parts
    are broken toward the lower rank index.

    `groups` (task #116): when set (= num_kv_heads for the Q dimension under
    the REPLICATED-KV geometry), the split is constrained so no rank's q
    packets straddle a kv-head-group boundary. It is a NO-OP whenever the raw
    split is already aligned, so all non-q dimensions (which pass groups=None)
    and already-aligned q splits stay byte-identical."""
    if groups:
        return _partition_units_kv_aligned(units, weights, groups)
    return _partition_units_raw(units, weights)


def partition_sizes(
    total: int,
    weights: Sequence[int],
    units: Optional[int] = None,
    groups: Optional[int] = None,
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
        return [s * scale for s in partition_units(units, weights, groups)]
    if groups:
        raise ValueError(
            "partition_sizes: groups (kv-boundary alignment) requires a "
            "unit count (the q-head packet count); got units=None."
        )
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
    groups: Optional[int] = None,
) -> list:
    """Per-rank sizes of a sharded dimension under the process-global
    shard plan. Without an installed ratio vector (or when this layer runs
    with its own tp_size, e.g. disable_tp layers use tp_size=1), this is
    the classic even split via divide(). `family` selects a named family
    plan (e.g. "mlp") and falls back to the base vector when that family
    has no own vector installed. `groups` (task #116, Q dimension only)
    constrains the split to kv-head-group boundaries; None keeps the plain
    proportional split (byte-identical)."""
    ratios = get_tp_partition_ratios(family)
    if not ratios or len(ratios) != tp_size:
        ensure_divisibility(total, tp_size)
        return [total // tp_size] * tp_size
    return partition_sizes(total, ratios, units, groups)


def tp_partition_size(
    total: int,
    tp_size: int,
    rank: int,
    units: Optional[int] = None,
    family: Optional[str] = None,
    groups: Optional[int] = None,
) -> int:
    """This rank's size of a sharded dimension under the global plan."""
    return tp_partition_sizes(total, tp_size, units, family, groups)[rank]


def tp_partition_offset(
    total: int,
    tp_size: int,
    rank: int,
    units: Optional[int] = None,
    family: Optional[str] = None,
    groups: Optional[int] = None,
) -> int:
    """This rank's start offset (prefix sum) in a sharded dimension."""
    return sum(tp_partition_sizes(total, tp_size, units, family, groups)[:rank])


def tp_vocab_ratios(tp_size: int) -> Optional[list]:
    """The EXPLICIT vocab family vector (--rank-vocab-ratio /
    SGLANG_UNEVEN_VOCAB_VECTOR) when it is active for a group of `tp_size`
    ranks and non-uniform; None otherwise.

    Deliberately does NOT fall back to the base --rank-tp-ratio vector
    (unlike get_tp_partition_ratios(family)): the vocab dimension of
    VocabParallelEmbedding / ParallelLMHead keeps the classic EVEN split
    under a plain uneven-TP plan ("vocab always even" by design, M22);
    only the explicit vocab flag opts into the ratio-weighted vocab
    split (per-rank shard widths ~ memory bandwidth, so the lm_head
    matvec finishes simultaneously on heterogeneous cards). A uniform
    vector IS the even split and reports as inactive, keeping the
    classic path byte-identical.
    """
    vec = _TP_PARTITION_FAMILIES.get("vocab")
    if not vec or len(vec) != tp_size or len(set(vec)) == 1:
        return None
    return vec


def tp_plan_active(tp_size: int, family: Optional[str] = None) -> bool:
    """True when an uneven-TP ratio plan is installed AND applies to a
    layer/group of the given tp_size (disable_tp layers with tp_size=1 and
    groups of a different size keep the classic even split)."""
    ratios = get_tp_partition_ratios(family)
    return bool(ratios) and len(ratios) == tp_size


# Widest vector of the jit activation kernels (elementwise/activation.cuh):
# kMaxVecBytes = 32 bytes on Blackwell -> 16 bf16 elements (the 16-byte
# path on Ampere gives 8). Uneven shard plans align to the WIDEST vector so
# one plan is valid on every arch of a potentially mixed rig; which rank
# lands on which arch is not known at plan/construction time.
ACTIVATION_VEC_ELEMS = 16


def block_aligned_units(total: int, units: Optional[int], block: Optional[int]):
    """Coarsen an element-granular unit family to whole weight-quant blocks.

    The one rule, in one place: a block-quantized weight (FP8 with
    ``weight_block_size``, AWQ/GPTQ groups) can only be split where a whole
    quantization block ends, so a family whose unit is FINER than the block
    has to be re-expressed in units of ``lcm(unit_elems, block)``.  Families
    that are already block-multiples (head-granular ones) pass through.

    Both ``_quant_block_aligned_units`` (layer construction) and the lane's
    nesting probes call this, and they must: a nesting verdict computed on
    the raw unit count says nothing about a dimension the layers partition
    in block units.  Those two verdicts genuinely disagree -- for
    intermediate 17408 with ``weight_block_size [128,128]`` the raw count is
    1088 and the real one 136, and a swept ratio grid finds both directions
    of disagreement, including "raw says nested, blocks say not".
    """
    if units is None or not block:
        return units
    if total % block != 0:
        # Dimension is not block-quantizable at all -- the quant method's own
        # skip/validation logic owns this case.
        return units
    unit_elems = total // units
    if unit_elems % block == 0:
        return units
    lcm = math.lcm(unit_elems, block)
    if total % lcm != 0:
        raise ValueError(
            f"Cannot align uneven-TP units (unit={unit_elems} elems) of a "
            f"{total}-wide dimension to the weight quant block {block}."
        )
    return total // lcm


def assert_activation_aligned_shards(
    total: int,
    tp_size: int,
    units: Optional[int],
    family: Optional[str] = "mlp",
    what: str = "MLP intermediate",
) -> None:
    """Fail fast at PLAN time (module construction) when any rank's shard
    of an activation-fed dimension (silu_and_mul / gelu_and_mul input)
    would violate the jit activation kernel's vector alignment. The kernel
    itself only raises at the FIRST FORWARD ("hidden size must be divisible
    by vector size", activation.cuh:168) — long after weights loaded — so
    an incompatible geometry must be rejected at boot instead (task #82).

    Only active under an installed uneven plan; the classic even-split
    path keeps its existing (per-arch, runtime) behavior untouched.
    """
    if not tp_plan_active(tp_size, family):
        return
    sizes = tp_partition_sizes(total, tp_size, units, family)
    bad = [r for r, s in enumerate(sizes) if s % ACTIVATION_VEC_ELEMS]
    if bad:
        raise ValueError(
            f"Uneven-TP shard plan for the {what} dimension of size {total} "
            f"(tp_size={tp_size}, units={units}, family={family!r}) yields "
            f"per-rank shards {sizes}; rank(s) {bad} are not divisible by "
            f"the activation kernel's widest vector "
            f"({ACTIVATION_VEC_ELEMS} elements). Pick a shard plan / unit "
            f"granularity whose per-rank shards are multiples of "
            f"{ACTIVATION_VEC_ELEMS}."
        )


# ---------------------------------------------------------------------------
# TP > num_kv_heads (task #62): REPLICATED-KV attention geometry.
#
# Under an uneven-TP plan the attention heads are normally split in whole
# kv-head units (every rank >= 1 whole kv head + its GQA q group). When the
# model has FEWER kv heads than ranks (e.g. Qwen3.6-35B-A3B global layers:
# kv=2, TP=3; or Qwen3.6-27B kv=4 at TP=5) that scheme cannot hand each rank
# a kv head. The affected attention layers then switch to REPLICATED-KV mode:
#   - ALL kv heads live on EVERY rank (num_kv_local == total). K/V are
#     RECOMPUTED per rank from byte-identical replicated projection weights
#     and the identical post-allreduce hidden state — no broadcast, exactly
#     the semantics of upstream's even tp%kv==0 replication path.
#   - q heads keep splitting, but in units of kv_total heads (NOT whole GQA
#     groups): flashinfer requires num_qo % num_kv == 0 per rank, and with
#     num_kv_local == kv_total that means q_local must be a multiple of
#     kv_total. E.g. A3B 16q/2kv over TP=3 -> units of 2 -> [6,6,4]
#     (whole-8er-GQA-group units would force the degenerate [8,8,0]).
#   - the KV cache is NOT duplicated: the uneven-DCP token-axis sharding
#     (owner rule + graph-validated LSE merge) keeps per-rank token shards
#     disjoint, so total capacity stays the sum. The DCP decode/extend paths
#     already plan the paged wrappers with the FULL head counts, so this
#     mode is comm-neutral (the per-layer kv-head all-gather in
#     _dcp_masked_write even becomes a no-op and is skipped).
# Layers whose kv-head count DOES cover the ranks (e.g. GDN linear-attention
# heads, or kv>=tp full-attention layers) keep the normal head sharding —
# geometry is derived per layer from (q_heads, kv_heads, tp_size) with no
# model special-casing.
# ---------------------------------------------------------------------------


def attn_kv_replicated(tp_size: int, total_num_kv_heads: int) -> bool:
    """True when attention layers with `total_num_kv_heads` kv heads must run
    the REPLICATED-KV geometry under the installed uneven-TP plan: fewer kv
    heads than ranks, so the whole-kv-head-unit split cannot give every rank
    a kv head. False on the default path (no plan) and whenever kv >= tp
    (the normal uneven unit split handles kv % tp != 0 fine).

    kv == tp is deliberately EXCLUDED, and a `<` -> `<=` flip was tried and
    REVERTED on measurement -- do not repeat it. At kv == tp the kv-boundary
    alignment has groups == ranks, so the only non-straddling q split is the
    even one (`_partition_units_kv_aligned` returns the raw split at
    `groups >= n`, and the #105 uniform-GQA ragged kernel then rejects any
    straddling split at the FIRST FORWARD). Measured on Qwen3.5-2B
    (q=8/kv=2, TP=2, --rank-tp-ratio 11,21):

      * with `<=`: REPLICATED-KV engaged, q split [2, 6], KV cache
        duplicated per rank, boot green -- then
        "ValueError: REPLICATED-KV current-chunk attention (#105): ...
        q heads (offset 2, 6 heads over 2 local kv slots) straddle a global
        kv-head boundary" on the first request. Strictly worse.
      * with `<` (this code): normal mode, even [4, 4] attention split, the
        non-uniform plan applied to every OTHER dimension, output coherent
        and token-identical to TP=1, no KV duplication.

    Truly uneven attention at kv == tp needs a ragged kernel that supports
    per-rank non-uniform GQA mapping (the #169 head-gather family), not a
    threshold flip."""
    return tp_plan_active(tp_size) and total_num_kv_heads < tp_size


def attn_q_partition_units(
    total_num_q_heads: int, total_num_kv_heads: int, tp_size: int
) -> int:
    """Unit count for partitioning the attention q heads (and everything
    partitioned proportionally to them: qkv q-block, o_proj input) under an
    uneven-TP plan.

    Normal mode (kv >= tp): kv heads are the indivisible units — every rank
    gets whole GQA groups. REPLICATED-KV mode (kv < tp): every rank holds
    ALL kv heads, so q splits in units of kv_total heads (unit count =
    q_total // kv_total = the GQA group size), keeping per-rank
    num_qo % num_kv == 0 for the attention kernels.

    `total_num_q_heads` must be the REAL q-head count (not a fused
    q+gate slot count) — callers with fused projections pass the real count
    and scale sizes themselves (the unit COUNT is fusion-invariant)."""
    if not attn_kv_replicated(tp_size, total_num_kv_heads):
        # Normal (or default-path) geometry: kv heads as the units. No
        # divisibility demands here — the default even path never reaches
        # unit-based partitioning at all.
        return total_num_kv_heads
    if total_num_q_heads % total_num_kv_heads != 0:
        raise ValueError(
            f"REPLICATED-KV geometry: total_num_q_heads "
            f"({total_num_q_heads}) must be a multiple of "
            f"total_num_kv_heads ({total_num_kv_heads})."
        )
    units = total_num_q_heads // total_num_kv_heads
    if units < tp_size:
        raise ValueError(
            f"REPLICATED-KV geometry: cannot split {total_num_q_heads} q "
            f"heads over {tp_size} ranks in units of {total_num_kv_heads} "
            f"(= kv_total) heads: only {units} units (< {tp_size} ranks). "
            f"tp_size must not exceed the GQA group size "
            f"({total_num_q_heads}/{total_num_kv_heads}={units})."
        )
    return units


def attn_q_partition_groups(total_num_kv_heads: int, tp_size: int) -> Optional[int]:
    """kv-group count to pass as `groups` when partitioning the Q dimension
    (task #116). Returns `total_num_kv_heads` under the REPLICATED-KV
    geometry (kv < tp), so the q-head split is constrained to never straddle
    a global kv-head-group boundary (the #105 ragged kernel cannot represent
    a straddling split). Returns None otherwise — the default/even path and
    the normal kv >= tp uneven split, where whole-kv-head units already keep
    every rank inside whole GQA groups, so no alignment constraint applies
    and the split stays byte-identical.

    This is THE single source of the Q-dimension `groups` value; the QKV q
    block, the o_proj input, and the attention backends must all derive their
    `groups` from it so their shards agree (a mismatch would mis-shard
    o_proj). It is cross-checked at layer construction (q_shard_groups)."""
    if not attn_kv_replicated(tp_size, total_num_kv_heads):
        return None
    return total_num_kv_heads


def tp_loaded_shard_start(
    loaded_full: int,
    tp_size: Optional[int],
    rank: int,
    shard_size: int,
    units: Optional[int] = None,
    family: Optional[str] = None,
    groups: Optional[int] = None,
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
    sizes = partition_sizes(loaded_full, ratios, units, groups)
    if sizes[rank] != shard_size:
        raise ValueError(
            f"uneven-TP shard mismatch: expected size {sizes[rank]} for "
            f"rank {rank} of dimension {loaded_full} under weight vector "
            f"{list(ratios)} (units={units}, groups={groups}), but the "
            f"parameter shard has {shard_size}."
        )
    return sum(sizes[:rank])


def solve_unit_rebalance_multi(
    free_bytes: Sequence[float],
    bytes_per_token: Sequence[float],
    families: dict,
    min_units: int = 1,
) -> Tuple[dict, int]:
    """Maximin solver for the uneven-TP self-calibration over one or
    more weight FAMILIES (the dense-MLP "mlp" family and the
    expert-weight "moe" family): find the per-rank unit counts per
    family that maximize the MINIMUM token capacity.

    `families` maps a family name to (units, bytes_per_unit): rank r
    currently owns units[r] units of that family; every unit it sheds
    frees bytes_per_unit[r] weight bytes (measured empirically as the
    rank's family parameter bytes divided by its unit count), which
    become KV budget. Rank r's profiled KV byte budget is free_bytes[r]
    and one KV token costs bytes_per_token[r] (both rank-local —
    per-token bytes scale with the rank's kv-head share):

        capacity_r = (free[r] + sum_f shed_units_f[r] * bpu_f[r])
                     / bytes_per_token[r]

    Since the scheduler's single max_total_num_tokens is the MIN over
    ranks, the objective is maximin. Greedy over single units suffices:
    the minimum only ever rises when the pinned (capacity-poorest) rank
    sheds a unit, so iteratively move — in whichever family raises the
    minimum most — one unit from the pinned rank to the rank that stays
    capacity-richest after receiving it, as long as the minimum strictly
    increases. Every rank keeps >= min_units units per family
    (partition_units requires >= 1 per rank).

    Conservation law: moving weight bytes between ranks leaves
    sum(free) unchanged, so the achievable maximin is bounded by
    sum(free) / sum(bytes_per_token) — the pure-TP balance point.
    Additional families do NOT raise that ceiling; they supply the
    shiftable weight mass needed to actually reach it (on MoE models
    the dense-MLP family alone is usually too small).

    Returns (new_units_by_family, projected_min_tokens).
    """
    n = len(free_bytes)
    if n != len(bytes_per_token):
        raise ValueError(
            "solve_unit_rebalance: input vectors must have equal length, "
            f"got {len(free_bytes)}/{len(bytes_per_token)}."
        )
    if any(b <= 0 for b in bytes_per_token):
        raise ValueError(
            "solve_unit_rebalance: bytes_per_token must be positive; got "
            f"bytes_per_token={list(bytes_per_token)}."
        )
    fams: dict = {}
    for name, (units, bytes_per_unit) in families.items():
        if not (n == len(units) == len(bytes_per_unit)):
            raise ValueError(
                "solve_unit_rebalance: input vectors must have equal "
                f"length, got {len(units)}/{len(bytes_per_unit)} for "
                f"family {name!r} vs {n} ranks."
            )
        if any(u < min_units for u in units):
            raise ValueError(
                f"solve_unit_rebalance: every rank must own >= "
                f"{min_units} unit(s) of family {name!r}; got "
                f"units={list(units)}."
            )
        fams[name] = (list(units), list(bytes_per_unit))

    u = {name: list(units) for name, (units, _) in fams.items()}

    def capacity(r: int) -> float:
        freed = sum((fams[name][0][r] - u[name][r]) * fams[name][1][r] for name in fams)
        return (free_bytes[r] + freed) / bytes_per_token[r]

    while fams:
        caps = [capacity(r) for r in range(n)]
        cur_min = min(caps)
        donor = min(range(n), key=lambda r: (caps[r], r))
        # Try a one-unit move in every family; commit the move that
        # raises the minimum most.
        best = None  # (new_min, family, receiver)
        for name in fams:
            if u[name][donor] <= min_units:
                continue
            bpu = fams[name][1]
            # Receiver: the rank that remains capacity-richest after
            # taking on the unit's weight bytes (hurts maximin least).
            receiver = max(
                (r for r in range(n) if r != donor),
                key=lambda r: (
                    (
                        free_bytes[r]
                        + sum((fams[f][0][r] - u[f][r]) * fams[f][1][r] for f in fams)
                        - bpu[r]
                    )
                    / bytes_per_token[r],
                    -r,
                ),
            )
            u[name][donor] -= 1
            u[name][receiver] += 1
            new_min = min(capacity(r) for r in range(n))
            u[name][donor] += 1
            u[name][receiver] -= 1
            if new_min > cur_min and (best is None or new_min > best[0]):
                best = (new_min, name, receiver)
        if best is None:
            break
        _, name, receiver = best
        u[name][donor] -= 1
        u[name][receiver] += 1

    projected = int(min(capacity(r) for r in range(n))) if n else 0
    return u, projected


def solve_unit_rebalance(
    free_bytes: Sequence[float],
    bytes_per_token: Sequence[float],
    units: Sequence[int],
    bytes_per_unit: Sequence[float],
    min_units: int = 1,
) -> Tuple[list, int]:
    """Single-family convenience wrapper around
    solve_unit_rebalance_multi (see there for the capacity model).

    Returns (new_units, projected_min_tokens)."""
    new_units, projected = solve_unit_rebalance_multi(
        free_bytes,
        bytes_per_token,
        {"_": (units, bytes_per_unit)},
        min_units=min_units,
    )
    return new_units["_"], projected


def suggest_unit_rebalance_multi(
    free_bytes: Sequence[float],
    bytes_per_token: Sequence[float],
    families: dict,
    imbalance_threshold: float = 1.10,
) -> Optional[Tuple[dict, int, int]]:
    """Decide whether shifting family units between uneven-TP ranks
    would meaningfully raise the (MIN-synced) KV token capacity.

    `families` maps a family name to (units, family_bytes) per rank —
    the values gathered after rank-local KV profiling (bytes_per_unit is
    derived as family_bytes / units per rank). Families with degenerate
    inputs (zero units or zero bytes on any rank) are dropped rather
    than blocking the calibration of the remaining families.

    Returns None when the capacities are already balanced (max/min <=
    imbalance_threshold), when nothing can be calibrated, or when the
    solver finds no strictly better partition. Otherwise returns
    (new_units_by_family, current_min_tokens, projected_min_tokens);
    new_units_by_family only contains families whose vector CHANGED.
    """
    n = len(free_bytes)
    if n < 2 or n != len(bytes_per_token):
        return None
    if any(f < 0 for f in free_bytes) or any(b <= 0 for b in bytes_per_token):
        return None
    usable = {}
    for name, (units, family_bytes) in families.items():
        if len(units) != n or len(family_bytes) != n:
            continue
        if any(u <= 0 for u in units) or any(b <= 0 for b in family_bytes):
            continue
        usable[name] = (
            list(units),
            [family_bytes[r] / units[r] for r in range(n)],
        )
    if not usable:
        return None
    capacities = [free_bytes[r] / bytes_per_token[r] for r in range(n)]
    cur_min = min(capacities)
    if cur_min <= 0:
        return None
    if max(capacities) / cur_min <= imbalance_threshold:
        return None
    new_units, projected = solve_unit_rebalance_multi(
        free_bytes, bytes_per_token, usable
    )
    changed = {name: vec for name, vec in new_units.items() if vec != usable[name][0]}
    if not changed or projected <= int(cur_min):
        return None
    return changed, int(cur_min), projected


def suggest_unit_rebalance(
    free_bytes: Sequence[float],
    bytes_per_token: Sequence[float],
    units: Sequence[int],
    family_bytes: Sequence[float],
    imbalance_threshold: float = 1.10,
) -> Optional[Tuple[list, int, int]]:
    """Single-family convenience wrapper around
    suggest_unit_rebalance_multi (see there for semantics).

    Returns None or (new_units, current_min_tokens, projected)."""
    result = suggest_unit_rebalance_multi(
        free_bytes,
        bytes_per_token,
        {"_": (units, family_bytes)},
        imbalance_threshold=imbalance_threshold,
    )
    if result is None:
        return None
    changed, cur_min, projected = result
    return changed["_"], cur_min, projected


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


#: Per-stage explicit LAYER SETS, as an alternative to the contiguous
#: ``SGLANG_PP_LAYER_PARTITION`` counts. Stages are separated by ``;`` and each
#: stage is a comma list of ranges and singletons:
#:
#:     SGLANG_PP_LAYER_SET="0-2,4-6,8-10;3,7,11"
#:
#: Why this exists: a stage has always been an INTERVAL here
#: (``start = sum(partitions[:pp_rank])`` below), so a family placement that
#: puts, say, every linear-attention layer on one card and the interleaved
#: full-attention layers on others is not expressible at all. That is an
#: ADDRESSING limit, independent of any transport.
#:
#: The count form is untouched and remains the default: with this variable
#: unset, every code path below is byte-identical to what it was.
PP_LAYER_SET_ENV = "SGLANG_PP_LAYER_SET"

#: #753: the mid-loop crossing wire. A gapped layer set is only safe when the
#: forward loop exchanges activations at every ownership boundary; without the
#: wire a stage runs its own layers back to back and silently skips the peer's.
#: Set this only when the wire is actually carrying the crossings.
PP_CROSSING_WIRE_ENV = "SGLANG_PP_CROSSING_WIRE"


def pp_crossing_wire_enabled() -> bool:
    """True when the #753 mid-loop crossing wire is switched on."""
    return os.getenv(PP_CROSSING_WIRE_ENV, "") not in ("", "0", "false", "False")



class PPLayerSetError(ValueError):
    """A layer-set map that cannot be used. Always names the offending layers."""


def _parse_layer_spec(spec: str) -> List[int]:
    """One stage's ``0-2,4,7-9`` into a sorted list. Duplicates are kept so the
    caller can report them rather than silently absorbing them."""
    out: List[int] = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            lo_s, _, hi_s = piece.partition("-")
            try:
                lo, hi = int(lo_s), int(hi_s)
            except ValueError as err:
                raise PPLayerSetError(
                    f"{PP_LAYER_SET_ENV}: {piece!r} is not a range of integers"
                ) from err
            if hi < lo:
                raise PPLayerSetError(
                    f"{PP_LAYER_SET_ENV}: range {piece!r} runs backwards"
                )
            out.extend(range(lo, hi + 1))
        else:
            try:
                out.append(int(piece))
            except ValueError as err:
                raise PPLayerSetError(
                    f"{PP_LAYER_SET_ENV}: {piece!r} is not an integer layer id"
                ) from err
    return out


def parse_pp_layer_sets(
    raw: str, num_hidden_layers: int, pp_size: int, *, allow_gapped: bool = False
) -> List[FrozenSet[int]]:
    """Parse and VALIDATE the per-stage layer sets.

    The validation is the point. A partition that is merely "probably right"
    produces a model where some layer is computed twice or never, and both are
    silent: a duplicated layer just costs time, and a missing one is a
    placeholder pass-through that returns its input unchanged. So every failure
    below names the exact layers involved.

    CONTIGUITY (#753). A stage may own a NON-CONTIGUOUS set only when the
    caller passes ``allow_gapped=True``, which is how the mid-loop crossing
    wire declares it can carry one. The default refuses, and the default is
    the safe reading rather than the convenient one:
    ``qwen3_5.py:1466-1518`` exchanges ``pp_proxy_tensors`` ONCE per rank, at
    the stage boundary. A rank owning ``{2, 4}`` therefore runs layer 2
    straight into layer 4 with layer 3 -- computed on a peer -- never
    exchanged. Nothing raises. The model returns fluent, confidently wrong
    output, which is the failure shape this whole file exists to prevent.

    ``allow_gapped`` deliberately relaxes ONLY contiguity. Coverage, range and
    single-ownership hold either way: a gapped set is admissible with the wire,
    a set that loses or duplicates a layer never is.
    """
    stages = [seg for seg in raw.split(";")]
    if len(stages) != pp_size:
        raise PPLayerSetError(
            f"{PP_LAYER_SET_ENV}: {len(stages)} stage(s) given but pp_size is "
            f"{pp_size}. Separate stages with ';'."
        )
    parsed = [_parse_layer_spec(seg) for seg in stages]

    seen: Dict[int, int] = {}
    duplicated: Dict[int, List[int]] = {}
    for rank, layers in enumerate(parsed):
        for layer in layers:
            if layer in seen:
                duplicated.setdefault(layer, [seen[layer]]).append(rank)
            else:
                seen[layer] = rank
    if duplicated:
        detail = "; ".join(
            f"layer {layer} on stages {sorted(set(ranks))}"
            for layer, ranks in sorted(duplicated.items())
        )
        raise PPLayerSetError(
            f"{PP_LAYER_SET_ENV}: a layer may be owned by exactly one stage, "
            f"but {detail}. A duplicated layer is computed twice and nothing "
            f"downstream would say so."
        )

    out_of_range = sorted(l for l in seen if l < 0 or l >= num_hidden_layers)
    if out_of_range:
        raise PPLayerSetError(
            f"{PP_LAYER_SET_ENV}: layer(s) {out_of_range} are outside "
            f"[0, {num_hidden_layers})."
        )

    missing = sorted(set(range(num_hidden_layers)) - set(seen))
    if missing:
        raise PPLayerSetError(
            f"{PP_LAYER_SET_ENV}: layer(s) {missing} are owned by no stage. "
            f"An unowned layer is a pass-through placeholder at run time, so "
            f"the model would answer with that layer silently skipped."
        )

    if not allow_gapped:
        gapped = []
        for rank, layers in enumerate(parsed):
            ordered = sorted(layers)
            if not ordered:
                continue
            holes = sorted(set(range(ordered[0], ordered[-1] + 1)) - set(ordered))
            if holes:
                gapped.append((rank, ordered[0], ordered[-1], holes))
        if gapped:
            detail = "; ".join(
                f"stage {rank} spans {lo}-{hi} but does not own "
                f"{holes if len(holes) <= 8 else holes[:8] + ['...']}"
                for rank, lo, hi, holes in gapped
            )
            raise PPLayerSetError(
                f"{PP_LAYER_SET_ENV}: a stage's layers must be CONTIGUOUS "
                f"without the mid-loop crossing wire, but {detail}. #753: the "
                f"forward loop exchanges pp_proxy_tensors once per rank, at the "
                f"stage boundary, so the layers this stage does not own are "
                f"never received -- it would run its own layers back to back "
                f"and answer fluently with the peer layers silently skipped. "
                f"Refusing is the only safe reading until the wire lands; the "
                f"wire enables this by passing allow_gapped=True."
            )

    return [frozenset(layers) for layers in parsed]


def refuse_noncontiguous_layer_descriptor(local_slot_of, where: str) -> None:
    """Refuse to build a layer descriptor that assumes contiguous ownership.

    Disaggregated transfer describes a stage's layers as a contiguous
    ``(start, end)`` pair -- either ``start_layer``/``end_layer`` directly, or a
    start plus a layer COUNT -- and the receiving side slices its buffer lists
    with it. Under ``SGLANG_PP_LAYER_SET`` those bounds are ``min(owned)`` and
    ``max(owned) + 1``, i.e. the SPAN, which for a stage owning
    ``[35, 39, ..., 63]`` names 29 layers of which 21 are not owned.

    No index translation repairs this: the descriptor has no way to carry a
    set. A wrong descriptor mismatches KV buffers silently, so the unsupported
    combination is refused where it is built rather than diagnosed later.
    Carrying a layer set across the wire is an open design question -- see
    docs/dev/DESIGN_pp_layer_set.md.

    Passing ``None`` (contiguous ownership) is a no-op.
    """
    if local_slot_of is None:
        return None
    raise NotImplementedError(
        f"{where}: disaggregated KV transfer requires contiguous layer "
        "ownership. The transfer descriptor carries a contiguous layer range, "
        "which cannot express the non-contiguous set "
        f"{sorted(local_slot_of)} owned by this stage under "
        "SGLANG_PP_LAYER_SET. Use contiguous PP partitioning for "
        "disaggregated serving."
    )


def get_pp_layer_set(
    num_hidden_layers: int, pp_rank: int, pp_size: int
) -> Optional[FrozenSet[int]]:
    """This stage's owned layer ids, or ``None`` when the set form is unused.

    ``None`` is the default and means "ask ``get_pp_indices``" -- it is what
    keeps the contiguous path byte-identical.
    """
    raw = os.getenv(PP_LAYER_SET_ENV, None)
    if raw is None or not raw.strip():
        return None
    # #754, folded into #753 because it is the SAME resolution seam. The env is
    # process-wide, but this function is called again by the TP stack during a
    # phase flip -- with pp_size=1, where a 3-stage string is not merely
    # inapplicable but invalid, and parse_pp_layer_sets refused it by stage
    # count ("3 stage(s) given but pp_size is 1"). A single stage owns every
    # layer, so the set form has nothing to express: answering None hands the
    # caller back to get_pp_indices, which is the correct contiguous answer
    # rather than a suppressed error.
    if pp_size <= 1:
        return None
    # #753: a gapped set is admissible only when the crossing wire is on.
    # The refusal lives in parse_pp_layer_sets and names why.
    return parse_pp_layer_sets(
        raw, num_hidden_layers, pp_size, allow_gapped=pp_crossing_wire_enabled()
    )[pp_rank]


def current_stage_layer_set() -> Optional[FrozenSet[int]]:
    """This stage's owned layer ids, read from the live process group.

    ``None`` on the contiguous path, which is what lets every caller degenerate
    to the interval arithmetic it replaces. Kept HERE, next to
    ``get_pp_layer_set``, so ownership has exactly one derivation -- the same
    argument ``memory_pool._owned_layers_for_pool`` makes and which that
    function now defers to rather than restating.
    """
    raw = os.getenv(PP_LAYER_SET_ENV, None)
    if raw is None or not raw.strip():
        # Contiguous path: nothing to resolve, and no reason to touch the
        # process group at all.
        return None
    try:
        from sglang.srt.distributed import get_pp_group
    except Exception:  # pragma: no cover - import shape varies in unit tests
        return None
    try:
        group = get_pp_group()
        num_layers = getattr(group, "num_hidden_layers", None)
        if num_layers is None:
            # No caller stamps ``num_hidden_layers`` onto the group object, so
            # returning None here made the set form UNREACHABLE on metal and
            # every consumer silently degraded to the span test -- exactly the
            # 15-vs-0 PP0 arena defect this function exists to prevent
            # (measured 2026-08-18 17:30: PP0 reserved 22.6 GiB from a
            # cell_size=0 configurator). The layer count is recoverable from
            # the set string itself: parse_pp_layer_sets requires the union to
            # cover [0, N) with no gaps, so N is exactly max(layer) + 1.
            num_layers = _num_layers_from_layer_set_raw(raw)
            if num_layers is None:
                return None
        return get_pp_layer_set(num_layers, group.rank_in_group, group.world_size)
    except Exception:  # pragma: no cover - no process group in unit tests
        return None


def _num_layers_from_layer_set_raw(raw: str) -> Optional[int]:
    """``max(layer) + 1`` over every layer named in a raw layer-set string.

    Exact for every string ``parse_pp_layer_sets`` accepts (full cover of
    ``[0, N)`` is enforced there); None on anything unparseable, so the caller
    degrades the same way it would on a missing env.
    """
    highest = -1
    try:
        for stage in raw.split(";"):
            for tok in stage.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                if "-" in tok:
                    _, hi = tok.split("-", 1)
                    highest = max(highest, int(hi))
                else:
                    highest = max(highest, int(tok))
    except ValueError:
        return None
    return highest + 1 if highest >= 0 else None


def pp_gapped_ownership_active(pp_world_size: int) -> bool:
    """True when THIS run needs the mid-loop crossing wire to be correct.

    Three conditions, all required: a layer set is configured, the wire is
    switched on, and at least one stage's ownership is actually non-contiguous.
    A contiguous set expressed through the set mechanism (DESIGN §9.1 step 1)
    is deliberately NOT gapped -- it is carried by the ordinary stage-boundary
    transport, and reporting it as gapped would move it onto a protocol it does
    not need.

    Callers use this to choose a PROTOCOL, not merely to log, so it answers
    False on anything it cannot parse: an unreadable set already fails loudly
    in ``parse_pp_layer_sets`` at model build, and guessing "gapped" here would
    disable the stage-boundary handoff for a run whose ownership is unknown.
    """
    if pp_world_size is None or int(pp_world_size) <= 1:
        return False
    if not pp_crossing_wire_enabled():
        return False
    raw = os.getenv(PP_LAYER_SET_ENV, None)
    if raw is None or not raw.strip():
        return False
    num_layers = _num_layers_from_layer_set_raw(raw)
    if num_layers is None:
        return False
    try:
        owned = parse_pp_layer_sets(
            raw, num_layers, int(pp_world_size), allow_gapped=True
        )
    except Exception:  # noqa: BLE001 - see docstring: unparseable is not gapped
        return False
    for stage in owned:
        if not stage:
            continue
        lo, hi = min(stage), max(stage)
        if len(stage) != hi - lo + 1:
            return True
    return False


#: "not supplied", so that an explicit ``owned=None`` can mean the CONTIGUOUS
#: path rather than "look it up". A plain None default could not express both.
_ASK_THE_GROUP = object()


def stage_owned_layer_ids(
    all_attn_layer_ids, start_layer: int, end_layer: int, owned=_ASK_THE_GROUP
) -> List[int]:
    """The full-attention layers THIS stage owns. Set-aware.

    THE INTERVAL IS THE SPAN, NOT THE SET, and treating it as the set is the
    consumer error ``get_pp_indices``' own docstring warns about: under
    ``SGLANG_PP_LAYER_SET``, ``start_layer``/``end_layer`` are ``min(owned)``
    and ``max(owned) + 1``, so for the stage owning ``[35, 39, ..., 63]`` the
    interval names 29 layers of which 21 belong to someone else.

    MEASURED COST OF GETTING THIS WRONG, gapped boot v6, 2026-08-18 16:03:07Z.
    The set was

        PP0  0-2,4-6,...,60-62     48 GDN layers, span [0, 63)
        PP1  3,7,11,15,19,23,27,31  8 full-attention layers
        PP2  35,39,43,47,51,55,59,63

    so the 16 full-attention layers are 3, 7, ... 63 and PP0 owns NONE of them.
    The interval test ``0 <= i < 63`` matched FIFTEEN of them on PP0, and the
    KV VMM arena was reserved for all fifteen:

        PP0  30 buffers -> 22.56 GiB   (logged "reserved=22.6 GiB")
        PP1  16 buffers -> 12.03 GiB   (logged "reserved=12.0 GiB")
        PP2  16 buffers -> 12.03 GiB

    On a 32.6 GiB card already holding 27.1 GiB that is 49.70 GiB, and the
    driver refused the next ``cuMemCreate``. The sizing CHAIN was not at fault
    and neither was the token count -- all three stages sized from the same
    754019-token universe. The stage's own configurator had already computed
    ``cell_size=0`` for PP0, i.e. it agreed PP0 carries no full-attention KV;
    the pool disagreed by fifteen layers, and the pool is the one that
    allocates. This function is what makes them agree.

    A stage owning zero full-attention layers is therefore a legitimate,
    reachable configuration, not an error: it gets an empty list, no KV
    buffers, and an arena of one granularity page.

    Byte-identical on every contiguous layout -- ``current_stage_layer_set``
    returns ``None`` there and the interval test below is exact, because span
    and set coincide.

    ``owned`` is injectable so the resolution can be tested against the
    specimen's own layer set without a process group; left alone it is read
    from the live one.
    """
    if owned is _ASK_THE_GROUP:
        owned = current_stage_layer_set()
    if owned is not None:
        return [i for i in all_attn_layer_ids if i in owned]
    return [i for i in all_attn_layer_ids if start_layer <= i < end_layer]


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


def derive_pp_layer_split(
    scores: List[int],
    is_full_attention: Optional[List[bool]] = None,
    num_hidden_layers: Optional[int] = None,
    attn_scores: Optional[List[int]] = None,
) -> List[int]:
    """Derive per-stage layer counts from per-stage capability scores
    (#201 slice 3 item 2, the --pp-stage-ratio planner).

    ``scores`` are relative per-stage weights (analogous to
    --rank-tp-ratio, but across pipeline stages). The split is contiguous
    (stage boundaries only), like SGLANG_PP_LAYER_PARTITION itself.

    Hybrid awareness (the slice-2 finding, DESIGN_201 par. 13d): a hybrid
    linear+full-attention model splits its KV after FULL-ATTENTION layers,
    not after layers -- a planner reading num_hidden_layers alone mis-sizes
    every hybrid. When ``is_full_attention`` marks a genuine hybrid
    (0 < full < all), each boundary is first targeted proportionally in
    LAYER space (compute tracks all layers) and then snapped into the
    layer range that puts the score-proportional number of FULL-ATTENTION
    layers on each side (KV mass tracks the scores too). For homogeneous
    models the snap window is the whole axis and the split is the plain
    proportional rounding.

    ``attn_scores`` (#485) decouples the two families. Without it BOTH
    targets are derived from ``scores``, so on a period-P hybrid the layer
    target lands at ``P * target_full`` -- the bottom of the snap window --
    whenever the cumulative fraction sits near a multiple of ``1/n_full``.
    That single-number coupling, not the hardware, is why the reachable
    splits on the 64-layer period-4 reference checkpoint looked quantized to
    four layers (PROD_BRINGUP_BENCH.md sec. 1e). Passing a separate
    ``attn_scores`` targets the FULL-ATTENTION mass (KV bytes, attention
    bandwidth) while ``scores`` continues to target total layer mass
    (weights, dense compute), so linear/GDN layers can move across a stage
    boundary at zero KV cost. The two vectors are independent; the snap
    window still guarantees the attention split is exactly the one
    ``attn_scores`` asks for.

    Refusals (never a silent even split -- the #202 lesson):
      * fewer layers than stages;
      * a hybrid stage that would end with ZERO full-attention layers
        (its KV pool would be empty; give the stage a larger score or use
        fewer stages).
    """
    if is_full_attention is not None:
        n_layers = len(is_full_attention)
        if num_hidden_layers is not None and num_hidden_layers != n_layers:
            raise ValueError(
                f"derive_pp_layer_split: num_hidden_layers={num_hidden_layers} "
                f"disagrees with len(is_full_attention)={n_layers}."
            )
    elif num_hidden_layers is not None:
        n_layers = num_hidden_layers
        is_full_attention = [True] * n_layers
    else:
        raise ValueError(
            "derive_pp_layer_split needs is_full_attention or num_hidden_layers."
        )
    n_stages = len(scores)
    if n_stages < 1 or any((not isinstance(s, int)) or s < 1 for s in scores):
        raise ValueError(
            f"--pp-stage-ratio entries must be positive integers, got {scores}."
        )
    if n_layers < n_stages:
        raise ValueError(
            f"--pp-stage-ratio: {n_stages} stages cannot split "
            f"{n_layers} layers (every stage needs at least one)."
        )
    if attn_scores is not None:
        if len(attn_scores) != n_stages:
            raise ValueError(
                f"derive_pp_layer_split: attn_scores has {len(attn_scores)} "
                f"entries but scores has {n_stages}."
            )
        if any((not isinstance(s, int)) or s < 1 for s in attn_scores):
            raise ValueError(
                f"--pp-attn-stage-ratio entries must be positive integers, "
                f"got {attn_scores}."
            )
    total_score = sum(scores)
    total_attn_score = sum(attn_scores) if attn_scores is not None else total_score
    full_positions = [i for i, f in enumerate(is_full_attention) if f]
    n_full = len(full_positions)
    hybrid = 0 < n_full < n_layers

    bounds: List[int] = []
    prev = 0
    cum_score = 0
    cum_attn_score = 0
    for i in range(n_stages - 1):
        cum_score += scores[i]
        cum_attn_score += attn_scores[i] if attn_scores is not None else scores[i]
        target_layers = round(n_layers * cum_score / total_score)
        if hybrid:
            # #485: the attention target rides its OWN vector when one is
            # given, so KV mass and layer mass are independent.
            target_full = round(n_full * cum_attn_score / total_attn_score)
            target_full = min(max(target_full, 0), n_full)
            # All boundaries b with exactly target_full full-attention
            # layers in [0, b): the window between the target_full-th and
            # the following full-attention position.
            lo = full_positions[target_full - 1] + 1 if target_full >= 1 else 0
            hi = full_positions[target_full] if target_full < n_full else n_layers
            boundary = min(max(target_layers, lo), hi)
        else:
            boundary = target_layers
        # Contiguity floor/ceiling: at least one layer per stage on both
        # sides of every boundary.
        boundary = min(max(boundary, prev + 1), n_layers - (n_stages - 1 - i))
        bounds.append(boundary)
        prev = boundary
    bounds.append(n_layers)

    counts = [bounds[0]] + [bounds[i] - bounds[i - 1] for i in range(1, n_stages)]
    if hybrid:
        per_stage_full = []
        start = 0
        for count in counts:
            per_stage_full.append(
                sum(1 for p in full_positions if start <= p < start + count)
            )
            start += count
        if any(f == 0 for f in per_stage_full):
            zero_stage = per_stage_full.index(0)
            raise ValueError(
                f"--pp-stage-ratio {scores}: the derived split {counts} gives "
                f"stage {zero_stage} zero of the model's {n_full} "
                f"full-attention layers -- its KV pool would be empty. A "
                f"hybrid model splits its KV after FULL-ATTENTION layers "
                f"(#201 slice 2 finding); give stage {zero_stage} a larger "
                f"score, use fewer stages, or pass --pp-layer-ratio "
                f"explicitly."
            )
    return counts


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
            key = f"broadcast_from/{src}/{self.broadcast_send_counter}"
            self.store.set(key, pickle.dumps(obj))
            self.broadcast_send_counter += 1
            self.entries.append((key, time.perf_counter()))
            return obj
        else:
            key = f"broadcast_from/{src}/{self.broadcast_recv_src_counter[src]}"
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
