# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================

"""Anti-hang guard for the weightless-KV fast lane (Variant C Stage 1).

THE #1 RISK of the weightless attention-worker split is a collective-sequence
divergence between the head rank (which runs the full model) and the weightless
KV worker ranks (which run only the per-attention-layer dispatch loop): if the
two sides ever issue a DIFFERENT number or order of DCP-group collectives, the
symmetric NCCL call pairs a collective on one rank with the wrong collective on
another -> a SILENT NCCL HANG (the #94 class).

This guard makes such a divergence FAIL LOUD instead of hanging:

  * Per-step handshake (``guard_dcp_step``): before every guarded DCP collective
    each rank exchanges, over the group's CPU (gloo) subgroup, a fixed-shape
    signature of (step-index, op-tag). All ranks must agree. This is FIXED SHAPE,
    so it can never itself hang on a shape mismatch, and it catches an ORDER or
    TYPE divergence at the earliest possible point (before the real collective's
    payload is even built), raising a clear error naming the diverging step.

  * End-of-forward reconciliation (``dcp_forward_guard``): a monitored barrier
    (with a bounded timeout) plus an all-gather of each rank's total step count.
    This catches a COUNT divergence (one side issued FEWER steps and left the
    forward early) as a timeout with the culprit rank named, instead of leaving
    the other side blocked forever on an unmatched collective.

The guard uses the CPU subgroup exclusively, so it is off the CUDA stream and
never interferes with the NCCL/attention path. It CANNOT run inside a captured
CUDA graph (a gloo collective is not capturable); the forward/graph integration
must disable it during capture and replay via ``set_guard_enabled(False)`` --
inside a captured region the collective sequence is fixed by construction, so
the eager capture pass (guard ON) is what validates the sequence once.

Everything here is a no-op unless the weightless-KV fast lane is active AND the
guard is enabled, so every other code path stays byte-identical.
"""

from __future__ import annotations

import datetime
import zlib
from typing import Optional

import torch

from sglang.srt.distributed.parallel_state import GroupCoordinator

# Per-forward monotonic step counter and a rolling running signature. Reset at
# the start of every guarded forward by ``dcp_forward_guard``.
_STEP: int = 0
_ENABLED: bool = True
# Timeout for the end-of-forward monitored barrier. Generous vs a decode step
# (~10s ms) yet finite, so a genuine divergence surfaces in seconds not never.
_BARRIER_TIMEOUT = datetime.timedelta(seconds=30)


def set_guard_enabled(enabled: bool) -> None:
    """Enable/disable the anti-hang guard for this process. The forward/graph
    integration disables it inside captured CUDA-graph regions (where a gloo
    collective is not capturable and the sequence is already fixed)."""
    global _ENABLED
    _ENABLED = enabled


def guard_enabled() -> bool:
    return _ENABLED


def _op_signature(op_tag: str, step: int) -> int:
    """Deterministic fixed-width signature of (step, op_tag). Uses crc32 so it
    is identical across ranks/machines and fits an int64."""
    return (step << 32) | zlib.crc32(op_tag.encode("utf-8"))


def reset_forward_guard() -> None:
    global _STEP
    _STEP = 0


def guard_dcp_step(op_tag: str, cp_group: GroupCoordinator) -> None:
    """Fixed-shape per-step handshake over the group's CPU subgroup. Raises
    (loud, on every rank) if any rank disagrees on (step-index, op-tag) -- an
    order/type divergence between the head rank and the weightless workers.

    No-op when the guard is disabled or the group is size 1 (nothing to
    diverge)."""
    global _STEP
    if not _ENABLED or cp_group.world_size <= 1:
        _STEP += 1
        return
    sig = _op_signature(op_tag, _STEP)
    local = torch.tensor([sig], dtype=torch.int64)
    gathered = [torch.empty_like(local) for _ in range(cp_group.world_size)]
    # Fixed shape [1] on every rank -> the collective itself cannot hang on a
    # shape mismatch; it can only surface a VALUE mismatch, which we check.
    torch.distributed.all_gather(gathered, local, group=cp_group.cpu_group)
    vals = [int(t.item()) for t in gathered]
    if any(v != sig for v in vals):
        rank = cp_group.rank_in_group
        raise RuntimeError(
            "weightless-KV anti-hang guard: DCP collective-sequence DIVERGENCE "
            f"at step {_STEP} on rank {rank} (op {op_tag!r}, sig {sig}). "
            f"Per-rank signatures: {vals}. The head rank and the weightless KV "
            "workers issued a different DCP collective order/type at this step "
            "-- this would otherwise be a silent NCCL hang. Fix the forward so "
            "every rank issues the identical DCP-collective sequence."
        )
    _STEP += 1


class dcp_forward_guard:
    """Context manager wrapping one model forward. Resets the per-forward step
    counter on entry; on exit reconciles the total step count across the group
    with a bounded monitored barrier so a COUNT divergence (one rank ran fewer
    DCP steps and left early) fails as a timeout naming the culprit, not an
    infinite hang.

    No-op when the guard is disabled or the group is size 1."""

    def __init__(self, cp_group: Optional[GroupCoordinator]):
        self.cp_group = cp_group

    def __enter__(self):
        reset_forward_guard()
        return self

    def __exit__(self, exc_type, exc, tb):
        # Do not run the reconciliation collective while unwinding an exception
        # (the other ranks may be raising too; adding a collective would mask
        # the real error or deadlock the teardown).
        if exc_type is not None:
            return False
        cp = self.cp_group
        if not _ENABLED or cp is None or cp.world_size <= 1:
            return False
        try:
            torch.distributed.monitored_barrier(
                group=cp.cpu_group, timeout=_BARRIER_TIMEOUT, wait_all_ranks=True
            )
        except Exception as e:  # noqa: BLE001 - re-raise as a clear guard error
            raise RuntimeError(
                "weightless-KV anti-hang guard: DCP end-of-forward barrier "
                f"TIMED OUT on rank {cp.rank_in_group}. A rank issued a "
                "different NUMBER of DCP collectives this forward (count "
                "divergence between the head rank and the weightless KV "
                f"workers). Underlying: {e}"
            ) from e
        # Reconcile the final step count too (cheap, catches any residual skew).
        local = torch.tensor([_STEP], dtype=torch.int64)
        gathered = [torch.empty_like(local) for _ in range(cp.world_size)]
        torch.distributed.all_gather(gathered, local, group=cp.cpu_group)
        counts = [int(t.item()) for t in gathered]
        if any(c != _STEP for c in counts):
            raise RuntimeError(
                "weightless-KV anti-hang guard: DCP step-count MISMATCH across "
                f"ranks {counts} (this rank {cp.rank_in_group} = {_STEP}). The "
                "head rank and the weightless KV workers issued a different "
                "number of DCP collectives this forward."
            )
        return False
