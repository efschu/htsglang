# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================

"""#639: check ONCE per extend batch that the prefix-length vector is the same
on every DCP rank, and refuse loudly when it is not.

The decision itself is pure and lives in ``lockstep.py`` next to the predicate
whose premise it verifies; this module is only the transport, kept separate so
``lockstep.py`` keeps its "no device, no process group" property and stays
hermetically pinnable.

COST, named exactly: ONE MIN ``all_reduce`` of a FOUR-element int64 CPU tensor
-- 32 bytes -- on the DCP group's gloo communicator, once per EXTEND batch.
Not once per layer: the sixteen full-attention layers of a forward all read
``forward_batch.extend_prefix_lens_cpu``, which is fixed by the time this runs,
so one ballot covers the whole forward. The alternative that was considered and
rejected -- making every rank enter the prefix branch unconditionally -- costs
two collectives per full-attention layer on every genuinely prefix-free extend,
i.e. 32 per first-chunk prefill on this checkpoint.

GATING is replicated by construction: the group's existence and its world size
come from ``--dcp-size``, and the kill switch is read ONCE at import into a
module constant so a mid-run environment edit cannot make one rank skip a
collective the others enter. A single-rank or non-DCP boot takes no collective
at all and is byte-identical.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import torch

from sglang.srt.layers.dcp.lockstep import (
    PrefixLensRankDivergence,
    format_prefix_lens_divergence,
    prefix_lens_ballot,
    prefix_lens_ballot_agrees,
)

logger = logging.getLogger(__name__)

__all__ = ["assert_prefix_lens_rank_uniform", "prefix_lens_check_enabled"]

#: Read ONCE, at import. A per-call ``os.environ`` read would let an operator
#: turn the check off in one process and leave the others entering a collective
#: nobody joins -- which is the exact failure class this file exists to report.
_ENABLED = os.environ.get("SGLANG_DCP_PREFIX_LENS_CHECK", "1") != "0"


def prefix_lens_check_enabled() -> bool:
    return _ENABLED


def _dcp_cpu_group():
    """The DCP group's gloo communicator, or None when there is nothing to
    compare against (no DCP, single rank, or bring-up before the group
    exists)."""
    try:
        from sglang.srt.distributed.parallel_state import get_dcp_group_no_assert
    except ImportError:  # pragma: no cover - import cycle safety
        return None

    group = get_dcp_group_no_assert()
    if group is None:
        return None
    cpu_group = getattr(group, "cpu_group", None)
    if cpu_group is None:
        return None
    try:
        if torch.distributed.get_world_size(cpu_group) <= 1:
            return None
    except Exception:  # noqa: BLE001 - an unusable group is "nothing to check"
        return None
    return cpu_group


def assert_prefix_lens_rank_uniform(prefix_lens: Optional[Sequence[int]]) -> None:
    """Refuse if this rank's extend prefix-length vector differs from a peer's.

    A DETECTOR, not a correction -- see the block comment above
    ``PrefixLensRankDivergence`` in ``lockstep.py``. Making the branch uniform
    does not make a divergent-prefix attention result right; it makes the
    failure uniform, immediate and self-describing instead of a 60-second BAR1
    stall whose diagnosis needs three live py-spy captures.

    Every rank reaches the same verdict from the same reduced ballot, so the
    raise is taken on all ranks or on none -- a detector that fired on one rank
    would itself be the rank-local-test-before-a-collective defect.
    """
    if not _ENABLED or prefix_lens is None:
        return
    cpu_group = _dcp_cpu_group()
    if cpu_group is None:
        return

    ballot = torch.tensor(prefix_lens_ballot(prefix_lens), dtype=torch.int64)
    torch.distributed.all_reduce(
        ballot, op=torch.distributed.ReduceOp.MIN, group=cpu_group
    )
    if prefix_lens_ballot_agrees(ballot.tolist()):
        return

    # Failure path only: pay for the vectors themselves. `all_gather_object`
    # tolerates the differing lengths that are part of what went wrong, and the
    # group is about to raise anyway, so the cost is irrelevant next to the
    # diagnosis it buys.
    world = torch.distributed.get_world_size(cpu_group)
    gathered: list = [None] * world
    try:
        torch.distributed.all_gather_object(
            gathered, list(prefix_lens), group=cpu_group
        )
    except Exception as exc:  # noqa: BLE001 - never lose the primary fault
        logger.error(
            "#639: prefix-length vectors diverged and the diagnostic gather "
            "failed too (%s: %s); raising with this rank's vector only.",
            type(exc).__name__,
            exc,
        )
        gathered = [list(prefix_lens)]

    raise PrefixLensRankDivergence(format_prefix_lens_divergence(gathered))
