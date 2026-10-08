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

from flliper.srt.environ import envs
from flliper.srt.layers.dcp.lockstep import (
    PrefixLensRankDivergence,
    format_prefix_lens_divergence,
    prefix_lens_ballot,
    prefix_lens_ballot_agrees,
)

logger = logging.getLogger(__name__)

__all__ = [
    "assert_prefix_lens_rank_uniform",
    "has_deferred",
    "prefix_lens_check_enabled",
    "resolve_deferred",
]

#: Read ONCE, at import. A per-call ``os.environ`` read would let an operator
#: turn the check off in one process and leave the others entering a collective
#: nobody joins -- which is the exact failure class this file exists to report.
_ENABLED = os.environ.get("FLLIPER_DCP_PREFIX_LENS_CHECK", "1") != "0"


def prefix_lens_check_enabled() -> bool:
    return _ENABLED


def _dcp_cpu_group():
    """The DCP group's gloo communicator, or None when there is nothing to
    compare against (no DCP, single rank, or bring-up before the group
    exists)."""
    try:
        from flliper.srt.distributed.parallel_state import get_dcp_group_no_assert
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


def assert_prefix_lens_rank_uniform(
    prefix_lens: Optional[Sequence[int]], defer: bool = False
) -> None:
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
    # an older deferred ballot is decided first: verdicts in issue order
    resolve_deferred()

    ballot = torch.tensor(prefix_lens_ballot(prefix_lens), dtype=torch.int64)
    if defer and _DEFER_ENABLED:
        work = torch.distributed.all_reduce(
            ballot, op=torch.distributed.ReduceOp.MIN, group=cpu_group, async_op=True
        )
        _PENDING.append((work, ballot, list(prefix_lens), cpu_group))
        return
    torch.distributed.all_reduce(
        ballot, op=torch.distributed.ReduceOp.MIN, group=cpu_group
    )
    _decide(ballot, prefix_lens, cpu_group)


#: nf-pd-post 01.10. (boot y6o): ballots issued but not yet decided, oldest
#: first -- ``(work, ballot, prefix_lens, cpu_group)``.
_PENDING: list = []

#: Read ONCE, at import, like ``_ENABLED``: whether a skip-extend batch's
#: ballot may be decided after its (forward-less) pass instead of inside
#: ``prepare_for_extend``. Every rank defers the same batches (the skip flag
#: is the group's vote), so the collective itself is issued at the same point
#: on every rank either way -- only the WAIT moves.
_DEFER_ENABLED = bool(envs.FLLIPER_DCP_PREFIX_LENS_DEFER_SKIP.get())


def has_deferred() -> bool:
    return bool(_PENDING)


def resolve_deferred() -> None:
    """Decide every deferred ballot (oldest first): wait for its reduce and
    refuse loudly on divergence, exactly as the immediate check would have.

    nf-pd-post 01.10. (boot y6o): a skip-extend batch (H24 E2) runs NO target
    forward, so its pass enters no collective whose shape the vector decides;
    waiting for the reduce inside ``prepare_for_extend`` only made TP0 wait for
    the slowest worker's load-back issue (TP0 ``prepare_ms`` = TP1
    ``START-LOADING kv_issue_ms`` - TP0's, flip for flip: 240 = 288 - 64, 244
    = 280 - 47, 250 = 302 - 55, 295 = 358 - 67) before it could stream P's
    token. The verdict is now taken after that batch's result (the first
    token) and, at the latest, before the next forward that can enter a
    collective (``Scheduler.run_batch``) -- the detector's promise ("not after
    the forward has already entered a collective it cannot leave") holds."""
    while _PENDING:
        work, ballot, prefix_lens, cpu_group = _PENDING.pop(0)
        work.wait()
        _decide(ballot, prefix_lens, cpu_group)


def _decide(ballot, prefix_lens, cpu_group) -> None:
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
