"""Per-stage forward boundaries for the SGLANG_MOE_OFFLOAD_TIMING instruments.

The prefill instruments (MOE-OFFLOAD-TIMING-PREFILL in expert_offload.py,
ATTN-TIMING-PREFILL in hybrid_linear_attn_backend.py) collect CUDA events per
layer and flush one line per forward "when the next forward starts". They
recognised that moment as ``layer_id in (0, None)``. Under pipeline
parallelism only stage 0 ever runs layer 0: stage 1 of the Next-Flash P group
starts at layer 29, stage 2 at layer 40, so on those ranks the instrument
never flushed -- it logged nothing and its event lists grew without bound
(fn7t 19.09.: 100 MOE-OFFLOAD-TIMING-PREFILL lines, every one of them PP0;
the two 3080 stages had no decomposition at all).

:class:`StageHead` names the forward boundary per process instead: the first
layer id an instrument sees is this stage's first layer, and every later call
with that id opens a new forward. On stage 0 that is layer 0 again, so the
single-stage and PP0 lines are unchanged.
"""

from __future__ import annotations

import logging
from typing import Optional

import msgspec

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# The key of a caller that carries no layer id (desk stubs): its own value,
# so a None-only process still opens a forward on every call, as before.
_NO_LAYER = -1


class StageHead(msgspec.Struct):
    """The layer id that opens a forward on this pipeline stage.

    Learned from the first call, not configured: an instrument inside a layer
    does not know the stage's layer range, but it sees the layers in forward
    order, and the first one it ever sees is the stage's first.
    """

    head: Optional[int] = None

    def opens_forward(self, layer_id: Optional[int]) -> bool:
        key = _NO_LAYER if layer_id is None else int(layer_id)
        if self.head is None:
            self.head = key
        return key == self.head


def timing_on() -> bool:
    return bool(envs.SGLANG_MOE_OFFLOAD_TIMING.get())


def log_ple_gather(rows: int, zero_rows: int, seconds: float, workers: int) -> None:
    """One line per prefill-sized PLE pread gather while the timing switch is on.

    The CPU gather runs INSIDE the forward (the embedding waits on it), so its
    wall time is idle GPU time that the rank's ``gpu-ms`` contains and none of
    the CUDA-event instruments names. Only stage 0 embeds: this is the one
    stage-0-only term, next to the expert stream that every stage pays."""
    if not timing_on():
        return
    logger.info(
        "PLE-GATHER-PREFILL rows=%d zero_rows=%d ms=%.1f workers=%d "
        "(host wall of the pread gather; the forward waits on it)",
        rows,
        zero_rows,
        seconds * 1000.0,
        workers,
    )
