# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The idle workbench: a queue of useful work for a rig nobody is using (#347).

ANALYSE #347 item 1. #341 made training a tenant of the rig -- it runs while
nothing is being served, holds a VRAM lease, checkpoints and releases when
demand arrives, and resumes at the next idle window. That machinery is not
specific to training. This package generalizes it into one interface, one
scheduler and one priority order, with training as tenant #1 of N.

See ``docs/dev/DESIGN_347_idle_workbench.md``. Nothing here imports torch.
"""

from sglang.srt.workbench.log import WorkLog, WorkLogEntry
from sglang.srt.workbench.scheduler import Workbench, WorkbenchConfig
from sglang.srt.workbench.service import WorkbenchService, build_service
from sglang.srt.workbench.tenant import (
    Feasibility,
    IdleWorkTenant,
    SegmentOutcome,
    SegmentStatus,
    SubprocessSegment,
    WorkEstimate,
    WorkEvent,
    WorkGrant,
    WorkSegment,
    price_segment,
)

__all__ = [
    "Feasibility",
    "IdleWorkTenant",
    "SegmentOutcome",
    "SegmentStatus",
    "SubprocessSegment",
    "WorkEstimate",
    "WorkEvent",
    "WorkGrant",
    "WorkLog",
    "WorkLogEntry",
    "WorkSegment",
    "Workbench",
    "WorkbenchConfig",
    "WorkbenchService",
    "build_service",
    "price_segment",
]
