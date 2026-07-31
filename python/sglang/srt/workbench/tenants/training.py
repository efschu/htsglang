# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The #341 training tenant, registered on the workbench (DESIGN #347 W4).

Training is tenant #1 of N, at the top of the priority order: a job a user
submitted outranks work the rig invented for itself.

This is an adapter and nothing else. It runs no training, prices no job and
holds no lease. :class:`~sglang.srt.training.tenant.TrainingTenant` keeps
doing all three, with its own idle check, its own D2 feasibility gate and its
own per-job ledger reservation, because the amount of VRAM a training job
needs depends on the job and can only be known once the job exists. What
moves here is a single decision: **who starts and stops that loop**.

Two schedulers deciding when training runs is one too many. With the
workbench enabled, the training service is started surface-only and this
adapter owns ``TrainingTenant.start()`` / ``stop()``. The training tenant's
code is unchanged -- same loop, same checkpoint-and-release preemption, same
lease -- so the #341 test suite passes unmodified.

The cost of that arrangement is stated rather than hidden: because training
prices per job, this tenant reports ``self_leased=True`` and the workbench's
own pricing step does not run for it. The workbench arbitrates *ordering*
for training and *ordering plus memory* for everyone else.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any, Callable, Optional

from sglang.srt.workbench.tenant import (
    EventSink,
    IdleWorkTenant,
    SegmentOutcome,
    SegmentStatus,
    WorkEstimate,
    WorkEvent,
    WorkGrant,
    WorkSegment,
)

logger = logging.getLogger(__name__)


class TrainingWorkTenant(IdleWorkTenant):
    """Ordering-only adapter over the #341 training service."""

    name = "training"
    priority = 10

    def __init__(
        self,
        service: Any,
        *,
        idle_settle_s: float = 5.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        super().__init__()
        self.service = service
        #: How long the training loop may report "nothing running and nothing
        #: queued" before the segment is considered finished. The loop needs a
        #: tick or two to pick a job up, and ending the segment in that gap
        #: would hand the cards to a lower-priority tenant while a training
        #: job was starting.
        self.idle_settle_s = float(idle_settle_s)
        self._clock = clock

    # -- availability -------------------------------------------------------

    def available(self) -> tuple[bool, str]:
        if not getattr(self.service.config, "enabled", False):
            return False, (
                "the training tenant is switched off; start the server with "
                "--enable-training-tenant to accept and run fine-tuning jobs"
            )
        return True, ""

    def describe(self) -> str:
        return (
            "fine-tuning jobs submitted through /v1/fine_tuning/jobs, run by "
            "the #341 tenant with checkpoint-and-release preemption"
        )

    # -- the queue ----------------------------------------------------------

    def pending(self) -> int:
        return len(self.service.jobs.queued())

    def estimate(self) -> WorkEstimate:
        """Ordering only: the training gate prices each job for itself."""
        jobs = self.service.jobs.queued()
        per_card = int(getattr(jobs[0], "reserved_bytes_per_card", 0)) if jobs else 0
        return WorkEstimate(
            per_card_bytes=per_card,
            posts={"training_job": per_card} if per_card else {},
            self_leased=True,
        )

    # -- running ------------------------------------------------------------

    async def start_segment(self, grant: WorkGrant, sink: EventSink) -> WorkSegment:
        segment = _TrainingSegment(
            service=self.service,
            sink=sink,
            poll_seconds=max(0.05, float(self.service.config.poll_seconds)),
            idle_settle_s=self.idle_settle_s,
            clock=self._clock,
        )
        return await segment.start()

    def snapshot(self) -> dict[str, Any]:
        body = super().snapshot()
        body.update(
            {
                "jobs_total": len(self.service.jobs),
                "queued_jobs": [j.id for j in self.service.jobs.queued()],
                "running_job": self.service.tenant.running_job_id,
                "artifact_root": str(self.service.config.artifact_root),
            }
        )
        return body


class _TrainingSegment(WorkSegment):
    """One stretch of training: the #341 loop, started and stopped by us."""

    def __init__(
        self,
        *,
        service: Any,
        sink: EventSink,
        poll_seconds: float,
        idle_settle_s: float,
        clock: Callable[[], float],
    ) -> None:
        self.service = service
        self.sink = sink
        self.poll_seconds = poll_seconds
        self.idle_settle_s = idle_settle_s
        self._clock = clock
        self._stopping: Optional[SegmentStatus] = None
        self._stopped = asyncio.Event()

    async def start(self) -> _TrainingSegment:
        self.service.tenant.start()
        self.sink(
            WorkEvent(
                "info",
                f"training loop started for {len(self.service.jobs.queued())} "
                "queued job(s)",
            )
        )
        return self

    async def wait(self) -> SegmentOutcome:
        """Ends when the training queue drains, or when we are asked to stop."""
        quiet_since: Optional[float] = None
        while True:
            if self._stopping is not None:
                await self._stopped.wait()
                return self._outcome(self._stopping)
            running = self.service.tenant.running_job_id
            queued = self.service.jobs.queued()
            if running is None and not queued:
                now = self._clock()
                quiet_since = quiet_since if quiet_since is not None else now
                if now - quiet_since >= self.idle_settle_s:
                    await self._halt()
                    return self._outcome(SegmentStatus.SUCCEEDED)
            else:
                quiet_since = None
            await asyncio.sleep(self.poll_seconds)

    async def preempt(self, *, timeout_s: float = 60.0) -> SegmentOutcome:
        return await self._stop(SegmentStatus.PREEMPTED, timeout_s)

    async def cancel(self, *, timeout_s: float = 30.0) -> SegmentOutcome:
        # Cancelling the *segment* is not cancelling the *jobs*: they stay
        # queued and resume from their checkpoints. A job is cancelled through
        # the protocol, at /v1/fine_tuning/jobs/{id}/cancel.
        return await self._stop(SegmentStatus.CANCELLED, timeout_s)

    async def _stop(self, status: SegmentStatus, timeout_s: float) -> SegmentOutcome:
        self._stopping = status
        tenant = self.service.tenant
        # Close the relaunch window first: without this, the loop can pick the
        # next queued job up between "the trainer checkpointed" and "the loop
        # was stopped", and that job would be launched onto cards the
        # workbench is in the middle of handing back.
        tenant.hold_new_work = True
        if tenant.running_job_id is not None:
            # The #341 preempt path: the trainer reaches a checkpoint, writes
            # it, exits, and the job goes back to the queue as PREEMPTED. This
            # is the hook the training tenant already exposes.
            tenant.force_preempt.set()
            self.sink(
                WorkEvent(
                    "info",
                    f"asked training job {tenant.running_job_id} to checkpoint "
                    f"and release within {timeout_s:.0f}s",
                )
            )
            deadline = self._clock() + max(0.1, timeout_s)
            while tenant.running_job_id is not None and self._clock() < deadline:
                await asyncio.sleep(min(self.poll_seconds, 0.5))
            if tenant.running_job_id is not None:
                self.sink(
                    WorkEvent(
                        "warn",
                        "the trainer did not reach a checkpoint in time; "
                        "stopping the loop, which kills the executor",
                    )
                )
        with contextlib.suppress(Exception):
            await tenant.stop()
        # The hold belongs to this segment, not to the tenant: the next idle
        # window must find the loop willing to work again.
        tenant.hold_new_work = False
        self._stopped.set()
        return self._outcome(status)

    async def _halt(self) -> None:
        tenant = self.service.tenant
        tenant.hold_new_work = True
        with contextlib.suppress(Exception):
            await tenant.stop()
        tenant.hold_new_work = False
        self._stopped.set()

    def _outcome(self, status: SegmentStatus) -> SegmentOutcome:
        jobs = self.service.jobs
        queued = jobs.queued()
        detail = {
            SegmentStatus.SUCCEEDED: "the training queue is empty",
            SegmentStatus.PREEMPTED: (
                f"training released the rig; {len(queued)} job(s) will resume "
                "at the next idle window"
            ),
            SegmentStatus.CANCELLED: "the training segment was cancelled",
            SegmentStatus.FAILED: "the training segment failed",
        }[status]
        return SegmentOutcome(
            status=status,
            detail=detail,
            artifact_path=str(self.service.config.artifact_root),
            data={"queued_jobs": [j.id for j in queued]},
        )


__all__ = ["TrainingWorkTenant"]
