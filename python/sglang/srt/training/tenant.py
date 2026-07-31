# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Training as an idle tenant of the rig (DESIGN #341 D4).

The contract is three sentences. Training runs only while no inference demand
exists. When demand arrives, training checkpoints, releases its VRAM lease and
parks. When the rig goes idle again, it resumes from that checkpoint.

Two mechanisms carry it, and they are deliberately separate:

**Idle detection is policy.** :class:`IdleMonitor` asks a set of sources when
the rig was last used and whether anything is serving now. A source that
cannot answer -- an unreachable registry, say -- contributes nothing rather
than vetoing. Absence of a control plane is not evidence of demand, and a
policy that refused to train without one would never train on the majority of
deployments.

**The VRAM lease is safety.** The ledger (#333 §3.3, #305-M1) is the thing
that actually prevents training from taking memory a serving engine holds. It
is an independent check with its own invariant and its own lock, and it does
not trust the monitor: if the monitor is wrong and the rig is busy, the
acquire is rejected with the holders named. Policy failing open is survivable
precisely because safety fails closed.

Preemption never appears in the protocol status. A preempted job is
``running`` with ``x-htsglang.tenant_state == "preempted"``, because that is
what it is: a job whose wall-clock exceeds its compute time.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from sglang.srt.training import activity
from sglang.srt.training.backends import (
    BackendEvent,
    BackendRun,
    BackendUnavailable,
    RunOutcome,
    RunSpec,
    RunStatus,
    TrainingBackend,
)
from sglang.srt.training.feasibility import MIB, MachineResources
from sglang.srt.training.store import JobStatus, JobStore, TenantState, TrainingJob

logger = logging.getLogger(__name__)

#: Registry engine class of a training tenant. Classes 1-3 are
#: autoregressive, diffusion and utility (#333); training is the fourth.
CLASS_TRAINING = 4

#: Engine classes whose activity counts as serving demand.
SERVING_CLASSES = (1, 2)


@dataclass(frozen=True)
class DemandSample:
    """One source's answer about serving demand."""

    source: str
    #: ``True`` when this source is certain something is being served now.
    busy: bool = False
    #: Wall-clock of the last observed activity. ``None`` means "no opinion",
    #: which is different from "was never used".
    last_activity_ts: Optional[float] = None
    detail: dict[str, Any] = field(default_factory=dict)


DemandSource = Callable[[], DemandSample]


@dataclass(frozen=True)
class IdleVerdict:
    """Whether the rig is idle, and the evidence either way."""

    idle: bool
    idle_for_s: Optional[float]
    grace_s: float
    samples: tuple[DemandSample, ...]

    def reason(self) -> str:
        busy = [s for s in self.samples if s.busy]
        if busy:
            return "serving demand reported by " + ", ".join(s.source for s in busy)
        if self.idle_for_s is None:
            return "no activity has ever been observed; the rig counts as idle"
        if self.idle:
            return (
                f"last serving activity {self.idle_for_s:.0f}s ago, past the "
                f"{self.grace_s:.0f}s grace window"
            )
        return (
            f"last serving activity {self.idle_for_s:.0f}s ago, inside the "
            f"{self.grace_s:.0f}s grace window"
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "idle": self.idle,
            "idle_for_s": (
                None if self.idle_for_s is None else round(self.idle_for_s, 1)
            ),
            "grace_s": self.grace_s,
            "reason": self.reason(),
            "sources": [
                {
                    "source": s.source,
                    "busy": s.busy,
                    "last_activity_ts": s.last_activity_ts,
                    **s.detail,
                }
                for s in self.samples
            ],
        }


def local_activity_source() -> DemandSample:
    """This process's own inbound generation requests."""
    stamp = activity.last_activity_ts()
    return DemandSample(
        source="local_requests",
        busy=False,
        last_activity_ts=stamp or None,
        detail={"requests_seen": activity.request_count()},
    )


def registry_activity_source(
    view_resolver: Optional[Callable[[], Any]] = None,
    *,
    serving_classes: Sequence[int] = SERVING_CLASSES,
) -> DemandSource:
    """Rig-wide activity, read from the registry snapshot.

    ``last_used_ts`` is stamped by the arbiter every time an engine is
    acquired for a request, so the maximum across serving-class engines is the
    rig's last inference. An unreachable registry yields no opinion.
    """

    def sample() -> DemandSample:
        resolver = view_resolver
        if resolver is None:
            from sglang.srt.entrypoints.openai.registry_view import (  # noqa: PLC0415
                fetch_registry_view,
            )

            resolver = fetch_registry_view
        try:
            view = resolver()
        except Exception as exc:  # noqa: BLE001 - a blind source is not a busy one
            return DemandSample(
                source="registry",
                last_activity_ts=None,
                detail={"reachable": False, "error": f"{type(exc).__name__}: {exc}"},
            )
        if not getattr(view, "reachable", False):
            return DemandSample(
                source="registry",
                last_activity_ts=None,
                detail={"reachable": False, "error": getattr(view, "error", None)},
            )
        stamps = [
            engine.last_used_ts
            for engine in getattr(view, "engines", ())
            if engine.klass in serving_classes and getattr(engine, "last_used_ts", 0.0)
        ]
        return DemandSample(
            source="registry",
            last_activity_ts=max(stamps) if stamps else None,
            detail={
                "reachable": True,
                "serving_engines": len(
                    [
                        e
                        for e in getattr(view, "engines", ())
                        if e.klass in serving_classes
                    ]
                ),
            },
        )

    return sample


class IdleMonitor:
    """Combines demand sources into a single idle verdict."""

    def __init__(
        self,
        sources: Sequence[DemandSource],
        *,
        grace_seconds: float = 120.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.sources = list(sources)
        self.grace_seconds = float(grace_seconds)
        self._clock = clock

    def sample(self) -> IdleVerdict:
        now = self._clock()
        samples: list[DemandSample] = []
        for source in self.sources:
            try:
                samples.append(source())
            except Exception as exc:  # noqa: BLE001 - one bad source is not a veto
                logger.debug("training: demand source failed: %s", exc)
        stamps = [s.last_activity_ts for s in samples if s.last_activity_ts]
        busy = any(s.busy for s in samples)
        idle_for = (now - max(stamps)) if stamps else None
        idle = not busy and (idle_for is None or idle_for >= self.grace_seconds)
        return IdleVerdict(
            idle=idle,
            idle_for_s=idle_for,
            grace_s=self.grace_seconds,
            samples=tuple(samples),
        )


@dataclass
class TenantConfig:
    """The tenant's knobs. Every one of them is a server flag."""

    enabled: bool = False
    artifact_root: Path = Path("/var/tmp/htsglang/training")
    #: Seconds of quiet before the rig counts as idle and training may start.
    grace_seconds: float = 120.0
    #: How often the loop re-checks demand while a job is running. This is the
    #: worst-case latency between a request arriving and preemption starting.
    poll_seconds: float = 2.0
    lease_seconds: float = 120.0
    #: Bounded teardown, same rule as #344: the serving tenant never waits
    #: indefinitely on a trainer that will not stop.
    preempt_timeout_s: float = 120.0
    cancel_timeout_s: float = 60.0
    #: Backoff after a rejected lease, so a full rig is not re-probed hotly.
    reject_backoff_s: float = 30.0
    default_backend: str = "auto"
    default_method: str = "lora"
    save_steps: int = 50

    def to_json(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "artifact_root": str(self.artifact_root),
            "grace_seconds": self.grace_seconds,
            "poll_seconds": self.poll_seconds,
            "lease_seconds": self.lease_seconds,
            "preempt_timeout_s": self.preempt_timeout_s,
            "cancel_timeout_s": self.cancel_timeout_s,
            "reject_backoff_s": self.reject_backoff_s,
            "default_backend": self.default_backend,
            "default_method": self.default_method,
            "save_steps": self.save_steps,
        }


@dataclass
class PlannedRun:
    """A job resolved to something the executor and the ledger can take."""

    backend: TrainingBackend
    spec: RunSpec
    card_uuids: tuple[str, ...]
    per_card_bytes: int


class TrainingTenant:
    """The scheduler loop: idle -> lease -> train -> preempt -> resume."""

    def __init__(
        self,
        jobs: JobStore,
        *,
        config: TenantConfig,
        monitor: IdleMonitor,
        planner: Callable[[TrainingJob], PlannedRun],
        reservation_store: Any = None,
        machine_resolver: Callable[[], MachineResources] = MachineResources,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.jobs = jobs
        self.config = config
        self.monitor = monitor
        self.planner = planner
        self.reservation_store = reservation_store
        self.machine_resolver = machine_resolver
        self._clock = clock
        self._task: Optional[asyncio.Task] = None
        self._stopping = asyncio.Event()
        self._wake = asyncio.Event()
        self._current: Optional[tuple[TrainingJob, BackendRun]] = None
        self._reservation: Any = None
        self.last_verdict: Optional[IdleVerdict] = None
        #: Set by tests and by the preempt demo to force a decision without
        #: waiting for the poll interval.
        self.force_preempt = asyncio.Event()

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.get_running_loop().create_task(self._loop())

    async def stop(self) -> None:
        self._stopping.set()
        self._wake.set()
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await self._release_lease()

    def wake(self) -> None:
        """Nudge the loop: a job was submitted or cancelled."""
        self._wake.set()

    def snapshot(self) -> dict[str, Any]:
        return {
            "config": self.config.to_json(),
            "running_job": self._current[0].id if self._current else None,
            "idle": self.last_verdict.to_json() if self.last_verdict else None,
            "queued_jobs": [j.id for j in self.jobs.queued()],
        }

    # -- the loop -----------------------------------------------------------

    async def _loop(self) -> None:
        logger.info("training tenant: started (%s)", self.config.to_json())
        while not self._stopping.is_set():
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a loop that dies stops all training
                logger.exception("training tenant: tick failed")
                await self._sleep(self.config.poll_seconds)

    async def _tick(self) -> None:
        pending = self.jobs.queued()
        cancelled_early = [j for j in pending if j.cancel_requested]
        for job in cancelled_early:
            self._finish_cancelled(job)
        pending = [j for j in pending if not j.cancel_requested]
        if not pending:
            await self._sleep(self.config.poll_seconds)
            return

        verdict = self.monitor.sample()
        self.last_verdict = verdict
        if not verdict.idle:
            for job in pending:
                self._note_waiting(job, verdict)
            await self._sleep(self.config.poll_seconds)
            return

        job = pending[0]
        try:
            planned = self.planner(job)
        except BackendUnavailable as exc:
            self._fail(
                job, exc.probe.reason, extension={"backend_probe": exc.probe.to_json()}
            )
            return
        except Exception as exc:  # noqa: BLE001 - a bad plan fails its own job
            self._fail(job, f"{type(exc).__name__}: {exc}")
            return

        if not await self._acquire_lease(job, planned):
            await self._sleep(self.config.reject_backoff_s)
            return

        try:
            await self._run(job, planned)
        finally:
            await self._release_lease()

    async def _sleep(self, seconds: float) -> None:
        self._wake.clear()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._wake.wait(), timeout=max(0.01, seconds))

    # -- the lease ----------------------------------------------------------

    async def _acquire_lease(self, job: TrainingJob, planned: PlannedRun) -> bool:
        if self.reservation_store is None or not planned.card_uuids:
            # No ledger configured: the tenant still runs, it simply has no
            # cross-process guard. Said out loud rather than assumed.
            self.jobs.append_event(
                job,
                "warn",
                "no VRAM ledger is configured; this job runs without a "
                "cross-process reservation and can collide with another tenant",
            )
            return True
        from sglang.srt.registry.ledger import (
            CardDemand,
            LedgerError,
            MultiCardReservation,
        )
        from sglang.srt.registry.ledger import (  # noqa: PLC0415
            TenantState as LedgerTenantState,
        )

        machine = self.machine_resolver()
        totals = {c.uuid: c.total_bytes for c in machine.cards}
        reservation = MultiCardReservation(
            self.reservation_store,
            tenant_id=f"training-{job.id}",
            klass=CLASS_TRAINING,
            totals=totals,
        )
        demands = [
            CardDemand(
                card_uuid=uuid,
                reserved_bytes=planned.per_card_bytes,
                posts={"training": planned.per_card_bytes},
            )
            for uuid in planned.card_uuids
        ]
        try:
            reservation.acquire(
                demands,
                state=LedgerTenantState.HOT,
                pid=os.getpid(),
                lease_seconds=self.config.lease_seconds,
            )
        except LedgerError as exc:
            self.jobs.append_event(
                job,
                "warn",
                f"VRAM lease rejected, staying queued: {exc}",
                data={"per_card_mib": planned.per_card_bytes // MIB},
            )
            return False
        self._reservation = reservation
        job.cards = planned.card_uuids
        job.reserved_bytes_per_card = planned.per_card_bytes
        self.jobs.append_event(
            job,
            "info",
            f"leased {planned.per_card_bytes // MIB} MiB on "
            f"{len(planned.card_uuids)} card(s): {', '.join(planned.card_uuids)}",
        )
        return True

    async def _release_lease(self) -> None:
        reservation, self._reservation = self._reservation, None
        if reservation is None:
            return
        with contextlib.suppress(Exception):
            reservation.release()

    # -- one attempt --------------------------------------------------------

    async def _run(self, job: TrainingJob, planned: PlannedRun) -> None:
        resuming = bool(planned.spec.resume_from)
        job.status = JobStatus.RUNNING
        job.tenant_state = TenantState.TRAINING
        job.backend = planned.backend.name
        job.output_dir = str(planned.spec.output_dir)
        self.jobs.append_event(
            job,
            "info",
            (
                f"resuming from {planned.spec.resume_from} on backend "
                f"{planned.backend.name}"
                if resuming
                else f"training started on backend {planned.backend.name} "
                f"({planned.spec.method.value})"
            ),
        )

        def sink(event: BackendEvent) -> None:
            self._absorb(job, event)

        try:
            run = await planned.backend.launch(planned.spec, sink)
        except BackendUnavailable as exc:
            self._fail(
                job, exc.probe.reason, extension={"backend_probe": exc.probe.to_json()}
            )
            return
        except Exception as exc:  # noqa: BLE001 - a launch failure fails the job
            self._fail(job, f"launch failed: {type(exc).__name__}: {exc}")
            return

        self._current = (job, run)
        try:
            outcome = await self._supervise(job, run)
        finally:
            self._current = None
        self._apply(job, outcome, planned)

    async def _supervise(self, job: TrainingJob, run: BackendRun) -> RunOutcome:
        waiter = asyncio.ensure_future(run.wait())
        try:
            while True:
                done, _ = await asyncio.wait(
                    {waiter}, timeout=max(0.01, self.config.poll_seconds)
                )
                if done:
                    return waiter.result()
                self._heartbeat()
                if job.cancel_requested:
                    self.jobs.append_event(job, "info", "cancel requested; stopping")
                    return await run.cancel(timeout_s=self.config.cancel_timeout_s)
                verdict = self.monitor.sample()
                self.last_verdict = verdict
                if not verdict.idle or self.force_preempt.is_set():
                    self.force_preempt.clear()
                    self.jobs.append_event(
                        job,
                        "info",
                        f"preempting: {verdict.reason()}",
                        data=verdict.to_json(),
                    )
                    return await run.preempt(timeout_s=self.config.preempt_timeout_s)
        finally:
            if not waiter.done():
                waiter.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await waiter

    def _heartbeat(self) -> None:
        if self._reservation is not None:
            with contextlib.suppress(Exception):
                self._reservation.heartbeat(lease_seconds=self.config.lease_seconds)

    # -- outcome ------------------------------------------------------------

    def _absorb(self, job: TrainingJob, event: BackendEvent) -> None:
        if event.checkpoint_path:
            checkpoint = self.jobs.append_checkpoint(
                job,
                step_number=event.step or job.last_step,
                path=event.checkpoint_path,
                metrics=_metrics_of(event.data),
            )
            self.jobs.append_event(
                job,
                event.level,
                event.message,
                data={**(event.data or {}), "checkpoint_id": checkpoint.id},
                event_type="metrics",
            )
            return
        if event.step:
            job.last_step = max(job.last_step, event.step)
        self.jobs.append_event(
            job, event.level, event.message, data=event.data, event_type=event.type
        )

    def _apply(
        self, job: TrainingJob, outcome: RunOutcome, planned: PlannedRun
    ) -> None:
        job.last_step = max(job.last_step, outcome.last_step)
        if outcome.last_checkpoint:
            job.resume_from = outcome.last_checkpoint
        if outcome.status is RunStatus.PREEMPTED:
            job.preemptions += 1
            job.tenant_state = TenantState.PREEMPTED
            # Status stays RUNNING: preemption is not a protocol state.
            self.jobs.append_event(
                job,
                "info",
                (
                    f"preempted at step {job.last_step} after "
                    f"{job.preemptions} interruption(s); VRAM lease released, will "
                    f"resume from {job.resume_from or 'the start'} at the next idle "
                    "window"
                ),
            )
            return
        if outcome.status is RunStatus.CANCELLED:
            self._finish_cancelled(job)
            return
        if outcome.status is RunStatus.FAILED:
            self._fail(job, outcome.error or "the executor failed without a message")
            return

        job.status = JobStatus.SUCCEEDED
        job.tenant_state = TenantState.DONE
        job.finished_at = int(self._clock())
        job.trained_tokens = outcome.trained_tokens
        name = outcome.fine_tuned_model or planned.spec.output_dir.name
        suffix = f":{job.suffix}" if job.suffix else ""
        job.fine_tuned_model = f"ft:{job.model}{suffix}:{name}"
        if outcome.artifact_path:
            job.extension["artifact_path"] = outcome.artifact_path
        self.jobs.append_event(
            job,
            "info",
            f"training succeeded at step {job.last_step}; "
            f"model {job.fine_tuned_model}",
        )

    def _fail(
        self,
        job: TrainingJob,
        message: str,
        *,
        extension: Optional[dict[str, Any]] = None,
    ) -> None:
        job.status = JobStatus.FAILED
        job.tenant_state = TenantState.DONE
        job.finished_at = int(self._clock())
        job.error = {"code": "training_failed", "message": message, "param": None}
        if extension:
            job.extension.update(extension)
        self.jobs.append_event(job, "error", message)

    def _finish_cancelled(self, job: TrainingJob) -> None:
        job.status = JobStatus.CANCELLED
        job.tenant_state = TenantState.DONE
        job.finished_at = int(self._clock())
        self.jobs.append_event(job, "info", "job cancelled")

    def _note_waiting(self, job: TrainingJob, verdict: IdleVerdict) -> None:
        if job.tenant_state is TenantState.PREEMPTED:
            return
        if job.status is JobStatus.VALIDATING_FILES:
            job.status = JobStatus.QUEUED
        if job.extension.get("_waiting_reason") == verdict.reason():
            return
        job.extension["_waiting_reason"] = verdict.reason()
        self.jobs.append_event(
            job, "info", f"waiting for an idle window: {verdict.reason()}"
        )


def _metrics_of(data: Optional[dict[str, Any]]) -> dict[str, Any]:
    """The subset of a metrics payload the checkpoint object carries."""
    if not data:
        return {}
    mapping = {
        "loss": "train_loss",
        "train_loss": "train_loss",
        "eval_loss": "valid_loss",
        "step": "step",
        "global_step": "step",
    }
    out: dict[str, Any] = {}
    for key, value in data.items():
        target = mapping.get(key)
        if target and isinstance(value, (int, float)):
            out[target] = float(value)
    return out
