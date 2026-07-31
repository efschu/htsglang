# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""One scheduler for every kind of idle work (DESIGN #347 W4).

The loop is deliberately the same shape as
:class:`~sglang.srt.training.tenant.TrainingTenant`'s, because it is the same
contract with the queue generalized:

    idle -> pick the highest-priority tenant with work -> price the segment
    -> claim the cross-session window -> lease the VRAM -> run one segment,
    watching demand throughout -> release everything -> report.

Three properties are worth stating rather than reading out of the code.

**Exactly one tenant runs at a time.** Idle work is opportunistic by
definition; two opportunistic tenants sharing a card would tune and measure
each other, and neither result would mean anything.

**Preemption is checked between segments as well as inside them.** The
worst-case latency between a request arriving and the rig being free is
``poll_seconds`` plus the running tenant's own preempt time -- never "until
the queue drains". That is what makes a queue of overnight work safe to leave
running on a rig that also serves.

**Nothing here decides whether the rig is idle.** The verdict comes from the
#341 :class:`~sglang.srt.training.tenant.IdleMonitor`, built from the same
sources the training service uses, so a rig that looks idle to training looks
idle to the workbench by construction.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from sglang.srt.training.feasibility import MachineResources
from sglang.srt.training.tenant import IdleMonitor, IdleVerdict
from sglang.srt.workbench.arb import ArbClaim, ArbDirectory, ArbRefused
from sglang.srt.workbench.log import WorkLog
from sglang.srt.workbench.tenant import (
    CLASS_WORKBENCH,
    MIB,
    IdleWorkTenant,
    SegmentOutcome,
    SegmentStatus,
    WorkEvent,
    WorkGrant,
    price_segment,
)

logger = logging.getLogger(__name__)


@dataclass
class WorkbenchConfig:
    """The workbench's knobs. Every one of them is a server flag."""

    enabled: bool = False
    artifact_root: Path = Path("/var/tmp/htsglang/workbench")
    #: Seconds of quiet before the rig counts as idle. Same meaning as the
    #: training flag; kept separate so a deployment can let training start
    #: sooner than the rig's self-maintenance, or the other way round.
    grace_seconds: float = 120.0
    #: How often demand is re-checked while a segment runs. The worst-case
    #: delay between a request arriving and preemption starting.
    poll_seconds: float = 2.0
    lease_seconds: float = 120.0
    preempt_timeout_s: float = 60.0
    cancel_timeout_s: float = 30.0
    #: Backoff after a refused window or a rejected lease, so a busy rig is
    #: not re-probed hotly.
    reject_backoff_s: float = 30.0
    #: Hard bound on one segment. A tenant whose work does not fit inside this
    #: must cut it into smaller items; that is what "segment" means.
    segment_timeout_s: float = 1800.0
    #: Cross-session arbitration directory. Empty disables the claim entirely,
    #: which is right on a rig where nothing else competes for the cards.
    arb_dir: str = ""
    arb_session: str = "operator"
    #: Well inside the 20-minute staleness window of the arbitration protocol,
    #: because a workbench window can be much longer than that.
    arb_heartbeat_s: float = 300.0
    max_events: int = 5000

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
            "segment_timeout_s": self.segment_timeout_s,
            "arb_dir": self.arb_dir,
            "arb_session": self.arb_session,
            "arb_heartbeat_s": self.arb_heartbeat_s,
        }


@dataclass
class TenantRuntime:
    """What the scheduler remembers about a tenant between ticks."""

    tenant: IdleWorkTenant
    segments_run: int = 0
    segments_preempted: int = 0
    segments_failed: int = 0
    last_outcome: Optional[SegmentOutcome] = None
    last_finished_ts: float = 0.0
    #: The last blocking reason reported, so the log is not repeated every
    #: two seconds while a condition persists.
    last_block_reason: str = ""

    def to_json(self) -> dict[str, Any]:
        body = self.tenant.snapshot()
        body.update(
            {
                "segments_run": self.segments_run,
                "segments_preempted": self.segments_preempted,
                "segments_failed": self.segments_failed,
                "last_outcome": (
                    self.last_outcome.to_json() if self.last_outcome else None
                ),
                "last_finished_ts": self.last_finished_ts or None,
                "blocked_reason": self.last_block_reason,
            }
        )
        return body


class Workbench:
    """The queue of useful idle work, and the loop that drains it."""

    def __init__(
        self,
        tenants: Sequence[IdleWorkTenant],
        *,
        config: WorkbenchConfig,
        monitor: IdleMonitor,
        reservation_store: Any = None,
        machine_resolver: Callable[[], MachineResources] = MachineResources,
        log: Optional[WorkLog] = None,
        arb: Optional[ArbDirectory] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.monitor = monitor
        self.reservation_store = reservation_store
        self.machine_resolver = machine_resolver
        self.log = log or WorkLog(max_entries=config.max_events)
        self.arb = arb
        self._clock = clock
        self._runtimes: dict[str, TenantRuntime] = {}
        for tenant in tenants:
            self.register(tenant)
        self._task: Optional[asyncio.Task] = None
        self._stopping = asyncio.Event()
        self._wake = asyncio.Event()
        self._current: Optional[tuple[IdleWorkTenant, Any]] = None
        self._reservation: Any = None
        self._claim: Optional[ArbClaim] = None
        self._last_arb_heartbeat = 0.0
        self.last_verdict: Optional[IdleVerdict] = None
        self.paused = False
        #: Set by tests and by the preempt demo to force a decision without
        #: waiting for the poll interval. Same hook the training tenant has.
        self.force_preempt = asyncio.Event()

    # -- registration -------------------------------------------------------

    def register(self, tenant: IdleWorkTenant) -> None:
        if tenant.name in self._runtimes:
            raise ValueError(f"a tenant named {tenant.name!r} is already registered")
        self._runtimes[tenant.name] = TenantRuntime(tenant=tenant)

    def tenant(self, name: str) -> IdleWorkTenant:
        runtime = self._runtimes.get(name)
        if runtime is None:
            known = ", ".join(sorted(self._runtimes)) or "none"
            raise KeyError(f"unknown workbench tenant {name!r}; registered: {known}")
        return runtime.tenant

    @property
    def tenants(self) -> list[IdleWorkTenant]:
        return [r.tenant for r in self._ordered()]

    def _ordered(self) -> list[TenantRuntime]:
        return sorted(
            self._runtimes.values(), key=lambda r: (r.tenant.priority, r.tenant.name)
        )

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._task is not None:
            return
        Path(self.config.artifact_root).mkdir(parents=True, exist_ok=True)
        self._task = asyncio.get_running_loop().create_task(self._loop())

    async def stop(self) -> None:
        self._stopping.set()
        self._wake.set()
        # The running segment is stopped BEFORE the loop task is cancelled.
        # Cancelling the task first unwinds ``_run_one``, which clears
        # ``_current`` on its way out, and the segment's child process would
        # then outlive the server holding a CUDA context -- the exact
        # "released the lease but not the memory" state the lease exists to
        # prevent.
        current = self._current
        if current is not None:
            with contextlib.suppress(Exception):
                await current[1].cancel(timeout_s=self.config.cancel_timeout_s)
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._current = None
        await self._release_all()

    def wake(self) -> None:
        """Nudge the loop: work was enqueued or the pause state changed."""
        self._wake.set()

    def pause(self, paused: bool = True) -> None:
        self.paused = bool(paused)
        self.log.append(
            "workbench",
            "info",
            "paused by request" if paused else "resumed by request",
        )
        if paused:
            self.force_preempt.set()
        self.wake()

    # -- snapshot -----------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        current = self._current
        body: dict[str, Any] = {
            "config": self.config.to_json(),
            "paused": self.paused,
            "running": current[0].name if current else None,
            "idle": self.last_verdict.to_json() if self.last_verdict else None,
            "tenants": [r.to_json() for r in self._ordered()],
            "pending_total": sum(r.tenant.pending() for r in self._runtimes.values()),
            "events_seq": self.log.last_seq,
        }
        if self.arb is not None:
            body["arb"] = self.arb.snapshot()
            body["arb"]["claim"] = self._claim.to_json() if self._claim else None
        return body

    # -- the loop -----------------------------------------------------------

    async def _loop(self) -> None:
        logger.info("idle workbench: started (%s)", self.config.to_json())
        self.log.append(
            "workbench",
            "info",
            "started with "
            + ", ".join(
                f"{r.tenant.name}(p{r.tenant.priority})" for r in self._ordered()
            ),
        )
        while not self._stopping.is_set():
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a loop that dies stops all idle work
                logger.exception("idle workbench: tick failed")
                self.log.append("workbench", "error", "tick failed; see the server log")
                await self._sleep(self.config.poll_seconds)

    async def _tick(self) -> None:
        if self.paused:
            await self._sleep(self.config.poll_seconds)
            return

        candidates = self._runnable()
        if not candidates:
            await self._sleep(self.config.poll_seconds)
            return

        verdict = self.monitor.sample()
        self.last_verdict = verdict
        if not verdict.idle:
            self._note_blocked(
                candidates[0], f"waiting for an idle window: {verdict.reason()}"
            )
            await self._sleep(self.config.poll_seconds)
            return

        machine = self.machine_resolver()
        for runtime in candidates:
            grant = self._grant_for(runtime, machine)
            if grant is None:
                continue
            await self._run_one(runtime, grant)
            return
        await self._sleep(self.config.poll_seconds)

    def _runnable(self) -> list[TenantRuntime]:
        out: list[TenantRuntime] = []
        for runtime in self._ordered():
            tenant = runtime.tenant
            if tenant.paused:
                continue
            available, reason = tenant.available()
            if not available:
                self._note_blocked(runtime, f"unavailable: {reason}")
                continue
            try:
                pending = tenant.pending()
            except Exception as exc:  # noqa: BLE001 - one bad tenant is not a veto
                self._note_blocked(
                    runtime, f"pending() failed: {type(exc).__name__}: {exc}"
                )
                continue
            if pending > 0:
                out.append(runtime)
        return out

    def _grant_for(
        self, runtime: TenantRuntime, machine: MachineResources
    ) -> Optional[WorkGrant]:
        tenant = runtime.tenant
        try:
            estimate = tenant.estimate()
        except Exception as exc:  # noqa: BLE001 - a bad estimate skips its own tenant
            self._note_blocked(runtime, f"estimate failed: {type(exc).__name__}: {exc}")
            return None
        feasibility = price_segment(
            estimate,
            machine,
            store=self.reservation_store,
            tenant_id=self._tenant_id(tenant),
        )
        if not feasibility.fits:
            self._note_blocked(runtime, feasibility.render())
            return None
        runtime.last_block_reason = ""
        return WorkGrant(
            card_uuids=feasibility.chosen_cards,
            card_indices=feasibility.chosen_indices,
            per_card_bytes=feasibility.per_card_bytes,
            artifact_root=Path(self.config.artifact_root) / tenant.name,
            deadline_ts=self._clock() + self.config.segment_timeout_s,
        )

    def _tenant_id(self, tenant: IdleWorkTenant) -> str:
        return f"workbench-{tenant.name}"

    async def _sleep(self, seconds: float) -> None:
        self._wake.clear()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._wake.wait(), timeout=max(0.01, seconds))

    def _note_blocked(self, runtime: TenantRuntime, reason: str) -> None:
        if runtime.last_block_reason == reason:
            return
        runtime.last_block_reason = reason
        self.log.append(runtime.tenant.name, "info", reason)

    # -- one segment --------------------------------------------------------

    async def _run_one(self, runtime: TenantRuntime, grant: WorkGrant) -> None:
        tenant = runtime.tenant
        estimate = tenant.estimate()

        if not await self._claim_window(runtime, grant):
            await self._sleep(self.config.reject_backoff_s)
            return
        if not estimate.self_leased and not await self._acquire_lease(runtime, grant):
            await self._release_all()
            await self._sleep(self.config.reject_backoff_s)
            return

        grant.artifact_root.mkdir(parents=True, exist_ok=True)

        def sink(event: WorkEvent) -> None:
            self.log.append(
                tenant.name,
                event.level,
                event.message,
                data=event.data,
                event_type=event.type,
            )

        self.log.append(
            tenant.name,
            "info",
            f"segment starting on card(s) {list(grant.card_indices) or 'none'} "
            f"with {grant.per_card_bytes // MIB} MiB each",
            data={"grant": grant.to_json(), "estimate": estimate.to_json()},
        )
        try:
            segment = await tenant.start_segment(grant, sink)
        except Exception as exc:  # noqa: BLE001 - a failed launch fails one segment
            runtime.segments_failed += 1
            runtime.last_outcome = SegmentOutcome(
                status=SegmentStatus.FAILED,
                detail="launch failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            self.log.append(
                tenant.name,
                "error",
                f"segment launch failed: {type(exc).__name__}: {exc}",
            )
            await self._release_all()
            await self._sleep(self.config.reject_backoff_s)
            return

        self._current = (tenant, segment)
        try:
            outcome = await self._supervise(tenant, segment, grant)
        finally:
            self._current = None
            await self._release_all()
        self._apply(runtime, outcome)

    async def _supervise(
        self, tenant: IdleWorkTenant, segment: Any, grant: WorkGrant
    ) -> SegmentOutcome:
        waiter = asyncio.ensure_future(segment.wait())
        try:
            while True:
                done, _ = await asyncio.wait(
                    {waiter}, timeout=max(0.01, self.config.poll_seconds)
                )
                if done:
                    return waiter.result()
                self._heartbeat()
                if self._stopping.is_set() or self.paused:
                    self.force_preempt.clear()
                    self.log.append(
                        tenant.name,
                        "info",
                        "preempting: the workbench is stopping or paused",
                    )
                    return await segment.preempt(
                        timeout_s=self.config.preempt_timeout_s
                    )
                if grant.deadline_ts and self._clock() > grant.deadline_ts:
                    self.log.append(
                        tenant.name,
                        "warn",
                        f"segment exceeded --workbench-segment-timeout-s "
                        f"({self.config.segment_timeout_s:.0f}s); cancelling it",
                    )
                    return await segment.cancel(timeout_s=self.config.cancel_timeout_s)
                verdict = self.monitor.sample()
                self.last_verdict = verdict
                if not verdict.idle or self.force_preempt.is_set():
                    self.force_preempt.clear()
                    self.log.append(
                        tenant.name,
                        "info",
                        f"preempting: {verdict.reason()}",
                        data=verdict.to_json(),
                    )
                    return await segment.preempt(
                        timeout_s=self.config.preempt_timeout_s
                    )
        finally:
            if not waiter.done():
                waiter.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await waiter

    def _apply(self, runtime: TenantRuntime, outcome: SegmentOutcome) -> None:
        runtime.last_outcome = outcome
        runtime.last_finished_ts = self._clock()
        if outcome.status is SegmentStatus.SUCCEEDED:
            runtime.segments_run += 1
            level = "info"
        elif outcome.status is SegmentStatus.PREEMPTED:
            runtime.segments_preempted += 1
            level = "info"
        elif outcome.status is SegmentStatus.FAILED:
            runtime.segments_failed += 1
            level = "error"
        else:
            level = "info"
        self.log.append(
            runtime.tenant.name,
            level,
            f"segment {outcome.status.value}: {outcome.detail or 'no detail'}"
            + (
                f" (artifact: {outcome.artifact_path})" if outcome.artifact_path else ""
            ),
            data=outcome.to_json(),
            event_type="segment",
        )

    # -- the window and the lease ------------------------------------------

    async def _claim_window(self, runtime: TenantRuntime, grant: WorkGrant) -> bool:
        if self.arb is None:
            return True
        try:
            self._claim = self.arb.claim(
                grant.card_indices, purpose=f"workbench-{runtime.tenant.name}"
            )
        except ArbRefused as exc:
            self._note_blocked(runtime, f"cross-session window refused: {exc}")
            return False
        except Exception as exc:  # noqa: BLE001 - an unusable arb dir is not a crash
            self._note_blocked(
                runtime, f"cross-session claim failed: {type(exc).__name__}: {exc}"
            )
            return False
        self._last_arb_heartbeat = self._clock()
        return True

    async def _acquire_lease(self, runtime: TenantRuntime, grant: WorkGrant) -> bool:
        if self.reservation_store is None or not grant.card_uuids:
            self.log.append(
                runtime.tenant.name,
                "warn",
                "no VRAM ledger is configured; this segment runs without a "
                "cross-process reservation and can collide with another tenant",
            )
            return True
        from sglang.srt.registry.ledger import (
            CardDemand,
            LedgerError,
            MultiCardReservation,
        )
        from sglang.srt.registry.ledger import TenantState as LedgerTenantState

        machine = self.machine_resolver()
        totals = {c.uuid: c.total_bytes for c in machine.cards}
        reservation = MultiCardReservation(
            self.reservation_store,
            tenant_id=self._tenant_id(runtime.tenant),
            klass=CLASS_WORKBENCH,
            totals=totals,
        )
        demands = [
            CardDemand(
                card_uuid=uuid,
                reserved_bytes=grant.per_card_bytes,
                posts={runtime.tenant.name: grant.per_card_bytes},
            )
            for uuid in grant.card_uuids
        ]
        try:
            reservation.acquire(
                demands,
                state=LedgerTenantState.HOT,
                pid=os.getpid(),
                lease_seconds=self.config.lease_seconds,
            )
        except LedgerError as exc:
            self._note_blocked(runtime, f"VRAM lease rejected: {exc}")
            return False
        self._reservation = reservation
        self.log.append(
            runtime.tenant.name,
            "info",
            f"leased {grant.per_card_bytes // MIB} MiB on "
            f"{len(grant.card_uuids)} card(s)",
        )
        return True

    def _heartbeat(self) -> None:
        if self._reservation is not None:
            with contextlib.suppress(Exception):
                self._reservation.heartbeat(lease_seconds=self.config.lease_seconds)
        claim = self._claim
        if claim is not None:
            now = self._clock()
            if now - self._last_arb_heartbeat >= self.config.arb_heartbeat_s:
                self._last_arb_heartbeat = now
                with contextlib.suppress(Exception):
                    claim.heartbeat()

    async def _release_all(self) -> None:
        reservation, self._reservation = self._reservation, None
        if reservation is not None:
            with contextlib.suppress(Exception):
                reservation.release()
        claim, self._claim = self._claim, None
        if claim is not None:
            with contextlib.suppress(Exception):
                claim.release()


__all__ = ["TenantRuntime", "Workbench", "WorkbenchConfig"]
