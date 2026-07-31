# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The interface every piece of idle work is wrapped behind (DESIGN #347 W2).

ANALYSE #347 item 1: training is one idle tenant among many. #341 built the
machinery -- idle detection, a VRAM lease, preemption by checkpoint-and-
release, an event channel -- for exactly one kind of work. This module is the
generalization of the *shape* of that work, and nothing else: it does not
re-implement idle detection (that is :class:`~sglang.srt.training.tenant.
IdleMonitor`, imported), and it does not re-implement the ledger (that is
:mod:`sglang.srt.registry.ledger`, imported).

Three properties define a tenant of the workbench:

**It can say what a segment costs, before running it.** :func:`price_segment`
is DESIGN #341 D2 lifted off training: every post named, no safety factor, no
implicit ceiling, priced against NVML totals and the current ledger. A
rejection carries the arithmetic.

**Its work is segmented.** A segment is a unit the tenant can abandon
cleanly -- one tuner shape, one probe run, one training attempt. The
scheduler grants one at a time, so "the rig is busy again" never has to wait
for a whole queue to drain.

**It is preemptible, not merely killable.** :class:`WorkSegment` has the same
three methods as :class:`~sglang.srt.training.backends.BackendRun` for the
reason given there: a tenant that can only be killed loses all of its work on
every interruption, and a tenant that loses all its work is a squatter.
"""

from __future__ import annotations

import abc
import asyncio
import contextlib
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

MIB = 1024 * 1024
GIB = 1024 * MIB

#: Registry engine class of idle-workbench work. Classes 1-3 are
#: autoregressive, diffusion and utility (#333); 4 is training (#341); 5 is
#: everything the rig does for itself while nobody is watching.
CLASS_WORKBENCH = 5

#: One CUDA context plus the allocator's first arena, per subprocess. The same
#: named post the training gate uses, for the same reason: it is real memory
#: the driver takes before the first tensor exists, and hiding it inside a
#: fudge factor is how a "fits" verdict turns into a runtime OOM.
DEFAULT_CUDA_CONTEXT_BYTES = 600 * MIB


class SegmentStatus(str, Enum):
    """How one segment ended."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    #: Stopped on the scheduler's request because serving demand arrived. Not
    #: an ending: the work item goes back on the tenant's queue.
    PREEMPTED = "preempted"
    CANCELLED = "cancelled"

    @property
    def is_progress(self) -> bool:
        return self is SegmentStatus.SUCCEEDED


@dataclass(frozen=True)
class WorkEvent:
    """One thing worth telling the operator about.

    Same shape as :class:`~sglang.srt.training.backends.BackendEvent`, on
    purpose: the two event channels are read by the same frontend building
    block (ANALYSE #347 item 5).
    """

    level: str
    message: str
    data: Optional[dict[str, Any]] = None
    type: str = "message"


#: Called synchronously from the scheduler's own asyncio task.
EventSink = Callable[[WorkEvent], None]


@dataclass(frozen=True)
class WorkEstimate:
    """What one segment of this tenant's work would cost.

    ``posts`` is the itemisation and ``per_card_bytes`` is their sum. Both are
    reported: a caller staring at a rejection needs to know *which* post is
    the big one, which is the whole reason DESIGN #341 D2 forbids a single
    opaque number.
    """

    #: Bytes on each card the segment touches.
    per_card_bytes: int
    #: Named contributions to ``per_card_bytes``.
    posts: Mapping[str, int] = field(default_factory=dict)
    #: Pin the work to these cards. Empty means "the scheduler picks".
    card_uuids: tuple[str, ...] = ()
    #: How many cards a segment needs when they are not pinned. ``0`` means
    #: every visible card, which is what a rig-wide measurement wants.
    cards_wanted: int = 1
    disk_bytes: int = 0
    ram_bytes: int = 0
    #: The tenant's own guess at wall-clock. Reported, never enforced; the
    #: enforced bound is ``--workbench-segment-timeout-s``.
    expected_seconds: float = 0.0
    #: ``True`` when the tenant prices and leases per work item inside its own
    #: gate, so the scheduler arbitrates ordering only. Used by the training
    #: adapter, whose demand depends on the submitted job.
    self_leased: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "per_card_mib": self.per_card_bytes // MIB,
            "posts_mib": {k: v // MIB for k, v in self.posts.items()},
            "card_uuids": list(self.card_uuids),
            "cards_wanted": self.cards_wanted,
            "disk_mib": self.disk_bytes // MIB,
            "ram_mib": self.ram_bytes // MIB,
            "expected_seconds": round(self.expected_seconds, 1),
            "self_leased": self.self_leased,
        }


@dataclass(frozen=True)
class Feasibility:
    """Whether a segment fits right now, with the arithmetic either way."""

    fits: bool
    reason: str
    chosen_cards: tuple[str, ...] = ()
    chosen_indices: tuple[int, ...] = ()
    per_card_bytes: int = 0
    shortfall_bytes: int = 0

    def render(self) -> str:
        return self.reason

    def to_json(self) -> dict[str, Any]:
        return {
            "fits": self.fits,
            "reason": self.reason,
            "cards": list(self.chosen_cards),
            "card_indices": list(self.chosen_indices),
            "per_card_mib": self.per_card_bytes // MIB,
            "shortfall_mib": self.shortfall_bytes // MIB,
        }


@dataclass(frozen=True)
class WorkGrant:
    """What the scheduler hands a tenant for one segment.

    Every tenant here launches a subprocess, and the fork's isolation rule is
    process-level: ``CUDA_VISIBLE_DEVICES`` pins the child, and inside it
    ``cuda:0`` is unambiguous. No in-process logical-to-physical table exists.

    That variable is set from the **UUIDs**, not the indices (AUDIT #331).
    ``CUDA_VISIBLE_DEVICES`` accepts either form, and the UUID form is the one
    that cannot be misread: an index in it is resolved in the driver's PCI
    order, which is not the order the parent's ``torch.cuda`` reports, so an
    index that travelled from a parent's CUDA view into a child's environment
    is a wrong-card bind waiting for the two orders to diverge. The Class-1/2/3
    registry adapters already pin by UUID; this is the same rule.
    ``card_indices`` stays for logging and for the ledger's NVML lookups.
    """

    card_uuids: tuple[str, ...]
    card_indices: tuple[int, ...]
    per_card_bytes: int
    artifact_root: Path
    #: Wall-clock after which the scheduler stops the segment regardless.
    deadline_ts: float = 0.0

    @property
    def visible_devices(self) -> str:
        if self.card_uuids:
            return ",".join(self.card_uuids)
        return ",".join(str(i) for i in self.card_indices)

    def to_json(self) -> dict[str, Any]:
        return {
            "cards": list(self.card_uuids),
            "card_indices": list(self.card_indices),
            "per_card_mib": self.per_card_bytes // MIB,
            "artifact_root": str(self.artifact_root),
            "deadline_ts": self.deadline_ts,
        }


@dataclass(frozen=True)
class SegmentOutcome:
    """How one segment ended, and what it left behind."""

    status: SegmentStatus
    detail: str = ""
    artifact_path: Optional[str] = None
    error: Optional[str] = None
    data: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "detail": self.detail,
            "artifact_path": self.artifact_path,
            "error": self.error,
            "data": dict(self.data),
        }


class WorkSegment(abc.ABC):
    """One running segment. One object per attempt, not per work item."""

    @abc.abstractmethod
    async def wait(self) -> SegmentOutcome:
        """Block until the segment ends by itself."""

    @abc.abstractmethod
    async def preempt(self, *, timeout_s: float = 60.0) -> SegmentOutcome:
        """Stop for now, cleanly, and leave the work item requeueable.

        Bounded: a segment that will not stop within ``timeout_s`` is
        escalated, because the whole point of preemption is that the serving
        tenant does not wait on it.
        """

    @abc.abstractmethod
    async def cancel(self, *, timeout_s: float = 30.0) -> SegmentOutcome:
        """Stop for good. The work item is dropped, not requeued."""


class IdleWorkTenant(abc.ABC):
    """One kind of useful idle work."""

    #: Stable identifier. Used in the API, in the ledger tenant id and in the
    #: event log, so it is a slug and not a sentence.
    name: str = "abstract"

    #: Lower runs first (DESIGN #347 W4). A per-tenant integer so a deployment
    #: that disagrees with the shipped order changes one number.
    priority: int = 50

    def __init__(self) -> None:
        self._paused = False

    # -- what the scheduler asks --------------------------------------------

    def available(self) -> tuple[bool, str]:
        """Can this tenant run on this host at all, and if not, precisely why.

        The :class:`~sglang.srt.training.backends.BackendProbe` pattern one
        level up: a missing script, an absent optional dependency or an
        unreadable queue file is a named skip, not a segment that fails over
        and over.
        """
        return True, ""

    @abc.abstractmethod
    def pending(self) -> int:
        """How many work items are waiting. ``0`` means nothing to do."""

    @abc.abstractmethod
    def estimate(self) -> WorkEstimate:
        """What the next segment would cost. Must not start anything."""

    @abc.abstractmethod
    async def start_segment(self, grant: WorkGrant, sink: EventSink) -> WorkSegment:
        """Start one segment against the granted cards."""

    def describe(self) -> str:
        """One line for the operator."""
        return self.name

    def snapshot(self) -> dict[str, Any]:
        available, reason = self.available()
        return {
            "name": self.name,
            "priority": self.priority,
            "description": self.describe(),
            "pending": self.pending(),
            "paused": self.paused,
            "available": available,
            "unavailable_reason": reason,
        }

    # -- control ------------------------------------------------------------

    @property
    def paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def enqueue(self, item: Mapping[str, Any]) -> str:
        """Add one work item. Returns a short identifier for it."""
        raise NotImplementedError(
            f"tenant {self.name!r} has no enqueue surface; its work is derived "
            "from the state of the rig rather than submitted"
        )


# ---------------------------------------------------------------------------
# Pricing (DESIGN #347 W3 -- DESIGN #341 D2, generalized off training)
# ---------------------------------------------------------------------------


def price_segment(
    estimate: WorkEstimate,
    machine: Any,
    *,
    store: Any = None,
    tenant_id: str = "workbench",
) -> Feasibility:
    """Would this segment fit on this machine, right now?

    ``machine`` is a :class:`~sglang.srt.training.feasibility.MachineResources`
    -- the same probe the training gate uses, so both gates see one machine.
    ``store`` is a :class:`~sglang.srt.registry.ledger.ReservationStore`; when
    given, other tenants' holdings and the corridor count against the demand,
    which is the difference between "would this fit on an empty rig" and
    "would this fit now".

    There is no safety factor here and no rounding down. Every post is in
    ``estimate.posts`` and the caller can argue with any of them.
    """
    if estimate.self_leased:
        return Feasibility(
            fits=True,
            reason=(
                "priced and leased by the tenant itself, per work item; the "
                "scheduler arbitrates ordering only"
            ),
            per_card_bytes=estimate.per_card_bytes,
        )

    cards = tuple(getattr(machine, "cards", ()) or ())
    if not cards:
        probe_error = getattr(machine, "probe_error", None)
        return Feasibility(
            fits=False,
            reason=(
                "no GPU is visible to this process"
                + (f" ({probe_error})" if probe_error else "")
            ),
            per_card_bytes=estimate.per_card_bytes,
            shortfall_bytes=estimate.per_card_bytes,
        )

    by_uuid = {c.uuid: c for c in cards}
    if estimate.card_uuids:
        unknown = [u for u in estimate.card_uuids if u not in by_uuid]
        if unknown:
            return Feasibility(
                fits=False,
                reason=(
                    "pinned to card(s) this machine does not have: "
                    + ", ".join(unknown)
                ),
                per_card_bytes=estimate.per_card_bytes,
            )
        chosen = [by_uuid[u] for u in estimate.card_uuids]
    elif estimate.cards_wanted <= 0:
        chosen = list(cards)
    else:
        # The emptiest cards first: idle work should land where it disturbs
        # the least, and a tie between equal cards is broken by index so the
        # choice is reproducible across ticks.
        ranked = sorted(cards, key=lambda c: (-c.available_bytes, c.index))
        chosen = ranked[: estimate.cards_wanted]
        if len(chosen) < estimate.cards_wanted:
            return Feasibility(
                fits=False,
                reason=(
                    f"{estimate.cards_wanted} card(s) wanted, {len(cards)} " "visible"
                ),
                per_card_bytes=estimate.per_card_bytes,
                shortfall_bytes=estimate.per_card_bytes,
            )

    need = int(estimate.per_card_bytes)
    if store is not None:
        report = _ledger_report(store, chosen, need, tenant_id)
        if report is not None and not report.fits:
            return Feasibility(
                fits=False,
                reason="the VRAM ledger refuses it: " + report.render(),
                per_card_bytes=need,
                shortfall_bytes=report.shortfall_bytes,
            )
    else:
        short = [c for c in chosen if c.available_bytes < need]
        if short:
            worst = max(need - c.available_bytes for c in short)
            lines = "; ".join(
                f"card {c.index} {c.name}: {need / MIB:.0f} MiB wanted, "
                f"{c.available_bytes / MIB:.0f} MiB claimable of "
                f"{c.total_bytes / MIB:.0f} MiB total"
                for c in short
            )
            return Feasibility(
                fits=False,
                reason=f"does not fit: {lines}",
                per_card_bytes=need,
                shortfall_bytes=worst,
            )

    disk_short = _disk_shortfall(machine, estimate)
    if disk_short:
        return Feasibility(
            fits=False,
            reason=disk_short,
            per_card_bytes=need,
        )
    ram_short = _ram_shortfall(machine, estimate)
    if ram_short:
        return Feasibility(fits=False, reason=ram_short, per_card_bytes=need)

    posts = ", ".join(f"{k} {v / MIB:.0f} MiB" for k, v in estimate.posts.items())
    return Feasibility(
        fits=True,
        reason=(
            f"{need / MIB:.0f} MiB on {len(chosen)} card(s)"
            + (f" ({posts})" if posts else "")
        ),
        chosen_cards=tuple(c.uuid for c in chosen),
        chosen_indices=tuple(c.index for c in chosen),
        per_card_bytes=need,
    )


def _ledger_report(store: Any, chosen: Sequence[Any], need: int, tenant_id: str):
    try:
        from sglang.srt.registry.ledger import CardDemand, plan_reservation

        return plan_reservation(
            store,
            [CardDemand(card_uuid=c.uuid, reserved_bytes=need) for c in chosen],
            {c.uuid: c.total_bytes for c in chosen},
            excluding_tenant=tenant_id,
        )
    except Exception as exc:  # noqa: BLE001 - an unreadable ledger is reportable
        logger.debug("workbench: ledger plan failed: %s", exc)
        return None


def _disk_shortfall(machine: Any, estimate: WorkEstimate) -> str:
    free = int(getattr(machine, "disk_free_bytes", 0) or 0)
    if estimate.disk_bytes and free and free < estimate.disk_bytes:
        return (
            f"needs {estimate.disk_bytes / MIB:.0f} MiB on "
            f"{getattr(machine, 'disk_path', '?')}, {free / MIB:.0f} MiB free"
        )
    return ""


def _ram_shortfall(machine: Any, estimate: WorkEstimate) -> str:
    free = int(getattr(machine, "ram_available_bytes", 0) or 0)
    if estimate.ram_bytes and free and free < estimate.ram_bytes:
        return (
            f"needs {estimate.ram_bytes / MIB:.0f} MiB of host RAM, "
            f"{free / MIB:.0f} MiB available"
        )
    return ""


# ---------------------------------------------------------------------------
# The shared segment body: a subprocess whose stdout is the event stream
# ---------------------------------------------------------------------------


class SubprocessSegment(WorkSegment):
    """A segment that is one child process, stopped by signalling it.

    Both non-training tenants in M1 are of this shape, and so are three of the
    four ANALYSE #347 candidates that are not built yet, so it lives here
    rather than in either tenant.

    Preemption is SIGTERM to the child's **process group**, then a bounded
    wait, then SIGKILL. The group matters: the tuner and the probe both spawn
    torch, which spawns its own helpers, and signalling only the direct child
    leaves those holding a CUDA context -- exactly the "released the lease but
    not the memory" failure the lease exists to prevent.

    The work item is requeued by the tenant on ``PREEMPTED``. This class does
    not know what a work item is; it only reports how the child ended.
    """

    def __init__(
        self,
        *,
        argv: Sequence[str],
        cwd: Optional[Path] = None,
        env: Optional[Mapping[str, str]] = None,
        sink: EventSink,
        label: str,
        artifact_path: Optional[str] = None,
        line_filter: Optional[Callable[[str], Optional[WorkEvent]]] = None,
    ) -> None:
        self.argv = [str(a) for a in argv]
        self.cwd = cwd
        self.env = dict(env or {})
        self.sink = sink
        self.label = label
        self.artifact_path = artifact_path
        self.line_filter = line_filter
        self._process: Any = None
        self._reader: Optional[asyncio.Task] = None
        self._tail: list[str] = []
        self._stopping: Optional[SegmentStatus] = None

    async def start(self) -> SubprocessSegment:
        environ = dict(os.environ)
        environ.update(self.env)
        self.sink(
            WorkEvent(
                "info",
                f"{self.label}: launching {' '.join(self.argv[:2])}",
                data={
                    "argv": self.argv,
                    "cuda_visible_devices": environ.get("CUDA_VISIBLE_DEVICES", ""),
                },
            )
        )
        self._process = await asyncio.create_subprocess_exec(
            *self.argv,
            cwd=str(self.cwd) if self.cwd else None,
            env=environ,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        self._reader = asyncio.ensure_future(self._read())
        return self

    async def _read(self) -> None:
        stream = self._process.stdout
        if stream is None:
            return
        while True:
            raw = await stream.readline()
            if not raw:
                return
            line = raw.decode("utf-8", "replace").rstrip()
            if not line:
                continue
            # Bounded: a chatty child must not become a memory leak with a
            # long fuse. The tail is what a failure report needs.
            self._tail.append(line)
            if len(self._tail) > 50:
                del self._tail[0]
            event = self.line_filter(line) if self.line_filter else None
            if event is not None:
                self.sink(event)

    async def wait(self) -> SegmentOutcome:
        code = await self._process.wait()
        if self._reader is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader
        if self._stopping is not None:
            return SegmentOutcome(
                status=self._stopping,
                detail=f"{self.label}: stopped on request (exit {code})",
                artifact_path=self.artifact_path,
            )
        if code == 0:
            return SegmentOutcome(
                status=SegmentStatus.SUCCEEDED,
                detail=f"{self.label}: finished",
                artifact_path=self.artifact_path,
            )
        return SegmentOutcome(
            status=SegmentStatus.FAILED,
            detail=f"{self.label}: exit {code}",
            error="\n".join(self._tail[-10:]) or f"exit code {code}",
            artifact_path=self.artifact_path,
        )

    async def preempt(self, *, timeout_s: float = 60.0) -> SegmentOutcome:
        return await self._stop(SegmentStatus.PREEMPTED, timeout_s)

    async def cancel(self, *, timeout_s: float = 30.0) -> SegmentOutcome:
        return await self._stop(SegmentStatus.CANCELLED, timeout_s)

    async def _stop(self, status: SegmentStatus, timeout_s: float) -> SegmentOutcome:
        self._stopping = status
        self._signal(signal.SIGTERM)
        try:
            await asyncio.wait_for(self._process.wait(), timeout=max(0.05, timeout_s))
        except (asyncio.TimeoutError, TimeoutError):
            self.sink(
                WorkEvent(
                    "warn",
                    f"{self.label}: did not exit within {timeout_s:.0f}s, killing it",
                )
            )
            self._signal(signal.SIGKILL)
            with contextlib.suppress(asyncio.TimeoutError, TimeoutError):
                await asyncio.wait_for(self._process.wait(), timeout=10.0)
        return await self.wait()

    def _signal(self, sig: int) -> None:
        process = self._process
        if process is None or process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            # Own children only, by process group id: never a broad pkill.
            os.killpg(os.getpgid(process.pid), sig)


def now() -> float:
    return time.time()


__all__ = [
    "CLASS_WORKBENCH",
    "DEFAULT_CUDA_CONTEXT_BYTES",
    "EventSink",
    "Feasibility",
    "GIB",
    "IdleWorkTenant",
    "MIB",
    "SegmentOutcome",
    "SegmentStatus",
    "SubprocessSegment",
    "WorkEstimate",
    "WorkEvent",
    "WorkGrant",
    "WorkSegment",
    "price_segment",
]
