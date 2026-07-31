# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Detecting a consumer that is gone, and releasing what it was holding.

Originally written for the video-enhance tenant (#344a, TASK_333 §8.2) and
generalized here to every endpoint class in the server (#344b). The argument
is unchanged and is worth restating, because it is the reason this exists at
all rather than being left to the transport:

**Why a timeout and not just the disconnect.** A client that closes the
connection is already handled: Starlette throws into the response generator
and its ``finally`` tears the job down. The case that is not handled is the
client that neither closes nor reads. The socket stays open, the TCP window
stays full, the sink coroutine never returns, and back-pressure -- working
exactly as designed -- stalls the whole chain and holds every resource
indefinitely. From the server's side that is indistinguishable from a very
slow consumer, and the only thing that separates them is *how long*.

So the duration is the policy, and it is configured per endpoint class: a
file-to-file enhance behind a slow disk is legitimately quiet for a long
time, a paused player longer still, and a chat client that has not accepted a
token in ninety seconds is holding KV blocks for nobody.

**What "progress" means.** Bytes accepted by the transport, not bytes
produced by the chain. A stalled client makes the chain stop producing, so
"the pipeline is idle" is a consequence of the stall and cannot be its
evidence. Only a write the transport took is proof the peer is still there.

**Two lessons that cost real debugging** (from #339, and they generalize):

1. A suspended generator never reaches its own ``finally``. Whatever the
   release has to do, it has to do itself -- it cannot rely on the stream's
   cleanup path running, because that path is exactly what is stuck.
2. Cancelling an executor does not unblock a stalled pipeline. A producer
   parked in ``ring.put`` on a full ring only observes the cancel flag after
   that await returns, and on a stalled pipeline it never does. The rings
   have to be closed.

**Grace, not pinning.** Between "quiet" and "declared dead" the attachment
sits in :class:`~sglang.srt.liveness.grace.AttachmentPhase.GRACE`, where its
claims are published to the process attachment registry as reclaimable. This
module does not reclaim anything; it makes the state visible to the ladder
that does.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Iterable

from sglang.srt.liveness.classes import DEFAULT_TIMEOUTS_S, EndpointClass
from sglang.srt.liveness.grace import (
    AttachmentPhase,
    AttachmentRegistry,
    ResourceClaim,
    global_attachment_registry,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ConsumerGone",
    "ConsumerWatchdog",
    "LivenessConfig",
    "LivenessPolicy",
    "LivenessState",
    "global_liveness_config",
    "set_global_liveness_config",
]

#: Fraction of a class's timeout after which the attachment enters grace.
#: Unmeasured. A quarter leaves three quarters of the window in which a
#: reclaimer may act before the client is declared dead, which is the
#: interesting part; a much smaller fraction would put healthy but bursty
#: consumers into grace constantly and make the signal useless.
DEFAULT_GRACE_FRACTION = 0.25


class ConsumerGone(RuntimeError):
    """The consumer was declared dead and what it held is being released."""

    def __init__(self, job_id: str, silent_for: float, timeout: float) -> None:
        super().__init__(
            f"job {job_id}: the consumer accepted no bytes for {silent_for:.1f}s, "
            f"over the {timeout:.1f}s limit for its endpoint class; releasing "
            "what it held"
        )
        self.job_id = job_id
        self.silent_for = silent_for
        self.timeout = timeout


@dataclass(frozen=True)
class LivenessPolicy:
    """How long silence is tolerated, and how long teardown may take."""

    endpoint_class: EndpointClass = EndpointClass.VIDEO_STREAM
    #: Seconds without a byte accepted by the transport before the consumer is
    #: declared dead. ``None`` falls back to the class default; zero or
    #: negative disables detection for this class.
    timeout_s: float | None = None
    #: How often the watchdog looks. It never needs to be finer than the
    #: timeout it is enforcing, and a coarse poll is what keeps this off the
    #: serving hot path -- the streaming loop only stamps a float.
    poll_interval_s: float = 1.0
    #: How long teardown itself may take before it is escalated to a cancel.
    #: A job that will not release in this long is a bug; the reservation
    #: must not be held hostage to it.
    teardown_timeout_s: float = 30.0
    #: Seconds of silence after which the attachment is offered to the
    #: reclamation ladder. ``None`` derives it from ``grace_fraction``.
    grace_after_s: float | None = None
    grace_fraction: float = DEFAULT_GRACE_FRACTION

    def resolved_timeout(self) -> float | None:
        if self.timeout_s is not None:
            return self.timeout_s if self.timeout_s > 0 else None
        return DEFAULT_TIMEOUTS_S.get(self.endpoint_class)

    def resolved_grace_after(self) -> float | None:
        """When grace starts. Never after the timeout it precedes."""
        timeout = self.resolved_timeout()
        if timeout is None:
            return None
        if self.grace_after_s is not None:
            grace = self.grace_after_s
        else:
            grace = timeout * self.grace_fraction
        return max(0.0, min(grace, timeout))

    def __post_init__(self) -> None:
        if self.poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        if self.teardown_timeout_s <= 0:
            raise ValueError("teardown_timeout_s must be positive")
        if not 0.0 <= self.grace_fraction <= 1.0:
            raise ValueError("grace_fraction must be between 0.0 and 1.0")


@dataclass
class LivenessState:
    """What a progress endpoint reports about a consumer."""

    job_id: str
    endpoint_class: str
    timeout_s: float | None
    started_at: float
    last_progress_at: float
    bytes_accepted: int = 0
    writes_accepted: int = 0
    declared_dead: bool = False
    declared_dead_at: float | None = None
    in_grace: bool = False

    def silent_for(self, now: float | None = None) -> float:
        return (now if now is not None else time.monotonic()) - self.last_progress_at

    def snapshot(self, now: float | None = None) -> dict:
        return {
            "endpoint_class": self.endpoint_class,
            "timeout_s": self.timeout_s,
            "bytes_accepted": self.bytes_accepted,
            "writes_accepted": self.writes_accepted,
            "silent_for_s": round(self.silent_for(now), 3),
            "declared_dead": self.declared_dead,
            "in_grace": self.in_grace,
        }


class ConsumerWatchdog:
    """Watches one consumer and releases what it held when it is gone.

    Constructed around a teardown coroutine rather than around the job, so
    the same watchdog serves an SSE token stream, an enhance response, a
    preview tap and anything else with a consumer -- and so the tests can
    assert on release without a pipeline behind it.

    ``claims`` is what this attachment holds. Declaring it is what puts the
    attachment on the reclamation ladder during grace; omitting it leaves the
    watchdog fully functional and the attachment invisible to reclaimers,
    which is the right default for something that holds only a queue.
    """

    def __init__(
        self,
        *,
        job_id: str,
        policy: LivenessPolicy,
        release: Callable[[], Awaitable[None]],
        clock: Callable[[], float] | None = None,
        claims: Iterable[ResourceClaim] = (),
        registry: AttachmentRegistry | None = None,
    ) -> None:
        self.job_id = job_id
        self.policy = policy
        self._release = release
        self._clock = clock or time.monotonic
        self._claims = tuple(claims)
        self._registry = registry
        self._registered = False
        now = self._clock()
        self.state = LivenessState(
            job_id=job_id,
            endpoint_class=policy.endpoint_class.value,
            timeout_s=policy.resolved_timeout(),
            started_at=now,
            last_progress_at=now,
        )
        self._task: asyncio.Task | None = None
        self._stopped = asyncio.Event()
        self._released = asyncio.Event()

    # -- the streaming side ------------------------------------------------

    def note_progress(self, byte_count: int) -> None:
        """Called after the transport has accepted ``byte_count`` bytes.

        After, never before. A write that was handed to the transport and is
        still blocked is not evidence of anything, which is the whole
        distinction this class exists to make.
        """
        self.state.last_progress_at = self._clock()
        self.state.bytes_accepted += byte_count
        self.state.writes_accepted += 1
        if self.state.in_grace:
            # The consumer came back. Take it off the ladder before a
            # reclaimer acts on a claim that is live again.
            self.state.in_grace = False
            self._set_phase(AttachmentPhase.ACTIVE, 0.0)

    @property
    def released(self) -> bool:
        return self._released.is_set()

    @property
    def in_grace(self) -> bool:
        return self.state.in_grace

    # -- the registry side -------------------------------------------------

    def _resolve_registry(self) -> AttachmentRegistry | None:
        if self._registry is not None:
            return self._registry
        if self._claims:
            return global_attachment_registry()
        return None

    def _set_phase(self, phase: AttachmentPhase, silent_for: float) -> None:
        if not self._registered:
            return
        registry = self._resolve_registry()
        if registry is not None:
            registry.set_phase(self.job_id, phase, silent_for_s=silent_for)

    # -- lifecycle ---------------------------------------------------------

    async def __aenter__(self) -> ConsumerWatchdog:
        self.start()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.stop()

    def start(self) -> None:
        registry = self._resolve_registry()
        if registry is not None:
            registry.register(
                self.job_id,
                endpoint_class=self.state.endpoint_class,
                claims=self._claims,
                timeout_s=self.state.timeout_s,
            )
            self._registered = True
        if self.state.timeout_s is None:
            # Detection disabled for this class. Say so once: a deployment
            # that turned it off should be able to find out from the log why
            # a dead client is still holding a card.
            logger.info(
                "job %s: no liveness timeout for endpoint class %s; a consumer "
                "that stops reading will hold this job until it disconnects",
                self.job_id,
                self.state.endpoint_class,
            )
            return
        self._task = asyncio.create_task(self._watch())

    async def stop(self) -> None:
        self._stopped.set()
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._unregister()

    def _unregister(self) -> None:
        if not self._registered:
            return
        registry = self._resolve_registry()
        if registry is not None:
            registry.unregister(self.job_id)
        self._registered = False

    async def _watch(self) -> None:
        timeout = self.state.timeout_s
        assert timeout is not None
        grace_after = self.policy.resolved_grace_after()
        interval = min(self.policy.poll_interval_s, timeout)
        try:
            while not self._stopped.is_set():
                try:
                    await asyncio.wait_for(self._stopped.wait(), timeout=interval)
                    return
                except asyncio.TimeoutError:
                    pass
                silent = self.state.silent_for(self._clock())
                if silent >= timeout:
                    await self._declare_dead(silent, timeout)
                    return
                if (
                    grace_after is not None
                    and silent >= grace_after
                    and not self.state.in_grace
                ):
                    # Not dead, not busy. The claims go on the ladder; the
                    # stream keeps its place and is still allowed to recover.
                    self.state.in_grace = True
                    self._set_phase(AttachmentPhase.GRACE, silent)
        except asyncio.CancelledError:
            raise

    async def _declare_dead(self, silent: float, timeout: float) -> None:
        self.state.declared_dead = True
        self.state.declared_dead_at = self._clock()
        self.state.in_grace = False
        self._set_phase(AttachmentPhase.DEAD, silent)
        logger.warning(
            "job %s: consumer accepted no bytes for %.1fs (limit %.1fs for %s); "
            "releasing",
            self.job_id,
            silent,
            timeout,
            self.state.endpoint_class,
        )
        try:
            # Teardown is bounded. A release that hangs would leave the
            # reservation held by a job nobody is watching, which is the exact
            # state this class exists to end.
            await asyncio.wait_for(
                self._release(), timeout=self.policy.teardown_timeout_s
            )
        except asyncio.TimeoutError:
            logger.error(
                "job %s: release did not complete within %.1fs; the reservation "
                "may still be held",
                self.job_id,
                self.policy.teardown_timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 - a bad release must not hang
            logger.error("job %s: release failed: %s", self.job_id, exc)
        finally:
            self._released.set()
            # Unregister here rather than leaving it to ``stop()``. On this
            # path ``stop()`` is never reached: the stream is suspended at its
            # yield and will not resume, so its ``finally`` does not run --
            # the same rule that puts the teardown in the release callback.
            # Without this the registry would grow one dead entry per dropped
            # client and report claims that were already given back.
            self._unregister()


@dataclass
class LivenessConfig:
    """Per-endpoint-class timeouts, as a server argument.

    One object rather than a flag per class so a deployment that adds an
    endpoint class does not have to add a flag, and so the whole policy is
    one thing a status endpoint can report.
    """

    timeouts_s: dict[str, float] = field(default_factory=dict)
    poll_interval_s: float = 1.0
    teardown_timeout_s: float = 30.0
    grace_fraction: float = DEFAULT_GRACE_FRACTION

    @classmethod
    def parse(cls, spec: str | None, **kwargs) -> LivenessConfig:
        """``video_stream=300,llm_stream=45`` into a config.

        A zero or negative value disables detection for that class, which is
        a real choice (a batch export nobody is watching by design) and is
        therefore expressible rather than clamped.
        """
        timeouts: dict[str, float] = {}
        known = {c.value for c in EndpointClass}
        for item in (spec or "").split(","):
            item = item.strip()
            if not item:
                continue
            name, sep, value = item.partition("=")
            name = name.strip()
            if not sep:
                raise ValueError(
                    f"liveness timeout spec {item!r} must be "
                    f"'<endpoint-class>=<seconds>'; known classes are "
                    f"{sorted(known)}"
                )
            if name not in known:
                raise ValueError(
                    f"unknown endpoint class {name!r}; known classes are "
                    f"{sorted(known)}"
                )
            try:
                timeouts[name] = float(value)
            except ValueError:
                raise ValueError(
                    f"liveness timeout for {name!r} must be a number of seconds, "
                    f"got {value!r}"
                ) from None
        return cls(timeouts_s=timeouts, **kwargs)

    def policy_for(self, endpoint_class: EndpointClass) -> LivenessPolicy:
        configured = self.timeouts_s.get(endpoint_class.value)
        return LivenessPolicy(
            endpoint_class=endpoint_class,
            timeout_s=configured,
            poll_interval_s=self.poll_interval_s,
            teardown_timeout_s=self.teardown_timeout_s,
            grace_fraction=self.grace_fraction,
        )

    def describe(self) -> dict:
        out: dict[str, float | None] = {}
        for endpoint_class in EndpointClass:
            out[endpoint_class.value] = self.policy_for(
                endpoint_class
            ).resolved_timeout()
        return out


_GLOBAL_CONFIG: LivenessConfig | None = None


def set_global_liveness_config(config: LivenessConfig | None) -> None:
    """Install the server-flag policy for code that cannot be handed one.

    The streaming guards live inside request handlers three layers below the
    place that parsed the flags, and threading a config through every
    ``serving_*`` constructor would be a wide change for a value that is
    constant for the process's whole life. One module-level install at
    startup, read-only afterwards.
    """
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = config


def global_liveness_config() -> LivenessConfig:
    """The installed policy, or an all-defaults one if nothing installed it.

    Never ``None``: an engine embedded in another process gets the documented
    class defaults rather than no detection at all.
    """
    return _GLOBAL_CONFIG if _GLOBAL_CONFIG is not None else LivenessConfig()
