# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Detecting a consumer that is gone, and releasing the job it was holding.

TASK_333 §8.2 (user directive, #344a): the server must notice that the other
side has stopped consuming -- bandwidth collapse, network loss, a silently
dead client -- and clean up quickly rather than hold a decoder, an encoder,
a set of engines and a VRAM reservation for a viewer who left.

**Why a timeout and not just the disconnect.** A client that closes the
connection is already handled: Starlette throws into the response generator
and its ``finally`` tears the job down. The case that is not handled is the
client that neither closes nor reads. The socket stays open, the TCP window
stays full, the sink coroutine never returns, and back-pressure -- working
exactly as designed -- stalls the whole chain and holds every resource
indefinitely. From the server's side that is indistinguishable from a very
slow viewer, and the only thing that separates them is *how long*.

So the duration is the policy, and it is a configured number rather than a
constant: a file-to-file enhance behind a slow disk is legitimately quiet
for a long time, while a live preview tap that has not accepted a frame in
ten seconds has no viewer. :class:`LivenessPolicy` is per endpoint class for
that reason.

**What "progress" means.** Bytes accepted by the transport, not bytes
produced by the chain. A stalled client makes the chain stop producing, so
"the pipeline is idle" is a consequence of the stall and cannot be its
evidence. Only a write the transport took is proof the peer is still there.

**What this does not do.** During the grace window the job's resources stay
where they are. The directive asks for them to join the normal reclamation
ladder (idle tenant #341, pressure staircase #287, spill) instead of being
idly pinned, and that ladder is not wired to this tenant yet. What is built
here is the detection and the prompt release; the demotion is a registered
follow-on and is listed as open in TASK_333 §6.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


class EndpointClass(str, Enum):
    """What kind of consumer is on the other end, and how patient to be."""

    #: A player or a downloader pulling an enhanced file. Legitimately slow:
    #: a viewer who pauses stops reading for as long as the pause lasts.
    VIDEO_STREAM = "video_stream"
    #: A preview tap. Drop-frame by construction, so a viewer that is there
    #: at all keeps accepting bytes; silence means nobody is watching.
    PREVIEW_TAP = "preview_tap"
    #: A progress poller. Not a stream; included so every endpoint class has
    #: a named policy rather than an implicit one.
    CONTROL = "control"
    #: An SSE tap on a training job's event log (#341-M1). Legitimately quiet
    #: -- a training step is seconds and a checkpoint is minutes apart -- so
    #: the stream sends keepalives and silence really is the consumer's.
    TRAINING_EVENTS = "training_events"


#: Defaults per endpoint class. The video-stream figure is deliberately
#: generous -- a paused player is a normal thing and reclaiming its job would
#: be worse than holding it -- and the preview figure is deliberately short,
#: because a preview has no reason to pause.
DEFAULT_TIMEOUTS_S: dict[EndpointClass, float] = {
    EndpointClass.VIDEO_STREAM: 300.0,
    EndpointClass.PREVIEW_TAP: 15.0,
    EndpointClass.CONTROL: 60.0,
    # A training tap costs one subscriber queue, not a decoder and a card, so
    # it can afford to be patient with a reader that pauses. Bounded all the
    # same: an abandoned tap must not accumulate.
    EndpointClass.TRAINING_EVENTS: 120.0,
}


class ConsumerGone(RuntimeError):
    """The consumer was declared dead and its job is being released."""

    def __init__(self, job_id: str, silent_for: float, timeout: float) -> None:
        super().__init__(
            f"job {job_id}: the consumer accepted no bytes for {silent_for:.1f}s, "
            f"over the {timeout:.1f}s limit for its endpoint class; releasing the "
            "decoder, the encoder and the reservation"
        )
        self.job_id = job_id
        self.silent_for = silent_for
        self.timeout = timeout


@dataclass(frozen=True)
class LivenessPolicy:
    """How long silence is tolerated, and how long teardown may take."""

    endpoint_class: EndpointClass = EndpointClass.VIDEO_STREAM
    #: Seconds without a byte accepted by the transport before the consumer is
    #: declared dead. ``None`` disables detection for this class.
    timeout_s: float | None = None
    #: How often the watchdog looks. It never needs to be finer than the
    #: timeout it is enforcing, and a coarse poll is what keeps this off the
    #: serving hot path -- the streaming loop only stamps a float.
    poll_interval_s: float = 1.0
    #: How long teardown itself may take before it is escalated to a cancel.
    #: A job that will not release in this long is a bug; the reservation
    #: must not be held hostage to it.
    teardown_timeout_s: float = 30.0

    def resolved_timeout(self) -> float | None:
        if self.timeout_s is not None:
            return self.timeout_s if self.timeout_s > 0 else None
        return DEFAULT_TIMEOUTS_S.get(self.endpoint_class)

    def __post_init__(self) -> None:
        if self.poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        if self.teardown_timeout_s <= 0:
            raise ValueError("teardown_timeout_s must be positive")


@dataclass
class LivenessState:
    """What the progress endpoint reports about a consumer."""

    job_id: str
    endpoint_class: str
    timeout_s: float | None
    started_at: float
    last_progress_at: float
    bytes_accepted: int = 0
    writes_accepted: int = 0
    declared_dead: bool = False
    declared_dead_at: float | None = None

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
        }


class ConsumerWatchdog:
    """Watches one consumer and releases its job when the consumer is gone.

    Constructed around a teardown coroutine rather than around the job, so
    the same watchdog serves the enhance response, a preview tap and anything
    else with a consumer -- and so the tests can assert on release without a
    pipeline behind it.
    """

    def __init__(
        self,
        *,
        job_id: str,
        policy: LivenessPolicy,
        release: Callable[[], Awaitable[None]],
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.job_id = job_id
        self.policy = policy
        self._release = release
        self._clock = clock or time.monotonic
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

    @property
    def released(self) -> bool:
        return self._released.is_set()

    # -- lifecycle ---------------------------------------------------------

    async def __aenter__(self) -> "ConsumerWatchdog":
        self.start()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.stop()

    def start(self) -> None:
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

    async def _watch(self) -> None:
        timeout = self.state.timeout_s
        assert timeout is not None
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
        except asyncio.CancelledError:
            raise

    async def _declare_dead(self, silent: float, timeout: float) -> None:
        self.state.declared_dead = True
        self.state.declared_dead_at = self._clock()
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


@dataclass
class LivenessConfig:
    """Per-endpoint-class timeouts, as a server argument.

    One object rather than three flags so a deployment that adds an endpoint
    class does not have to add a flag, and so the whole policy is one thing
    the progress endpoint can report.
    """

    timeouts_s: dict[str, float] = field(default_factory=dict)
    poll_interval_s: float = 1.0
    teardown_timeout_s: float = 30.0

    @classmethod
    def parse(cls, spec: str | None, **kwargs) -> "LivenessConfig":
        """``video_stream=300,preview_tap=15`` into a config.

        A zero or negative value disables detection for that class, which is
        a real choice (a batch export nobody is watching by design) and is
        therefore expressible rather than clamped.
        """
        timeouts: dict[str, float] = {}
        for item in (spec or "").split(","):
            item = item.strip()
            if not item:
                continue
            name, _, value = item.partition("=")
            name = name.strip()
            known = {c.value for c in EndpointClass}
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
        )

    def describe(self) -> dict:
        out = {}
        for endpoint_class in EndpointClass:
            policy = self.policy_for(endpoint_class)
            out[endpoint_class.value] = policy.resolved_timeout()
        return out


__all__ = [
    "ConsumerGone",
    "ConsumerWatchdog",
    "DEFAULT_TIMEOUTS_S",
    "EndpointClass",
    "LivenessConfig",
    "LivenessPolicy",
    "LivenessState",
]
