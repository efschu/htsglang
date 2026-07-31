# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Putting a watchdog on a response without rewriting the response.

Every long-lived endpoint in the tree already produces either an async
iterator of frames or a single long await. Rather than teach each handler
about liveness, this module wraps the thing it already returns:

* :func:`guarded_stream` wraps an async iterator and stamps progress *after*
  each yield, which is the only moment the process knows the transport took
  the bytes.
* :func:`guard_streaming_response` swaps that wrapper into a Starlette
  ``StreamingResponse`` in place, so a handler gains liveness in one line and
  keeps its status code, headers and background tasks untouched.
* :func:`guard_generate_stream` is the same thing preconfigured for the
  token-stream endpoints: the release aborts the request ids, which is what
  actually frees KV blocks and the running-batch slot.
* :func:`await_with_liveness` covers the shape that has no frames at all --
  the image and speech lanes, where one ``await`` can sit for fifteen minutes
  behind a client that hung up in the first second.

**The suspended-generator rule.** When the watchdog declares a consumer dead,
the wrapper is parked in ``yield`` and will never resume: its ``finally``
does not run, and neither does the wrapped generator's. Everything that must
happen on death happens in the release callback. The wrapper's ``finally`` is
for the normal and the disconnected paths only.
"""

from __future__ import annotations

import asyncio
import logging
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Sequence,
)

from sglang.srt.liveness.classes import EndpointClass
from sglang.srt.liveness.grace import ClaimKind, ResourceClaim
from sglang.srt.liveness.watchdog import (
    ConsumerGone,
    ConsumerWatchdog,
    LivenessConfig,
    global_liveness_config,
)

logger = logging.getLogger(__name__)

__all__ = [
    "await_with_liveness",
    "claims_for_rids",
    "guard_generate_stream",
    "guard_streaming_response",
    "guarded_stream",
]


def _frame_size(chunk: Any) -> int:
    """Bytes the transport took, best effort.

    A frame is normally ``bytes`` or ``str``. Anything else still counts as
    one accepted write, because the fact that control came back is the
    evidence; the size is only for the progress report.
    """
    try:
        return len(chunk)
    except TypeError:
        return 1


async def guarded_stream(
    source: AsyncIterable[Any],
    watchdog: ConsumerWatchdog,
    *,
    size_of: Callable[[Any], int] = _frame_size,
) -> AsyncIterator[Any]:
    """Yield ``source`` through, under ``watchdog``.

    Raises :class:`ConsumerGone` if the consumer was declared dead and then
    -- against expectation -- started reading again. That case is rare and
    the right answer is to end the response: its resources were released when
    it was declared dead, so continuing would stream from a torn-down job.
    """
    watchdog.start()
    try:
        async for chunk in source:
            if watchdog.released:
                raise ConsumerGone(
                    watchdog.job_id,
                    watchdog.state.silent_for(),
                    watchdog.state.timeout_s or 0.0,
                )
            yield chunk
            # After the yield, not before. Control returns here only once the
            # transport has taken the chunk, so this is the one place in the
            # process that knows the peer is still there.
            watchdog.note_progress(size_of(chunk))
    finally:
        # Reached on normal completion and on a client that actually closed.
        # Not reached when the watchdog declared the consumer dead: see the
        # suspended-generator rule in the module docstring.
        await watchdog.stop()


def guard_streaming_response(
    response: Any,
    watchdog: ConsumerWatchdog,
    *,
    size_of: Callable[[Any], int] = _frame_size,
) -> Any:
    """Wrap a ``StreamingResponse``'s body iterator in place and return it.

    Returns the response unchanged if it has no body iterator, so a caller
    that may have been handed an error response does not have to check.
    """
    body_iterator = getattr(response, "body_iterator", None)
    if body_iterator is None:
        return response
    response.body_iterator = guarded_stream(body_iterator, watchdog, size_of=size_of)
    return response


def _request_ids(obj: Any) -> tuple[str, ...]:
    """The rids behind one request object, single or batched."""
    rid = getattr(obj, "rid", None)
    if rid is None:
        return ()
    if isinstance(rid, str):
        return (rid,)
    if isinstance(rid, (list, tuple)):
        return tuple(str(r) for r in rid if r)
    return (str(rid),)


def guard_generate_stream(
    response: Any,
    *,
    tokenizer_manager: Any,
    obj: Any = None,
    rids: Sequence[str] | None = None,
    endpoint_class: EndpointClass = EndpointClass.LLM_STREAM,
    config: LivenessConfig | None = None,
) -> Any:
    """Put an LLM-stream watchdog on a token SSE response.

    The release is ``abort_request`` per rid. That is the only call that
    reaches the scheduler and frees KV blocks; the ``create_abort_task``
    background task the endpoints already carry fires *after* the response
    body ends, which for a client that stopped reading is never.

    A request with no rid -- there is one code path where the id is assigned
    later -- gets no watchdog rather than a watchdog that cannot release
    anything. Silent by design: it is the pre-existing behaviour, not a
    regression this introduces.
    """
    ids = tuple(rids) if rids is not None else _request_ids(obj)
    if not ids:
        return response
    if getattr(response, "body_iterator", None) is None:
        return response

    policy = (config or global_liveness_config()).policy_for(endpoint_class)

    async def release() -> None:
        for rid in ids:
            try:
                tokenizer_manager.abort_request(rid)
            except Exception as exc:  # noqa: BLE001 - one bad rid must not
                # leave the others held.
                logger.warning("liveness abort failed for rid %s: %s", rid, exc)

    watchdog = ConsumerWatchdog(
        job_id=f"stream-{ids[0]}",
        policy=policy,
        release=release,
        claims=claims_for_rids(ids),
    )
    return guard_streaming_response(response, watchdog)


async def await_with_liveness(
    awaitable: Awaitable[Any],
    *,
    raw_request: Any,
    endpoint_class: EndpointClass,
    job_id: str,
    config: LivenessConfig | None = None,
    claims: Iterable[ResourceClaim] = (),
    on_abandoned: Callable[[], Awaitable[None]] | None = None,
) -> Any:
    """Await one long call, giving up if the client hangs up first.

    The frame-based watchdog has nothing to watch here: an image generation
    writes once, at the end. The evidence available instead is Starlette's
    ``is_disconnected()``, which is cheap and is polled on the watchdog's own
    interval rather than in any hot loop.

    Raises :class:`ConsumerGone` when the client left. The caller decides what
    to answer; nobody is listening, so the point is only that the awaited
    task is cancelled and ``on_abandoned`` has run.
    """
    policy = (config or global_liveness_config()).policy_for(endpoint_class)
    timeout = policy.resolved_timeout()
    task = asyncio.ensure_future(awaitable)
    if raw_request is None or not hasattr(raw_request, "is_disconnected"):
        return await task

    from sglang.srt.liveness.grace import (  # noqa: PLC0415 - avoid a cycle
        AttachmentPhase,
        global_attachment_registry,
    )

    registry = global_attachment_registry() if claims else None
    if registry is not None:
        registry.register(
            job_id,
            endpoint_class=endpoint_class.value,
            claims=claims,
            timeout_s=timeout,
        )
    grace_after = policy.resolved_grace_after()
    waited = 0.0
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=policy.poll_interval_s)
            if done:
                return task.result()
            waited += policy.poll_interval_s
            try:
                disconnected = await raw_request.is_disconnected()
            except Exception:  # noqa: BLE001 - a transport that cannot answer
                # is not evidence of a dead client.
                disconnected = False
            if not disconnected:
                continue
            if (
                registry is not None
                and grace_after is not None
                and waited < grace_after
            ):
                registry.set_phase(job_id, AttachmentPhase.GRACE, silent_for_s=waited)
            logger.warning(
                "job %s: client disconnected after %.1fs of a %s call; "
                "cancelling the lane request",
                job_id,
                waited,
                endpoint_class.value,
            )
            task.cancel()
            if on_abandoned is not None:
                try:
                    await asyncio.wait_for(
                        on_abandoned(), timeout=policy.teardown_timeout_s
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("job %s: abandon handler failed: %s", job_id, exc)
            raise ConsumerGone(job_id, waited, timeout or 0.0)
    finally:
        if registry is not None:
            registry.unregister(job_id)
        if not task.done():
            task.cancel()


def claims_for_rids(rids: Sequence[str]) -> tuple[ResourceClaim, ...]:
    """KV claims for a set of request ids.

    No byte figure: the number of KV blocks behind an rid lives in the
    scheduler process, and asking for it across the ZMQ boundary on every
    attachment would put a round trip on the serving path to improve a
    reporting field. The claim is still useful without it -- a reclaimer that
    wants to know *which* requests are held by dead suspects gets the answer.
    """
    return tuple(ResourceClaim(kind=ClaimKind.KV, key=rid) for rid in rids)
