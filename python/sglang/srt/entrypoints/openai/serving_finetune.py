# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""``/v1/fine_tuning/jobs`` -- the standard training protocol (D3, #341-M1).

DESIGN #341 D3: external suites must be able to point at this fork and submit
training jobs without knowing anything about it, which means speaking the
de-facto standard rather than inventing a surface. So the routes, the field
names, the list envelopes, the cursor pagination and the job state machine are
OpenAI's, and everything the fork adds -- method ladder, target model path,
backend selection, tenant state -- rides in ``hyperparameters``, ``metadata``
or a namespaced ``x-htsglang`` block that a vanilla client ignores.

Two things this adapter owns that the service does not:

**Error shape.** A store exception becomes the OpenAI envelope with the
fork's numbers in the extension. An infeasible job is a 400 whose extension
carries the whole method ladder, so the caller can read off which rung fits
instead of guessing.

**The event stream.** ``?stream=true`` on the events endpoint is a live SSE
tap, and a tap has a consumer that can die. Per #344 the stream is bounded by
a :class:`~sglang.srt.video_enhance.liveness.ConsumerWatchdog`: a client that
stops reading is dropped promptly rather than pinning a subscriber queue
forever. The job it was watching is *not* touched -- fine-tuning jobs are
fire-and-forget by protocol, and a submitter that goes away has not cancelled
anything.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Optional

from fastapi.responses import ORJSONResponse, StreamingResponse

from sglang.srt.entrypoints.openai.errors import (
    error_type_for_status,
    openai_error_response,
)
from sglang.srt.entrypoints.openai.serving_files import store_error_response
from sglang.srt.training.service import BackendRejected, InfeasibleRequest
from sglang.srt.training.store import JobEvent, StoreError, page

logger = logging.getLogger(__name__)

#: Default page size for the events and checkpoints collections, matching
#: OpenAI's own.
DEFAULT_LIMIT = 20
MAX_LIMIT = 100


class OpenAIServingFineTuning:
    """The fine-tuning surface. All decisions live in the training service."""

    def __init__(self, service, *, liveness=None) -> None:
        self._service = service
        self._liveness = liveness

    # -- jobs ---------------------------------------------------------------

    async def create(self, body: dict[str, Any]) -> ORJSONResponse:
        try:
            job = self._service.create_job(body)
        except InfeasibleRequest as exc:
            # The rejection carries the arithmetic and the ladder, which is
            # the difference between "try something else" and "try what".
            return openai_error_response(
                exc.decision.render(),
                status_code=exc.status_code,
                err_type=error_type_for_status(exc.status_code),
                param="model",
                code=exc.code,
                extension={
                    "capability": "fine_tuning",
                    "reason": exc.code,
                    "feasibility": exc.decision.to_json(),
                    "what_would_make_it_work": list(exc.decision.remedies),
                },
            )
        except BackendRejected as exc:
            return store_error_response(
                exc,
                capability="fine_tuning",
                reason=exc.code,
                backend_probe=exc.probe.to_json(),
                what_would_make_it_work=list(exc.probe.remedies),
            )
        except StoreError as exc:
            return store_error_response(exc)
        return ORJSONResponse(job.to_json())

    async def list(
        self, *, after: Optional[str] = None, limit: int = DEFAULT_LIMIT
    ) -> ORJSONResponse:
        jobs = self._service.jobs.list()
        start = 0
        if after:
            for index, job in enumerate(jobs):
                if job.id == after:
                    start = index + 1
                    break
            else:
                return store_error_response(
                    _not_found(f"No such FineTuningJob object: {after}")
                )
        size = _clamp(limit)
        window = jobs[start : start + size]
        return ORJSONResponse(page(window, has_more=start + size < len(jobs)))

    async def retrieve(self, job_id: str) -> ORJSONResponse:
        try:
            return ORJSONResponse(self._service.jobs.get(job_id).to_json())
        except StoreError as exc:
            return store_error_response(exc)

    async def cancel(self, job_id: str) -> ORJSONResponse:
        try:
            return ORJSONResponse(self._service.cancel_job(job_id).to_json())
        except StoreError as exc:
            return store_error_response(exc)

    async def checkpoints(
        self, job_id: str, *, after: Optional[str] = None, limit: int = DEFAULT_LIMIT
    ) -> ORJSONResponse:
        try:
            job = self._service.jobs.get(job_id)
        except StoreError as exc:
            return store_error_response(exc)
        items = list(reversed(job.checkpoints))
        start = 0
        if after:
            for index, checkpoint in enumerate(items):
                if checkpoint.id == after:
                    start = index + 1
                    break
        size = _clamp(limit)
        window = items[start : start + size]
        return ORJSONResponse(page(window, has_more=start + size < len(items)))

    # -- events -------------------------------------------------------------

    async def events(
        self,
        job_id: str,
        *,
        after: Optional[str] = None,
        limit: int = DEFAULT_LIMIT,
        stream: bool = False,
    ):
        try:
            job = self._service.jobs.get(job_id)
        except StoreError as exc:
            return store_error_response(exc)
        if stream:
            return StreamingResponse(
                self._stream_events(job_id, after=after),
                media_type="text/event-stream",
            )
        try:
            window, has_more = self._service.jobs.events_after(
                job, after=after, limit=_clamp(limit)
            )
        except StoreError as exc:
            return store_error_response(exc)
        return ORJSONResponse(page(window, has_more=has_more))

    async def _stream_events(
        self, job_id: str, *, after: Optional[str]
    ) -> AsyncIterator[str]:
        """Backlog then live tail, bounded by a #344 consumer watchdog."""
        from sglang.srt.liveness import (  # noqa: PLC0415
            ClaimKind,
            ConsumerWatchdog,
            EndpointClass,
            LivenessConfig,
            ResourceClaim,
        )

        store = self._service.jobs
        job = store.get(job_id)
        queue = store.subscribe(job_id)
        config = self._liveness or LivenessConfig()
        policy = config.policy_for(EndpointClass.TRAINING_EVENTS)

        gone = asyncio.Event()

        async def release() -> None:
            # The consumer is dropped; the job is untouched. A fine-tuning job
            # is fire-and-forget by protocol, so a submitter that dies has not
            # cancelled it (#344 client-liveness rule).
            gone.set()
            store.unsubscribe(job_id, queue)

        watchdog = ConsumerWatchdog(
            job_id=f"ftevents-{job_id}",
            policy=policy,
            release=release,
            # A subscriber queue and nothing else. Declared anyway so a stuck
            # tap shows up in the attachment registry next to the streams that
            # do hold a card -- an operator asking "what is attached and
            # quiet" wants the complete answer, not the expensive half of it.
            claims=(ResourceClaim(kind=ClaimKind.SUBSCRIBER, key=job_id),),
        )
        watchdog.start()
        # Keepalives are what make silence attributable. Without them a job
        # that simply has nothing to say would look exactly like a dead
        # consumer, and the watchdog would drop a client that is fine.
        timeout = policy.resolved_timeout()
        keepalive_s = max(1.0, (timeout or 60.0) / 4.0)
        try:
            backlog, _ = store.events_after(
                job, after=after, limit=store.max_events_per_job
            )
            for event in backlog:
                frame = _frame(event)
                yield frame
                watchdog.note_progress(len(frame))

            while not gone.is_set():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=keepalive_s)
                except asyncio.TimeoutError:
                    if job.status.is_terminal:
                        break
                    frame = ": keepalive\n\n"
                    yield frame
                    watchdog.note_progress(len(frame))
                    continue
                frame = _frame(event)
                yield frame
                watchdog.note_progress(len(frame))
                if job.status.is_terminal and queue.empty():
                    break
            yield "data: [DONE]\n\n"
        finally:
            await watchdog.stop()
            store.unsubscribe(job_id, queue)


def _frame(event: JobEvent) -> str:
    return f"data: {json.dumps(event.to_json())}\n\n"


def _clamp(limit: int) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        return DEFAULT_LIMIT
    return max(1, min(MAX_LIMIT, value))


def _not_found(message: str) -> StoreError:
    error = StoreError(message)
    error.status_code = 404
    error.code = "not_found"
    return error
