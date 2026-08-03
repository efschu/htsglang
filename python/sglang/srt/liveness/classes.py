# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The endpoint classes a client can attach to, and how patient each one is.

One enum, one default table, one rationale table. Everything else in the
server -- the watchdogs, the server flags, the grace registry, the docs --
reads its classes from here, so adding a kind of long-lived attachment is a
single edit rather than a scavenger hunt (#344).

**On the defaults.** None of these numbers is measured. Each is a judgement
about what silence means for that kind of consumer, and the judgement is
recorded next to the number in :data:`DEFAULT_TIMEOUT_RATIONALE` so a
deployment that disagrees can see what it is disagreeing with. The rationale
is the deliverable; the number is a starting point.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "DEFAULT_TIMEOUTS_S",
    "DEFAULT_TIMEOUT_RATIONALE",
    "EndpointClass",
]


class EndpointClass(str, Enum):
    """What kind of consumer is on the other end, and how patient to be."""

    # -- generation ------------------------------------------------------
    #: Token SSE from /generate, /v1/chat/completions, /v1/completions,
    #: /v1/responses, /v1/messages and the Ollama routes. Holds KV blocks and
    #: a running-batch slot, which is the most contended resource on the rig.
    LLM_STREAM = "llm_stream"
    #: A non-streaming request whose answer is one write. There is nothing to
    #: watch mid-flight; the class exists so the policy table is complete and
    #: so a deployment can see that the omission is deliberate.
    EMBEDDING = "embedding"

    # -- video enhance (#333/#339) ---------------------------------------
    #: A player or a downloader pulling an enhanced file. Legitimately slow:
    #: a viewer who pauses stops reading for as long as the pause lasts.
    VIDEO_STREAM = "video_stream"
    #: A preview tap. Drop-frame by construction, so a viewer that is there
    #: at all keeps accepting bytes; silence means nobody is watching.
    PREVIEW_TAP = "preview_tap"
    #: A progress poller. Not a stream; included so every endpoint class has
    #: a named policy rather than an implicit one.
    CONTROL = "control"

    # -- training (#341) --------------------------------------------------
    #: An SSE tap on a training job's event log (#341-M1). Legitimately quiet
    #: -- a training step is seconds and a checkpoint is minutes apart -- so
    #: the stream sends keepalives and silence really is the consumer's.
    TRAINING_EVENTS = "training_events"

    # -- multimodal lanes (#335-M0 / #333-M3) -----------------------------
    #: A diffusion generation forwarded to the image lane. One long await,
    #: no intermediate writes, and a GPU job on the far side of it.
    IMAGE_GENERATION = "image_generation"
    #: Text-to-speech, same shape as the image lane and same failure mode.
    AUDIO_SPEECH = "audio_speech"
    #: Streaming transcription of an upload.
    AUDIO_TRANSCRIPTION = "audio_transcription"
    #: A /v1/realtime websocket ASR session. The transport already reports a
    #: close; the class covers the socket that stays open and goes silent.
    REALTIME_SESSION = "realtime_session"

    # -- infrastructure ----------------------------------------------------
    #: A holder of a VRAM lease in the registry ledger (#305-M1). Not an HTTP
    #: attachment: the "consumer" is a tenant process and the "progress" is
    #: its heartbeat.
    REGISTRY_LEASE = "registry_lease"
    #: An SSE tap on the dashboard/bench event stream. A devtool, but an
    #: unattended one, and it drives real load while it is attached.
    DASHBOARD_SSE = "dashboard_sse"


#: Seconds of silence tolerated per class before the consumer is declared
#: dead. Unmeasured; see :data:`DEFAULT_TIMEOUT_RATIONALE` for why each is
#: what it is.
DEFAULT_TIMEOUTS_S: dict[EndpointClass, float] = {
    EndpointClass.LLM_STREAM: 90.0,
    EndpointClass.EMBEDDING: 60.0,
    EndpointClass.VIDEO_STREAM: 300.0,
    EndpointClass.PREVIEW_TAP: 15.0,
    EndpointClass.CONTROL: 60.0,
    EndpointClass.TRAINING_EVENTS: 120.0,
    EndpointClass.IMAGE_GENERATION: 900.0,
    EndpointClass.AUDIO_SPEECH: 300.0,
    EndpointClass.AUDIO_TRANSCRIPTION: 120.0,
    EndpointClass.REALTIME_SESSION: 60.0,
    EndpointClass.REGISTRY_LEASE: 120.0,
    EndpointClass.DASHBOARD_SSE: 60.0,
}

#: Why each default is what it is. Read this before changing one; the numbers
#: encode an argument about the consumer, not a measurement of the server.
DEFAULT_TIMEOUT_RATIONALE: dict[EndpointClass, str] = {
    EndpointClass.LLM_STREAM: (
        "Covers the gap BETWEEN chunks, not the time to the first chunk: "
        "_handle_streaming_request awaits the first chunk out of the generator "
        "before it builds the StreamingResponse, and only that response is "
        "wrapped, so the watchdog does not exist yet while the request queues "
        "and prefills (pinned by TimeToFirstTokenIsOutsideTheBudgetTest, "
        "#505-C-01). What it must survive is therefore a slow reader and an "
        "inter-token stall, not TTFT. Short enough that a dead chat client does "
        "not hold KV blocks for minutes. The most contended resource in the "
        "process, so the least patient of the stream classes."
    ),
    EndpointClass.EMBEDDING: (
        "One write, so the timeout never fires in practice. Present so the "
        "class table is total and so an embedding endpoint that later grows a "
        "stream inherits a policy instead of an omission."
    ),
    EndpointClass.VIDEO_STREAM: (
        "Deliberately generous: a paused player is a normal thing and "
        "reclaiming its job would be worse than holding it."
    ),
    EndpointClass.PREVIEW_TAP: (
        "Deliberately short: a preview is drop-frame by construction, so a "
        "viewer that is there at all keeps accepting bytes."
    ),
    EndpointClass.CONTROL: (
        "A poller that has not polled in a minute has stopped caring."
    ),
    EndpointClass.TRAINING_EVENTS: (
        "A training tap costs one subscriber queue, not a decoder and a card, "
        "so it can afford to be patient. Bounded all the same: an abandoned "
        "tap must not accumulate."
    ),
    EndpointClass.IMAGE_GENERATION: (
        "Matches the image lane's own 900 s client timeout, so the liveness "
        "check never fires before the request it is guarding would have."
    ),
    EndpointClass.AUDIO_SPEECH: (
        "Matches the speech lane's own 300 s client timeout, for the same "
        "reason as the image lane."
    ),
    EndpointClass.AUDIO_TRANSCRIPTION: (
        "Chunked ASR writes per chunk, so silence longer than a couple of "
        "chunks is the consumer's, not the model's."
    ),
    EndpointClass.REALTIME_SESSION: (
        "A realtime socket that has sent no audio for a minute is not "
        "realtime. The transport close is the primary signal; this bounds the "
        "socket that stays open and says nothing."
    ),
    EndpointClass.REGISTRY_LEASE: (
        "The ledger's own DEFAULT_LEASE_SECONDS. Restated here so the lease "
        "and the client-liveness ladder cannot drift apart silently."
    ),
    EndpointClass.DASHBOARD_SSE: (
        "A devtool tap. Long enough for a bench run between writes, short "
        "enough that a closed browser tab stops driving load."
    ),
}
