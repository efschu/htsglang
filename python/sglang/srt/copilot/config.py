"""Copilot configuration.

Everything tunable in one dataclass so the launcher, the tests and the service
read the same defaults. No value here is a memory or GPU budget: this process
owns neither.
"""

from __future__ import annotations

from dataclasses import dataclass

# The runtime's realtime ASR surface fixes this: audio is base64 PCM16 inside
# JSON, and the pipeline rate the browser capture chain decimates to.
PIPELINE_SAMPLE_RATE = 16000

# Milliseconds of audio per client frame. Matches the #466 capture chain.
FRAME_MS = 20


@dataclass
class CopilotConfig:
    """Runtime-independent app configuration."""

    # --- which backend set ------------------------------------------------
    backend: str = "stub"
    """``"stub"`` or ``"rig"``. This single field is the whole difference
    between running against the stub set and running against the runtime
    (``backends.build_backend_set``). No app code branches on it."""

    # --- upstream runtime -------------------------------------------------
    runtime_base_url: str = "http://127.0.0.1:30000"
    """Base URL of the htsglang runtime. ``/v1/chat/completions`` and
    ``/v1/realtime`` are derived from it."""

    model: str = "default"
    """Model name passed to the chat endpoint."""

    api_key: str | None = None
    """Optional key for the runtime, when it runs with ``--api-key``."""

    # --- hint generation --------------------------------------------------
    hint_lane: str = "fast"
    """Scheduling lane for live hint requests. ``"fast"`` reaches
    ``GenerateReqInput.lane`` via the chat schema's ``lane`` field
    (``entrypoints/openai/protocol.py:869``) and only takes effect when the
    runtime runs with ``--enable-fast-lane``. On a runtime without that flag
    the field is inert, not an error."""

    hint_priority: int = 100
    """Request priority for live hints. Under ``--enable-priority-scheduling``
    this also seeds the radix node priority, which moves the topic prefix later
    in the eviction victim order (``mem_cache/evict_policy.py:41-47``). It does
    NOT pin the prefix -- no pin API exists (see DESIGN_502 §1.3(c))."""

    expander_priority: int = 0
    """Request priority for background briefing expansion. Deliberately the
    heavy-tier default so a live hint outranks it."""

    hint_max_tokens: int = 160
    expander_max_tokens: int = 400

    min_hint_interval_s: float = 2.0
    """Floor between two hint requests for one session, so a fast talker cannot
    turn every transcript delta into a decode."""

    expander_interval_s: float = 45.0
    """Cadence of the background briefing expander."""

    expander_first_delay_s: float = 12.0
    """When the FIRST expansion round runs after a session starts. Earlier than
    the cadence on purpose: the addendum worth having is the one that lands
    while the topic is still open, and a first round 45 seconds in is a feature
    nobody sees in a short call."""

    # --- topic warmth -----------------------------------------------------
    topic_touch_interval_s: float = 30.0
    """Cadence of the priming re-issue that refreshes a topic prefix's
    ``last_access_time`` and re-inserts it if it was evicted."""

    warm_ratio: float = 0.9
    """Fraction of a topic's primed prefix that must come back as
    ``cached_tokens`` for the topic to count as WARM."""

    partial_ratio: float = 0.25
    """Below this fraction the topic counts as COLD rather than PARTIAL."""

    # --- session ----------------------------------------------------------
    max_sessions: int = 8
    session_idle_timeout_s: float = 300.0
    journal_max_events: int = 512
    max_transcript_lines: int = 2000
    subscriber_queue_max: int = 256
    """Per-connection outbound queue depth. On overflow the connection is
    dropped with a named reason rather than silently falling behind; the client
    reconnects and replays from the journal, which retains more than this."""

    # --- stub backend set -------------------------------------------------
    # Only read when ``backend == "stub"``. These are the knobs the acceptance
    # runs turn, so they live with every other tunable instead of in argv.
    stub_script: str = "renewal-call"
    stub_word_ms: float = 200.0
    """Milliseconds between two scripted words. A speaker at 2.5 words/s."""

    stub_final_ms: float = 240.0
    """Delay between the last partial and the final line of an utterance."""

    stub_gap_ms: float = 700.0
    """Silence between two scripted utterances."""

    stub_audio_grace_s: float = 1.0
    """How long after the last audio frame a track still counts as streaming.
    Past it the script holds: no audio, no words."""

    stub_silence_hold_s: float = 5.0
    """How long a half-spoken utterance waits for audio to resume before it is
    finalised with what was said so far."""

    stub_hint_latency_ms: float = 160.0
    stub_hint_jitter_ms: float = 60.0
    stub_cold_penalty_ms: float = 700.0
    """Extra hint latency when the topic context is NOT prepared. This is the
    only thing the prepared-context mechanism claims to buy, so the stub models
    it instead of pretending every switch is instant."""

    stub_prepare_ms: float = 40.0
    stub_prepared_capacity: int = 3
    """Prepared topic contexts held at once. Smaller than the stub briefing's
    section count on purpose, so eviction and a cold switch are reachable."""

    # --- serving ----------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 30810

    def chat_url(self) -> str:
        return f"{self.runtime_base_url.rstrip('/')}/v1/chat/completions"

    def realtime_url(self) -> str:
        base = self.runtime_base_url.rstrip("/")
        if base.startswith("https://"):
            return "wss://" + base[len("https://") :] + "/v1/realtime"
        if base.startswith("http://"):
            return "ws://" + base[len("http://") :] + "/v1/realtime"
        return base + "/v1/realtime"
