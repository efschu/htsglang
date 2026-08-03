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
