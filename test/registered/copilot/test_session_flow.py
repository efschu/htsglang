"""Full conversation round trip against desk fakes, plus the scheduling shape.

    CUDA_VISIBLE_DEVICES=99 PYTHONPATH=<worktree>/python \
        python -m pytest test/registered/copilot/test_session_flow.py -v

Covers the P1 falsifier: two tracks in, attributed transcript lines out, hint
frames out, briefing extended by the background expander -- and the two
scheduling facts the design rests on (live hints carry ``lane="fast"``,
background expansion does not).
"""

import asyncio

from sglang.srt.copilot.briefing import parse_briefing
from sglang.srt.copilot.config import CopilotConfig
from sglang.srt.copilot.hints import DeskFakeHints, RequestKind
from sglang.srt.copilot.protocol import (
    ServerFrame,
    Track,
    encode_audio_frame,
    decode_audio_frame,
)
from sglang.srt.copilot.session import CopilotSession, SessionManager

BRIEFING = """# Client call

## Contract renewal
Ends in March.

## Migration timeline
Two clusters move in Q3.
"""


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def build(config=None, always_warm=True):
    config = config or CopilotConfig()
    clock = Clock()
    backend = DeskFakeHints(config=config, always_warm=always_warm)
    session = CopilotSession(
        session_id="test",
        config=config,
        hint_backend=backend,
        briefing=parse_briefing(BRIEFING),
        clock=clock,
    )
    return session, backend, clock


def frame(track: Track, seq: int = 0) -> bytes:
    return encode_audio_frame(track, bytes(640), seq)


class TestTrackAttribution:
    def test_two_tracks_produce_attributed_lines(self):
        session, _, _ = build()
        session.open_track(Track.SELF)
        session.open_track(Track.OTHER)

        async def run():
            await session.on_audio(decode_audio_frame(frame(Track.SELF)))
            await session.on_commit(Track.SELF)
            await session.on_audio(decode_audio_frame(frame(Track.OTHER)))
            await session.on_commit(Track.OTHER)

        asyncio.run(run())
        tracks = [ln.track for ln in session.transcript]
        assert tracks == [Track.SELF, Track.OTHER]

    def test_audio_for_an_unopened_track_is_dropped_not_reattributed(self):
        session, _, _ = build()
        session.open_track(Track.SELF)

        async def run():
            return await session.on_audio(decode_audio_frame(frame(Track.OTHER)))

        events = asyncio.run(run())
        assert session.dropped_frames == 1
        assert events[0].kind is ServerFrame.ERROR
        assert session.transcript == []


class TestAsrRefusalsDoNotKillTheConversation:
    """Regression: an empty commit used to raise out of the session.

    Found while executing the can-fail proof for the audio header -- the
    injected fault made a commit land on an empty buffer, the ``AsrError``
    propagated through the WebSocket handler and tore down the whole
    connection. One track's refusal must never end the conversation.
    """

    def test_empty_commit_becomes_an_error_frame(self):
        session, _, _ = build()
        session.open_track(Track.SELF)
        events = asyncio.run(session.on_commit(Track.SELF))
        assert len(events) == 1
        assert events[0].kind is ServerFrame.ERROR
        assert events[0].payload["stage"] == "asr"
        assert events[0].payload["track"] == "self"

    def test_the_other_track_still_works_afterwards(self):
        session, _, _ = build()
        session.open_track(Track.SELF)
        session.open_track(Track.OTHER)

        async def run():
            await session.on_commit(Track.SELF)  # refused, empty
            await session.on_audio(decode_audio_frame(frame(Track.OTHER)))
            return await session.on_commit(Track.OTHER)

        events = asyncio.run(run())
        assert any(e.kind is ServerFrame.TRANSCRIPT_LINE for e in events)


class TestHints:
    def test_a_final_line_produces_a_hint(self):
        session, backend, _ = build()
        session.open_track(Track.SELF)

        async def run():
            await session.on_audio(decode_audio_frame(frame(Track.SELF)))
            return await session.on_commit(Track.SELF)

        events = asyncio.run(run())
        kinds = [e.kind for e in events]
        assert ServerFrame.TRANSCRIPT_LINE in kinds
        assert ServerFrame.HINT in kinds
        hint = [e for e in events if e.kind is ServerFrame.HINT][0]
        assert hint.payload["bullets"]
        assert hint.payload["desk_fake"] is True

    def test_live_hints_ride_the_fast_lane(self):
        session, backend, _ = build()
        session.open_track(Track.SELF)

        async def run():
            await session.on_audio(decode_audio_frame(frame(Track.SELF)))
            await session.on_commit(Track.SELF)

        asyncio.run(run())
        hint_calls = [c for c in backend.calls if c.kind is RequestKind.HINT]
        assert hint_calls
        assert hint_calls[0].lane == "fast"
        assert hint_calls[0].priority == session.config.hint_priority

    def test_hint_rate_is_floored(self):
        config = CopilotConfig(min_hint_interval_s=5.0)
        session, backend, clock = build(config)
        session.open_track(Track.SELF)

        async def utterance():
            await session.on_audio(decode_audio_frame(frame(Track.SELF)))
            await session.on_commit(Track.SELF)

        asyncio.run(utterance())
        clock.advance(1.0)
        asyncio.run(utterance())
        assert len([c for c in backend.calls if c.kind is RequestKind.HINT]) == 1
        clock.advance(10.0)
        asyncio.run(utterance())
        assert len([c for c in backend.calls if c.kind is RequestKind.HINT]) == 2

    def test_hint_prompt_starts_with_the_topic_prefix_verbatim(self):
        """PREFIX DISCIPLINE: any reformatting is a radix miss."""
        session, backend, _ = build()
        session.open_track(Track.SELF)

        async def run():
            await session.on_audio(decode_audio_frame(frame(Track.SELF)))
            await session.on_commit(Track.SELF)

        asyncio.run(run())
        hint = [c for c in backend.calls if c.kind is RequestKind.HINT][0]
        prefix = session.topics.focused().prefix_text
        assert hint.prompt.startswith(prefix)


class TestPriming:
    def test_priming_records_tokens_without_claiming_warmth(self):
        session, backend, _ = build()
        asyncio.run(session.prime_due_topics())
        primes = [c for c in backend.calls if c.kind is RequestKind.PRIME]
        assert len(primes) == 2
        assert all(c.max_tokens == 0 for c in primes)
        assert all(c.lane is None for c in primes)
        for topic in session.topics.topics.values():
            assert topic.primed_tokens > 0
            assert topic.warmth.value == "unknown"

    def test_a_hint_after_priming_measures_warmth(self):
        session, _, _ = build()
        session.open_track(Track.SELF)

        async def run():
            await session.prime_due_topics()
            await session.on_audio(decode_audio_frame(frame(Track.SELF)))
            return await session.on_commit(Track.SELF)

        events = asyncio.run(run())
        hint = [e for e in events if e.kind is ServerFrame.HINT][0]
        assert hint.payload["warmth"] in ("warm", "partial", "cold")
        assert hint.payload["cached_tokens"] is not None

    def test_a_cold_runtime_is_reported_as_cold(self):
        """Can-fail proof for the probe wired into the session, not just the
        registry: with the pessimistic fake the same code path reports COLD."""
        session, _, _ = build(always_warm=False)
        session.open_track(Track.SELF)

        async def run():
            await session.prime_due_topics()
            await session.on_audio(decode_audio_frame(frame(Track.SELF)))
            return await session.on_commit(Track.SELF)

        events = asyncio.run(run())
        hint = [e for e in events if e.kind is ServerFrame.HINT][0]
        assert hint.payload["warmth"] == "cold"
        assert session.topics.miss_report()["misses"] == 1


class TestExpander:
    def test_expansion_appends_and_does_not_use_the_fast_lane(self):
        session, backend, _ = build()
        session.open_track(Track.SELF)

        async def run():
            await session.on_audio(decode_audio_frame(frame(Track.SELF)))
            await session.on_commit(Track.SELF)
            return await session.run_expansion()

        event = asyncio.run(run())
        assert event.kind is ServerFrame.BRIEFING_UPDATE
        assert event.payload["reason"] == "expanded"
        expansions = [c for c in backend.calls if c.kind is RequestKind.EXPANSION]
        assert expansions
        assert expansions[0].lane is None
        assert expansions[0].priority == session.config.expander_priority
        assert len(session.briefing.generated_sections) == 1

    def test_expansion_never_rewrites_user_text(self):
        session, _, _ = build()
        before = session.briefing.section("contract-renewal").body
        session.open_track(Track.SELF)

        async def run():
            await session.on_audio(decode_audio_frame(frame(Track.SELF)))
            await session.on_commit(Track.SELF)
            await session.run_expansion()

        asyncio.run(run())
        assert session.briefing.section("contract-renewal").body == before

    def test_expansion_cadence(self):
        config = CopilotConfig(expander_interval_s=60.0)
        session, _, clock = build(config)
        assert session.expansion_due() is True
        session._last_expansion_at = clock()
        assert session.expansion_due() is False
        clock.advance(61.0)
        assert session.expansion_due() is True


class TestSessionManager:
    def test_idle_sessions_are_collected(self):
        config = CopilotConfig(session_idle_timeout_s=10.0)
        clock = Clock()
        manager = SessionManager(
            config=config, hint_backend=DeskFakeHints(config=config), clock=clock
        )
        session = manager.open()
        assert manager.open(session.session_id) is session
        clock.advance(11.0)
        assert manager.collect() == [session.session_id]

    def test_the_session_limit_is_enforced(self):
        config = CopilotConfig(max_sessions=1)
        manager = SessionManager(
            config=config, hint_backend=DeskFakeHints(config=config)
        )
        manager.open()
        try:
            manager.open()
        except RuntimeError as exc:
            assert "session limit reached" in str(exc)
        else:
            raise AssertionError("the session limit did not fire")

    def test_briefings_are_not_shared_between_sessions(self):
        """One conversation's generated sections must not leak into another."""
        config = CopilotConfig()
        manager = SessionManager(
            config=config,
            hint_backend=DeskFakeHints(config=config),
            default_briefing=parse_briefing(BRIEFING),
        )
        a = manager.open()
        b = manager.open()
        a.briefing.append_generated("Leak check", "only in a", provenance="copilot")
        assert a.briefing.section("leak-check") is not None
        assert b.briefing.section("leak-check") is None
