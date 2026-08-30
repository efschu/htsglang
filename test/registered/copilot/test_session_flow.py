"""Full conversation round trip against desk fakes, plus the scheduling shape.

    CUDA_VISIBLE_DEVICES=99 PYTHONPATH=<worktree>/python \
        python -m pytest test/registered/copilot/test_session_flow.py -v

Covers the P1 falsifier: two tracks in, attributed transcript lines out, hint
frames out, briefing extended by the background expander -- and the two
scheduling facts the design rests on (live hints carry ``lane="fast"``,
background expansion does not).

The session is PUSH-shaped: nothing here returns the events it caused. They are
journalled and fanned out, so every assertion reads the journal or a
subscription, which is exactly what a reconnecting browser reads.
"""

import asyncio

from sglang.srt.copilot.briefing import parse_briefing
from sglang.srt.copilot.config import CopilotConfig
from sglang.srt.copilot.deskfakes import DeskFakeHints, desk_fake_backend_set
from sglang.srt.copilot.hints import RequestKind
from sglang.srt.copilot.protocol import (
    ServerFrame,
    Track,
    decode_audio_frame,
    encode_audio_frame,
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
    hints = DeskFakeHints(config=config, always_warm=always_warm)
    backends = desk_fake_backend_set(config, hints=hints)
    session = CopilotSession(
        session_id="test",
        config=config,
        backends=backends,
        briefing=parse_briefing(BRIEFING),
        clock=clock,
    )
    return session, hints, clock


def frame(track: Track, seq: int = 0) -> bytes:
    return encode_audio_frame(track, bytes(640), seq)


def journal_kinds(session, since=0):
    return [e.kind for e in session.journal.since(since)]


def journal_of(session, kind, since=0):
    return [e for e in session.journal.since(since) if e.kind is kind]


async def utterance(session, track: Track):
    """One complete utterance on one track, hints settled."""
    await session.on_audio(decode_audio_frame(frame(track)))
    await session.on_commit(track)
    await session.settle()


class TestTrackAttribution:
    def test_two_tracks_produce_attributed_lines(self):
        session, _, _ = build()

        async def run():
            await session.open_track(Track.SELF)
            await session.open_track(Track.OTHER)
            await utterance(session, Track.SELF)
            await utterance(session, Track.OTHER)

        asyncio.run(run())
        tracks = [ln.track for ln in session.transcript]
        assert tracks == [Track.SELF, Track.OTHER]

    def test_audio_for_an_unopened_track_is_dropped_not_reattributed(self):
        session, _, _ = build()

        async def run():
            await session.open_track(Track.SELF)
            await session.on_audio(decode_audio_frame(frame(Track.OTHER)))

        asyncio.run(run())
        assert session.dropped_frames == 1
        errors = journal_of(session, ServerFrame.ERROR)
        assert errors and errors[-1].payload["stage"] == "audio"
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

        async def run():
            await session.open_track(Track.SELF)
            await session.on_commit(Track.SELF)

        asyncio.run(run())
        errors = journal_of(session, ServerFrame.ERROR)
        assert len(errors) == 1
        assert errors[0].payload["stage"] == "asr"
        assert errors[0].payload["track"] == "self"

    def test_the_other_track_still_works_afterwards(self):
        session, _, _ = build()

        async def run():
            await session.open_track(Track.SELF)
            await session.open_track(Track.OTHER)
            await session.on_commit(Track.SELF)  # refused, empty
            await utterance(session, Track.OTHER)

        asyncio.run(run())
        lines = journal_of(session, ServerFrame.TRANSCRIPT_LINE)
        assert len(lines) == 1
        assert lines[0].payload["track"] == "other"


class TestHints:
    def test_a_final_line_produces_a_hint(self):
        session, _, _ = build()

        async def run():
            await session.open_track(Track.SELF)
            await utterance(session, Track.SELF)

        asyncio.run(run())
        kinds = journal_kinds(session)
        assert ServerFrame.TRANSCRIPT_LINE in kinds
        assert ServerFrame.HINT_PENDING in kinds
        assert ServerFrame.HINT in kinds
        hint = journal_of(session, ServerFrame.HINT)[0]
        assert hint.payload["bullets"]
        assert hint.payload["desk_fake"] is True

    def test_a_hint_names_the_transcript_line_it_answers(self):
        """The latency instrument is only honest if t0 is identified."""
        session, _, _ = build()

        async def run():
            await session.open_track(Track.SELF)
            await utterance(session, Track.SELF)

        asyncio.run(run())
        line = journal_of(session, ServerFrame.TRANSCRIPT_LINE)[0]
        hint = journal_of(session, ServerFrame.HINT)[0]
        assert hint.payload["source_kind"] == "line"
        assert hint.payload["source_line_id"] == line.payload["line_id"]
        assert hint.payload["pipeline_ms"] is not None

    def test_a_pending_frame_precedes_the_hint(self):
        """A read pane must be able to tell thinking from broken."""
        session, _, _ = build()

        async def run():
            await session.open_track(Track.SELF)
            await utterance(session, Track.SELF)

        asyncio.run(run())
        kinds = journal_kinds(session)
        assert kinds.index(ServerFrame.HINT_PENDING) < kinds.index(ServerFrame.HINT)

    def test_live_hints_ride_the_fast_lane(self):
        session, hints, _ = build()

        async def run():
            await session.open_track(Track.SELF)
            await utterance(session, Track.SELF)

        asyncio.run(run())
        hint_calls = [c for c in hints.calls if c.kind is RequestKind.HINT]
        assert hint_calls
        assert hint_calls[0].lane == "fast"
        assert hint_calls[0].priority == session.config.hint_priority

    def test_hint_rate_is_floored(self):
        config = CopilotConfig(min_hint_interval_s=5.0)
        session, hints, clock = build(config)

        async def run():
            await session.open_track(Track.SELF)
            await utterance(session, Track.SELF)
            clock.advance(1.0)
            await utterance(session, Track.SELF)
            first = len([c for c in hints.calls if c.kind is RequestKind.HINT])
            clock.advance(10.0)
            await utterance(session, Track.SELF)
            second = len([c for c in hints.calls if c.kind is RequestKind.HINT])
            return first, second

        first, second = asyncio.run(run())
        assert first == 1
        assert second == 2

    def test_only_one_decode_is_in_flight_at_a_time(self):
        """Stacked decodes would spend the fast lane on stale answers."""
        config = CopilotConfig(min_hint_interval_s=0.0)
        session, hints, _ = build(config)

        async def run():
            await session.open_track(Track.SELF)
            await session.on_audio(decode_audio_frame(frame(Track.SELF)))
            # Three commits back to back, each producing a distinct final line.
            await session.on_commit(Track.SELF)
            await session.on_audio(decode_audio_frame(frame(Track.SELF)))
            await session.on_commit(Track.SELF)
            await session.on_audio(decode_audio_frame(frame(Track.SELF)))
            await session.on_commit(Track.SELF)
            pending_before_settle = len(journal_of(session, ServerFrame.HINT_PENDING))
            await session.settle()
            return pending_before_settle

        pending = asyncio.run(run())
        assert pending == 1
        assert len(journal_of(session, ServerFrame.HINT)) == 1

    def test_a_failing_hint_backend_is_reported_not_swallowed(self):
        session, hints, _ = build()
        hints.fail = True

        async def run():
            await session.open_track(Track.SELF)
            await utterance(session, Track.SELF)

        asyncio.run(run())
        errors = [
            e
            for e in journal_of(session, ServerFrame.ERROR)
            if e.payload["stage"] == "hint"
        ]
        assert len(errors) == 1
        assert errors[0].payload["degraded"] is True
        assert session.hint_failures == 1
        assert journal_of(session, ServerFrame.HINT) == []

    def test_hint_prompt_starts_with_the_topic_prefix_verbatim(self):
        """PREFIX DISCIPLINE: any reformatting is a radix miss."""
        session, hints, _ = build()

        async def run():
            await session.open_track(Track.SELF)
            await utterance(session, Track.SELF)

        asyncio.run(run())
        hint = [c for c in hints.calls if c.kind is RequestKind.HINT][0]
        prefix = session.topics.focused().prefix_text
        assert hint.prompt.startswith(prefix)


class TestTopicSwitch:
    def test_focus_switches_the_context_and_answers_with_prepared_state(self):
        session, hints, _ = build()

        async def run():
            await session.open_track(Track.SELF)
            await session.prime_due_topics()
            await utterance(session, Track.SELF)
            mark = session.journal.next_seq
            await session.focus_topic("migration-timeline")
            await session.settle()
            return mark

        mark = asyncio.run(run())
        focus = journal_of(session, ServerFrame.TOPIC_STATE, mark)[0]
        assert focus.payload["reason"] == "focus"
        assert focus.payload["topic_id"] == "migration-timeline"
        assert focus.payload["prepared"] is True
        hint = journal_of(session, ServerFrame.HINT, mark)[0]
        assert hint.payload["topic_id"] == "migration-timeline"

    def test_a_switch_clears_the_hint_floor(self):
        """A stale card for the previous topic reads as a broken app."""
        config = CopilotConfig(min_hint_interval_s=1e9)
        session, _, _ = build(config)

        async def run():
            await session.open_track(Track.SELF)
            await utterance(session, Track.SELF)
            mark = session.journal.next_seq
            await session.focus_topic("migration-timeline")
            await session.settle()
            return mark

        mark = asyncio.run(run())
        assert journal_of(session, ServerFrame.HINT, mark)


class TestPriming:
    def test_priming_records_tokens_without_claiming_warmth(self):
        session, hints, _ = build()
        asyncio.run(session.prime_due_topics())
        assert session.backends.prep.prepares == 2
        for topic in session.topics.topics.values():
            assert topic.primed_tokens > 0
            assert topic.warmth.value == "unknown"

    def test_a_hint_after_priming_measures_warmth(self):
        session, _, _ = build()

        async def run():
            await session.open_track(Track.SELF)
            await session.prime_due_topics()
            await utterance(session, Track.SELF)

        asyncio.run(run())
        hint = journal_of(session, ServerFrame.HINT)[0]
        assert hint.payload["warmth"] in ("warm", "partial", "cold")
        assert hint.payload["cached_tokens"] is not None

    def test_a_cold_runtime_is_reported_as_cold(self):
        """Can-fail proof for the probe wired into the session, not just the
        registry: with the pessimistic fake the same code path reports COLD."""
        session, _, _ = build(always_warm=False)

        async def run():
            await session.open_track(Track.SELF)
            await session.prime_due_topics()
            await utterance(session, Track.SELF)

        asyncio.run(run())
        hint = journal_of(session, ServerFrame.HINT)[0]
        assert hint.payload["warmth"] == "cold"
        assert session.topics.miss_report()["misses"] == 1

    def test_an_unenumerable_prep_backend_is_not_guessed_at(self):
        """``DeskFakePrep`` omits ``held``, modelling the rig's blindness.

        The session must then report NO prep reconciliation at all rather than
        inventing one -- an eviction it cannot see must be discovered by the
        warmth probe, not assumed away.
        """
        session, _, _ = build()
        asyncio.run(session.prime_due_topics())
        assert "held" not in session.backends.prep.report()
        assert [
            e
            for e in journal_of(session, ServerFrame.TOPIC_STATE)
            if e.payload.get("reason") == "prep"
        ] == []


class TestExpander:
    def test_expansion_appends_and_does_not_use_the_fast_lane(self):
        session, hints, _ = build()

        async def run():
            await session.open_track(Track.SELF)
            await utterance(session, Track.SELF)
            return await session.run_expansion()

        event = asyncio.run(run())
        assert event.kind is ServerFrame.BRIEFING_UPDATE
        assert event.payload["reason"] == "expanded"
        expansions = [c for c in hints.calls if c.kind is RequestKind.EXPANSION]
        assert expansions
        assert expansions[0].lane is None
        assert expansions[0].priority == session.config.expander_priority
        assert len(session.briefing.generated_sections) == 1

    def test_expansion_never_rewrites_user_text(self):
        session, _, _ = build()
        before = session.briefing.section("contract-renewal").body

        async def run():
            await session.open_track(Track.SELF)
            await utterance(session, Track.SELF)
            await session.run_expansion()

        asyncio.run(run())
        assert session.briefing.section("contract-renewal").body == before

    def test_the_background_loop_expands_without_being_asked(self):
        """Acceptance: extension events appear DURING a session.

        Nothing in this test calls ``run_expansion``; the session's own task
        does, which is the only version of this feature a user ever sees.
        """
        config = CopilotConfig(expander_first_delay_s=0.01, expander_interval_s=0.01)
        session, _, _ = build(config)

        async def run():
            await session.open_track(Track.SELF)
            await utterance(session, Track.SELF)
            session.start_background()
            for _ in range(200):
                await asyncio.sleep(0.005)
                if session.expansion_count:
                    break
            await session.aclose()

        asyncio.run(run())
        assert session.expansion_count >= 1
        updates = [
            e
            for e in journal_of(session, ServerFrame.BRIEFING_UPDATE)
            if e.payload.get("reason") == "expanded"
        ]
        assert updates
        assert updates[0].payload["added_title"]

    def test_expansion_cadence(self):
        config = CopilotConfig(expander_interval_s=60.0)
        session, _, clock = build(config)
        assert session.expansion_due() is True
        session._last_expansion_at = clock()
        assert session.expansion_due() is False
        clock.advance(61.0)
        assert session.expansion_due() is True


class TestSubscriptions:
    def test_a_subscriber_receives_what_the_session_emits(self):
        session, _, _ = build()

        async def run():
            sub = session.subscribe()
            await session.open_track(Track.SELF)
            await utterance(session, Track.SELF)
            drained = []
            while not sub.queue.empty():
                drained.append(sub.queue.get_nowait())
            return drained

        drained = asyncio.run(run())
        kinds = [e.kind for e in drained]
        assert ServerFrame.TRACK_STATE in kinds
        assert ServerFrame.TRANSCRIPT_LINE in kinds
        assert ServerFrame.HINT in kinds

    def test_an_overflowing_subscriber_is_marked_not_thinned(self):
        """The queue must never quietly drop an event and carry on."""
        config = CopilotConfig(subscriber_queue_max=2)
        session, _, _ = build(config)

        async def run():
            sub = session.subscribe()
            for _ in range(5):
                session.emit(ServerFrame.SESSION_STATE, {})
            return sub

        sub = asyncio.run(run())
        assert sub.overflowed is True
        assert sub.queue.qsize() == 2
        # The journal keeps everything, which is what makes the reconnect a
        # complete recovery rather than a partial one.
        assert len(session.journal.since(0)) == 5


class TestSessionManager:
    def test_idle_sessions_are_collected(self):
        config = CopilotConfig(session_idle_timeout_s=10.0)
        clock = Clock()
        manager = SessionManager(
            config=config, backends=desk_fake_backend_set(config), clock=clock
        )

        async def run():
            session = await manager.open()
            assert await manager.open(session.session_id) is session
            clock.advance(11.0)
            return session.session_id, await manager.collect()

        sid, dead = asyncio.run(run())
        assert dead == [sid]

    def test_the_session_limit_is_enforced(self):
        config = CopilotConfig(max_sessions=1)
        manager = SessionManager(config=config, backends=desk_fake_backend_set(config))

        async def run():
            await manager.open()
            try:
                await manager.open()
            except RuntimeError as exc:
                return str(exc)
            return None

        message = asyncio.run(run())
        assert message is not None and "session limit reached" in message

    def test_briefings_are_not_shared_between_sessions(self):
        """One conversation's generated sections must not leak into another."""
        config = CopilotConfig()
        manager = SessionManager(
            config=config,
            backends=desk_fake_backend_set(config),
            default_briefing=parse_briefing(BRIEFING),
        )

        async def run():
            return await manager.open(), await manager.open()

        a, b = asyncio.run(run())
        a.briefing.extend_generated("Leak check", "only in a", provenance="copilot")
        assert a.briefing.section("leak-check") is not None
        assert b.briefing.section("leak-check") is None
