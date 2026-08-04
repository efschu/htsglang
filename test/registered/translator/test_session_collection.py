# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Abandoned sessions must not accumulate — measured, not hoped at.

This is a degradation class rather than a crash, which is why it survived so
long: the live service answered every probe correctly while first-audio
latency went from 4 s to **52 s** with six of eight session slots held by
clients that were long gone. In Spain — one phone, a flaky tunnel, reconnects
all day — that is the daily case, not the edge case.

Two deadlines, because two different things go wrong:

* a session with a client ATTACHED is alive however long the conversation
  pauses, and is only collected after ``idle_timeout_s``;
* a session whose socket DIED is collected after ``resume_grace_s`` — the
  window in which a reconnect can still resume it by token.

Sticky resume must survive inside that window, so the tests below check both
directions: a reconnect inside the grace gets the SAME session with its
speakers and transcript, and one after it does not.

Time is injected. A test that slept for a real grace period would either be
useless (grace too short to mean anything) or unbearable.

    CUDA_VISIBLE_DEVICES=99 python -m pytest \\
      test/registered/translator/test_session_collection.py -v
"""

import unittest

from sglang.srt.translator.languages import ConversationLanguages
from sglang.srt.translator.session import SessionManager
from test_session import LANG_A, LANG_B, make_session


class Clock:
    """Injected time. The whole point is not to sleep."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def manager(clock, max_sessions=3, idle_timeout_s=300.0, resume_grace_s=120.0):
    def factory(session_id, conversation):
        session, _asr, _mt, _tts = make_session(session_id=session_id)
        session._clock = clock
        session.last_activity = clock()
        return session

    return SessionManager(
        factory,
        max_sessions=max_sessions,
        idle_timeout_s=idle_timeout_s,
        resume_grace_s=resume_grace_s,
        clock=clock,
    )


class TestCollection(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.mgr = manager(self.clock)
        self.conv = ConversationLanguages.of([LANG_A, LANG_B])

    def test_capacity_comes_back_after_clients_die(self):
        """The falsifier for the leak, end to end.

        Fill every slot, let all of them lose their socket, and assert the
        capacity is measurably back -- not that a collector exists.
        """
        for _ in range(3):
            self.mgr.open(self.conv).attach()
        with self.assertRaises(RuntimeError):
            self.mgr.open(self.conv)          # full, as designed

        for sid in self.mgr.ids():
            self.mgr.get(sid).detach()
        self.clock.advance(121.0)

        collected = self.mgr.collect()
        self.assertEqual(len(collected), 3)
        self.assertEqual(len(self.mgr), 0)
        # And the capacity is usable, which is the thing that actually failed.
        fresh = self.mgr.open(self.conv)
        self.assertIsNotNone(fresh)

    def test_a_reconnect_inside_the_grace_resumes_the_same_session(self):
        session = self.mgr.open(self.conv)
        session.attach()
        sid = session.session_id
        session.transcript.append(
            turn_id="t1", speaker_id="speaker-1", speaker_label="speaker-1",
            source_language=LANG_A, source_text="hello",
        )
        session.detach()

        self.clock.advance(60.0)              # inside the 120 s grace
        self.assertEqual(self.mgr.collect(), [])
        resumed = self.mgr.open(self.conv, session_id=sid)
        self.assertIs(resumed, session)
        # Sticky state survived: this is what the grace exists to protect.
        self.assertEqual(len(resumed.transcript), 1)
        self.assertEqual(resumed.transcript.lines()[0].source_text, "hello")

    def test_after_the_grace_the_session_is_gone(self):
        session = self.mgr.open(self.conv)
        session.attach()
        sid = session.session_id
        session.detach()
        self.clock.advance(121.0)
        self.assertIn(sid, self.mgr.collect())
        # A later client with the old token gets a NEW session rather than an
        # error -- the conversation is lost, the app is not.
        fresh = self.mgr.open(self.conv, session_id=sid)
        self.assertEqual(len(fresh.transcript), 0)

    def test_an_attached_session_is_never_collected_for_being_quiet(self):
        """A conversation that pauses is not an abandoned one."""
        session = self.mgr.open(self.conv)
        session.attach()
        self.clock.advance(200.0)             # well past the resume grace
        self.assertEqual(self.mgr.collect(), [])
        self.assertEqual(len(self.mgr), 1)
        # ...but the idle timeout still applies, so a forgotten tab does go.
        self.clock.advance(200.0)
        self.assertEqual(len(self.mgr.collect()), 1)

    def test_reattaching_clears_the_detach_deadline(self):
        session = self.mgr.open(self.conv)
        session.attach()
        session.detach()
        self.clock.advance(100.0)
        session.attach()
        self.assertIsNone(session.detached_seconds())
        self.clock.advance(100.0)             # 200 s since the first detach
        self.assertEqual(self.mgr.collect(), [])

    def test_a_second_client_holds_the_session_open(self):
        session = self.mgr.open(self.conv)
        session.attach()
        session.attach()
        session.detach()
        self.assertEqual(session.attached, 1)
        self.assertIsNone(session.detached_seconds())
        self.clock.advance(200.0)
        self.assertEqual(self.mgr.collect(), [])

    def test_detaching_drops_work_nobody_is_listening_to(self):
        """The mechanism behind the 52 s.

        Queued turns for a client that left keep the synthesizer busy for
        nobody, and every other session pays for it.
        """
        session = self.mgr.open(self.conv)
        session.attach()
        from sglang.srt.translator.segmenter import Segment, SegmentReason
        from test_session import RATE, tone
        from sglang.srt.translator.backends import AudioChunk

        for i in range(2):
            session.enqueue(
                Segment(
                    index=i,
                    audio=AudioChunk(tone(220.0, 1.0), RATE),
                    reason=SegmentReason.RELEASED,
                    start_s=0.0,
                )
            )
        self.assertGreater(session.pending(), 0)
        session.detach()
        self.assertEqual(session.pending(), 0)

    def test_the_state_is_visible_before_it_is_felt(self):
        session = self.mgr.open(self.conv)
        session.attach()
        detail = self.mgr.to_json()
        self.assertEqual(len(detail), 1)
        self.assertEqual(detail[0]["attached"], 1)
        self.assertIsNone(detail[0]["detached_s"])
        session.detach()
        self.clock.advance(30.0)
        detail = self.mgr.to_json()
        self.assertEqual(detail[0]["attached"], 0)
        self.assertEqual(detail[0]["detached_s"], 30.0)


if __name__ == "__main__":
    unittest.main()
