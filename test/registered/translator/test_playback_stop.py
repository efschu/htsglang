# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Stop must actually stop: the queue, the talker, and the wire (§19.12).

A stop button that only silences the local speaker is a lie in three ways at
once, and each is visible to the user:

* the queued turns still run, so the translator starts talking again by
  itself a few seconds later;
* the in-flight synthesis keeps the talker, which is the system's capacity
  limit (measured RTF 1.23, §17.8.2) -- so "stop" would make the NEXT turn
  slower rather than faster;
* audio already on the wire arrives after the stop and plays over whatever
  comes next.

The protocol answer is one frame in, one ack out, plus a journal event so the
written record shows WHY a line stops mid-sentence -- a turn that simply ends
looks like a defect. The ack names the abandoned turn because the client
discards audio per ``turn_id``.

    CUDA_VISIBLE_DEVICES=99 python -m pytest \\
      test/registered/translator/test_playback_stop.py -v
"""

import asyncio
import unittest

from sglang.srt.translator.session import EventKind
from test_session import (  # noqa: E402  - sibling helper module
    VOICE_A_HZ,
    conversation_audio,
    make_session,
)


def one_segment(session, audio):
    segments = session.push_audio(audio)
    assert segments, "the test audio must close at least one segment"
    return segments[0]


class SlowTts:
    """A talker that holds its lock, so a stop has something to interrupt.

    It mirrors the real one where it matters: one lock for every session, and
    synthesis is an async generator, so cancelling the awaiting task is what
    releases the lock.
    """

    def __init__(self, inner):
        self._inner = inner
        self._lock = asyncio.Lock()
        self.started = asyncio.Event()
        self.finished = 0

    def __getattr__(self, name):
        # Everything this wrapper does not deliberately change comes from the
        # real fake: `sample_rate`, the language table, the reference rules.
        # Redeclaring them here would let the wrapper drift from the backend
        # it is standing in for.
        return getattr(self._inner, name)

    @property
    def busy(self):
        return self._lock.locked()

    async def synthesize(self, text, language, reference, reference_text, voice_id):
        async with self._lock:
            self.started.set()
            async for piece in self._inner.synthesize(
                text, language, reference, reference_text, voice_id
            ):
                yield piece
                # Long enough that a stop lands mid-synthesis rather than
                # racing the end of it.
                await asyncio.sleep(0.25)
            self.finished += 1


class TestTheQueueIsAbandoned(unittest.IsolatedAsyncioTestCase):
    async def test_a_stop_drops_the_queued_segments(self):
        session, _asr, _mt, _tts = make_session()
        audio = conversation_audio((VOICE_A_HZ, 1.2))
        for segment in session.push_audio(audio):
            session.enqueue(segment)
        self.assertGreater(session.pending(), 0, "need something queued to drop")
        outcome = session.abort_playback()
        self.assertEqual(session.pending(), 0)
        self.assertGreaterEqual(outcome["dropped_queued"], 1)

    async def test_the_stop_is_journalled_with_what_it_abandoned(self):
        """The record must explain a line that ends mid-sentence."""
        session, _asr, _mt, _tts = make_session()
        session.abort_playback(reason="user")
        events = [
            e for e in session.journal.since(0)[0]
            if e.kind is EventKind.PLAYBACK_STOPPED
        ]
        self.assertEqual(len(events), 1)
        payload = events[0].payload
        self.assertEqual(payload["reason"], "user")
        self.assertEqual(payload["dropped_queued"], 0)
        self.assertIn("aborted_turn_id", payload)
        self.assertEqual(payload["stop_epoch"], 1)

    async def test_the_epoch_advances_per_stop(self):
        session, _asr, _mt, _tts = make_session()
        self.assertEqual(session.abort_playback()["stop_epoch"], 1)
        self.assertEqual(session.abort_playback()["stop_epoch"], 2)


class TestTheTalkerIsFreedImmediately(unittest.IsolatedAsyncioTestCase):
    async def test_cancelling_the_turn_releases_the_talker(self):
        """The falsifier for "stop while speaking": the lock must come back."""
        session, _asr, _mt, inner = make_session()
        talker = SlowTts(inner)
        session.tts = talker
        segment = one_segment(session, conversation_audio((VOICE_A_HZ, 1.5)))

        task = asyncio.create_task(session.run_turn_multi(segment))
        await asyncio.wait_for(talker.started.wait(), timeout=10.0)
        self.assertTrue(talker.busy, "precondition: the talker is held")

        outcome = session.abort_playback()
        self.assertIsNotNone(
            outcome["aborted_turn_id"],
            "a stop during a turn must NAME the turn it killed; the client "
            "discards audio by turn_id and cannot act on None",
        )
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertFalse(
            talker.busy,
            "the talker is still held after the stop -- the next turn would "
            "queue behind a synthesis nobody is listening to",
        )
        self.assertEqual(talker.finished, 0, "the synthesis must not complete")

    async def test_the_next_turn_starts_cleanly_after_a_stop(self):
        """Stop must not poison the session: speaking again has to work."""
        session, _asr, _mt, inner = make_session()
        talker = SlowTts(inner)
        session.tts = talker
        segment = one_segment(session, conversation_audio((VOICE_A_HZ, 1.5)))

        task = asyncio.create_task(session.run_turn_multi(segment))
        await asyncio.wait_for(talker.started.wait(), timeout=10.0)
        session.abort_playback()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertIsNone(
            session._active_turn_id,
            "a stale active turn would make the NEXT stop name a dead turn",
        )
        # And the pipeline runs again, on the real talker this time.
        session.tts = inner
        results = await session.run_turn_multi(
            one_segment(session, conversation_audio((VOICE_A_HZ, 1.5)))
        )
        self.assertTrue(results, "the session must still translate after a stop")


if __name__ == "__main__":
    unittest.main()
