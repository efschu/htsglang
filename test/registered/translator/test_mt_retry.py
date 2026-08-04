# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""A momentarily unreachable MT server must not cost the turn.

The field defect: the user speaks several sentences, the first is translated
and later ones silently produce nothing. Measured root
(`probe_continuous_listen.py`): recognition and segmentation are fine across
pauses -- every utterance is recognized and none is dropped -- but the MT hop
raises `httpx` transport errors whose ``str()`` is EMPTY, so the turn failed
with a message that named nothing.

Co-tenancy is the normal state of this rig: the MT server shares its cards
with the talker and with whatever else is running, so windows where it cannot
answer in time open and close on their own. A turn must ride those out.

What is asserted here, and why each one has teeth:

* a retryable failure followed by success produces a NORMAL turn -- the user
  never learns there was trouble, except that it took longer;
* the wait is ANNOUNCED. A turn that silently hangs is the complaint this
  came from, so the retry emits the frame the client already renders into
  the translation slot;
* a REFUSAL is not retried. Retrying a 400 delays the failure by the whole
  backoff budget and changes nothing;
* a stream that already delivered tokens is not restarted, because the user
  has read those clauses and the talker may already have spoken them;
* retrying stops as soon as newer speech is waiting, so a stale sentence
  never costs a fresh one.

    CUDA_VISIBLE_DEVICES=99 python -m pytest \\
      test/registered/translator/test_mt_retry.py -v
"""

import asyncio
import collections
import unittest

from sglang.srt.translator.backends import BackendError
from sglang.srt.translator.session import EventKind
from test_session import (  # noqa: E402  - sibling helper module
    VOICE_A_HZ,
    conversation_audio,
    make_session,
)

SENTENCE = "Una frase."


class FlakyMt:
    """Fails a given number of times, then answers.

    ``retryable`` and ``after_tokens`` are the two axes the policy turns on:
    whether the failure is worth another attempt at all, and whether anything
    had already been delivered when it happened.
    """

    def __init__(self, failures, retryable=True, after_tokens=0):
        self.name = "flaky-mt"
        self.failures = failures
        self.retryable = retryable
        self.after_tokens = after_tokens
        self.attempts = 0

    def supported_languages(self):
        return ("a", "b")

    async def translate(self, text, source, target, *, context=None):
        return SENTENCE

    async def translate_stream(self, text, source, target, *, context=None):
        self.attempts += 1
        if self.failures > 0:
            self.failures -= 1
            for _ in range(self.after_tokens):
                yield SENTENCE + " "
            raise BackendError(
                "mt", "stream failed: ReadTimeout: (no detail)",
                retryable=self.retryable,
            )
        for word in SENTENCE.split(" "):
            yield word + " "


def build(mt, **kwargs):
    session, _asr, _mt, _tts = make_session(**kwargs)
    session.mt = mt
    return session


async def run_turn(session):
    audio = conversation_audio((VOICE_A_HZ, 2.0))
    for segment in session.push_audio(audio):
        session.enqueue(segment)
    await session.drain()
    return list(session.journal.since(0)[0])


def kinds(events, kind):
    return [e for e in events if e.kind is kind]


class TestMtRetry(unittest.IsolatedAsyncioTestCase):
    async def test_a_transient_failure_costs_time_and_not_the_turn(self):
        mt = FlakyMt(failures=1)
        session = build(mt)
        # No real waiting: the backoff is the policy under test, not the
        # clock. Patching sleep keeps the test honest AND fast.
        original = asyncio.sleep
        asyncio.sleep = lambda _s: original(0)
        try:
            events = await run_turn(session)
        finally:
            asyncio.sleep = original
        self.assertEqual(mt.attempts, 2, "the stream was not retried")
        finals = [
            e for e in kinds(events, EventKind.TURN_TRANSLATION)
            if not e.payload.get("partial")
        ]
        self.assertTrue(finals, "the retry did not produce a translation")
        self.assertFalse(
            kinds(events, EventKind.ERROR),
            "a turn that recovered must not also report a failure",
        )

    async def test_the_wait_is_announced_rather_than_silent(self):
        mt = FlakyMt(failures=1)
        session = build(mt)
        original = asyncio.sleep
        asyncio.sleep = lambda _s: original(0)
        try:
            events = await run_turn(session)
        finally:
            asyncio.sleep = original
        queued = [
            e for e in kinds(events, EventKind.TURN_QUEUED)
            if e.payload.get("reason") == "mt_unreachable"
        ]
        self.assertEqual(len(queued), 1, "the retry was not announced")
        self.assertEqual(queued[0].payload["attempt"], 1)
        self.assertGreater(queued[0].payload["retry_in_s"], 0)

    async def test_a_refusal_is_not_retried(self):
        mt = FlakyMt(failures=1, retryable=False)
        session = build(mt)
        events = await run_turn(session)
        self.assertEqual(mt.attempts, 1, "a refusal must not be retried")
        self.assertTrue(
            kinds(events, EventKind.ERROR),
            "the turn must fail rather than hang",
        )

    async def test_a_stream_that_already_spoke_is_not_restarted(self):
        """Duplicated clauses would be worse than a failed turn."""
        mt = FlakyMt(failures=1, after_tokens=3)
        session = build(mt)
        events = await run_turn(session)
        self.assertEqual(
            mt.attempts, 1,
            "a stream that already delivered tokens was restarted -- the user "
            "would read and hear those clauses twice",
        )
        self.assertTrue(kinds(events, EventKind.ERROR))

    async def test_a_turn_with_newer_speech_behind_it_does_not_retry(self):
        """A stale sentence must never cost a fresh one.

        The property is per turn, not per run: a turn processed while
        something is still waiting gets ONE attempt, while the last turn --
        which has an empty queue behind it -- is entitled to the full budget.
        Asserting a total would fail on the second half for being correct.
        """
        mt = FlakyMt(failures=99)
        # Coalescing OFF, and that is the point of the setup rather than a
        # workaround: this arm needs TWO turns, one waiting behind the other,
        # and the whole purpose of the bundler is to make a waiting segment
        # stop being a separate turn. The retry policy it exercises is not
        # affected -- the guard reads "is anything still queued", which a
        # bundle answers the same way a second segment does.
        session = build(mt, coalesce_queued=False)
        audio = conversation_audio((VOICE_A_HZ, 2.0))
        segments = session.push_audio(audio)
        self.assertTrue(segments)
        for _ in range(session._max_queued):
            session.enqueue(segments[0])
        original = asyncio.sleep
        asyncio.sleep = lambda _s: original(0)
        try:
            await session.drain()
        finally:
            asyncio.sleep = original
        events = list(session.journal.since(0)[0])
        opened = [e.payload["turn_id"] for e in kinds(events, EventKind.TURN_OPENED)]
        self.assertGreaterEqual(len(opened), 2, "need a turn with one behind it")
        retried = collections.Counter(
            e.payload["turn_id"] for e in kinds(events, EventKind.TURN_QUEUED)
            if e.payload.get("reason") == "mt_unreachable"
        )
        self.assertEqual(
            retried[opened[0]], 0,
            "the first turn burned backoff while a newer segment waited",
        )
        self.assertGreater(
            retried[opened[-1]], 0,
            "the last turn had an empty queue and should have used the budget "
            "-- if it did not, this test proves nothing about the guard",
        )


if __name__ == "__main__":
    unittest.main()
