# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""One talker, many conversations: the wait must be measured and visible.

DESIGN §17.8.2. Measured on the live service, a turn that arrives while
another conversation is being synthesized pays that whole synthesis -- a
median 4.8 s on top of a 3.6 s baseline. That is not a bug to fix by
scheduling: the talker runs at RTF 1.23, so one conversation already consumes
more than real time and two cannot both be served. The queue is correct. What
was wrong is that it was INVISIBLE -- the second conversation simply appeared
to have become slow, which is the shape the user reported.

So two claims are tested here, and neither is about speed:

1. the wait is attributed -- ``tts_wait_ms`` separates queueing from compute,
   per conversation, without one conversation reading another's number;
2. the wait is announced -- a turn behind another turn emits ``turn.queued``
   BEFORE it waits, not after.

The first test drives the REAL ``InProcessQwen3Tts.synthesize`` (its lock and
its publication, executed, not described) with only ``_generate`` replaced, so
no checkpoint and no GPU are needed to prove the shipped code path.

    CUDA_VISIBLE_DEVICES=99 python -m pytest \\
      test/registered/translator/test_tts_contention.py -v
"""

import asyncio
import unittest

import numpy as np

from sglang.srt.translator.backends import TTS_QUEUE_WAIT_S, AudioChunk
from sglang.srt.translator.inprocess_tts import (
    InProcessQwen3Tts,
    InProcessTtsConfig,
)
from sglang.srt.translator.session import EventKind, run_conversation
from test_session import (  # noqa: E402  - sibling helper module
    VOICE_A_HZ,
    conversation_audio,
    make_session,
)

SYNTHESIS_S = 0.30


def detached_backend(generate_seconds: float = SYNTHESIS_S) -> InProcessQwen3Tts:
    """A real InProcessQwen3Tts with only the checkpoint call replaced.

    ``__init__`` reads a checkpoint's geometry, which would make this a GPU
    test. Everything this test is about -- the lock, the wait it publishes,
    the ``busy`` predicate -- lives in ``synthesize``, so the instance is
    built directly and given exactly the attributes that method touches. The
    code under test is the shipped one; only the model is absent.
    """
    tts = object.__new__(InProcessQwen3Tts)
    tts.config = InProcessTtsConfig()
    tts.sample_rate = tts.config.sample_rate
    tts.min_reference_seconds = tts.config.min_reference_seconds
    tts._languages = ("de", "es")
    tts._model = object()
    tts._lock = asyncio.Lock()

    def _generate(text, language, reference, reference_text):
        # The real one blocks a worker thread; this one blocks for a known
        # time so the wait a second caller measures has a value to compare to.
        import time as _time

        _time.sleep(generate_seconds)
        return np.zeros(int(tts.sample_rate * 0.5), dtype=np.float32)

    tts._generate = _generate
    return tts


async def synthesize_once(tts, text: str) -> float:
    reference = AudioChunk(np.zeros(int(24000 * 4.0), dtype=np.float32), 24000)
    async for _ in tts.synthesize(text, "es", reference, "", None):
        pass
    return TTS_QUEUE_WAIT_S.get()


class TestQueueWaitIsAttributed(unittest.IsolatedAsyncioTestCase):
    async def test_a_lone_call_waits_for_nobody(self):
        tts = detached_backend()
        self.assertFalse(tts.busy)
        waited = await synthesize_once(tts, "hola")
        self.assertLess(waited, 0.05, "an idle talker cannot have been queued")

    async def test_the_second_caller_pays_the_first_synthesis(self):
        tts = detached_backend()
        first, second = await asyncio.gather(
            synthesize_once(tts, "uno"), synthesize_once(tts, "dos")
        )
        waits = sorted([first, second])
        self.assertLess(waits[0], 0.05, "one of the two ran immediately")
        # The loser waits for the whole of the winner's synthesis. Asserted as
        # a floor rather than a window: a loaded box may schedule late, and a
        # test that fails on a slow machine teaches people to ignore it.
        self.assertGreater(
            waits[1],
            SYNTHESIS_S * 0.8,
            "the queued call must report the wait it actually served",
        )

    async def test_one_conversation_never_reads_another_s_wait(self):
        """The number is per call, not per backend.

        An attribute on the shared backend would carry whichever turn finished
        last -- and it would be wrong exactly when two conversations run, i.e.
        the only time anyone reads it.
        """
        tts = detached_backend()
        waits = await asyncio.gather(
            *(synthesize_once(tts, f"turn {i}") for i in range(4))
        )
        self.assertEqual(len(set(round(w, 3) for w in waits)), 4,
                         f"four callers reported {waits}, so some shared a slot")
        self.assertAlmostEqual(min(waits), 0.0, delta=0.05)

    async def test_busy_is_true_only_while_a_turn_is_inside(self):
        tts = detached_backend()
        task = asyncio.create_task(synthesize_once(tts, "hola"))
        await asyncio.sleep(SYNTHESIS_S / 3)
        self.assertTrue(tts.busy, "a synthesis in flight must be visible")
        await task
        self.assertFalse(tts.busy, "the talker must free up again")


class QueuedTts:
    """A TTS that is always busy, and publishes a wait like the real one."""

    def __init__(self, inner, wait_s: float = 1.25):
        self._inner = inner
        self._wait_s = wait_s

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @property
    def busy(self) -> bool:
        return True

    def synthesize(self, *args, **kwargs):
        TTS_QUEUE_WAIT_S.set(self._wait_s)
        return self._inner.synthesize(*args, **kwargs)


class TestQueueIsAnnounced(unittest.IsolatedAsyncioTestCase):
    async def _events(self, tts_factory):
        session, _asr, _mt, tts = make_session()
        session.tts = tts_factory(tts)
        results = await run_conversation(
            session, [conversation_audio((VOICE_A_HZ, 1.2))]
        )
        kinds = [e.kind for e in session.journal.since(0)[0]]
        return session, results, kinds

    async def test_a_queued_turn_says_so(self):
        session, results, kinds = await self._events(QueuedTts)
        self.assertIn(
            EventKind.TURN_QUEUED, kinds,
            "a turn behind another conversation must announce the wait; "
            "silence is what made this look like a broken translator",
        )
        # And it is announced BEFORE the audio it is waiting for, or it is
        # not a warning, it is a post-mortem.
        self.assertLess(
            kinds.index(EventKind.TURN_QUEUED),
            kinds.index(EventKind.TURN_AUDIO),
            "the notice arrived after the audio it was warning about",
        )
        self.assertTrue(results, "the queued turn must still complete")

    async def test_the_wait_lands_in_the_stopwatch(self):
        _session, results, _kinds = await self._events(QueuedTts)
        timings = results[-1].timings
        self.assertAlmostEqual(timings.tts_wait_ms, 1250.0, delta=1.0)
        self.assertGreaterEqual(
            timings.tts_first_audio_ms, 0.0,
            "first-audio must still be reported alongside the wait",
        )

    async def test_an_idle_talker_announces_nothing(self):
        _session, results, kinds = await self._events(lambda tts: tts)
        self.assertNotIn(
            EventKind.TURN_QUEUED, kinds,
            "a turn that never waited must not claim it did",
        )
        self.assertEqual(results[-1].timings.tts_wait_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
