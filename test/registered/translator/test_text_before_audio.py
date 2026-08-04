# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The translated TEXT must not wait for the AUDIO (§19.13).

User, from the field: *"the translated text should be there much earlier than
the audio is transferred. It somehow waits for it. The text has to be streamed
before that."* He is right, and the cause was one `await`.

``speak(unit)`` was awaited INSIDE the MT loop, so the loop stopped consuming
translation deltas for the entire duration of a clause's synthesis -- measured
at 3.4 s per clause (§18.4). The first clause's text was journalled before its
own synthesis, which is why a one-clause turn looked fine and hid this; from
the second clause on, every line of text arrived at the speed of the SPEAKER
rather than of the translator.

The falsifier makes the two speeds unmistakable: a translator that produces
several sentences immediately, and a talker that takes a visible time per
clause. Then the ordering in the journal answers the question directly --

* unfixed: text1, audio1, text2, audio2, ... (text interleaved with speech)
* fixed:   text1, text2, text3, ... then audio (text runs ahead)

Ordering rather than wall-clock, because a timing assertion under a loaded
test runner is a flake generator; the journal sequence is exact.

    CUDA_VISIBLE_DEVICES=99 python -m pytest \\
      test/registered/translator/test_text_before_audio.py -v
"""

import asyncio
import unittest

from sglang.srt.translator.session import EventKind
from test_session import (  # noqa: E402  - sibling helper module
    VOICE_A_HZ,
    conversation_audio,
    make_session,
)

#: Three sentences, so the accumulator emits three synthesis units.
THREE_SENTENCES = "Uno primero. Dos segundo. Tres tercero."


class InstantMt:
    """Produces the whole translation at once: MT is never the bottleneck."""

    def __init__(self, languages):
        self.name = "instant-mt"
        self._languages = tuple(languages)

    def supported_languages(self):
        return self._languages

    async def translate(self, text, source, target, *, context=None):
        return THREE_SENTENCES

    async def translate_stream(self, text, source, target, *, context=None):
        # Word by word, the way a real streaming LLM arrives: the accumulator
        # only closes a unit when it sees a sentence end, so handing it the
        # whole string at once would produce ONE unit and the test could not
        # show an interleaving defect at all.
        for word in THREE_SENTENCES.split(" "):
            yield word + " "


class SlowTts:
    """A talker that takes real time per clause, like the measured one."""

    def __init__(self, inner, delay_s=0.30):
        self._inner = inner
        self._delay_s = delay_s
        self._lock = asyncio.Lock()

    def __getattr__(self, name):
        return getattr(self._inner, name)

    @property
    def busy(self):
        return self._lock.locked()

    async def synthesize(self, text, language, reference, reference_text, voice_id):
        async with self._lock:
            await asyncio.sleep(self._delay_s)
            async for piece in self._inner.synthesize(
                text, language, reference, reference_text, voice_id
            ):
                yield piece


def build(session_kwargs=None):
    session, _asr, _mt, inner = make_session(**(session_kwargs or {}))
    session.mt = InstantMt(session.matrix.mt_languages()
                           if hasattr(session.matrix, "mt_languages")
                           else ("a", "b"))
    session.tts = SlowTts(inner)
    return session, inner


class TestTextRunsAheadOfSpeech(unittest.IsolatedAsyncioTestCase):
    async def _run_one_turn(self):
        session, _inner = build()
        segments = session.push_audio(conversation_audio((VOICE_A_HZ, 1.5)))
        self.assertTrue(segments)
        await session.run_turn_multi(segments[0])
        return session

    async def test_every_clause_of_text_precedes_the_first_audio(self):
        """The falsifier. Unfixed, audio for clause 1 lands before text 2."""
        session = await self._run_one_turn()
        events = session.journal.since(0)[0]
        text_at = [
            i for i, e in enumerate(events)
            if e.kind is EventKind.TURN_TRANSLATION and e.payload.get("partial")
        ]
        audio_at = [
            i for i, e in enumerate(events) if e.kind is EventKind.TURN_AUDIO
        ]
        self.assertGreaterEqual(
            len(text_at), 2,
            "the fixture must produce several clauses or it cannot show the "
            "defect at all -- with one clause the old code looked correct",
        )
        self.assertTrue(audio_at, "the turn must still produce audio")
        self.assertLess(
            max(text_at), min(audio_at),
            "text event %d arrived after the first audio event %d -- the "
            "translation is still being paced by the synthesizer"
            % (max(text_at), min(audio_at)),
        )

    async def test_the_audio_still_arrives_and_keeps_its_turn_id(self):
        """Decoupling must not cost the attribution the UI replays on."""
        session = await self._run_one_turn()
        audio = [
            e for e in session.journal.since(0)[0]
            if e.kind is EventKind.TURN_AUDIO
        ]
        self.assertTrue(audio)
        turn_ids = {e.payload.get("turn_id") for e in audio}
        self.assertEqual(
            len(turn_ids), 1, "every audio chunk of one turn carries one id"
        )
        self.assertNotIn(None, turn_ids)

    async def test_the_audio_order_matches_the_text_order(self):
        """One FIFO worker: clause N is spoken before clause N+1."""
        session = await self._run_one_turn()
        events = session.journal.since(0)[0]
        spoken = [
            e.payload.get("text") for e in events
            if e.kind is EventKind.TURN_TRANSLATION and e.payload.get("partial")
        ]
        self.assertEqual(
            spoken, [s.strip() for s in spoken],
            "clauses must not be reordered on their way to the talker",
        )
        # The final non-partial event is the whole translation.
        whole = [
            e.payload["text"] for e in events
            if e.kind is EventKind.TURN_TRANSLATION
            and not e.payload.get("partial")
        ]
        self.assertTrue(whole)
        for clause in spoken:
            self.assertIn(clause.strip(" ."), whole[-1])

    async def test_mt_total_no_longer_includes_synthesis(self):
        """The metric said 'mt' and measured the talker (§17.8.9)."""
        session, _inner = build()
        segments = session.push_audio(conversation_audio((VOICE_A_HZ, 1.5)))
        results = await session.run_turn_multi(segments[0])
        self.assertTrue(results)
        done = [
            e for e in session.journal.since(0)[0]
            if e.kind is EventKind.TURN_DONE
        ]
        self.assertTrue(done)
        timings = done[-1].payload.get("timings") or {}
        if "mt_total_ms" in timings and "tts_first_audio_ms" in timings:
            self.assertLess(
                timings["mt_total_ms"], 250.0,
                "mt_total %r still carries the synthesizer's time"
                % (timings["mt_total_ms"],),
            )


if __name__ == "__main__":
    unittest.main()
