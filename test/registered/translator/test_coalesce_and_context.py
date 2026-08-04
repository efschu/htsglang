# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Over-fragmentation: bundling the backlog, and per-direction MT context.

The user's report (§17.8.21 open item 3) was that the pipeline "chops
everything into pieces even where context exists". Two mechanisms answer it
and they are independent, so they are tested independently:

  * segments WAITING behind a running turn are folded into one bundle, so the
    backlog becomes one recognition and one translation instead of N;
  * every MT call carries the last N turns of THIS session in THIS direction,
    so terminology and pronoun choice survive a turn boundary.

Every arm has its control in the same class -- the mechanism switched off, or
the state it reacts to removed -- and the control asserts the OLD behaviour
comes back. An arm with no control is not evidence that a mechanism works.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sglang.srt.translator.backends import AudioChunk  # noqa: E402
from sglang.srt.translator.segmenter import (  # noqa: E402
    Segment,
    SegmentReason,
)
from sglang.srt.translator.session import EventKind, run_conversation  # noqa: E402
from test_session import (  # noqa: E402
    LANG_A,
    LANG_B,
    RATE,
    VOICE_A_HZ,
    VOICE_B_HZ,
    conversation_audio,
    make_session,
)


def segment(index, seconds=2.0, rate=RATE, reason=SegmentReason.PAUSE):
    """One closed segment of `seconds` of silence, ready to enqueue."""
    samples = np.zeros(int(rate * seconds), dtype=np.float32)
    return Segment(
        audio=AudioChunk(samples, rate),
        reason=reason,
        index=index,
        start_s=float(index) * seconds,
    )


def coalesced_events(session):
    events, _gap = session.journal.since(0)
    return [e for e in events if e.kind is EventKind.TURN_COALESCED]


class TestBundlingTheBacklog(unittest.IsolatedAsyncioTestCase):
    """What is already waiting is bundled; what is running is untouched."""

    async def test_the_first_segment_is_never_bundled(self):
        """THE EAGER-SMALL CASE, and it is the common one.

        With nothing waiting there is no backlog, so there is no evidence that
        context exists and no latency already lost. A pipeline that keeps up
        must behave exactly as it did before this feature.
        """
        session, _asr, _mt, _tts = make_session()
        self.assertTrue(session.enqueue(segment(0)))
        self.assertEqual(session.pending(), 1)
        self.assertEqual(session.segments_coalesced, 0)
        self.assertEqual(coalesced_events(session), [])

    async def test_a_waiting_segment_is_folded_into_the_one_ahead_of_it(self):
        session, _asr, _mt, _tts = make_session()
        session.enqueue(segment(0, seconds=2.0))
        session.enqueue(segment(1, seconds=3.0))
        # One item, holding both utterances.
        self.assertEqual(session.pending(), 1)
        self.assertEqual(session.segments_coalesced, 1)
        self.assertAlmostEqual(session._queue[0].duration_s, 5.0, places=3)
        events = coalesced_events(session)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].payload["segment_index"], 1)
        self.assertEqual(events[0].payload["into_index"], 0)

    async def test_without_the_flag_each_segment_stays_its_own_turn(self):
        """THE CONTROL: the pre-feature pipeline, chopping as reported."""
        session, _asr, _mt, _tts = make_session(coalesce_queued=False)
        session.enqueue(segment(0, seconds=2.0))
        session.enqueue(segment(1, seconds=3.0))
        self.assertEqual(session.pending(), 2)
        self.assertEqual(session.segments_coalesced, 0)
        self.assertEqual(coalesced_events(session), [])

    async def test_the_bundle_keeps_the_identity_of_its_oldest_member(self):
        """`index` and `start_s` order the transcript and resume the client.

        The bundle occupies the position the FIRST segment claimed, so a
        client that resumes mid-bundle does not see the conversation jump.
        """
        session, _asr, _mt, _tts = make_session()
        session.enqueue(segment(7, seconds=2.0))
        session.enqueue(segment(8, seconds=2.0))
        self.assertEqual(session.pending(), 1, "nothing was bundled at all")
        self.assertAlmostEqual(session._queue[0].duration_s, 4.0, places=3)
        self.assertEqual(session._queue[0].index, 7)
        self.assertAlmostEqual(session._queue[0].start_s, 14.0, places=3)

    async def test_the_bundle_stops_growing_at_the_cap(self):
        """A long backlog must not become one unbounded ASR call."""
        session, _asr, _mt, _tts = make_session(
            coalesce_max_s=5.0, max_queued_turns=4
        )
        for index in range(4):
            session.enqueue(segment(index, seconds=2.0))
        # 2+2 fills the first bundle to 4 s; a third would make 6 s, over the
        # 5 s cap, so it starts a second bundle which then takes the fourth.
        self.assertEqual(session.pending(), 2)
        self.assertAlmostEqual(session._queue[0].duration_s, 4.0, places=3)
        self.assertAlmostEqual(session._queue[1].duration_s, 4.0, places=3)

    async def test_a_rate_mismatch_is_not_bundled(self):
        """Resampling here would hide a wire/pipeline disagreement."""
        session, _asr, _mt, _tts = make_session()
        session.enqueue(segment(0, seconds=1.0, rate=RATE))
        session.enqueue(segment(1, seconds=1.0, rate=RATE * 2))
        self.assertEqual(session.pending(), 2)
        self.assertEqual(session.segments_coalesced, 0)

    async def test_the_backlog_is_bundled_instead_of_dropped(self):
        """The overrun drop deleted real speech; bundling has nothing to drop.

        `max_queued_turns` is small (2 by default) and the old queue answered
        an overrun by discarding the OLDEST waiting segment -- a whole
        utterance, silently, exactly when the pipeline was already struggling.
        """
        session, _asr, _mt, _tts = make_session()
        for index in range(6):
            session.enqueue(segment(index, seconds=1.0))
        self.assertEqual(session.turns_dropped, 0)
        self.assertEqual(session.pending(), 1)
        self.assertAlmostEqual(session._queue[0].duration_s, 6.0, places=3)

    async def test_without_the_flag_the_backlog_is_dropped(self):
        """THE CONTROL for the arm above: the loss is real, not hypothetical."""
        session, _asr, _mt, _tts = make_session(coalesce_queued=False)
        for index in range(6):
            session.enqueue(segment(index, seconds=1.0))
        self.assertEqual(session.pending(), 2)
        self.assertEqual(session.turns_dropped, 4)

    async def test_the_bundle_is_announced_to_the_client(self):
        session, _asr, _mt, _tts = make_session()
        supports = session.state()["supports"]
        self.assertTrue(supports["coalesce_queued"])
        off, _a, _m, _t = make_session(coalesce_queued=False, session_id="s2")
        self.assertFalse(off.state()["supports"]["coalesce_queued"])


class TestMtContext(unittest.IsolatedAsyncioTestCase):
    """The last N turns of this session, in this direction, and no others."""

    async def test_the_second_turn_carries_the_first_as_context(self):
        session, _asr, mt, _tts = make_session()
        # Same voice twice: one direction, two turns.
        audio = conversation_audio((VOICE_A_HZ, 2.0), (VOICE_A_HZ, 2.0))
        results = await run_conversation(session, [audio])
        self.assertEqual(len(results), 2, [r.source_text for r in results])
        self.assertEqual(len(mt.contexts), 2, mt.contexts)
        self.assertEqual(mt.contexts[0], [], "turn 1 has no history to carry")
        self.assertEqual(
            mt.contexts[1],
            [(results[0].source_text, results[0].translations[LANG_B])],
            "turn 2 did not receive turn 1 as context",
        )

    async def test_without_the_knob_no_context_is_sent(self):
        """THE CONTROL: a context-free sentence translator, as before."""
        session, _asr, mt, _tts = make_session(mt_context_turns=0)
        audio = conversation_audio((VOICE_A_HZ, 2.0), (VOICE_A_HZ, 2.0))
        await run_conversation(session, [audio])
        self.assertTrue(mt.contexts, "MT was never called")
        self.assertTrue(
            all(ctx == [] for ctx in mt.contexts), mt.contexts
        )

    async def test_context_does_not_cross_directions(self):
        """A de->es history is not usable context for an es->de call.

        The roles of the two languages are swapped, so replaying the pairs
        would show the model an assistant answering in the SOURCE language --
        the single most effective way to make it stop translating.
        """
        session, _asr, mt, _tts = make_session()
        audio = conversation_audio((VOICE_A_HZ, 2.0), (VOICE_B_HZ, 2.0))
        results = await run_conversation(session, [audio])
        self.assertEqual(results[0].source_language, LANG_A)
        self.assertEqual(results[1].source_language, LANG_B)
        self.assertEqual(len(mt.contexts), 2, mt.contexts)
        self.assertEqual(
            mt.contexts[1], [],
            "the reverse direction inherited the forward direction's history",
        )

    async def test_context_does_not_cross_sessions(self):
        """One backend instance serves the whole process.

        This is the bug the session-owned store exists to fix: the history
        used to live on `OpenAiMt`, which `launch.py` builds once and hands to
        every session, so one conversation's sentences were prepended to
        another conversation's prompt.
        """
        first, _asr, mt, _tts = make_session(session_id="s1")
        second, _asr2, _mt2, _tts2 = make_session(session_id="s2")
        second.mt = mt  # the shared process-wide backend
        audio = conversation_audio((VOICE_A_HZ, 2.0), (VOICE_A_HZ, 2.0))
        await run_conversation(first, [audio])
        calls_before = len(mt.contexts)
        self.assertGreaterEqual(calls_before, 2)
        await run_conversation(second, [conversation_audio((VOICE_A_HZ, 2.0))])
        self.assertEqual(
            mt.contexts[calls_before], [],
            "session 2 was handed session 1's conversation",
        )

    async def test_the_context_is_capped(self):
        session, _asr, mt, _tts = make_session(mt_context_turns=1)
        audio = conversation_audio(
            (VOICE_A_HZ, 2.0), (VOICE_A_HZ, 2.0), (VOICE_A_HZ, 2.0)
        )
        await run_conversation(session, [audio])
        self.assertEqual(len(mt.contexts), 3, mt.contexts)
        self.assertTrue(all(len(ctx) <= 1 for ctx in mt.contexts), mt.contexts)
        self.assertEqual(len(mt.contexts[2]), 1)

    async def test_the_depth_is_announced_to_the_client(self):
        session, _asr, _mt, _tts = make_session(mt_context_turns=4)
        self.assertEqual(session.state()["supports"]["mt_context_turns"], 4)


class TestThePromptCarriesTheContext(unittest.IsolatedAsyncioTestCase):
    """The real backend puts the pairs on the wire, as dialogue turns."""

    def test_context_becomes_user_assistant_pairs(self):
        from sglang.srt.translator.mt import OpenAiMt

        backend = OpenAiMt()
        messages = backend._messages(
            "und weiter", "de", "es",
            [("hallo", "hola"), ("wie geht es dir", "como estas")],
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(
            [(m["role"], m["content"]) for m in messages[1:]],
            [
                ("user", "hallo"),
                ("assistant", "hola"),
                ("user", "wie geht es dir"),
                ("assistant", "como estas"),
                ("user", "und weiter"),
            ],
        )

    def test_without_context_the_prompt_is_system_plus_one_turn(self):
        """THE CONTROL: the shape every existing MT test was written against."""
        from sglang.srt.translator.mt import OpenAiMt

        backend = OpenAiMt()
        messages = backend._messages("und weiter", "de", "es", None)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1], {"role": "user", "content": "und weiter"})

    def test_the_backend_keeps_no_history_of_its_own(self):
        """The leak this replaces: state on a process-wide instance.

        Asserted structurally rather than behaviourally, because the failure
        mode is a field that quietly comes back.
        """
        from sglang.srt.translator.mt import MtConfig, OpenAiMt

        backend = OpenAiMt()
        self.assertFalse(hasattr(backend, "_history"))
        self.assertFalse(hasattr(backend, "remember"))
        self.assertFalse(hasattr(MtConfig(), "history_turns"))


if __name__ == "__main__":
    unittest.main()
