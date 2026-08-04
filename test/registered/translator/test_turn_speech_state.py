# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""A turn must say what it is DOING with each clause (``turn.speech``).

The gap the user experiences is the interval between a clause's translated
text appearing and its audio starting. Until now nothing on the wire named
that interval, so the screen could not tell "working on it" from "stuck" --
and with one talker serving every conversation (measured RTF 1.23, §17.8.2)
the interval is neither short nor rare.

``turn.speech`` carries ``{turn_id, target, unit_index, state}`` with state
``queued`` -> ``synthesizing`` -> ``spoken``, one frame per transition.

Two properties are asserted and they fail differently:

* the ORDER per unit, and the FIFO across units. The single worker is what
  makes the audio order equal the text order, so a ``synthesizing`` that
  overlaps another unit's would mean the ordering guarantee is gone;
* that a unit which will never be spoken is never ANNOUNCED. Reading mode
  (§17.1) skips the synthesizer entirely, so a ``queued`` emitted there is a
  spinner the client can never clear -- the failure mode of announcing state
  optimistically rather than at the transition that actually happened.

    CUDA_VISIBLE_DEVICES=99 python -m pytest \\
      test/registered/translator/test_turn_speech_state.py -v
"""

import asyncio
import unittest

from sglang.srt.translator.session import EventKind, OutputMode
from test_session import (  # noqa: E402  - sibling helper module
    VOICE_A_HZ,
    conversation_audio,
    make_session,
)

#: Three sentences, so the accumulator closes three synthesis units.
THREE_SENTENCES = "Uno primero. Dos segundo. Tres tercero."


class InstantMt:
    """The whole translation at once: MT is never the bottleneck here."""

    def __init__(self):
        self.name = "instant-mt"

    def supported_languages(self):
        return ("a", "b")

    async def translate(self, text, source, target, *, context=None):
        return THREE_SENTENCES

    async def translate_stream(self, text, source, target, *, context=None):
        # Word by word, as a streaming LLM arrives. Handing the accumulator
        # the whole string would close ONE unit and the FIFO assertion would
        # have nothing to order.
        for word in THREE_SENTENCES.split(" "):
            yield word + " "


class SlowTts:
    """A talker that takes real time per clause, like the measured one."""

    def __init__(self, inner, delay_s=0.20):
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


def build(**session_kwargs):
    session, _asr, _mt, inner = make_session(**session_kwargs)
    session.mt = InstantMt()
    session.tts = SlowTts(inner)
    return session


async def run_one_turn(session):
    audio = conversation_audio((VOICE_A_HZ, 2.0))
    for segment in session.push_audio(audio):
        session.enqueue(segment)
    await session.drain()
    return list(session.journal.since(0)[0])


def speech_events(events):
    return [e for e in events if e.kind is EventKind.TURN_SPEECH]


class TestTurnSpeechState(unittest.IsolatedAsyncioTestCase):
    async def test_every_unit_walks_queued_synthesizing_spoken(self):
        events = await run_one_turn(build())
        frames = speech_events(events)
        self.assertTrue(frames, "no turn.speech frame was emitted at all")

        by_unit = {}
        for event in frames:
            by_unit.setdefault(event.payload["unit_index"], []).append(
                event.payload["state"]
            )
        self.assertEqual(
            sorted(by_unit), list(range(len(by_unit))),
            "unit_index must be dense and start at 0",
        )
        for index, states in sorted(by_unit.items()):
            self.assertEqual(
                states, ["queued", "synthesizing", "spoken"],
                f"unit {index} did not walk the three states in order",
            )

    async def test_every_frame_names_its_turn_and_target(self):
        events = await run_one_turn(build())
        turn_ids = {e.payload["turn_id"] for e in speech_events(events)}
        self.assertEqual(len(turn_ids), 1)
        self.assertNotIn(None, turn_ids)
        for event in speech_events(events):
            # Without the target the client cannot place the state on the
            # right bubble in a conversation that fans out.
            self.assertTrue(event.payload["target"])

    async def test_the_worker_is_fifo_so_no_two_units_overlap(self):
        events = await run_one_turn(build())
        frames = speech_events(events)
        order = [(e.payload["unit_index"], e.payload["state"]) for e in frames]
        # One worker: a unit is only spoken before the NEXT one starts. This
        # is the property that makes the audio order equal the text order.
        active = None
        for index, state in order:
            if state == "synthesizing":
                self.assertIsNone(
                    active,
                    f"unit {index} started while unit {active} was still "
                    f"synthesizing -- the FIFO worker is not serializing",
                )
                active = index
            elif state == "spoken":
                self.assertEqual(active, index)
                active = None

    async def test_the_audio_of_a_unit_lands_between_its_two_states(self):
        events = await run_one_turn(build())
        # The states have to bracket the thing they describe, or they are
        # labels on the wrong interval.
        seq = {}
        for position, event in enumerate(events):
            if event.kind is EventKind.TURN_SPEECH:
                seq[(event.payload["unit_index"], event.payload["state"])] = position
        audio_at = [
            position for position, event in enumerate(events)
            if event.kind is EventKind.TURN_AUDIO
        ]
        self.assertTrue(audio_at, "the turn produced no audio to bracket")
        first_synth = min(
            position for (_, state), position in seq.items()
            if state == "synthesizing"
        )
        last_spoken = max(
            position for (_, state), position in seq.items() if state == "spoken"
        )
        self.assertGreater(min(audio_at), first_synth)
        self.assertLess(max(audio_at), last_spoken)

    async def test_reading_mode_announces_nothing(self):
        """The can-fail half: no synthesis, so no state to announce.

        Announcing ``queued`` for a unit the synthesizer will never see is
        the defect this asserts against -- the client would show a clause as
        pending forever, with no event that can ever clear it.
        """
        session = build()
        session.set_output_mode(OutputMode.SILENT)
        events = await run_one_turn(session)
        self.assertTrue(
            [e for e in events if e.kind is EventKind.TURN_TRANSLATION],
            "reading mode must still translate -- otherwise this proves nothing",
        )
        self.assertEqual(
            speech_events(events), [],
            "reading mode announced a synthesis state for audio it never makes",
        )


if __name__ == "__main__":
    unittest.main()
