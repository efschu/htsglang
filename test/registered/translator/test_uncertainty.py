# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Uncertain attributions: marked, ranked, and correctable (§17.4).

The user called this the important one, and the reason is in the numbers
rather than in the UI. ``probe_speaker_change.py --pool`` over the 17-voice
pool at the shipped 2.5 s window gives within-speaker p05 0.637 against
between-speaker p95 0.583 -- but the populations OVERLAP: the pool's worst
collision, boy-03 against girl-01, scores 0.734, which is above the
within-speaker minimum of 0.624. No threshold closes that gap. The machine
cannot be made right, so it has to be made correctable.

Two tests carry the design:

``test_an_unconfirmed_uncertain_line_never_moves_a_centroid`` is the falsifier
against the obvious implementation. Folding an ambiguous slice on the way past
corrupts an identity permanently and nothing downstream can tell.

``test_the_children_case_offers_a_new_speaker`` is the mandatory scenario: two
similar young voices where the right answer is frequently "neither of the two
you know". If ``speaker-N (new)`` were appended after the known candidates
instead of ranked among them, that answer would never be reachable.

    CUDA_VISIBLE_DEVICES=99 python -m pytest \\
      test/registered/translator/test_uncertainty.py -v
"""

import unittest

import numpy as np

from sglang.srt.translator.backends import AudioChunk
from sglang.srt.translator.session import EventKind, run_conversation
from sglang.srt.translator.speakers import (
    SpeakerEmbedding,
    SpeakerRegistry,
    SpeakerRegistryConfig,
)
from sglang.srt.translator.transcript_log import (
    CONFIDENCE_EXACT,
    CONFIDENCE_UNCERTAIN,
)
from test_session import (  # noqa: E402  - sibling helper module
    RATE,
    VOICE_A_HZ,
    conversation_audio,
    make_session,
    tone,
)


def unit(*components) -> SpeakerEmbedding:
    vector = np.array(components, dtype=np.float32)
    return SpeakerEmbedding(vector)


def at_similarity(reference: SpeakerEmbedding, cosine: float) -> SpeakerEmbedding:
    """An embedding at a chosen cosine to ``reference``, in 2 dimensions."""
    angle = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
    base = np.arctan2(reference.vector[1], reference.vector[0])
    return unit(np.cos(base + angle), np.sin(base + angle))


async def _resolved(value):
    """An awaitable that yields a value already in hand."""
    return value


class TestTheBand(unittest.TestCase):
    """The verdict, driven at chosen similarities rather than by luck."""

    def setUp(self):
        self.registry = SpeakerRegistry(
            SpeakerRegistryConfig(min_slice_s=0.5), clock=lambda: 0.0
        )
        self.anchor = unit(1.0, 0.0)
        self.audio = AudioChunk(tone(VOICE_A_HZ, 1.0), RATE)
        self.registry.assign(self.anchor, self.audio, "hola", "xx")

    def _verdict(self, cosine):
        probe = at_similarity(self.anchor, cosine)
        ranked = self.registry.rank(probe)
        # The similarity actually achieved, so the test reports the number the
        # code saw rather than the one it asked for.
        return ranked, self.registry.uncertainty(ranked, "speaker-2")

    def test_the_three_bands_produce_the_three_verdicts(self):
        for cosine, expected in ((0.90, False), (0.65, True), (0.30, False)):
            with self.subTest(cosine=cosine):
                ranked, (uncertain, _candidates) = self._verdict(cosine)
                self.assertAlmostEqual(ranked[0][1], cosine, places=2)
                self.assertEqual(uncertain, expected)

    def test_the_edges_are_the_measured_ones(self):
        cfg = self.registry.config
        self.assertAlmostEqual(cfg.uncertain_floor, 0.583)
        self.assertAlmostEqual(cfg.within_speaker_floor, 0.637)
        # Just inside and just outside the lower edge.
        _r, (below, _c) = self._verdict(cfg.uncertain_floor - 0.02)
        _r, (above, _c) = self._verdict(cfg.uncertain_floor + 0.02)
        self.assertFalse(below)
        self.assertTrue(above)

    def test_the_candidates_are_the_similarities_the_assignment_used(self):
        ranked, (_uncertain, candidates) = self._verdict(0.65)
        known = [c for c in candidates if not c["new"]]
        self.assertEqual(known[0]["speaker_id"], ranked[0][0])
        self.assertAlmostEqual(known[0]["similarity"], round(ranked[0][1], 3))

    def test_the_children_case_offers_a_new_speaker(self):
        """Two similar young voices; the answer is often 'neither'.

        Three mutually-ambiguous directions, which needs three dimensions:
        in two, a third vector ambiguous against the first is necessarily
        almost identical to the second, and the scenario collapses.
        """
        registry = SpeakerRegistry(
            SpeakerRegistryConfig(min_slice_s=0.5), clock=lambda: 0.0
        )
        # Pairwise cosine 0.6 between all three: the boy-03/girl-01 shape
        # from the pool sweep, where two real people score inside the band.
        moritz = unit(1.0, 0.0, 0.0)
        ben = unit(0.6, 0.8, 0.0)
        third = unit(0.6, 0.2, 0.7746)
        registry.assign(moritz, self.audio, "hola", "xx")
        registry.assign(ben, self.audio, "hola", "xx")
        ranked = registry.rank(third)
        self.assertLess(ranked[0][1], registry.config.match_threshold)
        self.assertGreater(ranked[0][1], registry.config.uncertain_floor)
        uncertain, candidates = registry.uncertainty(ranked, "speaker-3")
        self.assertTrue(uncertain)
        self.assertLessEqual(len(candidates), 3)
        self.assertTrue(
            any(c["new"] for c in candidates),
            f"a new speaker must be reachable, got {candidates}",
        )

    def test_a_known_speaker_outranks_new_only_above_the_between_p95(self):
        _r, (_u, low) = self._verdict(0.50)
        _r, (_u, high) = self._verdict(0.66)
        self.assertTrue(low[0]["new"])
        self.assertFalse(high[0]["new"])

    def test_a_declared_but_unheard_speaker_is_not_ranked(self):
        self.registry.create_speaker(label="declared")
        ranked = self.registry.rank(self.anchor)
        self.assertNotIn("declared", [sid for sid, _ in ranked])


class TestUncertaintyInSession(unittest.IsolatedAsyncioTestCase):
    async def _uncertain_session(self):
        """A session whose second turn lands inside the band."""
        session, _asr, _mt, _tts = make_session()
        anchor = unit(1.0, 0.0)
        probe = at_similarity(anchor, 0.65)
        vectors = iter([anchor, probe, probe])

        class ScriptedEmbedder:
            name = "scripted"
            min_seconds = 0.5

            async def embed(self, audio):
                return next(vectors, probe)

        session.embedder = ScriptedEmbedder()
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 1.2))])
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 1.2))])
        return session

    async def test_an_ambiguous_line_is_badged_with_candidates(self):
        session = await self._uncertain_session()
        line = session.transcript.lines()[-1]
        self.assertEqual(line.confidence, CONFIDENCE_UNCERTAIN)
        self.assertTrue(line.candidates)
        self.assertIn("candidates", line.to_json())
        # And a confident line is not badged.
        self.assertEqual(session.transcript.lines()[0].confidence, CONFIDENCE_EXACT)

    async def test_an_unconfirmed_uncertain_line_never_moves_a_centroid(self):
        """The falsifier against folding an ambiguous slice on the way past."""
        session = await self._uncertain_session()
        first_speaker = session.transcript.lines()[0].speaker_id
        before = np.array(session.speakers.get(first_speaker).centroid.vector)
        # A third ambiguous turn against the same profile.
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 1.2))])
        after = np.array(session.speakers.get(first_speaker).centroid.vector)
        np.testing.assert_allclose(before, after, rtol=0, atol=0)

    async def test_a_tap_settles_the_line_and_only_then_folds(self):
        session = await self._uncertain_session()
        line = session.transcript.lines()[-1]
        target = session.transcript.lines()[0].speaker_id
        before = np.array(session.speakers.get(target).centroid.vector)

        changed = session.resolve_line(line.line_id, target)
        self.assertEqual(changed.speaker_id, target)
        self.assertEqual(changed.confidence, CONFIDENCE_EXACT)
        self.assertEqual(changed.candidates, [])
        self.assertEqual(changed.to_json()["resolved_by"], "user")
        after = np.array(session.speakers.get(target).centroid.vector)
        self.assertFalse(
            np.allclose(before, after),
            "a confirmation must be what moves the centroid",
        )

    async def test_a_tap_is_undoable(self):
        session = await self._uncertain_session()
        line = session.transcript.lines()[-1]
        original = line.speaker_id
        target = session.transcript.lines()[0].speaker_id
        session.resolve_line(line.line_id, target)
        reverted = session.undo_resolution(line.line_id)
        self.assertEqual(reverted.speaker_id, original)
        self.assertEqual(reverted.confidence, CONFIDENCE_UNCERTAIN)
        self.assertTrue(reverted.candidates)
        self.assertIsNone(reverted.resolved_by)
        # Undoing twice is refused rather than looping.
        self.assertIsNone(session.undo_resolution(line.line_id))

    async def test_resolving_to_a_new_speaker_mints_one(self):
        session = await self._uncertain_session()
        line = session.transcript.lines()[-1]
        before = len(session.speakers)
        changed = session.resolve_line(line.line_id, None)
        self.assertEqual(len(session.speakers), before + 1)
        self.assertEqual(changed.confidence, CONFIDENCE_EXACT)
        self.assertNotEqual(changed.speaker_id, session.transcript.lines()[0].speaker_id)

    async def test_resolution_is_an_update_event_not_a_silent_rewrite(self):
        session = await self._uncertain_session()
        line = session.transcript.lines()[-1]
        before = session.journal.next_seq
        session.resolve_line(line.line_id, session.transcript.lines()[0].speaker_id)
        updates = [
            event
            for event in session.journal.since(before)[0]
            if event.kind is EventKind.TRANSCRIPT_UPDATE
        ]
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0].payload["line"]["line_id"], line.line_id)

    async def test_a_user_settled_line_is_never_re_decided(self):
        session = await self._uncertain_session()
        line = session.transcript.lines()[-1]
        target = session.transcript.lines()[0].speaker_id
        session.resolve_line(line.line_id, target)
        # More audio arrives; auto-resolution must leave the human's line be.
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 1.2))])
        settled = session.transcript.get(line.line_id)
        self.assertEqual(settled.speaker_id, target)
        self.assertEqual(settled.resolved_by, "user")

    async def test_later_evidence_resolves_an_earlier_badge_visibly(self):
        """The binds-proof for auto-resolution.

        A mechanism that never fires has reach zero, so this drives the case
        it exists for: a known speaker's centroid moves toward the ambiguous
        embedding until the earlier line is no longer in doubt. The badge must
        CHANGE through an update event rather than the line being rewritten
        where nobody is looking.
        """
        session = await self._uncertain_session()
        line = session.transcript.lines()[-1]
        self.assertEqual(line.confidence, CONFIDENCE_UNCERTAIN)
        first = session.transcript.lines()[0].speaker_id
        pending = session._pending[line.line_id]

        # The user confirms, with a speaker button, that a very similar
        # utterance was the FIRST speaker. That fold moves their centroid.
        session.arm_speaker(first)
        session.embedder = type(
            "Near", (), {
                "name": "near", "min_seconds": 0.5,
                "embed": staticmethod(lambda audio: _resolved(pending)),
            },
        )()
        before = session.journal.next_seq
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 1.2))])

        settled = session.transcript.get(line.line_id)
        self.assertEqual(settled.speaker_id, first)
        self.assertEqual(settled.confidence, CONFIDENCE_EXACT)
        self.assertEqual(settled.resolved_by, "auto")
        updates = [
            event
            for event in session.journal.since(before)[0]
            if event.kind is EventKind.TRANSCRIPT_UPDATE
            and event.payload["line"]["line_id"] == line.line_id
        ]
        self.assertTrue(updates, "the badge change must be on the wire")

    async def test_a_manual_line_gets_no_badge_and_no_candidates(self):
        session, _asr, _mt, _tts = make_session()
        new_id = session.add_speaker()
        session.arm_speaker(new_id)
        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 1.2))])
        line = session.transcript.lines()[-1]
        self.assertEqual(line.confidence, CONFIDENCE_EXACT)
        self.assertEqual(line.candidates, [])


if __name__ == "__main__":
    unittest.main()
