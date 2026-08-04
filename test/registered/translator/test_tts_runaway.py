# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The runaway talker: bound it where it decodes, not where it is awaited.

THE LIVE DEFECT (2026-08-04, session 6d78966660e0, turn 21244f89bf36). ASR
returned in 702 ms and the translation in 1.5 s; the synthesis then produced
nothing for six minutes. It was never stuck. py-spy showed the worker thread
moving through `_sample` the whole time, and the reference's per-step config
warning -- an accidental but exact step counter -- put it at 10.7 steps/s
against 12.5 steps/s on a healthy turn. The rate was fine. The COUNT was not:
2996 steps where a normal clause costs ~85, on a translation the MT stage had
returned in 230 ms. The talker had simply not emitted its codec EOS and was
running at its token ceiling.

Three things had to be true at once for that to cost six minutes, and this
file pins each of them:

1. the ceiling was unreachable. `max_new_tokens` was 2048, which at the
   measured 12.5 steps/s is ~164 s -- nearly twice the 90 s unit deadline. A
   talker that never stops could not finish inside the deadline, so the
   deadline could only abandon it;
2. the deadline lived in the event loop only. Cancelling the await does not
   reach a thread already inside `generate_voice_clone`, so the abort freed
   the lock while the talker kept decoding, `busy` went false, and the
   automatic redrive started a SECOND generation on the same module tree;
3. the failure named the wrong budget. A 90 s abort was logged as "did not
   finish within 300s", which is where the investigation lost its first hour.

    CUDA_VISIBLE_DEVICES=99 python -m pytest \\
      test/registered/translator/test_tts_runaway.py -v
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sglang.srt.translator.backends import AudioChunk  # noqa: E402
from sglang.srt.translator.inprocess_tts import (  # noqa: E402
    InProcessQwen3Tts,
    InProcessTtsConfig,
)
from sglang.srt.translator.session import EventKind, run_conversation  # noqa: E402

from test_session import (  # noqa: E402  - sibling helper module
    LANG_A,
    LANG_B,
    VOICE_A_HZ,
    conversation_audio,
    make_session,
)

#: What a healthy turn actually costs, measured on the live service on
#: 2026-08-04: 85 decode steps at 12.5 steps/s for one clause.
MEASURED_STEPS_PER_SECOND = 12.5
MEASURED_STEPS_PER_CLAUSE = 85
#: `TranslatorSession.tts_unit_timeout_s`. Duplicated as a literal on purpose:
#: if someone widens the session deadline, the arithmetic below should be
#: re-derived deliberately rather than silently follow it.
UNIT_DEADLINE_S = 90.0


class TestTheTokenCeilingIsReachable(unittest.TestCase):
    """A ceiling above the deadline is not a ceiling, it is a trap."""

    def test_the_token_ceiling_fits_inside_the_unit_deadline(self):
        config = InProcessTtsConfig()
        worst_case_s = config.max_new_tokens / MEASURED_STEPS_PER_SECOND
        self.assertLess(
            worst_case_s,
            UNIT_DEADLINE_S,
            f"a talker running to max_new_tokens={config.max_new_tokens} "
            f"costs {worst_case_s:.0f}s at the measured "
            f"{MEASURED_STEPS_PER_SECOND} steps/s, which the {UNIT_DEADLINE_S:.0f}s "
            "unit deadline cannot wait for -- the deadline could only ever "
            "abandon the thread, which is the 2026-08-04 defect",
        )

    def test_the_ceiling_still_clears_a_normal_clause_by_a_wide_margin(self):
        """The control. A ceiling that clips real speech is a worse bug."""
        config = InProcessTtsConfig()
        self.assertGreater(
            config.max_new_tokens,
            MEASURED_STEPS_PER_CLAUSE * 5,
            "the ceiling must not be anywhere near what a real clause costs",
        )

    def test_the_wall_clock_budget_leaves_the_session_room_to_react(self):
        config = InProcessTtsConfig()
        self.assertLess(
            config.max_generation_seconds,
            UNIT_DEADLINE_S,
            "the synthesis must give up BEFORE the session gives up on it, "
            "otherwise the thread is abandoned rather than stopped",
        )


class TestTheWallClockReachesTheDecodeLoop(unittest.TestCase):
    """The deadline has to arrive where the decoding happens."""

    @staticmethod
    def _detached() -> InProcessQwen3Tts:
        """A real instance without a checkpoint.

        `__init__` reads a checkpoint's geometry, which would make this a GPU
        test. `arm_generation_deadline` touches nothing but the config and the
        model it is handed, so the shipped method runs against a stand-in.
        """
        tts = object.__new__(InProcessQwen3Tts)
        tts.config = InProcessTtsConfig()
        return tts

    def test_the_deadline_lands_on_the_talker_that_decodes(self):
        from transformers import GenerationConfig

        talker = type("Talker", (), {"generation_config": GenerationConfig()})()
        model = type("Wrapper", (), {"model": type("Inner", (), {"talker": talker})()})()

        tts = self._detached()
        self.assertTrue(tts.arm_generation_deadline(model))
        self.assertEqual(
            talker.generation_config.max_time,
            tts.config.max_generation_seconds,
            "the wall-clock budget never reached the talker's own config, "
            "which is the only place transformers reads it from",
        )

    def test_a_checkpoint_without_a_talker_does_not_fail_the_turn(self):
        """Degrade, do not raise: an unarmed talker still has the deadline."""
        tts = self._detached()
        self.assertFalse(tts.arm_generation_deadline(object()))


class TestATalkerThatNeverStopsIsCutByTheClock(unittest.TestCase):
    """The proof, executed on the real transformers decode loop.

    Not a mock of the mechanism -- the mechanism. A tiny Llama on CPU with its
    EOS removed is the reduced form of the live failure: a model that will
    never stop on its own. The two arms differ in one field, the one this fix
    sets.
    """

    CEILING = 2000
    BUDGET_S = 0.15

    @staticmethod
    def _model_that_never_emits_eos():
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(0)
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=8192,
        )
        model = LlamaForCausalLM(config).eval()
        # The live failure, reduced: nothing will ever stop this generation
        # except a criterion imposed from outside.
        model.generation_config.eos_token_id = None
        model.generation_config.pad_token_id = 0
        model.generation_config.do_sample = False
        return model

    def _steps(self, model) -> int:
        import torch

        prompt = torch.tensor([[1, 2, 3]])
        output = model.generate(prompt, max_new_tokens=self.CEILING)
        return output.shape[1] - prompt.shape[1]

    def test_without_the_budget_it_runs_to_the_token_ceiling(self):
        """THE CAN-FAIL ARM. This is the state the live service was in.

        Without `max_time` the only thing that ends the generation is the
        token ceiling, which is precisely why 2048 tokens became six minutes
        of a card the user was waiting on.
        """
        model = self._model_that_never_emits_eos()
        self.assertEqual(
            self._steps(model),
            self.CEILING,
            "the reduction is wrong: this model was supposed to be unable to "
            "stop by itself, so the can-fail arm proves nothing",
        )

    def test_with_the_budget_the_clock_stops_it_first(self):
        model = self._model_that_never_emits_eos()
        model.generation_config.max_time = self.BUDGET_S

        started = time.monotonic()
        steps = self._steps(model)
        elapsed = time.monotonic() - started

        self.assertLess(
            steps,
            self.CEILING,
            "the generation still ran to its token ceiling, so "
            "generation_config.max_time is no longer the field transformers "
            "builds MaxTimeCriteria from -- the shipped fix is inert",
        )
        self.assertGreater(steps, 0, "it stopped before generating anything")
        # Asserted as a loose ceiling, not a window: the criterion is checked
        # between steps, so it overshoots by one step, and a loaded box makes
        # that step arbitrarily long. A test that fails under load teaches
        # people to ignore it.
        self.assertLess(
            elapsed,
            self.BUDGET_S + 5.0,
            "the budget was honoured so late that it was not a budget",
        )


class HangingTts:
    """A talker that accepts a unit and then never yields anything."""

    languages = (LANG_A, LANG_B)
    sample_rate = 16000
    busy = False
    min_reference_seconds = 0.0

    async def synthesize(self, *args, **kwargs):
        await asyncio.Event().wait()  # never set, on purpose
        yield  # pragma: no cover - unreachable, keeps this an async generator


class TestTheFailureNamesTheDeadlineThatFired(unittest.IsolatedAsyncioTestCase):
    """A message that names the wrong budget is worse than no message."""

    async def test_a_unit_stall_reports_the_unit_budget(self):
        session, _asr, _mt, _tts = make_session()
        session.tts = HangingTts()
        # The live shape: a unit budget far below the drain budget. The unit
        # deadline is what fires, and its TimeoutError reaches the drain's
        # handler through `await worker` -- which used to report the drain
        # constant for it.
        session.tts_unit_timeout_s = 1.0
        session.tts_drain_timeout_s = 30.0
        floor = session.journal.next_seq

        await run_conversation(session, [conversation_audio((VOICE_A_HZ, 1.4))])

        messages = [
            event.payload.get("message", "")
            for event in session.journal.since(floor)[0]
            if event.kind is EventKind.ERROR
        ]
        self.assertTrue(messages, "the turn died silently instead of loudly")
        self.assertTrue(
            any("within 1s" in message for message in messages),
            f"the failure did not name the budget that actually elapsed: "
            f"{messages}",
        )
        self.assertFalse(
            any("within 30s" in message for message in messages),
            f"a 1s abort was reported as a 30s one, which is the 2026-08-04 "
            f"misdirection: {messages}",
        )


class TestBusyStaysTrueWhileTheThreadLives(unittest.IsolatedAsyncioTestCase):
    """The lock follows the thread, not the coroutine that started it."""

    @staticmethod
    def _detached(release: threading.Event) -> InProcessQwen3Tts:
        tts = object.__new__(InProcessQwen3Tts)
        tts.config = InProcessTtsConfig()
        tts.sample_rate = tts.config.sample_rate
        tts.min_reference_seconds = tts.config.min_reference_seconds
        tts._languages = (LANG_A, LANG_B)
        tts._model = object()
        tts._lock = asyncio.Lock()
        tts._executor = None
        # Counts how many threads were inside the talker AT ONCE. The whole
        # point of the lock is that this never exceeds one.
        tts.inside = 0
        tts.peak_inside = 0
        guard = threading.Lock()

        def _generate(text, language, reference, reference_text,
                      sink=None, pacing=None):
            # Stands in for a talker that has not reached its EOS yet: this
            # thread cannot be cancelled from the event loop, exactly like the
            # real one.
            with guard:
                tts.inside += 1
                tts.peak_inside = max(tts.peak_inside, tts.inside)
            try:
                release.wait(timeout=30.0)
            finally:
                with guard:
                    tts.inside -= 1
            return np.zeros(int(tts.sample_rate * 0.5), dtype=np.float32)

        tts._generate = _generate
        return tts

    async def _drive(self, tts) -> None:
        reference = AudioChunk(np.zeros(int(24000 * 4.0), dtype=np.float32), 24000)
        async for _ in tts.synthesize("hola", LANG_B, reference, "", None):
            pass

    async def test_cancelling_the_await_does_not_free_the_talker(self):
        release = threading.Event()
        tts = self._detached(release)
        try:
            turn = asyncio.create_task(self._drive(tts))
            # Let the submit happen and the worker thread pick the job up.
            await asyncio.sleep(0.2)
            self.assertTrue(tts.busy, "the talker was never marked busy")

            # This is what the session's unit deadline does. It reaches the
            # coroutine; it cannot reach the thread.
            turn.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await turn

            self.assertTrue(
                tts.busy,
                "the talker reported free while its thread was still inside "
                "generate_voice_clone -- the lie that let the 2026-08-04 "
                "redrive start a second generation on the same model",
            )

            release.set()
            for _ in range(300):
                await asyncio.sleep(0.05)
                if not tts.busy:
                    break
            self.assertFalse(
                tts.busy,
                "the talker stayed locked after its thread finished, which "
                "would wedge every later turn",
            )
        finally:
            release.set()

    async def test_the_redrive_never_enters_the_talker_beside_the_first(self):
        """The consequence the lock exists for, asserted on the model.

        Not "did the second turn finish late" -- a second turn blocked on the
        same event finishes late either way, which would make this test pass
        against the very defect it is here for. The claim is narrower and is
        the one the live failure violated: two threads must never be inside
        `_generate` at the same moment.
        """
        release = threading.Event()
        tts = self._detached(release)
        try:
            first = asyncio.create_task(self._drive(tts))
            await asyncio.sleep(0.2)
            self.assertEqual(tts.inside, 1, "the first synthesis never began")

            # The unit deadline, reduced. The coroutine dies; the thread does
            # not.
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first

            # The automatic redrive, reduced: it arrives about a second later
            # and asks the same backend for the same line.
            second = asyncio.create_task(self._drive(tts))
            await asyncio.sleep(0.4)

            self.assertEqual(
                tts.peak_inside,
                1,
                "two generations were inside the talker at once -- the KV "
                "interleaving `synthesize` forbids, and what the 2026-08-04 "
                "redrive actually did",
            )

            release.set()
            await asyncio.wait_for(second, timeout=30.0)
        finally:
            release.set()


class TestTheTextIsWhatBoundsTheDecode(unittest.TestCase):
    """The second field round (2026-08-04, package client-20260804T141043Z).

    The ceiling and the wall clock above bounded the DAMAGE and did nothing
    about the cause: 119 s of synthesized audio for 13.6 s of user speech, two
    units pinned at the 70 s clock, `since_release_ms` 70683 and 70917.

    Reproduced off the phone with `scripts/translator/probe_tts_runaway.py` on
    CPU, and the reproduction killed the leading hypothesis on the way. The
    suspicion was a speaker merge, which had concatenated two profiles' clone
    references into one 7.74 s buffer just before the two bad turns. It is not
    that: in the log the first runaway unit entered `synthesizing` 496 ms
    BEFORE the merge frame was sent, nine units synthesized after the merge
    were healthy, and in the probe the two-speaker arms were the healthiest of
    the four (0 of 3 versus 1 of 3 for a single speaker). The reference is not
    the variable.

    What it is: the talker fails to emit its codec EOS on roughly one
    generation in ten, and the failure has two shapes -- one that keeps
    decoding to any ceiling offered (311 steps of continuous babble, measured
    per-second RMS 0.02-0.046 throughout, for the word "Gracias.") and one that
    does emit EOS eventually after producing far more than the text needs
    (102 steps for the same word against a 19-step median, the tail exactly
    zero). Neither is reachable by a constant token ceiling or a wall clock,
    because neither knows how much audio the TEXT is worth.
    """

    #: Measured with the probe on this checkpoint, Spanish clauses, shipped
    #: sampling: characters -> observed talker steps.
    MEASURED = {
        8: [11, 16, 16, 18, 19, 22, 22, 24, 28],     # "Gracias.", healthy runs
        63: [55, 56, 60],
        123: [89, 101, 110],
    }

    @staticmethod
    def _detached() -> InProcessQwen3Tts:
        """A real instance without a checkpoint.

        Only the codec's frame rate is read off the geometry, and it is the one
        number this file already asserts against (12.5 steps per second of
        audio, measured on the live service), so a stand-in carrying that one
        field keeps the test hermetic without pretending to be the checkpoint.
        """
        tts = object.__new__(InProcessQwen3Tts)
        tts.config = InProcessTtsConfig()
        tts.sample_rate = tts.config.sample_rate
        tts.geometry = type(
            "Geometry", (), {"frame_rate_hz": MEASURED_STEPS_PER_SECOND}
        )()
        return tts

    def test_the_budget_clears_every_healthy_generation_measured(self):
        tts = self._detached()
        for chars, samples in self.MEASURED.items():
            budget = tts.step_budget("x" * chars)
            self.assertGreater(
                budget,
                max(samples),
                f"a {chars}-character clause was measured at up to "
                f"{max(samples)} steps and the budget is {budget} -- a budget "
                "below real speech truncates the translation, which is a worse "
                "defect than the one it fixes",
            )

    def test_the_budget_cuts_the_failures_that_were_measured(self):
        tts = self._detached()
        # The two observed failures for the 8-character word.
        budget = tts.step_budget("Gracias.")
        for observed in (102, 311):
            self.assertLess(
                budget,
                observed,
                f"{observed} steps for one word must not be inside the budget",
            )

    def test_the_budget_grows_with_the_text_and_never_leaves_the_ceiling(self):
        tts = self._detached()
        self.assertLess(tts.step_budget("hola"), tts.step_budget("hola " * 20))
        self.assertLessEqual(
            tts.step_budget("x" * 100000),
            tts.config.max_new_tokens,
            "the text-derived budget must stay under the absolute ceiling",
        )
        self.assertGreaterEqual(tts.step_budget(""), 8)

    def test_a_runaway_is_redrawn_and_the_healthy_draw_is_kept(self):
        """The retry is what makes this a repair rather than a shorter leash."""
        tts = self._detached()
        rate = tts.sample_rate
        budget = tts.step_budget("Gracias.")
        runaway = np.full(int(budget / 12.5 * rate), 0.05, dtype=np.float32)
        healthy = np.full(int(20 / 12.5 * rate), 0.05, dtype=np.float32)
        drawn = []

        def fake(text, language, reference, reference_text, max_new_tokens,
                 emitter=None):
            drawn.append(max_new_tokens)
            return runaway if len(drawn) == 1 else healthy

        tts._generate_once = fake
        out = tts._generate("Gracias.", "es", None, "")
        self.assertEqual(len(drawn), 2, "the runaway draw was not re-drawn")
        self.assertEqual(drawn, [budget, budget],
                         "both draws must carry the text's budget")
        self.assertEqual(len(out), len(healthy),
                         "the healthy re-draw is the one the listener hears")

    def test_a_healthy_generation_is_drawn_once_and_returned_unchanged(self):
        """The control: nothing about a normal turn may change."""
        tts = self._detached()
        speech = np.full(int(1.5 * tts.sample_rate), 0.05, dtype=np.float32)
        drawn = []

        def fake(text, language, reference, reference_text, max_new_tokens,
                 emitter=None):
            drawn.append(max_new_tokens)
            return speech

        tts._generate_once = fake
        out = tts._generate("Gracias.", "es", None, "")
        self.assertEqual(len(drawn), 1, "a healthy draw must not be repeated")
        np.testing.assert_array_equal(out, speech)

    def test_both_draws_failing_still_returns_the_words(self):
        """A truncated utterance opens with the speech; babble is in the tail."""
        tts = self._detached()
        budget = tts.step_budget("Gracias.")
        runaway = np.full(int(budget / 12.5 * tts.sample_rate), 0.05, dtype=np.float32)
        calls = []

        def fake(text, language, reference, reference_text, max_new_tokens,
                 emitter=None):
            calls.append(max_new_tokens)
            return runaway

        tts._generate_once = fake
        out = tts._generate("Gracias.", "es", None, "")
        self.assertEqual(len(calls), tts.config.runaway_retries + 1)
        self.assertEqual(len(out), len(runaway))

    def test_trailing_digital_silence_is_cut_and_the_speech_is_not(self):
        tts = self._detached()
        rate = tts.sample_rate
        speech = np.full(int(2.0 * rate), 0.05, dtype=np.float32)
        # The measured shape: two seconds of speech, five of exact zero.
        padded = np.concatenate([speech, np.zeros(int(5.0 * rate), dtype=np.float32)])
        out = tts.trim_trailing_silence(padded)
        kept_tail = (len(out) - len(speech)) / rate
        self.assertAlmostEqual(kept_tail, tts.config.trailing_silence_keep_s, places=3)
        np.testing.assert_array_equal(out[: len(speech)], speech)

    def test_leading_and_interior_silence_survive(self):
        """Timing inside the utterance is not this method's business."""
        tts = self._detached()
        rate = tts.sample_rate
        lead = np.zeros(int(1.0 * rate), dtype=np.float32)
        gap = np.zeros(int(0.5 * rate), dtype=np.float32)
        word = np.full(int(0.5 * rate), 0.05, dtype=np.float32)
        out = tts.trim_trailing_silence(np.concatenate([lead, word, gap, word]))
        self.assertEqual(len(out), len(lead) + len(word) + len(gap) + len(word))

    def test_an_entirely_silent_result_is_left_for_the_session_gate(self):
        tts = self._detached()
        quiet = np.zeros(int(1.0 * tts.sample_rate), dtype=np.float32)
        np.testing.assert_array_equal(tts.trim_trailing_silence(quiet), quiet)


if __name__ == "__main__":
    unittest.main()
