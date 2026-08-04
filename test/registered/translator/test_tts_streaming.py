# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Incremental emission: start early, and still never gap.

THE DEFECT (2026-08-04, `/spinning/466-client-logs/MEASURE_TTS_LATENCY.md`).
The reference generates a complete unit and only then emits it, so
time-to-first-audio equalled full generation time by construction: in the field
session `ea7531b498c6` the whole utterance arrived as a 199 ms burst of 14
frames, 6866 ms after the translation was ready. TTS was 93 % of the wait, and
none of it was setup -- the fixed cost is 79 ms and the rest is ~74
autoregressive steps at 92.6 ms each.

What makes this fixable is that codec decode is 36 ms and FLAT across a 6x
range of utterance length, so audio can be taken out in pieces for a constant
charge. What makes it hard is the other measured number:

    RTF = 12.5 frames/s needed / 10.5 steps/s produced = 1.20

**generation is slower than playback.** Emitting at frame 1 would let playback
overtake the generator and starve before the end, so the bar's two halves --
"starts within 3 s" and "then runs continuously" -- pull against each other and
are satisfied together only by a pre-roll sized from the inequality

    P >= (R - 1) x D

This file pins the three things that have to be true at once: that the pre-roll
is derived rather than guessed, that the stream never underruns a playback
clock, and that the wire carries exactly the samples the burst path would have
sent -- no more, no fewer, in order. What those samples are WORTH, i.e. how
close a prefix decode is to a one-shot decode, is a property of the vendored
codec and is measured on the card by `probe_decode_strategies.py`; see
`TestWhatGoesOnTheWireIsWhatTheBurstWouldHaveSent` for the number and why it
cannot be pinned here.

    CUDA_VISIBLE_DEVICES=99 python -m pytest \\
      test/registered/translator/test_tts_streaming.py -v
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sglang.srt.translator.backends import TurnPacing  # noqa: E402
from sglang.srt.translator.inprocess_tts import (  # noqa: E402
    CODEC_FRAME_RATE_HZ,
    FIXED_CALL_SECONDS,
    InProcessQwen3Tts,
    InProcessTtsConfig,
    _UnitEmitter,
    release_frame,
)

#: Measured on the RTX 5090 on 2026-08-04 (MEASURE_TTS_LATENCY.md section 5).
RTF_IDLE = 1.158
RTF_27B_C1 = 1.215
RTF_27B_SATURATED = 1.256
STEPS_PER_S_IDLE = 10.79
STEPS_PER_S_SATURATED = 9.95
#: The field turn: 73.3 steps, 5.86 s of audio generated, 5.6 s emitted.
FIELD_TURN_FRAMES = 73.3
#: 3000 ms bar - 560 ms upstream (ASR 151 + diarization 57 + MT 325 +
#: dispatch 27) - 79 ms of TTS fixed cost. This is what is left for the
#: PRE-ROLL itself, i.e. for generating frames.
PREROLL_BUDGET_MS = 2361.0
#: What the same arithmetic allows measured from the synthesizer's own start,
#: which is where `tts_first_audio_ms` is taken: the pre-roll plus the 79 ms of
#: prefill, encodes and first decode that precede any sample. Assertions about
#: an observed time-to-first-audio use THIS; assertions about the pre-roll
#: budget use the line above. Confusing the two is a 79 ms error in the
#: direction that reads as a pass.
TTS_BUDGET_MS = PREROLL_BUDGET_MS + FIXED_CALL_SECONDS * 1000.0


class _ListSink:
    """Collects what the emitter decides to send, in order."""

    def __init__(self) -> None:
        self.chunks = []

    def push(self, chunk) -> None:
        self.chunks.append(chunk)

    def audio(self) -> np.ndarray:
        if not self.chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate([c.samples for c in self.chunks])


def _detached(**overrides) -> InProcessQwen3Tts:
    """A real instance without a checkpoint.

    Nothing exercised here touches a weight: the emission policy is arithmetic
    over frame counts and a waveform, which is exactly why it can be pinned
    without a card.
    """
    tts = object.__new__(InProcessQwen3Tts)
    tts.config = InProcessTtsConfig(**overrides)
    tts.sample_rate = tts.config.sample_rate
    tts.min_reference_seconds = tts.config.min_reference_seconds
    return tts


def _emitter(tts, sink, expected_frames, ceiling=800, streaming=True):
    return _UnitEmitter(tts, sink, None, expected_frames, ceiling, streaming)


class TestThePreRollComesFromTheInequality(unittest.TestCase):
    """`P >= (R - 1) x D`, not a constant somebody liked the look of."""

    def test_a_faster_card_needs_less_pre_roll(self):
        idle = release_frame(FIELD_TURN_FRAMES, RTF_IDLE, 2.5, 5.0, 800)
        loaded = release_frame(FIELD_TURN_FRAMES, RTF_27B_SATURATED, 2.5, 5.0, 800)
        self.assertLess(
            idle, loaded,
            "the pre-roll did not respond to the real-time factor at all, so "
            "it is a constant wearing a formula's clothes -- and the 7.8 % "
            "co-tenancy spread is precisely what it exists to track",
        )

    def test_a_longer_turn_needs_more_pre_roll(self):
        short = release_frame(20.0, RTF_27B_C1, 2.5, 5.0, 800)
        long = release_frame(200.0, RTF_27B_C1, 2.5, 5.0, 800)
        self.assertLess(short, long)

    def test_generation_at_or_faster_than_playback_needs_no_bank(self):
        """The state lever (d) is aiming at: R below 1.0, ceiling gone.

        At 12.5 steps/s -- a 1.26x speedup on the saturated measurement -- the
        deficit disappears and the pre-roll should collapse to the mechanical
        minimum rather than divide by a non-positive number.
        """
        for rtf in (1.0, 0.9):
            frames = release_frame(FIELD_TURN_FRAMES, rtf, 2.5, 5.0, 800)
            self.assertLessEqual(
                frames, 8,
                f"at RTF {rtf} playback cannot overtake the generator, so "
                f"holding {frames} frames is latency spent on nothing",
            )

    def test_the_pre_roll_never_exceeds_the_generation_it_is_waiting_for(self):
        """A hold longer than the generation would restore the burst."""
        ceiling = 30
        self.assertLessEqual(
            release_frame(10_000.0, 2.0, 2.5, 5.0, ceiling), ceiling
        )

    def test_the_field_turn_clears_the_bar_on_both_measured_cards(self):
        """The acceptance arithmetic, on the turn the field log actually has.

        Not a round number: 73.3 frames is the step count implied for
        `83162aef0055` unit 0 by `call_ms = 81 + 92.6 x steps`.
        """
        for rtf, rate in (
            (RTF_IDLE, STEPS_PER_S_IDLE),
            (RTF_27B_SATURATED, STEPS_PER_S_SATURATED),
        ):
            frames = release_frame(FIELD_TURN_FRAMES, rtf, 2.5, 5.0, 800)
            wall_ms = (FIXED_CALL_SECONDS + frames / rate) * 1000.0
            self.assertLess(
                wall_ms, PREROLL_BUDGET_MS,
                f"at RTF {rtf} the pre-roll costs {wall_ms:.0f} ms of the "
                f"{PREROLL_BUDGET_MS:.0f} ms the 3 s bar leaves after ASR, "
                f"diarization, MT and dispatch -- so the turn starts late",
            )

    def test_the_burst_path_is_what_it_is_measured_against(self):
        """THE CAN-FAIL ARM: the same turn, generated whole, misses the bar.

        Without this the numbers above prove only that some quantity is small.
        """
        whole_ms = (FIXED_CALL_SECONDS + FIELD_TURN_FRAMES / STEPS_PER_S_IDLE) * 1000.0
        self.assertGreater(
            whole_ms, PREROLL_BUDGET_MS,
            "generating this turn whole fits the budget, so the field's "
            "6866 ms is not reproduced by the model and nothing above is "
            "evidence of an improvement",
        )


class TestTheStreamNeverUnderrunsThePlaybackClock(unittest.TestCase):
    """The half of the bar that a fast start is not evidence for.

    "Starts within 3 s" is one sample; "runs continuously to the end" is every
    sample after it, and with R > 1 the buffer only ever shrinks -- so the
    dangerous moment is the LAST one. These drive the shipped emitter through a
    whole utterance against a playback clock and assert on the minimum.
    """

    def _run(self, frames_total, rate, expected_frames=None, config=None):
        """Simulate one unit end to end. Returns (min buffer s, ttfa s).

        The generation is modelled as one frame every `1/rate` seconds after a
        fixed cost, which is the measured shape: `call_ms = 81 + 92.6 x steps`.
        Playback starts when the first chunk is emitted and consumes one second
        of audio per wall second from then on, which is what the client's
        cursor scheduler does.
        """
        tts = _detached(**(config or {}))
        sink = _ListSink()
        emitter = _emitter(tts, sink, expected_frames or float(frames_total))
        samples_per_frame = int(tts.sample_rate / CODEC_FRAME_RATE_HZ)
        # Full-scale speech throughout: the trailing-silence rule is a separate
        # claim with its own tests, and letting it fire here would silently
        # shorten the utterance this one is about.
        speech = np.full(frames_total * samples_per_frame, 0.05, dtype=np.float32)

        first_emit = None
        worst = math.inf
        for frame in range(1, frames_total + 1):
            now = FIXED_CALL_SECONDS + frame / rate
            if not emitter.may_release(frame, now):
                continue
            emitter.offer(speech[: frame * samples_per_frame])
            if emitter.emitted and first_emit is None:
                first_emit = now
            if first_emit is None:
                continue
            buffered = emitter.emitted / tts.sample_rate - (now - first_emit)
            worst = min(worst, buffered)
        emitter.offer(speech, final=True)
        end = FIXED_CALL_SECONDS + frames_total / rate
        worst = min(worst, emitter.emitted / tts.sample_rate - (end - first_emit))
        return worst, first_emit

    def test_the_field_turn_is_gapless_on_an_idle_card(self):
        worst, ttfa = self._run(73, STEPS_PER_S_IDLE)
        self.assertGreaterEqual(
            worst, 0.0,
            f"playback ran dry by {-worst:.3f}s before the utterance ended: "
            "the stream started early enough and finished badly, which is the "
            "failure the inequality exists to prevent",
        )
        self.assertLess(ttfa, TTS_BUDGET_MS / 1000.0)

    def test_the_field_turn_is_gapless_with_the_27b_saturated(self):
        worst, ttfa = self._run(73, STEPS_PER_S_SATURATED)
        self.assertGreaterEqual(worst, 0.0, f"underran by {-worst:.3f}s")
        self.assertLess(ttfa, TTS_BUDGET_MS / 1000.0)

    def test_a_turn_near_the_crossover_is_gapless(self):
        """The longest turn that still clears both halves of the bar.

        The ranking put the saturated crossover at 9.2 s of speech. That figure
        assumed audio can be emitted the instant it exists; a real emitter
        cannot send a chunk that has not filled, and that backlog is charged to
        the pre-roll. The honest crossover for THIS implementation is therefore
        shorter -- ~93 frames, 7.4 s -- and 90 frames is the arm that has to
        hold. The interesting one, because the pre-roll is largest exactly
        where the budget is nearly spent.
        """
        worst, ttfa = self._run(90, STEPS_PER_S_SATURATED)
        self.assertGreaterEqual(worst, 0.0, f"underran by {-worst:.3f}s")
        self.assertLess(
            ttfa, TTS_BUDGET_MS / 1000.0,
            "a turn at the crossover started later than the budget allows",
        )

    def test_beyond_the_crossover_the_start_is_kept_and_the_gap_is_reported(self):
        """The limit of the lever, asserted rather than left as prose.

        Past the crossover the two halves of the bar cannot both hold: the
        deficit grows with duration while the budget does not. The choice made
        here is to keep the start time -- the half the user stated as a number
        -- and to log the projected shortfall. This pins that the choice is
        actually made, and in that direction, so a long monologue degrades to
        "starts on time, may gap" rather than "starts whenever".

        Lifting this needs per-step compute (lever (d), target 1.26x), not
        emission work. It is the reason (a) alone does not ship a monologue.
        """
        worst, ttfa = self._run(200, STEPS_PER_S_SATURATED)
        self.assertLess(
            ttfa, TTS_BUDGET_MS / 1000.0,
            "a long turn was allowed to spend more than the budget on its "
            "pre-roll, so it misses the 3 s bar AND still gaps",
        )
        self.assertLess(
            worst, 0.0,
            "a 16 s turn did not underrun at RTF 1.256, so the crossover this "
            "test claims to sit beyond is not where the arithmetic says",
        )

    def test_emitting_from_the_first_frame_does_underrun(self):
        """THE CAN-FAIL ARM. Without the hold, the same turn starves.

        This is the arm that makes the three above mean something: the
        simulation is capable of reporting an underrun, so a non-negative
        minimum is a result rather than a property of the harness.
        """
        worst, _ttfa = self._run(
            73, STEPS_PER_S_IDLE, config={"preroll_budget_seconds": 0.001},
        )
        self.assertLess(
            worst, 0.0,
            "a stream that starts at the first chunk did NOT run dry, so "
            "either R <= 1 in this simulation or the clock is not moving -- "
            "and every gapless claim above rests on this arm failing",
        )

    def test_a_late_clause_lengthens_a_pre_roll_that_has_not_committed(self):
        """MT can still be producing while the first unit banks its audio.

        The pre-roll is re-derived on every frame precisely so that a clause
        arriving during the hold is charged for. Once emission has begun it
        cannot be, which is stated as the cost rather than hidden.
        """
        tts = _detached()
        alone = release_frame(
            tts.expected_turn_frames("x" * 60, None), RTF_IDLE, 2.5, 5.0, 800
        )
        pacing = TurnPacing()
        pacing.note_queued("y" * 60)
        with_more = release_frame(
            tts.expected_turn_frames("x" * 60, pacing), RTF_IDLE, 2.5, 5.0, 800
        )
        self.assertGreater(
            with_more, alone,
            "a second clause already queued behind this one did not lengthen "
            "the pre-roll, so the turn banks enough audio to reach the end of "
            "clause one and nothing for clause two -- which moves the gap to "
            "the seam instead of removing it",
        )


class TestWhatGoesOnTheWireIsWhatTheBurstWouldHaveSent(unittest.TestCase):
    """The emitter sends every sample the burst path would, and no other.

    SCOPE, because an earlier draft of this file claimed more than it proved.
    These pin the emitter's SLICING: which sample positions are released, in
    what order, and that the concatenation closes on exactly the sample
    `trim_trailing_silence` would have ended on. They say nothing about the
    VALUES, which come from decoding a prefix of the codes.

    The values are not identical to a one-shot decode and cannot be made so.
    `probe_prefix_decode.py` and `probe_decode_strategies.py` measured why: the
    codec decoder reaches 25-50 frames forward in time, so a decode that has
    not seen the end of the utterance differs from one that has -- by 39.4 dB
    on a signal of RMS 0.068, with chunk seams of 0.0427 against the one-shot
    decode's own 0.0411 at the same positions. Exactness would mean holding
    back four seconds of audio, i.e. abandoning the feature. That is a GPU
    fact, so it is measured there and named here rather than asserted in a
    hermetic test that cannot see a decoder.
    """

    def _stream(self, waveform, tts=None, expected_frames=8.0):
        tts = tts or _detached()
        sink = _ListSink()
        emitter = _emitter(tts, sink, expected_frames)
        samples_per_frame = int(tts.sample_rate / CODEC_FRAME_RATE_HZ)
        total = len(waveform) // samples_per_frame
        for frame in range(1, total + 1):
            now = FIXED_CALL_SECONDS + frame / STEPS_PER_S_IDLE
            if emitter.may_release(frame, now):
                emitter.offer(waveform[: frame * samples_per_frame])
        trimmed = tts.trim_trailing_silence(waveform)
        emitter.offer(trimmed, final=True)
        return sink.audio(), trimmed

    def test_the_stream_concatenates_to_the_trimmed_waveform(self):
        rate = 24000
        speech = np.full(int(4.0 * rate), 0.05, dtype=np.float32)
        streamed, trimmed = self._stream(speech)
        np.testing.assert_array_equal(
            streamed, trimmed,
            "the concatenated stream does not cover the same samples the "
            "burst path returns: the emitter dropped, duplicated or reordered "
            "audio, which no amount of decode fidelity would repair",
        )

    def test_trailing_silence_is_never_sent_in_the_first_place(self):
        """The measured runaway tail: two seconds of speech, five of zero.

        `trim_trailing_silence` cuts it from a finished waveform. A stream has
        no finished waveform to cut, so the same expression is evaluated on the
        prefix and the samples are simply never emitted.
        """
        rate = 24000
        speech = np.full(int(2.0 * rate), 0.05, dtype=np.float32)
        padded = np.concatenate(
            [speech, np.zeros(int(5.0 * rate), dtype=np.float32)]
        )
        streamed, trimmed = self._stream(padded)
        np.testing.assert_array_equal(streamed, trimmed)
        self.assertLess(
            len(streamed) / rate, 2.5,
            "five seconds of digital silence reached the wire; the listener "
            "waits through all of it and the next unit cannot start",
        )

    def test_leading_and_interior_silence_survive_the_stream(self):
        """Timing inside an utterance is not the emitter's business either."""
        rate = 24000
        lead = np.zeros(int(1.0 * rate), dtype=np.float32)
        gap = np.zeros(int(0.5 * rate), dtype=np.float32)
        word = np.full(int(0.8 * rate), 0.05, dtype=np.float32)
        waveform = np.concatenate([lead, word, gap, word])
        streamed, trimmed = self._stream(waveform)
        np.testing.assert_array_equal(streamed, trimmed)
        self.assertEqual(len(streamed), len(waveform))

    def test_an_entirely_silent_unit_still_reaches_the_session_gate(self):
        """Silence that arrives on time is the worst failure shape there is.

        The session's non-silence gate reports it. The emitter must therefore
        not quietly swallow a silent unit on the way -- an emptied buffer would
        hide exactly the defect the gate exists to name.
        """
        rate = 24000
        quiet = np.zeros(int(2.0 * rate), dtype=np.float32)
        streamed, trimmed = self._stream(quiet)
        np.testing.assert_array_equal(streamed, trimmed)
        self.assertEqual(len(streamed), len(quiet))

    def test_the_wire_sees_the_same_number_of_buffers_as_before(self):
        """Every hold counter on the client is per-BUFFER, not per-sample.

        `holdDeferred`, `holdDrained`, `pushed`, `scheduled` and the
        `audio_buffers_*` telemetry all count calls, so a finer chunk would
        silently re-baseline all of them. 0.4 s is exactly 5 codec frames, and
        keeping it is what makes this change invisible to them.
        """
        tts = _detached()
        rate = tts.sample_rate
        speech = np.full(int(4.0 * rate), 0.05, dtype=np.float32)
        sink = _ListSink()
        emitter = _emitter(tts, sink, 8.0)
        samples_per_frame = int(rate / CODEC_FRAME_RATE_HZ)
        for frame in range(1, len(speech) // samples_per_frame + 1):
            now = FIXED_CALL_SECONDS + frame / STEPS_PER_S_IDLE
            if emitter.may_release(frame, now):
                emitter.offer(speech[: frame * samples_per_frame])
        emitter.offer(speech, final=True)
        span = int(tts.config.emit_chunk_seconds * rate)
        self.assertEqual(
            len(sink.chunks), math.ceil(len(speech) / span),
            "the unit reached the wire as a different number of buffers than "
            "the burst path would have sent, which re-baselines every "
            "per-buffer counter the client reports",
        )


class TestTheRedrawAndTheStreamAreExclusive(unittest.TestCase):
    """A re-draw can only replace audio nobody has heard.

    #564 re-draws a generation that overruns the text's budget. Once a chunk is
    on the wire it cannot be recalled, so the listener would get the opening of
    a runaway followed by a different take of the same clause. The two are
    mutually exclusive and this pins where the line is drawn and that it holds.
    """

    @staticmethod
    def _tts(**overrides) -> InProcessQwen3Tts:
        tts = _detached(**overrides)
        tts.geometry = type("Geometry", (), {"frame_rate_hz": CODEC_FRAME_RATE_HZ})()
        return tts

    def test_a_short_clause_is_generated_whole_and_keeps_its_redraw(self):
        """"Gracias." -- the clause the runaway was actually observed on.

        Eight characters, ~20 frames, under two seconds of generation: it does
        not need to be streamed to start in time, so it is not, and the re-draw
        survives exactly where the bar does not forbid it.
        """
        tts = self._tts()
        self.assertFalse(
            tts._should_stream("Gracias.", None),
            "a clause that generates whole inside the pre-roll budget was "
            "streamed anyway, which gives up the re-draw for nothing",
        )

    def test_a_normal_clause_is_streamed(self):
        tts = self._tts()
        self.assertTrue(tts._should_stream("x" * 63, None))

    def test_a_turn_that_has_started_speaking_streams_whatever_the_size(self):
        """A unit generated whole mid-turn is a hole as long as itself."""
        tts = self._tts()
        pacing = TurnPacing()
        pacing.started = True
        self.assertTrue(tts._should_stream("Gracias.", pacing))

    def test_a_runaway_with_nothing_emitted_is_still_redrawn(self):
        """The #564 control: the repair must survive where it can."""
        tts = self._tts()
        rate = tts.sample_rate
        budget = tts.step_budget("Gracias.")
        runaway = np.full(
            int(budget / CODEC_FRAME_RATE_HZ * rate), 0.05, dtype=np.float32
        )
        healthy = np.full(
            int(20 / CODEC_FRAME_RATE_HZ * rate), 0.05, dtype=np.float32
        )
        drawn = []

        def fake(text, language, reference, reference_text, max_new_tokens,
                 emitter=None):
            drawn.append(max_new_tokens)
            return runaway if len(drawn) == 1 else healthy

        tts._generate_once = fake
        out = tts._generate("Gracias.", "es", None, "")
        self.assertEqual(len(drawn), 2, "the runaway draw was not re-drawn")
        self.assertEqual(len(out), len(healthy))

    def test_a_runaway_that_has_already_spoken_is_truncated_not_redrawn(self):
        """The conflict, at the moment it actually bites.

        The fake emits while it generates, exactly as the streaming path does,
        and then overruns. Re-drawing here would play the listener the opening
        of the runaway and then a different take of the same words.
        """
        tts = self._tts()
        rate = tts.sample_rate
        budget = tts.step_budget("x" * 63)
        runaway = np.full(
            int(budget / CODEC_FRAME_RATE_HZ * rate), 0.05, dtype=np.float32
        )
        drawn = []

        def fake(text, language, reference, reference_text, max_new_tokens,
                 emitter=None):
            drawn.append(max_new_tokens)
            # What a streamed generation has done by the time it overruns.
            emitter.offer(runaway[: int(1.0 * rate)])
            return runaway

        tts._generate_once = fake
        sink = _ListSink()
        out = tts._generate("x" * 63, "es", None, "", sink, TurnPacing())
        self.assertEqual(
            len(drawn), 1,
            "the generation was re-drawn after audio had already been sent, "
            "so the listener hears the start of a runaway and then a second, "
            "different rendering of the same clause",
        )
        self.assertEqual(len(out), len(runaway))
        np.testing.assert_array_equal(
            sink.audio(), out,
            "the truncated unit did not reach the wire in one piece",
        )

    def test_the_stream_is_a_prefix_of_the_result_at_every_moment(self):
        """The property the whole resolution rests on.

        Emission is irrevocable, so whatever has been sent must remain a
        correct prefix of whatever the unit eventually returns. If that ever
        fails, no gating of the re-draw can save it.
        """
        tts = self._tts()
        rate = tts.sample_rate
        speech = np.full(int(3.0 * rate), 0.05, dtype=np.float32)

        def fake(text, language, reference, reference_text, max_new_tokens,
                 emitter=None):
            for cut in (0.8, 1.6, 2.4):
                emitter.offer(speech[: int(cut * rate)])
            return speech

        tts._generate_once = fake
        sink = _ListSink()
        out = tts._generate("x" * 63, "es", None, "", sink, TurnPacing())
        streamed = sink.audio()
        np.testing.assert_array_equal(streamed[: len(out)], out[: len(streamed)])
        np.testing.assert_array_equal(streamed, out)


class TestStoppingOneTurnIsANarrowingNotANewPath(unittest.TestCase):
    """The seam a per-turn stop and a re-synthesis will land on.

    Nothing here is a feature. `playback.stop` is global today -- it clears the
    whole queue and cancels the one drain task -- but the half that reaches the
    CLIENT is already per turn: `abort_playback` returns an `aborted_turn_id`,
    and its docstring records that the client discards audio per `turn_id` and
    that in-flight frames arrive after the event, which is what makes "drop
    frames whose turn is aborted" sufficient. So the missing half is server
    side, and these pin that the emitter does not stand in its way.

    They exist because the cost of getting this wrong is asymmetric: a stop
    that has to be retrofitted into an emitter which assumed it would always
    run to completion is a rewrite, and an emitter that merely reads one flag
    is not.
    """

    @staticmethod
    def _emitter_with(pacing):
        tts = _detached()
        sink = _ListSink()
        return tts, sink, _UnitEmitter(tts, sink, pacing, 60.0, 800, True)

    def test_a_cancelled_turn_stops_putting_audio_on_the_wire(self):
        pacing = TurnPacing(turn_id="t1", target="es")
        tts, sink, emitter = self._emitter_with(pacing)
        speech = np.full(int(3.0 * tts.sample_rate), 0.05, dtype=np.float32)
        emitter.released = True
        emitter.offer(speech[: int(1.0 * tts.sample_rate)])
        sent_before = emitter.emitted
        self.assertGreater(sent_before, 0, "nothing was streamed to begin with")

        pacing.cancel()
        emitter.offer(speech)
        emitter.offer(speech, final=True)
        self.assertEqual(
            emitter.emitted, sent_before,
            "audio kept reaching the wire after the turn was cancelled, so a "
            "per-turn stop would have to reach into the talker thread instead "
            "of the emitter -- and that thread cannot be cancelled at all",
        )

    def test_what_was_already_sent_is_not_taken_back(self):
        """A stop supersedes; it cannot un-send.

        The same boundary the re-draw obeys, which is the point: one answer,
        read by both.
        """
        pacing = TurnPacing(turn_id="t1")
        tts, sink, emitter = self._emitter_with(pacing)
        speech = np.full(int(2.0 * tts.sample_rate), 0.05, dtype=np.float32)
        emitter.released = True
        emitter.offer(speech[: int(0.8 * tts.sample_rate)])
        heard = sink.audio().copy()
        pacing.cancel()
        emitter.offer(speech, final=True)
        np.testing.assert_array_equal(sink.audio(), heard)
        self.assertTrue(pacing.started, "the point of no return was cleared")

    def test_the_point_of_no_return_is_recorded_on_the_turn_not_the_unit(self):
        """One flag, set where a chunk becomes reachable, read by everything.

        `started` has to live on the TURN: a re-draw asks about the unit in
        hand, but "may this turn's audio still be replaced" is a question about
        every unit of it, and a per-unit answer would say yes for clause two of
        a turn whose clause one is already playing.
        """
        pacing = TurnPacing(turn_id="t1")
        tts, sink, emitter = self._emitter_with(pacing)
        self.assertFalse(pacing.started)
        emitter.released = True
        emitter.offer(np.full(int(1.0 * tts.sample_rate), 0.05, dtype=np.float32))
        self.assertTrue(pacing.started)
        self.assertGreater(pacing.emitted_audio_s, 0.0)

    def test_a_second_generation_of_the_same_turn_is_expressible(self):
        """Nothing is keyed by turn_id, so re-synthesis is not blocked.

        The check that matters for the regeneration feature: the synthesizer
        holds no cache, registry or per-turn state that outlives one call, so
        a second generation is a second `TurnPacing` carrying the same
        `turn_id` and a higher `generation` -- distinguishable on the wire,
        and starting from a clean point of no return.
        """
        first = TurnPacing(turn_id="t1", target="es", generation=0)
        first.started = True
        first.emitted_audio_s = 4.4
        second = TurnPacing(turn_id="t1", target="es", generation=1)
        self.assertEqual(second.turn_id, first.turn_id)
        self.assertNotEqual(second.generation, first.generation)
        self.assertFalse(
            second.started,
            "a re-synthesis inherited the first generation's point of no "
            "return, so it would refuse to emit its own audio",
        )
        self.assertFalse(second.cancelled)

    def test_the_synthesizer_holds_no_state_keyed_by_turn(self):
        """The structural half of the claim above, asserted on the module.

        A registry keyed by turn would be the thing that makes a second
        generation of one turn impossible, and it is exactly the kind of thing
        that gets added later for a good local reason.
        """
        import sglang.srt.translator.inprocess_tts as module

        source = Path(module.__file__).with_suffix(".py").read_text()
        self.assertNotIn(
            "turn_id", source,
            "the synthesizer began keying state by turn_id; a second "
            "generation of the same turn may no longer be expressible",
        )


class TestTheRealTimeFactorIsMeasuredNotAssumed(unittest.TestCase):
    """The one detail salvaged from the withdrawn contention-control lever."""

    def test_the_first_generation_of_a_process_uses_the_pessimistic_default(self):
        """1.256 is the saturated measurement, not the idle one.

        An RTF guessed too low under-banks and gaps; one guessed too high only
        spends latency. The first generation has no evidence at all, so it gets
        the arm that fails safely.
        """
        tts = _detached()
        self.assertEqual(tts.observed_rtf(), RTF_27B_SATURATED)
        self.assertGreater(tts.config.default_rtf, RTF_IDLE)

    def test_a_live_generation_reports_the_rate_it_is_actually_running_at(self):
        """A card slower than the carried estimate is believed immediately."""
        tts = _detached()
        slow = 8.0
        frames = 40
        elapsed = FIXED_CALL_SECONDS + frames / slow
        self.assertAlmostEqual(
            tts.observed_rtf(frames, elapsed),
            CODEC_FRAME_RATE_HZ / slow,
            places=3,
        )

    def test_the_pre_release_measurement_cannot_lower_the_carried_estimate(self):
        """The bias found by the first GPU run, pinned so it cannot return.

        Nothing is decoded before the release point, so the live rate is the
        BARE talker rate -- while every frame after release is charged a codec
        decode on the same thread. Sizing the pre-roll on the pre-release rate
        therefore under-banks by exactly what the emission is about to cost:
        `probe_stream_emission.py` underran by 0.39-1.57 s on every streamed
        arm until the carried whole-generation rate was made a floor.
        """
        tts = _detached()
        tts._rtf_ema = RTF_27B_SATURATED
        frames = 40
        # A bare-talker rate well below the carried whole-call figure, which is
        # the situation on every single generation.
        elapsed = FIXED_CALL_SECONDS + frames / STEPS_PER_S_IDLE
        self.assertGreater(
            CODEC_FRAME_RATE_HZ / STEPS_PER_S_IDLE, 0.0,
        )
        self.assertEqual(
            tts.observed_rtf(frames, elapsed), RTF_27B_SATURATED,
            "the optimistic pre-release rate was allowed to lower the "
            "estimate, which is the under-banking the first probe run "
            "measured as a gap in the middle of every streamed turn",
        )

    def test_the_fixed_cost_is_removed_before_the_rate_is_taken(self):
        """Otherwise the estimate is a rate PLUS a constant.

        At the handful of frames where the estimate first becomes available,
        the measured 79 ms of prefill and encodes is a large share of the
        elapsed time; leaving it in reads as a contended card and buys a
        pre-roll nobody needed.
        """
        tts = _detached()
        slow = 8.0
        frames = tts.MIN_FRAMES_FOR_RATE
        elapsed = FIXED_CALL_SECONDS + frames / slow
        corrected = tts.observed_rtf(frames, elapsed)
        naive = CODEC_FRAME_RATE_HZ / (frames / elapsed)
        self.assertLess(corrected, naive)
        self.assertAlmostEqual(corrected, CODEC_FRAME_RATE_HZ / slow, places=3)

    def test_a_measured_rate_is_carried_to_the_next_generation(self):
        tts = _detached()
        tts._rtf_ema = None
        frames = 60
        tts._record_rtf(frames, FIXED_CALL_SECONDS + frames / STEPS_PER_S_SATURATED)
        self.assertAlmostEqual(
            tts.observed_rtf(), CODEC_FRAME_RATE_HZ / STEPS_PER_S_SATURATED, places=3
        )

    def test_one_draw_does_not_move_the_carried_rate_very_far(self):
        """The step COUNT of one generation is itself a sampler draw.

        Its standard deviation is 15 % of the median, against a co-tenancy term
        of 7.8 % -- so an estimator that chased single generations would report
        contention that was only sampling noise.
        """
        tts = _detached()
        tts._rtf_ema = RTF_IDLE
        tts._record_rtf(60, FIXED_CALL_SECONDS + 60 / 5.0)  # an absurd outlier
        self.assertLess(
            tts.observed_rtf() - RTF_IDLE,
            (CODEC_FRAME_RATE_HZ / 5.0 - RTF_IDLE) * 0.5,
            "one bad draw moved the carried rate more than half way",
        )


if __name__ == "__main__":
    unittest.main()
