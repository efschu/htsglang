# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Rung B: the talker and codec as in-process modules, under the ledger.

One process, one ledger, no second engine -- the 2026-08-03 architecture order.
The synthesizer is now an ``nn.Module`` living in the translator's own process
tree, its weights registered as the ``audio_modules`` asset class in the #286
register, parkable and evictable on the same importance ladder as every other
asset in the runtime. That is the whole difference from the revoked sidecar:
not where the code runs, but whether the memory is visible to the thing that
has to arbitrate it.

**Honest scope.** This rung drives the reference modeling code with our own
call sequence rather than reimplementing the talker against our layers. It buys
audible output inside the deadline. What it does not buy, and what #488's
native-lane rung exists for:

* no CUDA-graph capture over the nested code predictor;
* no cross-request batching -- one conversation at a time, which is the MVP.

**Incremental emission (2026-08-04).** This rung now emits audio as codec
frames are produced rather than after the whole unit, which is what turns a
6.9 s wait into a pre-roll. The reference still offers no streaming output of
any kind, so the frames are taken with a forward hook on the talker and decoded
in growing prefixes; both mechanisms are justified where they are used. Two
measured facts make it work and are worth having at the top of the file:

* **generation is SLOWER than playback** -- RTF 1.158 idle, 1.256 with the 27B
  saturated. Emission therefore cannot start at frame 1; it starts once enough
  audio is banked to cover the deficit the rest of the turn will run up, which
  is the inequality in ``release_frame``. This is also why the feature has a
  ceiling: past ~9 s of speech in one turn no pre-roll satisfies both halves of
  the bar, and lifting that needs per-step compute work, not emission work.
* **codec decode is 36 ms and flat** across a 6x range of utterance length, so
  decoding the prefix again on every emission costs a constant rather than a
  growing one.

Both numbers, and the arithmetic that turns them into a pre-roll, are in
``/spinning/466-client-logs/MEASURE_TTS_LATENCY.md``.

The four modules are registered separately on purpose: the codec decoder alone
is 229 MB and is the module a turn needs LAST, so it is the natural first
victim under pressure.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import logging
import math
import time
from pathlib import Path
from typing import AsyncIterator, Dict, Iterable, List, Optional, Sequence

import numpy as np

from sglang.srt.translator.backends import (
    TTS_QUEUE_WAIT_S,
    AudioChunk,
    BackendError,
)
from sglang.srt.translator.ledger import AudioAssetLedger
from sglang.srt.translator.talker_config import TalkerGeometry, read_talker_geometry
from sglang.srt.translator.tts_backends import (
    QWEN3_TTS_LANGUAGE_NAMES,
    languages_from_qwen3_tts_config,
)

logger = logging.getLogger(__name__)

__all__ = [
    "InProcessQwen3Tts",
    "InProcessTtsConfig",
    "CODEC_FRAME_RATE_HZ",
    "release_frame",
]

#: The codec's true frame rate. MEASURED on this checkpoint on 2026-08-04
#: (MEASURE_TTS_LATENCY.md section 1): steps taken divided by audio seconds
#: produced is exactly 12.500 in every arm, and the codec package is literally
#: `qwen_tts/core/tokenizer_12hz/`. `TalkerGeometry.frame_rate_hz` returns 13,
#: which is `position_id_per_seconds` -- the M-RoPE position scaffolding, not
#: the codec rate. Converting a waveform length back to frames with 13
#: over-counts by 4 %, a real bias in a guard whose whole job is to tell a
#: healthy generation from a runaway. Every frame/second conversion in this
#: module goes through this constant instead.
CODEC_FRAME_RATE_HZ = 12.5

#: Fixed per-turn cost before the first sample can exist: prefill 24 ms +
#: reference codec encode 15 ms + speaker encoder 4 ms + one codec decode 36 ms
#: = 79 ms, corroborated independently by the 81 ms intercept of the
#: least-squares fit `call_ms = 81 + 92.6 x steps` -- two methods, 2 % apart
#: (MEASURE_TTS_LATENCY.md section 2). Subtracted from elapsed time when the
#: live step rate is estimated, so that the estimate is a RATE and not a rate
#: plus a constant amortized over the handful of frames seen so far.
FIXED_CALL_SECONDS = 0.079


class _StopUnit(Exception):
    """Raised from the per-step tap to end one generation early.

    The talker's decode loop cannot be stopped from outside by any supported
    argument: `stopping_criteria` and `streamer` are both swallowed by the
    closed `talker_kwargs` dict literal (modeling_qwen3_tts.py:2044), which
    never splats `**kwargs`. Raising out of a forward hook is the one path that
    reaches the loop, and it is safe here because the talker keeps no state
    across calls that a later call does not overwrite: `rope_deltas` is
    rewritten on every prefill (modeling_qwen3_tts.py:1704) and the KV cache is
    a fresh `DynamicCache` per `generate`.
    """


def release_frame(
    expected_frames: float,
    rtf: float,
    holdback_frames: float,
    chunk_frames: float,
    ceiling_frames: int,
) -> int:
    """The frame at which emission may begin and still never underrun.

    THE INEQUALITY (MEASURE_TTS_LATENCY.md, "the arithmetic every lever is
    scored against"). Generation is SLOWER than playback -- RTF 1.158 idle,
    1.256 with the 27B saturated -- so a stream that starts at frame 1 is
    overtaken by its own playback and gaps before the end. Audio may only start
    once enough of it is banked to cover the deficit the rest of the utterance
    will run up.

    Write `g` for frames generated at the moment of release, `N` for the frames
    the whole turn will produce, `R` for the real-time factor and `B` for the
    frames that are generated but not yet sendable. Playback drains one second
    of audio per wall second while generation supplies only `1/R`, so requiring
    the buffer to still be non-negative when the last frame is emitted gives

        g - B >= (R - 1) x (N - g)

    which solves to the release point returned here:

        g* = ceil( ((R - 1) x N + B) / R )

    `B` is the term the ranking's idealized arithmetic left out and this
    implementation cannot. Two things withhold audio: the emission chunk `C`,
    since a chunk that has not filled cannot be sent, and the holdback `H`,
    audio kept back so a later trim can never need it returned. They do not
    add: the backlog is whatever the chunking rounds away, so it is
    `C x ceil(H / C)` -- plain `C` whenever the holdback is no larger than a
    chunk, which is how this module configures it. Charging `H + C` instead
    (as an earlier draft did) buys a pre-roll 0.4 s longer than the mechanism
    needs, and 0.4 s of a 2361 ms budget is not affordable.

    `N` is the WHOLE TURN, not one unit. A gap anywhere fails the bar and the
    worst moment is the last sample; splitting a turn into more units does not
    relax this on one card, because the deficit accrues at the same rate
    wherever the clause boundaries fall.

    Capped at `ceiling_frames` -- holding past the point where the generation
    is going to stop anyway buys nothing and would simply restore the burst.
    """
    backlog = chunk_frames * math.ceil(max(holdback_frames, 1e-9) / max(chunk_frames, 1e-9))
    if not (rtf > 1.0):
        # Generation is at or faster than playback: nothing to bank, and the
        # formula would divide the deficit by a non-positive number. This is
        # the state lever (d) is trying to reach; the code should not have to
        # be rewritten on the day it does.
        return int(max(1.0, min(float(ceiling_frames), backlog)))
    required = ((rtf - 1.0) * expected_frames + backlog) / rtf
    return int(max(1.0, min(float(ceiling_frames), math.ceil(required))))


@dataclasses.dataclass(frozen=True)
class InProcessTtsConfig:
    """Where the checkpoint is and how the talker is driven."""

    model_dir: Path = Path("/spinning/llm_stuff/translator-models/qwen3-tts-0.6b-base")
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    sample_rate: int = 24000
    min_reference_seconds: float = 3.0
    #: Cross-lingual mode: keep the reference transcript out of the LM context.
    #: The mechanism every leading zero-shot cloner relies on when the
    #: reference language differs from the output language. The GPU A/B
    #: compares this against in-context learning, so it stays a flag.
    x_vector_only_mode: bool = True
    #: Emission granularity. Deliberately left at the value the burst path
    #: used: 0.4 s is exactly 5 codec frames (5 x 1920 = 9600 samples at
    #: 24 kHz), so a unit still reaches the client as the SAME NUMBER of
    #: `turn.audio` buffers it always did and only their timing changes. That
    #: matters because every hold and playback counter on the client is
    #: per-buffer (`holdDeferred`, `holdDrained`, `pushed`, `scheduled`,
    #: `audio_buffers_scheduled`), so a finer chunk would silently re-baseline
    #: all of them; only `PLAYBACK_HOLD_MAX_SAMPLES` is measured in samples and
    #: therefore indifferent.
    emit_chunk_seconds: float = 0.4
    #: Emit audio as codec frames are produced instead of after the whole unit.
    #: The kill switch for the whole lever: with this false the module behaves
    #: exactly as it did before, which is what makes the burst path available
    #: as a control arm rather than only as a git revision.
    stream_within_unit: bool = True
    #: Real-time factor assumed before anything has been measured in this
    #: process. 1.256 is the MEASURED value with the 27B saturated at its
    #: `max_running_requests` ceiling (MEASURE_TTS_LATENCY.md section 5), i.e.
    #: the worst of the three real arms -- 1.158 idle, 1.215 at concurrency 1,
    #: 1.256 at concurrency 4. The pessimistic end is the right seed because an
    #: RTF guessed too LOW under-banks the buffer and gaps, while one guessed
    #: too HIGH only spends latency, and the very first generation of a process
    #: is the one arriving with no evidence at all. Every later turn uses the
    #: live measurement instead.
    default_rtf: float = 1.256
    #: Smoothing for the observed RTF carried between generations. The
    #: co-tenancy spread is only 7.8 % between an idle card and a saturated one
    #: (section 5), so this tracks a slow term and should not chase one draw:
    #: the step COUNT of a single generation is itself a sampler draw with a
    #: 15 % standard deviation (section "the noise floor"), and an EMA is what
    #: keeps that noise out of the pre-roll.
    rtf_ema_weight: float = 0.3
    #: THE LATENCY BUDGET, and the only number two separate decisions are made
    #: against. From the acceptance arithmetic (MEASURE_TTS_LATENCY.md, "the
    #: two conditions are one inequality"): the 3000 ms bar, less 560 ms
    #: upstream (ASR 151 + diarization 57 + MT 325 + dispatch 27), less the
    #: 79 ms of TTS fixed cost that precedes any sample, leaves 2361 ms.
    #:
    #: It is used twice, and deliberately by the same value:
    #:
    #: * as the CEILING on the pre-roll. When the gapless inequality asks for
    #:   more than this the request is refused rather than honoured. Past the
    #:   crossover no pre-roll satisfies both halves of the bar, and of the two
    #:   the start time is the one the user stated as a number. The shortfall
    #:   is logged with the figures that produced it.
    #: * as the line above which a unit is STREAMED at all. A unit whose whole
    #:   generation is expected to finish inside this budget does not need to
    #:   be streamed to start in time, so it is generated whole and keeps the
    #:   #564 re-draw. See `_should_stream`.
    #:
    #: Both are the same question -- "does this fit the time the bar leaves?" --
    #: so they are not allowed to drift apart into two tunables.
    preroll_budget_seconds: float = 2.361
    #: Where in the fit's OWN SPREAD the pre-roll is sized. `steps = 14 + 0.7 x
    #: chars` is a fit through medians, so half of all draws exceed it -- and
    #: the inequality it feeds fails asymmetrically: a duration guessed short
    #: gaps, one guessed long only spends latency that is already budgeted. So
    #: the pre-roll is sized at the upper end of the residual rather than at
    #: its centre, which is not a safety margin bolted onto a guess but the
    #: same measured spread the step budget's own factor was chosen against.
    #:
    #: Observed fit-to-actual ratios: 1.03x and 1.10x for the 63- and
    #: 123-character clauses of the 2026-08-04 sweep, 1.47x for the 8-character
    #: one, and 1.01x / 1.01x / 1.39x (58 chars) and 1.15x / 1.48x / 1.26x
    #: (112 chars) across the six streamed arms of
    #: `probe_stream_emission.py`. 1.5 covers every one of those. It costs
    #: ~0.45 s of pre-roll on a field-sized turn, against a budget with ~1.3 s
    #: of room at that length, and the run that motivated it underran by up to
    #: 1.30 s on exactly those arms.
    duration_estimate_factor: float = 1.5
    #: Continuous digital silence, in seconds, that ends a generation early.
    #: The measured runaway has two shapes and this catches the one the field
    #: log showed: per-second RMS [0.0, 0.046, 0.033, 0.0, 0.0, 0.0, 0.0, 0.0]
    #: for a one-word clause -- two seconds of speech and five of exact zero.
    #: Those five seconds are steps the listener will never hear (the emitter
    #: withholds them) but that still delay the NEXT unit, which under
    #: streaming is a gap rather than merely a wait. 1.6 s is deliberately far
    #: above any pause inside a clause; it only ever arms after speech has been
    #: produced, so it cannot truncate a slow start.
    silence_cut_seconds: float = 1.6
    #: Ceiling on talker decode steps for ONE unit. Measured on this
    #: checkpoint on 2026-08-04: the talker decodes at 12.5 steps/s and a
    #: normal clause costs ~85 steps, so 800 is a ~10x headroom over anything
    #: a unit legitimately needs. The previous 2048 was not a ceiling but a
    #: trap: at the measured rate it is ~164 s of decoding, nearly twice
    #: `session.tts_unit_timeout_s` (90 s). A talker that never emits its
    #: codec EOS -- which happens, and twice did on 2026-08-04 -- therefore
    #: could not finish inside the deadline no matter how healthy the card
    #: was, so the deadline could only ever abandon it, never catch it.
    #:
    #: Since 2026-08-04 this is the ABSOLUTE ceiling rather than the operating
    #: bound: what a generation may actually spend is derived from the text by
    #: `step_budget`, and this is the line that budget can never cross.
    max_new_tokens: int = 800
    #: Wall-clock ceiling for one `generate_voice_clone` call, held well below
    #: `session.tts_unit_timeout_s` (90 s) so a runaway synthesis returns a
    #: short waveform BY ITSELF before the session gives up on it. The token
    #: ceiling above cannot do this job alone: it bounds STEPS, and the time a
    #: step costs is not ours to promise while a 27B model shares the card.
    #: See `_generate` for why a deadline that lives only in the event loop is
    #: not enough.
    max_generation_seconds: float = 70.0
    temperature: float = 0.9
    top_p: float = 0.9
    #: Nucleus/top-k/sampling switch, exposed so the decode can be MEASURED
    #: rather than argued about. `None` means "leave it to the checkpoint's own
    #: generation_config.json", which is what this module did implicitly before
    #: these fields existed: top_k 50, and the package's `pick()` only falls
    #: back when the caller passes None (qwen3_tts_model.py:332-338). Motivated
    #: by the runaway investigation of 2026-08-04, where the question "does the
    #: talker terminate under the sampling we ship" could not be answered
    #: without editing the module.
    top_k: Optional[int] = None
    do_sample: bool = True
    #: WHAT THE TEXT NEEDS, in talker steps. The talker decodes one codec frame
    #: per step, and how many frames a clause needs is a property of the TEXT,
    #: not a constant -- which is what `max_new_tokens` above is, and why it
    #: could only ever bound the damage.
    #:
    #: Measured on this checkpoint with `scripts/translator/probe_tts_runaway.py`
    #: on 2026-08-04, Spanish clauses at the shipped sampling: 8 characters ->
    #: 19 steps (median of 10), 63 -> 55/56/60, 123 -> 101. Those three points
    #: sit on `steps = 14 + 0.7 * characters` to within a step, and at the
    #: codec's 12.5 Hz that slope is 17.9 characters per second of speech --
    #: a normal speaking rate, which is the sanity check that this is measuring
    #: speech and not an artefact.
    step_budget_base: int = 14
    step_budget_per_char: float = 0.7
    #: How far past what the text needs a generation may run before it is
    #: treated as a runaway. 2.5x is chosen against the MEASURED spread rather
    #: than as a round number: across 18 healthy generations the widest was
    #: 30 steps against a 19-step median for the same text (1.6x), so 2.5x
    #: clears every healthy sample by a margin while cutting the observed
    #: failures -- 102 steps for an 8-character word (5.2x) and the 311-step
    #: run that never emitted EOS at all -- long before the user hears them.
    #: Raise it if a legitimate clause is ever truncated; that event is logged
    #: with the numbers needed to justify the new value.
    step_budget_factor: float = 2.5
    #: A generation that hits its budget is re-drawn this many times. The
    #: failure is stochastic and rare (1 in 10 generations over-produced in the
    #: 2026-08-04 sweep, and the field session lost 2 units of ~16), so one
    #: fresh draw is expected to leave roughly 1 in 100 units truncated, and it
    #: costs one extra short generation only on a unit that was already wrong.
    runaway_retries: int = 1
    #: Trailing digital silence is CUT from the finished waveform. The
    #: over-produced tail measured on 2026-08-04 was exactly that -- per-second
    #: RMS [0.0, 0.046, 0.033, 0.0, 0.0, 0.0, 0.0, 0.0] for a one-word clause,
    #: i.e. two seconds of speech followed by five of digital zero -- and the
    #: session-level silence gate cannot see it, because that gate reads the
    #: PEAK of the whole turn and the speech at the front keeps the peak high.
    #: `trailing_silence_keep_s` leaves a short natural release rather than
    #: butting the next unit against the last spoken sample.
    trailing_silence_peak: float = 1e-3
    trailing_silence_keep_s: float = 0.2
    #: Park every audio module after a turn. Costs a restore on the next turn
    #: and frees ~2 GB between conversations; the right default for a tenant
    #: sharing a card with a 27B model.
    park_when_idle: bool = False


class _UnitEmitter:
    """Decides which samples of a unit-in-progress may go on the wire.

    Two rules, and they are the whole of the emission policy.

    THE HOLD. Nothing is emitted until `release_frame`'s inequality is
    satisfied, because generation is slower than playback and a stream that
    starts too early is overtaken by its own playback. Until then this only
    accumulates.

    THE PROSPECTIVE TRIM. Samples are emitted up to

        min(len(decoded) - H, last_loud + 1 + K)

    where `H` is a small withheld tail and `K` is `trailing_silence_keep_s`.
    The second term is exactly what `trim_trailing_silence` computes on the
    finished waveform, evaluated on the prefix instead -- so audio that the
    finished-waveform trim would have cut is never sent in the first place,
    and the stream carries the same SAMPLE POSITIONS the burst path would have
    sent, no more and no fewer. `H` is what makes that exact rather than
    approximate: `last_loud` can only move forward, so the second term can
    only grow, while `len - H` is what stops the first term from running past
    a trim point that has not been discovered yet.

    That is a claim about which samples are sent, not about their values. The
    values come from decoding a prefix, which is 39.4 dB from the one-shot
    decode for a reason measured in `_offer_prefix` and not removable here.

    Both terms are monotonic, so `emitted` never has to be taken back -- which
    is the property the whole design needs, because a chunk on the wire cannot
    be recalled.
    """

    def __init__(
        self,
        tts: "InProcessQwen3Tts",
        sink: Optional["_EmissionSink"],
        pacing: Optional[object],
        expected_turn_frames: float,
        ceiling_frames: int,
        streaming: bool,
    ) -> None:
        self._tts = tts
        self._sink = sink
        self._pacing = pacing
        self._config = tts.config
        self._rate = tts.sample_rate
        self.streaming = streaming and sink is not None
        self.expected_turn_frames = expected_turn_frames
        self.ceiling_frames = ceiling_frames
        #: Samples already handed to the wire for THIS unit.
        self.emitted = 0
        self.emitted_chunks = 0
        #: Exactly what was handed to the wire, kept so the unit can RETURN it.
        self.sent: List[np.ndarray] = []
        self.released = False
        self.release_at_frame: Optional[int] = None
        self.release_rtf: Optional[float] = None
        self.first_emit_monotonic: Optional[float] = None
        self._chunk = max(1, int(self._config.emit_chunk_seconds * self._rate))
        self._keep = int(self._config.trailing_silence_keep_s * self._rate)
        # The withheld tail: one codec frame. Any positive holdback makes the
        # emission exact -- `len - H` can never reach a trim point that has not
        # been discovered, because `last_loud` only moves forward -- so the
        # value is chosen to be as small as the guarantee allows rather than as
        # large as it tolerates. It costs nothing at all here: the emitter
        # sends whole chunks, so a holdback under one chunk is rounded away and
        # the backlog is the chunk either way (see `release_frame`).
        self._holdback = max(1, int(self._rate / CODEC_FRAME_RATE_HZ))

    # -- the policy ---------------------------------------------------------

    @property
    def holdback_frames(self) -> float:
        return self._holdback / self._rate * CODEC_FRAME_RATE_HZ

    @property
    def chunk_frames(self) -> float:
        return self._chunk / self._rate * CODEC_FRAME_RATE_HZ

    def emittable_end(self, waveform: np.ndarray, final: bool) -> int:
        """How far into `waveform` the wire may see, under both rules."""
        loud = np.flatnonzero(np.abs(waveform) >= self._config.trailing_silence_peak)
        if not loud.size:
            # No speech yet. Leading silence is real and `trim_trailing_silence`
            # keeps it, but it is only knowable as LEADING once something loud
            # follows -- so hold it. If the unit ends without ever being loud,
            # the whole waveform goes out unchanged, which is the branch that
            # keeps the session's non-silence gate able to see and report it.
            return len(waveform) if final else 0
        trim_end = int(loud[-1]) + 1 + self._keep
        if final:
            return min(len(waveform), trim_end)
        return max(0, min(len(waveform) - self._holdback, trim_end))

    def trailing_silence_seconds(self, waveform: np.ndarray) -> float:
        loud = np.flatnonzero(np.abs(waveform) >= self._config.trailing_silence_peak)
        if not loud.size:
            return 0.0
        return (len(waveform) - int(loud[-1]) - 1) / float(self._rate)

    def may_release(self, frames_done: int, elapsed: float) -> bool:
        """Has enough audio been banked to start without gapping later?

        The estimate is re-derived on every call rather than fixed when the
        unit started, because both of its inputs move: MT can still be adding
        clauses to this turn, and the step rate is a live measurement of a card
        that is shared with a 27B model.
        """
        if self.released:
            return True
        rtf = self._tts.observed_rtf(frames_done, elapsed)
        expected = max(self.expected_turn_frames, float(frames_done))
        target = release_frame(
            expected, rtf, self.holdback_frames, self.chunk_frames,
            self.ceiling_frames,
        )
        # THE BAR IS A HARD NUMBER AND THE GAP IS NOT. When the inequality asks
        # for a longer pre-roll than the budget allows, the request is refused
        # rather than honoured: past the crossover no pre-roll satisfies both
        # halves of the bar, and the user stated the start time in seconds and
        # the continuity in words.
        #
        # The ranking put that crossover at 14.9 s idle and 9.2 s saturated.
        # THIS implementation's is shorter, and the two reasons are both in the
        # code above rather than in the arithmetic: a chunk that has not filled
        # cannot be sent, and the duration is taken at the top of the fit's
        # residual rather than at its centre because the fit was measured
        # under-predicting real draws by up to 48 %. Together they put the cap
        # at roughly 73 characters of pending clause at RTF 1.25 -- comfortably
        # past the 58-character field turn, and short of a monologue. Lifting
        # it is lever (d)'s job, not this one's.
        budget_frames = int(
            self._config.preroll_budget_seconds / max(rtf, 1e-6) * CODEC_FRAME_RATE_HZ
        )
        if target > budget_frames:
            logger.warning(
                "pre-roll capped: the turn's estimated %.1f frames (%.2fs of "
                "speech) at RTF %.3f wants %d frames (%.2fs) of pre-roll but "
                "the %.2fs budget allows %d; releasing early and accepting a "
                "projected %.2fs shortfall at the end of the turn",
                expected, expected / CODEC_FRAME_RATE_HZ, rtf, target,
                target / CODEC_FRAME_RATE_HZ * rtf,
                self._config.preroll_budget_seconds, budget_frames,
                (target - budget_frames) / CODEC_FRAME_RATE_HZ,
            )
            target = budget_frames
        if frames_done >= max(1, target):
            self.released = True
            self.release_at_frame = frames_done
            self.release_rtf = rtf
        return self.released

    # -- the wire -----------------------------------------------------------

    @property
    def cancelled(self) -> bool:
        """Has this turn been told to stop putting audio on the wire?"""
        return bool(getattr(self._pacing, "cancelled", False))

    def offer(self, waveform: np.ndarray, final: bool = False) -> None:
        """Send whatever of `waveform` both rules now allow."""
        if self._sink is None or self.cancelled:
            return
        end = self.emittable_end(waveform, final)
        while end - self.emitted >= (1 if final else self._chunk):
            span = min(self._chunk, end - self.emitted)
            if span <= 0:
                break
            self._push(waveform[self.emitted : self.emitted + span])
            self.emitted += span

    def assembled(self) -> np.ndarray:
        """The utterance as the listener actually received it.

        WHY THE UNIT RETURNS THIS RATHER THAN ITS LAST DECODE. The intermediate
        prefix decodes and the final whole decode do not agree sample for
        sample -- the decoder reaches 25-50 frames forward in time, so a decode
        that could not see the end of the utterance is 39.4 dB from one that
        could (see `_offer_prefix`). Both are the same length.

        Returning the last decode would therefore mean the audio recorded for
        the turn is not quite the audio that was played, and the session's
        non-silence gate, the stored turn audio and every later comparison
        would be looking at something the user never heard. A discrepancy
        between what was played and what was recorded is the kind of thing that
        costs a future investigation its first hour, so it is removed rather
        than documented: the turn IS what the listener received.
        """
        if not self.sent:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.sent).astype(np.float32)

    def _push(self, samples: np.ndarray) -> None:
        chunk = AudioChunk(np.array(samples, dtype=np.float32), self._rate)
        self.sent.append(chunk.samples)
        if self.first_emit_monotonic is None:
            self.first_emit_monotonic = time.monotonic()
        self.emitted_chunks += 1
        # Set from the worker thread, read from the event loop. `started` is
        # the flag the re-draw is gated on, so it has to be true BEFORE the
        # chunk is reachable, never after.
        if self._pacing is not None:
            self._pacing.started = True
            self._pacing.emitted_audio_s += chunk.duration_s
        self._sink.push(chunk)


class _EmissionSink:
    """Carries chunks from the talker's worker thread to the event loop.

    The talker runs on `_talker_executor`'s single thread and cannot be
    cancelled once inside `generate_voice_clone`; the consumer is an async
    generator that a unit deadline may cancel at any moment. So the handoff is
    one-way and non-blocking: the producer never waits on the consumer, which
    means a cancelled turn cannot wedge the thread that holds the talker.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._queue: "asyncio.Queue[Optional[AudioChunk]]" = asyncio.Queue()
        self._closed = False

    def push(self, chunk: Optional[AudioChunk]) -> None:
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, chunk)
        except RuntimeError:  # pragma: no cover - loop already closed
            pass

    def close(self) -> None:
        # Idempotent: the generation closes the sink when it fails and the
        # future's done callback closes it again, and a second sentinel left
        # in the queue would be read as the end of the NEXT unit.
        if self._closed:
            return
        self._closed = True
        self.push(None)

    async def get(self) -> Optional[AudioChunk]:
        return await self._queue.get()


class InProcessQwen3Tts:
    """Zero-shot cloning TTS as an in-process, ledger-registered module."""

    def __init__(
        self,
        config: Optional[InProcessTtsConfig] = None,
        ledger: Optional[AudioAssetLedger] = None,
    ) -> None:
        self.config = config or InProcessTtsConfig()
        self.name = f"inprocess-qwen3-tts:{self.config.model_dir.name}"
        self.sample_rate = self.config.sample_rate
        self.min_reference_seconds = self.config.min_reference_seconds
        self.ledger = ledger if ledger is not None else AudioAssetLedger()

        # The geometry read validates the M-RoPE mapping and will refuse a
        # checkpoint that would build the wrong rotary. Doing it here means no
        # weight is touched before the trap is ruled out.
        self.geometry: TalkerGeometry = read_talker_geometry(self.config.model_dir)
        self._languages = tuple(
            languages_from_qwen3_tts_config(self.config.model_dir)
        )
        # The checkpoint's own table is name -> ISO code; the reference API
        # takes the ENGLISH NAME ("spanish"), not the code ("es"). Inverting
        # it here keeps the whole rest of the system on ISO codes -- which is
        # what requirement 5's language matrix intersects on -- and confines
        # the model's vocabulary to the one call that needs it.
        self._code_to_name = {
            code: name for name, code in QWEN3_TTS_LANGUAGE_NAMES.items()
        }
        self._model = None
        self._lock = asyncio.Lock()
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        #: Real-time factor carried between generations. `None` until one has
        #: been measured in this process; see `observed_rtf` for why a measured
        #: value beats the compiled-in one even though the spread is only 8 %.
        self._rtf_ema: Optional[float] = None

    # -- capability ---------------------------------------------------------

    @property
    def busy(self) -> bool:
        """True while another turn is inside the synthesizer.

        One talker serves every session, so a second conversation's turn does
        not run slowly -- it does not run at all until this clears. Measured
        (DESIGN §17.8.2): a turn arriving while another is synthesizing pays
        the whole of that synthesis, a median 4.8 s. The caller reads this to
        tell the user WHY, because a queue nobody can see is indistinguishable
        from a system that has become slow.
        """
        return self._lock.locked()

    def supported_languages(self) -> Iterable[str]:
        return self._languages

    #: Frames before a live step-rate estimate is trusted. The per-step cost is
    #: reproducible to 1-3 % (MEASURE_TTS_LATENCY.md, "the noise floor"), so a
    #: handful of frames is already a good rate -- but the FIRST frames also
    #: carry the prefill, so this is set where the fixed-cost correction below
    #: has something to correct.
    MIN_FRAMES_FOR_RATE = 6

    def observed_rtf(self, frames_done: int = 0, elapsed: float = 0.0) -> float:
        """Seconds of wall clock per second of audio, measured if possible.

        MEASURE IT, DO NOT COMPILE IT IN. The spread between an idle card and
        one sharing with a saturated 27B is 7.8 % of step rate -- RTF 1.158 to
        1.256 -- which is small but free to track, and it is the difference
        between a 14.9 s and a 9.2 s crossover. The compiled-in default is only
        the seed for the first generation of a process.

        Three sources, in order of how much they know:

        1. the current generation's own frames and elapsed time, which is the
           only number that reflects what the card is doing RIGHT NOW;
        2. the EMA of previous generations, for the frames before (1) exists;
        3. `config.default_rtf`, for the first generation of a process.

        `FIXED_CALL_SECONDS` is subtracted before dividing. Without it the
        estimate is a rate plus a constant: at 6 frames the measured 79 ms of
        prefill and encodes would inflate the apparent RTF by ~14 %, which
        reads as a contended card and buys a pre-roll nobody needed.
        """
        live = None
        if frames_done >= self.MIN_FRAMES_FOR_RATE:
            working = elapsed - FIXED_CALL_SECONDS
            if working > 0.0 and frames_done / working > 0.0:
                live = CODEC_FRAME_RATE_HZ / (frames_done / working)
        # `getattr` rather than the attribute: the hermetic tests raise this
        # class with `object.__new__` to avoid loading a checkpoint, and a
        # rate estimator is exactly the kind of thing that should degrade to
        # its documented default rather than refuse to run.
        carried = getattr(self, "_rtf_ema", None)
        if carried is None:
            carried = self.config.default_rtf
        if live is None:
            return carried
        # THE PRE-RELEASE MEASUREMENT IS STRUCTURALLY OPTIMISTIC, so the two
        # sources are combined rather than ranked. Nothing is decoded before
        # the release point -- there is nothing to send yet -- so `live` is the
        # bare talker rate. After release every emission adds a codec decode to
        # the same thread, and the 2026-08-04 probe measured what that is
        # worth: whole-call RTF 1.21-1.27 against a bare-talker 1.158. Sizing
        # the pre-roll on `live` alone therefore under-banks by the exact
        # amount the emission itself is about to cost, and the first probe run
        # duly underran by 0.39-1.57 s on every streamed arm.
        #
        # The EMA is the other way round: it is taken over WHOLE generations,
        # so it already carries the decode charge -- but from previous turns,
        # so it knows nothing about the card right now. Taking the larger keeps
        # whichever effect is currently binding. It can only ever cost latency,
        # and it is the side of the trade where a mistake is recoverable.
        return max(live, carried)

    def _record_rtf(self, frames_done: int, elapsed: float) -> None:
        if frames_done < self.MIN_FRAMES_FOR_RATE:
            return
        working = elapsed - FIXED_CALL_SECONDS
        if working <= 0.0:
            return
        measured = CODEC_FRAME_RATE_HZ / (frames_done / working)
        weight = self.config.rtf_ema_weight
        carried = getattr(self, "_rtf_ema", None)
        self._rtf_ema = (
            measured if carried is None
            else (1.0 - weight) * carried + weight * measured
        )

    def to_json(self) -> Dict[str, object]:
        return {
            "backend": self.name,
            "loaded": self._model is not None,
            "languages": list(self._languages),
            "x_vector_only_mode": self.config.x_vector_only_mode,
            "geometry": {
                "layers": self.geometry.num_hidden_layers,
                "hidden": self.geometry.hidden_size,
                "code_groups": self.geometry.num_code_groups,
                "frame_hz": self.geometry.frame_rate_hz,
            },
            "ledger": self.ledger.to_json(),
        }

    # -- lifecycle ----------------------------------------------------------

    def load(self) -> None:
        """Load the checkpoint and register every module with the ledger."""
        if self._model is not None:
            return
        import torch

        from sglang.srt.translator.qwen3_tts_compat import (
            ensure_qwen3_tts_importable,
            refresh_rotary_buffers,
            restore_cache_position,
            retarget_wrapper_device,
            verify_and_load_weights,
        )

        shims = ensure_qwen3_tts_importable()
        logger.info("qwen3-tts compat shims: %s", ", ".join(shims))

        from qwen_tts.core.models.modeling_qwen3_tts import (
            Qwen3TTSTalkerForConditionalGeneration,
        )
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

        # Drift 6, and the one that blocked every decode step: transformers 5.x
        # stopped creating `cache_position`, which is the flag the talker
        # branches on to tell prefill from decode. Must be installed on the
        # CLASS before any generate call. See qwen3_tts_compat.
        if restore_cache_position(Qwen3TTSTalkerForConditionalGeneration):
            logger.info("restored cache_position on the talker for decode steps")

        dtype = getattr(torch, self.config.dtype)
        # NO `device_map`. transformers 5.x routes it through `accelerate`
        # (integrations/accelerate.py:134 raises without it), and accelerate is
        # not in this venv -- deliberately, since two long-running services map
        # that venv and it buys nothing for a single-device tenant. Loading
        # plainly and moving afterwards is the same end state for a 0.6 B model
        # and keeps CPU and CUDA on ONE code path, so the desk path and the
        # window path cannot diverge.
        self._model = Qwen3TTSModel.from_pretrained(
            str(self.config.model_dir), dtype=dtype
        )
        inner = getattr(self._model, "model", self._model)
        if hasattr(inner, "to"):
            inner.to(self.config.device)
        # `inner.to()` does NOT finish the job, and both halves of what it
        # misses are silent on CPU. The wrapper cached its device in
        # __init__ (stale prompts -> a device error), and the codec hangs off a
        # PLAIN holder that is not in the module tree at all, so it stays on
        # the CPU and costs 90% of every call. Both are repaired and verified
        # here. See qwen3_tts_compat.retarget_wrapper_device.
        logger.info(
            "device placement: %s",
            retarget_wrapper_device(self._model, self.config.device),
        )
        # Non-persistent rotary buffers do not survive 5.x's meta-device
        # construction; unrefreshed they are NaN and the failure only surfaces
        # as a NaN probability tensor at sampling time. See
        # qwen3_tts_compat.refresh_rotary_buffers.
        inner = getattr(self._model, "model", self._model)
        refreshed = refresh_rotary_buffers(inner)
        logger.info("refreshed %d rotary buffers", refreshed)
        # Drift 7, and the reason this check is not optional: under 5.x
        # `from_pretrained` reported "Loading weights: 478/478" and loaded
        # NONE of them. A randomly initialised talker in front of a correctly
        # loaded vocoder does not fail -- it produces fluent babble that never
        # ends, and every cheap signal (finite, speech-shaped, high speaker
        # similarity) says it is fine. Only a comparison against the
        # checkpoint bytes can see it, so it runs on every load.
        report = verify_and_load_weights(inner, self.config.model_dir)
        logger.info(
            "weight verification: %d tensors checked, %d repaired",
            report["checked"],
            report["repaired"],
        )
        self._register_modules()

    def _register_modules(self) -> None:
        """Register the four weight blocks as separate ledgered assets.

        Separate, not one blob: they have different sizes and different
        last-needed points in a turn, and the register can only make a good
        victim choice at a grain that reflects that.
        """
        inner = getattr(self._model, "model", self._model)
        # `code2wav` is not an attribute of THIS checkpoint's model -- the codec
        # lives behind the plain `speech_tokenizer` holder. The old path
        # therefore registered nothing for it and logged a warning nobody read,
        # so the largest single audio asset was invisible to the register that
        # exists to arbitrate exactly such assets. A registration that never
        # binds has the same reach as no registration at all.
        candidates = [
            ("talker_trunk", self._resolve(inner, "talker.model")),
            ("code_predictor", self._resolve(inner, "talker.code_predictor")),
            ("speaker_encoder", self._resolve(inner, "speaker_encoder")),
            ("codec", self._resolve(inner, "speech_tokenizer.model")),
        ]
        missing = [name for name, module in candidates if module is None]
        if missing:
            # Loud, not a warning: an unledgered module is memory the #286
            # register cannot see, park or evict, and the one-runtime law rests
            # on that register being complete.
            raise BackendError(
                "tts",
                f"audio modules {missing} were not found on the loaded model, "
                "so their weights would be invisible to the asset register. "
                "The checkpoint's module layout changed; fix the paths rather "
                "than run with an incomplete ledger.",
            )
        for name, module in candidates:
            try:
                self.ledger.register(name, module)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("could not register %s with the ledger: %s", name, exc)

    @staticmethod
    def _resolve(root: object, dotted: str) -> Optional[object]:
        node = root
        for part in dotted.split("."):
            node = getattr(node, part, None)
            if node is None:
                return None
        return node

    def park(self) -> int:
        """Park every audio module. Returns bytes freed."""
        return self.ledger.park_all()

    def ensure_resident(self) -> Dict[str, float]:
        return self.ledger.ensure_resident()

    # -- synthesis ----------------------------------------------------------

    async def synthesize(
        self,
        text: str,
        language: str,
        reference: AudioChunk,
        reference_text: Optional[str] = None,
        voice_id: Optional[str] = None,
        pacing: Optional[object] = None,
    ) -> AsyncIterator[AudioChunk]:
        text = text.strip()
        if not text:
            return
        if language not in self._languages:
            raise BackendError(
                "tts",
                f"checkpoint cannot speak {language!r}; it speaks "
                f"{list(self._languages)}",
            )
        if reference is None:
            raise BackendError("tts", "in-process cloning needs reference audio")
        if reference.duration_s < self.min_reference_seconds:
            raise BackendError(
                "tts",
                f"reference is {reference.duration_s:.2f}s, need "
                f">= {self.min_reference_seconds}s",
            )

        # One turn at a time: the modules are shared mutable state and a
        # second concurrent generate would interleave KV caches.
        queued_at = time.monotonic()
        loop = asyncio.get_running_loop()
        sink = _EmissionSink(loop)
        pending = await self._generate_under_lock(
            text, language, reference, reference_text, queued_at, sink, pacing
        )

        # THE CHUNKS COME OFF THE QUEUE, NOT OFF THE RESULT. Everything the
        # generation decides to emit is pushed by the worker thread as it is
        # produced, so this loop can be yielding the opening of a clause while
        # the talker is still deciding how it ends. The burst path is the same
        # code with every chunk arriving at once, which is what keeps a single
        # consumer correct for both.
        streamed = 0
        while True:
            chunk = await sink.get()
            if chunk is None:
                break
            streamed += 1
            yield chunk
        # The future is awaited AFTER the queue is drained, to surface what the
        # thread raised. The done callback closes the sink, so this cannot
        # deadlock on a generation that failed before its own cleanup.
        waveform = await asyncio.wrap_future(pending, loop=loop)

        # THE NET UNDER THE EMISSION PATH. A unit that produced a waveform but
        # emitted nothing has to reach the wire anyway: audio that silently
        # fails to arrive is the worst failure shape this pipeline has, since
        # every event still reports success and the listener cannot tell it
        # from a network drop. Guarded on `streamed` so it can only ever fire
        # when nothing at all was sent, never as a second copy of a stream that
        # merely stopped early.
        # `cancelled` is checked FIRST and separately. A turn that was stopped
        # emitted nothing precisely because it was stopped, and the net below
        # would read that as a failure to deliver and send the whole utterance
        # -- the exact opposite of what was asked for, and the kind of thing
        # that is only obvious before the stop feature exists rather than after.
        if getattr(pacing, "cancelled", False):
            return
        if not streamed and waveform is not None and len(waveform):
            logger.warning(
                "nothing was emitted incrementally for a %.2fs unit; falling "
                "back to sending the finished waveform",
                len(waveform) / self.sample_rate,
            )
            span = max(1, int(self.config.emit_chunk_seconds * self.sample_rate))
            for start in range(0, len(waveform), span):
                yield AudioChunk(waveform[start : start + span], self.sample_rate)

        if self.config.park_when_idle:
            self.park()

    async def _generate_under_lock(
        self,
        text: str,
        language: str,
        reference: AudioChunk,
        reference_text: Optional[str],
        queued_at: float,
        sink: "_EmissionSink",
        pacing: Optional[object],
    ) -> "concurrent.futures.Future":
        """Run one synthesis, holding the talker until the THREAD is done.

        THE LIE (2026-08-04 12:51). The lock used to be an ``async with``
        around ``await run_in_executor``. That reads as "held for the whole
        synthesis", and it is not: a future handed back by the executor cannot
        be cancelled once its thread has started, so when the session's unit
        deadline cancelled the await, the context manager released the lock
        while the talker kept decoding. ``busy`` -- which is just
        ``self._lock.locked()`` -- then reported free. One second later the
        automatic redrive believed it and called ``generate_voice_clone`` on
        the same module tree, which is exactly the interleaving the comment in
        ``synthesize`` forbids. Two generations then shared the card and each
        halved the other's rate (measured: 12.5 steps/s alone, 10.7 with the
        overlap), so the redrive missed its deadline too.

        The repair is to make lock ownership follow the FUTURE rather than
        this coroutine: the done callback is the only place that releases, so
        the lock outlives any cancellation of the await and ``busy`` stays
        true for exactly as long as a thread is still inside the talker.
        """
        await self._lock.acquire()
        try:
            TTS_QUEUE_WAIT_S.set(time.monotonic() - queued_at)
            loop = asyncio.get_running_loop()
            pending = self._talker_executor().submit(
                self._generate, text, language, reference, reference_text,
                sink, pacing,
            )
        except BaseException:
            # Nothing was submitted, so no callback will ever fire: this is
            # the one path that has to release the lock itself.
            self._lock.release()
            raise
        # Fires on the worker thread the moment `_generate` returns or raises,
        # including long after this coroutine was cancelled. Exactly once per
        # submit, so the lock cannot be released twice.
        # Fires on the worker thread the moment `_generate` returns or raises,
        # so the sink is closed on exactly the same event that frees the lock.
        # Putting it here rather than inside `_generate` is what makes the
        # consumer safe against a `_generate` that never reaches its own
        # cleanup -- including the stand-ins the hermetic tests substitute for
        # it, which know nothing about sinks.
        def _finished(_future) -> None:
            loop.call_soon_threadsafe(self._release_talker)
            sink.close()

        pending.add_done_callback(_finished)
        return pending

    def _release_talker(self) -> None:
        if self._lock.locked():
            self._lock.release()

    def _talker_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """The ONE thread every synthesis runs on.

        Owned here rather than borrowed from the loop's default executor. That
        pool has several threads, so two callers that both believed the talker
        was free could genuinely be inside `generate_voice_clone` at the same
        time, on the same module tree -- the interleaving `synthesize` warns
        about. With a single worker the serialization is structural: the
        second generation cannot start before the first returns, whatever the
        lock happens to think.

        Built on demand so an instance raised with `object.__new__` -- how the
        hermetic tests avoid loading a checkpoint -- needs no extra wiring.
        """
        executor = getattr(self, "_executor", None)
        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="inprocess-tts"
            )
            self._executor = executor
        return executor

    def arm_generation_deadline(self, model: object) -> bool:
        """Bound the talker's decode loop by wall-clock time. True if armed.

        WHY NOT `stopping_criteria=`. The obvious spelling --
        ``generate_voice_clone(stopping_criteria=StoppingCriteriaList([
        MaxTimeCriteria(...)]))`` -- is silently dropped, which is worse than
        rejected. The value survives ``Qwen3TTSModel._merge_generate_kwargs``
        (``merged = dict(kwargs)``, inference/qwen3_tts_model.py:339) and
        reaches ``Qwen3TTSForConditionalGeneration.generate``, where the
        arguments actually handed to the talker are assembled as a CLOSED dict
        literal (``talker_kwargs``, core/models/modeling_qwen3_tts.py:2044)
        that never splats ``**kwargs``. So the criteria are accepted, carried
        two frames, and thrown away, and the synthesis runs unbounded anyway.

        ``generation_config.max_time`` is the same mechanism reached from the
        side the closed dict cannot block: transformers builds precisely a
        ``MaxTimeCriteria`` from it (generation/utils.py:1315) on every
        ``generate`` call, and ``talker_kwargs`` sets no ``max_time`` that
        could override it. Re-armed per call rather than once at load, because
        the clock starts when the criterion is CONSTRUCTED and a stale config
        is the kind of thing a checkpoint reload would leave behind.

        A talker cut off by time returns the frames it has: the reference
        keeps the full length when no codec EOS is present
        (``effective_lengths``, modeling_qwen3_tts.py:2287), so this yields a
        short utterance rather than an empty one.
        """
        talker = getattr(getattr(model, "model", None), "talker", None)
        generation_config = getattr(talker, "generation_config", None)
        if generation_config is None:
            # Never fail a turn over a checkpoint layout change: an unarmed
            # talker is still caught by the session deadline, just rudely.
            logger.warning(
                "could not arm the talker generation deadline: no "
                "generation_config on the talker; synthesis is bounded only "
                "by max_new_tokens=%d",
                self.config.max_new_tokens,
            )
            return False
        generation_config.max_time = self.config.max_generation_seconds
        return True

    def step_budget(self, text: str) -> int:
        """Talker steps this text may legitimately need.

        THE RUNAWAY (2026-08-04, package client-20260804T141043Z). One session
        synthesized 119 s of audio for 13.6 s of user speech, and two units ran
        until the 70 s wall-clock ceiling stopped them -- the user heard the
        right word, then dead air, and had to press stop. Reproduced off the
        phone with `scripts/translator/probe_tts_runaway.py`: the talker fails
        to emit its codec EOS on roughly one generation in ten, independently of
        the reference (the two-speaker "merged reference" arms were the
        healthiest of the four), and the failure is not binary -- a run that
        emitted EOS after 102 steps for the word "Gracias." is the same defect
        as one that never emitted it.

        Neither existing guard can catch that, and neither is meant to.
        `max_new_tokens` is a constant, so it says the same thing about one word
        as about a full sentence; `max_generation_seconds` is a wall clock, so
        it bounds how long the damage lasts, not whether it happens. The only
        thing that knows how much audio is correct here is the text, so that is
        what this bounds against.

        Deliberately NOT a safety margin on top of a guess: the coefficients are
        measured on this checkpoint (see the config fields), and the factor is
        checked against the spread of healthy generations rather than picked.
        """
        needed = self.config.step_budget_base + self.config.step_budget_per_char * len(text)
        budget = int(math.ceil(needed * self.config.step_budget_factor))
        # Never above the absolute ceiling, and never so small that the floor
        # term alone could truncate a one-word clause.
        return max(8, min(self.config.max_new_tokens, budget))

    def trim_trailing_silence(self, waveform: np.ndarray) -> np.ndarray:
        """Cut digital silence off the END of a finished utterance.

        The over-production measured on 2026-08-04 was silence, not babble: the
        talker keeps spending steps after the words are done instead of emitting
        EOS, and the codec renders those steps as zeros. Playing them costs the
        listener the whole wait, and the next unit cannot start because one
        talker serves the session -- so the dead air is paid twice.

        Only the tail is touched. Leading and interior silence carry timing that
        belongs to the utterance, and a trim that reached into them would be a
        prosody edit rather than a repair.
        """
        keep = int(self.config.trailing_silence_keep_s * self.sample_rate)
        loud = np.flatnonzero(np.abs(waveform) >= self.config.trailing_silence_peak)
        if not loud.size:
            # Entirely silent: not this method's call. The session-level
            # non-silence gate exists for that and reports it as a failure,
            # which a quietly emptied buffer here would hide.
            return waveform
        end = min(len(waveform), int(loud[-1]) + 1 + keep)
        return waveform[:end]

    def expected_frames(self, text: str) -> float:
        """Codec frames this text is worth, from the measured fit.

        `step_budget`'s coefficients without its 2.5x factor: the budget is
        "how far past what the text needs may a generation run", this is "what
        the text needs". The pre-roll is sized against the EXPECTATION, not
        against the tolerance -- banking audio for a runaway that will probably
        not happen would spend the latency budget on every healthy turn.
        """
        return self.config.step_budget_base + self.config.step_budget_per_char * len(text)

    def expected_turn_frames(self, text: str, pacing: Optional[object]) -> float:
        """What the whole TURN still owes, in codec frames.

        The unit in hand plus everything MT has already queued behind it, for
        the reason `TurnPacing` exists: the deficit accrues across clause
        boundaries because one talker serves them back to back.

        The queued clauses are charged one base term between them rather than
        one each. Their split into units is not knowable from a character
        count, and over-charging the base would inflate the pre-roll on every
        multi-clause turn; the per-character term, which dominates, is exact.
        """
        frames = self.expected_frames(text)
        pending = int(getattr(pacing, "pending_chars", 0) or 0)
        if pending > 0:
            frames += self.config.step_budget_base
            frames += self.config.step_budget_per_char * pending
        # Sized at the top of the fit's measured residual, not its centre: see
        # `duration_estimate_factor`. Applied here rather than inside
        # `expected_frames`, because `_should_stream` wants the CENTRAL
        # estimate -- it is asking what a unit typically costs, not how badly
        # it could overrun.
        return frames * self.config.duration_estimate_factor

    def _should_stream(self, text: str, pacing: Optional[object]) -> bool:
        """Whether this unit is emitted incrementally or as a finished burst.

        WHERE THE #564 RE-DRAW SURVIVES, AND WHY IT IS HERE. A re-draw can only
        replace audio nobody has heard: once the first chunk is on the wire the
        listener would get the opening of a runaway followed by a different
        take of the same clause, which is worse than either. So the two are
        exclusive, and the only honest question is where to draw the line.

        It is drawn where the BAR draws it. A unit whose whole generation is
        expected to finish inside `preroll_budget_seconds` does not need to
        be streamed to start in time, so it is generated whole and keeps the
        re-draw. Everything above that line is streamed and gives the re-draw
        up. At the measured 10.5 steps/s the line sits at ~24 frames, i.e.
        clauses up to ~14 characters -- which covers the clause the runaway was
        actually observed on ("Gracias.", 8 characters, 1 generation in 10) and
        does not cover a normal 5-second clause.

        WHAT GIVING IT UP COSTS, stated plainly. The measured runaway has two
        shapes (test_tts_runaway.py, "what it is"). The one that emits EOS late
        with an exactly-zero tail costs nothing here: the emitter's prospective
        trim never puts that tail on the wire, so the listener hears what a
        successful re-draw would have given them, sooner. The one that babbles
        -- 311 steps of continuous speech-shaped audio for one word -- is the
        real loss: on a streamed unit the listener now hears babble from the
        release point until `step_budget` stops it, where a re-draw would have
        replaced it. That is the price of the lever, it is not recoverable
        while R > 1, and it is bounded by the same budget as before.

        Once the turn has started speaking there is no choice left: a unit
        generated whole in the middle of a turn is a hole in the stream exactly
        as long as its own generation, so streaming is forced regardless of
        size.
        """
        if not self.config.stream_within_unit:
            return False
        if getattr(pacing, "started", False):
            return True
        rtf = self.observed_rtf()
        rate = CODEC_FRAME_RATE_HZ / max(rtf, 1e-6)
        whole_call_s = FIXED_CALL_SECONDS + self.expected_frames(text) / rate
        return whole_call_s > self.config.preroll_budget_seconds

    def _generate(
        self,
        text: str,
        language: str,
        reference: AudioChunk,
        reference_text: Optional[str],
        sink: Optional["_EmissionSink"] = None,
        pacing: Optional[object] = None,
    ) -> np.ndarray:
        """Synthesize one unit, emitting as it goes and re-drawing if it may.

        The retry is what makes the runaway guard a repair rather than a
        shorter leash: a generation that hits the text's budget did not stop on
        its own, the failure is a stochastic property of one draw, and a fresh
        draw is overwhelmingly likely to be healthy. It is now conditional on
        nothing having been emitted -- see `_should_stream` for why that
        condition is the whole shape of the conflict, and what it costs.
        """
        budget = self.step_budget(text)
        streaming = self._should_stream(text, pacing)
        emitter = None
        waveform = None
        attempt = 0
        try:
            while True:
                emitter = _UnitEmitter(
                    self, sink, pacing,
                    self.expected_turn_frames(text, pacing),
                    budget, streaming,
                )
                waveform = self._generate_once(
                    text, language, reference, reference_text, budget, emitter
                )
                frames = len(waveform) / self.sample_rate * CODEC_FRAME_RATE_HZ
                # One frame of slack: the codec's output length is a multiple of
                # its hop, so an exact comparison against the budget would be a
                # coin flip on rounding.
                if frames < budget - 1 or attempt >= self.config.runaway_retries:
                    break
                if emitter.emitted:
                    logger.warning(
                        "talker did not stop by itself: %.0f frames for %d "
                        "characters (budget %d, %.2fs of audio), but %.2fs has "
                        "already been sent, so the draw cannot be replaced -- "
                        "the unit is truncated at its budget instead",
                        frames, len(text), budget,
                        len(waveform) / self.sample_rate,
                        emitter.emitted / self.sample_rate,
                    )
                    break
                attempt += 1
                logger.warning(
                    "talker did not stop by itself: %.0f frames for %d characters "
                    "(budget %d, %.2fs of audio); nothing emitted yet, so "
                    "re-drawing (attempt %d of %d)",
                    frames, len(text), budget,
                    len(waveform) / self.sample_rate, attempt,
                    self.config.runaway_retries,
                )
            trimmed = self.trim_trailing_silence(waveform)
            if len(trimmed) != len(waveform):
                logger.info(
                    "trimmed %.2fs of trailing silence from a %.2fs utterance "
                    "(%d characters)",
                    (len(waveform) - len(trimmed)) / self.sample_rate,
                    len(waveform) / self.sample_rate, len(text),
                )
            # The stream and the burst path cover the same sample positions,
            # by construction: the emitter's two rules are both monotonic and
            # its trim term is the same expression `trim_trailing_silence`
            # evaluates, so sending the remainder here closes the utterance at
            # exactly the sample the burst path would have ended on.
            emitter.offer(trimmed, final=True)
            if emitter.cancelled:
                # A turn told to stop mid-stream is SHORTER than its waveform
                # by definition, so the completeness check below would report
                # the abort as a defect. What was sent is what the turn is.
                return emitter.assembled()
            if emitter.emitted != len(trimmed):
                logger.error(
                    "streamed %d samples of a %d-sample utterance: the emitted "
                    "prefix and the finished waveform disagree, so the listener "
                    "heard something the burst path would not have produced",
                    emitter.emitted, len(trimmed),
                )
            # What was SENT is what the turn is, so that the audio stored for
            # the turn and the audio the listener heard cannot differ. They are
            # the same length and the same samples up to the decoder's own
            # float-level shape dependence; see `_UnitEmitter.assembled`.
            return emitter.assembled() if emitter.emitted else trimmed
        except BaseException:
            # The sink is closed by the future's done callback either way; this
            # only makes the consumer stop waiting at the moment of the failure
            # rather than after whatever unwinding follows it.
            if sink is not None:
                sink.close()
            raise

    def _generate_once(
        self,
        text: str,
        language: str,
        reference: AudioChunk,
        reference_text: Optional[str],
        max_new_tokens: int,
        emitter: Optional["_UnitEmitter"] = None,
    ) -> np.ndarray:
        import torch

        if self._model is None:
            self.load()
        self.ensure_resident()
        self.arm_generation_deadline(self._model)

        reference_np = np.asarray(reference.samples, dtype=np.float32)
        # The wrapper builds the voice-clone prompt itself from (audio, sr).
        # Going through create_voice_clone_prompt() and passing the result as
        # voice_clone_prompt= is the documented alternative, but it hands the
        # talker a differently-shaped prompt and fails deep inside the text
        # projection -- so the simple form is also the correct one here.
        model_language = self._code_to_name.get(language, language)
        frames: List[object] = []
        tap = self._attach_frame_tap(frames, emitter)
        started = time.monotonic()
        cut_early = False
        output = None
        try:
            with torch.inference_mode():
                output = self._model.generate_voice_clone(
                    text=[text],
                    language=[model_language],
                    ref_audio=[(reference_np, reference.sample_rate)],
                    ref_text=[reference_text or ""],
                    x_vector_only_mode=self.config.x_vector_only_mode,
                    non_streaming_mode=True,
                    # The TEXT's budget, not the module constant.
                    # `max_new_tokens` in the config remains the absolute
                    # ceiling and `step_budget` never exceeds it.
                    max_new_tokens=max_new_tokens,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                )
        except _StopUnit as stop:
            cut_early = True
            logger.warning("talker generation ended early: %s", stop)
        finally:
            if tap is not None:
                tap.remove()
        self._record_rtf(len(frames), time.monotonic() - started)

        if tap is None:
            # No tap: either streaming is off for this unit or the checkpoint's
            # module layout has moved. Either way the reference's own return
            # value is the waveform, exactly as before.
            return self._to_waveform(output)

        waveform = self._decode_frames(frames)
        if not cut_early and output is not None:
            self._check_stream_matches_burst(waveform, self._to_waveform(output))
        return waveform

    # -- incremental emission ----------------------------------------------

    def _attach_frame_tap(
        self, frames: List[object], emitter: Optional["_UnitEmitter"]
    ) -> Optional[object]:
        """Hook the talker so each codec frame is seen as it is produced.

        WHY A HOOK AND NOT AN ARGUMENT. The reference has no streaming output
        of any kind -- no generator, no callback, no audio streamer -- and the
        two arguments that could have carried one are unreachable: the
        arguments handed to the talker are assembled as a CLOSED dict literal
        (`talker_kwargs`, modeling_qwen3_tts.py:2044) that never splats
        `**kwargs`, so `streamer=` and `stopping_criteria=` are accepted,
        carried two frames and thrown away. `generation_config` is the one door
        that dict cannot block, and `streamer` is not a field of it.

        WHICH MODULE. One level above what the latency probe hooked. The trunk
        (`talker.model`) returns hidden states and the code predictor returns
        logits; neither carries a sampled frame. `talker` itself is invoked
        once per decode step through `Module.__call__`, and its output's
        `hidden_states[-1]` IS the frame: the 16 codebook ids of one 80 ms
        codec frame (modeling_qwen3_tts.py:1738). `None` on the prefill call,
        which is the same test the reference itself uses when it assembles
        `talker_codes` after the loop.

        WHY THE FRAMES ARE COMPLETE. At batch 1 the reference's
        `effective_lengths` never trims: transformers samples the EOS token
        after a forward and breaks before feeding it back, so the EOS frame is
        never produced and `effective_lengths == talker_codes.shape[1]`
        (trimming only ever fires for batch > 1, where finished rows are
        pad-filled and keep being forwarded). Every frame seen here is a
        keeper, in order, and nothing is taken back afterwards.
        """
        if emitter is None or not emitter.streaming or self._model is None:
            return None
        inner = getattr(self._model, "model", self._model)
        talker = getattr(inner, "talker", None)
        if talker is None or not hasattr(talker, "register_forward_hook"):
            logger.warning(
                "cannot tap the talker for incremental emission: no `talker` "
                "on the loaded model, so this unit falls back to emitting the "
                "finished waveform"
            )
            return None
        state = {"decoded_at": 0, "started": time.monotonic()}

        def _on_step(_module, _inputs, output):
            hidden = getattr(output, "hidden_states", None)
            if not isinstance(hidden, (tuple, list)) or not hidden:
                return
            codes = hidden[-1]
            if codes is None:
                return  # the prefill call, which produces no frame
            frames.append(codes)
            self._offer_prefix(frames, emitter, state)

        return talker.register_forward_hook(_on_step)

    def _offer_prefix(
        self, frames: List[object], emitter: "_UnitEmitter", state: Dict[str, float]
    ) -> None:
        """Decode what exists so far and give the emitter its chance.

        Called from inside the talker's forward hook, i.e. on the worker thread
        between two decode steps, so everything it does is charged to the
        generation it is streaming. That is affordable for exactly one measured
        reason: codec decode is 36 ms and FLAT -- 35.7, 36.1 and 36.6 ms for
        1.50 s, 4.68 s and 9.20 s of audio (MEASURE_TTS_LATENCY.md section 3).
        A 6x change in length moves it 2.5 %, because the decode is
        dispatch-bound like everything else here. So re-decoding the whole
        prefix each time costs a constant, not a growing one.

        DECODING A PREFIX IS AN APPROXIMATION, AND HERE IS ITS SIZE. It was
        designed as an exact operation, on the argument that the decoder is
        causal -- left-only convolution padding, right-trimmed transposed
        convolutions, attention declaring `is_causal`. `probe_prefix_decode.py`
        falsified that: a prefix decode disagrees with the one-shot decode
        starting at SAMPLE ZERO, so it is not a boundary effect and no holdback
        removes it. `probe_decode_strategies.py` then located the dependence --
        25 frames of context reproduce the shipped waveform at 25.2 dB while 50
        frames reproduce it at 39.4 dB, identical to using the whole prefix, so
        the decoder's transformer reaches roughly 25-50 frames FORWARD in time
        whatever its `is_causal` flag says. Exactness would require holding
        back ~50 frames, four seconds of audio, which is the entire feature.

        So the shipped behaviour is measured instead of claimed:

        * **39.4 dB** against the one-shot decode, on a signal of RMS 0.068 and
          peak 0.438 -- about a 1 % error;
        * **seams of 0.0427 against the one-shot decode's own 0.0411** at the
          same sample positions. That control is the one that matters: speech
          is full of transients, and the question is not whether consecutive
          chunks join smoothly but whether they join less smoothly than the
          waveform already does there. 3.7 % is not a click.

        Using the whole prefix rather than a window is not arbitrary either --
        it is the best any incremental emitter can do, because the only context
        it lacks is audio that has not been generated yet.
        """
        done = len(frames)
        if not emitter.may_release(done, time.monotonic() - state["started"]):
            return
        stride = max(1, int(round(emitter.chunk_frames)))
        if done - int(state["decoded_at"]) < stride:
            return
        state["decoded_at"] = done
        waveform = self._decode_frames(frames)
        emitter.offer(waveform)
        silence = emitter.trailing_silence_seconds(waveform)
        if silence >= self.config.silence_cut_seconds:
            raise _StopUnit(
                f"{silence:.2f}s of digital silence after "
                f"{len(waveform) / self.sample_rate:.2f}s of audio, which is "
                f"the measured shape of a talker that said its words and then "
                f"failed to emit codec EOS; the silence is not on the wire and "
                f"the steps that would follow it only delay the next unit"
            )

    def _decode_frames(self, frames: List[object]) -> np.ndarray:
        """Codec frames -> waveform, through the reference's own decoder.

        Assembled exactly as the reference assembles `talker_codes`
        (modeling_qwen3_tts.py:2280): stack the per-step `(1, 16)` frames along
        a new step axis and take row 0, giving `(T, 16)`. Handed to the same
        public `speech_tokenizer.decode` the reference calls, so this is the
        shipped decode on the shipped inputs rather than a reimplementation of
        it. With `x_vector_only_mode=True` there is no reference-code prefix to
        cut afterwards (qwen3_tts_model.py:612-631), so the wrapper's output is
        the waveform.
        """
        import torch

        if not frames:
            return np.zeros(0, dtype=np.float32)
        inner = getattr(self._model, "model", self._model)
        codes = torch.stack(list(frames), dim=1)[0]
        with torch.inference_mode():
            wavs, _rate = inner.speech_tokenizer.decode([{"audio_codes": codes}])
        return np.asarray(wavs[0], dtype=np.float32).reshape(-1)

    def _check_stream_matches_burst(
        self, streamed: np.ndarray, burst: np.ndarray
    ) -> None:
        """Check that OUR decode of the tapped frames is the reference's decode.

        Narrower than it looks, and deliberately so. It does not check the
        prefix decodes -- those are known to differ, by a measured 39.4 dB.
        What it checks is that the frames pulled out of the forward hook are
        the same frames the reference itself would have decoded, by comparing a
        full decode of them against the waveform the reference returned from
        the very same generation. That is the assumption the tap rests on: that
        `hidden_states[-1]` per step, with the prefill entry dropped, is
        exactly `talker_codes`. The 2026-08-04 run found it exact on every arm
        (maximum absolute difference 0.0 across nine seeded pairs), which is
        also what rules out the tap perturbing the generation.

        Cheap: both arrays already exist and neither is on the card.
        """
        if len(streamed) != len(burst):
            logger.error(
                "streamed decode produced %d samples and the reference's own "
                "decode produced %d for the same generation: the prefix decode "
                "is not reproducing the shipped one",
                len(streamed), len(burst),
            )
            return
        if not len(streamed):
            return
        drift = float(np.abs(streamed - burst).max())
        if drift > 0.0:
            logger.warning(
                "a full decode of the tapped frames differs from the "
                "reference's own decode of the same generation by %.3e at its "
                "worst sample; the per-step tap is not seeing exactly the "
                "frames the reference decodes", drift,
            )

    @staticmethod
    def _to_waveform(output: object) -> np.ndarray:
        """Coerce whatever the reference returned into mono float32.

        Written defensively on purpose: the return shape is the least stable
        part of the reference API, and a wrong-shape coercion is a silent
        audio bug rather than a crash.
        """
        import torch

        candidate = output
        # Documented return is (List[np.ndarray], sample_rate); take the list
        # and ignore the rate, which the caller already knows.
        if (
            isinstance(candidate, tuple)
            and len(candidate) == 2
            and isinstance(candidate[1], (int, float))
        ):
            candidate = candidate[0]
        for attribute in ("audio_values", "waveform", "audios", "audio"):
            if hasattr(candidate, attribute):
                candidate = getattr(candidate, attribute)
                break
        if isinstance(candidate, dict):
            for key in ("audio_values", "waveform", "audio"):
                if key in candidate:
                    candidate = candidate[key]
                    break
        parts: List[np.ndarray] = []

        def flatten(item) -> None:
            if item is None:
                return
            if isinstance(item, (list, tuple)):
                for sub in item:
                    flatten(sub)
                return
            if torch.is_tensor(item):
                parts.append(item.detach().float().cpu().numpy().reshape(-1))
                return
            if isinstance(item, np.ndarray):
                parts.append(item.astype(np.float32).reshape(-1))

        flatten(candidate)
        if not parts:
            raise BackendError(
                "tts",
                f"could not find audio in the generation result "
                f"({type(output).__name__}); the reference API's return shape "
                "changed and coercing it blindly would be a silent audio bug",
            )
        return np.concatenate(parts).astype(np.float32)


def build_inprocess_tts(
    model_dir: Path,
    device: str = "cuda:0",
    languages: Optional[Sequence[str]] = None,
    **kwargs,
) -> InProcessQwen3Tts:
    """Convenience constructor used by the launcher."""
    del languages  # the checkpoint is the authority; see talker_config
    return InProcessQwen3Tts(
        InProcessTtsConfig(model_dir=Path(model_dir), device=device, **kwargs)
    )
