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
* no cross-request batching -- one conversation at a time, which is the MVP;
* **no true incremental streaming.** The reference generates a whole utterance
  and then decodes it. We chunk the finished waveform so the transport and the
  client see the same interface they always did, but the first-audio latency is
  whole-utterance, not first-frame. Since ``mt.py`` already splits translations
  into clauses and synthesizes them one at a time, the practical unit is a
  clause rather than a turn -- which is most of the benefit and none of the
  decode-loop surgery. The gap is stated rather than hidden, and the interface
  does not change when the native lane closes it.

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

__all__ = ["InProcessQwen3Tts", "InProcessTtsConfig"]


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
    #: Emit the finished waveform in chunks of this length, so the transport
    #: and the client see a stream. Not true incremental synthesis -- see the
    #: module docstring.
    emit_chunk_seconds: float = 0.4
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
        waveform = await self._generate_under_lock(
            text, language, reference, reference_text, queued_at
        )

        chunk = max(1, int(self.config.emit_chunk_seconds * self.sample_rate))
        for start in range(0, len(waveform), chunk):
            yield AudioChunk(waveform[start : start + chunk], self.sample_rate)

        if self.config.park_when_idle:
            self.park()

    async def _generate_under_lock(
        self,
        text: str,
        language: str,
        reference: AudioChunk,
        reference_text: Optional[str],
        queued_at: float,
    ) -> np.ndarray:
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
                self._generate, text, language, reference, reference_text
            )
        except BaseException:
            # Nothing was submitted, so no callback will ever fire: this is
            # the one path that has to release the lock itself.
            self._lock.release()
            raise
        # Fires on the worker thread the moment `_generate` returns or raises,
        # including long after this coroutine was cancelled. Exactly once per
        # submit, so the lock cannot be released twice.
        pending.add_done_callback(
            lambda _finished: loop.call_soon_threadsafe(self._release_talker)
        )
        return await asyncio.wrap_future(pending, loop=loop)

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

    def _generate(
        self,
        text: str,
        language: str,
        reference: AudioChunk,
        reference_text: Optional[str],
    ) -> np.ndarray:
        """Synthesize one unit, re-drawing a generation that ran away.

        The retry is what makes this a repair rather than a shorter leash: a
        generation that hits the text's budget did not stop on its own, the
        failure is a stochastic property of one draw, and a fresh draw is
        overwhelmingly likely to be healthy. A truncated attempt is kept only if
        every draw failed -- and it still opens with the correct words, because
        the runaway is in the tail.
        """
        budget = self.step_budget(text)
        attempt = 0
        waveform = None
        while True:
            waveform = self._generate_once(
                text, language, reference, reference_text, budget
            )
            frames = len(waveform) / self.sample_rate * self.geometry.frame_rate_hz
            # One frame of slack: the codec's output length is a multiple of its
            # hop, so an exact comparison against the budget would be a
            # coin flip on rounding.
            if frames < budget - 1 or attempt >= self.config.runaway_retries:
                break
            attempt += 1
            logger.warning(
                "talker did not stop by itself: %.0f steps for %d characters "
                "(budget %d, %.2fs of audio); re-drawing (attempt %d of %d)",
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
        return trimmed

    def _generate_once(
        self,
        text: str,
        language: str,
        reference: AudioChunk,
        reference_text: Optional[str],
        max_new_tokens: int,
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
        with torch.inference_mode():
            output = self._model.generate_voice_clone(
                text=[text],
                language=[model_language],
                ref_audio=[(reference_np, reference.sample_rate)],
                ref_text=[reference_text or ""],
                x_vector_only_mode=self.config.x_vector_only_mode,
                non_streaming_mode=True,
                # The TEXT's budget, not the module constant. `max_new_tokens`
                # in the config remains the absolute ceiling and `step_budget`
                # never exceeds it.
                max_new_tokens=max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )
        return self._to_waveform(output)

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
