#!/usr/bin/env python3
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Does incremental emission actually start earlier, and still not gap?

The claim under test (#566): audio emitted as codec frames are produced turns
a 6.9 s wait into a pre-roll, without introducing a gap and without changing
which samples the listener hears.

Three things are measured, and the third is the one that could quietly be
wrong:

1. **time-to-first-audio**, burst versus streamed, same text and same SEED so
   the two arms decode the identical token sequence and the difference cannot
   be a different draw. An A-versus-A pair runs first and interleaved, because
   `MEASURE_TTS_LATENCY.md` established that whole-call latency has a 15 %
   standard deviation driven by the sampler's step count -- so no per-call
   delta under that is readable, and the floor has to be shown before any
   delta is;
2. **the emission schedule against a playback clock.** Every chunk is
   timestamped as it is pushed, then replayed against a listener who starts at
   the first chunk and consumes one second of audio per wall second. The
   reported number is the WORST buffer level over the whole utterance, not the
   first one: a fast start is not evidence of continuity, and the dangerous
   moment is the last sample;
3. **what the wire carried.** Two separate comparisons, kept apart because an
   early draft of this probe conflated them and read a pass into it.
   `sent_vs_result_max_abs` asks whether the concatenation of the chunks IS
   what the unit returned -- it must be exactly 0, since the two are the same
   array by construction, and anything else means the emitter dropped or
   duplicated audio. `max_abs_drift` asks how far that is from the waveform
   the burst path returns for the same seed, which is a property of the
   VENDORED CODEC rather than of this change: a prefix decode cannot see the
   end of the utterance, and `probe_decode_strategies.py` measures what that
   is worth (39.4 dB, seams within 4 % of the signal's own transients).

Runs a SECOND talker beside the live translator, which is what the latency
round did: ~2.05 GB of weights against the free VRAM on the 5090. The live
service (PID 272333) and the 27B on 30030 are not touched.

    /spinning/htsglang-gpu/.venv/bin/python \\
      scripts/translator/probe_stream_emission.py --repeats 3
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from sglang.srt.translator.backends import AudioChunk, TurnPacing  # noqa: E402
from sglang.srt.translator.inprocess_tts import (  # noqa: E402
    CODEC_FRAME_RATE_HZ,
    InProcessQwen3Tts,
    InProcessTtsConfig,
)

VOICES = Path("/spinning/llm_stuff/translator-models/preset-voices")
RATE = 24000
REF_S = 3.22

#: The same real translations the step budget was fitted against, so the
#: character counts are the ones the model was measured on rather than
#: invented strings.
TEXTS = {
    "short": "Para nada.",
    "field": "Hola, soy Matthias y estoy de vacaciones aqui. Como estas?",
    "long": (
        "No es mas rapido en absoluto. Para nada. Que pasa? "
        "Hola, soy Matthias y estoy de vacaciones aqui, como estas hoy?"
    ),
    #: THE SHAPE THAT BROKE, added 2026-08-04 for #466 (d). The field package
    #: `client-20260804T175817Z` carried two mid-turn underruns and both sat
    #: immediately before the FINAL chunk of a long turn: turn acac91a3
    #: (talk_ms 5114) re-anchored at cursor_lead_ms -241, turn 57e07979
    #: (talk_ms 3469) at -1603. A turn of ~5 s of speech translates to ~8 s of
    #: Spanish, which at the fit `steps = 14 + 0.7 x chars` needs ~100 frames
    #: and therefore ~123 characters. The shorter texts above cannot reproduce
    #: it: at RTF > 1 the buffer deficit grows with DURATION, so a turn has to
    #: be long enough for the pre-roll to run out before the end.
    "monologue": (
        "Estamos de vacaciones aqui en la costa desde el martes pasado, "
        "y manana por la manana queremos ir a la playa temprano. Hace calor."
    ),
}


class RecordingSink:
    """A sink that timestamps every chunk the emitter decides to send."""

    def __init__(self) -> None:
        self.events: List[Dict[str, float]] = []
        self.chunks: List[AudioChunk] = []
        self.started = time.monotonic()

    def push(self, chunk: Optional[AudioChunk]) -> None:
        if chunk is None:
            return
        self.chunks.append(chunk)
        self.events.append(
            {"at_s": time.monotonic() - self.started,
             "samples": int(len(chunk.samples))}
        )

    def close(self) -> None:
        pass

    def audio(self) -> np.ndarray:
        if not self.chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate([c.samples for c in self.chunks])


def worst_buffer_seconds(events: List[Dict[str, float]]) -> Optional[float]:
    """Replay the emission schedule against a listener's clock.

    The client schedules each arriving buffer at the end of the previous one on
    a shared cursor, so playback advances in real time from the first chunk. If
    the cursor ever catches up to now, the client re-anchors 50 ms ahead AND
    re-applies its 8 ms onset ramp -- so an underrun here is not merely a gap,
    it is an audible click in the middle of a word.
    """
    if not events:
        return None
    start = events[0]["at_s"]
    emitted = 0.0
    worst = None
    for index, event in enumerate(events):
        # Sampled just BEFORE the chunk lands, which is where the trough
        # actually is: the buffer drains continuously and is refilled in steps,
        # so measuring it after each arrival reports the peak of the sawtooth
        # and would call a stream healthy that ran dry between every chunk.
        if index:
            buffered = emitted - (event["at_s"] - start)
            worst = buffered if worst is None else min(worst, buffered)
        emitted += event["samples"] / float(RATE)
    # Nothing arrives behind the last chunk, so the buffer cannot trough again.
    return worst


def load_reference() -> AudioChunk:
    path = VOICES / "man" / "man-03.de.wav"
    data, rate = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if rate != RATE:
        raise SystemExit(f"{path} is {rate} Hz, expected {RATE}")
    return AudioChunk(data[: int(REF_S * RATE)], RATE)


def run_one(backend, torch, text: str, reference, seed: int, streaming: bool) -> Dict:
    backend.config = dataclasses.replace(
        backend.config, stream_within_unit=streaming
    )
    sink = RecordingSink()
    pacing = TurnPacing()
    # Same seed on both arms: the talker samples from the global generator, so
    # this is what makes the burst and the stream decode the identical token
    # sequence. Without it the two arms differ by a draw whose step count has a
    # 15 % standard deviation, which is larger than the effect.
    torch.manual_seed(seed)
    started = time.monotonic()
    waveform = backend._generate(text, "es", reference, None, sink, pacing)
    total = time.monotonic() - started
    ttfa = sink.events[0]["at_s"] if sink.events else total
    audio_s = len(waveform) / RATE
    streamed = sink.audio()
    return {
        "streaming": streaming,
        "seed": seed,
        "chars": len(text),
        "ttfa_ms": ttfa * 1000.0,
        "total_ms": total * 1000.0,
        "audio_s": audio_s,
        "frames": audio_s * CODEC_FRAME_RATE_HZ,
        "rtf": total / audio_s if audio_s else None,
        "chunks": len(sink.events),
        "worst_buffer_s": worst_buffer_seconds(sink.events),
        "streamed_samples": int(len(streamed)),
        "result_samples": int(len(waveform)),
        "identical": bool(
            len(streamed) == len(waveform)
            and np.array_equal(streamed, waveform)
        ),
        # How far the concatenation of what was sent is from what the unit
        # returned. Reported as a number rather than a boolean because the
        # interesting question is not "equal?" but "by how much, and is that
        # the decoder's shape-dependent kernel choice or a real seam?".
        "sent_vs_result_max_abs": (
            float(np.abs(streamed - waveform).max())
            if len(streamed) == len(waveform) and len(streamed) else None
        ),
        "waveform": waveform,
        "streamed": streamed,
        "events": sink.events,
    }


def force_runaway(backend, torch, reference, factors=(0.35, 0.80)) -> Dict:
    """Make every draw overrun, and watch the conflict resolve on real audio.

    The runaway is stochastic -- about one generation in ten -- so waiting for
    one is not a test. `step_budget_factor` is instead driven below 1.0, which
    makes a perfectly healthy generation exceed its budget and takes the
    re-draw path deterministically. The factor also decides WHICH branch is
    reached, and both are wanted:

    * **0.35** puts the budget below the release point, so the generation is
      stopped before a single chunk was sent. Nothing has been emitted, the
      draw can still be replaced, and the #564 re-draw must fire: two draws.
      This is the control -- it proves the re-draw is still reachable at all.
    * **0.80** puts the budget well above the release point, so audio is
      already on the wire when the overrun is detected. The draw can no longer
      be replaced and the unit must be truncated instead: ONE draw, with every
      sample that was sent still present in what the unit returns.

    The second arm is the cost of this lever, executed rather than asserted.
    """
    original = backend.config
    results = {}
    try:
        # The factors are a parameter rather than a constant because WHICH
        # branch a factor reaches depends on the step RATE. The release point
        # is where the pre-roll has been banked, and a faster talker banks it
        # in fewer frames -- so a budget that sat below the release point at
        # 10.9 steps/s can sit above it at 11.7. The two branches below are the
        # thing being tested; the numbers that reach them are not fixed.
        for name, factor in ((n, f) for f in factors for n in ("field",)):
            backend.config = dataclasses.replace(
                original, step_budget_factor=factor, stream_within_unit=True
            )
            draws = {"n": 0}
            real = backend._generate_once

            def counting(*args, **kwargs):
                draws["n"] += 1
                return real(*args, **kwargs)

            backend._generate_once = counting
            sink = RecordingSink()
            torch.manual_seed(4242)
            try:
                waveform = backend._generate(
                    TEXTS[name], "es", reference, None, sink, TurnPacing()
                )
            finally:
                backend._generate_once = real
            sent = sink.audio()
            results[f"{name}_factor_{factor}"] = {
                "factor": factor,
                "draws": draws["n"],
                "budget_frames": backend.step_budget(TEXTS[name]),
                "chunks": len(sink.events),
                "sent_samples": int(len(sent)),
                "result_samples": int(len(waveform)),
                "sent_is_prefix_of_result": bool(
                    len(sent) <= len(waveform)
                    and np.array_equal(sent, waveform[: len(sent)])),
                "streamed": bool(len(sink.events) > 1),
            }
            print("forced runaway", name, factor,
                  json.dumps(results[f"{name}_factor_{factor}"]))
    finally:
        backend.config = original
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out", default="/spinning/466-client-logs/stream_emission.json")
    parser.add_argument("--texts", default="short,field,long")
    parser.add_argument("--runaway", action="store_true",
                        help="also run the forced-overrun arm")
    parser.add_argument("--runaway-factors", default="0.35,0.80",
                        help="step-budget factors to force an overrun at")
    parser.add_argument(
        "--code-predictor-loop", choices=("on", "off"), default="on",
        help="run the code predictor as one unrolled loop (on) or as 15 "
             "nested generate() calls (off). The step RATE is what decides "
             "whether the schedule below stays ahead of the playback clock, "
             "so this is the arm dimension for #466 (d); the emitter itself "
             "is unchanged either way.",
    )
    args = parser.parse_args()

    import torch

    from sglang.srt.translator import code_predictor_loop

    reference = load_reference()
    backend = InProcessQwen3Tts(InProcessTtsConfig())
    backend.load()
    code_predictor_loop.set_enabled(args.code_predictor_loop == "on")

    # JIT and allocator outliers, discarded exactly as the latency round did.
    for _ in range(args.warmup):
        torch.manual_seed(1)
        backend._generate(TEXTS["field"], "es", reference, None, None, None)

    records: List[Dict] = []
    pairs: List[Dict] = []
    names = [name.strip() for name in args.texts.split(",") if name.strip()]
    for repeat in range(args.repeats):
        for name in names:
            text = TEXTS[name]
            seed = 1000 + repeat
            # INTERLEAVED, not blocked: a card that drifts during the run must
            # not be able to masquerade as an arm effect.
            burst = run_one(backend, torch, text, reference, seed, False)
            stream = run_one(backend, torch, text, reference, seed, True)
            for record in (burst, stream):
                keep = {k: v for k, v in record.items()
                        if k not in ("waveform", "streamed", "events")}
                keep["text"] = name
                keep["repeat"] = repeat
                records.append(keep)
            same_len = len(burst["waveform"]) == len(stream["waveform"])
            drift = (
                float(np.abs(burst["waveform"] - stream["waveform"]).max())
                if same_len and len(burst["waveform"]) else None
            )
            pairs.append({
                "text": name, "repeat": repeat, "seed": seed,
                "burst_ttfa_ms": burst["ttfa_ms"],
                "stream_ttfa_ms": stream["ttfa_ms"],
                "burst_chunks": burst["chunks"],
                "stream_chunks": stream["chunks"],
                "burst_audio_s": burst["audio_s"],
                "stream_audio_s": stream["audio_s"],
                "same_length": same_len,
                "max_abs_drift": drift,
                "stream_is_prefix_exact": stream["identical"],
                "sent_vs_result_max_abs": stream["sent_vs_result_max_abs"],
                "worst_buffer_s": stream["worst_buffer_s"],
                "expected_frames_fit": 14 + 0.7 * len(TEXTS[name]),
                "actual_frames": stream["frames"],
            })
            print(json.dumps(pairs[-1], indent=None))

    summary: Dict[str, Dict] = {}
    for name in names:
        rows = [p for p in pairs if p["text"] == name]
        summary[name] = {
            "n": len(rows),
            "burst_ttfa_ms_median": statistics.median(
                r["burst_ttfa_ms"] for r in rows),
            "stream_ttfa_ms_median": statistics.median(
                r["stream_ttfa_ms"] for r in rows),
            "worst_buffer_s_min": min(
                (r["worst_buffer_s"] for r in rows
                 if r["worst_buffer_s"] is not None), default=None),
            "all_byte_identical": all(
                r["same_length"] and (r["max_abs_drift"] == 0.0)
                for r in rows),
            "all_stream_is_prefix_exact": all(
                r["stream_is_prefix_exact"] for r in rows),
            "sent_vs_result_max_abs": max(
                (r["sent_vs_result_max_abs"] for r in rows
                 if r["sent_vs_result_max_abs"] is not None), default=None),
            "fit_error_max": max(
                r["actual_frames"] / r["expected_frames_fit"] for r in rows),
            "chunk_counts_match": all(
                r["burst_chunks"] == r["stream_chunks"] for r in rows),
        }

    factors = tuple(float(f) for f in args.runaway_factors.split(","))
    runaway = (force_runaway(backend, torch, reference, factors)
               if args.runaway else {})

    Path(args.out).write_text(json.dumps(
        {"code_predictor_loop": args.code_predictor_loop,
         "code_predictor_loop_calls": code_predictor_loop.loop_stats(),
         "summary": summary, "pairs": pairs, "records": records,
         "forced_runaway": runaway}, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
