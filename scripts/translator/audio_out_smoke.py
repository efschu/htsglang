# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""End-to-end audio out: reference voice -> talker -> code predictor -> codec -> Opus.

The audible acceptance run for rung B. It drives the real
:class:`InProcessQwen3Tts` -- no fakes, no stubs on the audio path -- with a
REAL German speech clip as the cloning reference and Spanish text as the
target, which is the cross-lingual direction the whole project exists for.
The result is written as a wav so a human can listen to it, and pushed through
the Opus codec on the way so the transport leg is exercised rather than assumed.

    CUDA_VISIBLE_DEVICES=99 PYTHONPATH=<repo>/python \\
      /spinning/htsglang-gpu/.venv/bin/python \\
      scripts/translator/audio_out_smoke.py --out /tmp/de2es.wav

CPU by default: the point is that it produces correct audio at all, not how
fast. Pass ``--device cuda:0 --dtype bfloat16`` inside an arbitrated GPU window
for the latency number.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_REFERENCE = Path(
    "/spinning/llm_stuff/translator-models/xtts-v2/samples/de_sample.wav"
)
DEFAULT_TEXT = "Hola, buenos dias. Me alegro mucho de verte otra vez."


async def run(args) -> int:
    from sglang.srt.translator.audio import (
        AudioChunk,
        available_codecs,
        negotiate_codec,
    )
    from sglang.srt.translator.inprocess_tts import (
        InProcessQwen3Tts,
        InProcessTtsConfig,
    )

    import soundfile as sf

    reference_samples, reference_rate = sf.read(str(args.reference), dtype="float32")
    if reference_samples.ndim > 1:
        reference_samples = reference_samples.mean(axis=1)
    reference = AudioChunk(np.ascontiguousarray(reference_samples), reference_rate)
    print(
        f"[smoke] reference {args.reference.name}: "
        f"{reference.duration_s:.2f}s @ {reference_rate} Hz"
    )

    backend = InProcessQwen3Tts(
        InProcessTtsConfig(
            model_dir=args.model_dir,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            # Greedy-ish: the smoke wants a reproducible artefact, not variety.
            temperature=args.temperature,
        )
    )
    print(f"[smoke] languages: {list(backend.supported_languages())}")
    if args.language not in backend.supported_languages():
        print(f"[smoke] FAILED: checkpoint cannot speak {args.language!r}")
        return 1

    started = time.monotonic()
    backend.load()
    print(f"[smoke] loaded in {time.monotonic() - started:.1f}s")

    started = time.monotonic()
    chunks = []
    first_chunk_at = None
    async for chunk in backend.synthesize(
        text=args.text, language=args.language, reference=reference
    ):
        if first_chunk_at is None:
            first_chunk_at = time.monotonic() - started
        chunks.append(chunk.samples)
    elapsed = time.monotonic() - started

    if not chunks:
        print("[smoke] FAILED: the backend produced no audio at all")
        return 1

    waveform = np.concatenate(chunks).astype(np.float32)
    duration = len(waveform) / backend.sample_rate
    peak = float(np.abs(waveform).max())
    finite = bool(np.isfinite(waveform).all())
    print(
        f"[smoke] {len(chunks)} chunks, {duration:.2f}s of audio in {elapsed:.1f}s "
        f"(RTF {elapsed / duration:.2f}), first chunk at {first_chunk_at:.1f}s"
    )
    print(f"[smoke] peak {peak:.3f}, finite {finite}")

    if not finite:
        print("[smoke] FAILED: the waveform contains non-finite samples")
        return 1
    if peak < 1e-3:
        print("[smoke] FAILED: the waveform is silence")
        return 1
    if duration < 0.5:
        print(f"[smoke] FAILED: {duration:.2f}s is too short to be an utterance")
        return 1

    # Write the artefact BEFORE the transport leg: a codec problem must not
    # cost the audio that took minutes to generate.
    sf.write(str(args.out), waveform, backend.sample_rate)
    print(f"[smoke] wrote {args.out}")

    # The transport leg, on the real waveform rather than on a tone.
    print(f"[smoke] codecs available: {available_codecs()}")
    codec = negotiate_codec(["opus", "pcm16"])
    full = AudioChunk(waveform, backend.sample_rate)
    packets = list(codec.encode(full))
    decoded_parts = [codec.decode(packet).samples for packet in packets]
    decoded = AudioChunk(
        np.concatenate(decoded_parts).astype(np.float32)
        if decoded_parts
        else np.zeros(0, dtype=np.float32),
        codec.sample_rate,
    )
    payload = sum(len(packet) for packet in packets)
    print(
        f"[smoke] {codec.name}: {len(packets)} packets, {payload} bytes for "
        f"{duration:.2f}s ({payload * 8 / duration / 1000:.1f} kbps), "
        f"decoded {decoded.duration_s:.2f}s @ {decoded.sample_rate} Hz"
    )
    if decoded.duration_s < duration * 0.5:
        print("[smoke] FAILED: the codec round trip lost most of the audio")
        return 1

    if args.out_codec is not None:
        sf.write(str(args.out_codec), decoded.samples, decoded.sample_rate)
        print(f"[smoke] wrote {args.out_codec} (after the {codec.name} round trip)")

    report_speech_structure(waveform, backend.sample_rate)
    report_speaker_similarity(backend, reference, waveform)
    return 0


def report_speech_structure(waveform: np.ndarray, rate: int) -> None:
    """Is this speech, or is it a tone / noise / silence?

    Nobody in this loop can listen, so the audible claim needs an instrument.
    The one that separates speech from every plausible failure is the ENERGY
    ENVELOPE's modulation rate: connected speech modulates at the syllable
    rate, 3-8 Hz. A held tone, white noise and a DC-ish artefact are all flat
    there, and a truncated or silent output has no envelope at all. It does not
    prove the words are right -- only an ASR round trip does that -- but it
    does separate "speech-shaped" from "the model emitted something".
    """
    frame = max(1, rate // 100)  # 10 ms envelope, 100 Hz sampling
    frames = len(waveform) // frame
    envelope = np.abs(waveform[: frames * frame].reshape(frames, frame)).mean(axis=1)
    if frames < 64:
        print("[smoke] structure: too short to characterise")
        return
    voiced = float((envelope > 0.05 * envelope.max()).mean())
    centred = envelope - envelope.mean()
    spectrum = np.abs(np.fft.rfft(centred * np.hanning(len(centred))))
    freqs = np.fft.rfftfreq(len(centred), d=1.0 / 100.0)
    band = (freqs >= 2.0) & (freqs <= 10.0)
    peak_hz = float(freqs[band][np.argmax(spectrum[band])])
    syllabic = float(spectrum[band].sum() / max(spectrum[1:].sum(), 1e-9))
    print(
        f"[smoke] structure: voiced fraction {voiced:.2f}, envelope peak "
        f"{peak_hz:.1f} Hz, syllabic-band share {syllabic:.2f} "
        f"(speech: peak 3-8 Hz, share > 0.3)"
    )


def report_speaker_similarity(backend, reference, waveform: np.ndarray) -> None:
    """Cosine between the reference's x-vector and the output's.

    The checkpoint carries the speaker encoder that the cloning conditions on,
    so the similarity metric the design's A/B specifies is available here for
    free -- and it is the one number that says whether the VOICE carried across
    the language boundary, which is requirement 1.
    """
    import torch

    inner = getattr(backend._model, "model", backend._model)
    extract = getattr(inner, "extract_speaker_embedding", None)
    if extract is None:
        print("[smoke] similarity: this checkpoint exposes no speaker encoder")
        return
    rate = inner.speaker_encoder_sample_rate

    from sglang.srt.translator.qwen3_tts_compat import librosa_resample

    def embed(samples: np.ndarray, source_rate: int):
        resampled = librosa_resample(samples, source_rate, rate)
        with torch.inference_mode():
            return extract(audio=resampled, sr=rate).detach().float().reshape(-1)

    try:
        ref_vector = embed(reference.samples, reference.sample_rate)
        out_vector = embed(waveform, backend.sample_rate)
    except Exception as exc:  # noqa: BLE001 - reporting, not handling
        print(f"[smoke] similarity: unavailable ({type(exc).__name__}: {exc})")
        return
    cosine = float(
        torch.nn.functional.cosine_similarity(ref_vector, out_vector, dim=0)
    )
    print(
        f"[smoke] speaker similarity (reference vs output x-vector): "
        f"{cosine:.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--language", default="es")
    parser.add_argument("--out", type=Path, default=Path("/tmp/translator_de2es.wav"))
    parser.add_argument("--out-codec", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/spinning/llm_stuff/translator-models/qwen3-tts-0.6b-base"),
    )
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
