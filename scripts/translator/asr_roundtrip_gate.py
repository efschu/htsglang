# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The intelligibility gate (DESIGN_466 §7(b)(3)), executed.

Transcribes synthesized audio with the real recognizer and scores WER against
the text that was requested. A candidate above the threshold is out regardless
of how good it sounds; accent is never scored.

Two arms are run whenever a control clip is supplied, and the control is the
point: it is the audio produced by the *randomly initialised* talker, which is
finite, speech-shaped, correctly pitched and scores 0.986 on speaker
similarity. If the gate cannot fail on that, the gate is decoration. This is
the can-fail proof, and it runs every time rather than being asserted once.

The recognizer also exercises the routing whitelist on the REAL model:
constrained detection with the target language in the set, and with it removed
so the out-of-set fallback path is executed rather than reasoned about.

faster-whisper lives in its own tree and is APPENDED to ``sys.path``, never
prepended: the shared serving venv is not modified, and its own numpy /
tokenizers keep winning inside this process.

    CUDA_VISIBLE_DEVICES=99 PYTHONPATH=<repo>/python \\
      /spinning/htsglang-gpu/.venv/bin/python \\
      scripts/translator/asr_roundtrip_gate.py \\
        --audio /path/to/out.wav --text "the sentence that was asked for" \\
        --control /path/to/babble.wav
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

ASR_LIB = Path("/spinning/llm_stuff/translator-models/asr-lib")
ASR_MODELS = Path("/spinning/llm_stuff/translator-models/asr-models")

#: DESIGN_466 §7(b)(3) leaves the threshold to the measurement; this is it.
#: Rationale, so a future reader does not have to guess at it: the checkpoint's
#: published in-language WER is ~1 %, a round trip adds the recognizer's own
#: error on a few seconds of audio, and cross-lingual cloning degrades
#: pronunciation rather than content. 0.15 is therefore loose enough that a
#: genuinely intelligible utterance cannot fail it on ASR noise, and tight
#: enough that babble (which scores near or above 1.0) cannot pass. It is a
#: gate, not a quality score -- ranking happens on speaker similarity.
DEFAULT_THRESHOLD = 0.15


def _enable_asr_library() -> None:
    if str(ASR_LIB) not in sys.path:
        # APPEND: the venv's numpy/tokenizers must keep priority.
        sys.path.append(str(ASR_LIB))


async def run(args) -> int:
    _enable_asr_library()

    import numpy as np
    import soundfile as sf

    from sglang.srt.translator.backends import AudioChunk
    from sglang.srt.translator.scoring import word_error_rate

    try:
        from sglang.srt.translator.asr_backends import FasterWhisperAsr
    except ImportError as exc:  # pragma: no cover - environment dependent
        print(f"[gate] cannot import the recognizer adapter: {exc}")
        return 1

    started = time.monotonic()
    asr = FasterWhisperAsr(
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        download_root=ASR_MODELS,
        restrict_languages=args.restrict.split(",") if args.restrict else (),
        beam_size=args.beam_size,
    )
    print(f"[gate] {asr.name} ready in {time.monotonic() - started:.1f}s "
          f"({args.device}, {args.compute_type})")

    def load(path: Path) -> AudioChunk:
        samples, rate = sf.read(str(path), dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        return AudioChunk(np.ascontiguousarray(samples), rate)

    async def transcribe(path: Path) -> tuple:
        audio = load(path)
        t0 = time.monotonic()
        transcript = await asr.transcribe(audio)
        return transcript, audio.duration_s, time.monotonic() - t0

    # -- arm 1: the real output -------------------------------------------
    transcript, duration, elapsed = await transcribe(args.audio)
    result = word_error_rate(args.text, transcript.text)
    print(f"\n[gate] ARM: synthesized output  ({args.audio.name})")
    print(f"[gate]   asked for : {args.text}")
    print(f"[gate]   heard     : {transcript.text}")
    print(f"[gate]   language  : {transcript.language} "
          f"(confidence {transcript.language_confidence:.3f})")
    print(f"[gate]   WER {result.rate:.3f}  "
          f"(sub {result.substitutions} del {result.deletions} "
          f"ins {result.insertions}, ref {result.reference_words} words)")
    print(f"[gate]   {duration:.2f}s of audio transcribed in {elapsed:.2f}s")

    # -- arm 2: the can-fail proof ----------------------------------------
    control = None
    if args.control is not None and args.control.exists():
        control_transcript, control_duration, _ = await transcribe(args.control)
        control = word_error_rate(args.text, control_transcript.text)
        print(f"\n[gate] CONTROL: pre-fix babble  ({args.control.name})")
        print(f"[gate]   heard     : {control_transcript.text[:160]}")
        print(f"[gate]   language  : {control_transcript.language} "
              f"(confidence {control_transcript.language_confidence:.3f})")
        print(f"[gate]   WER {control.rate:.3f}  "
              f"({control.hypothesis_words} words for "
              f"{control_duration:.1f}s of audio)")

    # -- the constrained-detection paths, on the real model ---------------
    if args.check_whitelist:
        audio = load(args.audio)
        print("\n[gate] constrained detection on the real recognizer:")
        for label, whitelist in (
            ("target IN the set", args.whitelist_in.split(",")),
            ("target NOT in the set", args.whitelist_out.split(",")),
        ):
            asr.set_restrict_languages(whitelist)
            probe = await asr.transcribe(audio)
            print(f"[gate]   {label:22s} {tuple(whitelist)} -> "
                  f"{probe.language!r} at {probe.language_confidence:.3f}")
            if probe.language not in whitelist:
                print("[gate]   FAILED: constrained detection returned a "
                      "language outside the whitelist")
                return 1
            if not probe.text.strip():
                print("[gate]   FAILED: an utterance was discarded rather "
                      "than resolved to an in-set language")
                return 1

    # -- verdict ----------------------------------------------------------
    print("")
    passed = result.rate <= args.threshold
    print(f"[gate] threshold {args.threshold:.2f}: "
          f"{'PASS' if passed else 'FAIL'} at WER {result.rate:.3f}")
    if control is not None:
        if control.rate <= args.threshold:
            print("[gate] THE GATE IS BROKEN: the control babble passed it. "
                  "A gate that cannot fail on random-talker output is "
                  "decoration; fix the gate before trusting the arm above.")
            return 1
        print(f"[gate] can-fail proof: control scored {control.rate:.3f} "
              f"> {args.threshold:.2f} and is correctly rejected")
    return 0 if passed else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--control", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--restrict", default="")
    parser.add_argument("--check-whitelist", action="store_true")
    parser.add_argument("--whitelist-in", default="de,es")
    parser.add_argument("--whitelist-out", default="de,fr")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
