#!/usr/bin/env python3
"""Why in-process Qwen3-TTS keeps generating after the utterance is over.

FIELD EVIDENCE (2026-08-04, package client-20260804T141043Z). One session
produced 5952 audio buffers of 320 samples at 16 kHz -- 119 s of synthesized
speech -- for 13.6 s of user speech, and two turns ran until the wall-clock
ceiling in ``InProcessTtsConfig.max_generation_seconds`` (70 s) stopped them:
``since_release_ms`` 70683 and 70917. The user hears the correct word followed
by a minute of babble and has to press stop.

THE HYPOTHESIS UNDER TEST. In the same session the user merged two speakers at
t=30586 ms, and the clone reference of the surviving profile grew from 3.22 s
to 7.74 s (= 3.22 + 4.52, the two profiles' buffers concatenated by
``SpeakerProfile.reference_audio``). Both runaway turns are after the merge;
both turns before it were normal. That is a correlation in ONE log, so this
probe separates the two things the merge changed at once:

  A  one speaker, 3.22 s              the pre-merge baseline
  B  one speaker, 7.74 s              reference LENGTH alone
  C  two speakers spliced, 7.74 s     the merge: length AND a blended voice
  D  two speakers spliced, 7.74 s     a second pair, so C cannot be one unlucky
                                      voice pair

Only C and D carry the merge's second change. If B is healthy and C/D run away,
the defect is the blended speaker embedding, not the length. If B runs away too,
it is the length. If nothing runs away, the merge is not the mechanism and the
cause is elsewhere (see the module docstring of ``inprocess_tts`` for the other
candidate: a talker whose weights did not load produces exactly this symptom).

WHY CPU. The rig's cards carry the live translator the user is field-testing;
this probe must not compete for them. Everything here is the same code path --
``InProcessQwen3Tts`` builds one prompt and calls ``generate_voice_clone`` -- so
the only thing CPU changes is the wall clock, and the measurement is in STEPS.

WHY max_new_tokens IS LOWERED HERE. A healthy clause costs ~85 talker steps
(measured 2026-08-04, 12.5 steps/s). The shipped ceiling of 800 exists to bound
a runaway, and on CPU it would cost minutes per run. 300 steps is ~3.5x what a
healthy clause needs, so "reached the cap" still means "did not stop by itself"
and no healthy generation can touch it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from sglang.srt.translator.audio import AudioChunk  # noqa: E402
from sglang.srt.translator.inprocess_tts import (  # noqa: E402
    InProcessQwen3Tts,
    InProcessTtsConfig,
)

VOICES = Path("/spinning/llm_stuff/translator-models/preset-voices")
RATE = 24000
#: The field durations, kept exactly: the merged profile in the incident was
#: 3.22 s + 4.52 s of admitted reference audio.
SHORT_S = 3.22
SECOND_S = 4.52


def load(path: Path, seconds: float) -> np.ndarray:
    data, rate = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if rate != RATE:
        raise SystemExit(f"{path} is {rate} Hz, expected {RATE}")
    want = int(seconds * RATE)
    if len(data) < want:
        raise SystemExit(f"{path} is {len(data)/RATE:.2f}s, need {seconds}s")
    return data[:want]


def arms() -> dict:
    a = VOICES / "man" / "man-03.de.wav"        # 10.0 s, one speaker
    b = VOICES / "woman" / "woman-01.de.wav"    # 8.0 s, a different speaker
    c = VOICES / "man" / "man-06.de.wav"        # 10.88 s, second pair
    d = VOICES / "woman" / "woman-03.de.wav"    # 8.64 s, second pair
    return {
        "A_one_speaker_3.22s": load(a, SHORT_S),
        "B_one_speaker_7.74s": load(a, SHORT_S + SECOND_S),
        "C_two_speakers_7.74s": np.concatenate([load(a, SHORT_S), load(b, SECOND_S)]),
        "D_two_speakers_7.74s_pair2": np.concatenate([load(c, SHORT_S), load(d, SECOND_S)]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--text", default="Gracias.")
    ap.add_argument("--language", default="es")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only", default="", help="comma-separated arm prefixes")
    ap.add_argument("--embeddings-only", action="store_true")
    # THE SAMPLING AXIS. The shipped decode is temperature 0.9 with top_p 0.9;
    # the checkpoint's own generation_config.json asks for top_p 1.0 with
    # top_k 50. A nucleus that is tighter than the checkpoint's can drop the
    # codec EOS out of the sampled set at exactly the step where the utterance
    # should end, so this has to be measurable rather than assumed.
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--save-dir", default="")
    args = ap.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO)

    import torch

    torch.set_num_threads(args.threads)

    base = InProcessTtsConfig()
    config = dataclasses.replace(
        base,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=base.temperature if args.temperature is None else args.temperature,
        top_p=base.top_p if args.top_p is None else args.top_p,
        top_k=args.top_k,
        do_sample=not args.greedy,
        # The wall clock must not be what stops a run here, or every arm would
        # report the same number and the experiment would measure the ceiling
        # instead of the model.
        max_generation_seconds=100000.0,
    )
    tts = InProcessQwen3Tts(config)
    t0 = time.monotonic()
    tts.load()
    print(f"loaded on {args.device}/{args.dtype} in {time.monotonic() - t0:.1f}s", flush=True)
    print(f"decode[{args.label or 'shipped'}]: do_sample={config.do_sample} "
          f"temperature={config.temperature} top_p={config.top_p} "
          f"top_k={config.top_k} max_new_tokens={config.max_new_tokens}", flush=True)

    material = arms()
    if args.only:
        keep = tuple(args.only.split(","))
        material = {k: v for k, v in material.items() if k.startswith(keep)}

    # Stage 1: the speaker embedding itself, which is the ONLY thing the
    # reference contributes in x_vector_only mode (ref_code is None, see
    # qwen3_tts_model.create_voice_clone_prompt). Cheap, and it says whether the
    # merged reference lands anywhere near a real voice.
    inner = getattr(tts._model, "model", tts._model)
    embeds = {}
    for name, wave in material.items():
        emb = inner.extract_speaker_embedding(audio=wave, sr=RATE)
        embeds[name] = emb.detach().float().cpu().numpy().reshape(-1)
        print(f"embedding {name:28s} dim={embeds[name].size} "
              f"norm={np.linalg.norm(embeds[name]):.4f}", flush=True)
    names = list(embeds)
    for i, one in enumerate(names):
        for other in names[i + 1:]:
            u, v = embeds[one], embeds[other]
            cos = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
            print(f"cosine {one} <-> {other}: {cos:.4f}", flush=True)
    if args.embeddings_only:
        return 0

    # Stage 2: does the talker stop on its own?
    results = []
    for name, wave in material.items():
        reference = AudioChunk(wave.astype(np.float32), RATE)
        for run in range(args.runs):
            torch.manual_seed(1000 + run)
            start = time.monotonic()
            audio = tts._generate(args.text, args.language, reference, "")
            elapsed = time.monotonic() - start
            seconds = len(audio) / tts.sample_rate
            # The talker decodes one codec frame per step; the codec renders
            # `frame_rate_hz` frames per second of audio, so the step count is
            # recoverable from the waveform without instrumenting the loop.
            steps = seconds * tts.geometry.frame_rate_hz
            # Against whichever bound actually applies: before the text-derived
            # budget existed that was the module ceiling, and after it the
            # budget is what a generation can hit.
            bound = min(args.max_new_tokens, tts.step_budget(args.text))
            capped = steps >= bound - 2
            if args.save_dir:
                out = Path(args.save_dir)
                out.mkdir(parents=True, exist_ok=True)
                sf.write(str(out / f"{name}-run{run}.wav"), audio, tts.sample_rate)
                # WHERE the extra audio sits decides the fix. A tail that is
                # silence would be trimmed; a tail that is speech-shaped is
                # babble, and only a shorter leash plus a retry helps.
                step = tts.sample_rate  # one second per bucket
                rms = [round(float(np.sqrt(np.mean(audio[i:i + step] ** 2))), 4)
                       for i in range(0, len(audio), step)]
                print(f"    per-second rms: {rms}", flush=True)
            results.append({"arm": name, "run": run, "audio_s": round(seconds, 2),
                            "steps": round(steps), "capped": bool(capped),
                            "wall_s": round(elapsed, 1)})
            print(f"{name:28s} run{run}  audio={seconds:6.2f}s  steps={steps:5.0f}"
                  f"  {'RAN TO CAP' if capped else 'stopped by itself'}"
                  f"  wall={elapsed:6.1f}s", flush=True)

    print()
    print(json.dumps(results))
    for name in material:
        rows = [r for r in results if r["arm"] == name]
        hit = sum(1 for r in rows if r["capped"])
        mean = sum(r["audio_s"] for r in rows) / len(rows)
        print(f"{name:28s} ran to cap {hit}/{len(rows)}, mean audio {mean:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
