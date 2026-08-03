# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""What the intra-segment speaker-change test actually sees, on real audio.

Written after a front-door run split ONE German speaker into ``speaker-1`` and
``speaker-2``. The re-cut (``session._split_at_speaker_changes``) embeds
``speaker_change_window_s`` windows across a segment and cuts wherever an
adjacent pair falls below ``speaker_change_threshold`` (0.62). This prints
those adjacent similarities so the threshold can be read against the numbers
it actually decides on, instead of against intuition.

It also prints a BETWEEN-speaker pair, because a within-speaker number means
nothing on its own: if the embedder cannot separate two different voices at
this window length, the instrument is what is broken and the threshold is
irrelevant. That check runs first and the verdict is withheld if it fails.

CPU only, no GPU window.

    PYTHONPATH=<repo>/python /spinning/htsglang-gpu/.venv/bin/python \\
      scripts/translator/probe_speaker_change.py \\
      --audio /spinning/llm_stuff/translator-models/xtts-v2/samples/de_sample.wav \\
      --other <a-clip-of-a-DIFFERENT-voice.wav>
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np

DEFAULT_AUDIO = Path(
    "/spinning/llm_stuff/translator-models/xtts-v2/samples/de_sample.wav"
)
DEFAULT_EMBEDDER = Path(
    "/spinning/llm_stuff/translator-models/embedder/wespeaker_resnet34_LM.onnx"
)


def load(path: Path, rate: int):
    import soundfile as sf

    from sglang.srt.translator.audio import resample
    from sglang.srt.translator.backends import AudioChunk

    samples, sr = sf.read(str(path), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    chunk = AudioChunk(np.ascontiguousarray(samples), sr)
    return resample(chunk, rate) if sr != rate else chunk


async def windows_of(embedder, chunk, window_s: float):
    from sglang.srt.translator.backends import AudioChunk

    rate = chunk.sample_rate
    width = int(window_s * rate)
    count = int(len(chunk.samples) / rate // window_s)
    out = []
    for index in range(count):
        piece = chunk.samples[index * width : index * width + width]
        out.append(await embedder.embed(AudioChunk(piece, rate)))
    return out


async def run(args) -> int:
    from sglang.srt.translator.speakers import split_points_by_dispersion
    from sglang.srt.translator.asr_backends import OnnxSpeakerEmbedder

    embedder = OnnxSpeakerEmbedder(args.embedder_model)
    rate = 16000

    one = load(args.audio, rate)
    windows = await windows_of(embedder, one, args.window_s)
    print(f"[chg] {args.audio.name}: {one.duration_s:.2f}s -> "
          f"{len(windows)} windows of {args.window_s}s")
    if len(windows) < 2:
        print("[chg] fewer than two windows; the re-cut would not run at all")
        return 1

    # -- instrument first: can it tell two DIFFERENT voices apart here? -----
    between = None
    if args.other is not None and args.other.exists():
        other = load(args.other, rate)
        other_windows = await windows_of(embedder, other, args.window_s)
        if other_windows:
            between = max(
                windows[i].similarity(other_windows[j])
                for i in range(len(windows))
                for j in range(len(other_windows))
            )
            print(f"[chg] BETWEEN speakers ({args.other.name}), worst case "
                  f"(most similar pair): {between:.3f}")

    print("[chg] WITHIN one speaker, adjacent window pairs:")
    within = []
    for i in range(1, len(windows)):
        sim = windows[i - 1].similarity(windows[i])
        within.append(sim)
        verdict = "CUT" if sim < args.threshold else "keep"
        print(f"[chg]   window {i-1} vs {i}: {sim:.3f}  -> {verdict} "
              f"(threshold {args.threshold})")

    points = split_points_by_dispersion(windows, args.threshold)
    print(f"[chg] split_points_by_dispersion -> {points or '() (one speaker)'}")

    print("")
    if between is None:
        print("[chg] no --other given: the instrument's discrimination is "
              "UNPROVEN here, so no verdict on the threshold.")
        return 0
    worst_within = min(within)
    print(f"[chg] within-speaker minimum {worst_within:.3f}, "
          f"between-speaker maximum {between:.3f}")
    if between >= worst_within:
        print("[chg] THE INSTRUMENT CANNOT DISCRIMINATE at this window "
              "length: two different voices score at least as similar as one "
              "voice does with itself. No threshold can separate them; fix "
              "the window or the embedder, not the number.")
        return 1
    print(f"[chg] a threshold separates the two populations anywhere in "
          f"({between:.3f}, {worst_within:.3f}]. The shipped "
          f"{args.threshold} is "
          f"{'ABOVE the within-speaker floor -- it cuts one speaker' if args.threshold > worst_within else 'inside the separating band -- it does not cut one speaker'}.")
    return 0 if args.threshold <= worst_within else 2


async def sweep(args) -> int:
    """The same two populations over the whole preset pool, not one pair.

    A threshold picked from a single clip pair is exactly the desk-picked
    default this project has been bitten by. This walks every voice in the
    pool: adjacent windows of the SAME clip are the within-speaker
    population, windows of DIFFERENT voices are the between-speaker one, and
    the gap between the two distributions is what a threshold may live in.
    """
    from sglang.srt.translator.asr_backends import OnnxSpeakerEmbedder

    embedder = OnnxSpeakerEmbedder(args.embedder_model)
    rate = 16000
    clips = sorted(args.pool.rglob("*.wav"))
    if len(clips) < 2:
        print(f"[chg] need at least two clips under {args.pool}")
        return 1

    within: list = []
    per_voice: dict = {}
    for clip in clips:
        chunk = load(clip, rate)
        windows = await windows_of(embedder, chunk, args.window_s)
        if len(windows) >= 1:
            per_voice[clip.name] = windows
        for i in range(1, len(windows)):
            within.append((windows[i - 1].similarity(windows[i]), clip.name))

    names = sorted(per_voice)
    between: list = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if a.split(".")[0] == b.split(".")[0]:
                continue  # same voice, different language: not a between pair
            best = max(
                x.similarity(y) for x in per_voice[a] for y in per_voice[b]
            )
            between.append((best, f"{a} vs {b}"))

    if not within or not between:
        print("[chg] not enough windows to build both populations")
        return 1

    within_vals = sorted(v for v, _ in within)
    between_vals = sorted(v for v, _ in between)
    worst_within = min(within)
    best_between = max(between)
    print(f"[chg] pool sweep over {len(clips)} clips, {len(names)} with windows")
    print(f"[chg] WITHIN  n={len(within_vals)}  min {within_vals[0]:.3f}  "
          f"p05 {within_vals[len(within_vals) // 20]:.3f}  "
          f"median {within_vals[len(within_vals) // 2]:.3f}")
    print(f"[chg]   worst: {worst_within[1]} at {worst_within[0]:.3f}")
    print(f"[chg] BETWEEN n={len(between_vals)}  max {between_vals[-1]:.3f}  "
          f"p95 {between_vals[int(len(between_vals) * 0.95)]:.3f}  "
          f"median {between_vals[len(between_vals) // 2]:.3f}")
    print(f"[chg]   closest: {best_between[1]} at {best_between[0]:.3f}")
    print("")
    if best_between[0] >= worst_within[0]:
        print(f"[chg] the populations OVERLAP "
              f"({best_between[0]:.3f} >= {worst_within[0]:.3f}): no single "
              f"threshold is safe. The re-cut must not be the only guard.")
    else:
        print(f"[chg] separating band: ({best_between[0]:.3f}, "
              f"{worst_within[0]:.3f}]  -- midpoint "
              f"{(best_between[0] + worst_within[0]) / 2:.3f}")
    print(f"[chg] shipped threshold {args.threshold}: cuts "
          f"{sum(1 for v in within_vals if v < args.threshold)} of "
          f"{len(within_vals)} SAME-speaker pairs "
          f"(every one of those is a person split in two)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--other", type=Path, default=None,
                        help="a clip of a DIFFERENT voice; without it there "
                             "is no discrimination proof and no verdict")
    parser.add_argument("--embedder-model", type=Path, default=DEFAULT_EMBEDDER)
    parser.add_argument("--window-s", type=float, default=1.5)
    parser.add_argument("--threshold", type=float, default=0.62)
    parser.add_argument(
        "--pool", type=Path, default=None,
        help="sweep this voice pool instead of one pair, to get the two "
             "populations from many voices rather than from one sample",
    )
    args = parser.parse_args()
    return asyncio.run(sweep(args) if args.pool else run(args))


if __name__ == "__main__":
    sys.exit(main())
