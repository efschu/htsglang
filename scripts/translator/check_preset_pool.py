# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Is the rendered preset pool usable? Class shape, and voices that are DISTINCT.

A pool that loads is not a pool that works. The property the design actually
depends on (DESIGN_466 §4.3) is that two speakers assigned different presets
sound different to a listener -- "distinctness beats class match", because a
listener who cannot tell two speakers apart has lost more than one whose
preset is the wrong gender. Nothing about rendering guarantees that: the
VoiceDesign model is given eighteen natural-language descriptions and may well
collapse several of them onto one voice.

So this measures it, with the same instrument the pipeline itself uses to tell
speakers apart: the checkpoint's speaker encoder. For every pair of presets it
reports the cosine similarity of their x-vectors, and it flags any pair above
the registry's own speaker-merge threshold -- because a pair that similar is,
by the system's own definition, the SAME speaker, and handing them to two
participants would attribute words to the wrong person.

It also checks the axis that is easy to forget: the same preset in two
languages must still be the same VOICE. If ``man-01.de`` and ``man-01.es``
score low against each other, the descriptor is not producing a stable
identity and the preset will change voice when the conversation switches
language.

    CUDA_VISIBLE_DEVICES=99 PYTHONPATH=<repo>/python \\
      /spinning/htsglang-gpu/.venv/bin/python \\
      scripts/translator/check_preset_pool.py
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from sglang.srt.translator.speakers import SpeakerRegistryConfig

DEFAULT_POOL = Path("/spinning/llm_stuff/translator-models/preset-voices")
DEFAULT_MODEL = Path("/spinning/llm_stuff/translator-models/qwen3-tts-0.6b-base")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-root", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    #: Taken from the speaker registry rather than invented here, and it must
    #: STAY taken from there: `match_threshold` is the cosine above which the
    #: live pipeline calls two clips the same person. Two presets at or above
    #: it would be merged into one speaker identity at runtime, which is the
    #: precise operational meaning of "not distinct". A number chosen in this
    #: script could silently drift away from the one that actually decides.
    parser.add_argument(
        "--merge-threshold", type=float,
        default=SpeakerRegistryConfig.match_threshold,
    )
    #: Below this, a preset is not a stable identity across languages.
    parser.add_argument("--cross-language-floor", type=float, default=0.60)
    args = parser.parse_args()

    import soundfile as sf
    import torch

    from sglang.srt.translator.qwen3_tts_compat import (
        ensure_qwen3_tts_importable,
        librosa_resample,
        refresh_rotary_buffers,
        verify_and_load_weights,
    )

    clips = sorted(args.pool_root.glob("*/*.wav"))
    if not clips:
        print(f"[pool] no clips under {args.pool_root}")
        return 1

    by_class: dict = defaultdict(list)
    for clip in clips:
        by_class[clip.parent.name].append(clip)
    print(f"[pool] {len(clips)} clips in {len(by_class)} classes")
    for name in sorted(by_class):
        voices = sorted({c.name.split(".")[0] for c in by_class[name]})
        print(f"[pool]   {name:6s} {len(by_class[name]):3d} clips, "
              f"{len(voices)} voices: {', '.join(voices)}")

    ensure_qwen3_tts_importable()
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(str(args.model_dir), dtype=torch.float32)
    inner = getattr(model, "model", model)
    inner.to("cpu")
    refresh_rotary_buffers(inner)
    # Same reason as everywhere else: an unloaded speaker encoder produces
    # x-vectors that agree beautifully with each other and mean nothing.
    verify_and_load_weights(inner, args.model_dir)
    rate = inner.speaker_encoder_sample_rate

    vectors = {}
    for clip in clips:
        samples, source_rate = sf.read(str(clip), dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        resampled = librosa_resample(samples, source_rate, rate)
        with torch.inference_mode():
            vector = inner.extract_speaker_embedding(audio=resampled, sr=rate)
        vector = vector.detach().float().reshape(-1)
        vectors[clip] = vector / vector.norm()

    def cosine(a: Path, b: Path) -> float:
        return float(torch.dot(vectors[a], vectors[b]))

    # -- distinctness within each class -----------------------------------
    print("\n[pool] pairwise similarity WITHIN each class "
          f"(flagging >= {args.merge_threshold:.2f}, the registry's "
          "same-speaker line)")
    failures = 0
    for name in sorted(by_class):
        # One language at a time: comparing across languages here would mix
        # the two questions this script is deliberately keeping apart.
        for language in sorted({c.name.split(".")[1] for c in by_class[name]}):
            same = [c for c in by_class[name] if c.name.split(".")[1] == language]
            if len(same) < 2:
                continue
            scores = [
                (cosine(a, b), a.name, b.name)
                for a, b in itertools.combinations(same, 2)
            ]
            scores.sort(reverse=True)
            worst, left, right = scores[0]
            median = float(np.median([s[0] for s in scores]))
            flag = " <-- TOO SIMILAR" if worst >= args.merge_threshold else ""
            print(f"[pool]   {name:6s} {language}: {len(scores):3d} pairs, "
                  f"median {median:.3f}, closest {worst:.3f} "
                  f"({left} / {right}){flag}")
            if worst >= args.merge_threshold:
                failures += 1

    # -- identity stability across languages ------------------------------
    print(f"\n[pool] same preset across languages "
          f"(floor {args.cross_language_floor:.2f})")
    by_voice: dict = defaultdict(list)
    for clip in clips:
        by_voice[clip.name.split(".")[0]].append(clip)
    weak = []
    for voice in sorted(by_voice):
        renders = sorted(by_voice[voice])
        if len(renders) < 2:
            continue
        scores = [cosine(a, b) for a, b in itertools.combinations(renders, 2)]
        low = min(scores)
        if low < args.cross_language_floor:
            weak.append((voice, low))
    if weak:
        for voice, low in weak:
            print(f"[pool]   {voice}: {low:.3f} -- this preset changes voice "
                  "between languages")
        failures += len(weak)
    else:
        cross = [
            min(cosine(a, b) for a, b in itertools.combinations(sorted(v), 2))
            for v in by_voice.values()
            if len(v) > 1
        ]
        if cross:
            print(f"[pool]   all {len(cross)} presets stable; "
                  f"weakest {min(cross):.3f}, median {float(np.median(cross)):.3f}")

    print("")
    if failures:
        print(f"[pool] {failures} problem(s): the pool is NOT ready to hand out")
        return 1
    print("[pool] pool is usable: every class distinct, every preset stable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
