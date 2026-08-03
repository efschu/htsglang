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

So this measures it -- with the instrument the pipeline itself uses, and only
after PROVING that instrument can tell anyone apart at all.

**That validation is not ceremony; it caught a wrong verdict here.** The first
version of this script scored the pool on the TTS checkpoint's own speaker
encoder, which is loaded anyway and conditions the cloning. Every preset pair
came back at 0.94-0.99 and the pool was declared collapsed. It was not: that
encoder returns x-vectors whose shared component is ~98 % of the norm (mean
vector 9.83 against a median sample norm of 10.06), so cosine is ~0.98 between
ANY two clips. Measured on eight unrelated speakers it spans 0.949-0.990, and a
voice against its own clone scores 0.981 -- INSIDE that range. It is trained to
condition synthesis, not to discriminate identity, and mean-centering does not
rescue it either. A verdict from an instrument with 0.04 of dynamic range is
noise wearing a number.

So the encoder is `wespeaker_en_voxceleb_resnet34_LM` through ONNX Runtime --
the design's original §2.2 choice, which this detour vindicated -- and the
script refuses to report anything until the instrument clears a floor check. For every pair of presets it
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
import asyncio
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from sglang.srt.translator.speakers import SpeakerRegistryConfig

DEFAULT_POOL = Path("/spinning/llm_stuff/translator-models/preset-voices")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-root", type=Path, default=DEFAULT_POOL)
    parser.add_argument(
        "--embedder-model", type=Path,
        default=Path("/spinning/llm_stuff/translator-models/embedder/"
                     "wespeaker_resnet34_LM.onnx"),
    )
    parser.add_argument(
        "--control-dir", type=Path,
        default=Path("/spinning/llm_stuff/translator-models/xtts-v2/samples"),
        help="clips that are NOT the same voice; used to prove the encoder "
             "discriminates before the pool is judged",
    )
    parser.add_argument(
        "--min-instrument-range", type=float, default=0.30,
        help="required cosine spread over the control clips",
    )
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

    from sglang.srt.translator.asr_backends import OnnxSpeakerEmbedder
    from sglang.srt.translator.backends import AudioChunk

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

    embedder = OnnxSpeakerEmbedder(args.embedder_model)
    print(f"[pool] embedder {embedder.name}")

    def load(path: Path) -> AudioChunk:
        samples, source_rate = sf.read(str(path), dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        return AudioChunk(np.ascontiguousarray(samples), source_rate)

    async def embed_all(paths):
        out = {}
        for path in paths:
            out[path] = (await embedder.embed(load(path))).vector
        return out

    controls = sorted(args.control_dir.glob("*.wav")) if args.control_dir else []
    vectors = asyncio.run(embed_all(list(clips) + controls))

    def cosine(a: Path, b: Path) -> float:
        return float(np.dot(vectors[a], vectors[b]))

    # -- INSTRUMENT FLOOR, before any verdict -----------------------------
    # An encoder that scores everything alike would declare any pool
    # collapsed; one that scores everything different would pass any pool.
    # Neither failure is visible in the pool numbers themselves.
    if len(controls) >= 3:
        spread = [cosine(a, b) for a, b in itertools.combinations(controls, 2)]
        observed = max(spread) - min(spread)
        print(f"[pool] instrument floor over {len(controls)} unrelated clips: "
              f"cosine {min(spread):.3f}..{max(spread):.3f} "
              f"(range {observed:.3f})")
        if observed < args.min_instrument_range:
            print(f"[pool] REFUSING to judge the pool: this encoder only "
                  f"spans {observed:.3f} across clips that are not the same "
                  f"voice. Any verdict would be noise wearing a number.")
            return 1
    else:
        print("[pool] no control clips given (--control-dir); the instrument "
              "is UNVALIDATED and the verdict below is provisional")

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
