# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Give every preset ONE identity across all its languages.

**The bug this fixes, measured.** `render_preset_voices.py` renders each
preset independently per language: same descriptor, same pinned seed, but a
different sentence, through a model that designs a voice from a
natural-language description. The description constrains the voice CLASS, not
the timbre, so the seed does not save it. Measured with
`wespeaker_en_voxceleb_resnet34_LM`, the German and Spanish renders of the
same preset scored **0.044 to 0.505** against each other -- against a
different-speaker range whose median is 0.627. Every preset was two different
people wearing one name.

The user-visible failure is exactly the one §4.3 exists to prevent: a speaker
the listener has learned to recognise silently changes voice the moment the
conversation switches language.

**The fix uses the pipeline's own validated path.** One language is the
ANCHOR, rendered by VoiceDesign. Every other language is then produced by
CLONING that anchor with the serving checkpoint -- the same cross-lingual
zero-shot path a real speaker's turn takes, which is measured at WER 0.100 in
the round-trip gate. The identity therefore comes from one clip, and the
per-language text still comes from that language's own render sentence, so
nothing regresses on the reason the clips were per-language in the first
place.

Accent: the derived clip carries the anchor language's accent. That is
explicitly fine per the 2026-08-03 dated decision -- accent carry-over is
wanted and never penalised -- and it is a small price for an identity that
holds still.

    CUDA_VISIBLE_DEVICES=99 PYTHONPATH=<repo>/python \\
      /spinning/htsglang-gpu/.venv/bin/python \\
      scripts/translator/derive_preset_languages.py \\
        --anchor de --languages es
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np

REPO_PYTHON = Path(__file__).resolve().parents[2] / "python"
if str(REPO_PYTHON) not in sys.path:
    sys.path.insert(0, str(REPO_PYTHON))

from sglang.srt.translator.voice_presets import RENDER_SENTENCES  # noqa: E402


async def run(args) -> int:
    import soundfile as sf

    from sglang.srt.translator.backends import AudioChunk
    from sglang.srt.translator.inprocess_tts import (
        InProcessQwen3Tts,
        InProcessTtsConfig,
    )

    anchors = sorted(args.pool_root.glob(f"*/*.{args.anchor}.wav"))
    if not anchors:
        print(f"[derive] no {args.anchor} anchors under {args.pool_root}")
        return 1
    languages = [c.strip() for c in args.languages.split(",") if c.strip()]
    missing = [c for c in languages if c not in RENDER_SENTENCES]
    if missing:
        print(f"[derive] no render sentence for {missing}")
        return 1
    print(f"[derive] {len(anchors)} anchors in {args.anchor} -> {languages}")

    tts = InProcessQwen3Tts(
        InProcessTtsConfig(
            model_dir=args.model_dir,
            device=args.device,
            dtype=args.dtype,
            # The anchor clips are ~7 s, comfortably over the floor, but the
            # check stays on: a truncated anchor would otherwise be cloned
            # from silently.
            min_reference_seconds=3.0,
        )
    )
    tts.load()

    written = 0
    for anchor in anchors:
        samples, rate = sf.read(str(anchor), dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        reference = AudioChunk(np.ascontiguousarray(samples), rate)
        voice_id = anchor.name.split(".")[0]

        for language in languages:
            target = anchor.with_name(f"{voice_id}.{language}.wav")
            if target.exists() and not args.overwrite:
                print(f"[derive]   {target.name} exists, skipping")
                continue
            text = RENDER_SENTENCES[language]
            chunks = []
            async for chunk in tts.synthesize(
                text=text, language=language, reference=reference
            ):
                chunks.append(chunk.samples)
            if not chunks:
                print(f"[derive]   {target.name}: no audio, refusing to write")
                return 1
            waveform = np.concatenate(chunks).astype(np.float32)
            if not np.isfinite(waveform).all() or float(np.abs(waveform).max()) < 1e-3:
                print(f"[derive]   {target.name}: silence or non-finite, refusing")
                return 1
            sf.write(str(target), waveform, tts.sample_rate, subtype="PCM_16")
            target.with_suffix(".txt").write_text(text, encoding="utf-8")
            written += 1
            print(f"[derive]   wrote {target.name} "
                  f"({len(waveform) / tts.sample_rate:.1f}s, cloned from "
                  f"{anchor.name})", flush=True)

    print(f"[derive] done: {written} clips derived from {len(anchors)} anchors")
    print("[derive] now re-run check_preset_pool.py: cross-language similarity "
          "must clear the floor, and within-class distinctness must not have "
          "regressed -- cloning one anchor into two languages could in "
          "principle pull two presets together.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pool-root", type=Path,
        default=Path("/spinning/llm_stuff/translator-models/preset-voices"),
    )
    parser.add_argument("--anchor", default="de")
    parser.add_argument("--languages", default="es")
    parser.add_argument(
        "--model-dir", type=Path,
        default=Path("/spinning/llm_stuff/translator-models/qwen3-tts-0.6b-base"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--overwrite", action="store_true", default=True)
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
