#!/usr/bin/env python3
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Render the 18 preset voices into pool clips. GPU-window step, run once.

    python scripts/translator/render_preset_voices.py \
        --pool-root /spinning/llm_stuff/translator-models/preset-voices \
        --languages de,es

Uses Qwen3-TTS-VoiceDesign (Apache-2.0), which synthesizes a voice from a
natural-language description -- the serving checkpoint is a cloning model and
cannot invent a voice, so the pool has to be rendered by something that can.
Output is ordinary reference audio; nothing downstream knows it was synthetic.

Seeds are pinned per descriptor so a re-render after a lost pool reproduces the
SAME voices. Without that, a speaker the user has learned to recognise would
silently change identity between sessions.

`--dry-run` prints the plan and touches no GPU, which is how the desk suite
exercises this file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_PYTHON = Path(__file__).resolve().parents[2] / "python"
if str(REPO_PYTHON) not in sys.path:
    sys.path.insert(0, str(REPO_PYTHON))

from sglang.srt.translator.voice_presets import (  # noqa: E402
    PRESET_DESCRIPTORS,
    VOICE_DESIGN_MODEL,
    VOICE_DESIGN_REVISION,
    render_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool-root",
        type=Path,
        default=Path("/spinning/llm_stuff/translator-models/preset-voices"),
    )
    parser.add_argument(
        "--languages", default="de,es",
        help="comma-separated; one clip per preset per language",
    )
    parser.add_argument("--model", default=VOICE_DESIGN_MODEL)
    parser.add_argument("--revision", default=VOICE_DESIGN_REVISION)
    parser.add_argument("--card-uuid", default=None,
                        help="NVML UUID; pins the process before any CUDA init")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="re-render clips that already exist",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    languages = [c.strip() for c in args.languages.split(",") if c.strip()]
    plan = render_plan(languages, str(args.pool_root))

    print(f"{len(PRESET_DESCRIPTORS)} presets x {len(languages)} languages "
          f"= {len(plan)} clips")
    print(f"model {args.model} @ {args.revision}")
    print(f"pool  {args.pool_root}")

    if args.dry_run:
        for entry in plan:
            print(json.dumps(entry, ensure_ascii=False))
        # Report the shape the pool loader will see, so a sizing mistake shows
        # up here rather than at the third speaker in a Spanish bar.
        counts = {}
        for entry in plan:
            counts[entry["voice_class"]] = counts.get(entry["voice_class"], 0) + 1
        print(f"clips per class: {counts}")
        return 0

    if args.card_uuid:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.card_uuid)

    import soundfile
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()

    written = skipped = 0
    for entry in plan:
        path = Path(entry["path"])
        if path.exists() and not args.overwrite:
            skipped += 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)

        # Pin per clip, not once globally: a resumed run must reproduce the
        # same voice for the clips it re-renders, regardless of what was
        # rendered before it in this process.
        torch.manual_seed(entry["seed"])

        inputs = processor(
            text=entry["text"],
            voice_description=entry["description"],
            language=entry["language"],
            return_tensors="pt",
        ).to("cuda:0")
        with torch.inference_mode():
            audio = model.generate(**inputs, do_sample=True, temperature=0.9)
        waveform = audio.detach().float().cpu().numpy().reshape(-1)
        soundfile.write(str(path), waveform, args.sample_rate, subtype="PCM_16")

        transcript = path.with_suffix(".txt")
        transcript.write_text(entry["text"], encoding="utf-8")
        written += 1
        print(f"  wrote {path.name} ({len(waveform) / args.sample_rate:.1f}s)")

    print(f"done: {written} written, {skipped} already present")
    print("verify the pool loads and is not thin:")
    print("  python -c \"from sglang.srt.translator.voices import VoicePool; "
          f"p=VoicePool.from_directory('{args.pool_root}'); "
          "print(len(p), p.counts_by_class(), p.thin_classes())\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
