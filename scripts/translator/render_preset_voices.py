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

import numpy as np

REPO_PYTHON = Path(__file__).resolve().parents[2] / "python"
if str(REPO_PYTHON) not in sys.path:
    sys.path.insert(0, str(REPO_PYTHON))

#: The checkpoint's language table is keyed by ENGLISH NAME, not ISO code;
#: the rest of the system speaks ISO codes, so the translation happens here
#: and nowhere else -- the same confinement `inprocess_tts` uses.
LANGUAGE_NAMES = {
    "de": "german", "es": "spanish", "en": "english", "fr": "french",
    "it": "italian", "pt": "portuguese", "ru": "russian", "ja": "japanese",
    "ko": "korean", "zh": "chinese",
}

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
        "--model-dir", type=Path,
        default=Path("/spinning/llm_stuff/translator-models/qwen3-tts-1.7b-voicedesign"),
        help="local checkpoint; the pool is rendered in process, not over HTTP",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--limit", type=int, default=0,
        help="render at most N clips; render ONE and gate it before the batch",
    )
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

    from sglang.srt.translator.qwen3_tts_compat import (  # noqa: E402
        ensure_qwen3_tts_importable,
        refresh_rotary_buffers,
        restore_cache_position,
        verify_and_load_weights,
    )

    # The VoiceDesign checkpoint is the same architecture as the serving one
    # and therefore carries the same two silent transformers-5.x faults: the
    # missing `cache_position` that pins every decode step to the prefill
    # branch, and `from_pretrained` reporting a successful load while loading
    # nothing. Both produce fluent babble rather than an error, so a batch
    # rendered without these would yield 36 plausible-sounding, useless clips
    # -- and the pool is the thing every other path degrades TO.
    shims = ensure_qwen3_tts_importable()
    print(f"compat shims: {', '.join(shims)}")

    from qwen_tts.core.models.modeling_qwen3_tts import (  # noqa: E402
        Qwen3TTSTalkerForConditionalGeneration,
    )
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel  # noqa: E402

    restore_cache_position(Qwen3TTSTalkerForConditionalGeneration)

    dtype = getattr(torch, args.dtype)
    load_kwargs = {"dtype": dtype}
    if args.device.startswith("cuda"):
        load_kwargs["device_map"] = args.device
    model = Qwen3TTSModel.from_pretrained(str(args.model_dir), **load_kwargs)
    inner = getattr(model, "model", model)
    if not args.device.startswith("cuda"):
        inner.to(args.device)
    refresh_rotary_buffers(inner)
    report = verify_and_load_weights(inner, args.model_dir)
    print(f"weights: {report['checked']} checked, {report['repaired']} repaired")

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

        with torch.inference_mode():
            waveforms, sample_rate = model.generate_voice_design(
                text=[entry["text"]],
                instruct=[entry["description"]],
                language=[LANGUAGE_NAMES.get(entry["language"], entry["language"])],
                non_streaming_mode=True,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=0.9,
            )
        waveform = np.concatenate(
            [np.asarray(w, dtype=np.float32).reshape(-1) for w in waveforms]
        )
        seconds = len(waveform) / sample_rate
        if not np.isfinite(waveform).all() or float(np.abs(waveform).max()) < 1e-3:
            raise RuntimeError(
                f"{path.name} is silence or non-finite; refusing to write a "
                "clip the pool would later hand out as a voice"
            )
        soundfile.write(str(path), waveform, sample_rate, subtype="PCM_16")

        transcript = path.with_suffix(".txt")
        transcript.write_text(entry["text"], encoding="utf-8")
        written += 1
        print(f"  wrote {path.name} ({seconds:.1f}s)", flush=True)
        if args.limit and written >= args.limit:
            print(f"stopping after {written} clips (--limit)")
            break

    print(f"done: {written} written, {skipped} already present")
    print("verify the pool loads and is not thin:")
    print("  python -c \"from sglang.srt.translator.voices import VoicePool; "
          f"p=VoicePool.from_directory('{args.pool_root}'); "
          "print(len(p), p.counts_by_class(), p.thin_classes())\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
