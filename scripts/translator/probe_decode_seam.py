# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Instrument the talker's decode seam and print what actually arrives there.

Run it BEFORE and AFTER the ``cache_position`` shim: unfixed it prints the
prefill-shaped position tensor arriving on a one-token decode step and then the
``o_proj`` shape error; fixed it prints one position per query token and the
generation runs to completion.

    CUDA_VISIBLE_DEVICES=99 PYTHONPATH=<repo>/python \\
      /spinning/htsglang-gpu/.venv/bin/python \\
      scripts/translator/probe_decode_seam.py [--steps 6] [--no-shim]

CPU only, ~1.5 GB of fp32 weights, no GPU window needed. The point is the seam,
not the audio.
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import numpy as np

MODEL_DIR = Path("/spinning/llm_stuff/translator-models/qwen3-tts-0.6b-base")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument(
        "--no-shim",
        action="store_true",
        help="skip the cache_position restoration, to reproduce the failure",
    )
    args = parser.parse_args()

    import torch

    from sglang.srt.translator.qwen3_tts_compat import (
        ensure_qwen3_tts_importable,
        refresh_rotary_buffers,
        restore_cache_position,
    )

    shims = ensure_qwen3_tts_importable()
    print(f"[probe] shims: {', '.join(shims)}")

    from qwen_tts.core.models import modeling_qwen3_tts as mod
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    talker_cls = mod.Qwen3TTSTalkerForConditionalGeneration
    if args.no_shim:
        print("[probe] cache_position restoration DISABLED (reproducing the bug)")
    else:
        restored = restore_cache_position(talker_cls)
        print(f"[probe] cache_position restoration installed: {restored}")

    # Instrument the seam: what the talker forward receives, and which position
    # branch it takes.
    original_forward = talker_cls.forward
    calls = {"n": 0}

    @functools.wraps(original_forward)
    def traced_forward(self, *fargs, **fkwargs):
        calls["n"] += 1
        index = calls["n"]
        input_ids = fkwargs.get("input_ids")
        inputs_embeds = fkwargs.get("inputs_embeds")
        cache_position = fkwargs.get("cache_position")
        attention_mask = fkwargs.get("attention_mask")
        past = fkwargs.get("past_key_values")
        cached = past.get_seq_length() if past is not None else 0
        query_len = (
            inputs_embeds.shape[1]
            if inputs_embeds is not None
            else (input_ids.shape[1] if input_ids is not None else None)
        )
        branch = (
            "full-sequence get_rope_index"
            if (cache_position is None or cache_position[0] == 0 or self.rope_deltas is None)
            else "per-step arange"
        )
        print(
            f"[probe] call {index}: query_len={query_len} cached={cached} "
            f"input_ids={None if input_ids is None else tuple(input_ids.shape)} "
            f"embeds={None if inputs_embeds is None else tuple(inputs_embeds.shape)} "
            f"attn_mask={None if attention_mask is None else tuple(attention_mask.shape)} "
            f"cache_position={None if cache_position is None else tuple(cache_position.shape)} "
            f"-> {branch}"
        )
        out = original_forward(self, *fargs, **fkwargs)
        return out

    talker_cls.forward = traced_forward

    # And one step deeper: the positions that reach the rotary.
    talker_model_cls = mod.Qwen3TTSTalkerModel
    original_model_forward = talker_model_cls.forward

    @functools.wraps(original_model_forward)
    def traced_model_forward(self, *fargs, **fkwargs):
        position_ids = fkwargs.get("position_ids")
        inputs_embeds = fkwargs.get("inputs_embeds")
        query_len = None if inputs_embeds is None else inputs_embeds.shape[1]
        print(
            f"[probe]   -> talker model: query_len={query_len} "
            f"position_ids="
            f"{None if position_ids is None else tuple(position_ids.shape)}"
        )
        if position_ids is not None and query_len is not None:
            if position_ids.shape[-1] != query_len:
                print(
                    f"[probe]   !! CONTRACT VIOLATION: {position_ids.shape[-1]} "
                    f"positions for a {query_len}-token query"
                )
        return original_model_forward(self, *fargs, **fkwargs)

    talker_model_cls.forward = traced_model_forward

    print(f"[probe] loading {args.model_dir} on CPU (fp32) ...")
    model = Qwen3TTSModel.from_pretrained(str(args.model_dir), dtype=torch.float32)
    inner = getattr(model, "model", model)
    inner.to("cpu")
    print(f"[probe] refreshed {refresh_rotary_buffers(inner)} rotary buffers")

    # A 4 s reference tone is enough: the seam does not care what it sounds
    # like, and a synthetic reference keeps the probe hermetic.
    rate = 16000
    t = np.arange(4 * rate, dtype=np.float32) / rate
    reference = 0.1 * np.sin(2 * np.pi * 140.0 * t).astype(np.float32)

    try:
        with torch.inference_mode():
            model.generate_voice_clone(
                text=["Hola, buenos dias."],
                language=["spanish"],
                ref_audio=[(reference, rate)],
                ref_text=[""],
                x_vector_only_mode=True,
                non_streaming_mode=True,
                max_new_tokens=args.steps,
                do_sample=False,
            )
    except Exception as exc:  # noqa: BLE001 - the probe reports, it does not handle
        print(f"[probe] FAILED after {calls['n']} talker calls: {type(exc).__name__}: {exc}")
        return 1

    print(f"[probe] OK: {calls['n']} talker calls, no shape violation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
