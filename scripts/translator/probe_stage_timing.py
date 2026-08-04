# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Which stage of one synthesis actually costs the time.

Written because the first warm GPU measurement came back at RTF 28.9 on a
5090 for a 0.6 B talker -- roughly two orders of magnitude away from anything
the architecture predicts, which means the number is a symptom, not a limit.
``latency_window.py`` measures the stage as a black box; this opens it, since
a chain number cannot say WHICH link is wrong.

The three links of one ``generate_voice_clone`` call are timed separately,
with a CUDA synchronize around each so the numbers are wall time on the device
rather than queue-submission time:

1. ``create_voice_clone_prompt`` -- codec ENCODE of the reference plus the
   speaker encoder;
2. ``model.generate`` -- the talker's autoregressive loop, the part a serving
   engine would own;
3. ``speech_tokenizer.decode`` -- the codec DECODER turning codes into a
   waveform.

It also prints how many codec frames came out, because time per frame is the
only figure comparable across utterance lengths, and a frame count wildly
above the audio duration would mean a runaway rather than a slow loop.

    CUDA_VISIBLE_DEVICES=<uuid> PYTHONPATH=<repo>/python \\
      /spinning/htsglang-gpu/.venv/bin/python \\
      scripts/translator/probe_stage_timing.py --repeats 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

MODEL_DIR = Path("/spinning/llm_stuff/translator-models/qwen3-tts-0.6b-base")
REFERENCE = Path("/spinning/llm_stuff/translator-models/xtts-v2/samples/de_sample.wav")
TEXT = "Hola, buenos dias. Me alegro mucho de verte otra vez."


class _Clock:
    """Wall time with the device drained, so a stage cannot hide in the queue."""

    def __init__(self, torch, device: str) -> None:
        self._torch = torch
        self._device = device
        self.spans: dict = {}

    def sync(self) -> None:
        if self._device.startswith("cuda"):
            self._torch.cuda.synchronize()

    def measure(self, label: str, fn):
        self.sync()
        started = time.monotonic()
        result = fn()
        self.sync()
        self.spans[label] = time.monotonic() - started
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--text", default=TEXT)
    parser.add_argument("--language", default="spanish")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()

    import soundfile as sf
    import torch

    from sglang.srt.translator.inprocess_tts import (
        InProcessQwen3Tts,
        InProcessTtsConfig,
    )

    print(f"[stage] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[stage] no CUDA device visible; refusing to report a GPU number")
            return 1
        print(f"[stage] device: {torch.cuda.get_device_name(0)}")

    samples, rate = sf.read(str(args.reference), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = np.ascontiguousarray(samples)

    backend = InProcessQwen3Tts(
        InProcessTtsConfig(
            model_dir=args.model_dir, device=args.device, dtype=args.dtype
        )
    )
    started = time.monotonic()
    backend.load()
    print(f"[stage] loaded in {time.monotonic() - started:.1f}s (cold)")

    wrapper = backend._model  # noqa: SLF001 - a probe reads private state on purpose
    inner = getattr(wrapper, "model", wrapper)
    clock = _Clock(torch, args.device)

    for run in range(args.repeats):
        label = "warm-up" if run == 0 else f"run {run}"
        with torch.inference_mode():
            items = clock.measure(
                "prompt", lambda: wrapper.create_voice_clone_prompt(
                    ref_audio=[(samples, rate)], ref_text=[""], x_vector_only_mode=True
                )
            )
            prompt = wrapper._prompt_items_to_voice_clone_prompt(items)  # noqa: SLF001
            input_ids = wrapper._tokenize_texts(  # noqa: SLF001
                [wrapper._build_assistant_text(args.text)]  # noqa: SLF001
            )
            gen_kwargs = wrapper._merge_generate_kwargs(  # noqa: SLF001
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=0.9,
            )
            codes_list = clock.measure(
                "talker",
                lambda: inner.generate(
                    input_ids=input_ids,
                    ref_ids=None,
                    voice_clone_prompt=prompt,
                    languages=[args.language],
                    non_streaming_mode=True,
                    **gen_kwargs,
                )[0],
            )
            frames = int(codes_list[0].shape[0])
            wavs = clock.measure(
                "codec-decode",
                lambda: inner.speech_tokenizer.decode(
                    [{"audio_codes": c} for c in codes_list]
                )[0],
            )

        audio_s = len(np.asarray(wavs[0]).reshape(-1)) / backend.sample_rate
        total = sum(clock.spans.values())
        print(
            f"[stage] {label}: {frames} codec frames -> {audio_s:.2f}s audio  "
            f"total {total:.2f}s  RTF {total / max(audio_s, 1e-9):.2f}"
        )
        for name in ("prompt", "talker", "codec-decode"):
            span = clock.spans[name]
            print(
                f"[stage]     {name:14s} {span:8.3f}s  "
                f"({100 * span / total:5.1f}% of the call)"
            )
        if frames:
            print(
                f"[stage]     talker per frame {clock.spans['talker'] / frames * 1000:.1f} ms  "
                f"codec per frame {clock.spans['codec-decode'] / frames * 1000:.1f} ms"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
