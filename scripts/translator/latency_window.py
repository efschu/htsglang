# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The GPU latency anchor for the pipeline (DESIGN_466 §3), time-boxed.

Window 3 measured **71 ms warm first-audio for the TTS stage alone**, through
a serving engine that no longer exists. This measures the stages we actually
run, in process, on a card resolved by NVML UUID -- and it measures the chain,
not one stage, because §3's budget is a chain.

Three disciplines it does not skip:

* **A-vs-A floor first.** Two identical arms run before any number is
  reported, and the spread between them is printed as the noise floor. A
  stage difference smaller than that floor is not a difference. Every
  measurement canon violation this project has recorded started with skipping
  this.
* **Warm only.** Cold start is 20-30 s of kernel autotuning and weight
  paging, and quoting it against a 2-3 s conversational budget is meaningless.
  The warm-up runs are discarded explicitly rather than averaged in.
* **Time-boxed.** ``--budget-s`` caps the whole run; a window is a shared
  resource and an unbounded measurement loop is how one gets held too long.

The card is pinned by UUID before torch is imported, per the device-identity
rule -- torch's ordinal is not NVML's, and this rig has been bitten by that.

    # resolve the card first, never hardcode an index
    nvidia-smi --query-gpu=index,uuid,name --format=csv

    CUDA_VISIBLE_DEVICES=<uuid> PYTHONPATH=<repo>/python \\
      /spinning/htsglang-gpu/.venv/bin/python \\
      scripts/translator/latency_window.py --card-uuid GPU-... --budget-s 300
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ASR_LIB = Path("/spinning/llm_stuff/translator-models/asr-lib")
ASR_MODELS = Path("/spinning/llm_stuff/translator-models/asr-models")
TTS_MODEL = Path("/spinning/llm_stuff/translator-models/qwen3-tts-0.6b-base")
REFERENCE = Path("/spinning/llm_stuff/translator-models/xtts-v2/samples/de_sample.wav")
TEXT = "Hola, buenos dias. Me alegro mucho de verte otra vez."


def describe(label: str, samples):
    if not samples:
        print(f"[lat] {label:26s} no samples")
        return None
    median = statistics.median(samples)
    spread = max(samples) - min(samples)
    print(
        f"[lat] {label:26s} median {median * 1000:8.1f} ms   "
        f"min {min(samples) * 1000:8.1f}   max {max(samples) * 1000:8.1f}   "
        f"spread {spread * 1000:7.1f}   n={len(samples)}"
    )
    return median


async def run(args) -> int:
    if str(ASR_LIB) not in sys.path:
        sys.path.append(str(ASR_LIB))

    import soundfile as sf
    import torch

    from sglang.srt.translator.backends import AudioChunk
    from sglang.srt.translator.inprocess_tts import (
        InProcessQwen3Tts,
        InProcessTtsConfig,
    )

    print(f"[lat] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[lat] torch sees {torch.cuda.device_count()} device(s): "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    if not torch.cuda.is_available():
        print("[lat] no CUDA device visible; refusing to report a GPU number")
        return 1

    samples, rate = sf.read(str(args.reference), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    reference = AudioChunk(np.ascontiguousarray(samples), rate)

    deadline = time.monotonic() + args.budget_s

    # -- TTS ---------------------------------------------------------------
    tts = InProcessQwen3Tts(
        InProcessTtsConfig(
            model_dir=args.model_dir, device="cuda:0", dtype="bfloat16"
        )
    )
    t0 = time.monotonic()
    tts.load()
    print(f"[lat] TTS loaded in {time.monotonic() - t0:.1f}s (cold, not a budget number)")

    async def tts_once():
        started = time.monotonic()
        first = None
        total_samples = 0
        async for chunk in tts.synthesize(
            text=args.text, language=args.language, reference=reference
        ):
            if first is None:
                first = time.monotonic() - started
            total_samples += len(chunk.samples)
        elapsed = time.monotonic() - started
        seconds = total_samples / tts.sample_rate
        return first, elapsed, seconds

    print(f"[lat] warming TTS ({args.warmup} run(s), discarded)")
    for _ in range(args.warmup):
        await tts_once()

    ttfb, total, audio_s = [], [], []
    arm_a, arm_b = [], []
    iteration = 0
    while time.monotonic() < deadline and iteration < args.iterations:
        first, elapsed, seconds = await tts_once()
        if first is None:
            print("[lat] a run produced no audio; aborting rather than averaging it")
            return 1
        ttfb.append(first)
        total.append(elapsed)
        audio_s.append(seconds)
        # Interleaved A/A: identical work, alternating buckets. The spread
        # between the two medians is the floor any claim must clear.
        (arm_a if iteration % 2 == 0 else arm_b).append(first)
        iteration += 1

    print("")
    print(f"[lat] A-vs-A floor (identical work, interleaved, n={len(arm_a)}/{len(arm_b)})")
    median_a = describe("  arm A first-audio", arm_a)
    median_b = describe("  arm B first-audio", arm_b)
    if median_a is not None and median_b is not None:
        floor = abs(median_a - median_b)
        print(f"[lat]   noise floor {floor * 1000:.1f} ms -- "
              "a difference below this is not a difference")

    print("")
    print(f"[lat] TTS stage, warm, {len(ttfb)} runs")
    median_ttfb = describe("  first audio (TTFB)", ttfb)
    median_total = describe("  whole utterance", total)
    if median_total and audio_s:
        produced = statistics.median(audio_s)
        print(f"[lat]   {produced:.2f}s of audio, RTF {median_total / produced:.3f}")
    if median_ttfb:
        print("[lat]   against the §3 budget of 150-300 ms for TTS first audio, "
              "and window 3's 71 ms bar")

    # -- ASR ---------------------------------------------------------------
    if args.with_asr:
        from sglang.srt.translator.asr_backends import FasterWhisperAsr

        t0 = time.monotonic()
        asr = FasterWhisperAsr(
            model=args.asr_model,
            device="cuda",
            compute_type=args.asr_compute_type,
            download_root=ASR_MODELS,
            beam_size=1,
        )
        print(f"\n[lat] ASR loaded in {time.monotonic() - t0:.1f}s (cold)")
        for _ in range(args.warmup):
            await asr.transcribe(reference)
        asr_times = []
        while time.monotonic() < deadline and len(asr_times) < args.iterations:
            started = time.monotonic()
            await asr.transcribe(reference)
            asr_times.append(time.monotonic() - started)
        print(f"[lat] ASR stage, warm, {reference.duration_s:.2f}s utterance")
        describe("  transcribe", asr_times)

    # MT is deliberately NOT stubbed here. It is an HTTP hop to our own
    # 27B endpoint; with no server up, a number invented for it would be the
    # only fabricated row in the table. §3's 150-400 ms first-token stands
    # unmeasured until that server is running in the same window.
    print("\n[lat] MT not measured: no local endpoint in this window. "
          "§3's 150-400 ms MT first token remains unverified.")
    remaining = deadline - time.monotonic()
    print(f"[lat] budget: {args.budget_s - remaining:.0f}s used of {args.budget_s}s")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card-uuid", default=None,
                        help="NVML UUID; pins the process before torch is imported")
    parser.add_argument("--model-dir", type=Path, default=TTS_MODEL)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--text", default=TEXT)
    parser.add_argument("--language", default="es")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--budget-s", type=float, default=300.0)
    parser.add_argument("--with-asr", action="store_true")
    parser.add_argument("--asr-model", default="large-v3-turbo")
    parser.add_argument("--asr-compute-type", default="int8_float16")
    args = parser.parse_args()
    if args.card_uuid:
        # Before any torch import in this process: the pin must precede the
        # CUDA context, and after it cuda:0 is unambiguous.
        os.environ["CUDA_VISIBLE_DEVICES"] = args.card_uuid
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
