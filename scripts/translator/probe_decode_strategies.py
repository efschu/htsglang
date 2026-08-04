#!/usr/bin/env python3
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Which incremental decode is closest to the shipped one, and is it audible?

`probe_prefix_decode.py` falsified the structural argument that decoding a
growing prefix reproduces the one-shot decode: the disagreement starts at
sample ZERO, so it is not a right edge and no holdback removes it. The decoder
has a sequence-global dependence.

That leaves a measurement rather than an argument. Two incremental strategies
are possible without touching the vendored package, and the question is which
is closer to the shipped waveform and whether what remains can be heard:

* **growing prefix** -- decode `codes[:k]` each time and send the new tail.
  Every sample is produced by a decode that saw a different total length.
* **sliding window** -- decode `codes[k-context : k+stride]` and keep only the
  new frames' samples. This is the vendored `chunked_decode`'s OWN recipe
  (300-frame chunks, 25 frames of dropped left context) at a smaller chunk
  size, i.e. the same class of approximation the package already ships and
  accepts at every one of its own boundaries.

Reported for each: signal-to-noise against the one-shot decode, and the
discontinuity at the chunk seams, because a global error that is smooth is a
level change nobody notices while the same error concentrated at a seam is a
click every 0.4 s. The seam figure is compared against the seam in the
one-shot decode at the same sample positions -- speech has real transients, so
an absolute jump means nothing without that control.

    /spinning/htsglang-gpu/.venv/bin/python \\
      scripts/translator/probe_decode_strategies.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from sglang.srt.translator.backends import AudioChunk  # noqa: E402
from sglang.srt.translator.inprocess_tts import (  # noqa: E402
    InProcessQwen3Tts,
    InProcessTtsConfig,
)

VOICES = Path("/spinning/llm_stuff/translator-models/preset-voices")
RATE = 24000
SPF = 1920
STRIDE = 5
TEXT = "Hola, soy Matthias y estoy de vacaciones aqui. Como estas?"


def snr_db(reference: np.ndarray, test: np.ndarray) -> float:
    n = min(len(reference), len(test))
    noise = reference[:n] - test[:n]
    power = float(np.mean(reference[:n] ** 2))
    noise_power = float(np.mean(noise ** 2))
    if noise_power <= 0.0:
        return float("inf")
    return 10.0 * np.log10(power / noise_power)


def seam_steps(wave: np.ndarray, boundaries: List[int]) -> float:
    """Largest single-sample jump at the chunk boundaries."""
    steps = [
        abs(float(wave[b]) - float(wave[b - 1]))
        for b in boundaries
        if 0 < b < len(wave)
    ]
    return max(steps) if steps else 0.0


def main() -> int:
    import torch

    data, rate = sf.read(str(VOICES / "man" / "man-03.de.wav"), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    reference = AudioChunk(data[: int(3.22 * RATE)], RATE)

    backend = InProcessQwen3Tts(InProcessTtsConfig(stream_within_unit=False))
    backend.load()

    frames: List[object] = []
    inner = getattr(backend._model, "model", backend._model)

    def _on_step(_module, _inputs, output):
        hidden = getattr(output, "hidden_states", None)
        if isinstance(hidden, (tuple, list)) and hidden and hidden[-1] is not None:
            frames.append(hidden[-1])

    handle = inner.talker.register_forward_hook(_on_step)
    try:
        torch.manual_seed(1000)
        backend._generate_once(TEXT, "es", reference, None, 800, None)
    finally:
        handle.remove()

    total = len(frames)
    full = backend._decode_frames(frames)
    boundaries = [k * SPF for k in range(STRIDE, total, STRIDE)]

    results = {}

    # 1. growing prefix, which is what the module does today
    parts: List[np.ndarray] = []
    done = 0
    for k in range(STRIDE, total + 1, STRIDE):
        wave = backend._decode_frames(frames[:k])
        parts.append(wave[done : k * SPF])
        done = k * SPF
    if done < len(full):
        parts.append(backend._decode_frames(frames)[done:])
    grown = np.concatenate(parts)
    results["growing_prefix"] = {
        "samples": int(len(grown)),
        "snr_db": snr_db(full, grown),
        "max_abs": float(np.abs(full[: len(grown)] - grown[: len(full)]).max()),
        "seam_max_step": seam_steps(grown, boundaries),
    }

    # 2. sliding window with dropped left context, the vendor's own recipe
    for context in (25, 50):
        parts = []
        for start in range(0, total, STRIDE):
            end = min(start + STRIDE, total)
            back = min(context, start)
            wave = backend._decode_frames(frames[start - back : end])
            parts.append(wave[back * SPF :])
        slid = np.concatenate(parts)
        results[f"sliding_context_{context}"] = {
            "samples": int(len(slid)),
            "snr_db": snr_db(full, slid),
            "max_abs": float(np.abs(full[: len(slid)] - slid[: len(full)]).max()),
            "seam_max_step": seam_steps(slid, boundaries),
        }

    results["one_shot_control"] = {
        "samples": int(len(full)),
        "snr_db": float("inf"),
        "max_abs": 0.0,
        # THE CONTROL the seam numbers are meaningless without: real speech has
        # transients, so what matters is whether a seam is bigger than the
        # jumps already present at those sample positions.
        "seam_max_step": seam_steps(full, boundaries),
        "signal_peak": float(np.abs(full).max()),
        "signal_rms": float(np.sqrt(np.mean(full ** 2))),
    }

    print(json.dumps(results, indent=2))
    Path("/spinning/466-client-logs/decode_strategies.json").write_text(
        json.dumps({"frames": total, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
