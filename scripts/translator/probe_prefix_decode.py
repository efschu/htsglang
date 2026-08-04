#!/usr/bin/env python3
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Where does a prefix decode stop agreeing with the whole decode?

`probe_stream_emission.py` found that streaming a unit and decoding it in one
pass give waveforms that differ by up to 0.028 of full scale -- far too much
to be float noise, and the incremental emission is only sound if that
difference is understood.

The claim it tests was structural: the codec decoder is causal (left-only conv
padding, right-trimmed transposed convolutions, causal sliding-window
attention), so `decode(codes[:k])` should be a prefix of `decode(codes[:n])`
sample for sample. This measures the claim directly, on real codes, with no
generation involved -- one talker run supplies the frames and everything after
is decode.

The shape of the answer decides the fix. If the disagreement is spread over
the whole prefix, causality is simply false here and incremental decode is not
available. If it is confined to the LAST few frames of each prefix, then the
decoder has a right edge -- the tail of a prefix is computed as if the
utterance ended there -- and the repair is a holdback wide enough to cover it,
which costs pre-roll and nothing else.

    /spinning/htsglang-gpu/.venv/bin/python \\
      scripts/translator/probe_prefix_decode.py
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
SAMPLES_PER_FRAME = 1920
TEXT = "Hola, soy Matthias y estoy de vacaciones aqui. Como estas?"


def main() -> int:
    import torch

    data, rate = sf.read(str(VOICES / "man" / "man-03.de.wav"), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    reference = AudioChunk(data[: int(3.22 * RATE)], RATE)

    backend = InProcessQwen3Tts(InProcessTtsConfig(stream_within_unit=False))
    backend.load()

    # One generation, captured frame by frame, then nothing but decode.
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
    print(f"frames={total} full_samples={len(full)}")

    rows = []
    for k in range(10, total, 5):
        prefix = backend._decode_frames(frames[:k])
        span = min(len(prefix), k * SAMPLES_PER_FRAME)
        diff = np.abs(prefix[:span] - full[:span])
        bad = np.flatnonzero(diff > 1e-6)
        # How far back from the END of the prefix the first disagreement sits,
        # expressed in codec frames: this is the width of the right edge.
        edge_frames = (
            (span - int(bad[0])) / SAMPLES_PER_FRAME if bad.size else 0.0
        )
        rows.append({
            "k": k,
            "max_abs": float(diff.max()) if span else 0.0,
            "n_differing": int(bad.size),
            "first_bad_sample": int(bad[0]) if bad.size else None,
            "edge_frames": round(edge_frames, 2),
            "max_abs_excluding_last_16_frames": float(
                diff[: max(0, span - 16 * SAMPLES_PER_FRAME)].max()
            ) if span > 16 * SAMPLES_PER_FRAME else None,
        })
        print(json.dumps(rows[-1]))

    edges = [r["edge_frames"] for r in rows if r["n_differing"]]
    summary = {
        "frames": total,
        "prefixes_tested": len(rows),
        "prefixes_differing": sum(1 for r in rows if r["n_differing"]),
        "edge_frames_max": max(edges) if edges else 0.0,
        "interior_max_abs": max(
            (r["max_abs_excluding_last_16_frames"] for r in rows
             if r["max_abs_excluding_last_16_frames"] is not None), default=None),
    }
    print(json.dumps(summary, indent=2))
    Path("/spinning/466-client-logs/prefix_decode.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
