# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Does the unrolled code predictor put the real-time factor below 1.0?

THE NUMBER THAT DECIDES. Audio is consumed at 12.5 codec frames per second and
the talker produces one frame per decode step, so

    RTF = 12.5 / steps_per_second

and playback of a turn carrying D seconds of speech, started after a pre-roll P,
underruns exactly when P < (R - 1) x D. `MEASURE_TTS_LATENCY.md` measured
R = 1.158 idle and 1.256 with the 27B saturated; the field then produced the
predicted failure -- two underruns, both immediately before the FINAL chunk of a
long turn. R below 1.0 removes the inequality entirely: the deficit becomes a
surplus and turn length stops mattering. From the saturated 9.95 steps/s the
requirement is 12.5/9.95 = **1.26x**.

HOW THIS IS MEASURED, and why it is built this way.

* **Arms are INTERLEAVED, never blocked.** A card that drifts during the run
  must not be able to masquerade as an arm effect. The order alternates
  (off, on, on, off) within each round so that even a monotone drift cancels
  to first order rather than loading one arm.
* **Both arms share a SEED per round.** The step count of a generation is a
  draw from the sampler with a 15 % standard deviation -- larger than the
  effect on a single call. Because the unrolled loop is bit-identical to the
  reference (`probe_unrolled_predictor.py`), the same seed makes both arms
  decode the SAME token sequence, so the comparison is the same work done two
  ways rather than two different amounts of work.
* **An A-versus-A floor is established before any delta is read.** Each arm is
  run twice per round under the same seed. Their spread is the noise floor, and
  a delta smaller than it is not reported as a delta.
* **Warm-up draws are discarded**, as every earlier round on this card did.

The rate metric is `steps / whole_call_seconds`, the same convention as section
5 of the measurement doc (which reported 10.79 steps/s idle), so the numbers
here can be read straight against it. It includes the ~81 ms of fixed per-call
cost, which makes it slightly pessimistic -- the right direction for a bar.

Step count comes from the predictor's own call counter: the talker calls the
code predictor exactly once per decode step, in both arms, so the counter delta
IS the step count and no extra hook is needed to obtain it.

    CUDA_VISIBLE_DEVICES=<uuid-of-the-5090> PYTHONPATH=<repo>/python \\
      /spinning/htsglang-gpu/.venv/bin/python \\
      scripts/translator/probe_unrolled_rtf.py --draws 6
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from sglang.srt.translator import code_predictor_loop  # noqa: E402
from sglang.srt.translator.audio import AudioChunk  # noqa: E402
from sglang.srt.translator.inprocess_tts import (  # noqa: E402
    CODEC_FRAME_RATE_HZ,
    InProcessQwen3Tts,
    InProcessTtsConfig,
)

VOICES = Path("/spinning/llm_stuff/translator-models/preset-voices")
RATE = 24000
REF_S = 3.22

#: The 58-character field translation every earlier arm was measured on, so
#: `steps/s` here is comparable to the 10.79 / 10.29 / 9.95 of section 5.
TEXT_FIELD = "Hola, soy Matthias y estoy de vacaciones aqui. Como estas?"

#: And the long shape that actually broke in the field, because the buffer
#: deficit grows with duration and the failure sat before the LAST chunk.
TEXT_MONOLOGUE = (
    "Estamos de vacaciones aqui en la costa desde el martes pasado, "
    "y manana por la manana queremos ir a la playa temprano. Hace calor."
)


def load_reference() -> AudioChunk:
    path = VOICES / "man" / "man-03.de.wav"
    data, rate = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if rate != RATE:
        raise SystemExit(f"{path} is {rate} Hz, expected {RATE}")
    return AudioChunk(data[: int(REF_S * RATE)], RATE)


def _calls() -> int:
    stats = code_predictor_loop.loop_stats()
    return stats["calibrated"] + stats["unrolled"] + stats["fell_back"]


def run_one(backend, torch, text: str, reference, seed: int,
            enabled: bool) -> Dict:
    code_predictor_loop.set_enabled(enabled)
    torch.manual_seed(seed)
    before = _calls()
    torch.cuda.synchronize()
    started = time.perf_counter()
    waveform = backend._generate(text, "es", reference, None, None, None)
    torch.cuda.synchronize()
    call_s = time.perf_counter() - started
    steps = _calls() - before
    audio_s = len(waveform) / RATE
    steps_per_s = steps / call_s if call_s else float("nan")
    return {
        "loop": "on" if enabled else "off",
        "seed": seed,
        "chars": len(text),
        "steps": steps,
        "call_ms": call_s * 1000.0,
        "ms_per_step": call_s * 1000.0 / steps if steps else float("nan"),
        "steps_per_s": steps_per_s,
        # The doc's convention, and the one the gapless inequality uses.
        "rtf": CODEC_FRAME_RATE_HZ / steps_per_s if steps_per_s else None,
        # The end-to-end factor including the trailing-silence trim, kept
        # separate because the trim shortens the AUDIO after the steps have
        # already been spent and would flatter the rate above.
        "rtf_wall_over_audio": call_s / audio_s if audio_s else None,
        "audio_s": audio_s,
        "samples": int(len(waveform)),
    }


def summarize(rows: List[Dict], key: str) -> Dict:
    values = [r[key] for r in rows if r[key] is not None]
    if not values:
        return {}
    return {
        "n": len(values),
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--texts", default="field,monologue")
    parser.add_argument("--label", default="idle")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=logging.WARNING)
    import torch

    texts = {"field": TEXT_FIELD, "monologue": TEXT_MONOLOGUE}
    wanted = [n.strip() for n in args.texts.split(",") if n.strip()]

    reference = load_reference()
    backend = InProcessQwen3Tts(InProcessTtsConfig())
    backend.load()

    # Discarded as JIT and allocator outliers, exactly as the latency round
    # did. One per arm, so neither arm pays another's first-call cost.
    for _ in range(args.warmup):
        for enabled in (False, True):
            code_predictor_loop.set_enabled(enabled)
            torch.manual_seed(1)
            backend._generate(TEXT_FIELD, "es", reference, None, None, None)

    records: List[Dict] = []
    for draw in range(args.draws):
        for name in wanted:
            seed = 1000 + draw
            # (off, on, on, off): two draws of each arm per round, in an order
            # whose midpoint is the same for both, so a monotone drift cancels
            # instead of loading one arm. The repeated pairs are the A-vs-A.
            for slot, enabled in enumerate((False, True, True, False)):
                row = run_one(backend, torch, texts[name], reference, seed,
                              enabled)
                row.update({"draw": draw, "text": name, "slot": slot})
                records.append(row)
                # Flushed: a run this long is watched from its log, and a
                # block-buffered stdout makes a live probe look hung.
                print(json.dumps(row), flush=True)

    report: Dict[str, object] = {"label": args.label, "records": records}
    for name in wanted:
        for arm in ("off", "on"):
            rows = [r for r in records
                    if r["text"] == name and r["loop"] == arm]
            report[f"{name}_{arm}"] = {
                "steps_per_s": summarize(rows, "steps_per_s"),
                "rtf": summarize(rows, "rtf"),
                "ms_per_step": summarize(rows, "ms_per_step"),
                "steps": summarize(rows, "steps"),
            }
        off = report[f"{name}_off"]["steps_per_s"].get("median")
        on = report[f"{name}_on"]["steps_per_s"].get("median")
        if off and on:
            report[f"{name}_speedup"] = round(on / off, 4)
            # The A-vs-A floor: the spread WITHIN each arm, which any claimed
            # delta has to clear before it is a delta at all.
            for arm in ("off", "on"):
                block = report[f"{name}_{arm}"]["steps_per_s"]
                block["spread_pct"] = round(
                    100.0 * (block["max"] - block["min"]) / block["median"], 2
                )
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "records"},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
