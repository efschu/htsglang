# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Does the unrolled code predictor say the same thing as the reference?

The licence for #466 (d) was explicit that the AUDIO need not be byte-identical
-- this is a live translator, not a determinism harness. That is a licence about
bits, not about meaning. An unrolled loop that changed the sampling order, the
RNG draw sequence, the stopping rule or the residual-codebook order would change
what is SAID, and no amount of "the waveform differs only slightly" would catch
it: the codebooks are discrete, so a single different draw is a different sound,
not a quieter one.

So this probe does not measure a distance. It checks IDENTITY of the discrete
objects, which is the only statement about a sampler that means anything:

* **Part A, the predictor alone.** Same seed, same input hidden states, run the
  reference ``generate`` and the unrolled loop. The sixteen codebook ids of a
  frame must be equal ELEMENTWISE. Repeated over many draws and over both a
  cold and a warm RNG so the comparison covers more than one point of the
  sampler's stream.
* **Part B, the whole shipped call.** Same seed, same text, same reference clip,
  ``InProcessQwen3Tts._generate`` end to end with the loop off and on. The
  codec frame COUNT, the waveform LENGTH and the waveform SAMPLES are compared.
  A listener notices a different word, a different length, or a truncation;
  those are what this reports, in those terms.

Part B runs on the CPU deliberately. The claim under test is about control flow
and RNG consumption, both of which are device-independent, and the CPU arm needs
no window on a card the live translator is serving from. It is slow (~333 ms per
talker step, RTF 4.19, measured) but a two-clause text is under a minute.

    PYTHONPATH=<repo>/python /spinning/htsglang-gpu/.venv/bin/python \\
      scripts/translator/probe_unrolled_predictor.py --draws 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from sglang.srt.translator import code_predictor_loop  # noqa: E402
from sglang.srt.translator.audio import AudioChunk  # noqa: E402
from sglang.srt.translator.inprocess_tts import (  # noqa: E402
    InProcessQwen3Tts,
    InProcessTtsConfig,
)

VOICES = Path("/spinning/llm_stuff/translator-models/preset-voices")
RATE = 24000

#: A real translation from the 2026-08-04 server transcript, so the step count
#: is one the fit was made against rather than an invented string.
TEXT = "Hola, soy Matthias y estoy de vacaciones aqui. Como estas?"


def load_reference(path: Path, seconds: float) -> np.ndarray:
    data, rate = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if rate != RATE:
        raise SystemExit(f"{path} is {rate} Hz, expected {RATE}")
    want = int(seconds * RATE)
    if len(data) < want:
        raise SystemExit(f"{path} is {len(data) / RATE:.2f}s, need {seconds}s")
    return data[:want]


def part_a(torch, predictor, draws: int, prewarm: int) -> Dict:
    """Reference generate versus unrolled loop, on the predictor alone.

    ``prewarm`` draws from the global RNG before each pair, so the comparison
    does not only ever start from a freshly seeded generator: a loop that
    consumed a different NUMBER of random values would agree at draw 0 and
    diverge afterwards, and seeding before every single call would hide exactly
    that.
    """
    device = next(predictor.parameters()).device
    dtype = next(predictor.parameters()).dtype
    hidden = predictor.small_to_mtp_projection
    width = (
        hidden.in_features
        if hasattr(hidden, "in_features")
        else predictor.config.hidden_size
    )
    groups = predictor.config.num_code_groups - 1

    mismatches: List[Dict] = []
    checked = 0
    for draw in range(draws):
        # Sampled on the CPU and moved, never generated on the device: a
        # device-side `randn` is not architecture-identical, and the input has
        # to be the same bits in both arms for the comparison to mean anything.
        torch.manual_seed(9000 + draw)
        embeds = torch.randn(1, 2, width, dtype=torch.float32).to(
            device=device, dtype=dtype
        )
        arms = {}
        for name, enabled in (("reference", False), ("unrolled", True)):
            code_predictor_loop.set_enabled(enabled)
            torch.manual_seed(4242 + draw)
            for _ in range(prewarm):
                torch.rand(1)
            with torch.inference_mode():
                result = predictor.generate(
                    inputs_embeds=embeds,
                    max_new_tokens=groups,
                    do_sample=True,
                    top_p=1.0,
                    top_k=50,
                    temperature=0.9,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )
            arms[name] = result.sequences.detach().cpu()
            # The RNG must be left in the same place too, or the TALKER's own
            # next draw diverges even when this call agreed.
            arms[name + "_rng"] = torch.random.get_rng_state().clone()
        checked += 1
        same_codes = bool(torch.equal(arms["reference"], arms["unrolled"]))
        same_rng = bool(torch.equal(arms["reference_rng"], arms["unrolled_rng"]))
        if not (same_codes and same_rng):
            mismatches.append(
                {
                    "draw": draw,
                    "same_codes": same_codes,
                    "same_rng_state": same_rng,
                    "reference": arms["reference"].tolist(),
                    "unrolled": arms["unrolled"].tolist(),
                }
            )
    return {
        "draws": checked,
        "groups_per_draw": groups,
        "mismatches": mismatches,
        "identical": not mismatches,
    }


def part_b(torch, backend, text: str, reference: AudioChunk, seed: int) -> Dict:
    """The whole shipped call, loop off then on, from the same seed."""
    arms = {}
    for name, enabled in (("reference", False), ("unrolled", True)):
        code_predictor_loop.set_enabled(enabled)
        torch.manual_seed(seed)
        waveform = backend._generate(text, "es", reference, None)
        arms[name] = np.asarray(waveform, dtype=np.float32)

    left, right = arms["reference"], arms["unrolled"]
    same_length = left.shape == right.shape
    if same_length:
        identical = bool(np.array_equal(left, right))
        max_abs = float(np.max(np.abs(left - right))) if not identical else 0.0
    else:
        identical = False
        max_abs = float("nan")
    return {
        "samples_reference": int(left.size),
        "samples_unrolled": int(right.size),
        "seconds_reference": round(left.size / RATE, 3),
        "seconds_unrolled": round(right.size / RATE, 3),
        "same_length": same_length,
        "identical_samples": identical,
        "max_abs_difference": max_abs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/spinning/llm_stuff/translator-models/qwen3-tts-0.6b-base"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--draws", type=int, default=12)
    parser.add_argument("--prewarm", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-part-b", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=logging.INFO)
    import torch

    backend = InProcessQwen3Tts(
        InProcessTtsConfig(
            model_dir=args.model_dir,
            device=args.device,
            dtype=args.dtype,
        )
    )
    backend.load()
    inner = getattr(backend._model, "model", backend._model)
    predictor = backend._resolve(inner, "talker.code_predictor")
    if predictor is None:
        raise SystemExit("no talker.code_predictor on this checkpoint")

    report: Dict[str, object] = {
        "device": args.device,
        "dtype": args.dtype,
        "part_a": part_a(torch, predictor, args.draws, args.prewarm),
    }
    print(json.dumps({"event": "part_a", **report["part_a"]}, indent=1))

    if not args.skip_part_b:
        clip = AudioChunk(
            load_reference(VOICES / "man" / "man-03.de.wav", 3.22), RATE
        )
        report["part_b"] = part_b(torch, backend, TEXT, clip, args.seed)
        print(json.dumps({"event": "part_b", **report["part_b"]}, indent=1))

    report["stats"] = code_predictor_loop.loop_stats()
    ok = report["part_a"]["identical"] and (
        args.skip_part_b or report["part_b"]["identical_samples"]
    )
    report["verdict"] = "identical" if ok else "DIVERGED"
    print(json.dumps({"event": "verdict", "verdict": report["verdict"],
                      "stats": report["stats"]}))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
