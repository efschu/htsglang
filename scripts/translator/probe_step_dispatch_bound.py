# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Is the talker's decode step dispatch-bound or arithmetic-bound?

The latency breakdown for #466 measured 22.6 ms for one trunk decode step and
3.95 ms for one code-predictor call -- 0.81 and 0.79 ms per transformer layer,
against a batch-1 memory roofline of ~0.014 ms per layer. That is 57x, and
the conclusion drawn from it ("the cost is dispatch, not arithmetic") drives
the recommendation to spend effort on CUDA-graph capture rather than on
TensorRT. A recommendation that expensive should not rest on a roofline
division.

THE DISCRIMINATOR. Run the same forward at increasing batch size with
everything else held fixed:

* if the step is ARITHMETIC- or BANDWIDTH-bound, wall time grows roughly
  linearly with batch, because there is proportionally more work to do;
* if the step is DISPATCH- or LATENCY-bound, wall time is nearly FLAT in
  batch, because the same number of kernel launches and the same Python
  per-layer work now cover more rows.

Flatness is the signature. The batch axis is a diagnostic only -- the talker
serves one conversation at a time and this is not a proposal to batch it.

    CUDA_VISIBLE_DEVICES=<uuid> PYTHONPATH=<repo>/python \\
      python probe_step_dispatch_bound.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from sglang.srt.translator.inprocess_tts import (  # noqa: E402
    InProcessQwen3Tts,
    InProcessTtsConfig,
)


def time_forward(torch, module, batch, hidden, device, dtype, kv_len,
                 repeats, warmup):
    """Time one decode-shaped forward (query length 1) at a given batch."""
    hidden_states = torch.randn(
        batch, 1, hidden, device=device, dtype=dtype
    )
    samples = []
    with torch.inference_mode():
        for index in range(repeats + warmup):
            torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                module(inputs_embeds=hidden_states)
            except TypeError:
                # Some trunks take positional hidden states instead.
                module(hidden_states)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000.0
            if index >= warmup:
                samples.append(elapsed)
    del hidden_states
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path,
        default=Path("/spinning/llm_stuff/translator-models/qwen3-tts-0.6b-base"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batches", default="1,2,4,8,16,32")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=logging.WARNING)
    import torch

    backend = InProcessQwen3Tts(
        InProcessTtsConfig(
            model_dir=args.model_dir, device=args.device, dtype=args.dtype
        )
    )
    backend.load()
    inner = getattr(backend._model, "model", backend._model)
    dtype = getattr(torch, args.dtype)
    hidden = backend.geometry.hidden_size

    targets = {
        "talker_trunk": backend._resolve(inner, "talker.model"),
        "code_predictor_trunk": backend._resolve(
            inner, "talker.code_predictor.model"
        ),
    }
    results = {}
    for name, module in targets.items():
        if module is None:
            results[name] = {"error": "not resolved"}
            continue
        rows = {}
        for batch in [int(b) for b in args.batches.split(",")]:
            try:
                samples = time_forward(
                    torch, module, batch, hidden, args.device, dtype,
                    kv_len=1, repeats=args.repeats, warmup=args.warmup,
                )
            except Exception as exc:
                rows[batch] = {"error": str(exc)[:200]}
                continue
            rows[batch] = {
                "median_ms": round(statistics.median(samples), 4),
                "min_ms": round(min(samples), 4),
                "stdev_ms": round(statistics.stdev(samples), 4)
                if len(samples) > 1 else 0.0,
            }
        results[name] = rows
        base = rows.get(1, {}).get("median_ms")
        if base:
            print(f"== {name}: batch-1 = {base:.3f} ms")
            for batch, row in rows.items():
                if "median_ms" in row:
                    print(
                        f"   batch {batch:3d}: {row['median_ms']:8.3f} ms"
                        f"   x_batch1 = {row['median_ms']/base:5.2f}"
                        f"   (linear would be {batch:.2f})"
                    )
    print(json.dumps({"event": "summary", "results": results}))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
