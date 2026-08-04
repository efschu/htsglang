# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Does a ledger park/restore round trip leave the audio modules usable?

Found while pricing the fixed per-turn cost for #466: driving the shipped
`_generate` with an explicit `park()` in front of it dies with

    torch.OutOfMemoryError: CUDA out of memory. Tried to allocate more
    than 1EB memory

inside the Mimi codec's `_pad1d`, on the reference ENCODE.

THE READING UNDER TEST. `AudioAssetLedger.park` releases VRAM with
`module.to("meta")`, and `restore` brings the module back with
`to_empty(device=target)` followed by `load_state_dict(..., strict=True)`.
`state_dict()` does not contain NON-PERSISTENT buffers, and `to_empty`
allocates uninitialised storage for everything -- so any non-persistent
buffer survives the round trip as garbage rather than as its value.

This checkpoint is known to have exactly that shape: `qwen3_tts_compat`
carries `refresh_rotary_buffers` precisely because "non-persistent rotary
buffers do not survive 5.x's meta-device construction; unrefreshed they are
NaN". `restore` re-enters that same meta path and never re-runs the repair.

So this counts buffers that are finite before a park and non-finite after the
restore. A count of zero falsifies the reading and the OOM needs another
explanation; a non-zero count locates it.

    CUDA_VISIBLE_DEVICES=<uuid> PYTHONPATH=<repo>/python python probe_park_restore.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from sglang.srt.translator.inprocess_tts import (  # noqa: E402
    InProcessQwen3Tts,
    InProcessTtsConfig,
)


def survey(backend, torch) -> dict:
    """Per registered asset: how many tensors are finite, and how many exist."""
    out = {}
    for name in backend.ledger.names():
        asset = backend.ledger.get(name)
        module = asset.module
        persistent = set(module.state_dict().keys())
        total = 0
        nonfinite = 0
        nonpersistent_total = 0
        nonpersistent_nonfinite = 0
        for buffer_name, tensor in module.named_buffers():
            if tensor is None or not torch.is_tensor(tensor):
                continue
            if tensor.device.type == "meta" or not tensor.is_floating_point():
                continue
            total += 1
            finite = bool(torch.isfinite(tensor).all().item())
            if not finite:
                nonfinite += 1
            if buffer_name not in persistent:
                nonpersistent_total += 1
                if not finite:
                    nonpersistent_nonfinite += 1
        out[name] = {
            "float_buffers": total,
            "nonfinite": nonfinite,
            "nonpersistent_float_buffers": nonpersistent_total,
            "nonpersistent_nonfinite": nonpersistent_nonfinite,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path,
        default=Path("/spinning/llm_stuff/translator-models/qwen3-tts-0.6b-base"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
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
    before = survey(backend, torch)
    print(json.dumps({"phase": "loaded", "assets": before}, indent=2))

    # Checksums BEFORE the park, so "the values came back" is a comparison
    # rather than a claim. Buffers are included because they are the tensors
    # `load_state_dict` cannot refill when they are non-persistent.
    def fingerprint() -> dict:
        marks = {}
        for name in backend.ledger.names():
            module = backend.ledger.get(name).module
            for kind, items in (
                ("param", module.named_parameters()),
                ("buffer", module.named_buffers()),
            ):
                for tensor_name, tensor in items:
                    if tensor is None or not torch.is_tensor(tensor):
                        continue
                    if tensor.device.type == "meta":
                        continue
                    if not tensor.is_floating_point():
                        continue
                    # Reduce in fp32 WITHOUT materialising an fp32 copy of the
                    # tensor: `.float()` on the codec's larger weights is a
                    # 1.16 GiB allocation on a card that is already shared.
                    marks[f"{name}.{kind}.{tensor_name}"] = (
                        float(tensor.detach().abs().sum(dtype=torch.float32).item()),
                        tuple(tensor.shape),
                    )
        return marks

    marks_before = fingerprint()

    freed = backend.park()
    restored = backend.ensure_resident()
    after = survey(backend, torch)
    marks_after = fingerprint()

    changed = []
    missing = []
    for key, value in marks_before.items():
        if key not in marks_after:
            missing.append(key)
            continue
        other = marks_after[key]
        if other[1] != value[1]:
            changed.append({"tensor": key, "shape_before": value[1],
                            "shape_after": other[1]})
        elif abs(other[0] - value[0]) > max(1e-3, 1e-6 * abs(value[0])):
            changed.append({"tensor": key, "abs_sum_before": value[0],
                            "abs_sum_after": other[0]})
    print(
        json.dumps(
            {
                "phase": "value_preservation",
                "tensors_checked": len(marks_before),
                "missing_after": missing[:20],
                "changed_after": changed[:20],
                "changed_count": len(changed),
            },
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "phase": "after_park_restore",
                "freed_bytes": freed,
                "freed_mib": round(freed / (1 << 20), 1),
                "restore_ms": {k: round(v, 1) for k, v in restored.items()},
                "assets": after,
            },
            indent=2,
        )
    )

    verdict = {
        name: {
            "nonfinite_before": before[name]["nonfinite"],
            "nonfinite_after": after[name]["nonfinite"],
            "nonpersistent_buffers": after[name]["nonpersistent_float_buffers"],
        }
        for name in after
    }
    broke = [n for n, v in verdict.items()
             if v["nonfinite_after"] > v["nonfinite_before"]]
    print(json.dumps({"phase": "verdict", "per_asset": verdict,
                      "corrupted_assets": broke}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
