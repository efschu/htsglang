#!/usr/bin/env python3
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Derive a fused-tail SR artifact from the pinned fp32 one, and grade it (#457).

The surgery itself is ``video_enhance/fused_tail.py``; this is the build-time
driver around it, on the same pattern as ``export_sr_fp16.py``: derive, verify
loadability, write a provenance sidecar, then grade against a reference.

What makes this script worth running at a desk is that the whole gate is
CPU-executable. The candidate is an ONNX graph and the reference is
``resize.lanczos3_resize`` on the unfused graph's output, so onnxruntime's CPU
provider settles the question "does one engine compute what two stages
computed" at a small resolution, in seconds, with no card involved. The GPU
window then only has to answer what a desk cannot: the fp16 engine's ms/frame
and its parity on each arch.

``--arm`` selects which tail is built and graded:

``lanczos3``
    The production tail. Expect a very high PSNR -- the fused graph computes
    the reference filter, so the only difference is float rounding order.
``nearest``
    The can-fail arm. Sample-dropping decimation, which is a legitimate
    resize and a wrong one, so a gate that passes it is not a gate.
``bicubic_antialias``
    The route not taken, kept buildable so its cost against the Lanczos-3
    reference is a measured number rather than an argument.

Needs ``onnx`` for the graph surgery, which the serving path does not:

    PYTHONPATH=python:/tmp/onnxtools python \\
        scripts/video_enhance/export_sr_fused_tail.py --arm lanczos3 --grade
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sglang.srt.video_enhance.frame_math import Resolution


def _cpu_session(onnx_path: Path):
    """A CPU-provider session. The desk-side grader, not the serving path.

    ``backends.OnnxRuntimeBackend`` binds device pointers and refuses anything
    but the CUDA and TensorRT providers, which is right for the chain and
    useless for a host with no card. Grading the *graph* needs neither.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    return ort.InferenceSession(
        str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _run(session, array):
    name = session.get_inputs()[0].name
    return session.run(None, {name: array})[0]


def grade_on_cpu(
    source_onnx: Path,
    fused_onnx: Path,
    *,
    resolutions: list[Resolution],
    samples: int,
    net_scale: int,
) -> list[dict]:
    """Grade the fused graph against SR-then-Lanczos-3, both on the CPU provider.

    Inputs are sampled on the CPU with a fixed seed. That is the house rule for
    a reason recorded in ``export_sr_fp16.grade`` -- two architectures do not
    agree on ``torch.randn`` -- and it has a second benefit here: the same
    seed produces the same numbers on the GPU window's host, so the desk row
    and the window row are comparable rather than merely similar.
    """
    import torch

    from sglang.srt.video_enhance.fused_tail import grade_fused_tail

    reference_session = _cpu_session(source_onnx)
    candidate_session = _cpu_session(fused_onnx)
    results: list[dict] = []
    for resolution in resolutions:
        generator = torch.Generator().manual_seed(0x5EED)
        target = Resolution(resolution.width * net_scale, resolution.height * net_scale)
        for sample in range(samples):
            host = torch.rand(
                (1, 3, resolution.height, resolution.width),
                generator=generator,
                dtype=torch.float32,
            )
            array = host.numpy()
            sr_output = torch.from_numpy(_run(reference_session, array))
            candidate = torch.from_numpy(_run(candidate_session, array))
            if tuple(candidate.shape[2:]) != (target.height, target.width):
                raise SystemExit(
                    f"the fused graph emitted {tuple(candidate.shape)} at input "
                    f"{resolution}, expected 1x3x{target.height}x{target.width}. "
                    "The tail's stride or padding is wrong; nothing downstream "
                    "of this is meaningful."
                )
            verdict = grade_fused_tail(
                candidate,
                sr_output,
                target=target,
                note=f"{fused_onnx.name} @ {resolution} -> {target}",
            )
            results.append(
                {
                    "input_resolution": str(resolution),
                    "output_resolution": str(target),
                    "sample": sample,
                    "provider": "CPUExecutionProvider",
                    "sr_output_shape": list(sr_output.shape),
                    **verdict.as_manifest(),
                }
            )
    return results


def main(argv: list[str] | None = None) -> int:
    from sglang.srt.video_enhance.fused_tail import (
        FUSED_TAIL_SUFFIX,
        TAIL_KINDS,
        fuse_tail,
    )
    from sglang.srt.video_enhance.sr import REALESR_GENERAL_WDN_X4V3, fetch_model

    parser = argparse.ArgumentParser(description="#457 fused-tail SR artifact + parity")
    parser.add_argument("--model-dir", default="/spinning/llm_stuff/k3-models/sr")
    parser.add_argument("--out", default=None)
    parser.add_argument("--arm", default="lanczos3", choices=list(TAIL_KINDS))
    parser.add_argument("--grade", action="store_true")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument(
        "--resolutions",
        default="64x48,96x64",
        help=(
            "SR *input* resolutions to grade at. Small by default: the CPU "
            "provider runs the whole 34-convolution network per sample, and "
            "the property under test is a filter identity that does not depend "
            "on frame size."
        ),
    )
    parser.add_argument(
        "--expect-fail",
        action="store_true",
        help=(
            "invert the exit status: the run succeeds only if the gate REJECTS "
            "the arm. This is how the can-fail proof is executed rather than "
            "asserted."
        ),
    )
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)

    model_dir = Path(args.model_dir)
    source = fetch_model(REALESR_GENERAL_WDN_X4V3, model_dir)
    suffix = FUSED_TAIL_SUFFIX
    if args.arm != "lanczos3":
        suffix = f"_{args.arm}{FUSED_TAIL_SUFFIX}"
    out_path = Path(args.out or model_dir / (source.stem + suffix))

    manifest = fuse_tail(
        source, out_path, kind=args.arm, model_scale=REALESR_GENERAL_WDN_X4V3.scale
    )
    report: dict = {"artifact": manifest}
    print(json.dumps({"artifact": manifest}, indent=2), flush=True)

    if args.grade:
        resolutions = [Resolution.parse(r) for r in args.resolutions.split(",")]
        report["parity"] = grade_on_cpu(
            source,
            out_path,
            resolutions=resolutions,
            samples=args.samples,
            net_scale=int(manifest["net_scale"]),
        )
        print(json.dumps({"parity": report["parity"]}, indent=2))
        failed = [r for r in report["parity"] if not r["passed"]]
        if args.report:
            Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True))
        if args.expect_fail:
            if failed:
                print(
                    f"GATE REJECTED the {args.arm} arm in {len(failed)} of "
                    f"{len(report['parity'])} samples, as it must",
                )
                return 0
            print(
                f"the {args.arm} arm PASSED a gate it was built to fail; the "
                "gate is not measuring what it claims to",
                file=sys.stderr,
            )
            return 1
        if failed:
            print(
                f"PARITY GATE FAILED for {len(failed)} of {len(report['parity'])} "
                f"samples on the {args.arm} tail",
                file=sys.stderr,
            )
            return 1
        print("PARITY GATE PASSED")
    elif args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
