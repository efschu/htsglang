#!/usr/bin/env python3
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Build the fp16 SR engine with TensorRT directly, grade it, and time it.

**Why this exists instead of going through ONNX Runtime.** TASK_333 §2.1
chose ONNX Runtime's TensorRT execution provider precisely so that no second
TensorRT distribution would be needed. That choice is still right for
serving, and it is unavailable here: the wheel in this venv is
``onnxruntime-gpu 1.28.0``, whose TRT EP links ``libnvinfer.so.10`` and
``libnvonnxparser.so.10``, and the TensorRT installed is ``11.2.1.2``, which
provides ``libnvinfer.so.11``. A major-version mismatch, so the EP cannot
load and every session that asks for it silently falls back to CUDA. The
version pair is the finding; forcing them together with ``LD_PRELOAD`` in a
shared venv would produce a number nobody could trust and a rig nobody else
could use.

So the engine is built through the TensorRT Python API, which is the
supported way to use the TensorRT that is actually installed, and the result
is benched standalone. What it answers is the question the whole thread is
waiting on: **does an fp16 TensorRT engine move the 2.4-2.5x 3080-to-5090 SR
ratio that was measured on the fp32 CUDA provider?**

**TensorRT 11 is strongly typed and that simplifies the argument.** There is
no ``BuilderFlag.FP16`` any more; ``NetworkDefinitionCreationFlag.STRONGLY_TYPED``
is the only precision-relevant flag and precision comes from the graph. So an
fp16 ONNX yields an fp16 engine and an fp32 ONNX yields an fp32 engine, with
no builder-side permission involved -- which is exactly the ``STRONGLY_TYPED``
route ``vs-pipeline`` took (§10) and removes the usual "was it *allowed* to
run in half precision or did it actually" ambiguity from the comparison.

**Engine-build discipline**, recorded in the manifest next to every engine:

*   the **I/O signature** -- every binding's name, mode, dtype and shape, read
    back off the built engine rather than assumed from the ONNX;
*   the **dynamic-shape profile** -- min, opt and max, taken from the
    resolutions the chain actually asks for;
*   the **consumer matrix** -- for each resolution a real request would use,
    whether it falls inside the profile, so a shape the engine cannot serve
    is known at build time rather than at the first frame;
*   the identity of what was built: source ONNX sha256, TensorRT version, GPU
    name and compute capability. An engine is per-architecture, so an engine
    built on one card is not evidence about another and the manifest says
    which card it belongs to.

    PYTHONPATH=python python scripts/video_enhance/trt_engine_bench.py \\
        --card 1 --out /spinning/gpu-battery-results/trt
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from sglang.srt.video_enhance.engine_cache import sha256_file
from sglang.srt.video_enhance.frame_math import Resolution

#: The SR input sizes the chain actually asks for. The profile is built from
#: these rather than from round numbers, and the consumer matrix is checked
#: against them.
CONSUMER_RESOLUTIONS = ("960x540", "1280x720", "1920x1080")

#: Workspace ceiling for the builder. Generous but bounded: an unbounded
#: workspace on a card shared with an LLM tenant is how a build succeeds and
#: the next allocation fails.
WORKSPACE_BYTES = 4 << 30


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def build_engine(
    onnx_path: Path,
    *,
    profile: tuple[Resolution, Resolution, Resolution],
    workspace_bytes: int = WORKSPACE_BYTES,
) -> tuple[bytes, dict]:
    """Build a serialized engine and the manifest that describes it."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # Strongly typed: precision is the graph's, not the builder's. With no
    # FP16 builder flag in TensorRT 11 this is also the only way to say it.
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    payload = onnx_path.read_bytes()
    if not parser.parse(payload):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise SystemExit(
            f"TensorRT could not parse {onnx_path.name}:\n  " + "\n  ".join(errors)
        )

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)

    if network.num_inputs != 1:
        raise SystemExit(
            f"expected one network input, got {network.num_inputs}; this script "
            "builds the single-input SR graph"
        )
    input_name = network.get_input(0).name
    opt_profile = builder.create_optimization_profile()
    low, opt, high = profile
    opt_profile.set_shape(
        input_name,
        (1, 3, low.height, low.width),
        (1, 3, opt.height, opt.width),
        (1, 3, high.height, high.width),
    )
    config.add_optimization_profile(opt_profile)

    started = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    elapsed = time.perf_counter() - started
    if plan is None:
        raise SystemExit("TensorRT returned no engine; see the builder log above")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)

    # I/O signature read off the built engine, not assumed from the ONNX.
    bindings = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        bindings.append(
            {
                "name": name,
                "mode": str(engine.get_tensor_mode(name)).rsplit(".", 1)[-1],
                "dtype": str(engine.get_tensor_dtype(name)).rsplit(".", 1)[-1],
                "shape": list(engine.get_tensor_shape(name)),
            }
        )

    manifest = {
        "source_onnx": str(onnx_path),
        "source_sha256": sha256_file(onnx_path),
        "tensorrt_version": trt.__version__,
        "strongly_typed": True,
        "build_seconds": round(elapsed, 2),
        "workspace_bytes": workspace_bytes,
        "io_signature": bindings,
        "profile": {
            "input": input_name,
            "min": str(low),
            "opt": str(opt),
            "max": str(high),
        },
    }
    return bytes(plan), manifest


def consumer_matrix(profile: tuple[Resolution, Resolution, Resolution]) -> list[dict]:
    """For each resolution a request would use, can this engine serve it?

    Answered at build time. A shape outside the profile is not a slow path in
    TensorRT, it is a refusal at ``set_input_shape``, so discovering it at the
    first frame of a job means the job dies after admission.
    """
    low, _opt, high = profile
    rows = []
    for name in CONSUMER_RESOLUTIONS:
        res = Resolution.parse(name)
        inside = (
            low.width <= res.width <= high.width
            and low.height <= res.height <= high.height
        )
        rows.append({"resolution": name, "servable": inside})
    return rows


class TrtRunner:
    """One built engine, executed on torch tensors."""

    def __init__(self, plan: bytes, device: str) -> None:
        import tensorrt as trt

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(plan)
        self.context = self.engine.create_execution_context()
        self.device = device
        self.input_name = None
        self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name
        self._torch_dtype = {
            "HALF": "float16",
            "FLOAT": "float32",
        }.get(str(self.engine.get_tensor_dtype(self.input_name)).rsplit(".", 1)[-1])

    def run(self, tensor):
        """One inference. ``tensor`` is NCHW on the device."""
        import torch

        self.context.set_input_shape(self.input_name, tuple(tensor.shape))
        out_shape = tuple(self.context.get_tensor_shape(self.output_name))
        out_dtype = str(self.engine.get_tensor_dtype(self.output_name)).rsplit(".", 1)[
            -1
        ]
        out = torch.empty(
            out_shape,
            dtype=torch.float16 if out_dtype == "HALF" else torch.float32,
            device=tensor.device,
        )
        self.context.set_tensor_address(self.input_name, tensor.data_ptr())
        self.context.set_tensor_address(self.output_name, out.data_ptr())
        stream = torch.cuda.current_stream()
        self.context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        return out


def timeit(fn, *, iterations: int, warmup: int = 3) -> tuple[float, float]:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.mean(samples), statistics.pstdev(samples)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="#339-P4 direct TensorRT SR engine")
    parser.add_argument("--card", default="0", help="NVML index, for the record only")
    parser.add_argument("--model-dir", default="/spinning/llm_stuff/k3-models/sr")
    parser.add_argument("--fp16-onnx", default=None)
    parser.add_argument("--fp32-onnx", default=None)
    parser.add_argument("--out", default="/tmp/trt")
    parser.add_argument("--profile-min", default="960x540")
    parser.add_argument("--profile-opt", default="960x540")
    parser.add_argument("--profile-max", default="1920x1080")
    parser.add_argument("--bench-resolution", default="960x540")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--samples", type=int, default=3, help="parity samples")
    parser.add_argument("--skip-parity", action="store_true")
    args = parser.parse_args(argv)

    import torch

    model_dir = Path(args.model_dir)
    fp16_onnx = Path(
        args.fp16_onnx or model_dir / "realesr-general-wdn-x4v3_opset16_fp16.onnx"
    )
    fp32_onnx = Path(
        args.fp32_onnx or model_dir / "realesr-general-wdn-x4v3_opset16.onnx"
    )
    for path in (fp16_onnx, fp32_onnx):
        if not path.is_file():
            raise SystemExit(f"missing artifact: {path}")

    workdir = Path(args.out)
    workdir.mkdir(parents=True, exist_ok=True)

    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    _log(f"card {args.card} = {name}, sm{capability[0]}{capability[1]}")

    profile = (
        Resolution.parse(args.profile_min),
        Resolution.parse(args.profile_opt),
        Resolution.parse(args.profile_max),
    )
    report: dict = {
        "card": args.card,
        "device_name": name,
        # An engine is per-architecture. Recording the SM here is what stops
        # an engine built on one card being read as evidence about another.
        "compute_capability": f"sm{capability[0]}{capability[1]}",
        "torch": torch.__version__,
        "consumer_matrix": consumer_matrix(profile),
    }

    _log("building the fp16 engine (strongly typed)")
    plan, manifest = build_engine(fp16_onnx, profile=profile)
    engine_path = (
        workdir / f"sr_fp16_card{args.card}_sm{capability[0]}{capability[1]}.plan"
    )
    engine_path.write_bytes(plan)
    manifest["engine_path"] = str(engine_path)
    manifest["engine_bytes"] = len(plan)
    report["engine"] = manifest
    _log(f"built in {manifest['build_seconds']}s -> {engine_path.name}")
    print(json.dumps({"engine": manifest}, indent=2), flush=True)

    runner = TrtRunner(plan, "cuda:0")
    bench_res = Resolution.parse(args.bench_resolution)

    # -- parity against the fp32 ONNX on the CUDA provider -------------------
    if not args.skip_parity:
        from sglang.srt.video_enhance.backends import OnnxRuntimeBackend
        from sglang.srt.video_enhance.parity import grade as grade_pair

        reference = OnnxRuntimeBackend(
            fp32_onnx, provider="cuda", precision="fp32", device_id=0
        )
        rows = []
        try:
            generator = torch.Generator().manual_seed(0x5EED)
            for sample in range(args.samples):
                # Sampled on the CPU: two architectures do not agree on a
                # device RNG, and these numbers are compared across cards.
                host = torch.rand(
                    (1, 3, bench_res.height, bench_res.width),
                    generator=generator,
                    dtype=torch.float32,
                )
                ref = reference.run(host.to("cuda:0"))
                cand = runner.run(host.to("cuda:0").half())
                verdict = grade_pair(
                    cand.float(), ref.float(), note=f"TRT fp16 @ {bench_res}"
                )
                rows.append(
                    {
                        "resolution": str(bench_res),
                        "sample": sample,
                        "candidate_provider": "tensorrt-direct",
                        "reference_provider": reference.info.effective_provider,
                        **verdict.as_manifest(),
                    }
                )
        finally:
            reference.close()
        report["parity"] = rows
        print(json.dumps({"parity": rows}, indent=2), flush=True)

    # -- throughput, noise floor first ---------------------------------------
    host = torch.rand((1, 3, bench_res.height, bench_res.width), dtype=torch.float32)
    device_input = host.to("cuda:0").half()

    def once() -> None:
        runner.run(device_input)

    a_mean, a_std = timeit(once, iterations=args.iterations)
    b_mean, b_std = timeit(once, iterations=args.iterations)
    floor_pct = abs(a_mean - b_mean) / a_mean * 100.0
    report["bench"] = {
        "resolution": str(bench_res),
        "iterations": args.iterations,
        "ms_a": round(a_mean, 4),
        "ms_b": round(b_mean, 4),
        "stdev_a": round(a_std, 4),
        "stdev_b": round(b_std, 4),
        "ms": round(min(a_mean, b_mean), 4),
        "noise_floor_pct": round(floor_pct, 3),
    }
    _log(
        f"fp16 TRT SR @ {bench_res}: {min(a_mean, b_mean):.3f} ms "
        f"(A/B floor {floor_pct:.2f}%)"
    )

    out_json = workdir / f"trt_card{args.card}.json"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({"bench": report["bench"]}, indent=2))
    unservable = [r for r in report["consumer_matrix"] if not r["servable"]]
    if unservable:
        print(
            "consumer matrix: this engine cannot serve "
            f"{[r['resolution'] for r in unservable]} -- outside the profile",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
