#!/usr/bin/env python3
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Derive an fp16 SR artifact from the pinned fp32 one, and grade it.

Ported from ``efschu/vs-pipeline``'s ``build/build_esrgan_rtx.py``, which
solves exactly this for exactly this model. Two things are taken from it and
one is deliberately not:

*   **Taken: convert the initializers, then the graph I/O.** Casting the
    weights to fp16 and declaring the input and output fp16 makes the graph
    genuinely half precision rather than an fp32 graph a builder is *allowed*
    to run in half precision. The difference matters here for a reason beyond
    speed: the chain works in fp16, and an fp32-input graph forces a
    per-frame cast of every frame on its way into the stage.
*   **Taken: the shape profile from real consumer usage** -- min, opt and max
    from the resolutions the chain actually asks for, not round numbers.
*   **Not taken: TensorRT-RTX as the builder.** ``vs-pipeline`` builds with
    ``tensorrt_rtx`` and ships ``.engine`` files. This fork's SR path goes
    through ONNX Runtime's TensorRT execution provider (TASK_333 §2.1), which
    builds and caches its own engine from the ONNX, so what is produced here
    is an ONNX and the engine stays the EP's business. Pulling in a second
    TensorRT distribution to produce an artifact the existing one can produce
    would be a dependency for no capability.

The other half of ``build_rife_rtx_fp16.py``'s technique -- inserting Cast
nodes at the I/O so the *compute* stays fp32 while the interface is fp16 --
is a different trade and is exposed as ``--io-cast-only``. It is the right
choice when a network is numerically fragile and the parity gate rejects a
full fp16 conversion; it buys the interface bandwidth without the accuracy
risk. Which one this model needs is a measurement, and ``--grade`` is how it
is taken.

Needs ``onnx`` at the graph-surgery step, which the serving path does not.
It is a build-time tool, so ``onnx`` is deliberately not added to the
runtime requirements; install it wherever this is run, e.g.

    pip install --target /tmp/onnxtools onnx   # then drop its bundled numpy
    PYTHONPATH=python:/tmp/onnxtools python scripts/video_enhance/export_sr_fp16.py --grade
"""

from __future__ import annotations

import argparse

import json
import sys
import time
from pathlib import Path

from sglang.srt.video_enhance.engine_cache import sha256_file
from sglang.srt.video_enhance.frame_math import Resolution

FP16_SUFFIX = "_fp16.onnx"
#: Sidecar next to the derived artifact. A derived file has no upstream hash
#: to pin, so its identity is recorded here at the moment it is produced --
#: source hash, tool, method -- and the loader verifies against that rather
#: than trusting the filename.
PROVENANCE_SUFFIX = ".provenance.json"


def schema_pinned_float_inputs(model) -> set:
    """Initializer names the ONNX spec pins to fp32, whatever the compute is.

    Not every float input of every operator is part of its precision. Some are
    fixed by the schema to ``tensor(float)`` rather than to a type variable
    that follows the data: ``Resize`` declares ``roi`` and ``scales`` that way,
    because they are geometry, not pixels. Casting them along with the weights
    produces a model that ``onnx.checker.check_model`` accepts and ONNX Runtime
    refuses to load:

        Type 'tensor(float16)' of input parameter (onnx::Resize_275) of
        operator (Resize) in node (Resize_68) is invalid.

    That is what the first version of this converter did, and because the only
    validation at export time was the checker, the broken artifact was written,
    hashed and given a provenance sidecar. It sat on disk looking finished
    until something tried to load it.

    The fix is general rather than a list of op names: ask the operator's own
    schema which of its inputs are type variables and which are pinned, and
    leave the pinned ones alone. A future model that uses some other op with a
    fixed float input is then covered without anyone remembering to add it.
    """
    from onnx import defs

    opsets = {imp.domain: imp.version for imp in model.opset_import}
    pinned: set = set()
    for node in model.graph.node:
        version = opsets.get(node.domain, opsets.get("", 0))
        try:
            schema = defs.get_schema(node.op_type, version, node.domain)
        except Exception:  # noqa: BLE001 - an unknown op pins nothing
            continue
        variables = {tc.type_param_str for tc in schema.type_constraints}
        declared = list(schema.inputs)
        if not declared:
            continue
        for index, name in enumerate(node.input):
            if not name:
                continue
            # A variadic tail repeats the last declared input's constraint.
            spec = declared[index] if index < len(declared) else declared[-1]
            if spec.type_str not in variables and "float16" not in spec.type_str:
                pinned.add(name)
    return pinned


def convert_initializers_and_io(model):
    """Weights and graph I/O to fp16. The vs-pipeline conversion."""
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    pinned = schema_pinned_float_inputs(model)
    converted = 0
    for init in model.graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            continue
        if init.name in pinned:
            # Geometry, not pixels. See schema_pinned_float_inputs.
            continue
        array = numpy_helper.to_array(init).astype(np.float16)
        init.CopyFrom(
            helper.make_tensor(
                name=init.name,
                data_type=TensorProto.FLOAT16,
                dims=init.dims,
                vals=array.tobytes(),
                raw=True,
            )
        )
        converted += 1
    for tensor in list(model.graph.input) + list(model.graph.output):
        if tensor.type.tensor_type.elem_type == TensorProto.FLOAT:
            tensor.type.tensor_type.elem_type = TensorProto.FLOAT16
    return model, converted


def insert_io_casts(model):
    """fp16 interface over an fp32 graph, via Cast nodes at the boundary.

    The ``build_rife_rtx_fp16.py`` technique. The graph still computes in
    fp32, so accuracy is unchanged by construction, and only the interface
    changes precision. Kept as the fallback for a network the full conversion
    cannot pass the parity gate with.
    """
    from onnx import TensorProto, helper

    graph = model.graph
    inserted: list[str] = []

    for i, inp in enumerate(list(graph.input)):
        original = inp.name
        cast_out = f"__cast_in_{i}_to_fp32"
        name = f"Cast_in_{i}"
        inp.type.tensor_type.elem_type = TensorProto.FLOAT16
        graph.node.insert(
            0,
            helper.make_node(
                "Cast",
                inputs=[original],
                outputs=[cast_out],
                to=TensorProto.FLOAT,
                name=name,
            ),
        )
        inserted.append(name)
        for node in graph.node:
            if node.name in inserted:
                continue
            for j, ref in enumerate(node.input):
                if ref == original:
                    node.input[j] = cast_out

    for i, out in enumerate(list(graph.output)):
        original = out.name
        cast_in = f"__cast_out_in_{i}_fp32"
        name = f"Cast_out_{i}"
        out.type.tensor_type.elem_type = TensorProto.FLOAT16
        for node in graph.node:
            if node.name in inserted:
                continue
            for j, ref in enumerate(node.output):
                if ref == original:
                    node.output[j] = cast_in
        graph.node.append(
            helper.make_node(
                "Cast",
                inputs=[cast_in],
                outputs=[original],
                to=TensorProto.FLOAT16,
                name=name,
            )
        )
        inserted.append(name)

    if model.opset_import[0].version < 13:
        model.opset_import[0].version = 13
    return model, len(inserted)


def _assert_loadable(onnx_path: Path) -> None:
    """Open the artifact once on CPU. Raises if the runtime refuses it.

    Deliberately the CPU provider: this asks whether the *graph* is valid, not
    whether any accelerator is present, so it is the same check on a build box
    with no card as on this rig.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    try:
        ort.InferenceSession(
            str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with what to do about it
        onnx_path.unlink(missing_ok=True)
        raise SystemExit(
            f"the exported artifact {onnx_path.name} is not loadable and has been "
            f"removed rather than left on disk looking finished: {exc}\n"
            "If this is a type-constraint rejection, an initializer that the "
            "operator's schema pins to fp32 was converted; see "
            "schema_pinned_float_inputs. --io-cast-only is the fallback that "
            "keeps every computation in fp32."
        ) from exc


def export(source_onnx: Path, out_path: Path, *, io_cast_only: bool) -> dict:
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - build-time tool only
        raise SystemExit(
            "the fp16 export needs the 'onnx' package for the graph surgery. It "
            "is a build-time dependency and is deliberately not in the serving "
            "requirements; install it into a scratch target and put that on "
            "PYTHONPATH."
        ) from exc

    model = onnx.load(str(source_onnx))
    if io_cast_only:
        model, changed = insert_io_casts(model)
        method = "io_cast"
    else:
        model, changed = convert_initializers_and_io(model)
        method = "initializers_and_io"
    onnx.checker.check_model(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))

    # The checker is not the gate. It accepted the Resize/scales violation
    # that made the first fp16 artifact unloadable, so the artifact was
    # written, hashed and given a provenance sidecar that said it had been
    # built -- and nothing found out until a grading run tried to open it
    # weeks later. Loading it once, on CPU, costs a second and makes "built"
    # mean "loadable". An artifact that cannot be loaded must not leave a
    # sidecar behind claiming otherwise, so this runs before the manifest.
    _assert_loadable(out_path)

    manifest = {
        "derived_from": str(source_onnx),
        "derived_from_sha256": sha256_file(source_onnx),
        "sha256": sha256_file(out_path),
        "bytes": out_path.stat().st_size,
        "method": method,
        "changed_tensors": changed,
        "exported_at": time.time(),
        "tool": "scripts/video_enhance/export_sr_fp16.py",
        "ported_from": "efschu/vs-pipeline build/build_esrgan_rtx.py",
    }
    out_path.with_suffix(out_path.suffix + PROVENANCE_SUFFIX).write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    return manifest


def grade(
    source_onnx: Path,
    fp16_onnx: Path,
    *,
    resolutions: list[Resolution],
    samples: int,
    device_id: int,
) -> list[dict]:
    """Grade the fp16 engine against the fp32 reference on real tensors.

    The reference is the fp32 ONNX on the CUDA provider, which is what
    ``parity.py`` defines the gate against. Inputs are sampled on the CPU and
    moved to the device rather than generated with a device RNG: two
    architectures do not agree on ``torch.randn``, so a device-sampled input
    would make a cross-card comparison of these numbers meaningless.
    """
    import torch

    from sglang.srt.video_enhance.backends import OnnxRuntimeBackend
    from sglang.srt.video_enhance.engine_cache import ShapeTriplet
    from sglang.srt.video_enhance.parity import grade as grade_pair

    results: list[dict] = []
    for resolution in resolutions:
        generator = torch.Generator().manual_seed(0x5EED)
        reference_backend = OnnxRuntimeBackend(
            source_onnx, provider="cuda", precision="fp32", device_id=device_id
        )
        candidate_backend = OnnxRuntimeBackend(
            fp16_onnx,
            provider="tensorrt",
            precision="fp16",
            device_id=device_id,
            shapes=ShapeTriplet.static(resolution.width, resolution.height),
        )
        try:
            for sample in range(samples):
                host = torch.rand(
                    (1, 3, resolution.height, resolution.width),
                    generator=generator,
                    dtype=torch.float32,
                )
                device = f"cuda:{device_id}"
                reference = reference_backend.run(host.to(device))
                candidate = candidate_backend.run(host.to(device).half())
                verdict = grade_pair(
                    candidate.float(),
                    reference.float(),
                    note=f"{fp16_onnx.name} @ {resolution}",
                )
                results.append(
                    {
                        "resolution": str(resolution),
                        "sample": sample,
                        # What the numbers describe. A TensorRT session falls
                        # back to CUDA when the libraries are missing, so a
                        # record that only carried the *requested* provider
                        # could claim a TensorRT result taken on a host where
                        # libnvinfer was never installed.
                        "candidate_provider": candidate_backend.info.effective_provider,
                        "candidate_provider_requested": candidate_backend.info.provider,
                        "candidate_provider_fell_back": (
                            candidate_backend.info.provider_fell_back
                        ),
                        "reference_provider": (
                            reference_backend.info.effective_provider
                        ),
                        **verdict.as_manifest(),
                    }
                )
        finally:
            reference_backend.close()
            candidate_backend.close()
    return results


def main(argv: list[str] | None = None) -> int:
    from sglang.srt.video_enhance.sr import REALESR_GENERAL_WDN_X4V3, fetch_model

    parser = argparse.ArgumentParser(description="#339 fp16 SR artifact + parity")
    parser.add_argument("--model-dir", default="/spinning/llm_stuff/k3-models/sr")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--io-cast-only",
        action="store_true",
        help="fp16 interface over an fp32 graph, the RIFE-script technique",
    )
    parser.add_argument("--grade", action="store_true")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--resolutions",
        default="960x540,1280x720",
        help="input resolutions to grade at; the chain's real SR input sizes",
    )
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)

    model_dir = Path(args.model_dir)
    source = fetch_model(REALESR_GENERAL_WDN_X4V3, model_dir)
    out_path = Path(
        args.out
        or model_dir
        / (source.stem + ("_iocast" if args.io_cast_only else "") + FP16_SUFFIX)
    )

    manifest = export(source, out_path, io_cast_only=args.io_cast_only)
    report: dict = {"artifact": manifest}
    print(json.dumps({"artifact": manifest}, indent=2), flush=True)

    if args.grade:
        resolutions = [Resolution.parse(r) for r in args.resolutions.split(",")]
        report["parity"] = grade(
            source,
            out_path,
            resolutions=resolutions,
            samples=args.samples,
            device_id=args.device_id,
        )
        print(json.dumps({"parity": report["parity"]}, indent=2))
        failed = [r for r in report["parity"] if not r["passed"]]
        if args.report:
            Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True))
        if failed:
            print(
                f"PARITY GATE FAILED for {len(failed)} of {len(report['parity'])} "
                "samples; --io-cast-only keeps the compute in fp32 and is the "
                "fallback for a model the full conversion cannot carry",
                file=sys.stderr,
            )
            return 1
        print("PARITY GATE PASSED")
    elif args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
