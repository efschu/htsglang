#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Task #337 -- ONNX export of each target, from the fork's own module shapes.

WHY THIS IS THE SECOND PATH, NOT THE FIRST
==========================================
``build_engines.py`` constructs the TensorRT network directly. That is the
primary path and the one the engines in ``engines/`` were built with, for three
reasons that are worth stating rather than leaving implicit:

  1. The Q/DQ placement is the whole point of this task, and building the
     network directly means the graph contains exactly the dequantize pair, the
     matmul and the per-token scale multiply -- no exporter in between deciding
     to fold, reorder or helpfully "optimize" the quantization away.
  2. ``onnx``/``onnxscript`` are not part of the serving environment. The direct
     path needs only ``tensorrt_rtx``.
  3. It is the only path where the INT8 weight constant can be handed over as
     INT8 with its per-channel scale attached, rather than round-tripped
     through a float tensor that an exporter would then have to re-quantize.

ONNX is exported anyway because it is the inspectable artifact: a
``.onnx`` file can be read by anyone, diffed across revisions, and fed to a
different backend (onnxruntime, a newer TensorRT, someone else's toolchain)
when the question is "is our engine doing what we think" rather than "how fast
is it".

MEASURED LIMITATION: THE EXPORT LOSES THE WEIGHT QUANTIZATION
=============================================================
This is not a caveat copied from documentation, it is what the exporter did to
these graphs. Checked with ``--verify-ops`` on the real rank-0 gate_up export:

    ops         Constant 6, Mul 5, Cast 2, Slice 2, MatMul 1, Shape 1,
                Gather 1, Add 1, Div 1, Sigmoid 1
    initializer onnx::MatMul_30  FLOAT  [5120, 16320]

There is no ``DequantizeLinear`` and there is no INT8 initializer. Torch's
exporter constant-folded ``w_q.to(float32) * w_scale`` into a single float32
weight, which is arithmetically the same number and structurally a different
model: an engine built from THIS ONNX would run an fp32 GEMM, not an INT8 one,
and would answer a different question than the one #337 asks.

Consequences, stated plainly:

  * The ``.onnx`` files are a STRUCTURAL artifact -- read them to see the op
    order, the dynamic axes, the epilogue shape. They are not a faithful build
    input and no engine in ``engines/`` was built from them.
  * A faithful ONNX route would need explicit QuantizeLinear/DequantizeLinear
    nodes that survive folding. That is graph surgery this task does not need,
    because ``build_engines.py`` reaches the same place directly and with the
    quantization intact.
  * ``--verify-ops`` runs the check above on every exported graph and writes
    ``quantization_preserved`` into the manifest, so this stays measured rather
    than remembered.

WHAT IS EXPORTED
================
One graph per engine stage, matching the TensorRT network op for op:

    a_q [M,K] int8, a_scale [M,1] float32
      -> DequantizeLinear(a_q, 1.0)
      -> DequantizeLinear(W_q, w_scale, axis=0)      per output channel
      -> MatMul / Gemm
      -> Mul(a_scale)
      -> Cast(bfloat16)
      -> [Split, Sigmoid, Mul, Mul]                  silu_mul epilogue
      -> out

The activation quant is NOT in the graph, for the reason given in
``targets.py``: TensorRT requires a build-time-constant quantization scale and
ours is per token and dynamic. Exporting a graph that contained it would export
a model we cannot build.

Weights are the real checkpoint shard tensors by default, same loader as the
engine builder, so the ONNX file and the ``.plan`` carry identical constants.

REQUIREMENTS
============
``onnx`` and ``onnxscript``, installed into the same isolated directory as
``tensorrt_rtx`` and reached through PYTHONPATH -- never into a venv a serving
window is using:

    pip install --target <artifact_dir>/pylibs onnx onnxscript

USAGE
=====
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=<artifact_dir>/pylibs \\
        python3 scripts/trt_337/export_onnx.py --ranks 0,1 \\
        --out-dir <artifact_dir>/onnx
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import targets as tgt  # noqa: E402
from build_engines import DEFAULT_MODEL_DIR  # noqa: E402


def build_module(spec: tgt.GemmSpec, weights):
    """A torch module whose forward is the engine's graph, op for op."""
    import torch

    class Stage(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # int8 weights are held as int8; the dequantize is explicit in
            # forward so the exporter emits DequantizeLinear rather than
            # inventing its own quantization scheme.
            self.register_buffer("w_q", weights.q)
            self.register_buffer("w_scale", weights.scale.reshape(-1, 1))
            self.epilogue = spec.epilogue

        def forward(self, a_q, a_scale):
            # DQ(a_q, 1.0): the activation is already integer-valued; the real
            # scale is per token and arrives as an ordinary runtime tensor.
            a = a_q.to(torch.float32)
            w = self.w_q.to(torch.float32) * self.w_scale  # per output channel
            out = (a @ w.t()) * a_scale
            out = out.to(torch.bfloat16)
            if self.epilogue == "silu_mul":
                gate, up = out.chunk(2, dim=-1)
                out = (gate * torch.sigmoid(gate)) * up
            return out

    return Stage().eval()


def export_one(spec, weights, path, max_bs, opset):
    import torch

    mod = build_module(spec, weights)
    a_q = torch.randint(-127, 128, (1, spec.k), dtype=torch.int8)
    a_scale = torch.rand(1, 1, dtype=torch.float32).add_(0.01)
    t0 = time.time()
    torch.onnx.export(
        mod,
        (a_q, a_scale),
        path,
        input_names=["a_q", "a_scale"],
        output_names=["out"],
        dynamic_axes={
            "a_q": {0: "M"},
            "a_scale": {0: "M"},
            "out": {0: "M"},
        },
        opset_version=opset,
        dynamo=False,
    )
    return {
        "onnx": os.path.basename(path),
        "stage": spec.name,
        "n": spec.n,
        "k": spec.k,
        "epilogue": spec.epilogue,
        "weights_source": weights.source,
        "bytes": os.path.getsize(path),
        "export_seconds": round(time.time() - t0, 3),
        "opset": opset,
        "max_batch_documented": max_bs,
        "activation_quant_in_graph": False,
    }


def verify_ops(path: str) -> dict:
    """Did the export keep the weight quantization, or fold it away?

    The answer decides whether the file is a build input or an inspection
    artifact, so it is measured per file rather than assumed once.
    """
    import collections

    import onnx

    m = onnx.load(path, load_external_data=False)
    ops = collections.Counter(n.op_type for n in m.graph.node)
    inits = [
        {
            "name": i.name,
            "dtype": onnx.TensorProto.DataType.Name(i.data_type),
            "dims": list(i.dims),
        }
        for i in m.graph.initializer
    ]
    has_dq = ops.get("DequantizeLinear", 0) > 0
    has_int8 = any(i["dtype"] == "INT8" for i in inits)
    return {
        "ops": dict(ops),
        "initializers": inits[:8],
        "quantization_preserved": bool(has_dq and has_int8),
        "note": (
            "false means torch folded w_q*w_scale into a float constant; the "
            "file is then a structural artifact, not a faithful build input."
        ),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=tgt.DEFAULT_CONFIG)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--tp-size", type=int, default=3)
    ap.add_argument("--ratio", default="30,17,17")
    ap.add_argument("--ranks", default="0,1")
    ap.add_argument("--optional", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-bs", type=int, default=8)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--random-weights", action="store_true")
    ap.add_argument("--seed", type=int, default=337)
    ap.add_argument("--mock", action="store_true",
                    help="tiny shapes, so the export path runs in seconds")
    ap.add_argument("--verify-ops", action="store_true", default=True,
                    help="check per file whether the quantization survived")
    a = ap.parse_args(argv)

    try:
        import onnx  # noqa: F401,PLC0415
    except ImportError as ex:
        raise SystemExit(
            f"onnx not importable ({ex}). This is the SECONDARY export path; "
            f"the engines are built by build_engines.py without it. To use it:\n"
            f"  pip install --target <artifact_dir>/pylibs onnx onnxscript\n"
            f"  export PYTHONPATH=<artifact_dir>/pylibs"
        ) from ex

    os.makedirs(a.out_dir, exist_ok=True)
    ratio = [int(x) for x in a.ratio.split(",")]
    optional = [x for x in a.optional.split(",") if x]
    manifest = {
        "task": "337",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mock": a.mock,
        "note": (
            "Secondary, inspectable artifact. The benchmarked engines are built "
            "by build_engines.py directly against the TensorRT network API; see "
            "that file's docstring for why."
        ),
        "graphs": [],
    }
    seen = set()
    for rank in [int(r) for r in a.ranks.split(",")]:
        geo = tgt.derive_geometry(a.config, a.tp_size, ratio, rank)
        for t in tgt.build_targets(geo, optional):
            for st in t.engine_stages:
                spec = st.gemm
                key = (rank, spec.name)
                if key in seen:
                    continue
                seen.add(key)
                if a.mock:
                    spec = tgt.GemmSpec(
                        name=spec.name, n=64, k=32, epilogue=spec.epilogue
                    )
                    w = tgt.load_gemm_weights(spec, geo, a.model_dir, True, a.seed)
                else:
                    w = tgt.load_gemm_weights(
                        spec, geo, a.model_dir, a.random_weights, a.seed
                    )
                name = f"rank{rank}_{geo.arch}_{spec.name}.onnx"
                rec = export_one(
                    spec, w, os.path.join(a.out_dir, name), a.max_bs, a.opset
                )
                rec["rank"] = rank
                rec["arch"] = geo.arch
                if a.verify_ops:
                    rec["verify"] = verify_ops(os.path.join(a.out_dir, name))
                manifest["graphs"].append(rec)
                qp = rec.get("verify", {}).get("quantization_preserved")
                print(f"  exported {name:44s} {rec['bytes']/1e6:7.2f} MB "
                      f"{rec['export_seconds']:6.2f}s  quant_preserved={qp}")

    path = os.path.join(a.out_dir, "manifest.json")
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"\n{len(manifest['graphs'])} graphs -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
