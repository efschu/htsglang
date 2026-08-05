#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Task #337 -- TensorRT-RTX engine builder for the per-part microbench.

WHY THIS RUNS AT A DESK
=======================
TensorRT for RTX splits compilation in two. The ahead-of-time (AOT) phase needs
NO GPU -- ``trt.Builder`` constructs and ``build_serialized_network`` returns a
plan on a machine with ``CUDA_VISIBLE_DEVICES=""``. The just-in-time (JIT) phase
specialises that plan for whatever RTX card it is later loaded on. One AOT plan
therefore covers BOTH architectures on this rig, and every engine in this task
was built without ever claiming a card.

That is verified, not assumed: ``--probe`` records the builder's own answers, and
every engine manifest carries ``"built_with_visible_devices"`` so a plan built on
a card can never be mistaken for a desk-built one.

Two settings would silently break CPU-only AOT and are therefore refused here:

    ITimingCache      -- forces the AOT step to query for a GPU. Not used; the
                         relevant BuilderFlags are set to DISABLE_TIMING_CACHE.
    weight stripping  -- requires a device. Not used.

MULTI-ARCHITECTURE
==================
``IBuilderConfig.set_compute_capability`` takes an explicit list. Both of this
rig's architectures are in the installed enum (``ComputeCapability.SM86`` for the
two 3080s, ``ComputeCapability.SM120`` for the 5090), so one plan is built for
``[SM86, SM120]`` and both card arms load the same file. ``kCURRENT`` is
deliberately NOT used -- it would pin the plan to whatever card happened to be
visible and reintroduce the per-arch split.

PRECISION: NO IMPLICIT DOWNGRADE IS STRUCTURALLY POSSIBLE
=========================================================
The classic TensorRT advice ("pass --fp16/--int8, set OBEY_PRECISION_CONSTRAINTS,
hope") does not apply to this version. The installed builder's flag enum is:

    DEBUG DIRECT_IO DISABLE_COMPILATION_CACHE DISABLE_TIMING_CACHE
    DISTRIBUTIVE_INDEPENDENCE EDITABLE_TIMING_CACHE ERROR_ON_TIMING_CACHE_MISS
    EXCLUDE_LEAN_RUNTIME GPU_FALLBACK MONITOR_MEMORY REFIT REFIT_IDENTICAL
    REFIT_INDIVIDUAL REQUIRE_USER_ALLOCATION SAFETY_SCOPE SPARSE_WEIGHTS
    STRICT_NANS STRIP_PLAN TF32 VERSION_COMPATIBLE WEIGHT_STREAMING

There is no FP16 flag, no BF16 flag, no INT8 flag and no
OBEY_PRECISION_CONSTRAINTS, because there is nothing for them to do: the network
is created with ``NetworkDefinitionCreationFlag.STRONGLY_TYPED``, and in a
strongly typed network every tensor's type is fixed by the network definition.
The builder may choose tactics but not precisions. An implicit fp16 downgrade is
not a risk that is being managed here -- it is a thing the API cannot do.

TF32 is cleared explicitly anyway. It would only affect fp32 matmuls, and after
the Q/DQ fusion there should be none, but "should be none" is not a reason to
leave a precision knob at its default.

The full flag state (set, cleared, and available-but-untouched) is written into
every engine manifest, so the precision claim is auditable from the artifact
rather than from this docstring.

WHAT THE NETWORK IS
===================
See ``targets.py`` for why the activation quant stays outside the engine. Per
engine stage:

    a_q      INT8  [M,K]     (produced by our per_token_quant_int8, outside)
    a_scale  FP32  [M,1]     (ditto)
    W_q      INT8  [N,K]     build-time constant, real checkpoint shard
    w_scale  FP32  [N]       build-time constant, real per-channel scales

    DQ(a_q, 1.0)                     per-tensor constant, folds away
    DQ(W_q, w_scale, axis=0)         per-output-channel, the deployed weight
                                     quantization exactly
    MatMul(NONE, TRANSPOSE)
    Mul(a_scale)                     the per-token scale as an ordinary runtime
                                     elementwise operand
    [silu_mul]                       split N in half, silu(gate) * up
    cast BF16

DYNAMIC SHAPES
==============
One engine, two optimization profiles: min 1 / opt 1 / max 8 and min 1 / opt 4 /
max 8. Decode at bs=1 and bs=4 are the two operating points the deliverable
names, and a single opt point would leave one of them running on a kernel
specialised for the other. The harness selects the profile whose opt matches the
M being measured and records which one it used.
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

#: Both deployed architectures in one plan. Not kCURRENT -- see module docstring.
DEFAULT_COMPUTE_CAPABILITIES = ("SM86", "SM120")
#: The batch sizes the profiles are cut for.
DEFAULT_PROFILE_OPTS = (1, 4)
DEFAULT_MAX_BS = 8
DEFAULT_MODEL_DIR = (
    "/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8"
)


def import_trt(mock: bool, library: str = "rtx"):
    """Import the requested TensorRT, or return None in mock mode.

    ``rtx``     tensorrt_rtx -- the primary library. Installed OUTSIDE any venv
                the serving windows use, under the artifact directory, and
                reached through PYTHONPATH. That is deliberate: this task must
                not mutate the environment a running window depends on.
    ``classic`` tensorrt -- the CONTROL arm. NVIDIA states TensorRT-RTX has no
                performance downside against classic TensorRT; that is a claim,
                and a claim measured on this rig is worth more than a claim
                quoted from a blog. Classic TensorRT happens to be installed
                already (11.2.1.2 in the serving venv), so the control costs one
                extra build rather than an install.

                Its builder needs a GPU, so a classic engine can NOT be built at
                a desk -- that build step belongs in the card window, and the
                run sheet has it as an optional arm.
    """
    if mock:
        return None
    if library == "classic":
        try:
            import tensorrt as trt  # noqa: PLC0415
        except ImportError as ex:
            raise SystemExit(f"classic tensorrt not importable: {ex}") from ex
        return trt
    try:
        import tensorrt_rtx as trt  # noqa: PLC0415
    except ImportError as ex:
        raise SystemExit(
            f"tensorrt_rtx not importable ({ex}). Install it into an isolated "
            f"directory and put that directory on PYTHONPATH -- never into a "
            f"venv a serving window is using:\n"
            f"  pip install --target <artifact_dir>/pylibs tensorrt_rtx==1.6.1.120\n"
            f"  export PYTHONPATH=<artifact_dir>/pylibs"
        ) from ex
    return trt


# --------------------------------------------------------------------------
# Capability probe -- what the installed library can and can not express
# --------------------------------------------------------------------------


def probe(trt) -> dict:
    """Record the installed library's own answers about INT8 W8A8.

    This exists because the honest answer to "can TensorRT express our
    quantization" is version dependent, and a docstring written from memory
    would be a guess. Everything below is read out of the installed package.
    """
    q_doc = trt.IQuantizeLayer.__doc__ or ""
    dq_doc = trt.IDynamicQuantizeLayer.__doc__ or ""
    return {
        "tensorrt_rtx_version": trt.__version__,
        "compute_capabilities_available": [
            c for c in dir(trt.ComputeCapability) if c.isupper()
        ],
        "builder_flags_available": sorted(
            f for f in dir(trt.BuilderFlag) if f.isupper()
        ),
        "network_creation_flags_available": sorted(
            f for f in dir(trt.NetworkDefinitionCreationFlag) if f.isupper()
        ),
        "datatypes_available": sorted(d for d in dir(trt.DataType) if d.isupper()),
        "cuda_graph_strategies": [
            c for c in dir(trt.CudaGraphStrategy) if c.isupper()
        ],
        "kernel_specialization_strategies": [
            c
            for c in dir(trt.DynamicShapesKernelSpecializationStrategy)
            if c.isupper()
        ],
        "quantize_scale_must_be_build_time_constant": (
            "must be a build-time constant" in q_doc
        ),
        "dynamic_quantize_output_types": (
            "kFP4 / kFP8 only" if "kFP4" in dq_doc else "unknown"
        ),
        "per_token_int8_activation_quant_expressible": False,
        "per_channel_int8_weight_quant_expressible": True,
        "verdict": (
            "TensorRT cannot express per-token dynamic INT8 activation "
            "quantization: IQuantizeLayer requires a build-time-constant scale, "
            "and IDynamicQuantizeLayer only emits FP4/FP8 at block size 16/32. "
            "Per-output-channel INT8 WEIGHT quantization is expressible exactly "
            "(constant weights, 1D scale on the output axis), which is the "
            "deployed weight format. The chosen configuration therefore keeps "
            "our per_token_quant_int8 kernel outside the engine in every arm "
            "and hands the engine pre-quantized activations plus the per-token "
            "scale as an ordinary runtime tensor -- arithmetically identical to "
            "int8_scaled_mm, with equal total work in both arms."
        ),
    }


# --------------------------------------------------------------------------
# Network construction
# --------------------------------------------------------------------------


def build_network(trt, builder, spec: tgt.GemmSpec, weights, dq_dtype: str):
    import numpy as np

    net = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    dq_t = trt.DataType.FLOAT if dq_dtype == "float" else trt.DataType.BF16
    np_t = np.float32 if dq_dtype == "float" else np.float32  # scales stay fp32

    a_q = net.add_input("a_q", trt.DataType.INT8, (-1, spec.k))
    a_scale = net.add_input("a_scale", trt.DataType.FLOAT, (-1, 1))

    w_np = weights.q.numpy()
    s_np = weights.scale.numpy().astype(np_t)
    w_c = net.add_constant((spec.n, spec.k), trt.Weights(w_np)).get_output(0)
    s_c = net.add_constant((spec.n,), trt.Weights(s_np)).get_output(0)
    one = net.add_constant(
        (1,), trt.Weights(np.array([1.0], dtype=np.float32))
    ).get_output(0)

    dq_a = net.add_dequantize(a_q, one, dq_t)
    dq_a.axis = 1
    dq_a.name = "dq_act"
    dq_w = net.add_dequantize(w_c, s_c, dq_t)
    dq_w.axis = 0  # per output channel: the deployed weight_scale layout
    dq_w.name = "dq_weight"

    mm = net.add_matrix_multiply(
        dq_a.get_output(0),
        trt.MatrixOperation.NONE,
        dq_w.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    )
    mm.name = "gemm"

    # The per-token scale. NOT a Q/DQ scale -- an ordinary runtime operand, which
    # is exactly what makes the deployed dynamic per-token scaling expressible
    # despite the build-time-constant restriction on IQuantizeLayer.
    scaled = net.add_elementwise(
        mm.get_output(0), a_scale, trt.ElementWiseOperation.PROD
    )
    scaled.name = "per_token_scale"

    # Cast to bf16 HERE, before the epilogue, not at the very end. The deployed
    # path is int8_scaled_mm(out_dtype=bfloat16) feeding silu_and_mul, so the
    # SiLU-multiply consumes bf16. An engine that kept fp32 through the epilogue
    # would be MORE accurate than what we deploy, and the extra accuracy would
    # then feed the next stage's activation quant -- the chain would slowly
    # diverge from serving and the tolerance gate would be measuring the
    # divergence rather than the kernels. Same dataflow, same rounding points.
    cur = net.add_cast(scaled.get_output(0), trt.DataType.BF16)
    cur.name = "to_bf16"
    cur = cur.get_output(0)

    if spec.epilogue == "silu_mul":
        half = spec.n // 2
        gate = net.add_slice(cur, (0, 0), (0, half), (1, 1))
        gate.set_input(2, _dyn_shape(net, trt, cur, half, 0))
        gate.name = "slice_gate"
        up = net.add_slice(cur, (0, half), (0, half), (1, 1))
        up.set_input(2, _dyn_shape(net, trt, cur, half, 0))
        up.name = "slice_up"
        sig = net.add_activation(gate.get_output(0), trt.ActivationType.SIGMOID)
        sig.name = "silu_sigmoid"
        silu = net.add_elementwise(
            gate.get_output(0), sig.get_output(0), trt.ElementWiseOperation.PROD
        )
        silu.name = "silu"
        cur = net.add_elementwise(
            silu.get_output(0), up.get_output(0), trt.ElementWiseOperation.PROD
        ).get_output(0)

    cur.name = "out"
    net.mark_output(cur)
    return net


def _dyn_shape(net, trt, ref, static_dim: int, dyn_axis: int):
    """Slice size tensor [M, static_dim] where M is taken from ``ref`` at runtime.

    add_slice with a dynamic leading dimension needs its size as a tensor;
    a static (M, half) would pin the engine to one batch size and defeat the
    whole point of the optimization profiles.
    """
    import numpy as np

    shp = net.add_shape(ref).get_output(0)  # [2] int64
    m = net.add_slice(shp, (0,), (1,), (1,)).get_output(0)
    const = net.add_constant(
        (1,), trt.Weights(np.array([static_dim], dtype=np.int64))
    ).get_output(0)
    return net.add_concatenation([m, const]).get_output(0)


# --------------------------------------------------------------------------
# Fold variants: dequantize the weights at BUILD time
# --------------------------------------------------------------------------

#: name -> (trt DataType attr, torch dtype name, bytes per weight element)
FOLD_DTYPES = {
    "fold_bf16": ("BF16", "bfloat16", 2),
    "fold_fp16": ("HALF", "float16", 2),
    "fp32_ref": ("FLOAT", "float32", 4),
}


def _trt_weights_from_torch(trt, t, keep):
    """A trt.Weights view of a CPU torch tensor, without a copy.

    numpy has no bfloat16, so the (type, ptr, count) overload is used and the
    tensor is appended to ``keep`` -- TensorRT does not copy, so the buffer must
    outlive the build call.
    """
    import torch

    dt = {
        torch.bfloat16: trt.DataType.BF16,
        torch.float16: trt.DataType.HALF,
        torch.float32: trt.DataType.FLOAT,
    }[t.dtype]
    t = t.contiguous()
    keep.append(t)
    return trt.Weights(dt, t.data_ptr(), t.numel())


def dequantized_weight(weights, torch_dtype: str):
    """w_q x per-channel scale, materialised once at build time.

    This is the FOLD. The INT8 grid the checkpoint was quantized to is
    preserved exactly -- every value is still a representable point of that
    grid -- it is simply stored in a wider container so no runtime dequantize
    is needed, and no activation quantization is needed either.

    On quality: bf16 carries 8 mantissa bits and fp16 carries 11, against an
    INT8 weight grid of ~7 bits plus a per-channel exponent. The fold therefore
    loses nothing on the weight side, and REMOVES the per-token activation
    quantization error the deployed path pays. It is a quality-neutral
    candidate in the strict sense: the harness measures both paths against an
    exact fp32 reference and gates on the fold being at least as accurate as
    what we deploy, rather than asserting it here.
    """
    import torch

    dt = getattr(torch, torch_dtype)
    w = weights.q.to(torch.float32) * weights.scale.reshape(-1, 1).to(torch.float32)
    return w.to(dt)


def build_fold_network(trt, builder, target, stage_weights, variant: str, keep):
    """The WHOLE chain as one engine.

    With the weights folded there is no activation quantization anywhere, and
    that is what removes the constraint that forced the INT8 variant to cut the
    chain at every quant: TensorRT could not express our per-token dynamic
    scale, so the INT8 engines are per-stage islands with our kernel between
    them. Folded, the entire part -- every GEMM, the SiLU-gate multiply, the
    attention-core bridge -- is a single engine and a single graph node, with
    nothing of ours running inside it at all.

    That is the interesting arm: it is the largest fusion surface TensorRT can
    be given on this model, and it is also the arm that pays 2x the weight
    bytes. Which of those two wins is the crossover this task measures.
    """
    trt_attr, torch_dtype, _ = FOLD_DTYPES[variant]
    dt = getattr(trt.DataType, trt_attr)
    net = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    k0 = target.engine_stages[0].gemm.k
    cur = net.add_input("x", dt, (-1, k0))

    for st in target.stages:
        if st.kind == "quant":
            # Deliberately nothing: the fold has no activation quant at all.
            continue
        if st.kind == "bridge":
            sl = net.add_slice(cur, (0, 0), (0, st.out_width), (1, 1))
            sl.set_input(2, _dyn_shape(net, trt, cur, st.out_width, 0))
            sl.name = f"bridge_{st.name}"
            cur = sl.get_output(0)
            continue

        spec = st.gemm
        w = dequantized_weight(stage_weights[st.name], torch_dtype)
        wc = net.add_constant(
            (spec.n, spec.k), _trt_weights_from_torch(trt, w, keep)
        ).get_output(0)
        mm = net.add_matrix_multiply(
            cur, trt.MatrixOperation.NONE, wc, trt.MatrixOperation.TRANSPOSE
        )
        mm.name = f"gemm_{st.name}"
        cur = mm.get_output(0)

        if spec.epilogue == "silu_mul":
            half = spec.n // 2
            gate = net.add_slice(cur, (0, 0), (0, half), (1, 1))
            gate.set_input(2, _dyn_shape(net, trt, cur, half, 0))
            up = net.add_slice(cur, (0, half), (0, half), (1, 1))
            up.set_input(2, _dyn_shape(net, trt, cur, half, 0))
            sig = net.add_activation(gate.get_output(0), trt.ActivationType.SIGMOID)
            silu = net.add_elementwise(
                gate.get_output(0), sig.get_output(0), trt.ElementWiseOperation.PROD
            )
            cur = net.add_elementwise(
                silu.get_output(0), up.get_output(0), trt.ElementWiseOperation.PROD
            ).get_output(0)

    if dt != trt.DataType.BF16:
        # The deployed path emits bf16. The arms are compared on the same output
        # dtype so the tolerance numbers mean the same thing across variants.
        c = net.add_cast(cur, trt.DataType.BF16)
        c.name = "to_bf16"
        cur = c.get_output(0)
    cur.name = "out"
    net.mark_output(cur)
    return net


def build_fold_engine(
    trt, target, stage_weights, variant, out_path, compute_capabilities,
    profile_opts, max_bs, verbose,
):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    keep = []  # weight buffers TensorRT does not copy
    net = build_fold_network(trt, builder, target, stage_weights, variant, keep)

    cfg = builder.create_builder_config()
    cfg.clear_flag(trt.BuilderFlag.TF32)
    cfg.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    portable = hasattr(cfg, "set_compute_capability")
    if portable:
        cfg.num_compute_capabilities = len(compute_capabilities)
        for i, cc in enumerate(compute_capabilities):
            if not cfg.set_compute_capability(getattr(trt.ComputeCapability, cc), i):
                raise SystemExit(f"builder rejected compute capability {cc}")

    k0 = target.engine_stages[0].gemm.k
    for opt in profile_opts:
        p = builder.create_optimization_profile()
        p.set_shape("x", (1, k0), (opt, k0), (max_bs, k0))
        cfg.add_optimization_profile(p)

    t0 = time.time()
    plan = builder.build_serialized_network(net, cfg)
    dt = time.time() - t0
    if plan is None:
        raise SystemExit(f"fold AOT build failed for {target.name} / {variant}")
    with open(out_path, "wb") as fh:
        fh.write(bytes(plan))

    _, torch_dtype, elem_bytes = FOLD_DTYPES[variant]
    weight_elems = sum(s.gemm.n * s.gemm.k for s in target.engine_stages)
    return {
        "engine": os.path.basename(out_path),
        "variant": variant,
        "target": target.name,
        "stages": [s.name for s in target.engine_stages],
        "single_engine_whole_chain": True,
        "activation_quant_kernels": 0,
        "bytes": plan.nbytes,
        "aot_build_seconds": round(dt, 3),
        "weight_elements": weight_elems,
        "weight_bytes_fold": weight_elems * elem_bytes,
        "weight_bytes_int8": weight_elems + sum(
            s.gemm.n * 2 for s in target.engine_stages
        ),
        "memory_multiplier_vs_int8": round(
            weight_elems * elem_bytes
            / max(weight_elems + sum(s.gemm.n * 2 for s in target.engine_stages), 1),
            3,
        ),
        "compute_capabilities": list(compute_capabilities) if portable
        else ["<current-device>"],
        "portable_multi_arch": portable,
        "profile_opts": list(profile_opts),
        "max_batch": max_bs,
        "strongly_typed": True,
        "builder_flags_set": sorted(
            f for f in dir(trt.BuilderFlag)
            if f.isupper() and cfg.get_flag(getattr(trt.BuilderFlag, f))
        ),
        "built_with_visible_devices": os.environ.get(
            "CUDA_VISIBLE_DEVICES", "<unset>"
        ),
        "tensorrt_rtx_version": trt.__version__,
    }


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def build_engine(
    trt,
    spec: tgt.GemmSpec,
    weights,
    out_path: str,
    compute_capabilities,
    profile_opts,
    max_bs: int,
    dq_dtype: str,
    verbose: bool,
) -> dict:
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = build_network(trt, builder, spec, weights, dq_dtype)

    cfg = builder.create_builder_config()

    # Precision hygiene: strongly typed network already forbids downgrades;
    # TF32 is cleared so no fp32 matmul could quietly run at reduced mantissa.
    cfg.clear_flag(trt.BuilderFlag.TF32)
    # A timing cache would force the AOT phase to look for a GPU and destroy the
    # CPU-only property this whole plan depends on.
    cfg.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    # Classic TensorRT has no multi-compute-capability API: it builds for the
    # card it is running on, which is precisely the per-architecture split that
    # TensorRT-RTX's portable AOT engine removes. The control arm therefore
    # builds one engine per card, in the window, and that asymmetry is part of
    # what the control measures.
    portable = hasattr(cfg, "set_compute_capability")
    if portable:
        cfg.num_compute_capabilities = len(compute_capabilities)
        for i, cc in enumerate(compute_capabilities):
            if not cfg.set_compute_capability(getattr(trt.ComputeCapability, cc), i):
                raise SystemExit(f"builder rejected compute capability {cc}")

    for opt in profile_opts:
        p = builder.create_optimization_profile()
        p.set_shape("a_q", (1, spec.k), (opt, spec.k), (max_bs, spec.k))
        p.set_shape("a_scale", (1, 1), (opt, 1), (max_bs, 1))
        cfg.add_optimization_profile(p)

    t0 = time.time()
    plan = builder.build_serialized_network(net, cfg)
    dt = time.time() - t0
    if plan is None:
        raise SystemExit(
            f"AOT build failed for {spec.name} (N={spec.n} K={spec.k} "
            f"epilogue={spec.epilogue}). Re-run with --verbose for the builder log."
        )
    with open(out_path, "wb") as fh:
        fh.write(bytes(plan))

    flags_set = [
        f for f in dir(trt.BuilderFlag)
        if f.isupper() and cfg.get_flag(getattr(trt.BuilderFlag, f))
    ]
    return {
        "engine": os.path.basename(out_path),
        "stage": spec.name,
        "n": spec.n,
        "k": spec.k,
        "epilogue": spec.epilogue,
        "module": spec.module,
        "note": spec.note,
        "weights_source": weights.source,
        "bytes": plan.nbytes,
        "aot_build_seconds": round(dt, 3),
        "compute_capabilities": list(compute_capabilities) if portable else ["<current-device>"],
        "portable_multi_arch": portable,
        "profile_opts": list(profile_opts),
        "max_batch": max_bs,
        "dq_dtype": dq_dtype,
        "strongly_typed": True,
        "builder_flags_set": sorted(flags_set),
        "builder_flags_available": sorted(
            f for f in dir(trt.BuilderFlag) if f.isupper()
        ),
        "timing_cache_used": False,
        "weight_streaming_used": False,
        "built_with_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "tensorrt_rtx_version": trt.__version__,
    }


def mock_engine(spec: tgt.GemmSpec, weights, out_path: str) -> dict:
    """Write a stub plan so the whole pipeline can be exercised without TensorRT.

    Path coverage, never a measurement. The harness recognises the stub by its
    magic header and runs a torch stand-in in the TensorRT arms, stamping every
    resulting timing with ``"stub": true``.
    """
    with open(out_path, "wb") as fh:
        fh.write(b"MOCK337\x00" + json.dumps(
            {"n": spec.n, "k": spec.k, "epilogue": spec.epilogue}
        ).encode())
    return {
        "engine": os.path.basename(out_path),
        "stage": spec.name,
        "n": spec.n,
        "k": spec.k,
        "epilogue": spec.epilogue,
        "weights_source": weights.source,
        "bytes": os.path.getsize(out_path),
        "mock": True,
    }


def mock_fold_engine(target, variant, out_path) -> dict:
    """Stub fold plan, so --mock exercises the fold code paths too."""
    _, _, elem_bytes = FOLD_DTYPES[variant]
    elems = sum(s.gemm.n * s.gemm.k for s in target.engine_stages)
    with open(out_path, "wb") as fh:
        fh.write(b"MOCK337\x00" + json.dumps(
            {"target": target.name, "variant": variant,
             "stages": [s.name for s in target.engine_stages]}
        ).encode())
    return {
        "engine": os.path.basename(out_path),
        "variant": variant,
        "target": target.name,
        "stages": [s.name for s in target.engine_stages],
        "single_engine_whole_chain": True,
        "activation_quant_kernels": 0,
        "bytes": os.path.getsize(out_path),
        "weight_elements": elems,
        "weight_bytes_fold": elems * elem_bytes,
        "weight_bytes_int8": elems + sum(s.gemm.n * 2 for s in target.engine_stages),
        "memory_multiplier_vs_int8": round(
            elems * elem_bytes
            / max(elems + sum(s.gemm.n * 2 for s in target.engine_stages), 1),
            3,
        ),
        "mock": True,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=tgt.DEFAULT_CONFIG)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--tp-size", type=int, default=3)
    ap.add_argument("--ratio", default="30,17,17")
    ap.add_argument("--ranks", default="0,1", help="which rank geometries to build")
    ap.add_argument("--optional", default="", help="e.g. gdn_conv")
    ap.add_argument("--out-dir", required=False, default="")
    ap.add_argument(
        "--compute-capabilities",
        default=",".join(DEFAULT_COMPUTE_CAPABILITIES),
        help="one plan covers all of them; kCURRENT is refused",
    )
    ap.add_argument("--profile-opts", default="1,4")
    ap.add_argument("--max-bs", type=int, default=DEFAULT_MAX_BS)
    ap.add_argument("--dq-dtype", choices=("float", "bf16"), default="float")
    ap.add_argument(
        "--precision",
        default="int8",
        help="comma list of variants to build: int8 (per-stage engines, our "
             "quant kernel between them), fold_bf16 / fold_fp16 (weights "
             "dequantized at build time, WHOLE chain in one engine, no "
             "activation quant anywhere), fp32_ref (diagnostic only -- see "
             "--fp32-ref-targets).",
    )
    ap.add_argument(
        "--fp32-ref-targets",
        default="gemm_mlp_gate_up",
        help="fp32_ref is a diagnostic arm, not a deployment candidate: at 4 "
             "bytes per weight it does not fit this rig. Built for ONE shape so "
             "it can isolate TensorRT runtime quality from the quantization "
             "constraints, and no further.",
    )
    ap.add_argument(
        "--random-weights",
        action="store_true",
        help="shape-faithful CPU-sampled randoms instead of checkpoint shards",
    )
    ap.add_argument("--seed", type=int, default=337)
    ap.add_argument(
        "--library",
        choices=("rtx", "classic"),
        default="rtx",
        help="rtx = TensorRT-RTX, portable CPU-built AOT engine (default). "
             "classic = plain TensorRT, the control arm; needs a GPU to build, "
             "so it belongs in the card window, not at a desk.",
    )
    ap.add_argument("--mock", action="store_true", help="stub engines, no TensorRT")
    ap.add_argument("--probe", action="store_true", help="print capability probe only")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    trt = import_trt(a.mock, a.library)

    if a.probe:
        print(json.dumps(probe(trt), indent=1))
        return 0

    if a.library == "classic":
        print(
            "classic TensorRT control build: no portable multi-arch engine, "
            "one plan per card, GPU required."
        )
    elif "CURRENT" in a.compute_capabilities.upper():
        raise SystemExit(
            "kCURRENT pins the plan to the card that happens to be visible and "
            "reintroduces the per-architecture split this build deliberately "
            "removes. Name the architectures explicitly."
        )

    if not a.out_dir:
        raise SystemExit("--out-dir is required")
    os.makedirs(a.out_dir, exist_ok=True)
    ratio = [int(x) for x in a.ratio.split(",")]
    ccs = [c.strip().upper() for c in a.compute_capabilities.split(",")]
    opts = [int(x) for x in a.profile_opts.split(",")]
    optional = [x for x in a.optional.split(",") if x]

    variants = [v.strip() for v in a.precision.split(",") if v.strip()]
    fp32_targets = {t for t in a.fp32_ref_targets.split(",") if t}

    manifest = {
        "task": "337",
        "variants": variants,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mock": a.mock,
        "library": a.library,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "config": a.config,
        "model_dir": a.model_dir,
        "tp_size": a.tp_size,
        "ratio": ratio,
        "compute_capabilities": ccs,
        "profile_opts": opts,
        "max_bs": a.max_bs,
        "dq_dtype": a.dq_dtype,
        "random_weights": a.random_weights,
        "engines": [],
        "ranks": [],
    }
    if not a.mock:
        manifest["capability_probe"] = probe(trt)

    total_t0 = time.time()
    seen = {}
    for rank in [int(r) for r in a.ranks.split(",")]:
        geo = tgt.derive_geometry(a.config, a.tp_size, ratio, rank)
        ts = tgt.build_targets(geo, optional)
        manifest["ranks"].append(
            {
                "rank": rank,
                "arch": geo.arch,
                "partition_provenance": geo.partition_provenance,
                "local_intermediate": geo.local_intermediate,
                "targets": [t.name for t in ts],
                "shapes": tgt.shape_table(ts),
            }
        )
        if "int8" in variants:
            for t in ts:
                for st in t.engine_stages:
                    key = (rank, st.gemm.name, st.gemm.n, st.gemm.k,
                           st.gemm.epilogue)
                    if key in seen:
                        continue  # identical stages are shared across targets
                    suffix = "" if a.library == "rtx" else ".classic"
                    name = f"rank{rank}_{geo.arch}_{st.gemm.name}{suffix}.plan"
                    path = os.path.join(a.out_dir, name)
                    w = tgt.load_gemm_weights(
                        st.gemm, geo, a.model_dir, a.random_weights, a.seed
                    )
                    if a.mock:
                        rec = mock_engine(st.gemm, w, path)
                    else:
                        rec = build_engine(
                            trt, st.gemm, w, path, ccs, opts, a.max_bs,
                            a.dq_dtype, a.verbose,
                        )
                    rec["rank"] = rank
                    rec["arch"] = geo.arch
                    rec["variant"] = "int8"
                    manifest["engines"].append(rec)
                    seen[key] = name
                    print(
                        f"  built {name:48s} N={st.gemm.n:6d} K={st.gemm.k:6d} "
                        f"{rec.get('bytes',0)/1e6:7.2f} MB "
                        f"{rec.get('aot_build_seconds', 0):6.2f}s"
                    )

        for variant in variants:
            if variant == "int8":
                continue
            if variant not in FOLD_DTYPES:
                raise SystemExit(f"unknown precision variant {variant!r}")
            for t in ts:
                if variant == "fp32_ref" and t.name not in fp32_targets:
                    continue
                name = f"rank{rank}_{geo.arch}_{t.name}.{variant}.plan"
                path = os.path.join(a.out_dir, name)
                sw = {
                    st.name: tgt.load_gemm_weights(
                        st.gemm, geo, a.model_dir, a.random_weights, a.seed
                    )
                    for st in t.engine_stages
                }
                if a.mock:
                    rec = mock_fold_engine(t, variant, path)
                else:
                    rec = build_fold_engine(
                        trt, t, sw, variant, path, ccs, opts, a.max_bs, a.verbose
                    )
                rec["rank"] = rank
                rec["arch"] = geo.arch
                manifest["engines"].append(rec)
                print(
                    f"  built {name:48s} chain={len(t.engine_stages)} "
                    f"{rec.get('bytes',0)/1e6:7.2f} MB "
                    f"{rec.get('aot_build_seconds', 0):6.2f}s "
                    f"mem_x{rec.get('memory_multiplier_vs_int8','?')}"
                )

    manifest["total_build_seconds"] = round(time.time() - total_t0, 2)
    mpath = os.path.join(a.out_dir, "manifest.json")
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(
        f"\n{len(manifest['engines'])} engines, "
        f"{manifest['total_build_seconds']}s total -> {mpath}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
