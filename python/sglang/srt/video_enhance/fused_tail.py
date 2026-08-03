# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Fusing the 8K->4K tail resize into the SR graph (#457).

``sr.py``'s ``FUSED_TAIL_RESIZE_NOTE`` registered this as a follow-on and said
why: SRVGGNetCompact computes at input resolution and expands only in the
final pixel-shuffle, so everything expensive about the x4 output is memory
traffic in the tail. At 1080p input the tail writes a 189.84 MiB fp16 8K frame
that the next stage reads back and immediately reduces to 47.46 MiB. Ticket V
priced the two halves of that round trip at 25.42 ms (SR) and 24.37 ms
(resize) on the 5090, which is the pair that binds the chain.

Two routes were on the table. This module builds one of them and records what
the other would have cost.

Route (a) -- **built** -- appends the downscale to the pinned ONNX before the
engine is built, so one engine emits 4K directly. The engine's output drops
from 189.84 to 47.46 MiB per frame, the intermediate never exists, and
TensorRT is free to fuse the tail into the pixel-shuffle rather than round-trip
it through global memory.

Route (b) -- **not built** -- a CUDA op applied to the engine's output inside
the same stage. It removes the stage boundary but not the intermediate: the
8K frame is still written and still read. Its price, by the pricer's own
model, is exactly zero milliseconds -- ``stage_pipeline.price_placement``
charges a boundary only when it crosses a *card*, and ``sr``/``resize`` are
already forced co-resident, so the boundary route (b) removes was never priced
at anything. Whatever route (b) could win would have to come from a faster
resize kernel, which is a different piece of work from fusion and is named as
such in TASK_333 §17.7.

What makes route (a) exact rather than an approximation
-------------------------------------------------------
Lanczos-3 is not an ONNX operator, and the obvious substitute -- ``Resize``
with ``antialias=1`` -- is a different filter that would need a quality gate
to justify. It is not needed. For the one ratio the chain actually asks for,
**exactly 2:1**, the Lanczos-3 tap table collapses to a single vector shared
by every output pixel (see ``resize.halving_taps`` for the derivation), and
the resample becomes a stride-2 depthwise convolution with edge padding. The
fused graph therefore computes *the reference filter*, not a stand-in, and its
parity gate measures float rounding order rather than filter choice.

The ratio restriction is real and is enforced. The tap collapse happens for
any exact integer decimation (ratio ``1/q``) and for nothing else -- ``600 ->
400`` is 2/3 and its taps do vary with the output position -- and of those,
2:1 is the one the chain asks for and the only one that leaves an x4 model
with an integer net scale. Any other geometry is refused by name.

The pinned-artifact rule is unchanged. The source stays the sha256-pinned
styler00dollar ONNX; this produces a *derived* artifact with a provenance
sidecar naming the source hash, exactly as ``export_sr_fp16.py`` does, and the
loader verifies the chain rather than trusting the filename.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from sglang.srt.video_enhance.engine_cache import sha256_file
from sglang.srt.video_enhance.frame_math import Resolution
from sglang.srt.video_enhance.parity import ParityResult, grade
from sglang.srt.video_enhance.resize import (
    LANCZOS_A,
    halving_pad,
    halving_taps,
    is_exact_halving,
    lanczos3_resize,
)

__all__ = [
    "FUSED_TAIL_SUFFIX",
    "PROVENANCE_SUFFIX",
    "TAIL_KINDS",
    "FusedTailError",
    "FusedTailPlan",
    "append_halving_tail",
    "apply_tail_torch",
    "fuse_tail",
    "fused_tail_reference",
    "grade_fused_tail",
    "plan_fused_tail",
]

#: Suffix of the derived artifact and of the sidecar it is verified against.
FUSED_TAIL_SUFFIX = "_fusedtail.onnx"
PROVENANCE_SUFFIX = ".provenance.json"

#: The tails this module can append.
#:
#: ``lanczos3``
#:     The production one. Bit-for-bit the filter ``resize.lanczos3_resize``
#:     applies, expressed as a convolution.
#: ``nearest``
#:     Decimation by dropping samples. Present so the parity gate can be shown
#:     to *fail* -- a gate that has never rejected anything is an assertion,
#:     not a gate.
#: ``bicubic_antialias``
#:     ONNX ``Resize`` with ``antialias=1``, the route this module did not
#:     take. Kept buildable so the choice is settled by a measured PSNR
#:     against the Lanczos-3 reference rather than by argument. Needs opset 18.
TAIL_KINDS = ("lanczos3", "nearest", "bicubic_antialias")

#: ``Resize-18`` is the first opset with the ``antialias`` attribute. The
#: pinned artifact is opset 16, so the comparison arm has to raise the graph's
#: opset -- which the production arm does not, because a convolution and an
#: edge pad are both opset-16 operators.
ANTIALIAS_MIN_OPSET = 18


class FusedTailError(RuntimeError):
    """The tail cannot be fused for this geometry, or the graph surgery failed."""


@dataclass(frozen=True)
class FusedTailPlan:
    """What the surgery will do, decided before any proto is touched."""

    kind: str
    taps: tuple[float, ...]
    pad_begin: int
    pad_end: int
    stride: int
    #: Net scale of the fused graph: the model's own scale divided by 2.
    net_scale: int
    #: True when the graph's opset has to be raised for this tail.
    needs_opset: int | None = None

    def describe(self) -> str:
        return (
            f"{self.kind} tail, {len(self.taps)} taps, stride {self.stride}, "
            f"edge pad {self.pad_begin}/{self.pad_end}, net scale "
            f"x{self.net_scale}"
        )


def plan_fused_tail(
    *,
    kind: str = "lanczos3",
    model_scale: int = 4,
    a: int = LANCZOS_A,
) -> FusedTailPlan:
    """Decide the tail before touching a graph, and refuse what does not fit."""
    if kind not in TAIL_KINDS:
        raise FusedTailError(f"unknown tail kind {kind!r}; known: {list(TAIL_KINDS)}")
    if model_scale % 2 != 0:
        raise FusedTailError(
            f"a x{model_scale} model cannot carry an exact 2:1 tail: the fused "
            "graph's net scale would not be an integer. Only even-scale models "
            "have a halving tail whose taps are output-position independent"
        )
    pad_begin, pad_end = halving_pad(a)
    if kind == "nearest":
        # One-hot at the sample the window is centred on. Deliberately the
        # worst legitimate decimation, so the gate has something to reject.
        taps = tuple(1.0 if t == pad_begin else 0.0 for t in range(4 * a))
    elif kind == "lanczos3":
        taps = halving_taps(a)
    else:
        taps = ()
    return FusedTailPlan(
        kind=kind,
        taps=taps,
        pad_begin=pad_begin,
        pad_end=pad_end,
        stride=2,
        net_scale=model_scale // 2,
        needs_opset=ANTIALIAS_MIN_OPSET if kind == "bicubic_antialias" else None,
    )


def refuse_unless_halving(source: Resolution, target: Resolution) -> None:
    """Guard for a caller that has a concrete geometry in hand."""
    if not is_exact_halving(source, target):
        raise FusedTailError(
            f"the fused tail implements exactly 2:1 decimation; {source} -> "
            f"{target} is not that ratio. For a ratio p/q in lowest terms the "
            "Lanczos tap pattern repeats with period p in output space, so "
            "only p == 1 is one convolution and this geometry keeps the "
            "separate resize stage"
        )


def apply_tail_torch(x, plan: FusedTailPlan):
    """The tail as torch ops -- the reference twin of the ONNX nodes.

    Two jobs. It lets the registered test prove the load-bearing claim (that
    this convolution *is* ``lanczos3_resize``) without the build-time ``onnx``
    dependency, and it is route (b) in usable form: if the fused engine ever
    fails to build on an arch, this is the same arithmetic applied to the
    unfused engine's output, inside the SR stage, with no host copy.

    It does not save the 8K intermediate -- only the graph surgery does that.
    """
    import torch
    import torch.nn.functional as F

    if plan.kind == "bicubic_antialias":
        raise FusedTailError(
            "the bicubic_antialias arm exists as an ONNX comparison only; there "
            "is no torch twin because it is not the filter this fork ships"
        )
    if x.dim() != 4:
        raise ValueError(f"expected NCHW, got shape {tuple(x.shape)}")
    channels = x.shape[1]
    n = len(plan.taps)
    # fp32 accumulation for the same reason resize._resize_axis uses it: a
    # 12-tap normalised sum in fp16 is visibly short of the fp32 reference.
    work = x.to(torch.float32)
    taps = torch.tensor(plan.taps, dtype=torch.float32, device=x.device)
    for axis in (3, 2):
        pad = [0, 0, 0, 0]
        # F.pad's pad list is last-axis-first: (W_begin, W_end, H_begin, H_end).
        offset = 0 if axis == 3 else 2
        pad[offset] = plan.pad_begin
        pad[offset + 1] = plan.pad_end
        work = F.pad(work, pad, mode="replicate")
        shape = (channels, 1, 1, n) if axis == 3 else (channels, 1, n, 1)
        stride = (1, plan.stride) if axis == 3 else (plan.stride, 1)
        work = F.conv2d(
            work,
            taps.reshape(1, 1, -1).expand(channels, 1, n).reshape(shape),
            stride=stride,
            groups=channels,
        )
    return work.to(x.dtype)


# --------------------------------------------------------------------------
# Graph surgery
# --------------------------------------------------------------------------


def _separable_conv_tail(graph, plan: FusedTailPlan, produced: str, final: str):
    """Pad + stride-2 depthwise conv on W, then the same on H.

    Separable rather than one 12x12 kernel: 24 multiply-adds per output pixel
    instead of 144, and the intermediate is half-width so it is smaller than
    either endpoint. The horizontal pass runs first, matching the order
    ``resize.lanczos3_resize`` uses and the order
    ``frame_math.resize_footprint`` prices.
    """
    import numpy as np
    from onnx import helper, numpy_helper

    channels = 3
    taps = np.asarray(plan.taps, dtype=np.float32)
    n = taps.size

    nodes = []
    initializers = []

    # Pad takes its pads as an int64 input, NCHW order
    # [N_begin, C_begin, H_begin, W_begin, N_end, C_end, H_end, W_end].
    for axis, name in ((3, "w"), (2, "h")):
        pads = [0, 0, 0, 0, 0, 0, 0, 0]
        pads[axis] = plan.pad_begin
        pads[axis + 4] = plan.pad_end
        initializers.append(
            numpy_helper.from_array(
                np.asarray(pads, dtype=np.int64), name=f"fused_tail_pads_{name}"
            )
        )
        kernel = taps.reshape(1, 1, 1, n) if axis == 3 else taps.reshape(1, 1, n, 1)
        initializers.append(
            numpy_helper.from_array(
                np.repeat(kernel, channels, axis=0).copy(),
                name=f"fused_tail_weight_{name}",
            )
        )

    nodes.append(
        helper.make_node(
            "Pad",
            inputs=[produced, "fused_tail_pads_w"],
            outputs=["fused_tail_padded_w"],
            mode="edge",
            name="fused_tail_pad_w",
        )
    )
    nodes.append(
        helper.make_node(
            "Conv",
            inputs=["fused_tail_padded_w", "fused_tail_weight_w"],
            outputs=["fused_tail_half_w"],
            group=channels,
            kernel_shape=[1, n],
            strides=[1, plan.stride],
            pads=[0, 0, 0, 0],
            name="fused_tail_conv_w",
        )
    )
    nodes.append(
        helper.make_node(
            "Pad",
            inputs=["fused_tail_half_w", "fused_tail_pads_h"],
            outputs=["fused_tail_padded_h"],
            mode="edge",
            name="fused_tail_pad_h",
        )
    )
    nodes.append(
        helper.make_node(
            "Conv",
            inputs=["fused_tail_padded_h", "fused_tail_weight_h"],
            outputs=[final],
            group=channels,
            kernel_shape=[n, 1],
            strides=[plan.stride, 1],
            pads=[0, 0, 0, 0],
            name="fused_tail_conv_h",
        )
    )
    graph.node.extend(nodes)
    graph.initializer.extend(initializers)
    return len(nodes)


def _resize_tail(graph, plan: FusedTailPlan, produced: str, final: str):
    """The route not taken: ONNX ``Resize`` with ``antialias=1``."""
    import numpy as np
    from onnx import helper, numpy_helper

    graph.initializer.extend(
        [
            numpy_helper.from_array(
                np.asarray([], dtype=np.float32), name="fused_tail_roi"
            ),
            numpy_helper.from_array(
                np.asarray([1.0, 1.0, 0.5, 0.5], dtype=np.float32),
                name="fused_tail_scales",
            ),
        ]
    )
    graph.node.append(
        helper.make_node(
            "Resize",
            inputs=[produced, "fused_tail_roi", "fused_tail_scales"],
            outputs=[final],
            mode="cubic",
            antialias=1,
            coordinate_transformation_mode="half_pixel",
            cubic_coeff_a=-0.75,
            name="fused_tail_resize",
        )
    )
    return 1


def append_halving_tail(model, plan: FusedTailPlan):
    """Append the tail to ``model``'s single output, in place.

    The graph's output *name* is preserved -- consumers resolve it through
    ``session.get_outputs()[0].name`` and an artifact that renamed it would
    break them for no reason. The tensor the old last node produced is
    renamed instead, exactly as ``export_sr_fp16.insert_io_casts`` does.
    """
    graph = model.graph
    if len(graph.output) != 1:
        raise FusedTailError(
            f"the fused tail assumes one graph output, found {len(graph.output)}"
        )
    final = graph.output[0].name
    produced = f"__pre_fused_tail_{final}"
    rewired = 0
    for node in graph.node:
        for i, out in enumerate(node.output):
            if out == final:
                node.output[i] = produced
                rewired += 1
    if rewired != 1:
        raise FusedTailError(
            f"expected exactly one node to produce {final!r}, found {rewired}"
        )

    if plan.kind == "bicubic_antialias":
        added = _resize_tail(graph, plan, produced, final)
    else:
        added = _separable_conv_tail(graph, plan, produced, final)

    # The source artifact declares its output spatial dims with the *input's*
    # symbols ('width', 'height'), which was already inaccurate for an x4
    # model. Rather than propagate a wrong symbol, the fused output gets its
    # own names: the graph no longer claims any relation to the input's dims.
    shape = graph.output[0].type.tensor_type.shape
    for index, label in ((2, "fused_tail_dim2"), (3, "fused_tail_dim3")):
        if index < len(shape.dim):
            shape.dim[index].Clear()
            shape.dim[index].dim_param = label

    if plan.needs_opset is not None:
        for imp in model.opset_import:
            if imp.domain in ("", "ai.onnx") and imp.version < plan.needs_opset:
                imp.version = plan.needs_opset
    return model, added


def _assert_loadable(onnx_path: Path) -> None:
    """Open the artifact once on CPU; delete it if the runtime refuses it.

    Same discipline as ``export_sr_fp16._assert_loadable`` and for the same
    recorded reason: ``onnx.checker`` once accepted an artifact ONNX Runtime
    would not load, and the sidecar written next to it claimed it was built.
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
        raise FusedTailError(
            f"the fused artifact {onnx_path.name} is not loadable and has been "
            f"removed rather than left on disk looking finished: {exc}"
        ) from exc


def fuse_tail(
    source_onnx: Path,
    out_path: Path,
    *,
    kind: str = "lanczos3",
    model_scale: int = 4,
) -> dict:
    """Derive a fused-tail artifact from a pinned SR ONNX and record provenance.

    Returns the manifest that was written next to the artifact. The manifest
    is what ``sr.derived_fused_tail_model`` verifies against -- a file whose
    bytes no longer match it fails the same check a corrupted download does.
    """
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - build-time tool only
        raise FusedTailError(
            "the fused-tail export needs the 'onnx' package for the graph "
            "surgery. It is a build-time dependency and is deliberately not in "
            "the serving requirements; install it into a scratch target and put "
            "that on PYTHONPATH."
        ) from exc

    plan = plan_fused_tail(kind=kind, model_scale=model_scale)
    model = onnx.load(str(source_onnx))
    source_opset = {imp.domain: imp.version for imp in model.opset_import}
    model, added = append_halving_tail(model, plan)
    onnx.checker.check_model(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    _assert_loadable(out_path)

    manifest = {
        "derived_from": str(source_onnx),
        "derived_from_sha256": sha256_file(source_onnx),
        "sha256": sha256_file(out_path),
        "bytes": out_path.stat().st_size,
        "method": f"append_halving_tail:{plan.kind}",
        "tail": plan.describe(),
        "taps": list(plan.taps),
        "nodes_added": added,
        "model_scale": model_scale,
        "net_scale": plan.net_scale,
        "source_opset": source_opset,
        "opset": {imp.domain: imp.version for imp in model.opset_import},
        "exported_at": time.time(),
        "tool": "python/sglang/srt/video_enhance/fused_tail.py",
    }
    out_path.with_suffix(out_path.suffix + PROVENANCE_SUFFIX).write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    return manifest


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


def fused_tail_reference(sr_output, target: Resolution):
    """The reference a fused engine is graded against: SR output, then Lanczos-3.

    Deliberately the *existing* two-stage path rather than a re-derivation of
    it. The claim under test is that one engine produces what two stages
    produced, so the second stage's own code has to be on the reference side
    of the comparison.
    """
    if sr_output.dim() != 4:
        raise ValueError(f"expected NCHW, got shape {tuple(sr_output.shape)}")
    source = Resolution(sr_output.shape[3], sr_output.shape[2])
    refuse_unless_halving(source, target)
    return lanczos3_resize(sr_output, target)


def grade_fused_tail(
    candidate,
    sr_output,
    *,
    target: Resolution,
    note: str = "",
    **thresholds,
) -> ParityResult:
    """Grade a fused engine's 4K output against SR-then-Lanczos-3 on the same input.

    Thresholds default to ``parity``'s 40 dB / 0.995, the same gate the fp16
    engine passes at 48.1 dB. The Lanczos-3 tail is the reference filter
    expressed differently, so what this measures is float rounding order and a
    passing grade should clear the threshold by a wide margin; the
    ``nearest`` tail exists to show the gate still rejects a wrong filter.
    """
    reference = fused_tail_reference(sr_output, target)
    return grade(candidate, reference, note=note, **thresholds)
