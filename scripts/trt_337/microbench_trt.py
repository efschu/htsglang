#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Task #337 -- TensorRT-RTX vs. our own kernels, per part, per regime.

THE QUESTION, AND THE TRAP IN IT
================================
"Do per-part TensorRT engines beat our current kernels?" has an obvious wrong
way to answer it. TensorRT is not an alternative to CUDA graphs: TensorRT-RTX
captures the whole engine into a CUDA graph itself -- the installed runtime
exposes ``CudaGraphStrategy.WHOLE_GRAPH_CAPTURE`` as a first-class knob, and
TensorRT-LLM ships that way. TensorRT is *fusion + tactic selection + precision
optimization*, and then a CUDA graph on top.

So a measurement of "TensorRT (graphed) vs our kernels (eager)" would credit
TensorRT with removing a launch cost that a CUDA graph removes for free on
either side. #368 measured exactly how big that free win is on this rig: the
per-token INT8 activation quant costs 0.0266 ms eager and 0.0012 ms under graph
replay, ~21x, and the quant's share of the fused path falls from 61 % to 11 %.
Anyone quoting a TensorRT-vs-eager ratio here would be quoting that 21x with a
TensorRT label on it.

The verdict cell of this harness is therefore GRAPH VS GRAPH, and the other
cells exist to make the attribution visible rather than to be quoted:

    torch_eager        our kernels, per-op launches                 context
    torch_graph        our kernels, one captured graph              BASELINE
    trt_enqueue        engines via execute_async_v3, no graph       context
    trt_native_graph   engines with WHOLE_GRAPH_CAPTURE             context
    trt_outer_graph    engines + our quant in ONE captured graph    VERDICT

    verdict            trt_outer_graph / torch_graph
    launch axis        torch_eager / torch_graph   -- our own launch constant
    trt launch axis    trt_enqueue / trt_native_graph -- TensorRT's own
    DO NOT QUOTE       trt_*_graph / torch_eager   -- double counts the graph

``trt_outer_graph`` is the deployment form: one CUDA graph over the whole part,
containing our quant kernels and TensorRT's engine executions. ``torch_graph``
is the same part, same total work, captured the same way, running our kernels.
Both arms carry an identical amount of work, because the activation quant is
ours in every arm -- TensorRT cannot express it (see ``targets.py``), so it is
never inside an engine and never silently skipped.

DURATION RULE
=============
Every measured point runs for at least ``--min-point-seconds`` (default 10) of
real measured time, and every individual arm for at least
``--min-arm-seconds``. A calibration probe converts those into a burst size and
a round count; both are recorded. Short bursts at these shapes measure the CUDA
event pair as much as the kernel, and a point that finishes in 200 ms has not
settled thermally.

A-VS-A FLOOR
============
The arms named by ``--a2-arms`` (by default the two verdict arms) are each
instantiated TWICE on independent tensors and both copies sit in the same
rotation. The spread between the two copies IS the noise floor at that operating
point. A verdict ratio inside its own floor is not a result.

Distributions are reported as median / p5 / p95 over rounds, never as means.

TOLERANCE
=========
Byte identity is not expected and not required: the epilogue scaling happens in
a different order, and possibly a different intermediate precision, than CUTLASS
uses. The integer part is what must agree. The gate is a relative bound
(``--tolerance``, default 2e-2, which is ~2.5 bf16 ulps at the output
magnitudes these shapes produce), ``max_abs_diff`` and ``max_rel_diff`` are
always reported whether the gate passes or not, and a failure is recorded and
carried rather than silently swallowed -- an engine that is fast and wrong is a
result about TensorRT, not a reason to stop.

JIT WARM-UP
===========
The AOT plans in ``--engine-dir`` were built on a CPU with no GPU visible. The
first execution on a card triggers TensorRT's JIT specialisation. That cost is
measured and reported SEPARATELY from steady state, and the resulting runtime
cache is serialised to ``--runtime-cache-dir`` keyed by GPU identity, so a
second window run pays it once. Kernel specialisation strategy is set to EAGER
so no measurement can land on a generic fallback kernel; the strategy is
recorded in the output.

MOCK MODE
=========
``--mock`` runs every code path on CPU with torch stand-ins for both the sgl
kernels and the TensorRT runtime (stub engines written by
``build_engines.py --mock``), so the harness is proven to run end to end before
a card window is claimed. Output is stamped ``"mock": true`` and every timing
carries ``"stub": true``. It is path coverage, never a measurement.

USAGE
=====
    # desk, no card
    CUDA_VISIBLE_DEVICES="" python3 scripts/trt_337/microbench_trt.py --mock \\
        --engine-dir /tmp/337mock --out /tmp/337mock/bench.json

    # card, inside a claimed arbitration window
    CUDA_VISIBLE_DEVICES=<idx> PYTHONPATH=<pylibs>:<repo>/python \\
        python3 scripts/trt_337/microbench_trt.py \\
        --engine-dir <artifacts>/engines --rank 0 \\
        --runtime-cache-dir <artifacts>/rtcache \\
        --out <artifacts>/bench.<card>.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Callable

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import targets as tgt  # noqa: E402

ARMS = [
    "torch_eager",
    "torch_graph",
    "trt_enqueue",
    "trt_native_graph",
    "trt_outer_graph",
    "trt_fold_bf16_graph",
    "trt_fold_fp16_graph",
    "trt_fp32_ref_graph",
]
#: fold variants: the WHOLE chain is one engine and no activation quant runs at
#: all, so these arms have no torch-side stage structure to mirror -- they are
#: one engine call, captured in one graph, compared against torch_graph.
FOLD_ARMS = {
    "trt_fold_bf16_graph": "fold_bf16",
    "trt_fold_fp16_graph": "fold_fp16",
    "trt_fp32_ref_graph": "fp32_ref",
}
DEFAULT_A2 = "torch_graph,trt_outer_graph,trt_fold_bf16_graph"
FOLD_TORCH_DTYPE = {
    "fold_bf16": "bfloat16",
    "fold_fp16": "float16",
    "fp32_ref": "float32",
}
MOCK_MAGIC = b"MOCK337\x00"


# --------------------------------------------------------------------------
# Kernels: ours, and their CPU stand-ins
# --------------------------------------------------------------------------


@dataclass
class Kernels:
    per_token_quant_int8: Callable
    int8_scaled_mm: Callable
    silu_and_mul: Callable
    stub: bool = False
    missing: list = field(default_factory=list)


def _stub_per_token_quant_int8(x):
    import torch

    s = x.abs().to(torch.float32).amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    return (x.to(torch.float32) / s).round().clamp(-127, 127).to(torch.int8), s


def _stub_int8_scaled_mm(a, b, sa, sb, out_dtype, bias=None):
    import torch

    acc = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    out = acc * sa.to(torch.float32) * sb.to(torch.float32).reshape(1, -1)
    if bias is not None:
        out = out + bias
    return out.to(out_dtype)


def _stub_silu_and_mul(x):
    import torch

    a, b = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(a) * b


def load_kernels(mock: bool) -> Kernels:
    if mock:
        return Kernels(
            _stub_per_token_quant_int8, _stub_int8_scaled_mm, _stub_silu_and_mul,
            stub=True,
        )
    missing = []
    from sglang.srt.layers.quantization.int8_kernel import (  # noqa: PLC0415
        per_token_quant_int8,
    )
    from sgl_kernel import int8_scaled_mm  # noqa: PLC0415

    try:
        from sgl_kernel import silu_and_mul as _sam  # noqa: PLC0415

        def silu_and_mul(x):
            import torch

            out = torch.empty(
                x.shape[0], x.shape[1] // 2, dtype=x.dtype, device=x.device
            )
            _sam(x, out)
            return out

    except Exception as ex:
        missing.append(f"sgl_kernel.silu_and_mul: {type(ex).__name__}: {ex}")
        silu_and_mul = _stub_silu_and_mul
    return Kernels(per_token_quant_int8, int8_scaled_mm, silu_and_mul, missing=missing)


# --------------------------------------------------------------------------
# Engine wrappers -- real and stub, same interface
# --------------------------------------------------------------------------


class StubFoldEngine:
    """Torch stand-in for a fold engine: the whole chain, dense weights.

    Mirrors build_fold_network op for op so --mock exercises the fold arms.
    """

    def __init__(self, target, dense_weights, fold_dtype):
        self.target = target
        self.w = dense_weights
        self.fold_dtype = fold_dtype
        self.stub = True
        self.jit_seconds = 0.0

    def run_fold(self, x):
        import torch

        cur = x
        for st in self.target.stages:
            if st.kind == "quant":
                continue
            if st.kind == "bridge":
                cur = cur[:, : st.out_width].contiguous()
                continue
            cur = cur @ self.w[st.name].t()
            if st.gemm.epilogue == "silu_mul":
                half = cur.shape[-1] // 2
                gate, up = cur[:, :half], cur[:, half:]
                cur = torch.nn.functional.silu(gate) * up
        return cur.to(torch.bfloat16)


class StubEngine:
    """Torch stand-in with the real engine's interface.

    Exists so ``--mock`` exercises binding, shape setting, execution, tolerance
    comparison and JSON emission without TensorRT. It computes the same
    arithmetic the engine does, which also makes the tolerance path meaningful
    in mock mode (it should be near zero there).
    """

    def __init__(self, meta, w_q, w_scale, kernels):
        self.n, self.k = meta["n"], meta["k"]
        self.epilogue = meta["epilogue"]
        self.w_q, self.w_scale, self.kn = w_q, w_scale, kernels
        self.jit_seconds = 0.0
        self.stub = True

    def set_shape(self, m):
        self.m = m

    def run(self, a_q, a_scale):
        import torch

        # bf16 BEFORE the epilogue, mirroring the real engine's cast placement
        # and the deployed int8_scaled_mm(out_dtype=bfloat16) -> silu_and_mul
        # dataflow. Same rounding points, so mock-mode tolerance is ~0 and a
        # nonzero tolerance in a card run means something real.
        out = self.kn.int8_scaled_mm(
            a_q, self.w_q.t(), a_scale, self.w_scale, out_dtype=torch.bfloat16
        )
        if self.epilogue == "silu_mul":
            out = _stub_silu_and_mul(out)
        return out


class TrtEngine:
    """One AOT plan, deserialised and bound.

    The runtime config carries the two knobs this task cares about:
    ``cuda_graph_strategy`` (whether TensorRT wraps itself in a CUDA graph) and
    ``dynamic_shapes_kernel_specialization_strategy`` (EAGER, so a measurement
    never lands on the generic fallback kernel).
    """

    def __init__(self, trt, logger, plan_path, graph_strategy, spec_strategy,
                 runtime_cache_path, profile_opts, device):

        self.trt = trt
        self.stub = False
        with open(plan_path, "rb") as fh:
            plan = fh.read()
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(plan)
        if self.engine is None:
            raise SystemExit(f"failed to deserialize {plan_path}")

        self.rc = self.engine.create_runtime_config()
        self.rc.cuda_graph_strategy = getattr(trt.CudaGraphStrategy, graph_strategy)
        self.rc.dynamic_shapes_kernel_specialization_strategy = getattr(
            trt.DynamicShapesKernelSpecializationStrategy, spec_strategy
        )
        self.cache = self.rc.create_runtime_cache()
        self.rc.set_runtime_cache(self.cache)
        self.runtime_cache_path = runtime_cache_path
        if runtime_cache_path and os.path.exists(runtime_cache_path):
            with open(runtime_cache_path, "rb") as fh:
                self.cache.deserialize(fh.read())
            self.cache_preloaded = True
        else:
            self.cache_preloaded = False

        self.ctx = self.engine.create_execution_context(runtime_config=self.rc)
        self.profile_opts = list(profile_opts)
        self.device = device
        self.jit_seconds = 0.0
        self.active_profile = None
        self._out = None

    def select_profile(self, m: int, stream):
        """Pick the profile whose opt point is closest to this batch size."""
        idx = min(
            range(len(self.profile_opts)),
            key=lambda i: abs(self.profile_opts[i] - m),
        )
        if idx != self.active_profile:
            self.ctx.set_optimization_profile_async(idx, stream.cuda_stream)
            stream.synchronize()
            self.active_profile = idx
        return idx

    def bind(self, a_q, a_scale, m: int):
        import torch

        self.ctx.set_input_shape("a_q", (m, a_q.shape[1]))
        self.ctx.set_input_shape("a_scale", (m, 1))
        shp = tuple(self.ctx.get_tensor_shape("out"))
        if self._out is None or tuple(self._out.shape) != shp:
            self._out = torch.empty(shp, dtype=torch.bfloat16, device=self.device)
        self.ctx.set_tensor_address("a_q", a_q.data_ptr())
        self.ctx.set_tensor_address("a_scale", a_scale.data_ptr())
        self.ctx.set_tensor_address("out", self._out.data_ptr())
        return self._out

    def bind_fold(self, x, m: int):
        """One dense input, one output. No scale, no quantized activation."""
        import torch

        self.ctx.set_input_shape("x", (m, x.shape[1]))
        shp = tuple(self.ctx.get_tensor_shape("out"))
        if self._out is None or tuple(self._out.shape) != shp:
            self._out = torch.empty(shp, dtype=torch.bfloat16, device=self.device)
        self.ctx.set_tensor_address("x", x.data_ptr())
        self.ctx.set_tensor_address("out", self._out.data_ptr())
        return self._out

    def enqueue(self, stream):
        if not self.ctx.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("execute_async_v3 returned false")
        return self._out

    def save_cache(self):
        if not self.runtime_cache_path:
            return False
        mem = self.cache.serialize()
        if mem is None:
            return False
        os.makedirs(os.path.dirname(self.runtime_cache_path), exist_ok=True)
        with open(self.runtime_cache_path, "wb") as fh:
            fh.write(bytes(mem))
        return True

    def layer_information(self) -> str:
        """What TensorRT actually built -- the record that says whether an INT8
        tactic ran at all. A TensorRT loss caused by a silently-chosen fp32
        tactic is a different finding than a TensorRT loss on equal footing."""
        try:
            insp = self.engine.create_engine_inspector()
            insp.execution_context = self.ctx
            return insp.get_engine_information(self.trt.LayerInformationFormat.JSON)
        except Exception as ex:  # inspector availability is version dependent
            return json.dumps({"inspector_error": f"{type(ex).__name__}: {ex}"})


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------


def time_burst(fn, iters, cuda) -> float:
    import torch

    if cuda:
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1e3 / iters


def calibrate(fn, target_ms, cuda, max_iters) -> tuple:
    """Burst size for a target wall time per burst, plus the probe cost."""
    import torch

    for _ in range(3):
        fn()
    if cuda:
        torch.cuda.synchronize()
    probe = max(time_burst(fn, 3, cuda), 1e-5)
    return max(1, min(max_iters, int(target_ms / probe))), probe


def rounds_for_duration(per_arm_ms, n_arms, min_arm_s, min_point_s, floor) -> int:
    """Round count that satisfies BOTH duration rules.

    per_arm_ms is the wall time of one burst of one arm. One round runs every
    arm once, so a round costs ``per_arm_ms * n_arms``.
    """
    need_arm = math.ceil(min_arm_s * 1e3 / max(per_arm_ms, 1e-6))
    need_point = math.ceil(min_point_s * 1e3 / max(per_arm_ms * n_arms, 1e-6))
    return int(max(floor, need_arm, need_point))


def summarize(samples) -> dict:
    s = sorted(samples)
    return {
        "n": len(s),
        "median_ms": statistics.median(s),
        "p5_ms": s[max(0, int(0.05 * (len(s) - 1)))],
        "p95_ms": s[min(len(s) - 1, int(math.ceil(0.95 * (len(s) - 1))))],
        "min_ms": s[0],
        "max_ms": s[-1],
    }


def capture_graph(fn, iters):
    """A CUDA graph holding ``iters`` back-to-back bodies.

    Warmup on a side stream is required by the capture API, and it is also
    where Triton JIT and TensorRT JIT happen -- neither may occur inside the
    capture.
    """
    import torch

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    return g


# --------------------------------------------------------------------------
# Clock / power annotation
# --------------------------------------------------------------------------


class CardMonitor:
    def __init__(self, enabled: bool):
        self.h = None
        self.err = None
        if not enabled:
            self.err = "disabled"
            return
        try:
            import pynvml
            import torch

            pynvml.nvmlInit()
            self.nvml = pynvml
            want = str(torch.cuda.get_device_properties(0).uuid)
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                uid = pynvml.nvmlDeviceGetUUID(h)
                uid = uid.decode() if isinstance(uid, bytes) else uid
                if want in uid or uid.endswith(want):
                    self.h = h
                    self.index = i
                    self.uuid = uid
                    break
            if self.h is None:  # single-card CUDA_VISIBLE_DEVICES, fall back
                self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.index = 0
                self.uuid = pynvml.nvmlDeviceGetUUID(self.h)
        except Exception as ex:
            self.err = f"{type(ex).__name__}: {ex}"

    def sample(self) -> dict:
        if self.h is None:
            return {"error": self.err}
        p = self.nvml
        try:
            return {
                "sm_clock_mhz": p.nvmlDeviceGetClockInfo(self.h, p.NVML_CLOCK_SM),
                "mem_clock_mhz": p.nvmlDeviceGetClockInfo(self.h, p.NVML_CLOCK_MEM),
                "power_w": round(p.nvmlDeviceGetPowerUsage(self.h) / 1000.0, 1),
                "power_limit_w": round(
                    p.nvmlDeviceGetEnforcedPowerLimit(self.h) / 1000.0, 1
                ),
                "temp_c": p.nvmlDeviceGetTemperature(self.h, p.NVML_TEMPERATURE_GPU),
                "throttle_reasons": p.nvmlDeviceGetCurrentClocksThrottleReasons(self.h),
            }
        except Exception as ex:
            return {"error": f"{type(ex).__name__}: {ex}"}


# --------------------------------------------------------------------------
# Chain execution
# --------------------------------------------------------------------------


class Chain:
    """One target instantiated at one batch size, in one arm.

    Holds the buffers so a CUDA graph capture sees stable addresses, and exposes
    a single zero-argument callable per arm.
    """

    def __init__(self, target, engines, kn, m, device, dtype, seed, use_trt):
        import torch

        self.target = target
        self.kn = kn
        self.m = m
        self.device = device
        self.use_trt = use_trt
        self.engines = engines
        k0 = target.engine_stages[0].gemm.k
        g = torch.Generator(device="cpu").manual_seed(seed)
        # CPU-sampled, then moved: on-device randn is not architecture-identical
        # across sm86 and sm120, and both cards must see the same input bytes.
        self.x = (
            torch.randn(m, k0, generator=g, dtype=torch.float32).to(device).to(dtype)
        )
        self.stream = torch.cuda.current_stream() if device.type == "cuda" else None
        self.fold_engine = None
        self.exact_weights = {}
        self.x_fold = self.x

    def attach_fold(self, engine, fold_dtype):
        """Bind a fold engine and the input cast its network declares.

        The input is cast, not re-sampled: both arms must see the same numbers,
        and a fold arm fed different randoms would be measuring the randoms.
        """
        import torch

        self.fold_engine = engine
        self.x_fold = self.x.to(getattr(torch, fold_dtype)).contiguous()
        return self

    def attach_exact(self, weights):
        self.exact_weights = weights
        return self

    def _engine_for(self, stage):
        return self.engines[stage.name]

    @staticmethod
    def _bridge(cur, width):
        """Narrow to the width the next stage consumes.

        Stands in for the excluded attention core. Identical in both arms, so it
        cancels out of every ratio; ``contiguous`` because the quant kernel wants
        a packed row.
        """
        return cur[:, :width].contiguous()

    def torch_step(self):
        """The part exactly as the serving path runs it."""
        cur = self.x
        for st in self.target.stages:
            if st.kind == "quant":
                self.q, self.s = self.kn.per_token_quant_int8(cur)
                cur = None
            elif st.kind == "bridge":
                cur = self._bridge(cur, st.out_width)
            else:
                e = self._engine_for(st)
                out = self.kn.int8_scaled_mm(
                    self.q, e.w_t, self.s, e.w_scale, out_dtype=self.x.dtype
                )
                if e.epilogue == "silu_mul":
                    out = self.kn.silu_and_mul(out)
                cur = out
        return cur

    def fold_step(self):
        """The whole part as ONE engine call. No quant kernel runs at all.

        This is what the fold buys structurally, before any timing: the INT8
        arms are per-stage engine islands with our quant kernel between them,
        because TensorRT cannot express a per-token dynamic scale. Folded, there
        is nothing left to express -- one input, one engine, one graph node.
        """
        e = self.fold_engine
        if e.stub:
            return e.run_fold(self.x_fold)
        e.bind_fold(self.x_fold, self.m)
        return e.enqueue(self.stream)

    def exact_reference(self):
        """fp32 ground truth from the dequantized weights.

        w_q * w_scale is exact -- it IS the checkpoint's weight, just in a wider
        container. Computing the chain from it in fp32 gives the value both the
        deployed path and the fold are approximating, so each one's error can be
        measured against the same target instead of against each other. That is
        what turns "the fold is quality-neutral" from a claim into a number.
        """
        import torch

        cur = self.x.float()
        for st in self.target.stages:
            if st.kind == "quant":
                continue
            if st.kind == "bridge":
                cur = cur[:, : st.out_width].contiguous()
                continue
            w = self.exact_weights[st.name]
            cur = cur @ w.t()
            if st.gemm.epilogue == "silu_mul":
                half = cur.shape[-1] // 2
                gate, up = cur[:, :half], cur[:, half:]
                cur = torch.nn.functional.silu(gate) * up
        return cur

    def trt_step(self):
        """The same part with every GEMM stage served by an engine."""
        cur = self.x
        for st in self.target.stages:
            if st.kind == "quant":
                self.q, self.s = self.kn.per_token_quant_int8(cur)
                cur = None
            elif st.kind == "bridge":
                cur = self._bridge(cur, st.out_width)
            else:
                e = self._engine_for(st)
                if e.stub:
                    cur = e.run(self.q, self.s)
                else:
                    e.bind(self.q, self.s.float(), self.m)
                    cur = e.enqueue(self.stream)
        return cur


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def _dense_weight(spec, geo, model_dir, random_weights, seed, device, torch_dtype):
    """w_q x per-channel scale, in the fold's storage dtype.

    Same loader and same slicing as the INT8 arms, so the fold arm and the
    deployed arm differ in exactly one thing: where the dequantize happens.
    """
    import torch

    w = tgt.load_gemm_weights(spec, geo, model_dir, random_weights, seed)
    dense = w.q.to(torch.float32) * w.scale.reshape(-1, 1).to(torch.float32)
    return dense.to(getattr(torch, torch_dtype)).to(device)


def _exact_weights(target, geo, model_dir, random_weights, seed, device):
    """fp32 dequantized weights for the exact reference."""
    return {
        st.name: _dense_weight(
            st.gemm, geo, model_dir, random_weights, seed, device, "float32"
        )
        for st in target.engine_stages
    }


def build_torch_weights(target, manifest, rank, geo, model_dir, random_weights,
                        seed, device):
    """Weight objects the torch arm uses -- the SAME tensors the engines carry.

    Loaded from the same source with the same slicing, so a tolerance difference
    can only come from the kernels, never from the operands.
    """

    out = {}
    for st in target.engine_stages:
        w = tgt.load_gemm_weights(st.gemm, geo, model_dir, random_weights, seed)
        obj = type("W", (), {})()
        # serving layout: the INT8 scheme stores weight.t(), so mat_b is (k,n)
        obj.w_t = w.q.to(device).t()
        obj.w_scale = w.scale.reshape(-1, 1).to(device)
        obj.epilogue = st.gemm.epilogue
        obj.stub = True
        obj.run = None
        out[st.name] = obj
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--config", default=tgt.DEFAULT_CONFIG)
    ap.add_argument(
        "--model-dir",
        default="/spinning/llm_stuff/club-3090/models-cache/Qwen3.6-27B-INT8-W8A8",
    )
    ap.add_argument("--tp-size", type=int, default=3)
    ap.add_argument("--ratio", default="30,17,17")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--optional", default="")
    ap.add_argument("--targets", default="", help="comma list; default all")
    ap.add_argument("--m", default="1,2,4,8")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--a2-arms", default=DEFAULT_A2)
    ap.add_argument("--rounds-floor", type=int, default=9)
    ap.add_argument("--target-ms", type=float, default=20.0)
    ap.add_argument("--max-iters", type=int, default=4000)
    ap.add_argument("--min-arm-seconds", type=float, default=1.0)
    ap.add_argument("--min-point-seconds", type=float, default=10.0)
    ap.add_argument("--graph-bodies", type=int, default=32)
    ap.add_argument("--tolerance", type=float, default=2e-2)
    ap.add_argument("--kernel-specialization", default="EAGER",
                    choices=("EAGER", "LAZY", "NONE"))
    ap.add_argument("--runtime-cache-dir", default="")
    ap.add_argument("--random-weights", action="store_true")
    ap.add_argument("--seed", type=int, default=337)
    ap.add_argument("--out", default="")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--no-nvml", action="store_true")
    ap.add_argument(
        "--quality-reference",
        action="store_true",
        help="compute the exact fp32 reference even when no fold arm runs",
    )
    a = ap.parse_args(argv)

    import torch

    cuda = torch.cuda.is_available() and not a.mock
    device = torch.device("cuda" if cuda else "cpu")
    dtype = torch.bfloat16
    kn = load_kernels(a.mock)
    ratio = [int(x) for x in a.ratio.split(",")]
    ms = [int(x) for x in a.m.split(",")]
    arms = [x for x in a.arms.split(",") if x]
    a2_arms = [x for x in a.a2_arms.split(",") if x]
    if a.mock:
        # CUDA-graph arms cannot run on a CPU; everything else can, including
        # the fold arms, which are the point of the mock now that they exist.
        keep = ("torch_eager", "trt_enqueue", *FOLD_ARMS)
        arms = [x for x in arms if x in keep]
        a2_arms = [x for x in a2_arms if x in arms] or ["torch_eager"]

    geo = tgt.derive_geometry(a.config, a.tp_size, ratio, a.rank)
    all_targets = tgt.build_targets(geo, [x for x in a.optional.split(",") if x])
    if a.targets:
        want = set(a.targets.split(","))
        all_targets = [t for t in all_targets if t.name in want]

    with open(os.path.join(a.engine_dir, "manifest.json")) as fh:
        manifest = json.load(fh)
    profile_opts = manifest.get("profile_opts", [1, 4])

    mon = CardMonitor(cuda and not a.no_nvml)
    trt = None
    logger = None
    if not a.mock:
        import tensorrt_rtx as _trt  # noqa: PLC0415

        trt = _trt
        logger = trt.Logger(trt.Logger.WARNING)

    out = {
        "task": "337",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mock": a.mock,
        "environment": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            "device_name": torch.cuda.get_device_name(0) if cuda else "cpu",
            "capability": (
                "sm%d%d" % torch.cuda.get_device_capability(0) if cuda else "cpu"
            ),
            "tensorrt_rtx": trt.__version__ if trt else None,
            "kernels_stubbed": kn.stub,
            "kernels_missing": kn.missing,
        },
        "engine_manifest": {
            k: manifest.get(k)
            for k in ("compute_capabilities", "profile_opts", "dq_dtype",
                      "cuda_visible_devices", "random_weights", "mock")
        },
        "settings": {
            "rank": a.rank,
            "arch": geo.arch,
            "ratio": ratio,
            "m": ms,
            "arms": arms,
            "a2_arms": a2_arms,
            "target_ms": a.target_ms,
            "min_arm_seconds": a.min_arm_seconds,
            "min_point_seconds": a.min_point_seconds,
            "graph_bodies": a.graph_bodies,
            "tolerance": a.tolerance,
            "kernel_specialization": a.kernel_specialization,
            "seed": a.seed,
        },
        "partition_provenance": geo.partition_provenance,
        "shapes": tgt.shape_table(all_targets),
        "jit_warmup": [],
        "results": [],
    }

    t_all = time.time()
    for target in all_targets:
        # ---- engines (or stubs) for this target, loaded once ----
        engines = {}
        tw = build_torch_weights(
            target, manifest, a.rank, geo, a.model_dir, a.random_weights,
            a.seed, device,
        )
        for st in target.engine_stages:
            plan = os.path.join(
                a.engine_dir, f"rank{a.rank}_{geo.arch}_{st.gemm.name}.plan"
            )
            if not os.path.exists(plan):
                raise SystemExit(f"missing engine {plan}; run build_engines.py first")
            with open(plan, "rb") as fh:
                head = fh.read(len(MOCK_MAGIC))
            if head == MOCK_MAGIC or a.mock:
                w = tgt.load_gemm_weights(
                    st.gemm, geo, a.model_dir, a.random_weights, a.seed
                )
                engines[st.name] = StubEngine(
                    {"n": st.gemm.n, "k": st.gemm.k, "epilogue": st.gemm.epilogue},
                    w.q.to(device), w.scale.reshape(-1, 1).to(device), kn,
                )
            else:
                cache_path = ""
                if a.runtime_cache_dir:
                    cache_path = os.path.join(
                        a.runtime_cache_dir,
                        f"{out['environment']['capability']}_"
                        f"rank{a.rank}_{st.gemm.name}.cache",
                    )
                t0 = time.time()
                eng = TrtEngine(
                    trt, logger, plan, "DISABLED", a.kernel_specialization,
                    cache_path, profile_opts, device,
                )
                engines[st.name] = eng
                out["jit_warmup"].append(
                    {
                        "target": target.name,
                        "stage": st.name,
                        "deserialize_seconds": round(time.time() - t0, 3),
                        "runtime_cache_preloaded": eng.cache_preloaded,
                        "runtime_cache_path": cache_path,
                    }
                )

        # ---- fold engines: one per variant for the WHOLE target ----
        fold_engines = {}
        for arm, variant in FOLD_ARMS.items():
            if arm not in arms:
                continue
            plan = os.path.join(
                a.engine_dir,
                f"rank{a.rank}_{geo.arch}_{target.name}.{variant}.plan",
            )
            if not os.path.exists(plan):
                # fp32_ref is built for one shape only, by design.
                out.setdefault("skipped_fold_arms", []).append(
                    {"target": target.name, "arm": arm, "reason": "no engine built"}
                )
                continue
            with open(plan, "rb") as fh:
                head = fh.read(len(MOCK_MAGIC))
            torch_dtype = FOLD_TORCH_DTYPE[variant]
            if head == MOCK_MAGIC or a.mock:
                dense = {
                    st.name: _dense_weight(
                        st.gemm, geo, a.model_dir, a.random_weights, a.seed,
                        device, torch_dtype,
                    )
                    for st in target.engine_stages
                }
                fold_engines[arm] = (StubFoldEngine(target, dense, torch_dtype),
                                     torch_dtype)
            else:
                cache_path = ""
                if a.runtime_cache_dir:
                    cache_path = os.path.join(
                        a.runtime_cache_dir,
                        f"{out['environment']['capability']}_"
                        f"rank{a.rank}_{target.name}.{variant}.cache",
                    )
                t0 = time.time()
                eng = TrtEngine(
                    trt, logger, plan, "DISABLED", a.kernel_specialization,
                    cache_path, profile_opts, device, mode="fold",
                )
                fold_engines[arm] = (eng, torch_dtype)
                out["jit_warmup"].append(
                    {
                        "target": target.name,
                        "stage": f"{target.name}.{variant}",
                        "deserialize_seconds": round(time.time() - t0, 3),
                        "runtime_cache_preloaded": eng.cache_preloaded,
                        "runtime_cache_path": cache_path,
                    }
                )

        for m in ms:
            point = run_point(
                target, engines, tw, kn, m, device, dtype, arms, a2_arms, a,
                cuda, mon, trt, logger, profile_opts, out, fold_engines, geo,
            )
            point["target"] = target.name
            point["m"] = m
            out["results"].append(point)
            v = point.get("verdict", {})
            print(
                f"  {target.name:22s} M={m:<3d} "
                f"torch_graph={_med(point,'torch_graph'):8.4f} ms  "
                f"trt_outer_graph={_med(point,'trt_outer_graph'):8.4f} ms  "
                f"ratio={v.get('trt_over_torch_graph', float('nan')):6.3f}  "
                f"floor={v.get('a2_floor_frac', float('nan')):6.3f}  "
                f"maxdiff={point.get('tolerance',{}).get('max_abs_diff','-')}"
            )

        for e in engines.values():
            if not getattr(e, "stub", False) and e.save_cache():
                pass

    out["total_wall_s"] = round(time.time() - t_all, 1)
    text = json.dumps(out, indent=1)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            fh.write(text)
        print(f"\n-> {a.out}  ({out['total_wall_s']}s)")
    else:
        print(text)
    return 0


def _med(point, arm):
    r = point.get("arms", {}).get(arm)
    return r["median_ms"] if r else float("nan")


def run_point(target, engines, tw, kn, m, device, dtype, arms, a2_arms, a,
              cuda, mon, trt, logger, profile_opts, out, fold_engines=None,
              geo=None) -> dict:
    """One (target, M) operating point: every arm, interleaved, same rotation."""
    import torch

    rec = {"arms": {}, "clocks": {}, "notes": []}
    rec["clocks"]["before"] = mon.sample()

    def make_chain(seed_off, use_trt):
        return Chain(target, engines if use_trt else tw, kn, m, device, dtype,
                     a.seed + seed_off, use_trt)

    # Bind engines once at this M, and pay TensorRT's JIT specialisation OUTSIDE
    # any timed region -- it is reported separately, never folded into steady
    # state.
    jit = {}
    if not a.mock:
        stream = torch.cuda.current_stream()
        for st in target.engine_stages:
            e = engines[st.name]
            if e.stub:
                continue
            e.select_profile(m, stream)
        c = make_chain(0, True)
        torch.cuda.synchronize()
        t0 = time.time()
        c.trt_step()
        torch.cuda.synchronize()
        jit["first_execution_seconds"] = round(time.time() - t0, 4)
        t0 = time.time()
        c.trt_step()
        torch.cuda.synchronize()
        jit["second_execution_seconds"] = round(time.time() - t0, 4)
        rec["jit"] = jit

    # ---- callables per arm ----
    builders = {}
    if "torch_eager" in arms:
        builders["torch_eager"] = lambda off: make_chain(off, False).torch_step
    if "trt_enqueue" in arms:
        builders["trt_enqueue"] = lambda off: make_chain(off, True).trt_step

    def graphed(fn):
        g = capture_graph(fn, a.graph_bodies)
        return lambda: g.replay(), a.graph_bodies

    lanes = {}
    for name in arms:
        copies = [""] + (["#A2"] if name in a2_arms else [])
        for ci, suffix in enumerate(copies):
            off = 0 if ci == 0 else 977
            try:
                if name == "torch_eager":
                    fn, div = make_chain(off, False).torch_step, 1
                elif name == "trt_enqueue":
                    fn, div = make_chain(off, True).trt_step, 1
                elif name == "torch_graph":
                    if not cuda:
                        continue
                    fn, div = graphed(make_chain(off, False).torch_step)
                elif name in ("trt_native_graph", "trt_outer_graph"):
                    if not cuda or a.mock:
                        continue
                    if name == "trt_native_graph":
                        # TensorRT wraps ITSELF in a CUDA graph; our quant stays
                        # eager between engine calls.
                        rebuilt = {}
                        for st in target.engine_stages:
                            e = engines[st.name]
                            plan = e_plan_path(a, out, st)
                            ne = TrtEngine(
                                trt, logger, plan, "WHOLE_GRAPH_CAPTURE",
                                a.kernel_specialization, "", profile_opts, device,
                            )
                            ne.select_profile(m, torch.cuda.current_stream())
                            rebuilt[st.name] = ne
                        ch = Chain(target, rebuilt, kn, m, device, dtype,
                                   a.seed + off, True)
                        for _ in range(3):
                            ch.trt_step()
                        torch.cuda.synchronize()
                        fn, div = ch.trt_step, 1
                    else:
                        # The deployment form: engines with their own graph
                        # capture DISABLED, and the whole part -- our quant
                        # kernels included -- captured in one outer graph.
                        fn, div = graphed(make_chain(off, True).trt_step)
                elif name in FOLD_ARMS:
                    if not cuda and not a.mock:
                        continue
                    if not fold_engines or name not in fold_engines:
                        continue
                    eng, fold_dtype = fold_engines[name]
                    ch = make_chain(off, True).attach_fold(eng, fold_dtype)
                    if not a.mock and not eng.stub:
                        eng.select_profile(m, torch.cuda.current_stream())
                        for _ in range(3):
                            ch.fold_step()
                        torch.cuda.synchronize()
                    if cuda:
                        fn, div = graphed(ch.fold_step)
                    else:
                        fn, div = ch.fold_step, 1
                else:
                    continue
            except Exception as ex:
                rec["notes"].append(f"{name}{suffix}: not runnable: "
                                    f"{type(ex).__name__}: {ex}")
                continue
            lanes[name + suffix] = (fn, div)

    if not lanes:
        rec["notes"].append("no runnable arm at this point")
        return rec

    # ---- calibration and the duration rule ----
    iters = {}
    probes = {}
    for name, (fn, div) in lanes.items():
        it, probe = calibrate(fn, a.target_ms, cuda, a.max_iters)
        iters[name] = it
        probes[name] = probe
    slowest = max(probes[n] * iters[n] for n in lanes)
    rounds = rounds_for_duration(
        slowest, len(lanes), a.min_arm_seconds, a.min_point_seconds, a.rounds_floor
    )
    rec["calibration"] = {
        "iters": iters,
        "probe_ms": {k: round(v, 6) for k, v in probes.items()},
        "rounds": rounds,
        "predicted_point_seconds": round(slowest * len(lanes) * rounds / 1e3, 2),
    }

    # ---- interleaved rotation: one burst per arm per round ----
    samples = {n: [] for n in lanes}
    rotation = list(lanes)
    t_meas0 = time.time()
    for _ in range(rounds):
        for name in rotation:
            fn, div = lanes[name]
            samples[name].append(time_burst(fn, iters[name], cuda) / div)
    rec["measured_seconds"] = round(time.time() - t_meas0, 2)
    rec["duration_rule_met"] = rec["measured_seconds"] >= a.min_point_seconds

    for name, s in samples.items():
        rec["arms"][name] = summarize(s)
        rec["arms"][name]["iters"] = iters[name]
        rec["arms"][name]["stub"] = bool(a.mock)

    # ---- A-vs-A floor and the verdict ----
    verdict = {}
    for name in a2_arms:
        if name in rec["arms"] and name + "#A2" in rec["arms"]:
            x, y = rec["arms"][name]["median_ms"], rec["arms"][name + "#A2"]["median_ms"]
            verdict[f"a2_spread_{name}"] = abs(x - y) / max(x, y, 1e-12)
    verdict["a2_floor_frac"] = max(
        [v for k, v in verdict.items() if k.startswith("a2_spread_")] or [float("nan")]
    )
    for num, den, key in (
        ("trt_outer_graph", "torch_graph", "trt_over_torch_graph"),
        ("torch_eager", "torch_graph", "our_launch_axis"),
        ("trt_enqueue", "trt_native_graph", "trt_launch_axis"),
        ("trt_outer_graph", "torch_eager", "DO_NOT_QUOTE_trt_graph_over_eager"),
        # The fold crossover. Both sides are graph-captured, so what is left is
        # exactly the trade the fold makes: one engine and zero activation-quant
        # kernels, against 2x the weight bytes moved in a memory-bound GEMV.
        ("trt_fold_bf16_graph", "torch_graph", "fold_bf16_over_torch_graph"),
        ("trt_fold_fp16_graph", "torch_graph", "fold_fp16_over_torch_graph"),
        ("trt_fold_bf16_graph", "trt_outer_graph", "fold_bf16_over_trt_int8"),
        ("trt_fp32_ref_graph", "torch_graph", "DIAGNOSTIC_fp32_ref_over_torch_graph"),
    ):
        if num in rec["arms"] and den in rec["arms"]:
            verdict[key] = (
                rec["arms"][num]["median_ms"] / rec["arms"][den]["median_ms"]
            )
    r = verdict.get("trt_over_torch_graph")
    f = verdict.get("a2_floor_frac")
    if r is not None and f == f:
        verdict["inside_noise_floor"] = abs(1.0 - r) <= f
    verdict["reading"] = (
        "trt_over_torch_graph is the verdict for the same-precision question: "
        "< 1.0 means the engine wins with the launch axis removed from BOTH "
        "sides. fold_*_over_torch_graph is the fold crossover: the fold removes "
        "every activation-quant kernel (78 % of M=1 eager INT8 linear time per "
        "#368, though graph replay already shrinks that share to ~11 %) and "
        "pays 2x the weight bytes in a memory-bound GEMV -- so it should win at "
        "small M and lose as M grows, and WHERE it crosses decides which parts "
        "get the fold. DO_NOT_QUOTE_trt_graph_over_eager double counts the "
        "CUDA-graph win. DIAGNOSTIC_fp32_ref_* is not a deployment candidate at "
        "4 bytes per weight; it isolates TensorRT runtime quality from the "
        "quantization constraints and nothing else."
    )
    rec["verdict"] = verdict

    # ---- tolerance, and the quality question the fold arms raise ----
    rec["tolerance"] = compare_outputs(target, engines, tw, kn, m, device, dtype,
                                       a, cuda)
    if geo is not None and (fold_engines or a.quality_reference):
        rec["quality"] = compare_quality(
            target, engines, tw, kn, fold_engines or {}, m, device, dtype, a,
            geo, cuda,
        )
    rec["clocks"]["after"] = mon.sample()
    return rec


def e_plan_path(a, out, st):
    return os.path.join(
        a.engine_dir,
        f"rank{a.rank}_{out['settings']['arch']}_{st.gemm.name}.plan",
    )


def compare_quality(target, engines, tw, kn, fold_engines, m, device, dtype, a,
                    geo, cuda) -> dict:
    """Every path's error against the SAME exact fp32 reference.

    The fold's quality argument is that it removes error rather than adding it:
    the weight grid is untouched (w_q * w_scale is exact, just stored wider) and
    the per-token activation quantization disappears entirely. That is a
    testable claim, so it is tested rather than asserted.

    The gate is not a fixed tolerance -- it is RELATIVE TO WHAT WE DEPLOY: a fold
    passes if its error against exact is no larger than the deployed INT8 path's
    error against exact. Comparing the fold to the deployed OUTPUT instead would
    measure the deployed path's own quantization error and call it a fold defect.
    """
    import torch

    try:
        exact_w = _exact_weights(
            target, geo, a.model_dir, a.random_weights, a.seed, device
        )
        ref_chain = Chain(target, tw, kn, m, device, dtype, a.seed, False)
        ref_chain.attach_exact(exact_w)
        with torch.no_grad():
            exact = ref_chain.exact_reference()
            deployed = ref_chain.torch_step().float()
        if cuda:
            torch.cuda.synchronize()

        scale = exact.abs().amax().clamp(min=1e-6)

        def err(v):
            return float((v - exact).abs().amax() / scale)

        out = {
            "exact_absmax": float(scale),
            "deployed_int8_err_vs_exact": err(deployed),
            "note": (
                "errors are max-abs against an fp32 chain computed from the "
                "dequantized checkpoint weights. A fold arm is quality-neutral "
                "if its error is <= the deployed path's."
            ),
            "arms": {},
        }
        for arm, (eng, fold_dtype) in fold_engines.items():
            ch = Chain(target, engines, kn, m, device, dtype, a.seed, True)
            ch.attach_fold(eng, fold_dtype)
            with torch.no_grad():
                got = ch.fold_step().float()
            if cuda:
                torch.cuda.synchronize()
            finite = bool(torch.isfinite(got).all())
            e = err(got) if finite else float("nan")
            entry = {
                "err_vs_exact": e,
                "storage_dtype": fold_dtype,
                "finite": finite,
                "at_least_as_accurate_as_deployed": (
                    finite and e <= out["deployed_int8_err_vs_exact"]
                ),
                "err_vs_deployed_output": (
                    float((got - deployed).abs().amax() / scale)
                    if finite else float("nan")
                ),
            }
            if not finite:
                # fp16 keeps 11 mantissa bits but only 5 exponent bits, so its
                # max representable value is 65504. bf16 has fp32's exponent
                # range and cannot overflow where fp32 does not. In a chain of
                # several GEMMs the intermediate magnitudes compound, and that
                # is where the difference stops being academic. Recorded as a
                # RESULT about fp16 in this chain, not swallowed.
                entry["overflow"] = (
                    "non-finite output: fp16 exponent range exceeded somewhere "
                    "in the chain. bf16 carries fp32's exponent range and is "
                    "not exposed to this."
                )
                entry["max_abs_finite_intermediate"] = float(
                    got[torch.isfinite(got)].abs().amax()
                ) if torch.isfinite(got).any() else float("inf")
            out["arms"][arm] = entry
        return out
    except Exception as ex:
        return {"error": f"{type(ex).__name__}: {ex}"}


def compare_outputs(target, engines, tw, kn, m, device, dtype, a, cuda) -> dict:
    """Same input through both paths; report the difference, gate on it.

    Byte identity is not the claim -- the epilogue scale order differs. What the
    gate protects against is an engine that is fast because it is computing
    something else.
    """
    import torch

    try:
        ct = Chain(target, tw, kn, m, device, dtype, a.seed, False)
        ce = Chain(target, engines, kn, m, device, dtype, a.seed, True)
        with torch.no_grad():
            ref = ct.torch_step().float()
            got = ce.trt_step().float()
        if cuda:
            torch.cuda.synchronize()
        diff = (ref - got).abs()
        denom = ref.abs().amax().clamp(min=1e-6)
        max_abs = float(diff.amax())
        max_rel = float(max_abs / denom)
        cos = float(
            torch.nn.functional.cosine_similarity(
                ref.flatten(), got.flatten(), dim=0
            )
        )
        return {
            "max_abs_diff": max_abs,
            "max_rel_diff": max_rel,
            "cosine": cos,
            "ref_absmax": float(denom),
            "bound": a.tolerance,
            "pass": max_rel <= a.tolerance,
            "note": (
                "bf16 output; differences below the bound are epilogue ordering, "
                "not different arithmetic. A failure is recorded and the run "
                "continues -- a fast wrong engine is a result."
            ),
        }
    except Exception as ex:
        return {"error": f"{type(ex).__name__}: {ex}"}


if __name__ == "__main__":
    raise SystemExit(main())
