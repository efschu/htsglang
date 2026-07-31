# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Inference backends for the convolutional stages of the chain.

Two are implemented here, both against the same narrow protocol:

``OnnxRuntimeBackend``
    ONNX Runtime with either the CUDA execution provider (fp32, the parity
    reference) or the TensorRT execution provider (fp16, the production
    path). The TensorRT EP builds and caches a real TensorRT engine from the
    ONNX; it is a *dependency* rather than the source extraction DESIGN #333
    §9.3 anticipated, taken because the reuse order in that same section is
    dependency before port and this dependency did not exist in the
    prior-art survey. The extraction remains the route to features the EP
    does not expose -- see ``NativeTensorRTBackend``.

``NativeTensorRTBackend``
    The seam for the §9.3 extraction from ``efschu/vs-mlrt`` (``trt_utils.h``
    is already VapourSynth-free; ``inference_helper.h`` needs one call
    rewritten from ``vs_bitblt`` to ``cudaMemcpy2D``). Not built in this
    milestone; constructing it raises with that statement rather than
    pretending.

Both keep tensors on the device. ONNX Runtime's IO binding takes raw device
pointers, so a torch CUDA tensor is bound directly and no frame crosses PCIe
inside a stage.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sglang.srt.video_enhance.engine_cache import (
    EngineCache,
    EngineKey,
    ShapeTriplet,
    sha256_file,
)

logger = logging.getLogger(__name__)


class BackendUnavailable(RuntimeError):
    """The requested backend cannot run in this process."""


@dataclass(frozen=True)
class BackendInfo:
    """What actually ran, recorded so a measurement can be attributed."""

    runtime: str
    runtime_version: str
    #: The provider that was *requested*.
    provider: str
    precision: str
    engine_path: str | None
    built_engine: bool
    build_seconds: float | None
    #: The providers onnxruntime actually bound, in its own priority order.
    #: Empty only for a backend that does not go through a session.
    active_providers: tuple[str, ...] = ()
    #: True when the requested provider is not among the active ones -- the
    #: session fell back. A measurement taken through such a session does not
    #: describe the provider its label claims.
    provider_fell_back: bool = False

    @property
    def effective_provider(self) -> str:
        """What to attribute a measurement to."""
        return self.active_providers[0] if self.active_providers else self.provider


class InferenceBackend(Protocol):
    info: BackendInfo

    def run(self, tensor):  # -> torch.Tensor
        """Run the network on one NCHW device tensor and return an NCHW device tensor."""

    def close(self) -> None: ...


# --------------------------------------------------------------------------
# ONNX Runtime
# --------------------------------------------------------------------------

#: Mirrors the tactic and optimisation choices of the vs-mlrt / VSGAN trtexec
#: recipe recorded in ANALYSE_333 §1, expressed as TensorRT-EP options.
DEFAULT_TRT_EP_OPTIONS: dict[str, object] = {
    "trt_builder_optimization_level": 5,
    "trt_timing_cache_enable": True,
    "trt_engine_cache_enable": True,
}

#: CUDA execution provider options. ``cudnn_conv_algo_search`` defaults to
#: ``EXHAUSTIVE`` in ONNX Runtime, which benchmarks the available convolution
#: algorithms on the first call for each shape and keeps the fastest. That
#: choice is made from wall-clock timings, so two processes running the same
#: graph on the same card can select different algorithms and produce results
#: that differ in the last bits.
#:
#: Measured cost of leaving it at the default: a chunked run and a whole-clip
#: run of the same 96-frame source, same card, provably identical decoded
#: input, differed in 14 of 191 output frames. The pixels agree to within the
#: 8-bit output step almost everywhere, and occasionally one code value tips
#: -- which is enough to make "same input, same output" false and to make any
#: comparison between two chunkings unanswerable.
#:
#: ``HEURISTIC`` picks by shape rather than by stopwatch, so the choice is a
#: function of the graph and reproduces. It is not free -- the heuristic can
#: pick a slower algorithm than a benchmark would -- and that is the trade
#: taken here, deliberately: a reproducible chain is worth more than the last
#: few percent of an SR stage, and the multi-card seam cannot be verified at
#: all without it.
CUDA_EP_OPTIONS: dict[str, object] = {
    "cudnn_conv_algo_search": "HEURISTIC",
}


class OnnxRuntimeBackend:
    """ONNX Runtime session bound to device memory.

    ``precision="fp32"`` with the CUDA provider is the reference every engine
    is graded against by the parity gate. ``precision="fp16"`` with the
    TensorRT provider is the production path; note that the pinned SR artifact
    ``realesr-general-wdn-x4v3_opset16.onnx`` is distributed fp32-only, so
    fp16 comes from the builder downcasting at build time, exactly as the
    prior-art recipe does for every model regardless of source precision.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        provider: str = "tensorrt",
        precision: str = "fp16",
        shapes: ShapeTriplet | None = None,
        device_id: int = 0,
        cache: EngineCache | None = None,
        engine_key: EngineKey | None = None,
        input_name: str | None = None,
        output_name: str | None = None,
        ep_options: dict | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise BackendUnavailable(
                "onnxruntime-gpu is required for OnnxRuntimeBackend"
            ) from exc

        self._ort = ort
        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.is_file():
            raise FileNotFoundError(self.onnx_path)
        if precision not in ("fp32", "fp16"):
            raise ValueError("precision must be fp32 or fp16")
        if provider not in ("cuda", "tensorrt"):
            raise ValueError("provider must be cuda or tensorrt")
        if provider == "cuda" and precision != "fp32":
            # The CUDA EP runs the graph as exported. The pinned SR artifact is
            # fp32-only, so asking for fp16 here would silently be ignored.
            raise ValueError(
                "the CUDA provider runs the ONNX at its exported precision; "
                "use provider='tensorrt' for an fp16 engine"
            )

        self.provider = provider
        self.precision = precision
        self.device_id = device_id
        self.shapes = shapes
        self._cache = cache
        self._key = engine_key

        available = ort.get_available_providers()
        provider_name = (
            "TensorrtExecutionProvider"
            if provider == "tensorrt"
            else "CUDAExecutionProvider"
        )
        if provider_name not in available:
            raise BackendUnavailable(
                f"{provider_name} not available; onnxruntime reports {available}"
            )

        options: list = []
        engine_dir: Path | None = None
        if provider == "tensorrt":
            trt_opts = dict(DEFAULT_TRT_EP_OPTIONS)
            trt_opts.update(ep_options or {})
            trt_opts["device_id"] = device_id
            trt_opts["trt_fp16_enable"] = precision == "fp16"
            if cache is not None:
                engine_dir = Path(cache.root) / "ort_trt"
                engine_dir.mkdir(parents=True, exist_ok=True)
                trt_opts["trt_engine_cache_path"] = str(engine_dir)
                trt_opts["trt_timing_cache_path"] = str(engine_dir)
            if shapes is not None:
                name = input_name or _first_input_name(ort, self.onnx_path)
                trt_opts["trt_profile_min_shapes"] = _profile(name, shapes.min_wh)
                trt_opts["trt_profile_opt_shapes"] = _profile(name, shapes.opt_wh)
                trt_opts["trt_profile_max_shapes"] = _profile(name, shapes.max_wh)
            options.append((provider_name, trt_opts))
            options.append(
                ("CUDAExecutionProvider", dict(CUDA_EP_OPTIONS, device_id=device_id))
            )
        else:
            options.append((provider_name, dict(CUDA_EP_OPTIONS, device_id=device_id)))

        so = ort.SessionOptions()
        so.log_severity_level = 3
        started = time.perf_counter()
        self.session = ort.InferenceSession(
            str(self.onnx_path), sess_options=so, providers=options
        )
        elapsed = time.perf_counter() - started

        self.input_name = input_name or self.session.get_inputs()[0].name
        self.output_name = output_name or self.session.get_outputs()[0].name
        self._input_np_dtype = _np_dtype_of(self.session.get_inputs()[0].type)
        self._input_torch_dtype = _torch_dtype_of(self.session.get_inputs()[0].type)
        self._warned_about_cast = False

        engine_path = None
        built = False
        if engine_dir is not None:
            engines = sorted(engine_dir.glob("*.engine"))
            if engines:
                engine_path = str(engines[-1])
                # The EP writes the engine during session construction on a
                # miss, so a session that took minutes built one.
                built = elapsed > 20.0
            if cache is not None and self._key is not None and engine_path:
                self._record_provenance(engine_path, elapsed, built)

        # The provider that actually ran, not the one that was asked for.
        # A TensorRT session lists CUDAExecutionProvider as a fallback so a
        # subgraph the EP cannot take still runs -- which also means that on a
        # host with no TensorRT libraries the whole session silently becomes a
        # CUDA one. It works, it is fast, and every record it produces is
        # labelled "tensorrt". That is how a parity table can come to contain
        # TensorRT numbers taken on a machine where libnvinfer was never
        # installed. Recording what ORT chose makes the fallback visible in
        # the artifact rather than only in a log line nobody kept.
        active = list(self.session.get_providers())
        fell_back = provider_name not in active
        if fell_back:
            logger.warning(
                "%s was requested but onnxruntime is running this session on %s; "
                "any measurement taken through it describes %s, not %s",
                provider_name,
                active[0] if active else "no provider",
                active[0] if active else "no provider",
                provider_name,
            )

        self.info = BackendInfo(
            runtime="onnxruntime",
            runtime_version=ort.__version__,
            provider=provider_name,
            active_providers=tuple(active),
            provider_fell_back=fell_back,
            precision=precision,
            engine_path=engine_path,
            built_engine=built,
            build_seconds=elapsed if built else None,
        )

    def _record_provenance(self, engine_path: str, elapsed: float, built: bool) -> None:
        assert self._cache is not None and self._key is not None
        try:
            self._cache.store(
                self._key,
                None,
                built_path=engine_path,
                source_artifact={
                    "path": str(self.onnx_path),
                    "sha256": sha256_file(self.onnx_path),
                    "bytes": self.onnx_path.stat().st_size,
                },
                build={
                    "builder": "onnxruntime TensorrtExecutionProvider",
                    "session_construction_seconds": round(elapsed, 3),
                    "built_this_session": built,
                    "options": {
                        k: str(v) for k, v in sorted(DEFAULT_TRT_EP_OPTIONS.items())
                    },
                },
            )
        except OSError:
            # A cache that cannot be written must not take the run down; the
            # engine still exists in the EP's own directory.
            pass

    def run(self, tensor):
        """Run on an NCHW CUDA tensor, returning an NCHW CUDA tensor.

        Both sides are bound by device pointer, so nothing crosses PCIe.

        The dtype check is not a formality. ``bind_input`` is handed a raw
        device pointer and an element type; if the two disagree, ONNX Runtime
        reads the buffer at the size the *declared* type implies. Handing an
        fp16 tensor to a graph whose input is fp32 therefore makes it read
        twice the allocation and fault with ``CUDA failure 700: an illegal
        memory access``, from inside the session, with nothing in the message
        pointing at the dtype. That combination is not exotic -- the pinned SR
        artifact is fp32-only, and running it as the parity reference inside
        an otherwise fp16 chain produces it on the first frame.
        """
        import torch

        if not tensor.is_cuda:
            raise ValueError("OnnxRuntimeBackend requires a CUDA tensor")
        expected = self._input_torch_dtype
        if tensor.dtype is not expected:
            if not self._warned_about_cast:
                # Once per backend, not once per frame. The cast is a real
                # cost -- an fp16-to-fp32 copy of the input frame every frame
                # -- so it is reported rather than absorbed silently.
                logger.warning(
                    "%s expects %s input but the chain is handing it %s; casting "
                    "per frame. Build the engine at the chain's precision to "
                    "avoid the copy.",
                    self.onnx_path.name,
                    expected,
                    tensor.dtype,
                )
                self._warned_about_cast = True
            tensor = tensor.to(expected)
        tensor = tensor.contiguous()
        binding = self.session.io_binding()
        binding.bind_input(
            name=self.input_name,
            device_type="cuda",
            device_id=self.device_id,
            element_type=self._input_np_dtype,
            shape=tuple(tensor.shape),
            buffer_ptr=tensor.data_ptr(),
        )
        binding.bind_output(
            self.output_name, device_type="cuda", device_id=self.device_id
        )
        # ONNX Runtime's CUDA EP runs on its own CUDA stream, not torch's.
        # Binding a torch tensor by device pointer therefore crosses a stream
        # boundary in both directions, and neither direction is ordered by
        # default: ORT can start reading the input before torch has finished
        # producing it, and torch can start reading the output before ORT has
        # finished writing it. Nothing in the API hints at this -- the call
        # returns, and the tensor is there.
        #
        # It is not a rare race. Measured on the reference clip: the same
        # whole-clip run at ring depth 1, 2 and 4 produced three different
        # outputs, differing in 9, and then 178 of 191 frames, because a
        # deeper pipeline means more frames in flight and a wider window for
        # ORT's stream to overwrite a frame another stage is still reading.
        # It is also almost certainly the "output is not byte-stable across
        # two runs" defect recorded against M2: two runs differ whenever their
        # timing differs.
        #
        # Cloning the output does not help -- the clone reads the same memory
        # at the same unsynchronised moment. Only the stream ordering does.
        torch.cuda.current_stream().synchronize()
        binding.synchronize_inputs()
        self.session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        out = binding.get_outputs()[0]
        result = torch.from_dlpack(out._ortvalue.to_dlpack())
        # The clone is load-bearing. With the output left unbound, ONNX
        # Runtime allocates it from its own arena, and that allocation is
        # handed back to the arena for reuse -- so the tensor the previous
        # frame is still holding gets overwritten by the next inference. The
        # chain has several frames in flight by design (§8.4 rule 5), so the
        # overwrite lands on a frame that has not been encoded yet and the
        # output carries a band of frames belonging to later ones.
        #
        # Measured: a chunk of the reference clip run with ring depth 2
        # differed from the whole-clip run in 14 of 191 frames, with a mean
        # absolute difference of up to 126 of 255 -- not rounding, whole
        # frames. At ring depth 1 the band shrank to 7 frames, which is the
        # signature of an in-flight aliasing bug rather than of arithmetic.
        #
        # Binding a torch-owned output buffer would avoid the copy, and is
        # the follow-on: it needs the output shape threaded in from the chain,
        # which the stage knows and the backend currently does not.
        return result.clone()

    def close(self) -> None:
        self.session = None


def _profile(name: str, wh: tuple[int, int]) -> str:
    width, height = wh
    return f"{name}:1x3x{height}x{width}"


def _first_input_name(ort, path: Path) -> str:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    session = ort.InferenceSession(
        str(path), sess_options=so, providers=["CPUExecutionProvider"]
    )
    return session.get_inputs()[0].name


def _np_dtype_of(ort_type: str):
    import numpy as np

    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(uint8)": np.uint8,
    }
    if ort_type not in mapping:
        raise ValueError(f"unsupported ONNX input type {ort_type}")
    return mapping[ort_type]


def _torch_dtype_of(ort_type: str):
    """The torch dtype a bound input buffer must actually have.

    Kept next to ``_np_dtype_of`` because the two must not diverge: the numpy
    type is what ONNX Runtime is told the buffer contains, and the torch type
    is what it does contain. A mismatch is read as a buffer overrun.
    """
    import torch

    mapping = {
        "tensor(float)": torch.float32,
        "tensor(float16)": torch.float16,
        "tensor(uint8)": torch.uint8,
    }
    if ort_type not in mapping:
        raise ValueError(f"unsupported ONNX input type {ort_type}")
    return mapping[ort_type]


# --------------------------------------------------------------------------
# Native TensorRT, the §9.3 extraction seam
# --------------------------------------------------------------------------


class NativeTensorRTBackend:
    """Placeholder for the driver extracted from ``efschu/vs-mlrt``.

    The extraction is scoped in DESIGN #333 §9.3: reuse ``trt_utils.h``,
    ``cuda_helper.h``, ``cuda_utils.h`` and ``lanczos3_kernel.cu`` as they
    are, rewrite the one ``vs_bitblt`` call in ``inference_helper.h`` to
    ``cudaMemcpy2D``, and replace ``vs_tensorrt.cpp`` with a driver taking an
    engine path and device buffers. It buys explicit control over execution
    contexts, per-context workspace accounting and tiling -- none of which
    the ONNX Runtime EP exposes -- and it is what a per-context memory post
    (§6.2 ``trt_context_workspace_bytes``) can be measured against directly.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise BackendUnavailable(
            "NativeTensorRTBackend is the DESIGN #333 §9.3 extraction seam and is "
            "not built in M2. Use OnnxRuntimeBackend(provider='tensorrt')."
        )
