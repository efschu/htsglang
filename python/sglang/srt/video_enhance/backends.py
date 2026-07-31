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


class BackendUnavailable(RuntimeError):
    """The requested backend cannot run in this process."""


@dataclass(frozen=True)
class BackendInfo:
    """What actually ran, recorded so a measurement can be attributed."""

    runtime: str
    runtime_version: str
    provider: str
    precision: str
    engine_path: str | None
    built_engine: bool
    build_seconds: float | None


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
            options.append(("CUDAExecutionProvider", {"device_id": device_id}))
        else:
            options.append((provider_name, {"device_id": device_id}))

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

        self.info = BackendInfo(
            runtime="onnxruntime",
            runtime_version=ort.__version__,
            provider=provider_name,
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
        """
        import torch

        if not tensor.is_cuda:
            raise ValueError("OnnxRuntimeBackend requires a CUDA tensor")
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
        self.session.run_with_iobinding(binding)
        out = binding.get_outputs()[0]
        return torch.from_dlpack(out._ortvalue.to_dlpack())

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
