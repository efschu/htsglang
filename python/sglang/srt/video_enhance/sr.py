# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The super-resolution stage and its model artifacts.

The pinned first-build artifact is ``realesr-general-wdn-x4v3_opset16.onnx``
(SRVGGNetCompact, x4, 4,864,534 bytes, opset 16). Two properties of that
artifact drive the code here:

*   It is **fp32-only** as distributed -- no ``_fp16`` sibling exists. An
    fp16 engine is produced by the builder downcasting at build time, which
    is why the fp16 engine is a separate registered post with its own parity
    gate rather than a free precision switch.
*   The ``wdn`` denoise blend is **baked in**. Upstream's denoise-strength
    knob is a runtime interpolation between two checkpoints in the PyTorch
    script; ONNX export froze one point on that blend. Variable denoise
    strength at request time is therefore not available from this artifact
    and would need either multiple fixed-blend exports or a pre-export weight
    interpolation.
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from sglang.srt.video_enhance.asset_root import default_sr_model_dir
from sglang.srt.video_enhance.backends import BackendUnavailable, OnnxRuntimeBackend
from sglang.srt.video_enhance.engine_cache import (
    EngineCache,
    EngineKey,
    ShapeTriplet,
    sha256_file,
)
from sglang.srt.video_enhance.frame_math import PixelFormat, Resolution
from sglang.srt.video_enhance.frames import StageBase
from sglang.srt.video_enhance.parity import ParityGateFailure, grade


@dataclass(frozen=True)
class SrModel:
    """A pinned SR artifact, with the identity needed to verify a download."""

    model_id: str
    url: str
    filename: str
    sha256: str
    size_bytes: int
    scale: int
    exported_precision: str
    opset: int
    architecture: str
    note: str = ""


#: Verified by download on 2026-07-31: the fetched file is 4,864,534 bytes and
#: hashes to the value below, matching the size recorded in ANALYSE_333 §6.1.
REALESR_GENERAL_WDN_X4V3 = SrModel(
    model_id="realesr-general-wdn-x4v3",
    url=(
        "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/"
        "download/models/realesr-general-wdn-x4v3_opset16.onnx"
    ),
    filename="realesr-general-wdn-x4v3_opset16.onnx",
    sha256="293f5cc725c13d6af37e85e49650c414d012a4038b13db124b8cbc9d62ecb165",
    size_bytes=4864534,
    scale=4,
    exported_precision="fp32",
    opset=16,
    architecture="SRVGGNetCompact",
    note="denoise blend baked in at export; no fp16 sibling is distributed",
)

#: Native x2 models. Scaling strategy, in order of preference:
#:
#: 1. When the caller wants x2, use a NATIVE x2 model. No 8K intermediate is
#:    produced at all, so the model choice is a pipeline parameter of the
#:    target scale rather than a fixed stage.
#: 2. Where x4 is genuinely wanted, fuse the downscale into the tail of the SR
#:    engine instead of running it as a separate engine -- see
#:    ``FUSED_TAIL_RESIZE_NOTE``.
#: 3. Tail surgery on an x4 network to make it x2 without a finetune is NOT
#:    done. It is documented as considered and rejected: the final
#:    pixel-shuffle and the convolution feeding it are trained together, and
#:    replacing them without retraining is a quality risk with no measurement
#:    behind it.
#:
#: These entries are pinned by URL and size but their sha256 is filled in on
#: first verified fetch -- unlike the x4 artifact they were not downloaded
#: during this build, and writing a hash nobody verified would be worse than
#: recording none.
REALESRGAN_X2PLUS = SrModel(
    model_id="RealESRGAN_x2plus",
    url=(
        "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/"
        "download/models/RealESRGAN_x2plus.pth"
    ),
    filename="RealESRGAN_x2plus.pth",
    sha256="",
    size_bytes=0,
    scale=2,
    exported_precision="fp32",
    opset=0,
    architecture="RRDBNet",
    note="native x2; requires ONNX export before use, not yet fetched or verified",
)

#: Why the resize belongs inside the SR engine rather than after it.
#:
#: SRVGGNetCompact computes at input resolution throughout and only expands in
#: the final pixel-shuffle. Everything expensive about the x4 output is
#: therefore memory traffic in the tail, not compute: at 1080p input the tail
#: writes a 190 MiB fp16 8K frame that the very next stage reads back and
#: immediately reduces to 47 MiB. Fusing the Lanczos-3 downscale in as the
#: engine's last layer removes that round trip and one kernel launch per
#: frame, and it is exactly the operation ``efschu/vs-mlrt``'s
#: ``lanczos3_kernel.cu`` already implements.
#:
#: **Built in #457** as ONNX graph surgery on the pinned artifact --
#: ``video_enhance/fused_tail.py``. Ticket V supplied the before numbers this
#: note asked for: SR 25.42 ms, resize 24.37 ms on the 5090. The fused engine
#: emits 4K directly, so the 8K intermediate is never materialised. The fused
#: stage's own ms/frame is a GPU row and is BOOT-PENDING (TICKET_460 §5);
#: until it lands, every re-priced chain figure is labelled ESTIMATE.
FUSED_TAIL_RESIZE_NOTE = (
    "resize-fused-into-SR-tail is BUILT (#457, video_enhance/fused_tail.py): "
    "the pinned graph gains a stride-2 depthwise Lanczos-3 tail and the engine "
    "emits 4K directly. Its ms/frame is BOOT-PENDING (TICKET_460 §5)"
)

SR_MODELS: dict[str, SrModel] = {
    REALESR_GENERAL_WDN_X4V3.model_id: REALESR_GENERAL_WDN_X4V3,
    REALESRGAN_X2PLUS.model_id: REALESRGAN_X2PLUS,
}


def model_for_scale(scale: int) -> SrModel:
    """Pick the SR model whose native scale matches, preferring no intermediate.

    A request for x2 gets a native x2 model where one is registered, because
    running x4 and resizing back produces an 8K intermediate the request never
    asked for.
    """
    for model in SR_MODELS.values():
        if model.scale == scale:
            return model
    raise ArtifactError(
        f"no SR model registered with native scale x{scale}; registered scales "
        f"are {sorted({m.scale for m in SR_MODELS.values()})}"
    )


#: #251: follows SGLANG_VIDEO_MODEL_ROOT; unset -> the previous literal.
DEFAULT_MODEL_DIR = default_sr_model_dir()


class ArtifactError(RuntimeError):
    """A model artifact is missing, or its bytes do not match the pinned hash."""


def fetch_model(
    model: SrModel, model_dir: Path = DEFAULT_MODEL_DIR, *, allow_download: bool = True
) -> Path:
    """Return a local path to the artifact, verifying its hash.

    A hash mismatch is fatal rather than a re-download trigger: silently
    replacing a file whose content did not match what was pinned is how an
    engine ends up built from an artifact nobody can name.
    """
    model_dir = Path(model_dir)
    path = model_dir / model.filename
    sidecar = path.with_suffix(path.suffix + ".provenance.json")

    if path.is_file():
        digest = sha256_file(path)
        if digest != model.sha256:
            raise ArtifactError(
                f"{path} hashes to {digest}, pinned value is {model.sha256}"
            )
        return path

    if not allow_download:
        raise ArtifactError(f"{path} is absent and downloads are disabled")

    model_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(model.url, timeout=300) as response:  # noqa: S310
        tmp.write_bytes(response.read())
    digest = sha256_file(tmp)
    if digest != model.sha256:
        tmp.unlink(missing_ok=True)
        raise ArtifactError(
            f"downloaded {model.url} hashes to {digest}, pinned value is {model.sha256}"
        )
    tmp.replace(path)
    sidecar.write_text(
        json.dumps(
            {
                "url": model.url,
                "sha256": model.sha256,
                "bytes": model.size_bytes,
                "fetched_at": time.time(),
                "architecture": model.architecture,
                "opset": model.opset,
                "exported_precision": model.exported_precision,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return path


#: Suffix of the locally derived fp16 artifact and of its provenance sidecar.
#: See ``scripts/video_enhance/export_sr_fp16.py``.
FP16_SUFFIX = "_fp16.onnx"
PROVENANCE_SUFFIX = ".provenance.json"


def derived_fp16_model(
    model_dir: Path = DEFAULT_MODEL_DIR,
    base: SrModel = REALESR_GENERAL_WDN_X4V3,
) -> SrModel:
    """The locally derived fp16 sibling of a pinned artifact.

    The upstream distribution has no fp16 file, so this one is produced
    locally and cannot carry a hash pinned in source. Its identity comes from
    the sidecar the export wrote: the hash of the file it was derived from,
    the hash of the file itself, and the conversion method. Reading the hash
    from the sidecar rather than recomputing it silently is the point -- a
    file whose bytes no longer match what the export recorded fails
    ``fetch_model``'s check exactly as a corrupted download would.
    """
    path = Path(model_dir) / (Path(base.filename).stem + FP16_SUFFIX)
    sidecar = path.with_suffix(path.suffix + PROVENANCE_SUFFIX)
    if not sidecar.is_file():
        raise ArtifactError(
            f"no fp16 artifact derived yet: {sidecar} is absent. Run "
            "scripts/video_enhance/export_sr_fp16.py, which writes both the "
            "ONNX and the provenance it is verified against."
        )
    manifest = json.loads(sidecar.read_text())
    if manifest.get("derived_from_sha256") != base.sha256:
        raise ArtifactError(
            f"{path.name} was derived from an artifact hashing to "
            f"{manifest.get('derived_from_sha256')}, but the pinned source now "
            f"hashes to {base.sha256}. Re-export rather than use a derivative "
            "of a different model."
        )
    return SrModel(
        model_id=f"{base.model_id}-fp16",
        url="",
        filename=path.name,
        sha256=manifest["sha256"],
        size_bytes=int(manifest["bytes"]),
        scale=base.scale,
        exported_precision="fp16",
        opset=base.opset,
        architecture=base.architecture,
        note=(
            f"derived locally from {base.filename} by {manifest.get('method')}; "
            "graded by the parity gate before use"
        ),
    )


def derived_fused_tail_model(
    model_dir: Path = DEFAULT_MODEL_DIR,
    base: SrModel = REALESR_GENERAL_WDN_X4V3,
    *,
    kind: str = "lanczos3",
) -> SrModel:
    """The locally derived fused-tail sibling of a pinned artifact (#457).

    Identity comes from the sidecar the surgery wrote, on the same terms as
    :func:`derived_fp16_model`: the source hash, this file's hash, and the
    method. The returned model's ``scale`` is the **net** scale -- x2 for an
    x4 network with a 2:1 tail -- so a stage built from it computes the right
    output resolution without knowing that a fusion happened.
    """
    from sglang.srt.video_enhance.fused_tail import (
        FUSED_TAIL_SUFFIX,
        PROVENANCE_SUFFIX as FUSED_PROVENANCE_SUFFIX,
    )

    path = Path(model_dir) / (Path(base.filename).stem + FUSED_TAIL_SUFFIX)
    sidecar = path.with_suffix(path.suffix + FUSED_PROVENANCE_SUFFIX)
    if not sidecar.is_file():
        raise ArtifactError(
            f"no fused-tail artifact derived yet: {sidecar} is absent. Run "
            "scripts/video_enhance/export_sr_fused_tail.py, which writes both "
            "the ONNX and the provenance it is verified against."
        )
    manifest = json.loads(sidecar.read_text())
    if manifest.get("derived_from_sha256") != base.sha256:
        raise ArtifactError(
            f"{path.name} was derived from an artifact hashing to "
            f"{manifest.get('derived_from_sha256')}, but the pinned source now "
            f"hashes to {base.sha256}. Re-export rather than use a derivative "
            "of a different model."
        )
    method = manifest.get("method", "")
    if method != f"append_halving_tail:{kind}":
        raise ArtifactError(
            f"{path.name} carries tail {method!r}, not the requested {kind!r}. "
            "The comparison arms (nearest, bicubic_antialias) exist to be "
            "graded, not to be served."
        )
    return SrModel(
        model_id=f"{base.model_id}-fusedtail-{kind}",
        url="",
        filename=path.name,
        sha256=manifest["sha256"],
        size_bytes=int(manifest["bytes"]),
        scale=int(manifest["net_scale"]),
        exported_precision=base.exported_precision,
        opset=int(manifest.get("opset", {}).get("", base.opset)),
        architecture=base.architecture,
        note=(
            f"derived locally from {base.filename} by {method}; the tail resize "
            "is inside the graph, so this model's scale is the net scale and no "
            "separate resize stage follows it"
        ),
    )


def sr_engine_key(
    model: SrModel,
    onnx_path: Path,
    *,
    nvml_uuid: str,
    device_name: str,
    driver_version: str,
    runtime: str,
    runtime_version: str,
    precision: str,
    shapes: ShapeTriplet,
) -> EngineKey:
    return EngineKey(
        model_id=model.model_id,
        onnx_sha256=sha256_file(onnx_path),
        nvml_uuid=nvml_uuid,
        device_name=device_name,
        driver_version=driver_version,
        runtime=runtime,
        runtime_version=runtime_version,
        precision=precision,
        shapes=shapes,
        builder_flags=("builder_optimization_level=5", "timing_cache"),
    )


#: Torch dtype behind each RGB pixel format the chain declares. NV12 has no
#: entry: it is the codec-side layout and never reaches an SR engine.
_FORMAT_DTYPE_NAMES: dict[PixelFormat, str] = {
    PixelFormat.RGB_FP16: "float16",
    PixelFormat.RGB_FP32: "float32",
    PixelFormat.RGB_UINT8: "uint8",
}


def _as_format(tensor, fmt: PixelFormat):
    """Return ``tensor`` in the dtype the chain declared for this boundary."""
    import torch

    name = _FORMAT_DTYPE_NAMES.get(fmt)
    if name is None:
        raise ValueError(f"{fmt.value} is not an SR stage boundary format")
    wanted = getattr(torch, name)
    return tensor if tensor.dtype is wanted else tensor.to(wanted)


class SuperResolutionStage(StageBase):
    """x4 SR over one device-resident frame at a time.

    Batching is deliberately not exposed. The chain's concurrency comes from
    in-flight depth across stages (§8.4 rule 5), and a batch dimension would
    multiply the dominant memory post -- the activation ping-pong of §8.3 --
    without changing the arithmetic that the reservation is derived from.
    """

    name = "sr"

    def __init__(
        self,
        backend,
        *,
        source: Resolution,
        fmt: PixelFormat,
        scale: int = 4,
    ) -> None:
        self.backend = backend
        self.source = source
        self.format = fmt
        self.scale = scale
        self.out_resolution = source.scaled(scale)

    @classmethod
    def build(
        cls,
        *,
        source: Resolution,
        fmt: PixelFormat,
        model: SrModel = REALESR_GENERAL_WDN_X4V3,
        model_dir: Path = DEFAULT_MODEL_DIR,
        provider: str = "tensorrt",
        precision: str = "fp16",
        device_id: int = 0,
        cache: EngineCache | None = None,
        engine_key: EngineKey | None = None,
        shapes: ShapeTriplet | None = None,
    ) -> "SuperResolutionStage":
        onnx_path = fetch_model(model, model_dir)
        shapes = shapes or ShapeTriplet.static(source.width, source.height)
        backend = OnnxRuntimeBackend(
            onnx_path,
            provider=provider,
            precision=precision,
            shapes=shapes,
            device_id=device_id,
            cache=cache,
            engine_key=engine_key,
        )
        return cls(backend, source=source, fmt=fmt, scale=model.scale)

    def process(self, frames):
        out = []
        for frame in frames:
            if frame.end_of_stream:
                out.append(frame)
                continue
            if frame.resolution != self.source:
                raise ValueError(
                    f"SR stage configured for {self.source}, got {frame.resolution}"
                )
            result = self.backend.run(frame.data)
            # The chain declares one pixel format per boundary, and the
            # backend returns whatever the graph's output type is. Those two
            # are legitimately different -- the fp32 reference engine running
            # inside an fp16 chain is exactly the parity setup -- so the stage
            # restores the declared format rather than letting an fp32 tensor
            # leak downstream and be reinterpreted by the next stage.
            result = _as_format(result, self.format)
            out.append(
                frame.with_data(result, resolution=self.out_resolution).require_device(
                    self.name
                )
            )
        return out

    def close(self) -> None:
        self.backend.close()


def run_parity_gate(
    *,
    candidate_backend,
    reference_backend,
    sample_tensor,
    engine_name: str,
    cache: EngineCache | None = None,
    engine_key: EngineKey | None = None,
    raise_on_failure: bool = True,
):
    """Grade a built engine against the fp32 reference and record the verdict.

    Called once per engine build. The result lands in the engine's provenance
    manifest, so no engine can sit in the cache without a recorded verdict.
    """
    reference = reference_backend.run(sample_tensor)
    candidate = candidate_backend.run(sample_tensor)
    result = grade(candidate, reference, note=engine_name)
    if cache is not None and engine_key is not None:
        try:
            cache.attach_parity(engine_key, result.as_manifest())
        except FileNotFoundError:
            pass
    if raise_on_failure and not result.passed:
        raise ParityGateFailure(result, engine_name)
    return result


__all__ = [
    "ArtifactError",
    "BackendUnavailable",
    "DEFAULT_MODEL_DIR",
    "FUSED_TAIL_RESIZE_NOTE",
    "REALESR_GENERAL_WDN_X4V3",
    "SR_MODELS",
    "SrModel",
    "SuperResolutionStage",
    "derived_fp16_model",
    "derived_fused_tail_model",
    "fetch_model",
    "run_parity_gate",
    "sr_engine_key",
]
