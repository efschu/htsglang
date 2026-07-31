# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""On-disk cache for built inference engines, keyed by physical identity.

TensorRT engines are not portable. Upstream vs-mlrt states it plainly --
"Engines are system specific, don't use across multiple systems" -- so a
cache that is keyed by anything weaker than the full build identity will
eventually hand a process an engine built for a different card, a different
TensorRT, or a different shape range. The failure is not a clean error; it is
a wrong result or a crash minutes into a run.

The key therefore mirrors what ``model_loader/hibernate.py`` already uses for
parked GPU tensors -- NVML UUID plus content hash -- extended with the parts
of the build that change the engine:

    nvml_uuid | device_name | driver | trt_version | onnx_sha256
             | precision | min/opt/max shape triplet | builder flags

Every cache entry carries a provenance manifest recording where its source
artifact came from and how it was built. An engine file with no manifest is
treated as absent, not as usable: a cached artifact whose origin cannot be
stated is exactly the class of defect the #304 work registered.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

MANIFEST_SUFFIX = ".provenance.json"
MANIFEST_VERSION = 1


def sha256_file(path: str | os.PathLike[str], chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ShapeTriplet:
    """TensorRT's min / opt / max dynamic shape range, as ``(width, height)``."""

    min_wh: tuple[int, int]
    opt_wh: tuple[int, int]
    max_wh: tuple[int, int]

    def __post_init__(self) -> None:
        for name in ("min_wh", "opt_wh", "max_wh"):
            value = getattr(self, name)
            if len(value) != 2 or any(v <= 0 for v in value):
                raise ValueError(f"{name} must be a positive (width, height) pair")
        if any(a > b for a, b in zip(self.min_wh, self.opt_wh)):
            raise ValueError("min shape must not exceed opt shape")
        if any(a > b for a, b in zip(self.opt_wh, self.max_wh)):
            raise ValueError("opt shape must not exceed max shape")

    def token(self) -> str:
        return "min{}x{}_opt{}x{}_max{}x{}".format(
            *self.min_wh, *self.opt_wh, *self.max_wh
        )

    @classmethod
    def static(cls, width: int, height: int) -> "ShapeTriplet":
        wh = (width, height)
        return cls(min_wh=wh, opt_wh=wh, max_wh=wh)


@dataclass(frozen=True)
class EngineKey:
    """Everything that changes the bytes of a built engine."""

    model_id: str
    onnx_sha256: str
    nvml_uuid: str
    device_name: str
    driver_version: str
    runtime: str  # "tensorrt", "onnxruntime-trt", "torch"
    runtime_version: str
    precision: str  # "fp32" | "fp16" | "bf16"
    shapes: ShapeTriplet
    builder_flags: tuple[str, ...] = ()

    def digest(self) -> str:
        payload = json.dumps(
            {
                "model_id": self.model_id,
                "onnx_sha256": self.onnx_sha256,
                "nvml_uuid": self.nvml_uuid,
                "device_name": self.device_name,
                "driver_version": self.driver_version,
                "runtime": self.runtime,
                "runtime_version": self.runtime_version,
                "precision": self.precision,
                "shapes": self.shapes.token(),
                "builder_flags": sorted(self.builder_flags),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def filename(self) -> str:
        # The human-readable prefix is for operators reading `ls`; the digest is
        # what actually makes the name unique.
        return (
            f"{self.model_id}_{self.precision}_{self.shapes.token()}"
            f"_{self.digest()[:16]}.engine"
        )


@dataclass
class Provenance:
    """Recorded next to every cached engine. Absent manifest = absent engine."""

    manifest_version: int
    key: dict
    source_artifact: dict  # url, sha256, bytes, fetched_at
    build: dict  # command or api call, flags, duration_s, built_at, builder
    parity: dict | None = None  # filled in by the parity gate, see parity.py
    extra: dict = field(default_factory=dict)


class EngineCache:
    """Content-addressed engine store with a mandatory provenance manifest."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)

    def path_for(self, key: EngineKey) -> Path:
        return self.root / key.filename()

    def manifest_path_for(self, key: EngineKey) -> Path:
        return self.path_for(key).with_suffix(".engine" + MANIFEST_SUFFIX)

    def lookup(self, key: EngineKey) -> Path | None:
        """Return the cached engine, or None if it is missing or unprovenanced."""
        engine = self.path_for(key)
        manifest = self.manifest_path_for(key)
        if not engine.is_file() or not manifest.is_file():
            return None
        try:
            data = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if data.get("manifest_version") != MANIFEST_VERSION:
            return None
        if data.get("key", {}).get("digest") != key.digest():
            # A digest mismatch means the manifest belongs to a different build
            # that happened to collide on the filename. Treat as a miss.
            return None
        return engine

    def store(
        self,
        key: EngineKey,
        engine_bytes: bytes | None,
        *,
        built_path: str | os.PathLike[str] | None = None,
        source_artifact: dict,
        build: dict,
        parity: dict | None = None,
        extra: dict | None = None,
    ) -> Path:
        """Place an engine plus its manifest in the cache, atomically.

        Exactly one of ``engine_bytes`` or ``built_path`` must be given;
        ``built_path`` covers builders such as trtexec that write the file
        themselves.
        """
        if (engine_bytes is None) == (built_path is None):
            raise ValueError("pass exactly one of engine_bytes or built_path")
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.path_for(key)

        if engine_bytes is not None:
            _atomic_write_bytes(target, engine_bytes)
        else:
            src = Path(built_path)  # type: ignore[arg-type]
            if not src.is_file():
                raise FileNotFoundError(f"builder did not produce {src}")
            if src.resolve() != target.resolve():
                _atomic_write_bytes(target, src.read_bytes())

        manifest = Provenance(
            manifest_version=MANIFEST_VERSION,
            key={**_key_dict(key), "digest": key.digest()},
            source_artifact=source_artifact,
            build={**build, "stored_at": time.time()},
            parity=parity,
            extra=extra or {},
        )
        _atomic_write_bytes(
            self.manifest_path_for(key),
            json.dumps(asdict(manifest), indent=2, sort_keys=True).encode(),
        )
        return target

    def attach_parity(self, key: EngineKey, parity: dict) -> None:
        """Record a parity-gate result against an already-cached engine."""
        manifest_path = self.manifest_path_for(key)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"no manifest for {key.filename()}")
        data = json.loads(manifest_path.read_text())
        data["parity"] = parity
        _atomic_write_bytes(
            manifest_path, json.dumps(data, indent=2, sort_keys=True).encode()
        )

    def entries(self) -> list[tuple[Path, dict]]:
        out: list[tuple[Path, dict]] = []
        if not self.root.is_dir():
            return out
        for manifest in sorted(self.root.glob("*" + MANIFEST_SUFFIX)):
            engine = manifest.with_name(manifest.name[: -len(MANIFEST_SUFFIX)])
            if not engine.is_file():
                continue
            try:
                out.append((engine, json.loads(manifest.read_text())))
            except (OSError, json.JSONDecodeError):
                continue
        return out


def _key_dict(key: EngineKey) -> dict:
    data = asdict(key)
    data["shapes"] = {
        "min_wh": list(key.shapes.min_wh),
        "opt_wh": list(key.shapes.opt_wh),
        "max_wh": list(key.shapes.max_wh),
        "token": key.shapes.token(),
    }
    data["builder_flags"] = list(key.builder_flags)
    return data


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
