# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""RIFE frame interpolation as a chain stage.

DESIGN #333 §9.3 settles what is taken from ``HolyWu/vs-rife`` and what is
not: *dependency at the model level, port at the execution level*. Concretely:

*   the model version enum, the modulo-32/64/128 padding rule, the ``scale``
    semantics and the ``module.``/``encode.`` state-dict key rewriting are
    mirrored here exactly, verified line by line against
    ``vsrife/__init__.py`` at the commit recorded in
    ``_vendor/rife/README.md``;
*   the IFNet architectures are vendored verbatim under ``_vendor/rife/``;
*   the execution path -- padding, encode caching, timestep construction,
    unpadding -- is ours, because upstream's is a VapourSynth frame callback
    and this chain hands device tensors between stages with no host round
    trip (§8.1).

Two things this module deliberately does not do:

*   It does not assert a VRAM number for RIFE. §8.3 registers the per-frame-pair
    footprint as measurement post P4 and asserts nothing, so
    :meth:`RifeStage.footprint` surfaces
    :class:`~sglang.srt.video_enhance.frame_math.UnprobedFootprintError` until
    a measured value is supplied.
*   It does not build TensorRT engines. The seam is marked
    (:class:`RifeBackend`, :meth:`RifeStage.engine_resolution`) and the shape
    that seam needs is already correct -- the engine is built at the
    *post-resize* size, which is the whole reason resize precedes RIFE in
    §8.1 -- but the builder itself is a later step.

The arithmetic helpers (:func:`padding_modulo`, :func:`padded_shape`) import
no torch, so the planner and the fixed-cost calculation stay runnable on a
host with no CUDA and no torch at all.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from sglang.srt.video_enhance.chain import StageKind, StageSpec
from sglang.srt.video_enhance.frame_math import (
    Resolution,
    StageFootprint,
    dtype_bytes,
    rife_footprint,
)
from sglang.srt.video_enhance.frames import Frame, StageBase

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


# --------------------------------------------------------------------------
# Version enum
# --------------------------------------------------------------------------

#: Every model string ``HolyWu/vs-rife`` accepts, in upstream order (the
#: ``models`` list in ``vsrife/__init__.py``). Being in this tuple means the
#: name is real and its padding rule is known -- it does not mean the
#: architecture is vendored here. See :data:`SUPPORTED_VERSIONS`.
RIFE_VERSIONS: tuple[str, ...] = (
    "4.0",
    "4.1",
    "4.2",
    "4.3",
    "4.4",
    "4.5",
    "4.6",
    "4.7",
    "4.8",
    "4.9",
    "4.10",
    "4.11",
    "4.12",
    "4.12.lite",
    "4.13",
    "4.13.lite",
    "4.14",
    "4.14.lite",
    "4.15",
    "4.15.lite",
    "4.16.lite",
    "4.17",
    "4.17.lite",
    "4.18",
    "4.19",
    "4.20",
    "4.21",
    "4.22",
    "4.22.lite",
    "4.23",
    "4.24",
    "4.25",
    "4.25.lite",
    "4.25.heavy",
    "4.26",
    "4.26.heavy",
)

KNOWN_VERSIONS: frozenset[str] = frozenset(RIFE_VERSIONS)

#: The ``scale`` values upstream accepts. Mirrored so a bad value fails at
#: config time rather than producing a silently different padding.
VALID_SCALES: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)

WEIGHT_URL_TEMPLATE = (
    "https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v{version}.pkl"
)


class RifeVersionError(ValueError):
    """Base for anything wrong with a requested RIFE version."""


class UnknownRifeVersionError(RifeVersionError):
    """The version string is not one upstream ships."""


class UnsupportedRifeVersionError(RifeVersionError):
    """A real upstream version whose architecture is not vendored here."""


# --------------------------------------------------------------------------
# Vendored architectures
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VendoredArch:
    """What is needed to construct one vendored IFNet and its encode head."""

    version: str
    module: str
    #: Versions from 4.7 on prepend a small encoder ("Head") whose output is
    #: concatenated into every IFBlock input. 4.6 has none.
    encode_channels: int | None
    #: Number of IFBlock pyramid levels. 4.26 added a fifth.
    num_blocks: int

    @property
    def has_head(self) -> bool:
        return self.encode_channels is not None


#: Upstream ships one IFNet file per version, but several of those files are
#: byte-identical to each other at the pinned commit -- 4.15, 4.17 and 4.18
#: share one architecture and 4.15.lite, 4.16.lite and 4.17.lite share another
#: (sha256 proof in ``_vendor/rife/README.md``). Only the *weights* differ, and
#: the weights are pinned separately. Mapping the aliases onto the single
#: vendored module is therefore not a substitution of the kind
#: :func:`require_supported` refuses -- it is the same bytes under two names,
#: and it is what lets the #460 ladder offer 4.15/4.17 and the lite family
#: without carrying five more near-duplicate files.
_VENDORED: dict[str, VendoredArch] = {
    "4.6": VendoredArch("4.6", "IFNet_HDv3_v4_6", None, 4),
    "4.15": VendoredArch("4.15", "IFNet_HDv3_v4_18", 8, 4),
    "4.15.lite": VendoredArch("4.15.lite", "IFNet_HDv3_v4_17_lite", 4, 4),
    "4.16.lite": VendoredArch("4.16.lite", "IFNet_HDv3_v4_17_lite", 4, 4),
    "4.17": VendoredArch("4.17", "IFNet_HDv3_v4_18", 8, 4),
    "4.17.lite": VendoredArch("4.17.lite", "IFNet_HDv3_v4_17_lite", 4, 4),
    "4.18": VendoredArch("4.18", "IFNet_HDv3_v4_18", 8, 4),
    "4.26": VendoredArch("4.26", "IFNet_HDv3_v4_26", 4, 5),
}

#: The subset whose architecture actually lives in ``_vendor/rife/``. Asking
#: for anything else raises rather than substituting a neighbour: two RIFE
#: versions differ in weights *and* in forward signature, so a substitution
#: would either crash later or silently produce a different model's output.
SUPPORTED_VERSIONS: frozenset[str] = frozenset(_VENDORED)


def require_known(version: str) -> str:
    if version not in KNOWN_VERSIONS:
        raise UnknownRifeVersionError(
            f"unknown RIFE version {version!r}; HolyWu/vs-rife ships "
            f"{', '.join(RIFE_VERSIONS)}"
        )
    return version


def require_supported(version: str) -> VendoredArch:
    """Resolve a version to its vendored architecture, or explain why not."""
    require_known(version)
    try:
        return _VENDORED[version]
    except KeyError:
        raise UnsupportedRifeVersionError(
            f"RIFE version {version!r} exists upstream but its architecture is "
            f"not vendored here. Vendored: {', '.join(sorted(SUPPORTED_VERSIONS))}. "
            "Add the matching IFNet file under video_enhance/_vendor/rife/ with "
            "its provenance recorded in that directory's README, then extend "
            "_VENDORED. No substitution is made."
        ) from None


# --------------------------------------------------------------------------
# Padding
# --------------------------------------------------------------------------

DEFAULT_PADDING_MODULO = 32

# Read off the `modulo = ...` assignments in the model match-statement of
# vsrife/__init__.py. ANALYSE_333 §6.2 lists 4.25/4.25.heavy/4.26 for 64 and
# omits 4.26.heavy; the source assigns 64 there too, and the source wins.
_MODULO_OVERRIDES: dict[str, int] = {
    "4.25": 64,
    "4.25.lite": 128,
    "4.25.heavy": 64,
    "4.26": 64,
    "4.26.heavy": 64,
}


def padding_modulo(version: str) -> int:
    """Alignment the IFNet pyramid requires of its input, in pixels.

    Defined for every *known* version, not only the vendored ones: the value
    is metadata from the upstream dispatcher, and the planner needs it to size
    buffers for configurations this build cannot execute yet.
    """
    require_known(version)
    return _MODULO_OVERRIDES.get(version, DEFAULT_PADDING_MODULO)


def require_valid_scale(scale: float) -> float:
    if scale not in VALID_SCALES:
        raise ValueError(
            f"rife scale must be one of {VALID_SCALES}, got {scale!r} "
            "(HolyWu/vs-rife semantics, mirrored exactly)"
        )
    return scale


def padded_shape(res: Resolution, version: str, scale: float = 1.0) -> Resolution:
    """Input geometry the flownet is actually fed.

    ``vsrife/__init__.py``::

        tmp = max(modulo, int(modulo / scale))
        pw = math.ceil(w / tmp) * tmp
        ph = math.ceil(h / tmp) * tmp

    The ``int()`` truncates, and ``max`` keeps the alignment from ever
    dropping below ``modulo`` when ``scale > 1``. Reproduced exactly, because
    a different padding is a different tensor shape and therefore a different
    TensorRT engine and a different output.
    """
    modulo = padding_modulo(version)
    require_valid_scale(scale)
    tmp = max(modulo, int(modulo / scale))
    return Resolution(
        math.ceil(res.width / tmp) * tmp,
        math.ceil(res.height / tmp) * tmp,
    )


def padding_amounts(
    res: Resolution, version: str, scale: float = 1.0
) -> tuple[int, int]:
    """``(right, bottom)`` pixels of padding. Upstream pads only those edges."""
    padded = padded_shape(res, version, scale)
    return padded.width - res.width, padded.height - res.height


# --------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------

#: sha256 of the upstream release assets, recorded when they were first
#: fetched (4.6/4.18/4.26 on 2026-07-31, the rest on 2026-08-03 for the #460
#: ladder). A download that does not match is rejected: the release tag is
#: mutable on GitHub, so pinning the hash is the only thing that makes the
#: fetch reproducible. ``scripts/video_enhance/fetch_rife_weights.py`` is the
#: only thing that establishes a new entry here, and only with
#: ``--record-new-pin``; it refuses an unpinned re-download outright.
KNOWN_WEIGHT_SHA256: dict[str, str] = {
    "4.6": "008646e761f0e67cb77f0c6c44cfe3c3e5a05d9d9465311b9681ca650ce030db",
    "4.15": "bf969affb3f6e9a09eea8c1da6c51bb22d6da45e92cafb9273689d05e2360dd2",
    "4.15.lite": "54fe3fb51efb880f64ffe2048a88220ad2b0b3d03f7cb800742e90821dcfb24f",
    "4.16.lite": "290cf343fb5a933f4503920ca07826c4d30561bd9b573b11b054b33ea0dee642",
    "4.17": "b1ee3186270312a38316e4d53c77b31a60062cfa5636e13d6f0a1dd89bb7b128",
    "4.17.lite": "9303f59001bc40e801273828cc0da84b8f7696a9485cd7c45e7796f20ed2f086",
    "4.18": "955fc8b552417ae4808999f5d671d024438e904c0bebf766ae6fb61806ce021c",
    "4.26": "45c7f74156704769dc9f85cfcaf8552e1e926f9399dcfa3a553dee88fac6f53f",
}

_ENV_WEIGHT_DIR = "SGLANG_RIFE_WEIGHT_DIR"
_SIDECAR_SCHEMA = 1


def default_weight_dir() -> Path:
    return Path(
        os.environ.get(_ENV_WEIGHT_DIR, "~/.cache/sglang/video_enhance/rife")
    ).expanduser()


def weight_filename(version: str) -> str:
    return f"flownet_v{version}.pkl"


def _sidecar_path(weight_path: Path) -> Path:
    return weight_path.with_suffix(weight_path.suffix + ".json")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _sidecar_is_valid(weight_path: Path, expected_sha256: str | None) -> bool:
    """True when the cached file provably matches its recorded provenance."""
    sidecar = _sidecar_path(weight_path)
    if not weight_path.is_file() or not sidecar.is_file():
        return False
    try:
        record = json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if record.get("schema") != _SIDECAR_SCHEMA:
        return False
    if record.get("size_bytes") != weight_path.stat().st_size:
        return False
    recorded = record.get("sha256")
    if expected_sha256 is not None and recorded != expected_sha256:
        return False
    # Hashing 20-25 MiB is a few tens of milliseconds and is the only check
    # that catches a truncated or half-written cache entry.
    return recorded == sha256_file(weight_path)


def weights_are_cached(
    version: str,
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    expected_sha256: str | None = None,
) -> bool:
    """True when :func:`download_weights` would do no network I/O.

    Lets a caller that must not touch the network -- a test, or a GPU window
    that refuses to spend its time downloading -- check first.
    """
    require_known(version)
    directory = Path(cache_dir) if cache_dir is not None else default_weight_dir()
    if expected_sha256 is None:
        expected_sha256 = KNOWN_WEIGHT_SHA256.get(version)
    return _sidecar_is_valid(directory / weight_filename(version), expected_sha256)


def download_weights(
    version: str,
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    expected_sha256: str | None = None,
    force: bool = False,
) -> Path:
    """Fetch ``flownet_v<version>.pkl`` into ``cache_dir`` and return its path.

    Writes a sidecar ``<file>.json`` recording sha256, source URL, byte size
    and fetch time. A subsequent call that finds a sidecar which validates
    against the file on disk performs no network I/O.
    """
    require_supported(version)
    directory = Path(cache_dir) if cache_dir is not None else default_weight_dir()
    directory.mkdir(parents=True, exist_ok=True)
    weight_path = directory / weight_filename(version)
    if expected_sha256 is None:
        expected_sha256 = KNOWN_WEIGHT_SHA256.get(version)

    if not force and _sidecar_is_valid(weight_path, expected_sha256):
        return weight_path

    url = WEIGHT_URL_TEMPLATE.format(version=version)
    tmp_path = weight_path.with_suffix(weight_path.suffix + ".partial")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as out:
        while block := response.read(1 << 20):
            out.write(block)

    digest = sha256_file(tmp_path)
    if expected_sha256 is not None and digest != expected_sha256:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"sha256 mismatch for {url}: expected {expected_sha256}, got {digest}. "
            "The upstream release asset changed; verify it before updating "
            "KNOWN_WEIGHT_SHA256."
        )
    tmp_path.replace(weight_path)

    _sidecar_path(weight_path).write_text(
        json.dumps(
            {
                "schema": _SIDECAR_SCHEMA,
                "version": version,
                "source_url": url,
                "sha256": digest,
                "size_bytes": weight_path.stat().st_size,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    return weight_path


# --------------------------------------------------------------------------
# Module construction
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RifeModules:
    """The flownet and, for 4.7+, its separate encode head."""

    version: str
    flownet: Any  # torch.nn.Module
    encode: Any | None  # torch.nn.Module | None


def _torch_dtype(dtype: str) -> "torch.dtype":
    import torch

    dtype_bytes(dtype)  # rejects an unknown name with the shared message
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]


def _import_arch(arch: VendoredArch):
    import importlib

    return importlib.import_module(
        f"sglang.srt.video_enhance._vendor.rife.{arch.module}"
    )


def build_modules(
    version: str,
    *,
    scale: float = 1.0,
    ensemble: bool = False,
    device: str = "cpu",
    dtype: str = "fp32",
) -> RifeModules:
    """Construct the vendored IFNet with framework-default random weights.

    This is the shape-and-signature path: it is what a test uses to exercise
    the architecture without a 25 MiB checkpoint, and what
    :func:`weight_load` fills in. It produces no meaningful interpolation.
    """
    arch = require_supported(version)
    require_valid_scale(scale)
    module = _import_arch(arch)
    tdtype = _torch_dtype(dtype)
    flownet = module.IFNet(scale, ensemble).eval().to(device=device, dtype=tdtype)
    encode = None
    if arch.has_head:
        encode = module.Head().eval().to(device=device, dtype=tdtype)
    return RifeModules(version=version, flownet=flownet, encode=encode)


def weight_load(
    version: str,
    path: str | os.PathLike[str],
    *,
    scale: float = 1.0,
    ensemble: bool = False,
    device: str = "cpu",
    dtype: str = "fp32",
) -> RifeModules:
    """Load ``flownet_v<version>.pkl`` onto the vendored architecture.

    The checkpoints are saved from a ``DataParallel``-wrapped model, so every
    key carries a ``module.`` prefix; upstream ``init_module`` strips it and
    *drops any key that lacks it*, which is what makes ``strict=False``
    tolerable there. The same filter is applied here. The encode head is
    loaded from the ``encode.`` subtree of the same file, matching upstream:
    the head is a submodule of IFNet for the vendored versions, but it is also
    run standalone so its output can be cached across frame pairs.
    """
    import torch

    arch = require_supported(version)
    require_valid_scale(scale)
    module = _import_arch(arch)
    tdtype = _torch_dtype(dtype)

    raw = torch.load(str(path), map_location="cpu", mmap=True, weights_only=True)
    state_dict = {k.replace("module.", ""): v for k, v in raw.items() if "module." in k}
    if not state_dict:
        raise ValueError(
            f"{path}: no 'module.'-prefixed keys; this does not look like a "
            "HolyWu/vs-rife flownet checkpoint"
        )

    # Materialise on meta first so the random init is never allocated: these
    # tensors are all about to be overwritten by assign=True.
    with torch.device("meta"):
        flownet = module.IFNet(scale, ensemble)
    flownet.load_state_dict(state_dict, strict=False, assign=True)
    flownet.eval().to(device=device, dtype=tdtype)

    encode = None
    if arch.has_head:
        encode_state = {
            k.replace("encode.", ""): v for k, v in state_dict.items() if "encode." in k
        }
        with torch.device("meta"):
            encode = module.Head()
        encode.load_state_dict(encode_state, assign=True)
        encode.eval().to(device=device, dtype=tdtype)

    return RifeModules(version=version, flownet=flownet, encode=encode)


# --------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------


class RifeBackend(str, Enum):
    TORCH = "torch"
    #: Seam only. The engine has to be built at :meth:`RifeStage.engine_resolution`,
    #: i.e. the padded *post-resize* size -- §8.1's reason for the stage order.
    TENSORRT = "tensorrt"


#: Upstream's ``trt_max_shape`` default. ANALYSE_333 §6.2 identifies this, not
#: any architectural limit, as what makes 4K fail out of the box. The chain
#: always overrides it from the post-resize geometry; it is kept here so the
#: default we are diverging from is visible.
UPSTREAM_TRT_MAX_SHAPE = Resolution(1920, 1080)


class RifeStage(StageBase):
    """Interpolate ``multiplier - 1`` frames between each consecutive pair.

    Arity follows §8.1 and ``chain.StageSpec``: two frames in, ``multiplier-1``
    frames out. The source frames themselves are not re-emitted; the executor
    interleaves them.
    """

    name = "rife"

    def __init__(
        self,
        *,
        resolution: Resolution,
        version: str = "4.6",
        multiplier: int = 2,
        scale: float = 1.0,
        dtype: str = "fp16",
        device: str = "cuda",
        ensemble: bool = False,
        weights_path: str | os.PathLike[str] | None = None,
        modules: RifeModules | None = None,
        backend: RifeBackend = RifeBackend.TORCH,
    ) -> None:
        self.arch = require_supported(version)
        self.version = version
        self.resolution = resolution
        self.scale = require_valid_scale(scale)
        self.dtype = dtype
        self.device = device
        self.ensemble = ensemble
        self.backend = RifeBackend(backend)
        if multiplier < 2:
            raise ValueError(
                f"rife multiplier must be >= 2 (a multiplier of 1 means the stage "
                f"should not be in the chain at all), got {multiplier}"
            )
        self.multiplier = multiplier
        if modules is not None and modules.version != version:
            raise ValueError(
                f"pre-built modules are version {modules.version!r}, stage is "
                f"{version!r}"
            )
        self._weights_path = Path(weights_path) if weights_path is not None else None
        self._modules = modules
        self._grid: Any | None = None
        self._flow_div: Any | None = None
        # Keyed by Frame.order_key. Each source frame is the right member of
        # one pair and the left member of the next, so a two-entry cache halves
        # the encode-head work on a sequential stream.
        self._encode_cache: dict[tuple[int, int], Any] = {}

    # -- geometry ---------------------------------------------------------

    @property
    def arity_in(self) -> int:
        return 2

    @property
    def arity_out(self) -> int:
        return self.multiplier - 1

    @property
    def padded(self) -> Resolution:
        return padded_shape(self.resolution, self.version, self.scale)

    def engine_resolution(self) -> Resolution:
        """Shape a TensorRT engine for this stage must be built at.

        This is the padded post-resize size, not the source size and not
        upstream's 1920x1080 default. Getting it from the chain rather than
        from a config default is what §8.1 buys by placing resize first.
        """
        return self.padded

    def footprint(self, measured_bytes_per_pair: int | None = None) -> StageFootprint:
        """Per-frame-pair device bytes.

        DESIGN §8.3 registers this as measurement post P4 and asserts no
        number, so without a measured value this raises rather than guessing.
        """
        return rife_footprint(self.resolution, self.dtype, measured_bytes_per_pair)

    # -- lifecycle --------------------------------------------------------

    def warmup(self) -> None:
        import torch

        if self.backend is RifeBackend.TENSORRT:
            raise NotImplementedError(
                "the TensorRT backend for RIFE is a declared seam, not an "
                f"implementation. Build the engine at {self.engine_resolution()} "
                "(padded post-resize size) and plug it in behind "
                "RifeStage._flownet_forward."
            )

        if self._modules is None:
            if self._weights_path is None:
                raise ValueError(
                    "RifeStage needs either weights_path or a pre-built modules= "
                    "object; it will not run with random weights implicitly"
                )
            self._modules = weight_load(
                self.version,
                self._weights_path,
                scale=self.scale,
                ensemble=self.ensemble,
                device=self.device,
                dtype=self.dtype,
            )

        padded = self.padded
        # warp() works in fp32 regardless of the flownet dtype, so the grid and
        # the divisor stay fp32 -- matching upstream, where a half grid loses
        # enough precision at 4K to visibly shift the sampled positions.
        self._flow_div = torch.tensor(
            [(padded.width - 1.0) / 2.0, (padded.height - 1.0) / 2.0],
            dtype=torch.float32,
            device=self.device,
        )
        horizontal = (
            torch.linspace(
                -1.0, 1.0, padded.width, dtype=torch.float32, device=self.device
            )
            .view(1, 1, 1, padded.width)
            .expand(-1, -1, padded.height, -1)
        )
        vertical = (
            torch.linspace(
                -1.0, 1.0, padded.height, dtype=torch.float32, device=self.device
            )
            .view(1, 1, padded.height, 1)
            .expand(-1, -1, -1, padded.width)
        )
        self._grid = torch.cat([horizontal, vertical], 1)

    def close(self) -> None:
        self._modules = None
        self._grid = None
        self._flow_div = None
        self._encode_cache.clear()

    # -- execution --------------------------------------------------------

    def _prepare(self, frame: Frame) -> Any:
        import torch.nn.functional as F

        tensor = frame.data
        if getattr(tensor, "ndim", None) != 4 or tensor.shape[1] != 3:
            raise ValueError(
                f"rife expects an NCHW RGB tensor with C=3, got "
                f"{getattr(tensor, 'shape', type(tensor))}"
            )
        if str(tensor.device).split(":")[0] != self.device.split(":")[0]:
            raise ValueError(
                f"rife stage is on {self.device} but received a frame on "
                f"{tensor.device}; the chain does no implicit transfers (§8.1)"
            )
        tensor = tensor.to(_torch_dtype(self.dtype))
        right, bottom = padding_amounts(self.resolution, self.version, self.scale)
        if right or bottom:
            tensor = F.pad(tensor, (0, right, 0, bottom))
        return tensor

    def _encode_for(self, key: tuple[int, int], padded_tensor: Any) -> Any:
        cached = self._encode_cache.get(key)
        if cached is None:
            cached = self._modules.encode(padded_tensor)
            # Only the two frames of the current pair can still be needed.
            if len(self._encode_cache) >= 2:
                self._encode_cache.clear()
            self._encode_cache[key] = cached
        return cached

    def _flownet_forward(
        self, img0: Any, img1: Any, timestep: Any, f0: Any | None, f1: Any | None
    ) -> Any:
        """Single call into the model. The TensorRT seam replaces this body."""
        if self.arch.has_head:
            return self._modules.flownet(
                img0, img1, timestep, self._flow_div, self._grid, f0, f1
            )
        return self._modules.flownet(img0, img1, timestep, self._flow_div, self._grid)

    def process(self, frames: Sequence[Frame]) -> Sequence[Frame]:
        import torch

        if len(frames) != self.arity_in:
            raise ValueError(
                f"rife consumes {self.arity_in} frames per invocation, got {len(frames)}"
            )
        left, right = frames
        if left.end_of_stream or right.end_of_stream:
            # Nothing to interpolate across the end of the source; the
            # executor forwards the sentinel itself.
            return ()
        if self._modules is None or self._grid is None:
            raise RuntimeError("RifeStage.warmup() must run before process()")

        img0 = self._prepare(left)
        img1 = self._prepare(right)
        f0 = f1 = None
        if self.arch.has_head:
            f0 = self._encode_for(left.order_key, img0)
            f1 = self._encode_for(right.order_key, img1)

        padded = self.padded
        tdtype = _torch_dtype(self.dtype)
        out: list[Frame] = []
        for step in range(1, self.multiplier):
            t = step / self.multiplier
            timestep = torch.full(
                [1, 1, padded.height, padded.width],
                t,
                dtype=tdtype,
                device=self.device,
            )
            result = self._flownet_forward(img0, img1, timestep, f0, f1)
            result = result[:, :, : self.resolution.height, : self.resolution.width]
            out.append(
                Frame(
                    data=result,
                    resolution=self.resolution,
                    format=left.format,
                    index=left.index,
                    sub_index=step,
                    sub_count=self.multiplier,
                    pts=_interpolate_pts(left.pts, right.pts, t),
                )
            )
        return tuple(out)

    # -- construction from a resolved chain -------------------------------

    @classmethod
    def from_stage_spec(
        cls,
        spec: StageSpec,
        *,
        dtype: str = "fp16",
        device: str = "cuda",
        weights_path: str | os.PathLike[str] | None = None,
        modules: RifeModules | None = None,
        backend: RifeBackend = RifeBackend.TORCH,
    ) -> "RifeStage":
        if spec.kind is not StageKind.RIFE:
            raise ValueError(f"expected a RIFE stage spec, got {spec.kind.value}")
        return cls(
            resolution=spec.in_res,
            version=str(spec.options["version"]),
            multiplier=spec.arity_out + 1,
            scale=float(spec.options.get("scale", 1.0)),
            dtype=dtype,
            device=device,
            weights_path=weights_path,
            modules=modules,
            backend=backend,
        )


def _interpolate_pts(left: int | None, right: int | None, t: float) -> int | None:
    if left is None or right is None:
        return None
    return left + round((right - left) * t)
