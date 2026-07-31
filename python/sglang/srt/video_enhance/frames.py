# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The unit of work that travels through the chain, and the stage protocol.

A :class:`Frame` is a *device-resident* tensor plus the metadata the chain
needs to keep ordering and timing straight. The invariant DESIGN #333 §8.1
calls the single most important structural property of the chain --
device-to-device hand-off with no host round-trip -- is enforced here:
:meth:`Frame.require_device` is called at every stage boundary by the
executor, so a stage that returns a host tensor fails loudly at the boundary
rather than silently costing a PCIe round trip per frame.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, Sequence, runtime_checkable

from sglang.srt.video_enhance.frame_math import PixelFormat, Resolution


class HostResidencyError(RuntimeError):
    """A frame left a stage on the host where the chain requires device memory."""


@dataclass
class Frame:
    """One frame in flight.

    ``data`` is a ``torch.Tensor``. It is typed loosely on purpose: this
    module must import without torch so the arithmetic and planning paths stay
    usable on a CPU-only host.
    """

    data: object
    resolution: Resolution
    format: PixelFormat
    #: Monotonic index in the *output* timeline. Interpolated frames get
    #: fractional positions expressed as ``(index, sub_index, sub_count)``.
    index: int
    sub_index: int = 0
    sub_count: int = 1
    #: Presentation timestamp in the source time base, or None if unknown.
    pts: int | None = None
    #: True for the sentinel that tells downstream stages the source ended.
    end_of_stream: bool = False

    @property
    def order_key(self) -> tuple[int, int]:
        return (self.index, self.sub_index)

    def require_device(self, stage: str) -> "Frame":
        if self.end_of_stream:
            return self
        device = getattr(self.data, "device", None)
        if device is None:
            raise HostResidencyError(
                f"stage {stage!r} produced a frame whose payload is not a tensor"
            )
        if getattr(device, "type", None) != "cuda":
            raise HostResidencyError(
                f"stage {stage!r} produced a frame on {device}; the chain requires "
                "device-resident intermediates with no host round-trip (§8.1)"
            )
        return self

    def with_data(self, data: object, **changes) -> "Frame":
        return replace(self, data=data, **changes)

    @classmethod
    def eos(cls, index: int) -> "Frame":
        return cls(
            data=None,
            resolution=Resolution(2, 2),
            format=PixelFormat.NV12,
            index=index,
            end_of_stream=True,
        )


@runtime_checkable
class Stage(Protocol):
    """One executable stage.

    Stages are synchronous with respect to the caller and asynchronous with
    respect to the device: they may enqueue work on the current CUDA stream
    and return without synchronising. The executor is responsible for stream
    ordering; a stage must not call ``torch.cuda.synchronize``.
    """

    name: str

    def warmup(self) -> None:
        """Build engines, allocate pools, capture graphs. Called once, on the
        card, inside the arbiter's capture window."""

    def process(self, frames: Sequence[Frame]) -> Sequence[Frame]:
        """Consume ``arity_in`` frames, produce ``arity_out`` frames."""

    def close(self) -> None:
        """Release execution contexts and pools. Idempotent."""


class StageBase:
    """Default no-op implementations of the optional protocol members."""

    name: str = "stage"

    def warmup(self) -> None:  # pragma: no cover - trivial
        return None

    def close(self) -> None:  # pragma: no cover - trivial
        return None
