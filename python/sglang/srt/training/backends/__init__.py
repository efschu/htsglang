# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""The executor interface every training suite is wrapped behind (D1).

DESIGN #341 D1: existing suites are wrapped, never rebuilt and never vendored.
LLaMA-Factory, kohya sd-scripts and Unsloth are all Apache-2.0, so driving
them as subprocesses is license-clean and costs nothing but an interface.

This module is that interface. It is deliberately narrower than any of those
suites: launch a run, read its events, stop it at a checkpoint, cancel it.
Everything a suite does beyond that -- dataset formats, method taxonomies,
distributed strategies -- stays inside the suite's own configuration, which is
why a wrapper for kohya (M2, diffusion LoRA) or Unsloth (M2, fast path) is a
new subclass and not a change here.

**Why the interface has ``preempt`` at all.** A tenant that can only be killed
is not preemptible; it is interruptible, and every interruption costs the
whole run. ``preempt`` is the difference: it asks the executor to reach a
checkpoint and stop, and it returns the path to resume from. The three-line
contract -- checkpoint, release, resume -- is what makes training a tenant
rather than a squatter (D4).
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from sglang.srt.training.feasibility import TrainingMethod

logger = logging.getLogger(__name__)


class RunStatus(str, Enum):
    """How a backend run ended."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    #: Stopped at a checkpoint on the tenant's request. Not an ending: the
    #: scheduler will launch the same job again with ``resume_from`` set.
    PREEMPTED = "preempted"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class BackendProbe:
    """Whether this backend can run here, and if not, precisely why."""

    name: str
    available: bool
    reason: str = ""
    version: str = ""
    executable: str = ""
    remedies: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "available": self.available,
            "reason": self.reason,
            "version": self.version,
            "executable": self.executable,
            "what_would_make_it_work": list(self.remedies),
        }


@dataclass(frozen=True)
class RunSpec:
    """Everything one launch needs. Backend-agnostic on purpose."""

    job_id: str
    base_model_path: str
    method: TrainingMethod
    dataset_path: Path
    output_dir: Path
    n_epochs: int = 3
    micro_batch_size: int = 1
    learning_rate: float = 5e-5
    sequence_length: int = 2048
    seed: int = 0
    gradient_checkpointing: bool = True
    #: Steps between saved checkpoints. This is the preemption granularity:
    #: a preempt loses at most the work since the last save, so it is a
    #: tenant-policy number and not only a training one.
    save_steps: int = 50
    validation_path: Optional[Path] = None
    resume_from: Optional[str] = None
    #: Physical device indices this run may touch, as a CUDA_VISIBLE_DEVICES
    #: string. Process-level isolation, matching the rest of the fork: inside
    #: the child, ``cuda:0`` is unambiguous.
    cuda_visible_devices: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunOutcome:
    """How the run ended, and what it left behind."""

    status: RunStatus
    fine_tuned_model: Optional[str] = None
    artifact_path: Optional[str] = None
    last_checkpoint: Optional[str] = None
    last_step: int = 0
    trained_tokens: Optional[int] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class BackendEvent:
    """One thing worth telling the caller about."""

    level: str
    message: str
    data: Optional[dict[str, Any]] = None
    type: str = "message"
    #: Set when this event is a checkpoint being written, so the job store can
    #: turn it into a ``fine_tuning.job.checkpoint`` object.
    checkpoint_path: Optional[str] = None
    step: Optional[int] = None


#: Called synchronously from the backend's own asyncio task.
EventSink = Callable[[BackendEvent], None]


class BackendUnavailable(RuntimeError):
    """The requested backend is not installed or cannot run this method."""

    def __init__(self, probe: BackendProbe) -> None:
        super().__init__(probe.reason)
        self.probe = probe


class BackendRun(abc.ABC):
    """A launched run. One object per attempt, not per job."""

    @abc.abstractmethod
    async def wait(self) -> RunOutcome:
        """Block until the run ends by itself."""

    @abc.abstractmethod
    async def preempt(self, *, timeout_s: float = 120.0) -> RunOutcome:
        """Checkpoint and stop. Returns with ``last_checkpoint`` set.

        Bounded: an executor that will not stop within ``timeout_s`` is
        escalated to a kill, because the whole point of preemption is that
        the serving tenant does not wait on it.
        """

    @abc.abstractmethod
    async def cancel(self, *, timeout_s: float = 60.0) -> RunOutcome:
        """Stop for good. No checkpoint is required."""


class TrainingBackend(abc.ABC):
    """One wrapped training suite."""

    name: str = "abstract"
    #: Never selected by ``backend: auto``. A backend that does not really
    #: train must be asked for by name -- picking one automatically because
    #: the real suite is missing would report a succeeded job that trained
    #: nothing, which is the worst possible failure mode for this surface.
    explicit_only: bool = False

    @abc.abstractmethod
    def probe(self) -> BackendProbe:
        """Can this backend run on this host right now?"""

    @abc.abstractmethod
    def supported_methods(self) -> tuple[TrainingMethod, ...]:
        """Which rungs of the ladder this backend implements."""

    @abc.abstractmethod
    async def launch(self, spec: RunSpec, sink: EventSink) -> BackendRun:
        """Start a run. Raises :class:`BackendUnavailable` if it cannot."""

    def check(self, spec: RunSpec) -> None:
        """Common admission checks. Subclasses call this from ``launch``."""
        probe = self.probe()
        if not probe.available:
            raise BackendUnavailable(probe)
        if spec.method not in self.supported_methods():
            supported = ", ".join(m.value for m in self.supported_methods())
            raise BackendUnavailable(
                BackendProbe(
                    name=self.name,
                    available=False,
                    reason=(
                        f"backend {self.name!r} does not implement method "
                        f"{spec.method.value!r}; it implements {supported}"
                    ),
                    remedies=(
                        f"resubmit with one of: {supported}",
                        "or select a different backend in the x-htsglang block",
                    ),
                )
            )


_REGISTRY: dict[str, Callable[[], TrainingBackend]] = {}


def register_backend(name: str, factory: Callable[[], TrainingBackend]) -> None:
    _REGISTRY[name] = factory


def backend_names() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_backend(name: str) -> TrainingBackend:
    factory = _REGISTRY.get(name)
    if factory is None:
        raise KeyError(
            f"unknown training backend {name!r}; known backends are "
            f"{', '.join(backend_names())}"
        )
    return factory()


def resolve_backend(
    name: str, method: TrainingMethod
) -> tuple[TrainingBackend, list[BackendProbe]]:
    """Pick a backend by name, or the first available one that fits.

    Returns the backend and every probe taken on the way, so a rejection can
    report what was tried rather than only what failed last.
    """
    probes: list[BackendProbe] = []
    if name and name != "auto":
        backend = get_backend(name)
        probes.append(backend.probe())
        return backend, probes
    for candidate_name in backend_names():
        candidate = get_backend(candidate_name)
        if candidate.explicit_only:
            continue
        probe = candidate.probe()
        probes.append(probe)
        if probe.available and method in candidate.supported_methods():
            return candidate, probes
    raise BackendUnavailable(
        BackendProbe(
            name="auto",
            available=False,
            reason=(
                f"no installed training backend implements {method.value!r}. "
                "Probed: "
                + "; ".join(f"{p.name} ({p.reason or 'available'})" for p in probes)
            ),
            remedies=(
                "install LLaMA-Factory into the server's virtualenv "
                "(pip install llamafactory) and restart",
                "or select the 'mock' backend for a dry run of the job surface",
            ),
        )
    )


def _install_default_backends() -> None:
    from sglang.srt.training.backends.llamafactory import LlamaFactoryBackend
    from sglang.srt.training.backends.mock import MockBackend

    register_backend(LlamaFactoryBackend.name, LlamaFactoryBackend)
    register_backend(MockBackend.name, MockBackend)


_install_default_backends()

__all__ = [
    "BackendEvent",
    "BackendProbe",
    "BackendRun",
    "BackendUnavailable",
    "EventSink",
    "RunOutcome",
    "RunSpec",
    "RunStatus",
    "TrainingBackend",
    "backend_names",
    "get_backend",
    "register_backend",
    "resolve_backend",
]
