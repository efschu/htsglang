# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""A backend that performs the whole job lifecycle without a GPU.

Not a stub. It runs the same state machine a real backend runs -- steps,
periodic checkpoints, metric events, preemption at a step boundary, resume
from a checkpoint directory, a final artifact on disk -- with the arithmetic
replaced by a decaying loss curve. That makes it the vehicle for testing
everything between the socket and the executor: the OpenAI surface, the job
store, the event stream, the tenant's preempt/resume loop and the ledger
lease, on a machine with no free card.

It is never selected automatically (``explicit_only``). A run that trained
nothing must never be able to report ``succeeded`` because the real suite
happened not to be installed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from pathlib import Path
from typing import Optional

from sglang.srt.training.backends import (
    BackendEvent,
    BackendProbe,
    BackendRun,
    EventSink,
    RunOutcome,
    RunSpec,
    RunStatus,
    TrainingBackend,
)
from sglang.srt.training.feasibility import LADDER, TrainingMethod

logger = logging.getLogger(__name__)

#: Wall-clock per simulated step. Small enough that a test finishes, large
#: enough that a preempt can land mid-run deterministically.
DEFAULT_STEP_SECONDS = 0.02


class MockRun(BackendRun):
    """One simulated run, driven by an asyncio task."""

    def __init__(self, spec: RunSpec, sink: EventSink, *, step_seconds: float) -> None:
        self.spec = spec
        self._sink = sink
        self._step_seconds = step_seconds
        self._preempt = asyncio.Event()
        self._cancel = asyncio.Event()
        self._total_steps = max(1, int(spec.extra.get("total_steps", 200)))
        self._start_step = _step_of_checkpoint(spec.resume_from)
        self._step = self._start_step
        self._last_checkpoint: Optional[str] = spec.resume_from
        self._task = asyncio.get_running_loop().create_task(self._drive())

    async def wait(self) -> RunOutcome:
        return await self._task

    async def preempt(self, *, timeout_s: float = 120.0) -> RunOutcome:
        self._preempt.set()
        try:
            return await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout_s)
        except asyncio.TimeoutError:
            self._task.cancel()
            return RunOutcome(
                status=RunStatus.PREEMPTED,
                last_checkpoint=self._last_checkpoint,
                last_step=self._step,
                error=f"mock run did not stop within {timeout_s}s; cancelled",
            )

    async def cancel(self, *, timeout_s: float = 60.0) -> RunOutcome:
        self._cancel.set()
        try:
            return await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout_s)
        except asyncio.TimeoutError:
            self._task.cancel()
            return RunOutcome(
                status=RunStatus.CANCELLED,
                last_checkpoint=self._last_checkpoint,
                last_step=self._step,
            )

    # -- the simulated loop -------------------------------------------------

    async def _drive(self) -> RunOutcome:
        spec = self.spec
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        self._emit(
            "info",
            f"mock backend starting at step {self._start_step}/{self._total_steps} "
            f"({spec.method.value}, seed {spec.seed})",
        )
        try:
            while self._step < self._total_steps:
                if self._cancel.is_set():
                    self._emit("info", f"cancelled at step {self._step}")
                    return RunOutcome(
                        status=RunStatus.CANCELLED,
                        last_checkpoint=self._last_checkpoint,
                        last_step=self._step,
                    )
                if self._preempt.is_set():
                    path = self._write_checkpoint()
                    self._emit(
                        "info",
                        f"preempted at step {self._step}; checkpoint written to {path}",
                    )
                    return RunOutcome(
                        status=RunStatus.PREEMPTED,
                        last_checkpoint=path,
                        last_step=self._step,
                    )
                await asyncio.sleep(self._step_seconds)
                self._step += 1
                loss = _loss_at(self._step, self._total_steps)
                if self._step % max(1, self._total_steps // 20) == 0:
                    self._emit(
                        "info",
                        f"step {self._step}/{self._total_steps}: loss {loss:.4f}",
                        data={
                            "step": self._step,
                            "train_loss": round(loss, 6),
                            "learning_rate": spec.learning_rate,
                            "epoch": round(
                                self._step / self._total_steps * spec.n_epochs, 3
                            ),
                        },
                        event_type="metrics",
                    )
                if self._step % max(1, spec.save_steps) == 0:
                    self._write_checkpoint(loss=loss)
            artifact = self._write_artifact()
            self._emit("info", f"mock training complete; adapter at {artifact}")
            return RunOutcome(
                status=RunStatus.SUCCEEDED,
                fine_tuned_model=str(spec.output_dir.name),
                artifact_path=artifact,
                last_checkpoint=self._last_checkpoint,
                last_step=self._step,
                trained_tokens=self._step
                * spec.micro_batch_size
                * spec.sequence_length,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            logger.exception("mock backend failed")
            self._emit("error", f"mock backend failed: {exc}")
            return RunOutcome(
                status=RunStatus.FAILED,
                last_checkpoint=self._last_checkpoint,
                last_step=self._step,
                error=str(exc),
            )

    def _write_checkpoint(self, *, loss: Optional[float] = None) -> str:
        directory = self.spec.output_dir / f"checkpoint-{self._step}"
        directory.mkdir(parents=True, exist_ok=True)
        value = _loss_at(self._step, self._total_steps) if loss is None else loss
        (directory / "trainer_state.json").write_text(
            json.dumps(
                {
                    "global_step": self._step,
                    "job_id": self.spec.job_id,
                    "train_loss": value,
                    "saved_at": time.time(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (directory / "adapter_model.safetensors").write_bytes(
            b"htsglang-mock-adapter\n"
        )
        self._last_checkpoint = str(directory)
        self._sink(
            BackendEvent(
                level="info",
                message=f"checkpoint saved at step {self._step}",
                data={"step": self._step, "train_loss": round(value, 6)},
                type="metrics",
                checkpoint_path=str(directory),
                step=self._step,
            )
        )
        return str(directory)

    def _write_artifact(self) -> str:
        path = self.spec.output_dir / "adapter_model.safetensors"
        path.write_bytes(b"htsglang-mock-adapter\n")
        (self.spec.output_dir / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": self.spec.base_model_path,
                    "peft_type": "LORA",
                    "task_type": "CAUSAL_LM",
                    "htsglang_mock": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return str(path)

    def _emit(
        self,
        level: str,
        message: str,
        *,
        data: Optional[dict] = None,
        event_type: str = "message",
    ) -> None:
        self._sink(
            BackendEvent(level=level, message=message, data=data, type=event_type)
        )


def _loss_at(step: int, total: int) -> float:
    """A plausible decaying curve. Deterministic, so tests can assert on it."""
    return 2.5 * math.exp(-3.0 * step / max(1, total)) + 0.35


def _step_of_checkpoint(path: Optional[str]) -> int:
    if not path:
        return 0
    name = Path(path).name
    if name.startswith("checkpoint-"):
        try:
            return int(name.split("-", 1)[1])
        except ValueError:
            return 0
    return 0


class MockBackend(TrainingBackend):
    """Always available, never chosen automatically."""

    name = "mock"
    explicit_only = True

    def __init__(self, *, step_seconds: float = DEFAULT_STEP_SECONDS) -> None:
        self.step_seconds = step_seconds

    def probe(self) -> BackendProbe:
        return BackendProbe(
            name=self.name,
            available=True,
            version="1",
            reason="simulated executor; produces no trained weights",
        )

    def supported_methods(self) -> tuple[TrainingMethod, ...]:
        return tuple(LADDER)

    async def launch(self, spec: RunSpec, sink: EventSink) -> BackendRun:
        self.check(spec)
        return MockRun(spec, sink, step_seconds=self.step_seconds)
