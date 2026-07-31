# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Per-stage timing that does not itself change what it measures.

The standing rule for this project is ms per round rather than aggregate
throughput, collected per stage, with CUDA events read out of band. Reading a
CUDA event immediately after recording it forces a synchronisation and turns
an overlapped pipeline into a serialised one, so the measured ms/frame would
be an artifact of the measurement.

``StageTimer`` therefore records event pairs and defers the elapsed-time
readout: pending pairs are drained only when the caller asks for a summary,
or when the pending list grows past a bound. On a host without CUDA it falls
back to ``perf_counter``, which is correct there because there is no
asynchronous device queue to hide.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class StageTiming:
    stage: str
    samples: list[float] = field(default_factory=list)
    frames: int = 0

    def add(self, ms: float, frames: int = 1) -> None:
        self.samples.append(ms)
        self.frames += frames

    @property
    def count(self) -> int:
        return len(self.samples)

    def summary(self) -> dict:
        if not self.samples:
            return {"stage": self.stage, "count": 0, "frames": self.frames}
        ordered = sorted(self.samples)
        n = len(ordered)
        return {
            "stage": self.stage,
            "count": n,
            "frames": self.frames,
            "ms_mean": round(sum(ordered) / n, 4),
            "ms_median": round(ordered[n // 2], 4),
            "ms_p05": round(ordered[max(0, int(0.05 * n) - 1)], 4),
            "ms_p95": round(ordered[min(n - 1, int(0.95 * n))], 4),
            "ms_min": round(ordered[0], 4),
            "ms_max": round(ordered[-1], 4),
        }


class StageTimer:
    """Deferred-readout timer for one stage."""

    def __init__(
        self, stage: str, *, use_cuda_events: bool = True, max_pending: int = 256
    ):
        self.timing = StageTiming(stage=stage)
        self.max_pending = max_pending
        self._pending: list[tuple[object, object, int]] = []
        self._start_wall: float | None = None
        self._start_event = None
        self._frames = 1
        self._torch = None
        if use_cuda_events:
            try:
                import torch

                if torch.cuda.is_available():
                    self._torch = torch
            except ImportError:
                self._torch = None

    @property
    def uses_cuda_events(self) -> bool:
        return self._torch is not None

    def start(self, frames: int = 1) -> None:
        self._frames = frames
        if self._torch is not None:
            self._start_event = self._torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        else:
            self._start_wall = time.perf_counter()

    def stop(self) -> None:
        if self._torch is not None:
            end = self._torch.cuda.Event(enable_timing=True)
            end.record()
            self._pending.append((self._start_event, end, self._frames))
            self._start_event = None
            if len(self._pending) >= self.max_pending:
                self.drain()
        else:
            if self._start_wall is None:
                raise RuntimeError("stop() without start()")
            self.timing.add(
                (time.perf_counter() - self._start_wall) * 1000.0, self._frames
            )
            self._start_wall = None

    def drain(self) -> None:
        """Read out completed event pairs. Only pairs whose end event has
        already completed are read, so draining never blocks the pipeline."""
        if self._torch is None or not self._pending:
            return
        remaining: list[tuple[object, object, int]] = []
        for start, end, frames in self._pending:
            if end.query():
                self.timing.add(start.elapsed_time(end), frames)
            else:
                remaining.append((start, end, frames))
        self._pending = remaining

    def finish(self) -> StageTiming:
        """Synchronise once, at the end, and read out everything left."""
        if self._torch is not None and self._pending:
            self._torch.cuda.synchronize()
            for start, end, frames in self._pending:
                self.timing.add(start.elapsed_time(end), frames)
            self._pending = []
        return self.timing


class ChainTimers:
    """One timer per stage, summarised together."""

    def __init__(self, stages: list[str], *, use_cuda_events: bool = True) -> None:
        self.timers = {
            name: StageTimer(name, use_cuda_events=use_cuda_events) for name in stages
        }

    def __getitem__(self, stage: str) -> StageTimer:
        return self.timers[stage]

    def drain(self) -> None:
        for timer in self.timers.values():
            timer.drain()

    def summary(self) -> list[dict]:
        return [timer.finish().summary() for timer in self.timers.values()]

    def ms_per_frame(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, timer in self.timers.items():
            timing = timer.finish()
            if timing.frames:
                out[name] = sum(timing.samples) / timing.frames
        return out
