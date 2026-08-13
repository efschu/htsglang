# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Continuous corridor sampling: the recorder's answer to "that was a snapshot".

WHY THE BOOT MARKS ARE NOT CORRIDOR EVIDENCE. The corridor law is a
CONTINUOUS minimum -- free memory per card must never fall below 1024 MiB at
any instant under load, sampled finely enough to catch a trough. Every number
the phase marks produce is a BOOT SNAPSHOT: it says where a card came to rest,
and says nothing about the 1.5-second dip a flip cutover drives through it.
#631's own history is the proof that the difference decides things: a cutover
was measured entering the corridor at 3006 MiB free and sitting at 940 MiB --
84 MiB UNDER the law -- for 1.5 s, while every resting-level reading in the
same window looked lawful.

So this module samples at the corridor's own cadence (100 ms) for as long as a
load leg runs, and keeps the MINIMUM rather than the average, because a law
about a floor is broken by the worst instant and not by the typical one.

WHAT IT COSTS, AND WHY THAT IS AFFORDABLE. One sample is one
``nvmlDeviceGetMemoryInfo`` on the rank's own card plus two integer reads from
torch's allocator; it allocates no device memory and takes no lock the serving
path holds. The samples land in a FIXED-SIZE ring, so a leg of any length costs
constant host memory and a forgotten sampler cannot grow without bound. The
overhead is measured and reported by :func:`summary` rather than asserted --
``sample_cost_us`` is the instrument's own price, recorded next to its
readings, which is the only honest way to publish a measurement that competes
with the thing it measures.

OFF BY DEFAULT, and off in two independent ways: the thread is never started
unless :func:`start` is called, and :func:`start` returns None unless
:data:`TRACE_ENV` is set. A serving process that does not opt in does not
import a thread, does not touch NVML on a timer, and is byte-identical.
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import os
import threading
import time
from typing import Any, Deque, Dict, Optional

logger = logging.getLogger(__name__)

MIB = 1 << 20

#: Arms the sampler. Value is the sampling period in milliseconds; ``1`` and
#: any non-numeric truthy value mean the corridor cadence.
TRACE_ENV = "SGLANG_CORRIDOR_TRACE_MS"

#: The corridor's own sampling cadence. The law is stated as a continuous
#: minimum and audited at 100 ms elsewhere in this tree; sampling coarser than
#: the audit would produce a number that cannot be compared with it.
DEFAULT_PERIOD_MS = 100

#: Ring capacity. At 100 ms this is 30 minutes of history in ~5 MiB of host
#: RAM, which is longer than any load leg this rig runs, and bounded for the
#: ones it does not.
DEFAULT_CAPACITY = 18000

#: Fallback only. The law lives in ``managers.corridor_guard`` and is read
#: from there; this value exists so that an instrument can still report a
#: verdict if that import is ever unavailable, and it is deliberately the
#: same number so a fallback cannot change a verdict silently.
_LAW_MIB_FALLBACK = 1024


def corridor_law_mib() -> int:
    """The ONE declaration of the corridor law, read at call time.

    Imported lazily: this module sits under ``mem_ledger`` and the law is
    declared in ``managers.corridor_guard``, so a module-level import would
    tie an instrument's import graph to the scheduler's.
    """
    try:
        from sglang.srt.managers.corridor_guard import (
            corridor_law_mib as _law,
        )

        return int(_law())
    except Exception:  # pragma: no cover - defensive; see _LAW_MIB_FALLBACK
        return _LAW_MIB_FALLBACK


@dataclasses.dataclass
class Sample:
    monotonic: float
    nvml_free_bytes: int
    nvml_self_bytes: int
    torch_reserved_bytes: int
    torch_allocated_bytes: int
    kv_arena_backed_bytes: int


class CorridorTrace:
    """A bounded, self-timing ring of corridor samples for ONE card."""

    def __init__(
        self,
        period_ms: int = DEFAULT_PERIOD_MS,
        capacity: int = DEFAULT_CAPACITY,
        card_uuid: Optional[str] = None,
    ):
        self.period_s = max(0.001, period_ms / 1000.0)
        self.card_uuid = card_uuid
        self.samples: Deque[Sample] = collections.deque(maxlen=capacity)
        #: The instrument's own price, in microseconds per sample. Kept as a
        #: running max and total so the report can state both the typical cost
        #: and the worst one, rather than a mean that hides a stall.
        self.cost_total_us = 0.0
        self.cost_max_us = 0.0
        self.overruns = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- sampling ---------------------------------------------------------

    def _read(self) -> Optional[Sample]:
        from sglang.srt.mem_ledger.flight_recorder import _nvml_view, _torch_view

        nvml = _nvml_view()
        if "nvml_free_bytes" not in nvml:
            return None
        torch_view = _torch_view(None)
        arena = 0
        try:
            from sglang.srt.mem_cache.kv_vmm_backing import arena_census

            for row in arena_census().values():
                arena += int(row.get("backed", 0) or 0) + int(
                    row.get("retained", 0) or 0
                )
        except Exception:  # pragma: no cover
            arena = 0
        if self.card_uuid is None:
            self.card_uuid = nvml.get("card_uuid")
        return Sample(
            monotonic=time.monotonic(),
            nvml_free_bytes=int(nvml.get("nvml_free_bytes") or 0),
            nvml_self_bytes=int(nvml.get("nvml_self_bytes") or 0),
            torch_reserved_bytes=int(torch_view.get("reserved_bytes") or 0),
            torch_allocated_bytes=int(torch_view.get("allocated_bytes") or 0),
            kv_arena_backed_bytes=arena,
        )

    def sample_once(self) -> Optional[Sample]:
        """Take one sample and charge its cost to the instrument's own account."""
        began = time.perf_counter()
        try:
            sample = self._read()
        except Exception:  # pragma: no cover - a probe never breaks serving
            return None
        cost_us = (time.perf_counter() - began) * 1e6
        self.cost_total_us += cost_us
        self.cost_max_us = max(self.cost_max_us, cost_us)
        if sample is not None:
            self.samples.append(sample)
        return sample

    def _loop(self) -> None:
        next_at = time.monotonic()
        while not self._stop.is_set():
            self.sample_once()
            next_at += self.period_s
            delay = next_at - time.monotonic()
            if delay < 0:
                # The sampler could not keep its cadence. Counted, never
                # silently skipped: a trace with unrecorded gaps must not be
                # presented as a continuous minimum.
                self.overruns += 1
                next_at = time.monotonic()
                continue
            self._stop.wait(delay)

    # -- lifecycle --------------------------------------------------------

    def start(self) -> "CorridorTrace":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._loop, name="corridor-trace", daemon=True
        )
        self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> "CorridorTrace":
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=timeout)
        return self

    # -- reading ----------------------------------------------------------

    def summary(self, corridor_mib: Optional[int] = None) -> Dict[str, Any]:
        """The minimum, not the mean, plus what the instrument itself cost.

        ``corridor_mib`` defaults to the ONE declaration of the law
        (:data:`corridor_guard.CORRIDOR_LAW_MIB`) rather than to a literal
        of this module's own. A private copy of a threshold is how an
        instrument ends up reporting a different verdict from the gate it
        is meant to audit (#656).
        """
        corridor_mib = corridor_law_mib() if corridor_mib is None else int(corridor_mib)
        samples = list(self.samples)
        if not samples:
            return {"n": 0, "card_uuid": self.card_uuid}
        free = [s.nvml_free_bytes for s in samples]
        span = samples[-1].monotonic - samples[0].monotonic
        floor = min(free)
        return {
            "card_uuid": self.card_uuid,
            "n": len(samples),
            "span_s": round(span, 3),
            "period_ms": int(self.period_s * 1000),
            "free_min_mib": floor // MIB,
            "free_max_mib": max(free) // MIB,
            "free_last_mib": free[-1] // MIB,
            "corridor_mib": corridor_mib,
            # The verdict the law actually asks for: did the WORST instant hold.
            "breach": bool(floor // MIB < corridor_mib),
            "margin_mib": floor // MIB - corridor_mib,
            "arena_backed_min_mib": min(s.kv_arena_backed_bytes for s in samples)
            // MIB,
            "arena_backed_max_mib": max(s.kv_arena_backed_bytes for s in samples)
            // MIB,
            "sample_cost_us_mean": round(self.cost_total_us / max(1, len(samples)), 1),
            "sample_cost_us_max": round(self.cost_max_us, 1),
            # Duty cycle: what fraction of wall time the instrument spent
            # measuring. This is the number that decides whether the trace may
            # run beside a benchmark.
            "duty_pct": round(
                100.0 * (self.cost_total_us / 1e6) / span if span > 0 else 0.0, 4
            ),
            "overruns": self.overruns,
        }

    def dump(self, path: str, corridor_mib: Optional[int] = None) -> str:
        payload = {
            "summary": self.summary(corridor_mib=corridor_mib),
            "samples": [dataclasses.asdict(s) for s in self.samples],
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as handle:
            json.dump(payload, handle)
        os.replace(tmp, path)
        return path


def requested_period_ms() -> Optional[int]:
    raw = os.environ.get(TRACE_ENV)
    if not raw:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_PERIOD_MS
    return DEFAULT_PERIOD_MS if value <= 1 else value


def start(capacity: int = DEFAULT_CAPACITY) -> Optional[CorridorTrace]:
    """Start a trace if :data:`TRACE_ENV` asks for one, else return None."""
    period = requested_period_ms()
    if period is None:
        return None
    logger.info("corridor trace armed at %d ms", period)
    return CorridorTrace(period_ms=period, capacity=capacity).start()


__all__ = [
    "TRACE_ENV",
    "DEFAULT_PERIOD_MS",
    "CorridorTrace",
    "Sample",
    "requested_period_ms",
    "start",
]
