# SPDX-License-Identifier: Apache-2.0
"""PCIe bus as a BUDGETED item (#286, DESIGN_201 Nachtrag-13 Erg. 7c).

What stage-2 phase multiplexing saves in resident VRAM it pays in PCIe
traffic that otherwise belongs to expert streaming or KV spill -- so the bus
itself becomes a budgeted item that a single arbiter splits between the
consumers. This module adapts the #236/#242 pattern (``SpillBudgetConfig`` /
``SpillRateBucket`` in ``managers/kv_session_offload.py``, today wired to KV
sessions) into a consumer-neutral BUS BUDGET ARBITER:

* consumers (``expert_streaming``, ``stage2_phase``, ``kv_spill``) register
  with a WEIGHT and a PRIORITY CLASS and announce planned transfers;
* each consumer owns a guaranteed weighted share of the total rate as a
  DEBT-MODEL bucket (ported from ``SpillRateBucket``: ``ready()`` gates on a
  non-negative level, ``consume`` may push into debt which refill pays off
  -- a single oversized transfer is throttled, never starved forever). The
  guaranteed share IS the starvation protection: borrowing never dips
  another consumer's bucket below zero;
* on top of the guaranteed share, a consumer may BORROW the surplus of idle
  consumers (no pending unmet demand), but only while no consumer of a
  more important priority class has pending demand -- idle-first victim
  choice, the same shape as #236's spill victim rule;
* the total rate is INJECTABLE (``set_measured_rate``); real figures come
  from the GPU measurement phase, tests feed fakes. Rate 0 (the default) is
  the OPEN budget: every request granted, byte-identical to today.

The stage-2 planner (``OffloadRegister.on_phase_boundary``) asks this
arbiter -- in addition to the per-item overlap cost function -- via
``OffloadRegister.set_bus_arbiter``.

Pure Python, CPU-hermetic; tests live in
test/registered/unit/model_executor/test_offload_bus_budget.py.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

# Known consumer names (open vocabulary -- these are the wired ones).
BUS_CONSUMER_EXPERT_STREAMING = "expert_streaming"
BUS_CONSUMER_STAGE2_PHASE = "stage2_phase"
BUS_CONSUMER_KV_SPILL = "kv_spill"


class ByteRateBucket:
    """Byte-denominated port of ``SpillRateBucket`` (#236, see
    managers/kv_session_offload.py): token bucket with the DEBT MODEL --
    ``ready()`` gates on a non-negative level, ``consume`` may push the level
    into debt, refill pays it off. The average rate converges to the budget
    and nothing stalls permanently."""

    def __init__(self, rate_bytes_per_s: float, burst_seconds: float = 1.0):
        self.rate = float(rate_bytes_per_s)
        self.cap = max(1.0, self.rate * float(burst_seconds))
        self.level = self.cap
        self._last: Optional[float] = None

    def advance(self, now: float) -> None:
        if self._last is None:
            self._last = float(now)
            return
        dt = max(0.0, float(now) - self._last)
        self._last = float(now)
        self.level = min(self.cap, self.level + dt * self.rate)

    def ready(self) -> bool:
        return self.level >= 0.0

    def surplus(self) -> float:
        """Positive level above zero -- what borrowing may take without
        pushing THIS bucket into debt (the guaranteed-share floor)."""
        return max(0.0, self.level)

    def consume(self, nbytes: float) -> None:
        self.level -= float(max(0.0, nbytes))


@dataclass
class BusGrant:
    granted: bool
    reason: str
    borrowed_bytes: float = 0.0


@dataclass
class _Consumer:
    name: str
    weight: float
    priority: int  # lower value = more important class
    bucket: ByteRateBucket = field(default_factory=lambda: ByteRateBucket(0.0))
    pending_demand: bool = False
    granted_requests: int = 0
    denied_requests: int = 0
    granted_bytes: float = 0.0


class BusBudgetArbiter:
    """Splits the measured PCIe rate between the registered consumers by
    weight and priority class (see module docstring).

    ``total_rate_bytes_per_s`` 0.0 = OPEN budget (default): every request is
    granted and nothing is metered -- byte-identical to the pre-arbiter
    behaviour, mirroring ``SpillBudgetConfig.armed``. Real rates are fed by
    the GPU measurement phase via ``set_measured_rate``; tests inject
    fakes."""

    def __init__(
        self,
        total_rate_bytes_per_s: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
        burst_seconds: float = 1.0,
    ):
        if total_rate_bytes_per_s < 0:
            raise ValueError("total rate must be >= 0")
        self._rate = float(total_rate_bytes_per_s)
        self._clock = clock
        self._burst_seconds = float(burst_seconds)
        self._consumers: Dict[str, _Consumer] = {}
        self._lock = threading.Lock()

    # -- configuration -------------------------------------------------------
    @property
    def total_rate_bytes_per_s(self) -> float:
        return self._rate

    def register_consumer(
        self, name: str, weight: float = 1.0, priority: int = 1
    ) -> None:
        """Register (or reconfigure) one consumer. ``weight`` sets its
        guaranteed share of the total rate; ``priority`` its class for
        borrowing (LOWER value = more important)."""
        if weight <= 0:
            raise ValueError(f"consumer weight must be > 0, got {weight}")
        with self._lock:
            self._consumers[name] = _Consumer(name, float(weight), int(priority))
            self._rebuild_buckets_locked()

    def set_measured_rate(self, total_rate_bytes_per_s: float) -> None:
        """Feed the measured bus rate (GPU measurement phase; tests fake)."""
        if total_rate_bytes_per_s < 0:
            raise ValueError("total rate must be >= 0")
        with self._lock:
            self._rate = float(total_rate_bytes_per_s)
            self._rebuild_buckets_locked()

    def _rebuild_buckets_locked(self) -> None:
        total_weight = sum(c.weight for c in self._consumers.values())
        for c in self._consumers.values():
            share = c.weight / total_weight if total_weight > 0 else 0.0
            c.bucket = ByteRateBucket(self._rate * share, self._burst_seconds)

    # -- arbitration ---------------------------------------------------------
    def request(self, name: str, nbytes: int) -> BusGrant:
        """One planned transfer of ``nbytes``. Granted from the consumer's
        guaranteed weighted share (debt model), else by borrowing idle
        consumers' surplus (only while no more important class has pending
        demand). A denial is throttling, never a hard state: the bucket
        refills on its own (#236 wording: rate pressure recovers)."""
        nbytes = max(0, int(nbytes))
        with self._lock:
            consumer = self._consumers.get(name)
            if consumer is None:
                raise ValueError(
                    f"unknown bus consumer {name!r}; register_consumer() "
                    f"first (known: {sorted(self._consumers)})."
                )
            if self._rate <= 0:
                # OPEN budget: unmetered, byte-identical default.
                consumer.granted_requests += 1
                consumer.granted_bytes += nbytes
                consumer.pending_demand = False
                return BusGrant(True, "open budget (no measured rate set)")
            now = self._clock()
            for c in self._consumers.values():
                c.bucket.advance(now)
            # 1. Guaranteed weighted share (debt model).
            if consumer.bucket.ready():
                consumer.bucket.consume(nbytes)
                consumer.granted_requests += 1
                consumer.granted_bytes += nbytes
                consumer.pending_demand = False
                return BusGrant(True, "guaranteed share")
            # 2. Borrowing: only while no MORE IMPORTANT class is waiting,
            #    and only from idle consumers' surplus (never into their
            #    guaranteed floor -- that is the starvation protection).
            blocked_by = [
                c.name
                for c in self._consumers.values()
                if c.name != name
                and c.pending_demand
                and c.priority < consumer.priority
            ]
            if not blocked_by:
                borrowable = sum(
                    c.bucket.surplus()
                    for c in self._consumers.values()
                    if c.name != name and not c.pending_demand
                )
                # Borrowing covers exactly the announced bytes; the
                # consumer's own debt stays its own and is paid off by its
                # guaranteed refill (debt model). Victims are only ever
                # drained down to zero, never into their floors.
                if borrowable >= nbytes > 0:
                    taken = 0.0
                    for c in self._consumers.values():
                        if c.name == name or c.pending_demand:
                            continue
                        take = min(c.bucket.surplus(), nbytes - taken)
                        if take > 0:
                            c.bucket.consume(take)
                            taken += take
                        if taken >= nbytes:
                            break
                    consumer.granted_requests += 1
                    consumer.granted_bytes += nbytes
                    consumer.pending_demand = False
                    return BusGrant(True, "borrowed idle surplus", borrowed_bytes=taken)
            consumer.pending_demand = True
            consumer.denied_requests += 1
            reason = (
                f"weighted share exhausted (bucket level "
                f"{consumer.bucket.level:.0f} B); rate pressure recovers on "
                f"its own"
            )
            if blocked_by:
                reason += (
                    "; borrowing blocked by pending higher-priority demand "
                    f"from {sorted(blocked_by)}"
                )
            return BusGrant(False, reason)

    def clear_pending(self, name: str) -> None:
        """A consumer that no longer wants its announced transfer withdraws
        its pending demand (so it stops blocking lower classes' borrowing)."""
        with self._lock:
            consumer = self._consumers.get(name)
            if consumer is not None:
                consumer.pending_demand = False

    # -- telemetry -----------------------------------------------------------
    def as_dict(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return {
                c.name: {
                    "weight": c.weight,
                    "priority": c.priority,
                    "rate_bytes_per_s": c.bucket.rate,
                    "bucket_level": c.bucket.level,
                    "granted_requests": c.granted_requests,
                    "denied_requests": c.denied_requests,
                    "granted_bytes": c.granted_bytes,
                    "pending_demand": float(c.pending_demand),
                }
                for c in self._consumers.values()
            }
