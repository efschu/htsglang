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
"""The time-series store: a cascade of ring buffers at configurable
resolutions, with downsampling from each tier into the next.

Why a cascade and not one buffer: a dashboard asks two different questions —
"what is happening right now" (seconds, full detail) and "what did the last
six hours look like" (minutes, aggregated). One buffer can serve only one of
them: fine enough for the first is far too much data for the second.

    tier 0  period 1 s    retain 10 min      600 points, raw samples
    tier 1  period 15 s   retain  6 h      1 440 points, aggregated from 0
    tier 2  period 300 s  retain  7 d      2 016 points, aggregated from 1

Resolutions are configuration, not constants (:class:`TierSpec`) — DESIGN_216
names the missing resolution setting explicitly.

**Aggregation keeps min/mean/max, never just the mean.** A mean hides exactly
what this project needs to see: a card that spikes to 88 C for twenty seconds
and settles back reads as "warm" in a five-minute mean and as "thermally
throttled" in the max. Peaks are the signal, not noise to be smoothed away.

The store is pure Python (no numpy, no deps), thread-safe, and serialisable to
plain JSON, because the collector on node B ships its tiers to the aggregator
on node A over HTTP.
"""

from __future__ import annotations

import dataclasses
import math
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "TierSpec",
    "DEFAULT_TIERS",
    "Aggregate",
    "Point",
    "Tier",
    "TimeSeries",
    "parse_tier_spec",
    "parse_duration",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TierSpec:
    """One resolution level. ``period_s`` is the bucket width, ``retain_s``
    how far back this tier reaches. Capacity follows from the two, so the
    memory cost of a configuration is explicit rather than emergent."""

    name: str
    period_s: float
    retain_s: float

    def __post_init__(self):
        if self.period_s <= 0:
            raise ValueError(f"tier {self.name!r}: period_s must be > 0")
        if self.retain_s < self.period_s:
            raise ValueError(
                f"tier {self.name!r}: retain_s ({self.retain_s}) is shorter "
                f"than one bucket ({self.period_s})"
            )

    @property
    def capacity(self) -> int:
        return max(1, int(math.ceil(self.retain_s / self.period_s)))


#: Sensible default cascade: 1 s live detail, 15 s for a session, 5 min for a
#: week. ~4 000 points per metric key in total.
DEFAULT_TIERS: Tuple[TierSpec, ...] = (
    TierSpec("live", 1.0, 600.0),
    TierSpec("session", 15.0, 6 * 3600.0),
    TierSpec("history", 300.0, 7 * 86400.0),
)


def parse_duration(text: str) -> float:
    """``"90"`` / ``"90s"`` / ``"15m"`` / ``"6h"`` / ``"7d"`` -> seconds."""
    t = str(text).strip().lower()
    if not t:
        raise ValueError("empty duration")
    mult = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
    if t[-1] in mult:
        return float(t[:-1]) * mult[t[-1]]
    return float(t)


def parse_tier_spec(text: str) -> TierSpec:
    """Parse one ``--resolution`` CLI item: ``NAME:PERIOD:RETAIN``, e.g.
    ``"live:1s:10m"`` or ``"history:5m:7d"``."""
    parts = [p.strip() for p in str(text).split(":")]
    if len(parts) != 3 or not parts[0]:
        raise ValueError(
            f"--resolution {text!r}: expected 'NAME:PERIOD:RETAIN' "
            "(e.g. 'live:1s:10m')"
        )
    return TierSpec(parts[0], parse_duration(parts[1]), parse_duration(parts[2]))


# ---------------------------------------------------------------------------
# Points
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Aggregate:
    """min / mean / max over the samples that fell into one coarse bucket.

    ``n`` is the number of contributing raw samples; a bucket with ``n``
    below the expected count is a bucket with gaps, and a reader that cares
    (e.g. "was the collector even running?") can tell.
    """

    min: float
    max: float
    sum: float
    n: int

    @property
    def mean(self) -> float:
        return self.sum / self.n if self.n else float("nan")

    def merge(self, other: "Aggregate") -> "Aggregate":
        return Aggregate(
            min=min(self.min, other.min),
            max=max(self.max, other.max),
            sum=self.sum + other.sum,
            n=self.n + other.n,
        )

    def add(self, value: float) -> None:
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        self.sum += value
        self.n += 1

    @classmethod
    def of(cls, value: float) -> "Aggregate":
        return cls(min=value, max=value, sum=value, n=1)

    def to_json(self) -> List[float]:
        # Compact wire form: [min, mean, max, n]. Chosen over a dict because a
        # week of history is ~2 000 buckets per key per node.
        return [self.min, self.mean, self.max, self.n]

    @classmethod
    def from_json(cls, v) -> "Aggregate":
        if isinstance(v, (int, float)):
            return cls.of(float(v))
        mn, mean, mx, n = v
        return cls(min=float(mn), max=float(mx), sum=float(mean) * int(n), n=int(n))


@dataclasses.dataclass
class Point:
    """One bucket: a timestamp plus the aggregate per metric key.

    In the raw tier every aggregate has ``n == 1`` and ``min == mean == max``;
    the type is uniform across tiers so readers need no special case.
    """

    ts: float
    values: Dict[str, Aggregate]

    def to_json(self) -> dict:
        return {"t": round(self.ts, 3), "v": {k: a.to_json() for k, a in self.values.items()}}

    @classmethod
    def from_json(cls, d: dict) -> "Point":
        return cls(
            ts=float(d["t"]),
            values={k: Aggregate.from_json(v) for k, v in (d.get("v") or {}).items()},
        )


# ---------------------------------------------------------------------------
# Tier
# ---------------------------------------------------------------------------


class Tier:
    """A fixed-capacity ring of buckets at one resolution."""

    def __init__(self, spec: TierSpec):
        self.spec = spec
        self._points: List[Point] = []

    # -- writing ------------------------------------------------------------

    def _bucket_start(self, ts: float) -> float:
        p = self.spec.period_s
        return math.floor(ts / p) * p

    def add(self, ts: float, values: Dict[str, float]) -> Optional[Point]:
        """Fold a sample into this tier. Returns the bucket that was CLOSED by
        this write (i.e. is now final and may be pushed down to the next,
        coarser tier), or None when the sample landed in the open bucket."""
        start = self._bucket_start(ts)
        closed: Optional[Point] = None
        if self._points and self._points[-1].ts == start:
            cur = self._points[-1]
            for k, v in values.items():
                if v is None:
                    continue
                a = cur.values.get(k)
                if a is None:
                    cur.values[k] = Aggregate.of(float(v))
                else:
                    a.add(float(v))
            return None
        if self._points and start < self._points[-1].ts:
            # Out-of-order sample older than the open bucket: fold it into the
            # matching historical bucket if it still exists, else drop it. A
            # clock step backwards must not corrupt the ordering invariant
            # every reader relies on.
            for p in reversed(self._points):
                if p.ts == start:
                    for k, v in values.items():
                        if v is None:
                            continue
                        a = p.values.get(k)
                        if a is None:
                            p.values[k] = Aggregate.of(float(v))
                        else:
                            a.add(float(v))
                    return None
                if p.ts < start:
                    break
            return None
        if self._points:
            closed = self._points[-1]
        self._points.append(
            Point(
                ts=start,
                values={
                    k: Aggregate.of(float(v)) for k, v in values.items() if v is not None
                },
            )
        )
        self._trim()
        return closed

    def add_point(self, point: Point) -> Optional[Point]:
        """Fold an already-aggregated bucket in (used when a coarse tier
        consumes a finer tier's closed bucket, and on push ingest)."""
        start = self._bucket_start(point.ts)
        closed: Optional[Point] = None
        if self._points and self._points[-1].ts == start:
            cur = self._points[-1]
            for k, a in point.values.items():
                cur.values[k] = cur.values[k].merge(a) if k in cur.values else dataclasses.replace(a)
            return None
        if self._points and start < self._points[-1].ts:
            for p in reversed(self._points):
                if p.ts == start:
                    for k, a in point.values.items():
                        p.values[k] = p.values[k].merge(a) if k in p.values else dataclasses.replace(a)
                    return None
                if p.ts < start:
                    break
            return None
        if self._points:
            closed = self._points[-1]
        self._points.append(
            Point(ts=start, values={k: dataclasses.replace(a) for k, a in point.values.items()})
        )
        self._trim()
        return closed

    def _trim(self) -> None:
        excess = len(self._points) - self.spec.capacity
        if excess > 0:
            del self._points[:excess]

    # -- reading ------------------------------------------------------------

    def points(
        self, since: Optional[float] = None, until: Optional[float] = None
    ) -> List[Point]:
        out = self._points
        if since is not None:
            out = [p for p in out if p.ts >= since]
        if until is not None:
            out = [p for p in out if p.ts <= until]
        return list(out)

    @property
    def span(self) -> Tuple[Optional[float], Optional[float]]:
        if not self._points:
            return (None, None)
        return (self._points[0].ts, self._points[-1].ts + self.spec.period_s)

    def __len__(self) -> int:
        return len(self._points)


# ---------------------------------------------------------------------------
# TimeSeries
# ---------------------------------------------------------------------------


class TimeSeries:
    """A cascade of :class:`Tier`s over one *stream* (one node, or one node's
    one GPU — the caller decides the granularity by choosing metric keys).

    Writes go into the finest tier; whenever a bucket there closes it is
    folded into the next tier, and so on. Downsampling therefore costs O(1)
    amortised per sample and never re-reads history.
    """

    def __init__(self, tiers: Sequence[TierSpec] = DEFAULT_TIERS):
        tiers = list(tiers)
        if not tiers:
            raise ValueError("a TimeSeries needs at least one tier")
        for a, b in zip(tiers, tiers[1:]):
            if b.period_s <= a.period_s:
                raise ValueError(
                    "tiers must be ordered fine -> coarse: "
                    f"{a.name} ({a.period_s}s) then {b.name} ({b.period_s}s)"
                )
        self.tiers: List[Tier] = [Tier(t) for t in tiers]
        self._lock = threading.RLock()
        self._last_sample: Optional[Tuple[float, Dict[str, float]]] = None

    # -- writing ------------------------------------------------------------

    def add(self, ts: float, values: Dict[str, Any]) -> None:
        """Record one sample. Non-numeric and None values are ignored here —
        they belong in the sample's *metadata*, not in the numeric series."""
        numeric = {
            k: float(v)
            for k, v in values.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        with self._lock:
            self._last_sample = (ts, numeric)
            closed = self.tiers[0].add(ts, numeric)
            for i in range(1, len(self.tiers)):
                if closed is None:
                    break
                closed = self.tiers[i].add_point(closed)

    def ingest_points(self, tier_name: str, points: Iterable[Point]) -> int:
        """Ingest already-bucketed points (aggregator side, push protocol).
        Points are folded into the named tier and cascade onward from there.
        Returns the number of points accepted."""
        with self._lock:
            idx = self._tier_index(tier_name)
            n = 0
            for p in points:
                closed = self.tiers[idx].add_point(p)
                for i in range(idx + 1, len(self.tiers)):
                    if closed is None:
                        break
                    closed = self.tiers[i].add_point(closed)
                n += 1
            return n

    # -- reading ------------------------------------------------------------

    def _tier_index(self, name: str) -> int:
        for i, t in enumerate(self.tiers):
            if t.spec.name == name:
                return i
        raise KeyError(
            f"unknown resolution {name!r}; configured: "
            f"{', '.join(t.spec.name for t in self.tiers)}"
        )

    def resolutions(self) -> List[dict]:
        with self._lock:
            out = []
            for t in self.tiers:
                lo, hi = t.span
                out.append(
                    {
                        "name": t.spec.name,
                        "period_s": t.spec.period_s,
                        "retain_s": t.spec.retain_s,
                        "capacity": t.spec.capacity,
                        "points": len(t),
                        "from": lo,
                        "to": hi,
                    }
                )
            return out

    def pick_resolution(
        self, window_s: float, max_points: int = 600
    ) -> str:
        """Choose the finest tier whose bucket count over ``window_s`` stays
        within ``max_points``, falling back to the coarsest tier. This is what
        lets the UI ask for "the last 6 hours" without knowing the cascade."""
        with self._lock:
            for t in self.tiers:
                if window_s / t.spec.period_s <= max_points:
                    return t.spec.name
            return self.tiers[-1].spec.name

    def query(
        self,
        resolution: Optional[str] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
        keys: Optional[Sequence[str]] = None,
        window_s: Optional[float] = None,
        max_points: int = 600,
        now: Optional[float] = None,
    ) -> dict:
        """Read a window. Either name a ``resolution`` or give a ``window_s``
        and let the store pick one.

        The reply carries the resolution actually used and the bucket period,
        so a caller can never mistake a five-minute mean for a live reading.
        """
        with self._lock:
            if resolution is None:
                if window_s is None:
                    resolution = self.tiers[0].spec.name
                else:
                    resolution = self.pick_resolution(window_s, max_points)
            idx = self._tier_index(resolution)
            tier = self.tiers[idx]
            if since is None and window_s is not None:
                ref = now if now is not None else (tier.span[1] or 0.0)
                since = ref - window_s
            pts = tier.points(since, until)
            if keys is not None:
                kset = set(keys)
                pts = [
                    Point(p.ts, {k: v for k, v in p.values.items() if k in kset})
                    for p in pts
                ]
            return {
                "resolution": resolution,
                "period_s": tier.spec.period_s,
                "aggregated": tier.spec.period_s > self.tiers[0].spec.period_s,
                "points": [p.to_json() for p in pts],
            }

    def latest(self) -> Optional[Tuple[float, Dict[str, float]]]:
        """The most recent raw sample as written (not bucketed). This is the
        "now" number; everything else in the store is a bucket."""
        with self._lock:
            if self._last_sample is None:
                return None
            ts, vals = self._last_sample
            return (ts, dict(vals))

    # -- transfer -----------------------------------------------------------

    def export_since(
        self, cursors: Optional[Dict[str, float]] = None
    ) -> Tuple[dict, Dict[str, float]]:
        """Everything newer than ``cursors`` (tier name -> last exported ts),
        plus the new cursor set. This is the collector -> aggregator payload:
        incremental, so a push every few seconds carries only new buckets, and
        a reconnect after an outage back-fills whatever is still in the ring.
        """
        cursors = dict(cursors or {})
        with self._lock:
            out: Dict[str, List[dict]] = {}
            new_cursors = dict(cursors)
            for t in self.tiers:
                after = cursors.get(t.spec.name)
                pts = [
                    p for p in t.points() if after is None or p.ts > after
                ]
                if pts:
                    out[t.spec.name] = [p.to_json() for p in pts]
                    new_cursors[t.spec.name] = pts[-1].ts
            return out, new_cursors
