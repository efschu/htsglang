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
"""Persistent, resettable Joules-per-token counter for the dashboard.

Keeps a running (model, config_label, lanes) -> energy tally, PREFILL and
DECODE kept as two entirely separate accumulators (never blended into one
number), backed by NVML-measured power -- never the roofline estimate. This
is a thin persistence + accounting layer on top of the two places real
NVML-integrated energy already gets produced in this codebase:

  1. ``energy.py``'s ``EnergyHarness`` (design #146): boots a server, drives a
     controlled batch sweep, and integrates board power over the EXACT
     wall-clock window of each phase (prefill window = first-request-sent to
     last-request's-first-token; decode window = that to last-request-done),
     via a 20ms background ``PowerSampler`` + trapezoidal integration. This is
     the highest-fidelity source: ``record_harness_result`` feeds a completed
     ``MeasurementResult`` in wholesale after a scenario run.

  2. The dashboard's own landing-page live poll (``live_metrics.snapshot``,
     every ``LAND_POLL_MS`` -- about 2s -- against a RUNNING/production
     server): each poll already carries an instantaneous per-card NVML power
     reading (piggybacked on the poll that already happens for the tok/s
     widget -- no extra network round-trip) plus the exact non-cached
     prefill/decode token-rate deltas for that window. ``record_live_tick``
     turns one such window into an energy contribution with a plain
     RECTANGLE-rule estimate (instant total watts * dt) -- coarser than the
     harness's 20ms trapezoidal integration, but zero extra cost and still a
     real NVML reading, never a modelled number.

Toggle discipline (opt-in, default OFF): accumulation costs are the poll
already happening (READ side is always live; nothing here adds a network
call) plus a small periodic disk write while ENABLED. When disabled,
``record_live_tick``/``record_harness_result`` return immediately without
building a single dict or touching the store -- toggled off costs nothing
beyond the boolean check itself (see ``test_toggle_off_is_free``).

Phase attribution honesty: a live-poll window is attributed WHOLLY to
prefill when only prefill tokens moved in it, wholly to decode when only
decode tokens moved. When BOTH moved in the same ~2s window (bursty mixed
traffic) the split cannot be measured, only guessed -- so it is NOT
apportioned by some invented ratio. It is tracked separately in the record's
``mixed_*`` fields, fully visible, never silently folded into the pure
prefill/decode totals. The harness path never produces mixed windows (its
prefill/decode windows are exact request timestamps, not a coarse poll tick),
so every harness-fed contribution is phase-pure by construction.

Dual-group readiness: the key carries a ``lanes: List[str]`` dimension (one
lane today) rather than a hardcoded single/pair of GPU groups, so a second
lane (a second rig or a second co-scheduled group on this rig) is just
another list element, not a schema change.

Persistence mirrors ``hicache_savings.py`` (design #147)'s pattern exactly:
a single JSON file in the shared planner DATA dir (self_update code/data
separation, design #275), atomic writes, corrupt/missing file -> start empty
rather than crash the UI, and every WRITE path (toggle, tick, reset) goes
through ``self_update.data_write_guard`` so a dashboard older than the data
dir's schema stamp never corrupts a newer store format.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sglang.srt.planner.self_update import data_write_guard, planner_data_path

__all__ = [
    "JOULES_PER_KWH",
    "MEASURED_PROVENANCE",
    "SOURCE_LIVE_POLL",
    "SOURCE_HARNESS",
    "JtokCounterRecord",
    "JtokCounterStore",
    "record_live_tick",
    "record_harness_result",
    "DEFAULT_JTOK_STORE",
]

#: 1 kWh = 3.6e6 J -- same single conversion constant used by hicache_savings.
JOULES_PER_KWH = 3.6e6

#: Every number this module ever stores comes from an NVML power reading
#: integrated over a real wall-clock phase window -- never a roofline/model
#: estimate. There is only one provenance value; ``sources`` (below) records
#: WHICH measurement path(s) contributed (their sampling fidelity differs,
#: but both are genuine measurements).
MEASURED_PROVENANCE = "measured"

#: recording-path tags for JtokCounterRecord.sources.
SOURCE_LIVE_POLL = "live_poll"
SOURCE_HARNESS = "harness"


def _lane_key(lanes: Sequence[str]) -> Tuple[str, ...]:
    return tuple(str(x) for x in lanes)


# ---------------------------------------------------------------------------
# One persisted accumulator record, keyed by (model, config_label, lanes).
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class JtokCounterRecord:
    """Accumulated prefill/decode energy for one (model, config_label, lanes).

    ``lanes`` is a list (today always length 1) so a second co-scheduled lane
    is an additional list element, never a hardcoded second field."""

    model: str
    config_label: str
    lanes: List[str] = dataclasses.field(default_factory=list)
    #: pure-prefill accumulators: whole-window NVML joules / non-cached
    #: prefill tokens, only from windows where NO decode token moved.
    prefill_joules: float = 0.0
    prefill_tokens: float = 0.0
    #: pure-decode accumulators: symmetric, only from windows where NO
    #: prefill token moved.
    decode_joules: float = 0.0
    decode_tokens: float = 0.0
    #: windows where BOTH phases were active at once (live-poll only -- the
    #: harness's exact request-timestamp windows never produce these). Kept
    #: fully separate: never apportioned into the pure totals above by a
    #: guessed split.
    mixed_joules: float = 0.0
    mixed_prefill_tokens: float = 0.0
    mixed_decode_tokens: float = 0.0
    mixed_windows: int = 0
    #: which recording path(s) have ever fed this record (SOURCE_LIVE_POLL /
    #: SOURCE_HARNESS), for transparency about sampling fidelity.
    sources: List[str] = dataclasses.field(default_factory=list)
    created_at: Optional[float] = None
    updated_at: Optional[float] = None

    # -- key ----------------------------------------------------------------

    def key(self) -> Tuple[str, str, Tuple[str, ...]]:
        return (self.model, self.config_label, _lane_key(self.lanes))

    # -- accumulation ---------------------------------------------------

    def _note_source(self, source: str) -> None:
        if source not in self.sources:
            self.sources.append(source)

    def add_prefill(self, joules: float, tokens: float, source: str) -> None:
        if joules < 0 or tokens < 0:
            raise ValueError(f"joules/tokens must be >= 0, got ({joules}, {tokens})")
        if tokens == 0:
            return
        self.prefill_joules += float(joules)
        self.prefill_tokens += float(tokens)
        self._note_source(source)
        self._touch()

    def add_decode(self, joules: float, tokens: float, source: str) -> None:
        if joules < 0 or tokens < 0:
            raise ValueError(f"joules/tokens must be >= 0, got ({joules}, {tokens})")
        if tokens == 0:
            return
        self.decode_joules += float(joules)
        self.decode_tokens += float(tokens)
        self._note_source(source)
        self._touch()

    def add_mixed(self, joules: float, prefill_tokens: float, decode_tokens: float,
                   source: str) -> None:
        if joules < 0 or prefill_tokens < 0 or decode_tokens < 0:
            raise ValueError("joules/tokens must be >= 0")
        self.mixed_joules += float(joules)
        self.mixed_prefill_tokens += float(prefill_tokens)
        self.mixed_decode_tokens += float(decode_tokens)
        self.mixed_windows += 1
        self._note_source(source)
        self._touch()

    # -- reset ------------------------------------------------------------

    def reset(self) -> None:
        """Zero every accumulator (identity -- model/config_label/lanes --
        and ``sources`` history are kept: a reset counter still exists as a
        row, it just restarts from 0, matching hicache's touch/save
        discipline rather than deleting the record)."""
        self.prefill_joules = 0.0
        self.prefill_tokens = 0.0
        self.decode_joules = 0.0
        self.decode_tokens = 0.0
        self.mixed_joules = 0.0
        self.mixed_prefill_tokens = 0.0
        self.mixed_decode_tokens = 0.0
        self.mixed_windows = 0
        self._touch()

    # -- derived ------------------------------------------------------------

    def j_per_prefill_token(self) -> Optional[float]:
        return self.prefill_joules / self.prefill_tokens if self.prefill_tokens else None

    def j_per_decode_token(self) -> Optional[float]:
        return self.decode_joules / self.decode_tokens if self.decode_tokens else None

    def mixed_j_per_token_blended(self) -> Optional[float]:
        """Honest blended figure for the mixed bucket: total mixed joules over
        total mixed tokens (prefill+decode together) -- explicitly labelled
        'blended' wherever shown, never presented as a phase-separated
        number."""
        total_tok = self.mixed_prefill_tokens + self.mixed_decode_tokens
        return self.mixed_joules / total_tok if total_tok else None

    # -- (de)serialization ------------------------------------------------

    def _touch(self) -> None:
        now = time.time()
        if self.created_at is None:
            self.created_at = now
        self.updated_at = now

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    def to_view(self) -> dict:
        return {
            "model": self.model,
            "config_label": self.config_label,
            "lanes": list(self.lanes),
            "provenance": MEASURED_PROVENANCE,
            "sources": list(self.sources),
            "prefill_tokens": self.prefill_tokens,
            "j_per_prefill_token": self.j_per_prefill_token(),
            "decode_tokens": self.decode_tokens,
            "j_per_decode_token": self.j_per_decode_token(),
            "mixed_windows": self.mixed_windows,
            "mixed_prefill_tokens": self.mixed_prefill_tokens,
            "mixed_decode_tokens": self.mixed_decode_tokens,
            "mixed_j_per_token_blended": self.mixed_j_per_token_blended(),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_json(cls, d: dict) -> "JtokCounterRecord":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# The persistent store (JSON, keyed by (model, config_label, lanes)), plus
# the opt-in toggle (default OFF, persisted alongside the records).
# ---------------------------------------------------------------------------


class JtokCounterStore:
    """A tiny persistent map (model, config_label, lanes) -> JtokCounterRecord,
    plus the feature's own enabled toggle (default False)."""

    def __init__(self, records: Optional[Sequence[JtokCounterRecord]] = None,
                 enabled: bool = False):
        self._records: Dict[Tuple[str, str, Tuple[str, ...]], JtokCounterRecord] = {}
        for r in records or []:
            self._records[r.key()] = r
        self.enabled = bool(enabled)

    def __len__(self) -> int:
        return len(self._records)

    def records(self) -> List[JtokCounterRecord]:
        return list(self._records.values())

    def get(self, model: str, config_label: str,
            lanes: Sequence[str]) -> Optional[JtokCounterRecord]:
        return self._records.get((model, config_label, _lane_key(lanes)))

    def get_or_create(self, model: str, config_label: str,
                       lanes: Sequence[str]) -> JtokCounterRecord:
        key = (model, config_label, _lane_key(lanes))
        rec = self._records.get(key)
        if rec is None:
            rec = JtokCounterRecord(model=model, config_label=config_label,
                                     lanes=list(lanes))
            self._records[key] = rec
        return rec

    # -- reset --------------------------------------------------------------

    def reset_one(self, model: str, config_label: str, lanes: Sequence[str]) -> bool:
        rec = self.get(model, config_label, lanes)
        if rec is None:
            return False
        rec.reset()
        return True

    def reset_all(self) -> int:
        n = 0
        for rec in self._records.values():
            rec.reset()
            n += 1
        return n

    # -- grand total ----------------------------------------------------

    def grand_total(self) -> dict:
        pj = pt = dj = dt = 0.0
        for r in self._records.values():
            pj += r.prefill_joules
            pt += r.prefill_tokens
            dj += r.decode_joules
            dt += r.decode_tokens
        return {
            "prefill_tokens": pt,
            "j_per_prefill_token": (pj / pt) if pt else None,
            "decode_tokens": dt,
            "j_per_decode_token": (dj / dt) if dt else None,
        }

    def to_view(self) -> dict:
        return {
            "enabled": self.enabled,
            "records": [
                r.to_view()
                for r in sorted(
                    self._records.values(),
                    key=lambda r: (r.model, r.config_label, tuple(r.lanes)),
                )
            ],
            "grand_total": self.grand_total(),
        }

    # -- persistence ------------------------------------------------------

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        obj = {
            "enabled": self.enabled,
            "records": [r.to_json() for r in self._records.values()],
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "JtokCounterStore":
        store = cls()
        if not os.path.exists(path):
            return store
        try:
            with open(path) as f:
                data = json.load(f)
        except (ValueError, OSError):
            return store  # corrupt/absent store -> start empty, never crash the UI
        if isinstance(data, dict):
            store.enabled = bool(data.get("enabled", False))
            rows = data.get("records") or []
        else:
            rows = data or []  # tolerate a bare list (defensive, no such format was ever written)
        for d in rows:
            try:
                rec = JtokCounterRecord.from_json(d)
            except TypeError:
                continue
            store._records[rec.key()] = rec
        return store


#: Default on-disk location: the shared planner DATA dir (~/.cache/sglang),
#: NOT the package dir -- design #275 code/data separation, same call
#: convention as hicache_savings.DEFAULT_HICACHE_STORE / energy's
#: DEFAULT_RESULTS_STORE (no legacy in-package copy exists for this brand-new
#: store; the parameter is passed anyway for convention consistency).
DEFAULT_JTOK_STORE = planner_data_path(
    "jtok_counter.json",
    legacy=os.path.join(os.path.dirname(__file__), "jtok_counter.json"),
)


# ---------------------------------------------------------------------------
# Recording entry points.
# ---------------------------------------------------------------------------


def record_live_tick(
    store: JtokCounterStore,
    *,
    model: str,
    config_label: str,
    lanes: Sequence[str],
    dt: Optional[float],
    prefill_tok_s: float,
    decode_tok_s: float,
    total_watts: float,
) -> Optional[dict]:
    """Fold one landing-page live-poll window into the counter.

    Toggle-off is the null-overhead path: returns immediately, before any
    arithmetic or dict construction. When ``dt`` is missing/non-positive
    (first poll after a target change, or a clock-jitter duplicate scrape --
    the same conditions ``live_metrics._rates`` itself guards) there is no
    valid window to attribute, so this is also a no-op.

    ``total_watts`` is the CURRENT instantaneous whole-rig NVML power reading
    (sum of per-card ``power_watts`` from the same snapshot); the window's
    energy is estimated as ``total_watts * dt`` (rectangle rule) -- see the
    module docstring for why this is coarser than, but not less genuinely
    NVML-MEASURED than, the harness's trapezoidal integration.
    """
    if not store.enabled:
        return None
    if dt is None or dt <= 0:
        return None
    prefill_tokens = max(0.0, prefill_tok_s) * dt
    decode_tokens = max(0.0, decode_tok_s) * dt
    if prefill_tokens <= 0 and decode_tokens <= 0:
        return None  # idle window: no tokens moved, nothing to attribute energy to
    joules = max(0.0, total_watts) * dt
    rec = store.get_or_create(model, config_label, lanes)
    if prefill_tokens > 0 and decode_tokens > 0:
        rec.add_mixed(joules, prefill_tokens, decode_tokens, source=SOURCE_LIVE_POLL)
    elif prefill_tokens > 0:
        rec.add_prefill(joules, prefill_tokens, source=SOURCE_LIVE_POLL)
    else:
        rec.add_decode(joules, decode_tokens, source=SOURCE_LIVE_POLL)
    return rec.to_view()


def record_harness_result(
    store: JtokCounterStore,
    *,
    model: str,
    config_label: str,
    lanes: Sequence[str],
    measurements: Sequence[Any],
) -> List[dict]:
    """Fold a completed EnergyHarness ``MeasurementResult.measurements`` (a
    sequence of ``energy.BucketMeasurement``, duck-typed here to avoid a
    circular import with energy.py) into the counter, one contribution per
    bucket. Every harness window is phase-pure by construction (bounded by
    exact request timestamps), so this never touches the mixed_* fields."""
    if not store.enabled:
        return []
    out = []
    for m in measurements:
        total_prefill_tok = float(m.prompt_tokens) * m.n_requests
        total_decode_tok = float(m.decode_tokens) * m.n_requests
        rec = store.get_or_create(model, config_label, lanes)
        if total_prefill_tok > 0:
            rec.add_prefill(m.prefill_joules, total_prefill_tok, source=SOURCE_HARNESS)
        if total_decode_tok > 0:
            rec.add_decode(m.decode_joules, total_decode_tok, source=SOURCE_HARNESS)
        out.append(rec.to_view())
    return out


# ---------------------------------------------------------------------------
# Guarded read/write helpers (mirrors hicache_saved_read/record's guard use).
# ---------------------------------------------------------------------------


def jtok_read(path: str = DEFAULT_JTOK_STORE) -> dict:
    """Read-only view of the persisted store (never blocked by the schema
    write-guard -- reading an older-format store is always safe)."""
    return JtokCounterStore.load(path).to_view()


def jtok_set_enabled(enabled: bool, path: str = DEFAULT_JTOK_STORE) -> dict:
    guard = data_write_guard()
    if guard:
        return {"ok": False, "error": guard}
    store = JtokCounterStore.load(path)
    store.enabled = bool(enabled)
    store.save(path)
    return {"ok": True, **store.to_view()}


def jtok_reset(path: str = DEFAULT_JTOK_STORE, *,
               model: Optional[str] = None,
               config_label: Optional[str] = None,
               lanes: Optional[Sequence[str]] = None,
               reset_all: bool = False) -> dict:
    guard = data_write_guard()
    if guard:
        return {"ok": False, "error": guard}
    store = JtokCounterStore.load(path)
    if reset_all:
        n = store.reset_all()
        store.save(path)
        return {"ok": True, "reset_count": n, **store.to_view()}
    if model is None or config_label is None or lanes is None:
        return {"ok": False, "error": "reset requires model+config_label+lanes, "
                                       "or reset_all=true"}
    found = store.reset_one(model, config_label, lanes)
    if not found:
        return {"ok": False, "error": "no such counter"}
    store.save(path)
    return {"ok": True, **store.to_view()}
