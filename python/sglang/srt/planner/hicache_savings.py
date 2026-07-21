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
"""Persistent "HiCache energy-saved" accumulator (design #147).

The HiCache serves prefill tokens from a RAM (host) / disk (storage) prefix
cache instead of recomputing them. Every prefill token served from that cache
is a prefill recompute that DID NOT happen -> energy saved. This module keeps a
small, dependency-free JSON store, keyed by ``(model, config_label)``, that
ACCUMULATES over sessions how many prefill tokens the HiCache recovered, and
converts that running total into a kWh-saved and €-cent-saved BAND.

Four hard rules from the user (#147), enforced structurally here:

  1. THE RECOVERY COSTS NOTHING EXTRA. RAM and disk draw the same power whether
     or not you fetch from them — that draw is sunk cost. So the saved energy is
     PURELY the avoided prefill recompute:

         saved_J = recovered_prefill_tokens * j_per_prefill_token

     No fetch / paging / transfer energy is ever subtracted. There is no code
     path in this module that debits the saving; ``saved_joules_band`` is a bare
     multiply. (Test: ``test_no_fetch_energy_deducted``.)

  2. A VON–BIS BAND, NOT A POINT. ``j_per_prefill_token`` varies by batch-size
     bucket, so the saving is a range: the band edges are the MIN and MAX
     measured ``j_per_prefill_token`` across buckets (``band_from_buckets``).

  3. PERSISTENT, PER (model, config_label). The total keeps growing across
     restarts — ``HiCacheSavingsStore.load``/``save`` round-trips a JSON store on
     disk; recording appends to the running total, never resets it.

  4. saved_kWh = saved_J / 3.6e6 ; saved_ct = saved_kWh * price_ct_per_kWh
     (currency is €-cent, matching the rest of the dashboard).

Provenance discipline (mirrors ``results_store.py`` §5B.3a): the J/prefill-token
band is a MEASURED input. A measured band, once set, is NEVER overwritten by an
estimate; only a fresh measurement may replace it. The recovered-token COUNT is
an operational counter (a tally of cache hits), not a perf measurement, so it is
always safe to accumulate.

The recovered-token signal itself comes from the running server's Prometheus
counter ``sglang:cached_tokens_total{cache_source=...}`` (see
``metrics_collector.py``). Only the RAM/disk tiers count as HiCache recovery —
``cache_source="host"`` (RAM) and ``cache_source="storage_*"`` (disk backend).
``cache_source="device"`` is the on-GPU radix cache (already-resident KV, not the
RAM/disk offload) and is excluded; the ``"total"`` fallback label (emitted only
when the per-source breakdown is unavailable) is likewise excluded by default
because it lumps the device tier in. See ``HICACHE_CACHE_SOURCES`` /
``cached_tokens_by_source``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "JOULES_PER_KWH",
    "HICACHE_CACHE_SOURCES",
    "band_from_buckets",
    "cached_tokens_by_source",
    "hicache_recovered_from_metrics",
    "HiCacheSavingRecord",
    "HiCacheSavingsStore",
    "MEASURED_BAND_PROVENANCE",
    "DEFAULT_HICACHE_STORE",
]

#: 1 kWh = 3.6e6 J (design #147 rule 4). The single conversion constant.
JOULES_PER_KWH = 3.6e6

#: Provenance value that marks the J/prefill-token band as a genuine measurement
#: (mirrors ``results_store.MEASURED_PROVENANCE``). A measured band is never
#: overwritten by an estimate.
MEASURED_BAND_PROVENANCE = "measured"


def _is_hicache_source(source: str) -> bool:
    """A ``cache_source`` label counts as HiCache RAM/disk recovery iff it is the
    host (RAM) tier or a storage (disk) backend tier. ``device`` (GPU radix) and
    the ``total`` fallback are NOT HiCache offload and are excluded."""
    s = str(source)
    return s == "host" or s.startswith("storage")


#: Predicate object exposed for documentation/tests: the set membership test for
#: "is this cache_source a HiCache RAM/disk tier?".
HICACHE_CACHE_SOURCES = _is_hicache_source


# ---------------------------------------------------------------------------
# Prometheus: recovered-prefill-token counter, split by cache_source.
# ---------------------------------------------------------------------------

_CACHED_TOKENS_METRIC = "sglang:cached_tokens_total"
_CACHE_SOURCE_RE = re.compile(r'cache_source="([^"]*)"')


def cached_tokens_by_source(metrics_text: str) -> Dict[str, float]:
    """Parse ``sglang:cached_tokens_total{...,cache_source="X"} V`` lines out of a
    Prometheus ``/metrics`` scrape into ``{cache_source: summed_value}``.

    ``energy.parse_prometheus_metrics`` collapses ALL label sets of a metric into
    one number, which would fuse device+host+storage into a single figure — the
    exact opposite of what HiCache accounting needs. This keeps the per-source
    split so ``hicache_recovered_from_metrics`` can pick only the RAM/disk tiers.
    """
    out: Dict[str, float] = {}
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith(_CACHED_TOKENS_METRIC):
            continue
        brace = line.find("{")
        if brace < 0:
            # No labels at all -> a bare total (no per-source breakdown).
            rest = line[len(_CACHED_TOKENS_METRIC):].strip()
            source = "total"
        else:
            labels = line[brace + 1 : line.find("}", brace)]
            rest = line[line.find("}", brace) + 1 :].strip()
            m = _CACHE_SOURCE_RE.search(labels)
            source = m.group(1) if m else "total"
        val = rest.split()[0] if rest else ""
        try:
            out[source] = out.get(source, 0.0) + float(val)
        except ValueError:
            continue
    return out


def hicache_recovered_from_metrics(metrics_text: str) -> float:
    """Total recovered-prefill-token counter across the HiCache RAM/disk tiers.

    This is the cumulative ``sglang:cached_tokens_total`` restricted to
    ``cache_source in {host, storage_*}`` — the prompt tokens served from the
    RAM/disk prefix cache that would otherwise have been prefilled. It is an
    absolute counter value (monotonic within one server lifetime), fed to
    ``HiCacheSavingRecord.record_counter_snapshot`` which turns successive
    snapshots into a persisted running total."""
    by_source = cached_tokens_by_source(metrics_text)
    return sum(v for src, v in by_source.items() if _is_hicache_source(src))


# ---------------------------------------------------------------------------
# J/prefill-token band (min/max across buckets — never a scalar).
# ---------------------------------------------------------------------------


def band_from_buckets(
    j_per_prefill_token_by_bucket: Optional[Dict],
) -> Optional[Tuple[float, float]]:
    """(lo, hi) = (min, max) measured ``j_per_prefill_token`` across buckets.

    The per-token prefill energy varies with the batch-size bucket, so the saving
    is presented as a von–bis band whose edges are the cheapest and most
    expensive measured buckets (design #147 rule 2). Returns None when there is no
    measured bucket to derive a band from — never fabricates one."""
    if not j_per_prefill_token_by_bucket:
        return None
    vals = [float(v) for v in j_per_prefill_token_by_bucket.values() if v is not None]
    if not vals:
        return None
    return (min(vals), max(vals))


# ---------------------------------------------------------------------------
# One persisted accumulator record, keyed by (model, config_label).
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class HiCacheSavingRecord:
    """Accumulated HiCache recovery for one ``(model, config_label)``.

    ``recovered_prefill_tokens`` is the persisted running total; the kWh/ct bands
    are DERIVED on demand from that total and the measured J/prefill-token band —
    they are never stored as their own scalar (no false precision)."""

    model: str
    config_label: str
    #: cumulative HiCache-recovered prefill tokens (grows across sessions).
    recovered_prefill_tokens: float = 0.0
    #: measured J/prefill-token band edges (min/max across buckets). None until a
    #: measured band is supplied.
    j_per_prefill_token_lo: Optional[float] = None
    j_per_prefill_token_hi: Optional[float] = None
    #: provenance of the band above ("measured" | None). A measured band is never
    #: clobbered by an estimate (results_store §5B.3a discipline).
    band_provenance: Optional[str] = None
    #: last absolute ``cached_tokens_total`` counter snapshot, so the next scrape
    #: records only the DELTA (and a counter reset is handled, not double-counted).
    last_counter_total: Optional[float] = None
    created_at: Optional[float] = None
    updated_at: Optional[float] = None

    # -- key --------------------------------------------------------------

    def key(self) -> Tuple[str, str]:
        return (self.model, self.config_label)

    # -- band maintenance (provenance-guarded) ----------------------------

    def has_band(self) -> bool:
        return (
            self.j_per_prefill_token_lo is not None
            and self.j_per_prefill_token_hi is not None
        )

    def set_band(
        self, lo: float, hi: float, provenance: str = MEASURED_BAND_PROVENANCE
    ) -> bool:
        """Set the J/prefill-token band. Returns True if applied.

        Provenance discipline: an estimate MUST NOT overwrite a band that was
        already set from a measurement. A measurement may replace anything
        (a re-measure is authoritative). Same discipline as the results store."""
        if hi < lo:
            raise ValueError(f"band hi<lo: ({lo}, {hi})")
        if (
            self.band_provenance == MEASURED_BAND_PROVENANCE
            and provenance != MEASURED_BAND_PROVENANCE
        ):
            return False  # never downgrade a measured band with an estimate
        self.j_per_prefill_token_lo = float(lo)
        self.j_per_prefill_token_hi = float(hi)
        self.band_provenance = provenance
        self._touch()
        return True

    def set_band_from_buckets(
        self, buckets: Optional[Dict], provenance: str = MEASURED_BAND_PROVENANCE
    ) -> bool:
        band = band_from_buckets(buckets)
        if band is None:
            return False
        return self.set_band(band[0], band[1], provenance)

    # -- token accumulation ----------------------------------------------

    def add_tokens(self, n: float) -> None:
        """Accumulate ``n`` newly-recovered prefill tokens (manual delta path)."""
        if n < 0:
            raise ValueError(f"recovered tokens delta must be >= 0, got {n}")
        self.recovered_prefill_tokens += float(n)
        self._touch()

    def record_counter_snapshot(self, total: float) -> float:
        """Record an absolute ``cached_tokens_total`` counter snapshot and add the
        DELTA since the last snapshot to the running total. Returns the delta
        that was added.

        Guards a counter RESET (server restart): if the new absolute value is
        below the previous snapshot, the counter restarted, so the new value
        itself is the post-reset delta (mirrors the reset guard in
        ``energy.compute_live_rates``, but ACCUMULATING rather than rate-limiting
        to zero — we must not lose the tokens recovered since the restart)."""
        total = float(total)
        if total < 0:
            raise ValueError(f"counter total must be >= 0, got {total}")
        if self.last_counter_total is None:
            delta = total  # first snapshot: everything so far counts once
        elif total >= self.last_counter_total:
            delta = total - self.last_counter_total
        else:
            delta = total  # counter reset -> fresh accumulation
        self.last_counter_total = total
        if delta:
            self.recovered_prefill_tokens += delta
            self._touch()
        return delta

    # -- derived saving bands (NO fetch-energy deduction) -----------------

    def saved_joules_band(self) -> Optional[Tuple[float, float]]:
        """(lo, hi) Joules saved = recovered_prefill_tokens * (band edges).

        This is the WHOLE saving: purely the avoided prefill recompute. Nothing
        is subtracted for the RAM/disk fetch — that power is drawn regardless
        (design #147 rule 1). A bare multiply, deliberately."""
        if not self.has_band():
            return None
        lo = self.recovered_prefill_tokens * self.j_per_prefill_token_lo
        hi = self.recovered_prefill_tokens * self.j_per_prefill_token_hi
        return (lo, hi)

    def saved_kwh_band(self) -> Optional[Tuple[float, float]]:
        j = self.saved_joules_band()
        if j is None:
            return None
        return (j[0] / JOULES_PER_KWH, j[1] / JOULES_PER_KWH)

    def saved_ct_band(self, price_ct_per_kwh: float) -> Optional[Tuple[float, float]]:
        kwh = self.saved_kwh_band()
        if kwh is None:
            return None
        return (kwh[0] * price_ct_per_kwh, kwh[1] * price_ct_per_kwh)

    # -- (de)serialization ------------------------------------------------

    def _touch(self) -> None:
        now = time.time()
        if self.created_at is None:
            self.created_at = now
        self.updated_at = now

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    def to_view(self, price_ct_per_kwh: float) -> dict:
        """The dashboard-facing view: the record plus its derived bands (as
        ``[lo, hi]`` lists, or None when no measured band exists yet)."""
        def _band(b):
            return list(b) if b is not None else None

        return {
            "model": self.model,
            "config_label": self.config_label,
            "recovered_prefill_tokens": self.recovered_prefill_tokens,
            "j_per_prefill_token_band": (
                [self.j_per_prefill_token_lo, self.j_per_prefill_token_hi]
                if self.has_band()
                else None
            ),
            "band_provenance": self.band_provenance,
            "saved_kwh_band": _band(self.saved_kwh_band()),
            "saved_ct_band": _band(self.saved_ct_band(price_ct_per_kwh)),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_json(cls, d: dict) -> "HiCacheSavingRecord":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# The persistent store (JSON, keyed by (model, config_label)).
# ---------------------------------------------------------------------------


class HiCacheSavingsStore:
    """A tiny persistent map ``(model, config_label) -> HiCacheSavingRecord``.

    Persistence is a single JSON file (list of records). ``save`` writes
    atomically (tmp + os.replace) so a concurrent reader never sees a half file.
    The store ACCUMULATES: loading + recording + saving grows the totals; it
    never resets on restart (design #147 rule 3)."""

    def __init__(self, records: Optional[Sequence[HiCacheSavingRecord]] = None):
        self._records: Dict[Tuple[str, str], HiCacheSavingRecord] = {}
        for r in records or []:
            self._records[r.key()] = r

    def __len__(self) -> int:
        return len(self._records)

    def records(self) -> List[HiCacheSavingRecord]:
        return list(self._records.values())

    def get(self, model: str, config_label: str) -> Optional[HiCacheSavingRecord]:
        return self._records.get((model, config_label))

    def get_or_create(self, model: str, config_label: str) -> HiCacheSavingRecord:
        key = (model, config_label)
        rec = self._records.get(key)
        if rec is None:
            rec = HiCacheSavingRecord(model=model, config_label=config_label)
            self._records[key] = rec
        return rec

    # -- grand total ------------------------------------------------------

    def grand_total_saved_kwh_band(self) -> Optional[Tuple[float, float]]:
        """Sum of every record's saved-kWh band (lo with lo, hi with hi). Records
        without a measured band contribute nothing (not a zero that would drag a
        real band down — they are simply absent). None if no record has a band."""
        los, his = 0.0, 0.0
        any_band = False
        for r in self._records.values():
            b = r.saved_kwh_band()
            if b is None:
                continue
            any_band = True
            los += b[0]
            his += b[1]
        return (los, his) if any_band else None

    def grand_total_saved_ct_band(
        self, price_ct_per_kwh: float
    ) -> Optional[Tuple[float, float]]:
        kwh = self.grand_total_saved_kwh_band()
        if kwh is None:
            return None
        return (kwh[0] * price_ct_per_kwh, kwh[1] * price_ct_per_kwh)

    def grand_total_recovered_tokens(self) -> float:
        return sum(r.recovered_prefill_tokens for r in self._records.values())

    def to_view(self, price_ct_per_kwh: float) -> dict:
        return {
            "records": [
                r.to_view(price_ct_per_kwh)
                for r in sorted(
                    self._records.values(),
                    key=lambda r: (r.model, r.config_label),
                )
            ],
            "grand_total": {
                "recovered_prefill_tokens": self.grand_total_recovered_tokens(),
                "saved_kwh_band": (
                    list(self.grand_total_saved_kwh_band())
                    if self.grand_total_saved_kwh_band() is not None
                    else None
                ),
                "saved_ct_band": (
                    list(self.grand_total_saved_ct_band(price_ct_per_kwh))
                    if self.grand_total_saved_ct_band(price_ct_per_kwh) is not None
                    else None
                ),
            },
            "price_ct_per_kwh": price_ct_per_kwh,
        }

    # -- persistence ------------------------------------------------------

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump([r.to_json() for r in self._records.values()], f, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "HiCacheSavingsStore":
        store = cls()
        if not os.path.exists(path):
            return store
        try:
            with open(path) as f:
                data = json.load(f)
        except (ValueError, OSError):
            return store  # corrupt/absent store -> start empty, never crash the UI
        for d in data or []:
            try:
                rec = HiCacheSavingRecord.from_json(d)
            except TypeError:
                continue
            store._records[rec.key()] = rec
        return store


#: Default on-disk location, next to the measured results store.
DEFAULT_HICACHE_STORE = os.path.join(
    os.path.dirname(__file__), "hicache_savings.json"
)
