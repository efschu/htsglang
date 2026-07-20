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
"""Local benchmark results store + ingest guard (design #97 stage S5, §5B.1).

A dependency-free ``results.jsonl`` store (append-only, no server) of measured
runs, feeding the crowdsourced landscape (``landscape.py``). The schema is
built NOW; the measured perf/energy columns come from S2.5 (energy module,
GPU) and stay structurally ABSENT until then — never fabricated.

STRUCTURAL honesty (design §5A.5 / §5B.3a):
  * INGEST GUARD — the store holds only MEASURED runs. An entry that carries
    perf/energy numbers without measured provenance is REJECTED at ingest
    (``IngestRejected``), not stored with a caveat. A pure feasibility/
    capacity record (no perf numbers) is not a benchmark result and does not
    belong here — it is computed on the fly by the landscape from the planner.
  * BANDS, NOT SCALARS — kWh/seconds saved are stored as (lo, hi) bands; a
    bare scalar is rejected, because false precision is itself a dishonesty
    failure (§5A.4/§5A.5). Per-token energy is a per-batch-size-bucket CURVE,
    never one flat number.
  * ANONYMOUS — hardware is a class (card model + count) via the S2
    fingerprint; the §4.5 scrub already dropped UUIDs/paths/hostnames.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "QuantDescriptor",
    "Band",
    "ResultEntry",
    "ResultsStore",
    "IngestRejected",
    "MEASURED_PROVENANCE",
]

#: Provenance values that count as a genuine measurement (design §5B.3a). Only
#: these may carry perf/energy numbers into the store.
MEASURED_PROVENANCE = ("measured", "boot-log-measured")


class IngestRejected(ValueError):
    """An entry was refused at ingest (unmeasured perf, bad band, ...). The
    store never stores-with-a-caveat; a rejected number is simply absent."""


# ---------------------------------------------------------------------------
# Canonical quant descriptor (design §5B.3).
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class QuantDescriptor:
    """``(nominal_bits, scheme, group_size)`` — the canonical quant key.

    Exact-match is the default grouping; a "similar quant" view groups by
    ``nominal_bits`` only and MUST be labelled approximate (scheme/group-size
    shift both efficiency and quality — never silently mixed, §5B.3)."""

    nominal_bits: float
    scheme: str
    group_size: Optional[int] = None

    @classmethod
    def parse(cls, text: str) -> "QuantDescriptor":
        """Parse a loose quant string into the canonical descriptor.

        Handles: ``"4b/AWQ/g128"``, GGUF ``"Q4_K_M"`` / ``"Q3_K_XL"``,
        ``"fp8"`` / ``"fp8_e4m3"``, ``"bf16"`` / ``"fp16"``,
        ``"compressed-tensors g32"``, ``"AWQ g128"``, ``"GPTQ-Int4"``.
        """
        s = str(text).strip()
        low = s.lower()
        # Explicit canonical "Nb/SCHEME/gG".
        m = re.match(r"^\s*(\d+(?:\.\d+)?)b/([^/]+)(?:/g(\d+))?\s*$", s, re.I)
        if m:
            return cls(
                float(m.group(1)),
                m.group(2).upper(),
                int(m.group(3)) if m.group(3) else None,
            )
        # GGUF k-quant / legacy: Q4_K_M, Q3_K_XL, Q8_0, IQ2_XS ...
        m = re.match(r"^\s*(iq|q)(\d+)(?:_(.+?))?\s*$", low)
        if m:
            suffix = m.group(3)
            scheme = "IQ" if m.group(1) == "iq" else "Q"
            if suffix:
                scheme += "_" + suffix.upper()
            return cls(float(m.group(2)), scheme, None)
        # fp8 / fp16 / bf16.
        m = re.match(r"^\s*(fp8|fp16|bf16)", low)
        if m:
            bits = {"fp8": 8.0, "fp16": 16.0, "bf16": 16.0}[m.group(1)]
            return cls(bits, m.group(1), None)
        # scheme + optional group size (AWQ g128, GPTQ-Int4, compressed g32).
        gs = None
        gm = re.search(r"g(\d+)", low)
        if gm:
            gs = int(gm.group(1))
        bits = 16.0
        bm = re.search(r"int(\d+)|(\d+)\s*bit|(\d+)b\b", low)
        if bm:
            bits = float(next(g for g in bm.groups() if g))
        elif "awq" in low or "gptq" in low:
            bits = 4.0  # AWQ/GPTQ default to 4-bit
        scheme = re.split(r"[\s\-/]", s)[0].upper() or "unknown"
        return cls(bits, scheme, gs)

    def canonical(self) -> str:
        base = f"{self.nominal_bits:g}b/{self.scheme}"
        return base + (f"/g{self.group_size}" if self.group_size else "")

    def exact_key(self) -> Tuple:
        return (self.nominal_bits, self.scheme.upper(), self.group_size)

    def similar_key(self) -> float:
        """The 'similar quant' grouping key (nominal bits only) — approximate,
        must be labelled as such wherever used."""
        return self.nominal_bits


# ---------------------------------------------------------------------------
# Band (never a scalar — design §5A.4/§5A.5).
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Band:
    """A (lo, hi) range with an optional central value. Saved numbers that
    vary with the batch mix are bands, not false-precision scalars."""

    lo: float
    hi: float
    central: Optional[float] = None

    def __post_init__(self):
        if self.hi < self.lo:
            raise IngestRejected(f"band hi<lo: ({self.lo}, {self.hi})")

    def as_list(self):
        return [self.lo, self.hi] if self.central is None else [
            self.lo,
            self.central,
            self.hi,
        ]

    @classmethod
    def coerce(cls, value) -> "Band":
        if isinstance(value, Band):
            return value
        if isinstance(value, (list, tuple)):
            if len(value) == 2:
                return cls(float(value[0]), float(value[1]))
            if len(value) == 3:
                return cls(float(value[0]), float(value[2]), float(value[1]))
        raise IngestRejected(
            f"a saved-quantity must be a [lo, hi] band, not the scalar "
            f"{value!r} (false precision is treated as dishonesty here)."
        )


# ---------------------------------------------------------------------------
# One results entry (design §5B.1 schema).
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ResultEntry:
    # -- identity --------------------------------------------------------
    model: str
    quant: QuantDescriptor
    #: anonymous hardware class: [(count, name, total_mib), ...] (S2 §4.5).
    hardware_cards: List[tuple]
    #: the exact launch flags + env — the "reproduce this" block (§5B.3).
    reproduce_flags: List[str]
    provenance: str  # measured | boot-log-measured | planner-estimate

    # -- feasibility / capacity (available NOW: planner or boot log) -----
    fits: Optional[bool] = None
    max_context_tokens: Optional[float] = None
    max_total_num_tokens: Optional[int] = None
    tp_config: Optional[str] = None  # e.g. "tp3 uneven 5,3,3" for Mode-B key

    # -- MEASURED perf (from S2.5; absent until then) --------------------
    batch: Optional[int] = None
    concurrency: Optional[int] = None
    #: per-batch-size-bucket throughput curves {bucket: tok/s}; NOT a scalar.
    prefill_tok_s_by_bucket: Optional[Dict[int, float]] = None
    decode_tok_s_by_bucket: Optional[Dict[int, float]] = None
    peak_prefill_tok_s: Optional[float] = None
    peak_decode_tok_s: Optional[float] = None

    # -- MEASURED energy (from S2.5; absent until then) ------------------
    j_per_prefill_token_by_bucket: Optional[Dict[int, float]] = None
    j_per_decode_token_by_bucket: Optional[Dict[int, float]] = None
    batch_size_distribution: Optional[Dict[int, float]] = None
    attribution_coverage: Optional[float] = None
    per_card_efficiency: Optional[Dict[str, float]] = None
    #: bands, never scalars.
    kwh_saved: Optional[Band] = None
    prefill_seconds_saved: Optional[Band] = None

    kv_cache_dtype: str = "auto"
    timestamp: Optional[float] = None

    # -- helpers ---------------------------------------------------------

    def has_measured_perf(self) -> bool:
        return any(
            v is not None
            for v in (
                self.prefill_tok_s_by_bucket,
                self.decode_tok_s_by_bucket,
                self.peak_prefill_tok_s,
                self.peak_decode_tok_s,
                self.j_per_prefill_token_by_bucket,
                self.j_per_decode_token_by_bucket,
                self.kwh_saved,
                self.prefill_seconds_saved,
            )
        )

    def hardware_class(self) -> str:
        return ", ".join(f"{c[0]}x {c[1]}" for c in self.hardware_cards)

    def mode_b_key(self) -> Tuple:
        """Strict reproducibility key (design §5B.3 Mode B)."""
        return (
            self.model,
            self.quant.exact_key(),
            self.hardware_class(),
            self.tp_config,
            self.batch,
            self.concurrency,
            self.kv_cache_dtype,
        )

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["quant"] = {
            "nominal_bits": self.quant.nominal_bits,
            "scheme": self.quant.scheme,
            "group_size": self.quant.group_size,
            "canonical": self.quant.canonical(),
        }
        for k in ("kwh_saved", "prefill_seconds_saved"):
            b = getattr(self, k)
            d[k] = b.as_list() if b is not None else None
        return d

    @classmethod
    def from_json(cls, d: dict) -> "ResultEntry":
        d = dict(d)
        q = d.pop("quant")
        quant = QuantDescriptor(
            q["nominal_bits"], q["scheme"], q.get("group_size")
        )
        for k in ("kwh_saved", "prefill_seconds_saved"):
            if d.get(k) is not None:
                d[k] = Band.coerce(d[k])
        d["hardware_cards"] = [tuple(c) for c in d.get("hardware_cards", [])]
        d.pop("canonical", None)
        # Only known fields (forward-compatible ingest).
        known = {f.name for f in dataclasses.fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        return cls(quant=quant, **d)


# ---------------------------------------------------------------------------
# The store + ingest guard.
# ---------------------------------------------------------------------------


class ResultsStore:
    """Append-only ``results.jsonl`` of MEASURED runs (design §5B.1)."""

    def __init__(self, entries: Optional[Sequence[ResultEntry]] = None):
        self._entries: List[ResultEntry] = list(entries or [])

    def __len__(self):
        return len(self._entries)

    def entries(self) -> List[ResultEntry]:
        return list(self._entries)

    def check(self, entry: ResultEntry) -> None:
        """The ingest guard (design §5B.3a). Raises ``IngestRejected`` when the
        entry is not admissible; returns None when it is."""
        if entry.has_measured_perf():
            if entry.provenance not in MEASURED_PROVENANCE:
                raise IngestRejected(
                    f"entry carries perf/energy numbers but provenance is "
                    f"{entry.provenance!r}, not measured "
                    f"{MEASURED_PROVENANCE}. The DB stores only measured runs; "
                    "an unmeasured perf number is rejected, never shown with a "
                    "caveat (design §5B.3a)."
                )
            # Bands must be bands (already enforced by Band.__post_init__, but
            # a raw scalar slipping through the API is caught here).
            for name in ("kwh_saved", "prefill_seconds_saved"):
                v = getattr(entry, name)
                if v is not None and not isinstance(v, Band):
                    raise IngestRejected(
                        f"{name} must be a (lo, hi) Band, not {v!r}."
                    )
            if entry.attribution_coverage is not None and not (
                0.0 <= entry.attribution_coverage <= 1.0
            ):
                raise IngestRejected(
                    f"attribution_coverage must be in [0,1], got "
                    f"{entry.attribution_coverage}."
                )
        else:
            # A no-perf entry is a feasibility record; the measured store is
            # not its home (it is recomputed live by the landscape).
            if entry.provenance in MEASURED_PROVENANCE:
                raise IngestRejected(
                    "a measured-provenance entry must carry at least one "
                    "measured perf/energy field; got none."
                )
            raise IngestRejected(
                f"provenance {entry.provenance!r} carries no measured perf — "
                "feasibility/capacity records are not stored here (the "
                "landscape computes them live from the planner)."
            )

    def ingest(self, entry: ResultEntry) -> None:
        self.check(entry)
        if entry.timestamp is None:
            entry.timestamp = time.time()
        self._entries.append(entry)

    def try_ingest(self, entry: ResultEntry) -> Optional[str]:
        """Ingest, returning None on success or the rejection reason string."""
        try:
            self.ingest(entry)
            return None
        except IngestRejected as e:
            return str(e)

    # -- persistence (jsonl) -----------------------------------------------

    def append_to_file(self, path: str, entry: ResultEntry) -> None:
        self.ingest(entry)
        with open(path, "a") as f:
            f.write(json.dumps(entry.to_json()) + "\n")

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            for e in self._entries:
                f.write(json.dumps(e.to_json()) + "\n")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str, strict: bool = False) -> "ResultsStore":
        """Load a jsonl store, re-running the ingest guard on every line so a
        hand-edited file cannot smuggle an unmeasured perf number in. With
        ``strict`` a bad line raises; otherwise it is skipped."""
        store = cls()
        if not os.path.exists(path):
            return store
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                entry = ResultEntry.from_json(json.loads(line))
                try:
                    store.ingest(entry)
                except IngestRejected:
                    if strict:
                        raise
        return store
