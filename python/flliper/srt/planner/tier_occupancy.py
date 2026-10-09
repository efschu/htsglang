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
"""The spill / offload tier view for the landing page: one row per tier,
tier = TYPE x PLACE, each row honest about where its number came from.

This is the #407 memory-tier picture drawn from the sources that actually
exist today. The structure follows DESIGN_407 (a tier is a kind -- device
VRAM / host RAM / filesystem -- crossed with a location, local or remote), but
the DATA does not come from the #407 ``TierRegistry``: that registry is a
capability/profile description with no production consumer wired to it, and
``TierCapacity.reserved`` is never populated from a live source. Reading it
would produce zeros that look like measurements. The rows below therefore read
each consumer's own bookkeeping, through the ``sglang:spill_tier_*_bytes``
gauges the scheduler publishes (``observability/spill_tiers.py``), and every
tier with no live source is emitted as an explicit ABSENT row carrying the
reason.

Provenance follows #218 exactly -- ``measured`` / ``estimate`` / ``absent``,
with no "probably" tier (``planner/cost_model.Provenance``). In this module:

* ``measured`` -- the tier's own ledger answered, in the unit it speaks.
* ``absent``   -- no live source. The row is still DRAWN, with
  ``missing_reason``. A tier that is not configured on this rig (every remote
  tier, most of the time) is a visible absence, never a hidden row and never a
  zero.

There is no ``estimate`` row in the catalogue today: every source either has a
current-occupancy ledger or has none. If one is ever added, it is because a
figure is genuinely derived, and it says so in ``source``.

Host RAM totals: read from ``/proc/meminfo`` of the process running the
DASHBOARD, which is not necessarily the process running the server. That is
stated on the row (``total_scope``) rather than quietly used as if it were the
server's -- the runbook's rule is that a number the page shows must be
checkable from a shell, and this one is checkable only against the right host.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Dict, List, Optional, Tuple

__all__ = [
    "SPILL_USED_METRIC",
    "SPILL_TOTAL_METRIC",
    "MEASURED",
    "ABSENT",
    "TierRow",
    "spill_tier_bytes",
    "host_mem_total_bytes",
    "tier_rows",
    "tier_view",
]

SPILL_USED_METRIC = "sglang:spill_tier_used_bytes"
SPILL_TOTAL_METRIC = "sglang:spill_tier_total_bytes"

MEASURED = "measured"
ABSENT = "absent"

_TIER_LABEL_RE = re.compile(r'spill_tier="([^"]*)"')


@dataclasses.dataclass(frozen=True)
class TierRow:
    """One tier: what it is, where it is, how full, and how we know.

    ``used`` / ``total`` are None on an absent row -- a missing number is
    None, never 0, so a renderer cannot draw an empty bar for a tier that has
    no data.
    """

    id: str
    label: str
    kind: str           # "vram" | "host_ram" | "disk"
    location: str       # "local" | "remote"
    provenance: str     # "measured" | "absent"
    used: Optional[float] = None
    total: Optional[float] = None
    unit: str = "bytes"
    source: str = ""
    consumer: str = ""
    missing_reason: str = ""
    total_scope: str = ""

    @property
    def used_frac(self) -> Optional[float]:
        if self.used is None or not self.total:
            return None
        return min(1.0, self.used / self.total)

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["used_frac"] = self.used_frac
        return d


# ---------------------------------------------------------------------------
# Scrape parsing.
# ---------------------------------------------------------------------------
def spill_tier_bytes(metrics_text: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    """``(used_by_tier, total_by_tier)`` from a /metrics scrape.

    ``energy.parse_prometheus_metrics`` collapses every label set of a metric
    into one number, which would fuse all tiers into a single figure -- the
    opposite of a per-tier view -- so the ``spill_tier`` label is parsed here,
    the way ``hicache_savings.cached_tokens_by_source`` parses ``cache_source``.
    """
    used: Dict[str, float] = {}
    total: Dict[str, float] = {}
    for line in (metrics_text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(SPILL_USED_METRIC):
            target = used
        elif line.startswith(SPILL_TOTAL_METRIC):
            target = total
        else:
            continue
        brace = line.find("{")
        if brace < 0:
            continue      # a bare total carries no tier identity: not usable
        close = line.find("}", brace)
        m = _TIER_LABEL_RE.search(line[brace + 1:close])
        if not m:
            continue
        rest = line[close + 1:].strip()
        try:
            target[m.group(1)] = target.get(m.group(1), 0.0) + float(rest.split()[0])
        except (ValueError, IndexError):
            continue
    return used, total


def host_mem_total_bytes(meminfo_text: Optional[str] = None) -> Optional[int]:
    """Total host RAM in bytes, from /proc/meminfo. Reference denominator only.

    This is the only place this module consults /proc: every OCCUPANCY figure
    comes from a consumer's ledger, because /proc cannot attribute a byte to
    the tier that holds it.
    """
    text = meminfo_text
    if text is None:
        try:
            with open("/proc/meminfo", "r") as f:
                text = f.read()
        except OSError:
            return None
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1]) * 1024      # kB -> bytes
                except ValueError:
                    return None
    return None


# ---------------------------------------------------------------------------
# The tier catalogue.
# ---------------------------------------------------------------------------
#: Tiers with no live occupancy source in this tree. Each is DRAWN, absent,
#: with the reason -- so "nobody looked" and "looked, there is nothing to
#: read" stay distinguishable (the #17 doctrine applied to a dashboard row).
_ABSENT_TIERS = (
    ("vram_short_term_register", "VRAM spill posts (#286 short-term register)",
     "vram", "local",
     "the #286 register tracks classes, not bytes, in this build: no "
     "production path constructs the movement backend that owns a byte "
     "ledger, so there is no current-occupancy figure to read."),
    ("vram_peer", "peer-VRAM spill (cross-card park)", "vram", "local",
     "no consumer parks bytes in another card's VRAM today; the #224 park "
     "tiers are host-RAM and filesystem backends."),
    ("disk_nvme_experts", "NVMe expert tier (#389)", "disk", "local",
     "design only: no NVMe expert tier is implemented in this tree."),
    ("disk_hibernate", "hibernate staging images (#89/#456)", "disk", "local",
     "written once at boot or on POST /hibernate and static in between, so "
     "it is a state, not a live tier gauge; size on disk is a separate read."),
    ("remote_rig_vram", "rig-2 VRAM", "vram", "remote",
     "no remote-VRAM spill target is configured or implemented."),
    ("remote_rig_disk", "rig-2 disk", "disk", "remote",
     "no remote-disk spill target is configured."),
)

#: How a #224 park backend name maps onto (kind, location, label). ``file`` is
#: a local persistent store; ``mooncake`` is the paired rig's host RAM.
_PARK_BACKENDS = {
    "file": ("disk", "local", "KV session park - local file store (#224)"),
    "mooncake": ("host_ram", "remote", "KV session park - remote rig RAM (#224)"),
    "dynamic": ("host_ram", "remote", "KV session park - operator tier (#224)"),
}


def tier_rows(
    metrics_text: str,
    *,
    hicache: Optional[dict] = None,
    host_total: Optional[int] = None,
    metrics_available: bool = True,
) -> List[TierRow]:
    """Every tier, in a fixed order, measured or explicitly absent.

    The order is by locality then by kind (local VRAM, local RAM, local disk,
    remote), because that is the order a reader walks when asking "where did
    my bytes go".
    """
    used, total = spill_tier_bytes(metrics_text)
    absent_by_id = {t[0]: t for t in _ABSENT_TIERS}
    rows: List[TierRow] = []

    def _absent(tier_id: str, reason: Optional[str] = None) -> TierRow:
        _id, label, kind, loc, why = absent_by_id[tier_id]
        return TierRow(id=_id, label=label, kind=kind, location=loc,
                       provenance=ABSENT, missing_reason=reason or why)

    no_scrape = ("the server serves no /metrics, so no tier can report its "
                 "occupancy; this is unknown, not empty.")

    # --- local VRAM ------------------------------------------------------
    rows.append(_absent("vram_short_term_register"))
    rows.append(_absent("vram_peer"))

    # --- local host RAM --------------------------------------------------
    rows.append(_bytes_row(
        "expert_host_ram", "MoE expert pool (#77/#123)", "host_ram", "local",
        used.get("expert_host_ram"), total.get("expert_host_ram"),
        host_total,
        source="sum of the per-layer pinned host tensors",
        consumer="expert offload",
        missing=(no_scrape if not metrics_available else
                 "expert offload is not active on this server (no layer has "
                 "an offload cache installed)."),
    ))
    rows.append(_bytes_row(
        "kv_session_host_ram", "KV session spill (kvso)", "host_ram", "local",
        used.get("kv_session_host_ram"), total.get("kv_session_host_ram"),
        host_total,
        source="occupied spill regions x region bytes",
        consumer="kv session offload",
        missing=(no_scrape if not metrics_available else
                 "--enable-kv-session-offload is off on this server."),
    ))
    rows.append(_hicache_row(hicache, metrics_available, no_scrape))

    # --- local disk ------------------------------------------------------
    rows.append(_bytes_row(
        "hicache_file_disk", "HiCache file store (L3)", "disk", "local",
        used.get("hicache_file_disk"), total.get("hicache_file_disk"), None,
        source="HiCache file backend LRU evictor byte total",
        consumer="hicache L3",
        missing=(no_scrape if not metrics_available else
                 "no HiCache file backend with eviction tracking: without a "
                 "configured max size / min free space the backend keeps no "
                 "byte total, and a 0 there would be wrong, not empty."),
    ))
    rows.append(_absent("disk_nvme_experts"))
    rows.append(_absent("disk_hibernate"))

    # --- park destinations (#224): local file and remote rig -------------
    park_ids = sorted(k for k in used if k.startswith("park:"))
    seen_backends = set()
    for key in park_ids:
        backend = key.split(":", 1)[1]
        seen_backends.add(backend)
        kind, loc, label = _PARK_BACKENDS.get(
            backend, ("host_ram", "remote", f"KV session park - {backend} (#224)"))
        rows.append(TierRow(
            id=key, label=label, kind=kind, location=loc,
            provenance=MEASURED, used=used[key], total=total.get(key),
            source="parked session rows x bytes per token",
            consumer="kv session park",
            total_scope=("remote host" if loc == "remote" else "local disk"),
        ))
    if "mooncake" not in seen_backends and "dynamic" not in seen_backends:
        rows.append(TierRow(
            id="remote_rig_ram", label="rig-2 host RAM (#224 park target)",
            kind="host_ram", location="remote", provenance=ABSENT,
            missing_reason=(
                no_scrape if not metrics_available else
                "no remote park destination configured "
                "(--kv-session-offload-destinations names none)."),
        ))
    rows.append(_absent("remote_rig_vram"))
    rows.append(_absent("remote_rig_disk"))
    return rows


def _bytes_row(tier_id, label, kind, location, used, total, host_total,
               *, source, consumer, missing) -> TierRow:
    if used is None:
        return TierRow(id=tier_id, label=label, kind=kind, location=location,
                       provenance=ABSENT, missing_reason=missing)
    scope = ""
    if total is None and kind == "host_ram" and host_total:
        total, scope = host_total, "dashboard host /proc/meminfo MemTotal"
    return TierRow(id=tier_id, label=label, kind=kind, location=location,
                   provenance=MEASURED, used=used, total=total,
                   source=source, consumer=consumer, total_scope=scope)


def _hicache_row(hicache, metrics_available, no_scrape) -> TierRow:
    """HiCache host tier (L2). Its unit is TOKENS, not bytes: those are the
    only host-tier gauges metrics_collector exports, and converting them to
    bytes here would need a page size this module does not have. The row says
    tokens rather than inventing a byte figure."""
    if not hicache:
        return TierRow(
            id="hicache_host_ram", label="HiCache host tier (L2)",
            kind="host_ram", location="local", provenance=ABSENT,
            missing_reason=(no_scrape if not metrics_available else
                            "the server was not booted with a hierarchical "
                            "cache, so no host KV tier exists."))
    return TierRow(
        id="hicache_host_ram", label="HiCache host tier (L2)",
        kind="host_ram", location="local", provenance=MEASURED,
        used=hicache.get("host_used_tokens"),
        total=hicache.get("host_total_tokens"),
        unit="tokens", source="sglang:hicache_host_*_tokens",
        consumer="hicache L2",
        total_scope="host KV pool capacity, in KV tokens (no byte gauge exists)")


def tier_view(
    metrics_text: str,
    *,
    hicache: Optional[dict] = None,
    meminfo_text: Optional[str] = None,
    metrics_available: bool = True,
) -> dict:
    """The whole panel payload: rows plus the host-RAM reference sum.

    ``host_ram_used_bytes`` sums ONLY the local host-RAM rows that reported
    bytes -- the token-unit HiCache row is excluded by construction, because
    adding tokens to bytes is how a plausible wrong total gets made.
    """
    host_total = host_mem_total_bytes(meminfo_text)
    rows = tier_rows(metrics_text, hicache=hicache, host_total=host_total,
                     metrics_available=metrics_available)
    summed = 0.0
    counted, skipped = [], []
    for r in rows:
        if (r.kind == "host_ram" and r.location == "local"
                and r.provenance == MEASURED and r.unit == "bytes"
                and r.used is not None):
            summed += r.used
            counted.append(r.id)
        elif r.provenance == MEASURED and r.unit != "bytes":
            skipped.append(r.id)
    return {
        "rows": [r.to_json() for r in rows],
        "host_ram_total_bytes": host_total,
        "host_ram_total_scope": "dashboard host /proc/meminfo MemTotal",
        "host_ram_used_bytes": summed if counted else None,
        "host_ram_counted": counted,
        "host_ram_excluded_non_byte": skipped,
        "measured_tiers": [r.id for r in rows if r.provenance == MEASURED],
        "absent_tiers": [r.id for r in rows if r.provenance == ABSENT],
    }
