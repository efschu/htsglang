# SPDX-License-Identifier: Apache-2.0
"""Rate-table loaders for the barlink path dispatcher (#279).

Three sources fill the dispatcher's ``PathProfile`` entries; each has its
own parser here, and each parser consumes ONLY effective/measured values:

1. ``scripts/p2p_readiness/`` probe output (fresh merge):
   * ``capability_matrix.json`` -- per directed pair the EFFECTIVE aperture
     (``effective_max_single_copy_bytes`` / ``effective_max_region_chunked_
     bytes``). Nominal BAR1 figures are deliberately ignored: they are an
     upper bound, not a usability promise.
   * ``d2d_bench.json`` -- per directed pair and mode (direct | staged) the
     median-latency ladder -> measured profiles ``d2d_direct:<src>-><dst>``
     and ``host_staged:<src>-><dst>``.
2. #278 GDR matrix TSV rows (crossrig ladder, wire rows):
   ``pair  direction  modus  ro  depth  size_bytes  iters  p10_us
   median_us  p90_us  MB_per_s`` -> measured profiles
   ``gdr_direct@d<depth>:...`` / ``nic_staged@d<depth>:...`` (``+ro`` suffix
   when relaxed ordering was on). median_us is the half round-trip of one
   round carrying ``depth`` messages, so per-message time is median/depth.
3. The pending NCCL/system-RAM reference. The measurement does not exist
   yet; its FORMAT is defined here (``NCCL_REFERENCE_*``,
   ``new_nccl_reference_envelope``) so the run can write directly loadable
   JSON. Both p50 and p99 per row and an explicit ``load`` arm are part of
   the schema -- the #278 wrap-up flagged that the load axis was measured
   asymmetrically (p50 vs p99) and must be re-taken symmetrically.

Unknown or missing sources are LOUDLY logged and yield placeholder
profiles, which the dispatcher's placeholder-neutrality rule turns into
"status quo, decide nothing" (barlink_path_dispatcher hard rule 1).

Pure parsing, CPU-hermetic. Tests:
test/registered/unit/distributed/test_barlink_path_dispatcher.py.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from sglang.srt.distributed.device_communicators.barlink_path_dispatcher import (
    PROVENANCE_MEASURED,
    PROVENANCE_PLACEHOLDER,
    PathProfile,
    RatePoint,
)

logger = logging.getLogger(__name__)

GIB = 1024 * 1024 * 1024

# Path-name kinds the loaders emit (directed pairs: "<kind>:<src>-><dst>").
KIND_D2D_DIRECT = "d2d_direct"
KIND_HOST_STAGED = "host_staged"
KIND_GDR_DIRECT = "gdr_direct"
KIND_NIC_STAGED = "nic_staged"
KIND_NCCL = "nccl"


@dataclass
class LoadResult:
    """One loader's outcome: measured profiles, per-pair apertures, and the
    rows it could not use (errors never abort a load -- a partially usable
    artifact still beats a placeholder)."""

    profiles: List[PathProfile] = field(default_factory=list)
    # (src, dst) -> effective single-copy aperture bytes (capability matrix).
    apertures: Dict[Tuple[str, str], int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)


def pair_path(kind: str, src: str, dst: str) -> str:
    return f"{kind}:{src}->{dst}"


def placeholder_profile(name: str, source: str = "missing source") -> PathProfile:
    return PathProfile(
        name=name, provenance=PROVENANCE_PLACEHOLDER, source=source
    )


# ---------------------------------------------------------------------------
# source 1a: p2p_readiness capability matrix (EFFECTIVE apertures only)
# ---------------------------------------------------------------------------

P2P_CAPABILITY_KIND = "capability_matrix"
P2P_D2D_KIND = "d2d_bench"

# capability_matrix.py writes its ordered rows under "directed_pairs". The
# legacy name is still read so an older artifact stays loadable, but a payload
# carrying NEITHER key is an error rather than an empty result: a silently
# empty aperture table leaves every direct path unbounded and is
# indistinguishable from "this rig has no P2P".
P2P_CAPABILITY_ROW_KEYS = ("directed_pairs", "pairs")


def _check_envelope(payload: dict, expected_kind: str, res: LoadResult) -> bool:
    if not isinstance(payload, dict):
        res.errors.append(f"{expected_kind}: payload is not a JSON object")
        return False
    kind = payload.get("kind")
    if kind != expected_kind:
        res.errors.append(
            f"{expected_kind}: unexpected kind {kind!r} "
            f"(schema_version {payload.get('schema_version')!r})"
        )
        return False
    if "schema_version" not in payload:
        res.errors.append(f"{expected_kind}: missing schema_version")
        return False
    return True


def capability_matrix_rows(payload: dict) -> Tuple[str, List[dict]]:
    """(key, rows) for a capability matrix, or ("", []) when it carries none.

    One place knows what the producer calls its rows, so a consumer cannot
    quietly read a key that was never written.
    """
    for key in P2P_CAPABILITY_ROW_KEYS:
        rows = payload.get(key)
        if isinstance(rows, list):
            return key, rows
    return "", []


def load_p2p_capability_matrix(payload: dict) -> LoadResult:
    """Effective apertures per directed pair. NOMINAL fields
    (``dst_bar1_nominal_bytes`` etc.) are ignored by design."""
    res = LoadResult()
    if not _check_envelope(payload, P2P_CAPABILITY_KIND, res):
        return res
    key, rows = capability_matrix_rows(payload)
    if not key:
        res.errors.append(
            f"{P2P_CAPABILITY_KIND}: no rows under any of "
            f"{P2P_CAPABILITY_ROW_KEYS} (keys present: {sorted(payload)})"
        )
        return res
    for i, row in enumerate(rows):
        src = row.get("src_pci")
        dst = row.get("dst_pci")
        if not src or not dst:
            res.errors.append(f"{key}[{i}]: missing src_pci/dst_pci")
            continue
        eff = row.get("effective_max_single_copy_bytes")
        if eff is None:
            res.skipped.append(
                f"{key}[{i}] {src}->{dst}: no effective aperture measured "
                "(nominal-only rows are not consumed)"
            )
            continue
        res.apertures[(src, dst)] = int(eff)
    return res


def apply_apertures(
    profiles: List[PathProfile], apertures: Dict[Tuple[str, str], int]
) -> None:
    """Stamp the measured effective aperture onto every DIRECT profile of a
    measured pair (host-staged/NIC-staged paths do not map peer memory, so
    the BAR window is not their constraint)."""
    for prof in profiles:
        kind, _, pair = prof.name.partition(":")
        # gdr_direct kinds may carry "+ro" and "@d<depth>" suffixes.
        base_kind = kind.split("@", 1)[0].split("+", 1)[0]
        if base_kind not in (KIND_D2D_DIRECT, KIND_GDR_DIRECT):
            continue
        src, sep, dst = pair.partition("->")
        if sep and (src, dst) in apertures:
            prof.aperture_bytes = apertures[(src, dst)]


# ---------------------------------------------------------------------------
# source 1b: p2p_readiness d2d bench (median-latency ladder)
# ---------------------------------------------------------------------------


def load_p2p_d2d_bench(payload: dict) -> LoadResult:
    res = LoadResult()
    if not _check_envelope(payload, P2P_D2D_KIND, res):
        return res
    for i, row in enumerate(payload.get("pairs", [])):
        src = row.get("src_pci")
        dst = row.get("dst_pci")
        mode = row.get("mode")
        if not src or not dst or mode not in ("direct", "staged"):
            res.errors.append(f"pairs[{i}]: missing src_pci/dst_pci/mode")
            continue
        kind = KIND_D2D_DIRECT if mode == "direct" else KIND_HOST_STAGED
        points: List[RatePoint] = []
        capacity = None
        for j, pt in enumerate(row.get("points", [])):
            if "error" in pt:
                res.skipped.append(
                    f"pairs[{i}].points[{j}]: {pt['error']!r} (above the "
                    "effective aperture is a result, not a rate)"
                )
                continue
            try:
                points.append(
                    RatePoint(int(pt["size_bytes"]), float(pt["median_s"]) * 1e3)
                )
            except (KeyError, TypeError, ValueError) as e:
                res.errors.append(f"pairs[{i}].points[{j}]: {e!r}")
                continue
            if "gib_per_s" in pt:
                rate = float(pt["gib_per_s"]) * GIB
                capacity = rate if capacity is None else max(capacity, rate)
        if not points:
            res.skipped.append(f"pairs[{i}] {src}->{dst} {mode}: no usable points")
            continue
        prof = PathProfile(
            name=pair_path(kind, src, dst),
            provenance=PROVENANCE_MEASURED,
            points=points,
            capacity_bytes_per_s=capacity,
            source="p2p_readiness/d2d_bench",
        )
        prof.fit()
        res.profiles.append(prof)
    return res


# ---------------------------------------------------------------------------
# source 2: #278 GDR matrix TSV rows
# ---------------------------------------------------------------------------

GDR_TSV_COLUMNS = (
    "pair",
    "direction",
    "modus",
    "ro",
    "depth",
    "size_bytes",
    "iters",
    "p10_us",
    "median_us",
    "p90_us",
    "MB_per_s",
)


def _gdr_pair(label: str) -> Optional[Tuple[str, str]]:
    """``i_05:00.0_to_0a:00.0`` -> ("05:00.0", "0a:00.0"). The leading
    ``<letter>_`` arm prefix is optional."""
    body = label.split("_", 1)[1] if "_" in label and len(label.split("_", 1)[0]) <= 2 else label
    src, sep, dst = body.partition("_to_")
    if not sep or not src or not dst:
        return None
    return src, dst


def load_gdr_matrix_tsv(lines) -> LoadResult:
    """Parse #278 crossrig-ladder TSV rows (see module docstring for the
    column layout). Comment lines (#) are headers/verdicts, not data."""
    res = LoadResult()
    grouped: Dict[str, List[RatePoint]] = {}
    capacity: Dict[str, float] = {}
    for lineno, raw in enumerate(lines, start=1):
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) != len(GDR_TSV_COLUMNS):
            res.errors.append(
                f"line {lineno}: {len(cols)} columns, expected "
                f"{len(GDR_TSV_COLUMNS)}"
            )
            continue
        row = dict(zip(GDR_TSV_COLUMNS, cols))
        pair = _gdr_pair(row["pair"])
        if pair is None:
            res.errors.append(f"line {lineno}: unparsable pair {row['pair']!r}")
            continue
        if row["modus"] not in ("gdr", "stage"):
            res.errors.append(f"line {lineno}: unknown modus {row['modus']!r}")
            continue
        try:
            depth = int(row["depth"])
            size_bytes = int(row["size_bytes"])
            median_us = float(row["median_us"])
            mb_per_s = float(row["MB_per_s"]) if row["MB_per_s"] != "-" else None
        except ValueError as e:
            res.errors.append(f"line {lineno}: {e}")
            continue
        kind = KIND_GDR_DIRECT if row["modus"] == "gdr" else KIND_NIC_STAGED
        if row["ro"] == "on":
            kind += "+ro"
        kind += f"@d{depth}"
        name = pair_path(kind, *pair)
        # median_us is one round of `depth` messages: per-message time.
        grouped.setdefault(name, []).append(
            RatePoint(size_bytes, (median_us / max(depth, 1)) / 1e3)
        )
        if mb_per_s is not None:
            rate = mb_per_s * 1e6
            capacity[name] = max(capacity.get(name, 0.0), rate)
    for name, points in grouped.items():
        prof = PathProfile(
            name=name,
            provenance=PROVENANCE_MEASURED,
            points=points,
            capacity_bytes_per_s=capacity.get(name),
            source="#278 gdr matrix tsv",
        )
        prof.fit()
        res.profiles.append(prof)
    return res


# ---------------------------------------------------------------------------
# source 3: the pending NCCL/system-RAM reference -- FORMAT DEFINITION
# ---------------------------------------------------------------------------

NCCL_REFERENCE_KIND = "nccl_reference"
NCCL_REFERENCE_SCHEMA_VERSION = 1

# Per-row required fields. The measurement writes one row per
# (op, pair, size, load arm):
#   op          "all_reduce" | "send_recv" | ...
#   transport   NCCL's chosen transport from NCCL_DEBUG=INFO ("P2P"|"SHM"|"NET")
#   world       process-group size of the measurement
#   src_pci     PCI bus id (device-order trap: never a bare index)
#   dst_pci     PCI bus id; for symmetric collectives the sorted pair
#   size_bytes  payload
#   iters       samples behind the percentiles
#   p50_us      median half round-trip, microseconds
#   p99_us      tail -- MANDATORY: the #278 load axis was p50-vs-p99
#               asymmetric and must be re-taken with both sides' p99
#   load        "idle" | a named foreign-load arm (e.g. "nic_1mib_stream");
#               idle rows feed the cost model, load rows the pressure view
NCCL_REFERENCE_ROW_FIELDS = frozenset(
    {
        "op",
        "transport",
        "world",
        "src_pci",
        "dst_pci",
        "size_bytes",
        "iters",
        "p50_us",
        "p99_us",
        "load",
    }
)


def new_nccl_reference_envelope() -> dict:
    """What the measurement script should write: this envelope plus
    ``rows: [...]`` with the fields above."""
    return {
        "schema_version": NCCL_REFERENCE_SCHEMA_VERSION,
        "kind": NCCL_REFERENCE_KIND,
        "rows": [],
    }


def load_nccl_reference(payload: dict) -> LoadResult:
    """Load an nccl_reference JSON. Idle rows become the measured cost
    profiles ``nccl:<op>:<src>-><dst>``; rows under a load arm are kept as
    points of ``...@load=<arm>`` profiles (p99-based) for the pressure
    comparison, separate from the idle cost model."""
    res = LoadResult()
    if not _check_envelope(payload, NCCL_REFERENCE_KIND, res):
        return res
    if payload.get("schema_version") != NCCL_REFERENCE_SCHEMA_VERSION:
        res.errors.append(
            f"nccl_reference: schema_version "
            f"{payload.get('schema_version')!r}, expected "
            f"{NCCL_REFERENCE_SCHEMA_VERSION}"
        )
        return res
    grouped: Dict[str, List[RatePoint]] = {}
    for i, row in enumerate(payload.get("rows", [])):
        missing = NCCL_REFERENCE_ROW_FIELDS - set(row)
        if missing:
            res.errors.append(f"rows[{i}]: missing fields {sorted(missing)}")
            continue
        try:
            size_bytes = int(row["size_bytes"])
            p50_ms = float(row["p50_us"]) / 1e3
            p99_ms = float(row["p99_us"]) / 1e3
        except (TypeError, ValueError) as e:
            res.errors.append(f"rows[{i}]: {e!r}")
            continue
        kind = f"{KIND_NCCL}:{row['op']}"
        name = pair_path(kind, row["src_pci"], row["dst_pci"])
        if row["load"] == "idle":
            grouped.setdefault(name, []).append(RatePoint(size_bytes, p50_ms))
        else:
            grouped.setdefault(f"{name}@load={row['load']}", []).append(
                RatePoint(size_bytes, p99_ms)
            )
    for name, points in grouped.items():
        prof = PathProfile(
            name=name,
            provenance=PROVENANCE_MEASURED,
            points=points,
            source="nccl_reference",
        )
        prof.fit()
        res.profiles.append(prof)
    return res


# ---------------------------------------------------------------------------
# top-level: load whatever exists, placeholders for the rest -- loudly
# ---------------------------------------------------------------------------


def _read_json(path: str, res: LoadResult) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        res.errors.append(f"{path}: {e!r}")
        return None


def load_rate_tables(
    p2p_capability_json: Optional[str] = None,
    p2p_d2d_json: Optional[str] = None,
    gdr_tsv: Optional[str] = None,
    nccl_reference_json: Optional[str] = None,
) -> LoadResult:
    """Load every available source into one LoadResult (apertures already
    applied to the direct profiles). Missing or unreadable sources are
    loudly logged; the dispatcher then simply has fewer measured entries
    and hard rule 1 keeps the affected classes on the status quo."""
    merged = LoadResult()

    def merge(part: LoadResult) -> None:
        merged.profiles.extend(part.profiles)
        merged.apertures.update(part.apertures)
        merged.errors.extend(part.errors)
        merged.skipped.extend(part.skipped)

    sources = [
        ("p2p capability matrix", p2p_capability_json, load_p2p_capability_matrix),
        ("p2p d2d bench", p2p_d2d_json, load_p2p_d2d_bench),
        ("nccl reference", nccl_reference_json, load_nccl_reference),
    ]
    for label, path, loader in sources:
        if path is None or not os.path.exists(path):
            logger.warning(
                "barlink path dispatcher: rate source %r not available (%s) -- "
                "its paths stay PLACEHOLDER and their classes on the "
                "status-quo choice.",
                label,
                path or "not configured",
            )
            continue
        payload = _read_json(path, merged)
        if payload is not None:
            merge(loader(payload))
    if gdr_tsv is None or not os.path.exists(gdr_tsv):
        logger.warning(
            "barlink path dispatcher: rate source '#278 gdr matrix' not "
            "available (%s) -- its paths stay PLACEHOLDER and their classes "
            "on the status-quo choice.",
            gdr_tsv or "not configured",
        )
    else:
        try:
            with open(gdr_tsv) as f:
                merge(load_gdr_matrix_tsv(f))
        except OSError as e:
            merged.errors.append(f"{gdr_tsv}: {e!r}")
    apply_apertures(merged.profiles, merged.apertures)
    for err in merged.errors:
        logger.warning("barlink path dispatcher rate load: %s", err)
    return merged
