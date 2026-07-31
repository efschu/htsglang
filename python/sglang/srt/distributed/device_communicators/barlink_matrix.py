# SPDX-License-Identifier: Apache-2.0
"""barlink path matrix: the planner.

At startup, the planner measures the **directed** edges of the ensemble
and derives from them roles, domains, algorithm, chunk size, split, and a
fan-in staging schedule.

Four principles carry the whole design:

1. **Measured, not derived.** No ``lspci`` width is taken as truth.
   ``lspci`` says x8; whether the edge actually carries 12.7 GB/s is
   decided by the measurement. The link width may serve, at most, as a
   starting estimate, and is therefore only read for *tie-breaks* (ring
   ordering) and for the plausibility report, never as a basis for a
   decision.
2. **Capability and policy are kept separate.** What the hardware can do
   is measured (``Measurement``); what gets used is configured
   (``BarlinkConfig``). The two only meet in ``plan_collective()``.
3. **Explainable.** ``Plan.explanation()`` prints the measured
   capacities, the roles that follow from them, and the chosen algorithm
   together with predicted times. Without this, nobody can debug this on
   unfamiliar hardware.
4. **Overridable at every level.** Planner off, pin an individual role,
   set domains by hand, force the algorithm, force the chunk size, force
   the split -- and this is also the only way to measure the planner
   against itself.

**The entry belongs to the directed edge (source->destination), not to
the link.** PCIe is full duplex, outbound and inbound are separate
budgets; the assignment is allowed to be asymmetric.

Decision rules and their evidence
------------------------------------
All four come from measurements on this rig; the evidence lives in
``/spinning/nvidia-smallbar-p2p/MESSUNG_NEBENLAEUFIGKEIT.md``. Here they
are encoded as a *model* whose parameters come from the startup
measurement -- not as hardcoded numbers.

R1  **Mesh vs. ring.** Below saturation, the mesh's concurrency is free
    (measured ratio 0.99 at 20 KiB), and the mesh needs only 2 steps
    instead of 2(R-1) -> mesh. At saturation, concurrency buys nothing
    (1.03x at 1 MiB) **and** the split is even rather than proportional,
    so it wastes the fast edges -> ring. Encoded in ``_time_mesh`` /
    ``_time_ring``: the mesh term splits the measured fan-in cap *evenly*
    per source, the ring term does not.

R2  **Leaves.** Whoever's measured capacity sits clearly below the
    median carries no transit traffic. If everyone is roughly level,
    there are no leaves and the flat scheme is kept. Threshold
    ``leaf_threshold`` (default 0.6 x median).

R3  **Fan-in cap.** No card receives more than its measured cap,
    regardless of the number of sources (this rig: ~13 GB/s). The cap is
    split **evenly per source**, not proportionally -- the x8 source
    dropped from 12.81 to 6.75 GB/s, the x4 source kept its 6.46. Hence:
    never let a fast and a slow card write into the same destination at
    the same time; stagger them instead (``Plan.tiers``).

R4  **Full duplex does not double.** At saturation each direction drops
    to ~65%, sum 1.32x. Below saturation it is nearly free (0.99 at
    20 KiB). The factor is measured (``Measurement.duplex``) and enters
    the pipelining calculation as an overlap credit -- nowhere in the code
    is there a hardcoded 2.

What the planner does NOT do
--------------------------------
It does not choose a path per edge (BAR1 / NIC / system RAM). That is the
job of the composite transport (``barlink_bar1.py`` supplies one of the
sub-paths). The planner answers the separate question: which *role* does
each rank have, and with which *algorithm* is the collective decomposed.
The two levels meet via ``Sensor``: whoever has a real point-to-point path
hands it in as a pair sensor and gets real edge measurements instead of
the self-load estimate.

Rank uniformity
-------------------
The plan MUST be identical on every rank -- the collectives' SPMD
assumption depends on it. Hence: measure, share the quantized raw values
via ``all_gather_object``, every rank computes from the **same** data with
the same code, then reconcile the plan checksum across the group. If a
rank diverges, that is a named startup error, not a warning.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import pathlib
import re
import statistics
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

from sglang.srt.distributed.device_communicators import (
    barlink_env_guard,  # noqa: F401  (rejects retired SGLANG_HTCCL* vars)
)

logger = logging.getLogger(__name__)

# Fingerprint component: if the planning logic changes, an old cache
# entry becomes invalid even when the hardware is the same.
PLANNER_VERSION = 1

# Default size ladder for the measurement, in KiB. The three points are
# the operating regimes that matter: decode (20), transition (80),
# prefill (1024).
DEFAULT_SIZES_KIB = (20, 80, 1024)

ALGORITHMS = ("mesh", "ring", "star", "hierarchical")
PLANNER_MODES = ("auto", "fixed", "off")
NIC_MODES = ("never", "on_demand", "always")
ROLES = ("leaf", "domain", "hub")

# ---------------------------------------------------------------------------
# Reference values for this rig -- ONLY for the plausibility report.
#
# They feed into NO decision. If a startup measurement deviates sharply,
# that is logged, so a broken measurement setup stands out instead of
# passing as a rig quirk. Evidence: MESSUNG_NEBENLAEUFIGKEIT.md.
# ---------------------------------------------------------------------------
_REFERENCE_FANIN_GBPS = 13.16          # cap at 1 MiB, two sources
_REFERENCE_DUPLEX_TOTAL_1MIB = 1.32    # sum of both directions / one direction
_REFERENCE_DUPLEX_TOTAL_20KIB = 1.47


# ===========================================================================
# Configuration
# ===========================================================================


class ConfigError(ValueError):
    """A named configuration error -- never silently healed."""


@dataclass(frozen=True)
class MeasureConfig:
    """What the planner measures at startup, and how long it's allowed to take."""

    sizes_kib: tuple[int, ...] = DEFAULT_SIZES_KIB
    repeats: int = 32
    warmup: int = 8
    # Ceiling for the entire startup measurement. If it's exceeded, the
    # planner thins out the size ladder and then the repetition count,
    # and logs it -- it does NOT abort, and it does not quietly keep going
    # with fewer samples without saying so.
    budget_ms: float = 2000.0
    # Fan-in (R3) and duplex (R4) cost their own rounds. Can be disabled,
    # because they only produce real values with a pair sensor.
    fanin: bool = True
    duplex: bool = True
    # Cache of the matrix plus its fingerprint. None = default path.
    cache: Optional[str] = None
    cache_off: bool = False


@dataclass(frozen=True)
class CollectiveConfig:
    planner: str = "auto"                # auto | fixed | off
    algorithm: str = "auto"              # auto | mesh | ring | star | hierarchical
    chunk_kib: Optional[int] = None      # None == "auto"
    leaf_threshold: float = 0.6          # capacity < 0.6 x median -> leaf
    split: Any = "auto"                  # auto | even | proportional | {bdf: [...]}
    roles: Mapping[str, str] = field(default_factory=dict)      # bdf -> role
    domains: tuple[tuple[str, ...], ...] = ()                   # lists of BDFs
    # R3: within one fan-in wave, source capacities may differ by at most
    # this factor. Beyond it, they get staggered.
    tier_ratio: float = 1.5
    # Fraction of the measured capacity above which an edge counts as
    # saturated. Only used for the explanation/report -- the actual
    # decision is made via the cost comparison, not via this threshold.
    saturation_share: float = 0.75
    measure: MeasureConfig = field(default_factory=MeasureConfig)


@dataclass(frozen=True)
class BarlinkConfig:
    collective: CollectiveConfig = field(default_factory=CollectiveConfig)
    nic: str = "never"                   # never | on_demand | always


# -- File --------------------------------------------------------------

_MEASURE_KEYS = {
    "sizes_kib", "repeats", "warmup", "budget_ms",
    "fanin", "duplex", "cache", "cache_off",
}
_COLLECTIVE_KEYS = {
    "planner", "algorithm", "chunk_kib", "leaf_threshold", "split",
    "roles", "domains", "tier_ratio", "saturation_share", "measure",
}
_ROOT_KEYS = {"collective", "nic"}


def _bool(value: Any, where: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off"):
            return False
    raise ConfigError(f"{where}: {value!r} is not a boolean value")


def _check_key(given: Iterable[str], allowed: set[str], where: str) -> None:
    unknown = sorted(set(given) - allowed)
    if unknown:
        raise ConfigError(
            f"{where}: unknown keys {unknown}; allowed are "
            f"{sorted(allowed)}. (A silently ignored typo in the "
            f"configuration is exactly the kind of bug that later sends "
            f"someone hunting for performance that was switched off by "
            f"configuration.)"
        )


#: A PCI address, with or without a domain. Serves as a probe against
#: anything that merely looks like one (e.g. a bare bus number).
_IS_BDF = re.compile(r"^(?:[0-9a-fA-F]{4}:)?[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.\d$")


def _norm_bdf(s: str) -> str:
    """``05:00.0`` and ``0000:05:00.0`` are the same key."""
    s = str(s).strip().lower()
    if s.count(":") == 1:
        s = "0000:" + s
    return s


def _read_file(path: str) -> dict:
    p = pathlib.Path(path).expanduser()
    if not p.is_file():
        raise ConfigError(f"config file {path!r} does not exist")
    text = p.read_text()
    data: Any
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:  # pragma: no cover - PyYAML is a project dependency
            raise ConfigError(
                f"{path}: YAML requested, but PyYAML is missing ({e}). "
                f"JSON (.json) works without an extra package."
            ) from e
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping, not {type(data)}")
    # Both `barlink: {...}` and the bare subtree are accepted.
    if set(data) == {"barlink"}:
        data = data["barlink"] or {}
    elif "barlink" in data:
        data = data["barlink"] or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: `barlink` must be a mapping")
    return data


def _from_mapping(base: BarlinkConfig, d: Mapping[str, Any], where: str) -> BarlinkConfig:
    _check_key(d, _ROOT_KEYS, where)
    coll = base.collective
    nic = base.nic
    if "nic" in d:
        nic = str(d["nic"]).strip().lower().replace("-", "_")
    kd = d.get("collective") or {}
    if not isinstance(kd, Mapping):
        raise ConfigError(f"{where}.collective must be a mapping")
    _check_key(kd, _COLLECTIVE_KEYS, f"{where}.collective")
    changes: dict[str, Any] = {}
    if "planner" in kd:
        changes["planner"] = str(kd["planner"]).strip().lower()
    if "algorithm" in kd:
        changes["algorithm"] = str(kd["algorithm"]).strip().lower()
    if "chunk_kib" in kd:
        v = kd["chunk_kib"]
        changes["chunk_kib"] = None if str(v).strip().lower() == "auto" else int(v)
    if "leaf_threshold" in kd:
        changes["leaf_threshold"] = float(kd["leaf_threshold"])
    if "tier_ratio" in kd:
        changes["tier_ratio"] = float(kd["tier_ratio"])
    if "saturation_share" in kd:
        changes["saturation_share"] = float(kd["saturation_share"])
    if "split" in kd:
        changes["split"] = _read_split(kd["split"], f"{where}.collective")
    if "roles" in kd:
        r = kd["roles"] or {}
        if not isinstance(r, Mapping):
            raise ConfigError(f"{where}.collective.roles must be a mapping")
        changes["roles"] = {
            _norm_bdf(k): str(v).strip().lower() for k, v in r.items()
        }
    if "domains" in kd:
        dom = kd["domains"] or []
        if not isinstance(dom, Sequence) or isinstance(dom, (str, bytes)):
            raise ConfigError(f"{where}.collective.domains must be a list of lists")
        changes["domains"] = tuple(
            tuple(_norm_bdf(x) for x in group) for group in dom
        )
    if "measure" in kd:
        md = kd["measure"] or {}
        if not isinstance(md, Mapping):
            raise ConfigError(f"{where}.collective.measure must be a mapping")
        _check_key(md, _MEASURE_KEYS, f"{where}.collective.measure")
        mchanges: dict[str, Any] = {}
        if "sizes_kib" in md:
            mchanges["sizes_kib"] = tuple(int(x) for x in md["sizes_kib"])
        for key in ("repeats", "warmup"):
            if key in md:
                mchanges[key] = int(md[key])
        if "budget_ms" in md:
            mchanges["budget_ms"] = float(md["budget_ms"])
        for key in ("fanin", "duplex", "cache_off"):
            if key in md:
                mchanges[key] = _bool(md[key], f"{where}.collective.measure.{key}")
        if "cache" in md:
            mchanges["cache"] = None if md["cache"] is None else str(md["cache"])
        changes["measure"] = replace(coll.measure, **mchanges)
    return BarlinkConfig(collective=replace(coll, **changes), nic=nic)


def _read_split(v: Any, where: str) -> Any:
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("auto", "even", "proportional"):
            return s
        raise ConfigError(
            f"{where}.split: {v!r} unknown "
            f"(auto | even | proportional | mapping BDF -> weights)"
        )
    if isinstance(v, Mapping):
        out: dict[str, tuple[float, ...]] = {}
        for k, weights in v.items():
            if not isinstance(weights, Sequence) or isinstance(weights, (str, bytes)):
                raise ConfigError(f"{where}.split[{k!r}]: expected a list of weights")
            g = tuple(float(x) for x in weights)
            if not g or any(x < 0 for x in g) or sum(g) <= 0:
                raise ConfigError(f"{where}.split[{k!r}]: weights must be >= 0, sum > 0")
            out[_norm_bdf(k)] = g
        return out
    raise ConfigError(f"{where}.split: {v!r} not parseable")


# -- Environment -------------------------------------------------------------
#
# Order: default < file < environment variable. The environment always
# wins, because it is the way to measure a run against itself without
# changing any file.

_ENV_PREFIX = "SGLANG_BARLINK_"


def load_config(env: Optional[Mapping[str, str]] = None) -> BarlinkConfig:
    """Default < file (``SGLANG_BARLINK_CONFIG``) < environment.

    RANK UNIFORMITY: reads exclusively process-global state that is the
    same on every rank (environment, file). Nothing in here may ever
    depend on the rank -- otherwise two ranks would plan differently and
    the collectives' SPMD assumption would collapse.
    """
    env = os.environ if env is None else env
    k = BarlinkConfig()
    path = env.get(_ENV_PREFIX + "CONFIG")
    if path:
        k = _from_mapping(k, _read_file(path), where=path)

    over: dict[str, Any] = {}
    coll_over: dict[str, Any] = {}
    measure_over: dict[str, Any] = {}

    def _e(name: str) -> Optional[str]:
        return env.get(_ENV_PREFIX + name)

    if (v := _e("NIC")) is not None or (v := _e("MATRIX_NIC")) is not None:
        # `MATRIX_NIC` is the name used in ENTWURF_BARLINK_TRANSPORT.md;
        # it spells the value `on-demand`; both spellings are accepted.
        over["nic"] = v.strip().lower().replace("-", "_")
    if (v := _e("PLANNER")) is not None:
        coll_over["planner"] = v.strip().lower()
    if (v := _e("ALGORITHM")) is not None:
        coll_over["algorithm"] = v.strip().lower()
    if (v := _e("CHUNK_KIB")) is not None:
        coll_over["chunk_kib"] = None if v.strip().lower() == "auto" else int(v)
    if (v := _e("LEAF_THRESHOLD")) is not None:
        coll_over["leaf_threshold"] = float(v)
    if (v := _e("TIER_RATIO")) is not None:
        coll_over["tier_ratio"] = float(v)
    if (v := _e("SATURATION_SHARE")) is not None:
        coll_over["saturation_share"] = float(v)
    if (v := _e("SPLIT")) is not None:
        s = v.strip()
        coll_over["split"] = _read_split(
            json.loads(s) if s.startswith("{") else s, "SGLANG_BARLINK_SPLIT"
        )
    if (v := _e("ROLES")) is not None:
        # "0000:05:00.0=leaf,0000:0a:00.0=domain" or JSON
        s = v.strip()
        if s.startswith("{"):
            raw = json.loads(s)
        else:
            raw = {}
            for part in s.split(","):
                if not part.strip():
                    continue
                if "=" not in part:
                    raise ConfigError(
                        f"SGLANG_BARLINK_ROLES: {part!r} has no '='; expected "
                        f"'<bdf>=<role>[,<bdf>=<role>]' or JSON"
                    )
                bdf, role = part.split("=", 1)
                raw[bdf] = role
        coll_over["roles"] = {_norm_bdf(a): str(b).strip().lower() for a, b in raw.items()}
    if (v := _e("DOMAINS")) is not None:
        # "05:00.0+0b:00.0;0a:00.0" or a JSON list of lists
        s = v.strip()
        if s.startswith("["):
            raw = json.loads(s)
        else:
            raw = [g.split("+") for g in s.split(";") if g.strip()]
        coll_over["domains"] = tuple(tuple(_norm_bdf(x) for x in g) for g in raw)
    if (v := _e("MEASURE_SIZES_KIB")) is not None:
        measure_over["sizes_kib"] = tuple(int(x) for x in v.replace(";", ",").split(","))
    if (v := _e("MEASURE_REPEATS")) is not None:
        measure_over["repeats"] = int(v)
    if (v := _e("MEASURE_WARMUP")) is not None:
        measure_over["warmup"] = int(v)
    if (v := _e("MEASURE_BUDGET_MS")) is not None:
        measure_over["budget_ms"] = float(v)
    if (v := _e("MEASURE_FANIN")) is not None:
        measure_over["fanin"] = _bool(v, "SGLANG_BARLINK_MEASURE_FANIN")
    if (v := _e("MEASURE_DUPLEX")) is not None:
        measure_over["duplex"] = _bool(v, "SGLANG_BARLINK_MEASURE_DUPLEX")
    if (v := _e("MATRIX_CACHE")) is not None:
        measure_over["cache"] = v
    if (v := _e("MATRIX_CACHE_OFF")) is not None:
        measure_over["cache_off"] = _bool(v, "SGLANG_BARLINK_MATRIX_CACHE_OFF")

    coll = k.collective
    if measure_over:
        coll_over["measure"] = replace(coll.measure, **measure_over)
    if coll_over:
        coll = replace(coll, **coll_over)
    k = replace(k, collective=coll, **over)
    _validate(k)
    return k


def _validate(k: BarlinkConfig) -> None:
    if k.nic not in NIC_MODES:
        raise ConfigError(f"nic={k.nic!r} unknown; allowed {list(NIC_MODES)}")
    c = k.collective
    if c.planner not in PLANNER_MODES:
        raise ConfigError(f"collective.planner={c.planner!r}; allowed {list(PLANNER_MODES)}")
    if c.algorithm != "auto" and c.algorithm not in ALGORITHMS:
        raise ConfigError(
            f"collective.algorithm={c.algorithm!r}; allowed "
            f"{['auto', *ALGORITHMS]}"
        )
    if not 0.0 < c.leaf_threshold <= 1.0:
        raise ConfigError(
            f"collective.leaf_threshold={c.leaf_threshold} must lie in (0, 1] "
            f"(fraction of the median; 1.0 means 'everything below the "
            f"median is a leaf')"
        )
    if c.tier_ratio < 1.0:
        raise ConfigError("collective.tier_ratio must be >= 1")
    if not 0.0 < c.saturation_share <= 1.0:
        raise ConfigError("collective.saturation_share must lie in (0, 1]")
    if c.chunk_kib is not None and c.chunk_kib <= 0:
        raise ConfigError("collective.chunk_kib must be > 0 or 'auto'")
    for bdf, role in c.roles.items():
        if role not in ROLES:
            raise ConfigError(
                f"collective.roles[{bdf!r}]={role!r}; allowed {list(ROLES)}"
            )
    m = c.measure
    if not m.sizes_kib or any(g <= 0 for g in m.sizes_kib):
        raise ConfigError("collective.measure.sizes_kib: expected positive values")
    if m.repeats <= 0 or m.warmup < 0:
        raise ConfigError("collective.measure.repeats > 0, warmup >= 0")
    if m.budget_ms <= 0:
        raise ConfigError("collective.measure.budget_ms must be > 0")
    if c.planner == "off" and c.algorithm == "auto":
        raise ConfigError(
            "collective.planner=off requires a fixed collective.algorithm -- "
            "'off' + 'auto' would mean 'don't measure, but still choose', "
            "and that doesn't exist. No silent fallback."
        )
    if c.planner == "fixed" and c.measure.cache_off:
        raise ConfigError(
            "collective.planner=fixed needs the cache; measure.cache_off=1 takes "
            "away its only source. You probably mean planner=auto."
        )


# ===========================================================================
# Sensor: what gets measured
# ===========================================================================


class Sensor(Protocol):
    """The planner's measurement probe.

    Two forms, deliberately kept separate:

    ``self_load``   measures this rank's own PCIe link (GPU <-> pinned
                    host memory) and needs no peer path. Always
                    available, delivers **node capacities** per
                    direction.
    ``pair``        measures the directed rank->rank edge over a real
                    point-to-point path. Only available if a transport
                    supplies one (``barlink_bar1``). Delivers **edge
                    capacities**.

    A sensor that cannot do ``pair`` returns ``None``; the planner then
    falls back to the self-load estimate and records that in the
    explanation. It does NOT pretend to have measured the edge.
    """

    def name(self) -> str: ...

    def self_load(self, nbytes: int, direction: str) -> float:
        """GB/s. ``direction`` is ``"d2h"`` (outbound) or ``"h2d"`` (inbound)."""
        ...

    def self_load_duplex(self, nbytes: int) -> Optional[float]:
        """Sum of both directions running simultaneously, GB/s. ``None`` if not measurable."""
        ...

    def pair(self, dst: int, nbytes: int) -> Optional[float]:
        """GB/s from *this* rank to rank ``dst``. ``None`` = no pair path."""
        ...

    def pair_receive(self, source: int, nbytes: int) -> None:
        """Counterpart to ``pair`` on the destination side (synchronization only)."""
        ...


class SelfLoadSensor:
    """Default sensor: measures the rank's own link, with no peer path at all.

    This is a genuine measurement of this card in this slot -- not the
    ``lspci`` width. Its limitation is a different one: it measures GPU
    <-> host, not GPU <-> GPU. An edge that crosses a switch uplink or two
    root complexes can be considerably slower than the minimum of the two
    nodes' rates. The planner therefore marks edges derived from it as
    ``source="self_load"``, and the explanation says so.
    """

    def __init__(self, device, max_bytes: int = 4 << 20,
                 repeats: Optional[int] = None):
        import torch

        self.device = device
        self._torch = torch
        self._max_bytes = max_bytes
        # None = the size-dependent default (small messages need more
        # rounds, otherwise you end up measuring the clock instead of the
        # link).
        self._reps = repeats
        self._dev = torch.empty(max_bytes, dtype=torch.uint8, device=device)
        self._host = torch.empty(max_bytes, dtype=torch.uint8, pin_memory=True)
        self._host2 = torch.empty(max_bytes, dtype=torch.uint8, pin_memory=True)
        self._s1 = torch.cuda.Stream(device=device)
        self._s2 = torch.cuda.Stream(device=device)

    def name(self) -> str:
        return "self_load"

    def _run(self, nbytes: int, direction: str, n: int) -> None:
        d = self._dev[:nbytes]
        h = self._host[:nbytes]
        for _ in range(n):
            if direction == "d2h":
                h.copy_(d, non_blocking=True)   # D2H == card's outbound direction
            else:
                d.copy_(h, non_blocking=True)   # H2D == card's inbound direction

    def self_load(self, nbytes: int, direction: str) -> float:
        torch = self._torch
        nbytes = min(nbytes, self._max_bytes)
        self._run(nbytes, direction, 8)
        torch.cuda.synchronize(self.device)
        n = self._reps or _repeats_for(nbytes)
        t0 = time.perf_counter()
        self._run(nbytes, direction, n)
        torch.cuda.synchronize(self.device)
        dt = time.perf_counter() - t0
        return (n * nbytes) / dt / 1e9 if dt > 0 else 0.0

    def self_load_duplex(self, nbytes: int) -> Optional[float]:
        """R4 measured, not assumed: both directions at once.

        Two transfers on two streams, so the copy engines genuinely run in
        parallel. Returns the *sum* of both directions; the planner
        derives the duplex factor from that against the sum of the
        individual rates. On this rig, the sum at 1 MiB was 1.32x one
        direction -- i.e. deliberately not 2x.
        """
        torch = self._torch
        nbytes = min(nbytes, self._max_bytes)
        d = self._dev[:nbytes]
        h1 = self._host[:nbytes]
        h2 = self._host2[:nbytes]
        n = self._reps or _repeats_for(nbytes)

        def round(k: int) -> None:
            for _ in range(k):
                with torch.cuda.stream(self._s1):
                    h1.copy_(d, non_blocking=True)
                with torch.cuda.stream(self._s2):
                    d.copy_(h2, non_blocking=True)

        round(8)
        torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        round(n)
        torch.cuda.synchronize(self.device)
        dt = time.perf_counter() - t0
        return (2 * n * nbytes) / dt / 1e9 if dt > 0 else None

    def pair(self, dst: int, nbytes: int) -> Optional[float]:
        return None   # no point-to-point path -- honest rather than estimated

    def pair_receive(self, source: int, nbytes: int) -> None:
        return None


def _repeats_for(nbytes: int) -> int:
    """More repetitions for small messages.

    The reference measurement's 20-KiB rows sit close to clock resolution
    even at 64 repetitions; below ~64 rounds the percentages there are
    noise. Large messages don't need that many and it would eat into the
    time budget.
    """
    if nbytes <= 64 * 1024:
        return 64
    if nbytes <= 512 * 1024:
        return 32
    return 16


# ===========================================================================
# Measurement: the result of the startup measurement
# ===========================================================================


@dataclass
class Measurement:
    """Raw findings. Contains only what was measured, no policy.

    All rates in GB/s (10^9 bytes/s, as elsewhere in the repo).
    """

    world: int
    sizes: tuple[int, ...]      # bytes
    bdfs: tuple[str, ...]       # per rank
    names: tuple[str, ...]      # per rank (card name)
    sensor: str                 # which sensor measured
    # Node rates per rank and size.
    outbound: dict[int, list[float]] = field(default_factory=dict)   # per rank
    inbound: dict[int, list[float]] = field(default_factory=dict)
    duplex_total: dict[int, list[float]] = field(default_factory=dict)
    # Edge rates, if a pair sensor was present: (from, to) -> per size
    edge: dict[tuple[int, int], list[float]] = field(default_factory=dict)
    # R3: fan-in cap per destination and size, plus shares per source.
    fanin_cap: dict[int, list[float]] = field(default_factory=dict)
    fanin_shares: dict[tuple[int, int], list[float]] = field(default_factory=dict)
    # Line t(N) = latency + N/rate fitted from the size ladder, per rank:
    # the startup cost of one step, in seconds.
    latency_s: dict[int, float] = field(default_factory=dict)
    # Shared bottlenecks: by how much an edge slows down when ALL pairs
    # talk at once (mesh) instead of just one (ring). 1.0 means "no
    # interference" AND is also the value for "not measured" -- which of
    # the two applies is told by `mesh_factor_measured`.
    #
    # This is the term the ring-wins hypothesis hinges on: a mesh forces
    # EVERY pair to talk, including across switch uplinks and NUMA hops,
    # while a topology-ordered ring crosses shared bottlenecks only once
    # per round. Per-edge capacities alone cannot capture this -- they are
    # each measured on their own edge.
    mesh_factor: list[float] = field(default_factory=list)
    mesh_factor_measured: bool = False
    duration_s: float = 0.0
    hints: list[str] = field(default_factory=list)

    # -- derived views (no state) --------------------------------------

    def capacity(self, src: int, dst: int, gi: int) -> float:
        """Directed rank->rank capacity at size ``sizes[gi]``.

        A measured edge wins. Otherwise the self-load estimate
        ``min(source's outbound, destination's inbound)`` -- explicitly an
        **upper bound**, because it cannot see shared bottlenecks (switch
        uplink, a second root complex).
        """
        e = self.edge.get((src, dst))
        if e is not None:
            return e[gi]
        return min(self.outbound[src][gi], self.inbound[dst][gi])

    def edge_measured(self, src: int, dst: int) -> bool:
        return (src, dst) in self.edge

    def cap(self, dst: int, gi: int) -> float:
        d = self.fanin_cap.get(dst)
        if d is not None:
            return d[gi]
        return self.inbound[dst][gi]

    def cap_measured(self, dst: int) -> bool:
        return dst in self.fanin_cap

    def mesh_penalty(self, gi: int) -> float:
        """Factor applied to the mesh transfer term. >= 1."""
        if not self.mesh_factor or gi >= len(self.mesh_factor):
            return 1.0
        return max(1.0, self.mesh_factor[gi])

    def duplex_factor(self, rank: int, gi: int) -> float:
        """Sum of both directions / larger single direction. 1.0 = no gain."""
        d = self.duplex_total.get(rank)
        if d is None:
            return 1.0
        single = max(self.outbound[rank][gi], self.inbound[rank][gi])
        return d[gi] / single if single > 0 else 1.0

    def step_s(self) -> float:
        """Startup cost of one collective step, in seconds (median across ranks)."""
        if not self.latency_s:
            return 0.0
        return statistics.median(self.latency_s.values())

    def as_dict(self) -> dict:
        d = asdict(self)
        # Tuple keys aren't JSON-safe.
        d["edge"] = {f"{a}->{b}": v for (a, b), v in self.edge.items()}
        d["fanin_shares"] = {f"{a}->{b}": v for (a, b), v in self.fanin_shares.items()}
        d["outbound"] = {str(k): v for k, v in self.outbound.items()}
        d["inbound"] = {str(k): v for k, v in self.inbound.items()}
        d["duplex_total"] = {str(k): v for k, v in self.duplex_total.items()}
        d["fanin_cap"] = {str(k): v for k, v in self.fanin_cap.items()}
        d["latency_s"] = {str(k): v for k, v in self.latency_s.items()}
        return d

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "Measurement":
        def pair(s: str) -> tuple[int, int]:
            a, b = s.split("->")
            return int(a), int(b)

        m = Measurement(
            world=int(d["world"]),
            sizes=tuple(int(x) for x in d["sizes"]),
            bdfs=tuple(d["bdfs"]),
            names=tuple(d["names"]),
            sensor=str(d["sensor"]),
            duration_s=float(d.get("duration_s", 0.0)),
            hints=list(d.get("hints", [])),
        )
        m.outbound = {int(k): list(v) for k, v in d["outbound"].items()}
        m.inbound = {int(k): list(v) for k, v in d["inbound"].items()}
        m.duplex_total = {int(k): list(v) for k, v in d.get("duplex_total", {}).items()}
        m.edge = {pair(k): list(v) for k, v in d.get("edge", {}).items()}
        m.fanin_cap = {int(k): list(v) for k, v in d.get("fanin_cap", {}).items()}
        m.fanin_shares = {
            pair(k): list(v) for k, v in d.get("fanin_shares", {}).items()
        }
        m.latency_s = {int(k): float(v) for k, v in d.get("latency_s", {}).items()}
        m.mesh_factor = [float(x) for x in d.get("mesh_factor", [])]
        m.mesh_factor_measured = bool(d.get("mesh_factor_measured", False))
        return m


def _quant(x: float) -> float:
    """Quantize before deciding.

    The plan must come out bit-identical on every rank. The raw values do
    arrive as identical ``float``s via ``all_gather_object``, but a
    decision threshold that flips on the 12th decimal digit is a random
    number generator regardless. Three decimal digits in GB/s is about
    1 MB/s -- well below any measurement noise.
    """
    if not math.isfinite(x):
        return 0.0
    return round(float(x), 3)


# ===========================================================================
# The plan
# ===========================================================================


@dataclass(frozen=True)
class Stage:
    """One stage of the size ladder: up to ``max_bytes``, ``algorithm`` applies."""

    min_bytes: int                       # smallest size this was measured for
    max_bytes: int                       # inclusive; -1 == "and everything above"
    algorithm: str
    prediction_s: Mapping[str, float]    # algorithm -> predicted time
    reason: str


@dataclass(frozen=True)
class Plan:
    world: int
    bdfs: tuple[str, ...]
    roles: tuple[str, ...]                     # per rank
    domain: tuple[int, ...]                    # ranks of the reduction domain
    leaves: tuple[int, ...]
    parents: Mapping[int, tuple[int, ...]]     # leaf -> domain nodes
    split: Mapping[int, tuple[int, ...]]       # leaf -> per-mille share per parent
    ring_order: tuple[int, ...]
    ladder: tuple[Stage, ...]
    chunk_bytes: int
    tiers: Mapping[int, tuple[tuple[int, ...], ...]]  # destination -> waves of sources
    config_summary: Mapping[str, Any]
    measurement: Optional[Measurement] = None
    source: str = "measured"                   # measured | cached | fixed

    # -- queries a transport needs ---------------------------------------

    def algorithm_for(self, nbytes: int) -> str:
        for stage in self.ladder:
            if stage.max_bytes < 0 or nbytes <= stage.max_bytes:
                return stage.algorithm
        return self.ladder[-1].algorithm

    def is_leaf(self, rank: int) -> bool:
        return self.roles[rank] == "leaf"

    def checksum(self) -> str:
        """Fingerprint of the *decisions*, not of the raw measurement.

        Deliberately without ``measurement``: the raw values differ between
        ranks by measurement noise once every rank contributes its own
        numbers. What MUST agree is the plan.
        """
        core = {
            "world": self.world,
            "roles": list(self.roles),
            "domain": list(self.domain),
            "leaves": list(self.leaves),
            "parents": {str(k): list(v) for k, v in sorted(self.parents.items())},
            "split": {str(k): list(v) for k, v in sorted(self.split.items())},
            "ring_order": list(self.ring_order),
            "ladder": [(s.min_bytes, s.max_bytes, s.algorithm) for s in self.ladder],
            "chunk_bytes": self.chunk_bytes,
            "tiers": {
                str(k): [list(w) for w in v] for k, v in sorted(self.tiers.items())
            },
        }
        raw = json.dumps(core, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # -- Explanation --------------------------------------------------------

    def explanation(self) -> str:
        """Human-readable: what was measured, what was inferred, what was chosen.

        Without this block, nobody can debug this on unfamiliar hardware
        -- which is why it's mandatory output and not gated behind a
        debug flag.
        """
        z: list[str] = []
        a = z.append
        a("=" * 78)
        a(f"barlink path matrix: plan {self.checksum()} ({self.source}), "
          f"{self.world} ranks")
        a("=" * 78)
        m = self.measurement
        if m is not None:
            a(f"Sensor: {m.sensor}   Measurement duration: {m.duration_s * 1000:.0f} ms")
            a("")
            a("-- measured node rates (GB/s, outbound/inbound per size) --")
            a("   The sysfs column is shown ONLY for comparison and feeds "
              "into no decision:")
            a("   if it says x8 and the measurement says 6 GB/s, either the "
              "card is downclocked")
            a("   or the measurement setup is broken -- lspci is a starting "
              "estimate, not truth.")
            header = "  Rank  Card                   sysfs   " + "".join(
                f"{_kib(g):>18}" for g in m.sizes
            )
            a(header)
            for r in range(self.world):
                lb = link_width(self.bdfs[r])
                lbs = f"x{lb[0]}" if lb else "?"
                line = (f"  {r:>4}  {self.bdfs[r]:<14}{m.names[r][:8]:<8}"
                        f"{lbs:<8}")
                for gi in range(len(m.sizes)):
                    line += f"{m.outbound[r][gi]:>8.2f}/{m.inbound[r][gi]:<9.2f}"
                a(line)
            if m.duplex_total:
                a("")
                a("-- full duplex (sum of both directions / stronger single direction) --")
                a("   R4: measured, not assumed. 1.00 = the reverse direction "
                  "isn't free at all;")
                a("   2.00 would be the naive expectation. Reference for this rig: "
                  f"{_REFERENCE_DUPLEX_TOTAL_20KIB:.2f}x at 20 KiB, "
                  f"{_REFERENCE_DUPLEX_TOTAL_1MIB:.2f}x at 1 MiB.")
                for r in range(self.world):
                    if r not in m.duplex_total:
                        continue
                    values = "  ".join(
                        f"{_kib(g)}: {m.duplex_factor(r, gi):.2f}x"
                        for gi, g in enumerate(m.sizes)
                    )
                    a(f"  Rank {r}: {values}")
            a("")
            a("-- directed edge capacities (GB/s) --")
            a("   'M' = measured on the pair, 'S' = estimate min(outbound, "
              "inbound) from self-load")
            for gi, g in enumerate(m.sizes):
                a(f"  at {_kib(g)}:")
                a("        " + "".join(f"{'->' + str(j):>12}" for j in range(self.world)))
                for i in range(self.world):
                    line = f"    {i:>2}  "
                    for j in range(self.world):
                        if i == j:
                            line += f"{'-':>12}"
                        else:
                            mark = "M" if m.edge_measured(i, j) else "S"
                            line += f"{m.capacity(i, j, gi):>10.2f}{mark}"
                    a(line)
            a("")
            a("-- fan-in cap per destination (GB/s) --")
            a("   R3: no card receives more than the cap, regardless of how "
              "many sources.")
            a(f"   Reference for this rig: {_REFERENCE_FANIN_GBPS:.2f} GB/s at 1 MiB "
              "with two sources.")
            for j in range(self.world):
                mark = "measured" if m.cap_measured(j) else "estimated from inbound rate"
                values = "  ".join(
                    f"{_kib(g)}: {m.cap(j, gi):.2f}" for gi, g in enumerate(m.sizes)
                )
                a(f"  Destination {j} ({self.bdfs[j]}): {values}   [{mark}]")
            a(f"  Step cost (startup cost per collective step): "
              f"{m.step_s() * 1e6:.2f} us")
            a("")
            a("-- shared bottlenecks: mesh penalty vs. the ring --")
            if m.mesh_factor_measured:
                a("   Measured: all pairs simultaneously vs. one pair alone. "
                  ">1 means the mesh")
                a("   loses to uplinks/NUMA hops that a per-edge capacity "
                  "cannot see.")
            else:
                a("   NOT MEASURED -- set to 1.0. At 1.0, mesh and ring move "
                  "the same")
                a("   bytes through the same cap, and the mesh then always "
                  "wins on step")
                a("   count. Here the ring can only be chosen via the "
                  "window limit.")
            values = "  ".join(
                f"{_kib(g)}: {m.mesh_penalty(gi):.2f}x"
                for gi, g in enumerate(m.sizes)
            )
            a(f"  {values}")
            for h in m.hints:
                a(f"  ! {h}")
        else:
            a("No measurement (planner off or fixed configuration).")

        a("")
        a("-- roles (R2: capacity < leaf_threshold x median -> leaf) --")
        for r in range(self.world):
            extra = ""
            if r in self.parents:
                parent_list = ", ".join(str(x) for x in self.parents[r])
                shares = "/".join(f"{p / 10:.0f}%" for p in self.split.get(r, ()))
                extra = f"  -> parents [{parent_list}]  split {shares}"
            a(f"  Rank {r} ({self.bdfs[r]}): {self.roles[r]}{extra}")
        a(f"  Reduction domain: {list(self.domain)}   Leaves: {list(self.leaves)}")
        a(f"  Ring order (sorted by measured capacity): "
          f"{list(self.ring_order)}")

        a("")
        a("-- algorithm per size class --")
        for stage in self.ladder:
            bounds = (f"from {_kib(stage.min_bytes)}" if stage.max_bytes < 0
                      else f"{_kib(stage.min_bytes)}..{_kib(stage.max_bytes)}")
            predicted = "  ".join(
                f"{k}={v * 1e6:.1f}us" for k, v in sorted(stage.prediction_s.items())
            )
            a(f"  {bounds:>16}: {stage.algorithm:<14} [{predicted}]")
            a(f"                    {stage.reason}")
        a(f"  Chunk: {self.chunk_bytes // 1024} KiB")

        if self.tiers:
            a("")
            a("-- fan-in staging plan (R3) --")
            a("   A fast and a slow source may not write into the same "
              "destination at the")
            a("   same time: the cap is split evenly per source, the fast "
              "one gets pulled")
            a("   down to the slow one's share. More than one wave means "
              "staggered.")
            for dst, waves in sorted(self.tiers.items()):
                w = " | ".join("+".join(str(q) for q in wave) for wave in waves)
                a(f"  Destination {dst} ({self.bdfs[dst]}): {w}")

        a("")
        a("-- effective configuration --")
        for k, v in sorted(self.config_summary.items()):
            a(f"  {k}: {v}")
        a("=" * 78)
        return "\n".join(z)


def _kib(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n // (1024 * 1024)} MiB"
    return f"{n // 1024} KiB"


# ===========================================================================
# The cost model -- this is where R1, R3, and R4 live
# ===========================================================================


def _time_mesh(m: Measurement, gi: int, n_bytes: int, ranks: Sequence[int],
               waves: Mapping[int, tuple[tuple[int, ...], ...]],
               step_s: float) -> float:
    """Direct mesh: reduce-scatter + all-gather, exactly one step each.

    ``ranks`` are REAL ranks, not indices -- this function is also
    called for a subset (the reduction domain), and an
    index-into-the-subset would be a different rank there.

    Byte load per rank same as for the ring: 2N(R-1)/R. But only **2**
    steps instead of 2(R-1) -- that is the entire reason the mesh wins at
    small message sizes.

    R3 lives in the effective rate: ``R-1`` sources write into the same
    destination, and the cap is split **evenly per source**. The
    effective rate of one source is therefore
    ``min(edge capacity, cap / number of simultaneous sources)``. This is
    exactly the term that penalizes the mesh at saturation: the fast edge
    gets squeezed down to its share of the cap, and the excess is wasted.

    The staging plan multiplies the steps: whoever has to write in two
    waves pays for the steps twice.
    """
    r = len(ranks)
    if r < 2:
        return 0.0
    members = set(ranks)
    share = n_bytes / r             # everyone sends N/R to each neighbor
    worst = 0.0
    steps = 2
    for dst in ranks:
        w = tuple(
            tuple(q for q in wave if q in members)
            for wave in waves.get(dst, (tuple(x for x in ranks if x != dst),))
        )
        w = tuple(wave for wave in w if wave)
        cap = m.cap(dst, gi)
        # Waves run SEQUENTIALLY -- that's their whole purpose. So sum
        # them, don't take the maximum: staggered fan-in costs time, and
        # that cost must be visible in the comparison against the ring.
        total = 0.0
        for wave in w:
            concurrent = len(wave)
            per_source = cap / concurrent if concurrent else cap
            duration = 0.0
            for source in wave:
                rate = min(m.capacity(source, dst, gi), per_source)
                duration = max(duration, share / (rate * 1e9) if rate > 0 else float("inf"))
            total += duration
        worst = max(worst, total)
        steps = max(steps, 2 * len(w))
    return steps * step_s + 2 * worst * m.mesh_penalty(gi)


def _time_ring(m: Measurement, gi: int, n_bytes: int, order: Sequence[int],
               step_s: float) -> float:
    """Ring: 2(R-1) steps, but every rank receives from exactly ONE neighbor.

    No fan-in, no unfair split -- the reason the ring wins at saturation.
    The price paid for that is the step count, which for R=8 is fourteen
    instead of two.
    """
    world = len(order)
    if world < 2:
        return 0.0
    share = n_bytes / world
    slowest = 0.0
    for k in range(world):
        src, dst = order[k], order[(k + 1) % world]
        rate = m.capacity(src, dst, gi)
        slowest = max(slowest, share / (rate * 1e9) if rate > 0 else float("inf"))
    return 2 * (world - 1) * (step_s + slowest)


def _time_star(m: Measurement, gi: int, n_bytes: int, hub: int, world: int,
                step_s: float) -> float:
    """Star: everything to the hub, reduce there, distribute back.

    Two strictly separate phases with (R-1)N per direction on the hub. It
    is in the model because it is today's status quo and because
    ``algorithm: star`` must remain forceable -- not because it ever
    wins.
    """
    if world < 2:
        return 0.0
    inbound = m.cap(hub, gi)
    out_rates = [m.capacity(hub, j, gi) for j in range(world) if j != hub]
    outbound = min(out_rates) if out_rates else 0.0
    if inbound <= 0 or outbound <= 0:
        return float("inf")
    forward = (world - 1) * n_bytes / (inbound * 1e9)
    back = (world - 1) * n_bytes / (outbound * 1e9)
    return 2 * step_s + forward + back


def _time_hierarchical(m: Measurement, gi: int, n_bytes: int,
                       domain: Sequence[int], leaves: Sequence[int],
                       parents: Mapping[int, tuple[int, ...]],
                       split: Mapping[int, tuple[int, ...]],
                       inner_s: float, step_s: float, world: int) -> float:
    """Leaf + reduction domain, chunked and overlapped.

    The leaf sends N out and pulls N in -- the floor, regardless of
    algorithm. With chunking the two directions overlap, but **not for
    free**: the credit is the measured duplex factor (R4), not 2.
    """
    if not leaves:
        return inner_s
    worst_leaf = 0.0
    for b in leaves:
        leaf_parents = parents.get(b, tuple(domain))
        weights = split.get(
            b, tuple(1000 // max(len(leaf_parents), 1) for _ in leaf_parents)
        )
        total_weight = sum(weights) or 1
        up = 0.0
        down = 0.0
        for e, g in zip(leaf_parents, weights):
            part = n_bytes * (g / total_weight)
            r_up = m.capacity(b, e, gi)
            r_down = m.capacity(e, b, gi)
            up = max(up, part / (r_up * 1e9) if r_up > 0 else float("inf"))
            down = max(down, part / (r_down * 1e9) if r_down > 0 else float("inf"))
        # Overlap: the duplex factor says how much of the reverse
        # direction is actually free. f=1 -> no overlap (up+down),
        # f=2 -> full overlap (max). The measured value lies in between.
        f = max(1.0, min(2.0, m.duplex_factor(b, gi)))
        serial = up + down
        parallel = max(up, down)
        worst_leaf = max(worst_leaf, serial - (f - 1.0) * (serial - parallel))
    return 2 * step_s + worst_leaf + inner_s


# ===========================================================================
# Derived quantities: roles, domains, ring, staggering, chunk
# ===========================================================================


def _node_capacity(m: Measurement, r: int) -> float:
    """One number per rank for the role comparison (R2).

    Deliberately the **minimum** of outbound and inbound, condensed
    across sizes: a card that sends fast but receives slowly is just as
    unsuited for transit traffic as the reverse -- transit means both. For
    the edge choice itself the directions stay separate; they are
    combined only here, where a single ranking is needed.
    """
    gi = len(m.sizes) - 1        # largest measured size: that's where it separates
    return min(m.outbound[r][gi], m.inbound[r][gi])


def _roles(m: Measurement, k: CollectiveConfig) -> tuple[list[str], list[int], list[int]]:
    world = m.world
    cap = [_node_capacity(m, r) for r in range(world)]
    median_cap = statistics.median(cap) if cap else 0.0
    threshold = k.leaf_threshold * median_cap
    roles = ["domain"] * world
    for r in range(world):
        if world > 2 and cap[r] < threshold:
            roles[r] = "leaf"
    # Manually set roles override the measurement -- deliberately.
    for r in range(world):
        fixed = k.roles.get(m.bdfs[r])
        if fixed is not None:
            roles[r] = "domain" if fixed == "hub" else fixed
    # There must be at least two domain nodes, otherwise there is nothing
    # to reduce into. If necessary, pull back the strongest ones.
    dom = [r for r in range(world) if roles[r] != "leaf"]
    if len(dom) < min(2, world):
        strongest = sorted(range(world), key=lambda r: (-cap[r], r))[: min(2, world)]
        for r in strongest:
            roles[r] = "domain"
    dom = [r for r in range(world) if roles[r] != "leaf"]
    leaves = [r for r in range(world) if roles[r] == "leaf"]
    return roles, dom, leaves


def _domains_from_config(m: Measurement, k: CollectiveConfig) -> Optional[list[list[int]]]:
    if not k.domains:
        return None
    index = {bdf: r for r, bdf in enumerate(m.bdfs)}
    out: list[list[int]] = []
    for group in k.domains:
        ranks = []
        for bdf in group:
            if bdf not in index:
                raise ConfigError(
                    f"collective.domains names {bdf!r}, but this ensemble "
                    f"only has {sorted(index)}. No silent fallback -- "
                    f"either the address is wrong or the card is missing."
                )
            ranks.append(index[bdf])
        out.append(sorted(ranks))
    seen = [r for g in out for r in g]
    if len(seen) != len(set(seen)):
        raise ConfigError("collective.domains: one rank appears in two domains")
    return out


def _parent_and_split(
    m: Measurement, domain: Sequence[int], leaves: Sequence[int], k: CollectiveConfig
) -> tuple[dict[int, tuple[int, ...]], dict[int, tuple[int, ...]]]:
    """Who attaches to what, and in which ratio.

    The split ratio is its own lever: if a leaf concentrates its traffic
    on a single parent, less is left over for the fast cards among
    themselves (in lane units: an even 2+2 leaves B<->C six lanes,
    concentrated 4+0 only four). On a three-card rig this changes nothing
    about the total duration, but with more ranks it very much does --
    hence it's configurable.

    ``auto`` = proportional to the measured edge, BUT capped by R3: where
    the parents' capacities are too far apart, proportionality buys
    nothing, because the fan-in cap splits evenly regardless -- there, it
    is split evenly.
    """
    gi = len(m.sizes) - 1
    parents: dict[int, tuple[int, ...]] = {}
    split: dict[int, tuple[int, ...]] = {}
    mode = k.split
    for b in leaves:
        candidates = sorted(domain, key=lambda d: (-m.capacity(b, d, gi), d))
        if not candidates:
            continue
        parents[b] = tuple(candidates)
        if isinstance(mode, Mapping):
            weights = mode.get(m.bdfs[b])
            if weights is not None:
                if len(weights) != len(candidates):
                    raise ConfigError(
                        f"collective.split[{m.bdfs[b]!r}] names {len(weights)} "
                        f"weights, but the leaf has {len(candidates)} parents "
                        f"{[m.bdfs[d] for d in candidates]}"
                    )
                split[b] = _per_mille(weights)
                continue
            mode_b = "auto"
        else:
            mode_b = mode
        rates = [m.capacity(b, d, gi) for d in candidates]
        if mode_b == "even" or not any(rates):
            split[b] = _per_mille([1.0] * len(candidates))
        elif mode_b == "proportional":
            split[b] = _per_mille(rates)
        else:  # auto
            spread = (max(rates) / min(rates)) if min(rates) > 0 else float("inf")
            split[b] = _per_mille(
                [1.0] * len(candidates) if spread > k.tier_ratio else rates
            )
    return parents, split


def _per_mille(weights: Sequence[float]) -> tuple[int, ...]:
    s = sum(weights)
    if s <= 0:
        n = len(weights)
        return tuple([1000 // n] * (n - 1) + [1000 - (1000 // n) * (n - 1)])
    raw = [g / s * 1000.0 for g in weights]
    whole = [int(x) for x in raw]
    remainder = 1000 - sum(whole)
    # Largest fractional remainders get the leftover -- deterministic, so
    # every rank arrives at the same result.
    order_by_remainder = sorted(range(len(raw)), key=lambda i: (-(raw[i] - whole[i]), i))
    for i in order_by_remainder[:remainder]:
        whole[i] += 1
    return tuple(whole)


def _ring_order(m: Measurement, world: int) -> tuple[int, ...]:
    """Ring ordered by measured capacity.

    The ring wins, among other reasons, because it can lay out
    neighborhoods so that traffic stays local and shared bottlenecks are
    crossed only once per round. The order here is built **from the
    measurements** (a greedy tour over the strongest edges); PCI
    proximity is used only as a tie-break and never as a decision --
    ``lspci`` is a starting estimate, not truth.
    """
    if world < 2:
        return tuple(range(world))
    gi = len(m.sizes) - 1
    start = min(range(world), key=lambda r: (-_node_capacity(m, r), r))
    order = [start]
    open_set = set(range(world)) - {start}
    while open_set:
        last = order[-1]
        next_rank = min(
            open_set,
            key=lambda j: (
                -min(m.capacity(last, j, gi), m.capacity(j, last, gi)),
                -_pci_proximity(m.bdfs[last], m.bdfs[j]),
                j,
            ),
        )
        order.append(next_rank)
        open_set.discard(next_rank)
    return tuple(order)


def _pci_proximity(a: str, b: str) -> int:
    """Tie-break only: length of the shared sysfs path.

    Two cards under the same switch share a longer path than two on
    different root ports. This is a statement about topology, not about
    capacity -- which is why it lives here and nowhere else.
    """
    try:
        pa = os.path.realpath(f"/sys/bus/pci/devices/{a}")
        pb = os.path.realpath(f"/sys/bus/pci/devices/{b}")
    except OSError:
        return 0
    n = 0
    for x, y in zip(pa.split("/"), pb.split("/")):
        if x != y:
            break
        n += 1
    return n


def _tier_plan(
    m: Measurement, world: int, ratio: float
) -> dict[int, tuple[tuple[int, ...], ...]]:
    """R3 as a schedule: who may write into the same destination at the same time.

    Rule taken from the fan-in measurement: the cap is split evenly per
    source, not by capability. An x8 source next to an x4 source dropped
    from 12.81 to 6.75 GB/s, while the x4 source kept its 6.46. Hence:
    sources whose capacities differ by more than ``ratio`` belong in
    different waves.
    """
    gi = len(m.sizes) - 1
    plan: dict[int, tuple[tuple[int, ...], ...]] = {}
    for dst in range(world):
        sources = [q for q in range(world) if q != dst]
        if len(sources) < 2:
            plan[dst] = (tuple(sources),)
            continue
        sources.sort(key=lambda q: (-m.capacity(q, dst, gi), q))
        waves: list[list[int]] = []
        for q in sources:
            rate = m.capacity(q, dst, gi)
            placed = False
            for wave in waves:
                rates = [m.capacity(x, dst, gi) for x in wave] + [rate]
                lo, hi = min(rates), max(rates)
                if lo > 0 and hi / lo <= ratio:
                    wave.append(q)
                    placed = True
                    break
            if not placed:
                waves.append([q])
        plan[dst] = tuple(tuple(w) for w in waves)
    return plan


def _chunk_bytes(m: Measurement, k: CollectiveConfig, world: int) -> int:
    """Chunk size from measured step cost and measured rate.

    Per the collective design, pipelining is probably worth more than the
    topology choice (without chunking, the leaf's outbound and return
    trips run sequentially: 266 instead of 133 us). Chunks too small drown
    in per-step overhead, chunks too large stop overlapping.

    Rule: a chunk should take roughly four times as long to transfer as
    one step costs. Both numbers are measured.
    """
    if k.chunk_kib is not None:
        return k.chunk_kib * 1024
    gi = len(m.sizes) - 1
    rates = [m.capacity(i, j, gi) for i in range(world) for j in range(world) if i != j]
    rate = statistics.median(rates) if rates else 1.0
    step = m.step_s()
    if step <= 0 or rate <= 0:
        return 256 * 1024
    target = 4.0 * step * rate * 1e9
    kib = max(16, min(4096, 2 ** int(round(math.log2(max(target, 1.0) / 1024)))))
    return int(kib) * 1024


# ===========================================================================
# plan_collective(): measurement + configuration -> plan
# ===========================================================================


def _window_requirement(algorithm: str, nbytes: int, world: int) -> int:
    """Same calculation as ``barlink_bar1.window_requirement`` -- kept local here so
    the planner stays usable (and testable) without the BAR1 transport.

    **Counted against the ported kernels**, not estimated: mesh and ring
    BOTH need ``2(R-1)`` slots of ``ceil(N/R)``.

    * Mesh: ``R-1`` for the reduce-scatter and another ``R-1`` for the
      all-gather. Kept separate, because there is no ordering between "I
      read my RS slots" and "the other side writes its AG chunk".
    * Ring: one per step, and there are ``2(R-1)`` steps. Alternating
      between two slots would only work if the sender observed its
      SUCCESSOR's progress -- but it only observes its predecessor.
    * Star: ``R-1`` full buffers on the hub.

    CORRECTION: this used to say ``2*2*share`` for the ring, i.e. four
    slots regardless of R. At ``R=3`` that is the same value
    (``2(R-1) = 4``) and so it never stood out; from ``R=4`` on it was too
    small. An undersized requirement lets an algorithm be selected that
    the mapping cannot actually carry -- and the bug then only shows up
    in the transport.
    """
    if world < 2:
        return 0
    share = -(-nbytes // world)
    if algorithm in ("mesh", "ring", "hierarchical"):
        return 2 * (world - 1) * share
    if algorithm == "star":
        return 2 * (world - 1) * nbytes
    return 0


def plan_collective(m: Measurement, k: BarlinkConfig, source: str = "measured",
          window_bytes: Optional[int] = None) -> Plan:
    """Purely computational, with no I/O at all -- so it's testable without hardware.

    Exclusively a function of ``(m, k, window_bytes)``. Two ranks with
    the same inputs are guaranteed to get the same plan; that is exactly
    what ``BarlinkMatrixPlanner`` checks afterwards via the checksum.

    ``window_bytes`` is how much BAR1 can be mapped simultaneously per
    destination -- i.e. a **capability**, not a policy. If it is known,
    any algorithm whose window requirement exceeds it drops out. This is
    the second reason the ring can win: it needs two slots of ``N/R``,
    the mesh needs ``R-1``. ``None`` means "unknown" and rules nothing
    out -- not "unlimited".
    """
    c = k.collective
    world = m.world
    roles, dom, leaves = _roles(m, c)

    fixed_domains = _domains_from_config(m, c)
    if fixed_domains is not None:
        # Manually set domains: the first entry is the reduction domain,
        # everything not named becomes a leaf.
        named = {r for g in fixed_domains for r in g}
        dom = sorted(fixed_domains[0])
        leaves = [r for r in range(world) if r not in dom]
        roles = ["domain" if r in dom else "leaf" for r in range(world)]
        for r in range(world):
            if r not in named and c.roles.get(m.bdfs[r]) is None:
                roles[r] = "leaf"

    parents, split = _parent_and_split(m, dom, leaves, c)
    order = _ring_order(m, world)
    tiers = _tier_plan(m, world, c.tier_ratio)
    chunk = _chunk_bytes(m, c, world)
    step = m.step_s()
    hub = order[0] if order else 0

    stages: list[Stage] = []
    for gi, g in enumerate(m.sizes):
        prediction: dict[str, float] = {}
        prediction["mesh"] = _time_mesh(m, gi, g, range(world), tiers, step)
        prediction["ring"] = _time_ring(m, gi, g, order, step)
        prediction["star"] = _time_star(m, gi, g, hub, world, step)
        if leaves and len(dom) >= 2:
            # Within the domain, whatever wins there wins.
            inner_mesh = _time_mesh(m, gi, g, dom, tiers, step)
            inner_ring = _time_ring(
                m, gi, g, [d for d in order if d in dom] or dom, step
            )
            prediction["hierarchical"] = _time_hierarchical(
                m, gi, g, dom, leaves, parents, split,
                min(inner_mesh, inner_ring), step, world,
            )

        # Capability limit: whatever doesn't fit the window isn't in the
        # running. Deliberately BEFORE policy -- a configuration must not
        # be able to force an algorithm the hardware cannot map.
        too_large = {
            a: _window_requirement(a, g, world) for a in list(prediction)
            if window_bytes is not None
            and _window_requirement(a, g, world) > window_bytes
        }
        feasible = {a: v for a, v in prediction.items() if a not in too_large}

        if c.algorithm != "auto":
            chosen = c.algorithm
            if chosen in too_large:
                raise ConfigError(
                    f"collective.algorithm={chosen} forced, but at "
                    f"{_kib(g)} and {world} ranks it needs "
                    f"{too_large[chosen] // 1024} KiB of window; mappable "
                    f"is {window_bytes // 1024} KiB. No silent fallback -- "
                    f"either chunk smaller or choose a different algorithm."
                )
            reason = f"forced via collective.algorithm={c.algorithm}"
        else:
            if not feasible:
                raise ConfigError(
                    f"At {_kib(g)} and {world} ranks NO algorithm fits into "
                    f"the mappable window of {window_bytes // 1024} KiB "
                    f"(requirement: "
                    f"{ {a: b // 1024 for a, b in too_large.items()} } KiB). "
                    f"This is a startup error, not a quiet reroute: chunk "
                    f"smaller, or exclude the direct path for this size."
                )
            chosen = min(feasible, key=lambda a: (feasible[a], a))
            reason = _reason(m, gi, g, world, chosen, prediction, tiers,
                             c.saturation_share, leaves)
            if too_large:
                reason += (
                    "  Excluded for being too large for the window: "
                    + ", ".join(
                        f"{a} ({b // 1024} KiB > {window_bytes // 1024} KiB)"
                        for a, b in sorted(too_large.items())
                    ) + "."
                )
        stages.append(
            Stage(
                min_bytes=g,
                max_bytes=g if gi + 1 < len(m.sizes) else -1,
                algorithm=chosen,
                prediction_s={a: round(v, 9) for a, v in prediction.items()},
                reason=reason,
            )
        )

    stages = _smooth_ladder(stages)

    summary = {
        "planner": c.planner,
        "algorithm": c.algorithm,
        "chunk_kib": "auto" if c.chunk_kib is None else c.chunk_kib,
        "leaf_threshold": c.leaf_threshold,
        "split": c.split if isinstance(c.split, str) else "manual",
        "tier_ratio": c.tier_ratio,
        "roles (manual)": dict(c.roles) or "-",
        "domains (manual)": [list(g) for g in c.domains] or "-",
        "nic": k.nic,
        "measure.sizes_kib": list(c.measure.sizes_kib),
        "measure.budget_ms": c.measure.budget_ms,
    }
    return Plan(
        world=world,
        bdfs=m.bdfs,
        roles=tuple(roles),
        domain=tuple(dom),
        leaves=tuple(leaves),
        parents={k2: v for k2, v in sorted(parents.items())},
        split={k2: v for k2, v in sorted(split.items())},
        ring_order=order,
        ladder=tuple(stages),
        chunk_bytes=chunk,
        tiers=tiers,
        config_summary=summary,
        measurement=m,
        source=source,
    )


def _reason(m: Measurement, gi: int, g: int, world: int, chosen: str,
           prediction: Mapping[str, float],
           tiers: Mapping[int, tuple[tuple[int, ...], ...]],
           saturation_share: float, leaves: Sequence[int]) -> str:
    """One sentence saying why. Without it the choice isn't traceable."""
    mesh, ring = prediction.get("mesh", 0.0), prediction.get("ring", 0.0)
    share = g / world
    # Load the mesh would create on the most heavily addressed
    # destination: what the R-1 sources together want to deliver, against
    # that destination's measured cap.
    load = 0.0
    for dst in range(world):
        cap = m.cap(dst, gi)
        sources = [m.capacity(q, dst, gi) for q in range(world) if q != dst]
        if cap > 0 and sources:
            load = max(load, sum(sources) / cap)
    saturated = load > 1.0 / max(saturation_share, 1e-9)
    staggered = sum(1 for w in tiers.values() if len(w) > 1)
    t = []
    if chosen == "mesh":
        if saturated:
            t.append(
                f"Mesh despite saturation (demand/cap {load:.2f}): the step "
                f"count outweighs it. 2 steps instead of {2 * (world - 1)}, "
                f"and one step costs {m.step_s() * 1e6:.1f} us here."
            )
        else:
            t.append(
                f"Mesh: 2 steps instead of {2 * (world - 1)}; demand/cap "
                f"{load:.2f} is below saturation -- concurrency is nearly "
                f"free there (measured ratio 0.99 at 20 KiB)."
            )
        if staggered:
            t.append(f"{staggered} destination(s) need to be staggered for this.")
    elif chosen == "ring":
        t.append(
            f"Ring: demand/cap {load:.2f} (threshold "
            f"{1.0 / saturation_share:.2f}) -- at saturation, the mesh's "
            f"concurrency buys nothing (measured 1.03x at 1 MiB) and the "
            f"cap is split evenly instead of proportionally, wasting the "
            f"fast edges. In the ring, every rank receives from exactly "
            f"one neighbor."
        )
        if staggered:
            t.append(f"{staggered} destination(s) would have needed staggering in the mesh.")
    elif chosen == "hierarchical":
        t.append(
            f"Hierarchical: {len(leaves)} leaf/leaves carry no transit "
            f"traffic, sending only their own contribution and receiving "
            f"only the result; the domain reduces among itself."
        )
    elif chosen == "star":
        t.append("Star: chosen only because everything else predicts worse, "
                 "or because it was forced.")
    t.append(f"(mesh {mesh * 1e6:.1f} us / ring {ring * 1e6:.1f} us, "
             f"share per edge {share / 1024:.0f} KiB)")
    return " ".join(t)


def _smooth_ladder(stages: list[Stage]) -> list[Stage]:
    """Merge consecutive stages that use the same algorithm.

    A ladder with three entries of the same algorithm isn't a ladder, it's
    a single line -- and it reads that way in the explanation too.
    """
    out: list[Stage] = []
    for s in stages:
        if out and out[-1].algorithm == s.algorithm:
            # min_bytes stays as-is: the stage then spans from the
            # smallest to the largest size it applies to.
            out[-1] = replace(out[-1], max_bytes=s.max_bytes,
                              prediction_s=s.prediction_s, reason=s.reason)
        else:
            out.append(s)
    if out:
        out[-1] = replace(out[-1], max_bytes=-1)
    return out


# ===========================================================================
# Cache
# ===========================================================================


def _default_cache() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "sglang", "barlink_matrix.json")


def fingerprint(bdfs: Sequence[str], names: Sequence[str],
                  k: BarlinkConfig) -> str:
    """Card list, PCI addresses, driver version, patch state, measurement parameters.

    If any of these change, the matrix is measured again. The patch state
    is deliberately part of it: without the regkey the direct path
    doesn't carry, and a matrix measured with it becomes wrong afterwards.
    """
    m = k.collective.measure
    parts = {
        "version": PLANNER_VERSION,
        "bdfs": list(bdfs),
        "names": list(names),
        "driver": _driver_version(),
        "patch": _patch_state(),
        "sizes_kib": list(m.sizes_kib),
        "repeats": m.repeats,
        "fanin": m.fanin,
        "duplex": m.duplex,
    }
    raw = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _driver_version() -> str:
    try:
        with open("/proc/driver/nvidia/version") as f:
            return f.read().strip().split("\n")[0]
    except OSError:
        return "unknown"


def _patch_state() -> str:
    """Driver regkeys, as far as visible.

    ``RegistryDwords`` is where ``BarlinkPeerBar1`` and
    ``RMPcieP2PType`` get set. If nothing is there, the direct path isn't
    unlocked -- a matrix measured with it must not be reused for a run
    with the patch, and vice versa.
    """
    try:
        with open("/proc/driver/nvidia/params") as f:
            lines = [
                z.strip() for z in f
                if z.startswith(("RegistryDwords", "EnableResizableBar"))
            ]
        return "|".join(lines)
    except OSError:
        return "unknown"


def read_cache(path: str, fp: str) -> Optional[Measurement]:
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if d.get("fingerprint") != fp:
        return None
    try:
        return Measurement.from_dict(d["measurement"])
    except (KeyError, TypeError, ValueError) as e:
        logger.warning("barlink-Matrix: cache %s unreadable (%s); measuring again.",
                       path, e)
        return None


def write_cache(path: str, fp: str, m: Measurement) -> None:
    try:
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(
            {"fingerprint": fp, "measurement": m.as_dict()},
            sort_keys=True, indent=1,
        ))
        tmp.replace(p)
    except OSError as e:
        logger.warning("barlink-Matrix: cache %s not writable (%s).",
                       path, e)


# ===========================================================================
# The planner: what happens at startup
# ===========================================================================


class BarlinkMatrixPlanner:
    """Measures, plans, checks rank uniformity, explains.

    Usage::

        planner = BarlinkMatrixPlanner(cpu_group, device, config=load_config())
        plan = planner.plan()            # measures on the first call
        logger.info("%s", plan.explanation())

    The plan is **frozen** afterwards. This is not a convenience, it's a
    requirement: decode runs inside captured CUDA graphs, and the choice
    must be fixed at capture time. Switching per message would force a
    re-capture. Load-dependent dynamism is only allowed outside captured
    regions, and then via multiple captured variants, not via a change to
    this plan.
    """

    def __init__(self, cpu_group, device, config: Optional[BarlinkConfig] = None,
                 sensor: Optional[Sensor] = None,
                 window_bytes: Optional[int] = None):
        import torch.distributed as dist

        self.cpu_group = cpu_group
        self.device = device
        self.config = config if config is not None else load_config()
        self.rank = dist.get_rank(cpu_group)
        self.world = dist.get_world_size(cpu_group)
        self._sensor = sensor
        # Capability, not policy: how much BAR1 can be mapped
        # simultaneously per destination. None = unknown (rules nothing
        # out).
        self.window_bytes = window_bytes
        self._plan: Optional[Plan] = None

    # -- public -----------------------------------------------------------

    def plan(self) -> Plan:
        if self._plan is None:
            self._plan = self._build()
        return self._plan

    # -- internal -----------------------------------------------------------

    def _build(self) -> Plan:
        import torch.distributed as dist

        c = self.config.collective
        bdfs, names = self._cards()
        fp = fingerprint(bdfs, names, self.config)

        if c.planner == "off":
            m = self._synthetic_measurement(bdfs, names)
            p = plan_collective(m, self.config, source="fixed",
                      window_bytes=self.window_bytes)
            self._check_uniform(p)
            return p

        path = c.measure.cache or _default_cache()
        m = None
        if not c.measure.cache_off:
            # Only rank 0 reads and distributes -- otherwise two ranks
            # could see differently stale files and plan differently.
            carrier: list[Any] = [None]
            if self.rank == 0:
                found = read_cache(path, fp)
                carrier = [found.as_dict() if found is not None else None]
            dist.broadcast_object_list(
                carrier, src=dist.get_global_rank(self.cpu_group, 0),
                group=self.cpu_group,
            )
            if carrier[0] is not None:
                m = Measurement.from_dict(carrier[0])
                logger.info(
                    "barlink-Matrix: startup measurement skipped, cache %s "
                    "matches (fingerprint %s). Force a re-measure with "
                    "SGLANG_BARLINK_MATRIX_CACHE_OFF=1.", path, fp,
                )

        source = "cached"
        if m is None:
            if c.planner == "fixed":
                # 'fixed' deliberately means: don't measure, use the stored
                # result. If it's missing, that's a named error -- quietly
                # measuring anyway would be exactly the silent fallback
                # this design forbids.
                raise ConfigError(
                    f"collective.planner=fixed, but there is no result under "
                    f"{path!r} with fingerprint {fp} (card list, PCI "
                    f"addresses, driver version, patch state, measurement "
                    f"parameters). Either measure once with planner=auto, or "
                    f"point the path via SGLANG_BARLINK_MATRIX_CACHE at a "
                    f"valid result."
                )
            source = "measured"
            m = self._measure(bdfs, names)
            if not c.measure.cache_off and self.rank == 0:
                write_cache(path, fp, m)

        p = plan_collective(m, self.config, source=source,
                  window_bytes=self.window_bytes)
        self._check_uniform(p)
        return p

    # -- cards --------------------------------------------------------------

    def _cards(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        import torch
        import torch.distributed as dist

        bdf = bdf_of_card(self.device)
        try:
            name = torch.cuda.get_device_name(self.device)
        except Exception:                       # pragma: no cover
            name = "unknown"
        gathered: list[Any] = [None] * self.world
        dist.all_gather_object(gathered, (bdf, name), group=self.cpu_group)
        bdfs = tuple(str(x[0]) for x in gathered)   # type: ignore[index]
        names = tuple(str(x[1]) for x in gathered)  # type: ignore[index]
        if len(set(bdfs)) != len(bdfs):
            logger.warning(
                "barlink-Matrix: duplicate PCI addresses %s. Roles and "
                "domains are addressed via the PCI address; with "
                "duplicates a configuration cannot be applied "
                "unambiguously.", bdfs,
            )
        return bdfs, names

    # -- measurement ------------------------------------------------------------

    def _measure(self, bdfs, names) -> Measurement:
        import torch.distributed as dist

        c = self.config.collective
        mk = self._adjust_budget(c.measure)
        sizes = tuple(g * 1024 for g in mk.sizes_kib)
        f = self._sensor if self._sensor is not None else SelfLoadSensor(
            self.device, max_bytes=max(sizes),
            repeats=mk.repeats,
        )
        t_start = time.perf_counter()
        m = Measurement(
            world=self.world, sizes=sizes, bdfs=bdfs, names=names,
            sensor=f.name(),
        )

        # -- Phase 1: self-load, staggered. --------------------------------
        # Sequential, not simultaneous: otherwise R cards compete for the
        # same host memory and the numbers end up measuring each other.
        own_out: list[float] = []
        own_in: list[float] = []
        own_duplex: list[float] = []
        for owner in range(self.world):
            dist.barrier(group=self.cpu_group)
            if owner == self.rank:
                for g in sizes:
                    own_out.append(_quant(f.self_load(g, "d2h")))
                    own_in.append(_quant(f.self_load(g, "h2d")))
                if mk.duplex:
                    for g in sizes:
                        d = f.self_load_duplex(g)
                        own_duplex.append(_quant(d) if d is not None else 0.0)
            dist.barrier(group=self.cpu_group)

        gathered: list[Any] = [None] * self.world
        dist.all_gather_object(
            gathered, (own_out, own_in, own_duplex), group=self.cpu_group
        )
        for r, (a, e, d) in enumerate(gathered):    # type: ignore[misc]
            m.outbound[r] = list(a)
            m.inbound[r] = list(e)
            if d and any(x > 0 for x in d):
                m.duplex_total[r] = list(d)
            m.latency_s[r] = _latency_fit(sizes, list(a))

        # -- Phase 2: real edge measurement, if a pair sensor is present. ----
        has_pair = self._has_pair_sensor(f, sizes)
        if has_pair:
            for src in range(self.world):
                for dst in range(self.world):
                    if src == dst:
                        continue
                    dist.barrier(group=self.cpu_group)
                    values: list[float] = []
                    if self.rank == src:
                        for g in sizes:
                            r = f.pair(dst, g)
                            values.append(_quant(r) if r is not None else 0.0)
                    elif self.rank == dst:
                        for g in sizes:
                            f.pair_receive(src, g)
                    dist.barrier(group=self.cpu_group)
                    carrier: list[Any] = [values if self.rank == src else None]
                    dist.broadcast_object_list(
                        carrier, src=dist.get_global_rank(self.cpu_group, src),
                        group=self.cpu_group,
                    )
                    if carrier[0]:
                        m.edge[(src, dst)] = list(carrier[0])

            # -- Phase 3: fan-in (R3). All sources into the destination at once. --
            if mk.fanin:
                for dst in range(self.world):
                    dist.barrier(group=self.cpu_group)
                    values = []
                    if self.rank != dst:
                        for g in sizes:
                            r = f.pair(dst, g)
                            values.append(_quant(r) if r is not None else 0.0)
                    else:
                        for g in sizes:
                            f.pair_receive(-1, g)
                    dist.barrier(group=self.cpu_group)
                    rows: list[Any] = [None] * self.world
                    dist.all_gather_object(rows, values, group=self.cpu_group)
                    cap = []
                    for gi in range(len(sizes)):
                        s = sum(v[gi] for r, v in enumerate(rows) if r != dst and v)
                        cap.append(_quant(s))
                    m.fanin_cap[dst] = cap
                    for r, v in enumerate(rows):
                        if r != dst and v:
                            m.fanin_shares[(r, dst)] = list(v)
                m.hints.append(
                    "Fan-in: the sources start together at a barrier, but "
                    "then run the size ladder independently afterwards. "
                    "With strongly mismatched cards, the sizes don't fully "
                    "overlap; the measured cap is therefore more of a "
                    "LOWER bound on concurrency. An interleaved loop behind "
                    "a shared start gate (as in "
                    "sonden/nebenlauf_probe.cu) would be more accurate and "
                    "would need a sensor that interleaves both strands itself."
                )
            # -- Phase 4: shared bottlenecks. ALL pairs simultaneously. ------
            # The only way to measure the difference between "mesh" and
            # "ring" instead of just asserting it: in the mesh all pairs
            # talk simultaneously, in the ring only the neighbors. Whatever
            # is lost to switch uplinks and NUMA hops along the way shows
            # up in no per-edge capacity -- those were each measured on
            # their own edge, INDIVIDUALLY.
            #
            # Methodology as in sonden/nebenlauf_probe.cu: the same strand
            # twice, alone and together, and what's compared is RATES, not
            # wall-clock times. Comparing wall clock against a single
            # transfer would be off by the repetition count.
            #
            # Each round, rank i writes to rank (i+offset) mod R. This is
            # a perfect pairing: everyone sends exactly once and receives
            # exactly once, so there is no fan-in contention -- what's
            # measured here is exclusively the shared bottleneck, not the
            # cap.
            concurrent_rates: dict[tuple[int, int], list[float]] = {}
            for offset in range(1, self.world):
                dst = (self.rank + offset) % self.world
                dist.barrier(group=self.cpu_group)
                mine: list[float] = []
                for g in sizes:
                    r = f.pair(dst, g)
                    mine.append(_quant(r) if r is not None else 0.0)
                gathered2: list[Any] = [None] * self.world
                dist.all_gather_object(gathered2, mine, group=self.cpu_group)
                for src, values2 in enumerate(gathered2):
                    if values2:
                        concurrent_rates[(src, (src + offset) % self.world)] = list(values2)

            factors = []
            for gi in range(len(sizes)):
                # Worst-case degradation across all edges: the mesh step
                # is only as fast as its slowest edge.
                worst = 1.0
                for (src, dst), rates in concurrent_rates.items():
                    alone = m.capacity(src, dst, gi)
                    together = rates[gi]
                    if alone > 0 and together > 0:
                        worst = max(worst, alone / together)
                factors.append(_quant(worst))
            m.mesh_factor = [max(1.0, x) for x in factors]
            m.mesh_factor_measured = True
        else:
            m.hints.append(
                "No pair sensor: edge capacities are estimated from "
                "self-load (min(outbound, inbound)) and are therefore an "
                "UPPER BOUND -- shared bottlenecks such as a switch uplink "
                "or a second root complex are not visible in them. The "
                "fan-in cap is the inbound rate, not a measured BAR1 cap."
            )
            m.mesh_factor = [1.0] * len(sizes)
            m.mesh_factor_measured = False
            m.hints.append(
                "SHARED BOTTLENECKS UNMEASURED (mesh_factor set to 1.0). "
                "Without a pair sensor there is no way to measure how much "
                "an edge loses when ALL pairs talk at once instead of just "
                "one. This is exactly what the expectation that the ring "
                "wins at saturation depends on: with plain edge capacities "
                "and fan-in caps, mesh and ring move the same bytes through "
                "the same cap, and the mesh then ALWAYS wins on step count. "
                "As long as this factor is 1.0, the planner consistently "
                "never picks the ring -- that is a missing measurement, not "
                "a result. Overridable with SGLANG_BARLINK_MESH_FACTOR."
            )

        over = os.environ.get(_ENV_PREFIX + "MESH_FACTOR")
        if over:
            values = [float(x) for x in over.replace(";", ",").split(",")]
            if len(values) == 1:
                values = values * len(sizes)
            if len(values) != len(sizes):
                raise ConfigError(
                    f"SGLANG_BARLINK_MESH_FACTOR={over!r}: expected one "
                    f"value or {len(sizes)} values (per size "
                    f"{[g // 1024 for g in sizes]} KiB)."
                )
            m.mesh_factor = [max(1.0, x) for x in values]
            m.mesh_factor_measured = False
            m.hints.append(
                f"mesh_factor manually set to {m.mesh_factor} "
                f"(SGLANG_BARLINK_MESH_FACTOR) -- not measured."
            )

        m.duration_s = time.perf_counter() - t_start
        self._plausible(m)
        return m

    def _has_pair_sensor(self, f: Sensor, sizes) -> bool:
        """Decide group-wide, not per rank.

        A rank that believes it's measuring while the others are waiting
        at the barrier wedges the startup. Hence: ask everyone, and only
        measure if ALL of them can.
        """
        import torch.distributed as dist

        try:
            can = f.pair(-1, sizes[0]) is not None
        except NotImplementedError:
            can = False
        except Exception as e:
            logger.warning("barlink-Matrix: pair sensor reports an error (%s); "
                           "using the self-load estimate.", e)
            can = False
        rows: list[Any] = [None] * self.world
        dist.all_gather_object(rows, bool(can), group=self.cpu_group)
        return all(bool(x) for x in rows)

    def _adjust_budget(self, mk: MeasureConfig) -> MeasureConfig:
        """Rough pre-estimate against ``budget_ms``.

        Cost grows with R^2 once a pair sensor is present. Rather than
        letting startup time explode, the size ladder is thinned first
        (the middle size goes first -- it discriminates the least), then
        the repetition count. Whatever was thinned out is logged; there
        is no such thing as quietly measuring on thinner evidence.
        """
        pairs = self.world * (self.world - 1)
        # Round count per phase:
        #   1 self-load, staggered           -> R
        #   2 edges individually (pair sensor) -> R(R-1)
        #   3 fan-in, one destination per round -> R
        #   4 all pairs simultaneously        -> R-1
        # Without a pair sensor, 2 through 4 are skipped.
        with_pair = self._sensor is not None
        rounds = self.world + (pairs + self.world + self.world - 1 if with_pair else 0)

        def estimate(g_kib: Sequence[int], reps: int) -> float:
            # ~6 GB/s as a rough assumption for the pre-estimate; it only
            # decides how much gets measured, never what comes out of it.
            transfer = sum((g * 1024) * reps / 6e9 * 1000 for g in g_kib)
            # 2 barriers per round, gloo roughly 0.3 ms.
            return rounds * (transfer + 0.6 * len(g_kib))

        g = list(mk.sizes_kib)
        reps = mk.repeats
        dropped = []
        while estimate(g, reps) > mk.budget_ms and len(g) > 2:
            removed = g.pop(len(g) // 2)
            dropped.append(f"{removed} KiB")
        while estimate(g, reps) > mk.budget_ms and reps > 4:
            reps //= 2
        if dropped or reps != mk.repeats:
            logger.info(
                "barlink-Matrix: measurement budget %.0f ms -- sizes %s "
                "removed, repetitions %d -> %d. Measure fully with "
                "SGLANG_BARLINK_MEASURE_BUDGET_MS=<more>.",
                mk.budget_ms, dropped or "none", mk.repeats, reps,
            )
        return replace(mk, sizes_kib=tuple(g), repeats=reps)

    def _plausible(self, m: Measurement) -> None:
        """Reports deviations from this rig's reference values.

        No abort and no decision -- just a note, so that a broken
        measurement setup stands out before it passes as a rig quirk.
        Fourteen plausible assumptions have already failed against the
        hardware in this project; a silent measurement would be the
        fifteenth.
        """
        gi = len(m.sizes) - 1
        for r in range(m.world):
            if m.outbound[r][gi] <= 0.05 or m.inbound[r][gi] <= 0.05:
                m.hints.append(
                    f"Rank {r} ({m.bdfs[r]}): measured rate near zero "
                    f"(outbound={m.outbound[r][gi]:.2f}, "
                    f"inbound={m.inbound[r][gi]:.2f}) -- "
                    f"this is a measurement error, not a property of the "
                    f"card."
                )
            if m.duplex_total:
                f = m.duplex_factor(r, gi)
                if f > 1.9:
                    m.hints.append(
                        f"Rank {r}: duplex factor {f:.2f} at the largest "
                        f"size. Measured on this rig was "
                        f"{_REFERENCE_DUPLEX_TOTAL_1MIB:.2f}; a value close to "
                        f"2 suggests the streams weren't really running "
                        f"simultaneously."
                    )
        if m.fanin_cap:
            for j, d in m.fanin_cap.items():
                if d[gi] > 2.0 * m.inbound[j][gi] and m.inbound[j][gi] > 0:
                    m.hints.append(
                        f"Destination {j}: measured fan-in cap "
                        f"{d[gi]:.2f} GB/s lies far above the inbound rate "
                        f"{m.inbound[j][gi]:.2f} GB/s -- check whether the "
                        f"sources really ran simultaneously."
                    )

    def _synthetic_measurement(self, bdfs, names) -> Measurement:
        """``planner: off`` -- measure nothing, take everything from the configuration.

        Evenly-distributed placeholder rates, so that roles come
        exclusively from ``roles``/``domains``. The plan reports
        ``source=fixed``, so the explanation doesn't give the impression
        that anything was actually measured here.
        """
        sizes = tuple(g * 1024 for g in self.config.collective.measure.sizes_kib)
        m = Measurement(world=self.world, sizes=sizes, bdfs=bdfs, names=names,
                    sensor="none (planner=off)")
        for r in range(self.world):
            m.outbound[r] = [1.0] * len(sizes)
            m.inbound[r] = [1.0] * len(sizes)
            m.latency_s[r] = 0.0
        m.hints.append(
            "planner=off: nothing measured. Roles, domains, and algorithm "
            "come exclusively from the configuration."
        )
        return m

    def _check_uniform(self, p: Plan) -> None:
        """The plan MUST be the same on every rank."""
        import torch.distributed as dist

        sums: list[Any] = [None] * self.world
        dist.all_gather_object(sums, p.checksum(), group=self.cpu_group)
        if len(set(sums)) != 1:
            divergent = {r: s for r, s in enumerate(sums) if s != sums[0]}
            raise RuntimeError(
                f"barlink-Matrix: ranks ended up with DIFFERENT plans "
                f"({sums}). Divergent ranks: {divergent}. This is a "
                f"startup error, not a warning -- the collectives assume "
                f"that every rank runs the same decomposition. Most common "
                f"causes: an SGLANG_BARLINK_* variable or a config file "
                f"isn't the same on every rank, or window_bytes was "
                f"passed in differently per rank (it must be the minimum "
                f"across all destinations -- what matters is what can be "
                f"mapped EVERYWHERE)."
            )


def _latency_fit(sizes: Sequence[int], rates: Sequence[float]) -> float:
    """t(N) = latency + N/rate, fitted across the size ladder.

    The measured rates per size yield times; the line fitted through them
    separates startup cost from throughput. The startup cost is what a
    collective step costs at minimum -- and at 20 KiB the acknowledgment
    round-trip, at ~3 us of 13.47 us, was the single largest line item of
    the collective, which is exactly the size range this is about.

    CAVEAT: what's measured here is the copy engine's startup cost, not
    that of the collective's acknowledgment loop. This is a LOWER BOUND.
    Anyone who knows better sets ``SGLANG_BARLINK_STEP_US``.
    """
    over = os.environ.get(_ENV_PREFIX + "STEP_US")
    if over:
        return float(over) / 1e6
    points = [
        (float(n), n / (r * 1e9)) for n, r in zip(sizes, rates) if r > 0
    ]
    if len(points) < 2:
        return 0.0
    n_mean = sum(x for x, _ in points) / len(points)
    t_mean = sum(y for _, y in points) / len(points)
    numerator = sum((x - n_mean) * (y - t_mean) for x, y in points)
    denominator = sum((x - n_mean) ** 2 for x, _ in points)
    if denominator <= 0:
        return 0.0
    slope = numerator / denominator
    intercept = t_mean - slope * n_mean
    return max(0.0, intercept)


# ===========================================================================
# Utilities
# ===========================================================================


def bdf_of_card(device) -> str:
    """PCI address ``0000:05:00.0`` of the card behind ``device``.

    Via ``cudaDeviceGetPCIBusId``, because that's the only mapping that
    stays correct even with ``CUDA_VISIBLE_DEVICES`` set -- the ordinal
    alone is meaningless across process boundaries, and the configuration
    addresses cards by their PCI address.
    """
    import torch

    ordinal = device.index if hasattr(device, "index") else int(device)
    if ordinal is None:
        ordinal = torch.cuda.current_device()
    # First without ctypes, if torch offers the fields. WARNING:
    # `props.pci_bus_id` is the BUS NUMBER as an int (from
    # cudaDeviceProp.pciBusID), NOT the address string -- calling str() on
    # it gives "10", i.e. a sysfs path that doesn't exist. The address is
    # only formed from domain, bus, and slot; cudaDeviceProp carries no
    # function number, GPUs always sit on .0.
    try:
        props = torch.cuda.get_device_properties(ordinal)
        dom = getattr(props, "pci_domain_id", None)
        bus = getattr(props, "pci_bus_id", None)
        slot = getattr(props, "pci_device_id", None)
        if isinstance(bus, str) and _IS_BDF.match(bus.strip()):
            return _norm_bdf(bus)
        if None not in (dom, bus, slot):
            return _norm_bdf(f"{int(dom):04x}:{int(bus):02x}:{int(slot):02x}.0")
    except Exception:
        pass
    try:
        import ctypes

        lib = ctypes.CDLL(
            "libamdhip64.so" if torch.version.hip is not None else "libcudart.so"
        )
        fn = (lib.hipDeviceGetPCIBusId if torch.version.hip is not None
              else lib.cudaDeviceGetPCIBusId)
        buffer = ctypes.create_string_buffer(32)
        fn.restype = ctypes.c_int
        if fn(buffer, ctypes.c_int(32), ctypes.c_int(int(ordinal))) == 0:
            return _norm_bdf(buffer.value.decode())
    except Exception as e:                      # pragma: no cover
        logger.warning("barlink-Matrix: could not determine PCI address (%s).", e)
    return f"unknown-{ordinal}"


def link_width(bdf: str) -> Optional[tuple[int, str]]:
    """``(width, speed)`` from sysfs -- ONLY as a starting estimate.

    This value feeds into NO decision. It's shown in the explanation next
    to the measurement, so that it stands out if the two disagree: if
    sysfs says x8 and the measurement says 6 GB/s, either the card is
    downclocked or the measurement setup is broken -- and this exact
    question has repeatedly been the deciding factor in this project.
    """
    try:
        base = f"/sys/bus/pci/devices/{bdf}"
        with open(f"{base}/current_link_width") as f:
            width = int(f.read().strip())
        with open(f"{base}/current_link_speed") as f:
            speed = f.read().strip()
        return width, speed
    except (OSError, ValueError):
        return None
