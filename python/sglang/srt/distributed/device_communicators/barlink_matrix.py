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
    ``blatt_schwelle`` (default 0.6 x median).

R3  **Fan-in cap.** No card receives more than its measured cap,
    regardless of the number of sources (this rig: ~13 GB/s). The cap is
    split **evenly per source**, not proportionally -- the x8 source
    dropped from 12.81 to 6.75 GB/s, the x4 source kept its 6.46. Hence:
    never let a fast and a slow card write into the same destination at
    the same time; stagger them instead (``Plan.staffeln``).

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
PLANER_VERSION = 1

# Default size ladder for the measurement, in KiB. The three points are
# the operating regimes that matter: decode (20), transition (80),
# prefill (1024).
VORGABE_GROESSEN_KIB = (20, 80, 1024)

ALGORITHMEN = ("mesh", "ring", "star", "hierarchisch")
PLANER_MODI = ("auto", "fest", "aus")
NIC_MODI = ("nie", "bei_bedarf", "immer")
ROLLEN = ("blatt", "domaene", "nabe")

# ---------------------------------------------------------------------------
# Reference values for this rig -- ONLY for the plausibility report.
#
# They feed into NO decision. If a startup measurement deviates sharply,
# that is logged, so a broken measurement setup stands out instead of
# passing as a rig quirk. Evidence: MESSUNG_NEBENLAEUFIGKEIT.md.
# ---------------------------------------------------------------------------
_BELEG_FANIN_GBPS = 13.16          # cap at 1 MiB, two sources
_BELEG_DUPLEX_SUMME_1MIB = 1.32    # sum of both directions / one direction
_BELEG_DUPLEX_SUMME_20KIB = 1.47


# ===========================================================================
# Configuration
# ===========================================================================


class ConfigError(ValueError):
    """A named configuration error -- never silently healed."""


@dataclass(frozen=True)
class MeasureConfig:
    """What the planner measures at startup, and how long it's allowed to take."""

    groessen_kib: tuple[int, ...] = VORGABE_GROESSEN_KIB
    wiederholungen: int = 32
    vorlauf: int = 8
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
    cache_aus: bool = False


@dataclass(frozen=True)
class CollectiveConfig:
    planer: str = "auto"                 # auto | fest | aus  (auto | fixed | off)
    algorithmus: str = "auto"            # auto | mesh | ring | star | hierarchisch (hierarchical)
    chunk_kib: Optional[int] = None      # None == "auto"
    blatt_schwelle: float = 0.6          # capacity < 0.6 x median -> leaf
    aufteilung: Any = "auto"             # auto | gleich | proportional | {bdf: [...]}  (auto | even | proportional)
    roles: Mapping[str, str] = field(default_factory=dict)      # bdf -> role
    domaenen: tuple[tuple[str, ...], ...] = ()                   # lists of BDFs
    # R3: within one fan-in wave, source capacities may differ by at most
    # this factor. Beyond it, they get staggered.
    staffel_verhaeltnis: float = 1.5
    # Fraction of the measured capacity above which an edge counts as
    # saturated. Only used for the explanation/report -- the actual
    # decision is made via the cost comparison, not via this threshold.
    saettigung_anteil: float = 0.75
    mess: MeasureConfig = field(default_factory=MeasureConfig)


@dataclass(frozen=True)
class BarlinkConfig:
    kollektiv: CollectiveConfig = field(default_factory=CollectiveConfig)
    nic: str = "nie"                     # nie | bei_bedarf | immer  (never | on_demand | always)


# -- File --------------------------------------------------------------

_MESS_SCHLUESSEL = {
    "groessen_kib", "wiederholungen", "vorlauf", "budget_ms",
    "fanin", "duplex", "cache", "cache_aus",
}
_KOLLEKTIV_SCHLUESSEL = {
    "planer", "algorithmus", "chunk_kib", "blatt_schwelle", "aufteilung",
    "roles", "domaenen", "staffel_verhaeltnis", "saettigung_anteil", "mess",
}
_WURZEL_SCHLUESSEL = {"kollektiv", "nic"}


def _bool(wert: Any, wo: str) -> bool:
    if isinstance(wert, bool):
        return wert
    if isinstance(wert, str):
        s = wert.strip().lower()
        if s in ("1", "ja", "true", "an", "yes"):
            return True
        if s in ("0", "nein", "false", "aus", "no"):
            return False
    raise ConfigError(f"{wo}: {wert!r} is not a boolean value")


def _check_key(gegeben: Iterable[str], erlaubt: set[str], wo: str) -> None:
    unbekannt = sorted(set(gegeben) - erlaubt)
    if unbekannt:
        raise ConfigError(
            f"{wo}: unknown keys {unbekannt}; allowed are "
            f"{sorted(erlaubt)}. (A silently ignored typo in the "
            f"configuration is exactly the kind of bug that later sends "
            f"someone hunting for performance that was switched off by "
            f"configuration.)"
        )


#: A PCI address, with or without a domain. Serves as a probe against
#: anything that merely looks like one (e.g. a bare bus number).
_IST_BDF = re.compile(r"^(?:[0-9a-fA-F]{4}:)?[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.\d$")


def _norm_bdf(s: str) -> str:
    """``05:00.0`` and ``0000:05:00.0`` are the same key."""
    s = str(s).strip().lower()
    if s.count(":") == 1:
        s = "0000:" + s
    return s


def _read_file(pfad: str) -> dict:
    p = pathlib.Path(pfad).expanduser()
    if not p.is_file():
        raise ConfigError(f"config file {pfad!r} does not exist")
    text = p.read_text()
    daten: Any
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:  # pragma: no cover - PyYAML is a project dependency
            raise ConfigError(
                f"{pfad}: YAML requested, but PyYAML is missing ({e}). "
                f"JSON (.json) works without an extra package."
            ) from e
        daten = yaml.safe_load(text)
    else:
        daten = json.loads(text)
    if daten is None:
        return {}
    if not isinstance(daten, dict):
        raise ConfigError(f"{pfad}: expected a mapping, not {type(daten)}")
    # Both `barlink: {...}` and the bare subtree are accepted.
    if set(daten) == {"barlink"}:
        daten = daten["barlink"] or {}
    elif "barlink" in daten:
        daten = daten["barlink"] or {}
    if not isinstance(daten, dict):
        raise ConfigError(f"{pfad}: `barlink` must be a mapping")
    return daten


def _from_mapping(basis: BarlinkConfig, d: Mapping[str, Any], wo: str) -> BarlinkConfig:
    _check_key(d, _WURZEL_SCHLUESSEL, wo)
    koll = basis.kollektiv
    nic = basis.nic
    if "nic" in d:
        nic = str(d["nic"]).strip().lower().replace("-", "_")
    kd = d.get("kollektiv") or {}
    if not isinstance(kd, Mapping):
        raise ConfigError(f"{wo}.kollektiv must be a mapping")
    _check_key(kd, _KOLLEKTIV_SCHLUESSEL, f"{wo}.kollektiv")
    aend: dict[str, Any] = {}
    if "planer" in kd:
        aend["planer"] = str(kd["planer"]).strip().lower()
    if "algorithmus" in kd:
        aend["algorithmus"] = str(kd["algorithmus"]).strip().lower()
    if "chunk_kib" in kd:
        v = kd["chunk_kib"]
        aend["chunk_kib"] = None if str(v).strip().lower() == "auto" else int(v)
    if "blatt_schwelle" in kd:
        aend["blatt_schwelle"] = float(kd["blatt_schwelle"])
    if "staffel_verhaeltnis" in kd:
        aend["staffel_verhaeltnis"] = float(kd["staffel_verhaeltnis"])
    if "saettigung_anteil" in kd:
        aend["saettigung_anteil"] = float(kd["saettigung_anteil"])
    if "aufteilung" in kd:
        aend["aufteilung"] = _read_split(kd["aufteilung"], f"{wo}.kollektiv")
    if "roles" in kd:
        r = kd["roles"] or {}
        if not isinstance(r, Mapping):
            raise ConfigError(f"{wo}.kollektiv.roles must be a mapping")
        aend["roles"] = {
            _norm_bdf(k): str(v).strip().lower() for k, v in r.items()
        }
    if "domaenen" in kd:
        dom = kd["domaenen"] or []
        if not isinstance(dom, Sequence) or isinstance(dom, (str, bytes)):
            raise ConfigError(f"{wo}.kollektiv.domaenen must be a list of lists")
        aend["domaenen"] = tuple(
            tuple(_norm_bdf(x) for x in gruppe) for gruppe in dom
        )
    if "mess" in kd:
        md = kd["mess"] or {}
        if not isinstance(md, Mapping):
            raise ConfigError(f"{wo}.kollektiv.mess must be a mapping")
        _check_key(md, _MESS_SCHLUESSEL, f"{wo}.kollektiv.mess")
        maend: dict[str, Any] = {}
        if "groessen_kib" in md:
            maend["groessen_kib"] = tuple(int(x) for x in md["groessen_kib"])
        for schl in ("wiederholungen", "vorlauf"):
            if schl in md:
                maend[schl] = int(md[schl])
        if "budget_ms" in md:
            maend["budget_ms"] = float(md["budget_ms"])
        for schl in ("fanin", "duplex", "cache_aus"):
            if schl in md:
                maend[schl] = _bool(md[schl], f"{wo}.kollektiv.mess.{schl}")
        if "cache" in md:
            maend["cache"] = None if md["cache"] is None else str(md["cache"])
        aend["mess"] = replace(koll.mess, **maend)
    return BarlinkConfig(kollektiv=replace(koll, **aend), nic=nic)


def _read_split(v: Any, wo: str) -> Any:
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("auto", "gleich", "proportional"):
            return s
        raise ConfigError(
            f"{wo}.aufteilung: {v!r} unknown "
            f"(auto | gleich | proportional | mapping BDF -> weights)"
        )
    if isinstance(v, Mapping):
        aus: dict[str, tuple[float, ...]] = {}
        for k, gew in v.items():
            if not isinstance(gew, Sequence) or isinstance(gew, (str, bytes)):
                raise ConfigError(f"{wo}.aufteilung[{k!r}]: expected a list of weights")
            g = tuple(float(x) for x in gew)
            if not g or any(x < 0 for x in g) or sum(g) <= 0:
                raise ConfigError(f"{wo}.aufteilung[{k!r}]: weights must be >= 0, sum > 0")
            aus[_norm_bdf(k)] = g
        return aus
    raise ConfigError(f"{wo}.aufteilung: {v!r} not parseable")


# -- Environment -------------------------------------------------------------
#
# Order: default < file < environment variable. The environment always
# wins, because it is the way to measure a run against itself without
# changing any file.

_ENV_PRAEFIX = "SGLANG_BARLINK_"


def load_config(env: Optional[Mapping[str, str]] = None) -> BarlinkConfig:
    """Default < file (``SGLANG_BARLINK_CONFIG``) < environment.

    RANK UNIFORMITY: reads exclusively process-global state that is the
    same on every rank (environment, file). Nothing in here may ever
    depend on the rank -- otherwise two ranks would plan differently and
    the collectives' SPMD assumption would collapse.
    """
    env = os.environ if env is None else env
    k = BarlinkConfig()
    pfad = env.get(_ENV_PRAEFIX + "KONFIG")
    if pfad:
        k = _from_mapping(k, _read_file(pfad), wo=pfad)

    ueber: dict[str, Any] = {}
    kueber: dict[str, Any] = {}
    mueber: dict[str, Any] = {}

    def _e(name: str) -> Optional[str]:
        return env.get(_ENV_PRAEFIX + name)

    if (v := _e("NIC")) is not None or (v := _e("MATRIX_NIC")) is not None:
        # `MATRIX_NIC` is the name used in ENTWURF_BARLINK_TRANSPORT.md;
        # `bei-bedarf` there, `bei_bedarf` here -- both are accepted.
        ueber["nic"] = v.strip().lower().replace("-", "_")
    if (v := _e("PLANER")) is not None:
        kueber["planer"] = v.strip().lower()
    if (v := _e("ALGORITHMUS")) is not None:
        kueber["algorithmus"] = v.strip().lower()
    if (v := _e("CHUNK_KIB")) is not None:
        kueber["chunk_kib"] = None if v.strip().lower() == "auto" else int(v)
    if (v := _e("BLATT_SCHWELLE")) is not None:
        kueber["blatt_schwelle"] = float(v)
    if (v := _e("STAFFEL_VERHAELTNIS")) is not None:
        kueber["staffel_verhaeltnis"] = float(v)
    if (v := _e("SAETTIGUNG_ANTEIL")) is not None:
        kueber["saettigung_anteil"] = float(v)
    if (v := _e("AUFTEILUNG")) is not None:
        s = v.strip()
        kueber["aufteilung"] = _read_split(
            json.loads(s) if s.startswith("{") else s, "SGLANG_BARLINK_SPLIT"
        )
    if (v := _e("ROLLEN")) is not None:
        # "0000:05:00.0=blatt,0000:0a:00.0=domaene" or JSON
        s = v.strip()
        if s.startswith("{"):
            roh = json.loads(s)
        else:
            roh = {}
            for teil in s.split(","):
                if not teil.strip():
                    continue
                if "=" not in teil:
                    raise ConfigError(
                        f"SGLANG_BARLINK_ROLES: {teil!r} has no '='; expected "
                        f"'<bdf>=<role>[,<bdf>=<role>]' or JSON"
                    )
                bdf, rolle = teil.split("=", 1)
                roh[bdf] = rolle
        kueber["roles"] = {_norm_bdf(a): str(b).strip().lower() for a, b in roh.items()}
    if (v := _e("DOMAENEN")) is not None:
        # "05:00.0+0b:00.0;0a:00.0" or a JSON list of lists
        s = v.strip()
        if s.startswith("["):
            roh = json.loads(s)
        else:
            roh = [g.split("+") for g in s.split(";") if g.strip()]
        kueber["domaenen"] = tuple(tuple(_norm_bdf(x) for x in g) for g in roh)
    if (v := _e("MESS_GROESSEN_KIB")) is not None:
        mueber["groessen_kib"] = tuple(int(x) for x in v.replace(";", ",").split(","))
    if (v := _e("MESS_WIEDERHOLUNGEN")) is not None:
        mueber["wiederholungen"] = int(v)
    if (v := _e("MESS_VORLAUF")) is not None:
        mueber["vorlauf"] = int(v)
    if (v := _e("MESS_BUDGET_MS")) is not None:
        mueber["budget_ms"] = float(v)
    if (v := _e("MESS_FANIN")) is not None:
        mueber["fanin"] = _bool(v, "SGLANG_BARLINK_MEASURE_FANIN")
    if (v := _e("MESS_DUPLEX")) is not None:
        mueber["duplex"] = _bool(v, "SGLANG_BARLINK_MEASURE_DUPLEX")
    if (v := _e("MATRIX_CACHE")) is not None:
        mueber["cache"] = v
    if (v := _e("MATRIX_CACHE_AUS")) is not None:
        mueber["cache_aus"] = _bool(v, "SGLANG_BARLINK_MATRIX_CACHE_OFF")

    koll = k.kollektiv
    if mueber:
        kueber["mess"] = replace(koll.mess, **mueber)
    if kueber:
        koll = replace(koll, **kueber)
    k = replace(k, kollektiv=koll, **ueber)
    _validate(k)
    return k


def _validate(k: BarlinkConfig) -> None:
    if k.nic not in NIC_MODI:
        raise ConfigError(f"nic={k.nic!r} unknown; allowed {list(NIC_MODI)}")
    c = k.kollektiv
    if c.planer not in PLANER_MODI:
        raise ConfigError(f"kollektiv.planer={c.planer!r}; allowed {list(PLANER_MODI)}")
    if c.algorithmus != "auto" and c.algorithmus not in ALGORITHMEN:
        raise ConfigError(
            f"kollektiv.algorithmus={c.algorithmus!r}; allowed "
            f"{['auto', *ALGORITHMEN]}"
        )
    if not 0.0 < c.blatt_schwelle <= 1.0:
        raise ConfigError(
            f"kollektiv.blatt_schwelle={c.blatt_schwelle} must lie in (0, 1] "
            f"(fraction of the median; 1.0 means 'everything below the "
            f"median is a leaf')"
        )
    if c.staffel_verhaeltnis < 1.0:
        raise ConfigError("kollektiv.staffel_verhaeltnis must be >= 1")
    if not 0.0 < c.saettigung_anteil <= 1.0:
        raise ConfigError("kollektiv.saettigung_anteil must lie in (0, 1]")
    if c.chunk_kib is not None and c.chunk_kib <= 0:
        raise ConfigError("kollektiv.chunk_kib must be > 0 or 'auto'")
    for bdf, rolle in c.roles.items():
        if rolle not in ROLLEN:
            raise ConfigError(
                f"kollektiv.roles[{bdf!r}]={rolle!r}; allowed {list(ROLLEN)}"
            )
    m = c.mess
    if not m.groessen_kib or any(g <= 0 for g in m.groessen_kib):
        raise ConfigError("kollektiv.mess.groessen_kib: expected positive values")
    if m.wiederholungen <= 0 or m.vorlauf < 0:
        raise ConfigError("kollektiv.mess.wiederholungen > 0, vorlauf >= 0")
    if m.budget_ms <= 0:
        raise ConfigError("kollektiv.mess.budget_ms must be > 0")
    if c.planer == "aus" and c.algorithmus == "auto":
        raise ConfigError(
            "kollektiv.planer=aus requires a fixed kollektiv.algorithmus -- "
            "'aus' + 'auto' would mean 'don't measure, but still choose', "
            "and that doesn't exist. No silent fallback."
        )
    if c.planer == "fest" and c.mess.cache_aus:
        raise ConfigError(
            "kollektiv.planer=fest needs the cache; mess.cache_aus=1 takes "
            "away its only source. You probably mean planer=auto."
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

    def self_load(self, nbytes: int, richtung: str) -> float:
        """GB/s. ``richtung`` is ``"aus"`` (D2H) or ``"ein"`` (H2D)."""
        ...

    def self_load_duplex(self, nbytes: int) -> Optional[float]:
        """Sum of both directions running simultaneously, GB/s. ``None`` if not measurable."""
        ...

    def pair(self, ziel: int, nbytes: int) -> Optional[float]:
        """GB/s from *this* rank to rank ``ziel``. ``None`` = no pair path."""
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
                 wiederholungen: Optional[int] = None):
        import torch

        self.device = device
        self._torch = torch
        self._max_bytes = max_bytes
        # None = the size-dependent default (small messages need more
        # rounds, otherwise you end up measuring the clock instead of the
        # link).
        self._reps = wiederholungen
        self._dev = torch.empty(max_bytes, dtype=torch.uint8, device=device)
        self._host = torch.empty(max_bytes, dtype=torch.uint8, pin_memory=True)
        self._host2 = torch.empty(max_bytes, dtype=torch.uint8, pin_memory=True)
        self._s1 = torch.cuda.Stream(device=device)
        self._s2 = torch.cuda.Stream(device=device)

    def name(self) -> str:
        return "self_load"

    def _lauf(self, nbytes: int, richtung: str, n: int) -> None:
        d = self._dev[:nbytes]
        h = self._host[:nbytes]
        for _ in range(n):
            if richtung == "aus":
                h.copy_(d, non_blocking=True)   # D2H == card's outbound direction
            else:
                d.copy_(h, non_blocking=True)   # H2D == card's inbound direction

    def self_load(self, nbytes: int, richtung: str) -> float:
        torch = self._torch
        nbytes = min(nbytes, self._max_bytes)
        self._lauf(nbytes, richtung, 8)
        torch.cuda.synchronize(self.device)
        n = self._reps or _repeats_for(nbytes)
        t0 = time.perf_counter()
        self._lauf(nbytes, richtung, n)
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

    def pair(self, ziel: int, nbytes: int) -> Optional[float]:
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

    welt: int
    groessen: tuple[int, ...]                      # bytes
    bdfs: tuple[str, ...]                          # per rank
    namen: tuple[str, ...]                         # per rank (card name)
    fuehler: str                                   # which sensor measured
    # Node rates per rank and size.
    aus: dict[int, list[float]] = field(default_factory=dict)     # rank -> per size
    ein: dict[int, list[float]] = field(default_factory=dict)
    duplex_summe: dict[int, list[float]] = field(default_factory=dict)
    # Edge rates, if a pair sensor was present: (from, to) -> per size
    kante: dict[tuple[int, int], list[float]] = field(default_factory=dict)
    # R3: fan-in cap per destination and size, plus shares per source.
    fanin_deckel: dict[int, list[float]] = field(default_factory=dict)
    fanin_anteile: dict[tuple[int, int], list[float]] = field(default_factory=dict)
    # Line t(N) = latency + N/rate fitted from the size ladder, per rank:
    # the startup cost of one step, in seconds.
    latenz_s: dict[int, float] = field(default_factory=dict)
    # Shared bottlenecks: by how much an edge slows down when ALL pairs
    # talk at once (mesh) instead of just one (ring). 1.0 means "no
    # interference" AND is also the value for "not measured" -- which of
    # the two applies is told by `netz_faktor_gemessen`.
    #
    # This is the term the ring-wins hypothesis hinges on: a mesh forces
    # EVERY pair to talk, including across switch uplinks and NUMA hops,
    # while a topology-ordered ring crosses shared bottlenecks only once
    # per round. Per-edge capacities alone cannot capture this -- they are
    # each measured on their own edge.
    netz_faktor: list[float] = field(default_factory=list)
    netz_faktor_gemessen: bool = False
    dauer_s: float = 0.0
    hinweise: list[str] = field(default_factory=list)

    # -- derived views (no state) --------------------------------------

    def capacity(self, von: int, nach: int, gi: int) -> float:
        """Directed rank->rank capacity at size ``groessen[gi]``.

        A measured edge wins. Otherwise the self-load estimate
        ``min(source's outbound, destination's inbound)`` -- explicitly an
        **upper bound**, because it cannot see shared bottlenecks (switch
        uplink, a second root complex).
        """
        e = self.kante.get((von, nach))
        if e is not None:
            return e[gi]
        return min(self.aus[von][gi], self.ein[nach][gi])

    def edge_measured(self, von: int, nach: int) -> bool:
        return (von, nach) in self.kante

    def cap(self, nach: int, gi: int) -> float:
        d = self.fanin_deckel.get(nach)
        if d is not None:
            return d[gi]
        return self.ein[nach][gi]

    def cap_measured(self, nach: int) -> bool:
        return nach in self.fanin_deckel

    def mesh_penalty(self, gi: int) -> float:
        """Factor applied to the mesh transfer term. >= 1."""
        if not self.netz_faktor or gi >= len(self.netz_faktor):
            return 1.0
        return max(1.0, self.netz_faktor[gi])

    def duplex_factor(self, rang: int, gi: int) -> float:
        """Sum of both directions / larger single direction. 1.0 = no gain."""
        d = self.duplex_summe.get(rang)
        if d is None:
            return 1.0
        einzeln = max(self.aus[rang][gi], self.ein[rang][gi])
        return d[gi] / einzeln if einzeln > 0 else 1.0

    def step_s(self) -> float:
        """Startup cost of one collective step, in seconds (median across ranks)."""
        if not self.latenz_s:
            return 0.0
        return statistics.median(self.latenz_s.values())

    def as_dict(self) -> dict:
        d = asdict(self)
        # Tuple keys aren't JSON-safe.
        d["kante"] = {f"{a}->{b}": v for (a, b), v in self.kante.items()}
        d["fanin_anteile"] = {f"{a}->{b}": v for (a, b), v in self.fanin_anteile.items()}
        d["aus"] = {str(k): v for k, v in self.aus.items()}
        d["ein"] = {str(k): v for k, v in self.ein.items()}
        d["duplex_summe"] = {str(k): v for k, v in self.duplex_summe.items()}
        d["fanin_deckel"] = {str(k): v for k, v in self.fanin_deckel.items()}
        d["latenz_s"] = {str(k): v for k, v in self.latenz_s.items()}
        return d

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "Measurement":
        def pair(s: str) -> tuple[int, int]:
            a, b = s.split("->")
            return int(a), int(b)

        m = Measurement(
            welt=int(d["welt"]),
            groessen=tuple(int(x) for x in d["groessen"]),
            bdfs=tuple(d["bdfs"]),
            namen=tuple(d["namen"]),
            fuehler=str(d["fuehler"]),
            dauer_s=float(d.get("dauer_s", 0.0)),
            hinweise=list(d.get("hinweise", [])),
        )
        m.aus = {int(k): list(v) for k, v in d["aus"].items()}
        m.ein = {int(k): list(v) for k, v in d["ein"].items()}
        m.duplex_summe = {int(k): list(v) for k, v in d.get("duplex_summe", {}).items()}
        m.kante = {pair(k): list(v) for k, v in d.get("kante", {}).items()}
        m.fanin_deckel = {int(k): list(v) for k, v in d.get("fanin_deckel", {}).items()}
        m.fanin_anteile = {
            pair(k): list(v) for k, v in d.get("fanin_anteile", {}).items()
        }
        m.latenz_s = {int(k): float(v) for k, v in d.get("latenz_s", {}).items()}
        m.netz_faktor = [float(x) for x in d.get("netz_faktor", [])]
        m.netz_faktor_gemessen = bool(d.get("netz_faktor_gemessen", False))
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
    """One stage of the size ladder: up to ``max_bytes``, ``algorithmus`` applies."""

    von_bytes: int                       # smallest size this was measured for
    max_bytes: int                       # inclusive; -1 == "and everything above"
    algorithmus: str
    vorhersage_s: Mapping[str, float]    # algorithm -> predicted time
    grund: str


@dataclass(frozen=True)
class Plan:
    welt: int
    bdfs: tuple[str, ...]
    roles: tuple[str, ...]                     # per rank
    domaene: tuple[int, ...]                    # ranks of the reduction domain
    blaetter: tuple[int, ...]
    eltern: Mapping[int, tuple[int, ...]]       # leaf -> domain nodes
    aufteilung: Mapping[int, tuple[int, ...]]   # leaf -> per-mille share per parent
    ringfolge: tuple[int, ...]
    leiter: tuple[Stage, ...]
    chunk_bytes: int
    staffeln: Mapping[int, tuple[tuple[int, ...], ...]]   # destination -> waves of sources
    konfig_zusammenfassung: Mapping[str, Any]
    messung: Optional[Measurement] = None
    source: str = "gemessen"                    # gemessen | zwischenspeicher | fest  (measured | cached | fixed)

    # -- queries a transport needs ---------------------------------------

    def algorithm_for(self, nbytes: int) -> str:
        for stufe in self.leiter:
            if stufe.max_bytes < 0 or nbytes <= stufe.max_bytes:
                return stufe.algorithmus
        return self.leiter[-1].algorithmus

    def is_leaf(self, rang: int) -> bool:
        return self.roles[rang] == "blatt"

    def checksum(self) -> str:
        """Fingerprint of the *decisions*, not of the raw measurement.

        Deliberately without ``messung``: the raw values differ between
        ranks by measurement noise once every rank contributes its own
        numbers. What MUST agree is the plan.
        """
        kern = {
            "welt": self.welt,
            "roles": list(self.roles),
            "domaene": list(self.domaene),
            "blaetter": list(self.blaetter),
            "eltern": {str(k): list(v) for k, v in sorted(self.eltern.items())},
            "aufteilung": {str(k): list(v) for k, v in sorted(self.aufteilung.items())},
            "ringfolge": list(self.ringfolge),
            "leiter": [(s.von_bytes, s.max_bytes, s.algorithmus) for s in self.leiter],
            "chunk_bytes": self.chunk_bytes,
            "staffeln": {
                str(k): [list(w) for w in v] for k, v in sorted(self.staffeln.items())
            },
        }
        roh = json.dumps(kern, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(roh.encode()).hexdigest()[:16]

    # -- Erklaerung ---------------------------------------------------------

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
          f"{self.welt} ranks")
        a("=" * 78)
        m = self.messung
        if m is not None:
            a(f"Sensor: {m.fuehler}   Measurement duration: {m.dauer_s * 1000:.0f} ms")
            a("")
            a("-- measured node rates (GB/s, outbound/inbound per size) --")
            a("   The sysfs column is shown ONLY for comparison and feeds "
              "into no decision:")
            a("   if it says x8 and the measurement says 6 GB/s, either the "
              "card is downclocked")
            a("   or the measurement setup is broken -- lspci is a starting "
              "estimate, not truth.")
            kopf = "  Rank  Card                   sysfs   " + "".join(
                f"{_kib(g):>18}" for g in m.groessen
            )
            a(kopf)
            for r in range(self.welt):
                lb = link_width(self.bdfs[r])
                lbs = f"x{lb[0]}" if lb else "?"
                zeile = (f"  {r:>4}  {self.bdfs[r]:<14}{m.namen[r][:8]:<8}"
                         f"{lbs:<8}")
                for gi in range(len(m.groessen)):
                    zeile += f"{m.aus[r][gi]:>8.2f}/{m.ein[r][gi]:<9.2f}"
                a(zeile)
            if m.duplex_summe:
                a("")
                a("-- full duplex (sum of both directions / stronger single direction) --")
                a("   R4: measured, not assumed. 1.00 = the reverse direction "
                  "isn't free at all;")
                a("   2.00 would be the naive expectation. Reference for this rig: "
                  f"{_BELEG_DUPLEX_SUMME_20KIB:.2f}x at 20 KiB, "
                  f"{_BELEG_DUPLEX_SUMME_1MIB:.2f}x at 1 MiB.")
                for r in range(self.welt):
                    if r not in m.duplex_summe:
                        continue
                    werte = "  ".join(
                        f"{_kib(g)}: {m.duplex_factor(r, gi):.2f}x"
                        for gi, g in enumerate(m.groessen)
                    )
                    a(f"  Rank {r}: {werte}")
            a("")
            a("-- directed edge capacities (GB/s) --")
            a("   'M' = measured on the pair, 'S' = estimate min(outbound, "
              "inbound) from self-load")
            for gi, g in enumerate(m.groessen):
                a(f"  at {_kib(g)}:")
                a("        " + "".join(f"{'->' + str(j):>12}" for j in range(self.welt)))
                for i in range(self.welt):
                    zeile = f"    {i:>2}  "
                    for j in range(self.welt):
                        if i == j:
                            zeile += f"{'-':>12}"
                        else:
                            mark = "M" if m.edge_measured(i, j) else "S"
                            zeile += f"{m.capacity(i, j, gi):>10.2f}{mark}"
                    a(zeile)
            a("")
            a("-- fan-in cap per destination (GB/s) --")
            a("   R3: no card receives more than the cap, regardless of how "
              "many sources.")
            a(f"   Reference for this rig: {_BELEG_FANIN_GBPS:.2f} GB/s at 1 MiB "
              "with two sources.")
            for j in range(self.welt):
                mark = "measured" if m.cap_measured(j) else "estimated from inbound rate"
                werte = "  ".join(
                    f"{_kib(g)}: {m.cap(j, gi):.2f}" for gi, g in enumerate(m.groessen)
                )
                a(f"  Destination {j} ({self.bdfs[j]}): {werte}   [{mark}]")
            a(f"  Step cost (startup cost per collective step): "
              f"{m.step_s() * 1e6:.2f} us")
            a("")
            a("-- shared bottlenecks: mesh penalty vs. the ring --")
            if m.netz_faktor_gemessen:
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
            werte = "  ".join(
                f"{_kib(g)}: {m.mesh_penalty(gi):.2f}x"
                for gi, g in enumerate(m.groessen)
            )
            a(f"  {werte}")
            for h in m.hinweise:
                a(f"  ! {h}")
        else:
            a("No measurement (planner off or fixed configuration).")

        a("")
        a("-- roles (R2: capacity < blatt_schwelle x median -> leaf) --")
        for r in range(self.welt):
            zusatz = ""
            if r in self.eltern:
                el = ", ".join(str(x) for x in self.eltern[r])
                anteile = "/".join(f"{p / 10:.0f}%" for p in self.aufteilung.get(r, ()))
                zusatz = f"  -> parents [{el}]  split {anteile}"
            a(f"  Rank {r} ({self.bdfs[r]}): {self.roles[r]}{zusatz}")
        a(f"  Reduction domain: {list(self.domaene)}   Leaves: {list(self.blaetter)}")
        a(f"  Ring order (sorted by measured capacity): "
          f"{list(self.ringfolge)}")

        a("")
        a("-- algorithm per size class --")
        for stufe in self.leiter:
            grenze = (f"from {_kib(stufe.von_bytes)}" if stufe.max_bytes < 0
                      else f"{_kib(stufe.von_bytes)}..{_kib(stufe.max_bytes)}")
            vor = "  ".join(
                f"{k}={v * 1e6:.1f}us" for k, v in sorted(stufe.vorhersage_s.items())
            )
            a(f"  {grenze:>16}: {stufe.algorithmus:<14} [{vor}]")
            a(f"                    {stufe.grund}")
        a(f"  Chunk: {self.chunk_bytes // 1024} KiB")

        if self.staffeln:
            a("")
            a("-- fan-in staging plan (R3) --")
            a("   A fast and a slow source may not write into the same "
              "destination at the")
            a("   same time: the cap is split evenly per source, the fast "
              "one gets pulled")
            a("   down to the slow one's share. More than one wave means "
              "staggered.")
            for ziel, wellen in sorted(self.staffeln.items()):
                w = " | ".join("+".join(str(q) for q in welle) for welle in wellen)
                a(f"  Destination {ziel} ({self.bdfs[ziel]}): {w}")

        a("")
        a("-- effective configuration --")
        for k, v in sorted(self.konfig_zusammenfassung.items()):
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


def _time_mesh(m: Measurement, gi: int, n_bytes: int, raenge: Sequence[int],
               wellen: Mapping[int, tuple[tuple[int, ...], ...]],
               step_s: float) -> float:
    """Direct mesh: reduce-scatter + all-gather, exactly one step each.

    ``raenge`` are REAL ranks, not indices -- this function is also
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
    r = len(raenge)
    if r < 2:
        return 0.0
    menge = set(raenge)
    anteil = n_bytes / r             # everyone sends N/R to each neighbor
    schlimmste = 0.0
    schritte = 2
    for ziel in raenge:
        w = tuple(
            tuple(q for q in welle if q in menge)
            for welle in wellen.get(ziel, (tuple(x for x in raenge if x != ziel),))
        )
        w = tuple(welle for welle in w if welle)
        cap = m.cap(ziel, gi)
        # Waves run SEQUENTIALLY -- that's their whole purpose. So sum
        # them, don't take the maximum: staggered fan-in costs time, and
        # that cost must be visible in the comparison against the ring.
        summe = 0.0
        for welle in w:
            gleichzeitig = len(welle)
            pro_quelle = cap / gleichzeitig if gleichzeitig else cap
            dauer = 0.0
            for source in welle:
                rate = min(m.capacity(source, ziel, gi), pro_quelle)
                dauer = max(dauer, anteil / (rate * 1e9) if rate > 0 else float("inf"))
            summe += dauer
        schlimmste = max(schlimmste, summe)
        schritte = max(schritte, 2 * len(w))
    return schritte * step_s + 2 * schlimmste * m.mesh_penalty(gi)


def _time_ring(m: Measurement, gi: int, n_bytes: int, folge: Sequence[int],
               step_s: float) -> float:
    """Ring: 2(R-1) steps, but every rank receives from exactly ONE neighbor.

    No fan-in, no unfair split -- the reason the ring wins at saturation.
    The price paid for that is the step count, which for R=8 is fourteen
    instead of two.
    """
    welt = len(folge)
    if welt < 2:
        return 0.0
    anteil = n_bytes / welt
    langsamste = 0.0
    for k in range(welt):
        von, nach = folge[k], folge[(k + 1) % welt]
        rate = m.capacity(von, nach, gi)
        langsamste = max(langsamste, anteil / (rate * 1e9) if rate > 0 else float("inf"))
    return 2 * (welt - 1) * (step_s + langsamste)


def _time_star(m: Measurement, gi: int, n_bytes: int, nabe: int, welt: int,
                step_s: float) -> float:
    """Star: everything to the hub, reduce there, distribute back.

    Two strictly separate phases with (R-1)N per direction on the hub. It
    is in the model because it is today's status quo and because
    ``algorithmus: star`` must remain forceable -- not because it ever
    wins.
    """
    if welt < 2:
        return 0.0
    ein = m.cap(nabe, gi)
    aus_raten = [m.capacity(nabe, j, gi) for j in range(welt) if j != nabe]
    aus = min(aus_raten) if aus_raten else 0.0
    if ein <= 0 or aus <= 0:
        return float("inf")
    hin = (welt - 1) * n_bytes / (ein * 1e9)
    rueck = (welt - 1) * n_bytes / (aus * 1e9)
    return 2 * step_s + hin + rueck


def _time_hierarchical(m: Measurement, gi: int, n_bytes: int,
                       domaene: Sequence[int], blaetter: Sequence[int],
                       eltern: Mapping[int, tuple[int, ...]],
                       aufteilung: Mapping[int, tuple[int, ...]],
                       innen_s: float, step_s: float, welt: int) -> float:
    """Leaf + reduction domain, chunked and overlapped.

    The leaf sends N out and pulls N in -- the floor, regardless of
    algorithm. With chunking the two directions overlap, but **not for
    free**: the credit is the measured duplex factor (R4), not 2.
    """
    if not blaetter:
        return innen_s
    schlimmstes_blatt = 0.0
    for b in blaetter:
        el = eltern.get(b, tuple(domaene))
        gew = aufteilung.get(b, tuple(1000 // max(len(el), 1) for _ in el))
        gesamt = sum(gew) or 1
        hoch = 0.0
        runter = 0.0
        for e, g in zip(el, gew):
            teil = n_bytes * (g / gesamt)
            r_hoch = m.capacity(b, e, gi)
            r_runter = m.capacity(e, b, gi)
            hoch = max(hoch, teil / (r_hoch * 1e9) if r_hoch > 0 else float("inf"))
            runter = max(runter, teil / (r_runter * 1e9) if r_runter > 0 else float("inf"))
        # Overlap: the duplex factor says how much of the reverse
        # direction is actually free. f=1 -> no overlap (hoch+runter),
        # f=2 -> full overlap (max). The measured value lies in between.
        f = max(1.0, min(2.0, m.duplex_factor(b, gi)))
        seriell = hoch + runter
        parallel = max(hoch, runter)
        schlimmstes_blatt = max(schlimmstes_blatt, seriell - (f - 1.0) * (seriell - parallel))
    return 2 * step_s + schlimmstes_blatt + innen_s


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
    gi = len(m.groessen) - 1        # largest measured size: that's where it separates
    return min(m.aus[r][gi], m.ein[r][gi])


def _roles(m: Measurement, k: CollectiveConfig) -> tuple[list[str], list[int], list[int]]:
    welt = m.welt
    cap = [_node_capacity(m, r) for r in range(welt)]
    med = statistics.median(cap) if cap else 0.0
    schwelle = k.blatt_schwelle * med
    roles = ["domaene"] * welt
    for r in range(welt):
        if welt > 2 and cap[r] < schwelle:
            roles[r] = "blatt"
    # Manually set roles override the measurement -- deliberately.
    for r in range(welt):
        fest = k.roles.get(m.bdfs[r])
        if fest is not None:
            roles[r] = "domaene" if fest == "nabe" else fest
    # There must be at least two domain nodes, otherwise there is nothing
    # to reduce into. If necessary, pull back the strongest ones.
    dom = [r for r in range(welt) if roles[r] != "blatt"]
    if len(dom) < min(2, welt):
        stark = sorted(range(welt), key=lambda r: (-cap[r], r))[: min(2, welt)]
        for r in stark:
            roles[r] = "domaene"
    dom = [r for r in range(welt) if roles[r] != "blatt"]
    blaetter = [r for r in range(welt) if roles[r] == "blatt"]
    return roles, dom, blaetter


def _domains_from_config(m: Measurement, k: CollectiveConfig) -> Optional[list[list[int]]]:
    if not k.domaenen:
        return None
    index = {bdf: r for r, bdf in enumerate(m.bdfs)}
    aus: list[list[int]] = []
    for gruppe in k.domaenen:
        raenge = []
        for bdf in gruppe:
            if bdf not in index:
                raise ConfigError(
                    f"kollektiv.domaenen names {bdf!r}, but this ensemble "
                    f"only has {sorted(index)}. No silent fallback -- "
                    f"either the address is wrong or the card is missing."
                )
            raenge.append(index[bdf])
        aus.append(sorted(raenge))
    gesehen = [r for g in aus for r in g]
    if len(gesehen) != len(set(gesehen)):
        raise ConfigError("kollektiv.domaenen: one rank appears in two domains")
    return aus


def _parent_and_split(
    m: Measurement, domaene: Sequence[int], blaetter: Sequence[int], k: CollectiveConfig
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
    gi = len(m.groessen) - 1
    eltern: dict[int, tuple[int, ...]] = {}
    aufteilung: dict[int, tuple[int, ...]] = {}
    modus = k.aufteilung
    for b in blaetter:
        kand = sorted(domaene, key=lambda d: (-m.capacity(b, d, gi), d))
        if not kand:
            continue
        eltern[b] = tuple(kand)
        if isinstance(modus, Mapping):
            gew = modus.get(m.bdfs[b])
            if gew is not None:
                if len(gew) != len(kand):
                    raise ConfigError(
                        f"kollektiv.aufteilung[{m.bdfs[b]!r}] names {len(gew)} "
                        f"weights, but the leaf has {len(kand)} parents "
                        f"{[m.bdfs[d] for d in kand]}"
                    )
                aufteilung[b] = _promille(gew)
                continue
            modus_b = "auto"
        else:
            modus_b = modus
        raten = [m.capacity(b, d, gi) for d in kand]
        if modus_b == "gleich" or not any(raten):
            aufteilung[b] = _promille([1.0] * len(kand))
        elif modus_b == "proportional":
            aufteilung[b] = _promille(raten)
        else:  # auto
            spanne = (max(raten) / min(raten)) if min(raten) > 0 else float("inf")
            aufteilung[b] = _promille(
                [1.0] * len(kand) if spanne > k.staffel_verhaeltnis else raten
            )
    return eltern, aufteilung


def _promille(gew: Sequence[float]) -> tuple[int, ...]:
    s = sum(gew)
    if s <= 0:
        n = len(gew)
        return tuple([1000 // n] * (n - 1) + [1000 - (1000 // n) * (n - 1)])
    roh = [g / s * 1000.0 for g in gew]
    ganz = [int(x) for x in roh]
    rest = 1000 - sum(ganz)
    # Largest fractional remainders get the leftover -- deterministic, so
    # every rank arrives at the same result.
    ordn = sorted(range(len(roh)), key=lambda i: (-(roh[i] - ganz[i]), i))
    for i in ordn[:rest]:
        ganz[i] += 1
    return tuple(ganz)


def _ring_order(m: Measurement, welt: int) -> tuple[int, ...]:
    """Ring ordered by measured capacity.

    The ring wins, among other reasons, because it can lay out
    neighborhoods so that traffic stays local and shared bottlenecks are
    crossed only once per round. The order here is built **from the
    measurements** (a greedy tour over the strongest edges); PCI
    proximity is used only as a tie-break and never as a decision --
    ``lspci`` is a starting estimate, not truth.
    """
    if welt < 2:
        return tuple(range(welt))
    gi = len(m.groessen) - 1
    start = min(range(welt), key=lambda r: (-_node_capacity(m, r), r))
    folge = [start]
    offen = set(range(welt)) - {start}
    while offen:
        letzter = folge[-1]
        naechster = min(
            offen,
            key=lambda j: (
                -min(m.capacity(letzter, j, gi), m.capacity(j, letzter, gi)),
                -_pci_proximity(m.bdfs[letzter], m.bdfs[j]),
                j,
            ),
        )
        folge.append(naechster)
        offen.discard(naechster)
    return tuple(folge)


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
    m: Measurement, welt: int, verhaeltnis: float
) -> dict[int, tuple[tuple[int, ...], ...]]:
    """R3 as a schedule: who may write into the same destination at the same time.

    Rule taken from the fan-in measurement: the cap is split evenly per
    source, not by capability. An x8 source next to an x4 source dropped
    from 12.81 to 6.75 GB/s, while the x4 source kept its 6.46. Hence:
    sources whose capacities differ by more than ``verhaeltnis`` belong in
    different waves.
    """
    gi = len(m.groessen) - 1
    plan: dict[int, tuple[tuple[int, ...], ...]] = {}
    for ziel in range(welt):
        quellen = [q for q in range(welt) if q != ziel]
        if len(quellen) < 2:
            plan[ziel] = (tuple(quellen),)
            continue
        quellen.sort(key=lambda q: (-m.capacity(q, ziel, gi), q))
        wellen: list[list[int]] = []
        for q in quellen:
            rate = m.capacity(q, ziel, gi)
            gelegt = False
            for welle in wellen:
                raten = [m.capacity(x, ziel, gi) for x in welle] + [rate]
                lo, hi = min(raten), max(raten)
                if lo > 0 and hi / lo <= verhaeltnis:
                    welle.append(q)
                    gelegt = True
                    break
            if not gelegt:
                wellen.append([q])
        plan[ziel] = tuple(tuple(w) for w in wellen)
    return plan


def _chunk_bytes(m: Measurement, k: CollectiveConfig, welt: int) -> int:
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
    gi = len(m.groessen) - 1
    raten = [m.capacity(i, j, gi) for i in range(welt) for j in range(welt) if i != j]
    rate = statistics.median(raten) if raten else 1.0
    schritt = m.step_s()
    if schritt <= 0 or rate <= 0:
        return 256 * 1024
    ziel = 4.0 * schritt * rate * 1e9
    kib = max(16, min(4096, 2 ** int(round(math.log2(max(ziel, 1.0) / 1024)))))
    return int(kib) * 1024


# ===========================================================================
# plan_collective(): measurement + configuration -> plan
# ===========================================================================


def _window_requirement(algorithmus: str, nbytes: int, welt: int) -> int:
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

    CORRECTION: this used to say ``2*2*anteil`` for the ring, i.e. four
    slots regardless of R. At ``R=3`` that is the same value
    (``2(R-1) = 4``) and so it never stood out; from ``R=4`` on it was too
    small. An undersized requirement lets an algorithm be selected that
    the mapping cannot actually carry -- and the bug then only shows up
    in the transport.
    """
    if welt < 2:
        return 0
    anteil = -(-nbytes // welt)
    if algorithmus in ("mesh", "ring", "hierarchisch"):
        return 2 * (welt - 1) * anteil
    if algorithmus == "star":
        return 2 * (welt - 1) * nbytes
    return 0


def plan_collective(m: Measurement, k: BarlinkConfig, source: str = "gemessen",
          fenster_bytes: Optional[int] = None) -> Plan:
    """Purely computational, with no I/O at all -- so it's testable without hardware.

    Exclusively a function of ``(m, k, fenster_bytes)``. Two ranks with
    the same inputs are guaranteed to get the same plan; that is exactly
    what ``BarlinkMatrixPlanner`` checks afterwards via the checksum.

    ``fenster_bytes`` is how much BAR1 can be mapped simultaneously per
    destination -- i.e. a **capability**, not a policy. If it is known,
    any algorithm whose window requirement exceeds it drops out. This is
    the second reason the ring can win: it needs two slots of ``N/R``,
    the mesh needs ``R-1``. ``None`` means "unknown" and rules nothing
    out -- not "unlimited".
    """
    c = k.kollektiv
    welt = m.welt
    roles, dom, blaetter = _roles(m, c)

    fest_dom = _domains_from_config(m, c)
    if fest_dom is not None:
        # Manually set domains: the first entry is the reduction domain,
        # everything not named becomes a leaf.
        genannt = {r for g in fest_dom for r in g}
        dom = sorted(fest_dom[0])
        blaetter = [r for r in range(welt) if r not in dom]
        roles = ["domaene" if r in dom else "blatt" for r in range(welt)]
        for r in range(welt):
            if r not in genannt and c.roles.get(m.bdfs[r]) is None:
                roles[r] = "blatt"

    eltern, aufteilung = _parent_and_split(m, dom, blaetter, c)
    folge = _ring_order(m, welt)
    staffeln = _tier_plan(m, welt, c.staffel_verhaeltnis)
    chunk = _chunk_bytes(m, c, welt)
    schritt = m.step_s()
    nabe = folge[0] if folge else 0

    stufen: list[Stage] = []
    for gi, g in enumerate(m.groessen):
        vorhersage: dict[str, float] = {}
        vorhersage["mesh"] = _time_mesh(m, gi, g, range(welt), staffeln, schritt)
        vorhersage["ring"] = _time_ring(m, gi, g, folge, schritt)
        vorhersage["star"] = _time_star(m, gi, g, nabe, welt, schritt)
        if blaetter and len(dom) >= 2:
            # Within the domain, whatever wins there wins.
            innen_netz = _time_mesh(m, gi, g, dom, staffeln, schritt)
            innen_ring = _time_ring(
                m, gi, g, [d for d in folge if d in dom] or dom, schritt
            )
            vorhersage["hierarchisch"] = _time_hierarchical(
                m, gi, g, dom, blaetter, eltern, aufteilung,
                min(innen_netz, innen_ring), schritt, welt,
            )

        # Capability limit: whatever doesn't fit the window isn't in the
        # running. Deliberately BEFORE policy -- a configuration must not
        # be able to force an algorithm the hardware cannot map.
        zu_gross = {
            a: _window_requirement(a, g, welt) for a in list(vorhersage)
            if fenster_bytes is not None
            and _window_requirement(a, g, welt) > fenster_bytes
        }
        moeglich = {a: v for a, v in vorhersage.items() if a not in zu_gross}

        if c.algorithmus != "auto":
            gewaehlt = c.algorithmus
            if gewaehlt in zu_gross:
                raise ConfigError(
                    f"kollektiv.algorithmus={gewaehlt} forced, but at "
                    f"{_kib(g)} and {welt} ranks it needs "
                    f"{zu_gross[gewaehlt] // 1024} KiB of window; mappable "
                    f"is {fenster_bytes // 1024} KiB. No silent fallback -- "
                    f"either chunk smaller or choose a different algorithm."
                )
            grund = f"forced via kollektiv.algorithmus={c.algorithmus}"
        else:
            if not moeglich:
                raise ConfigError(
                    f"At {_kib(g)} and {welt} ranks NO algorithm fits into "
                    f"the mappable window of {fenster_bytes // 1024} KiB "
                    f"(requirement: "
                    f"{ {a: b // 1024 for a, b in zu_gross.items()} } KiB). "
                    f"This is a startup error, not a quiet reroute: chunk "
                    f"smaller, or exclude the direct path for this size."
                )
            gewaehlt = min(moeglich, key=lambda a: (moeglich[a], a))
            grund = _reason(m, gi, g, welt, gewaehlt, vorhersage, staffeln,
                           c.saettigung_anteil, blaetter)
            if zu_gross:
                grund += (
                    "  Excluded for being too large for the window: "
                    + ", ".join(
                        f"{a} ({b // 1024} KiB > {fenster_bytes // 1024} KiB)"
                        for a, b in sorted(zu_gross.items())
                    ) + "."
                )
        stufen.append(
            Stage(
                von_bytes=g,
                max_bytes=g if gi + 1 < len(m.groessen) else -1,
                algorithmus=gewaehlt,
                vorhersage_s={a: round(v, 9) for a, v in vorhersage.items()},
                grund=grund,
            )
        )

    stufen = _smooth_ladder(stufen)

    zus = {
        "planer": c.planer,
        "algorithmus": c.algorithmus,
        "chunk_kib": "auto" if c.chunk_kib is None else c.chunk_kib,
        "blatt_schwelle": c.blatt_schwelle,
        "aufteilung": c.aufteilung if isinstance(c.aufteilung, str) else "manual",
        "staffel_verhaeltnis": c.staffel_verhaeltnis,
        "roles (manual)": dict(c.roles) or "-",
        "domaenen (manual)": [list(g) for g in c.domaenen] or "-",
        "nic": k.nic,
        "mess.groessen_kib": list(c.mess.groessen_kib),
        "mess.budget_ms": c.mess.budget_ms,
    }
    return Plan(
        welt=welt,
        bdfs=m.bdfs,
        roles=tuple(roles),
        domaene=tuple(dom),
        blaetter=tuple(blaetter),
        eltern={k2: v for k2, v in sorted(eltern.items())},
        aufteilung={k2: v for k2, v in sorted(aufteilung.items())},
        ringfolge=folge,
        leiter=tuple(stufen),
        chunk_bytes=chunk,
        staffeln=staffeln,
        konfig_zusammenfassung=zus,
        messung=m,
        source=source,
    )


def _reason(m: Measurement, gi: int, g: int, welt: int, gewaehlt: str,
           vorhersage: Mapping[str, float],
           staffeln: Mapping[int, tuple[tuple[int, ...], ...]],
           saettigung_anteil: float, blaetter: Sequence[int]) -> str:
    """One sentence saying why. Without it the choice isn't traceable."""
    mesh, ring = vorhersage.get("mesh", 0.0), vorhersage.get("ring", 0.0)
    anteil = g / welt
    # Load the mesh would create on the most heavily addressed
    # destination: what the R-1 sources together want to deliver, against
    # that destination's measured cap.
    last = 0.0
    for ziel in range(welt):
        cap = m.cap(ziel, gi)
        quellen = [m.capacity(q, ziel, gi) for q in range(welt) if q != ziel]
        if cap > 0 and quellen:
            last = max(last, sum(quellen) / cap)
    gesaettigt = last > 1.0 / max(saettigung_anteil, 1e-9)
    gestaffelt = sum(1 for w in staffeln.values() if len(w) > 1)
    t = []
    if gewaehlt == "mesh":
        if gesaettigt:
            t.append(
                f"Mesh despite saturation (demand/cap {last:.2f}): the step "
                f"count outweighs it. 2 steps instead of {2 * (welt - 1)}, "
                f"and one step costs {m.step_s() * 1e6:.1f} us here."
            )
        else:
            t.append(
                f"Mesh: 2 steps instead of {2 * (welt - 1)}; demand/cap "
                f"{last:.2f} is below saturation -- concurrency is nearly "
                f"free there (measured ratio 0.99 at 20 KiB)."
            )
        if gestaffelt:
            t.append(f"{gestaffelt} destination(s) need to be staggered for this.")
    elif gewaehlt == "ring":
        t.append(
            f"Ring: demand/cap {last:.2f} (threshold "
            f"{1.0 / saettigung_anteil:.2f}) -- at saturation, the mesh's "
            f"concurrency buys nothing (measured 1.03x at 1 MiB) and the "
            f"cap is split evenly instead of proportionally, wasting the "
            f"fast edges. In the ring, every rank receives from exactly "
            f"one neighbor."
        )
        if gestaffelt:
            t.append(f"{gestaffelt} destination(s) would have needed staggering in the mesh.")
    elif gewaehlt == "hierarchisch":
        t.append(
            f"Hierarchical: {len(blaetter)} leaf/leaves carry no transit "
            f"traffic, sending only their own contribution and receiving "
            f"only the result; the domain reduces among itself."
        )
    elif gewaehlt == "star":
        t.append("Star: chosen only because everything else predicts worse, "
                 "or because it was forced.")
    t.append(f"(mesh {mesh * 1e6:.1f} us / ring {ring * 1e6:.1f} us, "
             f"share per edge {anteil / 1024:.0f} KiB)")
    return " ".join(t)


def _smooth_ladder(stufen: list[Stage]) -> list[Stage]:
    """Merge consecutive stages that use the same algorithm.

    A ladder with three entries of the same algorithm isn't a ladder, it's
    a single line -- and it reads that way in the explanation too.
    """
    aus: list[Stage] = []
    for s in stufen:
        if aus and aus[-1].algorithmus == s.algorithmus:
            # von_bytes stays as-is: the stage then spans from the
            # smallest to the largest size it applies to.
            aus[-1] = replace(aus[-1], max_bytes=s.max_bytes,
                              vorhersage_s=s.vorhersage_s, grund=s.grund)
        else:
            aus.append(s)
    if aus:
        aus[-1] = replace(aus[-1], max_bytes=-1)
    return aus


# ===========================================================================
# Cache
# ===========================================================================


def _default_cache() -> str:
    basis = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(basis, "sglang", "barlink_matrix.json")


def fingerprint(bdfs: Sequence[str], namen: Sequence[str],
                  k: BarlinkConfig) -> str:
    """Card list, PCI addresses, driver version, patch state, measurement parameters.

    If any of these change, the matrix is measured again. The patch state
    is deliberately part of it: without the regkey the direct path
    doesn't carry, and a matrix measured with it becomes wrong afterwards.
    """
    m = k.kollektiv.mess
    teile = {
        "version": PLANER_VERSION,
        "bdfs": list(bdfs),
        "namen": list(namen),
        "treiber": _driver_version(),
        "patch": _patch_state(),
        "groessen_kib": list(m.groessen_kib),
        "wiederholungen": m.wiederholungen,
        "fanin": m.fanin,
        "duplex": m.duplex,
    }
    roh = json.dumps(teile, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(roh.encode()).hexdigest()[:24]


def _driver_version() -> str:
    try:
        with open("/proc/driver/nvidia/version") as f:
            return f.read().strip().split("\n")[0]
    except OSError:
        return "unknown"


def _patch_state() -> str:
    """Driver regkeys, as far as visible.

    ``RegistryDwords`` is where ``RMSmallBarP2PPeerBar1`` and
    ``RMPcieP2PType`` get set. If nothing is there, the direct path isn't
    unlocked -- a matrix measured with it must not be reused for a run
    with the patch, and vice versa.
    """
    try:
        with open("/proc/driver/nvidia/params") as f:
            zeilen = [
                z.strip() for z in f
                if z.startswith(("RegistryDwords", "EnableResizableBar"))
            ]
        return "|".join(zeilen)
    except OSError:
        return "unknown"


def read_cache(pfad: str, fa: str) -> Optional[Measurement]:
    try:
        with open(pfad) as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if d.get("fingerprint") != fa:
        return None
    try:
        return Measurement.from_dict(d["messung"])
    except (KeyError, TypeError, ValueError) as e:
        logger.warning("barlink-Matrix: cache %s unreadable (%s); measuring again.",
                       pfad, e)
        return None


def write_cache(pfad: str, fa: str, m: Measurement) -> None:
    try:
        p = pathlib.Path(pfad)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(
            {"fingerprint": fa, "messung": m.as_dict()},
            sort_keys=True, indent=1,
        ))
        tmp.replace(p)
    except OSError as e:
        logger.warning("barlink-Matrix: cache %s not writable (%s).",
                       pfad, e)


# ===========================================================================
# The planner: what happens at startup
# ===========================================================================


class BarlinkMatrixPlanner:
    """Measures, plans, checks rank uniformity, explains.

    Usage::

        planer = BarlinkMatrixPlanner(cpu_group, device, config=load_config())
        plan = planer.plan()            # measures on the first call
        logger.info("%s", plan.explanation())

    The plan is **frozen** afterwards. This is not a convenience, it's a
    requirement: decode runs inside captured CUDA graphs, and the choice
    must be fixed at capture time. Switching per message would force a
    re-capture. Load-dependent dynamism is only allowed outside captured
    regions, and then via multiple captured variants, not via a change to
    this plan.
    """

    def __init__(self, cpu_group, device, config: Optional[BarlinkConfig] = None,
                 fuehler: Optional[Sensor] = None,
                 fenster_bytes: Optional[int] = None):
        import torch.distributed as dist

        self.cpu_group = cpu_group
        self.device = device
        self.config = config if config is not None else load_config()
        self.rank = dist.get_rank(cpu_group)
        self.welt = dist.get_world_size(cpu_group)
        self._fuehler = fuehler
        # Capability, not policy: how much BAR1 can be mapped
        # simultaneously per destination. None = unknown (rules nothing
        # out).
        self.fenster_bytes = fenster_bytes
        self._plan: Optional[Plan] = None

    # -- public -----------------------------------------------------------

    def plan(self) -> Plan:
        if self._plan is None:
            self._plan = self._build()
        return self._plan

    # -- internal -----------------------------------------------------------

    def _build(self) -> Plan:
        import torch.distributed as dist

        c = self.config.kollektiv
        bdfs, namen = self._cards()
        fa = fingerprint(bdfs, namen, self.config)

        if c.planer == "aus":
            m = self._synthetic_measurement(bdfs, namen)
            p = plan_collective(m, self.config, source="fest",
                      fenster_bytes=self.fenster_bytes)
            self._check_uniform(p)
            return p

        pfad = c.mess.cache or _default_cache()
        m = None
        if not c.mess.cache_aus:
            # Only rank 0 reads and distributes -- otherwise two ranks
            # could see differently stale files and plan differently.
            traeger: list[Any] = [None]
            if self.rank == 0:
                gefunden = read_cache(pfad, fa)
                traeger = [gefunden.as_dict() if gefunden is not None else None]
            dist.broadcast_object_list(
                traeger, src=dist.get_global_rank(self.cpu_group, 0),
                group=self.cpu_group,
            )
            if traeger[0] is not None:
                m = Measurement.from_dict(traeger[0])
                logger.info(
                    "barlink-Matrix: startup measurement skipped, cache %s "
                    "matches (fingerprint %s). Force a re-measure with "
                    "SGLANG_BARLINK_MATRIX_CACHE_OFF=1.", pfad, fa,
                )

        source = "zwischenspeicher"
        if m is None:
            if c.planer == "fest":
                # 'fest' deliberately means: don't measure, use the stored
                # result. If it's missing, that's a named error -- quietly
                # measuring anyway would be exactly the silent fallback
                # this design forbids.
                raise ConfigError(
                    f"kollektiv.planer=fest, but there is no result under "
                    f"{pfad!r} with fingerprint {fa} (card list, PCI "
                    f"addresses, driver version, patch state, measurement "
                    f"parameters). Either measure once with planer=auto, or "
                    f"point the path via SGLANG_BARLINK_MATRIX_CACHE at a "
                    f"valid result."
                )
            source = "gemessen"
            m = self._measure(bdfs, namen)
            if not c.mess.cache_aus and self.rank == 0:
                write_cache(pfad, fa, m)

        p = plan_collective(m, self.config, source=source,
                  fenster_bytes=self.fenster_bytes)
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
        gesammelt: list[Any] = [None] * self.welt
        dist.all_gather_object(gesammelt, (bdf, name), group=self.cpu_group)
        bdfs = tuple(str(x[0]) for x in gesammelt)   # type: ignore[index]
        namen = tuple(str(x[1]) for x in gesammelt)  # type: ignore[index]
        if len(set(bdfs)) != len(bdfs):
            logger.warning(
                "barlink-Matrix: duplicate PCI addresses %s. Roles and "
                "domains are addressed via the PCI address; with "
                "duplicates a configuration cannot be applied "
                "unambiguously.", bdfs,
            )
        return bdfs, namen

    # -- measurement ------------------------------------------------------------

    def _measure(self, bdfs, namen) -> Measurement:
        import torch.distributed as dist

        c = self.config.kollektiv
        mk = self._adjust_budget(c.mess)
        groessen = tuple(g * 1024 for g in mk.groessen_kib)
        f = self._fuehler if self._fuehler is not None else SelfLoadSensor(
            self.device, max_bytes=max(groessen),
            wiederholungen=mk.wiederholungen,
        )
        t_start = time.perf_counter()
        m = Measurement(
            welt=self.welt, groessen=groessen, bdfs=bdfs, namen=namen,
            fuehler=f.name(),
        )

        # -- Phase 1: self-load, staggered. --------------------------------
        # Sequential, not simultaneous: otherwise R cards compete for the
        # same host memory and the numbers end up measuring each other.
        eigen_aus: list[float] = []
        eigen_ein: list[float] = []
        eigen_dup: list[float] = []
        for besitzer in range(self.welt):
            dist.barrier(group=self.cpu_group)
            if besitzer == self.rank:
                for g in groessen:
                    eigen_aus.append(_quant(f.self_load(g, "aus")))
                    eigen_ein.append(_quant(f.self_load(g, "ein")))
                if mk.duplex:
                    for g in groessen:
                        d = f.self_load_duplex(g)
                        eigen_dup.append(_quant(d) if d is not None else 0.0)
            dist.barrier(group=self.cpu_group)

        gesammelt: list[Any] = [None] * self.welt
        dist.all_gather_object(
            gesammelt, (eigen_aus, eigen_ein, eigen_dup), group=self.cpu_group
        )
        for r, (a, e, d) in enumerate(gesammelt):    # type: ignore[misc]
            m.aus[r] = list(a)
            m.ein[r] = list(e)
            if d and any(x > 0 for x in d):
                m.duplex_summe[r] = list(d)
            m.latenz_s[r] = _latency_fit(groessen, list(a))

        # -- Phase 2: real edge measurement, if a pair sensor is present. ----
        hat_paar = self._has_pair_sensor(f, groessen)
        if hat_paar:
            for von in range(self.welt):
                for nach in range(self.welt):
                    if von == nach:
                        continue
                    dist.barrier(group=self.cpu_group)
                    werte: list[float] = []
                    if self.rank == von:
                        for g in groessen:
                            r = f.pair(nach, g)
                            werte.append(_quant(r) if r is not None else 0.0)
                    elif self.rank == nach:
                        for g in groessen:
                            f.pair_receive(von, g)
                    dist.barrier(group=self.cpu_group)
                    traeger: list[Any] = [werte if self.rank == von else None]
                    dist.broadcast_object_list(
                        traeger, src=dist.get_global_rank(self.cpu_group, von),
                        group=self.cpu_group,
                    )
                    if traeger[0]:
                        m.kante[(von, nach)] = list(traeger[0])

            # -- Phase 3: fan-in (R3). All sources into the destination at once. --
            if mk.fanin:
                for ziel in range(self.welt):
                    dist.barrier(group=self.cpu_group)
                    werte = []
                    if self.rank != ziel:
                        for g in groessen:
                            r = f.pair(ziel, g)
                            werte.append(_quant(r) if r is not None else 0.0)
                    else:
                        for g in groessen:
                            f.pair_receive(-1, g)
                    dist.barrier(group=self.cpu_group)
                    alle: list[Any] = [None] * self.welt
                    dist.all_gather_object(alle, werte, group=self.cpu_group)
                    cap = []
                    for gi in range(len(groessen)):
                        s = sum(v[gi] for r, v in enumerate(alle) if r != ziel and v)
                        cap.append(_quant(s))
                    m.fanin_deckel[ziel] = cap
                    for r, v in enumerate(alle):
                        if r != ziel and v:
                            m.fanin_anteile[(r, ziel)] = list(v)
                m.hinweise.append(
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
            # Each round, rank i writes to rank (i+versatz) mod R. This is
            # a perfect pairing: everyone sends exactly once and receives
            # exactly once, so there is no fan-in contention -- what's
            # measured here is exclusively the shared bottleneck, not the
            # cap.
            gemeinsam: dict[tuple[int, int], list[float]] = {}
            for versatz in range(1, self.welt):
                nach = (self.rank + versatz) % self.welt
                dist.barrier(group=self.cpu_group)
                meine: list[float] = []
                for g in groessen:
                    r = f.pair(nach, g)
                    meine.append(_quant(r) if r is not None else 0.0)
                gesammelt2: list[Any] = [None] * self.welt
                dist.all_gather_object(gesammelt2, meine, group=self.cpu_group)
                for von, werte2 in enumerate(gesammelt2):
                    if werte2:
                        gemeinsam[(von, (von + versatz) % self.welt)] = list(werte2)

            faktoren = []
            for gi in range(len(groessen)):
                # Worst-case degradation across all edges: the mesh step
                # is only as fast as its slowest edge.
                schlimmster = 1.0
                for (von, nach), raten in gemeinsam.items():
                    allein = m.capacity(von, nach, gi)
                    zus = raten[gi]
                    if allein > 0 and zus > 0:
                        schlimmster = max(schlimmster, allein / zus)
                faktoren.append(_quant(schlimmster))
            m.netz_faktor = [max(1.0, x) for x in faktoren]
            m.netz_faktor_gemessen = True
        else:
            m.hinweise.append(
                "No pair sensor: edge capacities are estimated from "
                "self-load (min(outbound, inbound)) and are therefore an "
                "UPPER BOUND -- shared bottlenecks such as a switch uplink "
                "or a second root complex are not visible in them. The "
                "fan-in cap is the inbound rate, not a measured BAR1 cap."
            )
            m.netz_faktor = [1.0] * len(groessen)
            m.netz_faktor_gemessen = False
            m.hinweise.append(
                "SHARED BOTTLENECKS UNMEASURED (netz_faktor set to 1.0). "
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

        ueber = os.environ.get(_ENV_PRAEFIX + "NETZ_FAKTOR")
        if ueber:
            werte = [float(x) for x in ueber.replace(";", ",").split(",")]
            if len(werte) == 1:
                werte = werte * len(groessen)
            if len(werte) != len(groessen):
                raise ConfigError(
                    f"SGLANG_BARLINK_MESH_FACTOR={ueber!r}: expected one "
                    f"value or {len(groessen)} values (per size "
                    f"{[g // 1024 for g in groessen]} KiB)."
                )
            m.netz_faktor = [max(1.0, x) for x in werte]
            m.netz_faktor_gemessen = False
            m.hinweise.append(
                f"netz_faktor manually set to {m.netz_faktor} "
                f"(SGLANG_BARLINK_MESH_FACTOR) -- not measured."
            )

        m.dauer_s = time.perf_counter() - t_start
        self._plausible(m)
        return m

    def _has_pair_sensor(self, f: Sensor, groessen) -> bool:
        """Decide group-wide, not per rank.

        A rank that believes it's measuring while the others are waiting
        at the barrier wedges the startup. Hence: ask everyone, and only
        measure if ALL of them can.
        """
        import torch.distributed as dist

        try:
            kann = f.pair(-1, groessen[0]) is not None
        except NotImplementedError:
            kann = False
        except Exception as e:
            logger.warning("barlink-Matrix: pair sensor reports an error (%s); "
                           "using the self-load estimate.", e)
            kann = False
        alle: list[Any] = [None] * self.welt
        dist.all_gather_object(alle, bool(kann), group=self.cpu_group)
        return all(bool(x) for x in alle)

    def _adjust_budget(self, mk: MeasureConfig) -> MeasureConfig:
        """Rough pre-estimate against ``budget_ms``.

        Cost grows with R^2 once a pair sensor is present. Rather than
        letting startup time explode, the size ladder is thinned first
        (the middle size goes first -- it discriminates the least), then
        the repetition count. Whatever was thinned out is logged; there
        is no such thing as quietly measuring on thinner evidence.
        """
        paare = self.welt * (self.welt - 1)
        # Round count per phase:
        #   1 self-load, staggered           -> R
        #   2 edges individually (pair sensor) -> R(R-1)
        #   3 fan-in, one destination per round -> R
        #   4 all pairs simultaneously        -> R-1
        # Without a pair sensor, 2 through 4 are skipped.
        mit_paar = self._fuehler is not None
        runden = self.welt + (paare + self.welt + self.welt - 1 if mit_paar else 0)

        def estimate(g_kib: Sequence[int], reps: int) -> float:
            # ~6 GB/s as a rough assumption for the pre-estimate; it only
            # decides how much gets measured, never what comes out of it.
            uebertragung = sum((g * 1024) * reps / 6e9 * 1000 for g in g_kib)
            # 2 barriers per round, gloo roughly 0.3 ms.
            return runden * (uebertragung + 0.6 * len(g_kib))

        g = list(mk.groessen_kib)
        reps = mk.wiederholungen
        gekuerzt = []
        while estimate(g, reps) > mk.budget_ms and len(g) > 2:
            weg = g.pop(len(g) // 2)
            gekuerzt.append(f"{weg} KiB")
        while estimate(g, reps) > mk.budget_ms and reps > 4:
            reps //= 2
        if gekuerzt or reps != mk.wiederholungen:
            logger.info(
                "barlink-Matrix: measurement budget %.0f ms -- sizes %s "
                "removed, repetitions %d -> %d. Measure fully with "
                "SGLANG_BARLINK_MEASURE_BUDGET_MS=<more>.",
                mk.budget_ms, gekuerzt or "none", mk.wiederholungen, reps,
            )
        return replace(mk, groessen_kib=tuple(g), wiederholungen=reps)

    def _plausible(self, m: Measurement) -> None:
        """Reports deviations from this rig's reference values.

        No abort and no decision -- just a note, so that a broken
        measurement setup stands out before it passes as a rig quirk.
        Fourteen plausible assumptions have already failed against the
        hardware in this project; a silent measurement would be the
        fifteenth.
        """
        gi = len(m.groessen) - 1
        for r in range(m.welt):
            if m.aus[r][gi] <= 0.05 or m.ein[r][gi] <= 0.05:
                m.hinweise.append(
                    f"Rank {r} ({m.bdfs[r]}): measured rate near zero "
                    f"(aus={m.aus[r][gi]:.2f}, ein={m.ein[r][gi]:.2f}) -- "
                    f"this is a measurement error, not a property of the "
                    f"card."
                )
            if m.duplex_summe:
                f = m.duplex_factor(r, gi)
                if f > 1.9:
                    m.hinweise.append(
                        f"Rank {r}: duplex factor {f:.2f} at the largest "
                        f"size. Measured on this rig was "
                        f"{_BELEG_DUPLEX_SUMME_1MIB:.2f}; a value close to "
                        f"2 suggests the streams weren't really running "
                        f"simultaneously."
                    )
        if m.fanin_deckel:
            for j, d in m.fanin_deckel.items():
                if d[gi] > 2.0 * m.ein[j][gi] and m.ein[j][gi] > 0:
                    m.hinweise.append(
                        f"Destination {j}: measured fan-in cap "
                        f"{d[gi]:.2f} GB/s lies far above the inbound rate "
                        f"{m.ein[j][gi]:.2f} GB/s -- check whether the "
                        f"sources really ran simultaneously."
                    )

    def _synthetic_measurement(self, bdfs, namen) -> Measurement:
        """``planer: aus`` -- measure nothing, take everything from the configuration.

        Evenly-distributed placeholder rates, so that roles come
        exclusively from ``roles``/``domaenen``. The plan reports
        ``source=fest``, so the explanation doesn't give the impression
        that anything was actually measured here.
        """
        groessen = tuple(g * 1024 for g in self.config.kollektiv.mess.groessen_kib)
        m = Measurement(welt=self.welt, groessen=groessen, bdfs=bdfs, namen=namen,
                    fuehler="none (planer=aus)")
        for r in range(self.welt):
            m.aus[r] = [1.0] * len(groessen)
            m.ein[r] = [1.0] * len(groessen)
            m.latenz_s[r] = 0.0
        m.hinweise.append(
            "planer=aus: nothing measured. Roles, domains, and algorithm "
            "come exclusively from the configuration."
        )
        return m

    def _check_uniform(self, p: Plan) -> None:
        """The plan MUST be the same on every rank."""
        import torch.distributed as dist

        summen: list[Any] = [None] * self.welt
        dist.all_gather_object(summen, p.checksum(), group=self.cpu_group)
        if len(set(summen)) != 1:
            abweichler = {r: s for r, s in enumerate(summen) if s != summen[0]}
            raise RuntimeError(
                f"barlink-Matrix: ranks ended up with DIFFERENT plans "
                f"({summen}). Divergent ranks: {abweichler}. This is a "
                f"startup error, not a warning -- the collectives assume "
                f"that every rank runs the same decomposition. Most common "
                f"causes: an SGLANG_BARLINK_* variable or a config file "
                f"isn't the same on every rank, or fenster_bytes was "
                f"passed in differently per rank (it must be the minimum "
                f"across all destinations -- what matters is what can be "
                f"mapped EVERYWHERE)."
            )


def _latency_fit(groessen: Sequence[int], raten: Sequence[float]) -> float:
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
    ueber = os.environ.get(_ENV_PRAEFIX + "SCHRITT_US")
    if ueber:
        return float(ueber) / 1e6
    punkte = [
        (float(n), n / (r * 1e9)) for n, r in zip(groessen, raten) if r > 0
    ]
    if len(punkte) < 2:
        return 0.0
    n_mittel = sum(x for x, _ in punkte) / len(punkte)
    t_mittel = sum(y for _, y in punkte) / len(punkte)
    zaehler = sum((x - n_mittel) * (y - t_mittel) for x, y in punkte)
    nenner = sum((x - n_mittel) ** 2 for x, _ in punkte)
    if nenner <= 0:
        return 0.0
    steigung = zaehler / nenner
    achsenabschnitt = t_mittel - steigung * n_mittel
    return max(0.0, achsenabschnitt)


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
        if isinstance(bus, str) and _IST_BDF.match(bus.strip()):
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
        puffer = ctypes.create_string_buffer(32)
        fn.restype = ctypes.c_int
        if fn(puffer, ctypes.c_int(32), ctypes.c_int(int(ordinal))) == 0:
            return _norm_bdf(puffer.value.decode())
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
        basis = f"/sys/bus/pci/devices/{bdf}"
        with open(f"{basis}/current_link_width") as f:
            breite = int(f.read().strip())
        with open(f"{basis}/current_link_speed") as f:
            tempo = f.read().strip()
        return breite, tempo
    except (OSError, ValueError):
        return None
