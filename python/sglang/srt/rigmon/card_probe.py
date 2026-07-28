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
"""The short card probe: measured card rates + the ORDERED pair matrix.

``probe.py`` defines what a short probe IS (:class:`~sglang.srt.rigmon.probe.
CardRate`, :class:`~sglang.srt.rigmon.probe.LinkRate`, the 30 s budget) and
reads the stage-0 profile ``uneven_perf`` already writes. What neither of them
does is RUN the measurement for the quantities the planner's speed advice
actually needs. This module is that runner, and it is deliberately a binder
rather than a third probe:

* **membw and bf16 GEMM come from** ``uneven_perf._bench_membw_rates`` /
  ``_bench_gemm_tflops`` — the same kernels the boot path calibrates against,
  called directly. Two probes that measure the same card with different
  kernels would disagree, and the disagreement would be invisible.
* **The GEMV rate** (task #231's decode divisor) is one of those three membw
  rates and is carried through unchanged, so the decode roofline and the
  dashboard quote one number, not two.
* **fp8 GEMM, H2D, D2H, and the ordered pair matrix** are added here, because
  nothing measures them today.

**Ordered, not symmetric.** ``uneven_perf`` stores one entry per unordered
pair. A route can be asymmetric — on this rig GPU0 sits on a x4 link, so it
uploads and downloads differently from a card on a x16 slot — so both
directions are measured and each carries its own number.

**The path is part of the number.** A pair figure without the path it took is
not comparable with anything. Every link records ``transport``:

``cuda p2p``
    ``can_device_access_peer`` is true both ways and the copy crossed the bus
    directly.
``host staging (pinned)``
    No P2P. The bytes went device -> pinned host buffer -> device, which is
    what the driver does underneath a cross-device copy anyway; measuring it
    explicitly means the number names its own path instead of hiding it.

On the reference rig no pair has P2P (chipset, no NVLink), so the matrix it
produces is a host-staging matrix and says so in every row. That is not a
degraded measurement — it is the transfer rate this hardware actually offers,
and a placement decision made against a nameplate P2P figure would be wrong.

**fp8 where it compiles, absent where it does not.** ``gemm_fp8_tflops`` is
measured through ``torch._scaled_mm`` on cards that have an fp8 tensor path
(sm89+); on Ampere it is ``None`` with ``fp8_note`` saying why. An estimated
fp8 number would be worse than none: the whole point of the probe is that the
planner stops ranking cards off a datasheet.

**State travels with the numbers.** Driver version, SM clock, clock ceiling,
temperature and active throttle reasons are captured per card at measurement
time, following the tagging convention from task #149. A point taken under
throttling is KEPT AND MARKED, never dropped.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "CARD_PROBE_VERSION",
    "CardProbeMeasurement",
    "PairMeasurement",
    "CardProbeProfile",
    "P2P_DIRECT",
    "HOST_STAGING",
    "card_probe_cache_path",
    "default_cache_path",
    "load_card_probe",
    "save_card_probe",
    "run_card_probe",
    "measure_card",
    "measure_pair_matrix",
    "to_probe_result",
    "measured_card_rates",
    "ProbeJob",
    "ProbeJobStore",
    "JOBS",
    "format_text",
    "PENDING",
    "RUNNING",
    "OK",
    "ERROR",
]

#: Bump when the measured FIELDS change meaning; a cached profile with a
#: different version is re-probed rather than reinterpreted.
CARD_PROBE_VERSION = 1

CACHE_DIR = os.path.expanduser("~/.cache/sglang")

#: Transport labels. The label is not decoration: a host-staging figure and a
#: p2p figure are different quantities and must never be averaged together.
P2P_DIRECT = "cuda p2p"
HOST_STAGING = "host staging (pinned)"

#: A probe older than this is reported as stale rather than used silently.
#: Same horizon as ``probe.DEFAULT_MAX_AGE_S`` -- one convention, not two.
DEFAULT_MAX_AGE_S = 7 * 24 * 3600.0

#: Transfer sizes. 64 MiB is large enough that the copy is bandwidth-bound
#: rather than launch-bound on every card here, and small enough that a
#: pinned staging buffer of that size is affordable on a 20 GB card.
_XFER_BYTES = 64 * 1024 * 1024
_XFER_ITERS = 12
_XFER_WARMUP = 3

#: fp8 GEMM shape. Same M/K/N as ``uneven_perf``'s bf16 GEMM so the two
#: numbers are directly comparable -- an fp8 figure measured at a different
#: shape would not answer "how much does fp8 buy on this card".
_FP8_M, _FP8_K, _FP8_N = 2048, 5120, 17408
_FP8_ITERS = 40
_FP8_WARMUP = 8

#: Minimum compute capability with an fp8 tensor path (Ada/Hopper and up).
_FP8_MIN_CC = (8, 9)


# ---------------------------------------------------------------------------
# Result shapes
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CardProbeMeasurement:
    """One card's measured rates, plus the state they were measured in."""

    uuid: str
    name: str
    cuda_index: int
    total_mib: Optional[int] = None

    # -- compute ---------------------------------------------------------
    gemm_bf16_tflops: Optional[float] = None
    gemm_fp8_tflops: Optional[float] = None
    #: Why fp8 is absent, when it is. Never a substitute number.
    fp8_note: str = ""

    # -- device memory (D2D) --------------------------------------------
    membw_read_gbs: Optional[float] = None
    membw_copy_gbs: Optional[float] = None
    #: The decode-shaped weight read -- task #231's divisor. Carried through
    #: from the same kernel the roofline uses; not re-derived here.
    membw_gemv_gbs: Optional[float] = None

    # -- host <-> device -------------------------------------------------
    h2d_gbs: Optional[float] = None
    d2h_gbs: Optional[float] = None

    # -- state (#149 tagging convention) ---------------------------------
    sm_clock_mhz: Optional[int] = None
    sm_clock_max_mhz: Optional[int] = None
    temp_c: Optional[float] = None
    throttle_reasons: List[str] = dataclasses.field(default_factory=list)

    seconds: Optional[float] = None

    @property
    def membw_gbs(self) -> Optional[float]:
        """The card's streaming bandwidth score -- the best a pure stream
        reaches. Kept as a derived property so the three rates stay apart in
        storage; collapsing them at write time would lose the GEMV rate at
        exactly the point it is needed."""
        vals = [v for v in (self.membw_read_gbs, self.membw_copy_gbs) if v]
        return max(vals) if vals else None

    @property
    def throttled(self) -> bool:
        return bool(self.throttle_reasons)

    @property
    def clock_ratio(self) -> Optional[float]:
        if not self.sm_clock_mhz or not self.sm_clock_max_mhz:
            return None
        return self.sm_clock_mhz / self.sm_clock_max_mhz

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["membw_gbs"] = self.membw_gbs
        d["throttled"] = self.throttled
        d["clock_ratio"] = self.clock_ratio
        return d

    @classmethod
    def from_json(cls, d: dict) -> "CardProbeMeasurement":
        known = {f.name for f in dataclasses.fields(cls)}
        d = {k: v for k, v in dict(d).items() if k in known}
        d["throttle_reasons"] = list(d.get("throttle_reasons") or [])
        return cls(**d)


@dataclasses.dataclass
class PairMeasurement:
    """One ORDERED pair, src -> dst, with the path it was measured over."""

    src_uuid: str
    dst_uuid: str
    bandwidth_gbs: Optional[float] = None
    latency_us: Optional[float] = None
    transport: str = HOST_STAGING
    #: Whether the driver reports peer access in THIS direction. Recorded even
    #: when the copy ran over host staging, because "no p2p" is the finding.
    peer_access: bool = False
    bytes_moved: int = _XFER_BYTES
    note: str = ""

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "PairMeasurement":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in dict(d).items() if k in known})


@dataclasses.dataclass
class CardProbeProfile:
    """Everything one probe run produced, with its age and its context."""

    version: int = CARD_PROBE_VERSION
    created: float = 0.0
    created_str: str = ""
    duration_s: Optional[float] = None
    driver: Optional[str] = None
    torch_version: Optional[str] = None
    cuda_version: Optional[str] = None
    node_id: str = "local"
    cards: List[CardProbeMeasurement] = dataclasses.field(default_factory=list)
    pairs: List[PairMeasurement] = dataclasses.field(default_factory=list)
    notes: List[str] = dataclasses.field(default_factory=list)

    # -- lookup ----------------------------------------------------------

    def by_uuid(self) -> Dict[str, CardProbeMeasurement]:
        return {c.uuid: c for c in self.cards}

    def pair(self, src: str, dst: str) -> Optional[PairMeasurement]:
        for p in self.pairs:
            if p.src_uuid == src and p.dst_uuid == dst:
                return p
        return None

    def age_s(self, now: Optional[float] = None) -> Optional[float]:
        if not self.created:
            return None
        return max(0.0, (now if now is not None else time.time()) - self.created)

    def is_stale(
        self, now: Optional[float] = None, max_age_s: float = DEFAULT_MAX_AGE_S
    ) -> bool:
        age = self.age_s(now)
        return age is not None and age > max_age_s

    @property
    def transports(self) -> List[str]:
        return sorted({p.transport for p in self.pairs if p.transport})

    def rate_caveats(self) -> List[str]:
        """The caveats that are about the RATES rather than the run.

        Kept separate because ``probe.ProbeResult`` derives its own warnings
        about throttling, staleness and mirrored pairs from the same facts.
        Anything projected onto that shape must carry only what it cannot
        derive, or every warning is printed twice.
        """
        out: List[str] = []
        missing_fp8 = [c for c in self.cards if c.gemm_fp8_tflops is None]
        if missing_fp8 and len(missing_fp8) != len(self.cards):
            names = ", ".join(sorted({c.name for c in missing_fp8}))
            out.append(
                f"No fp8 GEMM rate for {names}: "
                f"{missing_fp8[0].fp8_note or 'not measurable'}. "
                "Rank these cards by their bf16 figure; an fp8 plan cannot be "
                "priced on them."
            )
        if HOST_STAGING in self.transports and P2P_DIRECT not in self.transports:
            out.append(
                "No pair on this host has peer access, so the pair matrix is a "
                "HOST-STAGING matrix (device -> pinned host -> device). It is "
                "the transfer rate this hardware actually offers, not a "
                "degraded stand-in for a p2p number."
            )
        return out

    def caveats(self, now: Optional[float] = None) -> List[str]:
        """Everything that changes how these numbers must be read, as data.

        The surface has to show it: a recommendation derived from a throttled
        or stale probe carries that provenance for as long as it is on screen.
        """
        out: List[str] = []
        for c in self.cards:
            if c.throttled:
                ratio = c.clock_ratio
                at = f" at {ratio * 100:.0f} % of its maximum clock" if ratio else ""
                out.append(
                    f"{c.name} was throttled while measured "
                    f"({', '.join(c.throttle_reasons)}){at}. The point is kept "
                    "and marked; any rate derived from it understates this card."
                )
        out.extend(self.rate_caveats())
        if self.is_stale(now):
            age = self.age_s(now) or 0.0
            out.append(
                f"This probe is {age / 3600:.0f} h old. Driver, clocks and "
                "thermal conditions drift; re-probe before deriving a "
                "configuration from it."
            )
        return out

    def to_json(self) -> dict:
        return {
            "version": self.version,
            "created": self.created,
            "created_str": self.created_str,
            "duration_s": self.duration_s,
            "driver": self.driver,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "node_id": self.node_id,
            "uuids": sorted(c.uuid for c in self.cards),
            "cards": [c.to_json() for c in self.cards],
            "pairs": [p.to_json() for p in self.pairs],
            "transports": self.transports,
            "notes": list(self.notes),
            "caveats": self.caveats(),
        }

    @classmethod
    def from_json(cls, d: dict) -> "CardProbeProfile":
        return cls(
            version=int(d.get("version", 0)),
            created=float(d.get("created") or 0.0),
            created_str=str(d.get("created_str") or ""),
            duration_s=d.get("duration_s"),
            driver=d.get("driver"),
            torch_version=d.get("torch_version"),
            cuda_version=d.get("cuda_version"),
            node_id=str(d.get("node_id") or "local"),
            cards=[CardProbeMeasurement.from_json(c) for c in d.get("cards") or []],
            pairs=[PairMeasurement.from_json(p) for p in d.get("pairs") or []],
            notes=list(d.get("notes") or []),
        )


# ---------------------------------------------------------------------------
# Persistence (the ~/.cache/sglang convention, atomic replace)
# ---------------------------------------------------------------------------


def card_probe_cache_path(uuids: Sequence[str], driver: Optional[str]) -> str:
    """Cache key = (sorted card UUIDs, driver version, probe version).

    Keyed on the driver on purpose: a driver update moves clock behaviour and
    the p2p verdict, and silently reusing rates across one is how a stale
    number outlives the hardware state it described.
    """
    key = json.dumps([sorted(uuids), driver or "", CARD_PROBE_VERSION])
    digest = hashlib.sha1(key.encode()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"card_probe-{digest}.json")


def default_cache_path() -> Optional[str]:
    """The cache path for the cards visible right now, or None when the
    inventory cannot be read (no NVML / no CUDA)."""
    try:
        gpus, driver = _inventory()
    except Exception:
        return None
    if not gpus:
        return None
    return card_probe_cache_path([g["uuid"] for g in gpus], driver)


def save_card_probe(profile: CardProbeProfile, path: Optional[str] = None) -> str:
    path = path or card_probe_cache_path(
        [c.uuid for c in profile.cards], profile.driver
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(profile.to_json(), f, indent=1)
    os.replace(tmp, path)
    return path


def load_card_probe(path: Optional[str] = None) -> Optional[CardProbeProfile]:
    """Load the cached probe for the cards visible now, or None.

    NEVER triggers a measurement: every consumer of this is a read path
    (placement, lever profiles, the dashboard), and a multi-second GPU probe
    as a side effect of rendering a page would be a surprise.
    """
    path = path or default_cache_path()
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError) as e:
        logger.warning("card probe %s unreadable (%s); ignoring it.", path, e)
        return None
    if int(d.get("version", -1)) != CARD_PROBE_VERSION:
        logger.info(
            "card probe %s was written by version %s, this is version %s; "
            "ignoring it rather than reinterpreting its fields.",
            path,
            d.get("version"),
            CARD_PROBE_VERSION,
        )
        return None
    return CardProbeProfile.from_json(d)


# ---------------------------------------------------------------------------
# Inventory and state
# ---------------------------------------------------------------------------


def _inventory() -> Tuple[List[dict], Optional[str]]:
    """Per-CUDA-device {cuda_index, uuid, name, total_mib} + driver version.

    Delegates to ``uneven_perf._nvml_gpu_inventory``, which already bridges
    CUDA enumeration order to NVML's via PCI bus ids. The two orders do not
    match on this rig, and a probe that assumed they did would attach every
    measurement to the wrong card.
    """
    from sglang.srt.uneven_perf import _nvml_gpu_inventory

    return _nvml_gpu_inventory()


def _card_states() -> Dict[str, dict]:
    """UUID-keyed clock/temperature/throttle snapshot, or {} when unavailable.

    Reuses the rigmon collector backend rather than a second NVML reader, so
    "throttled" means the same thing here and on the live page.

    Sampled immediately AFTER a card's kernels, never before: an idle card
    sits near its minimum clock, and a state read one moment too early
    reports 6 % of the clock ceiling for a measurement that ran at 100 % of
    it. That is not a harmless inaccuracy — the throttle check exists to say
    whether a rate understates its card, and an idle sample makes every rate
    look throttled.
    """
    states: Dict[str, dict] = {}
    try:
        from sglang.srt.rigmon.sources import select_backend

        backend = select_backend()
        try:
            for c in backend.sample(with_profiling=False):
                if not c.uuid:
                    continue
                states[c.uuid] = {
                    "sm_clock_mhz": c.sm_clock_mhz,
                    "sm_clock_max_mhz": c.sm_clock_max_mhz,
                    "temp_c": c.temp_c,
                    "throttle_reasons": list(c.performance_throttles()),
                }
        finally:
            backend.close()
    except Exception as e:  # pragma: no cover - depends on the host
        logger.warning("card probe: no card state available (%s)", e)
    return states


# ---------------------------------------------------------------------------
# Per-card measurement
# ---------------------------------------------------------------------------


def _fp8_supported(dev) -> Tuple[bool, str]:
    """(supported, reason). The reason is what gets stored when it is False."""
    import torch

    major, minor = torch.cuda.get_device_capability(dev)
    if (major, minor) < _FP8_MIN_CC:
        return False, (
            f"compute capability {major}.{minor} has no fp8 tensor path "
            f"(needs {_FP8_MIN_CC[0]}.{_FP8_MIN_CC[1]}+)"
        )
    if not hasattr(torch, "_scaled_mm"):
        return False, "this torch build has no torch._scaled_mm"
    return True, ""


def _bench_gemm_fp8_tflops(dev) -> Tuple[Optional[float], str]:
    """fp8 e4m3 GEMM throughput via ``torch._scaled_mm``, or (None, why).

    Per-tensor scales, ``use_fast_accum=True``: that is the configuration the
    fp8 serving path uses, so the number prices the path that would actually
    run rather than a slower reference variant.
    """
    import torch

    ok, why = _fp8_supported(dev)
    if not ok:
        return None, why
    try:
        a = torch.randn(_FP8_M, _FP8_K, dtype=torch.bfloat16, device=dev).to(
            torch.float8_e4m3fn
        )
        # Column-major right operand: _scaled_mm requires mat2 to be
        # transposed-contiguous, and building it any other way fails at the
        # first call rather than measuring something.
        b = (
            torch.randn(_FP8_N, _FP8_K, dtype=torch.bfloat16, device=dev)
            .to(torch.float8_e4m3fn)
            .t()
        )
        scale = torch.ones((), dtype=torch.float32, device=dev)

        # Bound as defaults, not captured: the buffers are freed below, and a
        # late-binding closure over a deleted name is only ever a trap (the
        # same convention ``uneven_perf._bench_membw_rates`` follows).
        def fn(a=a, b=b, scale=scale):
            return torch._scaled_mm(
                a, b, scale, scale, out_dtype=torch.bfloat16, use_fast_accum=True
            )

        for _ in range(_FP8_WARMUP):
            fn()
        torch.cuda.synchronize(dev)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(torch.cuda.current_stream(dev))
        for _ in range(_FP8_ITERS):
            fn()
        e.record(torch.cuda.current_stream(dev))
        torch.cuda.synchronize(dev)
        ms = s.elapsed_time(e) / _FP8_ITERS
        flops = 2.0 * _FP8_M * _FP8_K * _FP8_N
        del a, b, scale
        torch.cuda.empty_cache()
        return flops / (ms / 1e3) / 1e12, ""
    except Exception as ex:
        torch.cuda.empty_cache()
        return None, f"fp8 GEMM did not run on this card: {type(ex).__name__}: {ex}"


def _bench_h2d_d2h(dev) -> Tuple[Optional[float], Optional[float]]:
    """Pinned-memory host<->device bandwidth (GB/s), each direction alone.

    Pinned because that is what the runtime uses for weight loading and for
    every host-staged transfer; a pageable figure would measure the copy into
    the driver's bounce buffer, not the link.
    """
    import torch

    try:
        host = torch.empty(_XFER_BYTES, dtype=torch.uint8, pin_memory=True)
        dst = torch.empty(_XFER_BYTES, dtype=torch.uint8, device=dev)
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        return None, None
    try:
        # Buffers bound as defaults; they are freed in the finally below.
        h2d = _time_copy_gbs(
            dev, lambda dst=dst, host=host: dst.copy_(host, non_blocking=False)
        )
        d2h = _time_copy_gbs(
            dev, lambda dst=dst, host=host: host.copy_(dst, non_blocking=False)
        )
        return h2d, d2h
    finally:
        del host, dst
        torch.cuda.empty_cache()


def _time_copy_gbs(dev, fn, nbytes: int = _XFER_BYTES) -> Optional[float]:
    """Best-of wall-clock GB/s for a blocking copy.

    Wall clock rather than CUDA events: a host<->device or cross-device copy
    is not confined to one device's stream, and an event pair on one device
    would time only the part of it that device saw. Best-of, because a
    scheduling hiccup in one iteration is noise, not the link.
    """
    import torch

    for _ in range(_XFER_WARMUP):
        fn()
    torch.cuda.synchronize(dev)
    best = float("inf")
    for _ in range(_XFER_ITERS):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(dev)
        best = min(best, time.perf_counter() - t0)
    if best <= 0 or best == float("inf"):
        return None
    return nbytes / 1e9 / best


def measure_card(
    cuda_index: int,
    uuid: str,
    name: str,
    total_mib: Optional[int] = None,
    state_fn=None,
) -> CardProbeMeasurement:
    """Measure one card: membw (D2D), GEMM bf16 + fp8, H2D and D2H.

    The membw and bf16 GEMM kernels are ``uneven_perf``'s, called directly.
    Re-implementing them here would produce a second opinion about the same
    card, and there is no way to tell two such opinions apart after the fact.

    ``state_fn`` is called once the kernels are done and before the card idles
    back down, so the clock and throttle state describe the run rather than
    the pause after it.
    """
    import torch

    from sglang.srt.uneven_perf import _bench_gemm_tflops, _bench_membw_rates

    t0 = time.time()
    dev = torch.device(f"cuda:{cuda_index}")
    torch.cuda.set_device(dev)

    rates = _bench_membw_rates(dev)
    gemm_bf16 = _bench_gemm_tflops(dev)
    gemm_fp8, fp8_note = _bench_gemm_fp8_tflops(dev)
    h2d, d2h = _bench_h2d_d2h(dev)

    st: dict = (state_fn or (lambda: {}))() or {}
    return CardProbeMeasurement(
        uuid=uuid,
        name=name,
        cuda_index=cuda_index,
        total_mib=total_mib,
        gemm_bf16_tflops=round(gemm_bf16, 2),
        gemm_fp8_tflops=round(gemm_fp8, 2) if gemm_fp8 is not None else None,
        fp8_note=fp8_note,
        membw_read_gbs=round(rates.read_gbs, 1),
        membw_copy_gbs=round(rates.copy_gbs, 1),
        membw_gemv_gbs=round(rates.gemv_gbs, 1),
        h2d_gbs=round(h2d, 2) if h2d is not None else None,
        d2h_gbs=round(d2h, 2) if d2h is not None else None,
        sm_clock_mhz=st.get("sm_clock_mhz"),
        sm_clock_max_mhz=st.get("sm_clock_max_mhz"),
        temp_c=st.get("temp_c"),
        throttle_reasons=list(st.get("throttle_reasons") or []),
        seconds=round(time.time() - t0, 2),
    )


# ---------------------------------------------------------------------------
# The ordered pair matrix
# ---------------------------------------------------------------------------


def _peer_ok(src: int, dst: int) -> bool:
    import torch

    try:
        return bool(torch.cuda.can_device_access_peer(src, dst))
    except Exception:
        return False


def _measure_one_pair(
    src: int, dst: int, staging=None
) -> Tuple[Optional[float], Optional[float], str, bool]:
    """(bandwidth_gbs, latency_us, transport, peer_access) for src -> dst.

    With peer access the copy is issued device-to-device and the driver keeps
    it on the bus. Without it the bytes are staged through a pinned host
    buffer explicitly, so the reported figure names the path it took instead
    of hiding a host round trip inside a one-line cross-device copy.
    """
    import torch

    peer = _peer_ok(src, dst)
    sdev = torch.device(f"cuda:{src}")
    ddev = torch.device(f"cuda:{dst}")
    a = torch.empty(_XFER_BYTES, dtype=torch.uint8, device=sdev)
    b = torch.empty(_XFER_BYTES, dtype=torch.uint8, device=ddev)
    small_src = torch.empty(4096, dtype=torch.uint8, device=sdev)
    small_dst = torch.empty(4096, dtype=torch.uint8, device=ddev)
    host = staging
    owns_host = False
    if not peer and host is None:
        host = torch.empty(_XFER_BYTES, dtype=torch.uint8, pin_memory=True)
        owns_host = True
    try:
        # Every buffer is bound as a default rather than captured: they are
        # freed in the finally below, and a closure over a deleted name is
        # only ever a trap.
        hsmall = None if peer else torch.empty(4096, dtype=torch.uint8, pin_memory=True)
        transport = P2P_DIRECT if peer else HOST_STAGING

        def copy(src, dst, stage=None):
            """One transfer, either straight across or via the staging buffer.

            One function for both paths rather than one per branch: the two
            differ only in whether the bytes stop at the host, and writing
            them twice is how the two drift apart.
            """
            if stage is None:
                dst.copy_(src, non_blocking=False)
                torch.cuda.synchronize(ddev)
                return
            stage.copy_(src, non_blocking=False)
            torch.cuda.synchronize(sdev)
            dst.copy_(stage, non_blocking=False)
            torch.cuda.synchronize(ddev)

        def big(a=a, b=b, stage=host if not peer else None):
            copy(a, b, stage)

        def small(s=small_src, d=small_dst, stage=hsmall):
            copy(s, d, stage)

        torch.cuda.set_device(sdev)
        gbs = _time_copy_gbs(sdev, big)
        # Latency at 4 kB: small enough that the number is dominated by the
        # per-transfer cost rather than the bytes, which is what a latency
        # figure is for.
        for _ in range(_XFER_WARMUP):
            small()
        best = float("inf")
        for _ in range(20):
            t0 = time.perf_counter()
            small()
            best = min(best, time.perf_counter() - t0)
        lat_us = best * 1e6
        return (
            round(gbs, 2) if gbs is not None else None,
            round(lat_us, 1),
            transport,
            peer,
        )
    finally:
        del a, b, small_src, small_dst
        if owns_host:
            del host
        torch.cuda.empty_cache()


def measure_pair_matrix(gpus: Sequence[dict]) -> List[PairMeasurement]:
    """Every ORDERED pair, both directions measured separately.

    Nothing is mirrored here. ``probe.from_hardware_profile`` has to mirror
    because the stage-0 profile only holds one direction; this probe measures
    the reverse, which is the whole reason it exists.
    """
    import torch

    out: List[PairMeasurement] = []
    staging = None
    try:
        staging = torch.empty(_XFER_BYTES, dtype=torch.uint8, pin_memory=True)
    except RuntimeError:
        staging = None
    try:
        for a in gpus:
            for b in gpus:
                if a["uuid"] == b["uuid"]:
                    continue
                gbs, lat, transport, peer = _measure_one_pair(
                    a["cuda_index"], b["cuda_index"], staging=staging
                )
                out.append(
                    PairMeasurement(
                        src_uuid=a["uuid"],
                        dst_uuid=b["uuid"],
                        bandwidth_gbs=gbs,
                        latency_us=lat,
                        transport=transport,
                        peer_access=peer,
                        note=(
                            ""
                            if peer
                            else "no peer access in this direction; bytes went "
                            "device -> pinned host -> device"
                        ),
                    )
                )
    finally:
        del staging
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run_card_probe(
    node_id: str = "local",
    include_pairs: bool = True,
    path: Optional[str] = None,
    save: bool = True,
    progress=None,
) -> CardProbeProfile:
    """Run the short probe over every visible card and cache the result.

    ``progress(done, total, label)`` is called between steps so a long-running
    endpoint can report where it is without the caller polling the GPU.
    """
    import torch

    t0 = time.time()
    gpus, driver = _inventory()

    n_pairs = len(gpus) * (len(gpus) - 1) if include_pairs else 0
    total = len(gpus) + (1 if n_pairs else 0)
    done = 0

    cards: List[CardProbeMeasurement] = []
    for g in gpus:
        if progress:
            progress(done, total, f"card {g['name']} (cuda:{g['cuda_index']})")
        cards.append(
            measure_card(
                cuda_index=g["cuda_index"],
                uuid=g["uuid"],
                name=g["name"],
                total_mib=g.get("total_mib"),
                state_fn=lambda uuid=g["uuid"]: _card_states().get(uuid),
            )
        )
        done += 1

    pairs: List[PairMeasurement] = []
    if n_pairs:
        if progress:
            progress(done, total, f"{n_pairs} ordered pairs")
        pairs = measure_pair_matrix(gpus)
        done += 1
    if progress:
        progress(done, total, "done")

    profile = CardProbeProfile(
        version=CARD_PROBE_VERSION,
        created=t0,
        created_str=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
        duration_s=round(time.time() - t0, 1),
        driver=driver,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        node_id=node_id,
        cards=cards,
        pairs=pairs,
    )
    if len(gpus) < 2:
        profile.notes.append(
            "Only one card is visible, so there is no pair matrix to measure."
        )
    if save:
        saved = save_card_probe(profile, path)
        profile.notes.append(f"cached to {saved}")
    return profile


# ---------------------------------------------------------------------------
# Binding into the probe data model
# ---------------------------------------------------------------------------


def to_probe_result(profile: CardProbeProfile):
    """Project a :class:`CardProbeProfile` onto ``probe.ProbeResult``.

    So the transport chooser, the pair-matrix renderer and the cross-rig join
    keep one input shape whether the numbers came from the stage-0 profile or
    from this probe. Every link here is ``MEASURED``: both directions were.
    """
    from sglang.srt.rigmon.probe import (
        MEASURED,
        CardRate,
        CardState,
        LinkRate,
        ProbeResult,
    )

    result = ProbeResult(
        created=profile.created,
        duration_s=profile.duration_s,
        nodes=[profile.node_id],
        # Only what ProbeResult cannot derive for itself: it produces its own
        # warnings about throttling and staleness from the state below.
        notes=list(profile.notes) + profile.rate_caveats(),
    )
    for c in profile.cards:
        result.cards.append(
            CardRate(
                node_id=profile.node_id,
                uuid=c.uuid,
                name=c.name,
                total_mib=c.total_mib,
                gemm_tflops=c.gemm_bf16_tflops,
                membw_gbs=c.membw_gbs,
                gemm_dtype="bfloat16",
                gemm_fp8_tflops=c.gemm_fp8_tflops,
                membw_gemv_gbs=c.membw_gemv_gbs,
                h2d_gbs=c.h2d_gbs,
                d2h_gbs=c.d2h_gbs,
                state=CardState(
                    sm_clock_mhz=c.sm_clock_mhz,
                    sm_clock_max_mhz=c.sm_clock_max_mhz,
                    temp_c=c.temp_c,
                    throttle_reasons=tuple(c.throttle_reasons),
                    sampled_at=profile.created,
                ),
            )
        )
    for p in profile.pairs:
        result.links.append(
            LinkRate(
                src=f"{profile.node_id}/{p.src_uuid}",
                dst=f"{profile.node_id}/{p.dst_uuid}",
                latency_us=p.latency_us,
                bandwidth_gbs=p.bandwidth_gbs,
                transport=p.transport,
                direction=MEASURED,
                same_node=True,
                latency_bytes=4096,
                bandwidth_bytes=p.bytes_moved,
                note=p.note,
            )
        )
    return result


def measured_card_rates(
    profile: Optional[CardProbeProfile] = None,
) -> Dict[str, Dict[str, Any]]:
    """UUID -> the measured rates, for consumers that rank cards by speed.

    Returns ``{}`` when no probe is cached, which is the signal to fall back
    to the nameplate table WITH a caveat rather than to silently substitute.
    """
    profile = profile if profile is not None else load_card_probe()
    if profile is None:
        return {}
    return {
        c.uuid: {
            "name": c.name,
            "membw_gbs": c.membw_gbs,
            "membw_gemv_gbs": c.membw_gemv_gbs,
            "gemm_bf16_tflops": c.gemm_bf16_tflops,
            "gemm_fp8_tflops": c.gemm_fp8_tflops,
            "h2d_gbs": c.h2d_gbs,
            "d2h_gbs": c.d2h_gbs,
            "throttled": c.throttled,
        }
        for c in profile.cards
    }


# ---------------------------------------------------------------------------
# The job: start, poll, never block the caller
# ---------------------------------------------------------------------------

PENDING = "pending"
RUNNING = "running"
OK = "ok"
ERROR = "error"

#: A finished job is kept this long so a page reloaded after the run still
#: finds its outcome.
JOB_TTL_S = 3600.0

#: A probe that has not finished by now is not going to; the estimate for
#: three cards is ~25 s, and the subprocess also pays torch's import.
JOB_TIMEOUT_S = 600.0


@dataclasses.dataclass
class ProbeJob:
    """One probe run, observed from outside the process that does it."""

    job_id: str
    state: str = PENDING
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    path: Optional[str] = None
    error: Optional[str] = None
    remedy: Optional[str] = None
    profile: Optional[CardProbeProfile] = None

    def to_json(self) -> dict:
        return {
            "job_id": self.job_id,
            "state": self.state,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": (
                round((self.finished_at or time.time()) - self.started_at, 1)
                if self.started_at
                else None
            ),
            "path": self.path,
            "error": self.error,
            "remedy": self.remedy,
            "profile": self.profile.to_json() if self.profile else None,
        }


class ProbeJobStore:
    """Starts probes and answers "is it done yet", nothing more.

    Two properties are the reason this exists rather than a direct call:

    * **The HTTP request returns immediately.** A ~25 s measurement held open
      across a request would time the browser out and lose the result on a
      reload; the state lives here instead of in the page.
    * **The measurement runs in a SUBPROCESS.** A probe allocates a CUDA
      context on every card, and the dashboard process must not acquire one —
      it runs next to a live server and would take VRAM from it permanently.
      This is the same isolation ``uneven_perf.get_hardware_profile`` uses.
    """

    def __init__(self):
        import threading

        self._lock = threading.Lock()
        self._jobs: Dict[str, ProbeJob] = {}
        #: Overridable for tests: run the probe inline instead of threading.
        self.synchronous = False
        #: Overridable for tests: what actually performs the run.
        self.runner = _run_probe_subprocess

    def jobs(self) -> List[ProbeJob]:
        with self._lock:
            return list(self._jobs.values())

    def get(self, job_id: str) -> Optional[ProbeJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def active(self) -> Optional[ProbeJob]:
        with self._lock:
            for j in self._jobs.values():
                if j.state in (PENDING, RUNNING):
                    return j
        return None

    def start(self, node_id: str = "local") -> ProbeJob:
        """Start a run, or hand back the one already going.

        Two concurrent probes on the same cards would measure each other, so a
        second request joins the first rather than starting a rival.
        """
        import threading
        import uuid as _uuid

        running = self.active()
        if running is not None:
            return running
        job = ProbeJob(
            job_id=_uuid.uuid4().hex[:12], state=RUNNING, started_at=time.time()
        )
        with self._lock:
            self._expire_locked(time.time())
            self._jobs[job.job_id] = job
        if self.synchronous:
            self._run(job, node_id)
        else:
            threading.Thread(target=self._run, args=(job, node_id), daemon=True).start()
        return job

    def _run(self, job: ProbeJob, node_id: str) -> None:
        try:
            profile, path = self.runner(node_id)
            with self._lock:
                job.profile = profile
                job.path = path
                job.state = OK
                job.finished_at = time.time()
        except Exception as e:
            with self._lock:
                job.state = ERROR
                job.error = f"{type(e).__name__}: {e}"
                job.remedy = (
                    "Check that the cards are visible to this process "
                    "(nvidia-smi) and that nothing else is holding all of "
                    "their memory; the probe needs a small allocation per card."
                )
                job.finished_at = time.time()

    def _expire_locked(self, now: float) -> None:
        for jid, j in list(self._jobs.items()):
            if j.finished_at and now - j.finished_at > JOB_TTL_S:
                del self._jobs[jid]


def _run_probe_subprocess(node_id: str = "local") -> Tuple[CardProbeProfile, str]:
    """Run the probe in a separate interpreter and read back what it wrote."""
    import subprocess

    path = default_cache_path()
    if not path:
        raise RuntimeError("no CUDA cards are visible, so there is nothing to probe")
    cmd = [
        sys.executable,
        "-m",
        "sglang.srt.rigmon.card_probe",
        "--run",
        "--node-id",
        node_id,
        "--out",
        path,
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=JOB_TIMEOUT_S, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "card probe failed (rc=%d): %s"
            % (proc.returncode, (proc.stderr or proc.stdout or "")[-1200:])
        )
    profile = load_card_probe(path)
    if profile is None:
        raise RuntimeError(f"card probe wrote nothing readable to {path}")
    return profile, path


#: Module-level store, mirroring ``pairing.STORE``: the dashboard is one
#: process and the job outlives the request that started it.
JOBS = ProbeJobStore()


# ---------------------------------------------------------------------------
# CLI / module entry point
# ---------------------------------------------------------------------------


def format_text(profile: CardProbeProfile) -> str:
    """The probe as a table. Every rate names its unit; every pair names its
    path."""
    lines: List[str] = []
    ctx = [f"driver {profile.driver}"] if profile.driver else []
    if profile.torch_version:
        ctx.append(f"torch {profile.torch_version}")
    if profile.cuda_version:
        ctx.append(f"cuda {profile.cuda_version}")
    lines.append(
        f"card probe  {profile.created_str}  ({profile.duration_s} s)"
        + ("  [" + ", ".join(ctx) + "]" if ctx else "")
    )
    lines.append("")
    head = (
        f"{'card':28s} {'membw':>9s} {'gemv':>9s} {'bf16':>9s} {'fp8':>9s} "
        f"{'H2D':>8s} {'D2H':>8s}  state"
    )
    lines.append(head)
    lines.append(
        f"{'':28s} {'GB/s':>9s} {'GB/s':>9s} {'TFLOP/s':>9s} {'TFLOP/s':>9s} "
        f"{'GB/s':>8s} {'GB/s':>8s}"
    )
    for c in profile.cards:
        state = []
        if c.temp_c is not None:
            state.append(f"{c.temp_c:.0f} C")
        if c.clock_ratio is not None:
            state.append(f"{c.clock_ratio * 100:.0f} % clock")
        if c.throttled:
            state.append("THROTTLED: " + ",".join(c.throttle_reasons))
        lines.append(
            f"{c.name[:28]:28s} "
            f"{_num(c.membw_gbs, 0):>9s} {_num(c.membw_gemv_gbs, 0):>9s} "
            f"{_num(c.gemm_bf16_tflops, 1):>9s} {_num(c.gemm_fp8_tflops, 1):>9s} "
            f"{_num(c.h2d_gbs, 1):>8s} {_num(c.d2h_gbs, 1):>8s}  " + ", ".join(state)
        )
    if profile.pairs:
        by_uuid = profile.by_uuid()
        lines.append("")
        lines.append("ordered pair matrix (src -> dst):")
        for pr in profile.pairs:
            src = by_uuid.get(pr.src_uuid)
            dst = by_uuid.get(pr.dst_uuid)
            lines.append(
                f"  {(src.name if src else pr.src_uuid)[:20]:20s} -> "
                f"{(dst.name if dst else pr.dst_uuid)[:20]:20s} "
                f"{_num(pr.bandwidth_gbs, 2):>8s} GB/s  "
                f"{_num(pr.latency_us, 1):>8s} us   via {pr.transport}"
            )
    for n in profile.notes:
        lines.append(f"note: {n}")
    for cav in profile.caveats():
        lines.append(f"CAVEAT: {cav}")
    return "\n".join(lines)


def _num(v: Optional[float], digits: int) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m sglang.srt.rigmon.card_probe",
        description="The short card probe: per-card rates + the ordered pair "
        "matrix. Roughly 30 s of GPU time for this rig.",
    )
    p.add_argument(
        "--run",
        action="store_true",
        help="measure now (default is to print the cached probe)",
    )
    p.add_argument("--node-id", default="local")
    p.add_argument("--out", default=None, help="cache path override")
    p.add_argument(
        "--no-pairs", action="store_true", help="cards only, skip the pair matrix"
    )
    p.add_argument("--json", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.run:
        profile = run_card_probe(
            node_id=args.node_id,
            include_pairs=not args.no_pairs,
            path=args.out,
        )
    else:
        cached = load_card_probe(args.out)
        if cached is None:
            print(
                "no cached card probe for these cards. Run it with --run "
                "(about 30 s of GPU time); until then the planner ranks cards "
                "on nameplate specs.",
                file=sys.stderr,
            )
            return 1
        profile = cached
    print(
        json.dumps(profile.to_json(), indent=1) if args.json else format_text(profile)
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(_main())
