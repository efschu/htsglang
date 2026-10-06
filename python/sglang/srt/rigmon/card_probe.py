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

**Profile arms (order 950, profile editor S2).** ``hardware_profile`` unites
this probe's caches with the stage-0 profile into one ``flliper.hardware/1``
view, so the probe also carries what that view shows per card and nothing
else measures: the int8 W8A8 rate, the NVFP4 W4A8-on-int8 rate (sm_8x only:
that is the only place the serving path takes it), the NVFP4 W4A16 Marlin
rate, and (order 1006) the NATIVE NVFP4 W4A4 rate on the FP4 tensor cores (sm_12x
only; every older card stores "no native FP4 tensor cores" as its reason) -- each through the kernel the serving path calls, at the probe shape of
``uneven_perf`` -- the 4 kB host<->device latency per card, and SM count, L2
size and compute capability from the device properties. A lane that cannot run
stores its REASON in ``lane_notes``, never a number.

**The BAR1 stretch per pair (order 1006).** ``bar1_probe`` measures it: one child
process per card, the production transport (``barlink_bar1.build_bar1`` with its
byte-level proof), the transport's own pair sensor for the rate and a 4 kB
write+sync for the latency, one directed pair at a time. The result sits in
``CardProbeProfile.bar1_pairs`` (never mixed into ``pairs``: a BAR1 write and a
host-staged copy are different quantities) and ``bar1_reason`` says why a pair
has no number. ``--no-bar1`` skips the step and then says so
(``BAR1_NOT_MEASURED``).

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

# One text for "the BAR1 stretch is not measured", shared with the hardware profile.
from sglang.srt.rigmon.hardware_profile import BAR1_NOT_MEASURED

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
    "BAR1_NOT_MEASURED",
    "lane_environment_issue",
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

#: Pipelined host staging (order 1006): chunk size, repeats (median). The serial D2H-then-H2D figure is the SUM of the two
#: one-way times; a pipelined path overlaps them, which is what the operating host-staging / barlink-host planes do.
_PIPE_CHUNK = 8 * 1024 * 1024
_PIPE_REPEATS = 7
_PIPE_WARMUP = 2
#: Pair latency (4 kB, blocking copy + sync): median of this many samples.
_PAIR_LAT_ITERS = 200

#: fp8 GEMM shape. Same M/K/N as ``uneven_perf``'s bf16 GEMM so the two
#: numbers are directly comparable -- an fp8 figure measured at a different
#: shape would not answer "how much does fp8 buy on this card".
_FP8_M, _FP8_K, _FP8_N = 2048, 5120, 17408
_FP8_ITERS = 40
_FP8_WARMUP = 8

#: Minimum compute capability with an fp8 tensor path (Ada/Hopper and up).
_FP8_MIN_CC = (8, 9)

#: Host<->device latency: one 4 kB pinned copy, MEDIAN of N (order 1006: the
#: headline is the typical latency; the minimum is stored beside it). 4 kB is
#: the same size the pair latency uses, so the card-to-host and card-to-card
#: figures are directly comparable in size (the pair latency is best-of-N).
_LAT_BYTES = 4096
_LAT_ITERS = 200

#: The NVFP4 W4A8-on-int8 kernel is the sm_8x one (``nvfp4_w4a8_int8``: "sm_86
#: W4A8 kernel for native-mixed"); native-mixed resolves it on sm_8x only. On
#: any other architecture the serving path does not take the lane, so it is
#: not asked there and the note says why.
_W4A8_CC_MAJOR = 8

#: Lane keys of ``CardProbeMeasurement.lane_notes`` / ``arm_seconds`` (the same
#: names ``uneven_perf`` gives its lanes).
LANE_INT8 = "int8_native"
LANE_W4A8_INT8 = "nvfp4_w4a8"
LANE_W4A16 = "nvfp4_marlin"
LANE_FP8 = "fp8_native"
LANE_W4A4 = "nvfp4_w4a4"

#: Native NVFP4 (W4A4) needs FP4 tensor cores: Blackwell (compute capability 10.0+). The path the serving model
#: takes on a consumer Blackwell card (sm_12x) is the fork's sm_120a CUTLASS GEMM behind ``fp4_gemm`` (what the
#: ``auto`` backend resolves there); datacenter Blackwell resolves another backend (cute-dsl) this probe does not ask.
_W4A4_MIN_MAJOR = 10
_W4A4_PROBE_MAJOR = 12



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
    #: int8 W8A8 (``sgl_kernel.int8_scaled_mm``), NVFP4 W4A8 on the int8 tensor
    #: cores (sm_8x), NVFP4 W4A16 Marlin -- prefill shape, TFLOPS (TOPS for the
    #: int8 ones). ``None`` = not measured; ``lane_notes`` says why.
    gemm_int8_tflops: Optional[float] = None
    gemm_w4a8_int8_tflops: Optional[float] = None
    gemm_w4a16_tflops: Optional[float] = None
    #: Native NVFP4 W4A4 on the FP4 tensor cores (sm_12x): the model's apply path
    #: (activation quantisation + the fork's sm_120a CUTLASS GEMM), TFLOPS.
    #: ``None`` = not measured; ``lane_notes["nvfp4_w4a4"]`` says why (a card
    #: without FP4 tensor cores has its reason there, never a number).
    gemm_w4a4_tflops: Optional[float] = None
    #: lane key -> why that lane has no number on this card.
    lane_notes: Dict[str, str] = dataclasses.field(default_factory=dict)
    #: Device properties (no kernel needed): SM count, L2 size, "M.m".
    sm_count: Optional[int] = None
    l2_mib: Optional[float] = None
    compute_capability: Optional[str] = None
    #: Wall seconds per measurement arm -- the run's duration line is written
    #: from these, not guessed.
    arm_seconds: Dict[str, float] = dataclasses.field(default_factory=dict)

    # -- device memory (D2D) --------------------------------------------
    membw_read_gbs: Optional[float] = None
    membw_copy_gbs: Optional[float] = None
    #: The decode-shaped weight read -- task #231's divisor. Carried through
    #: from the same kernel the roofline uses; not re-derived here.
    membw_gemv_gbs: Optional[float] = None

    # -- host <-> device -------------------------------------------------
    h2d_gbs: Optional[float] = None
    d2h_gbs: Optional[float] = None
    #: 4 kB pinned copy, median of N, wall clock (microseconds).
    h2d_lat_us: Optional[float] = None
    d2h_lat_us: Optional[float] = None
    #: Same measurement, the minimum of the N samples (order 1006: the headline
    #: above is the median). ``None`` in probes written before that order.
    h2d_lat_min_us: Optional[float] = None
    d2h_lat_min_us: Optional[float] = None

    # -- the card's PCIe link AS READ right after the transfer arm (order 1006; NVML, by UUID) -----
    #: Generation / width the link had under that traffic (an idle card sits trained down) and the card's maxima.
    pcie_gen_cur: Optional[int] = None
    pcie_width_cur: Optional[int] = None
    pcie_gen_max: Optional[int] = None
    pcie_width_max: Optional[int] = None

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
        d["lane_notes"] = dict(d.get("lane_notes") or {})
        d["arm_seconds"] = dict(d.get("arm_seconds") or {})
        return cls(**d)


@dataclasses.dataclass
class PairMeasurement:
    """One ORDERED pair, src -> dst, with the path it was measured over."""

    src_uuid: str
    dst_uuid: str
    #: Headline rate of the path. Host staging (order 1006): the PIPELINED rate (chunked, double buffer, D2H and H2D
    #: overlapped); ``bandwidth_serial_gbs`` is the serial D2H-then-H2D figure earlier probes stored here.
    bandwidth_gbs: Optional[float] = None
    #: Host staging only: whole copy D2H, then whole copy H2D, no overlap (the pre-1006 figure). ``None`` on every other
    #: path, and on probes written before order 1006 (whose ``bandwidth_gbs`` IS this serial figure).
    bandwidth_serial_gbs: Optional[float] = None
    latency_us: Optional[float] = None
    #: SECOND latency (order 1006): the same transfer issued back to back in a stream with one host sync at the end, so the
    #: per-operation launch + sync floor of ``latency_us`` is paid once. ``latency_device_kind`` says exactly what it is
    #: (BAR1: a write rate per 4 kB write, NOT a round trip; NCCL: ping-pong round trip / 2). ``None`` where not measured.
    latency_device_us: Optional[float] = None
    latency_device_kind: str = ""
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
    #: The BAR1 stretch per ORDERED pair (``bar1_probe``), transport label
    #: ``bar1_probe.BAR1_DIRECT``. Empty together with ``bar1_attempted=False``
    #: = the step did not run (an older probe, ``--no-bar1``, one card).
    bar1_pairs: List[PairMeasurement] = dataclasses.field(default_factory=list)
    bar1_attempted: bool = False
    #: Why at least one BAR1 pair has no number ("" = all measured).
    bar1_reason: str = ""
    bar1_seconds: Optional[float] = None
    bar1_window_mib: Optional[float] = None
    #: NCCL send/recv per ORDERED pair (``nccl_probe``): the "without barlink" reference column of the D2D table.
    nccl_pairs: List[PairMeasurement] = dataclasses.field(default_factory=list)
    nccl_attempted: bool = False
    nccl_reason: str = ""
    nccl_seconds: Optional[float] = None

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
            "bar1_pairs": [p.to_json() for p in self.bar1_pairs],
            "bar1_attempted": self.bar1_attempted,
            "bar1_reason": self.bar1_reason,
            "bar1_seconds": self.bar1_seconds,
            "bar1_window_mib": self.bar1_window_mib,
            "nccl_pairs": [p.to_json() for p in self.nccl_pairs],
            "nccl_attempted": self.nccl_attempted,
            "nccl_reason": self.nccl_reason,
            "nccl_seconds": self.nccl_seconds,
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
            bar1_pairs=[PairMeasurement.from_json(p) for p in d.get("bar1_pairs") or []],
            bar1_attempted=bool(d.get("bar1_attempted")),
            bar1_reason=str(d.get("bar1_reason") or ""),
            bar1_seconds=d.get("bar1_seconds"),
            bar1_window_mib=d.get("bar1_window_mib"),
            nccl_pairs=[PairMeasurement.from_json(p) for p in d.get("nccl_pairs") or []],
            nccl_attempted=bool(d.get("nccl_attempted")),
            nccl_reason=str(d.get("nccl_reason") or ""),
            nccl_seconds=d.get("nccl_seconds"),
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


def matching_cached_probe_json(
    *,
    cache_dir: Optional[str] = None,
    inventory: Optional[Tuple[Optional[Sequence[str]], Optional[str]]] = None,
) -> Optional[dict]:
    """The cached probe that describes THIS rig, as raw JSON, or ``None``.

    #513 (audit #506, finding A3-1): the key above is deliberate -- a driver
    update moves clock behaviour and the p2p verdict -- but the two callers
    that wanted the raw payload (``planner/solver_api.cached_card_probe`` and
    ``planner/rig_profile_source._latest_card_probe``) globbed
    ``card_probe-*.json`` and took the NEWEST by mtime. A probe taken while
    the arbiter had handed out only two of three cards, or one measured under
    a driver that has since been rolled back, then became the rig's profile
    for every later solver call. This resolves by key instead.

    #363: the key is right, the COMPARISON was not. #513 matched the probe's
    card set against the caller's for EQUALITY, which is the correct test for
    the process that WRITES a probe and the wrong one for every process that
    reads one, because a scheduler rank runs under a narrowed
    ``CUDA_VISIBLE_DEVICES`` and sees one card where the writer saw three:

        CUDA_VISIBLE_DEVICES=0,1,2 -> FOUND
        CUDA_VISIBLE_DEVICES=1     -> NONE
        CUDA_VISIBLE_DEVICES=0     -> NONE

    measured on metal in the ACT window. No rank could ever match, so the
    planner feed was permanently ``PlannerFeedUnavailable`` and every stage
    table on a real multi-rank boot read "0 flip target(s)".

    The comparison is CONTAINMENT now: a probe matches when it DESCRIBES every
    card the caller can see (``visible <= probe``), same driver. Containment is
    directional, so #513's protection still points the way #513 aimed it -- a
    two-card probe cannot serve a three-card view, because two cards do not
    describe three. What it stops refusing is the narrowed view, which is the
    only view a rank ever has.

    Matching is on UUID at every step and never on a device index: a UUID is
    the one name for a card that survives ``CUDA_VISIBLE_DEVICES`` narrowing
    unchanged, and an index is precisely what narrowing renumbers.

    The WHOLE probe is returned, not the caller's slice of it. Load-bearing:
    ``key_solver.rates_from_probe`` indexes cards by ``cuda_index`` in the
    ``--rank-gpu-id`` space (the full-inventory space), so a rank solving a
    three-rank layout needs all three cards' rates while seeing only its own.

    Preference, when several probes describe the view: an exact card-set match
    first (those cards were measured under this view's own contention), then
    the TIGHTEST superset, then newest by mtime as the tie-break. Newest is
    only ever a tie-break among probes already proven to describe the same rig.

    A miss is the correct answer when nothing matches: every consumer already
    has a "no probe" remedy path, and that includes the case where the
    inventory itself cannot be read -- a probe we cannot attribute to this rig
    is exactly what this function exists to refuse.

    ``cache_dir`` and ``inventory`` are injection points for tests; production
    callers pass neither.
    """
    directory = cache_dir or CACHE_DIR
    if inventory is None:
        try:
            gpus, driver = _inventory()
            uuids: Optional[Sequence[str]] = [g["uuid"] for g in gpus]
        except Exception:
            uuids, driver = None, None
    else:
        uuids, driver = inventory
    if not uuids:
        logger.info(
            "card probe: the live card inventory could not be read, so no "
            "cached probe can be attributed to this rig; reporting no probe "
            "rather than the newest file on disk."
        )
        return None

    want_uuids = sorted(str(u) for u in uuids)

    def _read(path: str) -> Optional[dict]:
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict) or not data.get("cards"):
            return None
        if int(data.get("version", -1)) != CARD_PROBE_VERSION:
            return None
        return data

    exact = os.path.join(
        directory, os.path.basename(card_probe_cache_path(want_uuids, driver))
    )
    if os.path.exists(exact):
        data = _read(exact)
        if data is not None:
            return dict(data)

    # Second chance by CONTENT: a file whose name was written by a different
    # path convention, and -- #363 -- any probe that DESCRIBES this view even
    # though it was written by a process that could see more cards. Still a
    # match on the rig by UUID, never on the mtime alone.
    want_set = set(want_uuids)
    candidates = []
    try:
        names = os.listdir(directory)
    except OSError:
        return None
    for name in names:
        if not (name.startswith("card_probe-") and name.endswith(".json")):
            continue
        path = os.path.join(directory, name)
        data = _read(path)
        if data is None:
            continue
        got = {str(c.get("uuid")) for c in data.get("cards") or []}
        # CONTAINMENT, not equality: every card this caller can see must be
        # described. Directional on purpose -- a probe that knows fewer cards
        # than the caller sees is still the #513 refusal.
        if not want_set <= got:
            continue
        if str(data.get("driver") or "") != str(driver or ""):
            continue
        # Rank the match: exact card set first, then the tightest superset,
        # then newest. `extra` is 0 for an exact match, so one sort key covers
        # both steps.
        extra = len(got) - len(want_set)
        candidates.append((-extra, os.path.getmtime(path), data))
    if not candidates:
        return None
    best = max(candidates, key=lambda c: (c[0], c[1]))
    if best[0] != 0:
        logger.info(
            "card probe: this process sees %d card(s); the matching probe "
            "describes %d, including all of them. Using it -- a narrowed "
            "CUDA_VISIBLE_DEVICES is a smaller VIEW of this rig, not another "
            "rig, and the solver indexes cards in the full --rank-gpu-id "
            "space regardless of what this process can see.",
            len(want_set),
            len(want_set) - best[0],
        )
    return dict(best[2])


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


def _pcie_link_of(uuid: str) -> Dict[str, Optional[int]]:
    """The PCIe link of the card ``uuid`` as NVML reports it NOW (generation / width, current and maximum), or ``{}``.

    Read right after the host-transfer arm, so the link is trained up and the "current" values describe the run, not the
    idle state (an idle card drops to Gen1). By UUID, never by index. NVML trouble is an empty dict, never an exception:
    the profile then shows the link rows as "nicht gemessen"."""
    try:
        from sglang.srt.rigmon.hardware_profile import read_nvml

        for c in read_nvml()[0]:
            if c.get("uuid") == uuid:
                return {
                    "gen_cur": c.get("pcie_cur_gen"),
                    "width_cur": c.get("pcie_cur_width"),
                    "gen_max": c.get("pcie_max_gen"),
                    "width_max": c.get("pcie_max_width"),
                }
    except Exception as ex:  # pragma: no cover - depends on the host
        logger.warning("card probe: no PCIe link state (%s)", ex)
    return {}


def _bench_h2d_d2h_latency(
    dev,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """(h2d_p50_us, d2h_p50_us, h2d_min_us, d2h_min_us): one 4 kB pinned copy
    each way, ``_LAT_ITERS`` samples, wall clock, blocking copy + device sync.

    The headline is the MEDIAN (the typical latency a staged transfer pays);
    the minimum is returned beside it so a noisy run can be told from a clean
    one. Same size and same clock as the pair latency (``_measure_one_pair``)."""
    import torch

    try:
        host = torch.empty(_LAT_BYTES, dtype=torch.uint8, pin_memory=True)
        dst = torch.empty(_LAT_BYTES, dtype=torch.uint8, device=dev)
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        return None, None, None, None

    def samples_us(fn) -> List[float]:
        for _ in range(_XFER_WARMUP):
            fn()
        torch.cuda.synchronize(dev)
        out = []
        for _ in range(_LAT_ITERS):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize(dev)
            out.append((time.perf_counter() - t0) * 1e6)
        return out

    try:
        h2d = samples_us(lambda dst=dst, host=host: dst.copy_(host, non_blocking=False))
        d2h = samples_us(lambda dst=dst, host=host: host.copy_(dst, non_blocking=False))
        return (
            round(_median(h2d), 1),
            round(_median(d2h), 1),
            round(min(h2d), 1),
            round(min(d2h), 1),
        )
    finally:
        del host, dst
        torch.cuda.empty_cache()


def _median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def _device_properties(dev) -> Tuple[Optional[int], Optional[float], Optional[str]]:
    """(sm_count, l2_mib, "M.m") from the CUDA device properties -- no kernel.

    ``L2_cache_size`` is a recent torch attribute; a build without it leaves
    the L2 figure absent rather than filled from a table."""
    import torch

    props = torch.cuda.get_device_properties(dev)
    sm = getattr(props, "multi_processor_count", None)
    l2 = getattr(props, "L2_cache_size", None)
    cc = f"{props.major}.{props.minor}"
    return (
        int(sm) if sm is not None else None,
        round(int(l2) / (1024 * 1024), 2) if l2 else None,
        cc,
    )


#: Module cache of the lane-probe environment check (same interpreter for
#: every card of one run): ``None`` = not yet asked, ``""`` = fine.
_LANE_ENV_ISSUE: Optional[str] = None


def lane_environment_issue() -> str:
    """Why THIS interpreter cannot measure the sgl_kernel-backed lanes
    honestly, or ``""``.

    The ``uneven_perf`` rule (#310): a missing or mocked ``sgl_kernel`` is a
    fact about the interpreter, not about the card, and must never be written
    into a cache keyed by card. The reason is therefore returned to the CALLER
    (logged, printed on stderr by the CLI) and the lane is left without a
    number AND without a persisted note."""
    global _LANE_ENV_ISSUE
    if _LANE_ENV_ISSUE is None:
        try:
            from sglang.srt.uneven_perf import (
                ProbeEnvironmentError,
                _check_lane_probe_environment,
            )

            try:
                _check_lane_probe_environment()
                _LANE_ENV_ISSUE = ""
            except ProbeEnvironmentError as ex:
                _LANE_ENV_ISSUE = str(ex)
        except Exception as ex:  # pragma: no cover - depends on the host
            _LANE_ENV_ISSUE = f"lane environment check unavailable: {type(ex).__name__}: {ex}"
    return _LANE_ENV_ISSUE


def _bench_gemm_int8(dev) -> Tuple[Optional[float], str]:
    """int8 W8A8 via ``sgl_kernel.int8_scaled_mm`` -- ``uneven_perf``'s lane
    probe, called as is (same kernel, same shape, same timing harness)."""
    from sglang.srt.uneven_perf import _bench_gemm_int8_native_tflops

    return _bench_gemm_int8_native_tflops(dev)


def _bench_gemm_w4a8_int8(dev) -> Tuple[Optional[float], str]:
    """NVFP4 weights on the int8 tensor cores (W4A8), or ``(None, why)``.

    Served by ``nvfp4_w4a8_linear`` (per-token int8 activation quantisation +
    the N4A tiled GEMM) from the NATIVE byte layout with the 128x4 swizzled
    block scales -- the apply path of a native-mixed sm_8x rank, built the way
    ``benchmark/nvfp4_native/n4d_decode_bench`` builds its copies. Asked only
    where the serving path takes it (sm_8x)."""
    import torch

    major, minor = torch.cuda.get_device_capability(dev)
    if major != _W4A8_CC_MAJOR:
        return None, (
            f"compute capability {major}.{minor}: the NVFP4 W4A8-on-int8 kernel is "
            f"the sm_{_W4A8_CC_MAJOR}x one (native-mixed resolves it there only); "
            "this architecture serves NVFP4 natively (W4A4) or through Marlin"
        )
    from sglang.srt import uneven_perf as up

    m, k, n = up._PROBE_GEMM_M, up._PROBE_GEMM_K, up._PROBE_GEMM_N
    w = sw = gs = x = None
    try:
        from sglang.jit_kernel.nvfp4_w4a8 import nvfp4_w4a8_linear
        from sglang.srt.layers.quantization import nvfp4_native_mixed as nm
    except Exception as ex:
        return None, f"NVFP4 W4A8 kernels unavailable: {type(ex).__name__}: {ex}"
    try:
        w = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev)
        # E4M3 scale codes 0x28..0x47 (about 0.1..1.9): finite, no NaN codes.
        raw = torch.randint(0x28, 0x48, (n, k // 16), dtype=torch.uint8, device=dev)
        sw = nm.swizzle_128x4(raw).contiguous().view(torch.float8_e4m3fn)
        del raw
        gs = torch.tensor([0.0025], dtype=torch.float32, device=dev)
        x = torch.randn(m, k, dtype=torch.bfloat16, device=dev)

        def fn(x=x, w=w, sw=sw, gs=gs, n=n):
            return nvfp4_w4a8_linear(x, w, sw, gs, n)

        fn()  # a dispatch/JIT failure must surface as a note, not in the warmup loop
        return up._time_gemm_tflops(dev, fn), ""
    except Exception as ex:
        return None, f"NVFP4 W4A8 GEMM did not run: {type(ex).__name__}: {ex}"
    finally:
        del w, sw, gs, x
        torch.cuda.empty_cache()


def _bench_gemm_w4a16(dev) -> Tuple[Optional[float], str]:
    """NVFP4 weight-only through Marlin (W4A16), or ``(None, why)``.

    The real serving helpers: ``prepare_nvfp4_layer_for_marlin`` repacks, and
    ``apply_fp4_marlin_linear`` runs it (the same pair
    ``scripts/nvfp4/phi0_lane_microbench`` and the compressed-tensors W4A4
    scheme use). Runs on every architecture -- W4A16 families take it even on
    Blackwell."""
    import torch

    from sglang.srt import uneven_perf as up

    m, k, n = up._PROBE_GEMM_M, up._PROBE_GEMM_K, up._PROBE_GEMM_N
    layer = x = None
    try:
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            apply_fp4_marlin_linear,
            prepare_nvfp4_layer_for_marlin,
        )
    except Exception as ex:
        return None, f"NVFP4 Marlin kernels unavailable: {type(ex).__name__}: {ex}"
    try:
        layer = torch.nn.Module()
        layer.params_dtype = torch.bfloat16
        layer.input_size_per_partition = k
        layer.output_size_per_partition = n
        layer.weight = torch.nn.Parameter(
            torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev),
            requires_grad=False,
        )
        layer.weight_scale = torch.nn.Parameter(
            torch.ones((n, k // 16), dtype=torch.bfloat16, device=dev).to(
                torch.float8_e4m3fn
            ),
            requires_grad=False,
        )
        layer.weight_global_scale = torch.nn.Parameter(
            torch.tensor(1.0, dtype=torch.float32, device=dev), requires_grad=False
        )
        prepare_nvfp4_layer_for_marlin(layer)
        x = torch.randn(m, k, dtype=torch.bfloat16, device=dev)

        def fn(layer=layer, x=x, n=n, k=k):
            return apply_fp4_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                weight_global_scale=layer.weight_global_scale,
                workspace=layer.workspace,
                size_n=n,
                size_k=k,
                bias=None,
            )

        fn()
        return up._time_gemm_tflops(dev, fn), ""
    except Exception as ex:
        return None, f"NVFP4 Marlin GEMM did not run: {type(ex).__name__}: {ex}"
    finally:
        del layer, x
        torch.cuda.empty_cache()


def _w4a4_unsupported_reason(cc: Optional[str]) -> str:
    """Why this card has no native NVFP4 W4A4 lane, or ``""`` when it is asked.

    A card fact (compute capability), independent of the interpreter, so the reason is stored even when the
    sgl_kernel lanes are skipped. Asked only on sm_12x, where the serving path takes the fork's sm_120a kernel."""
    try:
        major = int(str(cc).split(".")[0])
    except (TypeError, ValueError):
        return f"compute capability unknown ({cc!r}): native NVFP4 not asked"
    if major < _W4A4_MIN_MAJOR:
        return (
            f"compute capability {cc}: no native FP4 tensor cores (needs {_W4A4_MIN_MAJOR}.0+); "
            "this card serves NVFP4 through Marlin (W4A16) or the int8 cores (W4A8), not natively"
        )
    if major != _W4A4_PROBE_MAJOR:
        return (
            f"compute capability {cc}: native FP4 exists, but the kernel the serving path takes here is not the "
            f"sm_{_W4A4_PROBE_MAJOR}0a CUTLASS GEMM this probe measures; not asked on this architecture"
        )
    return ""


def _bench_gemm_w4a4_native(dev) -> Tuple[Optional[float], str]:
    """Native NVFP4 W4A4 on the FP4 tensor cores, or ``(None, why)``.

    The model's own path: ``ModelOptFp4LinearMethod.apply`` with the fork's CUTLASS backend -- per-block FP4
    activation quantisation (``fp4_quantize``), padding, then ``fp4_gemm`` -> ``cutlass_scaled_fp4_mm`` (the sm_120a
    arm; what ``--fp4-gemm-backend auto`` resolves on a consumer Blackwell). The layer is built the way
    ``benchmark/nvfp4_native/bench_5090_nvfp4`` builds it (random valid E2M1 bytes / E4M3 scales), at the probe
    shape of ``uneven_perf`` (M x 5120 x 17408: the 27B MLP width), so the figure is comparable with the other
    lanes. It INCLUDES the activation quantisation, as the W4A8 lane does; the backend global is restored."""
    import torch

    from sglang.srt import uneven_perf as up

    m, k, n = up._PROBE_GEMM_M, up._PROBE_GEMM_K, up._PROBE_GEMM_N
    layer = x = None
    changed = False
    saved = None
    fp4_utils = None
    try:
        from sglang.srt.layers.quantization import fp4_utils
        from sglang.srt.layers.quantization.modelopt_quant import (
            ModelOptFp4Config,
            ModelOptFp4LinearMethod,
        )

        if not fp4_utils.has_fork_nvfp4_cutlass_kernel():
            return None, (
                "the fork's sm_120a NVFP4 CUTLASS GEMM is not available on this device "
                "(has_fork_nvfp4_cutlass_kernel() is False)"
            )
    except Exception as ex:
        return None, f"NVFP4 native kernels unavailable: {type(ex).__name__}: {ex}"
    try:
        saved, changed = fp4_utils.FP4_GEMM_RUNNER_BACKEND, True
        fp4_utils.FP4_GEMM_RUNNER_BACKEND = fp4_utils.Fp4GemmRunnerBackend.CUTLASS
        method = ModelOptFp4LinearMethod(
            ModelOptFp4Config(is_checkpoint_nvfp4_serialized=True, group_size=16)
        )
        layer = torch.nn.Module()
        with torch.device(dev):
            method.create_weights(
                layer, k, [n], k, n, torch.bfloat16, weight_loader=None
            )
        layer.weight.data.copy_(
            torch.randint(0, 256, layer.weight.shape, dtype=torch.uint8, device=dev)
        )
        layer.weight_scale.data.copy_(
            (torch.rand(layer.weight_scale.shape, device=dev) * 2 + 0.25).to(
                torch.float8_e4m3fn
            )
        )
        layer.input_scale.data.fill_(0.01)
        layer.weight_scale_2.data.fill_(0.001)
        method.process_weights_after_loading(layer)
        x = torch.randn(m, k, dtype=torch.bfloat16, device=dev)

        def fn(method=method, layer=layer, x=x):
            return method.apply(layer, x)

        fn()  # a dispatch/JIT failure must surface as a note, not in the warmup loop
        return up._time_gemm_tflops(dev, fn), ""
    except Exception as ex:
        return None, f"NVFP4 native GEMM did not run: {type(ex).__name__}: {ex}"
    finally:
        if changed:
            fp4_utils.FP4_GEMM_RUNNER_BACKEND = saved
        del layer, x
        torch.cuda.empty_cache()


def _timed(arm_seconds: Dict[str, float], key: str, fn):
    """Run ``fn()`` and record its wall seconds under ``key``."""
    t0 = time.time()
    try:
        return fn()
    finally:
        arm_seconds[key] = round(time.time() - t0, 2)


def measure_card(
    cuda_index: int,
    uuid: str,
    name: str,
    total_mib: Optional[int] = None,
    state_fn=None,
) -> CardProbeMeasurement:
    """Measure one card: membw (D2D), GEMM bf16 + fp8 + int8 + NVFP4 lanes (W4A8, Marlin W4A16, native W4A4),
    H2D and D2H (rate and latency), and the device properties.

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

    arm: Dict[str, float] = {}
    sm_count, l2_mib, cc = _device_properties(dev)
    rates = _timed(arm, "membw", lambda: _bench_membw_rates(dev))
    gemm_bf16 = _timed(arm, "bf16", lambda: _bench_gemm_tflops(dev))
    gemm_fp8, fp8_note = _timed(arm, LANE_FP8, lambda: _bench_gemm_fp8_tflops(dev))
    h2d, d2h = _timed(arm, "h2d_d2h", lambda: _bench_h2d_d2h(dev))
    link = _timed(arm, "pcie_link", lambda: _pcie_link_of(uuid))
    h2d_lat, d2h_lat, *lat_min = _timed(
        arm, "h2d_d2h_lat", lambda: _bench_h2d_d2h_latency(dev)
    )
    h2d_lat_min, d2h_lat_min = (lat_min + [None, None])[:2]

    # The sgl_kernel-backed lanes. An interpreter that cannot measure them
    # honestly leaves them empty WITHOUT a persisted note (#310).
    lane_notes: Dict[str, str] = {}
    int8 = w4a8 = w4a16 = w4a4 = None
    # Native W4A4: the card's own verdict first (compute capability), asked only where the serving path takes it.
    w4a4_why = _w4a4_unsupported_reason(cc)
    if w4a4_why:
        lane_notes[LANE_W4A4] = w4a4_why
    env_issue = lane_environment_issue()
    if env_issue:
        logger.warning("card probe: lanes int8/w4a8/w4a16 skipped -- %s", env_issue)
    else:
        int8, note = _timed(arm, LANE_INT8, lambda: _bench_gemm_int8(dev))
        if int8 is None:
            lane_notes[LANE_INT8] = note
        w4a8, note = _timed(arm, LANE_W4A8_INT8, lambda: _bench_gemm_w4a8_int8(dev))
        if w4a8 is None:
            lane_notes[LANE_W4A8_INT8] = note
        w4a16, note = _timed(arm, LANE_W4A16, lambda: _bench_gemm_w4a16(dev))
        if w4a16 is None:
            lane_notes[LANE_W4A16] = note
        if not w4a4_why:
            w4a4, note = _timed(arm, LANE_W4A4, lambda: _bench_gemm_w4a4_native(dev))
            if w4a4 is None:
                lane_notes[LANE_W4A4] = note

    st: dict = (state_fn or (lambda: {}))() or {}
    return CardProbeMeasurement(
        uuid=uuid,
        name=name,
        cuda_index=cuda_index,
        total_mib=total_mib,
        gemm_bf16_tflops=round(gemm_bf16, 2),
        gemm_fp8_tflops=round(gemm_fp8, 2) if gemm_fp8 is not None else None,
        fp8_note=fp8_note,
        gemm_int8_tflops=round(int8, 2) if int8 is not None else None,
        gemm_w4a8_int8_tflops=round(w4a8, 2) if w4a8 is not None else None,
        gemm_w4a16_tflops=round(w4a16, 2) if w4a16 is not None else None,
        gemm_w4a4_tflops=round(w4a4, 2) if w4a4 is not None else None,
        lane_notes=lane_notes,
        sm_count=sm_count,
        l2_mib=l2_mib,
        compute_capability=cc,
        arm_seconds=arm,
        membw_read_gbs=round(rates.read_gbs, 1),
        membw_copy_gbs=round(rates.copy_gbs, 1),
        membw_gemv_gbs=round(rates.gemv_gbs, 1),
        h2d_gbs=round(h2d, 2) if h2d is not None else None,
        d2h_gbs=round(d2h, 2) if d2h is not None else None,
        h2d_lat_us=h2d_lat,
        d2h_lat_us=d2h_lat,
        h2d_lat_min_us=h2d_lat_min,
        d2h_lat_min_us=d2h_lat_min,
        pcie_gen_cur=link.get("gen_cur"),
        pcie_width_cur=link.get("width_cur"),
        pcie_gen_max=link.get("gen_max"),
        pcie_width_max=link.get("width_max"),
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


def _staged_pipelined_gbs(sdev, ddev, a, b, nbytes: int) -> Optional[float]:
    """Pipelined host staging src -> dst, GB/s (median of ``_PIPE_REPEATS``).

    The copy is cut into ``_PIPE_CHUNK`` pieces and moved through TWO pinned buffers: while chunk k travels host -> destination
    on a stream of the destination card, chunk k+1 travels source -> host on a stream of the source card. A buffer is reused
    only after the H2D that read it has finished. Wall clock around the whole copy, synchronised at both ends, so the figure is
    what one staged transfer achieves end to end -- not either direction alone."""
    import torch

    n = nbytes // _PIPE_CHUNK
    if n < 2:
        return None
    try:
        host = [torch.empty(_PIPE_CHUNK, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        return None
    s_d = torch.cuda.Stream(device=sdev)
    s_h = torch.cuda.Stream(device=ddev)

    def once() -> float:
        d_ev = [torch.cuda.Event() for _ in range(n)]
        h_ev = [torch.cuda.Event() for _ in range(n)]
        torch.cuda.synchronize(sdev)
        torch.cuda.synchronize(ddev)
        t0 = time.perf_counter()
        for k in range(n):
            if k >= 2:
                h_ev[k - 2].synchronize()          # the buffer of chunk k is free once H2D k-2 has read it
            with torch.cuda.stream(s_d):
                host[k % 2].copy_(a[k * _PIPE_CHUNK:(k + 1) * _PIPE_CHUNK], non_blocking=True)
                d_ev[k].record(s_d)
            if k >= 1:
                d_ev[k - 1].synchronize()          # chunk k-1 is in host memory: start its H2D while D2H k runs
                with torch.cuda.stream(s_h):
                    b[(k - 1) * _PIPE_CHUNK:k * _PIPE_CHUNK].copy_(host[(k - 1) % 2], non_blocking=True)
                    h_ev[k - 1].record(s_h)
        d_ev[n - 1].synchronize()
        with torch.cuda.stream(s_h):
            b[(n - 1) * _PIPE_CHUNK:n * _PIPE_CHUNK].copy_(host[(n - 1) % 2], non_blocking=True)
            h_ev[n - 1].record(s_h)
        h_ev[n - 1].synchronize()
        torch.cuda.synchronize(sdev)
        torch.cuda.synchronize(ddev)
        return time.perf_counter() - t0

    try:
        for _ in range(_PIPE_WARMUP):
            once()
        times = [once() for _ in range(_PIPE_REPEATS)]
    finally:
        del host
    med = _median(times)
    return (n * _PIPE_CHUNK) / 1e9 / med if med > 0 else None


def _measure_one_pair(
    src: int, dst: int, staging=None
) -> Tuple[Optional[float], Optional[float], str, bool, Optional[float]]:
    """(bandwidth_gbs, latency_us, transport, peer_access, bandwidth_serial_gbs) for src -> dst.

    Host staging: ``bandwidth_gbs`` is the PIPELINED rate (``_staged_pipelined_gbs``), ``bandwidth_serial_gbs`` the serial
    D2H-then-H2D figure (the sum of the two one-way times, which is what earlier probes stored as the pair rate). Over peer
    access the copy is direct and there is no serial figure (``None``).

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
        gbs = _time_copy_gbs(sdev, big)          # serial for host staging (whole D2H, then whole H2D); direct for p2p
        pipelined = None if peer else _staged_pipelined_gbs(sdev, ddev, a, b, _XFER_BYTES)
        # Latency at 4 kB: small enough that the number is dominated by the
        # per-transfer cost rather than the bytes, which is what a latency
        # figure is for. MEDIAN of 200 (order 1006; was best of 20).
        for _ in range(_XFER_WARMUP):
            small()
        lats = []
        for _ in range(_PAIR_LAT_ITERS):
            t0 = time.perf_counter()
            small()
            lats.append((time.perf_counter() - t0) * 1e6)
        lat_us = _median(lats)
        if peer:
            head, serial = gbs, None
        else:
            head, serial = (pipelined if pipelined is not None else None), gbs
        return (
            round(head, 2) if head is not None else None,
            round(lat_us, 1),
            transport,
            peer,
            round(serial, 2) if serial is not None else None,
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
                gbs, lat, transport, peer, serial = _measure_one_pair(
                    a["cuda_index"], b["cuda_index"], staging=staging
                )
                out.append(
                    PairMeasurement(
                        src_uuid=a["uuid"],
                        dst_uuid=b["uuid"],
                        bandwidth_gbs=gbs,
                        bandwidth_serial_gbs=serial,
                        latency_us=lat,
                        transport=transport,
                        peer_access=peer,
                        note=(
                            ""
                            if peer
                            else "no peer access in this direction; bytes went "
                            "device -> pinned host -> device. bandwidth_gbs = "
                            f"PIPELINED ({_PIPE_CHUNK >> 20} MiB chunks, two pinned "
                            "buffers, D2H and H2D overlapped, median of "
                            f"{_PIPE_REPEATS}); bandwidth_serial_gbs = whole copy "
                            "D2H then whole copy H2D, no overlap. latency = 4 kB "
                            f"two-hop copy with a sync after each, median of {_PAIR_LAT_ITERS}"
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


def _run_bar1_step(gpus: Sequence[dict], timeout_s: Optional[float]):
    """The BAR1 stretch in child processes; a failure is a reason, never an exception.

    The cards go to the children by UUID, in the probe's own (CUDA) order."""
    try:
        from sglang.srt.rigmon import bar1_probe

        return bar1_probe.run_bar1_probe(
            [{"uuid": g["uuid"]} for g in gpus],
            timeout_s=timeout_s or bar1_probe.DEFAULT_TIMEOUT_S,
        )
    except Exception as ex:  # noqa: BLE001 -- the card rates above are worth keeping
        from sglang.srt.rigmon.bar1_probe import Bar1Result, merge_reports

        res = merge_reports([g["uuid"] for g in gpus], [])
        res.reason = f"BAR1 step failed: {type(ex).__name__}: {ex}"
        return res


def _run_nccl_step(gpus: Sequence[dict], timeout_s: Optional[float]):
    """NCCL send/recv per ordered pair in child processes; a failure is a reason, never an exception."""
    try:
        from sglang.srt.rigmon import nccl_probe

        return nccl_probe.run_nccl_probe(
            [{"uuid": g["uuid"]} for g in gpus],
            timeout_s=timeout_s or nccl_probe.DEFAULT_TIMEOUT_S,
        )
    except Exception as ex:  # noqa: BLE001
        from sglang.srt.rigmon.nccl_probe import NcclResult

        res = NcclResult(reason=f"NCCL step failed: {type(ex).__name__}: {ex}")
        res.pairs = []
        for a in gpus:
            for b in gpus:
                if a["uuid"] != b["uuid"]:
                    res.pairs.append({"src_uuid": a["uuid"], "dst_uuid": b["uuid"], "bandwidth_gbs": None,
                                      "latency_us": None, "transport": "nccl send/recv", "peer_access": False,
                                      "bytes_moved": 0, "note": res.reason})
        return res


def run_card_probe(
    node_id: str = "local",
    include_pairs: bool = True,
    path: Optional[str] = None,
    save: bool = True,
    progress=None,
    bar1: bool = False,
    bar1_timeout_s: Optional[float] = None,
    nccl: bool = False,
    nccl_timeout_s: Optional[float] = None,
) -> CardProbeProfile:
    """Run the short probe over every visible card and cache the result.

    ``bar1`` (order 1006): with two or more cards, also measure the BAR1 stretch
    per ordered pair in child processes (``bar1_probe``); a failed or skipped
    step is recorded with its reason, it never fails the probe. Off by default
    for library callers (the planner's card-rate pass, ``rigmon`` CLI): only the
    ``--run`` command line -- the one "Hardwareprofil messen" starts -- turns it
    on (``--no-bar1`` turns it off again).

    ``nccl`` (order 1006, same rules): NCCL send/recv per ordered pair (``nccl_probe``), the "without barlink" reference
    column of the D2D table. The cache file is written after the card and pair stages and again after each optional way, so a
    run cut short by the window keeps what it measured.

    ``progress(done, total, label)`` is called between steps so a long-running
    endpoint can report where it is without the caller polling the GPU.
    """
    import torch

    t0 = time.time()
    gpus, driver = _inventory()

    n_pairs = len(gpus) * (len(gpus) - 1) if include_pairs else 0
    do_bar1 = bool(bar1 and len(gpus) >= 2)
    do_nccl = bool(nccl and len(gpus) >= 2)
    total = len(gpus) + (1 if n_pairs else 0) + (1 if do_bar1 else 0) + (1 if do_nccl else 0)
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
        # what the card and pair stages measured survives a window that ends during an optional way
        save_card_probe(profile, path)

    n_ord = len(gpus) * (len(gpus) - 1)
    bar1_res = None
    if do_bar1:
        if progress:
            progress(done, total, f"BAR1 stretch ({n_ord} ordered pairs)")
        bar1_res = _run_bar1_step(gpus, bar1_timeout_s)
        done += 1
        profile.bar1_attempted = True
        profile.bar1_pairs = [PairMeasurement.from_json(p) for p in bar1_res.pairs]
        profile.bar1_reason = bar1_res.reason
        profile.bar1_seconds = bar1_res.seconds
        profile.bar1_window_mib = bar1_res.window_mib
        if bar1_res.reason:
            profile.notes.append(f"BAR1 stretch: {bar1_res.reason}")
        profile.duration_s = round(time.time() - t0, 1)
        if save:
            save_card_probe(profile, path)
    nccl_res = None
    if do_nccl:
        if progress:
            progress(done, total, f"NCCL send/recv ({n_ord} ordered pairs)")
        nccl_res = _run_nccl_step(gpus, nccl_timeout_s)
        done += 1
        profile.nccl_attempted = True
        profile.nccl_pairs = [PairMeasurement.from_json(p) for p in nccl_res.pairs]
        profile.nccl_reason = nccl_res.reason
        profile.nccl_seconds = nccl_res.seconds
        if nccl_res.reason:
            profile.notes.append(f"NCCL send/recv: {nccl_res.reason}")
    if progress:
        progress(done, total, "done")
    profile.duration_s = round(time.time() - t0, 1)
    if len(gpus) >= 2 and bar1_res is None:
        # The BAR1 step did not run (--no-bar1): say it, the matrix has no BAR1 column.
        profile.notes.append(BAR1_NOT_MEASURED)
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
    if any(
        c.gemm_int8_tflops is not None
        or c.gemm_w4a8_int8_tflops is not None
        or c.gemm_w4a16_tflops is not None
        or c.gemm_w4a4_tflops is not None
        or c.sm_count is not None
        for c in profile.cards
    ):
        lines.append("")
        lines.append(
            f"{'card':28s} {'int8':>8s} {'w4a8':>8s} {'w4a16':>8s} {'w4a4':>8s} {'H2D lat':>8s} "
            f"{'D2H lat':>8s} {'SMs':>5s} {'L2 MiB':>7s} {'cc':>5s}"
        )
        for c in profile.cards:
            lines.append(
                f"{c.name[:28]:28s} {_num(c.gemm_int8_tflops, 1):>8s} "
                f"{_num(c.gemm_w4a8_int8_tflops, 1):>8s} "
                f"{_num(c.gemm_w4a16_tflops, 1):>8s} {_num(c.gemm_w4a4_tflops, 1):>8s} "
                f"{_num(c.h2d_lat_us, 1):>8s} "
                f"{_num(c.d2h_lat_us, 1):>8s} "
                f"{'-' if c.sm_count is None else c.sm_count:>5} "
                f"{_num(c.l2_mib, 1):>7s} {c.compute_capability or '-':>5s}"
            )
            for lane, why in sorted(c.lane_notes.items()):
                lines.append(f"    {lane}: {why}")
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
                + (
                    f"  (serial D2H+H2D {_num(pr.bandwidth_serial_gbs, 2)} GB/s)"
                    if pr.bandwidth_serial_gbs is not None
                    else ""
                )
            )
    if profile.bar1_pairs:
        by_uuid = profile.by_uuid()
        lines.append("")
        lines.append("BAR1 stretch per ordered pair (writes src -> dst's BAR1):")
        for pr in profile.bar1_pairs:
            src = by_uuid.get(pr.src_uuid)
            dst = by_uuid.get(pr.dst_uuid)
            tail = "" if pr.bandwidth_gbs is not None else f"   not measured: {pr.note}"
            lines.append(
                f"  {(src.name if src else pr.src_uuid)[:20]:20s} -> "
                f"{(dst.name if dst else pr.dst_uuid)[:20]:20s} "
                f"{_num(pr.bandwidth_gbs, 3):>8s} GB/s  "
                f"{_num(pr.latency_us, 1):>8s} us{tail}"
            )
    if profile.nccl_pairs:
        by_uuid = profile.by_uuid()
        lines.append("")
        lines.append("NCCL send/recv per ordered pair (reference, no barlink):")
        for pr in profile.nccl_pairs:
            src = by_uuid.get(pr.src_uuid)
            dst = by_uuid.get(pr.dst_uuid)
            tail = "" if pr.bandwidth_gbs is not None else f"   not measured: {pr.note}"
            lines.append(
                f"  {(src.name if src else pr.src_uuid)[:20]:20s} -> "
                f"{(dst.name if dst else pr.dst_uuid)[:20]:20s} "
                f"{_num(pr.bandwidth_gbs, 3):>8s} GB/s  "
                f"{_num(pr.latency_us, 1):>8s} us   [{pr.transport}]{tail}"
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
    p.add_argument(
        "--no-bar1",
        action="store_true",
        help="skip the BAR1 stretch per pair (child processes; needs the "
        "dmabuf_holder module and the driver's peer-BAR1 reg key)",
    )
    p.add_argument(
        "--bar1-timeout-s",
        type=float,
        default=None,
        help="wall cap of the BAR1 step (default bar1_probe.DEFAULT_TIMEOUT_S)",
    )
    p.add_argument(
        "--no-nccl",
        action="store_true",
        help="skip NCCL send/recv per ordered pair (child processes, two per pair-run; "
        "the reference column of the D2D table)",
    )
    p.add_argument(
        "--nccl-timeout-s",
        type=float,
        default=None,
        help="wall cap of the whole NCCL step (default nccl_probe.DEFAULT_TIMEOUT_S)",
    )
    p.add_argument("--json", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.run:
        profile = run_card_probe(
            node_id=args.node_id,
            include_pairs=not args.no_pairs,
            path=args.out,
            bar1=not args.no_bar1,
            bar1_timeout_s=args.bar1_timeout_s,
            nccl=not args.no_nccl,
            nccl_timeout_s=args.nccl_timeout_s,
        )
        # #310: an interpreter that cannot measure the sgl_kernel lanes says so
        # to the caller, never into the card-keyed cache.
        issue = lane_environment_issue()
        if issue:
            print(f"WARNING lanes int8/w4a8/w4a16/w4a4 not measured: {issue}", file=sys.stderr)
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
