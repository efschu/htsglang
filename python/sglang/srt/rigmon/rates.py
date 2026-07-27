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
"""Rate semantics: what is a group quantity, what is a per-rank quantity.

**Tokens are not attributable to a card.** Under tensor parallelism every rank
holds a shard of every layer and the group produces a token jointly; there is
no rank-local token count to report. Splitting group tok/s by shard size, by
FLOP share, or by anything else would produce a number that looks per-card and
means nothing. So this module publishes exactly one throughput figure —
:class:`GroupThroughput` — and it belongs to the group.

**Work and energy, however, ARE attributable**, and they carry more decision
value than a split token count would. Each card reports its own power, clock,
temperature, activity and idle share. From those come three per-rank
quantities that are measured rather than allocated (:class:`RankShare`):

1. **work share** — achieved FLOP/s and achieved bytes/s against the group sum;
2. **work per watt** — the same work against the power it cost;
3. **roofline position** — achieved FLOP/s against achieved bytes/s, i.e.
   whether this rank is compute- or bandwidth-bound in the phase running now.

The join that makes those readable is peak against achieved: the short probe
(``uneven_perf``) measures each card's PEAK GEMM throughput and PEAK memory
bandwidth; the collector measures the ACHIEVED fraction. Together they yield
the sentence DESIGN_216 asks for — "this rank runs 34 % of its memory
bandwidth and 8 % of its tensor pipe, draws 210 W and waits 46 % of the time".

Three caveats are emitted as data (:class:`Caveat`), not left in comments,
because they change how the numbers must be read:

* **Waiting costs power.** In lockstep a waiting rank still draws current, so
  a fast card looks BAD on work-per-watt. That is a symptom of imbalance, not
  of inefficiency. Active and wait share are therefore always reported
  separately, and work-per-watt is given both against total power and against
  power above the card's measured idle floor.
* **``power.draw`` is not strictly comparable across card generations** — what
  it includes varies, and on older parts it is partly estimated. Per-card
  values are shown; a group sum is marked as approximate.
* **Profiling counters are vendor-specific.** Where the fine counters are
  missing, the coarse utilization fallback is labelled as such, and any share
  derived from it inherits that label.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence

from sglang.srt.rigmon.sources import CardSample

__all__ = [
    "Caveat",
    "PeakCapability",
    "GroupThroughput",
    "RankShare",
    "RankView",
    "peaks_from_hw_profile",
    "idle_floor_from_power_profile",
    "group_throughput",
    "rank_shares",
    "pacing_rank",
]


# ---------------------------------------------------------------------------
# Caveats
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Caveat:
    """A qualification that belongs in the DISPLAY, not only in the code."""

    key: str
    text: str
    severity: str = "note"  # note | warning

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


CAVEAT_WAIT_COSTS_POWER = Caveat(
    "wait_costs_power",
    "In lockstep a waiting rank still draws power, so a fast card scores low "
    "on work per watt. That is imbalance, not inefficiency — read active and "
    "wait share alongside it.",
)
CAVEAT_POWER_COMPARABILITY = Caveat(
    "power_comparability",
    "power.draw is not strictly comparable across card generations (differing "
    "scope, partly estimated on older parts). Per-card values stand; the group "
    "sum is approximate.",
)
CAVEAT_COARSE_ACTIVITY = Caveat(
    "coarse_activity",
    "Fine profiling counters (GPM/DCGM) are unavailable on this host; SM and "
    "DRAM activity come from coarse NVML utilization, and every share derived "
    "from them is coarse too. Tensor-pipe activity has no coarse equivalent "
    "and is absent rather than approximated.",
    severity="warning",
)
CAVEAT_NO_PEAKS = Caveat(
    "no_peaks",
    "No hardware probe result available, so achieved rates cannot be put "
    "against peak capability. Run the short probe to turn activity fractions "
    "into FLOP/s and bytes/s.",
    severity="warning",
)
CAVEAT_ENGINE_WORK_ESTIMATED = Caveat(
    "engine_work_estimated",
    "Work per rank comes from the engine's own counters, derived from the "
    "shapes actually executed. That is an exact attribution of work to ranks "
    "— it is not a hardware measurement of what the silicon achieved, so it "
    "does not capture kernel inefficiency.",
)
CAVEAT_SINGLE_RANK_EXPORT = Caveat(
    "single_rank_export",
    "Only one TP rank exports metrics, so the per-rank columns repeat rank 0 "
    "instead of describing each rank. Start the server with "
    "--enable-metrics-for-all-schedulers for a real per-rank breakdown.",
    severity="warning",
)


# ---------------------------------------------------------------------------
# Peak capability (from the short probe)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PeakCapability:
    """A card's measured ceiling, from ``uneven_perf``'s stage-0 probe.

    Datasheet numbers are deliberately not accepted here: this project has
    repeatedly found nominal values misleading (GPU0 on x4, no P2P pair, NVML
    order diverging from torch order).
    """

    uuid: Optional[str]
    name: str
    gemm_tflops: Optional[float] = None
    membw_gbs: Optional[float] = None
    #: Measured idle power floor, from the energy module's power profile.
    idle_w: Optional[float] = None
    #: State the probe ran under — a probe taken while throttled understates
    #: the card, and any recommendation derived from it inherits that.
    throttled_at_probe: bool = False
    probe_created: Optional[str] = None


def peaks_from_hw_profile(
    profile: Optional[dict], power_profile: Optional[dict] = None
) -> Dict[str, PeakCapability]:
    """Build a UUID-keyed peak table from a cached ``hw_profile-*.json`` (and
    optionally ``power_profile.json`` for the idle floor)."""
    out: Dict[str, PeakCapability] = {}
    if not profile:
        return out
    idle = idle_floor_from_power_profile(power_profile)
    created = profile.get("created")
    for uuid, g in (profile.get("gpus") or {}).items():
        out[uuid] = PeakCapability(
            uuid=uuid,
            name=str(g.get("name", "unknown")),
            gemm_tflops=g.get("gemm_tflops"),
            membw_gbs=g.get("membw_gbs"),
            idle_w=idle.get(uuid),
            probe_created=created,
        )
    return out


def idle_floor_from_power_profile(power_profile: Optional[dict]) -> Dict[str, float]:
    """UUID -> measured idle watts, from the energy module's power profile."""
    out: Dict[str, float] = {}
    for c in (power_profile or {}).get("cards", []) or []:
        if c.get("uuid") and c.get("p_idle_w") is not None:
            out[str(c["uuid"])] = float(c["p_idle_w"])
    return out


# ---------------------------------------------------------------------------
# Group throughput — the one real tok/s
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class GroupThroughput:
    """Throughput of the TP GROUP. There is no per-card counterpart, by
    construction — see the module docstring."""

    gen_tok_s: Optional[float] = None
    prompt_tok_s: Optional[float] = None
    running_reqs: Optional[float] = None
    queued_reqs: Optional[float] = None
    accept_length: Optional[float] = None
    source: str = "none"
    #: Why there is no per-card breakdown. Carried so the UI can say it.
    per_card_note: str = (
        "Throughput belongs to the tensor-parallel group: all ranks compute "
        "every token together. A per-card token count would be invented. Per "
        "card, see work share, work per watt and roofline position instead."
    )

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


def group_throughput(
    metrics: Optional[Dict[str, float]],
    prev_metrics: Optional[Dict[str, float]] = None,
    dt_s: Optional[float] = None,
) -> GroupThroughput:
    """Group tok/s from engine metrics.

    Prefers differencing the monotonic ``generation_tokens_total`` counter over
    the window, because ``gen_throughput`` is an engine-internal moving average
    whose window is not the dashboard's. Falls back to the gauge and says so.
    """
    if not metrics:
        return GroupThroughput(source="engine unreachable")
    gt = GroupThroughput(
        running_reqs=metrics.get("num_running_reqs"),
        queued_reqs=metrics.get("num_queue_reqs"),
        accept_length=metrics.get("spec_accept_length"),
    )
    if prev_metrics and dt_s and dt_s > 0:
        for key, attr in (
            ("generation_tokens_total", "gen_tok_s"),
            ("prompt_tokens_total", "prompt_tok_s"),
        ):
            cur, prev = metrics.get(key), prev_metrics.get(key)
            if cur is not None and prev is not None and cur >= prev:
                setattr(gt, attr, (cur - prev) / dt_s)
        if gt.gen_tok_s is not None:
            gt.source = f"counter delta over {dt_s:.1f}s"
            return gt
    if metrics.get("gen_throughput") is not None:
        gt.gen_tok_s = metrics["gen_throughput"]
        gt.source = "engine gauge sglang:gen_throughput (engine-internal window)"
    else:
        gt.source = "no throughput metric exposed"
    return gt


# ---------------------------------------------------------------------------
# Per-rank shares — measured work, not allocated tokens
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RankShare:
    """What is genuinely per-card, with the peak/achieved join applied."""

    rank: Optional[int]
    gpu_index: int
    name: str
    uuid: Optional[str] = None

    # -- raw state ----------------------------------------------------------
    power_w: Optional[float] = None
    temp_c: Optional[float] = None
    sm_clock_mhz: Optional[int] = None
    clock_ratio: Optional[float] = None
    pstate: Optional[int] = None
    throttles: List[str] = dataclasses.field(default_factory=list)

    # -- time split (exact, from activity) ----------------------------------
    active_share: Optional[float] = None
    wait_share: Optional[float] = None

    # -- achieved vs peak ---------------------------------------------------
    membw_achieved_frac: Optional[float] = None
    tensor_achieved_frac: Optional[float] = None
    achieved_gbs: Optional[float] = None
    achieved_tflops: Optional[float] = None

    # -- group-relative -----------------------------------------------------
    byte_work_share: Optional[float] = None
    flop_work_share: Optional[float] = None

    # -- energy -------------------------------------------------------------
    idle_w: Optional[float] = None
    dynamic_w: Optional[float] = None
    gbs_per_total_w: Optional[float] = None
    gbs_per_dynamic_w: Optional[float] = None

    # -- roofline -----------------------------------------------------------
    intensity_flop_per_byte: Optional[float] = None
    balance_flop_per_byte: Optional[float] = None
    bound_by: str = "unknown"

    #: "nvml-gpm" | "nvml-utilization (coarse fallback)" | "none"
    activity_source: str = "none"
    #: Which source the work and active-share figures actually came from:
    #: "engine forward-time counter" (exact per-rank attribution), "nvml-gpm"
    #: (measured silicon), or the coarse utilization fallback.
    work_source: str = "none"

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class RankView:
    """The assembled per-rank picture plus the caveats that must be shown."""

    ranks: List[RankShare]
    caveats: List[Caveat]
    group_power_w: Optional[float] = None
    group_power_approximate: bool = True
    pacer_rank: Optional[int] = None
    pacer_basis: Optional[str] = None

    def to_json(self) -> dict:
        return {
            "ranks": [r.to_json() for r in self.ranks],
            "caveats": [c.to_json() for c in self.caveats],
            "group_power_w": self.group_power_w,
            "group_power_approximate": self.group_power_approximate,
            "pacer_rank": self.pacer_rank,
            "pacer_basis": self.pacer_basis,
        }


def engine_rank_rates(
    per_rank: Optional[Dict[int, Dict[str, float]]],
    prev_per_rank: Optional[Dict[int, Dict[str, float]]],
    dt_s: Optional[float],
) -> Dict[int, Dict[str, float]]:
    """Differentiate the engine's per-rank counters into rates.

    ``forward_execution_seconds_total`` is GPU-busy time, so its delta divided
    by wall time IS the rank's active share — an exact figure that needs no
    profiling counter and is available on any card. The estimated FLOP and byte
    counters likewise become achieved rates.
    """
    if not per_rank or not prev_per_rank or not dt_s or dt_s <= 0:
        return {}
    out: Dict[int, Dict[str, float]] = {}
    for rank, cur in per_rank.items():
        prev = prev_per_rank.get(rank)
        if not prev:
            continue
        row: Dict[str, float] = {}
        busy = cur.get("forward_execution_seconds_total")
        busy0 = prev.get("forward_execution_seconds_total")
        if busy is not None and busy0 is not None and busy >= busy0:
            row["active_share"] = max(0.0, min(1.0, (busy - busy0) / dt_s))
        flops, flops0 = (
            cur.get("estimated_flops_per_gpu_total"),
            prev.get("estimated_flops_per_gpu_total"),
        )
        if flops is not None and flops0 is not None and flops >= flops0:
            row["achieved_tflops"] = (flops - flops0) / dt_s / 1e12
        rb, rb0 = (
            cur.get("estimated_read_bytes_per_gpu_total"),
            prev.get("estimated_read_bytes_per_gpu_total"),
        )
        wb, wb0 = (
            cur.get("estimated_write_bytes_per_gpu_total"),
            prev.get("estimated_write_bytes_per_gpu_total"),
        )
        moved = 0.0
        have = False
        if rb is not None and rb0 is not None and rb >= rb0:
            moved += rb - rb0
            have = True
        if wb is not None and wb0 is not None and wb >= wb0:
            moved += wb - wb0
            have = True
        if have:
            row["achieved_gbs"] = moved / dt_s / 1e9
        if row:
            out[rank] = row
    return out


def rank_shares(
    cards: Sequence[CardSample],
    peaks: Optional[Dict[str, PeakCapability]] = None,
    rank_gpu_id: Optional[Sequence[int]] = None,
    engine_rates: Optional[Dict[int, Dict[str, float]]] = None,
    single_rank_export: bool = False,
) -> RankView:
    """Assemble the per-rank view from a card sample plus probe peaks.

    ``rank_gpu_id`` maps rank -> physical GPU index (the fork's flag). With
    co-located ranks several ranks share one card; the card's state is then
    reported for each of them and the group sums count the card once, because
    a co-located pair does not draw double the power.

    ``engine_rates`` (from :func:`engine_rank_rates`) takes precedence over the
    device counters where present. Precedence order for work and active share:
    engine counters (exact attribution) > GPM (measured silicon) > coarse NVML
    utilization (a busy-time proxy). Whichever was used is recorded per rank.
    """
    peaks = peaks or {}
    engine_rates = engine_rates or {}
    caveats: List[Caveat] = [CAVEAT_POWER_COMPARABILITY]
    if not peaks:
        caveats.append(CAVEAT_NO_PEAKS)
    if engine_rates:
        caveats.append(CAVEAT_ENGINE_WORK_ESTIMATED)
    if single_rank_export:
        caveats.append(CAVEAT_SINGLE_RANK_EXPORT)
    if not engine_rates and any(
        c.activity_source.startswith("nvml-utilization")
        or c.activity_source.startswith("nvidia-smi")
        for c in cards
    ):
        caveats.append(CAVEAT_COARSE_ACTIVITY)

    by_index = {c.index: c for c in cards}
    if rank_gpu_id:
        pairs = list(enumerate(rank_gpu_id))
    else:
        pairs = [(None, c.index) for c in cards]

    shares: List[RankShare] = []
    for rank, gpu in pairs:
        c = by_index.get(gpu)
        if c is None:
            continue
        pk = peaks.get(c.uuid or "") or PeakCapability(uuid=c.uuid, name=c.name)
        s = RankShare(
            rank=rank,
            gpu_index=c.index,
            name=c.name,
            uuid=c.uuid,
            power_w=c.power_w,
            temp_c=c.temp_c,
            sm_clock_mhz=c.sm_clock_mhz,
            clock_ratio=c.clock_ratio(),
            pstate=c.pstate,
            throttles=c.performance_throttles(),
            activity_source=c.activity_source,
            idle_w=pk.idle_w,
        )
        er = engine_rates.get(rank) if rank is not None else None
        # Time split. `sm_active` is the share of the window in which this card
        # had work resident; the remainder is where it waited on the group.
        if er and "active_share" in er:
            s.active_share = er["active_share"]
            s.wait_share = 1.0 - s.active_share
            s.work_source = "engine forward-time counter"
        elif c.sm_active is not None:
            s.active_share = max(0.0, min(1.0, c.sm_active))
            s.wait_share = 1.0 - s.active_share
            s.work_source = c.activity_source
        # Achieved rates. Engine counters win: they attribute work to the rank
        # that did it, which the device counters cannot do for co-located ranks.
        if er and "achieved_gbs" in er:
            s.achieved_gbs = er["achieved_gbs"]
            if pk.membw_gbs:
                s.membw_achieved_frac = min(1.0, s.achieved_gbs / pk.membw_gbs)
        elif c.dram_active is not None:
            s.membw_achieved_frac = max(0.0, min(1.0, c.dram_active))
            if pk.membw_gbs:
                s.achieved_gbs = s.membw_achieved_frac * pk.membw_gbs
        if er and "achieved_tflops" in er:
            s.achieved_tflops = er["achieved_tflops"]
            if pk.gemm_tflops:
                s.tensor_achieved_frac = min(1.0, s.achieved_tflops / pk.gemm_tflops)
        elif c.tensor_active is not None:
            s.tensor_achieved_frac = max(0.0, min(1.0, c.tensor_active))
            if pk.gemm_tflops:
                s.achieved_tflops = s.tensor_achieved_frac * pk.gemm_tflops
        # Energy decomposition: the idle floor is measured, so "power above
        # idle" is a measured quantity too — unlike any attempt to split power
        # between active and waiting time, which would be invented.
        if c.power_w is not None and pk.idle_w is not None:
            s.dynamic_w = max(0.0, c.power_w - pk.idle_w)
        if s.achieved_gbs is not None and c.power_w:
            s.gbs_per_total_w = s.achieved_gbs / c.power_w
            if s.dynamic_w:
                s.gbs_per_dynamic_w = s.achieved_gbs / s.dynamic_w
        # Roofline: achieved FLOP/s against achieved bytes/s, compared with the
        # card's own measured machine balance.
        if s.achieved_tflops is not None and s.achieved_gbs:
            s.intensity_flop_per_byte = (s.achieved_tflops * 1e12) / (
                s.achieved_gbs * 1e9
            )
        if pk.gemm_tflops and pk.membw_gbs:
            s.balance_flop_per_byte = (pk.gemm_tflops * 1e12) / (pk.membw_gbs * 1e9)
        if s.intensity_flop_per_byte is not None and s.balance_flop_per_byte:
            if s.active_share is not None and s.active_share < 0.02:
                s.bound_by = "idle"
            elif s.intensity_flop_per_byte < s.balance_flop_per_byte:
                s.bound_by = "memory"
            else:
                s.bound_by = "compute"
        elif s.active_share is not None and s.active_share < 0.02:
            s.bound_by = "idle"
        shares.append(s)

    # Group-relative shares. Sum over DISTINCT cards, then attribute; a
    # co-located rank pair shares one card's work, it does not double it.
    def _sum_distinct(attr: str) -> Optional[float]:
        seen, total, any_val = set(), 0.0, False
        for s in shares:
            if s.gpu_index in seen:
                continue
            seen.add(s.gpu_index)
            v = getattr(s, attr)
            if v is not None:
                total += v
                any_val = True
        return total if any_val else None

    total_gbs = _sum_distinct("achieved_gbs")
    total_tflops = _sum_distinct("achieved_tflops")
    for s in shares:
        if total_gbs and s.achieved_gbs is not None:
            s.byte_work_share = s.achieved_gbs / total_gbs
        if total_tflops and s.achieved_tflops is not None:
            s.flop_work_share = s.achieved_tflops / total_tflops

    if any(s.wait_share is not None for s in shares):
        caveats.append(CAVEAT_WAIT_COSTS_POWER)

    group_power = _sum_distinct("power_w")
    pacer, basis = pacing_rank(shares)
    return RankView(
        ranks=shares,
        caveats=caveats,
        group_power_w=group_power,
        group_power_approximate=True,
        pacer_rank=pacer,
        pacer_basis=basis,
    )


def pacing_rank(shares: Sequence[RankShare]) -> tuple:
    """Which rank sets the group's pace.

    Under lockstep the pacing rank is the one that waits LEAST: everyone else
    is waiting on it. Reported with its basis, and only when the spread is
    large enough to mean anything — a five-point difference in a coarse
    utilization reading is noise, and naming a pacer from it would be a guess
    dressed as a finding.
    """
    have = [s for s in shares if s.wait_share is not None]
    if len(have) < 2:
        return (None, None)
    have_sorted = sorted(have, key=lambda s: s.wait_share)
    best, second = have_sorted[0], have_sorted[1]
    spread = second.wait_share - best.wait_share
    if spread < 0.10:
        return (
            None,
            f"no clear pacer: wait shares within {spread * 100:.0f} points of "
            "each other, which is inside the noise of this activity source",
        )
    coarse = best.activity_source.startswith(("nvml-utilization", "nvidia-smi"))
    basis = (
        f"lowest wait share ({best.wait_share * 100:.0f}% vs next "
        f"{second.wait_share * 100:.0f}%) — the group waits on this rank"
    )
    if coarse:
        basis += "; from coarse utilization, so treat as a direction, not a measurement"
    return (best.rank if best.rank is not None else best.gpu_index, basis)
