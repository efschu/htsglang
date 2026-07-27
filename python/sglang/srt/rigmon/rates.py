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

**Milliseconds per round is the yardstick; tok/s is a companion.** The two
measure the same machine at different resolutions, and the gap is an order of
magnitude or more. Measured boot-to-boot on this rig: ms per verify round has
a 0.08-0.37 % noise floor at the device level and 0.98-1.46 % at the host
level, while raw tok/s sits at 2.7-4.2 %. Two reasons behind the numbers:

* tok/s tracks the output CONTENT (r = 0.90 on this project's own runs), so
  two arms that generated different text are not comparable at all, while a
  round time is content-blind as long as batch size and resident tokens match;
* under speculation ``tok/s = accept length / round time`` mixes two
  quantities. A round time without its accept length and verify count cannot
  be placed, which is why :class:`RoundTime` carries all three or none.

So tok/s may be shown next to a round time, never alone and never as the basis
for a decision. :class:`GroupThroughput` keeps that in its own field notes.

**Tokens are not attributable to a card.** Under tensor parallelism every rank
holds a shard of every layer and the group produces a token jointly; there is
no rank-local token count to report. Splitting group tok/s by shard size, by
FLOP share, or by anything else would produce a number that looks per-card and
means nothing. So this module publishes exactly one throughput figure —
:class:`GroupThroughput` — and it belongs to the group.

**Nor is a per-rank round time worth showing raw.** In lockstep the WALL time
of a round is near-identical on every rank by construction: the ranks enter
and leave the collective together. Three near-equal bars would hide the one
thing that differs. The informative decomposition is COMPUTE time against
COLLECTIVE WAIT per rank — the rank with the least wait is the one the others
are waiting for. :class:`RankShare` therefore carries
``compute_ms_per_round`` and ``collective_wait_ms_per_round`` as separate
fields, and leaves them ``None`` when the source cannot separate them rather
than filling both from one number.

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
    "RoundTime",
    "RankShare",
    "RankView",
    "PREFILL_CATEGORIES",
    "VERIFY_CATEGORIES",
    "DECODE_CATEGORIES",
    "DRAFT_CATEGORIES",
    "peaks_from_hw_profile",
    "idle_floor_from_power_profile",
    "group_throughput",
    "round_time",
    "phase_seconds",
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
CAVEAT_TOKS_IS_NOT_THE_YARDSTICK = Caveat(
    "toks_is_not_the_yardstick",
    "Throughput is shown for orientation, not for decisions. Its boot-to-boot "
    "noise floor is 2.7-4.2 % against 0.08-0.37 % for ms per verify round, it "
    "tracks the output content (r = 0.90), and under speculation it is accept "
    "length divided by round time — two quantities in one number. Compare "
    "round times.",
)
CAVEAT_NO_FORWARD_TIMER = Caveat(
    "no_forward_timer",
    "The engine exports no forward-time counter, so round times are "
    "unavailable — not zero. sglang:forward_execution_seconds_total is fed "
    "only by the CUDA-event device timer, which the scheduler installs under "
    "SGLANG_ENABLE_METRICS_DEVICE_TIMER=1. Set it (together with "
    "--enable-metrics-for-all-schedulers for the per-rank split) and the "
    "phase-labelled round times appear.",
    severity="warning",
)
CAVEAT_ROUNDS_DERIVED = Caveat(
    "rounds_derived",
    "The engine publishes no round counter, so the number of rounds in the "
    "window is derived as generated tokens divided by accept length. That "
    "holds while the accept length is stable across the window and drifts "
    "with it otherwise; the round time inherits that, which is why accept "
    "length is reported next to it rather than folded in.",
)
CAVEAT_WAIT_NOT_SEPARABLE = Caveat(
    "wait_not_separable",
    "Compute and collective wait could not be separated for at least one "
    "rank, so no wait figure is given for it. Under lockstep the WALL time "
    "per round is near-identical on every rank by construction; showing it "
    "per card would draw equal bars and hide the imbalance that matters.",
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


#: Forward-mode categories, grouped into the phases a reader thinks in. The
#: engine already labels its forward-time counter with these, so the phase
#: split costs a parser branch rather than a measurement path.
PREFILL_CATEGORIES = ("extend", "split_prefill")
VERIFY_CATEGORIES = ("target_verify",)
DECODE_CATEGORIES = ("decode",)
DRAFT_CATEGORIES = ("eagle_draft", "eagle_draft_extend")


def phase_seconds(
    per_rank_phase: Optional[Dict[int, Dict[str, float]]],
    prev_per_rank_phase: Optional[Dict[int, Dict[str, float]]],
) -> Dict[int, Dict[str, float]]:
    """Difference the phase-labelled forward-time counters over the window.

    Returns ``rank -> {category: seconds spent in that category}``. Categories
    are kept individually; grouping them into prefill/verify/decode is the
    caller's business, because a reader who wants the drafter's own cost
    separately must not have to unpick a pre-summed number.
    """
    if not per_rank_phase or not prev_per_rank_phase:
        return {}
    out: Dict[int, Dict[str, float]] = {}
    for rank, cur in per_rank_phase.items():
        prev = prev_per_rank_phase.get(rank)
        if not prev:
            continue
        row = {}
        for category, value in cur.items():
            before = prev.get(category)
            if before is not None and value >= before:
                row[category] = value - before
        if row:
            out[rank] = row
    return out


@dataclasses.dataclass
class RoundTime:
    """Milliseconds per forward round, per phase — the comparison yardstick.

    Every field that can be misread alone travels with what places it:
    ``accept_length`` and ``verify_ct`` sit next to the verify-round time
    because under speculation a round time without them says nothing about
    tokens, and ``rounds_source`` says whether the round count was counted or
    derived.
    """

    #: The speculative verify round: the unit the decode lever moves.
    ms_per_verify_round: Optional[float] = None
    #: The non-speculative decode step, for a server running without spec.
    ms_per_decode_round: Optional[float] = None
    #: Prefill time per 1000 prompt tokens. Deliberately not "per pass": the
    #: engine publishes no pass counter, and under chunked prefill a "pass" is
    #: a chunk whose size is a server setting. Per 1000 tokens is comparable
    #: across arms as long as the chunk size is held fixed.
    ms_per_1k_prefill_tokens: Optional[float] = None
    #: The drafter's own passes, separate from the verify they feed.
    ms_per_draft_pass: Optional[float] = None

    #: Accepted tokens per verify round. Without it a round time cannot be
    #: converted into tokens, and a comparison of round times across differing
    #: accept lengths is comparing different work.
    accept_length: Optional[float] = None
    #: Verify rounds counted in the window.
    verify_ct: Optional[float] = None
    #: Decode/verify rounds the window was divided by.
    rounds: Optional[float] = None
    #: "engine counter" | "derived from generated tokens / accept length" |
    #: "unavailable".
    rounds_source: str = "unavailable"
    #: Which phases the group actually spent time in during this window.
    phase_seconds: Dict[str, float] = dataclasses.field(default_factory=dict)

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


def round_time(
    metrics: Optional[Dict[str, float]],
    prev_metrics: Optional[Dict[str, float]],
    phase_delta: Optional[Dict[int, Dict[str, float]]],
    dt_s: Optional[float] = None,
) -> RoundTime:
    """Assemble the round-time yardstick from one collector window.

    The phase seconds are GPU-busy seconds per rank; a round is executed by
    every rank at once, so the group's time in a phase is the MAXIMUM over the
    ranks, not the sum — the group leaves the round when its slowest rank does.
    """
    rt = RoundTime()
    if not phase_delta:
        return rt

    group: Dict[str, float] = {}
    for row in phase_delta.values():
        for category, seconds in row.items():
            group[category] = max(group.get(category, 0.0), seconds)
    rt.phase_seconds = group
    if metrics:
        rt.accept_length = metrics.get("spec_accept_length")

    def _phase_total(categories) -> float:
        return sum(group.get(c, 0.0) for c in categories)

    # Round count. The engine publishes no counter for it, so it is derived
    # from generated tokens over accept length; without an accept length the
    # rounds stay unknown and every per-round figure stays None rather than
    # being silently divided by the token count.
    rounds = None
    if prev_metrics and metrics:
        gen = metrics.get("generation_tokens_total")
        gen0 = prev_metrics.get("generation_tokens_total")
        if gen is not None and gen0 is not None and gen >= gen0:
            produced = gen - gen0
            accept = rt.accept_length
            if accept and accept > 0:
                rounds = produced / accept
                rt.rounds_source = (
                    "derived from generated tokens / accept length"
                )
            elif produced > 0 and not _phase_total(VERIFY_CATEGORIES):
                # No speculation: one generated token per decode round.
                rounds = produced
                rt.rounds_source = "generated tokens (one per decode round)"
    rt.rounds = rounds

    if rounds and rounds > 0:
        verify_s = _phase_total(VERIFY_CATEGORIES)
        decode_s = _phase_total(DECODE_CATEGORIES)
        draft_s = _phase_total(DRAFT_CATEGORIES)
        if verify_s:
            rt.ms_per_verify_round = verify_s * 1000.0 / rounds
            rt.verify_ct = rounds
        if decode_s:
            rt.ms_per_decode_round = decode_s * 1000.0 / rounds
        if draft_s:
            rt.ms_per_draft_pass = draft_s * 1000.0 / rounds

    # Prefill is not divided by decode rounds: a prefill pass is a different
    # unit. A per-pass time would need a pass count the engine does not
    # publish, so what is reported is time per 1000 prompt tokens when the
    # prompt counter moved, and nothing otherwise.
    prefill_s = _phase_total(PREFILL_CATEGORIES)
    if prefill_s and prev_metrics and metrics:
        prompt = metrics.get("prompt_tokens_total")
        prompt0 = prev_metrics.get("prompt_tokens_total")
        if prompt is not None and prompt0 is not None and prompt > prompt0:
            rt.ms_per_1k_prefill_tokens = prefill_s * 1000.0 / ((prompt - prompt0) / 1000.0)
    return rt


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

    # -- round time, decomposed --------------------------------------------
    #: GPU-busy milliseconds this rank spent per round. The half of the round
    #: that differs between ranks.
    compute_ms_per_round: Optional[float] = None
    #: Milliseconds per round this rank spent NOT computing while the group's
    #: round was in flight, i.e. waiting on the collective. Together with
    #: ``compute_ms_per_round`` it sums to the group's round time, which is
    #: near-identical on every rank and therefore not reported per rank.
    collective_wait_ms_per_round: Optional[float] = None
    #: This rank's GPU seconds in the window, per forward category. Kept so a
    #: reader can ask "where did this rank's time go" without a second scrape.
    phase_seconds: Dict[str, float] = dataclasses.field(default_factory=dict)

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
    phase_delta: Optional[Dict[int, Dict[str, float]]] = None,
    round_times: Optional["RoundTime"] = None,
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

    ``phase_delta`` (from :func:`phase_seconds`) and ``round_times`` (from
    :func:`round_time`) add the compute-against-wait decomposition. Both are
    optional: without them the per-rank rows carry no round-time fields at all,
    which is the honest state — a raw per-rank round time would be the group's
    round time repeated, and that hides exactly what the decomposition shows.
    """
    peaks = peaks or {}
    engine_rates = engine_rates or {}
    phase_delta = phase_delta or {}
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

        # Round time, decomposed. Compute comes from this rank's own GPU-busy
        # seconds in the round-bearing phases; wait is what is left of the
        # group's round after that. Only the engine's forward-time counter can
        # carry this: a coarse utilization sample says how busy the CARD was,
        # which for co-located ranks is not the same question.
        rank_phase = phase_delta.get(rank) if rank is not None else None
        if rank_phase:
            s.phase_seconds = dict(rank_phase)
        if rank_phase and round_times and round_times.rounds:
            round_categories = VERIFY_CATEGORIES + DECODE_CATEGORIES + DRAFT_CATEGORIES
            busy_s = sum(rank_phase.get(cat, 0.0) for cat in round_categories)
            if busy_s > 0:
                s.compute_ms_per_round = busy_s * 1000.0 / round_times.rounds
                group_ms = round_times.ms_per_verify_round
                if group_ms is None:
                    group_ms = round_times.ms_per_decode_round
                if group_ms is not None:
                    if round_times.ms_per_draft_pass:
                        group_ms += round_times.ms_per_draft_pass
                    s.collective_wait_ms_per_round = max(
                        0.0, group_ms - s.compute_ms_per_round
                    )

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
    if shares and not any(s.compute_ms_per_round is not None for s in shares):
        caveats.append(CAVEAT_WAIT_NOT_SEPARABLE)
    elif any(
        s.compute_ms_per_round is not None and s.collective_wait_ms_per_round is None
        for s in shares
    ):
        caveats.append(CAVEAT_WAIT_NOT_SEPARABLE)

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

    The millisecond decomposition is preferred where it exists: a wait
    expressed in ms per round is directly readable as "this is what the group
    loses to imbalance each round", and it comes from the CUDA-event forward
    timer rather than from a sampled utilization figure.
    """
    ms = [s for s in shares if s.collective_wait_ms_per_round is not None]
    if len(ms) >= 2:
        ms_sorted = sorted(ms, key=lambda s: s.collective_wait_ms_per_round)
        best, second = ms_sorted[0], ms_sorted[1]
        spread = second.collective_wait_ms_per_round - best.collective_wait_ms_per_round
        round_ms = (best.compute_ms_per_round or 0.0) + best.collective_wait_ms_per_round
        # Below a twentieth of the round the difference is not worth naming a
        # pacer over; the ranks are effectively balanced.
        if round_ms > 0 and spread < 0.05 * round_ms:
            return (
                None,
                f"no clear pacer: collective wait within {spread:.2f} ms per "
                f"round of each other, under 5 % of the {round_ms:.2f} ms "
                "round",
            )
        return (
            best.rank if best.rank is not None else best.gpu_index,
            f"lowest collective wait ({best.collective_wait_ms_per_round:.2f} ms "
            f"per round vs next {second.collective_wait_ms_per_round:.2f} ms) — "
            f"the group waits on this rank, and the difference is what "
            f"rebalancing could recover",
        )

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
