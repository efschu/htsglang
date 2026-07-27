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
"""Benchmark scenarios: measure ONE limiting factor, not aggregate throughput.

A total-throughput number answers "is it faster" and nothing else. The
questions worth asking are about single limiting factors — what does the power
target do, what does host RAM clock do to the spill path, how many concurrent
prefills does a second rig buy, what does spilling cost in latency under
concurrency. Each of those needs its own parameter axis, its own measured
quantity and its own stopping rule.

So a scenario is **data**, not code:

    question  ->  parameter space  ->  measured quantity  ->  stopping rule

New questions are new entries (in the registry, or in a user JSON file loaded
at runtime), not new code paths. :func:`suggest` picks a starting scenario from
a free-text question, which is the one-click path; every field stays editable
afterwards, which is the dropdown path.

Two obligations are structural rather than optional:

**Window separation.** A measurement that averages across a state change
reports neither state. The spill path is the worked example: the window right
after a restore contains the speculative-resume backfill, so throughput there
is neither the spilled value nor the steady one. Every scenario therefore
declares its windows explicitly, and a result carries per-window figures — a
single mean over a transition is not accepted.

**Noise floor first.** Before any A-vs-B comparison the same configuration is
measured against itself. Without that floor there is no threshold below which a
difference must not be reported, and this project has repeatedly found
differences that were content variance rather than effect.

**Milliseconds per round is the primary metric; throughput is a companion.**
Measured boot-to-boot on this rig the noise floor of ms per verify round is
0.08-0.37 % at the device level and 0.98-1.46 % at the host level, against
2.7-4.2 % for raw tok/s — a resolution difference of 10-30x. Beyond the
numbers, tok/s follows the output CONTENT (r = 0.90), so two arms that
generated different text are not comparable at all, and under speculation
``tok/s = accept length / round time`` fuses two quantities into one. Every
scenario here therefore makes a round time its primary metric and carries
accept length and verify count alongside it; throughput may be reported next
to them, never as the comparison.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "Axis",
    "HarnessBinding",
    "build_harness_command",
    "Window",
    "Metric",
    "StopRule",
    "Scenario",
    "ScenarioPlan",
    "SCENARIOS",
    "suggest",
    "plan_scenario",
    "render_scenario_text",
    "load_scenarios",
]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Axis:
    """One parameter that is varied. ``values`` is the suggested sweep; the UI
    offers it as a dropdown and lets it be edited."""

    key: str
    label: str
    unit: str = ""
    values: List[Any] = dataclasses.field(default_factory=list)
    #: "control" (a host setting), "flag" (a server flag), "workload"
    kind: str = "flag"
    #: Facility keys from :mod:`sglang.srt.rigmon.facilities` this axis needs.
    requires: List[str] = dataclasses.field(default_factory=list)
    note: str = ""

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Window:
    """A measurement window. Results are reported PER window, never merged."""

    key: str
    label: str
    description: str
    #: What opens the window, in terms a harness can detect.
    starts_at: str = ""
    ends_at: str = ""
    #: Windows that must not be pooled into a headline figure.
    exclude_from_headline: bool = False

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class Metric:
    """The measured quantity. ``direction`` says which way is better, so a
    result table can be read without knowing the metric.

    Frozen because the shared instances below (the round-time yardstick, the
    accept length, the demoted throughput) are referenced by several scenarios
    at once; a mutable shared instance is the shape of bug this project has
    hit repeatedly.
    """

    key: str
    label: str
    unit: str
    direction: str = "higher_better"  # higher_better | lower_better | neutral
    source: str = ""
    primary: bool = False
    #: Reported for orientation but not used to decide. Set with a reason: a
    #: metric that is shown without being trusted has to say which it is, or a
    #: reader will compare it anyway.
    context_only: str = ""

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class StopRule:
    """When to stop sweeping. Without one, a sweep either runs forever or stops
    at an arbitrary point that later reads as the conclusion."""

    key: str
    label: str
    condition: str
    kind: str = "threshold"  # threshold | budget | saturation | failure

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class HarnessBinding:
    """How a scenario drives an EXISTING benchmark harness.

    The scenarios here describe measurements; they do not run them. sglang
    already ships the load generators (``sglang.benchmark.serving`` for the
    serving path, ``sglang.benchmark.offline_throughput`` and
    ``bench_one_batch`` for the offline ones) together with a dataset registry.
    Rebuilding any of that would mean maintaining a second load generator and
    a second dataset story for no gain, so a scenario binds to one instead:
    it says which harness, which harness flag each axis maps to, and which
    harness output field each metric reads from.
    """

    module: str
    provides: str = ""
    fixed_flags: List[str] = dataclasses.field(default_factory=list)
    #: scenario axis key -> harness flag
    axis_flags: Dict[str, str] = dataclasses.field(default_factory=dict)
    #: scenario metric key -> field in the harness result
    metric_fields: Dict[str, str] = dataclasses.field(default_factory=dict)
    #: Axes the harness cannot set, because they are host or server settings.
    external_axes: List[str] = dataclasses.field(default_factory=list)
    note: str = ""

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Scenario:
    key: str
    question: str
    #: What is expected and what would falsify it. A scenario without a
    #: falsifier tends to confirm whatever it was built to confirm.
    hypothesis: str
    falsifier: str
    axes: List[Axis]
    metrics: List[Metric]
    windows: List[Window] = dataclasses.field(default_factory=list)
    stop_rules: List[StopRule] = dataclasses.field(default_factory=list)
    #: Things that will corrupt the result if not held fixed.
    confounders: List[str] = dataclasses.field(default_factory=list)
    requires: List[str] = dataclasses.field(default_factory=list)
    est_runtime_min: Optional[int] = None
    #: Which existing harness runs this, and how the axes map onto it.
    harness: Optional[HarnessBinding] = None
    #: Free-text keywords used by :func:`suggest`.
    keywords: List[str] = dataclasses.field(default_factory=list)
    #: A-vs-A noise floor before any comparison. Off only with a reason.
    noise_floor_required: bool = True

    @property
    def primary_metric(self) -> Optional[Metric]:
        for m in self.metrics:
            if m.primary:
                return m
        return self.metrics[0] if self.metrics else None

    def to_json(self) -> dict:
        return {
            "key": self.key,
            "question": self.question,
            "hypothesis": self.hypothesis,
            "falsifier": self.falsifier,
            "axes": [a.to_json() for a in self.axes],
            "metrics": [m.to_json() for m in self.metrics],
            "windows": [w.to_json() for w in self.windows],
            "stop_rules": [s.to_json() for s in self.stop_rules],
            "harness": self.harness.to_json() if self.harness else None,
            "confounders": list(self.confounders),
            "requires": list(self.requires),
            "est_runtime_min": self.est_runtime_min,
            "keywords": list(self.keywords),
            "noise_floor_required": self.noise_floor_required,
        }

    @classmethod
    def from_json(cls, d: dict) -> Scenario:
        return cls(
            key=d["key"],
            question=d["question"],
            hypothesis=d.get("hypothesis", ""),
            falsifier=d.get("falsifier", ""),
            axes=[Axis(**a) for a in d.get("axes", [])],
            metrics=[Metric(**m) for m in d.get("metrics", [])],
            windows=[Window(**w) for w in d.get("windows", [])],
            stop_rules=[StopRule(**s) for s in d.get("stop_rules", [])],
            harness=(
                HarnessBinding(**d["harness"]) if d.get("harness") else None
            ),
            confounders=list(d.get("confounders", [])),
            requires=list(d.get("requires", [])),
            est_runtime_min=d.get("est_runtime_min"),
            keywords=list(d.get("keywords", [])),
            noise_floor_required=bool(d.get("noise_floor_required", True)),
        )


# ---------------------------------------------------------------------------
# Shared metric vocabulary
# ---------------------------------------------------------------------------

#: The comparison yardstick. Used wherever a scenario compares two arms of a
#: generating server; the noise floor quoted in the module docstring is the
#: reason it outranks throughput.
MS_PER_VERIFY_ROUND = Metric(
    "ms_per_verify_round",
    "time per verify round",
    "ms",
    "lower_better",
    source="collector: forward_execution_seconds_total{category=target_verify} "
    "over the round count (needs SGLANG_ENABLE_METRICS_DEVICE_TIMER=1)",
    primary=True,
)
MS_PER_DECODE_ROUND = Metric(
    "ms_per_decode_round",
    "time per decode step",
    "ms",
    "lower_better",
    source="collector: forward_execution_seconds_total{category=decode}; the "
    "yardstick on a server running without speculation",
)
MS_PER_1K_PREFILL = Metric(
    "ms_per_1k_prefill_tokens",
    "prefill time per 1000 prompt tokens",
    "ms",
    "lower_better",
    source="collector: forward_execution_seconds_total{category=extend|"
    "split_prefill} over prompt tokens; comparable only at a fixed chunk size",
)
#: Never omitted next to a round time: under speculation a round time without
#: its accept length describes a different amount of work in each arm.
ACCEPT_LENGTH = Metric(
    "accept_length",
    "accepted tokens per verify round",
    "tokens",
    "higher_better",
    source="engine: sglang:spec_accept_length, or meta_info.spec_accept_length "
    "per request",
)
VERIFY_CT = Metric(
    "verify_ct",
    "verify rounds in the window",
    "rounds",
    "neutral",
    source="engine: meta_info.spec_verify_ct, or generated tokens / accept "
    "length when only the exposition is available",
)
#: Throughput, demoted on purpose.
TOK_S_CONTEXT = Metric(
    "tok_s",
    "group throughput",
    "tok/s",
    source="engine counter delta",
    context_only="2.7-4.2 % boot-to-boot noise against 0.08-0.37 % for the "
    "round time, follows the output content (r = 0.90), and under speculation "
    "fuses accept length with round time. Shown for orientation; compare the "
    "round times.",
)


# ---------------------------------------------------------------------------
# Shared window vocabulary
# ---------------------------------------------------------------------------

#: The three windows any spill/offload measurement must keep apart. The middle
#: one exists because a restore is followed by a speculative-resume backfill:
#: the throughput observed there belongs to the catch-up, not to the restored
#: steady state, and pooling it produces a figure that describes neither.
SPILL_WINDOWS = [
    Window(
        "pre_spill",
        "before the spill",
        "Baseline with the session resident. The reference the other windows "
        "are read against.",
        starts_at="request accepted, KV resident",
        ends_at="spill decision taken",
    ),
    Window(
        "during_spill",
        "while spilled",
        "The victim session's KV lives in host memory. This is the cost the "
        "feature actually imposes while it is engaged.",
        starts_at="spill complete",
        ends_at="restore begins",
    ),
    Window(
        "restore_transient",
        "restore transient (includes backfill)",
        "From the restore until steady state. Contains the page-in AND the "
        "speculative-resume backfill, so throughput here is a catch-up rate, "
        "not a serving rate. Reported separately and never pooled into a "
        "headline number.",
        starts_at="restore begins",
        ends_at="backfill drained and step time stable over N consecutive steps",
        exclude_from_headline=True,
    ),
    Window(
        "steady_after",
        "steady state after restore",
        "The figure that answers 'what does it cost once it has recovered'. "
        "Comparable with pre_spill; the transient is not.",
        starts_at="step time stable",
        ends_at="end of run",
    ),
]


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

SCENARIOS: Dict[str, Scenario] = {}


def _register(s: Scenario) -> Scenario:
    SCENARIOS[s.key] = s
    return s


_register(
    Scenario(
        key="noise_floor",
        question="How large is a difference that means nothing?",
        hypothesis=(
            "Repeating the identical configuration yields a spread; anything "
            "below it is not reportable."
        ),
        falsifier=(
            "If the A-vs-A spread is as large as the effect under study, the "
            "effect is not measurable on this rig and must not be reported."
        ),
        axes=[
            Axis(
                "repeat",
                "identical repeats",
                "runs",
                [5],
                kind="workload",
                note="Same config, same prompts, interleaved with the comparison "
                "arm rather than run as a block, so drift cannot masquerade as "
                "an effect.",
            )
        ],
        metrics=[
            MS_PER_VERIFY_ROUND,
            ACCEPT_LENGTH,
            VERIFY_CT,
            MS_PER_1K_PREFILL,
            Metric("ttft_p50", "TTFT median", "ms", "lower_better"),
            TOK_S_CONTEXT,
        ],
        stop_rules=[
            StopRule(
                "spread_known",
                "spread established",
                "Stop once the relative standard deviation over the repeats is "
                "stable; report it as the detection threshold for every "
                "comparison that follows. Establish it separately for the "
                "round time and for throughput — they differ by 10-30x, and a "
                "threshold taken from the coarser one would discard real "
                "effects.",
                kind="saturation",
            )
        ],
        confounders=[
            "Output CONTENT drives throughput more than most settings do; fix "
            "the prompt set and validate the outputs.",
            "Clocks unpinned: a drifting P-state inflates the floor.",
        ],
        harness=HarnessBinding(
            module="sglang.benchmark.serving",
            provides="load generation, dataset selection, TTFT/ITL/e2e percentiles",
            fixed_flags=[
                "--backend sglang",
                "--dataset-name random",
                "--warmup-requests 8",
                "--seed 1",
                "--flush-cache",
                "--output-details",
            ],
            axis_flags={'repeat': '(repeat the whole invocation)'},
            metric_fields={
                'ms_per_verify_round': '(collector: round_time over the same wall clock)',
                'accept_length': '(engine: sglang:spec_accept_length; meta_info.spec_accept_length per request)',
                'verify_ct': '(engine: meta_info.spec_verify_ct, summed over the window)',
                'ms_per_1k_prefill_tokens': '(collector: round_time.ms_per_1k_prefill_tokens)',
                'ttft_p50': 'median_ttft_ms',
                'tok_s': 'output_throughput',
            },
            external_axes=[],
            note=(
                "Percentiles and throughput come from the harness; card state "
                "and per-rank work come from the collector over the same wall "
                "clock, joined by timestamp."
            ),
        ),
        est_runtime_min=10,
        keywords=["noise", "floor", "rauschen", "a-vs-a", "threshold", "spread"],
        noise_floor_required=False,
    )
)

_register(
    Scenario(
        key="power_target_sweep",
        question="What does the power target do to throughput and to energy per token?",
        hypothesis=(
            "Throughput falls sub-linearly with the power cap while energy per "
            "token improves, so there is a knee well below the stock limit."
        ),
        falsifier=(
            "If tok/s tracks the cap linearly there is no efficiency knee and "
            "the stock limit is already the right setting."
        ),
        axes=[
            Axis(
                "power_limit_w",
                "power limit per card",
                "W",
                ["60%", "70%", "80%", "90%", "100%"],
                kind="control",
                requires=["power_target"],
                note="Percentages of each card's enforced limit, so a "
                "heterogeneous rig sweeps comparably instead of pinning a "
                "3080 and a 5090 to the same absolute watts.",
            )
        ],
        metrics=[
            MS_PER_VERIFY_ROUND,
            ACCEPT_LENGTH,
            Metric("j_per_token", "energy per token", "J/tok", "lower_better",
                   source="NVML total-energy counter difference, not integrated power"),
            TOK_S_CONTEXT,
            Metric("clock_ratio", "achieved clock / max clock", "fraction"),
            Metric("throttle_share", "share of samples throttled", "fraction",
                   "lower_better"),
        ],
        stop_rules=[
            StopRule(
                "throughput_cliff",
                "throughput cliff",
                "Stop lowering the cap once tok/s drops more than 25 % below "
                "the stock-limit arm: past the cliff the sweep only measures "
                "how badly the card is starved.",
            ),
            StopRule(
                "thermal_invalid",
                "thermal contamination",
                "Abort a point whose samples show sw_thermal_slowdown: a "
                "thermally limited card is not measuring the power cap.",
                kind="failure",
            ),
        ],
        confounders=[
            "Temperature: sweep from low cap upward, or let the card cool "
            "between points, so the high-cap arms are not measured hot.",
            "Unpinned clocks — pin them if the host allows it.",
            "The slowest rank paces the group; a cap applied to the fast card "
            "may not move group throughput at all.",
        ],
        harness=HarnessBinding(
            module="sglang.benchmark.serving",
            provides="load generation, dataset selection, TTFT/ITL/e2e percentiles",
            fixed_flags=[
                "--backend sglang",
                "--dataset-name random",
                "--warmup-requests 8",
                "--seed 1",
                "--flush-cache",
                "--output-details",
            ],
            axis_flags={},
            metric_fields={
                'ms_per_verify_round': '(collector: round_time over the same wall clock)',
                'accept_length': '(engine: sglang:spec_accept_length; meta_info.spec_accept_length per request)',
                'j_per_token': '(collector: energy counter delta / generated tokens)',
                'tok_s': 'output_throughput',
            },
            external_axes=['power_limit_w'],
            note=(
                "Percentiles and throughput come from the harness; card state "
                "and per-rank work come from the collector over the same wall "
                "clock, joined by timestamp."
            ),
        ),
        requires=["power_target"],
        est_runtime_min=30,
        keywords=["power", "target", "watt", "energy", "efficiency", "cap",
                  "leistung", "energie", "strom", "tdp"],
    )
)

_register(
    Scenario(
        key="ram_clock_spill",
        question="What does a higher host RAM clock do to the spill/offload path?",
        hypothesis=(
            "The spill path is host-memory bound, so restore bandwidth scales "
            "with RAM clock while the resident path is unaffected."
        ),
        falsifier=(
            "If restore bandwidth is flat across RAM clocks the bottleneck is "
            "PCIe or the copy engine, and RAM tuning buys nothing."
        ),
        axes=[
            Axis(
                "ram_clock",
                "host RAM clock",
                "MT/s",
                ["stock (JEDEC)", "XMP/EXPO profile", "manual overclock"],
                kind="control",
                requires=["ram_timings"],
                note="Firmware setting: it cannot be changed from software, so "
                "each point is a separate boot. Record the value read back, not "
                "the value requested.",
            ),
            Axis(
                "spill_state",
                "spill engaged",
                "",
                [False, True],
                kind="flag",
            ),
        ],
        metrics=[
            Metric("restore_gbs", "restore bandwidth", "GB/s", primary=True,
                   source="bytes paged in / restore window duration"),
            Metric("restore_ms", "restore duration", "ms", "lower_better"),
            Metric(
                "steady_ms_per_verify_round",
                "time per verify round once restored",
                "ms",
                "lower_better",
                source="collector, per window; comparable with the pre_spill "
                "window, which the restore transient is not",
            ),
            ACCEPT_LENGTH,
            dataclasses.replace(
                TOK_S_CONTEXT,
                key="steady_tok_s",
                label="steady-state throughput after restore",
            ),
        ],
        windows=SPILL_WINDOWS,
        stop_rules=[
            StopRule(
                "bandwidth_saturated",
                "bandwidth saturated",
                "Stop once restore bandwidth stops rising with RAM clock: the "
                "path has moved to another bottleneck and further tuning is "
                "measuring nothing.",
                kind="saturation",
            )
        ],
        confounders=[
            "A restore is followed by a resume backfill; measure the restore "
            "transient and the steady state separately (see windows).",
            "GPU0 sits on a x4 link, so the host-to-device stage may cap the "
            "measurement before RAM does. Report the stages separately.",
            "Fixed cost per handover and cost per token are different terms; "
            "vary the window size to separate them.",
        ],
        harness=HarnessBinding(
            module="sglang.benchmark.serving",
            provides="load generation, dataset selection, TTFT/ITL/e2e percentiles",
            fixed_flags=[
                "--backend sglang",
                "--dataset-name random",
                "--warmup-requests 8",
                "--seed 1",
                "--flush-cache",
                "--output-details",
            ],
            axis_flags={},
            metric_fields={
                'steady_ms_per_verify_round': '(collector: round_time over the same wall clock)',
                'restore_gbs': '(collector: spill-path counters, per window)',
                'restore_ms': '(collector: restore window duration)',
                'steady_tok_s': 'output_throughput',
            },
            external_axes=['ram_clock', 'spill_state'],
            note=(
                "Percentiles and throughput come from the harness; card state "
                "and per-rank work come from the collector over the same wall "
                "clock, joined by timestamp."
            ),
        ),
        requires=["ram_timings"],
        est_runtime_min=45,
        keywords=["ram", "memory", "clock", "spill", "offload", "takt",
                  "arbeitsspeicher", "restore", "overclock"],
    )
)

_register(
    Scenario(
        key="concurrent_prefill_capacity",
        question="How many concurrent request prefills does a second rig make possible?",
        hypothesis=(
            "A satellite absorbs prefill peaks, so TTFT under load stays flat "
            "up to a higher concurrency than the hub alone sustains."
        ),
        falsifier=(
            "If hub queueing time is small under realistic load, the offload "
            "never wins regardless of transport cost — measure the queue first."
        ),
        axes=[
            Axis(
                "concurrency",
                "concurrent requests",
                "requests",
                [1, 2, 4, 8, 16],
                kind="workload",
            ),
            Axis(
                "topology",
                "prefill placement",
                "",
                ["hub only", "hub + satellite"],
                kind="flag",
            ),
            Axis(
                "prompt_len",
                "prompt length",
                "tokens",
                [2048, 8192, 32768],
                kind="workload",
                note="The handover has a length-independent fixed cost, so the "
                "short-prompt point is where the offload should LOSE. Include "
                "it deliberately.",
            ),
        ],
        metrics=[
            Metric("ttft_p95", "TTFT 95th percentile", "ms", "lower_better",
                   primary=True),
            Metric("ttft_p50", "TTFT median", "ms", "lower_better"),
            Metric("queue_wait_ms", "queueing time before prefill", "ms",
                   "lower_better",
                   source="the term the offload actually attacks"),
            MS_PER_1K_PREFILL,
            dataclasses.replace(
                TOK_S_CONTEXT, key="prefill_tok_s", label="prompt throughput"
            ),
        ],
        windows=[
            Window(
                "warmup",
                "warm-up",
                "Graph capture and cache warming. Excluded from the result.",
                exclude_from_headline=True,
            ),
            Window(
                "loaded",
                "under sustained load",
                "Steady arrival rate at the target concurrency. The window the "
                "TTFT figures come from.",
            ),
        ],
        stop_rules=[
            StopRule(
                "ttft_budget",
                "TTFT budget exceeded",
                "Stop raising concurrency once p95 TTFT exceeds the budget; the "
                "largest concurrency still inside it IS the answer.",
                kind="budget",
            ),
            StopRule(
                "queue_negligible",
                "queue too small to matter",
                "If hub queueing time stays far below prefill duration, stop: "
                "the premise of the offload does not hold on this workload.",
                kind="failure",
            ),
        ],
        confounders=[
            "Both sides must run the same weights and quantisation, otherwise "
            "the handover changes output quality and the comparison is void.",
            "Chunked prefill mixes phases in one batch; hold the chunk size "
            "fixed across arms.",
        ],
        harness=HarnessBinding(
            module="sglang.benchmark.serving",
            provides="load generation, dataset selection, TTFT/ITL/e2e percentiles",
            fixed_flags=[
                "--backend sglang",
                "--dataset-name random",
                "--warmup-requests 8",
                "--seed 1",
                "--flush-cache",
                "--output-details",
            ],
            axis_flags={'concurrency': '--max-concurrency', 'prompt_len': '--random-input-len'},
            metric_fields={
                'ttft_p95': 'p95_ttft_ms',
                'ttft_p50': 'median_ttft_ms',
                'ms_per_1k_prefill_tokens': '(collector: round_time.ms_per_1k_prefill_tokens)',
                'queue_wait_ms': '(engine: sglang:num_queue_reqs and the scheduler wait histogram)',
                'prefill_tok_s': 'input_throughput',
            },
            external_axes=['topology'],
            note=(
                "Percentiles and throughput come from the harness; card state "
                "and per-rank work come from the collector over the same wall "
                "clock, joined by timestamp."
            ),
        ),
        est_runtime_min=40,
        keywords=["prefill", "concurrent", "ttft", "queue", "satellite",
                  "zweitrig", "gleichzeitig", "latenz", "wartezeit"],
    )
)

_register(
    Scenario(
        key="spill_latency_under_concurrency",
        question=(
            "What does the spill offload do to end-to-end latency with several "
            "concurrent requests?"
        ),
        hypothesis=(
            "Spilling a victim session buys capacity at a bounded latency cost "
            "for that session and leaves the others unaffected."
        ),
        falsifier=(
            "If the non-victim sessions slow down too, the cost is not confined "
            "to the victim and the trade is worse than it appears."
        ),
        axes=[
            Axis("sessions", "concurrent sessions", "", [2, 3, 4], kind="workload"),
            Axis("spill_enabled", "spill offload", "", [False, True], kind="flag"),
            Axis(
                "context_len",
                "context length per session",
                "tokens",
                [8192, 32768],
                kind="workload",
            ),
        ],
        metrics=[
            Metric(
                "victim_ms_per_verify_round",
                "victim session: time per verify round",
                "ms",
                "lower_better",
                primary=True,
                source="collector, per window",
            ),
            Metric(
                "bystander_ms_per_verify_round",
                "non-victim sessions: time per verify round",
                "ms",
                "lower_better",
                source="the term that decides whether the cost is confined to "
                "the victim",
            ),
            ACCEPT_LENGTH,
            Metric("e2e_p95", "end-to-end latency p95", "ms", "lower_better"),
            Metric("restore_ms", "restore duration", "ms", "lower_better"),
        ],
        windows=SPILL_WINDOWS,
        stop_rules=[
            StopRule(
                "capacity_reached",
                "capacity reached",
                "Stop adding sessions once the configuration no longer fits "
                "without spilling more than one victim: beyond that the "
                "measurement is about a different mechanism.",
                kind="saturation",
            )
        ],
        confounders=[
            "The window right after a restore contains the speculative-resume "
            "backfill. A figure that pools it with the steady state describes "
            "neither — the windows in this scenario exist for that reason.",
            "Which session is chosen as victim changes the result; record it.",
            "Output content variance exceeds many real effects; fix the prompts "
            "and validate the outputs.",
        ],
        harness=HarnessBinding(
            module="sglang.benchmark.serving",
            provides="load generation, dataset selection, TTFT/ITL/e2e percentiles",
            fixed_flags=[
                "--backend sglang",
                "--dataset-name random",
                "--warmup-requests 8",
                "--seed 1",
                "--flush-cache",
                "--output-details",
            ],
            axis_flags={'sessions': '--max-concurrency', 'context_len': '--random-input-len'},
            metric_fields={
                'victim_ms_per_verify_round': '(collector: round_time of the spilled session, per window)',
                'bystander_ms_per_verify_round': '(collector: round_time of the others, per window)',
                'accept_length': '(engine: sglang:spec_accept_length; meta_info.spec_accept_length per request)',
                'e2e_p95': 'p95_e2e_latency_ms',
                'restore_ms': '(collector: restore window duration)',
            },
            external_axes=['spill_enabled'],
            note=(
                "Percentiles and throughput come from the harness; card state "
                "and per-rank work come from the collector over the same wall "
                "clock, joined by timestamp."
            ),
        ),
        est_runtime_min=50,
        keywords=["spill", "latency", "concurrent", "session", "offload",
                  "latenz", "gleichzeitig", "victim", "restore"],
    )
)

_register(
    Scenario(
        key="mlp_split_crossover",
        question=(
            "At what mix of prompt and output tokens does concentrating MLP "
            "weight on the compute-strong rank start to pay on THIS rig?"
        ),
        hypothesis=(
            "Concentration buys prefill per prompt token and costs decode per "
            "output token, so there is a prompt:output ratio where the two "
            "cancel. Below it the base split is faster; above it the "
            "concentration is."
        ),
        falsifier=(
            "If the prefill saving stays inside the prefill noise floor, or if "
            "the decode cost does not grow with concentration, there is no "
            "crossover to find on this rig and no vector should be proposed. "
            "The ratio is a property of this card mix, this model and this "
            "quantisation path; a value carried over from another rig "
            "falsifies nothing."
        ),
        axes=[
            Axis(
                "mlp_vector",
                "MLP weight split",
                "",
                ["base", "3,1,1", "6,1,1"],
                kind="flag",
                note="--rank-mlp-ratio, set at boot. 'base' is the plain "
                "VRAM-proportional split and is the reference every other "
                "point is read against.",
            ),
            Axis(
                "phase",
                "which half is being measured",
                "",
                ["random-ids", "random"],
                kind="workload",
                note="The harness dataset, and the two halves cannot share "
                "one. Prefill needs unique random input ids so the radix "
                "cache cannot serve a repeat and the prompt length is exact; "
                "decode needs natural text, because on random ids the model "
                "falls into a degenerate output mode whose speculative "
                "acceptance drives the per-token time more than the weight "
                "split does.",
            ),
            Axis(
                "prompt_tokens",
                "prompt length",
                "tokens",
                [500, 1000, 2000, 4000, 8000, 11000],
                kind="workload",
                note="The prefill gain is a SLOPE difference and saturates as "
                "a percentage past roughly 2000 tokens, so the short points "
                "are what separate the fixed cost from the slope.",
            ),
            Axis(
                "output_tokens",
                "generated tokens",
                "tokens",
                [1, 256],
                kind="workload",
                note="1 for the prefill points, so the measurement is prefill "
                "and nothing else; a decode-length run for the other half.",
            ),
        ],
        metrics=[
            Metric(
                "break_even_prompt_to_output",
                "break-even prompt:output ratio",
                "prompt:output",
                "lower_better",
                source="derived from this scenario's own arms: decode ms per "
                "output token divided by prefill ms saved per prompt token. "
                "Not read from the harness — it is the quantity the two "
                "measured slopes produce.",
                primary=True,
            ),
            MS_PER_1K_PREFILL,
            MS_PER_VERIFY_ROUND,
            ACCEPT_LENGTH,
            VERIFY_CT,
            Metric(
                "max_total_num_tokens",
                "KV pool size",
                "tokens",
                "higher_better",
                source="engine: /get_server_info, read once per boot. Falls "
                "out of every arm at no extra cost, and concentration spends "
                "it, so it belongs in the same table as the two time terms.",
            ),
            Metric(
                "cached_token_fraction",
                "prompt tokens served from the radix cache",
                "fraction",
                "lower_better",
                source="server log: #cached-token summed over the prefill "
                "chunks, over prompt tokens. Must be 0, and the proof travels "
                "with the result — a prefill figure taken over a warm cache "
                "measures the cache.",
            ),
            TOK_S_CONTEXT,
        ],
        stop_rules=[
            StopRule(
                "prefill_slope_saturated",
                "prefill slope established",
                "Stop extending the prompt length once the per-prompt-token "
                "slope between two consecutive lengths moves by less than the "
                "prefill noise floor. The gain is a slope difference; past "
                "saturation a longer prompt adds cost, not information.",
                kind="saturation",
            ),
            StopRule(
                "cache_bypass_broken",
                "cache bypass broken",
                "Abort a prefill point whose chunks report a non-zero "
                "#cached-token. The positive control is the same prompt sent "
                "twice in one boot; where it has been run it went 5505 -> "
                "1319 ms, which is the size of the artefact being avoided.",
                kind="failure",
            ),
            StopRule(
                "effect_below_floor",
                "no crossover on this rig",
                "If the prefill difference against the base arm stays inside "
                "the prefill noise floor at every length, report that there is "
                "no crossover here and propose nothing. A ratio computed from "
                "a difference below the floor is a number without a "
                "measurement behind it.",
                kind="failure",
            ),
        ],
        confounders=[
            "The radix cache. Repeating a prompt turns a prefill measurement "
            "into a cache measurement; unique random input ids per request are "
            "the bypass, and the #cached-token proof is part of the result.",
            "Raw ms per output token is acceptance-driven, not weight-driven. "
            "On random prompts it produced a 4.7x phantom depth term and one "
            "negative step time; report ms per speculative step on natural "
            "text and carry the accept length with it.",
            "The KV ownership vector must be pinned across arms "
            "(SGLANG_UNEVEN_TOKEN_VECTOR), or the weight split and the token "
            "split move together and neither is attributable.",
            "~/.cache/sglang/kv_budget-<confighash>.json carries a KV budget "
            "from one boot into the next; the runner neutralises it, and a "
            "manual run must too.",
            "Clocks and temperature. The decode half is the one a clock drop "
            "distorts most, so a throttled point is marked rather than "
            "silently averaged in.",
        ],
        harness=HarnessBinding(
            module="sglang.benchmark.serving",
            provides="load generation, dataset selection, TTFT/ITL/e2e percentiles",
            fixed_flags=[
                "--backend sglang",
                "--warmup-requests 8",
                "--seed 1",
                "--flush-cache",
                "--output-details",
            ],
            axis_flags={
                "phase": "--dataset-name",
                "prompt_tokens": "--random-input-len",
                "output_tokens": "--random-output-len",
            },
            metric_fields={
                "ms_per_1k_prefill_tokens": "(collector: round_time.ms_per_1k_prefill_tokens)",
                "ms_per_verify_round": "(collector: round_time over the same wall clock)",
                "accept_length": "(engine: sglang:spec_accept_length; meta_info.spec_accept_length per request)",
                "verify_ct": "(engine: meta_info.spec_verify_ct, summed over the window)",
                "max_total_num_tokens": "(engine: /get_server_info, once per boot)",
                "cached_token_fraction": "(server log: #cached-token per prefill chunk)",
                "tok_s": "output_throughput",
            },
            external_axes=["mlp_vector"],
            note=(
                "The MLP split is a boot flag, so each vector is an arm rather "
                "than a point. The phase is a point: one invocation measures "
                "prefill or decode, never both, because the two need different "
                "prompt material."
            ),
        ),
        est_runtime_min=190,
        keywords=["crossover", "kipppunkt", "break-even", "mlp", "split",
                  "concentration", "konzentration", "prefill", "decode",
                  "ratio", "verhaeltnis", "gewichtsverteilung"],
    )
)


# ---------------------------------------------------------------------------
# Suggestion and planning
# ---------------------------------------------------------------------------


def suggest(question: str, registry: Optional[Dict[str, Scenario]] = None) -> Optional[Scenario]:
    """Pick a starting scenario from a free-text question.

    Deliberately a ranking, not a fixed menu: the suggestion follows the
    question. A question that matches nothing returns None rather than the
    first entry, because a confidently wrong scenario costs more than none.
    """
    registry = registry if registry is not None else SCENARIOS
    words = {w.strip(".,;:!?()").lower() for w in str(question).split()}
    if not words:
        return None
    best, best_score = None, 0
    for s in registry.values():
        score = sum(1 for k in s.keywords if k in words)
        score += sum(
            1 for k in s.keywords if any(k in w for w in words if len(w) > 3)
        )
        if score > best_score:
            best, best_score = s, score
    return best


@dataclasses.dataclass
class ScenarioPlan:
    """A scenario resolved against this host's facilities."""

    scenario: Scenario
    runnable: bool
    blocked_axes: List[Dict[str, Any]]
    missing_facilities: List[Dict[str, Any]]
    preflight: List[str]

    def to_json(self) -> dict:
        return {
            "scenario": self.scenario.to_json(),
            "runnable": self.runnable,
            "blocked_axes": self.blocked_axes,
            "missing_facilities": self.missing_facilities,
            "preflight": self.preflight,
        }


def plan_scenario(
    scenario: Scenario, facility_list: Optional[Sequence] = None
) -> ScenarioPlan:
    """Resolve a scenario against the host.

    An axis whose control is unreachable is reported as blocked WITH the
    remedy, and the axis stays in the plan rather than being dropped: a sweep
    silently reduced to one point looks like a completed measurement.
    """
    by_key = {f.key: f for f in (facility_list or [])}
    blocked, missing = [], []
    needed = set(scenario.requires)
    for a in scenario.axes:
        needed.update(a.requires)
    for key in sorted(needed):
        f = by_key.get(key)
        if f is None:
            missing.append(
                {
                    "key": key,
                    "available": None,
                    "reason": "facility not probed on this host",
                    "remedy": [],
                }
            )
        elif not f.available:
            missing.append(
                {
                    "key": key,
                    "available": False,
                    "reason": f.reason,
                    "remedy": f.remedy,
                    "impossible_in_container": f.impossible_in_container,
                }
            )
    missing_keys = {m["key"] for m in missing}
    for a in scenario.axes:
        if set(a.requires) & missing_keys:
            blocked.append(
                {
                    "axis": a.key,
                    "label": a.label,
                    "requires": a.requires,
                    "effect": (
                        "this axis cannot be swept here; the scenario would "
                        "collapse to a single point, which is not the "
                        "measurement it claims to be"
                    ),
                }
            )
    preflight = []
    if scenario.noise_floor_required:
        preflight.append(
            "Run the noise-floor scenario first (A vs A, interleaved) and take "
            "its spread as the threshold below which no difference is reported."
        )
    if any(w.exclude_from_headline for w in scenario.windows):
        names = ", ".join(w.key for w in scenario.windows if w.exclude_from_headline)
        preflight.append(
            f"Report windows separately; {names} must not enter a headline "
            "figure."
        )
    for c in scenario.confounders:
        preflight.append(f"Hold fixed / record: {c}")
    return ScenarioPlan(
        scenario=scenario,
        runnable=not blocked,
        blocked_axes=blocked,
        missing_facilities=missing,
        preflight=preflight,
    )


def render_scenario_text(plan: ScenarioPlan) -> str:
    s = plan.scenario
    lines = [
        f"Scenario: {s.key}",
        f"Question: {s.question}",
        f"Hypothesis: {s.hypothesis}",
        f"Falsified by: {s.falsifier}",
        "",
        "Parameter space:",
    ]
    for a in s.axes:
        vals = ", ".join(str(v) for v in a.values)
        lines.append(f"  - {a.label} [{a.kind}] {a.unit}: {vals}")
        if a.note:
            lines.append(f"      note: {a.note}")
    lines.append("")
    lines.append("Measured:")
    for m in s.metrics:
        star = " (primary)" if m.primary else ""
        lines.append(f"  - {m.label} [{m.unit}, {m.direction}]{star}")
        if m.context_only:
            # A metric shown without being trusted has to say which it is,
            # here as well as in the JSON, or it gets compared anyway.
            lines.append(f"      context only: {m.context_only}")
    if s.windows:
        lines.append("")
        lines.append("Windows (reported separately):")
        for w in s.windows:
            flag = "  [excluded from headline]" if w.exclude_from_headline else ""
            lines.append(f"  - {w.key}: {w.description}{flag}")
    lines.append("")
    lines.append("Stop when:")
    for r in s.stop_rules:
        lines.append(f"  - {r.label}: {r.condition}")
    if plan.preflight:
        lines.append("")
        lines.append("Before running:")
        for p in plan.preflight:
            lines.append(f"  - {p}")
    if not plan.runnable:
        lines.append("")
        lines.append("NOT RUNNABLE HERE:")
        for b in plan.blocked_axes:
            lines.append(f"  - axis {b['axis']}: needs {', '.join(b['requires'])}")
        for m in plan.missing_facilities:
            lines.append(f"  - {m['key']}: {m.get('reason')}")
            for r in m.get("remedy") or []:
                lines.append(f"      remedy: {r}")
    if s.est_runtime_min:
        lines.append("")
        lines.append(f"Estimated runtime: ~{s.est_runtime_min} min")
    return "\n".join(lines)


def build_harness_command(
    scenario: Scenario,
    point: Dict[str, Any],
    base_url: str = "http://127.0.0.1:30000",
    num_prompts: int = 64,
) -> Dict[str, Any]:
    """Turn one point of the parameter space into a harness invocation.

    Axes the harness cannot set (a host control, a server flag that needs a
    restart) come back under ``external`` as steps to take BEFORE the command,
    rather than being dropped. A sweep whose control axis silently vanished
    would run the same point repeatedly and look like a completed measurement.
    """
    h = scenario.harness
    if h is None:
        return {
            "runnable": False,
            "reason": f"scenario {scenario.key} declares no harness binding",
        }
    flags = list(h.fixed_flags) + [f"--base-url {base_url}", f"--num-prompts {num_prompts}"]
    external: List[Dict[str, Any]] = []
    unmapped: List[str] = []
    for axis in scenario.axes:
        if axis.key not in point:
            continue
        value = point[axis.key]
        flag = h.axis_flags.get(axis.key)
        if flag and flag.startswith("--"):
            flags.append(f"{flag} {value}")
        elif axis.key in h.external_axes:
            external.append(
                {
                    "axis": axis.key,
                    "label": axis.label,
                    "value": value,
                    "kind": axis.kind,
                    "apply": (
                        "set on the host before the run"
                        if axis.kind == "control"
                        else "restart the server with this setting"
                    ),
                }
            )
        else:
            unmapped.append(axis.key)
    return {
        "runnable": not unmapped,
        "module": h.module,
        "command": f"python -m {h.module} " + " ".join(flags),
        "external": external,
        "unmapped_axes": unmapped,
        "metric_fields": dict(h.metric_fields),
        "windows": [w.key for w in scenario.windows],
    }


def load_scenarios(path: str, into: Optional[Dict[str, Scenario]] = None) -> Dict[str, Scenario]:
    """Load extra scenarios from JSON, so a new question needs no new code."""
    target = SCENARIOS if into is None else into
    with open(path) as f:
        data = json.load(f)
    for d in data.get("scenarios", []):
        s = Scenario.from_json(d)
        target[s.key] = s
    return target
