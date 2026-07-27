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
"""The executor: walks a scenario's arms and produces results that may be read.

:mod:`sglang.srt.planner.scenarios` says what to measure,
:mod:`sglang.srt.planner.comparison` says what may be concluded, and until now
nothing ran in between. This module is that middle piece. It boots a server per
measurement point through :class:`~sglang.srt.planner.server_manager.
SglangSupervisor`, drives ``sglang.benchmark.serving`` as the load generator,
reads the engine counters over the same wall clock through
:mod:`sglang.srt.rigmon.sources` and :mod:`sglang.srt.rigmon.rates`, and emits
:class:`~sglang.srt.planner.comparison.ArmResult` objects.

Five rules are enforced by the SHAPE of the code, not by an operator
remembering them. Each exists because its absence has already produced a wrong
number in this project.

**The noise floor comes first, and it is boot-to-boot.** :meth:`Study.plan`
puts the A-vs-A boots at the head of the schedule and :meth:`Study.run` will
not compare anything before they have run. Repeats are separate BOOTS by
construction — there is no within-boot repeat mode, because a within-boot floor
is measured under a warm cache and a settled clock and is therefore narrower
than the effect it is supposed to bound. A within-boot floor produced an
apparent +2.80 % gain in this campaign that survived nothing.

**Arms are interleaved, never blocked.** ``A,B,A,B`` rather than ``A,A,B,B``.
A 3080 coming out of idle runs unthrottled for the first tens of seconds, so a
blocked schedule systematically favours whichever arm went first.

**Windows are kept apart.** A point produces one :class:`WindowResult` per
declared window and never a pooled mean. A window the runner cannot drive is
emitted EMPTY with the reason, so its absence cannot be read as agreement, and
``exclude_from_headline`` travels from the scenario onto the result where
:func:`~sglang.srt.planner.comparison.headline` already refuses it.

**The KV budget is neutralised before every arm.**
``~/.cache/sglang/kv_budget-<confighash>.json`` is written by one boot and
consumed by the next with the same fingerprint, which has produced a 4x swing
in ``max_total_num_tokens`` from boot ORDER alone. Every point therefore either
clears the file or pins ``SGLANG_UNEVEN_TOKEN_VECTOR``; a policy that does
neither is rejected at construction.

**A point has a time budget and an age and a state.** 10-20 s of load per
point, a hard 60 s ceiling, and an abort with a NAMED reason rather than an
overrun. Clock, temperature and throttle reasons per card travel with every
point; a throttled point is KEPT and MARKED, never dropped.

Teardown is scoped to the process GROUP the supervisor spawned and to the
harness process group this module spawned. There is no ``pkill`` here, and the
free-VRAM gate waits only on pids this process started — a foreign process is
reported, never waited for, because a gate phrased against the whole card
blocks forever on somebody else's server.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shlex
import signal
import statistics
import subprocess
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sglang.srt.planner.comparison import (
    ArmResult,
    Comparison,
    NoiseFloor,
    WindowResult,
    compare_arms,
)
from sglang.srt.planner.scenarios import Scenario, build_harness_command

__all__ = [
    "TimeBudget",
    "RunPolicy",
    "Arm",
    "ScheduledBoot",
    "WindowStep",
    "PointResult",
    "StudyResult",
    "Study",
    "KvBudgetUnpinned",
    "WithinBootRefused",
    "NoiseFloorMissing",
    "build_schedule",
    "window_plan",
    "neutralise_kv_budget",
    "own_vram_gate",
    "card_state",
    "window_metrics",
    "noise_floor_from_points",
    "arm_result_from_points",
    "suggest_num_prompts",
    "HarnessOutcome",
    "SubprocessHarness",
    "load_study",
    "render_study_text",
    "render_dry_run_text",
]


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


class WithinBootRefused(ValueError):
    """Raised when a noise floor is asked for from repeats inside one boot."""


class KvBudgetUnpinned(ValueError):
    """Raised when neither budget reset nor a pinned token vector was chosen."""


class NoiseFloorMissing(RuntimeError):
    """Raised when a comparison is requested before the A-vs-A arm has run."""


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TimeBudget:
    """How long one measurement point's load may take.

    The band rather than a single number: below ``target_low_s`` a point is
    dominated by ramp-up and its figures are not a steady-state rate; above
    ``target_high_s`` a sweep stops fitting into a session. ``ceiling_s`` is not
    a target at all — it is where the runner stops waiting and says why.
    """

    target_low_s: float = 10.0
    target_high_s: float = 20.0
    ceiling_s: float = 60.0

    def __post_init__(self):
        if not 0 < self.target_low_s <= self.target_high_s:
            raise ValueError(
                f"time budget band is inverted: {self.target_low_s} > "
                f"{self.target_high_s}"
            )
        if self.ceiling_s < self.target_high_s:
            raise ValueError(
                f"ceiling {self.ceiling_s}s is below the target band's upper "
                f"end {self.target_high_s}s; the ceiling would fire on every "
                "point that is merely slow"
            )

    def verdict(self, duration_s: float) -> str:
        """ "" (in band) | "short" | "over_target" | "ceiling"."""
        if duration_s > self.ceiling_s:
            return "ceiling"
        if duration_s > self.target_high_s:
            return "over_target"
        if duration_s < self.target_low_s:
            return "short"
        return ""


@dataclasses.dataclass
class RunPolicy:
    """Everything about HOW the points are run, separate from WHAT is run."""

    budget: TimeBudget = dataclasses.field(default_factory=TimeBudget)
    #: Boots of the baseline arm, each a full boot, that establish the floor.
    #: Two is the minimum that has a spread at all; three is the default
    #: because a two-point range is one unlucky boot wide.
    noise_floor_boots: int = 3
    #: How often the interleaved A,B,... sequence is repeated.
    comparison_repeats: int = 2
    #: Only value accepted. Named rather than implied so that a future caller
    #: asking for the cheap variant gets the reason instead of the number.
    noise_floor_mode: str = "boot_to_boot"
    #: Load size handed to the harness. Held FIXED across arms: adjusting it
    #: per arm to hit the time band would make the arms incomparable, which is
    #: why :func:`suggest_num_prompts` only advises.
    num_prompts: int = 64
    #: Seconds to let the server settle after readiness before the first
    #: window opens.
    settle_s: float = 5.0
    boot_timeout_s: float = 600.0
    stop_grace_s: float = 20.0
    #: #188: clear ``kv_budget-<hash>.json`` before every point.
    reset_kv_budget: bool = True
    #: ...or pin the split instead, which makes the file irrelevant.
    pin_token_vector: Optional[str] = None
    #: How long the runner waits for ITS OWN previous processes to hand back
    #: VRAM. Foreign occupancy is recorded, never waited for.
    own_vram_timeout_s: float = 120.0
    #: Extra env for every boot (e.g. the two switches per-rank columns need).
    env: Dict[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.noise_floor_mode != "boot_to_boot":
            raise WithinBootRefused(
                f"noise_floor_mode={self.noise_floor_mode!r} is not available. "
                "The floor is measured boot-to-boot because that is the "
                "comparison it has to bound: two arms are two boots. Repeats "
                "inside one boot run on a warm cache and a settled clock and "
                "produce a narrower spread than the thing being measured — in "
                "this campaign a within-boot floor certified an apparent "
                "+2.80 % gain that boot-to-boot repetition did not reproduce."
            )
        if self.noise_floor_boots < 2:
            raise ValueError(
                f"noise_floor_boots={self.noise_floor_boots}: a spread needs at "
                "least two boots. With one there is no floor, and every "
                "comparison downstream is 'unknown'."
            )
        if self.comparison_repeats < 1:
            raise ValueError("comparison_repeats must be >= 1")
        if not self.reset_kv_budget and not self.pin_token_vector:
            raise KvBudgetUnpinned(
                "the persisted KV budget is neither reset nor pinned. "
                "~/.cache/sglang/kv_budget-<confighash>.json is written by one "
                "boot and consumed by the next with the same fingerprint, so "
                "the arm that happens to boot first fixes max_total_num_tokens "
                "for the arm that follows — a 4x swing has been observed from "
                "boot order alone. Set reset_kv_budget=True, or pin the split "
                "with pin_token_vector."
            )


@dataclasses.dataclass
class Arm:
    """One configuration under test.

    ``settings`` is a ``server_manager.LaunchSettings``; it is not imported
    here so this module stays usable (and testable) without the sglang import
    chain the supervisor pulls in.
    """

    label: str
    settings: Any
    #: Launch env for this arm ON TOP of the policy's.
    env: Dict[str, str] = dataclasses.field(default_factory=dict)
    #: Comparability terms known before the run (model, quant, prompt set...).
    #: Measured terms (accept length, batch size) are filled in from the run.
    conditions: Dict[str, Any] = dataclasses.field(default_factory=dict)
    note: str = ""


# ---------------------------------------------------------------------------
# The schedule
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ScheduledBoot:
    """One boot. There is no other unit: a repeat IS a boot."""

    order: int
    arm: str
    repeat: int
    #: "noise_floor" | "comparison"
    role: str

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


def build_schedule(
    arms: Sequence[str], policy: Optional[RunPolicy] = None
) -> List[ScheduledBoot]:
    """The boot order: floor first, then interleaved.

    The floor boots come first unconditionally — a floor computed after the
    comparison would be a floor for a differently warmed machine, and one
    computed from the comparison's own points would be the effect measuring its
    own threshold. The comparison boots then alternate between arms so that
    monotone drift (a card warming up over the session) lands on both arms
    equally instead of on whichever went first.
    """
    policy = policy or RunPolicy()
    if not arms:
        raise ValueError("a schedule needs at least one arm")
    out: List[ScheduledBoot] = []
    order = 0
    baseline = arms[0]
    for r in range(policy.noise_floor_boots):
        out.append(ScheduledBoot(order, baseline, r, "noise_floor"))
        order += 1
    if len(arms) > 1:
        for r in range(policy.comparison_repeats):
            for label in arms:
                out.append(ScheduledBoot(order, label, r, "comparison"))
                order += 1
    return out


# ---------------------------------------------------------------------------
# The window plan
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class WindowStep:
    """One window of one point, and how the runner intends to fill it."""

    window: str
    excluded_from_headline: bool = False
    #: True when this window is filled by running the scenario's harness.
    drives_harness: bool = False
    #: Set when the runner cannot fill it; the window is still emitted.
    undrivable_reason: str = ""

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


#: The single window a scenario without declared windows gets. Named rather
#: than left implicit so a result cannot claim a window the scenario never
#: described.
DEFAULT_WINDOW = "steady"


def window_plan(
    scenario: Scenario, drivers: Optional[Dict[str, Callable]] = None
) -> List[WindowStep]:
    """Turn a scenario's declared windows into steps this runner can execute.

    A scenario that declares no windows gets one harness-driven ``steady``
    window. A scenario that declares windows gets one step per window: the
    first one the runner can drive carries the harness, the rest need a driver
    from the caller because their boundaries are state changes (a spill, a
    restore) that only the caller can cause. A window without a driver is kept
    in the plan with the reason attached — dropping it would turn a partial
    measurement into what looks like a complete one.
    """
    drivers = drivers or {}
    if not scenario.windows:
        return [WindowStep(DEFAULT_WINDOW, False, True)]
    steps: List[WindowStep] = []
    harness_placed = False
    for w in scenario.windows:
        if w.key in drivers:
            steps.append(WindowStep(w.key, w.exclude_from_headline, False))
            continue
        if not harness_placed and not w.exclude_from_headline:
            steps.append(WindowStep(w.key, w.exclude_from_headline, True))
            harness_placed = True
            continue
        steps.append(
            WindowStep(
                w.key,
                w.exclude_from_headline,
                False,
                undrivable_reason=(
                    f"window {w.key!r} opens at {w.starts_at or 'a state change'!r}, "
                    "which this runner cannot cause on its own. Supply a driver "
                    "for it, or read this window as not measured — it is "
                    "reported empty rather than omitted so its absence is not "
                    "mistaken for agreement."
                ),
            )
        )
    return steps


# ---------------------------------------------------------------------------
# Preflight: the #188 trap and the VRAM gate
# ---------------------------------------------------------------------------


def neutralise_kv_budget(
    policy: RunPolicy,
    cache_dir: Optional[str] = None,
    lister: Optional[Callable[[str], List[str]]] = None,
    resetter: Optional[Callable[[str], dict]] = None,
) -> Dict[str, Any]:
    """Make the persisted KV budget irrelevant to THIS boot. Run before each.

    Two ways out, and the policy has to have picked one (the constructor
    refuses a policy that picked neither):

    * pin ``SGLANG_UNEVEN_TOKEN_VECTOR`` — the split no longer comes from the
      measured budget, so the file cannot move it. Preferred when the arms
      differ in something that would change the fingerprint anyway.
    * clear the file — the boot re-measures. Correct only when nothing else
      holds VRAM at that moment, which is why the returned record carries what
      was removed: if a later result looks odd, the removal is in the log.
    """
    from sglang.srt.rigmon import kvbudget

    cache_dir = cache_dir or kvbudget.CACHE_DIR
    lister = lister or kvbudget.list_budget_files
    resetter = resetter or kvbudget.reset_budget

    if policy.pin_token_vector:
        return {
            "strategy": "pinned",
            "env": {"SGLANG_UNEVEN_TOKEN_VECTOR": policy.pin_token_vector},
            "removed": [],
            "note": (
                "the split is pinned, so the persisted budget cannot set it; "
                "the file is left alone"
            ),
        }

    removed: List[Dict[str, Any]] = []
    for path in lister(cache_dir):
        try:
            removed.append(resetter(path))
        except Exception as e:  # a missing file is fine, a locked one is not
            removed.append({"removed": False, "path": path, "reason": str(e)})
    return {
        "strategy": "reset",
        "env": {},
        "removed": removed,
        "note": (
            "the budget files were cleared so this boot re-measures instead of "
            "inheriting the previous arm's measurement"
        ),
    }


def own_vram_gate(
    own_pids: Sequence[int],
    indices: Sequence[int],
    nvml=None,
    timeout_s: float = 120.0,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    poll_s: float = 1.0,
) -> Dict[str, Any]:
    """Wait until no process WE started still holds VRAM on the target cards.

    Deliberately not a "the cards are free" gate. Other agents and other
    servers run on this box; a gate phrased against total occupancy never
    opens, and the operator's only escape is to disable it, which removes the
    check entirely. So: our own pids block and are waited for, foreign pids are
    reported and never waited for, and even the own-pid wait has a deadline
    after which the gate returns ``clear=False`` with the pids named.
    """
    own = {int(p) for p in own_pids if p}
    record: Dict[str, Any] = {
        "clear": True,
        "reason": "",
        "waited_s": 0.0,
        "foreign": [],
        "own_holding": [],
    }
    if nvml is None:
        record["reason"] = "NVML unavailable; the gate could not look and did not block"
        return record

    start = clock()
    deadline = start + timeout_s
    while True:
        foreign: List[Dict[str, Any]] = []
        holding: List[Dict[str, Any]] = []
        try:
            count = nvml.nvmlDeviceGetCount()
            for i in indices or range(count):
                if not 0 <= i < count:
                    continue
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                for proc in nvml.nvmlDeviceGetComputeRunningProcesses(handle) or []:
                    entry = {
                        "gpu": i,
                        "pid": int(getattr(proc, "pid", 0)),
                        "used_mib": int(
                            (getattr(proc, "usedGpuMemory", 0) or 0) / 2**20
                        ),
                    }
                    (holding if entry["pid"] in own else foreign).append(entry)
        except Exception as e:
            record["reason"] = f"NVML query failed ({e}); the gate did not block"
            return record

        record["foreign"] = foreign
        record["own_holding"] = holding
        record["waited_s"] = clock() - start
        if not holding:
            if foreign:
                record["reason"] = (
                    f"{len(foreign)} foreign process(es) hold VRAM on these "
                    "cards. Recorded, not waited for: this run does not own "
                    "them, and a gate that waits for them never opens. The "
                    "measurement runs alongside them and the residency is in "
                    "the provenance."
                )
            return record
        if clock() >= deadline:
            record["clear"] = False
            pids = ", ".join(str(h["pid"]) for h in holding)
            record["reason"] = (
                f"our own process(es) {pids} still hold VRAM after "
                f"{timeout_s:.0f}s. Aborting this point rather than measuring "
                "on a card we have not finished vacating."
            )
            return record
        sleep(poll_s)


def card_state(samples: Sequence[Any]) -> List[Dict[str, Any]]:
    """Clock, temperature and throttle reasons per card, for one instant.

    Every point carries this. A recommendation derived from a card at 88 C on
    a software thermal slowdown is a recommendation about a summer afternoon,
    and without the state written down there is no way to tell afterwards.
    """
    out: List[Dict[str, Any]] = []
    for s in samples or []:
        throttles = []
        try:
            throttles = list(s.performance_throttles())
        except Exception:
            throttles = list(getattr(s, "throttle", []) or [])
        ratio = None
        try:
            ratio = s.clock_ratio()
        except Exception:
            pass
        out.append(
            {
                "index": getattr(s, "index", None),
                "name": getattr(s, "name", ""),
                "sm_clock_mhz": getattr(s, "sm_clock_mhz", None),
                "sm_clock_max_mhz": getattr(s, "sm_clock_max_mhz", None),
                "clock_ratio": ratio,
                "temp_c": getattr(s, "temp_c", None),
                "power_w": getattr(s, "power_w", None),
                "throttle_reasons": throttles,
                "throttled": bool(throttles),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Metrics for one window
# ---------------------------------------------------------------------------


def _is_harness_field(spec: str) -> bool:
    """Whether a scenario's ``metric_fields`` entry names a real harness field.

    The registry writes ``(collector: ...)`` for the metrics the harness does
    NOT produce, so the leading parenthesis is a discriminator the data already
    carries; no second table has to be kept in sync with it.
    """
    return bool(spec) and not spec.strip().startswith("(")


def window_metrics(
    engine_before,
    engine_after,
    dt_s: Optional[float],
    harness_result: Optional[Dict[str, Any]] = None,
    metric_fields: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, float], Dict[str, int], List[str]]:
    """Metrics for one window, from the two sources joined over one wall clock.

    Round times come from the engine's phase-labelled forward-time counter
    differenced across the window (:func:`~sglang.srt.rigmon.rates.round_time`),
    percentiles and throughput from the harness's own result. Nothing is
    invented in between: a metric neither side produced is absent, and the
    third return value says why, so an empty column reads as "the device timer
    was off" rather than as zero.
    """
    from sglang.srt.rigmon.rates import group_throughput, phase_seconds, round_time

    notes: List[str] = []
    metrics: Dict[str, float] = {}
    samples: Dict[str, int] = {}

    before_phase = getattr(engine_before, "per_rank_phase", None) or {}
    after_phase = getattr(engine_after, "per_rank_phase", None) or {}
    before_metrics = getattr(engine_before, "metrics", None) or {}
    after_metrics = getattr(engine_after, "metrics", None) or {}

    phase_delta = phase_seconds(after_phase, before_phase)
    if not phase_delta:
        notes.append(
            "no phase-labelled forward time in this window: the round times are "
            "absent, not zero. Boot with SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 "
            "(and --enable-metrics-for-all-schedulers for the per-rank split)."
        )
    rt = round_time(after_metrics, before_metrics, phase_delta, dt_s)
    for key in (
        "ms_per_verify_round",
        "ms_per_decode_round",
        "ms_per_1k_prefill_tokens",
        "ms_per_draft_pass",
        "accept_length",
        "verify_ct",
    ):
        value = getattr(rt, key, None)
        if value is not None:
            metrics[key] = float(value)
            samples[key] = 1

    gt = group_throughput(after_metrics, before_metrics, dt_s)
    if gt.gen_tok_s is not None:
        metrics["tok_s"] = float(gt.gen_tok_s)
        samples["tok_s"] = 1
        notes.append(f"tok_s source: {gt.source}")

    for metric_key, spec in (metric_fields or {}).items():
        if not _is_harness_field(spec):
            continue
        if not harness_result or spec not in harness_result:
            continue
        value = harness_result[spec]
        if isinstance(value, (int, float)):
            metrics[metric_key] = float(value)
            samples[metric_key] = int(harness_result.get("completed") or 0) or 1

    return metrics, samples, notes


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class HarnessOutcome:
    """What one harness invocation produced."""

    ok: bool
    duration_s: float
    result: Optional[Dict[str, Any]] = None
    returncode: Optional[int] = None
    reason: str = ""
    command: str = ""

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        # The per-request detail arrays are large and belong in the raw file,
        # not in every result record.
        if isinstance(d.get("result"), dict):
            d["result"] = {
                k: v for k, v in d["result"].items() if not isinstance(v, list)
            }
        return d


class SubprocessHarness:
    """Runs ``sglang.benchmark.serving`` in its own process group.

    Its own group so that a point that hits the ceiling can be ended by
    signalling exactly that group. Nothing here matches on process names, and
    nothing signals a pid this class did not spawn.
    """

    def __init__(self, python_exe: Optional[str] = None, env: Optional[Dict] = None):
        import sys

        self.python_exe = python_exe or sys.executable
        self.env = env

    def run(self, command: str, timeout_s: float) -> HarnessOutcome:
        argv = shlex.split(command)
        if argv and argv[0] == "python":
            argv[0] = self.python_exe
        fd, out_path = tempfile.mkstemp(prefix="rigrun_", suffix=".jsonl")
        os.close(fd)
        os.unlink(out_path)
        argv += ["--output-file", out_path]

        env = dict(os.environ)
        env.update(self.env or {})
        start = time.time()
        proc = subprocess.Popen(
            argv,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        pgid = os.getpgid(proc.pid)
        try:
            proc.communicate(timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    break
                try:
                    proc.communicate(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    continue
        duration = time.time() - start

        result = None
        try:
            with open(out_path) as f:
                lines = [ln for ln in f.read().splitlines() if ln.strip()]
            if lines:
                result = json.loads(lines[-1])
        except Exception:
            result = None
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass

        if timed_out:
            return HarnessOutcome(
                ok=False,
                duration_s=duration,
                result=result,
                reason=(
                    f"the load exceeded the {timeout_s:.0f}s ceiling and was "
                    "ended (its own process group, SIGTERM then SIGKILL). The "
                    "point is reported aborted rather than overrunning; size "
                    "the load down or raise the ceiling deliberately."
                ),
                command=command,
            )
        if proc.returncode != 0:
            return HarnessOutcome(
                ok=False,
                duration_s=duration,
                result=result,
                returncode=proc.returncode,
                reason=f"harness exited with rc={proc.returncode}",
                command=command,
            )
        if result is None:
            return HarnessOutcome(
                ok=False,
                duration_s=duration,
                returncode=0,
                reason=(
                    "the harness exited cleanly but wrote no result record; "
                    "nothing may be read from this point"
                ),
                command=command,
            )
        return HarnessOutcome(
            ok=True,
            duration_s=duration,
            result=result,
            returncode=0,
            command=command,
        )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PointResult:
    """One boot, one point of the parameter space, one set of windows."""

    arm: str
    repeat: int
    role: str
    order: int = 0
    started_at: float = 0.0
    #: Load seconds, per window.
    durations_s: Dict[str, float] = dataclasses.field(default_factory=dict)
    #: "" | "short" | "over_target" | "ceiling", per window.
    budget_verdicts: Dict[str, str] = dataclasses.field(default_factory=dict)
    windows: List[WindowResult] = dataclasses.field(default_factory=list)
    #: Card state before and after the point; both, because the question is
    #: usually whether the card heated up DURING it.
    state_before: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    state_after: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    #: A throttled point is kept and marked. Dropping it would silently select
    #: for cool moments and report a rig that does not exist.
    throttled: bool = False
    #: Non-empty when the point did not complete; names the reason.
    aborted: str = ""
    conditions: Dict[str, Any] = dataclasses.field(default_factory=dict)
    provenance: Dict[str, Any] = dataclasses.field(default_factory=dict)
    notes: List[str] = dataclasses.field(default_factory=list)

    @property
    def usable(self) -> bool:
        return not self.aborted and bool(self.windows)

    def window(self, key: str) -> Optional[WindowResult]:
        for w in self.windows:
            if w.window == key:
                return w
        return None

    def age_s(self, now: Optional[float] = None) -> float:
        return max(0.0, (now if now is not None else time.time()) - self.started_at)

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["usable"] = self.usable
        return d


@dataclasses.dataclass
class StudyResult:
    """Everything one study produced, including what it failed to produce."""

    scenario: str
    schedule: List[ScheduledBoot] = dataclasses.field(default_factory=list)
    points: List[PointResult] = dataclasses.field(default_factory=list)
    noise: Optional[NoiseFloor] = None
    arms: List[ArmResult] = dataclasses.field(default_factory=list)
    comparisons: List[Comparison] = dataclasses.field(default_factory=list)
    aborted: str = ""
    notes: List[str] = dataclasses.field(default_factory=list)

    def points_for(self, arm: str, role: str = "") -> List[PointResult]:
        return [p for p in self.points if p.arm == arm and (not role or p.role == role)]

    def to_json(self) -> dict:
        return {
            "scenario": self.scenario,
            "schedule": [s.to_json() for s in self.schedule],
            "points": [p.to_json() for p in self.points],
            "noise": self.noise.to_json() if self.noise else None,
            "arms": [a.to_json() for a in self.arms],
            "comparisons": [c.to_json() for c in self.comparisons],
            "aborted": self.aborted,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Reducing points
# ---------------------------------------------------------------------------


def noise_floor_from_points(
    points: Sequence[PointResult],
    window: str = DEFAULT_WINDOW,
    source: str = "",
) -> NoiseFloor:
    """The detection threshold per metric, from boot-to-boot A-vs-A repeats.

    The statistic is the full relative range, ``(max - min) / median``, not a
    standard deviation. The comparison downstream puts ONE run of A against ONE
    run of B, so the question is how far apart two same-configuration runs can
    land — that is the range, and a standard deviation would understate it by
    construction. With few boots the range is a lower bound on the true one, so
    the count travels in ``source``.

    Metrics measured in fewer than two boots get no entry, which makes
    :func:`~sglang.srt.planner.comparison.compare_metric` return ``unknown``
    for them. That is the intended outcome: a metric without a floor has no
    resolution, and reporting it as unchanged would be a claim the data does
    not support.
    """
    usable = [p for p in points if p.usable]
    by_metric: Dict[str, List[float]] = {}
    for p in usable:
        w = p.window(window)
        if w is None:
            continue
        for key, value in w.metrics.items():
            if isinstance(value, (int, float)):
                by_metric.setdefault(key, []).append(float(value))

    relative: Dict[str, float] = {}
    for key, values in by_metric.items():
        if len(values) < 2:
            continue
        mid = statistics.median(values)
        if not mid:
            continue
        relative[key] = (max(values) - min(values)) / abs(mid)

    throttled = sum(1 for p in usable if p.throttled)
    parts = [
        source or "A-vs-A",
        f"{len(usable)} boot-to-boot repeats",
        f"window {window!r}",
        "spread = (max - min) / median",
    ]
    if throttled:
        parts.append(
            f"{throttled} of the repeats were throttled; they widen the floor "
            "rather than being dropped"
        )
    return NoiseFloor(relative=relative, source="; ".join(parts))


def arm_result_from_points(
    label: str,
    points: Sequence[PointResult],
    scenario: Optional[Scenario] = None,
) -> ArmResult:
    """Fold repeats of one arm into one :class:`ArmResult`.

    The median across repeats, per metric per window — the median because a
    single boot that lost a card to a thermal event should not move the figure
    the way a mean would, and because the noise floor next to it is a range
    statistic that already carries the spread.

    Aborted points are excluded from the figures and counted in the note:
    their windows are partial, so folding them in would report a load that did
    not run. Throttled points ARE folded in and the note says how many.
    """
    usable = [p for p in points if p.usable]
    windows: Dict[str, Dict[str, List[float]]] = {}
    samples: Dict[str, Dict[str, int]] = {}
    excluded: Dict[str, bool] = {}
    notes: Dict[str, List[str]] = {}
    for p in usable:
        for w in p.windows:
            slot = windows.setdefault(w.window, {})
            excluded[w.window] = (
                excluded.get(w.window, False) or w.excluded_from_headline
            )
            for key, value in w.metrics.items():
                slot.setdefault(key, []).append(float(value))
            for key, n in w.samples.items():
                cur = samples.setdefault(w.window, {})
                cur[key] = cur.get(key, 0) + int(n)
            if w.note:
                bucket = notes.setdefault(w.window, [])
                if w.note not in bucket:
                    bucket.append(w.note)

    order = (
        [w.key for w in scenario.windows]
        if scenario is not None and scenario.windows
        else list(windows)
    )
    for key in windows:
        if key not in order:
            order.append(key)

    results: List[WindowResult] = []
    for key in order:
        if key not in windows:
            continue
        slot = windows[key]
        results.append(
            WindowResult(
                window=key,
                metrics={m: statistics.median(v) for m, v in slot.items() if v},
                samples=dict(samples.get(key, {})),
                excluded_from_headline=excluded.get(key, False),
                note="; ".join(notes.get(key, [])),
            )
        )

    conditions: Dict[str, Any] = {}
    for p in usable:
        conditions.update(p.conditions)
    accepts = [
        w.metrics["accept_length"]
        for p in usable
        for w in p.windows
        if "accept_length" in w.metrics
    ]
    if accepts:
        conditions["accept_length"] = statistics.median(accepts)

    throttled = sum(1 for p in usable if p.throttled)
    dropped = [p for p in points if not p.usable]
    provenance = {
        "boots_run": len(points),
        "boots_used": len(usable),
        "boots_throttled": throttled,
        "boots_aborted": [{"repeat": p.repeat, "reason": p.aborted} for p in dropped],
        "state_note": (
            f"{throttled} of {len(usable)} boots ran throttled; kept and "
            "marked, not dropped"
            if throttled
            else "no throttling observed in the boots used"
        ),
        "newest_point_age_s": (min(p.age_s() for p in usable) if usable else None),
    }
    return ArmResult(
        label=label,
        scenario=scenario.key if scenario is not None else "",
        windows=results,
        conditions=conditions,
        provenance=provenance,
    )


def suggest_num_prompts(
    duration_s: float, num_prompts: int, budget: Optional[TimeBudget] = None
) -> Dict[str, Any]:
    """Advise a load size for the time band — advice, never applied mid-study.

    Changing the load between arms changes the amount of work behind each
    figure, which is exactly the condition
    :func:`~sglang.srt.planner.comparison.compare_arms` calls
    ``not_comparable``. So the runner reports the number to use NEXT TIME and
    leaves the current study on the size it started with.
    """
    budget = budget or TimeBudget()
    verdict = budget.verdict(duration_s)
    if not verdict or duration_s <= 0 or num_prompts <= 0:
        return {"verdict": verdict, "suggested": num_prompts, "advice": ""}
    midpoint = (budget.target_low_s + budget.target_high_s) / 2.0
    suggested = max(1, int(round(num_prompts * midpoint / duration_s)))
    return {
        "verdict": verdict,
        "suggested": suggested,
        "advice": (
            f"a point took {duration_s:.1f}s at --num-prompts {num_prompts}; "
            f"{suggested} would land near {midpoint:.0f}s. Apply it to the NEXT "
            "study, not this one: changing the load between arms makes them "
            "not_comparable."
        ),
    }


# ---------------------------------------------------------------------------
# The study
# ---------------------------------------------------------------------------


class Study:
    """Runs a scenario's arms and returns results that carry their own caveats.

    Every external dependency is injectable, which is what makes the whole
    executor testable on a box with no GPU: the supervisor, the card sampler,
    the engine scraper factory, the harness, the clock and the sleep. The
    default wiring is the real one.
    """

    def __init__(
        self,
        scenario: Scenario,
        arms: Sequence[Arm],
        policy: Optional[RunPolicy] = None,
        supervisor=None,
        sampler=None,
        scraper_factory: Optional[Callable[[str], Any]] = None,
        harness=None,
        window_drivers: Optional[Dict[str, Callable]] = None,
        point: Optional[Dict[str, Any]] = None,
        nvml=None,
        kv_cache_dir: Optional[str] = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not arms:
            raise ValueError("a study needs at least one arm")
        labels = [a.label for a in arms]
        if len(set(labels)) != len(labels):
            raise ValueError(f"arm labels must be unique, got {labels}")
        self.scenario = scenario
        self.arms = list(arms)
        self.policy = policy or RunPolicy()
        self.supervisor = supervisor
        self.sampler = sampler
        self.scraper_factory = scraper_factory
        self.harness = harness
        self.window_drivers = dict(window_drivers or {})
        self.point = dict(point or {})
        self.nvml = nvml
        self.kv_cache_dir = kv_cache_dir
        self.clock = clock
        self.sleep = sleep
        self._own_pids: List[int] = []

    # -- planning --------------------------------------------------------

    def plan(self) -> List[ScheduledBoot]:
        return build_schedule([a.label for a in self.arms], self.policy)

    def steps(self) -> List[WindowStep]:
        return window_plan(self.scenario, self.window_drivers)

    def dry_run(self) -> Dict[str, Any]:
        """Everything the study would do, without doing any of it.

        The point of a dry run here is not reassurance: it is that the two
        failure modes which cost the most (a harness command that cannot
        express an axis, and a missing device timer that empties every round
        time) are both visible before the first boot, and both take multiple
        boots to notice afterwards.
        """
        plan = self.plan()
        arms = []
        for arm in self.arms:
            cmd = self.harness_command(arm)
            arms.append(
                {
                    "label": arm.label,
                    "runnable": bool(cmd.get("runnable")),
                    "reason": cmd.get("reason", ""),
                    "command": cmd.get("command", ""),
                    "env": self._launch_env(arm),
                    "launch": _launch_argv(arm),
                }
            )
        primary = self.scenario.primary_metric
        return {
            "scenario": self.scenario.key,
            "question": self.scenario.question,
            "primary_metric": primary.key if primary else "",
            "boots": len(plan),
            "schedule": [b.to_json() for b in plan],
            "windows": [s.to_json() for s in self.steps()],
            "arms": arms,
            "preflight": self.preflight(),
            "budget": dataclasses.asdict(self.policy.budget),
            "estimated_load_s": len(plan)
            * len([s for s in self.steps() if not s.undrivable_reason])
            * self.policy.budget.target_high_s,
            "runnable": all(a["runnable"] for a in arms),
        }

    def preflight(self) -> List[str]:
        """What the operator has to know before this runs, in order."""
        tail = (
            f"then {self.policy.comparison_repeats}x interleaved over "
            f"{len(self.arms)} arms"
            if len(self.arms) > 1
            else "one arm only, so the floor is the whole schedule"
        )
        out = [
            f"schedule: {len(self.plan())} boots "
            f"({self.policy.noise_floor_boots} A-vs-A first, {tail})",
            (
                "the floor is measured boot-to-boot; repeats inside one boot "
                "are not offered"
            ),
        ]
        if self.policy.pin_token_vector:
            out.append(
                "SGLANG_UNEVEN_TOKEN_VECTOR is pinned to "
                f"{self.policy.pin_token_vector!r}, so the persisted KV budget "
                "cannot set the split"
            )
        else:
            out.append(
                "the persisted KV budget is cleared before every boot; make "
                "sure no other process holds VRAM at that moment, or the "
                "re-measurement inherits its occupancy"
            )
        for step in self.steps():
            if step.undrivable_reason:
                out.append(f"window {step.window}: {step.undrivable_reason}")
        env = self._launch_env(self.arms[0])
        if env.get("SGLANG_ENABLE_METRICS_DEVICE_TIMER") != "1":
            out.append(
                "SGLANG_ENABLE_METRICS_DEVICE_TIMER is not set to 1: the "
                "forward-time counter will be ABSENT and every round time with "
                "it. Set it, and --enable-metrics-for-all-schedulers for the "
                "per-rank split."
            )
        return out

    def _launch_env(self, arm: Arm) -> Dict[str, str]:
        env = dict(self.policy.env)
        env.update(arm.env)
        if self.policy.pin_token_vector:
            env["SGLANG_UNEVEN_TOKEN_VECTOR"] = self.policy.pin_token_vector
        return env

    # -- the harness command ---------------------------------------------

    def harness_command(self, arm: Arm) -> Dict[str, Any]:
        settings = arm.settings
        base_url = "http://{}:{}".format(
            getattr(settings, "host", "127.0.0.1"),
            getattr(settings, "port", 30000),
        )
        cmd = build_harness_command(
            self.scenario,
            self.point,
            base_url=base_url,
            num_prompts=self.policy.num_prompts,
        )
        if not cmd.get("runnable"):
            reason = cmd.get("reason") or (
                "axes {} have no harness flag and are not declared external, so "
                "the sweep would silently collapse to one point".format(
                    cmd.get("unmapped_axes")
                )
            )
            cmd["reason"] = reason
        elif cmd.get("external"):
            cmd["runnable"] = False
            cmd["reason"] = (
                "axes {} are host or server controls the runner does not set: "
                "{}. Apply them before the study, or split the study so each "
                "value is its own arm — running them unset would repeat the "
                "same point and look like a completed sweep.".format(
                    [e["axis"] for e in cmd["external"]],
                    "; ".join(
                        f"{e['label']} = {e['value']} ({e['apply']})"
                        for e in cmd["external"]
                    ),
                )
            )
        return cmd

    # -- one point -------------------------------------------------------

    def run_point(self, boot: ScheduledBoot) -> PointResult:
        arm = next(a for a in self.arms if a.label == boot.arm)
        result = PointResult(
            arm=boot.arm,
            repeat=boot.repeat,
            role=boot.role,
            order=boot.order,
            started_at=self.clock(),
            conditions=dict(arm.conditions),
        )

        cmd = self.harness_command(arm)
        if not cmd.get("runnable"):
            result.aborted = cmd.get("reason", "the harness command is not runnable")
            return result

        budget = neutralise_kv_budget(self.policy, self.kv_cache_dir)
        result.provenance["kv_budget"] = budget

        indices = sorted(set(getattr(arm.settings, "rank_gpu_id", None) or []))
        gate = own_vram_gate(
            self._own_pids,
            indices,
            nvml=self.nvml,
            timeout_s=self.policy.own_vram_timeout_s,
            clock=self.clock,
            sleep=self.sleep,
        )
        result.provenance["vram_gate"] = gate
        if not gate["clear"]:
            result.aborted = gate["reason"]
            return result

        env = self._launch_env(arm)
        env.update(budget.get("env") or {})
        result.provenance["launch_env"] = dict(env)

        try:
            self._boot(arm, env)
        except Exception as e:
            result.aborted = f"boot failed: {type(e).__name__}: {e}"
            self._teardown()
            return result

        try:
            result.state_before = card_state(self._sample_cards())
            if self.policy.settle_s:
                self.sleep(self.policy.settle_s)
            self._run_windows(result, arm, cmd)
            result.state_after = card_state(self._sample_cards())
        finally:
            self._teardown()

        result.throttled = any(
            c.get("throttled")
            for c in list(result.state_before) + list(result.state_after)
        )
        if result.throttled:
            result.notes.append(
                "a card was throttled during this point. The point is kept and "
                "marked: dropping throttled points selects for cool moments and "
                "reports a rig that does not exist."
            )
        result.conditions.setdefault("batch_size", self.policy.num_prompts)
        return result

    def _boot(self, arm: Arm, env: Dict[str, str]) -> None:
        if self.supervisor is None:
            raise RuntimeError("no supervisor was supplied; nothing can be booted")
        settings = arm.settings
        try:
            settings.extra_env = dict(getattr(settings, "extra_env", None) or {})
            settings.extra_env.update(env)
        except Exception:
            pass
        self.supervisor.start(
            settings,
            wait_ready=True,
            ready_timeout_s=self.policy.boot_timeout_s,
        )
        pid = getattr(self.supervisor, "proc", None)
        pid = getattr(pid, "pid", None)
        if pid:
            self._own_pids.append(int(pid))

    def _teardown(self) -> None:
        if self.supervisor is None:
            return
        try:
            self.supervisor.stop(grace_s=self.policy.stop_grace_s)
        except Exception:
            pass

    def _sample_cards(self) -> List[Any]:
        if self.sampler is None:
            return []
        try:
            return list(self.sampler.sample())
        except Exception:
            return []

    def _scraper(self, arm: Arm):
        base_url = "http://{}:{}".format(
            getattr(arm.settings, "host", "127.0.0.1"),
            getattr(arm.settings, "port", 30000),
        )
        if self.scraper_factory is not None:
            return self.scraper_factory(base_url)
        from sglang.srt.rigmon.sources import EngineScraper

        return EngineScraper(base_url)

    def _run_windows(self, result: PointResult, arm: Arm, cmd: Dict[str, Any]) -> None:
        scraper = self._scraper(arm)
        metric_fields = cmd.get("metric_fields") or {}
        for step in self.steps():
            if step.undrivable_reason:
                result.windows.append(
                    WindowResult(
                        window=step.window,
                        excluded_from_headline=step.excluded_from_headline,
                        note=step.undrivable_reason,
                    )
                )
                continue

            before = scraper.scrape()
            t0 = self.clock()
            outcome: Optional[HarnessOutcome] = None
            failure = ""
            if step.drives_harness:
                harness = self.harness or SubprocessHarness(
                    python_exe=getattr(arm.settings, "python_exe", None)
                )
                outcome = harness.run(
                    cmd["command"], timeout_s=self.policy.budget.ceiling_s
                )
                if not outcome.ok:
                    failure = outcome.reason
            else:
                driver = self.window_drivers.get(step.window)
                try:
                    driver(arm, step.window)
                except Exception as e:
                    failure = f"window driver failed: {type(e).__name__}: {e}"
            t1 = self.clock()
            after = scraper.scrape()

            duration = outcome.duration_s if outcome is not None else max(0.0, t1 - t0)
            result.durations_s[step.window] = duration
            verdict = self.policy.budget.verdict(duration)
            result.budget_verdicts[step.window] = verdict

            metrics, samples, notes = window_metrics(
                before,
                after,
                max(1e-9, t1 - t0),
                harness_result=outcome.result if outcome else None,
                metric_fields=metric_fields,
            )
            if verdict == "ceiling":
                notes.append(
                    f"the load ran {duration:.1f}s against a "
                    f"{self.policy.budget.ceiling_s:.0f}s ceiling"
                )
            elif verdict:
                notes.append(
                    f"{duration:.1f}s is {verdict.replace('_', ' ')} for the "
                    f"{self.policy.budget.target_low_s:.0f}-"
                    f"{self.policy.budget.target_high_s:.0f}s band"
                )
            if failure:
                notes.append(failure)
            result.windows.append(
                WindowResult(
                    window=step.window,
                    metrics=metrics,
                    samples=samples,
                    excluded_from_headline=step.excluded_from_headline,
                    note="; ".join(n for n in notes if n),
                )
            )
            if failure:
                result.aborted = failure
                return

            if outcome is not None and outcome.result:
                self._absorb_conditions(result, outcome.result)
                result.provenance.setdefault("harness", []).append(outcome.to_json())

    def _absorb_conditions(
        self, result: PointResult, harness_result: Dict[str, Any]
    ) -> None:
        """Copy the comparability terms the harness reports into the point."""
        mapping = {
            "batch_size": "max_concurrency",
            "prompt_set": "dataset_name",
            "resident_tokens": "total_input_tokens",
        }
        for key, field in mapping.items():
            value = harness_result.get(field)
            if value is not None:
                result.conditions[key] = value
        info = harness_result.get("server_info")
        if isinstance(info, dict):
            for key, field in (("model", "model_path"), ("quant", "quantization")):
                value = info.get(field)
                if value is not None:
                    result.conditions.setdefault(key, value)

    # -- the whole study -------------------------------------------------

    def run(self, noise: Optional[NoiseFloor] = None) -> StudyResult:
        """Floor first, then the interleaved comparison, then the verdicts.

        The order is not a convention here: the floor boots are executed and
        reduced BEFORE the first comparison boot starts, so nothing downstream
        can be handed a threshold derived from the runs it is meant to judge.
        A floor supplied by the caller replaces the A-vs-A boots only if it
        already covers the scenario's primary metric; otherwise the boots run
        anyway and the note says why.
        """
        schedule = self.plan()
        out = StudyResult(scenario=self.scenario.key, schedule=schedule)
        out.notes.extend(self.preflight())

        floor_boots = [b for b in schedule if b.role == "noise_floor"]
        compare_boots = [b for b in schedule if b.role == "comparison"]
        primary = self.scenario.primary_metric
        primary_key = primary.key if primary else ""

        supplied_covers = bool(
            noise and (not primary_key or noise.for_metric(primary_key) is not None)
        )
        if supplied_covers:
            out.noise = noise
            out.notes.append(
                "using the supplied noise floor "
                f"({noise.source or 'no source recorded'}); the A-vs-A boots "
                "were skipped"
            )
        else:
            if noise is not None:
                out.notes.append(
                    "the supplied noise floor has no entry for the primary "
                    f"metric {primary_key!r}, so the A-vs-A arm runs anyway "
                    "rather than the comparison inheriting a floor that does "
                    "not cover it"
                )
            for boot in floor_boots:
                out.points.append(self.run_point(boot))
            out.noise = noise_floor_from_points(
                out.points_for(self.arms[0].label, "noise_floor"),
                window=self._headline_window(),
                source=f"noise_floor arm {self.arms[0].label!r}",
            )

        if primary_key and out.noise.for_metric(primary_key) is None:
            out.notes.append(
                f"no floor for the primary metric {primary_key!r} came out of "
                f"{len(floor_boots)} A-vs-A boots. Every comparison below will "
                "read 'unknown' — which is the correct answer, not a failure of "
                "the comparison: without a threshold, 'no difference' cannot be "
                "told apart from 'no resolution'."
            )

        for boot in compare_boots:
            out.points.append(self.run_point(boot))

        for arm in self.arms:
            points = [
                p for p in out.points if p.arm == arm.label and p.role == "comparison"
            ] or out.points_for(arm.label)
            out.arms.append(arm_result_from_points(arm.label, points, self.scenario))

        if len(out.arms) > 1:
            baseline = out.arms[0]
            for candidate in out.arms[1:]:
                out.comparisons.extend(
                    compare_arms(
                        baseline, candidate, scenario=self.scenario, noise=out.noise
                    )
                )

        durations = [d for p in out.points for d in p.durations_s.values() if d > 0]
        if durations:
            advice = suggest_num_prompts(
                statistics.median(durations),
                self.policy.num_prompts,
                self.policy.budget,
            )
            if advice["advice"]:
                out.notes.append(advice["advice"])

        aborted = [p for p in out.points if p.aborted]
        if aborted and len(aborted) == len(out.points):
            out.aborted = (
                "every point aborted; the first reason was: " f"{aborted[0].aborted}"
            )
        return out

    def _headline_window(self) -> str:
        for step in self.steps():
            if not step.excluded_from_headline and not step.undrivable_reason:
                return step.window
        return DEFAULT_WINDOW


def _launch_argv(arm: Arm) -> List[str]:
    try:
        return list(arm.settings.launch_command())
    except Exception:
        return []


def load_study(path: str, **overrides) -> Study:
    """Build a :class:`Study` from a JSON description.

    A study is data for the same reason a scenario is: a new comparison is a
    new file, not a new code path, and the file is the thing that gets
    attached to a result when somebody asks six weeks later what exactly ran.

    ::

        {
          "scenario": "noise_floor",
          "policy": {"noise_floor_boots": 3, "comparison_repeats": 2,
                     "num_prompts": 64,
                     "env": {"SGLANG_ENABLE_METRICS_DEVICE_TIMER": "1"}},
          "point": {},
          "arms": [
            {"label": "even",   "settings": {...}, "conditions": {...}},
            {"label": "uneven", "settings": {...}, "env": {...}}
          ]
        }

    ``settings`` is a ``server_manager.LaunchSettings`` field mapping and is
    validated on construction, so a bad rank map fails here rather than after
    the first boot.
    """
    from sglang.srt.planner.scenarios import SCENARIOS, load_scenarios
    from sglang.srt.planner.server_manager import LaunchSettings

    with open(path) as f:
        spec = json.load(f)

    registry = dict(SCENARIOS)
    if spec.get("scenario_file"):
        load_scenarios(spec["scenario_file"], into=registry)
    key = spec.get("scenario")
    scenario = registry.get(key)
    if scenario is None:
        raise KeyError(
            f"no scenario {key!r}. Known: {', '.join(sorted(registry))}. "
            "A new question is a new scenario entry (or a scenario_file), not "
            "a new flag."
        )

    policy_spec = dict(spec.get("policy") or {})
    budget = policy_spec.pop("budget", None)
    policy = RunPolicy(**policy_spec)
    if budget:
        policy.budget = TimeBudget(**budget)

    arms = []
    for entry in spec.get("arms") or []:
        settings = LaunchSettings(**entry["settings"]).validate()
        arms.append(
            Arm(
                label=entry["label"],
                settings=settings,
                env=dict(entry.get("env") or {}),
                conditions=dict(entry.get("conditions") or {}),
                note=entry.get("note", ""),
            )
        )
    if not arms:
        raise ValueError(f"{path}: a study needs at least one arm")

    return Study(
        scenario, arms, policy=policy, point=spec.get("point") or {}, **overrides
    )


def render_dry_run_text(dry: Dict[str, Any]) -> str:
    lines = [
        f"Study: {dry['scenario']}",
        f"Question: {dry['question']}",
        f"Yardstick: {dry['primary_metric'] or '(none declared)'}",
        "",
        f"{dry['boots']} boots, load only ~{dry['estimated_load_s']:.0f}s at the "
        "top of the band (boot and teardown come on top):",
    ]
    for b in dry["schedule"]:
        lines.append(f"  {b['order']:>2}. {b['arm']} r{b['repeat']}  [{b['role']}]")
    lines += ["", "Windows:"]
    for w in dry["windows"]:
        mark = " [excluded from headline]" if w["excluded_from_headline"] else ""
        how = "harness" if w["drives_harness"] else "driver"
        if w["undrivable_reason"]:
            how = "NOT MEASURED"
        lines.append(f"  {w['window']:<20} {how}{mark}")
        if w["undrivable_reason"]:
            lines.append(f"      {w['undrivable_reason']}")
    lines += ["", "Arms:"]
    for a in dry["arms"]:
        lines.append(f"  {a['label']}: {'runnable' if a['runnable'] else 'BLOCKED'}")
        if a["reason"]:
            lines.append(f"      {a['reason']}")
        if a["command"]:
            lines.append(f"      {a['command']}")
    lines += ["", "Before running:"]
    for p in dry["preflight"]:
        lines.append(f"  - {p}")
    return "\n".join(lines)


def render_study_text(result: StudyResult) -> str:
    """A plain reading of a study: what ran, in what state, and what it means."""
    lines = [f"Study: {result.scenario}", ""]
    if result.aborted:
        lines += [f"ABORTED: {result.aborted}", ""]
    lines.append(f"Boots: {len(result.points)} of {len(result.schedule)} planned")
    for p in result.points:
        marks = []
        if p.throttled:
            marks.append("throttled")
        if p.aborted:
            marks.append("aborted")
        for window, verdict in p.budget_verdicts.items():
            if verdict:
                marks.append(f"{window}:{verdict}")
        suffix = f"  [{', '.join(marks)}]" if marks else ""
        lines.append(f"  {p.order:>2}. {p.arm} r{p.repeat} ({p.role}){suffix}")
        if p.aborted:
            lines.append(f"      {p.aborted}")
    if result.noise:
        lines += ["", "Noise floor:", f"  source: {result.noise.source}"]
        for key, value in sorted(result.noise.relative.items()):
            lines.append(f"  {key:<28} {value * 100:.2f} %")
        if not result.noise.relative:
            lines.append("  (empty — every comparison below reads 'unknown')")
    if result.comparisons:
        lines += ["", "Comparisons:"]
        for c in result.comparisons:
            lines.append(f"  [{c.verdict}] {c.metric} in {c.window}: {c.reason}")
    if result.notes:
        lines += ["", "Notes:"]
        for n in result.notes:
            if n:
                lines.append(f"  - {n}")
    return "\n".join(lines)
