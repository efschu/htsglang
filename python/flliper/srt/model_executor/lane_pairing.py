"""Pairing objective of the two-class scheduler (#274 slice D).

WHAT THIS DECIDES.  On a card that carries two classes -- the serving group
and a dual-group lane -- concurrency pays only when the two classes do not
both saturate the SMs.  The empirical basis is pinned:

* C3 (INTEGRATION_R3_VALIDATION, slice C): a decode-shaped lane next to the
  serving group yields E = 1.440 card equivalents, a 2048-token prefill lane
  yields E = 1.130.  The +9.7 % the protected prefill lane pays was traced
  CAUSALLY to SM compute competition (prefill_wait_ms 0.01 -- not preemption
  granularity, not the submission path).
* D1: choosing the decode-shaped pairing over the prefill-shaped one is
  +24.3 % aggregate on an independent lane configuration.
* #284: under load the lane's loss splits about evenly into SM competition
  (cost ratio 2.04 on the same captured graph) and a GIL-bound submission gap
  (occupancy 0.378 at duty 1.0); finer grains would worsen the GIL half, so
  chunking is explicitly NOT the lever.

THE OBJECTIVE.  Maximize E = sum of card-equivalent shares by ORDER, not by
idling: prefer pairing a saturating grain of one class with a non-saturating
grain of the other; avoid saturating+saturating when the queue offers an
alternative.  The policy is a work-conserving reorder of the lane's own job
queue -- it never idles the lane and never touches the serving group's batch
composition, because C3 also pinned that saturating+saturating CONCURRENCY
still beats taking turns (E 1.130 vs 0.974): avoiding a bad pairing means
picking a different job, never serializing.

CLASSIFICATION.  A grain is one forward.  The cheap, deterministic measure of
SM saturation is the GEMM row count R of that forward -- the number of times
each weight byte is reused.  A forward is compute-bound (SM-saturating) once
R exceeds the card's flops-per-byte ratio divided by the model's
flops-per-weight-byte, i.e. roughly

    R_sat = (gemm_flops / mem_bandwidth) / (2 / bytes_per_weight)

which is ~117 rows for bf16 weights on a 5090-class card and ~26 rows for
Q3_K GGUF weights.  The default threshold (64) sits between the two, and the
anchors this policy exists for are each an order of magnitude away from it:
a 2048-row prefill chunk is saturating under any calibration, a <=16-row
speculative decode batch is not.  The threshold is a flag, not a constant,
because the ratio is a property of card and weight format.

Occupancy from the LaneShareMeter is deliberately NOT an automatic input,
for two measured reasons: (a) stream occupancy is duty, not SM width -- #284
measured occ 0.975 for a batch-size-1 decode lane, which pairs WELL (E 1.44);
(b) the meter's windows are the judge of this policy's A/B, and a policy
conditioned on its judge self-conditions the measurement (the same principle
that keeps LaneShareGate report-only).  The meter's occupancy/cost
decomposition is instead the CALIBRATION evidence: measurement windows report
the labels this module assigned next to the occupancy and cost ratios the
meter measured, so the threshold can be recalibrated from data.

RANK RELEVANCE.  Lanes exist on the shared rank only (build_dual_group_lanes
returns [] everywhere else) and have no communicator by contract; the policy
reorders lane jobs on that one rank and never reads or writes anything a
collective depends on.  The serving group's batch composition -- the thing
that IS collectively relevant -- is published to the policy read-only.
Pairing decisions therefore need no cross-rank consensus (#287 pattern not
required).

GIL BUDGET.  Decisions happen at job-pick boundaries only (once per lane job,
i.e. once per ~50-130 forwards), labels are computed once at enqueue, and the
serving-side signal is one tuple store per run_batch.  Nothing is added to
the Python between two lane forwards, which #284 names as the submission-gap
half of the loss.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

# Phases a grain can be in. The phase is carried for readability of the
# evidence rows; the CLASSIFIER only looks at rows.
PHASE_PREFILL = "prefill"
PHASE_DECODE = "decode"
PHASE_IDLE = "idle"

# Default GEMM-row threshold above which a forward counts as SM-saturating.
# See the module docstring for the derivation; calibratable via
# --dual-group-lane-pairing-sat-rows.
DEFAULT_SAT_ROWS = 64

# Default staleness of the serving grain signal. A serving iteration is
# 17-35 ms on the measured vehicle (#284); a signal older than a few
# iterations means the serving group is idle or draining, and an idle class
# saturates nothing.
DEFAULT_STALE_MS = 100.0

# How long one GEMM row of a saturating prefill grain keeps that grain
# current, in ms. CARD-FOUND (slice D boot 1): the signal is published when
# the batch LAUNCHES, and a 1600-row prefill forward then runs for ~1.1 s --
# under a flat 100 ms staleness the policy read IDLE during exactly the
# grains it exists to flag (1 of 11 picks saw a saturating serving grain on
# a serving load that was in prefill most of the wall clock). A saturating
# prefill grain therefore stays current for rows * ms_per_row, a duration
# estimate from its own size; the next publish replaces it anyway, so the
# overshoot only exists when the serving group drains right after a prefill.
# 1.0 ms/row is conservative against the measured ~0.7 ms/row (1400 tok/s
# serving prefill on the C3 vehicle); calibratable via
# --dual-group-lane-pairing-prefill-ms-per-row.
DEFAULT_PREFILL_MS_PER_ROW = 1.0

# Default starvation cap: a queue head skipped in favour of better-pairing
# jobs for longer than this runs regardless of the pairing. The value bounds
# added head latency at a handful of lane jobs, far below any lane job's own
# runtime budget.
DEFAULT_MAX_DEFER_MS = 500.0

# How many prefill ROWS one decode step is worth in lane device time, for the
# job-dominance rule below. CARD-FOUND (slice D boot 2): a queued job's next
# grain is always its prefill, so classifying the JOB by that one forward
# called a 71-token-prompt, 128-new-token job "saturating" (71 rows >= 64) --
# every queued job then classified saturating and the policy answered "queue
# all saturating: FIFO" on all six picks that saw a saturating serving grain.
# The 71-row prefill lasts ~35 ms; the 128 decode steps after it, ~2-5 s. A
# job pick allocates the JOB's whole runtime next to the serving group, so
# the label has to follow the job's DOMINANT phase: saturating only when the
# prefill's device time outweighs the decode tail, i.e.
#   prefill_rows >= max_new_tokens * decode_step_rows.
# The default derives from the measured lane costs on the C3 vehicle: a lane
# decode step is ~17 ms and a lane prefill row ~1.5 ms (mixed-floor cost
# 1.4976 ms/token at 93 % prefill tokens, boot 1), giving ~11 rows per step;
# 12 keeps it conservative. Calibratable via
# --dual-group-lane-pairing-decode-step-rows.
DEFAULT_DECODE_STEP_ROWS = 12


@dataclass(frozen=True)
class GrainLabel:
    """The saturation classification of one forward-sized unit of work."""

    phase: str
    rows: int
    saturating: bool
    reason: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "rows": self.rows,
            "saturating": self.saturating,
            "reason": self.reason,
        }


IDLE_LABEL = GrainLabel(PHASE_IDLE, 0, False, "no fresh grain: class is idle")


def classify_rows(
    phase: str, rows: int, sat_rows: int = DEFAULT_SAT_ROWS
) -> GrainLabel:
    """Label one grain from its GEMM row count.

    Deterministic and total: any non-negative row count yields a label, and
    the same inputs always yield the same label (the hermetic tests pin
    this).  ``sat_rows`` is the calibratable threshold; the phase is carried
    as evidence, not consulted -- a 2048-row "decode" (which does not exist
    today) would saturate exactly like a 2048-row prefill.
    """
    rows = int(rows)
    if rows <= 0:
        return GrainLabel(phase, rows, False, "empty grain")
    if rows >= int(sat_rows):
        return GrainLabel(phase, rows, True, f"rows {rows} >= sat_rows {int(sat_rows)}")
    return GrainLabel(phase, rows, False, f"rows {rows} < sat_rows {int(sat_rows)}")


def lane_job_grain_rows(job: Mapping[str, Any], spec_steps: int) -> Tuple[str, int]:
    """(phase, rows) of the NEXT forward a lane job would run.

    A queued job's next grain is always its whole-prompt prefill
    (DualGroupLane._prefill builds ONE ScheduleBatch over the full prompt);
    an active job's next grain is one decode step (1 row) or one speculative
    round (spec_steps + 1 rows on the verify).  Decode grains are therefore
    never saturating at any sane calibration -- the policy's whole decision
    space is which PREFILL to start next to what the serving group is doing.
    """
    if job.get("prefill_ms") is None:
        return PHASE_PREFILL, len(job.get("input_ids") or ())
    if job.get("spec"):
        return PHASE_DECODE, int(spec_steps) + 1
    return PHASE_DECODE, 1


class ServingGrainSignal:
    """The serving group's current grain, published for the lane's policy.

    One writer (the scheduler thread, one tuple store per run_batch), any
    number of readers (lane worker threads).  The tuple is replaced
    atomically under the GIL; readers never see a torn pair.  Staleness IS
    the idle signal: run_batch simply stops being called when the serving
    group drains, so a fresh "idle" publish would need a hook on the idle
    path -- aging out needs none.

    EXCEPT for the grains the policy exists to flag (card-found, slice D
    boot 1): the label is published when the batch LAUNCHES, and a big
    prefill forward then runs for around a second -- far past any staleness
    that still detects idleness for 17-35 ms decode iterations.  A
    saturating prefill grain therefore stays current for its own estimated
    duration, rows * ms_per_row; every later publish replaces it outright,
    so the estimate only governs the gap where nothing is published at all.
    """

    __slots__ = ("_current", "stale_s", "ms_per_row")

    def __init__(
        self,
        stale_ms: float = DEFAULT_STALE_MS,
        ms_per_row: float = DEFAULT_PREFILL_MS_PER_ROW,
    ):
        self.stale_s = max(0.001, float(stale_ms) / 1000.0)
        self.ms_per_row = max(0.0, float(ms_per_row))
        self._current: Tuple[float, GrainLabel] = (0.0, IDLE_LABEL)

    def publish(self, label: GrainLabel, now: Optional[float] = None) -> None:
        self._current = (time.monotonic() if now is None else now, label)

    def _current_for_s(self, label: GrainLabel) -> float:
        if label.saturating and label.phase == PHASE_PREFILL:
            return max(self.stale_s, label.rows * self.ms_per_row / 1000.0)
        return self.stale_s

    def read(self, now: Optional[float] = None) -> GrainLabel:
        t, label = self._current
        age = (time.monotonic() if now is None else now) - t
        if age > self._current_for_s(label):
            return IDLE_LABEL
        return label

    def snapshot(self) -> Dict[str, Any]:
        t, label = self._current
        return {"age_s": round(time.monotonic() - t, 4), "label": label.to_json()}


@dataclass
class PairingDecision:
    """One pick, with the evidence it was made on (pinned by tests)."""

    index: int
    reason: str
    serving: GrainLabel
    picked: GrainLabel


@dataclass
class PairingPolicy:
    """Work-conserving pairing reorder of one lane's job queue.

    ``enabled`` is the runtime A/B switch (set_internal_state command
    ``dual_group_lane_pairing``): disabled, ``pick`` returns index 0
    unconditionally, which is byte-identical to the FIFO ``pop(0)`` the lane
    ran before this module existed -- the regression tests pin that.
    """

    sat_rows: int = DEFAULT_SAT_ROWS
    max_defer_ms: float = DEFAULT_MAX_DEFER_MS
    decode_step_rows: int = DEFAULT_DECODE_STEP_ROWS
    spec_steps: int = 3
    enabled: bool = True
    signal: Optional[ServingGrainSignal] = None
    # Counters, monotone, for stats()/measurement windows.
    picks_total: int = 0
    reordered_total: int = 0
    starvation_overrides_total: int = 0
    serving_saturating_picks: int = 0
    last_decision: Optional[PairingDecision] = None
    _deferred_since: Dict[int, float] = field(default_factory=dict)

    def label_job(self, job: Mapping[str, Any]) -> GrainLabel:
        """Label a job for the PICK: its dominant phase, not its first forward.

        A queued job's next grain is always its prefill, but picking a job
        allocates the job's WHOLE runtime next to the serving group.  A job
        whose prefill clears the row threshold while its decode tail dwarfs
        that prefill in device time (card-found in boot 2: 71 rows / ~35 ms
        of prefill in front of 128 steps / seconds of decode) is
        decode-dominated: deferring it defers almost no saturating work and
        starves the policy of exactly the non-saturating alternative it is
        looking for.  Saturating therefore additionally requires
        prefill_rows >= max_new_tokens * decode_step_rows.
        """
        phase, rows = lane_job_grain_rows(job, self.spec_steps)
        label = classify_rows(phase, rows, self.sat_rows)
        if not label.saturating or phase != PHASE_PREFILL:
            return label
        decode_tail = int(job.get("max_new_tokens") or 0)
        tail_rows = decode_tail * int(self.decode_step_rows)
        if rows < tail_rows:
            return GrainLabel(
                phase,
                rows,
                False,
                f"decode-dominated job: prefill {rows} rows < "
                f"{decode_tail} new tokens x {int(self.decode_step_rows)} "
                "rows/step",
            )
        return label

    def pick(
        self, jobs: Sequence[Mapping[str, Any]], now: Optional[float] = None
    ) -> int:
        """Index of the job to run next.  O(queue length), integer compares.

        Rules, in order (each pinned by a hermetic test):
        1. Disabled or trivial queue: FIFO (index 0).
        2. Serving grain not saturating (incl. idle/stale): FIFO -- any lane
           job pairs, and a saturating lane prefill FILLS the gap best where
           it would otherwise wait behind short jobs for no reason.
        3. Serving saturating: first NON-saturating job in queue order.
           A head skipped longer than max_defer_ms runs regardless
           (starvation cap).
        4. Serving saturating and every queued job saturating: FIFO.
           Concurrency still beats serialization for that pairing (C3:
           E 1.130 vs 0.974), so the policy never idles the lane.
        """
        if not self.enabled or len(jobs) <= 1:
            return 0
        now = time.monotonic() if now is None else now
        self.picks_total += 1
        serving = IDLE_LABEL if self.signal is None else self.signal.read(now)
        if not serving.saturating:
            self._note(
                PairingDecision(
                    0, "serving not saturating: FIFO", serving, self.label_job(jobs[0])
                ),
                jobs,
            )
            return 0
        self.serving_saturating_picks += 1
        head_label = self.label_job(jobs[0])
        if not head_label.saturating:
            self._note(
                PairingDecision(0, "head already non-saturating", serving, head_label),
                jobs,
            )
            return 0
        # Head is saturating against a saturating serving grain. Starvation
        # cap first: a head we have been skipping runs now.
        head_key = id(jobs[0])
        since = self._deferred_since.get(head_key)
        if since is not None and (now - since) * 1000.0 >= self.max_defer_ms:
            self.starvation_overrides_total += 1
            self._note(
                PairingDecision(
                    0,
                    f"starvation cap: head deferred {((now - since) * 1000.0):.0f} ms"
                    f" >= {self.max_defer_ms:.0f} ms",
                    serving,
                    head_label,
                ),
                jobs,
            )
            return 0
        for i in range(1, len(jobs)):
            label = self.label_job(jobs[i])
            if not label.saturating:
                self.reordered_total += 1
                self._deferred_since.setdefault(head_key, now)
                self._note(
                    PairingDecision(
                        i,
                        "saturating+saturating avoided: picked first "
                        "non-saturating job",
                        serving,
                        label,
                    ),
                    jobs,
                )
                return i
        self._note(
            PairingDecision(
                0, "queue all saturating: FIFO (work-conserving)", serving, head_label
            ),
            jobs,
        )
        return 0

    def _note(
        self, decision: PairingDecision, jobs: Sequence[Mapping[str, Any]]
    ) -> None:
        self.last_decision = decision
        if decision.index == 0 and jobs:
            # The head ran (or will run): its deferral record is spent.
            self._deferred_since.pop(id(jobs[0]), None)
        if len(self._deferred_since) > 64:
            # The map is keyed by id() of queue entries; entries picked via
            # index 0 are cleaned above, and a bounded sweep keeps abandoned
            # ids (jobs dropped on error paths) from accumulating.
            live = {id(j) for j in jobs}
            for k in list(self._deferred_since):
                if k not in live:
                    del self._deferred_since[k]

    def snapshot(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "enabled": self.enabled,
            "sat_rows": self.sat_rows,
            "decode_step_rows": self.decode_step_rows,
            "max_defer_ms": self.max_defer_ms,
            "picks_total": self.picks_total,
            "reordered_total": self.reordered_total,
            "starvation_overrides_total": self.starvation_overrides_total,
            "serving_saturating_picks": self.serving_saturating_picks,
        }
        if self.signal is not None:
            out["serving_signal"] = self.signal.snapshot()
        d = self.last_decision
        if d is not None:
            out["last_decision"] = {
                "index": d.index,
                "reason": d.reason,
                "serving": d.serving.to_json(),
                "picked": d.picked.to_json(),
            }
        return out


def serving_batch_grain(
    forward_mode_is_extend: bool,
    extend_num_tokens: Optional[int],
    batch_size: int,
    rows_per_seq: int,
    sat_rows: int,
) -> GrainLabel:
    """Label the serving group's next forward from batch shape alone.

    ``rows_per_seq`` folds speculation in: a target-verify forward runs
    (num_draft_tokens) rows per sequence, a plain decode runs 1.  Computed
    by the caller once at init from the server args, so this stays integer
    arithmetic on the launch path.
    """
    if forward_mode_is_extend:
        rows = int(extend_num_tokens or 0)
        return classify_rows(PHASE_PREFILL, rows, sat_rows)
    return classify_rows(PHASE_DECODE, int(batch_size) * int(rows_per_seq), sat_rows)
