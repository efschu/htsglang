"""#699: liveness from PROGRESS, because health 200 and the watchdog are both blind.

On 2026-08-16 the server sat wedged for 52+ minutes while `/health` returned
200 throughout. Two independent blind spots produced that, and both are
structural rather than bad luck.

**`/health` answers "is the process up", not "is work moving".** A scheduler
that accepts connections and answers a probe is healthy by that definition even
when nothing has advanced for an hour.

**The existing watchdog is blind to admission wedges BY CONSTRUCTION.**
``create_scheduler_watchdog`` (``managers/scheduler_components/invariant_checker.py:536-540``)
wires:

    get_counter = lambda: scheduler.forward_ct
    is_active   = lambda: (scheduler.is_initializing
                           or scheduler.cur_batch_for_debug is not None)

so it only arms while a batch EXISTS. An admission wedge is exactly the state
where no batch exists while work is pending (`#running-req: 0` on 90.6% of
prefill rounds in the specimen). ``is_active`` is therefore False for the whole
wedge and the timer never starts. The watchdog is not slow; it is switched off.

**The fix is to key on PENDING WORK rather than on a running batch**, and to
read progress from counters that move under *every* healthy regime:

* ``prefill_chunks`` — the only one that moves during a long pure-prefill run,
  where a 640-chunk prompt completes nothing for minutes;
* ``decode_steps`` — moves in pure decode, where no chunk is admitted;
* ``completions`` — moves when requests finish.

Any one of them advancing is progress. Requiring all three would call a
pure-prefill server wedged; requiring only completions would do the same, which
is precisely the trap this file exists to avoid.

**An idle box is NOT a wedge, and that refusal is load-bearing.** Zero progress
with zero pending work is a server with nothing to do. Alarming on it would
train everyone to ignore the alarm, which is how a real wedge goes unnoticed for
52 minutes. The distinction is queue depth and pending tokens, and nothing else.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Optional


class ProgressLivenessError(ValueError):
    """A liveness question that cannot be answered as posed."""


HEALTHY = "healthy"
IDLE = "idle"
WEDGED = "wedged"
INHIBITED = "inhibited"
UNKNOWN = "unknown"

ACTION_NONE = "none"
ACTION_ALARM = "alarm"
ACTION_RESTART = "restart"

#: Counters that count as forward progress. Any ONE advancing is enough.
PROGRESS_FIELDS = ("completions", "decode_steps", "prefill_chunks")


@dataclasses.dataclass(frozen=True)
class ProgressSample:
    """One observation of the scheduler's own emitted counters."""

    t_s: float
    completions: int
    decode_steps: int
    prefill_chunks: int
    pending_requests: int
    pending_tokens: int
    #: Batch ATTEMPTS (scheduler.forward_ct). Distinct from the commit counters
    #: above: a batch that re-runs without committing advances this and nothing
    #: else, which is the #701 retry-loop silhouette. Kept separate so that
    #: shape can be named rather than read as health.
    attempts: int = 0
    #: Deliberate pause: a phase flip, a maintenance hold, a GPU-arb claim.
    #: Progress legitimately stops, so an alarm here would be a false positive.
    inhibited: bool = False
    inhibit_reason: str = ""

    @property
    def has_work(self) -> bool:
        return self.pending_requests > 0 or self.pending_tokens > 0


@dataclasses.dataclass(frozen=True)
class LivenessPolicy:
    """Thresholds. Every one is a deployment input, not a constant here."""

    window_s: float = 60.0
    min_samples: int = 2
    #: Consecutive wedged assessments before the alarm is raised. Guards against
    #: a single slow window (a flip is 2-4.2 s; a long chunk is ~0.3 s).
    confirmations: int = 2
    #: Treat "attempts advancing with zero commits and work pending" as a
    #: wedge. Off restores the plain disjunctive rule.
    retry_loop_detection: bool = True
    #: Alarms before the restart policy fires. None disables restart entirely.
    restart_after_alarms: int | None = 3
    #: Minimum seconds between restarts, so a wedge that survives a restart
    #: does not become a restart loop.
    restart_cooldown_s: float = 300.0

    def __post_init__(self) -> None:
        if self.window_s <= 0:
            raise ProgressLivenessError("window_s must be positive.")
        if self.min_samples < 2:
            raise ProgressLivenessError(
                "at least two samples are needed to form a delta."
            )
        if self.confirmations < 1:
            raise ProgressLivenessError("confirmations must be at least 1.")


@dataclasses.dataclass(frozen=True)
class LivenessReport:
    verdict: str
    action: str
    deltas: dict[str, int]
    pending_requests: int
    pending_tokens: int
    span_s: float
    detail: str

    @property
    def progressing(self) -> bool:
        return any(v > 0 for v in self.deltas.values())

    def to_monitoring_dict(self) -> dict:
        """Flat shape for a metrics/liveness endpoint.

        Deliberately exposes the DELTAS and the pending counts alongside the
        verdict: an operator who can see only 'wedged' cannot tell an admission
        wedge from a stalled forward pass, and those need different responses.
        """
        out = {
            "liveness_verdict": self.verdict,
            "liveness_action": self.action,
            "liveness_span_s": round(self.span_s, 3),
            "liveness_pending_requests": self.pending_requests,
            "liveness_pending_tokens": self.pending_tokens,
            "liveness_progressing": int(self.progressing),
            "liveness_detail": self.detail,
        }
        for k, v in self.deltas.items():
            out[f"liveness_delta_{k}"] = v
        return out


def _window(samples: Sequence[ProgressSample], window_s: float):
    if not samples:
        return []
    newest = samples[-1].t_s
    return [s for s in samples if newest - s.t_s <= window_s]


def assess(
    samples: Sequence[ProgressSample],
    policy: LivenessPolicy | None = None,
    consecutive_wedges: int = 0,
    alarms_raised: int = 0,
    since_last_restart_s: float = float("inf"),
) -> LivenessReport:
    """Verdict over the sliding window, plus the action it warrants.

    ``consecutive_wedges`` is how many prior assessments in a row already said
    WEDGED; the caller carries it, because the confirmation count is about
    persistence rather than about any one window.
    """
    policy = policy or LivenessPolicy()
    win = _window(samples, policy.window_s)
    empty = dict.fromkeys(PROGRESS_FIELDS, 0)

    if len(win) < policy.min_samples:
        return LivenessReport(
            UNKNOWN,
            ACTION_NONE,
            empty,
            win[-1].pending_requests if win else 0,
            win[-1].pending_tokens if win else 0,
            0.0,
            f"only {len(win)} sample(s) in the window; not enough to form a "
            "delta. Refusing to judge rather than alarming on a cold start.",
        )

    first, last = win[0], win[-1]
    span = last.t_s - first.t_s

    if any(s.inhibited for s in win):
        reason = next((s.inhibit_reason for s in win if s.inhibit_reason), "")
        return LivenessReport(
            INHIBITED,
            ACTION_NONE,
            empty,
            last.pending_requests,
            last.pending_tokens,
            span,
            f"progress is deliberately paused ({reason or 'no reason given'}); "
            "a flip or maintenance hold stops the counters legitimately.",
        )

    deltas = {f: getattr(last, f) - getattr(first, f) for f in PROGRESS_FIELDS}

    if any(v < 0 for v in deltas.values()):
        return LivenessReport(
            UNKNOWN,
            ACTION_NONE,
            deltas,
            last.pending_requests,
            last.pending_tokens,
            span,
            "a counter moved backwards, so the scheduler restarted inside the "
            "window. A reset is not evidence of a wedge and must not be read as "
            "one; the window will refill.",
        )

    attempts_delta = last.attempts - first.attempts
    committed = any(v > 0 for v in deltas.values())

    if (
        policy.retry_loop_detection
        and not committed
        and attempts_delta > 0
        and last.has_work
    ):
        wedges = consecutive_wedges + 1
        detail = (
            f"RETRY LOOP: {attempts_delta} batch attempt(s) over {span:,.1f}s "
            "committed nothing, with "
            f"{last.pending_requests} request(s) and {last.pending_tokens} "
            "token(s) pending. Work is being re-attempted rather than "
            "advanced -- the #701 self-deadlock silhouette. An attempts-only "
            "watchdog reads this as healthy."
        )
        if wedges < policy.confirmations:
            return LivenessReport(
                WEDGED,
                ACTION_NONE,
                deltas,
                last.pending_requests,
                last.pending_tokens,
                span,
                detail + f" Confirmation {wedges}/{policy.confirmations}.",
            )
        return LivenessReport(
            WEDGED,
            ACTION_ALARM,
            deltas,
            last.pending_requests,
            last.pending_tokens,
            span,
            detail,
        )

    if committed:
        return LivenessReport(
            HEALTHY,
            ACTION_NONE,
            deltas,
            last.pending_requests,
            last.pending_tokens,
            span,
            "at least one progress counter advanced: "
            + ", ".join(f"{k}+{v}" for k, v in deltas.items() if v > 0),
        )

    if not last.has_work:
        return LivenessReport(
            IDLE,
            ACTION_NONE,
            deltas,
            last.pending_requests,
            last.pending_tokens,
            span,
            "no progress, and nothing pending. An idle box is not a wedge; "
            "alarming here is how a real alarm gets ignored.",
        )

    wedges = consecutive_wedges + 1
    if wedges < policy.confirmations:
        return LivenessReport(
            WEDGED,
            ACTION_NONE,
            deltas,
            last.pending_requests,
            last.pending_tokens,
            span,
            f"no progress with {last.pending_requests} request(s) and "
            f"{last.pending_tokens} token(s) pending, confirmation "
            f"{wedges}/{policy.confirmations}.",
        )

    action = ACTION_ALARM
    detail = (
        f"WEDGED: zero progress over {span:,.1f}s with "
        f"{last.pending_requests} request(s) and {last.pending_tokens} token(s) "
        "pending. Work is available and nothing is advancing -- this is the "
        "state /health reports as 200."
    )
    if (
        policy.restart_after_alarms is not None
        and alarms_raised + 1 >= policy.restart_after_alarms
    ):
        if since_last_restart_s >= policy.restart_cooldown_s:
            action = ACTION_RESTART
            detail += (
                f" Restart policy: {alarms_raised + 1} alarms reached the "
                f"threshold of {policy.restart_after_alarms}."
            )
        else:
            detail += (
                f" Restart withheld: only {since_last_restart_s:,.0f}s since the "
                f"last one against a {policy.restart_cooldown_s:,.0f}s cooldown. "
                "A wedge that survives a restart must not become a restart loop."
            )
    return LivenessReport(
        WEDGED,
        action,
        deltas,
        last.pending_requests,
        last.pending_tokens,
        span,
        detail,
    )


def build_liveness_is_active(scheduler) -> callable:
    """Replacement gate for ``WatchdogRaw.is_active``.

    The shipped gate arms only while a batch exists
    (``invariant_checker.py:536-540``), which switches the watchdog OFF for the
    entire duration of an admission wedge. This one arms whenever WORK IS
    PENDING, which is the condition that actually distinguishes a wedge from an
    idle box.
    """

    def _active() -> bool:
        if getattr(scheduler, "is_initializing", False):
            return True
        if getattr(scheduler, "cur_batch_for_debug", None) is not None:
            return True
        return bool(getattr(scheduler, "waiting_queue", ()) or ())

    return _active


# ---------------------------------------------------------------------------
# Binding to the deploy tree's real attributes (/spinning/wt-678-deploy).
#
# Verified by reading, not by running. What actually exists:
#
#   scheduler.forward_ct                      monotone; `+= 1` at the TOP of
#                                             run_batch (scheduler.py:6933), so
#                                             it counts batch ATTEMPTS
#   len(scheduler.waiting_queue)              pending requests (:7538, :7725)
#   len(scheduler.running_batch.reqs)         running requests (:7539, :7726)
#   load_inquirer._get_num_pending_tokens()   pending tokens
#                                             (load_inquirer.py:54-72: sums
#                                             req.seqlen over the waiting queue
#                                             plus the chunked remainder)
#
# WHAT DOES NOT EXIST, stated because it changes what the signal can promise:
# there is **no monotone completion, decode-step or committed-chunk counter**.
# `metrics_reporter.num_generated_tokens` is reset every reporting interval
# (metrics_reporter.py:849, :1080), so it cannot carry a delta across a window.
#
# CONSEQUENCE, and it corrects an overstatement of mine. `forward_ct` alone IS
# sufficient to catch the 16:23 wedge: no batch forms, so it never increments,
# and the defect was purely the `is_active` gate. Where it is NOT sufficient is
# the retry-loop shape -- a batch that re-runs without committing anything
# advances ATTEMPTS while nothing progresses, which is exactly the #701
# self-deadlock silhouette. Distinguishing those needs a committed-chunk
# counter, and that is a one-line instrumentation ask rather than a redesign.
#
# So the binding below is honest about its reach: it fills the attempt signal
# from a real monotone attribute and leaves the commit signals at zero rather
# than inventing motion they cannot observe.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SchedulerBinding:
    """Attribute paths on the deploy tree's Scheduler, kept as data.

    Held as names rather than lambdas so a binding can be asserted against a
    synthetic object in a hermetic test, and so a rename shows up as a refusal
    instead of a silently dead signal.
    """

    attempts: str = "forward_ct"
    #: Monotone committed-chunk counter (#701 ledger ride-along). When absent
    #: the commit signal stays at zero rather than borrowing the attempt count.
    committed_chunks: str = "chunked_admission_ledger.committed_chunks"
    waiting_queue: str = "waiting_queue"
    running_batch_reqs: str = "running_batch.reqs"
    #: Optional; when absent the pending-token term falls back to summing
    #: ``seqlen`` over the waiting queue, which is what load_inquirer does.
    pending_tokens_fn: str = "load_inquirer._get_num_pending_tokens"


def _resolve(obj, dotted: str):
    cur = obj
    for part in dotted.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def sample_from_scheduler(
    scheduler,
    t_s: float,
    binding: SchedulerBinding | None = None,
    inhibited: bool = False,
    inhibit_reason: str = "",
) -> ProgressSample:
    """Read one :class:`ProgressSample` from a live scheduler object.

    ``forward_ct`` lands in ``attempts``, and the #701 ledger's monotone
    ``committed_chunks`` lands in ``prefill_chunks``. Keeping them apart is what
    lets the retry-loop shape be named: attempts advancing with commits flat is
    a wedge, not health. When the ledger is absent the commit signal stays at
    ZERO rather than borrowing the attempt count -- an invented commit would
    make a retry loop look like progress, which is the failure being hunted.
    ``completions`` and ``decode_steps`` stay at zero for the same reason: no
    monotone source exists for them.
    """
    binding = binding or SchedulerBinding()
    attempts = _resolve(scheduler, binding.attempts)
    if attempts is None:
        raise ProgressLivenessError(
            f"no attribute {binding.attempts!r} on the scheduler: the binding is "
            "stale. Refusing to sample rather than reporting a frozen counter, "
            "which would read as a permanent wedge."
        )

    waiting = _resolve(scheduler, binding.waiting_queue) or ()
    pending_requests = len(waiting)

    fn = _resolve(scheduler, binding.pending_tokens_fn)
    if callable(fn):
        pending_tokens = int(fn())
    else:
        pending_tokens = int(sum(getattr(r, "seqlen", 0) for r in waiting))

    commits = _resolve(scheduler, binding.committed_chunks)
    return ProgressSample(
        t_s=t_s,
        completions=0,
        decode_steps=0,
        prefill_chunks=int(commits) if commits is not None else 0,
        attempts=int(attempts),
        pending_requests=pending_requests,
        pending_tokens=pending_tokens,
        inhibited=inhibited,
        inhibit_reason=inhibit_reason,
    )


# ---------------------------------------------------------------------------
# #861e: THE COMPLETION-PROGRESS CLOCK -- thrash is neither wedge nor progress.
# ---------------------------------------------------------------------------

#: A run of samples in which decode advances and completions do not. Two is too
#: few (a long generation legitimately spans samples); this is the shortest run
#: that cannot be one slow request.
THRASH_MIN_SAMPLES = 4


def thrash_verdict(
    samples: "Sequence[ProgressSample]",
    *,
    min_samples: int = THRASH_MIN_SAMPLES,
) -> Optional[str]:
    """DECODE ADVANCES, COMPLETIONS DO NOT. The detector W37-D lacked.

    THE GAP, measured. W37-D/d4 ran 102 flips, 69 decode batches, GPU at
    98/47/57 %, and produced ZERO completions in seven minutes. Every existing
    detector stayed silent and each was right to:

      * ADMISSION-WEDGE measures FIRST-TOKEN age -- and first tokens kept
        arriving, one per flip cycle, so there was no wedge;
      * the #699 liveness assess() measures PROGRESS -- and decode_steps kept
        advancing, so there was progress;
      * health returns 200 throughout, as it always does.

    Nothing measured the quantity a user actually has: **completions per unit
    time**. Per-rid output grew (rid be636087: n=2,3,4,...,13 across epochs) at
    exactly ONE TOKEN PER FLIP CYCLE, so a 64-token request needed 64 cycles.
    That is a livelock WITH progress, and it is invisible to every rung above.

    THE SHAPE, stated so it cannot be confused with its neighbours:
        decode_steps STRICTLY INCREASING  (the box is working -- not a wedge)
        completions   FLAT                 (nobody is being served)
        has_work      TRUE                 (there is something to serve)
        not inhibited                      (no flip/maintenance pause claimed)

    BOTH PINS, because a one-sided detector is how the last three terms
    shipped blind: it must FIRE on that shape and be SILENT when completions
    advance, when nothing is decoding, when there is no work, and while
    progress is legitimately inhibited.

    Returns the alarm detail, or None.
    """
    window = [s for s in samples][-min_samples:]
    if len(window) < min_samples:
        return None
    if any(s.inhibited for s in window):
        return None
    if not all(s.has_work for s in window):
        return None
    decode_delta = window[-1].decode_steps - window[0].decode_steps
    completion_delta = window[-1].completions - window[0].completions
    if decode_delta <= 0 or completion_delta > 0:
        return None
    span = max(1e-9, window[-1].t_s - window[0].t_s)
    return (
        f"COMPLETION-PROGRESS STALL (#861e): {decode_delta} decode step(s) in "
        f"{span:.1f}s and ZERO completions, with work pending "
        f"({window[-1].pending_requests} req / {window[-1].pending_tokens} tok). "
        f"The box IS working -- this is not a wedge and every first-token and "
        f"liveness rung is correctly silent -- but nobody is being served. "
        f"W37-D measured this exact shape: 102 flips, 69 decode batches, GPU "
        f"98%, one output token per flip cycle, 0 completions in 7 minutes. "
        f"Suspect a policy term reading a state the cutover manufactures "
        f"(#861e) before suspecting the model."
    )
