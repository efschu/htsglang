# SPDX-License-Identifier: Apache-2.0
"""Observe-only driver of the #363 regime classifier (phase 2).

DESIGN_363_regime_controller.md section 6 ships v1 as observe-only, and this
is it: one object per scheduler that classifies the regime at the between-tick
boundary, names the stage it WOULD select, and calls no actuator. Its output
is the input trace the remaining falsifiers need (F2 self-conditioning replay,
F4 do-nothing baseline), so the observe phase is the instrument that earns the
actuating phase rather than a courtesy step in front of it.

WHAT IT MAY NOT DO, IN CODE
---------------------------
1. NO ACTUATOR. There is no import of ``kv_reshard`` or ``vram_dial`` in this
   module and no call that changes server state. A unit test greps for that.
2. NO RANK-LOCAL BRANCH. Every branch reads tier-R state only -- the
   forward-mode mix, the KV occupancy, the queue composition, the replicated
   round counter. The one rank-local input (this rank's forward device ms)
   is accumulated, quantized and packed into the consensus payload; it is
   read back only as a group summary, one boundary later. Reading it locally
   before a collective is the #94/#194/#259/#312 hang class, and observe-only
   is exactly where that has to be proven, not assumed.
3. NO RAISE. #287 raises on a desync because continuing would run a collective
   under a geometry the ranks disagree about. Nothing here acts, so nothing
   here can hang -- and an instrument that takes the server down while
   proving it is safe has failed at its own job. A desync is counted and
   logged loudly instead, and ``desyncs == 0`` over a real workload is the
   gate that phase 3 has to pass before any actuator is wired.

THE CADENCE AND THE COLLECTIVE
------------------------------
Sampling is every round; the verdict and the one MIN-reduction happen every
``consensus_interval``-th round, gated by the REPLICATED round counter and
never by local state, so every rank enters the collective in the same round or
none of them does. This is #287 rule 2 verbatim; the payload layout and the
``(v, -v)`` MIN trick come from ``regime_classifier.pack_proposal``.

THE ONE-BOUNDARY LAG, STATED
----------------------------
The rank spread only exists after the reduction, and the classification that
produced the proposal happened before it. The spread is therefore reported
with this boundary's record and carried into the NEXT sample. In observe-only
nothing consumes it, so the lag costs nothing; when it becomes a veto input in
phase 3 it is one boundary stale by construction, and that is a property of
the reduction, not a shortcut taken here.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
from typing import Callable, Dict, List, Optional

from sglang.srt.managers.regime_classifier import (
    DEFAULT_WINDOW_ROUNDS,
    REGIMES,
    DwellGate,
    RegimeSample,
    RegimeSensor,
    StageTable,
    pack_proposal,
    unpack_reduced,
)

logger = logging.getLogger(__name__)

LOG_PREFIX = "REGIME-OBSERVE"

#: Mode values. ``off`` builds nothing; ``observe`` classifies and logs;
#: ``act`` additionally issues the reachable half of a stage flip through an
#: INJECTED commit callable (``regime_act.RegimeActuator.apply``). This module
#: never imports that one -- see the no-actuator property below.
MODE_OFF = "off"
MODE_OBSERVE = "observe"
MODE_ACT = "act"
MODES = (MODE_OFF, MODE_OBSERVE, MODE_ACT)

#: ``--regime-controller`` is the flag (phase 3). This env var is the phase-2
#: spelling, kept as an OVERRIDE so an observe run already in somebody's
#: launch script keeps working.
#:
#: RETIREMENT: it will be removed once the flag has been through one release.
#: It can only ever select ``observe`` or ``off`` -- authorizing ``act`` needs
#: the entry gate, and an env var is not a place to put an authorization.
ENV_MODE = "SGLANG_REGIME_OBSERVE"

#: Emit one record every N verdicts. The verdict itself is cheap (integer
#: arithmetic over a window plus one small reduction); the log line is what
#: costs, so it is the thing that is throttled.
DEFAULT_LOG_EVERY = 1


def observe_mode() -> str:
    """The mode the ENV OVERRIDE asks for: ``off`` or ``observe``.

    An unknown value is refused loudly rather than rounded to the nearest
    mode: a typo that silently disables an observability run wastes the run.
    ``act`` is refused here by name -- the entry gate of DESIGN_363 section
    11.7 is checked at argument time against declared evidence, and an
    environment variable cannot carry evidence.
    """
    raw = (os.environ.get(ENV_MODE) or "").strip().lower()
    if not raw or raw in ("0", "false", "no", MODE_OFF):
        return MODE_OFF
    if raw in ("1", "true", "yes", MODE_OBSERVE):
        return MODE_OBSERVE
    if raw == MODE_ACT:
        raise ValueError(
            f"{ENV_MODE}=act is refused: acting is authorized by the entry "
            f"gate (DESIGN_363 section 11.7), which is checked at argument "
            f"time against --regime-gate-evidence. Use "
            f"--regime-controller act so the refusal can name what is "
            f"missing; an env var cannot carry evidence."
        )
    raise ValueError(
        f"{ENV_MODE}={raw!r} is not a known mode; use {MODE_OFF} or "
        f"{MODE_OBSERVE} (or 1/0)."
    )


def resolve_mode(server_args=None) -> str:
    """The effective mode: the flag, with the env override on top.

    Precedence is the override direction, deliberately: the env var can turn
    observation ON for a server whose launch script predates the flag, and it
    can never turn acting on.
    """
    mode = str(getattr(server_args, "regime_controller", MODE_OFF) or MODE_OFF)
    if mode not in MODES:
        raise ValueError(
            f"--regime-controller={mode!r} is not a known mode; use one of "
            f"{', '.join(MODES)}."
        )
    env = observe_mode()
    if env == MODE_OBSERVE and mode == MODE_OFF:
        logger.info(
            "%s: %s=observe overrides --regime-controller off. The env var is "
            "the phase-2 spelling and will be retired; prefer the flag.",
            LOG_PREFIX,
            ENV_MODE,
        )
        return MODE_OBSERVE
    return mode


class RegimeObserver:
    """Classifies the regime at the between-tick boundary. Actuates nothing.

    ``collective_min`` is the injectable consensus channel, same contract as
    the #287 runtime: a packed int payload in, the element-wise MIN across the
    TP group out. Hermetic tests inject mocks -- including one that merges
    DIFFERENT ranks' payloads, which is how the desync path is exercised
    without a second process.
    """

    def __init__(
        self,
        *,
        sensor: Optional[RegimeSensor] = None,
        table: Optional[StageTable] = None,
        consensus_interval: int = 8,
        window_rounds: int = DEFAULT_WINDOW_ROUNDS,
        tp_size: int = 1,
        collective_min: Optional[Callable[[List[int]], List[int]]] = None,
        log_every: int = DEFAULT_LOG_EVERY,
        current_stage: Optional[str] = None,
        trace_path: Optional[str] = None,
        trace_rank: Optional[int] = None,
        mode: str = MODE_OBSERVE,
        table_plan=None,
        commit_fn: Optional[Callable] = None,
        dwell: Optional["DwellGate"] = None,
    ):
        if consensus_interval < 1:
            raise ValueError(
                f"consensus_interval must be >= 1, got {consensus_interval}"
            )
        if window_rounds < 1:
            raise ValueError(f"window_rounds must be >= 1, got {window_rounds}")
        if log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {log_every}")
        # Deliberately NOT the #287 refusal: a multi-rank group without a
        # channel is a degraded observation, not a hang risk, because nothing
        # here acts on the verdict. It is recorded as such rather than
        # refused, so an operator can still get a regime trace off a rig whose
        # CPU group is not wired for this.
        self._sensor = sensor if sensor is not None else RegimeSensor()
        self._table = table
        self._interval = int(consensus_interval)
        self._window = int(window_rounds)
        self._tp_size = int(tp_size)
        self._collective_min = collective_min
        self._log_every = int(log_every)
        self._current_stage = current_stage
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}, got {mode!r}")
        # The act path is a pair of INJECTED callables, never an import: this
        # module must stay unable to reach #297/#330 so the phase-2 F4
        # property survives phase 3 as a fact about the import graph.
        if mode == MODE_ACT and commit_fn is None:
            raise ValueError(
                "mode 'act' needs a commit_fn (regime_act.RegimeActuator.apply). "
                "Constructing an acting observer with nothing to act through "
                "would run as an expensive observe under a misleading name."
            )
        if mode != MODE_ACT and commit_fn is not None:
            raise ValueError(
                f"mode {mode!r} was given a commit_fn. Observe must not hold a "
                f"path to an actuator even an unused one -- that is the "
                f"property the no-actuator test pins."
            )
        self._mode = mode
        # One JSON object per verdict, appended. This is the artifact both
        # card gates read: gate 1 counts desyncs off it, gate 2 replays it
        # open-loop. Written from the scheduler process that owns the
        # observer, so on a multi-rank boot each rank writes its own file and
        # the reader can see that they agreed (or did not).
        self._trace_path = str(trace_path) if trace_path else None
        self._trace_fh = None
        self._trace_rank = trace_rank
        self._table_plan = table_plan
        self._commit_fn = commit_fn
        self._dwell = dwell
        self.actuations = 0
        self.vetoes: Dict[str, int] = {}

        self._round = 0
        self._epoch = 0
        self._verdicts = 0
        # Tier-R window accumulators.
        self._prefill_rounds = 0
        self._decode_rounds = 0
        self._idle_rounds = 0
        # Tier-L accumulator. Rank-local, never branched on; it leaves this
        # object only inside the packed payload.
        self._rank_ms_sum = 0.0
        self._rank_ms_n = 0
        # Carried forward from the previous boundary's reduction (see the
        # one-boundary lag in the module docstring).
        self._last_spread_pct: Optional[float] = None
        self._last_record: Optional[Dict] = None

        self.desyncs = 0
        self.proposals = 0
        self.consensus_rounds = 0
        self.uncoordinated = bool(self._tp_size > 1 and collective_min is None)

        logger.info(
            "%s armed (OBSERVE-ONLY: classifies and logs, actuates nothing): "
            "window %d rounds, verdict every %d rounds over %d rank(s), "
            "stage table %s%s",
            LOG_PREFIX,
            self._window,
            self._interval,
            self._tp_size,
            (
                f"{len(self._table)} stage(s)"
                if self._table is not None
                else "NOT DECLARED (regime is logged, no stage is named)"
            ),
            (
                "; WARNING no consensus channel on a multi-rank group -- the "
                "rank-uniformity of the verdict is NOT being checked"
                if self.uncoordinated
                else ""
            ),
        )

    # -- state ---------------------------------------------------------------
    @property
    def round_index(self) -> int:
        return self._round

    @property
    def regime(self) -> str:
        return self._sensor.regime

    @property
    def trace_path(self) -> Optional[str]:
        """Where the verdict trace is being written, or ``None``."""
        return self._trace_path

    @property
    def last_record(self) -> Optional[Dict]:
        return self._last_record

    # -- the per-round hook --------------------------------------------------
    #: Execution phase of one scheduler round. IDLE is its own value and not
    #: a kind of prefill: an idle window is no measurement, not 0 % prefill
    #: (DESIGN_363 section 3.2). The 2026-08-01 gate window found this the
    #: hard way -- the hook passed a BOOLEAN, idle fell on the prefill side,
    #: and the rig read PREFILL_HEAVY on 93 600 of 93 603 verdicts.
    PHASE_PREFILL = "prefill"
    PHASE_DECODE = "decode"
    PHASE_IDLE = "idle"
    PHASES = (PHASE_PREFILL, PHASE_DECODE, PHASE_IDLE)

    def on_round(
        self,
        *,
        phase: str,
        held_tokens: int,
        capacity_tokens: int,
        running_bs: int,
        queued_reqs: int = 0,
        queued_prompt_tokens: int = 0,
        max_queued_prompt_tokens: int = 0,
        rank_forward_ms: Optional[float] = None,
    ) -> Optional[Dict]:
        """One between-tick boundary.

        ``phase`` is one of ``prefill`` / ``decode`` / ``idle``. Every argument
        except ``rank_forward_ms`` must be REPLICATED across
        the TP group -- the caller's obligation, stated here because it is the
        whole uniformity argument. ``rank_forward_ms`` is the one rank-local
        input and is treated as such: accumulated, never branched on, and
        released only through the reduction.

        Returns the verdict record at a consensus boundary, ``None`` between
        boundaries.
        """
        if phase not in self.PHASES:
            raise ValueError(
                f"unknown execution phase {phase!r}; known: "
                f"{', '.join(self.PHASES)}. An idle round is IDLE, not a "
                f"kind of prefill."
            )
        self._round += 1
        if phase == self.PHASE_PREFILL:
            self._prefill_rounds += 1
        elif phase == self.PHASE_DECODE:
            self._decode_rounds += 1
        else:
            # Counted in the round index (the replicated cadence gate) and in
            # NEITHER share. A window of nothing but idle rounds therefore has
            # no forward rounds at all, its shares are absent, and the
            # classifier answers MIXED -- which is the designed behaviour and
            # what the boolean hook silently prevented.
            self._idle_rounds += 1
        if rank_forward_ms is not None:
            self._rank_ms_sum += float(rank_forward_ms)
            self._rank_ms_n += 1

        # Rule 2: the cadence gate is the REPLICATED round counter. Every rank
        # passes or skips this line in the same round, so the collective below
        # is entered by all of them or by none.
        if self._round % self._interval != 0:
            return None

        sample = RegimeSample(
            round_index=self._round,
            prefill_rounds=self._prefill_rounds,
            decode_rounds=self._decode_rounds,
            held_tokens=max(0, int(held_tokens)),
            capacity_tokens=max(0, int(capacity_tokens)),
            queued_reqs=max(0, int(queued_reqs)),
            queued_prompt_tokens=max(0, int(queued_prompt_tokens)),
            max_queued_prompt_tokens=max(0, int(max_queued_prompt_tokens)),
            running_bs=max(0, int(running_bs)),
            rank_ms_spread_pct=self._last_spread_pct,
        )
        # Tier-R only, pure, identical on every rank by construction.
        regime = self._sensor.observe(sample)

        target, why = (None, "no stage table declared; regime only")
        if self._table is not None:
            target, why = self._table.select(
                regime, sample, current=self._current_stage
            )
        stage_index = self._stage_index(target)

        mean_ms = (self._rank_ms_sum / self._rank_ms_n) if self._rank_ms_n else None
        reduced = self._consensus(regime, stage_index, mean_ms)

        self._verdicts += 1
        if target is not None:
            self.proposals += 1
        record = {
            "round": self._round,
            "epoch": self._epoch,
            "regime": regime,
            "prefill_share": sample.prefill_share,
            "decode_share": sample.decode_share,
            "occupancy": sample.occupancy,
            # The NUMERATOR and the denominator, not only their ratio. A
            # replay (#363 gate 2) has to vary the denominator -- that is the
            # whole self-conditioning mechanism -- so a trace carrying only
            # the ratio silently holds occupancy constant under the very
            # change it is meant to expose, and the counterfactual comes back
            # clean for the wrong reason. Found by the gate-2 tool's own
            # smoke.
            "held_tokens": sample.held_tokens,
            "capacity_tokens": sample.capacity_tokens,
            "queued_reqs": sample.queued_reqs,
            "queued_prompt_tokens": sample.queued_prompt_tokens,
            "idle_rounds": self._idle_rounds,
            "would_flip_to": target.name if target is not None else None,
            "reason": why,
            "rank_mean_forward_ms": mean_ms,
            # What the SAMPLE carried in (the previous boundary's reduction)
            # and what THIS boundary's reduction produced. Two fields, because
            # the one-boundary lag is a property of the reduction and a reader
            # comparing them has to be able to see it.
            "sample_spread_pct": sample.rank_ms_spread_pct,
            "rank_ms_spread_pct": (
                reduced.get("rank_ms_spread_pct") if reduced else None
            ),
            "agreed": bool(reduced["agreed"]) if reduced else None,
            "actuated": False,
            "action": None,
        }
        # ACT: the only place a proposal becomes a move. Off in every other
        # mode by construction -- ``_commit_fn`` is None and the constructor
        # refuses to accept one outside act mode.
        if self._mode == MODE_ACT and target is not None:
            allowed, veto = self._act_interlocks(target, sample, reduced)
            if not allowed:
                self._count_veto(veto)
                record["reason"] = f"{why}; ACT VETO: {veto}"
            else:
                result = self._commit_fn(target, self._table[self._current_stage])
                record["action"] = result.as_dict()
                record["actuated"] = bool(result.armed)
                if result.armed:
                    self.actuations += 1
                    self._current_stage = target.name
                    if self._dwell is not None:
                        self._dwell.record_flip(self._round)
                else:
                    self._count_veto("actuator refused")
        self._last_record = record
        self._write_trace(record)
        self._maybe_log(record, reduced)

        self._prefill_rounds = 0
        self._decode_rounds = 0
        self._idle_rounds = 0
        self._rank_ms_sum = 0.0
        self._rank_ms_n = 0
        self._epoch += 1
        return record

    # -- act-mode interlocks --------------------------------------------------
    def _act_interlocks(self, target, sample, reduced) -> tuple:
        """Every guard a proposal must clear before it becomes a move.

        ``(allowed, veto_reason)``. Each one is carried over from a phase
        whose falsifier proved it necessary, and each is BINDING in act mode
        where observe only reported it.
        """
        # 1. Selectability (phase 3). A stage the boot table judged
        #    unreachable -- weights differ, or the KV vector was never
        #    declared -- is not a candidate, whatever the regime says.
        if self._table is None or self._current_stage is None:
            return False, (
                "act needs a boot stage table AND a known current stage; "
                "without both there is no 'from' for the move and no set the "
                "'to' was validated against"
            )
        if self._table_plan is not None:
            ok, why = self._table_plan.is_selectable(target.name)
            if not ok:
                return False, why

        # 2. Dwell (F1). Hysteresis says the signal is real; dwell says the
        #    move is affordable. A load that alternates faster than a flip
        #    amortizes passes the first and must fail the second.
        if self._dwell is not None:
            ok, why = self._dwell.allows(self._round)
            if not ok:
                return False, why

        # 3. Group agreement. Acting on a verdict the ranks did not share is
        #    the #94/#194/#259 hang under an actuator -- the exact thing the
        #    observe phase existed to rule out.
        if reduced is not None and not reduced["agreed"]:
            return False, (
                "the TP ranks classified differently this boundary; a flip "
                "under a disputed verdict is the hang the observe phase "
                "existed to rule out"
            )
        if self._tp_size > 1 and self._collective_min is None:
            return False, (
                "multi-rank group with no consensus channel: the verdict's "
                "rank-uniformity is unchecked, so it may not move anything"
            )

        # 4. The one-boundary-stale veto, BINDING in act mode (phase 2 built
        #    it, phase 3 makes it bite). The spread is by construction one
        #    boundary old; what act adds is that it must EXIST. Flipping with
        #    no group timing at all means the plan has never been checked
        #    against the rig it is about to move.
        if self._tp_size > 1 and sample.rank_ms_spread_pct is None:
            return False, (
                "no group timing from the previous boundary (one-boundary-"
                "stale veto): every rank was blind, so the planner's split "
                "has not been checked against this rig and a flip would be "
                "unvalidated"
            )
        return True, ""

    def _count_veto(self, reason: str) -> None:
        key = reason.split(":")[0][:48]
        self.vetoes[key] = self.vetoes.get(key, 0) + 1

    # -- rule 2: the consensus channel ---------------------------------------
    def _consensus(
        self, regime: str, stage_index: int, mean_ms: Optional[float]
    ) -> Optional[Dict]:
        """MIN-reduce this rank's proposal; record the group's view.

        Entered unconditionally at every boundary when a channel exists -- the
        gate above is the round counter, not the verdict, so a rank whose
        classification differs still arrives here and the disagreement becomes
        a reduced value instead of a missing peer.
        """
        if self._collective_min is None:
            self._last_spread_pct = None
            return None
        self.consensus_rounds += 1
        payload = pack_proposal(regime, stage_index, self._epoch, mean_ms)
        reduced = unpack_reduced(self._collective_min(payload))
        self._last_spread_pct = reduced.get("rank_ms_spread_pct")
        if not reduced["agreed"]:
            self.desyncs += 1
            # Loud, and NOT fatal: see rule 3 in the module docstring. The
            # count is the phase-3 gate, so it has to survive the run.
            logger.warning(
                "%s DESYNC at round %d: the ranks classified differently "
                "(%s; this rank: regime=%r stage_index=%d epoch=%d). Nothing "
                "was actuated, so this is a measurement, not a hang -- but a "
                "non-zero desync count BLOCKS wiring any actuator, because "
                "the same disagreement under an actuator is the "
                "#94/#194/#259 hang.",
                LOG_PREFIX,
                self._round,
                "; ".join(reduced["disagreements"]),
                regime,
                stage_index,
                self._epoch,
            )
        return reduced

    def _stage_index(self, target) -> int:
        if target is None or self._table is None:
            return -1
        for i, stage in enumerate(self._table.stages):
            if stage.name == target.name:
                return i
        return -1

    def _maybe_log(self, record: Dict, reduced: Optional[Dict]) -> None:
        if self._verdicts % self._log_every != 0:
            return
        spread = record["rank_ms_spread_pct"]
        logger.info(
            "%s round %d: regime=%s (prefill %s, decode %s, occupancy %s, "
            "queue %d req / %d tok); would flip to %s -- %s; rank ms mean %s, "
            "spread %s; consensus %s. NOT ACTUATED (observe-only).",
            LOG_PREFIX,
            record["round"],
            record["regime"],
            _pct(record["prefill_share"]),
            _pct(record["decode_share"]),
            _pct(record["occupancy"]),
            record["queued_reqs"],
            record["queued_prompt_tokens"],
            record["would_flip_to"] or "nothing",
            record["reason"],
            (
                f"{record['rank_mean_forward_ms']:.2f}"
                if record["rank_mean_forward_ms"] is not None
                else "absent (no timed forward this window)"
            ),
            f"{spread:.1f}%" if spread is not None else "absent",
            (
                "not checked (no channel)"
                if reduced is None
                else ("agreed" if reduced["agreed"] else "DESYNC")
            ),
        )

    # -- the trace, which is what the card gates actually consume -----------
    def _write_trace(self, record: Dict) -> None:
        """Append one verdict. Never raises into the scheduler loop: a full
        disk must not stop a server from serving, and a trace that stops
        mid-run is visible to the reader as a short file."""
        if self._trace_path is None:
            return
        try:
            if self._trace_fh is None:
                self._trace_fh = open(self._trace_path, "a", buffering=1)
                self._trace_fh.write(
                    json.dumps(
                        {
                            "kind": "header",
                            "mode": self._mode,
                            "rank": self._trace_rank,
                            "interval": self._interval,
                        }
                    )
                    + "\n"
                )
            self._trace_fh.write(
                json.dumps({"kind": "verdict", "rank": self._trace_rank, **record})
                + "\n"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "%s trace write failed (%r); disabling the trace for this "
                "run. The classifier is unaffected.",
                LOG_PREFIX,
                exc,
            )
            self._trace_path = None

    def close_trace(self) -> None:
        """Write the summary line and close. Idempotent.

        The summary is the LAST line by construction, so a reader that finds
        one knows the run ended cleanly and a reader that does not knows the
        server was killed -- which is the difference between "zero desyncs"
        and "zero desyncs so far".
        """
        if self._trace_fh is None:
            return
        try:
            self._trace_fh.write(
                json.dumps(
                    {"kind": "summary", "rank": self._trace_rank, **self.summary()}
                )
                + "\n"
            )
            self._trace_fh.close()
        except Exception:  # noqa: BLE001
            pass
        finally:
            self._trace_fh = None

    def summary(self) -> Dict:
        """What the observe run has to show for itself.

        ``desyncs`` is the phase-3 gate; ``proposals`` is how often the
        controller WOULD have moved, which is the number that says whether
        actuating is worth building at all on this workload.
        """
        return {
            "rounds": self._round,
            "verdicts": self._verdicts,
            "regime": self._sensor.regime,
            "transitions": self._sensor.transitions,
            "proposals": self.proposals,
            "consensus_rounds": self.consensus_rounds,
            "desyncs": self.desyncs,
            "uncoordinated": self.uncoordinated,
            "mode": self._mode,
            "actuations": self.actuations,
            "vetoes": dict(self.vetoes),
        }


def _pct(value: Optional[float]) -> str:
    return "absent" if value is None else f"{100.0 * value:.0f}%"


def build_regime_observer(scheduler) -> Optional[RegimeObserver]:
    """Construct the observer for one scheduler, or ``None`` when off.

    Off is the default and costs one flag read at build time and nothing per
    round: the scheduler's attribute stays ``None`` and the call site is one
    predictable branch.

    In ``act`` mode the actuator is built HERE and injected, so the observer
    still holds no import path to #297/#330. The entry gate was already
    cleared at argument time (``server_args._handle_regime_controller``); this
    function does not re-litigate it, it only wires what the gate authorized.
    """
    server_args = scheduler.server_args
    mode = resolve_mode(server_args)
    if mode == MODE_OFF:
        return None
    tp_size = int(getattr(server_args, "tp_size", 1) or 1)
    collective = None
    if tp_size > 1:
        cpu_group = getattr(scheduler, "tp_cpu_group", None)
        if cpu_group is not None:
            from sglang.srt.managers.kv_pressure_runtime import (
                default_collective_min,
            )

            # The same bounded MIN all-reduce the #287 ladder uses. Sharing it
            # is the point: a second consensus channel would be a second set
            # of assumptions about what a dead peer looks like.
            collective = default_collective_min(cpu_group)

    interval = int(getattr(server_args, "kv_pressure_consensus_interval", 8) or 8)

    # The boot stage table (#363 phase 3). Absent is a real state, not an
    # empty table pretending to be a full one: on a rig with no probe the
    # observer reports the regime and says the table is absent.
    table_plan = getattr(scheduler, "regime_stage_table", None)
    table = table_plan.table if table_plan is not None else None
    current = table_plan.booted if table_plan is not None else None
    if table_plan is not None:
        logger.info(
            "%s stage table: %d stage(s), %d reachable at runtime, %d flip "
            "target(s), booted on %r",
            LOG_PREFIX,
            len(table_plan),
            len(table_plan.reachable),
            len(table_plan.flip_targets),
            table_plan.booted,
        )
        for line in table_plan.describe():
            logger.info("%s   stage %s", LOG_PREFIX, line)
        for note in table_plan.notes:
            logger.info("%s   note: %s", LOG_PREFIX, note)

    commit_fn = None
    dwell = None
    if mode == MODE_ACT:
        # Imported HERE, in the act branch only. A module-scope import would
        # give observe a path to the actuators, which is the property the
        # no-actuator test pins.
        from sglang.srt.managers.regime_act import build_regime_actuator

        commit_fn = build_regime_actuator(scheduler, table_plan).apply
        dwell = _dwell_for(table_plan)

    observer = RegimeObserver(
        consensus_interval=interval,
        tp_size=tp_size,
        collective_min=collective,
        mode=mode,
        table=table,
        table_plan=table_plan,
        current_stage=current,
        commit_fn=commit_fn,
        dwell=dwell,
        trace_path=_rank_trace_path(
            getattr(server_args, "regime_trace", None), scheduler
        ),
        trace_rank=_tp_rank_of(scheduler),
    )
    install_trace_shutdown_hook(observer)
    return observer


#: Rounds of dwell to assume before the live mean round time is known. The
#: real value is derived (DESIGN_363 section 3.3) from the flip's instrumented
#: cost and the measured round time; until a flip has been timed this is the
#: conservative stand-in, and it is deliberately long.
BOOTSTRAP_DWELL_ROUNDS = 512


def _dwell_for(table_plan) -> DwellGate:
    """The dwell gate for the most expensive selectable flip in the table.

    One gate for the table rather than one per stage: the gate answers "may
    anything move now", and sizing it to the cheapest stage would let an
    expensive flip follow a cheap one inside the expensive one's own
    amortization window.
    """
    if table_plan is None:
        return DwellGate(min_rounds=BOOTSTRAP_DWELL_ROUNDS)
    costs = [p.stage.flip_cost_s for p in table_plan.flip_targets]
    if not costs:
        return DwellGate(min_rounds=BOOTSTRAP_DWELL_ROUNDS)
    # Round time is not measured yet at build time; DESIGN_363 section 3.3's
    # derivation runs once the observer has timed a window. Until then the
    # bootstrap value stands, scaled by the worst flip so a 6 s move is not
    # held for the same span as a 0.3 s one.
    worst = max(costs)
    return DwellGate(min_rounds=max(BOOTSTRAP_DWELL_ROUNDS, int(worst * 128)))


def _tp_rank_of(scheduler) -> Optional[int]:
    ps = getattr(scheduler, "ps", None)
    rank = getattr(ps, "tp_rank", None) if ps is not None else None
    if rank is None:
        rank = getattr(scheduler, "tp_rank", None)
    return None if rank is None else int(rank)


def _rank_trace_path(base: Optional[str], scheduler) -> Optional[str]:
    """One trace file PER RANK, derived from the single ``--regime-trace``.

    Every rank of a TP group runs its own observer and its own consensus
    proposal, so the desync count is a property of the GROUP and the reader
    refuses a multi-rank verdict assembled from one rank's file. Left sharing
    a path they append into one interleaved file: the lines stay atomic (the
    first gates window wrote 93 603 of them with none torn), but the ranks
    become indistinguishable and the three summary lines overwrite each
    other's meaning. Suffixing here keeps the flag a single path for the
    operator and still gives the reader what its own contract asks for.
    """
    if not base:
        return None
    # The rank lives on the ParallelState, not on the scheduler. Reading
    # ``scheduler.tp_rank`` returned None and every rank silently shared one
    # file -- found in the 2026-08-01 re-run, where three ranks interleaved
    # 104 862 verdicts into one path.
    ps = getattr(scheduler, "ps", None)
    rank = getattr(ps, "tp_rank", None) if ps is not None else None
    if rank is None:
        rank = getattr(scheduler, "tp_rank", None)
    if rank is None:
        return base
    root, ext = os.path.splitext(str(base))
    return f"{root}.rank{int(rank)}{ext or '.jsonl'}"


def close_regime_trace(scheduler) -> None:
    """Close the observer's trace, from anywhere, safely.

    Called from the scheduler process's ``finally`` -- the one block that runs
    on BOTH the graceful path (a ShutdownReq breaks the event loop) and the
    exception path (including ``KeyboardInterrupt``, which ``except
    Exception`` does not catch but ``finally`` still honours). Idempotent, and
    it never raises: a teardown helper that can throw turns a clean shutdown
    into a confusing one and an already-failing shutdown into a lost
    traceback.
    """
    observer = getattr(scheduler, "regime_observer", None)
    if observer is None:
        return
    try:
        observer.close_trace()
    except Exception as exc:  # noqa: BLE001
        logger.warning("%s could not close the verdict trace (%r)", LOG_PREFIX, exc)


def install_trace_shutdown_hook(observer) -> None:
    """Make the summary line land on every way this process actually stops.

    The reader refuses a trace without a summary line, because "zero desyncs"
    and "zero desyncs SO FAR" are different claims (DESIGN_363 section 13.3).
    That refusal is only useful if a run which DID end cleanly always produces
    one, so all three stop paths are covered:

    * a **ShutdownReq** breaks the event loop -> the scheduler's ``finally``
      calls :func:`close_regime_trace`;
    * a **scheduler-loop exception** (or ``KeyboardInterrupt``) -> the same
      ``finally``;
    * **SIGTERM** -> nothing above runs. Python's default disposition
      terminates the process without unwinding, so neither ``finally`` nor
      ``atexit`` fires. Hence the handler installed here.

    The SIGTERM handler CHAINS: it closes the trace and then hands the signal
    on to whatever was installed before, or re-raises it against the default
    disposition. It must not change how the server dies -- only what it
    writes on the way out.
    """
    if observer is None or observer.trace_path is None:
        return
    atexit.register(observer.close_trace)

    previous = signal.getsignal(signal.SIGTERM)

    def _close_then_chain(signum, frame):
        observer.close_trace()
        if callable(previous):
            return previous(signum, frame)
        # SIG_DFL / SIG_IGN: restore it and re-deliver, so the process dies
        # exactly as it would have without this hook.
        signal.signal(signum, previous if previous is not None else signal.SIG_DFL)
        if previous is signal.SIG_IGN:
            return None
        os.kill(os.getpid(), signum)
        return None

    try:
        signal.signal(signal.SIGTERM, _close_then_chain)
    except ValueError:
        # Not the main thread of this process. The other two paths still
        # cover it, and a scheduler that cannot install the handler must not
        # fail to boot over a trace file.
        logger.info(
            "%s: SIGTERM trace hook not installed (not the main thread); the "
            "shutdown and exception paths still write the summary.",
            LOG_PREFIX,
        )


def build_regime_stage_table(scheduler):
    """The boot stage table for one scheduler, or ``None``.

    The booted stage is assembled from what this server ACTUALLY launched with
    -- the MLP vector, the KV token vector and the per-rank budgets in force --
    so reachability is computed against where we are rather than against a
    plan. Candidates come from the planner through
    ``regime_stages.planner_candidates``; with no feed bound the table holds
    the booted stage alone, which is the honest state on a rig with no probe
    and is reported as such.

    Never raises into the scheduler loop: a table that cannot be built is a
    logged absence, because an observability feature must not be able to stop
    a server from serving.
    """
    from sglang.srt.managers.regime_stages import (
        build_stage_table,
        planner_candidates,
    )

    server_args = scheduler.server_args
    try:
        booted = _booted_stage(scheduler)
        if booted is None:
            return None
        declared = _declared_vectors(scheduler)
        candidates, notes = planner_candidates(server_args)
        plan = build_stage_table(
            booted=booted, candidates=candidates, declared_vectors=declared
        )
        for note in notes:
            logger.info("%s stage feed: %s", LOG_PREFIX, note)
        return plan
    except Exception as exc:  # noqa: BLE001 -- a table is not worth a server
        logger.warning(
            "%s could not build the boot stage table (%r); running without "
            "one. The regime is still classified and logged; no stage is "
            "named and nothing could be selected even in act mode.",
            LOG_PREFIX,
            exc,
        )
        return None


def _booted_stage(scheduler):
    """The configuration this server is actually running, as a Stage."""
    from sglang.srt.managers.regime_classifier import REGIME_MIXED, Stage

    server_args = scheduler.server_args
    kv_vector = _current_kv_vector(scheduler)
    if kv_vector is None:
        return None
    budgets = tuple(
        int(x) for x in (getattr(server_args, "vram_budget_mib", None) or ())
    )
    if not budgets:
        budgets = tuple(
            int(x) for x in (getattr(server_args, "rank_gpu_memory_mib", None) or ())
        )
    capacity = int(getattr(scheduler, "max_total_num_tokens", 0) or 0)
    if capacity <= 0:
        return None
    weights = getattr(server_args, "rank_mlp_ratio", None)
    return Stage(
        name="booted",
        # The booted stage serves whatever regime the server is in; it is the
        # reference, not a regime's optimum, so it claims the resting one.
        regime=REGIME_MIXED,
        weight_vector=tuple(int(x) for x in weights) if weights else None,
        kv_token_vector=kv_vector,
        vram_budget_mib=budgets,
        max_total_num_tokens=capacity,
        # The reference stage's gain against itself is zero by definition, and
        # StageTable exempts the reference from the band rule for that reason.
        measured_gain_pct=0.0,
        measured_band_pct=0.0,
        flip_cost_s=0.0,
    )


def _current_kv_vector(scheduler):
    reshard_rt = getattr(scheduler, "kv_reshard_runtime", None)
    if reshard_rt is not None:
        return tuple(int(x) for x in reshard_rt.current_vector)
    try:
        from sglang.srt.distributed.utils import get_cp_token_ratios

        ratios = get_cp_token_ratios()
    except Exception:
        return None
    return tuple(int(x) for x in ratios) if ratios else None


def _declared_vectors(scheduler):
    reshard_rt = getattr(scheduler, "kv_reshard_runtime", None)
    if reshard_rt is None:
        return ()
    return tuple(tuple(int(x) for x in v) for v in reshard_rt.allowed_vectors)


def rank_forward_ms_from(scheduler) -> Optional[float]:
    """This rank's last measured forward device ms, or ``None``.

    The honest reading of the #252 sensing, per DESIGN_363 section 7.1: the
    compute/wait split is installed on plain-prefill forwards only and reports
    NOTHING for a graph-covered forward rather than a wrong zero. So this
    returns a number when the last prefill was measurable and ``None``
    otherwise, and the absence travels all the way into the packed payload as
    a sentinel -- a blind rank must not read as an infinitely fast one.
    """
    reporter = getattr(scheduler, "metrics_reporter", None)
    log = getattr(reporter, "rank_prefill_log", None) if reporter else None
    if log is None or not getattr(log, "last_split_known", False):
        return None
    return getattr(log, "last_gpu_ms", None)


__all__ = [
    "ENV_MODE",
    "LOG_PREFIX",
    "MODES",
    "MODE_ACT",
    "MODE_OBSERVE",
    "MODE_OFF",
    "REGIMES",
    "RegimeObserver",
    "build_regime_observer",
    "build_regime_stage_table",
    "close_regime_trace",
    "install_trace_shutdown_hook",
    "observe_mode",
    "rank_forward_ms_from",
    "resolve_mode",
]
