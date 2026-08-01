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

import logging
import os
from typing import Callable, Dict, List, Optional

from sglang.srt.managers.regime_classifier import (
    DEFAULT_WINDOW_ROUNDS,
    REGIMES,
    RegimeSample,
    RegimeSensor,
    StageTable,
    pack_proposal,
    unpack_reduced,
)

logger = logging.getLogger(__name__)

LOG_PREFIX = "REGIME-OBSERVE"

#: Mode values. ``off`` builds nothing; ``observe`` classifies and logs.
#: Actuating modes do not exist yet and are refused by name rather than
#: silently treated as ``observe``.
MODE_OFF = "off"
MODE_OBSERVE = "observe"
MODES = (MODE_OFF, MODE_OBSERVE)

#: Phase-2 gate. The flag proper (``--regime-controller``) lands with phase 3,
#: when there is an action to authorize; an env read keeps the observe-only
#: ship inside one module instead of spreading a knob for a no-op across
#: server_args and environ.
ENV_MODE = "SGLANG_REGIME_OBSERVE"

#: Emit one record every N verdicts. The verdict itself is cheap (integer
#: arithmetic over a window plus one small reduction); the log line is what
#: costs, so it is the thing that is throttled.
DEFAULT_LOG_EVERY = 1


def observe_mode() -> str:
    """``off`` unless the env var explicitly asks for ``observe``.

    An unknown value is refused loudly rather than rounded to the nearest
    mode: a typo that silently disables an observability run wastes the run.
    """
    raw = (os.environ.get(ENV_MODE) or "").strip().lower()
    if not raw or raw in ("0", "false", "no", MODE_OFF):
        return MODE_OFF
    if raw in ("1", "true", "yes", MODE_OBSERVE):
        return MODE_OBSERVE
    raise ValueError(
        f"{ENV_MODE}={raw!r} is not a known mode; use one of "
        f"{', '.join(MODES)} (or 1/0). Actuating modes do not exist yet: "
        f"#363 phase 2 ships observe-only."
    )


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

        self._round = 0
        self._epoch = 0
        self._verdicts = 0
        # Tier-R window accumulators.
        self._prefill_rounds = 0
        self._decode_rounds = 0
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
    def last_record(self) -> Optional[Dict]:
        return self._last_record

    # -- the per-round hook --------------------------------------------------
    def on_round(
        self,
        *,
        prefill_active: bool,
        held_tokens: int,
        capacity_tokens: int,
        running_bs: int,
        queued_reqs: int = 0,
        queued_prompt_tokens: int = 0,
        max_queued_prompt_tokens: int = 0,
        rank_forward_ms: Optional[float] = None,
    ) -> Optional[Dict]:
        """One between-tick boundary.

        Every argument except ``rank_forward_ms`` must be REPLICATED across
        the TP group -- the caller's obligation, stated here because it is the
        whole uniformity argument. ``rank_forward_ms`` is the one rank-local
        input and is treated as such: accumulated, never branched on, and
        released only through the reduction.

        Returns the verdict record at a consensus boundary, ``None`` between
        boundaries.
        """
        self._round += 1
        if prefill_active:
            self._prefill_rounds += 1
        else:
            self._decode_rounds += 1
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
            "queued_reqs": sample.queued_reqs,
            "queued_prompt_tokens": sample.queued_prompt_tokens,
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
        }
        self._last_record = record
        self._maybe_log(record, reduced)

        self._prefill_rounds = 0
        self._decode_rounds = 0
        self._rank_ms_sum = 0.0
        self._rank_ms_n = 0
        self._epoch += 1
        return record

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
            "actuations": 0,
        }


def _pct(value: Optional[float]) -> str:
    return "absent" if value is None else f"{100.0 * value:.0f}%"


def build_regime_observer(scheduler) -> Optional[RegimeObserver]:
    """Construct the observer for one scheduler, or ``None`` when off.

    Off is the default and costs one env read at build time and nothing per
    round: the scheduler's attribute stays ``None`` and the call site is one
    predictable branch.
    """
    if observe_mode() != MODE_OBSERVE:
        return None

    server_args = scheduler.server_args
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
    return RegimeObserver(
        consensus_interval=interval,
        tp_size=tp_size,
        collective_min=collective,
        # No stage table yet: the planner-solved stages are declared at boot
        # in phase 3, when something can select between them. Until then the
        # observer reports the regime and says the table is absent rather
        # than inventing one.
        table=None,
    )


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
    "MODE_OBSERVE",
    "MODE_OFF",
    "REGIMES",
    "RegimeObserver",
    "build_regime_observer",
    "observe_mode",
    "rank_forward_ms_from",
]
