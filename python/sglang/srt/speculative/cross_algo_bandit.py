"""Cross-algorithm speculative decoding (T156 stage 4) -- the rung bandit.

Non-stationary bandit over the co-resident speculative rungs
(``--speculative-cross-algorithm-force auto``):

* arms: ``("nextn", k)`` for every k in the adaptive-config candidate set
  (default [1, 2, 3]) plus ``("dflash", block_size)``;
* objective per rung r (design 4b):
      ``score(r) = EMA[accepted tokens per verify round](r)
                   / EMA[round duration seconds](r)``
  i.e. a live tokens/second-equivalent, both terms MEASURED per rung by
  ``RungMetrics`` (adaptive_spec_params) -- never assumed;
* selection: argmax, but only when the best rung beats the ACTIVE rung by a
  relative ``deadzone`` margin.

Anti-flap guards (all mandatory, stage-1 lessons):

* MINIMUM DWELL -- after any switch, the rung is held ``min_dwell_rounds``
  verify rounds before the next decision (switch cost is ~1 round; the
  default 64 rounds caps switching overhead at ~1.6%).
* BURN-IN -- the first ``burn_in_rounds`` results after the attribution
  stream changes rungs are excluded from the estimator (measured stage 3:
  DFLASH accepts dip ~4% in the first rounds after a switch).
* STALENESS PROBING -- a rung that has not been active for
  ``probe_interval_rounds`` has a frozen estimate; the bandit periodically
  runs one short ``probe_window_rounds`` window on the best inactive rung.
  Never-measured rungs are probed as soon as the dwell allows (cold start);
  rungs whose stale score is clearly inferior
  (``score * probe_inferior_factor < score(active)``) are left alone.

RANK UNIFORMITY (design 4c): the round-duration term is per-rank WALL CLOCK
and NOT rank-uniform, so -- unlike the stage-1 k-ladder, which every rank
recomputes identically -- ``decide()`` runs on RANK 0 ONLY and the chosen
rung id is broadcast to all ranks over the TP CPU (gloo) group at a
rank-uniform round boundary (CrossAlgoWorker.maybe_switch_rung). The
broadcast lives in eager scheduler code, never inside a graph capture.

Tunables: every CrossBanditConfig field can be overridden via
``SGLANG_CROSS_BANDIT_<FIELDNAME>`` (upper-cased), e.g.
``SGLANG_CROSS_BANDIT_MIN_DWELL_ROUNDS=128``. Decision history JSONL:
``SGLANG_CROSS_BANDIT_TRACE=<path>`` (rank 0; one record per decision point
with per-rung scores, event, active/target rung).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, fields
from typing import Optional

from sglang.srt.speculative.adaptive_spec_params import RungMetrics

logger = logging.getLogger(__name__)

# A rung id: ("nextn", k) or ("dflash", block_size).
Rung = tuple

ENV_PREFIX = "SGLANG_CROSS_BANDIT_"


def rung_str(rung: Rung) -> str:
    return f"{rung[0]}:{rung[1]}"


@dataclass
class CrossBanditConfig:
    # Rounds between decision points (each is one rank-0 decide + one CPU
    # broadcast of the rung id). 16 rounds ~ 0.5 s at bs=1.
    decide_interval_rounds: int = 16
    # Minimum residence on a rung before any new decision applies.
    min_dwell_rounds: int = 64
    # Results excluded from the estimator after the attributed rung changes.
    burn_in_rounds: int = 4
    # Relative score margin the best rung must clear to displace the active
    # one. NOT a safety margin against measurement noise (the slow EMA is
    # that); it prices in the ~1-round switch cost + post-switch dip.
    deadzone: float = 0.06
    # Staleness age (rounds since last active) after which a rung's estimate
    # is considered stale and it becomes probe-eligible. At ~30 rounds/s this
    # is ~17 s; probe overhead = window/interval per eligible arm (measured
    # 2026-07-22: 256 caused ~13% probe+switch time with two near-equal
    # arms, netting below the best static rung).
    probe_interval_rounds: int = 512
    # Length of a staleness probe window.
    probe_window_rounds: int = 32
    # Optimistic (UCB-flavored) probe gate: a stale rung is re-probed only if
    # its decayed BEST-EVER score * factor >= score(active). The best-ever
    # score -- not the frozen current one -- separates rungs that were merely
    # probed during unfavorable content (worth re-checking; e.g. DFLASH on a
    # prose stretch) from rungs that have NEVER been competitive (their score
    # is structurally capped; e.g. NEXTN k=1) and should stop eating probe
    # windows. Measured 2026-07-22: with a frozen-score gate, k=1 (frozen 57)
    # outranked DFLASH (frozen 56) for probes and the bandit missed a
    # content shift DFLASH would have won.
    probe_inferior_factor: float = 1.2
    # Per-decision multiplicative decay of the remembered best score
    # (non-stationarity: an ancient lucky burst must not keep a rung
    # probe-eligible forever). 0.995 per decision point ~= 26% decay per
    # 1000 rounds at decide_interval 16.
    probe_best_decay: float = 0.995
    # Round-duration samples a rung needs before it is scoreable.
    min_time_samples: int = 3

    @classmethod
    def from_env(cls) -> "CrossBanditConfig":
        kwargs = {}
        for f in fields(cls):
            raw = os.environ.get(ENV_PREFIX + f.name.upper())
            if raw is not None:
                cast = float if isinstance(f.default, float) else int
                kwargs[f.name] = cast(raw)
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        for name in (
            "decide_interval_rounds",
            "min_dwell_rounds",
            "probe_interval_rounds",
            "probe_window_rounds",
            "min_time_samples",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"cross bandit: {name} must be >= 1")
        if self.burn_in_rounds < 0:
            raise ValueError("cross bandit: burn_in_rounds must be >= 0")
        if self.deadzone < 0.0:
            raise ValueError("cross bandit: deadzone must be >= 0")
        if self.probe_inferior_factor < 1.0:
            raise ValueError("cross bandit: probe_inferior_factor must be >= 1")
        if not (0.0 < self.probe_best_decay <= 1.0):
            raise ValueError("cross bandit: probe_best_decay must be in (0, 1]")
        if self.probe_window_rounds >= self.probe_interval_rounds:
            raise ValueError(
                "cross bandit: probe_window_rounds must be < probe_interval_rounds"
            )


class CrossAlgoBandit:
    """See the module docstring. ``observe_result`` may run on every rank
    (timing is rank-local, harmless); ``decide`` must run on rank 0 only and
    its result must be broadcast."""

    def __init__(
        self,
        rungs: list,
        initial: Rung,
        cfg: Optional[CrossBanditConfig] = None,
        trace_path: Optional[str] = None,
    ):
        assert initial in rungs, f"initial rung {initial} not in {rungs}"
        assert len(set(rungs)) == len(rungs), f"duplicate rungs: {rungs}"
        self.cfg = cfg if cfg is not None else CrossBanditConfig.from_env()
        self.rungs = list(rungs)
        self.metrics = RungMetrics()
        self.active: Rung = initial
        self.switch_count = 0
        self._last_switch_round = 0
        self._last_active_round = {r: 0 for r in self.rungs}
        # Probe state: while _probe_until is set, `active` is the probe rung
        # and _pre_probe_rung the incumbent to fall back to.
        self._probe_until: Optional[int] = None
        self._pre_probe_rung: Optional[Rung] = None
        # Burn-in: attribution tracking over the (possibly delayed, under
        # overlap) result stream. Results are processed in production order,
        # so "first N results attributed to a different rung than the
        # previous result" == "first N rounds after the switch".
        self._attr_rung: Optional[Rung] = None
        self._attr_count = 0
        # Decayed best-ever score per rung (the optimistic probe gate).
        self._best_score: dict = {}
        self._trace_path = trace_path
        self._t0 = time.monotonic()

    # ------------------------------------------------------------------
    # Measurement (any rank)
    # ------------------------------------------------------------------
    def observe_result(
        self,
        rung: Rung,
        num_correct_drafts_per_req: list,
        batch_size: int,
        now: Optional[float] = None,
    ) -> None:
        if not num_correct_drafts_per_req:
            return
        if rung != self._attr_rung:
            self._attr_rung = rung
            self._attr_count = 0
        self._attr_count += 1
        self.metrics.observe(
            rung,
            num_correct_drafts_per_req,
            batch_size,
            now=now,
            record=self._attr_count > self.cfg.burn_in_rounds,
        )

    def score(self, rung: Rung) -> Optional[float]:
        return self.metrics.reward(
            rung, min_time_samples=self.cfg.min_time_samples
        )

    def scores(self) -> dict:
        return {r: self.score(r) for r in self.rungs}

    # ------------------------------------------------------------------
    # Decision (rank 0 only)
    # ------------------------------------------------------------------
    def decide(self, round_idx: int) -> Rung:
        """Return the rung to run from the next round boundary on.

        Pure control flow over the rank-local metrics; returns the CURRENT
        rung when nothing warrants a change (the caller broadcasts the
        returned id unconditionally -- receiving one's own current rung is
        the no-decision case on every rank).
        """
        scores = self.scores()
        active = self.active
        self._last_active_round[active] = round_idx
        # Maintain the decayed best-ever scores (probe gate; see config).
        for r, s in scores.items():
            best = self._best_score.get(r)
            if best is not None:
                best *= self.cfg.probe_best_decay
            if s is not None and (best is None or s > best):
                best = s
            if best is not None:
                self._best_score[r] = best

        # 1) A probe window in progress: hold until it ends, then adopt the
        #    probe rung iff it measured better than the incumbent. NO deadzone
        #    here, deliberately: the deadzone prices the switch cost of
        #    LEAVING a rung, but at probe end the switch is already paid --
        #    staying on the probe rung is free, RETURNING costs a switch. A
        #    plain > (tie -> return, for stability) therefore biases
        #    correctly; requiring the deadzone on top lost a genuinely better
        #    rung by 0.4 points (measured 2026-07-22, probe 78 vs incumbent
        #    74 with deadzone bar 78.4).
        if self._probe_until is not None:
            if round_idx < self._probe_until:
                return self._emit(round_idx, "probe_hold", scores, active)
            incumbent = self._pre_probe_rung
            self._probe_until = None
            self._pre_probe_rung = None
            probe_s = scores.get(active)
            inc_s = scores.get(incumbent)
            if probe_s is not None and (inc_s is None or probe_s > inc_s):
                # Probe won; it is already active. Restart the dwell clock so
                # the adoption is not immediately re-contested.
                self._last_switch_round = round_idx
                return self._emit(round_idx, "probe_adopt", scores, active)
            self._switch(incumbent, round_idx)
            return self._emit(round_idx, "probe_return", scores, incumbent)

        # 2) Minimum dwell.
        if round_idx - self._last_switch_round < self.cfg.min_dwell_rounds:
            return self._emit(round_idx, "dwell_hold", scores, active)

        cur_s = scores.get(active)

        # 3) Exploit: argmax over measured rungs, deadzone-guarded.
        measured = {r: s for r, s in scores.items() if s is not None}
        if measured and cur_s is not None:
            best = max(measured, key=measured.get)
            if best != active and measured[best] > cur_s * (
                1.0 + self.cfg.deadzone
            ):
                self._switch(best, round_idx)
                return self._emit(round_idx, "switch", scores, best)

        # 4) Explore: probe the best stale / never-measured inactive rung.
        cand = self._probe_candidate(scores, round_idx)
        if cand is not None:
            self._pre_probe_rung = active
            self._probe_until = round_idx + self.cfg.probe_window_rounds
            self._switch(cand, round_idx)
            return self._emit(round_idx, "probe_start", scores, cand)

        return self._emit(round_idx, "hold", scores, active)

    def _probe_candidate(self, scores: dict, round_idx: int) -> Optional[Rung]:
        cur_s = scores.get(self.active)
        unmeasured = []
        stale = []
        for r in self.rungs:
            if r == self.active:
                continue
            s = scores.get(r)
            if s is None:
                # Cold start: gather data as soon as the dwell allows.
                unmeasured.append(r)
                continue
            if (
                round_idx - self._last_active_round[r]
            ) < self.cfg.probe_interval_rounds:
                continue
            # Optimistic gate + priority key: the decayed best-ever score
            # (falls back to the frozen current score before any best is
            # recorded). A rung that has NEVER been competitive stops eating
            # probe windows; one that ever shone stays re-checkable.
            best = self._best_score.get(r, s)
            if (
                cur_s is not None
                and best * self.cfg.probe_inferior_factor < cur_s
            ):
                continue
            stale.append((best, r))
        if unmeasured:
            return unmeasured[0]
        if stale:
            return max(stale, key=lambda t: t[0])[1]
        return None

    def _switch(self, target: Rung, round_idx: int) -> None:
        self.active = target
        self.switch_count += 1
        self._last_switch_round = round_idx
        self._last_active_round[target] = round_idx

    def _emit(self, round_idx: int, event: str, scores: dict, target: Rung) -> Rung:
        if self._trace_path is not None:
            rec = {
                "t_s": round(time.monotonic() - self._t0, 3),
                "round": round_idx,
                "event": event,
                "active": rung_str(self.active),
                "target": rung_str(target),
                "switches": self.switch_count,
                "scores": {
                    rung_str(r): (None if s is None else round(s, 2))
                    for r, s in scores.items()
                },
                "best": {
                    rung_str(r): round(b, 2)
                    for r, b in self._best_score.items()
                },
            }
            try:
                with open(self._trace_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except OSError:
                logger.exception("cross bandit: trace write failed; disabling")
                self._trace_path = None
        return target
