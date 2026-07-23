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

CONTEXT GATE (--speculative-cross-algorithm-ctx-gate): the caller may pass an
``eligible`` rung set into ``decide``; ineligible rungs (the short-context
DFLASH drafter above its context threshold) are never selected, never probed
(logged "skipped: ctx gate"), and the active rung is evicted when it becomes
ineligible (overriding dwell). ``eligible=None`` == gate off == pre-gate
behavior.

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
    # ---- Analytic k selection within the NEXTN family (task A). --------
    # 1 (default): the NEXTN k rungs are never probed; k* is computed
    # analytically from the shared per-position accept statistics
    # (NextnAcceptModel) and k moves within the family are plain pointer
    # swaps governed by k_dwell_rounds + k_switch_margin. 0: legacy
    # behavior (every k rung probed like any arm).
    analytic_k: int = 1
    # Minimum rounds between k moves WITHIN the NEXTN family. 16, not the
    # family dwell of 64: a family switch pays warm-keep conversion, a
    # post-switch accept dip, and (under offload) a tag map/unmap, so 64
    # rounds cap that at ~1.6% -- a k move swaps pre-built states of the
    # SAME draft head (~ms, KV continuous across k, no accept dip), so its
    # only flap cost is the swap itself and 16 rounds (~0.5 s) keep it
    # negligible while still damping EMA jitter.
    k_dwell_rounds: int = 16
    # Relative analytic-score margin a k move must clear (flap guard on
    # top of the dwell; the EMA inputs are smooth but not constant).
    k_switch_margin: float = 0.02
    # Round-time model fallback slope: with only ONE measured k, the cost
    # of one extra k is assumed slope * round_s(k_measured). 0.03 because
    # the marginal k adds one 1-layer MTP draft step + one verify token
    # against a 64-layer target forward (stage-1 k-ladder data: a few
    # percent per step at bs=1). Self-calibrating: the first k move
    # produces the second measured point and the linear fit takes over.
    k_time_slope_frac: float = 0.03
    # Per-position accept estimator: EMA weight and the samples a position
    # needs before its estimate is trusted (below it, the deepest trusted
    # position's value is extrapolated).
    p_ema_alpha: float = 0.05
    p_min_samples: int = 8
    # ---- Request-boundary exploration burst (task B). ------------------
    # 1 (default): at a new request's decode start the decision point is
    # pulled forward, the family dwell is waived once, and the staleness /
    # clearly-inferior probe gates are bypassed once -- so an eligible
    # DFLASH rung is tested EARLY on the new content instead of waiting
    # out probe_interval_rounds (up to 512 rounds > a whole 1024-token
    # request). 0: boundaries are ignored (legacy).
    boundary_burst: int = 1
    # A boundary burst never re-probes a rung probed this recently
    # (absolute rounds; guards against rapid tiny requests re-probing on
    # every turn).
    boundary_min_age_rounds: int = 64

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
        if self.k_dwell_rounds < 1:
            raise ValueError("cross bandit: k_dwell_rounds must be >= 1")
        if self.k_switch_margin < 0.0:
            raise ValueError("cross bandit: k_switch_margin must be >= 0")
        if self.k_time_slope_frac < 0.0:
            raise ValueError("cross bandit: k_time_slope_frac must be >= 0")
        if not (0.0 < self.p_ema_alpha <= 1.0):
            raise ValueError("cross bandit: p_ema_alpha must be in (0, 1]")
        if self.p_min_samples < 1:
            raise ValueError("cross bandit: p_min_samples must be >= 1")
        if self.boundary_min_age_rounds < 0:
            raise ValueError(
                "cross bandit: boundary_min_age_rounds must be >= 0"
            )


class NextnAcceptModel:
    """Analytic k selection within the NEXTN/MTP family (task A).

    Every NEXTN k rung drafts with the SAME MTP head: a k-rung round drafts
    k positions autoregressively and verify accepts a PREFIX of them. The
    per-position prefix accept probabilities
        p_i = P(position i accepted | positions 1..i-1 accepted)
    are therefore k-invariant -- position i's draft distribution is
    conditioned only on the accepted prefix, not on how many further
    positions the configured k would draft -- so results measured at the
    ACTIVE k predict every other k analytically and the per-k probe
    windows of the original bandit are unnecessary inside the family.

    ESTIMATOR: one verify result with accept count a under rung k updates
    positions 1..a with success and, when a < k, position a+1 with failure;
    positions beyond a+1 are unobserved (the chain broke earlier). Each
    position keeps a Bernoulli EMA. Positions with fewer than
    ``p_min_samples`` observations -- and every position beyond the current
    k, which is never observed at all -- extrapolate the DEEPEST trusted
    position's value (p_i = p_k for i > k, per the task spec; no invented
    decay tail, no invented optimism).

    EXPECTED YIELD per round at k (the +1 is the bonus token):
        E_accept(k) = 1 + sum_{i=1..k} prod_{j<=i} p_j

    ROUND-TIME MODEL: measured round_s EMA (RungMetrics) for every k that
    actually ran; unmeasured k use a linear model round_s(k) = a + b*k --
    the marginal k costs one extra 1-layer MTP draft step plus one extra
    verify token, both ~constant at fixed batch size, hence linear.
    Calibrated by least squares over the measured (k, round_s) points
    (slope clamped >= 0: a negative fit is measurement noise, draft steps
    do not have negative cost) when >= 2 ks are measured; with a single
    measured point the slope falls back to
    ``k_time_slope_frac * round_s(k0)`` (see the config field comment).
    Self-calibrating: the first analytic k move makes the moved-to k a
    measured point and the fit replaces the fallback.

    k* = argmax_k E_accept(k) / round_s(k) over the configured k arms.

    RANK UNIFORMITY: observe() consumes ONLY the rank-0-broadcast accept
    counts and is safe on every rank; round_s is rank-local WALL CLOCK, so
    score()/k_star() must stay rank-0-only + broadcast (design 4c),
    exactly like the bandit's measured rewards.
    """

    def __init__(
        self,
        ks: list,
        p_ema_alpha: float = 0.05,
        p_min_samples: int = 8,
        k_time_slope_frac: float = 0.03,
    ):
        self.ks = sorted(int(k) for k in ks)
        assert self.ks and self.ks[0] >= 1
        self._alpha = float(p_ema_alpha)
        self._min_samples = int(p_min_samples)
        self._slope_frac = float(k_time_slope_frac)
        # position (1-based) -> [ema, samples]
        self._p: dict = {}

    # -- estimation (any rank) ----------------------------------------
    def observe(self, k: int, num_correct_drafts_per_req: list) -> None:
        for a in num_correct_drafts_per_req:
            a = int(a)
            for i in range(1, min(a, k) + 1):
                self._update(i, 1.0)
            if a < k:
                self._update(a + 1, 0.0)

    def _update(self, pos: int, value: float) -> None:
        m = self._p.setdefault(pos, [0.0, 0])
        m[0] = value if m[1] == 0 else (
            (1 - self._alpha) * m[0] + self._alpha * value
        )
        m[1] += 1

    def p_prefix(self, kmax: int) -> Optional[list]:
        """[p_1..p_kmax] with extrapolation; None before position 1 is
        trusted (no usable accept data yet)."""
        ps = []
        last = None
        for i in range(1, kmax + 1):
            m = self._p.get(i)
            if m is not None and m[1] >= self._min_samples:
                last = m[0]
            if last is None:
                return None
            ps.append(last)
        return ps

    def e_accept(self, k: int) -> Optional[float]:
        ps = self.p_prefix(k)
        if ps is None:
            return None
        total, prefix = 1.0, 1.0
        for p in ps:
            prefix *= p
            total += prefix
        return total

    # -- round-time model (rank 0 only: wall clock) --------------------
    def _measured_points(self, metrics, min_time_samples: int) -> list:
        pts = []
        for k in self.ks:
            t = metrics.round_s(("nextn", k), min_time_samples)
            if t is not None:
                pts.append((k, t))
        return pts

    def round_s(self, k: int, metrics, min_time_samples: int = 3) -> Optional[float]:
        pts = self._measured_points(metrics, min_time_samples)
        for pk, pt in pts:
            if pk == k:
                return pt
        if not pts:
            return None
        if len(pts) == 1:
            k0, t0 = pts[0]
            return max(1e-6, t0 + self._slope_frac * t0 * (k - k0))
        n = len(pts)
        mean_k = sum(p[0] for p in pts) / n
        mean_t = sum(p[1] for p in pts) / n
        var = sum((p[0] - mean_k) ** 2 for p in pts)
        cov = sum((p[0] - mean_k) * (p[1] - mean_t) for p in pts)
        b = max(0.0, cov / var) if var > 0 else 0.0
        a = mean_t - b * mean_k
        return max(1e-6, a + b * k)

    def score(self, k: int, metrics, min_time_samples: int = 3) -> Optional[float]:
        e = self.e_accept(k)
        if e is None:
            return None
        t = self.round_s(k, metrics, min_time_samples)
        if t is None:
            return None
        return e / t

    def k_star(self, metrics, min_time_samples: int = 3):
        """(best k, {k: score}) over the configured arms; (None, {}) until
        both accept and time data exist."""
        smap = {}
        for k in self.ks:
            s = self.score(k, metrics, min_time_samples)
            if s is not None:
                smap[k] = s
        if not smap:
            return None, {}
        return max(smap, key=smap.get), smap

    def snapshot(self) -> dict:
        return {
            i: {"p": round(m[0], 3), "n": m[1]}
            for i, m in sorted(self._p.items())
        }


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
        self.initial: Rung = initial
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
        # Context-gate bookkeeping: rungs whose probe suppression was already
        # logged in the CURRENT contiguous ineligible phase (reset when the
        # rung becomes eligible again), and the eligibility set of the last
        # decide (trace only).
        self._gate_logged: set = set()
        self._gated_rungs: tuple = ()
        # Analytic k selection (task A): one shared accept model over the
        # NEXTN k arms. With analytic_k on, the k arms are NEVER probed --
        # the model predicts every k from the active k's accepts -- and k
        # moves are governed by k_dwell_rounds/k_switch_margin instead of
        # probe windows. The family decision (NEXTN(k*) vs DFLASH) stays
        # with the bandit.
        nextn_ks = sorted(r[1] for r in self.rungs if r[0] == "nextn")
        self.accept_model: Optional[NextnAcceptModel] = None
        if self.cfg.analytic_k and nextn_ks:
            self.accept_model = NextnAcceptModel(
                nextn_ks,
                p_ema_alpha=self.cfg.p_ema_alpha,
                p_min_samples=self.cfg.p_min_samples,
                k_time_slope_frac=self.cfg.k_time_slope_frac,
            )
        self._last_k_switch_round = 0
        # Trace-only mirrors of the last decide's analytic view.
        self._last_k_star = None
        self._last_k_scores: dict = {}
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
        record = self._attr_count > self.cfg.burn_in_rounds
        self.metrics.observe(
            rung,
            num_correct_drafts_per_req,
            batch_size,
            now=now,
            record=record,
        )
        # Task A: feed the per-position accept estimators from the same
        # rank-uniform accept stream, under the same burn-in gate.
        if record and self.accept_model is not None and rung[0] == "nextn":
            self.accept_model.observe(int(rung[1]), num_correct_drafts_per_req)

    def score(self, rung: Rung) -> Optional[float]:
        return self.metrics.reward(
            rung, min_time_samples=self.cfg.min_time_samples
        )

    def scores(self) -> dict:
        return {r: self.score(r) for r in self.rungs}

    # ------------------------------------------------------------------
    # Decision (rank 0 only)
    # ------------------------------------------------------------------
    def decide(
        self,
        round_idx: int,
        eligible: Optional[frozenset] = None,
        gate_note: str = "",
        request_boundary: bool = False,
    ) -> Rung:
        """Return the rung to run from the next round boundary on.

        Pure control flow over the rank-local metrics; returns the CURRENT
        rung when nothing warrants a change (the caller broadcasts the
        returned id unconditionally -- receiving one's own current rung is
        the no-decision case on every rank).

        ``eligible`` (context gate, T156): when given, rungs OUTSIDE the set
        are (a) never selected by the exploit argmax, (b) never probed --
        the probe suppression is the actual overhead saving -- and (c) the
        active rung is EVICTED when it falls out of the set, to the initial
        (boot) rung when that is eligible (overriding dwell: the gate is a
        hard eligibility criterion, not a preference). ``None`` (gate off)
        is byte-identical
        to the pre-gate behavior. ``gate_note`` is appended to gate log
        lines (the caller knows the context length; this class does not).

        ``request_boundary`` (task B): True when a new request entered
        decode since the last decision point. With ``boundary_burst`` on,
        this decide waives the family dwell once and relaxes the probe
        gates (staleness age -> boundary_min_age_rounds, inferior gate
        bypassed), so an eligible inactive DFLASH rung is tested EARLY on
        the fresh content. An in-flight probe window is never interrupted.

        With ``analytic_k`` on (task A), the NEXTN k arms carry ANALYTIC
        scores (NextnAcceptModel) in place of per-arm measurements, are
        never probed, and k moves inside the family run through the k-step
        below (k dwell + margin, no probe window). The bandit's remaining
        job is the FAMILY decision NEXTN(k*) vs DFLASH.
        """
        scores = self.scores()
        active = self.active
        self._last_active_round[active] = round_idx
        boundary = bool(request_boundary and self.cfg.boundary_burst)
        # Task A: overlay the NEXTN arms with analytic scores (E_accept(k)
        # / round_s(k)); one shared measurement stream prices every k.
        k_star, k_scores = None, {}
        if self.accept_model is not None:
            k_star, k_scores = self.accept_model.k_star(
                self.metrics, self.cfg.min_time_samples
            )
            for k, s in k_scores.items():
                scores[("nextn", k)] = s
        self._last_k_star, self._last_k_scores = k_star, k_scores
        if eligible is not None:
            self._gate_logged &= {r for r in self.rungs if r not in eligible}
            self._gated_rungs = tuple(
                rung_str(r) for r in self.rungs if r not in eligible
            )
        else:
            self._gate_logged.clear()
            self._gated_rungs = ()
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
            if eligible is not None and active not in eligible:
                # The probe rung fell out of eligibility mid-window (context
                # crossed the gate threshold): abort the probe and return to
                # the incumbent immediately -- its partial estimate is kept.
                incumbent = self._pre_probe_rung
                self._probe_until = None
                self._pre_probe_rung = None
                if incumbent is not None and (
                    eligible is None or incumbent in eligible
                ):
                    self._switch(incumbent, round_idx)
                    logger.info(
                        "cross bandit: probe of %s aborted: ctx gate%s; "
                        "returning to %s.",
                        rung_str(active),
                        gate_note,
                        rung_str(incumbent),
                    )
                    return self._emit(
                        round_idx, "probe_abort_gate", scores, incumbent
                    )
                # No usable incumbent: the probe state is cleared and the
                # (ineligible) probe rung stays active -- the gate-eviction
                # step below relocates it.
            elif round_idx < self._probe_until:
                return self._emit(round_idx, "probe_hold", scores, active)
            else:
                incumbent = self._pre_probe_rung
                self._probe_until = None
                self._pre_probe_rung = None
                probe_s = scores.get(active)
                inc_s = scores.get(incumbent)
                if probe_s is not None and (inc_s is None or probe_s > inc_s):
                    # Probe won; it is already active. Restart the dwell
                    # clock so the adoption is not immediately re-contested.
                    self._last_switch_round = round_idx
                    return self._emit(round_idx, "probe_adopt", scores, active)
                self._switch(incumbent, round_idx)
                return self._emit(round_idx, "probe_return", scores, incumbent)

        # 1b) Context-gate eviction: the ACTIVE rung is no longer eligible.
        #     Deliberately OVERRIDES the minimum dwell -- eligibility is a
        #     hard criterion (the drafter cannot usefully run past the gate),
        #     not a preference the anti-flap guards may defer.
        if eligible is not None and active not in eligible:
            elig = [r for r in self.rungs if r in eligible]
            if elig:
                # Eviction target: the INITIAL (boot/primary) rung, not the
                # frozen-score argmax. At eviction time every eligible rung's
                # score is FROZEN from the pre-crossing (short-context)
                # regime and predicts nothing about the regime the batch just
                # entered; argmax over those is arbitrary, and the optimistic
                # probe gate can then lock the arbitrary pick in (measured
                # 2026-07-22: frozen-argmax eviction landed on nextn:2 and
                # held it through the whole 10k/50k battery segment at
                # -12..-15% vs the nextn:3 control). Falling back to the
                # boot rung matches the single-algorithm default; argmax and
                # probing re-explore from there on FRESH measurements.
                if self.initial in eligible:
                    target = self.initial
                else:
                    measured = {
                        r: scores[r] for r in elig if scores[r] is not None
                    }
                    target = (
                        max(measured, key=measured.get) if measured else elig[0]
                    )
                self._switch(target, round_idx)
                logger.info(
                    "cross bandit: active rung %s evicted: ctx gate%s; -> %s.",
                    rung_str(active),
                    gate_note,
                    rung_str(target),
                )
                return self._emit(round_idx, "gate_evict", scores, target)
            # No eligible rung at all (misconfiguration): hold.
            return self._emit(round_idx, "gate_hold", scores, active)

        # 1c) Analytic k-step (task A): move WITHIN the NEXTN family to k*.
        #     Runs even during the family dwell -- a k move swaps pre-built
        #     states of the same draft head (~ms, no accept dip), so only
        #     the short k dwell + margin guard it. Does NOT touch the
        #     family dwell clock: residence in the FAMILY is what the
        #     family dwell prices.
        if (
            self.accept_model is not None
            and active[0] == "nextn"
            and k_star is not None
            and k_star != active[1]
            and (eligible is None or ("nextn", k_star) in eligible)
            and round_idx - self._last_k_switch_round
            >= self.cfg.k_dwell_rounds
        ):
            cur_ks = k_scores.get(active[1])
            new_ks = k_scores.get(k_star)
            if new_ks is not None and (
                cur_ks is None
                or new_ks > cur_ks * (1.0 + self.cfg.k_switch_margin)
            ):
                target = ("nextn", k_star)
                self.active = target
                self.switch_count += 1
                self._last_active_round[target] = round_idx
                self._last_k_switch_round = round_idx
                return self._emit(round_idx, "k_step", scores, target)

        # 2) Minimum dwell (waived once at a request boundary, task B).
        if (
            not boundary
            and round_idx - self._last_switch_round < self.cfg.min_dwell_rounds
        ):
            return self._emit(round_idx, "dwell_hold", scores, active)

        cur_s = scores.get(active)

        # 3) Exploit: argmax over measured ELIGIBLE rungs, deadzone-guarded.
        #    With analytic_k on, nextn-to-nextn moves belong to the k-step
        #    above (margin + k dwell, no deadzone) -- the exploit argmax
        #    only decides FAMILY changes then.
        measured = {
            r: s
            for r, s in scores.items()
            if s is not None and (eligible is None or r in eligible)
        }
        if measured and cur_s is not None:
            best = max(measured, key=measured.get)
            intra_family_k = (
                self.accept_model is not None
                and best[0] == "nextn"
                and active[0] == "nextn"
            )
            if (
                best != active
                and not intra_family_k
                and measured[best] > cur_s * (1.0 + self.cfg.deadzone)
            ):
                self._switch(best, round_idx)
                return self._emit(round_idx, "switch", scores, best)

        # 4) Explore: probe the best stale / never-measured inactive rung.
        cand = self._probe_candidate(
            scores, round_idx, eligible, gate_note, boundary=boundary
        )
        if cand is not None:
            self._pre_probe_rung = active
            self._probe_until = round_idx + self.cfg.probe_window_rounds
            self._switch(cand, round_idx)
            return self._emit(
                round_idx,
                "probe_start_boundary" if boundary else "probe_start",
                scores,
                cand,
            )

        return self._emit(round_idx, "hold", scores, active)

    def _probe_candidate(
        self,
        scores: dict,
        round_idx: int,
        eligible: Optional[frozenset] = None,
        gate_note: str = "",
        boundary: bool = False,
    ) -> Optional[Rung]:
        cur_s = scores.get(self.active)
        # Task B boundary burst: fresh content -> the frozen estimates of
        # inactive rungs predict little; probe with a much shorter
        # staleness age and without the inferior gate (once per boundary).
        interval = (
            self.cfg.boundary_min_age_rounds
            if boundary
            else self.cfg.probe_interval_rounds
        )
        unmeasured = []
        stale = []
        for r in self.rungs:
            if r == self.active:
                continue
            if self.accept_model is not None and r[0] == "nextn":
                # Task A: the k arms are priced analytically from the
                # shared accept stream -- probing them buys no information.
                continue
            gated = eligible is not None and r not in eligible
            s = scores.get(r)
            if s is None:
                # Cold start: gather data as soon as the dwell allows.
                if gated:
                    self._log_gate_skip(r, gate_note)
                    continue
                unmeasured.append(r)
                continue
            if (round_idx - self._last_active_round[r]) < interval:
                continue
            if gated:
                # Context gate: an ineligible rung is NEVER probed -- this
                # suppression (not the selection filter) is the overhead
                # saving: pre-gate, the stale DFLASH rung was probed
                # pointlessly on 50k contexts every probe interval. Logged
                # once per contiguous ineligible phase (only when the rung
                # would otherwise be a probe candidate: stale or unmeasured).
                self._log_gate_skip(r, gate_note)
                continue
            # Optimistic gate + priority key: the decayed best-ever score
            # (falls back to the frozen current score before any best is
            # recorded). A rung that has NEVER been competitive stops eating
            # probe windows; one that ever shone stays re-checkable.
            best = self._best_score.get(r, s)
            if (
                not boundary
                and cur_s is not None
                and best * self.cfg.probe_inferior_factor < cur_s
            ):
                continue
            stale.append((best, r))
        if unmeasured:
            return unmeasured[0]
        if stale:
            return max(stale, key=lambda t: t[0])[1]
        return None

    def _log_gate_skip(self, rung: Rung, gate_note: str) -> None:
        if rung in self._gate_logged:
            return
        self._gate_logged.add(rung)
        logger.info(
            "cross bandit: probe of %s skipped: ctx gate%s.",
            rung_str(rung),
            gate_note,
        )

    def note_switch_rejected(self, target: Rung, actual: Rung) -> None:
        """The worker refused to APPLY a decided switch (pre-swap OOM
        guard, first-boot convergence). Roll the decision state back to
        *actual* so future decisions reason from the rung that is really
        running; retry suppression lives in the worker. A probe window
        whose probe rung never physically started is cancelled."""
        if self.active != target:
            return
        self.active = actual
        if self._probe_until is not None and self._pre_probe_rung == actual:
            self._probe_until = None
            self._pre_probe_rung = None

    def _switch(self, target: Rung, round_idx: int) -> None:
        self.active = target
        self.switch_count += 1
        self._last_switch_round = round_idx
        self._last_active_round[target] = round_idx
        if target[0] == "nextn":
            # Entering the family (or returning to it) restarts the k-step
            # settling window too.
            self._last_k_switch_round = round_idx

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
            if self._gated_rungs:
                rec["gated"] = list(self._gated_rungs)
            if self.accept_model is not None:
                rec["k_star"] = self._last_k_star
                rec["k_scores"] = {
                    k: round(s, 2) for k, s in self._last_k_scores.items()
                }
                rec["p"] = self.accept_model.snapshot()
            try:
                with open(self._trace_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except OSError:
                logger.exception("cross bandit: trace write failed; disabling")
                self._trace_path = None
        return target
