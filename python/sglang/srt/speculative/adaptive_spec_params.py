"""Adaptive speculative decoding parameters.

Adjusts speculative_num_steps at runtime based on observed acceptance lengths.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
import time
from collections import deque
from functools import cached_property
from typing import TYPE_CHECKING

from sglang.srt.utils import log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

DEFAULT_ADAPTIVE_CONFIG: dict[str, dict] = {
    # Candidate ceiling is 3 (not 7): with per-position accept p <= 0.8 the
    # expected accepted chain length is sum_{i=1..k} p^i, so k=3 -> k=7 buys
    # < 1 extra accepted token for 4 extra draft forwards (net-negative), and
    # every extra candidate step costs a fully pre-captured graph set of VRAM.
    # Users with p >~ 0.85 workloads can raise the ceiling via
    # --speculative-adaptive-config.
    "1": {
        "candidate_steps": [1, 2, 3],
        "up_hysteresis": 0.0,
        "down_hysteresis": -0.25,
        "ceiling_coeff": 0,
    },
    "8": {
        "candidate_steps": [0, 1, 3],
        "up_hysteresis": 0.0,
        "down_hysteresis": 0.0,
        "ceiling_coeff": 0,
    },
    "32": {
        "candidate_steps": [0, 1],
        "up_hysteresis": 0.0,
        "down_hysteresis": 0.0,
        "ceiling_coeff": 0,
    },
    "64": {
        "candidate_steps": [0],
        "up_hysteresis": 0.0,
        "down_hysteresis": 0.0,
        "ceiling_coeff": 0,
    },
}

# Frozen-KV MTP default. Differs from DEFAULT_ADAPTIVE_CONFIG in two ways:
# 1. Every candidate step is >= 1: the frozen seed / draft-extend path has no
#    step-0 (nospec) branch, and FrozenKVMTPWorkerV2._assert_adaptive_supported
#    hard-rejects any config resolving to a step < 1. Using the generic default
#    (which contains step 0 in the bs>=8 slots) would crash at init.
# 2. The candidate ceiling is 3, not 7: at per-position accept p<=0.8 the
#    expected accepted chain length gained by k=3 -> k=7 is < 1 token for 4
#    extra draft forwards (net-negative), and each extra candidate costs a
#    full pre-captured graph set of VRAM.
FROZEN_MTP_DEFAULT_ADAPTIVE_CONFIG: dict[str, dict] = {
    "1": {
        "candidate_steps": [1, 2, 3],
        "up_hysteresis": 0.0,
        "down_hysteresis": -0.25,
        "ceiling_coeff": 0,
    },
    "8": {
        "candidate_steps": [1, 2],
        "up_hysteresis": 0.0,
        "down_hysteresis": 0.0,
        "ceiling_coeff": 0,
    },
    "32": {
        "candidate_steps": [1],
        "up_hysteresis": 0.0,
        "down_hysteresis": 0.0,
        "ceiling_coeff": 0,
    },
}


# Built-in profile for high-predictability workloads (per-position accept
# probability p >~ 0.85: code boilerplate, structured/tabular emission, ...).
# Adds k=4/5 ladder rungs: with graph-memory OFFLOAD (#93) an extra rung
# costs only its capture time at boot plus the max-state-sized alias pool --
# inactive rungs hold no physical VRAM -- so exposing them is nearly free.
# The rungs are climbed only on SUSTAINED high acceptance: up_hysteresis 0.25
# (T75 recommendation) raises every rise threshold, because flapping into
# k=4/5 wastes 4-5 serial draft forwards per rejected chain. No step-0 slots,
# so the profile is valid for EAGLE and FROZEN_KV_MTP alike (high-accept
# workloads keep drafting profitable even at bs 32). This profile is NOT the
# default: at p <= 0.8 the expected extra accepted chain length from k=3 ->
# k=5 is < 0.6 tokens for 2 extra serial draft forwards (net-negative, see
# DEFAULT_ADAPTIVE_CONFIG's ceiling rationale) -- select it explicitly via
# --speculative-adaptive-config high-accept.
HIGH_ACCEPT_ADAPTIVE_CONFIG: dict[str, dict] = {
    "1": {
        "candidate_steps": [1, 2, 3, 4, 5],
        "up_hysteresis": 0.25,
        "down_hysteresis": -0.25,
        "ceiling_coeff": 0,
    },
    "8": {
        "candidate_steps": [1, 2, 3],
        "up_hysteresis": 0.25,
        "down_hysteresis": 0.0,
        "ceiling_coeff": 0,
    },
    "32": {
        "candidate_steps": [1],
        "up_hysteresis": 0.0,
        "down_hysteresis": 0.0,
        "ceiling_coeff": 0,
    },
}

# --speculative-adaptive-config accepts these names instead of a JSON path.
# "default" resolves to the per-algorithm built-in default.
BUILTIN_ADAPTIVE_PROFILES: dict[str, dict[str, dict] | None] = {
    "default": None,
    "high-accept": HIGH_ACCEPT_ADAPTIVE_CONFIG,
}


def default_adaptive_config_for(algorithm: str | None) -> dict[str, dict]:
    """Pick the built-in adaptive config for *algorithm*.

    Used by every resolver call site (server_args buffer sizing, the
    speculative-hook param init, and the workers) so they cannot disagree
    about which default applies.
    """
    if algorithm == "FROZEN_KV_MTP":
        return FROZEN_MTP_DEFAULT_ADAPTIVE_CONFIG
    return DEFAULT_ADAPTIVE_CONFIG


def adaptive_unsupported_reason(server_args: ServerArgs) -> str | None:
    """Return why adaptive spec cannot run under the given server args, or None if supported."""
    from sglang.srt.arg_groups.overrides import resolved_view

    if server_args.speculative_algorithm not in ("EAGLE", "EAGLE3", "FROZEN_KV_MTP"):
        return (
            f"speculative_algorithm={server_args.speculative_algorithm} "
            "(only EAGLE/EAGLE3/FROZEN_KV_MTP are supported)"
        )
    if (
        server_args.speculative_eagle_topk is not None
        and server_args.speculative_eagle_topk != 1
    ):
        return (
            f"speculative_eagle_topk={server_args.speculative_eagle_topk} "
            "(only topk=1 is supported)"
        )
    if resolved_view(server_args).enable_dp_attention:
        return (
            "enable_dp_attention=True is not supported "
            "(adaptive tier decisions are not synchronized across DP ranks)"
        )
    if resolved_view(server_args).enable_multi_layer_eagle:
        return (
            "enable_multi_layer_eagle=True is not supported "
            "(MultiLayerEagleWorkerV2 does not implement adaptive)"
        )
    if server_args.enable_two_batch_overlap:
        return (
            "enable_two_batch_overlap=True is not supported "
            "(adaptive state swap would discard the TboAttnBackend wrapper)"
        )
    if server_args.enable_pdmux:
        return (
            "enable_pdmux=True is not supported "
            "(adaptive state swap does not update decode_attn_backend_group)"
        )
    return None


def _load_adaptive_config(
    cfg_path: str | None,
    algorithm: str | None = None,
) -> tuple[dict, dict[int, dict]]:
    """Load and validate adaptive config.

    *cfg_path* may be a JSON file path or a built-in profile name
    (``BUILTIN_ADAPTIVE_PROFILES``). Uses
    ``default_adaptive_config_for(algorithm)`` when it is ``None``.
    """
    if cfg_path is not None:
        if cfg_path in BUILTIN_ADAPTIVE_PROFILES:
            profile = BUILTIN_ADAPTIVE_PROFILES[cfg_path]
            cfg = (
                profile
                if profile is not None
                else default_adaptive_config_for(algorithm)
            )
        else:
            with open(cfg_path) as f:
                cfg = json.load(f)
    else:
        cfg = default_adaptive_config_for(algorithm)

    bs_entries: dict[int, dict] = {}
    for key, entry in cfg.items():
        if not key.isdigit():
            continue

        steps = entry.get("candidate_steps")
        if (
            not isinstance(steps, list)
            or not steps
            or not all(isinstance(s, int) and s >= 0 for s in steps)
        ):
            raise ValueError(
                f"BS {key}: candidate_steps must be a list of non-negative ints, "
                f"got {steps!r}"
            )
        bs_entries[int(key)] = entry

    if not bs_entries:
        raise ValueError(
            "speculative_adaptive_config must contain at least one integer-string "
            'BS key, e.g. {"1": {"candidate_steps": [1,2,3]}}. '
            f"Got keys: {list(cfg.keys())}"
        )
    return cfg, bs_entries


def resolve_candidate_steps_from_config(
    cfg_path: str | None = None,
    algorithm: str | None = None,
) -> list[int]:
    """Union of every BS slot's candidate steps; sizes the runtime buffers."""
    _, bs_entries = _load_adaptive_config(cfg_path, algorithm=algorithm)
    all_steps: set[int] = set()
    for entry in bs_entries.values():
        all_steps.update(entry["candidate_steps"])
    return sorted(all_steps)


class AdaptiveStepSlot:
    """Tracks acceptance over a short trailing window and adapts num_steps.

    The core idea: if drafts are consistently accepted, try more steps;
    if drafts are consistently rejected early, reduce steps to avoid waste.

    DECISION SIGNAL (T156 stage-1 controller repair): decisions read the MEAN
    of a short trailing WINDOW of per-round batch-average accept counts, not
    the server-lifetime EMA. The 2026-07-22 flap incident (12 switches / 18 s,
    median dwell 1.0 s, perfect 3<->2 alternation) showed the lifetime EMA is
    the wrong granularity: it averages code bursts (accept ~3.0-3.5) and prose
    lulls (~1.0) into a mid value (observed 0.93-1.80) that swings WIDER than
    the 0.25 hysteresis band on every update. A short window tracks the
    within-generation modality that actually drives throughput (r=0.901) and
    has a defined spread, which feeds the noise-scaled deadzone below.
    ``ema_accept_len`` is still maintained, but only as the observability
    gauge (``spec_ema_accept_len``); it no longer drives decisions.

    ANTI-FLAP GUARDS (both mandatory; each alone was insufficient):
    - Minimum dwell: after any applied switch, at least ``min_dwell_rounds``
      verify rounds must pass before the next switch is considered. Counted
      in ROUNDS, not wall-clock seconds: every rank replays this decision
      independently from the rank-0-broadcast accept counts (#50 invariant),
      and wall time is NOT rank-uniform — a seconds-based dwell would let
      ranks switch on different rounds and desynchronize CUDA graphs.
    - Noise-scaled deadzone: a switch must clear its threshold by
      ``deadzone_sigma`` standard errors of the window mean, on top of the
      configured up/down hysteresis. A noisy estimator (mixed workload)
      widens the effective deadzone automatically; a stable estimator keeps
      the controller responsive.

    The window is cleared on every applied switch: accept counts are capped
    by the active step count, so samples from the previous rung would bias
    the next decision. Refilling to ``window_size // 2`` before the next
    decision acts as a second, data-driven dwell.

    Only updates every `update_interval` batches; num_steps can be selected
    from different candidate sets on different batch_sizes.
    """

    def __init__(self, initial_steps: int, cfg: dict):
        candidates = sorted(set(cfg["candidate_steps"]))
        assert len(candidates) >= 1, "candidate_steps must have at least 1 value"
        self.candidate_steps = candidates

        self.ema_alpha = cfg.get("ema_alpha", 0.2)
        self.update_interval = cfg.get("update_interval", 5)
        self.warmup_batches = cfg.get("warmup_batches", 10)
        self.down_hysteresis = cfg.get("down_hysteresis", -0.25)
        self.up_hysteresis = cfg.get("up_hysteresis", 0.0)
        self.ceiling_coeff = cfg.get("ceiling_coeff", 0)

        # Stage-1 anti-flap knobs (see class docstring). window_size=16 is
        # ~0.5 s of bs=1 decode rounds: long enough to smooth single-round
        # noise, short enough to follow a code<->prose modality change.
        # min_dwell_rounds=64 is ~2 s at the observed ~30 rounds/s, i.e. the
        # design target dwell, expressed rank-uniformly in rounds.
        self.window_size = int(cfg.get("window_size", 16))
        self.min_dwell_rounds = int(cfg.get("min_dwell_rounds", 64))
        self.deadzone_sigma = float(cfg.get("deadzone_sigma", 1.0))
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")
        if self.min_dwell_rounds < 0:
            raise ValueError(
                f"min_dwell_rounds must be >= 0, got {self.min_dwell_rounds}"
            )
        if self.deadzone_sigma < 0:
            raise ValueError(
                f"deadzone_sigma must be >= 0, got {self.deadzone_sigma}"
            )

        if initial_steps in self.candidate_steps:
            self.current_steps = initial_steps
        else:
            self.current_steps = self.candidate_steps[len(self.candidate_steps) // 2]

        # Observability gauge only (spec_ema_accept_len); not a decision input.
        # Initialize at current steps - 1 (neutral starting point).
        self.ema_accept_len = float(self.current_steps - 1)
        self._batch_count = 0
        # Per-round batch-average accept counts; the decision estimator.
        self._accept_window: deque[float] = deque(maxlen=self.window_size)
        # Start dwell "expired" so the first decision after warmup is not
        # additionally delayed (warmup_batches already covers startup).
        self._rounds_since_switch = self.min_dwell_rounds

    def update(self, num_correct_drafts_per_req: list[int]) -> bool:
        """Feed observed accept lengths. Returns True if params changed.

        Args:
            num_correct_drafts_per_req: Per-request accepted draft token counts from last verify.
        """
        if not num_correct_drafts_per_req:
            return False

        if self.current_steps > 0:
            batch_avg = sum(num_correct_drafts_per_req) / len(
                num_correct_drafts_per_req
            )
            self.ema_accept_len = (
                1 - self.ema_alpha
            ) * self.ema_accept_len + self.ema_alpha * batch_avg
            self._accept_window.append(batch_avg)

        self._batch_count += 1
        self._rounds_since_switch += 1
        if self._batch_count <= self.warmup_batches:
            return False

        if (self._batch_count - self.warmup_batches) % self.update_interval != 0:
            return False

        # Minimum dwell: hard rank-uniform flap limiter (class docstring).
        if self._rounds_since_switch < self.min_dwell_rounds:
            return False

        # After a switch (window cleared) require the window to be at least
        # half full again, so the decision rests on data from the ACTIVE rung.
        # Zero-step intervals collect no accept data; the probe path below
        # must stay reachable, so the fill gate only applies while drafting.
        if self.current_steps > 0 and len(self._accept_window) < max(
            1, self.window_size // 2
        ):
            return False

        return self._recompute_params()

    def _window_mean_and_margin(self) -> tuple[float, float]:
        """Window mean and the noise margin the deadzone scales with.

        The margin is ``deadzone_sigma`` standard errors of the window mean:
        switch thresholds must be cleared by more than the estimator's own
        statistical noise, otherwise a mixed workload whose per-round accept
        counts straddle a threshold (the observed 3<->2 flap) re-crosses it
        on every update. Pure Python on rank-uniform inputs in identical
        order -> bit-identical on every rank.
        """
        window = self._accept_window
        n = len(window)
        mean = sum(window) / n
        if n < 2 or self.deadzone_sigma == 0.0:
            return mean, 0.0
        var = sum((x - mean) ** 2 for x in window) / (n - 1)
        stderr = math.sqrt(var / n)
        return mean, self.deadzone_sigma * stderr

    def _recompute_params(self) -> bool:
        """Recompute steps from the windowed estimator. Returns True if changed."""
        old_steps = self.current_steps
        current_idx = self.candidate_steps.index(old_steps)
        old_idx = current_idx

        # Probe the smallest positive step after a zero-step nospec interval.
        if old_steps == 0:
            current_idx = min(current_idx + 1, len(self.candidate_steps) - 1)
            target = self.candidate_steps[current_idx]
            if target > 0 and self.ema_accept_len < 0:
                # A slot initialized at steps=0 has no draft acceptance history;
                # start the first positive-step probe from that step's neutral EMA.
                self.ema_accept_len = float(target - 1)
            return self._apply_target_steps(old_steps, target)

        accept_est, noise_margin = self._window_mean_and_margin()

        # TODO: Consider limiting step changes to avoid overshooting.
        while current_idx > 0:
            prev_step = self.candidate_steps[current_idx - 1]
            # A zero-step candidate disables drafting. Treat zero accepted drafts
            # as low enough to reach it when it is the floor candidate.
            drop_threshold = 0.5 if prev_step == 0 else prev_step - 0.5
            drop_threshold += self.down_hysteresis - noise_margin
            if accept_est <= drop_threshold:
                current_idx -= 1
            else:
                break

        moved_down = current_idx < old_idx
        if not moved_down:
            while current_idx < len(self.candidate_steps) - 1:
                current_step = self.candidate_steps[current_idx]
                rise_threshold = current_step - 0.5 + self.up_hysteresis + noise_margin
                if accept_est > rise_threshold:
                    current_idx += 1
                else:
                    break

        target = self.candidate_steps[current_idx]
        # Estimator ceiling: only caps downward — never blocks step-ups, so
        # the system can explore higher steps and let the estimator catch up.
        if self.ceiling_coeff > 0:
            ceiling = max(1, math.ceil(accept_est * self.ceiling_coeff))
            if target > ceiling and target <= old_steps:
                while current_idx > 0 and self.candidate_steps[current_idx] > ceiling:
                    current_idx -= 1
                target = self.candidate_steps[current_idx]

        return self._apply_target_steps(old_steps, target)

    def _apply_target_steps(self, old_steps: int, target: int) -> bool:
        if target != old_steps:
            self.current_steps = target
            window_repr = (
                f"window_mean={sum(self._accept_window) / len(self._accept_window):.2f} "
                f"(n={len(self._accept_window)})"
                if self._accept_window
                else "window empty"
            )
            log_info_on_rank0(
                logger,
                f"Adaptive spec params updated: steps {old_steps} -> {target} "
                f"({window_repr}, ema_accept_len={self.ema_accept_len:.2f}, "
                f"dwell={self._rounds_since_switch} rounds)",
            )
            # Accept counts are capped by the step count, so samples from the
            # outgoing rung would bias the next decision — start fresh.
            self._accept_window.clear()
            self._rounds_since_switch = 0
            return True
        return False


class RungMetrics:
    """Live per-rung reward measurements (T156 stage 4).

    Tracks, per rung key, an EMA of
    - accepted tokens per verify round (accept_len incl. the bonus token), and
    - decode-round wall time attributed to that rung,
    i.e. the two factors of the stage-4 bandit objective
    ``reward(r) = E[accepted_tokens_per_verify(r)] / E[round_seconds(r)]``.

    The rung key is any hashable: the stage-1 k-ladder feeds plain step ints,
    the stage-4 cross-algorithm bandit feeds ``(algo, k)`` tuples. Keys must
    be homogeneous per instance (snapshot() sorts them).

    DETERMINISM WARNING (#50): ``round_s_ema`` is per-rank WALL CLOCK and is
    NOT rank-uniform. In the stage-1 k-ladder it is OBSERVABILITY-ONLY and
    must never feed the step decision — every rank recomputes that decision
    independently from the rank-0-broadcast accept counts, and a
    timing-dependent input would make the argmax diverge across ranks (graph
    desync -> NCCL hang). The stage-4 bandit consumes ``reward()`` on RANK 0
    ONLY and broadcasts the chosen rung id (design 4c, cross_algo_worker).
    """

    # Slow EMA: rung residence is dwell-limited (>= ~2 s), so per-rung samples
    # arrive in bursts; a small alpha keeps the estimate stable across visits.
    EMA_ALPHA = 0.05
    # Deltas above this are scheduler idle gaps (decode rounds are ~10-50 ms),
    # not round cost; feeding them would poison the duration estimate.
    IDLE_CUTOFF_S = 1.0
    # Rank-0 INFO snapshot every N observed rounds (~1 min at 30 rounds/s).
    LOG_EVERY_ROUNDS = 2000

    def __init__(self):
        # rung key -> [accept_len_ema, accept_samples, round_s_ema, time_samples]
        self._per_rung: dict = {}
        self._last_ts: float | None = None
        self._last_batch_size: int | None = None
        self._round_count = 0

    def observe(
        self,
        steps,
        num_correct_drafts_per_req: list[int],
        batch_size: int,
        now: float | None = None,
        record: bool = True,
    ) -> None:
        """Attribute one verify result (and its round duration) to *steps*.

        *steps* must be the rung that PRODUCED the result (the caller derives
        it from the result's draft-token stride), not the currently active
        one: under overlap scheduling the result is processed after the
        worker may already have switched rungs.

        ``record=False`` (stage-4 burn-in: the first rounds after a rung
        switch are systematically atypical) advances the round-duration
        timestamp chain WITHOUT updating any EMA — skipping the call entirely
        would make the next recorded duration span multiple rounds.
        """
        if not num_correct_drafts_per_req:
            return
        now = time.monotonic() if now is None else now
        if not record:
            self._last_ts = now
            self._last_batch_size = batch_size
            return
        m = self._per_rung.setdefault(steps, [0.0, 0, 0.0, 0])

        accept_len = 1.0 + sum(num_correct_drafts_per_req) / len(
            num_correct_drafts_per_req
        )
        m[0] = accept_len if m[1] == 0 else (
            (1 - self.EMA_ALPHA) * m[0] + self.EMA_ALPHA * accept_len
        )
        m[1] += 1

        # Round duration = gap to the previous verify completion on this rank.
        # Only comparable when the batch size did not change in between, and
        # only meaningful when decode rounds are back-to-back (idle cutoff).
        if self._last_ts is not None and batch_size == self._last_batch_size:
            dt = now - self._last_ts
            if 0.0 < dt < self.IDLE_CUTOFF_S:
                m[2] = dt if m[3] == 0 else (
                    (1 - self.EMA_ALPHA) * m[2] + self.EMA_ALPHA * dt
                )
                m[3] += 1
        self._last_ts = now
        self._last_batch_size = batch_size

        self._round_count += 1
        if self._round_count % self.LOG_EVERY_ROUNDS == 0:
            log_info_on_rank0(logger, f"Adaptive rung metrics: {self.snapshot()}")

    def reward(self, steps, min_time_samples: int = 3) -> float | None:
        """The stage-4 objective for one rung:
        EMA[accepted tokens per verify round] / EMA[round seconds].

        None until the rung has accept data AND at least *min_time_samples*
        round-duration samples (a single dt gap is too noisy to rank on).
        RANK-LOCAL wall clock — consume on rank 0 only (see class docstring).
        """
        m = self._per_rung.get(steps)
        if m is None or m[1] == 0 or m[3] < min_time_samples or m[2] <= 0.0:
            return None
        return m[0] / m[2]

    def round_s(self, steps, min_time_samples: int = 1) -> float | None:
        """EMA round duration of one rung, or None below *min_time_samples*.
        RANK-LOCAL wall clock -- same consumption rule as reward()."""
        m = self._per_rung.get(steps)
        if m is None or m[3] < min_time_samples or m[2] <= 0.0:
            return None
        return m[2]

    def snapshot(self) -> dict:
        """Per-rung metric snapshot for logging / the stage-4 objective."""
        return {
            steps: {
                "accept_len_ema": round(m[0], 3),
                "accept_samples": m[1],
                "round_s_ema": round(m[2], 5),
                "time_samples": m[3],
            }
            for steps, m in sorted(self._per_rung.items())
        }


class AdaptiveSpeculativeParams:
    """Routes ``batch_size`` to the correct per-BS slot.

    A slot is a per-BS configuration of adaptive step selection.
    """

    def __init__(
        self,
        initial_steps: int,
        cfg_path: str | None = None,
        algorithm: str | None = None,
    ):
        cfg, bs_entries = _load_adaptive_config(cfg_path, algorithm=algorithm)
        self._bs_list: list[int] = sorted(bs_entries)
        self._slots: dict[int, AdaptiveStepSlot] = {}
        self._cuda_graph_bs: list[int] | None = None

        # BS-axis debounce: the EMA axis has hysteresis, but without this the
        # slot choice follows batch_size instantly, so a concurrency level
        # oscillating across a slot boundary (e.g. 4 streams flapping around a
        # CUDA-graph bs edge) would thrash graph swaps every decode step. A
        # slot switch on the bs axis therefore requires `bs_debounce`
        # CONSECUTIVE decode steps routed to the new slot; any step routed back
        # to the active slot resets the run. Driven purely by the batch_size
        # sequence (identical on all TP/DCP ranks), so it is rank-deterministic.
        # bs_debounce=1 restores instant switching.
        self._bs_debounce = int(cfg.get("bs_debounce", 3))
        if self._bs_debounce < 1:
            raise ValueError(f"bs_debounce must be >= 1, got {self._bs_debounce}")
        self._active_slot_bs: int | None = None
        self._pending_slot_bs: int | None = None
        self._pending_slot_count = 0

        # Per-rung reward measurement (stage-4 bandit preparation).
        # Observability-only in stage 1 — see the RungMetrics docstring.
        self.rung_metrics = RungMetrics()

        for bs, entry in sorted(bs_entries.items()):
            self._slots[bs] = AdaptiveStepSlot(
                initial_steps=initial_steps,
                cfg={**cfg, **entry},
            )

        first_slot = self._slots[self._bs_list[0]]
        log_info_on_rank0(
            logger,
            f"AdaptiveSpeculativeParams initialized: "
            f"steps={first_slot.current_steps}, "
            f"candidate_steps={first_slot.candidate_steps}",
        )

    @cached_property
    def candidate_steps(self) -> list[int]:
        """Union of all BS slots' candidate steps."""
        return sorted({s for p in self._slots.values() for s in p.candidate_steps})

    def set_cuda_graph_bs(self, cuda_graph_bs: list[int] | None) -> None:
        self._cuda_graph_bs = sorted(cuda_graph_bs) if cuda_graph_bs else None

    def get_steps_for_batch(self, batch_size: int) -> int:
        """Step count for this decode batch; advances the bs-axis debounce."""
        return self._slots[
            self._debounced_slot_bs(batch_size, advance=True)
        ].current_steps

    def current_ema_accept_len(self) -> float:
        """EMA accept length of the currently ACTIVE (debounced) BS slot.

        This is the smoothed per-step signal the controller actually uses to
        decide when to raise/lower ``num_steps`` (see AdaptiveStepSlot.update).
        Exposed read-only for observability (``spec_ema_accept_len`` gauge / the
        live monitoring widget); it never advances the debounce or mutates
        state, so reading it is side-effect free and safe from any thread.
        """
        bs = self._active_slot_bs
        if bs is None or bs not in self._slots:
            bs = self._bs_list[0]
        return float(self._slots[bs].ema_accept_len)

    def on_verify_complete(
        self, num_correct_drafts_per_req: list[int], batch_size: int
    ) -> int | None:
        """Feed verify results to the ACTIVE (debounced) BS slot's EMA.

        Uses the debounced slot, not the raw-routed one: the batch actually ran
        with the active slot's step count, so its accept lengths belong to that
        slot's EMA. Does not advance the debounce (only the per-decode-step
        activation in ``get_steps_for_batch`` does).

        Returns the new step if a switch is warranted, else ``None``.
        """
        params = self._slots[self._debounced_slot_bs(batch_size, advance=False)]
        if params.update(num_correct_drafts_per_req):
            return params.current_steps
        return None

    def note_verify_observation(
        self,
        steps: int,
        num_correct_drafts_per_req: list[int],
        batch_size: int,
    ) -> None:
        """Record per-rung reward measurements (accept length + round time).

        Observability / stage-4 preparation only: never consulted by the step
        decision in stage 1 (its duration term is rank-local wall clock — see
        the RungMetrics determinism warning).
        """
        self.rung_metrics.observe(steps, num_correct_drafts_per_req, batch_size)

    def cuda_graph_bs_for_step(self, step: int) -> list[int] | None:
        """Return cuda_graph_bs values that can reach *step* at runtime.

        Returns ``None`` when CUDA graphs are disabled (``set_cuda_graph_bs``
        was never called or was called with ``None``).
        """
        if self._cuda_graph_bs is None:
            return None
        return [
            v
            for v in self._cuda_graph_bs
            if step in self._slots[self._find_closest_bs(v)].candidate_steps
        ]

    def _route(self, batch_size: int) -> AdaptiveStepSlot:
        """Raw (un-debounced) map: *batch_size* → pad to CUDA-graph BS → slot."""
        return self._slots[self._slot_bs_for(batch_size)]

    def _slot_bs_for(self, batch_size: int) -> int:
        return self._find_closest_bs(self._pad_to_cuda_graph_bs(batch_size))

    def _debounced_slot_bs(self, batch_size: int, advance: bool) -> int:
        """Slot key for *batch_size* under the bs-axis debounce.

        With ``advance=True`` (one call per decode step) a differing target
        slot must be seen ``bs_debounce`` consecutive times before it becomes
        active; ``advance=False`` only reads the currently active slot.
        """
        target = self._slot_bs_for(batch_size)
        if self._active_slot_bs is None or self._bs_debounce <= 1:
            self._active_slot_bs = target
            return target
        if target == self._active_slot_bs:
            if advance:
                # Any step back in the active slot resets the pending run.
                self._pending_slot_bs = None
                self._pending_slot_count = 0
            return target
        if advance:
            if target == self._pending_slot_bs:
                self._pending_slot_count += 1
            else:
                self._pending_slot_bs = target
                self._pending_slot_count = 1
            if self._pending_slot_count >= self._bs_debounce:
                log_info_on_rank0(
                    logger,
                    f"Adaptive bs-slot switch: {self._active_slot_bs} -> {target} "
                    f"after {self._pending_slot_count} consecutive decode steps",
                )
                self._active_slot_bs = target
                self._pending_slot_bs = None
                self._pending_slot_count = 0
                return target
        return self._active_slot_bs

    def _pad_to_cuda_graph_bs(self, batch_size: int) -> int:
        if self._cuda_graph_bs is None:
            return batch_size
        idx = bisect.bisect_left(self._cuda_graph_bs, batch_size)
        return (
            self._cuda_graph_bs[idx] if idx < len(self._cuda_graph_bs) else batch_size
        )

    def _find_closest_bs(self, target: int) -> int:
        idx = bisect.bisect_right(self._bs_list, target) - 1
        return self._bs_list[max(0, idx)]
