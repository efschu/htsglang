"""Cross-algorithm lazy single-capture + measured ctx retirement (T156 #156-4).

Pure, CPU-testable control logic for three coupled optimizations of the
cross-algo (NEXTN <-> DFLASH) meta worker. NOTHING here touches CUDA, the
model runner, or any collective -- ``cross_algo_worker`` owns the plumbing,
this module owns the decisions, so the whole state machine is unit-testable
without a GPU.

THE PROBLEM (measured, RESULT_draft_crossover.md 2026-07-24):
pure DFLASH runs 138 tok/s on code below ctx 4096, but the cross-algo worker
with DFLASH active only reaches 116-120 -- a standing ~13-15% "switching
tax". It is not the drafter: it is the SHARED TARGET producing BOTH hidden
state variants on every single round (aux concat for DFLASH, final hidden
for the NEXTN/MTP draft) plus the per-round warm-keep of whichever rung is
idle, purely so that a switch could happen at any moment.

THE THREE PIECES

1. ``RetirePolicy`` / ``retire_verdict`` -- MEASURED, MONOTONE ctx retirement
   of the DFLASH rung. ctx only ever grows inside a session, so a DFLASH
   acceptance collapse that is CONTEXT-driven can never heal: retire the rung
   permanently, stop probing it, drop its capture -- from then on zero dual
   tax. The trap is that low DFLASH acceptance is AMBIGUOUS: on prose DFLASH
   is bad at EVERY context length (content-driven, temporary, reversible).
   The rule therefore separates the two by WHERE the low acceptance happens:
   below the collapse band it counts as content (keep DFLASH), inside the
   band it counts as ctx collapse (retire permanently).

2. ``LazyCaptureController`` -- lazy single capture. Steady state runs only
   the ACTIVE rung's capture (and no warm-keep at all); the dual/eager
   window is entered only around an actual probe.

3. The same controller's signal gate -- signal-driven instead of periodic
   probing. NEXTN's own accept-length EMA is a FREE content signal (high =
   structured/code = a DFLASH probe is plausible; low = prose = DFLASH never
   wins there, do not pay the dual tax to find out again).

RANK UNIFORMITY (the NCCL-hang invariant, hard requirement):
every input consumed here is rank-uniform -- round counters, CPU-side
context lengths, and ``accept_len_ema`` (which is fed exclusively from the
rank-0-broadcast accept counts, #50). Wall-clock (``round_s``/``reward``)
is deliberately ABSENT from this module. Therefore every rank runs the same
transitions in the same round and reaches the same collectives. The single
wall-clock-dependent choice in the whole design -- "did the probe win?" --
is NOT made here: the controller only reports that a decision is due, rank 0
resolves it, and the answer arrives through the EXISTING rung-id broadcast
and is fed back via :meth:`LazyCaptureController.commit`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Part 1: measured, monotone ctx retirement of the DFLASH rung.
# ---------------------------------------------------------------------------

RETIRE_CTX_FACTOR_ENV = "SGLANG_CROSS_RETIRE_CTX_FACTOR"
RETIRE_BAND_FRAC_ENV = "SGLANG_CROSS_RETIRE_BAND_FRAC"
RETIRE_ACCEPT_RATIO_ENV = "SGLANG_CROSS_RETIRE_ACCEPT_RATIO"

# Fraction of the collapse ctx below which a low DFLASH acceptance is read as
# CONTENT (prose), not as ctx collapse -- see the module docstring.
DEFAULT_RETIRE_BAND_FRAC = 0.8
# DFLASH is considered collapsed once its accept EMA drops below this
# multiple of the NEXTN accept EMA. Relative, not absolute: the absolute
# accept level is model- and content-dependent, the RATIO to the baseline
# rung is what says "DFLASH stopped being worth its block size here".
DEFAULT_RETIRE_ACCEPT_RATIO = 1.0

# NOTE -- NO DEFAULT COLLAPSE CTX EXISTS ON PURPOSE.
# The collapse point of the z-lab DFLASH drafter is NOT MEASURED yet: the
# crossover suite (RESULT_draft_crossover.md) only reached ctx 6000, where
# DFLASH was still ahead; the collapse is merely SUSPECTED around 8-10k. A
# baked-in guess (4096 from the old policy table, or 8000 from the guess)
# would silently retire a rung that still wins, or keep one that no longer
# does. So: unset == disabled == exactly today's behavior, and 'auto'
# requires an explicitly configured factor. Fill this in from
# RESULT_dflash_ctx_curve.md once that measurement lands.


@dataclass(frozen=True)
class RetirePolicy:
    """Resolved retirement parameters. ``collapse_ctx is None`` == disabled
    == today's behavior (no retirement path is ever reached)."""

    collapse_ctx: Optional[int] = None
    band_frac: float = DEFAULT_RETIRE_BAND_FRAC
    accept_ratio: float = DEFAULT_RETIRE_ACCEPT_RATIO
    source: str = "off"

    @property
    def enabled(self) -> bool:
        return self.collapse_ctx is not None

    @property
    def band_low(self) -> Optional[int]:
        if self.collapse_ctx is None:
            return None
        return int(self.band_frac * self.collapse_ctx)

    def to_stash(self) -> Dict[str, Any]:
        return {
            "collapse_ctx": self.collapse_ctx,
            "band_frac": self.band_frac,
            "accept_ratio": self.accept_ratio,
            "source": self.source,
        }

    @classmethod
    def from_stash(cls, stash: Optional[Dict[str, Any]]) -> "RetirePolicy":
        if not stash:
            return cls()
        return cls(
            collapse_ctx=(
                None
                if stash.get("collapse_ctx") is None
                else int(stash["collapse_ctx"])
            ),
            band_frac=float(stash.get("band_frac", DEFAULT_RETIRE_BAND_FRAC)),
            accept_ratio=float(
                stash.get("accept_ratio", DEFAULT_RETIRE_ACCEPT_RATIO)
            ),
            source=str(stash.get("source", "off")),
        )


# Verdict reasons (stable strings: logged, traced and asserted in tests).
RETIRE_DISABLED = "disabled"
RETIRE_KEEP_CONTENT = "keep_content_band"
RETIRE_KEEP_MEASURED = "keep_measured_ok"
RETIRE_KEEP_NO_DATA = "keep_no_data"
RETIRE_CTX_COLLAPSE = "retire_ctx_collapse"
RETIRE_CTX_HARD = "retire_ctx_beyond_collapse"


def retire_verdict(
    max_ctx: int,
    dflash_accept: Optional[float],
    nextn_accept: Optional[float],
    policy: RetirePolicy,
) -> Tuple[bool, str]:
    """Should the DFLASH rung be retired PERMANENTLY for this session?

    Pure function of rank-uniform inputs (batch max context + the two accept
    EMAs). Returns ``(retire_now, reason)``.

    Three regions, by where the batch context sits:

    * ``ctx < band_low``  -- low DFLASH acceptance here is a CONTENT effect
      (DFLASH is weak on prose at every length). NEVER retire: a prose
      passage must not kill DFLASH for the code that may follow.
    * ``band_low <= ctx < collapse_ctx`` -- the approach to the measured
      collapse. Retire only on a MEASURED relative collapse against the
      NEXTN baseline (both EMAs present, dflash < ratio * nextn). Without
      data: keep.
    * ``ctx >= collapse_ctx`` -- at/past the measured collapse point the
      rung is known dead; retire unconditionally (no accept data needed).

    Monotonicity is a property of the CALLER (the flag is latched and never
    cleared) plus of ctx itself, which only grows within a session.
    """
    if not policy.enabled:
        return False, RETIRE_DISABLED
    ctx = int(max_ctx)
    collapse = int(policy.collapse_ctx)
    if ctx >= collapse:
        return True, RETIRE_CTX_HARD
    band_low = policy.band_low
    if band_low is None or ctx < band_low:
        return False, RETIRE_KEEP_CONTENT
    if dflash_accept is None or nextn_accept is None:
        return False, RETIRE_KEEP_NO_DATA
    if dflash_accept < policy.accept_ratio * nextn_accept:
        return True, RETIRE_CTX_COLLAPSE
    return False, RETIRE_KEEP_MEASURED


def parse_retire_ctx_value(raw: Optional[str]):
    """Parse ``--speculative-cross-algorithm-retire-ctx``:
    ``'off'`` (default) | ``'auto'`` | positive token count."""
    if raw is None:
        return "off"
    val = str(raw).strip().lower()
    if val in ("auto", "off"):
        return val
    try:
        tokens = int(val)
    except ValueError:
        tokens = -1
    if tokens < 1:
        raise ValueError(
            "--speculative-cross-algorithm-retire-ctx must be 'off', 'auto', "
            f"or a positive token count; got {raw!r}."
        )
    return tokens


def resolve_retire_policy(
    raw: Optional[str], sliding_window: Optional[int]
) -> RetirePolicy:
    """Resolve the retirement policy once at argument time (rank-uniform by
    construction: it runs in the launcher process and travels to every
    scheduler inside the pickled cross-shapes stash).

    ``'auto'`` derives ``collapse_ctx = factor * sliding_window`` from the
    DFLASH drafter's own structural horizon -- but ONLY when the factor is
    explicitly configured via ``SGLANG_CROSS_RETIRE_CTX_FACTOR``. There is
    deliberately no default factor: the collapse point is unmeasured (see
    the module comment), and 'auto' without a measurement would be a guess
    dressed up as a derivation. Unresolvable 'auto' degrades to DISABLED
    (today's behavior) with a loud log line, never to a guess.
    """
    parsed = parse_retire_ctx_value(raw)
    band_frac = float(os.environ.get(RETIRE_BAND_FRAC_ENV, DEFAULT_RETIRE_BAND_FRAC))
    if not (0.0 < band_frac <= 1.0):
        raise ValueError(f"{RETIRE_BAND_FRAC_ENV} must be in (0, 1]; got {band_frac}.")
    accept_ratio = float(
        os.environ.get(RETIRE_ACCEPT_RATIO_ENV, DEFAULT_RETIRE_ACCEPT_RATIO)
    )
    if accept_ratio <= 0.0:
        raise ValueError(
            f"{RETIRE_ACCEPT_RATIO_ENV} must be > 0; got {accept_ratio}."
        )
    if parsed == "off":
        return RetirePolicy(source="off (default: no DFLASH retirement)")
    if parsed == "auto":
        raw_factor = os.environ.get(RETIRE_CTX_FACTOR_ENV)
        if raw_factor is None:
            return RetirePolicy(
                source=(
                    "auto requested but the collapse ctx is UNMEASURED: set "
                    f"{RETIRE_CTX_FACTOR_ENV} (or pass an explicit token "
                    "count) once the ctx curve is measured -- retirement "
                    "stays DISABLED (today's behavior)"
                )
            )
        factor = float(raw_factor)
        if factor <= 0:
            raise ValueError(f"{RETIRE_CTX_FACTOR_ENV} must be > 0; got {factor}.")
        if not sliding_window:
            return RetirePolicy(
                source=(
                    "auto: the DFLASH drafter declares no sliding_window, so "
                    "no structural horizon to scale -- retirement DISABLED"
                )
            )
        return RetirePolicy(
            collapse_ctx=int(factor * int(sliding_window)),
            band_frac=band_frac,
            accept_ratio=accept_ratio,
            source=f"auto: {factor:g} * sliding_window {int(sliding_window)}",
        )
    return RetirePolicy(
        collapse_ctx=int(parsed),
        band_frac=band_frac,
        accept_ratio=accept_ratio,
        source="explicit --speculative-cross-algorithm-retire-ctx",
    )


# ---------------------------------------------------------------------------
# Part 2+3: lazy single capture with signal-driven probing.
# ---------------------------------------------------------------------------

LAZY_ENV_PREFIX = "SGLANG_CROSS_LAZY_"

# Phases of the capture state machine.
PHASE_STEADY = "steady"
PHASE_WARMUP = "warmup"
PHASE_MEASURE = "measure"

# Decision kinds handed to the worker (each one means: a rank-0 decision is
# due THIS round boundary and its answer must come back through commit()).
DECIDE_STEADY_K = "steady_k"  # k move inside the adopted NEXTN family
DECIDE_ENTER_MEASURE = "enter_measure"  # warm-up done -> run the candidate
DECIDE_END_MEASURE = "end_measure"  # window done -> adopt or return
DECIDE_RETIRE_EVICT = "retire_evict"  # DFLASH retired while it was adopted


@dataclass(frozen=True)
class LazyCaptureConfig:
    """Tunables; every field is overridable via ``SGLANG_CROSS_LAZY_<FIELD>``.

    The three round-count knobs deliberately MIRROR the existing bandit
    knobs (``probe_window_rounds``, ``probe_interval_rounds``,
    ``min_dwell_rounds``) rather than inventing a parallel vocabulary --
    the user-facing schema ("probe briefly every so often, keep the winner
    for a long dwell") is the same one, only the trigger changes from a
    fixed cadence to the accept signal.
    """

    # HAKEN 2 (NEXTN cold start): rounds of dual capture + warm-keep run
    # BEFORE the candidate rung is measured. A candidate whose draft KV has
    # a hole for the whole steady segment starts with a burn-in dip; measuring
    # that dip would systematically UNDER-rate the candidate (worst for
    # NEXTN, whose 1-layer MTP draft KV must be re-extended) and the probe
    # would always come out in favor of the incumbent. 5 rounds mirrors the
    # bandit's own burn_in_rounds=4 plus one round of slack.
    warmup_rounds: int = 5
    # Length of the measured part of the window (bandit: probe_window_rounds).
    probe_window_rounds: int = 32
    # MINIMUM gap between the end of one window and the start of the next
    # (bandit: probe_interval_rounds). An upper bound on probe cadence, not
    # a periodic trigger -- the signal decides whether a probe happens at all.
    probe_interval_rounds: int = 512
    # Minimum residence on a freshly adopted family (bandit: min_dwell_rounds).
    min_dwell_rounds: int = 64
    # Cadence of the (cheap, capture-neutral) k decision inside the adopted
    # NEXTN family; mirrors the bandit's decide_interval_rounds.
    decide_interval_rounds: int = 16
    # ---- signal gate (part 3) -----------------------------------------
    # The gate compares NEXTN's accept EMA against the best NEXTN accept EMA
    # seen this session. Relative, so no model-specific absolute number has
    # to be guessed: on the measured z-lab drafter code sits at ~4.3 and
    # prose at ~2.9-3.6, i.e. prose lands at ~0.67-0.84 of the code peak,
    # which the 0.90/0.75 hysteresis band separates cleanly.
    signal_high_frac: float = 0.90
    signal_low_frac: float = 0.75
    # Optional absolute floor on NEXTN's accept EMA (0 = off). Left off by
    # default: an absolute threshold is a per-model constant and none has
    # been measured for anything but the z-lab drafter.
    signal_min_accept: float = 0.0
    # Per-round decay of the remembered best NEXTN accept. 1.0 (default) =
    # no decay: a session that once saw structured content and then never
    # again SHOULD stop probing -- that is the point.
    signal_best_decay: float = 1.0
    # After a probe that LOST, the signal must improve by this fraction over
    # its level at that failed probe before the same candidate is retried.
    # Without it, a flat-signal session re-probes every probe_interval and
    # pays the dual tax for an answer it already has.
    reprobe_improve_frac: float = 0.05
    # ---- warm-keep stride (the fairness/tax dial) ---------------------
    # 0 (default): NO warm-keep in the steady state -- maximum tax saving.
    # The cost is that the INACTIVE rung's draft KV develops a gap over the
    # segment (neither draft is stateless: the MTP rung owns a 1-layer draft
    # KV pool, the solo DFLASH rung a pool fed from the target's aux hidden
    # states), and the warm-up window only re-primes its TAIL. Both drafts
    # are recency-dominated, so a tail may well be enough -- but if the GPU
    # measurement shows the probe is biased toward the incumbent by a holey
    # candidate KV, N >= 1 keeps the inactive rung warm every Nth round
    # instead: N=4 pays 25% of today's warm-keep cost and caps the gap at 3
    # rounds. This is the pre-wired fallback, not a second mechanism.
    warmkeep_stride: int = 0

    @classmethod
    def from_env(cls, **overrides) -> "LazyCaptureConfig":
        kwargs = dict(overrides)
        for f in fields(cls):
            raw = os.environ.get(LAZY_ENV_PREFIX + f.name.upper())
            if raw is not None:
                kwargs[f.name] = (
                    float(raw) if isinstance(f.default, float) else int(raw)
                )
        cfg = cls(**kwargs)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        for name in (
            "warmup_rounds",
            "probe_window_rounds",
            "probe_interval_rounds",
            "min_dwell_rounds",
            "decide_interval_rounds",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"cross lazy: {name} must be >= 1")
        if not (0.0 < self.signal_low_frac <= self.signal_high_frac <= 1.0):
            raise ValueError(
                "cross lazy: need 0 < signal_low_frac <= signal_high_frac <= 1; "
                f"got {self.signal_low_frac} / {self.signal_high_frac}"
            )
        if self.signal_min_accept < 0.0:
            raise ValueError("cross lazy: signal_min_accept must be >= 0")
        if self.warmkeep_stride < 0:
            raise ValueError("cross lazy: warmkeep_stride must be >= 0")
        if not (0.0 < self.signal_best_decay <= 1.0):
            raise ValueError("cross lazy: signal_best_decay must be in (0, 1]")
        if self.reprobe_improve_frac < 0.0:
            raise ValueError("cross lazy: reprobe_improve_frac must be >= 0")


@dataclass(frozen=True)
class LazyVerdict:
    """What the worker must do for the round that is about to run."""

    phase: str
    adopted: str
    candidate: Optional[str]
    # The family that actually RUNS this round (steady/warm-up: the
    # incumbent; measure: the candidate).
    running: str
    # Target-model aux-hidden capture required this round (DFLASH's input).
    aux_capture: bool
    # Run the per-round warm-keep of the INACTIVE rung (re-primes its draft
    # KV so the incoming rung is not measured cold).
    warmkeep: bool
    # A rank-0 decision is due at this round boundary; its answer must be
    # broadcast and fed back through commit().
    decision_kind: Optional[str]
    # The family the worker must run from this round on (None: unchanged).
    family_target: Optional[str]
    retired: bool

    @property
    def decision_due(self) -> bool:
        return self.decision_kind is not None

    @property
    def eager(self) -> bool:
        """Must the TARGET VERIFY run eager this round?

        HAKEN 1: the aux-capture setting is baked into every captured graph.
        Under lazy capture the two target-verify graph SETS are baked with
        exactly the setting their own rung needs -- the DFLASH set aux-ON,
        the NEXTN set aux-OFF -- so replay is correct whenever the running
        rung's own setting is wanted. Eager is required in exactly one case:
        aux is needed while the aux-OFF NEXTN set is the active one (a
        warm-up/probe window entered from a NEXTN steady state, or a strided
        warm-keep round there).

        Note what this is NOT: it is not a second, dual-capture GRAPH
        variant. Design variant (b) -- capturing a third target-verify set
        with both captures on -- was rejected: it would hand back exactly
        the graph memory (and the occupancy/workspace pressure) that the
        cross-algo worker already pays twice over, permanently, to buy a few
        eager rounds per probe window; and the eager verify path is not a
        new code path but the existing ``can_run_cuda_graph=False``
        fallback, exercised on every non-graph-eligible batch anyway. A
        window is ~37 rounds out of >=512, and only in the NEXTN direction
        -- a low single-digit percentage of rounds, paid only when a probe
        is actually worth running.
        """
        return self.aux_capture and self.running == "nextn"


class LazyCaptureController:
    """The lazy-capture / signal-probe state machine.

    Lifecycle per round boundary, driven by the worker:

    1. :meth:`observe` -- feed the rank-uniform measurements (round index,
       batch max ctx, both accept EMAs). Latches retirement.
    2. :meth:`step` -- advance the phase machine, return a
       :class:`LazyVerdict`.
    3. if ``verdict.decision_due``: rank 0 resolves the rung, the id is
       broadcast, and every rank calls :meth:`commit` with the SAME
       broadcast family -- which is what keeps the adopt/return outcome
       (the only wall-clock-dependent choice in the design) rank-uniform.

    Phase machine (every family change goes through a warm-up, no
    exceptions -- a direct steady->switch would hand the incoming rung a
    cold draft KV and, worse, an unpopulated draft-seed field):

        steady --signal+dwell+interval--> warmup --warmup_rounds-->
        measure --probe_window_rounds--> (adopt candidate | return) -> steady
    """

    def __init__(
        self,
        cfg: LazyCaptureConfig,
        retire: Optional[RetirePolicy] = None,
        initial_family: str = "nextn",
        start_round: int = 0,
    ):
        assert initial_family in ("nextn", "dflash")
        self.cfg = cfg
        self.retire_policy = retire if retire is not None else RetirePolicy()
        self.adopted = initial_family
        self.phase = PHASE_STEADY
        self.candidate: Optional[str] = None
        self.retired = False
        self.retire_reason: Optional[str] = None
        self.retire_round: Optional[int] = None
        self.window_count = 0
        self.adopt_count = 0

        self._round = int(start_round)
        self._phase_until = 0
        self._last_window_end = int(start_round)
        self._last_adopt_round = int(start_round)
        self._last_k_decide = int(start_round)
        self._awaiting_commit: Optional[str] = None
        # Signal state (part 3).
        self._nextn_accept: Optional[float] = None
        self._dflash_accept: Optional[float] = None
        self._best_nextn: Optional[float] = None
        self._signal_high = True  # cold start: the first probe is allowed
        self._failed_probe_signal: Dict[str, float] = {}
        # Families that have actually RUN a measured window this session.
        # Everything outside this set is a cold start (see _probe_allowed).
        self._measured: set = {initial_family}

    # -- introspection (logging / tests / worker boot) ------------------
    @property
    def aux_required_now(self) -> bool:
        """Steady-state aux setting -- what the target GRAPHS must be baked
        with. Consulted at boot, before any round has run."""
        if self.phase != PHASE_STEADY:
            return True
        return self.adopted == "dflash"

    def snapshot(self) -> dict:
        return {
            "phase": self.phase,
            "adopted": self.adopted,
            "candidate": self.candidate,
            "retired": self.retired,
            "retire_reason": self.retire_reason,
            "windows": self.window_count,
            "adopts": self.adopt_count,
            "signal_high": self._signal_high,
            "nextn_accept": self._nextn_accept,
            "best_nextn_accept": self._best_nextn,
        }

    # -- 1) measurement intake (rank-uniform) --------------------------
    def observe(
        self,
        round_idx: int,
        max_ctx: int,
        nextn_accept: Optional[float],
        dflash_accept: Optional[float],
    ) -> None:
        self._round = int(round_idx)
        self._nextn_accept = nextn_accept
        self._dflash_accept = dflash_accept
        # Signal: NEXTN's own accept EMA relative to the best seen. Only the
        # NEXTN family's accept is a content signal -- it is the always-
        # available baseline rung; DFLASH's accept is only observed while
        # DFLASH runs and would therefore be circular.
        if nextn_accept is not None:
            if self._best_nextn is None:
                self._best_nextn = nextn_accept
            else:
                self._best_nextn *= self.cfg.signal_best_decay
                if nextn_accept > self._best_nextn:
                    self._best_nextn = nextn_accept
            ratio = (
                1.0
                if not self._best_nextn
                else nextn_accept / self._best_nextn
            )
            floor_ok = nextn_accept >= self.cfg.signal_min_accept
            if self._signal_high:
                if ratio <= self.cfg.signal_low_frac or not floor_ok:
                    self._signal_high = False
            else:
                if ratio >= self.cfg.signal_high_frac and floor_ok:
                    self._signal_high = True
        # Retirement is MONOTONE: latched once, never cleared.
        if not self.retired:
            do_retire, reason = retire_verdict(
                max_ctx, dflash_accept, nextn_accept, self.retire_policy
            )
            if do_retire:
                self.retired = True
                self.retire_reason = reason
                self.retire_round = self._round
                logger.info(
                    "cross lazy: DFLASH rung RETIRED permanently at round %d "
                    "(ctx %d, reason=%s, collapse_ctx=%s, dflash_accept=%s, "
                    "nextn_accept=%s). No further DFLASH probes, no aux "
                    "capture, no warm-keep for the rest of this session.",
                    self._round,
                    int(max_ctx),
                    reason,
                    self.retire_policy.collapse_ctx,
                    dflash_accept,
                    nextn_accept,
                )

    # -- 2) phase machine ----------------------------------------------
    def step(self, round_idx: int) -> LazyVerdict:
        self._round = int(round_idx)
        if self._awaiting_commit is not None:
            # A due decision was not committed yet: re-emit it unchanged
            # rather than advancing, so a dropped commit can never desync
            # the phase from the broadcast rung.
            return self._verdict(self._awaiting_commit, self._pending_target())

        if self.phase == PHASE_STEADY:
            return self._step_steady()
        if self.phase == PHASE_WARMUP:
            return self._step_warmup()
        return self._step_measure()

    def _step_steady(self) -> LazyVerdict:
        # Retirement while DFLASH is the adopted family: leave it, but --
        # like every other family change -- through a warm-up window, so the
        # incoming NEXTN rung's draft KV and seeds are re-primed first.
        if self.retired and self.adopted == "dflash":
            self._enter_warmup("nextn")
            return self._verdict(None, None)
        if self._probe_allowed():
            self._enter_warmup(self._other(self.adopted))
            return self._verdict(None, None)
        # Plain steady state. The k choice inside the adopted NEXTN family is
        # capture-neutral (same draft head, same aux setting), so it keeps
        # running on the cheap cadence.
        if (
            self.adopted == "nextn"
            and (self._round - self._last_k_decide)
            >= self.cfg.decide_interval_rounds
        ):
            self._last_k_decide = self._round
            return self._arm(DECIDE_STEADY_K, "nextn")
        return self._verdict(None, None)

    def _step_warmup(self) -> LazyVerdict:
        if self._round >= self._phase_until:
            self.phase = PHASE_MEASURE
            self._phase_until = self._round + self.cfg.probe_window_rounds
            kind = (
                DECIDE_RETIRE_EVICT
                if (self.retired and self.candidate == "nextn")
                else DECIDE_ENTER_MEASURE
            )
            return self._arm(kind, self.candidate)
        return self._verdict(None, None)

    def _step_measure(self) -> LazyVerdict:
        if self._round >= self._phase_until:
            return self._arm(DECIDE_END_MEASURE, None)
        return self._verdict(None, None)

    # -- 3) commit the (broadcast) outcome ------------------------------
    def commit(self, family: str, round_idx: int) -> None:
        """Feed back the family that was actually broadcast and applied.

        Called on EVERY rank with the identical value, which is what makes
        the adopt/return outcome rank-uniform even though rank 0 resolved it
        from wall-clock scores.
        """
        assert family in ("nextn", "dflash"), family
        self._round = int(round_idx)
        kind = self._awaiting_commit
        self._awaiting_commit = None
        if kind is None or kind == DECIDE_STEADY_K:
            return
        if kind in (DECIDE_ENTER_MEASURE, DECIDE_RETIRE_EVICT):
            # A retire eviction is not a probe: adopt immediately, no
            # measurement window.
            if kind == DECIDE_RETIRE_EVICT:
                self._adopt(family)
                return
            if family != self.candidate:
                # The worker did not actually switch (the first-boot swap
                # guard refused the incoming rung's graph tag). Measuring
                # the incumbent against itself is meaningless -- abort the
                # window and fall back to the steady state.
                self._adopt(self.adopted)
            return
        # DECIDE_END_MEASURE: rank 0 said which family wins.
        if self.candidate is not None:
            self._measured.add(self.candidate)
        won = family == self.candidate
        if not won and self.candidate is not None:
            # Remember the signal level at which this candidate lost, so the
            # same probe is not repeated until the signal genuinely improves.
            if self._nextn_accept is not None:
                self._failed_probe_signal[self.candidate] = self._nextn_accept
        self._adopt(family)

    # -- internals ------------------------------------------------------
    def _adopt(self, family: str) -> None:
        if family != self.adopted:
            self.adopt_count += 1
            self._last_adopt_round = self._round
        self.adopted = family
        self.candidate = None
        self.phase = PHASE_STEADY
        self._phase_until = 0
        self._last_window_end = self._round
        self._last_k_decide = self._round

    def _enter_warmup(self, candidate: str) -> None:
        self.phase = PHASE_WARMUP
        self.candidate = candidate
        self._phase_until = self._round + self.cfg.warmup_rounds
        self.window_count += 1

    def _arm(self, kind: str, target: Optional[str]) -> LazyVerdict:
        self._awaiting_commit = kind
        self._pending_family = target
        return self._verdict(kind, target)

    def _pending_target(self) -> Optional[str]:
        return getattr(self, "_pending_family", None)

    @staticmethod
    def _other(family: str) -> str:
        return "dflash" if family == "nextn" else "nextn"

    def _probe_allowed(self) -> bool:
        cand = self._other(self.adopted)
        if cand == "dflash" and self.retired:
            return False
        # The family dwell always binds -- it prices the switch itself.
        if (self._round - self._last_adopt_round) < self.cfg.min_dwell_rounds:
            return False
        if cand not in self._measured:
            # COLD START: this family has never run in this session, so there
            # is nothing for the interval or the signal gate to reason ABOUT.
            # Both exist to avoid re-asking a question already answered; at
            # round 0 no question has been answered. Waiting out
            # probe_interval_rounds here would strand the session on the boot
            # family for the first ~512 rounds -- measured 2026-07-24: the
            # lazy arm ran NEXTN for 517 rounds before its first probe, so
            # three of four measurement runs never saw the other family at
            # all. Mirrors the bandit's own rule ("never-measured rungs are
            # probed as soon as the dwell allows", cross_algo_bandit
            # _probe_candidate), so both explorers agree on cold start.
            return True
        if (
            self._round - self._last_window_end
        ) < self.cfg.probe_interval_rounds:
            return False
        if not self._signal_high:
            return False
        prev = self._failed_probe_signal.get(cand)
        if prev is not None:
            if self._nextn_accept is None:
                return False
            if self._nextn_accept < prev * (1.0 + self.cfg.reprobe_improve_frac):
                return False
        return True

    def _verdict(
        self, kind: Optional[str], family_target: Optional[str]
    ) -> LazyVerdict:
        in_window = self.phase in (PHASE_WARMUP, PHASE_MEASURE)
        running = (
            self.candidate
            if (self.phase == PHASE_MEASURE and self.candidate)
            else self.adopted
        )
        # Warm-keep of the inactive rung: always inside a window (that IS the
        # window's job), plus optionally on a stride in the steady state.
        warmkeep = in_window
        if not in_window and self.cfg.warmkeep_stride > 0:
            warmkeep = (self._round % self.cfg.warmkeep_stride) == 0
        # A retired DFLASH rung is never kept warm again and never needs the
        # aux capture -- that is the permanent tax saving of part 1.
        if self.retired and running == "nextn":
            warmkeep = False
        # Aux capture: DFLASH's own input while it runs, plus whatever the
        # warm-keep of an inactive DFLASH rung needs.
        aux = running == "dflash" or (warmkeep and running == "nextn")
        return LazyVerdict(
            phase=self.phase,
            adopted=self.adopted,
            candidate=self.candidate,
            running=running,
            aux_capture=aux,
            warmkeep=warmkeep,
            decision_kind=kind,
            family_target=family_target,
            retired=self.retired,
        )
