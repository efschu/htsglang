"""Adaptive verify width for the dense 27B DFlash path (``FLLIPER_ADAPTIVE_K``).

The DFlash draft proposes a fixed block of ``B`` tokens per row (slot 0 is the
anchor = last round's bonus token, slots 1..B-1 are drafts). Today the target
verifies all ``B`` slots every round. This module picks, per round, a verify
WIDTH ``k`` in ``[min_width, B]`` -- the target then verifies only slots
``0..k-1`` -- such that expected committed tokens per millisecond is maximal:

    k* = argmax_k  E[tokens](k) / step_ms(k, bs)

    E[tokens](k) = 1 + sum_{j=1}^{k-1} prod_{i<=j} accept_rate[i]
    step_ms(k, bs) = fixed_ms + verify_ms(k, bs)          (dense + attention,
                                                            no expert term)

which is Basalt's ``Controller::choose`` (basalt-ref src/spec/controller.cpp:
8-66) restricted to one draft source. ``k == B`` is exactly today's round.

What is reused instead of rebuilt (``adaptive_chain``, the EAGLE chain policy):
``choose_chain_length`` (the argmax, smallest k on ties),
``normalize_survival`` (clamp, monotone, 4-digit quantisation so TP ranks
cannot split on ULP noise) and ``switch_is_profitable`` (the hysteresis
margin). A DFlash verify width ``k`` is a chain of ``k - 1`` drafts in that
module's terms.

Two lessons of the 27B A/B of 2026-09-20 (see ``adaptive_chain`` docstring)
are structural here:

* acceptance is LEARNED per position from verified rounds only (Basalt's
  ``observe``), and unverified positions are refreshed by a periodic probe
  round at ``k = B`` -- never assumed flat;
* the cost of every candidate width comes from ONE measured curve (a file,
  ``FLLIPER_ADAPTIVE_K_CURVE``), not from the incumbent's own timing. Without
  a curve there is nothing to trade against and the width stays ``B``.

Rank uniformity (docs_new/multi_layer_eagle_adaptive_len.md §4.5): a width
change is a group decision. Every input here is rank-invariant -- the accept
counts are the rank-synced ones the scheduler hands to
``on_verify_complete_cpu``, ``batch_size`` is the batch's row count, the curve
is loaded once at boot, the round counter advances once per decode round on
every rank -- and every function is pure Python float arithmetic in a fixed
order, so all ranks compute bit-identical states. ``state_digest`` exists so
the worker can cross-check that claim cheaply.

Nothing in this file touches torch or CUDA.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import zlib
from typing import Sequence

import msgspec

from flliper.srt.speculative.adaptive_chain import (
    choose_chain_length,
    expected_tokens,
    normalize_survival,
    switch_is_profitable,
)
from flliper.srt.speculative.dspark_components.dspark_sps import SpsAdditiveCostTable

logger = logging.getLogger(__name__)

#: Rounds between two "ADAPTIVE-K census" lines (and rank cross-checks).
DEFAULT_CENSUS_EVERY = 256


class VerifyCostCurve(msgspec.Struct, frozen=True, kw_only=True):
    """Round cost per (batch rows, verify width) plus the candidate widths.

    The cost model is DSpark's ``SpsAdditiveCostTable`` (reused, not rebuilt):
    ``step_time = bias + alpha(bs) + theta(bs + budget)`` with
    ``budget = bs * (k - 1)`` verified drafts, i.e. ``theta`` is read at the
    verified ROW count ``bs * k``. A1's 27B fit ``T = A + c * rows`` is
    ``bias = A``, ``alpha = 0``, ``theta`` linear in rows.

    ``widths`` are the verify widths that get a pre-captured target state
    (the tiers); ``prior_survival`` (optional, measured) seeds the
    per-position acceptance: ``S_j = P(correct drafts >= j)``.
    """

    widths: tuple[int, ...]
    sps: SpsAdditiveCostTable
    prior_survival: tuple[float, ...] = ()
    source: str = ""


class AcceptEma(msgspec.Struct, frozen=True, kw_only=True):
    """Per-position conditional acceptance, learned online.

    ``accept_rate[j-1]`` = P(draft slot ``j`` is correct | slots ``1..j-1``
    were), for ``j = 1..B-1`` (Leviathan's alpha, per position).
    ``observed_ct[j-1]`` counts rounds in which slot ``j`` was verified.
    """

    accept_rate: tuple[float, ...]
    observed_ct: tuple[int, ...]


class AdaptiveKConfig(msgspec.Struct, frozen=True, kw_only=True):
    """Boot-static knobs (env ``FLLIPER_ADAPTIVE_K_*``)."""

    block_size: int
    min_width: int = 2
    ema_alpha: float = 0.1
    prior_accept_rate: float = 0.8
    margin_pct: float = 2.0
    min_dwell: int = 8
    warmup_rounds: int = 64
    probe_every: int = 32
    census_every: int = DEFAULT_CENSUS_EVERY
    force_width: int = 0
    #: every N-th round, while this bs has no adopted cost fit, verify the
    #: NARROWEST tier so both ends of the fit get measured (NF probe idea,
    #: Basalt draft_policy). 0 = never.
    fit_probe_every: int = 8


class AdaptiveKState(msgspec.Struct, frozen=True, kw_only=True):
    """Everything the decision depends on; replaced, never mutated."""

    round_ct: int = 0
    width: int
    incumbent: int
    held_rounds: int = 0
    switch_ct: int = 0
    probe_ct: int = 0
    verify_row_ct: int = 0
    num_accept_tokens_ct: int = 0
    width_hist: tuple[int, ...]
    ema: AcceptEma
    fit_probe_ct: int = 0
    #: (bs, a_ms, c_ms_per_row) adopted group-wide at a census: the online
    #: cost model that replaces the curve's prior for that bs.
    adopted_fit: tuple[tuple[int, float, float], ...] = ()


class RowCostFit(msgspec.Struct, frozen=True, kw_only=True):
    """Exponentially weighted least squares of ``ms = a + c * rows`` for ONE bs.

    Port of NF's ``formhp_adaptive_k.SpanFit`` (fLLiper 0d9fe5dcc) with the
    dense 27B regressor ``rows`` in place of Form B's distinct-expert share
    U(N). Fed with the measured decode ROUND PERIOD (begin of round N to
    begin of round N+1 at the same bs, no other forward between): the
    objective is committed tokens per wall ms, so a host-bound shape whose
    period does not shrink with the rows fits c ~ 0 and keeps the full width.
    """

    sw: float = 0.0
    sr: float = 0.0
    srr: float = 0.0
    sy: float = 0.0
    sry: float = 0.0
    count: int = 0
    rows_min: int = 0
    rows_max: int = 0


# --------------------------------------------------------------------------
# Cost curve
# --------------------------------------------------------------------------


#: Rows up to which a ``linear`` shorthand is tabulated (theta clamps beyond).
_LINEAR_MAX_ROWS = 4096


def linear_sps(*, a_ms: float, c_ms_per_row: float) -> SpsAdditiveCostTable:
    """``T = a + c * rows`` as a DSpark additive table (seconds)."""
    if not (math.isfinite(a_ms) and a_ms > 0.0 and math.isfinite(c_ms_per_row)):
        raise ValueError(f"bad linear curve a={a_ms!r} c={c_ms_per_row!r}")
    return SpsAdditiveCostTable(
        bias_seconds=a_ms / 1000.0,
        bs_probes=[1],
        alpha_seconds=[0.0],
        m_probes=[0, _LINEAR_MAX_ROWS],
        theta_seconds=[0.0, max(0.0, c_ms_per_row) * _LINEAR_MAX_ROWS / 1000.0],
    )


def flat_curve(*, widths: Sequence[int], step_ms: float = 1.0) -> VerifyCostCurve:
    """A curve whose cost does not depend on the width (tests, diagnostics)."""
    return VerifyCostCurve(
        widths=tuple(sorted({int(w) for w in widths})),
        sps=linear_sps(a_ms=float(step_ms), c_ms_per_row=0.0),
        source="flat",
    )


def parse_curve_text(*, text: str, source: str) -> VerifyCostCurve:
    """Parse a curve file: JSON, or Markdown carrying a fenced ```json block.

    Format (``sps`` is DSpark's ``SpsAdditiveCostTable`` JSON, seconds; the
    ``linear`` shorthand ``{"a_ms": A, "c_ms_per_row": c}`` may replace it)::

        {"widths": [5, 8],
         "linear": {"a_ms": 21.5, "c_ms_per_row": 0.93},
         "prior_survival": [0.7411, 0.5243, ...]}
    """
    stripped = text.strip()
    if not stripped.startswith("{"):
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
        if m is None:
            raise ValueError("no JSON object and no ```json block found")
        stripped = m.group(1)
    doc = json.loads(stripped)
    if "sps" in doc:
        sps = msgspec.convert(doc["sps"], type=SpsAdditiveCostTable)
    else:
        lin = doc["linear"]
        sps = linear_sps(a_ms=float(lin["a_ms"]), c_ms_per_row=float(lin["c_ms_per_row"]))
    widths = tuple(sorted({int(w) for w in doc["widths"]}))
    if not widths or widths[0] < 1:
        raise ValueError(f"bad widths {doc['widths']!r}")
    prior = tuple(float(s) for s in doc.get("prior_survival", ()))
    return VerifyCostCurve(widths=widths, sps=sps, prior_survival=prior, source=source)


def load_curve_file(*, path: str) -> VerifyCostCurve | None:
    """Load the curve at *path*; None (logged) when absent or unreadable."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return parse_curve_text(text=f.read(), source=path)
    except (OSError, ValueError, KeyError, TypeError) as e:
        logger.warning("ADAPTIVE-K curve %s unusable (%s: %s)", path, type(e).__name__, e)
        return None


def round_cost_ms(
    *,
    curve: VerifyCostCurve,
    width: int,
    batch_size: int,
    adopted: tuple[tuple[int, float, float], ...] = (),
) -> float:
    """Cost (ms) of a round verifying *width* slots for *batch_size* rows.

    The group-adopted online fit for this bs wins over the curve's prior.
    Returns ``inf`` for a width the curve does not carry, which makes it
    ineligible in ``choose_chain_length``.
    """
    if width not in curve.widths:
        return math.inf
    bs = max(1, int(batch_size))
    for fit_bs, a_ms, c_ms in adopted:
        if fit_bs == bs:
            return a_ms + c_ms * bs * int(width)
    return 1000.0 * curve.sps.step_time(num_reqs=bs, budget=bs * (int(width) - 1))


#: NF SpanFit constants (fLLiper 0d9fe5dcc): forgetting factor (half-life ~34
#: rounds) and spans needed before a fit counts.
FIT_FORGET = 0.98
FIT_MIN_SPANS = 6


def fit_observe(*, fit: RowCostFit, rows: int, ms: float, forget: float = FIT_FORGET) -> RowCostFit:
    """Fold one (rows, period ms) sample in; non-finite / non-positive ignored."""
    if not (math.isfinite(ms) and ms > 0.0):
        return fit
    r = float(rows)
    first = fit.count == 0
    return RowCostFit(
        sw=forget * fit.sw + 1.0,
        sr=forget * fit.sr + r,
        srr=forget * fit.srr + r * r,
        sy=forget * fit.sy + ms,
        sry=forget * fit.sry + r * ms,
        count=fit.count + 1,
        rows_min=int(rows) if first else min(fit.rows_min, int(rows)),
        rows_max=int(rows) if first else max(fit.rows_max, int(rows)),
    )


def fit_coef(*, fit: RowCostFit, min_spread: int) -> tuple[float, float] | None:
    """(a_ms, c_ms_per_row), or None until identifiable (both ends measured).

    ``c < 0`` (noise: cost falling with rows) is clamped to 0, which prices
    every width the same and so keeps the full width.
    """
    if fit.count < FIT_MIN_SPANS or fit.rows_max - fit.rows_min < max(1, int(min_spread)):
        return None
    mu = fit.sr / fit.sw
    var = fit.srr / fit.sw - mu * mu
    if var <= 1e-12:
        return None
    c = max(0.0, (fit.sry / fit.sw - mu * fit.sy / fit.sw) / var)
    return fit.sy / fit.sw - c * mu, c


def adopt_fits(
    *, state: AdaptiveKState, fits: Sequence[tuple[int, float, float]]
) -> AdaptiveKState:
    """Install the group-agreed cost fits (one rank's, broadcast at a census)."""
    merged = {bs: (a, c) for bs, a, c in state.adopted_fit}
    for bs, a, c in fits:
        merged[int(bs)] = (float(a), float(c))
    return msgspec.structs.replace(
        state, adopted_fit=tuple((bs, a, c) for bs, (a, c) in sorted(merged.items()))
    )


def candidate_widths(*, cfg: AdaptiveKConfig, curve: VerifyCostCurve | None) -> tuple[int, ...]:
    """Widths the controller may return, ascending; always ends with ``B``.

    Only widths the curve measured (and that the worker therefore pre-captures)
    qualify. ``(B,)`` -- i.e. today's fixed round -- when there is no curve, the
    curve lacks ``B`` (nothing to compare the alternatives against), or a
    forced width is set (then ``(force,)`` if it is a curve width).
    """
    b = int(cfg.block_size)
    if curve is None or b not in curve.widths:
        return (b,)
    if cfg.force_width:
        f = int(cfg.force_width)
        return (f,) if (f == b or (f in curve.widths and cfg.min_width <= f < b)) else (b,)
    return tuple(w for w in curve.widths if max(1, cfg.min_width) <= w <= b)


# --------------------------------------------------------------------------
# Acceptance
# --------------------------------------------------------------------------


def initial_ema(*, cfg: AdaptiveKConfig, curve: VerifyCostCurve | None = None) -> AcceptEma:
    """Prior: the curve's measured survival (as conditional rates), else flat."""
    n = max(0, int(cfg.block_size) - 1)
    rates = [float(cfg.prior_accept_rate)] * n
    survival = list(curve.prior_survival) if curve is not None else []
    prev = 1.0
    for j, s in enumerate(survival[:n]):
        rates[j] = min(1.0, max(0.0, s / prev)) if prev > 0.0 else 0.0
        prev = s
    return AcceptEma(accept_rate=tuple(rates), observed_ct=tuple(0 for _ in range(n)))


def observe_round(
    *, ema: AcceptEma, num_correct_drafts: Sequence[int], width: int, alpha: float
) -> AcceptEma:
    """Fold one verified round into the per-position EMA (Basalt ``observe``).

    Row ``r`` with ``a_r`` correct drafts at width ``k`` tested slots
    ``1..min(a_r + 1, k - 1)``: slots ``<= a_r`` were hits, slot ``a_r + 1``
    (when verified) the miss. Per slot the round's hit fraction over the rows
    that tested it moves the EMA once -- batch-size independent smoothing.
    Rows that accepted every verified slot pull the unverified tail toward the
    deepest verified rate (Basalt's pull), weighted by their share; without
    it the tail keeps its prior forever and the width can never grow.
    """
    rates = list(ema.accept_rate)
    seen = list(ema.observed_ct)
    n = len(rates)
    drafts = max(0, min(int(width) - 1, n))
    rows = [max(0, min(int(a), drafts)) for a in num_correct_drafts]
    if not rows or drafts == 0:
        return ema
    for j in range(1, drafts + 1):
        tested = sum(1 for a in rows if a >= j - 1)
        if tested == 0:
            break
        hits = sum(1 for a in rows if a >= j)
        rates[j - 1] += alpha * (hits / tested - rates[j - 1])
        seen[j - 1] += 1
    full_share = sum(1 for a in rows if a == drafts) / len(rows)
    if drafts < n and full_share > 0.0:
        deepest = rates[drafts - 1]
        for j in range(drafts, n):
            rates[j] += alpha * full_share * (deepest - rates[j])
    return AcceptEma(accept_rate=tuple(rates), observed_ct=tuple(seen))


def survival_curve(*, ema: AcceptEma) -> list[float]:
    """Cumulative product of the per-position rates, normalised (quantised)."""
    out, run = [], 1.0
    for p in ema.accept_rate:
        run *= p
        out.append(run)
    return normalize_survival(out, len(out))


def expected_accept_tokens(*, ema: AcceptEma, width: int) -> float:
    """E[committed tokens] of a round at *width* (drafts + the bonus token)."""
    return expected_tokens(survival_curve(ema=ema), int(width) - 1)


# --------------------------------------------------------------------------
# Decision
# --------------------------------------------------------------------------


def choose_width(
    *,
    cfg: AdaptiveKConfig,
    curve: VerifyCostCurve | None,
    ema: AcceptEma,
    batch_size: int,
    incumbent: int,
    adopted: tuple[tuple[int, float, float], ...] = (),
) -> int:
    """argmax_k E[tokens](k) / step_ms(k), with the hysteresis margin.

    The argmax alone is ``adaptive_chain.choose_chain_length`` over chain
    length ``k - 1``. A different width is only taken when it beats the
    incumbent's rate by more than ``margin_pct`` (``switch_is_profitable``
    with zero swap cost: every candidate state is resident, the switch is a
    pointer swap).
    """
    widths = candidate_widths(cfg=cfg, curve=curve)
    if len(widths) == 1:
        return widths[0]
    assert curve is not None
    b = int(cfg.block_size)
    survival = survival_curve(ema=ema)

    def cost(chain: int) -> float:
        return round_cost_ms(
            curve=curve, width=chain + 1, batch_size=batch_size, adopted=adopted
        )

    best = 1 + choose_chain_length(
        survival,
        cost,
        k_max=b - 1,
        k_min=widths[0] - 1,
        candidates=[w - 1 for w in widths],
    )
    if best == incumbent or incumbent not in widths:
        return best
    profitable = switch_is_profitable(
        e_incumbent=expected_tokens(survival, incumbent - 1),
        c_incumbent=cost(incumbent - 1),
        e_candidate=expected_tokens(survival, best - 1),
        c_candidate=cost(best - 1),
        swap_ms=0.0,
        dwell_rounds=1,
        margin=max(0.0, float(cfg.margin_pct)) / 100.0,
    )
    return best if profitable else incumbent


def initial_state(*, cfg: AdaptiveKConfig, curve: VerifyCostCurve | None) -> AdaptiveKState:
    b = int(cfg.block_size)
    return AdaptiveKState(
        width=b,
        incumbent=b,
        width_hist=tuple(0 for _ in range(b + 1)),
        ema=initial_ema(cfg=cfg, curve=curve),
    )


def plan_round(
    *,
    state: AdaptiveKState,
    cfg: AdaptiveKConfig,
    curve: VerifyCostCurve | None,
    batch_size: int,
) -> AdaptiveKState:
    """Advance one decode round and fix its verify width (``state.width``).

    Order: forced/curve-less -> ``B``; warm-up -> ``B``; a probe round every
    ``probe_every`` rounds -> ``B`` without giving up the incumbent; otherwise
    the incumbent, re-decided once it has been held ``min_dwell`` rounds.
    """
    b = int(cfg.block_size)
    round_ct = state.round_ct + 1
    widths = candidate_widths(cfg=cfg, curve=curve)
    incumbent, held, switch_ct, probe_ct = (
        state.incumbent,
        state.held_rounds + 1,
        state.switch_ct,
        state.probe_ct,
    )
    fit_probe_ct = state.fit_probe_ct
    if len(widths) == 1:
        incumbent = width = widths[0]
    elif round_ct <= cfg.warmup_rounds:
        incumbent = width = b
    elif cfg.probe_every > 0 and round_ct % cfg.probe_every == 0 and incumbent != b:
        width = b
        probe_ct += 1
    elif (
        cfg.fit_probe_every > 0
        and round_ct % cfg.fit_probe_every == 0
        and incumbent == b
        and all(bs != int(batch_size) for bs, _, _ in state.adopted_fit)
    ):
        # No adopted cost fit for this bs: measure the narrow end (the full
        # end is measured by every other round). Does not move the incumbent.
        width = widths[0]
        fit_probe_ct += 1
    else:
        if held > cfg.min_dwell:
            chosen = choose_width(
                cfg=cfg,
                curve=curve,
                ema=state.ema,
                batch_size=batch_size,
                incumbent=incumbent,
                adopted=state.adopted_fit,
            )
            if chosen != incumbent:
                incumbent, held, switch_ct = chosen, 0, switch_ct + 1
        width = incumbent
    hist = list(state.width_hist)
    hist[width] += 1
    return msgspec.structs.replace(
        state,
        round_ct=round_ct,
        width=width,
        incumbent=incumbent,
        held_rounds=held,
        switch_ct=switch_ct,
        probe_ct=probe_ct,
        fit_probe_ct=fit_probe_ct,
        width_hist=tuple(hist),
    )


def observe(
    *,
    state: AdaptiveKState,
    cfg: AdaptiveKConfig,
    num_correct_drafts: Sequence[int],
    width: int,
) -> AdaptiveKState:
    """Fold a verified round (rank-synced counts, its own width) into *state*."""
    rows = list(num_correct_drafts)
    if not rows:
        return state
    return msgspec.structs.replace(
        state,
        ema=observe_round(
            ema=state.ema, num_correct_drafts=rows, width=width, alpha=cfg.ema_alpha
        ),
        verify_row_ct=state.verify_row_ct + len(rows),
        num_accept_tokens_ct=state.num_accept_tokens_ct
        + sum(1 + max(0, min(int(a), int(width) - 1)) for a in rows),
    )


def state_digest(*, state: AdaptiveKState) -> int:
    """CRC32 of the encoded state: equal on every rank or the ranks diverged."""
    return zlib.crc32(msgspec.msgpack.encode(state))


def census_line(
    *,
    state: AdaptiveKState,
    cfg: AdaptiveKConfig,
    curve: VerifyCostCurve | None,
    batch_size: int,
) -> str:
    """One "ADAPTIVE-K" census line: histogram, switches, rates, tok/ms per width."""
    hist = " ".join(
        f"k{w}:{n}" for w, n in enumerate(state.width_hist) if n > 0
    ) or "-"
    rates = "[" + " ".join(f"{p:.3f}" for p in state.ema.accept_rate) + "]"
    mean = (
        state.num_accept_tokens_ct / state.verify_row_ct if state.verify_row_ct else 0.0
    )
    scores = []
    for w in candidate_widths(cfg=cfg, curve=curve):
        c = (
            round_cost_ms(curve=curve, width=w, batch_size=batch_size, adopted=state.adopted_fit)
            if curve
            else math.nan
        )
        e = expected_accept_tokens(ema=state.ema, width=w)
        scores.append(f"k{w}:E{e:.2f}/{c:.2f}ms")
    fits = " ".join(f"bs{b}:{a:.1f}+{c:.2f}r" for b, a, c in state.adopted_fit) or "-"
    return (
        f"ADAPTIVE-K census rounds={state.round_ct} hist={hist} "
        f"incumbent={state.incumbent} switches={state.switch_ct} "
        f"probes={state.probe_ct} fit_probes={state.fit_probe_ct} "
        f"accept_length={mean:.3f} rates={rates} "
        f"bs={batch_size} score={' '.join(scores)} fit={fits} "
        f"curve={curve.source if curve else 'none(k=B)'} "
        f"digest={state_digest(state=state):08x} "
        "(legend: k = verified slots per row incl. the anchor; E = expected "
        "committed tokens per row; ms = round cost from the adopted fit "
        "a+c*rows, else the curve prior)"
    )


class AdaptiveVerifyWidth:
    """Thin holder over the pure functions above; one per DFlash worker.

    ``next_width`` must be called exactly once per decode round on every rank
    (the round counter is part of the rank-uniform state); ``observe`` with
    every verified result's rank-synced counts.
    """

    def __init__(self, *, cfg: AdaptiveKConfig, curve: VerifyCostCurve | None) -> None:
        self.cfg = cfg
        self.curve = curve
        self.widths = candidate_widths(cfg=cfg, curve=curve)
        self.state = initial_state(cfg=cfg, curve=curve)
        self._last_batch_size = 1

    @property
    def active(self) -> bool:
        """More than one width to choose from (else every round is ``B``)."""
        return len(self.widths) > 1

    def next_width(self, *, batch_size: int) -> int:
        self._last_batch_size = int(batch_size)
        self.state = plan_round(
            state=self.state, cfg=self.cfg, curve=self.curve, batch_size=batch_size
        )
        return self.state.width

    def observe(self, *, num_correct_drafts: Sequence[int], width: int) -> None:
        self.state = observe(
            state=self.state,
            cfg=self.cfg,
            num_correct_drafts=num_correct_drafts,
            width=width,
        )

    def census_due(self) -> bool:
        every = int(self.cfg.census_every)
        return every > 0 and self.state.round_ct > 0 and self.state.round_ct % every == 0

    def adopt(self, *, fits: Sequence[tuple[int, float, float]]) -> None:
        """Install group-agreed cost fits (call on every rank, same round)."""
        self.state = adopt_fits(state=self.state, fits=fits)

    def census(self) -> str:
        return census_line(
            state=self.state,
            cfg=self.cfg,
            curve=self.curve,
            batch_size=self._last_batch_size,
        )
