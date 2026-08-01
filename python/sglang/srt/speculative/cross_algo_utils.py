"""Cross-algorithm speculative decoding (T156 stage 2) -- argument shaping.

Under ``--speculative-cross-algorithm`` ONE server hosts TWO co-resident
speculative rungs over the SAME target model:

* the NEXTN/MTP rung (EAGLEWorkerV2, draft = the target checkpoint's MTP
  head, placement SPLIT -- TP-sharded exactly like a plain NEXTN server), and
* the DFLASH rung (DFlashWorkerV2, draft = a separate DFLASH checkpoint,
  placement SOLO on TP rank 0 -- forced, because DFlashAttention requires
  heads % tp == 0, which is unsatisfiable under uneven tp3).

The two rungs need DIFFERENT values in the same ``speculative_*`` server-args
fields (algorithm, draft path, num_steps, num_draft_tokens, placement, ...).
This module resolves both value sets ("shapes") once at argument-handling
time:

* the FORCED rung's shape (``--speculative-cross-algorithm-force``) is applied
  to the real ServerArgs, so the whole boot pipeline (scheduler, target model
  runner, pool sizing, graph capture, batch stamping) behaves exactly like the
  corresponding single-algorithm server -- that is the stage-2 contract, and
  it is what makes the forced arm comparable to its single-algo control;
* the OTHER rung's shape is stashed on the ServerArgs instance
  (``speculative_cross_shapes``) and consumed by CrossAlgoWorker to build the
  secondary sub-worker from a deep-copied, re-shaped ServerArgs.

Stage 3 (per-batch switching) only changes who stamps
``batch.spec_algorithm``; the shapes and the dual resource build stay.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

CROSS_SHAPES_ATTR = "speculative_cross_shapes"

# ---------------------------------------------------------------------------
# Dual hidden-state capture (stage 3, schedule mode only).
#
# Under per-batch switching BOTH rungs must stay warm on every verify round:
# the NEXTN/MTP draft-extend consumes the target's FINAL hidden states, the
# DFLASH draft-KV append consumes the concatenated AUX layer hidden states.
# With this process-local flag on, LogitsProcessor stores the final hidden
# states in `hidden_states` (the EAGLE convention -- every EAGLE consumer
# stays untouched) and the aux concat in the separate
# `cross_aux_hidden_states` field (read by the DFLASH rung under the cross
# gate). Off (default, incl. the stage-2 force modes and every single-algo
# server): behavior is byte-identical to before.
# ---------------------------------------------------------------------------
_DUAL_CAPTURE_ACTIVE = False


def set_dual_capture_active(enabled: bool) -> None:
    global _DUAL_CAPTURE_ACTIVE
    _DUAL_CAPTURE_ACTIVE = bool(enabled)


def dual_capture_active() -> bool:
    return _DUAL_CAPTURE_ACTIVE


# Numeric shape fields swapped onto the REAL ServerArgs on every rung switch
# (schedule mode). Audited runtime readers that must see the ACTIVE rung's
# value: dflash_info_v2.prepare_for_decode (block size), the eager verify
# sizing fallback, metrics. The algorithm STRING and every boot-sizing
# consumer (max_speculative_num_draft_tokens & friends) stay fixed.
RUNTIME_SHAPE_FIELDS = (
    "speculative_num_steps",
    "speculative_eagle_topk",
    "speculative_num_draft_tokens",
    "speculative_dflash_block_size",
)


def apply_runtime_shape(server_args, shape: Dict[str, Any], reason: str) -> None:
    """Atomically point the REAL ServerArgs' numeric spec-shape fields at the
    active rung's values (single override call, single stamp point)."""
    server_args.override(
        reason, **{f: shape[f] for f in RUNTIME_SHAPE_FIELDS}
    )

# Fields a rung shape may override on the deep-copied ServerArgs of the
# secondary sub-worker (and that the forced shape writes on the real one).
_SHAPE_FIELDS = (
    "speculative_algorithm",
    "speculative_draft_model_path",
    "speculative_draft_model_revision",
    "speculative_num_steps",
    "speculative_eagle_topk",
    "speculative_num_draft_tokens",
    "speculative_dflash_block_size",
    "speculative_draft_attention_backend",
    "speculative_draft_placement",
    "speculative_draft_gpu",
    "speculative_adaptive",
    "speculative_adaptive_config",
    "speculative_draft_window_size",
    "speculative_use_rejection_sampling",
)


def cross_algo_enabled(server_args) -> bool:
    return bool(getattr(server_args, "speculative_cross_algorithm", False))


def cross_schedule_interval(server_args) -> Optional[int]:
    """Verify-round switch interval under force=schedule:N, else None.

    Stage-3 debug policy only; the runtime-switching code paths themselves
    are gated by ``cross_switching_active`` (schedule AND auto).
    """
    if not cross_algo_enabled(server_args):
        return None
    shapes = getattr(server_args, CROSS_SHAPES_ATTR, None)
    if shapes is None or shapes.get("force") != "schedule":
        return None
    return int(shapes["schedule_interval"])


# The per-batch-switching modes (stage 3 schedule + stage 4 auto + the
# deterministic drafter policy). Everything these modes share -- dual hidden
# capture, dual prefill, per-round warm-keeping, the scheduler pre-decode
# switch hook, DFLASH request validation for every request -- gates on
# membership here; the static force=nextn|dflash modes keep their stage-2
# behavior untouched.
SWITCHING_MODES = ("schedule", "auto", "policy")


def cross_switching_active(server_args) -> bool:
    """True when the active-rung policy switches rungs at runtime."""
    if not cross_algo_enabled(server_args):
        return False
    shapes = getattr(server_args, CROSS_SHAPES_ATTR, None)
    return shapes is not None and shapes.get("force") in SWITCHING_MODES


def parse_cross_force(force: Optional[str]) -> tuple[str, Optional[int]]:
    """Parse --speculative-cross-algorithm-force into (mode, interval)."""
    if force in ("nextn", "dflash", "auto", "policy"):
        return force, None
    if isinstance(force, str) and force.startswith("schedule:"):
        try:
            interval = int(force.split(":", 1)[1])
        except ValueError:
            interval = -1
        if interval < 1:
            _fail(
                "force=schedule:N needs a positive integer switch interval "
                f"(verify rounds); got {force!r}."
            )
        return "schedule", interval
    _fail(
        "--speculative-cross-algorithm-force must be 'nextn', 'dflash', "
        "'schedule:N' (switch every N verify rounds), 'auto' (stage-4 "
        "acceptance-driven bandit), or 'policy' (deterministic ctx-threshold "
        f"drafter policy, see --speculative-drafter-policy); got {force!r}."
    )


# ---------------------------------------------------------------------------
# Context gate for the DFLASH rung (auto mode).
#
# The z-lab DFLASH drafter is a SHORT-CONTEXT model: trained at ctx 4096 with
# an interleaved-SWA stack in which 4 of its 5 layers physically see only the
# last `sliding_window` (2048) tokens. The measured regime map
# (dflash_verdict.md, 2026-07-22 fixed-bench data) shows its acceptance
# ADVANTAGE over NEXTN evaporates past ~10k context; its only win region is
# short/mid-context code. Without a gate the bandit discovers this only
# reactively, through probe cycles that each cost a probe window on the wrong
# rung (97% of observed switches were probes; DFLASH kept being probed at 50k
# where it cannot win).
#
# The gate makes context an ELIGIBILITY criterion: while the batch context is
# at/above the threshold, the DFLASH rung is (a) never selectable as the
# active rung and (b) never probed -- (b) is the actual overhead killer.
# ---------------------------------------------------------------------------

CTX_GATE_FACTOR_ENV = "SGLANG_CROSS_CTX_GATE_FACTOR"
CTX_GATE_NEAR_ENV = "SGLANG_CROSS_CTX_GATE_NEAR_FRAC"
# Threshold = factor * sliding_window for SWA drafters (see
# derive_ctx_gate_threshold for the formula rationale).
DEFAULT_CTX_GATE_FACTOR = 4.0
# Turn-start pre-emption nearness: a request whose max_new_tokens budget can
# carry it past the threshold makes DFLASH ineligible from its decode start
# ONLY when the context is already this close to the threshold.
DEFAULT_CTX_GATE_NEAR_FRAC = 0.8


def parse_ctx_gate_value(raw: Optional[str]):
    """Parse --speculative-cross-algorithm-ctx-gate: 'auto' | 'off' | int."""
    if raw is None:
        return "auto"
    val = str(raw).strip().lower()
    if val in ("auto", "off"):
        return val
    try:
        tokens = int(val)
    except ValueError:
        tokens = -1
    if tokens < 1:
        _fail(
            "--speculative-cross-algorithm-ctx-gate must be 'auto', 'off', "
            f"or a positive token count; got {raw!r}."
        )
    return tokens


def _read_drafter_ctx_fields(
    draft_model_path: str,
) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    """Read (max_position_embeddings, effective sliding_window, error) from
    the drafter's config.json. ``sliding_window`` is None when SWA is
    declared disabled. Shared by the ctx-gate and drafter-policy
    derivations."""
    cfg_path = os.path.join(str(draft_model_path), "config.json")
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except (OSError, ValueError) as e:
        return None, None, f"drafter config unreadable ({cfg_path}: {e})"
    mpe = cfg.get("max_position_embeddings")
    sliding = cfg.get("sliding_window")
    if cfg.get("use_sliding_window") is False:
        sliding = None
    return mpe, sliding, None


def derive_ctx_gate_threshold(
    draft_model_path: str,
) -> Tuple[Optional[int], str]:
    """Derive the DFLASH context-gate threshold from the DRAFTER's config.json.

    Returns (threshold_tokens or None, human-readable derivation source).

    Formula (auto mode):

    * SWA drafter (``sliding_window`` declared, not disabled):
          threshold = CTX_GATE_FACTOR * sliding_window   (capped at
          max_position_embeddings when declared)
      Rationale: ``max_position_embeddings`` is routinely inherited from the
      TARGET (the z-lab Qwen3.6-27B drafter declares 262144) and says nothing
      about the drafter's usable range; the sliding window is the drafter's
      OWN structural horizon -- beyond it, most of its layers are blind to
      the prompt. The default factor 4.0 puts the z-lab drafter
      (window 2048, trained at ctx 4096 = 2 * window) at 4 * 2048 = 8192:
      2x its training context, comfortably above its measured win region
      (short/mid-context code, paired wins at ~2-4k, code parity ~21k is
      acceptance-peak noise on its own outputs) and below the ~10k point
      where the regime map shows its acceptance edge fully evaporated.
    * No SWA declared: threshold = max_position_embeddings -- a genuine
      long-context drafter lifts (effectively removes) the gate
      automatically.
    * Neither field / config unreadable: no gate (None), logged.

    The factor is configurable via SGLANG_CROSS_CTX_GATE_FACTOR.
    """
    mpe, sliding, err = _read_drafter_ctx_fields(draft_model_path)
    if err is not None:
        return None, f"{err}; gate disabled"
    factor = float(
        os.environ.get(CTX_GATE_FACTOR_ENV, DEFAULT_CTX_GATE_FACTOR)
    )
    if factor <= 0:
        _fail(f"{CTX_GATE_FACTOR_ENV} must be > 0; got {factor}.")
    if sliding:
        threshold = int(factor * int(sliding))
        source = f"{factor:g} * sliding_window {int(sliding)}"
        if mpe:
            threshold = min(threshold, int(mpe))
            source += f" (cap max_position_embeddings {int(mpe)})"
        return threshold, source
    if mpe:
        return int(mpe), "max_position_embeddings (no sliding window declared)"
    return None, (
        f"drafter config under {draft_model_path} declares neither "
        "sliding_window nor max_position_embeddings; gate disabled"
    )


def resolve_ctx_gate(server_args, dflash_draft_path: str) -> Dict[str, Any]:
    """Resolve the ctx-gate stash entry at argument time (rank-uniform by
    construction: runs once in the launcher process, travels to every
    scheduler process inside the pickled shapes stash)."""
    parsed = parse_ctx_gate_value(
        getattr(server_args, "speculative_cross_algorithm_ctx_gate", "auto")
    )
    near_frac = float(
        os.environ.get(CTX_GATE_NEAR_ENV, DEFAULT_CTX_GATE_NEAR_FRAC)
    )
    if not (0.0 < near_frac <= 1.0):
        _fail(f"{CTX_GATE_NEAR_ENV} must be in (0, 1]; got {near_frac}.")
    if parsed == "off":
        return {"threshold": None, "near_frac": near_frac, "source": "off"}
    if parsed == "auto":
        threshold, source = derive_ctx_gate_threshold(dflash_draft_path)
        return {"threshold": threshold, "near_frac": near_frac, "source": source}
    return {
        "threshold": int(parsed),
        "near_frac": near_frac,
        "source": "explicit --speculative-cross-algorithm-ctx-gate",
    }


def ctx_gate_eligible(
    ctx_and_remaining: List[Tuple[int, Optional[int]]],
    threshold: int,
    near_frac: float,
) -> Tuple[bool, int]:
    """Pure eligibility predicate for the DFLASH rung.

    ``ctx_and_remaining``: per request, (current context length, remaining
    max_new_tokens budget or None for unbounded). Returns
    (eligible, max_ctx).

    Ineligible when ANY request either
    * already sits at/above the threshold (bs>1: the max over requests is
      what the batch pays in verify-attention cost), or
    * TURN-START PRE-EMPTION: is already NEAR the threshold
      (ctx > near_frac * threshold) and its remaining new-token budget can
      carry it past -- then DFLASH is not selected from the start instead of
      switching away mid-stream shortly after. Requests far below the
      threshold are NOT preempted on budget alone (no answer-length
      estimator: a 2k-ctx request with a 16k budget usually stops early).
      A genuine mid-stream crossing is handled by the per-round-boundary
      eligibility check (switch cost ~1 round; no special path).
    """
    eligible = True
    max_ctx = 0
    for ctx, remaining in ctx_and_remaining:
        ctx = int(ctx)
        if ctx > max_ctx:
            max_ctx = ctx
        if ctx >= threshold:
            eligible = False
            continue
        if ctx > near_frac * threshold and (
            remaining is None or ctx + int(remaining) > threshold
        ):
            eligible = False
    return eligible, max_ctx


# ---------------------------------------------------------------------------
# Deterministic drafter policy (force=policy, T156 follow-up).
#
# A GENERIC ordered table of (start_ctx -> rung): the active rung is a pure
# lookup of the batch's max context in the table -- no bandit, no probing, no
# dwell. Configured via --speculative-drafter-policy
# ("0:dflash:16,4096:nextn:3", arbitrary stage count) or derived ("auto").
#
# The ctx GATE stays active on top as a SAFETY filter: a policy-chosen but
# gate-ineligible DFLASH stage falls back to the next stage above.
# ---------------------------------------------------------------------------

POLICY_CTX_FACTOR_ENV = "SGLANG_CROSS_POLICY_CTX_FACTOR"
# Default policy switch point = factor * sliding_window = the drafter's
# TRAINING context (z-lab recipe: trained at ctx = 2 * window).
DEFAULT_POLICY_CTX_FACTOR = 2.0
# Sentinel k of a 'nextn:auto' policy stage: the k is resolved at runtime
# by the task-A analytic model (NextnAcceptModel k*), falling back to the
# boot k until the model has data. 0 is safe as a sentinel: real k arms
# are >= 1 (enforced at table validation).
POLICY_K_AUTO = 0


def derive_policy_switch_ctx(
    draft_model_path: str,
) -> Tuple[Optional[int], str]:
    """DFLASH->NEXTN switch point of the DEFAULT drafter-policy table.

    DELIBERATELY DIFFERENT from derive_ctx_gate_threshold (4 * window),
    although both read the same drafter config: the ctx GATE is an
    eligibility UPPER BOUND for the self-correcting bandit -- generosity is
    cheap there, because below the gate the bandit still measures live and
    walks away from a losing DFLASH rung. The POLICY *forces* the drafter
    with no corrective, so its switch point must sit at the PROVEN
    win-region boundary: the drafter's TRAINING context, 2 * sliding_window
    under the z-lab recipe (window 2048 trained at ctx 4096). The measured
    regime map backs this: paired-2k DFLASH wins 4/4, 10k it loses clearly,
    and 4-8k is unmeasured -- forcing DFLASH past its training context
    would be a bet, not a policy.

    Returns (switch_ctx or None, derivation source). SWA drafter:
    factor * sliding_window (factor via SGLANG_CROSS_POLICY_CTX_FACTOR,
    default 2.0), capped at max_position_embeddings. No SWA declared:
    max_position_embeddings (a genuine long-context drafter needs no early
    switch point). Neither: None.
    """
    mpe, sliding, err = _read_drafter_ctx_fields(draft_model_path)
    if err is not None:
        return None, err
    factor = float(
        os.environ.get(POLICY_CTX_FACTOR_ENV, DEFAULT_POLICY_CTX_FACTOR)
    )
    if factor <= 0:
        _fail(f"{POLICY_CTX_FACTOR_ENV} must be > 0; got {factor}.")
    if sliding:
        switch_ctx = int(factor * int(sliding))
        source = (
            f"{factor:g} * sliding_window {int(sliding)} "
            "(drafter training context)"
        )
        if mpe:
            switch_ctx = min(switch_ctx, int(mpe))
            source += f" (cap max_position_embeddings {int(mpe)})"
        return switch_ctx, source
    if mpe:
        return int(mpe), "max_position_embeddings (no sliding window declared)"
    return None, (
        f"drafter config under {draft_model_path} declares neither "
        "sliding_window nor max_position_embeddings"
    )


# A policy table: list of (start_ctx, rung) stages, start_ctx strictly
# ascending, stage 0 at ctx 0. rung = ("nextn", k) | ("dflash", block).
PolicyTable = List[Tuple[int, Tuple[str, int]]]


def resolve_drafter_policy_table(
    raw: Optional[str],
    *,
    arm_ks,
    dflash_block: int,
    primary_k: int,
    draft_model_path: str,
) -> Tuple[PolicyTable, str]:
    """Resolve --speculative-drafter-policy into a validated table.

    ``raw`` None or 'auto': the derived two-stage default -- the gate-gated
    rung (DFLASH) from ctx 0, NEXTN with the ANALYTIC k (task A;
    'nextn:auto', boot k until the model has data) from the drafter's
    training context (derive_policy_switch_ctx). Explicit: comma-separated
    'start_ctx:family:value' stages; every referenced rung must exist in
    the configured arm set (NEXTN k in *arm_ks* or 'auto', DFLASH block ==
    *dflash_block*), stage starts strictly ascending, first stage at 0.
    Hard errors, no silent fallback.
    """
    if raw is None or str(raw).strip().lower() == "auto":
        switch_ctx, source = derive_policy_switch_ctx(draft_model_path)
        if switch_ctx is None:
            _fail(
                "force=policy: cannot derive the default policy switch "
                f"point ({source}); pass an explicit "
                "--speculative-drafter-policy table."
            )
        return (
            [
                (0, ("dflash", int(dflash_block))),
                (int(switch_ctx), ("nextn", POLICY_K_AUTO)),
            ],
            f"auto: {source}",
        )
    entries = [e.strip() for e in str(raw).split(",") if e.strip()]
    if not entries:
        _fail("--speculative-drafter-policy: empty table.")
    table: PolicyTable = []
    for entry in entries:
        parts = entry.split(":")
        if len(parts) != 3:
            _fail(
                f"--speculative-drafter-policy entry {entry!r}: expected "
                "'start_ctx:family:value', e.g. '0:dflash:16', "
                "'4096:nextn:3' or '4096:nextn:auto'."
            )
        family = parts[1].strip().lower()
        raw_val = parts[2].strip().lower()
        nextn_auto = family == "nextn" and raw_val == "auto"
        try:
            start = int(parts[0])
            val = POLICY_K_AUTO if nextn_auto else int(parts[2])
        except ValueError:
            _fail(
                f"--speculative-drafter-policy entry {entry!r}: start_ctx "
                "and value must be integers ('auto' only as a nextn k)."
            )
        if family not in ("nextn", "dflash"):
            _fail(
                f"--speculative-drafter-policy entry {entry!r}: family must "
                "be 'nextn' or 'dflash'."
            )
        if start < 0:
            _fail(
                f"--speculative-drafter-policy entry {entry!r}: start_ctx "
                "must be >= 0."
            )
        if family == "dflash" and val != int(dflash_block):
            _fail(
                f"--speculative-drafter-policy entry {entry!r}: the resident "
                f"DFLASH rung has block size {dflash_block}; dflash:{val} "
                "is not a configured arm."
            )
        if family == "nextn" and not nextn_auto and val not in arm_ks:
            _fail(
                f"--speculative-drafter-policy entry {entry!r}: nextn k={val} "
                f"is not in the configured arm set {sorted(arm_ks)} "
                "(adaptive-config candidate steps + the boot k; widen via "
                "--speculative-adaptive-config)."
            )
        table.append((start, (family, val)))
    if table[0][0] != 0:
        _fail(
            "--speculative-drafter-policy: the first stage must start at "
            f"ctx 0; got {table[0][0]}."
        )
    for (a, _ra), (b, _rb) in zip(table, table[1:]):
        if b <= a:
            _fail(
                "--speculative-drafter-policy: stage start contexts must be "
                f"strictly ascending; got {a} followed by {b}."
            )
    return table, "explicit --speculative-drafter-policy"


def policy_rung_str(rung) -> str:
    fam, val = rung
    if fam == "nextn" and val == POLICY_K_AUTO:
        return "nextn:auto"
    return f"{fam}:{val}"


def policy_lookup_index(table: PolicyTable, max_ctx: int) -> int:
    """Index of the table stage covering *max_ctx* (last start <= ctx)."""
    idx = 0
    for i, (start, _rung) in enumerate(table):
        if max_ctx >= start:
            idx = i
        else:
            break
    return idx


def policy_select(
    table: PolicyTable,
    ctx_and_remaining: List[Tuple[int, Optional[int]]],
    gate_threshold: Optional[int],
    near_frac: float,
    fallback_rung: Tuple[str, int],
) -> Tuple[Tuple[str, int], int, bool]:
    """Deterministic drafter-policy selection for one round boundary.

    Pure function of rank-uniform inputs (CPU seq lens / budgets + the
    static table), so every rank computes the same verdict locally -- no
    broadcast, unlike the bandit. Returns (rung, max_ctx, gate_fell_back).

    Table lookup on the batch max context; the ctx GATE is applied on top
    as a SAFETY filter: when the looked-up stage is DFLASH but the gate
    (threshold/near-preemption over *all* requests) says ineligible, the
    selection falls back to the next non-DFLASH stage above, or
    *fallback_rung* (the primary NEXTN rung) when the table has none.
    """
    max_ctx = 0
    for ctx, _remaining in ctx_and_remaining:
        if int(ctx) > max_ctx:
            max_ctx = int(ctx)
    idx = policy_lookup_index(table, max_ctx)
    rung = tuple(table[idx][1])
    fell_back = False
    if rung[0] == "dflash" and gate_threshold is not None:
        eligible, _ = ctx_gate_eligible(
            ctx_and_remaining, gate_threshold, near_frac
        )
        if not eligible:
            fell_back = True
            rung = None
            for j in range(idx + 1, len(table)):
                cand = tuple(table[j][1])
                if cand[0] != "dflash":
                    rung = cand
                    break
            if rung is None:
                rung = tuple(fallback_rung)
    return rung, max_ctx, fell_back


# ---------------------------------------------------------------------------
# #156-4: DFLASH ctx retirement + lazy single capture (see cross_algo_lazy).
#
# Both are resolved at ARGUMENT time and stashed, exactly like the ctx gate:
# the resolution reads env vars and the drafter config in the launcher
# process, and every scheduler process receives the frozen result through the
# pickled shapes stash -- so no rank can ever derive a different policy.
# ---------------------------------------------------------------------------


def _resolve_retire_stash(server_args, dflash_draft_path: str) -> Dict[str, Any]:
    from sglang.srt.speculative.cross_algo_lazy import resolve_retire_policy

    raw = getattr(server_args, "speculative_cross_algorithm_retire_ctx", "off")
    _, sliding, err = _read_drafter_ctx_fields(dflash_draft_path)
    if err is not None:
        sliding = None
    try:
        policy = resolve_retire_policy(raw, sliding)
    except ValueError as e:
        _fail(str(e))
    if policy.enabled:
        logger.info(
            "Cross-algo DFLASH retirement: collapse ctx %d tokens (%s); "
            "content band below %d (low acceptance there is content, not ctx "
            "collapse); collapse criterion dflash_accept < %.2f * "
            "nextn_accept.",
            policy.collapse_ctx,
            policy.source,
            policy.band_low,
            policy.accept_ratio,
        )
    else:
        # debug, not info: the default is OFF, and flag-OFF must not even
        # change the boot log.
        logger.debug("Cross-algo DFLASH retirement: disabled (%s).", policy.source)
    return policy.to_stash()


def _resolve_lazy_stash(server_args, force: str) -> Optional[Dict[str, Any]]:
    """None == lazy capture off == today's code path, byte for byte."""
    from sglang.srt.speculative.cross_algo_lazy import LazyCaptureConfig

    if not getattr(
        server_args, "speculative_cross_algorithm_lazy_capture", False
    ):
        return None
    if force != "auto":
        _fail(
            "--speculative-cross-algorithm-lazy-capture requires "
            "--speculative-cross-algorithm-force auto (it owns the family "
            "decision, the probe windows and the capture mode together); got "
            f"force={force!r}."
        )
    # Seed the round-count knobs from the bandit config so a user who tuned
    # SGLANG_CROSS_BANDIT_* does not have to re-tune a parallel vocabulary;
    # SGLANG_CROSS_LAZY_* wins where both are set.
    from sglang.srt.speculative.cross_algo_bandit import CrossBanditConfig

    bandit = CrossBanditConfig.from_env()
    try:
        cfg = LazyCaptureConfig.from_env(
            probe_window_rounds=bandit.probe_window_rounds,
            probe_interval_rounds=bandit.probe_interval_rounds,
            min_dwell_rounds=bandit.min_dwell_rounds,
            decide_interval_rounds=bandit.decide_interval_rounds,
        )
    except ValueError as e:
        _fail(str(e))
    logger.info("Cross-algo lazy single capture: ON (%s).", cfg)
    return {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}


def get_cross_shapes(server_args) -> Dict[str, Dict[str, Any]]:
    shapes = getattr(server_args, CROSS_SHAPES_ATTR, None)
    if shapes is None:
        raise RuntimeError(
            "--speculative-cross-algorithm is set but the rung shapes were "
            "not resolved (normalize_cross_algorithm_args did not run in "
            "this process; the stash did not survive the process boundary?)."
        )
    return shapes


def _fail(msg: str) -> None:
    raise ValueError(f"--speculative-cross-algorithm: {msg}")


def normalize_cross_algorithm_args(server_args: "ServerArgs") -> None:
    """Validate the cross-algorithm request, resolve both rung shapes, apply
    the forced shape to *server_args* and stash the other shape.

    Must run inside ``handle_speculative_decoding`` AFTER the NEXTN->EAGLE
    alias resolution and BEFORE the per-algorithm handler dispatch, so the
    dispatch sees the forced shape.
    """
    from sglang.srt.arg_groups.speculative_hook import _handle_dflash

    force, schedule_interval = parse_cross_force(
        server_args.speculative_cross_algorithm_force
    )
    if server_args.speculative_algorithm != "EAGLE":
        _fail(
            "requires --speculative-algorithm NEXTN (or EAGLE) as the MTP "
            f"rung; got {server_args.speculative_algorithm!r}."
        )
    if server_args.speculative_draft_model_path is None:
        _fail(
            "requires --speculative-draft-model-path pointing at the DFLASH draft "
            "checkpoint (the MTP rung drafts from the target checkpoint and "
            "needs no path)."
        )
    if server_args.device != "cuda":
        _fail(f"only supports CUDA; got device={server_args.device!r}.")
    if server_args.pp_size != 1 or server_args.dp_size != 1:
        _fail(
            "supports pure single-node TP only; got "
            f"pp_size={server_args.pp_size}, dp_size={server_args.dp_size}."
        )
    if server_args.enable_dp_attention:
        _fail("does not support --enable-dp-attention.")
    if server_args.ep_size != 1:
        _fail(f"does not support expert parallelism (ep_size={server_args.ep_size}).")
    if server_args.nnodes != 1:
        _fail(f"is single-node only (nnodes={server_args.nnodes}).")
    if server_args.disaggregation_mode != "null":
        _fail(
            "does not support PD disaggregation "
            f"(disaggregation_mode={server_args.disaggregation_mode!r})."
        )
    if server_args.enable_multi_layer_eagle:
        _fail("does not support --enable-multi-layer-eagle.")
    if server_args.speculative_use_rejection_sampling:
        _fail("does not support --speculative-use-rejection-sampling.")
    if (
        server_args.speculative_eagle_topk is not None
        and int(server_args.speculative_eagle_topk) != 1
    ):
        _fail(
            "requires topk == 1 (linear MTP chain); got "
            f"--speculative-eagle-topk {server_args.speculative_eagle_topk}."
        )
    if server_args.speculative_draft_window_size is not None:
        _fail(
            "does not support --speculative-draft-window-size: the DFLASH "
            "rung runs draft-solo on rank 0, and solo placement has no "
            "compact draft cache."
        )
    if server_args.speculative_draft_placement != "split":
        _fail(
            "owns the draft placement (NEXTN rung: split, DFLASH rung: solo "
            "on rank 0); leave --speculative-draft-placement unset."
        )
    if server_args.speculative_draft_gpu is not None:
        _fail("owns the solo rank (rank 0); leave --speculative-draft-gpu unset.")

    user_steps = server_args.speculative_num_steps
    user_adaptive = bool(server_args.speculative_adaptive)
    user_adaptive_config = server_args.speculative_adaptive_config
    dflash_draft_path = server_args.speculative_draft_model_path
    dflash_draft_revision = server_args.speculative_draft_model_revision

    # ------------------------------------------------------------------
    # GGUF target detection.
    #
    # For an FP8/HF target the NEXTN/MTP rung drafts from the TARGET
    # checkpoint: ModelConfig swaps the arch to the *MTP variant on
    # is_draft_model and the None->model_path fallback (model_config.py) picks
    # the right directory over the HF loader -- so eagle_shape's draft path is
    # None there (see the plain NEXTN server shape).
    #
    # A GGUF target breaks that assumption: the Qwen3.5/3.6-GGUF NEXTN head
    # lives in the target .gguf file (blk.N.nextn.*), and the loader only
    # takes the GGUF adapter + MTP name-map when it goes through the .gguf
    # file with load_format=gguf. Left at None the draft falls back to the
    # target path but is routed through the HF-safetensors loader
    # (download_weights_from_hf -> "repository not found"), the reported boot
    # crash. So under a GGUF target the NEXTN rung must draft from the target
    # GGUF file explicitly, as gguf.
    #
    # _handle_load_format / _gguf_quantization run later (post-__post_init__
    # resolution pass), so detect robustly from the raw flags AND the model
    # file, not the not-yet-resolved quantization value.
    # ------------------------------------------------------------------
    from sglang.srt.utils.hf_transformers.common import check_gguf_file

    _gguf_target = (
        server_args.load_format == "gguf"
        or server_args.quantization == "gguf"
        or check_gguf_file(server_args.model_path)
    )
    if _gguf_target:
        if force == "dflash":
            _fail(
                "GGUF-target cross-algo supports the NEXTN-primary modes "
                "(force nextn/policy/auto/schedule) only; force=dflash on a "
                "GGUF target is not yet supported (the NEXTN rung must draft "
                "from the target GGUF's MTP head, which cannot be the primary "
                "under force=dflash)."
            )
        # Primary NEXTN draft loads the target GGUF via the GGUF loader.
        # Reject a conflicting draft load-format instead of silently routing
        # the GGUF draft through the HF-safetensors loader (the boot crash).
        if server_args.speculative_draft_load_format in (None, "gguf"):
            server_args.speculative_draft_load_format = "gguf"
        else:
            _fail(
                "GGUF-target cross-algo requires the NEXTN draft to load as "
                "gguf; got --speculative-draft-load-format "
                f"{server_args.speculative_draft_load_format!r}. Drop the flag "
                "or set it to 'gguf'."
            )
        # A GGUF target defaults speculative_draft_model_quantization to 'gguf'
        # (_handle_missing_default_values, run earlier); None here therefore
        # means the user passed --speculative-draft-model-quantization unquant.
        # Reject that (and any other value): the GGUF MTP draft must load as
        # gguf.
        if server_args.speculative_draft_model_quantization != "gguf":
            got = server_args.speculative_draft_model_quantization or "unquant"
            _fail(
                "GGUF-target cross-algo requires the NEXTN draft quantization "
                f"to be gguf; got --speculative-draft-model-quantization {got!r}"
                ". Drop the flag (it defaults to the gguf target)."
            )

    # ------------------------------------------------------------------
    # Resolve the DFLASH shape on a probe copy: _handle_dflash performs the
    # full validation + block-size inference + draft-attention-backend
    # resolution; reusing it verbatim keeps the cross path in lockstep with
    # the single-algorithm DFLASH path.
    # ------------------------------------------------------------------
    probe = copy.deepcopy(server_args)
    probe.speculative_algorithm = "DFLASH"
    probe.speculative_num_steps = None
    probe.speculative_eagle_topk = None
    probe.speculative_num_draft_tokens = None
    probe.speculative_adaptive = False
    probe.speculative_adaptive_config = None
    probe.speculative_draft_placement = "solo"
    _handle_dflash(probe)

    dflash_shape: Dict[str, Any] = {
        "speculative_algorithm": "DFLASH",
        "speculative_draft_model_path": dflash_draft_path,
        "speculative_draft_model_revision": dflash_draft_revision,
        "speculative_num_steps": 1,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": int(probe.speculative_num_draft_tokens),
        "speculative_dflash_block_size": int(probe.speculative_num_draft_tokens),
        "speculative_draft_attention_backend": (
            probe.speculative_draft_attention_backend
        ),
        # DFLASH rung placement is SOLO on rank 0 by requirement:
        # DFlashAttention needs heads % tp == 0, unsatisfiable under uneven
        # tp3; measured 2026-07-22 on bdc1714551.
        "speculative_draft_placement": "solo",
        "speculative_draft_gpu": None,
        "speculative_adaptive": False,
        "speculative_adaptive_config": None,
        "speculative_draft_window_size": None,
        "speculative_use_rejection_sampling": False,
    }
    if _gguf_target:
        # The DFLASH rung keeps its OWN draft checkpoint (a separate
        # HF-safetensors DFLASH model, e.g. qwen3.6-27b-dflash), NOT the GGUF
        # target. Under a GGUF target the global load_format/quantization are
        # gguf, which would (wrongly) send this HF checkpoint through the GGUF
        # loader. Force the secondary DFLASH sub-worker's ServerArgs copy back
        # to the HF loader. These extra keys are applied to the deep-copied
        # secondary args by apply_shape_to_args_copy (override(**shape)); the
        # forced-shape loop only touches _SHAPE_FIELDS, so they never mutate
        # the real (target) load_format.
        dflash_shape["load_format"] = "auto"
        dflash_shape["speculative_draft_model_quantization"] = None

    eagle_steps = int(user_steps) if user_steps is not None else 3
    eagle_shape: Dict[str, Any] = {
        "speculative_algorithm": "EAGLE",
        # FP8/HF: the MTP rung drafts from the TARGET checkpoint (ModelConfig
        # swaps the arch to the *MTP variant when is_draft_model=True and the
        # path falls back to model_path) -- exactly the plain NEXTN server
        # shape, so None is correct. GGUF: the MTP head lives in the target
        # .gguf file and must be loaded via the GGUF loader; point the draft
        # explicitly at the target GGUF file (see the _gguf_target block
        # above for the full rationale). draft_load_format/quantization for
        # this primary rung were forced to gguf on the real args above.
        "speculative_draft_model_path": (
            server_args.model_path if _gguf_target else None
        ),
        "speculative_draft_model_revision": None,
        "speculative_num_steps": eagle_steps,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": eagle_steps + 1,
        "speculative_dflash_block_size": None,
        "speculative_draft_attention_backend": None,
        "speculative_draft_placement": "split",
        "speculative_draft_gpu": None,
        # Adaptive k-laddering is an EAGLE-side feature. When DFLASH is the
        # forced primary, the secondary EAGLE rung is built WITHOUT adaptive:
        # its controller would otherwise clobber the target's active
        # attention backend / decode graph runner on activation, and the
        # launcher decided the torch_memory_saver LD_PRELOAD hook from the
        # GLOBAL (dflash-shaped, adaptive-off) args, so an offload-mode
        # controller in the copy could not initialize. The schedule mode
        # (stage 3) ALSO builds the NEXTN rung without adaptive: the
        # k-controller's _activate would pause the mapped DFLASH rung tag
        # mid-segment (use-after-unmap). The auto mode (stage 4) replaces
        # the k-controller entirely: the bandit runs the NEXTN k rungs as
        # its own arms (design 4b), so adaptive stays off there too.
        "speculative_adaptive": user_adaptive if force == "nextn" else False,
        "speculative_adaptive_config": (
            user_adaptive_config if force == "nextn" else None
        ),
        "speculative_draft_window_size": None,
        "speculative_use_rejection_sampling": False,
    }
    if force != "nextn" and user_adaptive:
        logger.info(
            "--speculative-cross-algorithm force=%s: the NEXTN rung is built "
            "without --speculative-adaptive (%s); see cross_algo_utils for "
            "the rationale.",
            force,
            (
                "the stage-4 bandit owns the k rungs"
                if force == "auto"
                else f"single k={eagle_steps} state"
            ),
        )

    # #156-4 flags are only meaningful for the rung-set modes (auto/policy);
    # fail fast instead of silently ignoring them.
    retire_raw = getattr(
        server_args, "speculative_cross_algorithm_retire_ctx", "off"
    )
    if (
        retire_raw is not None
        and str(retire_raw).strip().lower() != "off"
        and force not in ("auto", "policy")
    ):
        _fail(
            "--speculative-cross-algorithm-retire-ctx needs "
            "--speculative-cross-algorithm-force auto or policy (the DFLASH "
            f"rung must be a runtime choice to be retired); got force={force!r}."
        )
    if (
        getattr(server_args, "speculative_cross_algorithm_lazy_capture", False)
        and force != "auto"
    ):
        _fail(
            "--speculative-cross-algorithm-lazy-capture requires "
            "--speculative-cross-algorithm-force auto; got force="
            f"{force!r}."
        )

    drafter_policy_raw = getattr(server_args, "speculative_drafter_policy", None)
    if drafter_policy_raw is not None and force != "policy":
        _fail(
            "--speculative-drafter-policy is only meaningful with "
            f"--speculative-cross-algorithm-force policy; got force={force!r}."
        )

    # auto mode (stage 4): the NEXTN k rungs are bandit ARMS. The k set comes
    # from the user's --speculative-adaptive-config (positive candidate steps
    # union; the built-in default yields [1, 2, 3], the high-accept profile
    # [1..5]); the boot k (primary rung) is always included. policy mode
    # reuses the same set as the ARM SET a policy table may reference, then
    # restricts the built states to the ks the table actually uses.
    bandit_ks: Optional[list] = None
    policy_table: Optional[PolicyTable] = None
    policy_source = ""
    if force in ("auto", "policy"):
        from sglang.srt.speculative.adaptive_spec_params import (
            resolve_candidate_steps_from_config,
        )

        bandit_ks = sorted(
            {
                s
                for s in resolve_candidate_steps_from_config(
                    cfg_path=user_adaptive_config, algorithm="EAGLE"
                )
                if s >= 1
            }
            | {eagle_steps}
        )
        dflash_tokens = int(dflash_shape["speculative_num_draft_tokens"])
        if force == "policy":
            policy_table, policy_source = resolve_drafter_policy_table(
                drafter_policy_raw,
                arm_ks=set(bandit_ks),
                dflash_block=dflash_tokens,
                primary_k=eagle_steps,
                draft_model_path=dflash_draft_path,
            )
            # Build only the NEXTN k states the table references (+ the boot
            # k, always the resident baseline) -- unreferenced candidate ks
            # would cost boot time and VRAM for nothing. A 'nextn:auto'
            # stage needs the FULL candidate set: the analytic k* roams
            # over all of it.
            has_auto_stage = any(
                fam == "nextn" and v == POLICY_K_AUTO
                for _s, (fam, v) in policy_table
            )
            if not has_auto_stage:
                bandit_ks = sorted(
                    {eagle_steps}
                    | {v for _s, (fam, v) in policy_table if fam == "nextn"}
                )
        if any(k + 1 == dflash_tokens for k in bandit_ks):
            _fail(
                f"bandit k set {bandit_ks} contains k={dflash_tokens - 1}, "
                "whose draft-token stride collides with the DFLASH block "
                f"size {dflash_tokens}; verify results could not be "
                "attributed to a unique rung."
            )

    # schedule/auto mode: the NEXTN shape is the PRIMARY (global) shape. This
    # is load-bearing, not a taste choice: scheduler.spec_algorithm and the
    # FutureMap relay type derive from the global speculative_algorithm, and
    # only the EAGLE type relays topk/hidden draft seeds -- the union both
    # rungs need. The numeric shape fields are re-pointed at the active rung
    # on every switch (CrossAlgoWorker), the algorithm STRING never moves.
    forced_shape = dflash_shape if force == "dflash" else eagle_shape
    other_name = "nextn" if force == "dflash" else "dflash"
    other_shape = eagle_shape if force == "dflash" else dflash_shape

    # Apply the forced shape to the REAL args. handle_speculative_decoding is
    # still inside ServerArgs.__post_init__, so plain assignment is the
    # correct mutation style here (mirrors _handle_dflash & co).
    for field in _SHAPE_FIELDS:
        setattr(server_args, field, forced_shape[field])

    # Stash the other rung's shape for CrossAlgoWorker. Instance attribute
    # (not a dataclass field) on purpose: it is derived state, survives
    # pickling to the scheduler processes via __dict__, and stays off the
    # CLI surface. get_cross_shapes() raises loudly if it went missing.
    # schedule/auto mode additionally stashes BOTH shapes (+ the interval /
    # the bandit k set) so the meta-worker can swap the numeric fields per
    # switch.
    stash: Dict[str, Any] = {other_name: other_shape, "force": force}
    if force in SWITCHING_MODES:
        stash["nextn"] = eagle_shape
        stash["dflash"] = dflash_shape
        if force == "schedule":
            stash["schedule_interval"] = schedule_interval
        else:
            stash["bandit_ks"] = bandit_ks
            # DFLASH context gate (auto: eligibility; policy: safety
            # filter): resolved once here, at argument time, from the
            # DRAFTER's config.json.
            gate = resolve_ctx_gate(server_args, dflash_draft_path)
            stash["ctx_gate"] = gate
            if gate["threshold"] is not None:
                logger.info(
                    "Cross-algo ctx gate: DFLASH rung eligible below %d "
                    "tokens (%s; near_frac=%.2f).",
                    gate["threshold"],
                    gate["source"],
                    gate["near_frac"],
                )
            else:
                logger.info("Cross-algo ctx gate: disabled (%s).", gate["source"])
            # DFLASH ctx-collapse retirement (#156-4 part 1) and lazy
            # single capture (#156-4 parts 2+3). Both resolved here, at
            # argument time, so they travel to every scheduler process
            # inside the pickled stash -- rank-uniform by construction.
            stash["retire"] = _resolve_retire_stash(
                server_args, dflash_draft_path
            )
            stash["lazy_capture"] = _resolve_lazy_stash(server_args, force)
            if force == "policy":
                stash["policy_table"] = policy_table
                logger.info(
                    "Cross-algo drafter policy (%s): %s.",
                    policy_source,
                    ", ".join(
                        f"ctx>={start}: {policy_rung_str(rung)}"
                        for start, rung in policy_table
                    ),
                )
    object.__setattr__(server_args, CROSS_SHAPES_ATTR, stash)
    logger.info(
        "Cross-algorithm speculative decoding: force=%s%s; primary rung=%s; "
        "secondary rung=%s resident (algorithm=%s, draft_tokens=%s, "
        "placement=%s).",
        force,
        (
            f" (switch every {schedule_interval} verify rounds)"
            if force == "schedule"
            else (
                f" (bandit arms: NEXTN k={bandit_ks} + DFLASH)"
                if force == "auto"
                else (
                    f" (policy stages: {policy_table})"
                    if force == "policy"
                    else ""
                )
            )
        ),
        "nextn" if force != "dflash" else "dflash",
        other_name,
        other_shape["speculative_algorithm"],
        other_shape["speculative_num_draft_tokens"],
        other_shape["speculative_draft_placement"],
    )


def apply_shape_to_args_copy(server_args: "ServerArgs", shape: Dict[str, Any]):
    """Deep-copy *server_args* and apply a rung *shape* to the copy (via the
    audited post-resolution mutation point). The copy is what the secondary
    sub-worker keeps as ``self.server_args`` for its whole lifetime."""
    args_copy = copy.deepcopy(server_args)
    args_copy.override("cross_algo.secondary_shape", **shape)
    return args_copy
