# SPDX-License-Identifier: Apache-2.0
"""Pure comparison primitives for the #124 determinism harness.

Every function here is a pure function of :class:`Trajectory` pairs (CPU
tensors are fine -- nothing touches CUDA) and returns a structured
:class:`Verdict`. These are the primitives that would have flagged the FP8
"byte-identical to no-offload" overclaim (retracted ``0fb3d8007``): a
46%-token-match rerun fails ``check_machine_zero`` instantly, while
``check_near_tie_only_divergence`` PASSES it as a legitimate near-tie fork
at token 115 -- i.e. the harness distinguishes "wrong claim" from "broken
code".

Delta math is done in float64 regardless of input dtype so fp16/bf16
trajectories compare exactly.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .trajectory import FlipKind, Trajectory, Verdict, combine

__all__ = [
    "per_step_max_abs_delta",
    "classify_flip",
    "check_machine_zero",
    "check_argmax_clean_trajectory",
    "check_delta_band",
    "check_non_compounding",
    "check_self_determinism",
    "check_near_tie_only_divergence",
]


def _shape_mismatch(check: str, ref: Trajectory, test: Trajectory) -> Optional[Verdict]:
    """Common guard: incomparable trajectories fail structurally (never crash)."""
    if len(ref) != len(test):
        return Verdict(
            ok=False,
            check=check,
            message=f"trajectory length mismatch: ref={len(ref)} test={len(test)}",
        )
    if ref.logits.shape != test.logits.shape:
        return Verdict(
            ok=False,
            check=check,
            message=(
                f"logits shape mismatch: ref={tuple(ref.logits.shape)} "
                f"test={tuple(test.logits.shape)}"
            ),
        )
    return None


def per_step_max_abs_delta(ref: Trajectory, test: Trajectory) -> torch.Tensor:
    """``[T]`` float64 tensor of per-step ``max |ref - test|`` over the vocab."""
    if ref.logits.shape != test.logits.shape:
        raise ValueError(
            f"shape mismatch {tuple(ref.logits.shape)} vs {tuple(test.logits.shape)}"
        )
    return (
        (ref.logits.to(torch.float64) - test.logits.to(torch.float64))
        .abs()
        .amax(dim=-1)
    )


def classify_flip(
    ref: Trajectory, step: int, test_token: int, near_tie_margin: float
) -> Tuple[FlipKind, float]:
    """Classify an argmax flip at ``step``: genuine near-tie or corruption?

    Discriminator: the REFERENCE's own margin against the test-chosen token,
    ``ref.logits[step].max() - ref.logits[step, test_token]``. If that margin
    is <= ``near_tie_margin``, the flipped-to token was itself a near-tie
    candidate in the reference -> NEAR_TIE (an fp-reassociation coin-flip,
    the empirically observed benign class). Otherwise the reference was
    confident and the flip is real corruption.

    Checking the margin *to the chosen token* (not just top-1 vs top-2) also
    covers k-way ties and rejects a "flip into a confident-loser token even
    though top-2 happened to be close".
    """
    margin = ref.margin_to(step, test_token)
    kind = FlipKind.NEAR_TIE if margin <= near_tie_margin else FlipKind.CORRUPTION
    return kind, margin


def check_machine_zero(
    ref: Trajectory, test: Trajectory, check: str = "machine_zero"
) -> Verdict:
    """MACHINE_ZERO assertion: bit-exact equality.

    Token trajectories identical AND ``torch.equal`` on the logits (same
    shape, same dtype, max |delta| == 0). No band, no tolerance.
    """
    bad = _shape_mismatch(check, ref, test)
    if bad is not None:
        return bad
    for t, (a, b) in enumerate(zip(ref.token_ids, test.token_ids)):
        if int(a) != int(b):
            kind, margin = classify_flip(ref, t, int(b), near_tie_margin=0.0)
            return Verdict(
                ok=False,
                check=check,
                message=f"token flip at step {t}: ref={int(a)} test={int(b)}",
                first_divergence_step=t,
                flip_kind=kind,
                max_abs_delta=float(per_step_max_abs_delta(ref, test).amax()),
                margin_at_divergence=margin,
            )
    if ref.logits.dtype != test.logits.dtype:
        return Verdict(
            ok=False,
            check=check,
            message=(
                f"logits dtype mismatch: ref={ref.logits.dtype} "
                f"test={test.logits.dtype} (bit-exactness is dtype-strict)"
            ),
        )
    if not torch.equal(ref.logits, test.logits):
        deltas = per_step_max_abs_delta(ref, test)
        first = int((deltas > 0).nonzero()[0])
        return Verdict(
            ok=False,
            check=check,
            message="logits are not bit-identical",
            first_divergence_step=first,
            max_abs_delta=float(deltas.amax()),
        )
    return Verdict(
        ok=True, check=check, message="bit-identical (max|d|=0)", max_abs_delta=0.0
    )


def check_argmax_clean_trajectory(
    ref: Trajectory,
    test: Trajectory,
    near_tie_margin: float,
    check: str = "argmax_clean_trajectory",
) -> Verdict:
    """Token-level discriminator of the DECODE_CLASS: the full decode token
    trajectory must be identical to the reference (0 flips over T steps).

    Any flip fails; the verdict classifies it (NEAR_TIE -> likely benign
    fp-band coin-flip, rerun/seed-sensitivity territory; CORRUPTION -> real
    regression) so a failing gate is immediately diagnosable.
    """
    bad = _shape_mismatch(check, ref, test)
    if bad is not None:
        return bad
    for t, (a, b) in enumerate(zip(ref.token_ids, test.token_ids)):
        if int(a) != int(b):
            kind, margin = classify_flip(ref, t, int(b), near_tie_margin)
            return Verdict(
                ok=False,
                check=check,
                message=(
                    f"argmax flip at step {t}: ref={int(a)} test={int(b)} "
                    f"({kind.value}, ref margin {margin:.3e})"
                ),
                first_divergence_step=t,
                flip_kind=kind,
                margin_at_divergence=margin,
            )
    return Verdict(ok=True, check=check, message=f"argmax-clean over {len(ref)} steps")


def check_delta_band(
    ref: Trajectory,
    test: Trajectory,
    band: float,
    upto: Optional[int] = None,
    check: str = "delta_band",
) -> Verdict:
    """Bounded per-step |delta|: every compared step's max-abs logits delta
    must lie within the fp-reassociation ``band``.

    ``upto`` (exclusive) limits the check to the pre-divergence prefix when
    the trajectories legitimately fork later (SELF_DET_NEAR_TIE use).
    """
    bad = _shape_mismatch(check, ref, test)
    if bad is not None:
        return bad
    deltas = per_step_max_abs_delta(ref, test)
    if upto is not None:
        deltas = deltas[:upto]
    if deltas.numel() == 0:
        return Verdict(ok=True, check=check, message="no steps to compare (empty)")
    worst = int(deltas.argmax())
    worst_val = float(deltas[worst])
    if worst_val > band:
        return Verdict(
            ok=False,
            check=check,
            message=f"per-step |d| {worst_val:.3e} exceeds band {band:.3e} at step {worst}",
            first_divergence_step=worst,
            max_abs_delta=worst_val,
        )
    return Verdict(
        ok=True,
        check=check,
        message=f"all per-step |d| <= band {band:.3e}",
        max_abs_delta=worst_val,
    )


def check_non_compounding(
    ref: Trajectory,
    test: Trajectory,
    band: float,
    growth_factor: float = 4.0,
    floor_frac: float = 0.1,
    upto: Optional[int] = None,
    check: str = "non_compounding",
) -> Verdict:
    """Drift detector: the per-step delta must not systematically GROW along
    the trajectory (fp-reassociation noise is stationary; state corruption
    -- e.g. a poisoned KV slot, a stale graph buffer -- compounds).

    Fails when the late-third median per-step delta exceeds
    ``growth_factor x`` the early-third median AND is material
    (> ``floor_frac x band``). This catches under-band drift that the plain
    band check cannot see, before it reaches an argmax flip.
    """
    bad = _shape_mismatch(check, ref, test)
    if bad is not None:
        return bad
    deltas = per_step_max_abs_delta(ref, test)
    if upto is not None:
        deltas = deltas[:upto]
    n = int(deltas.numel())
    if n < 6:
        return Verdict(
            ok=True, check=check, message=f"trajectory too short for drift check (T={n})"
        )
    third = n // 3
    early = float(deltas[:third].median())
    late = float(deltas[-third:].median())
    floor = floor_frac * band
    if late > floor and late > growth_factor * max(early, 0.0) :
        return Verdict(
            ok=False,
            check=check,
            message=(
                f"compounding drift: late-third median |d| {late:.3e} > "
                f"{growth_factor:g} x early-third median {early:.3e} "
                f"(and > {floor:.3e} = {floor_frac:g} x band)"
            ),
            max_abs_delta=float(deltas.amax()),
        )
    return Verdict(
        ok=True,
        check=check,
        message=f"no drift (early median {early:.3e}, late median {late:.3e})",
    )


def check_self_determinism(run_a: Trajectory, run_b: Trajectory) -> Verdict:
    """run == run: two runs of the IDENTICAL config (same boot flags, same
    pinned seed) must be bit-identical -- tokens and logits.

    This holds for BOTH offload paths (sorted-slot deterministic residency)
    and for the weightless lane (self-det 5/5 gates); losing it means
    history-dependent state leaked into the forward (e.g. LRU residency
    drift, an unscrubbed flashinfer workspace -- memory
    ``hetero-spec-determinismus`` root #3).
    """
    v = check_machine_zero(run_a, run_b, check="self_determinism")
    if v.ok:
        v.message = "run==run bit-identical"
    return v


def check_near_tie_only_divergence(
    ref: Trajectory,
    test: Trajectory,
    band: float,
    near_tie_margin: float,
    check: str = "near_tie_only_divergence",
) -> Verdict:
    """Divergence-vs-reference is tolerated ONLY at a genuine near-tie.

    Walk the token trajectories to the first mismatch:

    * every step BEFORE the mismatch (and the mismatch step itself) must have
      per-step |delta| within ``band``;
    * at the mismatch step, the REFERENCE margin to the test-chosen token
      must be <= ``near_tie_margin`` (a genuine coin-flip near-tie);
    * steps AFTER a legitimate near-tie fork are NOT compared -- the token
      histories differ from there on, so the cascade is expected and
      meaningless to diff (exactly the FP8 rerun pattern: first divergence
      at token 115 on a near-tie, then a legitimate fork; 118/256 match is
      NOT a failure of this class).

    A confident-margin flip, or over-band deltas before the fork, is real
    corruption and fails.
    """
    bad = _shape_mismatch(check, ref, test)
    if bad is not None:
        return bad
    fork: Optional[int] = None
    for t, (a, b) in enumerate(zip(ref.token_ids, test.token_ids)):
        if int(a) != int(b):
            fork = t
            break
    upto = len(ref) if fork is None else fork + 1  # include the flip step in band check
    band_v = check_delta_band(ref, test, band, upto=upto, check="pre_fork_delta_band")
    if not band_v.ok:
        band_v.check = check
        band_v.message = "over-band divergence before/at the fork: " + band_v.message
        band_v.flip_kind = FlipKind.CORRUPTION
        return band_v
    if fork is None:
        return Verdict(
            ok=True,
            check=check,
            message=f"no divergence over {len(ref)} steps (within band {band:.3e})",
            max_abs_delta=band_v.max_abs_delta,
        )
    kind, margin = classify_flip(ref, fork, int(test.token_ids[fork]), near_tie_margin)
    if kind is FlipKind.CORRUPTION:
        return Verdict(
            ok=False,
            check=check,
            message=(
                f"confident-token flip at step {fork}: ref margin {margin:.3e} > "
                f"near-tie margin {near_tie_margin:.3e} -- real corruption"
            ),
            first_divergence_step=fork,
            flip_kind=kind,
            margin_at_divergence=margin,
            max_abs_delta=band_v.max_abs_delta,
        )
    return Verdict(
        ok=True,
        check=check,
        message=(
            f"legitimate near-tie fork at step {fork} (ref margin {margin:.3e} <= "
            f"{near_tie_margin:.3e}); post-fork steps not comparable by design"
        ),
        first_divergence_step=fork,
        flip_kind=kind,
        margin_at_divergence=margin,
        max_abs_delta=band_v.max_abs_delta,
    )
