"""#700: ReplaySSM divergence classifier. NOT the enable gate -- see below.

**RETIRED AS A GATE (2026-08-16).** The byte-identity criterion is structurally
unpassable and has been withdrawn as the enable condition. The kernel's own
registered test, ``test_linear_replayssm_decode.py``, settles it: its docstring
states the kernel is "algebraically equivalent but floating-point REORDERED",
and the suite contains ZERO exact-equality assertions -- every check is
tolerance-based (atol 2e-6 at L=1, 1e-4/1e-3 at L>=4). It passes 6/36 subtests
on GPU. ReplaySSM produces a nonzero delta BY DESIGN; the reordering IS the
optimization, so no probe at any level can show byte-identity.

``--enable-linear-replayssm`` therefore stays OFF and its enable path is
Quality-Last: lossy features go to the back of the queue and need quality-suite
evidence. A criterion nothing can pass is not a gate.

**What this module is still for:** regression-pinning the tolerance band. If a
future kernel change widens divergence beyond the envelope the registered test
encodes, that is a real regression, and the classifier's distinction between
"same tokens, nonzero delta" and "tokens changed" is the right shape to catch
it. Do not cite it as the enable gate.

Historical context below; the wiring half remains correct and useful.

---

The wiring contradiction, resolved:

1. **Wiring** -- resolved. The kernel header's "NOT yet wired into the memory
   pool / radix cache / scheduler / backend dispatch" was stale and false in all
   four places it named; corrected in
   ``layers/attention/fla/fused_recurrent_linear_replayssm.py``.
2. **Output fidelity** -- this module. The flag claims only "numerically
   correct", and the buffered reconstruction sums rank-1 updates in a different
   order than the sequential path, so byte-identity must be assumed ABSENT until
   measured. On a lossless queue that decides the enable outright.

The decision logic lives here, apart from the GPU probe, so the rules are
testable without a window and cannot quietly change when the probe is edited.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence, Tuple

#: GDN prefill is non-reproducible above roughly this many tokens (upstream,
#: pre-existing). A longer probe measures that, not ReplaySSM.
GDN_DETERMINISM_CEILING_TOKENS = 109


@dataclasses.dataclass(frozen=True)
class ProbePlan:
    max_tokens: int
    gate: str  # "gdn" (scalar) | "kda" (per-K)
    sample_device: str  # inputs must be sampled on CPU and moved
    arms: Tuple[str, ...]


def validate_probe_plan(plan: ProbePlan) -> None:
    """Refuse a probe that cannot support an identity claim."""
    if int(plan.max_tokens) > GDN_DETERMINISM_CEILING_TOKENS:
        raise ValueError(
            f"probe of {plan.max_tokens} tokens exceeds the ~"
            f"{GDN_DETERMINISM_CEILING_TOKENS}-token GDN prefill determinism "
            "ceiling. Above it the baseline does not reproduce itself, so the "
            "probe would measure the known non-determinism rather than "
            "ReplaySSM. Keep it short and decode-only."
        )
    if str(plan.sample_device).lower() != "cpu":
        raise ValueError(
            f"inputs sampled on {plan.sample_device!r}: torch.randn is not "
            "arch-identical across devices, so a GPU-sampled input cannot "
            "support a byte-identity claim. Sample on CPU and move."
        )
    if str(plan.gate).lower() != "gdn":
        raise ValueError(
            f"gate {plan.gate!r}: only the GDN scalar gate informs this "
            "decision. The flag's own help states KDA decode is SLOWER than the "
            "packed baseline, so a KDA arm cannot justify enabling the feature."
        )


@dataclasses.dataclass(frozen=True)
class IdentityResult:
    byte_identical: bool
    changes_emitted_tokens: bool
    max_abs_logit_delta: float


def classify_identity(
    baseline_tokens: Sequence[int],
    treatment_tokens: Sequence[int],
    max_abs_logit_delta: float,
) -> IdentityResult:
    """Classify one comparison.

    Byte-identity requires BOTH the same emitted tokens and a zero logit delta.
    Same-tokens-with-nonzero-delta is the likely real outcome and must not be
    rounded up to identical: it is a lossy result that happens not to have
    changed this particular sample's argmax.
    """
    same_tokens = list(baseline_tokens) == list(treatment_tokens)
    delta = float(max_abs_logit_delta)
    return IdentityResult(
        byte_identical=same_tokens and delta == 0.0,
        changes_emitted_tokens=not same_tokens,
        max_abs_logit_delta=delta,
    )


@dataclasses.dataclass(frozen=True)
class GateVerdict:
    enable: bool
    lossy: bool
    reason: str


def gate_verdict(
    a_vs_a: Optional[IdentityResult], a_vs_b: Optional[IdentityResult]
) -> GateVerdict:
    """Decide the enable, with a printable reason for every refusal."""
    if a_vs_a is None or a_vs_b is None:
        return GateVerdict(
            enable=False,
            lossy=False,
            reason=(
                "the identity measurement has not been run. Refusing to enable "
                "a capability whose output fidelity nobody has checked -- an "
                "unmeasured gate is a refusal, never a default-on."
            ),
        )
    if not a_vs_a.byte_identical:
        return GateVerdict(
            enable=False,
            lossy=False,
            reason=(
                "the A-vs-A floor did not hold: the baseline does not reproduce "
                f"itself (max |delta| {a_vs_a.max_abs_logit_delta:g}, tokens "
                f"{'changed' if a_vs_a.changes_emitted_tokens else 'stable'}). "
                "Until it does, any A-vs-B difference measures the harness, not "
                "ReplaySSM."
            ),
        )
    if not a_vs_b.byte_identical:
        return GateVerdict(
            enable=False,
            lossy=True,
            reason=(
                "not byte-identical on a clean A-vs-A floor (max |delta| "
                f"{a_vs_b.max_abs_logit_delta:g}, emitted tokens "
                f"{'CHANGED' if a_vs_b.changes_emitted_tokens else 'unchanged'})"
                ". ReplaySSM is therefore a lossy capability and goes last by "
                "standing policy, whatever its throughput is worth."
            ),
        )
    return GateVerdict(
        enable=True,
        lossy=False,
        reason=(
            "byte-identical on a clean A-vs-A floor: same emitted tokens and a "
            "zero logit delta. Safe to enable on the lossless queue."
        ),
    )
