"""#701 — chunked-prefill admission as pool arithmetic.

Design: ``DESIGN_701_chunked_admission.md``.

The defect this replaces: chunked prefill bounds the COMPUTE per step, not the
KV COMMITMENT, and admission charged the budget one chunk (``trunc_len``) while
committing the pool to the request's entire remaining length. A 327,680-token
request was admitted on a 512-token affordability check; its own prefix then
locked, and a locked chain cannot be evicted to fund its own growth. One request
walked the pool 0.95 -> 1.00 with no second actor.

Per the binding generality clause this module is pool ARITHMETIC and contains no
rig threshold, no tuned fraction, and no model or hardware name. There is
deliberately **no chunk-size parameter**: the chunk is what the old code
substituted for the commitment, so the substitution is made unrepresentable.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class PoolState:
    """Measured pool arithmetic at the admission decision point.

    ``evictable_unlocked_tokens`` must EXCLUDE locked chains. A locked chain is
    counted by some evictable-size accessors but cannot actually be freed, and
    treating it as fundable is what let the specimen in.

    ``spillable_tokens`` is zero until a spill/retract capability for a
    request's own prefix exists. It is present so that capability raises the
    fundable total without this rule being re-derived.
    """

    free_tokens: float
    evictable_unlocked_tokens: float
    locked_tokens: float
    total_capacity_tokens: float
    spillable_tokens: float = 0.0

    @property
    def fundable_tokens(self) -> float:
        return (
            float(self.free_tokens)
            + float(self.evictable_unlocked_tokens)
            + float(self.spillable_tokens)
        )


@dataclasses.dataclass(frozen=True)
class AdmissionDecision:
    verdict: str  # "admit" | "defer" | "refuse"
    reason: str
    required_tokens: float
    fundable_tokens: float

    @property
    def admitted(self) -> bool:
        return self.verdict == "admit"


def decide_chunked_admission(
    remaining_tokens: int, pool: PoolState
) -> AdmissionDecision:
    """Decide admission from the request's FULL remaining length.

    Three verdicts, all derived:

    * ``refuse`` -- the request exceeds total pool capacity, so it can never fit
      at any future time and admitting it can only deadlock. The only hard
      error, and it comes from capacity, not from a tuned fraction.
    * ``defer`` -- it does not fit now but may once other requests finish and
      unlock their chains.
    * ``admit`` -- it fits.
    """
    required = float(remaining_tokens)
    fundable = pool.fundable_tokens
    capacity = float(pool.total_capacity_tokens)

    if required > capacity:
        return AdmissionDecision(
            verdict="refuse",
            reason=(
                f"remaining length {required:,.0f} tokens exceeds total pool "
                f"capacity {capacity:,.0f}: the request can never fit at any "
                "future time, so admitting it could only deadlock the instance."
            ),
            required_tokens=required,
            fundable_tokens=fundable,
        )

    if required > fundable:
        return AdmissionDecision(
            verdict="defer",
            reason=(
                f"remaining length {required:,.0f} tokens exceeds the fundable "
                f"{fundable:,.0f} (free {pool.free_tokens:,.0f} + unlocked "
                f"evictable {pool.evictable_unlocked_tokens:,.0f} + spillable "
                f"{pool.spillable_tokens:,.0f}); {pool.locked_tokens:,.0f} "
                "locked tokens cannot fund it. Deferring rather than admitting "
                "into a self-deadlock."
            ),
            required_tokens=required,
            fundable_tokens=fundable,
        )

    return AdmissionDecision(
        verdict="admit",
        reason=(
            f"remaining length {required:,.0f} tokens is fundable from {fundable:,.0f}."
        ),
        required_tokens=required,
        fundable_tokens=fundable,
    )


def effective_running_bs(running_bs: int, resident_chunked: int) -> int:
    """#631 defect O, as a counting truth.

    A chunked request is resident-but-batchless: it holds a locked prefix in the
    pool while appearing in no batch, so a raw ``running_bs`` reads 0 and every
    consumer -- the KV-pressure ladder, the min-free-slots delayer, the idle and
    flip detectors -- concludes the instance is idle while it is in fact
    wedged. Resident-but-batchless counts as RUNNING.
    """
    return int(running_bs) + int(resident_chunked)
