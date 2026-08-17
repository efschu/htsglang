"""#677: phase window length SOLVED from flip cost and load, not set as a constant.

A static window is wrong in both directions: too short at high load (the system
never clears its backlog) and too long at low load (decodes wait behind an empty
prefill window). The length that works is a function of the flip cost, the
offered load and the latency budget, so it is solved here.

**The amortization argument.** Over one cycle ``C = T_prefill + T_decode +
flips x F`` the work arriving in ``C`` must also be cleared in ``C``. With
utilisations ``rho_p`` and ``rho_d`` that gives ``T_p = rho_p*C``,
``T_d = rho_d*C`` and therefore

    C >= flips * F / (1 - rho)            STABILITY FLOOR

Below that floor the backlog grows without bound however the windows are split.
Flip overhead is ``flips*F/C``, so it FALLS as the cycle lengthens — meaning
throughput always prefers a longer window and the real limit is latency:

    C <= (ttft_budget - F) / rho_d        LATENCY CEILING

because a request arriving just after the prefill window shuts waits out the
whole decode window plus one flip before its prefill can start.

**The economic window is therefore the LARGEST admissible cycle**, floored by
stability — not a midpoint, and not a constant.

**The sharpest consequence, and it is why #690's flip cost matters more than it
looks.** The two constraints move in OPPOSITE directions with ``F``: the floor
rises as ``flips*F`` and the ceiling falls as ``-F/rho_d``. So a more expensive
flip closes the feasible band from BOTH ends, and past some ``F`` the band shuts
entirely — no window length satisfies both, at any split. Halving the flip cost
does not just halve an overhead line; it reopens configurations that were
infeasible.

Two further refusals, because a window policy that quietly does the impossible
is worse than one that stops:

* **The seam must be able to arm.** If the layout's free column no longer clears
  its arming floor (#707 closed form, and the depth bound it implies), there is
  no flip to schedule at any window length.
* **The decode window must be worth entering.** Batch formation (#689) collapses
  to batch-size 1 below a queue threshold, so flipping to decode early buys a
  fraction of the decode rate for a full flip cost. That sets a floor on the
  prefill window: stay until the queue can form a batch.

No rig constants: every quantity is injected.
"""

from __future__ import annotations

import dataclasses


class PhaseWindowError(ValueError):
    """A window question that cannot be answered as posed."""


@dataclasses.dataclass(frozen=True)
class WindowInputs:
    """Measured inputs. Utilisations are fractions of wall time, not rates."""

    flip_cost_s: float
    prefill_utilisation: float
    decode_utilisation: float
    ttft_budget_s: float
    arrival_rate_per_s: float
    #: #689: below this queue depth the decode batch collapses toward size 1.
    min_decode_queue: int = 0
    flips_per_cycle: int = 2
    #: Whether the seam can arm at all under the current layout (#707).
    seam_can_arm: bool = True

    def __post_init__(self) -> None:
        if self.flip_cost_s < 0.0:
            raise PhaseWindowError("flip cost cannot be negative.")
        if self.flips_per_cycle < 1:
            raise PhaseWindowError("a cycle contains at least one flip.")
        for name in ("prefill_utilisation", "decode_utilisation"):
            v = float(getattr(self, name))
            if v < 0.0:
                raise PhaseWindowError(f"{name} cannot be negative.")
        if self.arrival_rate_per_s < 0.0:
            raise PhaseWindowError("arrival rate cannot be negative.")

    @property
    def utilisation(self) -> float:
        return float(self.prefill_utilisation) + float(self.decode_utilisation)


@dataclasses.dataclass(frozen=True)
class WindowPolicy:
    feasible: bool
    refusal: str | None
    cycle_s: float
    prefill_window_s: float
    decode_window_s: float
    stability_floor_s: float
    latency_ceiling_s: float
    batch_floor_s: float
    flip_overhead_fraction: float
    binding_constraint: str


def stability_floor_s(inputs: WindowInputs) -> float:
    """``flips * F / (1 - rho)``. Diverges as the offered load approaches 1."""
    rho = inputs.utilisation
    if rho >= 1.0:
        raise PhaseWindowError(
            f"offered load is {rho:.3f} >= 1: no window length is stable, because "
            "the phases cannot clear what arrives even with zero flip cost. This "
            "is a capacity problem, not a scheduling one."
        )
    return inputs.flips_per_cycle * float(inputs.flip_cost_s) / (1.0 - rho)


def latency_ceiling_s(inputs: WindowInputs) -> float:
    """``(budget - F) / rho_d``: the wait a request meets behind a decode window."""
    if inputs.decode_utilisation <= 0.0:
        return float("inf")
    return (float(inputs.ttft_budget_s) - float(inputs.flip_cost_s)) / float(
        inputs.decode_utilisation
    )


def batch_floor_s(inputs: WindowInputs) -> float:
    """Cycle length needed for the prefill window to form a decode batch.

    #689: entering decode below the queue threshold buys batch-size 1 for a full
    flip cost. The prefill window must therefore last long enough to accumulate
    ``min_decode_queue`` arrivals, i.e. ``T_p >= q/lambda`` and
    ``C = T_p/rho_p``.
    """
    if inputs.min_decode_queue <= 0:
        return 0.0
    if inputs.arrival_rate_per_s <= 0.0 or inputs.prefill_utilisation <= 0.0:
        raise PhaseWindowError(
            "a decode-batch floor needs a positive arrival rate and prefill "
            "utilisation; with neither, the queue never fills and the flip is "
            "never worth entering."
        )
    return (
        float(inputs.min_decode_queue)
        / float(inputs.arrival_rate_per_s)
        / float(inputs.prefill_utilisation)
    )


def solve_window(inputs: WindowInputs) -> WindowPolicy:
    """The economic cycle: the LARGEST admissible one, floored by stability."""
    if not inputs.seam_can_arm:
        return WindowPolicy(
            feasible=False,
            refusal=(
                "the seam cannot arm under this layout, so there is no flip to "
                "schedule at any window length. Fix the layout (its free column "
                "must clear the arming floor) before asking how long to stay in "
                "a phase."
            ),
            cycle_s=0.0,
            prefill_window_s=0.0,
            decode_window_s=0.0,
            stability_floor_s=float("nan"),
            latency_ceiling_s=float("nan"),
            batch_floor_s=float("nan"),
            flip_overhead_fraction=float("nan"),
            binding_constraint="seam",
        )

    floor = stability_floor_s(inputs)
    ceiling = latency_ceiling_s(inputs)
    bfloor = batch_floor_s(inputs)
    hard_floor = max(floor, bfloor)
    binding = "stability" if floor >= bfloor else "batch-formation"

    if ceiling < hard_floor:
        return WindowPolicy(
            feasible=False,
            refusal=(
                f"no admissible window: the {binding} floor is {hard_floor:,.1f} s "
                f"but the latency ceiling is {ceiling:,.1f} s. A flip costing "
                f"{inputs.flip_cost_s:,.1f} s raises the floor and lowers the "
                "ceiling at the same time, so the band shuts from both ends. "
                "Reduce the flip cost, shed load, or relax the budget -- no split "
                "of the windows rescues this."
            ),
            cycle_s=0.0,
            prefill_window_s=0.0,
            decode_window_s=0.0,
            stability_floor_s=floor,
            latency_ceiling_s=ceiling,
            batch_floor_s=bfloor,
            flip_overhead_fraction=float("nan"),
            binding_constraint=binding,
        )

    cycle = ceiling if ceiling != float("inf") else hard_floor
    overhead = inputs.flips_per_cycle * float(inputs.flip_cost_s) / cycle
    return WindowPolicy(
        feasible=True,
        refusal=None,
        cycle_s=cycle,
        prefill_window_s=float(inputs.prefill_utilisation) * cycle,
        decode_window_s=float(inputs.decode_utilisation) * cycle,
        stability_floor_s=floor,
        latency_ceiling_s=ceiling,
        batch_floor_s=bfloor,
        flip_overhead_fraction=overhead,
        binding_constraint="latency" if ceiling != float("inf") else binding,
    )
