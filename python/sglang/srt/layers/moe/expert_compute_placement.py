# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Link-proportional COMPUTE placement for cold MoE experts (#394 slice 3).

What the earlier slices did, and what they left on the table
------------------------------------------------------------
Slice 2 moved cold-expert BYTE OWNERSHIP: a rank's own cold experts are
apportioned over the group's shared DRAM segments in proportion to each rank's
measured host->device bandwidth. It did not move a single expert's COMPUTE. A
rank that must run expert ``e`` still pulls ``e`` across its OWN link, whether
the bytes started in its private pinned pool or in a peer's segment, so
per-rank H2D volume is unchanged by construction (`docs/dev/BENCH_394_...`).

The lever that is left is the assignment itself. Under the #82 GGUF expert-dim
shard the ranks own DISJOINT expert ranges and the owner is the rank that
executes the expert -- foreign ids remap to a zero padding expert and the
existing TP all-reduce sums the disjoint contributions
(``fused_moe_triton/layer.py``, ``forward_local``). The width of that range per
rank is nothing but the "moe" family vector, i.e. ``--rank-moe-ratio``. So
moving compute link-proportionally does not need a new mechanism; it needs the
right vector, solved from the same measured provenance chain slice 2 already
resolves.

Measured on the reference rig (2026-08-02 battery, DeepSeek-V4-Flash UD-IQ3_XXS,
TP=3, `docs/dev/BENCH_394_v4flash_club3090.md`): per-rank H2D shares were
42.1 / 28.9 / 29.0 % against link shares of 42.0 / 18.8 / 39.1 %. The x4-slotted
rank is the clock at 1.40x the group mean, and equalising the transfer times is
worth 1.54x on the transfer term for a bench-length generation (1.36x on the
short-probe mix).

The solve
---------
Two quantities per rank, both already known at launch:

* ``b_r`` -- this rank's share under the CURRENT plan (the base ``--rank-tp-ratio``
  vector, or an explicit ``--rank-moe-ratio``), normalised to sum 1;
* ``f_r`` -- this rank's MoE resident fraction (``--rank-moe-resident-fraction``),
  the GPU-resident part of its own expert shard;
* ``l_r`` -- this rank's link weight from the #394 provenance chain
  (env > measured card probe > NVML nameplate > refusal), normalised to sum 1.

The expert mass splits into a RESIDENT part that costs VRAM and never crosses
the link, and a COLD part that crosses the link on every activation. Only the
cold part is on the clock. So the placement is:

    resident_r = f_r * b_r                     # held FIXED -- VRAM-neutral
    cold_total = 1 - sum_r resident_r
    u          = normalise(l_r / c_r)          # per-unit streaming cost
    share_r    = resident_r + cold_total * u_r  # cold mass follows the links

and the installed vector is the smallest integer weight vector that represents
``share`` to a hundredth of a percentage point.

``c_r`` is the per-rank COLD-TRAFFIC COEFFICIENT: bytes actually streamed per
unit of non-resident expert mass. It is 1.0 everywhere -- the first-order model
"a cold expert is fetched, a resident one is not" -- and that is what
``--rank-moe-ratio link`` solves with. A prior boot's #390 dump can calibrate it
(see :func:`cold_traffic_coefficients_from_measurement`), because the
reference-rig battery showed the first-order model has a residual: measured H2D
shares were 42.1 / 28.9 / 29.0 % where ``b * (1 - f)`` predicts
37.2 / 26.8 / 36.0 %, i.e. the ranks re-fetch at materially different rates.

CALIBRATION IS FALSIFIED ON HARDWARE, and it is a separate, experimental symbol
(``--rank-moe-ratio link-calibrated``) rather than an implicit upgrade of
``link``. The 2026-08-03 confirmation window measured both arms on the reference
recipe and the calibrated one LOST -- on its END-TO-END leg, which is the
economically decisive one: -0.94 % against -7.67 %, i.e. inside the same-window
A-vs-A floor (CV 2.12 %, spread 4.09 %) and therefore indistinguishable from the
baseline. The transfer term is explicitly NOT a leg of the rejection: the
"1.439x against 1.496x" that sentence used to carry divided two counters sampled
at different fractions of their runs, and work-matched -- both arms read at a
common work point, the rule of #482, enforced by ``read_arm.py --against``
since #523 -- the same window gives the calibrated arm 1.4573x against plain
``link``'s 1.4253x, so it slightly WINS that term. The cause is the assumption
the coefficient rests on, and it is the
falsifier ``ARM3_COMPUTE.md`` named in advance: a coefficient treats the hit rate
as a property of the RANK, and it is not -- it tracks the SIZE of the owned
expert range, because a smaller range fits the cache better. In that window tp1's
hit rate rose 0.8450 -> 0.9050 as its range shrank 72 -> 58 experts, while tp2's
fell 0.8474 -> 0.7814 as its range grew 72 -> 89. The calibrated solve read tp2's
below-average coefficient (0.9586) as spare capacity, moved three more experts
onto it, each cost more traffic than modelled, and tp2 became the new clock.
A range-size-aware hit-rate model would be needed before calibration is worth
its complexity; a single-shot coefficient is not a property of the rig.
Evidence: ``/spinning/gpu-battery-results/2026-08-03_439_confirm/RESULTS.md``
§6, and ``planner/rejected.py`` key ``moe_link_calibrated_coefficients``.

Holding ``resident_r`` fixed is the whole reason this is safe to install: every
rank keeps EXACTLY the resident expert count it has today, so no card's VRAM
ledger moves, and only the streamed remainder is redistributed. A solve that
simply set ``share_r = l_r`` would minimise the same objective but would also
move resident weight onto whichever card has the fattest slot, which on a
heterogeneous rig is not the card with the spare VRAM.

Three properties fall out, and all three are pinned by tests:

* ``f_r == 1`` everywhere (offload OFF) => there is no cold mass at all, no
  link can matter, and the launch is REFUSED by name rather than resolved to a
  no-op that would serve the baseline under the treatment's name.
* ``l == b`` (the plan already matches the links) => ``share == b``: identity,
  and today's behaviour field for field. This is the identity condition, NOT
  "uniform links": on uniform links over an UNEVEN plan the solve equalises the
  cold mass, which is the correct optimum and a real change. Byte-identical
  defaults come from the flag being opt-in, not from the solve being a no-op.
* ``f_r -> 0`` (everything streamed) => ``share -> l``: ANALYSE_393's Path A'
  in its pure form.

The objective it optimises is the slowest-rank rule: with cold mass
``c_r = cold_total * l_r`` and uniform routing density over the expert set, the
per-rank transfer time is ``c_r / l_r``, which is the same constant on every
rank. That is the min-max point; no other split of a fixed cold mass has a
smaller maximum.

What is deliberately NOT modelled
---------------------------------
Nothing in this module is fitted to any rig, and nothing estimates a quantity
it was not given. In particular there is no model of the offload cache's
eviction behaviour: the residual it produces enters only through a coefficient
measured on a real boot, and an uncalibrated solve reports itself as
uncalibrated rather than quietly assuming the residual is zero.

Scope
-----
Single-node, pure TP, expert-dim-sharded MoE. Expert parallelism
(``moe_ep_size > 1``) has its own dispatch and is refused by name: the vector
here indexes the TP group, and installing it on a differently shaped MoE group
would attribute one rank's card to another.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "COMPUTE_BASE_PLAN_ENV",
    "COMPUTE_PLACEMENT_LINK",
    "COMPUTE_PLACEMENT_LINK_CALIBRATED",
    "COMPUTE_PLACEMENT_SYMBOLS",
    "COMPUTE_POLICY_ENV",
    "TRAFFIC_COEFFICIENT_ENV",
    "ComputePlacement",
    "NoComputeLever",
    "cold_traffic_coefficients_from_measurement",
    "compute_base_plan",
    "compute_policy_label",
    "resident_fraction_held_at_base_plan",
    "resolve_moe_compute_placement_flag",
    "solve_link_proportional_expert_vector",
    "vram_neutral_resident_fraction",
]

#: The symbolic value ``--rank-moe-ratio`` accepts in place of a vector. This
#: is the SHIPPED slice-3 solve: uncalibrated, first-order traffic model, no
#: prior boot required. It is the only one of the two that has beaten its own
#: baseline above a same-window floor on hardware (1.4307x transfer term
#: work-matched, -6.42 % end-to-end, 2026-08-03 green-corridor window).
COMPUTE_PLACEMENT_LINK = "link"

#: The EXPERIMENTAL calibrated variant, kept because the mechanism is sound and
#: a better hit-rate model could revive it, and named separately because it is
#: falsified as it stands (see the module docstring). It requires
#: :data:`TRAFFIC_COEFFICIENT_ENV`; ``link`` never reads that variable, so a
#: leftover export cannot turn the shipped solve into the falsified one behind
#: the operator's back.
COMPUTE_PLACEMENT_LINK_CALIBRATED = "link-calibrated"

#: Every symbolic value the flag accepts, for the parser and the validators.
COMPUTE_PLACEMENT_SYMBOLS = (
    COMPUTE_PLACEMENT_LINK,
    COMPUTE_PLACEMENT_LINK_CALIBRATED,
)

#: How a worker learns WHICH policy produced the explicit vector it was handed.
#: The launcher resolves the symbol once and every worker inherits the numbers,
#: so without this channel a #390 dump could not tell a solved vector from one
#: an operator typed -- and an A/B arm that cannot identify itself is an A/B
#: whose null result was never tested. Same channel and same reason as the
#: rank->card vector (#407 cut 2) and the cold-tier launch id (#394 slice 2).
COMPUTE_POLICY_ENV = "SGLANG_MOE_COMPUTE_POLICY"

#: The "moe" family vector the solve held the RESIDENT mass against, i.e. the
#: plan the boot would have run without the flag. Published by the launcher
#: next to :data:`COMPUTE_POLICY_ENV` and read by the residency sizing in every
#: worker; see :func:`resident_fraction_held_at_base_plan` for why residency
#: cannot be sized off the installed vector.
COMPUTE_BASE_PLAN_ENV = "SGLANG_MOE_COMPUTE_BASE_PLAN"

#: Optional per-rank cold-traffic coefficients, comma-separated, mean-1. The
#: operator computes them from a PRIOR boot's #390 dump with
#: :func:`cold_traffic_coefficients_from_measurement`; there is deliberately no
#: automatic path from a dump into a launch, because a coefficient measured on
#: one recipe is not a property of the rig.
TRAFFIC_COEFFICIENT_ENV = "SGLANG_MOE_COLD_TRAFFIC_COEFFICIENTS"

#: Largest denominator the integer vector may use. The search below stops at
#: the SMALLEST denominator that represents the solved shares to
#: :data:`_VECTOR_TOLERANCE`, so this is a ceiling, not the scale itself.
_VECTOR_SCALE = 10000

#: How far the installed integer vector's shares may sit from the exact solve,
#: as an absolute share difference. 1e-4 is a hundredth of a percentage point
#: of the expert set -- below one expert on any checkpoint anyone runs, so the
#: rounding cannot change a single expert's owner.
_VECTOR_TOLERANCE = 1e-4


class NoComputeLever(ValueError):
    """The requested placement cannot move anything, and says why.

    A ValueError rather than a silent fall back to the base plan: an operator
    who asked for link-proportional compute placement is running an experiment,
    and quietly serving the baseline under the treatment's name is how an A/B
    reports a null that was never tested.
    """


@dataclass(frozen=True)
class ComputePlacement:
    """The solved vector plus everything needed to argue with it.

    ``weights`` is what gets installed as ``--rank-moe-ratio``. The three share
    vectors are kept because the integer rounding is lossy and the predictions
    the proof window is read against (per-rank cold mass, hence per-rank H2D)
    live in the exact shares.
    """

    weights: Tuple[int, ...]
    base_shares: Tuple[float, ...]
    resident_shares: Tuple[float, ...]
    cold_shares: Tuple[float, ...]
    shares: Tuple[float, ...]
    link_shares: Tuple[float, ...]
    link_source: str
    link_provenance: str
    link_detail: str = ""
    #: Bytes streamed per unit of non-resident expert mass, per rank. All ones
    #: when nothing calibrated it.
    traffic_coefficients: Tuple[float, ...] = ()

    @property
    def is_identity(self) -> bool:
        """True when the solve reproduces the plan it started from.

        Compared on the ROUNDED vector, because that is what gets installed:
        a share difference too small to survive the integer rounding changes
        nothing that any rank can observe.
        """
        return _reduce(self.weights) == _integer_vector(self.base_shares)

    @property
    def is_calibrated(self) -> bool:
        """True when a measurement, not the first-order model, set the shares."""
        return bool(self.traffic_coefficients) and any(
            abs(c - 1.0) > 1e-9 for c in self.traffic_coefficients
        )

    def predicted_transfer_shares(self) -> Tuple[float, ...]:
        """Per-rank transfer TIME share, normalised so the group sums to 1.

        This is the readout the proof window compares arms on. At the optimum
        every entry equals ``1/world``; the current plan's entries are the
        thing the x4 rank's 1.40x came from.
        """
        coefficients = self.traffic_coefficients or tuple(1.0 for _ in self.cold_shares)
        times = [
            (cold * coeff / link if link > 0 else float("inf"))
            for cold, coeff, link in zip(
                self.cold_shares, coefficients, self.link_shares
            )
        ]
        total = sum(times)
        if not total or total == float("inf"):
            return tuple(0.0 for _ in times)
        return tuple(t / total for t in times)

    def describe(self) -> str:
        vec = ",".join(str(w) for w in self.weights)
        cold = ", ".join(f"rank{i}={c:.4f}" for i, c in enumerate(self.cold_shares))
        link = ", ".join(f"rank{i}={w:.4f}" for i, w in enumerate(self.link_shares))
        tail = f" ({self.link_detail})" if self.link_detail else ""
        if self.is_calibrated:
            calib = "; traffic coefficients " + ", ".join(
                f"rank{i}={c:.4f}" for i, c in enumerate(self.traffic_coefficients)
            )
        else:
            calib = (
                "; UNCALIBRATED traffic model (a cold expert is fetched, a "
                "resident one is not) -- feed a prior boot's per-rank H2D "
                "through cold_traffic_coefficients_from_measurement to remove "
                "the residual"
            )
        return (
            f"--rank-moe-ratio {vec} "
            f"[link source={self.link_source} provenance={self.link_provenance}"
            f"{tail}; link shares {link}; predicted cold-mass shares {cold}"
            f"{calib}; resident shares held at the current plan]"
        )


def _normalize(values: Sequence[float], what: str) -> Tuple[float, ...]:
    out = [float(v) for v in values]
    for i, v in enumerate(out):
        if not (v > 0.0) or v != v or v == float("inf"):
            raise ValueError(
                f"{what}: entry {i} is {v!r}; every entry must be a finite "
                "positive number"
            )
    total = sum(out)
    return tuple(v / total for v in out)


def _integer_vector(
    shares: Sequence[float],
    max_scale: int = _VECTOR_SCALE,
    tolerance: float = _VECTOR_TOLERANCE,
) -> Tuple[int, ...]:
    """Smallest integer weight vector that represents ``shares`` to ``tolerance``.

    The obvious implementation -- largest-remainder rounding onto a fixed large
    denominator -- is wrong here in a way that matters. Three equal shares onto
    a denominator of 10000 give ``3334,3333,3333``: an asymmetry invented by the
    rounding, on a group where the solve said there is none, and one that the
    layer would then turn into genuinely unequal expert ranges. Searching for
    the smallest denominator instead returns ``1,1,1``, and the general property
    is that a vector expressible exactly in small integers is returned in small
    integers.

    Deterministic and pure: the search order is fixed, so every process that
    calls this with the same shares gets the same vector without a collective.
    """
    n = len(shares)
    if n == 0:
        return ()
    for scale in range(n, int(max_scale) + 1):
        candidate = [max(1, int(round(s * scale))) for s in shares]
        total = sum(candidate)
        error = max(abs(c / total - s) for c, s in zip(candidate, shares))
        if error <= tolerance:
            return _reduce(candidate)
    # Unreachable for any well-formed share vector; a largest-remainder split on
    # the ceiling is still an exact partition, so fall back rather than raise.
    return _reduce(_largest_remainder(shares, int(max_scale)))


def _largest_remainder(shares: Sequence[float], scale: int) -> Tuple[int, ...]:
    """Round shares to integers summing to ``scale``, every entry >= 1.

    Largest remainder (Hamilton) rather than repeated rounding, for the same
    reason ``plan_proportional_shares`` uses it: the result is a pure function
    of its inputs, so the launcher and any process that re-derives the vector
    agree without talking. Ties go to the lower rank index.
    """
    n = len(shares)
    if n == 0:
        return ()
    exact = [max(0.0, float(s)) * scale for s in shares]
    floors = [max(1, int(math.floor(x))) for x in exact]
    remaining = scale - sum(floors)
    if remaining > 0:
        order = sorted(range(n), key=lambda i: (-(exact[i] - math.floor(exact[i])), i))
        for i in order[:remaining]:
            floors[i] += 1
    elif remaining < 0:
        # Only reachable when the >= 1 clamp inflated the sum, i.e. a share so
        # small it rounds to nothing. Take the excess back off the largest
        # entries, never below 1.
        order = sorted(range(n), key=lambda i: (-floors[i], i))
        idx = 0
        while remaining < 0 and any(f > 1 for f in floors):
            i = order[idx % n]
            if floors[i] > 1:
                floors[i] -= 1
                remaining += 1
            idx += 1
    return tuple(floors)


def _reduce(weights: Sequence[int]) -> Tuple[int, ...]:
    """gcd-reduce, so two vectors that mean the same split compare equal."""
    values = [int(w) for w in weights]
    if not values:
        return ()
    g = 0
    for v in values:
        g = math.gcd(g, v)
    if g <= 1:
        return tuple(values)
    return tuple(v // g for v in values)


def cold_traffic_coefficients_from_measurement(
    base_plan: Sequence[float],
    resident_fractions: Sequence[float],
    measured_h2d: Sequence[float],
) -> Tuple[float, ...]:
    """Bytes streamed per unit of non-resident expert mass, from a real boot.

    ``measured_h2d`` is one arm's per-rank H2D total (any consistent unit --
    only ratios are used), taken from the #390 dump's ``h2d_bytes``.
    ``base_plan`` and ``resident_fractions`` are the ones THAT boot ran, not the
    ones the next boot will.

    Why this exists: the first-order model says a rank streams its non-resident
    mass and nothing else, so its H2D share should be ``b_r (1 - f_r)``. On the
    2026-08-02 battery it was not -- the ranks re-fetch at different rates. This
    turns that residual into a number instead of leaving it in the error term of
    a prediction. A coefficient of 1.0 everywhere means the model was right.

    Pure, and it refuses rather than extrapolates: a rank with no non-resident
    mass in the measured boot has no ratio to report.
    """
    n = len(base_plan)
    if len(resident_fractions) != n or len(measured_h2d) != n:
        raise ValueError(
            f"vector lengths disagree: base_plan has {n} entries, "
            f"resident_fractions {len(resident_fractions)}, measured_h2d "
            f"{len(measured_h2d)}"
        )
    b = _normalize(base_plan, "base plan")
    cold = []
    for r in range(n):
        f = float(resident_fractions[r])
        if not (0.0 < f <= 1.0):
            raise ValueError(
                f"resident fraction for rank {r} is {f}; it must be in (0, 1]"
            )
        mass = b[r] * (1.0 - f)
        if mass <= 0.0:
            raise ValueError(
                f"rank {r} held its whole shard resident in the measured boot "
                "(resident fraction 1.0), so that boot says nothing about what "
                "it costs that rank to stream an expert"
            )
        cold.append(mass)
    h2d = _normalize(measured_h2d, "measured H2D")
    cold_shares = _normalize(cold, "non-resident mass")
    raw = [h2d[r] / cold_shares[r] for r in range(n)]
    mean = sum(raw) / n
    # Normalised to mean 1 so the coefficients read as "this rank streams x
    # times as much per owned cold expert as the group average", and so a
    # calibrated solve on a group that turns out to be uniform is the
    # uncalibrated solve exactly.
    return tuple(v / mean for v in raw)


def solve_link_proportional_expert_vector(
    base_plan: Sequence[float],
    resident_fractions: Sequence[float],
    link_weights: Sequence[float],
    traffic_coefficients: Optional[Sequence[float]] = None,
    link_source: str = "?",
    link_provenance: str = "?",
    link_detail: str = "",
) -> ComputePlacement:
    """Solve the "moe" family vector so cold-expert mass follows the links.

    Pure: no environment, no NVML, no torch. Every caller that wants the
    resolved-from-the-rig version goes through
    :func:`resolve_moe_compute_placement_flag`, which supplies these vectors
    and nothing else.

    ``traffic_coefficients`` defaults to all ones (the first-order model). When
    supplied -- normally from
    :func:`cold_traffic_coefficients_from_measurement` -- the cold mass is
    apportioned by ``l_r / c_r`` instead of ``l_r``, which is the point at which
    every rank's MEASURED streaming cost, not its modelled one, is equalised.

    Raises :class:`NoComputeLever` when the inputs leave nothing to move.
    """
    n = len(base_plan)
    if n == 0:
        raise ValueError("base_plan is empty; there is no group to place")
    if len(resident_fractions) != n or len(link_weights) != n:
        raise ValueError(
            f"vector lengths disagree: base_plan has {n} entries, "
            f"resident_fractions {len(resident_fractions)}, link_weights "
            f"{len(link_weights)}; all three index the same TP group"
        )
    if traffic_coefficients is None:
        coefficients = tuple(1.0 for _ in range(n))
    else:
        if len(traffic_coefficients) != n:
            raise ValueError(
                f"traffic_coefficients has {len(traffic_coefficients)} entries "
                f"but the TP group has {n}"
            )
        coefficients = tuple(float(c) for c in traffic_coefficients)
        for i, c in enumerate(coefficients):
            if not (c > 0.0) or c != c or c == float("inf"):
                raise ValueError(
                    f"traffic coefficient for rank {i} is {c!r}; every entry "
                    "must be a finite positive number"
                )

    b = _normalize(base_plan, "base plan")
    link = _normalize(link_weights, "link weights")
    # Per-unit streaming cost: a rank that streams twice as many bytes per owned
    # cold expert must receive half as many of them for the same link.
    u = _normalize(
        [link[r] / coefficients[r] for r in range(n)], "link/traffic weights"
    )

    f: List[float] = []
    for i, value in enumerate(resident_fractions):
        v = float(value)
        if not (0.0 < v <= 1.0):
            raise ValueError(
                f"resident fraction for rank {i} is {v}; it must be in (0, 1] "
                "-- 1.0 means the whole shard stays on the GPU and 0.0 could "
                "not run"
            )
        f.append(v)

    resident = tuple(f[r] * b[r] for r in range(n))
    cold_total = 1.0 - sum(resident)
    if cold_total <= 1e-9:
        raise NoComputeLever(
            "link-proportional expert compute placement has nothing to move: "
            f"the resident fractions {[round(x, 4) for x in f]} keep the whole "
            "expert set on the GPUs, so no expert crosses a PCIe link and the "
            "link widths cannot change any rank's transfer time. Enable expert "
            "offload (--rank-moe-resident-fraction < 1) or drop the flag."
        )

    shares = tuple(resident[r] + cold_total * u[r] for r in range(n))
    cold = tuple(cold_total * u[r] for r in range(n))
    weights = _integer_vector(shares)

    return ComputePlacement(
        weights=weights,
        base_shares=b,
        resident_shares=resident,
        cold_shares=cold,
        shares=shares,
        traffic_coefficients=coefficients,
        link_shares=link,
        link_source=link_source,
        link_provenance=link_provenance,
        link_detail=link_detail,
    )


def compute_policy_label() -> str:
    """What produced the installed vector, for the #390 dump. Never raises."""
    return os.environ.get(COMPUTE_POLICY_ENV, "").strip() or "base-plan"


# ===========================================================================
# VRAM neutrality, and the fixed point the first battery walked into (#439).
#
# The solve above holds ``resident_r = f_r * b_r`` FIXED -- that invariant is
# the reason the arm is installable without re-deriving the VRAM ledger, and
# it is asserted on the solve's own output. The RUNTIME did not honour it.
# Residency is sized from the rank's own shard extent
# (``expert_offload.plan_load_time_staging``: ``R = resident_slot_count(E,
# f)``), and the installed vector is exactly what changes that extent. So the
# realised resident mass was ``f_r * share_r``, not ``f_r * b_r``: on the
# 2026-08-02 reference recipe the solved vector 160,79,119 raised tp2's
# resident expert mass by 19.5 % over a card the baseline already left ~515 MiB
# free, and the boot died in ``_ensure_tiers -> allocate``.
#
# THE FIXED POINT. The obvious correction -- lower the resident FRACTION until
# the mass comes back -- feeds straight back into the solve, because the solve
# reads the fractions as an input. The night battery measured that: rescaled
# fractions 0.4812,0.5296,0.3516 produced 161,91,113 instead of 160,79,119, a
# placement for which the correction just computed is already wrong. Iterating
# to a fixed point is not in the spec and would make the installed vector a
# function of a numerical process rather than of the rig.
#
# THE DECISION, and it breaks the loop by construction rather than solving it:
#
#   * The SOLVE keeps reading the fractions the OPERATOR set, against the BASE
#     plan. Its inputs are untouched, so 160,79,119 is the vector, once.
#   * Residency is then held at the PRE-LINK baseline in the runtime: rank r
#     keeps exactly the resident slots ``resident_slot_count(base_extent_r,
#     f_r)`` gives it, sized off the base plan the launcher published here.
#   * That correction is expressed as a DERIVED per-rank fraction, and it is
#     carried on a channel the solve does not read. Nothing feeds back; there
#     is no iteration and no second solve.
#
# The derived fraction is a SIZING quantity only. ``--rank-moe-resident-
# fraction`` and ``SGLANG_MOE_RESIDENT_EXPERT_FRACTION`` keep meaning exactly
# what they meant -- the operator's fraction of the BASE shard -- which is also
# what keeps the treatment arm comparable with the baseline arm that shares it.
# ===========================================================================


def compute_base_plan(world: int) -> Optional[Tuple[int, ...]]:
    """The pre-link "moe" family vector, or ``None`` when link mode is off.

    Never raises and never guesses: a malformed or wrong-length value reads as
    absent, which restores today's sizing exactly. The channel only ever
    carries a vector the launcher itself wrote.
    """
    raw = os.environ.get(COMPUTE_BASE_PLAN_ENV, "").strip()
    if not raw:
        return None
    try:
        values = tuple(int(p) for p in raw.replace(";", ",").split(",") if p.strip())
    except ValueError:
        return None
    if len(values) != int(world) or any(v <= 0 for v in values):
        return None
    return values


def vram_neutral_resident_fraction(
    fraction: float,
    base_experts: int,
    local_experts: int,
    base_width: int = 1,
    local_width: int = 1,
) -> float:
    """The fraction that reproduces the BASE plan's resident bytes. Pure.

    Resident bytes on a rank are ``R * width``: ``R`` resident expert slots of
    ``width`` elements each. Under the #82 expert-dim shard the vector moves
    ``R``'s domain (the rank's expert count) and leaves ``width`` alone; on
    every other MoE path it moves ``width`` and leaves the expert count alone.
    One expression covers both, because the invariant is the product:

        target_slots = R_base * base_width / local_width

    The returned fraction is the midpoint of the interval that ``ceil`` maps
    onto ``round(target_slots)``, i.e. ``(target - 0.5) / local_experts``.
    The midpoint rather than ``target / local_experts`` because the latter does
    not survive its own round trip in binary floating point -- 56/114 times 114
    is 56.000000000000014, whose ceiling is 57, which would put one extra
    expert on the card this function exists to protect.

    Clamped into ``[1, local_experts]`` slots. The upper clamp is reachable
    only if a rank was solved BELOW its own resident mass, which the solve
    forbids (``share_r = resident_r + cold_total * u_r`` with both terms
    positive) except through integer rounding. It is not silent: the caller
    warns by name, because a rank in that state holds LESS than its baseline
    and the arm is no longer VRAM-neutral for it.
    """
    f = float(fraction)
    if f >= 1.0:
        return f
    from sglang.srt.layers.moe.expert_offload import resident_slot_count

    e_base = max(1, int(base_experts))
    e_local = max(1, int(local_experts))
    target = resident_slot_count(e_base, f) * float(base_width) / float(local_width)
    target = min(max(target, 1.0), float(e_local))
    return (target - 0.5) / e_local


def resident_fraction_held_at_base_plan(
    fraction: float,
    *,
    num_experts: int,
    num_local_experts: int,
    moe_tp_size: int,
    moe_tp_rank: int,
    expert_sharded: bool,
    intermediate_size: Optional[int] = None,
    intermediate_units: Optional[int] = None,
) -> float:
    """``fraction``, corrected so this rank's resident bytes do not move.

    A no-op -- returning ``fraction`` unchanged, identity object and all --
    whenever link mode did not publish a base plan, the installed vector is the
    base plan, or offload is off. That is what keeps every launch that does not
    pass ``--rank-moe-ratio link`` byte-identical.

    ``expert_sharded`` selects which dimension the "moe" family actually
    partitions on this layer (see :func:`vram_neutral_resident_fraction`). The
    ``+ 1`` on the expert-shard counts is the #82 zero padding expert, which
    ``_gguf_owned_expert_count`` includes and which is resident on every rank.
    """
    f = float(fraction)
    if f >= 1.0:
        return f
    world = int(moe_tp_size)
    base = compute_base_plan(world)
    if base is None:
        return f

    from sglang.srt.distributed.utils import (
        get_tp_partition_ratios,
        partition_sizes,
        partition_units,
    )

    installed = get_tp_partition_ratios("moe")
    if not installed or len(installed) != world:
        return f
    if _reduce(list(base)) == _reduce(list(installed)):
        # The solve was the identity, or the operator installed the base plan
        # by hand. Nothing moved, so nothing is corrected.
        return f

    rank = int(moe_tp_rank)
    if expert_sharded:
        e_base = partition_units(int(num_experts), list(base))[rank] + 1
        e_local = partition_units(int(num_experts), list(installed))[rank] + 1
        w_base = w_local = 1
    else:
        if not intermediate_size or not intermediate_units:
            # Nothing to hold the width against; leave the fraction alone
            # rather than invent a geometry.
            return f
        e_base = e_local = int(num_local_experts)
        w_base = partition_sizes(
            int(intermediate_size), list(base), int(intermediate_units)
        )[rank]
        w_local = partition_sizes(
            int(intermediate_size), list(installed), int(intermediate_units)
        )[rank]

    from sglang.srt.layers.moe.expert_offload import resident_slot_count

    desired = resident_slot_count(e_base, f) * w_base / w_local
    corrected = vram_neutral_resident_fraction(f, e_base, e_local, w_base, w_local)
    if desired > e_local:
        logger.warning(
            "MoE residency cannot be held at the base plan on rank %d: the "
            "base plan gives it %.1f resident slots but the solved placement "
            "leaves it only %d slots to hold them in. This rank will hold "
            "LESS than its baseline, so the arm is not VRAM-neutral for it. "
            "The solve does not produce this (a rank's solved share is always "
            "its resident share plus a positive cold term); check for a "
            "hand-installed --rank-moe-ratio.",
            rank,
            desired,
            e_local,
        )
    if corrected != f:
        logger.info(
            "MoE resident fraction held at the base plan (#439): rank %d "
            "%.6f -> %.6f (base extent %d x %d, installed %d x %d). The "
            "operator's fraction always means a fraction of the BASE shard, "
            "so the resident bytes are the ones the VRAM ledger was solved "
            "for; only the STREAMED remainder follows the links.",
            rank,
            f,
            corrected,
            e_base,
            w_base,
            e_local,
            w_local,
        )
    return corrected


def _resident_fraction_vector(server_args, world: int) -> Tuple[float, ...]:
    """The per-rank resident fraction as the RUNTIME will read it.

    Deliberately re-derived from the same module the layers use rather than
    read off ``server_args`` directly: the flag may be a scalar that broadcasts,
    the environment variable is a second source, and the two are cross-checked
    rather than ranked. A solve that used a different fraction than the one the
    cache sizes itself from would be solving a different rig.

    ``server_args`` is handed DOWN into that module rather than left for it to
    find. This resolver runs in the launcher, before the arguments reach the
    runtime context, so the module's own flag source could not see them: #439's
    first battery attempt read ``[1.0, 1.0, 1.0]`` here while the launch
    carried ``--rank-moe-resident-fraction 0.485,0.42,0.42``, and the arm was
    refused as having "nothing to move". The cross-check against the
    environment variable is unaffected.
    """
    from sglang.srt.layers.moe.resident_fraction import resident_fraction_vector

    values = resident_fraction_vector(tp_size=world, server_args=server_args)
    if len(values) == 1 and world > 1:
        values = tuple(values) * world
    if len(values) != world:
        raise ValueError(
            f"the MoE resident-fraction vector has {len(values)} entries but "
            f"the TP group has {world}"
        )
    return tuple(float(v) for v in values)


def _traffic_coefficients_from_env(world: int) -> Optional[Tuple[float, ...]]:
    """The measured coefficients, or ``None`` when nobody supplied any.

    A malformed or wrong-length vector is a hard error, never a silent fall
    back to the uncalibrated model: an operator who typed coefficients took a
    measurement, and running a different solve than the one they asked for is
    how a measurement arm turns into a lie. Same contract as
    ``SGLANG_MOE_HOST_SHARD_RATIO``.
    """
    raw = os.environ.get(TRAFFIC_COEFFICIENT_ENV, "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    try:
        values = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(
            f"{TRAFFIC_COEFFICIENT_ENV}={raw!r} is not a comma-separated list "
            f"of positive floats ({exc})"
        ) from exc
    if len(values) != int(world):
        raise ValueError(
            f"{TRAFFIC_COEFFICIENT_ENV}={raw!r} has {len(values)} entries but "
            f"the TP group has {int(world)}; give exactly one per rank"
        )
    return tuple(values)


def _traffic_coefficients_for_symbol(
    symbol: str, world: int
) -> Optional[Tuple[float, ...]]:
    """The coefficients this symbol is allowed to solve with.

    The whole point of the split introduced after the 2026-08-03 window: the
    coefficients enter through the SYMBOL the operator typed, never through an
    environment variable that happens to still be exported. ``link`` is the
    uncalibrated solve by definition, and a set coefficient variable under it
    is refused rather than ignored -- an operator who took a measurement gets
    told which flag spends it, and an operator who left one exported from the
    previous arm does not silently run the falsified solve.
    """
    env_coefficients = _traffic_coefficients_from_env(world)
    if symbol == COMPUTE_PLACEMENT_LINK:
        if env_coefficients is not None:
            raise NoComputeLever(
                f"--rank-moe-ratio {COMPUTE_PLACEMENT_LINK} is the UNCALIBRATED "
                f"solve and does not read {TRAFFIC_COEFFICIENT_ENV}, but that "
                "variable is set. Calibration is a separate, experimental "
                f"symbol: pass --rank-moe-ratio "
                f"{COMPUTE_PLACEMENT_LINK_CALIBRATED} to spend those "
                "coefficients, or unset the variable to run the shipped solve. "
                "Read why before choosing: the calibrated solve is falsified on "
                "the reference recipe (2026-08-03, -0.94 % end to end inside "
                "the same-window 4.09 % floor, against -7.67 % uncalibrated) "
                "because a per-rank hit rate tracks the SIZE of the owned "
                "expert range, not the rank."
            )
        return None
    if env_coefficients is None:
        raise NoComputeLever(
            f"--rank-moe-ratio {COMPUTE_PLACEMENT_LINK_CALIBRATED} has nothing "
            f"to calibrate with: {TRAFFIC_COEFFICIENT_ENV} is unset. Derive one "
            "coefficient per rank from a PRIOR boot's #390 dump with "
            "cold_traffic_coefficients_from_measurement, or use "
            f"--rank-moe-ratio {COMPUTE_PLACEMENT_LINK}, which needs no prior "
            "boot and measured faster on the reference recipe."
        )
    logger.warning(
        "--rank-moe-ratio %s is EXPERIMENTAL and its modelling assumption is "
        "FALSIFIED on the reference recipe: a per-rank cold-traffic coefficient "
        "treats the hit rate as a property of the rank, and the 2026-08-03 "
        "window measured it tracking the SIZE of the owned expert range instead "
        "(tp1 0.8450 -> 0.9050 as its range shrank 72 -> 58; tp2 0.8474 -> "
        "0.7814 as its range grew 72 -> 89). The calibrated arm measured "
        "-0.94 %% end to end inside a 4.09 %% same-window floor, against "
        "-7.67 %% for plain '%s'. That end-to-end leg and the mechanism above "
        "are the whole rejection: work-matched, the calibrated arm reads "
        "1.4573x on the transfer term against 1.4253x and slightly WINS it, so "
        "the transfer term is not cited. Use it to test a better hit-rate "
        "model, not to serve.",
        COMPUTE_PLACEMENT_LINK_CALIBRATED,
        COMPUTE_PLACEMENT_LINK,
    )
    return env_coefficients


def resolve_moe_compute_placement_flag(server_args) -> Optional[ComputePlacement]:
    """Resolve ``--rank-moe-ratio link`` into an explicit vector, in the LAUNCHER.

    A no-op -- and no import of anything heavy -- unless a symbolic value is
    set. Returns the placement it installed, or ``None`` when there was nothing
    to resolve.

    Two symbols, and which one was typed is the ONLY thing that decides whether
    the solve is calibrated. ``link`` is the shipped, uncalibrated solve;
    ``link-calibrated`` is the experimental one and is refused without explicit
    coefficients. Before the 2026-08-03 window the distinction was implicit --
    ``link`` plus a set :data:`TRAFFIC_COEFFICIENT_ENV` silently produced the
    calibrated policy -- which is exactly the shape of accident that A/B window
    discipline exists to prevent, and it is now the falsified solve that the
    accident would install.

    ONE process resolves and every worker inherits the numbers through the
    pickled ``ServerArgs``. That is not an optimisation: the vector decides
    which rank executes which expert, so a per-worker re-derivation would put
    the group's expert coverage at the mercy of three independent NVML reads.
    Rank-uniformity here is structural, not checked.
    """
    symbol = getattr(server_args, "rank_moe_ratio", None)
    if symbol not in COMPUTE_PLACEMENT_SYMBOLS:
        return None

    world = int(getattr(server_args, "tp_size", 1) or 1)
    # Before anything that touches NVML or the card registry: which solve was
    # asked for is a property of the launch arguments alone, and getting it
    # wrong is the one failure that produces a plausible boot rather than an
    # error.
    coefficients = _traffic_coefficients_for_symbol(symbol, world)

    base_plan = getattr(server_args, "rank_tp_ratio", None)
    if not isinstance(base_plan, list) or len(base_plan) != world:
        # ServerArgs validation already refuses this; repeated here because the
        # launcher is a second entry point (an Engine built in-process).
        raise NoComputeLever(
            f"--rank-moe-ratio {symbol} needs a resolved uneven-TP base plan of "
            f"length {world} to hold the resident mass fixed against; got "
            f"--rank-tp-ratio {base_plan!r}"
        )

    ep_size = int(getattr(server_args, "ep_size", 1) or 1)
    if ep_size > 1:
        raise NoComputeLever(
            f"--rank-moe-ratio {symbol} is a TP-group placement, but ep_size="
            f"{ep_size} gives the MoE its own dispatch and its own group "
            "shape. The vector would index a different group than the one it "
            "was solved for; pass an explicit vector if that is intended."
        )

    from sglang.srt.layers.moe.expert_offload import (
        HOST_SHARD_RATIO_ENV,
        HOST_SHARD_SOURCE_ENV,
        resolve_host_shard_ratio,
    )
    from sglang.srt.registry.rank_cards import (
        rank_card_vector,
        resolve_rank_card_vector,
    )

    # The vector the launcher published a moment ago (#407 cut 2). Read rather
    # than recomputed so this resolution and the cold tier's cannot disagree,
    # and so a hand-set SGLANG_RANK_CARD_UUIDS reaches this path too. Falls back
    # to resolving for the in-process Engine, which has no publish step.
    cards = rank_card_vector(world)
    if not cards.present:
        cards = resolve_rank_card_vector(server_args)
    if not cards.present:
        raise NoComputeLever(
            f"--rank-moe-ratio {symbol} cannot be solved: the rank -> physical "
            f"card vector is not available ({cards.reason}). The link weights are "
            "keyed by card UUID, and guessing the mapping from CUDA "
            "enumeration order is the #392/#397 defect this chain exists to "
            "avoid."
        )

    ratio = resolve_host_shard_ratio(world, card_uuids=cards.uuids)
    if ratio.provenance == "absent":
        raise NoComputeLever(
            f"--rank-moe-ratio {symbol} cannot be solved: no link bandwidth of "
            f"usable provenance for these cards ({ratio.detail}). An equal "
            "ratio is the shape a refusal takes, not a placement -- run the "
            "card probe (python -m sglang.srt.rigmon.card_probe) or set "
            "SGLANG_MOE_HOST_SHARD_RATIO explicitly."
        )

    if ratio.source == HOST_SHARD_SOURCE_ENV and ratio.is_equal:
        # The one configuration trap this flag has. HOST_SHARD_RATIO_ENV does
        # double duty: it is the STRONGEST source of the link weights, and an
        # equal value is also how the cold tier is held at the baseline. A
        # measurement arm that pins it equal for the second reason silently
        # hands this solve a uniform link profile -- and the result is not the
        # base plan either, it is the cold-mass-equalising vector, which is a
        # third thing nobody asked for. Hold byte ownership at the baseline
        # with SGLANG_MOE_COLD_TIER_SHM instead.
        logger.warning(
            "--rank-moe-ratio %s is solving against an EQUAL link ratio that "
            "came from %s. That variable is both the link-weight source and "
            "the cold-tier baseline switch; if it was set to hold byte "
            "ownership at the baseline, unset it and use "
            "SGLANG_MOE_COLD_TIER_SHM for that instead -- otherwise this boot "
            "equalises the cold mass across ranks rather than following the "
            "links, which is neither the baseline nor the treatment.",
            symbol,
            HOST_SHARD_RATIO_ENV,
        )

    fractions = _resident_fraction_vector(server_args, world)
    placement = solve_link_proportional_expert_vector(
        base_plan,
        fractions,
        ratio.weights,
        traffic_coefficients=coefficients,
        link_source=ratio.source,
        link_provenance=ratio.provenance,
        link_detail=ratio.detail,
    )

    # The single audited post-resolution mutation point: the symbol was
    # resolvable only here, and the override records the provenance so a
    # republish of these arguments resolves the same vector.
    server_args.override(
        "moe compute placement (#394 slice 3)",
        rank_moe_ratio=list(placement.weights),
    )
    if placement.is_identity:
        policy = "link-identity"
    elif placement.is_calibrated:
        policy = "link-proportional-calibrated"
    else:
        policy = "link-proportional"
    os.environ[COMPUTE_POLICY_ENV] = policy
    # The plan the resident mass is held against, published on the same
    # environment channel and in the same process as the policy label, and for
    # the same reason: the workers must not re-derive it. Without it the
    # residency sizing in every worker would follow the vector it just
    # installed, and the arm would stop being VRAM-neutral (#439).
    os.environ[COMPUTE_BASE_PLAN_ENV] = ",".join(str(int(b)) for b in base_plan)
    if placement.is_identity:
        # Loud, because an identity resolution under a flag that was asked for
        # explicitly is nearly always a configuration accident rather than a
        # converged rig -- most often an equal SGLANG_MOE_HOST_SHARD_RATIO set
        # to hold byte ownership at the baseline, which is also the FIRST
        # source of the link weights this solve reads.
        logger.warning(
            "--rank-moe-ratio %s -> %s. The solve is the IDENTITY: this "
            "group's link shares already match its plan shares, so no expert "
            "changes owner and this boot is the base plan field for field. If "
            "that is not what you expected, check whether the link ratio came "
            "from an equal %s.",
            symbol,
            placement.describe(),
            HOST_SHARD_RATIO_ENV,
        )
    else:
        predicted = placement.predicted_transfer_shares()
        logger.info(
            "--rank-moe-ratio %s -> %s. Predicted per-rank transfer-time "
            "shares %s (the optimum is %.4f on every rank; the slowest rank is "
            "the clock).",
            symbol,
            placement.describe(),
            [round(p, 4) for p in predicted],
            1.0 / world,
        )
    return placement
