"""#690: what the fixed flip cost is MADE OF, and what each part is worth.

The flip has been carried as a scalar (~2.0-4.2 s) with an unexplained
residual. It is not a scalar: the runtime already reports a five-way split on
every completed flip, and reading 765 of those lines decomposes it.

The headline is that the part everyone reaches for first is the minority
share. ``read + exchange + write`` — the KV seam move — is about a third of the
flip. **Sixty-two percent sits outside it**, in two components:

* ``movers`` (the largest single term) is the pre-cutover leg:
  ``_build_gdn_leg`` plus ``stacks.refill``
  (``managers/phase_flip_runtime.py:1843-1846``), i.e. GDN state plus the
  **weights arena refill** — an H2D copy of the target layout's image;
* ``cutover`` is the switch itself: group routing, owner-rule refresh,
  component rebuild, scheduler swap (``build_production_flip_cutover``,
  ``:1147``).

The residual is only ~3%, so the "unexplained" part of the cost is small and
the named parts are the story.

**Why this module exists rather than a table.** #677 showed that flip cost sets
both the stability floor and the latency ceiling, in opposite directions, so a
reduction does not merely shave an overhead line — it **reopens (F, rho)
configurations that are refused outright**. A lever is therefore priced in
configurations reopened, not in milliseconds saved, and that conversion is what
:func:`reopened_load` performs.

No rig constants: measured components are inputs.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence


class FlipCostError(ValueError):
    """A flip-cost question that cannot be answered as posed."""


@dataclasses.dataclass(frozen=True)
class FlipComponents:
    """One flip's reported split, in milliseconds.

    Field names match the runtime's own log line so a reader can grep from the
    number back to the source.
    """

    read_ms: float
    exchange_ms: float
    write_ms: float
    movers_ms: float
    cutover_ms: float
    total_ms: float

    @property
    def residual_ms(self) -> float:
        """Total minus the named parts. Small means the split is complete."""
        return self.total_ms - (
            self.read_ms
            + self.exchange_ms
            + self.write_ms
            + self.movers_ms
            + self.cutover_ms
        )

    @property
    def kv_seam_ms(self) -> float:
        """read + exchange + write: the part usually mistaken for the whole."""
        return self.read_ms + self.exchange_ms + self.write_ms

    def share(self, component_ms: float) -> float:
        if self.total_ms <= 0.0:
            raise FlipCostError("total flip time must be positive.")
        return component_ms / self.total_ms


@dataclasses.dataclass(frozen=True)
class Lever:
    """A named reduction, the component it attacks, and what it costs."""

    name: str
    #: Components it removes or overlaps away, by field name.
    attacks: tuple[str, ...]
    #: Fraction of those components it removes (1.0 = deletes them).
    removes_fraction: float
    mechanism: str
    #: What it costs elsewhere. Empty string means "scheduling only".
    price: str

    def apply(self, c: FlipComponents) -> float:
        """Resulting total flip time in ms."""
        if not 0.0 <= self.removes_fraction <= 1.0:
            raise FlipCostError(
                f"{self.name}: removes_fraction must be in [0,1], got "
                f"{self.removes_fraction}."
            )
        saved = 0.0
        for field in self.attacks:
            if not hasattr(c, field):
                raise FlipCostError(f"{self.name}: no component named {field!r}.")
            saved += float(getattr(c, field))
        return float(c.total_ms) - saved * float(self.removes_fraction)


def overlap_lever(name: str, hide: str, behind: str, mechanism: str) -> Lever:
    """A lever that hides one component behind another rather than removing it.

    The saving is capped by the shorter of the two: overlapping a 1.5 s leg
    behind a 1.1 s one saves 1.1 s, not 1.5. Modelled as a fractional removal
    resolved against the actual pair at call time.
    """
    return Lever(
        name=name,
        attacks=(hide,),
        removes_fraction=1.0,
        mechanism=mechanism,
        price="scheduling only; requires the two legs to be independent",
    )


def apply_overlap(c: FlipComponents, hide: str, behind: str) -> float:
    """Total after hiding ``hide`` behind ``behind``, capped by the shorter."""
    a = float(getattr(c, hide))
    b = float(getattr(c, behind))
    return float(c.total_ms) - min(a, b)


def max_feasible_load(
    flip_cost_s: float,
    ttft_budget_s: float,
    decode_share: float,
    flips_per_cycle: int = 2,
    resolution: float = 0.01,
) -> float:
    """Largest total utilisation that admits ANY window, at this flip cost.

    Solves #677's two constraints against each other: the stability floor
    ``flips*F/(1-rho)`` must not exceed the latency ceiling
    ``(budget-F)/rho_d``. Because the floor rises and the ceiling falls with
    ``F``, this is the quantity a flip-cost reduction actually buys.

    ``decode_share`` is ``rho_d / rho`` — how the load splits between phases —
    held fixed while ``rho`` is swept, so the comparison is like for like.
    """
    if not 0.0 < decode_share <= 1.0:
        raise FlipCostError("decode_share must be in (0, 1].")
    if ttft_budget_s <= flip_cost_s:
        return 0.0
    best = 0.0
    rho = resolution
    while rho < 1.0:
        rho_d = rho * decode_share
        floor = flips_per_cycle * flip_cost_s / (1.0 - rho)
        ceiling = (ttft_budget_s - flip_cost_s) / rho_d if rho_d > 0 else float("inf")
        if ceiling >= floor:
            best = rho
        rho += resolution
    return best


def reopened_load(
    before_s: float,
    after_s: float,
    ttft_budget_s: float,
    decode_share: float,
    flips_per_cycle: int = 2,
) -> tuple[float, float]:
    """``(max load before, max load after)`` a lever is applied.

    The unit a flip-cost reduction should be argued in: not "saves 1.2 s" but
    "moves the feasible ceiling from rho=0.52 to rho=0.71".
    """
    return (
        max_feasible_load(before_s, ttft_budget_s, decode_share, flips_per_cycle),
        max_feasible_load(after_s, ttft_budget_s, decode_share, flips_per_cycle),
    )


def rank_levers(
    c: FlipComponents, levers: Sequence[Lever]
) -> tuple[tuple[Lever, float], ...]:
    """Levers sorted by resulting flip time, cheapest flip first."""
    scored = [(lv, lv.apply(c)) for lv in levers]
    scored.sort(key=lambda p: p[1])
    return tuple(scored)
