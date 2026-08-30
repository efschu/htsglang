"""#1012: planner-solved BAR1 receive windows, one per collective group.

``--barlink-bar1-window-mib 24,PP_0=96,FLIP_TP_0=48,FLIP_DCP_0=32`` is FIVE
hand-pins (the bare default plus four group overrides), and a fifth group
arrived on 2026-08-30 that nobody had budgeted for.

THE DEFECT CLASS THIS CLOSES IS NOT "the numbers are wrong" -- it is that
NOBODY CHOOSES THEM TOGETHER. ``barlink_matrix_transport.window_for`` sizes one
group at a time against what is free at the moment that group builds, and
charges it to a per-device ``_LEDGER``. So the aperture is allocated
FIRST-COME BY BUILD ORDER: whichever group builds last meets whatever the
earlier ones left. That is exactly how the first #704b boot died, and the
arithmetic is quoted from the refusal itself in ``boot_855_gdncov.sh:64-86``::

    world:0 23 + pp:0 95 + flip_tp:0 47 + flip_dcp:0 31 = 196 MiB already held
    NVML free 44 - reserve 32 = 12 MiB left      ->  decoupled_kv:0 got 8

and that boot script says so in its own words: the 8 is "gemessen-und-
eingepasst", measured-and-fitted, "das ist der GRUND fuer die 8, kein Optimum".
It also states the consequence this module has to answer: with five groups BAR1
is nearly exhausted (~204 MiB bound, single-digit MiB free), so "jeder weitere
Gruppenbau braucht erst einen Fenster-Solve".

WHY A SOLVER IS THE RIGHT ANSWER HERE AND NOT SECOND BOOKKEEPING (the
2026-08-29 upstream-minimal law: before repairing a fork mechanism, ask whether
it may exist at all). It may, and the existing layer is kept, not duplicated:

* upstream has no equivalent -- upstream never pins a BAR1 aperture per
  collective group, because it has no BAR1 direct path;
* this module does NOT re-implement ``window_for``. ``window_for`` stays the
  authority on what a group may take on a card at build time, keeps its
  ledger, and keeps refusing an explicit window that does not fit. This
  module only decides what to ASK for, once, for all groups together, before
  the first group builds -- it replaces an implicit first-come policy with an
  explicit one, which is the opposite of a second ledger.

WHAT A WINDOW BUYS, precisely -- it is a THROUGHPUT knob under a hard floor.
``barlink_bar1.a2a_rounds`` is ``max(1, ceil(largest_block / slot))``, and
``window_for``'s own warning states the cost of a smaller window: "all_reduce
and all_to_all decompose oversized payloads into rounds, so the direct path is
kept and pays in launches, not in coverage -- until a payload needs more than
the round cap allows, at which point the dispatcher warns (outside a capture)
or raises (inside one)". The caps are ``SGLANG_BARLINK_BAR1_AR_MAX_ROUNDS`` /
``_A2A_MAX_ROUNDS``, 16 each. So:

* above the cap the direct path is LOST (and inside a CUDA-graph capture it
  raises) -- that is the hard floor this solver refuses below;
* below the cap a smaller window costs launches, monotonically -- that is the
  objective this solver minimizes.

REFUSAL, NOT CLIPPING. If the aperture cannot even fund every group's
round-cap floor, this refuses at the desk and names the shortfall, mirroring
``Bar1WindowRefused``'s own reasoning: "an explicit window is an instruction,
and silently serving a smaller one would make every later size decision of
this group rest on a number nobody chose."

THE APERTURE IS THE GROUP MINIMUM, NOT THIS RANK'S. ``window_for`` is
explicitly "a local PROPOSAL, not the final decision ... ``_build_up`` takes
the minimum across the group", because a per-rank-different region would mean
a per-rank-different slot layout. On this rig that binding card is a 3080 at
256 MiB gross, never the 5090. Feed :func:`solve_bar1_windows` the MINIMUM
usable aperture across the group's ranks; passing rank 0's would silently
oversize every window.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Sequence, Tuple

MIB = 1024 * 1024

#: Round caps of the two decomposed operations. Above these the direct path is
#: declined outside a capture and RAISES inside one, so this is a hard floor
#: rather than a preference. Mirrors SGLANG_BARLINK_BAR1_AR_MAX_ROUNDS /
#: SGLANG_BARLINK_BAR1_A2A_MAX_ROUNDS in barlink_bar1.
DEFAULT_MAX_ROUNDS = 16

#: What barlink leaves untouched in BAR1 (RM's own occupancy is not in the
#: sysfs gross number). Mirrors barlink_matrix_transport.RESERVE_MIB_DEFAULT.
RESERVE_MIB_DEFAULT = 32


@dataclasses.dataclass(frozen=True)
class Bar1GroupDemand:
    """One collective group's claim on the aperture."""

    #: The live group name as barlink builds it, e.g. ``"pp:0"``. The env key
    #: is derived from it exactly as ``_group_key`` does, so the emitted flag
    #: reproduces the consumer's keys instead of inventing them.
    name: str
    #: Largest block this group will carry, in bytes -- the group-wide maximum
    #: over all R*R blocks, which is what ``a2a_rounds`` is denominated in.
    largest_block_bytes: int
    #: Where that number came from. Printed in the provenance line so a
    #: derived block and a measured one are never confused.
    basis: str = "derived"

    @property
    def env_key(self) -> str:
        """``pp:0`` -> ``PP_0``. Mirrors ``_group_key``."""
        return "".join(c if c.isalnum() else "_" for c in self.name).upper()

    def rounds_at(self, window_mib: int) -> int:
        """Rounds this group pays at a given window. ``ceil(block/slot)``."""
        if window_mib <= 0:
            raise ValueError(f"{self.name}: window must be positive")
        return max(1, -(-int(self.largest_block_bytes) // (window_mib * MIB)))

    def floor_mib(self, max_rounds: int) -> int:
        """Smallest window that keeps this group on the direct path."""
        need = -(-int(self.largest_block_bytes) // (max_rounds * MIB))
        return max(1, need)

    def ideal_mib(self) -> int:
        """Smallest window that carries the largest block in ONE round."""
        return max(1, -(-int(self.largest_block_bytes) // MIB))


@dataclasses.dataclass(frozen=True)
class Bar1WindowSolution:
    windows_mib: Tuple[Tuple[str, int], ...]
    rounds: Tuple[Tuple[str, int], ...]
    usable_mib: int
    allocated_mib: int
    #: Groups that reached one round, i.e. pay nothing for decomposition.
    single_round: Tuple[str, ...]
    basis: Tuple[Tuple[str, str], ...]

    def flag_value(self, default_mib: int) -> str:
        """The ``--barlink-bar1-window-mib`` string, in the flag's own form.

        A bare default followed by ``GROUP=VALUE`` overrides -- the shape the
        flag documents and the shape the boot scripts already carry, so a
        solved value is a drop-in for the hand-pinned one.
        """
        parts = [str(default_mib)]
        parts.extend(
            f"{key}={mib}"
            for key, mib in self.windows_mib
            if mib != default_mib
        )
        return ",".join(parts)

    def summary(self) -> str:
        per = ", ".join(
            f"{k}={m} MiB/{r} round(s)"
            for (k, m), (_, r) in zip(self.windows_mib, self.rounds)
        )
        return (
            f"{self.allocated_mib} of {self.usable_mib} usable MiB allocated; "
            f"{per}; {len(self.single_round)} of {len(self.windows_mib)} "
            f"group(s) at one round"
        )


class Bar1ApertureRefused(ValueError):
    """The aperture cannot fund every group's round-cap floor."""

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons: Tuple[str, ...] = tuple(reasons)
        super().__init__(
            "BAR1 aperture cannot carry this group set: " + "; ".join(reasons)
        )


def solve_bar1_windows(
    groups: Sequence[Bar1GroupDemand],
    usable_mib: int,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
) -> Bar1WindowSolution:
    """Size every group's BAR1 window at once, from predicted block sizes.

    ``usable_mib`` is the MINIMUM across the group's ranks of
    ``NVML free - reserve`` -- the same quantity ``window_for`` computes, taken
    at its binding card (see the module docstring).

    Two phases, both explainable line by line:

    1. **Floors.** Every group gets the smallest window that keeps it under
       the round cap, because below that the direct path is lost outright.
       If the floors alone exceed the aperture, refuse and name the gap.
    2. **Marginal rounds.** Remaining MiB go, one at a time, to whichever
       group's next MiB removes the most rounds. Rounds fall as
       ``ceil(block/window)``, so the marginal benefit is strictly
       non-increasing per group and a greedy pass is not a heuristic here --
       it cannot be beaten by moving a MiB from one group to another.

    Nothing is clipped and nothing is rounded down onto a group silently:
    every group's window and round count is reported.
    """
    if not groups:
        raise ValueError("solve_bar1_windows needs at least one group")
    if usable_mib <= 0:
        raise Bar1ApertureRefused(
            [
                f"no usable aperture at all ({usable_mib} MiB after the "
                f"reserve). Free BAR1 elsewhere, or lower "
                f"SGLANG_BARLINK_BAR1_RESERVE_MIB"
            ]
        )

    floors = {g.name: g.floor_mib(max_rounds) for g in groups}
    total_floor = sum(floors.values())
    if total_floor > usable_mib:
        worst = max(groups, key=lambda g: floors[g.name])
        raise Bar1ApertureRefused(
            [
                f"the round-cap floors of {len(groups)} group(s) need "
                f"{total_floor} MiB but only {usable_mib} MiB is usable "
                f"(short by {total_floor - usable_mib} MiB)",
                f"the largest floor is {worst.name} at {floors[worst.name]} "
                f"MiB for a {worst.largest_block_bytes / MIB:.1f} MiB block "
                f"at the {max_rounds}-round cap",
                "below a floor the direct path is declined outside a CUDA "
                "graph capture and RAISES inside one, so this is refused "
                "rather than clipped",
            ]
        )

    window = dict(floors)
    remaining = usable_mib - total_floor

    while remaining > 0:
        best_name: Optional[str] = None
        best_gain = 0
        for g in groups:
            cur = window[g.name]
            if cur >= g.ideal_mib():
                continue  # already one round; more MiB buys nothing
            gain = g.rounds_at(cur) - g.rounds_at(cur + 1)
            if gain > best_gain:
                best_gain, best_name = gain, g.name
        if best_name is None:
            # Either everyone is at one round, or no single MiB changes any
            # round count. Give the rest to the group still furthest from one
            # round, so the slack is attributed rather than left unassigned.
            hungry = [g for g in groups if window[g.name] < g.ideal_mib()]
            if not hungry:
                break
            target = max(hungry, key=lambda g: g.rounds_at(window[g.name]))
            step = min(remaining, target.ideal_mib() - window[target.name])
            window[target.name] += step
            remaining -= step
            continue
        window[best_name] += 1
        remaining -= 1

    ordered = [(g.env_key, window[g.name]) for g in groups]
    rounds = [(g.env_key, g.rounds_at(window[g.name])) for g in groups]
    return Bar1WindowSolution(
        windows_mib=tuple(ordered),
        rounds=tuple(rounds),
        usable_mib=int(usable_mib),
        allocated_mib=int(sum(window.values())),
        single_round=tuple(
            g.env_key for g in groups if g.rounds_at(window[g.name]) == 1
        ),
        basis=tuple((g.env_key, g.basis) for g in groups),
    )


def advisory_lines(
    solution: Bar1WindowSolution,
    default_mib: int,
    incumbent: Optional[str] = None,
) -> List[str]:
    """The provenance block: SOLVED, NOT INSTALLED.

    This deliberately follows ``uneven_perf``'s existing shape at :7851 --
    "JOINT PREFILL LAYOUT (#485, DESK/PREDICTED -- SOLVED, NOT INSTALLED) ...
    Launch it with:" -- rather than materializing the flag the way #1017's
    vector solve does.

    THE REASON IS A HONEST GAP IN THE INPUT, NOT CAUTION FOR ITS OWN SAKE.
    The solve is only as good as the per-group ``largest_block_bytes`` fed to
    it, and today those are DERIVED from geometry, while the runtime computes
    the real value group-wide at the call site
    (``barlink.BarlinkCommunicator.all_to_all_single`` takes a ``_group_max``
    over the send/recv matrix). Until a boot reports the measured blocks, an
    installed window could undersize a group by exactly the ratio the
    prediction is wrong -- costing rounds against a hand-pin that, whatever
    else is true of it, was fitted to a real boot. So the solve is printed
    with its inputs, and installation waits on one measurement.
    """
    lines = [
        "BAR1 WINDOW SOLVE (#1012, DESK/PREDICTED -- SOLVED, NOT INSTALLED): "
        + solution.summary(),
    ]
    for (key, mib), (_, rounds) in zip(solution.windows_mib, solution.rounds):
        basis = dict(solution.basis).get(key, "derived")
        lines.append(
            f"  {key}: {mib} MiB -> {rounds} round(s)  [block basis: {basis}]"
        )
    lines.append(f"  Launch it with: --barlink-bar1-window-mib {solution.flag_value(default_mib)}")
    if incumbent:
        lines.append(f"  Incumbent hand-pin, unchanged by this solve: {incumbent}")
    lines.append(
        "  NOT INSTALLED: the per-group block sizes above are DERIVED from "
        "model geometry. The runtime computes the real largest block "
        "group-wide at the call site, so calibrate against a boot that "
        "reports them before pinning this."
    )
    return lines


def activation_block_bytes(tokens: int, hidden: int, dtype_bytes: int) -> int:
    """Largest activation block a width-parallel group carries.

    The formula is not invented here -- it is the one
    ``barlink_matrix_transport._requested`` states for its own worked example:
    "the tp group carries 20 MiB during prefill (chunked_prefill_size 2048 x
    hidden 5120 x 2 B)". 2048*5120*2 = 20.0 MiB, so this reproduces the
    documented number exactly rather than approximating it.
    """
    return int(tokens) * int(hidden) * int(dtype_bytes)


def kv_block_bytes(
    tokens: int, kv_heads: int, head_dim: int, dtype_bytes: int
) -> int:
    """Largest KV block a seam/KV-carrying group stages (K and V both)."""
    return int(tokens) * int(kv_heads) * int(head_dim) * 2 * int(dtype_bytes)
