# SPDX-License-Identifier: Apache-2.0
"""#302a -- dynamic expert HEAT MIGRATION: re-rank the resident set at runtime.

The expert offload (#77/#123) picks its resident set ONCE, at load time
(:func:`expert_offload.plan_load_time_staging`), and never revisits it. Stage-1
hot residency (``SGLANG_MOE_HOT_RESIDENCY``) improves the *choice* but keeps the
*one-shot* shape: it calibrates over the first few forwards and then freezes for
the life of the process. Neither reacts to what the router actually sends an
hour into a serving session.

This module is the second stage: a decayed heat window per layer, a periodic
re-rank, and an EQUAL-COUNT swap of the resident set against the pinned host
pool. It is off by default (``SGLANG_MOE_HEAT_MIGRATION``).

Desk evidence this exists at all (`scripts/dev/302a_heat_desk/`, run against the
recorded `expert_stats_*.json` of four independent boots):

* the recorded static hit rates 0.7623 / 0.8427 / 0.8463 are reproduced exactly
  from the JSONs, so the simulation measures the real placement;
* the ORACLE ceiling for the SAME resident-set size is 0.9836 / 0.9844 / 0.9850
  -- the static layout leaves 13.9 to 22.1 percentage points on the table;
* a ranking learned on a DIFFERENT boot on a DIFFERENT day still captures 40 to
  83 % of that ceiling, so the signal survives staleness and is not an artefact
  of scoring a ranking on the run it was learned from.

WHAT MOVES AND WHAT DOES NOT
----------------------------
Only WHICH physical expert occupies which GPU slot / host pool row changes. The
resident COUNT is invariant by construction: swaps are pairs, one expert in and
one expert out, so ``len(resident_ids)`` and the GPU buffer are byte-for-byte
the same size before and after. That is the #439 sizing latch's invariant
(residency is held at the BASE plan) and this module must never break it -- a
migration that grew residency would silently re-price every VRAM figure the
#400 ledger and the KV-regain path were computed from.

Output identity: a token's MoE result depends on its routed experts' weights and
the top-k reduction order, neither of which a slot permutation touches. The same
bytes are multiplied; only their address changes. This is the identical argument
:meth:`expert_offload.MoEExpertOffloadCache._apply_hotset_freeze` already makes
for the Stage-1 freeze.

Eviction doctrine: `DESIGN_407_memtier_registry.md` §8 puts cold experts at rung
3 of one global importance ladder, and within a class the order is coldest
first. :func:`plan_heat_swaps` implements exactly that within-class order and
owns no other victim policy -- when the #407 registry grows its cut-4 victim
interface, the candidate list below is what registers against it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "HeatMigrationConfig",
    "HeatMigrationStats",
    "HeatWindow",
    "plan_heat_swaps",
    "refuse_heat_migration_under_graph_capture",
]


@dataclass(frozen=True)
class HeatMigrationConfig:
    """Policy knobs. ``enabled=False`` is the default and changes nothing."""

    enabled: bool = False
    # Re-rank every N recorded forwards on this layer. Small values re-rank on
    # noise and pay H2D for it; large values track a drifting workload slowly.
    period_forwards: int = 512
    # Multiplied into every expert's accumulated count at each round boundary,
    # so the window is exponentially weighted rather than whole-run. 1.0 = never
    # forget (whole-run heat), 0.0 = only the last period counts.
    decay: float = 0.5
    # #516 LONGER-HORIZON MISS BUDGET. 0.0 = OFF, and off is byte-identical:
    # every swap the periodic policy would have made is still made.
    #
    # When > 0 it is the window MISS RATE below which the resident set is left
    # alone: placement that is already meeting the budget is not re-ranked, so
    # the swap is spent only where the miss rate says it is needed. This is the
    # principled form of the hazard the `period_forwards` comment above already
    # names -- "small values re-rank on noise and pay H2D for it" -- because a
    # window whose miss rate is fine is exactly a window whose top-R movement
    # is noise.
    #
    # Measured on the recorded #302a series (see
    # scripts/dev/302a_heat_desk/simulate_miss_budget.py): at 0.04 the trigger
    # beat swap-every-window on ALL NINE recorded rank/series combinations,
    # worst case +0.0021 hit rate, mean +0.0052, at 26% of the swaps. Simulation
    # only -- it has not run on metal.
    miss_budget: float = 0.0
    # A candidate must be this much hotter than the victim it would displace.
    # Without it a pair whose heats differ by one activation swaps every round
    # and the migration pays PCIe forever for nothing.
    hysteresis: float = 0.25
    # Absolute companion to `hysteresis`, and the reason both exist. A purely
    # RELATIVE margin is scale-free: down in the tail of the routing
    # distribution, where a dozen near-identical cold experts each draw a
    # handful of activations per window, "40 % hotter" is three activations of
    # sampling noise and the resident set churns on it forever. The absolute
    # floor is the worth-it side of the same decision (`DESIGN_363` §20.1): a
    # swap costs one expert-row D2H plus one H2D, so it is only worth taking
    # when the observed heat difference is large enough that the next window
    # plausibly repeats it. Both conditions must hold.
    min_gain: float = 8.0
    # Upper bound on swaps per layer per round: the H2D burst is
    # ``swaps x expert_bytes`` and lands between two forwards.
    max_swaps: int = 4
    # Minimum total observed activations in the window before ranking at all.
    min_observations: int = 32

    @classmethod
    def from_env(cls) -> "HeatMigrationConfig":
        from sglang.srt.environ import envs

        return cls(
            enabled=bool(envs.SGLANG_MOE_HEAT_MIGRATION.get()),
            period_forwards=max(1, int(envs.SGLANG_MOE_HEAT_PERIOD.get())),
            miss_budget=max(0.0, float(envs.SGLANG_MOE_HEAT_MISS_BUDGET.get())),
            decay=min(1.0, max(0.0, float(envs.SGLANG_MOE_HEAT_DECAY.get()))),
            hysteresis=max(0.0, float(envs.SGLANG_MOE_HEAT_HYSTERESIS.get())),
            min_gain=max(0.0, float(envs.SGLANG_MOE_HEAT_MIN_GAIN.get())),
            max_swaps=max(0, int(envs.SGLANG_MOE_HEAT_MAX_SWAPS.get())),
            min_observations=max(0, int(envs.SGLANG_MOE_HEAT_MIN_OBS.get())),
        )


@dataclass
class HeatMigrationStats:
    """Counters, surfaced in the #390 dump under ``"heat_migration"``."""

    rounds: int = 0  # re-rank decisions taken
    rounds_migrating: int = 0  # of those, rounds that moved at least one expert
    swaps: int = 0  # (promote, demote) pairs executed
    promoted: int = 0  # experts moved host pool -> GPU
    demoted: int = 0  # experts moved GPU -> host pool
    skipped_hysteresis: int = 0  # candidates rejected by the margin
    skipped_delegated: int = 0  # candidates a peer's tier owns (#394)
    skipped_cap: int = 0  # candidates dropped by max_swaps
    h2d_bytes: int = 0  # promote traffic
    d2h_bytes: int = 0  # demote traffic
    # Hit rate over the window that most recently triggered a re-rank, so a
    # reader can see whether the migrations are tracking anything.
    window_hit_activations: int = 0
    window_miss_activations: int = 0
    last_window_hit_rate: float = 0.0
    # #516: rounds where the miss budget held and no swap was planned.
    budget_held_rounds: int = 0

    def as_dict(self) -> dict:
        d = {
            "rounds": self.rounds,
            "rounds_migrating": self.rounds_migrating,
            "swaps": self.swaps,
            "promoted": self.promoted,
            "demoted": self.demoted,
            "skipped_hysteresis": self.skipped_hysteresis,
            "skipped_delegated": self.skipped_delegated,
            "skipped_cap": self.skipped_cap,
            "h2d_bytes": self.h2d_bytes,
            "d2h_bytes": self.d2h_bytes,
            "last_window_hit_rate": self.last_window_hit_rate,
            "budget_held_rounds": self.budget_held_rounds,
        }
        return d


def plan_heat_swaps(
    heat: Dict[int, float],
    resident_ids: Iterable[int],
    *,
    pinned: Optional[FrozenSet[int]] = None,
    delegated: Optional[FrozenSet[int]] = None,
    hysteresis: float = 0.25,
    min_gain: float = 8.0,
    max_swaps: int = 4,
    stats: Optional[HeatMigrationStats] = None,
) -> List[Tuple[int, int]]:
    """The whole policy, as a pure function -- ``[(promote, demote), ...]``.

    Invariants, each of which has its own test:

    * **equal counts.** The return value is a list of PAIRS, so applying it
      leaves ``len(resident_ids)`` unchanged. There is no code path that returns
      an unmatched promotion; VRAM-neutrality is structural, not checked.
    * **coldest-first victims** (`DESIGN_407` §8, within-class order): victims
      are drawn from the resident set in ascending heat.
    * **hottest-first candidates**: promotions are drawn from the non-resident
      set in descending heat.
    * **pinned experts never leave.** The #82 expert-dim pad expert at id E-1 is
      the reason: every foreign token collapses onto it, so demoting it would
      make the hottest expert in the layer a miss on every single forward.
    * **delegated experts are never promoted** (#394). Their bytes live in a
      PEER's shared segment, so this rank has no local pool row to write the
      displaced victim into. Fetching them still works; only migration declines.
    * **deterministic.** Ties break by ascending expert id on both sides, so two
      ranks with identical heat produce identical plans.
    * **hysteresis, relative AND absolute.** A pair is only swapped when
      ``heat[promote] > heat[demote] * (1 + hysteresis)`` AND
      ``heat[promote] - heat[demote] >= min_gain``. Both are monotone in the
      same direction as the two sort orders, so the first pair that fails means
      every later pair fails too -- the loop stops there rather than continuing
      to test. See ``min_gain``'s field comment for why the relative term alone
      is not enough.
    """
    # ``None`` reaches here from the planner, whose ``delegated_ids`` is unset
    # on every launch without a shared cold tier -- which is every launch today.
    pinned = pinned or frozenset()
    delegated = delegated or frozenset()
    resident = [int(e) for e in resident_ids]
    resident_set = set(resident)
    if max_swaps <= 0:
        return []

    def h(e: int) -> float:
        return float(heat.get(e, 0.0))

    # Victims: resident, not pinned, coldest first.
    victims = sorted((e for e in resident if e not in pinned), key=lambda e: (h(e), e))
    # Candidates: known to the heat window, not resident, hottest first.
    candidates = sorted(
        (e for e in heat if e not in resident_set and h(e) > 0.0),
        key=lambda e: (-h(e), e),
    )

    swaps: List[Tuple[int, int]] = []
    vi = 0
    for cand in candidates:
        if len(swaps) >= max_swaps:
            if stats is not None:
                stats.skipped_cap += 1
            continue
        if cand in delegated:
            if stats is not None:
                stats.skipped_delegated += 1
            continue
        if vi >= len(victims):
            break
        victim = victims[vi]
        if h(cand) <= h(victim) * (1.0 + hysteresis) or h(cand) - h(victim) < min_gain:
            # Candidates descend, victims ascend: no later pair can pass.
            if stats is not None:
                stats.skipped_hysteresis += 1
            break
        swaps.append((cand, victim))
        vi += 1
    return swaps


class HeatWindow:
    """Per-layer decayed routing counts plus the "is a re-rank due" clock.

    Kept deliberately separate from the offload cache so the policy can be
    exercised hermetically -- no CUDA, no layer, no weights.
    """

    __slots__ = ("cfg", "heat", "forwards_since_round", "observations", "stats")

    def __init__(self, cfg: HeatMigrationConfig):
        self.cfg = cfg
        self.heat: Dict[int, float] = {}
        self.forwards_since_round = 0
        self.observations = 0
        self.stats = HeatMigrationStats()

    def observe(
        self,
        rows: Sequence[Sequence[int]],
        resident_ids: Optional[FrozenSet[int]] = None,
        resident_count: int = 0,
    ) -> None:
        """Fold one forward's ``topk_ids.tolist()`` into the window.

        The ids are already on the host at the offload path's fetch-decision
        point, so this adds no device synchronisation -- the same argument the
        #390 instrument makes at the same call site.
        """
        heat = self.heat
        hit = 0
        miss = 0
        for row in rows:
            for e in row:
                if e < 0:
                    continue
                heat[e] = heat.get(e, 0.0) + 1.0
                if (
                    e in resident_ids
                    if resident_ids is not None
                    else e < resident_count
                ):
                    hit += 1
                else:
                    miss += 1
        self.observations += hit + miss
        self.stats.window_hit_activations += hit
        self.stats.window_miss_activations += miss
        self.forwards_since_round += 1

    def due(self) -> bool:
        return (
            self.cfg.enabled
            and self.forwards_since_round >= self.cfg.period_forwards
            and self.observations >= self.cfg.min_observations
        )

    def close_round(self) -> None:
        """End the window: record its hit rate, decay the counts, reset clocks."""
        s = self.stats
        tot = s.window_hit_activations + s.window_miss_activations
        s.last_window_hit_rate = s.window_hit_activations / tot if tot else 0.0
        s.window_hit_activations = 0
        s.window_miss_activations = 0
        s.rounds += 1
        self.forwards_since_round = 0
        self.observations = 0
        decay = self.cfg.decay
        if decay >= 1.0:
            return
        if decay <= 0.0:
            self.heat.clear()
            return
        # Drop entries that have decayed into irrelevance so the dict cannot
        # grow to hold every expert forever at a negligible weight.
        self.heat = {e: v * decay for e, v in self.heat.items() if v * decay >= 1e-3}

    def budget_holds(self) -> bool:
        """True when the closing window's miss rate is within the budget.

        Pure and side-effect free, so the decision can be pinned without a
        window object's history. ``miss_budget <= 0`` disables it and this
        always returns False, which is what makes the OFF path byte-identical:
        a False here means "the budget has nothing to say", not "swap".

        Reads the CLOSING window's counters, i.e. the same numbers
        ``close_round`` is about to fold into ``last_window_hit_rate``. It must
        run BEFORE the round closes, which is where :meth:`plan` calls it.
        """
        budget = float(self.cfg.miss_budget)
        if budget <= 0.0:
            return False
        s = self.stats
        total = s.window_hit_activations + s.window_miss_activations
        if total <= 0:
            return False
        return (s.window_miss_activations / total) <= budget

    def plan(
        self,
        resident_ids: Iterable[int],
        *,
        pinned: Optional[FrozenSet[int]] = None,
        delegated: Optional[FrozenSet[int]] = None,
    ) -> List[Tuple[int, int]]:
        # #516: spend the swap only where the miss rate says it is needed. A
        # window already inside the budget is left alone -- its top-R movement
        # is the noise the module's own period comment warns about paying for.
        if self.budget_holds():
            self.stats.budget_held_rounds += 1
            return []
        return plan_heat_swaps(
            self.heat,
            resident_ids,
            pinned=pinned,
            delegated=delegated,
            hysteresis=self.cfg.hysteresis,
            min_gain=self.cfg.min_gain,
            max_swaps=self.cfg.max_swaps,
            stats=self.stats,
        )


def refuse_heat_migration_under_graph_capture(cfg: HeatMigrationConfig) -> None:
    """Heat migration and a captured decode graph are mutually exclusive.

    Same structural reason the Stage-1 live calibration is refused there: the
    capturable path snapshots the residency layout into device LUTs and UVA pool
    views at capture time (``install_capturable_buffers``), and a migration
    after that point would leave those LUTs pointing at the previous occupant of
    a slot -- a silently wrong gather, not a crash. On the eager path the
    remap is rebuilt from ``planner.resident_slot`` on every forward, which is
    why migration is safe there and only there.
    """
    from sglang.srt.environ import envs

    if not cfg.enabled:
        return
    if bool(envs.SGLANG_MOE_OFFLOAD_CUDA_GRAPH.get()):
        raise RuntimeError(
            "SGLANG_MOE_HEAT_MIGRATION=1 cannot be combined with "
            "SGLANG_MOE_OFFLOAD_CUDA_GRAPH=1: the capturable path freezes the "
            "residency layout into device LUTs at capture time, and a "
            "migration after that would move an expert out from under a "
            "captured gather. Use the eager offload path (the shipped one, see "
            "#452) or turn heat migration off."
        )
