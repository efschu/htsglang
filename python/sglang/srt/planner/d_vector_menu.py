# SPDX-License-Identifier: Apache-2.0
"""The D-VECTOR MENU: which decode layout the flip installs, chosen from the
backlog instead of from a boot flag.

USER ORDER 2026-09-20, verbatim: the shard distribution and the DCP token key
in the D layout should change "ja nur am flip", according to the number of
simultaneous requests and the context depth.

WHAT IS NEW HERE AND WHAT IS NOT
--------------------------------
NOT new, and deliberately not re-solved (upstream-minimal; the fork's own
"NO SECOND SOLVER" discipline at ``weg2/launcher.py:1313``):

  * the weight vectors themselves -- they are the #1241 operating points
    ``D_OPERATING_POINTS = ("decode-bs1", "decode-bs6")``, already solved by
    ``d_tp_ratio_decision`` (``weg2/launcher.py:7379``) off the measured card
    library. This module CHOOSES among them; it never invents one.
  * the pool arithmetic -- ``cp_token_context_budget`` (``distributed/
    utils.py:1377``) is the GESAMTPOOL formula and the only one used here.
  * the token split rounding -- ``partition_units`` (``distributed/
    utils.py:1466``), largest-remainder, every rank >= 1.
  * the hysteresis and dwell shapes -- ``clears_band`` / ``DwellGate``
    (``managers/regime_classifier.py``), whose #360 rule is that a delta
    inside its own A-vs-A band is not a delta.

NEW, because nothing in the tree does it:

  1. a menu ENTRY is derived end-to-end from one weight vector: the weight
     vector fixes the per-rank weight bytes, which fix the free VRAM, which
     fix the KV rows, which fix the token key AND the GESAMTPOOL. One input,
     four outputs, no hand-pin anywhere (Planner-Alleinzustaendigkeit);
  2. a PAIRWISE switch price. ``Stage.flip_cost_s`` is a scalar per stage --
     the cost of flipping INTO it. Under the diff-flip ("liegen lassen /
     BAR1-Diff / Host zuletzt") the cost of a flip depends on WHICH vector
     was resident before, because the bytes a rank already holds are free and
     only the delta crosses BAR1. That is a function of the PAIR, and it is
     the term that prices a menu switch against standing still;
  3. selection from the BACKLOG SHAPE (how many requests, how deep) rather
     than from the regime label.

THE ONE PHYSICAL FACT THE WHOLE DESIGN TURNS ON
-----------------------------------------------
P is a LAYER cut and D is a COLUMN cut. Rank r's P stage is every tensor of
layers ``[.., ..)``; rank r's D shard is a column range of EVERY layer. They
intersect, and the intersection is what "liegen lassen" keeps:

    resident_r = (layers_r / n_layers) * (column_share_r) * W

so the bytes rank r must pull over BAR1 to complete its D shard are

    need_r - resident_r = column_share_r * W * (1 - layers_r / n_layers)

This is why the flip is not free even when nothing changes, and why changing
the vector is not catastrophic either: the marginal cost of A -> B is only the
part of B's column range that A's column range did not already cover.

THE TAKTGEBER IS THE SLOWEST LINK, NOT THE TOTAL. Every leg runs in parallel
across pairs, so the wall cost of a move is ``max_r(in_r / link_r)``. On this
rig rank 1 sits on x4 (~6.5 GB/s) against x8 for ranks 0 and 2, so a byte
routed to rank 1 costs roughly twice what the same byte costs elsewhere. A
model that summed bytes and divided by a mean would price the menu wrong in
exactly the direction that matters.

WHAT THIS MODULE REFUSES TO DO
------------------------------
It does not install anything. It is a pure function from (backlog, census,
calibration) to a verdict, so that every rank can evaluate it on the same
inputs and get the same answer bit-for-bit, and so that a desk test can pin
the arithmetic without a GPU. The install stays with the one authority that
has it today (``set_cp_token_ratios`` / ``set_tp_partition_ratios`` at the
cutover), and the verdict rides PP0's existing one-decider channel
(``take_flip_decision`` / ``apply_flip_decision``). Divergence between ranks
is a CRASH/STOP, not a vote -- see :func:`require_rank_uniform`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Dict, Optional, Sequence, Tuple

from sglang.srt.distributed.utils import cp_token_context_budget, partition_units

__all__ = [
    "DVectorMenuError",
    "DBacklog",
    "RankCensus",
    "DVector",
    "SwitchPrice",
    "MenuVerdict",
    "MenuCalibration",
    "TOKEN_GRAIN",
    "derive_vector",
    "build_menu",
    "switch_price",
    "select_d_vector",
    "require_rank_uniform",
]


class DVectorMenuError(RuntimeError):
    """Loud failure of the D-vector menu family."""


#: Rounding grain of the token key. The SAME 64 the runtime's own token
#: resolver uses (``distributed/utils.py:1258`` ``_CP_TOKEN_UNITS = 64``);
#: quoted rather than re-chosen so a menu vector and a runtime-resolved vector
#: land on the same lattice and cannot differ by rounding alone.
TOKEN_GRAIN = 64


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DBacklog:
    """What the D phase is about to serve, as PP0 sees it at the flip.

    The three scalars are exactly the ones the scheduler already reduces the
    waiting queue to for the regime observer (``managers/scheduler.py``
    ``queued_reqs`` / ``queued_prompt_tokens`` / ``max_queued_prompt_tokens``),
    plus the tokens the running batch already holds. Nothing here needs a new
    sensor; what was missing was a consumer that reads the DEPTH and not only
    the COUNT -- the flip's own arming condition is documented as "deliberately
    coarse: count only, no token lengths", which is precisely the blindness
    this module removes.

    ``depths`` is optional and, when present, is authoritative: it is the exact
    per-request length vector. When it is absent the two aggregates still
    determine everything this module needs, which keeps the menu usable from
    the coarse sample.
    """

    queued_reqs: int
    queued_prompt_tokens: int
    max_queued_prompt_tokens: int
    held_tokens: int = 0
    depths: Optional[Tuple[int, ...]] = None

    def __post_init__(self) -> None:
        for field in (
            "queued_reqs",
            "queued_prompt_tokens",
            "max_queued_prompt_tokens",
            "held_tokens",
        ):
            if int(getattr(self, field)) < 0:
                raise DVectorMenuError(
                    f"DBacklog.{field} is {getattr(self, field)}; a negative "
                    f"backlog term is a sensor defect, not a small backlog, "
                    f"and silently clamping it would hide the defect behind a "
                    f"plausible vector."
                )
        if self.depths is not None:
            if len(self.depths) != self.queued_reqs:
                raise DVectorMenuError(
                    f"DBacklog carries {len(self.depths)} depths against "
                    f"queued_reqs={self.queued_reqs}. The two describe the "
                    f"same queue, so a disagreement means one of them is from "
                    f"a different round and the menu would be chosen for a "
                    f"backlog that never existed."
                )
            if any(int(d) < 0 for d in self.depths):
                raise DVectorMenuError("DBacklog.depths carries a negative length")

    @property
    def concurrency(self) -> int:
        """Requests the decode phase will run at once, at least one.

        A flip into D with an empty queue still decodes the running batch, so
        zero queued requests is not zero concurrency.
        """
        return max(1, int(self.queued_reqs))

    @property
    def deepest(self) -> int:
        """The deepest single context the pool must be able to hold."""
        if self.depths:
            return max(int(d) for d in self.depths)
        return int(self.max_queued_prompt_tokens)

    @property
    def demand_tokens(self) -> int:
        """The GESAMTPOOL this backlog needs, held plus queued.

        Summed, not maxed: under DCP every rank owns a slice of EVERY
        sequence, so concurrent requests share one global pool and the pool a
        backlog demands is the sum of its depths. The per-rank figures are
        never the denominator here (D-KV-GESAMTPOOL law).
        """
        if self.depths:
            queued = sum(int(d) for d in self.depths)
        else:
            queued = int(self.queued_prompt_tokens)
        return int(self.held_tokens) + queued

    def fingerprint(self) -> str:
        return _digest(dataclasses.asdict(self))


@dataclasses.dataclass(frozen=True)
class RankCensus:
    """Per-rank physical facts, all measured or budgeted, none assumed.

    Every field is a census/budget/calibration term that already exists on a
    boot line; the menu multiplies them and never invents one. The units are
    stated because the single most expensive defect class in this tree is a
    number whose denominator was guessed.
    """

    #: Total VRAM per rank, bytes (NVML total minus the operator's reserve).
    total_bytes: Tuple[int, ...]
    #: Total D-phase checkpoint bytes across ALL ranks (the thing the column
    #: split divides). Not per rank.
    weight_bytes_total: int
    #: The P-phase LAYER split, per rank, and the model's layer count. This is
    #: what makes the diff-flip cheap: rank r keeps the intersection of its own
    #: P stage with its D column range.
    p_layers: Tuple[int, ...]
    n_layers: int
    #: Bytes one KV token costs across ALL ranks together, from the pool's own
    #: row schema. NOT per rank: under uneven TP the KV heads are split by the
    #: SAME vector as the weights, so a rank's row bytes are a function of the
    #: vector and are derived in :func:`derive_vector` like everything else.
    #: Passing them per-rank was the modelling defect this field replaces --
    #: it held the 5090's rows expensive under every vector, including the ones
    #: that give the 5090 fewer heads.
    kv_row_bytes_total: int
    #: Measured BAR1 leg rate per rank, bytes/s. Rank 1 on this rig is x4 and
    #: is the taktgeber; these come from the transport's own per-pair line
    #: (``weight_exchange_transport`` ``gbs``), never from a nominal figure.
    link_bytes_per_s: Tuple[float, ...]
    #: Bytes that are neither weights nor KV and cannot be displaced: the
    #: retained P-phase residue outside the D column range, activation and
    #: prefill transients, the graph set, the mixer. Booked explicitly because
    #: the memory law forbids an implicit reserve.
    other_resident_bytes: Tuple[int, ...] = ()
    #: One resident decode/verify/draft graph set, bytes per rank.
    graph_set_bytes: Tuple[int, ...] = ()
    #: Seconds to capture one graph set. Measured, not modelled.
    graph_capture_s: float = 0.0
    #: "Liegen lassen": the rank keeps its WHOLE P stage resident through the
    #: D phase, so the return flip costs nothing. Its cost is the part of the P
    #: stage that lies OUTSIDE the rank's D column range -- the intersection is
    #: already paid for as weight bytes, and charging it twice would make every
    #: vector look ~10 GB more expensive than it is. Vector-dependent, so it is
    #: computed in :func:`derive_vector` rather than passed as a flat term.
    #:
    #: Set False to price the pre-order behaviour, where the P stage lives only
    #: in the host image and the return trip pays for it.
    p_stage_resident: bool = True
    #: The kv<tp case (#1258): the KV heads cannot be split as finely as the
    #: weights, so every rank carries a REPLICATED full row. Then the row bytes
    #: are the same on every rank and do NOT follow the vector -- which removes
    #: the main reason a menu switch pays for itself on this rig, so it is worth
    #: censusing rather than assuming.
    kv_replicated: bool = False

    def __post_init__(self) -> None:
        n = len(self.total_bytes)
        if n == 0:
            raise DVectorMenuError("RankCensus needs at least one rank")
        for name in ("p_layers", "link_bytes_per_s"):
            if len(getattr(self, name)) != n:
                raise DVectorMenuError(
                    f"RankCensus.{name} has {len(getattr(self, name))} entries "
                    f"against {n} ranks. A per-rank term of the wrong length "
                    f"cannot be matched to a rank, and pairing it by position "
                    f"anyway is how a 3080 term gets charged to the 5090."
                )
        if sum(self.p_layers) != self.n_layers:
            raise DVectorMenuError(
                f"RankCensus.p_layers {list(self.p_layers)} sums to "
                f"{sum(self.p_layers)} against n_layers={self.n_layers}. The "
                f"P stage split must cover the model exactly; a gap or an "
                f"overlap would make the resident term of the diff-flip wrong "
                f"in a direction that looks like free bytes."
            )
        if self.kv_row_bytes_total <= 0:
            raise DVectorMenuError("RankCensus.kv_row_bytes_total must be > 0")
        if any(v <= 0 for v in self.link_bytes_per_s):
            raise DVectorMenuError(
                "RankCensus.link_bytes_per_s must be > 0 per rank; a zero link "
                "would price every move as free or as infinite."
            )
        if self.weight_bytes_total <= 0:
            raise DVectorMenuError("RankCensus.weight_bytes_total must be > 0")
        if self.graph_capture_s < 0:
            raise DVectorMenuError("RankCensus.graph_capture_s must be >= 0")

    @property
    def n_ranks(self) -> int:
        return len(self.total_bytes)

    def _padded(self, values: Sequence[int]) -> Tuple[int, ...]:
        if not values:
            return tuple(0 for _ in range(self.n_ranks))
        if len(values) != self.n_ranks:
            raise DVectorMenuError(
                f"optional per-rank term has {len(values)} entries against "
                f"{self.n_ranks} ranks"
            )
        return tuple(int(v) for v in values)

    @property
    def other(self) -> Tuple[int, ...]:
        return self._padded(self.other_resident_bytes)

    @property
    def graphs(self) -> Tuple[int, ...]:
        return self._padded(self.graph_set_bytes)

    def kv_row_bytes_for(
        self, bounds: Sequence[Tuple[float, float]]
    ) -> Tuple[int, ...]:
        """What one KV token costs on each rank under a given column split.

        Under uneven TP the KV heads ride the SAME vector as the weights, so a
        rank that takes more columns also pays more per token. Under the
        replicated regime it does not, and then the vector cannot buy cheaper
        rows at all -- which is worth knowing, because it removes the main
        reason a menu switch pays for itself on this rig.
        """
        if self.kv_replicated:
            return tuple(int(self.kv_row_bytes_total) for _ in range(self.n_ranks))
        return tuple(
            max(1, int(round((hi - lo) * self.kv_row_bytes_total))) for lo, hi in bounds
        )

    def fingerprint(self) -> str:
        return _digest(dataclasses.asdict(self))


@dataclasses.dataclass(frozen=True)
class MenuCalibration:
    """The knobs that turn a price into a decision. All defaults are stated.

    ``band_pct`` is the A-vs-A band of the throughput signal (#360). A gain
    inside its own band is not a gain; ``margin`` is the house 2x for a
    threshold.
    """

    #: Concurrency at or above which the round is compute-bound rather than
    #: bandwidth-bound. The #1241 operating points are named bs1 and bs6, so
    #: the boundary between them is where the menu switches arms.
    deep_concurrency_max: int = 1
    #: Expected decode rounds the chosen vector will serve before the next
    #: flip. The payback horizon; from the measured mean D-phase length.
    horizon_rounds: int = 0
    #: Mean decode round seconds, measured.
    mean_round_s: float = 0.0
    #: A-vs-A band of the tok/s signal, percent.
    band_pct: float = 0.0
    #: Threshold margin over the band (#360 section 3.4).
    margin: float = 2.0
    #: Minimum flips a vector must be held before the menu may move again.
    min_dwell_flips: int = 1

    def __post_init__(self) -> None:
        if self.deep_concurrency_max < 1:
            raise DVectorMenuError("deep_concurrency_max must be >= 1")
        if self.min_dwell_flips < 1:
            raise DVectorMenuError("min_dwell_flips must be >= 1")
        if self.band_pct < 0 or self.margin <= 0:
            raise DVectorMenuError("band_pct must be >= 0 and margin > 0")
        if self.horizon_rounds < 0 or self.mean_round_s < 0:
            raise DVectorMenuError("horizon_rounds and mean_round_s must be >= 0")


# ---------------------------------------------------------------------------
# A menu entry
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DVector:
    """One D layout: the triple the flip installs, plus what it funds.

    A TUPLE, NOT THREE KNOBS -- the same rule ``Stage`` states: the weight
    vector, the token key and the per-rank KV cap are solved together and the
    flip moves all of them or none. Moving one alone is how a pool collapses
    under a live working set, and under the weighted owner rule it is also how
    a slot id goes out of bounds: the rule would split rows under a vector the
    pools were not sized for.
    """

    name: str
    #: ``--rank-tp-ratio``: the weight COLUMN split.
    weight_ratio: Tuple[int, ...]
    #: The DCP TOKEN key: rank r owns ``token_ratio[r]`` of every
    #: ``sum(token_ratio)`` context tokens, cyclically.
    token_ratio: Tuple[int, ...]
    #: Per-rank KV rows the layout may allocate. The cap, per rank.
    kv_cap_rows: Tuple[int, ...]
    #: THE GESAMTPOOL. The only pool figure that may be quoted for a layout.
    pool_tokens: int
    #: Per-rank weight bytes under ``weight_ratio``.
    weight_bytes: Tuple[int, ...]
    #: Per-rank bytes that survive a flip into this vector because they are the
    #: intersection of the rank's own P stage with its D column range.
    resident_bytes: Tuple[int, ...]
    #: Per-rank bytes left for KV once everything else is placed. ONE writer,
    #: so a second consumer cannot re-derive it with a different set of terms.
    free_bytes: Tuple[int, ...] = ()
    #: Per-rank KV bytes per token UNDER THIS VECTOR. Carried on the entry
    #: rather than on the census because it is vector-dependent.
    kv_row_bytes: Tuple[int, ...] = ()

    def fingerprint(self) -> str:
        return _digest(dataclasses.asdict(self))

    def describe(self) -> str:
        return (
            f"{self.name}: weights {list(self.weight_ratio)} "
            f"tokens {list(self.token_ratio)} "
            f"GESAMTPOOL {self.pool_tokens} "
            f"(per-rank caps {list(self.kv_cap_rows)} rows -- quoted only as "
            f"the constraint, never as the pool)"
        )


def _column_bounds(ratio: Sequence[int]) -> Tuple[Tuple[float, float], ...]:
    """Cumulative column range per rank, as fractions of the whole checkpoint.

    Contiguous by construction, which is what makes an A -> B overlap a simple
    interval intersection instead of a set difference over shard ids.
    """
    total = float(sum(ratio))
    if total <= 0:
        raise DVectorMenuError(f"weight ratio {list(ratio)} sums to {total}")
    bounds = []
    acc = 0.0
    for w in ratio:
        lo = acc
        acc += float(w) / total
        bounds.append((lo, acc))
    # Pin the last edge exactly, so float drift cannot leave a sliver of the
    # checkpoint owned by nobody.
    bounds[-1] = (bounds[-1][0], 1.0)
    return tuple(bounds)


def derive_vector(
    name: str, weight_ratio: Sequence[int], census: RankCensus
) -> DVector:
    """One weight vector in, a full menu entry out. No hand numbers.

    The chain, each step from a census term:

      weight_ratio -> per-rank weight bytes
                   -> free VRAM = total - weights - other
                   -> KV rows    = free // kv_row_bytes
                   -> token key  = partition_units(64, rows)   [capacity-prop]
                   -> GESAMTPOOL = cp_token_context_budget(key, rows)

    The token key is capacity-proportional because that is what MAXIMISES the
    pool under the weighted owner rule -- ``cp_token_context_budget``'s own
    docstring says so, and ``--rank-kv-ratio capacity`` is the runtime arm that
    installs it. So the menu does not need a second token axis to express "KV
    toward the cards with more room": that tilt FALLS OUT of the weight vector.
    A heavier 5090 column share leaves the 5090 less free VRAM, the
    capacity-proportional key then hands more tokens to the 3080s, and the
    tilt the order asks for is a consequence rather than a knob.
    """
    if len(weight_ratio) != census.n_ranks:
        raise DVectorMenuError(
            f"weight ratio {list(weight_ratio)} has {len(weight_ratio)} entries "
            f"against {census.n_ranks} ranks"
        )
    if any(int(w) <= 0 for w in weight_ratio):
        raise DVectorMenuError(
            f"weight ratio {list(weight_ratio)} has a non-positive entry. A "
            f"rank with weight share 0 holds no shard while still being in the "
            f"collective, which the column split cannot express."
        )
    ratio = tuple(int(w) for w in weight_ratio)
    bounds = _column_bounds(ratio)
    W = int(census.weight_bytes_total)

    weight_bytes = tuple(int(round((hi - lo) * W)) for lo, hi in bounds)
    resident = tuple(
        int(round((hi - lo) * W * (census.p_layers[r] / census.n_layers)))
        for r, (lo, hi) in enumerate(bounds)
    )

    other = census.other
    graphs = census.graphs
    # The P stage kept for the return trip, minus the part already counted as
    # this vector's weight bytes. Zero when the intersection is the whole P
    # stage, which is the best case the diff-flip can reach.
    p_only = tuple(
        (
            int(round((census.p_layers[r] / census.n_layers) * W * (1.0 - (hi - lo))))
            if census.p_stage_resident
            else 0
        )
        for r, (lo, hi) in enumerate(bounds)
    )
    free = []
    for r in range(census.n_ranks):
        f = (
            int(census.total_bytes[r])
            - weight_bytes[r]
            - p_only[r]
            - other[r]
            - graphs[r]
        )
        if f <= 0:
            raise DVectorMenuError(
                f"vector {name!r} leaves rank {r} with {f} free bytes: total "
                f"{census.total_bytes[r]} minus D-shard {weight_bytes[r]} minus "
                f"retained P stage {p_only[r]} minus other-resident {other[r]} "
                f"minus graphs {graphs[r]}. This is a physical impossibility, "
                f"not a tight fit, and the menu refuses it here rather than "
                f"letting the allocator find it later."
            )
        free.append(f)

    row_bytes = census.kv_row_bytes_for(bounds)
    rows = tuple(free[r] // int(row_bytes[r]) for r in range(census.n_ranks))
    if any(x < 1 for x in rows):
        raise DVectorMenuError(
            f"vector {name!r} funds {list(rows)} KV rows; a rank with no row "
            f"owns no tokens while still holding a weight shard, which the "
            f"owner rule cannot express."
        )
    key = tuple(partition_units(TOKEN_GRAIN, list(rows)))
    pool = int(cp_token_context_budget(list(key), list(rows)))
    return DVector(
        name=str(name),
        weight_ratio=ratio,
        token_ratio=key,
        kv_cap_rows=rows,
        pool_tokens=pool,
        weight_bytes=weight_bytes,
        resident_bytes=resident,
        free_bytes=tuple(free),
        kv_row_bytes=row_bytes,
    )


def build_menu(
    positions: Dict[str, Sequence[int]], census: RankCensus
) -> Tuple[DVector, ...]:
    """Derive every menu entry from its weight vector.

    ``positions`` is the DECLARED CEILING SET, not a free choice at flip time.
    The discipline is ``KvReshardRuntime``'s (``managers/kv_reshard.py``): a
    vector that was not declared before the pools were built cannot be armed,
    because the pools were pre-sized for the declared set and a fourth vector
    would point the owner rule at a split no allocation backs. The menu is
    therefore a CHOICE WITHIN a boot-declared set, never an open solve.
    """
    if not positions:
        raise DVectorMenuError(
            "the D-vector menu is empty. An empty menu is not 'hold the "
            "current vector' -- it is a boot that declared no ceiling set, and "
            "silently holding would hide that."
        )
    menu = tuple(derive_vector(n, v, census) for n, v in sorted(positions.items()))
    seen: Dict[Tuple[int, ...], str] = {}
    for v in menu:
        if v.weight_ratio in seen:
            raise DVectorMenuError(
                f"menu entries {seen[v.weight_ratio]!r} and {v.name!r} carry "
                f"the same weight vector {list(v.weight_ratio)}. Two names for "
                f"one layout make a switch between them cost a full recapture "
                f"for no change at all."
            )
        seen[v.weight_ratio] = v.name
    return menu


# ---------------------------------------------------------------------------
# The pairwise price
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SwitchPrice:
    """What moving from one menu entry to another costs, in bytes and seconds.

    Three terms, kept apart because they are paid to three different things
    and only one of them is avoidable by holding the vector:

      * ``weight_in_bytes``  -- the column range B needs that A did not hold.
        Zero when A == B, which is the whole point of the diff-flip.
      * ``kv_in_bytes``      -- resident D-KV rows whose OWNER changes when
        the token key changes. Zero under a full-reset flip that drops D-KV;
        non-zero under the "D-KV bleibt resident" direction, which is exactly
        why that direction makes a vector switch more expensive, not less.
      * ``graph_capture_s``  -- a new weight vector changes per-rank head
        counts, so every captured decode/verify/draft shape is invalid.

    Seconds are ``max_r(bytes_r / link_r)`` per leg, NOT the total over a mean
    rate: the legs run in parallel across pairs and the barrier waits for the
    slowest rank. On this rig that is the x4 rank.
    """

    src: str
    dst: str
    weight_in_bytes: Tuple[int, ...]
    kv_in_bytes: Tuple[int, ...]
    weight_s: float
    kv_s: float
    graph_capture_s: float
    taktgeber_rank: int

    @property
    def total_bytes(self) -> int:
        return int(sum(self.weight_in_bytes) + sum(self.kv_in_bytes))

    @property
    def total_s(self) -> float:
        """The legs serialize: the arena refill completes before the owner rule
        is switched, and the capture follows both."""
        return float(self.weight_s + self.kv_s + self.graph_capture_s)

    def describe(self) -> str:
        return (
            f"{self.src} -> {self.dst}: {self.total_bytes / 1e9:.2f} GB over "
            f"BAR1 ({sum(self.weight_in_bytes) / 1e9:.2f} weights + "
            f"{sum(self.kv_in_bytes) / 1e9:.2f} KV), "
            f"{self.total_s:.3f} s = {self.weight_s:.3f} weights + "
            f"{self.kv_s:.3f} KV + {self.graph_capture_s:.3f} capture; "
            f"taktgeber rank {self.taktgeber_rank}"
        )


def switch_price(
    src: DVector,
    dst: DVector,
    census: RankCensus,
    *,
    resident_kv_tokens: int = 0,
    graph_set_resident: bool = False,
) -> SwitchPrice:
    """Price A -> B under the diff-flip.

    ``resident_kv_tokens`` is the GESAMTPOOL actually occupied when the switch
    is taken -- the D-KV that "bleibt resident". Pass 0 to price a full-reset
    flip, which is the pre-order behaviour and the cheap case for a switch
    precisely because it has already thrown the valuable thing away.

    ``graph_set_resident`` says B's captured shapes are still in VRAM from an
    earlier residence, so no recapture is owed. Whether two sets CAN stay
    resident is a VRAM question and is answered by
    :func:`two_graph_sets_fit`, not assumed here.
    """
    n = census.n_ranks
    a = _column_bounds(src.weight_ratio)
    b = _column_bounds(dst.weight_ratio)
    W = int(census.weight_bytes_total)

    # A rank keeps whatever of B's range it already covered -- either because
    # A's column range covered it, or because it is inside its own P stage and
    # was left lying there. The union of the two, not the sum: double-counting
    # the overlap would price the move as cheaper than it is.
    p_frac = tuple(census.p_layers[r] / census.n_layers for r in range(n))
    weight_in = []
    for r in range(n):
        need_lo, need_hi = b[r]
        have_lo, have_hi = a[r]
        overlap = max(0.0, min(need_hi, have_hi) - max(need_lo, have_lo))
        # The part of B's range that A did not hold is still partly resident
        # via the P stage, which covers that column range on p_frac of layers.
        missing = (need_hi - need_lo) - overlap
        weight_in.append(int(round(missing * (1.0 - p_frac[r]) * W)))

    # Token-key ownership churn. Rank r holds share_a of the resident rows and
    # must end with share_b; what it gains arrives over BAR1.
    sa = float(sum(src.token_ratio))
    sb = float(sum(dst.token_ratio))
    kv_in = []
    for r in range(n):
        share_a = src.token_ratio[r] / sa
        share_b = dst.token_ratio[r] / sb
        gained = max(0.0, share_b - share_a) * int(resident_kv_tokens)
        # Charged at the DESTINATION vector's row bytes: the rows arrive in the
        # layout they will be read in, not the one they left.
        kv_in.append(int(round(gained * int(dst.kv_row_bytes[r]))))

    link = census.link_bytes_per_s
    w_times = [weight_in[r] / float(link[r]) for r in range(n)]
    k_times = [kv_in[r] / float(link[r]) for r in range(n)]
    weight_s = max(w_times) if n else 0.0
    kv_s = max(k_times) if n else 0.0
    # The taktgeber is the rank that binds the larger of the two legs; when
    # nothing moves at all there is no clock and rank 0 is named by convention
    # with both legs at zero, which the caller can see.
    binding = w_times if weight_s >= kv_s else k_times
    taktgeber = max(range(n), key=lambda r: binding[r]) if n else 0

    recapture = 0.0
    if src.weight_ratio != dst.weight_ratio and not graph_set_resident:
        recapture = float(census.graph_capture_s)

    return SwitchPrice(
        src=src.name,
        dst=dst.name,
        weight_in_bytes=tuple(weight_in),
        kv_in_bytes=tuple(kv_in),
        weight_s=weight_s,
        kv_s=kv_s,
        graph_capture_s=recapture,
        taktgeber_rank=taktgeber,
    )


@dataclasses.dataclass(frozen=True)
class SecondGraphSetTrade:
    """What keeping BOTH vectors' graph sets resident costs and buys.

    THIS IS A TRADE, NOT A FIT CHECK, and the distinction is the whole point.
    A fit check would always answer "no": ``derive_vector`` hands every free
    byte to the KV pool, so the slack after the pool is ~0 under every vector
    by construction. The real question is what the second set is WORTH, and
    that is a price in POOL TOKENS against a saving in RECAPTURE SECONDS.

    No reserve is invented to pay for it (the memory law: reserves never, "nicht
    ein Byte"). The bytes come out of the KV pool explicitly and the caller sees
    exactly how many tokens that costs.
    """

    #: GESAMTPOOL the vector funds with one graph set resident.
    pool_with_one: int
    #: GESAMTPOOL it funds when a second set is held too.
    pool_with_two: int
    #: Recapture seconds a switch avoids while both sets are resident.
    recapture_saved_s: float
    #: True when holding the second set is physically possible at all.
    possible: bool
    reason: str

    @property
    def pool_cost_tokens(self) -> int:
        return int(self.pool_with_one - self.pool_with_two)


def second_graph_set_trade(v: DVector, census: RankCensus) -> SecondGraphSetTrade:
    """Price holding a SECOND resident graph set under vector ``v``.

    ``graph_capture_s`` is the dominant term of :func:`switch_price` -- on this
    rig it is roughly two orders of magnitude above the BAR1 legs -- so whether
    two sets can stay resident is the single biggest lever on whether a menu is
    worth having at all. It is answered here by re-deriving the vector against a
    census that carries the second set, which keeps ONE derivation path.
    """
    graphs = census.graphs
    if not any(graphs):
        return SecondGraphSetTrade(
            pool_with_one=v.pool_tokens,
            pool_with_two=v.pool_tokens,
            recapture_saved_s=0.0,
            possible=False,
            reason=(
                "no graph-set size is censused, so the question cannot be "
                "answered; an unmeasured set is not a set that fits."
            ),
        )
    doubled = dataclasses.replace(census, graph_set_bytes=tuple(2 * g for g in graphs))
    try:
        with_two = derive_vector(v.name, v.weight_ratio, doubled)
    except DVectorMenuError as e:
        return SecondGraphSetTrade(
            pool_with_one=v.pool_tokens,
            pool_with_two=0,
            recapture_saved_s=0.0,
            possible=False,
            reason=(f"a second graph set does not fit under {v.name!r} at all: {e}"),
        )
    return SecondGraphSetTrade(
        pool_with_one=v.pool_tokens,
        pool_with_two=with_two.pool_tokens,
        recapture_saved_s=float(census.graph_capture_s),
        possible=True,
        reason=(
            f"holding both graph sets under {v.name!r} costs "
            f"{v.pool_tokens - with_two.pool_tokens} GESAMTPOOL tokens "
            f"({v.pool_tokens} -> {with_two.pool_tokens}) and saves "
            f"{census.graph_capture_s:.1f} s of recapture on every switch."
        ),
    )


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MenuVerdict:
    """One decision, with the reason and the arithmetic that produced it."""

    chosen: DVector
    current: Optional[str]
    switched: bool
    mandatory: bool
    reason: str
    price: Optional[SwitchPrice]
    payback_rounds: Optional[int]

    def fingerprint(self) -> str:
        """What every rank must agree on, bit-for-bit.

        The NAME and the three vectors, not the prose: two ranks may print a
        different reason string without disagreeing about the layout, but they
        may never install different vectors.
        """
        return _digest(
            {
                "name": self.chosen.name,
                "weight": list(self.chosen.weight_ratio),
                "token": list(self.chosen.token_ratio),
                "cap": list(self.chosen.kv_cap_rows),
                "pool": self.chosen.pool_tokens,
                "switched": bool(self.switched),
            }
        )

    def describe(self) -> str:
        head = "SWITCH" if self.switched else "HOLD"
        if self.mandatory:
            head += " (MANDATORY)"
        out = f"D-VECTOR MENU {head} -> {self.chosen.describe()}. {self.reason}"
        if self.price is not None:
            out += f" Price: {self.price.describe()}."
        return out


def _wants_position(backlog: DBacklog, calib: MenuCalibration) -> str:
    """Which operating point the backlog's SHAPE asks for.

    One request deep is a bandwidth-bound round (the shard is streamed once
    per token); several requests at once is a compute-bound round (the same
    shard is read once for all of them). That is the #1241 distinction, and
    the backlog is what says which side of it we are on.
    """
    if backlog.concurrency <= calib.deep_concurrency_max:
        return "deep"
    return "wide"


def select_d_vector(
    backlog: DBacklog,
    menu: Sequence[DVector],
    census: RankCensus,
    calib: MenuCalibration,
    *,
    current: Optional[str] = None,
    preference: Optional[Dict[str, str]] = None,
    gain_pct: Optional[Dict[str, float]] = None,
    resident_kv_tokens: int = 0,
    flips_on_current: int = 0,
    resident_graph_sets: Sequence[str] = (),
) -> MenuVerdict:
    """Choose the D vector for THIS flip. Pure and deterministic.

    ``preference`` maps a backlog shape ("deep" / "wide") to a menu entry
    name; it is the boot's declaration of which operating point serves which
    shape, not a runtime solve. ``gain_pct`` is the MEASURED throughput of each
    entry relative to the current one at this shape -- a menu entry with no
    measurement cannot be switched INTO on a throughput argument, only on a
    capacity one, which is the #578 unmeasured rule applied here.

    Order of authority, highest first:

      1. CAPACITY. A vector whose GESAMTPOOL cannot hold the backlog is not a
         candidate, and if the current vector cannot hold it the switch is
         MANDATORY and skips hysteresis entirely. Refusing to grow the pool
         because a move "has not paid for itself" would refuse the one move
         that is about correctness rather than speed.
      2. DWELL. A vector must be held ``min_dwell_flips`` flips before the menu
         may move again. Hysteresis asks whether the signal is real; dwell asks
         whether the move is affordable, and a load that genuinely alternates
         faster than a switch amortizes gives a clean signal and must still be
         refused.
      3. HYSTERESIS. The projected gain over the horizon must clear the switch
         price, AND the gain must clear its own A-vs-A band by the margin.
    """
    if not menu:
        raise DVectorMenuError("select_d_vector called with an empty menu")
    by_name = {v.name: v for v in menu}
    if current is not None and current not in by_name:
        raise DVectorMenuError(
            f"current vector {current!r} is not in the menu "
            f"{sorted(by_name)}. The menu is the declared ceiling set, so a "
            f"current vector outside it means the pools were built for a "
            f"layout the menu cannot describe."
        )
    cur = by_name.get(current) if current else None
    demand = backlog.demand_tokens
    shape = _wants_position(backlog, calib)

    feasible = [v for v in menu if v.pool_tokens >= demand]
    # Widest pool first, then name, so the choice is stable across ranks and
    # across runs -- a tie broken by dict order would be a divergence waiting
    # for a rehash.
    widest = max(menu, key=lambda v: (v.pool_tokens, v.name))

    if not feasible:
        target = widest
        reason = (
            f"no menu entry funds the backlog's demand of {demand} tokens "
            f"({backlog.concurrency} requests, deepest {backlog.deepest}); the "
            f"widest pool {target.name!r} at {target.pool_tokens} is taken and "
            f"the remainder stays queued for admission control. The menu does "
            f"not invent capacity it cannot back."
        )
        return _finish(
            target,
            cur,
            census,
            reason,
            True,
            resident_kv_tokens,
            resident_graph_sets,
            None,
        )

    if shape == "deep":
        # THE ORDER'S OWN RULE for 'bs1-tief', not a declared preference:
        # "KV-Schwerpunkt so, dass der Gesamtpool maximal wird". One deep
        # request is bound by whether its context fits at all, so the entry
        # with the widest GESAMTPOOL wins regardless of which operating point
        # its weight vector was solved for.
        #
        # On this rig that is NOT the bandwidth-shaped vector, and the menu
        # surfaces the tension rather than hiding it: giving the 5090 MORE
        # columns leaves the 3080s far more room, and the 3080s are where the
        # cheap KV rows are, so the compute-shaped vector funds the larger
        # pool. A preference map that hard-wired deep -> bs1 would have picked
        # the smaller pool for the one shape that needs the bigger one.
        target = max(feasible, key=lambda v: (v.pool_tokens, v.name))
    else:
        want_name = (preference or {}).get(shape)
        if want_name is not None and want_name not in by_name:
            raise DVectorMenuError(
                f"preference maps shape {shape!r} to {want_name!r}, which is "
                f"not in the menu {sorted(by_name)}"
            )
        target = by_name.get(want_name) if want_name else None
        if target is None or target not in feasible:
            # Either no preference was declared for this shape, or the
            # preferred entry cannot hold the backlog. Capacity outranks the
            # preference; it is never the other way round.
            target = max(feasible, key=lambda v: (v.pool_tokens, v.name))

    if cur is None:
        return _finish(
            target,
            cur,
            census,
            f"no vector is installed yet; the backlog is {backlog.concurrency} "
            f"requests, deepest {backlog.deepest}, demand {demand} tokens, "
            f"shape {shape!r} -> {target.name!r} at GESAMTPOOL "
            f"{target.pool_tokens}.",
            True,
            resident_kv_tokens,
            resident_graph_sets,
            None,
        )

    if cur.pool_tokens < demand:
        return _finish(
            target,
            cur,
            census,
            f"MANDATORY: the installed vector {cur.name!r} funds "
            f"{cur.pool_tokens} tokens against a demand of {demand}. A backlog "
            f"the pool cannot hold is a correctness problem, so hysteresis and "
            f"dwell do not apply; moving to {target.name!r} at "
            f"{target.pool_tokens}.",
            True,
            resident_kv_tokens,
            resident_graph_sets,
            None,
            mandatory=True,
        )

    if target.name == cur.name:
        return _hold(
            cur,
            f"the backlog's shape {shape!r} already wants the installed "
            f"vector {cur.name!r} (GESAMTPOOL {cur.pool_tokens} >= demand "
            f"{demand}); the diff-flip keeps every resident byte.",
        )

    if flips_on_current < calib.min_dwell_flips:
        return _hold(
            cur,
            f"dwell: {flips_on_current} flips on {cur.name!r} against a "
            f"minimum of {calib.min_dwell_flips}. The backlog wants "
            f"{target.name!r}, but a load that alternates faster than a switch "
            f"amortizes is a workload that alternates, not a regime that "
            f"changed.",
        )

    price = switch_price(
        cur,
        target,
        census,
        resident_kv_tokens=resident_kv_tokens,
        graph_set_resident=target.name in tuple(resident_graph_sets),
    )
    gain = float((gain_pct or {}).get(target.name, 0.0))
    if abs(gain) <= calib.margin * calib.band_pct:
        return _hold(
            cur,
            f"the measured gain of {target.name!r} over {cur.name!r} is "
            f"{gain:+.1f}% against an A-vs-A band of {calib.band_pct:.1f}% and "
            f"a {calib.margin:g}x threshold margin. A delta inside its own band "
            f"is not a delta, and {price.total_s:.3f} s of BAR1 and recapture "
            f"is a real price for it.",
        )
    if gain <= 0:
        return _hold(
            cur,
            f"{target.name!r} measures {gain:+.1f}% against {cur.name!r}; "
            f"the backlog's shape prefers it but the measurement does not.",
        )

    saved_s = calib.horizon_rounds * calib.mean_round_s * (gain / 100.0)
    payback = (
        math.ceil(price.total_s / (calib.mean_round_s * gain / 100.0))
        if calib.mean_round_s > 0 and gain > 0
        else None
    )
    if saved_s < price.total_s:
        return _hold(
            cur,
            f"the switch to {target.name!r} costs {price.total_s:.3f} s "
            f"({price.total_bytes / 1e9:.2f} GB over BAR1, taktgeber rank "
            f"{price.taktgeber_rank}) and would save {saved_s:.3f} s over the "
            f"{calib.horizon_rounds}-round horizon at {gain:+.1f}%. Payback "
            f"needs {payback} rounds. The move does not pay for itself before "
            f"the next flip undoes it.",
        )
    return _finish(
        target,
        cur,
        census,
        f"backlog shape {shape!r} ({backlog.concurrency} requests, deepest "
        f"{backlog.deepest}, demand {demand}) wants {target.name!r} at "
        f"{gain:+.1f}% over band {calib.band_pct:.1f}%; {saved_s:.3f} s saved "
        f"over {calib.horizon_rounds} rounds against {price.total_s:.3f} s of "
        f"switch, payback in {payback} rounds.",
        True,
        resident_kv_tokens,
        resident_graph_sets,
        price,
        payback=payback,
    )


def _hold(cur: DVector, reason: str) -> MenuVerdict:
    return MenuVerdict(
        chosen=cur,
        current=cur.name,
        switched=False,
        mandatory=False,
        reason=reason,
        price=None,
        payback_rounds=None,
    )


def _finish(
    target: DVector,
    cur: Optional[DVector],
    census: RankCensus,
    reason: str,
    switched: bool,
    resident_kv_tokens: int,
    resident_graph_sets: Sequence[str],
    price: Optional[SwitchPrice],
    *,
    mandatory: bool = False,
    payback: Optional[int] = None,
) -> MenuVerdict:
    if price is None and cur is not None and cur.name != target.name:
        price = switch_price(
            cur,
            target,
            census,
            resident_kv_tokens=resident_kv_tokens,
            graph_set_resident=target.name in tuple(resident_graph_sets),
        )
    really_switched = bool(switched and (cur is None or cur.name != target.name))
    return MenuVerdict(
        chosen=target,
        current=None if cur is None else cur.name,
        switched=really_switched,
        mandatory=mandatory,
        reason=reason,
        price=price,
        payback_rounds=payback,
    )


# ---------------------------------------------------------------------------
# One verdict, or none
# ---------------------------------------------------------------------------


def require_rank_uniform(
    fingerprints: Sequence[str], *, where: str = "the D-vector menu"
) -> str:
    """Every rank chose the same layout, or the boot stops.

    RAENGE NIE UNEINS. Two ranks installing different vectors is not a
    degraded mode that a majority can repair: the weighted owner rule would
    split the same token space two ways, so one rank's slot id addresses
    another rank's row, and the corruption is silent. The only safe answer to a
    disagreement is to stop.

    Returns the agreed fingerprint so the caller can log ONE value.
    """
    if not fingerprints:
        raise DVectorMenuError(
            f"{where}: no fingerprints were gathered. An empty agreement is "
            f"not agreement -- it is a reduction that did not run, and "
            f"treating it as consensus would let every rank keep its own "
            f"answer."
        )
    unique = sorted(set(fingerprints))
    if len(unique) != 1:
        raise DVectorMenuError(
            f"{where}: ranks disagree about the D vector. {len(unique)} "
            f"distinct verdicts across {len(fingerprints)} ranks: {unique}. "
            f"The menu is a pure function of the backlog, the census and the "
            f"calibration, so a disagreement means those inputs differ between "
            f"ranks -- CRASH/STOP rather than install two layouts over one "
            f"token space."
        )
    return unique[0]


def _digest(payload) -> str:
    """A stable digest of a plain structure.

    ``sort_keys`` and ``default=list`` so that tuple-vs-list and dict ordering
    -- neither of which is a disagreement about the layout -- cannot make two
    ranks look like they disagree.
    """
    blob = json.dumps(payload, sort_keys=True, default=list, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
