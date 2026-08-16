"""#704b: PP-KV decoupling -- collective cost and token-vector feasibility.

Decoupling token-shards the full-attention layers' KV across ALL ranks instead
of pinning it to the rank that owns the layer. The owning stage then computes
attention over the distributed pool: broadcast Q, compute a partial attention
per shard, merge with LSE.

Everything here is arithmetic done BEFORE any measurement, because two numbers
decide whether the feature is worth building and both can be got from the
census and the model config.

**1. The collective cost is driven by Q and the OUTPUT, not by the KV.** Per
attention layer per chunk each remote participant receives a full Q block and
returns a full partial-output block plus its LSE. Those are sized by
``chunk_tokens x num_attention_heads x head_dim``, and they do not shrink when
a rank holds fewer KV rows. A rank holding 1% of the shard returns exactly the
same bytes as one holding 99%. The intuition "shard less aggressively to save
bandwidth" is therefore false, and the only real levers are the number of
remote participants and the chunk size relative to compute.

**2. The phase-uniform token vector fights the deep cuts it exists to unlock.**
The prize is choosing the same token vector as the TP phase, which makes the KV
layout phase-uniform and deletes the seam KV move. But this rig's TP vector
``[14,10,8]`` puts 43.75% of all KV rows on rank0, and rank0 is exactly the
rank a deep PP cut loads with weights. The two goals contradict, and the design
has to say which yields rather than assume they compose.

A distinction this module keeps separable, because conflating them would pay
for both or neither:

* **structural uniformity** -- both phases token-shard the same layers, so the
  layout KIND is the same. This is what a content-addressed cache key needs
  (#703), and it is free;
* **vector identity** -- both phases use the same SHARES, so no row moves at
  the seam. This additionally kills the fixed flip cost (#690), and it costs
  ladder depth.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

GIB = 1024.0**3


class DecoupledKvError(ValueError):
    """A decoupling configuration that cannot be honoured."""


@dataclasses.dataclass(frozen=True)
class KvGeometry:
    """Model geometry, all of it read from config -- none of it fitted."""

    num_attention_heads: int
    head_dim: int
    num_key_value_heads: int
    kv_dtype_bytes: int
    activation_dtype_bytes: int
    num_attn_layers: int

    @property
    def kv_bytes_per_token_per_attn_layer(self) -> int:
        """K and V for one token of one attention layer."""
        return int(2 * self.num_key_value_heads * self.head_dim * self.kv_dtype_bytes)

    @property
    def q_bytes_per_token(self) -> int:
        return int(
            self.num_attention_heads * self.head_dim * self.activation_dtype_bytes
        )

    @property
    def out_bytes_per_token(self) -> int:
        # The partial attention output has the same shape as Q.
        return self.q_bytes_per_token

    @property
    def lse_bytes_per_token(self) -> int:
        # One float32 log-sum-exp per query head.
        return int(self.num_attention_heads * 4)


def collective_bytes_per_chunk(
    geometry: KvGeometry, n_remote_ranks: int, chunk_tokens: int
) -> int:
    """Wire bytes for one chunk across ALL attention layers.

    Each remote participant costs one Q block out and one partial output plus
    LSE back, per attention layer. Deliberately expressed per REMOTE RANK
    rather than per shard fraction: the traffic does not depend on how the KV
    is divided, only on how many ranks take part.
    """
    if n_remote_ranks < 0:
        raise DecoupledKvError("n_remote_ranks cannot be negative.")
    if chunk_tokens <= 0:
        raise DecoupledKvError("chunk_tokens must be positive.")
    per_token = (
        geometry.q_bytes_per_token
        + geometry.out_bytes_per_token
        + geometry.lse_bytes_per_token
    )
    return int(n_remote_ranks * chunk_tokens * per_token * geometry.num_attn_layers)


def _normalise(shares: Sequence[float], n_ranks: int | None = None) -> list[float]:
    vals = [float(s) for s in shares]
    if n_ranks is not None and len(vals) != n_ranks:
        raise DecoupledKvError(
            f"the share vector covers {len(vals)} ranks but the memory census "
            f"covers {n_ranks}."
        )
    if any(v < 0.0 for v in vals):
        raise DecoupledKvError(f"share vector {shares} has a non-negative violation.")
    total = sum(vals)
    if total <= 0.0:
        raise DecoupledKvError(f"share vector {shares} sums to zero.")
    return [v / total for v in vals]


@dataclasses.dataclass(frozen=True)
class VectorFit:
    shares: tuple[float, ...]
    need_gib: tuple[float, ...]
    free_gib: tuple[float, ...]
    fits: tuple[bool, ...]
    world_pool_tokens: float


def vector_feasibility(
    shares: Sequence[float],
    free_gib: Sequence[float],
    total_tokens: int,
    geometry: KvGeometry,
) -> VectorFit:
    """Can each rank hold its share of the decoupled pool?

    Under decoupling a rank stores ``share_i * total_tokens`` rows for ALL
    attention layers, so its footprint stops depending on which layers it owns
    -- that independence is the whole point. What remains rank-dependent is
    free memory, and a share vector that ignores it is how a phase-uniform
    layout becomes an OOM.
    """
    norm = _normalise(shares, len(free_gib))
    per_row = geometry.num_attn_layers * geometry.kv_bytes_per_token_per_attn_layer
    need = tuple(s * float(total_tokens) * per_row / GIB for s in norm)
    fits = tuple(n <= float(f) for n, f in zip(need, free_gib))
    # The pool this vector admits: the tightest rank scaled by its own share.
    pool = min(
        (float(f) * GIB / per_row) / s if s > 0 else float("inf")
        for f, s in zip(free_gib, norm)
    )
    return VectorFit(
        shares=tuple(norm),
        need_gib=need,
        free_gib=tuple(float(f) for f in free_gib),
        fits=fits,
        world_pool_tokens=pool,
    )


def deepest_feasible_rank0_layers(
    shares: Sequence[float],
    free_gib: Sequence[float],
    total_tokens: int,
    geometry: KvGeometry,
    weight_mib_per_layer: float,
    base_rank0_layers: int,
) -> int:
    """The depth ceiling a fixed share vector imposes on rank0.

    Deepening the cut moves weights onto rank0 and takes the same bytes away
    from its KV share, which the vector does not shrink. So a phase-uniform
    vector buys seam-free flips at the price of a hard limit on how deep the
    ladder may go, and that limit is solved here rather than discovered on
    metal.
    """
    fit = vector_feasibility(shares, free_gib, total_tokens, geometry)
    slack_mib = (fit.free_gib[0] - fit.need_gib[0]) * 1024.0
    if slack_mib < 0.0:
        raise DecoupledKvError(
            f"rank0 already cannot hold its {fit.shares[0]:.1%} share "
            f"({fit.need_gib[0]:,.2f} GiB) against {fit.free_gib[0]:,.2f} GiB "
            f"free at the base cut of {base_rank0_layers} layers."
        )
    return int(base_rank0_layers + slack_mib // float(weight_mib_per_layer))


def seam_rebalance_bytes(
    from_shares: Sequence[float],
    to_shares: Sequence[float],
    total_tokens: int,
    geometry: KvGeometry,
) -> int:
    """Rows that must move at the seam when only the SHARES differ.

    This is the quantity that separates the two halves of the prize. If both
    phases token-shard the same layers, the layout KIND is already uniform and
    a content-addressed key works (#703) whatever the shares are. If the shares
    are also identical this returns zero, and the seam byte move disappears
    entirely (#690).

    Only the ranks that GAIN rows are counted: what one rank sheds another
    takes, and counting both would double the wire traffic.
    """
    a = _normalise(from_shares)
    b = _normalise(to_shares, len(a))
    per_row = geometry.num_attn_layers * geometry.kv_bytes_per_token_per_attn_layer
    gained = sum(max(0.0, y - x) for x, y in zip(a, b))
    return round(gained * float(total_tokens) * per_row)


def fixed_vector_for_ladder(
    rung_free_gib: Sequence[Sequence[float]],
    total_tokens: int,
    geometry: KvGeometry,
) -> tuple[float, ...]:
    """One share vector held across every rung, so rung changes move no rows.

    The vector does NOT have to follow the cut. Letting it do so re-optimises
    each rung at the price of a KV rebalance on every step (measured on this
    rig: 0.80-1.07 GiB per step, 3.48 GiB across the ladder) and buys nothing,
    because a single vector chosen against the TIGHTEST rung is feasible at all
    of them.

    "Tightest" is per rank and not a single rung: deepening the cut starves
    rank0 of free bytes while FREEING rank1 and rank2, so the binding
    constraint for each rank comes from a different rung. Taking the per-rank
    minimum over rungs is what makes the result feasible everywhere, and the
    caller must still verify that with :func:`vector_feasibility` per rung --
    a vector proportional to the per-rank minima is the natural candidate, not
    a theorem.
    """
    if not rung_free_gib:
        raise DecoupledKvError("no rungs supplied; there is no ladder to fit.")
    n_ranks = len(rung_free_gib[0])
    for row in rung_free_gib:
        if len(row) != n_ranks:
            raise DecoupledKvError(
                "the rungs disagree on how many ranks the ladder has."
            )
    worst = [min(float(row[r]) for row in rung_free_gib) for r in range(n_ranks)]
    if sum(worst) <= 0.0:
        raise DecoupledKvError(
            "every rank is out of free memory at some rung; no shared vector "
            "exists and the ladder cannot be served decoupled."
        )
    return tuple(_normalise(worst))


@dataclasses.dataclass(frozen=True)
class QuantisedShares:
    """Integer token-axis ratios for the EXISTING uneven-DCP owner rule."""

    ratios: tuple[int, ...]
    period: int
    realized_shares: tuple[float, ...]
    max_share_error: float


def quantize_shares(
    target: Sequence[float],
    period: int,
    free_gib: Sequence[float],
    total_tokens: int,
    geometry: KvGeometry,
) -> QuantisedShares:
    """Round a FEASIBLE fractional target onto the integer owner rule.

    ``distributed/utils.py:407`` defines the token axis as integer ratios: slot
    ``L`` belongs to rank ``r`` iff ``(L % S)`` is in ``[lo_r, hi_r)`` with
    ``S = sum(ratios)``. So the realized share is exactly ``ratio_r / S`` and
    the solver's fractions have to be quantised before they can drive it.

    This is a ROUNDING step, not a re-optimisation. An infeasible target is
    refused rather than repaired into something feasible but unrelated -- the
    caller is expected to have obtained a feasible target from
    :func:`vector_feasibility` or :func:`fixed_vector_for_ladder`, and silently
    substituting a different vector would hide that it did not.

    Rounding itself is not neutral: rounding a tight rank UP buys rows it
    cannot hold, and the resulting OOM would read as a decoupling bug rather
    than a rounding bug. So the rounded vector is re-checked and repaired by
    moving slots from any over-committed rank to the rank with the most slack.
    """
    if period < len(target):
        raise DecoupledKvError(
            f"period {period} cannot give {len(target)} ranks a slot each."
        )
    norm = _normalise(target, len(free_gib))

    # The target itself must be feasible; quantisation does not rescue it.
    base = vector_feasibility(norm, free_gib, total_tokens, geometry)
    if not all(base.fits):
        bad = [i for i, ok in enumerate(base.fits) if not ok]
        raise DecoupledKvError(
            f"the target share vector is not feasible on rank(s) {bad}: "
            f"needs {[round(base.need_gib[i], 2) for i in bad]} GiB against "
            f"{[round(base.free_gib[i], 2) for i in bad]} GiB free. Quantisation "
            "rounds a feasible target; it does not repair an infeasible one, "
            "because substituting a different vector would hide that the "
            "target was wrong."
        )

    # Largest-remainder apportionment: exact sum, minimal rounding drift.
    raw = [s * period for s in norm]
    ratios = [int(x) for x in raw]
    rem = period - sum(ratios)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - ratios[i], reverse=True)
    for i in range(rem):
        ratios[order[i % len(order)]] += 1

    if any(r <= 0 for r in ratios):
        zero = [i for i, r in enumerate(ratios) if r <= 0]
        raise DecoupledKvError(
            f"rank(s) {zero} quantise to zero slots at period {period}. A "
            "zero-slot rank holds no rows but is still asked for Q and a "
            "partial output, paying the full collective cost for nothing. Use "
            "a finer period, or drop those ranks from the shard set explicitly "
            "so the cost model sees fewer participants."
        )

    # Repair rounding-induced infeasibility only.
    for _ in range(period):
        fit = vector_feasibility(ratios, free_gib, total_tokens, geometry)
        over = [i for i, ok in enumerate(fit.fits) if not ok]
        if not over:
            break
        donor = max(over, key=lambda i: fit.need_gib[i] - fit.free_gib[i])
        slack = [fit.free_gib[i] - fit.need_gib[i] for i in range(len(ratios))]
        taker = max(range(len(ratios)), key=lambda i: slack[i])
        if ratios[donor] <= 1 or taker == donor:
            raise DecoupledKvError(
                f"no feasible integer split exists at period {period}: rank "
                f"{donor} is over budget even at its smallest non-zero share."
            )
        ratios[donor] -= 1
        ratios[taker] += 1

    realized = tuple(r / period for r in ratios)
    return QuantisedShares(
        ratios=tuple(ratios),
        period=int(period),
        realized_shares=realized,
        max_share_error=max(abs(a - b) for a, b in zip(realized, norm)),
    )
