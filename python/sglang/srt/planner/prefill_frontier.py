"""#702: solve the PP cut for PREFILL SPEED, with the pool price stated honestly.

The capacity solver answers "what cut holds the most context". This one answers
the question actually asked: **more layers on the fast card, what does it
cost?** The two objectives disagree, so the answer is a frontier rather than a
winner, and the caller picks.

Three prices are charged against every candidate, because quoting only the first
is how a cut gets recommended that cannot serve:

1. **Compute speedup** — the pipelined stage time against the incumbent's. This
   is an UPPER BOUND until the two-point timing calibration lands (slice 1a-i):
   ``fixed_ms`` defaults to zero, attributing all measured time to layers, which
   is the optimistic end of the family.
2. **Pool**, in the two regimes that actually exist. **Coupled** (KV stays with
   its layer) prices by the min-rule and collapses as the cut deepens.
   **Decoupled** (#704b, KV token-sharded) prices by the sum-rule and is
   *exactly* cut-independent, because total weight bytes and total GDN state are
   both invariant under a re-cut — only their distribution moves.
3. **Collective overhead**, which exists only in the decoupled regime, from the
   measured link profile.

The result that makes this worth solving rather than tabulating: **net speedup
without cross-chunk pipelining is not monotone in depth.** Past some cut the
collective overhead grows faster than the compute gain, so deeper is *worse*,
not merely diminishing. Candidates beyond that point are marked
``needs_pipelining`` — they are unreachable until the §4.2g lever is built, and
recommending one without it would be recommending a regression.

No rig constants: every quantity is injected. Today's figures are calibration
data and live in tests.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence


class PrefillFrontierError(ValueError):
    """A frontier question that cannot be answered as posed."""


@dataclasses.dataclass(frozen=True)
class FrontierPoint:
    counts: tuple[int, ...]
    attn_counts: tuple[int, ...]
    #: Pipelined compute speedup vs the incumbent. UPPER BOUND (fixed_ms=0).
    compute_speedup: float
    coupled_pool_tokens: float
    decoupled_pool_tokens: float
    #: Collective overhead fraction; decoupled regime only.
    overhead: float
    #: Speedup net of overhead, with today's machinery.
    net_no_pipelining: float
    #: Speedup net of overhead once cross-chunk pipelining exists.
    net_pipelined: float
    #: True when no shallower candidate is worse without pipelining, i.e. this
    #: cut is only reachable once the lever is built.
    needs_pipelining: bool


@dataclasses.dataclass(frozen=True)
class PrefillFrontier:
    points: tuple[FrontierPoint, ...]
    incumbent: tuple[int, ...]
    incumbent_pool_tokens: float
    #: Self-labelling. False whenever any input was extrapolated rather than
    #: measured, so a caller cannot mistake a projection for an observation.
    measured: bool

    def best_without_pipelining(self) -> FrontierPoint:
        return max(self.points, key=lambda p: p.net_no_pipelining)

    def best_with_pipelining(self) -> FrontierPoint:
        return max(self.points, key=lambda p: p.net_pipelined)


def solve_prefill_frontier(
    total_layers: int,
    n_stages: int,
    incumbent: Sequence[int],
    incumbent_pool_tokens: float,
    ms_per_layer: Sequence[float],
    attn_counts_for: Callable[[Sequence[int]], Sequence[int]],
    available_bytes_for: Callable[[Sequence[int], Sequence[int]], Sequence[float]],
    kv_bytes_per_token_per_attn_layer: int,
    total_attn_layers: int,
    gather_mib_per_attn_layer: float,
    link_mib_per_s: Sequence[float],
    max_rank0_layers: int | None = None,
    measured: bool = False,
) -> PrefillFrontier:
    """Enumerate cuts and price each on speed, pool and overhead.

    ``available_bytes_for(counts, attn_counts)`` returns the per-rank bytes
    left for KV under that cut. It is injected because it depends on the
    boot's own budget instruments, which no arithmetic here can invent.
    """
    incumbent = tuple(int(c) for c in incumbent)
    if len(incumbent) != n_stages:
        raise PrefillFrontierError(
            f"incumbent has {len(incumbent)} stages against n_stages={n_stages}."
        )
    if sum(incumbent) != int(total_layers):
        raise PrefillFrontierError(
            f"incumbent {incumbent} sums to {sum(incumbent)}, not {total_layers}."
        )
    if len(ms_per_layer) != n_stages or len(link_mib_per_s) != n_stages:
        raise PrefillFrontierError("timing/link vectors must cover every stage.")

    def pipelined(counts: Sequence[int]) -> float:
        return max(float(ms_per_layer[i]) * int(counts[i]) for i in range(n_stages))

    base = pipelined(incumbent)
    points: list[FrontierPoint] = []

    for n0 in range(int(incumbent[0]), int(total_layers) - (n_stages - 1) + 1):
        if max_rank0_layers is not None and n0 > int(max_rank0_layers):
            break
        # For a given lead depth, the tail split that minimises pipelined time.
        best: tuple[int, ...] | None = None
        best_ms = float("inf")
        for tail in _tails(int(total_layers) - n0, n_stages - 1):
            counts = (n0, *tail)
            ms = pipelined(counts)
            if ms < best_ms - 1e-12:
                best_ms, best = ms, counts
        if best is None:
            continue

        attn = tuple(int(a) for a in attn_counts_for(best))
        if any(a <= 0 for a in attn):
            continue
        avail = [float(x) for x in available_bytes_for(best, attn)]
        if min(avail) <= 0.0:
            continue

        cell = int(kv_bytes_per_token_per_attn_layer)
        coupled = min(avail[i] / (attn[i] * cell) for i in range(n_stages))
        decoupled = sum(avail) / (int(total_attn_layers) * cell)

        comp = [float(ms_per_layer[i]) * best[i] for i in range(n_stages)]
        exp = [
            attn[i]
            * float(gather_mib_per_attn_layer)
            / float(link_mib_per_s[i])
            * 1000.0
            for i in range(n_stages)
        ]
        with_gather = max(comp[i] + exp[i] for i in range(n_stages))
        overhead = with_gather / best_ms - 1.0

        points.append(
            FrontierPoint(
                counts=best,
                attn_counts=attn,
                compute_speedup=base / best_ms,
                coupled_pool_tokens=coupled,
                decoupled_pool_tokens=decoupled,
                overhead=overhead,
                net_no_pipelining=base / with_gather,
                net_pipelined=base / best_ms,
                needs_pipelining=False,
            )
        )

    # A candidate needs the lever when every shallower candidate beats it
    # without one: depth you cannot cash in until the overhead is hidden.
    running = float("-inf")
    finished: list[FrontierPoint] = []
    for p in points:
        needs = p.net_no_pipelining < running - 1e-12
        finished.append(dataclasses.replace(p, needs_pipelining=needs))
        running = max(running, p.net_no_pipelining)

    return PrefillFrontier(
        points=tuple(finished),
        incumbent=incumbent,
        incumbent_pool_tokens=float(incumbent_pool_tokens),
        measured=bool(measured),
    )


def _tails(remaining: int, stages: int) -> list[tuple[int, ...]]:
    if stages == 1:
        return [(remaining,)] if remaining >= 1 else []
    out: list[tuple[int, ...]] = []
    for n in range(1, remaining - (stages - 1) + 1):
        for rest in _tails(remaining - n, stages - 1):
            out.append((n, *rest))
    return out
