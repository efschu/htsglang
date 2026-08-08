"""#651: expert weights resident on DISK, faulted onto the device on demand.

WHY THIS TIER HAS TO EXIST. ``spill_tiers`` already names ``expert_host_ram``,
and on a discrete-GPU box that is the right answer: host RAM is a separate pool,
so moving an expert there frees real VRAM. On the gfx1103 laptop it frees
nothing, because the GPU allocates from host RAM through GTT -- an expert moved
"to host RAM" has not left the memory being reclaimed. 29.5 GiB total, 22.7 GiB
of Q4 weights, and the desktop competing for the remainder is a configuration
that cannot be fixed by moving bytes sideways.

Disk is different in the one way that matters here: pages backed by the GGUF
file are CLEAN PAGE CACHE. The kernel may evict them under pressure without
writing anything back, so their residency is advisory rather than a claim on
memory. That is what actually returns RAM on this machine.

WHY A SPILL LIST CANNOT BE FROZEN. The routing census that licenses this design
(FINAL_651.md 9.7.1) measured 214,400 lookups over 566 tokens from five short
prompts: the coldest 20% of experts took 0.03% of lookups and 19.3% were never
routed at all. That licenses the SHAPE of the distribution -- a deep cold tail
-- and nothing more. Five prompts cannot tell you WHICH experts a user's actual
work will need, and an expert that is cold for "count to forty" is not
necessarily cold for Rust. Freezing that list would bake a five-prompt sample
into the serving configuration.

So residency here is a CACHE, not a partition:

  * a hot region sized once, holding the experts believed hot;
  * a small staging pool of rows that any cold expert can be faulted into;
  * LRU eviction within the staging pool;
  * and a ``refresh`` that re-derives the hot set from live counters, so the
    residency follows the workload instead of the census.

WHAT THIS MODULE DOES NOT DO. It moves no memory and touches no device. It
answers one question -- "which device row currently holds expert (layer, e),
and if none, which row should it be faulted into" -- so that the answer can be
unit-tested without a GPU, which is the only way the eviction logic gets
exercised at all. The copy itself belongs to the caller.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

#: Row value meaning "this expert is not on the device right now".
NOT_RESIDENT: int = -1


@dataclass
class FaultResult:
    """What the caller must do to make an expert usable.

    ``row`` is always a valid device row on return. ``copied_from_disk`` says
    whether the caller has to perform the transfer, and ``evicted`` names the
    expert whose row was taken -- reported so a caller can account for the
    eviction rather than discover it later as a miss.
    """

    row: int
    copied_from_disk: bool
    evicted: Optional[int] = None


@dataclass
class LayerResidency:
    """Residency for one MoE layer.

    Hot experts own a device row for the lifetime of the configuration. Cold
    experts share the staging rows. The split is per layer because routing is
    per layer: an expert index means nothing across layers, and a global LRU
    would let one layer's traffic evict another's working set.
    """

    num_experts: int
    hot_experts: Tuple[int, ...]
    staging_rows: int

    #: expert id -> device row, or NOT_RESIDENT.
    _row_of: Dict[int, int] = field(default_factory=dict, init=False)
    #: device row -> expert id currently in it (staging rows only).
    _expert_in_row: Dict[int, int] = field(default_factory=dict, init=False)
    #: staging rows, least-recently-used first.
    _lru: List[int] = field(default_factory=list, init=False)

    faults: int = field(default=0, init=False)
    hits: int = field(default=0, init=False)
    evictions: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.staging_rows < 1:
            # A pool of zero would make every cold expert unservable, which is
            # a silent correctness failure rather than a slow one.
            raise ValueError("staging_rows must be >= 1")
        if len(set(self.hot_experts)) != len(self.hot_experts):
            raise ValueError("hot_experts contains duplicates")
        for e in self.hot_experts:
            if not 0 <= e < self.num_experts:
                raise ValueError(f"hot expert {e} out of range")
        self._rebuild()

    def _rebuild(self) -> None:
        self._row_of = {e: i for i, e in enumerate(self.hot_experts)}
        base = len(self.hot_experts)
        self._expert_in_row = {}
        self._lru = [base + i for i in range(self.staging_rows)]

    @property
    def total_rows(self) -> int:
        """Device rows this layer needs -- the compacted tensor's height."""
        return len(self.hot_experts) + self.staging_rows

    def row_of(self, expert: int) -> int:
        return self._row_of.get(expert, NOT_RESIDENT)

    def touch(self, expert: int) -> None:
        """Mark a resident staging expert as recently used."""
        row = self._row_of.get(expert, NOT_RESIDENT)
        if row is not NOT_RESIDENT and row in self._expert_in_row:
            self._lru.remove(row)
            self._lru.append(row)

    def acquire(self, expert: int) -> FaultResult:
        """Ensure ``expert`` has a device row, faulting it in if needed."""
        if not 0 <= expert < self.num_experts:
            raise ValueError(f"expert {expert} out of range")

        row = self._row_of.get(expert, NOT_RESIDENT)
        if row is not NOT_RESIDENT:
            self.hits += 1
            self.touch(expert)
            return FaultResult(row=row, copied_from_disk=False)

        # Cold and not resident: take the least recently used staging row.
        victim_row = self._lru.pop(0)
        evicted = self._expert_in_row.pop(victim_row, None)
        if evicted is not None:
            del self._row_of[evicted]
            self.evictions += 1

        self._row_of[expert] = victim_row
        self._expert_in_row[victim_row] = expert
        self._lru.append(victim_row)
        self.faults += 1
        return FaultResult(row=victim_row, copied_from_disk=True, evicted=evicted)

    def refresh_hot(self, hot_experts: Iterable[int]) -> None:
        """Re-derive the hot set, e.g. from live routing counters.

        Everything staged is dropped: after a change of hot set the staging
        rows may belong to experts that are now hot, and reconciling that in
        place would be more code than simply refilling on demand. Refresh is a
        rare, deliberate operation; a fault is cheap.
        """
        hot = tuple(hot_experts)
        if len(set(hot)) != len(hot):
            raise ValueError("hot_experts contains duplicates")
        for e in hot:
            if not 0 <= e < self.num_experts:
                raise ValueError(f"hot expert {e} out of range")
        self.hot_experts = hot
        self._rebuild()

    def miss_rate(self) -> float:
        total = self.hits + self.faults
        return (self.faults / total) if total else 0.0


class ExpertDiskResidency:
    """Residency across all MoE layers, with a lock.

    The lock is not decoration: faulting mutates the row map that a forward
    pass is simultaneously reading, and on this server the scheduler thread and
    a refresh triggered from the HTTP layer are different threads.
    """

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        hot_per_layer: Dict[int, Iterable[int]],
        staging_rows: int,
    ) -> None:
        self._lock = threading.Lock()
        self.layers: Dict[int, LayerResidency] = {}
        for layer in range(num_layers):
            self.layers[layer] = LayerResidency(
                num_experts=num_experts,
                hot_experts=tuple(hot_per_layer.get(layer, ())),
                staging_rows=staging_rows,
            )

    def acquire(self, layer: int, expert: int) -> FaultResult:
        with self._lock:
            return self.layers[layer].acquire(expert)

    def row_of(self, layer: int, expert: int) -> int:
        with self._lock:
            return self.layers[layer].row_of(expert)

    def refresh_from_counts(self, counts, hot_fraction: float) -> None:
        """Re-derive every layer's hot set from a [layers, experts] count matrix.

        ``counts`` is anything indexable as ``counts[layer][expert]`` -- the
        expert-distribution recorder's ``logical_count`` summed over passes is
        exactly this shape, which is the point: the residency follows measured
        traffic instead of a census taken once.
        """
        if not 0.0 < hot_fraction <= 1.0:
            raise ValueError("hot_fraction must be in (0, 1]")
        with self._lock:
            for layer, res in self.layers.items():
                row = list(counts[layer])
                keep = max(1, int(round(len(row) * hot_fraction)))
                ranked = sorted(range(len(row)), key=lambda e: row[e], reverse=True)
                res.refresh_hot(sorted(ranked[:keep]))

    def totals(self) -> Dict[str, float]:
        with self._lock:
            hits = sum(r.hits for r in self.layers.values())
            faults = sum(r.faults for r in self.layers.values())
            evictions = sum(r.evictions for r in self.layers.values())
        total = hits + faults
        return {
            "hits": hits,
            "faults": faults,
            "evictions": evictions,
            "miss_rate": (faults / total) if total else 0.0,
        }


def plan_hot_sets(
    counts,
    hot_fraction: float,
) -> Dict[int, List[int]]:
    """Choose each layer's hot experts from a count matrix.

    Separate from the residency object so a spill plan can be computed, printed
    and argued about before anything is allocated -- the sizing decision is the
    part a human needs to see.
    """
    if not 0.0 < hot_fraction <= 1.0:
        raise ValueError("hot_fraction must be in (0, 1]")
    plan: Dict[int, List[int]] = {}
    for layer in range(len(counts)):
        row = list(counts[layer])
        keep = max(1, int(round(len(row) * hot_fraction)))
        ranked = sorted(range(len(row)), key=lambda e: row[e], reverse=True)
        plan[layer] = sorted(ranked[:keep])
    return plan


def bytes_saved(
    num_layers: int,
    num_experts: int,
    bytes_per_expert: int,
    hot_fraction: float,
    staging_rows: int,
) -> int:
    """Device bytes this configuration does NOT hold.

    Staging rows are charged, because they are real device memory: a plan that
    spills 80% of experts into a staging pool almost as large as the hot region
    has saved far less than its spill fraction suggests.
    """
    hot = max(1, int(round(num_experts * hot_fraction)))
    resident_rows = hot + staging_rows
    spilled_rows = max(0, num_experts - resident_rows)
    return spilled_rows * bytes_per_expert * num_layers
