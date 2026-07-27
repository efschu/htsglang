# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the per-wave expert LUT build
(``MoEExpertOffloadCache._build_lut``).

Two gates:
  * EQUIVALENCE -- the vectorized build is bit-identical to the historical
    per-entry ``lut[e] = s`` loop for every input shape the wave loop can
    produce (static and hot residency, empty / full / permuted slot maps,
    int32 and int64 topk dtypes), and drives ``_remap`` to the same ids.
  * COST -- the build issues a CONSTANT number of tensor ops, independent of
    how many experts a wave needs. The per-entry loop scaled with the needed
    count (one blocking device scalar store each), which is what made the
    offload prefill path host-bound; this gate fails on that implementation.

Run:  python -m pytest tests/moe_offload/test_build_lut.py -q
"""

import os
import sys

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from sglang.srt.layers.moe.expert_offload import (  # noqa: E402
    ExpertResidencyPlanner,
    MoEExpertOffloadCache,
    plan_token_waves,
)


def _cache(num_local_experts):
    """A cache instance carrying only what ``_build_lut`` reads (no layer, no
    CUDA, no install)."""
    c = MoEExpertOffloadCache.__new__(MoEExpertOffloadCache)
    c.num_local_experts = num_local_experts
    return c


def _ref_build_lut(num_local_experts, slot_of_needed, dtype, device="cpu"):
    """Pre-fix reference: one scalar store per needed expert."""
    lut = torch.full((num_local_experts,), -1, dtype=dtype, device=device)
    for e, s in slot_of_needed.items():
        lut[e] = s
    return lut


class _OpCounter(TorchDispatchMode):
    def __init__(self):
        self.n = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.n += 1
        return func(*args, **(kwargs or {}))


def _resolve_maps(E, R, C, seed, hot=False):
    """Slot maps exactly as the wave loop produces them, for a random chunk."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(E, generator=g)[:8] for _ in range(64)])
    kw = {}
    if hot:
        hot_ids = sorted(torch.randperm(E, generator=g)[:R].tolist())
        kw = {
            "resident_ids": frozenset(hot_ids),
            "resident_slot": {e: i for i, e in enumerate(hot_ids)},
        }
    planner = ExpertResidencyPlanner(
        num_local_experts=E, resident_count=R, scratch=C, **kw
    )
    rows_list = ids.tolist()
    waves = plan_token_waves(rows_list, R, C, planner.resident_ids)
    out = []
    for rows in waves:
        needed = sorted({e for r in rows for e in rows_list[r] if e >= 0})
        out.append((planner.resolve(needed)[0], ids[rows]))
    return out


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("hot", [False, True])
def test_matches_per_entry_reference(dtype, hot):
    E, R, C = 64, 16, 8
    cache = _cache(E)
    maps = _resolve_maps(E, R, C, seed=7, hot=hot)
    assert len(maps) > 4  # the chunk really did wave-split
    for slot_of_needed, wave_ids in maps:
        got = cache._build_lut(slot_of_needed, dtype, "cpu")
        ref = _ref_build_lut(E, slot_of_needed, dtype)
        assert got.dtype == ref.dtype
        assert torch.equal(got, ref)
        # and the remap the LUT feeds is identical for the wave's own ids
        wave_ids = wave_ids.to(dtype)
        assert torch.equal(
            MoEExpertOffloadCache._remap(wave_ids, got),
            MoEExpertOffloadCache._remap(wave_ids, ref),
        )


def test_edge_cases():
    E = 32
    cache = _cache(E)
    cases = [
        {},  # nothing needed
        {0: 0},  # single resident
        {31: 16},  # single spill, high id
        {e: e for e in range(E)},  # every expert needed
        {5: 2, 1: 0, 30: 17, 2: 1},  # unsorted insertion order
        {0: 0, 31: 8, 8: 8},  # slot reused by two ids
    ]
    for slot_of_needed in cases:
        for dtype in (torch.int32, torch.int64):
            got = cache._build_lut(slot_of_needed, dtype, "cpu")
            assert torch.equal(got, _ref_build_lut(E, slot_of_needed, dtype))


def test_build_cost_is_independent_of_needed_count():
    """The hot-path gate: op count must not grow with the wave's expert count.

    The per-entry loop issued one store per needed expert -- on CUDA a blocking
    host->device scalar write, ~640k of them per 2048-token prefill chunk.
    """
    E = 256
    cache = _cache(E)
    small = {e: e for e in range(4)}
    large = {e: e for e in range(64)}
    counts = []
    for slot_of_needed in (small, large):
        counter = _OpCounter()
        with counter:
            cache._build_lut(slot_of_needed, torch.int64, "cpu")
        counts.append(counter.n)
    assert counts[0] == counts[1], (
        f"LUT build cost scales with the needed-expert count "
        f"({counts[0]} ops for 4 experts vs {counts[1]} for 64); it must be a "
        f"single vectorized scatter"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
