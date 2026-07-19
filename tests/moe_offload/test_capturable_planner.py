# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the Stage-3 CUDA-graph-capturable routing math.

Proves (without CUDA) that the fixed-shape on-device index math
(``build_capturable_luts`` + ``prepare_capturable_remap``) reproduces the eager
``ExpertResidencyPlanner.resolve()`` + ``_build_lut`` + ``_remap`` slot map and
fetch plan BIT-FOR-BIT for the single-wave case -- the Δ=0 correctness gate for
graph capture, established before any GPU boot.

Run:  python -m pytest tests/moe_offload/test_capturable_planner.py -q
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from sglang.srt.layers.moe.expert_offload import (  # noqa: E402
    ExpertResidencyPlanner,
    build_capturable_luts,
    prepare_capturable_remap,
)


def _ref_remap(planner, resident_slot, spill_pool_index, topk_ids):
    """Eager reference: resolve() + _build_lut + _remap, plus the fetch pool
    rows, computed straight from the planner semantics."""
    ids = topk_ids.tolist()
    needed = sorted({e for row in ids for e in row if e >= 0})
    slot_of_needed, fetch_plan = planner.resolve(needed)
    E = planner.num_local_experts
    lut = {e: -1 for e in range(E)}
    lut.update(slot_of_needed)
    ref = [[lut[e] if e >= 0 else -1 for e in row] for row in ids]
    # pool row for each fetched scratch slot, in slot order (R, R+1, ...)
    def pool_row(e):
        return spill_pool_index[e] if spill_pool_index is not None else e - planner.resident_count
    fetch_rows = [pool_row(e) for e, _slot in fetch_plan]
    return torch.tensor(ref, dtype=topk_ids.dtype), fetch_rows, len(fetch_plan)


def _make_hotset(E, R, seed):
    """Construct a frozen hot-set exactly as _freeze_hotset would: hot experts
    sorted -> slots [0,R); cold sorted -> pool rows [0,E-R)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(E, generator=g).tolist()
    hot = sorted(perm[:R])
    cold = sorted(e for e in range(E) if e not in set(hot))
    resident_slot = {e: i for i, e in enumerate(hot)}
    spill_pool_index = {e: j for j, e in enumerate(cold)}
    return frozenset(hot), resident_slot, spill_pool_index


def _random_single_wave_ids(E, R, C, resident_ids, seed, T=3, k=8):
    """Random topk_ids whose unique SPILL experts <= C (single-wave)."""
    g = torch.Generator().manual_seed(seed)
    is_spill_id = (lambda e: e not in resident_ids) if resident_ids is not None else (lambda e: e >= R)
    for _ in range(200):
        ids = torch.randint(0, E, (T, k), generator=g)
        # inject some -1 padding
        mask = torch.rand(T, k, generator=g) < 0.15
        ids = torch.where(mask, torch.full_like(ids, -1), ids)
        spill = {int(e) for row in ids.tolist() for e in row if e >= 0 and is_spill_id(int(e))}
        if len(spill) <= C:
            return ids
    pytest.skip("could not sample a single-wave case")


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("hot", [False, True])
def test_capturable_matches_resolve(seed, hot):
    E, R, C = 256, 64, 16
    if hot:
        resident_ids, resident_slot, spill_pool_index = _make_hotset(E, R, seed)
    else:
        resident_ids = resident_slot = spill_pool_index = None
    planner = ExpertResidencyPlanner(
        num_local_experts=E, resident_count=R, scratch=C,
        resident_ids=resident_ids, resident_slot=resident_slot,
    )
    ids = _random_single_wave_ids(E, R, C, resident_ids, seed)

    ref_remap, ref_fetch_rows, ref_num_spill = _ref_remap(
        planner, resident_slot, spill_pool_index, ids
    )

    rslot_lut, is_spill, spool_lut = build_capturable_luts(
        E, R, resident_slot, spill_pool_index, device="cpu"
    )
    remapped, src_row, num_spill = prepare_capturable_remap(
        ids, rslot_lut, is_spill, spool_lut, R, C
    )

    # (a) remap bit-identical to the eager path
    assert torch.equal(remapped, ref_remap), (
        f"remap mismatch\n got {remapped.tolist()}\n ref {ref_remap.tolist()}"
    )
    # (b) num_spill matches
    assert int(num_spill) == ref_num_spill
    # (c) fetch source rows match, in scratch-slot order
    assert src_row[:ref_num_spill].tolist() == ref_fetch_rows
    # (d) no remapped id lands in an unused scratch slot [R+num_spill, R+C)
    used_max = R + ref_num_spill
    vals = remapped[remapped >= 0]
    assert not ((vals >= used_max) & (vals < R + C)).any(), (
        "a remapped id referenced an unused scratch slot"
    )


def test_all_resident_no_fetch():
    E, R, C = 256, 64, 16
    planner = ExpertResidencyPlanner(num_local_experts=E, resident_count=R, scratch=C)
    ids = torch.tensor([[0, 1, 2, 63], [5, 10, 63, 0]], dtype=torch.long)  # all < R
    ref_remap, ref_fetch_rows, ref_num_spill = _ref_remap(planner, None, None, ids)
    rslot_lut, is_spill, spool_lut = build_capturable_luts(E, R, None, None, "cpu")
    remapped, src_row, num_spill = prepare_capturable_remap(ids, rslot_lut, is_spill, spool_lut, R, C)
    assert torch.equal(remapped, ref_remap)
    assert int(num_spill) == 0 == ref_num_spill


def test_exactly_c_spill_boundary():
    E, R, C = 256, 64, 16
    planner = ExpertResidencyPlanner(num_local_experts=E, resident_count=R, scratch=C)
    spill = list(range(R, R + C))  # exactly C distinct spill experts
    ids = torch.tensor([spill[:8], spill[8:]], dtype=torch.long)
    ref_remap, ref_fetch_rows, ref_num_spill = _ref_remap(planner, None, None, ids)
    rslot_lut, is_spill, spool_lut = build_capturable_luts(E, R, None, None, "cpu")
    remapped, src_row, num_spill = prepare_capturable_remap(ids, rslot_lut, is_spill, spool_lut, R, C)
    assert torch.equal(remapped, ref_remap)
    assert int(num_spill) == C == ref_num_spill
    assert src_row.tolist() == ref_fetch_rows  # all C slots used


def test_all_padding():
    E, R, C = 32, 8, 8
    planner = ExpertResidencyPlanner(num_local_experts=E, resident_count=R, scratch=C)
    ids = torch.full((2, 4), -1, dtype=torch.long)
    ref_remap, _, ref_num_spill = _ref_remap(planner, None, None, ids)
    rslot_lut, is_spill, spool_lut = build_capturable_luts(E, R, None, None, "cpu")
    remapped, src_row, num_spill = prepare_capturable_remap(ids, rslot_lut, is_spill, spool_lut, R, C)
    assert torch.equal(remapped, ref_remap)  # all -1
    assert int(num_spill) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
