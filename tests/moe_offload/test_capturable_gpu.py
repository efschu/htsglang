# SPDX-License-Identifier: Apache-2.0
"""GPU isolation proof for the Stage-3 capturable offload path (requires CUDA).

Proves that ``prepare_capturable_remap`` + the captured UVA gather
(``device_view_of_pinned`` + ``index_select``) is BIT-IDENTICAL to the eager
offload path (``resolve`` + ``_build_lut`` + ``_remap`` + per-expert ``_fetch``
row copies) on CUDA -- both when run eagerly and when captured into a CUDA
graph and replayed with fresh inputs. Static and frozen-hot-set layouts.

Context (measured on the 35B validation vehicle, 2026-07-19): an end-to-end
captured-decode vs eager-decode server comparison can NOT reach machine-zero on
stock sglang -- at fraction=1.0 with no offload code active at all, graph vs
eager already shows an argmax-token logprob Δ of ~3.4e-2 (capture-gated
dual-stream branches in qwen2_moe/qwen3_5 + capture-context kernel selection).
This test therefore pins the Δ=0 claim where it is provable: the Stage-3
machinery itself contributes EXACTLY ZERO divergence.

Run:  python -m pytest tests/moe_offload/test_capturable_gpu.py -q
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from sglang.srt.layers.moe.expert_offload import (  # noqa: E402
    ExpertResidencyPlanner,
    build_capturable_luts,
    device_view_of_pinned,
    prepare_capturable_remap,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

E, R, C, K = 256, 64, 16, 8


def _eager_reference(planner, topk_ids, pools, resident_bufs, spill_pool_index):
    """Replicates run_waves' single-wave fast path + _fetch exactly."""
    ids_list = topk_ids.tolist()
    needed = sorted({e for row in ids_list for e in row if e >= 0})
    slot_of_needed, fetch_plan = planner.resolve(needed)
    for attr, spill in pools.items():
        dst = resident_bufs[attr]
        for expert_id, slot in fetch_plan:
            row = (
                spill_pool_index[expert_id]
                if spill_pool_index is not None
                else expert_id - R
            )
            dst[slot].copy_(spill[row], non_blocking=True)
    torch.cuda.synchronize()
    lut = torch.full((E,), -1, dtype=topk_ids.dtype, device=topk_ids.device)
    for e, s in slot_of_needed.items():
        lut[e] = s
    return torch.where(topk_ids >= 0, lut[topk_ids.clamp(min=0)], topk_ids)


def _build_case(hot_layout: bool):
    dev = torch.device("cuda:0")
    torch.manual_seed(7)
    full = {
        "w13_qweight": torch.randint(
            -(2**31), 2**31 - 1, (E, 64, 32), dtype=torch.int32
        ),
        "w2_scales": torch.randn(E, 8, 96, dtype=torch.float16),
    }
    if hot_layout:
        g = torch.Generator().manual_seed(99)
        hot = sorted(torch.randperm(E, generator=g)[:R].tolist())
        cold = [e for e in range(E) if e not in set(hot)]
        resident_slot = {e: i for i, e in enumerate(hot)}
        spill_pool_index = {e: j for j, e in enumerate(cold)}
        resident_ids = frozenset(hot)
    else:
        hot, cold = list(range(R)), list(range(R, E))
        resident_slot = spill_pool_index = resident_ids = None

    pools, resident_a, resident_b = {}, {}, {}
    for attr, t in full.items():
        pool = torch.empty((E - R,) + t.shape[1:], dtype=t.dtype).pin_memory()
        for j, e in enumerate(cold):
            pool[j].copy_(t[e])
        buf = torch.zeros((R + C,) + t.shape[1:], dtype=t.dtype, device=dev)
        for i, e in enumerate(hot):
            buf[i].copy_(t[e])
        pools[attr] = pool
        resident_a[attr] = buf          # eager side
        resident_b[attr] = buf.clone()  # capturable side
    torch.cuda.synchronize()

    planner = ExpertResidencyPlanner(
        num_local_experts=E,
        resident_count=R,
        scratch=C,
        resident_ids=resident_ids,
        resident_slot=resident_slot,
    )
    lut_r, lut_s, lut_p = build_capturable_luts(
        E, R, resident_slot, spill_pool_index, device=dev
    )
    pool_dev = {a: device_view_of_pinned(p) for a, p in pools.items()}
    scratch_dst = {a: resident_b[a][R : R + C] for a in pools}

    def capturable(topk_ids):
        remapped, src_row, _ = prepare_capturable_remap(
            topk_ids, lut_r, lut_s, lut_p, R, C
        )
        for a, pv in pool_dev.items():
            torch.index_select(pv, 0, src_row, out=scratch_dst[a])
        return remapped

    return planner, pools, resident_a, resident_b, spill_pool_index, capturable


def _assert_used_slots_equal(resident_a, resident_b, remapped, ctx):
    # Only slots referenced by tokens matter: unused scratch is inert by design
    # (eager leaves it stale, capturable fills pool row 0 -- design §7.3).
    used = torch.unique(remapped[remapped >= 0]).tolist()
    for a in resident_a:
        for s in used:
            assert torch.equal(resident_a[a][s], resident_b[a][s]), (
                f"slot {s} content diverged (attr {a}, {ctx})"
            )


@pytest.mark.parametrize("hot_layout", [False, True], ids=["static", "hotset"])
def test_eager_equivalence(hot_layout):
    planner, pools, ra, rb, spi, capturable = _build_case(hot_layout)
    dev = torch.device("cuda:0")
    for trial in range(50):
        g = torch.Generator().manual_seed(1000 + trial)
        t = int(torch.randint(1, 3, (1,), generator=g))
        ids = torch.randint(0, E, (t, K), generator=g).to(dev)
        if trial % 7 == 0:
            ids[0, -1] = -1  # padding
        rem_e = _eager_reference(planner, ids, pools, ra, spi)
        rem_c = capturable(ids)
        torch.cuda.synchronize()
        assert torch.equal(rem_e, rem_c), f"remap diverged (trial {trial})"
        _assert_used_slots_equal(ra, rb, rem_c, f"trial {trial}")


@pytest.mark.parametrize("hot_layout", [False, True], ids=["static", "hotset"])
def test_captured_replay_equivalence(hot_layout):
    planner, pools, ra, rb, spi, capturable = _build_case(hot_layout)
    dev = torch.device("cuda:0")
    static_ids = torch.zeros((2, K), dtype=torch.int64, device=dev)
    out = {}
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            out["rem"] = capturable(static_ids)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out["rem"] = capturable(static_ids)

    for trial in range(20):
        g = torch.Generator().manual_seed(5000 + trial)
        ids = torch.randint(0, E, (2, K), generator=g).to(dev)
        static_ids.copy_(ids)
        graph.replay()
        torch.cuda.synchronize()
        rem_e = _eager_reference(planner, ids, pools, ra, spi)
        torch.cuda.synchronize()
        assert torch.equal(rem_e, out["rem"]), f"captured remap diverged ({trial})"
        _assert_used_slots_equal(ra, rb, rem_e, f"replay trial {trial}")
