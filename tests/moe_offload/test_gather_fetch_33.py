"""Task #33 (19.09.): the eager expert fetch gathers per tensor (UVA) instead of
one memcpy per expert; the desk (no CUDA stream) keeps the memcpy path."""
import os
import types

import pytest
import torch

from sglang.srt.layers.moe import expert_offload as eo


def test_fetch_mode_default_gather_and_memcpy_opt_out(monkeypatch):
    monkeypatch.delenv("SGLANG_MOE_OFFLOAD_FETCH", raising=False)
    assert eo._fetch_mode() == "gather"
    monkeypatch.setenv("SGLANG_MOE_OFFLOAD_FETCH", "memcpy")
    assert eo._fetch_mode() == "memcpy"
    monkeypatch.setenv("SGLANG_MOE_OFFLOAD_FETCH", "nonsense")
    assert eo._fetch_mode() == "gather"


def _cache_stub(R=2, C=2, E=6, rows=4):
    # resident buf [R+C, rows], pinned spill [E-R, rows]; static layout id-R
    st = types.SimpleNamespace()
    st._resident = {"w": torch.zeros((R + C, rows), dtype=torch.int32)}
    st._pinned = {"w": torch.arange((E - R) * rows, dtype=torch.int32).view(E - R, rows)}
    st._scratch_holds = {}
    st.resident_count = R
    st._spill_pool_index = None
    st._cold_tier = None
    st._remote_ids = set()
    st._stream = None
    st.planner = types.SimpleNamespace(stats=types.SimpleNamespace(h2d_bytes=0, remote_h2d_bytes=0))
    return st


def test_desk_fetch_without_a_stream_takes_the_memcpy_path_and_is_correct():
    st = _cache_stub()
    plan = [(4, 2), (5, 3)]          # experts 4,5 (spill rows 2,3) -> scratch slots 2,3
    eo.MoEExpertOffloadCache._fetch(st, plan)
    assert torch.equal(st._resident["w"][2], st._pinned["w"][2])
    assert torch.equal(st._resident["w"][3], st._pinned["w"][3])
    assert st._scratch_holds == {2: 4, 3: 5}
    assert st.planner.stats.h2d_bytes == 2 * 4 * 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="UVA gather needs a card")
def test_gpu_gather_fetch_matches_memcpy(monkeypatch):
    R, C, E, rows = 2, 3, 8, 1024
    pinned = torch.arange((E - R) * rows, dtype=torch.int32).view(E - R, rows).pin_memory()
    plan = [(3, 2), (6, 3), (7, 4)]
    out = {}
    for mode in ("memcpy", "gather"):
        monkeypatch.setenv("SGLANG_MOE_OFFLOAD_FETCH", mode)
        st = _cache_stub(R, C, E, rows)
        st._resident = {"w": torch.zeros((R + C, rows), dtype=torch.int32, device="cuda")}
        st._pinned = {"w": pinned}
        st._stream = torch.cuda.Stream()
        eo.MoEExpertOffloadCache._fetch(st, plan)
        torch.cuda.synchronize()
        out[mode] = st._resident["w"].cpu().clone()
    assert torch.equal(out["memcpy"], out["gather"])
    assert torch.equal(out["gather"][2], pinned[1]) and torch.equal(out["gather"][4], pinned[5])
