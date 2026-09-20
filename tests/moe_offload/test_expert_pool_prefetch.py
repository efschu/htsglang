"""Speculative expert prefetch for the device-planned pool (19.09.).

The reference step program run with ``prefetch=True`` is the ORACLE the Triton
program must match, exactly as ``step_reference`` is for the real step. What is
pinned here is the part a GPU boot cannot show cheaply: that a prediction never
evicts the target layer's last real working set, that a prefetched row becomes
an ordinary hit at the next real step, that staging is never spent on a guess,
and that the switch being off changes nothing.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from sglang.srt.layers.moe import expert_pool_device as ep

# residents 0,1 in rows 0,1; LRU rows 2..7; staging rows 8..13
E, ROWS, R, S = 10, 14, 2, 6
HOST = [-1, -1, 0, 1, 2, 3, 4, 5, 6, 7]


def _pool(rows=ROWS, staging=S):
    t = ep.allocate_pool_tables("cpu", E, rows, R, staging, {0: 0, 1: 1}, HOST)
    b = ep.allocate_step_buffers("cpu", E, 16)
    pf = ep.allocate_step_buffers("cpu", E, 16)
    return t, b, pf


def _ids(*v):
    return torch.tensor(v, dtype=torch.int32)


# ---- (a) the prefetch semantics in the reference program --------------------
def test_a_prefetched_row_is_a_plain_hit_at_the_next_real_step():
    t, b, pf = _pool()
    pairs, _ = ep.step_reference(t, _ids(2, 3), pf, prefetch=True)
    assert pairs == [(0, 2), (1, 3)]  # host rows of experts 2,3 into free LRU rows
    assert ep.take_prefetch_report(t) == (2, 2, 0, 0)  # predicted, fetched, hits, skipped

    pairs, _ = ep.step_reference(t, _ids(2, 3), b)  # the real step needs exactly those
    assert pairs == []  # nothing copied synchronously any more
    assert b.routes[:2].tolist() == [2, 3]
    predicted, fetched, hits, skipped = ep.take_prefetch_report(t)
    assert (predicted, fetched, hits, skipped) == (0, 0, 2, 0)


def test_a_wrong_prediction_costs_one_lru_row_and_is_counted_as_wasted():
    t, b, pf = _pool()
    ep.step_reference(t, _ids(2, 3), pf, prefetch=True)
    ep.step_reference(t, _ids(2), b)  # only expert 2 was really needed
    predicted, fetched, hits, skipped = ep.take_prefetch_report(t)
    assert (predicted, fetched, hits) == (2, 2, 1)
    assert fetched - hits == 1  # expert 3's row: wasted, but still resident
    assert t.hot_phys.tolist()[3] == 3  # and free for a later step to hit


def test_the_last_real_working_set_is_never_a_prefetch_victim():
    t, b, pf = _pool()
    ep.step_reference(t, _ids(2, 3, 4, 5, 6, 7), b)  # fills all six LRU rows, clock 1
    before = t.row_key.tolist()
    pairs, _ = ep.step_reference(t, _ids(8, 9), pf, prefetch=True)
    assert pairs == []  # no victim: every LRU row carries this clock
    assert t.row_key.tolist() == before  # not one row taken from the real step
    assert ep.take_prefetch_report(t) == (2, 0, 0, 2)  # both predictions skipped
    # the real step, which DOES advance the clock, may evict them again
    pairs, _ = ep.step_reference(t, _ids(8, 9), b)
    assert len(pairs) == 2


def test_two_predicted_misses_in_one_prefetch_never_evict_each_other():
    t, b, pf = _pool(rows=10, staging=6)  # residents 0,1; LRU rows 2,3; staging 4..9
    pairs, _ = ep.step_reference(t, _ids(2, 3, 4), pf, prefetch=True)
    assert len(pairs) == 2 and pairs[0][1] != pairs[1][1]  # two rows, both distinct
    assert ep.take_prefetch_report(t) == (3, 2, 0, 1)  # the third found no victim
    assert sorted(t.row_key.tolist()[2:4]) == [2, 3]


def test_the_prefetch_never_spends_a_staging_row():
    t, b, pf = _pool(rows=10, staging=6)  # LRU rows 2,3; staging rows 4..9
    ep.step_reference(t, _ids(2, 3, 4, 5), pf, prefetch=True)
    assert t.row_key.tolist()[4:] == [-1] * 6  # staging untouched
    assert int(pf.staged_count[0]) == 0
    # the real step still stages what does not fit, exactly as before
    pairs, _ = ep.step_reference(t, _ids(2, 3, 6), b)
    assert int(b.staged_count[0]) == 1 and pairs[-1][1] == 4


def test_a_resident_prediction_is_counted_but_fetches_nothing():
    t, b, pf = _pool()
    pairs, _ = ep.step_reference(t, _ids(0, 1), pf, prefetch=True)  # both residents
    assert pairs == []
    assert ep.take_prefetch_report(t) == (2, 0, 0, 0)


def test_the_prefetch_leaves_clock_forwards_misses_and_routes_alone():
    t, b, pf = _pool()
    ep.step_reference(t, _ids(2), b)
    clock, forwards, misses = int(t.clock[0]), int(t.forwards[0]), int(t.misses_total[0])
    routes, step_map = pf.routes.tolist(), pf.step_map.tolist()
    ep.step_reference(t, _ids(3, 4), pf, prefetch=True)
    assert (int(t.clock[0]), int(t.forwards[0])) == (clock, forwards)
    assert int(t.misses_total[0]) == misses  # a prefetch is not a miss
    assert pf.routes.tolist() == routes and pf.step_map.tolist() == step_map


def test_the_prefetch_refuses_more_lanes_than_the_plan_allows():
    t, b, pf = _pool(rows=6, staging=2)
    # rows 6 = 2 residents + 2 LRU + 2 staging: the bound is lru + staging = 4
    # lanes (Task #40); three lanes are fine now, five are refused
    ep.step_reference(t, _ids(2, 3, 4), b)
    with pytest.raises(ValueError):
        ep.step_reference(t, _ids(2, 3, 4, 5, 6), b)  # real step: more lanes than the bound
    with pytest.raises(ValueError):
        ep.step_reference(t, _ids(2, 3, 4, 5, 6), pf, prefetch=True)  # and the prefetch too


def test_an_eager_pass_clears_the_prefetch_marks():
    t, b, pf = _pool()
    ep.step_reference(t, _ids(2, 3), pf, prefetch=True)
    ep.take_prefetch_report(t)
    ep.sync_tables(t, {2: 2, 3: 3})  # run_waves rewrote those rows behind our back
    assert t.pf_row.tolist()[R:] == [-1] * (ROWS - R)
    ep.step_reference(t, _ids(2, 3), b)
    assert ep.take_prefetch_report(t)[2] == 0  # no stale row counted as a hit


# ---- (b) the prediction helper: global top-k -> this rank's local ids --------
def _shard_stub(num_experts, lo, hi, generic):
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    class _Shard:
        _build_expert_shard_topk_remap = FusedMoE._build_expert_shard_topk_remap
        pool_prefetch_local_ids = FusedMoE.pool_prefetch_local_ids

        def __init__(self):
            self.num_experts = num_experts
            self._gguf_expert_shard = True
            self._gguf_expert_range = (lo, hi)
            self._expert_shard_generic = generic
            self._param = torch.nn.Parameter(torch.zeros(1))

        def parameters(self):
            return iter([self._param])

    s = _Shard()
    s._build_expert_shard_topk_remap()
    return s


@pytest.mark.parametrize("generic", [True, False])
def test_the_prediction_is_narrowed_to_this_ranks_own_experts(generic):
    lo, hi = 4, 9  # this rank owns global experts 4..8
    s = _shard_stub(16, lo, hi, generic)
    ids = torch.tensor([[3, 4, 8, 9], [0, 15, 5, 6]], dtype=torch.int32)
    local = s.pool_prefetch_local_ids(ids)
    n_local = hi - lo
    base = 1 if generic else 0  # generic shards keep local 0 as the zero pad
    assert local.tolist() == [
        [-1, base + 0, base + 4, -1],
        [-1, -1, base + 1, base + 2],
    ]
    kept = local[local >= 0]
    assert kept.numel() == 4
    # every surviving candidate addresses a row this rank actually holds, and
    # never the all-zero pad slot (which is resident and has no host row)
    assert int(kept.min()) >= base and int(kept.max()) <= base + n_local - 1
    pad = 0 if generic else n_local
    assert (kept != pad).all()  # the all-zero pad slot is never a prefetch target


def test_a_rank_that_owns_none_of_the_predicted_experts_prefetches_nothing():
    s = _shard_stub(16, 0, 4, True)
    local = s.pool_prefetch_local_ids(torch.tensor([[7, 8, 9]], dtype=torch.int32))
    assert local.tolist() == [[-1, -1, -1]]


# ---- (c) the switch is off by default ---------------------------------------
def test_the_switch_is_off_by_default_and_links_nothing():
    import sglang.srt.layers.moe.expert_offload as eo
    from sglang.srt.models.qwen2_moe import link_moe_pool_prefetch

    eo._POOL_PREFETCH = None
    os.environ.pop("SGLANG_MOE_POOL_PREFETCH", None)
    assert eo.pool_prefetch_enabled() is False

    class _Blk:
        pool_prefetch_next = None
        gate = object()
        experts = object()

    class _Layer:
        def __init__(self):
            self.mlp = _Blk()

    layers = [_Layer(), _Layer(), _Layer()]
    assert link_moe_pool_prefetch(layers) == 0
    assert all(l.mlp.pool_prefetch_next is None for l in layers)

    eo._POOL_PREFETCH = None
    os.environ["SGLANG_MOE_POOL_PREFETCH"] = "1"
    try:
        assert eo.pool_prefetch_enabled() is True
        assert link_moe_pool_prefetch(layers) == 2  # last block has no successor
        assert layers[0].mlp.pool_prefetch_next is layers[1].mlp
        assert layers[2].mlp.pool_prefetch_next is None
    finally:
        os.environ.pop("SGLANG_MOE_POOL_PREFETCH", None)
        eo._POOL_PREFETCH = None


def test_the_real_step_is_unchanged_when_no_prefetch_ever_ran():
    """Byte-for-byte the old behaviour: the pool tables carry the new marks,
    but with the switch off nothing writes them and nothing reads them."""
    t, b, _ = _pool()
    pairs, _ = ep.step_reference(t, _ids(2, 2, 3), b)
    assert pairs == [(0, 2), (1, 3)]
    assert b.routes[:3].tolist() == [2, 2, 3]
    assert t.pf_row.tolist() == [-1] * ROWS
    assert ep.take_prefetch_report(t) == (0, 0, 0, 0)


def test_the_side_stream_reads_the_layers_own_ids_buffer(monkeypatch):
    """fn6n (19.09.): the prefetch handed the caller's top-k temporary to the
    side stream; under graph capture that block is recycled on the capture
    stream, so the replay raced the step kernel's reads -> illegal address on
    all three ranks. The step must read the layer's OWN ids buffer, filled on
    the main stream before the fork, and only ever the lanes it was given."""
    import torch

    from sglang.srt.layers.moe import expert_offload as eo
    from sglang.srt.layers.moe import expert_pool_device as ep

    E, W = 12, 8
    pf = ep.allocate_step_buffers("cpu", E, W)
    seen = {}

    class _Ev:
        def record(self, _stream=None):
            seen["recorded"] = seen.get("recorded", 0) + 1

    class _Stream:
        def wait_event(self, _ev):
            seen["waited"] = True

    class _Ctx:
        def __init__(self, _s):
            pass

        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(eo, "pool_prefetch_stream", lambda: _Stream())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: _Stream())
    monkeypatch.setattr(torch.cuda, "stream", _Ctx)

    def fake_step(tables, ids, buffers, prefetch=False):
        seen["ids"] = ids
        seen["prefetch"] = prefetch
        seen["buffers"] = buffers

    monkeypatch.setattr(ep, "step", fake_step)
    monkeypatch.setattr(ep, "copy_rows", lambda *a, **k: seen.setdefault("copied", True))

    class _Cache:
        _pool_ready = True
        _pool_tables = object()
        _pool_pf_buffers = pf
        _pool_pf_begin = _Ev()
        _pool_pf_done = _Ev()
        _pool_pf_armed = False
        _pool_srcs = []
        _pool_dsts = []

    cache = _Cache()
    predicted = torch.tensor([[3, 7, 11, -1, 9, 2]], dtype=torch.int64)
    assert eo.MoEExpertOffloadCache.prefetch_pool(cache, predicted) is True
    ids = seen["ids"]
    assert seen["prefetch"] is True and seen["buffers"] is pf
    assert ids.data_ptr() == pf.ids.data_ptr() and ids.dtype == torch.int32
    assert ids.tolist() == [3, 7, 11, -1, 9, 2]
    assert pf.ids.tolist() == [3, 7, 11, -1, 9, 2, -1, -1]
    # the caller's temporary may be recycled after the fork -- the step's copy stays
    predicted.fill_(0)
    assert ids.tolist() == [3, 7, 11, -1, 9, 2]
    assert cache._pool_pf_armed is True and seen["waited"] and seen["copied"]
    # more lanes than the plan width: only the first W are taken
    wide = torch.arange(W + 5, dtype=torch.int32)
    eo.MoEExpertOffloadCache.prefetch_pool(cache, wide)
    assert seen["ids"].tolist() == list(range(W))


def test_the_kernel_takes_the_missing_expert_from_the_screened_lanes():
    """The Triton step reads the ids buffer exactly once: the per-miss expert
    comes from the screened ``safe`` lanes in registers, never from a second
    ``tl.load`` that a racing writer could feed an unscreened value."""
    import inspect

    from sglang.srt.layers.moe import expert_pool_device as ep

    src = inspect.getsource(ep._step_kernel)
    assert "expert = tl.load(ids_ptr + i)" not in src
    assert "expert = tl.sum(tl.where(lane == i, safe, 0), 0)" in src
    assert src.count("tl.load(ids_ptr") == 1
