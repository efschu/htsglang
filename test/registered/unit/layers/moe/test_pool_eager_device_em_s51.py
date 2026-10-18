"""S51 EAGER-DEVICE-EM (18.10., NF D rc13): an eager forward PAST the
D-Mini-Extend graph form runs a device-planned expert-major pass instead of
the host plan.

rc13 D TP0 (boot ..dauer10182002_6767e25d89_1018_200305.D.log): extends of
64-255 tokens cost 1.33 s median (PDFLIP-VRAM-PEAK chunk window, n=105) for
~5.3 GiB of expert misses, ~0.40 s at the 5090's Gen4 x8 link (14.4 GB/s card
probe). The rest is the host plan: per MoE layer one routing D2H, the Python
plan, the per-wave launch chain -- the GPU idles through it (y3u: a 2-6-token
extend cost 403-739 gpu-ms on the host plan, ~25 ms on the device step).

What must hold, black-box through ``run_eager_pool`` on a CPU cache:

* past the graph form the EM pass runs: no host plan, no host table sync, the
  step's forward counter moves once, one owner per expert;
* its output equals the host plan's bit for bit (same rows, same expert bytes,
  same weights, same combine), also over two waves, sliced, and with padded
  (-1) lanes, which stay zero;
* every miss crosses once (misses_total = the forward's distinct cold
  experts), warm LRU experts are hits;
* a lane served in wave 1 is never recomputed by wave 2 (the danger
  direction: wave 2 evicts wave-1 rows);
* the bounds: switch, device-step switch, host routing instrument, streaming
  partials, id cap, step width, wave cap keep the host plan;
* the metal check has its own budget;
* S51 PINNED-WAVE-IDX: the host loop stages its pair indices through
  ``_h2d_i64`` and computes the same output either way.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import logging
from collections import namedtuple
from types import SimpleNamespace

import pytest
import torch

from flliper.srt.environ import envs
from flliper.srt.layers.moe import expert_offload as eo
from flliper.srt.layers.moe import expert_pool_device as ep
from flliper.srt.layers.moe.topk import StandardTopKOutput
from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

# residents 0..3; scratch C=12 -> 9 LRU + 3 staging rows; E - R = 20 > 12:
# the EM bound is two waves
E, R, C, S, W, WIDTH = 24, 4, 12, 3, 3, 64

DispatchOutput = namedtuple("DispatchOutput", "hidden_states hidden_states_scale topk_output")
CombineOutput = namedtuple("CombineOutput", "hidden_states")

# 6 tokens at top-3 = 18 ids (past a graph form of 6): 11 distinct spill
# experts, fits one wave
ROUTES = [[5, 6, 0], [7, 5, 1], [9, 11, 2], [13, 17, 3], [6, 9, 21], [19, 22, 0]]
# 8 tokens at top-3: 16 distinct spill experts (> 12 rows) -> two waves
WIDE = [[4, 5, 0], [6, 7, 1], [8, 9, 2], [10, 11, 3], [12, 13, 0], [14, 15, 1],
        [16, 17, 2], [18, 19, 5]]


def _combine_ref(partials, out, rsf):
    """CPU stand-in for the combine kernel: the k-ordered sum times rsf."""
    acc = partials[:, 0].clone()
    for k in range(1, partials.shape[1]):
        acc = acc + partials[:, k]
    out.copy_(acc * (1.0 if rsf is None else rsf))
    return out


def _pool_cache(monkeypatch, warm=(), *, scratch=C, staging=S, width=WIDTH, graph_limit=6):
    monkeypatch.setenv("FLLIPER_MOE_SCRATCH_SLOTS", str(scratch))
    monkeypatch.setitem(eo._PARTIALS_MODE, "mode", "table")
    monkeypatch.setattr(eo, "combine_topk_partials", _combine_ref)
    layer = SimpleNamespace(
        num_local_experts=E, layer_id=7,
        moe_runner_config=SimpleNamespace(routed_scaling_factor=1.0),
    )
    cache = eo.MoEExpertOffloadCache(layer, R / E)
    assert (cache.resident_count, cache.scratch) == (R, scratch)
    spill = torch.zeros((E - R, W), dtype=torch.float32)
    for row in range(E - R):
        spill[row].fill_(R + row)  # a row's bytes ARE its expert id
    bank = torch.full((R + scratch, W), -1.0, dtype=torch.float32)
    for e in range(R):
        bank[e].fill_(e)
    cache._pinned = {"w13": spill}
    cache._resident = {"w13": bank}
    cache._installed = True
    hot_slot_of, host_row = cache._pool_layout()
    cache._pool_tables = ep.allocate_pool_tables("cpu", E, R + scratch, R, staging, hot_slot_of, host_row)
    cache._pool_buffers = ep.allocate_step_buffers("cpu", E, width)
    cache._pool_srcs = [spill]
    cache._pool_dsts = [bank]
    cache._pool_ready = True
    cache._pool_eager_device_checks = 0
    cache._pool_eager_device_em_checks = 0  # the check has its own test
    cache._pool_eager_device_limit = graph_limit  # the captured graph form
    for host, dst in ep.seed_lru_rows(cache._pool_tables, list(warm)):
        bank[dst].copy_(spill[host])
    return cache


def _weights(routes, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(len(routes), len(routes[0]), generator=g)


def _apply_over(bank, calls=None):
    """The MoE apply: a lane on a row reads that row's bytes times its weight;
    a pair on a row that does not hold its expert computes the wrong id."""

    def _apply(sub):
        if calls is not None:
            calls.append(int(sub.topk_output.topk_ids.shape[0]))
        rows = sub.topk_output.topk_ids.long()
        per_pair = bank[rows.clamp(min=0)][..., 0] * sub.topk_output.topk_weights
        per_pair = torch.where(rows >= 0, per_pair, torch.zeros_like(per_pair))
        out = per_pair.sum(dim=-1, keepdim=True).expand(-1, W).contiguous()
        return CombineOutput(hidden_states=out)

    return _apply


def _extend(cache, routes=ROUTES, seed=0, calls=None):
    ids = torch.tensor(routes, dtype=torch.int32)
    topk = StandardTopKOutput(topk_weights=_weights(routes, seed), topk_ids=ids, router_logits=None)
    out = cache.run_eager_pool(
        DispatchOutput(torch.zeros(ids.shape[0], W), None, topk),
        _apply_over(cache._resident["w13"], calls),
    )
    return out.hidden_states


def _truth(routes=ROUTES, seed=0):
    w = _weights(routes, seed)
    ids = torch.tensor(routes, dtype=torch.float32)
    contrib = torch.where(ids >= 0, ids * w, torch.zeros_like(w))
    acc = contrib[:, 0].clone()
    for k in range(1, contrib.shape[1]):
        acc = acc + contrib[:, k]
    return acc


def _forbid_host_plan(monkeypatch, cache):
    def _no(*_a, **_k):
        raise AssertionError("the host plan ran")

    monkeypatch.setattr(cache, "run_waves", _no)
    monkeypatch.setattr(cache, "sync_pool_from_host", _no)
    monkeypatch.setattr(cache, "begin_eager_pool", _no)
    monkeypatch.setattr(cache, "run_pool_waves", _no)  # nor the summed H95 waves


def _host_twin(monkeypatch, warm=()):
    with envs.FLLIPER_OPT_MOE_POOL_EAGER_DEVICE_EM.override(False):
        host = _pool_cache(monkeypatch, warm=warm)
    assert host.eager_device_em_waves(18)[1] == "em_switch_off"
    return host


def test_past_the_graph_form_the_em_pass_runs_not_the_host_plan(monkeypatch, caplog):
    cache = _pool_cache(monkeypatch)
    assert cache.eager_device_waves(18) == (0, "over_graph_form")
    assert cache.eager_device_em_waves(18) == (2, "device_em")
    _forbid_host_plan(monkeypatch, cache)
    t = cache._pool_tables
    before = int(t.forwards[0])
    with caplog.at_level(logging.INFO, logger=eo.__name__):
        got = _extend(cache)
    assert int(t.forwards[0]) == before + 1, "one pool forward, the later wave does not count"
    assert torch.equal(got[:, 0], _truth())
    assert ep.bijection_breaks(t) == 0
    assert int(t.error[0]) == 0
    bank = cache._resident["w13"]
    kept = 0
    for e in {e for row in ROUTES for e in row if e >= R}:
        r = int(t.hot_phys[e])
        if r < 0:
            continue  # served from a staging row for this step only
        kept += 1
        assert t.lru_start <= r < t.pool_rows, (e, r)
        assert float(bank[r, 0]) == float(e)  # the row holds the expert's bytes
    assert kept == t.pool_rows - t.lru_start, "the LRU rows took the misses first"
    lines = [r.getMessage() for r in caplog.records if "EAGER-DEVICE-EM layer 7" in r.getMessage()]
    assert lines and "T=6 ids=18 experts=24 waves=2" in lines[-1], lines


def test_the_em_output_is_bit_identical_to_the_host_plan(monkeypatch):
    dev = _pool_cache(monkeypatch, warm=[6, 13])
    on = _extend(dev, seed=3)
    host = _host_twin(monkeypatch, warm=[6, 13])
    off = _extend(host, seed=3)
    assert torch.equal(on, off)
    assert torch.equal(on[:, 0], _truth(seed=3))


def test_two_waves_serve_every_lane_once_and_each_miss_crosses_once(monkeypatch):
    dev = _pool_cache(monkeypatch, warm=[8, 9])
    t = dev._pool_tables
    t.misses_total.zero_()
    calls = []
    got = _extend(dev, WIDE, seed=5, calls=calls)
    spill = {e for row in WIDE for e in row if e >= R}
    assert len(spill) == 16 > C
    assert int(t.misses_total[0]) == len(spill - {8, 9}), "a miss crossed twice or not at all"
    assert len(calls) == 2, calls  # one apply per wave (one slice each)
    assert all(n == len(WIDE) * 3 for n in calls), "every wave runs over all lanes"
    assert torch.equal(got[:, 0], _truth(WIDE, seed=5))
    host = _host_twin(monkeypatch, warm=[8, 9])
    assert torch.equal(got, _extend(host, WIDE, seed=5))
    assert ep.bijection_breaks(t) == 0 and int(t.error[0]) == 0


def test_a_wave1_lane_is_never_recomputed_from_a_row_wave2_rewrote(monkeypatch):
    # every LRU + staging row is taken by wave 1, so wave 2 must evict wave-1
    # rows: a wave-1 lane recomputed in wave 2 would read another expert's
    # bytes (or its partial would be overwritten) and miss the truth
    dev = _pool_cache(monkeypatch)
    got = _extend(dev, WIDE, seed=11)
    assert torch.equal(got[:, 0], _truth(WIDE, seed=11))


def test_padded_lanes_stay_zero_and_slicing_changes_nothing(monkeypatch):
    routes = [[5, -1, 0], [7, 5, -1], [9, 11, 2], [-1, 17, 3], [6, 9, 21], [19, 22, 0]]
    whole = _pool_cache(monkeypatch, warm=[5])
    a = _extend(whole, routes, seed=2)
    assert torch.equal(a[:, 0], _truth(routes, seed=2))
    with envs.FLLIPER_OPT_MOE_POOL_EAGER_DEVICE_EM_SLICE.override(5):
        sliced = _pool_cache(monkeypatch, warm=[5])
        calls = []
        b = _extend(sliced, routes, seed=2, calls=calls)
    assert torch.equal(a, b)
    assert sorted(set(calls)) == [3, 5], calls  # 18 lanes in slices of 5 per wave


def test_the_bounds_keep_the_host_plan(monkeypatch):
    cache = _pool_cache(monkeypatch)
    assert cache.eager_device_em_waves(18) == (2, "device_em")
    cache._route_note = True
    assert cache.eager_device_em_waves(18) == (0, "host_instrument")
    cache._route_note = False
    assert cache.eager_device_em_waves(0) == (0, "empty")
    monkeypatch.setitem(eo._PARTIALS_MODE, "mode", "stream")
    assert cache.eager_device_em_waves(18) == (0, "em_host_partials")
    monkeypatch.setitem(eo._PARTIALS_MODE, "mode", "table")
    with envs.FLLIPER_OPT_MOE_POOL_EAGER_DEVICE_EM_MAX_IDS.override(17):
        assert cache.eager_device_em_waves(18) == (0, "em_max_ids")
    with envs.FLLIPER_OPT_MOE_POOL_EAGER_DEVICE_EM_MAX_IDS.override(0):
        assert cache.eager_device_em_waves(10**6) == (2, "device_em")
    with envs.FLLIPER_OPT_MOE_POOL_EAGER_DEVICE_EM_WAVES.override(1):
        assert cache.eager_device_em_waves(18) == (0, "em_waves_cap")
    narrow = _pool_cache(monkeypatch, width=16)  # E=24 lanes do not fit
    assert narrow.eager_device_em_waves(18) == (0, "em_width")
    with envs.FLLIPER_OPT_MOE_POOL_EAGER_DEVICE_STEP.override(False):
        off = _pool_cache(monkeypatch)
    assert off.eager_device_em_waves(18) == (0, "em_switch_off")
    calls = []
    real = off.run_waves
    monkeypatch.setattr(off, "run_waves", lambda *a, **k: calls.append(1) or real(*a, **k))
    assert torch.equal(_extend(off)[:, 0], _truth())
    assert calls, "with the device step off the host plans"


def test_the_em_metal_check_has_its_own_budget(monkeypatch, caplog):
    cache = _pool_cache(monkeypatch, warm=[6, 13])
    cache._pool_eager_device_checks = 0
    cache._pool_eager_device_em_checks = 1
    with caplog.at_level(logging.INFO, logger=eo.__name__):
        got = _extend(cache)
    assert cache._pool_eager_device_em_checks == 0
    assert torch.equal(got[:, 0], _truth())
    lines = [r.getMessage() for r in caplog.records if "EAGER-DEVICE-STEP CHECK" in r.getMessage()]
    assert len(lines) == 1 and "verdict=MATCH " in lines[0] and "waves=2" in lines[0], lines


def test_the_em_metal_check_stops_by_name_when_a_device_row_lies(monkeypatch):
    cache = _pool_cache(monkeypatch, warm=[6, 13])
    cache._pool_eager_device_em_checks = 1
    r = int(cache._pool_tables.hot_phys[6])
    cache._resident["w13"][r].fill_(99.0)
    with pytest.raises(RuntimeError, match="EAGER-DEVICE-STEP MISMATCH layer 7"):
        _extend(cache)


def test_small_forwards_keep_the_mini_extend_step(monkeypatch):
    cache = _pool_cache(monkeypatch, graph_limit=64)
    calls = []
    real = cache.run_pool_em
    monkeypatch.setattr(cache, "run_pool_em", lambda *a, **k: calls.append(1) or real(*a, **k))
    with envs.FLLIPER_OPT_MOE_POOL_OVERFLOW_WAVES.override(2):
        assert cache.eager_device_waves(18) == (2, "device")
        _extend(cache)
    assert not calls, "inside the graph form the D-Mini-Extend step stays"
    # past the decode wave cap the EM pass takes it (not the host plan)
    with envs.FLLIPER_OPT_MOE_POOL_OVERFLOW_WAVES.override(0):
        assert cache.eager_device_waves(18) == (0, "waves_cap")
        assert torch.equal(_extend(cache)[:, 0], _truth())
    assert calls == [1]


def test_pinned_wave_idx_stages_through_h2d_and_changes_nothing(monkeypatch):
    real = eo._h2d_i64
    out, calls = {}, {}
    for pinned in (True, False):
        seen = []
        monkeypatch.setattr(eo, "_h2d_i64", lambda arr, dev, _s=seen: _s.append(len(arr)) or real(arr, dev))
        applies = []
        with envs.FLLIPER_OPT_MOE_OFFLOAD_PINNED_WAVE_IDX.override(pinned):
            host = _host_twin(monkeypatch, warm=[8, 9])
            ids = torch.tensor(WIDE, dtype=torch.int32)
            topk = StandardTopKOutput(topk_weights=_weights(WIDE, 4), topk_ids=ids, router_logits=None)
            out[pinned] = host.run_eager_pool(
                DispatchOutput(torch.zeros(ids.shape[0], W), None, topk),
                _apply_over(host._resident["w13"], applies),
            ).hidden_states
        calls[pinned] = (len(seen), len(applies))
    assert torch.equal(out[True], out[False])
    assert torch.equal(out[True][:, 0], _truth(WIDE, seed=4))
    # exactly one pinned stage per applied wave slice, nothing else moved
    assert calls[True][1] == calls[False][1] >= 2
    assert calls[True][0] - calls[False][0] == calls[True][1], calls
