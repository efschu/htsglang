"""D-Mini-Extend (30.09., y3u): an eager forward under the device-planned pool
whose routed ids fit the widest CAPTURED decode step runs that step's device
plan (``prepare_pool`` / ``run_pool_waves``) instead of the host plan.

y3u D TP0 ('Prefill rank batch' 00:36:43Z / 00:40:39Z ff.): extends of 2-6 new
tokens cost 403-739 gpu-ms, while a graphed verify round of 4 rows costs 30-40
ms. The host plan pays, per MoE layer, one D2H of the routing, the host plan,
and ``sync_pool_from_host`` (sticky error, report, prefetch report, table
read-back and write) -- every one a device sync that serializes the layer's
CPU launch behind its H2D misses. The device step has the same misses and no
host read.

What must hold, black-box through ``run_eager_pool`` on a CPU cache:

* a small eager forward runs the device step: no host plan, no host sync of
  the tables, the step's forward counter moves, one owner per expert;
* its output is bit-identical to the host plan's, every lane from its
  expert's bytes; warm LRU experts are hits, only the rest are misses;
* the limit is the graph form (captured max step ids, never above the step
  buffers' width); past it, past the wave cap, with a host-side routing
  instrument or with the switch off the host plans as before;
* the metal check (SGLANG_DEBUG_MOE_POOL_EAGER_DEVICE_CHECK) compares against
  the plain host plan (fresh fetch) and stops by name on a deviation.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import logging
from collections import namedtuple
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import expert_offload as eo
from sglang.srt.layers.moe import expert_pool_device as ep
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

# residents 0..3 in rows 0..3; scratch C=12 -> 9 LRU + 3 staging rows
E, R, C, S, W, WIDTH = 24, 4, 12, 3, 3, 64

DispatchOutput = namedtuple("DispatchOutput", "hidden_states hidden_states_scale topk_output")
CombineOutput = namedtuple("CombineOutput", "hidden_states")

# a 4-token extend at top-3: 12 ids, spill {5,6,7,9,11,13,17} + residents 0..3
ROUTES = [[5, 6, 0], [7, 5, 1], [9, 11, 2], [13, 17, 3]]
SPILL = sorted({e for row in ROUTES for e in row if e >= R})


def _pool_cache(monkeypatch, warm=(), *, scratch=C, staging=S):
    """A CPU pool cache whose bank rows carry their expert id as bytes; the
    experts in ``warm`` sit in LRU rows as a decode round left them."""
    monkeypatch.setenv("SGLANG_MOE_SCRATCH_SLOTS", str(scratch))
    monkeypatch.setitem(eo._PARTIALS_MODE, "mode", "stream")  # no combine kernel on CPU
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
    cache._pool_buffers = ep.allocate_step_buffers("cpu", E, WIDTH)
    cache._pool_srcs = [spill]
    cache._pool_dsts = [bank]
    cache._pool_ready = True
    cache._pool_eager_device_checks = 0  # the check has its own tests
    for host, dst in ep.seed_lru_rows(cache._pool_tables, list(warm)):
        bank[dst].copy_(spill[host])
    return cache


def _weights(routes, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(len(routes), len(routes[0]), generator=g)


def _apply_over(bank):
    """The MoE apply: a lane routed to a row reads that row's bytes; a lane
    on a row that does not hold its expert computes the wrong expert id."""

    def _apply(sub):
        rows = sub.topk_output.topk_ids.long()
        per_pair = bank[rows.clamp(min=0)][..., 0] * sub.topk_output.topk_weights
        out = per_pair.sum(dim=-1, keepdim=True).expand(-1, W).contiguous()
        return CombineOutput(hidden_states=out)

    return _apply


def _extend(cache, routes=ROUTES, seed=0):
    ids = torch.tensor(routes, dtype=torch.int32)
    topk = StandardTopKOutput(topk_weights=_weights(routes, seed), topk_ids=ids, router_logits=None)
    out = cache.run_eager_pool(
        DispatchOutput(torch.zeros(ids.shape[0], W), None, topk),
        _apply_over(cache._resident["w13"]),
    )
    return out.hidden_states


def _truth(routes=ROUTES, seed=0):
    w = _weights(routes, seed)
    ids = torch.tensor(routes, dtype=torch.float32)
    return (ids * w).sum(dim=-1)


def _forbid_host_plan(monkeypatch, cache):
    def _no(*_a, **_k):
        raise AssertionError("the host plan ran")

    monkeypatch.setattr(cache, "run_waves", _no)
    monkeypatch.setattr(cache, "sync_pool_from_host", _no)
    monkeypatch.setattr(cache, "begin_eager_pool", _no)


def test_a_small_eager_forward_runs_the_device_step_not_the_host_plan(monkeypatch):
    cache = _pool_cache(monkeypatch)
    _forbid_host_plan(monkeypatch, cache)
    t = cache._pool_tables
    before = int(t.forwards[0])
    got = _extend(cache)
    assert int(t.forwards[0]) == before + 1, "the device step did not run"
    assert torch.allclose(got[:, 0], _truth(), rtol=0, atol=1e-6)
    assert ep.bijection_breaks(t) == 0
    bank = cache._resident["w13"]
    for e in SPILL:  # every miss now lives in one LRU row holding its bytes
        r = int(t.hot_phys[e])
        assert t.lru_start <= r < t.pool_rows, (e, r)
        assert float(bank[r, 0]) == float(e)


def test_the_output_is_bit_identical_to_the_host_plan(monkeypatch):
    routes = [[5, 6, 0], [7, 5, 1], [9, 11, 2], [13, 17, 3], [6, 9, 21]]
    dev = _pool_cache(monkeypatch, warm=[6, 13])
    on = _extend(dev, routes, seed=3)
    with envs.SGLANG_OPT_MOE_POOL_EAGER_DEVICE_STEP.override(False):
        host = _pool_cache(monkeypatch, warm=[6, 13])
    assert host._pool_eager_device_step is False
    off = _extend(host, routes, seed=3)
    assert torch.equal(on, off)


def test_warm_lru_experts_are_hits_and_only_the_rest_are_misses(monkeypatch):
    cache = _pool_cache(monkeypatch, warm=[5, 9])
    t = cache._pool_tables
    row_of = {e: int(t.hot_phys[e]) for e in (5, 9)}
    cache._pool_tables.misses_total.zero_()
    _extend(cache)
    assert int(t.misses_total[0]) == len(set(SPILL) - {5, 9})
    for e, r in row_of.items():  # a hit keeps its row and is stamped as used
        assert int(t.hot_phys[e]) == r
        assert int(t.row_use[r]) == int(t.clock[0])


def test_the_limit_is_the_graph_form(monkeypatch):
    cache = _pool_cache(monkeypatch)
    monkeypatch.setattr(cache, "_pool_max_step_ids", lambda: 40)
    assert cache.eager_device_limit() == 40  # captured bs1 x 4 rows x top-10
    cache._pool_eager_device_limit = None
    monkeypatch.setattr(cache, "_pool_max_step_ids", lambda: 1000)
    assert cache.eager_device_limit() == WIDTH  # never above the step buffers
    cache._pool_eager_device_limit = None
    monkeypatch.setattr(cache, "_pool_max_step_ids", lambda: None)
    assert cache.eager_device_limit() == WIDTH


def test_past_the_graph_form_the_host_plans(monkeypatch):
    cache = _pool_cache(monkeypatch)
    cache._pool_eager_device_limit = 6
    assert cache.eager_device_waves(12) == (0, "over_graph_form")
    calls = []
    real = cache.run_waves
    monkeypatch.setattr(cache, "run_waves", lambda *a, **k: calls.append(1) or real(*a, **k))
    got = _extend(cache)
    assert calls, "the host plan must run past the graph form"
    assert torch.allclose(got[:, 0], _truth(), rtol=0, atol=1e-6)


def test_the_wave_cap_bounds_the_device_step(monkeypatch):
    # C=6 (3 LRU + 3 staging): 12 ids -> min(12, 20) = 12 needs 2 waves
    cache = _pool_cache(monkeypatch, scratch=6, staging=3)
    with envs.SGLANG_OPT_MOE_POOL_OVERFLOW_WAVES.override(0):
        assert cache.eager_device_waves(12) == (0, "waves_cap")
    with envs.SGLANG_OPT_MOE_POOL_OVERFLOW_WAVES.override(2):
        assert cache.eager_device_waves(12) == (2, "device")
        _forbid_host_plan(monkeypatch, cache)
        got = _extend(cache)
    assert torch.allclose(got[:, 0], _truth(), rtol=0, atol=1e-6)
    assert ep.bijection_breaks(cache._pool_tables) == 0
    assert int(cache._pool_tables.error[0]) == 0


def test_a_host_routing_instrument_or_the_switch_keep_the_host_plan(monkeypatch):
    cache = _pool_cache(monkeypatch)
    assert cache.eager_device_waves(12) == (1, "device")
    cache._route_note = True
    assert cache.eager_device_waves(12) == (0, "host_instrument")
    cache._route_note = False
    assert cache.eager_device_waves(0) == (0, "empty")
    with envs.SGLANG_OPT_MOE_POOL_EAGER_DEVICE_STEP.override(False):
        off = _pool_cache(monkeypatch)
    assert off.eager_device_waves(12) == (0, "switch_off")


def test_the_metal_check_matches_the_plain_host_plan(monkeypatch, caplog):
    with envs.SGLANG_DEBUG_MOE_POOL_EAGER_DEVICE_CHECK.override(1):
        cache = _pool_cache(monkeypatch, warm=[6, 13])
        cache._pool_eager_device_checks = int(envs.SGLANG_DEBUG_MOE_POOL_EAGER_DEVICE_CHECK.get())
    with caplog.at_level(logging.INFO, logger=eo.__name__):
        got = _extend(cache)
    assert cache._pool_eager_device_checks == 0
    assert torch.allclose(got[:, 0], _truth(), rtol=0, atol=1e-6)
    lines = [r.getMessage() for r in caplog.records if "EAGER-DEVICE-STEP CHECK" in r.getMessage()]
    assert len(lines) == 1 and "verdict=MATCH " in lines[0], lines
    assert ep.bijection_breaks(cache._pool_tables) == 0
    # the budget is spent: the next forward runs without the reference
    _extend(cache)
    lines = [r.getMessage() for r in caplog.records if "EAGER-DEVICE-STEP CHECK" in r.getMessage()]
    assert len(lines) == 1


def test_the_metal_check_stops_by_name_when_a_device_row_lies(monkeypatch):
    cache = _pool_cache(monkeypatch, warm=[6, 13])
    cache._pool_eager_device_checks = 1
    # the tables say row r holds expert 6, its bytes say otherwise
    r = int(cache._pool_tables.hot_phys[6])
    cache._resident["w13"][r].fill_(99.0)
    with pytest.raises(RuntimeError, match="EAGER-DEVICE-STEP MISMATCH layer 7"):
        _extend(cache)


def test_compare_moe_outputs_verdicts():
    a = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    assert eo.compare_moe_outputs(a, a.clone())[0] == "MATCH"
    b = a.clone()
    b[0, 0] += 1e-3  # fp rounding of another summation order
    assert eo.compare_moe_outputs(b, a)[0] == "MATCH~fp"
    c = a.clone()
    c[1, 1] = 0.0
    assert eo.compare_moe_outputs(c, a)[0] == "MISMATCH"
    d = a.clone()
    d[0, 1] = float("nan")
    assert eo.compare_moe_outputs(d, a)[0] == "MISMATCH"
    assert eo.compare_moe_outputs(a[:1], a)[0] == "MISMATCH"
