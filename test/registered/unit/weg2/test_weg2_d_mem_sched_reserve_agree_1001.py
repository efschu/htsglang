"""D hang 01.10. (NF y6h, boot 10011531): at 16:07:53 the REPLICATED stage
machine split over the D group -- TP0 S6->S7 ("grow to hold 261597",
gate_reserve=4964), TP1/TP2 S6->S8 ("grow to hold 263225",
gate_reserve=6592) for the same census (used 171362, incoming 81151, air
4120). The gate reservation carries the scheduler's ``new_token_ratio``,
which is rank-local. Three seconds later every D rank sat in the
``drain_retired_prefetch`` all_reduce and D served nothing until the stop.

The fix agrees the reservation (MAX over the group) before the machine
steps on it, so every rank takes the same stage.
"""

import types

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GRID = [32768, 65536, 98304, 131072, 163840, 196608, 229376, 262144, 393216, 524288]
CLIP = 4096


def _req(rid, n_in, n_out=0, max_new=32768):
    return types.SimpleNamespace(
        rid=rid, origin_input_ids=[0] * n_in, output_ids=[0] * n_out, finished=lambda: False,
        sampling_params=types.SimpleNamespace(max_new_tokens=max_new))


class _Group:
    """A three-rank D group as the CPU collective sees it: MIN over what
    every rank contributes to the same call (the ranks' own reservations)."""

    def __init__(self):
        self.contrib = {}

    def gmin_for(self, rank):
        def gmin(vals):
            peers = [v for r, v in self.contrib.items() if r != rank]
            if len(vals) == 1 and peers:
                return [min([int(vals[0])] + peers)]
            return [int(v) for v in vals]
        return gmin


@pytest.fixture
def env(monkeypatch):
    from sglang.srt.weg2 import d_seat_vram as dsv

    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.setenv("SGLANG_OPT_WEG2_D_SEAT_VRAM", "1")
    monkeypatch.setenv("SGLANG_WEG2_D_KV_STAGE_TOKENS", ",".join(str(t) for t in GRID))
    monkeypatch.setenv("SGLANG_WEG2_D_KV_STAGE_BY_DEMAND", "1")
    monkeypatch.setenv("SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION", str(CLIP))
    monkeypatch.setattr(dsv, "_engage_kv_cap", lambda alloc, t, p: t)
    monkeypatch.setattr(dsv, "max_live_page", lambda alloc: 0)

    def make(ratio, gmin):
        # y6h D argv: chunk 4096, 6 seats, verify 4 -> air 4120
        sched = types.SimpleNamespace(
            server_args=types.SimpleNamespace(max_running_requests=6, chunked_prefill_size=4096,
                                              speculative_num_draft_tokens=4, page_size=1),
            running_batch=types.SimpleNamespace(reqs=[], batch_is_full=False), waiting_queue=[],
            chunked_req=None, last_batch=None, page_size=1, token_to_kv_pool_allocator=object(),
            new_token_ratio_tracker=types.SimpleNamespace(current=ratio),
            _weg2_group_min_ints=gmin)
        setattr(sched, dsv.CTL_ATTR, False)
        setattr(sched, dsv.PHASE_ATTR, dsv.PhaseState(epoch="e34", n=6, cap=6, done=True,
                                                      stage=6, stage_tokens=GRID[6]))
        # 16:07:53: two running (used 171362), weg2-34-246 queued (81151)
        sched.running_batch.reqs = [_req("weg2-34-244", 85681), _req("weg2-34-245", 85681)]
        sched.waiting_queue = [_req("weg2-34-246", 81151, max_new=1000)]
        return sched

    return dsv, make


def _rank_stages(dsv, make, ratios):
    group = _Group()
    scheds = {r: make(ratio, group.gmin_for(r)) for r, ratio in ratios.items()}
    for r, s in scheds.items():
        running, admissible = dsv._demand_lists(s)
        group.contrib[r] = -dsv.admission_reserve(s, running, admissible)
    for s in scheds.values():
        dsv.runtime_tick(s)
    return {r: getattr(s, dsv.PHASE_ATTR).stage for r, s in scheds.items()}


def test_the_census_is_the_y6h_one(env):
    dsv, make = env
    s = make(0.5, lambda vals: vals)
    used, incoming, _ = dsv.global_demand(s)
    assert (used, incoming) == (171362, 81151)
    assert used + incoming + 4120 == 256633          # S7 holds 262144, room 5511


def test_rank_local_ratios_still_take_one_stage(env):
    """TP0's ratio prices the queue at 2 x 2048 + 1001 = 5097 (fits S7), the
    workers' at 2 x 2458 + 1001 = 5917 (needs S8). Base: TP0 S7, TP1/TP2 S8
    -- the y6h split. Fixed: all three S8 (MAX, the refusing rank's price)."""
    dsv, make = env
    stages = _rank_stages(dsv, make, {0: 0.5, 1: 0.6, 2: 0.6})
    assert len(set(stages.values())) == 1, stages
    assert stages[0] == 8


def test_equal_ratios_are_unchanged(env):
    dsv, make = env
    assert _rank_stages(dsv, make, {0: 0.5, 1: 0.5, 2: 0.5}) == {0: 7, 1: 7, 2: 7}


def test_no_queue_no_collective(env):
    """A decode round with nothing queued pays no group call."""
    dsv, make = env
    calls = []
    s = make(0.5, lambda vals: calls.append(list(vals)) or vals)
    s.waiting_queue = []
    assert dsv._agree_reserve(s, [], 0) == 0
    dsv.runtime_tick(s)
    assert not any(len(c) == 1 and c[0] != 0 for c in calls)
