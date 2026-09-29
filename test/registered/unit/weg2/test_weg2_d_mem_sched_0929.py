"""D-MEM-SCHED (29.09.): the #251c/d KV stage BETWEEN wakes -- the replicated
machine, the runtime tick on the scheduler and the coldest-first row switch
(27B's conditions, user orders 10:20-10:50Z; gap (a) of the coverage table)."""

import types

import pytest
import torch

from sglang.srt.layers.moe import expert_pool_device as epd
from sglang.srt.weg2 import d_mem_sched as ms

S0 = 262144
STEP = 16384
AIR = ms.air_tokens(4096, 6, 4)  # one chunk + one round of six seats
#: the launcher's #251c/d ladder of the cut form (EXPERTEN-KV-DYNAMISCH-D §9)
LADDER = (262144, 393216, 524288)


def _sched(top=S0 + 16 * STEP, k=ms.HYSTERESIS_ROUNDS):
    # a fine grid for the state-machine cases (the machine takes any ladder)
    return ms.MemSched(stage_tokens=tuple(range(S0 // 4, top + 1, STEP)), air_tokens=AIR,
                       hysteresis_rounds=k)


# --- KV side -----------------------------------------------------------------


def test_grow_takes_the_smallest_stage_holding_the_need():
    s = _sched()
    st = s.step(100_000, 20_000)
    assert st.changed and s.stage_tokens[st.stage] >= 100_000 + 20_000 + AIR
    assert st.stage == 0 or s.stage_tokens[st.stage - 1] < 100_000 + 20_000 + AIR
    assert s.counters["stage_up"] == 1


def test_above_the_top_only_the_admission_waits_never_an_error():
    s = _sched(top=S0)
    st = s.step(S0 - 1000, 50_000)
    assert st.admit_wait and st.stage == len(s.stage_tokens) - 1
    assert s.counters["admit_wait_stage"] == 1


def test_an_end_event_shrinks_at_once_no_hysteresis():
    s = _sched()
    s.step(300_000)
    hi = s.stage
    st = s.step(120_000, ended=True)
    assert st.changed and s.stage < hi
    assert s.counters["stage_down_on_end"] == 1


def test_unbacked_pages_of_the_ended_request_hold_until_the_backup_ack():
    s = _sched()
    s.step(300_000)
    hi = s.stage
    st = s.step(120_000, ended=True, unbacked_tokens=180_000)
    assert s.stage == hi and not st.changed
    assert s.counters["stage_down_waited_backup"] == 1
    st = s.step(120_000, ended=True, unbacked_tokens=0)
    assert st.changed and s.stage < hi


def test_bs_swinging_3_4_at_a_boundary_does_not_flap():
    s = _sched()
    edge = s.stage_tokens[6]
    s.step(edge - AIR + 10)  # grows to stage 7
    j = s.stage
    for i in range(200):
        s.step(edge - AIR + (10 if i % 2 else -3000))  # 3 <-> 4 seats worth
    assert s.stage == j
    assert s.counters["stage_flap"] == 0 and s.counters["stage_down"] == 0


def test_no_event_below_for_k_rounds_shrinks_with_a_stage_gap():
    s = _sched(k=8)
    s.step(400_000)
    hi = s.stage
    for _ in range(7):
        assert not s.step(100_000).changed
    st = s.step(100_000)
    assert st.changed and s.stage < hi
    assert s.stage_tokens[s.stage] >= 100_000 + AIR + STEP


def test_six_sessions_finishing_one_by_one_shrink_each_time():
    s = _sched()
    per = 60_000
    s.step(6 * per)
    last = s.stage
    for left in range(5, 0, -1):
        st = s.step(left * per, ended=True)
        assert st.changed and s.stage < last
        last = s.stage
    assert s.counters["stage_down_on_end"] == 5 and s.counters["stage_flap"] == 0


def test_the_stage_is_the_same_on_every_rank_from_global_inputs():
    # uneven DCP 1,2,2: the ranks hold different shares, the scheduler's token
    # count is global -- three replicas fed the same inputs agree every step
    ranks = [_sched() for _ in range(3)]
    for used, inc, end in [(10_000, 50_000, False), (200_000, 0, False),
                           (320_000, 8_000, False), (90_000, 0, True)]:
        assert len({r.step(used, inc, ended=end).stage for r in ranks}) == 1


def test_two_sessions_of_200k_take_the_ladder_above_262k():
    s = ms.MemSched(stage_tokens=LADDER, air_tokens=AIR)
    st = s.step(200_000, 200_000)
    assert st.changed and not st.admit_wait and s.stage_tokens[st.stage] == 524288


def test_above_the_ladder_the_admission_waits():
    s = ms.MemSched(stage_tokens=LADDER, air_tokens=AIR)
    st = s.step(400_000, 200_000)
    assert st.admit_wait and s.stage == 2 and s.counters["admit_wait_stage"] == 1


# --- coldest first -------------------------------------------------------------


def _tables(keys, uses, *, lru_start, seat_base, seat_rows, seat_on, clock, E=16):
    hot = torch.full((E,), -1, dtype=torch.int32)
    for r, e in enumerate(keys):
        if 0 <= e < E:
            hot[e] = r
    return types.SimpleNamespace(
        num_experts=E, lru_start=lru_start, seat_base=seat_base, seat_rows=seat_rows,
        seat_on=seat_on, row_key=torch.tensor(keys, dtype=torch.int32),
        row_use=torch.tensor(uses, dtype=torch.int64), hot_phys=hot,
        pf_row=torch.full((len(keys),), -1, dtype=torch.int64),
        clock=torch.tensor([clock], dtype=torch.int64))


def test_the_hot_tail_expert_moves_into_the_coldest_kept_row():
    # rows 0-1 resident, 2-5 LRU (row 3 free), seat rows 6-7 ON; 7 is hot
    keys = [0, 1, 2, -1, 4, 5, 6, 7]
    uses = [0, 0, 5, 0, 1, 9, 2, 50]
    moves = epd.coldest_first_moves(keys, uses, lru_start=2, keep_hi=6, drop_lo=6, drop_hi=8,
                                    num_experts=16, clock=60)
    assert moves[0] == (7, 3)          # hottest source -> the free row first
    assert moves[1] == (6, 4)          # next -> the coldest held row (use 1 < 2)
    assert len(moves) == 2


def test_rows_used_in_the_running_step_are_never_a_destination():
    keys = [2, 3, 7]
    uses = [60, 61, 50]
    assert epd.coldest_first_moves(keys, uses, lru_start=0, keep_hi=2, drop_lo=2, drop_hi=3,
                                   num_experts=16, clock=60) == []


def test_shrink_keeps_the_hot_expert_on_the_card_and_drops_the_cold_one():
    keys = [0, 1, 2, -1, 4, 5, 6, 7]
    uses = [0, 0, 5, 0, 1, 9, 2, 50]
    t = _tables(keys, uses, lru_start=2, seat_base=6, seat_rows=2, seat_on=2, clock=60)
    seen = []
    epd.set_seat_rows_on(t, 0, device_write=True, move_rows=seen.extend)
    assert seen == [(7, 3), (6, 4)]
    assert int(t.hot_phys[7]) == 3 and int(t.hot_phys[6]) == 4
    assert int(t.hot_phys[4]) == -1            # the cold one went to the store
    assert t.row_key[6:8].tolist() == [epd.SEAT_OFF_KEY] * 2
    assert int(t.row_use[3]) == 50


def test_without_a_mover_the_shrink_is_the_old_tail_drop():
    keys = [0, 1, 2, -1, 4, 5, 6, 7]
    uses = [0, 0, 5, 0, 1, 9, 2, 50]
    t = _tables(keys, uses, lru_start=2, seat_base=6, seat_rows=2, seat_on=2, clock=60)
    epd.set_seat_rows_on(t, 0, device_write=True)
    assert int(t.hot_phys[7]) == -1 and int(t.hot_phys[4]) == 4


def test_the_bank_move_copies_every_row_buffer_the_fetch_writes():
    from sglang.srt.layers.moe.expert_offload import MoEExpertOffloadCache

    cache = object.__new__(MoEExpertOffloadCache)
    w13 = torch.arange(8 * 3, dtype=torch.float32).view(8, 3)
    scale = torch.arange(8, dtype=torch.float32).view(8, 1)
    cache._resident = {"w13_weight": w13.clone(), "w13_weight_scale": scale.clone()}
    cache._move_bank_rows([(7, 3), (6, 4)])
    assert torch.equal(cache._resident["w13_weight"][3], w13[7])
    assert torch.equal(cache._resident["w13_weight"][4], w13[6])
    assert torch.equal(cache._resident["w13_weight_scale"][3], scale[7])
    assert torch.equal(cache._resident["w13_weight"][0], w13[0])      # others untouched
    with pytest.raises(RuntimeError, match="coldest-first move needs row 9"):
        cache._move_bank_rows([(9, 1)])


def test_the_emergency_stop_exists_and_is_off_by_default():
    from sglang.srt.environ import envs

    assert envs.SGLANG_WEG2_DISABLE_D_ELASTIC_ROWS.get() is False


# --- live floor: a stage never ends below a held page ---------------------------


def test_a_live_page_above_the_target_holds_the_shrink_pending():
    s = _sched()
    s.step(300_000)
    hi = s.stage
    want = s._smallest_holding(120_000 + AIR)
    floor = s.stage_tokens[want + 3] - 10
    st = s.step(120_000, ended=True, floor_tokens=floor)
    assert s.stage == want + 3 < hi and st.changed        # down as far as the floor allows
    assert s.pending == want and s.counters["stage_down_waited_backup"] == 1
    st = s.step(120_000, floor_tokens=floor)              # still held: no second count
    assert not st.changed and s.counters["stage_down_waited_backup"] == 1
    st = s.step(120_000, floor_tokens=0)                  # drained: no new hysteresis
    assert st.changed and s.stage == want and s.pending is None
    assert s.counters["stage_down_on_end"] == 1 and s.counters["stage_flap"] == 0


def test_growth_cancels_a_pending_shrink():
    s = _sched()
    s.step(300_000)
    s.step(120_000, ended=True, floor_tokens=290_000)
    assert s.pending is not None
    s.step(350_000)
    assert s.pending is None and s.stage_tokens[s.stage] >= 350_000 + AIR


def test_the_floor_is_asked_only_when_a_shrink_is_possible():
    s = _sched()
    s.step(300_000)
    assert not s.shrink_candidate(300_000)
    assert s.shrink_candidate(100_000)


# --- the runtime tick on a scheduler ---------------------------------------------


def _req(rid, n_in, n_out=0):
    return types.SimpleNamespace(rid=rid, origin_input_ids=[0] * n_in, output_ids=[0] * n_out)


@pytest.fixture
def tick_env(monkeypatch):
    from sglang.srt.weg2 import d_seat_vram as dsv

    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.setenv("SGLANG_OPT_WEG2_D_SEAT_VRAM", "1")
    grid = [S0 + i * STEP for i in range(8)]
    monkeypatch.setenv("SGLANG_WEG2_D_KV_STAGE_TOKENS", ",".join(str(t) for t in grid))
    monkeypatch.setenv("SGLANG_WEG2_D_KV_STAGE_BY_DEMAND", "1")
    caps, floor = [], {"page": 0}
    monkeypatch.setattr(dsv, "_engage_kv_cap", lambda alloc, t, p: caps.append(int(t)) or t)
    monkeypatch.setattr(dsv, "max_live_page", lambda alloc: floor["page"])
    votes = []

    def gmin(vals):
        votes.append(list(vals))
        return vals

    sched = types.SimpleNamespace(
        server_args=types.SimpleNamespace(max_running_requests=6, chunked_prefill_size=4096,
                                          speculative_num_draft_tokens=4),
        running_batch=types.SimpleNamespace(reqs=[]), waiting_queue=[], chunked_req=None,
        page_size=64, token_to_kv_pool_allocator=object(), _weg2_group_min_ints=gmin)
    setattr(sched, dsv.CTL_ATTR, False)  # no pages on this rank (a 3080 worker)
    setattr(sched, dsv.PHASE_ATTR, dsv.PhaseState(epoch="e1", n=6, cap=6, done=True, stage=0,
                                                  stage_tokens=S0))
    return dsv, sched, caps, floor, votes, grid


def test_tick_grows_the_stage_for_the_queue_head(tick_env):
    dsv, sched, caps, _floor, votes, grid = tick_env
    sched.running_batch.reqs = [_req("a", 200_000)]
    sched.waiting_queue = [_req("b", 90_000)]
    st = dsv.runtime_tick(sched)
    assert st.changed and st.stage > 0
    phase = getattr(sched, dsv.PHASE_ATTR)
    assert phase.stage == st.stage and phase.stage_tokens == grid[st.stage]
    assert caps == [grid[st.stage]] and votes == []    # growth asks no collective


def test_tick_shrinks_in_the_next_round_after_a_finish(tick_env):
    dsv, sched, caps, floor, votes, grid = tick_env
    sched.running_batch.reqs = [_req("a", 200_000), _req("b", 150_000)]
    dsv.runtime_tick(sched)
    hi = getattr(sched, dsv.PHASE_ATTR).stage
    sched.running_batch.reqs = [_req("a", 200_000)]    # b finished
    st = dsv.runtime_tick(sched)
    assert st.changed and getattr(sched, dsv.PHASE_ATTR).stage < hi
    assert len(votes) == 1                              # one floor vote, only here
    ms = getattr(sched, dsv.MEM_SCHED_ATTR)
    assert ms.counters["stage_down_on_end"] == 1


def test_tick_waits_for_the_held_pages_and_caps_below(tick_env):
    dsv, sched, caps, floor, votes, grid = tick_env
    sched.running_batch.reqs = [_req("a", 200_000), _req("b", 150_000)]
    dsv.runtime_tick(sched)
    hi = getattr(sched, dsv.PHASE_ATTR).stage
    floor["page"] = grid[hi] // 64                      # b's pages still live (unbacked)
    sched.running_batch.reqs = [_req("a", 200_000)]
    st = dsv.runtime_tick(sched)
    ms = getattr(sched, dsv.MEM_SCHED_ATTR)
    assert not st.changed and ms.pending is not None
    want = ms.pending
    assert caps[-1] == grid[want]                       # new pages go below the wanted end
    floor["page"] = 0                                   # backed and evicted
    st = dsv.runtime_tick(sched)
    assert st.changed and ms.pending is None
    assert getattr(sched, dsv.PHASE_ATTR).stage == want


def test_tick_off_by_the_stop_and_asleep(tick_env, monkeypatch):
    dsv, sched, *_ = tick_env
    sched.running_batch.reqs = [_req("a", 200_000)]
    monkeypatch.setenv("SGLANG_WEG2_DISABLE_D_ELASTIC_ROWS", "1")
    assert dsv.runtime_tick(sched) is None
    monkeypatch.delenv("SGLANG_WEG2_DISABLE_D_ELASTIC_ROWS")
    sched.weg2_dormant = True
    assert dsv.runtime_tick(sched) is None


def test_a_new_wake_epoch_restarts_from_the_wake_stage(tick_env):
    dsv, sched, *_ = tick_env
    sched.running_batch.reqs = [_req("a", 300_000)]
    dsv.runtime_tick(sched)
    phase = getattr(sched, dsv.PHASE_ATTR)
    phase.epoch, phase.stage = "e2", 0
    sched.running_batch.reqs = []
    dsv.runtime_tick(sched)
    ms = getattr(sched, dsv.MEM_SCHED_ATTR)
    assert ms._epoch == "e2" and ms.counters["stage_up"] >= 1   # counters carry over


def test_the_scheduler_runs_the_tick_after_the_ack_flush():
    import inspect

    from sglang.srt.managers import scheduler

    src = inspect.getsource(scheduler.Scheduler.get_next_batch_to_run)
    assert src.index("flush_write_through_acks()") < src.index("_weg2_d_seat_vram.runtime_tick(self)")
