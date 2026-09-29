"""D-MEM-SCHED UNMERGED-EXTEND (y3j 09291933, needle weg2-40-151 MISS).

The tick runs at the top of ``get_next_batch_to_run``, BEFORE the extend batch
that ran last is merged into the running batch. Its census read the extend's
requests as gone: used=0, an END event, a pending shrink to S0 whose cap
(32768) sat below 87296 live tokens. The bs2 decode found no page below the
cap, the retraction emptied the batch and the needle was pressure-parked
behind an 11-minute agent turn until its client gave up."""

import logging
import types

import pytest

S0 = 32768
GRID = [S0 * (i + 1) for i in range(8)]  # 32768 .. 262144, the y3j #251c form below 262k


def _req(rid, n_in, n_out=0, finished=False):
    return types.SimpleNamespace(rid=rid, origin_input_ids=[0] * n_in, output_ids=[0] * n_out,
                                 finished=lambda: finished)


def _batch(reqs, extend=True):
    mode = types.SimpleNamespace(is_extend=lambda: extend)
    return types.SimpleNamespace(reqs=list(reqs), forward_mode=mode)


@pytest.fixture
def env(monkeypatch):
    from sglang.srt.weg2 import d_seat_vram as dsv

    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.setenv("SGLANG_OPT_WEG2_D_SEAT_VRAM", "1")
    monkeypatch.setenv("SGLANG_WEG2_D_KV_STAGE_TOKENS", ",".join(str(t) for t in GRID))
    monkeypatch.setenv("SGLANG_WEG2_D_KV_STAGE_BY_DEMAND", "1")
    caps, floor = [], {"page": 0}
    monkeypatch.setattr(dsv, "_engage_kv_cap", lambda alloc, t, p: caps.append(int(t)) or t)
    monkeypatch.setattr(dsv, "max_live_page", lambda alloc: floor["page"])
    sched = types.SimpleNamespace(
        server_args=types.SimpleNamespace(max_running_requests=6, chunked_prefill_size=4096,
                                          speculative_num_draft_tokens=4),
        running_batch=types.SimpleNamespace(reqs=[]), waiting_queue=[], chunked_req=None,
        last_batch=None, page_size=64, token_to_kv_pool_allocator=object(),
        _weg2_group_min_ints=lambda vals: vals)
    setattr(sched, dsv.CTL_ATTR, False)
    setattr(sched, dsv.PHASE_ATTR, dsv.PhaseState(epoch="e1", n=6, cap=6, done=True, stage=2,
                                                  stage_tokens=GRID[2]))
    return dsv, sched, caps, floor


def test_the_census_counts_the_extend_the_merge_has_not_taken_yet(env):
    dsv, sched, _caps, _floor = env
    a, b = _req("weg2-27-76", 23168), _req("weg2-40-151", 64072)
    sched.last_batch = _batch([a, b])
    used, incoming, rids = dsv.global_demand(sched)
    assert used == 23168 + 64072 and incoming == 0
    assert rids == frozenset({"weg2-27-76", "weg2-40-151"})


def test_the_census_counts_each_request_once_and_skips_decode_and_finished(env):
    dsv, sched, _caps, _floor = env
    a, b, done = _req("a", 1000), _req("b", 2000), _req("c", 5000, finished=True)
    sched.running_batch.reqs = [a]
    sched.last_batch = _batch([a, b, done])            # a already merged (aliased slot)
    assert dsv.global_demand(sched)[0] == 3000
    sched.last_batch = _batch([a, b], extend=False)     # a decode round: nothing unmerged
    assert dsv.global_demand(sched)[0] == 1000
    sched.last_batch = None
    assert dsv.global_demand(sched)[0] == 1000


def test_y3j_20_02_20_no_end_event_no_cap_below_the_live_pages(env):
    dsv, sched, caps, floor = env
    a, b = _req("weg2-27-76", 23168), _req("weg2-40-151", 64072)
    sched.running_batch.reqs = [a]                      # the round before the wake's extend
    dsv.runtime_tick(sched)
    ms = getattr(sched, dsv.MEM_SCHED_ATTR)
    # the extend ran with both; the tick of the next pass precedes the merge
    sched.running_batch.reqs = []
    sched.last_batch = _batch([a, b])
    floor["page"] = 87296 // 64
    dsv.runtime_tick(sched)
    assert ms.pending is None                           # base: pending S0
    assert ms.counters["stage_down_on_end"] == 0
    assert all(c >= 87296 for c in caps), caps          # base: cap 32768 below 87296 live
    assert getattr(sched, dsv.PHASE_ATTR).stage == 2


def test_the_marker_names_the_unmerged_extend(env, caplog):
    dsv, sched, _caps, _floor = env
    sched.last_batch = _batch([_req("weg2-40-151", 64072)])
    with caplog.at_level(logging.INFO):
        dsv.global_demand(sched)
    assert any("WEG2 D-MEM-SCHED UNMERGED-EXTEND" in r.getMessage() and "64072" in r.getMessage()
               for r in caplog.records)
