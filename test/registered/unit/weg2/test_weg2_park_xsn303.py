"""Punkt 2 (user order 18.09.): per-chunk admission + in-flight park on P.
Pure decision tests; the wiring ratchets check the scheduler/adder sites."""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import park as pk  # noqa: E402


def test_knobs_group_p_only_and_chunk_admit_rides_with_the_park():
    assert pk.park_active({"SGLANG_WEG2_GROUP": "P"})
    assert not pk.park_active({"SGLANG_WEG2_GROUP": "D"})
    assert not pk.park_active({})
    assert pk.chunk_admit_active({"SGLANG_WEG2_GROUP": "P"})
    # chunk admission without the park is refused: one knob covers both
    assert not pk.chunk_admit_active({"SGLANG_WEG2_GROUP": "P", pk.PARK_ENV: "0"})
    assert not pk.chunk_admit_active({"SGLANG_WEG2_GROUP": "P", pk.CHUNK_ADMIT_ENV: "0"})


def test_group_env_mirrors_corridor_guard():
    from sglang.srt.managers.corridor_guard import GROUP_ENV
    assert pk.GROUP_ENV == GROUP_ENV


def test_chunk_admit_tokens_charges_the_next_chunk_only():
    assert pk.chunk_admit_tokens(99572, 4096, anchor_gap=1) == 4097
    assert pk.chunk_admit_tokens(900, 4096) == 900          # shorter than a chunk
    assert pk.chunk_admit_tokens(99572, None) == 99572      # chunking off: whole
    assert pk.chunk_admit_tokens(99572, 0) == 99572


def test_the_youngest_parks_only_under_pressure_and_never_alone():
    running = [("old", 1.0), ("mid", 2.0), ("young", 3.0)]
    assert pk.park_verdict(running, need_tokens=4096, rem_total_tokens=100) == "young"
    assert pk.park_verdict(running, need_tokens=4096, rem_total_tokens=4096) is None
    assert pk.park_verdict([("only", 1.0)], need_tokens=4096, rem_total_tokens=0) is None
    # a protected (in-flight on a follower) youngest yields to the next youngest
    assert pk.park_verdict(running, 4096, 0, protected=("young",)) == "mid"
    assert pk.park_verdict(running, 4096, 0, protected=("young", "mid", "old")) is None


def test_resume_and_head_of_queue():
    assert pk.resume_ok(4096, 4096) and not pk.resume_ok(4097, 4096)
    a, b, p = object(), object(), object()
    assert pk.head_of_queue([a, p, b], p) == [p, a, b]
    assert pk.head_of_queue([a, b], p) == [p, a, b]


def test_park_req_is_a_frozen_ring_object():
    r = pk.Weg2ParkReq(rid="weg2-1-2", span=12000, epoch=3)
    assert r.reason == "kv-pressure"
    try:
        r.rid = "x"  # type: ignore[misc]
    except Exception:
        pass
    else:
        raise AssertionError("Weg2ParkReq must be frozen")


# ---- wiring (stage 2): adder gate, #679 flag, head-of-step park ---------------

def test_adder_gate_charges_the_next_chunk_on_group_p_only():
    from sglang.srt.managers import schedule_policy as sp
    src = open(sp.__file__).read()
    # #36415: the mamba gap reserve is read once, before the host load-back
    i = src.index("total_tokens += mamba_gap_reserve")
    blk = src[i:i + 1200]
    assert "_weg2_chunk_admit()" in blk and "weg2_parked_span" in blk
    assert "chunk_admit_tokens as _cat" in blk
    j = src.index("if grant <= 0:")
    assert "req.weg2_pool_parked = True" in src[j:j + 600]
    # the knobs are read once and default to OFF outside group P
    sp._WEG2_CHUNK_ADMIT = None; sp._WEG2_PARK_ON = None
    os.environ.pop("SGLANG_WEG2_GROUP", None)
    assert sp._weg2_chunk_admit() is False and sp._weg2_park_on() is False
    sp._WEG2_CHUNK_ADMIT = None; sp._WEG2_PARK_ON = None


def _park_stand_in(sch, *, waiting, inflight=0, parked=True, pool_idx=3):
    calls = []

    class _Req:
        rid = "weg2-1-7"
        req_pool_idx = pool_idx
        weg2_pool_parked = parked
        inflight_middle_chunks = inflight
        prefix_indices = list(range(40960))
        origin_input_ids = list(range(99000))

        def finished(self):
            return False

        def reset_for_retract(self):
            calls.append("reset")

    class _S:
        chunked_req = _Req()
        tree_cache = object()
        waiting_queue = list(waiting)

        def _add_request_to_queue(self, req, is_retracted=False):
            calls.append(("queue", is_retracted))

    s = _S()
    return s, calls


def test_head_of_step_park_gives_rows_back_and_requeues():
    from sglang.srt.managers import scheduler as sch
    os.environ["SGLANG_WEG2_GROUP"] = "P"
    seen = []
    orig = sch.release_kv_cache
    sch.release_kv_cache = lambda req, tree, is_insert=True: seen.append(is_insert)
    try:
        s, calls = _park_stand_in(sch, waiting=[object()])
        req = s.chunked_req
        sch.Scheduler.process_pending_weg2_park(s)
        assert seen == [True]                       # inserted, not discarded
        assert s.chunked_req is None
        assert calls == ["reset", ("queue", True)]
        assert req.weg2_parked_span == 40960 and req.weg2_pool_parked is False
        assert s._weg2_park_n == 1
    finally:
        sch.release_kv_cache = orig
        os.environ.pop("SGLANG_WEG2_GROUP", None)


def test_head_of_step_park_waits_for_inflight_chunks_and_needs_a_waiter():
    from sglang.srt.managers import scheduler as sch
    os.environ["SGLANG_WEG2_GROUP"] = "P"
    seen = []
    orig = sch.release_kv_cache
    sch.release_kv_cache = lambda req, tree, is_insert=True: seen.append(is_insert)
    try:
        s, calls = _park_stand_in(sch, waiting=[object()], inflight=1)
        sch.Scheduler.process_pending_weg2_park(s)
        assert seen == [] and s.chunked_req is not None      # a chunk still in flight
        s, calls = _park_stand_in(sch, waiting=[])
        sch.Scheduler.process_pending_weg2_park(s)
        assert seen == [] and s.chunked_req is not None      # nobody to make room for
        s, calls = _park_stand_in(sch, waiting=[object()], parked=False)
        sch.Scheduler.process_pending_weg2_park(s)
        assert seen == []                                    # not #679-parked
        os.environ["SGLANG_WEG2_GROUP"] = "D"
        s, calls = _park_stand_in(sch, waiting=[object()])
        sch.Scheduler.process_pending_weg2_park(s)
        assert seen == []                                    # group D never parks here
    finally:
        sch.release_kv_cache = orig
        os.environ.pop("SGLANG_WEG2_GROUP", None)


def test_the_park_runs_at_the_head_of_the_step():
    from sglang.srt.managers import scheduler as sch
    src = open(sch.__file__).read()
    i = src.index("self.process_pending_chunked_abort()\n")
    assert "self.process_pending_weg2_park()" in src[i:i + 200]
