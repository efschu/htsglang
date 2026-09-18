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
