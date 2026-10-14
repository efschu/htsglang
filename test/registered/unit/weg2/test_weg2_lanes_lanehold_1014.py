"""PRIORITY LANES: the lane hold never takes the floor lane itself (NF metal boot f4afb10f28, 14.10.2026 09:01Z, FIXES-NACH-FREEZE Nr. 48).

THE HANG.  weg2-139-485 (omp, ``priority`` 1, 23 929 tokens) hung from 09:01:26Z.
  front 09:01:18,944 ``LANE-ARRIVE rid=weg2-139-485 lane=1 floor=0->1 epoch=3`` (the request raised the floor itself); 09:01:22,297
  ``W50-REROUTE rid=weg2-139-485 ... reason=x_refusal_midstream path=midstream`` (D kept the stream, P prefills its context);
  P 09:01:18 ``PR LANE-FLOOR floor=1 epoch=3`` on PP0/1/2; P.log Z.360848 (09:01:51) ``H91 WEG2-INTAKE-ADMISSIBLE rid=weg2-139-485 ...
  decline=loop_skips(weg2_lane_held=1(first=weg2-139-485))``: P skipped the request of the very lane that holds the floor.
ROOT.  The P-only leg 1 of a RESUME-VIA-P rid is built by ``Front._rvp_take`` from D's needs-P record: ``{"rid", "input_ids",
  "sampling_params"}`` -- no ``priority``.  ``Pending.lane`` is right (``_lane_for(rid)``), the PAYLOAD that ``leg1`` POSTs to P's
  ``/generate`` is not: the scheduler's ``Req.priority`` is None, ``lanes_p.req_lane`` reads lane 0, and ``lanes_p.holds(0, floor=1)`` holds
  it.  The hold rule itself (``lane < floor``) is right; what was wrong is the lane the request arrived at P with.
THE FIX.  (1) ``_rvp_take`` writes the lane into the payload it builds (both branches, text and vision) -- ``Front._lane_payload``;
  (2) ``leg1`` verifies, for EVERY P leg, that the payload carries the lane the front booked for the Pending, writes it when it does not and
  says so (``WEG2 LANE-LEG-PRIORITY``): a floor-lane leg that would reach P as lane 0 is named, never silently held.  Lane 0 stays absent
  (no ``priority`` key): lane-0 operation is byte-identical.  No timer, no switch.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.weg2 import front as front_mod  # noqa: E402
from sglang.srt.weg2 import lanes as LN  # noqa: E402
from sglang.srt.weg2 import lanes_p as LP  # noqa: E402
from sglang.srt.weg2 import resume_via_p as rvp  # noqa: E402
from sglang.srt.weg2.front import Front  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_weg2_lanes_l4_1008 import _front, _on, _switch  # noqa: E402


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_WEG2_LANES", "0")  # the helpers of the L4 file name the switch themselves
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.delenv(rvp.ENV, raising=False)
    monkeypatch.setenv(rvp.ENV_OPEN_STREAM, "0")


def _sched_req(payload):
    """What P's scheduler makes of the posted JSON: a ``Req`` whose ``priority`` is the payload's field (``GenerateReqInput.priority``)."""
    return types.SimpleNamespace(priority=payload.get(LN.FIELD))


def _p_holds(payload, floor):
    """P's admission loop (scheduler.py ``_lane_floor and _lanes_p.holds(req_lane(req), floor)``) on the request the payload becomes."""
    return LP.holds(LP.req_lane(_sched_req(payload)), floor)


def _rvp_front(rid, lane, tag=""):
    f = _front("P")
    f.tag = tag
    f._rvp_state()
    f._lane_state().note(rid, lane)
    f._lane_state().set_floor(lane)
    return f


# ======================================================================== the repro: W50-REROUTE midstream of the floor lane
@_on
async def test_w50_reroute_midstream_leg_of_the_floor_lane_is_not_held_by_p():
    """The boot's case: floor 1, a lane-1 stream D refused midstream -> ``_rvp_take`` queues the P-only leg; P's lane hold must NOT take it.
    Red on f4afb10f28: the payload has no ``priority``, P reads lane 0 < floor 1 (``loop_skips(weg2_lane_held=1)``)."""
    rid = "weg2-139-485"
    f = _rvp_front(rid, 1)
    rvp.write_request(rid, list(range(23929)), 23929, 12288, "x_refusal_midstream")
    assert f._rvp_take() == 1
    p = f.queue[-1]
    assert p.resume_via_p and p.p_only and p.lane == 1
    assert p.payload.get("priority") == 1, "the P-only leg must reach P as the lane the front booked"
    assert not _p_holds(p.payload, floor=1), "the floor lane is never held (P.log Z.360848: decline=loop_skips(weg2_lane_held=1))"


@_on
async def test_the_lane_resume_reprefill_leg_carries_its_lane_too():
    """Same site, other cause: a lane resume that found no prefix is a RESUME-VIA-P leg (``LANE-REPREFILL``); its lane is the lane it was held at."""
    rid = "weg2-60-548"
    f = _rvp_front(rid, 2)
    rvp.write_request(rid, [1, 2, 3], 3, 12288, "x_refusal_midstream")
    assert f._rvp_take() == 1
    assert f.queue[-1].payload["priority"] == 2
    assert not _p_holds(f.queue[-1].payload, floor=2)
    assert not _p_holds(f.queue[-1].payload, floor=1)


# ======================================================================== the hold keeps its meaning: lanes BELOW the floor are held
@_on
async def test_a_lower_lane_leg_is_still_held_by_p():
    """Counter-test: lane 1 RVP leg under floor 2 stays held (``lane < floor``) -- the hold rule is not weakened."""
    rid = "weg2-7-70"
    f = _rvp_front(rid, 1)
    f._lane_state().set_floor(2)
    rvp.write_request(rid, [1, 2, 3], 3, 12288, "x_refusal_midstream")
    assert f._rvp_take() == 1
    assert f.queue[-1].payload["priority"] == 1
    assert _p_holds(f.queue[-1].payload, floor=2)


@_on
async def test_lane_0_requests_stay_held_and_carry_no_priority_key():
    """Counter-test: a lane-0 request is held under floor 1 as before, and its P-only leg has NO ``priority`` key (absent stays absent:
    lane-0 operation without lanes action is byte-identical)."""
    rid = "weg2-8-80"
    f = _rvp_front(rid, 0)
    f._lane_state().set_floor(1)
    rvp.write_request(rid, [1, 2, 3], 3, 12288, "x_refusal_midstream")
    assert f._rvp_take() == 1
    assert "priority" not in f.queue[-1].payload
    assert _p_holds(f.queue[-1].payload, floor=1), "lane 0 under floor 1 is held, by design"
    assert not _p_holds(f.queue[-1].payload, floor=0)


def test_switch_off_payload_of_the_rvp_leg_is_unchanged():
    """SGLANG_WEG2_LANES=0: the payload ``_rvp_take`` builds has exactly the keys it had (no ``priority``)."""
    rid = "weg2-9-90"
    with _switch(False):
        f = _front("P")
        f.tag = ""
        f._rvp_state()
        rvp.write_request(rid, [1, 2, 3], 3, 12288, "x_refusal_midstream")

        async def body():
            assert f._rvp_take() == 1

        asyncio.run(body())
    assert set(f.queue[-1].payload) == {"rid", "input_ids", "sampling_params"}


@_on
async def test_vision_branch_carries_the_lane():
    """The vision leg is the ORIGINAL request body (``_rvp_bodies``); it carries the normalised lane too (the stored body may predate the
    normalisation or lack the field)."""
    rid = "weg2-10-100"
    f = _rvp_front(rid, 3)
    f._rvp_bodies[rid] = ("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}], "stream": True})
    rvp.write_request(rid, [1, 2, 3], 3, 12288, rvp.REASON_VISION)
    assert f._rvp_take() == 1
    q = f.queue[-1]
    assert q.path == "/v1/chat/completions" and q.payload["priority"] == 3
    assert f._rvp_bodies[rid][1].get("priority") is None, "the stored original is not mutated"


# ======================================================================== the net in leg1: every P leg reaches P as its lane
class _Resp:
    def __init__(self, status, body):
        self.status, self._body = status, body

    async def read(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _leg1_front(posted):
    f = _front("P")
    f.p_leg1_stall_s = 0.0
    body = json.dumps({"usage": {"prompt_tokens": 50, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 0}}}).encode()

    def post(url, json=None):
        posted.append(json)
        return _Resp(200, body)

    f.session = types.SimpleNamespace(post=post)
    return f


def _pending(f, rid, lane, payload, **kw):
    p = front_mod.Pending(rid=rid, path="/generate", payload=payload, text="x", t_arrive=time.time(),
                          fut=asyncio.get_event_loop().create_future(), est_prompt=50, est_uncached=50, lane=lane, **kw)
    f._lane_state().note(rid, lane)
    return p


@_on
async def test_leg1_net_writes_the_booked_lane_into_a_payload_that_lost_it(caplog):
    """Net: any other synthetic P leg (or a future one) that reaches ``leg1`` with ``Pending.lane`` > 0 and a payload without it is corrected
    and named -- ``WEG2 LANE-LEG-PRIORITY`` -- before the POST."""
    posted = []
    f = _leg1_front(posted)
    p = _pending(f, "weg2-11-110", 1, {"rid": "weg2-11-110", "input_ids": [1, 2], "sampling_params": {"max_new_tokens": 1}})
    with caplog.at_level("WARNING"):
        await f.leg1(p)
    assert posted[-1]["priority"] == 1 and not _p_holds(posted[-1], floor=1)
    assert p.payload.get("priority") is None, "leg1 works on a copy"
    marks = [r.getMessage() for r in caplog.records if r.getMessage().startswith(LN.MARK_LEG_PRIORITY)]
    assert len(marks) == 1 and "rid=weg2-11-110" in marks[0] and "lane=1" in marks[0]


@_on
async def test_leg1_net_is_silent_when_the_payload_already_carries_the_lane(caplog):
    posted = []
    f = _leg1_front(posted)
    p = _pending(f, "weg2-12-120", 2, {"rid": "weg2-12-120", "input_ids": [1], "priority": 2, "sampling_params": {}})
    with caplog.at_level("WARNING"):
        await f.leg1(p)
    assert posted[-1]["priority"] == 2
    assert not [r for r in caplog.records if r.getMessage().startswith(LN.MARK_LEG_PRIORITY)]


@_on
async def test_leg1_net_leaves_lane_0_alone(caplog):
    posted = []
    f = _leg1_front(posted)
    p = _pending(f, "weg2-13-130", 0, {"rid": "weg2-13-130", "input_ids": [1], "sampling_params": {}})
    with caplog.at_level("WARNING"):
        await f.leg1(p)
    assert "priority" not in posted[-1]
    assert not [r for r in caplog.records if r.getMessage().startswith(LN.MARK_LEG_PRIORITY)]


@_on
async def test_leg1_net_only_raises_never_lowers_a_payload_lane(caplog):
    """The net writes the booked lane where the payload lacks it; it never rewrites a payload that already names a lane (a Pending whose
    booked lane is 0 keeps the payload's own priority)."""
    posted = []
    f = _leg1_front(posted)
    p = _pending(f, "weg2-15-150", 0, {"rid": "weg2-15-150", "input_ids": [1], "priority": 2, "sampling_params": {}})
    with caplog.at_level("WARNING"):
        await f.leg1(p)
    assert posted[-1]["priority"] == 2
    assert not [r for r in caplog.records if r.getMessage().startswith(LN.MARK_LEG_PRIORITY)]


def test_lane_payload_helper_switch_off_writes_nothing_even_for_a_named_lane():
    """SGLANG_WEG2_LANES=0: ``_lane_payload`` returns before it looks, whatever lane it is handed."""
    with _switch(False):
        f = _front("P")
        pl = {"rid": "r"}
        assert Front._lane_payload(f, "r", pl, lane=2) == {"rid": "r"}
    with _switch(True):
        f = _front("P")
        pl = {"rid": "r"}
        assert Front._lane_payload(f, "r", pl, lane=2) == {"rid": "r", "priority": 2}
        assert Front._lane_payload(f, "r", pl, lane=0) == {"rid": "r", "priority": 2}, "lane 0 never lowers or adds"
        assert Front._lane_payload(f, "r", "not-a-dict", lane=2) == "not-a-dict"


def test_leg1_net_switch_off_posts_the_payload_as_it_is():
    posted = []
    with _switch(False):
        async def body():
            f = _leg1_front(posted)
            p = _pending(f, "weg2-14-140", 1, {"rid": "weg2-14-140", "input_ids": [1], "sampling_params": {}})
            await f.leg1(p)

        asyncio.run(body())
    assert "priority" not in posted[-1]


# ======================================================================== P's own rule
def test_p_hold_rule_is_lane_below_floor_only():
    """The P-side rule the boot hit, pinned: ``holds(lane, floor)`` is ``lane < floor`` and nothing else; the floor lane and above pass."""
    assert not LP.holds(1, 1) and not LP.holds(2, 1) and not LP.holds(5, 5)
    assert LP.holds(0, 1) and LP.holds(1, 2)
    assert not LP.holds(0, 0)
    assert LP.req_lane(types.SimpleNamespace(priority=None)) == 0, "a request that reached P without the field is lane 0 -- hence the payload fix"
