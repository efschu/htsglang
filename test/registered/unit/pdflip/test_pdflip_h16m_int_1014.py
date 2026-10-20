"""h16m integration 1014: #968 ROW-ROOM (75e3ee8abd) + h18 port + LBV-ADMIT (e23abfcb05) + PRIO-AS-AGE (fa99a484ae).

What the merge itself decides, and the review items of done/nf-prio-alter-review-1014.md taken into it:

* WAKE-READ-FLOOR x PRIO-AS-AGE (port risk 4 of done/nf-h18port-1014.md): the wake read's floor reads ``priority`` as a
  lane.  User decision 14.10. 18:27Z: priority is a SORT KEY, no lane floor / hold / preempt.  The one place that
  decides whether a floor may act is ``prio_age.floor_mode()``; ``park_l3.below_floor`` follows it.
* N2: the leg1 hotfix a4856f6e5d (every P leg carries the rid's lane as ``priority``) -- in the age mode P's batch
  order hangs on exactly this field.
* N3: counted, not changed -- the X gate's abstention for a p>0 rid outside the W9 canon names the priority.
* N4: PDFLIP MODE-PARITY -- front, P and D state their mode; the front compares and stops by name on a mismatch.
* N5: on group P a stage sorts only on a single stage or where the followers execute PP0's row.
* N1 (MIN-DWELL on the PRIO-URGENT flip) is NOT in this branch: the user decides after a metal measurement.
"""
from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import os
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flliper.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from flliper.srt.pdflip import d_lane, park_l3  # noqa: E402
from flliper.srt.pdflip import lanes as LN  # noqa: E402
from flliper.srt.pdflip import mode_parity as MP  # noqa: E402
from flliper.srt.pdflip import p_row_authority as PROW  # noqa: E402
from flliper.srt.pdflip import prio_age as PA  # noqa: E402
from flliper.srt.pdflip.front import Front, Pending  # noqa: E402
from test_pdflip_lanes_l4_1008 import _front, _note  # noqa: E402
from test_pdflip_wake_read_seat_gate_1014 import _req as _wreq, _sched as _wsched  # noqa: E402

NOW = 1_000_000.0


@contextlib.contextmanager
def _mode(prio_as_age: bool, lanes: bool = True):
    keys = ("FLLIPER_PDFLIP_LANES", PA.ENV)
    old = {k: os.environ.get(k) for k in keys}
    os.environ["FLLIPER_PDFLIP_LANES"] = "1" if lanes else "0"
    os.environ[PA.ENV] = "1" if prio_as_age else "0"
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(autouse=True)
def _production_mode(monkeypatch):
    """Production default: lanes on, priority as age on."""
    monkeypatch.delenv(park_l3.FLOOR_ENV, raising=False)
    with _mode(True):
        yield


# ============================================================ WAKE-READ-FLOOR x PRIO-AS-AGE
def _hold(n=5):
    out = []
    for i in range(n):
        r = _wreq("pdflip-2-%d" % (30 + i), 100)
        r.priority = 0
        out.append(r)
    return out


def test_age_mode_a_wake_read_never_waits_on_a_floor(monkeypatch):
    """Age mode: even a floor attribute left at 1 (the RPC that could raise it is refused there) holds no read back --
    the floor decision belongs to prio_age.floor_mode alone."""
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    h = _wsched()
    setattr(h, d_lane.FLOOR_ATTR, 1)
    hold = _hold()
    assert not PA.floor_mode()
    assert not any(park_l3.below_floor(h, r) for r in hold)
    park_l3.issue_deferred_reads(h, hold)
    assert h.issued, "the age mode has no floor: the seat/page gates alone decide"


def test_floor_mode_keeps_the_ported_wake_read_floor(monkeypatch):
    """Floor mode (FLLIPER_PDFLIP_PRIO_AS_AGE=0): the h17 port acts exactly as before."""
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    with _mode(False):
        assert PA.floor_mode()
        h = _wsched()
        setattr(h, d_lane.FLOOR_ATTR, 1)
        hold = _hold()
        assert all(park_l3.below_floor(h, r) for r in hold)
        park_l3.issue_deferred_reads(h, hold)
        assert h.issued == []


def test_age_mode_three_ranks_issue_the_same_reads_even_with_split_floor_attrs(monkeypatch):
    """Rank agreement: under the age mode the floor attribute is not an input at all -- three ranks with different
    (stale) floor attributes issue the same reads."""
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    ranks = []
    for f in (1, 0, 0):
        s = _wsched()
        setattr(s, d_lane.FLOOR_ATTR, f)
        park_l3.issue_deferred_reads(s, _hold())
        ranks.append(tuple(s.issued))
    assert len(set(ranks)) == 1 and ranks[0]


def test_park_l3_reads_the_one_floor_decider():
    src = inspect.getsource(park_l3.below_floor)
    assert "floor_mode()" in src


# ============================================================ N2: the leg1 hotfix a4856f6e5d
class _Resp:
    status = 200

    async def read(self):
        return json.dumps({"meta_info": {"prompt_tokens": 10, "cached_tokens": 0}}).encode()


class _Cm:
    async def __aenter__(self):
        return _Resp()

    async def __aexit__(self, *a):
        return False


def _posted_leg1(f, p):
    posted = []

    def post(url, json=None, **kw):  # noqa: A002 - aiohttp's keyword
        posted.append(json)
        return _Cm()

    f.session = types.SimpleNamespace(post=post)

    async def body():
        try:
            await f.leg1(p)
        except Exception:  # noqa: BLE001 -- the rest of leg 1 (instruments) is not under test
            pass

    asyncio.run(body())
    assert posted, "leg1 posted nothing"
    return posted[0]


def _pend(rid, lane, payload):
    loop = asyncio.new_event_loop()
    try:
        return Pending(rid, "/generate", payload, "x", NOW, loop.create_future(), est_prompt=5000,
                       est_uncached=5000, lane=lane)
    finally:
        loop.close()


def test_n2_rvp_text_leg_without_priority_reaches_p_with_the_lane_of_the_pending():
    """The RESUME-VIA-P text leg {rid, input_ids, sampling_params} carries no priority; leg1 writes the Pending's lane."""
    f = _front("P")
    p = _pend("pdflip-145-486", 1, {"rid": "pdflip-145-486", "input_ids": [1, 2, 3], "sampling_params": {}})
    body = _posted_leg1(f, p)
    assert body[LN.FIELD] == 1
    # what P's scheduler makes of it: Req.priority = the field -> the age order puts it first
    assert PA.lane_of_req(types.SimpleNamespace(priority=body.get(LN.FIELD))) == 1


def test_n2_lane_falls_back_to_the_front_booking_of_the_rid():
    """Pending.lane 0 but the front booked the rid at lane 3 (``_lane_for``): the leg carries 3."""
    f = _front("P")
    _note(f, "pdflip-7-70", 3)
    p = _pend("pdflip-7-70", 0, {"rid": "pdflip-7-70", "input_ids": [1], "sampling_params": {}})
    assert _posted_leg1(f, p)[LN.FIELD] == 3


def test_n2_lane_0_stays_absent_and_a_carried_priority_is_not_overwritten():
    f = _front("P")
    p0 = _pend("pdflip-8-80", 0, {"rid": "pdflip-8-80", "input_ids": [1], "sampling_params": {}})
    assert LN.FIELD not in _posted_leg1(f, p0), "lane 0: byte-identical body, no priority key"
    p5 = _pend("pdflip-9-90", 2, {"rid": "pdflip-9-90", "input_ids": [1], "sampling_params": {}, LN.FIELD: 5})
    assert _posted_leg1(f, p5)[LN.FIELD] == 5, "the request's own priority wins"


def test_n2_p_batch_order_of_two_rvp_legs_follows_the_carried_priority(monkeypatch):
    """End to end on the P side: the bodies leg1 posts become Reqs; the P stage (single stage) orders p1 before an
    older p0."""
    from flliper.srt.managers.scheduler import Scheduler

    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    f = _front("P")
    old = _posted_leg1(f, _pend("pdflip-1-1", 0, {"rid": "pdflip-1-1", "input_ids": [1], "sampling_params": {}}))
    new = _posted_leg1(f, _pend("pdflip-2-2", 1, {"rid": "pdflip-2-2", "input_ids": [1], "sampling_params": {}}))
    q = [types.SimpleNamespace(rid=b["rid"], priority=b.get(LN.FIELD)) for b in (old, new)]
    s = types.SimpleNamespace(waiting_queue=q, enable_priority_scheduling=False,
                              ps=types.SimpleNamespace(pp_size=1, pp_rank=0))
    Scheduler._pdflip_prio_age_order(s)
    assert [r.rid for r in s.waiting_queue] == ["pdflip-2-2", "pdflip-1-1"]


# ============================================================ N5: P sorts only on a single stage or the row form
def _p_stage(pp_size, row_only=False, counters=None):
    q = [types.SimpleNamespace(rid="a", priority=0), types.SimpleNamespace(rid="b", priority=1)]
    s = types.SimpleNamespace(waiting_queue=q, enable_priority_scheduling=False,
                              ps=types.SimpleNamespace(pp_size=pp_size, pp_rank=0), pp_flip_counters=counters)
    if row_only:
        setattr(s, PROW.ROW_ONLY_ATTR, True)
    return s


def _order(s):
    from flliper.srt.managers.scheduler import Scheduler

    Scheduler._pdflip_prio_age_order(s)
    return [r.rid for r in s.waiting_queue]


def test_n5_carrierless_p_stage_does_not_sort(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    assert _order(_p_stage(3)) == ["a", "b"], "carrierless: each stage plans itself -- P's order is the front's pick"


def test_n5_row_form_and_single_stage_sort(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    monkeypatch.delenv("FLLIPER_PP_ROW_AUTHORITY", raising=False)
    assert _order(_p_stage(3, row_only=True)) == ["b", "a"], "row form: PP0's sort reaches the followers via the row"
    assert _order(_p_stage(1)) == ["b", "a"], "single stage"


def test_n5_row_killed_by_switch_does_not_sort(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    monkeypatch.setenv("FLLIPER_PP_ROW_AUTHORITY", "0")
    assert _order(_p_stage(3, row_only=True)) == ["a", "b"]


def test_n5_group_d_unchanged(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    s = _p_stage(1)
    del s.ps
    assert _order(s) == ["b", "a"]


def test_n5_every_p_stage_of_the_carrierless_form_keeps_the_same_order(monkeypatch):
    """Rank agreement: three carrierless stages, one queue -> one order (none sorts)."""
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    orders = []
    for rank in range(3):
        s = _p_stage(3)
        s.ps.pp_rank = rank
        orders.append(tuple(_order(s)))
    assert len(set(orders)) == 1


# ============================================================ N3: the X-gate abstention names the priority
def test_n3_abstention_of_a_p1_rid_is_counted_and_named(caplog):
    from flliper.srt.managers.scheduler import Scheduler

    s = types.SimpleNamespace(server_args=types.SimpleNamespace(tp_prefill_max_tokens=4096),
                              ps=types.SimpleNamespace(tp_size=3))
    req = types.SimpleNamespace(rid="pdflip-40-400", priority=1)
    with caplog.at_level(logging.INFO):
        assert Scheduler._pdflip_x_refuses(s, req, None) is False   # abstains (admits), uniform
    assert s._pdflip_x_abstained_prio == 1
    assert any("verdict=abstain_no_group_match" in m and "priority=1" in m for m in caplog.messages)
    req0 = types.SimpleNamespace(rid="pdflip-41-401", priority=None)
    assert Scheduler._pdflip_x_refuses(s, req0, None) is False
    assert s._pdflip_x_abstained_prio == 1 and s._pdflip_x_abstained == 2


# ============================================================ N4: PDFLIP MODE-PARITY
def test_n4_local_mode_reads_the_two_switches():
    assert MP.local_mode() == {"lanes": 1, "prio_as_age": 1, "age": 1, "floor": 0}
    with _mode(False):
        assert MP.local_mode() == {"lanes": 1, "prio_as_age": 0, "age": 0, "floor": 1}


def test_n4_compare_and_info_parsing():
    on = {"lanes": 1, "prio_as_age": 1, "age": 1, "floor": 0}
    off = {"lanes": 1, "prio_as_age": 0, "age": 0, "floor": 1}
    assert MP.compare(on, {"P": on, "D": dict(on)}) == []
    diff = MP.compare(on, {"P": on, "D": off})
    assert diff == ["prio_as_age: front=1 D=0"]
    assert MP.compare(on, {"P": None, "D": on})[0].startswith("group P answered without")
    assert MP.mode_of_info([{"internal_states": [{MP.INFO_KEY: on}]}]) == (True, on)
    assert MP.mode_of_info({"internal_states": [{"x": 1}]}) == (True, None)
    assert MP.mode_of_info(None) == (False, None)
    ok, line = MP.verdict_line(on, {"P": on, "D": off})
    assert not ok and line.startswith(MP.MARK + " MISMATCH") and "prio_as_age: front=1 D=0" in line


class _GetResp:
    def __init__(self, body, status=200):
        self.status, self._body = status, body

    async def json(self):
        return self._body


class _GetCm:
    def __init__(self, body, status=200):
        self.r = _GetResp(body, status)

    async def __aenter__(self):
        if isinstance(self.r._body, Exception):
            raise self.r._body
        return self.r

    async def __aexit__(self, *a):
        return False


def _run_parity(bodies_by_group, monkeypatch, real_stop=False):
    """``bodies_by_group``: name -> list of bodies served one per read (an Exception = no answer)."""
    f = _front("P")
    f.MODE_PARITY_RETRY_S = 0.0
    calls = {"P": 0, "D": 0}

    def get(url, timeout=None):
        name = "P" if url.startswith(f.groups["P"].url) else "D"
        seq = bodies_by_group[name]
        b = seq[min(calls[name], len(seq) - 1)]
        calls[name] += 1
        return _GetCm(b)

    f.session = types.SimpleNamespace(get=get)
    stops = []
    if not real_stop:
        monkeypatch.setattr(f, "do_stop", lambda name, detail: stops.append((name, detail)))
    asyncio.run(f._mode_parity_boot())
    return f, stops, calls


def _info(mode):
    return {"internal_states": [{MP.INFO_KEY: mode}]}


def test_n4_front_ok_when_front_p_and_d_agree(monkeypatch, caplog):
    m = MP.local_mode()
    with caplog.at_level(logging.INFO):
        f, stops, _ = _run_parity({"P": [_info(m)], "D": [_info(m)]}, monkeypatch)
    assert stops == [] and f.counters["mode_parity_checked"] == 1
    assert any(x.startswith(MP.MARK + " OK") for x in caplog.messages)
    assert any(x.startswith(MP.MARK + " role=front") for x in caplog.messages)


def test_n4_mismatch_is_the_named_front_stop(monkeypatch):
    other = dict(MP.local_mode(), prio_as_age=0, age=0, floor=1)
    f, stops, _ = _run_parity({"P": [_info(MP.local_mode())], "D": [_info(other)]}, monkeypatch)
    assert len(stops) == 1 and stops[0][0] == MP.STOP_NAME == "W-MODE-PARITY"
    assert "prio_as_age: front=1 D=0" in stops[0][1]
    assert f.counters["mode_parity_mismatch"] == 1


def test_n4_a_group_answering_without_the_block_stops(monkeypatch):
    f, stops, _ = _run_parity({"P": [_info(MP.local_mode())], "D": [{"internal_states": [{}]}]}, monkeypatch)
    assert stops and "group D answered without" in stops[0][1]


def test_n4_a_group_not_up_yet_is_read_again_not_judged(monkeypatch):
    m = MP.local_mode()
    f, stops, calls = _run_parity({"P": [_info(m)], "D": [ConnectionError("down"), None, _info(m)]}, monkeypatch)
    assert stops == [] and calls["D"] == 3 and calls["P"] == 1


def test_n4_switch_off_names_but_does_not_stop(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_MODE_PARITY_CHECK", "0")
    other = dict(MP.local_mode(), lanes=0, prio_as_age=0, age=0, floor=0)
    f, stops, _ = _run_parity({"P": [_info(other)], "D": [_info(MP.local_mode())]}, monkeypatch)
    assert stops == [] and f.counters["mode_parity_mismatch"] == 1


def test_n4_real_stop_puts_the_front_into_stop(monkeypatch):
    other = dict(MP.local_mode(), prio_as_age=0, age=0, floor=1)
    f, _, _ = _run_parity({"P": [_info(other)], "D": [_info(MP.local_mode())]}, monkeypatch, real_stop=True)
    assert f.state == "STOP" and f.stop.name == "W-MODE-PARITY"


def test_n4_wiring_pins():
    from flliper.srt.managers.scheduler import Scheduler

    assert "self._mode_parity_boot()" in inspect.getsource(Front.startup)
    assert "_mode_parity.log_rank(self)" in inspect.getsource(Scheduler.init_request_receiver)
    gis = inspect.getsource(Scheduler.get_internal_state)
    assert "ret[_mode_parity.INFO_KEY] = _mode_parity.local_mode()" in gis


def test_n4_rank_line(caplog, monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    s = types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=2, tp_rank=0))
    with caplog.at_level(logging.INFO):
        assert MP.log_rank(s) == MP.local_mode()
    assert any(m.startswith(MP.MARK + " role=group group=P pp_rank=2 tp_rank=0 lanes=1 prio_as_age=1")
               for m in caplog.messages)
