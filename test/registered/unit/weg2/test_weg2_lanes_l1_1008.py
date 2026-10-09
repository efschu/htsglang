"""PRIORITY LANES 1008, part L1 (field + state), plan deskq/PLAN-PRIO-LANES-1008.md sections 0-2.

What is pinned (all hermetic, no GPU, no network):

* the field: ``priority`` flows OpenAI protocol / Anthropic adapter -> the front's raw payload read -> Pending.lane,
  the request book row and state.json (``front.lane_floor`` / ``lane_epoch`` / ``lanes``); a missing field is lane 0,
  a negative / malformed one is lane 0 with a ``WEG2 LANE-FIELD`` warning;
* the switch (``SGLANG_WEG2_LANES``, default off): OFF = the front does not read ``priority``, no lane key in
  state.json / the request book / ``request_done`` / any log line, the Anthropic adapter drops the field as before;
* ``weg2/lanes.py``: names fixed for L2-L4 (markers, the RPC), ``LaneState`` (floor, epoch, counters, no controller);
* the progress beacon's lane trailer (16 bytes behind the 32 only with the switch on);
* the catalog entries (CURATED) and the two edges K135/K136.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
import tempfile
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.weg2 import arrival_seat_rule as _asr  # noqa: E402,F401
from sglang.srt.weg2 import front_requests as FRQ  # noqa: E402
from sglang.srt.weg2 import front_state_ipc as FSI  # noqa: E402
from sglang.srt.weg2 import lanes as LN  # noqa: E402
from sglang.srt.weg2 import progress_beacon as PB  # noqa: E402
from sglang.srt.weg2.front import Front, Pending  # noqa: E402

X = 4096
HERE = os.path.dirname(os.path.abspath(__file__))
WEG2 = os.path.join(HERE, "..", "..", "..", "..", "python", "sglang", "srt", "weg2")


class _Req:
    def __init__(self, path, payload):
        self.path = path
        self._payload = payload
        self._d = {}

    async def json(self):
        return self._payload

    def __setitem__(self, k, v):
        self._d[k] = v

    def get(self, k, default=None):
        return self._d.get(k, default)


def _front(awake="D"):
    f = Front("http://p", "http://d", awake, "t", "", 0, 0, {}, 45.0, tp_prefill_max_tokens=X)
    f.state = "serving"
    f.admit_d = True
    return f


def _arrive(payload, path="/v1/chat/completions"):
    """One request handed to the front while D flips to P: it is queued for P (the plan's Pending)."""
    async def go():
        f = _front("D")
        f.state = "flipping"
        f._flip_dst = "P"

        async def fake_leg2(*a, **k):
            return "served"
        f.leg2 = fake_leg2
        task = asyncio.ensure_future(f.handle_generate(_Req(path, payload)))
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=0.5)
        except asyncio.TimeoutError:
            pass
        out = (f, list(f.queue))
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return out

    with envs.SGLANG_WEG2_ENABLE_ARRIVAL_SEAT_RULE.override(True):
        return asyncio.run(go())


def _body(priority=...):
    p = {"messages": [{"role": "user", "content": "x" * 2300}]}
    if priority is not ...:
        p["priority"] = priority
    return p


# ------------------------------------------------------------------ names and env
def test_the_names_l2_to_l4_are_written_against():
    assert LN.RPC_LANE_FLOOR == "/weg2/lane_floor"
    assert (LN.RPC_KEY_FLOOR, LN.RPC_KEY_EPOCH) == ("floor", "epoch")
    assert LN.MARK_PREEMPT == "WEG2 LANE-PREEMPT"
    assert LN.MARK_RESUME == "WEG2 LANE-RESUME"
    assert LN.MARK_DEFER == "WEG2 LANE-DEFER"
    assert LN.MARK_REPREFILL == "WEG2 LANE-REPREFILL"
    assert LN.MARK_D_PARK_PARK == "WEG2-D-PARK park(lane)"
    assert LN.MARK_D_PARK_REQUEUE == "WEG2-D-PARK requeue(lane)"
    assert LN.MARK_PR_FLOOR == "PR LANE-FLOOR"
    assert LN.LANE_STATES == ("active", "held", "parked", "resuming", "reprefill")
    assert LN.LANE_COUNT_KEYS == ("pending", "running_p", "running_d", "parked")
    assert (LN.ENV_LANES, LN.ENV_KEEPALIVE_S, LN.ENV_PREEMPT_CHUNK_TOKENS) == (
        "SGLANG_WEG2_LANES", "SGLANG_WEG2_LANE_KEEPALIVE_S", "SGLANG_WEG2_LANE_PREEMPT_CHUNK_TOKENS")


def test_env_defaults_and_readers():
    assert envs.SGLANG_WEG2_LANES.get() is False
    assert envs.SGLANG_WEG2_LANE_KEEPALIVE_S.get() == 15
    assert envs.SGLANG_WEG2_LANE_PREEMPT_CHUNK_TOKENS.get() == 0
    assert (LN.enabled(), LN.keepalive_s(), LN.preempt_chunk_tokens()) == (False, 15, 0)
    with envs.SGLANG_WEG2_LANES.override(True), envs.SGLANG_WEG2_LANE_KEEPALIVE_S.override(7), \
            envs.SGLANG_WEG2_LANE_PREEMPT_CHUNK_TOKENS.override(2048):
        assert (LN.enabled(), LN.keepalive_s(), LN.preempt_chunk_tokens()) == (True, 7, 2048)


# ------------------------------------------------------------------ the field
@pytest.mark.parametrize("raw,want", [
    (None, (0, None)), (0, (0, None)), (1, (1, None)), (7, (7, None)),
    ("2", (2, None)), (" 3 ", (3, None)), (2.0, (2, None)),
    (-1, (0, "negative")), ("-5", (0, "negative")), (-0.0 - 3.0, (0, "negative")),
    (True, (0, "invalid")), (False, (0, "invalid")), ("x", (0, "invalid")), ("", (0, "invalid")),
    (2.5, (0, "invalid")), (float("nan"), (0, "invalid")), (float("inf"), (0, "invalid")),
    ([1], (0, "invalid")), ({"a": 1}, (0, "invalid")),
    (10 ** 30, (LN.LANE_MAX, "clamped")),
])
def test_parse_lane(raw, want):
    assert LN.parse_lane(raw) == want


def test_lane_of_reads_the_raw_payload_and_never_raises():
    assert LN.lane_of({"priority": 4}) == 4
    assert LN.lane_of({}) == 0
    assert LN.lane_of({"priority": -2}) == 0
    assert LN.lane_of(None) == 0 and LN.lane_of("priority") == 0 and LN.lane_of([1]) == 0


# ------------------------------------------------------------------ field flow: protocol layers
def test_the_openai_protocol_carries_the_field_to_the_scheduler_request():
    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest, CompletionRequest

    chat = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], priority=3)
    comp = CompletionRequest(model="m", prompt="hi", priority=2)
    assert chat.priority == 3 and comp.priority == 2
    assert ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}]).priority is None


def _convert_anthropic(**kwargs):
    from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
    from sglang.srt.entrypoints.anthropic.serving import AnthropicServing

    class _NoReasoning:
        def apply_reasoning_enabled(self, request, enabled):
            pass

    base = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}
    base.update(kwargs)
    serving = AnthropicServing.__new__(AnthropicServing)
    serving._merge_inline_system = False
    serving.openai_serving_chat = _NoReasoning()
    return AnthropicServing._convert_to_chat_completion_request(serving, AnthropicMessagesRequest(**base))


def test_the_anthropic_adapter_forwards_priority_only_with_the_switch():
    assert _convert_anthropic(priority=3).priority is None, "switch off: dropped, exactly as before the field existed"
    with envs.SGLANG_WEG2_LANES.override(True):
        assert _convert_anthropic(priority=3).priority == 3
        assert _convert_anthropic(priority=0).priority == 0
        assert _convert_anthropic().priority is None, "absent stays absent"
        assert _convert_anthropic(priority=2, stream=True).priority == 2


@pytest.mark.parametrize("raw", ["abc", 1.5, [1], {"a": 1}, float("nan"), float("inf"), True, ""])
@pytest.mark.parametrize("switch", [False, True])
def test_a_malformed_anthropic_priority_never_refuses_the_request(raw, switch):
    """Before L1 `extra="ignore"` swallowed any `priority`; declaring the field must not make /v1/messages
    refuse a request it ran before (switch off: nothing changes). Malformed -> the field is dropped."""
    with envs.SGLANG_WEG2_LANES.override(switch):
        assert _convert_anthropic(priority=raw).priority is None


@pytest.mark.parametrize("raw,want", [("3", 3), (2.0, 2), (-4, 0), (10 ** 30, LN.LANE_MAX)])
def test_the_anthropic_adapter_forwards_the_normalised_lane(raw, want):
    with envs.SGLANG_WEG2_LANES.override(True):
        assert _convert_anthropic(priority=raw).priority == want
    assert _convert_anthropic(priority=raw).priority is None  # switch off: dropped


def test_the_anthropic_model_declares_the_field():
    from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest

    r = AnthropicMessagesRequest(model="m", messages=[{"role": "user", "content": "hi"}], max_tokens=1, priority=5)
    assert r.priority == 5


# ------------------------------------------------------------------ field flow: front (switch on)
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages", "/generate", "/v1/completions"])
def test_priority_reaches_pending_book_and_state(path):
    with envs.SGLANG_WEG2_LANES.override(True):
        f, queued = _arrive(_body(2), path)
        assert len(queued) == 1 and queued[0].lane == 2
        rid = queued[0].rid
        assert Front._req_book(f).rows[rid]["lane"] == 2
        st = f.state_dict()
        assert st["lane_floor"] == 0 and st["lane_epoch"] == 0
        assert st["lanes"] == {"2": {"pending": 1, "running_p": 0, "running_d": 0, "parked": 0}}
        ipc = f._ipc_front_fields()
        assert (ipc["lane_floor"], ipc["lane_epoch"], ipc["lanes"]) == (0, 0, st["lanes"])
        assert not set(ipc) & set(FSI.HOST_FRONT_KEYS)
        json.dumps(st)  # state.json / /weg2/state must serialise
        json.dumps(ipc)


def test_a_missing_priority_is_lane_0_without_a_warning(caplog):
    caplog.set_level(logging.INFO)
    with envs.SGLANG_WEG2_LANES.override(True):
        f, queued = _arrive(_body())
        lanes = f.state_dict()["lanes"]
    assert queued[0].lane == 0
    assert lanes == {"0": {"pending": 1, "running_p": 0, "running_d": 0, "parked": 0}}
    assert not any(LN.MARK_FIELD in m for m in caplog.messages)


@pytest.mark.parametrize("raw,why", [(-3, "negative"), ("abc", "invalid"), (True, "invalid"), (1.5, "invalid")])
def test_a_negative_or_malformed_priority_is_lane_0_with_a_warning(raw, why, caplog):
    caplog.set_level(logging.INFO)
    with envs.SGLANG_WEG2_LANES.override(True):
        f, queued = _arrive(_body(raw))
    assert queued[0].lane == 0
    line = next(m for m in caplog.messages if LN.MARK_FIELD in m)
    assert queued[0].rid in line and "lane 0" in line and why in line
    assert [r for r in caplog.records if LN.MARK_FIELD in r.getMessage()][0].levelno == logging.WARNING


@pytest.mark.parametrize("raw,want", [
    (-3, 0), ("abc", 0), (True, 0), (1.5, 0), ("2", 2), (2.0, 2), (4, 4), (0, 0), (10 ** 30, LN.LANE_MAX),
])
def test_the_forwarded_payload_carries_the_normalised_lane(raw, want):
    """Front and scheduler must agree on one number: P and D get `payload["priority"]` as the front read it
    (negative -> 0, malformed -> 0), so `req.priority >= floor` can never lock a lane-0 request out and the
    P server never refuses a request for a malformed priority."""
    with envs.SGLANG_WEG2_LANES.override(True):
        body = _body(raw)
        f, queued = _arrive(body)
    assert queued[0].lane == want
    assert queued[0].payload["priority"] == want and type(queued[0].payload["priority"]) is int
    assert body["priority"] == want, "the dict the front posts on to P and D"


def test_an_absent_priority_stays_absent_in_the_forwarded_payload():
    with envs.SGLANG_WEG2_LANES.override(True):
        body = _body()
        _arrive(body)
    assert "priority" not in body


def test_switch_off_the_forwarded_payload_is_untouched():
    for raw in (-3, "abc", True, 1.5, 5):
        body = _body(raw)
        _arrive(body)
        assert body["priority"] == raw and type(body["priority"]) is type(raw)


def test_the_lane_note_goes_with_the_rid_and_request_done_carries_the_lane():
    with envs.SGLANG_WEG2_LANES.override(True):
        f, queued = _arrive(_body(4))
        rid = queued[0].rid
        assert f._lane_for(rid) == 4
        rec, _ev = Front._req_book(f).done(rid, 100.0, 200, 0)
        assert rec["lane"] == 4
        f._lane_end(rid)
        assert f._lane_for(rid) == 0


# ------------------------------------------------------------------ switch OFF: nothing changes
def test_switch_off_the_front_does_not_read_priority_and_writes_no_lane_field(caplog):
    caplog.set_level(logging.DEBUG)
    f, queued = _arrive(_body(5))
    assert queued[0].lane == 0, "off: the field is not read"
    assert "_lane_state_obj" not in f.__dict__, "off: no lane state is even built"
    st = f.state_dict()
    ipc = f._ipc_front_fields()
    for d in (st, ipc):
        assert not [k for k in d if k.startswith("lane")]
    assert "lane" not in Front._req_book(f).rows[queued[0].rid]
    rec, _ = Front._req_book(f).done(queued[0].rid, 100.0, 200, 0)
    assert "lane" not in rec
    assert not any("LANE" in m.upper() and "WEG2" in m for m in caplog.messages)
    assert f._lane_block() == {}


def test_state_keys_with_the_switch_on_are_the_off_keys_plus_the_three_lane_keys():
    f_off, _ = _arrive(_body(1))
    with envs.SGLANG_WEG2_LANES.override(True):
        f_on, _ = _arrive(_body(1))
        on_state, on_ipc = f_on.state_dict(), f_on._ipc_front_fields()
    assert set(on_state) - set(f_off.state_dict()) == {"lane_floor", "lane_epoch", "lanes"}
    assert set(on_ipc) - set(f_off._ipc_front_fields()) == {"lane_floor", "lane_epoch", "lanes"}
    assert set(f_off.state_dict()) - set(on_state) == set()


def test_pending_without_a_lane_is_lane_0():
    p = Pending(rid="r", path="/generate", payload={}, text="x", t_arrive=0.0, fut=None)  # type: ignore[arg-type]
    assert p.lane == 0


# ------------------------------------------------------------------ LaneState (no controller logic)
def test_lane_state_floor_epoch_and_counters():
    s = LN.LaneState()
    assert (s.lane_floor, s.lane_epoch) == (0, 0)
    assert s.set_floor(0) is False and s.lane_epoch == 0, "no change, no epoch"
    assert s.set_floor(2) is True and (s.lane_floor, s.lane_epoch) == (2, 1)
    assert s.set_floor(2) is False and s.lane_epoch == 1
    assert s.set_floor(0) is True and (s.lane_floor, s.lane_epoch) == (0, 2)
    assert s.set_floor(-4) is False, "a floor below 0 is 0"
    for rid, lane in (("a", 0), ("b", 1), ("c", 1), ("d", 2), ("e", 2)):
        s.note(rid, lane)
    assert s.lane_for("zzz") == 0
    c = s.counts(pending=["a", "x"], running_p=["b"], running_d=["c", "d"], parked=["e"])
    assert c == {
        "0": {"pending": 2, "running_p": 0, "running_d": 0, "parked": 0},   # a, and the unnoted x (lane 0)
        "1": {"pending": 0, "running_p": 1, "running_d": 1, "parked": 0},
        "2": {"pending": 0, "running_p": 0, "running_d": 1, "parked": 1},
    }
    assert list(c) == ["0", "1", "2"]
    # a rid in several sets counts once, parked > running_d > running_p > pending
    assert s.counts(pending=["b"], running_p=["b"], running_d=["b"], parked=["b"]) == {
        "1": {"pending": 0, "running_p": 0, "running_d": 0, "parked": 1}}
    assert s.counts() == {}
    s.end("e")
    assert s.lane_for("e") == 0
    blk = s.state_block(pending=["a"])
    assert blk == {"lane_floor": 0, "lane_epoch": 2,
                   "lanes": {"0": {"pending": 1, "running_p": 0, "running_d": 0, "parked": 0}}}


def test_lane_state_is_bounded():
    s = LN.LaneState()
    for i in range(LN.LaneState.MAX_RIDS + 10):
        s.note(f"r{i}", 1)
    assert len(s.rid_lane) == LN.LaneState.MAX_RIDS
    assert s.lane_for("r0") == 0 and s.lane_for(f"r{LN.LaneState.MAX_RIDS + 9}") == 1


def test_lane_state_has_no_controller_surface():
    """L1 stores and counts; the controller (L4) decides. No method names a transition."""
    pub = {n for n in dir(LN.LaneState) if not n.startswith("_")}
    assert pub <= {"MAX_RIDS", "note", "lane_for", "end", "set_floor", "counts", "state_block",
                   "lane_floor", "lane_epoch", "rid_lane"}


# ------------------------------------------------------------------ the request book
def test_request_book_lane_row_and_record():
    b = FRQ.RequestBook(0.0)
    b.arrive("r1", 1.0, 0)
    assert "lane" not in b.rows["r1"]
    b.lane("r1", 3)
    assert b.rows["r1"]["lane"] == 3
    b.lane("unknown", 3)  # a rid the book never saw: no row, no error
    rec, _ = b.done("r1", 2.0, 200, 0)
    assert rec["lane"] == 3
    b.arrive("r2", 1.0, 0)
    rec2, _ = b.done("r2", 2.0, 200, 0)
    assert "lane" not in rec2, "a row the switch-on front never noted carries no lane"


# ------------------------------------------------------------------ the progress beacon's lane trailer
def _beacon(tmp, lanes_on):
    w = PB._Writer()
    with mock.patch.object(PB, "beacon_dir", return_value=tmp), \
            mock.patch.dict(os.environ, {"SGLANG_WEG2_GROUP": "D"}), \
            envs.SGLANG_WEG2_LANES.override(lanes_on):
        w.beat(1, True)
        w.beat(1, False)
        w.beat_lane(3, 9)
    return w


def test_beacon_file_stays_32_bytes_with_the_switch_off():
    with tempfile.TemporaryDirectory() as tmp:
        _beacon(tmp, False)
        path = os.path.join(tmp, f"D-pid{os.getpid()}.bin")
        assert os.path.getsize(path) == 32
        assert PB.read_group_lane(tmp, "D", 1, session_of=lambda pid: 1) == {}, "no trailer, no lane reading"
        assert PB.read_group(tmp, "D", 1, session_of=lambda pid: 1)[os.getpid()][0] == 1


def test_beacon_lane_trailer_with_the_switch_on_leaves_the_32_bytes_readable():
    with tempfile.TemporaryDirectory() as tmp:
        _beacon(tmp, True)
        path = os.path.join(tmp, f"D-pid{os.getpid()}.bin")
        assert os.path.getsize(path) == 48
        raw = open(path, "rb").read()
        assert struct.unpack_from("<qq", raw, 32) == (3, 9)
        assert PB.read_group_lane(tmp, "D", 1, session_of=lambda pid: 1) == {os.getpid(): (3, 9)}
        assert PB.read_group_lane(tmp, "D", 1, session_of=lambda pid: 2) == {}, "another session's rank is not read"
        # today's readers (32 bytes) read on unchanged
        ct, ts, td = PB.read_group(tmp, "D", 1, session_of=lambda pid: 1)[os.getpid()]
        assert ct == 1 and ts > 0 and td >= ts
        assert PB.progress({os.getpid(): (0, 0, 0)}, {os.getpid(): (ct, ts, td)}) is not None


# ------------------------------------------------------------------ catalog, edges, launch snapshot
def test_catalog_entries_and_edges_for_the_three_envs():
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("t1008_curated", os.path.join(WEG2, "profile_catalog_curated.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["t1008_curated"] = mod
    spec.loader.exec_module(mod)
    for n in ("SGLANG_WEG2_LANES", "SGLANG_WEG2_LANE_KEEPALIVE_S", "SGLANG_WEG2_LANE_PREEMPT_CHUNK_TOKENS"):
        c = mod.CURATED[n]
        assert c["kind"] == "env" and len(c["text"]) > 40 and c.get("satz_quelle") and c["depends"] == []
        assert hasattr(envs, n), n
    with open(os.path.join(WEG2, "kantenkatalog_1004.json"), encoding="utf-8") as fh:
        kanten = {k["id"]: k for k in json.load(fh)["kanten"]}
    for kid, von in (("K135", "SGLANG_WEG2_LANE_KEEPALIVE_S"), ("K136", "SGLANG_WEG2_LANE_PREEMPT_CHUNK_TOKENS")):
        k = kanten[kid]
        assert (k["von"], k["nach"], k["rel"]) == (von, "SGLANG_WEG2_LANES", "braucht")
        # the lane envs exist on the NF line only (27B 84d04adae1 has none): the edge documents NF code only
        assert k["beleg"]["datei"] == "python/sglang/srt/environ.py" and k["baeume"] == ["nf"]


def test_the_launch_snapshot_shows_the_lane_envs_without_a_launcher_change():
    """The envs reach the front and both groups by inheritance (launcher: fenv = dict(os.environ), build_env:
    canonical_env(dict(os.environ))); state.json launch.* lists every SGLANG_ name."""
    from sglang.srt.weg2 import state_file

    snap = state_file.launch_snapshot(["x"], {"SGLANG_WEG2_LANES": "1", "SGLANG_WEG2_LANE_KEEPALIVE_S": "15"})
    assert snap["env"] == {"SGLANG_WEG2_LANES": "1", "SGLANG_WEG2_LANE_KEEPALIVE_S": "15"}


def test_the_lane_keys_reach_state_json_through_the_one_writer():
    """front.lane_floor / front.lane_epoch / front.lanes are keys of the front's own subtree: state_file accepts them."""
    from sglang.srt.weg2 import state_file

    with tempfile.TemporaryDirectory() as root, envs.SGLANG_WEG2_LANES.override(True):
        sd = state_file.init(root, "nflanes-boot-20261008T180000Z-1008", "boot", {})
        f, queued = _arrive(_body(2))
        assert FSI.publish_front_fields(sd, f._ipc_front_fields())
        fr = state_file.read(sd)["front"]
        assert fr["lane_floor"] == 0 and fr["lane_epoch"] == 0
        assert fr["lanes"] == {"2": {"pending": 1, "running_p": 0, "running_d": 0, "parked": 0}}
