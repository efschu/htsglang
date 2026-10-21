"""P-COMMIT MESSTOR (21.10.2026, review nf-rc30-pcommit-review-1021 §2.5): instruments only, no behaviour change.

Before the "arrival during PP0's launch" class (rc30: 28 drains / 40.6 s) is built, rc31 must say per boot how many
seconds qualify. Pinned (hermetic, CPU):
  1. W2C WOULD-ANNOUNCE: at a launch end WITHOUT an announcement PP0 probes its request socket (zero-timeout poll,
     nothing consumed) and C2(b); the next pass top names generate vs control requests; ``qualifies`` only for a
     readable socket + C2(b) + arrivals only. The frame is unchanged (proxy_extra returns the same extras).
  2. S50 WOULD-ACK: a follower's real ack line names how long after its own read end it left, whether own == told
     and whether its key chain digests to PP0's; the ack itself is unchanged.
  3. #PDRAIN sums the drain wait behind qualifying rids (``w2c_would_ms_sum``) and behind paced rids
     (``paced_ms_sum``), only for drains with wait_ms >= 500 (review B4).
  4. Wiring.
"""

import inspect
import logging
import os
import sys
import time as _time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flliper.srt.managers import pdflip_store_told as told  # noqa: E402
from flliper.srt.managers import pdflip_told_ack_early as tae  # noqa: E402
from flliper.srt.managers import pdflip_told_fallback as fb  # noqa: E402
from flliper.srt.pdflip import pp_chain_announce as w2c  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

RID = "pdflip-141-1469"     # metal fwd217: front 09:33:17.378, PP0 inside its fwd216 launch until 09:33:20.8


class GenerateReqInput:      # the type NAME is what the classifier reads (io_struct's class name)
    def __init__(self, rid):
        self.rid = rid


class AbortReq:
    def __init__(self, rid):
        self.rid = rid


class _RR:
    def __init__(self, readable):
        self.readable = readable
        self.polled = 0

    def control_message_pending(self):
        self.polled += 1
        return self.readable


def _pp0(readable=True, others=False, carried=False):
    batch = SimpleNamespace(reqs=[SimpleNamespace(rid="pdflip-142-1471")])
    return SimpleNamespace(
        ps=SimpleNamespace(pp_rank=0, pp_size=3, tp_size=1),
        _pdflip_told_paced_on=True, _pdflip_store_told_armed=True, pdflip_dormant=False,
        mbs=[None, batch, batch if others else None],         # slot 1 just launched (fwd216)
        chunked_req=SimpleNamespace(rid="c") if carried else None,
        request_receiver=_RR(readable),
        waiting_queue=[],
    )


@pytest.fixture(autouse=True)
def _reset():
    w2c.reset()
    tae.reset_would()
    yield


# ---------------------------------------------------------------------------
# 1. W2C WOULD-ANNOUNCE


def _would_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("W2C WOULD-ANNOUNCE")]


def test_metal_fwd217_readable_socket_with_c2b_and_an_arrival_qualifies(caplog):
    s = _pp0()
    w2c.note_would_announce(s, 1)
    assert s._w2c_would is not None and s.request_receiver.polled == 1
    with caplog.at_level(logging.INFO, logger=w2c.logger.name):
        w2c.pp0_would_at_top(s, [GenerateReqInput(RID)])
    line = _would_lines(caplog)
    assert len(line) == 1 and "qualifies=1" in line[0] and "c2b=1" in line[0] and "gen=1 ctrl=0" in line[0]
    assert RID in s._w2c_would_rids and s._w2c_would is None
    assert w2c.would_stats() == {"n": 1, "qual": 1}


@pytest.mark.parametrize("case", ["other_slot_in_flight", "carried_chunk", "control_in_chain", "arrival_missing"])
def test_not_qualifying(case, caplog):
    s = _pp0(others=case == "other_slot_in_flight", carried=case == "carried_chunk")
    w2c.note_would_announce(s, 1)
    reqs = {"control_in_chain": [GenerateReqInput(RID), AbortReq("x")], "arrival_missing": []}.get(
        case, [GenerateReqInput(RID)])
    with caplog.at_level(logging.INFO, logger=w2c.logger.name):
        w2c.pp0_would_at_top(s, reqs)
    line = _would_lines(caplog)
    assert len(line) == 1 and "qualifies=0" in line[0]
    assert not getattr(s, "_w2c_would_rids", None)


@pytest.mark.parametrize("case", ["socket_empty", "not_paced", "dormant", "follower"])
def test_no_probe_record(case, caplog):
    s = _pp0(readable=case != "socket_empty")
    if case == "not_paced":
        s._pdflip_told_paced_on = False
    elif case == "dormant":
        s.pdflip_dormant = True
    elif case == "follower":
        s.ps.pp_rank = 1
    w2c.note_would_announce(s, 1)
    assert getattr(s, "_w2c_would", None) is None
    with caplog.at_level(logging.INFO, logger=w2c.logger.name):
        w2c.pp0_would_at_top(s, [GenerateReqInput(RID)])
    assert _would_lines(caplog) == []


def test_frame_is_unchanged_and_an_announced_frame_is_not_probed(monkeypatch):
    s = _pp0()
    base = {"row": 1}
    monkeypatch.setattr(w2c, "_pp0_announce", lambda sch, mb: None)
    assert w2c.proxy_extra(s, 1, base) is base                    # instrument only: the same extras object
    assert s._w2c_would is not None
    s2 = _pp0()
    monkeypatch.setattr(w2c, "_pp0_announce", lambda sch, mb: 7)
    out = w2c.proxy_extra(s2, 1, base)
    assert out[w2c.KEY] == 7 and getattr(s2, "_w2c_would", None) is None and s2.request_receiver.polled == 0


def test_a_broken_probe_never_raises():
    s = _pp0()

    def boom():
        raise RuntimeError("poll")

    s.request_receiver = SimpleNamespace(control_message_pending=boom)
    w2c.note_would_announce(s, 1)
    assert getattr(s, "_w2c_would", None) is None
    s._w2c_would = {"t": 0.0}                                    # malformed record: the top closes it silently
    w2c.pp0_would_at_top(s, [GenerateReqInput(RID)])
    assert s._w2c_would is None


# ---------------------------------------------------------------------------
# 2. S50 WOULD-ACK


def test_would_ack_names_the_gap_between_read_end_and_the_real_ack(caplog):
    # metal 09:33:20 (S50 doc / rc13 shape): the read ended in the follower's forward, the ack left ~2.5 s later
    tae.note_read_end(SimpleNamespace(request_id=RID))
    tae._READ_END[RID] -= 2.547                                  # 'PDFLIP-LOAD-DEVICE ... harvest_ms=2547' (PP1)
    tae.note_read_ahead(None, RID, 72448, True)
    with caplog.at_level(logging.INFO, logger=tae.logger.name):
        tae.note_ack(SimpleNamespace(ps=SimpleNamespace(pp_rank=1)), RID, 72448, 72448)
    line = [r.getMessage() for r in caplog.records if r.getMessage().startswith("S50 WOULD-ACK")]
    assert len(line) == 1 and "eq=1" in line[0] and "digest_eq=1" in line[0]
    st = tae.would_stats()
    assert st["n"] == 1 and st["eq_n"] == 1 and 2540 <= st["gain_ms_sum"] <= 2700
    assert RID not in tae._READ_END and RID not in tae._AHEAD


def test_would_ack_depth_mismatch_or_unknown_read_end_is_not_counted(caplog):
    tae.note_read_end(SimpleNamespace(request_id="a"))
    with caplog.at_level(logging.INFO, logger=tae.logger.name):
        tae.note_ack(None, "a", 1000, 2000)                       # own != told: the combined Admit could not use it
        tae.note_ack(None, "b", 2000, 2000)                       # no read end seen on this process
    st = tae.would_stats()
    assert st["n"] == 2 and st["eq_n"] == 0 and st["gain_ms_sum"] == 0.0


def test_read_end_record_is_bounded_and_never_raises():
    for i in range(tae._READ_END_MAX + 10):
        tae.note_read_end(SimpleNamespace(request_id=f"r{i}"))
    assert len(tae._READ_END) == tae._READ_END_MAX
    tae.note_read_end(None)
    tae.note_read_end(object())


def test_follower_absorb_notes_the_read_ahead_with_the_digest(monkeypatch):
    req = SimpleNamespace(rid=RID)
    s = SimpleNamespace(waiting_queue=[req])
    monkeypatch.setattr(told, "_pp0_keys_digest", lambda r: "abc:3")
    told._note_would_ack_ahead(s, RID, 72448, SimpleNamespace(keys_digest="abc:3"))
    assert tae._AHEAD[RID][1:] == (72448, True)
    told._note_would_ack_ahead(s, RID, 72448, SimpleNamespace(keys_digest="zzz:3"))
    assert tae._AHEAD[RID][2] is False
    told._note_would_ack_ahead(s, RID, 72448, SimpleNamespace())  # no digest on the wire: unknown
    assert tae._AHEAD[RID][2] is None


# ---------------------------------------------------------------------------
# 3. #PDRAIN sums


def test_pdrain_sums_the_drain_behind_qualifying_rids(caplog):
    req = SimpleNamespace(rid=RID, multimodal_inputs=None)
    s = SimpleNamespace(waiting_queue=[req], _pdflip_store_held={}, _pdflip_told_pacing={RID: object()},
                        tree_cache=None, chunked_req=None, _w2c_would_rids={RID: 0.0, "gone": 0.0})
    s._pdrain_entry = (_time.monotonic() - 3.987, told.pdrain_snapshot(s))
    with caplog.at_level(logging.INFO, logger=told.logger.name):
        told._log_pdrain(s, 1, _time.monotonic(), (_time.monotonic() - 4.0, 15019, 58880))
    line = [r.getMessage() for r in caplog.records if r.getMessage().startswith("#PDRAIN")]
    assert len(line) == 1 and "w2c_would=1" in line[0] and "cause=paced" in line[0]
    sums = s._pdrain_ms_sums
    assert 3980 <= sums["w2c_would"] <= 4100 and sums["paced"] == sums["w2c_would"] == sums["pause"]
    assert "gone" not in s._w2c_would_rids                       # pruned: a rid never qualifies twice


# ---------------------------------------------------------------------------
# 4. wiring


def test_wiring():
    assert "note_would_announce(scheduler, mb_id)" in inspect.getsource(w2c.proxy_extra)
    pub = inspect.getsource(told.pp0_publish)
    assert pub.index("pp0_would_at_top(scheduler, recv_reqs)") < pub.index("_twin.tick(scheduler)")
    pump = inspect.getsource(fb.follower_pump)
    assert pump.index("st.outbox.append((rid, own))") < pump.index("_tae.note_ack(scheduler, rid, own, told)")
    absorb = inspect.getsource(told._follower_absorb_impl)
    assert absorb.index("_fb.follower_expect(scheduler, rid, told)") < absorb.index("_note_would_ack_ahead(")
    import flliper.srt.managers.cache_controller as cc

    assert "_told_ack_early.note_read_end(operation)" in inspect.getsource(cc)
