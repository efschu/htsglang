"""F3 (29.09., FLIPZEIT-VERLAUF-0929.md): after D's wake the wake's hand-offs
extend and decode FIRST; a flip-parked resume with a real tail extend waits.

Metal (rc12z4 ep9, 05:55:49): the legs were done after 1.78 s, then D resumed
the flip-parked pdflip-1-19 first -- 64 tail tokens, one eager expert pass of
2500 ms ('HOST-ANON-PASS phase=EXTEND tokens=64 wall_ms=2500') -- while the six
held requests of the same wake waited behind the park barrier
('SETTLE-RELEASE held_after_wake_s=3.0'); the first decode came 8.3 s after
the flip began. Flip time is P end -> the first DECODE token, so the parked
resume's tail sat on the critical path of every held request.

The metal path is ``d_park_runtime.admission`` -> ``d_seats.admission_gate``
-> ``AdmissionGate.skip`` in the admission loop (scheduler.py
``_d_park_gate.skip``). RED on ad95392095: the switch does not exist, the
parked resume goes first and every hand-off of the wake is skipped as
``pdflip_d_park_first``.
"""
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.pdflip import d_park_read, d_park_runtime  # noqa: E402
from flliper.srt.pdflip import d_seats as ds  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(__file__)

WAKE = 3


@pytest.fixture(autouse=True)
def _group_d(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv("FLLIPER_PDFLIP_D_PARK", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_SEAT_ROTATE", "0")
    monkeypatch.setenv("FLLIPER_PDFLIP_PARK_READ_CAP", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_ENABLE_D_DECODE_FIRST", "1")
    monkeypatch.delenv("FLLIPER_PDFLIP_D_DECODE_FIRST_TAIL", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_D_DECODE_FIRST_ROUNDS", raising=False)


def _req(rid, seq, *, parked=False, cohort=WAKE, tokens=130, resumable=64):
    r = types.SimpleNamespace(rid=rid, kv_arrival_seq=seq,
                              origin_input_ids=list(range(tokens - 30)), output_ids=list(range(30)))
    if cohort is not None:
        r._pdflip_settled_wake = cohort          # WT: released by this wake
    if parked:
        ds.mark_parked(r, ds.SITE_FLIP)
        setattr(r, d_park_read.CAP_ATTR, (resumable, tokens))  # the park's resumable depth
    return r


def _sched(waiting, settle=()):
    return types.SimpleNamespace(waiting_queue=list(waiting), pdflip_post_wake_settle=list(settle),
                                 pdflip_dormant_hold=[], _pdflip_wake_seq=WAKE)


def _running(*reqs):
    return types.SimpleNamespace(reqs=list(reqs))


def _decode_round(sched):
    d_park_runtime.note_decode_round(
        sched, types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_decode=lambda: True)))


def test_the_wake_hand_offs_go_first_the_parked_resume_keeps_its_seat():
    p = _req("pdflip-1-19", 1, parked=True)            # tail 130 - 64 = 66 tokens
    h1, h2 = _req("pdflip-2-20", 5), _req("pdflip-2-21", 6)
    s = _sched([p, h1, h2])
    gate = d_park_runtime.admission(s, _running())
    assert gate is not None and gate.barrier
    assert gate.skip(p, admitted=[]) == "pdflip_d_park_decode_first"
    assert gate.skip(h1, admitted=[]) is None
    assert gate.skip(h2, admitted=[h1.rid]) is None
    # the resume still holds its seat against a newcomer that is not of this wake
    late = _req("pdflip-9-99", 9, cohort=None)
    assert gate.skip(late, admitted=[h1.rid, h2.rid]) == "pdflip_d_park_first"


def test_the_resume_follows_the_first_decode_round():
    p = _req("pdflip-1-19", 1, parked=True)
    h1, h2 = _req("pdflip-2-20", 5), _req("pdflip-2-21", 6)
    s = _sched([p])                                   # h1, h2 extended in pass 0
    gate = d_park_runtime.admission(s, _running(h1, h2))
    assert gate.skip(p, admitted=[]) == "pdflip_d_park_decode_first"   # their decode first
    _decode_round(s)
    gate = d_park_runtime.admission(s, _running(h1, h2))
    assert gate.skip(p, admitted=[]) is None                          # then the resume


def test_a_settle_read_of_the_wake_keeps_the_resume_behind_its_member():
    p = _req("pdflip-1-19", 1, parked=True)
    h1 = _req("pdflip-2-20", 5)
    reading = _req("pdflip-2-22", 7)
    s = _sched([p], settle=[reading])
    _decode_round(s)
    gate = d_park_runtime.admission(s, _running(h1))
    assert gate.skip(p, admitted=[]) == "pdflip_d_park_decode_first"
    for _ in range(40):                                # bounded: the round cap closes it
        _decode_round(s)
    gate = d_park_runtime.admission(s, _running(h1))
    assert gate.skip(p, admitted=[]) is None


def test_a_short_tail_rides_the_first_pass():
    p = _req("pdflip-1-19", 1, parked=True, tokens=130, resumable=124)   # tail 6 <= 8
    h1 = _req("pdflip-2-20", 5)
    gate = d_park_runtime.admission(_sched([p, h1]), _running())
    assert gate.skip(p, admitted=[]) is None
    assert gate.skip(h1, admitted=[]) == "pdflip_d_park_first"


def test_nothing_else_to_run_never_idles_for_the_resume():
    p = _req("pdflip-1-19", 1, parked=True)
    gate = d_park_runtime.admission(_sched([p]), _running())
    assert gate.skip(p, admitted=[]) is None


def test_switch_off_is_the_park_first_gate(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_ENABLE_D_DECODE_FIRST", "0")
    p = _req("pdflip-1-19", 1, parked=True)
    h1 = _req("pdflip-2-20", 5)
    gate = d_park_runtime.admission(_sched([p, h1]), _running())
    assert gate.skip(p, admitted=[]) is None
    assert gate.skip(h1, admitted=[]) == "pdflip_d_park_first"
    assert not getattr(gate, "deferred", frozenset())
