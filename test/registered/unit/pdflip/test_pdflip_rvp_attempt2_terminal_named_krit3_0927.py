# SPDX-License-Identifier: Apache-2.0
"""KRIT3 (NF rc12s dkrnfh91dprbar1dauer09271719, pdflip-3-16 / pdflip-3-18, 17:33-17:35).

pdflip-3-16 was refused mid-stream at 17:33:33 (n=1) and again at 17:33:36 (n=2) in the SAME D phase --
the front dropped the second needs-p, one path per rid -- then, after its one P leg, twice at 17:35:47
(n=3 kept, n=4 over the bound of 3): in-band W50 after the first byte, "re-route impossible",
PDFLIP-SERVED status=200 completion_tokens=0.

(1) D counts RESUME-VIA-P attempts per P LEG (per D wake), bound 2: the first P leg and exactly one more.
(2) A committed stream that ends without an answer ends NAMED (LEG2-TERMINAL-NAMED; an error event in the
    wire of the path, synthesized when D's stream did not carry one).
"""

import json
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.managers.scheduler_components import weight_updater as _wu  # noqa: E402,F401
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import resume_via_p as rvp  # noqa: E402


@pytest.fixture
def d_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv("FLLIPER_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.delenv("FLLIPER_PDFLIP_RVP_ATTEMPT_PER_WAKE", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_RESUME_VIA_P", raising=False)
    return tmp_path


def _req(rid="pdflip-3-16"):
    return types.SimpleNamespace(rid=rid, stream=True, multimodal_inputs=None, output_ids=[1, 2],
                                 origin_input_ids=[7] * 10, full_untruncated_fill_ids=None)


def _sched():
    return types.SimpleNamespace(ps=types.SimpleNamespace(tp_rank=0), pdflip_d_parked=[])


def _refuse(s, r, extent):
    """The scheduler's W31 answer: eligible -> keep_on_d, else the named abort."""
    if not rvp.eligible(r, sched=s):
        return "W50"
    rvp.keep_on_d(s, r, extent, 12288)
    return "held"


def _needs_p(d):
    with open(os.path.join(d, rvp.SUBDIR, "pdflip-3-16.json")) as f:
        return json.load(f)


def test_metal_sequence_gets_exactly_one_more_p_leg_then_ends_by_name(d_env, monkeypatch):
    s, r = _sched(), _req()
    rvp.note_wake(s)
    assert _refuse(s, r, 24065) == "held"                     # 17:33:33
    assert (getattr(r, rvp.ATTEMPTS_ATTR), _needs_p(d_env)["attempt"]) == (1, 1)
    assert _refuse(s, r, 24065) == "held"                     # 17:33:36, same phase: the same hold
    assert getattr(r, rvp.ATTEMPTS_ATTR) == 1
    rvp.note_wake(s)                                           # P leg ran, D woke
    assert _refuse(s, r, 14401) == "held"                     # 17:35:47: attempt=2
    assert (getattr(r, rvp.ATTEMPTS_ATTR), _needs_p(d_env)["attempt"]) == (2, 2)
    assert _refuse(s, r, 14401) == "held"                     # same pass again: still the same hold
    rvp.note_wake(s)                                           # the second P leg ran
    assert _refuse(s, r, 9000) == "W50"                       # a refusal after it ends by name


def test_switch_off_counts_every_refusal_bound_3(d_env, monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_RVP_ATTEMPT_PER_WAKE", "0")
    s, r = _sched(), _req()
    rvp.note_wake(s)
    assert [_refuse(s, r, 24065) for _ in range(4)] == ["held", "held", "held", "W50"]


def test_wake_counter_needs_no_scheduler():
    rvp.note_wake(None)
    s = _sched()
    rvp.note_wake(s)
    rvp.note_wake(s)
    assert getattr(s, rvp.SCHED_WAKE_ATTR) == 2


def test_the_wake_is_counted_in_the_resume_and_the_scheduler_passes_itself():
    wu = open(_wu.__file__).read()
    i = wu.index("    def resume_memory_occupation(self, recv_req")
    blk = wu[i:i + 2500]
    assert blk.index("return replay") < blk.index("note_wake(getattr(self, \"scheduler\", None))")
    from flliper.srt.managers import scheduler as S

    assert "_pdflip_rvp.eligible(req, sched=self)" in open(S.__file__).read()


A = "/v1/messages"
O = "/v1/chat/completions"
W50_TAIL = (b'event: message_start\ndata: {}\n\nevent: ping\ndata: {}\n\n'
            b'event: error\ndata: {"type":"error","error":{"type":"overloaded_error","message":'
            b'"W50 PdFlipTpPrefillExceeded: ... extent after prefix matching is 14401"}}\n\n')


def test_terminal_reason():
    assert F.leg2_terminal_reason(W50_TAIL, A, finished=False) == "W50"
    assert F.leg2_terminal_reason(W50_TAIL, A, finished=True) == "W50"
    assert F.leg2_terminal_reason(b"event: content_block_delta\n", A, finished=False) == "truncated"
    assert F.leg2_terminal_reason(b"event: message_stop\n", A, finished=True) is None
    assert F.leg2_terminal_reason(b'data: {"text":"x"}\n', "/generate", finished=False) is None
    assert F.leg2_terminal_reason(b"", O, finished=False) is None


def test_error_event_detection_and_chunks():
    assert F.stream_error_event_in(W50_TAIL, A)
    assert not F.stream_error_event_in(b"event: content_block_delta\n", A)
    a = F.named_error_chunk(A, "W50 x")
    assert a.startswith(b"event: error\ndata: ") and F.stream_error_event_in(a, A)
    assert json.loads(a.split(b"data: ", 1)[1])["error"]["message"] == "W50 x"
    o = F.named_error_chunk(O, "W164 y")
    assert o.startswith(b"data: ") and F.stream_error_event_in(o, O)
    assert json.loads(o[6:])["error"]["code"] == 503


def test_switch():
    assert F._terminal_named_enabled({})
    assert not F._terminal_named_enabled({"FLLIPER_PDFLIP_LEG2_TERMINAL_NAMED": "0"})


def test_wiring_before_write_eof():
    src = open(F.__file__).read()
    i = src.index('_tn = leg2_terminal_reason(bytes(tail), request.path, client_io["finished"])')
    blk = src[i - 400:i + 2200]
    assert 'if not client_io["gone"] and _terminal_named_enabled():' in blk
    assert "PDFLIP LEG2-TERMINAL-NAMED rid=%s reason=%s" in blk
    assert blk.index("await _push(named_error_chunk(") < blk.index("await resp.write_eof()")
    assert "RESUME-VIA-P attempt=%s rid=%s" in src


# NF rc12t (dkrnfh91dprbar1dauer09271756) pdflip-4-16: PARK-RUNNING 18:07:42 -> PARK-RESUME 18:08:25 ->
# W50-REROUTE midstream 18:08:32 (d_extent 82045, D holds, no bytes) -> 18:08:55 D-TP0 "W88
# PdFlipStoreLoadNotProgressing arm=host_pool_shortfall span=82044 ... terminal, answered 503" -> the front
# booked verdict=serve 0/0 and RESUME-VIA-P dropped the queued P leg at the stream end.
W88_MSG = ("W88 PdFlipStoreLoadNotProgressing rid=pdflip-4-16 arm=host_pool_shortfall span=82044 site=retry "
           "no_progress_passes=122 bound_passes=64 -- terminal, answered 503")
W88_ANTH = (b'event: message_start\ndata: {"type":"message_start"}\n\n'
            b'event: ping\ndata: {"type":"ping"}\n\n' * 3
            + b'event: error\ndata: ' + json.dumps({"type": "error", "error": {
                "type": "overloaded_error", "message": W88_MSG}}).encode() + b'\n\n')
W88_OAI = (b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
           b'data: ' + json.dumps({"error": {"message": W88_MSG, "code": 503}}).encode() + b'\n\n')


def test_rc12t_pdflip_4_16_w88_after_the_commit_ends_named():
    assert F.leg2_terminal_reason(W88_ANTH, A, finished=False) == "error:W88"
    # even when the adapter closed the envelope after the error: named, never "serve"
    assert F.leg2_terminal_reason(W88_ANTH + b"event: message_stop\ndata: {}\n\n", A, finished=True) == "error:W88"
    assert F.leg2_terminal_reason(W88_OAI, O, finished=False) == "error:W88"
    assert F.leg2_terminal_reason(W88_OAI + b'data: {"choices":[{"finish_reason":"abort"}]}\n\n', O,
                                  finished=True) == "error:W88"
    # D's error event is already in the stream: forwarded, nothing synthesized
    assert F.stream_error_event_in(W88_ANTH, A) and F.stream_error_event_in(W88_OAI, O)


def test_any_error_event_without_a_w_code_is_still_terminal():
    tail = b'event: message_start\ndata: {}\n\nevent: error\ndata: {"type":"error","error":{"message":"boom"}}\n\n'
    assert F.leg2_terminal_reason(tail, A, finished=True) == "error"


def test_held_stream_closed_silently_when_rvp_drops_its_p_leg_is_named():
    # RESUME-VIA-P dropped its P leg and D closed the stream without an error event or end marker
    silent = b'event: message_start\ndata: {}\n\n' + b'event: ping\ndata: {}\n\n' * 5
    assert F.leg2_terminal_reason(silent, A, finished=False) == "truncated"
    assert F.leg2_terminal_reason(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n', O,
                                  finished=False) == "truncated"


def test_a_complete_answer_is_not_touched():
    ok = (b'event: message_start\ndata: {}\n\nevent: content_block_delta\ndata: {"delta":{"text":'
          b'"the word \\"error\\" in prose"}}\n\nevent: message_stop\ndata: {}\n\n')
    assert F.leg2_terminal_reason(ok, A, finished=True) is None
    oai = (b'data: {"choices":[{"delta":{"content":"an \\"error\\" word"}}]}\n\n'
           b'data: {"choices":[{"finish_reason":"stop"}]}\n\n')
    assert F.leg2_terminal_reason(oai, O, finished=True) is None
