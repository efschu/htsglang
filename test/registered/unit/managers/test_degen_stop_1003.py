"""DEGEN-STOP (managers/degen_stop.py): stage 2 of the decode-tail repetition
watch ends a looping request with finish_reason=length.

Metal 03.10. (27B dual y8w, D): rid weg2-0-58 looped one 210-token reasoning
pattern for 64000 tokens; the D seat and its KV pressed P into a pause
cascade. Stage 1 named it ("DEGEN-SUSPECT ... stop=off") and nothing ended it.

Pinned, without a GPU: the switch is OFF by default (the user's call) and then
only logs; armed, the detokenizer sends ONE AbortReq(degen_stop, length) to
the tokenizer manager, the tokenizer manager forwards it to the scheduler
instead of finalizing the stream as an abort echo, and the scheduler finishes
the running request with FINISH_LENGTH (every other abort stays FINISH_ABORT).
"""

import logging
import random
import types

from sglang.srt.managers import degen_detect as dd

THINK_END = 151668


def _rand(n, seed=1, vocab=50000):
    r = random.Random(seed)
    return [r.randrange(1000, vocab) for _ in range(n)]


def _detok(stop: bool, monkeypatch):
    """A DetokenizerManager with only what init_running_status's DEGEN part reads."""
    from sglang.srt.managers import detokenizer_manager as dm

    monkeypatch.setenv("SGLANG_WEG2_DEGEN_STOP", "1" if stop else "0")
    sent = []
    monkeypatch.setattr(dm, "sock_send", lambda sock, obj: sent.append(obj))
    mgr = dm.DetokenizerManager.__new__(dm.DetokenizerManager)
    mgr.send_to_tokenizer = object()
    mgr.tokenizer = None
    srv = types.SimpleNamespace(
        disable_tokenizer_batch_decode=False, tool_call_parser=None,
        soft_watchdog_timeout=None, enable_metrics=False,
    )
    dm.DetokenizerManager.init_running_status(mgr, srv)
    mgr.degen.think_end_id = THINK_END
    return dm, mgr, sent


def _feed(dm, mgr, rid, pattern, rounds=200):
    recv = types.SimpleNamespace(rids=[rid], decode_ids=[pattern], read_offsets=[0],
                                 finished_reasons=[None])
    for _ in range(rounds):
        recv.decode_ids = [pattern]
        dm.DetokenizerManager._observe_degen(mgr, recv)


def test_off_by_default_logs_only(monkeypatch, caplog):
    from sglang.srt.environ import envs

    monkeypatch.delenv("SGLANG_WEG2_DEGEN_STOP", raising=False)
    assert envs.SGLANG_WEG2_DEGEN_STOP.get() is False
    dm, mgr, sent = _detok(False, monkeypatch)
    with caplog.at_level(logging.WARNING):
        _feed(dm, mgr, "r-off", _rand(7, seed=3))
    assert any("DEGEN-SUSPECT rid=r-off" in r.getMessage() for r in caplog.records)
    assert not any("DEGEN-STOP" in r.getMessage() for r in caplog.records)
    assert sent == []


def test_armed_detokenizer_sends_one_length_abort(monkeypatch, caplog):
    from sglang.srt.managers.io_struct import AbortReq

    dm, mgr, sent = _detok(True, monkeypatch)
    with caplog.at_level(logging.WARNING):
        _feed(dm, mgr, "r-loop", _rand(7, seed=4), rounds=400)
    assert len(sent) == 1, sent
    req = sent[0]
    assert isinstance(req, AbortReq) and req.rid == "r-loop" and req.degen_stop is True
    assert req.finished_reason["type"] == "length" and req.finished_reason["length"] > 0
    stop_lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("DEGEN-STOP rid=r-loop")]
    assert stop_lines and "part=reasoning" in stop_lines[0] and "period=7" in stop_lines[0]
    assert "out_len=" in stop_lines[0]


def test_tokenizer_manager_forwards_instead_of_finalizing():
    from sglang.srt.managers.degen_stop import degen_stop_abort_req
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

    dispatched = []
    state = types.SimpleNamespace(finished=False)
    tm = types.SimpleNamespace(
        rid_to_state={"r-loop": state},
        _dispatch_to_scheduler=lambda obj: dispatched.append(obj),
    )
    req = degen_stop_abort_req("r-loop", "reasoning", 7, 9, 3072)
    TokenizerManager._handle_abort_req(tm, req)
    assert dispatched == [req]
    assert state.finished is False  # the scheduler's length output ends the stream
    # a request that already finished is not forwarded
    dispatched.clear()
    TokenizerManager._handle_abort_req(tm, degen_stop_abort_req("gone", "content", 7, 9, 10))
    assert dispatched == []


def test_scheduler_finishes_running_request_with_length():
    from sglang.srt.managers.degen_stop import degen_stop_abort_req, running_abort_finish
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_LENGTH

    fin = running_abort_finish(degen_stop_abort_req("r", "reasoning", 7, 9, 3072))
    assert isinstance(fin, FINISH_LENGTH) and fin.to_json() == {"type": "length", "length": 3072}
    assert isinstance(running_abort_finish(AbortReq(rid="r")), FINISH_ABORT)
    # an origin-injected abort with a finish reason but no degen_stop stays abort
    assert isinstance(running_abort_finish(AbortReq(rid="r", finished_reason={"type": "length", "length": 5})),
                      FINISH_ABORT)


def test_scheduler_running_abort_uses_the_hook():
    import inspect

    from sglang.srt.managers import scheduler

    src = inspect.getsource(scheduler.Scheduler._abort_request_now)
    assert "req.to_finish = running_abort_finish(recv_req)" in src
    # the detector itself still never enters the scheduler
    assert "degen_detect" not in inspect.getsource(scheduler)
