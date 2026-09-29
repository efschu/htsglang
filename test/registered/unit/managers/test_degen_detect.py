"""DEGEN-SUSPECT (managers/degen_detect.py): the decode-tail repetition
instrument, stage 1 = log only.

27B boot ...dkr27breleasedraftbar1w109281340 (D, 13:44-13:51): rid pdflip-1-13
generated for minutes at ~190 tok/s and held the D->P drain. That boot's logs
carry NO output ids or text of pdflip-1-13 (only '#1469 RETAIN ...
token_ids_len=43548' and the front's route/refill lines), so the positive case
here is synthetic: a reasoning loop of one 37-token pattern.

Pinned: a real loop is named once per rid and period with part=reasoning or
content; legitimate structure (markdown table, JSON array, numbered list,
repeated code lines with changing names) stays under the threshold; the
detector lives in the detokenizer only -- nothing in the scheduler's decode
round -- and its cost per token is measured here; stage 2 is OFF.
"""

import logging
import random
import time
import types

import pytest

from flliper.srt.managers import degen_detect as dd

THINK_END = 151668


def _stream(det, rid, ids, batch=4, prompt_tail=(), finished_at_end=False):
    hits = []
    for k in range(0, len(ids), batch):
        chunk = ids[k:k + batch]
        last = finished_at_end and k + batch >= len(ids)
        if k == 0:
            h = det.observe_chunk(rid, list(prompt_tail) + chunk, len(prompt_tail), finished=last)
        else:
            h = det.observe_chunk(rid, chunk, 0, finished=last)
        if h:
            hits.append(h)
    return hits


def _lines(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("DEGEN-SUSPECT")]


def _rand(n, seed=1, vocab=50000):
    rng = random.Random(seed)
    return [rng.randrange(1000, vocab) for _ in range(n)]


def test_a_reasoning_loop_is_named_once_with_its_period(caplog):
    det = dd.DegenDetector(think_end_id=THINK_END)
    pattern = _rand(37, seed=7)
    ids = _rand(300) + pattern * 60 + pattern[:11]        # 2531 ids
    with caplog.at_level(logging.WARNING, logger=dd.logger.name):
        hits = _stream(det, "pdflip-1-13", ids)
        hits += _stream(det, "pdflip-1-13", pattern * 80)   # it keeps looping
    lines = _lines(caplog)
    assert len(lines) == 1, lines                         # once per rid and period
    assert "rid=pdflip-1-13 part=reasoning period=37 " in lines[0], lines[0]
    reps = int(lines[0].split("reps=")[1].split()[0])
    assert reps >= dd.MIN_REPS and "stop=off" in lines[0] and "tok_s=" in lines[0]
    assert hits and hits[0][0] == 37


def test_a_content_loop_after_think_end_says_content(caplog):
    det = dd.DegenDetector(think_end_id=THINK_END)
    ids = _rand(500, seed=3) + [THINK_END] + _rand(9, seed=4) * 120
    with caplog.at_level(logging.WARNING, logger=dd.logger.name):
        _stream(det, "r-content", ids)
    lines = _lines(caplog)
    assert len(lines) == 1 and "part=content period=9 " in lines[0], lines


def test_an_empty_think_in_the_prompt_tail_starts_in_content(caplog):
    det = dd.DegenDetector(think_end_id=THINK_END)
    with caplog.at_level(logging.WARNING, logger=dd.logger.name):
        _stream(det, "r-nothink", _rand(5, seed=5) * 300, prompt_tail=[11, THINK_END, 12, 13, 14])
    assert "part=content" in _lines(caplog)[0]


def test_a_short_period_run_below_the_span_does_not_hit(caplog):
    # a 200-element JSON array of zeros: "0" "," x200 = 400 ids of period 2
    det = dd.DegenDetector(think_end_id=THINK_END)
    ids = _rand(600, seed=8) + [15, 11] * 200 + _rand(1200, seed=9)
    with caplog.at_level(logging.WARNING, logger=dd.logger.name):
        _stream(det, "r-json", ids)
    assert _lines(caplog) == []


@pytest.mark.parametrize("name", ["table", "list", "code"])
def test_legitimate_structure_does_not_hit(caplog, name):
    rng = random.Random(11)
    ids = []
    for row in range(300):
        if name == "table":       # | cell | cell | cell |  with changing cells
            ids += [91, rng.randrange(2000, 9000), 91, rng.randrange(2000, 9000), 91, 198]
        elif name == "list":      # "- item N: <words>"
            ids += [12, 482, 1000 + row, 25] + _rand(rng.randrange(3, 9), seed=row)
        else:                     # x_N = foo(x_{N-1}, N)
            ids += [87, 1000 + row, 284, 7061, 7, 87, 999 + row, 11, 2000 + row, 8, 198]
    det = dd.DegenDetector(think_end_id=THINK_END)
    with caplog.at_level(logging.WARNING, logger=dd.logger.name):
        _stream(det, "r-" + name, ids)
    assert _lines(caplog) == [], name


def test_finished_request_drops_its_state_and_states_are_bounded():
    det = dd.DegenDetector()
    _stream(det, "r-done", _rand(100), finished_at_end=True)
    assert "r-done" not in det.states
    for i in range(dd.MAX_STATES + 10):
        det.observe(f"r{i}", [1])
    assert len(det.states) == dd.MAX_STATES


def test_cost_per_token_is_small_and_off_the_decode_round():
    # random text (no hit) and a long loop (hit, then re-checked every 256)
    for label, ids in (("random", _rand(40000, seed=21)), ("loop", _rand(37, seed=22) * 1100)):
        det = dd.DegenDetector(think_end_id=THINK_END)
        t0 = time.perf_counter()
        _stream(det, "r-cost", ids, batch=1)
        us = (time.perf_counter() - t0) / len(ids) * 1e6
        print(f"DEGEN cost {label}: {us:.2f} us/token amortized "
              f"(window {dd.WINDOW}, check every {dd.CHECK_EVERY}, detokenizer process)")
        assert us < 50.0, (label, us)
    # the decode round never calls it: only the detokenizer does
    import inspect

    from flliper.srt.managers import detokenizer_manager, scheduler
    from flliper.srt.managers.scheduler_components import output_streamer

    assert "degen_detect" not in inspect.getsource(scheduler)
    assert "degen_detect" not in inspect.getsource(output_streamer)
    assert "self._observe_degen(recv_obj)" in inspect.getsource(detokenizer_manager)


def test_the_detokenizer_hook_feeds_ids_and_skips_the_prompt_surround(caplog):
    from flliper.srt.managers.detokenizer_manager import DetokenizerManager

    mgr = types.SimpleNamespace(degen=dd.DegenDetector(think_end_id=THINK_END))
    pattern = _rand(13, seed=31)
    first = [101, 102, 103, 104, 105] + pattern * 4       # 5 prompt-surround ids
    recv = types.SimpleNamespace(rids=["r-hook"], decode_ids=[first], read_offsets=[5],
                                 finished_reasons=[None])
    DetokenizerManager._observe_degen(mgr, recv)
    assert mgr.degen.states["r-hook"].out_len == len(pattern) * 4
    with caplog.at_level(logging.WARNING, logger=dd.logger.name):
        for _ in range(100):
            recv.decode_ids = [pattern]
            recv.read_offsets = [0]
            DetokenizerManager._observe_degen(mgr, recv)
    assert "period=13 " in _lines(caplog)[0]


def test_stage_two_is_off_by_default_and_only_a_hook():
    from flliper.srt.environ import envs

    assert envs.FLLIPER_PDFLIP_DEGEN_DETECT.get() is True
    assert envs.FLLIPER_PDFLIP_DEGEN_STOP.get() is False
    called = []
    det = dd.DegenDetector(stop=True, on_stop=lambda *a: called.append(a))
    _stream(det, "r-stop", _rand(7, seed=2) * 200)
    assert called and called[0][0] == "r-stop" and called[0][2] == 7
