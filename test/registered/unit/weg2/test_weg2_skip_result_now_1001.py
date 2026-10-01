"""nf-pd-post 01.10. (boot y6o): P's token -- the flip's first streamed token --
waited for the pass AFTER the skip-extend batch.

The overlap loop processes a batch's result after the NEXT batch's launch. A
skip-extend batch (H24 E2) runs no target forward, so that deferral overlapped
nothing and only held P's token behind the next pass:

* 21:37:57 (typical): TP0 n=0 schedule 534 + run 76 ms, then ``gap_ms=188``
  -- the next pass's TP recv broadcast waited for TP1 (its load-back issue,
  START-LOADING kv_issue 280 ms) -- then n=1 run 10 ms, then "Prefill rank
  batch" (the skip's result) and LEG2-FIRST-CONTENT 801 ms after the wake RPC.
* 21:37:36 (first wake of the boot): n=1 decode run 531 ms (cold, TP1 late);
  the skip's result came after it, first content 1426 ms after the wake RPC.

The loop below is the REAL ``Scheduler.event_loop_overlap`` on a stub
scheduler; it records the order of recv / run / process.
"""
import os
import types
from collections import deque

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(__file__)


def _mode(name):
    return types.SimpleNamespace(
        name=name, is_extend=lambda: name == "EXTEND", is_decode=lambda: name == "DECODE")


class _Batch:
    def __init__(self, tag, mode, skip=False):
        self.tag, self.forward_mode, self.weg2_skip_extend = tag, _mode(mode), skip
        self.reqs = [object(), object()]

    def copy(self):
        return self


def _drive(monkeypatch, script):
    from sglang.srt.managers import scheduler as sch

    events = []
    for name in ("_h58_span", "_stage_sync", "_weg2_resume_first_token_note",
                 "_weg2_wake_cohort_note", "_weg2_decode_first_note"):
        monkeypatch.setattr(sch, name, lambda *a, **k: None)
    monkeypatch.setattr(sch.index_race_guard, "poll", lambda *a, **k: None)
    plans = deque(script)
    s = types.SimpleNamespace(gracefully_exit=False, _engine_paused=False, idle_sleeper=None,
                              enable_unified_memory=False, _pending_spill_batch=None,
                              is_generation=True, last_batch=None, running_batch=None,
                              cur_batch_for_debug=None)
    s.request_receiver = types.SimpleNamespace(recv_requests=lambda: events.append("recv") or [])
    s.process_input_requests = lambda reqs: None
    s._dual_group_lane_tick = lambda: None
    s._apply_war_barrier = lambda: None

    def next_batch(running_batch=None, last_batch=None):
        if not plans:
            s.gracefully_exit = True
            return types.SimpleNamespace(running_batch=running_batch, batch_to_run=None)
        return types.SimpleNamespace(running_batch=running_batch, batch_to_run=plans.popleft())

    s.get_next_batch_to_run = next_batch
    s.is_disable_overlap_for_batch = lambda batch, last_batch=None: False
    s.run_batch = lambda b: events.append(f"run:{b.tag}") or types.SimpleNamespace(
        delay_sample_func=None, tag=b.tag)
    s.process_batch_result = lambda b, r: events.append(f"process:{b.tag}")
    s._weg2_post_wake_pass_log = lambda b: None
    s.launch_batch_sample_if_needed = lambda r, b: None
    s.on_idle = lambda: events.append("idle")
    with envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(0):
        sch.Scheduler.event_loop_overlap(s)
    return events


def test_y6o_p_token_is_processed_before_the_next_pass(monkeypatch):
    ev = _drive(monkeypatch, [_Batch("S", "EXTEND", skip=True), _Batch("D1", "DECODE"),
                              _Batch("D2", "DECODE")])
    # base: ['recv', 'run:S', 'recv', 'run:D1', 'process:S', ...] -- P's token
    # behind the next pass's recv broadcast and launch
    assert ev[:5] == ["recv", "run:S", "process:S", "recv", "run:D1"], ev
    # every result exactly once, in launch order
    assert [e for e in ev if e.startswith("process:")] == ["process:S", "process:D1", "process:D2"]


def test_decode_rounds_keep_their_overlap(monkeypatch):
    ev = _drive(monkeypatch, [_Batch("S", "EXTEND", skip=True), _Batch("D1", "DECODE"),
                              _Batch("D2", "DECODE")])
    # D1's result is still processed after D2's launch (the decode overlap)
    assert ev.index("process:D1") > ev.index("run:D2"), ev


def test_a_real_extend_keeps_the_deferred_order(monkeypatch):
    ev = _drive(monkeypatch, [_Batch("E", "EXTEND"), _Batch("D1", "DECODE")])
    assert ev.index("process:E") > ev.index("run:D1"), ev


def test_switch_off_restores_the_deferred_order(monkeypatch):
    with envs.SGLANG_WEG2_ENABLE_SKIP_RESULT_NOW.override(False):
        ev = _drive(monkeypatch, [_Batch("S", "EXTEND", skip=True), _Batch("D1", "DECODE")])
    assert ev.index("process:S") > ev.index("run:D1"), ev


def test_a_skip_after_a_pending_decode_keeps_fifo(monkeypatch):
    """The skip's result is processed AFTER the earlier batch's -- never ahead."""
    ev = _drive(monkeypatch, [_Batch("D0", "DECODE"), _Batch("S", "EXTEND", skip=True),
                              _Batch("D1", "DECODE")])
    p = [e for e in ev if e.startswith("process:")]
    assert p == ["process:D0", "process:S", "process:D1"], ev
    assert ev.index("process:S") < ev.index("run:D1"), ev


def test_result_now_reads_only_rank_uniform_inputs():
    from sglang.srt.weg2 import skip_first as sf

    res = types.SimpleNamespace(delay_sample_func=None)
    assert sf.result_now(_Batch("S", "EXTEND", skip=True), res) is True
    assert sf.result_now(_Batch("E", "EXTEND"), res) is False
    assert sf.result_now(_Batch("D", "DECODE", skip=True), res) is False
    assert sf.result_now(None, res) is False
    assert sf.result_now(_Batch("S", "EXTEND", skip=True), None) is False
    delayed = types.SimpleNamespace(delay_sample_func=lambda: None)
    assert sf.result_now(_Batch("S", "EXTEND", skip=True), delayed) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
