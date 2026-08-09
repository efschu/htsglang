"""#631: pins for the per-rank output_ids trace across a flip cutover.

WHAT THE INSTRUMENT IS FOR. A request crossing ``pp_to_tp`` under load
loses exactly one output token and one crossing ``tp_to_pp`` gains a
duplicate. Every transport suspect is measured out (HANDOFF_656 section
8), and the emit code shows why the append site is what is left: each
emit hands the client ``output_ids[off : n]`` and sets the cursor to
``n``, so the client's array IS the concatenation of consecutive
half-open slices of this rank's own list.

These pins cover the three properties the instrument must have to be
worth a boot:

1. it costs the default path nothing (no request enumeration outside a
   flip window),
2. its per-pass lines carry the DELTA, because absolute lengths cannot
   show which pass skipped,
3. its emit-continuity check can actually FAIL -- a gap and an overlap
   are both reported, and they are the drop and duplicate faces.
"""

import pytest

from sglang.srt.managers.phase_flip_output_trace import (
    EmitContinuity,
    OutputTrace,
    snapshot_rows,
    trace_cutover,
    trace_emit,
    trace_tick,
)


class _Req:
    def __init__(self, rid, output_ids, send_token_offset=0):
        self.rid = rid
        self.output_ids = list(output_ids)
        self.send_token_offset = send_token_offset


class _Runtime:
    def __init__(self, pending=None, epoch=7):
        self.pending = pending
        self.epoch = epoch


class _Scheduler:
    def __init__(self, pending=None, reqs=()):
        self.phase_flip_runtime = _Runtime(pending)
        self._reqs = list(reqs)


@pytest.fixture(autouse=True)
def _enable_trace(monkeypatch):
    monkeypatch.setenv("SGLANG_PHASE_FLIP_OUTPUT_TRACE", "1")


@pytest.fixture
def patched_live(monkeypatch):
    """Route the module's resident-request lookup at the fake scheduler."""
    import sglang.srt.managers.phase_flip_output_trace as mod

    monkeypatch.setattr(mod, "_resident_reqs", lambda sched: sched._reqs)
    return mod


# --------------------------------------------------------------------
# 1. Snapshot shape
# --------------------------------------------------------------------


def test_snapshot_rows_carry_rid_length_offset_and_tail():
    rows = snapshot_rows([_Req("2ede3499aaaa", [11, 12, 13, 14], 3)])
    assert rows == [("2ede3499", 4, 3, (12, 13, 14))]


def test_snapshot_rows_survive_a_request_with_no_output_yet():
    assert snapshot_rows([_Req("abc", [], 0)]) == [("abc", 0, 0, ())]


# --------------------------------------------------------------------
# 2. Cost on the default path
# --------------------------------------------------------------------


def test_tick_does_not_enumerate_requests_while_no_flip_is_pending(monkeypatch):
    """THE COST PIN. Outside an armed window and outside the post-cutover
    countdown the tick must not touch the resident set at all -- it runs
    on the round hook of a serving instance, once per pass, for ever."""
    import sglang.srt.managers.phase_flip_output_trace as mod

    def _boom(_sched):
        raise AssertionError("the resident set was enumerated on an idle pass")

    monkeypatch.setattr(mod, "_resident_reqs", _boom)
    trace_tick(_Scheduler(pending=None), "tp_top")


def test_tick_is_a_no_op_when_the_env_switch_is_off(monkeypatch):
    monkeypatch.setenv("SGLANG_PHASE_FLIP_OUTPUT_TRACE", "0")
    import sglang.srt.managers.phase_flip_output_trace as mod

    def _boom(_sched):
        raise AssertionError("tracing ran with the switch off")

    monkeypatch.setattr(mod, "_resident_reqs", _boom)
    sched = _Scheduler(pending="pp_to_tp", reqs=[_Req("a", [1])])
    trace_tick(sched, "pp_end")
    trace_cutover(sched, "pp_to_tp")


def test_tick_records_while_a_flip_is_pending(patched_live):
    sched = _Scheduler(pending="pp_to_tp", reqs=[_Req("a", [1, 2])])
    trace_tick(sched, "pp_end")
    trace = getattr(sched, "_phase_flip_output_trace")
    assert len(trace.ring) == 1
    assert trace.ring[0][0] == "pp_end"


# --------------------------------------------------------------------
# 3. The ring, the dump and the countdown
# --------------------------------------------------------------------


def test_ring_keeps_only_the_passes_adjacent_to_the_cutover(patched_live):
    trace = OutputTrace(pre=3, post=2)
    for i in range(6):
        trace.observe("pp_end", [("a", i, i, (i,))])
    assert len(trace.ring) == 3
    assert [rows[0][1] for _site, rows in trace.ring] == [3, 4, 5]


def test_cutover_dumps_the_ring_and_arms_the_countdown():
    trace = OutputTrace(pre=3, post=2)
    trace.observe("pp_end", [("a", 10, 10, (5,))])
    trace.observe("pp_end", [("a", 11, 11, (6,))])
    lines = trace.cutover("pp_to_tp", 4)
    assert len(lines) == 2
    assert "pre[-2]" in lines[0] and "pre[-1]" in lines[1]
    assert "n=11 (+1)" in lines[1]
    assert trace.armed_after
    assert not trace.ring


def test_countdown_reports_exactly_post_passes_then_stops():
    trace = OutputTrace(pre=2, post=3)
    trace.cutover("tp_to_pp", 1)
    seen = [trace.after("tp_top", [("a", n, n, (n,))]) for n in range(5)]
    assert sum(line is not None for line in seen) == 3
    assert seen[3] is None and seen[4] is None
    assert not trace.armed_after


def test_post_lines_carry_the_delta_not_only_the_length():
    """THE DELTA IS THE MEASUREMENT. Three ranks' absolute lengths can
    differ legitimately (they sample at different points of the pass);
    a pass that appends one token on two ranks and none on the third is
    the defect, and only the delta shows it."""
    trace = OutputTrace(pre=2, post=3)
    trace.observe("pp_end", [("a", 80, 80, (17,))])
    trace.cutover("pp_to_tp", 2)
    line = trace.after("tp_top", [("a", 82, 81, (17, 220))])
    assert "n=82 (+2)" in line
    assert "dir=pp_to_tp" in line and "ep=2" in line


def test_a_request_first_seen_after_the_cutover_is_marked_new():
    trace = OutputTrace(pre=2, post=2)
    trace.observe("pp_end", [("a", 5, 5, (1,))])
    trace.cutover("pp_to_tp", 0)
    line = trace.after("tp_top", [("b", 1, 0, (9,))])
    assert "b n=1 (new)" in line


# --------------------------------------------------------------------
# 4. The emit-continuity detector, and the proof that it can fail
# --------------------------------------------------------------------


def test_consecutive_emits_are_silent():
    cont = EmitContinuity()
    assert cont.observe("a", 0, 3) is None
    assert cont.observe("a", 3, 1) is None
    assert cont.observe("a", 4, 2) is None


def test_can_fail_a_gap_is_reported_and_is_the_drop_face():
    cont = EmitContinuity()
    assert cont.observe("a", 0, 3) is None
    complaint = cont.observe("a", 4, 1)
    assert complaint is not None
    assert "GAP of 1 token" in complaint


def test_can_fail_an_overlap_is_reported_and_is_the_duplicate_face():
    cont = EmitContinuity()
    assert cont.observe("a", 0, 4) is None
    complaint = cont.observe("a", 3, 2)
    assert complaint is not None
    assert "OVERLAP of 1 token" in complaint


def test_continuity_is_tracked_per_request():
    cont = EmitContinuity()
    assert cont.observe("a", 0, 3) is None
    assert cont.observe("b", 0, 5) is None
    assert cont.observe("a", 3, 1) is None
    assert cont.observe("b", 5, 1) is None


def test_emit_hook_is_inert_outside_the_post_cutover_window(patched_live, caplog):
    """The emit hook sits in the output streamer, on every decode pass of
    every request. Outside the window it must not log a line."""
    import sglang.srt.managers.phase_flip_output_trace as mod

    mod._ACTIVE_TRACE = None
    with caplog.at_level("INFO"):
        trace_emit("abcdef12", 0, 1, 1)
    assert "OUTTRACE" not in caplog.text


def test_emit_hook_logs_and_complains_inside_the_window(patched_live, caplog):
    sched = _Scheduler(pending="pp_to_tp", reqs=[_Req("abcdef12", [1, 2, 3])])
    trace_tick(sched, "pp_end")
    trace_cutover(sched, "pp_to_tp")
    with caplog.at_level("INFO"):
        trace_emit("abcdef12", 0, 2, 3)
        trace_emit("abcdef12", 3, 1, 4)
    assert "emit rid=abcdef12 off=0->2 sent=2" in caplog.text
    assert "EMIT DISCONTINUITY" in caplog.text
    assert "GAP of 1 token" in caplog.text


def test_cutover_logs_the_at_cutover_line(patched_live, caplog):
    sched = _Scheduler(pending="tp_to_pp", reqs=[_Req("abcdef12", [1, 2, 3], 3)])
    trace_tick(sched, "tp_top")
    with caplog.at_level("INFO"):
        trace_cutover(sched, "tp_to_pp")
    assert "at-cutover dir=tp_to_pp ep=7" in caplog.text
    assert "abcdef12 n=3" in caplog.text
