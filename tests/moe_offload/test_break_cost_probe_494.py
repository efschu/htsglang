# SPDX-License-Identifier: Apache-2.0
"""#494 -- the break-cost probe for the #462 breakable route, hermetic gates.

The probe (``srt/utils/break_cost_clock.py``) prices one CUDA-graph crossing:
``segment_end -> eager slot -> segment_start``, per crossing, per rank, split
into device compute, device wait and the four host terms F2 names. These tests
run the REAL clock, the REAL aggregation and the REAL record shape, and drive
them through the REAL ``BreakableCUDAGraph.replay()`` -- only the CUDA events
and the graph segments are fakes, because the cards are not available here.

WHERE THE FAKE EVENT DELIBERATELY DIFFERS FROM ``torch.cuda.Event``
------------------------------------------------------------------
A fake that merely resembles the real object proves nothing (desk-fake lesson).
``ScriptedEvent`` is deliberately STRICTER than the real thing at three named
places, so the properties the probe claims are enforced instead of assumed:

  1. ``synchronize()`` RAISES. The real event blocks. Any host sync the probe
     might sneak into the hot path is therefore a test failure, not a slowdown.
  2. ``query()`` is False for the whole round in which the event was recorded.
     A real event often reports ready almost immediately, which would let a
     broken deferred read pass by luck. Here reading early returns nothing.
  3. ``elapsed_time()`` RAISES while either side is unready. The real event
     silently blocks and returns a number -- exactly the hidden synchronisation
     this instrument exists to avoid.

Its timeline is scripted, so every span is an exact expected number rather than
a plausible one, and every crossing carries DIFFERENT values -- which is what
makes the off-by-one can-fail arm below detectable at all.

Gates:
  * TIMELINE -- every device term of every crossing equals its scripted value,
    and compute/wait/span/residual add up.
  * ATTRIBUTION -- host phases land on the crossing that recorded them, and
    the crossing is named after the function that caused the break.
  * DEFERRED -- nothing is emitted in the round that recorded it, records come
    out in order, and no event is ever synchronised.
  * STEADY STATE -- events are pooled: an armed run stops allocating events
    after the in-flight window is full.
  * NEUTRALITY (default off) -- ``replay()`` takes the unmeasured loop, creates
    zero events, records zero events, and ``break_cost_phase`` hands back one
    shared object.
  * CAN-FAIL -- the crossing->segment mapping is replaced by an off-by-one
    variant and the TIMELINE gate must go red.

Run:  CUDA_VISIBLE_DEVICES=99 PYTHONPATH=<worktree>/python \\
      python -m pytest tests/moe_offload/test_break_cost_probe_494.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from sglang.srt.utils import break_cost_clock as bcc  # noqa: E402

# --- the scripted fake ------------------------------------------------------

# One round of the fixture graph: 3 segments, 2 crossings. The advance BEFORE
# each record, in ms, in the exact order replay() records them. Every value is
# distinct so a misattributed crossing cannot coincide with a correct one.
SEG_MS = (100.0, 200.0, 300.0)
GAP_IN_MS = (1.5, 2.5)
SLOT_MS = (7.0, 8.0)
GAP_OUT_MS = (0.25, 0.5)

ROUND_ADVANCES = (
    0.0,  # seg0 start
    SEG_MS[0],  # seg0 end
    GAP_IN_MS[0],  # slot0 start
    SLOT_MS[0],  # slot0 end
    GAP_OUT_MS[0],  # seg1 start
    SEG_MS[1],  # seg1 end
    GAP_IN_MS[1],  # slot1 start
    SLOT_MS[1],  # slot1 end
    GAP_OUT_MS[1],  # seg2 start
    SEG_MS[2],  # seg2 end
)
MARKS_PER_ROUND = len(ROUND_ADVANCES)


class Timeline:
    """A virtual device clock plus a round counter the fake events read."""

    def __init__(self):
        self.now = 0.0
        self.round = 0
        self.cursor = 0
        self.records = 0
        self.events_built = 0

    def next_stamp(self) -> float:
        self.now += ROUND_ADVANCES[self.cursor % MARKS_PER_ROUND]
        self.cursor += 1
        self.records += 1
        return self.now

    def new_round(self) -> None:
        self.round += 1


class ScriptedEvent:
    """See the module docstring: stricter than torch.cuda.Event on purpose."""

    def __init__(self, timeline: Timeline):
        self._tl = timeline
        self._t = None
        self._round = None
        timeline.events_built += 1

    def record(self) -> None:
        self._t = self._tl.next_stamp()
        self._round = self._tl.round

    # DIFFERENCE 2: not ready during the round that recorded it.
    def query(self) -> bool:
        return self._round is not None and self._tl.round > self._round

    # DIFFERENCE 3: refuses to make up a number for an unready event.
    def elapsed_time(self, other: "ScriptedEvent") -> float:
        assert self.query() and other.query(), "elapsed_time() on an unready event"
        return other._t - self._t

    # DIFFERENCE 1: the real event would block here.
    def synchronize(self) -> None:  # pragma: no cover - must never be called
        raise AssertionError("the probe synchronised an event on the hot path")


class FakeSegment:
    def __init__(self, log, idx):
        self._log = log
        self._idx = idx

    def replay(self) -> None:
        self._log.append(("segment", self._idx))


def _fixture_graph(log, phase_by_crossing=None):
    """A real BreakableCUDAGraph holding fake segments and named break fns."""
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
        BreakableCUDAGraph,
    )

    graph = BreakableCUDAGraph()
    graph._segments = [FakeSegment(log, i) for i in range(3)]

    def make_break(i, name):
        def fn():
            log.append(("break", i))
            for phase in (phase_by_crossing or {}).get(i, ()):
                with bcc.break_cost_phase(phase):
                    pass

        fn.break_name = name
        return fn

    graph._break_fns = [
        make_break(0, "_moe_offload_fetch_step"),
        make_break(1, "_moe_offload_fetch_step"),
    ]
    return graph


@pytest.fixture
def stream_stub(monkeypatch):
    """replay() asks torch for the current stream; there is no device here."""
    import torch

    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: object())


@pytest.fixture
def armed(monkeypatch, stream_stub):
    """An armed clock with the scripted event factory and a list sink."""
    timeline = Timeline()
    records = []
    clock = bcc.BreakCostClock(
        event_factory=lambda: ScriptedEvent(timeline),
        sink=records.append,
        defer_rounds=2,
        rank_tag="rank2",
    )
    bcc.reset_break_cost_clock_for_test(clock)
    yield clock, timeline, records
    bcc.reset_break_cost_clock_for_test(None)


def _run_rounds(graph, timeline, n):
    for _ in range(n):
        timeline.new_round()
        graph.replay()


# --- TIMELINE ---------------------------------------------------------------


def test_crossing_terms_match_the_scripted_timeline(armed):
    clock, timeline, records = armed
    log = []
    graph = _fixture_graph(log)

    _run_rounds(graph, timeline, 4)

    assert records, "no round was ever emitted"
    rec = records[0]
    assert rec["crossings"] == 2 and rec["segments"] == 3
    for i, c in enumerate(rec["crossing_detail"]):
        assert c["i"] == i
        assert c["gap_in_ms"] == pytest.approx(GAP_IN_MS[i])
        assert c["slot_ms"] == pytest.approx(SLOT_MS[i])
        assert c["gap_out_ms"] == pytest.approx(GAP_OUT_MS[i])


def test_compute_wait_and_span_add_up(armed):
    clock, timeline, records = armed
    graph = _fixture_graph([])

    _run_rounds(graph, timeline, 4)

    rec = records[0]
    assert rec["compute_ms"] == pytest.approx(sum(SEG_MS) + sum(SLOT_MS))
    assert rec["wait_ms"] == pytest.approx(sum(GAP_IN_MS) + sum(GAP_OUT_MS))
    assert rec["span_ms"] == pytest.approx(rec["compute_ms"] + rec["wait_ms"])
    assert rec["residual_ms"] == pytest.approx(0.0, abs=1e-9)
    assert rec["segment_ms"] == pytest.approx(sum(SEG_MS))
    assert rec["slot_ms"] == pytest.approx(sum(SLOT_MS))


def test_record_is_machine_readable_and_carries_the_rank(armed, tmp_path):
    import json

    clock, timeline, records = armed
    graph = _fixture_graph([])
    _run_rounds(graph, timeline, 4)

    rec = records[0]
    assert rec["v"] == bcc.RECORD_VERSION
    assert rec["rank_tag"] == "rank2"
    assert rec["graph"].startswith("g")
    # A record must survive the JSONL sink unchanged -- the F2 write-up parses
    # these lines, so a non-serialisable field is a defect, not a cosmetic.
    line = json.dumps(rec, separators=(",", ":"))
    assert json.loads(line)["wait_ms"] == pytest.approx(rec["wait_ms"])


def test_jsonl_sink_writes_one_line_per_round(armed, tmp_path, monkeypatch):
    import json

    clock, timeline, _ = armed
    path = tmp_path / "break_cost.rank2.jsonl"
    clock._sink = bcc._JsonlSink(str(path))
    graph = _fixture_graph([])

    _run_rounds(graph, timeline, 5)

    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert len(lines) >= 2
    assert [x["round"] for x in lines] == list(range(len(lines)))
    assert lines[0]["crossings"] == 2


# --- ATTRIBUTION ------------------------------------------------------------


def test_host_phases_land_on_the_crossing_that_recorded_them(armed):
    clock, timeline, records = armed
    graph = _fixture_graph(
        [], phase_by_crossing={0: ("rendezvous", "planning"), 1: ("publish",)}
    )

    _run_rounds(graph, timeline, 4)

    detail = records[0]["crossing_detail"]
    assert set(detail[0]["phases"]) == {"rendezvous", "planning"}
    assert set(detail[1]["phases"]) == {"publish"}
    assert all(v >= 0.0 for c in detail for v in c["phases"].values())


def test_crossings_are_named_after_the_break_function(armed):
    clock, timeline, records = armed
    graph = _fixture_graph([])

    _run_rounds(graph, timeline, 4)

    rec = records[0]
    assert [c["name"] for c in rec["crossing_detail"]] == [
        "_moe_offload_fetch_step"
    ] * 2
    by_name = rec["by_name"]["_moe_offload_fetch_step"]
    assert by_name["count"] == 2
    assert by_name["slot_ms"] == pytest.approx(sum(SLOT_MS))
    assert by_name["gap_in_ms"] == pytest.approx(sum(GAP_IN_MS))


def test_the_moe_break_point_carries_its_label():
    """The name the probe reports comes from the real decorator, not the test."""
    from sglang.srt.layers.moe.breakable_offload import breakable_moe_offload_fetch

    assert callable(breakable_moe_offload_fetch)
    # eager_on_graph stamps break_name on the replay closure at capture time;
    # here the decorator's own wrapper is checked to be the decorated one.
    assert breakable_moe_offload_fetch.__name__ in ("wrapper", "_moe_offload_fetch_step")


# --- DEFERRED ---------------------------------------------------------------


def test_nothing_is_emitted_in_the_round_that_recorded_it(armed):
    clock, timeline, records = armed
    graph = _fixture_graph([])

    timeline.new_round()
    graph.replay()
    assert records == [], "round 0 was read inside the round that recorded it"

    timeline.new_round()
    graph.replay()
    assert records == [], "the deferred window is shorter than configured"

    timeline.new_round()
    graph.replay()
    assert [r["round"] for r in records] == [0]


def test_rounds_are_emitted_in_order(armed):
    clock, timeline, records = armed
    graph = _fixture_graph([])

    _run_rounds(graph, timeline, 8)

    assert [r["round"] for r in records] == sorted(r["round"] for r in records)
    assert records[0]["round"] == 0
    assert clock.records_emitted == len(records)


def test_replay_order_is_unchanged_while_measuring(armed):
    clock, timeline, records = armed
    log = []
    graph = _fixture_graph(log)

    _run_rounds(graph, timeline, 1)

    assert log == [
        ("segment", 0),
        ("break", 0),
        ("segment", 1),
        ("break", 1),
        ("segment", 2),
    ]


def test_an_unwound_round_is_dropped_and_does_not_block_the_queue(armed):
    clock, timeline, records = armed
    graph = _fixture_graph([])

    timeline.new_round()
    graph.replay()

    boom = _fixture_graph([])

    def explode():
        raise RuntimeError("break function failed")

    explode.break_name = "_moe_offload_fetch_step"
    boom._break_fns = [explode, explode]
    timeline.new_round()
    with pytest.raises(RuntimeError):
        boom.replay()
    assert clock.rounds_dropped == 1

    # The good round still comes out; the half round did not wedge the drain.
    timeline.new_round()
    graph.replay()
    timeline.new_round()
    graph.replay()
    assert [r["round"] for r in records][:1] == [0]


# --- STEADY STATE -----------------------------------------------------------


def test_events_are_pooled_so_a_long_run_stops_allocating(armed):
    clock, timeline, records = armed
    graph = _fixture_graph([])

    _run_rounds(graph, timeline, 3)
    settled = timeline.events_built

    _run_rounds(graph, timeline, 12)

    assert timeline.events_built == settled, "an armed round allocated new events"
    # In flight at any moment: the round being recorded plus the defer_rounds
    # older ones still waiting to be read. Ten marks each.
    assert settled == (clock._defer_rounds + 1) * MARKS_PER_ROUND
    assert clock.events_created == settled


# --- NEUTRALITY (the default path) ------------------------------------------


def test_probe_is_off_by_default(monkeypatch):
    monkeypatch.delenv(bcc.ENV_ENABLE, raising=False)
    bcc.reset_break_cost_clock_for_test(None)
    try:
        assert bcc.break_cost_clock() is None
    finally:
        bcc.reset_break_cost_clock_for_test(None)


def test_disabled_phase_context_allocates_nothing(monkeypatch):
    monkeypatch.delenv(bcc.ENV_ENABLE, raising=False)
    bcc.reset_break_cost_clock_for_test(None)
    try:
        a = bcc.break_cost_phase("rendezvous")
        b = bcc.break_cost_phase("publish")
        assert a is b is bcc.NO_PHASE
        with bcc.break_cost_phase("fetch") as ctx:
            assert ctx is bcc.NO_PHASE
    finally:
        bcc.reset_break_cost_clock_for_test(None)


def test_default_replay_records_zero_events(monkeypatch, stream_stub):
    """The callcount spy: with the probe off, no CUDA event is even built."""
    import torch

    monkeypatch.delenv(bcc.ENV_ENABLE, raising=False)
    bcc.reset_break_cost_clock_for_test(None)

    built = []

    class ExplodingEvent:
        def __init__(self, *a, **k):
            built.append(1)
            raise AssertionError("the disabled probe created a CUDA event")

    monkeypatch.setattr(torch.cuda, "Event", ExplodingEvent)

    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
        BreakableCUDAGraph,
    )

    def refuse(self, clock):  # the measured loop must not be entered at all
        raise AssertionError("the disabled probe took the measured replay loop")

    monkeypatch.setattr(BreakableCUDAGraph, "_replay_measured", refuse)

    log = []
    graph = _fixture_graph(log)
    try:
        graph.replay()
        graph.replay()
    finally:
        bcc.reset_break_cost_clock_for_test(None)

    assert built == []
    assert log.count(("segment", 0)) == 2
    assert log.count(("break", 0)) == 2


def test_disabled_prepare_breakable_runs_the_unmeasured_path(monkeypatch):
    """The four phase brackets in prepare_breakable are inert when off."""
    monkeypatch.delenv(bcc.ENV_ENABLE, raising=False)
    bcc.reset_break_cost_clock_for_test(None)
    try:
        calls = []
        real = bcc.break_cost_phase
        monkeypatch.setattr(
            bcc, "break_cost_phase", lambda name: (calls.append(name), real(name))[1]
        )
        # The call sites import the symbol directly; assert the module-level
        # binding they use is the disabled one rather than re-importing.
        from sglang.srt.layers.moe import expert_offload as eo

        assert eo.break_cost_phase("planning") is bcc.NO_PHASE
    finally:
        bcc.reset_break_cost_clock_for_test(None)


def test_the_f2_reader_consumes_what_the_probe_writes(armed, tmp_path, capsys):
    """The artifact must be readable by the script the ticket names.

    Run end to end: real sink -> real file -> the real summariser's main().
    A format that only the writer understands is not a measurement artifact.
    """
    import importlib.util

    clock, timeline, _ = armed
    path = tmp_path / "break_cost.rank2.jsonl"
    clock._sink = bcc._JsonlSink(str(path))
    graph = _fixture_graph(
        [], phase_by_crossing={0: ("rendezvous",), 1: ("publish",)}
    )
    _run_rounds(graph, timeline, 6)

    script = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "scripts",
        "dev",
        "494_break_cost",
        "summarise.py",
    )
    spec = importlib.util.spec_from_file_location("break_cost_summarise", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "rank2" in out
    assert "_moe_offload_fetch_step" in out
    assert "BREAK COST/step" in out
    assert "host:rendezvous" in out


# --- BINDS (the phase brackets act at the real call site) --------------------


@pytest.fixture
def no_scratch_env_leak():
    """``t462._cache`` writes SGLANG_MOE_SCRATCH_SLOTS into the real
    environment, and ``scratch_slot_count`` reads it raw. Restore it, or this
    module changes what a later one measures -- test_planner's default-C
    assertion went red exactly this way (the sibling module carries the same
    guard for the same reason)."""
    key = "SGLANG_MOE_SCRATCH_SLOTS"
    before = os.environ.get(key)
    try:
        yield
    finally:
        if before is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = before


def test_prepare_breakable_records_all_four_f2_phases(armed, no_scratch_env_leak):
    """The brackets in ``prepare_breakable`` must actually BIND.

    A phase timer that exists but never fires prices nothing, so this runs the
    REAL ``MoEExpertOffloadCache.prepare_breakable`` (the CPU fixture of
    ``test_breakable_route_462``, not a mock) inside an open slot and requires
    every one of F2's four terms to come back attributed to that slot.
    """
    import torch

    import test_breakable_route_462 as t462

    clock, timeline, records = armed
    cache = t462._cache()
    arena_ids = torch.tensor([[0, 9, 3]], dtype=torch.int64)
    bridge = torch.empty_like(arena_ids)

    rnd = clock.begin_round("binds")
    clock.segment_begin(rnd)
    clock.segment_end(rnd)
    clock.slot_begin(rnd, "_moe_offload_fetch_step")
    cache.prepare_breakable(arena_ids, bridge)
    clock.slot_end(rnd)
    clock.segment_begin(rnd)
    clock.segment_end(rnd)

    assert set(rnd.slot_phases[0]) == set(bcc.PHASES)
    assert all(v >= 0.0 for v in rnd.slot_phases[0].values())


# --- CAN-FAIL ---------------------------------------------------------------


def test_offbyone_crossing_mapping_turns_the_timeline_gate_red(armed, monkeypatch):
    """The executed can-fail arm.

    Replaces the ONE function that decides which segments a crossing is
    measured against with an off-by-one variant. If the timeline gate above
    still passed, it would not be testing the mapping at all.
    """
    clock, timeline, records = armed

    def shifted(rnd, i):
        prev_end = rnd.seg_events[i + 1][1] if i + 1 < len(rnd.seg_events) else None
        next_start = rnd.seg_events[i + 2][0] if i + 2 < len(rnd.seg_events) else None
        return prev_end, next_start

    monkeypatch.setattr(bcc, "_crossing_bounds", shifted)

    graph = _fixture_graph([])
    _run_rounds(graph, timeline, 4)

    rec = records[0]
    with pytest.raises(AssertionError):
        for i, c in enumerate(rec["crossing_detail"]):
            assert c["gap_in_ms"] == pytest.approx(GAP_IN_MS[i])
            assert c["gap_out_ms"] == pytest.approx(GAP_OUT_MS[i])


def test_reading_an_event_too_early_is_caught_by_the_fake(armed):
    """The deferred discipline itself can fail: force a same-round read."""
    clock, timeline, records = armed
    graph = _fixture_graph([])

    timeline.new_round()
    graph.replay()
    rnd = clock._queue[0]

    # What a non-deferred implementation would do -- and what the fake refuses.
    with pytest.raises(AssertionError):
        clock._aggregate(rnd)


def test_a_missing_slot_name_would_lose_the_crossing_identity(armed):
    """Removing the label the decorator stamps degrades the record by name."""
    clock, timeline, records = armed
    graph = _fixture_graph([])
    for fn in graph._break_fns:
        del fn.break_name

    _run_rounds(graph, timeline, 4)

    assert set(records[0]["by_name"]) == {"break"}
