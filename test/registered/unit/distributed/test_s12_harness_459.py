# SPDX-License-Identifier: Apache-2.0
"""#459: the two desk-provable halves of the s12 harness follow-up.

PLACEMENT NOTE. The natural home of this file is next to
``test/registered/unit/distributed/test_s12_log_analyse.py``, which already
owns the s12 parser fixtures. That directory belongs to another change in
flight, so the file lands here and should be moved when the two merge.

(2a) THE CLOCK RAMP (#475 SS6). ``s12_prefill_kurve.py`` measured one draw per
invocation, so a floor round was three separate invocations with the harness's
own orchestration between them: #435 sub-arm B2 drew 1597.7 / 1720.2 / 1820.2
tok/s 48-51 s apart, with ~12 s of work each, and reported that 13.0 % as an
A-vs-A noise floor where #424 reported 3.0 % on the same instrument. The
CollectiveClock lines say the collective axis was flat to 1.6 % and the whole
spread was compute, and monotone -- a clock ramp, not exchangeable samples,
and loose in the direction that lets a real regression pass as "within the
floor" (#435 sub-arm B: -8.5 % scored as parity).

The warm-up existed but was INVISIBLE: ``mode_measure`` ran it and threw the
result away without recording that it had run, so a point could not be
distinguished from one taken with no warm-up at all. The tests below pin the
three properties the artifact now has to carry: which draw was discarded, what
it measured, and the measured idle gap before every draw.

(2b) THE SPEC-OFF PARSE. ``RE_DECODE`` required the accept block that the
scheduler only writes when speculation is ON
(``metrics_reporter.py:968-972``, ``:1018``), and allowed nothing between it
and the graph flag. So a spec-off boot parsed ZERO decode ticks and s12
reported 0/0 -- and, less obviously, so did a CAP_ACCEPT boot, which DOES
write an accept block but puts ``cap len:`` behind it (``:1020``).

Both shapes are pinned below. The spec-off sample is DERIVED from the spec-on
one by deleting exactly the segment the emitter appends at ``:1018``, rather
than typed fresh: the two lines then cannot drift apart, which is the #315
lesson about coupling an assertion to its real source.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
)
BATTERY = os.path.join(REPO_ROOT, "scripts", "gpu_battery")
sys.path.insert(0, BATTERY)

from s12_log_analyse import (  # noqa: E402  one parser, one place
    RE_DECODE,
    decode_tick_aggregat,
    parse_decode,
)
from s12_prefill_kurve import (  # noqa: E402
    BACK_TO_BACK_MAX_GAP_S,
    WARMUP_DRAW_INDEX,
    draw_label,
    floor_from_points,
    run_draw_series,
)

# ---------------------------------------------------------------------------
# the two decode line shapes
# ---------------------------------------------------------------------------

#: Verbatim from the 2026-07-30 s12 run (the same line
#: ``test_s12_log_analyse.py`` pins), speculation ON.
LINE_SPEC_ON = (
    "[2026-07-30 13:35:13 TP0] Decode batch, #running-req: 1, #full token: 257, "
    "full token usage: 0.00, mamba num: 3, mamba usage: 0.03, accept len: 2.83, "
    "accept rate: 0.61, cuda graph: True, gen throughput (token/s): 93.65, "
    "#queue-req: 0"
)

#: The segment ``metrics_reporter.py:1018`` appends, and ONLY that segment.
#: With ``spec_algorithm.is_none()`` the whole block at :968-972 is skipped and
#: the message goes straight from the pool usage parts to the graph flag.
ACCEPT_SEGMENT = "accept len: 2.83, accept rate: 0.61, "
LINE_SPEC_OFF = LINE_SPEC_ON.replace(ACCEPT_SEGMENT, "")

#: Speculation on, ragged verify in CAP_ACCEPT mode: ``cap len:`` (:1020) and
#: ``block accept len:`` (:1022) sit between the accept block and the graph
#: flag. The pre-#459 pattern matched nothing here either.
LINE_SPEC_ON_CAP = LINE_SPEC_ON.replace(
    "cuda graph:", "cap len: 3.40, block accept len: 2.10, cuda graph:"
)

#: ``LOG_FORWARD_ITERS`` (:962) puts the iteration counter behind "Decode
#: batch". Same line otherwise.
LINE_SPEC_OFF_ITER = LINE_SPEC_OFF.replace("Decode batch,", "Decode batch [4711],")


class TestTheSpecOffShapeIsRealAndDifferent:
    """The precondition: the two sample lines must actually differ in the way
    the emitter differs, or the parser tests below pin nothing."""

    def test_the_spec_off_line_is_the_spec_on_line_minus_the_accept_block(self):
        assert ACCEPT_SEGMENT in LINE_SPEC_ON
        assert "accept len" not in LINE_SPEC_OFF
        assert "accept rate" not in LINE_SPEC_OFF
        # Everything else is untouched: same rank, same rate, same graph flag.
        assert "gen throughput (token/s): 93.65" in LINE_SPEC_OFF
        assert "#running-req: 1" in LINE_SPEC_OFF

    def test_the_old_pattern_could_not_have_matched_it(self):
        """The pre-#459 pattern, spelled out, against both shapes. This is the
        defect, executed rather than described."""
        import re

        old = re.compile(
            r"\[(?P<zeit>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) TP(?P<rang>\d+)\] "
            r"Decode batch, #running-req: (?P<running_req>\d+),.*?"
            r"accept len: (?P<accept_len>[\d.]+), "
            r"accept rate: (?P<accept_rate>[\d.]+), "
            r"cuda graph: (?P<cuda_graph>\w+), "
            r"gen throughput \(token/s\): (?P<gen_tok_s>[\d.]+)"
        )
        assert old.search(LINE_SPEC_ON) is not None
        assert old.search(LINE_SPEC_OFF) is None
        assert old.search(LINE_SPEC_ON_CAP) is None
        assert old.search(LINE_SPEC_OFF_ITER) is None


class TestParseDecodeBothModes:
    def test_spec_on_is_parsed_exactly_as_before(self):
        """The regression guard: the spec-on numbers may not move by a digit."""
        (row,) = parse_decode([LINE_SPEC_ON])
        assert row["running_req"] == 1
        assert row["rang"] == 0
        assert row["accept_len"] == 2.83
        assert row["accept_rate"] == 0.61
        assert row["gen_tok_s"] == 93.65
        assert row["cuda_graph"] is True
        assert row["spec"] is True

    def test_spec_off_parses_its_real_decode_numbers(self):
        """THE FALSIFIER for (2b). Red before the fix: ``parse_decode`` returned
        [] and s12 reported 0/0 ticks for the whole boot."""
        (row,) = parse_decode([LINE_SPEC_OFF])
        assert row["running_req"] == 1
        assert row["gen_tok_s"] == 93.65
        assert row["cuda_graph"] is True
        # Absent, not zero: a 0.0 here would read downstream as a MEASURED
        # accept length of zero and would divide into ms/verify.
        assert row["spec"] is False
        assert row["accept_len"] is None
        assert row["accept_rate"] is None

    def test_the_cap_accept_shape_parses_with_its_accept_numbers(self):
        (row,) = parse_decode([LINE_SPEC_ON_CAP])
        assert row["spec"] is True
        assert row["accept_len"] == 2.83
        assert row["gen_tok_s"] == 93.65

    def test_the_iteration_tag_does_not_hide_a_tick(self):
        (row,) = parse_decode([LINE_SPEC_OFF_ITER])
        assert row["gen_tok_s"] == 93.65
        assert row["spec"] is False

    def test_the_graph_backend_is_captured_not_spelled(self):
        """``_graph_backend_label`` is a device lookup
        (metrics_reporter.py:284-288): cpu/npu/musa write their own word."""
        line = LINE_SPEC_OFF.replace("cuda graph:", "npu graph:")
        (row,) = parse_decode([line])
        assert row["graph_backend"] == "npu"
        assert row["cuda_graph"] is True

    def test_a_line_that_is_not_a_decode_tick_is_not_a_tick(self):
        assert parse_decode(["[2026-07-30 13:35:13 TP0] Prefill batch, whatever"]) == []
        assert RE_DECODE.search("nonsense") is None


class TestAggregationWithoutAccept:
    def test_a_spec_off_window_reports_its_rate_and_no_accept(self):
        agg = decode_tick_aggregat(parse_decode([LINE_SPEC_OFF]), running_req=1)
        assert agg["ticks"] == 1
        assert agg["ticks_gewertet"] == 1
        assert agg["ticks_with_accept"] == 0
        assert agg["gen_tok_s_median"] == 93.65
        assert agg["ms_pro_token"] == pytest.approx(1000.0 / 93.65)
        # The two accept-derived numbers are absent, and say so.
        assert agg["accept_len_median"] is None
        assert agg["ms_pro_verify"] is None

    def test_a_spec_on_window_is_unchanged(self):
        agg = decode_tick_aggregat(parse_decode([LINE_SPEC_ON]), running_req=1)
        assert agg["ticks_with_accept"] == 1
        assert agg["accept_len_median"] == 2.83
        assert agg["ms_pro_verify"] == pytest.approx(1000.0 / 93.65 * 2.83)


# ---------------------------------------------------------------------------
# (2a) the floor draws
# ---------------------------------------------------------------------------


class _Clock:
    """A monotonic clock the test drives, so a gap is asserted rather than
    slept through."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _series(draws, *, warmup=True, work_s=12.0, gap_s=0.0, rates=None, clock=None):
    """Run ``run_draw_series`` against a fake server on a fake clock."""
    import s12_prefill_kurve as s12

    clock = clock or _Clock()
    seen: list = []
    rates = list(rates or [])

    def draw(index):
        seen.append(index)
        clock.now += work_s
        rate = rates.pop(0) if rates else 1000.0
        return {"prefill_tok_s": rate, "requests": 7, "roh": []}

    def between():
        clock.now += gap_s

    old = s12.time.monotonic
    s12.time.monotonic = clock
    try:
        series, records = run_draw_series(
            draw, draws, warmup=warmup, between=between
        )
    finally:
        s12.time.monotonic = old
    return series, records, seen


class TestTheWarmupDrawIsDiscardedExplicitly:
    def test_the_artifact_names_the_discarded_draw_and_its_number(self):
        """THE FALSIFIER for (2a), half one. Before the fix the warm-up ran and
        vanished: the emitted point carried no field that could tell a
        warmed-up run from a cold one."""
        series, records, seen = _series(3, rates=[800.0, 1000.0, 1010.0, 1005.0])
        assert seen == [0, 1, 2, 3]
        assert [i for i, _, _ in records] == [1, 2, 3]
        warm = series["warmup_draw"]
        assert warm["draw"] == WARMUP_DRAW_INDEX == 0
        assert warm["discarded"] is True
        assert "clock ramp" in warm["reason"]
        # Its own number is kept, not hidden: 800 against the 1000-1010 of the
        # measured draws IS the ramp, and a reader can see it.
        assert warm["prefill_tok_s"] == 800.0
        assert series["discarded_draws"] == [0]

    def test_a_run_without_a_warmup_says_so_instead_of_implying_one(self):
        series, records, seen = _series(2, warmup=False)
        assert seen == [1, 2]
        assert series["warmup_draw"] is None
        assert series["discarded_draws"] == []
        # No warm-up means the first measured draw has no gap before it, and
        # "not measured" must not read as "zero".
        assert series["gap_before_s"][0] is None


class TestTheDrawsRunBackToBack:
    def test_the_gap_before_every_draw_is_recorded(self):
        """Half two. The property is measured, not asserted: the artifact
        carries the gaps, so the claim can be refuted from the file."""
        series, _, _ = _series(3, gap_s=0.4)
        assert series["gap_before_s"] == [0.4, 0.4, 0.4]
        assert series["max_gap_s"] == 0.4
        assert series["back_to_back"] is True

    def test_the_ramp_shape_is_reported_as_not_back_to_back(self):
        """The #475 SS6 arms: 48-51 s of idle between draws. The artifact has
        to come out saying so rather than looking like a tight run."""
        series, _, _ = _series(3, gap_s=49.0)
        assert series["max_gap_s"] == 49.0
        assert series["back_to_back"] is False
        assert series["back_to_back_max_gap_s"] == BACK_TO_BACK_MAX_GAP_S

    def test_a_single_draw_with_a_warmup_still_carries_the_verdict(self):
        """The DEFAULT path. One measured draw is still preceded by the
        discarded warm-up, and the gap to it is what the artifact records."""
        series, records, _ = _series(1, gap_s=0.2)
        assert series["draws"] == 1
        assert len(records) == 1
        assert series["gap_before_s"] == [0.2]
        assert series["back_to_back"] is True


class TestTheFloorSummaryReadsBackTheProperties:
    def _points(self, rates, back_to_back=True, max_gap=0.4):
        series = {
            "draws": len(rates),
            "warmup_draw": {"draw": 0, "discarded": True, "prefill_tok_s": 800.0},
            "discarded_draws": [0],
            "back_to_back": back_to_back,
            "max_gap_s": max_gap,
        }
        return [
            {
                "arm": draw_label("int8_match_B2", i, len(rates)),
                "base_arm": "int8_match_B2",
                "draw": i,
                "sessions": 8,
                "floor_series": series,
                "prefill": {"prefill_tok_s": r},
            }
            for i, r in enumerate(rates, start=1)
        ]

    def test_the_draws_are_filed_under_the_suffix_the_tooling_strips(self):
        """``window_accounting.py:92`` maps a draw back to its arm with
        ``re.sub(r"_floorP\\d$", "", arm)``; the names have to match it."""
        import re

        assert draw_label("int8_match_B2", 1, 1) == "int8_match_B2"
        assert draw_label("int8_match_B2", 2, 3) == "int8_match_B2_floorP2"
        assert (
            re.sub(r"_floorP\d$", "", draw_label("int8_match_B2", 2, 3))
            == "int8_match_B2"
        )

    def test_the_measured_ramp_is_reported_as_monotone(self):
        """The #435 sub-arm B2 draws, verbatim. A monotone series is a drift:
        its spread is not a noise floor even though the arithmetic produces a
        number, so the artifact reports both."""
        (row,) = floor_from_points(self._points([1597.7, 1720.2, 1820.2]))
        assert row["arm"] == "int8_match_B2"
        assert row["draws"] == 3
        assert row["monotone"] is True
        assert row["spread_pct"] == pytest.approx(13.93, abs=0.01)
        assert row["discarded_draws"] == [0]
        assert row["warmup_prefill_tok_s"] == 800.0

    def test_a_non_monotone_series_is_reported_as_such(self):
        """#424 int8_decode: 120.6 / 113.3 / 113.4 -- the arm that reached
        steady state fastest and reported a usable floor."""
        (row,) = floor_from_points(self._points([120.6, 113.3, 113.4]))
        assert row["monotone"] is False
        assert row["spread_pct"] == pytest.approx(6.44, abs=0.01)

    def test_single_draw_points_carry_no_floor_row(self):
        points = [
            {
                "arm": "bar1",
                "draw": 1,
                "sessions": 1,
                "floor_series": {"draws": 1},
                "prefill": {"prefill_tok_s": 1190.7},
            }
        ]
        assert floor_from_points(points) == []


class TestTheEmittedArtifactCarriesIt:
    """End to end on the emitted file: the point lines themselves, not only
    the helper's return value. An artifact that cannot show the two properties
    is not evidence for them."""

    def test_a_floor_series_writes_one_point_per_draw_with_the_evidence(
        self, tmp_path, monkeypatch
    ):
        import s12_prefill_kurve as s12

        clock = _Clock()
        rates = iter([800.0, 1000.0, 1010.0, 1005.0])
        monkeypatch.setattr(s12.time, "monotonic", clock)
        monkeypatch.setattr(s12, "_flush_cache", lambda port: "flushed")

        def fake_measure(port, arm, sessions, seconds, target_tokens):
            clock.now += seconds
            return {
                "prefill_tok_s": next(rates),
                "requests": 7,
                "roh": [{"slot": 0}],
                "fehler": [],
            }

        monkeypatch.setattr(s12, "measure_prefill", fake_measure)

        class Args:
            out_dir = str(tmp_path)
            port = 1
            arm = "int8_match_B2"
            sessions = 8
            folge = 3
            point_seconds = 12.0
            warmup_seconds = 6.0
            prompt_tokens = 2048
            with_decode = 0
            decode_batches = ""
            server_log = ""
            floor_draws = 3

        assert s12.mode_measure(Args()) == 0

        lines = [
            json.loads(x)
            for x in open(os.path.join(str(tmp_path), "punkte.jsonl"))
            if x.strip()
        ]
        assert [p["arm"] for p in lines] == [
            "int8_match_B2_floorP1",
            "int8_match_B2_floorP2",
            "int8_match_B2_floorP3",
        ]
        for p in lines:
            assert p["base_arm"] == "int8_match_B2"
            assert p["warmup_seconds"] == 6.0
            series = p["floor_series"]
            # WHICH draw was discarded, and what it measured.
            assert series["discarded_draws"] == [0]
            assert series["warmup_draw"]["draw"] == 0
            assert series["warmup_draw"]["prefill_tok_s"] == 800.0
            # THAT the draws were back-to-back, with the numbers behind it.
            assert series["back_to_back"] is True
            assert series["gap_before_s"] == [0.0, 0.0, 0.0]
        assert [p["draw"] for p in lines] == [1, 2, 3]
        assert [p["prefill"]["prefill_tok_s"] for p in lines] == [
            1000.0,
            1010.0,
            1005.0,
        ]
        # The raw rows are filed per draw, not overwritten by the last one.
        raw = sorted(f for f in os.listdir(str(tmp_path)) if f.startswith("roh_"))
        assert raw == [
            "roh_int8_match_B2_floorP1_8.jsonl",
            "roh_int8_match_B2_floorP2_8.jsonl",
            "roh_int8_match_B2_floorP3_8.jsonl",
        ]

    def test_the_default_single_draw_point_keeps_its_name_and_gains_the_proof(
        self, tmp_path, monkeypatch
    ):
        """Backward compatibility: one draw, plain arm name, same shape as
        before -- plus the warm-up record that was missing."""
        import s12_prefill_kurve as s12

        clock = _Clock()
        rates = iter([800.0, 1190.7])
        monkeypatch.setattr(s12.time, "monotonic", clock)
        monkeypatch.setattr(s12, "_flush_cache", lambda port: "flushed")

        def fake_measure(port, arm, sessions, seconds, target_tokens):
            clock.now += seconds
            return {"prefill_tok_s": next(rates), "requests": 7, "roh": []}

        monkeypatch.setattr(s12, "measure_prefill", fake_measure)

        class Args:
            out_dir = str(tmp_path)
            port = 1
            arm = "bar1"
            sessions = 1
            folge = 1
            point_seconds = 15.0
            warmup_seconds = 8.0
            prompt_tokens = 2048
            with_decode = 0
            decode_batches = ""
            server_log = ""
            floor_draws = 1

        assert s12.mode_measure(Args()) == 0
        (point,) = [
            json.loads(x)
            for x in open(os.path.join(str(tmp_path), "punkte.jsonl"))
            if x.strip()
        ]
        assert point["arm"] == "bar1"
        assert point["prefill"]["prefill_tok_s"] == 1190.7
        assert point["floor_series"]["warmup_draw"]["prefill_tok_s"] == 800.0
        assert point["floor_series"]["discarded_draws"] == [0]
        assert point["gap_before_s"] == 0.0
