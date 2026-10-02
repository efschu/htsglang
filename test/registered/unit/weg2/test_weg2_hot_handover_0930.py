"""AP L15-09 (2026-09-30): the hot D->P handover state machine and its plan picker.

Pure tests -- no CUDA, no I/O, no env. Plain pytest functions (CustomTestCase
would wrap failures in retry() and hide the assertion).
"""

import re

import pytest

from sglang.srt.weg2.hot_handover import Handover, HandoverPlan, decide


# ---------------------------------------------------------------- a. decide --


def test_decide_picks_first_hot_candidate_that_fits():
    candidates = [
        {"rid": "r-cold", "hot_in_d": False, "prefix_tokens": 10, "anchor_depth": 10},
        {"rid": "r-hot", "hot_in_d": True, "prefix_tokens": 200, "anchor_depth": 200},
        {"rid": "r-hot2", "hot_in_d": True, "prefix_tokens": 50, "anchor_depth": 50},
    ]
    plan = decide(candidates, p_free_rows=500)
    assert plan is not None
    assert plan.rid == "r-hot"          # input order, not best-fit
    assert plan.n_tokens == 200
    assert plan.anchor_depth == 200
    assert plan.p_row0 == 0             # rows [0, n) right after P's reset


def test_decide_skips_candidate_without_anchor_at_depth():
    # F3: never KV without its anchor at depth -> anchor_depth must equal
    # prefix_tokens exactly, even for a hot candidate that would fit.
    candidates = [
        {"rid": "r-shallow", "hot_in_d": True, "prefix_tokens": 100, "anchor_depth": 50},
        {"rid": "r-deeper", "hot_in_d": True, "prefix_tokens": 100, "anchor_depth": 128},
        {"rid": "r-ok", "hot_in_d": True, "prefix_tokens": 80, "anchor_depth": 80},
    ]
    plan = decide(candidates, p_free_rows=100)
    assert plan is not None
    assert plan.rid == "r-ok"


def test_decide_skips_what_does_not_fit():
    candidates = [
        {"rid": "r-big", "hot_in_d": True, "prefix_tokens": 500, "anchor_depth": 500},
        {"rid": "r-fits", "hot_in_d": True, "prefix_tokens": 500, "anchor_depth": 500},
    ]
    # first is skipped only because p_free_rows cannot carry it; the second
    # candidate in input order is identical in size so None is the answer:
    assert decide(candidates, p_free_rows=499) is None
    # equality still fits (prefix_tokens <= p_free_rows):
    plan = decide([candidates[0]], p_free_rows=500)
    assert plan is not None and plan.rid == "r-big"


def test_decide_none_when_cold_or_empty():
    assert decide([], p_free_rows=1000) is None
    assert decide(
        [{"rid": "x", "hot_in_d": False, "prefix_tokens": 1, "anchor_depth": 1}],
        p_free_rows=1000,
    ) is None


# ----------------------------------------------------------- b. the machine --


def _h(rid="r1", n=100):
    return Handover(HandoverPlan(rid=rid, n_tokens=n, anchor_depth=n))


def test_happy_path_planned_deposited_adopted():
    h = _h()
    assert h.state == "planned"
    h.deposit_done(bytes_local=40 << 20, bytes_lane=60 << 20)
    assert h.state == "d_deposited"
    h.adopt(manifest_complete=True, fp_agree=True)
    assert h.state == "p_adopted"
    assert h.reason == ""


def test_adopt_without_complete_manifest_falls_back():
    h = _h()
    h.deposit_done(bytes_local=1, bytes_lane=1)
    h.adopt(manifest_complete=False, fp_agree=True)
    assert h.state == "fallen_back"
    assert h.reason == "incomplete_manifest"


def test_adopt_with_disagreeing_fingerprint_falls_back():
    h = _h()
    h.deposit_done(bytes_local=1, bytes_lane=1)
    h.adopt(manifest_complete=True, fp_agree=False)
    assert h.state == "fallen_back"
    assert h.reason == "disagree"


def test_incomplete_manifest_wins_over_disagree():
    # both halves bad -> the named cause is the manifest, checked first
    h = _h()
    h.deposit_done(bytes_local=1, bytes_lane=1)
    h.adopt(manifest_complete=False, fp_agree=False)
    assert h.reason == "incomplete_manifest"


def test_fall_back_is_legal_from_planned_and_from_d_deposited():
    h = _h()
    h.fall_back("no_lane")
    assert h.state == "fallen_back" and h.reason == "no_lane"
    h2 = _h(rid="r2")
    h2.deposit_done(bytes_local=1, bytes_lane=1)
    h2.fall_back("lane_lost")
    assert h2.state == "fallen_back" and h2.reason == "lane_lost"


def test_illegal_transitions_raise_value_error_naming_from_and_to():
    h = _h()
    with pytest.raises(ValueError, match=r"planned.*p_adopted"):
        h.adopt(manifest_complete=True, fp_agree=True)   # skips d_deposited
    h.deposit_done(bytes_local=1, bytes_lane=1)
    with pytest.raises(ValueError, match=r"d_deposited -> d_deposited"):
        h.deposit_done(bytes_local=1, bytes_lane=1)      # no re-deposit
    h.adopt(manifest_complete=True, fp_agree=True)
    for call in (lambda: h.adopt(True, True), lambda: h.deposit_done(1, 1),
                 lambda: h.fall_back("late")):
        with pytest.raises(ValueError):
            call()                                       # terminal states stay terminal


def test_line_reports_the_state():
    h = _h(rid="abc", n=42)
    assert h.line == "HOT-HANDOVER rid=abc n=42 state=planned reason="
    h.deposit_done(bytes_local=1, bytes_lane=1)
    h.adopt(manifest_complete=False, fp_agree=True)
    assert h.line == ("HOT-HANDOVER rid=abc n=42 state=fallen_back "
                      "reason=incomplete_manifest")


# ----------------------------------------------------- c. the ordering note --


def test_module_docstring_carries_six_file_line_anchors():
    import sglang.srt.weg2.hot_handover as mod

    anchors = re.findall(r"\w+\.py:\d+", mod.__doc__ or "")
    assert len(anchors) >= 6, f"ordering note needs >=6 file.py:line anchors, got {anchors}"
