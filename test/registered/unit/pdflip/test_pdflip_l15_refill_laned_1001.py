"""L15-W2A: l15_refill.refill_plan_laned -- the lane-tagged refill plan.

Same rows and the same order as l15_restore.refill_plan, each row extended
with the token's L2 page lane: lane = span.l2_lanes[i] when the record
carries it (P > 1, P1 contract), -1 otherwise (old records, P == 1).
Pure module tests: manifest built from l15_manifest.HoldSpan, no CUDA.
"""

from flliper.srt.pdflip import l15_refill
from flliper.srt.pdflip.l15_manifest import HoldSpan, Manifest
from flliper.srt.pdflip.l15_restore import refill_plan

# rank 0 owns every slot % 8 (prefix (0, 8)), compact_row == slot, so the
# expected rows read directly off the slot numbers.
_PREFIX = (0, 8)
_CAPS = (0,)


def _manifest(lanes_a, lanes_b):
    span_a = HoldSpan(
        rid="A", depth=4, slots=(0, 1, 2, 3), anchor_slot=0,
        l2_slots=(10, 10, 11, -1), l2_gens=(7, 7, 8, 0), l2_lanes=lanes_a,
    )
    span_b = HoldSpan(
        rid="B", depth=2, slots=(4, 5), anchor_slot=4,
        l2_slots=(20, 21), l2_gens=(3, 3), l2_lanes=lanes_b,
    )
    return Manifest(
        epoch=1, pid=1, spans=(span_a, span_b), rows_by_rank=(8,),
        anchor_slots=1,
    )


def _laned():
    return l15_refill.refill_plan_laned(
        _manifest((2, 0, 1, 1), (1, 0)), 0, _PREFIX, _CAPS
    )


def test_rows_and_order_equal_refill_plan():
    m = _manifest((2, 0, 1, 1), (1, 0))
    base = refill_plan(m, 0, _PREFIX, _CAPS)
    laned = l15_refill.refill_plan_laned(m, 0, _PREFIX, _CAPS)
    assert base  # the fixture must actually produce rows
    assert len(laned) == len(base)
    assert [e[:3] for e in laned] == base
    # token 3 of span A has l2_slot -1 (no L2 entry): skipped, not -1-lane
    assert [e[0] for e in laned] == [0, 1, 2, 4, 5]


def test_lanes_come_from_l2_lanes_at_the_span_token_index():
    # 4-tuples, lane last; A: l2_lanes[0..2], B: l2_lanes[0..1]
    assert [e[3] for e in _laned()] == [2, 0, 1, 1, 0]


def test_old_record_without_lanes_lanes_are_minus_one():
    laned = l15_refill.refill_plan_laned(
        _manifest((), ()), 0, _PREFIX, _CAPS
    )
    assert [e[3] for e in laned] == [-1, -1, -1, -1, -1]


def test_short_lanes_record_tail_is_minus_one():
    laned = l15_refill.refill_plan_laned(
        _manifest((1,), ()), 0, _PREFIX, _CAPS
    )
    # span A tokens 1, 2 and both span B tokens have no lane entry
    assert [e[3] for e in laned] == [1, -1, -1, -1, -1]


def test_cap_positive_rank_has_no_plan():
    m = _manifest((2, 0, 1, 1), (1, 0))
    assert l15_refill.refill_plan_laned(m, 0, _PREFIX, (1,)) == []
