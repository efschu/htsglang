"""L15-12 wake-side restore: verdict, TP0 refill plan, sample check, lines."""

from sglang.srt.weg2 import l15_restore as R
from sglang.srt.weg2.l15_manifest import (
    HoldSpan,
    Manifest,
    fingerprint,
    to_json,
)


def _span(rid, slots, l2_slots, l2_gens, depth=None, anchor=0):
    return HoldSpan(
        rid=rid,
        depth=depth if depth is not None else len(slots),
        slots=tuple(slots),
        anchor_slot=anchor,
        l2_slots=tuple(l2_slots),
        l2_gens=tuple(l2_gens),
    )


def _manifest(spans, rows_by_rank=(2, 2, 4), epoch=7, pid=4242):
    return Manifest(
        epoch=epoch,
        pid=pid,
        spans=tuple(spans),
        rows_by_rank=tuple(rows_by_rank),
        anchor_slots=1,
    )


def _write(tmp_path, m, name="l15_hold.json"):
    path = tmp_path / name
    path.write_text(to_json(m))
    return str(path)


# Two spans over 3 ranks, prefix [0, 2, 4, 8] (S = 8):
# span a: slots 1,9 (rank 0), 10,11 (rank 1); span b: slots 0,8 (rank 0).
_SPANS = [
    _span("a", [1, 9, 10, 11], [100, 101, -1, 103], [7, 8, 8, 9]),
    _span("b", [0, 8], [-1, 201], [5, 6], depth=2),
]
_PREFIX = [0, 2, 4, 8]


def test_load_for_wake_reads_back(tmp_path):
    m = _manifest(_SPANS)
    path = _write(tmp_path, m)
    got = R.load_for_wake(path, pid_alive=lambda pid: True)
    assert got is not None
    assert got.epoch == 7 and len(got.spans) == 2
    assert fingerprint(got) == fingerprint(m)


def test_load_for_wake_absent_or_dead_pid(tmp_path):
    m = _manifest(_SPANS)
    path = _write(tmp_path, m)
    assert R.load_for_wake(path, pid_alive=lambda pid: False) is None
    absent = str(tmp_path / "nothing_here.json")
    assert R.load_for_wake(absent, pid_alive=lambda pid: True) is None


def test_verdict_three_states():
    assert R.verdict(None, None, None) == "none"
    assert R.verdict(5, 5, 5) == "hold"
    assert R.verdict(5, 4, 5) == "fallback"
    assert R.verdict(None, 4, 5) == "fallback"
    assert R.verdict(5, None, 5) == "fallback"


def test_refill_plan_rank_zero_cap_zero():
    m = _manifest(_SPANS)
    plan = R.refill_plan(m, 0, _PREFIX, (0, 2, 4))
    # rank 0 owns L%8 in [0,2): slots 1,9 (span a) and 0,8 (span b).
    # compact row = (L//8)*2 + (L%8 - 0); slot 0's L2 entry is -1 (missing).
    assert plan == [(1, 100, 7), (3, 101, 8), (2, 201, 6)]


def test_refill_plan_cap_positive_is_empty():
    m = _manifest(_SPANS)
    assert R.refill_plan(m, 1, _PREFIX, (0, 5, 4)) == []


def test_refill_plan_owner_rule_other_rank():
    m = _manifest(_SPANS)
    plan = R.refill_plan(m, 1, _PREFIX, (0, 0, 4))
    # rank 1 owns L%8 in [2,4): slots 10 (L2 -1, missing) and 11.
    assert plan == [(3, 103, 9)]


def test_missing_l2_entries_counted_not_planned():
    m = _manifest(_SPANS)
    assert R.count_missing(m, 0, _PREFIX) == 1
    assert R.count_missing(m, 1, _PREFIX) == 1
    assert R.count_missing(m, 2, _PREFIX) == 0
    plan = R.refill_plan(m, 0, _PREFIX, (0, 2, 4))
    assert all(slot >= 0 for _, slot, _ in plan)


def test_sample_rows_stable_and_capped():
    plan = [(i, i * 10, 1) for i in range(200)]
    first = R.sample_rows(plan, 64)
    second = R.sample_rows(plan, 64)
    assert len(first) <= 64
    assert first == second
    assert all(b > a for a, b in zip(first, first[1:]))
    assert R.sample_rows([3, 1, 2], 64) == [1, 2, 3]
    assert R.sample_rows([], 64) == []


def test_check_line_and_restore_line_formats():
    assert (
        R.check_line(2, 40, 1, 3) == "L15-CHECK rank=2 ok=40 bad=1 missing=3"
    )
    assert R.restore_line(7, "hold", (2, 2, 4), 5, 1) == (
        "L15-RESTORE epoch=7 verdict=hold keep_rows_by_rank=2,2,4 "
        "refill_rows=5 missing=1"
    )
