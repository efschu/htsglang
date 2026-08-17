"""#690: the fixed flip cost, decomposed from the runtime's own log lines.

Calibration data, from 765 unique `PHASE-FLIP DONE` lines across the boot
captures in /spinning/evidence-665-f1. Medians:

    total     3357 ms   (min 2401, max 4874 -- the reported 2.0-4.2 s band)
    read        30.6      0.9%
    exchange   847.8     25.3%
    write      302.6      9.0%
    movers    1458.4     43.4%
    cutover    670.7     20.0%
    residual    99.2      3.0%

Both directions agree closely (pp_to_tp 3298 ms, tp_to_pp 3417 ms).

NOTE ON THE MEDIANS: medians do not sum, so the FlipComponents below is a
SYNTHETIC composite -- each field is that component's own median, and they add
to slightly more than the median total (hence a ~3% residual by construction).
The per-event median of (movers+cutover)/total is 61.6%; against the composite
it reads 63.4%. Both are honest; the tests use the composite because that is
the object the solver is handed.

Hermetic: pure arithmetic, no CUDA, no server.
"""

import pytest
from sglang.srt.planner.flip_cost_model import (
    FlipComponents,
    FlipCostError,
    Lever,
    apply_overlap,
    max_feasible_load,
    rank_levers,
    reopened_load,
)

MEASURED = FlipComponents(
    read_ms=30.6,
    exchange_ms=847.8,
    write_ms=302.6,
    movers_ms=1458.4,
    cutover_ms=670.7,
    total_ms=3357.0,
)
BUDGET_S = 10.0
DECODE_SHARE = 0.5


def test_the_kv_seam_move_is_the_MINORITY_of_the_flip():
    """The premise of the whole task, checked against the logs.

    Everyone reaches for read+exchange+write first. It is about a third.
    """
    assert MEASURED.share(MEASURED.kv_seam_ms) == pytest.approx(0.353, abs=0.01)
    outside = MEASURED.movers_ms + MEASURED.cutover_ms
    assert MEASURED.share(outside) == pytest.approx(0.634, abs=0.01)
    assert outside > MEASURED.kv_seam_ms


def test_the_residual_is_small_so_the_named_split_is_the_story():
    """The 'unexplained' part of the cost is ~3%, not the bulk."""
    assert MEASURED.share(MEASURED.residual_ms) < 0.05
    assert MEASURED.residual_ms > 0.0


def test_movers_is_the_single_largest_component():
    """movers = GDN state + the weights arena refill (an H2D image copy)."""
    parts = {
        "read": MEASURED.read_ms,
        "exchange": MEASURED.exchange_ms,
        "write": MEASURED.write_ms,
        "movers": MEASURED.movers_ms,
        "cutover": MEASURED.cutover_ms,
    }
    assert max(parts, key=parts.get) == "movers"
    assert MEASURED.share(MEASURED.movers_ms) > 0.40


def test_overlapping_movers_is_capped_by_the_shorter_leg():
    """Hiding a 1458 ms leg behind an 1150 ms one saves 1150, not 1458."""
    seam = MEASURED.exchange_ms + MEASURED.write_ms
    c = FlipComponents(
        read_ms=MEASURED.read_ms,
        exchange_ms=seam,
        write_ms=0.0,
        movers_ms=MEASURED.movers_ms,
        cutover_ms=MEASURED.cutover_ms,
        total_ms=MEASURED.total_ms,
    )
    after = apply_overlap(c, hide="movers_ms", behind="exchange_ms")
    assert after == pytest.approx(MEASURED.total_ms - seam)
    assert after > MEASURED.total_ms - MEASURED.movers_ms, (
        "cannot save more than exists"
    )


def test_a_lever_that_deletes_the_seam_move_reports_the_right_total():
    """#704b's phase-uniform vector deletes exchange+write outright."""
    lv = Lever(
        name="phase-uniform KV vector (#704b)",
        attacks=("exchange_ms", "write_ms"),
        removes_fraction=1.0,
        mechanism="same token vector in both phases, so no rows move at the seam",
        price="caps ladder depth at n0<=37",
    )
    assert lv.apply(MEASURED) == pytest.approx(
        MEASURED.total_ms - MEASURED.exchange_ms - MEASURED.write_ms
    )


def test_levers_rank_by_resulting_flip_time():
    levers = [
        Lever("tiny", ("read_ms",), 1.0, "m", ""),
        Lever("seam", ("exchange_ms", "write_ms"), 1.0, "m", ""),
        Lever("movers", ("movers_ms",), 1.0, "m", ""),
    ]
    ranked = rank_levers(MEASURED, levers)
    # movers (1458 ms) exceeds exchange+write (1150 ms), so deleting the movers
    # leg beats deleting the entire KV seam move. That ordering is the finding,
    # not an artifact: the biggest single lever is not the one about KV.
    assert [lv.name for lv, _ in ranked] == ["movers", "seam", "tiny"]
    assert ranked[0][1] < ranked[-1][1]


def test_a_cheaper_flip_raises_the_MAXIMUM_FEASIBLE_LOAD():
    """THE conversion this module exists for.

    #677 showed flip cost sets the stability floor and the latency ceiling in
    opposite directions, so a reduction reopens configurations rather than
    shaving an overhead. The unit of a lever is therefore load, not
    milliseconds.
    """
    dear = max_feasible_load(4.2, BUDGET_S, DECODE_SHARE)
    mid = max_feasible_load(3.357, BUDGET_S, DECODE_SHARE)
    cheap = max_feasible_load(1.0, BUDGET_S, DECODE_SHARE)
    assert cheap > mid > dear
    assert dear > 0.0


def test_reopened_load_reports_the_before_and_after_pair():
    before, after = reopened_load(3.357, 2.2, BUDGET_S, DECODE_SHARE)
    assert after > before
    assert 0.0 < before < 1.0 and 0.0 < after < 1.0


def test_a_flip_costing_more_than_the_budget_admits_no_load_at_all():
    assert max_feasible_load(12.0, BUDGET_S, DECODE_SHARE) == 0.0


def test_the_model_is_not_rig_specific():
    """Foreign profile: a fast flip on a differently-shaped split."""
    foreign = FlipComponents(
        read_ms=1.0,
        exchange_ms=5.0,
        write_ms=2.0,
        movers_ms=3.0,
        cutover_ms=4.0,
        total_ms=16.0,
    )
    assert foreign.residual_ms == pytest.approx(1.0)
    assert foreign.share(foreign.kv_seam_ms) == pytest.approx(0.5)
    # And its levers rank on its own numbers, not on this rig's.
    ranked = rank_levers(
        foreign,
        [
            Lever("seam", ("exchange_ms", "write_ms"), 1.0, "m", ""),
            Lever("cutover", ("cutover_ms",), 1.0, "m", ""),
        ],
    )
    assert ranked[0][0].name == "seam"


def test_malformed_inputs_are_refused():
    with pytest.raises(FlipCostError, match="no component named"):
        Lever("bad", ("nope_ms",), 1.0, "m", "").apply(MEASURED)
    with pytest.raises(FlipCostError, match=r"\[0,1\]"):
        Lever("bad", ("read_ms",), 1.5, "m", "").apply(MEASURED)
    with pytest.raises(FlipCostError, match="decode_share"):
        max_feasible_load(1.0, BUDGET_S, 0.0)
    with pytest.raises(FlipCostError, match="total flip time"):
        FlipComponents(0, 0, 0, 0, 0, 0).share(1.0)
