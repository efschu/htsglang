# SPDX-License-Identifier: Apache-2.0
"""L15-16: the cap-0 rank's park plan (pure)."""

from __future__ import annotations

from sglang.srt.weg2.l15_park import ParkPiece, park_bytes, park_plan, parked_rows_on


def test_one_capped_rank_takes_everything_after_its_own_rows():
    pieces, why = park_plan(keep_rows=[700, 400, 500], caps=[0, 2000, 600])
    assert why is None
    assert pieces == [ParkPiece(src=0, dst=1, src_row=0, dst_row=400, rows=700)]
    assert parked_rows_on(pieces, 1) == 700 and park_bytes(pieces, 10) == 7000


def test_split_over_two_capped_ranks_largest_free_first():
    pieces, why = park_plan(keep_rows=[700, 400, 500], caps=[0, 900, 800])
    assert why is None
    assert pieces == [ParkPiece(0, 1, 0, 400, 500), ParkPiece(0, 2, 500, 500, 200)]


def test_refused_by_name_when_the_free_rows_do_not_suffice():
    pieces, why = park_plan(keep_rows=[700, 400, 500], caps=[0, 600, 600])
    assert pieces == [] and "needs" in why and "free" in why


def test_rank_agnostic_any_cap0_position_and_nothing_to_park():
    pieces, why = park_plan(keep_rows=[300, 50, 0], caps=[1000, 0, 400])
    assert why is None and pieces == [ParkPiece(1, 0, 0, 300, 50)]
    assert park_plan([10, 20], [100, 100]) == ([], None)


def test_capped_rank_over_its_cap_and_length_mismatch_refuse():
    assert park_plan([10, 200], [0, 100])[1] is not None
    assert park_plan([10, 20], [0, 100, 5])[1] is not None
