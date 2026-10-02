# SPDX-License-Identifier: Apache-2.0
"""L15-14a: the deposit region (D) and the front's deposit book."""

from __future__ import annotations

from sglang.srt.weg2.l15_deposit import DepositBook, DepositRegion, deposit_region

PREFIX = [0, 7, 11, 16]


def test_region_starts_after_the_held_blocks_and_ends_at_the_capped_ranks_limit():
    # held: 3 blocks (rows 21/12/15 on ratios 7/4/5); caps: rank0 0, rank1 40, rank2 60
    reg = deposit_region([21, 12, 15], PREFIX, caps=[0, 40, 60], anchor_slots=3,
                         anchor_cap=8)
    assert reg == DepositRegion(e0=48, e1=160, a0=3, a1=9, skip_ranks=(0,))


def test_no_room_or_no_capped_rank_gives_none():
    assert deposit_region([21, 12, 15], PREFIX, caps=[0, 0, 0], anchor_slots=1,
                          anchor_cap=8) is None
    assert deposit_region([70, 40, 50], PREFIX, caps=[0, 40, 60], anchor_slots=1,
                          anchor_cap=8) is None


def test_book_hands_out_contiguous_ranges_and_anchor_rows_until_full():
    book = DepositBook(DepositRegion(e0=48, e1=100, a0=3, a1=5, skip_ranks=(0,)))
    assert book.assign("a", 30) == (48, 30, 3)
    assert book.assign("a", 30) == (48, 30, 3)       # idempotent per rid
    assert book.assign("b", 30) is None              # 78+30 > 100
    assert book.assign("c", 22) == (78, 22, 4)
    assert book.assign("d", 1) is None               # region full


def test_front_deposits_follow_the_published_epoch(tmp_path):
    import json

    from sglang.srt.weg2.l15_deposit import FrontDeposits, read_deposit_hint

    def publish(epoch, e0, e1):
        (tmp_path / "D.0.json").write_text(json.dumps(
            {"epoch": epoch, "deposit": {"e0": e0, "e1": e1, "a0": 2, "a1": 6,
                                         "skip_ranks": [0]}}))

    fd = FrontDeposits(str(tmp_path))
    assert fd.assign("x", 10) is None                  # nothing published
    publish(4, 48, 100)
    assert fd.assign("a", 30) == (48, 30, 2)
    assert read_deposit_hint(str(tmp_path), "a") == {
        "epoch": 4, "e_start": 48, "n": 30, "anchor_row": 2}
    publish(6, 64, 200)                                # D slept again
    assert fd.assign("b", 30) == (64, 30, 2)           # fresh book


def test_region_is_read_from_a_capped_rank_when_rank0_publishes_none(tmp_path):
    import json

    from sglang.srt.weg2.l15_deposit import read_region

    (tmp_path / "D.2.json").write_text(json.dumps(
        {"epoch": 9, "deposit": {"e0": 16, "e1": 64, "a0": 2, "a1": 5,
                                 "skip_ranks": [0]}}))
    (tmp_path / "D.1.json").write_text(json.dumps({"epoch": 9}))
    ep, reg = read_region(str(tmp_path))
    assert ep == 9 and (reg.e0, reg.e1, reg.skip_ranks) == (16, 64, (0,))
