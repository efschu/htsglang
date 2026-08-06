"""Hermetic unit tests for bar1_flag_align (issue 616c)."""

from __future__ import annotations

import sglang.srt.debug_utils.bar1_flag_align as mod


# ---------------------------------------------------------------------------
# Helpers: build snapshot dicts without touching the file system
# ---------------------------------------------------------------------------


def _snap(rank: int, world: int, group: str, values: dict[int, int]) -> dict:
    return {"rank": rank, "world": world, "group": group, "values": values}


_REAL_RANK0 = (
    "barlink-BAR1 abort flag snapshot rank 0/3 group tp:0: "
    "64 lines of 2097152 bytes, first dword per line -- "
    "0:0 1:285932 2:285932 3:0 4:285932 5:285932 6:0 7:0 "
    "8:283536 9:0 10:0 11:283536 12:0 13:0 14:283536 15:0 "
    "16:0 17:283536 18:0 19:285934 20:285935 21:0 "
    "22:0 23:0 24:0 25:0 26:0 27:0 28:0 29:0 30:0 31:0 "
    "32:0 33:0 34:0 35:0 36:0 37:0 38:0 39:0 40:0 41:0 "
    "42:0 43:0 44:0 45:0 46:0 47:0 48:0 49:0 50:0 51:0 "
    "52:0 53:0 54:0 55:0 56:0 57:0 58:0 59:0 60:0 61:0 "
    "62:0 63:0. Compare against the peers'."
)


# ---------------------------------------------------------------------------
# Test 1: parse_snapshot pulls rank, world, group and the value map
# ---------------------------------------------------------------------------


def test_parse_snapshot_realistic():
    result = mod.parse_snapshot(_REAL_RANK0)
    assert result is not None
    assert result["rank"] == 0
    assert result["world"] == 3
    assert result["group"] == "tp:0"
    # Spot-check a few values
    assert result["values"][1] == 285932
    assert result["values"][8] == 283536
    assert result["values"][0] == 0
    assert result["values"][20] == 285935


# ---------------------------------------------------------------------------
# Test 2: parse_snapshot returns None on garbage
# ---------------------------------------------------------------------------


def test_parse_snapshot_garbage():
    assert mod.parse_snapshot("this is not a snapshot line") is None
    assert mod.parse_snapshot("") is None
    assert mod.parse_snapshot("barlink-BAR1 abort flag snapshot") is None
    assert mod.parse_snapshot(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test 3: align() maps line index -> (block, sender) correctly for world=3
# ---------------------------------------------------------------------------


def test_align_line_mapping():
    # Minimal snapshots: one rank, a few values
    snaps = [_snap(0, 3, "tp:0", {5: 100, 20: 200})]
    aligned = mod.align(snaps, world=3)

    # line 5 = block 1 sender 2   (5 // 3 = 1, 5 % 3 = 2)
    entry_b1s2 = [e for e in aligned if e["block"] == 1 and e["sender"] == 2][0]
    assert entry_b1s2["by_rank"][0] == 100

    # line 20 = block 6 sender 2  (20 // 3 = 6, 20 % 3 = 2)
    entry_b6s2 = [e for e in aligned if e["block"] == 6 and e["sender"] == 2][0]
    assert entry_b6s2["by_rank"][0] == 200


# ---------------------------------------------------------------------------
# Test 4: report() flags a disagreement when ranks differ in the SAME cell
# ---------------------------------------------------------------------------


def test_report_flags_disagreement():
    # Two ranks, same (block=0, sender=1) -> line 1
    # rank0 has 100, rank1 has 200 -> disagreement, spread 100
    snaps = [
        _snap(0, 2, "tp:0", {1: 100}),
        _snap(1, 2, "tp:0", {1: 200}),
    ]
    text = mod.report(snaps, world=2)
    assert "DISAGREEMENTS" in text
    assert "spread=100" in text
    assert "(block=0, sender=1)" in text


# ---------------------------------------------------------------------------
# Test 5: report() does NOT flag a disagreement merely because two ranks
# have different maxima across DIFFERENT cells.  This is the exact error
# the tool exists to prevent.
#
# Real example restricted to ring blocks (0, 1, 6), which all agree:
#   rank0 nonzero ring: 1:285932 2:285932  4:285932 5:285932
#                     6:283536       11:283536   14:283536 17:283536
#   rank1 nonzero ring: 0:285938  3:285937 2:285937 5:285937
#                     6:283536       9:283536 12:283536   15:283536
#
# rank0 max=285932, rank1 max=285938  (different)
# But aligned cell-by-cell they all agree (285937 vs 285932 difference is
# because rank1 has values in its OWN sender slots which rank0 sees as 0).
#
# We build a simplified version: two ranks on the SAME line, same value.
# ---------------------------------------------------------------------------


def test_report_no_false_disagreement_different_cells():
    """Different ranks write different lines -- no disagreement.

    rank0 holds value 99 on line 0 (block 0 sender 0).
    rank1 holds value 199 on line 1 (block 0 sender 1).
    These are different (block,sender) cells -> must NOT be flagged.
    """
    snaps = [
        _snap(0, 2, "tp:0", {0: 99}),
        _snap(1, 2, "tp:0", {1: 199}),
    ]
    text = mod.report(snaps, world=2)
    assert "DISAGREEMENTS" not in text
    assert "No disagreements found" in text


# ---------------------------------------------------------------------------
# Test 6: Cells where only one rank holds a value are NOT disagreements.
# ---------------------------------------------------------------------------


def test_report_single_rank_value():
    """Only rank0 has a nonzero value at (block=0, sender=1).

    rank1 has 0 there.  This is not a disagreement.
    """
    snaps = [
        _snap(0, 3, "tp:0", {1: 42}),
        _snap(1, 3, "tp:0", {1: 0}),
        _snap(2, 3, "tp:0", {1: 0}),
    ]
    text = mod.report(snaps, world=3)
    assert "DISAGREEMENTS" not in text
    assert "No disagreements found" in text
