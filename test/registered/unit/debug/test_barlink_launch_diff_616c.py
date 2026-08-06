"""Hermetic tests for sglang.srt.debug_utils.barlink_launch_diff."""

from sglang.srt.debug_utils.barlink_launch_diff import (
    diff_ranks,
    main,
    parse_file,
    parse_line,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _line(
    ts, rank, last_op, last_nbytes, captured_launches="False", last_op_captured="False"
):
    return (
        f"{ts} group=? rank={rank}/? "
        f"last_op={last_op} last_nbytes={last_nbytes} "
        f"unchecked=0 "
        f"captured_launches={captured_launches} "
        f"last_op_captured={last_op_captured}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParseLine:
    def test_extracts_all_fields_with_correct_types(self):
        rec = parse_line(_line("12:34:56", 2, "all_gather", 192512, "True", "True"))
        assert rec is not None
        assert rec["ts"] == "12:34:56"
        assert rec["rank"] == 2
        assert isinstance(rec["rank"], int)
        assert rec["last_op"] == "all_gather"
        assert rec["last_nbytes"] == 192512
        assert isinstance(rec["last_nbytes"], int)
        assert rec["captured_launches"] is True
        assert isinstance(rec["captured_launches"], bool)
        assert rec["last_op_captured"] is True
        assert isinstance(rec["last_op_captured"], bool)

    def test_boolean_false_parsed(self):
        rec = parse_line(_line("00:00:00", 0, "broadcast", 32, "False", "False"))
        assert rec["captured_launches"] is False
        assert rec["last_op_captured"] is False

    def test_returns_none_on_garbage(self):
        assert parse_line("this is total garbage") is None
        assert parse_line("") is None
        assert parse_line("  \n  ") is None


class TestDiffRanks:
    def _write_rank_file(self, tmp_path, rank, lines):
        """Helper: write lines (with rank substituted) to a tmp file and return records."""
        content = "\n".join(ln.replace("R", str(rank)) for ln in lines)
        p = tmp_path / f"rank{rank}.log"
        p.write_text(content)
        return parse_file(str(p)), str(p)

    def test_no_disagreement_identical_logs(self, tmp_path):
        """Three identical logs should report no disagreement with a count."""
        lines = [
            _line("10:00:00", "R", "broadcast", 32),
            _line("10:00:01", "R", "all_gather", 64),
            _line("10:00:02", "R", "reduce", 128),
        ]
        records_by_rank = {}
        for r in range(3):
            recs, _ = self._write_rank_file(tmp_path, r, lines)
            records_by_rank[r] = recs
        result = diff_ranks(records_by_rank)
        assert "No disagreements" in result
        assert "Compared 3" in result

    def test_first_disagreement_reported(self, tmp_path):
        """Rank 2 differs at the SECOND of three timestamps; second is reported."""
        lines_base = [
            _line("10:00:00", "R", "broadcast", 32),
            _line("10:00:01", "R", "all_gather", 64),
            _line("10:00:02", "R", "reduce", 128),
        ]
        lines_r2 = [
            _line("10:00:00", "2", "broadcast", 32),
            _line("10:00:01", "2", "all_reduce", 999),
            _line("10:00:02", "2", "reduce", 128),
        ]
        records_by_rank = {}
        for r in range(3):
            src = lines_r2 if r == 2 else lines_base
            recs, _ = self._write_rank_file(tmp_path, r, src)
            records_by_rank[r] = recs
        result = diff_ranks(records_by_rank)
        assert "Disagreement" in result
        assert "10:00:01" in result

    def test_missing_timestamps_ignored(self, tmp_path):
        """Timestamps present in only some ranks are skipped."""
        r0_content = "\n".join(
            [
                _line("10:00:00", "0", "op0", 10),
                _line("10:00:01", "0", "op1", 20),
            ]
        )
        r1_content = "\n".join(
            [
                _line("10:00:01", "1", "op1", 20),
            ]
        )
        p0 = tmp_path / "r0.log"
        p0.write_text(r0_content)
        p1 = tmp_path / "r1.log"
        p1.write_text(r1_content)
        records_by_rank = {0: parse_file(str(p0)), 1: parse_file(str(p1))}
        result = diff_ranks(records_by_rank)
        assert "No disagreements" in result
        assert "Compared 1" in result


class TestMain:
    def test_prints_report_for_three_files(self, tmp_path, capsys):
        """main() exercises the full CLI path and prints a report."""
        lines = [
            _line("10:00:00", "R", "broadcast", 32),
            _line("10:00:01", "R", "all_gather", 64),
        ]
        files = []
        for r in range(3):
            p = tmp_path / f"rank{r}.log"
            p.write_text("\n".join(ln.replace("R", str(r)) for ln in lines))
            files.append(str(p))

        main(files)
        captured = capsys.readouterr()
        assert "No disagreements" in captured.out
        assert "Compared 2" in captured.out
