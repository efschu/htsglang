"""Hermetic unit tests for barlink_launch_diff (issue #616c)."""

from sglang.srt.debug_utils.barlink_launch_diff import (
    diff_ranks,
    main,
    parse_file,
    parse_line,
)

# ---------------------------------------------------------------------------
# parse_line
# ---------------------------------------------------------------------------

REAL_LINE = (
    "17:47:59 group=? rank=0/? last_op=all_gather last_nbytes=192512 "
    "unchecked=0 captured_launches=True last_op_captured=False "
    "result_i=-1 result_counter=0"
)


class TestParseLine:
    def test_real_sample_types(self):
        """A real sample line extracts all fields with correct types."""
        rec = parse_line(REAL_LINE)
        assert rec is not None
        assert rec["ts"] == "17:47:59"
        assert isinstance(rec["rank"], int)
        assert rec["rank"] == 0
        assert rec["last_op"] == "all_gather"
        assert isinstance(rec["last_nbytes"], int)
        assert rec["last_nbytes"] == 192512
        assert isinstance(rec["captured_launches"], bool)
        assert rec["captured_launches"] is True
        assert isinstance(rec["last_op_captured"], bool)
        assert rec["last_op_captured"] is False

    def test_garbage_returns_none(self):
        """parse_line returns None on a garbage line and does NOT raise."""
        assert parse_line("THIS IS NOT A VALID LOG LINE @@@") is None
        assert parse_line("") is None
        assert parse_line("   \t  ") is None
        assert parse_line("rank=0 last_op=foo") is None  # missing timestamp

    def test_parenthesized_op(self):
        """last_op=(none) is parsed correctly."""
        line = (
            "11:56:29 group=? rank=2/? last_op=(none) last_nbytes=0 "
            "unchecked=0 captured_launches=False last_op_captured=False "
            "result_i=None result_counter=None"
        )
        rec = parse_line(line)
        assert rec is not None
        assert rec["last_op"] == "(none)"
        assert rec["last_nbytes"] == 0

    def test_broadcast_line(self):
        """A broadcast line parses with correct values."""
        line = (
            "11:56:29 group=? rank=1/? last_op=broadcast last_nbytes=32 "
            "unchecked=10 captured_launches=False last_op_captured=False "
            "result_i=-1 result_counter=0"
        )
        rec = parse_line(line)
        assert rec is not None
        assert rec["last_op"] == "broadcast"
        assert rec["last_nbytes"] == 32
        assert rec["captured_launches"] is False


# ---------------------------------------------------------------------------
# parse_file
# ---------------------------------------------------------------------------


class TestParseFile:
    def test_parses_all_valid_lines(self, tmp_path):
        """parse_file returns records for every valid line, skips bad ones."""
        f = tmp_path / "log.txt"
        f.write_text(
            "11:56:29 group=? rank=0/? last_op=broadcast last_nbytes=32 "
            "unchecked=10 captured_launches=False last_op_captured=False "
            "result_i=-1 result_counter=0\n"
            "this is garbage\n"
            "11:56:30 group=? rank=0/? last_op=all_gather last_nbytes=64 "
            "unchecked=0 captured_launches=True last_op_captured=True "
            "result_i=0 result_counter=1\n"
        )
        records = parse_file(str(f))
        assert len(records) == 2
        assert records[0]["last_op"] == "broadcast"
        assert records[1]["last_op"] == "all_gather"


# ---------------------------------------------------------------------------
# diff_ranks
# ---------------------------------------------------------------------------


def _make_record(ts: str, rank: int, last_op: str, last_nbytes: int) -> dict:
    return {
        "ts": ts,
        "rank": rank,
        "last_op": last_op,
        "last_nbytes": last_nbytes,
        "captured_launches": False,
        "last_op_captured": False,
    }


class TestDiffRanks:
    def test_no_disagreement_identical(self):
        """All three ranks have identical records -> no disagreement."""
        ts1, ts2 = "11:00:01", "11:00:02"
        records_by_rank = {
            0: [
                _make_record(ts1, 0, "broadcast", 32),
                _make_record(ts2, 0, "all_gather", 64),
            ],
            1: [
                _make_record(ts1, 1, "broadcast", 32),
                _make_record(ts2, 2, "all_gather", 64),
            ],
            2: [
                _make_record(ts1, 2, "broadcast", 32),
                _make_record(ts2, 2, "all_gather", 64),
            ],
        }
        report = diff_ranks(records_by_rank)
        assert "No disagreement" in report
        assert "Compared 2" in report

    def test_first_disagreeing_timestamp_named(self):
        """rank2 diverges at the second timestamp; report names that one."""
        ts1, ts2, ts3 = "11:00:01", "11:00:02", "11:00:03"
        records_by_rank = {
            0: [
                _make_record(ts1, 0, "broadcast", 32),
                _make_record(ts2, 0, "all_gather", 64),
                _make_record(ts3, 0, "all_gather", 64),
            ],
            1: [
                _make_record(ts1, 1, "broadcast", 32),
                _make_record(ts2, 1, "all_gather", 64),
                _make_record(ts3, 1, "all_gather", 64),
            ],
            2: [
                _make_record(ts1, 2, "broadcast", 32),
                _make_record(ts2, 2, "reduce_scatter", 128),  # divergence
                _make_record(ts3, 2, "all_gather", 64),
            ],
        }
        report = diff_ranks(records_by_rank)
        assert "DISAGREEMENT at timestamp 11:00:02" in report
        # Must NOT report the third timestamp.
        assert "11:00:03" not in report

    def test_ignores_non_common_timestamps(self):
        """A timestamp only in one rank is not compared."""
        ts1, ts_only_r0 = "11:00:01", "11:00:99"
        records_by_rank = {
            0: [
                _make_record(ts1, 0, "broadcast", 32),
                _make_record(ts_only_r0, 0, "all_gather", 999),
            ],
            1: [_make_record(ts1, 1, "broadcast", 32)],
        }
        report = diff_ranks(records_by_rank)
        assert "No disagreement" in report
        assert "Compared 1" in report
        # The unique timestamp must not appear in the report.
        assert "11:00:99" not in report

    def test_multiset_differentiation(self):
        """Same ops but different counts per rank is a disagreement."""
        ts1 = "11:00:01"
        records_by_rank = {
            0: [
                _make_record(ts1, 0, "broadcast", 32),
                _make_record(ts1, 0, "all_gather", 64),
            ],
            1: [
                _make_record(ts1, 1, "broadcast", 32),
            ],
        }
        report = diff_ranks(records_by_rank)
        assert "DISAGREEMENT at timestamp 11:00:01" in report


# ---------------------------------------------------------------------------
# main (CLI)
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_writes_report(self, tmp_path, capsys):
        """main() writes a report to stdout for three temp files."""
        files = []
        for rank in range(3):
            f = tmp_path / f"rank{rank}.log"
            lines = []
            for ts in ["11:00:01", "11:00:02"]:
                lines.append(
                    f"{ts} group=? rank={rank}/? last_op=broadcast last_nbytes=32 "
                    f"unchecked=0 captured_launches=False last_op_captured=False "
                    f"result_i=-1 result_counter=0\n"
                )
            f.write_text("".join(lines))
            files.append(str(f))

        main(argv=files)
        captured = capsys.readouterr()
        assert "No disagreement" in captured.out
        assert "Compared 2" in captured.out
