"""Hermetic unit tests for rank_phase_summary (issue #616c)."""

from __future__ import annotations

import textwrap

import pytest

from sglang.srt.debug_utils.rank_phase_summary import (
    main,
    parse_rank_batch_line,
    report,
    summarize,
)

# ---------------------------------------------------------------------------
# Fixture: canonical synthetic log line
# ---------------------------------------------------------------------------

_LINE = (
    "[2026-08-06 19:15:53 TP0] Prefill rank batch, "
    "#new-token: 53, #cached-token: 0, #chunks: 1, "
    "gpu-ms: 111.0 (compute 32.8, wait 78.1) "
    "(wait by family: tp.all_reduce 62.5/129x, dcp.all_gather 5.3/16x, tp.all_gather 0.1/1x)"
)

_LINE_TP1 = (
    "[2026-08-06 19:15:54 TP1] Prefill rank batch, "
    "#new-token: 10, #cached-token: 0, #chunks: 1, "
    "gpu-ms: 200.0 (compute 50.0, wait 150.0) "
    "(wait by family: tp.all_reduce 140.0/100x, dcp.all_gather 10.0/10x)"
)

_LINE_TP1_SECOND = (
    "[2026-08-06 19:16:00 TP1] Prefill rank batch, "
    "#new-token: 20, #cached-token: 0, #chunks: 1, "
    "gpu-ms: 300.0 (compute 100.0, wait 200.0) "
    "(wait by family: tp.all_reduce 180.0/200x, dcp.all_gather 20.0/20x)"
)


class TestParseRankBatchLine:
    """Test parse_rank_batch_line extraction."""

    def test_extracts_all_fields(self):
        d = parse_rank_batch_line(_LINE)
        assert d is not None
        assert d["ts"] == "2026-08-06 19:15:53"
        assert d["rank"] == "TP0"
        assert d["gpu_ms"] == 111.0
        assert d["compute_ms"] == 32.8
        assert d["wait_ms"] == 78.1
        assert isinstance(d["wait_by_family"], dict)
        assert d["wait_by_family"]["tp.all_reduce"] == (62.5, 129)
        assert d["wait_by_family"]["dcp.all_gather"] == (5.3, 16)
        assert d["wait_by_family"]["tp.all_gather"] == (0.1, 1)
        # Check numeric types
        assert isinstance(d["gpu_ms"], float)
        assert isinstance(d["wait_by_family"]["tp.all_reduce"][0], float)
        assert isinstance(d["wait_by_family"]["tp.all_reduce"][1], int)

    def test_returns_none_on_garbage(self):
        assert parse_rank_batch_line("totally unrelated log line") is None
        assert parse_rank_batch_line("") is None
        assert (
            parse_rank_batch_line("[2026-08-06 19:15:53 TP0] some other message")
            is None
        )

    def test_no_raise_on_malformed_families(self):
        bad = (
            "[2026-08-06 19:15:53 TP0] Prefill rank batch, "
            "gpu-ms: 50.0 (compute 10.0, wait 40.0) "
            "(wait by family: BROKEN)"
        )
        d = parse_rank_batch_line(bad)
        # Should parse the main fields, family section is empty because regex fails
        assert d is not None
        assert d["gpu_ms"] == 50.0
        assert d["wait_by_family"] == {}

    def test_missing_family_section(self):
        no_fam = (
            "[2026-08-06 19:15:53 TP0] Prefill rank batch, "
            "gpu-ms: 50.0 (compute 10.0, wait 40.0)"
        )
        d = parse_rank_batch_line(no_fam)
        assert d is not None
        assert d["gpu_ms"] == 50.0
        assert d["wait_by_family"] == {}


class TestSummarize:
    """Test summarize aggregation."""

    def test_aggregates_per_rank(self):
        # TP0 has 1 sample, TP1 has 2 samples
        lines = [_LINE, _LINE_TP1, _LINE_TP1_SECOND]
        s = summarize(lines)
        assert "TP0" in s
        assert "TP1" in s
        assert len(s) == 2

        # TP0: single sample
        assert s["TP0"]["count"] == 1
        assert s["TP0"]["mean_gpu_ms"] == pytest.approx(111.0)
        assert s["TP0"]["max_gpu_ms"] == pytest.approx(111.0)

        # TP1: two samples, hand-computed means
        assert s["TP1"]["count"] == 2
        assert s["TP1"]["mean_gpu_ms"] == pytest.approx((200.0 + 300.0) / 2)  # 250.0
        assert s["TP1"]["max_gpu_ms"] == pytest.approx(300.0)
        assert s["TP1"]["mean_compute_ms"] == pytest.approx((50.0 + 100.0) / 2)  # 75.0
        assert s["TP1"]["mean_wait_ms"] == pytest.approx((150.0 + 200.0) / 2)  # 175.0

        # TP1 total family waits
        fam = s["TP1"]["total_wait_by_family"]
        assert fam["tp.all_reduce"] == (140.0 + 180.0, 100 + 200)  # (320.0, 300)
        assert fam["dcp.all_gather"] == (10.0 + 20.0, 10 + 20)  # (30.0, 30)

    def test_skips_unparseable_lines(self):
        lines = [_LINE, "garbage", _LINE_TP1]
        s = summarize(lines)
        assert len(s) == 2  # TP0 and TP1

    def test_empty_input(self):
        assert summarize([]) == {}


class TestReport:
    """Test report output content."""

    def _setup_summary(self):
        """Create a summary where TP1 has clearly higher mean wait."""
        return {
            "TP0": {
                "count": 1,
                "mean_gpu_ms": 111.0,
                "max_gpu_ms": 111.0,
                "mean_compute_ms": 32.8,
                "mean_wait_ms": 78.1,
                "total_wait_by_family": {
                    "tp.all_reduce": (62.5, 129),
                    "dcp.all_gather": (5.3, 16),
                },
            },
            "TP1": {
                "count": 1,
                "mean_gpu_ms": 200.0,
                "max_gpu_ms": 200.0,
                "mean_compute_ms": 50.0,
                "mean_wait_ms": 150.0,
                "total_wait_by_family": {
                    "tp.all_reduce": (140.0, 100),
                    "dcp.all_gather": (10.0, 10),
                },
            },
        }

    def test_names_highest_mean_wait_rank(self):
        summary = self._setup_summary()
        r = report(summary)
        # TP1 has higher mean_wait_ms (150 vs 78.1)
        assert "TP1" in r
        assert "150.0" in r

    def test_names_largest_total_wait_family(self):
        summary = self._setup_summary()
        r = report(summary)
        # tp.all_reduce: 62.5+140.0=202.5, dcp.all_gather: 5.3+10.0=15.3
        assert "tp.all_reduce" in r
        # Check it is named as the dominant family
        assert "Family with largest total wait: tp.all_reduce" in r

    def test_contains_barrier_sentence(self):
        summary = self._setup_summary()
        r = report(summary)
        assert "slowest rank sets the pace at each barrier" in r

    def test_empty_summary(self):
        r = report({})
        assert "No timing data found" in r


class TestMain:
    """Test CLI main() via file + capsys."""

    def test_main_prints_report(self, tmp_path, capsys):
        log_content = textwrap.dedent(
            "[2026-08-06 19:15:53 TP0] Prefill rank batch, "
            "gpu-ms: 111.0 (compute 32.8, wait 78.1) "
            "(wait by family: tp.all_reduce 62.5/129x)\n"
        )
        log_file = tmp_path / "server.log"
        log_file.write_text(log_content)

        main([str(log_file)])
        captured = capsys.readouterr()
        assert "Per-Rank Phase Timing Summary" in captured.out
        assert "TP0" in captured.out
        assert "CONCLUSION" in captured.out

    def test_main_exits_on_no_args(self, capsys):
        with pytest.raises(SystemExit):
            main([])
