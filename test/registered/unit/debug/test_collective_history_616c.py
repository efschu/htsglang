"""#616c: ring-buffer history of the last N collectives per rank.

Hermetic: no CUDA, no torch, no distributed backend required.
"""

import unittest

from sglang.srt.distributed.collective_census import (
    CollectiveCensus,
    format_local_history,
)


class TestRingBufferBounds(unittest.TestCase):
    """Test that bumping N+10 times keeps only the last DEFAULT_HISTORY_LEN
    entries."""

    def test_n_plus_10_keeps_only_last_n(self):
        from sglang.srt.distributed.collective_census import DEFAULT_HISTORY_LEN

        c = CollectiveCensus()
        first_family = "alpha"
        # Bump DEFAULT_HISTORY_LEN + 10 times.
        for i in range(DEFAULT_HISTORY_LEN + 10):
            family = first_family if i == 0 else f"family_{i}"
            c.bump(family, nbytes=i * 100)

        # The formatted output reports exactly DEFAULT_HISTORY_LEN entries.
        line = c.format_local_history(rank=0)
        self.assertIn(
            f"{DEFAULT_HISTORY_LEN} entries",
            line,
            msg=f"expected '{DEFAULT_HISTORY_LEN} entries' in output",
        )

        # The very first bumped family:nbytes pair must be absent.
        self.assertNotIn(
            "alpha:0",
            line,
            msg="the first bumped entry must have been evicted from the ring buffer",
        )


class TestOldestFirstOrder(unittest.TestCase):
    """Test that formatted history lists entries in the order they were bumped."""

    def test_three_distinct_families_appear_in_order(self):
        c = CollectiveCensus()
        c.bump("aaa_first", nbytes=10)
        c.bump("bbb_second", nbytes=20)
        c.bump("ccc_third", nbytes=30)

        line = c.format_local_history(rank=0)
        pos_a = line.index("aaa_first:10")
        pos_b = line.index("bbb_second:20")
        pos_c = line.index("ccc_third:30")

        self.assertLess(pos_a, pos_b, "aaa_first must appear before bbb_second")
        self.assertLess(pos_b, pos_c, "bbb_second must appear before ccc_third")


class TestFormattedLineContents(unittest.TestCase):
    """Test that the formatted line contains the rank number and entry count."""

    def test_rank_and_count_present(self):
        c = CollectiveCensus()
        c.bump("ar", nbytes=64)
        c.bump("bg", nbytes=128)
        c.bump("ar", nbytes=64)

        line = c.format_local_history(rank=7)
        self.assertIn("rank 7", line)
        self.assertIn("3 entries", line)


class TestInstanceIsolation(unittest.TestCase):
    """Test that two CollectiveCensus instances do not share history."""

    def test_separate_instances_independent(self):
        a = CollectiveCensus()
        b = CollectiveCensus()

        a.bump("family_a", nbytes=1)

        # Instance b must report no entries because only a was bumped.
        line_b = b.format_local_history(rank=1)
        self.assertIn("0 entries", line_b)
        self.assertIn("no entries recorded", line_b)

        # Instance a must report exactly its own entry.
        line_a = a.format_local_history(rank=0)
        self.assertIn("family_a:1", line_a)
        self.assertNotIn("0 entries", line_a)


class TestEmptyHistory(unittest.TestCase):
    """Test that an instance with no bumps reports the no-entries wording."""

    def test_no_bumps_shows_no_entries_recorded(self):
        c = CollectiveCensus()
        line = c.format_local_history(rank=42)
        self.assertIn("42", line)
        self.assertIn("no entries recorded", line)


class TestNbytesDefault(unittest.TestCase):
    """Test that nbytes defaults to 0 and appears as ':0' in the output."""

    def test_bump_without_nbytes_defaults_to_zero(self):
        c = CollectiveCensus()
        c.bump("tp.all_reduce")  # no nbytes argument

        line = c.format_local_history(rank=0)
        self.assertIn(
            "tp.all_reduce:0",
            line,
            msg="bump without nbytes must default to 0 and show ':0'",
        )


class TestModuleLevelFormatLocalHistory(unittest.TestCase):
    """Test the module-level format_local_history function delegates correctly."""

    def setUp(self):
        # Import the singleton to clear it for each test.
        from sglang.srt.distributed import collective_census as cc

        self._cc = cc
        cc._CENSUS._counts.clear()
        cc._CENSUS._history.clear()

    def tearDown(self):
        self._cc._CENSUS._counts.clear()
        self._cc._CENSUS._history.clear()

    def test_module_level_delegates_to_singleton(self):
        self._cc._CENSUS.bump("ar", nbytes=128)
        line = format_local_history(rank=3)
        self.assertIn("rank 3", line)
        self.assertIn("ar:128", line)


if __name__ == "__main__":
    unittest.main()
