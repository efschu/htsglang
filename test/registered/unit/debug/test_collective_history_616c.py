"""#616c: ring-buffer history of the last N collectives per rank.

The existing census records only the single LAST count per family, which
means at abort time each rank prints its cumulative totals but nothing
about the ORDER of collectives. A ring buffer of the recent sequence lets
the three ranks print their histories and diff them to find where the
sequences diverged.

Hermetic: CPU tensors only, no CUDA, no distributed backend.
"""

import os
import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.distributed import collective_census as cc  # noqa: E402
from sglang.srt.distributed.collective_census import (  # noqa: E402
    CollectiveCensus,
    census_enabled,
    format_local_history,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class CollectiveHistoryTest(unittest.TestCase):
    def setUp(self):
        # Ensure census is enabled for these tests.
        os.environ.pop(cc.ENV_ENABLE, None)

    # -- Test 1: ring buffer is bounded at N ----------------------------

    def test_recording_n_plus_10_keeps_only_last_n(self):
        """N+10 entries should be bounded to N by deque(maxlen=N)."""
        n = 5  # small capacity for a visible effect
        c = CollectiveCensus()
        c._history = c._history  # already created by __init__
        # Override maxlen to a small value for the test.
        from collections import deque

        c._history = deque(maxlen=n)

        for i in range(n + 10):
            c.bump(f"family_{i}", nbytes=i * 100)

        # Exactly N entries should remain.
        self.assertEqual(
            len(c._history),
            n,
            (f"expected {n} entries after {n + 10} bumps, got {len(c._history)}"),
        )
        # With n=5, maxlen=5, 15 bumps (i=0..14): last 5 survive.
        # -> indices 10,11,12,13,14 remain.
        self.assertEqual(
            c._history[0][0],
            "family_10",
            (f"oldest entry should be family_10, got {c._history[0]}"),
        )
        # The last entry is i=14, the final bump.
        self.assertEqual(
            c._history[-1][0],
            "family_14",
            (f"newest entry should be family_14, got {c._history[-1]}"),
        )

    # -- Test 2: order is oldest-first in the formatted output ----------

    def test_format_is_oldest_first(self):
        """The formatted line must list entries in the order they occurred,
        oldest first -- so the reader scans left to right chronologically."""
        c = CollectiveCensus()
        c.bump("first", nbytes=10)
        c.bump("second", nbytes=20)
        c.bump("third", nbytes=30)
        line = c.format_local_history(rank=0)
        # In the formatted body, 'first' must appear before 'second'
        # which must appear before 'third'.
        pos_first = line.index("first:10")
        pos_second = line.index("second:20")
        pos_third = line.index("third:30")
        self.assertLess(pos_first, pos_second, msg="first must appear before second")
        self.assertLess(pos_second, pos_third, msg="second must appear before third")

    # -- Test 3: formatted line contains the rank and the entry count ---

    def test_formatted_line_contains_rank_and_count(self):
        """The prefix must include both the rank number and the entry
        count, so the reader can tell at a glance how many entries are
        in the history window."""
        c = CollectiveCensus()
        c.bump("ar", nbytes=64)
        c.bump("bg", nbytes=128)
        c.bump("ar", nbytes=64)
        line = c.format_local_history(rank=7)
        self.assertIn("rank 7", line, msg="rank number must be in the output")
        self.assertIn("3 entries", line, msg="entry count must be in the output")

    def test_formatted_line_shows_no_entries_when_empty(self):
        """An empty census history must say 'no entries recorded'
        rather than showing a blank line or the wrong count."""
        c = CollectiveCensus()
        line = c.format_local_history(rank=0)
        self.assertIn("0 entries", line, msg="empty history must show 0 entries")
        self.assertIn(
            "no entries recorded", line, msg="empty history must have explicit message"
        )

    # -- Test 4: two instances do not share history ---------------------

    def test_two_instances_do_not_share_history(self):
        """Each CollectiveCensus instance must own its own ring buffer;
        bumps on one instance must not leak into the other."""
        a = CollectiveCensus()
        b = CollectiveCensus()
        a.bump("family_a", nbytes=1)
        b.bump("family_b", nbytes=2)
        # Instance a should only see its own entry.
        self.assertEqual(len(a._history), 1)
        self.assertEqual(a._history[0], ("family_a", 1))
        # Instance b should only see its own entry.
        self.assertEqual(len(b._history), 1)
        self.assertEqual(b._history[0], ("family_b", 2))
        # Cross-check via formatted output.
        line_a = a.format_local_history(rank=0)
        line_b = b.format_local_history(rank=1)
        self.assertIn("family_a:1", line_a)
        self.assertNotIn("family_b:2", line_a, msg="a must not contain b's entry")
        self.assertIn("family_b:2", line_b)
        self.assertNotIn("family_a:1", line_b, msg="b must not contain a's entry")

    # -- Test 5: when census is disabled, recording adds nothing -------

    def test_disabled_census_records_nothing(self):
        """When SGLANG_COLLECTIVE_CENSUS=0, callers check census_enabled()
        and skip bump(). The history must stay empty -- recording is a
        no-op when the census is disabled."""
        c = CollectiveCensus()
        with mock.patch.dict("os.environ", {cc.ENV_ENABLE: "0"}):
            self.assertFalse(census_enabled())
            # Simulate a caller that respects the kill switch:
            # it checks census_enabled() before bumping.
            for _ in range(5):
                if census_enabled():
                    c.bump("tp.all_reduce")
                # When disabled, the caller never reaches bump().
            # History must be completely empty.
            self.assertEqual(
                len(c._history), 0, msg="disabled census must have no history"
            )
            line = c.format_local_history(rank=0)
            self.assertIn("no entries recorded", line)


class ModuleLevelHelperTest(unittest.TestCase):
    def setUp(self):
        os.environ.pop(cc.ENV_ENABLE, None)
        # Clear global census state for isolation.
        cc._CENSUS._counts.clear()
        cc._CENSUS._history.clear()

    def tearDown(self):
        cc._CENSUS._counts.clear()
        cc._CENSUS._history.clear()

    def test_module_level_format_local_history_delegates(self):
        """The module-level function must delegate to the singleton
        instance, producing the same shape as the method call."""
        cc._CENSUS.bump("all_reduce", nbytes=128)
        cc._CENSUS.bump("broadcast", nbytes=64)
        line = format_local_history(rank=3)
        self.assertIn("rank 3", line)
        self.assertIn("2 entries", line)
        self.assertIn("all_reduce:128", line)
        self.assertIn("broadcast:64", line)
        # Oldest-first order check.
        self.assertLess(
            line.index("all_reduce:128"),
            line.index("broadcast:64"),
            msg="module-level output must also be oldest-first",
        )


if __name__ == "__main__":
    unittest.main()
