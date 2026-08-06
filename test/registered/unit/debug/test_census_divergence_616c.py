"""#616c: hermetic tests for Divergence and CollectiveCensus._diff.

No CUDA, no torch, no distributed backend required.
"""

import unittest

from sglang.srt.distributed.collective_census import (
    CollectiveCensus,
    Divergence,
)


class TestDiffAllIdentical(unittest.TestCase):
    """Three identical count dicts produce zero divergences."""

    def test_three_ranks_same_counts_no_divergence(self):
        gathered = [
            {"tp.all_reduce": 129, "tp.broadcast": 45},
            {"tp.all_reduce": 129, "tp.broadcast": 45},
            {"tp.all_reduce": 129, "tp.broadcast": 45},
        ]
        result = CollectiveCensus._diff(gathered)
        self.assertEqual(
            result,
            [],
            msg="identical counts across all ranks must yield no divergences",
        )


class TestDiffOneFamilyOneRank(unittest.TestCase):
    """One family differing on a single rank yields exactly one Divergence
    whose describe() text names the family and the differing counts."""

    def test_single_divergence_identified(self):
        gathered = [
            {"tp.all_reduce": 129},
            {"tp.all_reduce": 129},
            {"tp.all_reduce": 128},  # rank 2 is behind by 1
        ]
        result = CollectiveCensus._diff(gathered)
        self.assertEqual(len(result), 1, "expected exactly one Divergence")
        div = result[0]
        self.assertEqual(div.family, "tp.all_reduce")
        self.assertEqual(list(div.counts), [129, 129, 128])
        self.assertEqual(div.leader, 129)
        self.assertEqual(div.behind, [(2, 1)])

    def test_describe_contains_family_and_counts(self):
        gathered = [
            {"tp.all_reduce": 129},
            {"tp.all_reduce": 129},
            {"tp.all_reduce": 128},
        ]
        result = CollectiveCensus._diff(gathered)
        text = result[0].describe()
        self.assertIn("tp.all_reduce", text, "describe must name the family")
        self.assertIn("129", text, "describe must mention the leader count")
        self.assertIn("128", text, "describe must mention the lagging count")


class TestDiffMissingFamily(unittest.TestCase):
    """A family present on one rank and absent on another is handled without
    raising. Per line 361 of collective_census.py:
        counts = [int((per_rank or {}).get(family, 0)) for per_rank in gathered]
    a missing key reads as zero, producing a Divergence (0 vs non-zero)."""

    def test_missing_key_treated_as_zero(self):
        gathered = [
            {"tp.all_reduce": 10},
            {},  # rank 1 never issued this collective; key absent -> count 0
        ]
        result = CollectiveCensus._diff(gathered)
        # Must not raise; must produce a divergence for 10 vs 0.
        self.assertEqual(len(result), 1)
        self.assertEqual(list(result[0].counts), [10, 0])
        self.assertEqual(result[0].behind, [(1, 10)])


class TestDiffTwoFamiliesSimultaneous(unittest.TestCase):
    """Two families diverging at once produce two Divergence entries."""

    def test_two_diverging_families(self):
        gathered = [
            {"tp.all_reduce": 130, "tp.broadcast": 50},
            {"tp.all_reduce": 128, "tp.broadcast": 47},
        ]
        result = CollectiveCensus._diff(gathered)
        self.assertEqual(len(result), 2, "expected two Divergence entries")
        # _diff iterates sorted(families), so order is deterministic.
        fams = [d.family for d in result]
        self.assertEqual(fams, ["tp.all_reduce", "tp.broadcast"])
        self.assertEqual(list(result[0].counts), [130, 128])
        self.assertEqual(list(result[1].counts), [50, 47])


class TestDivergenceDescribeStability(unittest.TestCase):
    """Divergence.describe() must be stable/deterministic for the same
    input -- a log line that changes between calls is useless for diffing."""

    def test_describe_is_idempotent(self):
        div = Divergence("tp.all_reduce", [129, 128, 130])
        first = div.describe()
        second = div.describe()
        self.assertEqual(
            first,
            second,
            "describe() must return the same string on repeated calls",
        )

    def test_describe_exact_text(self):
        div = Divergence("tp.all_reduce", [129, 129, 128])
        text = div.describe()
        # leader=129, rank 2 is behind by 1
        expected = "tp.all_reduce: counts [129, 129, 128] -- rank 2 behind by 1"
        self.assertEqual(text, expected)


class TestDiffEmptyInput(unittest.TestCase):
    """Empty input (no ranks) must not raise."""

    def test_empty_gathered_list(self):
        result = CollectiveCensus._diff([])
        self.assertEqual(result, [])

    def test_all_none_gathered(self):
        result = CollectiveCensus._diff([None, None])
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
