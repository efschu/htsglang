"""#583: the collective census names the family and the rank that skipped.

WHAT THIS PINS
--------------
Crashes 9, 10 and 11 (2026-08-05) all ended with the same useless shape:
ranks 0 and 1 standing in one collective, rank 2 standing in a different
one, and nothing anywhere saying which collective rank 2 had SKIPPED. The
counts existed per rank (the prefill line already prints
``tp.all_reduce 243.4/129x``); they were simply never compared ACROSS ranks.

These tests pin the comparison: given per-rank counts that differ by one on
one family -- the exact arithmetic of crashes 9 and 11, where rank 2 issued
128 ``tp.all_reduce`` against its peers' 129 -- the census must report the
FAMILY, the per-rank counts, and which rank is behind and by how much.

They also pin the two properties that decide whether the instrument is
present when it is needed:

  * it is ARMED BY DEFAULT (an instrument you must switch on before a crash
    explains no crashes);
  * it WARNS AND NEVER RAISES, including when the group is wedged or absent
    -- it must never become the reason a healthy forward fails, nor mask the
    defect it is watching for.

Hermetic: no CUDA, no process group, no model. The cross-rank gather is
driven through a fake that behaves like ``all_gather_object``.
"""

import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.distributed import collective_census as cc  # noqa: E402
from sglang.srt.distributed.collective_census import (  # noqa: E402
    CollectiveCensus,
    census_enabled,
    census_interval,
    format_local_census,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fake_all_gather_object(per_rank, boom=None):
    """Stands in for ``torch.distributed.all_gather_object``: fills the output
    list with every rank's dict, or fails the way a wedged group would."""

    def _impl(out, obj, group=None):
        if boom is not None:
            raise boom
        for i, d in enumerate(per_rank):
            out[i] = d

    return _impl


def _compare(per_rank, rank=0, boom=None):
    """Run one comparison as `rank` would see it. Returns (result, census)."""
    import torch.distributed as dist

    c = CollectiveCensus()
    for k, v in per_rank[rank].items():
        for _ in range(v):
            c.bump(k)
    with mock.patch.object(
        dist, "all_gather_object", _fake_all_gather_object(per_rank, boom)
    ):
        found = c.compare_across_ranks(object(), len(per_rank), rank)
    return found, c


#: Crashes 9 and 11, in numbers: 64 layers x 2 + 1 = 129 tp all-reduces per
#: forward, and rank 2 one short.
CRASH_9_COUNTS = [
    {"tp.all_reduce": 129, "dcp.all_gather": 16, "tp.all_gather": 1},
    {"tp.all_reduce": 129, "dcp.all_gather": 16, "tp.all_gather": 1},
    {"tp.all_reduce": 128, "dcp.all_gather": 16, "tp.all_gather": 1},
]


class CollectiveCensusTest(unittest.TestCase):
    # -- THE FALSIFIER: an injected divergence must be named --------------

    def test_the_one_missing_all_reduce_is_named(self):
        found, _ = _compare(CRASH_9_COUNTS)
        self.assertIsNotNone(found)
        self.assertEqual(len(found), 1, msg=f"expected one family, got {found}")
        d = found[0]
        self.assertEqual(d.family, "tp.all_reduce")
        self.assertEqual(d.counts, (129, 129, 128))
        # The whole point: WHICH rank, and by HOW MUCH.
        self.assertEqual(d.behind, [(2, 1)])
        text = d.describe()
        self.assertIn("tp.all_reduce", text)
        self.assertIn("rank 2 behind by 1", text)

    def test_agreeing_ranks_report_nothing(self):
        """The detector must be able to say NO -- otherwise the test above
        would pass against something that always reports a divergence."""
        agree = [dict(CRASH_9_COUNTS[0]) for _ in range(3)]
        found, _ = _compare(agree)
        self.assertEqual(found, [])

    def test_a_family_missing_entirely_on_one_rank_still_diverges(self):
        """A rank that skipped the ONLY collective of some family has no key
        for it at all. That is the case most worth catching, so a missing key
        must read as zero rather than drop the family from the diff."""
        per_rank = [
            {"tp.all_reduce": 129, "tp.all_gather": 1},
            {"tp.all_reduce": 129, "tp.all_gather": 1},
            {"tp.all_reduce": 129},  # never issued the final gather
        ]
        found, _ = _compare(per_rank)
        self.assertEqual([d.family for d in found], ["tp.all_gather"])
        self.assertEqual(found[0].behind, [(2, 1)])

    def test_it_does_not_care_which_rank_is_the_odd_one_out(self):
        """Rank 0 being the straggler must read the same way as rank 2."""
        per_rank = [
            {"tp.all_reduce": 128},
            {"tp.all_reduce": 129},
            {"tp.all_reduce": 129},
        ]
        found, _ = _compare(per_rank)
        self.assertEqual(found[0].behind, [(0, 1)])

    def test_every_rank_reaches_the_same_verdict(self):
        """All ranks gather the same vector, so the report is rank-uniform --
        the instrument must not itself become a source of disagreement."""
        seen = [_compare(CRASH_9_COUNTS, rank=r)[0][0].behind for r in range(3)]
        self.assertEqual(seen, [[(2, 1)], [(2, 1)], [(2, 1)]])

    # -- warn-never-raise --------------------------------------------------

    def test_a_wedged_or_failing_group_does_not_raise(self):
        found, c = _compare(CRASH_9_COUNTS, boom=RuntimeError("gloo is gone"))
        self.assertIsNone(found)
        self.assertEqual(c.comparisons, 0)

    def test_no_group_and_single_rank_are_silent_no_ops(self):
        c = CollectiveCensus()
        self.assertIsNone(c.compare_across_ranks(None, 3, 0))
        self.assertIsNone(c.compare_across_ranks(object(), 1, 0))

    # -- cadence -----------------------------------------------------------

    def test_the_cadence_gate_is_the_replicated_round_counter(self):
        """It must open on a round number, never on the counts -- a cadence
        read off local state would itself be a rank-local test before a group
        collective."""
        c = CollectiveCensus()
        due = [(c.next_round(), c.due(4))[1] for _ in range(8)]
        self.assertEqual(due, [False, False, False, True] * 2)

    def test_a_non_positive_interval_disables_only_the_comparison(self):
        c = CollectiveCensus()
        c.next_round()
        self.assertFalse(c.due(0))
        c.bump("tp.all_reduce")
        # Counting stays armed, so the abort-time dump still has content.
        self.assertEqual(c.snapshot(), {"tp.all_reduce": 1})

    # -- armed by default --------------------------------------------------

    def test_armed_by_default(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertTrue(census_enabled())
            self.assertEqual(census_interval(), cc.DEFAULT_INTERVAL)

    def test_the_kill_switch_works(self):
        with mock.patch.dict("os.environ", {cc.ENV_ENABLE: "0"}):
            self.assertFalse(census_enabled())

    def test_a_junk_interval_falls_back_instead_of_raising(self):
        with mock.patch.dict("os.environ", {cc.ENV_INTERVAL: "not-a-number"}):
            self.assertEqual(census_interval(), cc.DEFAULT_INTERVAL)

    # -- the abort-time dump ----------------------------------------------

    def test_the_local_dump_takes_no_collective_and_names_the_counts(self):
        """The wedge-proof half: usable when the peers are already dead."""
        cc._CENSUS._counts.clear()
        for _ in range(128):
            cc._CENSUS.bump("tp.all_reduce")
        line = format_local_census(rank=2)
        self.assertIn("rank 2", line)
        self.assertIn("tp.all_reduce 128x", line)
        cc._CENSUS._counts.clear()

    def test_the_local_dump_is_honest_when_nothing_was_counted(self):
        cc._CENSUS._counts.clear()
        self.assertIn("no collectives counted", format_local_census(rank=0))


if __name__ == "__main__":
    unittest.main()
