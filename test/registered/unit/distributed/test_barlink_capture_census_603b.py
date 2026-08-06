"""#603b -- the collectives baked into a CUDA graph, diffed across ranks.

WHY A SEPARATE INSTRUMENT WAS NEEDED, pinned here so the next reader does not
re-derive it: the #583 collective census counts HOST-side calls, and a
captured collective is a host call exactly once per boot -- at capture. Every
replay after that makes no host call at all, so the count census is
structurally blind to the sequence a replayed graph runs, which is where this
crash family wedges. These tests pin the three properties that make the
capture census able to answer that question:

1. IT RECORDS ONLY UNDER CAPTURE, and it records op, size AND kernel variant.
   Equal counts are not enough: the BAR1 planners turn bytes into a round
   count, and ``_kernel`` turns bytes into a cooperative-vs-single-block
   choice. Two ranks that agree on "one all_reduce here" and disagree on
   either of those still deadlock.

2. THE COMPARISON IS POSITIONAL, the same discipline as the #431 comparator
   (``barlink_uniformity.first_divergence``): bar1 sequences every collective
   on one shared device round counter and waits on flag EQUALITY, so decision
   *i* on rank A must be decision *i* on rank B. A set- or count-based
   comparison passes on two ranks running the same collectives in a different
   order, which hangs exactly as hard.

3. IT CAN FAIL. Each agreement test has a divergence twin that must be
   detected -- a comparator that cannot go red certifies nothing.

The #431 ``CollectiveDecision`` recorder is deliberately NOT reused: it has no
field for the kernel variant and no notion of which graph a decision belongs
to, and it is an unbounded host-path recorder that is off by default, whereas
this one is capture-scoped, segmented per graph and armed by default at zero
replay cost.

CPU only: no device allocation, no real process group. ``all_gather_object``
is substituted by a stand-in that serves the peers' payloads, so the real
two-phase comparison code runs exactly as it does on a rig.
"""

import unittest
from typing import List, Optional, Tuple

from sglang.srt.distributed.device_communicators import barlink_capture_census as cc
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FakeDist:
    """Serves each rank's payload to every rank, like a real all_gather_object.

    Payloads are supplied per phase, in rank order. The comparison makes two
    gathers (digests, then details for the disagreeing segments only), so the
    stand-in pops one prepared round per call and asserts it was given one --
    a silent extra gather would mean the protocol changed under the test.
    """

    def __init__(self, rounds: List[List[object]]) -> None:
        self._rounds = list(rounds)
        self.calls = 0

    def all_gather_object(self, out_list, obj, group=None):  # noqa: ARG002
        if not self._rounds:
            raise AssertionError(
                "compare_across_ranks made more all_gather_object calls than "
                "the test prepared payloads for"
            )
        payload = self._rounds.pop(0)
        self.calls += 1
        for i, value in enumerate(payload):
            out_list[i] = value


def _census(records: List[Tuple[str, str, int, Optional[int]]]) -> cc.CaptureCensus:
    """Build one rank's census from (segment, op, nbytes, variant) rows."""
    census = cc.CaptureCensus()
    open_key = None
    for segment, op, nbytes, variant in records:
        if segment != open_key:
            if open_key is not None:
                census.close_segment()
            census.open_segment(segment)
            open_key = segment
        census.note(op, nbytes, variant)
    if open_key is not None:
        census.close_segment()
    return census


#: One rank's decode graph: two layer all_reduces and a small broadcast.
UNIFORM = [
    ("full/bs1", "all_reduce", 40960, 0),
    ("full/bs1", "all_reduce", 40960, 0),
    ("full/bs1", "broadcast", 128, 0),
]


def _compare(censuses: List[cc.CaptureCensus], monkeypatched) -> List[bool]:
    """Run the real comparison on every rank against the same stand-in."""
    results = []
    for rank, census in enumerate(censuses):
        results.append(census.compare_across_ranks(monkeypatched, len(censuses), rank))
    return results


class TestCaptureCensusRecording(CustomTestCase):
    def test_records_op_size_and_variant_in_order(self):
        census = _census(UNIFORM)
        rendered = census.details(["full/bs1"])["full/bs1"]
        self.assertEqual(len(rendered), 3)
        self.assertTrue(rendered[0].startswith("all_reduce|40960|v0|"))
        self.assertTrue(rendered[2].startswith("broadcast|128|v0|"))

    def test_segments_are_kept_apart(self):
        census = _census(
            [("full/bs1", "all_reduce", 8, 0), ("full/bs2", "all_reduce", 8, 0)]
        )
        self.assertEqual(sorted(census.segments()), ["full/bs1", "full/bs2"])

    def test_two_graphs_with_the_same_shape_key_stay_apart(self):
        """The decode runner and the EAGLE draft runner can emit the SAME
        ShapeKey through the same backend. Measured on-card (boot 12:31:33,
        TP=3 NEXTN): with both merged under one key the record showed four
        segments of 210 collectives that were really eight graphs. Merging
        does not break the cross-rank diff -- both ranks merge identically --
        but it reports a position inside a sequence no single graph runs, so
        the callsite it names cannot be looked up.
        """
        census = cc.CaptureCensus()
        for _ in range(2):
            census.open_segment("full/bs1")
            census.note("all_reduce", 8, 0)
            census.close_segment()
        self.assertEqual(sorted(census.segments()), ["full/bs1", "full/bs1#2"])
        self.assertEqual(len(census.segments()["full/bs1"]), 1)
        self.assertEqual(len(census.segments()["full/bs1#2"]), 1)

    def test_records_outside_a_segment_are_counted_not_dropped(self):
        """An unsegmented capture path must be reported, never silently lost.

        A graph this module does not segment is a graph the diff cannot see,
        which is the one failure mode that would make a clean report a lie.
        """
        census = cc.CaptureCensus()
        census.note("all_reduce", 8, 0)
        self.assertEqual(census.unsegmented, 1)
        self.assertIn("<unsegmented>", census.segments())

    def test_note_is_disabled_by_the_kill_switch(self):
        import os

        previous = os.environ.get(cc.ENV_ENABLE)
        os.environ[cc.ENV_ENABLE] = "0"
        try:
            before = len(cc.capture_census().segments())
            cc.note("all_reduce", 8, 0)
            self.assertEqual(len(cc.capture_census().segments()), before)
        finally:
            if previous is None:
                os.environ.pop(cc.ENV_ENABLE, None)
            else:
                os.environ[cc.ENV_ENABLE] = previous


class TestCaptureCensusComparison(CustomTestCase):
    """The comparator, including the can-fail twin of every agreement."""

    def _run(self, per_rank: List[List[Tuple[str, str, int, Optional[int]]]]):
        censuses = [_census(rows) for rows in per_rank]
        digests = [c.digests() for c in censuses]
        bad = sorted(
            {
                k
                for k in {key for d in digests for key in d}
                if len({d.get(k, (-1, "missing")) for d in digests}) > 1
            }
        )
        rounds: List[List[object]] = [list(digests)]
        if bad:
            rounds.append([c.details(bad) for c in censuses])
        fake = _FakeDist(rounds)
        original = cc.__dict__.get("_test_dist_hook")
        import torch.distributed as dist

        saved = dist.all_gather_object
        dist.all_gather_object = fake.all_gather_object
        try:
            # Only rank 0 is run: every rank executes identical code against
            # identical gathered data, so one pass exercises the whole path.
            return censuses[0].compare_across_ranks(object(), len(censuses), 0), fake
        finally:
            dist.all_gather_object = saved
            del original

    def test_identical_sequences_agree(self):
        agree, fake = self._run([UNIFORM, UNIFORM, UNIFORM])
        self.assertTrue(agree)
        self.assertEqual(fake.calls, 1, "an agreeing boot must not ship details")

    def test_different_kernel_variant_is_caught(self):
        """THE #603b SHAPE. Same op, same size, different recorded kernel.

        This is the divergence no other instrument in the tree can see: the
        census counts one call either way, and the launch record's unchecked
        counter does not advance under capture at all.
        """
        diverging = list(UNIFORM)
        diverging[2] = ("full/bs1", "broadcast", 128, 1)
        agree, fake = self._run([UNIFORM, UNIFORM, diverging])
        self.assertFalse(agree)
        self.assertEqual(fake.calls, 2, "a divergence must ship the details")

    def test_different_payload_size_is_caught(self):
        diverging = list(UNIFORM)
        diverging[0] = ("full/bs1", "all_reduce", 40961, 0)
        agree, _ = self._run([UNIFORM, diverging, UNIFORM])
        self.assertFalse(agree)

    def test_different_collective_count_is_caught(self):
        agree, _ = self._run([UNIFORM, UNIFORM[:2], UNIFORM])
        self.assertFalse(agree)

    def test_reordered_sequence_is_caught(self):
        """Same multiset, different order -- hangs as hard, must go red.

        The positional discipline of the #431 comparator, restated for
        capture-time records: bar1 waits on flag EQUALITY over one shared
        round counter, so position is the identity of a collective.
        """
        reordered = [UNIFORM[2], UNIFORM[0], UNIFORM[1]]
        agree, _ = self._run([UNIFORM, reordered, UNIFORM])
        self.assertFalse(agree)

    def test_missing_segment_on_one_rank_is_caught(self):
        extra = UNIFORM + [("full/bs2", "all_reduce", 40960, 0)]
        agree, _ = self._run([extra, UNIFORM, extra])
        self.assertFalse(agree)


class TestCaptureCensusDiscipline(CustomTestCase):
    """It is an instrument: it must never be the reason a boot fails."""

    def test_comparison_never_raises(self):
        census = _census(UNIFORM)

        class _Boom:
            def __getattr__(self, name):
                raise RuntimeError("gloo group is gone")

        import torch.distributed as dist

        saved = dist.all_gather_object

        def _explode(*args, **kwargs):
            raise RuntimeError("collective failed")

        dist.all_gather_object = _explode
        try:
            # Reports "agree" on failure by design: an instrument that cannot
            # make its measurement must not manufacture a divergence.
            self.assertTrue(census.compare_across_ranks(_Boom(), 3, 0))
        finally:
            dist.all_gather_object = saved

    def test_dump_never_raises_on_an_unwritable_directory(self):
        census = _census(UNIFORM)
        self.assertIsNone(census.dump_to_file(0, "/proc/nonexistent/nope"))

    def test_single_rank_is_trivially_uniform(self):
        census = _census(UNIFORM)
        self.assertTrue(census.compare_across_ranks(object(), 1, 0))


if __name__ == "__main__":
    unittest.main()
