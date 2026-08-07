"""#639: the DCP extend prefix-length vector is checked for rank-uniformity
once per extend batch, and a divergent one is refused by name.

WHAT THIS PINS, and what it deliberately does NOT claim
-------------------------------------------------------
``weightless_has_prefix`` (lockstep.py:139-143) reduces to
``any(extend_prefix_lens_cpu)`` and its docstring states the requirement it
depends on: "The answer decides whether the Q all-gather and the LSE merge are
issued at all, so it must be identical on the head rank and on every
weightless worker. Both inputs are replicated." The requirement is stated; the
premise was never checked, and it is false. The vector's provenance is one
line:

    schedule_batch.py  prefix_lens = [len(r.prefix_indices) for r in reqs]

-- the content of THIS rank's radix cache, which is exactly the quantity #616B
had to install a group-MIN floor to keep uniform. The line two above it builds
``extend_num_tokens`` from the same expression, so ONE vector decides both the
SHAPE of every per-layer TP collective and WHICH DCP collectives run at all.

Caught live 2026-08-07 08:26 with all three ranks alive:

    TP0        qwen3_5.py:1241 prepare_mlp -> ALL_REDUCE   (past attention)
    TP1, TP2   qwen3_5.py:1234 self_attention -> _forward_extend_dcp
               -> cp_lse_ag_out_ar_mha_uneven -> _ag_lse -> ALL_GATHER

Seven lines apart in one layer body, on two different group collectives. TP0
had taken ``_forward_extend_dcp``'s ``if not has_prefix: ... return``.

THIS IS A DETECTOR, NOT A CORRECTION. Once the vector diverges the attention
result is already wrong; refusing only makes the failure uniform and immediate.
It is deliberately NOT an OR-ballot over the derived boolean: OR-ing would
adopt the position that a divergent prefix vector is legitimate, which is the
opposite of the position #616B took when it made that vector uniform on the
paths it reached. Refusal is neutral between "extend the floor to this path"
and "divergence is legal here" and forecloses neither. The open question --
which of those is right -- is answered by the detector's first live firing,
because it prints the per-rank vectors.

Hermetic: no CUDA, no real process group. The reduce is a real elementwise MIN
over a fixed set of per-rank ballots built by the production packer.
"""

import inspect
import unittest
from unittest import mock

import torch

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ImportError:  # pragma: no cover - registration is a CI-time marker

    def register_cpu_ci(*args, **kwargs):
        return None


from sglang.srt.layers.dcp import prefix_lens_check
from sglang.srt.layers.dcp.lockstep import (
    PrefixLensRankDivergence,
    format_prefix_lens_divergence,
    prefix_lens_ballot,
    prefix_lens_ballot_agrees,
    weightless_has_prefix,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


#: The 08:26 shape: TP0's batch carries no matched prefix, its peers' does.
DEGRADED_0826 = [[0], [2048], [2048]]


class _FakeGroupDist:
    """A real elementwise MIN all_reduce plus a real all_gather_object over a
    fixed set of per-rank vectors."""

    ReduceOp = torch.distributed.ReduceOp

    def __init__(self, per_rank):
        self.per_rank = per_rank
        self.reduce_calls = 0
        self.gather_calls = 0
        self.widths = []

    def get_world_size(self, group=None):
        return len(self.per_rank)

    def all_reduce(self, t, op=None, group=None):
        self.reduce_calls += 1
        self.widths.append(t.numel())
        assert op is self.ReduceOp.MIN, "the ballot is a MIN reduction"
        ballots = [prefix_lens_ballot(v) for v in self.per_rank]
        for i in range(t.numel()):
            t[i] = min(b[i] for b in ballots)

    def all_gather_object(self, out, obj, group=None):
        self.gather_calls += 1
        for i, v in enumerate(self.per_rank):
            out[i] = list(v)


def _run_check(own_vector, per_rank):
    """Drive the REAL ``assert_prefix_lens_rank_uniform`` for one rank."""
    fake = _FakeGroupDist(per_rank)
    with mock.patch.object(torch, "distributed", fake), mock.patch.object(
        prefix_lens_check, "_dcp_cpu_group", lambda: object()
    ):
        prefix_lens_check.assert_prefix_lens_rank_uniform(own_vector)
    return fake


class PrefixLensBallotTest(CustomTestCase):
    def test_the_ballot_agrees_when_every_rank_sends_the_same_vector(self):
        same = [[0, 2048], [0, 2048], [0, 2048]]
        ballots = [prefix_lens_ballot(v) for v in same]
        reduced = [min(b[i] for b in ballots) for i in range(4)]
        self.assertTrue(prefix_lens_ballot_agrees(reduced))

    def test_the_ballot_catches_a_content_difference(self):
        ballots = [prefix_lens_ballot(v) for v in DEGRADED_0826]
        reduced = [min(b[i] for b in ballots) for i in range(4)]
        self.assertFalse(prefix_lens_ballot_agrees(reduced))

    def test_the_ballot_catches_a_length_difference(self):
        """A rank that admitted a different NUMBER of requests is the same
        defect one step earlier; the length field names it as such instead of
        reporting an opaque digest mismatch."""
        per_rank = [[2048], [2048, 512], [2048]]
        ballots = [prefix_lens_ballot(v) for v in per_rank]
        reduced = [min(b[i] for b in ballots) for i in range(4)]
        self.assertFalse(prefix_lens_ballot_agrees(reduced))
        self.assertNotEqual(reduced[0], -reduced[1], "the LENGTH must be the tell")

    def test_the_digest_is_process_stable(self):
        """`hash()` is salted for str/bytes and is a per-process value in
        exactly the situation this compares across processes. The packer must
        not depend on it."""
        src = inspect.getsource(prefix_lens_ballot)
        digest_src = inspect.getsource(
            __import__(
                "sglang.srt.layers.dcp.lockstep", fromlist=["_prefix_lens_digest"]
            )._prefix_lens_digest
        )
        self.assertNotIn("hash(", src)
        self.assertNotIn("hash(", digest_src)
        # Same input, same answer, and different inputs separate.
        self.assertEqual(prefix_lens_ballot([0, 7]), prefix_lens_ballot([0, 7]))
        self.assertNotEqual(prefix_lens_ballot([0, 7]), prefix_lens_ballot([7, 0]))

    def test_the_digest_separates_a_zero_from_an_absent_entry(self):
        """`[0]` and `[]` are different batches and must not collide: the
        first is a prefix-free request, the second is no request at all."""
        self.assertNotEqual(prefix_lens_ballot([0]), prefix_lens_ballot([]))


class PrefixLensDetectorTest(CustomTestCase):
    def test_a_uniform_vector_passes_with_exactly_one_collective(self):
        for rank in range(3):
            same = [[0, 2048]] * 3
            fake = _run_check(same[rank], same)
            self.assertEqual(fake.reduce_calls, 1, "one ballot, no more")
            self.assertEqual(fake.widths, [4])
            self.assertEqual(fake.gather_calls, 0, "the gather is failure-path only")

    def test_the_degraded_0826_vector_is_refused_on_every_rank(self):
        """THE FALSIFIER. Every rank must reach the SAME verdict from the same
        reduced ballot -- a detector that fired on one rank only would itself
        be the rank-local-test-before-a-collective defect it exists to report.
        """
        for rank in range(3):
            with self.assertRaises(PrefixLensRankDivergence) as ctx:
                _run_check(DEGRADED_0826[rank], DEGRADED_0826)
            msg = str(ctx.exception)
            # The vectors themselves must be in the message: the whole point of
            # paying for the check is that the next occurrence self-diagnoses.
            self.assertIn("rank 0", msg)
            self.assertIn("rank 2", msg)
            self.assertIn("[0]", msg)
            self.assertIn("[2048]", msg)
            self.assertIn("has_prefix=False", msg)
            self.assertIn("has_prefix=True", msg)

    def test_the_message_names_it_a_detector_not_a_correction(self):
        msg = format_prefix_lens_divergence(DEGRADED_0826)
        self.assertIn("DETECTOR, not a correction", msg)

    def test_no_dcp_group_takes_no_collective(self):
        """Single-rank and non-DCP boots must be byte-identical: no group, no
        ballot, no behaviour change."""
        fake = _FakeGroupDist(DEGRADED_0826)
        with mock.patch.object(torch, "distributed", fake), mock.patch.object(
            prefix_lens_check, "_dcp_cpu_group", lambda: None
        ):
            prefix_lens_check.assert_prefix_lens_rank_uniform([0])
        self.assertEqual(fake.reduce_calls, 0)

    def test_a_none_vector_takes_no_collective(self):
        fake = _FakeGroupDist(DEGRADED_0826)
        with mock.patch.object(torch, "distributed", fake), mock.patch.object(
            prefix_lens_check, "_dcp_cpu_group", lambda: object()
        ):
            prefix_lens_check.assert_prefix_lens_rank_uniform(None)
        self.assertEqual(fake.reduce_calls, 0)

    def test_the_kill_switch_is_read_once_at_import(self):
        """A per-call environ read would let an operator disable the check in
        one process and leave the others entering a collective nobody joins --
        the exact failure class this file reports."""
        src = inspect.getsource(prefix_lens_check.assert_prefix_lens_rank_uniform)
        self.assertNotIn("os.environ", src)
        self.assertIn("_ENABLED", src)

    def test_prepare_for_extend_calls_the_check_at_the_vector(self):
        """Deletion falsifier: the call must sit at the line that MATERIALISES
        the vector, not somewhere the forward has already entered a collective
        it cannot leave."""
        from sglang.srt.managers.schedule_batch import ScheduleBatch

        src = inspect.getsource(ScheduleBatch.prepare_for_extend)
        self.assertIn("assert_prefix_lens_rank_uniform(prefix_lens)", src)
        vector_at = src.index("prefix_lens = [len(r.prefix_indices) for r in reqs]")
        check_at = src.index("assert_prefix_lens_rank_uniform(prefix_lens)")
        self.assertLess(vector_at, check_at)

    def test_the_predicate_this_protects_still_reads_that_vector(self):
        """If ``weightless_has_prefix`` stops consuming the vector, this
        detector is guarding nothing and the justification has to be rewritten
        rather than silently kept."""
        self.assertFalse(weightless_has_prefix(False, [0]))
        self.assertTrue(weightless_has_prefix(False, [2048]))
        src = inspect.getsource(weightless_has_prefix)
        self.assertIn("return any(extend_prefix_lens_cpu)", src)


if __name__ == "__main__":
    unittest.main()
