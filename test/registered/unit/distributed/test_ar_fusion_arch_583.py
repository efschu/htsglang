"""#583: the all-reduce FUSION ARCH term is a group decision, not a per-rank one.

THE DEFECT THIS PINS CLOSED
---------------------------
``apply_flashinfer_allreduce_fusion`` gated on
``is_sm90_supported() or is_sm100_supported()`` -- a query against THIS
RANK's device. That value reaches
``should_fuse_mlp_allreduce_with_next_layer``, and a True there makes the
layer SKIP ``postprocess_layer``, i.e. it removes a collective from that
layer.

So a per-rank answer is a rank-local predicate deciding whether a group
collective is entered: the ranks that fuse issue one fewer all-reduce per
layer than the ranks that do not. Worse than a hang -- because the
mispaired payloads are frequently the same size, the ranks complete each
other's collectives and compute garbage silently for a while before
anything aborts.

Dormant on a homogeneous cluster, and dormant on the rig this was found on
(sm86 + sm120: both answer False). LIVE on any heterogeneous TP group
straddling the sm90 boundary -- e.g. an sm89 card beside an sm90 one.

The fix is the same shape as ``decide_spec_kernel_backend``: every rank
contributes its capability, the MINIMUM rules, and if one rank lacks the
arch then NO rank fuses.

Hermetic: no CUDA, no process group. The reduction is driven through a fake.
"""

import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers import communicator as comm  # noqa: E402
from sglang.srt.layers.communicator import (  # noqa: E402
    ar_fusion_arch_supported,
    decide_ar_fusion_arch,
    reset_ar_fusion_arch,
    set_ar_fusion_arch,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeGroup:
    def __init__(self, world_size):
        self.world_size = world_size
        self.cpu_group = object()


def _decide_as(local_ok, world_locals):
    """Run the decision on a rank whose own capability is `local_ok`, in a
    group whose per-rank capabilities are `world_locals`."""
    import torch

    reset_ar_fusion_arch()
    with (
        mock.patch.object(comm, "local_ar_fusion_arch_supported", lambda: local_ok),
        mock.patch.object(
            torch.distributed,
            "all_reduce",
            lambda t, op=None, group=None: t.fill_(min(world_locals)),
        ),
    ):
        return decide_ar_fusion_arch(_FakeGroup(len(world_locals)))


class ArFusionArchGroupDecisionTest(unittest.TestCase):
    def tearDown(self):
        reset_ar_fusion_arch()

    # -- THE FALSIFIER -----------------------------------------------------

    def test_a_mixed_group_does_not_fuse_on_any_rank(self):
        """The divergent case: one rank has the arch, one does not. Before
        the fix the capable rank fused (skipping a collective) while the
        other did not. After it, neither does -- one answer."""
        world = [1, 0]  # rank 0 capable, rank 1 not
        answers = [_decide_as(bool(own), world) for own in world]
        self.assertEqual(answers, [False, False])
        self.assertEqual(len(set(answers)), 1)

    def test_the_prefix_per_rank_predicate_really_did_diverge(self):
        """The fixture must be a genuinely divergent case, or the test above
        proves nothing: the OLD predicate was the local value verbatim."""
        world = [1, 0]
        local_answers = [bool(own) for own in world]
        self.assertEqual(local_answers, [True, False])
        self.assertEqual(len(set(local_answers)), 2)

    def test_a_uniformly_capable_group_still_fuses(self):
        """The gate must be able to answer YES -- otherwise the test above
        would pass against something that always disables fusion."""
        world = [1, 1]
        answers = [_decide_as(bool(own), world) for own in world]
        self.assertEqual(answers, [True, True])

    def test_a_uniformly_incapable_group_is_unchanged(self):
        world = [0, 0]
        self.assertEqual([_decide_as(bool(o), world) for o in world], [False, False])

    # -- the recorded decision ---------------------------------------------

    def test_the_decision_is_what_the_fusion_predicate_reads(self):
        set_ar_fusion_arch(False)
        self.assertFalse(ar_fusion_arch_supported())
        reset_ar_fusion_arch()
        set_ar_fusion_arch(True)
        self.assertTrue(ar_fusion_arch_supported())

    def test_a_conflicting_second_decision_is_refused_not_silently_won(self):
        """Two disagreeing decisions in one process mean the collective ran
        twice with different inputs -- the very divergence being closed."""
        set_ar_fusion_arch(True)
        with self.assertRaises(RuntimeError):
            set_ar_fusion_arch(False)

    def test_undecided_falls_back_to_local_and_not_to_a_silent_false(self):
        """Single-rank boots and unit tests never run the collective; the
        local value is correct there, and must not read as a stale False."""
        reset_ar_fusion_arch()
        with mock.patch.object(comm, "local_ar_fusion_arch_supported", lambda: True):
            self.assertTrue(ar_fusion_arch_supported())

    def test_an_unavailable_group_does_not_block_boot(self):
        """Warn-never-raise on the plumbing: no group means the local value
        stands, which is what a single-rank boot wants anyway."""
        reset_ar_fusion_arch()
        with mock.patch.object(comm, "local_ar_fusion_arch_supported", lambda: True):
            self.assertTrue(decide_ar_fusion_arch(_FakeGroup(1)))


if __name__ == "__main__":
    unittest.main()
