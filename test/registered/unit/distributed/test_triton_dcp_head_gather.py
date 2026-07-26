"""The Triton DCP head collectives carry per-rank head counts, not an average.

Both DCP forwards used a bare equal-shape pair:

    q_all = group.all_gather(q_local, dim=1)      # every rank contributes h
    out   = cp_lse_ag_out_rs_mha(...)             # slice = H // world * rank

whose precondition -- all ranks of the DCP group hold the SAME number of q
heads -- was stated nowhere. Under a --rank-tp-ratio plan the shards are
unequal ([16,8,8] for total_q=32 / kv=8 / tp=3, the fork's own worked
example), the ranks disagree on the collective's byte count, and the merge
slices the wrong heads. torch neither refuses nor repairs either half.

This is the same family as the shared-buffer sightings recorded in the
integration notes: a returned buffer plus an ordering/shape assumption that
lives only in a comment. Here the assumption becomes an argument.

Falsifiers, both run below on a faithful stand-in for the collective:
  * the gather, on [4,2,2]: the old expression returns a tensor of the wrong
    width in the wrong head order, seen from a small rank AND from the large
    one;
  * the merge, on [16,8,8]: ``H // world * rank`` puts rank 1 on heads 10:20
    instead of 16:24, and leaves heads 30 and 31 unreachable from every rank.

Equal counts must stay byte-identical -- that is the reachable configuration
today -- so that direction is asserted too.

CPU only: tensors are tiny and the collective is stood in for.
"""

import inspect
import pathlib
import unittest
from unittest import mock

import torch

from sglang.srt.distributed.utils import (
    get_tp_partition_ratios,
    set_tp_partition_ratios,
)
from sglang.srt.layers.attention.triton_backend import (
    TritonAttnBackend,
    _plan_aware_dcp_group_q_head_counts,
)
from sglang.srt.layers.dcp import cp_local_head_bounds
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class FakeDcpGroup:
    """Stand-in for a GroupCoordinator's head-dim all_gather.

    Models what the real collective does with a disagreeing shape: the output
    is sized from the LOCAL tensor (world_size * local), so a peer's larger
    contribution is truncated and a smaller one reads past its end. That is the
    corruption the equal-shape call site was exposed to; it is reproduced here
    rather than asserted about.
    """

    def __init__(self, rank, shards):
        self.rank_in_group = rank
        self.world_size = len(shards)
        self.shards = shards

    def all_gather(self, x, dim=0):
        assert dim == 1, "only the head-dim gather is modelled here"
        parts = []
        for t in self.shards:
            if t.shape == x.shape:
                parts.append(t)
                continue
            fitted = t.new_zeros(x.shape)
            width = min(t.shape[1], x.shape[1])
            fitted[:, :width] = t[:, :width]
            parts.append(fitted)
        return torch.cat(parts, dim=1)


class _HFTextConfig:
    swa_num_key_value_heads = None


class _ModelConfig:
    def __init__(self, num_q, num_kv):
        self.num_attention_heads = num_q
        self._num_kv = num_kv
        self.hf_text_config = _HFTextConfig()

    def get_total_num_kv_heads(self):
        return self._num_kv


def _backend(dcp_size, model_config):
    """A TritonAttnBackend carrying only what the head collectives read."""
    be = object.__new__(TritonAttnBackend)
    be.dcp_size = dcp_size
    be.dcp_model_config = model_config
    return be


def _shards(counts, tokens=2, head_dim=1):
    """Per-rank q shards whose values ARE their global head index."""
    out = []
    base = 0
    for c in counts:
        t = torch.zeros(tokens, c, head_dim)
        for h in range(c):
            t[:, h, :] = base + h
        out.append(t)
        base += c
    return out


class _Parallel:
    def __init__(self, tp_size=3, tp_rank=0):
        self.attn_tp_size = tp_size
        self.attn_tp_rank = tp_rank


class TestTritonDcpHeadGather(CustomTestCase):
    def setUp(self):
        self._saved = get_tp_partition_ratios()
        self._patch = mock.patch(
            "sglang.srt.layers.attention.triton_backend.get_parallel",
            return_value=_Parallel(),
        )
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        set_tp_partition_ratios(self._saved)

    # ------------------------------------------------------------- counts

    def test_counts_without_a_plan_replicate_the_model_s_own_head_count(self):
        """The default path must be an identity: whatever the layer reports is
        what every peer is assumed to hold, exactly as before this helper."""
        set_tp_partition_ratios(None)
        cfg = _ModelConfig(num_q=32, num_kv=8)
        self.assertEqual(
            _plan_aware_dcp_group_q_head_counts(cfg, 4, local_heads=8),
            [8, 8, 8, 8],
        )
        # dcp off -> the group is this rank alone
        self.assertEqual(
            _plan_aware_dcp_group_q_head_counts(cfg, 1, local_heads=8), [8]
        )

    def test_counts_under_a_plan_are_the_real_unequal_split(self):
        """Same worked example the workspace-sizing fix uses: [12,6,6] over
        total_q=32 / kv=8 / tp=3 is the head split [16,8,8].

        The point is that the counts are PER RANK: [16,8,8] and a naive
        [32//3]*3 both live near 32, and only the per-rank values place the
        heads correctly.
        """
        cfg = _ModelConfig(num_q=32, num_kv=8)
        set_tp_partition_ratios([12, 6, 6])
        counts = _plan_aware_dcp_group_q_head_counts(cfg, 3, local_heads=16)
        self.assertEqual(counts, [16, 8, 8])
        self.assertEqual(sum(counts), 32, "the split must be exhaustive")

    # ------------------------------------------------------------- gather

    def test_unequal_shards_gather_in_global_head_order(self):
        """THE falsifier for the gather half."""
        counts = [4, 2, 2]
        shards = _shards(counts)
        expected = torch.cat(shards, dim=1)  # heads 0..7, once each

        set_tp_partition_ratios(None)
        for rank in range(3):
            group = FakeDcpGroup(rank, shards)
            be = _backend(3, _ModelConfig(num_q=8, num_kv=2))
            # this rank's group really does hold [4,2,2]; the plan machinery
            # that derives that is exercised separately above
            be._dcp_group_q_head_counts = lambda _local, _c=counts: list(_c)

            # the OLD expression, verbatim
            old = group.all_gather(shards[rank], dim=1)
            # the NEW one
            new = be._dcp_gather_q_heads(shards[rank], group)

            self.assertEqual(new.shape, expected.shape, f"rank {rank} width")
            self.assertTrue(torch.equal(new, expected), f"rank {rank} order")
            self.assertNotEqual(
                old.shape[1],
                expected.shape[1],
                f"rank {rank}: the old gather happened to be the right width, "
                f"so this case does not falsify anything",
            )

    def test_equal_shards_are_byte_identical_to_the_plain_collective(self):
        """The reachable configuration today. Any drift here is a regression
        on the working even-DCP path."""
        counts = [3, 3, 3]
        shards = _shards(counts)
        set_tp_partition_ratios(None)
        for rank in range(3):
            group = FakeDcpGroup(rank, shards)
            be = _backend(3, _ModelConfig(num_q=9, num_kv=3))
            plain = group.all_gather(shards[rank], dim=1)
            new = be._dcp_gather_q_heads(shards[rank], group)
            self.assertTrue(torch.equal(new, plain), f"rank {rank}")

    # -------------------------------------------------------------- merge

    def test_the_equal_slice_reads_the_wrong_heads_when_shards_differ(self):
        """THE falsifier for the merge half, at the arithmetic that decides it.

        cp_lse_ag_out_rs_mha slices ``H // world_size * rank``. With [16,8,8]
        that is 10 heads per rank: rank 1 reads 10:20 where it owns 16:24, and
        heads 30/31 are read by nobody.
        """
        counts = [16, 8, 8]
        total = sum(counts)
        world = len(counts)
        equal_width = total // world

        covered = set()
        for rank in range(world):
            group = FakeDcpGroup(rank, [torch.zeros(1, c, 1) for c in counts])
            start, stop = cp_local_head_bounds(group, counts)
            self.assertEqual(stop - start, counts[rank])
            covered.update(range(start, stop))

            old_start = equal_width * rank
            old_stop = equal_width * (rank + 1)
            self.assertNotEqual(
                (old_start, old_stop),
                (start, stop),
                f"rank {rank}: the equal slice coincides here, no falsifier",
            )
        self.assertEqual(covered, set(range(total)), "prefix sums must tile")

        old_covered = set()
        for rank in range(world):
            old_covered.update(range(equal_width * rank, equal_width * (rank + 1)))
        self.assertEqual(
            set(range(total)) - old_covered,
            {30, 31},
            "the equal slice must be shown to drop the tail heads",
        )

    def test_equal_shards_keep_the_old_slice_exactly(self):
        counts = [4, 4, 4]
        for rank in range(3):
            group = FakeDcpGroup(rank, [torch.zeros(1, c, 1) for c in counts])
            self.assertEqual(cp_local_head_bounds(group, counts), (4 * rank, 4 * rank + 4))

    # --------------------------------------------------------- call sites

    def test_both_dcp_forwards_go_through_the_head_aware_helpers(self):
        """A helper nobody calls is not a fix. Pins that neither forward has a
        bare equal-shape head collective left."""
        src = pathlib.Path(
            inspect.getfile(TritonAttnBackend)
        ).read_text()
        # one call in forward_extend, one in forward_decode, for each helper
        self.assertEqual(src.count("self._dcp_gather_q_heads("), 2)
        self.assertEqual(src.count("self._dcp_merge_q_heads("), 2)
        self.assertIn("def _dcp_gather_q_heads(", src)
        self.assertIn("def _dcp_merge_q_heads(", src)
        self.assertNotIn("group.all_gather(q_local, dim=1)", src)
        self.assertNotIn("group.all_gather(q_for_decode, dim=1)", src)
        self.assertNotIn("cp_lse_ag_out_rs_mha(", src)


if __name__ == "__main__":
    unittest.main()
