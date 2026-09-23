"""DFLASH2 draft shard, stage 2: the three REPLICATED blocks (task #37).

Stage 1 sharded the draft's attention and MLP (uneven, by the
``--rank-tp-ratio`` plan). What stayed replicated on every TP rank was
measured on boot xsn402 as 522 MB per rank of a 1264 MB draft:

  * the candidate selector's two [vocab, r] codebooks  (254 MB)
  * ``fc``, the K-target-layer context projection       (133 MB)
  * the ten grouped-conv kernel projections             (131 MB)

Stage 2 shards all three with the parallel primitives the fork already has,
and these tests pin the property that makes that legal: for one simulated
TP=3 group, the per-rank pieces RECONSTRUCT the replicated result exactly.

No GPU, no distributed group: the layers are built under an injected
parallel context (``get_parallel().override``) plus an installed shard plan,
the TP-group handle fetch is stubbed, and the all-reduce is replaced by the
identity so the test can sum the three ranks' contributions itself -- which
is precisely what the real collective does.

The three mechanisms are deliberately NOT the same:

  A) codebooks   -> VocabParallelEmbedding: row band per rank, masked lookup,
                    all-reduce. The vocab axis splits EVENLY even under an
                    uneven base plan ("vocab always even", see
                    ``tp_vocab_ratios``); only ``--rank-vocab-ratio`` opts the
                    vocab into a ratio-weighted split.
  B) fc          -> RowParallelLinear(input_is_parallel=False): the replicated
                    input is cut along the contraction axis by the plan,
                    all-reduce of the output.
  C) conv kernel -> RowParallelLinear(input_is_parallel=False) as well, NOT
                    column-parallel + all-gather: under an uneven plan the
                    per-rank group bands have different widths, and the fork's
                    ``all_gather(dim=-1)`` is equal-size-only.

B and C both rely on the plan-aware input cut added to
``RowParallelLinear.forward``; its default-path inertness is pinned here too.
"""

import contextlib
import math
import unittest
from unittest.mock import patch

import torch

import sglang.srt.layers.linear as linear_mod
import sglang.srt.layers.vocab_parallel_embedding as vpe_mod
from sglang.srt.distributed.utils import (
    set_tp_partition_ratios,
    tp_partition_sizes,
)
from sglang.srt.layers.linear import RowParallelLinear
from sglang.srt.models.dflash import CandidateSelector, DFlashGroupedConv
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

# The layout the DFlash2 draft actually runs on (boot xsn402): TP=3 with
# --rank-tp-ratio 39,13,12 (one 5090 + two 3080s).
PLAN = [39, 13, 12]
TP = len(PLAN)
FP = torch.float32


class _CollectiveCounter:
    """Stands in for an all-reduce: counts the calls and returns the input.

    Returning the input unchanged makes each rank's forward produce its own
    PARTIAL contribution, so a test can sum the three of them and compare
    against the unsharded reference -- the sum IS what the real collective
    computes.
    """

    def __init__(self):
        self.calls = 0
        self.shapes = []

    def __call__(self, x, *args, **kwargs):
        self.calls += 1
        self.shapes.append(tuple(x.shape))
        return x


class _Stage2Case(CustomTestCase):
    def setUp(self):
        set_tp_partition_ratios(None)
        self._server_args = get_context().override_server_args()
        self._server_args.install()

    def tearDown(self):
        self._server_args.restore()
        set_tp_partition_ratios(None)

    @contextlib.contextmanager
    def _group(self, counter=None):
        """Stub the TP-group handle and the all-reduce.

        ``get_tp_group()`` is a handle fetch (the symmetric-memory context
        manager takes it as an argument), not a collective -- it is stubbed
        only because no process group exists in a CPU unit test.
        """
        counter = counter if counter is not None else _CollectiveCounter()
        with contextlib.ExitStack() as stack:
            for mod in (vpe_mod, linear_mod):
                stack.enter_context(patch.object(mod, "get_tp_group", lambda: None))
                stack.enter_context(
                    patch.object(mod, "tensor_model_parallel_all_reduce", counter)
                )
            yield counter

    @staticmethod
    @contextlib.contextmanager
    def _rank(tp_rank, tp_size=TP):
        with get_parallel().override(
            tp_size=tp_size,
            tp_rank=tp_rank,
            attn_tp_size=tp_size,
            attn_tp_rank=tp_rank,
        ):
            yield


# ---------------------------------------------------------------------------
# A) candidate-selector codebooks
# ---------------------------------------------------------------------------


class TestSelectorCodebooksAreVocabParallel(_Stage2Case):
    VOCAB = 1152  # a multiple of lcm(64, 3), so every rank owns real rows
    RANK_DIM = 8
    HIDDEN = 16
    TOP_K = 4

    def _selector(self, tp_rank, tp_size=TP):
        with self._rank(tp_rank, tp_size):
            return CandidateSelector(
                hidden_size=self.HIDDEN,
                vocab_size=self.VOCAB,
                state_rank=self.RANK_DIM,
                top_k=self.TOP_K,
                prefix="candidate_selector",
            )

    def test_tp1_lookup_is_the_plain_table_and_runs_no_collective(self):
        with self._group() as counter:
            sel = self._selector(0, tp_size=1)
            table = torch.randn(self.VOCAB, self.RANK_DIM, dtype=FP)
            sel.successor_codebook.weight.data.copy_(table)
            ids = torch.randint(0, self.VOCAB, (2, 3, self.TOP_K))
            with self._rank(0, tp_size=1):
                out = sel.successor_codebook(ids)
        self.assertTrue(torch.equal(out, table[ids]))
        self.assertEqual(counter.calls, 0)

    def test_tp3_masked_lookups_sum_to_the_replicated_table(self):
        """Every rank asks for EVERY id (candidate ids are global). The ranks
        that do not own a row contribute an exactly-zero row, so the sum is
        bit-exact, not merely close."""
        set_tp_partition_ratios(PLAN)
        table = torch.randn(self.VOCAB, self.RANK_DIM, dtype=FP)
        ids = torch.randint(0, self.VOCAB, (2, 3, self.TOP_K))
        with self._group() as counter:
            selectors = [self._selector(r) for r in range(TP)]
            widths = [
                int(s.successor_codebook.weight.shape[0]) for s in selectors
            ]
            for sel in selectors:
                param = sel.successor_codebook.weight
                sel.successor_codebook.weight_loader(param, table)
            parts = []
            for rank, sel in enumerate(selectors):
                with self._rank(rank):
                    parts.append(sel.successor_codebook(ids))

        # The vocab axis stays EVEN under a base --rank-tp-ratio plan.
        self.assertEqual(widths, [self.VOCAB // TP] * TP)
        self.assertTrue(torch.equal(sum(parts), table[ids]))
        # One all-reduce per lookup per rank, all with the full row shape.
        self.assertEqual(counter.calls, TP)
        self.assertEqual(
            counter.shapes, [tuple(ids.shape) + (self.RANK_DIM,)] * TP
        )

    def test_each_rank_only_holds_its_band(self):
        """The point of the exercise: no rank carries the whole table."""
        set_tp_partition_ratios(PLAN)
        with self._group():
            selectors = [self._selector(r) for r in range(TP)]
        for rank, sel in enumerate(selectors):
            for codebook in (sel.predecessor_codebook, sel.successor_codebook):
                self.assertEqual(
                    int(codebook.weight.shape[0]), self.VOCAB // TP
                )
                self.assertEqual(int(codebook.weight.shape[1]), self.RANK_DIM)
                self.assertEqual(
                    int(codebook.shard_indices.org_vocab_start_index),
                    rank * (self.VOCAB // TP),
                )

    def test_build_lattice_matches_the_replicated_formula(self):
        """TP=1 regression for the lattice itself: moving the two lookups out
        of the compiled ``_score_edges`` must not move a number."""
        with self._group() as counter:
            sel = self._selector(0, tp_size=1)
            for param in sel.parameters():
                torch.nn.init.normal_(param)
            b, slots, k = 2, 3, self.TOP_K
            candidate_ids = torch.randint(0, self.VOCAB, (b, slots, k))
            unary_logits = torch.randn(b, slots, k, dtype=FP)
            hidden_states = torch.randn(b, slots, self.HIDDEN, dtype=FP)
            anchor_token_ids = torch.randint(0, self.VOCAB, (b,))
            with self._rank(0, tp_size=1):
                scores = sel.build_lattice(
                    candidate_ids=candidate_ids,
                    unary_logits=unary_logits,
                    hidden_states=hidden_states,
                    anchor_token_ids=anchor_token_ids,
                )

        pred_table = sel.predecessor_codebook.weight
        succ_table = sel.successor_codebook.weight
        hidden = sel.hidden_projection(hidden_states)
        predecessor_ids = torch.cat(
            [
                anchor_token_ids[:, None, None].expand(-1, 1, k),
                candidate_ids[:, :-1],
            ],
            dim=1,
        )
        expected = unary_logits[:, :, None] + torch.einsum(
            "blpr,blcr->blpc",
            pred_table[predecessor_ids] * hidden[:, :, None],
            succ_table[candidate_ids],
        )
        torch.testing.assert_close(scores, expected)
        self.assertEqual(counter.calls, 0)

    def test_build_lattice_issues_exactly_two_collectives_per_rank(self):
        """Rank-uniform collective SEQUENCE: two all-reduces, same order, same
        payload shape on every rank. Shard widths may differ; the sequence may
        not -- a divergent one is a hang, not a wrong number."""
        set_tp_partition_ratios(PLAN)
        b, slots, k = 2, 3, self.TOP_K
        candidate_ids = torch.randint(0, self.VOCAB, (b, slots, k))
        unary_logits = torch.randn(b, slots, k, dtype=FP)
        hidden_states = torch.randn(b, slots, self.HIDDEN, dtype=FP)
        anchor_token_ids = torch.randint(0, self.VOCAB, (b,))
        per_rank_shapes = []
        for rank in range(TP):
            with self._group() as counter:
                sel = self._selector(rank)
                with self._rank(rank):
                    sel.build_lattice(
                        candidate_ids=candidate_ids,
                        unary_logits=unary_logits,
                        hidden_states=hidden_states,
                        anchor_token_ids=anchor_token_ids,
                    )
            self.assertEqual(counter.calls, 2)
            per_rank_shapes.append(counter.shapes)
        self.assertEqual(per_rank_shapes[0], per_rank_shapes[1])
        self.assertEqual(per_rank_shapes[1], per_rank_shapes[2])
        self.assertEqual(
            per_rank_shapes[0], [(b, slots, k, self.RANK_DIM)] * 2
        )


# ---------------------------------------------------------------------------
# B) fc -- the context projection
# ---------------------------------------------------------------------------


class TestFcIsRowParallel(_Stage2Case):
    HIDDEN = 64
    FEATURES = 5  # the shipped checkpoint's len(target_layer_ids)

    @property
    def fc_in(self):
        return self.FEATURES * self.HIDDEN

    def _fc(self, tp_rank, tp_size=TP):
        units = self.fc_in // math.gcd(self.fc_in, 16)
        with self._rank(tp_rank, tp_size):
            return RowParallelLinear(
                self.fc_in,
                self.HIDDEN,
                bias=False,
                input_is_parallel=False,
                params_dtype=FP,
                prefix="fc",
                tp_units=units,
            )

    def test_tp1_is_the_unsharded_matmul(self):
        full = torch.randn(self.HIDDEN, self.fc_in, dtype=FP)
        x = torch.randn(7, self.fc_in, dtype=FP)
        with self._group() as counter:
            fc = self._fc(0, tp_size=1)
            fc.weight_loader(fc.weight, full)
            with self._rank(0, tp_size=1):
                out, bias = fc(x)
        self.assertIsNone(bias)
        self.assertEqual(counter.calls, 0)
        torch.testing.assert_close(out, torch.nn.functional.linear(x, full))

    def test_tp3_partials_sum_to_the_unsharded_matmul(self):
        set_tp_partition_ratios(PLAN)
        full = torch.randn(self.HIDDEN, self.fc_in, dtype=FP)
        x = torch.randn(7, self.fc_in, dtype=FP)
        with self._group() as counter:
            fcs = [self._fc(r) for r in range(TP)]
            for fc in fcs:
                fc.weight_loader(fc.weight, full)
            parts = []
            for rank, fc in enumerate(fcs):
                with self._rank(rank):
                    out, _ = fc(x)
                parts.append(out)

        widths = [int(fc.weight.shape[1]) for fc in fcs]
        self.assertEqual(
            widths,
            tp_partition_sizes(
                self.fc_in, TP, self.fc_in // math.gcd(self.fc_in, 16)
            ),
        )
        self.assertEqual(sum(widths), self.fc_in)
        torch.testing.assert_close(
            sum(parts), torch.nn.functional.linear(x, full), atol=2e-5, rtol=2e-5
        )
        self.assertEqual(counter.calls, TP)
        self.assertEqual(counter.shapes, [(7, self.HIDDEN)] * TP)

    def test_shipped_geometry_splits_by_the_ratio(self):
        """The dimension the real draft carries: K=5 context features of
        hidden 5120 = 25600, in 16-element units, over (39, 13, 12)."""
        set_tp_partition_ratios(PLAN)
        shipped_in = 5 * 5120
        units = shipped_in // math.gcd(shipped_in, 16)
        self.assertEqual(units, 1600)
        self.assertEqual(
            tp_partition_sizes(shipped_in, TP, units), [15600, 5200, 4800]
        )

    def test_input_cut_matches_the_weight_shard(self):
        """The cut and the shard are two separate code paths (forward vs.
        weight_loader) reading the same plan; this pins that they agree.

        With the even ``split_tensor_along_last_dim`` that the default path
        uses, 320 would not even divide by 3 -- the layer would raise in
        ``divide()`` rather than compute the wrong thing.
        """
        set_tp_partition_ratios(PLAN)
        full = torch.randn(self.HIDDEN, self.fc_in, dtype=FP)
        # One-hot rows: rank r's output is non-zero only for the input columns
        # it actually owns, so the cut is directly observable.
        x = torch.eye(self.fc_in, dtype=FP)
        with self._group():
            offset = 0
            for rank in range(TP):
                fc = self._fc(rank)
                fc.weight_loader(fc.weight, full)
                width = int(fc.weight.shape[1])
                with self._rank(rank):
                    out, _ = fc(x)
                expected = torch.zeros(self.fc_in, self.HIDDEN, dtype=FP)
                expected[offset : offset + width] = full[
                    :, offset : offset + width
                ].T
                torch.testing.assert_close(out, expected)
                offset += width
            self.assertEqual(offset, self.fc_in)

    def test_no_plan_keeps_the_classic_even_split(self):
        """Inertness of the plan-aware cut: without an installed plan the
        even path runs, byte for byte."""
        hidden, in_features = 4, 12
        full = torch.randn(hidden, in_features, dtype=FP)
        x = torch.randn(3, in_features, dtype=FP)
        with self._group():
            parts = []
            for rank in range(TP):
                with self._rank(rank):
                    fc = RowParallelLinear(
                        in_features,
                        hidden,
                        bias=False,
                        input_is_parallel=False,
                        params_dtype=FP,
                        prefix="fc",
                    )
                fc.weight_loader(fc.weight, full)
                self.assertEqual(int(fc.weight.shape[1]), in_features // TP)
                with self._rank(rank):
                    out, _ = fc(x)
                parts.append(out)
        torch.testing.assert_close(
            sum(parts), torch.nn.functional.linear(x, full), atol=2e-6, rtol=2e-6
        )


# ---------------------------------------------------------------------------
# C) grouped-conv kernel projection
# ---------------------------------------------------------------------------


class TestConvKernelProjectionIsSharded(_Stage2Case):
    HIDDEN = 64
    GROUP_SIZE = 8
    TAPS = 2
    BLOCK = 4

    @property
    def num_groups(self):
        return self.HIDDEN // self.GROUP_SIZE

    @property
    def proj_out(self):
        return 2 * self.TAPS * self.num_groups

    def _conv(self, tp_rank, tp_size=TP):
        with self._rank(tp_rank, tp_size):
            return DFlashGroupedConv(
                self.HIDDEN,
                self.BLOCK,
                self.TAPS,
                self.GROUP_SIZE,
                prefix="layers.0.attention_conv",
            )

    def _reference_coefficients(self, x, full):
        return torch.nn.functional.linear(x, full).reshape(
            *x.shape[:-1], 2, self.TAPS, self.num_groups
        )

    def test_tp1_projection_is_the_unsharded_one(self):
        full = torch.randn(self.proj_out, self.HIDDEN, dtype=FP)
        x = torch.randn(self.BLOCK * 2, self.HIDDEN, dtype=FP)
        with self._group() as counter:
            conv = self._conv(0, tp_size=1)
            conv.kernel_projection.weight_loader(
                conv.kernel_projection.weight, full
            )
            with self._rank(0, tp_size=1):
                prepared, kernel = conv.prepare(x)
        expected = self._reference_coefficients(x, full)
        torch.testing.assert_close(kernel, expected[..., 1, :, :])
        self.assertEqual(tuple(prepared.shape), tuple(x.shape))
        self.assertEqual(counter.calls, 0)

    def test_tp3_partials_sum_to_the_unsharded_projection(self):
        set_tp_partition_ratios(PLAN)
        full = torch.randn(self.proj_out, self.HIDDEN, dtype=FP)
        x = torch.randn(self.BLOCK * 2, self.HIDDEN, dtype=FP)
        with self._group() as counter:
            convs = [self._conv(r) for r in range(TP)]
            for conv in convs:
                conv.kernel_projection.weight_loader(
                    conv.kernel_projection.weight, full
                )
            kernels = []
            for rank, conv in enumerate(convs):
                with self._rank(rank):
                    _, kernel = conv.prepare(x)
                kernels.append(kernel)

        widths = [int(c.kernel_projection.weight.shape[1]) for c in convs]
        self.assertEqual(
            widths,
            tp_partition_sizes(
                self.HIDDEN, TP, self.HIDDEN // math.gcd(self.HIDDEN, 16)
            ),
        )
        self.assertEqual(sum(widths), self.HIDDEN)
        expected = self._reference_coefficients(x, full)[..., 1, :, :]
        torch.testing.assert_close(
            sum(kernels), expected, atol=2e-5, rtol=2e-5
        )
        # One all-reduce per conv per rank, over the FULL coefficient block --
        # the group axis is never split across the wire, which is exactly why
        # this is row-parallel and not column-parallel + all-gather.
        self.assertEqual(counter.calls, TP)
        self.assertEqual(
            counter.shapes, [(x.shape[0], self.proj_out)] * TP
        )

    def test_weight_shards_concatenate_back_to_the_checkpoint_tensor(self):
        """The shard decomposition itself, independent of any forward."""
        set_tp_partition_ratios(PLAN)
        full = torch.randn(self.proj_out, self.HIDDEN, dtype=FP)
        with self._group():
            shards = []
            for rank in range(TP):
                conv = self._conv(rank)
                param = conv.kernel_projection.weight
                conv.kernel_projection.weight_loader(param, full)
                shards.append(param.detach().clone())
        self.assertTrue(torch.equal(torch.cat(shards, dim=1), full))

    def test_base_kernel_stays_replicated(self):
        """2 * taps * hidden elements; sharding it would buy ~41 kB per conv
        and cost a collective."""
        set_tp_partition_ratios(PLAN)
        with self._group():
            for rank in range(TP):
                conv = self._conv(rank)
                self.assertEqual(
                    tuple(conv.base_kernel.shape), (2, self.TAPS, self.HIDDEN)
                )


# ---------------------------------------------------------------------------
# solo placement
# ---------------------------------------------------------------------------


class TestSoloPlacementRunsNoCollective(_Stage2Case):
    """``--speculative-draft-placement solo``: ModelRunner builds the draft
    (host AND shadows) inside ``get_parallel().override(tp_size=1, ...)``, and
    the parallel layers cache tp_size at construction. So the three new
    sharded modules must degrade to plain local ops -- no mask, no all-reduce
    -- even while the surrounding target group runs TP=3 with a plan
    installed.

    That is the property the solo path needs: only the host calls the
    selector, so a collective in there would be a three-rank hang.
    """

    def test_selector_fc_and_conv_issue_no_collective_under_a_solo_build(self):
        set_tp_partition_ratios(PLAN)  # the target group's plan stays installed
        with self._group() as counter:
            # Built the way ModelRunner builds a solo draft.
            with self._rank(0, tp_size=1):
                sel = CandidateSelector(
                    hidden_size=16,
                    vocab_size=1152,
                    state_rank=8,
                    top_k=4,
                    prefix="candidate_selector",
                )
                conv = DFlashGroupedConv(64, 4, 2, 8, prefix="c")
                fc = RowParallelLinear(
                    320,
                    64,
                    bias=False,
                    input_is_parallel=False,
                    params_dtype=FP,
                    prefix="fc",
                    tp_units=20,
                )

            # Every module holds the FULL weight: nothing was sharded.
            self.assertEqual(
                int(sel.successor_codebook.weight.shape[0]), 1152
            )
            self.assertEqual(int(fc.weight.shape[1]), 320)
            self.assertEqual(int(conv.kernel_projection.weight.shape[1]), 64)

            with self._rank(0, tp_size=1):
                sel.build_lattice(
                    candidate_ids=torch.randint(0, 1152, (2, 3, 4)),
                    unary_logits=torch.randn(2, 3, 4, dtype=FP),
                    hidden_states=torch.randn(2, 3, 16, dtype=FP),
                    anchor_token_ids=torch.randint(0, 1152, (2,)),
                )
                fc(torch.randn(5, 320, dtype=FP))
                conv.prepare(torch.randn(8, 64, dtype=FP))

        self.assertEqual(counter.calls, 0)


if __name__ == "__main__":
    unittest.main()
