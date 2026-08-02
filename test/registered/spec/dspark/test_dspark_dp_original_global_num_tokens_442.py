# SPDX-License-Identifier: Apache-2.0
"""#442 / upstream sgl-project/sglang#33287 + issue #33286.

``DraftBlockProposer`` builds its draft ``ForwardBatch`` by hand instead of
going through ``ForwardBatch.init_new``. That standard conversion copies the
unscaled per-DP-rank token counts across:

    ret.original_global_num_tokens_cpu = batch.global_num_tokens
        -- forward_batch_info.py:807

``_fill_dp_moe_sync_metadata`` filled only the speculatively SCALED fields
(``global_num_tokens_cpu`` and friends) and left ``original_...`` at its
``None`` default, so a hand-built DSpark draft batch was not interchangeable
with a converted one.

Reachability in this tree, stated honestly: upstream crashes in
``decode_cuda_graph_runner.can_run_graph`` with ``TypeError: 'NoneType' object
is not iterable`` because their bucket selection reads
``original_global_num_tokens_cpu``. Ours still reads ``global_num_tokens_cpu``
(``decode_cuda_graph_runner.py:853-859``), so that specific crash is not
reachable here today. The divergence from the standard conversion is real
regardless, and it becomes load-bearing the moment that bucket-selection line
is refreshed from upstream. What is pinned here is the invariant, not the
crash: a DSpark draft batch carries the same original counts a converted batch
would.
"""

from __future__ import annotations

import types
import unittest

import torch

from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockProposer

# Two DP ranks with different token counts, so a scaled/unscaled mix-up is
# visible rather than accidentally equal.
GLOBAL_NUM_TOKENS = [7, 3]
GLOBAL_NUM_TOKENS_FOR_LOGPROB = [2, 1]
# What the draft block reports after speculative scaling.
SPEC_SCALE = 4


class _SpecInfoStub:
    def get_spec_adjusted_global_num_tokens(self, batch):
        return (
            [x * SPEC_SCALE for x in batch.global_num_tokens],
            list(batch.global_num_tokens_for_logprob),
        )


def _make_proposer() -> DraftBlockProposer:
    return DraftBlockProposer(
        draft_model=None,
        draft_model_runner=types.SimpleNamespace(device="cpu"),
        gamma=2,
        mask_token_id=0,
        draft_block_spec_info=_SpecInfoStub(),
        dp_moe_sync=True,
    )


def _make_batch():
    return types.SimpleNamespace(
        global_num_tokens=list(GLOBAL_NUM_TOKENS),
        global_num_tokens_for_logprob=list(GLOBAL_NUM_TOKENS_FOR_LOGPROB),
        can_run_dp_cuda_graph=False,
    )


def _make_forward_batch():
    # The real ForwardBatch defaults every one of these to None; a namespace
    # with the same defaults keeps the test free of a model runner.
    return types.SimpleNamespace(
        original_global_num_tokens_cpu=None,
        global_num_tokens_cpu=None,
        global_num_tokens_for_logprob_cpu=None,
        global_num_tokens_gpu=None,
        global_num_tokens_for_logprob_gpu=None,
        can_run_dp_cuda_graph=None,
    )


class TestDsparkDpOriginalGlobalNumTokens(unittest.TestCase):
    def test_original_counts_are_preserved_unscaled(self) -> None:
        forward_batch = _make_forward_batch()
        _make_proposer()._fill_dp_moe_sync_metadata(forward_batch, _make_batch())

        # Fails on the unfixed tree: the field stays None.
        self.assertEqual(
            forward_batch.original_global_num_tokens_cpu,
            GLOBAL_NUM_TOKENS,
            "A hand-built DSpark draft batch must carry the same original "
            "per-DP-rank counts that ForwardBatch.init_new copies across.",
        )

    def test_scaled_counts_are_still_the_scaled_ones(self) -> None:
        """The fix must not overwrite the speculatively scaled fields."""

        forward_batch = _make_forward_batch()
        _make_proposer()._fill_dp_moe_sync_metadata(forward_batch, _make_batch())

        self.assertEqual(
            forward_batch.global_num_tokens_cpu,
            [x * SPEC_SCALE for x in GLOBAL_NUM_TOKENS],
        )
        self.assertNotEqual(
            forward_batch.global_num_tokens_cpu,
            forward_batch.original_global_num_tokens_cpu,
            "Scaled and original counts must stay distinguishable, otherwise "
            "this test could pass on a tree that conflates them.",
        )
        torch.testing.assert_close(
            forward_batch.global_num_tokens_gpu,
            torch.tensor(
                [x * SPEC_SCALE for x in GLOBAL_NUM_TOKENS], dtype=torch.int64
            ),
        )

    def test_no_dp_sync_leaves_the_batch_untouched(self) -> None:
        """The non-DP path must not acquire a new write."""

        proposer = _make_proposer()
        proposer._dp_moe_sync = False
        forward_batch = _make_forward_batch()
        proposer._fill_dp_moe_sync_metadata(forward_batch, _make_batch())

        self.assertIsNone(forward_batch.original_global_num_tokens_cpu)
        self.assertIsNone(forward_batch.global_num_tokens_cpu)


if __name__ == "__main__":
    unittest.main()
