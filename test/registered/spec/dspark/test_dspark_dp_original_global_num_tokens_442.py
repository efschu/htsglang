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
from unittest.mock import patch

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


#: Tokens in the hand-built draft batch. Deliberately unequal to every entry
#: of GLOBAL_NUM_TOKENS so a field that copied the wrong number is visible.
DRAFT_NUM_TOKENS = 6


def _make_forward_batch():
    # The real ForwardBatch defaults every one of these to None; a namespace
    # with the same defaults keeps the test free of a model runner.
    return types.SimpleNamespace(
        input_ids=torch.arange(DRAFT_NUM_TOKENS),
        original_global_num_tokens_cpu=None,
        global_num_tokens_cpu=None,
        global_num_tokens_for_logprob_cpu=None,
        global_num_tokens_gpu=None,
        global_num_tokens_for_logprob_gpu=None,
        num_token_non_padded=None,
        num_token_non_padded_cpu=None,
        can_run_dp_cuda_graph=None,
    )


def _fill(forward_batch, proposer=None, *, ep: bool = False, batch=None):
    """Run the function under test with the EP switch pinned.

    ``enable_num_token_non_padded()`` reads the live moe-EP group
    (``forward_batch_info.py:1527-1528``) and asserts when none exists, which
    is every hermetic process. Pinning it makes the EP axis an explicit
    parameter of each test instead of an ambient one.
    """
    with patch(
        "sglang.srt.speculative.dspark_components.dspark_draft."
        "enable_num_token_non_padded",
        return_value=ep,
    ):
        (proposer or _make_proposer())._fill_dp_moe_sync_metadata(
            forward_batch, batch if batch is not None else _make_batch()
        )
    return forward_batch


class TestDsparkDpOriginalGlobalNumTokens(unittest.TestCase):
    def test_original_counts_are_preserved_unscaled(self) -> None:
        forward_batch = _fill(_make_forward_batch())

        # Fails on the unfixed tree: the field stays None.
        self.assertEqual(
            forward_batch.original_global_num_tokens_cpu,
            GLOBAL_NUM_TOKENS,
            "A hand-built DSpark draft batch must carry the same original "
            "per-DP-rank counts that ForwardBatch.init_new copies across.",
        )

    def test_scaled_counts_are_still_the_scaled_ones(self) -> None:
        """The fix must not overwrite the speculatively scaled fields."""

        forward_batch = _fill(_make_forward_batch())

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
        forward_batch = _fill(_make_forward_batch(), proposer)

        self.assertIsNone(forward_batch.original_global_num_tokens_cpu)
        self.assertIsNone(forward_batch.global_num_tokens_cpu)
        self.assertIsNone(forward_batch.num_token_non_padded)
        self.assertIsNone(forward_batch.num_token_non_padded_cpu)


class TestDsparkDpNumTokenNonPadded(unittest.TestCase):
    """Upstream sglang #33098: the EP token accounting fields.

    The global vectors above are what the DP all-reduce reads. The EXPERT
    parallel path reads something else entirely -- ``num_token_non_padded`` and
    its ``_cpu`` twin (``layers/moe/topk.py``, ``hash_topk.py``,
    ``mega_moe.py``) -- and a hand-built DSpark draft batch left both at their
    ``None`` default, so a draft forward under DP+EP accounted its tokens
    against nothing. Same defect family as #442: the hand-built batch is not
    interchangeable with one ``ForwardBatch.init_new`` produced.

    ``enable_num_token_non_padded()`` is ``moe_ep_size > 1``
    (``forward_batch_info.py:1527-1528``), so the device tensor exists only
    under EP; the ``_cpu`` field is written either way, which is what upstream
    does.
    """

    def _fill(self, ep: bool):
        return _fill(_make_forward_batch(), ep=ep)

    def test_under_ep_the_device_tensor_carries_the_draft_token_count(self):
        forward_batch = self._fill(ep=True)
        self.assertIsNotNone(
            forward_batch.num_token_non_padded,
            "the EP accounting path would read None",
        )
        self.assertEqual(forward_batch.num_token_non_padded.item(), DRAFT_NUM_TOKENS)
        self.assertEqual(forward_batch.num_token_non_padded.dtype, torch.int32)

    def test_the_cpu_twin_is_written_with_or_without_ep(self):
        for ep in (True, False):
            forward_batch = self._fill(ep=ep)
            self.assertEqual(forward_batch.num_token_non_padded_cpu, DRAFT_NUM_TOKENS)

    def test_without_ep_no_device_tensor_is_allocated(self):
        """Neutrality: a non-EP boot pays nothing for this."""
        forward_batch = self._fill(ep=False)
        self.assertIsNone(forward_batch.num_token_non_padded)

    def test_the_count_is_the_draft_batch_not_a_global_vector(self):
        """The two are different numbers here, so a copy-paste is visible."""
        forward_batch = self._fill(ep=True)
        self.assertNotIn(DRAFT_NUM_TOKENS, GLOBAL_NUM_TOKENS)
        self.assertNotEqual(
            forward_batch.num_token_non_padded_cpu,
            forward_batch.global_num_tokens_cpu,
        )

    def test_the_scaled_and_original_vectors_are_untouched_by_the_addition(self):
        forward_batch = self._fill(ep=True)
        self.assertEqual(
            forward_batch.original_global_num_tokens_cpu, GLOBAL_NUM_TOKENS
        )
        self.assertEqual(
            forward_batch.global_num_tokens_cpu,
            [x * SPEC_SCALE for x in GLOBAL_NUM_TOKENS],
        )


if __name__ == "__main__":
    unittest.main()
