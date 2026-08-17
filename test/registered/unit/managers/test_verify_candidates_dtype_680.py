# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#680: the verify path's `candidates` must be int64, at ~1.0 pool usage.

THE DEATH. 2026-08-16 00:25:07, nineteen minutes into the c4b88e1923 boot under
five-lane load, all three ranks simultaneously:

    RuntimeError: Expected 'candidates' to be of type long (torch.int64)
      sgl_kernel tree_speculative_sampling_target_only
      <- eagle_utils.eagle_sample          (eagle_utils.py:1191)
      <- eagle_worker_v2.verify            (:2777)
      <- forward_batch_generation          (:2138)
      <- scheduler.run_batch

The iteration before it: full token usage 0.97, mamba 0.50, a 257-token chunked
prefill admitted, 143,984 tokens pending.

THE MECHANISM, and the working hypothesis was half right. It is NOT a float
tensor from a defaulted torch.empty -- it is int32, deliberately:

    eagle_sample:              candidates = verify_input.draft_token.reshape(...)
    _build_trivial_verify_input: draft_token = draft_input.bonus_tokens
    verify:                    bonus_tokens = torch.empty_like(accept_lens,
                                                  dtype=torch.int32)
                               ... and on the empty branch
                               bonus_tokens = torch.empty((0,), dtype=torch.int32)

So `draft_token` -- which IS the kernel's `candidates` -- arrives int32 from one
construction path. int32 is CORRECT at its source: bonus_tokens is an *input* to
build_tree_kernel_efficient on the ordinary path, never the candidates tensor.
The dtype is wrong at the door it came through, not where it was made.

WHY ONLY UNDER PRESSURE. `_build_trivial_verify_input` is taken when drafting is
disabled at high batch size, or on a phase-flip draft bootstrap -- both load
states. The ordinary path takes draft_token from build_tree_kernel_efficient,
which returns int64, which is why this never fired on a quiet instance.

LATENT, NOT INTRODUCED BY #679. The int32 dates to upstream #24724
(2026-05-08), months before the park. What #679 changed is survival: the
instance used to die at the allocator before it could keep taking the trivial
path at 0.97 usage. The park removed the earlier death and exposed this one.
That changes the framing, not the necessity.

THE FIX IS AT THE CONSTRUCTION SITE, NOT THE KERNEL CALL. A `.to(long)` in
eagle_sample would fix this crash and leave every other consumer of the field
free to be handed the wrong dtype. EagleVerifyInput.__post_init__ enforces the
contract for EVERY construction path, present and future -- and the class
already declared that contract in its own `create_idle_input` default
(`torch.empty((0,), dtype=torch.long)`). A default is not an invariant; this
makes it one.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.eagle_info import EagleVerifyInput

#: The crash iteration: pool essentially full.
USAGE_AT_CRASH = 0.97


def _verify_input(draft_token, *, draft_token_num=1):
    """An EagleVerifyInput built the way the trivial path builds it."""
    return EagleVerifyInput(
        draft_token=draft_token,
        custom_mask=None,
        positions=None,
        retrieve_index=None,
        retrieve_next_token=None,
        retrieve_next_sibling=None,
        retrieve_cum_len=None,
        spec_steps=0,
        topk=1,
        draft_token_num=draft_token_num,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        seq_lens_sum=0,
        seq_lens_cpu=None,
    )


def _candidates(verify_input, bs):
    """Exactly what eagle_sample does: candidates = draft_token.reshape(...)."""
    return verify_input.draft_token.reshape(bs, verify_input.draft_token_num)


class TheKernelContractTest(unittest.TestCase):
    """`candidates` must be int64. The kernel says so by raising."""

    def test_the_crash_input_now_yields_int64_candidates(self):
        """THE CRASH, as one assertion.

        bonus_tokens as the trivial path builds it -- int32, one per request --
        handed in as draft_token.
        """
        bonus_tokens = torch.empty_like(
            torch.zeros(4, dtype=torch.int32), dtype=torch.int32
        )
        vi = _verify_input(bonus_tokens)
        self.assertEqual(
            _candidates(vi, bs=4).dtype,
            torch.int64,
            "this is the tensor tree_speculative_sampling_target_only rejects",
        )

    def test_the_EMPTY_branch_of_the_trivial_path_too(self):
        """verify's other bonus_tokens branch: torch.empty((0,), int32).

        Reached when the batch is idle/degenerate -- which is precisely the
        near-exhaustion shape this ticket is about.
        """
        vi = _verify_input(torch.empty((0,), dtype=torch.int32))
        self.assertEqual(vi.draft_token.dtype, torch.int64)
        self.assertEqual(_candidates(vi, bs=0).dtype, torch.int64)

    def test_a_float_tensor_is_defended_too(self):
        """The original hypothesis was a defaulted-float tensor. It was not the
        cause here, but a field that must be int64 must be int64 whatever
        arrives -- and torch.empty/torch.tensor([]) do default to float."""
        vi = _verify_input(torch.zeros(2, dtype=torch.float32))
        self.assertEqual(vi.draft_token.dtype, torch.int64)

    def test_int64_passes_through_unchanged(self):
        """The ordinary path -- build_tree_kernel_efficient's output -- must be
        untouched in dtype AND in value."""
        tok = torch.arange(6, dtype=torch.int64)
        vi = _verify_input(tok, draft_token_num=3)
        self.assertEqual(vi.draft_token.dtype, torch.int64)
        self.assertTrue(torch.equal(vi.draft_token, torch.arange(6)))

    def test_the_values_survive_the_conversion(self):
        """A dtype fix that changed token ids would be a correctness bug wearing
        a crash fix's clothes."""
        vi = _verify_input(torch.tensor([7, 11, 13, 17], dtype=torch.int32))
        self.assertTrue(
            torch.equal(
                vi.draft_token, torch.tensor([7, 11, 13, 17], dtype=torch.int64)
            )
        )

    def test_none_is_not_an_error(self):
        """Some constructions carry no draft_token at all; enforcing a dtype on
        absence would turn a legal state into a crash."""
        self.assertIsNone(_verify_input(None).draft_token)


class TheViolationStaysVisibleTest(unittest.TestCase):
    """CONVERT, BUT DO NOT HIDE. A fix that silences its own trigger is how the
    defect comes back: the next path to hand over the wrong dtype would be
    absorbed without a word."""

    def setUp(self):
        EagleVerifyInput._draft_token_dtype_announced.clear()

    def tearDown(self):
        EagleVerifyInput._draft_token_dtype_announced.clear()

    def test_a_wrong_dtype_is_announced(self):
        with self.assertLogs(
            "sglang.srt.speculative.eagle_info", level="WARNING"
        ) as cm:
            _verify_input(torch.zeros(2, dtype=torch.int32))
        joined = "\n".join(cm.output)
        self.assertIn("int32", joined)
        self.assertIn("#680", joined)

    def test_it_is_announced_ONCE_per_dtype_not_once_per_verify(self):
        """This runs on every verify of every round. A per-call warning would
        be its own outage."""
        with self.assertLogs(
            "sglang.srt.speculative.eagle_info", level="WARNING"
        ) as cm:
            for _ in range(50):
                _verify_input(torch.zeros(2, dtype=torch.int32))
        self.assertEqual(len([m for m in cm.output if "#680" in m]), 1)

    def test_a_compliant_path_says_nothing(self):
        with self.assertNoLogs("sglang.srt.speculative.eagle_info", level="WARNING"):
            _verify_input(torch.zeros(2, dtype=torch.int64))


class TheFixIsAtTheConstructionSiteTest(unittest.TestCase):
    """DESIGN choice, pinned: the contract is enforced where the object is
    built, so it covers every path -- not at the one kernel call that happened
    to crash."""

    def test_enforcement_runs_from_post_init(self):
        import inspect

        src = inspect.getsource(EagleVerifyInput.__post_init__)
        self.assertIn("_enforce_draft_token_dtype", src)

    def test_the_trivial_path_still_hands_over_bonus_tokens(self):
        """The source of the wrong dtype is deliberately NOT changed:
        bonus_tokens is int32 correctly for its other consumers. If this ever
        becomes false the enforcement is what kept the instance alive, and the
        note in __post_init__ should be revisited rather than quietly stale.
        """
        import inspect

        from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

        src = inspect.getsource(EAGLEWorkerV2._build_trivial_verify_input)
        self.assertIn("draft_token=draft_input.bonus_tokens", src)

    def test_eagle_sample_still_derives_candidates_from_the_field(self):
        """If candidates ever stops coming from draft_token, this enforcement
        stops protecting the kernel and the pin must move with it."""
        import inspect

        from sglang.srt.speculative import eagle_utils

        src = inspect.getsource(eagle_utils)
        self.assertIn("candidates = verify_input.draft_token.reshape(", src)


if __name__ == "__main__":
    unittest.main()
