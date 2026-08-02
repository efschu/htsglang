"""#426 -- the DCP LSE correction's log base must follow the attention backend.

Upstream sgl-project/sglang#33064: ``correct_attn_out`` re-normalizes each
rank's partial attention output by ``exp(local_lse - global_lse)`` and its
Triton kernel used ``exp2``/``log2`` unconditionally. FlashInfer MLA returns
base-2 LSE, so that was right; FlashMLA returns NATURAL-log LSE (its AOT tests
validate ``softmax_lse`` against ``torch.logsumexp``), so for FlashMLA it
computes different softmax weights -- wrong model output, no error, no NaN.

This is latent rather than academic for this fork. Our uneven-DCP feature
(#345, #346, #297) is built on ``correct_attn_out``, and the rig has no
FlashMLA-capable card (sm86/sm120 route through FlashInfer MLA), so nothing is
wrong today. It becomes wrong silently the day a Hopper-class card joins the
group -- which is exactly the kind of defect that has to be pinned before the
hardware arrives, not after.

Three layers are pinned, GPU-free:

1. THE ARITHMETIC -- a faithful CPU emulation of the kernel's reduction,
   parameterized by base, reproducing the reporter's numbers (8.0 correct,
   7.2330317 for the base-2 misreading). This is the divergence itself.
2. THE WIRING -- ``correct_attn_out`` must hand the kernel an ``LSE_BASE_E``
   constexpr that follows its argument, and ``cp_lse_ag_out_rs_mla`` must
   forward the caller's choice.
3. THE SELECTION -- ``lse_is_base_e`` must answer True for FlashMLA and False
   for every other backend, so the default stays base-2 exactly as before.

The Triton kernel body itself is not executed here (no GPU); what is executed
is every decision that reaches it.
"""

from __future__ import annotations

import contextlib
import unittest
from unittest import mock

import torch

import sglang.srt.layers.dcp.comm as dcp_comm
from sglang.srt.layers.dcp.comm import cp_lse_ag_out_rs_mla
from sglang.srt.layers.dcp.kernels import CPTritonContext, correct_attn_out
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

#: The reporter's minimal example: two DCP shards with partition functions
#: Z = [2, 8], local output 10 on the second shard.
ISSUE_PARTITION_FUNCTIONS = [2.0, 8.0]
ISSUE_LOCAL_OUTPUT = 10.0
ISSUE_CORRECT = 8.0
ISSUE_BASE2_MISREADING = 7.2330317031552385


def lse_is_base_e(backend):
    """The predicate under test, resolved by name.

    A tree without it answers "base 2 for everything", which is the pre-#426
    behavior -- so the assertions below fail on their own terms instead of
    turning into a collection error.
    """
    fn = getattr(dcp_comm, "lse_is_base_e", None)
    return False if fn is None else fn(backend)


def _combine(lses: torch.Tensor, cp_rank: int, out: torch.Tensor, base_e: bool):
    """CPU emulation of ``_correct_attn_cp_out_kernel``'s reduction.

    Same online-softmax shape as the kernel: subtract the max, exponentiate,
    sum, take the log, then scale the local output by ``exp(local - global)``.
    Only the base moves.
    """
    exp = torch.exp if base_e else torch.exp2
    log = torch.log if base_e else torch.log2

    lse_max = lses.max(dim=0).values
    lse_max = torch.where(torch.isneginf(lse_max), torch.zeros_like(lse_max), lse_max)
    final_lse = log(exp(lses - lse_max).sum(dim=0)) + lse_max
    factor = exp(lses[cp_rank] - final_lse)
    return out * factor, final_lse


class _CapturingContext:
    """A CPTritonContext stand-in that records the launch instead of running it."""

    def __init__(self):
        self.calls = []

    def call_kernel(self, kernel, grid, *regular_args, **const_args):
        self.calls.append((kernel, grid, regular_args, const_args))


class TestTheArithmeticDiverges(CustomTestCase):
    """The divergence is a different answer, not a rounding difference."""

    def test_natural_log_lse_read_as_base_two_returns_the_reporters_number(self):
        lses = torch.log(torch.tensor(ISSUE_PARTITION_FUNCTIONS)).view(2, 1, 1)
        out = torch.tensor([[ISSUE_LOCAL_OUTPUT]])

        corrected_e, _ = _combine(lses, 1, out, base_e=True)
        corrected_2, _ = _combine(lses, 1, out, base_e=False)

        self.assertAlmostEqual(corrected_e.item(), ISSUE_CORRECT, places=5)
        self.assertAlmostEqual(corrected_2.item(), ISSUE_BASE2_MISREADING, places=5)
        self.assertGreater(abs(corrected_e.item() - corrected_2.item()), 0.7)

    def test_the_correct_answer_is_the_partition_function_ratio(self):
        """Independent derivation, so the expectation is not the same code."""
        total = sum(ISSUE_PARTITION_FUNCTIONS)
        expected = ISSUE_LOCAL_OUTPUT * ISSUE_PARTITION_FUNCTIONS[1] / total
        self.assertAlmostEqual(expected, ISSUE_CORRECT, places=12)

    def test_base_two_lse_still_needs_the_base_two_reduction(self):
        """Control: FlashInfer MLA's convention must keep giving 8.0 too."""
        lses = torch.log2(torch.tensor(ISSUE_PARTITION_FUNCTIONS)).view(2, 1, 1)
        out = torch.tensor([[ISSUE_LOCAL_OUTPUT]])
        corrected, _ = _combine(lses, 1, out, base_e=False)
        self.assertAlmostEqual(corrected.item(), ISSUE_CORRECT, places=5)


class TestTheKernelIsToldTheBase(CustomTestCase):
    """The falsifier: unfixed, ``correct_attn_out`` has no base to be told."""

    def _launch(self, is_lse_base_on_e):
        out = torch.zeros(2, 3, 4)
        lses = torch.zeros(2, 2, 3)
        new_output = torch.zeros(3, 2, 4)
        ctx = _CapturingContext()
        correct_attn_out(
            out,
            lses,
            cp_rank=1,
            ctx=ctx,
            new_output=new_output,
            is_lse_base_on_e=is_lse_base_on_e,
        )
        self.assertEqual(len(ctx.calls), 1)
        return ctx.calls[0][3]

    def test_the_constexpr_follows_the_argument(self):
        for is_lse_base_on_e in (True, False):
            with self.subTest(is_lse_base_on_e=is_lse_base_on_e):
                const_args = self._launch(is_lse_base_on_e)
                self.assertIn("LSE_BASE_E", const_args)
                self.assertIs(const_args["LSE_BASE_E"], is_lse_base_on_e)

    def test_the_default_is_base_two(self):
        """Omitting the argument must reproduce the pre-#426 behavior."""
        out = torch.zeros(1, 1, 4)
        lses = torch.zeros(2, 1, 1)
        ctx = _CapturingContext()
        correct_attn_out(out, lses, cp_rank=0, ctx=ctx, new_output=torch.zeros(1, 1, 4))
        self.assertIs(ctx.calls[0][3]["LSE_BASE_E"], False)

    def test_the_other_constexprs_are_unchanged(self):
        const_args = self._launch(False)
        self.assertEqual(const_args["HEAD_DIM"], 4)
        self.assertEqual(const_args["N_ROUNDED"], 2)


class TestTheContextDoesNotReplayTheWrongSpecialization(CustomTestCase):
    """A single-slot JIT cache would serve the first base to every later call.

    That is the "shared buffer plus an ordering assumption" family, and adding
    a constexpr is precisely what makes it reachable.
    """

    def test_two_bases_through_one_context_compile_separately(self):
        launched = []

        def fake_kernel_getitem(grid):
            def launch(*args, **kwargs):
                launched.append(kwargs.get("LSE_BASE_E"))
                return mock.MagicMock(__getitem__=lambda self, g: lambda *a: None)

            return launch

        kernel = mock.MagicMock()
        kernel.__getitem__.side_effect = fake_kernel_getitem

        ctx = CPTritonContext()
        ctx.call_kernel(kernel, (1,), 1, 2, HEAD_DIM=4, N_ROUNDED=2, LSE_BASE_E=False)
        ctx.call_kernel(kernel, (1,), 1, 2, HEAD_DIM=4, N_ROUNDED=2, LSE_BASE_E=True)
        self.assertEqual(launched, [False, True])

        # ...and a repeat of the first specialization replays, not recompiles.
        ctx.call_kernel(kernel, (1,), 1, 2, HEAD_DIM=4, N_ROUNDED=2, LSE_BASE_E=False)
        self.assertEqual(launched, [False, True])


class TestTheBackendSelectsTheBase(CustomTestCase):
    def test_only_flashmla_is_natural_log(self):
        self.assertTrue(lse_is_base_e("flashmla"))
        for backend in ("flashinfer", "fa3", "triton", "dsv4", "nsa", "aiter", None):
            with self.subTest(backend=backend):
                self.assertFalse(lse_is_base_e(backend))

    def test_the_collective_forwards_the_callers_choice(self):
        for is_lse_base_on_e in (True, False):
            with self.subTest(is_lse_base_on_e=is_lse_base_on_e):
                seen = {}

                def fake_correct(out, lses, cp_rank, ctx, new_output, **kwargs):
                    seen.update(kwargs)
                    return new_output, None

                group = mock.MagicMock()
                group.world_size = 2
                group.rank_in_group = 1
                group.all_gather.side_effect = lambda x, dim=0: torch.cat([x, x], dim)
                group.reduce_scatter_along_dim.side_effect = lambda x, dim=0: x

                with contextlib.ExitStack() as stack:
                    stack.enter_context(
                        mock.patch.object(dcp_comm, "correct_attn_out", fake_correct)
                    )
                    stack.enter_context(
                        mock.patch.object(
                            dcp_comm,
                            "use_symmetric_memory",
                            lambda _g: contextlib.nullcontext(),
                        )
                    )
                    cp_lse_ag_out_rs_mla(
                        torch.zeros(2, 4, 8),
                        torch.zeros(2, 4),
                        group,
                        is_lse_base_on_e=is_lse_base_on_e,
                    )
                self.assertIs(seen.get("is_lse_base_on_e"), is_lse_base_on_e)


class TestTheMlaCallSiteAsksTheBackend(CustomTestCase):
    """The producing backend, not the collective, owns the base -- so the model
    call site must be the place that decides."""

    def test_forward_mla_derives_the_flag_from_current_attention_backend(self):
        from sglang.srt.models.deepseek_common.attention_forward_methods import (
            forward_mla,
        )

        source = forward_mla.__file__
        with open(source) as handle:
            text = handle.read()
        self.assertIn("is_lse_base_on_e=lse_is_base_e(", text)
        self.assertIn("self.current_attention_backend", text)


if __name__ == "__main__":
    unittest.main()
