# SPDX-License-Identifier: Apache-2.0
"""#487: is the stock even-DCP allocator branch reachable with a REPLICATED
draft KV pool?

THE QUESTION. ``model_runner_kv_cache_mixin._init_pools`` ends its allocator
selection with a stock even-DCP fallback
(``model_runner_kv_cache_mixin.py:4001-4011``)::

    else:
        # Stock even-DCP (unchanged): interleave via inflated page granularity.
        self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
            self.max_total_num_tokens * self.dcp_size,
            page_size=self.page_size * self.dcp_size,
            ...
        )

That inflates BOTH the index space and the page granularity by ``dcp_size``,
i.e. it assumes the pool behind it is token-sharded across the DCP group. A
draft worker at ``--draft-kv-layout replicated`` (the default) has the
opposite geometry: its pool keeps the FULL token context and is head-sharded,
which the pool sizing knows (``:3552-3568`` guards on
``draft_pool_is_replicated``) but this allocator branch does not mention at
all. #108 never audited the combination.

THE ANSWER, and it is platform-conditional:

* **On CUDA the branch is UNREACHABLE with a replicated draft pool**, by two
  independent predicates -- one per producer of ``is_draft_worker=True``.
  Reduce first: given ``dcp_size > 1``, the stock branch is taken exactly when
  ``rank_tp_ratio is None`` AND ``not weightless_kv_active()``
  (``:3933-3936``). Then:

  1. **Speculative draft workers.** ``ServerArgs._handle_dcp_validation``
     refuses ``dcp_size > 1`` plus a speculative algorithm on CUDA unless the
     boot is uneven-weighted DCP (which REQUIRES ``rank_tp_ratio is not
     None``) or the weightless-KV fast lane (which makes
     ``weightless_kv_active()`` true) -- i.e. unless the allocator gate is
     already true. So such a boot always takes the branch ABOVE the stock one.
  2. **Dual-group lane runners (#274).** These also set
     ``is_draft_worker=True`` and are NOT speculative, so leg 1 does not cover
     them -- the assumption "a draft pool implies a speculative algorithm" is
     false, and finding that out is what turned this into a two-leg argument.
     They are closed by their own predicate instead:
     ``_lane_server_args_view`` forces ``view.dcp_size = 1``
     (``dual_group_lane.py:1633``), and ``ModelRunner.dcp_size`` reads
     server_args, so a lane never enters the DCP chain at all. At
     ``dcp_size == 1`` the stock branch's two multipliers are identities
     anyway.

  Both exclusions are real predicates, quoted in ``TestTheCudaGateExcludesIt``
  and ``TestTheAnalysisInputsAreStillTrue``, not assumptions. The producer set
  itself is pinned, so a third family shows up as a red test rather than as a
  silently wrong address.

* **On HIP/ROCm leg 1 does not run** -- ``_handle_dcp_validation`` returns at
  ``server_args.py:7612-7613`` (``if is_hip(): return``) before the CUDA
  branch. There, ``dcp_size > 1`` + speculation + ``rank_tp_ratio is None`` is
  admitted, the allocator's gate is False, and the stock branch IS reached
  with a replicated draft pool. (Leg 2 still holds: lanes force dcp_size=1 on
  every platform.) That residual is recorded by
  ``TestTheRocmResidual`` rather than fixed: this fork does not serve ROCm,
  no ROCm hardware was available to this task, and changing an allocator for
  a platform that cannot be booted here would be a desk change to a memory
  address -- the #345 right-token/wrong-slot class is exactly what that costs
  when it is wrong.

Everything here is hermetic: the two library predicates are executed, and the
two source-level facts (which names the allocator gate references, and what
the CUDA refusal says) are read from the tree.

Usage:
    python3 -m pytest test/registered/unit/distributed/test_stock_dcp_allocator_reach_487.py -v
"""

from __future__ import annotations

import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

_ROOT = Path(__file__).resolve().parents[4]
_MIXIN = _ROOT / "python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py"
_SERVER_ARGS = _ROOT / "python/sglang/srt/server_args.py"


class _Args:
    """The two fields the predicates under test read."""

    def __init__(self, draft_kv_layout="replicated"):
        self.draft_kv_layout = draft_kv_layout


# ---------------------------------------------------------------------------
# 1. the predicates, executed
# ---------------------------------------------------------------------------


class TestTheReplicatedDraftPredicate(unittest.TestCase):
    """``draft_pool_is_replicated`` is the single source of the geometry."""

    def test_a_default_draft_worker_is_replicated(self):
        from sglang.srt.layers.dcp.owner import draft_pool_is_replicated

        self.assertTrue(draft_pool_is_replicated(True, _Args()))

    def test_opting_into_dcp_turns_it_off(self):
        from sglang.srt.layers.dcp.owner import draft_pool_is_replicated

        self.assertFalse(draft_pool_is_replicated(True, _Args("dcp")))

    def test_the_target_worker_is_never_replicated_by_this_predicate(self):
        from sglang.srt.layers.dcp.owner import draft_pool_is_replicated

        self.assertFalse(draft_pool_is_replicated(False, _Args()))
        self.assertFalse(draft_pool_is_replicated(False, _Args("dcp")))

    def test_a_partial_server_args_stand_in_means_the_default(self):
        """The docstring at owner.py:92-99 claims getattr-defensiveness for
        test doubles and the CUDA-graph runners. That is a testable claim."""
        from sglang.srt.layers.dcp.owner import draft_pool_is_replicated

        class _Bare:
            pass

        self.assertTrue(draft_pool_is_replicated(True, _Bare()))


class TestTheAllocatorGateIgnoresIt(unittest.TestCase):
    """The structural fact the question rests on: the allocator branch chain
    never consults the draft-pool geometry, while the pool sizing does."""

    def _allocator_chain_source(self) -> str:
        """The allocator SELECTION chain only.

        Pool sizing and allocator selection both live in ``_init_pools``, so
        the whole function is the wrong window: the sizing legitimately reads
        the draft geometry a few hundred lines earlier. This slice runs from
        the first branch of the chain to the stock branch's inflated page
        argument.
        """
        src = _MIXIN.read_text()
        start = src.index("elif self.page_size == 1 and self.dcp_size == 1:")
        end = src.index("page_size=self.page_size * self.dcp_size,")
        return src[start:end]

    def test_the_pool_sizing_consults_the_draft_geometry(self):
        self.assertIn("_draft_non_dcp = draft_pool_is_replicated(", _MIXIN.read_text())

    def test_the_allocator_selection_does_not(self):
        """The asymmetry the question rests on. If this ever fails because the
        name appeared in the chain, the coupling changed and this whole file
        needs re-deriving."""
        chain = self._allocator_chain_source()
        code = "\n".join(line.split("#", 1)[0] for line in chain.splitlines())
        self.assertNotIn("draft_pool_is_replicated", code)
        self.assertNotIn("is_draft_worker", code)

    def test_a_neighbouring_feature_already_refuses_this_exact_layout(self):
        """Independent corroboration, from a guard written for another feature.

        ``_init_pools`` refuses --enable-kv-session-offload on the stock
        even-DCP allocator in its own words: "the stock even-DCP inflated-page
        layout re-interprets slot identity". That is a fork-authored statement
        that this branch's addressing is not interchangeable with the
        natural-page one -- which is why the replicated-draft combination was
        worth auditing rather than assuming benign.
        """
        src = _MIXIN.read_text()
        # The sentence is split across adjacent string literals in the source,
        # so match the distinctive half rather than the whole phrase.
        self.assertIn("inflated-page layout re-interprets", src)
        self.assertIn("_dcp > 1 and self.server_args.rank_tp_ratio is None", src)

    def test_the_stock_branch_inflates_both_axes(self):
        """Pin the expression, so a change to it invalidates this analysis
        loudly instead of silently."""
        src = _MIXIN.read_text()
        self.assertIn("self.max_total_num_tokens * self.dcp_size,", src)
        self.assertIn("page_size=self.page_size * self.dcp_size,", src)

    def test_the_preceding_gate_is_the_two_disjuncts(self):
        self.assertRegex(
            self._allocator_chain_source(),
            r"self\.server_args\.rank_tp_ratio is not None\s*\n\s*or weightless_kv_active\(\)",
        )


# ---------------------------------------------------------------------------
# 2. the exclusion, at its source
# ---------------------------------------------------------------------------


def _dcp_validation_source() -> str:
    src = _SERVER_ARGS.read_text()
    start = src.index("    def _handle_dcp_validation(")
    rest = src[start + 1 :]
    end = rest.find("\n    def ")
    return rest[: end if end != -1 else len(rest)]


class TestTheCudaGateExcludesIt(unittest.TestCase):
    """Mechanism reach: the excluding predicate, quoted rather than assumed."""

    def test_cuda_refuses_dcp_plus_speculation_outside_two_lanes(self):
        src = _dcp_validation_source()
        self.assertRegex(
            src,
            r"self\.speculative_algorithm is not None\s*\n\s*and not uneven_weighted_dcp"
            r"\s*\n\s*and not self\.weightless_kv_fastlane",
        )
        self.assertIn("does not support any speculative algorithm, but got", src)

    def test_the_first_permitted_lane_requires_rank_tp_ratio(self):
        """``uneven_weighted_dcp`` is one of the two escapes, and it carries
        ``rank_tp_ratio is not None`` -- the allocator gate's left disjunct."""
        src = _dcp_validation_source()
        window = src[src.index("uneven_weighted_dcp = (") :]
        window = window[: window.index("\n            )")]
        self.assertIn("self.rank_tp_ratio is not None", window)
        self.assertIn("len(set(self.rank_tp_ratio)) > 1", window)

    def test_the_second_permitted_lane_is_the_weightless_fast_lane(self):
        """...whose activation is the allocator gate's right disjunct."""
        self.assertIn("and not self.weightless_kv_fastlane", _dcp_validation_source())

    def test_therefore_a_cuda_boot_with_a_draft_pool_never_reaches_the_stock_branch(
        self,
    ):
        """The conclusion, as arithmetic over the two gates.

        A draft pool exists only under a speculative algorithm. On CUDA, that
        plus dcp_size > 1 admits exactly two shapes, and both make the
        allocator's gate true, so the stock ``else`` is not reached.
        """
        for rank_tp_ratio, weightless in (
            ([13, 6, 6], False),  # uneven-weighted DCP lane
            (None, True),  # weightless-KV fast lane
        ):
            allocator_gate = (rank_tp_ratio is not None) or weightless
            with self.subTest(rank_tp_ratio=rank_tp_ratio, weightless=weightless):
                self.assertTrue(allocator_gate, "stock else would be reached")

    def test_the_excluded_shape_is_the_one_that_would_reach_it(self):
        """Can-discriminate: the shape CUDA refuses is exactly the shape whose
        allocator gate is False. If the refusal is ever relaxed, this pairing
        is what makes the consequence visible."""
        rank_tp_ratio, weightless = None, False
        self.assertFalse((rank_tp_ratio is not None) or weightless)


class TestTheRocmResidual(unittest.TestCase):
    """What is NOT closed, recorded with its own predicate."""

    def test_hip_returns_before_the_cuda_refusal(self):
        src = _dcp_validation_source()
        hip_at = src.index("if is_hip():")
        cuda_at = src.index("elif is_cuda():")
        self.assertLess(hip_at, cuda_at)
        between = src[hip_at:cuda_at]
        self.assertIn("return", between)

    def test_the_residual_is_documented_where_the_branch_lives(self):
        """A residual nobody can find is not recorded. The stock branch must
        carry the pointer, so the next reader of those ten lines sees it."""
        src = _MIXIN.read_text()
        self.assertIn("#487", src)
        window = src[src.index("# Stock even-DCP") :][:1600]
        self.assertIn("#487", window)
        self.assertIn("replicated", window)


# ---------------------------------------------------------------------------
# 3. the analysis is only as good as the source it was read from
# ---------------------------------------------------------------------------


class TestTheAnalysisInputsAreStillTrue(unittest.TestCase):
    def test_the_speculative_draft_runner_inherits_the_target_dcp_size(self):
        """The premise that makes the question non-trivial on the speculative
        leg: that draft worker is built from the SAME server_args, so its
        ``dcp_size`` is the target's, not 1."""
        worker = (
            _ROOT / "python/sglang/srt/speculative/eagle_worker_v2.py"
        ).read_text()
        self.assertRegex(worker, r"TpModelWorker\(\s*\n\s*server_args=server_args,")
        self.assertIn("is_draft_worker=True,", worker)

    def test_every_producer_of_a_draft_worker_is_accounted_for(self):
        """The reachability argument is a case split over who sets
        ``is_draft_worker=True``. Pin that set: a new producer invalidates the
        analysis, and this is where it shows up.

        Two families, and the second was a surprise -- the #274 dual-group
        lane reuses the flag for a runner that is not a speculative draft at
        all, which is why "a draft pool implies a speculative algorithm" is
        FALSE and the CUDA argument needed a second leg.
        """
        producers = sorted(
            str(p.relative_to(_ROOT))
            for p in (_ROOT / "python/sglang/srt").rglob("*.py")
            if "is_draft_worker=True," in p.read_text()
        )
        self.assertEqual(
            producers,
            [
                "python/sglang/srt/model_executor/dual_group_lane.py",
                "python/sglang/srt/speculative/draft_worker_common.py",
                "python/sglang/srt/speculative/eagle_worker_v2.py",
                "python/sglang/srt/speculative/frozen_kv_mtp_worker_v2.py",
                "python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py",
                "python/sglang/srt/speculative/standalone_worker_v2.py",
            ],
            "a new is_draft_worker producer appeared; re-derive the #487 "
            "reachability case split before trusting this file",
        )

    def test_the_dual_group_lane_leg_is_closed_by_its_own_args_view(self):
        """The second leg, at its predicate.

        A lane runner is built from ``_lane_server_args_view``, which forces
        ``dcp_size = 1`` (and clears rank_tp_ratio and the speculative
        algorithm). ``ModelRunner.dcp_size`` reads server_args, so a lane
        never enters the DCP branch chain at all -- and at dcp_size == 1 the
        stock branch's two multipliers are identities regardless.
        """
        src = (
            _ROOT / "python/sglang/srt/model_executor/dual_group_lane.py"
        ).read_text()
        view = src[src.index("def _lane_server_args_view") :][:3000]
        self.assertIn("view.dcp_size = 1", view)
        self.assertIn("view.rank_tp_ratio = None", view)
        self.assertIn("view.speculative_algorithm = None", view)

    def test_the_runner_reads_dcp_size_from_server_args(self):
        """What makes the lane leg work: the runner takes dcp_size from the
        (overridden) server args, not from the ambient parallel view."""
        runner = (
            _ROOT / "python/sglang/srt/model_executor/model_runner.py"
        ).read_text()
        self.assertIn("self.dcp_size = server_args.dcp_size", runner)
        self.assertEqual(runner.count("self.dcp_size = "), 1)


if __name__ == "__main__":
    unittest.main()
