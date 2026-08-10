"""#656 spec item 8: draft CUDA graphs must be REMOVABLE to be measured.

The user's rule is explicit: measure draft graphs, and if NEXTN spec
gains nothing from them, leave them out. On this rig they are already
ON -- draft decode, draft extend and draft verify all capture, at
0.12-0.30 GB per rank per kind -- and there was no way to turn them off,
so the rule could not be applied and the item stayed open through eleven
successors. This flag is the missing instrument.

TWO THINGS RIDE ON IT, not one.

1. Spec item 8 itself: an A/B that can only be run in one direction is
   not an A/B.
2. Spec item 6. `resolve_spill_depth` refuses spill rungs 2-3 because
   "the TP decode CUDA graphs bake addresses into" the draft weights, so
   restoring them into a fresh arena would corrupt the graphs. No draft
   graphs, no baked addresses, no blocker -- the removal experiment is a
   PREREQUISITE for the spill route to >=600000, not an independent
   errand. The two spec items are sequenced, and this is the sequencer.

Default stays ON so a boot that does not ask is byte-identical.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from sglang.srt.speculative.base_spec_worker import should_capture_draft_graphs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _args(**kw):
    base = dict(disable_draft_cuda_graph=False, disable_cuda_graph=False)
    base.update(kw)
    return SimpleNamespace(**base)


class DraftGraphsCaptureByDefault(unittest.TestCase):
    def test_an_unasked_boot_still_captures(self):
        self.assertTrue(should_capture_draft_graphs(_args()))

    def test_missing_attribute_is_treated_as_capture(self):
        """Server args objects predating the flag must not silently disable."""
        self.assertTrue(should_capture_draft_graphs(SimpleNamespace()))

    def test_none_args_captures(self):
        self.assertTrue(should_capture_draft_graphs(None))


class TheFlagCanActuallyRefuse(unittest.TestCase):
    """Can-fail proof. A knob that cannot say no measures nothing."""

    def test_the_flag_disables_draft_capture(self):
        self.assertFalse(
            should_capture_draft_graphs(_args(disable_draft_cuda_graph=True))
        )

    def test_disabling_all_graphs_also_disables_draft(self):
        """--disable-cuda-graph must not leave the draft graphs behind.

        Otherwise a boot that asked for no graphs would still pay the
        draft capture and its per-rank memory, which is exactly the
        corridor the flip is fighting for.
        """
        self.assertFalse(should_capture_draft_graphs(_args(disable_cuda_graph=True)))


if __name__ == "__main__":
    unittest.main()


class TheGateIsReachedFromTheCallSite(unittest.TestCase):
    """The pure helper being right is not the same as the flag working.

    ``EagleDraftWorkerBase``'s own class comment records that several draft
    variants bypass ``EagleDraftWorker.__init__``, so a gate reading only
    ``self.server_args`` would be silently inert on exactly the workers
    this rig runs -- inert in the safe direction, but inert. This test
    drives the real method and asserts on the CALL, not on the predicate.
    """

    def _worker(self, server_args):
        from sglang.srt.speculative.base_spec_worker import EagleDraftWorkerBase

        calls = []

        # A minimal concrete subclass: the ABC's abstract members are the
        # forward path, none of which this test touches.
        class _W(EagleDraftWorkerBase):
            def draft(self):  # pragma: no cover - never called
                raise AssertionError

            def draft_extend(self):  # pragma: no cover - never called
                raise AssertionError

        w = _W.__new__(_W)
        w.server_args = server_args
        w.draft_worker = SimpleNamespace(
            init_cuda_graphs=lambda **kw: calls.append(("draft", kw))
        )
        w._capture_cuda_graphs = lambda: calls.append(("target", {}))
        return w, calls

    def test_the_flag_actually_skips_the_capture(self):
        w, calls = self._worker(_args(disable_draft_cuda_graph=True))
        w.init_cuda_graphs()
        self.assertEqual(calls, [], "capture ran despite the flag")

    def test_without_the_flag_both_captures_still_run(self):
        w, calls = self._worker(_args())
        w.init_cuda_graphs()
        self.assertEqual([c[0] for c in calls], ["draft", "target"])
