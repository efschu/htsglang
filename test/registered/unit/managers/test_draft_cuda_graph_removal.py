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

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.speculative import eagle_worker_v2
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


class TheWorkerThisRigActuallyRuns(unittest.TestCase):
    """The base-class gate above was INERT on this rig. Measured, not feared.

    The boot of 2026-08-10 19:06 carried ``--disable-draft-cuda-graph`` on
    the command line of the live process and captured the draft graphs
    anyway: the log held ZERO occurrences of the disable message and idle
    VRAM moved by -108 / -234 / +274 MiB across the three cards -- mixed
    sign, i.e. boot noise, not a freed capture.

    ``--speculative-algorithm NEXTN`` runs ``EAGLEWorkerV2``, a thin wrapper
    that delegates straight to ``EagleDraftWorker`` -- and
    ``EagleDraftWorker`` OVERRIDES ``init_cuda_graphs`` and never called
    the gate. So the flag
    parsed, propagated, reached the worker, and did nothing. This is the
    same shape as the span-forwarding defect of HANDOFF_670 section 1d: a
    switch tested against a base class that the production path does not
    execute. Test the override, not the ancestor.

    ``_capture_cuda_graphs`` is tested separately and that is not
    thoroughness for its own sake: the phase flip RE-CAPTURES draft graphs
    when it re-arms a layout, entering through ``_capture_cuda_graphs``
    without passing ``init_cuda_graphs``. A gate on the entry point alone
    would hold at boot and let the graphs come back at the first flip --
    disabled for exactly as long as nobody looked.
    """

    def _fake(self, server_args):
        calls = []
        w = eagle_worker_v2.EagleDraftWorker.__new__(
            eagle_worker_v2.EagleDraftWorker
        )
        w.server_args = server_args
        w._spec_solo_active = False
        w._spec_solo_is_host = True
        w.draft_worker = SimpleNamespace(
            init_cuda_graphs=lambda **kw: calls.append(("draft", kw))
        )
        w.draft_runner = SimpleNamespace(
            tp_group=None,
            canary_manager=None,
            init_prefill_cuda_graph=lambda **kw: calls.append(("prefill", kw)),
        )
        w._capture_cuda_graphs = lambda: calls.append(("capture", {}))
        w.draft_tp_context = lambda _g: contextlib.nullcontext()
        return w, calls

    def test_the_flag_skips_the_override_the_rig_runs(self):
        """The prefill capture is skipped; the other two calls still happen.

        Both survivors are deliberate and each refuses internally:

        * ``draft_worker.init_cuda_graphs`` also BUILDS THE EAGER RUNNER.
          Skipping it is what killed the 19:15 boot -- see
          ``TheEagerRunnerMustStillExist``.
        * ``_capture_cuda_graphs`` is how both runner attributes get nulled,
          exactly as the shadow-rank branch does it, and it refuses in turn
          per ``TheFlipCannotBringTheGraphsBack``.

        So the assertion here is narrow on purpose: what this gate owns is
        the prefill capture. The other refusals are owned, and proved,
        where they live.
        """
        w, calls = self._fake(_args(disable_draft_cuda_graph=True))
        with mock.patch.object(
            eagle_worker_v2, "speculative_moe_backend_context",
            lambda: contextlib.nullcontext(),
        ), mock.patch.object(
            eagle_worker_v2, "speculative_moe_a2a_backend_context",
            lambda: contextlib.nullcontext(),
        ), mock.patch.object(
            eagle_worker_v2, "check_cuda_graph_backend", lambda *a, **k: True
        ):
            w.init_cuda_graphs()
        self.assertNotIn(
            "prefill", [c[0] for c in calls],
            "the draft prefill graph was captured despite the flag "
            "(check_cuda_graph_backend is forced True here so that this "
            "assertion can actually fail)",
        )
        # `draft` IS expected: see the eager-runner note below.
        self.assertEqual([c[0] for c in calls], ["draft", "capture"])

    def test_without_the_flag_the_override_still_captures(self):
        """Can-fail proof in the other direction: the default is untouched."""
        w, calls = self._fake(_args())
        with mock.patch.object(
            eagle_worker_v2, "speculative_moe_backend_context",
            lambda: contextlib.nullcontext(),
        ), mock.patch.object(
            eagle_worker_v2, "speculative_moe_a2a_backend_context",
            lambda: contextlib.nullcontext(),
        ), mock.patch.object(
            eagle_worker_v2, "check_cuda_graph_backend", lambda *a, **k: False
        ):
            w.init_cuda_graphs()
        self.assertIn("draft", [c[0] for c in calls])
        self.assertIn("capture", [c[0] for c in calls])


class TheFlipCannotBringTheGraphsBack(unittest.TestCase):
    """``_capture_cuda_graphs`` is the flip's re-arm entry. Gate it there too.

    The assertion is on HOW FAR EXECUTION GOT, not on a captured graph:
    reading ``server_args.model_impl`` is the first thing past the gate, so
    a recording property makes "returned at the gate" and "fell through it"
    two distinguishable, dependency-free outcomes on a CPU box.
    """

    class _RecordingArgs:
        def __init__(self, disable_draft_cuda_graph):
            self.disable_draft_cuda_graph = disable_draft_cuda_graph
            self.disable_cuda_graph = False
            self.reads = []

        @property
        def model_impl(self):
            self.reads.append("model_impl")
            # Returns the one value that makes the real method stop right
            # here, so no CUDA is needed to prove the gate was passed.
            return "mindspore"

    def _fake(self, server_args):
        w = eagle_worker_v2.EagleDraftWorker.__new__(
            eagle_worker_v2.EagleDraftWorker
        )
        w.server_args = server_args
        w._spec_solo_active = False
        w._spec_solo_is_host = True
        w.cuda_graph_runner = "stale"
        w.cuda_graph_runner_for_draft_extend = "stale"
        return w

    def test_recapture_refuses_when_the_flag_is_set(self):
        args = self._RecordingArgs(True)
        w = self._fake(args)
        with mock.patch.object(
            eagle_worker_v2, "check_cuda_graph_backend", lambda *a, **k: False
        ):
            w._capture_cuda_graphs()
        self.assertEqual(
            args.reads, [], "re-capture fell through the gate; a flip would "
            "restore the draft graphs the boot asked to remove"
        )
        # The terminal state must match the shadow-rank path's: both
        # runners nulled, so every later `is None` check reads correctly.
        self.assertIsNone(w.cuda_graph_runner)
        self.assertIsNone(w.cuda_graph_runner_for_draft_extend)

    def test_recapture_proceeds_without_the_flag(self):
        args = self._RecordingArgs(False)
        w = self._fake(args)
        with mock.patch.object(
            eagle_worker_v2, "check_cuda_graph_backend", lambda *a, **k: False
        ):
            w._capture_cuda_graphs()
        self.assertEqual(args.reads, ["model_impl"])


class TheEagerRunnerMustStillExist(unittest.TestCase):
    """The regression that took all three ranks down at 19:18 on 2026-08-10.

        AttributeError: 'ModelRunner' object has no attribute 'eager_runner'
        ... in _draft_extend_for_decode -> draft_runner.forward

    The first version of this flag skipped ``init_cuda_graphs`` on the draft
    worker entirely. But ``ModelRunner.init_cuda_graphs`` does not only
    capture graphs -- it CONSTRUCTS the eager runner and aliases the prefill
    and decode runners onto it. The eager path is not the absence of the
    graph path; it is an object somebody has to build. Refusing around the
    call skipped the construction, and the failure did not surface at boot:
    it surfaced at the first draft extend, three minutes later, on every
    rank at once.

    Hence the refusal lives INSIDE ``init_cuda_graphs``, and hence this test
    asserts on the surviving state rather than on a skipped call.
    """

    def _runner(self, disable_draft):
        from sglang.srt.model_executor.model_runner import ModelRunner

        r = ModelRunner.__new__(ModelRunner)
        r.is_draft_worker = True
        # is_phase_flip_pp_stack is a computed property and is False for any
        # draft runner by construction, so the PP carve above cannot mask
        # what this test is asserting about the draft carve.
        r.server_args = _args(
            disable_draft_cuda_graph=disable_draft, enable_phase_flip=True
        )
        assert r.is_phase_flip_pp_stack is False
        return r

    def test_a_refused_draft_capture_still_leaves_a_usable_runner(self):
        from sglang.srt.model_executor import model_runner as mr

        r = self._runner(True)
        sentinel = object()
        with mock.patch.object(
            mr, "EagerRunner", lambda _self: sentinel
        ), mock.patch.object(
            mr.GraphSharedOutput, "create_for_model_runner",
            staticmethod(lambda _self: "shared"),
        ):
            r.init_cuda_graphs()

        # The four coupled assignments. `eager_runner` alone is not enough:
        # the forward path reaches the aliases, and an alias left unset is
        # the same crash one call deeper.
        self.assertIs(r.eager_runner, sentinel)
        self.assertIs(r.prefill_cuda_graph_runner, sentinel)
        self.assertIs(r.decode_cuda_graph_runner, sentinel)
        self.assertEqual(r.graph_mem_usage, 0)

    def test_the_carve_can_fail_when_the_flag_is_absent(self):
        """Can-fail proof: without the flag this must NOT take the eager exit.

        Detected by letting the real method run on past the carve, where it
        immediately reaches _harmonize_cuda_graph_plan -- a collective this
        test has no group for. Raising there is the proof it did not return
        early; returning cleanly would mean the carve swallowed a boot that
        never asked for it.
        """
        r = self._runner(False)
        with mock.patch.object(
            type(r), "_harmonize_cuda_graph_plan",
            lambda _self: (_ for _ in ()).throw(RuntimeError("went past")),
        ):
            with self.assertRaisesRegex(RuntimeError, "went past"):
                r.init_cuda_graphs()
