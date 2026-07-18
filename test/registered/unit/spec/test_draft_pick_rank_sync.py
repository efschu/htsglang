"""Rank-consistency guard for speculative draft picks (EagleDraftWorker).

On heterogeneous GPUs under (uneven) TP, per-rank logits differ in the last
bits, so any per-rank argmax/topk/sampling decision that feeds the draft
chain, the candidate tree, or the verify input desynchronizes the KV /
recurrent state across ranks unless its result is broadcast from rank 0
(#50 debugging campaign, root 1).

Two layers of protection:
1. Unit tests for the `_broadcast_draft_picks` helper semantics.
2. An AST ratchet: every draft-pick decision point in eagle_worker_v2.py
   (fast_topk / fast_sample / sample_draft_proposal / torch.argmax) must be
   followed by a `_broadcast_draft_picks` call inside the same function, so
   a new unsynced pick site cannot be added silently.
"""

import ast
import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.speculative.eagle_worker_v2 as eagle_worker_v2
from sglang.srt.speculative.eagle_worker_v2 import _broadcast_draft_picks
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

# Names whose call results are per-rank draft decisions.
_PICK_CALL_NAMES = {"fast_topk", "fast_sample", "sample_draft_proposal"}
# Functions that hold pick sites but legitimately need no broadcast because
# their inputs/outputs are already rank-consistent (none currently).
_EXEMPT_FUNCTIONS: set = set()


class TestBroadcastDraftPicks(CustomTestCase):
    def _patched(self, world_size, calls):
        fake_group = SimpleNamespace(
            world_size=world_size,
            broadcast=lambda t, src: calls.append((id(t), src)),
        )
        return (
            patch(
                "sglang.srt.distributed.get_tp_group",
                return_value=fake_group,
            ),
            patch(
                "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
                return_value=False,
            ),
        )

    def test_broadcasts_all_non_none_tensors_from_rank0(self):
        calls = []
        p1, p2 = self._patched(3, calls)
        a = torch.randn(2, 1)
        b = torch.randn(2, 1)
        with p1, p2:
            _broadcast_draft_picks(a, None, b)
        self.assertEqual(calls, [(id(a), 0), (id(b), 0)])

    def test_world_size_one_is_a_no_op(self):
        calls = []
        p1, p2 = self._patched(1, calls)
        with p1, p2:
            _broadcast_draft_picks(torch.randn(2, 1))
        self.assertEqual(calls, [])


class _PickSiteVisitor(ast.NodeVisitor):
    """Collect (function, lineno, callname) of draft-pick decision points and
    the linenos of _broadcast_draft_picks calls, per enclosing function."""

    def __init__(self):
        self.func_stack = []
        # func qualname -> list[(lineno, callname)]
        self.pick_sites = {}
        # func qualname -> list[lineno]
        self.broadcast_sites = {}

    def _qualname(self):
        return ".".join(self.func_stack) if self.func_stack else "<module>"

    def visit_FunctionDef(self, node):
        self.func_stack.append(node.name)
        self.generic_visit(node)
        self.func_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
            # torch.argmax(...) — attribute call on torch
            if name == "argmax" and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "torch":
                    name = "torch.argmax"
        qn = self._qualname()
        if name in _PICK_CALL_NAMES or name == "torch.argmax":
            self.pick_sites.setdefault(qn, []).append((node.lineno, name))
        if name == "_broadcast_draft_picks":
            self.broadcast_sites.setdefault(qn, []).append(node.lineno)
        self.generic_visit(node)


class TestPickSiteRatchet(CustomTestCase):
    def test_every_pick_site_is_followed_by_a_broadcast(self):
        source = inspect.getsource(eagle_worker_v2)
        tree = ast.parse(source)
        visitor = _PickSiteVisitor()
        visitor.visit(tree)

        # Sanity: the known sites must be found, otherwise the ratchet is dead.
        all_picks = [
            (qn, lineno, name)
            for qn, sites in visitor.pick_sites.items()
            for lineno, name in sites
        ]
        self.assertGreaterEqual(
            len(all_picks),
            4,
            f"expected at least 4 draft-pick sites, ratchet broken? {all_picks}",
        )

        unsynced = []
        for qn, sites in visitor.pick_sites.items():
            if qn in _EXEMPT_FUNCTIONS:
                continue
            broadcasts = visitor.broadcast_sites.get(qn, [])
            for lineno, name in sites:
                # A pick is covered if a broadcast happens later in the same
                # function (all current sites broadcast right after picking).
                if not any(b > lineno for b in broadcasts):
                    unsynced.append((qn, lineno, name))
        self.assertEqual(
            unsynced,
            [],
            "Per-rank draft pick(s) without a following _broadcast_draft_picks "
            "in the same function — on heterogeneous GPUs this desynchronizes "
            f"TP ranks (see module docstring): {unsynced}",
        )


class _FuncScopedVisitor(ast.NodeVisitor):
    """Base visitor that tracks the enclosing function qualname."""

    def __init__(self):
        self.func_stack = []

    def _qualname(self):
        return ".".join(self.func_stack) if self.func_stack else "<module>"

    def visit_FunctionDef(self, node):
        self.func_stack.append(node.name)
        self.generic_visit(node)
        self.func_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef


class TestAdaptiveAcceptFeedRatchet(CustomTestCase):
    """Determinism ratchet for the adaptive draft-length controller (#50).

    The EMA that picks the speculative step count must be fed ONLY the
    rank-0-broadcast accept counts. The chain is:

      eagle_utils.eagle_sample:
          tp_group.broadcast(num_correct_drafts, src=0)
          -> returns num_correct_drafts + 1 (= accept_lens)
      batch_result_processor._resolve_spec_v2_tokens:
          accept_lens = result.accept_lens.tolist()
          result.num_correct_drafts_per_req_cpu = [x - 1 for x in accept_lens]
          model_worker.on_verify_complete_cpu(
              result.num_correct_drafts_per_req_cpu, ...)
      worker.on_verify_complete_cpu:
          adaptive_controller.on_verify_complete(...)

    A future refactor feeding the controller a locally computed (per-rank)
    statistic would silently reintroduce rank-divergent graph swaps. These
    AST checks pin each link of the chain.
    """

    def test_eagle_sample_broadcasts_num_correct_drafts(self):
        import sglang.srt.speculative.eagle_utils as eagle_utils

        tree = ast.parse(inspect.getsource(eagle_utils))

        class V(_FuncScopedVisitor):
            found = False

            def visit_Call(self, node):
                if (
                    "eagle_sample" in self.func_stack
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "broadcast"
                    and any(
                        isinstance(a, ast.Name) and a.id == "num_correct_drafts"
                        for a in node.args
                    )
                ):
                    self.found = True
                self.generic_visit(node)

        v = V()
        v.visit(tree)
        self.assertTrue(
            v.found,
            "eagle_sample no longer broadcasts num_correct_drafts from rank 0 "
            "— the adaptive EMA (and verify itself) would diverge across "
            "ranks on heterogeneous GPUs (#50).",
        )

    def test_adaptive_feed_derives_from_broadcast_accept_lens(self):
        import sglang.srt.managers.scheduler_components.batch_result_processor as brp

        tree = ast.parse(inspect.getsource(brp))

        class V(_FuncScopedVisitor):
            def __init__(self):
                super().__init__()
                self.feed_calls = []  # (func, first_arg_node)
                self.percpu_assign_funcs = {}  # func -> value node
                self.accept_lens_assign_funcs = {}  # func -> value node

            def visit_Call(self, node):
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "on_verify_complete_cpu"
                ):
                    self.feed_calls.append(
                        (self._qualname(), node.args[0] if node.args else None)
                    )
                self.generic_visit(node)

            def visit_Assign(self, node):
                for t in node.targets:
                    if (
                        isinstance(t, ast.Attribute)
                        and t.attr == "num_correct_drafts_per_req_cpu"
                        and isinstance(t.value, ast.Name)
                        and t.value.id == "result"
                    ):
                        self.percpu_assign_funcs[self._qualname()] = node.value
                    if isinstance(t, ast.Name) and t.id == "accept_lens":
                        self.accept_lens_assign_funcs[self._qualname()] = node.value
                self.generic_visit(node)

        v = V()
        v.visit(tree)

        self.assertGreaterEqual(
            len(v.feed_calls), 1, "no on_verify_complete_cpu feed site found"
        )
        for func, arg in v.feed_calls:
            # The fed value must be exactly result.num_correct_drafts_per_req_cpu.
            self.assertIsInstance(arg, ast.Attribute, f"feed in {func} not a result attr")
            self.assertEqual(arg.attr, "num_correct_drafts_per_req_cpu")
            self.assertIsInstance(arg.value, ast.Name)
            self.assertEqual(arg.value.id, "result")

            # In the same function: num_correct_drafts_per_req_cpu is a pure
            # elementwise transform of accept_lens ...
            value = v.percpu_assign_funcs.get(func)
            self.assertIsNotNone(
                value, f"{func}: result.num_correct_drafts_per_req_cpu not assigned"
            )
            self.assertIsInstance(value, ast.ListComp)
            self.assertEqual(len(value.generators), 1)
            src_iter = value.generators[0].iter
            self.assertIsInstance(src_iter, ast.Name)
            self.assertEqual(src_iter.id, "accept_lens")

            # ... and accept_lens comes straight off result.accept_lens (the
            # rank-0-broadcast verify output), not a local recompute.
            al_value = v.accept_lens_assign_funcs.get(func)
            self.assertIsNotNone(al_value, f"{func}: accept_lens not assigned")
            self.assertIsInstance(al_value, ast.Call)
            self.assertIsInstance(al_value.func, ast.Attribute)
            self.assertEqual(al_value.func.attr, "tolist")
            src = al_value.func.value
            self.assertIsInstance(src, ast.Attribute)
            self.assertEqual(src.attr, "accept_lens")
            self.assertIsInstance(src.value, ast.Name)
            self.assertEqual(src.value.id, "result")

    def test_controller_fed_only_inside_on_verify_complete_cpu(self):
        # The only production call sites of adaptive_controller.on_verify_complete
        # must live inside an on_verify_complete_cpu hook (which receives the
        # broadcast-derived accept counts checked above).
        import os

        import sglang.srt.speculative as spec_pkg

        offenders = []
        found_any = False
        # Namespace package: __file__ is None, use __path__.
        spec_dir = list(spec_pkg.__path__)[0]
        for fname in sorted(os.listdir(spec_dir)):
            if not fname.endswith(".py"):
                continue
            with open(os.path.join(spec_dir, fname)) as f:
                tree = ast.parse(f.read())

            class V(_FuncScopedVisitor):
                def __init__(self, fname):
                    super().__init__()
                    self.fname = fname
                    self.hits = []

                def visit_Call(self, node):
                    if (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr == "on_verify_complete"
                        and isinstance(node.func.value, ast.Attribute)
                        and node.func.value.attr == "adaptive_controller"
                    ):
                        self.hits.append((self._qualname(), node.lineno))
                    self.generic_visit(node)

            v = V(fname)
            v.visit(tree)
            for qn, lineno in v.hits:
                found_any = True
                if not qn.endswith("on_verify_complete_cpu"):
                    offenders.append((fname, qn, lineno))

        self.assertTrue(
            found_any,
            "no adaptive_controller.on_verify_complete call site found — "
            "ratchet is dead, update the test",
        )
        self.assertEqual(
            offenders,
            [],
            "adaptive_controller.on_verify_complete called outside an "
            "on_verify_complete_cpu hook — the EMA would be fed a value whose "
            f"rank-0-broadcast provenance is unchecked (#50): {offenders}",
        )


if __name__ == "__main__":
    unittest.main()
