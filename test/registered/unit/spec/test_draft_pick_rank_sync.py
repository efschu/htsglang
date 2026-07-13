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


if __name__ == "__main__":
    unittest.main()
