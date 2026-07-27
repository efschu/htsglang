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

            @staticmethod
            def _arg_names(node):
                # Flatten tuple/list literals so both call shapes count:
                #   tp_group.broadcast(num_correct_drafts, src=0)
                #   capture_safe_tp_broadcast(g, (predict, ..., num_correct_drafts), src=0)
                names = []
                for a in node.args:
                    if isinstance(a, (ast.Tuple, ast.List)):
                        names.extend(
                            e.id for e in a.elts if isinstance(e, ast.Name)
                        )
                    elif isinstance(a, ast.Name):
                        names.append(a.id)
                return names

            def visit_Call(self, node):
                is_broadcast_call = (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "broadcast"
                ) or (
                    # Capture/co-location-safe variant (pynccl-preferred);
                    # see spec_utils.capture_safe_tp_broadcast.
                    isinstance(node.func, ast.Name)
                    and node.func.id == "capture_safe_tp_broadcast"
                )
                if (
                    "eagle_sample" in self.func_stack
                    and is_broadcast_call
                    and "num_correct_drafts" in self._arg_names(node)
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


def _annotate_parents(tree):
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node
    return tree


def _call_name(node):
    """Name of a Call node, with torch.argmax spelled out."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        if (
            node.func.attr == "argmax"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
        ):
            return "torch.argmax"
        return node.func.attr
    return None


def _enclosing_func(node):
    cur = getattr(node, "parent", None)
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return cur
        cur = getattr(cur, "parent", None)
    return None


def _innermost_loop(node):
    """The innermost For/While enclosing *node* within its function, or None."""
    cur = getattr(node, "parent", None)
    while cur is not None and not isinstance(
        cur, (ast.FunctionDef, ast.AsyncFunctionDef)
    ):
        if isinstance(cur, (ast.For, ast.AsyncFor, ast.While)):
            return cur
        cur = getattr(cur, "parent", None)
    return None


# Multi-layer EAGLE pick sites. `replay` is a pick site too: the per-step
# draft-extend graph computes fast_topk IN-graph
# (MultiLayerEagleDraftExtendCudaGraphRunner._compute_topk), so the worker
# receives an already-made per-rank pick that the source-level names below
# cannot see.
_ML_PICK_CALL_NAMES = {
    "fast_topk",
    "fast_sample",
    "sample_draft_proposal",
    "torch.argmax",
    "replay",
}


class TestMultiLayerPickSiteRatchet(CustomTestCase):
    """#185 / #50-family: the multi-layer EAGLE worker draws its chain picks
    per rank in `_draft_extend_for_prefill` / `_draft_extend_for_decode`, one
    per MTP rung. Each rung's pick feeds the NEXT rung's input_ids (the chain
    rotation) and thus that rung's KV writes, so a single divergent pick on a
    heterogeneous-GPU TP rank desynchronizes the group for the rest of the
    sequence — exactly what `_broadcast_draft_picks` exists to prevent in
    eagle_worker_v2.

    Stricter than the eagle_worker_v2 ratchet above: the broadcast must sit in
    the SAME innermost loop as the pick, so one broadcast at the end of the
    function cannot vacuously cover k rungs.
    """

    def _sites(self):
        import sglang.srt.speculative.multi_layer_eagle_worker_v2 as ml

        tree = _annotate_parents(ast.parse(inspect.getsource(ml)))
        picks, broadcasts, rotations = [], [], []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name in _ML_PICK_CALL_NAMES:
                picks.append(node)
            elif name == "_broadcast_draft_picks":
                broadcasts.append(node)
            elif name == "rotate_input_ids":
                rotations.append(node)
        return picks, broadcasts, rotations

    def test_every_pick_site_is_broadcast_inside_its_own_rung_loop(self):
        picks, broadcasts, _ = self._sites()
        self.assertGreaterEqual(
            len(picks),
            5,
            "expected at least 5 multi-layer draft-pick sites "
            "(prefill loop, graph replay, eager loop) — ratchet broken?",
        )

        unsynced = []
        for pick in picks:
            func = _enclosing_func(pick)
            loop = _innermost_loop(pick)
            covered = any(
                _enclosing_func(b) is func
                and _innermost_loop(b) is loop
                and b.lineno > pick.lineno
                for b in broadcasts
            )
            if not covered:
                unsynced.append(
                    (func.name if func else "<module>", pick.lineno, _call_name(pick))
                )
        self.assertEqual(
            unsynced,
            [],
            "Multi-layer EAGLE draft pick(s) without a _broadcast_draft_picks "
            "in the same rung loop — on heterogeneous GPUs every one of these "
            f"desynchronizes the TP group (#185, #50 family): {unsynced}",
        )

    def test_broadcast_precedes_the_chain_rotation(self):
        """Ordering, not just presence: the pick is rotated into the NEXT
        rung's input_ids, so the sync has to land BEFORE the rotation."""
        picks, broadcasts, rotations = self._sites()
        late = []
        for pick in picks:
            loop = _innermost_loop(pick)
            if loop is None:
                continue
            rots = [
                r
                for r in rotations
                if _innermost_loop(r) is loop and r.lineno > pick.lineno
            ]
            if not rots:
                continue
            first_rot = min(r.lineno for r in rots)
            if not any(
                _innermost_loop(b) is loop and pick.lineno < b.lineno < first_rot
                for b in broadcasts
            ):
                late.append((pick.lineno, _call_name(pick), first_rot))
        self.assertEqual(
            late,
            [],
            "Draft pick rotated into the next rung's input_ids before being "
            f"rank-0-broadcast (pick_lineno, call, rotate_lineno): {late}",
        )


class TestBroadcastAdoptsRank0Picks(CustomTestCase):
    """Semantics of the sync itself: after the broadcast every rank holds
    rank 0's picks, and a world_size-1 run is byte-identically untouched."""

    def test_divergent_rank_adopts_rank0_values(self):
        rank0 = (torch.tensor([[5]]), torch.tensor([[0.7]]))
        rank1_index = torch.tensor([[7]])
        rank1_p = torch.tensor([[0.6]])

        payloads = iter([rank0[0], rank0[1]])
        fake_group = SimpleNamespace(
            world_size=2,
            broadcast=lambda t, src: t.copy_(next(payloads)),
        )
        with (
            patch("sglang.srt.distributed.get_tp_group", return_value=fake_group),
            patch(
                "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
                return_value=False,
            ),
        ):
            _broadcast_draft_picks(rank1_index, rank1_p)

        self.assertTrue(torch.equal(rank1_index, rank0[0]))
        self.assertTrue(torch.equal(rank1_p, rank0[1]))

    def test_single_rank_path_is_untouched(self):
        index = torch.tensor([[7]])
        before = index.clone()
        fake_group = SimpleNamespace(
            world_size=1,
            broadcast=lambda t, src: t.fill_(-1),
        )
        with (
            patch("sglang.srt.distributed.get_tp_group", return_value=fake_group),
            patch(
                "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
                return_value=False,
            ),
        ):
            _broadcast_draft_picks(index)
        self.assertTrue(torch.equal(index, before))


if __name__ == "__main__":
    unittest.main()
