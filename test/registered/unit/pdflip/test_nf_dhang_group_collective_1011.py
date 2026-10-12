# SPDX-License-Identifier: Apache-2.0
"""NF D-HANG 1011: a group collective after the group ADMIT is entered by EVERY rank.

SPECIMEN (NF int24d, 11.10. 10:25:45Z, group D, TP0/TP1/TP2): the ranks parted at the admission of
pdflip-0-15. TP0 stood in the extend forward, TP1/TP2 inside the admission pass (watchdog: "1 queued,
0 running", CLASS=A-PIPELINE-DEAD). The exact blocking line is unproven (no stack was written; see
test_wedge_stack_forward_hang_1011). One live instance of the class "ranks decide rank-locally over
a collective" was found and is what this file pins:

    schedule_policy.add_one_req: a worker reads the group's ADMIT, loads back, finds no device room
    and returns NO_TOKEN ALONE (PDFLIP-LOADBACK-WAIT); the host admits.
    -> ``HolPass.head`` is set on that worker only
    -> scheduler.py entered the D-WALL-HEAD group MIN "iff _hol.head is not None"
    -> only that worker entered the all_reduce; the host went on into the forward.

The fix: ``pick_unservable_head`` is called by every rank whenever D-HOL is on (a replicated
condition); the head is the PAYLOAD. These tests run THREE ranks as threads against a thread-barrier
all_reduce (a rank that does not enter leaves the others blocked -> BrokenBarrierError), through the
real ``Scheduler._pdflip_group_min_flags``.

MUTANT (old path): ``test_old_entry_condition_blocks_the_group`` shows the harness goes red on the
old call shape; ``test_scheduler_calls_the_vote_under_hol_on_not_under_head`` is the same mutant on
the scheduler source (swap ``if _hol.on:`` back to ``if _hol.head is not None:`` -> red).
"""

import ast
import os
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from flliper.srt.managers.scheduler import Scheduler  # noqa: E402
from flliper.srt.pdflip import d_wall_head as W  # noqa: E402
from flliper.srt.pdflip.hol_overtake import HolPass  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

N = 3
HEAD = "pdflip-0-15"
AVAILABLE = 1408
SCHED_PY = Path(__file__).resolve()
for _p in SCHED_PY.parents:
    if (_p / "python" / "flliper" / "srt" / "managers" / "scheduler.py").exists():
        SCHEDULER_SRC = _p / "python" / "flliper" / "srt" / "managers" / "scheduler.py"
        break


class _Fabric:
    """A thread-barrier MIN all_reduce over N ranks; a rank that never calls leaves the rest blocked."""

    def __init__(self, n=N, timeout=3.0):
        self.barrier = threading.Barrier(n, timeout=timeout)
        self.lock = threading.Lock()
        self.vals = {}
        self.entered = {}

    def all_reduce(self, tensor, op=None, group=None):
        me = threading.current_thread().name
        with self.lock:
            self.vals[me] = tensor.clone()
            self.entered[me] = self.entered.get(me, 0) + 1
        self.barrier.wait()
        with self.lock:
            stacked = torch.stack([v for v in self.vals.values()])
        tensor.copy_(stacked.min(dim=0).values)
        self.barrier.wait()
        with self.lock:
            self.vals.pop(me, None)


def _rank(fabric, rank):
    ns = types.SimpleNamespace(
        ps=types.SimpleNamespace(tp_size=N, pp_size=1, tp_rank=rank),
        tp_cpu_group=object(),
        _pp_blocked_recv_since=None,
        _pp_blocked_recv_arm=None,
    )
    ns._pdflip_group_min_flags = types.MethodType(Scheduler._pdflip_group_min_flags, ns)
    return ns


class _Tree:
    """A tree under a measured backup wall whose evictable room is not payable."""

    def __init__(self, walled=True):
        self.walled = walled

    def evictable_size(self):
        return 430528


def _req(rid=HEAD, n=4813):
    return types.SimpleNamespace(
        rid=rid,
        full_untruncated_fill_ids=list(range(n)),
        prefix_indices=torch.empty(0, dtype=torch.int64),
    )


def _run_ranks(fn, wall=None):
    """Run ``fn(rank_index, sched)`` on N threads; return (results, errors, fabric).

    ``wall``: rank -> this rank's own wall verdict (default: every rank's wall says unservable).
    ``wall_unservable`` is patched ONCE, here, on the calling thread and answers per rank thread --
    patching it inside the rank threads would interleave the patch/restore pairs and leave a mock
    behind for every later test in the process."""
    fab = _Fabric()
    res, err = {}, {}
    wall = wall or {}

    def per_rank_wall(**kw):
        rank = int(threading.current_thread().name[len("rank"):])
        return bool(wall.get(rank, True))

    def body(i):
        try:
            res[i] = fn(i, _rank(fab, i))
        except BaseException as exc:  # noqa: BLE001
            err[i] = exc

    threads = [threading.Thread(target=body, args=(i,), name=f"rank{i}") for i in range(N)]
    with mock.patch.object(torch.distributed, "is_initialized", return_value=True), mock.patch.object(
        torch.distributed, "all_reduce", side_effect=fab.all_reduce
    ), mock.patch.object(W, "wall_unservable", side_effect=per_rank_wall), mock.patch.dict(
        os.environ, {"FLLIPER_PDFLIP_GROUP": "D"}
    ):
        for t in threads:
            t.start()
        for t in threads:
            t.join(15)
    return res, err, fab


def _scheduler_pass(i, sched, *, head_on):
    """The D-WALL-HEAD tail of one admission pass, as scheduler.py runs it, for rank ``i``.

    ``head_on``: which ranks had a NO_TOKEN in the loop (HolPass.head set) -- the worker-only
    PDFLIP-LOADBACK-WAIT shape is ``head_on = {1, 2}`` (the host admitted)."""
    hol = HolPass(types.SimpleNamespace(ps=types.SimpleNamespace(pp_size=1)))  # env: group D, set once by _run_ranks
    if i in head_on:
        hol.head = HEAD
    waiting = [_req()]
    got = None
    if hol.on:  # THE FIXED ENTRY CONDITION: replicated (switch + group D + pp_size 1)
        got = W.pick_unservable_head(
            head_rid=hol.head,
            waiting=waiting,
            tree=_Tree(),
            available=AVAILABLE,
            group_min=lambda flags: sched._pdflip_group_min_flags(flags, site="d-wall-head"),
        )
    return got


class ThreeRankWallHeadTest(unittest.TestCase):
    def test_worker_only_no_token_every_rank_enters_and_the_answer_is_uniform(self):
        """THE SPECIMEN SHAPE: the host admitted, TP1/TP2 got a local NO_TOKEN. All three enter the
        MIN once and all three get the same answer (None: the host has no head)."""
        res, err, fab = _run_ranks(lambda i, s: _scheduler_pass(i, s, head_on={1, 2}))
        self.assertEqual(err, {})
        self.assertEqual(res, {0: None, 1: None, 2: None})
        self.assertEqual(fab.entered, {"rank0": 1, "rank1": 1, "rank2": 1})

    def test_every_rank_with_the_same_wall_unservable_head_answers_it(self):
        res, err, fab = _run_ranks(lambda i, s: _scheduler_pass(i, s, head_on={0, 1, 2}))
        self.assertEqual(err, {})
        self.assertTrue(all(res[i] is not None and res[i].rid == HEAD for i in range(N)), res)
        self.assertEqual(fab.entered, {"rank0": 1, "rank1": 1, "rank2": 1})

    def test_one_rank_with_room_keeps_the_head_everywhere(self):
        def one(i, s):
            return W.pick_unservable_head(
                head_rid=HEAD, waiting=[_req()], tree=_Tree(), available=AVAILABLE,
                group_min=lambda f: s._pdflip_group_min_flags(f, site="d-wall-head"))

        res, err, fab = _run_ranks(one, wall={2: False})
        self.assertEqual(err, {})
        self.assertEqual(res, {0: None, 1: None, 2: None})

    def test_no_head_anywhere_still_costs_exactly_one_vote_per_rank(self):
        res, err, fab = _run_ranks(lambda i, s: _scheduler_pass(i, s, head_on=set()))
        self.assertEqual(err, {})
        self.assertEqual(res, {0: None, 1: None, 2: None})
        self.assertEqual(fab.entered, {"rank0": 1, "rank1": 1, "rank2": 1})

    def test_different_heads_are_answered_by_nobody(self):
        def split(i, s):
            rid = HEAD if i != 2 else "pdflip-0-16"
            return W.pick_unservable_head(
                head_rid=rid, waiting=[_req(rid)], tree=_Tree(), available=AVAILABLE,
                group_min=lambda f: s._pdflip_group_min_flags(f, site="d-wall-head"))

        res, err, fab = _run_ranks(split)
        self.assertEqual(err, {})
        self.assertEqual(res, {0: None, 1: None, 2: None})

    def test_old_entry_condition_blocks_the_group(self):
        """MUTANT = the old path: the vote entered iff this rank's own head is set. With the
        worker-only NO_TOKEN only TP1/TP2 enter -> the barrier never completes. This is what the
        3-rank harness must detect (red for the old shape, green above for the new)."""

        def old(i, s):
            hol = HolPass.__new__(HolPass)
            hol.head = HEAD if i in {1, 2} else None
            if hol.head is not None:  # the old condition
                return W.pick_unservable_head(
                    head_rid=hol.head, waiting=[_req()], tree=_Tree(), available=AVAILABLE,
                    group_min=lambda f: s._pdflip_group_min_flags(f, site="d-wall-head"))
            return None

        res, err, fab = _run_ranks(old)
        # rank 0 never entered; the barrier of the two that did could not complete (production: a
        # gloo all_reduce that never returns; here the fabric times out)
        self.assertEqual(fab.entered.get("rank0", 0), 0)
        self.assertTrue(fab.barrier.broken, "the ranks that entered must be left blocked")
        # and the group's decision is split: the timed-out ranks fall back to their local flags and
        # answer the head, rank 0 does not -- the picture the new entry condition makes impossible
        self.assertIsNone(res[0])
        self.assertTrue(res[1] is not None and res[2] is not None, res)


class BlockedStampTest(unittest.TestCase):
    """I3: a named call site stamps the blocked-receive pair for the duration of the all_reduce."""

    def _sched(self):
        return _rank(None, 0)

    def test_site_names_the_arm_while_waiting_and_restores_after(self):
        s = self._sched()
        s._pp_blocked_recv_since, s._pp_blocked_recv_arm = 12.5, "chain-recv/outer"
        seen = {}

        def ar(t, op=None, group=None):
            seen["arm"], seen["since"] = s._pp_blocked_recv_arm, s._pp_blocked_recv_since

        with mock.patch.object(torch.distributed, "is_initialized", return_value=True), mock.patch.object(
            torch.distributed, "all_reduce", side_effect=ar
        ):
            out = s._pdflip_group_min_flags([True, False], site="d-wall-head")
        self.assertEqual(seen["arm"], "group-min/d-wall-head")
        self.assertIsNotNone(seen["since"])
        self.assertNotEqual(seen["since"], 12.5)
        self.assertEqual((s._pp_blocked_recv_since, s._pp_blocked_recv_arm), (12.5, "chain-recv/outer"))
        self.assertEqual(out, [1, 0])

    def test_no_site_no_stamp_and_a_raise_restores(self):
        s = self._sched()
        seen = []

        def ar(t, op=None, group=None):
            seen.append(s._pp_blocked_recv_arm)
            if len(seen) == 2:
                raise RuntimeError("gloo gone")

        with mock.patch.object(torch.distributed, "is_initialized", return_value=True), mock.patch.object(
            torch.distributed, "all_reduce", side_effect=ar
        ):
            s._pdflip_group_min_flags([True])
            out = s._pdflip_group_min_flags([True], site="x")
        self.assertEqual(seen, [None, "group-min/x"])
        self.assertEqual(out, [1])  # local flags on a raise, as before
        self.assertEqual((s._pp_blocked_recv_since, s._pp_blocked_recv_arm), (None, None))


def _pass_function():
    tree = ast.parse(SCHEDULER_SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_get_new_batch_prefill_raw":
            return node
    raise AssertionError("_get_new_batch_prefill_raw not found")


class SchedulerSourcePinTest(unittest.TestCase):
    def test_scheduler_calls_the_vote_under_hol_on_not_under_head(self):
        """MUTANT = swap the guard back to ``_hol.head is not None``: red. The vote's entry
        condition may read only replicated state; a rank-local head is its payload."""
        fn = _pass_function()
        calls = []

        class V(ast.NodeVisitor):
            def __init__(self):
                self.guards = []

            def visit_If(self, n):
                self.guards.append(ast.unparse(n.test))
                for b in n.body:
                    self.visit(b)
                self.guards.pop()
                for b in n.orelse:
                    self.visit(b)

            def visit_Call(self, n):
                if ast.unparse(n.func).endswith("pick_unservable_head"):
                    calls.append(list(self.guards))
                self.generic_visit(n)

        V().visit(fn)
        self.assertEqual(len(calls), 1, "exactly one D-WALL-HEAD vote in the admission pass")
        guards = calls[0]
        self.assertIn("_hol.on", guards)
        self.assertFalse(
            any("_hol.head" in g for g in guards),
            f"the vote's entry condition reads the rank-local head: {guards}",
        )


if __name__ == "__main__":
    unittest.main()
