# SPDX-License-Identifier: Apache-2.0
"""H62 (NF D rounds, 24.09.): the SECOND HiCache CPU collective of a D round on a
rank-uniform cadence, behind SGLANG_HICACHE_RETIRED_AGREE_EVERY (default off).

MEASURED (boot fnFL2x165, D = TP 3, all 72 DECODE-HOST-PERIOD windows):
allreduce_n = 2 x hicache_calls (128/64 ... 134/67), while hicache_ms ~= drain_ms
~= 0.4-0.5 ms and allreduce_ms = 0.9-1.0 ms. One HiCache collective sits inside
``check_hicache_events`` (the storage-queue agreement, gated by
SGLANG_HICACHE_DRAIN_AGREE_EVERY since 0619f1280f) and one OUTSIDE it: the #939
retired-prefetch agreement ``UnifiedRadixCache.drain_retired_prefetch``, called
once per scheduler iteration from ``Scheduler._drain_prefetch_progress`` (TP
loop: ``_update_uniform_pool_budget``). The retired list was empty on every rank
for the whole boot (no #939 REAPED line) -- a gloo all_reduce per round for
nothing.

WHAT MUST HOLD. Unset is the unchanged every-round path. On, every rank enters
the agreement on exactly the same rounds -- a rank alone in a gloo all_reduce is
the wedge, and the #939 gloo test measured the quieter failure too: gloo pairs
one rank's round-1 op with another's round-2 op and nobody notices -- and every
retired record is still reaped on every rank. The in-process group below checks
(label, round) identity at EVERY collective, which a real group cannot; the
gloo class runs the real code over three real ranks.
"""

import os
import threading
import types
import unittest
import warnings

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.environ import envs
from sglang.srt.mem_cache import unified_radix_cache as u
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

ENV = "SGLANG_HICACHE_RETIRED_AGREE_EVERY"
LABEL = "drain_retired_prefetch"
SPAN = 4
_TIMEOUT_S = 5.0


class _Group:
    """WORLD ranks on threads, one all_reduce at a time. Every rank must enter
    the SAME collective on the SAME round, else every rank raises; a rank that
    never arrives leaves the others in ``Barrier.wait`` until the timeout."""

    def __init__(self, world, timeout=_TIMEOUT_S):
        self._slots = [None] * world
        self._out = None
        self._err = None
        self._b1 = threading.Barrier(world, timeout=timeout)
        self._b2 = threading.Barrier(world, timeout=timeout)

    def all_reduce(self, rank, rnd, label, tensor, op):
        self._slots[rank] = (label, rnd, tensor.clone())
        if self._b1.wait() == 0:
            keys = sorted({(s[0], s[1]) for s in self._slots})
            if len(keys) != 1:
                self._err, self._out = f"ranks in different collectives: {keys}", None
            else:
                stack = torch.stack([s[2] for s in self._slots])
                self._err = None
                self._out = (
                    stack.min(dim=0).values
                    if op == dist.ReduceOp.MIN
                    else stack.max(dim=0).values
                )
        self._b2.wait()
        if self._err:
            raise AssertionError(self._err)
        tensor.copy_(self._out)


def _make_record(req_id):
    operation = types.SimpleNamespace(request_id=req_id, binding_generation=None)
    return u._OngoingPrefetch(
        anchor_node=None,
        prefetch_key=list(range(SPAN)),
        host_indices=torch.arange(SPAN, dtype=torch.int64),
        operation=operation,
        anchor_lock_params=None,
        comp_xfers={},
    )


def _rank_cache(group, rank, log, tp_world_size):
    """The REAL drain_retired_prefetch (and gate) on a cache that carries only
    what it reads; the all_reduce goes to the in-process group and is logged
    with the round it was entered on."""
    c = u.UnifiedRadixCache.__new__(u.UnifiedRadixCache)
    c.attn_cp_group = None
    c.attn_tp_group = None
    c.tp_world_size = tp_world_size
    c.cache_controller = types.SimpleNamespace(
        prefetch_tokens_occupied=0,
        terminate_prefetch=lambda op: (SPAN, []),
        append_host_mem_release=lambda host_indices=None, generation=None: None,
    )
    c._retired_prefetch = []
    c._retired_prefetch_reaped = 0
    c.dec_host_lock_ref = lambda node, params: None
    cur = {"round": 0}

    def all_reduce(tensor, op, label="hicache"):
        log.append((cur["round"], label))
        group.all_reduce(rank, cur["round"], label, tensor, op)

    def can_terminate(operation, tail_hold=0):
        # the production call is a MAX all_reduce over the same group -- what
        # the agreement protects is that every rank enters it together
        all_reduce(torch.tensor([0], dtype=torch.int), dist.ReduceOp.MAX,
                   label="can_terminate_prefetch")
        return True

    c._all_reduce_attn_groups = all_reduce
    c.can_terminate_prefetch = can_terminate
    return c, cur


def _run(arrivals, *, rounds, world=3, tp_world_size=3, mutate=None, timeout=_TIMEOUT_S):
    group = _Group(world, timeout=timeout)
    logs = [[] for _ in range(world)]
    ranks = [_rank_cache(group, r, logs[r], tp_world_size) for r in range(world)]
    if mutate is not None:
        for c, _cur in ranks:
            mutate(c)
    outcome = [None] * world

    def rank_main(r):
        c, cur = ranks[r]
        try:
            for rnd in range(1, rounds + 1):
                cur["round"] = rnd
                for req in arrivals(r, rnd):
                    c._retired_prefetch.append(_make_record(req))
                c.drain_retired_prefetch()
            outcome[r] = "ok"
        except Exception as exc:  # noqa: BLE001 -- the outcome IS the finding
            outcome[r] = f"{type(exc).__name__}: {exc}"

    threads = [threading.Thread(target=rank_main, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    return outcome, logs, [c for c, _cur in ranks]


def _entered(log, label=LABEL):
    return [rnd for rnd, lab in log if lab == label]


def _idle(r, rnd):
    return ()


class TheDefaultIsTheUnchangedPath(CustomTestCase):
    def setUp(self):
        super().setUp()
        self._saved = os.environ.pop(ENV, None)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = self._saved

    def test_unset_agrees_on_every_round(self):
        outcome, logs, caches = _run(_idle, rounds=12)
        self.assertEqual(outcome, ["ok"] * 3)
        for log in logs:
            self.assertEqual(_entered(log), list(range(1, 13)))
        for c in caches:
            self.assertFalse(hasattr(c, "_retired_gate_round"))

    def test_every_1_is_the_default(self):
        with envs.SGLANG_HICACHE_RETIRED_AGREE_EVERY.override("1"):
            outcome, logs, _c = _run(_idle, rounds=5)
        self.assertEqual(outcome, ["ok"] * 3)
        self.assertEqual(_entered(logs[0]), [1, 2, 3, 4, 5])

    def test_garbage_and_nonpositive_values_are_every_round(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for v in ("x", "0", "-3", ""):
                with envs.SGLANG_HICACHE_RETIRED_AGREE_EVERY.override(v):
                    self.assertEqual(u._hicache_retired_agree_every(), 1, v)
        with envs.SGLANG_HICACHE_RETIRED_AGREE_EVERY.override("8"):
            self.assertEqual(u._hicache_retired_agree_every(), 8)

    def test_a_group_without_a_collective_never_engages_the_gate(self):
        """Group P (TP 1 / PP 3): the agreement reduces over nobody, so the
        cadence never engages there, even with the variable in its env."""
        with envs.SGLANG_HICACHE_RETIRED_AGREE_EVERY.override("8"):
            outcome, logs, caches = _run(_idle, rounds=9, world=1, tp_world_size=1)
        self.assertEqual(outcome, ["ok"])
        self.assertFalse(caches[0]._drain_agreement_is_collective())
        self.assertEqual(_entered(logs[0]), list(range(1, 10)))
        self.assertFalse(hasattr(caches[0], "_retired_gate_round"))


class TheCadenceIsRankUniform(CustomTestCase):
    """Three D ranks whose records retire on DIFFERENT rounds."""

    def setUp(self):
        super().setUp()
        self._override = envs.SGLANG_HICACHE_RETIRED_AGREE_EVERY.override("8")
        self._override.__enter__()

    def tearDown(self):
        self._override.__exit__(None, None, None)

    def test_every_rank_enters_on_the_same_rounds_and_every_record_is_reaped(self):
        def arrivals(r, rnd):
            out = []
            if rnd == 10 + r:  # rank 2 retires req-a two rounds after rank 0
                out.append("req-a")
            if rnd == 40:
                out.append("req-b")
            if rnd == (47 if r == 2 else 41):  # a latecomer inside a cold stretch
                out.append("req-c")
            return out

        outcome, logs, caches = _run(arrivals, rounds=80)
        self.assertEqual(outcome, ["ok"] * 3)
        # the whole collective sequence (agreement AND can_terminate), round by round
        self.assertEqual(logs[0], logs[1])
        self.assertEqual(logs[1], logs[2])
        for c in caches:
            self.assertEqual(c._retired_prefetch, [], "an agreed record was never reaped")
            self.assertEqual(c._retired_prefetch_reaped, 3)
        entered = _entered(logs[0])
        self.assertLess(len(entered), 80 // 3, "the cadence did not thin the agreement")
        self.assertEqual(len(_entered(logs[0], "can_terminate_prefetch")), 3)

    def test_an_idle_group_agrees_every_nth_round(self):
        outcome, logs, caches = _run(_idle, rounds=64)
        self.assertEqual(outcome, ["ok"] * 3)
        for log in logs:
            self.assertEqual(_entered(log), [1, 9, 17, 25, 33, 41, 49, 57])
        for c in caches:
            self.assertEqual(c._retired_gate_skipped, 56)

    def test_a_group_with_records_agrees_every_round(self):
        outcome, logs, caches = _run(lambda r, rnd: (f"req-{rnd}",), rounds=30)
        self.assertEqual(outcome, ["ok"] * 3)
        self.assertEqual(_entered(logs[0]), list(range(1, 31)))
        for c in caches:
            self.assertEqual(c._retired_prefetch_reaped, 30)

    def test_the_harness_catches_a_rank_local_gate(self):
        """The mutant: skip on THIS rank's empty list (the #580 shape). Rank 0
        enters at round 10 while ranks 1 and 2 skip it -- the group must fail,
        or the tests above prove nothing."""

        def mutate(c):
            c._retired_agreement_due = lambda: bool(c._retired_prefetch)

        def arrivals(r, rnd):
            return ("req-a",) if rnd == 10 + r else ()

        outcome, _logs, _c = _run(arrivals, rounds=20, mutate=mutate, timeout=1.0)
        self.assertTrue(any(o != "ok" for o in outcome), outcome)


# ---- three real gloo ranks -------------------------------------------------

WORLD = 3
ROUNDS = 40
SETUP_BUDGET_S = 60.0
OBSERVE_BUDGET_S = 30.0


def _free_port():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _gloo_arrivals(rank, rnd):
    out = []
    if rnd == 5 + 2 * rank:
        out.append("req-a")
    if rnd == (30 if rank == 1 else 26):
        out.append("req-b")
    return out


def _gloo_rank(rank, port, q):
    status, log, reaped, left = "ok", [], -1, -1
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ[ENV] = "8"
        dist.init_process_group(backend="gloo", rank=rank, world_size=WORLD,
                                init_method="env://")
        group = dist.group.WORLD
        c = u.UnifiedRadixCache.__new__(u.UnifiedRadixCache)
        c.attn_cp_group = None
        c.attn_tp_group = None
        c.tp_group = group
        c.tp_world_size = WORLD
        c.cache_controller = types.SimpleNamespace(
            prefetch_tokens_occupied=0,
            terminate_prefetch=lambda op: (SPAN, []),
            append_host_mem_release=lambda host_indices=None, generation=None: None,
        )
        c._retired_prefetch = []
        c._retired_prefetch_reaped = 0
        c.dec_host_lock_ref = lambda node, params: None
        c._wait_bounded = lambda work, label: work.wait()
        cur = {"round": 0}
        real = types.MethodType(u.UnifiedRadixCache._all_reduce_attn_groups, c)

        def logged(tensor, op, label="hicache"):
            log.append((cur["round"], label))
            real(tensor, op, label=label)

        def can_terminate(operation, tail_hold=0):
            logged(torch.tensor([0], dtype=torch.int), dist.ReduceOp.MAX,
                   label="can_terminate_prefetch")
            return True

        c._all_reduce_attn_groups = logged
        c.can_terminate_prefetch = can_terminate
        for rnd in range(1, ROUNDS + 1):
            cur["round"] = rnd
            for req in _gloo_arrivals(rank, rnd):
                c._retired_prefetch.append(_make_record(req))
            c.drain_retired_prefetch()
        reaped, left = c._retired_prefetch_reaped, len(c._retired_prefetch)
    except Exception as exc:  # noqa: BLE001 -- reported to the parent
        status = f"error:{type(exc).__name__}:{exc}"
    finally:
        try:
            q.put((rank, status, log, reaped, left))
        except Exception:  # noqa: BLE001
            pass
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


class TheCadenceHoldsOverRealGlooRanks(CustomTestCase):
    def test_three_gloo_ranks_agree_on_the_same_rounds_and_reap_everything(self):
        ctx = mp.get_context("spawn")
        port = _free_port()
        q = ctx.Queue()
        procs = [ctx.Process(target=_gloo_rank, args=(r, port, q)) for r in range(WORLD)]
        for p in procs:
            p.start()
        results, budget = [], SETUP_BUDGET_S
        try:
            for _ in range(WORLD):
                results.append(q.get(timeout=budget))
                budget = OBSERVE_BUDGET_S
        finally:
            for p in procs:
                p.join(timeout=10)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
        self.assertEqual(len(results), WORLD, "a rank never reported (wedge)")
        by_rank = {r: (status, log, reaped, left) for r, status, log, reaped, left in results}
        for r in range(WORLD):
            self.assertEqual(by_rank[r][0], "ok", by_rank[r][0])
            self.assertEqual(by_rank[r][2], 2, f"rank {r} reaped {by_rank[r][2]}")
            self.assertEqual(by_rank[r][3], 0)
        self.assertEqual(by_rank[0][1], by_rank[1][1])
        self.assertEqual(by_rank[1][1], by_rank[2][1])
        entered = _entered(by_rank[0][1])
        self.assertLess(len(entered), ROUNDS // 2, entered)
        self.assertEqual(entered[0], 1)


if __name__ == "__main__":
    unittest.main()
