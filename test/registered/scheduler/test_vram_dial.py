# SPDX-License-Identifier: Apache-2.0
"""VRAM budget dial + KV capacity re-raise (#330): hermetic contract tests.

CPU-only, no torch.distributed, no GPU: the consensus channel and the commit
actuator are injected. Multi-rank tests drive REAL threads through a
barrier-backed MIN channel, so "fails loudly, never hangs" is demonstrated
with actual concurrency.

The load-bearing gates, mapped to DESIGN_330:
* capacity math -- compute_target_c equals a brute-force scan of feasible
  units under budget, VA-reservation and draft bounds (section 3);
* C re-raise -- a vector change (the #297 cutover) auto-arms growth and
  commits (c_new, backing, "grow") at an idle boundary (section 4);
* shrink discipline -- a smaller target NEVER commits without an explicit
  dial (no spontaneous cache flush); an authorized dial commits "shrink"
  and reports released bytes (section 4);
* floor rejection -- a below-floor dial is rejected with the exact numbers
  and changes nothing (section 2, GPU proof obligation 3);
* consensus -- op_seq delivery skew holds uniformly then commits; divergent
  budgets (poisoned rank) raise the same KvCapacityError on every rank,
  nobody hangs (section 4);
* reshard interplay -- the fit guard refuses a row-heavier vector after
  growth with an actionable message, and a funded pending reshard target is
  pre-provisioned so the guard opens (section 5);
* allocator growth -- grow_size keeps live allocations and appends exactly
  the new ids (section 4).
"""

import threading
import unittest

import torch

from sglang.srt.managers.vram_dial import (
    MIB,
    BootCapacityPlan,
    KvCapacityError,
    KvCapacityRuntime,
    RankState,
    validate_vram_dial_compat,
    verify_pool_reached_capacity,
)
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import graph_safe_store_bound
from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class _BarrierMinChannel:
    """Element-wise MIN across N rank threads (the #287 mock, restated). A
    broken barrier raises in every waiting thread, so a hang cannot
    masquerade as a pass."""

    def __init__(self, n, timeout=15.0):
        self.n = n
        self._barrier = threading.Barrier(n, timeout=timeout)
        self._slots = [None] * n
        self._result = None

    def channel_for(self, rank):
        def _reduce(vals):
            self._slots[rank] = list(vals)
            index = self._barrier.wait()
            if index == 0:
                self._result = [min(col) for col in zip(*self._slots)]
            self._barrier.wait()
            return list(self._result)

        return _reduce


def _mk_ranks(
    floors, budgets, trbs, drbs, reserves, draft_reserves, boot_rows
):
    return [
        RankState(
            floor_bytes=floors[r],
            budget_bytes=budgets[r],
            target_row_bytes=trbs[r],
            draft_token_bytes=drbs[r],
            target_reserve_rows=reserves[r],
            draft_reserve_tokens=draft_reserves[r],
            boot_backed_rows=boot_rows[r],
            card_uuid=f"GPU-test-{r}",
            device_index=r,
        )
        for r in range(len(floors))
    ]


class _Commits:
    def __init__(self, released=0):
        self.calls = []
        self.released = released

    def __call__(self, c_new, backing, mode):
        self.calls.append((c_new, backing, mode))
        return self.released if mode in ("shrink", "adjust") else 0


def _mk_runtime(
    *,
    n=1,
    rank=0,
    vec=(1,),
    floors=None,
    budgets=None,
    trbs=None,
    drbs=None,
    reserves=None,
    draft_reserves=None,
    boot_rows=None,
    current_c,
    interval=1,
    ready=True,
    collective=None,
    pending=None,
    commits=None,
    user_cap=None,
):
    floors = floors or [0] * n
    trbs = trbs or [100] * n
    drbs = drbs or [0] * n
    reserves = reserves or [10**9] * n
    draft_reserves = draft_reserves or [0] * n
    vec_holder = {"vec": tuple(vec)}
    pending_holder = {"pending": pending}
    commits = commits if commits is not None else _Commits()
    if n > 1 and collective is None:
        # Single-threaded group-math tests: a width-1 barrier channel is a
        # passthrough MIN (the threaded tests inject real N-wide channels).
        collective = _BarrierMinChannel(1).channel_for(0)
    if boot_rows is None:
        s = sum(vec_holder["vec"])
        boot_rows = [
            (current_c // s + 1) * vec_holder["vec"][r] for r in range(n)
        ]
    if budgets is None:
        # Natural boot budgets: floor + backed KV bytes.
        budgets = [
            floors[r]
            + boot_rows[r] * trbs[r]
            + (current_c + 1) * drbs[r]
            for r in range(n)
        ]
    rt = KvCapacityRuntime(
        n_ranks=n,
        my_rank=rank,
        ranks=_mk_ranks(
            floors, budgets, trbs, drbs, reserves, draft_reserves, boot_rows
        ),
        page_size=1,
        current_c=current_c,
        user_cap=user_cap,
        consensus_interval=interval,
        collective_min=collective,
        ready_fn=lambda: ready,
        current_vector_fn=lambda: vec_holder["vec"],
        pending_reshard_fn=lambda: pending_holder["pending"],
        commit_fn=commits,
    )
    rt._test_vec = vec_holder
    rt._test_pending = pending_holder
    rt._test_commits = commits
    return rt


class TestCapacityMath(CustomTestCase):
    def _brute_force_c(self, rt, vec):
        s = sum(vec)
        best = 0
        for unit in range(0, 4000):
            c = unit * s
            ok = True
            for r, st in enumerate(rt._ranks):
                rows = (unit + 1) * vec[r]
                kv = rows * st.target_row_bytes + (c + 1) * st.draft_token_bytes
                if st.floor_bytes + kv > st.budget_bytes:
                    ok = False
                if rows > st.target_reserve_rows:
                    ok = False
                if st.draft_token_bytes > 0 and c > st.draft_reserve_tokens:
                    ok = False
            if ok:
                best = c
        return best

    def test_target_c_matches_bruteforce(self):
        vec = (7, 3, 3)
        rt = _mk_runtime(
            n=3,
            vec=vec,
            floors=[10_000, 8_000, 8_000],
            trbs=[100, 80, 80],
            drbs=[10, 10, 10],
            reserves=[2000, 2000, 2000],
            draft_reserves=[4000, 4000, 4000],
            current_c=1300,
        )
        for v in [(7, 3, 3), (2, 11, 10), (1, 1, 1)]:
            self.assertEqual(
                rt.compute_target_c(v),
                self._brute_force_c(rt, v),
                msg=f"vector {v}",
            )

    def test_user_cap_clamps(self):
        rt = _mk_runtime(n=1, vec=(1,), current_c=100, user_cap=42)
        self.assertLessEqual(rt.compute_target_c(), 42)

    def test_natural_boot_is_stable(self):
        """With natural budgets and no dial, nothing is armed: no capacity
        change, no backing change, pending_work is False."""
        rt = _mk_runtime(
            n=2, vec=(1, 1), floors=[500, 500], current_c=198, interval=1
        )
        self.assertFalse(rt.pending_work())
        self.assertIsNone(rt.on_round())
        self.assertEqual(rt.epoch, 0)
        self.assertEqual(rt._test_commits.calls, [])


class TestGrowReRaise(CustomTestCase):
    def _grow_setup(self, **kw):
        # Two ranks, boot vector (1,1) at C=198; switching to (1,3) funds
        # C=396 (the C re-raise scenario in miniature). Group math is driven
        # single-threaded through a width-1 barrier channel (passthrough MIN)
        # unless the test injects its own.
        kw.setdefault("collective", _BarrierMinChannel(1).channel_for(0))
        return _mk_runtime(
            n=2,
            vec=(1, 1),
            floors=[0, 0],
            budgets=[10_000, 40_000],
            trbs=[100, 100],
            reserves=[1000, 1000],
            current_c=198,
            **kw,
        )

    def test_vector_change_auto_arms_growth_and_commits(self):
        # Two-rank group math, driven single-threaded through a width-1
        # barrier channel (a passthrough MIN).
        ch = _BarrierMinChannel(1)
        rt = self._grow_setup(collective=ch.channel_for(0))
        self.assertIsNone(rt.on_round())  # boot state: stable
        # Growth requires one prior dial (op_seq > 0, the headroom grant);
        # simulate it -- replicated state, identical on every rank.
        rt._op_seq = 1
        rt._test_vec["vec"] = (1, 3)  # the #297 cutover installed a new vector
        self.assertTrue(rt.pending_work())
        stats = rt.on_round()
        self.assertIsNotNone(stats)
        self.assertEqual(stats["kind"], "grow")
        self.assertEqual(stats["new_tokens"], 396)
        self.assertEqual(rt.current_c, 396)
        self.assertEqual(rt.epoch, 1)
        c_new, backing, mode = rt._test_commits.calls[-1]
        self.assertEqual((c_new, mode), (396, "grow"))
        self.assertEqual(backing, (396 // 4 + 1) * 1)  # rank 0 rows
        # Stable afterwards.
        self.assertIsNone(rt.on_round())
        self.assertFalse(rt.pending_work())

    def test_growth_waits_for_idle(self):
        rt = self._grow_setup(ready=False)
        rt._op_seq = 1
        rt._test_vec["vec"] = (1, 3)
        self.assertIsNone(rt.on_round())
        self.assertEqual(rt.epoch, 0)
        self.assertEqual(rt._test_commits.calls, [])

    def test_cadence_gate(self):
        rt = self._grow_setup(interval=4)
        rt._op_seq = 1
        rt._test_vec["vec"] = (1, 3)
        for _ in range(3):
            self.assertIsNone(rt.on_round())
        self.assertIsNotNone(rt.on_round())  # 4th round is the boundary

    def test_no_growth_before_first_dial(self):
        """Card-run finding: chunk-rounding slack in the natural budgets
        funded a spontaneous boot grow past the fitted ceiling, breaking
        declared-vector resharding. Growth stays held until op_seq > 0."""
        # Boot backing already covers both vectors (the fitted-ceiling
        # situation), so the vector flip alone needs no backing adjust.
        rt = self._grow_setup(boot_rows=[200, 200])
        rt._test_vec["vec"] = (1, 3)  # growth funded...
        self.assertIsNone(rt.on_round())  # ...but never armed without a dial
        self.assertEqual(rt.epoch, 0)
        self.assertFalse(rt.pending_work())
        rt._op_seq = 1  # the headroom grant
        self.assertTrue(rt.pending_work())
        stats = rt.on_round()
        self.assertEqual(stats["kind"], "grow")


class TestShrinkDiscipline(CustomTestCase):
    def _dial_setup(self, **kw):
        # One rank, MiB-scale numbers so the MiB-based dial API applies:
        # floor 1000 MiB, C=100 tokens at 100 MiB/row -> natural budget
        # 1000 + 101*100 = 11100 MiB.
        return _mk_runtime(
            n=1,
            vec=(1,),
            floors=[1000 * MIB],
            trbs=[100 * MIB],
            reserves=[1000],
            current_c=100,
            commits=_Commits(released=4 * 1024 * MIB),
            **kw,
        )

    def test_no_spontaneous_shrink(self):
        rt = self._dial_setup()
        # Sabotage: lower the budget WITHOUT the dial path (simulates a
        # would-be spontaneous shrink); the runtime must hold, not commit.
        rt._ranks[0].budget_bytes = 6100 * MIB
        self.assertIsNone(rt.on_round())
        self.assertEqual(rt.epoch, 0)
        self.assertEqual(rt.current_c, 100)

    def test_authorized_dial_shrinks_and_reports_release(self):
        rt = self._dial_setup()
        ok, msg = rt.apply_budget_request(device="rank:0", budget_mib=6100)
        self.assertTrue(ok, msg)
        self.assertIn("shrink armed", msg)
        stats = rt.on_round()
        self.assertIsNotNone(stats)
        self.assertEqual(stats["kind"], "shrink")
        self.assertEqual(stats["new_tokens"], 50)
        self.assertEqual(stats["released_bytes"], 4 * 1024 * MIB)
        c_new, backing, mode = rt._test_commits.calls[-1]
        self.assertEqual((c_new, backing, mode), (50, 51, "shrink"))
        # Authorization is consumed: a later, lower target holds again.
        self.assertFalse(rt._shrink_authorized)

    def test_dial_up_restores_capacity(self):
        rt = self._dial_setup()
        rt.apply_budget_request(device="rank:0", budget_mib=6100)
        rt.on_round()
        self.assertEqual(rt.current_c, 50)
        ok, msg = rt.apply_budget_request(device="rank:0", budget_mib=11100)
        self.assertTrue(ok, msg)
        stats = rt.on_round()
        self.assertEqual(stats["kind"], "grow")
        self.assertEqual(rt.current_c, 100)

    def test_release_fraction_and_mib(self):
        rt = self._dial_setup()
        # release 50% of the dialable span (11100-1000 = 10100 -> 5050).
        ok, msg = rt.apply_budget_request(
            device="rank:0", release_fraction=0.5
        )
        self.assertTrue(ok, msg)
        self.assertEqual(rt._ranks[0].budget_bytes, (11100 - 5050) * MIB)
        ok, msg = rt.apply_budget_request(device="rank:0", release_mib=-1000)
        self.assertTrue(ok, msg)
        self.assertEqual(rt._ranks[0].budget_bytes, (11100 - 5050 + 1000) * MIB)

    def test_below_floor_rejected_with_exact_numbers(self):
        rt = self._dial_setup()
        old_budget = rt._ranks[0].budget_bytes
        old_seq = rt.op_seq
        ok, msg = rt.apply_budget_request(device="rank:0", budget_mib=900)
        self.assertFalse(ok)
        self.assertIn("below the pinned floor", msg)
        self.assertIn("1000 MiB", msg)  # the floor
        self.assertIn("900 MiB", msg)  # the request
        self.assertIn("1200 MiB", msg)  # min viable = floor + 2 rows
        self.assertEqual(rt._ranks[0].budget_bytes, old_budget)
        self.assertEqual(rt.op_seq, old_seq)
        self.assertIsNone(rt.on_round())  # and nothing changes

    def test_exactly_one_mode_required(self):
        rt = self._dial_setup()
        ok, msg = rt.apply_budget_request(device="rank:0")
        self.assertFalse(ok)
        ok, msg = rt.apply_budget_request(
            device="rank:0", budget_mib=6100, release_mib=10
        )
        self.assertFalse(ok)

    def test_device_resolution(self):
        rt = _mk_runtime(
            n=3,
            vec=(1, 1, 1),
            floors=[MIB] * 3,
            trbs=[MIB] * 3,
            reserves=[10**6] * 3,
            current_c=99,
        )
        self.assertEqual(rt._resolve_targets("all"), [0, 1, 2])
        self.assertEqual(rt._resolve_targets("rank:2"), [2])
        self.assertEqual(rt._resolve_targets("cuda:1"), [1])
        self.assertEqual(rt._resolve_targets("GPU-test-0"), [0])
        for bad in ("rank:7", "cuda:9", "GPU-nope", "bogus"):
            with self.assertRaises(KvCapacityError):
                rt._resolve_targets(bad)

    def test_budget_clamped_to_va_ceiling(self):
        rt = self._dial_setup()
        ok, msg = rt.apply_budget_request(
            device="rank:0", budget_mib=10**7
        )
        self.assertTrue(ok, msg)
        self.assertLessEqual(
            rt._ranks[0].budget_bytes,
            rt.effective_budget_ceiling_bytes(0),
        )


class TestReshardInterplay(CustomTestCase):
    def _setup(self):
        # Two ranks, vector (1,1), grown state where vector (3,1) would need
        # more rows on rank 0 than are backed.
        return _mk_runtime(
            n=2,
            vec=(1, 1),
            floors=[0, 0],
            budgets=[15_000, 15_000],
            trbs=[100, 100],
            reserves=[1000, 1000],
            current_c=198,
            # The user cap pins the ceiling at the boot value so the budget
            # slack exercises the BACKING (pre-provision) axis, not growth.
            user_cap=198,
        )

    def test_fit_check_refuses_with_dial_hint(self):
        rt = self._setup()
        # (3,1) at C=198 needs (198//4+1)*3 = 150 rows on rank 0; backed 100.
        ok, msg = rt.reshard_fit_check((3, 1))
        self.assertFalse(ok)
        self.assertIn("rank 0", msg)
        self.assertIn("150 rows", msg)
        self.assertIn("/vram_budget", msg)

    def test_funded_pending_target_is_preprovisioned(self):
        rt = self._setup()
        rt._test_pending["pending"] = (3, 1)
        # Budgets fund 150 rows on rank 0 (15000 >= 150*100): an "adjust"
        # commit raises the backing, then the fit guard opens.
        self.assertTrue(rt.pending_work())
        stats = rt.on_round()
        self.assertIsNotNone(stats)
        self.assertEqual(stats["kind"], "adjust")
        self.assertEqual(stats["new_tokens"], 198)  # ceiling untouched
        self.assertEqual(rt._ranks[0].backed_rows, 150)
        ok, msg = rt.reshard_fit_check((3, 1))
        self.assertTrue(ok, msg)

    def test_unfunded_pending_target_stays_refused(self):
        rt = self._setup()
        rt._ranks[0].budget_bytes = 100 * 100 + 50  # funds only 100 rows
        rt._test_pending["pending"] = (3, 1)
        self.assertIsNone(rt.on_round())  # nothing to commit
        ok, msg = rt.reshard_fit_check((3, 1))
        self.assertFalse(ok)


class TestConsensusDiscipline(CustomTestCase):
    def _run_ranks(self, n, body):
        """Run body(rank) on n threads; propagate the first exception per
        rank; assert nobody hangs (barrier timeout raises)."""
        errors = [None] * n
        results = [None] * n

        def _wrap(r):
            try:
                results[r] = body(r)
            except BaseException as e:  # noqa: BLE001 - test harness
                errors[r] = e

        threads = [
            threading.Thread(target=_wrap, args=(r,), daemon=True)
            for r in range(n)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30.0)
            self.assertFalse(t.is_alive(), "a rank thread hung")
        return results, errors

    def _mk_group(self, n, ch, **kw):
        # Natural boot budgets (stable at rest); tests mutate budgets to arm.
        return [
            _mk_runtime(
                n=n,
                rank=r,
                vec=(1,) * n,
                floors=[0] * n,
                trbs=[100] * n,
                reserves=[1000] * n,
                current_c=100 * n,
                collective=ch.channel_for(r),
                **kw,
            )
            for r in range(n)
        ]

    def test_desync_falsifier_poisoned_budget_all_loud_none_hang(self):
        n = 3
        ch = _BarrierMinChannel(n)
        rts = self._mk_group(n, ch)
        # Poison rank 1's replicated budget copy directly (bypassing the
        # replicated RPC): its C_target diverges. To make every rank ARM,
        # give each a uniform pending grow via a raised budget on rank 0's
        # copy of everyone -- simplest: poison budgets so all arm but with
        # different targets.
        for r, rt in enumerate(rts):
            for q in range(n):
                rt._ranks[q].budget_bytes = 40_000
            rt._shrink_authorized = False
            rt._op_seq = 1  # growth is armed only after a dial
        rts[1]._ranks[0].budget_bytes = 30_000  # the poison
        results, errors = self._run_ranks(n, lambda r: rts[r].on_round())
        for r in range(n):
            self.assertIsInstance(
                errors[r],
                KvCapacityError,
                msg=f"rank {r} did not raise: {errors[r]!r} / {results[r]!r}",
            )
            self.assertIn("DESYNC", str(errors[r]))
        for rt in rts:
            self.assertEqual(rt.epoch, 0)
            self.assertEqual(rt._test_commits.calls, [])

    def test_op_seq_skew_holds_then_commits(self):
        n = 2
        ch = _BarrierMinChannel(n)
        rts = self._mk_group(n, ch)
        # Rank 0 already received the dial; rank 1 has not (delivery skew).
        ok, msg = rts[0].apply_budget_request(device="all", release_fraction=0.5)
        self.assertTrue(ok, msg)

        results, errors = self._run_ranks(n, lambda r: rts[r].on_round())
        self.assertEqual(errors, [None, None])
        self.assertEqual(results, [None, None])  # uniform hold, no commit
        # The RPC arrives on rank 1; the next boundary commits on both.
        ok, msg = rts[1].apply_budget_request(device="all", release_fraction=0.5)
        self.assertTrue(ok, msg)
        results, errors = self._run_ranks(n, lambda r: rts[r].on_round())
        self.assertEqual(errors, [None, None])
        for r in range(n):
            self.assertIsNotNone(results[r])
            self.assertEqual(results[r]["kind"], "shrink")
            self.assertEqual(rts[r].epoch, 1)

    def test_group_grow_commits_on_every_rank(self):
        n = 3
        ch = _BarrierMinChannel(n)
        rts = self._mk_group(n, ch)
        for rt in rts:
            for q in range(n):
                rt._ranks[q].budget_bytes = 40_000  # uniform raise
            rt._op_seq = 1  # growth is armed only after a dial
        results, errors = self._run_ranks(n, lambda r: rts[r].on_round())
        self.assertEqual(errors, [None] * n)
        for r in range(n):
            self.assertEqual(results[r]["kind"], "grow")
            self.assertEqual(results[r]["new_tokens"], rts[0].current_c)


class TestBootPlan(CustomTestCase):
    def test_reserve_rows_covers_every_vector_at_its_own_cap(self):
        plan = BootCapacityPlan(
            vectors=((7, 3, 3), (2, 11, 10)),
            caps={(7, 3, 3): 1300, (2, 11, 10): 4600},
        )
        # rank 1: (1300//13+1)*3 = 303 vs (4600//23+1)*11 = 2211.
        self.assertEqual(plan.reserve_rows_for_rank(1), 2211)
        self.assertEqual(plan.max_cap, 4600)


class TestStatus(CustomTestCase):
    def test_status_shape(self):
        rt = _mk_runtime(
            n=2,
            vec=(1, 1),
            floors=[MIB, MIB],
            trbs=[MIB, MIB],
            reserves=[10**6] * 2,
            current_c=100,
        )
        st = rt.status()
        self.assertEqual(st["max_total_num_tokens"], 100)
        self.assertEqual(len(st["ranks"]), 2)
        for row in st["ranks"]:
            for key in (
                "budget_mib",
                "floor_mib",
                "backed_rows",
                "min_viable_budget_mib",
                "card_uuid",
            ):
                self.assertIn(key, row)


class TestAllocatorGrow(CustomTestCase):
    def test_token_allocator_grow_keeps_live_allocations(self):
        alloc = TokenToKVPoolAllocator(
            10, dtype=torch.float16, device="cpu", kvcache=None, need_sort=False
        )
        got = alloc.alloc(4)
        self.assertEqual(len(got), 4)
        self.assertEqual(alloc.available_size(), 6)
        alloc.grow_size(16)
        self.assertEqual(alloc.size, 16)
        self.assertEqual(alloc.available_size(), 12)
        # The new ids are exactly 11..16 and nothing live was recycled.
        remaining = torch.cat([alloc.free_pages, alloc.release_pages])
        self.assertEqual(
            sorted(set(remaining.tolist()) & set(range(11, 17))),
            list(range(11, 17)),
        )
        self.assertFalse(set(got.tolist()) & set(remaining.tolist()))
        with self.assertRaises(ValueError):
            alloc.grow_size(8)

    def test_paged_allocator_grow(self):
        alloc = PagedTokenToKVPoolAllocator(
            32,
            page_size=4,
            dtype=torch.float16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        before = alloc.available_size()
        alloc.grow_size(64)
        self.assertEqual(alloc.size, 64)
        self.assertEqual(alloc.num_pages, 16)
        self.assertEqual(alloc.available_size(), before + 32)
        with self.assertRaises(ValueError):
            alloc.grow_size(65)  # not page-aligned

    def test_paged_allocator_natural_page_resize_updates_num_pages(self):
        # The uneven-DCP lane runs the paged allocator at page_size=1; the
        # old base.resize left num_pages stale there, so clear() rebuilt the
        # free list at the OLD size (#330 finding).
        alloc = PagedTokenToKVPoolAllocator(
            32,
            page_size=1,
            dtype=torch.float16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )

        class _Cfg:
            max_total_num_tokens = 16

        alloc.resize(_Cfg)
        self.assertEqual(alloc.num_pages, 16)
        self.assertEqual(alloc.available_size(), 16)
        alloc.grow_size(48)
        self.assertEqual(alloc.num_pages, 48)
        self.assertEqual(alloc.available_size(), 48)


class _FakePool:
    """Minimal stand-in for a dial participant pool: the two capacity bounds
    a store has to satisfy after a commit, and nothing else."""

    def __init__(self, backed_rows, store_bound_rows, page_size=1):
        self.full_pool_backed_rows = int(backed_rows)
        self.store_bound_rows = int(store_bound_rows)
        self.page_size = int(page_size)


class _FakeParticipant:
    def __init__(self, pool, is_target=False):
        self.pool = pool
        self.is_target = is_target


class TestGraphSafeStoreBound(CustomTestCase):
    """#352: the store_kvcache bound a CUDA graph bakes in at capture time
    must stay valid for every capacity the pool can later reach."""

    def test_off_dial_lane_is_byte_identical(self):
        # Buffers allocated at exactly size + page_size: the live bound IS the
        # row count, so the default path passes exactly what it passed before.
        self.assertEqual(graph_safe_store_bound(4097, 4097), 4097)

    def test_dial_lane_uses_the_lifetime_bound(self):
        # Buffers span the VA reserve while the pool is backed well below it.
        self.assertEqual(graph_safe_store_bound(251966, 522198), 522198)

    def test_grown_ceiling_is_admitted_by_a_pre_growth_capture(self):
        # The exact #330 card-run numbers: boot C = 251965, growth to 341861
        # with a 522197-row VA reserve, page_size 1. Before the fix the graph
        # carried 251966 and asserted on every legal id above the boot
        # ceiling -- which is what >= 10 x 30k concurrent sessions are the
        # first load to reach.
        c_boot, c_new, reserve, page = 251965, 341861, 522197, 1
        captured = graph_safe_store_bound(c_boot + page, reserve + page)
        self.assertGreater(captured, c_new)
        self.assertLess(c_boot + page, c_new)  # the pre-fix bound rejected ids

    def test_never_narrows_below_the_live_bound(self):
        self.assertEqual(graph_safe_store_bound(600, 400), 600)

    def test_bound_does_not_depend_on_capture_mode(self):
        # The correctness of the bound must NOT hinge on every capture site
        # remembering to raise the capture flag: PrefillCudaGraphRunner does
        # not raise it, and a gate there would re-create #352 for its graphs.
        eager = graph_safe_store_bound(251966, 522198)
        with model_capture_mode():
            captured = graph_safe_store_bound(251966, 522198)
        self.assertEqual(eager, captured)


class TestCapacityReachesEveryConsumer(CustomTestCase):
    """#345/#352 invariant: after a commit every participant pool either
    carries the new capacity or the commit fails loudly naming the numbers.
    A consumer that does neither must not pass silently."""

    def test_pool_that_grew_passes(self):
        p = _FakeParticipant(_FakePool(backed_rows=341861, store_bound_rows=522198))
        verify_pool_reached_capacity(p, 341861)

    def test_pool_frozen_at_the_boot_ceiling_is_caught(self):
        p = _FakeParticipant(_FakePool(backed_rows=251965, store_bound_rows=522198))
        with self.assertRaises(KvCapacityError) as cm:
            verify_pool_reached_capacity(p, 341861)
        msg = str(cm.exception)
        self.assertIn("251965", msg)
        self.assertIn("341861", msg)

    def test_pool_that_cannot_host_the_capacity_vetoes_with_numbers(self):
        # store_bound_rows is the lifetime ceiling captured graphs carry; a
        # capacity above it is structurally unhostable and must be refused
        # here rather than as a device assert on a legal slot id.
        p = _FakeParticipant(_FakePool(backed_rows=341861, store_bound_rows=341861))
        with self.assertRaises(KvCapacityError) as cm:
            verify_pool_reached_capacity(p, 341861)
        msg = str(cm.exception)
        self.assertIn("341861", msg)
        self.assertIn("341862", msg)

    def test_padding_slot_is_counted_in_the_veto(self):
        # Exactly enough rows for the slots but not for the reserved padding
        # slot is still a refusal: padded/dummy tokens write at index `size`.
        p = _FakeParticipant(
            _FakePool(backed_rows=100, store_bound_rows=100, page_size=4)
        )
        with self.assertRaises(KvCapacityError):
            verify_pool_reached_capacity(p, 100)
        ok = _FakeParticipant(
            _FakePool(backed_rows=100, store_bound_rows=104, page_size=4)
        )
        verify_pool_reached_capacity(ok, 100)

    def test_pool_without_the_bounds_is_a_no_op_here(self):
        # A pool family exposing neither bound cannot be verified at commit
        # time; the guard against such a participant is the boot-time
        # pool-family refusal in build_kv_capacity_runtime, not this function.
        verify_pool_reached_capacity(_FakeParticipant(object()), 341861)


class TestDialRefusals(CustomTestCase):
    """DESIGN_330 section 7 refusals that guard capacity consumers with no
    grow path (#352 audit)."""

    class _Args:
        device = "cuda"
        enable_memory_saver = False
        disaggregation_mode = "null"
        hicache_storage_backend = None
        enable_kv_session_offload = False
        dual_group_lane = None
        dp_size = 1
        kv_canary = "none"
        enable_hisparse = False
        weightless_kv_fastlane = False
        speculative_algorithm = "NEXTN"
        speculative_drafter_policy = None

    def _args(self, **over):
        a = TestDialRefusals._Args()
        for k, v in over.items():
            setattr(a, k, v)
        return a

    def test_supported_lane_passes(self):
        validate_vram_dial_compat(self._args())

    def test_dflash_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            validate_vram_dial_compat(self._args(speculative_algorithm="DFLASH"))
        self.assertIn("DFLASH", str(cm.exception))

    def test_dflash_via_drafter_policy_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            validate_vram_dial_compat(
                self._args(speculative_drafter_policy="0:dflash:16,4096:nextn:3")
            )
        self.assertIn("DFLASH", str(cm.exception))

    def test_hisparse_and_weightless_are_refused(self):
        with self.assertRaises(ValueError):
            validate_vram_dial_compat(self._args(enable_hisparse=True))
        with self.assertRaises(ValueError):
            validate_vram_dial_compat(self._args(weightless_kv_fastlane=True))


if __name__ == "__main__":
    unittest.main()
