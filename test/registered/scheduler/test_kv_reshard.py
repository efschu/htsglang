# SPDX-License-Identifier: Apache-2.0
"""Phase-boundary KV resharding (#297): hermetic contract tests.

CPU-only, no torch.distributed, no GPU: the consensus channel and the
pairwise byte channel are injected. The multi-rank tests drive REAL threads
through barrier-backed mock channels, so "fails loudly, never hangs" and
"no half move is ever visible" are demonstrated with actual concurrency.

The load-bearing gates, mapped to DESIGN_297:
* transition coverage -- every live slot is handled exactly once, and the
  sender/receiver row lists of every pair agree (section 1);
* byte identity -- after a threaded multi-rank move, the KV reassembled
  under the NEW owner rule equals the original under the OLD rule,
  byte-for-byte, including the 7,3,3 -> 2,11,10 card vector (section 7);
* aliasing falsifier -- heavy old/new row overlap inside one buffer stays
  byte-identical (reads-before-writes, section 3);
* desync falsifier -- a poisoned rank raises the same loud KvReshardError
  on every rank, nobody hangs, and NO pool byte moved (sections 3/6);
* readiness/arming skew -- legal divergence holds uniformly, then commits
  once the group agrees (section 6).
"""

import threading
import unittest

import torch

from sglang.srt.layers.dcp.reshard_plan import (
    KvReshardError,
    build_transition,
    owner_of,
    parse_reshard_vectors,
    reshard_ceiling_rows,
    reshard_vector_set,
    rows_of,
)
from sglang.srt.managers.kv_reshard import KvPoolView, KvReshardRuntime
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# Pure plan arithmetic.
# ---------------------------------------------------------------------------


class TestPlanArithmetic(CustomTestCase):
    def test_parse_and_vector_set(self):
        self.assertEqual(
            parse_reshard_vectors("7,3,3;2,11,10"), [(7, 3, 3), (2, 11, 10)]
        )
        self.assertEqual(
            reshard_vector_set("2,11,10", 3, (7, 3, 3)),
            [(7, 3, 3), (2, 11, 10)],
        )
        # Boot vector is deduplicated, never doubled.
        self.assertEqual(
            reshard_vector_set("7,3,3;2,11,10", 3, (7, 3, 3)),
            [(7, 3, 3), (2, 11, 10)],
        )
        for bad in ("", "7,x,3", "0,3,3", "7,3;2,11,10"):
            with self.assertRaises(KvReshardError):
                if ";" in bad or "," in bad:
                    parse_reshard_vectors(bad)
                    reshard_vector_set(bad, 3, (7, 3, 3))
                else:
                    parse_reshard_vectors(bad)
        with self.assertRaises(KvReshardError):
            reshard_vector_set("2,11,10", 2, (7, 3))

    def test_ceiling_rows_is_the_per_vector_maximum(self):
        c = 69784
        # (C // 13 + 1) * 7 vs (C // 23 + 1) * 2 for rank 0.
        self.assertEqual(
            reshard_ceiling_rows(c, [(7, 3, 3), (2, 11, 10)], 0),
            max((c // 13 + 1) * 7, (c // 23 + 1) * 2),
        )
        self.assertEqual(
            reshard_ceiling_rows(c, [(7, 3, 3), (2, 11, 10)], 1),
            max((c // 13 + 1) * 3, (c // 23 + 1) * 11),
        )

    def test_owner_rule_matches_bruteforce(self):
        torch.manual_seed(7)
        for vector in [(7, 3, 3), (2, 11, 10), (1, 1), (5, 2, 9, 1)]:
            s = sum(vector)
            prefix = [0]
            for v in vector:
                prefix.append(prefix[-1] + v)
            slots = torch.randperm(50 * s)[:200].sort().values.to(torch.int64)
            owners = owner_of(slots, vector)
            for i, slot in enumerate(slots.tolist()):
                res = slot % s
                expect = next(
                    r for r in range(len(vector)) if prefix[r] <= res < prefix[r + 1]
                )
                self.assertEqual(int(owners[i]), expect, f"slot {slot} {vector}")
                rank = expect
                row = int(rows_of(slots[i : i + 1], vector, rank)[0])
                self.assertEqual(row, (slot // s) * vector[rank] + (res - prefix[rank]))

    def test_transition_covers_every_slot_exactly_once(self):
        torch.manual_seed(11)
        old, new = (7, 3, 3), (2, 11, 10)
        slots = torch.randperm(4000)[:1500].sort().values.to(torch.int64)
        trs = [build_transition(slots, old, new, r) for r in range(3)]
        # Sender/receiver symmetry: rank r sends to p exactly as many rows
        # as p expects from r.
        for r in range(3):
            for p in range(3):
                if r == p:
                    continue
                sent = trs[r].outgoing_rows.get(p)
                recv = trs[p].incoming_rows.get(r)
                n_sent = 0 if sent is None else sent.numel()
                n_recv = 0 if recv is None else recv.numel()
                self.assertEqual(n_sent, n_recv, f"pair {r}->{p}")
        # Coverage: every slot is exactly one of (stationary, local-move,
        # cross-rank) -- counted over the group.
        owners_old = owner_of(slots, old)
        owners_new = owner_of(slots, new)
        cross = int((owners_old != owners_new).sum())
        total_sent = sum(tr.outgoing_slots for tr in trs)
        total_recv = sum(tr.incoming_slots for tr in trs)
        self.assertEqual(total_sent, cross)
        self.assertEqual(total_recv, cross)
        # Retained-moving + stationary = slots whose owner is unchanged.
        stationary_or_local = int((owners_old == owners_new).sum())
        total_local_moves = sum(tr.retained_moving for tr in trs)
        self.assertLessEqual(total_local_moves, stationary_or_local)

    def test_unsorted_slots_are_refused(self):
        with self.assertRaisesRegex(KvReshardError, "sorted"):
            build_transition(
                torch.tensor([5, 3, 9], dtype=torch.int64), (2, 1), (1, 2), 0
            )


# ---------------------------------------------------------------------------
# Threaded multi-rank harness: barrier-backed consensus + mailbox exchange.
# ---------------------------------------------------------------------------


class _BarrierMinChannel:
    """Element-wise MIN across N rank threads (the #287 mock, restated).

    A broken barrier (a rank that never arrives) raises in every waiting
    thread -- the mock inherits the loud-failure property of the production
    bounded collective, so a hang cannot masquerade as a pass."""

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


class _MailboxExchange:
    """Pairwise byte channel across N rank threads.

    Every rank posts its outgoing payloads, a barrier makes all posts
    visible, then every rank collects what was addressed to it. Same
    loud-failure property as the consensus mock."""

    def __init__(self, n, timeout=15.0):
        self.n = n
        self._barrier = threading.Barrier(n, timeout=timeout)
        self._mail = {}

    def exchange_for(self, rank):
        def _exchange(outgoing, incoming_nbytes):
            for peer, payload in outgoing.items():
                self._mail[(rank, peer)] = payload.clone()
            self._barrier.wait()
            received = {}
            for peer, nbytes in incoming_nbytes.items():
                payload = self._mail.get((peer, rank))
                received[peer] = (
                    payload
                    if payload is not None
                    else torch.empty(0, dtype=torch.uint8)
                )
            self._barrier.wait()
            return received

        return _exchange


def _make_pools(n_ranks, vectors, num_slots, layers=2, heads=2, dim=4, seed=3):
    """Per-rank CPU pools pre-filled from ONE global reference KV.

    Returns (views, pools, global_ref, slots): ``global_ref[layer][L]`` is
    the [heads, 2*dim] K||V reference row of live slot ``L``; each rank's
    pool holds its owned slots at the OLD-layout rows."""
    torch.manual_seed(seed)
    old = vectors[0]
    rows_needed = [reshard_ceiling_rows(num_slots, vectors, r) for r in range(n_ranks)]
    slots = torch.arange(num_slots, dtype=torch.int64)
    live = slots[torch.randperm(num_slots)[: int(num_slots * 0.8)]].sort().values
    global_k = [
        torch.randn(num_slots, heads, dim, dtype=torch.bfloat16) for _ in range(layers)
    ]
    global_v = [
        torch.randn(num_slots, heads, dim, dtype=torch.bfloat16) for _ in range(layers)
    ]
    pools = []
    views = []
    owners = owner_of(live, old)
    for r in range(n_ranks):
        k_bufs = [
            torch.zeros(rows_needed[r], heads, dim, dtype=torch.bfloat16)
            for _ in range(layers)
        ]
        v_bufs = [
            torch.zeros(rows_needed[r], heads, dim, dtype=torch.bfloat16)
            for _ in range(layers)
        ]
        mine = live[owners == r]
        rows = rows_of(mine, old, r)
        for layer in range(layers):
            k_bufs[layer][rows] = global_k[layer][mine]
            v_bufs[layer][rows] = global_v[layer][mine]
        pools.append((k_bufs, v_bufs))
        views.append(KvPoolView(k_bufs, v_bufs))
    return views, pools, (global_k, global_v), live


def _check_new_layout(pools, global_ref, live, new_vector):
    """Byte-identity gate: every live slot's K/V rows under the NEW owner
    rule equal the global reference."""
    global_k, global_v = global_ref
    owners = owner_of(live, new_vector)
    for r, (k_bufs, v_bufs) in enumerate(pools):
        mine = live[owners == r]
        rows = rows_of(mine, new_vector, r)
        for layer in range(len(k_bufs)):
            if not torch.equal(k_bufs[layer][rows], global_k[layer][mine]):
                return False, f"rank {r} layer {layer} K mismatch"
            if not torch.equal(v_bufs[layer][rows], global_v[layer][mine]):
                return False, f"rank {r} layer {layer} V mismatch"
    return True, ""


def _run_ranks(
    n_ranks,
    *,
    vectors,
    views,
    live,
    targets,
    ready_flags,
    rounds=8,
    interval=2,
    exchange_factory=None,
):
    """One runtime per rank on a real thread. ``targets[r]`` is armed before
    the loop (None = not armed); ``ready_flags[r]`` is a mutable [bool] or a
    zero-arg callable. ``exchange_factory(rank)`` overrides the mailbox
    channel. Returns (runtimes, exceptions, cutovers)."""
    channel = _BarrierMinChannel(n_ranks)
    mailbox = _MailboxExchange(n_ranks)
    if exchange_factory is None:
        exchange_factory = mailbox.exchange_for
    cutovers = [[] for _ in range(n_ranks)]
    runtimes = []
    for r in range(n_ranks):
        runtimes.append(
            KvReshardRuntime(
                dcp_size=n_ranks,
                dcp_rank=r,
                allowed_vectors=vectors,
                current_vector=vectors[0],
                consensus_interval=interval,
                collective_min=channel.channel_for(r),
                exchange=exchange_factory(r),
                pool_view=views[r],
                live_slots_fn=lambda: live,
                ready_fn=lambda r=r: (
                    ready_flags[r]() if callable(ready_flags[r]) else ready_flags[r][0]
                ),
                cutover_fn=lambda vec, r=r: cutovers[r].append(tuple(vec)),
            )
        )
    exceptions = [None] * n_ranks

    def _worker(r):
        try:
            if targets[r] is not None:
                runtimes[r].arm(targets[r], source=f"test-rank{r}")
            for _ in range(rounds):
                runtimes[r].on_round()
        except BaseException as e:  # noqa: BLE001 -- the assertion target
            exceptions[r] = e

    threads = [threading.Thread(target=_worker, args=(r,)) for r in range(n_ranks)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)
    alive = [t for t in threads if t.is_alive()]
    if alive:
        raise AssertionError(
            f"{len(alive)} rank threads still alive after 30s -- a reshard "
            f"hang, the exact failure mode this suite exists to catch"
        )
    return runtimes, exceptions, cutovers


# ---------------------------------------------------------------------------
# The gates.
# ---------------------------------------------------------------------------


class TestByteIdentity(CustomTestCase):
    def _roundtrip(self, vectors, n_ranks, num_slots=600):
        views, pools, ref, live = _make_pools(n_ranks, vectors, num_slots)
        ready = [[True] for _ in range(n_ranks)]
        runtimes, excs, cutovers = _run_ranks(
            n_ranks,
            vectors=vectors,
            views=views,
            live=live,
            targets=[vectors[1]] * n_ranks,
            ready_flags=ready,
        )
        self.assertEqual([e for e in excs if e is not None], [])
        for r in range(n_ranks):
            self.assertEqual(cutovers[r], [vectors[1]])
            self.assertEqual(runtimes[r].epoch, 1)
            self.assertEqual(runtimes[r].current_vector, vectors[1])
        ok, why = _check_new_layout(pools, ref, live, vectors[1])
        self.assertTrue(ok, why)

    def test_card_vector_7_3_3_to_2_11_10(self):
        self._roundtrip([(7, 3, 3), (2, 11, 10)], 3)

    def test_reverse_direction(self):
        self._roundtrip([(2, 11, 10), (7, 3, 3)], 3)

    def test_two_ranks(self):
        self._roundtrip([(3, 1), (1, 3)], 2)

    def test_aliasing_falsifier_heavy_local_overlap(self):
        # (2,1) -> (1,2): rank 0 keeps every residue-0 slot but its rows
        # shift (ratio 2 -> 1), rank 1 keeps residue-2 -> heavy overlap of
        # old and new row ranges INSIDE the same buffers. Byte identity
        # proves reads-before-writes.
        self._roundtrip([(2, 1), (1, 2)], 2, num_slots=900)

    def test_empty_live_set_still_cuts_over(self):
        vectors = [(7, 3, 3), (2, 11, 10)]
        views, pools, ref, _ = _make_pools(3, vectors, 300)
        ready = [[True] for _ in range(3)]
        empty = torch.empty(0, dtype=torch.int64)
        runtimes, excs, cutovers = _run_ranks(
            3,
            vectors=vectors,
            views=views,
            live=empty,
            targets=[vectors[1]] * 3,
            ready_flags=ready,
        )
        self.assertEqual([e for e in excs if e is not None], [])
        for r in range(3):
            self.assertEqual(cutovers[r], [(2, 11, 10)])


class TestConsensusDiscipline(CustomTestCase):
    def test_desync_falsifier_poisoned_target_all_loud_none_hang(self):
        vectors = [(7, 3, 3), (2, 11, 10), (4, 5, 4)]
        views, pools, ref, live = _make_pools(3, vectors, 600)
        before = [
            [b.clone() for b in pools[r][0]] + [b.clone() for b in pools[r][1]]
            for r in range(3)
        ]
        ready = [[True] for _ in range(3)]
        # Rank 2 is poisoned with a DIFFERENT target vector.
        targets = [(2, 11, 10), (2, 11, 10), (4, 5, 4)]
        runtimes, excs, cutovers = _run_ranks(
            3,
            vectors=vectors,
            views=views,
            live=live,
            targets=targets,
            ready_flags=ready,
        )
        for r in range(3):
            self.assertIsInstance(
                excs[r], KvReshardError, f"rank {r} must raise loudly"
            )
            self.assertIn("DESYNC", str(excs[r]))
            self.assertEqual(cutovers[r], [], f"rank {r} must not cut over")
            self.assertEqual(runtimes[r].epoch, 0)
        # NO half move: not one pool byte changed on any rank.
        for r in range(3):
            after = [b for b in pools[r][0]] + [b for b in pools[r][1]]
            for b_old, b_new in zip(before[r], after):
                self.assertTrue(
                    torch.equal(b_old, b_new),
                    f"rank {r}: pool bytes moved before consensus",
                )

    def test_exchange_failure_aborts_with_pool_untouched(self):
        # The pool is untouched through pack and exchange (DESIGN_297
        # section 3): a failing byte channel must leave every buffer
        # byte-identical, keep the target armed (a later boundary may
        # retry), and raise on every rank -- never a half move.
        vectors = [(7, 3, 3), (2, 11, 10)]
        views, pools, ref, live = _make_pools(3, vectors, 600)
        before = [
            [b.clone() for b in pools[r][0]] + [b.clone() for b in pools[r][1]]
            for r in range(3)
        ]

        def _broken_exchange_factory(rank):
            def _exchange(outgoing, incoming_nbytes):
                raise KvReshardError(f"injected exchange failure on rank {rank}")

            return _exchange

        ready = [[True] for _ in range(3)]
        runtimes, excs, cutovers = _run_ranks(
            3,
            vectors=vectors,
            views=views,
            live=live,
            targets=[vectors[1]] * 3,
            ready_flags=ready,
            exchange_factory=_broken_exchange_factory,
        )
        for r in range(3):
            self.assertIsInstance(excs[r], KvReshardError)
            self.assertIn("injected exchange failure", str(excs[r]))
            self.assertEqual(cutovers[r], [])
            self.assertEqual(runtimes[r].epoch, 0)
            self.assertEqual(runtimes[r].pending, (2, 11, 10))
            after = [b for b in pools[r][0]] + [b for b in pools[r][1]]
            for b_old, b_new in zip(before[r], after):
                self.assertTrue(
                    torch.equal(b_old, b_new),
                    f"rank {r}: pool bytes moved despite exchange failure",
                )

    def test_readiness_skew_holds_uniformly_then_commits(self):
        vectors = [(7, 3, 3), (2, 11, 10)]
        views, pools, ref, live = _make_pools(3, vectors, 600)
        # Rank 1 reports NOT ready at the first two consensus boundaries and
        # ready afterwards -- deterministic skew, no wall clock involved.
        rank1_polls = {"n": 0}

        def _rank1_ready():
            rank1_polls["n"] += 1
            return rank1_polls["n"] > 2

        ready = [[True], _rank1_ready, [True]]
        runtimes, excs, cutovers = _run_ranks(
            3,
            vectors=vectors,
            views=views,
            live=live,
            targets=[vectors[1]] * 3,
            ready_flags=ready,
            rounds=10,
        )
        self.assertEqual([e for e in excs if e is not None], [])
        self.assertGreater(
            rank1_polls["n"], 2, "the skewed rank must have been polled past the hold"
        )
        for r in range(3):
            self.assertEqual(
                cutovers[r], [(2, 11, 10)], f"rank {r} must commit exactly once"
            )
        ok, why = _check_new_layout(pools, ref, live, vectors[1])
        self.assertTrue(ok, why)

    def test_arming_skew_waits_without_desync(self):
        vectors = [(7, 3, 3), (2, 11, 10)]
        views, pools, ref, live = _make_pools(3, vectors, 300)
        ready = [[True] for _ in range(3)]
        # Rank 2 never arms: the group must WAIT (min-armed semantics),
        # never desync, never move.
        targets = [(2, 11, 10), (2, 11, 10), None]
        runtimes, excs, cutovers = _run_ranks(
            3,
            vectors=vectors,
            views=views,
            live=live,
            targets=targets,
            ready_flags=ready,
        )
        self.assertEqual([e for e in excs if e is not None], [])
        for r in range(3):
            self.assertEqual(cutovers[r], [])
            self.assertEqual(runtimes[r].epoch, 0)

    def test_arm_validation(self):
        vectors = [(7, 3, 3), (2, 11, 10)]
        views, _, _, live = _make_pools(1 + 2, vectors, 300)
        channel = _BarrierMinChannel(1)
        mailbox = _MailboxExchange(1)
        rt = KvReshardRuntime(
            dcp_size=3,
            dcp_rank=0,
            allowed_vectors=vectors,
            current_vector=vectors[0],
            collective_min=channel.channel_for(0),
            exchange=mailbox.exchange_for(0),
            pool_view=views[0],
            live_slots_fn=lambda: live,
            ready_fn=lambda: True,
            cutover_fn=lambda vec: None,
        )
        ok, msg = rt.arm((2, 11), source="test")
        self.assertFalse(ok)
        ok, msg = rt.arm((7, 3, 3), source="test")
        self.assertFalse(ok)
        self.assertIn("already the current", msg)
        ok, msg = rt.arm((9, 9, 9), source="test")
        self.assertFalse(ok)
        self.assertIn("ceiling set", msg)
        ok, msg = rt.arm((2, 11, 10), source="test")
        self.assertTrue(ok)

    def test_guard_refuses_arming(self):
        vectors = [(7, 3, 3), (2, 11, 10)]
        views, _, _, live = _make_pools(3, vectors, 300)
        channel = _BarrierMinChannel(1)
        mailbox = _MailboxExchange(1)
        rt = KvReshardRuntime(
            dcp_size=3,
            dcp_rank=0,
            allowed_vectors=vectors,
            current_vector=vectors[0],
            collective_min=channel.channel_for(0),
            exchange=mailbox.exchange_for(0),
            pool_view=views[0],
            live_slots_fn=lambda: live,
            ready_fn=lambda: True,
            cutover_fn=lambda vec: None,
            guards=("PD disaggregation",),
        )
        ok, msg = rt.arm((2, 11, 10), source="test")
        self.assertFalse(ok)
        self.assertIn("PD disaggregation", msg)

    def test_boot_vector_outside_ceiling_is_refused(self):
        vectors = [(2, 11, 10)]
        views, _, _, live = _make_pools(3, [(7, 3, 3), (2, 11, 10)], 300)
        with self.assertRaisesRegex(KvReshardError, "ceiling"):
            KvReshardRuntime(
                dcp_size=3,
                dcp_rank=0,
                allowed_vectors=vectors,
                current_vector=(7, 3, 3),
                collective_min=lambda vals: vals,
                exchange=lambda o, i: {},
                pool_view=views[0],
                live_slots_fn=lambda: live,
                ready_fn=lambda: True,
                cutover_fn=lambda vec: None,
            )


class TestLadderWiring(CustomTestCase):
    """The #287 dcp_ratio rung drives arm() when a reshard runtime is bound."""

    def test_flip_arms_reshard_with_operating_point_vector(self):
        from sglang.srt.managers.admission_limiter import AdmissionLimiter
        from sglang.srt.managers.kv_pressure_runtime import KvPressureRuntime
        from sglang.srt.model_executor.kv_pressure_ladder import (
            STEP_BASE,
            STEP_RELIEF,
            KvPressureLadder,
            KvPressureSensor,
            LadderStep,
            OperatingPoint,
            PressureLadder,
            StageOperatingGrid,
        )

        grid = StageOperatingGrid(
            [
                OperatingPoint("prefill", 0.05, (7, 3, 3), (2, 11, 10)),
                OperatingPoint("decode", 0.05, (7, 3, 3), (2, 11, 10)),
            ]
        )
        table = PressureLadder(
            [
                LadderStep(name="base", step_type=STEP_BASE),
                LadderStep(
                    name="dcp_ratio",
                    step_type=STEP_RELIEF,
                    relief_feature="dcp_ratio",
                    operating_grid=grid,
                ),
            ]
        )
        sensor = KvPressureSensor(
            ascend_threshold=0.85,
            ascend_window=2,
            descend_threshold=0.55,
            descend_window=6,
            pre_stage_threshold=0.70,
            pre_stage_window=2,
            abort_stage_window=8,
            horizon_rounds=4,
        )
        ladder = KvPressureLadder(table, sensor, pre_stage_enabled=False)
        armed = []

        def _arm(vec):
            armed.append(tuple(vec))
            return True, "armed (test)"

        runtime = KvPressureRuntime(
            ladder,
            admission_limiter=AdmissionLimiter(64, 64, auto=True),
            reshard_arm=_arm,
        )
        # dcp_ratio must NOT be inventoried planned-only when the arm is bound.
        self.assertNotIn("dcp_ratio", runtime.planned_only_reliefs)
        capacity = 10_000
        for occ in [0.30, 0.50, 0.86, 0.90, 0.92, 0.94, 0.96, 0.97]:
            runtime.on_round(
                held_tokens=int(occ * capacity),
                capacity_tokens=capacity,
                running_bs=8,
                phase="decode",
            )
        self.assertEqual(armed, [(2, 11, 10)])

    def test_without_reshard_arm_stays_planned_only(self):
        from sglang.srt.managers.admission_limiter import AdmissionLimiter
        from sglang.srt.managers.kv_pressure_runtime import KvPressureRuntime
        from sglang.srt.model_executor.kv_pressure_ladder import (
            STEP_BASE,
            STEP_RELIEF,
            KvPressureLadder,
            KvPressureSensor,
            LadderStep,
            PressureLadder,
        )

        table = PressureLadder(
            [
                LadderStep(name="base", step_type=STEP_BASE),
                LadderStep(
                    name="dcp_ratio",
                    step_type=STEP_RELIEF,
                    relief_feature="dcp_ratio",
                ),
            ]
        )
        sensor = KvPressureSensor(
            ascend_threshold=0.85,
            ascend_window=2,
            descend_threshold=0.55,
            descend_window=6,
            pre_stage_threshold=0.70,
            pre_stage_window=2,
            abort_stage_window=8,
            horizon_rounds=4,
        )
        runtime = KvPressureRuntime(
            KvPressureLadder(table, sensor, pre_stage_enabled=False),
            admission_limiter=AdmissionLimiter(64, 64, auto=True),
        )
        self.assertIn("dcp_ratio", runtime.planned_only_reliefs)


if __name__ == "__main__":
    unittest.main()
